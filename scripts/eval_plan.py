"""Planning success with CEM in the simulator, on the root eval.py tasks (same seed -> same tasks),
optionally with videos of the plans: simulation | the world model's decoded imagination.

    python -m scripts.eval_plan --run pusht_LeWM-D --epoch 10
    python -m scripts.eval_plan --run pusht_LeWM-D --epoch 10 --num-videos 10

Model: --run/--epoch loads <swm>/checkpoints/<run>/weights_epoch_<epoch>.pt (and its decoder for
videos). --sample-steps/--plan-noise/--risk-weight/--mse-head override flow attributes.

Every run appends one json line to results/plan/<--log>.

Videos (--num-videos N > 0, needs the run's decoder): at every replan the chosen plan (CEM mean)
is rolled out by the world model (one noise sample for flow variants) and every predicted latent is
decoded. The model predicts once per action block (FRAMESKIP env steps), so the video has one frame
per block boundary: the observation after k blocks of the current plan next to the model's
prediction for that same moment (k = 0 is the encoded and decoded current observation). At a
replan, the old plan's last prediction is shown against the state it led to, then the new plan
starts at k = 0. Written for the first N episodes to results/plan/<run>_e<epoch>/ (or --out):
ep<k>_<success|fail>.mp4, a tiled overview.mp4, and results.json.

--exec-noise adds N(0, (exec_noise * action std)^2) to every executed env action (normalised units
as in training); videos show the noisy execution.

Hallucination recording (--halluc-set NAME [--halluc-out ROOT], needs the decoder and --mse-head for
flow runs): every replan is written as one window of the analysis/halluc export format in
ROOT/NAME (default ROOT: results/halluc/<run>_e<epoch>):
the sim observations after k = 0..5 blocks of the executed plan, the decoded imagination of that plan
(MSE head, as planned) and the predictors u_r / u_f / u_s / motion. Plans cut short by the episode end
(budget or success) keep the blocks they reached (meta n_imag < 5).
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from helpers.constants import FRAMESKIP, TASKS, run_dir
from helpers.models import load_run
from analysis.halluc.rollout import SetWriter, imagine, set_root
from helpers.planning import cem_solver, fit_scalers, image_transform, load_planning_dataset, pick_tasks
from helpers.video import decode, encode, imagenet_norm, to_uint8, write_video


class PlanPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that times every solver call and optionally
    - records sim frames and, at each replan, the decoded imagined rollout (decoder given), and
    - adds Gaussian noise to the executed actions (exec_noise > 0)."""

    def __init__(self, *args, decoder=None, exec_noise=0.0, act_std=None, seed=0, halluc=None, **kw):
        super().__init__(*args, **kw)
        self.decoder = decoder
        self.halluc = halluc  # None or dict(n_seeds=, seed=): record replan windows with predictors
        self.exec_noise, self.act_std = exec_noise, act_std
        self.rng = np.random.default_rng(seed)
        self.norm = imagenet_norm()
        self.plan_times, self.plan_envs = [], []  # seconds per solver call, envs replanned in it
        self.frames, self.imag, self.since_plan, self.goal = None, None, None, None
        self._pending = None

    def set_env(self, env):
        super().set_env(env)
        n = env.num_envs
        self.frames = [[] for _ in range(n)]  # (sim frame, imagined frame, goal, k) per block boundary
        self.imag = [None] * n                 # decoded rollout of the current plan
        self.since_plan = [0] * n
        self.goal = [None] * n
        self.steps = [0] * n                   # env steps taken
        self.cur = [None] * n                  # halluc: the current plan's window being filled
        self.chunks = [[] for _ in range(n)]   # halluc: finished windows per env
        solve = self.solver

        def solve_timed(info, init_action=None):
            torch.cuda.synchronize()
            t = time.time()
            out = solve(info, init_action=init_action)
            torch.cuda.synchronize()
            self.plan_times.append(time.time() - t)
            self.plan_envs.append(out["actions"].size(0))
            if self.decoder is not None:
                self._imagine(info, out["actions"])
            return out

        self.solver_call = solve_timed

    @torch.no_grad()
    def _imagine(self, info, actions):
        """Roll the chosen plans out with the world model and decode every latent."""
        model = self.solver.model
        dev = next(model.parameters()).device
        if self.halluc is not None:
            out = imagine(model, self.decoder, info["pixels"].to(dev), actions.to(dev, torch.float32),
                          n_seeds=self.halluc["n_seeds"], seed=self.halluc["seed"])
            H = info["pixels"].size(1)
            self._pending = [{"wm": out["wm"][b, H - 1:], "z": out["z"][b, H - 1:],
                              "act": actions[b].cpu().numpy(),
                              "pred": {k: out[k][b] for k in ("u_r", "u_f", "u_s", "drift", "motion")}}
                             for b in range(actions.size(0))]
            return
        embs = []
        for b in range(actions.size(0)):  # one env at a time: get_cost's goal broadcast assumes batch 1, as in CEM
            ex = {k: v[b:b + 1].to(dev).unsqueeze(1) if torch.is_tensor(v) else v for k, v in info.items()}
            model.get_cost(ex, actions[b:b + 1].to(dev, torch.float32).unsqueeze(1))  # (1, 1, horizon, block*A)
            embs.append(ex["predicted_emb"][:, 0])                   # (1, H + horizon, D): context + predictions
        emb = torch.cat(embs)
        H = info["pixels"].size(1)
        B, T, D = emb.shape
        img = decode(self.decoder, emb.reshape(B * T, D), self.norm)
        img = img.reshape(B, T, *img.shape[1:])
        self._pending = [im[H - 1:] for im in img]                   # [now, +1 block, +2 blocks, ...]

    def get_action(self, info_dict, **kwargs):
        n = self.env.num_envs
        # same replan rule as WorldModelPolicy.get_action: (empty buffer or flushed) and not terminated
        flush = info_dict.get("_needs_flush")
        flush = np.zeros(n, bool) if flush is None else np.asarray(flush, bool).reshape(n, -1)[:, -1]
        term = info_dict.get("terminated")
        dead = np.zeros(n, bool) if term is None else np.asarray(term, bool).reshape(n, -1)[:, -1]
        replan = [(len(self._action_buffer[i]) == 0 or flush[i]) and not dead[i] for i in range(n)]
        self._pending = None
        # route the solver call through solve_timed (dunder calls skip instance attributes,
        # so swap in a callable proxy for the duration of the call)
        solver = self.solver
        self.solver = _CallWrapper(solver, self.solver_call)
        try:
            action = super().get_action(info_dict, **kwargs)
        finally:
            self.solver = solver
        if self.decoder is not None:
            self._record(info_dict, action, replan)
        if self.exec_noise > 0:
            action = action + self.exec_noise * self.act_std * self.rng.standard_normal(action.shape)
            space = self.env.action_space
            action = np.clip(action, space.low, space.high).astype(np.float32)
        return action

    def _record(self, info_dict, action, replan):
        n = self.env.num_envs
        px, goal = info_dict.get("pixels"), info_dict.get("goal")
        if goal is not None:
            for i in range(n):
                self.goal[i] = to_uint8(goal[i])
        if self._pending is not None:
            rows = [i for i in range(n) if replan[i]]
            for row, i in enumerate(rows):
                # the plan being replaced: its last prediction vs the state it led to
                k = self.since_plan[i] // FRAMESKIP
                if self.imag[i] is not None and self.since_plan[i] % FRAMESKIP == 0 and k < len(self.imag[i]):
                    self.frames[i].append((to_uint8(px[i]), self.imag[i][k], self.goal[i], k))
                    if self.cur[i] is not None:
                        self.cur[i]["real"].append(to_uint8(px[i]))
                new = self._pending[row]
                if isinstance(new, dict):  # halluc recording
                    self._close_chunk(i)
                    self.cur[i] = {**new, "real": [], "step": self.steps[i]}
                    new = new["wm"]
                self.imag[i], self.since_plan[i] = new, 0
        for i in range(n):
            if self.imag[i] is None or np.isnan(action[i]).all():
                continue
            # one video frame per action block: the observation after k blocks of the current plan
            # next to the world model's prediction for that same moment
            if self.since_plan[i] % FRAMESKIP == 0:
                k = self.since_plan[i] // FRAMESKIP
                if k < len(self.imag[i]):
                    self.frames[i].append((to_uint8(px[i]), self.imag[i][k], self.goal[i], k))
                    if self.cur[i] is not None:
                        self.cur[i]["real"].append(to_uint8(px[i]))
            self.since_plan[i] += 1
            self.steps[i] += 1

    def _close_chunk(self, i):
        c, self.cur[i] = self.cur[i], None
        if c is not None and len(c["real"]) >= 2:  # context + at least one imagined block
            self.chunks[i].append(c)

    def close_chunks(self):
        for i in range(len(self.cur)):
            self._close_chunk(i)


class _CallWrapper:
    """Callable proxy: calling it runs fn, attribute access goes to the wrapped solver."""

    def __init__(self, obj, fn):
        self._obj, self._fn = obj, fn

    def __call__(self, *a, **kw):
        return self._fn(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._obj, name)


def compose(sim, imag, goal, k, ok, label):
    """Side-by-side frame: sim (goal inset) | imagination, with a header strip."""
    import cv2

    h, w = sim.shape[:2]
    imag = cv2.resize(imag, (w, h), interpolation=cv2.INTER_LINEAR)
    left = sim.copy()
    if goal is not None:
        g = cv2.resize(goal, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        left[2:2 + g.shape[0], w - g.shape[1] - 2:w - 2] = g
        cv2.rectangle(left, (w - g.shape[1] - 3, 1), (w - 2, g.shape[0] + 2), (255, 255, 255), 1)
    frame = np.concatenate([left, np.full((h, 4, 3), 255, np.uint8), imag], 1)
    head = np.full((36, frame.shape[1], 3), 255, np.uint8)  # line 1: episode + outcome, line 2: panel titles
    font, black = cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 0)
    col = (0, 160, 0) if ok else (200, 0, 0)
    cv2.circle(head, (9, 9), 5, col, -1)
    cv2.putText(head, label, (20, 13), font, 0.4, black, 1)
    cv2.putText(head, "simulation (goal top-right)", (4, 30), font, 0.4, black, 1)
    cv2.putText(head, f"imagined: plan +{k} block{'s' if k != 1 else ''}", (w + 8, 30), font, 0.4, black, 1)
    return np.concatenate([head, frame], 0)


def write_videos(policy, episodes, starts, succ, n, fps, out):
    """ep<i>_<outcome>.mp4 for the first n episodes and a tiled overview.mp4 (up to 16, 4 wide)."""
    tiles = []
    for i, frames in enumerate(policy.frames[:n]):
        if not frames:
            continue
        label = f"ep {episodes[i]} step {starts[i]} -> {'success' if succ[i] else 'fail'}"
        vid = [compose(s, im, g, k, succ[i], label) for s, im, g, k in frames]
        write_video(vid, out / f"ep{i:02d}_{'success' if succ[i] else 'fail'}", fps)
        tiles.append(vid)
    if not tiles:
        return
    tiles = tiles[:16]
    L = max(len(v) for v in tiles)
    tiles = [v + [v[-1]] * (L - len(v)) for v in tiles]  # pad with the last frame
    blank = np.zeros_like(tiles[0][0])
    cols = 4
    grid = []
    for t in range(L):
        cells = [v[t] for v in tiles] + [blank] * (-len(tiles) % cols)
        rows = [np.concatenate(cells[r:r + cols], 1) for r in range(0, len(cells), cols)]
        g = np.concatenate(rows, 0)[::2, ::2]
        grid.append(g)
    write_video(grid, out / "overview", fps)


def write_halluc(policy, model, decoder, episodes, starts, succ, a):
    """Every recorded replan as one window of the analysis/halluc export format."""
    policy.close_chunks()
    rms = lambda x: np.linalg.norm(x, axis=-1) / np.sqrt(x.shape[-1])
    w = SetWriter(a.halluc_out, a.halluc_set, ctx=1, annotate=False)
    n = 0
    for i, chunks in enumerate(policy.chunks):
        for r, c in enumerate(chunks):
            T = len(c["real"])  # 1 context + n_imag blocks reached
            sid = f"{a.halluc_set}_s{a.seed}_e{i:03d}_r{r}"
            meta = {"episode": int(episodes[i]), "start": int(starts[i]), "env": i, "replan": r,
                    "env_step": c["step"], "n_imag": T - 1, "success": bool(succ[i]), "run": a.run,
                    "epoch": a.epoch, "frameskip": FRAMESKIP, "source": "plan", "seed": a.seed,
                    "n_seeds": a.halluc_seeds, "head": "mse", "exec_noise": a.exec_noise}
            pred = {k: v[:T - 1] for k, v in c["pred"].items()}
            real = np.stack(c["real"])
            zr = encode(model, real)  # real latents: recon (decoder ceiling) and the oracle columns
            recon = decode(decoder, torch.from_numpy(zr).cuda())
            pred["oracle_e_z"] = rms(c["z"][1:T] - zr[1:])
            pred["oracle_motion_real"] = rms(zr[1:] - zr[:-1])
            w.add(sid, a.task, real, c["wm"][:T], recon, c["act"], c["z"][:T], zr, pred, meta,
                  {"idx": n, "start": int(starts[i]) + c["step"], "episode": f"s{a.seed}_e{i:03d}",
                   "data_dir": "plan", "pick": "all", "horizon_motion": float("nan"),
                   "replan": r, "n_imag": T - 1, "success": int(succ[i])})
            n += 1
    w.close()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="trained run under <swm>/checkpoints")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--task", default="pusht", choices=[t for t in TASKS if "env" in TASKS[t]])
    p.add_argument("--dataset", default=None, help="name or path; default: the task's .lance if present, else .h5")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--plan-noise", type=int, default=None)
    p.add_argument("--risk-weight", type=float, default=None)
    p.add_argument("--mse-head", action="store_true", help="flow checkpoint, but plan with its MSE head")
    p.add_argument("--exec-noise", type=float, default=0.0, help="action noise executed in the env (normalised units)")
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="plan_results.jsonl", help="json-lines log in results/plan")
    p.add_argument("--num-videos", type=int, default=0, help="save videos for the first N episodes (needs a decoder)")
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--out", default=None, help="video dir (default: results/plan/<run>_e<epoch>)")
    p.add_argument("--halluc-set", default=None, help="record every replan as a hallucination window in this set")
    p.add_argument("--halluc-out", default=None, help="sets root (default: results/halluc/<run>_e<epoch>)")
    p.add_argument("--halluc-seeds", type=int, default=4, help="flow-head samples for u_f / u_s")
    a = p.parse_args()
    if a.halluc_set:
        a.halluc_out = a.halluc_out or str(set_root(a.run, a.epoch))

    task = TASKS[a.task]
    dataset = load_planning_dataset(a.dataset or task["dataset"], task["cols"])
    process = fit_scalers(dataset, task["cols"])
    tf = image_transform()

    model, decoder = load_run(a.run, a.epoch, decoder=a.num_videos > 0 or bool(a.halluc_out))
    if a.halluc_out and (decoder is None or (getattr(model, "flow_head", None) is not None and not a.mse_head)):
        raise SystemExit("--halluc-set needs the run's decoder, and --mse-head for flow runs "
                         "(the recorded imagination is the MSE head's)")
    if a.num_videos > 0 and decoder is None:
        raise SystemExit(f"--num-videos needs a decoder, {a.run} has no decoder_epoch_{a.epoch}.pt")
    overrides = {k: v for k, v in dict(sample_steps=a.sample_steps, plan_noise=a.plan_noise,
                                       risk_weight=a.risk_weight).items() if v is not None}
    if a.mse_head:
        overrides["flow_variant"] = "none"
    for k, v in overrides.items():
        assert hasattr(model, k), k
        setattr(model, k, v)

    world = swm.World(env_name=task["env"], num_envs=a.num_eval, max_episode_steps=2 * a.budget, image_shape=(224, 224))
    policy = PlanPolicy(
        solver=cem_solver(model, a.cem_samples, a.seed),
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=FRAMESKIP),
        process=process, transform={"pixels": tf, "goal": tf}, decoder=decoder,
        exec_noise=a.exec_noise, act_std=np.sqrt(process["action"].var_), seed=a.seed,
        halluc=dict(n_seeds=a.halluc_seeds, seed=a.seed) if a.halluc_out else None,
    )
    world.set_policy(policy)
    episodes, starts = pick_tasks(dataset, a.num_eval, a.goal_offset, a.seed)
    sk = task["state_key"]
    t0 = time.time()
    m = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=a.goal_offset, eval_budget=a.budget,
        episodes_idx=episodes, callables=[
            {"method": "_set_state", "args": {"state": {"value": sk}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": f"goal_{sk}"}}},
        ],
    )
    secs = time.time() - t0
    succ = np.asarray(m["episode_successes"]).astype(bool)

    res = {**vars(a),
           "success_rate": 100 * float(succ.mean()),  # percent, as eval.py
           "successes": succ.astype(int).tolist(),
           "seconds": round(secs, 1), "plans": len(policy.plan_times),
           # CEM plans one env at a time (batch_size=1): seconds per env per replan
           "sec_per_plan_per_env": round(sum(policy.plan_times) / max(sum(policy.plan_envs), 1), 3),
           "gpu": torch.cuda.get_device_name(), "max_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if a.halluc_out:
        res["halluc_windows"] = write_halluc(policy, model, decoder, episodes, starts, succ, a)
    if a.num_videos > 0:
        out = Path(a.out) if a.out else run_dir("plan", a.run, a.epoch)
        out.mkdir(parents=True, exist_ok=True)
        write_videos(policy, episodes, starts, succ, a.num_videos, a.fps, out)
        (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    with open(run_dir("plan") / a.log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
