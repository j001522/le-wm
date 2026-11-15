"""Planning success with CEM in the simulator, on the root eval.py tasks (same seed -> same tasks),
optionally with videos of the plans: simulation | the world model's decoded imagination.

    python -m flow_dynamics.eval_plan --model head.pt                              (stage 1, released encoder)
    python -m flow_dynamics.eval_plan --run pusht_LeWM-D --epoch 10                (trained run)
    python -m flow_dynamics.eval_plan --run pusht_LeWM-D --epoch 10 --num-videos 10

Model: --run/--epoch loads <swm>/checkpoints/<run>/weights_epoch_<epoch>.pt (and its decoder for
videos); otherwise --model is 'pretrained' or a stage-1 checkpoint in flow_dynamics/save with the
released encoder. --sample-steps/--plan-noise/--risk-weight/--mse-head override flow attributes.

Every run appends one json line to flow_dynamics/save/<--log>.

Videos (--num-videos N > 0, runs with a decoder only): at every replan the chosen plan (CEM mean)
is rolled out by the world model (one noise sample for flow variants) and every predicted latent is
decoded. The model predicts once per action block (FRAMESKIP env steps), so the video has one frame
per block boundary: the observation after k blocks of the current plan next to the model's
prediction for that same moment (k = 0 is the encoded and decoded current observation). At a
replan, the old plan's last prediction is shown against the state it led to, then the new plan
starts at k = 0. Written for the first N episodes: <out>/<run>/ep<k>_<success|fail>.mp4, a tiled
overview.mp4, and results.json.

--exec-noise adds N(0, (exec_noise * action std)^2) to every executed env action (normalised units
as in training); videos show the noisy execution.
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from flow_dynamics.common import (ARTIFACTS, FRAMESKIP, TASKS, cem_solver, fit_scalers, full_model,
                                  image_transform, load_planning_dataset, load_run, pick_tasks)


def to_uint8(x):
    """Last frame of an env observation as (H, W, 3) uint8 (tolerates a history axis / CHW)."""
    x = np.asarray(x)
    while x.ndim > 3:
        x = x[-1]
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255 if x.max() <= 1.0 else np.clip(x, 0, 255)).astype(np.uint8)
    return x.copy()  # the env reuses its observation buffers: keep a snapshot, not a view


class PlanPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that times every solver call and optionally
    - records sim frames and, at each replan, the decoded imagined rollout (decoder given), and
    - adds Gaussian noise to the executed actions (exec_noise > 0)."""

    def __init__(self, *args, decoder=None, exec_noise=0.0, act_std=None, seed=0, **kw):
        super().__init__(*args, **kw)
        self.decoder = decoder
        self.exec_noise, self.act_std = exec_noise, act_std
        self.rng = np.random.default_rng(seed)
        st = spt.data.dataset_stats.ImageNet
        self.img_mean = torch.tensor(st["mean"]).cuda()[:, None, None]
        self.img_std = torch.tensor(st["std"]).cuda()[:, None, None]
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
        embs = []
        for b in range(actions.size(0)):  # one env at a time: get_cost's goal broadcast assumes batch 1, as in CEM
            ex = {k: v[b:b + 1].to(dev).unsqueeze(1) if torch.is_tensor(v) else v for k, v in info.items()}
            model.get_cost(ex, actions[b:b + 1].to(dev, torch.float32).unsqueeze(1))  # (1, 1, horizon, block*A)
            embs.append(ex["predicted_emb"][:, 0])                   # (1, H + horizon, D): context + predictions
        emb = torch.cat(embs)
        H = info["pixels"].size(1)
        B, T, D = emb.shape
        img = self.decoder(emb.reshape(B * T, D).float())
        img = (img * self.img_std + self.img_mean).clamp(0, 1).mul(255).byte()
        img = img.reshape(B, T, *img.shape[1:]).permute(0, 1, 3, 4, 2).cpu().numpy()
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
                self.imag[i], self.since_plan[i] = self._pending[row], 0
        for i in range(n):
            if self.imag[i] is None or np.isnan(action[i]).all():
                continue
            # one video frame per action block: the observation after k blocks of the current plan
            # next to the world model's prediction for that same moment
            if self.since_plan[i] % FRAMESKIP == 0:
                k = self.since_plan[i] // FRAMESKIP
                if k < len(self.imag[i]):
                    self.frames[i].append((to_uint8(px[i]), self.imag[i][k], self.goal[i], k))
            self.since_plan[i] += 1


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
    import imageio.v2 as imageio

    tiles = []
    for i, frames in enumerate(policy.frames[:n]):
        if not frames:
            continue
        label = f"ep {episodes[i]} step {starts[i]} -> {'success' if succ[i] else 'fail'}"
        vid = [compose(s, im, g, k, succ[i], label) for s, im, g, k in frames]
        imageio.mimsave(out / f"ep{i:02d}_{'success' if succ[i] else 'fail'}.mp4", vid, fps=fps, macro_block_size=1)
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
        grid.append(g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2])  # libx264 needs even dimensions
    imageio.mimsave(out / "overview.mp4", grid, fps=fps, macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="trained run under <swm>/checkpoints (overrides --model)")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--model", default="pretrained", help="'pretrained' or a stage-1 checkpoint in flow_dynamics/save")
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
    p.add_argument("--log", default="plan_results.jsonl", help="json-lines log in flow_dynamics/save")
    p.add_argument("--num-videos", type=int, default=0, help="save videos for the first N episodes (needs a decoder)")
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--out", default=str(ARTIFACTS / "videos"), help="videos go to <out>/<run>/")
    a = p.parse_args()

    task = TASKS[a.task]
    dataset = load_planning_dataset(a.dataset or task["dataset"], task["cols"])
    process = fit_scalers(dataset, task["cols"])
    tf = image_transform()

    if a.run:
        model, decoder = load_run(a.run, a.epoch, decoder=a.num_videos > 0)
        name = f"{a.run}/e{a.epoch}"
        if a.num_videos > 0 and decoder is None:
            raise SystemExit(f"--num-videos needs a decoder, {a.run} has no decoder_epoch_{a.epoch}.pt")
    else:
        model, decoder, name = full_model(a.model, a.task), None, a.model
        if a.num_videos > 0:
            raise SystemExit("--num-videos needs a trained run with a decoder (--run)")
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

    res = {**vars(a), "model": name,
           "success_rate": 100 * float(succ.mean()),  # percent, as eval.py
           "successes": succ.astype(int).tolist(),
           "seconds": round(secs, 1), "plans": len(policy.plan_times),
           # CEM plans one env at a time (batch_size=1): seconds per env per replan
           "sec_per_plan_per_env": round(sum(policy.plan_times) / max(sum(policy.plan_envs), 1), 3),
           "gpu": torch.cuda.get_device_name(), "max_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if a.num_videos > 0:
        out = Path(a.out) / a.run
        out.mkdir(parents=True, exist_ok=True)
        write_videos(policy, episodes, starts, succ, a.num_videos, a.fps, out)
        (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    with open(ARTIFACTS / a.log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
