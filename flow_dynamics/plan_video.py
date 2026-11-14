"""Plan with CEM in the simulator and save videos: left the simulation (goal inset top-right),
right the world model's decoded imagination of the plan being executed.

python -m flow_dynamics.plan_video --run pusht_LeWM-D --epoch 10 --num-eval 50 --num-videos 10

Same task sampling, CEM settings and success metric as eval_plan.py. At every replan the chosen
plan (CEM mean) is rolled out by the world model (one noise sample for flow variants) and every
predicted latent is decoded with the run's detached decoder. The model predicts once per action
block (FRAMESKIP env steps), so the video has one frame per block boundary: the observation after k
blocks of the current plan next to the model's prediction for that same moment (k = 0 is the encoded
and decoded current observation). At a replan, the old plan's last prediction is shown against the
state it led to, then the new plan starts at k = 0. Success is measured on all --num-eval episodes; videos are written for the first --num-videos:
<out>/<run>/ep<k>_<success|fail>.mp4, a tiled overview.mp4, and results.json.
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from stable_worldmodel.solver import CEMSolver
from torchvision.transforms import v2 as transforms

import module
from flow_dynamics.common import ARTIFACTS, FRAMESKIP, SWM

COLS, STATE_KEY = ["action", "proprio", "state"], "state"


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


def pick_tasks(dataset, n, goal_offset, seed):
    """eval_plan.pick_tasks, also for lance (whose column_names omit the episode_idx index column)."""
    try:
        ep = dataset.get_col_data("episode_idx")
    except Exception:
        ep = dataset.get_col_data("ep_idx")
    step = dataset.get_col_data("step_idx")
    max_step = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(max_step, ep, step)
    valid = np.nonzero(step <= max_step[ep] - goal_offset)[0]  # eval.py: len - offset - 1
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, size=n, replace=False)])
    return ep[rows].tolist(), step[rows].tolist()


class RecordingPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that records sim frames and, at each replan, the decoded imagined rollout."""

    def __init__(self, *args, decoder=None, img_mean=None, img_std=None, **kw):
        super().__init__(*args, **kw)
        self.decoder, self.img_mean, self.img_std = decoder, img_mean, img_std
        self.frames, self.imag, self.since_plan, self.goal = None, None, None, None
        self.plan_times = []

    def set_env(self, env):
        super().set_env(env)
        n = env.num_envs
        self.frames = [[] for _ in range(n)]          # (sim frame, imagined frame) per env step
        self.imag = [None] * n                         # decoded rollout of the current plan
        self.since_plan = [0] * n
        self.goal = [None] * n
        solve = self.solver

        def solve_and_imagine(info, init_action=None):
            t = time.time()
            out = solve(info, init_action=init_action)
            self.plan_times.append(time.time() - t)
            self._imagine(info, out["actions"])
            return out

        self.solver_call = solve_and_imagine

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
        # route the solver call through solve_and_imagine (dunder calls skip instance attributes,
        # so swap in a callable proxy for the duration of the call)
        solver = self.solver
        self.solver = _CallWrapper(solver, self.solver_call)
        try:
            action = super().get_action(info_dict, **kwargs)
        finally:
            self.solver = solver
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
        return action


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="pusht_LeWM-D", help="checkpoint dir under $STABLEWM_HOME/checkpoints")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--dataset", default="pusht_expert_train.lance")
    p.add_argument("--env", default="swm/PushT-v1")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--num-videos", type=int, default=10, help="episodes to save videos for (the first N)")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--out", default=str(ARTIFACTS / "videos"))
    a = p.parse_args()

    out = Path(a.out) / a.run
    out.mkdir(parents=True, exist_ok=True)
    ds_path = a.dataset if os.path.isabs(a.dataset) else str(SWM / "datasets" / a.dataset)
    dataset = swm.data.load_dataset(ds_path, keys_to_cache=COLS)
    process = {}
    for col in COLS:
        v = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    stats = spt.data.dataset_stats.ImageNet
    tf = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**stats), transforms.Resize(224),
    ])

    model = swm.wm.utils.load_pretrained(f"{a.run}/weights_epoch_{a.epoch}.pt").cuda().eval().requires_grad_(False)
    dck = torch.load(SWM / "checkpoints" / a.run / f"decoder_epoch_{a.epoch}.pt", map_location="cpu")
    decoder = module.PixelDecoder(**{k: v for k, v in dck["cfg"].items() if k != "_target_"}).cuda().eval()
    decoder.load_state_dict(dck["state_dict"])
    img_mean = torch.tensor(stats["mean"]).cuda()[:, None, None]
    img_std = torch.tensor(stats["std"]).cuda()[:, None, None]

    world = swm.World(env_name=a.env, num_envs=a.num_eval, max_episode_steps=2 * a.budget, image_shape=(224, 224))
    solver = CEMSolver(model=model, batch_size=1, num_samples=a.cem_samples, var_scale=1.0,
                       n_steps=30, topk=30, device="cuda", seed=a.seed)
    policy = RecordingPolicy(
        solver=solver, config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=process, transform={"pixels": tf, "goal": tf},
        decoder=decoder, img_mean=img_mean, img_std=img_std,
    )
    world.set_policy(policy)
    episodes, starts = pick_tasks(dataset, a.num_eval, a.goal_offset, a.seed)
    t0 = time.time()
    m = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=a.goal_offset, eval_budget=a.budget,
        episodes_idx=episodes, callables=[
            {"method": "_set_state", "args": {"state": {"value": STATE_KEY}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": f"goal_{STATE_KEY}"}}},
        ],
    )
    secs = time.time() - t0
    succ = np.asarray(m["episode_successes"]).astype(bool)

    tiles = []
    for i, frames in enumerate(policy.frames[: a.num_videos]):
        if not frames:
            continue
        label = f"ep {episodes[i]} step {starts[i]} -> {'success' if succ[i] else 'fail'}"
        vid = [compose(s, im, g, k, succ[i], label) for s, im, g, k in frames]
        imageio.mimsave(out / f"ep{i:02d}_{'success' if succ[i] else 'fail'}.mp4", vid, fps=a.fps, macro_block_size=1)
        tiles.append(vid)
    if tiles:  # overview: up to 16 episodes tiled 4 wide, padded with their last frame
        tiles = tiles[:16]
        L = max(len(v) for v in tiles)
        tiles = [v + [v[-1]] * (L - len(v)) for v in tiles]
        blank = np.zeros_like(tiles[0][0])
        cols = 4
        grid = []
        for t in range(L):
            cells = [v[t] for v in tiles] + [blank] * (-len(tiles) % cols)
            rows = [np.concatenate(cells[r:r + cols], 1) for r in range(0, len(cells), cols)]
            g = np.concatenate(rows, 0)[::2, ::2]
            grid.append(g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2])  # libx264 needs even dimensions
        imageio.mimsave(out / "overview.mp4", grid, fps=a.fps, macro_block_size=1)

    res = {**vars(a), "success_rate": float(succ.mean()), "successes": succ.astype(int).tolist(),
           "seconds": round(secs, 1), "plans": len(policy.plan_times),
           "sec_per_plan_call": round(float(np.mean(policy.plan_times)), 2) if policy.plan_times else None,
           "gpu": torch.cuda.get_device_name(), "max_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
