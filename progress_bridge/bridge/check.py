"""Offline bridge check on episodes that are not in the graph. No simulator needed.

For start/goal pairs a fixed number of env steps apart in unseen episodes, compare
generated paths with straight-line interpolation (the control):
  - end error:   ||x(1) - z_goal||
  - off-manifold: distance from each path point to the nearest dataset node
  - residual:    r = min_a ||P(z, a) - z'||^2 between consecutive path points
                 spaced one predictor step apart
  - subgoal:     squared final-state cost of a CEM plan toward the first subgoal,
                 next to plans toward the goal and toward the real frame one cycle ahead
"""

import argparse

import h5py
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver

from bridge.common import TASKS, add_task_arg, latents_path, load_lewm
from bridge.eval import load_bridge
from bridge.planner import LatentCost, first_subgoal, sample_paths
from bridge.teacher import reachability_residual


@torch.no_grad()
def nearest_dist(points, nodes, chunk=4096):
    out = [torch.cdist(p, nodes).min(dim=1).values for p in points.reshape(-1, points.shape[-1]).split(chunk)]
    return torch.cat(out).reshape(points.shape[:-1])


def cem_cost(model, z, target, horizon=5):
    solver = CEMSolver(LatentCost(model), batch_size=128, num_samples=300, n_steps=30, topk=30, device="cuda", seed=0)
    solver.configure(
        action_space=Box(-1, 1, shape=(1, 2)), n_envs=len(z),
        config=PlanConfig(horizon=horizon, receding_horizon=horizon, action_block=5),
    )
    n = len(z)
    return np.asarray(solver.solve({
        "pixels": torch.zeros(n, 1, 1), "goal": torch.zeros(n, 1, 1),
        "emb": z[:, None], "goal_emb": target[:, None],
    })["costs"])


def q(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return "p10 {:7.2f}  p50 {:7.2f}  p90 {:7.2f}".format(*np.percentile(x, [10, 50, 90]))


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--bridge", default=None, help="defaults to bridge.pt in the task's artifacts")
    p.add_argument("--tasks", type=int, default=128)
    p.add_argument("--cem-tasks", type=int, default=32)
    p.add_argument("--offset", type=int, default=None, help="defaults to the task's goal offset")
    p.add_argument("--cycle", type=int, default=25, help="env steps to the real reference subgoal")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--c", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    task = TASKS[a.task]
    a.offset = a.offset or task.goal_offset
    a.bridge = a.bridge or task.artifacts / "bridge.pt"
    g = np.load(task.artifacts / "graph.npz")
    nodes = torch.from_numpy(g["z"]).cuda()
    bridge, ck = load_bridge(a.bridge)
    sigma, r = ck["sigma"], ck["r"]
    print(f"checkpoint step {ck['step']} | sigma {sigma} | r {r:.2f}")

    with h5py.File(latents_path(task.dataset), "r") as f:
        ep_len, off = f["ep_len"][:], f["ep_offset"][:]
        pool = np.setdiff1d(np.flatnonzero(ep_len > a.offset), g["episodes"])
        ep = np.sort(rng.choice(pool, a.tasks, replace=False))
        pairs = []
        for e in ep:
            t = off[e] + rng.integers(0, ep_len[e] - a.offset)
            pairs.append((f["z"][t], f["z"][t + a.offset], f["z"][t + a.cycle]))
    zs = torch.from_numpy(np.stack([x[0] for x in pairs]).astype(np.float32)).cuda()
    zg = torch.from_numpy(np.stack([x[1] for x in pairs]).astype(np.float32)).cuda()
    real_sub = torch.from_numpy(np.stack([x[2] for x in pairs]).astype(np.float32)).cuda()
    print(f"{len(ep)} tasks, offset {a.offset} | ||start-goal|| {q((zs - zg).norm(dim=1))}")

    paths = sample_paths(bridge, zs, zg, sigma, a.steps, a.c)  # (B, steps+1, D)
    w = torch.linspace(0, 1, a.steps + 1, device=zs.device)[None, :, None]
    line = zs[:, None] + w * (zg - zs)[:, None]

    print(f"end error ||x(1)-goal||     {q((paths[:, -1] - zg).norm(dim=1))}")
    print(f"off-manifold, bridge mid    {q(nearest_dist(paths[:, 1:-1], nodes).mean(1))}")
    print(f"off-manifold, straight mid  {q(nearest_dist(line[:, 1:-1], nodes).mean(1))}")
    print(f"typical real hop            {float(g['median_hop']):7.2f}, d_link {float(g['d_link']):.2f}")

    model = load_lewm(task.ckpt)
    # Compare points 1/24 of the path apart, about one predictor step on a 100-step path.
    gap = max(1, a.steps // 24)
    for name, curve in (("bridge", paths), ("straight", line)):
        u = curve[:, :-gap].reshape(-1, curve.shape[-1])
        v = curve[:, gap:].reshape(-1, curve.shape[-1])
        res = reachability_residual(model, u, v, steps=20)
        print(f"step residual, {name:<9}    {q(res)} | ||step|| {q((v - u).norm(dim=-1))}")

    sub = first_subgoal(paths, r)
    print(f"||subgoal-start||           {q((sub - zs).norm(dim=1))} (target r {r:.2f})")
    m = a.cem_tasks
    cost_sub = cem_cost(model, zs[:m], sub[:m])
    cost_goal = cem_cost(model, zs[:m], zg[:m])
    cost_line = cem_cost(model, zs[:m], first_subgoal(line, r)[:m])
    print(f"CEM cost to real +{a.cycle} frame {q(cem_cost(model, zs[:m], real_sub[:m]))}")
    print(f"CEM cost to bridge subgoal  {q(cost_sub)}")
    print(f"CEM cost to straight subgoal{q(cost_line)}")
    print(f"CEM cost to final goal      {q(cost_goal)}")

    many = sample_paths(bridge, zs[:8].repeat_interleave(16, 0), zg[:8].repeat_interleave(16, 0), sigma, a.steps, a.c)
    subs = first_subgoal(many, r).reshape(8, 16, -1)
    print(f"subgoal spread within task  {q(torch.cdist(subs, subs).flatten(1).sum(1) / (16 * 15))}")


if __name__ == "__main__":
    main()
