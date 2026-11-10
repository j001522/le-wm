"""Evaluate plain CEM vs the bridge planner on identical long-range tasks.

Starts come from episodes that are NOT in the graph, so the bridge is tested on
states it never saw as nodes. Each goal is the state `--goal-offset` env steps
later in the same episode.
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from stable_worldmodel.solver import CEMSolver
from torchvision.transforms import v2 as transforms

from bridge.common import TASKS, add_task_arg, load_lewm
from bridge.decoder import load_decoder
from bridge.graph_planner import GraphSolver, load_edges
from bridge.model import BridgeDiT
from bridge.planner import BridgeSolver, IndexedPolicy, LatentCost
from bridge.visualize import add_subgoal_panel, save_plan_figures


def callables_for(state_key):
    return [
        {"method": "_set_state", "args": {"state": {"value": state_key}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": f"goal_{state_key}"}}},
    ]


def load_bridge(path):
    ck = torch.load(path, map_location="cpu")
    bridge = BridgeDiT(v_scale=ck["v_scale"], sigma=ck["sigma"]).cuda().eval().requires_grad_(False)
    bridge.load_state_dict(ck["model"])
    return bridge, ck


def pick_tasks(dataset, n, goal_offset, exclude, seed):
    key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep, step = dataset.get_col_data(key), dataset.get_col_data("step_idx")
    max_step = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(max_step, ep, step)
    ok = (step + goal_offset <= max_step[ep]) & ~np.isin(ep, exclude)
    rows = np.flatnonzero(ok)
    # At most one task per episode, so tasks do not repeat near-identical states.
    rng = np.random.default_rng(seed)
    rows = rows[rng.permutation(len(rows))]
    _, first = np.unique(ep[rows], return_index=True)
    rows = np.sort(rng.choice(rows[first], n, replace=False))
    return ep[rows].tolist(), step[rows].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planner", choices=["cem", "bridge", "graph"], required=True)
    p.add_argument("--snap", action="store_true", help="bridge: replace each subgoal with its nearest graph node")
    p.add_argument("--ahead", type=float, default=25.0, help="graph: path cost to the subgoal")
    add_task_arg(p)
    p.add_argument("--bridge", default=None, help="defaults to bridge.pt in the task's artifacts")
    p.add_argument("--num-eval", type=int, default=20)
    p.add_argument("--goal-offset", type=int, default=None, help="defaults to the task's value")
    p.add_argument("--budget", type=int, default=None, help="defaults to the task's value")
    p.add_argument("--n-paths", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=96)
    p.add_argument("--c", type=float, default=10.0)
    p.add_argument("--close-dist", type=float, default=None)
    p.add_argument("--r", type=float, default=None, help="override the subgoal distance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video", default=None, help="directory for one panel mp4 per task")
    p.add_argument("--trace", default=None, help="save the bridge planner's per-replan trace here")
    a = p.parse_args()
    task = TASKS[a.task]
    a.goal_offset = a.goal_offset or task.goal_offset
    a.budget = a.budget or task.budget
    decoder_path = task.artifacts / "decoder.pt"

    world = swm.World(
        env_name=task.env, num_envs=a.num_eval,
        max_episode_steps=2 * a.budget, image_shape=(224, 224),
    )
    dataset = swm.data.HDF5Dataset(
        task.dataset, keys_to_cache=list(task.columns), cache_dir=swm.data.utils.get_cache_dir(),
    )
    process = {}
    for col in task.columns:
        values = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    tf = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(224),
    ])

    model = load_lewm(task.ckpt)
    model.interpolate_pos_encoding = True
    cem_args = dict(batch_size=16, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda", seed=a.seed)
    if a.planner == "cem":
        # LeWM.get_cost only broadcasts its goal correctly with batch_size=1.
        solver = CEMSolver(model=model, **{**cem_args, "batch_size": 1})
    else:
        graph = np.load(task.artifacts / "graph.npz")
        nodes = torch.from_numpy(graph["z"]).cuda()
        bridge_path = a.bridge or task.artifacts / "bridge.pt"
        if a.planner == "graph":
            # r comes from the bridge checkpoint so the "close to goal" test matches the bridge's.
            paths = np.load(task.artifacts / "paths.npz")
            r = torch.load(bridge_path, map_location="cpu")["r"]
            solver = GraphSolver(
                model, CEMSolver(model=LatentCost(model), **cem_args), nodes, load_edges(graph, paths),
                float(paths["threshold"]), a.close_dist or a.r or r, a.ahead,
            )
        else:
            bridge, ck = load_bridge(bridge_path)
            solver = BridgeSolver(
                model, bridge, CEMSolver(model=LatentCost(model), **cem_args), sigma=ck["sigma"],
                r=a.r or ck["r"], close_dist=a.close_dist, n_paths=a.n_paths, steps=a.sample_steps, c=a.c,
                snap_nodes=nodes if a.snap else None,
            )
    world.set_policy(IndexedPolicy(
        solver=solver, config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=process, transform={"pixels": tf, "goal": tf},
    ))

    exclude = np.unique(np.load(task.artifacts / "graph.npz")["episodes"])
    episodes, starts = pick_tasks(dataset, a.num_eval, a.goal_offset, exclude, a.seed)
    first = dataset.load_chunk(np.array(episodes), np.array(starts), np.array(starts) + 1)
    start_state = np.stack([c[task.state_key][0].numpy() for c in first])
    t0 = time.time()
    m = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=a.goal_offset, eval_budget=a.budget,
        episodes_idx=episodes, callables=callables_for(task.state_key), video=a.video,
    )
    goal, final = world.infos[f"goal_{task.state_key}"][:, -1], world.infos[task.state_key][:, -1]
    lo, hi = task.tracked

    def goal_dist(state):
        return np.linalg.norm(state[:, lo:hi] - goal[:, lo:hi], axis=1).round(1).tolist()

    if a.trace and hasattr(solver, "trace"):
        torch.save({"trace": solver.trace, "episodes": episodes, "starts": starts}, a.trace)
    print(json.dumps({
        "episodes": episodes, "starts": starts,
        "dist_start": goal_dist(start_state), "dist_final": goal_dist(final),
        "task": a.task, "planner": a.planner, "goal_offset": a.goal_offset, "budget": a.budget,
        "success_rate": float(m["success_rate"]),
        "successes": np.asarray(m["episode_successes"]).astype(int).tolist(),
        "stats": getattr(solver, "stats", None), "seconds": round(time.time() - t0, 1),
    }), flush=True)
    if a.video and a.planner != "cem" and decoder_path.exists():
        decoder = load_decoder(decoder_path)
        add_subgoal_panel(a.video, solver.trace, decoder)
        if a.planner == "bridge":
            save_plan_figures(a.video, solver.trace, decoder, bridge, ck["sigma"], a.sample_steps, a.c)


if __name__ == "__main__":
    main()
