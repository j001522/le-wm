"""Planning success with CEM in the simulator, the released encoder, and a chosen predictor.
Tasks are sampled as in the root eval.py (same seed -> same tasks).

python -m flow_dynamics.eval_plan --model head.pt --num-eval 50 --goal-offset 25 --budget 50
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

from flow_dynamics.common import ARTIFACTS, full_model

ENV = {"pusht": ("swm/PushT-v1", "pusht_expert_train", ["action", "proprio", "state"], "state")}


class NoisyExecutionPolicy(swm.policy.WorldModelPolicy):
    """Plans as usual, then the environment executes action + N(0, (exec_noise * std)^2)
    per env step (std = dataset action std, i.e. noise in normalised units as in training)."""

    def __init__(self, *args, exec_noise=0.0, act_std=None, seed=0, **kw):
        super().__init__(*args, **kw)
        self.exec_noise, self.act_std = exec_noise, act_std
        self.rng = np.random.default_rng(seed)

    def get_action(self, info_dict, **kwargs):
        a = super().get_action(info_dict, **kwargs)
        if self.exec_noise > 0:
            a = a + self.exec_noise * self.act_std * self.rng.standard_normal(a.shape)
            space = self.env.action_space
            a = np.clip(a, space.low, space.high).astype(np.float32)
        return a


def pick_tasks(dataset, n, goal_offset, seed):
    """Same procedure as eval.py."""
    key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep, step = dataset.get_col_data(key), dataset.get_col_data("step_idx")
    max_step = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(max_step, ep, step)
    valid = np.nonzero(step <= max_step[ep] - goal_offset)[0]  # eval.py: len - offset - 1
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, size=n, replace=False)])
    return ep[rows].tolist(), step[rows].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pretrained")
    p.add_argument("--task", default="pusht")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--plan-noise", type=int, default=None)
    p.add_argument("--risk-weight", type=float, default=None)
    p.add_argument("--e2e", default=None,
                   help="end-to-end checkpoint saved by train.py (e.g. <run name>/weights_epoch_10.pt under the swm "
                        "checkpoint dir); replaces --model")
    p.add_argument("--mse-head", action="store_true", help="flow checkpoint, but plan with its MSE head")
    p.add_argument("--exec-noise", type=float, default=0.0, help="action noise executed in the env (normalised units)")
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="plan_results.jsonl")
    a = p.parse_args()

    env, ds_name, cols, state_key = ENV[a.task]
    world = swm.World(env_name=env, num_envs=a.num_eval, max_episode_steps=2 * a.budget, image_shape=(224, 224))
    dataset = swm.data.HDF5Dataset(ds_name, keys_to_cache=cols, cache_dir=swm.data.utils.get_cache_dir())
    process = {}
    for col in cols:
        v = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    tf = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(224),
    ])

    kw = {k: v for k, v in dict(sample_steps=a.sample_steps, plan_noise=a.plan_noise,
                                  risk_weight=a.risk_weight).items() if v is not None}
    if a.mse_head:
        kw["flow_variant"] = "none"
    if a.e2e:
        model = swm.wm.utils.load_pretrained(a.e2e).cuda().eval().requires_grad_(False)
        for k, v in kw.items():
            setattr(model, k, v)
        a.model = a.e2e
    else:
        model = full_model(a.model, a.task, **kw)
    solver = CEMSolver(model=model, batch_size=1, num_samples=a.cem_samples, var_scale=1.0,
                       n_steps=30, topk=30, device="cuda", seed=a.seed)
    calls = {"n": 0, "t": 0.0}
    get_cost = model.get_cost

    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.time()
        out = get_cost(*args, **kwargs)
        torch.cuda.synchronize()
        calls["n"] += 1
        calls["t"] += time.time() - t
        return out

    model.get_cost = timed
    world.set_policy(NoisyExecutionPolicy(
        solver=solver, config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=process, transform={"pixels": tf, "goal": tf},
        exec_noise=a.exec_noise, act_std=np.sqrt(process["action"].var_), seed=a.seed,
    ))
    episodes, starts = pick_tasks(dataset, a.num_eval, a.goal_offset, a.seed)
    t0 = time.time()
    m = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=a.goal_offset, eval_budget=a.budget,
        episodes_idx=episodes, callables=[
            {"method": "_set_state", "args": {"state": {"value": state_key}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": f"goal_{state_key}"}}},
        ],
    )
    res = {
        **vars(a), "success_rate": float(m["success_rate"]),
        "successes": np.asarray(m["episode_successes"]).astype(int).tolist(),
        "seconds": round(time.time() - t0, 1),
        # get_cost runs once per CEM iteration (30 per plan) for all envs at once
        "sec_per_plan_per_env": round(30 * calls["t"] / max(calls["n"], 1), 3),
    }
    print(json.dumps(res), flush=True)
    with open(ARTIFACTS / a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
