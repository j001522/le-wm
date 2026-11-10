"""Step 5: bridge sampling and the MPC solver that plugs into WorldModelPolicy."""

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy


class LatentCost:
    """CEM cost toward a latent goal, skipping all image encoding.

    The info dict holds 'emb' and 'goal_emb' latents. LeWM.rollout needs only the
    time dimension of 'pixels', so a dummy tensor is enough there.
    """

    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info, candidates):
        return self.model.criterion(self.model.rollout(info, candidates))


@torch.no_grad()
def sample_paths(bridge, z_start, z_goal, sigma, steps=48, c=1.0, return_noisy=False):
    """Integrate tau from 0 to 1. Returns clean estimates (B, steps + 1, D).

    x += [v - c*gamma*eta] dt + sqrt(2c) * gamma * sqrt(dt) * noise
    clean = x - gamma*eta

    With return_noisy, returns (clean, x), the second being the raw iterates.
    """
    x, dt = z_start.clone(), 1.0 / steps
    clean, noisy = [x], [x]
    for k in range(steps):
        tau = torch.full((len(x),), k * dt, device=x.device)
        gamma = (sigma * tau * (1 - tau))[:, None]
        v, eta = bridge(x, tau, z_start, z_goal)
        x = x + (v - c * gamma * eta) * dt
        if c > 0:
            x = x + (2 * c) ** 0.5 * gamma * dt**0.5 * torch.randn_like(x)
        noisy.append(x)
        next_tau = tau + dt
        next_gamma = (sigma * next_tau * (1 - next_tau))[:, None]
        if k + 1 < steps:
            _, eta = bridge(x, next_tau, z_start, z_goal)
            clean.append(x - next_gamma * eta)
        else:
            clean.append(x)  # gamma(1) = 0
    clean = torch.stack(clean, dim=1)
    return (clean, torch.stack(noisy, dim=1)) if return_noisy else clean


def first_subgoal(paths, r):
    """First point of each path at distance >= r from its start, else the last point."""
    far = (paths - paths[:, :1]).norm(dim=-1) >= r
    idx = torch.where(far.any(1), far.float().argmax(1), torch.full_like(far[:, 0], paths.shape[1] - 1, dtype=torch.long))
    return paths[torch.arange(len(paths), device=paths.device), idx]


class IndexedPolicy(WorldModelPolicy):
    """Tells the solver which env each row is: the policy drops finished envs from the batch."""

    def _prepare_info(self, info_dict):
        out = super()._prepare_info(info_dict)
        out["env_id"] = np.arange(self.env.num_envs)
        return out


class BridgeSolver:
    """Solver for WorldModelPolicy: bridge subgoals, then batched CEM toward them.

    For each env: if z is within `close_dist` of the goal, plan straight to it.
    Otherwise sample `n_paths` paths, take each path's first subgoal, run CEM
    toward all of them, and return the plan with the lowest final cost.
    """

    def __init__(self, model, bridge, cem, sigma, r, close_dist=None, n_paths=16, steps=48, c=1.0, snap_nodes=None):
        self.model, self.bridge, self.cem = model, bridge, cem
        self.sigma, self.r = sigma, r
        self.close_dist = r if close_dist is None else close_dist
        self.n_paths, self.steps, self.c = n_paths, steps, c
        self.snap_nodes = snap_nodes  # if given, subgoals are replaced by the nearest of these
        self.stats = {"bridge": 0, "direct": 0}
        self.trace = []  # one entry per replan, for visualisation

    def configure(self, *, action_space, n_envs, config):
        self.cem.configure(action_space=action_space, n_envs=n_envs, config=config)

    n_envs = property(lambda self: self.cem.n_envs)
    action_dim = property(lambda self: self.cem.action_dim)
    horizon = property(lambda self: self.cem.horizon)

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    @torch.no_grad()
    def encode_last(self, pixels):
        return self.model.encode({"pixels": pixels.cuda().float()})["emb"][:, -1]

    def propose(self, z, z_goal):
        """Subgoals for the given envs: n_paths per env, in env order."""
        start = z.repeat_interleave(self.n_paths, dim=0)
        goal = z_goal.repeat_interleave(self.n_paths, dim=0)
        paths = sample_paths(self.bridge, start, goal, self.sigma, self.steps, self.c)
        subgoals = first_subgoal(paths, self.r)
        if self.snap_nodes is not None:
            subgoals = self.snap_nodes[torch.cdist(subgoals, self.snap_nodes).argmin(1)]
        return subgoals

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        z = self.encode_last(info_dict["pixels"])
        z_goal = self.encode_last(info_dict["goal"])
        direct = (z - z_goal).norm(dim=-1) < self.close_dist
        self.stats["direct"] += int(direct.sum())
        self.stats["bridge"] += int((~direct).sum())

        owner = [torch.nonzero(direct)[:, 0]]
        subgoals = [z_goal[direct]]
        far = torch.nonzero(~direct)[:, 0]
        if len(far):
            owner.append(far.repeat_interleave(self.n_paths))
            subgoals.append(self.propose(z[far], z_goal[far]))
        owner, subgoals = torch.cat(owner), torch.cat(subgoals)

        n = len(owner)
        out = self.cem.solve({
            "pixels": torch.zeros(n, 1, 1), "goal": torch.zeros(n, 1, 1),
            "emb": z[owner][:, None], "goal_emb": subgoals[:, None],
        })
        costs = torch.as_tensor(out["costs"], device=owner.device)
        actions = out["actions"]
        env_ids = info_dict.get("env_id", np.arange(len(z)))
        chosen = []
        for env in range(len(z)):
            mine = torch.nonzero(owner == env)[:, 0]
            best = mine[costs[mine].argmin()]
            chosen.append(actions[best.cpu()])
            self.trace.append({
                "env": int(env_ids[env]), "obs": info_dict["pixels"][env, -1].cpu(), "goal_img": info_dict["goal"][env, -1].cpu(),
                "z": z[env].cpu(), "z_goal": z_goal[env].cpu(), "direct": bool(direct[env]),
                "subgoal": subgoals[best].cpu(), "subgoals": subgoals[mine].cpu(), "cost": float(costs[best]),
            })
        return {"actions": torch.stack(chosen), "costs": costs}
