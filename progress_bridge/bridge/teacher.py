"""Step 3: teacher paths from Dijkstra on the graph, with lazy edge checking.

Only cross-episode edges that lie on a used path are checked. An edge i -> j
passes if some action makes the frozen predictor land near z_j:
    r(i, j) = min_a ||P(z_i, a) - z_j||^2
Edges that fail are deleted and the affected sources re-run until no used edge
is unchecked. The threshold comes from the residuals of real within-episode
hops, and a CEM spot-check reports how well the fast Adam estimate matches it.
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from gymnasium.spaces import Box
from scipy.sparse.csgraph import dijkstra
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver

from bridge.common import TASKS, add_task_arg, load_lewm, predict_one
from bridge.planner import LatentCost


def reachability_residual(model, zs, zt, steps=20, lr=0.2, a_clip=3.0, batch=8192, a_dim=10):
    """Batched min_a ||P(z, a) - z'||^2 over normalised action chunks, by Adam."""
    out = []
    for i in range(0, len(zs), batch):
        z, t = zs[i : i + batch].cuda(), zt[i : i + batch].cuda()
        a = torch.zeros(len(z), a_dim, device=z.device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=lr)
        best = torch.full((len(z),), float("inf"), device=z.device)
        with torch.enable_grad():
            for _ in range(steps):
                res = (predict_one(model, z, a) - t).pow(2).sum(-1)
                best = torch.minimum(best, res.detach())
                opt.zero_grad()
                res.sum().backward()
                opt.step()
                a.data.clamp_(-a_clip, a_clip)
        res = (predict_one(model, z, a.detach()) - t).pow(2).sum(-1)
        out.append(torch.minimum(best, res).detach().cpu())
    return torch.cat(out)


def cem_residual(model, zs, zt, num_samples=300, n_steps=30, topk=30):
    """Reference value for the spot-check: the same residual found by full CEM."""
    solver = CEMSolver(
        LatentCost(model), batch_size=256, num_samples=num_samples,
        n_steps=n_steps, topk=topk, device="cuda", seed=0,
    )
    solver.configure(
        action_space=Box(-1, 1, shape=(1, 2)), n_envs=len(zs),
        config=PlanConfig(horizon=1, receding_horizon=1, action_block=5),
    )
    n = len(zs)
    return torch.as_tensor(np.asarray(solver.solve({
        "pixels": torch.zeros(n, 1, 1), "goal": torch.zeros(n, 1, 1),
        "emb": zs[:, None], "goal_emb": zt[:, None],
    })["costs"]))


def walk_back(pred, source, targets):
    """Vectorised path reconstruction from a Dijkstra predecessor array.

    Returns a list of source-to-target node arrays, one per target.
    """
    cur = targets.copy()
    steps = [cur.copy()]
    while True:
        alive = cur != source
        if not alive.any():
            break
        cur = np.where(alive, pred[cur], cur)
        steps.append(cur.copy())
    table = np.stack(steps)  # (max_hops + 1, n_targets), padded with the source
    hops = (table != source).sum(0)
    return [table[h::-1, j] for j, h in enumerate(hops)]


class EdgeTable:
    """Directed edges, addressed by the key src * N + dst."""

    def __init__(self, src, dst, cost, is_cross, n):
        self.n = n
        keys = src.astype(np.int64) * n + dst
        order = np.argsort(keys)
        self.keys, self.cost, self.is_cross = keys[order], cost[order], is_cross[order]
        self.alive = np.ones(len(keys), bool)
        self.checked = np.zeros(len(keys), bool)
        self.residual = np.full(len(keys), np.nan, dtype=np.float32)

    def lookup(self, u, v):
        return np.searchsorted(self.keys, u.astype(np.int64) * self.n + v)

    def csr(self):
        k = self.keys[self.alive]
        return sp.csr_matrix(
            (self.cost[self.alive], (k // self.n, k % self.n)), shape=(self.n, self.n)
        )


def paths_for_sources(graph, sources, targets_of, n_targets, rng, chunk=64):
    """Shortest paths {source: [node arrays]}.

    Most nodes are unreachable from a given source, so targets are drawn from the
    reachable set on a source's first visit. They stay fixed afterwards, and any
    that a later edge removal cuts off are skipped.
    """
    out = {}
    for i in range(0, len(sources), chunk):
        batch = sources[i : i + chunk]
        dist, pred = dijkstra(graph, directed=True, indices=batch, return_predecessors=True)
        for row, s in enumerate(batch):
            if s not in targets_of:
                reach = np.flatnonzero(np.isfinite(dist[row]))
                targets_of[s] = rng.choice(reach, min(n_targets, len(reach)), replace=False)
            t = targets_of[s]
            t = t[np.isfinite(dist[row, t])]
            out[s] = walk_back(pred[row], s, t) if len(t) else []
    return out


def uses_edge(paths, n, edge_keys):
    """True if any of the node paths traverses one of the given edges."""
    if not paths:
        return False
    keys = np.concatenate([q[:-1].astype(np.int64) * n + q[1:] for q in paths])
    return bool(np.isin(keys, edge_keys).any())


def calibrate(model, z, table, n_real, n_cross, n_cem, percentile, rng):
    z_t = torch.from_numpy(z)
    u, v = table.keys // table.n, table.keys % table.n
    real = rng.choice(np.flatnonzero(~table.is_cross), n_real, replace=False)
    cross = rng.choice(np.flatnonzero(table.is_cross), n_cross, replace=False)
    r_real = reachability_residual(model, z_t[u[real]], z_t[v[real]]).numpy()
    r_cross = reachability_residual(model, z_t[u[cross]], z_t[v[cross]]).numpy()
    threshold = float(np.percentile(r_real, percentile))
    cem_real = None

    def show(name, x):
        print(f"  {name:<22} p10 {np.percentile(x, 10):7.2f}  p50 {np.percentile(x, 50):7.2f}  "
              f"p90 {np.percentile(x, 90):7.2f}  p99 {np.percentile(x, 99):7.2f}")

    print("calibration, squared residual:")
    show("real hops, Adam", r_real)
    show("cross candidates, Adam", r_cross)
    for name, idx, r in (("real hops, CEM", real, r_real), ("cross candidates, CEM", cross, r_cross)):
        idx, r = idx[:n_cem], r[:n_cem]
        c = cem_residual(model, z_t[u[idx]], z_t[v[idx]]).numpy()
        show(name, c)
        if cem_real is None:
            cem_real = c
        agree = ((r <= threshold) == (c <= threshold)).mean()
        print(f"  {'':<22} Adam/CEM pass-fail agreement {agree:.1%}, mean Adam-CEM {np.mean(r - c):+.2f}")
    if threshold > 3 * np.percentile(cem_real, 99):
        print("WARNING: Adam stalls on real hops, so this threshold is too loose. "
              "Pass --threshold near the CEM p99 of real hops.")
    print(f"threshold {threshold:.2f} (p{percentile:g} of real hops); "
          f"{(r_cross <= threshold).mean():.1%} of candidates would pass")
    return threshold


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--sources", type=int, default=1000)
    p.add_argument("--targets-per-source", type=int, default=200)
    p.add_argument("--min-hops", type=int, default=2)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--threshold-percentile", type=float, default=95)
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    task = TASKS[a.task]
    out = task.artifacts / "paths.npz"
    g = np.load(task.artifacts / "graph.npz")
    z, n = g["z"], len(g["z"])
    model = load_lewm(task.ckpt)
    table = EdgeTable(g["src"], g["dst"], g["cost"], g["is_cross"], n)

    threshold = a.threshold
    if threshold is None:
        threshold = calibrate(model, z, table, 4000, 4000, 128, a.threshold_percentile, rng)

    sources = rng.choice(n, min(a.sources, n), replace=False)
    targets_of = {}
    z_t = torch.from_numpy(z)

    paths, dirty, t0 = {}, sources, time.time()
    for round_i in range(a.max_rounds):
        paths.update(paths_for_sources(table.csr(), dirty, targets_of, a.targets_per_source, rng))

        idx = np.unique(np.concatenate(
            [table.lookup(q[:-1], q[1:]) for s in dirty for q in paths[s]] or [np.zeros(0, int)]
        ))
        new = idx[table.is_cross[idx] & ~table.checked[idx]]

        if len(new) == 0:
            print(f"round {round_i}: no unchecked edges left")
            break
        u, v = table.keys[new] // n, table.keys[new] % n
        res = reachability_residual(model, z_t[u], z_t[v]).numpy()
        table.checked[new] = True
        table.residual[new] = res
        bad = new[res > threshold]
        table.alive[bad] = False
        print(f"round {round_i}: {len(dirty)} sources, checked {len(new)} edges, "
              f"removed {len(bad)} ({time.time() - t0:.0f}s)", flush=True)
        if len(bad) == 0:
            break

        dead = table.keys[bad]
        dirty = np.array([s for s in sources if uses_edge(paths[s], n, dead)], dtype=sources.dtype)
    else:
        # Out of rounds: paths may still use a bad edge. Dropping those keeps the set clean.
        dead = table.keys[~table.alive]
        for s in sources:
            paths[s] = [q for q in paths[s] if not uses_edge([q], n, dead)]
        print("hit --max-rounds; dropped paths through removed edges")

    nodes, ptr, cum, src_of, stitched, cross_hops, all_hops = [], [0], [], [], 0, 0, 0
    for s in sources:
        for path in paths[s]:
            if len(path) - 1 < a.min_hops:
                continue
            edge = table.lookup(path[:-1], path[1:])
            c = table.cost[edge]
            n_x = int(table.is_cross[edge].sum())
            stitched += n_x > 0
            cross_hops += n_x
            all_hops += len(edge)
            nodes.append(path)
            cum.append(np.concatenate([[0.0], np.cumsum(c)]))
            ptr.append(ptr[-1] + len(path))
            src_of.append(s)
    hops, total = np.diff(ptr) - 1, np.array([c[-1] for c in cum])

    n_cross = int(table.is_cross.sum())
    print(f"stitching: {stitched / len(hops):.1%} of paths use a cross edge; "
          f"{cross_hops / all_hops:.1%} of hops are cross edges")
    print(f"paths {len(hops)} | hops p10/50/90 {np.percentile(hops, [10, 50, 90])} | "
          f"cost p10/50/90 {np.round(np.percentile(total, [10, 50, 90]), 1)}")
    print(f"cross edges: {n_cross} candidates, {int(table.checked.sum())} checked, "
          f"{int((~table.alive).sum())} removed")

    np.savez(
        out,
        nodes=np.concatenate(nodes).astype(np.int32),
        ptr=np.array(ptr, dtype=np.int64),
        cum=np.concatenate(cum).astype(np.float32),
        source=np.array(src_of, dtype=np.int32),
        threshold=threshold,
        edge_keys=table.keys,
        edge_alive=table.alive,
        edge_residual=table.residual,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
