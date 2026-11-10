"""Steps 1b + 2: pick nodes from the latent cache and build the state graph.

Directed graph, because real dynamics are directed (a pushed block does not
un-push itself): within-episode edges point forward in time. Close pairs from
different episodes get a candidate edge in each direction. No CEM happens here:
candidate links are checked lazily in teacher.py.
"""

import argparse

import h5py
import numpy as np
import torch

from bridge.common import FRAMESKIP, TASKS, add_task_arg, latents_path


def select_nodes(ep_len, ep_offset, stride, max_nodes, rng):
    """Every `stride`-th frame of randomly chosen episodes, up to max_nodes.

    Whole episodes are kept, so consecutive nodes stay one predictor step apart.
    """
    frames, episodes, total = [], [], 0
    for ep in rng.permutation(len(ep_len)):
        idx = np.arange(0, ep_len[ep], stride)
        if total + len(idx) > max_nodes:
            break
        frames.append(ep_offset[ep] + idx)
        episodes.append(np.full(len(idx), ep))
        total += len(idx)
    frames, episodes = np.concatenate(frames), np.concatenate(episodes)
    order = np.argsort(frames)
    return frames[order], episodes[order]


@torch.no_grad()
def cross_episode_neighbours(z, episodes, k, d_link, chunk=2048):
    """k nearest nodes from other episodes, keeping only those within d_link."""
    zt = torch.from_numpy(z).cuda()
    ep = torch.from_numpy(episodes).cuda()
    src, dst, dist = [], [], []
    for i in range(0, len(z), chunk):
        d = torch.cdist(zt[i : i + chunk], zt)
        d[ep[i : i + chunk, None] == ep[None]] = float("inf")
        val, idx = d.topk(k, dim=1, largest=False)
        keep = val < d_link
        src.append((torch.arange(len(val), device=val.device)[:, None] + i).expand_as(idx)[keep])
        dst.append(idx[keep])
        dist.append(val[keep])
    return (torch.cat(src).cpu().numpy(), torch.cat(dst).cpu().numpy(), torch.cat(dist).cpu().numpy())


def build_edges(z, episodes, stride, k, link_percentile):
    same = episodes[:-1] == episodes[1:]
    w_src = np.flatnonzero(same)
    w_dst = w_src + 1
    hop = np.linalg.norm(z[w_dst] - z[w_src], axis=1)
    median_hop = float(np.median(hop))
    d_link = float(np.percentile(hop, link_percentile))

    c_src, c_dst, c_d = cross_episode_neighbours(z, episodes, k, d_link)
    # Make the candidate set symmetric, then drop duplicates.
    keys = np.concatenate([c_src, c_dst]) * len(z) + np.concatenate([c_dst, c_src])
    keys, first = np.unique(keys, return_index=True)
    c_d = np.concatenate([c_d, c_d])[first]
    c_src, c_dst = keys // len(z), keys % len(z)
    c_cost = np.maximum(c_d / median_hop * stride, 1e-3)

    return {
        "src": np.concatenate([w_src, c_src]).astype(np.int32),
        "dst": np.concatenate([w_dst, c_dst]).astype(np.int32),
        "cost": np.concatenate([np.full(len(w_src), float(stride)), c_cost]).astype(np.float32),
        "is_cross": np.concatenate([np.zeros(len(w_src), bool), np.ones(len(c_src), bool)]),
        "median_hop": median_hop,
        "d_link": d_link,
    }


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--stride", type=int, default=FRAMESKIP)
    p.add_argument("--max-nodes", type=int, default=50_000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--link-percentile", type=float, default=25)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    task = TASKS[a.task]
    out = task.artifacts / "graph.npz"

    rng = np.random.default_rng(a.seed)
    with h5py.File(latents_path(task.dataset), "r") as f:
        ep_len, ep_offset = f["ep_len"][:], f["ep_offset"][:]
        frames, episodes = select_nodes(ep_len, ep_offset, a.stride, a.max_nodes, rng)
        z = f["z"][:][frames].astype(np.float32)

    e = build_edges(z, episodes, a.stride, a.k, a.link_percentile)
    n_cross = int(e["is_cross"].sum())
    print(
        f"nodes {len(z)} from {len(np.unique(episodes))} episodes | "
        f"median hop {e['median_hop']:.2f}, d_link {e['d_link']:.2f} | "
        f"within edges {len(e['src']) - n_cross}, cross candidates {n_cross} "
        f"({n_cross / len(z):.1f} per node)"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, z=z, frames=frames, episodes=episodes, stride=a.stride, **e)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
