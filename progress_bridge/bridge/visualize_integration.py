"""Show the sampler at every integration step, for the first replan of each task in a trace.

For each task, one freshly sampled path (not the one the planner used) gives:
  env_<i>.mp4: start | noisy x | denoised x - gamma*eta | nearest real frame | goal
  env_<i>.png: the same at start, steps 20, 40, 60, 80 and goal

The decoder blurs even real states and never saw noisy latents, so the nearest
real frame is the panel to trust for what state a latent is close to.
"""

import argparse
from pathlib import Path

import h5py
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from bridge.common import TASKS, add_task_arg, dataset_path
from bridge.decoder import load_decoder
from bridge.eval import load_bridge
from bridge.planner import sample_paths
from bridge.visualize import PANEL, unnormalize

HEADER = 44
GAP = 6
SUBGOAL_COLOR = (220, 40, 40)


def to_panel(img, label, border=None):
    """Image array (float 0-1 or uint8) -> labelled PANEL-sized uint8 panel."""
    arr = img if img.dtype == np.uint8 else (img.clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((PANEL, PANEL), Image.BICUBIC)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), label, fill=(60, 60, 60))
    if border:
        draw.rectangle((0, 0, PANEL - 1, PANEL - 1), outline=border, width=4)
    return np.array(pil)


def compose(panels, lines):
    """One video frame: text header above a row of panels. Width and height are even."""
    n = len(panels)
    w, h = n * PANEL + (n - 1) * GAP, PANEL + HEADER
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    for j, p in enumerate(panels):
        frame[HEADER:, j * (PANEL + GAP) : j * (PANEL + GAP) + PANEL] = p
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(lines):
        draw.text((6, 4 + 13 * i), line, fill=(0, 0, 0))
    return np.array(pil)


def first_reach(dist, r):
    """Index of the first point at distance >= r from the start, or the last point."""
    far = np.flatnonzero(dist >= r)
    return int(far[0]) if len(far) else len(dist) - 1


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--trace", required=True, help="eval.py trace, for the tasks' start and goal")
    p.add_argument("--out", required=True)
    p.add_argument("--bridge", default=None, help="defaults to bridge.pt in the task's artifacts")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--c", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    task = TASKS[a.task]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    trace = torch.load(a.trace)["trace"]
    first = {}
    for t in trace:
        first.setdefault(t["env"], t)
    envs = sorted(first)
    bridge, ck = load_bridge(a.bridge or task.artifacts / "bridge.pt")
    decoder = load_decoder(task.artifacts / "decoder.pt")
    g = np.load(task.artifacts / "graph.npz")
    nodes = torch.from_numpy(g["z"]).cuda()
    r = ck["r"]

    zs = torch.stack([first[e]["z"] for e in envs]).cuda()
    zg = torch.stack([first[e]["z_goal"] for e in envs]).cuda()
    clean, noisy = sample_paths(bridge, zs, zg, ck["sigma"], a.steps, a.c, return_noisy=True)

    def decode(z):
        with torch.no_grad():
            return decoder(z).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    shown = [s for s in (20, 40, 60, 80) if s < a.steps]
    with h5py.File(dataset_path(task.dataset), "r") as f:
        for b, env in enumerate(envs):
            near = torch.cdist(clean[b], nodes).min(1)
            near_dist, near_idx = near.values.cpu().numpy(), near.indices.cpu().numpy()
            real = {i: f["pixels"][int(g["frames"][i])] for i in np.unique(near_idx)}
            den, noi = decode(clean[b]), decode(noisy[b])
            start, goal = unnormalize(first[env]["obs"]), unnormalize(first[env]["goal_img"])
            from_start = (clean[b] - zs[b]).norm(dim=-1).cpu().numpy()
            to_goal = (clean[b] - zg[b]).norm(dim=-1).cpu().numpy()
            sub = first_reach(from_start, r)
            start_panel, goal_panel = to_panel(start, "start (real)"), to_panel(goal, "goal (real)")

            frames = []
            for k in range(a.steps + 1):
                is_sub = k == sub
                marker = "  <- SUBGOAL, first point at distance >= r" if is_sub else (
                    f"  (subgoal was step {sub})" if k > sub else "")
                frames.append(compose(
                    [
                        start_panel,
                        to_panel(noi[k], "noisy x (decoded)"),
                        to_panel(den[k], "denoised (decoded)", SUBGOAL_COLOR if is_sub else None),
                        to_panel(real[near_idx[k]], f"nearest real frame, dist {near_dist[k]:.1f}"),
                        goal_panel,
                    ],
                    [
                        f"integration step {k}/{a.steps}   tau {k / a.steps:.2f}{marker}",
                        f"latent distance: from start {from_start[k]:.1f} (r = {r:.1f}), to goal {to_goal[k]:.1f}",
                    ],
                ))
            # Stop on the subgoal for a second and on the last frame for two.
            frames = frames[: sub + 1] + [frames[sub]] * a.fps + frames[sub + 1 :] + [frames[-1]] * (2 * a.fps)
            iio.imwrite(out / f"env_{env}.mp4", np.stack(frames), fps=a.fps, macro_block_size=2)

            zs_dec, zg_dec = decode(zs[b][None])[0], decode(zg[b][None])[0]
            rows = ("noisy x (decoded)", "denoised (decoded)", "nearest real frame")
            fig, axes = plt.subplots(3, len(shown) + 2, figsize=(1.9 * (len(shown) + 2), 6.4), squeeze=False)
            for ax in axes.flat:
                ax.set_xticks([]), ax.set_yticks([])
            for c, s in enumerate(shown, start=1):
                axes[0, c].imshow(noi[s])
                axes[1, c].imshow(den[s])
                axes[2, c].imshow(real[near_idx[s]])
                axes[0, c].set_title(f"step {s} (tau {s / a.steps:.2f})", fontsize=9)
                axes[1, c].set_xlabel(f"start {from_start[s]:.1f} | goal {to_goal[s]:.1f}", fontsize=8)
                axes[2, c].set_xlabel(f"dist {near_dist[s]:.1f}", fontsize=8)
            for c, title, z_img, real_img in ((0, "start", zs_dec, start), (-1, "goal", zg_dec, goal)):
                axes[0, c].imshow(z_img)
                axes[1, c].imshow(z_img)
                axes[2, c].imshow(real_img)
                axes[0, c].set_title(title, fontsize=9)
                axes[2, c].set_xlabel("real image", fontsize=8)
            for row, label in enumerate(rows):
                axes[row, 0].set_ylabel(label, fontsize=9)
            fig.suptitle(
                f"{a.task} task {env}: subgoal at step {sub} (tau {sub / a.steps:.2f}), r = {r:.1f}, "
                f"final distance to goal latent {to_goal[-1]:.1f}", fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(out / f"env_{env}.png", dpi=120)
            plt.close(fig)
    print(f"wrote {len(envs)} videos and figures to {out}")


if __name__ == "__main__":
    main()
