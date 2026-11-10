"""Visualise bridge episodes from an eval.py trace.

Default: a filmstrip of one episode, one row per replan. Columns: observation |
its decoding (decoder sanity check) | chosen subgoal decoded | nearest dataset
node to that subgoal | goal.

--videos DIR: add a "decoded subgoal" panel to the agent/dataset/goal mp4s in DIR.
--plans DIR: write plan_env_<i>.png per task to DIR, see save_plan_figures.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from PIL import Image, ImageDraw
from stable_worldmodel.plot import save_panel_videos

from bridge.common import TASKS, add_task_arg, dataset_path
from bridge.decoder import load_decoder
from bridge.planner import sample_paths

PANEL = 224  # panel size in the eval videos
REPLAN_EVERY = 25  # env steps per replan: horizon 5 x action_block 5


def unnormalize(x):
    stats = spt.data.dataset_stats.ImageNet
    mean, std = torch.tensor(stats["mean"])[:, None, None], torch.tensor(stats["std"])[:, None, None]
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def panel_x(j, w=PANEL):
    """Left edge of panel j in save_panel_videos' layout."""
    return max(12, w // 14) + j * (w + max(10, w // 16))


def subgoal_frames(entries, decoder, n_frames):
    """One image per video frame: the decoded subgoal in force at that frame, labelled."""
    stills = []
    with torch.no_grad():
        decoded = decoder(torch.stack([t["subgoal"] for t in entries]).cuda()).clamp(0, 1)
    for k, (t, x) in enumerate(zip(entries, decoded)):
        img = Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        img = img.resize((PANEL, PANEL), Image.BICUBIC)
        label = f"replan {k}: {'goal (direct)' if t['direct'] else 'bridge'}"
        ImageDraw.Draw(img).text((6, 4), label, fill=(60, 60, 60))
        stills.append(np.array(img))
    return np.stack([stills[min(i // REPLAN_EVERY, len(stills) - 1)] for i in range(n_frames)])


def add_subgoal_panel(video_dir, trace, decoder):
    """Rewrite env_<i>.mp4 in video_dir with a fourth panel. Safe to run repeatedly."""
    by_env = defaultdict(list)
    for t in trace:
        by_env[t["env"]].append(t)
    panels = defaultdict(list)
    top = panel_x(0)
    for env in sorted(by_env):
        video = iio.imread(Path(video_dir) / f"env_{env}.mp4")
        for j, label in enumerate(("agent", "dataset", "goal")):
            x = panel_x(j)
            panels[label].append(video[:, top : top + PANEL, x : x + PANEL])
        panels["decoded subgoal"].append(subgoal_frames(by_env[env], decoder, len(video)))
    save_panel_videos(video_dir, dict(panels))


def save_plan_figures(out_dir, trace, decoder, bridge, sigma, steps=96, c=10.0, n_tau=9):
    """One figure per task, from its first replan.

    Row 1: start image, the subgoal pursued at each replan, goal image.
    Row 2: raw iterate x(tau) of one freshly sampled path, decoded.
    Row 3: the denoised estimate x - gamma*eta at the same points, decoded.
    Generation runs on latents, so rows 2-3 show progress and noise removal at once.
    The decoder never saw noisy latents, so row 2 only indicates what the noise does.
    """
    by_env = defaultdict(list)
    for t in trace:
        by_env[t["env"]].append(t)

    def decode(z):
        with torch.no_grad():
            return decoder(z.cuda()).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, steps, n_tau).round().astype(int)
    for env, entries in by_env.items():
        first = entries[0]
        clean, noisy = sample_paths(
            bridge, first["z"][None].cuda(), first["z_goal"][None].cuda(), sigma, steps, c, return_noisy=True
        )
        subs = decode(torch.stack([t["subgoal"] for t in entries]))
        plan = [unnormalize(first["obs"]), *subs, unnormalize(first["goal_img"])]
        names = ["start"] + [f"replan {k}" + (" (goal)" if t["direct"] else "") for k, t in enumerate(entries)] + ["goal"]

        cols = max(len(plan), n_tau)
        fig, axes = plt.subplots(3, cols, figsize=(1.5 * cols, 5.0), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, img, name in zip(axes[0], plan, names):
            ax.imshow(img)
            ax.set_title(name, fontsize=8)
        for row, images, label in (
            (1, decode(noisy[0, idx]), "noisy x(tau)"),
            (2, decode(clean[0, idx]), "denoised"),
        ):
            for ax, img, i in zip(axes[row], images, idx):
                ax.imshow(img)
                if row == 1:
                    ax.set_title(f"tau={i / steps:.2f}", fontsize=8)
            axes[row, 0].text(-0.08, 0.5, label, rotation=90, va="center", ha="right",
                              fontsize=8, transform=axes[row, 0].transAxes)
        fig.tight_layout()
        fig.savefig(out_dir / f"plan_env_{env}.png", dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--out", help="filmstrip png")
    p.add_argument("--videos", help="directory of panel videos to extend")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--plans", help="directory for per-task plan figures")
    add_task_arg(p)
    p.add_argument("--bridge", default=None, help="for --plans; defaults to bridge.pt in the task's artifacts")
    a = p.parse_args()
    task = TASKS[a.task]

    trace = torch.load(a.trace)["trace"]
    dec = load_decoder(task.artifacts / "decoder.pt")
    if a.videos:
        add_subgoal_panel(a.videos, trace, dec)
        print(f"added subgoal panels in {a.videos}")
    if a.plans:
        from bridge.eval import load_bridge

        bridge, ck = load_bridge(a.bridge or task.artifacts / "bridge.pt")
        save_plan_figures(a.plans, trace, dec, bridge, ck["sigma"])
        print(f"wrote plan figures in {a.plans}")
    if not a.out:
        return

    rows = [t for t in trace if t["env"] == a.env]
    g = np.load(task.artifacts / "graph.npz")
    nodes = torch.from_numpy(g["z"]).cuda()

    def decode(z):
        with torch.no_grad():
            return dec(z[None].cuda())[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    titles = ["observation", "decoded observation", "decoded subgoal", "nearest real frame", "goal"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 2.2 * len(rows)), squeeze=False)
    with h5py.File(dataset_path(task.dataset), "r") as f:
        for i, t in enumerate(rows):
            d = torch.cdist(t["subgoal"][None].cuda(), nodes)[0]
            near = int(d.argmin())
            panels = [
                unnormalize(t["obs"]), decode(t["z"]), decode(t["subgoal"]),
                f["pixels"][int(g["frames"][near])], unnormalize(t["goal_img"]),
            ]
            for ax, img in zip(axes[i], panels):
                ax.imshow(img)
                ax.set_xticks([]), ax.set_yticks([])
            axes[i, 0].set_ylabel(f"replan {i}\n{'direct' if t['direct'] else 'bridge'}", fontsize=8)
            to_goal = float((t["z"] - t["z_goal"]).norm())
            hop = float((t["subgoal"] - t["z"]).norm())
            axes[i, 0].set_xlabel(f"to goal {to_goal:.1f}", fontsize=8)
            axes[i, 2].set_xlabel(f"from obs {hop:.1f}, CEM cost {t['cost']:.0f}", fontsize=8)
            axes[i, 3].set_xlabel(f"distance {float(d[near]):.1f}", fontsize=8)
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
