"""Latent-space maps in the style of the LeWM paper (Fig. 9), with imagined trajectories on top.

python -m analysis.support.latent_map --run lewm_flow_e2e --epoch 14

1. Grid: the block is placed on a grid of x-y positions (fixed angle), the agent at a fixed
   offset from it; each state is rendered and encoded. latent_grid.png = the state grid
   coloured by position (left) and a t-SNE of its latents with the same colours (right).
2. Shared map: one t-SNE of training-frame latents and the example episodes saved by
   eval_edge (true latents and imagined latents, sample 0). With a latent cache
   of this run (scripts.cache_latents --run ... --epoch ...), the t-SNE is fitted on
   --n-fit cached latents and every cached frame is then placed on the map, so the coverage is
   the full dataset; the example episodes' own dataset episodes are left out. Without a cache,
   --n-bg random frames are encoded. latent_map.png: one
   panel per example over the training coverage (black = none, white = most); green = true
   path, coloured = imagined path, numbers = frame counter of the example videos (0-2 context).
No probe: every point on the map is a latent, the true ones from the encoder, the imagined
ones from the predictor.
"""

import argparse

import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers the swm/ gym envs)
import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE

from analysis.support.eval_edge import COLORS, GROUPS, valid_start
from helpers.constants import SWM, run_dir
from helpers.latents import latents_path
from helpers.models import load_run
from helpers.video import encode, imagenet_norm

CORNERS = np.array([[0.55, 0.55, 0.65],   # low x, low y: grey-violet
                    [0.85, 0.25, 0.20],   # high x, low y: red
                    [0.35, 0.55, 0.85],   # low x, high y: blue
                    [0.30, 0.65, 0.35]])  # high x, high y: green


def grid_colors(xy, lo, hi):
    u = (xy - lo) / (hi - lo)
    a, b = u[:, :1], u[:, 1:]
    return ((1 - a) * (1 - b) * CORNERS[0] + a * (1 - b) * CORNERS[1] + (1 - a) * b * CORNERS[2]
            + a * b * CORNERS[3])


@torch.no_grad()
def place(z, z_fit, e_fit, k=10, chunk=4096):
    """Map positions for latents z (N, D): inverse-distance mean of the t-SNE positions of their k
    nearest fitted latents. Standard out-of-sample placement for an existing t-SNE."""
    out = []
    for i in range(0, len(z), chunk):
        d = torch.cdist(z[i : i + chunk].float(), z_fit)
        dk, ik = d.topk(k, largest=False)
        w = 1.0 / (dk + 1e-6)
        out.append(((w[..., None] * e_fit[ik]).sum(1) / w.sum(1, keepdim=True)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def isolation(z_all, ep_all, keep, eps, HS, rng, k=10, n_ref=2000, chunk=32):
    """Latent isolation: mean distance to the k nearest cached training frames, leaving out the
    query's own episode, as a percentile of the same score for n_ref random training frames."""
    ep_t = torch.from_numpy(ep_all).cuda()

    def score(q, q_ep):
        out = []
        for i in range(0, len(q), chunk):
            d = torch.cdist(q[i : i + chunk].float(), z_all.float() if z_all.dtype != torch.float32 else z_all)
            d[ep_t[None] == q_ep[i : i + chunk, None]] = float("inf")
            out.append(d.topk(k, largest=False)[0].mean(1).cpu())
        return torch.cat(out).numpy()

    ref = rng.choice(keep, n_ref, replace=False)
    ref_score = np.sort(score(z_all[torch.from_numpy(ref).cuda()], ep_t[torch.from_numpy(ref).cuda()]))
    res = {}
    for g, e in eps.items():
        own = int(e["episode"]) if "episode" in e else -1
        for kind, zz in (("true", e["z_true"]), ("imag", e["z_imag"][HS:])):
            q = torch.from_numpy(zz).cuda()
            sc = score(q, torch.full((len(q),), own, device="cuda"))
            res[(g, kind)] = 100 * np.searchsorted(ref_score, sc) / len(ref_score)
    return res



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--grid", type=int, default=15, help="grid points per axis")
    p.add_argument("--lo", type=float, default=110)
    p.add_argument("--hi", type=float, default=390)
    p.add_argument("--angle", type=float, default=0.0, help="block angle on the grid")
    p.add_argument("--n-bg", type=int, default=8000, help="training frames in the background (no cache)")
    p.add_argument("--n-fit", type=int, default=100000, help="cached latents the t-SNE is fitted on")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = run_dir("edge", args.run, args.epoch)

    model, _ = load_run(args.run, args.epoch, decoder=False)
    norm = imagenet_norm()

    # 1. grid of states: block on the grid, agent at the first offset that never overlaps it
    xs = np.linspace(args.lo, args.hi, args.grid)
    gxy = np.array([(x, y) for y in xs for x in xs])
    env = gym.make("swm/PushT-v1").unwrapped
    best = None
    for off in ((-100, 0), (100, 0), (0, -100), (-80, -80)):
        states = np.array([[x + off[0], y + off[1], x, y, args.angle, 0.0, 0.0] for x, y in gxy])
        ok = np.array([valid_start(env, s) for s in states])
        if best is None or ok.sum() > best[2].sum():
            best = (off, states, ok)
    off, states, ok = best
    if not ok.all():  # e.g. the block touching a wall: physics would move it
        print(f"dropping {(~ok).sum()} grid states that are not physically valid", flush=True)
    gxy, states = gxy[ok], states[ok]
    frames = []
    for s in states:
        env.reset(seed=0)
        env._set_state(s)
        frames.append(env.render().copy())
    env.close()
    z_grid = encode(model, np.stack(frames), norm)
    cols = grid_colors(gxy, args.lo, args.hi)
    print(f"grid: {len(gxy)} states, agent offset {off}", flush=True)

    e = TSNE(2, perplexity=args.perplexity, init="pca", random_state=args.seed).fit_transform(z_grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].scatter(gxy[:, 0], gxy[:, 1], c=cols, s=14)
    axes[0].set_title("Push-T state grid (block x-y; agent offset fixed)", fontsize=10)
    axes[0].set_xlabel("block x"), axes[0].set_ylabel("block y")
    axes[1].scatter(e[:, 0], e[:, 1], c=cols, s=14)
    axes[1].set_title(f"t-SNE of the latents ({args.run}, epoch {args.epoch})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "latent_grid.png", dpi=130)
    plt.close(fig)

    # 2. shared map with the example episodes
    eps = {g: np.load(out / f"example_{g}_latents.npz") for g in GROUPS}
    HS = int(eps[GROUPS[0]]["history"])
    cache = latents_path(args.dataset, args.run, args.epoch)
    if cache.exists():
        # full dataset: t-SNE on a sample, then every cached frame placed by its nearest fitted latents
        with h5py.File(cache, "r") as f:
            z_all = f["z"][:]  # (N, D) float16
            ep_all = f["episode_idx"][:]
        z_gpu = torch.from_numpy(z_all).cuda()
        drop = np.zeros(len(z_all), bool)
        for g in GROUPS:  # replayed dataset episodes are in the cache: leave them out of the coverage
            e = int(eps[g]["episode"]) if "episode" in eps[g] else -1
            if e >= 0:
                drop |= ep_all == e
                print(f"{g} example = dataset episode {e}: excluded from the coverage", flush=True)
        keep = np.flatnonzero(~drop)
        fit = np.sort(rng.choice(keep, args.n_fit, replace=False))
        z_bg = z_all[fit].astype(np.float32)
    else:
        print(f"no latent cache at {cache}; encoding {args.n_bg} random frames instead", flush=True)
        with h5py.File(SWM / "datasets" / f"{args.dataset}.h5", "r") as f:
            idx = np.sort(rng.choice(len(f["pixels"]), args.n_bg, replace=False))
            bg_frames = np.stack([f["pixels"][i] for i in idx])
        z_bg = encode(model, bg_frames, norm)
    parts = [z_bg] + [x for g in GROUPS for x in (eps[g]["z_true"], eps[g]["z_imag"][HS:])]
    Z = np.concatenate(parts)
    print("t-SNE on", len(Z), "latents ...", flush=True)
    E = TSNE(2, perplexity=args.perplexity, init="pca", random_state=args.seed).fit_transform(Z)
    cuts = np.cumsum([len(x) for x in parts])[:-1]
    E_parts = np.split(E, cuts)
    E_bg = E_parts[0]
    n_cov = len(E_bg)
    if cache.exists():
        print(f"placing {len(keep)} cached frames on the map ...", flush=True)
        # anchors = every t-SNE point (training sample and episodes), so frames near an episode land near it
        E_bg = place(z_gpu[torch.from_numpy(keep).cuda()], torch.from_numpy(Z).cuda(), torch.from_numpy(E).cuda())
        n_cov = len(keep)
        iso = isolation(z_gpu, ep_all, keep, eps, HS, rng)
        del z_gpu

    # training coverage: smoothed density of training-frame latents on the map (per unit area)
    full_lo, full_hi = E.min(0), E.max(0)
    full_pad = 0.03 * (full_hi - full_lo)
    full_box = (full_lo - full_pad, full_hi + full_pad)
    sigma = (full_box[1] - full_box[0]).max() / 250  # kernel width in map units

    def density(box, bins=400, smooth=True):
        h, xe, ye = np.histogram2d(E_bg[:, 0], E_bg[:, 1], bins=bins, range=[[box[0][0], box[1][0]],
                                                                          [box[0][1], box[1][1]]])
        bw = np.array([xe[1] - xe[0], ye[1] - ye[0]])
        if smooth:
            h = gaussian_filter(h, sigma / bw)
        return h.T / (bw[0] * bw[1])  # rows = y, frames per unit area

    d_full = density(full_box)
    vmax = np.percentile(d_full[d_full > 0], 99)
    true_c = "#39ff14"

    rows = 3 if cache.exists() else 2
    fig, axes = plt.subplots(rows, len(GROUPS), figsize=(6 * len(GROUPS), 12.5 + 3.5 * (rows - 2)),
                             gridspec_kw={"height_ratios": [1, 1, 0.45][:rows]})
    for k, g in enumerate(GROUPS):
        true, imag = E_parts[1 + 2 * k], E_parts[2 + 2 * k]
        imag = np.concatenate([true[:HS], imag])  # imagination starts from the context frames
        pts = np.concatenate([true, imag])
        lo, hi = pts.min(0), pts.max(0)
        c, half = (lo + hi) / 2, max(0.65 * (hi - lo).max(), 2.0)  # square zoom box
        box = (c - half, c + half)
        for row in (0, 1):
            ax = axes[row, k]
            view = full_box if row == 0 else box
            if row == 0:  # smoothed coverage, black = no training frames, white = densest 1%
                ax.imshow(d_full, cmap="gray", norm=matplotlib.colors.PowerNorm(0.5, 0, vmax), origin="lower",
                          extent=(view[0][0], view[1][0], view[0][1], view[1][1]), aspect="auto",
                          interpolation="bilinear")
            elif cache.exists():  # zoom: unsmoothed count of training frames per pixel
                d = density(box, bins=160, smooth=False)
                ax.imshow(np.ma.masked_equal(d, 0), cmap="gray", norm=matplotlib.colors.LogNorm(), origin="lower",
                          extent=(view[0][0], view[1][0], view[0][1], view[1][1]), interpolation="nearest")
                ax.set_facecolor("black")
                ax.set_aspect("equal")
            else:  # zoom without cache: the individual sampled training frames
                ax.set_facecolor("black")
                inb = np.all((E_bg >= box[0]) & (E_bg <= box[1]), 1)
                ax.scatter(E_bg[inb, 0], E_bg[inb, 1], s=22, c="white", linewidths=0, label="training frames")
                ax.set_aspect("equal")
            ax.plot(true[:, 0], true[:, 1], "-o", c=true_c, ms=4, lw=1.4, label="true (encoded frames)")
            ax.plot(imag[HS - 1 :, 0], imag[HS - 1 :, 1], "--o", c=COLORS[g], ms=4.5, lw=1.6, mec="white", mew=0.5,
                    label="imagined (open loop)")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim(view[0][0], view[1][0]), ax.set_ylim(view[0][1], view[1][1])
            if row == 0:
                ax.add_patch(plt.Rectangle(box[0], *(box[1] - box[0]), fill=False, ec="yellow", lw=1.2))
                ax.set_title(f"{g} example: whole map (yellow box = zoom below)", fontsize=10)
                continue
            for t, (x, y) in enumerate(true):
                ax.annotate(str(t), (x, y), fontsize=8, color=true_c, xytext=(4, 4), textcoords="offset points")
            for t, (x, y) in enumerate(imag[HS:], start=HS):
                ax.annotate(str(t), (x, y), fontsize=8, color=COLORS[g], xytext=(4, -10), textcoords="offset points",
                            fontweight="bold")
            ax.set_title(f"{g}, zoomed (training frames per pixel)", fontsize=10)
    if cache.exists():
        for k, g in enumerate(GROUPS):
            ax = axes[2, k]
            ax.plot(np.arange(len(iso[(g, "true")])), iso[(g, "true")], "-o", c="#1a9e1a", ms=4, label="true")
            ax.plot(np.arange(HS, HS + len(iso[(g, "imag")])), iso[(g, "imag")], "--o", c=COLORS[g], ms=4,
                    label="imagined")
            ax.axhline(99, c="0.5", ls=":", lw=1)
            ax.axvline(HS - 0.5, c="0.7", lw=1)
            ax.set_ylim(0, 101)
            ax.set_xticks(np.arange(len(iso[(g, "true")])))
            ax.set_xlabel("frame (0-%d context)" % (HS - 1))
            ax.set_title(f"{g}: latent isolation (percentile among training frames)", fontsize=10)
            ax.grid(alpha=0.3)
        axes[2, 0].set_ylabel("isolation percentile\n(100 = more isolated than\nevery training frame)")
        axes[2, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].legend(fontsize=8, loc="lower left", markerscale=1.3)
    how = (f"t-SNE fitted on {len(parts[0])} training latents + the episodes; all {n_cov} cached training frames "
           "placed by their nearest t-SNE points" if cache.exists() else f"t-SNE of {n_cov} training latents")
    fig.suptitle(f"Latent map ({args.run}, epoch {args.epoch}): {how}, plus the three example episodes (their own "
                 "dataset episodes left out).\nMaps: black = no training frames, white = most; green = true path, "
                 f"group colour = imagined path; numbers = video frames (0-{HS - 1} context). Bottom: isolation in the "
                 "full 192-D latent space (mean distance to 10 nearest training frames, own episode excluded).",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "latent_map.png", dpi=120)
    plt.close(fig)
    print("wrote", out / "latent_grid.png", out / "latent_map.png")


if __name__ == "__main__":
    main()
