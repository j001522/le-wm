"""Which 2-D projection of the latent space keeps its structure?

python -m analysis.support.map_check --run lewm_flow_e2e --epoch 14

Fits each projection on cached training latents plus the state grid of latent_map,
then checks:
  grid order      Spearman between grid distances in state space (block x-y) and on the map
  continuity      for real episodes (every FRAMESKIP-th frame), map distance between consecutive
                  frames divided by the median map distance to the nearest training point;
                  median over steps and share of steps that jump > 10x that spacing
Writes results/edge/<run>_e<epoch>/map_check.png (grid coloured as in latent_grid.png, episodes as lines).
"""

import argparse

import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers the swm/ gym envs)
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from analysis.support.eval_edge import valid_start
from analysis.support.latent_map import grid_colors
from helpers.constants import FRAMESKIP, run_dir
from helpers.latents import latents_path
from helpers.models import load_run
from helpers.video import encode, imagenet_norm


def grid_latents(model, norm, n=15, lo=110, hi=390):
    xs = np.linspace(lo, hi, n)
    gxy = np.array([(x, y) for y in xs for x in xs])
    env = gym.make("swm/PushT-v1").unwrapped
    states = np.array([[x - 100, y, x, y, 0.0, 0.0, 0.0] for x, y in gxy])
    ok = np.array([valid_start(env, s) for s in states])
    frames = []
    for s in states[ok]:
        env.reset(seed=0)
        env._set_state(s)
        frames.append(env.render().copy())
    env.close()
    return encode(model, np.stack(frames), norm), gxy[ok], grid_colors(gxy[ok], lo, hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--n-fit", type=int, default=20000)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--frames", type=int, default=13)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=2, help="map dimensions (2 or 3); 3 = metrics only, plot of dims 1-2")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    model, _ = load_run(args.run, args.epoch, decoder=False)
    norm = imagenet_norm()
    z_grid, gxy, cols = grid_latents(model, norm)

    with h5py.File(latents_path(args.dataset, args.run, args.epoch), "r") as f:
        z_all = f["z"][:]
        L, O = f["ep_len"][:], f["ep_offset"][:]
    long = np.flatnonzero(L > args.frames * FRAMESKIP)
    eps = rng.choice(long, args.n_episodes, replace=False)
    ep_rows = [O[e] + FRAMESKIP * np.arange(args.frames) for e in eps]
    excl = np.zeros(len(z_all), bool)
    for e in eps:
        excl[O[e] : O[e] + L[e]] = True
    fit = rng.choice(np.flatnonzero(~excl), args.n_fit, replace=False)
    z_ep = np.concatenate([z_all[r] for r in ep_rows]).astype(np.float32)
    X = np.concatenate([z_all[fit].astype(np.float32), z_grid, z_ep])
    n_fit, n_grid = len(fit), len(z_grid)

    methods = {
        "t-SNE, perplexity 30": lambda: TSNE(args.dims, perplexity=30, init="pca", random_state=0).fit_transform(X),
        "t-SNE, perplexity 300": lambda: TSNE(args.dims, perplexity=300, init="pca", random_state=0).fit_transform(X),
        "PCA of latents": lambda: PCA(args.dims).fit(X[:n_fit]).transform(X),
    }
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 6.4))
    for ax, (name, fn) in zip(axes, methods.items()):
        print(name, "...", flush=True)
        E = fn()
        Ef, Eg, Ee = E[:n_fit], E[n_fit : n_fit + n_grid], E[n_fit + n_grid :]
        rho = spearmanr(pdist(gxy), pdist(Eg))[0]
        spacing = np.median(NearestNeighbors(n_neighbors=2).fit(Ef).kneighbors(Ef)[0][:, 1])
        steps = np.concatenate([np.linalg.norm(np.diff(Ee[i * args.frames : (i + 1) * args.frames], axis=0), axis=1)
                                for i in range(args.n_episodes)]) / spacing
        # same check in the latent space itself, for reference
        lat_spacing = np.median(NearestNeighbors(n_neighbors=2).fit(X[:n_fit]).kneighbors(X[:n_fit])[0][:, 1])
        lat_steps = np.concatenate([np.linalg.norm(np.diff(z_ep[i * args.frames : (i + 1) * args.frames], axis=0),
                                                   axis=1) for i in range(args.n_episodes)]) / lat_spacing
        msg = (f"grid order (Spearman) {rho:.2f} | episode step / spacing: median {np.median(steps):.1f}, "
               f"jumps >10x {np.mean(steps > 10):.0%}")
        print(" ", msg, f"| in latent space: median {np.median(lat_steps):.1f}, >10x {np.mean(lat_steps > 10):.0%}")
        ax.scatter(Ef[:, 0], Ef[:, 1], s=1, c="0.85", linewidths=0)
        for i in range(args.n_episodes):
            seg = Ee[i * args.frames : (i + 1) * args.frames]
            ax.plot(seg[:, 0], seg[:, 1], "-", lw=0.8, alpha=0.8)
        ax.scatter(Eg[:, 0], Eg[:, 1], s=10, c=cols, zorder=3)
        ax.set_title(f"{name}\n{msg}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{args.n_fit} training latents (grey), the state grid (coloured as in latent_grid.png) and "
                 f"{args.n_episodes} real episodes (lines, {args.frames} frames each)", fontsize=10)
    fig.tight_layout()
    out = run_dir("edge", args.run, args.epoch) / ("map_check.png" if args.dims == 2 else f"map_check_{args.dims}d.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
