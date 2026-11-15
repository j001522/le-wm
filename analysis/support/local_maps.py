"""Local latent maps: one t-SNE per episode, of the training latents around its trajectory.

python -m analysis.support.local_maps --run lewm_flow_e2e --epoch 14 --groups typical outside --n 6

A whole-dataset t-SNE tears trajectories apart (map_check): single frames vary in
about 5 directions, more than a 2-D map can keep. A small neighbourhood is close to flat, so it
projects with little tearing. For each episode (true latents + imagined latents of
eval_edge, saved in latents_<group>.npz):
  1. neighbourhood = every cached training frame within --radius x r0 of any trajectory point in
     the 192-D latent space (own dataset episode excluded), where r0 = median distance from a
     training frame to its 10th nearest training frame. A fixed radius (not k nearest) keeps the
     coverage honest: a point far from the data gets no neighbours. If more than --max-points
     are found, all are subsampled at the same rate.
  2. t-SNE of the neighbourhood plus the trajectory.
  3. Panel: training coverage (black = none, white = most), true path (green), imagined path.
     Title: final pixel error of the rollout and a fidelity check: Spearman between the pairwise
     distances of the trajectory points (true + imagined) in the latent space and on the map
     (1 = the map keeps which points are closer / farther; this is what the paths are read for).
Episodes per group are taken at evenly spaced quantiles of the final pixel error.
Output: results/edge/<run>_e<epoch>/local_maps.png
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from analysis.support.eval_edge import COLORS, features
from analysis.support.latent_map import grid_colors
from helpers.constants import SWM, run_dir
from helpers.latents import latents_path

TRUE_C = "#39ff14"
IMAG_C = "#ff2bd6"


@torch.no_grad()
def kth_distance(z_all, ep_t, rows, k=10, chunk=32):
    out = []
    for i in range(0, len(rows), chunk):
        r = rows[i : i + chunk]
        d = torch.cdist(z_all[r].float(), z_all.float())
        d[ep_t[None] == ep_t[r][:, None]] = float("inf")
        out.append(d.topk(k, largest=False)[0][:, -1].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def neighbourhood(z_all, ep_t, traj, own, radius, max_points, rng):
    d = torch.cdist(torch.from_numpy(traj).cuda(), z_all.float())  # (P, N)
    d[:, ep_t == own] = float("inf")
    idx = torch.nonzero((d <= radius).any(0)).squeeze(1).cpu().numpy()
    rate = min(1.0, max_points / max(len(idx), 1))
    if rate < 1:
        idx = np.sort(rng.choice(idx, int(len(idx) * rate), replace=False))
    near = (d <= radius).sum(1).cpu().numpy()  # training frames within the radius of each point
    return idx, rate, near


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--groups", nargs="+", default=["typical", "outside"])
    p.add_argument("--n", type=int, default=6, help="episodes per group")
    p.add_argument("--radius", type=float, default=6.0, help="neighbourhood radius in units of r0")
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--perplexity", type=float, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--color-by", choices=["block", "agent", "density"], default="block",
                   help="training frames coloured by their real block / agent x-y (2-D colours of latent_grid.png), "
                        "or grey density")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = run_dir("edge", args.run, args.epoch)

    with h5py.File(latents_path(args.dataset, args.run, args.epoch), "r") as f:
        z_all = torch.from_numpy(f["z"][:]).cuda()
        ep_t = torch.from_numpy(f["episode_idx"][:]).cuda()
    with h5py.File(SWM / "datasets" / f"{args.dataset}.h5", "r") as f:
        state_all = f["state"][:]  # same row order as the latent cache
    r0 = float(np.median(kth_distance(z_all, ep_t, torch.from_numpy(rng.choice(len(z_all), 2000)).cuda())))
    radius = args.radius * r0
    print(f"r0 (median distance to 10th nearest training frame) = {r0:.3f}; radius = {radius:.3f}", flush=True)

    stats_rows = []
    fig, axes = plt.subplots(len(args.groups), args.n, figsize=(3.9 * args.n, 4.3 * len(args.groups)), squeeze=False)
    for gi, g in enumerate(args.groups):
        data = np.load(out / f"latents_{g}.npz")
        HS = int(data["history"])
        final = data["pix_total"][:, -1]
        order = np.argsort(final)
        picks = order[np.linspace(0, len(order) - 1, args.n).round().astype(int)]
        for col, i in enumerate(picks):
            ax = axes[gi, col]
            z_true, z_imag = data["z_true"][i], data["z_imag"][i][HS:]
            traj = np.concatenate([z_true, z_imag]).astype(np.float32)
            idx, rate, near = neighbourhood(z_all, ep_t, traj, int(data["episode"][i]), radius, args.max_points, rng)
            Zn = z_all[torch.from_numpy(idx).cuda()].float().cpu().numpy()
            X = np.concatenate([Zn, traj])
            E = TSNE(2, perplexity=min(args.perplexity, (len(X) - 1) / 3), init="pca",
                     random_state=args.seed).fit_transform(X)
            En, Et = E[: len(Zn)], E[len(Zn) :]
            true, imag = Et[: len(z_true)], np.concatenate([Et[:HS], Et[len(z_true) :]])

            lo, hi = E.min(0), E.max(0)
            pad = 0.05 * (hi - lo)
            box = (lo - pad, hi + pad)
            if len(En) > 1:
                st = state_all[idx]
                if args.color_by == "density":
                    h, xe, ye = np.histogram2d(En[:, 0], En[:, 1], bins=200,
                                               range=[[box[0][0], box[1][0]], [box[0][1], box[1][1]]])
                    sp = np.median(NearestNeighbors(n_neighbors=2).fit(En).kneighbors(En)[0][:, 1])
                    h = gaussian_filter(h, max(1.0, 1.5 * sp / (xe[1] - xe[0]))) / rate
                    ax.imshow(h.T, cmap="gray", origin="lower", extent=(box[0][0], box[1][0], box[0][1], box[1][1]),
                              aspect="auto", norm=matplotlib.colors.PowerNorm(0.5, 0, max(h.max(), 1e-9)))
                else:
                    sl = slice(2, 4) if args.color_by == "block" else slice(0, 2)
                    order = rng.permutation(len(En))
                    xy = st[order, sl]
                    lo_c, hi_c = np.percentile(xy, 5, 0), np.percentile(xy, 95, 0)  # local colour range
                    hi_c = np.maximum(hi_c, lo_c + 1)
                    ax.scatter(En[order, 0], En[order, 1], s=3, c=grid_colors(np.clip(xy, lo_c, hi_c), lo_c, hi_c),
                               linewidths=0, alpha=0.9)
                    crange = (f"{args.color_by} x {lo_c[0]:.0f}-{hi_c[0]:.0f}, y {lo_c[1]:.0f}-{hi_c[1]:.0f}")
                # fidelity 1: do distances among the trajectory points (true + imagined) keep their order?
                rho = spearmanr(pdist(traj), pdist(Et))[0]
                # fidelity 2: do nearby states stay nearby on the map? (neighbourhood frames, real states)
                sub = rng.choice(len(En), min(600, len(En)), replace=False)
                rho_s = spearmanr(pdist(features(st[sub])), pdist(En[sub]))[0] if len(sub) > 3 else np.nan
                rho_b = spearmanr(pdist(st[sub, 2:4]), pdist(En[sub]))[0] if len(sub) > 3 else np.nan
                # without any map: does latent distance track state distance among these real frames?
                rho_ls = spearmanr(pdist(features(st[sub])), pdist(Zn[sub]))[0] if len(sub) > 3 else np.nan
                rho_lb = spearmanr(pdist(st[sub, 2:4]), pdist(Zn[sub]))[0] if len(sub) > 3 else np.nan
                stats_rows.append((g, int(i), rho, rho_s, rho_b, rho_ls, rho_lb))
                cont = (f"map fidelity: trajectory {rho:.2f}, state {rho_s:.2f}, block {rho_b:.2f}\n"
                        f"latent vs state (no map): state {rho_ls:.2f}, block {rho_lb:.2f}")
                if args.color_by != "density":
                    cont += f"\ncolours: {crange}"
            else:
                cont = "no training frames within the radius"
            ax.set_facecolor("black")
            tc, ic = (TRUE_C, COLORS[g]) if args.color_by == "density" else ("white", IMAG_C)
            ax.plot(true[:, 0], true[:, 1], "-o", c=tc, ms=3.5, lw=1.4, mec="black", mew=0.4)
            ax.plot(imag[HS - 1 :, 0], imag[HS - 1 :, 1], "--o", c=ic, ms=4, lw=1.6, mec="black", mew=0.4)
            for t in (0, len(z_true) - 1):
                ax.annotate(str(t), true[t], fontsize=8, color=tc, xytext=(4, 4), textcoords="offset points")
            ax.annotate(str(len(z_true) - 1), imag[-1], fontsize=8, color=ic, xytext=(4, -10),
                        textcoords="offset points", fontweight="bold")
            ax.set_xlim(box[0][0], box[1][0]), ax.set_ylim(box[0][1], box[1][1])
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{g} #{i}: final pixel err {final[i]:.3f}; {len(idx) / rate:.0f} training frames\n{cont}\n"
                         f"support (frames within radius): true end {near[len(z_true) - 1]}, imagined end {near[-1]}",
                         fontsize=7.5)
            print(f"{g} #{i}: {len(idx)} pts (rate {rate:.2f}), {cont}, near true end {near[len(z_true) - 1]}, "
                  f"imagined end {near[-1]}", flush=True)
    bg = ("Black = no training frames, white = most. Green = true path, group colour = imagined path."
          if args.color_by == "density" else
          f"Dots = training frames coloured by their real {args.color_by} x-y (colours of latent_grid.png, stretched "
          "to each panel's 5-95% range). "
          "White = true path, magenta = imagined path.")
    fig.suptitle(f"Local latent maps ({args.run}, epoch {args.epoch}): one t-SNE per episode of the training latents "
                 f"within {args.radius:g} r0 = {radius:.2f} of its trajectory (own episode excluded).\n{bg} Labels = "
                 "first and last frame. Columns: episodes ordered by final pixel error. Map fidelity = Spearman between "
                 "latent (trajectory) or state distances and map distances; latent vs state = the same without the map.",
                 fontsize=10)
    fig.tight_layout()
    print("\nSpearman with state distances, neighbourhood training frames (600 per episode):")
    print("  group     ep  | map: trajectory  state  block | latent (no map): state  block")
    for g, i, *v in stats_rows:
        print(f"  {g:8s} {i:3d}  |       {v[0]:.2f}       {v[1]:.2f}   {v[2]:.2f}  |                  {v[3]:.2f}   {v[4]:.2f}")
    name = "local_maps.png" if args.color_by == "density" else f"local_maps_{args.color_by}.png"
    fig.savefig(out / name, dpi=110)
    print("wrote", out / name)


if __name__ == "__main__":
    main()
