"""Does training support around a rollout predict how close imagination stays to reality?

python -m analysis.support.support --run lewm_flow_e2e --epoch 14

All numbers are computed in the full 192-D latent space (no projection). For every rollout saved
by eval_edge (latents_<group>.npz):
  support(z)   number of cached training frames within R of latent z, own dataset episode
               excluded; R = radius x r0, r0 = median distance from a training frame to its 10th
               nearest training frame (other episodes)
  ctx support  mean log(1 + support) over the HS context frames: known before imagining
  gap          ||ẑ_k - z_k|| between imagined (sample 0) and true latent at imagination step k
  pixel error  object-pixel MSE of the decoded imagination vs the real frame (mean over samples)
Tests, per imagination step: Spearman(ctx support, gap) and Spearman(ctx support, pixel error),
pooled and within each group, for several radii. Outputs in results/edge/<run>_e<epoch>/:
support.png (gap vs step by context-support tercile, with bootstrap 95% bands) and support.json.
The per-point support counts are also saved for the maps (support_points_<group>.npz).
"""

import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from analysis.support.local_maps import kth_distance
from helpers.constants import run_dir
from helpers.latents import latents_path


@torch.no_grad()
def counts(z_all, ep_t, q, own, radii, chunk=64):
    """q (P, D) latents, own (P,) episode to exclude -> (len(radii), P) counts within each radius."""
    out = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i : i + chunk].float(), z_all.float())
        d[ep_t[None] == own[i : i + chunk, None]] = float("inf")
        out.append(torch.stack([(d <= r).sum(1) for r in radii]).cpu())
    return torch.cat(out, 1).numpy()


def boot_band(x, n=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean(0) for _ in range(n)])
    return np.percentile(m, 2.5, 0), np.percentile(m, 97.5, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--groups", nargs="+", default=["typical", "outside"])
    p.add_argument("--radii", type=float, nargs="+", default=[4.0, 6.0, 8.0], help="in units of r0")
    p.add_argument("--main-radius", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = run_dir("edge", args.run, args.epoch)

    with h5py.File(latents_path(args.dataset, args.run, args.epoch), "r") as f:
        z_all = torch.from_numpy(f["z"][:]).cuda()
        ep_t = torch.from_numpy(f["episode_idx"][:]).cuda()
    r0 = float(np.median(kth_distance(z_all, ep_t, torch.from_numpy(rng.choice(len(z_all), 2000)).cuda())))
    radii = [r * r0 for r in args.radii]
    print(f"r0 = {r0:.3f}; radii {np.round(radii, 2).tolist()}", flush=True)

    rows = []
    for g in args.groups:
        d = np.load(out / f"latents_{g}.npz")
        HS = int(d["history"])
        zt, zi = d["z_true"], d["z_imag"]  # (N, T, D); zi[:, :HS] = context
        N, T, D = zt.shape
        own = np.repeat(d["episode"], T)
        sup_t = counts(z_all, ep_t, torch.from_numpy(zt.reshape(-1, D)).cuda(), torch.from_numpy(own).cuda(), radii)
        sup_i = counts(z_all, ep_t, torch.from_numpy(zi.reshape(-1, D)).cuda(), torch.from_numpy(own).cuda(), radii)
        sup_t, sup_i = sup_t.reshape(len(radii), N, T), sup_i.reshape(len(radii), N, T)
        np.savez(out / f"support_points_{g}.npz", true=sup_t, imag=sup_i, radii=np.array(args.radii), r0=r0)
        gap = np.linalg.norm(zi[:, HS:] - zt[:, HS:], axis=-1)  # (N, H)
        pix = d["pix_total"]
        for n in range(N):
            rows.append({"group": g, "ctx": np.log1p(sup_t[:, n, :HS]).mean(1), "gap": gap[n], "pix": pix[n],
                         "sup_true": sup_t[:, n, HS:], "sup_imag": sup_i[:, n, HS:]})
        print(f"{g}: context support at R={args.main_radius:g} r0, median {np.median(sup_t[args.radii.index(args.main_radius), :, :HS]):.0f} "
              f"frames | imagined end median {np.median(sup_i[args.radii.index(args.main_radius), :, -1]):.0f}, "
              f"true end {np.median(sup_t[args.radii.index(args.main_radius), :, -1]):.0f}", flush=True)

    H = rows[0]["gap"].shape[0]
    res = {"r0": r0, "radii_r0": args.radii, "tests": {}}
    print("\nSpearman(context support, gap / pixel error) at imagination steps 1, 5, 10 (negative = more support,"
          " closer to reality)")
    for ri, rr in enumerate(args.radii):
        for scope in ["pooled"] + args.groups:
            sel = [r for r in rows if scope == "pooled" or r["group"] == scope]
            ctx = np.array([r["ctx"][ri] for r in sel])
            gap = np.array([r["gap"] for r in sel])
            pix = np.array([r["pix"] for r in sel])
            rg = [spearmanr(ctx, gap[:, h]) for h in range(H)]
            rp = [spearmanr(ctx, pix[:, h]) for h in range(H)]
            res["tests"][f"R{rr:g}_{scope}"] = {"gap": [[float(x.statistic), float(x.pvalue)] for x in rg],
                                                "pix": [[float(x.statistic), float(x.pvalue)] for x in rp], "n": len(sel)}
            fmt = lambda xs: " ".join(f"{xs[h].statistic:+.2f}{'*' if xs[h].pvalue < 0.05 else ' '}" for h in (0, 4, H - 1))  # noqa: E731
            print(f"  R={rr:g} r0 {scope:8s} n={len(sel):3d} | gap {fmt(rg)} | pixel {fmt(rp)}")
    print("  (* = p < 0.05)")

    # figure: gap and pixel error vs step, by tercile of context support (main radius), pooled
    ri = args.radii.index(args.main_radius)
    ctx = np.array([r["ctx"][ri] for r in rows])
    t1, t2 = np.percentile(ctx, [100 / 3, 200 / 3])
    bins = {"low support": ctx <= t1, "mid support": (ctx > t1) & (ctx <= t2), "high support": ctx > t2}
    colors = {"low support": "C3", "mid support": "C1", "high support": "C0"}
    hs = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, key, name in ((axes[0], "gap", "latent gap ||imagined - true||"),
                          (axes[1], "pix", "pixel error, imagined vs real")):
        for b, m in bins.items():
            v = np.array([r[key] for r, mm in zip(rows, m) if mm])
            lo, hi = boot_band(v, rng=rng)
            groups = {g: sum(1 for r, mm in zip(rows, m) if mm and r["group"] == g) for g in args.groups}
            ax.plot(hs, v.mean(0), "o-", ms=3, c=colors[b],
                    label=f"{b} (n={m.sum()}: " + ", ".join(f"{k} {c}" for k, c in groups.items()) + ")")
            ax.fill_between(hs, lo, hi, color=colors[b], alpha=0.2)
        ax.set_xlabel("imagination step")
        ax.set_title(f"{name}, by context support tercile", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5)
    ax = axes[2]
    for g, c in zip(args.groups, ("C0", "C3")):
        sel = [r for r in rows if r["group"] == g]
        ax.scatter([r["ctx"][ri] for r in sel], [r["gap"][-1] for r in sel], s=14, c=c, label=g)
    rho = spearmanr(ctx, [r["gap"][-1] for r in rows])
    ax.set_xlabel(f"context support: mean log(1 + training frames within {args.main_radius:g} r0)")
    ax.set_ylabel(f"latent gap at step {H}")
    ax.set_title(f"one point per rollout; Spearman {rho.statistic:+.2f} (p={rho.pvalue:.1g})", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Training support vs imagination error ({args.run}, epoch {args.epoch}); support counted in the "
                 f"192-D latent space within {args.main_radius:g} r0 = {args.main_radius * r0:.2f}, own episode "
                 "excluded; bands = bootstrap 95% CI of the mean", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "support.png", dpi=120)
    (out / "support.json").write_text(json.dumps(res))
    print("wrote", out / "support.png", out / "support.json")


if __name__ == "__main__":
    main()
