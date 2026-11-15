"""Natural hallucinations: open-loop imagination from typical, edge and outside start states.

python -m analysis.support.eval_edge --run lewm_flow_e2e --epoch 14

1. Rarity of a Push-T state = mean distance to its K nearest neighbours in the training data,
   in the features (agent xy, block xy) / 512 and ANGLE_W * (cos, sin)(block angle).
   Reference = even episodes; candidates from odd episodes, so a state is never scored
   against its own episode.
2. Three groups of start states (--n each):
   typical  dataset states with rarity below the median
   edge     dataset states with rarity in the top 1% (rare but seen)
   outside  random board configurations (uniform positions and angle, physically valid)
            with rarity between the dataset's p99.9 and OUT_MAX x p99.9: just beyond the data
   Dataset starts use their own recorded actions; outside starts borrow the actions recorded
   after their nearest dataset neighbour.
3. From each start the simulator gives HS context frames and the true future; the model
   imagines --horizon frames open loop (--samples flow draws, shared noise across starts).
   Per imagined frame:
     err   ||ẑ_k - z_k||^2 / D               hallucination (truth known only here)
     u_f   flow endpoint drift               proxy available while imagining
     u_r   ||ẑ_k - E(D(ẑ_k))|| / sqrt(D)     round trip of the imagined latent, also available
4. Outputs in results/edge/<run>_e<epoch>/: map.png (where the groups sit in the data),
   signals.png (err, u_f, u_r vs imagination step per group), scatter.png (proxies vs err,
   Spearman), one video + signal plot per group, results.json, map_regions.png (each board bin
   coloured by the rarity class of putting the block / agent there), map_full.png (the full
   state features projected to 2-D with PCA and t-SNE). --maps-only redraws the maps.
"""

import argparse
import json

import cv2
import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers the swm/ gym envs)
import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from helpers.constants import FRAMESKIP, SWM, run_dir
from helpers.models import load_run
from helpers.video import decode, imagenet_norm, put, write_video

HS = 3
K = 10
ANGLE_W = 0.25
OUT_MAX = 1.5
OBJ_THR = 0.15  # object pixel: differs from the white background by more than this
GROUPS = ("typical", "edge", "outside")
COLORS = {"typical": "C0", "edge": "C1", "outside": "C3"}


def features(s):
    return np.concatenate([s[:, :4] / 512.0, ANGLE_W * np.stack([np.cos(s[:, 4]), np.sin(s[:, 4])], 1)], 1)


def valid_start(env, state):
    """Set the state and check the physics does not push agent or block apart (no overlap)."""
    env.reset(seed=0)
    env._set_state(state)
    env.step(np.zeros(2))
    obs = env._get_obs()
    return np.abs(obs[2:4] - state[2:4]).max() < 2 and abs(obs[4] - state[4]) < 0.02


def simulate(env, state, actions, n_frames):
    env.reset(seed=0)
    env._set_state(state)
    frames = [env.render().copy()]
    for k in range(1, n_frames):
        for t in range(FRAMESKIP * (k - 1), FRAMESKIP * k):
            env.step(actions[t])
        frames.append(env.render().copy())
    return np.stack(frames)


@torch.no_grad()
def imagine(model, decoder, frames, act, args, norm):
    """frames (N, T, 224, 224, 3), act (N, T, 10). Returns per-step signals (N, H) and latents of sample 0."""
    mean, std = norm
    N, T = frames.shape[:2]
    H = T - HS
    z = []
    for i in range(0, N, 16):
        px = torch.from_numpy(frames[i : i + 16]).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
        z.append(model.encode({"pixels": (px - mean) / std})["emb"])
    z = torch.cat(z)  # (N, T, D)
    D = z.size(-1)
    a = torch.from_numpy(act).cuda()
    M = args.samples
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eps = torch.randn(1, M, H, D, device="cuda", generator=gen).expand(N, -1, -1, -1).reshape(N * M, H, D)
    rep = lambda t: t.repeat_interleave(M, 0)  # noqa: E731
    roll, drift = model.rollout_latents(rep(z[:, :HS]), rep(a[:, :HS]), rep(a[:, HS : T - 1]), HS, eps)
    z_hat = roll[:, HS:].reshape(N, M, H, D)
    u_f = drift.reshape(N, M, H)
    err = (z_hat - z[:, None, HS:]).pow(2).mean(-1)
    u_r = []
    for i in range(0, N * M * H, 128):
        zz = z_hat.reshape(-1, D)[i : i + 128]
        u_r.append((zz - model.encode({"pixels": decoder(zz)[:, None]})["emb"][:, 0]).norm(dim=-1) / D**0.5)
    u_r = torch.cat(u_r).reshape(N, M, H)
    sig = {k: v.mean(1).cpu().numpy() for k, v in (("err", err), ("u_f", u_f), ("u_r", u_r))}
    sig["move"] = ((z[:, HS:] - z[:, HS - 1 : -1]).norm(dim=-1) / D**0.5).cpu().numpy()  # true latent movement
    # latent error relative to doing nothing (copy the last context frame)
    err_copy = (z[:, HS - 1 : HS] - z[:, HS:]).pow(2).mean(-1)
    sig["err_rel"] = (err.mean(1) / err_copy).cpu().numpy()

    # pixel errors on object pixels (|pixel - white background| > OBJ_THR in either image), RGB in [0, 1]
    def to01(x):
        return (x * std + mean).clamp(0, 1)

    def obj_mse(a_, b_):
        mask = (((1 - a_).amax(1) > OBJ_THR) | ((1 - b_).amax(1) > OBJ_THR)).float()  # (B, H, W)
        return ((a_ - b_).pow(2).mean(1) * mask).sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)

    real = torch.from_numpy(frames[:, HS:]).cuda().permute(0, 1, 4, 2, 3).float() / 255.0  # (N, H, 3, 224, 224)
    pix = {"pix_total": [], "pix_dyn": []}
    pix_dec = []
    for n in range(N):
        dz = to01(decoder(z[n, HS:]))  # decoded true latents (H, 3, 224, 224)
        pix_dec.append(obj_mse(dz, real[n]))
        tot, dyn = [], []
        for m in range(M):
            di = to01(decoder(z_hat[n, m]))
            tot.append(obj_mse(di, real[n]))
            dyn.append(obj_mse(di, dz))
        pix["pix_total"].append(torch.stack(tot).mean(0))
        pix["pix_dyn"].append(torch.stack(dyn).mean(0))
    sig["pix_dec"] = torch.stack(pix_dec).cpu().numpy()
    for k, v in pix.items():
        sig[k] = torch.stack(v).cpu().numpy()
    return sig, torch.cat([z[:, :HS], z_hat[:, 0]], 1), z  # context + imagined sample 0, and true latents (N, T, D)


def build_reference(args, rng):
    """Load the data, fit the nearest-neighbour rarity score, and score held-out candidates."""
    with h5py.File(SWM / "datasets" / f"{args.dataset}.h5", "r") as f:
        s = f["state"][:]
        ep = f["episode_idx"][:]
        act = f["action"][:]
        L, O = f["ep_len"][:], f["ep_offset"][:]
    need = (HS + args.horizon - 1) * FRAMESKIP
    step = np.arange(len(s)) - O[ep]
    room = (L[ep] - step) >= need  # enough recorded actions after this state
    ok = room & ~np.isnan(act).any(1)
    ref = np.flatnonzero((ep % 2 == 0) & ok)[:: args.ref_stride]
    nn = NearestNeighbors(n_neighbors=K).fit(features(s[ref]))
    cand = rng.choice(np.flatnonzero((ep % 2 == 1) & ok), args.n_cand, replace=False)
    score = nn.kneighbors(features(s[cand]))[0].mean(1)
    p50, p99, p999 = np.percentile(score, [50, 99, 99.9])
    return s, act, need, ref, nn, cand, score, (p50, p99, p999), ep


def pick_starts(args, rng):
    s, act, need, ref, nn, cand, score, (p50, p99, p999), ep = build_reference(args, rng)

    env = gym.make("swm/PushT-v1").unwrapped
    starts = {}
    typ = cand[score < p50]
    edge = cand[score >= p99]
    for g, pool in (("typical", typ), ("edge", edge)):
        chosen = []
        for i in rng.permutation(pool):
            st = s[i].copy()
            if valid_start(env, st):
                chosen.append((st, act[i : i + need], float(score[np.flatnonzero(cand == i)[0]]), int(ep[i])))
            if len(chosen) == args.n:
                break
        starts[g] = chosen

    # outside: uniform board configurations just beyond the data
    chosen, tries = [], 0
    while len(chosen) < args.n and tries < 200:
        tries += 1
        u = np.column_stack([rng.uniform(20, 492, (4096, 4)), rng.uniform(0, 2 * np.pi, 4096),
                             np.zeros((4096, 2))])
        d, idx = nn.kneighbors(features(u))
        sc = d.mean(1)
        for j in np.flatnonzero((sc > p999) & (sc < OUT_MAX * p999)):
            if valid_start(env, u[j]):
                r = ref[idx[j, 0]]  # nearest dataset neighbour: borrow its recorded actions
                chosen.append((u[j], act[r : r + need], float(sc[j]), -1))
            if len(chosen) == args.n:
                break
    starts["outside"] = chosen
    env.close()
    a_ok = ~np.isnan(act).any(1)
    stats = {"p50": p50, "p99": p99, "p999": p999, "a_mean": act[a_ok].mean(0), "a_std": act[a_ok].std(0), "nn": nn,
             "cand_states": s[cand], "cand_score": score}
    ref_states = s[ref]
    return starts, stats, ref_states


CLASS_NAMES = ("typical", "between", "edge", "outside", "far outside")
CLASS_COLORS = ("#1f77b4", "#d9d9d9", "#ff7f0e", "#d62728", "#000000")
PANELS = (("block position", slice(2, 4)), ("agent position", slice(0, 2)))


def rarity_class(score, stats):
    """0 typical (< p50), 1 between (p50-p99), 2 edge (p99-p99.9), 3 outside (p99.9-OUT_MAX*p99.9), 4 far outside."""
    edges = [stats["p50"], stats["p99"], stats["p999"], OUT_MAX * stats["p999"]]
    return np.digitize(score, edges)


def make_maps(ref_states, stats, start_states, out, rng, cand_states, cand_score, bins=48, per_bin=128, min_data=5):
    """map.png: data density (black = never visited, white = most common) with the start states.
    map_regions.png: each bin coloured by a rarity class. Bins with >= min_data held-out training
    states: median rarity of those states. Other bins: median rarity of putting the block (or agent)
    there with the rest of the state from per_bin random training states."""
    starts_kw = dict(s=14, edgecolors="black", linewidths=0.4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, sl) in zip(axes, PANELS):
        ax.set_facecolor("black")
        ax.hist2d(ref_states[:, sl][:, 0], ref_states[:, sl][:, 1], bins=64, range=[[0, 512], [0, 512]],
                  cmap="gray", norm=matplotlib.colors.LogNorm(), cmin=1)
        for g in GROUPS:
            ax.scatter(start_states[g][:, sl][:, 0], start_states[g][:, sl][:, 1], c=COLORS[g], label=g, **starts_kw)
        ax.set_title(f"{name}: training data (black = never visited, white = most visited, log)", fontsize=9)
        ax.set_xlim(0, 512), ax.set_ylim(512, 0), ax.set_aspect("equal")
    axes[0].legend(fontsize=8, facecolor="white")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=120)
    plt.close(fig)

    centres = (np.arange(bins) + 0.5) * 512 / bins
    gx, gy = np.meshgrid(centres, centres)  # gy rows = y
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    for ax, (name, sl) in zip(axes, PANELS):
        base = ref_states[rng.choice(len(ref_states), per_bin, replace=False)]
        q = np.repeat(base[None], bins * bins, 0)  # (bins*bins, per_bin, 7)
        q[:, :, sl] = np.stack([gx.ravel(), gy.ravel()], 1)[:, None, :]
        score = stats["nn"].kneighbors(features(q.reshape(-1, q.shape[-1])))[0].mean(1)
        med = np.median(score.reshape(bins, bins, per_bin), -1)
        bx = np.clip((cand_states[:, sl][:, 0] * bins / 512).astype(int), 0, bins - 1)
        by = np.clip((cand_states[:, sl][:, 1] * bins / 512).astype(int), 0, bins - 1)
        for yy in range(bins):
            for xx in range(bins):
                hit = (bx == xx) & (by == yy)
                if hit.sum() >= min_data:
                    med[yy, xx] = np.median(cand_score[hit])
        cls = rarity_class(med, stats)
        ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=len(CLASS_NAMES) - 0.5, extent=(0, 512, 512, 0),
                  interpolation="nearest")
        for g in GROUPS:
            ax.scatter(start_states[g][:, sl][:, 0], start_states[g][:, sl][:, 1], c=COLORS[g], **starts_kw)
        ax.set_title(f"{name}: rarity class (data where visited, else 'if it were here')", fontsize=8.5)
        ax.set_xlim(0, 512), ax.set_ylim(512, 0), ax.set_aspect("equal")
    handles = [matplotlib.patches.Patch(color=c, label=f"{n}") for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               title=f"Bin colour: median rarity of the held-out training states in it (if >= {min_data}); otherwise of "
                     f"{per_bin} training states with the block/agent moved there.\n"
                     f"typical < p50 < between < p99 < edge < p99.9 < outside < {OUT_MAX}x p99.9 < far outside."
                     f"   Dots: start states, coloured by their group.",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out / "map_regions.png", dpi=120)
    plt.close(fig)


FEATURE_NAMES = ("agent x", "agent y", "block x", "block y", "cos angle", "sin angle")


def make_full_map(stats, start_states, start_rarity, out, rng, cand_states, cand_score, n_data=12000, n_rand=6000):
    """map_full.png: the full 6-D rarity features projected to 2-D with PCA and t-SNE.
    Points: held-out training states (scored against the reference set) and random board
    configurations (to show never-visited regions), plus the start states."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    di = rng.choice(len(cand_states), n_data, replace=False)
    rand = np.column_stack([rng.uniform(20, 492, (n_rand, 4)), rng.uniform(0, 2 * np.pi, n_rand)])
    rand_score = stats["nn"].kneighbors(features(rand))[0].mean(1)
    starts = np.concatenate([start_states[g][:, :5] for g in GROUPS])
    X = np.concatenate([features(cand_states[di]), features(rand), features(starts)])
    score = np.concatenate([cand_score[di], rand_score, np.concatenate([start_rarity[g] for g in GROUPS])])
    is_data = np.r_[np.ones(n_data, bool), np.zeros(n_rand + len(starts), bool)]
    n0 = n_data + n_rand
    group_of = np.concatenate([[g] * len(start_states[g]) for g in GROUPS])

    pca = PCA(2).fit(X[:n0])
    emb = {"PCA": pca.transform(X)}
    print("t-SNE on", len(X), "points ...", flush=True)
    emb["t-SNE"] = TSNE(2, perplexity=50, init="pca", random_state=0).fit_transform(X)
    cls = rarity_class(score, stats)
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5))
    for col, (name, E) in enumerate(emb.items()):
        lo, hi = E[:n0].min(0), E[:n0].max(0)
        pad = 0.03 * (hi - lo)
        rng_xy = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
        ax = axes[0, col]
        ax.set_facecolor("black")
        ax.hist2d(E[is_data, 0], E[is_data, 1], bins=70, range=rng_xy, cmap="gray",
                  norm=matplotlib.colors.LogNorm(), cmin=1)
        ax.set_title(f"{name}: held-out training states (black = none, white = most, log)", fontsize=9)
        ax2 = axes[1, col]
        order = np.argsort(-cls[:n0])  # draw rare first, typical on top
        ax2.scatter(E[:n0][order, 0], E[:n0][order, 1], c=cls[:n0][order], cmap=cmap, vmin=-0.5,
                    vmax=len(CLASS_NAMES) - 0.5, s=3, linewidths=0)
        ax2.set_title(f"{name}: training states + random layouts, coloured by rarity class", fontsize=9)
        for a in (ax, ax2):
            for g in GROUPS:
                m = group_of == g
                a.scatter(E[n0:][m, 0], E[n0:][m, 1], c=COLORS[g], s=16, edgecolors="black", linewidths=0.5,
                          label=f"{g} starts")
            a.set_xlim(*rng_xy[0]), a.set_ylim(*rng_xy[1])
            a.set_xticks([]), a.set_yticks([])
        if name == "PCA":
            for i, a in ((0, ax2.set_xlabel), (1, ax2.set_ylabel)):
                load = pca.components_[i]
                top = np.argsort(-np.abs(load))[:3]
                a(f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%} var): "
                  + ", ".join(f"{load[j]:+.2f} {FEATURE_NAMES[j]}" for j in top), fontsize=8)
    axes[0, 0].legend(fontsize=8, facecolor="white", loc="upper right")
    handles = [matplotlib.patches.Patch(color=c, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               title="Features: agent xy and block xy / 512, block angle as 0.25 (cos, sin) - the space rarity is "
                     "computed in.\ntypical < p50 < between < p99 < edge < p99.9 < outside < "
                     f"{OUT_MAX}x p99.9 < far outside. Dots: start states, coloured by group.", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "map_full.png", dpi=120)
    plt.close(fig)


def video(frames, imag, sig, title, args, path):
    T, up = len(frames), 2
    out = []
    for k in range(T):
        left = cv2.resize(frames[k], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(imag[k], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        put(left, "simulator", (8, 22), 0.6, (40, 40, 40))
        put(right, "imagination (decoded, open loop)" if k >= HS else "context (reconstruction)", (8, 22), 0.6,
            (40, 40, 40))
        bar = np.full((64, left.shape[1] * 2 + 8, 3), 255, np.uint8)
        put(bar, f"frame {k:2d} / {T - 1}", (10, 26), 0.8, (0, 0, 0), 2)
        if k >= HS:
            h = k - HS
            put(bar, f"u_f {sig['u_f'][h]:.3f}   u_r {sig['u_r'][h]:.3f}   pixel err {sig['pix_total'][h]:.3f}", (10, 54), 0.6,
                (60, 60, 60))
        put(bar, title, (330, 26), 0.6, (60, 60, 60))
        gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
        out.append(np.concatenate([bar, np.concatenate([left, gap, right], 1)], 0))
    write_video(out, path, args.fps, gif=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    ks = np.arange(HS, T)
    for ax, key, name in zip(axes, ("pix_total", "u_f", "u_r"),
                             ("pixel error, imagined vs real (object pixels)", "$u_f$: flow drift",
                              "$u_r$: round trip of imagined latent")):
        ax.plot(ks, sig[key], "o-", ms=3)
        ax.set_title(name, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame (same counter as the video)")
    axes[-1].set_xticks(np.arange(T))
    axes[-1].set_xlim(-0.5, T - 0.5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path.parent / f"{path.name}_signals.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--n", type=int, default=64, help="start states per group")
    p.add_argument("--n-cand", type=int, default=40000)
    p.add_argument("--ref-stride", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10, help="imagined frames")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--maps-only", action="store_true", help="redraw the maps from results.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = run_dir("edge", args.run, args.epoch)

    if args.maps_only:
        s, _, _, ref, nn, cand, score, (p50, p99, p999), _ = build_reference(args, rng)
        groups = json.loads((out / "results.json").read_text())["groups"]
        st = {"p50": p50, "p99": p99, "p999": p999, "nn": nn}
        starts = {g: np.array(groups[g]["start_states"]) for g in GROUPS}
        make_maps(s[ref], st, starts, out, rng, s[cand], score)
        make_full_map(st, starts, {g: np.array(groups[g]["rarity"]) for g in GROUPS}, out, rng, s[cand], score)
        print("wrote", out / "map.png", out / "map_regions.png", out / "map_full.png")
        return

    starts, stats, ref_states = pick_starts(args, rng)
    print({g: len(v) for g, v in starts.items()}, "rarity p50/p99/p99.9:",
          [round(float(stats[k]), 4) for k in ("p50", "p99", "p999")])

    model, decoder = load_run(args.run, args.epoch)
    norm = imagenet_norm()
    T = HS + args.horizon
    env = gym.make("swm/PushT-v1").unwrapped
    out = run_dir("edge", args.run, args.epoch)
    out.mkdir(parents=True, exist_ok=True)
    res, all_sig = {}, {}
    for g in GROUPS:
        frames = np.stack([simulate(env, s0, a, T) for s0, a, *_ in starts[g]])
        act = np.stack([np.nan_to_num((a - stats["a_mean"]) / stats["a_std"]) for _, a, *_ in starts[g]])
        act = np.concatenate([act, np.zeros((len(act), FRAMESKIP, 2))], 1).reshape(len(act), T, -1).astype(np.float32)
        sig, zs, z_true = imagine(model, decoder, frames, act, args, norm)
        np.savez(out / f"latents_{g}.npz", z_true=z_true.cpu().numpy(), z_imag=zs.cpu().numpy(), history=HS,
                 episode=np.array([st[3] for st in starts[g]]), pix_total=sig["pix_total"])  # all rollouts
        all_sig[g] = sig
        rar = np.array([r for _, _, r, *_ in starts[g]])
        res[g] = {"rarity": rar.tolist(), "start_states": [np.round(s0, 2).tolist() for s0, *_ in starts[g]],
                  **{k: np.round(v, 5).tolist() for k, v in sig.items()}}
        print(f"{g:8s} rarity {rar.mean():.3f} | step 1 / {args.horizon}: pix_total "
              + " | ".join(f"{k} {np.median(sig[k][:, 0]):.4f}/{np.median(sig[k][:, -1]):.4f}"
                           for k in ("pix_total", "pix_dyn", "pix_dec", "err", "err_rel"))
              + f" | u_f {sig['u_f'].mean():.4f} | u_r {sig['u_r'].mean():.4f}")
        # video: the start with median final error in its group
        i = int(np.argsort(sig["pix_total"][:, -1])[len(rar) // 2])
        np.savez(out / f"example_{g}_latents.npz", z_true=z_true[i].cpu().numpy(), z_imag=zs[i].cpu().numpy(),
                 history=HS, episode=starts[g][i][3])  # episode: dataset episode index, -1 = not from the dataset  # for latent_map
        video(frames[i], decode(decoder, zs[i], norm), {k: v[i] for k, v in sig.items()},
              f"{g} start (rarity {rar[i]:.3f}; data p99 = {stats['p99']:.3f})", args, out / f"example_{g}")
    env.close()

    make_maps(ref_states, stats, {g: np.array(res[g]["start_states"]) for g in GROUPS}, out, rng,
              stats["cand_states"], stats["cand_score"])
    make_full_map(stats, {g: np.array(res[g]["start_states"]) for g in GROUPS},
                  {g: np.array(res[g]["rarity"]) for g in GROUPS}, out, rng, stats["cand_states"], stats["cand_score"])

    # signals vs imagination step, per group (median and IQR)
    hs = np.arange(1, args.horizon + 1)
    panels = (("pix_total", "pixel error, imagined vs real (what you see)", True),
              ("pix_dyn", "pixel error, imagined vs decoded truth (dynamics)", True),
              ("pix_dec", "pixel error, decoded truth vs real (decoder floor)", True),
              ("err_rel", "latent error / copy-last-frame error", True),
              ("u_f", "$u_f$: flow drift", False),
              ("u_r", "$u_r$: round trip of imagined latent", False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, name, log) in zip(axes.ravel(), panels):
        for g in GROUPS:
            v = all_sig[g][key]
            ax.plot(hs, np.median(v, 0), "o-", ms=3, c=COLORS[g], label=g)
            ax.fill_between(hs, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=COLORS[g], alpha=0.15)
        ax.set_title(name, fontsize=9.5, loc="left")
        ax.set_xlabel("imagination step")
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both" if log else "major")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Object-pixel MSE (RGB in [0,1]); median and IQR over start states", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "signals.png", dpi=120)
    plt.close(fig)

    # proxies vs truth, every (start, step)
    cat = lambda key: np.concatenate([all_sig[g][key] for g in GROUPS])  # noqa: E731  (N_all, H)
    corr = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, truth in enumerate(("pix_total", "pix_dyn")):
        for col, key in enumerate(("u_f", "u_r")):
            ax = axes[row, col]
            for g in GROUPS:
                ax.scatter(all_sig[g][key].ravel(), all_sig[g][truth].ravel(), s=5, alpha=0.4, c=COLORS[g], label=g)
            per_h = [spearmanr(cat(key)[:, h], cat(truth)[:, h])[0] for h in range(args.horizon)]
            roll = spearmanr(cat(key).mean(1), cat(truth).mean(1))[0]
            corr[f"{key}~{truth}"] = {"per_step": per_h, "rollout": roll}
            ax.set_yscale("log")
            ax.set_xlabel(key)
            ax.set_ylabel(truth)
            ax.set_title(f"{key} vs {truth}: Spearman step 1 {per_h[0]:.2f}, step {args.horizon} {per_h[-1]:.2f}, "
                         f"rollout {roll:.2f}", fontsize=8.5)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=120)
    plt.close(fig)
    for k, v in corr.items():
        print(f"Spearman {k:14s} per step", np.round(v["per_step"], 2), f"rollout {v['rollout']:.2f}")

    (out / "results.json").write_text(json.dumps({"args": vars(args), "rarity_percentiles": {
        k: float(stats[k]) for k in ("p50", "p99", "p999")}, "spearman": corr, "groups": res}))
    print("wrote", out)


if __name__ == "__main__":
    main()
