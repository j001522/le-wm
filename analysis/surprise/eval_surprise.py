"""Surprise tests on Push-T: one mid-episode event per experiment.

python -m analysis.surprise.eval_surprise --run lewm_flow_e2e --epoch 14 --events color goal block

Events (each is its own experiment, applied at --event-step):
  color  the block changes colour (--color)
  goal   the goal pose jumps (--goal x y angle)
  block  the block jumps to a new pose, velocity zeroed (--block x y angle; default: the centre
         of the board quadrant farthest from block and agent, rotated 90°)

A dataset episode is replayed in the simulator (replay is exact: same states and frames). The
same episode is also replayed without the event (control) with the same flow noise.

Per model frame k (one frame every FRAMESKIP env steps):
  u_f(k): flow endpoint drift of the step that predicts z_k, ||x - x̂(τ=0)|| / sqrt(D),
          averaged over --samples noise draws (band = min..max)
  u_r(k): encoder-decoder round trip, ||z_k - E(D(z_k))|| / sqrt(D), z_k = E(o_k)
  err(k): one-step error ||ẑ_k - z_k||^2 / D (saved in the json, not plotted)
  move(k): latent movement ||z_k - z_{k-1}|| / sqrt(D); u_f and u_r are also plotted divided by it

--imagine onestep (default): ẑ_k is predicted from the true last HS frames, so the
  imagination sees the event through its history.
--imagine openloop: ẑ_k is rolled out from the first HS frames only; it never sees it.

Outputs in results/surprise/<run>_e<epoch>/, per event: <tag>.mp4 (H.264) and .gif (left: simulator,
right: decoded imagination, frame counter on top), <tag>_signals.png (u_f and u_r vs frame,
event marked, control in grey), <tag>.json (all per-frame numbers).
"""

import argparse
import json

import cv2
import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (pixel compression filter)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch

from helpers.constants import FRAMESKIP, SWM, run_dir
from helpers.models import load_run
from helpers.video import imagenet_norm, put, write_video

HS = 3
BOARD = 512  # Push-T window size in simulator units
NAMES = {"color": "block colour change", "goal": "goal teleport", "block": "block teleport"}



def make_hook(event, args):
    if event == "color":
        color = np.array(args.color, np.uint8)
        return lambda env: env.variation_space["block"]["color"].set_value(color)
    if event == "goal":
        return lambda env: setattr(env, "goal_pose", np.array(args.goal, np.float64))

    def teleport_block(env):
        if args.block is None:  # quadrant centre farthest from both block and agent, angle +90°
            cands = np.array([(128, 128), (384, 128), (128, 384), (384, 384)], np.float64)
            dist = np.minimum(np.linalg.norm(cands - np.array(env.block.position), axis=1),
                              np.linalg.norm(cands - np.array(env.agent.position), axis=1))
            x, y = cands[dist.argmax()]
            pose = (x, y, env.block.angle + np.pi / 2)
        else:
            pose = args.block
        env.block.position = tuple(pose[:2])
        env.block.angle = float(pose[2])
        env.block.velocity = (0.0, 0.0)
        env.block.angular_velocity = 0.0
        env.space.reindex_shapes_for_body(env.block)
        gap = np.linalg.norm(np.array(env.agent.position) - np.array(pose[:2]))
        print(f"block teleported to ({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:.2f}); agent at {gap:.0f} units")

    return teleport_block


def simulate(state0, actions, n_frames, events):
    """Replay actions from state0; return n_frames frames (one per FRAMESKIP env steps).
    events: {frame index: callable(env)}, applied before that frame is rendered."""
    env = gym.make("swm/PushT-v1").unwrapped
    env.reset(seed=0)
    env._set_state(state0)
    frames = []
    for k in range(n_frames):
        if k > 0:
            for t in range(FRAMESKIP * (k - 1), FRAMESKIP * k):
                env.step(actions[t])
        if k in events:
            events[k](env)
        frames.append(env.render().copy())
    env.close()
    return np.stack(frames)


@torch.no_grad()
def analyse(model, decoder, frames, act, args, norm):
    """frames (T, 224, 224, 3) uint8, act (T, 10) normalised. Returns per-frame signals and decoded imagination."""
    mean, std = norm
    px = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255.0
    px = (px - mean) / std
    z = model.encode({"pixels": px[None]})["emb"][0]  # (T, D)
    T, D = z.shape
    rec = decoder(z)
    z_rt = model.encode({"pixels": rec[None]})["emb"][0]
    u_r = ((z - z_rt).norm(dim=-1) / D**0.5).cpu().numpy()

    M = args.samples
    a = torch.from_numpy(act).cuda()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eps = torch.randn(M, T, D, device="cuda", generator=gen)  # same noise for control and event runs
    z_hat = torch.full((M, T, D), float("nan"), device="cuda")
    u_f = torch.full((M, T), float("nan"), device="cuda")
    if args.imagine == "onestep":
        for k in range(HS, T):
            roll, drift = model.rollout_latents(
                z[None, k - HS : k].expand(M, -1, -1), a[None, k - HS : k].expand(M, -1, -1),
                a[None, k:k].expand(M, -1, -1), HS, eps[:, k : k + 1])
            z_hat[:, k], u_f[:, k] = roll[:, -1], drift[:, 0]
    else:
        roll, drift = model.rollout_latents(
            z[None, :HS].expand(M, -1, -1), a[None, :HS].expand(M, -1, -1),
            a[None, HS : T - 1].expand(M, -1, -1), HS, eps[:, HS:])
        z_hat[:, HS:], u_f[:, HS:] = roll[:, HS:], drift
    err = (z_hat - z[None]).pow(2).mean(-1)  # (M, T)

    imag = torch.cat([rec[:HS], decoder(z_hat[0, HS:])])  # sample 0; context frames are reconstructions
    imag = ((imag * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    uf = u_f.cpu().numpy()
    # latent movement from the previous frame, same units as u_f and u_r (undefined at frame 0)
    move = np.concatenate([[np.nan], ((z[1:] - z[:-1]).norm(dim=-1) / D**0.5).cpu().numpy()])
    return {"u_r": u_r, "u_f": uf.mean(0), "u_f_min": uf.min(0), "u_f_max": uf.max(0),
            "err": err.mean(0).cpu().numpy(), "move": move,
            "u_f_norm": uf.mean(0) / move, "u_f_norm_min": uf.min(0) / move, "u_f_norm_max": uf.max(0) / move,
            "u_r_norm": u_r / move}, imag



def event_video(frames, imag, sig, event, ke, args, path):
    T, up = len(frames), 2
    out = []
    for k in range(T):
        left = cv2.resize(frames[k], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(imag[k], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        put(left, "simulator", (8, 22), 0.6, (40, 40, 40))
        put(right, "imagination (decoded)" if k >= HS else "context (reconstruction)", (8, 22), 0.6, (40, 40, 40))
        bar = np.full((64, left.shape[1] * 2 + 8, 3), 255, np.uint8)
        uf = "  -  " if np.isnan(sig["u_f"][k]) else f"{sig['u_f'][k]:.3f}"
        put(bar, f"frame {k:2d} / {T - 1}", (10, 26), 0.8, (0, 0, 0), 2)
        put(bar, f"u_f {uf}   u_r {sig['u_r'][k]:.3f}", (10, 54), 0.6, (60, 60, 60))
        on = k >= ke
        put(bar, f"{NAMES[event]} @ frame {ke}" + ("  [active]" if on else ""), (330, 26), 0.6,
            (200, 30, 30) if on else (150, 150, 150), 2 if k == ke else 1)
        gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
        out.append(np.concatenate([bar, np.concatenate([left, gap, right], 1)], 0))
    write_video(out, path, args.fps, gif=True)


def plot(sig, ctrl, event, ke, args, path):
    T = len(sig["u_r"])
    ks = np.arange(T)
    rows = (("u_f", "$u_f$: flow endpoint drift", False),
            ("u_r", "$u_r$: encoder-decoder round trip", False),
            ("u_f_norm", r"$u_f\,/\,\|z_k - z_{k-1}\|$: drift per unit of latent movement (log scale)", True),
            ("u_r_norm", r"$u_r\,/\,\|z_k - z_{k-1}\|$: round trip per unit of latent movement (log scale)", True))
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 10), sharex=True)
    for ax, (key, title, log) in zip(axes, rows):
        ax.plot(ks, ctrl[key], color="0.6", lw=1.5, label="control (no event)")
        ax.plot(ks, sig[key], color="C0", lw=2, marker="o", ms=3, label=NAMES[event])
        if key.startswith("u_f"):
            ax.fill_between(ks, sig[key + "_min"], sig[key + "_max"], color="C0", alpha=0.2,
                            label=f"min-max over {args.samples} noise draws")
        ax.axvline(ke, color="C3", ls="--", lw=1.2, label=f"event @ frame {ke}")
        ax.set_title(title, fontsize=10, loc="left")
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both" if log else "major")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel(f"frame (same counter as the video; 1 frame = {FRAMESKIP} env steps)")
    axes[-1].set_xticks(ks[:: max(1, T // 20)])
    axes[-1].set_xlim(-0.5, T - 0.5)
    fig.suptitle(f"{NAMES[event]}: {args.run} epoch {args.epoch}, episode {args.episode}, "
                 f"imagination {args.imagine}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path.parent / f"{path.name}_signals.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--episode", type=int, default=-1, help="dataset episode; -1 = first with enough steps")
    p.add_argument("--frames", type=int, default=40, help="model frames (FRAMESKIP env steps each)")
    p.add_argument("--events", nargs="+", choices=list(NAMES), default=list(NAMES))
    p.add_argument("--event-step", type=int, default=20, help="frame at which the event happens")
    p.add_argument("--color", type=int, nargs=3, default=[220, 40, 40], help="new block RGB")
    p.add_argument("--goal", type=float, nargs=3, default=[150.0, 370.0, -1.0], help="new goal x y angle")
    p.add_argument("--block", type=float, nargs=3, default=None, help="new block x y angle (default: farthest quadrant)")
    p.add_argument("--imagine", choices=["onestep", "openloop"], default="onestep")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args()

    with h5py.File(SWM / "datasets" / f"{args.dataset}.h5", "r") as f:
        L, O = f["ep_len"][:], f["ep_offset"][:]
        need = args.frames * FRAMESKIP
        if args.episode < 0:
            args.episode = int(np.flatnonzero(L >= need)[0])
        assert L[args.episode] >= need, f"episode {args.episode} has {L[args.episode]} steps, need {need}"
        o = O[args.episode]
        state0 = f["state"][o]
        actions = f["action"][o : o + need]
        all_act = f["action"][:]
    ok = ~np.isnan(all_act).any(1)
    a_mean, a_std = all_act[ok].mean(0), all_act[ok].std(0)
    act = np.nan_to_num((actions - a_mean) / a_std).reshape(args.frames, -1).astype(np.float32)

    model, decoder = load_run(args.run, args.epoch)
    norm = imagenet_norm()

    ke = args.event_step
    ctrl, _ = analyse(model, decoder, simulate(state0, actions, args.frames, {}), act, args, norm)
    out = run_dir("surprise", args.run, args.epoch)
    out.mkdir(parents=True, exist_ok=True)
    for event in args.events:
        frames = simulate(state0, actions, args.frames, {ke: make_hook(event, args)})
        sig, imag = analyse(model, decoder, frames, act, args, norm)
        path = out / f"{event}_ep{args.episode}_{args.imagine}"
        event_video(frames, imag, sig, event, ke, args, path)
        plot(sig, ctrl, event, ke, args, path)
        res = {"args": vars(args), "event": event, "event_frame": ke,
               "event_run": {k: np.round(v, 5).tolist() for k, v in sig.items()},
               "control_run": {k: np.round(v, 5).tolist() for k, v in ctrl.items()}}
        path.with_suffix(".json").write_text(json.dumps(res, default=lambda x: None if x != x else x))
        w = slice(ke, min(ke + 3, args.frames))
        print(f"{NAMES[event]} @ {ke}: u_f {np.nanmean(sig['u_f'][w]):.4f} (control {np.nanmean(ctrl['u_f'][w]):.4f}), "
              f"u_r {np.mean(sig['u_r'][w]):.4f} (control {np.mean(ctrl['u_r'][w]):.4f}), "
              f"err {np.nanmean(sig['err'][w]):.4f} (control {np.nanmean(ctrl['err'][w]):.4f})")
        print("  wrote", path.with_suffix(".mp4").name, path.with_suffix(".gif").name, f"{path.name}_signals.png")


if __name__ == "__main__":
    main()
