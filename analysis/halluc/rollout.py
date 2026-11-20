"""Hallucination experiments: open-loop imagination of one plan-length window plus the label-free
predictors, and the frozen export format (the same as dreamer4's export_annotation_set.py, so
annotate.html and the metric definitions carry over unchanged).

Imagination (as the planner with --mse-head): context latents -> MSE head, autoregressive, history 3.
Predictors per imagined step k = 1..P (dreamer4 definitions, LeWM latents, RMS = norm / sqrt(D)):
    u_r     RMS(z_k - enc(dec(z_k)))                decoder -> encoder round-trip residual
    u_f     flow-head instability: mean RMS change of x_hat between consecutive Euler steps over
            the second half of the ODE, mean over seeds (flow head on the same trunk state as z_k)
    u_s     variance of the flow-head sample over seeds, mean over latent dims
    drift   flow-head one-sample conditional std |x - x_hat(tau=0)| / sqrt(D), mean over seeds
    motion  RMS(z_k - z_{k-1});  <u>_norm = u / max(motion, 1e-3)
Oracle columns (need the real future, not predictors): oracle_e_z = RMS(z_k - enc(real_k)),
oracle_motion_real = RMS(enc(real_k) - enc(real_{k-1})).

Export layout (<root> = results/halluc/<run>_e<epoch>, <set dir> = <root>/<set>):
    data/<id>.npz       real, wm, recon (T,H,W,3) uint8; actions (T,A); act_mask; z, z_real (T,D); meta (json)
    frames/<id>/{gt,wm}_NN.png, key.csv, manifest.js, annotate.html   (annotation sets only)
    <root>/_hidden_metrics/pred_<set>/<id>_<task>.csv   per imagined frame t = ctx..T-1
"""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from helpers.constants import run_dir
from helpers.video import decode, encode

ANNOTATE_HTML = Path(__file__).with_name("annotate.html")  # dreamer4's annotation tool, unchanged
PRED_COLS = ["t", "paper_psnr_wm", "paper_psnr_copy", "motion", "u_r", "u_r_norm", "u_f", "u_f_norm",
             "u_s", "u_s_norm", "drift", "drift_norm", "oracle_e_z", "oracle_motion_real"]


def set_root(run, epoch):
    """results/halluc/<run>_e<epoch>: every set of one checkpoint, and its _hidden_metrics/."""
    return run_dir("halluc", run, epoch)


def _rms(x):
    return x.norm(dim=-1) / x.size(-1) ** 0.5


@torch.no_grad()
def flow_uncertainty(model, hist, act_hist, n_seeds, seed, steps=None):
    """Flow-head statistics for the next latent given the history: u_f, u_s, drift, each (N,)."""
    if getattr(model, "flow_head", None) is None:
        nan = torch.full((hist.size(0),), float("nan"), device=hist.device)
        return nan, nan, nan
    N, D = hist.size(0), hist.size(-1)
    g = torch.Generator(device=hist.device).manual_seed(seed)
    eps = torch.randn(n_seeds * N, D, generator=g, device=hist.device)
    rep = lambda x: x.repeat(n_seeds, *[1] * (x.ndim - 1))  # seed-major: row s*N + i
    old = model.flow_variant
    model.flow_variant = "head"
    try:
        x, info = model.sample_next(rep(hist), rep(act_hist), eps, steps)
    finally:
        model.flow_variant = old
    st = info["steps"]  # (S*N, K-1) RMS change of x_hat between Euler steps
    u_f = st[:, st.size(1) // 2:].mean(1).view(n_seeds, N).mean(0)
    u_s = x.view(n_seeds, N, D).var(0, unbiased=n_seeds > 1).mean(-1)
    drift = info["total"].view(n_seeds, N).mean(0)
    return u_f, u_s, drift


@torch.no_grad()
def imagine(model, decoder, ctx_px, actions, history_size=3, n_seeds=4, seed=0, real_px=None,
            flow_steps=None):
    """Open-loop imagination of a window.
    ctx_px: (N, H, 3, 224, 224) ImageNet-normalised context frames; actions: (N, P, A) normalised
    action blocks, actions[:, k] taken after the frame at block k (k = 0 is the last context frame).
    real_px: optional (N, H+P, 3, 224, 224) normalised real frames (oracle columns, recon).
    Returns dict of numpy arrays: z (N, H+P, D), wm (N, H+P, h, w, 3) uint8 decoded (context = recon),
    and per imagined step (N, P): u_r, u_f, u_s, drift, motion (+ oracle_e_z, oracle_motion_real)."""
    dev = next(model.parameters()).device
    ctx_px, actions = ctx_px.to(dev).float(), actions.to(dev).float()
    N, H = ctx_px.shape[:2]
    P = actions.size(1)
    assert H == 1, "the planner's context is one frame (history_len = 1); longer contexts need their actions"
    emb = model.encode({"pixels": ctx_px})["emb"]  # (N, H, D)
    act = actions[:, :1]
    uf, us, dr = [], [], []
    for k in range(P):
        act_emb = model.action_encoder(act)
        hist, ah = emb[:, -history_size:], act_emb[:, -history_size:]
        z = model.predict(hist, ah)[:, -1]  # MSE head = the deterministic imagination
        u_f, u_s, d = flow_uncertainty(model, hist, ah, n_seeds, seed + 1000 * k, flow_steps)
        uf.append(u_f), us.append(u_s), dr.append(d)
        emb = torch.cat([emb, z[:, None]], 1)
        if k + 1 < P:
            act = torch.cat([act, actions[:, k + 1:k + 2]], 1)
    Dz = emb.size(-1)
    flat = emb.reshape(-1, Dz)
    wm = decode(decoder, flat)  # uint8 (N*T, h, w, 3)
    z_rt = torch.from_numpy(encode(model, wm)).to(dev)  # enc(dec(z))
    u_r = _rms(flat - z_rt).view(N, H + P)[:, H:]
    motion = _rms(emb[:, H:] - emb[:, H - 1:-1])
    out = {"z": emb.cpu().numpy(), "wm": wm.reshape(N, H + P, *wm.shape[1:]),
           "u_r": u_r.cpu().numpy(), "u_f": torch.stack(uf, 1).cpu().numpy(),
           "u_s": torch.stack(us, 1).cpu().numpy(), "drift": torch.stack(dr, 1).cpu().numpy(),
           "motion": motion.cpu().numpy()}
    if real_px is not None:
        real_px = real_px.to(dev).float()
        zr = model.encode({"pixels": real_px})["emb"]
        out["z_real"] = zr.cpu().numpy()
        out["recon"] = decode(decoder, zr.reshape(-1, Dz)).reshape(N, H + P, *wm.shape[1:])
        out["oracle_e_z"] = _rms(emb[:, H:] - zr[:, H:]).cpu().numpy()
        out["oracle_motion_real"] = _rms(zr[:, H:] - zr[:, H - 1:-1]).cpu().numpy()
    return out


def paper_dpsnr(real, wm, ctx):
    """Per horizon frame PSNR of the WM and of the repeated last context frame vs real (dreamer4 metric_io)."""
    r = real[ctx:].astype(np.float32) / 255.0
    w = wm[ctx:].astype(np.float32) / 255.0
    c = real[ctx - 1].astype(np.float32)[None] / 255.0
    psnr = lambda m: 10 * np.log10(1.0 / np.maximum(m, 1e-12))
    return psnr(((w - r) ** 2).mean(axis=(1, 2, 3))), psnr(((c - r) ** 2).mean(axis=(1, 2, 3)))


class SetWriter:
    """Writes one export set in the dreamer4 format. annotate=True also writes the PNG frames,
    manifest.js and annotate.html for hand labelling (ids are shuffled by the caller)."""

    def __init__(self, root, name, ctx, annotate=False):
        self.root, self.name, self.ctx, self.annotate = Path(root), name, ctx, annotate
        self.dir = self.root / name
        if (self.dir / "key.csv").exists():
            raise SystemExit(f"{self.dir} already holds an export; refusing to overwrite "
                             "(annotations refer to these exact frames)")
        (self.dir / "data").mkdir(parents=True, exist_ok=True)
        self.pred_dir = self.root / "_hidden_metrics" / f"pred_{name}"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.key, self.T, self.size = [], None, None

    def add(self, sid, task, real, wm, recon, actions, z, z_real, pred, meta, key_extra):
        """real/wm/recon (T,h,w,3) uint8; pred: dict of (P,) arrays (P = T - ctx)."""
        import imageio.v2 as imageio

        T = real.shape[0]
        self.T, self.size = T, list(real.shape[1:3])
        meta = {"id": sid, "task": task, "ctx": self.ctx, "horizon": T - self.ctx, **meta}
        np.savez_compressed(self.dir / "data" / f"{sid}.npz", real=real, wm=wm, recon=recon,
                            actions=actions, act_mask=np.ones(len(actions), bool), z=z, z_real=z_real,
                            **{f"pred_{k}": np.asarray(v) for k, v in pred.items()}, meta=json.dumps(meta))
        pw, pc = paper_dpsnr(real, wm, self.ctx)
        mo = np.maximum(np.asarray(pred["motion"]), 1e-3)
        with open(self.pred_dir / f"{sid}_{task}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PRED_COLS)
            w.writeheader()
            for i in range(T - self.ctx):
                r = {"t": self.ctx + i, "paper_psnr_wm": pw[i], "paper_psnr_copy": pc[i]}
                for k in ("motion", "u_r", "u_f", "u_s", "drift", "oracle_e_z", "oracle_motion_real"):
                    r[k] = float(pred[k][i]) if k in pred else float("nan")
                for k in ("u_r", "u_f", "u_s", "drift"):
                    r[f"{k}_norm"] = r[k] / mo[i]
                w.writerow(r)
        if self.annotate:
            fdir = self.dir / "frames" / sid
            fdir.mkdir(parents=True, exist_ok=True)
            for t in range(T):
                imageio.imwrite(fdir / f"gt_{t:02d}.png", real[t])
                imageio.imwrite(fdir / f"wm_{t:02d}.png", wm[t])
        self.key.append({"id": sid, "task": task, **key_extra})

    def close(self):
        if not self.key:
            print(f"[out] {self.dir}: nothing written")
            return
        key = sorted(self.key, key=lambda r: r["id"])
        with open(self.dir / "key.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
            w.writeheader()
            w.writerows(key)
        if self.annotate:
            manifest = {"set": self.name, "ctx": self.ctx, "T": self.T, "size": self.size,
                        "seqs": [r["id"] for r in key]}
            (self.dir / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest) + ";\n")
            shutil.copy(ANNOTATE_HTML, self.dir / "annotate.html")
        print(f"[out] {self.dir}: {len(key)} sequences" + ("; open annotate.html" if self.annotate else ""))
