"""Per-frame hallucination label metrics on frozen export sets (no model needed).

    python -m analysis.halluc.metrics --run pusht_LeWM-MSE-dFH-D --sets pusht_v1
    python -m analysis.halluc.metrics --run ... --sets auto_val_pusht --no-dino   # the label metric only

Same definitions as dreamer4 (AIC-Thesis/dreamer4/mmbench2/src mask_metrics.py, dino_metrics.py,
metric_io.paper_dpsnr), so LeWM and dreamer4 numbers are comparable. Per imagined frame t (t >= ctx),
for X in {wm, recon} against real[t]:

mask (analysis/halluc/masks.py foreground, M_X and M_real):
    hausdorff   max of the two directed max distances (px) between M_X and M_real (Euclidean distance
                transform; an empty mask counts as the image diagonal; both empty = 0)
    hd95        the same with the 95th percentile;  chamfer  mean of the two directed means
    iou         |M_X & M_real| / |M_X | M_real|;     psnr_m   PSNR over M_real | M_wm
dino (DINOv2 patch tokens at 224 px, 16x16 patches, HF facebook/dinov2-base and -with-registers-base):
    <model>_<ref>_{mean,p95,fg}   1 - cos(wm patch, ref patch), ref in {real, recon}, mean / 95th pct /
                                  mean over patches touching the foreground (M_real | M_wm)
paper dPSNR columns paper_psnr_wm, paper_psnr_copy (copy = real[ctx-1] repeated) in both.

dreamer4's masked LPIPS is left out (not a selected metric; lpips is not in the LeWM env).
Output: <root>/_hidden_metrics/{mask,dino}_<set>/<id>_<task>.csv (one row per imagined frame).
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from analysis.halluc.masks import foreground_mask, preset_for_task
from analysis.halluc.rollout import paper_dpsnr, set_root

HF_NAMES = {"base": "facebook/dinov2-base", "base_reg": "facebook/dinov2-with-registers-base"}
HF_CACHE = os.environ.get("HALLUC_HF_CACHE", str(Path.home() / ".cache/huggingface/hub"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CANDS = ("wm", "recon")


def load_set(root, name):
    out = []
    for p in sorted(glob.glob(str(Path(root) / name / "data" / "*.npz"))):
        d = np.load(p)
        meta = json.loads(str(d["meta"]))
        out.append({"name": f"{meta['id']}_{meta['task']}", "task": meta["task"], "ctx": int(meta["ctx"]),
                    "real": d["real"], "wm": d["wm"], "recon": d["recon"]})
    return out


# ------------------------------------------------------------------ mask

def shape_metrics(mx, mr, cap):
    if not mx.any() and not mr.any():
        return {"chamfer": 0.0, "hausdorff": 0.0, "hd95": 0.0, "iou": 1.0}
    dt_r = np.minimum(ndimage.distance_transform_edt(~mr), cap) if mr.any() else np.full(mr.shape, cap)
    dt_x = np.minimum(ndimage.distance_transform_edt(~mx), cap) if mx.any() else np.full(mx.shape, cap)
    a, b = dt_r[mx], dt_x[mr]  # d(M_X -> M_real) per M_X pixel, d(M_real -> M_X) per M_real pixel
    stat = lambda v, f: float(f(v)) if v.size else 0.0
    return {"chamfer": (stat(a, np.mean) + stat(b, np.mean)) / 2,
            "hausdorff": max(stat(a, np.max), stat(b, np.max)),
            "hd95": max(stat(a, lambda v: np.percentile(v, 95)), stat(b, lambda v: np.percentile(v, 95))),
            "iou": float((mx & mr).sum() / (mx | mr).sum())}


def mask_rows(seq):
    real, ctx = seq["real"], seq["ctx"]
    T, H, W, _ = real.shape
    cap = float(np.hypot(H, W))
    pr = preset_for_task(seq["task"])
    pw, pc = paper_dpsnr(real, seq["wm"], ctx)
    rows = []
    for i, t in enumerate(range(ctx, T)):
        mr = foreground_mask(real[t], pr)
        m = {c: foreground_mask(seq[c][t], pr) for c in CANDS}
        R = mr | m["wm"]
        r = {"t": t, "fg_real": int(mr.sum()), "region": int(R.sum()),
             "paper_psnr_wm": float(pw[i]), "paper_psnr_copy": float(pc[i])}
        for c in CANDS:
            d2 = (((seq[c][t].astype(np.float32) - real[t].astype(np.float32)) / 255.0) ** 2).mean(-1)
            r[f"psnr_m_{c}"] = 10 * np.log10(1.0 / max(float(d2[R].mean()), 1e-12)) if R.any() else np.nan
            r.update({f"{k}_{c}": v for k, v in shape_metrics(m[c], mr, cap).items()})
        rows.append(r)
    return rows


# ------------------------------------------------------------------ DINOv2

class Dino:
    def __init__(self, name, device, res=224):
        from transformers import AutoModel

        self.name, self.res, self.grid, self.device = name, res, res // 14, device
        self.model = AutoModel.from_pretrained(HF_NAMES[name], cache_dir=HF_CACHE,
                                               local_files_only=True).to(device).eval()

    @torch.no_grad()
    def __call__(self, frames_u8, batch=32):
        out = []
        for s in range(0, len(frames_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[s:s + batch])).permute(0, 3, 1, 2).float() / 255.0
            if x.shape[-1] != self.res:
                x = F.interpolate(x, size=(self.res, self.res), mode="bicubic", align_corners=False)
            h = self.model(pixel_values=((x - MEAN) / STD).to(self.device)).last_hidden_state
            tok = h[:, -self.grid * self.grid:]  # patch tokens come last (after CLS [+ registers])
            out.append(tok.float().reshape(tok.shape[0], self.grid, self.grid, -1).cpu())
        return torch.cat(out)


def patch_fg(mask, grid):
    H, W = mask.shape
    if H % grid or W % grid:
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        return (F.adaptive_max_pool2d(m, grid)[0, 0] > 0).numpy()
    return mask.reshape(grid, H // grid, grid, W // grid).any(axis=(1, 3))


def dino_rows(seq, dinos):
    real, wm, rec, ctx = seq["real"], seq["wm"], seq["recon"], seq["ctx"]
    hz = np.arange(ctx, real.shape[0])
    pr = preset_for_task(seq["task"])
    pw, pc = paper_dpsnr(real, wm, ctx)
    fg = [foreground_mask(real[t], pr) | foreground_mask(wm[t], pr) for t in hz]
    maps = {}
    for dn in dinos:
        f_real, f_wm, f_rec = dn(real[hz]), dn(wm[hz]), dn(rec[hz])
        for ref, f in (("real", f_real), ("recon", f_rec)):
            maps[(dn.name, ref, dn.grid)] = (1.0 - F.cosine_similarity(f_wm, f, dim=-1)).numpy()
    rows = []
    for i, t in enumerate(hz):
        r = {"t": int(t), "paper_psnr_wm": float(pw[i]), "paper_psnr_copy": float(pc[i])}
        for (mname, ref, g), m in maps.items():
            d, pf = m[i], patch_fg(fg[i], g)
            r[f"{mname}_{ref}_mean"] = float(d.mean())
            r[f"{mname}_{ref}_p95"] = float(np.percentile(d, 95))
            r[f"{mname}_{ref}_fg"] = float(d[pf].mean()) if pf.any() else np.nan
        rows.append(r)
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pusht_LeWM-MSE-dFH-D")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--root", default=None, help="default: results/halluc/<run>_e<epoch>")
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--no-dino", action="store_true")
    p.add_argument("--dino", nargs="+", default=["base", "base_reg"], choices=list(HF_NAMES))
    a = p.parse_args()
    root = Path(a.root) if a.root else set_root(a.run, a.epoch)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dinos = [] if a.no_dino else [Dino(n, dev) for n in a.dino]
    for s in a.sets:
        seqs = load_set(root, s)
        print(f"[{s}] {len(seqs)} sequences", flush=True)
        hd = []
        for k, seq in enumerate(seqs):
            rows = mask_rows(seq)
            hd.append(np.mean([r["hausdorff_wm"] for r in rows]))
            write(root / "_hidden_metrics" / f"mask_{s}" / f"{seq['name']}.csv", rows)
            if dinos:
                write(root / "_hidden_metrics" / f"dino_{s}" / f"{seq['name']}.csv", dino_rows(seq, dinos))
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(seqs)}", flush=True)
        print(f"[{s}] mean Hausdorff (wm, px): median {np.median(hd):.2f}, p90 {np.percentile(hd, 90):.2f}")


if __name__ == "__main__":
    main()
