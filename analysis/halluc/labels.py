"""Score the label metrics (and the predictors) against the hand labels; fits the label threshold.

    python -m analysis.halluc.labels --run pusht_LeWM-MSE-dFH-D --sets pusht_v1

Port of dreamer4's eval_annotations.py (same statistics, so the two models' tables compare).
Inputs per set, under results/halluc/<run>_e<epoch>/:
    <set>/key.csv, <set>/annotations/*.json   annotate.html exports; the newest row per id wins
    _hidden_metrics/{mask,dino,pred}_<set>/   per-frame CSVs (analysis.halluc.metrics, the export)
Labels: verdict yes = 1, no = 0; unsure is dropped.
Per window every per-frame metric is summarised over the imagined frames as mean / last4 / worst and
oriented so that higher = more hallucinated; the paper's dPSNR (pooled over the horizon) is a baseline.
Per task (sets pooled, and per set): AUROC with a 95% cluster-bootstrap CI over episodes, the
balanced-accuracy-optimal threshold, its grouped 5-fold CV balanced accuracy (folds by episode),
and Spearman correlations between the metrics. Output: <root>/eval_<sets>/results.csv (the
threshold for analysis.halluc.score), correlations.csv.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

# per-frame column -> (source, direction): +1 = higher means more hallucinated
METRICS = {
    "iou": ("mask", "iou_wm", -1),
    "hd95": ("mask", "hd95_wm", +1),
    "hausdorff": ("mask", "hausdorff_wm", +1),
    "chamfer": ("mask", "chamfer_wm", +1),
    "psnr_masked": ("mask", "psnr_m_wm", -1),
    "dino_reg_recon": ("dino", "base_reg_recon_mean", +1),
    "dino_base_recon": ("dino", "base_recon_mean", +1),
    "dino_reg_real": ("dino", "base_reg_real_mean", +1),
    "dino_reg_recon_fg": ("dino", "base_reg_recon_fg", +1),
    "dino_reg_recon_p95": ("dino", "base_reg_recon_p95", +1),
}
SIGN = {**{m: v[2] for m, v in METRICS.items()}, "paper_dpsnr": -1, "paper_label": +1}
NOT_SCORES = ("t", "paper_psnr_wm", "paper_psnr_copy")
N_BOOT = 1000


# ------------------------------------------------------------------ loading

def frames_of(root, name, sid, task):
    frames = {}
    for src in ("mask", "dino", "pred"):
        p = Path(root) / "_hidden_metrics" / f"{src}_{name}" / f"{sid}_{task}.csv"
        if p.exists():
            frames[src] = list(csv.DictReader(open(p)))
    return frames


def load_key(root, name):
    return {r["id"]: r for r in csv.DictReader(open(Path(root) / name / "key.csv"))}


def load_set(root, name):
    """Hand-labelled windows of one set: dicts with y (0/1), group (episode) and per-frame CSV rows."""
    labels = {}
    for p in sorted(glob.glob(str(Path(root) / name / "annotations" / "*.json"))):
        for r in json.load(open(p))["rows"]:
            old = labels.get(r["id"])
            if old is None or (r.get("updated") or "") >= (old.get("updated") or ""):
                labels[r["id"]] = r
    seqs = []
    for sid, k in load_key(root, name).items():
        a = labels.get(sid)
        if a is None or a.get("verdict") not in ("yes", "no"):
            continue
        frames = frames_of(root, name, sid, k["task"])
        if frames:
            seqs.append({"set": name, "id": sid, "task": k["task"], "group": f"{name}:{k['episode']}",
                         "y": int(a["verdict"] == "yes"), "key": k, "frames": frames})
    return seqs


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def scores(seq):
    """{(metric, agg): oriented score} for one window. Predictors are "pred:<column>" (higher = worse)."""
    out = {}
    for m, (src, col, sign) in METRICS.items():
        rows = seq["frames"].get(src)
        if not rows or col not in rows[0]:
            continue
        v = np.array([fnum(r[col]) for r in rows]) * sign
        if not np.isfinite(v).any():
            continue
        out[(m, "mean")] = np.nanmean(v)
        out[(m, "last4")] = np.nanmean(v[-4:]) if np.isfinite(v[-4:]).any() else np.nan
        out[(m, "worst")] = np.nanmax(v)
    rows = seq["frames"].get("pred") or []
    for col in (rows[0] if rows else []):
        if col in NOT_SCORES:
            continue
        v = np.array([fnum(r[col]) for r in rows])
        if np.isfinite(v).any():
            out[("pred:" + col, "mean")] = np.nanmean(v)
            out[("pred:" + col, "last4")] = np.nanmean(v[-4:])
            out[("pred:" + col, "worst")] = np.nanmax(v)
    rows = seq["frames"].get("mask") or seq["frames"].get("dino") or seq["frames"].get("pred")
    pw = np.array([fnum(r["paper_psnr_wm"]) for r in rows])
    pc = np.array([fnum(r["paper_psnr_copy"]) for r in rows])
    mse = lambda p: np.mean(10 ** (-p / 10))
    dpsnr = 10 * np.log10(mse(pc) / mse(pw))
    out[("paper_dpsnr", "pooled")] = -dpsnr  # lower dPSNR = worse
    out[("paper_label", "dpsnr<=0")] = float(dpsnr <= 0)
    return out


# ------------------------------------------------------------------ statistics

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s)  # average ranks for ties
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot(s, y, g, n=1000, seed=0):
    """95% CI of the AUROC, resampling episodes (windows of one episode are correlated)."""
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in groups}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(groups, len(groups), replace=True)])
        a = auroc(s[pick], y[pick])
        if np.isfinite(a):
            vals.append(a)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def bal_acc(pred, y):
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return np.nanmean([tpr, tnr]), tpr, tnr


def best_threshold(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    cand = np.unique(s)
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 401))
    best = (-1, np.nan)
    for c in cand:
        b = bal_acc(s >= c, y)[0]
        if b > best[0]:
            best = (b, c)
    return best[1]


def cv_bal_acc(s, y, g, k=5, seed=0):
    """Grouped k-fold: threshold fit on the other folds, balanced accuracy of the pooled held-out predictions."""
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    rng.shuffle(groups)
    pred = np.zeros(len(s), bool)
    for f in np.array_split(groups, min(k, len(groups))):
        te = np.isin(g, f)
        if len(np.unique(y[~te])) < 2:
            return np.nan
        pred[te] = s[te] >= best_threshold(s[~te], y[~te])
    ok = np.isfinite(s)
    return bal_acc(pred[ok], y[ok])[0]


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return np.corrcoef(ra, rb)[0, 1] if ok.sum() > 2 else np.nan


# ------------------------------------------------------------------ main

def evaluate(seqs, title, out_rows):
    y = np.array([q["y"] for q in seqs])
    g = np.array([q["group"] for q in seqs])
    sc = [scores(q) for q in seqs]
    keys = sorted({k for s in sc for k in s}, key=lambda k: (k[0].startswith("paper"), k[0], k[1]))
    print(f"\n=== {title}: n={len(y)}  yes={y.sum()}  no={len(y) - y.sum()}  episodes={len(np.unique(g))}")
    if y.sum() == 0 or y.sum() == len(y):
        print("   single class: AUROC undefined")
        return {}
    res = {}
    for k in keys:
        s = np.array([d.get(k, np.nan) for d in sc])
        a = auroc(s, y)
        lo, hi = cluster_boot(s, y, g, n=N_BOOT)
        thr = best_threshold(s, y)
        ba, tpr, tnr = bal_acc(s >= thr, y) if np.isfinite(thr) else (np.nan,) * 3
        sign = SIGN.get(k[0], 1)
        res[k] = dict(auroc=a, lo=lo, hi=hi, thr=thr * sign if np.isfinite(thr) else np.nan,
                      bal_acc=ba, tpr=tpr, tnr=tnr, cv_bal_acc=cv_bal_acc(s, y, g), s=s)
        out_rows.append({"subset": title, "metric": k[0], "agg": k[1], "n": len(y), "yes": int(y.sum()),
                         "auroc": a, "auroc_lo": lo, "auroc_hi": hi, "threshold_raw": res[k]["thr"],
                         "direction": ">=" if sign > 0 else "<=", "bal_acc_fit": ba, "tpr": tpr, "tnr": tnr,
                         "bal_acc_cv": res[k]["cv_bal_acc"]})
    print(f"   {'metric':24s} {'agg':9s} {'AUROC [95% CI]':22s} {'threshold':>14s} {'balAcc fit':>10s} "
          f"{'TPR/TNR':>11s} {'balAcc CV':>9s}")
    for (m, agg), r in sorted(res.items(), key=lambda kv: -np.nan_to_num(kv[1]["auroc"], nan=-1)):
        d = ">=" if SIGN.get(m, 1) > 0 else "<="
        thr = f"{d} {r['thr']:.4g}" if np.isfinite(r["thr"]) else "-"
        print(f"   {m:24s} {agg:9s} {r['auroc']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]      {thr:>14s} "
              f"{r['bal_acc']:10.3f} {r['tpr']:5.2f}/{r['tnr']:.2f} {r['cv_bal_acc']:9.3f}")
    return res


def main():
    from analysis.halluc.rollout import set_root

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pusht_LeWM-MSE-dFH-D")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--root", default=None, help="default: results/halluc/<run>_e<epoch>")
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    global N_BOOT
    N_BOOT = a.n_boot
    root = Path(a.root) if a.root else set_root(a.run, a.epoch)

    seqs = [q for s in a.sets for q in load_set(root, s)]
    if not seqs:
        raise SystemExit("no labelled windows with metrics (annotations/*.json and _hidden_metrics/ present?)")
    out = root / f"eval_{'+'.join(a.sets)}"
    out.mkdir(parents=True, exist_ok=True)
    out_rows, corr_rows = [], []
    for task in sorted({q["task"] for q in seqs}):
        tq = [q for q in seqs if q["task"] == task]
        res = evaluate(tq, f"{task} [{'+'.join(a.sets)}]", out_rows)
        for s in a.sets:
            sq = [q for q in tq if q["set"] == s]
            if sq and len(sq) != len(tq):
                evaluate(sq, f"{task} [{s}]", out_rows)
        names = [k for k in res if k[1] in ("mean", "pooled")]
        for x in names:
            corr_rows.append({"task": task, "metric": x[0],
                              **{y[0]: spearman(res[x]["s"], res[y]["s"]) for y in names}})
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    if corr_rows:
        keys = sorted({k for r in corr_rows for k in r}, key=lambda k: (k not in ("task", "metric"), k))
        with open(out / "correlations.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(corr_rows)
    print(f"\n[out] {out}/results.csv, correlations.csv")


if __name__ == "__main__":
    main()
