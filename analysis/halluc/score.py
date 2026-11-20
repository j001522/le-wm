"""Score the label-free predictors against automatic hallucination labels.

    python -m analysis.halluc.score --run pusht_LeWM-MSE-dFH-D --sets auto_val_pusht --thr-from eval_pusht_v1
    python -m analysis.halluc.score --run pusht_LeWM-MSE-dFH-D --sets plan_pusht --thr-from eval_pusht_v1

Label (per window): the label metric chosen on the hand labels (default: masked Hausdorff, mean over
the imagined frames) >= its threshold. The threshold is the balanced-accuracy-optimal one that
analysis.halluc.labels fitted on the hand-labelled sets (--thr-from <eval dir>/results.csv, row
metric/agg, subset = the task over all hand-labelled sets), or --thr.
Per-window metric and predictor scores are analysis.halluc.labels.scores (same summaries and
orientation as on the hand labels): "pred:<col>" for every column of the pred CSVs.
Reported per set and pooled: n, hallucination rate, AUROC with a 95% cluster-bootstrap CI (episodes),
AUROC within equal-count motion bins (motion = predicted latent motion, mean over the window), and
for planning sets the same restricted to complete windows (all 5 blocks reached).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.halluc import labels as ev
from analysis.halluc.rollout import set_root

N_MBINS, MIN_BIN = 5, 3


def load(root, name):
    seqs = []
    for sid, k in ev.load_key(root, name).items():
        frames = ev.frames_of(root, name, sid, k["task"])
        if "pred" not in frames:
            continue
        seqs.append({"set": name, "id": sid, "task": k["task"], "group": f"{name}:{k['episode']}",
                     "key": k, "frames": frames})
    return seqs


def threshold(path, metric, agg, task):
    rows = [r for r in csv.DictReader(open(path)) if r["metric"] == metric and r["agg"] == agg
            and r["subset"].startswith(task + " [")]
    if not rows:
        raise SystemExit(f"no {metric}/{agg} row for {task} in {path}")
    r = max(rows, key=lambda r: r["subset"].count("+"))  # the row over all hand-labelled sets
    return float(r["threshold_raw"]), r


def eq_motion_auroc(v, y, m):
    ok = np.isfinite(v) & np.isfinite(m)
    v, y, m = v[ok], y[ok], m[ok]
    edges = np.quantile(m, np.linspace(0, 1, N_MBINS + 1))
    b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, N_MBINS - 1)
    aus, ws = [], []
    for k in range(N_MBINS):
        i = b == k
        if min(int(y[i].sum()), int((1 - y[i]).sum())) >= MIN_BIN:
            aus.append(ev.auroc(v[i], y[i]))
            ws.append(int(i.sum()))
    return (float(np.average(aus, weights=ws)), len(aus)) if aus else (np.nan, 0)


def report(title, seqs, label_key, thr_oriented, n_boot, rows):
    sc = [ev.scores(q) for q in seqs]
    lab = np.array([d.get(label_key, np.nan) for d in sc])
    keep = np.isfinite(lab)
    sc = [d for d, k in zip(sc, keep) if k]
    seqs = [q for q, k in zip(seqs, keep) if k]
    y = (lab[keep] >= thr_oriented).astype(int)
    g = np.array([q["group"] for q in seqs])
    motion = np.array([d.get(("pred:motion", "mean"), np.nan) for d in sc])
    print(f"\n=== {title}: n={len(y)}  hallucinated={y.sum()} ({100 * y.mean():.1f}%)  "
          f"episodes={len(np.unique(g))}")
    if y.sum() == 0 or y.sum() == len(y):
        print("   single class: AUROC undefined")
        return
    keys = sorted({k for d in sc for k in d if k[0].startswith("pred:") or k[0].startswith("paper_")
                   or k[0] in ("dino_reg_recon",)})
    out = []
    for k in keys:
        s = np.array([d.get(k, np.nan) for d in sc])
        if not np.isfinite(s).any():
            continue
        a = ev.auroc(s, y)
        lo, hi = ev.cluster_boot(s, y, g, n=n_boot)
        em, nb = eq_motion_auroc(s, y, motion)
        out.append((k, a, lo, hi, em, nb))
        rows.append({"subset": title, "score": k[0], "agg": k[1], "n": len(y), "halluc": int(y.sum()),
                     "rate": y.mean(), "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
                     "auroc_eq_motion": em, "motion_bins": nb})
    print(f"   {'score':26s} {'agg':7s} {'AUROC [95% CI]':20s} {'eq-motion':>9s}")
    for k, a, lo, hi, em, nb in sorted(out, key=lambda r: -np.nan_to_num(r[1], nan=-1)):
        if k[1] not in ("mean", "pooled"):
            continue
        print(f"   {k[0]:26s} {k[1]:7s} {a:.3f} [{lo:.2f}, {hi:.2f}]    {em:9.3f} ({nb} bins)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pusht_LeWM-MSE-dFH-D")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--root", default=None, help="default: results/halluc/<run>_e<epoch>")
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--label-metric", default="hausdorff", help="eval_annotations METRICS name")
    p.add_argument("--label-agg", default="mean")
    p.add_argument("--thr", type=float, default=None, help="raw threshold (default: from --thr-from)")
    p.add_argument("--thr-from", default=None,
                   help="analysis.halluc.labels output (eval_<sets> dir under the root, or a results.csv)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=None, help="default: <root>/score_<sets>")
    a = p.parse_args()

    a.root = Path(a.root) if a.root else set_root(a.run, a.epoch)
    if a.thr_from and not str(a.thr_from).endswith(".csv"):
        a.thr_from = str((a.root / a.thr_from if not Path(a.thr_from).is_absolute() else Path(a.thr_from))
                         / "results.csv")
    seqs = [q for s in a.sets for q in load(a.root, s)]
    task = seqs[0]["task"]
    if a.thr is None:
        if not a.thr_from:
            raise SystemExit("give --thr or --thr-from")
        a.thr, src = threshold(a.thr_from, a.label_metric, a.label_agg, task)
        print(f"[label] {a.label_metric}/{a.label_agg} {src['direction']} {a.thr:.4g} from {a.thr_from} "
              f"({src['subset']}: hand-label AUROC {float(src['auroc']):.3f}, balAcc CV {float(src['bal_acc_cv']):.3f})")
    sign = ev.SIGN.get(a.label_metric, 1)
    label_key = (a.label_metric, a.label_agg)
    out = Path(a.out or Path(a.root) / f"score_{'+'.join(a.sets)}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.sets:
        sq = [q for q in seqs if q["set"] == s]
        report(s, sq, label_key, a.thr * sign, a.n_boot, rows)
        full = [q for q in sq if q["key"].get("n_imag") not in (None, "") and int(q["key"]["n_imag"]) == 5]
        if full and len(full) != len(sq):
            report(f"{s} [complete windows]", full, label_key, a.thr * sign, a.n_boot, rows)
        for flag in ("1", "0"):
            sub = [q for q in sq if q["key"].get("success") == flag]
            if sub:
                report(f"{s} [success={flag}]", sub, label_key, a.thr * sign, a.n_boot, rows)
    if len(a.sets) > 1:
        report("+".join(a.sets), seqs, label_key, a.thr * sign, a.n_boot, rows)
    with open(out / "predictors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "label.json").write_text(json.dumps({"metric": a.label_metric, "agg": a.label_agg,
                                                "threshold_raw": a.thr, "sets": a.sets}, indent=1))
    print(f"\n[out] {out}/predictors.csv")


if __name__ == "__main__":
    main()
