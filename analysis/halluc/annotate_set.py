"""Build a hand-labelling set from windows already exported (e.g. the planning replans).

    python -m analysis.halluc.annotate_set --sets plan_pusht_s42 plan_pusht_s43 plan_pusht_s44 \
        plan_pusht_s45 --set plan_v1 --n 120 --blocks 4

Planning windows are cut short when the episode ends (budget or success), so they have different
lengths; annotate.html needs one length. Windows with at least --blocks imagined blocks are kept and
truncated to exactly --blocks (T = 1 + blocks); --n of them are drawn at random (no selection on any
metric, so the AUROCs measured on the set stay unbiased) and get new shuffled ids. Writes the
analysis/halluc/rollout.py SetWriter layout with annotate=True (frames, manifest.js, annotate.html)
and the pred CSVs; key.csv keeps the source set/id. Run analysis.halluc.metrics on the new set after.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.halluc.rollout import SetWriter, set_root


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pusht_LeWM-MSE-dFH-D")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--root", default=None, help="default: results/halluc/<run>_e<epoch>")
    p.add_argument("--sets", nargs="+", required=True, help="source sets")
    p.add_argument("--set", required=True, help="new annotation set")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--blocks", type=int, default=4, help="imagined blocks per window (shorter ones are dropped)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.root) if a.root else set_root(a.run, a.epoch)

    cand = []
    for s in a.sets:
        for r in csv.DictReader(open(root / s / "key.csv")):
            n_imag = int(r["n_imag"]) if r.get("n_imag") not in (None, "") else None
            if n_imag is None or n_imag >= a.blocks:
                cand.append((s, r))
    rng = np.random.default_rng(a.seed)
    pick = [cand[i] for i in rng.permutation(len(cand))[:a.n]]
    print(f"[select] {len(pick)} of {len(cand)} windows with >= {a.blocks} imagined blocks from {a.sets}")

    T = 1 + a.blocks
    w = SetWriter(root, a.set, ctx=1, annotate=True)
    for k, (s, r) in enumerate(pick):  # pick is already in random order -> shuffled ids
        d = np.load(root / s / "data" / f"{r['id']}.npz")
        meta = json.loads(str(d["meta"]))
        pred = {key[len("pred_"):]: d[key][:a.blocks] for key in d.files if key.startswith("pred_")}
        sid = f"{a.set}_{k + 1:04d}"
        meta.update({"source_set": s, "source_id": r["id"], "n_imag": a.blocks})
        for x in ("id", "task", "ctx", "horizon"):
            meta.pop(x, None)
        key = {x: r[x] for x in r if x not in ("id", "task", "idx")}
        key["n_imag"] = a.blocks
        w.add(sid, r["task"], d["real"][:T], d["wm"][:T], d["recon"][:T], d["actions"], d["z"][:T],
              d["z_real"][:T], pred, meta, {"idx": k, **key, "source_set": s, "source_id": r["id"]})
    w.close()


if __name__ == "__main__":
    main()
