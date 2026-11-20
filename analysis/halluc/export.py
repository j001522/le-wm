"""Open-loop imagination on validation windows, frozen for hand labelling and metric scoring.

    python -m analysis.halluc.export --run pusht_LeWM-MSE-dFH-D --set pusht_v1 --n 100 --annotate
    python -m analysis.halluc.export --run pusht_LeWM-MSE-dFH-D --set auto_val_pusht --n 2000 \
        --exclude-sets pusht_v1

Windows: the training run's validation split (train.py: spt.data.random_split over the 4-frame
training windows, [0.9, 0.1], torch.Generator().manual_seed(3072)); a window is used through its
(episode, start). Val windows come from training episodes (the split is by window, not episode).
Each exported window is one replan of the planner: 1 context frame + P = 5 imagined action blocks
(FRAMESKIP env steps each), i.e. real frames at start + 5k, k = 0..5, and the 25 dataset actions.
Windows are picked at random, pairwise non-overlapping within an episode, never overlapping the
windows of --exclude-sets. Output: analysis/halluc/rollout.py SetWriter layout under
results/halluc/<run>_e<epoch>/ (or --root).
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from helpers.constants import FRAMESKIP, TASKS
from analysis.halluc.rollout import SetWriter, imagine, set_root
from helpers.models import load_run
from helpers.planning import dataset_path, fit_scalers
from helpers.video import imagenet_norm


def val_windows(dataset, train_split=0.9, seed=3072):
    """(episode, start) of every window in the run's validation split."""
    import stable_pretraining as spt

    _, val = spt.data.random_split(dataset, lengths=[train_split, 1 - train_split],
                                   generator=torch.Generator().manual_seed(seed))
    ci = np.asarray(dataset.clip_indices)
    return ci[np.asarray(val.indices)]


def excluded(root, sets):
    """{episode: [starts]} of the windows in the given export sets."""
    ex = {}
    for s in sets:
        with open(Path(root) / s / "key.csv") as f:
            for r in csv.DictReader(f):
                ex.setdefault(int(r["episode"]), []).append(int(r["start"]))
    return ex


def pick(windows, lengths, n, span, seed, exclude):
    rng = np.random.default_rng(seed)
    taken = {e: list(v) for e, v in exclude.items()}
    out = []
    for i in rng.permutation(len(windows)):
        ep, st = map(int, windows[i])
        if st + span > lengths[ep] or any(abs(st - s) < span for s in taken.get(ep, [])):
            continue
        taken.setdefault(ep, []).append(st)
        out.append((ep, st))
        if len(out) == n:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="pusht_LeWM-MSE-dFH-D")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--task", default="pusht")
    p.add_argument("--set", required=True)
    p.add_argument("--root", default=None, help="default: results/halluc/<run>_e<epoch>")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--annotate", action="store_true", help="also PNG frames + annotate.html; shuffled ids")
    p.add_argument("--exclude-sets", nargs="*", default=[])
    p.add_argument("--blocks", type=int, default=5, help="imagined action blocks (planner horizon)")
    p.add_argument("--n-seeds", type=int, default=4, help="flow-head samples for u_f / u_s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--id-prefix", default=None, help="default: <set>_")
    a = p.parse_args()

    import stable_worldmodel as swm

    a.root = a.root or str(set_root(a.run, a.epoch))
    task = TASKS[a.task]
    cols = task.get("cols", ["action"])
    # the training dataset (train.py: num_steps = history 3 + 1 prediction, frameskip 5)
    train_ds = swm.data.load_dataset(str(dataset_path(task["dataset"])), num_steps=4, frameskip=FRAMESKIP,
                                     keys_to_load=["pixels"] + cols, keys_to_cache=cols, transform=None)
    windows = val_windows(train_ds)
    span = (a.blocks + 1) * FRAMESKIP  # frames start + 5k (k = 0..P) and action blocks 0..P
    ex = excluded(a.root, a.exclude_sets)
    chosen = pick(windows, np.asarray(train_ds.lengths), a.n, span, a.seed, ex)
    print(f"[select] {len(chosen)} of {len(windows)} val windows (span {span} steps, excluded "
          f"{sum(map(len, ex.values()))} windows of {a.exclude_sets})", flush=True)
    process = fit_scalers(train_ds, ["action"])  # the planner's action normalisation
    sc = process["action"]

    model, decoder = load_run(a.run, a.epoch, decoder=True)
    assert decoder is not None, f"{a.run} has no decoder"
    mean, std = imagenet_norm()
    order = np.random.default_rng(a.seed + 1).permutation(len(chosen)) if a.annotate else np.arange(len(chosen))
    prefix = a.id_prefix if a.id_prefix is not None else f"{a.set}_"
    writer = SetWriter(a.root, a.set, ctx=1, annotate=a.annotate)
    torch.manual_seed(a.seed)
    for b0 in range(0, len(chosen), a.batch):
        idx = list(range(b0, min(b0 + a.batch, len(chosen))))
        eps = np.array([chosen[i][0] for i in idx])
        sts = np.array([chosen[i][1] for i in idx])
        chunks = train_ds.load_chunk(eps, sts, sts + span)
        px = torch.stack([torch.as_tensor(np.asarray(c["pixels"])) for c in chunks])  # (B, P+1, 3, h, w) uint8
        raw_act = np.stack([np.asarray(c["action"]) for c in chunks])                  # (B, P+1, 5*A)
        B, T = raw_act.shape[:2]
        act = np.nan_to_num(sc.transform(raw_act.reshape(-1, sc.n_features_in_)), nan=0.0)  # train.py: NaN -> 0
        act = torch.from_numpy(act.reshape(B, T, -1).astype(np.float32))
        pxn = ((px.cuda().float() / 255.0) - mean[None]) / std[None]
        out = imagine(model, decoder, pxn[:, :1], act[:, :a.blocks], n_seeds=a.n_seeds, seed=a.seed,
                      real_px=pxn)
        real = px.permute(0, 1, 3, 4, 2).numpy()
        for j, i in enumerate(idx):
            sid = f"{prefix}{order[i] + 1:04d}"
            pred = {k: out[k][j] for k in ("u_r", "u_f", "u_s", "drift", "motion", "oracle_e_z",
                                           "oracle_motion_real")}
            meta = {"episode": int(eps[j]), "start": int(sts[j]), "run": a.run, "epoch": a.epoch,
                    "frameskip": FRAMESKIP, "source": "val_split", "seed": a.seed, "n_seeds": a.n_seeds,
                    "head": "mse"}
            writer.add(sid, a.task, real[j], out["wm"][j], out["recon"][j], act[j].numpy(), out["z"][j],
                       out["z_real"][j], pred, meta,
                       {"idx": int(order[i]), "start": int(sts[j]), "episode": int(eps[j]),
                        "data_dir": "val_split", "pick": "random",
                        "horizon_motion": float(out["oracle_motion_real"][j].mean())})
        print(f"  [export] {idx[-1] + 1}/{len(chosen)}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
