"""Planning setup shared with the root eval.py: dataset, normalisation, transform, tasks, CEM."""

from pathlib import Path

import numpy as np
import torch

from helpers.constants import SWM


def dataset_path(name):
    """Dataset name or path -> path. A bare name (no extension) resolves to <swm>/datasets/<name>.lance
    if it exists (Snellius), else <name>.h5 (local)."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    p = SWM / "datasets" / name
    if p.suffix in (".lance", ".h5"):
        return p
    for ext in (".lance", ".h5"):
        if p.with_suffix(ext).exists():
            return p.with_suffix(ext)
    raise FileNotFoundError(f"no {name}.lance or {name}.h5 in {SWM / 'datasets'}")


def load_planning_dataset(name, cols):
    """swm dataset (h5 or lance) with only the low-dimensional columns cached."""
    import stable_worldmodel as swm

    return swm.data.load_dataset(str(dataset_path(name)), keys_to_cache=cols)


def fit_scalers(dataset, cols):
    """Per-column StandardScaler, as in eval.py (goal_<col> shares <col>'s scaler)."""
    from sklearn import preprocessing

    process = {}
    for col in cols:
        v = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def image_transform(size=224):
    """Observation transform of eval.py: uint8 -> float, ImageNet normalisation, resize."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size),
    ])


def pick_tasks(dataset, n, goal_offset, seed):
    """Start (episode, step) pairs, same procedure and seed behaviour as eval.py.
    Works for h5 and lance (whose column_names omit the episode_idx index column)."""
    try:
        ep = dataset.get_col_data("episode_idx")
    except Exception:
        ep = dataset.get_col_data("ep_idx")
    step = dataset.get_col_data("step_idx")
    max_step = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(max_step, ep, step)
    valid = np.nonzero(step <= max_step[ep] - goal_offset)[0]  # eval.py: len - offset - 1
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, size=n, replace=False)])
    return ep[rows].tolist(), step[rows].tolist()


def cem_solver(model, num_samples=300, seed=42):
    """CEM with the settings of config/eval/solver/cem.yaml."""
    from stable_worldmodel.solver import CEMSolver

    return CEMSolver(model=model, batch_size=1, num_samples=num_samples, var_scale=1.0,
                     n_steps=30, topk=30, device="cuda", seed=seed)
