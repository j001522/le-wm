"""Paths, frame skip and per-task settings shared by every script."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))  # datasets/, checkpoints/
RESULTS = ROOT / "results"  # every script's outputs (gitignored)
FRAMESKIP = 5
TASKS = {
    # env / cols / state_key: planning in the simulator (scripts/eval_plan.py)
    "pusht": dict(dataset="pusht_expert_train", env="swm/PushT-v1", cols=["action", "proprio", "state"],
                  state_key="state"),
    "tworoom": dict(dataset="tworoom"),
}


def run_dir(experiment, run=None, epoch=None):
    """results/<experiment>[/<run>_e<epoch>], created."""
    d = RESULTS / experiment
    if run is not None:
        d = d / (f"{run}_e{epoch}" if epoch is not None else run)
    d.mkdir(parents=True, exist_ok=True)
    return d
