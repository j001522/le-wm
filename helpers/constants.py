"""Paths, frame skip and per-task settings shared by every script."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))  # datasets/, checkpoints/
RESULTS = ROOT / "results"            # every script's outputs (gitignored)
STAGE1 = SWM / "checkpoints" / "stage1"  # stage-1 dynamics checkpoints (cached-latent prototype)
FRAMESKIP = 5
TASKS = {
    # latents: released-encoder latent cache; env / cols / state_key: planning in the simulator
    "pusht": dict(ckpt="quentinll/lewm-pusht", dataset="pusht_expert_train",
                  latents="pusht_expert_train_latents_v2.h5",
                  env="swm/PushT-v1", cols=["action", "proprio", "state"], state_key="state"),
    "tworoom": dict(ckpt="quentinll/lewm-tworooms", dataset="tworoom", latents="tworoom_latents_v2.h5"),
}


def run_dir(experiment, run=None, epoch=None):
    """results/<experiment>[/<run>_e<epoch>], created."""
    d = RESULTS / experiment
    if run is not None:
        d = d / (f"{run}_e{epoch}" if epoch is not None else run)
    d.mkdir(parents=True, exist_ok=True)
    return d
