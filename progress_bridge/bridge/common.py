"""Shared helpers: artifact paths, frozen LeWM loading, one-step dynamics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "save" / "progress_bridge"  # ROOT = progress_bridge/
CKPT = "quentinll/lewm-pusht"
DATASET = "pusht_expert_train"
FRAMESKIP = 5  # env steps per predictor step (LeWM's training frameskip)


@dataclass(frozen=True)
class Task:
    ckpt: str
    dataset: str
    env: str
    columns: tuple  # dataset columns the policy normalises
    state_key: str  # column holding the env state
    tracked: tuple  # state slice whose distance to the goal is reported
    artifacts: Path
    goal_offset: int
    budget: int


TASKS = {
    "pusht": Task(
        CKPT, DATASET, "swm/PushT-v1", ("action", "proprio", "state"), "state", (2, 4),
        ARTIFACTS, goal_offset=100, budget=200,
    ),
    "tworoom": Task(
        "quentinll/lewm-tworooms", "tworoom", "swm/TwoRoom-v1", ("action", "proprio"), "proprio", (0, 2),
        ARTIFACTS / "tworoom", goal_offset=75, budget=150,
    ),
}


def add_task_arg(parser):
    parser.add_argument("--task", choices=sorted(TASKS), default="pusht")


def dataset_path(name=DATASET) -> Path:
    return Path(swm.data.utils.get_cache_dir()) / "datasets" / f"{name}.h5"


def latents_path(name=DATASET) -> Path:
    return dataset_path(name).with_name(f"{name}_latents_v2.h5")


def load_lewm(ckpt=CKPT, device="cuda"):
    model = swm.wm.utils.load_pretrained(ckpt)
    return model.to(device).eval().requires_grad_(False)


def predict_one(model, z, a):
    """P(z, a): z (B, D), a (B, 10) normalised action chunk -> next z (B, D)."""
    return model.predict(z[:, None], model.action_encoder(a[:, None]))[:, -1]


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
