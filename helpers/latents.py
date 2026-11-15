"""Cached latents: every dataset frame encoded once (scripts/cache_latents.py).

Cache h5: z (N, 192) f16 for every raw frame, action (N, A), episode_idx, ep_len, ep_offset.
A predictor step spans FRAMESKIP raw frames and its action is the row-major flattening of the
FRAMESKIP raw actions (as in swm datasets).
"""

import h5py
import numpy as np
import torch

from helpers.constants import FRAMESKIP, SWM, TASKS


def latents_path(dataset, run=None, epoch=None):
    """Latent cache of the released encoder (run=None) or of a trained run."""
    if run is None:
        task = next(t for t in TASKS.values() if t["dataset"] == dataset)
        return SWM / "datasets" / task["latents"]
    return SWM / "datasets" / f"{dataset}_latents_{run}_e{epoch}.h5"


class LatentWindows:
    """All cached latents on the GPU; samples windows of L predictor steps.
    Episodes are split once into train/val by a fixed permutation."""

    def __init__(self, task="pusht", device="cuda", val_frac=0.05, seed=0):
        with h5py.File(SWM / "datasets" / TASKS[task]["latents"], "r") as f:
            z = f["z"][:]
            act = f["action"][:]
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_off = f["ep_offset"][:].astype(np.int64)
        ok = ~np.isnan(act).any(1)
        self.act_mean, self.act_std = act[ok].mean(0), act[ok].std(0)  # utils.get_column_normalizer
        act = np.nan_to_num((act - self.act_mean) / self.act_std, nan=0.0)
        self.device = device
        self.z = torch.from_numpy(z).to(device)  # f16
        self.act = torch.from_numpy(act.astype(np.float32)).to(device)
        perm = np.random.default_rng(seed).permutation(len(self.ep_len))
        n_val = int(round(val_frac * len(perm)))
        self.split = {"val": np.sort(perm[:n_val]), "train": np.sort(perm[n_val:])}
        self._starts = {}

    @property
    def dim(self):
        return self.z.size(1)

    def starts(self, split, L):
        """Global start frames of every window of L steps inside one episode."""
        key = (split, L)
        if key not in self._starts:
            span = (L - 1) * FRAMESKIP
            eps = self.split[split]
            n = np.maximum(self.ep_len[eps] - span, 0)
            s = np.concatenate([o + np.arange(k) for o, k in zip(self.ep_off[eps], n)])
            self._starts[key] = torch.from_numpy(s).to(self.device)
        return self._starts[key]

    def gather(self, starts, L):
        """starts (B,) -> z (B, L, D) f32, action chunks (B, L, FRAMESKIP*A).
        The last chunk may run past the episode end (it is never used as a condition
        for a target inside the window)."""
        k = torch.arange(L, device=self.device) * FRAMESKIP
        zi = starts[:, None] + k
        ai = (zi[..., None] + torch.arange(FRAMESKIP, device=self.device)).clamp(max=len(self.act) - 1)
        return self.z[zi].float(), self.act[ai].flatten(2)

    def sample(self, split, B, L, generator=None):
        s = self.starts(split, L)
        idx = torch.randint(len(s), (B,), device=self.device, generator=generator)
        return self.gather(s[idx], L)
