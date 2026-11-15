"""Latent caches: every dataset frame encoded once by a trained run (scripts/cache_latents.py).

Cache h5: z (N, 192) f16 for every raw frame, action (N, A), episode_idx, ep_len, ep_offset.
"""

from helpers.constants import SWM


def latents_path(dataset, run, epoch):
    """Latent cache of a trained run."""
    return SWM / "datasets" / f"{dataset}_latents_{run}_e{epoch}.h5"
