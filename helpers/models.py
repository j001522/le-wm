"""Model loading for trained runs (weights + detached visualisation decoder)."""

import torch

from helpers.constants import SWM


def load_run(run, epoch, decoder=True, device="cuda"):
    """Trained run under <swm>/checkpoints/<run>: (model, decoder or None).
    decoder=True loads decoder_epoch_<epoch>.pt if the run has one."""
    import stable_worldmodel as swm

    import module

    model = swm.wm.utils.load_pretrained(f"{run}/weights_epoch_{epoch}.pt").to(device).eval().requires_grad_(False)
    path = SWM / "checkpoints" / run / f"decoder_epoch_{epoch}.pt"
    dec = None
    if decoder and path.exists():
        ck = torch.load(path, map_location="cpu")
        dec = module.PixelDecoder(**{k: v for k, v in ck["cfg"].items() if k != "_target_"})
        dec.load_state_dict(ck["state_dict"])
        dec = dec.to(device).eval().requires_grad_(False)
    return model, dec
