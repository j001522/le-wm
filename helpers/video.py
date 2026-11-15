"""Images and videos: encode frames, decode latents, text overlays, mp4/gif writing."""

import numpy as np
import torch


def imagenet_norm(device="cuda"):
    """(mean, std), shaped (3, 1, 1): the encoder's input normalisation and the decoder's output space."""
    import stable_pretraining as spt

    st = spt.data.dataset_stats.ImageNet
    return (torch.tensor(st["mean"], device=device)[:, None, None],
            torch.tensor(st["std"], device=device)[:, None, None])


def to_uint8(x):
    """Last frame of an env observation as (H, W, 3) uint8 (tolerates a history axis / CHW)."""
    x = np.asarray(x)
    while x.ndim > 3:
        x = x[-1]
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255 if x.max() <= 1.0 else np.clip(x, 0, 255)).astype(np.uint8)
    return x.copy()  # the env reuses its observation buffers: keep a snapshot, not a view


@torch.no_grad()
def encode(model, frames, norm=None, bs=64):
    """uint8 frames (N, H, W, 3) -> latents (N, D) numpy."""
    mean, std = norm or imagenet_norm()
    out = []
    for i in range(0, len(frames), bs):
        px = torch.from_numpy(frames[i : i + bs]).cuda().permute(0, 3, 1, 2).float() / 255.0
        out.append(model.encode({"pixels": ((px - mean) / std)[:, None]})["emb"][:, 0].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def decode(decoder, z, norm=None):
    """Latents (N, D) -> uint8 images (N, H, W, 3) numpy."""
    mean, std = norm or imagenet_norm(z.device)
    img = decoder(z.float())
    return ((img * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def put(img, text, org, scale, color, thick=1):
    """Text with a white halo, readable on any background."""
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def write_video(frames, path, fps, gif=False):
    """frames: list of (H, W, 3) uint8. Writes <path>.mp4 (H.264, yuv420p: plays in browsers,
    Windows and macOS players) and, with gif=True, a half-size looping <path>.gif."""
    import cv2
    import imageio.v2 as imageio

    h, w = frames[0].shape[:2]
    even = [f[: h // 2 * 2, : w // 2 * 2] for f in frames]  # libx264 needs even dimensions
    with imageio.get_writer(path.with_suffix(".mp4"), fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=2, ffmpeg_log_level="error") as wr:
        for f in even:
            wr.append_data(f)
    if gif:
        imageio.mimsave(path.with_suffix(".gif"), [cv2.resize(f, (w // 2, h // 2)) for f in frames],
                        duration=1000 / fps, loop=0)
