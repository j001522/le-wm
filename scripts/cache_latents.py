"""Encode every dataset frame once, with the released LeWM encoder or an end-to-end checkpoint.

python -m scripts.cache_latents --task pusht                              (released encoder)
python -m scripts.cache_latents --task pusht --run lewm_flow_e2e --epoch 14 (any trained run)
About 40 min for PushT on an RTX 4060 laptop.

Output h5 at <swm cache>/datasets/<latents name> (released) or
<dataset>_latents_<run>_e<epoch>.h5 (trained run), rows = global frame indices:
    z (N, 192) f2, action (N, A) f4, episode_idx (N,), ep_len (E,), ep_offset (E,)
"""

import argparse
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter used for pixels)
import numpy as np
import stable_pretraining as spt
import torch

import stable_worldmodel as swm

from helpers.constants import SWM, TASKS
from helpers.latents import latents_path
from helpers.models import full_model


@torch.no_grad()
def encode_frames(model, frames, mean, std):
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    return model.encode({"pixels": x[:, None]})["emb"][:, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pusht", choices=sorted(TASKS))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--run", default=None, help="checkpoint run name under <swm cache>/checkpoints (default: released)")
    p.add_argument("--epoch", type=int, default=None)
    a = p.parse_args()

    task = TASKS[a.task]
    src_path = SWM / "datasets" / f"{task['dataset']}.h5"
    if a.run is None:
        out = SWM / "datasets" / task["latents"]
        model = full_model("pretrained", a.task)
    else:
        assert a.epoch is not None, "--epoch is required with --run"
        out = latents_path(task["dataset"], a.run, a.epoch)
        model = swm.wm.utils.load_pretrained(f"{a.run}/weights_epoch_{a.epoch}.pt").cuda().eval().requires_grad_(False)
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"]).cuda()[None, :, None, None]
    std = torch.tensor(stats["std"]).cuda()[None, :, None, None]
    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        n = src["pixels"].shape[0]
        for key in ("action", "ep_len", "ep_offset"):
            dst.create_dataset(key, data=src[key][:])
        episode_key = "episode_idx" if "episode_idx" in src else "ep_idx"
        dst.create_dataset("episode_idx", data=src[episode_key][:])
        z = dst.create_dataset("z", shape=(n, 192), dtype="f2", chunks=(4096, 192))
        t0 = time.time()
        for i in range(0, n, a.batch_size):
            z[i : i + a.batch_size] = encode_frames(
                model, src["pixels"][i : i + a.batch_size], mean, std
            ).astype(np.float16)
            if i % (a.batch_size * 200) == 0:
                print(f"{i}/{n} ({time.time() - t0:.0f}s)", flush=True)
        dst.attrs["encoder"] = "released" if a.run is None else f"{a.run}/weights_epoch_{a.epoch}.pt"
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
