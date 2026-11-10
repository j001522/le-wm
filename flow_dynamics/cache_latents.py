"""Encode every dataset frame once with the frozen released LeWM encoder.

python -m flow_dynamics.cache_latents --task pusht   (about 40 min for PushT)

Output h5 at <swm cache>/datasets/<latents name>, rows = global frame indices:
    z (N, 192) f2, action (N, A) f4, episode_idx (N,), ep_len (E,), ep_offset (E,)
"""

import argparse
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter used for pixels)
import numpy as np
import stable_pretraining as spt
import torch

from flow_dynamics.common import SWM, TASKS, full_model


@torch.no_grad()
def encode_frames(model, frames, mean, std):
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    return model.encode({"pixels": x[:, None]})["emb"][:, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pusht", choices=sorted(TASKS))
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()

    task = TASKS[a.task]
    src_path = SWM / "datasets" / f"{task['dataset']}.h5"
    out = SWM / "datasets" / task["latents"]
    model = full_model("pretrained", a.task)
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
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
