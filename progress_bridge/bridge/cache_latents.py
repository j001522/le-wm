"""Step 1a: encode every dataset frame once with the frozen LeWM encoder.

Layout of the output h5 (rows are global frame indices):
    z (N, 192) f2, action (N, A) f4, episode_idx (N,), ep_len (E,), ep_offset (E,)

`--reuse-z` copies latents from an existing cache instead of re-encoding (about
40 min for PushT); it still takes every index from the source dataset.
"""

import argparse
import time

import h5py
import numpy as np
import stable_pretraining as spt
import torch

from bridge.common import TASKS, add_task_arg, dataset_path, latents_path, load_lewm


@torch.no_grad()
def encode_frames(model, frames, mean, std):
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    return model.encode({"pixels": x[:, None]})["emb"][:, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--reuse-z", default=None)
    a = p.parse_args()

    task = TASKS[a.task]
    out = latents_path(task.dataset)
    with h5py.File(dataset_path(task.dataset), "r") as src, h5py.File(out, "w") as dst:
        n = src["pixels"].shape[0]
        for key in ("action", "ep_len", "ep_offset"):
            dst.create_dataset(key, data=src[key][:])
        episode_key = "episode_idx" if "episode_idx" in src else "ep_idx"
        dst.create_dataset("episode_idx", data=src[episode_key][:])
        z = dst.create_dataset("z", shape=(n, 192), dtype="f2", chunks=(4096, 192))

        if a.reuse_z:
            with h5py.File(a.reuse_z, "r") as old:
                assert old["z"].shape[0] == n, "cache does not match dataset"
                for i in range(0, n, 65536):
                    z[i : i + 65536] = old["z"][i : i + 65536]
        else:
            model = load_lewm(task.ckpt)
            stats = spt.data.dataset_stats.ImageNet
            mean = torch.tensor(stats["mean"]).cuda()[None, :, None, None]
            std = torch.tensor(stats["std"]).cuda()[None, :, None, None]
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
