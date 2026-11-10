"""Optional decoder z -> 96x96 image, used only to look at latents.

The LeWM checkpoint has no decoder. This one is trained afterwards on cached
latents with the frozen encoder untouched, so it has no effect on planning. It is
a regression: on off-manifold latents it shows a plausible blend rather than a
real state, so read it next to the nearest dataset frame.
"""

import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bridge.common import TASKS, add_task_arg, dataset_path, latents_path

SIZE = 96


class Decoder(nn.Module):
    def __init__(self, z_dim=192, ch=32):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(z_dim, 8 * ch * 6 * 6)

        def up(a, b):
            return nn.Sequential(
                nn.Upsample(scale_factor=2), nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
            )

        self.net = nn.Sequential(
            up(8 * ch, 4 * ch), up(4 * ch, 2 * ch), up(2 * ch, ch), up(ch, ch), nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 8 * self.ch, 6, 6))


def load_frames(path, allowed, n_chunks, rng, stride=5):
    """Sample frames from random HDF5 chunks where `allowed` holds. Returns rows and uint8 images.

    Pixels are compressed in chunks, so a scattered read decodes a whole chunk per
    frame. Reading whole chunks and keeping every stride-th frame avoids that.
    """
    with h5py.File(path, "r") as f:
        size = f["pixels"].chunks[0]
        candidates = np.unique(np.flatnonzero(allowed) // size)
        chosen = np.sort(rng.choice(candidates, min(n_chunks, len(candidates)), replace=False))
        rows, out = [], []
        for c in chosen:
            lo = int(c) * size
            pick = np.flatnonzero(allowed[lo : lo + size])[::stride]
            if len(pick) == 0:
                continue
            x = torch.from_numpy(f["pixels"][lo : lo + size][pick]).cuda().permute(0, 3, 1, 2).float()
            rows.append(lo + pick)
            out.append(F.interpolate(x, size=SIZE, mode="area").round().byte())
    return np.concatenate(rows), torch.cat(out)


def load_decoder(path):
    d = Decoder().cuda().eval().requires_grad_(False)
    d.load_state_dict(torch.load(path, map_location="cpu"))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, default=1500, help="HDF5 chunks to sample; about 20 frames each")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-episodes", type=int, default=500)
    add_task_arg(p)
    a = p.parse_args()
    task = TASKS[a.task]
    out = task.artifacts / "decoder.pt"

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    with h5py.File(latents_path(task.dataset), "r") as f:
        ep, z_all = f["episode_idx"][:], f["z"][:]
    held = rng.choice(ep.max() + 1, a.val_episodes, replace=False)
    is_val = np.isin(ep, held)

    t0 = time.time()
    tr, x_tr = load_frames(dataset_path(task.dataset), ~is_val, a.chunks, rng)
    va, x_va = load_frames(dataset_path(task.dataset), is_val, 100, rng)
    z_tr, z_va = (torch.from_numpy(z_all[r].astype(np.float32)).cuda() for r in (tr, va))
    del z_all
    print(f"loaded {len(tr)} + {len(va)} frames in {time.time() - t0:.0f}s", flush=True)

    model = Decoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for step in range(1, a.steps + 1):
        idx = torch.randint(len(z_tr), (a.batch,), device="cuda")
        loss = F.l1_loss(model(z_tr[idx]), x_tr[idx].float() / 255)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                val = np.mean([
                    F.l1_loss(model(z), x.float() / 255).item()
                    for z, x in zip(z_va.split(256), x_va.split(256))
                ])
            model.train()
            print(f"step {step} train L1 {loss.item():.4f} | unseen-episode L1 {val:.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            torch.save(model.state_dict(), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
