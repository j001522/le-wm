"""Decode real and predicted latents of an end-to-end model with its decoder.

python -m flow_dynamics.visualize_e2e --run lewm_flow_e2e --epoch 14 --episodes 3

For each episode, one figure with rows: real frames, reconstruction of their latents,
flow rollout (one sample), MSE-head rollout. The rollout starts from 3 real frames and
uses the real actions. Images go to flow_dynamics/save/viz/.
"""

import argparse

import h5py
import hdf5plugin  # noqa: F401  (pixel compression filter)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

import module
from flow_dynamics.common import ARTIFACTS, FRAMESKIP, SWM

HS = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lewm_flow_e2e")
    p.add_argument("--epoch", type=int, default=14)
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=10, help="predictor steps to roll out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ck = SWM / "checkpoints" / a.run
    model = swm.wm.utils.load_pretrained(f"{a.run}/weights_epoch_{a.epoch}.pt").cuda().eval().requires_grad_(False)
    dck = torch.load(ck / f"decoder_epoch_{a.epoch}.pt", map_location="cpu")
    cfg = {k: v for k, v in dck["cfg"].items() if k != "_target_"}
    decoder = module.PixelDecoder(**cfg).cuda().eval()
    decoder.load_state_dict(dck["state_dict"])

    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"]).cuda()[:, None, None]
    std = torch.tensor(stats["std"]).cuda()[:, None, None]
    T = HS + a.steps
    out_dir = ARTIFACTS / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(SWM / "datasets" / f"{a.dataset}.h5", "r") as f:
        act_all = f["action"]
        ok = ~np.isnan(act_all[:500000]).any(1)
        a_mean, a_std = act_all[:500000][ok].mean(0), act_all[:500000][ok].std(0)
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        rng = np.random.default_rng(a.seed)
        long_eps = np.flatnonzero(ep_len > T * FRAMESKIP + 1)
        for ep in rng.choice(long_eps, a.episodes, replace=False):
            start = ep_off[ep] + rng.integers(0, ep_len[ep] - T * FRAMESKIP)
            idx = start + FRAMESKIP * np.arange(T)
            px = torch.from_numpy(f["pixels"][start : start + T * FRAMESKIP : FRAMESKIP]).cuda()
            px = ((px.permute(0, 3, 1, 2).float() / 255.0) - mean) / std  # (T, 3, 224, 224)
            act = np.nan_to_num((f["action"][start : start + T * FRAMESKIP] - a_mean) / a_std)
            act = torch.from_numpy(act.reshape(T, -1).astype(np.float32)).cuda()  # (T, 10)

            with torch.no_grad():
                z = model.encode({"pixels": px[None]})["emb"][0]  # (T, D)
                rows = {"real": px, "reconstruction": decoder(z)}
                for name, variant in (("flow sample", model.flow_variant), ("MSE head", "none")):
                    model.flow_variant = variant
                    torch.manual_seed(a.seed)
                    eps = torch.randn(1, a.steps, z.size(-1), device="cuda")
                    roll, _ = model.rollout_latents(z[None, :HS], act[None, :HS], act[None, HS : T - 1], HS, eps)
                    pred = roll[0, HS:]
                    rows[name] = torch.cat([decoder(z[:HS]), decoder(pred)])
                    rows[name + " err"] = (pred - z[HS:]).pow(2).mean(-1)
                model.flow_variant = "head"

            img_rows = [k for k in rows if not k.endswith("err")]
            fig, axes = plt.subplots(len(img_rows), T, figsize=(1.6 * T, 1.75 * len(img_rows)))
            for r, name in enumerate(img_rows):
                ims = (rows[name] * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                for t in range(T):
                    ax = axes[r, t]
                    ax.imshow(ims[t])
                    ax.set_xticks([]), ax.set_yticks([])
                    if t == 0:
                        ax.set_ylabel(name, fontsize=8)
                    if r == 0:
                        ax.set_title("context" if t < HS else f"+{t - HS + 1}", fontsize=8)
                    if t >= HS and name + " err" in rows:
                        ax.set_xlabel(f"mse {rows[name + ' err'][t - HS]:.3f}", fontsize=6)
            fig.suptitle(f"{a.run} epoch {a.epoch}, episode {ep}, frames {idx[0]}..{idx[-1]}", fontsize=9)
            fig.tight_layout()
            path = out_dir / f"{a.run}_e{a.epoch}_ep{ep}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print("wrote", path)


if __name__ == "__main__":
    main()
