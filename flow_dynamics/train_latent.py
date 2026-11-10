"""Stage 1: train the dynamics on cached frozen-encoder latents.

python -m flow_dynamics.train_latent --variant head --steps 20000 --out head.pt
variant none = MSE predictor (fine-tuned control), head = B, transformer = A.
"""

import argparse
import math
import time

import numpy as np
import torch

from flow_dynamics.common import ARTIFACTS, LatentWindows, build_dynamics, save_model

HS = 3  # history size = context tokens


def losses(model, z, a, a_mse_weight, flow_kw):
    act_emb = model.action_encoder(a)
    tgt = z[:, 1:]
    if not model.is_flow:
        mse = (model.predict(z[:, :HS], act_emb[:, :HS]) - tgt).pow(2).mean()
        return {"loss": mse, "pred_loss": mse}
    fl = model.flow_loss(z, act_emb, HS, 1, **flow_kw)
    out = {"flow_loss": fl["flow_loss"], "tau": fl["tau"], "err": fl["err"]}
    if fl["mse_pred"] is not None:
        out["pred_loss"] = (fl["mse_pred"] - tgt).pow(2).mean()
        out["loss"] = fl["flow_loss"] + a_mse_weight * out["pred_loss"]
    else:
        out["loss"] = fl["flow_loss"]
    return out


@torch.no_grad()
def validate(model, data, flow_kw, n=8192, mse_weight=1.0, action_noise=0.0):
    model.eval()
    g = torch.Generator("cuda").manual_seed(123)
    z, a = data.sample("val", n, HS + 1, g)
    a = a + action_noise * torch.randn(a.shape, device=a.device, generator=g)
    torch.manual_seed(0)
    out = losses(model, z, a, mse_weight, flow_kw)
    res = {k: float(v) for k, v in out.items() if k.endswith("loss")}
    if "tau" in out:  # flow loss per tau bin
        tau, err = out["tau"].flatten(), out["err"].flatten()
        edges = torch.tensor([0, .1, .3, .5, .7, .9, .97, 1.0], device=tau.device)
        b = torch.bucketize(tau, edges[1:-1])
        res["by_tau"] = [float(err[b == i].mean()) for i in range(len(edges) - 1)]
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pusht")
    p.add_argument("--variant", default="head", choices=["none", "head", "transformer"])
    p.add_argument("--init", default="pretrained", choices=["pretrained", "none"])
    p.add_argument("--freeze-trunk", action="store_true", help="train only the flow head")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4, help="new modules")
    p.add_argument("--trunk-lr", type=float, default=3e-5, help="pretrained modules")
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--taus-per-token", type=int, default=4)
    p.add_argument("--tau-std", type=float, default=1.0)
    p.add_argument("--tau-weight", action="store_true")
    p.add_argument("--ctx-tau", type=float, default=0.9, help="variant A: signal level of the history at inference")
    p.add_argument("--ctx-frac", type=float, default=0.0, help="variant A: fraction of windows with history at ctx_tau")
    p.add_argument("--head-width", type=int, default=512)
    p.add_argument("--head-depth", type=int, default=4)
    p.add_argument("--action-noise", type=float, default=0.0,
                   help="std of Gaussian noise on the actions the model sees (normalised units); "
                        "latents still follow the real actions, so the next latent becomes stochastic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    data = LatentWindows(a.task)
    model = build_dynamics(a.variant, a.task, a.init, a.head_width, a.head_depth, ctx_tau=a.ctx_tau).cuda().train()
    new = ("flow_head.", "predictor.t.", "predictor.out.")
    groups = {"new": [], "old": []}
    for k, v in model.named_parameters():
        is_new = k.startswith(new) or a.init == "none"
        if a.freeze_trunk and not is_new:
            v.requires_grad_(False)
            continue
        groups["new" if is_new else "old"].append(v)
    if a.freeze_trunk:
        model.predictor.eval(), model.action_encoder.eval()  # no dropout in the frozen trunk
    opt = torch.optim.AdamW([
        {"params": groups["new"], "lr": a.lr},
        {"params": groups["old"], "lr": a.trunk_lr},
    ], weight_decay=1e-3)
    base = [g["lr"] for g in opt.param_groups]
    warm = min(1000, a.steps // 10)
    flow_kw = dict(taus_per_token=a.taus_per_token, tau_std=a.tau_std, tau_weight=a.tau_weight, ctx_frac=a.ctx_frac)
    out = ARTIFACTS / a.out
    meta = {**vars(a), "history_size": HS}
    print(f"{a.variant}: {sum(v.numel() for v in groups['new'])/1e6:.2f}M new, "
          f"{sum(v.numel() for v in groups['old'])/1e6:.2f}M pretrained trainable params", flush=True)
    print("val at init", validate(model, data, flow_kw, mse_weight=a.mse_weight, action_noise=a.action_noise), flush=True)

    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        f = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (a.steps - warm)))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * f
        z, act = data.sample("train", a.batch_size, HS + 1)
        act = act + a.action_noise * torch.randn_like(act)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = losses(model, z, act, a.mse_weight, flow_kw)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log.append([float(o.get("flow_loss", torch.zeros(())).detach()), float(o.get("pred_loss", torch.zeros(())).detach())])
        if step % 1000 == 0 or step == a.steps:
            fl, pl = np.mean(log, 0)
            log = []
            v = validate(model, data, flow_kw, mse_weight=a.mse_weight, action_noise=a.action_noise)
            print(f"step {step} train flow {fl:.4f} mse {pl:.4f} | val {v} | {time.time() - t0:.0f}s", flush=True)
            save_model(model, out, {**meta, "step": step, "val": v})
    print("saved", out)


if __name__ == "__main__":
    main()
