"""Stage 1 evaluation on held-out episodes: k-step latent rollout error, uncertainty
proxy vs ensemble spread, sample statistics, and rollout cost.

python -m flow_dynamics.eval_rollout --models pretrained mse.pt head.pt
'pretrained' is the released LeWM predictor. Each flow model is scored as:
  sample  one Euler sample (sample_steps), common noise
  mean    one step from tau=0 = x̂(tau=0), the conditional-mean predictor
  ens     mean of --ens samples
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from flow_dynamics.common import ARTIFACTS, LatentWindows, build_dynamics, load_model
from module import SIGReg

HS = 3
KS = (1, 5, 10, 20)


def rollout(model, z, a, n_pred, steps=None, eps=None):
    """History z[:, :HS] with actions a[:, :HS], future actions after that. Returns
    predictions (N, n_pred, D) for z[:, HS : HS+n_pred] and drift (N, n_pred) or None."""
    out, drift = model.rollout_latents(z[:, :HS], a[:, :HS], a[:, HS : HS + n_pred - 1], HS, eps, steps)
    return out[:, HS:], drift


def errs(pred, z):
    e = (pred - z[:, HS : HS + pred.size(1)]).pow(2).mean(-1)  # (N, n)
    return {k: float(e[:, k - 1].mean()) for k in KS}


@torch.no_grad()
def evaluate(model, z, a, ens=16, chunk=512, seed=0):
    n = max(KS)
    N, D = z.size(0), z.size(-1)
    if not model.is_flow:
        pred = torch.cat([rollout(model, z[i : i + chunk], a[i : i + chunk], n)[0] for i in range(0, N, chunk)])
        return {"det": errs(pred, z)}, pred
    g = torch.Generator("cuda").manual_seed(seed)
    eps = torch.randn(ens, N, n, D, device="cuda", generator=g)
    samples, drifts = [], []
    for e in range(ens):
        torch.manual_seed(seed + e)  # variant A context noise
        p, d = zip(*[rollout(model, z[i : i + chunk], a[i : i + chunk], n, eps=eps[e, i : i + chunk])
                     for i in range(0, N, chunk)])
        samples.append(torch.cat(p)), drifts.append(torch.cat(d))
    S = torch.stack(samples)  # (E, N, n, D)
    mean_pred = torch.cat([rollout(model, z[i : i + chunk], a[i : i + chunk], n, steps=1, eps=eps[0, i : i + chunk])[0]
                           for i in range(0, N, chunk)])
    spread = S.std(0).pow(2).mean(-1).sqrt()  # (N, n) per-dim RMS ensemble std
    drift = drifts[0]
    res = {}
    if model.flow_variant == "head":  # same trunk, its MSE head (pred_proj)
        model.flow_variant = "none"
        res["mse_head"] = evaluate(model, z, a, ens, chunk, seed)[0]["det"]
        model.flow_variant = "head"
    res.update({
        "sample": errs(S[0], z),
        "mean": errs(mean_pred, z),
        "ens": errs(S.mean(0), z),
        "spread": {k: float(spread[:, k - 1].mean()) for k in KS},
        "drift": {k: float(drift[:, k - 1].mean()) for k in KS},
        "calib_err_over_var": {k: float((S.mean(0) - z[:, HS:HS + n]).pow(2).mean(-1)[:, k - 1].mean()
                                        / spread[:, k - 1].pow(2).mean()) for k in KS},
        "spearman_drift_spread": {k: float(spearmanr(drift[:, k - 1].cpu(), spread[:, k - 1].cpu())[0]) for k in KS},
        "spearman_spread_err": {k: float(spearmanr(spread[:, k - 1].cpu(),
                                (S.mean(0) - z[:, HS:HS + n]).pow(2).mean(-1)[:, k - 1].cpu())[0]) for k in KS},
    })
    return res, S[0]


@torch.no_grad()
def sigreg_stat(x, reps=8):
    torch.manual_seed(0)
    sr = SIGReg().cuda()
    return float(np.mean([float(sr(x[None])) for _ in range(reps)]))


@torch.no_grad()
def plan_time(model, N=300, n=5, reps=5):
    """Seconds for one rollout of N candidates over n predictor steps (one CEM iteration's model cost)."""
    z = torch.randn(N, HS + n, model.action_encoder.embed[-1].out_features, device="cuda")
    a = torch.randn(N, HS + n, model.action_encoder.patch_embed.in_channels, device="cuda")
    eps = torch.randn(N, n, z.size(-1), device="cuda")
    rollout(model, z, a, n, eps=eps)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        rollout(model, z, a, n, eps=eps)
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["pretrained", "mse.pt", "head.pt"])
    p.add_argument("--task", default="pusht")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--ens", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=None, help="override Euler steps")
    p.add_argument("--action-noise", type=float, default=0.0, help="noise on the actions the models see")
    p.add_argument("--out", default="eval_rollout.json")
    a = p.parse_args()

    data = LatentWindows(a.task)
    g = torch.Generator("cuda").manual_seed(7)
    z, act = data.sample("val", a.n, HS + max(KS), g)
    act = act + a.action_noise * torch.randn(act.shape, device=act.device, generator=g)
    real = sigreg_stat(z[:, HS])
    print(f"{a.n} held-out windows; SIGReg stat on real latents {real:.3f}; copy-last err "
          f"{errs(z[:, HS - 1 : HS].expand(-1, max(KS), -1), z)}")
    results = {"real_sigreg": real}
    for name in a.models:
        model = (build_dynamics("none", a.task).cuda().eval() if name == "pretrained"
                 else load_model(ARTIFACTS / name)[0])
        if a.sample_steps:
            model.sample_steps = a.sample_steps
        res, one = evaluate(model, z, act, a.ens)
        res["sigreg_k1"] = sigreg_stat(one[:, 0])
        res["sigreg_k20"] = sigreg_stat(one[:, -1])
        res["plan_time_s"] = plan_time(model)
        results[name] = res
        print(f"\n== {name}")
        for k, v in res.items():
            print(f"  {k:22s} " + (", ".join(f"{kk}: {vv:.4f}" for kk, vv in v.items()) if isinstance(v, dict) else f"{v:.4f}"))
    (ARTIFACTS / a.out).write_text(json.dumps(results, indent=1))
    print("wrote", ARTIFACTS / a.out)


if __name__ == "__main__":
    main()
