"""Step 4: train the bridge with flow matching on teacher paths.

    gamma = sigma * tau * (1 - tau)
    x     = I(tau) + gamma * eps
    v*    = dI/dtau + sigma * (1 - 2*tau) * eps
    eta*  = eps
    v     = v_net * v_scale + sigma * (1 - 2*tau) * eta.detach()
    loss  = MSE(v / v_scale, v* / v_scale) + MSE(eta, eta*)

v_scale is the rms of v*. Without it, the velocity term is much larger than the
noise term and the noise head barely trains. See BridgeDiT for why v uses eta.
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from bridge.common import FRAMESKIP, TASKS, add_task_arg
from bridge.model import BridgeDiT


class TeacherPaths:
    """Random sub-segments of teacher paths, as curves I(tau) in latent space.

    A sub-segment of a shortest path is itself a shortest path, so this gives
    starts beyond the Dijkstra sources without any extra graph search.
    """

    def __init__(self, z, nodes, ptr, cum, device="cuda"):
        self.z = torch.as_tensor(z, device=device)
        self.nodes = torch.as_tensor(nodes, device=device).long()
        self.ptr = torch.as_tensor(ptr, device=device)
        self.cum = torch.as_tensor(cum, device=device).double()
        self.device = device
        # Per-path ascending costs, offset by path id, give one sorted array to search.
        lengths = self.ptr[1:] - self.ptr[:-1]
        path_of = torch.repeat_interleave(torch.arange(len(lengths), device=device), lengths)
        self.offset = 1e7
        self.sorted_cum = self.cum + path_of.double() * self.offset
        self.n_paths = len(lengths)
        self.lengths = lengths

    def curve(self, paths, first, last, tau):
        """I(tau) and dI/dtau for the segment first..last of each path (node positions)."""
        i, j = self.ptr[paths] + first, self.ptr[paths] + last
        c = self.cum[i] + tau.double() * (self.cum[j] - self.cum[i])
        k = torch.searchsorted(self.sorted_cum, c + paths.double() * self.offset, right=True) - 1
        k = torch.minimum(torch.maximum(k, i), j - 1)
        za, zb = self.z[self.nodes[k]], self.z[self.nodes[k + 1]]
        span = self.cum[k + 1] - self.cum[k]
        w = ((c - self.cum[k]) / span).float()[:, None]
        slope = ((self.cum[j] - self.cum[i]) / span).float()[:, None]
        return za + w * (zb - za), slope * (zb - za), self.z[self.nodes[i]], self.z[self.nodes[j]]

    def sample(self, batch, subset=None, min_hops=2):
        pool = torch.randint(self.n_paths, (batch,), device=self.device) if subset is None else (
            subset[torch.randint(len(subset), (batch,), device=self.device)]
        )
        n = self.lengths[pool]
        first = (torch.rand(batch, device=self.device) * (n - min_hops).float()).long()
        extra = (torch.rand(batch, device=self.device) * (n - first - min_hops).float()).long()
        last = first + min_hops + extra
        tau = torch.rand(batch, device=self.device)
        return self.curve(pool, first, last, tau) + (tau,)


def targets(I, dI, tau, sigma):
    eps = torch.randn_like(I)
    gamma = (sigma * tau * (1 - tau))[:, None]
    x = I + gamma * eps
    v = dI + (sigma * (1 - 2 * tau))[:, None] * eps
    return x, v, eps


def loss_fn(model, batch, sigma):
    I, dI, z_start, z_goal, tau = batch
    x, v_star, eps = targets(I, dI, tau, sigma)
    v_net, eta = model.forward_raw(x, tau, z_start, z_goal)
    v = v_net + model.noise_coefficient(tau) * eta.detach() / model.v_scale
    lv, le = F.mse_loss(v, v_star / model.v_scale), F.mse_loss(eta, eps)
    return lv + le, lv.detach(), le.detach()


def planning_radius(z, episodes, hops):
    """Median latent distance covered by `hops` graph steps along real episodes."""
    same = episodes[:-hops] == episodes[hops:]
    return float(np.median(np.linalg.norm(z[:-hops][same] - z[hops:][same], axis=1)))


def main():
    p = argparse.ArgumentParser()
    add_task_arg(p)
    p.add_argument("--out", default=None, help="defaults to bridge.pt in the task's artifacts")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sigma", type=float, default=4.0, help="noise norm at tau=0.5 is sigma/4*sqrt(192)")
    p.add_argument("--cycle-env-steps", type=int, default=25, help="env steps LeWM plans over per cycle")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    task = TASKS[a.task]
    a.out = a.out or str(task.artifacts / "bridge.pt")
    g, t = np.load(task.artifacts / "graph.npz"), np.load(task.artifacts / "paths.npz")
    data = TeacherPaths(g["z"], t["nodes"], t["ptr"], t["cum"])

    # Hold out every path of 5% of the Dijkstra sources.
    rng = np.random.default_rng(a.seed)
    sources = np.unique(t["source"])
    held = rng.choice(sources, max(1, len(sources) // 20), replace=False)
    is_val = torch.as_tensor(np.isin(t["source"], held), device="cuda")
    tr, va = torch.nonzero(~is_val)[:, 0], torch.nonzero(is_val)[:, 0]

    stride = int(g["stride"])
    r = planning_radius(g["z"], g["episodes"], a.cycle_env_steps // stride)
    I, dI, _, _, tau = data.sample(8192, tr)
    _, v, _ = targets(I, dI, tau, a.sigma)
    v_scale = float(v.pow(2).mean().sqrt())
    print(f"paths train {len(tr)}, val {len(va)} | r {r:.2f} | sigma {a.sigma} "
          f"(mid-path noise norm {a.sigma / 4 * 192**0.5:.2f}) | v_scale {v_scale:.2f}")

    model = BridgeDiT(v_scale=v_scale, sigma=a.sigma).cuda()
    print(f"params {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M")
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    def save(step):
        torch.save({
            "model": ema.state_dict(), "step": step, "sigma": a.sigma, "r": r,
            "v_scale": v_scale, "stride": stride, "frameskip": FRAMESKIP,
        }, a.out)

    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        loss, lv, le = loss_fn(model, data.sample(a.batch, tr), a.sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        log.append((lv.item(), le.item()))
        if step % 1000 == 0 or step == a.steps:
            with torch.no_grad():
                _, vv, ve = loss_fn(ema, data.sample(4096, va), a.sigma)
            lv_tr, le_tr = np.mean(log, axis=0)
            log = []
            print(f"step {step} train v {lv_tr:.4f} eta {le_tr:.4f} | val v {vv.item():.4f} "
                  f"eta {ve.item():.4f} | {step / (time.time() - t0):.0f} it/s", flush=True)
            save(step)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
