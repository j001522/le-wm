"""BridgeDiT: predicts path velocity v and noise eta from (x, tau, z_start, z_goal)."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def tau_embedding(tau, n_freqs=10):
    """Fourier features with octave frequencies, to resolve fine structure on [0, 1]."""
    freqs = 2.0 ** torch.arange(n_freqs, device=tau.device, dtype=tau.dtype) * math.pi
    args = tau[:, None] * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def attend(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.attn_out(out.transpose(1, 2).reshape(b, t, d))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + gate1 * self.attend(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return x


class BridgeDiT(nn.Module):
    """Small DiT over three tokens: x, z_start, z_goal.

    The velocity target is dI/dtau + sigma*(1 - 2*tau)*eps. Its eps term is what
    pulls a drifting x back onto the path, but it is a tiny share of the target
    variance, so a plain regression barely learns it. forward() therefore builds
    it from the noise head: v = v_net * v_scale + sigma*(1 - 2*tau)*eta, where
    v_scale is the rms of the velocity target per dimension.
    """

    def __init__(self, z_dim=192, dim=256, depth=6, heads=4, v_scale=1.0, sigma=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.register_buffer("v_scale", torch.tensor(float(v_scale)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.proj = nn.Linear(z_dim, dim)
        self.type_emb = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.tau_mlp = nn.Sequential(nn.Linear(20, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head_v = nn.Linear(dim, z_dim)
        self.head_eta = nn.Linear(dim, z_dim)
        for head in (self.head_v, self.head_eta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_raw(self, x, tau, z_start, z_goal):
        tokens = self.proj(torch.stack([x, z_start, z_goal], dim=1)) + self.type_emb
        c = self.tau_mlp(tau_embedding(tau))
        for block in self.blocks:
            tokens = block(tokens, c)
        h = self.final_norm(tokens[:, 0])
        return self.head_v(h), self.head_eta(h)

    def noise_coefficient(self, tau):
        return (self.sigma * (1 - 2 * tau))[:, None]

    def forward(self, x, tau, z_start, z_goal):
        v, eta = self.forward_raw(x, tau, z_start, z_goal)
        return v * self.v_scale + self.noise_coefficient(tau) * eta, eta
