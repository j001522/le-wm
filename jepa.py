"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        flow_head=None,
        flow_variant="none",
        sample_steps=10,
        sample_power=2.0,
        ctx_tau=0.9,
        plan_noise=1,
        risk_weight=0.0,
        plan_seed=0,
    ):
        super().__init__()
        # flow_variant: "none" (MSE predictor), "head" (B: trunk + FlowHead),
        # "transformer" (A: predictor is a FlowARPredictor)
        if flow_variant == "none" and flow_head is not None:
            flow_variant = "head"
        self.flow_head = flow_head
        self.flow_variant = flow_variant
        self.sample_steps, self.sample_power, self.ctx_tau = sample_steps, sample_power, ctx_tau
        self.plan_noise, self.risk_weight, self.plan_seed = plan_noise, risk_weight, plan_seed

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3, eps=None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        D = emb.size(-1)
        if self.is_flow and eps is None:
            eps = self.plan_eps(emb.size(0), n_steps + 1, D, emb.device)
        emb, drift = self.rollout_latents(emb, act, act_future, history_size, eps)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        if drift is not None:
            info["drift"] = rearrange(drift, "(b s) ... -> b s ...", b=B, s=S)

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        M = self.plan_noise if self.is_flow else 1
        B, S = action_candidates.shape[:2]
        if M > 1:  # candidate s, noise m sits at flat index s*M + m (see plan_eps)
            action_candidates = action_candidates.repeat_interleave(M, dim=1)
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        if M > 1:
            cost = cost.view(B, S, M)
            cost = cost.mean(-1) + self.risk_weight * cost.std(-1)
        
        return cost

    ###########################
    ## Flow-matching dynamics ##
    ###########################

    @property
    def is_flow(self):
        return self.flow_variant != "none"

    def plan_eps(self, n, steps, dim, device):
        """Common random numbers: the same noise on every call (all candidates, all
        CEM iterations), so CEM ranks actions, not noise. Row i uses noise i % plan_noise."""
        g = torch.Generator(device=device).manual_seed(self.plan_seed)
        M = self.plan_noise
        base = torch.randn(M, steps, dim, generator=g, device=device)
        return base.repeat((n + M - 1) // M, 1, 1)[:n]

    def tau_grid(self, steps, device):
        k = torch.linspace(0, 1, steps + 1, device=device)
        return 1 - (1 - k) ** self.sample_power  # 0 ... 1, dense near 1

    def act_prev(self, act_emb):
        """Variant A conditioning: token i gets a_{i-1}; token 0 gets zeros (as in training)."""
        return torch.cat([torch.zeros_like(act_emb[:, :1]), act_emb[:, :-1]], 1)

    @torch.no_grad()
    def sample_next(self, hist, act_hist, eps, steps=None):
        """Sample the next latent by Euler ODE in x-prediction form.
        hist: (N, T, D) latents, act_hist: (N, T, D) action embeddings aligned with hist
        (act_hist[:, i] is the action taken after hist[:, i]), eps: (N, D).
        steps=1 returns the conditional mean x̂(tau=0) (the deterministic mode).
        Returns the next latent (N, D) and the endpoint-drift info."""
        steps = steps or self.sample_steps
        taus = self.tau_grid(steps, eps.device)
        if self.flow_variant == "head":
            h = self.predictor(hist, act_hist)[:, -1]  # trunk once per rollout step

            def denoise(x, t):
                return self.flow_head(x, t.expand(x.shape[:-1]), h)
        else:
            N, T, _ = hist.shape
            # lightly noised history, as in training with diffusion forcing
            noisy = self.ctx_tau * hist + (1 - self.ctx_tau) * torch.randn_like(hist)
            cond = self.act_prev(torch.cat([act_hist, torch.zeros_like(act_hist[:, :1])], 1))

            def denoise(x, t):
                tau = torch.cat([torch.full((N, T), self.ctx_tau, device=x.device), t.expand(N, 1)], 1)
                return self.predictor(torch.cat([noisy, x[:, None]], 1), cond, tau)[:, -1]

        x, x_hats = eps, []
        for t0, t1 in zip(taus[:-1], taus[1:]):
            x_hat = denoise(x, t0)
            x_hats.append(x_hat)
            x = x + (t1 - t0) * (x_hat - x) / (1 - t0)  # lands exactly on x̂ when t1 = 1
        x_hats = torch.stack(x_hats, 1)  # (N, K, D)
        sd = x.size(-1) ** 0.5
        info = {
            "total": (x - x_hats[:, 0]).norm(dim=-1) / sd,  # one-sample conditional-std estimate
            "steps": (x_hats[:, 1:] - x_hats[:, :-1]).norm(dim=-1) / sd,  # (N, K-1)
            "mean": x_hats[:, 0],  # x̂(tau=0) = conditional mean
        }
        return x, info

    def step(self, hist, act_hist, eps=None, steps=None):
        """One autoregressive step: next latent (N, D) and total drift (N,) or None."""
        if not self.is_flow:
            return self.predict(hist, act_hist)[:, -1], None
        z, d = self.sample_next(hist, act_hist, eps, steps)
        return z, d["total"]

    def rollout_latents(self, emb, act, act_future, history_size=3, eps=None, steps=None):
        """Autoregressive latent rollout.
        emb: (N, H, D) context latents, act: (N, H, A) raw actions taken after them,
        act_future: (N, n, A). eps: (N, n+1, D) noise (flow only).
        Returns latents (N, H+n+1, D) and drift (N, n+1) or None."""
        HS = history_size
        drifts = []
        n = act_future.size(1)
        for t in range(n + 1):
            act_emb = self.action_encoder(act)
            z, d = self.step(emb[:, -HS:], act_emb[:, -HS:], None if eps is None else eps[:, t], steps)
            emb = torch.cat([emb, z[:, None]], dim=1)
            drifts.append(d)
            if t < n:
                act = torch.cat([act, act_future[:, t : t + 1]], dim=1)
        drift = torch.stack(drifts, 1) if self.is_flow else None
        return emb, drift

    def flow_loss(self, emb, act_emb, ctx_len, n_preds, detach_target=True,
                  taus_per_token=4, tau_mean=0.0, tau_std=1.0, tau_weight=False, ctx_frac=0.0,
                  detach_head=False):
        """Flow-matching (x-prediction) loss on encoded windows.
        emb, act_emb: (B, T, D). ctx_frac (variant A): fraction of windows whose history uses
        tau = ctx_tau, as at inference. Returns flow_loss, the MSE prediction (or None) and tau/err for logging."""
        def sample_tau(shape):
            return torch.sigmoid(tau_mean + tau_std * torch.randn(shape, device=emb.device))

        if self.flow_variant == "head":
            ctx_emb, ctx_act = emb[:, :ctx_len], act_emb[:, :ctx_len]
            tgt = emb[:, n_preds:]
            z1 = tgt.detach() if detach_target else tgt
            h = self.predictor(ctx_emb, ctx_act)  # trunk, before pred_proj
            K = taus_per_token
            h_f = h.detach() if detach_head else h
            h_k, z_k = (t.unsqueeze(2).expand(-1, -1, K, -1) for t in (h_f, z1))
            tau = sample_tau(h_k.shape[:-1])  # (B, T, K)
            x_tau = (1 - tau[..., None]) * torch.randn_like(z_k) + tau[..., None] * z_k
            err = (self.flow_head(x_tau, tau, h_k) - z_k).pow(2).mean(-1)
            b, t = h.shape[:2]
            mse_pred = rearrange(self.pred_proj(rearrange(h, "b t d -> (b t) d")), "(b t) d -> b t d", b=b)
        else:  # diffusion forcing: independent tau per frame, every position trained
            z = emb.detach() if detach_target else emb
            tau = sample_tau(z.shape[:-1])  # (B, T)
            if ctx_frac > 0:  # inference regime: lightly noised history at ctx_tau, random last frame
                rows = torch.rand(z.size(0), device=z.device) < ctx_frac
                tau[rows, :-1] = self.ctx_tau
            x_tau = (1 - tau[..., None]) * torch.randn_like(z) + tau[..., None] * z
            err = (self.predictor(x_tau, self.act_prev(act_emb), tau) - z).pow(2).mean(-1)
            mse_pred = None
        w = (0.1 + 0.9 * tau) if tau_weight else 1.0
        return {"flow_loss": (w * err).mean(), "mse_pred": mse_pred, "tau": tau.detach(), "err": err.detach()}
