"""Run: python -m tests.test_flow (from the repo root)."""

import importlib.util
import subprocess
import sys
import tempfile

import torch
from torch import nn

import jepa

D = 8

# Push-T model shapes, as in the trained runs' config.json (random init, no image encoder)
_PRED = dict(input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64,
             dropout=0.1, emb_dropout=0.0)


def build_model(variant="none", **kw):
    """JEPA without encoder: variant none (MSE) | head (B) | transformer (A)."""
    import module

    mlp = lambda: module.MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)  # noqa: E731
    act = module.Embedder(input_dim=10, emb_dim=192)
    if variant == "transformer":
        return jepa.JEPA(None, module.FlowARPredictor(num_frames=4, **_PRED), act, None, None,
                         flow_variant="transformer", **kw)
    head = module.FlowHead(dim=192, width=512, depth=4) if variant == "head" else None
    return jepa.JEPA(None, module.ARPredictor(num_frames=3, **_PRED), act, None, mlp(), flow_head=head,
                     flow_variant=variant, **kw)


class GaussianDenoiser(nn.Module):
    """Exact x-prediction denoiser for z ~ N(mu(h), sigma^2 I) on the path x = (1-t) eps + t z."""

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x, tau, h):
        t, s2 = tau[..., None], self.sigma ** 2
        return h + t * s2 / ((1 - t) ** 2 + t ** 2 * s2) * (x - t * h)


class Trunk(nn.Module):
    def forward(self, emb, act):
        return emb + act  # h = mu


def toy_model(sigma, **kw):
    return jepa.JEPA(None, Trunk(), nn.Identity(), flow_head=GaussianDenoiser(sigma), **kw)


def test_sampler_matches_gaussian_and_drift_is_sigma():
    for sigma in (0.05, 0.3, 1.0):
        m = toy_model(sigma, sample_steps=200, sample_power=2.0)
        N = 20000
        mu = torch.randn(N, 1, D)
        x, info = m.sample_next(mu, torch.zeros_like(mu), torch.randn(N, D))
        std = (x - mu[:, 0]).std().item()
        assert abs(std - sigma) / sigma < 0.05, (sigma, std)
        assert torch.allclose(info["mean"], mu[:, 0], atol=1e-5)  # x̂(0) is the conditional mean
        # total drift = |x - mu| / sqrt(D): its RMS is exactly sigma
        rms = info["total"].pow(2).mean().sqrt().item()
        assert abs(rms - sigma) / sigma < 0.05, (sigma, rms)


def test_one_step_is_the_mean():
    m = toy_model(0.5)
    mu = torch.randn(64, 1, D)
    x, _ = m.sample_next(mu, torch.zeros_like(mu), torch.randn(64, D), steps=1)
    assert torch.allclose(x, mu[:, 0], atol=1e-5)


def test_deterministic_target_gives_zero_drift():
    m = toy_model(1e-6, sample_steps=10)
    mu = torch.randn(64, 1, D)
    _, info = m.sample_next(mu, torch.zeros_like(mu), torch.randn(64, D))
    assert info["total"].max() < 1e-4


def test_plan_eps_is_common_random_numbers():
    m = toy_model(0.5, plan_noise=3, plan_seed=5)
    e1, e2 = m.plan_eps(12, 4, D, "cpu"), m.plan_eps(12, 4, D, "cpu")
    assert torch.equal(e1, e2)
    assert torch.equal(e1[0], e1[3]) and torch.equal(e1[1], e1[7]) and not torch.equal(e1[0], e1[1])


def _original_jepa():
    src = subprocess.run(["git", "show", "HEAD:jepa.py"], capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(src), f.close()
    spec = importlib.util.spec_from_file_location("jepa_orig", f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(12, 192)

    def forward(self, x, interpolate_pos_encoding=True):
        class O:
            pass
        o = O()
        o.last_hidden_state = self.lin(x.flatten(1))[:, None]
        return o


def test_mse_rollout_unchanged_from_original():
    torch.manual_seed(0)
    new = build_model("none").eval()
    new.encoder = FakeEncoder()
    old = _original_jepa().JEPA(new.encoder, new.predictor, new.action_encoder, None, new.pred_proj).eval()
    info = lambda: {"pixels": torch.randn(1, 4, 1, 3, 2, 2, generator=torch.Generator().manual_seed(1)),
                    "goal": torch.randn(1, 4, 1, 3, 2, 2, generator=torch.Generator().manual_seed(2)),
                    "action": torch.zeros(1, 4, 1, 10)}
    acts = torch.randn(1, 4, 6, 10)
    with torch.no_grad():
        a, b = new.get_cost(info(), acts), old.get_cost(info(), acts)
    assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_flow_variants_plan_with_noise_repeats():
    for variant in ("head", "transformer"):
        m = build_model(variant, plan_noise=3, risk_weight=0.5).eval()
        m.encoder = FakeEncoder()
        info = {"pixels": torch.randn(1, 4, 1, 3, 2, 2), "goal": torch.randn(1, 4, 1, 3, 2, 2),
                "action": torch.zeros(1, 4, 1, 10)}
        acts = torch.randn(1, 4, 6, 10)
        with torch.no_grad():
            c1 = m.get_cost(dict(info), acts)
            c2 = m.get_cost(dict(info), acts)
        assert c1.shape == (1, 4)
        if variant == "head":  # variant A also draws context noise from the global RNG
            assert torch.allclose(c1, c2)


def test_transformer_flow_loss_is_finite():
    """Variant A: shapes and finiteness of the loss."""
    m = build_model("transformer")
    z, a = torch.randn(4, 4, 192), torch.randn(4, 4, 192)
    out = m.flow_loss(z, a, 3, 1)
    assert out["flow_loss"].isfinite() and out["mse_pred"] is None


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(fails)
