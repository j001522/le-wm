"""Stage 1 helpers: cached frozen-encoder latents and dynamics-only JEPA models.

Latents come from the cache `<swm cache>/datasets/<name>_latents_v2.h5`, written by
`python -m flow_dynamics.cache_latents --task pusht`: z (N, 192) f16 for every raw frame,
action (N, A), ep_len, ep_offset. A predictor step spans FRAMESKIP raw frames and its
action is the row-major flattening of the FRAMESKIP raw actions (as in swm datasets).
"""

import json
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "flow_dynamics" / "save"
SWM = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
FRAMESKIP = 5
TASKS = {
    # env / cols / state_key: planning in the simulator (eval_plan.py)
    "pusht": dict(ckpt="quentinll/lewm-pusht", dataset="pusht_expert_train",
                  latents="pusht_expert_train_latents_v2.h5",
                  env="swm/PushT-v1", cols=["action", "proprio", "state"], state_key="state"),
    "tworoom": dict(ckpt="quentinll/lewm-tworooms", dataset="tworoom", latents="tworoom_latents_v2.h5"),
}


class LatentWindows:
    """All cached latents on the GPU; samples windows of L predictor steps.
    Episodes are split once into train/val by a fixed permutation."""

    def __init__(self, task="pusht", device="cuda", val_frac=0.05, seed=0):
        with h5py.File(SWM / "datasets" / TASKS[task]["latents"], "r") as f:
            z = f["z"][:]
            act = f["action"][:]
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_off = f["ep_offset"][:].astype(np.int64)
        ok = ~np.isnan(act).any(1)
        self.act_mean, self.act_std = act[ok].mean(0), act[ok].std(0)  # utils.get_column_normalizer
        act = np.nan_to_num((act - self.act_mean) / self.act_std, nan=0.0)
        self.device = device
        self.z = torch.from_numpy(z).to(device)  # f16
        self.act = torch.from_numpy(act.astype(np.float32)).to(device)
        perm = np.random.default_rng(seed).permutation(len(self.ep_len))
        n_val = int(round(val_frac * len(perm)))
        self.split = {"val": np.sort(perm[:n_val]), "train": np.sort(perm[n_val:])}
        self._starts = {}

    @property
    def dim(self):
        return self.z.size(1)

    def starts(self, split, L):
        """Global start frames of every window of L steps inside one episode."""
        key = (split, L)
        if key not in self._starts:
            span = (L - 1) * FRAMESKIP
            eps = self.split[split]
            n = np.maximum(self.ep_len[eps] - span, 0)
            s = np.concatenate([o + np.arange(k) for o, k in zip(self.ep_off[eps], n)])
            self._starts[key] = torch.from_numpy(s).to(self.device)
        return self._starts[key]

    def gather(self, starts, L):
        """starts (B,) -> z (B, L, D) f32, action chunks (B, L, FRAMESKIP*A).
        The last chunk may run past the episode end (it is never used as a condition
        for a target inside the window)."""
        k = torch.arange(L, device=self.device) * FRAMESKIP
        zi = starts[:, None] + k
        ai = (zi[..., None] + torch.arange(FRAMESKIP, device=self.device)).clamp(max=len(self.act) - 1)
        return self.z[zi].float(), self.act[ai].flatten(2)

    def sample(self, split, B, L, generator=None):
        s = self.starts(split, L)
        idx = torch.randint(len(s), (B,), device=self.device, generator=generator)
        return self.gather(s[idx], L)


def pretrained_state(task="pusht"):
    p = SWM / "checkpoints" / f"models--{TASKS[task]['ckpt'].replace('/', '--')}"
    return json.loads((p / "config.json").read_text()), torch.load(p / "weights.pt", map_location="cpu")


def _local(cfg):
    """swm checkpoint config -> this repo's classes, without the image encoder/projector."""
    s = json.dumps(cfg).replace("stable_worldmodel.wm.lewm.module.Predictor", "module.ARPredictor")
    s = s.replace("stable_worldmodel.wm.lewm.module.", "module.")
    cfg = json.loads(s)
    cfg["_target_"] = "jepa.JEPA"
    cfg.pop("encoder"), cfg.pop("projector")
    return cfg


def build_dynamics(variant="head", task="pusht", init="pretrained", head_width=512, head_depth=4, **jepa_kw):
    """A JEPA with only the dynamics parts (encoder=None).
    variant: none (MSE predictor) | head (B) | transformer (A).
    init='pretrained' copies the released predictor/action encoder/pred_proj
    (variant A copies the transformer weights; its 4th position embedding is new)."""
    cfg, sd = pretrained_state(task)
    cfg = _local(cfg)
    D = cfg["predictor"]["input_dim"]
    if variant == "head":
        cfg["flow_head"] = {"_target_": "module.FlowHead", "dim": D, "width": head_width, "depth": head_depth}
    if variant == "transformer":
        cfg["predictor"]["_target_"] = "module.FlowARPredictor"
        cfg["predictor"]["num_frames"] += 1
        cfg.pop("pred_proj")
    cfg["encoder"] = None
    model = hydra.utils.instantiate(cfg, flow_variant=variant, **jepa_kw)
    if init == "pretrained":
        sd = {k: v for k, v in sd.items() if not k.startswith(("encoder.", "projector."))}
        if variant == "transformer":
            sd = {k.replace("predictor.", "predictor.net.", 1): v for k, v in sd.items() if not k.startswith("pred_proj")}
            pe = sd["predictor.net.pos_embedding"]
            sd["predictor.net.pos_embedding"] = torch.cat([pe, pe[:, -1:]], 1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith(("flow_head.", "predictor.t.", "predictor.out.")) for k in missing), missing
    return model


def save_model(model, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    model = build_dynamics(m["variant"], m["task"], init="none", head_width=m.get("head_width", 512),
                           head_depth=m.get("head_depth", 4), ctx_tau=m.get("ctx_tau", 0.9))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), m


def full_model(name="pretrained", task="pusht", device="cuda", **flow_kw):
    """Root JEPA with the released encoder/projector and either the released predictor
    ('pretrained') or the dynamics from a stage-1 checkpoint in ARTIFACTS.
    flow_kw overrides sampling/planning attributes (sample_steps, plan_noise, risk_weight, ...)."""
    cfg, sd = pretrained_state(task)
    cfg = _local(cfg)
    cfg["encoder"], cfg["projector"] = pretrained_state(task)[0]["encoder"], json.loads(
        json.dumps(pretrained_state(task)[0]["projector"]).replace("stable_worldmodel.wm.lewm.module.", "module."))
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(sd)
    if name != "pretrained":
        dyn, _ = load_model(ARTIFACTS / name, "cpu")
        model.predictor, model.action_encoder = dyn.predictor, dyn.action_encoder
        model.pred_proj, model.flow_head, model.flow_variant = dyn.pred_proj, dyn.flow_head, dyn.flow_variant
    for k, v in flow_kw.items():
        assert hasattr(model, k), k
        setattr(model, k, v)
    return model.to(device).eval().requires_grad_(False)


# ---------------------------------------------------------------- planning setup (as the root eval.py)


def dataset_path(name):
    """Dataset name or path -> path. A bare name (no extension) resolves to <swm>/datasets/<name>.lance
    if it exists (Snellius), else <name>.h5 (local)."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    p = SWM / "datasets" / name
    if p.suffix in (".lance", ".h5"):
        return p
    for ext in (".lance", ".h5"):
        if p.with_suffix(ext).exists():
            return p.with_suffix(ext)
    raise FileNotFoundError(f"no {name}.lance or {name}.h5 in {SWM / 'datasets'}")


def load_planning_dataset(name, cols):
    """swm dataset (h5 or lance) with only the low-dimensional columns cached."""
    import stable_worldmodel as swm

    return swm.data.load_dataset(str(dataset_path(name)), keys_to_cache=cols)


def fit_scalers(dataset, cols):
    """Per-column StandardScaler, as in eval.py (goal_<col> shares <col>'s scaler)."""
    from sklearn import preprocessing

    process = {}
    for col in cols:
        v = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def image_transform(size=224):
    """Observation transform of eval.py: uint8 -> float, ImageNet normalisation, resize."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size),
    ])


def pick_tasks(dataset, n, goal_offset, seed):
    """Start (episode, step) pairs, same procedure and seed behaviour as eval.py.
    Works for h5 and lance (whose column_names omit the episode_idx index column)."""
    try:
        ep = dataset.get_col_data("episode_idx")
    except Exception:
        ep = dataset.get_col_data("ep_idx")
    step = dataset.get_col_data("step_idx")
    max_step = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(max_step, ep, step)
    valid = np.nonzero(step <= max_step[ep] - goal_offset)[0]  # eval.py: len - offset - 1
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, size=n, replace=False)])
    return ep[rows].tolist(), step[rows].tolist()


def cem_solver(model, num_samples=300, seed=42):
    """CEM with the settings of config/eval/solver/cem.yaml."""
    from stable_worldmodel.solver import CEMSolver

    return CEMSolver(model=model, batch_size=1, num_samples=num_samples, var_scale=1.0,
                     n_steps=30, topk=30, device="cuda", seed=seed)


def load_run(run, epoch, decoder=True, device="cuda"):
    """Trained run under <swm>/checkpoints/<run>: (model, decoder or None).
    decoder=True loads decoder_epoch_<epoch>.pt if the run has one."""
    import stable_worldmodel as swm

    import module

    model = swm.wm.utils.load_pretrained(f"{run}/weights_epoch_{epoch}.pt").to(device).eval().requires_grad_(False)
    path = SWM / "checkpoints" / run / f"decoder_epoch_{epoch}.pt"
    dec = None
    if decoder and path.exists():
        ck = torch.load(path, map_location="cpu")
        dec = module.PixelDecoder(**{k: v for k, v in ck["cfg"].items() if k != "_target_"})
        dec.load_state_dict(ck["state_dict"])
        dec = dec.to(device).eval().requires_grad_(False)
    return model, dec
