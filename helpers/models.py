"""Model loading: trained runs (+ decoder), the released LeWM, and stage-1 dynamics models.

Stage-1 models (scripts/stage1) are JEPAs with only the dynamics parts, trained on cached
released-encoder latents; full_model() puts them back on the released encoder.
"""

import json

import hydra
import torch

from helpers.constants import STAGE1, SWM, TASKS


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
    ('pretrained') or the dynamics from a stage-1 checkpoint in STAGE1.
    flow_kw overrides sampling/planning attributes (sample_steps, plan_noise, risk_weight, ...)."""
    cfg, sd = pretrained_state(task)
    cfg = _local(cfg)
    cfg["encoder"], cfg["projector"] = pretrained_state(task)[0]["encoder"], json.loads(
        json.dumps(pretrained_state(task)[0]["projector"]).replace("stable_worldmodel.wm.lewm.module.", "module."))
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(sd)
    if name != "pretrained":
        dyn, _ = load_model(STAGE1 / name, "cpu")
        model.predictor, model.action_encoder = dyn.predictor, dyn.action_encoder
        model.pred_proj, model.flow_head, model.flow_variant = dyn.pred_proj, dyn.flow_head, dyn.flow_variant
    for k, v in flow_kw.items():
        assert hasattr(model, k), k
        setattr(model, k, v)
    return model.to(device).eval().requires_grad_(False)
