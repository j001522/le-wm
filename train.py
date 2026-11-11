import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label

    if not self.model.is_flow:
        # LeWM loss
        pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    else:
        # flow-matching dynamics (+ MSE head as encoder signal, SIGReg on clean embeddings)
        fc = cfg.flow
        fl = self.model.flow_loss(
            emb, act_emb, ctx_len, n_preds, detach_target=fc.detach_target,
            taus_per_token=fc.taus_per_token, tau_mean=fc.tau_logit_normal.mean,
            tau_std=fc.tau_logit_normal.std, tau_weight=fc.get("tau_weight", False), ctx_frac=fc.get("ctx_frac", 0.0),
        )
        output["flow_loss"] = fl["flow_loss"]
        mse_pred = fl["mse_pred"]
        output["pred_loss"] = ((mse_pred - tgt_emb).pow(2).mean()
                               if mse_pred is not None else torch.zeros((), device=emb.device))
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = (output["flow_loss"] + fc.mse_weight * output["pred_loss"]
                          + lambd * output["sigreg_loss"])

    if getattr(self, "decoder", None) is not None:
        # visualisation decoder: detached latents -> encoder-space pixels; no gradient reaches the WM
        frames = cfg.decoder.frames
        z, px = output["emb"].detach(), output["pixels"]
        if frames == "last":
            z, px = z[:, -1:], px[:, -1:]
        rec = self.decoder(z.flatten(0, 1))
        output["dec_loss"] = (rec.float() - px.flatten(0, 1).float()).pow(2).mean()
        output["loss"] = output["loss"] + output["dec_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # seed init, SIGReg projections and dropout, so runs (e.g. with vs without decoder) are comparable
    pl.seed_everything(cfg.seed, workers=True)
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    extra = {}
    if cfg.get("decoder", {}).get("enabled", False):
        # own RNG stream, so adding the decoder leaves the WM's initialisation unchanged
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 1)
            extra["decoder"] = hydra.utils.instantiate(cfg.decoder.net)
        # own optimizer: gradient clipping and steps are per optimizer, so the WM's are untouched
        optimizers["decoder_opt"] = {
            "modules": "decoder",
            "optimizer": dict(cfg.decoder.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
        **extra,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
        extra_dir=run_dir, decoder_cfg=cfg.decoder.net if "decoder" in extra else None,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
