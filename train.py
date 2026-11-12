import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


class ResumeCkpt(pl.Callback):
    """Full trainer state to a fixed path every n steps and at each epoch end (atomic replace)."""

    def __init__(self, path, every_n_steps):
        self.path, self.n = Path(path), every_n_steps

    def _save(self, trainer):
        tmp = self.path.with_suffix(".tmp")
        trainer.save_checkpoint(tmp)
        if trainer.is_global_zero:
            os.replace(tmp, self.path)

    def on_train_batch_end(self, trainer, *args):
        if self.n and trainer.global_step % self.n == 0:
            self._save(trainer)

    def on_train_epoch_end(self, trainer, *args):
        self._save(trainer)


class LinkSptRun(pl.Callback):
    """Symlink the spt RegistryLogger run dir (metrics.csv) into <run_dir>/spt_runs/<job>_<run_id>."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir) / "spt_runs"

    def on_fit_start(self, trainer, *args):
        if not trainer.is_global_zero:
            return
        for lg in trainer.loggers:
            src = getattr(lg, "run_dir", None)
            if type(lg).__name__ == "RegistryLogger" and src:
                self.dir.mkdir(exist_ok=True)
                dst = self.dir / f"job{os.environ.get('SLURM_JOB_ID', 'local')}_{getattr(lg, 'run_id', '')}"
                if not dst.exists():
                    dst.symlink_to(Path(src).resolve())


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
            detach_head=fc.get("detach_head", False),
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

    # metrics: <run_dir>/tb (TensorBoard, appended across resumes). spt.Manager drops any CSVLogger
    # and adds its own RegistryLogger (metrics.csv in a timestamped spt run dir); LinkSptRun links
    # that dir into <run_dir>/spt_runs/ so every job's CSV is findable next to the checkpoints.
    logger = []
    try:
        logger.append(TensorBoardLogger(save_dir=run_dir, name="tb", version="", default_hp_metric=False))
    except ModuleNotFoundError as e:
        print(f"TensorBoard logging disabled: {e}")
    if cfg.wandb.enabled:
        logger.append(WandbLogger(**cfg.wandb.config))
    for lg in logger:
        lg.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=cfg.ckpt_epoch_interval,
        extra_dir=run_dir, decoder_cfg=cfg.decoder.net if "decoder" in extra else None,
    )

    # full-state checkpoint (optimizer, scheduler, RNG) in run_dir, for resuming across jobs
    # (a plain Callback, not ModelCheckpoint: spt.Manager redirects every ModelCheckpoint to its own
    # timestamped cache dir, where a resubmitted job would never find it)
    resume_ckpt = ResumeCkpt(run_dir / "last.ckpt", cfg.ckpt_every_n_steps)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, resume_ckpt, LinkSptRun(run_dir)],
        num_sanity_val_steps=1,
        logger=logger or None,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / "last.ckpt"
    if ckpt_path.exists():
        print(f"resuming full state from {ckpt_path}")
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        weights_only=False,
    )

    manager()
    return


if __name__ == "__main__":
    run()
