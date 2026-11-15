# Experiments: LeWM + flow-matching dynamics

Keep LeWM's autoregressive latent rollout, but **sample** each next latent with a flow
(x-prediction, linear path `x_t = (1-t) eps + t z`) instead of regressing its mean.

## Where the code is
- `module.py`: `TimeEmb`, `AdaLNResBlock`, `FlowHead` (variant B), `FlowARPredictor` (variant A)
- `jepa.py`: `flow_variant` none|head|transformer, `sample_next` (Euler in x-pred form +
  endpoint drift), `rollout_latents`, `flow_loss`, common-random-number planning noise
  (`plan_eps`), `plan_noise`/`risk_weight` aggregation in `get_cost`. With no flow head,
  behaviour equals the original (tested).
- `train.py` + `config/train/model/lewm_flow.yaml` (B) / `lewm_flow_ar.yaml` (A): end-to-end
  training, e.g. `python train.py model=lewm_flow decoder.enabled=true`. The Push-T variants trained
  on Snellius (LeWM-D, LeWM-MSE-dFH-D, LeWM-FH-D, LeWM-MSE-FH-D, LeFWM-D) and their settings are in
  `jobs/snellius/common.sh`.
- `helpers/`: shared code for the scripts
  - `constants.py`: paths (`SWM` = `$STABLEWM_HOME`, `RESULTS`), `TASKS`, `run_dir`
  - `models.py`: `load_run` (trained run + decoder)
  - `latents.py`: latent-cache paths
  - `planning.py`: tasks, scalers, transform and CEM as in the root `eval.py`
  - `video.py`: encode frames, decode latents, overlays, mp4/gif
- `scripts/` (`python -m scripts.<name> --run <run> --epoch N`)
  - `eval_plan.py`: CEM success in the simulator on the root `eval.py` tasks; `--num-videos N`
    for videos (simulation | decoded imagination of the plan)
  - `visualize_e2e.py`: decoded real / reconstructed / rolled-out frames
  - `cache_latents.py`: encode every dataset frame with the run's encoder
- `tests/test_flow.py`: `python -m tests.test_flow`
- Outputs: `results/<experiment>/<run>_e<epoch>/` (gitignored); weights in `$STABLEWM_HOME/checkpoints/`.

## End-to-end (encoder + flow B from scratch, with decoder), 2026-11-11

    python train.py model=lewm_flow data.dataset.name=pusht_expert_train.h5 decoder.enabled=true \
      loader.batch_size=32 num_workers=2 loader.prefetch_factor=2 loader.persistent_workers=false \
      loss.sigreg.weight=0.3 trainer.max_epochs=20 +trainer.limit_train_batches=5000 \
      +trainer.limit_val_batches=200 output_model_name=lewm_flow_e2e subdir=lewm_flow_e2e

Settings sized for an 8 GB GPU / 7 GB RAM laptop. SIGReg 0.3 compensates batch 32: the statistic
scales with batch size, so use ~0.09 at batch 128. The run stopped at epoch 14 (70k steps, about
2.2M windows vs about 178M for the released model) when its session ended. Validation losses were
still falling (flow .0071, MSE head .0101, decoder .0108, SIGReg 1.24; no collapse).

- **Decoder** (`module.PixelDecoder`, 224x224, `decoder.frames=last`): trained on the same
  batches from detached latents, with its own optimizer, gradient clipping and RNG stream.
  Verified: with `+trainer.deterministic=true` the WM weights are bit-identical with and without
  the decoder. Reconstructions are sharp; see `python -m scripts.visualize_e2e`.
- **Planning** (`eval_plan.py --run lewm_flow_e2e --epoch 14`, same 50 tasks):
  MSE head 72%, 1 flow sample 52%, mean mode 34% (released LeWM 92%). The model is undertrained.
