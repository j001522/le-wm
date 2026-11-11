# LeWM + flow-matching dynamics

Keep LeWM's autoregressive latent rollout, but **sample** each next latent with a flow
(x-prediction, linear path `x_t = (1-t) eps + t z`) instead of regressing its mean.

## Where the code is
- `module.py`: `TimeEmb`, `AdaLNResBlock`, `FlowHead` (variant B), `FlowARPredictor` (variant A)
- `jepa.py`: `flow_variant` none|head|transformer, `sample_next` (Euler in x-pred form +
  endpoint drift), `rollout_latents`, `flow_loss`, common-random-number planning noise
  (`plan_eps`), `plan_noise`/`risk_weight` aggregation in `get_cost`. With no flow head,
  behaviour equals the original (tested).
- `train.py` + `config/train/model/lewm_flow.yaml` (B) / `lewm_flow_ar.yaml` (A): end-to-end
  training (stages 2-3), e.g. `python train.py model=lewm_flow loader.batch_size=32`.
- `flow_dynamics/`: stage 1 (frozen encoder, cached latents)
  - `common.py`: `LatentWindows` (all 2.3M PushT latents on the GPU, 5% of episodes held out),
    `build_dynamics` (init from the released predictor), `full_model` (encoder + any dynamics)
  - `train_latent.py`: train variant none/head/transformer; `--action-noise` makes the
    next latent stochastic given the actions the model sees
  - `eval_rollout.py`: k-step rollout error, ensemble spread, drift-vs-spread rank correlation,
    calibration, SIGReg statistic of samples, model cost of one CEM iteration
  - `eval_plan.py`: CEM success in the simulator on the root `eval.py` tasks
  - `tests/test_flow.py`: `python -m flow_dynamics.tests.test_flow`
  - `save/`: checkpoints, logs, eval json

## Stage-1 recipe
    python -m flow_dynamics.train_latent --variant head --steps 20000 --out head.pt
    python -m flow_dynamics.train_latent --variant none --steps 20000 --out mse.pt
    python -m flow_dynamics.eval_rollout --models pretrained mse.pt head.pt
    python -m flow_dynamics.eval_plan --model head.pt

## Stage-1 results (Push-T, 2026-11-10)
All dynamics start from the released predictor and train on cached latents
(5% of episodes held out). Rollout error = per-dim MSE on 2048 held-out windows (k steps,
history 3). Planning = CEM 300x30 in the simulator, the 50 `eval.py` tasks (seed 42),
25-step goals, budget 50. One seed: differences of 1-3 tasks are noise.
JSON: `save/eval_*.json`, `save/plan_results.jsonl`, `save/plan_noisy.jsonl`.

**Clean Push-T (deterministic)**

| model | k=1 | k=5 | k=10 | k=20 | planning |
|---|---|---|---|---|---|
| released predictor | .0073 | .071 | .250 | .599 | 92% |
| `mse.pt` (MSE, 10k steps) | .0040 | .035 | .127 | .350 | **98%** |
| `head.pt` (B, 20k) MSE head | .0038 | .033 | .126 | .346 | 94% |
| `head.pt` 1 sample | .0039 | .040 | .160 | .442 | 96% (2.4x time) |
| `head.pt` x̂(τ=0) mean mode | .0036 | .038 | .153 | .436 | 94% |
| `head.pt` 4 samples / mean of 16 | .0034 | .033 | .128 | .333 | 96% (4 samples, 7.7x time) |
| `ar.pt` (A, 10k) mean of 16 | .047 | .43 | 1.02 | 1.37 | - |
| `ar_ctx.pt` (A, `--ctx-frac 0.5`) mean of 16 | .033 | .29 | .70 | 1.01 | - |

- The gain over the released model is fine-tuning, not the flow (`mse.pt` matches B).
- B's flow loss is smooth in τ, samples have real-latent SIGReg statistics, and the
  flow does not hurt the trunk. One sample adds about the ensemble variance to the error.
- Spread ranks error (Spearman .5-.7) but is overconfident (err/var 3-9): the error is model
  error, not aleatoric. Endpoint drift tracks spread only at k=1 (.63 -> .06 at k=20).
- Variant A trained with logit-normal τ per frame rarely sees near-clean history (inference uses
  τ=0.9); matching it (`ctx_frac`) helps a lot but A is still 10x worse than B at 10k steps and
  costs 30x the MSE model per CEM iteration.

**Stochastic test: model sees actions + N(0,1) noise (normalised units)**

| model | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| `mse_an1.pt` | .0148 | .176 | .531 | 1.12 |
| `head_an1.pt` MSE head | .0148 | .176 | .533 | 1.11 |
| `head_an1.pt` 1 sample | .021 | .237 | .661 | 1.42 |
| `head_an1.pt` mean of 16 | .0155 | .179 | **.460** | **.855** |

- With real aleatoric uncertainty, the 16-sample mean beats MSE by 13% at k=10 and 24% at k=20
  (MSE rollouts feed blurred means back in). Calibration improves (err/var 1.4-2.5).

**Planning with executed-action noise σ=1** (`--exec-noise 1.0`): released 26%, `mse.pt` 36%,
`mse_an1` 18%, `head_an1` 1 sample 10%. Caveat: the `*_an1` models learned p(z' | true + noise),
which shrinks the action effect, while planning needs p(z' | commanded), executed = commanded + noise.
They are the wrong conditional for planning; a proper test needs rollouts logged with commanded
actions and noisy execution.

**Variant A follow-up.** The first A runs trained the whole transformer at the pretrained
learning rate (3e-5). 16-sample mean, clean Push-T, 10k steps each:

| run | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| `ar_ctx` lr 3e-5, `--ctx-frac 0.5` | .033 | .29 | .70 | 1.01 |
| `ar_lr` `--trunk-lr 3e-4 --ctx-frac 0.5` | .014 | .13 | .35 | .68 |
| `ar_lr_c97` + `--ctx-tau 0.97` | .014 | .14 | .37 | .72 |
| `ar_lr_w` + `--tau-weight` (Dreamer 4 ramp) | .013 | .13 | .34 | .65 |
| `ar_f09` `--ctx-frac 0.9` | .010 | .10 | .31 | .63 |
| `ar_f09_scratch` + `--init none` | **.0089** | **.096** | **.29** | **.60** |
| B / `mse.pt` (reference) | .0034 / .0040 | .033 / .035 | .13 / .13 | .33 / .35 |

The context signal level doesn't matter, and one-step mean mode equals the ensemble, so the error
is in the network. Training more windows in the inference regime helps most, and the released
weights don't help A (they were trained for a different output). A still gets one target per
window vs B's 3 tokens x 4 taus. Next: dense clean-context training (clean context tokens plus
noisy target tokens in one sequence, with an attention mask so every frame is a target with
clean history).

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
  the decoder. Reconstructions are sharp; see `python -m flow_dynamics.visualize_e2e`.
- **Planning** (`eval_plan.py --e2e lewm_flow_e2e/weights_epoch_14.pt`, same 50 tasks):
  MSE head 72%, 1 flow sample 52%, mean mode 34% (released LeWM 92%). The model is undertrained;
  the flow-vs-MSE ordering matches stage 1.
