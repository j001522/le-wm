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

**Variant A follow-up (in progress).** The first A runs trained the whole transformer at the
pretrained learning rate (3e-5). With `--trunk-lr 3e-4 --ctx-frac 0.5` (`ar_lr.pt`) the
16-sample mean improves to .014 / .13 / .35 / .68 (k=1/5/10/20), still about 4x B at k=1.
Context signal 0.97 vs 0.9 makes no difference, and one-step mean mode equals the ensemble at
k=1, so the error is in the network, not the sampler. Next: `--ctx-frac 0.9` (more training
in the inference regime), training from scratch, and Dreamer 4's ramp loss weight.
