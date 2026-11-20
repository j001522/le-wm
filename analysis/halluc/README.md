# halluc/ — do the label-free predictors detect hallucinations?

LeWM port of the dreamer4 hallucination experiments (AIC-Thesis/dreamer4, `logs/plan_figures/main`),
with the same window format, annotation tool, label metrics and statistics, so the two models'
tables compare. Model: a flow run with a decoder (`pusht_LeWM-MSE-dFH-D`); imagination = the MSE head
(as `eval_plan --mse-head`), predictors from the detached flow head and the decoder.

Everything for one checkpoint lives in `results/halluc/<run>_e<epoch>/`:

    <set>/data/<id>.npz            real, wm, recon (T,224,224,3) uint8, actions, z, z_real, pred_*, meta
    <set>/key.csv                  id -> episode, start, ... (planning sets: replan, n_imag, success)
    <set>/frames/, manifest.js, annotate.html, annotations/   hand-labelling sets only
    _hidden_metrics/{pred,mask,dino}_<set>/<id>_<task>.csv    one row per imagined frame (not for the annotator)
    eval_<sets>/                   label metrics vs hand labels (results.csv holds the thresholds)
    score_<sets>/                  predictors vs automatic labels

A window = one replan: 1 context frame + 5 imagined action blocks (FRAMESKIP = 5 env steps each), T = 6.

## Order (pusht)

Jobs: `sbatch -J <name> jobs/snellius/halluc.sbatch <module> <args>` runs `python -m <module> <args>`
from the le-wm checkout (thesis branch).

1. **Val windows → hand labels.** `analysis.halluc.export --set pusht_v1 --n 100 --annotate`
   (random, non-overlapping windows of the training run's val split; ids shuffled). Open
   `results/halluc/<run>_e10/pusht_v1/annotate.html` in a browser, label, export the JSON into
   `pusht_v1/annotations/`.
2. **Label metrics.** `analysis.halluc.metrics --sets pusht_v1` (masked Hausdorff / hd95 / IoU /
   chamfer / masked PSNR, DINOv2 patch distance, paper dPSNR), then
   `analysis.halluc.labels --sets pusht_v1` → AUROC per metric vs the hand labels, threshold, CV
   balanced accuracy. Pick the label metric (dreamer4: Hausdorff, mean over the horizon).
3. **Rest of the val set.** `analysis.halluc.export --set auto_val_pusht --n 2000 --exclude-sets pusht_v1`,
   `analysis.halluc.metrics --sets auto_val_pusht`, then
   `analysis.halluc.score --sets auto_val_pusht --thr-from eval_pusht_v1` → AUROC of u_r, u_f, u_s
   (and drift, /motion variants, motion alone, dPSNR) against the automatic label, with a
   cluster-bootstrap CI over episodes and an equal-motion AUROC.
4. **Planning.** `scripts.eval_plan --run <run> --mse-head --num-eval 50 --seed S --halluc-set plan_pusht_sS`
   records every replan as a window (sim frames | decoded imagination of the executed plan |
   predictors). Then `analysis.halluc.metrics --sets plan_pusht_s0 ...` and
   `analysis.halluc.score --sets plan_pusht_s0 ... --thr-from eval_pusht_v1` (also per success /
   complete windows: a plan cut short by the episode end keeps the blocks it reached).

### Hand labels from planning windows

Val windows are rarely hallucinated (pusht_LeWM-MSE-dFH-D: 7/100 windows with mean masked Hausdorff
> 8 px), planning replans much more often (39/224), so the planning windows make a better
hand-labelling set:

    python -m analysis.halluc.annotate_set --sets plan_pusht_s42 plan_pusht_s43 plan_pusht_s44 \
        plan_pusht_s45 --set plan_v1 --n 120 --blocks 4
    python -m analysis.halluc.metrics --sets plan_v1

Windows with >= 4 imagined blocks, truncated to 4 (annotate.html needs one length), drawn at random.
Then `labels --sets plan_v1` (or `--sets pusht_v1 plan_v1`) and `score ... --thr-from eval_plan_v1`.

## Definitions

Predictors per imagined step (RMS = norm / sqrt(D)), `rollout.py`:
- `u_r` = RMS(z − E(D(z))), decoder→encoder round trip of the imagined latent
- `u_f` = flow-head instability: mean RMS change of x̂ between Euler steps over the second half of
  the ODE, mean over 4 noise seeds (flow head on the trunk state that produced z)
- `u_s` = variance of the flow-head sample over the seeds; `drift` = |x − x̂(τ=0)|/√D
- `motion` = RMS(z_k − z_{k−1}); `<u>_norm` = u / motion; `oracle_*` use the real future (not predictors)

Label metrics, `metrics.py` (dreamer4 definitions): foreground masks by colour (`masks.py`; pusht:
T block, agent and green goal), Hausdorff = max of the two directed max distances between the
real and imagined masks; DINOv2(-reg) base patch cosine distance vs real and vs the decoder
reconstruction; paper dPSNR vs repeating the context frame. `labels.py` = dreamer4's
`eval_annotations.py` statistics; `annotate.html` = dreamer4's annotation tool, unchanged.

Caveat: LeWM's val split is a random 10 % of training *windows*, so val windows come from training episodes.
