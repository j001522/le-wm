# Analysis (thesis branch)

Experiments, maps and plots on trained runs. Every script takes `--run <run> --epoch N`
(a checkpoint dir under `$STABLEWM_HOME/checkpoints`) and writes to
`results/<experiment>/<run>_e<epoch>/`. Shared code (model/decoder loading, video, latent
caches) is in `helpers/` on main.

## support/ — does imagination stay close to reality where training data is dense?
Run in this order (later scripts read `eval_edge`'s outputs; all write to `results/edge/<run>_e<epoch>/`):

    python -m scripts.cache_latents --task pusht --run <run> --epoch N   # latent cache, ~40 min (needed by the maps)
    python -m analysis.support.eval_edge  --run <run> --epoch N          # typical / edge / outside starts, videos, signals
    python -m analysis.support.latent_map --run <run> --epoch N          # paper-style latent map + imagined paths
    python -m analysis.support.map_check  --run <run> --epoch N          # which 2-D projection keeps the structure
    python -m analysis.support.local_maps --run <run> --epoch N          # per-episode local t-SNE
    python -m analysis.support.support    --run <run> --epoch N          # training support vs imagination error

## surprise/ — do the uncertainty signals react to unexpected events?

    python -m analysis.surprise.eval_surprise --run <run> --epoch N --events color goal block

`eval_edge` and `eval_surprise` read `u_f` (flow endpoint drift), which only flow models produce
(`rollout_latents` returns no drift without one), so they need a flow run, not e.g. `pusht_LeWM-D`.
