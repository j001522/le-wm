# Progress-time bridge planner (paused)

Subgoal planner on top of frozen LeWM (PushT, Two-Room). Moved here from the repo root
on 2026-11-10 when work switched to `flow_dynamics/`.

- `bridge/`: code (latent cache, graph, teacher paths, bridge DiT, planners, eval)
- `tests/`: unit tests
- `save/progress_bridge/`: checkpoints, graphs, eval logs (`bridge.common.ARTIFACTS`)

Run everything from this folder, e.g.

    cd progress_bridge
    ../.venv/bin/python -m bridge.eval --task pusht --planner bridge
    ../.venv/bin/python -c "import tests.test_bridge as t; [f() for n, f in vars(t).items() if n.startswith('test_')]"

It uses `stable_worldmodel`'s own LeWM class, not the root `jepa.py`/`module.py`, so changes
there don't affect it. The latent caches (`~/.stable_worldmodel/datasets/*_latents_v2.h5`)
are shared with `flow_dynamics/`.

Last results (one seed, 20 tasks): PushT bridge 9/20 vs plain CEM 0/20; Two-Room 15/20 vs 5/20.
