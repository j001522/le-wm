#!/bin/bash
# TensorBoard over all runs' tb/ dirs (tensorboard lives in $B/pylib, not in the venv, so the plain
# `tensorboard` command does not exist).   usage: bash tensorboard.sh [port] [logdir]
# Then open it from your laptop:
#   VS Code Remote-SSH: the PORTS tab forwards it automatically (or add the port there by hand);
#   plain ssh:          ssh -N -L <port>:<login node>:<port> <user>@snellius.surf.nl  -> http://localhost:<port>
B=/gpfs/work4/0/prjs0951/Giacomo/LeWM
PORT=${1:-6006}
LOGDIR=${2:-$B/stablewm/checkpoints}
source $B/.venv/bin/activate
echo "login node: $(hostname -s)   tunnel: ssh -N -L $PORT:$(hostname -s):$PORT <user>@snellius.surf.nl"
PYTHONPATH=$B/pylib exec python -m tensorboard.main --logdir "$LOGDIR" --port "$PORT" --bind_all
