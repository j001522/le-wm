#!/bin/bash
# Hallucination experiments (analysis/halluc, thesis branch): runs one python module with the given
# arguments from the thesis checkout ($REPO, default $B/le-wm-thesis, a git worktree of le-wm).
#   sbatch -J halluc_export  halluc.sbatch analysis.halluc.export  --set pusht_v1 --n 100 --annotate
#   sbatch -J halluc_metrics halluc.sbatch analysis.halluc.metrics --sets pusht_v1
#   sbatch -J halluc_plan -t 03:00:00 halluc.sbatch scripts.eval_plan --run pusht_LeWM-MSE-dFH-D --mse-head \
#       --num-eval 50 --halluc-set plan_pusht
# See analysis/halluc/README.md for the full order.
#SBATCH -p gpu_mig --gpus=1 -c 9 -t 01:00:00 -J halluc
#SBATCH -o /gpfs/work4/0/prjs0951/Giacomo/LeWM/logs/%x_%j.out
source /gpfs/work4/0/prjs0951/Giacomo/LeWM/le-wm/jobs/snellius/common.sh
REPO=${REPO:-$B/le-wm-thesis}
export PYTHONPATH=$REPO:$B/pylib
export HF_HUB_OFFLINE=1                    # DINOv2 weights from the local HF cache (compute nodes are offline)
cd $REPO
echo "$(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) $(git status --short | wc -l) changed"
nvidia-smi -L
echo "python -m $*"
python -m "$@"
echo "exited with $?"
