#!/bin/bash
# Planning videos (simulation | decoded imagination) for several runs, one after the other.
#   usage: [EVAL=10 VIDEOS=10 OUT=$B/eval_videos/<tag>] sbatch plan_videos.sbatch pusht_LeWM-D pusht_LeFWM-D ...
# Success rate is measured on EVAL episodes, videos are saved for the first VIDEOS of them.
# Memory: ~1 GB of GPU memory at 4 episodes, ~11 GB at 50, so a MIG slice is enough.
#SBATCH -p gpu_mig --gpus=1 -c 9 -t 01:00:00 -J lewm_videos
#SBATCH -o /gpfs/work4/0/prjs0951/Giacomo/LeWM/logs/%x_%j.out
source /gpfs/work4/0/prjs0951/Giacomo/LeWM/le-wm/jobs/snellius/common.sh
cd $B/le-wm
nvidia-smi -L
for run in "$@"; do
  t=$(date +%s)
  python -m flow_dynamics.plan_video --run $run --num-eval ${EVAL:-10} --num-videos ${VIDEOS:-10} \
    --out ${OUT:-$B/eval_videos/blocks} > $B/logs/plan_video_${SLURM_JOB_ID}_$run.log 2>&1
  echo "$run exit $? in $(( $(date +%s)-t ))s: $(grep -oE '"success_rate": [0-9.]+' $B/logs/plan_video_${SLURM_JOB_ID}_$run.log)"
done
