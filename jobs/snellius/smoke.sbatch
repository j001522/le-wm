#!/bin/bash
# Smoke test on one GPU: the 4 variants one at a time (500 steps each), then a resume check of
# LeWM-D, with GPU-utilisation sampling. Checkpoints go to checkpoints/smoke_<ds>_<variant>.
#   usage: for p in gpu_h100 gpu_a100; do DS=cube sbatch -p $p -J lewm_smoke_cube smoke.sbatch; done
# The first of the twins to start cancels the other.
#SBATCH --gpus=1 -c 16 -t 01:00:00
#SBATCH -o /gpfs/work4/0/prjs0951/Giacomo/LeWM/logs/%x_%j.out
scancel -u $USER -n $SLURM_JOB_NAME --state=PENDING
B=/gpfs/work4/0/prjs0951/Giacomo/LeWM
export DS=${DS:-pusht} SP=smoke_${DS:-pusht}
rm -rf $B/stablewm/checkpoints/${SP}_*   # fresh run dirs so the resume check is meaningful
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 20 > $B/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv &
NW=14 PAR=1 STEPS=500 CK=200 bash $B/le-wm/jobs/snellius/smoke_par.sh
