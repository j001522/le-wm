#!/bin/bash
# One variant on one GPU, paper recipe (10 epochs).
#   usage:   [DS=pusht|cube] sbatch -p <partition> -J <ds>_<variant> train1.sbatch <variant> [hydra overrides...]
#   example: for p in gpu_h100 gpu_a100; do sbatch -p $p -J pusht_LeWM-D train1.sbatch LeWM-D; done
#   variants: LeWM-D  LeWM-MSE-dFH-D  LeWM-FH-D  LeWM-MSE-FH-D  LeFWM-D   (see common.sh)
# Submit to a100 and h100 under the same job name: the first to start cancels its pending twin.
# Resubmittable: train.py resumes from $STABLEWM_HOME/checkpoints/<ds>_<variant>/last.ckpt.
#SBATCH --gpus=1 -c 16 -t 1-00:00:00
#SBATCH -o /gpfs/work4/0/prjs0951/Giacomo/LeWM/logs/%x_%j.out
scancel -u $USER -n $SLURM_JOB_NAME --state=PENDING
echo "running $SLURM_JOB_NAME in job $SLURM_JOB_ID on $SLURM_JOB_PARTITION $(hostname) at $(date)"
source /gpfs/work4/0/prjs0951/Giacomo/LeWM/le-wm/jobs/snellius/common.sh
v=$1; shift; name=${DS:-pusht}_$v
[ -n "$(variant_args $v | grep model=)" ] || { echo "unknown variant $v"; exit 1; }
stage $(ds_file)
# paper recipe: 10 epochs on every environment (the repo config's max_epochs=100 is not it)
$TRAIN $(variant_args $v) trainer.max_epochs=10 num_workers=14 \
  output_model_name=$name subdir=$name wandb.config.name=$name "$@"
echo "exited with $? at $(date)"
