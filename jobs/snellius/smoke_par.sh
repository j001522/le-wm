#!/bin/bash
# Quick correctness smoke: the 4 variants (PAR at a time, STEPS steps each), then a resume check of
# LeWM-D. Env: DS (pusht|cube), SP (run-name prefix), PAR, STEPS, CK (ckpt every), NW (workers),
# SMOKE_EXTRA (extra overrides). Run via smoke.sbatch, or inside an allocation:
#   srun --jobid=<id> --overlap --gres=gpu:1 bash smoke_par.sh
source /gpfs/work4/0/prjs0951/Giacomo/LeWM/le-wm/jobs/snellius/common.sh
echo "running in job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi -L
t0=$(date +%s); stage $(ds_file); echo "=== staged to $LOCAL_DATASET_DIR in $(( $(date +%s)-t0 ))s"
run() {  # variant subdir extra...
  local v=$1 s=$2; shift 2
  $TRAIN $(variant_args $v) num_workers=${NW:-2} loader.prefetch_factor=2 output_model_name=$s subdir=$s \
    +trainer.limit_val_batches=5 wandb.enabled=false $SMOKE_EXTRA "$@" > $W/logs/smoke_${SLURM_JOB_ID}_$s.full 2>&1
  echo "=== $s exit $?: $(tr '\r' '\n' < $W/logs/smoke_${SLURM_JOB_ID}_$s.full | grep -E "it/s|s/it" | tail -1 | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]|loss[^,]*" | tr '\n' ' ')"
  grep -E "Error|Traceback" $W/logs/smoke_${SLURM_JOB_ID}_$s.full | grep -v Warning | tail -3
}
t0=$(date +%s)
PAR=${PAR:-4}; n=0
for v in LeWM-D LeWM-MSE-dFH-D LeWM-FH-D LeFWM-D; do
  run $v ${SP:-smoke}_$v trainer.max_epochs=1 +trainer.limit_train_batches=${STEPS:-100} ckpt_every_n_steps=${CK:-50} &
  n=$((n+1)); [ $((n % PAR)) -eq 0 ] && wait
done
wait; echo "=== 4 variants ($PAR at a time) took $(( $(date +%s)-t0 ))s"
echo "=== resume LeWM-D: $(ls $STABLEWM_HOME/checkpoints/${SP:-smoke}_LeWM-D/ | tr '\n' ' ')"
run LeWM-D ${SP:-smoke}_LeWM-D trainer.max_epochs=2 +trainer.limit_train_batches=${STEPS:-100} ckpt_every_n_steps=${CK:-50}
grep -E "resuming|Restoring|restored" $W/logs/smoke_${SLURM_JOB_ID}_${SP:-smoke}_LeWM-D.full | head -3
echo "=== smoke done at $(date)"
