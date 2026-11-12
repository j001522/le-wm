#!/bin/bash
# Convert a downloaded .h5 dataset to .lance (CPU only; PushT ~16 min, cube ~24 min).
#   usage: sbatch to_lance.sbatch pusht_expert_train      (writes $STABLEWM_HOME/datasets/<name>.lance)
#SBATCH -p genoa -n 1 -c 8 -t 03:00:00 -J lewm_lance
#SBATCH -o /gpfs/work4/0/prjs0951/Giacomo/LeWM/logs/%x_%j.out
source /gpfs/work4/0/prjs0951/Giacomo/LeWM/le-wm/jobs/snellius/common.sh
python $J/to_lance.py $STABLEWM_HOME/datasets/$1.h5 $STABLEWM_HOME/datasets/$1.lance
