# Sourced by the Snellius job scripts in this folder. Everything lives under $B (project space):
#   $B/.venv               uv venv (py3.10, stable-worldmodel 0.1.1, stable-pretraining 0.1.8)
#   $B/pylib               extra packages installed with --target (tensorboard)
#   $B/stablewm/datasets   *.h5 (as downloaded) and *.lance (converted with to_lance.py; used for training)
#   $B/stablewm/checkpoints/<ds>_<variant>/  weights_epoch_N.pt, decoder_epoch_N.pt, last.ckpt (resume),
#                          tb/ (TensorBoard), spt_runs/ (links to stable-pretraining run dirs: metrics.csv)
#   $B/logs                slurm and per-run logs;  $B/spt  stable-pretraining run dirs;  $B/run  hydra outputs
B=/gpfs/work4/0/prjs0951/Giacomo/LeWM
W=$B
J=$B/le-wm/jobs/snellius
source $B/.venv/bin/activate
export STABLEWM_HOME=$B/stablewm
export HF_HOME=$B/hf
export SPT_CACHE_DIR=$B/spt                # stable-pretraining run dirs (default is ~/.cache)
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$B/logs
export PYTHONPATH=$B/le-wm:$B/pylib${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1           # no __pycache__ (the project space is inode-limited)
export TMPDIR=${TMPDIR:-$B/tmp}; mkdir -p $TMPDIR
TRAIN="python $B/le-wm/train.py"
mkdir -p $B/run $B/logs
cd $B/run                                  # hydra outputs/ land here
# stage the dataset on node-local NVMe (GPFS random reads are slow); falls back to GPFS if it does not fit
stage() {
  local need=$(du -cb $(printf "$STABLEWM_HOME/datasets/%s " "$@") -L | tail -1 | cut -f1)
  local free=$(( $(df -B1 --output=avail $TMPDIR | tail -1) ))
  if [ $free -lt $(( need + 5*1024**3 )) ]; then
    echo "stage: only $((free/1024**3)) GB free in $TMPDIR, reading datasets from GPFS"; return
  fi
  # swm.data.load_dataset(name, cache_dir) reads <cache_dir>/datasets/<name>
  mkdir -p $TMPDIR/swm/datasets
  for f in "$@"; do [ -e $TMPDIR/swm/datasets/$f ] || cp -rL $STABLEWM_HOME/datasets/$f $TMPDIR/swm/datasets/; done
  export LOCAL_DATASET_DIR=$TMPDIR/swm
}
# dataset (DS=pusht|cube) -> file name and hydra overrides. Training reads .lance (per-frame JPEG): the
# downloaded .h5 compresses pixels in 100-frame chunks, which made data loading ~8x slower than the GPU.
# Cube: the lance reader matches a keys_to_merge regex only against loaded keys (h5 matches all
# columns), so the proprio columns are listed explicitly, in the h5 reader's (alphabetical) order.
ds_file() { case ${DS:-pusht} in pusht) echo pusht_expert_train.lance ;; cube) echo cube_single_expert.lance ;; esac; }
ds_args() {
  case ${DS:-pusht} in
    pusht) echo "data=pusht data.dataset.name=$(ds_file)" ;;
    cube)  echo "data=ogb data.dataset.name=$(ds_file) data.dataset.keys_to_merge.proprio=[proprio_effector_pos,proprio_effector_yaw,proprio_gripper_contact,proprio_gripper_opening,proprio_gripper_vel,proprio_joint_pos,proprio_joint_vel]" ;;
  esac
}
# variant -> hydra overrides. All from scratch, LeWM recipe, D = detached visualisation decoder.
#   LeWM-D          LeWM + D
#   LeWM-MSE-dFH-D  LeWM + MSE + detached flow head + D
#   LeWM-FH-D       LeWM + flow head (no MSE; flow gradients reach the encoder) + D
#   LeWM-MSE-FH-D   LeWM + MSE + flow head, both training the encoder (flow target not detached, as in LeWM-FH-D) + D
#   LeFWM-D         flow inside the transformer + D
# (O, B, B2, A are the old short names)
variant_args() {
  local common="$(ds_args) decoder.enabled=true ckpt_epoch_interval=1"
  case $1 in
    O|LeWM-D)          echo "$common model=lewm" ;;
    B|LeWM-MSE-dFH-D)  echo "$common model=lewm_flow flow.mse_weight=1.0 flow.detach_head=true" ;;
    B2|LeWM-FH-D)      echo "$common model=lewm_flow flow.mse_weight=0.0 flow.detach_head=false flow.detach_target=false" ;;
    LeWM-MSE-FH-D)     echo "$common model=lewm_flow flow.mse_weight=1.0 flow.detach_head=false flow.detach_target=false" ;;
    A|LeFWM-D)         echo "$common model=lewm_flow_ar flow.detach_target=false flow.ctx_frac=0.9" ;;
  esac
}
