#!/bin/bash
#SBATCH --job-name=xmap-data
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal,gpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-data-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-data-%A_%a.err

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MAPS=(
  source_train_00_dense_grid source_train_00_dense_grid
  source_train_01_two_corridors source_train_01_two_corridors
  source_train_02_open_campus source_train_02_open_campus
  source_train_03_irregular_blocks source_train_03_irregular_blocks
  source_valid_00_ring_spokes source_valid_01_staggered
)
SEEDS=(1 2 1 2 1 2 1 2 1 1)
MAP_ID=${MAPS[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}
printf -v SEED_PAD '%02d' "$SEED"

ROOT=/home/hgraef/zramp-workspace
SCRATCH_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu
PY=$SCRATCH_ROOT/miniforge3/envs/radiodiff-replay/bin/python
SOURCE_ROOT=$ROOT/SUMO/policy_pretraining_maps
EXP_ROOT=$SCRATCH_ROOT/data/cross_map_policy_pretraining_v1
CASE_ROOT=$EXP_ROOT/cases/$MAP_ID/seed_$SEED_PAD
OUTPUT=$CASE_ROOT/policy_case.pt
LOCAL_ROOT=/tmp/hgraef_xmap_data_${SLURM_ARRAY_JOB_ID}_$TASK_ID

cleanup() {
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

export PYTHONPATH=$ROOT
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR=$LOCAL_ROOT/tmp
mkdir -p "$TMPDIR" "$CASE_ROOT" "$SCRATCH_ROOT/logs"
cd "$ROOT"

if [ -f "$OUTPUT" ]; then
  echo "Dataset already exists: $OUTPUT"
  exit 0
fi

"$PY" -u -m online_policy_learning.build_cross_map_policy_dataset \
  --source-manifest "$SOURCE_ROOT/$MAP_ID/source_manifest.json" \
  --mobility "$CASE_ROOT/mobility.json" \
  --measurements "$CASE_ROOT/structured_measurements.npz" \
  --output "$LOCAL_ROOT/policy_case.pt" \
  --checkpoints 50,100,200,350,550,750,1000 \
  --providers-per-receiver 6 --device cpu

rsync -a "$LOCAL_ROOT/policy_case.pt" "$OUTPUT"
rsync -a "$LOCAL_ROOT/policy_case.summary.json" "$CASE_ROOT/policy_case.summary.json"
echo "Completed dataset map=$MAP_ID seed=$SEED at $(date -Is)"
