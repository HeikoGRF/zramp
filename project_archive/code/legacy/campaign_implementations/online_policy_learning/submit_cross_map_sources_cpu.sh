#!/bin/bash
#SBATCH --job-name=xmap-source
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal,gpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-9%5
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-source-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-source-%A_%a.err

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
ENV_PREFIX=$SCRATCH_ROOT/miniforge3/envs/sionna-trace
PY=$ENV_PREFIX/bin/python
SOURCE_ROOT=$ROOT/SUMO/policy_pretraining_maps
EXP_ROOT=$SCRATCH_ROOT/data/cross_map_policy_pretraining_v1
CASE_ROOT=$EXP_ROOT/cases/$MAP_ID/seed_$SEED_PAD
LOCAL_ROOT=/tmp/hgraef_xmap_source_${SLURM_ARRAY_JOB_ID}_$TASK_ID

cleanup() {
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

export SUMO_HOME=$ENV_PREFIX/lib/python3.10/site-packages/sumo
export PATH=$ENV_PREFIX/bin:$PATH
export PYTHONPATH=$ROOT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$LOCAL_ROOT" "$CASE_ROOT" "$SCRATCH_ROOT/logs"
cd "$ROOT"

MANIFEST=$SOURCE_ROOT/$MAP_ID/source_manifest.json
ROUTES=$CASE_ROOT/roles.rou.xml
PLAN=$CASE_ROOT/roles.json
CONFIG=$CASE_ROOT/roles.sumocfg
MOBILITY=$CASE_ROOT/mobility.json
MEASUREMENTS=$CASE_ROOT/structured_measurements.npz

if [ ! -f "$MOBILITY" ]; then
  "$PY" SUMO/build_policy_pretraining_routes.py \
    --source-manifest "$MANIFEST" --routes "$ROUTES" --plan "$PLAN" \
    --config "$CONFIG" --seed "$SEED" --num-vehicles 24 \
    --regular-count 4 --steps 1000
  "$PY" SUMO/export_role_mobility_trace_150.py \
    --config "$CONFIG" --plan "$PLAN" \
    --output "$LOCAL_ROOT/mobility.json" --steps 1000
  rsync -a "$LOCAL_ROOT/mobility.json" "$MOBILITY"
fi

if [ ! -f "$MEASUREMENTS" ]; then
  "$PY" SUMO/generate_structured_radio_trace.py \
    --source-manifest "$MANIFEST" --mobility "$MOBILITY" \
    --output "$LOCAL_ROOT/structured_measurements.npz"
  rsync -a "$LOCAL_ROOT/structured_measurements.npz" "$MEASUREMENTS"
fi

echo "Completed source map=$MAP_ID seed=$SEED at $(date -Is)"
