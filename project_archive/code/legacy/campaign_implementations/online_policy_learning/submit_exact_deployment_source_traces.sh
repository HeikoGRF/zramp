#!/bin/bash
#SBATCH --job-name=xdep-trace
#SBATCH --account=disco-med
#SBATCH --partition=gpu.normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-9%10
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xdep-trace-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xdep-trace-%A_%a.err
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
printf -v SEED_PAD "%02d" "$SEED"
ROOT=/home/hgraef/zramp-workspace
SCRATCH_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu
PY=$SCRATCH_ROOT/miniforge3/envs/radiodiff-replay/bin/python
SOURCE_ROOT=$ROOT/SUMO/policy_pretraining_maps
INPUT_ROOT=$SCRATCH_ROOT/data/cross_map_policy_pretraining_v1/cases/$MAP_ID/seed_$SEED_PAD
EXP_ROOT=$SCRATCH_ROOT/data/cross_map_exact_deployment_policy_v1
CASE_ROOT=$EXP_ROOT/sources/$MAP_ID/seed_$SEED_PAD
LOCAL_ROOT=/tmp/hgraef_xdep_trace_${SLURM_ARRAY_JOB_ID}_$TASK_ID
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
export PYTHONPATH=$ROOT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$LOCAL_ROOT" "$CASE_ROOT" "$SCRATCH_ROOT/logs"
cd "$ROOT"
MANIFEST=$SOURCE_ROOT/$MAP_ID/source_manifest.json
MOBILITY=$INPUT_ROOT/mobility.json
STATIC=$LOCAL_ROOT/static_structured.npz
REPLAY=$LOCAL_ROOT/exact_replay.npz
CONTACTS=$LOCAL_ROOT/clear_contacts.npz
"$PY" SUMO/generate_structured_radio_trace.py \
  --source-manifest "$MANIFEST" --mobility "$MOBILITY" \
  --output "$STATIC" --static --opaque-buildings --steps 200
"$PY" cross_map_policy_generalization/prepare_policy_source_exact_trace.py \
  --source-manifest "$MANIFEST" --mobility "$MOBILITY" \
  --structured "$STATIC" --output "$REPLAY" --contacts "$CONTACTS" \
  --steps 200 --evaluation-pairs 2048 \
  --evaluation-unavailable-fraction 0.25 --seed "$((20260727 + SEED))"
rsync -a "$STATIC" "$CASE_ROOT/static_structured.npz"
rsync -a "$REPLAY" "$CASE_ROOT/exact_replay.npz"
rsync -a "$CONTACTS" "$CASE_ROOT/clear_contacts.npz"
echo "Completed exact source trace map=$MAP_ID seed=$SEED at $(date -Is)"
