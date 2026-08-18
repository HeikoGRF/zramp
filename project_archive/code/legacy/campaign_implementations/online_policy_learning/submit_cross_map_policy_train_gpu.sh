#!/bin/bash
#SBATCH --job-name=xmap-policy
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal,gpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-policy-%j.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/xmap-policy-%j.err

set -euo pipefail

ROOT=/home/hgraef/zramp-workspace
SCRATCH_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu
PY=$SCRATCH_ROOT/miniforge3/envs/radiodiff-replay/bin/python
EXP_ROOT=$SCRATCH_ROOT/data/cross_map_policy_pretraining_v1
OUTPUT=$EXP_ROOT/trained_policy
LOCAL_ROOT=/tmp/hgraef_xmap_policy_${SLURM_JOB_ID}

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
mkdir -p "$TMPDIR" "$LOCAL_ROOT/output" "$OUTPUT" "$SCRATCH_ROOT/logs"
cd "$ROOT"

CASES=("$EXP_ROOT"/cases/*/seed_*/policy_case.pt)
if [ "${#CASES[@]}" -ne 10 ] || [ ! -f "${CASES[0]}" ]; then
  echo "Expected 10 policy cases, found ${#CASES[@]}" >&2
  exit 3
fi

"$PY" -u -m online_policy_learning.train_cross_map_policy \
  --cases "${CASES[@]}" --output-dir "$LOCAL_ROOT/output" \
  --device cpu --epochs 30 --hidden-dim 8 --gain-hidden-dim 64 \
  --embedding-dim 64 --pair-feature-mode relational

rsync -a "$LOCAL_ROOT/output/" "$OUTPUT/"
echo "Completed centralized source-map policy training at $(date -Is)"
