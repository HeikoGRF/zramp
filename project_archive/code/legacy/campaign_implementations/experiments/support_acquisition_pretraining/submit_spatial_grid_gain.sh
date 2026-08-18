#!/usr/bin/env bash
#SBATCH --job-name=spatial-grid-gain
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/spatial_grid_gain_v1_300}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain_spatial_grid_gain.py \
    --output "${OUTPUT_DIR}" \
    --seed "${SEED:-20260811}" \
    --max-steps "${MAX_STEPS:-3000}" \
    --min-steps "${MIN_STEPS:-750}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-8}" \
    --training-cache-batches "${TRAINING_CACHE_BATCHES:-128}" \
    --validation-batches "${VALIDATION_BATCHES:-16}" \
    --validation-every "${VALIDATION_EVERY:-25}" \
    --patience "${PATIENCE:-30}" \
    --learning-rate "${LEARNING_RATE:-0.0003}" \
    --grid-resolution 300 \
    --spatial-size "${SPATIAL_SIZE:-16}" \
    --learned-channels "${LEARNED_CHANNELS:-2}" \
    --hidden-channels "${HIDDEN_CHANNELS:-24}" \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --maximum-relative-gain "${MAXIMUM_RELATIVE_GAIN:-4}" \
    --max-planes "${MAX_PLANES:-512}" \
    --max-bank-size "${MAX_BANK_SIZE:-24}" \
    --max-axes "${MAX_AXES:-96}" \
    --max-sample-count "${MAX_SAMPLE_COUNT:-4096}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-16}" \
    --target-workers "${TARGET_WORKERS:-4}" \
    --device cpu
