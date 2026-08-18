#!/usr/bin/env bash
#SBATCH --job-name=scalar-support-gain
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/synthetic_scalar_gain_v1_excessive}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain_scalar_gain.py \
    --output "${OUTPUT_DIR}" \
    --seed "${SEED:-20260810}" \
    --max-steps "${MAX_STEPS:-50000}" \
    --min-steps "${MIN_STEPS:-5000}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-8}" \
    --validation-batches "${VALIDATION_BATCHES:-64}" \
    --validation-every "${VALIDATION_EVERY:-250}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-24}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA:-0.00001}" \
    --learning-rate "${LEARNING_RATE:-0.0005}" \
    --latent-dim "${LATENT_DIM:-64}" \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --max-planes "${MAX_PLANES:-256}" \
    --max-bank-size "${MAX_BANK_SIZE:-20}" \
    --queries-per-world "${QUERIES_PER_WORLD:-512}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-8}" \
    --min-world-m "${MIN_WORLD_M:-40}" \
    --max-world-m "${MAX_WORLD_M:-1800}" \
    --max-axes "${MAX_AXES:-48}" \
    --max-sample-count "${MAX_SAMPLE_COUNT:-4096}" \
    --device cpu
