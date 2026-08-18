#!/usr/bin/env bash
#SBATCH --job-name=synth-support-acq-v2
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/synthetic_v2_normalized}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain.py \
    --output "${OUTPUT_DIR}" \
    --seed "${SEED:-20260809}" \
    --steps "${TRAIN_STEPS:-6000}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-8}" \
    --validation-batches "${VALIDATION_BATCHES:-48}" \
    --validation-every "${VALIDATION_EVERY:-200}" \
    --learning-rate "${LEARNING_RATE:-0.0008}" \
    --latent-dim "${LATENT_DIM:-64}" \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --max-planes "${MAX_PLANES:-256}" \
    --max-bank-size "${MAX_BANK_SIZE:-15}" \
    --queries-per-world "${QUERIES_PER_WORLD:-512}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-6}" \
    --min-world-m "${MIN_WORLD_M:-50}" \
    --max-world-m "${MAX_WORLD_M:-1200}" \
    --max-axes "${MAX_AXES:-36}" \
    --normalization-version v2 \
    --device cpu
