#!/usr/bin/env bash
#SBATCH --job-name=unit-square-gain
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/synthetic_unit_square_gain_v3_production_matched}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain_shared_frame_gain.py \
    --output "${OUTPUT_DIR}" \
    --seed "${SEED:-20260810}" \
    --max-steps "${MAX_STEPS:-15000}" \
    --min-steps "${MIN_STEPS:-4000}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-8}" \
    --validation-batches "${VALIDATION_BATCHES:-64}" \
    --validation-every "${VALIDATION_EVERY:-100}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-30}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA:-0.00001}" \
    --learning-rate "${LEARNING_RATE:-0.0005}" \
    --latent-dim "${LATENT_DIM:-64}" \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --max-planes "${MAX_PLANES:-512}" \
    --max-bank-size "${MAX_BANK_SIZE:-24}" \
    --grid-resolution "${GRID_RESOLUTION:-300}" \
    --grid-layout "${GRID_LAYOUT:-regular}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-16}" \
    --max-axes "${MAX_AXES:-96}" \
    --training-cache-batches "${TRAINING_CACHE_BATCHES:-1024}" \
    --max-sample-count "${MAX_SAMPLE_COUNT:-4096}" \
    --target-workers "${TARGET_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" \
    --gradient-clip-norm "${GRADIENT_CLIP_NORM:-0}" \
    --device cpu
