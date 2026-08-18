#!/usr/bin/env bash
#SBATCH --job-name=patch-grid-codec
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
LATENT_CHANNELS=${LATENT_CHANNELS:-4}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/patch_grid_codec_v1_c${LATENT_CHANNELS}}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${REPO_ROOT}"
EXTRA_ARGS=()
if [[ -n "${AUTOENCODER_STATE:-}" ]]; then
    EXTRA_ARGS+=(--autoencoder-state "${AUTOENCODER_STATE}")
fi
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain_patch_grid_codec.py \
    "${EXTRA_ARGS[@]}" \
    --output "${OUTPUT_DIR}" \
    --support-profile-kind "${SUPPORT_PROFILE_KIND:-plane-envelope}" \
    --seed "${SEED:-20260811}" \
    --patch-size "${PATCH_SIZE:-10}" \
    --latent-channels "${LATENT_CHANNELS}" \
    --codebook-size "${CODEBOOK_SIZE:-0}" \
    --codebook-groups "${CODEBOOK_GROUPS:-1}" \
    --codebook-maximum-codes "${CODEBOOK_MAXIMUM_CODES:-50000}" \
    --codebook-iterations "${CODEBOOK_ITERATIONS:-40}" \
    --hidden-dim "${HIDDEN_DIM:-64}" \
    --acquisition-hidden-dim "${ACQUISITION_HIDDEN_DIM:-96}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-4}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-16}" \
    --training-cache-batches "${TRAINING_CACHE_BATCHES:-96}" \
    --validation-batches "${VALIDATION_BATCHES:-16}" \
    --stress-validation-batches "${STRESS_VALIDATION_BATCHES:-8}" \
    --ae-steps "${AE_STEPS:-3000}" \
    --ae-min-steps "${AE_MIN_STEPS:-1000}" \
    --ae-validation-every "${AE_VALIDATION_EVERY:-100}" \
    --ae-patience "${AE_PATIENCE:-15}" \
    --ae-batch-size "${AE_BATCH_SIZE:-16}" \
    --gain-steps "${GAIN_STEPS:-3000}" \
    --gain-min-steps "${GAIN_MIN_STEPS:-750}" \
    --gain-validation-every "${GAIN_VALIDATION_EVERY:-100}" \
    --gain-patience "${GAIN_PATIENCE:-12}" \
    --gain-pairs-per-bin "${GAIN_PAIRS_PER_BIN:-16}" \
    --gain-natural-pairs "${GAIN_NATURAL_PAIRS:-96}" \
    --ae-learning-rate "${AE_LEARNING_RATE:-0.0003}" \
    --gain-learning-rate "${GAIN_LEARNING_RATE:-0.0001}" \
    --max-planes "${MAX_PLANES:-512}" \
    --max-bank-size "${MAX_BANK_SIZE:-24}" \
    --max-axes "${MAX_AXES:-96}" \
    --stress-max-planes "${STRESS_MAX_PLANES:-1024}" \
    --stress-max-bank-size "${STRESS_MAX_BANK_SIZE:-48}" \
    --stress-max-axes "${STRESS_MAX_AXES:-160}" \
    --sample-count-min "${SAMPLE_COUNT_MIN:-64}" \
    --sample-count-max "${SAMPLE_COUNT_MAX:-1048576}" \
    --stress-sample-count-max "${STRESS_SAMPLE_COUNT_MAX:-16777216}" \
    --target-workers "${TARGET_WORKERS:-12}" \
    --device cpu
