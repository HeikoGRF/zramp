#!/usr/bin/env bash
#SBATCH --job-name=grid-ae-gain
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
LATENT_DIM=${LATENT_DIM:-512}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/support_acquisition_pretraining/grid_autoencoder_gain_v1_d${LATENT_DIM}}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/support_acquisition_pretraining/pretrain_grid_autoencoder_gain.py \
    --output "${OUTPUT_DIR}" \
    --seed "${SEED:-20260811}" \
    --grid-resolution 300 \
    --latent-dim "${LATENT_DIM}" \
    --base-channels "${BASE_CHANNELS:-8}" \
    --hidden-dim "${HIDDEN_DIM:-384}" \
    --groups-per-batch "${GROUPS_PER_BATCH:-4}" \
    --candidates-per-bank "${CANDIDATES_PER_BANK:-16}" \
    --training-cache-batches "${TRAINING_CACHE_BATCHES:-96}" \
    --validation-batches "${VALIDATION_BATCHES:-16}" \
    --stress-validation-batches "${STRESS_VALIDATION_BATCHES:-8}" \
    --ae-steps "${AE_STEPS:-2500}" \
    --ae-min-steps "${AE_MIN_STEPS:-1000}" \
    --ae-validation-every "${AE_VALIDATION_EVERY:-100}" \
    --ae-patience "${AE_PATIENCE:-12}" \
    --ae-batch-size "${AE_BATCH_SIZE:-16}" \
    --gain-steps "${GAIN_STEPS:-6000}" \
    --gain-min-steps "${GAIN_MIN_STEPS:-1500}" \
    --gain-validation-every "${GAIN_VALIDATION_EVERY:-100}" \
    --gain-patience "${GAIN_PATIENCE:-20}" \
    --ae-learning-rate "${AE_LEARNING_RATE:-0.0003}" \
    --gain-learning-rate "${GAIN_LEARNING_RATE:-0.0003}" \
    --finetune-steps "${FINETUNE_STEPS:-2000}" \
    --finetune-min-steps "${FINETUNE_MIN_STEPS:-500}" \
    --finetune-validation-every "${FINETUNE_VALIDATION_EVERY:-100}" \
    --finetune-patience "${FINETUNE_PATIENCE:-12}" \
    --finetune-learning-rate "${FINETUNE_LEARNING_RATE:-0.0001}" \
    --max-planes "${MAX_PLANES:-512}" \
    --max-bank-size "${MAX_BANK_SIZE:-24}" \
    --max-axes "${MAX_AXES:-96}" \
    --stress-max-planes "${STRESS_MAX_PLANES:-1024}" \
    --stress-max-bank-size "${STRESS_MAX_BANK_SIZE:-48}" \
    --stress-max-axes "${STRESS_MAX_AXES:-160}" \
    --sample-count-min "${SAMPLE_COUNT_MIN:-64}" \
    --sample-count-max "${SAMPLE_COUNT_MAX:-65536}" \
    --stress-sample-count-max "${STRESS_SAMPLE_COUNT_MAX:-1048576}" \
    --target-workers "${TARGET_WORKERS:-12}" \
    --encoder-microbatch "${ENCODER_MICROBATCH:-32}" \
    --device cpu
