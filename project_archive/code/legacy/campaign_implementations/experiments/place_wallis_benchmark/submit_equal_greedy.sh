#!/usr/bin/env bash
#SBATCH --job-name=wallis-equal-greedy
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=arton10,arton11

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/place_wallis_benchmark/methods/equal_greedy_eval50_tail10x25}

mkdir -p "${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
RUN_ARGS=(
    --results-dir "${RESULTS_DIR}"
    --method-tag "${METHOD_TAG:-}"
    --seed "${SEED:-1}"
    --local-lr "${LOCAL_LR:-5.0e-4}"
    --local-batch-size "${LOCAL_BATCH_SIZE:-64}"
    --replay-capacity "${REPLAY_CAPACITY:-4096}"
    --new-data-epochs "${NEW_DATA_EPOCHS:-2}"
    --replay-batches "${REPLAY_BATCHES:-8}"
    --recent-replay-batches "${RECENT_REPLAY_BATCHES:-4}"
    --recent-window "${RECENT_WINDOW:-512}"
    --full-dataset-epochs "${FULL_DATASET_EPOCHS:-0}"
    --gradient-clip-norm "${GRADIENT_CLIP_NORM:-1.0}"
    --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    --tail-eval-count "${TAIL_EVAL_COUNT:-10}"
    --tail-eval-stride "${TAIL_EVAL_STRIDE:-25}"
    --progress-every "${PROGRESS_EVERY:-10}"
    --resume-if-exists
    --quiet
)
if [[ -n "${TRACE_PATH:-}" ]]; then
    RUN_ARGS+=(--trace "${TRACE_PATH}")
fi
if [[ -n "${TESTSET_PATH:-}" ]]; then
    RUN_ARGS+=(--testset "${TESTSET_PATH}")
fi
if [[ -n "${NET_PATH:-}" ]]; then
    RUN_ARGS+=(--net "${NET_PATH}")
fi
if [[ -n "${SIM_STEPS:-}" ]]; then
    RUN_ARGS+=(--sim-steps "${SIM_STEPS}")
fi
"${PYTHON_BIN}" experiments/place_wallis_benchmark/run_equal_greedy.py "${RUN_ARGS[@]}"
