#!/usr/bin/env bash
#SBATCH --job-name=wallis-expert-bank
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=arton10,arton11

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
BANK_CAPACITY=${BANK_CAPACITY:-6}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/place_wallis_benchmark/methods/support_expert_bank_k${BANK_CAPACITY}_eval50_tail10x25}

mkdir -p "${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
RUN_ARGS=(
    --results-dir "${RESULTS_DIR}"
    --method-tag "${METHOD_TAG:-}"
    --bank-capacity "${BANK_CAPACITY}"
    --transfer-cost "${TRANSFER_COST:-0}"
    --probe-count "${PROBE_COUNT:-512}"
    --replay-capacity "${REPLAY_CAPACITY:-0}"
    --full-dataset-epochs "${FULL_DATASET_EPOCHS:-1}"
    --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    --tail-eval-count "${TAIL_EVAL_COUNT:-10}"
    --tail-eval-stride "${TAIL_EVAL_STRIDE:-25}"
    --progress-every "${PROGRESS_EVERY:-10}"
    --bank-support-routing "${BANK_SUPPORT_ROUTING:-individual}"
    --teacher-distillation-batches-per-step "${TEACHER_DISTILLATION_BATCHES_PER_STEP:-0}"
    --angle-deg "${PLANE_ANGLE_DEG:-7}"
    --lateral-merge-m "${PLANE_LATERAL_GAP_M:-1}"
    --longitudinal-gap-m "${PLANE_LONGITUDINAL_GAP_M:-3}"
    --initial-half-width-m "${PLANE_INITIAL_HALF_WIDTH_M:-1.75}"
    --max-envelope-inflation "${PLANE_MAX_ENVELOPE_INFLATION:-1.2}"
    --max-corridor-width-m "${PLANE_MAX_CORRIDOR_WIDTH_M:-12}"
    --link-length-margin-m "${PLANE_LINK_LENGTH_MARGIN_M:-0}"
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
if [[ "${DOMINANCE_PRUNED:-0}" == "1" ]]; then
    RUN_ARGS+=(--dominance-pruned)
    RUN_ARGS+=(--min-unique-coverage "${MIN_UNIQUE_COVERAGE:-0.005}")
fi
if [[ -n "${LEARNED_ACQUISITION_BUNDLE:-}" ]]; then
    RUN_ARGS+=(--learned-acquisition-bundle "${LEARNED_ACQUISITION_BUNDLE}")
    RUN_ARGS+=(--acquisition-probability-threshold "${ACQUISITION_PROBABILITY_THRESHOLD:-0.5}")
    RUN_ARGS+=(--acquisition-relative-gain-penalty "${ACQUISITION_RELATIVE_GAIN_PENALTY:-0}")
fi
if [[ "${CELL_GRID_SUPPORT:-0}" == "1" ]]; then
    RUN_ARGS+=(--cell-grid-support)
    RUN_ARGS+=(--cell-grid-confidence "${CELL_GRID_CONFIDENCE:-binary}")
    RUN_ARGS+=(--cell-grid-min-intensity "${CELL_GRID_MIN_INTENSITY:-1}")
fi
if [[ "${CELL_GRID_WEIGHTED_SINGLE:-0}" == "1" ]]; then
    RUN_ARGS+=(--cell-grid-support --cell-grid-weighted-single)
    RUN_ARGS+=(--cell-grid-confidence "${CELL_GRID_CONFIDENCE:-binary}")
    RUN_ARGS+=(--cell-grid-min-intensity "${CELL_GRID_MIN_INTENSITY:-1}")
    RUN_ARGS+=(--weighted-pulls-per-receiver-step "${WEIGHTED_PULLS_PER_RECEIVER_STEP:-1}")
    RUN_ARGS+=(--weighted-pull-interval-steps "${WEIGHTED_PULL_INTERVAL_STEPS:-1}")
    RUN_ARGS+=(--weighted-pull-schedule-anchor "${WEIGHTED_PULL_SCHEDULE_ANCHOR:-entry}")
    RUN_ARGS+=(--weighted-selection "${WEIGHTED_SELECTION:-experience}")
    if [[ "${CELL_GRID_WEIGHTED_ACQUISITION:-0}" == "1" ]]; then
        RUN_ARGS+=(--cell-grid-weighted-acquisition)
        if [[ "${CELL_GRID_WEIGHTED_ACQUISITION_FIXED_BUDGET:-0}" == "1" ]]; then
            RUN_ARGS+=(--cell-grid-weighted-acquisition-fixed-budget)
        fi
    fi
fi
if [[ -n "${BASELINE_MODE:-}" ]]; then
    RUN_ARGS+=(--baseline-mode "${BASELINE_MODE}")
fi
"${PYTHON_BIN}" experiments/place_wallis_benchmark/run_support_expert_bank.py "${RUN_ARGS[@]}"
