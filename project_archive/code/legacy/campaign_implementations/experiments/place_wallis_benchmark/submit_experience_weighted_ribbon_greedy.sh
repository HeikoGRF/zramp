#!/usr/bin/env bash
#SBATCH --job-name=wallis-ribbon-exp
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/place_wallis_benchmark/methods/experience_weighted_ribbon_greedy_eval50_tail10x25}

mkdir -p "${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
RUN_ARGS=(
    --results-dir "${RESULTS_DIR}"
    --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    --tail-eval-count "${TAIL_EVAL_COUNT:-10}"
    --tail-eval-stride "${TAIL_EVAL_STRIDE:-25}"
    --progress-every "${PROGRESS_EVERY:-10}"
    --experience-weighted
    --resume-if-exists
    --quiet
)
if [[ -n "${SIM_STEPS:-}" ]]; then
    RUN_ARGS+=(--sim-steps "${SIM_STEPS}")
fi
"${PYTHON_BIN}" experiments/place_wallis_benchmark/run_capsule_greedy.py "${RUN_ARGS[@]}"
