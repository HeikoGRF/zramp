#!/usr/bin/env bash
#SBATCH --job-name=support-eb-k4
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/support_expert_bank/replay10_capsule_baseline_k4_cost0}

mkdir -p "${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
RUN_ARGS=(
    --results-dir "${RESULTS_DIR}"
    --bank-capacity "${BANK_CAPACITY:-4}"
    --transfer-cost 0
    --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    --progress-every "${PROGRESS_EVERY:-10}"
    --resume-if-exists
    --quiet
)
if [[ -n "${SIM_STEPS:-}" ]]; then
    RUN_ARGS+=(--sim-steps "${SIM_STEPS}")
fi
"${PYTHON_BIN}" experiments/support_expert_bank/run_support_expert_bank.py "${RUN_ARGS[@]}"
