#!/usr/bin/env bash
#SBATCH --job-name=rbf-greedy
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/rbf_greedy/pair_rbf_sigma2_merge3}

mkdir -p "${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
RUN_ARGS=(
    --results-dir "${RESULTS_DIR}"
    --checkpoint-every 50
    --progress-every 10
    --resume-if-exists
    --quiet
)
if [[ -n "${SIM_STEPS:-}" ]]; then
    RUN_ARGS+=(--sim-steps "${SIM_STEPS}")
fi
for setting in \
    RBF_SIGMA_M:rbf-sigma-m \
    PROTOTYPE_MERGE_RADIUS_M:prototype-merge-radius-m
do
    env_name=${setting%%:*}
    arg_name=${setting#*:}
    value=${!env_name:-}
    if [[ -n "${value}" ]]; then
        RUN_ARGS+=("--${arg_name}" "${value}")
    fi
done
"${PYTHON_BIN}" experiments/rbf_greedy/run_rbf_greedy.py "${RUN_ARGS[@]}"
