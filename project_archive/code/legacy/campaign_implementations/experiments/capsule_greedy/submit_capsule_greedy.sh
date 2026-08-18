#!/usr/bin/env bash
#SBATCH --job-name=capsule-greedy
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/artifacts/capsule_greedy/opaque_no_vehicle_blockers}

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
if [[ "${CAPSULE_PROFILE:-strict}" == "relaxed-streets" ]]; then
    RUN_ARGS+=(
        --angle-deg 15
        --lateral-merge-m 12
        --longitudinal-gap-m 20
        --sigma-perp-m 8
        --sigma-parallel-m 15
        --sigma-angle-deg 15
    )
fi
for setting in \
    ANGLE_DEG:angle-deg \
    LATERAL_MERGE_M:lateral-merge-m \
    LONGITUDINAL_GAP_M:longitudinal-gap-m \
    SIGMA_PERP_M:sigma-perp-m \
    SIGMA_PARALLEL_M:sigma-parallel-m \
    SIGMA_ANGLE_DEG:sigma-angle-deg
do
    env_name=${setting%%:*}
    arg_name=${setting#*:}
    value=${!env_name:-}
    if [[ -n "${value}" ]]; then
        RUN_ARGS+=("--${arg_name}" "${value}")
    fi
done
"${PYTHON_BIN}" experiments/capsule_greedy/run_capsule_greedy.py "${RUN_ARGS[@]}"
