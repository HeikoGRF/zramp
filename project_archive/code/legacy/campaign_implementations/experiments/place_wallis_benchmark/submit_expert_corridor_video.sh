#!/usr/bin/env bash
#SBATCH --job-name=wallis-corridor-video
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/artifacts/place_wallis_benchmark/visualizations/expert_corridor_growth_k6_vehicle517}

mkdir -p "${OUTPUT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" experiments/place_wallis_benchmark/visualize_vehicle_expert_growth.py \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
