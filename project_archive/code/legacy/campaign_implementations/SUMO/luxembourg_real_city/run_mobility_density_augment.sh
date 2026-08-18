#!/usr/bin/env bash
#SBATCH --job-name=mobility-density
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
: "${SOURCE_MOBILITY:?SOURCE_MOBILITY must be set}"
: "${OUTPUT_MOBILITY:?OUTPUT_MOBILITY must be set}"
: "${DENSITY_FACTOR:?DENSITY_FACTOR must be set}"

if [[ -s "${OUTPUT_MOBILITY}" ]]; then
    echo "already complete: ${OUTPUT_MOBILITY}"
    exit 0
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/augment_mobility_density.py     --input "${SOURCE_MOBILITY}"     --output "${OUTPUT_MOBILITY}"     --factor "${DENSITY_FACTOR}"

