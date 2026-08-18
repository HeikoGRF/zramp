#!/usr/bin/env bash
#SBATCH --job-name=wallis-merge
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers}

: "${MERGED_TRACE_OUTPUT:?MERGED_TRACE_OUTPUT must be set}"
MIN_RSSI_DBM=${MIN_RSSI_DBM:--100}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/merge_rssi_trace_shards.py \
    --shard-dir "${DATA_ROOT}/rssi/shards" \
    --pattern 'step_*.npz' \
    --output "${MERGED_TRACE_OUTPUT}" \
    --expected-start 0 \
    --expected-end 1799 \
    --min-rssi-dbm "${MIN_RSSI_DBM}"
