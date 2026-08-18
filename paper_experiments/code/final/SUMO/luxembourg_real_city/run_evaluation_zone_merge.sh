#!/usr/bin/env bash
#SBATCH --job-name=zone-merge
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${MERGED_TRACE_OUTPUT:?MERGED_TRACE_OUTPUT must be set}"
MIN_RSSI_DBM=${MIN_RSSI_DBM:--100}
THREADS=${SLURM_CPUS_PER_TASK:-8}

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/merge_rssi_trace_shards.py \
    --shard-dir "${DATA_ROOT}/rssi/shards" \
    --pattern 'step_*.npz' \
    --output "${MERGED_TRACE_OUTPUT}" \
    --expected-start 0 \
    --expected-end 1799 \
    --min-rssi-dbm "${MIN_RSSI_DBM}"
