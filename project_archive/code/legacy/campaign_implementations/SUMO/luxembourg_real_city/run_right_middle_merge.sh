#!/usr/bin/env bash
#SBATCH --job-name=bonnevoie-merge
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/merge-%j.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/merge-%j.err

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/merge_rssi_trace_shards.py \
    --shard-dir "${DATA_ROOT}/rssi/shards" \
    --pattern 'step_*.npz' \
    --output "${MERGED_TRACE_OUTPUT:-${DATA_ROOT}/rssi/gare_bonnevoie_vehicles_0745_0815_1s_right_middle_opaque_no_vehicle_blockers_r20k_d3_llvm.npz}" \
    --expected-start 0 \
    --expected-end 1799
