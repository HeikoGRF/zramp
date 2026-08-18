#!/usr/bin/env bash
#SBATCH --job-name=zone-testset
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
SUMO_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie/map/sumo
MAP_SIZE_M=${MAP_SIZE_M:-300}
: "${CROP_NAME:?CROP_NAME must be set}"
: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${MAP_ROOT:?MAP_ROOT must be set}"

TASK_CACHE=/tmp/hgraef-drjit-testset-${SLURM_JOB_ID}
THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "${TASK_CACHE}" "${DATA_ROOT}/testset"
trap 'rm -rf -- "${TASK_CACHE}"' EXIT

export MI_DEFAULT_VARIANT=llvm_ad_rgb
export DRJIT_CACHE_DIR="${TASK_CACHE}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/generate_street_testset.py \
    --scene "${MAP_ROOT}/sionna/${CROP_NAME}_scene.xml" \
    --scene-manifest "${MAP_ROOT}/sionna/${CROP_NAME}_scene_manifest.json" \
    --sumo-net-3d "${SUMO_ROOT}/lust3d.net.xml" \
    --output "${TESTSET_OUTPUT:-${DATA_ROOT}/testset/${CROP_NAME}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz}" \
    --region-bounds 0 0 "${MAP_SIZE_M}" "${MAP_SIZE_M}" \
    --samples "${TESTSET_SAMPLES:-10000}" \
    --senders "${TESTSET_SENDERS:-200}" \
    --seed "${TESTSET_SEED:-1}" \
    --street-spacing-m 2 \
    --region-margin-m 0 \
    --min-distance-m 1 \
    --antenna-height-m 1.5 \
    --num-rays "${NUM_RAYS:-20000}" \
    --max-depth "${MAX_DEPTH:-3}" \
    --tx-batch-size "${TX_BATCH_SIZE:-20}" \
    --frequency-hz 3500000000 \
    --tx-power-dbm 23 \
    --rssi-min-dbm -100 \
    --rssi-max-dbm 0 \
    --disable-refraction
