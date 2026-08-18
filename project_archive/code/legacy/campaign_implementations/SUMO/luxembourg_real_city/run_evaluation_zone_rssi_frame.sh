#!/usr/bin/env bash
#SBATCH --job-name=zone-rssi
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
SUMO_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie/map/sumo
MAP_SIZE_M=${MAP_SIZE_M:-300}

: "${CROP_NAME:?CROP_NAME must be set}"
: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${MAP_ROOT:?MAP_ROOT must be set}"
: "${MOBILITY:?MOBILITY must be set}"
: "${FRAME_OFFSET:?FRAME_OFFSET must be set}"
: "${SLURM_ARRAY_TASK_ID:?this worker must run as a Slurm array}"

FRAME_STRIDE=${FRAME_STRIDE:-5}
FRAME_STEP=$((SLURM_ARRAY_TASK_ID * FRAME_STRIDE + FRAME_OFFSET))
if (( FRAME_STEP > 1799 )); then
    exit 0
fi

OUTPUT=${TRACE_OUTPUT:-${DATA_ROOT}/rssi/shards/step_$(printf '%04d' "${FRAME_STEP}").npz}
if [[ -s "${OUTPUT}" ]]; then
    echo "already complete: ${OUTPUT}"
    exit 0
fi

NODE_COUNT="$(${PYTHON_BIN} -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["vehicle_ids"]))' "${MOBILITY}")"
TASK_CACHE=/tmp/hgraef-drjit-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p "${TASK_CACHE}" "${DATA_ROOT}/rssi/shards"
trap 'rm -rf -- "${TASK_CACHE}"' EXIT

export MI_DEFAULT_VARIANT=llvm_ad_rgb
export DRJIT_CACHE_DIR="${TASK_CACHE}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/generate_pilot_rssi_trace.py \
    --mobility "${MOBILITY}" \
    --scene "${MAP_ROOT}/sionna/${CROP_NAME}_scene.xml" \
    --scene-manifest "${MAP_ROOT}/sionna/${CROP_NAME}_scene_manifest.json" \
    --radio-net "${MAP_ROOT}/sionna/${CROP_NAME}_radio_bounds.net.xml" \
    --sumo-net-3d "${SUMO_ROOT}/lust3d.net.xml" \
    --output "${OUTPUT}" \
    --nodes "${NODE_COUNT}" \
    --steps 1799 \
    --start-step "${FRAME_STEP}" \
    --end-step "${FRAME_STEP}" \
    --num-zones 1 \
    --region-bounds 0 0 "${MAP_SIZE_M}" "${MAP_SIZE_M}" \
    --num-rays "${NUM_RAYS:-20000}" \
    --max-depth "${MAX_DEPTH:-3}" \
    --tx-batch-size "${TX_BATCH_SIZE:-20}" \
    --frequency-hz 3500000000 \
    --tx-power-dbm 23 \
    --rssi-min-dbm -120 \
    --rssi-max-dbm 0 \
    --fidelity-pairs 0 \
    --disable-refraction \
    --vehicle-type-file "${SUMO_ROOT}/vtypes.add.xml" \
    --vehicle-antenna-clearance-m 0.1 \
    --vehicle-roof-antennas
