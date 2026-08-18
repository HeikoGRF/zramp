#!/usr/bin/env bash
#SBATCH --job-name=bonnevoie-rssi
#SBATCH --account=disco-med
#SBATCH --partition=gpu.normal
#SBATCH --exclude=arton10,arton11
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --array=0-359%200
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/rssi-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/rssi-%A_%a.err

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers}
MAP_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie/map
MOBILITY=${MOBILITY:-${DATA_ROOT}/mobility/gare_bonnevoie_all_vehicles_0745_0815_1s_full1800.json}

: "${FRAME_OFFSET:?submit with FRAME_OFFSET=0,1,2,3,or 4}"
FRAME_STEP=$((SLURM_ARRAY_TASK_ID * 5 + FRAME_OFFSET))
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
mkdir -p "${TASK_CACHE}" "${DATA_ROOT}/rssi/shards"

export MI_DEFAULT_VARIANT=llvm_ad_rgb
export DRJIT_CACHE_DIR="${TASK_CACHE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/generate_pilot_rssi_trace.py \
    --mobility "${MOBILITY}" \
    --scene "${MAP_ROOT}/sionna/gare_bonnevoie_balanced_scene.xml" \
    --scene-manifest "${MAP_ROOT}/sionna/gare_bonnevoie_balanced_scene_manifest.json" \
    --radio-net "${MAP_ROOT}/sionna/gare_bonnevoie_balanced_radio_bounds.net.xml" \
    --sumo-net-3d "${MAP_ROOT}/sumo/lust3d.net.xml" \
    --output "${OUTPUT}" \
    --nodes "${NODE_COUNT}" \
    --steps 1799 \
    --start-step "${FRAME_STEP}" \
    --end-step "${FRAME_STEP}" \
    --num-zones 1 \
    --region-bounds 400 200 800 600 \
    --num-rays "${NUM_RAYS:-20000}" \
    --max-depth "${MAX_DEPTH:-3}" \
    --tx-batch-size "${TX_BATCH_SIZE:-20}" \
    --frequency-hz 3500000000 \
    --tx-power-dbm 23 \
    --rssi-min-dbm -120 \
    --rssi-max-dbm 0 \
    --fidelity-pairs 0 \
    --disable-refraction \
    --vehicle-type-file "${MAP_ROOT}/sumo/vtypes.add.xml" \
    --vehicle-antenna-clearance-m 0.1 \
    --vehicle-roof-antennas
