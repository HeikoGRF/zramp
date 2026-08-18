#!/usr/bin/env bash
# Submit all 1,800 Place Wallis frames and merge links with RSSI >= -100 dBm.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
FRAME_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_place_wallis_rssi_frame.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_place_wallis_merge.sh
MOBILITY=${MOBILITY:-${DATA_ROOT}/mobility/place_wallis_all_vehicles_0745_0815_1s_full1800.json}
SCENE=${REPO_ROOT}/SUMO/luxembourg_real_city/place_wallis/map/sionna/place_wallis_300m_scene.xml
MERGED_TRACE_OUTPUT=${MERGED_TRACE_OUTPUT:-${DATA_ROOT}/rssi/place_wallis_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz}

if [[ ! -s "${MOBILITY}" || ! -s "${SCENE}" ]]; then
    echo "Place Wallis inputs are missing; run:" >&2
    echo "  ${REPO_ROOT}/SUMO/luxembourg_real_city/prepare_place_wallis_inputs.sh" >&2
    exit 1
fi

mkdir -p "${DATA_ROOT}/logs" "${DATA_ROOT}/rssi/shards"

array_ids=()
for offset in 0 1 2 3 4; do
    job_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name=wallis-rssi \
        --array=0-359%200 \
        --output="${DATA_ROOT}/logs/rssi-%A_%a.out" \
        --error="${DATA_ROOT}/logs/rssi-%A_%a.err" \
        --export="ALL,DATA_ROOT=${DATA_ROOT},MOBILITY=${MOBILITY},FRAME_OFFSET=${offset}" \
        "${FRAME_SCRIPT}")"
    array_ids+=("${job_id}")
    echo "Place Wallis frame offset ${offset}: ${job_id}"
done

dependency="afterok:$(IFS=:; echo "${array_ids[*]}")"
merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
    --job-name=wallis-merge \
    --output="${DATA_ROOT}/logs/merge-%j.out" \
    --error="${DATA_ROOT}/logs/merge-%j.err" \
    --dependency="${dependency}" \
    --export="ALL,DATA_ROOT=${DATA_ROOT},MERGED_TRACE_OUTPUT=${MERGED_TRACE_OUTPUT},MIN_RSSI_DBM=-100" \
    "${MERGE_SCRIPT}")"

printf 'FRAME_ARRAY_JOB_IDS=%s\n' "$(IFS=,; echo "${array_ids[*]}")"
printf 'MERGE_JOB_ID=%s\n' "${merge_id}"
printf 'MERGED_TRACE_OUTPUT=%s\n' "${MERGED_TRACE_OUTPUT}"
