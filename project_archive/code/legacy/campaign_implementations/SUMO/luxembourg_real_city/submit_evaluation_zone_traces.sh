#!/usr/bin/env bash
# Submit four CPU ray-tracing arrays, their merge jobs, and static held-out test sets.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/evaluation_zones
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-200}
FRAME_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_rssi_frame.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
TESTSET_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_testset.sh
ZONES=(ville_haute_300m belair_300m limpertsberg_s_300m hollerich_w_300m)

for crop in "${ZONES[@]}"; do
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    map_root=${ZONE_PARENT}/${crop}/map
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    scene=${map_root}/sionna/${crop}_scene.xml
    manifest=${map_root}/sionna/${crop}_scene_manifest.json

    for required in "${mobility}" "${scene}" "${manifest}"; do
        if [[ ! -s "${required}" ]]; then
            printf 'Missing required input: %s\nRun prepare_evaluation_zones.sh first.\n' "${required}" >&2
            exit 1
        fi
    done
    mkdir -p "${data_root}/logs" "${data_root}/rssi/shards" "${data_root}/testset"

    array_ids=()
    for offset in 0 1 2 3 4; do
        job_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${crop}-rssi" \
            --array="0-359%${MAX_CONCURRENT_PER_ARRAY}" \
            --output="${data_root}/logs/rssi-%A_%a.out" \
            --error="${data_root}/logs/rssi-%A_%a.err" \
            --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root},MOBILITY=${mobility},FRAME_OFFSET=${offset}" \
            "${FRAME_SCRIPT}")"
        array_ids+=("${job_id}")
        printf '%s frame offset %d: %s\n' "${crop}" "${offset}" "${job_id}"
    done

    dependency="afterok:$(IFS=:; echo "${array_ids[*]}")"
    merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name="${crop}-merge" \
        --output="${data_root}/logs/merge-%j.out" \
        --error="${data_root}/logs/merge-%j.err" \
        --dependency="${dependency}" \
        --export="ALL,DATA_ROOT=${data_root},MERGED_TRACE_OUTPUT=${merged},MIN_RSSI_DBM=-100" \
        "${MERGE_SCRIPT}")"
    testset_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name="${crop}-testset" \
        --output="${data_root}/logs/testset-%j.out" \
        --error="${data_root}/logs/testset-%j.err" \
        --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root}" \
        "${TESTSET_SCRIPT}")"

    printf '%s merge: %s\n' "${crop}" "${merge_id}"
    printf '%s test set: %s\n' "${crop}" "${testset_id}"
    printf '%s merged output: %s\n' "${crop}" "${merged}"
done
