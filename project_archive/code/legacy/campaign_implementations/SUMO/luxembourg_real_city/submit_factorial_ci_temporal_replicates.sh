#!/usr/bin/env bash
# Submit four additional matched temporal realizations for every factorial cell,
# their ray-traced radio traces, and ten non-random paper methods per realization.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
CI_DATA_ROOT=${CI_DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_ci_temporal5_paper_final_v1}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_cell_grid_ci_temporal5_paper_final_v1}
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones}
ORIGINAL_DATA_PARENT=${ORIGINAL_DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-20}
SCAN_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_ci_lust_full_day_scan.sh
SELECT_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_ci_select_temporal_windows.sh
MOBILITY_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_ci_temporal_mobility.sh
CHUNK_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_ci_factorial_zone_rssi_chunk.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${REPO_ROOT}/trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt
WINDOWS_TSV=${CI_DATA_ROOT}/selected_windows.tsv

ZONES=(
    factor_b1_v1_300m factor_b1_v2_300m factor_b1_v3_300m
    factor_b2_v1_300m factor_b2_v2_300m factor_b2_v3_300m
    factor_b3_v1_300m factor_b3_v2_300m factor_b3_v3_300m
)
REPLICATES=(2 3 4 5)
INTERVALS=(5 10 20 40 80)

for required in "${SCAN_SCRIPT}" "${SELECT_SCRIPT}" "${MOBILITY_SCRIPT}" \
    "${CHUNK_SCRIPT}" "${MERGE_SCRIPT}" "${SIM_SCRIPT}" "${BUNDLE}"; do
    test -s "${required}"
done
mkdir -p "${CI_DATA_ROOT}/logs" "${RESULTS_ROOT}"
manifest=${RESULTS_ROOT}/submitted_jobs.tsv
if [[ -e "${manifest}" ]]; then
    echo "Submission manifest already exists; refusing duplicate submission: ${manifest}" >&2
    exit 1
fi
printf 'zone\treplicate\tstage\tconfiguration\tjob_id\tdependency\toutput\n' > "${manifest}"

scan_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
    --output="${CI_DATA_ROOT}/logs/full-day-scan-%j.out" \
    --error="${CI_DATA_ROOT}/logs/full-day-scan-%j.err" \
    "${SCAN_SCRIPT}")
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    - - full_day_scan - "${scan_id}" - "${CI_DATA_ROOT}/full_day_scan" >> "${manifest}"

select_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
    --dependency="afterok:${scan_id}" \
    --output="${CI_DATA_ROOT}/logs/window-select-%j.out" \
    --error="${CI_DATA_ROOT}/logs/window-select-%j.err" \
    "${SELECT_SCRIPT}")
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    - - select_windows - "${select_id}" "afterok:${scan_id}" "${WINDOWS_TSV}" >> "${manifest}"

submit_simulation() {
    local zone=$1
    local replicate=$2
    local trace=$3
    local testset=$4
    local net=$5
    local dependency=$6
    local configuration=$7
    local method_export=$8
    local short=${zone%_300m}
    local result_dir=${RESULTS_ROOT}/methods/${short}/rep${replicate}/${configuration}_paper_final_full1800_eval50_tail10x25
    local common_export
    local job_id

    mkdir -p "${result_dir}"
    common_export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${trace},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${result_dir},SIM_STEPS=1799,REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CELL_GRID_CONFIDENCE=binary,CELL_GRID_MIN_INTENSITY=1"
    job_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name="ci-${short#factor_}-r${replicate}-${configuration}" \
        --output="${result_dir}/slurm-%j.out" \
        --error="${result_dir}/slurm-%j.err" \
        --dependency="${dependency}" \
        --export="${common_export},${method_export}" \
        "${SIM_SCRIPT}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${zone}" "${replicate}" simulation "${configuration}" "${job_id}" \
        "${dependency}" "${result_dir}" >> "${manifest}"
}

for zone in "${ZONES[@]}"; do
    short=${zone%_300m}
    map_root=${ZONE_PARENT}/${zone}/map
    testset=${ORIGINAL_DATA_PARENT}/${zone}_30min_opaque_buildings_no_vehicle_blockers/testset/${zone}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    net=${map_root}/sionna/${zone}_radio_bounds.net.xml
    test -s "${testset}"
    test -s "${net}"

    for replicate in "${REPLICATES[@]}"; do
        replicate_root=${CI_DATA_ROOT}/replicates/${zone}/rep${replicate}
        mobility=${replicate_root}/mobility/${zone}_rep${replicate}_all_vehicles_1s_full1800.json
        trace=${replicate_root}/rssi/${zone}_rep${replicate}_vehicles_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
        mkdir -p "${replicate_root}/logs" "${replicate_root}/rssi/shards"

        mobility_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="ci-${short#factor_}-r${replicate}-mob" \
            --output="${replicate_root}/logs/mobility-%j.out" \
            --error="${replicate_root}/logs/mobility-%j.err" \
            --dependency="afterok:${select_id}" \
            --export="ALL,CROP_NAME=${zone},REPLICATE=${replicate},CI_DATA_ROOT=${CI_DATA_ROOT},WINDOWS_TSV=${WINDOWS_TSV}" \
            "${MOBILITY_SCRIPT}")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${zone}" "${replicate}" mobility - "${mobility_id}" \
            "afterok:${select_id}" "${mobility}" >> "${manifest}"

        array_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="ci-${short#factor_}-r${replicate}-ray" \
            --array="0-179%${MAX_CONCURRENT_PER_ARRAY}" \
            --exclude="arton10,arton11" \
            --output="${replicate_root}/logs/rssi-%A_%a.out" \
            --error="${replicate_root}/logs/rssi-%A_%a.err" \
            --dependency="afterok:${mobility_id}" \
            --export="ALL,CROP_NAME=${zone},DATA_ROOT=${replicate_root},MAP_ROOT=${map_root},MOBILITY=${mobility},FRAME_STRIDE=10" \
            "${CHUNK_SCRIPT}")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${zone}" "${replicate}" raytrace - "${array_id}" \
            "afterok:${mobility_id}" "${replicate_root}/rssi/shards" >> "${manifest}"

        merge_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="ci-${short#factor_}-r${replicate}-merge" \
            --output="${replicate_root}/logs/merge-%j.out" \
            --error="${replicate_root}/logs/merge-%j.err" \
            --dependency="afterok:${array_id}" \
            --export="ALL,DATA_ROOT=${replicate_root},MERGED_TRACE_OUTPUT=${trace},MIN_RSSI_DBM=-100" \
            "${MERGE_SCRIPT}")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${zone}" "${replicate}" merge - "${merge_id}" \
            "afterok:${array_id}" "${trace}" >> "${manifest}"

        dependency="afterok:${merge_id}"
        submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
            "${dependency}" iso \
            "CELL_GRID_SUPPORT=1,BASELINE_MODE=local-only,METHOD_TAG=ci_rep${replicate}_local_only"
        submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
            "${dependency}" central \
            "CELL_GRID_SUPPORT=1,BASELINE_MODE=central,METHOD_TAG=ci_rep${replicate}_central"
        submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
            "${dependency}" full \
            "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=0,WEIGHTED_PULL_INTERVAL_STEPS=1,WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=ci_rep${replicate}_intensity_full"
        submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
            "${dependency}" top5 \
            "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=5,WEIGHTED_PULL_INTERVAL_STEPS=1,WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=ci_rep${replicate}_intensity_top5"
        submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
            "${dependency}" every1 \
            "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=1,WEIGHTED_PULL_INTERVAL_STEPS=1,WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=ci_rep${replicate}_intensity_every1"
        for interval in "${INTERVALS[@]}"; do
            submit_simulation "${zone}" "${replicate}" "${trace}" "${testset}" "${net}" \
                "${dependency}" "every${interval}" \
                "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=1,WEIGHTED_PULL_INTERVAL_STEPS=${interval},WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=ci_rep${replicate}_intensity_every${interval}"
        done
    done
done

echo "Submitted temporal CI pipeline. Manifest: ${manifest}"
