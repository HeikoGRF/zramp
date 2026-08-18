#!/usr/bin/env bash
# Submit a full-1800-step synchronized communication sweep on three maps with
# matched medium building density and high vehicle density.  The established
# factor_b2_v3 map reuses its radio trace and test set; the two new maps are
# prepared first and their simulations depend on successful asset generation.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
FACTOR_ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones
MATCHED_ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/matched_b2v3_zones
MATCHED_MANIFEST=${REPO_ROOT}/SUMO/luxembourg_real_city/matched_b2v3_zones_crop_manifest.json
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-200}
PREP_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_prepare.sh
CHUNK_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_rssi_chunk.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
TESTSET_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_testset.sh
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${BUNDLE:-${REPO_ROOT}/trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_cell_grid_synchronized_matched_b2v3_v1}
ZONES=(factor_b2_v3_300m matched_b2v3_a_300m matched_b2v3_b_300m)
INTERVALS=(5 10 20 40 80)

for required in "${PREP_SCRIPT}" "${CHUNK_SCRIPT}" "${MERGE_SCRIPT}" \
    "${TESTSET_SCRIPT}" "${SIM_SCRIPT}" "${MATCHED_MANIFEST}" "${BUNDLE}"; do
    test -s "${required}"
done

mkdir -p "${RESULTS_ROOT}"
submission_table=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_jobs.tsv}
if [[ "${APPEND_SUBMISSIONS:-0}" != "1" || ! -s "${submission_table}" ]]; then
    printf 'zone\tstage\tconfiguration\tjob_id\n' > "${submission_table}"
fi

submit_simulation() {
    local crop=$1
    local short=$2
    local trace=$3
    local testset=$4
    local net=$5
    local dependency=$6
    local run_name=$7
    local job_suffix=$8
    local method_export=$9
    local results_dir=${RESULTS_ROOT}/methods/${short}/${run_name}
    local common_export
    local sim_id
    local -a dependency_args=()

    mkdir -p "${results_dir}"
    if [[ -n "${dependency}" ]]; then
        dependency_args+=(--dependency="${dependency}")
    fi
    common_export="ALL,TRACE_PATH=${trace},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},SIM_STEPS=1799,REPLAY_CAPACITY=10000,FULL_DATASET_EPOCHS=1,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CELL_GRID_CONFIDENCE=binary,CELL_GRID_MIN_INTENSITY=1"
    sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name="${short}-${job_suffix}" \
        --output="${results_dir}/slurm-%j.out" \
        --error="${results_dir}/slurm-%j.err" \
        "${dependency_args[@]}" \
        --export="${common_export},${method_export}" \
        "${SIM_SCRIPT}")"
    printf '%s\tsimulation\t%s\t%s\n' \
        "${crop}" "${run_name}" "${sim_id}" >> "${submission_table}"
}

for crop in "${ZONES[@]}"; do
    short=${crop%_300m}
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    testset=${data_root}/testset/${crop}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    sim_dependency=

    if [[ "${crop}" == factor_b2_v3_300m ]]; then
        map_root=${FACTOR_ZONE_PARENT}/${crop}/map
        for reusable in "${mobility}" "${merged}" "${testset}" \
            "${map_root}/sionna/${crop}_radio_bounds.net.xml"; do
            test -s "${reusable}"
        done
    else
        map_root=${MATCHED_ZONE_PARENT}/${crop}/map
        mkdir -p "${data_root}/logs" "${data_root}/rssi/shards" \
            "${data_root}/testset"

        prep_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-prep" \
            --output="${data_root}/logs/prepare-%j.out" \
            --error="${data_root}/logs/prepare-%j.err" \
            --export="ALL,CROP_NAME=${crop},DATA_PARENT=${DATA_PARENT},ZONE_PARENT=${MATCHED_ZONE_PARENT},MANIFEST=${MATCHED_MANIFEST}" \
            "${PREP_SCRIPT}")"
        printf '%s\tprep\t-\t%s\n' "${crop}" "${prep_id}" >> "${submission_table}"

        array_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-rssi" \
            --array="0-359%${MAX_CONCURRENT_PER_ARRAY}" \
            --exclude="arton10,arton11" \
            --output="${data_root}/logs/rssi-chunk-%A_%a.out" \
            --error="${data_root}/logs/rssi-chunk-%A_%a.err" \
            --dependency="afterok:${prep_id}" \
            --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root},MOBILITY=${mobility}" \
            "${CHUNK_SCRIPT}")"
        printf '%s\trssi_chunks\t-\t%s\n' "${crop}" "${array_id}" >> "${submission_table}"

        merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-merge" \
            --output="${data_root}/logs/merge-%j.out" \
            --error="${data_root}/logs/merge-%j.err" \
            --dependency="afterok:${array_id}" \
            --export="ALL,DATA_ROOT=${data_root},MERGED_TRACE_OUTPUT=${merged},MIN_RSSI_DBM=-100" \
            "${MERGE_SCRIPT}")"
        printf '%s\tmerge\t-\t%s\n' "${crop}" "${merge_id}" >> "${submission_table}"

        testset_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-test" \
            --exclude="arton10,arton11" \
            --output="${data_root}/logs/testset-%j.out" \
            --error="${data_root}/logs/testset-%j.err" \
            --dependency="afterok:${prep_id}" \
            --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root}" \
            "${TESTSET_SCRIPT}")"
        printf '%s\ttestset\t-\t%s\n' "${crop}" "${testset_id}" >> "${submission_table}"
        sim_dependency="afterok:${merge_id}:${testset_id}"
    fi

    net=${map_root}/sionna/${crop}_radio_bounds.net.xml
    mkdir -p "${RESULTS_ROOT}/methods/${short}"

    submit_simulation "${crop}" "${short}" "${merged}" "${testset}" "${net}" \
        "${sim_dependency}" \
        cell_grid_local_only_sync_full1800_eval50_tail10x25 iso \
        "CELL_GRID_SUPPORT=1,BASELINE_MODE=local-only,METHOD_TAG=local_only_sync_full1800"
    submit_simulation "${crop}" "${short}" "${merged}" "${testset}" "${net}" \
        "${sim_dependency}" \
        cell_grid_central_sync_full1800_eval50_tail10x25 central \
        "CELL_GRID_SUPPORT=1,BASELINE_MODE=central,METHOD_TAG=central_sync_full1800"
    submit_simulation "${crop}" "${short}" "${merged}" "${testset}" "${net}" \
        "${sim_dependency}" \
        cell_grid_intensity_greedy_sync_full1800_eval50_tail10x25 greedy \
        "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=0,WEIGHTED_PULL_INTERVAL_STEPS=1,WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=intensity_greedy_sync_full1800"

    for interval in "${INTERVALS[@]}"; do
        submit_simulation "${crop}" "${short}" "${merged}" "${testset}" "${net}" \
            "${sim_dependency}" \
            "cell_grid_intensity_top1_global_every${interval}_full1800_eval50_tail10x25" \
            "e${interval}" \
            "CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=1,WEIGHTED_PULL_INTERVAL_STEPS=${interval},WEIGHTED_PULL_SCHEDULE_ANCHOR=global,METHOD_TAG=intensity_top1_global_every${interval}_full1800"
    done
    printf '%s simulations=8 dependency=%s\n' "${crop}" "${sim_dependency:-none}"
done

printf 'Submission manifest: %s\n' "${submission_table}"
