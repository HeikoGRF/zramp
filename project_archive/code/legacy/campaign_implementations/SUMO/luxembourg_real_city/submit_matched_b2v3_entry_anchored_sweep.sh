#!/usr/bin/env bash
# Submit the entry-anchored counterparts to the synchronized Every-k sweep.
# Every vehicle pulls immediately on entry and then maintains its own k-step
# counter. ISO, central, and greedy are shared with the synchronized sweep.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
FACTOR_ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones
MATCHED_ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/matched_b2v3_zones
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${BUNDLE:-${REPO_ROOT}/trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_cell_grid_synchronized_matched_b2v3_v1}
SUBMISSION_TABLE=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_jobs.tsv}
ZONES=(factor_b2_v3_300m matched_b2v3_a_300m matched_b2v3_b_300m)
INTERVALS=(5 10 20 40 80)

for required in "${SIM_SCRIPT}" "${BUNDLE}" "${SUBMISSION_TABLE}"; do
    test -s "${required}"
done

dependency_job() {
    local crop=$1
    local stage=$2
    awk -F '\t' -v crop="${crop}" -v stage="${stage}" \
        '$1 == crop && $2 == stage { value=$4 } END { print value }' \
        "${SUBMISSION_TABLE}"
}

for crop in "${ZONES[@]}"; do
    short=${crop%_300m}
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    testset=${data_root}/testset/${crop}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    dependency=
    dependency_args=()

    if [[ "${crop}" == factor_b2_v3_300m ]]; then
        map_root=${FACTOR_ZONE_PARENT}/${crop}/map
        for reusable in "${mobility}" "${merged}" "${testset}"; do
            test -s "${reusable}"
        done
    else
        map_root=${MATCHED_ZONE_PARENT}/${crop}/map
        merge_id=$(dependency_job "${crop}" merge)
        testset_id=$(dependency_job "${crop}" testset)
        test -n "${merge_id}"
        test -n "${testset_id}"
        dependency="afterok:${merge_id}:${testset_id}"
        dependency_args+=(--dependency="${dependency}")
    fi

    net=${map_root}/sionna/${crop}_radio_bounds.net.xml
    for interval in "${INTERVALS[@]}"; do
        run_name=cell_grid_intensity_top1_entry_every${interval}_full1800_eval50_tail10x25
        results_dir=${RESULTS_ROOT}/methods/${short}/${run_name}
        if [[ -e "${results_dir}" ]]; then
            printf 'refusing to reuse existing result directory: %s\n' \
                "${results_dir}" >&2
            exit 1
        fi
        mkdir -p "${results_dir}"

        export_values="ALL,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},SIM_STEPS=1799,REPLAY_CAPACITY=10000,FULL_DATASET_EPOCHS=1,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CELL_GRID_CONFIDENCE=binary,CELL_GRID_MIN_INTENSITY=1,CELL_GRID_WEIGHTED_SINGLE=1,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},WEIGHTED_SELECTION=grid-intensity,WEIGHTED_PULLS_PER_RECEIVER_STEP=1,WEIGHTED_PULL_INTERVAL_STEPS=${interval},WEIGHTED_PULL_SCHEDULE_ANCHOR=entry,METHOD_TAG=intensity_top1_entry_every${interval}_full1800"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-entry${interval}" \
            --output="${results_dir}/slurm-%j.out" \
            --error="${results_dir}/slurm-%j.err" \
            "${dependency_args[@]}" \
            --export="${export_values}" \
            "${SIM_SCRIPT}")"
        printf '%s\tsimulation\t%s\t%s\n' \
            "${crop}" "${run_name}" "${sim_id}" >> "${SUBMISSION_TABLE}"
    done
    printf '%s entry-anchored simulations=5 dependency=%s\n' \
        "${crop}" "${dependency:-none}"
done

printf 'Submission manifest: %s\n' "${SUBMISSION_TABLE}"
