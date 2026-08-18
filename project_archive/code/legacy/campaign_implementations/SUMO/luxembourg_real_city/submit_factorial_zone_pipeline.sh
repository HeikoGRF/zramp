#!/usr/bin/env bash
# Submit nine preparation jobs, CPU ray-tracing arrays, merges, test sets, and
# the 27-run learned-acquisition communication sweep.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-200}
PREP_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_prepare.sh
CHUNK_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_rssi_chunk.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
TESTSET_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_testset.sh
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${BUNDLE:-${REPO_ROOT}/results/legacy_experiments/support_acquisition_pretraining/synthetic_regular_grid300_gain_v2_optimized_fast/frozen_best_step11300_sweep/bundle.pt}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_factorial_3x3_benchmark}
ZONES=(
    factor_b1_v1_300m factor_b1_v2_300m factor_b1_v3_300m
    factor_b2_v1_300m factor_b2_v2_300m factor_b2_v3_300m
    factor_b3_v1_300m factor_b3_v2_300m factor_b3_v3_300m
)
PENALTIES=(2 10 50)

if [[ -n "${ZONE_FILTER:-}" ]]; then
    read -r -a ZONES <<< "${ZONE_FILTER}"
fi
for required in "${PREP_SCRIPT}" "${CHUNK_SCRIPT}" "${MERGE_SCRIPT}" \
    "${TESTSET_SCRIPT}" "${SIM_SCRIPT}" "${BUNDLE}"; do
    test -s "${required}"
done
mkdir -p "${RESULTS_ROOT}"
submission_table=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_jobs.tsv}
if [[ "${APPEND_SUBMISSIONS:-0}" != "1" || ! -s "${submission_table}" ]]; then
    printf 'zone\tstage\tpenalty_pct\tjob_id\n' > "${submission_table}"
fi

for crop in "${ZONES[@]}"; do
    short=${crop%_300m}
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    map_root=${ZONE_PARENT}/${crop}/map
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    testset=${data_root}/testset/${crop}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    net=${map_root}/sionna/${crop}_radio_bounds.net.xml
    mkdir -p "${data_root}/logs" "${data_root}/rssi/shards" "${data_root}/testset" \
        "${RESULTS_ROOT}/methods/${short}"

    prep_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name="${short}-prep" \
        --output="${data_root}/logs/prepare-%j.out" \
        --error="${data_root}/logs/prepare-%j.err" \
        --export="ALL,CROP_NAME=${crop},DATA_PARENT=${DATA_PARENT}" \
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
    for penalty_pct in "${PENALTIES[@]}"; do
        penalty="$(awk -v value="${penalty_pct}" 'BEGIN { printf "%.6f", value / 100.0 }')"
        run_name=support_expert_bank_regular_grid300_step11300_zero_width_append_allD1_kappa${penalty_pct}_angle12_eval50_tail10x25
        results_dir=${RESULTS_ROOT}/methods/${short}/${run_name}
        mkdir -p "${results_dir}"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-k${penalty_pct}" \
            --output="${results_dir}/slurm-%j.out" \
            --error="${results_dir}/slurm-%j.err" \
            --dependency="${sim_dependency}" \
            --export="ALL,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BANK_CAPACITY=6,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},ACQUISITION_RELATIVE_GAIN_PENALTY=${penalty},REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,METHOD_TAG=regular_grid300_step11300_zero_width_append_allD1_kappa${penalty_pct}_angle12" \
            "${SIM_SCRIPT}")"
        printf '%s\tsimulation\t%s\t%s\n' "${crop}" "${penalty_pct}" "${sim_id}" >> "${submission_table}"
    done
    printf '%s prep=%s merge=%s testset=%s simulations=3\n' \
        "${crop}" "${prep_id}" "${merge_id}" "${testset_id}"
done

printf 'Submission manifest: %s\n' "${submission_table}"
