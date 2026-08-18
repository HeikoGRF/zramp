#!/usr/bin/env bash
# Build and evaluate nested 100 m and 200 m crops centered in three existing
# 300 m factorial zones. All outputs are isolated from the original campaign.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/nested_size_zones}
MANIFEST=${MANIFEST:-${REPO_ROOT}/SUMO/luxembourg_real_city/nested_size_zones_crop_manifest.json}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-200}
PREP_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_prepare.sh
CHUNK_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_rssi_chunk.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
TESTSET_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_testset.sh
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${BUNDLE:-${REPO_ROOT}/trained_models/experimental/acquisition_pretraining/patch_grid_acquisition_v8_c16_pq4x256/bundle.pt}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_nested_size_benchmark_v1}
SUBMISSION_TABLE=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_jobs.tsv}

ZONES=(
    factor_b1_v2_hotspot_100m factor_b1_v2_200m
    factor_b2_v2_hotspot_100m factor_b2_v2_200m
    factor_b3_v2_hotspot_100m factor_b3_v2_200m
)
PENALTIES=(2 10 50)
BASELINES=(local-only central)

if [[ -n "${ZONE_FILTER:-}" ]]; then
    read -r -a ZONES <<< "${ZONE_FILTER}"
fi
if [[ -n "${PENALTY_FILTER:-}" ]]; then
    read -r -a PENALTIES <<< "${PENALTY_FILTER}"
fi
for required in "${MANIFEST}" "${PREP_SCRIPT}" "${CHUNK_SCRIPT}"     "${MERGE_SCRIPT}" "${TESTSET_SCRIPT}" "${SIM_SCRIPT}" "${BUNDLE}"; do
    test -s "${required}"
done
if [[ -e "${SUBMISSION_TABLE}" && "${ALLOW_RESUBMIT:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing submission manifest: ${SUBMISSION_TABLE}" >&2
    exit 2
fi

mkdir -p "${RESULTS_ROOT}"
printf 'zone\tsize_m\tstage\tmethod\tpenalty_pct\tjob_id\tdependency\tresults_dir\n'     > "${SUBMISSION_TABLE}"

for crop in "${ZONES[@]}"; do
    size_tag=${crop##*_}
    size_m=${size_tag%m}
    case "${size_m}" in
        100|200) ;;
        *) echo "Unsupported nested crop size: ${crop}" >&2; exit 2 ;;
    esac

    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    map_root=${ZONE_PARENT}/${crop}/map
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    testset=${data_root}/testset/${crop}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    net=${map_root}/sionna/${crop}_radio_bounds.net.xml
    method_root=${RESULTS_ROOT}/methods/${crop}

    mkdir -p "${data_root}/logs" "${data_root}/rssi/shards"         "${data_root}/testset" "${method_root}"

    prep_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="ns-${crop}-prep"         --output="${data_root}/logs/prepare-%j.out"         --error="${data_root}/logs/prepare-%j.err"         --export="ALL,CROP_NAME=${crop},DATA_PARENT=${DATA_PARENT},ZONE_PARENT=${ZONE_PARENT},MANIFEST=${MANIFEST}"         "${PREP_SCRIPT}")"
    printf '%s\t%s\tprep\t-\t-\t%s\t-\t-\n'         "${crop}" "${size_m}" "${prep_id}" >> "${SUBMISSION_TABLE}"

    array_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="ns-${crop}-rssi"         --array="0-359%${MAX_CONCURRENT_PER_ARRAY}"         --exclude="arton10,arton11"         --output="${data_root}/logs/rssi-chunk-%A_%a.out"         --error="${data_root}/logs/rssi-chunk-%A_%a.err"         --dependency="afterok:${prep_id}"         --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root},MOBILITY=${mobility},MAP_SIZE_M=${size_m}"         "${CHUNK_SCRIPT}")"
    printf '%s\t%s\trssi_chunks\t-\t-\t%s\t%s\t-\n'         "${crop}" "${size_m}" "${array_id}" "${prep_id}" >> "${SUBMISSION_TABLE}"

    merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="ns-${crop}-merge"         --output="${data_root}/logs/merge-%j.out"         --error="${data_root}/logs/merge-%j.err"         --dependency="afterok:${array_id}"         --export="ALL,DATA_ROOT=${data_root},MERGED_TRACE_OUTPUT=${merged},MIN_RSSI_DBM=-100"         "${MERGE_SCRIPT}")"
    printf '%s\t%s\tmerge\t-\t-\t%s\t%s\t-\n'         "${crop}" "${size_m}" "${merge_id}" "${array_id}" >> "${SUBMISSION_TABLE}"

    testset_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="ns-${crop}-test"         --exclude="arton10,arton11"         --output="${data_root}/logs/testset-%j.out"         --error="${data_root}/logs/testset-%j.err"         --dependency="afterok:${prep_id}"         --export="ALL,CROP_NAME=${crop},DATA_ROOT=${data_root},MAP_ROOT=${map_root},MAP_SIZE_M=${size_m}"         "${TESTSET_SCRIPT}")"
    printf '%s\t%s\ttestset\t-\t-\t%s\t%s\t-\n'         "${crop}" "${size_m}" "${testset_id}" "${prep_id}" >> "${SUBMISSION_TABLE}"

    sim_dependency="afterok:${merge_id}:${testset_id}"
    for penalty_pct in "${PENALTIES[@]}"; do
        penalty="$(awk -v value="${penalty_pct}" 'BEGIN { printf "%.6f", value / 100.0 }')"
        method_tag=pq4x256_s3650_cached_zero_width_kappa${penalty_pct}_angle12
        run_name=support_expert_bank_${method_tag}_eval50_tail10x25
        results_dir=${method_root}/${run_name}
        mkdir -p "${results_dir}"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable             --job-name="ns-${crop}-k${penalty_pct}"             --output="${results_dir}/slurm-%j.out"             --error="${results_dir}/slurm-%j.err"             --dependency="${sim_dependency}"             --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BANK_CAPACITY=6,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},ACQUISITION_RELATIVE_GAIN_PENALTY=${penalty},REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=${method_tag}"             "${SIM_SCRIPT}")"
        printf '%s\t%s\tsimulation\texpert-bank\t%s\t%s\t%s\t%s\n'             "${crop}" "${size_m}" "${penalty_pct}" "${sim_id}"             "${sim_dependency}" "${results_dir}" >> "${SUBMISSION_TABLE}"
    done

    for mode in "${BASELINES[@]}"; do
        mode_tag=${mode//-/_}
        run_name=baseline_${mode_tag}_zero_width_angle12_eval50_tail10x25
        results_dir=${method_root}/${run_name}
        mkdir -p "${results_dir}"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable             --job-name="ns-${crop}-${mode_tag}"             --output="${results_dir}/slurm-%j.out"             --error="${results_dir}/slurm-%j.err"             --dependency="${sim_dependency}"             --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BASELINE_MODE=${mode},BANK_CAPACITY=1,REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=nested_size_${mode_tag}_zero_width_angle12"             "${SIM_SCRIPT}")"
        printf '%s\t%s\tsimulation\t%s\t-\t%s\t%s\t%s\n'             "${crop}" "${size_m}" "${mode}" "${sim_id}"             "${sim_dependency}" "${results_dir}" >> "${SUBMISSION_TABLE}"
    done

    printf '%s: prep=%s rssi=%s merge=%s test=%s simulations=5\n'         "${crop}" "${prep_id}" "${array_id}" "${merge_id}" "${testset_id}"
done

printf 'Submission manifest: %s\n' "${SUBMISSION_TABLE}"
