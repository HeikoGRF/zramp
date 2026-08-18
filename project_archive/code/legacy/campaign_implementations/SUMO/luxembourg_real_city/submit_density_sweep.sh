#!/usr/bin/env bash
# Controlled vehicle-density sweep on the medium-building 100 m hotspot.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
MAX_CONCURRENT_PER_ARRAY=${MAX_CONCURRENT_PER_ARRAY:-200}
SOURCE_CROP=${SOURCE_CROP:-factor_b2_v2_hotspot_100m}
MAP_SIZE_M=${MAP_SIZE_M:-100}
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/nested_size_zones}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_density_benchmark_v1}
SUBMISSION_TABLE=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_jobs.tsv}
AUGMENT_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_mobility_density_augment.sh
CHUNK_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_factorial_zone_rssi_chunk.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh
SIM_SCRIPT=${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh
BUNDLE=${BUNDLE:-${REPO_ROOT}/trained_models/experimental/acquisition_pretraining/patch_grid_acquisition_v8_c16_pq4x256/bundle.pt}

FACTORS=(2 4)
PENALTIES=(2 10 50)
BASELINES=(local-only central)
if [[ -n "${FACTOR_FILTER:-}" ]]; then
    read -r -a FACTORS <<< "${FACTOR_FILTER}"
fi

source_data=${DATA_PARENT}/${SOURCE_CROP}_30min_opaque_buildings_no_vehicle_blockers
source_mobility=${source_data}/mobility/${SOURCE_CROP}_all_vehicles_0745_0815_1s_full1800.json
testset=${source_data}/testset/${SOURCE_CROP}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
map_root=${ZONE_PARENT}/${SOURCE_CROP}/map
net=${map_root}/sionna/${SOURCE_CROP}_radio_bounds.net.xml

for required in "${source_mobility}" "${testset}" "${net}" "${AUGMENT_SCRIPT}"     "${CHUNK_SCRIPT}" "${MERGE_SCRIPT}" "${SIM_SCRIPT}" "${BUNDLE}"; do
    test -s "${required}"
done
if [[ -e "${SUBMISSION_TABLE}" ]]; then
    echo "Refusing to overwrite existing submission manifest: ${SUBMISSION_TABLE}" >&2
    exit 2
fi

mkdir -p "${RESULTS_ROOT}"
printf 'source_crop\tdensity_factor\tstage\tmethod\tpenalty_pct\tjob_id\tdependency\tresults_dir\n'     > "${SUBMISSION_TABLE}"

for factor in "${FACTORS[@]}"; do
    case "${factor}" in
        2|4) ;;
        *) echo "Unsupported density factor: ${factor}" >&2; exit 2 ;;
    esac
    variant=${SOURCE_CROP}_density${factor}x
    data_root=${DATA_PARENT}/${variant}_30min_opaque_buildings_no_vehicle_blockers
    mobility=${data_root}/mobility/${variant}_all_vehicles_0745_0815_1s_full1800.json
    merged=${data_root}/rssi/${variant}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    method_root=${RESULTS_ROOT}/methods/${variant}
    mkdir -p "${data_root}/logs" "${data_root}/mobility"         "${data_root}/rssi/shards" "${method_root}"

    augment_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="dens${factor}-augment"         --output="${data_root}/logs/augment-%j.out"         --error="${data_root}/logs/augment-%j.err"         --export="ALL,SOURCE_MOBILITY=${source_mobility},OUTPUT_MOBILITY=${mobility},DENSITY_FACTOR=${factor}"         "${AUGMENT_SCRIPT}")"
    printf '%s\t%s\taugment\t-\t-\t%s\t-\t-\n'         "${SOURCE_CROP}" "${factor}" "${augment_id}" >> "${SUBMISSION_TABLE}"

    array_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="dens${factor}-rssi"         --array="0-359%${MAX_CONCURRENT_PER_ARRAY}"         --exclude="arton10,arton11"         --output="${data_root}/logs/rssi-chunk-%A_%a.out"         --error="${data_root}/logs/rssi-chunk-%A_%a.err"         --dependency="afterok:${augment_id}"         --export="ALL,CROP_NAME=${SOURCE_CROP},DATA_ROOT=${data_root},MAP_ROOT=${map_root},MOBILITY=${mobility},MAP_SIZE_M=${MAP_SIZE_M}"         "${CHUNK_SCRIPT}")"
    printf '%s\t%s\trssi_chunks\t-\t-\t%s\t%s\t-\n'         "${SOURCE_CROP}" "${factor}" "${array_id}" "${augment_id}"         >> "${SUBMISSION_TABLE}"

    merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable         --job-name="dens${factor}-merge"         --output="${data_root}/logs/merge-%j.out"         --error="${data_root}/logs/merge-%j.err"         --dependency="afterok:${array_id}"         --export="ALL,DATA_ROOT=${data_root},MERGED_TRACE_OUTPUT=${merged},MIN_RSSI_DBM=-100"         "${MERGE_SCRIPT}")"
    printf '%s\t%s\tmerge\t-\t-\t%s\t%s\t-\n'         "${SOURCE_CROP}" "${factor}" "${merge_id}" "${array_id}"         >> "${SUBMISSION_TABLE}"

    sim_dependency="afterok:${merge_id}"
    for penalty_pct in "${PENALTIES[@]}"; do
        penalty="$(awk -v value="${penalty_pct}" 'BEGIN { printf "%.6f", value / 100.0 }')"
        method_tag=pq4x256_s3650_density${factor}x_zero_width_kappa${penalty_pct}_angle12
        run_name=support_expert_bank_${method_tag}_eval50_tail10x25
        results_dir=${method_root}/${run_name}
        mkdir -p "${results_dir}"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable             --job-name="dens${factor}-k${penalty_pct}"             --output="${results_dir}/slurm-%j.out"             --error="${results_dir}/slurm-%j.err"             --dependency="${sim_dependency}"             --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BANK_CAPACITY=6,LEARNED_ACQUISITION_BUNDLE=${BUNDLE},ACQUISITION_RELATIVE_GAIN_PENALTY=${penalty},REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=${method_tag}"             "${SIM_SCRIPT}")"
        printf '%s\t%s\tsimulation\texpert-bank\t%s\t%s\t%s\t%s\n'             "${SOURCE_CROP}" "${factor}" "${penalty_pct}" "${sim_id}"             "${sim_dependency}" "${results_dir}" >> "${SUBMISSION_TABLE}"
    done

    for mode in "${BASELINES[@]}"; do
        mode_tag=${mode//-/_}
        run_name=baseline_${mode_tag}_density${factor}x_zero_width_angle12_eval50_tail10x25
        results_dir=${method_root}/${run_name}
        mkdir -p "${results_dir}"
        sim_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable             --job-name="dens${factor}-${mode_tag}"             --output="${results_dir}/slurm-%j.out"             --error="${results_dir}/slurm-%j.err"             --dependency="${sim_dependency}"             --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${merged},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BASELINE_MODE=${mode},BANK_CAPACITY=1,REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=density${factor}x_${mode_tag}_zero_width_angle12"             "${SIM_SCRIPT}")"
        printf '%s\t%s\tsimulation\t%s\t-\t%s\t%s\t%s\n'             "${SOURCE_CROP}" "${factor}" "${mode}" "${sim_id}"             "${sim_dependency}" "${results_dir}" >> "${SUBMISSION_TABLE}"
    done
    printf '%s density=%sx: augment=%s rssi=%s merge=%s simulations=5\n'         "${SOURCE_CROP}" "${factor}" "${augment_id}" "${array_id}" "${merge_id}"
done

printf 'Submission manifest: %s\n' "${SUBMISSION_TABLE}"

