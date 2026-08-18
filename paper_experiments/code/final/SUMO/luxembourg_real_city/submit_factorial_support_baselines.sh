#!/usr/bin/env bash
# Submit local-only and ideal-central support-gated MLP baselines on the
# prepared 3x3 building-density x traffic-density Luxembourg zones.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
SIM_SCRIPT=${SIM_SCRIPT:-${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_factorial_3x3_benchmark}

ZONES=(
    factor_b1_v1_300m factor_b1_v2_300m factor_b1_v3_300m
    factor_b2_v1_300m factor_b2_v2_300m factor_b2_v3_300m
    factor_b3_v1_300m factor_b3_v2_300m factor_b3_v3_300m
)
MODES=(local-only central)

test -s "${SIM_SCRIPT}"
submission_table=${SUBMISSION_TABLE:-${RESULTS_ROOT}/submitted_support_baseline_jobs.tsv}
mkdir -p "${RESULTS_ROOT}"
printf 'zone\tbuilding_score\tvehicle_score\tmode\tjob_id\tresults_dir\n' \
    > "${submission_table}"

submitted=0
for crop in "${ZONES[@]}"; do
    short=${crop%_300m}
    building_score=${short#factor_b}
    building_score=${building_score%%_v*}
    vehicle_score=${short##*_v}
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    trace=${data_root}/rssi/${crop}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
    testset=${data_root}/testset/${crop}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    net=${ZONE_PARENT}/${crop}/map/sionna/${crop}_radio_bounds.net.xml
    for required in "${trace}" "${testset}" "${net}"; do
        test -s "${required}"
    done

    for mode in "${MODES[@]}"; do
        mode_tag=${mode//-/_}
        run_name=baseline_${mode_tag}_zero_width_angle12_eval50_tail10x25
        results_dir=${RESULTS_ROOT}/methods/${short}/${run_name}
        mkdir -p "${results_dir}"
        job_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="${short}-${mode_tag}" \
            --output="${results_dir}/slurm-%j.out" \
            --error="${results_dir}/slurm-%j.err" \
            --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${trace},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${results_dir},BASELINE_MODE=${mode},BANK_CAPACITY=1,REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,PLANE_ANGLE_DEG=12,PLANE_INITIAL_HALF_WIDTH_M=0,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=factorial_${mode_tag}_zero_width_angle12" \
            "${SIM_SCRIPT}")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${crop}" "${building_score}" "${vehicle_score}" \
            "${mode}" "${job_id}" "${results_dir}" \
            >> "${submission_table}"
        submitted=$((submitted + 1))
    done
done

printf 'Submitted %d baseline simulations. Manifest: %s\n' \
    "${submitted}" "${submission_table}"
