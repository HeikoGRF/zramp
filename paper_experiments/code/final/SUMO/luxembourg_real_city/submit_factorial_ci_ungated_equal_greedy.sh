#!/usr/bin/env bash
# Submit an ungated equal-weight all-neighbour averaging baseline for the
# original factorial window and the four additional temporal CI realizations.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
SLURM_CONFIG=${SLURM_CONFIG:-/tmp/slurm-itet.conf}
CI_DATA_ROOT=${CI_DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_ci_temporal5_paper_final_v1}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/artifacts/luxembourg_cell_grid_ci_temporal5_paper_final_v1}
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones}
ORIGINAL_DATA_PARENT=${ORIGINAL_DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
SIM_SCRIPT=${SIM_SCRIPT:-${REPO_ROOT}/experiments/place_wallis_benchmark/submit_equal_greedy.sh}

ZONES=(
    factor_b1_v1_300m factor_b1_v2_300m factor_b1_v3_300m
    factor_b2_v1_300m factor_b2_v2_300m factor_b2_v3_300m
    factor_b3_v1_300m factor_b3_v2_300m factor_b3_v3_300m
)
REPLICATES=(1 2 3 4 5)

test -s "${SIM_SCRIPT}"
mkdir -p "${RESULTS_ROOT}"
manifest=${RESULTS_ROOT}/submitted_ungated_equal_greedy_jobs.tsv
if [[ ! -e "${manifest}" ]]; then
    printf 'zone\treplicate\tconfiguration\tjob_id\toutput\n' > "${manifest}"
fi

submitted=0
for zone in "${ZONES[@]}"; do
    short=${zone%_300m}
    data_root=${ORIGINAL_DATA_PARENT}/${zone}_30min_opaque_buildings_no_vehicle_blockers
    testset=${data_root}/testset/${zone}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz
    net=${ZONE_PARENT}/${zone}/map/sionna/${zone}_radio_bounds.net.xml
    test -s "${testset}"
    test -s "${net}"

    for replicate in "${REPLICATES[@]}"; do
        if [[ "${replicate}" == 1 ]]; then
            trace=${data_root}/rssi/${zone}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
        else
            trace=${CI_DATA_ROOT}/replicates/${zone}/rep${replicate}/rssi/${zone}_rep${replicate}_vehicles_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
        fi
        test -s "${trace}"

        configuration=ungated_equal_greedy
        result_dir=${RESULTS_ROOT}/methods/${short}/rep${replicate}/${configuration}_paper_final_full1800_eval50_tail10x25
        if awk -F '\t' -v z="${zone}" -v r="${replicate}" \
            '$1 == z && $2 == r {found = 1} END {exit !found}' \
            "${manifest}"; then
            echo "Already recorded ${zone} rep${replicate}; skipping"
            continue
        fi
        if [[ -e "${result_dir}" ]]; then
            echo "Unrecorded result directory already exists: ${result_dir}" >&2
            exit 1
        fi
        mkdir -p "${result_dir}"

        job_id=$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
            --job-name="ci-ug-${short#factor_}-r${replicate}" \
            --output="${result_dir}/slurm-%j.out" \
            --error="${result_dir}/slurm-%j.err" \
            --export="ALL,PYTHONUNBUFFERED=1,TRACE_PATH=${trace},TESTSET_PATH=${testset},NET_PATH=${net},RESULTS_DIR=${result_dir},SIM_STEPS=1799,SEED=1,REPLAY_CAPACITY=0,FULL_DATASET_EPOCHS=1,CHECKPOINT_EVERY=50,TAIL_EVAL_COUNT=10,TAIL_EVAL_STRIDE=25,METHOD_TAG=ci_rep${replicate}_ungated_equal_greedy" \
            "${SIM_SCRIPT}")
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${zone}" "${replicate}" "${configuration}" "${job_id}" \
            "${result_dir}" >> "${manifest}"
        submitted=$((submitted + 1))
        echo "Submitted ${zone} rep${replicate}: ${job_id}"
    done
done

printf 'Submitted %d ungated equal-greedy jobs. Manifest: %s\n' \
    "${submitted}" "${manifest}"
