#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hgraef/zramp-workspace"
RESULTS_ROOT="${REPO_ROOT}/artifacts/luxembourg_cell_grid_intensity_budget_synchronized_9map_v1"
REUSED_ROOT="${REPO_ROOT}/artifacts/luxembourg_cell_grid_synchronized_matched_b2v3_v1"
REUSED_TABLE="${REUSED_ROOT}/submitted_jobs.tsv"
DATA_ROOT="/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city"
BUNDLE="${REPO_ROOT}/trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt"
SUBMIT_SCRIPT="${REPO_ROOT}/experiments/place_wallis_benchmark/submit_support_expert_bank.sh"

for required in "${REUSED_TABLE}" "${BUNDLE}" "${SUBMIT_SCRIPT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required input is missing: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${RESULTS_ROOT}/methods"
MANIFEST="${RESULTS_ROOT}/submitted_jobs.tsv"
if [[ ! -e "${MANIFEST}" ]]; then
    printf 'zone\tstage\tconfiguration\tjob_id\tsource\n' > "${MANIFEST}"
fi

ZONES=(
    factor_b1_v1_300m
    factor_b1_v2_300m
    factor_b1_v3_300m
    factor_b2_v1_300m
    factor_b2_v2_300m
    factor_b2_v3_300m
    factor_b3_v1_300m
    factor_b3_v2_300m
    factor_b3_v3_300m
)
INTERVALS=(5 10 20 40 80)

for zone in "${ZONES[@]}"; do
    zone_dir="${RESULTS_ROOT}/methods/${zone}"
    mkdir -p "${zone_dir}"

    for interval in "${INTERVALS[@]}"; do
        result_name="cell_grid_intensity_top1_global_every${interval}_full1800_eval50_tail10x25"
        configuration="global_every${interval}"

        if awk -F '\t' -v z="${zone}" -v c="${configuration}" \
            '$1 == z && $3 == c {found = 1} END {exit !found}' "${MANIFEST}"; then
            echo "Already recorded ${zone} ${configuration}; skipping"
            continue
        fi

        if [[ "${zone}" == "factor_b2_v3_300m" ]]; then
            source_dir="${REUSED_ROOT}/methods/factor_b2_v3/${result_name}"
            if [[ ! -d "${source_dir}" ]]; then
                echo "Reusable result directory is missing: ${source_dir}" >&2
                exit 1
            fi
            if [[ ! -e "${zone_dir}/${result_name}" && ! -L "${zone_dir}/${result_name}" ]]; then
                ln -s "${source_dir}" "${zone_dir}/${result_name}"
            fi
            job_id="$(awk -F '\t' -v z='factor_b2_v3_300m' -v c="${result_name}" '$1 == z && $3 == c {print $4; exit}' "${REUSED_TABLE}")"
            if [[ -z "${job_id}" ]]; then
                echo "Could not resolve reused job ID for ${configuration}" >&2
                exit 1
            fi
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "${zone}" "reused_simulation" "${configuration}" "${job_id}" "${source_dir}" >> "${MANIFEST}"
            continue
        fi

        crop_data="${DATA_ROOT}/${zone}_30min_opaque_buildings_no_vehicle_blockers"
        trace="${crop_data}/rssi/${zone}_vehicles_0745_0815_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz"
        testset="${crop_data}/testset/${zone}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz"
        net="${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones/${zone}/map/sionna/${zone}_radio_bounds.net.xml"

        for required in "${trace}" "${testset}" "${net}"; do
            if [[ ! -f "${required}" ]]; then
                echo "Required zone asset is missing: ${required}" >&2
                exit 1
            fi
        done

        short="${zone%_300m}"
        export RESULTS_DIR="${zone_dir}/${result_name}"
        export TRACE_PATH="${trace}"
        export TESTSET_PATH="${testset}"
        export NET_PATH="${net}"
        export SIM_STEPS=1799
        export REPLAY_CAPACITY=10000
        export FULL_DATASET_EPOCHS=1
        export CHECKPOINT_EVERY=50
        export TAIL_EVAL_COUNT=10
        export TAIL_EVAL_STRIDE=25
        export PLANE_ANGLE_DEG=12
        export PLANE_INITIAL_HALF_WIDTH_M=0
        export CELL_GRID_CONFIDENCE=binary
        export CELL_GRID_MIN_INTENSITY=1
        export CELL_GRID_WEIGHTED_SINGLE=1
        export LEARNED_ACQUISITION_BUNDLE="${BUNDLE}"
        export WEIGHTED_SELECTION=grid-intensity
        export WEIGHTED_PULLS_PER_RECEIVER_STEP=1
        export WEIGHTED_PULL_INTERVAL_STEPS="${interval}"
        export WEIGHTED_PULL_SCHEDULE_ANCHOR=global
        export METHOD_TAG="intensity_top1_global_every${interval}_full1800"

        job_id="$(env SLURM_CONF=/tmp/slurm-itet.conf sbatch --parsable \
            --job-name="${short}-global${interval}" \
            "${SUBMIT_SCRIPT}")"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${zone}" "simulation" "global_every${interval}" "${job_id}" "${RESULTS_DIR}" >> "${MANIFEST}"
        echo "Submitted ${zone} global_every${interval}: ${job_id}"
    done
done

echo "Sweep manifest: ${MANIFEST}"
