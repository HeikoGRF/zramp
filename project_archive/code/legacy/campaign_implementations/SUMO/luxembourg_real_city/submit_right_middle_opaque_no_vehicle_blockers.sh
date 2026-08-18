#!/usr/bin/env bash
# Generate the opaque-building trace without vehicle blocker geometry.

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
DATA_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers
SLURM_CONFIG=/tmp/slurm-itet.conf
FRAME_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_right_middle_rssi_frame.sh
TESTSET_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_right_middle_testset.sh
MERGE_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_right_middle_merge.sh
MERGED_TRACE_OUTPUT=${DATA_ROOT}/rssi/gare_bonnevoie_vehicles_0745_0815_1s_right_middle_opaque_no_vehicle_blockers_r20k_d3_llvm.npz
TESTSET_OUTPUT=${DATA_ROOT}/testset/right_middle_street_pairs_10000_opaque_no_vehicle_blockers_static.npz

mkdir -p "${DATA_ROOT}/logs" "${DATA_ROOT}/rssi/shards" "${DATA_ROOT}/testset"

testset_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
    --job-name=bonnevoie-novblk-test \
    --output="${DATA_ROOT}/logs/testset-%j.out" \
    --error="${DATA_ROOT}/logs/testset-%j.err" \
    --export="ALL,DATA_ROOT=${DATA_ROOT},TESTSET_OUTPUT=${TESTSET_OUTPUT}" \
    "${TESTSET_SCRIPT}")"
echo "opaque no-blocker test set: ${testset_id}"

array_ids=()
for offset in 0 1 2 3 4; do
    job_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
        --job-name=bonnevoie-novblk \
        --array=0-359%200 \
        --output="${DATA_ROOT}/logs/rssi-%A_%a.out" \
        --error="${DATA_ROOT}/logs/rssi-%A_%a.err" \
        --export="ALL,DATA_ROOT=${DATA_ROOT},FRAME_OFFSET=${offset}" \
        "${FRAME_SCRIPT}")"
    array_ids+=("${job_id}")
    echo "no-blocker frame offset ${offset}: ${job_id}"
done

dependency="afterok:$(IFS=:; echo "${array_ids[*]}")"
merge_id="$(env SLURM_CONF="${SLURM_CONFIG}" sbatch --parsable \
    --job-name=bonnevoie-novblk-merge \
    --output="${DATA_ROOT}/logs/merge-%j.out" \
    --error="${DATA_ROOT}/logs/merge-%j.err" \
    --dependency="${dependency}" \
    --export="ALL,DATA_ROOT=${DATA_ROOT},MERGED_TRACE_OUTPUT=${MERGED_TRACE_OUTPUT}" \
    "${MERGE_SCRIPT}")"
echo "no-blocker merge: ${merge_id} (${dependency})"

printf 'FRAME_ARRAY_JOB_IDS=%s\n' "$(IFS=,; echo "${array_ids[*]}")"
printf 'TESTSET_JOB_ID=%s\n' "${testset_id}"
printf 'MERGE_JOB_ID=%s\n' "${merge_id}"
printf 'DATA_ROOT=%s\n' "${DATA_ROOT}"
