#!/usr/bin/env bash
#SBATCH --job-name=ci-temporal-mobility
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
SUMO_BIN=${SUMO_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/sumo}
SOURCE_ROOT=${SOURCE_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/extracted/lust3d_v1}
CI_DATA_ROOT=${CI_DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_ci_temporal5_paper_final_v1}
WINDOWS_TSV=${WINDOWS_TSV:-${CI_DATA_ROOT}/selected_windows.tsv}
CROP_MANIFEST=${CROP_MANIFEST:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones_crop_manifest.json}

: "${CROP_NAME:?CROP_NAME must be set}"
: "${REPLICATE:?REPLICATE must be set}"

row=$(awk -F '\t' -v zone="${CROP_NAME}" -v replicate="${REPLICATE}" \
    '$1 == zone && $2 == replicate {print $0}' "${WINDOWS_TSV}")
if [[ -z "${row}" ]]; then
    echo "No selected window for ${CROP_NAME} replicate ${REPLICATE}" >&2
    exit 1
fi
start_s=$(awk -F '\t' '{print $3}' <<< "${row}")
end_s=$(awk -F '\t' '{print $4}' <<< "${row}")

replicate_root=${CI_DATA_ROOT}/replicates/${CROP_NAME}/rep${REPLICATE}
fcd=${replicate_root}/fcd/${CROP_NAME}_rep${REPLICATE}_${start_s}_${end_s}_period1.fcd.xml.gz
mobility=${replicate_root}/mobility/${CROP_NAME}_rep${REPLICATE}_all_vehicles_1s_full1800.json
mkdir -p "${replicate_root}/fcd" "${replicate_root}/mobility" "${replicate_root}/logs" \
    "${replicate_root}/rssi/shards"

if [[ ! -s "${fcd}" ]]; then
    "${SUMO_BIN}" \
        --net-file "${SOURCE_ROOT}/lust3d.net.xml" \
        --route-files "${SOURCE_ROOT}/buslines.rou.xml,${SOURCE_ROOT}/local.0.rou.xml,${SOURCE_ROOT}/local.1.rou.xml,${SOURCE_ROOT}/local.2.rou.xml,${SOURCE_ROOT}/transit.rou.xml" \
        --additional-files "${SOURCE_ROOT}/vtypes.add.xml,${SOURCE_ROOT}/busstops.add.xml,${SOURCE_ROOT}/e1detectors.add.xml,${SOURCE_ROOT}/lust3d.poly.xml" \
        --begin 0 \
        --end "${end_s}" \
        --step-length 1 \
        --fcd-output "${fcd}" \
        --device.fcd.begin "${start_s}" \
        --device.fcd.period 1 \
        --summary-output "${replicate_root}/logs/summary.xml" \
        --summary-output.period 60 \
        --ignore-junction-blocker 20 \
        --time-to-teleport 600 \
        --max-depart-delay 600 \
        --routing-algorithm dijkstra \
        --device.rerouting.probability 0.70 \
        --device.rerouting.period 300 \
        --device.rerouting.pre-period 300 \
        --xml-validation never \
        --xml-validation.net never \
        --log "${replicate_root}/logs/sumo.log" \
        --error-log "${replicate_root}/logs/sumo.errors.log" \
        --duration-log.statistics \
        --no-step-log \
        --seed 1
fi

if [[ ! -s "${mobility}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/export_crop_mobility_trace.py" \
        --fcd "${fcd}" \
        --crop-manifest "${CROP_MANIFEST}" \
        --crop "${CROP_NAME}" \
        --output "${mobility}" \
        --begin "${start_s}" \
        --steps 1799 \
        --sample-period 1 \
        --all-participants \
        --z-reference 230
fi

"${PYTHON_BIN}" -c '
import json, sys
p = json.load(open(sys.argv[1]))
assert p["max_step"] == 1799 and len(p["active_counts_by_step"]) == 1800
print(p["crop_name"], len(p["vehicle_ids"]), min(p["active_counts_by_step"]), max(p["active_counts_by_step"]))
' "${mobility}"
