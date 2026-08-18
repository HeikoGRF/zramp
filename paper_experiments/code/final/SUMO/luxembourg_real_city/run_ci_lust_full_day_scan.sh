#!/usr/bin/env bash
#SBATCH --job-name=ci-lust-scan
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

SUMO_BIN=${SUMO_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/sumo}
SOURCE_ROOT=${SOURCE_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/extracted/lust3d_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_ci_temporal5_paper_final_v1/full_day_scan}
FCD_OUTPUT=${FCD_OUTPUT:-${OUTPUT_ROOT}/lust3d_full_day_period5_seed01.fcd.xml.gz}

mkdir -p "${OUTPUT_ROOT}"
if [[ -s "${FCD_OUTPUT}" && -s "${OUTPUT_ROOT}/complete.json" ]]; then
    echo "Full-day scan already complete: ${FCD_OUTPUT}"
    exit 0
fi

"${SUMO_BIN}" \
    --net-file "${SOURCE_ROOT}/lust3d.net.xml" \
    --route-files "${SOURCE_ROOT}/buslines.rou.xml,${SOURCE_ROOT}/local.0.rou.xml,${SOURCE_ROOT}/local.1.rou.xml,${SOURCE_ROOT}/local.2.rou.xml,${SOURCE_ROOT}/transit.rou.xml" \
    --additional-files "${SOURCE_ROOT}/vtypes.add.xml,${SOURCE_ROOT}/busstops.add.xml,${SOURCE_ROOT}/e1detectors.add.xml,${SOURCE_ROOT}/lust3d.poly.xml" \
    --begin 0 \
    --end 86400 \
    --step-length 1 \
    --fcd-output "${FCD_OUTPUT}" \
    --device.fcd.begin 0 \
    --device.fcd.period 5 \
    --summary-output "${OUTPUT_ROOT}/summary.xml" \
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
    --log "${OUTPUT_ROOT}/sumo.log" \
    --error-log "${OUTPUT_ROOT}/sumo.errors.log" \
    --duration-log.statistics \
    --no-step-log \
    --seed 1

printf '{"status":"complete","fcd":"%s","period_seconds":5,"seed":1}\n' \
    "${FCD_OUTPUT}" > "${OUTPUT_ROOT}/complete.json"
