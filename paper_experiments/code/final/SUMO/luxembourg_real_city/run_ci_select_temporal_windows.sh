#!/usr/bin/env bash
#SBATCH --job-name=ci-window-select
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_ci_temporal5_paper_final_v1}
FCD=${FCD:-${DATA_ROOT}/full_day_scan/lust3d_full_day_period5_seed01.fcd.xml.gz}
OUTPUT_TSV=${OUTPUT_TSV:-${DATA_ROOT}/selected_windows.tsv}
OUTPUT_JSON=${OUTPUT_JSON:-${DATA_ROOT}/selected_windows.json}

test -s "${FCD}"
"${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/select_ci_temporal_windows.py" \
    --fcd "${FCD}" \
    --crop-manifest "${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones_crop_manifest.json" \
    --targets "${REPO_ROOT}/artifacts/luxembourg_zone_factorial_3x3_v1/proposed_zones.csv" \
    --output-tsv "${OUTPUT_TSV}" \
    --output-json "${OUTPUT_JSON}" \
    --window-seconds 1800 \
    --candidate-stride-seconds 300 \
    --separation-buffer-seconds 900 \
    --existing-start-seconds 27900

test "$(awk 'END {print NR}' "${OUTPUT_TSV}")" -eq 46
