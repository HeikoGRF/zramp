#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 LUST3D_SOURCE_ROOT OUTPUT_DIRECTORY" >&2
    echo "Runs the full-day SUMO scan and selects four additional windows per map." >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }

source_root=$(cd -- "$1" && pwd -P)
mkdir -p "$2"
output_root=$(cd -- "$2" && pwd -P)
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
paper_root=$(cd -- "$script_dir/.." && pwd -P)
code_root="$paper_root/code/final"
scan_root="$output_root/full_day_scan"
fcd="$scan_root/lust3d_full_day_period5_seed01.fcd.xml.gz"
python_bin=${PYTHON_BIN:-python3}
sumo_bin=${SUMO_BIN:-sumo}

env \
    SUMO_BIN="$sumo_bin" \
    SOURCE_ROOT="$source_root" \
    OUTPUT_ROOT="$scan_root" \
    FCD_OUTPUT="$fcd" \
    bash "$code_root/SUMO/luxembourg_real_city/run_ci_lust_full_day_scan.sh"

"$python_bin" "$code_root/SUMO/luxembourg_real_city/select_ci_temporal_windows.py" \
    --fcd "$fcd" \
    --crop-manifest "$paper_root/input_data/zone_metadata/factorial_zones_crop_manifest.json" \
    --targets "$paper_root/input_data/zone_metadata/proposed_zones.csv" \
    --output-tsv "$output_root/selected_windows.tsv" \
    --output-json "$output_root/selected_windows.json" \
    --window-seconds "${WINDOW_SECONDS:-1800}" \
    --candidate-stride-seconds "${CANDIDATE_STRIDE_SECONDS:-300}" \
    --separation-buffer-seconds "${SEPARATION_BUFFER_SECONDS:-900}" \
    --existing-start-seconds "${EXISTING_START_SECONDS:-27900}" \
    --earliest-start-seconds "${EARLIEST_START_SECONDS:-3600}"

[[ $(awk 'END {print NR - 1}' "$output_root/selected_windows.tsv") -eq 45 ]]
echo "Selected windows: $output_root/selected_windows.tsv"
echo "Use them by setting WINDOWS_TSV to that path during input generation."

