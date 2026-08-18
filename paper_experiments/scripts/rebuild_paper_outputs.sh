#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
paper_root=$(cd -- "$script_dir/.." && pwd -P)
artifact_root=${1:-$paper_root/results/paper}
output_root=${2:-$paper_root/figures/data}
python_bin=${PYTHON_BIN:-python3}

"$python_bin" "$paper_root/code/final/analysis/aggregate_paper_results_across_timeframes.py" \
    --artifact-root "$artifact_root" \
    --output-dir "$output_root/statistical_aggregation"

"$python_bin" "$paper_root/code/final/analysis/build_plot_ready_paper_tables.py" \
    --aggregation-dir "$output_root/statistical_aggregation" \
    --output-dir "$output_root/plot_ready_tables"

echo "Rebuilt paper tables under: $output_root"
