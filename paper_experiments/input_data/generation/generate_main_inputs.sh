#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  generate_main_inputs.sh --check SOURCE_ROOT
  generate_main_inputs.sh --submit SOURCE_ROOT OUTPUT_ROOT
  generate_main_inputs.sh --verify OUTPUT_ROOT

SOURCE_ROOT is the extracted LuST3d 1.0.0 directory. OUTPUT_ROOT is an
external working directory; generated data is intentionally not written into
the submission archive.

Environment overrides:
  PYTHON_BIN, SUMO_BIN, WINDOWS_TSV, SLURM_CONFIG, MAX_CONCURRENT_PER_ARRAY,
  SBATCH_ACCOUNT, SBATCH_PARTITION, SBATCH_EXCLUDE
USAGE
}

mode=${1:-}
case "$mode" in
    --check)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        source_root=$(cd -- "$2" && pwd -P)
        output_root=
        ;;
    --submit)
        [[ $# -eq 3 ]] || { usage >&2; exit 2; }
        source_root=$(cd -- "$2" && pwd -P)
        mkdir -p "$3"
        output_root=$(cd -- "$3" && pwd -P)
        ;;
    --verify)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        source_root=
        output_root=$(cd -- "$2" && pwd -P)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
archive_root=$(cd -- "$script_dir/../.." && pwd -P)
code_root="$archive_root/code/final"
windows_tsv=${WINDOWS_TSV:-$archive_root/input_data/prepared_traces/luxembourg_ci_temporal5_paper_final_v1/selected_windows.tsv}
crop_manifest="$archive_root/input_data/zone_metadata/factorial_zones_crop_manifest.json"
map_parent="$archive_root/input_data/maps/luxembourg_real_city/factorial_zones"
testset_parent="$archive_root/input_data/prepared_traces/luxembourg_real_city"
source_checksums="$script_dir/external_lust3d_files.sha256"

python_bin=${PYTHON_BIN:-python3}
sumo_bin=${SUMO_BIN:-sumo}
max_concurrent=${MAX_CONCURRENT_PER_ARRAY:-20}
slurm_config=${SLURM_CONFIG:-}

zones=(
    factor_b1_v1_300m factor_b1_v2_300m factor_b1_v3_300m
    factor_b2_v1_300m factor_b2_v2_300m factor_b2_v3_300m
    factor_b3_v1_300m factor_b3_v2_300m factor_b3_v3_300m
)
replicates=(1 2 3 4 5)

check_external_source() {
    [[ -d "$source_root" ]] || { echo "Missing source directory: $source_root" >&2; exit 1; }
    (
        cd "$source_root"
        sha256sum -c --quiet "$source_checksums"
    )
}

check_archive_assets() {
    [[ $(awk 'END {print NR - 1}' "$windows_tsv") -eq 45 ]]
    [[ -s "$crop_manifest" ]]
    for zone in "${zones[@]}"; do
        map_root="$map_parent/$zone/map/sionna"
        testset_root="$testset_parent/${zone}_30min_opaque_buildings_no_vehicle_blockers/testset"
        [[ -s "$map_root/${zone}_scene.xml" ]]
        [[ -s "$map_root/${zone}_scene_manifest.json" ]]
        [[ -s "$map_root/${zone}_radio_bounds.net.xml" ]]
        [[ -s "$testset_root/${zone}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz" ]]
        for replicate in "${replicates[@]}"; do
            awk -F '\t' -v zone="$zone" -v replicate="$replicate" \
                '$1 == zone && $2 == replicate {found = 1} END {exit !found}' \
                "$windows_tsv"
        done
    done
}

check_environment() {
    command -v "$python_bin" >/dev/null 2>&1 || [[ -x "$python_bin" ]]
    command -v "$sumo_bin" >/dev/null 2>&1 || [[ -x "$sumo_bin" ]]
    "$python_bin" -c '
from importlib.metadata import version
import numpy
for package in ("drjit", "mitsuba", "sionna-rt"):
    print(package, version(package))
'
    "$sumo_bin" --version >/dev/null
}

verify_outputs() {
    local checked=0
    local zone replicate replicate_root mobility trace
    for zone in "${zones[@]}"; do
        for replicate in "${replicates[@]}"; do
            replicate_root="$output_root/replicates/$zone/rep${replicate}"
            mobility="$replicate_root/mobility/${zone}_rep${replicate}_all_vehicles_1s_full1800.json"
            trace="$replicate_root/rssi/${zone}_rep${replicate}_vehicles_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz"
            [[ -s "$mobility" && -s "$trace" ]]
            "$python_bin" - "$mobility" "$trace" <<'PY'
import json
import sys
import numpy as np

mobility_path, trace_path = sys.argv[1:]
with open(mobility_path, encoding="utf-8") as stream:
    mobility = json.load(stream)
assert mobility["format"] == "sumo_crop_mobility_trace_v1"
assert mobility["max_step"] == 1799
assert len(mobility["active_counts_by_step"]) == 1800
with np.load(trace_path, allow_pickle=False) as archive:
    meta = json.loads(str(archive["meta_json"].item()))
    assert meta["format"] == "sumo_rssi_trace_v3"
    assert meta["start_step"] == 0 and meta["end_step"] == 1799
    assert len(archive["node_active"]) == 1800
PY
            checked=$((checked + 1))
        done
    done
    echo "Verified $checked map/replicate mobility and RSSI pairs in $output_root"
}

if [[ "$mode" == --verify ]]; then
    check_environment
    verify_outputs
    exit 0
fi

check_external_source
check_archive_assets
check_environment

if [[ "$mode" == --check ]]; then
    echo "Source, environment, nine maps, 45 windows, and nine fixed test sets are ready."
    exit 0
fi

command -v sbatch >/dev/null 2>&1
[[ "$max_concurrent" =~ ^[1-9][0-9]*$ ]]
case "$archive_root$output_root$source_root" in
    *,*) echo "Paths containing commas are not supported by Slurm --export." >&2; exit 1 ;;
esac

mobility_script="$code_root/SUMO/luxembourg_real_city/run_ci_temporal_mobility.sh"
chunk_script="$code_root/SUMO/luxembourg_real_city/run_ci_factorial_zone_rssi_chunk.sh"
merge_script="$code_root/SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh"
for required in "$mobility_script" "$chunk_script" "$merge_script"; do
    [[ -s "$required" ]]
done

scheduler_args=()
[[ -n "${SBATCH_ACCOUNT:-}" ]] && scheduler_args+=(--account="$SBATCH_ACCOUNT")
[[ -n "${SBATCH_PARTITION:-}" ]] && scheduler_args+=(--partition="$SBATCH_PARTITION")
[[ -n "${SBATCH_EXCLUDE:-}" ]] && scheduler_args+=(--exclude="$SBATCH_EXCLUDE")

submit_job() {
    if [[ -n "$slurm_config" ]]; then
        env SLURM_CONF="$slurm_config" sbatch --parsable "${scheduler_args[@]}" "$@"
    else
        sbatch --parsable "${scheduler_args[@]}" "$@"
    fi
}

mkdir -p "$output_root/logs"
submission_manifest="$output_root/submitted_input_jobs.tsv"
if [[ -e "$submission_manifest" ]]; then
    echo "Refusing duplicate submission; manifest already exists: $submission_manifest" >&2
    exit 1
fi
printf 'zone\treplicate\tstage\tjob_id\tdependency\toutput\n' > "$submission_manifest"

for zone in "${zones[@]}"; do
    short=${zone%_300m}
    map_root="$map_parent/$zone/map"
    for replicate in "${replicates[@]}"; do
        replicate_root="$output_root/replicates/$zone/rep${replicate}"
        mobility="$replicate_root/mobility/${zone}_rep${replicate}_all_vehicles_1s_full1800.json"
        trace="$replicate_root/rssi/${zone}_rep${replicate}_vehicles_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz"
        mkdir -p "$replicate_root/logs" "$replicate_root/rssi/shards"

        mobility_id=$(submit_job \
            --job-name="input-${short#factor_}-r${replicate}-mob" \
            --output="$replicate_root/logs/mobility-%j.out" \
            --error="$replicate_root/logs/mobility-%j.err" \
            --export="ALL,REPO_ROOT=$code_root,PYTHON_BIN=$python_bin,SUMO_BIN=$sumo_bin,SOURCE_ROOT=$source_root,CI_DATA_ROOT=$output_root,WINDOWS_TSV=$windows_tsv,CROP_MANIFEST=$crop_manifest,CROP_NAME=$zone,REPLICATE=$replicate" \
            "$mobility_script")
        printf '%s\t%s\tmobility\t%s\t-\t%s\n' \
            "$zone" "$replicate" "$mobility_id" "$mobility" >> "$submission_manifest"

        ray_id=$(submit_job \
            --job-name="input-${short#factor_}-r${replicate}-ray" \
            --array="0-179%${max_concurrent}" \
            --output="$replicate_root/logs/rssi-%A_%a.out" \
            --error="$replicate_root/logs/rssi-%A_%a.err" \
            --dependency="afterok:${mobility_id}" \
            --export="ALL,REPO_ROOT=$code_root,PYTHON_BIN=$python_bin,SUMO_ROOT=$source_root,CROP_NAME=$zone,DATA_ROOT=$replicate_root,MAP_ROOT=$map_root,MOBILITY=$mobility,FRAME_STRIDE=10" \
            "$chunk_script")
        printf '%s\t%s\traytrace\t%s\tafterok:%s\t%s\n' \
            "$zone" "$replicate" "$ray_id" "$mobility_id" "$replicate_root/rssi/shards" >> "$submission_manifest"

        merge_id=$(submit_job \
            --job-name="input-${short#factor_}-r${replicate}-merge" \
            --output="$replicate_root/logs/merge-%j.out" \
            --error="$replicate_root/logs/merge-%j.err" \
            --dependency="afterok:${ray_id}" \
            --export="ALL,REPO_ROOT=$code_root,PYTHON_BIN=$python_bin,DATA_ROOT=$replicate_root,MERGED_TRACE_OUTPUT=$trace,MIN_RSSI_DBM=-100" \
            "$merge_script")
        printf '%s\t%s\tmerge\t%s\tafterok:%s\t%s\n' \
            "$zone" "$replicate" "$merge_id" "$ray_id" "$trace" >> "$submission_manifest"
    done
done

echo "Submitted 45 mobility, ray-tracing, and merge pipelines."
echo "Submission manifest: $submission_manifest"
echo "After completion, run:"
echo "  PYTHON_BIN=$python_bin bash $script_dir/generate_main_inputs.sh --verify $output_root"
