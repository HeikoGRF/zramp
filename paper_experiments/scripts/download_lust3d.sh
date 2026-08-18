#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 DESTINATION_DIRECTORY" >&2
    echo "Downloads and verifies LuST3D 1.0.0, then extracts it below DESTINATION_DIRECTORY." >&2
}

[[ $# -eq 1 ]] || { usage; exit 2; }

destination=$1
archive="$destination/LuST3d.zip"
source_root="$destination/lust3d_v1"
url='https://zenodo.org/records/20799415/files/LuST3d.zip?download=1'
expected_sha256='7d86b3ee4f5bbbfe3898ab3561d4ea13290eba1e33b0d55b29d4f2d50d897d3e'

mkdir -p "$destination"
if [[ ! -s "$archive" ]]; then
    command -v curl >/dev/null
    curl --fail --location "$url" --output "$archive"
fi

printf '%s  %s\n' "$expected_sha256" "$archive" | sha256sum --check -

if [[ ! -d "$source_root" ]]; then
    command -v unzip >/dev/null
    mkdir -p "$source_root"
    unzip -q "$archive" -d "$source_root"
fi

required=(
    lust3d.net.xml
    lust3d.poly.xml
    buslines.rou.xml
    local.0.rou.xml
    local.1.rou.xml
    local.2.rou.xml
    transit.rou.xml
    vtypes.add.xml
)
for name in "${required[@]}"; do
    [[ -s "$source_root/$name" ]] || {
        echo "Extracted LuST3D file is missing: $source_root/$name" >&2
        exit 1
    }
done

echo "LuST3D is ready at: $source_root"

