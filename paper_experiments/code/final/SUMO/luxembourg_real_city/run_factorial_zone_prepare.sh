#!/usr/bin/env bash
#SBATCH --job-name=factor-zone-prep
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
SOURCE_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie
ZONE_PARENT=${ZONE_PARENT:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones}
MANIFEST=${MANIFEST:-${REPO_ROOT}/SUMO/luxembourg_real_city/factorial_zones_crop_manifest.json}
TERRAIN_URL=${TERRAIN_URL:-https://download.data.public.lu/resources/bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/20241223-093912/MNT_Lidar2024.tif}
GDAL_WARP_BIN=${GDAL_WARP_BIN:-/usr/sepp/bin/gdalwarp}
GDAL_TRANSLATE_BIN=${GDAL_TRANSLATE_BIN:-/usr/pack/gdal-3.x-sr/envs/gdal/bin/gdal_translate}

: "${CROP_NAME:?CROP_NAME must be set}"
test -x "${PYTHON_BIN}"
test -x "${GDAL_WARP_BIN}"
test -x "${GDAL_TRANSLATE_BIN}"

zone_root=${ZONE_PARENT}/${CROP_NAME}
map_root=${zone_root}/map
terrain_root=${map_root}/terrain
sionna_root=${map_root}/sionna
data_root=${DATA_PARENT}/${CROP_NAME}_30min_opaque_buildings_no_vehicle_blockers
mobility=${data_root}/mobility/${CROP_NAME}_all_vehicles_0745_0815_1s_full1800.json
terrain_tif=${terrain_root}/${CROP_NAME}_buffer200m_dtm_2024_10m_utm32.tif
terrain_xyz=${terrain_root}/${CROP_NAME}_buffer200m_dtm_2024_10m_utm32.xyz

read -r terrain_xmin terrain_ymin terrain_xmax terrain_ymax < <(
    "${PYTHON_BIN}" -c '
import json, sys
manifest, crop = sys.argv[1:]
x0, y0, x1, y1 = json.load(open(manifest))["crops"][crop]["bounds_sumo_xy_m"]
print(x0 + 285248.66, y0 + 5492198.13, x1 + 285648.66, y1 + 5492598.13)
' "${MANIFEST}" "${CROP_NAME}"
)

mkdir -p "${terrain_root}" "${sionna_root}" "${data_root}/mobility" \
    "${data_root}/logs" "${data_root}/rssi/shards" "${data_root}/testset"

if [[ ! -s "${terrain_tif}" ]]; then
    "${GDAL_WARP_BIN}" \
        -overwrite \
        -t_srs EPSG:32632 \
        -te "${terrain_xmin}" "${terrain_ymin}" "${terrain_xmax}" "${terrain_ymax}" \
        -tr 10 10 \
        -r bilinear \
        -dstnodata nan \
        "/vsicurl/${TERRAIN_URL}" \
        "${terrain_tif}"
fi

if [[ ! -s "${terrain_xyz}" ]]; then
    "${GDAL_TRANSLATE_BIN}" -of XYZ "${terrain_tif}" "${terrain_xyz}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/build_crop_sionna_scene.py" \
    --crop-manifest "${MANIFEST}" \
    --crop "${CROP_NAME}" \
    --sumo-net "${SOURCE_ROOT}/map/sumo/lust3d.net.xml" \
    --polygons "${SOURCE_ROOT}/map/sumo/lust3d.poly.xml" \
    --output-dir "${sionna_root}" \
    --buffer 200 \
    --z-reference 230 \
    --terrain-xyz "${terrain_xyz}" \
    --terrain-source-url "${TERRAIN_URL}"

if [[ ! -s "${mobility}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/export_crop_mobility_trace.py" \
        --fcd "${SOURCE_ROOT}/traces/lust3d_0745_0815_period1.fcd.xml.gz" \
        --crop-manifest "${MANIFEST}" \
        --crop "${CROP_NAME}" \
        --output "${mobility}" \
        --begin 27900 \
        --steps 1799 \
        --sample-period 1 \
        --all-participants \
        --z-reference 230
fi

"${PYTHON_BIN}" -c '
import json, sys
p = json.load(open(sys.argv[1])); c = p["active_counts_by_step"]
assert p["max_step"] == 1799 and len(c) == 1800
print("{} participants={} active_min={} active_median={} active_max={}".format(
    p["crop_name"], p["num_nodes"], min(c), sorted(c)[len(c)//2], max(c)
))
' "${mobility}"
