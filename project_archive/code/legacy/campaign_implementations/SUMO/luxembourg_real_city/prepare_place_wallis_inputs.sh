#!/usr/bin/env bash
# Build the Place Wallis 300 m scene and export all 1-second in-zone vehicles.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers}
SOURCE_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie
PLACE_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/place_wallis
MANIFEST=${REPO_ROOT}/SUMO/luxembourg_real_city/place_wallis_crop_manifest.json
TERRAIN_URL=https://download.data.public.lu/resources/bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/20241223-093912/MNT_Lidar2024.tif
TERRAIN_TIF=${PLACE_ROOT}/map/terrain/place_wallis_buffer200m_dtm_2024_10m_utm32.tif
TERRAIN_XYZ=${PLACE_ROOT}/map/terrain/place_wallis_buffer200m_dtm_2024_10m_utm32.xyz
GDAL_TRANSLATE_BIN=${GDAL_TRANSLATE_BIN:-/usr/pack/gdal-3.x-sr/envs/gdal/bin/gdal_translate}
MOBILITY=${DATA_ROOT}/mobility/place_wallis_all_vehicles_0745_0815_1s_full1800.json

mkdir -p "${PLACE_ROOT}/map/terrain" "${PLACE_ROOT}/map/sionna" "${DATA_ROOT}/mobility"

if [[ ! -s "${TERRAIN_TIF}" ]]; then
    command -v gdalwarp >/dev/null
    gdalwarp \
        -overwrite \
        -t_srs EPSG:32632 \
        -te 292598.66 5498098.13 293298.66 5498798.13 \
        -tr 10 10 \
        -r bilinear \
        -dstnodata nan \
        "/vsicurl/${TERRAIN_URL}" \
        "${TERRAIN_TIF}"
fi

if [[ ! -s "${TERRAIN_XYZ}" ]]; then
    test -x "${GDAL_TRANSLATE_BIN}"
    "${GDAL_TRANSLATE_BIN}" -of XYZ "${TERRAIN_TIF}" "${TERRAIN_XYZ}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/build_crop_sionna_scene.py" \
    --crop-manifest "${MANIFEST}" \
    --crop place_wallis_300m \
    --sumo-net "${SOURCE_ROOT}/map/sumo/lust3d.net.xml" \
    --polygons "${SOURCE_ROOT}/map/sumo/lust3d.poly.xml" \
    --output-dir "${PLACE_ROOT}/map/sionna" \
    --buffer 200 \
    --z-reference 230 \
    --terrain-xyz "${TERRAIN_XYZ}" \
    --terrain-source-url "${TERRAIN_URL}"

if [[ ! -s "${MOBILITY}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/export_crop_mobility_trace.py" \
        --fcd "${SOURCE_ROOT}/traces/lust3d_0745_0815_period1.fcd.xml.gz" \
        --crop-manifest "${MANIFEST}" \
        --crop place_wallis_300m \
        --output "${MOBILITY}" \
        --begin 27900 \
        --steps 1799 \
        --sample-period 1 \
        --all-participants \
        --z-reference 230
fi

"${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["max_step"] == 1799; assert len(p["active_counts_by_step"]) == 1800; print("mobility participants={} active_min={} active_max={}".format(p["num_nodes"], min(p["active_counts_by_step"]), max(p["active_counts_by_step"])))' "${MOBILITY}"
printf 'PLACE_ROOT=%s\n' "${PLACE_ROOT}"
printf 'MOBILITY=%s\n' "${MOBILITY}"
