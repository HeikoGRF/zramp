#!/usr/bin/env bash
# Build four 300 m Luxembourg scenes and export their complete 1-second traffic cohorts.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
PYTHON_BIN=${PYTHON_BIN:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python}
DATA_PARENT=${DATA_PARENT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city}
SOURCE_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie
ZONE_PARENT=${REPO_ROOT}/SUMO/luxembourg_real_city/evaluation_zones
MANIFEST=${REPO_ROOT}/SUMO/luxembourg_real_city/evaluation_zones_crop_manifest.json
TERRAIN_URL=${TERRAIN_URL:-https://download.data.public.lu/resources/bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/20241223-093912/MNT_Lidar2024.tif}
GDAL_WARP_BIN=${GDAL_WARP_BIN:-gdalwarp}
GDAL_TRANSLATE_BIN=${GDAL_TRANSLATE_BIN:-/usr/pack/gdal-3.x-sr/envs/gdal/bin/gdal_translate}

ZONES=(ville_haute_300m belair_300m limpertsberg_s_300m hollerich_w_300m)

terrain_extent() {
    case "$1" in
        ville_haute_300m) printf '%s\n' '292198.66 5499198.13 292898.66 5499898.13' ;;
        belair_300m) printf '%s\n' '290698.66 5498548.13 291398.66 5499248.13' ;;
        limpertsberg_s_300m) printf '%s\n' '291898.66 5499848.13 292598.66 5500548.13' ;;
        hollerich_w_300m) printf '%s\n' '291298.66 5497648.13 291998.66 5498348.13' ;;
        *) return 1 ;;
    esac
}

test -x "${PYTHON_BIN}"
command -v "${GDAL_WARP_BIN}" >/dev/null
test -x "${GDAL_TRANSLATE_BIN}"

for crop in "${ZONES[@]}"; do
    zone_root=${ZONE_PARENT}/${crop}
    map_root=${zone_root}/map
    terrain_root=${map_root}/terrain
    sionna_root=${map_root}/sionna
    data_root=${DATA_PARENT}/${crop}_30min_opaque_buildings_no_vehicle_blockers
    mobility=${data_root}/mobility/${crop}_all_vehicles_0745_0815_1s_full1800.json
    terrain_tif=${terrain_root}/${crop}_buffer200m_dtm_2024_10m_utm32.tif
    terrain_xyz=${terrain_root}/${crop}_buffer200m_dtm_2024_10m_utm32.xyz
    read -r terrain_xmin terrain_ymin terrain_xmax terrain_ymax < <(terrain_extent "${crop}")

    mkdir -p "${terrain_root}" "${sionna_root}" "${data_root}/mobility" "${data_root}/logs" "${data_root}/rssi/shards" "${data_root}/testset"

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

    scene_args=(
        --crop-manifest "${MANIFEST}"
        --crop "${crop}"
        --sumo-net "${SOURCE_ROOT}/map/sumo/lust3d.net.xml"
        --polygons "${SOURCE_ROOT}/map/sumo/lust3d.poly.xml"
        --output-dir "${sionna_root}"
        --buffer 200
        --z-reference 230
        --terrain-xyz "${terrain_xyz}"
        --terrain-source-url "${TERRAIN_URL}"
    )
    if [[ "${crop}" == limpertsberg_s_300m ]]; then
        scene_args+=(--bridge-deck)
    fi
    "${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/build_crop_sionna_scene.py" "${scene_args[@]}"

    if [[ ! -s "${mobility}" ]]; then
        "${PYTHON_BIN}" "${REPO_ROOT}/SUMO/luxembourg_real_city/export_crop_mobility_trace.py" \
            --fcd "${SOURCE_ROOT}/traces/lust3d_0745_0815_period1.fcd.xml.gz" \
            --crop-manifest "${MANIFEST}" \
            --crop "${crop}" \
            --output "${mobility}" \
            --begin 27900 \
            --steps 1799 \
            --sample-period 1 \
            --all-participants \
            --z-reference 230
    fi

    "${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); c=p["active_counts_by_step"]; assert p["max_step"] == 1799; assert len(c) == 1800; print("{} participants={} active_min={} active_median={} active_max={}".format(p["crop_name"], p["num_nodes"], min(c), sorted(c)[len(c)//2], max(c)))' "${mobility}"
done

printf 'Prepared %d zones under %s\n' "${#ZONES[@]}" "${ZONE_PARENT}"
