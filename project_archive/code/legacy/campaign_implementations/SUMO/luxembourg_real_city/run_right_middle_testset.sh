#!/usr/bin/env bash
#SBATCH --job-name=bonnevoie-testset
#SBATCH --account=disco-med
#SBATCH --partition=gpu.normal
#SBATCH --exclude=arton10,arton11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/testset-%j.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/logs/testset-%j.err

set -euo pipefail

REPO_ROOT=/home/hgraef/zramp-workspace
PYTHON_BIN=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/miniforge3/envs/sionna-trace/bin/python
DATA_ROOT=${DATA_ROOT:-/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers}
MAP_ROOT=${REPO_ROOT}/SUMO/luxembourg_real_city/gare_bonnevoie/map
TASK_CACHE=/tmp/hgraef-drjit-testset-${SLURM_JOB_ID}
mkdir -p "${TASK_CACHE}" "${DATA_ROOT}/testset"

export MI_DEFAULT_VARIANT=llvm_ad_rgb
export DRJIT_CACHE_DIR="${TASK_CACHE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" SUMO/luxembourg_real_city/generate_street_testset.py \
    --scene "${MAP_ROOT}/sionna/gare_bonnevoie_balanced_scene.xml" \
    --scene-manifest "${MAP_ROOT}/sionna/gare_bonnevoie_balanced_scene_manifest.json" \
    --sumo-net-3d "${MAP_ROOT}/sumo/lust3d.net.xml" \
    --output "${TESTSET_OUTPUT:-${DATA_ROOT}/testset/right_middle_street_pairs_10000_opaque_no_vehicle_blockers_static.npz}" \
    --region-bounds 400 200 800 600 \
    --samples "${TESTSET_SAMPLES:-10000}" \
    --senders "${TESTSET_SENDERS:-200}" \
    --seed "${TESTSET_SEED:-1}" \
    --street-spacing-m 2 \
    --region-margin-m 0 \
    --min-distance-m 1 \
    --antenna-height-m 1.5 \
    --num-rays "${NUM_RAYS:-20000}" \
    --max-depth "${MAX_DEPTH:-3}" \
    --tx-batch-size "${TX_BATCH_SIZE:-20}" \
    --frequency-hz 3500000000 \
    --tx-power-dbm 23 \
    --rssi-min-dbm -120 \
    --rssi-max-dbm 0 \
    --disable-refraction
