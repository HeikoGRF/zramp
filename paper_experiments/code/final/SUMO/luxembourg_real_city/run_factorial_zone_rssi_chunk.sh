#!/usr/bin/env bash
#SBATCH --job-name=factor-rssi-chunk
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --exclude=arton10,arton11

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
FRAME_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_rssi_frame.sh

for frame_offset in 0 1 2 3 4; do
    FRAME_OFFSET=${frame_offset} bash "${FRAME_SCRIPT}"
done
