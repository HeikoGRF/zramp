#!/usr/bin/env bash
#SBATCH --job-name=ci-factor-rssi-chunk
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/hgraef/zramp-workspace}
FRAME_SCRIPT=${REPO_ROOT}/SUMO/luxembourg_real_city/run_evaluation_zone_rssi_frame.sh
FRAME_STRIDE=${FRAME_STRIDE:-10}

export FRAME_STRIDE
for ((frame_offset = 0; frame_offset < FRAME_STRIDE; frame_offset++)); do
    FRAME_OFFSET=${frame_offset} bash "${FRAME_SCRIPT}"
done
