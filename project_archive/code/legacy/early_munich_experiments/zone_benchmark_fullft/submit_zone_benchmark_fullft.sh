#!/bin/bash
#SBATCH --job-name=radio-zones-fullft
#SBATCH --time=02:00:00
#SBATCH --account=projects
#SBATCH --output=logs/radio-zones-fullft-%j.out
#SBATCH --error=logs/radio-zones-fullft-%j.err

source ~/.bashrc
conda activate sionna

cd /home/hgraef/zramp-workspace/zone_benchmark_fullft

mkdir -p logs results

srun --gpus 1 python -u run_zone_benchmark_fullft.py

