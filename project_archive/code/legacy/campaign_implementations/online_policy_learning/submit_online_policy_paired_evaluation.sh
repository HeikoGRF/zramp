#!/bin/bash
#SBATCH --job-name=bz150-ev-pair
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal,gpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%4
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/bz150-ev-pair-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/bz150-ev-pair-%A_%a.err

set -euo pipefail

METHODS=(random policy random policy)
SEEDS=(1 1 2 2)
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
METHOD=${PAIR_METHOD_OVERRIDE:-${METHODS[$TASK]}}
SEED=${PAIR_SEED_OVERRIDE:-${SEEDS[$TASK]}}
CARS=${PAIR_CARS:-40}
REGULARS=${PAIR_REGULARS:-4}
SIM_STEPS=${PAIR_SIM_STEPS:-300}
FINAL_STEPS=${PAIR_FINAL_STEPS:-100,200,300}
EXPERIMENT=${PAIR_EXPERIMENT:-single_zone_urban_150_mergeable_information_policy_v1}
SCENARIO_NAME=${PAIR_SCENARIO_NAME:-single_zone_urban_150}
WINDOW_STEPS=${PAIR_WINDOW_STEPS:-20}
MAX_WALL_SECONDS=${PAIR_MAX_WALL_SECONDS:-1700}
printf -v CARS_PAD '%03d' "$CARS"
printf -v REGULARS_PAD '%02d' "$REGULARS"
printf -v SEED_PAD '%02d' "$SEED"

ROOT=/home/hgraef/zramp-workspace
SCRATCH_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu
PY=$SCRATCH_ROOT/miniforge3/envs/radiodiff-replay/bin/python
SCENARIO=$ROOT/SUMO/$SCENARIO_NAME
SOURCE=${PAIR_SOURCE_ROOT:-$SCRATCH_ROOT/data/single_zone_urban_150_roles_factorial_v1/n${CARS_PAD}_r${REGULARS_PAD}}
TRACE=${PAIR_TRACE:-$SOURCE/traces/seed_$SEED_PAD.npz}
CONFIG=${PAIR_CONFIG:-$SOURCE/route_plans/seed_$SEED_PAD/roles.sumocfg}
RUN_DIR=$SCRATCH_ROOT/data/$EXPERIMENT/n${CARS_PAD}_r${REGULARS_PAD}/s${WINDOW_STEPS}/$METHOD/seed_$SEED_PAD
LOCAL_ROOT=/tmp/hgraef_bz150_ev_pair_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
LOCAL_TRACE=$LOCAL_ROOT/trace.npz
LOCAL_RUN=$LOCAL_ROOT/results

finalize() {
  status=$?
  trap - EXIT
  set +e
  if [ -d "$LOCAL_RUN" ]; then
    mkdir -p "$RUN_DIR"
    rsync -a "$LOCAL_RUN/" "$RUN_DIR/"
  fi
  rm -rf -- "$LOCAL_ROOT"
  exit "$status"
}
trap finalize EXIT

THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_DYNAMIC=FALSE
export TORCH_NUM_THREADS=$THREADS
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=$LOCAL_ROOT/tmp
export XDG_CACHE_HOME=$LOCAL_ROOT/xdg-cache
export MPLCONFIGDIR=$LOCAL_ROOT/matplotlib

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$LOCAL_RUN" "$SCRATCH_ROOT/logs"
rsync -a "$TRACE" "$LOCAL_TRACE"

MODULE=online_policy_learning.run_online_local_validation_policy
METHOD_ARGS=()
if [ "$METHOD" = policy ]; then
  MODULE=online_policy_learning.run_online_policy_variants
  METHOD_ARGS=(
    --corrected-variant frozen-samples
    --share-training-samples
    --policy-sample-capacity 512
    --policy-sample-bundle-capacity 32
    --head-replay-batches-per-step 2
    --policy-min-samples 32
    --policy-exploration-start 0.35
    --policy-exploration-decay-samples 256
    --normalize-policy-rewards
    --policy-ranking-loss-weight 1.0
    --policy-ranking-margin-db 0.02
    --policy-ranking-temperature-db 0.10
    --policy-ranking-receiver-cosine-min 0.80
  )
fi

cd "$ROOT"
echo "Starting method=$METHOD N=$CARS R=$REGULARS S=$WINDOW_STEPS seed=$SEED host=$(hostname)"
"$PY" -u -m "$MODULE" "$SEED" "$CARS" 1 \
  --selection-mode "$METHOD" \
  --token-window-steps "$WINDOW_STEPS" \
  --policy-warmup-steps 0 \
  --contact-aware-window-timing \
  --match-random-pull-opportunities \
  --unconditional-evidence-union \
  --mergeable-max-delta-rows 1 \
  --policy-training-target information-gain \
  --symmetric-pulls \
  --policy-reward-metric normalized-improvement \
  --policy-reward-scope directional \
  --realistic-network \
  --network-candidate-top-k 0 \
  --network-resource-count 4 \
  --network-bandwidth-hz 10000000 \
  --network-direction-airtime-s 0.125 \
  --network-efficiency 0.6 \
  --network-max-spectral-efficiency 6 \
  --network-min-sinr-db 5 \
  --network-missing-power-dbm -120 \
  --model-transfer-snr-min-db 57.5 \
  --results-dir "$LOCAL_RUN" \
  --measurement-trace-in "$LOCAL_TRACE" \
  --sumo-config "$CONFIG" \
  --sumo-net "$SCENARIO/${SCENARIO_NAME}.net.xml" \
  --dynamic-map "$SCENARIO/${SCENARIO_NAME}_dynamic.json" \
  --sim-steps "$SIM_STEPS" \
  --num-zones 1 \
  --noise-floor-dbm -105 \
  --snr-min-db 5 \
  --rssi-model mergeable-evidence \
  --mergeable-basis-dim 192 \
  --mergeable-ridge 1 \
  --predictor-prior max-loss \
  --predictor-time \
  --predictor-time-step-duration 1 \
  --predictor-time-unit 1 \
  --predictor-learned-time-scale 1000 \
  --learned-time-scale 1000 \
  --predictor-zone-local-coordinates \
  --local-sample-weighting spatial-balanced \
  --local-spatial-balance-bins 4 \
  --merge-strategy average \
  --aux-baselines none \
  --exact-hidden-dim 32 \
  --gain-hidden-dim 64 \
  --embedding-dim 64 \
  --pair-feature-mode relational \
  --exploration-prob 0.10 \
  --validation-capacity 400 \
  --diagnostic-regular-count "$REGULARS" \
  --fidelity-eval-every 0 \
  --fidelity-pairs-per-zone 500 \
  --final-fidelity-pairs-per-zone 500 \
  --final-steps "$FINAL_STEPS" \
  --progress-every 25 \
  --log-rmse-every 0 \
  --flush-every 25 \
  --max-wall-seconds "$MAX_WALL_SECONDS" \
  --quiet \
  "${METHOD_ARGS[@]}"

echo "Completed method=$METHOD at $(date -Is)"
