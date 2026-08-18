#!/bin/bash
#SBATCH --job-name=bz150time
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal,gpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-179%20
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/single-zone-150-corrected-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/single-zone-150-corrected-%A_%a.err

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
N_VALUES=(10 10 20 20 40 40)
R_VALUES=(1 2 2 4 4 8)
WINDOWS=(1 2 5 10 20)
METHODS=(frozen_samples shared_policy random)

METHOD=${METHODS[$((TASK_ID % 3))]}
INDEX=$((TASK_ID / 3))
S=${WINDOWS[$((INDEX % 5))]}
INDEX=$((INDEX / 5))
SEED=$((INDEX % 2 + 1))
COMBO=$((INDEX / 2))
N=${N_VALUES[$COMBO]}
R=${R_VALUES[$COMBO]}

printf -v N_PAD '%03d' "$N"
printf -v R_PAD '%02d' "$R"
printf -v S_PAD '%02d' "$S"
printf -v SEED_PAD '%02d' "$SEED"

ROOT=/home/hgraef/zramp-workspace
SCRATCH_ROOT=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu
PY=$SCRATCH_ROOT/miniforge3/envs/radiodiff-replay/bin/python
SCENARIO=$ROOT/SUMO/single_zone_urban_150
RESULTS_ROOT=$SCRATCH_ROOT/data/single_zone_urban_150_contact_timing_sweep_v1
RUN_DIR=$RESULTS_ROOT/n${N_PAD}_r${R_PAD}/s$S_PAD/$METHOD/seed_$SEED_PAD

if [ "$N" -eq 20 ] && [ "$R" -eq 2 ]; then
  SOURCE_ROOT=$SCRATCH_ROOT/data/single_zone_urban_150_roles_sweep_v1
  TRACE=$SOURCE_ROOT/traces/seed_$SEED_PAD.npz
  CONFIG=$SOURCE_ROOT/route_plans/seed_$SEED_PAD/roles.sumocfg
else
  SOURCE_ROOT=$SCRATCH_ROOT/data/single_zone_urban_150_roles_factorial_v1/n${N_PAD}_r${R_PAD}
  TRACE=$SOURCE_ROOT/traces/seed_$SEED_PAD.npz
  CONFIG=$SOURCE_ROOT/route_plans/seed_$SEED_PAD/roles.sumocfg
fi

LOCAL_ROOT=/tmp/hgraef_bz150_corrected_${SLURM_ARRAY_JOB_ID}_$TASK_ID
LOCAL_TRACE=$LOCAL_ROOT/trace.npz
LOCAL_RUN=$LOCAL_ROOT/results
THREADS=${SLURM_CPUS_PER_TASK:-8}
CHECKPOINTS=100,200,300,400,500,600,700,800,900,1000

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

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export TORCH_NUM_THREADS=$THREADS
export TORCH_NUM_INTEROP_THREADS=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=$LOCAL_ROOT/tmp
export XDG_CACHE_HOME=$LOCAL_ROOT/xdg-cache
export MPLCONFIGDIR=$LOCAL_ROOT/matplotlib

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$LOCAL_RUN" "$SCRATCH_ROOT/logs"
if [ ! -f "$TRACE" ]; then
  echo "Missing trace: $TRACE" >&2
  exit 3
fi
if [ ! -f "$CONFIG" ]; then
  echo "Missing SUMO config: $CONFIG" >&2
  exit 3
fi
rsync -a "$TRACE" "$LOCAL_TRACE"

WARMUP=0
SELECTION=random
MODULE=online_policy_learning.run_online_local_validation_policy
METHOD_ARGS=()
if [ "$METHOD" = frozen_samples ]; then
  SELECTION=policy
  MODULE=online_policy_learning.run_online_policy_variants
  METHOD_ARGS=(
    --corrected-variant frozen-samples
    --share-training-samples
    --policy-sample-capacity 512
    --policy-sample-bundle-capacity 32
    --head-replay-batches-per-step 1
    --policy-min-samples 32
    --policy-exploration-start 0.20
    --policy-exploration-decay-samples 128
  )
elif [ "$METHOD" = shared_policy ]; then
  SELECTION=policy
  MODULE=online_policy_learning.run_online_policy_variants
  METHOD_ARGS=(--corrected-variant shared-policy)
fi

cd "$ROOT"
echo "Starting corrected map=150 N=$N R=$R S=$S method=$METHOD seed=$SEED host=$(hostname)"
"$PY" -u -m "$MODULE" "$SEED" "$N" 1 \
  --selection-mode "$SELECTION" \
  --token-window-steps "$S" \
  --policy-warmup-steps "$WARMUP" \
  --contact-aware-window-timing \
  --policy-reward-metric normalized-improvement \
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
  --sumo-net "$SCENARIO/single_zone_urban_150.net.xml" \
  --dynamic-map "$SCENARIO/single_zone_urban_150_dynamic.json" \
  --sim-steps 1000 \
  --num-zones 1 \
  --noise-floor-dbm -105 \
  --snr-min-db 5 \
  --rssi-model small \
  --predictor-prior max-loss \
  --predictor-time \
  --predictor-time-step-duration 1 \
  --predictor-time-unit 1 \
  --predictor-learned-time-scale 1000 \
  --learned-time-scale 1000 \
  --predictor-zone-local-coordinates \
  --local-lr 0.001 \
  --local-batch-size 128 \
  --local-epochs 1 \
  --local-batches-per-step 2 \
  --local-initialization-anchor-strength 0.00001 \
  --local-sample-weighting spatial-balanced \
  --local-spatial-balance-bins 4 \
  --merge-strategy average \
  --aux-baselines none \
  --exact-hidden-dim 8 \
  --gain-hidden-dim 64 \
  --embedding-dim 64 \
  --pair-feature-mode relational \
  --exploration-prob 0.10 \
  --validation-capacity 400 \
  --diagnostic-regular-count "$R" \
  --aggregation-tolerance 0.01 \
  --aggregation-max-iterations 16 \
  --fidelity-eval-every 0 \
  --fidelity-pairs-per-zone 50 \
  --final-fidelity-pairs-per-zone 500 \
  --final-steps "$CHECKPOINTS" \
  --progress-every 25 \
  --log-rmse-every 0 \
  --flush-every 25 \
  --max-wall-seconds 7000 \
  --quiet \
  "${METHOD_ARGS[@]}"

echo "Completed corrected map=150 N=$N R=$R S=$S method=$METHOD seed=$SEED at $(date -Is)"
