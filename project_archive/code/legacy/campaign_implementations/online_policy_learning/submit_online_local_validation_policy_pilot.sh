#!/bin/bash
#SBATCH --job-name=bz1nssweep
#SBATCH --account=disco-med
#SBATCH --partition=cpu.normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/single-zone-ns-sweep-%A_%a.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu/logs/single-zone-ns-sweep-%A_%a.err

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
VEHICLES=(10 20 30 40)
WINDOWS=(1 5 10 20)
METHODS=(policy random)
METHOD_INDEX=$((TASK_ID % 2))
INDEX=$((TASK_ID / 2))
WINDOW_INDEX=$((INDEX % 4))
INDEX=$((INDEX / 4))
SEED=$((INDEX % 2 + 1))
N_INDEX=$((INDEX / 2))
N=${VEHICLES[$N_INDEX]}
S=${WINDOWS[$WINDOW_INDEX]}
METHOD=${METHODS[$METHOD_INDEX]}
printf -v N_PAD '%03d' "$N"
printf -v SEED_PAD '%02d' "$SEED"
printf -v S_PAD '%02d' "$S"

ROOT=${ROOT_OVERRIDE:-/home/hgraef/zramp-workspace}
SCRATCH_ROOT=${SCRATCH_ROOT_OVERRIDE:-/usr/itetnas04/data-scratch-01/hgraef/data/radiodiff_grid_9z_dynamic1000_methods_x86gpu}
PY=${PY_OVERRIDE:-${SCRATCH_ROOT}/miniforge3/envs/radiodiff-replay/bin/python}
DATA_ROOT=${DATA_ROOT_OVERRIDE:-${SCRATCH_ROOT}/data}
SCENARIO=${SCENARIO_OVERRIDE:-${ROOT}/SUMO/single_zone_urban_220}
TRACE_ROOT=${TRACE_ROOT_OVERRIDE:-${DATA_ROOT}/single_zone_urban_220_roles_n_s_sionna_tail5_v1}
RESULTS_ROOT=${RESULTS_ROOT_OVERRIDE:-${DATA_ROOT}/single_zone_urban_220_roles_n_s_policy_random_s0102_v1}
STEM=single_zone_urban_220_roles_n${N_PAD}_seed${SEED_PAD}
CONFIG=${SCENARIO}/${STEM}.sumocfg
if [ "$N" -eq 20 ] && [ "$SEED" -eq 1 ]; then
  TRACE=${DATA_ROOT}/single_zone_urban_220_roles_sionna_tail5_v2/seed_01.npz
  CONFIG=${SCENARIO}/single_zone_urban_220_roles_seed01.sumocfg
else
  TRACE=${TRACE_ROOT}/n${N_PAD}/seed_${SEED_PAD}.npz
fi
RUN_DIR=${RESULTS_ROOT}/n${N_PAD}/s${S_PAD}/${METHOD}/seed_${SEED_PAD}
LOCAL_ROOT=${LOCAL_BASE_OVERRIDE:-/tmp}/hgraef_bz1_ns_${SLURM_ARRAY_JOB_ID}_${TASK_ID}
LOCAL_TRACE=${LOCAL_ROOT}/trace.npz
LOCAL_RUN=${LOCAL_ROOT}/results
THREADS=${SLURM_CPUS_PER_TASK:-8}

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
export VECLIB_MAXIMUM_THREADS=$THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export TORCH_NUM_THREADS=$THREADS
export TORCH_NUM_INTEROP_THREADS=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TMPDIR=${LOCAL_ROOT}/tmp
export XDG_CACHE_HOME=${LOCAL_ROOT}/xdg-cache
export MPLCONFIGDIR=${LOCAL_ROOT}/matplotlib

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$LOCAL_RUN" "${SCRATCH_ROOT}/logs"
if [ ! -f "$TRACE" ]; then
  echo "Missing trace: $TRACE" >&2
  exit 3
fi
rsync -a "$TRACE" "$LOCAL_TRACE"
cd "$ROOT"
echo "Starting N=$N S=$S method=$METHOD seed=$SEED on CPU host=$(hostname) at $(date -Is)"
"$PY" -u -m online_policy_learning.run_online_local_validation_policy "$SEED" "$N" 1 \
  --selection-mode "$METHOD" --token-window-steps "$S" \
  --validation-capacity 400 --diagnostic-regular-count 2 \
  --model-transfer-snr-min-db 57.5 \
  --results-dir "$LOCAL_RUN" --measurement-trace-in "$LOCAL_TRACE" \
  --sumo-config "$CONFIG" \
  --sumo-net "$SCENARIO/single_zone_urban_220.net.xml" \
  --dynamic-map "$SCENARIO/single_zone_urban_220_dynamic.json" \
  --sim-steps 1000 --num-zones 1 \
  --noise-floor-dbm -105 --snr-min-db 5 \
  --predictor-prior max-loss --rssi-model small \
  --predictor-time --merge-strategy average --aux-baselines none \
  --exact-hidden-dim 8 --gain-hidden-dim 64 \
  --pair-feature-mode relational \
  --embedding-dim 32 --learned-time-dim 16 \
  --exploration-prob 0.10 --head-replay-batches-per-step 1 \
  --local-sample-weighting uniform \
  --aggregation-tolerance 0.01 --aggregation-max-iterations 16 \
  --fidelity-eval-every 0 --fidelity-pairs-per-zone 50 \
  --final-fidelity-pairs-per-zone 500 \
  --final-steps 900,925,950,975,1000 \
  --progress-every 25 --log-rmse-every 0 --flush-every 25 --quiet

echo "Completed N=$N S=$S method=$METHOD seed=$SEED at $(date -Is)"
