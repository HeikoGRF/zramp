# Paper experiments

This is the authoritative reproduction path for the reported result. The design
contains nine factorial Luxembourg maps, five temporal realizations per map,
and eleven methods: ISO, Central, Full, Top 5, Top 1, five lower communication
frequencies, and ungated equal averaging.

The final method ranks candidate support by deterministic accumulated grid
intensity and uses intensity counts as merge weights. The runtime bundle under
`trained_models/paper_runtime/` initializes the support representation; the
reported ranking and merge weights do not use a learned acquisition score.

## Contents

- `code/final/`: paper preprocessing, simulation entry points, original
  Slurm launchers, and result aggregation.
- `code/shared_runtime/`: required simulator and model implementation.
- `environment/`: recorded Python, SUMO, Sionna, Mitsuba, Dr.Jit, NumPy,
  SciPy, Pandas, and PyTorch environment.
- `input_data/`: nine radio maps, fixed test sets, zone metadata, exact
  temporal windows, and input-generation provenance.
- `experiment_configs/paper/`: captured per-run configurations and submission
  manifests.
- `trained_models/paper_runtime/`: the small bundle required by the runner.
- `results/paper/`: the raw result trees used by the paper aggregation.
- `figures/`: regenerated CSV tables for the paper figures. The authoritative
  LaTeX/TikZ plotting code is provided in `../MANUSCRIPT.zip`.
- `scripts/`: portable high-level workflow helpers.

## 1. Create the environment

From this directory:

```bash
conda env create -f environment/environment.yml
conda activate zramp-archive
```

See [environment/README.md](environment/README.md) for the recorded versions
and CPU/GPU requirements. Trace generation is a substantial Sionna RT workload
and was designed for a Slurm cluster.

## 2. Download and verify LuST3D

Choose a working directory outside the repository:

```bash
bash scripts/download_lust3d.sh /path/to/zramp-work/source
```

The verified source will be
`/path/to/zramp-work/source/lust3d_v1`. The script checks the published
LuST3D 1.0.0 archive digest and the input-generation preflight checks every
extracted file used by the project.

## 3. Choose the temporal windows

For an exact reproduction, use the tracked table:

```text
input_data/prepared_traces/luxembourg_ci_temporal5_paper_final_v1/selected_windows.tsv
```

To repeat the selection procedure with a new full-day SUMO scan:

```bash
bash scripts/select_timeframes.sh \
  /path/to/zramp-work/source/lust3d_v1 \
  /path/to/zramp-work/window-selection
```

This writes `selected_windows.tsv` and `selected_windows.json`. Selecting new
windows defines a new experimental realization; it does not reproduce the
reported sample exactly.

## 4. Generate the 45 mobility and Sionna traces

First run the complete preflight:

```bash
bash input_data/generation/generate_main_inputs.sh \
  --check /path/to/zramp-work/source/lust3d_v1
```

Submit all 45 map/window pipelines using the stored paper windows:

```bash
SBATCH_ACCOUNT=your-account \
SBATCH_PARTITION=your-partition \
MAX_CONCURRENT_PER_ARRAY=20 \
  bash input_data/generation/generate_main_inputs.sh \
  --submit /path/to/zramp-work/source/lust3d_v1 \
           /path/to/zramp-work/generated-main-inputs
```

To use a newly selected table, add:

```bash
WINDOWS_TSV=/path/to/zramp-work/window-selection/selected_windows.tsv
```

After all Slurm dependency chains finish:

```bash
bash input_data/generation/generate_main_inputs.sh \
  --verify /path/to/zramp-work/generated-main-inputs
```

The generation details and exact radio settings are documented in
[input_data/generation/README.md](input_data/generation/README.md).

## 5. Run the simulations

The high-level runner validates every required trace, map, test set, and model
before doing any work. Without `--execute`, it only prints the commands:

```bash
python3 scripts/run_paper_experiments.py \
  --generated-input-root /path/to/zramp-work/generated-main-inputs
```

Run one representative subset first:

```bash
python3 scripts/run_paper_experiments.py \
  --generated-input-root /path/to/zramp-work/generated-main-inputs \
  --output-root /path/to/zramp-work/reproduced-results \
  --zones factor_b2_v2_300m \
  --replicates 1 \
  --methods iso central every10 \
  --execute
```

Run the complete 495-run matrix by omitting the filters:

```bash
python3 scripts/run_paper_experiments.py \
  --generated-input-root /path/to/zramp-work/generated-main-inputs \
  --output-root /path/to/zramp-work/reproduced-results \
  --execute
```

The runner is sequential and scheduler-independent. On a cluster, split the
printed dry-run commands into jobs or use the original Slurm launchers under
`code/final/SUMO/luxembourg_real_city/` after setting their path and scheduler
overrides.

## 6. Rebuild statistics and plotting tables

For the tracked raw paper results:

```bash
bash scripts/rebuild_paper_outputs.sh
```

For a complete reproduced result tree:

```bash
bash scripts/rebuild_paper_outputs.sh \
  /path/to/zramp-work/reproduced-results \
  /path/to/zramp-work/reproduced-tables
```

The aggregation first averages the nine maps within each temporal realization,
then computes the reported 95% Student-t confidence intervals across the five
temporal means.

## Exact dependency inventory

See
[code/final/FINAL_RESULTS_DEPENDENCIES.md](code/final/FINAL_RESULTS_DEPENDENCIES.md)
for the original launcher-to-input-to-result mapping and the four raw result
trees consumed by the paper.
