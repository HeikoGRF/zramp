The current five-replicate results depend on the following files and artifact trees. The numerical results were read primarily from each run’s `metrics.json` and `sharing_events.csv`.

## 1. Sweep orchestration

Main methods and temporal repetitions:

- [submit_factorial_ci_temporal_replicates.sh](SUMO/luxembourg_real_city/submit_factorial_ci_temporal_replicates.sh) — replicates 2–5 for ISO, Central, All, Top-5, Top-1, and Every5–80.
- [submit_factorial_synchronized_interval_sweep_paper_final.sh](SUMO/luxembourg_real_city/submit_factorial_synchronized_interval_sweep_paper_final.sh) — replicate 1 for Every5–80.
- [submit_factorial_ci_ungated_equal_greedy.sh](SUMO/luxembourg_real_city/submit_factorial_ci_ungated_equal_greedy.sh) — all five ungated-greedy repetitions.
- [submit_factorial_support_baselines.sh](SUMO/luxembourg_real_city/submit_factorial_support_baselines.sh) — local and centralized baselines.
- [submit_factorial_pq_sweep.sh](SUMO/luxembourg_real_city/submit_factorial_pq_sweep.sh) — learned Expert Bank penalty sweep.

Submission manifests:

- [temporal jobs](../../results/paper/luxembourg_cell_grid_ci_temporal5_paper_final_v1/submitted_jobs.tsv)
- [ungated-greedy jobs](../../results/paper/luxembourg_cell_grid_ci_temporal5_paper_final_v1/submitted_ungated_equal_greedy_jobs.tsv)
- [replicate-1 interval jobs](../../results/paper/luxembourg_cell_grid_synchronized_9map_paper_final_v1/submitted_jobs.tsv)
- [expert-bank penalty jobs](../../../project_archive/results/legacy_experiments/luxembourg_factorial_3x3_benchmark/submitted_pq4x256_kappa_sweep_jobs.tsv)
- [original support baselines](../../../project_archive/results/legacy_experiments/luxembourg_factorial_3x3_benchmark/submitted_support_baseline_jobs.tsv)

## 2. Temporal-window and radio-trace preparation

- [run_ci_lust_full_day_scan.sh](SUMO/luxembourg_real_city/run_ci_lust_full_day_scan.sh)
- [run_ci_select_temporal_windows.sh](SUMO/luxembourg_real_city/run_ci_select_temporal_windows.sh)
- [select_ci_temporal_windows.py](SUMO/luxembourg_real_city/select_ci_temporal_windows.py)
- [run_ci_temporal_mobility.sh](SUMO/luxembourg_real_city/run_ci_temporal_mobility.sh)
- [export_crop_mobility_trace.py](SUMO/luxembourg_real_city/export_crop_mobility_trace.py)
- [scan_fcd_crops.py](SUMO/luxembourg_real_city/scan_fcd_crops.py)
- [run_ci_factorial_zone_rssi_chunk.sh](SUMO/luxembourg_real_city/run_ci_factorial_zone_rssi_chunk.sh)
- [run_evaluation_zone_rssi_frame.sh](SUMO/luxembourg_real_city/run_evaluation_zone_rssi_frame.sh)
- [generate_pilot_rssi_trace.py](SUMO/luxembourg_real_city/generate_pilot_rssi_trace.py)
- [run_evaluation_zone_merge.sh](SUMO/luxembourg_real_city/run_evaluation_zone_merge.sh)
- [merge_rssi_trace_shards.py](SUMO/luxembourg_real_city/merge_rssi_trace_shards.py)
- [run_evaluation_zone_testset.sh](SUMO/luxembourg_real_city/run_evaluation_zone_testset.sh)
- [generate_street_testset.py](SUMO/luxembourg_real_city/generate_street_testset.py)
- [run_factorial_zone_prepare.sh](SUMO/luxembourg_real_city/run_factorial_zone_prepare.sh)
- [build_crop_sionna_scene.py](SUMO/luxembourg_real_city/build_crop_sionna_scene.py)

The archived generation-only entry point and provenance are:

- [generate_main_inputs.sh](../../input_data/generation/generate_main_inputs.sh)
- [generation instructions](../../input_data/generation/README.md)
- [external LuST3D file checksums](../../input_data/generation/external_lust3d_files.sha256)
- [omitted intermediate inventory](../../input_data/generation/omitted_generated_inputs.tsv)

The authoritative temporal design is retained under:

```text
input_data/prepared_traces/luxembourg_ci_temporal5_paper_final_v1/
├── selected_windows.tsv
└── selected_windows.json
```

Fixed test sets remain at their original paths under:

```text
input_data/prepared_traces/luxembourg_real_city/
  factor_b{1..3}_v{1..3}_300m_30min_opaque_buildings_no_vehicle_blockers/
    testset/*.npz
```

Generated mobility/RSSI files are written to external working storage. Their
original paths, sizes, roles, and SHA-256 digests are recorded in the omitted
intermediate inventory rather than bundled in the submission.

## 3. Maps and factorial metadata

- [factorial_zones_crop_manifest.json](../../input_data/zone_metadata/factorial_zones_crop_manifest.json) — spatial bounds of the nine maps.
- [proposed_zones.csv](../../input_data/zone_metadata/proposed_zones.csv) — building fractions and vehicle-level metadata.
- [proposed_zones.json](../../input_data/zone_metadata/proposed_zones.json)
- [survey_candidate_zones.py](SUMO/luxembourg_real_city/survey_candidate_zones.py)

Per-map geometry follows this pattern:

```text
input_data/maps/luxembourg_real_city/factorial_zones/
  factor_b{1..3}_v{1..3}_300m/map/
  ├── sionna/
  │   ├── *_scene.xml
  │   ├── *_scene_manifest.json
  │   ├── *_radio_bounds.net.xml
  │   ├── *_buildings_buffer200m.ply
  │   └── *_terrain_ground_buffer200m.ply
  └── terrain/
      ├── *.tif
      └── *.xyz
```

## 4. Simulation implementation

Main method, local/central modes, support-all/Top-\(k\), interval variants, and the exploratory learned Expert Bank:

- [submit_support_expert_bank.sh](experiments/place_wallis_benchmark/submit_support_expert_bank.sh)
- [run_support_expert_bank.py](experiments/place_wallis_benchmark/run_support_expert_bank.py)
- [cell_grid_methods.py](experiments/place_wallis_benchmark/cell_grid_methods.py)
- [cell_grid_support.py](experiments/place_wallis_benchmark/cell_grid_support.py)
- [training_utils.py](experiments/place_wallis_benchmark/training_utils.py)
- [tail_metrics.py](experiments/place_wallis_benchmark/tail_metrics.py)

Ungated equal-greedy baseline:

- [submit_equal_greedy.sh](experiments/place_wallis_benchmark/submit_equal_greedy.sh)
- [run_equal_greedy.py](experiments/place_wallis_benchmark/run_equal_greedy.py)

Required shared runtime:

- [shared-runtime explanation](../shared_runtime/README.md)
- [generalized support core](../shared_runtime/zramp_runtime/support_expert_bank.py)
- [SUMO trace-replay engine](../shared_runtime/SUMO/sumo_rl.py)
- [SUMO/Sionna map](../shared_runtime/SUMO/sumo_sionna_map.py)
- [dynamic obstacles](../shared_runtime/SUMO/dynamic_obstacles.py)
- [predictor model](../shared_runtime/model.py)
- [simulation configuration](../shared_runtime/rl_reward_experiment/config.py)

Acquisition model:

- [patch_grid_codec_model.py](experiments/support_acquisition_pretraining/patch_grid_codec_model.py)
- [pretrain_patch_grid_codec.py](experiments/support_acquisition_pretraining/pretrain_patch_grid_codec.py)
- [required acquisition helpers](../shared_runtime/experiments/support_acquisition_pretraining/)
- [cell-grid acquisition bundle](../../trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt)
- [Expert Bank bundle](../../../project_archive/trained_models/experimental/acquisition_pretraining/patch_grid_acquisition_v8_c16_pq4x256/bundle.pt)

The paper runner loads the cell-grid bundle to initialize its support
representation, but its intensity-count ranking and merge weights are
deterministic and recorded with `acquisition_model_used: false`. The patch-grid
Expert Bank bundle is used only by the separate development experiment.

## 5. Exact result trees used for the current five-replicate results

Replicate 1, ISO and Central:

```text
results/paper/luxembourg_cell_grid_factorial_sweeps_v1/methods/
  factor_b{1..3}_v{1..3}/
  cell_grid_{local_only,central}_eval50_tail10x25/
```

Replicate 1, All/Top-5/Top-1:

```text
results/paper/luxembourg_cell_grid_intensity_budget_sweep_v1/methods/
  factor_b{1..3}_v{1..3}/
  cell_grid_intensity_{greedy,top5,top1}_eval50_tail10x25/
```

Replicate 1, Every5–80:

```text
results/paper/luxembourg_cell_grid_synchronized_9map_paper_final_v1/methods/
  factor_b{1..3}_v{1..3}_300m/
  cell_grid_intensity_top1_global_every{5,10,20,40,80}_paper_final_full1800_eval50_tail10x25/
```

Replicates 2–5:

```text
results/paper/luxembourg_cell_grid_ci_temporal5_paper_final_v1/methods/
  factor_b{1..3}_v{1..3}/rep{2..5}/
  {iso,central,full,top5,every1,every5,every10,every20,every40,every80}_paper_final_full1800_eval50_tail10x25/
```

Ungated greedy, all five repetitions:

```text
results/paper/luxembourg_cell_grid_ci_temporal5_paper_final_v1/methods/
  factor_b{1..3}_v{1..3}/rep{1..5}/
  ungated_equal_greedy_paper_final_full1800_eval50_tail10x25/
```

Related learned-acquisition comparison (not consumed by the five-replicate aggregator):

```text
results/legacy_experiments/luxembourg_factorial_3x3_benchmark/methods/
  factor_b{1..3}_v{1..3}/
  support_expert_bank_pq4x256_s3650_cached_zero_width_kappa{1,2,5,10,20}_angle12_eval50_tail10x25/
```

Within every run directory, the relevant files are:

```text
metrics.json                         # overall, feasible and non-feasible tail RMSE
fidelity.csv                         # individual evaluation-time values
sharing_events.csv                   # communication bytes per step
config.json                          # complete run configuration
communication_overhead_assumptions.json
checkpoint_status.json
progress.json
slurm-*.out
slurm-*.err
```

`metrics.json` and `sharing_events.csv` are the files directly used for the reported numerical tables.

## 6. Final analysis, plot data, and manuscript files

Authoritative five-replicate analysis and plot-data export:

- [aggregate_paper_results_across_timeframes.py](analysis/aggregate_paper_results_across_timeframes.py)
- [build_plot_ready_paper_tables.py](analysis/build_plot_ready_paper_tables.py)

The resulting CSV tables are under `../../figures/data/`. The complete LaTeX/Overleaf work tree in the repository-root
`MANUSCRIPT.zip` provides the authoritative plotting and manuscript source.

The two archived analysis scripts are the authoritative pipeline for the reported five-replicate values. They recompute finite tail means, average the nine maps within each temporal replicate, calculate 95% Student-t confidence intervals across the five replicate means, and export validated plot-ready CSV files.

For B1V1-rep5 and B3V1-rep5, the aggregation must filter the shared `NaN` at tail step 1724 and average the remaining nine valid tail evaluations.
