# Deterministic single-model rules versus the learned Expert Bank

**Status:** development comparison; only the deterministic grid-intensity
methods were carried into the final paper experiment.

## Question

This experiment asked whether the complicated pretrained learned-acquisition
Expert Bank was necessary, and whether cumulative sample count or accumulated
cell-grid intensity was the more useful deterministic signal for selecting and
merging neighboring models.

All methods use the same first temporal realization of the nine factorial
Luxembourg maps: the 07:45--08:15 LuST window, 1,800 one-second steps, and the
same fixed 10,000-pair evaluation sets. The compared groups are:

- deterministic sample-count ranking and parameter weighting with Top-1,
  Top-2, Top-4, or all available models;
- deterministic cell-grid-intensity ranking and parameter weighting with
  Top-1, Top-5, or all available models;
- pretrained relative-gain acquisition with an uncapped Expert Bank at
  `kappa` values 0.02, 0.10, and 0.50; and
- local-only and idealized centralized references.

## Result

Lower tail RMSE is better. Means below average the available map-level results
within this one temporal realization.

| Method | Completed maps | Mean tail RMSE | Mean model transfers |
|---|---:|---:|---:|
| Centralized reference | 9/9 | 11.294 dB | not comparable |
| Grid intensity, Top-1 | 9/9 | 14.752 dB | 66,824 |
| Grid intensity, Top-5 | 9/9 | 14.782 dB | 331,129 |
| Grid intensity, all available | 9/9 | 14.844 dB | 2,600,027 |
| Sample count, Top-4 | 9/9 | 14.866 dB | 265,627 |
| Sample count, Top-1 | 9/9 | 14.887 dB | 66,824 |
| Sample count, Top-2 | 9/9 | 14.942 dB | 133,409 |
| Sample count, all available | 9/9 | 14.985 dB | 2,600,027 |
| Learned Expert Bank, `kappa=0.02` | 8/9 | 18.365 dB | 291,926 |
| Learned Expert Bank, `kappa=0.10` | 9/9 | 19.350 dB | 61,970 |
| Learned Expert Bank, `kappa=0.50` | 9/9 | 20.320 dB | 14,089 |
| Local only | 9/9 | 28.076 dB | 0 |

The main finding is that the deterministic one-model designs were much more
accurate than the learned Expert Bank. At similar transfer counts, intensity
Top-1 obtained 14.752 dB while the learned `kappa=0.10` method obtained
19.350 dB. Pulling additional models did not improve the deterministic
methods: Top-1 intensity was slightly better than Top-5 and unrestricted
pulling while transferring far fewer models.

The comparison between deterministic signals is much weaker. Grid-intensity
Top-1 averaged 14.752 dB and won five maps; sample-count Top-1 averaged
14.887 dB and won four. This 0.135 dB difference supports treating the two as
approximately tied, with a small descriptive advantage for grid intensity.

## Limitations

- This is one temporal realization, not the five-realization confidence-
  interval experiment used for the paper conclusions.
- The `kappa=0.02` Expert Bank lacks a completed `factor_b2_v3` run. Its mean
  is descriptive over eight maps and is not ranked with complete methods.
- Maps are averaged scenario conditions, not independent statistical
  replicates. No confidence interval is claimed here.
- `model_transfers` counts transferred predictor states. It is not a complete
  byte-cost comparison because advertisement formats differ, and the
  centralized reference is not represented by peer model transfers.

## Evidence and code

The repository keeps the raw files once rather than copying roughly 80 MiB
into this archive directory:

- sample-count, learned Expert-Bank, local, and central runs:
  `../../../../paper_experiments/results/paper/luxembourg_cell_grid_factorial_sweeps_v1/`;
- deterministic intensity runs:
  `../../../../paper_experiments/results/paper/luxembourg_cell_grid_intensity_budget_sweep_v1/`;
- captured configurations and original submission manifests:
  `../../../../paper_experiments/experiment_configs/paper/luxembourg_cell_grid_factorial_sweeps_v1/`;
- method implementation:
  `../../../../paper_experiments/code/final/experiments/place_wallis_benchmark/cell_grid_methods.py`;
- shared simulation and learned-acquisition implementation:
  `../../../../paper_experiments/code/shared_runtime/zramp_runtime/support_expert_bank.py`;
- retained acquisition bundle:
  `../../../../paper_experiments/trained_models/paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/bundle.pt`.

Every completed raw run contains `config.json`, `metrics.json`,
`final_fidelity.json`, fidelity histories, communication assumptions, and
sharing records. The original manifests and per-run metadata are retained,
but the one-off array wrapper that launched the sample-count and learned
penalty sweeps was not recovered as a standalone file.

The reader-facing tables are:

- `../../../figures/data/exploratory_experiment_summaries/deterministic_single_model_aggregate.csv`;
- `../../../figures/data/exploratory_experiment_summaries/deterministic_single_model_per_map.csv`.

Regenerate and validate both tables from the repository root with:

```bash
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```
