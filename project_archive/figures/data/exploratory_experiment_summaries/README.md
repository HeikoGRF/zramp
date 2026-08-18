# Compact experimental summaries

These CSV files are reader-facing indexes generated from the retained raw
experimental outputs. Regenerate them from the archive root with:

```bash
python3 code/legacy/analysis/build_exploratory_experiment_summaries.py
```

- `contact_timing_sweep_summary.csv` contains 90 condition/variant groups. Each row
  averages two seeds from the 180-run contact-timing sweep.
- `four_zone_decision_ablation_summary.csv` contains 13 decision variants, each averaged
  over three seeds.
- `cross_map_policy_holdout_summary.csv` records per-map final RMSE and an across-map
  descriptive mean for the two learned-policy campaigns. The map-level rows,
  not the across-map descriptive rows, are the underlying observations.
- `online_local_validation_policy_summary.csv` contains 23 two-seed groups
  from 46 completed online-policy runs.
- `kirchberg_online_private_validation_policy_summary.csv` gives the
  descriptive mean and sample standard deviation for three real-map
  learned-policy seeds.
- `receiver_in_zone_global_sender_policy_sweep_summary.csv` records all K/S
  groups, completed and incomplete attempt counts, and paired differences for
  the asymmetric Gare-Bonnevoie evaluation.
- `deterministic_single_model_aggregate.csv` compares the map-mean RMSE and
  model-transfer counts of deterministic sample-count and grid-intensity
  methods, learned Expert Banks, and local/central references in temporal
  realization 1.
- `deterministic_single_model_per_map.csv` exposes the 108 expected map/method
  cells underlying that aggregate, including the one missing learned
  `kappa=0.02` result.

For every policy table, the paired delta is learned policy minus uninformed
selection, so negative values favor the learned policy.

These are summaries of exploratory experiments and are not inputs to the
paper's five-realization confidence intervals. The deterministic comparison
reuses raw paper-replicate-1 inputs and reference runs, but its sample-count
and learned-acquisition variants are development evidence only.
