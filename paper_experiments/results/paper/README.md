# Authoritative paper results

Every final-paper run consumed by
`code/final/analysis/aggregate_five_replicates.py` is retained under these
roots. The replicate-1 factorial root also preserves same-protocol
sample-count and learned-acquisition development runs; these are excluded from
the paper aggregation and documented under
`../../../project_archive/results/legacy_experiments/deterministic_single_model_vs_learned_expert_bank/`.

- `luxembourg_cell_grid_factorial_sweeps_v1/`: replicate-1 ISO and Central,
  plus the explicitly documented development comparison above.
- `luxembourg_cell_grid_intensity_budget_sweep_v1/`: replicate-1 Full, Top-5,
  and Top-1.
- `luxembourg_cell_grid_synchronized_9map_paper_final_v1/`: replicate-1
  Every-5 through Every-80.
- `luxembourg_cell_grid_ci_temporal5_paper_final_v1/`: replicates 2-5 and all
  ungated-greedy replicates.

The aggregator reads `metrics.json` and `sharing_events.csv` from the exact run
paths listed in `code/final/FINAL_RESULTS_DEPENDENCIES.md`. Rebuild all
reported CSV tables from the `paper_experiments/` directory with:

```bash
bash scripts/rebuild_paper_outputs.sh
```

The regenerated outputs are under `figures/data/`. Operational scheduler logs
and superseded partial copies are not part of this reader-facing result tier.
