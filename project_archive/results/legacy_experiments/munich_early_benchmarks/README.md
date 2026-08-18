# Munich early benchmarks

## Question and role in the project

These were the first end-to-end radio-map learning experiments. They tested
local training, greedy model sharing, centralized learning, later full model
fine-tuning, and zone-aware variants before the project moved to the controlled
Luxembourg trace-replay design.

The experiments established that the complete simulation and learning pipeline
could run and that model sharing could change reconstruction behavior. They use
different data, metrics, and training assumptions from the final experiment,
so their numerical results must not be compared directly with the final
nine-map results.

## Evidence

- `raw_logs/` contains the original simulation logs, sharing events, summaries,
  and the analysis program used on them.
- `later_comparisons/` contains per-step and aggregate comparison CSVs.
- The matching source, benchmark drivers, and older analysis summaries are in
  `code/legacy/early_munich_experiments/`.

This campaign is included to document the early feasibility phase, not as a
reproducible instance of the later ZRAMP protocol.
