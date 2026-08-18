# Paper source code

`final/` contains the paper-facing entry points. `shared_runtime/` contains
the maintained simulator machinery imported by those entry points and by
selected historical experiments. There is no second copy of this runtime under
`project_archive/`.

The recommended interface is
`../scripts/run_paper_experiments.py`. It builds explicit direct commands for
all methods, checks their inputs, supports map/replicate/method filters, and
writes the same four result layouts understood by the final aggregator.

The original Slurm launchers are retained under
`final/SUMO/luxembourg_real_city/` as exact provenance. Their historical
absolute defaults are not portable; set their `REPO_ROOT`, input, output,
Python, and Slurm variables before using them on another system.

The primary simulation entry points are:

- `final/experiments/place_wallis_benchmark/run_support_expert_bank.py` for
  ISO, Central, and the deterministic cell-grid intensity method;
- `final/experiments/place_wallis_benchmark/run_equal_greedy.py` for ungated
  equal averaging;
- `final/analysis/aggregate_paper_results_across_timeframes.py` for the paper
  statistics across the five temporal realizations; and
- `final/analysis/build_plot_ready_paper_tables.py` for validated plot-ready
  tables.

The exact inventory is
`final/FINAL_RESULTS_DEPENDENCIES.md`.
