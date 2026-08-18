# Paper figure data

This directory intentionally contains CSV data rather than rendered plots.
The authoritative LaTeX/TikZ plotting code is supplied in
`../../MANUSCRIPT.zip`.

- `data/statistical_aggregation/` contains all 495 run-level rows, the
  per-timeframe method averages, and the five-timeframe confidence intervals.
- `data/plot_ready_tables/` contains the validated CSV tables arranged for the
  plots in the paper.

Regenerate every CSV file with `../scripts/rebuild_paper_outputs.sh`.
