# Historical figure data

This directory intentionally contains only compact CSV summaries, not rendered
plots. `data/exploratory_experiment_summaries/` records the documented
exploratory campaigns.

Regenerate the validated summary tables from the repository root with:

```bash
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```

Paper figure data is kept once under `../../paper_experiments/figures/`. The
authoritative LaTeX/TikZ plotting code is provided in `../../MANUSCRIPT.zip`.
