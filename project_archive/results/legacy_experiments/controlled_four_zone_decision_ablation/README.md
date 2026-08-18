# Controlled four-zone decision ablation

## Question and design

This 39-run experiment isolated early decision-rule choices in one controlled
four-zone setup. Thirteen variants were each run with seeds 1, 2, and 3 for
1,000 steps. The directory names encode the compared rule family and budget:
current behavior, older beta-weight choices, signed top-budget selection, and
unsigned top-budget selection with budgets 1, 2, or 5.

## Result

The lowest three-seed mean final RMSE was 16.914 dB for
`unsigned-top-b_b5`. The next-lowest mean was 17.955 dB for `old_beta_1`.
This supported continued investigation of unsigned/top-budget selection, but
the experiment used a small controlled setup and a different evaluation design
from the final nine-map, five-replicate study.

It is therefore development evidence, not a paper comparison.

## Evidence and reproduction

`figures/data/exploratory_experiment_summaries/four_zone_decision_summary.csv` contains all 13
three-seed means and sample standard deviations. Regenerate it with
`code/legacy/analysis/build_exploratory_experiment_summaries.py`.

Each run retains `config.json`, `final_fidelity.json`, `fidelity.csv`,
`sharing_events.csv`, policy-training summaries, and completion state. The
campaign-specific implementation is in
`code/legacy/campaign_implementations/online_policy_learning/decision_ablation.py` and
`run_decision_ablation.py`; the map builder is
`code/legacy/campaign_implementations/SUMO/build_controlled_4zone_300.py`. Archived
configuration copies are under
`experiment_configs/experimental/controlled_four_zone_decision_ablation/`.
The original array wrapper was not recovered, so the archive retains the exact
driver and per-run settings without presenting a reconstructed wrapper as an
original artifact.
