# Cross-map online-policy generalization

## Question and design

This campaign tested whether a peer-model selection policy trained from
source-map online interaction would retain its apparent advantage on unseen
synthetic map families. It includes six training runs, candidate-selection
diagnostics, source-map closed-loop sweeps, oracle audits, threshold studies,
and sequential evaluation on three untouched map families.

`selection_report.json` records strong in-domain ranking metrics for several
candidates. For example, selected candidates closed a large fraction of the
source-domain oracle gap. The crucial question was whether this translated to
the sequential unseen-map simulation.

## Result

The selected policy's final RMSEs on the three archived unseen runs were
31.236, 26.193, and 32.154 dB, for a descriptive across-map mean of 29.861 dB.
The campaign did not retain a paired run-level baseline for these first
holdouts, so that mean alone cannot establish an advantage. The later aligned
campaign supplied the paired comparison.

The gap between promising source-domain selection diagnostics and insufficient
unseen-map evidence is the reason this line was not used in the reported result.

## Evidence and reproduction

Start with `selection_report.json`, then `sequential_unseen/` and the analysis
JSON files. The compact values are in
`figures/data/exploratory_experiment_summaries/cross_map_policy_holdout_summary.csv`.
All per-training-run checkpoints and reports remain in this result tree; the
chosen checkpoint is also indexed under
`trained_models/experimental/cross_map_online_policy_generalization/`.

The exact sequential implementation is in
`code/legacy/campaign_implementations/online_policy_learning/`; recovered
source-only trainers, policy selection, oracle variants, and holdout drivers are in
`code/legacy/campaign_implementations/cross_map_policy_generalization/`. See
`code/legacy/CAMPAIGN_CODE_MAP.md` for the stage-by-stage mapping. Common
predictor and replay modules remain in `code/shared_runtime/`. Captured configs
are under `experiment_configs/experimental/cross_map_online_policy_generalization/`.
