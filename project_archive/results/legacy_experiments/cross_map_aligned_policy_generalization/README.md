# Cross-map aligned-policy generalization

## Question and design

This follow-up tested whether a target more closely aligned with deployment and
larger encoder/head architectures could repair the cross-map generalization
problem. Six architectures were trained with two seeds each. The archive
contains capacity audits, adaptation tests, two holdout generations, training
reports, and the checkpoints.

## Result

On the three paired `sequential_unseen_holdouts_v2` maps, the learned policy's
final RMSEs were 31.595, 25.698, and 32.920 dB. The paired uninformed baseline
produced 31.441, 26.451, and 31.962 dB. Their descriptive across-map means were
30.071 and 29.952 dB, respectively; lower is better.

The policy won on one map and lost on two. With only three heterogeneous maps
and no consistent advantage, the result did not support including the learned
policy in the final paper comparison.

## Evidence and reproduction

The per-map and across-map values are in
`figures/data/exploratory_experiment_summaries/cross_map_policy_holdout_summary.csv`
and can be regenerated with
`code/legacy/analysis/build_exploratory_experiment_summaries.py`. Start detailed
review with `analysis/aligned_encoder_capacity_audit.json`, then inspect
`sequential_unseen_holdouts_v2/` and the training reports.

All architecture checkpoints remain in this raw result tree. One representative
candidate is also indexed under
`trained_models/experimental/cross_map_aligned_policy_generalization/`.
Training and audit source is retained under
`code/legacy/campaign_implementations/cross_map_policy_generalization/`, including
`train_exact_deployment_source_policy.py`,
`train_cross_map_encoder_audit.py`, and the four audit scripts named in
`code/legacy/CAMPAIGN_CODE_MAP.md`. The underlying exact simulation is under
`code/legacy/campaign_implementations/online_policy_learning/`; common runtime
code is in `code/shared_runtime/`.
