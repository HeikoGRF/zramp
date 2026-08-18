# Cross-map policy generalization experiments

This package contains the later attempts to train a model-selection policy on
source maps and evaluate it sequentially on unseen synthetic map families.
It includes source-domain policy training, candidate selection, oracle audits,
reward/encoder alignment studies, and unseen-map holdout drivers.

The two reader-facing campaigns are:

- [`cross_map_online_policy_generalization`](../../../../results/legacy_experiments/cross_map_online_policy_generalization/),
  which selected candidates from source-domain online-policy diagnostics; and
- [`cross_map_aligned_policy_generalization`](../../../../results/legacy_experiments/cross_map_aligned_policy_generalization/),
  which tested aligned deployment targets and larger encoders with paired
  unseen-map evaluations.

Start with `train_onpolicy_augmented_source_policy.py` and
`select_onpolicy_augmented_policy.py` for the first campaign. The aligned
follow-up centers on `train_exact_deployment_source_policy.py` and
`train_cross_map_encoder_audit.py`. The `run_*`, `*_simulation.py`, and
`*_audit.py` files provide the retained evaluation and diagnostic stages.

See [CAMPAIGN_CODE_MAP.md](../../CAMPAIGN_CODE_MAP.md) for stage-level details.
