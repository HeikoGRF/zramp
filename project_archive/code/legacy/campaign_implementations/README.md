# Documented legacy campaign implementations

This directory contains the substantial experimental implementations that are
connected to retained results. None of them is the supported reproduction path
for the final paper experiment.

- `online_policy_learning/` contains the online local-validation policy,
  contact-timing study, decision ablations, and early cross-map policy tools.
- `cross_map_policy_generalization/` contains the later source-map training,
  policy-selection, audit, and unseen-map evaluation stages.
- `experiments/` contains Place Wallis method development, Expert Bank
  variants, support-shape prototypes, and learned support-selection training.
- `SUMO/` contains Luxembourg map, mobility, radio-trace, and campaign
  launchers used by the historical experiments.

Use [CAMPAIGN_CODE_MAP.md](../CAMPAIGN_CODE_MAP.md) to connect each source
group to its configurations and retained evidence.
