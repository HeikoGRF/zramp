# Online policy learning and decision experiments

This package implements the exploratory methods in which each vehicle learned
which peer model to request while the simulation was running. These methods
were investigated but were not used in the final paper experiment.

## Online local-validation policy

The main method is implemented by `online_local_validation_policy.py` and
launched through `run_online_local_validation_policy.py`. Each vehicle
maintained private held-out local observations, evaluated received models by
their local validation improvement, and trained a private gain-prediction head
online. `local_validation_reward.py` contains the validation split and reward
calculation.

`online_policy_variants.py` and `run_online_policy_variants.py` contain the
later frozen-encoder and sample-sharing variants. Their completed evidence is
under
[`online_local_validation_policy`](../../../../results/legacy_experiments/online_local_validation_policy/).

## Other studies in this package

- `decision_ablation.py` and `run_decision_ablation.py` test early decision,
  sign, and pull-budget choices.
- `submit_contact_timing_and_policy_variant_sweep.sh` defines the retained
  contact-timing campaign.
- `build_cross_map_policy_dataset.py` and `train_cross_map_policy.py` are early
  source-map dataset and policy-training stages. The later cross-map work is
  organized in `../cross_map_policy_generalization/`.

See [CAMPAIGN_CODE_MAP.md](../../CAMPAIGN_CODE_MAP.md) for the complete mapping
from source files to archived campaigns.
