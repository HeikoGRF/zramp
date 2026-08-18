# Kirchberg online private-validation policy

## Purpose and status

This is a **legacy development experiment**, not a paper result. It records
whether a peer-selection policy could be trained and executed online from local
held-out validation gains on a real Luxembourg mobility and radio trace.

The retained campaign uses the northern Kirchberg LuST scene, one evaluated
zone, three seeds, and 1,799 simulation steps. Vehicles maintain floating
predictors, evaluate bilateral merges on private validation records, and train
a local policy during the simulation to rank feasible peers.

Only the three learned-policy runs are included here. No comparative claim is
made from this campaign. The separate archived policy studies provide the
reviewable evidence for why learned peer selection was not adopted as the final
paper method.

## Result and interpretation

The three final RMSE values are 21.172, 20.467, and 22.685 dB; their descriptive
mean is **21.441 dB** with a sample standard deviation of **1.134 dB**. These
values demonstrate that the full online-learning pipeline ran to completion on
a real LuST/Sionna scene. Without an included within-campaign comparator, they
do not establish that the policy is better than another selection rule.

The policy learned a nonzero relationship between predicted and observed
validation gain. That diagnostic is retained as implementation evidence, not
as proof of improved final radio-map accuracy.

## Evidence retained

- `run_inventory.csv`: completion state and final RMSE for all three runs.
- `runs/seed_*/learned_policy/config.json`: captured simulator settings.
- `final_fidelity.json` and `fidelity_history.csv`: final and per-evaluation
  reconstruction metrics.
- `learning_summary.json`, `local_policy_training.csv`, and
  `policy_training_by_step.csv`: online training diagnostics.
- `validation_outcomes_by_step.csv`: compact per-step validation and adoption
  outcomes.
- `communication_history.csv` and `communication_overhead_assumptions.json`:
  communication totals and accounting assumptions.
- `../../../figures/data/exploratory_experiment_summaries/kirchberg_online_private_validation_policy_summary.csv`:
  the reviewer-facing descriptive aggregation.

There is no separate trained-model checkpoint: the policies were trained and
used online inside each run. The histories above are the model evidence.

## Compacting and reproduction

Multi-gigabyte per-decision diagnostics and scheduler logs are not duplicated
in this submission. The compact per-step histories can be recreated from the
original or regenerated external campaign output with:

```bash
python3 project_archive/code/legacy/analysis/prepare_real_map_policy_campaigns.py \
  --data-root /path/to/luxembourg_real_city
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```

The implementation is under
`../../../code/legacy/campaign_implementations/online_policy_learning/`.
Luxembourg trace generation is under
`../../../code/legacy/campaign_implementations/SUMO/luxembourg_real_city/`.
The original one-off cluster submission command was not preserved as a
standalone file; captured run configs and the reusable Python entry point are
retained. New outputs should be written outside this repository.
