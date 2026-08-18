# Online local-validation policy configurations

This directory mirrors the 46 completed runs under
`results/legacy_experiments/online_local_validation_policy/runs/`.

Each run contributes:

- `config.json`, containing the simulation and learning hyperparameters; and
- `communication_overhead_assumptions.json`, describing the online policy,
  private validation split, reward target, payloads, and communication rules.

The complete training histories, validation records, final metrics, and
completion state remain beside the results rather than being duplicated here.
Original absolute cluster paths are retained as provenance and are not portable
defaults.

See the result campaign README and
`figures/data/exploratory_experiment_summaries/online_local_validation_policy_summary.csv`
for the
design and outcome.
