# Shared runtime

This directory is a required dependency of the paper-facing code in
`../final/`. It is kept separate because its modules are inherited from a
larger experimental codebase and contain generalized branches that are not
themselves reported paper methods.

The final entry points activate this directory through
`../final/archive_paths.py`. No file here should be launched as the primary
paper reproduction command.

Contents:

- `SUMO/`, `rl_reward_experiment/`, and the top-level model/map modules:
  simulation and prepared-trace replay engine.
- `zramp_runtime/support_expert_bank.py`: generalized implementation core used
  by the paper runner and the reported learned-acquisition Expert Bank.
- `experiments/place_wallis_benchmark/`: support-geometry compatibility base
  required by the generalized core.
- `experiments/support_acquisition_pretraining/`: helper models and data
  generation functions required to load or retrain the archived patch bundles.
