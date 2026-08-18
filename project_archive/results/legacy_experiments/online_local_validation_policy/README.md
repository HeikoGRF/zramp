# Online local-validation policy experiments

## Question and design

These experiments tested whether each vehicle could learn which peer model to
request from experience gathered during the simulation. The learned policy used
private local validation reservoirs: streamed observations were split into
training, optimization-validation, and reward-validation subsets, and received
models were scored by their improvement on the receiver's held-out data.

The runs used a frozen common encoder and trained private gain heads online.
Their captured assumptions record `pretrained_policy_loaded: false`; this is
therefore distinct from the later cross-map checkpoint studies.

The archive retains four related source campaigns:

- `single_zone_urban_150_communities_policy_pilot_v1`;
- `single_zone_urban_150_mergeable_information_policy_n40_validation_v1`;
- `single_zone_urban_150_mergeable_information_policy_n40_ranked_v2`; and
- `single_zone_urban_150_mergeable_information_policy_sweep_v1`.

Only completed runs are present: 46 runs forming 23 two-seed groups. Six
time-limited runs from the broad sweep were omitted instead of being presented
as completed evidence.

## Result

Across the nine groups with a paired learned-policy and uninformed-comparator
result, learned selection had lower final RMSE in four groups and higher final
RMSE in five. The unweighted mean of the nine paired group differences was
+0.035 dB, where a positive difference means the learned policy was worse.

The individual effects varied substantially by node/contact setting, and each
group contains only two seeds. These results show that the online learner was
operational, but they do not establish a reliable advantage. This negative
result helped motivate the deterministic support-intensity rule used in the
paper.

The `novelty` pilot condition is retained as an additional development
comparator. It is not counted among the nine learned-policy comparisons above.

## Evidence layout

Every run under `runs/` retains the same compact set:

- `config.json` and `communication_overhead_assumptions.json`;
- `progress.json`, proving that the archived run completed;
- `fidelity.csv` and `final_fidelity.json`;
- `cross_validation_pulls.csv`, recording local validation outcomes;
- `exact_policy_training.csv`, `local_policy_training.csv`,
  `local_policy_summary.json`, and `learning_summary.json`; and
- `sharing_events.csv`, providing the realized communication history.

The generated group-level table is
`figures/data/exploratory_experiment_summaries/online_local_validation_policy_summary.csv`.
Regenerate it from the archive root with:

```bash
python3 code/legacy/analysis/build_exploratory_experiment_summaries.py
```

In the summary table, `paired_mean_delta_vs_uninformed_db` is computed
seed-by-seed before averaging; negative values favor the learned policy.

## Source code

The implementation is retained in
`code/legacy/campaign_implementations/online_policy_learning/online_local_validation_policy.py`, with the
CLI in `run_online_local_validation_policy.py`. The private validation reservoirs and reward
logic are in `local_validation_reward.py`; the corrected frozen-encoder/sample-sharing
variant is in `online_policy_variants.py`.

Two recovered reusable launchers document the commands used across this line
of work: `submit_online_local_validation_policy_pilot.sh` for the early pilot and
`submit_online_policy_paired_evaluation.sh` for the later mergeable-
information pairs. Campaign-specific overrides are recoverable from the
retained run configs and communication-assumption files.

## Deliberate omissions

The original campaigns also produced multi-megabyte per-decision diagnostics,
duplicate `*_partial` files, and six time-limited runs. Those files are not
needed to audit the conclusion and are outside the reader-facing archive.

These online policies existed only in simulator memory and were not serialized
as model checkpoints. Their training histories and evaluation records are the
available artifacts; no missing `.pt` file is required to interpret this
campaign.
