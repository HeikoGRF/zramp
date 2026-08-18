# Trained models from exploratory methods

`experimental/` contains selected learned-acquisition and cross-map policy
checkpoints referenced by documented historical campaigns. Per-run checkpoints
that are needed to interpret a campaign remain alongside that campaign's
results.

- `experimental/acquisition_pretraining/` contains the selected learned
  support/model-selection bundle.
- `experimental/cross_map_online_policy_generalization/` contains the selected
  policy from the first source-map training campaign.
- `experimental/cross_map_aligned_policy_generalization/` contains a
  representative aligned-reward/encoder candidate.

The bundle required by the final paper runner is kept once under
`../../paper_experiments/trained_models/paper_runtime/`.
