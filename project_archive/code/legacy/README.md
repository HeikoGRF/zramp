# Legacy and experimental source

None of this code produced the paper's authoritative five-replicate result.
It is retained only where it explains a documented development or negative
experiment in `results/legacy_experiments/`.

Use `CAMPAIGN_CODE_MAP.md` as the reader-facing index from every retained
legacy result campaign to its implementation, launcher, and analysis code.

## Campaign mapping

- `early_munich_experiments/` contains the early Munich dataset, benchmark,
  zone, and full-fine-tuning code. It maps to
  `results/legacy_experiments/munich_early_benchmarks/`.
- `campaign_implementations/experiments/place_wallis_benchmark/` contains the
  support-shape, Expert Bank, dominance, and learned-acquisition variants used
  by the Place Wallis and early Luxembourg campaigns.
- `campaign_implementations/experiments/support_acquisition_pretraining/`
  contains the learned support-selection variants whose outputs are under
  `results/legacy_experiments/support_acquisition_pretraining/`.
- `campaign_implementations/SUMO/luxembourg_real_city/` contains launchers and
  map preparation for the factorial Expert Bank, matched-zone, nested-size,
  density, additional-map, and earlier synchronized sweeps.
- `experiments/capsule_probe/`, `experiments/capsule_greedy/`, and
  `experiments/support_acquisition_pretraining/` are small predecessor modules
  used by those documented method-development stages.
- `analysis/prepare_real_map_policy_campaigns.py` creates compact, reviewable
  records from the preserved Kirchberg and Gare-Bonnevoie campaign roots while
  omitting their multi-gigabyte per-decision diagnostics.
- `analysis/build_exploratory_experiment_summaries.py` regenerates the compact
  deterministic-versus-learned, contact, four-zone, online local-validation
  policy, real-Luxembourg policy, and cross-map tables in
  `figures/data/exploratory_experiment_summaries/`.
- `campaign_implementations/online_policy_learning/` contains the recovered
  online local-validation learner, policy variants, contact-timing launcher, and
  controlled four-zone ablation driver used by the archived campaigns, including
  the Kirchberg and Gare-Bonnevoie real-map studies.
- `campaign_implementations/cross_map_policy_generalization/` contains the
  cross-map policy trainers, source-only audits, oracle variants, and
  sequential holdout drivers corresponding to the two retained result trees.

The large simulator implementation shared by several historical campaigns is
under `../../../paper_experiments/code/shared_runtime/`. The legacy Place
Wallis runner is a compatibility adapter to that maintained copy, avoiding two
diverging implementations.

Legacy source is provenance and experiment context, not the supported paper
reproduction path. Use `../../../paper_experiments/code/final/` for the latter.
Unsupported scratch snippets without an interpretable archived result are not
included.

The newly recovered campaign files come from the preserved 2026-07-30 source
snapshot. Two later exact-sequential files are taken from the final working
tree because they contain the same campaign behavior plus compatibility fixes.
Repository-only package initializers add
`paper_experiments/code/shared_runtime` to `sys.path`; that is the only
structural adaptation needed for the curated layout.
