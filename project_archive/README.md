# Project archive

This directory preserves the substantial development, sensitivity, and
negative experiments that informed the final study. The final paper implementation
and its raw evidence are intentionally not duplicated here; they are the
authoritative material under `../paper_experiments/`.

## Reading order

1. Read [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md) for the scientific
   status and conclusion of every retained campaign.
2. Read
   [RERUNNING_LEGACY_EXPERIMENTS.md](RERUNNING_LEGACY_EXPERIMENTS.md) for
   environment, input, path, and launcher guidance.
3. Use the README inside a result campaign before inspecting individual run
   directories.
4. Use `code/legacy/CAMPAIGN_CODE_MAP.md` to map evidence back to source.

## Structure

- `code/legacy/`: source and launchers tied to documented historical results.
- `experiment_configs/experimental/`: captured configurations and submission
  manifests.
- `trained_models/experimental/`: selected learned-acquisition and cross-map
  policy checkpoints.
- `results/legacy_experiments/`: curated raw evidence and campaign summaries.
- `figures/`: compact CSV summaries of documented exploratory experiments.
- `../MANUSCRIPT.zip`: complete Overleaf work tree.

Historical source reuses
`../paper_experiments/code/shared_runtime/`. Compact maps and test sets are
also shared from `../paper_experiments/input_data/`. This keeps one maintained
copy of common code and data while the repository as a whole remains
self-contained.

## Rebuild compact legacy tables

The two compact real-map policy result trees were prepared from the preserved
source campaign roots by
`code/legacy/analysis/prepare_real_map_policy_campaigns.py`. The submitted
compact records are already present; rerunning that extractor requires the
regenerated or original external campaign outputs.


From the repository root:

```bash
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```

This validates the retained deterministic-versus-learned, contact-timing,
four-zone, synthetic-map and real-Luxembourg online-policy, and cross-map records and regenerates
`figures/data/exploratory_experiment_summaries/`.

## Manuscript source

The complete Overleaf project tree is stored once in `../MANUSCRIPT.zip`,
including all LaTeX, bibliography, style, image, and generated-table inputs.
