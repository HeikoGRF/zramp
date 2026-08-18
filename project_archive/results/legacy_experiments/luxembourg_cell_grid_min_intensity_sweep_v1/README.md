# Minimum-intensity sensitivity sweep

This campaign tested how the cell-grid evidence threshold changes the sharing
result. It is a sensitivity study and is not consumed by the paper
aggregation.

The exact one-off scheduler command was not recovered. The archive retains the
cell-grid implementation, captured run configurations, final metrics, and
communication records, so the settings and conclusion remain reviewable and
the runs can be reconstructed without guessing parameters.

From the repository root, first source
`project_archive/scripts/activate_legacy_paths.sh`. Use the runner and
cell-grid implementation under
`project_archive/code/legacy/campaign_implementations/experiments/place_wallis_benchmark/`,
take arguments from this campaign's `config.json` files, and map trace, test
set, and map paths to `paper_experiments/input_data/`. Write reconstructed
outputs outside the repository.

