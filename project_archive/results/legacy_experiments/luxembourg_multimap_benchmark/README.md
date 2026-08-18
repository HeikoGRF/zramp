# Additional Luxembourg maps

This campaign broadened early method evaluation to additional real-city
regions. It did not use the controlled factorial and temporal design of the
final paper, so it is retained as development coverage rather than final
evidence.

Evaluation-zone map, mobility, radio-trace, and test-set preparation scripts
are under
`project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/`.
Run-specific settings are preserved under
`project_archive/experiment_configs/experimental/luxembourg_multimap_benchmark/`
and in each result's `config.json`.

To rerun, source `project_archive/scripts/activate_legacy_paths.sh`, regenerate
LuST inputs with the workflow under `paper_experiments/`, select the relevant
evaluation-zone launcher, replace its historical data and scheduler paths, and
write new outputs outside the repository.

