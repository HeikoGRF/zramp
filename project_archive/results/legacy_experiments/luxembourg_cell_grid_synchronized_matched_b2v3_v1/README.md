# Matched-zone synchronization study

This repeatability check compared two zones with similar high-level map
factors. It motivated the broader nine-map, five-window final design and is not
part of the final estimator.

The reusable launchers are:

- `project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_matched_b2v3_synchronized_sweep.sh`
- `project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_matched_b2v3_entry_anchored_sweep.sh`

Before submitting, source
`project_archive/scripts/activate_legacy_paths.sh`, replace the historical
cluster paths, and map the two zone assets and test sets to
`paper_experiments/input_data/`. Captured run configurations in
`project_archive/experiment_configs/experimental/luxembourg_cell_grid_synchronized_matched_b2v3_v1/`
are the authority for the exact settings.

