# Factorial Expert-Bank benchmark

This development campaign extended the learned support-acquisition and
Expert-Bank ideas to the nine Luxembourg factorial maps. It provided useful
multi-map evidence, but the learned bank was not selected for the final paper
comparison.

The primary retained launchers are:

- `project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_factorial_zone_pipeline.sh`
- `project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_factorial_pq_sweep.sh`
- `project_archive/code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_factorial_support_baselines.sh`

The runner is under
`project_archive/code/legacy/campaign_implementations/experiments/place_wallis_benchmark/`.
It reuses `paper_experiments/code/shared_runtime/`; the experimental bundle is
under `project_archive/trained_models/experimental/acquisition_pretraining/`.
Use the paper input-generation workflow for Luxembourg traces, then replace
the launchers' historical cluster paths and write outputs outside the
repository.

