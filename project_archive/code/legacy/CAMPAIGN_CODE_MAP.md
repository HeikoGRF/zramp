# Legacy campaign-to-code map

This file answers one question: which archived source files explain each
legacy result that is shown? Paths are relative to `code/legacy/` unless noted
otherwise. The final paper experiment is intentionally outside this map and
lives in `../../../paper_experiments/code/final/`.

## Early real-city and support experiments

| Result campaign | Primary implementation and launcher | Status of launch record |
|---|---|---|
| `munich_early_benchmarks` | `early_munich_experiments/` | Original benchmark drivers and analysis are retained. |
| `place_wallis_benchmark` | `campaign_implementations/experiments/place_wallis_benchmark/`; input generation in `campaign_implementations/SUMO/luxembourg_real_city/` | Original method launchers are retained. |
| `support_acquisition_pretraining` | `campaign_implementations/experiments/support_acquisition_pretraining/` | Original model definitions and training launchers are retained. |
| `luxembourg_factorial_3x3_benchmark` | `campaign_implementations/experiments/place_wallis_benchmark/run_support_expert_bank.py`; `campaign_implementations/SUMO/luxembourg_real_city/submit_factorial_zone_pipeline.sh`, `submit_factorial_pq_sweep.sh`, and `submit_factorial_support_baselines.sh` | Original launchers and submission manifests are retained. |
| `deterministic_single_model_vs_learned_expert_bank` | Maintained method code in `../../../paper_experiments/code/final/experiments/place_wallis_benchmark/cell_grid_methods.py` and shared runtime in `../../../paper_experiments/code/shared_runtime/zramp_runtime/support_expert_bank.py`; summaries from `analysis/build_exploratory_experiment_summaries.py` | Raw runs, per-run configs, and original submission manifests are retained once in the two replicate-1 paper result roots. The one-off sample-count/penalty array wrapper was not recovered as a standalone file. |
| `luxembourg_cell_grid_min_intensity_sweep_v1` | Cell-grid implementation in `campaign_implementations/experiments/place_wallis_benchmark/cell_grid_methods.py` and `cell_grid_support.py`; captured run configs remain with the results | The exact one-off scheduler command was not recovered; implementation and run configs are retained. |
| `luxembourg_cell_grid_synchronized_matched_b2v3_v1` | `campaign_implementations/SUMO/luxembourg_real_city/submit_matched_b2v3_synchronized_sweep.sh` and `submit_matched_b2v3_entry_anchored_sweep.sh` | Original launchers are retained. |
| `luxembourg_nested_size_benchmark_v1` | `campaign_implementations/SUMO/luxembourg_real_city/submit_nested_size_pipeline.sh` | Original launcher and submission manifests are retained. |
| `luxembourg_density_benchmark_v1` | `campaign_implementations/SUMO/luxembourg_real_city/augment_mobility_density.py` and `submit_density_sweep.sh` | Original generator, launcher, and manifests are retained. |
| `luxembourg_multimap_benchmark` | Evaluation-zone generation and launch code under `campaign_implementations/SUMO/luxembourg_real_city/` | The reusable original pipeline is retained; run-specific settings are in the result configs. |

The large Expert-Bank implementation used by several rows above is held once
in `../../../paper_experiments/code/shared_runtime/zramp_runtime/support_expert_bank.py`; the historical
entry point is a compatibility adapter.

## Online-policy and controlled-ablation experiments

| Result campaign | Primary implementation | Launcher or analysis |
|---|---|---|
| `contact_timing_sweep` | `campaign_implementations/online_policy_learning/online_local_validation_policy.py`, `online_policy_variants.py`, and `run_online_policy_variants.py` | `submit_contact_timing_and_policy_variant_sweep.sh`; compact table from `analysis/build_exploratory_experiment_summaries.py` |
| `controlled_four_zone_decision_ablation` | `campaign_implementations/online_policy_learning/decision_ablation.py`, `utility_selection.py`, `run_decision_ablation.py`, plus `campaign_implementations/SUMO/build_controlled_4zone_300.py` | The driver accepts every archived ablation variant. The original array wrapper was not recovered; exact settings are preserved in each run's `config.json`. |
| `online_local_validation_policy` | `campaign_implementations/online_policy_learning/online_local_validation_policy.py`, `run_online_local_validation_policy.py`, `local_validation_reward.py`, `online_policy_variants.py` | `submit_online_local_validation_policy_pilot.sh` and `submit_online_policy_paired_evaluation.sh` are the recovered reusable launchers; group table from `analysis/build_exploratory_experiment_summaries.py`. |
| `kirchberg_online_private_validation_policy` | `campaign_implementations/online_policy_learning/online_local_validation_policy.py`, `run_online_local_validation_policy.py`, and `local_validation_reward.py` | The one-off array wrapper was not recovered; captured configs and compact learned-policy records are retained. `analysis/prepare_real_map_policy_campaigns.py` extracts them and the common summary builder aggregates them. |
| `receiver_in_zone_global_sender_policy_sweep` | The same online-policy implementation; trace construction in `campaign_implementations/SUMO/luxembourg_real_city/generate_pilot_rssi_trace.py` with `--fidelity-global-senders` | The one-off array wrapper was not recovered. The protocol JSON and captured metadata preserve K, S, method, seed, and the asymmetric evaluation geometry; the two analysis scripts compact and aggregate the records. |

The online policies were trained inside the simulator. Their retained training
histories are therefore the model evidence; there is no separate checkpoint
for this campaign.

## Cross-map learned-policy experiments

Shared source-map construction is in:

- `campaign_implementations/SUMO/build_policy_pretraining_maps.py`;
- `campaign_implementations/SUMO/build_policy_pretraining_routes.py`;
- `campaign_implementations/online_policy_learning/build_cross_map_policy_dataset.py`;
- `campaign_implementations/online_policy_learning/train_cross_map_policy.py`; and
- `campaign_implementations/online_policy_learning/submit_cross_map_*.sh` plus
  `submit_exact_deployment_source_traces.sh`.

The corresponding files under
`campaign_implementations/cross_map_policy_generalization/` are grouped below.

| Stage represented in results | Archived source |
|---|---|
| Exact source traces and sequential holdouts | `prepare_policy_source_exact_trace.py`, `role_exact_simulation.py`, `run_role_exact_sequential.py` |
| Parameter-geometry and oracle comparisons | `parameter_geometry_role_simulation.py`, `parameter_novelty_oracle_simulation.py`, `rmse_alpha_oracle_simulation.py`, `rmse_gain_oracle_simulation.py`, and their `run_*.py` entry points |
| Maturity and relative-maturity variants | `source_maturity_reward_simulation.py`, `source_relative_maturity_oracle.py`, their entry points, `train_maturity_source_policy.py`, and `train_relative_maturity_source_policy.py` |
| On-policy candidate training and selection | `train_onpolicy_augmented_source_policy.py` and `select_onpolicy_augmented_policy.py` |
| Aligned reward and encoder training | `train_exact_deployment_source_policy.py`, `train_cross_map_encoder_audit.py`, and `make_encoder_only_policy_checkpoint.py` |
| Retained aligned-policy audits | `evaluate_additional_source_holdouts.py`, `exact_layer_geometry_reward_audit.py`, `frozen_encoder_shared_reward_audit.py`, and `shared_exact_reward_adaptation_audit.py` |

These files map to `cross_map_online_policy_generalization` and
`cross_map_aligned_policy_generalization`. The original ad-hoc scheduler commands for
some cross-map audit stages were not preserved as standalone shell files; the
Python entry points, checkpoints, reports, role configs, and completed outputs
are retained. This distinction is explicit so the archive does not present a
reconstructed command as an original record.

## Shared code and generated summaries

- `../../../paper_experiments/code/shared_runtime/` contains the predictor,
  SUMO replay, reward, and radio modules imported by both old and final
  experiments.
- `analysis/build_exploratory_experiment_summaries.py` regenerates every compact
  legacy table stored under `figures/data/exploratory_experiment_summaries/`.
- Result-specific `README.md` files state the conclusion and limitations before
  the reader reaches raw run directories.
