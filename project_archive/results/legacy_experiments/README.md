# Legacy experimental and negative results

This directory records substantial work that informed the project but is not
part of the final paper comparison. The aim is to show the development path
without asking a reader to infer conclusions from unlabelled run directories.

## Recommended reading order

1. Read `EXPERIMENT_CATALOG.md` at the archive root for the overall chronology.
2. Read the campaign README where one is present.
3. Use the compact tables in
   `figures/data/exploratory_experiment_summaries/` for cross-run context.
4. Inspect individual configs, final metrics, and fidelity trajectories only
   when checking a particular result.

## Campaign map

| Campaign | Question | Main conclusion | Primary evidence |
|---|---|---|---|
| `place_wallis_benchmark` | Which spatial-support and expert-bank ideas are promising on one real-city map? | Support-aware banks looked promising, but this was not a robust multi-map result. | `README.md`, `results.csv` |
| `support_acquisition_pretraining` | Can a learned representation identify useful peer support/models to request? | Training worked sufficiently for follow-up tests, but learned selection scoring was not used by the final deterministic ranking rule. | training summaries, checkpoints |
| `luxembourg_factorial_3x3_benchmark` | Does the learned Expert Bank transfer across nine factorial maps? | Useful development evidence, but it was not selected for the final comparison. | metrics and submitted-job tables |
| `deterministic_single_model_vs_learned_expert_bank` | Do simple deterministic one-model rules outperform the pretrained learned-acquisition Expert Bank? | The deterministic methods were substantially more accurate; intensity Top-1 had only a small descriptive advantage over sample-count Top-1 in this one-realization study. | `README.md`, aggregate CSV, per-map CSV, and indexed raw runs |
| `luxembourg_cell_grid_min_intensity_sweep_v1` | How sensitive is cell-grid support to the minimum evidence threshold? | Sensitivity study only; not consumed by the paper aggregation. | run metrics and configs |
| `luxembourg_cell_grid_synchronized_matched_b2v3_v1` | Are synchronized results stable on two matched zones? | Repeatability check only; the final design used all nine factorial maps and five temporal replicates. | run metrics and configs |
| `luxembourg_nested_size_benchmark_v1` | How does the approach behave on nested 100 m and 200 m crops? | Map size was confounded with realized vehicle density, so this was retained as a stress test rather than a controlled paper result. | `README.md`, run metrics |
| `luxembourg_density_benchmark_v1` | How do augmented vehicle densities affect behavior? | Auxiliary robustness evidence, not part of the factorial paper estimator. | `README.md`, run metrics |
| `luxembourg_multimap_benchmark` | How do earlier methods behave on additional city maps? | Broadened development coverage but did not match the final controlled design. | run metrics and configs |
| `munich_early_benchmarks` | Can the full local/sharing/central pipeline work end to end? | Demonstrated feasibility, but predates the Luxembourg replay protocol. | raw logs and existing summaries |
| `contact_timing_sweep` | Are apparent sharing gains stable across contact and update timing? | Results were condition-dependent; no learned sharing variant was consistently best. | `README.md`, compact CSV, 180 final records |
| `controlled_four_zone_decision_ablation` | Which early sign, budget, and reward choices help in a controlled setup? | The unsigned top-budget-5 variant had the lowest mean final RMSE in this 39-run ablation, but the setup was not the final estimator. | `README.md`, compact CSV |
| `online_local_validation_policy` | Can vehicles learn useful peer-selection policies online from private held-out validation improvements? | The learner was operational but was better in four and worse in five of nine paired groups; no reliable advantage was established. | `README.md`, compact CSV, 46 completed runs |
| `kirchberg_online_private_validation_policy` | Can the online private-validation policy run end to end on a real Kirchberg LuST trace? | Yes as an implementation study: all three learned-policy runs completed, with mean final RMSE 21.441 dB; no within-campaign comparison is included. | `README.md`, compact CSV, three completed runs |
| `receiver_in_zone_global_sender_policy_sweep` | Does the policy help when held-out receivers are in one Gare-Bonnevoie zone but transmitters span the map? | Mixed two-seed results depended on K/S; 30 of 32 attempts completed and no robust advantage was established. | `README.md`, protocol JSON, compact CSV |
| `cross_map_online_policy_generalization` | Does a policy trained on source maps retain its advantage on unseen maps? | In-domain selection diagnostics were promising, but unseen-map behavior was not enough for the paper. | `README.md`, selection report, holdouts |
| `cross_map_aligned_policy_generalization` | Do aligned targets and larger encoders solve cross-map transfer? | On the three paired v2 holdouts, policy mean RMSE was 30.071 dB versus 29.952 dB for the uninformed baseline; no reliable advantage was established. | `README.md`, compact CSV, audit JSON |

## Scope of the raw records

The archive retains final metrics, trajectories, communication records,
training summaries, configs, completion state, and checkpoints needed to
review these conclusions. Duplicate partial outputs, scheduler logs, profiling
files, and multi-gigabyte per-decision debug tables were excluded from the
submission view. Their removal does not change any compact table above.

Exact configuration copies are indexed in `experiment_configs/experimental/`.
The implementation and launchers are documented in `code/legacy/README.md` and
mapped campaign by campaign in `code/legacy/CAMPAIGN_CODE_MAP.md`.
