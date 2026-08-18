# Experiment catalog

This is the reader's map of the project. It separates evidence used for the
paper conclusions from substantial exploratory work. Directory order is not
intended as a strict date log; it represents the development progression that
can be reconstructed from the retained artifacts.

## Status meanings

- **Paper**: consumed by the authoritative five-replicate aggregation.
- **Development**: a substantial experiment with an interpretable outcome,
  retained to explain how the final design was reached.
- **Negative**: a tested idea that did not establish a robust advantage.
- **Sensitivity**: a useful auxiliary check that was not part of the paper
  estimator.

## Final paper experiment

| Status | Question | Design and outcome | Code | Results |
|---|---|---|---|---|
| Paper | How do deterministic intensity-count cell-grid sharing policies trade reconstruction error against communication? | Nine factorial Luxembourg maps, five temporal replicates, local and central baselines, All/Top-5/Top-1 budgets, five communication intervals, and ungated greedy. Confidence intervals use the five temporal replicates after averaging maps within each replicate. | `../paper_experiments/code/final/`, `../paper_experiments/code/shared_runtime/` | `../paper_experiments/results/paper/`, `../paper_experiments/figures/data/statistical_aggregation/` |

The four exact raw roots and every run-name mapping are listed in
`../paper_experiments/code/final/FINAL_RESULTS_DEPENDENCIES.md`. The final ranking and merge weights
are deterministic and do not use learned acquisition scores. The runner still
loads the cell-grid support bundle as infrastructure; see `trained_models/README.md`.

## Development and negative experiments

| Status | Campaign | What was tested | What was learned | Code and evidence |
|---|---|---|---|---|
| Development | Munich early benchmarks | Local, greedy-sharing, central, full-fine-tuning, and zone-aware end-to-end prototypes. | Established pipeline feasibility but used a protocol not comparable with the final study. | `code/legacy/early_munich_experiments/`; `results/legacy_experiments/munich_early_benchmarks/` |
| Development | Place Wallis benchmark | RBF, capsule, plane, support-bank, dominance, and learned-acquisition variants on one real-city map. | Support-aware banks looked promising on one map; this motivated multi-map validation and a simpler final rule. | `code/legacy/campaign_implementations/experiments/place_wallis_benchmark/`; campaign `README.md` and `results.csv` |
| Development | Learned support-selection pretraining | Autoencoder representations and learned scores for choosing useful peer support/models. | Produced workable bundles for follow-up experiments, but these models are not part of the final deterministic method. | `code/legacy/campaign_implementations/experiments/support_acquisition_pretraining/`; `results/legacy_experiments/support_acquisition_pretraining/` |
| Development | Factorial Expert Bank | Learned acquisition and support baselines on the nine factorial maps. | Extended the one-map idea, but the learned bank was not selected for the final paper comparison. | factorial launchers under `code/legacy/campaign_implementations/SUMO/`; `results/legacy_experiments/luxembourg_factorial_3x3_benchmark/` |
| Development | Deterministic single-model versus learned Expert Bank | On the first temporal realization of all nine factorial maps, compared sample-count and grid-intensity single-model rules, three learned-acquisition penalties, and local/central references. | Deterministic single-model methods substantially outperformed the learned Expert Bank. Intensity Top-1 had the best deterministic mean (14.752 dB), but its 0.135 dB advantage over sample-count Top-1 was small and was not evaluated across all five temporal realizations. | `results/legacy_experiments/deterministic_single_model_vs_learned_expert_bank/`; two generated CSV tables; raw runs retained once under `../paper_experiments/results/paper/` |
| Sensitivity | Minimum-intensity sweep | Cell-grid evidence thresholds. | Quantified threshold sensitivity; not an input to the paper estimator. | shared runtime; `results/legacy_experiments/luxembourg_cell_grid_min_intensity_sweep_v1/` |
| Sensitivity | Matched B2V3 zones | Repeatability on two zones with similar high-level map factors. | Motivated broader factorial and temporal replication rather than reliance on a matched pair. | `submit_matched_b2v3_*` launchers; corresponding result directory |
| Sensitivity | Nested map sizes | 100 m hotspot and 200 m nested crops. | Useful stress test, but map size and realized traffic density were confounded. | `submit_nested_size_pipeline.sh`; campaign `README.md` |
| Sensitivity | Controlled density | Phase-shifted trajectory duplication at 1x, 2x, and 4x density. | Auxiliary robustness evidence; the construction does not model congestion feedback. | `submit_density_sweep.sh`, `augment_mobility_density.py`; campaign `README.md` |
| Development | Additional maps | Earlier-method evaluation on more city regions. | Broadened coverage but did not use the final controlled factorial/temporal design. | archived Luxembourg pipeline code; `results/legacy_experiments/luxembourg_multimap_benchmark/` |
| Negative | Contact timing | Six contact conditions, five settings, three variants, and two seeds. | Outcomes depended on the condition; the learned variant was not consistently best. | Recovered launcher and exact-policy code under `code/legacy/campaign_implementations/online_policy_learning/`; campaign `README.md`; compact CSV |
| Development | Four-zone decision ablation | Sign, budget, and earlier beta-weight choices across 13 variants and three seeds. | Unsigned Top-5 was best in this controlled study, informing later deterministic support choices. | `decision_ablation.py` and `run_decision_ablation.py`; campaign `README.md`; compact CSV |
| Negative | Online local-validation policy | Private per-vehicle policies trained during simulation from held-out local validation improvements across four single-zone campaigns. | Across nine paired learned-policy groups, the learner was better in four and worse in five; the unweighted mean paired difference was +0.035 dB, so no reliable advantage was established. | Exact learner and recovered launchers under `code/legacy/campaign_implementations/online_policy_learning/`; campaign `README.md`; compact CSV |
| Development | Real-map Kirchberg online policy | Online private-validation peer selection on a northern Kirchberg LuST trace, with three learned-policy seeds and 1,799 steps. | All runs completed with descriptive mean final RMSE 21.441 dB. The campaign demonstrates real-map execution but contains no included within-campaign comparator. | Online-policy source; captured configs; campaign `README.md`; compact CSV |
| Negative | Receiver-in-zone, global-transmitter policy sweep | Asymmetric Gare-Bonnevoie fidelity pairs with receivers in zone 1 and transmitters across the 800 m scene, sweeping pull budgets and token windows. | Results were schedule-dependent across two seeds; 30 of 32 attempts completed, and no robust learned-policy advantage was established. | Trace-generator option and online-policy source; protocol JSON; campaign `README.md`; compact CSV |
| Negative | Cross-map online-policy generalization | Source-domain online-policy training and candidate selection followed by three unseen synthetic maps. | Strong source-domain diagnostics did not provide sufficient unseen-map evidence. | `code/legacy/campaign_implementations/cross_map_policy_generalization/`; `results/legacy_experiments/cross_map_online_policy_generalization/` |
| Negative | Cross-map aligned-policy generalization | Six architectures, two seeds, deployment-aligned targets, and paired v2 unseen-map holdouts. | Mean policy RMSE was 30.071 dB versus 29.952 dB for the paired uninformed baseline; no reliable advantage was established. | Cross-map trainers/audits under `code/legacy/campaign_implementations/cross_map_policy_generalization/`; `results/legacy_experiments/cross_map_aligned_policy_generalization/` |

## How much raw detail is retained

For paper results, the raw inputs consumed by the aggregation are preserved.
For development experiments, the archive keeps configs, final metrics,
fidelity histories, communication records, learning summaries, completion
state, and checkpoints. Operational scheduler output, duplicate partial files,
profiling records, and large per-decision debug tables are not part of the
reader-facing archive.

The complete result-to-source index is `code/legacy/CAMPAIGN_CODE_MAP.md`.
