# Rerunning archived experiments

The archived campaigns span several generations of the project. They are
preserved at different reproduction levels: some retain an original launcher,
some retain a Python entry point plus exact run configurations, and some are
best treated as reviewable provenance because an ad-hoc scheduler wrapper or
external source input was not preserved.

## Common setup

From the repository root:

```bash
conda env create -f paper_experiments/environment/environment.yml
conda activate zramp-archive
source project_archive/scripts/activate_legacy_paths.sh
```

Use `paper_experiments/input_data/` for retained maps, fixed test sets, zone
metadata, and LuST generation provenance. Use
`paper_experiments/scripts/download_lust3d.sh` and the input-generation
workflow when a campaign needs regenerated Luxembourg mobility or radio data.
Write new outputs outside the Git repository.

Legacy shell launchers preserve their original cluster defaults. Before
submitting one, inspect it and replace its repository, data, Python, output,
account, partition, and Slurm configuration paths. Captured `config.json`
files are the authority for run-level settings.

## Campaign reproduction map

| Campaign | Reproduction entry point | Retained status |
|---|---|---|
| Place Wallis support methods | `code/legacy/campaign_implementations/experiments/place_wallis_benchmark/` | Original method launchers, runner code, map/test-set assets, and results retained. |
| Learned support-selection pretraining | `code/legacy/campaign_implementations/experiments/support_acquisition_pretraining/` | Training entry points, launchers, summaries, and selected checkpoints retained. |
| Factorial Expert Bank | `code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_factorial_zone_pipeline.sh` and `submit_factorial_pq_sweep.sh` | Original launchers, nine-map assets, configurations, manifests, and results retained. |
| Deterministic single-model versus learned Expert Bank | Maintained cell-grid method and shared runtime under `../paper_experiments/code/`; exact run settings and manifests under `../paper_experiments/experiment_configs/paper/luxembourg_cell_grid_factorial_sweeps_v1/` | All source, one-realization raw outputs, and compact summaries are retained. The original one-off sample-count/penalty array wrapper was not recovered; use the per-run configs and method metadata as the historical authority. |
| Minimum-intensity sensitivity | Cell-grid implementation under the Place Wallis source plus captured run configs | Exact one-off scheduler command was not recovered; implementation, settings, and outputs are retained. |
| Matched-zone sensitivity | `submit_matched_b2v3_synchronized_sweep.sh` and `submit_matched_b2v3_entry_anchored_sweep.sh` | Original reusable launchers and run configurations retained. |
| Nested map sizes | `submit_nested_size_pipeline.sh` | Original launcher, relevant map/test-set assets, manifests, and results retained. |
| Controlled density | `augment_mobility_density.py` and `submit_density_sweep.sh` | Generator, launcher, configurations, and results retained. |
| Additional Luxembourg maps | Evaluation-zone preparation and launchers in the same SUMO directory | Reusable pipeline retained; run-specific settings are in captured configs. |
| Early Munich benchmarks | `code/legacy/early_munich_experiments/` | Original drivers and results retained; external dataset assumptions predate the final Luxembourg workflow. |
| Contact timing | `code/legacy/campaign_implementations/online_policy_learning/submit_contact_timing_and_policy_variant_sweep.sh` | Online-policy code, recovered launcher, configurations, and 180 final records retained. |
| Four-zone decision ablation | `decision_ablation.py` and `run_decision_ablation.py` | Exact driver and all run configs retained; the original array wrapper was not recovered. |
| Online local-validation policy | `run_online_local_validation_policy.py`, `local_validation_reward.py`, and the recovered policy launchers | Completed configs, training histories, validation records, and final metrics retained. |
| Kirchberg real-map online policy | `run_online_local_validation_policy.py` plus the LuST/Sionna input workflow | Three completed learned-policy runs, captured configs, compact training and validation histories, and final metrics retained; the one-off array wrapper was not recovered. |
| Receiver-in-zone, global-transmitter sweep | The same policy runner; `generate_pilot_rssi_trace.py --fidelity-global-senders` creates the asymmetric held-out protocol | Thirty completed and two incomplete attempts, protocol metadata, available exact configs, and compact histories retained; the one-off array wrapper was not recovered. |
| Cross-map policy studies | `code/legacy/campaign_implementations/cross_map_policy_generalization/` and cross-map scripts under `online_policy_learning/` | Python stages, checkpoints, reports, configs, and completed holdouts retained; some ad-hoc scheduler commands were not standalone files. |

The more detailed file-level mapping is
[code/legacy/CAMPAIGN_CODE_MAP.md](code/legacy/CAMPAIGN_CODE_MAP.md).

## Validate the archived conclusions

The common aggregation is fully portable:

```bash
python3 project_archive/code/legacy/analysis/prepare_real_map_policy_campaigns.py
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```

For a particular raw run, compare its `config.json`, `metrics.json`,
`final_fidelity.json`, communication records, and training summaries with
the campaign README. This is often more useful than re-executing older
cluster-specific launchers.
