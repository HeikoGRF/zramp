# Receiver-in-zone, global-transmitter policy sweep

## Purpose and status

This is a **mixed, exploratory legacy experiment**, not a paper result. It
studies an asymmetric held-out evaluation geometry on the real Gare-Bonnevoie
Luxembourg scene: the receiver is inside zone 1, while the transmitter may be
anywhere in the full 800 m by 800 m scene. This is the experiment in which only
one endpoint of an evaluated radio link had to be in the zone.

The endpoint rule applies to the fixed held-out fidelity pairs. It must not be
read as a claim that every training measurement or contact link used the same
geometry. `evaluation_protocol.json` records this scope explicitly.

The scene is split into a 2 by 2 grid of 400 m cells. Evaluation uses 10,000
held-out pairs, 20 transmitter candidates, dynamic vehicle blockers, and zone
1 receivers. The sweep varies:

- `pulls_per_receiver_step` (K): 1, 2, or 4 maximum pulls;
- `token_window_steps` (S): 1, 2, 5, 10, 25, or 50 where available; and
- selection: online `learned_policy` versus `uninformed_selection`.

Each planned cell has two seeds. There were 32 attempts: 30 completed. The two
learned-policy runs at K=4, S=1 stopped at their wall-time limits and are
reported as incomplete, not converted into final comparisons.

## Result

The two-seed evidence is condition-dependent. Relative to uninformed
selection, the learned policy reduced mean final RMSE at K=1/S=1 by 1.010 dB,
at K=1/S=2 by 0.385 dB, and at K=2/S=1 by 0.874 dB. It increased mean final
RMSE at K=1 with S=5, 10, 25, and 50. Only K=1/S=1 and K=2/S=1 improved in both
paired seeds. This is exploratory evidence of schedule sensitivity, not a
robust learned-policy advantage.

## Evidence retained

- `evaluation_protocol.json`: exact interpretation of the asymmetric held-out
  pairs and a pointer to the trace-generator option.
- `run_inventory.csv`: all 32 attempts, completion state, dimensions, and final
  RMSE where a run completed.
- `runs/pulls_*/...`: captured run metadata, final metrics, training summaries,
  compact validation histories, fidelity histories, and communication records.
- `../../../figures/data/exploratory_experiment_summaries/receiver_in_zone_global_sender_policy_sweep_summary.csv`:
  grouped means, paired differences, and incomplete-run counts.
- `../../../experiment_configs/experimental/receiver_in_zone_global_sender_policy_sweep/`:
  the standalone original config records that existed in the source runs.

Three historical run directories did not contain a standalone `config.json`.
The archive does not fabricate those files. Their method, seed, K, S, completion
state, and policy settings remain explicit in the directory structure,
`run_inventory.csv`, `learning_summary.json`, and
`communication_overhead_assumptions.json`; shared simulator settings are also
recorded by the other exact configs in the same campaign.

## Compacting and reproduction

Large per-decision tables and scheduler logs are excluded. The retained
per-step histories preserve training, validation, communication, and fidelity
progress without turning this legacy campaign into a multi-gigabyte archive.
Regenerate the compact package and its table with:

```bash
python3 project_archive/code/legacy/analysis/prepare_real_map_policy_campaigns.py \
  --data-root /path/to/luxembourg_real_city
python3 project_archive/code/legacy/analysis/build_exploratory_experiment_summaries.py
```

The online-policy implementation is under
`../../../code/legacy/campaign_implementations/online_policy_learning/`. The
trace generator is
`../../../code/legacy/campaign_implementations/SUMO/luxembourg_real_city/generate_pilot_rssi_trace.py`;
use its `--fidelity-global-senders` option for this held-out evaluation
geometry. The exact one-off cluster array wrapper was not recovered. Regenerate
LuST mobility and Sionna traces with the repository workflow and write outputs
outside the repository.
