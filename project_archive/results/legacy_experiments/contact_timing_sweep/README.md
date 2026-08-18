# Contact-timing sweep

## Question and design

This exploratory single-zone campaign tested whether sharing behavior remained
stable as contact and update timing changed. It contains 180 completed runs:
six top-level `n..._r...` conditions, five `s...` settings, three variants, two
seeds, and 1,000 simulation steps per run.

The variants are:

- `random`: uninformed action selection;
- `shared_policy`: the learned policy shared across the experiment; and
- `frozen_samples`: a non-updating reference using frozen evidence.

The recovered launcher confirms that `n` is the node count, `r` is the
diagnostic regular-role count used to select the prepared trace, and `s` is the
token/contact window in simulation steps.

## Result

Across the 30 condition groups per variant, the unweighted means of the
per-condition final RMSE means were 20.266 dB for `frozen_samples`, 21.016 dB
for `shared_policy`, and 21.384 dB for `random`. Individual conditions changed
the ordering. This established that apparent policy gains depended strongly on
the timing/contact setting and did not justify using this single-zone campaign
as the reported result.

## Evidence and reproduction

`figures/data/exploratory_experiment_summaries/contact_timing_summary.csv` contains all 90 grouped
rows, each averaged over two seeds. Regenerate it with
`code/legacy/analysis/build_exploratory_experiment_summaries.py`.

Within a run, start with `config.json`, `final_fidelity.json`, and
`fidelity.csv`; then inspect `sharing_events.csv` and the learning summaries.
The exact online learner and corrected variants are under
`code/legacy/campaign_implementations/online_policy_learning/`. The original array launcher
is retained there as `submit_contact_timing_and_policy_variant_sweep.sh`; it records
the complete parameter grid and command. Machine-specific paths must be
replaced with the archive paths before resubmission.
