# Support-acquisition pretraining

This campaign trained several representations intended to predict useful
support acquisitions. The training was sufficient for follow-up development,
but learned acquisition scoring was not used by the final deterministic
intensity-count rule.

Training code and original launchers are under
`project_archive/code/legacy/campaign_implementations/experiments/support_acquisition_pretraining/`.
Selected checkpoints are under
`project_archive/trained_models/experimental/acquisition_pretraining/`, while
the retained training histories and summaries are in this result directory.

To rerun, create the paper environment, source
`project_archive/scripts/activate_legacy_paths.sh`, choose the launcher for
the representation documented by its training summary, replace historical
input and output paths, and submit it on suitable hardware.

