#!/usr/bin/env python3
"""Run Place Wallis greedy sharing with finite-segment capsule support."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.place_wallis_benchmark import finite_capsule_support as support
from experiments.place_wallis_benchmark import run_capsule_greedy as runner


runner.Capsule = support.Capsule
runner.CapsuleGatedMLP = support.CapsuleGatedMLP
runner.CapsuleParams = support.CapsuleParams
runner.CapsuleRow = support.CapsuleRow
runner.add_capsule_vectorized = support.add_capsule_vectorized
runner.deserialize_capsules = support.deserialize_capsules
runner.remote_union = support.remote_union
runner.capsule_delta = support.capsule_delta
runner.serialize_capsules = support.serialize_capsules
runner.ribbon_self_test = support.self_test
runner.SUPPORT_VARIANT = "finite-corridor-capsules"
runner.SUPPORT_RECORD_FLOATS = 5
runner.SUPPORT_PAYLOAD_DESCRIPTION = "two endpoints and observation mass"
runner.SUPPORT_MERGE_DESCRIPTION = (
    "finite-segment angle, lateral-distance, and longitudinal-gap merge"
)
runner.CapsuleGreedySimulation.checkpoint_format = (
    "place_wallis_finite_capsule_greedy_metrics_v1"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
