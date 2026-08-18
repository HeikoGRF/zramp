#!/usr/bin/env python3
"""Run greedy sharing with bounded overlapping straight support planes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.place_wallis_benchmark import run_capsule_greedy as runner
from experiments.place_wallis_benchmark import overlapping_plane_support as support


runner.Capsule = support.OverlappingPlane
runner.CapsuleGatedMLP = support.OverlappingPlaneGatedMLP
runner.CapsuleParams = support.OverlappingPlaneParams
runner.CapsuleRow = support.PlaneRow
runner.add_capsule_vectorized = support.add_plane
runner.deserialize_capsules = support.deserialize_planes
runner.remote_union = support.remote_union
runner.capsule_delta = support.plane_delta
runner.serialize_capsules = support.serialize_planes
runner.ribbon_self_test = support.self_test
runner.SUPPORT_VARIANT = "overlapping-straight-planes"
runner.SUPPORT_RECORD_FLOATS = 11
runner.SUPPORT_PAYLOAD_DESCRIPTION = (
    "fixed-axis endpoints, four borders, mass, maximum observed link length, "
    "and direction spread"
)
runner.SUPPORT_MERGE_DESCRIPTION = (
    "overlapping fixed-axis corridor planes merged only for longitudinal "
    "overlap within a 12 m maximum envelope width"
)
runner.CapsuleGreedySimulation.checkpoint_format = (
    "place_wallis_overlapping_plane_greedy_checkpoint_v1"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
