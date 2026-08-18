#!/usr/bin/env python3
"""Isolated entry point for the corrected 150 m policy variants."""

from __future__ import annotations

import sys

from online_policy_learning.online_policy_variants import (
    AllNeighborPolicySharingSimulation,
    FrozenEncoderSampleSharingSimulation,
)
from online_policy_learning.run_online_local_validation_policy import main


def run(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        marker = values.index("--corrected-variant")
        variant = values[marker + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "--corrected-variant must be frozen-samples or shared-policy"
        ) from exc
    del values[marker : marker + 2]
    classes = {
        "frozen-samples": FrozenEncoderSampleSharingSimulation,
        "shared-policy": AllNeighborPolicySharingSimulation,
    }
    try:
        simulation_cls = classes[variant]
    except KeyError as exc:
        raise ValueError(
            "--corrected-variant must be frozen-samples or shared-policy"
        ) from exc
    return main(values, simulation_cls=simulation_cls)


if __name__ == "__main__":
    raise SystemExit(run())
