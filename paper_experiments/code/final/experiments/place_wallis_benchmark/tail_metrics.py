"""Shared temporal aggregation for definitive Place Wallis results."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_tail_evaluation_steps(
    sim_steps: int,
    *,
    count: int,
    stride: int,
) -> tuple[int, ...]:
    """Return endpoints of equal windows covering the final count*stride steps."""
    if count <= 0 or stride <= 0:
        raise ValueError("tail evaluation count and stride must be positive")
    period_start = int(sim_steps) - int(count) * int(stride)
    if period_start < 0:
        raise ValueError("simulation is too short for the tail evaluation")
    return tuple(
        period_start + int(stride) * (index + 1)
        for index in range(int(count))
    )


def temporal_metric_summary(
    history: list[dict[str, Any]],
    *,
    evaluation_steps: tuple[int, ...],
    metric_keys: tuple[str, ...],
) -> dict[str, object]:
    by_step = {
        int(row["step"]): row
        for row in history
        if int(row.get("step", -1)) in evaluation_steps
    }
    observed_steps = tuple(step for step in evaluation_steps if step in by_step)
    means: dict[str, float | None] = {}
    standard_deviations: dict[str, float | None] = {}
    values: dict[str, list[float]] = {}
    for key in metric_keys:
        rows = [
            float(by_step[step][key])
            for step in observed_steps
            if key in by_step[step]
        ]
        values[key] = rows
        means[key] = float(np.mean(rows)) if rows else None
        standard_deviations[key] = float(np.std(rows)) if rows else None
    stride = (
        evaluation_steps[1] - evaluation_steps[0]
        if len(evaluation_steps) > 1
        else 1
    )
    return {
        "period_start_step": int(evaluation_steps[0] - stride),
        "evaluation_steps": list(evaluation_steps),
        "observed_steps": list(observed_steps),
        "expected_evaluations": int(len(evaluation_steps)),
        "observed_evaluations": int(len(observed_steps)),
        "complete": bool(observed_steps == evaluation_steps),
        "aggregation": "arithmetic mean over equally weighted evaluation times",
        "mean": means,
        "standard_deviation": standard_deviations,
        "values": values,
    }
