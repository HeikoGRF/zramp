"""Experience-weighted one-shot predictor aggregation."""

from __future__ import annotations

from typing import Mapping

import torch


TensorState = Mapping[str, torch.Tensor]



def experience_weights(
    experiences: list[float] | tuple[float, ...],
    *,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Return normalized effective-experience weights."""

    if not experiences:
        raise ValueError("at least one candidate experience is required")
    if float(epsilon) <= 0.0:
        raise ValueError("experience epsilon must be positive")
    values = torch.tensor(experiences, dtype=torch.float64).clamp_min(0.0)
    values = values + float(epsilon)
    return (values / values.sum()).to(dtype=torch.float32)


def weighted_average(
    states: list[TensorState] | tuple[TensorState, ...],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Form one detached weighted state average without mutating candidates."""

    if not states:
        raise ValueError("at least one candidate state is required")
    flat_weights = weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if int(flat_weights.numel()) != len(states):
        raise ValueError("weights and candidate states must have the same length")
    if not torch.isfinite(flat_weights).all():
        raise ValueError("aggregation weights must be finite")
    if float(flat_weights.sum()) <= 0.0:
        raise ValueError("aggregation weights must have positive mass")
    flat_weights = flat_weights / flat_weights.sum()
    keys = tuple(states[0].keys())
    for state in states[1:]:
        if tuple(state.keys()) != keys:
            raise ValueError("candidate model states have different keys")

    averaged: dict[str, torch.Tensor] = {}
    for name in keys:
        tensors = [state[name].detach().to(device="cpu") for state in states]
        reference = tensors[0]
        if reference.is_floating_point() or reference.is_complex():
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for weight, tensor in zip(flat_weights, tensors):
                accumulator.add_(tensor.to(dtype=torch.float64), alpha=float(weight))
            averaged[name] = accumulator.to(dtype=reference.dtype)
        else:
            averaged[name] = reference.clone()
    return averaged

