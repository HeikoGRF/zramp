"""Per-step stochastic pull-capacity and provider selection helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch


def sample_pull_capacity(
    budget: float,
    num_neighbors: int,
    *,
    rng: random.Random,
) -> int:
    """Sample floor(B) plus an independent Bernoulli fractional slot."""

    if not math.isfinite(float(budget)) or float(budget) < 0.0:
        raise ValueError("pull budget must be a finite non-negative number")
    neighbors = max(0, int(num_neighbors))
    whole = int(math.floor(float(budget)))
    fractional = float(budget) - float(whole)
    sampled = whole + int(fractional > 0.0 and rng.random() < fractional)
    return min(neighbors, sampled)


def select_top_k(
    scores: torch.Tensor,
    provider_ids: Sequence[int],
    num_slots: int,
) -> list[int]:
    """Select at most K strictly positive scores with stable ID tie-breaking."""

    values = scores.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if int(values.numel()) != len(provider_ids):
        raise ValueError("scores and provider_ids must have the same length")
    limit = max(0, min(int(num_slots), len(provider_ids)))
    ranked = sorted(
        (
            (float(score), int(provider_id))
            for score, provider_id in zip(values.tolist(), provider_ids)
            if math.isfinite(float(score)) and float(score) > 0.0
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [provider_id for _score, provider_id in ranked[:limit]]


def select_random_subset(
    provider_ids: Sequence[int],
    num_slots: int,
    *,
    rng: random.Random,
) -> list[int]:
    """Select exactly min(K, N) unique feasible providers for exploration."""

    ids = sorted({int(provider_id) for provider_id in provider_ids})
    limit = max(0, min(int(num_slots), len(ids)))
    return sorted(rng.sample(ids, limit)) if limit else []
