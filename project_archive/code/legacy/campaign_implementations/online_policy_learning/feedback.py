"""Delayed pairwise utility targets over future same-zone samples."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .metadata import ModelMetadata


@dataclass(frozen=True)
class ModelSnapshot:
    metadata: ModelMetadata
    state: dict[str, torch.Tensor]


@dataclass
class PendingPull:
    observation: torch.Tensor
    receiver_snapshot: ModelSnapshot
    provider_snapshot: ModelSnapshot
    reference_pairwise_state: dict[str, torch.Tensor]
    receiver_idx: int
    provider_idx: int
    mode: str
    zone: int
    timestep: int
    horizon: int
    initial_samples_x: list[list[float]] = field(default_factory=list)
    initial_samples_y: list[float] = field(default_factory=list)
    samples_x: list[list[float]] = field(default_factory=list)
    samples_y: list[float] = field(default_factory=list)
    sample_steps: list[int] = field(default_factory=list)

    @property
    def maturity_step(self) -> int:
        return int(self.timestep + self.horizon)


def advance_pending_pull(
    pull: PendingPull,
    *,
    step: int,
    receiver_zone: int,
    samples_x: list[list[float]],
    samples_y: list[float],
) -> bool:
    """Ingest only t+1..t+T same-zone samples; return whether it matured."""

    current = int(step)
    if (
        int(pull.timestep) < current <= pull.maturity_step
        and int(receiver_zone) == int(pull.zone)
    ):
        pull.samples_x.extend([list(row) for row in samples_x])
        pull.samples_y.extend(float(value) for value in samples_y)
        pull.sample_steps.extend([current] * len(samples_x))
    return current >= pull.maturity_step
