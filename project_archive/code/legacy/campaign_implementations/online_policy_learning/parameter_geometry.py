"""Parameter-space maturity, stability, and aggregation geometry.

The geometry is defined relative to the common predictor initialization.  It
does not use validation samples or local sample counts.  Local SGD stability
and post-aggregation retention are tracked separately so a large merge jump
cannot masquerade as stable local learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


TensorState = Mapping[str, torch.Tensor]


def _floating_names(reference: TensorState) -> tuple[str, ...]:
    return tuple(
        name
        for name, value in reference.items()
        if torch.is_floating_point(value) and "_support_" not in name
    )


def normalized_parameter_delta(
    state: TensorState,
    reference: TensorState,
    *,
    scale_floor: float = 0.05,
) -> torch.Tensor:
    """Flatten a layer-normalized displacement from common initialization."""

    pieces: list[torch.Tensor] = []
    for name in _floating_names(reference):
        initial = reference[name].detach().to(device="cpu", dtype=torch.float64)
        current = state[name].detach().to(device="cpu", dtype=torch.float64)
        if current.shape != initial.shape:
            raise ValueError(f"parameter shape changed for {name}")
        rms = float(torch.sqrt(torch.mean(torch.square(initial))).item())
        scale = max(float(scale_floor), rms)
        pieces.append(((current - initial) / scale).reshape(-1))
    if not pieces:
        return torch.empty((0,), dtype=torch.float64)
    return torch.cat(pieces)


def parameter_delta_state(
    state: TensorState,
    reference: TensorState,
) -> dict[str, torch.Tensor]:
    """Return a state-shaped displacement used by the learned encoder."""

    result: dict[str, torch.Tensor] = {}
    for name, initial in reference.items():
        current = state[name].detach().to(device="cpu")
        if torch.is_floating_point(initial):
            result[name] = current - initial.detach().to(device="cpu")
        else:
            result[name] = current.clone()
    return result


def _rms(vector: torch.Tensor) -> float:
    if int(vector.numel()) == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(torch.square(vector))).item())


@dataclass(frozen=True)
class GeometrySummary:
    radial_distance: float
    training_stability: float
    merge_stability: float
    maturity: float


@dataclass(frozen=True)
class GeometryAggregation:
    alpha: float
    gross_reward: float
    receiver: GeometrySummary
    provider: GeometrySummary
    pair_distance: float
    normalized_novelty: float
    cosine: float
    cancellation_ratio: float
    trust_ratio: float
    objective_before: float
    objective_after: float
    evaluations: tuple[tuple[float, float], ...]


class ParameterGeometryTracker:
    """Separate local-SGD consistency from retention of imported updates."""

    def __init__(
        self,
        reference: TensorState,
        *,
        ema_decay: float = 0.9,
        stability_warmup_updates: int = 8,
        retention_updates: int = 8,
        scale_floor: float = 0.05,
    ) -> None:
        self.reference = {
            name: value.detach().to(device="cpu").clone()
            for name, value in reference.items()
        }
        self.ema_decay = float(ema_decay)
        self.stability_warmup_updates = max(1, int(stability_warmup_updates))
        self.retention_updates = max(1, int(retention_updates))
        self.scale_floor = float(scale_floor)
        self.local_updates = 0
        self.local_updates_since_merge = 0
        self._ema_update: torch.Tensor | None = None
        self._ema_update_norm = 0.0
        self._ema_update_norm_sq = 0.0
        self._merge_stability = 1.0
        self._merge_observations = 0
        self._pending_premerge: torch.Tensor | None = None
        self._pending_jump: torch.Tensor | None = None
        self._pending_updates = 0

    def vector(self, state: TensorState) -> torch.Tensor:
        return normalized_parameter_delta(
            state, self.reference, scale_floor=self.scale_floor
        )

    def radial_distance(self, state: TensorState) -> float:
        return _rms(self.vector(state))

    def training_stability(self) -> float:
        if self.local_updates <= 0 or self._ema_update is None:
            return 0.0
        mean_rms = _rms(self._ema_update)
        second_rms = math.sqrt(max(self._ema_update_norm_sq, 0.0))
        directional_coherence = (
            0.0 if second_rms <= 1.0e-12 else mean_rms / second_rms
        )
        norm_stability = (
            0.0
            if self._ema_update_norm_sq <= 1.0e-12
            else self._ema_update_norm
            / math.sqrt(self._ema_update_norm_sq)
        )
        confidence = 1.0 - math.exp(
            -float(self.local_updates) / float(self.stability_warmup_updates)
        )
        return float(
            np.clip(
                math.sqrt(
                    max(0.0, directional_coherence)
                    * max(0.0, norm_stability)
                )
                * confidence,
                0.0,
                1.0,
            )
        )

    def merge_stability(self) -> float:
        return float(np.clip(self._merge_stability, 0.0, 1.0))

    def summary(
        self, state: TensorState, *, radial_scale: float
    ) -> GeometrySummary:
        radial = self.radial_distance(state)
        radial_maturity = radial / (radial + max(float(radial_scale), 1.0e-9))
        training = self.training_stability()
        merge = self.merge_stability()
        # Maturity describes stable local training. Post-merge retention is
        # tracked separately and must not make a complementary expert look bad.
        maturity = radial_maturity * (0.2 + 0.8 * training)
        return GeometrySummary(
            radial_distance=float(radial),
            training_stability=float(training),
            merge_stability=float(merge),
            maturity=float(np.clip(maturity, 0.0, 1.0)),
        )

    def _observe_pending_retention(self, state: TensorState) -> None:
        if self._pending_premerge is None or self._pending_jump is None:
            return
        current = self.vector(state)
        jump = self._pending_jump
        denominator = float(torch.dot(jump, jump).item())
        retention = 1.0
        if denominator > 1.0e-12:
            retained = float(
                torch.dot(current - self._pending_premerge, jump).item()
            ) / denominator
            retention = float(np.clip(retained, 0.0, 1.0))
        decay = self.ema_decay
        if self._merge_observations <= 0:
            self._merge_stability = retention
        else:
            self._merge_stability = (
                decay * self._merge_stability + (1.0 - decay) * retention
            )
        self._merge_observations += 1
        self._pending_updates += 1
        if self._pending_updates >= self.retention_updates:
            self._pending_premerge = None
            self._pending_jump = None
            self._pending_updates = 0

    def observe_local_update(
        self, before: TensorState, after: TensorState
    ) -> None:
        before_vector = self.vector(before)
        after_vector = self.vector(after)
        update = after_vector - before_vector
        update_rms = _rms(update)
        decay = self.ema_decay
        if self._ema_update is None:
            self._ema_update = update.clone()
            self._ema_update_norm = update_rms
            self._ema_update_norm_sq = update_rms * update_rms
        else:
            self._ema_update.mul_(decay).add_(update, alpha=1.0 - decay)
            self._ema_update_norm = (
                decay * self._ema_update_norm + (1.0 - decay) * update_rms
            )
            self._ema_update_norm_sq = (
                decay * self._ema_update_norm_sq
                + (1.0 - decay) * update_rms * update_rms
            )
        self.local_updates += 1
        self.local_updates_since_merge += 1
        self._observe_pending_retention(after)

    def inherit_merge(
        self,
        provider: "ParameterGeometryTracker",
        *,
        alpha: float,
        before: TensorState,
        after: TensorState,
    ) -> None:
        """Carry stability through aggregation, then monitor actual retention."""

        # The repository-wide convention is
        #   aggregate = alpha * receiver + (1 - alpha) * provider.
        # Therefore provider history is inherited with weight (1 - alpha).
        weight = 1.0 - float(np.clip(alpha, 0.0, 1.0))
        if provider._ema_update is not None:
            if self._ema_update is None:
                self._ema_update = provider._ema_update.clone()
            else:
                self._ema_update.mul_(1.0 - weight).add_(
                    provider._ema_update, alpha=weight
                )
            self._ema_update_norm = (
                (1.0 - weight) * self._ema_update_norm
                + weight * provider._ema_update_norm
            )
            self._ema_update_norm_sq = (
                (1.0 - weight) * self._ema_update_norm_sq
                + weight * provider._ema_update_norm_sq
            )
            self.local_updates = max(
                self.local_updates, provider.local_updates
            )
        self._merge_stability = (
            (1.0 - weight) * self.merge_stability()
            + weight * provider.merge_stability()
        )
        self._merge_observations = max(
            self._merge_observations, provider._merge_observations
        )
        premerge = self.vector(before)
        postmerge = self.vector(after)
        self._pending_premerge = premerge
        self._pending_jump = postmerge - premerge
        self._pending_updates = 0
        self.local_updates_since_merge = 0


def select_geometry_aggregation(
    receiver_state: TensorState,
    provider_state: TensorState,
    receiver_tracker: ParameterGeometryTracker,
    provider_tracker: ParameterGeometryTracker,
    *,
    alpha_grid: Sequence[float],
    radial_scale: float = 0.10,
    cancellation_penalty: float = 1.0,
    trust_penalty: float = 1.0,
    trust_radius: float = 1.0,
) -> GeometryAggregation:
    """Choose receiver weight alpha using only model-space geometry.

    ``alpha=1`` retains the receiver and is the no-aggregation baseline;
    ``alpha=0`` copies the provider.  This matches ``interpolate_states``.
    """

    a = receiver_tracker.vector(receiver_state)
    b = provider_tracker.vector(provider_state)
    if a.shape != b.shape:
        raise ValueError("receiver and provider parameter vectors differ")
    receiver = receiver_tracker.summary(receiver_state, radial_scale=radial_scale)
    provider = provider_tracker.summary(provider_state, radial_scale=radial_scale)
    difference = b - a
    distance = _rms(difference)
    denominator = receiver.radial_distance + provider.radial_distance + radial_scale
    novelty = float(np.clip(distance / max(denominator, 1.0e-12), 0.0, 1.0))
    dot = float(torch.dot(a, b).item())
    norm_a = float(torch.linalg.vector_norm(a).item())
    norm_b = float(torch.linalg.vector_norm(b).item())
    cosine = 1.0 if norm_a <= 1.0e-12 or norm_b <= 1.0e-12 else dot / (norm_a * norm_b)
    cosine = float(np.clip(cosine, -1.0, 1.0))

    def objective(alpha: float) -> tuple[float, float, float]:
        value = float(np.clip(alpha, 0.0, 1.0))
        candidate = value * a + (1.0 - value) * b
        radial_candidate = _rms(candidate)
        radial_linear = (
            value * receiver.radial_distance
            + (1.0 - value) * provider.radial_distance
        )
        cancellation = (
            1.0
            if radial_linear <= 1.0e-12
            else float(np.clip(radial_candidate / radial_linear, 0.0, 1.0))
        )
        trust = (1.0 - value) * distance / max(
            max(
                receiver.radial_distance,
                provider.radial_distance,
            )
            + radial_scale,
            1.0e-12,
        )
        endpoint_loss = novelty * novelty * (
            receiver.maturity * (1.0 - value) * (1.0 - value)
            + provider.maturity * value * value
        )
        cancel_loss = (
            float(cancellation_penalty)
            * (receiver.maturity + provider.maturity)
            * max(0.0, -cosine)
            * (1.0 - cancellation)
            * (1.0 - cancellation)
        )
        trust_excess = max(0.0, trust - float(trust_radius))
        trust_loss = float(trust_penalty) * trust_excess * trust_excess
        return endpoint_loss + cancel_loss + trust_loss, cancellation, trust

    candidates = sorted(
        {float(np.clip(value, 0.0, 1.0)) for value in alpha_grid} | {1.0}
    )
    rows = [(alpha, *objective(alpha)) for alpha in candidates]
    # If objectives tie, retain more of the receiver. This prevents redundant
    # or exactly opposing models from causing a full-provider replacement.
    best = min(rows, key=lambda row: (row[1], abs(1.0 - row[0])))
    baseline = next(row for row in rows if row[0] == 1.0)
    reward = max(0.0, float(baseline[1]) - float(best[1]))
    return GeometryAggregation(
        alpha=float(best[0]),
        gross_reward=float(reward),
        receiver=receiver,
        provider=provider,
        pair_distance=float(distance),
        normalized_novelty=float(novelty),
        cosine=float(cosine),
        cancellation_ratio=float(best[2]),
        trust_ratio=float(best[3]),
        objective_before=float(baseline[1]),
        objective_after=float(best[1]),
        evaluations=tuple((float(row[0]), float(row[1])) for row in rows),
    )
