"""Directional pulls with bidirectional predictor exchange and private CV.

For a directed contact ``A <- B`` the policy first sees compact provider
metadata. A pass produces the immediate bandit target zero. A pull exchanges
both predictor states once, chooses a continuous interpolation weight on two
private optimization-validation sets, and measures the reward on two disjoint
reward-validation sets. Only A can adopt the aggregate; B is never mutated by
the directed action.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import torch
import torch.optim as optim

from model import (
    is_mergeable_evidence_state,
    mergeable_evidence_delta_state,
    mergeable_evidence_diagonal_information_gain,
    mergeable_evidence_direction_nbytes,
    mergeable_evidence_union_states,
)

from rl_reward_experiment.reward_modes import RewardMode, Transition

from .experience_diversity import experience_score
from .metadata import build_metadata, build_observation
from .simulation import BootstrapPolicySharingSimulation
from .utility import aggregate_policy_states


TensorState = Mapping[str, torch.Tensor]
VALIDATION_CYCLE = 10
VALIDATION_OPT_SLOT = 8
VALIDATION_REWARD_SLOT = 9
FLOAT32_BYTES = 4


def validation_allocation_counts(
    samples_seen: int, batch_size: int
) -> tuple[int, int, int]:
    """Return exact streaming 80/10/10 counts for one untrained batch.

    Every consecutive block of ten samples contributes eight training rows,
    one optimization-validation row, and one reward-validation row. Counting
    slots before selecting rows also works for one-sample batches.
    """

    seen = int(samples_seen)
    size = int(batch_size)
    if seen < 0 or size < 0:
        raise ValueError("samples_seen and batch_size must be non-negative")
    n_opt = 0
    n_reward = 0
    for ordinal in range(seen, seen + size):
        slot = ordinal % VALIDATION_CYCLE
        n_opt += int(slot == VALIDATION_OPT_SLOT)
        n_reward += int(slot == VALIDATION_REWARD_SLOT)
    return size - n_opt - n_reward, n_opt, n_reward


def _coordinate_rows(
    values: np.ndarray | torch.Tensor | list[list[float]],
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.size == 0 and rows.ndim == 1:
        rows = np.empty((0, 4), dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("sample features must have shape [num_samples, feature_dim]")
    if int(rows.shape[1]) < 4:
        raise ValueError("sample features must contain four normalized coordinates")
    return rows[:, :4]


def _farthest_candidate(
    coordinates: np.ndarray,
    remaining: list[int],
    references: np.ndarray,
) -> int:
    """Choose one deterministic maximin row, with a dispersion first point."""

    candidates = coordinates[np.asarray(remaining, dtype=np.int64)]
    if int(references.shape[0]) > 0:
        distances = np.linalg.norm(
            candidates[:, None, :] - references[None, :, :], axis=2
        )
        scores = distances.min(axis=1)
    elif len(remaining) > 1:
        distances = np.linalg.norm(
            candidates[:, None, :] - candidates[None, :, :], axis=2
        )
        scores = distances.sum(axis=1) / float(len(remaining) - 1)
    else:
        scores = np.zeros((len(remaining),), dtype=np.float64)
    best_position = max(
        range(len(remaining)),
        key=lambda position: (float(scores[position]), -int(remaining[position])),
    )
    return int(remaining[best_position])


def select_validation_indices(
    batch_features: np.ndarray | torch.Tensor | list[list[float]],
    opt_existing: np.ndarray | torch.Tensor | list[list[float]],
    reward_existing: np.ndarray | torch.Tensor | list[list[float]],
    n_opt: int,
    n_reward: int,
) -> tuple[list[int], list[int], list[int]]:
    """Max-distance split of a new batch into disjoint train/opt/reward rows."""

    coordinates = _coordinate_rows(batch_features)
    opt_reference = _coordinate_rows(opt_existing)
    reward_reference = _coordinate_rows(reward_existing)
    opt_count = int(n_opt)
    reward_count = int(n_reward)
    if opt_count < 0 or reward_count < 0:
        raise ValueError("validation counts must be non-negative")
    if opt_count + reward_count > int(coordinates.shape[0]):
        raise ValueError("validation counts exceed the new batch size")

    remaining = list(range(int(coordinates.shape[0])))
    opt_selected: list[int] = []
    reward_selected: list[int] = []
    while len(opt_selected) < opt_count or len(reward_selected) < reward_count:
        if len(opt_selected) < opt_count:
            refs = (
                np.concatenate(
                    [
                        opt_reference,
                        coordinates[np.asarray(opt_selected, dtype=np.int64)],
                    ],
                    axis=0,
                )
                if opt_selected
                else opt_reference
            )
            chosen = _farthest_candidate(coordinates, remaining, refs)
            opt_selected.append(chosen)
            remaining.remove(chosen)
        if len(reward_selected) < reward_count:
            refs = (
                np.concatenate(
                    [
                        reward_reference,
                        coordinates[np.asarray(reward_selected, dtype=np.int64)],
                    ],
                    axis=0,
                )
                if reward_selected
                else reward_reference
            )
            chosen = _farthest_candidate(coordinates, remaining, refs)
            reward_selected.append(chosen)
            remaining.remove(chosen)
    return sorted(remaining), sorted(opt_selected), sorted(reward_selected)


def validation_quality(
    features: np.ndarray | torch.Tensor | list[list[float]],
) -> float:
    """Apply the existing quantity-times-diversity score in spatial 4-D."""

    coordinates = _coordinate_rows(features)
    rows = torch.as_tensor(coordinates, dtype=torch.float32)
    # The existing helper deliberately caps the pairwise-distance sample at
    # 512 rows. Use a private generator so recalculating cached metadata is
    # deterministic and does not perturb the simulation RNG.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return float(experience_score(rows, generator=generator))


def quality_weighted_loss(
    loss_a: float,
    quality_a: float,
    loss_b: float,
    quality_b: float,
    *,
    epsilon: float = 1.0e-12,
) -> float | None:
    """Combine two private MSEs, or return ``None`` for zero total quality."""

    qa = max(0.0, float(quality_a))
    qb = max(0.0, float(quality_b))
    if qa + qb <= 0.0:
        return None
    eps = max(0.0, float(epsilon))
    return float((qa * float(loss_a) + qb * float(loss_b)) / (qa + qb + eps))


def interpolate_states(
    state_a: TensorState,
    state_b: TensorState,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Return ``alpha*A + (1-alpha)*B`` without mutating either endpoint."""

    if (
        is_mergeable_evidence_state(state_a)
        and is_mergeable_evidence_state(state_b)
    ):
        return mergeable_evidence_union_states(
            dict(state_a), dict(state_b)
        )

    weight = float(alpha)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if tuple(state_a.keys()) != tuple(state_b.keys()):
        raise ValueError("candidate model states have different keys")
    if weight == 1.0:
        return {
            name: value.detach().cpu().clone() for name, value in state_a.items()
        }
    if weight == 0.0:
        return {
            name: value.detach().cpu().clone() for name, value in state_b.items()
        }

    # Local-support maps carry evidence per cell. Plain averaging would halve
    # a learned delta whenever the other vehicle never observed that cell.
    grid_key = "grid.weight"
    support_key = "support.weight"
    support_aware = (
        grid_key in state_a
        and support_key in state_a
        and grid_key in state_b
        and support_key in state_b
    )
    support_left: torch.Tensor | None = None
    support_right: torch.Tensor | None = None
    if support_aware:
        support_left = state_a[support_key].detach().cpu().to(dtype=torch.float64)
        support_right = state_b[support_key].detach().cpu().to(dtype=torch.float64)
        if support_left.shape != support_right.shape:
            raise ValueError("candidate support tensors have different shapes")

    aggregate: dict[str, torch.Tensor] = {}
    for name in state_a:
        left = state_a[name].detach().cpu()
        right = state_b[name].detach().cpu()
        if left.shape != right.shape:
            raise ValueError(f"candidate tensor {name!r} has different shapes")
        if support_aware and name == grid_key:
            assert support_left is not None and support_right is not None
            left_mass = support_left * weight
            right_mass = support_right * (1.0 - weight)
            total_mass = left_mass + right_mass
            numerator = left.to(dtype=torch.float64) * left_mass
            numerator.add_(right.to(dtype=torch.float64) * right_mass)
            ordinary = left.to(dtype=torch.float64) * weight
            ordinary.add_(right.to(dtype=torch.float64), alpha=1.0 - weight)
            mixed = torch.where(
                total_mass > 0.0,
                numerator / total_mass.clamp_min(1.0e-12),
                ordinary,
            )
            aggregate[name] = mixed.to(dtype=left.dtype)
            continue
        if support_aware and name == support_key:
            assert support_left is not None and support_right is not None
            aggregate[name] = (
                support_left * weight + support_right * (1.0 - weight)
            ).to(dtype=left.dtype)
            continue
        if left.is_floating_point() or left.is_complex():
            mixed = left.to(dtype=torch.float64) * weight
            mixed.add_(right.to(dtype=torch.float64), alpha=1.0 - weight)
            aggregate[name] = mixed.to(dtype=left.dtype)
        else:
            aggregate[name] = left.clone()
    return aggregate


@dataclass(frozen=True)
class BoundedMinimum:
    alpha: float
    value: float
    evaluations: tuple[tuple[float, float], ...]


def minimize_bounded(
    objective: Callable[[float], float],
    *,
    tolerance: float = 1.0e-3,
    max_iterations: int = 64,
) -> BoundedMinimum:
    """Deterministic golden-section minimization including both endpoints."""

    tol = float(tolerance)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    evaluations: list[tuple[float, float]] = []
    cache: dict[float, float] = {}

    def evaluate(alpha: float) -> float:
        key = float(alpha)
        if key not in cache:
            value = float(objective(key))
            if not math.isfinite(value):
                raise ValueError("bounded objective returned a non-finite value")
            cache[key] = value
            evaluations.append((key, value))
        return cache[key]

    evaluate(0.0)
    evaluate(1.0)
    low, high = 0.0, 1.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_value = evaluate(left)
    right_value = evaluate(right)
    for _ in range(int(max_iterations)):
        if high - low <= tol:
            break
        if left_value <= right_value:
            high = right
            right = left
            right_value = left_value
            left = high - ratio * (high - low)
            left_value = evaluate(left)
        else:
            low = left
            left = right
            left_value = right_value
            right = low + ratio * (high - low)
            right_value = evaluate(right)
    evaluate((low + high) / 2.0)
    minimum = min(value for _alpha, value in evaluations)
    tie_tolerance = max(1.0e-12, abs(minimum) * 1.0e-12)
    tied = [
        (alpha, value)
        for alpha, value in evaluations
        if abs(value - minimum) <= tie_tolerance
    ]
    alpha, value = max(tied, key=lambda item: item[0])
    return BoundedMinimum(float(alpha), float(value), tuple(evaluations))


def normalized_improvement(
    before: float, after: float, *, epsilon: float = 1.0e-12
) -> float:
    """Return the normalized relative MSE improvement used as raw reward."""

    denominator = float(before) + max(0.0, float(epsilon))
    if denominator <= 0.0:
        return 0.0
    return float((float(before) - float(after)) / denominator)


@dataclass
class ValidationSubset:
    features: list[list[float]] = field(default_factory=list)
    targets: list[float] = field(default_factory=list)
    sample_steps: list[int] = field(default_factory=list)
    samples_seen: int = 0
    quality: float = 0.0
    pairwise_distance_sum: float = 0.0

    def refresh_quality(self, feature_dim: int = 4) -> None:
        rows = (
            self.features
            if self.features
            else np.empty((0, max(4, int(feature_dim))), dtype=np.float32)
        )
        self.quality = validation_quality(rows)
        count = len(self.features)
        if count < 2:
            self.pairwise_distance_sum = 0.0
        elif count <= 512:
            # Q = n * mean_distance = 2 * pairwise_sum / (n - 1).
            self.pairwise_distance_sum = (
                float(self.quality) * float(count - 1) / 2.0
            )
        else:
            # Above the existing quality helper's 512-row cap, the score is
            # based on a deterministic subsample rather than the full sum.
            self.pairwise_distance_sum = float("nan")

    def _append_bounded(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        steps: np.ndarray,
        indices: list[int],
        *,
        capacity: int,
        reservoir_seed: int,
    ) -> None:
        retained = False
        for index in indices:
            row = np.asarray(features[index], dtype=np.float32).tolist()
            coordinate = np.asarray(row[:4], dtype=np.float64)
            target = float(np.asarray(targets[index]).reshape(-1)[0])
            sample_step = int(steps[index])
            ordinal = int(self.samples_seen)
            self.samples_seen += 1
            if len(self.features) < capacity:
                slot = len(self.features)
            else:
                seed = (
                    int(reservoir_seed)
                    + ordinal * 0x9E3779B97F4A7C15
                ) & 0xFFFFFFFFFFFFFFFF
                slot = int(
                    np.random.default_rng(seed).integers(0, ordinal + 1)
                )
                if slot >= capacity:
                    continue

            current = (
                _coordinate_rows(self.features)
                if self.features
                else np.empty((0, 4), dtype=np.float64)
            )
            if capacity <= 512 and math.isfinite(self.pairwise_distance_sum):
                if slot == len(self.features):
                    if len(current):
                        self.pairwise_distance_sum += float(
                            np.linalg.norm(current - coordinate, axis=1).sum()
                        )
                else:
                    others = np.delete(current, slot, axis=0)
                    if len(others):
                        self.pairwise_distance_sum += float(
                            np.linalg.norm(others - coordinate, axis=1).sum()
                            - np.linalg.norm(
                                others - current[slot], axis=1
                            ).sum()
                        )
            if slot == len(self.features):
                self.features.append(row)
                self.targets.append(target)
                self.sample_steps.append(sample_step)
            else:
                self.features[slot] = row
                self.targets[slot] = target
                self.sample_steps[slot] = sample_step
            retained = True

        if not retained:
            return
        count = len(self.features)
        if capacity <= 512 and math.isfinite(self.pairwise_distance_sum):
            self.quality = (
                float(count)
                if count < 2
                else float(
                    2.0 * self.pairwise_distance_sum / float(count - 1)
                )
            )
        else:
            self.refresh_quality(int(features.shape[1]))

    def append(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        steps: np.ndarray,
        indices: list[int],
        *,
        capacity: int | None = None,
        reservoir_seed: int = 0,
    ) -> None:
        limit = None if capacity is None else int(capacity)
        if limit is not None:
            if limit <= 0:
                raise ValueError("validation capacity must be positive")
            self._append_bounded(
                features,
                targets,
                steps,
                indices,
                capacity=limit,
                reservoir_seed=int(reservoir_seed),
            )
            return
        self.samples_seen += len(indices)
        old_count = len(self.features)
        old_coordinates = (
            _coordinate_rows(self.features)
            if self.features
            else np.empty((0, 4), dtype=np.float64)
        )
        appended_coordinates: list[np.ndarray] = []
        for index in indices:
            self.features.append(
                np.asarray(features[index], dtype=np.float32).tolist()
            )
            self.targets.append(float(np.asarray(targets[index]).reshape(-1)[0]))
            self.sample_steps.append(int(steps[index]))
            appended_coordinates.append(
                np.asarray(features[index], dtype=np.float64)[:4]
            )
        if indices:
            new_count = len(self.features)
            if new_count <= 512 and math.isfinite(self.pairwise_distance_sum):
                new_coordinates = np.asarray(
                    appended_coordinates, dtype=np.float64
                ).reshape(-1, 4)
                added_sum = 0.0
                if old_count:
                    added_sum += float(
                        np.linalg.norm(
                            new_coordinates[:, None, :]
                            - old_coordinates[None, :, :],
                            axis=2,
                        ).sum()
                    )
                if len(new_coordinates) > 1:
                    upper = np.triu_indices(len(new_coordinates), k=1)
                    pair_distances = np.linalg.norm(
                        new_coordinates[:, None, :]
                        - new_coordinates[None, :, :],
                        axis=2,
                    )
                    added_sum += float(pair_distances[upper].sum())
                self.pairwise_distance_sum += added_sum
                if new_count < 2:
                    self.quality = float(new_count)
                else:
                    self.quality = float(
                        2.0
                        * self.pairwise_distance_sum
                        / float(new_count - 1)
                    )
            else:
                self.refresh_quality(int(features.shape[1]))

    def snapshot(self) -> dict[str, object]:
        return {
            "features": [list(row) for row in self.features],
            "targets": list(self.targets),
            "sample_steps": list(self.sample_steps),
            "samples_seen": int(self.samples_seen),
            "quality": float(self.quality),
            "pairwise_distance_sum": float(self.pairwise_distance_sum),
        }

    @classmethod
    def restore(cls, raw: object, *, feature_dim: int) -> "ValidationSubset":
        data = raw if isinstance(raw, dict) else {}
        subset = cls(
            features=[list(row) for row in data.get("features", [])],
            targets=[float(value) for value in data.get("targets", [])],
            sample_steps=[int(value) for value in data.get("sample_steps", [])],
            samples_seen=max(
                0,
                int(data.get("samples_seen", len(data.get("features", [])))),
            ),
        )
        if not (
            len(subset.features)
            == len(subset.targets)
            == len(subset.sample_steps)
        ):
            raise ValueError(
                "cached validation features, targets, and timestamps differ"
            )
        # Reactivation intentionally recalculates quality from the restored
        # private rows; the cached scalar is audit data, not trusted state.
        subset.refresh_quality(feature_dim)
        return subset


@dataclass
class ZoneValidationState:
    samples_seen: int = 0
    optimization: ValidationSubset = field(default_factory=ValidationSubset)
    reward: ValidationSubset = field(default_factory=ValidationSubset)

    def snapshot(self) -> dict[str, object]:
        return {
            "samples_seen": int(self.samples_seen),
            "optimization": self.optimization.snapshot(),
            "reward": self.reward.snapshot(),
        }

    @classmethod
    def restore(cls, raw: object, *, feature_dim: int) -> "ZoneValidationState":
        data = raw if isinstance(raw, dict) else {}
        return cls(
            samples_seen=max(0, int(data.get("samples_seen", 0))),
            optimization=ValidationSubset.restore(
                data.get("optimization", {}), feature_dim=feature_dim
            ),
            reward=ValidationSubset.restore(
                data.get("reward", {}), feature_dim=feature_dim
            ),
        )


@dataclass(frozen=True)
class PullResult:
    valid: bool
    reason: str
    alpha: float | None = None
    objective_evaluations: int = 0
    before_loss: float | None = None
    after_loss: float | None = None
    reward: float | None = None
    adopted: bool = False
    joint_reward: float | None = None
    receiver_before_loss: float | None = None
    receiver_after_loss: float | None = None
    provider_before_loss: float | None = None
    provider_after_loss: float | None = None
    receiver_reward: float | None = None
    provider_reward: float | None = None
    receiver_information_gain: float | None = None
    provider_information_gain: float | None = None
    parameter_geometry_reward: float | None = None
    parameter_geometry_alpha: float | None = None
    receiver_adopted: bool = False
    provider_adopted: bool = False
    model_messages: int = 2
    scalar_loss_messages: int = 0
    scalar_control_messages: int = 0
    scalar_messages: int = 0


PULL_LOG_FIELDS = (
    "step",
    "mode",
    "receiver_idx",
    "provider_idx",
    "zone",
    "valid",
    "reason",
    "alpha",
    "objective_evaluations",
    "receiver_opt_quality",
    "provider_opt_quality",
    "receiver_reward_quality",
    "provider_reward_quality",
    "before_loss",
    "after_loss",
    "reward",
    "joint_reward",
    "receiver_before_loss",
    "receiver_after_loss",
    "provider_before_loss",
    "provider_after_loss",
    "receiver_reward",
    "provider_reward",
    "receiver_information_gain",
    "provider_information_gain",
    "parameter_geometry_reward",
    "parameter_geometry_alpha",
    "policy_reward",
    "adopted",
    "receiver_adopted",
    "provider_adopted",
    "metadata_bytes",
    "model_messages",
    "model_bytes",
    "scalar_loss_messages",
    "scalar_loss_bytes",
    "scalar_control_messages",
    "scalar_control_bytes",
    "scalar_messages",
    "scalar_bytes",
)


class BidirectionalCrossValidationMode(RewardMode):
    """Immediate contextual-bandit reward mode for the CV protocol."""

    def __init__(self, cfg, mode_id: str) -> None:
        super().__init__(cfg)
        self.id = str(mode_id)
        self.name = "Bidirectional private cross-validation"

    def on_encounter(
        self,
        sim: "BidirectionalCrossValidationSimulation",
        step: int,
        ns_i,
        ns_j,
        az: int,
        action: int,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
        j_view: Optional[object] = None,
    ) -> Optional[Transition]:
        sim._record_validation_metadata(self.id, ns_j, j_view=j_view)
        decision_key = (self.id, sim.node_idx(ns_i))
        if int(action) == 0:
            sim._last_protocol_valid[decision_key] = True
            return (state, 0, 0.0, next_state, bool(done), sim.node_idx(ns_i))
        result = sim._execute_validation_pull(
            step=int(step),
            mode=self.id,
            receiver=ns_i,
            provider=ns_j,
            zone=int(az),
            provider_view=j_view,
        )
        sim._last_protocol_valid[decision_key] = bool(result.valid)
        raw_reward = 0.0 if result.reward is None else float(result.reward)
        return (
            state,
            1,
            raw_reward - float(sim.communication_penalty),
            next_state,
            bool(done),
            sim.node_idx(ns_i),
        )


class BidirectionalCrossValidationSimulation(BootstrapPolicySharingSimulation):
    """zRAMP method implementing decentralized cross-validation pulls."""

    policy_transfer_rule = "optional-synchronous-all-feasible-policy-average"

    def __init__(
        self,
        *args,
        communication_penalty: float = 0.0,
        aggregation_tolerance: float = 1.0e-3,
        aggregation_max_iterations: int = 64,
        validation_epsilon: float = 1.0e-12,
        validation_capacity: int | None = None,
        share_policy_every_contact: bool = False,
        symmetric_predictor_pull: bool = False,
        pull_reward_metric: str = "normalized-improvement",
        random_pull_probability: float | None = None,
        policy_temperature: float = 1.0,
        hard_warmup_steps: int = 0,
        hard_warmup_pull_probability: float = 0.5,
        train_all_observations: bool = False,
        **kwargs,
    ) -> None:
        penalty = float(communication_penalty)
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(
                "communication_penalty must be finite and non-negative"
            )
        if float(aggregation_tolerance) <= 0.0:
            raise ValueError("aggregation_tolerance must be positive")
        if int(aggregation_max_iterations) <= 0:
            raise ValueError("aggregation_max_iterations must be positive")
        if float(validation_epsilon) < 0.0:
            raise ValueError("validation_epsilon must be non-negative")
        reward_metric = str(pull_reward_metric).strip().lower()
        if reward_metric not in {"normalized-improvement", "rmse-gain"}:
            raise ValueError(
                "pull_reward_metric must be normalized-improvement or rmse-gain"
            )
        capacity = None if validation_capacity is None else int(validation_capacity)
        if capacity is not None and capacity < 2:
            raise ValueError("validation_capacity must be at least two")
        random_probability = (
            None
            if random_pull_probability is None
            else float(random_pull_probability)
        )
        if random_probability is not None and not 0.0 <= random_probability <= 1.0:
            raise ValueError("random_pull_probability must be in [0, 1]")
        temperature = float(policy_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("policy_temperature must be finite and positive")
        warmup_steps = int(hard_warmup_steps)
        if warmup_steps < 0:
            raise ValueError("hard_warmup_steps must be non-negative")
        warmup_probability = float(hard_warmup_pull_probability)
        if (
            not math.isfinite(warmup_probability)
            or not 0.0 <= warmup_probability <= 1.0
        ):
            raise ValueError("hard_warmup_pull_probability must be in [0, 1]")

        cfg = args[0] if args else kwargs.get("cfg")
        node_count = int(getattr(cfg, "num_nodes", 0))
        self.communication_penalty = penalty
        self.aggregation_tolerance = float(aggregation_tolerance)
        self.aggregation_max_iterations = int(aggregation_max_iterations)
        self.validation_epsilon = float(validation_epsilon)
        self.validation_capacity = capacity
        self.validation_optimization_capacity = (
            None if capacity is None else capacity // 2
        )
        self.validation_reward_capacity = (
            None if capacity is None else capacity - capacity // 2
        )
        self.random_pull_probability = random_probability
        self.policy_temperature = temperature
        self.hard_warmup_steps = warmup_steps
        self.hard_warmup_pull_probability = warmup_probability
        self.train_all_observations = bool(train_all_observations)
        self.share_policy_every_contact = bool(share_policy_every_contact)
        self.symmetric_predictor_pull = bool(symmetric_predictor_pull)
        self.pull_reward_metric = reward_metric
        self._zone_validation: list[ZoneValidationState] = [
            ZoneValidationState() for _ in range(node_count)
        ]
        self._cv_step_metadata_bytes: Counter[str] = Counter()
        self._cv_step_model_bytes: Counter[str] = Counter()
        self._cv_step_scalar_bytes: Counter[str] = Counter()
        self._cv_step_model_messages: Counter[str] = Counter()
        self._last_exploratory: dict[tuple[str, int], int] = {}
        self._cv_step_scalar_messages: Counter[str] = Counter()
        self._cv_step_scalar_loss_messages: Counter[str] = Counter()
        self._cv_step_scalar_control_messages: Counter[str] = Counter()
        self._cv_step_pulls: Counter[str] = Counter()
        self._cv_step_valid_pulls: Counter[str] = Counter()
        self._cv_step_policy_bytes: Counter[str] = Counter()
        self._cv_step_policy_messages: Counter[str] = Counter()
        self._cv_step_sample_bytes: Counter[str] = Counter()
        self._cv_step_sample_messages: Counter[str] = Counter()
        self._cv_receiver_aggregations: Counter[tuple[str, int]] = Counter()
        self._cv_last_provider_pull_step: dict[
            tuple[str, int, int, int], int
        ] = {}
        self._cv_log_file = None
        self._cv_log_writer: csv.DictWriter | None = None
        self._cv_log_count = 0
        self._last_predicted_gain: dict[tuple[str, int], float] = {}
        self._last_protocol_valid: dict[tuple[str, int], bool] = {}

        kwargs["zone_model_memory"] = True
        kwargs["local_policy_share"] = False
        kwargs["policy_temperature"] = temperature
        kwargs.setdefault("aux_baselines", "none")
        super().__init__(*args, **kwargs)
        # Predictor-pull-triggered sharing stays disabled; this public flag
        # reports the dedicated synchronous all-contact exchange configured
        # above in progress/config outputs.
        self.local_policy_share = self.share_policy_every_contact
        if len(self._zone_validation) != len(self.nodes):
            self._zone_validation = [ZoneValidationState() for _ in self.nodes]
        self.reward_modes = {
            mode_id: BidirectionalCrossValidationMode(self.cfg, mode_id)
            for mode_id in self.agents
        }
        self._cv_eval_model = self._make_predictor().to(self.device)
        self._cv_eval_model.eval()
        action_policy = str(self.cfg.rl_action_policy).strip().lower()
        self.zramp_policy_mode = (
            f"bidirectional-private-cv-net-reward-{action_policy}"
        )
        if action_policy == "softmax":
            self.local_policy_initial_pull = (
                "net-reward-temperature-softmax-from-first-contact"
            )
            self.local_policy_initial_pull_probability = 0.5
        elif action_policy == "argmax":
            if self.hard_warmup_steps > 0:
                self.local_policy_initial_pull = (
                    "random-warmup-then-positive-net-reward"
                )
                self.local_policy_initial_pull_probability = float(
                    self.hard_warmup_pull_probability
                )
            else:
                self.local_policy_initial_pull = "positive-net-reward-only"
                self.local_policy_initial_pull_probability = 0.0
        elif action_policy in {"accept", "always_accept"}:
            self.local_policy_initial_pull = "always-pull"
            self.local_policy_initial_pull_probability = 1.0
        else:
            self.local_policy_initial_pull = "always-pass"
            self.local_policy_initial_pull_probability = 0.0
        if self.random_pull_probability is not None:
            self.zramp_policy_mode = "bidirectional-private-cv-random"
            self.local_policy_initial_pull = "fixed-random-from-first-contact"
            self.local_policy_initial_pull_probability = float(
                self.random_pull_probability
            )
        self._cv_log_path = (
            Path(self.cfg.results_dir) / "cross_validation_pulls.csv"
        )
        self._communication_assumptions = self._build_communication_assumptions()

    # --------------------------------------------------------- zone lifecycle

    def _snapshot_variant(self, variant) -> dict[str, object]:
        snapshot = super()._snapshot_variant(variant)
        snapshot.update(
            {
                "m_samples": int(variant.m_samples),
                "n_samples": int(variant.n_samples),
                "quality": float(variant.quality),
                "t_wait": int(variant.t_wait),
                "last_rmse": float(variant.last_rmse),
                "last_rmse_available": bool(variant.last_rmse_available),
                "rmse_ema_short": float(variant.rmse_ema_short),
                "rmse_ema_long": float(variant.rmse_ema_long),
                "rmse_batches": int(variant.rmse_batches),
                "model_signature": variant.model_signature.detach().cpu().clone(),
                "step_saved": int(getattr(self, "_current_sumo_step", 0)),
            }
        )
        return snapshot

    def _restore_variant(self, variant, snapshot: dict[str, object]) -> None:
        if "m_samples" not in snapshot:
            super()._restore_variant(variant, snapshot)
            return
        self._load_model_state(
            variant.model, snapshot["weights"]  # type: ignore[arg-type]
        )
        variant.opt = optim.Adam(
            variant.model.parameters(), lr=self.cfg.local_lr
        )
        variant.m_samples = int(snapshot["m_samples"])
        variant.n_samples = int(snapshot["n_samples"])
        variant.quality = float(snapshot["quality"])
        elapsed = max(
            0,
            int(getattr(self, "_current_sumo_step", 0))
            - int(snapshot.get("step_saved", 0)),
        )
        variant.t_wait = max(0, int(snapshot["t_wait"]) + elapsed)
        variant.last_rmse = float(snapshot["last_rmse"])
        variant.last_rmse_available = bool(snapshot["last_rmse_available"])
        variant.rmse_ema_short = float(snapshot["rmse_ema_short"])
        variant.rmse_ema_long = float(snapshot["rmse_ema_long"])
        variant.rmse_batches = int(snapshot["rmse_batches"])
        variant.model_signature = (  # type: ignore[union-attr]
            snapshot["model_signature"].detach().cpu().clone()
        )
        variant.recovery_steps_left = 0
        variant.recovery_accepts_left = 0
        variant.recovery_cooldown_left = 0

    def _save_node_zone_memory(self, node_idx: int, zone: int) -> None:
        super()._save_node_zone_memory(node_idx, zone)
        if not self.zone_model_memory:
            return
        ns = self.nodes[int(node_idx)]
        cached = self._node_zone_memory[int(node_idx)][int(zone)]
        cached.update(
            {
                "train_features": [
                    list(row) for row in ns.current_visit_samples_x
                ],
                "train_targets": list(ns.current_visit_samples_y),
                "train_sample_steps": list(ns.current_visit_sample_steps),
                "validation": self._zone_validation[int(node_idx)].snapshot(),
            }
        )

    def _restore_node_zone_memory(self, node_idx: int, zone: int) -> bool:
        restored = super()._restore_node_zone_memory(node_idx, zone)
        if not restored:
            return False
        cached = self._node_zone_memory[int(node_idx)][int(zone)]
        ns = self.nodes[int(node_idx)]
        ns.current_visit_samples_x.extend(
            list(row) for row in cached.get("train_features", [])
        )
        ns.current_visit_samples_y.extend(
            float(value) for value in cached.get("train_targets", [])
        )
        ns.current_visit_sample_steps.extend(
            int(value) for value in cached.get("train_sample_steps", [])
        )
        if not (
            len(ns.current_visit_samples_x)
            == len(ns.current_visit_samples_y)
            == len(ns.current_visit_sample_steps)
        ):
            raise ValueError(
                "cached training features, targets, and timestamps differ"
            )
        self._zone_validation[int(node_idx)] = ZoneValidationState.restore(
            cached.get("validation", {}),
            feature_dim=self._predictor_input_dim(),
        )
        return True

    def _reset_node_for_zone_change(self, ns, new_az: int) -> None:
        node_idx = next(
            (
                idx
                for idx, candidate in enumerate(getattr(self, "nodes", []))
                if candidate is ns
            ),
            -1,
        )
        cached = bool(
            node_idx >= 0
            and hasattr(self, "_node_zone_memory")
            and int(new_az) in self._node_zone_memory[node_idx]
        )
        super()._reset_node_for_zone_change(ns, new_az)
        if node_idx >= 0 and not cached:
            self._zone_validation[node_idx] = ZoneValidationState()

    # ----------------------------------------------------------- sample split

    def _train_local(
        self,
        ns,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        sample_count_increment: int | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        del sample_weights
        node_idx = self.node_idx(ns)
        n_new = (
            int(X.shape[0])
            if sample_count_increment is None
            else max(0, int(sample_count_increment))
        )
        if n_new > len(ns.current_visit_samples_x):
            raise ValueError("new-sample count exceeds the current zone buffer")
        state = self._zone_validation[node_idx]
        if n_new > 0 and bool(getattr(self, "train_all_observations", False)):
            # Retain every observation for predictor training while also
            # keeping small, disjoint optimization/reward reservoirs. The
            # reservoir rows are copies, not withheld holdouts.
            start = len(ns.current_visit_samples_x) - n_new
            new_X = np.asarray(
                ns.current_visit_samples_x[start:], dtype=np.float32
            )
            new_y = np.asarray(
                ns.current_visit_samples_y[start:], dtype=np.float32
            ).reshape(-1, 1)
            new_steps = np.asarray(
                ns.current_visit_sample_steps[start:], dtype=np.int64
            )
            _n_train, n_opt, n_reward = validation_allocation_counts(
                state.samples_seen, n_new
            )
            empty = np.empty((0, int(new_X.shape[1])), dtype=np.float32)
            opt_existing = (
                np.asarray(state.optimization.features, dtype=np.float32)
                if state.optimization.features
                else empty
            )
            reward_existing = (
                np.asarray(state.reward.features, dtype=np.float32)
                if state.reward.features
                else empty
            )
            _unused_train, opt_indices, reward_indices = (
                self._select_validation_indices(
                    new_X,
                    opt_existing,
                    reward_existing,
                    n_opt,
                    n_reward,
                    node_idx=node_idx,
                    samples_seen=state.samples_seen,
                )
            )
            reservoir_base_seed = (
                int(getattr(getattr(self, "cfg", None), "seed", 0))
                * 1_000_003
                + int(node_idx) * 97_409
            )
            state.optimization.append(
                new_X,
                new_y,
                new_steps,
                opt_indices,
                capacity=getattr(
                    self, "validation_optimization_capacity", None
                ),
                reservoir_seed=reservoir_base_seed + 11,
            )
            state.reward.append(
                new_X,
                new_y,
                new_steps,
                reward_indices,
                capacity=getattr(
                    self, "validation_reward_capacity", None
                ),
                reservoir_seed=reservoir_base_seed + 23,
            )
            state.samples_seen += n_new
            n_train = n_new
        elif n_new > 0:
            start = len(ns.current_visit_samples_x) - n_new
            new_X = np.asarray(
                ns.current_visit_samples_x[start:], dtype=np.float32
            )
            new_y = np.asarray(
                ns.current_visit_samples_y[start:], dtype=np.float32
            ).reshape(-1, 1)
            new_steps = np.asarray(
                ns.current_visit_sample_steps[start:], dtype=np.int64
            )
            expected_train, n_opt, n_reward = validation_allocation_counts(
                state.samples_seen, n_new
            )
            empty = np.empty(
                (0, int(new_X.shape[1])), dtype=np.float32
            )
            opt_existing = (
                np.asarray(state.optimization.features, dtype=np.float32)
                if state.optimization.features
                else empty
            )
            reward_existing = (
                np.asarray(state.reward.features, dtype=np.float32)
                if state.reward.features
                else empty
            )
            train_indices, opt_indices, reward_indices = (
                self._select_validation_indices(
                    new_X,
                    opt_existing,
                    reward_existing,
                    n_opt,
                    n_reward,
                    node_idx=node_idx,
                    samples_seen=state.samples_seen,
                )
            )
            if len(train_indices) != expected_train:
                raise RuntimeError("validation allocation count changed")
            reservoir_base_seed = (
                int(getattr(getattr(self, "cfg", None), "seed", 0))
                * 1_000_003
                + int(node_idx) * 97_409
            )
            state.optimization.append(
                new_X,
                new_y,
                new_steps,
                opt_indices,
                capacity=getattr(
                    self, "validation_optimization_capacity", None
                ),
                reservoir_seed=reservoir_base_seed + 11,
            )
            state.reward.append(
                new_X,
                new_y,
                new_steps,
                reward_indices,
                capacity=getattr(
                    self, "validation_reward_capacity", None
                ),
                reservoir_seed=reservoir_base_seed + 23,
            )
            state.samples_seen += n_new

            del ns.current_visit_samples_x[start:]
            del ns.current_visit_samples_y[start:]
            del ns.current_visit_sample_steps[start:]
            for index in train_indices:
                ns.current_visit_samples_x.append(new_X[index].tolist())
                ns.current_visit_samples_y.append(float(new_y[index, 0]))
                ns.current_visit_sample_steps.append(int(new_steps[index]))
            n_train = len(train_indices)
        else:
            n_train = 0

        feature_dim = self._predictor_input_dim()
        train_X = (
            np.asarray(ns.current_visit_samples_x, dtype=np.float32)
            if ns.current_visit_samples_x
            else np.empty((0, feature_dim), dtype=np.float32)
        )
        train_y = np.asarray(
            ns.current_visit_samples_y, dtype=np.float32
        ).reshape(-1, 1)
        train_steps = np.asarray(
            ns.current_visit_sample_steps, dtype=np.float32
        )
        weights = self._sample_recency_weights(
            train_steps,
            current_step=int(getattr(self, "_current_sumo_step", 0)),
        )
        augment = getattr(self, "_augment_predictor_training_arrays", None)
        if callable(augment):
            train_X, train_y, train_steps, weights = augment(
                node_idx=int(node_idx),
                train_X=train_X,
                train_y=train_y,
                train_steps=train_steps,
                sample_weights=weights,
            )
            train_X = np.asarray(train_X, dtype=np.float32)
            train_y = np.asarray(train_y, dtype=np.float32).reshape(-1, 1)
            train_steps = np.asarray(train_steps, dtype=np.float32).reshape(-1)
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            row_count = int(train_X.shape[0])
            if not (
                int(train_y.shape[0])
                == int(train_steps.shape[0])
                == int(weights.shape[0])
                == row_count
            ):
                raise ValueError(
                    "augmented predictor training arrays have different lengths"
                )
        super()._train_local(
            ns,
            train_X,
            train_y,
            sample_count_increment=n_train,
            sample_weights=weights,
        )

    def _select_validation_indices(
        self,
        batch_features: np.ndarray,
        opt_existing: np.ndarray,
        reward_existing: np.ndarray,
        n_opt: int,
        n_reward: int,
        *,
        node_idx: int,
        samples_seen: int,
    ) -> tuple[list[int], list[int], list[int]]:
        del node_idx, samples_seen
        return select_validation_indices(
            batch_features,
            opt_existing,
            reward_existing,
            n_opt,
            n_reward,
        )

    # ------------------------------------------------------------ metadata/Q

    def _metadata_for(self, node_idx: int, mode: str):
        validation = self._zone_validation[int(node_idx)]
        return build_metadata(
            self.nodes[int(node_idx)].variants[mode],
            share_model_age=(
                "relative_provider_freshness" in self.policy_state_features
            ),
            validation_opt_quality=validation.optimization.quality,
            validation_reward_quality=validation.reward.quality,
            share_validation_quality=True,
        )

    def _make_peer_view(self, ns_j, mode: str):
        view = super()._make_peer_view(ns_j, mode)
        node_idx = self.node_idx(ns_j)
        if node_idx >= 0:
            metadata = self._metadata_for(node_idx, mode)
            view._validation_opt_quality = (  # type: ignore[attr-defined]
                metadata.validation_opt_quality
            )
            view._validation_reward_quality = (  # type: ignore[attr-defined]
                metadata.validation_reward_quality
            )
            view._validation_metadata_wire_nbytes = (  # type: ignore[attr-defined]
                metadata.wire_nbytes
            )
            view._cv_metadata = metadata  # type: ignore[attr-defined]
        return view

    def _steps_since_provider_pull(
        self,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        zone: int,
    ) -> int | None:
        key = (str(mode), int(receiver_idx), int(provider_idx), int(zone))
        previous = self._cv_last_provider_pull_step.get(key)
        if previous is None:
            return None
        return max(0, int(getattr(self, "_current_sumo_step", 0)) - previous)

    def _state_features(
        self,
        mode: str,
        ns_i,
        ns_j,
        az: int,
        neighbor_count: int,
        j_view: Optional[object] = None,
    ) -> torch.Tensor:
        receiver_idx = self.node_idx(ns_i)
        provider_idx = self.node_idx(ns_j)
        receiver_metadata = self._metadata_for(receiver_idx, mode)
        provider_metadata = (
            getattr(j_view, "_cv_metadata")
            if j_view is not None and hasattr(j_view, "_cv_metadata")
            else self._metadata_for(provider_idx, mode)
        )
        return build_observation(
            receiver_metadata,
            provider_metadata,
            neighbor_count=int(neighbor_count),
            zone_neighbor_count=int(neighbor_count),
            zone_buffer_samples=len(ns_i.current_visit_samples_x),
            steps_since_provider_pull=self._steps_since_provider_pull(
                mode, receiver_idx, provider_idx, int(az)
            ),
            receiver_aggregations_this_step=int(
                self._cv_receiver_aggregations[(str(mode), receiver_idx)]
            ),
            feature_names=tuple(self.policy_state_features),
        )

    def _record_validation_metadata(
        self, mode: str, provider, *, j_view=None
    ) -> None:
        if (
            j_view is not None
            and hasattr(j_view, "_validation_metadata_wire_nbytes")
        ):
            wire_bytes = int(j_view._validation_metadata_wire_nbytes)
        else:
            provider_idx = self.node_idx(provider)
            wire_bytes = int(self._metadata_for(provider_idx, mode).wire_nbytes)
        self._cv_step_metadata_bytes[mode] += wire_bytes

    def _normalized_contact_links(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None,
    ) -> list[tuple[int, int, int]]:
        if contact_links is None:
            return [
                (int(zone), int(a), int(b))
                for zone, indices in zone_nodes.items()
                for offset, a in enumerate(sorted(indices))
                for b in sorted(indices)[offset + 1 :]
            ]
        unique: set[tuple[int, int, int]] = set()
        for zone, a, b in contact_links:
            ia, ib = sorted((int(a), int(b)))
            if ia == ib or not (
                0 <= ia < len(self.nodes) and 0 <= ib < len(self.nodes)
            ):
                continue
            if int(self.nodes[ia].current_az) != int(
                self.nodes[ib].current_az
            ):
                continue
            unique.add((int(zone), ia, ib))
        return sorted(unique)

    def _share_policies_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> None:
        if not self.share_policy_every_contact or not links:
            return
        neighbors: dict[int, list[int]] = defaultdict(list)
        for _zone, a, b in links:
            neighbors[int(a)].append(int(b))
            neighbors[int(b)].append(int(a))
        neighbors = {
            receiver: sorted(set(providers))
            for receiver, providers in neighbors.items()
        }

        states = {
            mode: [self._clone_state(agent.policy) for agent in agents]
            for mode, agents in self.local_agents.items()
        }
        experiences = {
            mode: [self._policy_experience(mode, idx) for idx in range(len(agents))]
            for mode, agents in self.local_agents.items()
        }
        versions = {
            mode: list(self._local_policy_versions[mode])
            for mode in self.local_agents
        }
        for mode, agents in self.local_agents.items():
            for receiver_idx in sorted(neighbors):
                provider_ids = neighbors[receiver_idx]
                source_ids = [receiver_idx, *provider_ids]
                shared_state, _weights = aggregate_policy_states(
                    [states[mode][idx] for idx in source_ids],
                    [experiences[mode][idx] for idx in source_ids],
                )
                agents[receiver_idx].policy.load_state_dict(shared_state)
                self._local_policy_versions[mode][receiver_idx] = (
                    max(versions[mode][idx] for idx in source_ids) + 1
                )
                transferred = sum(
                    self._state_nbytes(states[mode][idx])
                    for idx in provider_ids
                )
                self._cv_step_policy_bytes[mode] += int(transferred)
                self._cv_step_policy_messages[mode] += len(provider_ids)
                self._local_policy_pull_updates[mode] += len(provider_ids)
                self._last_local_policy_pull_updates += len(provider_ids)
                self._last_local_policy_pull_updates_by_mode[mode] += len(
                    provider_ids
                )

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        links = self._normalized_contact_links(zone_nodes, contact_links)
        self._share_policies_with_all_feasible_neighbors(links)
        super()._gossip_step(step, zone_nodes, contact_links=links)

    @staticmethod
    def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    @staticmethod
    def _state_nbytes(state: TensorState) -> int:
        return int(
            sum(
                int(value.numel()) * int(value.element_size())
                for value in state.values()
            )
        )

    def _subset_arrays(
        self, subset: ValidationSubset
    ) -> tuple[np.ndarray, np.ndarray]:
        if not subset.features:
            return (
                np.empty(
                    (0, self._predictor_input_dim()), dtype=np.float32
                ),
                np.empty((0, 1), dtype=np.float32),
            )
        return (
            np.asarray(subset.features, dtype=np.float32),
            np.asarray(subset.targets, dtype=np.float32).reshape(-1, 1),
        )

    def _private_mse(
        self, mode: str, state: TensorState, subset: ValidationSubset
    ) -> float:
        del mode
        if subset.quality <= 0.0 or not subset.features:
            return 0.0
        prepared = self._prepare_validation_pair(
            subset,
            ValidationSubset(),
            quality_a=float(subset.quality),
            quality_b=0.0,
        )
        loss, _unused = self._pair_mses(state, prepared)
        return float(loss)

    def _prepare_validation_pair(
        self,
        subset_a: ValidationSubset,
        subset_b: ValidationSubset,
        *,
        quality_a: float,
        quality_b: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Materialize two private subsets once for repeated state tests."""

        arrays: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        counts: list[int] = []
        for subset, quality in (
            (subset_a, float(quality_a)),
            (subset_b, float(quality_b)),
        ):
            if quality > 0.0 and subset.features:
                X, y_rssi = self._subset_arrays(subset)
            else:
                X = np.empty(
                    (0, self._predictor_input_dim()), dtype=np.float32
                )
                y_rssi = np.empty((0, 1), dtype=np.float32)
            arrays.append(X)
            targets.append(y_rssi)
            counts.append(int(X.shape[0]))
        if sum(counts) == 0:
            return (
                torch.empty(
                    (0, self._predictor_input_dim()),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                counts[0],
                counts[1],
            )
        features = torch.as_tensor(
            np.concatenate(arrays, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        y_rssi = np.concatenate(targets, axis=0)
        target_loss = torch.as_tensor(
            self._rssi_to_loss_db(y_rssi).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        return features, target_loss, counts[0], counts[1]

    def _pair_mses(
        self,
        state: TensorState,
        prepared: tuple[torch.Tensor, torch.Tensor, int, int],
    ) -> tuple[float, float]:
        """Evaluate one candidate once over concatenated A/B private rows."""

        features, target_loss, count_a, count_b = prepared
        if int(features.shape[0]) == 0:
            return 0.0, 0.0
        self._load_model_state(self._cv_eval_model, dict(state))
        self._cv_eval_model.eval()
        lo = float(self._loss_min_db())
        hi = float(self._loss_max_db())
        with torch.inference_mode():
            prediction = self._cv_eval_model(features).reshape(-1)
            prediction_loss = torch.clamp(
                prediction * (hi - lo) + lo,
                min=lo,
                max=hi,
            )
            squared = torch.square(prediction_loss - target_loss)
            loss_a = (
                float(squared[:count_a].mean().item())
                if count_a > 0
                else 0.0
            )
            loss_b = (
                float(squared[count_a : count_a + count_b].mean().item())
                if count_b > 0
                else 0.0
            )
        return loss_a, loss_b

    def _spike_recovery_settings(
        self, mode: str
    ) -> tuple[bool, dict[str, float | int]]:
        _enabled, settings = super()._spike_recovery_settings(mode)
        return False, settings

    # --------------------------------------------------------------- protocol

    def _execute_validation_pull(
        self,
        *,
        step: int,
        mode: str,
        receiver,
        provider,
        zone: int,
        provider_view: Optional[object],
        diagnostic: bool = False,
    ) -> PullResult:
        receiver_idx = self.node_idx(receiver)
        provider_idx = self.node_idx(provider)
        write_pull_log = (
            (lambda *_args, **_kwargs: None)
            if diagnostic
            else self._write_pull_log
        )
        if not hasattr(self, "_cv_last_provider_pull_step"):
            self._cv_last_provider_pull_step = {}
        if not diagnostic:
            self._cv_last_provider_pull_step[
                (str(mode), receiver_idx, provider_idx, int(zone))
            ] = int(step)
            if bool(getattr(self, "symmetric_predictor_pull", False)):
                self._cv_last_provider_pull_step[
                    (str(mode), provider_idx, receiver_idx, int(zone))
                ] = int(step)
        receiver_validation = self._zone_validation[receiver_idx]
        provider_validation = self._zone_validation[provider_idx]
        receiver_variant = receiver.variants[mode]
        provider_variant = provider.variants[mode]
        state_a = self._clone_state(receiver_variant.model)
        state_b = (
            {
                name: value.detach().cpu().clone()
                for name, value in provider_view._model_state.items()  # type: ignore[attr-defined]
            }
            if provider_view is not None
            else self._clone_state(provider.variants[mode].model)
        )

        symmetric_pull = bool(
            getattr(self, "symmetric_predictor_pull", False)
        )
        mergeable_evidence = bool(
            is_mergeable_evidence_state(state_a)
            and is_mergeable_evidence_state(state_b)
        )
        max_delta_rows = max(
            0, int(getattr(self, "mergeable_max_delta_rows", 0))
        )
        if mergeable_evidence:
            model_bytes = mergeable_evidence_direction_nbytes(
                state_a, state_b, max_rows=max_delta_rows
            )
            model_bytes += mergeable_evidence_direction_nbytes(
                state_b, state_a, max_rows=max_delta_rows
            )
        else:
            model_bytes = self._state_nbytes(state_a) + self._state_nbytes(
                state_b
            )
        if not diagnostic:
            self._cv_step_pulls[mode] += 1
            self._cv_step_model_messages[mode] += 2
            self._cv_step_model_bytes[mode] += model_bytes

        qa_opt = self._validation_subset_weight(
            receiver_validation.optimization
        )
        qb_opt = self._validation_subset_weight(
            provider_validation.optimization
        )
        qa_reward = self._validation_subset_weight(
            receiver_validation.reward
        )
        qb_reward = self._validation_subset_weight(
            provider_validation.reward
        )
        metadata_bytes = int(self._metadata_for(provider_idx, mode).wire_nbytes)

        if qa_opt + qb_opt <= 0.0:
            result = PullResult(
                valid=False, reason="zero_optimization_quality"
            )
            write_pull_log(
                step,
                mode,
                receiver_idx,
                provider_idx,
                zone,
                result,
                qa_opt,
                qb_opt,
                qa_reward,
                qb_reward,
                metadata_bytes,
                model_bytes,
            )
            return result
        if qa_reward + qb_reward <= 0.0:
            result = PullResult(valid=False, reason="zero_reward_quality")
            write_pull_log(
                step,
                mode,
                receiver_idx,
                provider_idx,
                zone,
                result,
                qa_opt,
                qb_opt,
                qa_reward,
                qb_reward,
                metadata_bytes,
                model_bytes,
            )
            return result

        optimization_pair = self._prepare_validation_pair(
            receiver_validation.optimization,
            provider_validation.optimization,
            quality_a=qa_opt,
            quality_b=qb_opt,
        )
        reward_pair = self._prepare_validation_pair(
            receiver_validation.reward,
            provider_validation.reward,
            quality_a=qa_reward,
            quality_b=qb_reward,
        )

        if mergeable_evidence:
            provider_delta = mergeable_evidence_delta_state(
                state_a, state_b, max_rows=max_delta_rows
            )
            receiver_delta = mergeable_evidence_delta_state(
                state_b, state_a, max_rows=max_delta_rows
            )
            basis_dim = max(1, int(state_a["weight"].numel()))
            receiver_information_gain = (
                mergeable_evidence_diagonal_information_gain(
                    state_a, provider_delta
                )
                / basis_dim
            )
            provider_information_gain = (
                mergeable_evidence_diagonal_information_gain(
                    state_b, receiver_delta
                )
                / basis_dim
            )
            aggregate_a = mergeable_evidence_union_states(
                state_a, provider_delta
            )
            aggregate_b = (
                mergeable_evidence_union_states(state_b, receiver_delta)
                if symmetric_pull
                else state_b
            )
            loss_a, _unused = self._pair_mses(
                aggregate_a, optimization_pair
            )
            _unused, loss_b = self._pair_mses(
                aggregate_b, optimization_pair
            )
            combined = quality_weighted_loss(
                loss_a,
                qa_opt,
                loss_b,
                qb_opt,
                epsilon=self.validation_epsilon,
            )
            if combined is None:
                raise RuntimeError("optimization validation quality vanished")
            optimum = BoundedMinimum(
                alpha=0.5,
                value=float(combined),
                evaluations=((0.5, float(combined)),),
            )
        else:
            receiver_information_gain = None
            provider_information_gain = None
            def objective(alpha: float) -> float:
                candidate = interpolate_states(state_a, state_b, alpha)
                loss_a, loss_b = self._pair_mses(
                    candidate, optimization_pair
                )
                combined = quality_weighted_loss(
                    loss_a,
                    qa_opt,
                    loss_b,
                    qb_opt,
                    epsilon=self.validation_epsilon,
                )
                if combined is None:
                    raise RuntimeError(
                        "optimization validation quality vanished"
                    )
                return combined

            alpha_grid_size = int(
                getattr(self, "aggregation_alpha_grid_size", 0)
            )
            if alpha_grid_size >= 2:
                evaluations = tuple(
                    (float(alpha), float(objective(float(alpha))))
                    for alpha in np.linspace(0.0, 1.0, alpha_grid_size)
                )
                best_alpha, best_value = min(
                    evaluations,
                    key=lambda row: (float(row[1]), float(row[0])),
                )
                optimum = BoundedMinimum(
                    alpha=float(best_alpha),
                    value=float(best_value),
                    evaluations=evaluations,
                )
            else:
                optimum = minimize_bounded(
                    objective,
                    tolerance=self.aggregation_tolerance,
                    max_iterations=self.aggregation_max_iterations,
                )
            aggregate = interpolate_states(
                state_a, state_b, optimum.alpha
            )
            aggregate_a = aggregate
            aggregate_b = aggregate if symmetric_pull else state_b
        if symmetric_pull:
            # Compare each endpoint against the predictor it owned before the
            # exchange. The legacy directional protocol instead uses the
            # receiver predictor as the baseline on both private holdouts.
            before_a, _unused = self._pair_mses(state_a, reward_pair)
            _unused, before_b = self._pair_mses(state_b, reward_pair)
        else:
            before_a, before_b = self._pair_mses(state_a, reward_pair)
        after_a, _unused = self._pair_mses(aggregate_a, reward_pair)
        _unused, after_b = self._pair_mses(aggregate_b, reward_pair)
        before = quality_weighted_loss(
            before_a,
            qa_reward,
            before_b,
            qb_reward,
            epsilon=self.validation_epsilon,
        )
        after = quality_weighted_loss(
            after_a,
            qa_reward,
            after_b,
            qb_reward,
            epsilon=self.validation_epsilon,
        )
        if before is None or after is None:
            raise RuntimeError("reward validation quality vanished")
        reward_metric = str(
            getattr(self, "pull_reward_metric", "normalized-improvement")
        )
        if reward_metric == "rmse-gain":
            joint_reward = math.sqrt(max(float(before), 0.0)) - math.sqrt(
                max(float(after), 0.0)
            )
            receiver_reward = (
                math.sqrt(max(float(before_a), 0.0))
                - math.sqrt(max(float(after_a), 0.0))
                if qa_reward > 0.0
                else None
            )
            provider_reward = (
                math.sqrt(max(float(before_b), 0.0))
                - math.sqrt(max(float(after_b), 0.0))
                if qb_reward > 0.0
                else None
            )
        else:
            joint_reward = normalized_improvement(
                before, after, epsilon=self.validation_epsilon
            )
            receiver_reward = (
                normalized_improvement(
                    before_a, after_a, epsilon=self.validation_epsilon
                )
                if qa_reward > 0.0
                else None
            )
            provider_reward = (
                normalized_improvement(
                    before_b, after_b, epsilon=self.validation_epsilon
                )
                if qb_reward > 0.0
                else None
            )
        receiver_adopted = bool(
            qa_reward > 0.0 and float(after_a) < float(before_a)
        )
        provider_adopted = bool(
            symmetric_pull
            and qb_reward > 0.0
            and float(after_b) < float(before_b)
        )
        unconditional_union = bool(
            mergeable_evidence
            and getattr(self, "unconditional_evidence_union", False)
        )
        if unconditional_union:
            receiver_adopted = bool(
                int(provider_delta["evidence_keys"].numel()) > 0
            )
            provider_adopted = bool(
                symmetric_pull
                and int(receiver_delta["evidence_keys"].numel()) > 0
            )
        # In the legacy one-way protocol, preserve the jointly validated
        # receiver decision. In the bilateral protocol, each endpoint uses
        # only its own private held-out measurements to decide whether to
        # install the common aggregate.
        if not symmetric_pull and not unconditional_union:
            receiver_adopted = bool(after < before)
        adopted = bool(receiver_adopted or provider_adopted)
        if adopted and not diagnostic:
            if not hasattr(self, "_cv_receiver_aggregations"):
                self._cv_receiver_aggregations = Counter()
            endpoints: list[
                tuple[int, object, Mapping[str, torch.Tensor]]
            ] = []
            if receiver_adopted:
                endpoints.append(
                    (receiver_idx, receiver_variant, aggregate_a)
                )
            if provider_adopted:
                endpoints.append(
                    (provider_idx, provider_variant, aggregate_b)
                )
            for endpoint_idx, variant, endpoint_state in endpoints:
                self._load_model_state(variant.model, endpoint_state)
                variant.opt = optim.Adam(
                    variant.model.parameters(), lr=self.cfg.local_lr
                )
                variant.t_wait = 0
                variant.last_rmse_available = False
                self._refresh_variant_signature(variant)
                self._cv_receiver_aggregations[
                    (str(mode), int(endpoint_idx))
                ] += 1

        objective_evaluations = len(optimum.evaluations)
        # Both protocols return one optimization loss and two reward losses
        # from the provider. Parameter interpolation additionally sends every
        # trial alpha and the final alpha. Exact evidence union has no alpha.
        # Symmetric mode returns the common reward/adopt decision once.
        scalar_loss_messages = objective_evaluations + 2
        if mergeable_evidence:
            scalar_control_messages = int(symmetric_pull)
        else:
            scalar_control_messages = (
                objective_evaluations + 1 + int(symmetric_pull)
            )
        scalar_messages = scalar_loss_messages + scalar_control_messages
        scalar_bytes = scalar_messages * FLOAT32_BYTES
        if not diagnostic:
            self._cv_step_scalar_loss_messages[mode] += scalar_loss_messages
            self._cv_step_scalar_control_messages[mode] += scalar_control_messages
            self._cv_step_scalar_messages[mode] += scalar_messages
            self._cv_step_scalar_bytes[mode] += scalar_bytes
            self._cv_step_valid_pulls[mode] += 1
        result = PullResult(
            valid=True,
            reason=(
                "adopted_pair"
                if receiver_adopted and provider_adopted
                else "adopted_receiver"
                if receiver_adopted and symmetric_pull
                else "adopted_provider"
                if provider_adopted
                else "retained_pair"
                if symmetric_pull
                else "adopted"
                if adopted
                else "retained_receiver"
            ),
            alpha=(None if mergeable_evidence else float(optimum.alpha)),
            objective_evaluations=objective_evaluations,
            before_loss=float(before),
            after_loss=float(after),
            # The action belongs to the initiating receiver, so its local
            # gain is the directional policy label. The joint gain remains a
            # diagnostic and aggregation-quality metric.
            reward=float(
                receiver_reward
                if symmetric_pull and receiver_reward is not None
                else joint_reward
            ),
            adopted=adopted,
            joint_reward=float(joint_reward),
            receiver_before_loss=float(before_a),
            receiver_after_loss=float(after_a),
            provider_before_loss=float(before_b),
            provider_after_loss=float(after_b),
            receiver_reward=(
                None if receiver_reward is None else float(receiver_reward)
            ),
            provider_reward=(
                None if provider_reward is None else float(provider_reward)
            ),
            receiver_information_gain=receiver_information_gain,
            provider_information_gain=provider_information_gain,
            receiver_adopted=bool(receiver_adopted),
            provider_adopted=bool(provider_adopted),
            scalar_loss_messages=scalar_loss_messages,
            scalar_control_messages=scalar_control_messages,
            scalar_messages=scalar_messages,
        )
        write_pull_log(
            step,
            mode,
            receiver_idx,
            provider_idx,
            zone,
            result,
            qa_opt,
            qb_opt,
            qa_reward,
            qb_reward,
            metadata_bytes,
            model_bytes,
        )
        return result

    @staticmethod
    def _validation_subset_weight(subset: ValidationSubset) -> float:
        """Return the weight used to combine one vehicle's private losses."""

        return max(0.0, float(subset.quality))

    # --------------------------------------------------------------- actions

    def _select_action_from_local_agent(
        self, mode_id: str, node_idx: int, state: torch.Tensor
    ) -> int:
        agents = self.local_agents.get(mode_id)
        if agents is None or not (0 <= int(node_idx) < len(agents)):
            return super()._select_action_from_local_agent(
                mode_id, node_idx, state
            )
        agent = agents[int(node_idx)]
        policy = str(agent.action_policy).strip().lower()
        agent._decisions += 1
        key = (mode_id, int(node_idx))
        if self.random_pull_probability is not None:
            self._last_predicted_gain[key] = float("nan")
            self._last_exploratory[key] = 1
            return int(agent._py_rng.random() < self.random_pull_probability)
        with torch.no_grad():
            q = agent.policy(state.unsqueeze(0).to(agent.device)).squeeze(0)
            raw_gain = float((q[1] - q[0]).item())
        self._last_predicted_gain[key] = raw_gain
        if policy in {"reject", "always_reject"}:
            self._last_exploratory[key] = 0
            return 0
        if policy in {"accept", "always_accept"}:
            self._last_exploratory[key] = 0
            return 1
        if policy == "argmax":
            current_step = int(getattr(self, "_current_sumo_step", 0))
            if (
                self.hard_warmup_steps > 0
                and current_step <= self.hard_warmup_steps
            ):
                self._last_exploratory[key] = 1
                return int(
                    agent._py_rng.random()
                    < self.hard_warmup_pull_probability
                )
            self._last_exploratory[key] = 0
            return int(raw_gain > 0.0)
        if policy != "softmax":
            raise ValueError(f"Unknown RL action policy {policy!r}")
        self._last_exploratory[key] = 0
        probabilities = torch.softmax(
            q / self.policy_temperature, dim=0
        ).detach().cpu().tolist()
        draw = agent._py_rng.random()
        return 0 if draw < float(probabilities[0]) else 1

    def _record_decision_row(self, row: dict) -> None:
        normalized = dict(row)
        key = (
            str(normalized.get("mode", "")),
            int(normalized.get("node_i", -1)),
        )
        protocol_valid = self._last_protocol_valid.pop(key, None)
        if protocol_valid is False:
            # Invalid private validation is terminal and immediately charged
            # the pull communication cost; it is never a deferred reward.
            normalized["deferred"] = False
        normalized.setdefault(
            "predicted_gain", self._last_predicted_gain.pop(key, "")
        )
        normalized.setdefault("gain_threshold", 0.0)
        normalized.setdefault("exploratory", self._last_exploratory.pop(key, ""))
        super()._record_decision_row(normalized)

    # ---------------------------------------------------------- communication

    def _reset_policy_step_counters(self) -> None:
        super()._reset_policy_step_counters()
        self._cv_step_metadata_bytes.clear()
        self._cv_step_model_bytes.clear()
        self._cv_step_scalar_bytes.clear()
        self._cv_step_model_messages.clear()
        self._cv_step_scalar_messages.clear()
        self._cv_step_scalar_loss_messages.clear()
        self._cv_step_scalar_control_messages.clear()
        self._cv_step_pulls.clear()
        self._cv_step_valid_pulls.clear()
        self._cv_step_sample_bytes.clear()
        self._cv_step_sample_messages.clear()
        self._cv_step_policy_bytes.clear()
        self._cv_step_policy_messages.clear()
        self._cv_receiver_aggregations.clear()

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        model_bytes = int(assumptions.get("B_model_bytes", 0))
        legacy_merge_metadata = int(
            assumptions.get("B_accepted_merge_meta_bytes_per_pull", 0)
        )
        metadata_bytes = int(
            assumptions.get(
                "B_decision_meta_bytes_per_directed_decision", 0
            )
        ) + 8 + (
            4
            if "relative_provider_freshness" in self.policy_state_features
            else 0
        )
        policy_bytes = int(assumptions.get("B_policy_bytes", 0))
        objective_evaluations = len(
            minimize_bounded(
                lambda _alpha: 0.0,
                tolerance=self.aggregation_tolerance,
                max_iterations=self.aggregation_max_iterations,
            ).evaluations
        )
        scalar_loss_messages = objective_evaluations + 2
        scalar_control_messages = objective_evaluations + 1 + int(
            self.symmetric_predictor_pull
        )
        scalar_messages = scalar_loss_messages + scalar_control_messages
        valid_pull_bytes = 2 * model_bytes + scalar_messages * FLOAT32_BYTES
        assumptions.update(
            {
                "zramp_policy_mode": str(self.zramp_policy_mode),
                "local_policy_share": bool(self.share_policy_every_contact),
                "policy_transfer_rule": self.policy_transfer_rule,
                "policy_pull_rule": (
                    "fixed-state-independent-bernoulli"
                    if self.random_pull_probability is not None
                    else "learned-net-reward-softmax-or-hard-positive"
                ),
                "policy_temperature": float(self.policy_temperature),
                "hard_warmup_steps": int(self.hard_warmup_steps),
                "hard_warmup_pull_probability": float(
                    self.hard_warmup_pull_probability
                ),
                "random_pull_probability": (
                    -1.0
                    if self.random_pull_probability is None
                    else float(self.random_pull_probability)
                ),
                "random_pull_ignores_policy_scores": bool(
                    self.random_pull_probability is not None
                ),
                "B_policy_model_bytes_per_directed_contact": (
                    policy_bytes if self.share_policy_every_contact else 0
                ),
                "B_policy_messages_per_directed_contact": (
                    1 if self.share_policy_every_contact else 0
                ),
                "communication_penalty_beta": float(
                    self.communication_penalty
                ),
                "communication_penalty_application": (
                    "subtract-once-from-every-pull-policy-reward"
                ),
                "policy_reward_target": (
                    "quality-weighted-RMSE-gain-dB-minus-beta-for-pull;"
                    "pass-zero"
                    if self.pull_reward_metric == "rmse-gain"
                    else "normalized-quality-weighted-MSE-improvement-minus-beta-"
                    "for-pull;pass-zero"
                ),
                "raw_validation_reward_target": (
                    "quality-weighted-RMSE-gain-dB"
                    if self.pull_reward_metric == "rmse-gain"
                    else "normalized-quality-weighted-MSE-improvement"
                ),
                "pull_reward_metric": str(self.pull_reward_metric),
                "pass_reward": 0.0,
                "B_decision_meta_bytes_per_directed_decision": metadata_bytes,
                "B_validation_quality_bytes_per_directed_decision": 8,
                "B_greedy_accepted_pull_bytes": (
                    model_bytes + legacy_merge_metadata
                ),
                "B_accepted_merge_meta_bytes_per_pull": 0,
                "B_accepted_pull_bytes": valid_pull_bytes,
                "B_local_zramp_accepted_pull_bytes": valid_pull_bytes,
                "B_model_messages_per_pull": 2,
                "B_bidirectional_model_bytes_per_pull": 2 * model_bytes,
                "B_optimizer_objective_evaluations_per_valid_pull": (
                    objective_evaluations
                ),
                "B_scalar_validation_loss_messages_per_valid_pull": (
                    scalar_loss_messages
                ),
                "B_scalar_control_messages_per_valid_pull": (
                    scalar_control_messages
                ),
                "B_scalar_messages_per_valid_pull": scalar_messages,
                "B_scalar_loss_bytes_per_message": FLOAT32_BYTES,
                "predictor_aggregation": (
                    "deterministic-bounded-continuous-alpha"
                ),
                "symmetric_predictor_pull": bool(
                    self.symmetric_predictor_pull
                ),
                "predictor_pull_reward_baseline": (
                    "each-endpoint-own-predictor-on-own-private-reward-set"
                    if self.symmetric_predictor_pull
                    else "receiver-predictor-on-both-private-reward-sets"
                ),
                "unconditional_evidence_union": bool(
                    getattr(self, "unconditional_evidence_union", False)
                ),
                "predictor_pull_adoption": (
                    "unconditional-provenance-union-at-both-endpoints"
                    if getattr(
                        self, "unconditional_evidence_union", False
                    )
                    and self.symmetric_predictor_pull
                    else "unconditional-provenance-union-at-receiver"
                    if getattr(
                        self, "unconditional_evidence_union", False
                    )
                    else "same-jointly-accepted-aggregate-at-both-endpoints"
                    if self.symmetric_predictor_pull
                    else "jointly-accepted-aggregate-at-receiver-only"
                ),
                "aggregation_alpha_convention": (
                    "alpha*self+(1-alpha)*provider"
                ),
                "aggregation_tolerance": float(self.aggregation_tolerance),
                "aggregation_max_iterations": int(
                    self.aggregation_max_iterations
                ),
                "validation_split": (
                    "streaming-80-train-10-opt-10-reward"
                ),
                "validation_selection": (
                    "deterministic-max-distance-normalized-coordinate-4d"
                ),
                "validation_quality": (
                    "sample-count-times-mean-pairwise-coordinate-distance"
                ),
                "zone_cache_scope": (
                    "predictor-training-opt-validation-reward-validation-"
                    "metadata-timestamps"
                ),
                "metadata_note": (
                    "Every directed decision receives provider predictor "
                    "metadata plus two validation-quality float32 values. A "
                    "pull exchanges exactly two full predictors; provider "
                    "optimization losses and two provider reward losses are "
                    "scalar float32 messages. Alpha queries and the selected "
                    "alpha are counted as scalar control messages. When "
                    "enabled, every feasible directed contact transfers one "
                    "full policy model in a synchronous pre-decision round. "
                    "Private samples stay local."
                ),
            }
        )
        return assumptions

    def _communication_overhead_row(
        self,
        *,
        feasible_decisions: int,
        greedy_events: int,
        rl_events: Counter,
    ) -> dict[str, int | float]:
        del feasible_decisions
        assumptions = self._communication_assumptions
        model_bytes = int(assumptions.get("B_model_bytes", 0))
        greedy_payload = int(
            assumptions.get(
                "B_greedy_accepted_pull_bytes",
                model_bytes
                + int(
                    assumptions.get(
                        "B_accepted_merge_meta_bytes_per_pull", 0
                    )
                ),
            )
        )
        greedy_bytes = int(greedy_events) * greedy_payload
        self._comm_cumulative_bytes["greedy"] += greedy_bytes
        row: dict[str, int | float] = {
            "greedy_comm_bytes": int(greedy_bytes),
            "greedy_comm_mb": float(greedy_bytes) / 1_000_000.0,
            "greedy_comm_cumulative_mb": float(
                self._comm_cumulative_bytes["greedy"]
            )
            / 1_000_000.0,
            "local_policy_initial_pull_probability": float(
                self.local_policy_initial_pull_probability
            ),
        }
        for mode in self.agents:
            samples = int(self._cv_step_sample_bytes[mode])
            metadata = int(self._cv_step_metadata_bytes[mode])
            models = int(self._cv_step_model_bytes[mode])
            scalars = int(self._cv_step_scalar_bytes[mode])
            policy = int(self._cv_step_policy_bytes[mode])
            scalar_loss_messages = int(
                self._cv_step_scalar_loss_messages[mode]
            )
            scalar_control_messages = int(
                self._cv_step_scalar_control_messages[mode]
            )
            scalar_loss_bytes = scalar_loss_messages * FLOAT32_BYTES
            scalar_control_bytes = (
                scalar_control_messages * FLOAT32_BYTES
            )
            total = metadata + models + scalars + policy + samples
            self._comm_cumulative_bytes[mode] += total
            greedy_cumulative = float(self._comm_cumulative_bytes["greedy"])
            row.update(
                {
                    f"{mode}_metadata_bytes": metadata,
                    f"{mode}_model_bytes": models,
                    f"{mode}_scalar_loss_bytes": scalar_loss_bytes,
                    f"{mode}_scalar_control_bytes": scalar_control_bytes,
                    f"{mode}_scalar_bytes": scalars,
                    f"{mode}_policy_bytes": policy,
                    f"{mode}_policy_messages": int(
                        self._cv_step_policy_messages[mode]
                    ),
                    f"{mode}_training_sample_bytes": samples,
                    f"{mode}_training_sample_messages": int(
                        self._cv_step_sample_messages[mode]
                    ),
                    f"{mode}_model_messages": int(
                        self._cv_step_model_messages[mode]
                    ),
                    f"{mode}_scalar_loss_messages": int(
                        scalar_loss_messages
                    ),
                    f"{mode}_scalar_control_messages": (
                        scalar_control_messages
                    ),
                    f"{mode}_scalar_messages": int(
                        self._cv_step_scalar_messages[mode]
                    ),
                    f"{mode}_pull_events": int(self._cv_step_pulls[mode]),
                    f"{mode}_valid_pull_events": int(
                        self._cv_step_valid_pulls[mode]
                    ),
                    f"{mode}_selected_providers": int(
                        rl_events.get(mode, 0)
                    ),
                    f"{mode}_comm_bytes": total,
                    f"{mode}_comm_mb": float(total) / 1_000_000.0,
                    f"{mode}_comm_cumulative_mb": float(
                        self._comm_cumulative_bytes[mode]
                    )
                    / 1_000_000.0,
                    f"{mode}_comm_vs_greedy_ratio": (
                        float(total) / float(greedy_bytes)
                        if greedy_bytes > 0
                        else float("nan")
                    ),
                    f"{mode}_comm_cumulative_vs_greedy_ratio": (
                        float(self._comm_cumulative_bytes[mode])
                        / greedy_cumulative
                        if greedy_cumulative > 0.0
                        else float("nan")
                    ),
                }
            )
        return row

    # ---------------------------------------------------------------- logging

    def _ensure_pull_log(self) -> csv.DictWriter:
        if self._cv_log_writer is None:
            self._cv_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._cv_log_file = open(
                self._cv_log_path, "w", newline="", encoding="utf-8"
            )
            self._cv_log_writer = csv.DictWriter(
                self._cv_log_file, fieldnames=PULL_LOG_FIELDS
            )
            self._cv_log_writer.writeheader()
        return self._cv_log_writer

    def _write_pull_log(
        self,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        zone: int,
        result: PullResult,
        qa_opt: float,
        qb_opt: float,
        qa_reward: float,
        qb_reward: float,
        metadata_bytes: int,
        model_bytes: int,
    ) -> None:
        row = {
            "step": int(step),
            "mode": str(mode),
            "receiver_idx": int(receiver_idx),
            "provider_idx": int(provider_idx),
            "zone": int(zone),
            "valid": int(result.valid),
            "reason": result.reason,
            "alpha": "" if result.alpha is None else float(result.alpha),
            "objective_evaluations": int(result.objective_evaluations),
            "receiver_opt_quality": float(qa_opt),
            "provider_opt_quality": float(qb_opt),
            "receiver_reward_quality": float(qa_reward),
            "provider_reward_quality": float(qb_reward),
            "before_loss": (
                "" if result.before_loss is None else float(result.before_loss)
            ),
            "after_loss": (
                "" if result.after_loss is None else float(result.after_loss)
            ),
            "policy_reward": (
                (0.0 if result.reward is None else float(result.reward))
                - float(self.communication_penalty)
            ),
            "reward": "" if result.reward is None else float(result.reward),
            "joint_reward": (
                "" if result.joint_reward is None else float(result.joint_reward)
            ),
            "receiver_before_loss": (
                ""
                if result.receiver_before_loss is None
                else float(result.receiver_before_loss)
            ),
            "receiver_after_loss": (
                ""
                if result.receiver_after_loss is None
                else float(result.receiver_after_loss)
            ),
            "provider_before_loss": (
                ""
                if result.provider_before_loss is None
                else float(result.provider_before_loss)
            ),
            "provider_after_loss": (
                ""
                if result.provider_after_loss is None
                else float(result.provider_after_loss)
            ),
            "receiver_reward": (
                "" if result.receiver_reward is None else float(result.receiver_reward)
            ),
            "provider_reward": (
                "" if result.provider_reward is None else float(result.provider_reward)
            ),
            "receiver_information_gain": (
                ""
                if result.receiver_information_gain is None
                else float(result.receiver_information_gain)
            ),
            "provider_information_gain": (
                ""
                if result.provider_information_gain is None
                else float(result.provider_information_gain)
            ),
            "parameter_geometry_reward": (
                ""
                if result.parameter_geometry_reward is None
                else float(result.parameter_geometry_reward)
            ),
            "parameter_geometry_alpha": (
                ""
                if result.parameter_geometry_alpha is None
                else float(result.parameter_geometry_alpha)
            ),
            "adopted": int(result.adopted),
            "receiver_adopted": int(result.receiver_adopted),
            "provider_adopted": int(result.provider_adopted),
            "metadata_bytes": int(metadata_bytes),
            "model_messages": int(result.model_messages),
            "model_bytes": int(model_bytes),
            "scalar_loss_messages": int(result.scalar_loss_messages),
            "scalar_loss_bytes": int(
                result.scalar_loss_messages * FLOAT32_BYTES
            ),
            "scalar_control_messages": int(
                result.scalar_control_messages
            ),
            "scalar_control_bytes": int(
                result.scalar_control_messages * FLOAT32_BYTES
            ),
            "scalar_messages": int(result.scalar_messages),
            "scalar_bytes": int(result.scalar_messages * FLOAT32_BYTES),
        }
        self._ensure_pull_log().writerow(row)
        self._cv_log_count += 1
        if (
            self._cv_log_count % 10000 == 0
            and self._cv_log_file is not None
        ):
            self._cv_log_file.flush()

    def _flush_pull_log(self) -> None:
        if self._cv_log_file is not None:
            self._cv_log_file.flush()

    def _close_pull_log(self) -> None:
        if self._cv_log_file is not None:
            self._cv_log_file.flush()
            self._cv_log_file.close()
        self._cv_log_file = None
        self._cv_log_writer = None

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._flush_pull_log()

    def run(self) -> None:
        self._ensure_pull_log()
        try:
            super().run()
        finally:
            self._close_pull_log()
