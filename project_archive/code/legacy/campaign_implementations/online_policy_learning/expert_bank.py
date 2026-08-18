"""Bounded, support-aware predictor banks for decentralized radio maps.

The bank keeps predictor parameters separate.  A received expert is copied and
frozen, and inference combines *outputs* only where locally learned support
profiles overlap.  Support profiles contain aggregate moments and optional
spatial occupancy mass, never raw coordinates or propagation measurements.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


def _support_features(
    values: np.ndarray | torch.Tensor,
    *,
    time_scale: float,
) -> np.ndarray:
    """Map link coordinates to a compact, symmetric support space."""

    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or int(rows.shape[1]) not in {4, 5}:
        raise ValueError("support features need shape [samples, 4 or 5]")
    midpoint = 0.5 * (rows[:, :2] + rows[:, 2:4])
    displacement = np.abs(rows[:, :2] - rows[:, 2:4])
    components = [midpoint, displacement]
    if int(rows.shape[1]) == 5:
        if not math.isfinite(float(time_scale)) or float(time_scale) <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        components.append(rows[:, 4:5] / float(time_scale))
    return np.concatenate(components, axis=1)


@dataclass(frozen=True)
class SupportProfile:
    """Aggregate one-class support metadata learned from private inputs."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    radius_squared: float
    count: int
    time_scale: float = 1800.0
    grid_shape: tuple[int, ...] = ()
    grid_mass: tuple[float, ...] = ()
    support_threshold: float = 0.0

    @classmethod
    def fit(
        cls,
        features: np.ndarray | torch.Tensor,
        *,
        coverage_quantile: float = 0.95,
        minimum_scale: float = 0.025,
        time_scale: float = 1800.0,
        spatial_grid_points: int | None = None,
        calibration_features: np.ndarray | torch.Tensor | None = None,
        target_recall: float = 0.9,
    ) -> "SupportProfile":
        raw = np.asarray(features, dtype=np.float64)
        rows = _support_features(raw, time_scale=float(time_scale))
        if int(rows.shape[0]) == 0:
            raise ValueError("cannot fit support without private inputs")
        quantile = float(coverage_quantile)
        if not 0.5 <= quantile < 1.0:
            raise ValueError("coverage_quantile must be in [0.5, 1)")
        floor = float(minimum_scale)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("minimum_scale must be finite and positive")
        mean = np.mean(rows, axis=0)
        scale = np.maximum(np.std(rows, axis=0), floor)
        # Bounds are aggregate support metadata rather than retained samples.
        # A small scale-relative margin avoids rejecting ordinary validation
        # extrema while preserving real gaps between route communities.
        margin = 0.05 * scale
        lower = np.min(rows, axis=0) - margin
        upper = np.max(rows, axis=0) + margin
        distance_squared = np.mean(np.square((rows - mean) / scale), axis=1)
        radius_squared = max(
            1.0e-6,
            float(np.quantile(distance_squared, quantile)),
        )
        grid_shape: tuple[int, ...] = ()
        grid_mass: tuple[float, ...] = ()
        support_threshold = 0.0
        if spatial_grid_points is not None:
            points = int(spatial_grid_points)
            if points < 2:
                raise ValueError("spatial_grid_points must be at least two")
            if not 0.0 < float(target_recall) <= 1.0:
                raise ValueError("target_recall must be in (0, 1]")
            grid_shape = (points,) * 4
            mass = cls._record_grid_mass(raw[:, :4], grid_shape)
            calibration = (
                raw[:, :4]
                if calibration_features is None
                else np.asarray(calibration_features, dtype=np.float64)[:, :4]
            )
            if int(calibration.shape[0]) == 0:
                raise ValueError("grid support calibration cannot be empty")
            calibration_mass = cls._interpolate_grid_mass(
                calibration, grid_shape, mass
            )
            support_threshold = max(
                0.0,
                float(
                    np.quantile(
                        calibration_mass,
                        max(0.0, 1.0 - float(target_recall)),
                    )
                ),
            )
            grid_mass = tuple(float(value) for value in mass)
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            lower=tuple(float(value) for value in lower),
            upper=tuple(float(value) for value in upper),
            radius_squared=radius_squared,
            count=int(rows.shape[0]),
            time_scale=float(time_scale),
            grid_shape=grid_shape,
            grid_mass=grid_mass,
            support_threshold=float(support_threshold),
        )

    @staticmethod
    def _grid_corners(
        features: np.ndarray, grid_shape: tuple[int, ...]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or int(rows.shape[1]) != len(grid_shape):
            raise ValueError("grid support features and dimensions differ")
        coordinates = np.clip(rows, 0.0, 1.0)
        sizes = np.asarray(grid_shape, dtype=np.int64)
        scaled = coordinates * (sizes.astype(np.float64) - 1.0)
        lower = np.floor(scaled).astype(np.int64)
        upper = np.minimum(lower + 1, sizes - 1)
        fraction = scaled - lower.astype(np.float64)
        strides = np.asarray(
            [
                math.prod(grid_shape[index + 1 :])
                for index in range(len(grid_shape))
            ],
            dtype=np.int64,
        )
        corners: list[tuple[np.ndarray, np.ndarray]] = []
        for corner in range(1 << len(grid_shape)):
            index = np.zeros(int(rows.shape[0]), dtype=np.int64)
            weight = np.ones(int(rows.shape[0]), dtype=np.float64)
            for dimension in range(len(grid_shape)):
                use_upper = bool(corner & (1 << dimension))
                point = (
                    upper[:, dimension]
                    if use_upper
                    else lower[:, dimension]
                )
                component = (
                    fraction[:, dimension]
                    if use_upper
                    else 1.0 - fraction[:, dimension]
                )
                index += point * strides[dimension]
                weight *= component
            corners.append((index, weight))
        return corners

    @classmethod
    def _record_grid_mass(
        cls, features: np.ndarray, grid_shape: tuple[int, ...]
    ) -> np.ndarray:
        mass = np.zeros(math.prod(grid_shape), dtype=np.float64)
        for index, weight in cls._grid_corners(features, grid_shape):
            np.add.at(mass, index, np.square(weight))
        return mass

    @classmethod
    def _interpolate_grid_mass(
        cls,
        features: np.ndarray,
        grid_shape: tuple[int, ...],
        mass: np.ndarray,
    ) -> np.ndarray:
        values = np.zeros(int(np.asarray(features).shape[0]), dtype=np.float64)
        flat = np.asarray(mass, dtype=np.float64).reshape(-1)
        for index, weight in cls._grid_corners(features, grid_shape):
            values += weight * flat[index]
        return values

    @property
    def dimension(self) -> int:
        return len(self.mean)

    @property
    def wire_nbytes(self) -> int:
        # mean, diagonal scale, bounds, radius and count as float32 values.
        grid_metadata = len(self.grid_mass) + (2 if self.grid_mass else 0)
        return 4 * (4 * self.dimension + 2 + grid_metadata)

    @property
    def has_grid_support(self) -> bool:
        return bool(self.grid_shape and self.grid_mass)

    def confidence(
        self, features: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        if self.has_grid_support:
            raw = np.asarray(features, dtype=np.float64)
            mass = self._interpolate_grid_mass(
                raw[:, :4],
                self.grid_shape,
                np.asarray(self.grid_mass, dtype=np.float64),
            )
            if float(self.support_threshold) <= 0.0:
                # A zero validation quantile means the requested recall is
                # unattainable with the observed grid. Preserve the intended
                # "any nonzero learned mass" gate and rank by mass without an
                # unstable division by the smallest representable float.
                maximum = max(float(np.max(mass, initial=0.0)), 1.0e-12)
                return np.where(mass > 0.0, 1.0 + mass / maximum, 0.0).astype(
                    np.float32
                )
            return (mass / float(self.support_threshold)).astype(np.float32)
        rows = _support_features(features, time_scale=self.time_scale)
        if int(rows.shape[1]) != self.dimension:
            raise ValueError("support-profile and query dimensions differ")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        distance_squared = np.mean(np.square((rows - mean) / scale), axis=1)
        outside = np.maximum(distance_squared - self.radius_squared, 0.0)
        within_bounds = np.all(
            (rows >= np.asarray(self.lower, dtype=np.float64))
            & (rows <= np.asarray(self.upper, dtype=np.float64)),
            axis=1,
        )
        confidence = np.exp(-0.5 * outside)
        return np.where(within_bounds, confidence, 0.0).astype(np.float32)


    @staticmethod
    def coarse_cell_indices(
        features: np.ndarray | torch.Tensor, *, grid_points: int
    ) -> np.ndarray:
        """Map link coordinates to deterministic coarse certificate cells."""

        points = int(grid_points)
        if points < 2:
            raise ValueError("coarse certificate grid needs at least two points")
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or int(rows.shape[1]) not in {4, 5}:
            raise ValueError("cell features need shape [samples, 4 or 5]")
        bins = np.minimum(
            np.floor(np.clip(rows[:, :4], 0.0, 1.0) * points).astype(np.int64),
            points - 1,
        )
        return np.ravel_multi_index(bins.T, (points,) * 4).astype(np.int64)

    @staticmethod
    def coarse_cell_centers(*, grid_points: int) -> np.ndarray:
        points = int(grid_points)
        if points < 2:
            raise ValueError("coarse certificate grid needs at least two points")
        axis = (np.arange(points, dtype=np.float64) + 0.5) / float(points)
        return np.stack(
            np.meshgrid(axis, axis, axis, axis, indexing="ij"), axis=-1
        ).reshape(-1, 4)

    def _compute_supported_coarse_cells(
        self, *, grid_points: int
    ) -> tuple[int, ...]:
        centers = self.coarse_cell_centers(grid_points=int(grid_points))
        if self.dimension == 5:
            centers = np.concatenate(
                (
                    centers,
                    np.full(
                        (len(centers), 1),
                        float(self.mean[-1]) * float(self.time_scale),
                    ),
                ),
                axis=1,
            )
        return tuple(
            int(index)
            for index in np.flatnonzero(self.confidence(centers) >= 1.0)
        )

    @functools.cached_property
    def _supported_cells_grid3(self) -> tuple[int, ...]:
        return self._compute_supported_coarse_cells(grid_points=3)

    def supported_coarse_cells(self, *, grid_points: int) -> tuple[int, ...]:
        if int(grid_points) == 3:
            return self._supported_cells_grid3
        return self._compute_supported_coarse_cells(
            grid_points=int(grid_points)
        )

    def novelty(self, other: "SupportProfile") -> float:
        """Symmetric separation in [0, 1] between aggregate supports."""

        if self.dimension != other.dimension:
            raise ValueError("support profiles have different dimensions")
        first = np.asarray(self.mean, dtype=np.float64)
        second = np.asarray(other.mean, dtype=np.float64)
        pooled = np.sqrt(
            np.square(np.asarray(self.scale, dtype=np.float64))
            + np.square(np.asarray(other.scale, dtype=np.float64))
        )
        distance_squared = float(
            np.mean(np.square((first - second) / np.maximum(pooled, 1.0e-8)))
        )
        return float(1.0 - math.exp(-0.5 * distance_squared))


@dataclass(frozen=True)
class ValidationCertificate:
    """Scalar utility evidence; no validation rows or predictions are stored."""

    validator_id: str
    marginal_gain_db: float
    coverage_quality: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.marginal_gain_db)):
            raise ValueError("marginal_gain_db must be finite")
        if not math.isfinite(float(self.coverage_quality)):
            raise ValueError("coverage_quality must be finite")



@dataclass(frozen=True)
class CellValidationCertificate:
    """Aggregate error evidence bound to one exact predictor capsule."""

    expert_hash: str
    validator_id: str
    epoch: int
    cell_index: int
    grid_points: int
    sample_count: int
    squared_error_sum_db2: float
    prior_squared_error_sum_db2: float

    def __post_init__(self) -> None:
        if len(str(self.expert_hash)) < 16:
            raise ValueError("expert_hash must be a content hash")
        if not str(self.validator_id):
            raise ValueError("validator_id cannot be empty")
        if int(self.epoch) < 0:
            raise ValueError("certificate epoch cannot be negative")
        if int(self.grid_points) < 2:
            raise ValueError("certificate grid needs at least two points")
        if int(self.cell_index) < -1 or int(self.cell_index) >= int(
            self.grid_points
        ) ** 4:
            raise ValueError("certificate cell index is out of range")
        if int(self.sample_count) <= 0:
            raise ValueError("certificate sample_count must be positive")
        if not math.isfinite(float(self.squared_error_sum_db2)) or float(
            self.squared_error_sum_db2
        ) < 0.0:
            raise ValueError("certificate squared error must be finite/nonnegative")
        if not math.isfinite(float(self.prior_squared_error_sum_db2)) or float(
            self.prior_squared_error_sum_db2
        ) < 0.0:
            raise ValueError("certificate prior error must be finite/nonnegative")

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (
            str(self.expert_hash),
            str(self.validator_id),
            int(self.epoch),
            int(self.grid_points),
            int(self.cell_index),
        )

    @property
    def rmse_db(self) -> float:
        return float(
            math.sqrt(float(self.squared_error_sum_db2) / int(self.sample_count))
        )

    @property
    def prior_rmse_db(self) -> float:
        return float(
            math.sqrt(
                float(self.prior_squared_error_sum_db2) / int(self.sample_count)
            )
        )

    @property
    def wire_nbytes(self) -> int:
        return 16 + len(self.validator_id.encode("utf-8")) + 6 * 4


def predictor_content_hash(
    model: nn.Module,
    support: SupportProfile,
    *,
    lineage_id: str,
    version: int,
) -> str:
    """Return a stable BLAKE2 digest of one versioned predictor capsule."""

    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(lineage_id).encode("utf-8"))
    digest.update(np.asarray([int(version)], dtype="<i8").tobytes())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    digest.update(
        np.asarray(
            [
                *support.mean,
                *support.scale,
                *support.lower,
                *support.upper,
                support.radius_squared,
                float(support.count),
                support.time_scale,
                support.support_threshold,
            ],
            dtype="<f8",
        ).tobytes()
    )
    digest.update(np.asarray(support.grid_shape, dtype="<i8").tobytes())
    digest.update(np.asarray(support.grid_mass, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass
class PredictorExpert:
    """One immutable knowledge lineage carried inside an expert bank."""

    lineage_id: str
    model: nn.Module
    support: SupportProfile
    own_validation_rmse_db: float
    experience: int
    version: int = 0
    policy_embedding: torch.Tensor | None = None
    certificates: list[ValidationCertificate] = field(default_factory=list)
    cell_certificates: list[CellValidationCertificate] = field(
        default_factory=list
    )
    content_hash: str = ""
    locally_owned: bool = True

    def __post_init__(self) -> None:
        if not self.lineage_id:
            raise ValueError("lineage_id cannot be empty")
        if not math.isfinite(float(self.own_validation_rmse_db)):
            raise ValueError("own_validation_rmse_db must be finite")
        if int(self.experience) < 0:
            raise ValueError("experience cannot be negative")
        if int(self.version) < 0:
            raise ValueError("expert version cannot be negative")
        computed_hash = predictor_content_hash(
            self.model,
            self.support,
            lineage_id=self.lineage_id,
            version=self.version,
        )
        if self.content_hash and str(self.content_hash) != computed_hash:
            raise ValueError("expert content hash does not match its capsule")
        self.content_hash = computed_hash
        if any(
            row.expert_hash != self.content_hash
            for row in self.cell_certificates
        ):
            raise ValueError("certificate is bound to a different expert")
        if self.policy_embedding is not None:
            embedding = self.policy_embedding.detach().to(
                dtype=torch.float32, device="cpu"
            ).reshape(-1)
            if int(embedding.numel()) == 0 or not bool(
                torch.isfinite(embedding).all()
            ):
                raise ValueError("policy embedding must be finite and nonempty")
            self.policy_embedding = embedding.clone()

    def transferred_copy(self) -> "PredictorExpert":
        cloned = copy.deepcopy(self)
        cloned.locally_owned = False
        if cloned.policy_embedding is not None:
            cloned.policy_embedding = (
                cloned.policy_embedding.detach().cpu().clone()
            )
        cloned.model.eval()
        for parameter in cloned.model.parameters():
            parameter.requires_grad_(False)
        return cloned

    def refresh_content_hash(self, *, clear_certificates: bool = True) -> None:
        self.content_hash = predictor_content_hash(
            self.model,
            self.support,
            lineage_id=self.lineage_id,
            version=self.version,
        )
        if clear_certificates:
            self.certificates = []
            self.cell_certificates = []

    def add_certificate(self, certificate: ValidationCertificate) -> None:
        rows = {
            row.validator_id: row for row in self.certificates
        }
        rows[str(certificate.validator_id)] = certificate
        self.certificates = [rows[key] for key in sorted(rows)]

    def add_cell_certificate(
        self, certificate: CellValidationCertificate
    ) -> None:
        if certificate.expert_hash != self.content_hash:
            raise ValueError("certificate is bound to a different expert")
        def logical_key(
            row: CellValidationCertificate,
        ) -> tuple[str, str, int, int]:
            return (
                row.expert_hash,
                row.validator_id,
                int(row.grid_points),
                int(row.cell_index),
            )

        rows = {
            logical_key(row): row
            for row in self.cell_certificates
        }
        key = logical_key(certificate)
        current = rows.get(key)
        if current is None or int(certificate.epoch) >= int(current.epoch):
            rows[key] = certificate
        bounded = sorted(
            rows.values(),
            key=lambda row: (
                ":source" in row.validator_id,
                int(row.cell_index) == -1,
                int(row.sample_count),
                int(row.epoch),
                row.key,
            ),
            reverse=True,
        )[:128]
        self.cell_certificates = sorted(bounded, key=lambda row: row.key)

    @property
    def certificate_wire_nbytes(self) -> int:
        return int(sum(row.wire_nbytes for row in self.cell_certificates))

    def certified_rmse_db(
        self,
        *,
        cell_index: int = -1,
        grid_points: int = 3,
        uncertainty_db: float = 10.0,
    ) -> float:
        exact = [
            row
            for row in self.cell_certificates
            if int(row.grid_points) == int(grid_points)
            and int(row.cell_index) == int(cell_index)
        ]
        rows = exact or [
            row
            for row in self.cell_certificates
            if int(row.grid_points) == int(grid_points)
            and int(row.cell_index) == -1
        ]
        if not rows:
            return float(self.own_validation_rmse_db + uncertainty_db)
        count = int(sum(row.sample_count for row in rows))
        squared_error = float(
            sum(row.squared_error_sum_db2 for row in rows)
        )
        return float(
            math.sqrt(squared_error / max(1, count))
            + float(uncertainty_db) / math.sqrt(max(1, count))
        )

    def certified_prior_rmse_db(
        self, *, cell_index: int = -1, grid_points: int = 3
    ) -> float:
        exact = [
            row
            for row in self.cell_certificates
            if int(row.grid_points) == int(grid_points)
            and int(row.cell_index) == int(cell_index)
        ]
        rows = exact or [
            row
            for row in self.cell_certificates
            if int(row.grid_points) == int(grid_points)
            and int(row.cell_index) == -1
        ]
        if not rows:
            return 135.0
        count = int(sum(row.sample_count for row in rows))
        squared_error = float(
            sum(row.prior_squared_error_sum_db2 for row in rows)
        )
        return float(math.sqrt(squared_error / max(1, count)))

    def external_utility_db(self) -> float:
        weighted_gains: list[float] = []
        weights: list[float] = []
        for row in self.certificates:
            weighted_gains.append(float(row.marginal_gain_db))
            weights.append(max(0.0, float(row.coverage_quality)))
        for row in self.cell_certificates:
            weighted_gains.append(float(row.prior_rmse_db - row.rmse_db))
            weights.append(float(row.sample_count))
        if not weighted_gains:
            return 0.0
        gain_array = np.asarray(weighted_gains, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        if float(np.sum(weight_array)) <= 0.0:
            return float(np.mean(gain_array))
        return float(np.sum(weight_array * gain_array) / np.sum(weight_array))


@dataclass(frozen=True)
class ExpertManifestEntry:
    lineage_id: str
    version: int
    content_hash: str
    experience: int
    supported_cells: tuple[int, ...]
    certificate_count: int
    capsule_nbytes: int

    @property
    def wire_nbytes(self) -> int:
        return (
            16
            + len(self.lineage_id.encode("utf-8"))
            + 5 * 4
            + 4 * len(self.supported_cells)
        )


@dataclass(frozen=True)
class BankPrediction:
    normalized_loss: np.ndarray
    weights: np.ndarray
    supported: np.ndarray


@dataclass(frozen=True)
class RetentionDecision:
    kept_lineages: tuple[str, ...]
    evicted_lineage: str | None
    objective_by_eviction: dict[str, float]


@dataclass(frozen=True)
class AcquisitionReward:
    """Private cross-validation label for one directional expert acquisition."""

    joint_gain_db: float
    receiver_gain_db: float
    provider_gain_db: float
    receiver_before_rmse_db: float
    receiver_after_rmse_db: float
    provider_before_rmse_db: float
    provider_after_rmse_db: float
    duplicate_lineage: bool


def pareto_safe_acquisition(
    reward: AcquisitionReward, *, epsilon_db: float = 1.0e-9
) -> bool:
    """Accept only positive acquisitions that preserve both distributions."""

    tolerance = abs(float(epsilon_db))
    return bool(
        reward.joint_gain_db > 0.0
        and reward.receiver_gain_db >= -tolerance
        and reward.provider_gain_db >= -tolerance
    )


class ExpertBank:
    """A bounded output-level mixture of immutable predictor experts."""

    def __init__(
        self,
        experts: Iterable[PredictorExpert] = (),
        *,
        capacity: int = 4,
        temperature: float = 0.15,
        logit_bias_by_lineage: dict[str, float] | None = None,
        support_threshold: float = 0.05,
        prior_normalized_loss: float = 1.0,
        uncertainty_floor_db: float = 1.0,
        routing: str = "soft-mixture",
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.logit_bias_by_lineage = {
            str(key): float(value)
            for key, value in (logit_bias_by_lineage or {}).items()
        }
        if not all(
            math.isfinite(value)
            for value in self.logit_bias_by_lineage.values()
        ):
            raise ValueError("expert logit biases must be finite")
        self.support_threshold = float(support_threshold)
        if not 0.0 <= self.support_threshold < 1.0:
            raise ValueError("support_threshold must be in [0, 1)")
        self.prior_normalized_loss = float(prior_normalized_loss)
        self.uncertainty_floor_db = float(uncertainty_floor_db)
        self.routing = str(routing).strip().lower()
        if self.routing not in {
            "soft-mixture",
            "hard-max-support",
            "hard-certified",
        }:
            raise ValueError(
                "routing must be soft-mixture, hard-max-support, or hard-certified"
            )
        self.experts: list[PredictorExpert] = []
        for expert in experts:
            self.append(expert, allow_probation=True)
        if len(self.experts) > self.capacity + 1:
            raise ValueError("initial bank exceeds capacity plus probation")

    @property
    def lineages(self) -> tuple[str, ...]:
        return tuple(expert.lineage_id for expert in self.experts)

    def append(
        self,
        expert: PredictorExpert,
        *,
        allow_probation: bool = False,
    ) -> bool:
        if expert.lineage_id in self.lineages:
            return False
        limit = self.capacity + int(bool(allow_probation))
        if len(self.experts) >= limit:
            raise RuntimeError("expert bank is full")
        self.experts.append(expert)
        return True

    def expert_for_lineage(self, lineage_id: str) -> PredictorExpert | None:
        return next(
            (
                expert
                for expert in self.experts
                if expert.lineage_id == str(lineage_id)
            ),
            None,
        )

    def manifest(self, *, grid_points: int = 3) -> tuple[ExpertManifestEntry, ...]:
        return tuple(
            ExpertManifestEntry(
                lineage_id=expert.lineage_id,
                version=int(expert.version),
                content_hash=expert.content_hash,
                experience=int(expert.experience),
                supported_cells=expert.support.supported_coarse_cells(
                    grid_points=int(grid_points)
                ),
                certificate_count=len(expert.cell_certificates),
                capsule_nbytes=(
                    sum(
                        int(parameter.numel()) * int(parameter.element_size())
                        for parameter in expert.model.state_dict().values()
                    )
                    + expert.support.wire_nbytes
                    + expert.certificate_wire_nbytes
                ),
            )
            for expert in sorted(
                self.experts,
                key=lambda row: (row.lineage_id, int(row.version)),
            )
        )

    @property
    def manifest_wire_nbytes(self) -> int:
        return int(sum(row.wire_nbytes for row in self.manifest()))

    def usable_provider_experts(
        self, provider: "ExpertBank"
    ) -> tuple[PredictorExpert, ...]:
        rows: list[PredictorExpert] = []
        for candidate in provider.experts:
            current = self.expert_for_lineage(candidate.lineage_id)
            if current is None or (
                not current.locally_owned
                and int(candidate.version) > int(current.version)
            ):
                rows.append(candidate)
        return tuple(
            sorted(rows, key=lambda row: (row.lineage_id, int(row.version)))
        )

    def certified_risk_db(
        self,
        *,
        cell_index: int,
        grid_points: int = 3,
        prior_rmse_db: float = 135.0,
    ) -> float:
        supported = [
            expert
            for expert in self.experts
            if int(cell_index)
            in expert.support.supported_coarse_cells(
                grid_points=int(grid_points)
            )
        ]
        if not supported:
            return float(prior_rmse_db)
        return float(
            min(
                expert.certified_rmse_db(
                    cell_index=int(cell_index),
                    grid_points=int(grid_points),
                )
                for expert in supported
            )
        )

    def certified_marginal_gain_db(
        self,
        candidate: PredictorExpert,
        *,
        grid_points: int = 3,
    ) -> float:
        cells = sorted(
            {
                int(row.cell_index)
                for row in candidate.cell_certificates
                if int(row.grid_points) == int(grid_points)
                and int(row.cell_index) >= 0
            }
        )
        if not cells:
            cells = list(
                candidate.support.supported_coarse_cells(
                    grid_points=int(grid_points)
                )
            )
        if not cells:
            current = min(
                (
                    row.certified_rmse_db(
                        cell_index=-1, grid_points=int(grid_points)
                    )
                    for row in self.experts
                ),
                default=candidate.certified_prior_rmse_db(
                    cell_index=-1, grid_points=int(grid_points)
                ),
            )
            return float(
                current
                - candidate.certified_rmse_db(
                    cell_index=-1, grid_points=int(grid_points)
                )
            )
        gains: list[float] = []
        weights: list[float] = []
        for cell in cells:
            candidate_risk = candidate.certified_rmse_db(
                cell_index=int(cell), grid_points=int(grid_points)
            )
            prior = candidate.certified_prior_rmse_db(
                cell_index=int(cell), grid_points=int(grid_points)
            )
            current_risk = self.certified_risk_db(
                cell_index=int(cell),
                grid_points=int(grid_points),
                prior_rmse_db=prior,
            )
            matching = [
                row.sample_count
                for row in candidate.cell_certificates
                if int(row.grid_points) == int(grid_points)
                and int(row.cell_index) == int(cell)
            ]
            gains.append(float(current_risk - candidate_risk))
            weights.append(float(sum(matching) if matching else 1))
        return float(np.average(gains, weights=weights))

    def install(
        self,
        expert: PredictorExpert,
        *,
        allow_probation: bool = False,
    ) -> str:
        """Add a lineage or refresh an older frozen snapshot in place.

        Refreshes never consume another bank slot. A locally owned
        authoritative expert cannot be replaced by a received snapshot.
        """

        current = self.expert_for_lineage(expert.lineage_id)
        if current is None:
            self.append(expert, allow_probation=allow_probation)
            return "added"
        if current.locally_owned or int(expert.version) <= int(current.version):
            return "ignored"
        position = self.experts.index(current)
        self.experts[position] = expert
        return "refreshed"

    @staticmethod
    def _model_prediction(
        model: nn.Module, features: np.ndarray
    ) -> np.ndarray:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        model.eval()
        rows: list[np.ndarray] = []
        values = np.asarray(features, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, int(values.shape[0]), 4096):
                batch = torch.as_tensor(
                    values[start : start + 4096],
                    dtype=torch.float32,
                    device=device,
                )
                rows.append(
                    model(batch).detach().cpu().numpy().reshape(-1)
                )
        return (
            np.concatenate(rows).astype(np.float32)
            if rows
            else np.empty((0,), dtype=np.float32)
        )

    def predict(
        self,
        features: np.ndarray | torch.Tensor,
        *,
        temperature: float | None = None,
    ) -> BankPrediction:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("prediction features must be a matrix")
        count = int(values.shape[0])
        if not self.experts:
            return BankPrediction(
                normalized_loss=np.full(
                    count, self.prior_normalized_loss, dtype=np.float32
                ),
                weights=np.empty((count, 0), dtype=np.float32),
                supported=np.zeros(count, dtype=bool),
            )
        tau = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("temperature must be finite and positive")
        predictions = np.stack(
            [self._model_prediction(expert.model, values) for expert in self.experts],
            axis=1,
        )
        confidence = np.stack(
            [expert.support.confidence(values) for expert in self.experts],
            axis=1,
        ).astype(np.float64)
        if self.routing in {"hard-max-support", "hard-certified"}:
            if any(
                not expert.support.has_grid_support
                for expert in self.experts
            ):
                raise ValueError("hard support routing requires grid support")
            eligible = confidence >= 1.0
            supported = np.any(eligible, axis=1)
            if self.routing == "hard-certified":
                risks = np.empty_like(confidence, dtype=np.float64)
                for expert_index, expert in enumerate(self.experts):
                    certificate_grid = next(
                        (
                            int(row.grid_points)
                            for row in expert.cell_certificates
                        ),
                        3,
                    )
                    cells = SupportProfile.coarse_cell_indices(
                        values, grid_points=certificate_grid
                    )
                    risks[:, expert_index] = np.asarray(
                        [
                            expert.certified_rmse_db(
                                cell_index=int(cell),
                                grid_points=certificate_grid,
                            )
                            for cell in cells
                        ],
                        dtype=np.float64,
                    )
                choice = np.argmin(
                    np.where(
                        eligible,
                        risks - 1.0e-6 * confidence,
                        np.inf,
                    ),
                    axis=1,
                )
            else:
                choice = np.argmax(
                    np.where(eligible, confidence, -np.inf), axis=1
                )
            weights = np.zeros_like(confidence, dtype=np.float32)
            active_rows = np.flatnonzero(supported)
            if int(active_rows.size) > 0:
                weights[active_rows, choice[active_rows]] = 1.0
            routed = np.full(
                count, self.prior_normalized_loss, dtype=np.float32
            )
            routed[supported] = predictions[
                active_rows, choice[active_rows]
            ]
            return BankPrediction(
                normalized_loss=routed,
                weights=weights,
                supported=supported,
            )
        variance = np.asarray(
            [
                max(
                    self.uncertainty_floor_db,
                    float(expert.own_validation_rmse_db),
                )
                ** 2
                for expert in self.experts
            ],
            dtype=np.float64,
        )
        # Temperature controls only spatial/time support sharpness. Private
        # RMSEs are measured on different local distributions and therefore
        # remain a gentle calibration term rather than being amplified as the
        # gate approaches hard routing.
        logits = (
            np.log(np.maximum(confidence, 1.0e-12)) / tau
            - np.log(variance[None, :])
        )
        logits = logits + np.asarray(
            [
                self.logit_bias_by_lineage.get(expert.lineage_id, 0.0)
                for expert in self.experts
            ],
            dtype=np.float64,
        )[None, :]
        eligible = confidence >= self.support_threshold
        supported = np.any(eligible, axis=1)
        logits = np.where(eligible, logits, -np.inf)
        weights = np.zeros_like(confidence, dtype=np.float64)
        if np.any(supported):
            active = logits[supported]
            maximum = np.max(active, axis=1, keepdims=True)
            exponent = np.exp(active - maximum)
            exponent = np.where(np.isfinite(exponent), exponent, 0.0)
            weights[supported] = exponent / np.maximum(
                np.sum(exponent, axis=1, keepdims=True), 1.0e-12
            )
        mixture = np.sum(weights * predictions, axis=1)
        mixture = np.where(
            supported, mixture, self.prior_normalized_loss
        ).astype(np.float32)
        return BankPrediction(
            normalized_loss=mixture,
            weights=weights.astype(np.float32),
            supported=supported,
        )

    def rmse_db(
        self,
        features: np.ndarray | torch.Tensor,
        target_dbm: np.ndarray | torch.Tensor,
        *,
        rssi_min_dbm: float = -120.0,
        rssi_max_dbm: float = 15.0,
        temperature: float | None = None,
    ) -> float:
        normalized = self.predict(features, temperature=temperature).normalized_loss
        prediction = float(rssi_max_dbm) - (
            float(rssi_max_dbm) - float(rssi_min_dbm)
        ) * np.clip(normalized, 0.0, 1.0)
        target = np.asarray(target_dbm, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean(np.square(prediction - target))))

    @staticmethod
    def _mean_pairwise_novelty(experts: Sequence[PredictorExpert]) -> float:
        if len(experts) < 2:
            return 0.0
        values = [
            left.support.novelty(right.support)
            for index, left in enumerate(experts)
            for right in experts[index + 1 :]
        ]
        return float(np.mean(values))

    @staticmethod
    def _external_utility(experts: Sequence[PredictorExpert]) -> float:
        return float(
            np.mean([expert.external_utility_db() for expert in experts])
        ) if experts else 0.0

    def resolve_probation(
        self,
        validation_features: np.ndarray | torch.Tensor,
        validation_target_dbm: np.ndarray | torch.Tensor,
        *,
        diversity_weight_db: float = 0.25,
        external_utility_weight: float = 0.5,
        rssi_min_dbm: float = -120.0,
        rssi_max_dbm: float = 15.0,
    ) -> RetentionDecision:
        if len(self.experts) <= self.capacity:
            return RetentionDecision(self.lineages, None, {})
        if len(self.experts) != self.capacity + 1:
            raise RuntimeError("probation resolution expects exactly K+1 experts")
        original = list(self.experts)
        objectives: dict[str, float] = {}
        candidates: list[tuple[float, str, list[PredictorExpert]]] = []
        # The sole locally owned lineage is the carrier's continuously trained
        # specialist and must remain refreshable/advertisable. Probation only
        # chooses among frozen received lineages when such a choice exists.
        evictable = [row for row in original if not row.locally_owned]
        if not evictable:
            evictable = original
        for evicted in evictable:
            kept = [row for row in original if row.lineage_id != evicted.lineage_id]
            trial = ExpertBank(
                kept,
                capacity=self.capacity,
                temperature=self.temperature,
                logit_bias_by_lineage=self.logit_bias_by_lineage,
                support_threshold=self.support_threshold,
                prior_normalized_loss=self.prior_normalized_loss,
                uncertainty_floor_db=self.uncertainty_floor_db,
                routing=self.routing,
            )
            local_rmse = trial.rmse_db(
                validation_features,
                validation_target_dbm,
                rssi_min_dbm=rssi_min_dbm,
                rssi_max_dbm=rssi_max_dbm,
            )
            objective = (
                local_rmse
                - float(diversity_weight_db)
                * self._mean_pairwise_novelty(kept)
                - float(external_utility_weight)
                * self._external_utility(kept)
            )
            objectives[evicted.lineage_id] = float(objective)
            candidates.append((float(objective), evicted.lineage_id, kept))
        _objective, evicted_lineage, selected = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        self.experts = selected
        return RetentionDecision(
            kept_lineages=self.lineages,
            evicted_lineage=evicted_lineage,
            objective_by_eviction=objectives,
        )


def bilateral_zone_reward(
    receiver_before_rmse_db: float,
    receiver_after_rmse_db: float,
    receiver_coverage_quality: float,
    provider_before_rmse_db: float,
    provider_after_rmse_db: float,
    provider_coverage_quality: float,
    *,
    overlap: float = 0.0,
) -> tuple[float, float, float]:
    """Return coverage-weighted joint gain and both private endpoint gains."""

    overlap_value = float(overlap)
    if not 0.0 <= overlap_value <= 1.0:
        raise ValueError("overlap must be in [0, 1]")
    receiver_gain = float(receiver_before_rmse_db) - float(
        receiver_after_rmse_db
    )
    provider_gain = float(provider_before_rmse_db) - float(
        provider_after_rmse_db
    )
    # Half of shared support is assigned to each endpoint; unique support is
    # retained in full.  This avoids treating overlapping private holdouts as
    # twice as much evidence while preserving unequal validation quality.
    receiver_weight = max(0.0, float(receiver_coverage_quality)) * (
        1.0 - 0.5 * overlap_value
    )
    provider_weight = max(0.0, float(provider_coverage_quality)) * (
        1.0 - 0.5 * overlap_value
    )
    total = receiver_weight + provider_weight
    joint = (
        0.0
        if total <= 0.0
        else (
            receiver_weight * receiver_gain
            + provider_weight * provider_gain
        )
        / total
    )
    return float(joint), receiver_gain, provider_gain


def support_overlap(first: SupportProfile, second: SupportProfile) -> float:
    """Compact overlap proxy used only to weight scalar validation rewards."""

    return float(1.0 - first.novelty(second))


def cross_validated_acquisition_reward(
    receiver_bank: ExpertBank,
    provider_expert: PredictorExpert,
    receiver_validation_features: np.ndarray | torch.Tensor,
    receiver_validation_target_dbm: np.ndarray | torch.Tensor,
    provider_validation_features: np.ndarray | torch.Tensor,
    provider_validation_target_dbm: np.ndarray | torch.Tensor,
    *,
    receiver_coverage_quality: float,
    provider_coverage_quality: float,
    overlap: float,
    rssi_min_dbm: float = -120.0,
    rssi_max_dbm: float = 15.0,
) -> AcquisitionReward:
    """Score the same receiver bank on both private validation distributions.

    In a deployment the receiver evaluates both bank states on its private
    holdout. The provider temporarily receives the receiver bank models,
    evaluates the same before/after states on its own holdout, and returns four
    scalar losses. Neither endpoint sends coordinates, labels, or predictions.
    """

    receiver_before = receiver_bank.rmse_db(
        receiver_validation_features,
        receiver_validation_target_dbm,
        rssi_min_dbm=rssi_min_dbm,
        rssi_max_dbm=rssi_max_dbm,
    )
    provider_before = receiver_bank.rmse_db(
        provider_validation_features,
        provider_validation_target_dbm,
        rssi_min_dbm=rssi_min_dbm,
        rssi_max_dbm=rssi_max_dbm,
    )
    current = receiver_bank.expert_for_lineage(provider_expert.lineage_id)
    duplicate = bool(
        current is not None
        and (
            current.locally_owned
            or int(provider_expert.version) <= int(current.version)
        )
    )
    if duplicate:
        receiver_after = receiver_before
        provider_after = provider_before
    else:
        after_bank = ExpertBank(
            list(receiver_bank.experts),
            capacity=max(receiver_bank.capacity, len(receiver_bank.experts) + 1),
            temperature=receiver_bank.temperature,
            logit_bias_by_lineage=receiver_bank.logit_bias_by_lineage,
            support_threshold=receiver_bank.support_threshold,
            prior_normalized_loss=receiver_bank.prior_normalized_loss,
            uncertainty_floor_db=receiver_bank.uncertainty_floor_db,
            routing=receiver_bank.routing,
        )
        after_bank.install(
            provider_expert.transferred_copy(), allow_probation=True
        )
        receiver_after = after_bank.rmse_db(
            receiver_validation_features,
            receiver_validation_target_dbm,
            rssi_min_dbm=rssi_min_dbm,
            rssi_max_dbm=rssi_max_dbm,
        )
        provider_after = after_bank.rmse_db(
            provider_validation_features,
            provider_validation_target_dbm,
            rssi_min_dbm=rssi_min_dbm,
            rssi_max_dbm=rssi_max_dbm,
        )
    joint, receiver_gain, provider_gain = bilateral_zone_reward(
        receiver_before,
        receiver_after,
        receiver_coverage_quality,
        provider_before,
        provider_after,
        provider_coverage_quality,
        overlap=overlap,
    )
    return AcquisitionReward(
        joint_gain_db=float(joint),
        receiver_gain_db=float(receiver_gain),
        provider_gain_db=float(provider_gain),
        receiver_before_rmse_db=float(receiver_before),
        receiver_after_rmse_db=float(receiver_after),
        provider_before_rmse_db=float(provider_before),
        provider_after_rmse_db=float(provider_after),
        duplicate_lineage=bool(duplicate),
    )
