"""Tiny learned provider policy for bounded predictor expert banks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .online_local_validation_policy import ExactModelTrajectoryPolicy, ExactPrivateState
from .expert_bank import ExpertBank, PredictorExpert, SupportProfile


def profile_vector(profile: SupportProfile) -> torch.Tensor:
    """Serialize aggregate support metadata into a stable float vector."""

    values = [
        *profile.mean,
        *profile.scale,
        *profile.lower,
        *profile.upper,
        profile.radius_squared / max(1.0, float(profile.dimension)),
        np.log1p(profile.count) / 10.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def support_vector(features: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return the compact support metadata already transmitted with an expert."""

    profile = SupportProfile.fit(features, time_scale=1800.0)
    return profile_vector(profile)


def empty_support_vector(feature_dim: int) -> torch.Tensor:
    """Return the no-private-support marker for a 4-D/5-D link input."""

    dimension = int(feature_dim)
    if dimension not in {4, 5}:
        raise ValueError("support features need four coordinates and optional time")
    return torch.zeros(4 * dimension + 2, dtype=torch.float32)


def bank_support_vector(bank: ExpertBank, *, feature_dim: int) -> torch.Tensor:
    """Summarize a bounded bank without revealing any expert's raw samples."""

    if not bank.experts:
        return empty_support_vector(feature_dim)
    profiles = [expert.support for expert in bank.experts]
    dimension = profiles[0].dimension
    if any(profile.dimension != dimension for profile in profiles):
        raise ValueError("bank support dimensions differ")
    weights = np.asarray(
        [max(1, int(profile.count)) for profile in profiles], dtype=np.float64
    )
    weights /= float(np.sum(weights))
    means = np.asarray([profile.mean for profile in profiles], dtype=np.float64)
    scales = np.asarray([profile.scale for profile in profiles], dtype=np.float64)
    mean = np.sum(weights[:, None] * means, axis=0)
    variance = np.sum(
        weights[:, None] * (np.square(scales) + np.square(means - mean)),
        axis=0,
    )
    aggregate = SupportProfile(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in np.sqrt(np.maximum(variance, 1.0e-12))),
        lower=tuple(
            float(value)
            for value in np.min(
                np.asarray([profile.lower for profile in profiles]), axis=0
            )
        ),
        upper=tuple(
            float(value)
            for value in np.max(
                np.asarray([profile.upper for profile in profiles]), axis=0
            )
        ),
        radius_squared=float(
            np.sum(
                weights
                * np.asarray(
                    [profile.radius_squared for profile in profiles],
                    dtype=np.float64,
                )
            )
        ),
        count=int(sum(profile.count for profile in profiles)),
        time_scale=float(profiles[0].time_scale),
    )
    return profile_vector(aggregate)


class SupportAugmentedExactPolicy(nn.Module):
    """Encode exact predictor/trajectory state and compact support metadata.

    Support is not a third model or an extra measurement payload: it is the
    same aggregate profile used by :class:`ExpertBank` for inference routing.
    """

    def __init__(
        self,
        *,
        group_widths: tuple[int, ...],
        trajectory_dim: int,
        support_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        gain_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.pair_feature_mode = "relational"
        self.exact = ExactModelTrajectoryPolicy(
            group_widths=group_widths,
            trajectory_dim=trajectory_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            gain_hidden_dim=gain_hidden_dim,
            pair_feature_mode="relational",
        )
        support_hidden = max(4, int(hidden_dim))
        self.support_encoder = nn.Sequential(
            nn.LayerNorm(int(support_dim)),
            nn.Linear(int(support_dim), support_hidden),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(embedding_dim) + support_hidden, self.embedding_dim),
            nn.Tanh(),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(5 * self.embedding_dim, int(gain_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(gain_hidden_dim), 1),
        )

    def encode_many(
        self,
        states: list[ExactPrivateState],
        supports: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if supports is None:
            supports = [state.support for state in states]
            if any(support is None for support in supports):
                raise ValueError("support-augmented states require support metadata")
            supports = [support for support in supports if support is not None]
        if len(states) != len(supports):
            raise ValueError("exact states and support encodings must align")
        exact = self.exact.encode_many(states)
        if not supports:
            return exact
        metadata = self.support_encoder(
            torch.stack(supports).to(exact.device)
        )
        return self.fusion(torch.cat((exact, metadata), dim=-1))

    def encode(self, state: ExactPrivateState) -> torch.Tensor:
        return self.encode_many([state])[0]

    @property
    def trajectory_encoder(self) -> nn.GRU:
        """Compatibility view used by replacement-vehicle initialization."""

        return self.exact.trajectory_encoder

    @staticmethod
    def pair_features(
        receiver: torch.Tensor, provider: torch.Tensor
    ) -> torch.Tensor:
        difference = provider - receiver
        return torch.cat(
            (
                receiver,
                provider,
                difference,
                torch.abs(difference),
                receiver * provider,
            ),
            dim=-1,
        )

    def score_embeddings(
        self, receiver: torch.Tensor, provider: torch.Tensor
    ) -> torch.Tensor:
        return self.gain_head(
            self.pair_features(receiver, provider)
        ).squeeze(-1)


EXPERT_PULL_FEATURE_DIM = 9


def expert_pull_features(
    receiver_bank: ExpertBank,
    candidate: PredictorExpert,
    *,
    grid_points: int = 3,
) -> np.ndarray:
    """Explicit, stable context for decentralized provider learning."""

    candidate_cells = set(
        candidate.support.supported_coarse_cells(grid_points=int(grid_points))
    )
    receiver_cells = {
        cell
        for expert in receiver_bank.experts
        for cell in expert.support.supported_coarse_cells(
            grid_points=int(grid_points)
        )
    }
    novel = candidate_cells - receiver_cells
    overlap = candidate_cells & receiver_cells
    certificate_samples = sum(
        row.sample_count
        for row in candidate.cell_certificates
        if int(row.grid_points) == int(grid_points)
    )
    current = receiver_bank.expert_for_lineage(candidate.lineage_id)
    version_delta = (
        int(candidate.version)
        if current is None
        else max(0, int(candidate.version) - int(current.version))
    )
    return np.asarray(
        [
            1.0,
            np.clip(
                receiver_bank.certified_marginal_gain_db(
                    candidate, grid_points=int(grid_points)
                )
                / 30.0,
                -3.0,
                3.0,
            ),
            len(novel) / max(1, len(candidate_cells)),
            len(overlap) / max(1, len(candidate_cells)),
            np.log1p(max(0, int(candidate.experience))) / np.log1p(100_000),
            np.log1p(certificate_samples) / np.log1p(10_000),
            min(1.0, version_delta / 10.0),
            len(receiver_bank.experts) / max(1, receiver_bank.capacity),
            np.clip(candidate.external_utility_db() / 30.0, -3.0, 3.0),
        ],
        dtype=np.float64,
    )


class DecentralizedLinUCB:
    """Small per-vehicle contextual bandit with deduplicated online updates."""

    def __init__(
        self,
        *,
        feature_dim: int = EXPERT_PULL_FEATURE_DIM,
        regularization: float = 1.0,
        exploration: float = 0.35,
        reward_scale_db: float = 30.0,
        sample_capacity: int = 4096,
    ) -> None:
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if float(regularization) <= 0.0:
            raise ValueError("regularization must be positive")
        if float(exploration) < 0.0:
            raise ValueError("exploration cannot be negative")
        if float(reward_scale_db) <= 0.0:
            raise ValueError("reward_scale_db must be positive")
        self.feature_dim = int(feature_dim)
        self.exploration = float(exploration)
        self.reward_scale_db = float(reward_scale_db)
        self.sample_capacity = int(sample_capacity)
        self.a = float(regularization) * np.eye(
            self.feature_dim, dtype=np.float64
        )
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.sample_ids: set[str] = set()
        self.updates = 0

    def score(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float64).reshape(-1)
        if int(row.size) != self.feature_dim:
            raise ValueError("context feature dimension differs")
        inverse = np.linalg.inv(self.a)
        mean = float(row @ inverse @ self.b)
        uncertainty = float(np.sqrt(max(0.0, row @ inverse @ row)))
        return float(
            self.reward_scale_db
            * (mean + self.exploration * uncertainty)
        )

    def update(
        self,
        features: np.ndarray,
        reward_db: float,
        *,
        sample_id: str | None = None,
        propensity: float = 1.0,
    ) -> bool:
        if sample_id is not None and str(sample_id) in self.sample_ids:
            return False
        row = np.asarray(features, dtype=np.float64).reshape(-1)
        if int(row.size) != self.feature_dim:
            raise ValueError("context feature dimension differs")
        probability = max(1.0e-3, min(1.0, float(propensity)))
        weight = min(10.0, 1.0 / probability)
        target = np.clip(
            float(reward_db) / self.reward_scale_db, -3.0, 3.0
        )
        self.a += weight * np.outer(row, row)
        self.b += weight * row * target
        self.updates += 1
        if sample_id is not None and self.sample_capacity > 0:
            self.sample_ids.add(str(sample_id))
            if len(self.sample_ids) > self.sample_capacity:
                self.sample_ids = set(sorted(self.sample_ids)[-self.sample_capacity :])
        return True
