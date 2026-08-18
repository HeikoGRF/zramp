#!/usr/bin/env python3
"""Minimal permutation-invariant encoder and scalar acquisition model."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


SCALAR_PLANE_FEATURE_SCHEMA = (
    "midpoint_x_centered_rms",
    "midpoint_y_centered_rms",
    "axis_cos_2theta",
    "axis_sin_2theta",
    "length_rms",
    "low_start_rms",
    "high_start_rms",
    "low_end_rms",
    "high_end_rms",
    "log1p_sample_count",
    "max_link_length_rms",
    "angle_spread_fraction_pi",
)


class PlaneSetEncoder(nn.Module):
    """Encode any non-empty unordered plane set using mean/max pooling."""

    def __init__(
        self,
        *,
        feature_dim: int = len(SCALAR_PLANE_FEATURE_SCHEMA),
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.plane_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        plane_to_expert: torch.Tensor,
        expert_count: int,
    ) -> torch.Tensor:
        values = self.plane_mlp(features)
        indices = plane_to_expert[:, None].expand(-1, values.shape[1])
        total = values.new_zeros((int(expert_count), values.shape[1]))
        total.scatter_add_(0, indices, values)
        counts = values.new_zeros((int(expert_count), 1))
        counts.scatter_add_(
            0,
            plane_to_expert[:, None],
            values.new_ones((values.shape[0], 1)),
        )
        mean = total / counts.clamp_min(1.0)
        maximum = values.new_full(
            (int(expert_count), values.shape[1]), -torch.inf
        )
        maximum.scatter_reduce_(
            0, indices, values, reduce="amax", include_self=True
        )
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        return self.output_mlp(torch.cat((mean, maximum), dim=1))


class ScalarGainModel(nn.Module):
    """Predict one gain from a candidate encoding and an unordered bank."""

    def __init__(
        self, *, latent_dim: int = 64, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.bank_token_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(
                self.latent_dim + 2 * self.hidden_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        expert_embeddings: torch.Tensor,
        centers: torch.Tensor,
        scales: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate = expert_embeddings[candidate_indices]
        candidate_center = centers[candidate_indices]
        candidate_scale = scales[candidate_indices].clamp_min(1.0e-6)
        safe_bank = bank_indices.clamp_min(0)
        bank = expert_embeddings[safe_bank]
        bank_center = centers[safe_bank]
        bank_scale = scales[safe_bank].clamp_min(1.0e-6)

        relative_scale = torch.sqrt(
            bank_scale * candidate_scale[:, None]
        ).clamp_min(1.0e-6)
        relative_center = (
            bank_center - candidate_center[:, None, :]
        ) / relative_scale[:, :, None]
        log_scale_ratio = torch.log(
            bank_scale / candidate_scale[:, None]
        )
        tokens = self.bank_token_mlp(torch.cat((
            bank,
            relative_center,
            log_scale_ratio[:, :, None],
        ), dim=2))
        mask = bank_mask[:, :, None]
        mean = torch.where(mask, tokens, 0.0).sum(dim=1)
        mean = mean / mask.sum(dim=1).clamp_min(1)
        maximum = torch.where(mask, tokens, -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        return self.gain_head(
            torch.cat((candidate, mean, maximum), dim=1)
        ).squeeze(1)


def normalize_plane_set(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return scale-free features plus physical centroid and RMS scale."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    if len(values) == 0:
        raise ValueError("an expert must contain at least one plane")
    start, end = values[:, 0:2], values[:, 2:4]
    midpoint = 0.5 * (start + end)
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-9)
    axis = vector / length[:, None]
    center = midpoint.mean(axis=0)
    widths = np.maximum(
        values[:, 5] - values[:, 4],
        values[:, 7] - values[:, 6],
    )
    scale = float(np.sqrt(np.mean(
        np.sum((midpoint - center) ** 2, axis=1)
        + (0.5 * length) ** 2
        + (0.5 * widths) ** 2
    )))
    scale = max(scale, np.finfo(np.float64).eps)
    features = np.column_stack((
        (midpoint - center) / scale,
        axis[:, 0] ** 2 - axis[:, 1] ** 2,
        2.0 * axis[:, 0] * axis[:, 1],
        length / scale,
        values[:, 4:8] / scale,
        np.log1p(np.maximum(values[:, 8], 0.0)),
        values[:, 9] / scale,
        np.deg2rad(values[:, 10]) / math.pi,
    )).astype(np.float32)
    return features, center.astype(np.float32), scale
