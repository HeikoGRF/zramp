#!/usr/bin/env python3
"""Encoding-only support acquisition in one shared normalized map frame."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.support_acquisition_pretraining.scalar_gain_model import (
    PlaneSetEncoder,
)


SHARED_FRAME_PLANE_FEATURE_SCHEMA = (
    "midpoint_x_unit_square",
    "midpoint_y_unit_square",
    "axis_cos_2theta",
    "axis_sin_2theta",
    "length_unit_square",
    "low_start_unit_square",
    "high_start_unit_square",
    "low_end_unit_square",
    "high_end_unit_square",
    "log1p_sample_count",
    "max_link_length_unit_square",
    "angle_spread_fraction_pi",
)


def normalize_plane_set_shared_frame(
    rows: np.ndarray,
    *,
    map_size: float,
) -> np.ndarray:
    """Represent planes in a common square map frame without canonicalizing a set."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    if len(values) == 0:
        return np.empty((0, len(SHARED_FRAME_PLANE_FEATURE_SCHEMA)), dtype=np.float32)
    scale = float(map_size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_size must be finite and positive")
    start, end = values[:, 0:2], values[:, 2:4]
    midpoint = 0.5 * (start + end)
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-9)
    axis = vector / length[:, None]
    features = np.column_stack((
        midpoint / scale,
        axis[:, 0] ** 2 - axis[:, 1] ** 2,
        2.0 * axis[:, 0] * axis[:, 1],
        length / scale,
        values[:, 4:8] / scale,
        np.log1p(np.maximum(values[:, 8], 0.0)),
        values[:, 9] / scale,
        np.deg2rad(values[:, 10]) / math.pi,
    )).astype(np.float32)
    return features


class EncodingOnlyGainModel(nn.Module):
    """Predict one nonnegative log-gain from candidate and bank encodings."""

    def __init__(self, *, latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gain_head = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        set_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> torch.Tensor:
        candidate = set_embeddings[candidate_indices]
        bank = set_embeddings[bank_indices]
        raw_gain = self.gain_head(torch.cat((candidate, bank), dim=1)).squeeze(1)
        return F.softplus(raw_gain)


__all__ = [
    "EncodingOnlyGainModel",
    "PlaneSetEncoder",
    "SHARED_FRAME_PLANE_FEATURE_SCHEMA",
    "normalize_plane_set_shared_frame",
]
