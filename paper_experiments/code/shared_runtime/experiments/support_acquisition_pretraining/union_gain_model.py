#!/usr/bin/env python3
"""Simpler scalar acquisition from candidate and bank-union encodings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.support_acquisition_pretraining.scalar_gain_model import (
    PlaneSetEncoder,
    SCALAR_PLANE_FEATURE_SCHEMA,
    normalize_plane_set,
)


class UnionGainModel(nn.Module):
    """Predict nonnegative gain from two unordered plane-set encodings."""

    def __init__(
        self, *, latent_dim: int = 64, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gain_head = nn.Sequential(
            nn.Linear(2 * self.latent_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        set_embeddings: torch.Tensor,
        centers: torch.Tensor,
        scales: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> torch.Tensor:
        candidate = set_embeddings[candidate_indices]
        bank = set_embeddings[bank_indices]
        candidate_center = centers[candidate_indices]
        bank_center = centers[bank_indices]
        candidate_scale = scales[candidate_indices].clamp_min(1.0e-6)
        bank_scale = scales[bank_indices].clamp_min(1.0e-6)
        common_scale = torch.sqrt(candidate_scale * bank_scale).clamp_min(
            1.0e-6
        )
        relative_center = (
            bank_center - candidate_center
        ) / common_scale[:, None]
        log_scale_ratio = torch.log(bank_scale / candidate_scale)
        raw_gain = self.gain_head(torch.cat((
            candidate,
            bank,
            relative_center,
            log_scale_ratio[:, None],
        ), dim=1)).squeeze(1)
        return F.softplus(raw_gain)


__all__ = [
    "PlaneSetEncoder",
    "SCALAR_PLANE_FEATURE_SCHEMA",
    "UnionGainModel",
    "normalize_plane_set",
]
