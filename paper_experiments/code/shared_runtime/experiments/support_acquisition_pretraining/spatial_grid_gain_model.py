#!/usr/bin/env python3
"""Map-aligned support-grid encoder and scalar acquisition model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGridEncoder(nn.Module):
    """Compress an exact support-intensity grid into an aligned advertisement."""

    def __init__(
        self,
        *,
        spatial_size: int = 16,
        learned_channels: int = 2,
        count_scale: float = 4096.0,
    ) -> None:
        super().__init__()
        self.spatial_size = int(spatial_size)
        self.learned_channels = int(learned_channels)
        self.count_scale = float(count_scale)
        self.latent_channels = 2 + self.learned_channels
        self.latent_dim = self.latent_channels * self.spatial_size ** 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 12, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, self.learned_channels, 3, stride=2, padding=1),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        raw = profiles.float().unsqueeze(1) if profiles.ndim == 3 else profiles.float()
        raw = raw.clamp_min(0.0)
        values = torch.log1p(raw) / math.log1p(self.count_scale)
        # Preserve cell-integrated intensity for an analytic coarse-gain base.
        average = F.adaptive_avg_pool2d(raw, self.spatial_size) / self.count_scale
        maximum = F.adaptive_max_pool2d(values, self.spatial_size)
        learned = F.adaptive_avg_pool2d(self.features(values), self.spatial_size)
        return torch.cat((average, maximum, learned), dim=1).flatten(1)


class SpatialGridGainModel(nn.Module):
    """Predict bounded log-relative gain from aligned grid advertisements."""

    def __init__(
        self,
        *,
        spatial_size: int = 16,
        latent_channels: int = 4,
        hidden_channels: int = 24,
        hidden_dim: int = 128,
        count_scale: float = 4096.0,
        maximum_relative_gain: float = 4.0,
    ) -> None:
        super().__init__()
        self.spatial_size = int(spatial_size)
        self.latent_channels = int(latent_channels)
        self.latent_dim = self.latent_channels * self.spatial_size ** 2
        self.maximum_relative_gain = float(maximum_relative_gain)
        self.count_scale = float(count_scale)
        pair_channels = 5 * self.latent_channels
        self.pair = nn.Sequential(
            nn.Conv2d(pair_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        set_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> torch.Tensor:
        candidate = set_embeddings[candidate_indices].reshape(
            -1, self.latent_channels, self.spatial_size, self.spatial_size
        )
        bank = set_embeddings[bank_indices].reshape_as(candidate)
        candidate_average = candidate[:, 0].clamp_min(0.0)
        bank_average = bank[:, 0].clamp_min(0.0)
        coarse_relative_gain = (
            torch.relu(candidate_average - bank_average).sum(dim=(1, 2))
            / bank_average.sum(dim=(1, 2)).clamp_min(1.0 / self.count_scale)
        )
        cap = math.log1p(self.maximum_relative_gain)
        coarse_log_gain = torch.log1p(
            coarse_relative_gain.clamp_max(self.maximum_relative_gain)
        )
        pair = self.pair(torch.cat((
            candidate,
            bank,
            candidate - bank,
            torch.abs(candidate - bank),
            candidate * bank,
        ), dim=1))
        mean = pair.mean(dim=(2, 3))
        maximum = pair.amax(dim=(2, 3))
        deviation = pair.var(dim=(2, 3), unbiased=False).sqrt()
        residual = cap * torch.tanh(
            self.head(torch.cat((mean, maximum, deviation), dim=1)).squeeze(1)
        )
        return torch.clamp(coarse_log_gain + residual, min=0.0, max=cap)


__all__ = ["SpatialGridEncoder", "SpatialGridGainModel"]
