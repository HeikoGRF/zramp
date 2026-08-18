#!/usr/bin/env python3
"""Spatially aligned patch codec for exact support-intensity grids."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.support_acquisition_pretraining.grid_autoencoder_gain_model import (
    normalized_reconstruction_loss,
)


class PatchGridCodec(nn.Module):
    """Encode every 10x10 grid patch into a small learned local code.

    The patch networks have no biases, so an empty support patch is encoded
    and decoded as exactly zero.  Patch positions are never pooled together.
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 300,
        patch_size: int = 10,
        latent_channels: int = 4,
        hidden_dim: int = 64,
        codebook_size: int = 0,
        codebook_groups: int = 1,
    ) -> None:
        super().__init__()
        self.grid_resolution = int(grid_resolution)
        self.patch_size = int(patch_size)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.codebook_groups = int(codebook_groups)
        if self.grid_resolution % self.patch_size:
            raise ValueError("patch size must divide the grid resolution")
        if min(self.patch_size, self.latent_channels, self.hidden_dim) < 1:
            raise ValueError("codec dimensions must be positive")
        if self.latent_channels % self.codebook_groups:
            raise ValueError("codebook groups must divide latent channels")
        self.codebook_subdim = self.latent_channels // self.codebook_groups
        self.patches_per_axis = self.grid_resolution // self.patch_size
        self.patch_count = self.patches_per_axis**2
        self.patch_values = self.patch_size**2
        self.grid_points = self.grid_resolution**2
        self.latent_dim = self.patch_count * self.latent_channels
        self.advertisement_dim = self.latent_dim + 1
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.patch_values, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_channels, bias=False),
        )
        self.patch_decoder = nn.Sequential(
            nn.Linear(self.latent_channels, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.patch_values, bias=False),
        )
        self.register_buffer(
            "codebook",
            torch.zeros(
                self.codebook_groups,
                self.codebook_size,
                self.codebook_subdim,
            ),
        )
        self.register_buffer("codebook_ready", torch.tensor(False))

    def split_profile(
        self, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = profiles.float()
        if raw.ndim == 3:
            raw = raw.unsqueeze(1)
        if raw.ndim != 4 or tuple(raw.shape[-2:]) != (
            self.grid_resolution,
            self.grid_resolution,
        ):
            raise ValueError("profiles have the wrong grid shape")
        raw = raw.clamp_min(0.0)
        mass = raw.sum(dim=(1, 2, 3))
        distribution = raw / mass[:, None, None, None].clamp_min(1.0)
        transformed = torch.log1p(float(self.grid_points) * distribution)
        transformed = transformed / math.log1p(float(self.grid_points))
        return transformed, torch.log1p(mass)

    def _patches(self, transformed: torch.Tensor) -> torch.Tensor:
        return F.unfold(
            transformed,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)

    def _fold(self, patches: torch.Tensor) -> torch.Tensor:
        return F.fold(
            patches.transpose(1, 2),
            output_size=(self.grid_resolution, self.grid_resolution),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def encode_shape(self, transformed: torch.Tensor) -> torch.Tensor:
        codes = self.patch_encoder(self._patches(transformed))
        if self.codebook_size and bool(self.codebook_ready):
            codes, _indices = self.quantize_codes(codes)
        return codes.flatten(1)

    def raw_patch_codes(self, transformed: torch.Tensor) -> torch.Tensor:
        return self.patch_encoder(self._patches(transformed))

    def set_codebook(self, centroids: torch.Tensor) -> None:
        values = torch.as_tensor(
            centroids, dtype=self.codebook.dtype, device=self.codebook.device
        )
        if tuple(values.shape) != tuple(self.codebook.shape):
            raise ValueError("codebook has the wrong shape")
        if self.codebook_size and bool(torch.any(values[:, 0] != 0.0)):
            raise ValueError("code zero must represent an empty patch")
        self.codebook.copy_(values)
        self.codebook_ready.fill_(True)

    def quantize_codes(
        self, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.codebook_size < 2:
            raise ValueError("quantization requires at least two codewords")
        active = torch.any(codes != 0.0, dim=2)
        flattened = codes.reshape(
            -1, self.codebook_groups, self.codebook_subdim
        )
        index_parts: list[torch.Tensor] = []
        quantized_parts: list[torch.Tensor] = []
        for group in range(self.codebook_groups):
            distances = torch.cdist(
                flattened[:, group], self.codebook[group, 1:]
            )
            group_indices = torch.argmin(distances, dim=1) + 1
            index_parts.append(group_indices)
            quantized_parts.append(F.embedding(
                group_indices, self.codebook[group]
            ))
        indices = torch.stack(index_parts, dim=1)
        indices = torch.where(
            active.reshape(-1, 1), indices, torch.zeros_like(indices)
        ).reshape(*codes.shape[:2], self.codebook_groups)
        quantized = torch.cat(quantized_parts, dim=1)
        quantized = torch.where(
            active.reshape(-1, 1), quantized, torch.zeros_like(quantized)
        ).reshape_as(codes)
        return quantized, indices

    def advertisement_indices(
        self, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed, log_mass = self.split_profile(profiles)
        raw = self.raw_patch_codes(transformed)
        _quantized, indices = self.quantize_codes(raw)
        return indices, log_mass

    def decode_shape(self, latent: torch.Tensor) -> torch.Tensor:
        codes = latent.reshape(-1, self.patch_count, self.latent_channels)
        return self._fold(self.patch_decoder(codes))

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        transformed, log_mass = self.split_profile(profiles)
        latent = self.encode_shape(transformed)
        return torch.cat((latent, log_mass[:, None]), dim=1)

    def reconstruct(
        self, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed, log_mass = self.split_profile(profiles)
        reconstruction = self.decode_shape(self.encode_shape(transformed))
        return reconstruction, transformed, log_mass

    def decode_advertisements(self, advertisements: torch.Tensor) -> torch.Tensor:
        """Decode advertisements into nonnegative grids with exact total mass."""

        latent = advertisements[:, : self.latent_dim]
        log_mass = advertisements[:, self.latent_dim]
        transformed = self.decode_shape(latent).clamp(0.0, 1.0)
        weights = torch.expm1(
            transformed[:, 0] * math.log1p(float(self.grid_points))
        ) / float(self.grid_points)
        distribution = weights / weights.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1.0e-12)
        return distribution * torch.expm1(log_mass)[:, None, None]

    def active_patch_counts(self, advertisements: torch.Tensor) -> torch.Tensor:
        codes = advertisements[:, : self.latent_dim].reshape(
            -1, self.patch_count, self.latent_channels
        )
        return torch.any(codes != 0.0, dim=2).sum(dim=1)


class PatchGridGainModel(nn.Module):
    """Predict one gain by comparing aligned candidate/bank patch codes."""

    def __init__(
        self,
        *,
        patch_count: int = 900,
        latent_channels: int = 16,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.patch_count = int(patch_count)
        self.latent_channels = int(latent_channels)
        self.latent_dim = self.patch_count * self.latent_channels
        pair_values = 5 * self.latent_channels + 6
        self.patch_head = nn.Sequential(
            nn.Linear(pair_values, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.output = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -8.0)

    def forward(
        self,
        set_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> torch.Tensor:
        candidate = set_embeddings[candidate_indices]
        bank = set_embeddings[bank_indices]
        candidate_codes = candidate[:, : self.latent_dim].reshape(
            -1, self.patch_count, self.latent_channels
        )
        bank_codes = bank[:, : self.latent_dim].reshape_as(candidate_codes)
        candidate_mass = candidate[:, self.latent_dim]
        bank_mass = bank[:, self.latent_dim]
        masses = torch.stack((
            candidate_mass,
            bank_mass,
            candidate_mass - bank_mass,
        ), dim=1)[:, None, :].expand(-1, self.patch_count, -1)
        candidate_active = torch.any(candidate_codes != 0.0, dim=2).float()
        bank_active = torch.any(bank_codes != 0.0, dim=2).float()
        activity = torch.stack((
            candidate_active,
            bank_active,
            candidate_active * (1.0 - bank_active),
        ), dim=2)
        features = self.patch_head(torch.cat((
            candidate_codes,
            bank_codes,
            candidate_codes - bank_codes,
            torch.abs(candidate_codes - bank_codes),
            candidate_codes * bank_codes,
            masses,
            activity,
        ), dim=2))
        raw_contribution = self.output(features).squeeze(2)
        contribution = F.softplus(raw_contribution) * candidate_active
        return torch.log1p(contribution.sum(dim=1))


__all__ = [
    "PatchGridCodec",
    "PatchGridGainModel",
    "normalized_reconstruction_loss",
]
