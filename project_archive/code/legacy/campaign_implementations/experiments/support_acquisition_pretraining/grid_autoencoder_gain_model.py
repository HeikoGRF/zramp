#!/usr/bin/env python3
"""Scale-separated full-grid autoencoder and scalar acquisition model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridAutoencoder(nn.Module):
    """Encode an exact 300x300 intensity field without early spatial pooling.

    The total grid mass is transmitted explicitly as the final encoding value.
    The convolutional latent therefore only has to describe the normalized
    spatial distribution of that mass.  The transform applied to the
    distribution is invertible and depends only on the number of grid points.
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 300,
        latent_dim: int = 512,
        base_channels: int = 8,
    ) -> None:
        super().__init__()
        if int(grid_resolution) != 300:
            raise ValueError("the scale-separated encoder currently requires 300x300")
        if int(latent_dim) < 1 or int(base_channels) < 1:
            raise ValueError("latent dimensions and channels must be positive")
        self.grid_resolution = int(grid_resolution)
        self.grid_points = self.grid_resolution**2
        self.latent_dim = int(latent_dim)
        self.advertisement_dim = self.latent_dim + 1
        channels = int(base_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, channels, 5, stride=2, padding=2),       # 300 -> 150
            nn.SiLU(),
            nn.Conv2d(channels, 2 * channels, 4, stride=2, padding=1),  # 150 -> 75
            nn.SiLU(),
            nn.Conv2d(2 * channels, 3 * channels, 5, stride=3, padding=1),  # 75 -> 25
            nn.SiLU(),
            nn.Conv2d(3 * channels, 4 * channels, 5, stride=5),  # 25 -> 5
            nn.SiLU(),
        )
        self.bottleneck_shape = (4 * channels, 5, 5)
        bottleneck_values = math.prod(self.bottleneck_shape)
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_values, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, bottleneck_values),
            nn.SiLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(4 * channels, 3 * channels, 5, stride=5),  # 5 -> 25
            nn.SiLU(),
            nn.ConvTranspose2d(3 * channels, 2 * channels, 3, stride=3),  # 25 -> 75
            nn.SiLU(),
            nn.ConvTranspose2d(
                2 * channels, channels, 4, stride=2, padding=1
            ),  # 75 -> 150
            nn.SiLU(),
            nn.ConvTranspose2d(
                channels, 1, 4, stride=2, padding=1
            ),  # 150 -> 300
        )
        # The decoder is used only during pretraining.  A small linear output
        # avoids both a dense sigmoid background and saturation at zero.
        nn.init.normal_(self.decoder_conv[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.decoder_conv[-1].bias)

    def split_profile(
        self, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an invertibly transformed mass distribution and log mass."""

        raw = profiles.float()
        if raw.ndim == 3:
            raw = raw.unsqueeze(1)
        if raw.ndim != 4 or tuple(raw.shape[-2:]) != (
            self.grid_resolution,
            self.grid_resolution,
        ):
            raise ValueError("profiles must have shape [N,300,300] or [N,1,300,300]")
        raw = raw.clamp_min(0.0)
        mass = raw.sum(dim=(1, 2, 3))
        distribution = raw / mass[:, None, None, None].clamp_min(1.0)
        transformed = torch.log1p(float(self.grid_points) * distribution)
        transformed = transformed / math.log1p(float(self.grid_points))
        return transformed, torch.log1p(mass)

    def encode_shape(self, transformed: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_conv(transformed))

    def decode_shape(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.from_latent(latent).reshape(-1, *self.bottleneck_shape)
        return self.decoder_conv(hidden)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        transformed, log_mass = self.split_profile(profiles)
        latent = self.encode_shape(transformed)
        return torch.cat((latent, log_mass[:, None]), dim=1)

    def reconstruct(
        self, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed, log_mass = self.split_profile(profiles)
        latent = self.encode_shape(transformed)
        reconstructed = self.decode_shape(latent)
        return reconstructed, transformed, log_mass


class GridEncodingGainModel(nn.Module):
    """Predict log1p relative gain from candidate and bank encodings."""

    def __init__(
        self,
        *,
        latent_dim: int = 512,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.advertisement_dim = self.latent_dim + 1
        pair_dim = 5 * self.latent_dim
        mass_dim = 3
        self.spatial_head = nn.Sequential(
            nn.Linear(pair_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.mass_head = nn.Sequential(
            nn.Linear(mass_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(int(hidden_dim) + 32, max(64, int(hidden_dim) // 2)),
            nn.SiLU(),
            nn.Linear(max(64, int(hidden_dim) // 2), 1),
        )

    def forward(
        self,
        set_embeddings: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> torch.Tensor:
        candidate = set_embeddings[candidate_indices]
        bank = set_embeddings[bank_indices]
        candidate_shape, candidate_mass = (
            candidate[:, : self.latent_dim], candidate[:, self.latent_dim]
        )
        bank_shape, bank_mass = (
            bank[:, : self.latent_dim], bank[:, self.latent_dim]
        )
        spatial = self.spatial_head(torch.cat((
            candidate_shape,
            bank_shape,
            candidate_shape - bank_shape,
            torch.abs(candidate_shape - bank_shape),
            candidate_shape * bank_shape,
        ), dim=1))
        mass = self.mass_head(torch.stack((
            candidate_mass,
            bank_mass,
            candidate_mass - bank_mass,
        ), dim=1))
        return F.softplus(self.output(torch.cat((spatial, mass), dim=1)).squeeze(1))


def normalized_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-profile squared error relative to target field energy."""

    error = torch.square(reconstruction - target).sum(dim=(1, 2, 3))
    energy = torch.square(target).sum(dim=(1, 2, 3))
    relative = error / energy.clamp_min(1.0e-12)
    # Relative error is undefined for an empty profile.  For exactly empty
    # support, use ordinary per-point MSE so the correct zero field remains
    # the optimum without creating an unbounded loss.
    empty = error / float(target.shape[1] * target.shape[2] * target.shape[3])
    return torch.where(energy > 0.0, relative, empty).mean()


__all__ = [
    "GridAutoencoder",
    "GridEncodingGainModel",
    "normalized_reconstruction_loss",
]
