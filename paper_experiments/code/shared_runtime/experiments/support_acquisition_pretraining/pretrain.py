#!/usr/bin/env python3
"""Map-independent synthetic pretraining for support-driven acquisition."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/support_acquisition_pretraining/synthetic_v1"
PLANE_FEATURE_SCHEMA = (
    "midpoint_x_centered_scaled",
    "midpoint_y_centered_scaled",
    "axis_cos_2theta",
    "axis_sin_2theta",
    "length_scaled",
    "low_start_scaled",
    "high_start_scaled",
    "low_end_scaled",
    "high_end_scaled",
    "support_maturity",
    "max_link_length_scaled",
    "angle_spread_scaled_90deg",
)

PLANE_FEATURE_SCHEMA_V2 = (
    "midpoint_x_centered_rms_bounded",
    "midpoint_y_centered_rms_bounded",
    "axis_cos_2theta",
    "axis_sin_2theta",
    "length_rms_bounded",
    "low_start_rms_bounded",
    "high_start_rms_bounded",
    "low_end_rms_bounded",
    "high_end_rms_bounded",
    "support_maturity_centered",
    "max_link_length_rms_bounded",
    "angle_spread_centered",
)


@dataclass(frozen=True)
class SyntheticConfig:
    min_world_m: float = 80.0
    max_world_m: float = 600.0
    min_axes: int = 6
    max_axes: int = 24
    min_planes: int = 1
    max_planes: int = 128
    min_bank_size: int = 1
    max_bank_size: int = 12
    candidates_per_bank: int = 4
    queries_per_world: int = 384
    mass_scale: float = 3.0
    normalization_version: str = "v1"


@dataclass
class AcquisitionBatch:
    plane_features: torch.Tensor
    plane_to_expert: torch.Tensor
    expert_centers: torch.Tensor
    expert_scales: torch.Tensor
    candidate_indices: torch.Tensor
    bank_indices: torch.Tensor
    bank_mask: torch.Tensor
    targets: torch.Tensor
    positive_targets: torch.Tensor
    group_indices: torch.Tensor
    plane_counts: torch.Tensor

    def to(self, device: torch.device) -> "AcquisitionBatch":
        return AcquisitionBatch(**{
            name: value.to(device)
            for name, value in self.__dict__.items()
        })


class PlaneSetEncoder(nn.Module):
    """Permutation-invariant encoder for any positive number of planes."""

    def __init__(self, feature_dim: int = 12, hidden_dim: int = 64,
                 latent_dim: int = 32) -> None:
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

    def forward(self, features: torch.Tensor, plane_to_expert: torch.Tensor,
                expert_count: int) -> torch.Tensor:
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


class AcquisitionModel(nn.Module):
    """Predict marginal hard coverage, intensity, and any-gain probability."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 scale_invariant: bool = False) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.scale_invariant = bool(scale_invariant)
        pair_metadata = 3 if self.scale_invariant else 5
        context_metadata = 1 if self.scale_invariant else 2
        pair_dim = 2 * self.latent_dim + pair_metadata
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.latent_dim + context_metadata,
                      self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

    def forward(
        self,
        expert_embeddings: torch.Tensor,
        centers: torch.Tensor,
        scales: torch.Tensor,
        candidate_indices: torch.Tensor,
        bank_indices: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = expert_embeddings[candidate_indices]
        candidate_center = centers[candidate_indices]
        candidate_scale = scales[candidate_indices].clamp_min(1.0e-6)
        safe_bank = bank_indices.clamp_min(0)
        bank = expert_embeddings[safe_bank]
        bank_center = centers[safe_bank]
        bank_scale = scales[safe_bank].clamp_min(1.0e-6)
        repeated_candidate = candidate[:, None, :].expand_as(bank)
        relative_scale = torch.sqrt(
            bank_scale * candidate_scale[:, None]
        ).clamp_min(1.0e-6)
        relative_center = (
            bank_center - candidate_center[:, None, :]
        ) / relative_scale[:, :, None]
        relative_center = relative_center.clamp(-20.0, 20.0)
        log_scale_ratio = torch.log(
            bank_scale / candidate_scale[:, None]
        ).clamp(-8.0, 8.0)
        if self.scale_invariant:
            pair_features = torch.cat((
                bank,
                repeated_candidate,
                relative_center / 20.0,
                (log_scale_ratio / 8.0)[:, :, None],
            ), dim=2)
        else:
            pair_features = torch.cat((
                bank,
                repeated_candidate,
                relative_center,
                log_scale_ratio[:, :, None],
                torch.log(bank_scale)[:, :, None],
                torch.log(candidate_scale)[:, None, None].expand(
                    -1, bank.shape[1], -1
                ),
            ), dim=2)
        pair_values = self.pair_mlp(pair_features)
        mask = bank_mask[:, :, None]
        masked = torch.where(mask, pair_values, 0.0)
        mean = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        maximum = torch.where(mask, pair_values, -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        normalized_count = (
            torch.log1p(bank_mask.sum(dim=1).to(candidate.dtype))
            / math.log(21.0)
        )[:, None]
        if self.scale_invariant:
            context = torch.cat((
                candidate, normalized_count, mean, maximum,
            ), dim=1)
        else:
            context = torch.cat((
                candidate,
                torch.log(candidate_scale)[:, None],
                normalized_count,
                mean,
                maximum,
            ), dim=1)
        raw = self.head(context)
        return torch.sigmoid(raw[:, :2]), raw[:, 2]


def _log_uniform_int(
    rng: np.random.Generator, low: int, high: int
) -> int:
    value = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    return int(np.clip(value, low, high))


def _make_axes(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    count = int(rng.integers(cfg.min_axes, cfg.max_axes + 1))
    world = float(math.exp(rng.uniform(
        math.log(cfg.min_world_m), math.log(cfg.max_world_m)
    )))
    centers = rng.uniform(-0.45 * world, 0.45 * world, size=(count, 2))
    angles = rng.uniform(0.0, math.pi, size=count)
    lengths = rng.uniform(0.12 * world, 0.75 * world, size=count)
    widths = rng.uniform(2.0, min(12.0, 0.06 * world), size=count)
    return np.column_stack((centers, angles, lengths, widths))


def _choose_focus(
    rng: np.random.Generator,
    axis_count: int,
    preferred: np.ndarray | None = None,
) -> np.ndarray:
    maximum = min(axis_count, 8)
    count = int(rng.integers(1, maximum + 1))
    if preferred is not None and len(preferred):
        pool = np.asarray(preferred, dtype=np.int64)
        replace_axes = len(pool) < count
        return np.asarray(
            rng.choice(pool, size=count, replace=replace_axes), dtype=np.int64
        )
    return np.asarray(
        rng.choice(axis_count, size=count, replace=False), dtype=np.int64
    )


def _make_expert(
    rng: np.random.Generator,
    axes: np.ndarray,
    cfg: SyntheticConfig,
    *,
    preferred: np.ndarray | None = None,
) -> np.ndarray:
    plane_count = _log_uniform_int(rng, cfg.min_planes, cfg.max_planes)
    focus = _choose_focus(rng, len(axes), preferred)
    rows = np.empty((plane_count, 11), dtype=np.float64)
    for index in range(plane_count):
        axis_index = int(rng.choice(focus))
        cx, cy, theta, axis_length, road_half_width = axes[axis_index]
        direction = np.asarray([math.cos(theta), math.sin(theta)])
        normal = np.asarray([-direction[1], direction[0]])
        segment_center = (
            np.asarray([cx, cy])
            + rng.uniform(-0.42, 0.42) * axis_length * direction
            + rng.normal(0.0, 0.35 * road_half_width) * normal
        )
        segment_length = float(np.clip(
            rng.lognormal(math.log(max(5.0, 0.28 * axis_length)), 0.55),
            3.0,
            max(4.0, 0.95 * axis_length),
        ))
        local_theta = float(theta + rng.normal(0.0, math.radians(6.0)))
        local_direction = np.asarray(
            [math.cos(local_theta), math.sin(local_theta)]
        )
        start = segment_center - 0.5 * segment_length * local_direction
        end = segment_center + 0.5 * segment_length * local_direction
        if end[0] < start[0] or (
            abs(float(end[0] - start[0])) < 1.0e-12 and end[1] < start[1]
        ):
            start, end = end, start
        width = float(rng.uniform(1.0, max(1.1, road_half_width)))
        low_start = -width * float(rng.uniform(0.65, 1.25))
        high_start = width * float(rng.uniform(0.65, 1.25))
        low_end = -width * float(rng.uniform(0.65, 1.25))
        high_end = width * float(rng.uniform(0.65, 1.25))
        mass = float(rng.integers(1, 25))
        max_link_length = segment_length * float(rng.uniform(0.25, 1.0))
        rows[index] = (
            start[0], start[1], end[0], end[1],
            low_start, high_start, low_end, high_end,
            mass, max_link_length, rng.uniform(0.0, 18.0),
        )
    return rows


def _sample_queries(
    rng: np.random.Generator, axes: np.ndarray, count: int
) -> np.ndarray:
    queries = np.empty((count, 2, 2), dtype=np.float64)

    def point(axis_index: int) -> np.ndarray:
        cx, cy, theta, length, half_width = axes[axis_index]
        direction = np.asarray([math.cos(theta), math.sin(theta)])
        normal = np.asarray([-direction[1], direction[0]])
        return (
            np.asarray([cx, cy])
            + rng.uniform(-0.5, 0.5) * length * direction
            + rng.normal(0.0, 0.5 * half_width) * normal
        )

    for index in range(count):
        left = int(rng.integers(len(axes)))
        right = left if rng.random() < 0.7 else int(rng.integers(len(axes)))
        queries[index, 0] = point(left)
        queries[index, 1] = point(right)
    return queries


def support_intensity(
    rows: np.ndarray, queries: np.ndarray, mass_scale: float
) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(len(queries), dtype=np.float64)
    start, end = rows[:, 0:2], rows[:, 2:4]
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-9)
    axis = vector / length[:, None]
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    midpoint = 0.5 * (start + end)
    relative = queries[:, :, None, :] - midpoint[None, None, :, :]
    along = np.einsum("qepc,pc->qep", relative, axis)
    lateral = np.einsum("qepc,pc->qep", relative, normal)
    fraction = np.clip(
        (along + 0.5 * length[None, None, :]) / length[None, None, :],
        0.0,
        1.0,
    )
    low = (
        (1.0 - fraction) * rows[None, None, :, 4]
        + fraction * rows[None, None, :, 6]
    )
    high = (
        (1.0 - fraction) * rows[None, None, :, 5]
        + fraction * rows[None, None, :, 7]
    )
    query_length = np.linalg.norm(queries[:, 1] - queries[:, 0], axis=1)
    supported = (
        ((lateral >= low) & (lateral <= high)).all(axis=1)
        & (np.abs(along) <= 0.5 * length[None, None, :]).all(axis=1)
        & (query_length[:, None] <= rows[None, :, 9])
    )
    maturity = 1.0 - np.exp(-rows[:, 8] / float(mass_scale))
    return np.max(np.where(supported, maturity[None, :], 0.0), axis=1)


def normalize_expert(
    rows: np.ndarray, mass_scale: float, normalization_version: str = "v1"
) -> tuple[np.ndarray, np.ndarray, float]:
    start, end = rows[:, 0:2], rows[:, 2:4]
    midpoint = 0.5 * (start + end)
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-6)
    axis = vector / length[:, None]
    center = midpoint.mean(axis=0)
    widths = np.maximum(
        rows[:, 5] - rows[:, 4], rows[:, 7] - rows[:, 6]
    )
    scale = float(np.sqrt(np.mean(
        np.sum((midpoint - center) ** 2, axis=1)
        + (0.5 * length) ** 2
        + (0.5 * widths) ** 2
    )))
    scale = max(scale, 1.0)
    features = np.column_stack((
        (midpoint - center) / scale,
        axis[:, 0] ** 2 - axis[:, 1] ** 2,
        2.0 * axis[:, 0] * axis[:, 1],
        length / scale,
        rows[:, 4:8] / scale,
        1.0 - np.exp(-rows[:, 8] / float(mass_scale)),
        rows[:, 9] / scale,
        rows[:, 10] / 90.0,
    )).astype(np.float32)
    if normalization_version == "v2":
        features[:, 0:2] = np.clip(features[:, 0:2] / 3.0, -1.0, 1.0)
        features[:, 4] = (
            2.0 * np.clip(features[:, 4] / 4.0, 0.0, 1.0) - 1.0
        )
        features[:, 5:9] = np.clip(features[:, 5:9] / 2.0, -1.0, 1.0)
        features[:, 9] = 2.0 * features[:, 9] - 1.0
        features[:, 10] = (
            2.0 * np.clip(features[:, 10] / 4.0, 0.0, 1.0) - 1.0
        )
        features[:, 11] = (
            2.0 * np.clip(features[:, 11] * 5.0, 0.0, 1.0) - 1.0
        )
    elif normalization_version != "v1":
        raise ValueError(f"unknown normalization version: {normalization_version}")
    return features, center.astype(np.float32), scale


def _candidate_focus(
    rng: np.random.Generator,
    axis_count: int,
    bank_axes: np.ndarray,
) -> np.ndarray:
    unused = np.setdiff1d(np.arange(axis_count), bank_axes)
    mode = int(rng.integers(3))
    if mode == 0 or len(unused) == 0:
        return bank_axes
    if mode == 1:
        overlap_count = max(1, min(len(bank_axes), len(bank_axes) // 2))
        novel_count = max(1, min(len(unused), overlap_count))
        return np.concatenate((
            np.asarray(rng.choice(bank_axes, overlap_count, replace=False)),
            np.asarray(rng.choice(unused, novel_count, replace=False)),
        ))
    return unused


def make_batch(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    group_count: int,
) -> AcquisitionBatch:
    all_features: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scales: list[float] = []
    plane_to_expert: list[np.ndarray] = []
    candidate_indices: list[int] = []
    bank_rows_out: list[list[int]] = []
    targets: list[tuple[float, float]] = []
    positives: list[float] = []
    group_indices: list[int] = []
    plane_counts: list[int] = []
    expert_index = 0

    for group in range(int(group_count)):
        axes = _make_axes(rng, cfg)
        bank_size = int(rng.integers(
            cfg.min_bank_size, cfg.max_bank_size + 1
        ))
        bank_axis_count = int(rng.integers(1, len(axes) + 1))
        bank_axes = np.asarray(
            rng.choice(len(axes), bank_axis_count, replace=False),
            dtype=np.int64,
        )
        bank_rows = [
            _make_expert(rng, axes, cfg, preferred=bank_axes)
            for _ in range(bank_size)
        ]
        bank_indices = list(range(expert_index, expert_index + bank_size))
        for rows in bank_rows:
            features, center, scale = normalize_expert(rows, cfg.mass_scale, cfg.normalization_version)
            all_features.append(features)
            centers.append(center)
            scales.append(scale)
            plane_to_expert.append(
                np.full(len(rows), expert_index, dtype=np.int64)
            )
            plane_counts.append(len(rows))
            expert_index += 1
        queries = _sample_queries(rng, axes, cfg.queries_per_world)
        complete_bank = np.concatenate(bank_rows, axis=0)
        bank_support = support_intensity(
            complete_bank, queries, cfg.mass_scale
        )
        bank_hard = bank_support >= 0.5

        for _candidate in range(cfg.candidates_per_bank):
            preferred = _candidate_focus(
                rng, len(axes), bank_axes
            )
            rows = _make_expert(rng, axes, cfg, preferred=preferred)
            candidate_support = support_intensity(
                rows, queries, cfg.mass_scale
            )
            union = np.maximum(bank_support, candidate_support)
            hard_gain = float(np.mean((union >= 0.5) & ~bank_hard))
            intensity_gain = float(np.mean(union - bank_support))
            features, center, scale = normalize_expert(rows, cfg.mass_scale, cfg.normalization_version)
            all_features.append(features)
            centers.append(center)
            scales.append(scale)
            plane_to_expert.append(
                np.full(len(rows), expert_index, dtype=np.int64)
            )
            plane_counts.append(len(rows))
            candidate_indices.append(expert_index)
            bank_rows_out.append(bank_indices)
            targets.append((hard_gain, intensity_gain))
            positives.append(float(intensity_gain > 1.0e-8))
            group_indices.append(group)
            expert_index += 1

    max_bank = max(map(len, bank_rows_out))
    padded_bank = np.zeros((len(bank_rows_out), max_bank), dtype=np.int64)
    bank_mask = np.zeros((len(bank_rows_out), max_bank), dtype=bool)
    for row_index, values in enumerate(bank_rows_out):
        padded_bank[row_index, :len(values)] = values
        bank_mask[row_index, :len(values)] = True
    return AcquisitionBatch(
        plane_features=torch.from_numpy(np.concatenate(all_features)),
        plane_to_expert=torch.from_numpy(np.concatenate(plane_to_expert)),
        expert_centers=torch.from_numpy(np.stack(centers)),
        expert_scales=torch.from_numpy(np.asarray(scales, dtype=np.float32)),
        candidate_indices=torch.from_numpy(
            np.asarray(candidate_indices, dtype=np.int64)
        ),
        bank_indices=torch.from_numpy(padded_bank),
        bank_mask=torch.from_numpy(bank_mask),
        targets=torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        positive_targets=torch.from_numpy(
            np.asarray(positives, dtype=np.float32)
        ),
        group_indices=torch.from_numpy(
            np.asarray(group_indices, dtype=np.int64)
        ),
        plane_counts=torch.from_numpy(
            np.asarray(plane_counts, dtype=np.int64)
        ),
    )


def predict(
    encoder: PlaneSetEncoder,
    acquisition: AcquisitionModel,
    batch: AcquisitionBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = encoder(
        batch.plane_features,
        batch.plane_to_expert,
        int(batch.expert_centers.shape[0]),
    )
    return acquisition(
        embeddings,
        batch.expert_centers,
        batch.expert_scales,
        batch.candidate_indices,
        batch.bank_indices,
        batch.bank_mask,
    )


def ranking_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group in torch.unique(group_indices):
        mask = group_indices == group
        truth = targets[mask]
        if float((truth.max() - truth.min()).detach()) <= 1.0e-6:
            continue
        losses.append(F.cross_entropy(
            predicted[mask][None, :] / 0.05,
            truth.argmax()[None],
        ))
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean()


def loss_function(
    gains: torch.Tensor,
    logits: torch.Tensor,
    batch: AcquisitionBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = F.smooth_l1_loss(gains, batch.targets, beta=0.02)
    classification = F.binary_cross_entropy_with_logits(
        logits, batch.positive_targets
    )
    ranking = ranking_loss(
        gains[:, 1], batch.targets[:, 1], batch.group_indices
    )
    total = regression + 0.25 * classification + 0.25 * ranking
    return total, {
        "loss": float(total.detach()),
        "regression_loss": float(regression.detach()),
        "classification_loss": float(classification.detach()),
        "ranking_loss": float(ranking.detach()),
    }


@torch.no_grad()
def evaluate(
    encoder: PlaneSetEncoder,
    acquisition: AcquisitionModel,
    *,
    cfg: SyntheticConfig,
    seed: int,
    batches: int,
    groups_per_batch: int,
    device: torch.device,
) -> dict[str, float | int]:
    encoder.eval()
    acquisition.eval()
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    positive_targets: list[np.ndarray] = []
    top_correct = 0
    group_total = 0
    regret_sum = 0.0
    plane_min = sys.maxsize
    plane_max = 0
    plane_sum = 0
    plane_total = 0
    for _ in range(int(batches)):
        batch = make_batch(rng, cfg, groups_per_batch).to(device)
        gains, logits = predict(encoder, acquisition, batch)
        predicted_np = gains.detach().cpu().numpy()
        target_np = batch.targets.detach().cpu().numpy()
        predictions.append(predicted_np)
        targets.append(target_np)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        positive_targets.append(batch.positive_targets.cpu().numpy())
        counts = batch.plane_counts.cpu().numpy()
        plane_min = min(plane_min, int(counts.min()))
        plane_max = max(plane_max, int(counts.max()))
        plane_sum += int(counts.sum())
        plane_total += int(len(counts))
        groups = batch.group_indices.cpu().numpy()
        for group in np.unique(groups):
            mask = groups == group
            predicted_choice = int(np.argmax(predicted_np[mask, 1]))
            true_values = target_np[mask, 1]
            true_choice = int(np.argmax(true_values))
            top_correct += int(predicted_choice == true_choice)
            regret_sum += float(
                true_values[true_choice] - true_values[predicted_choice]
            )
            group_total += 1
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    positive = np.concatenate(positive_targets)
    selected_positive = probability >= 0.5
    true_positive = positive >= 0.5
    return {
        "examples": int(len(target)),
        "groups": int(group_total),
        "coverage_mae": float(np.mean(np.abs(prediction[:, 0] - target[:, 0]))),
        "intensity_mae": float(np.mean(np.abs(prediction[:, 1] - target[:, 1]))),
        "coverage_rmse": float(np.sqrt(np.mean((prediction[:, 0] - target[:, 0]) ** 2))),
        "intensity_rmse": float(np.sqrt(np.mean((prediction[:, 1] - target[:, 1]) ** 2))),
        "top1_accuracy": float(top_correct / max(1, group_total)),
        "mean_selection_regret": float(regret_sum / max(1, group_total)),
        "any_gain_accuracy": float(np.mean(selected_positive == true_positive)),
        "positive_fraction": float(np.mean(true_positive)),
        "plane_count_min": int(plane_min),
        "plane_count_max": int(plane_max),
        "plane_count_mean": float(plane_sum / max(1, plane_total)),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def self_test() -> None:
    cfg = SyntheticConfig(
        max_planes=24,
        max_bank_size=4,
        candidates_per_bank=3,
        queries_per_world=64,
    )
    batch = make_batch(np.random.default_rng(7), cfg, 3)
    assert int(batch.plane_counts.min()) < int(batch.plane_counts.max())
    encoder = PlaneSetEncoder()
    acquisition = AcquisitionModel()
    gains, logits = predict(encoder, acquisition, batch)
    assert gains.shape == (9, 2)
    assert logits.shape == (9,)
    loss, _ = loss_function(gains, logits, batch)
    loss.backward()
    assert torch.isfinite(loss)
    v2_cfg = replace(cfg, normalization_version="v2")
    v2_batch = make_batch(np.random.default_rng(8), v2_cfg, 2)
    assert float(v2_batch.plane_features.min()) >= -1.000001
    assert float(v2_batch.plane_features.max()) <= 1.000001
    v2_encoder = PlaneSetEncoder()
    v2_acquisition = AcquisitionModel(scale_invariant=True)
    v2_gains, v2_logits = predict(v2_encoder, v2_acquisition, v2_batch)
    assert v2_gains.shape == (6, 2)
    assert v2_logits.shape == (6,)
    v2_loss, _ = loss_function(v2_gains, v2_logits, v2_batch)
    v2_loss.backward()
    assert torch.isfinite(v2_loss)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--groups-per-batch", type=int, default=6)
    parser.add_argument("--validation-batches", type=int, default=24)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-planes", type=int, default=128)
    parser.add_argument("--max-bank-size", type=int, default=12)
    parser.add_argument("--queries-per-world", type=int, default=384)
    parser.add_argument("--candidates-per-bank", type=int, default=4)
    parser.add_argument("--min-world-m", type=float, default=80.0)
    parser.add_argument("--max-world-m", type=float, default=600.0)
    parser.add_argument("--max-axes", type=int, default=24)
    parser.add_argument(
        "--normalization-version", choices=("v1", "v2"), default="v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("synthetic acquisition pretraining self-test passed")
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cfg = SyntheticConfig(
        min_world_m=float(args.min_world_m),
        max_world_m=float(args.max_world_m),
        max_axes=int(args.max_axes),
        max_planes=int(args.max_planes),
        max_bank_size=int(args.max_bank_size),
        candidates_per_bank=int(args.candidates_per_bank),
        queries_per_world=int(args.queries_per_world),
        normalization_version=str(args.normalization_version),
    )
    validation_cfg = replace(
        cfg,
        min_world_m=min(60.0, 0.75 * cfg.min_world_m),
        max_world_m=max(750.0, 1.25 * cfg.max_world_m),
        max_axes=max(cfg.max_axes, 32),
        max_planes=max(cfg.max_planes, 320 if cfg.normalization_version == "v2" else 160),
        max_bank_size=max(cfg.max_bank_size, 20 if cfg.normalization_version == "v2" else 16),
    )
    encoder = PlaneSetEncoder(
        latent_dim=int(args.latent_dim), hidden_dim=int(args.hidden_dim)
    ).to(device)
    acquisition = AcquisitionModel(
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        scale_invariant=(args.normalization_version == "v2"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *acquisition.parameters()],
        lr=float(args.learning_rate),
        weight_decay=1.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.steps)),
        eta_min=0.05 * float(args.learning_rate),
    )
    feature_schema = (
        PLANE_FEATURE_SCHEMA_V2
        if args.normalization_version == "v2"
        else PLANE_FEATURE_SCHEMA
    )
    format_version = "v2" if args.normalization_version == "v2" else "v1"
    rng = np.random.default_rng(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_step = 0

    for step in range(1, int(args.steps) + 1):
        encoder.train()
        acquisition.train()
        batch = make_batch(rng, cfg, int(args.groups_per_batch)).to(device)
        gains, logits = predict(encoder, acquisition, batch)
        loss, loss_parts = loss_function(gains, logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *acquisition.parameters()], 5.0
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(args.validation_every) == 0 or step == int(args.steps):
            metrics = evaluate(
                encoder,
                acquisition,
                cfg=validation_cfg,
                seed=int(args.seed) + 100_000,
                batches=int(args.validation_batches),
                groups_per_batch=max(2, int(args.groups_per_batch) // 2),
                device=device,
            )
            validation_score = float(metrics["mean_selection_regret"])
            if args.normalization_version == "v2":
                validation_score = (
                    2.0 * float(metrics["mean_selection_regret"])
                    + 0.25 * float(metrics["coverage_mae"])
                    + 0.25 * float(metrics["intensity_mae"])
                )
            row: dict[str, float | int] = {
                "step": step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "checkpoint_score": validation_score,
                **loss_parts,
                **metrics,
            }
            history.append(row)
            print(
                f"[SYNTH-ACQ] step={step:04d}/{args.steps} "
                f"loss={loss_parts['loss']:.4f} "
                f"top1={100 * float(metrics['top1_accuracy']):.1f}% "
                f"regret={float(metrics['mean_selection_regret']):.5f} "
                f"intensity_mae={float(metrics['intensity_mae']):.5f} "
                f"planes={metrics['plane_count_min']}-"
                f"{metrics['plane_count_max']}",
                flush=True,
            )
            if validation_score < best_score:
                best_score = validation_score
                best_step = step
                torch.save(
                    {
                        "format": f"synthetic_plane_set_encoder_{format_version}",
                        "state_dict": encoder.state_dict(),
                        "feature_schema": feature_schema,
                        "latent_dim": int(args.latent_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "normalization": (
                            "per-expert centroid and RMS physical extent; "
                            "centroid and scale retained as acquisition metadata"
                        ),
                        "synthetic_config": asdict(cfg),
                    },
                    output / "encoder.pt",
                )
                torch.save(
                    {
                        "format": f"synthetic_support_acquisition_model_{format_version}",
                        "state_dict": acquisition.state_dict(),
                        "latent_dim": int(args.latent_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "scale_invariant": bool(args.normalization_version == "v2"),
                        "outputs": (
                            "marginal_hard_coverage",
                            "marginal_support_intensity",
                            "any_positive_gain_logit",
                        ),
                        "synthetic_config": asdict(cfg),
                    },
                    output / "acquisition.pt",
                )
                torch.save(
                    {
                        "format": f"synthetic_support_acquisition_bundle_{format_version}",
                        "encoder_state_dict": encoder.state_dict(),
                        "acquisition_state_dict": acquisition.state_dict(),
                        "feature_schema": feature_schema,
                        "latent_dim": int(args.latent_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "scale_invariant": bool(args.normalization_version == "v2"),
                        "synthetic_config": asdict(cfg),
                        "validation_config": asdict(validation_cfg),
                        "best_step": best_step,
                        "validation": metrics,
                    },
                    output / "bundle.pt",
                )

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final = {
        "schema": f"synthetic_support_acquisition_pretraining_{format_version}",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "normalization_version": str(args.normalization_version),
        "training_examples_generated": int(args.steps) * int(args.groups_per_batch) * int(cfg.candidates_per_bank),
        "checkpoint_selection": (
            "2*selection_regret + 0.25*coverage_mae + "
            "0.25*intensity_mae for normalized v2"
        ),
        "training_targets": (
            "exact marginal hard coverage and support-intensity gains from "
            "synthetic plane geometry"
        ),
        "encoder": {
            "path": str(output / "encoder.pt"),
            "architecture": "DeepSets mean/max pooling",
            "latent_dim": int(args.latent_dim),
            "feature_schema": feature_schema,
        },
        "acquisition": {
            "path": str(output / "acquisition.pt"),
            "architecture": "candidate-conditioned DeepSets bank comparison",
            "scale_invariant": bool(args.normalization_version == "v2"),
        },
        "bundle": str(output / "bundle.pt"),
        "synthetic_training_config": asdict(cfg),
        "synthetic_validation_config": asdict(validation_cfg),
        "optimization": {
            "steps": int(args.steps),
            "groups_per_batch": int(args.groups_per_batch),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "device": str(device),
        },
        "best_step": int(best_step),
        "best_validation": next(
            row for row in history if int(row["step"]) == best_step
        ),
        "final_validation": history[-1],
    }
    atomic_json(output / "metrics.json", final)
    print(f"[SYNTH-ACQ] saved={output} best_step={best_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
