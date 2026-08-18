#!/usr/bin/env python3
"""Pretrain encoding-only support acquisition in a shared map frame."""

from __future__ import annotations

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.support_acquisition_pretraining.pretrain import (  # noqa: E402
    _candidate_focus,
    _choose_focus,
    _log_uniform_int,
)
from experiments.support_acquisition_pretraining.grid_gain import (  # noqa: E402
    DEFAULT_GRID_RESOLUTION,
    GRID_LAYOUT_STAGGERED,
    GRID_LAYOUTS,
    grid_support_counts,
    self_test as grid_gain_self_test,
)
from experiments.support_acquisition_pretraining.pretrain_union_gain import atomic_json  # noqa: E402
from experiments.support_acquisition_pretraining.shared_frame_gain_model import (  # noqa: E402
    EncodingOnlyGainModel,
    PlaneSetEncoder,
    SHARED_FRAME_PLANE_FEATURE_SCHEMA,
    normalize_plane_set_shared_frame,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/support_acquisition_pretraining/synthetic_unit_square_gain_v1"
)


@dataclass(frozen=True)
class UnitSquareConfig:
    min_axes: int = 6
    max_axes: int = 48
    min_planes: int = 1
    max_planes: int = 256
    min_bank_size: int = 1
    max_bank_size: int = 15
    candidates_per_bank: int = 8
    grid_resolution: int = DEFAULT_GRID_RESOLUTION
    grid_layout: str = GRID_LAYOUT_STAGGERED
    redundant_fraction: float = 0.45
    subset_fraction: float = 0.20
    evolved_fraction: float = 0.30


@dataclass
class SharedFrameBatch:
    plane_features: torch.Tensor
    plane_to_set: torch.Tensor
    candidate_indices: torch.Tensor
    bank_indices: torch.Tensor
    targets: torch.Tensor
    group_indices: torch.Tensor
    plane_counts: torch.Tensor
    set_count: int

    def to(self, device: torch.device) -> "SharedFrameBatch":
        return SharedFrameBatch(
            plane_features=self.plane_features.to(device),
            plane_to_set=self.plane_to_set.to(device),
            candidate_indices=self.candidate_indices.to(device),
            bank_indices=self.bank_indices.to(device),
            targets=self.targets.to(device),
            group_indices=self.group_indices.to(device),
            plane_counts=self.plane_counts.to(device),
            set_count=int(self.set_count),
        )


def _grid_gain_group_task(
    payload: tuple[np.ndarray, list[np.ndarray], np.ndarray, int, str]
) -> np.ndarray:
    """Return exact log-gain targets without transferring dense profiles."""

    bank_rows, candidate_rows, exact_zero, resolution, layout = payload
    bank_support = grid_support_counts(
        bank_rows,
        resolution=int(resolution),
        map_size=1.0,
        layout=str(layout),
    )
    bank_strength = float(np.sum(bank_support, dtype=np.float64))
    targets = np.empty(len(candidate_rows), dtype=np.float32)
    for index, rows in enumerate(candidate_rows):
        if bool(exact_zero[index]):
            targets[index] = 0.0
            continue
        candidate_support = grid_support_counts(
            rows,
            resolution=int(resolution),
            map_size=1.0,
            layout=str(layout),
        )
        marginal = float(np.sum(
            np.maximum(candidate_support - bank_support, 0.0),
            dtype=np.float64,
        ))
        targets[index] = math.log1p(
            marginal / max(bank_strength, 1.0)
        )
    return targets


def make_unit_square_axes(
    rng: np.random.Generator,
    cfg: UnitSquareConfig,
) -> np.ndarray:
    """Return random road axes whose endpoints lie in [0, 1]^2."""

    count = int(rng.integers(cfg.min_axes, cfg.max_axes + 1))
    endpoints = rng.uniform(0.0, 1.0, size=(count, 2, 2))
    vector = endpoints[:, 1] - endpoints[:, 0]
    lengths = np.linalg.norm(vector, axis=1)
    while bool(np.any(lengths <= np.finfo(np.float64).eps)):
        invalid = lengths <= np.finfo(np.float64).eps
        endpoints[invalid] = rng.uniform(
            0.0, 1.0, size=(int(invalid.sum()), 2, 2)
        )
        vector = endpoints[:, 1] - endpoints[:, 0]
        lengths = np.linalg.norm(vector, axis=1)
    centers = endpoints.mean(axis=1)
    angles = np.arctan2(vector[:, 1], vector[:, 0])
    half_widths = 0.5 * lengths * rng.beta(1.0, 20.0, size=count)
    return np.column_stack((centers, angles, lengths, half_widths))


def make_unit_square_expert(
    rng: np.random.Generator,
    axes: np.ndarray,
    cfg: UnitSquareConfig,
    max_sample_count: int,
    *,
    preferred: np.ndarray | None = None,
    plane_count: int | None = None,
) -> np.ndarray:
    """Generate one variable-length plane set from unit-square road axes."""

    if plane_count is None:
        plane_count = _log_uniform_int(rng, cfg.min_planes, cfg.max_planes)
    plane_count = int(plane_count)
    focus = _choose_focus(rng, len(axes), preferred)
    rows = np.empty((plane_count, 11), dtype=np.float64)
    log_max_count = math.log(float(max(1, max_sample_count)))
    for index in range(plane_count):
        axis_index = int(rng.choice(focus))
        cx, cy, theta, axis_length, road_half_width = axes[axis_index]
        direction = np.asarray([math.cos(theta), math.sin(theta)])
        axis_start = np.asarray([cx, cy]) - 0.5 * axis_length * direction
        fractions = np.sort(rng.uniform(0.0, 1.0, size=2))
        start = axis_start + fractions[0] * axis_length * direction
        end = axis_start + fractions[1] * axis_length * direction
        if end[0] < start[0] or (
            abs(float(end[0] - start[0])) < 1.0e-12
            and end[1] < start[1]
        ):
            start, end = end, start
        segment_length = float(np.linalg.norm(end - start))
        low_start = -road_half_width * float(rng.random())
        high_start = road_half_width * float(rng.random())
        low_end = -road_half_width * float(rng.random())
        high_end = road_half_width * float(rng.random())
        sample_count = max(
            1.0,
            math.floor(math.exp(rng.uniform(0.0, log_max_count))),
        )
        rows[index] = (
            start[0],
            start[1],
            end[0],
            end[1],
            low_start,
            high_start,
            low_end,
            high_end,
            sample_count,
            segment_length * float(rng.random()),
            math.degrees(math.pi * float(rng.beta(1.0, 20.0))),
        )
    return rows


def make_production_candidate(
    rng: np.random.Generator,
    axes: np.ndarray,
    cfg: UnitSquareConfig,
    max_sample_count: int,
    bank_experts: list[np.ndarray],
    bank_axes: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Draw a redundant, near-version, or novel advertised expert."""

    draw = float(rng.random())
    source = bank_experts[int(rng.integers(len(bank_experts)))]
    redundant_end = float(cfg.redundant_fraction)
    subset_end = redundant_end + float(cfg.subset_fraction)
    evolved_end = subset_end + float(cfg.evolved_fraction)
    if draw < redundant_end:
        rows = source.copy()
        rows[:, 8] *= rng.uniform(0.35, 1.0, size=len(rows))
        return rows, True
    if draw < subset_end:
        count = int(rng.integers(1, len(source) + 1))
        chosen = rng.choice(len(source), size=count, replace=False)
        rows = source[np.asarray(chosen, dtype=np.int64)].copy()
        rows[:, 8] *= rng.uniform(0.35, 1.0, size=len(rows))
        return rows, True
    if draw < evolved_end:
        # Vehicle models retain their old data; a successor keeps all previous
        # support and adds only the planes learned since its last advertisement.
        retained = source.copy()
        added_count = int(rng.integers(1, min(9, cfg.max_planes) + 1))
        added = make_unit_square_expert(
            rng,
            axes,
            cfg,
            max_sample_count,
            preferred=bank_axes,
            plane_count=added_count,
        )
        return np.concatenate((retained, added), axis=0), False
    return (
        make_unit_square_expert(
            rng,
            axes,
            cfg,
            max_sample_count,
            preferred=_candidate_focus(rng, len(axes), bank_axes),
        ),
        False,
    )


def make_batch(
    rng: np.random.Generator,
    cfg: UnitSquareConfig,
    group_count: int,
    max_sample_count: int,
    executor: Executor | None = None,
) -> SharedFrameBatch:
    all_features: list[np.ndarray] = []
    plane_to_set: list[np.ndarray] = []
    candidate_indices: list[int] = []
    bank_indices: list[int] = []
    group_indices: list[int] = []
    plane_counts: list[int] = []
    profile_groups: list[tuple[np.ndarray, list[np.ndarray], np.ndarray]] = []
    set_index = 0

    def append_set(rows: np.ndarray) -> int:
        nonlocal set_index
        features = normalize_plane_set_shared_frame(rows, map_size=1.0)
        index = set_index
        all_features.append(features)
        plane_to_set.append(np.full(len(rows), index, dtype=np.int64))
        plane_counts.append(len(rows))
        set_index += 1
        return index

    for group in range(int(group_count)):
        axes = make_unit_square_axes(rng, cfg)
        bank_size = int(rng.integers(
            cfg.min_bank_size, cfg.max_bank_size + 1
        ))
        bank_axis_count = int(rng.integers(1, len(axes) + 1))
        bank_axes = np.asarray(
            rng.choice(len(axes), bank_axis_count, replace=False),
            dtype=np.int64,
        )
        bank_experts = [
            make_unit_square_expert(
                rng, axes, cfg, max_sample_count, preferred=bank_axes
            )
            for _ in range(bank_size)
        ]
        bank_union = np.concatenate(bank_experts, axis=0)
        bank_index = append_set(bank_union)
        candidates_and_flags = [
            make_production_candidate(
                rng,
                axes,
                cfg,
                max_sample_count,
                bank_experts,
                bank_axes,
            )
            for _candidate in range(cfg.candidates_per_bank)
        ]
        candidate_rows = [item[0] for item in candidates_and_flags]
        exact_zero = np.asarray(
            [item[1] for item in candidates_and_flags], dtype=np.bool_
        )
        candidate_indices.extend(append_set(rows) for rows in candidate_rows)
        bank_indices.extend([bank_index] * len(candidate_rows))
        group_indices.extend([group] * len(candidate_rows))
        profile_groups.append((bank_union, candidate_rows, exact_zero))

    target_payloads = [
        (
            bank_union,
            candidate_rows,
            exact_zero,
            cfg.grid_resolution,
            cfg.grid_layout,
        )
        for bank_union, candidate_rows, exact_zero in profile_groups
    ]
    if executor is None:
        group_targets = [
            _grid_gain_group_task(payload) for payload in target_payloads
        ]
    else:
        group_targets = list(executor.map(
            _grid_gain_group_task, target_payloads
        ))
    targets = [
        float(target)
        for values in group_targets
        for target in values
    ]

    return SharedFrameBatch(
        plane_features=torch.from_numpy(np.concatenate(all_features)),
        plane_to_set=torch.from_numpy(np.concatenate(plane_to_set)),
        candidate_indices=torch.from_numpy(
            np.asarray(candidate_indices, dtype=np.int64)
        ),
        bank_indices=torch.from_numpy(np.asarray(bank_indices, dtype=np.int64)),
        targets=torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        group_indices=torch.from_numpy(
            np.asarray(group_indices, dtype=np.int64)
        ),
        plane_counts=torch.from_numpy(np.asarray(plane_counts, dtype=np.int64)),
        set_count=int(set_index),
    )


def predict(
    encoder: PlaneSetEncoder,
    acquisition: EncodingOnlyGainModel,
    batch: SharedFrameBatch,
) -> torch.Tensor:
    embeddings = encoder(
        batch.plane_features,
        batch.plane_to_set,
        int(batch.set_count),
    )
    return acquisition(
        embeddings,
        batch.candidate_indices,
        batch.bank_indices,
    )


@torch.no_grad()
def evaluate(
    encoder: PlaneSetEncoder,
    acquisition: EncodingOnlyGainModel,
    *,
    cfg: UnitSquareConfig,
    max_sample_count: int,
    seed: int,
    batches: int,
    groups_per_batch: int,
    device: torch.device,
    executor: Executor | None = None,
    prepared_batches: list[SharedFrameBatch] | None = None,
) -> dict[str, float | int]:
    encoder.eval()
    acquisition.eval()
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    top_correct = 0
    group_total = 0
    regret_sum = 0.0
    plane_min = 2**63 - 1
    plane_max = 0
    plane_sum = 0
    plane_total = 0
    if prepared_batches is None:
        batch_source = (
            make_batch(
                rng,
                cfg,
                groups_per_batch,
                max_sample_count,
                executor=executor,
            )
            for _ in range(int(batches))
        )
    else:
        if len(prepared_batches) != int(batches):
            raise ValueError("prepared validation batch count differs")
        batch_source = iter(prepared_batches)
    for prepared_batch in batch_source:
        batch = prepared_batch.to(device)
        prediction_np = predict(
            encoder, acquisition, batch
        ).detach().cpu().numpy()
        target_np = batch.targets.detach().cpu().numpy()
        predictions.append(prediction_np)
        targets.append(target_np)
        counts = batch.plane_counts.detach().cpu().numpy()
        plane_min = min(plane_min, int(counts.min()))
        plane_max = max(plane_max, int(counts.max()))
        plane_sum += int(counts.sum())
        plane_total += int(len(counts))
        groups = batch.group_indices.detach().cpu().numpy()
        for group in np.unique(groups):
            mask = groups == group
            predicted_choice = int(np.argmax(prediction_np[mask]))
            true_values = target_np[mask]
            true_choice = int(np.argmax(true_values))
            top_correct += int(predicted_choice == true_choice)
            regret_sum += float(
                true_values[true_choice] - true_values[predicted_choice]
            )
            group_total += 1
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    absolute_error = np.abs(prediction - target)
    relative_prediction = np.expm1(prediction)
    relative_target = np.expm1(target)
    result: dict[str, float | int] = {
        "examples": int(len(target)),
        "groups": int(group_total),
        "gain_log_mae": float(np.mean(absolute_error)),
        "gain_log_rmse": float(np.sqrt(np.mean(np.square(absolute_error)))),
        "gain_relative_mae": float(np.mean(
            np.abs(relative_prediction - relative_target)
        )),
        "gain_relative_rmse": float(np.sqrt(np.mean(np.square(
            relative_prediction - relative_target
        )))),
        "gain_prediction_bias_log": float(np.mean(prediction - target)),
        "top1_accuracy": float(top_correct / max(1, group_total)),
        "mean_selection_regret_log": float(regret_sum / max(1, group_total)),
        "positive_target_fraction": float(np.mean(target > 0.0)),
        "plane_count_min": int(plane_min),
        "plane_count_max": int(plane_max),
        "plane_count_mean": float(plane_sum / max(1, plane_total)),
    }
    for percent in (2, 5, 10, 50, 100):
        threshold = percent / 100.0
        predicted_pass = relative_prediction >= threshold
        true_pass = relative_target >= threshold
        true_positive = int(np.sum(predicted_pass & true_pass))
        predicted_positive = int(np.sum(predicted_pass))
        actual_positive = int(np.sum(true_pass))
        result[f"threshold_{percent}_predicted_pass_fraction"] = float(
            np.mean(predicted_pass)
        )
        result[f"threshold_{percent}_true_pass_fraction"] = float(
            np.mean(true_pass)
        )
        result[f"threshold_{percent}_precision"] = float(
            true_positive / max(1, predicted_positive)
        )
        result[f"threshold_{percent}_recall"] = float(
            true_positive / max(1, actual_positive)
        )
    return result


def save_best(
    output: Path,
    *,
    encoder: PlaneSetEncoder,
    acquisition: EncodingOnlyGainModel,
    cfg: UnitSquareConfig,
    validation_cfg: UnitSquareConfig,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float | int],
) -> None:
    common = {
        "feature_schema": SHARED_FRAME_PLANE_FEATURE_SCHEMA,
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "synthetic_config": asdict(cfg),
        "validation_config": asdict(validation_cfg),
        "max_sample_count": int(args.max_sample_count),
        "coordinate_frame": "shared unit square [0,1]^2",
        "target": (
            "log1p(sum(max(candidate_grid_count-bank_grid_count,0))/"
            "max(sum(bank_grid_count),1))"
        ),
        "grid_resolution": int(cfg.grid_resolution),
        "grid_points": int(cfg.grid_resolution ** 2),
        "grid_layout": str(cfg.grid_layout),
        "best_step": int(step),
        "validation": metrics,
        "candidate_distribution": {
            "redundant": float(cfg.redundant_fraction),
            "subset": float(cfg.subset_fraction),
            "evolved_version": float(cfg.evolved_fraction),
            "novel": float(1.0 - cfg.redundant_fraction
                           - cfg.subset_fraction
                           - cfg.evolved_fraction),
        },
    }
    torch.save(
        {
            "format": "unit_square_grid_plane_set_encoder_v2",
            "state_dict": encoder.state_dict(),
            **common,
        },
        output / "encoder.pt",
    )
    torch.save(
        {
            "format": "encoding_only_grid_gain_model_v2",
            "state_dict": acquisition.state_dict(),
            "output": "one_scalar_log1p_relative_support_count_gain",
            **common,
        },
        output / "acquisition.pt",
    )
    torch.save(
        {
            "format": "synthetic_unit_square_grid_gain_bundle_v2",
            "encoder_state_dict": encoder.state_dict(),
            "acquisition_state_dict": acquisition.state_dict(),
            "output": "one_scalar_log1p_relative_support_count_gain",
            **common,
        },
        output / "bundle.pt",
    )


def self_test() -> None:
    grid_gain_self_test()
    cfg = UnitSquareConfig(
        max_axes=8,
        max_planes=16,
        max_bank_size=4,
        candidates_per_bank=3,
        grid_resolution=16,
    )
    batch = make_batch(np.random.default_rng(7), cfg, 3, 128)
    assert int(batch.plane_counts.min()) < int(batch.plane_counts.max())
    assert bool(torch.isfinite(batch.targets).all())
    assert bool((batch.targets >= 0.0).all())
    assert bool((batch.plane_features[:, 0:2] >= 0.0).all())
    assert bool((batch.plane_features[:, 0:2] <= 1.0).all())
    rng = np.random.default_rng(11)
    rows = make_unit_square_expert(
        rng,
        make_unit_square_axes(rng, cfg),
        cfg,
        128,
    )
    unit_features = normalize_plane_set_shared_frame(rows, map_size=1.0)
    scaled_rows = rows.copy()
    scaled_rows[:, 0:8] *= 300.0
    scaled_rows[:, 9] *= 300.0
    metre_features = normalize_plane_set_shared_frame(
        scaled_rows, map_size=300.0
    )
    assert np.allclose(unit_features, metre_features, atol=1.0e-6)
    encoder = PlaneSetEncoder(hidden_dim=32, latent_dim=16)
    acquisition = EncodingOnlyGainModel(latent_dim=16, hidden_dim=32)
    prediction = predict(encoder, acquisition, batch)
    assert prediction.shape == batch.targets.shape
    assert bool((prediction >= 0.0).all())
    loss = F.mse_loss(prediction, batch.targets)
    loss.backward()
    assert torch.isfinite(loss)
    encoder.eval()
    with torch.no_grad():
        original = encoder(
            batch.plane_features,
            batch.plane_to_set,
            batch.set_count,
        )
        order = torch.randperm(len(batch.plane_features))
        permuted = encoder(
            batch.plane_features[order],
            batch.plane_to_set[order],
            batch.set_count,
        )
    assert torch.allclose(original, permuted, atol=1.0e-6, rtol=1.0e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--min-steps", type=int, default=10_000)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=32)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-planes", type=int, default=512)
    parser.add_argument("--max-bank-size", type=int, default=24)
    parser.add_argument(
        "--grid-resolution", type=int, default=DEFAULT_GRID_RESOLUTION
    )
    parser.add_argument(
        "--grid-layout",
        choices=GRID_LAYOUTS,
        default=GRID_LAYOUT_STAGGERED,
    )
    parser.add_argument("--candidates-per-bank", type=int, default=16)
    parser.add_argument("--max-axes", type=int, default=96)
    parser.add_argument("--training-cache-batches", type=int, default=1024)
    parser.add_argument("--max-sample-count", type=int, default=4096)
    parser.add_argument("--target-workers", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("unit-square acquisition pretraining self-test passed")
        return 0
    if args.min_steps > args.max_steps:
        raise ValueError("minimum steps cannot exceed maximum steps")
    if args.early_stopping_patience <= 0:
        raise ValueError("early-stopping patience must be positive")
    if args.grid_resolution <= 0:
        raise ValueError("grid resolution must be positive")
    if args.target_workers <= 0:
        raise ValueError("target workers must be positive")
    if args.gradient_clip_norm < 0.0:
        raise ValueError("gradient clip norm cannot be negative")
    if args.training_cache_batches < 0:
        raise ValueError("training cache batch count cannot be negative")
    mixture_total = 0.45 + 0.20 + 0.30
    if not 0.0 <= mixture_total <= 1.0:
        raise ValueError("synthetic candidate fractions must sum to at most one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cfg = UnitSquareConfig(
        max_axes=int(args.max_axes),
        max_planes=int(args.max_planes),
        max_bank_size=int(args.max_bank_size),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=int(args.grid_resolution),
        grid_layout=str(args.grid_layout),
    )
    validation_cfg = cfg
    encoder = PlaneSetEncoder(
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
    ).to(device)
    acquisition = EncodingOnlyGainModel(
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *acquisition.parameters()],
        lr=float(args.learning_rate),
    )
    rng = np.random.default_rng(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_executor: Executor | None = (
        ProcessPoolExecutor(max_workers=int(args.target_workers))
        if int(args.target_workers) > 1
        else None
    )
    started = time.monotonic()
    validation_rng = np.random.default_rng(int(args.seed) + 100_000)
    prepared_validation_batches = [
        make_batch(
            validation_rng,
            validation_cfg,
            max(2, int(args.groups_per_batch) // 2),
            int(args.max_sample_count),
            executor=target_executor,
        )
        for _ in range(int(args.validation_batches))
    ]
    training_cache: list[SharedFrameBatch] = []
    cache_count = int(args.training_cache_batches)
    if cache_count > 0:
        cache_rng = np.random.default_rng(int(args.seed) + 300_000)
        progress_every = max(1, cache_count // 20)
        print(
            f"[UNIT-ACQ] preparing {cache_count} exact synthetic batches",
            flush=True,
        )
        for cache_index in range(cache_count):
            training_cache.append(make_batch(
                cache_rng,
                cfg,
                int(args.groups_per_batch),
                int(args.max_sample_count),
                executor=target_executor,
            ))
            if (cache_index + 1) % progress_every == 0 or (
                cache_index + 1 == cache_count
            ):
                print(
                    f"[UNIT-ACQ] cache={cache_index + 1}/{cache_count}",
                    flush=True,
                )
    cache_order = np.arange(cache_count, dtype=np.int64)
    order_rng = np.random.default_rng(int(args.seed) + 400_000)
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_step = 0
    stale_validations = 0
    actual_steps = 0
    early_stopped = False

    for step in range(1, int(args.max_steps) + 1):
        encoder.train()
        acquisition.train()
        if training_cache:
            cache_offset = (step - 1) % cache_count
            if cache_offset == 0:
                order_rng.shuffle(cache_order)
            prepared = training_cache[int(cache_order[cache_offset])]
        else:
            prepared = make_batch(
                rng,
                cfg,
                int(args.groups_per_batch),
                int(args.max_sample_count),
                executor=target_executor,
            )
        batch = prepared.to(device)
        predicted = predict(encoder, acquisition, batch)
        loss = F.mse_loss(predicted, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.gradient_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [*encoder.parameters(), *acquisition.parameters()],
                max_norm=float(args.gradient_clip_norm),
            )
        optimizer.step()
        actual_steps = step

        if (
            step == 1
            or step % int(args.validation_every) == 0
            or step == int(args.max_steps)
        ):
            metrics = evaluate(
                encoder,
                acquisition,
                cfg=validation_cfg,
                max_sample_count=int(args.max_sample_count),
                seed=int(args.seed) + 100_000,
                batches=int(args.validation_batches),
                groups_per_batch=max(2, int(args.groups_per_batch) // 2),
                device=device,
                executor=target_executor,
                prepared_batches=prepared_validation_batches,
            )
            score = float(metrics["gain_log_rmse"])
            row: dict[str, float | int] = {
                "step": int(step),
                "training_loss": float(loss.detach()),
                **metrics,
            }
            history.append(row)
            improved = score < (
                best_score - float(args.early_stopping_min_delta)
            )
            if improved:
                best_score = score
                best_step = step
                stale_validations = 0
                save_best(
                    output,
                    encoder=encoder,
                    acquisition=acquisition,
                    cfg=cfg,
                    validation_cfg=validation_cfg,
                    args=args,
                    step=step,
                    metrics=metrics,
                )
            else:
                stale_validations += 1
            print(
                f"[UNIT-ACQ] step={step:05d}/{args.max_steps} "
                f"train_mse={float(loss.detach()):.6f} "
                f"val_rmse={score:.6f} "
                f"top1={100 * float(metrics['top1_accuracy']):.1f}% "
                f"regret={float(metrics['mean_selection_regret_log']):.6f} "
                f"best={best_step} stale={stale_validations}/"
                f"{args.early_stopping_patience}",
                flush=True,
            )
            if (
                step >= int(args.min_steps)
                and stale_validations >= int(args.early_stopping_patience)
            ):
                early_stopped = True
                break

    bundle = torch.load(
        output / "bundle.pt", map_location=device, weights_only=False
    )
    encoder.load_state_dict(bundle["encoder_state_dict"])
    acquisition.load_state_dict(bundle["acquisition_state_dict"])
    holdout = evaluate(
        encoder,
        acquisition,
        cfg=validation_cfg,
        max_sample_count=int(args.max_sample_count),
        seed=int(args.seed) + 200_000,
        batches=int(args.validation_batches),
        groups_per_batch=max(2, int(args.groups_per_batch) // 2),
        device=device,
        executor=target_executor,
    )
    if target_executor is not None:
        target_executor.shutdown()
    with (output / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    elapsed = time.monotonic() - started
    atomic_json(output / "metrics.json", {
        "schema": "synthetic_unit_square_grid_gain_pretraining_v2",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "data_generation": (
            "cached artificial unit-square scenes with redundant, subset, "
            "evolved-version, and novel advertisements"
        ),
        "training_target": (
            "one scalar log1p relative intensity-weighted point-grid gain"
        ),
        "gain_grid": {
            "frame": "fixed normalized unit square",
            "layout": (
                "regular cell-centred square lattice"
                if cfg.grid_layout == "regular"
                else "deterministic rows with alternating half-cell horizontal shift"
            ),
            "resolution": int(cfg.grid_resolution),
            "points": int(cfg.grid_resolution ** 2),
            "aggregation": "pointwise maximum raw plane sample count",
            "zero_width_plane_support": 0,
        },
        "training_loss": "mean squared error only",
        "probability_output": False,
        "feature_schema": SHARED_FRAME_PLANE_FEATURE_SCHEMA,
        "architecture": {
            "plane_encoder": "shared MLP plus mean/max pooling",
            "bank_encoder": "same encoder over the union plane set",
            "acquisition_head": "candidate and bank encodings to one scalar",
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
        },
        "synthetic_config": asdict(cfg),
        "validation_config": asdict(validation_cfg),
        "max_sample_count": int(args.max_sample_count),
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": float(args.learning_rate),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "max_steps": int(args.max_steps),
            "actual_steps": int(actual_steps),
            "groups_per_batch": int(args.groups_per_batch),
            "target_workers": int(args.target_workers),
            "training_cache_batches": int(args.training_cache_batches),
            "unique_training_examples_generated": (
                (
                    int(args.training_cache_batches)
                    if int(args.training_cache_batches) > 0
                    else int(actual_steps)
                )
                * int(args.groups_per_batch)
                * int(cfg.candidates_per_bank)
            ),
            "training_example_exposures": (
                int(actual_steps)
                * int(args.groups_per_batch)
                * int(cfg.candidates_per_bank)
            ),
            "elapsed_seconds": float(elapsed),
        },
        "early_stopping": {
            "enabled": True,
            "minimum_steps": int(args.min_steps),
            "validation_every_steps": int(args.validation_every),
            "patience_validations": int(args.early_stopping_patience),
            "minimum_delta": float(args.early_stopping_min_delta),
            "triggered": bool(early_stopped),
            "best_step": int(best_step),
            "best_validation_rmse_log_gain": float(best_score),
        },
        "best_validation": bundle["validation"],
        "holdout_validation": holdout,
        "artifacts": {
            "encoder": str(output / "encoder.pt"),
            "acquisition": str(output / "acquisition.pt"),
            "bundle": str(output / "bundle.pt"),
        },
    })
    print(
        f"[UNIT-ACQ] saved={output} best_step={best_step} "
        f"actual_steps={actual_steps} early_stopped={int(early_stopped)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
