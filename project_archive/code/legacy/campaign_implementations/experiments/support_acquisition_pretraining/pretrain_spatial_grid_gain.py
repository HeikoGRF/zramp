#!/usr/bin/env python3
"""Pretrain acquisition from exact map-aligned 300x300 support profiles."""

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

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.place_wallis_benchmark.cell_grid_support import (  # noqa: E402
    cell_grid_from_support_rows,
)
from experiments.support_acquisition_pretraining.grid_gain import (  # noqa: E402
    GRID_LAYOUT_REGULAR,
    grid_support_counts,
    relative_point_grid_gain,
)
from experiments.support_acquisition_pretraining.pretrain_shared_frame_gain import (  # noqa: E402
    UnitSquareConfig,
    make_production_candidate,
    make_unit_square_axes,
    make_unit_square_expert,
)
from experiments.support_acquisition_pretraining.pretrain_union_gain import atomic_json  # noqa: E402
from experiments.support_acquisition_pretraining.spatial_grid_gain_model import (  # noqa: E402
    SpatialGridEncoder,
    SpatialGridGainModel,
)


DEFAULT_OUTPUT = ROOT / "artifacts/support_acquisition_pretraining/spatial_grid_gain_v1"
PROFILE_KIND_PLANE_ENVELOPE = "plane-envelope"
PROFILE_KIND_CELL_TRAVERSAL = "cell-traversal"


@dataclass
class GridBatch:
    profiles: torch.Tensor
    candidate_indices: torch.Tensor
    bank_indices: torch.Tensor
    targets: torch.Tensor
    group_indices: torch.Tensor

    def to(self, device: torch.device) -> "GridBatch":
        return GridBatch(
            profiles=self.profiles.to(device=device, dtype=torch.float32),
            candidate_indices=self.candidate_indices.to(device),
            bank_indices=self.bank_indices.to(device),
            targets=self.targets.to(device),
            group_indices=self.group_indices.to(device),
        )


def _profile_group_task(
    payload: tuple[np.ndarray, list[np.ndarray], int, str]
) -> tuple[np.ndarray, np.ndarray]:
    bank_rows, candidate_rows, resolution, layout = payload
    bank = grid_support_counts(
        bank_rows,
        resolution=resolution,
        map_size=1.0,
        layout=layout,
    )
    # Sample counts can exceed float16 once vehicles retain all observations.
    # Preserve the exact intensity envelope used by the gain target.
    profiles = np.empty(
        (1 + len(candidate_rows), resolution, resolution), dtype=np.float32
    )
    profiles[0] = bank.reshape(resolution, resolution).astype(np.float32)
    targets = np.empty(len(candidate_rows), dtype=np.float32)
    for index, rows in enumerate(candidate_rows):
        candidate = grid_support_counts(
            rows,
            resolution=resolution,
            map_size=1.0,
            layout=layout,
        )
        profiles[index + 1] = candidate.reshape(
            resolution, resolution
        ).astype(np.float32)
        relative, _absolute = relative_point_grid_gain(bank, candidate)
        targets[index] = math.log1p(relative)
    return profiles, targets


def _cell_profile_group_task(
    payload: tuple[np.ndarray, list[np.ndarray], int, str]
) -> tuple[np.ndarray, np.ndarray]:
    bank_rows, candidate_rows, resolution, _layout = payload
    bank = cell_grid_from_support_rows(bank_rows, resolution=resolution)
    profiles = np.empty(
        (1 + len(candidate_rows), resolution, resolution), dtype=np.float32
    )
    profiles[0] = bank
    targets = np.empty(len(candidate_rows), dtype=np.float32)
    for index, rows in enumerate(candidate_rows):
        candidate = cell_grid_from_support_rows(rows, resolution=resolution)
        profiles[index + 1] = candidate
        relative, _absolute = relative_point_grid_gain(bank, candidate)
        targets[index] = math.log1p(relative)
    return profiles, targets


def make_batch(
    rng: np.random.Generator,
    cfg: UnitSquareConfig,
    groups: int,
    max_sample_count: int,
    executor: Executor | None = None,
    *,
    profile_kind: str = PROFILE_KIND_PLANE_ENVELOPE,
) -> GridBatch:
    if profile_kind not in {
        PROFILE_KIND_PLANE_ENVELOPE,
        PROFILE_KIND_CELL_TRAVERSAL,
    }:
        raise ValueError(f"unknown support profile kind {profile_kind!r}")
    payloads: list[tuple[np.ndarray, list[np.ndarray], int, str]] = []
    for _group in range(int(groups)):
        axes = make_unit_square_axes(rng, cfg)
        bank_size = int(rng.integers(cfg.min_bank_size, cfg.max_bank_size + 1))
        bank_axis_count = int(rng.integers(1, len(axes) + 1))
        bank_axes = np.asarray(
            rng.choice(len(axes), bank_axis_count, replace=False),
            dtype=np.int64,
        )
        experts = [
            make_unit_square_expert(
                rng, axes, cfg, max_sample_count, preferred=bank_axes
            )
            for _ in range(bank_size)
        ]
        candidates = [
            make_production_candidate(
                rng, axes, cfg, max_sample_count, experts, bank_axes
            )[0]
            for _ in range(cfg.candidates_per_bank)
        ]
        payloads.append((
            np.concatenate(experts, axis=0),
            candidates,
            int(cfg.grid_resolution),
            str(cfg.grid_layout),
        ))
    task = (
        _cell_profile_group_task
        if profile_kind == PROFILE_KIND_CELL_TRAVERSAL
        else _profile_group_task
    )
    results = (
        list(executor.map(task, payloads))
        if executor is not None
        else [task(payload) for payload in payloads]
    )
    profiles: list[np.ndarray] = []
    candidate_indices: list[int] = []
    bank_indices: list[int] = []
    targets: list[np.ndarray] = []
    group_indices: list[int] = []
    offset = 0
    for group, (group_profiles, group_targets) in enumerate(results):
        profiles.append(group_profiles)
        bank_indices.extend([offset] * len(group_targets))
        candidate_indices.extend(range(offset + 1, offset + 1 + len(group_targets)))
        targets.append(group_targets)
        group_indices.extend([group] * len(group_targets))
        offset += len(group_profiles)
    return GridBatch(
        profiles=torch.from_numpy(np.concatenate(profiles)),
        candidate_indices=torch.as_tensor(candidate_indices, dtype=torch.long),
        bank_indices=torch.as_tensor(bank_indices, dtype=torch.long),
        targets=torch.from_numpy(np.concatenate(targets)),
        group_indices=torch.as_tensor(group_indices, dtype=torch.long),
    )


def predict(
    encoder: SpatialGridEncoder,
    acquisition: SpatialGridGainModel,
    batch: GridBatch,
) -> torch.Tensor:
    embeddings = encoder(batch.profiles)
    return acquisition(embeddings, batch.candidate_indices, batch.bank_indices)


@torch.no_grad()
def evaluate(
    encoder: SpatialGridEncoder,
    acquisition: SpatialGridGainModel,
    batches: list[GridBatch],
    *,
    device: torch.device,
    maximum_relative_gain: float,
) -> dict[str, float | int]:
    encoder.eval()
    acquisition.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    top1 = 0
    groups_total = 0
    regret = 0.0
    for prepared in batches:
        batch = prepared.to(device)
        prediction = predict(encoder, acquisition, batch).cpu().numpy()
        target = batch.targets.cpu().numpy()
        predictions.append(prediction)
        targets.append(target)
        groups = batch.group_indices.cpu().numpy()
        for group in np.unique(groups):
            mask = groups == group
            predicted_choice = int(np.argmax(prediction[mask]))
            true_values = target[mask]
            optimum = float(np.max(true_values))
            chosen = float(true_values[predicted_choice])
            top1 += int(chosen >= optimum - 1.0e-6)
            regret += optimum - chosen
            groups_total += 1
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    cap = math.log1p(maximum_relative_gain)
    capped_target = np.minimum(target, cap)
    result: dict[str, float | int] = {
        "examples": int(len(target)),
        "groups": int(groups_total),
        "gain_log_mae_capped": float(np.mean(np.abs(prediction - capped_target))),
        "gain_log_rmse_capped": float(np.sqrt(np.mean((prediction - capped_target) ** 2))),
        "gain_log_rmse_uncapped": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "gain_prediction_bias_log_capped": float(np.mean(prediction - capped_target)),
        "top1_accuracy_tie_aware": float(top1 / max(1, groups_total)),
        "mean_selection_regret_log": float(regret / max(1, groups_total)),
        "positive_target_fraction": float(np.mean(target > 0.0)),
    }
    relative_prediction = np.expm1(prediction)
    relative_target = np.expm1(target)
    for percent in (2, 5, 10, 50, 100):
        threshold = percent / 100.0
        predicted_pass = relative_prediction >= threshold
        true_pass = relative_target >= threshold
        true_positive = int(np.sum(predicted_pass & true_pass))
        predicted_positive = int(np.sum(predicted_pass))
        actual_positive = int(np.sum(true_pass))
        result[f"threshold_{percent}_precision"] = float(
            true_positive / max(1, predicted_positive)
        )
        result[f"threshold_{percent}_recall"] = float(
            true_positive / max(1, actual_positive)
        )
        result[f"threshold_{percent}_predicted_pass_fraction"] = float(
            np.mean(predicted_pass)
        )
        result[f"threshold_{percent}_true_pass_fraction"] = float(
            np.mean(true_pass)
        )
    return result


def balanced_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Combine population calibration with equal decision-range coverage."""

    per_example = F.smooth_l1_loss(
        prediction, target, beta=0.02, reduction="none"
    ) + 0.25 * torch.square(prediction - target)
    boundaries = prediction.new_tensor([
        0.0,
        math.log1p(0.02),
        math.log1p(0.10),
        math.log1p(0.50),
        math.log1p(1.00),
    ])
    bins = torch.bucketize(target, boundaries, right=True)
    # Exact zeros form their own bin; positive sub-2% gains remain separate.
    bins = torch.where(target <= 0.0, torch.zeros_like(bins), bins + 1)
    bin_losses = [
        per_example[bins == index].mean()
        for index in torch.unique(bins)
    ]
    return 0.5 * per_example.mean() + 0.5 * torch.stack(bin_losses).mean()


def save_bundle(
    output: Path,
    encoder: SpatialGridEncoder,
    acquisition: SpatialGridGainModel,
    args: argparse.Namespace,
    cfg: UnitSquareConfig,
    stress_cfg: UnitSquareConfig,
    step: int,
    validation: dict[str, float | int],
) -> None:
    common = {
        "format": "synthetic_spatial_grid_gain_bundle_v3",
        "encoder_state_dict": encoder.state_dict(),
        "acquisition_state_dict": acquisition.state_dict(),
        "grid_resolution": int(cfg.grid_resolution),
        "grid_layout": str(cfg.grid_layout),
        "spatial_size": int(args.spatial_size),
        "learned_channels": int(args.learned_channels),
        "latent_channels": int(2 + args.learned_channels),
        "latent_dim": int(encoder.latent_dim),
        "hidden_channels": int(args.hidden_channels),
        "hidden_dim": int(args.hidden_dim),
        "count_scale": float(args.max_sample_count),
        "maximum_relative_gain": float(args.maximum_relative_gain),
        "best_step": int(step),
        "synthetic_config": asdict(cfg),
        "stress_config": asdict(stress_cfg),
        "validation": validation,
        "advertisement": "map-aligned learned spatial encoding only",
        "target": "exact 300x300 relative intensity gain, capped only above operational range",
    }
    torch.save(common, output / "bundle.pt")


def self_test() -> None:
    encoder = SpatialGridEncoder(spatial_size=8, learned_channels=2)
    acquisition = SpatialGridGainModel(
        spatial_size=8, latent_channels=4, hidden_channels=8, hidden_dim=16
    )
    profiles = torch.zeros((3, 32, 32))
    profiles[0, 4:12, 4:12] = 10
    profiles[1, 4:12, 4:12] = 10
    profiles[2, 20:28, 20:28] = 10
    embeddings = encoder(profiles)
    assert embeddings.shape == (3, encoder.latent_dim)
    prediction = acquisition(
        embeddings,
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
    )
    assert prediction.shape == (2,)
    assert bool(torch.isfinite(prediction).all())
    prediction.sum().backward()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--min-steps", type=int, default=3000)
    parser.add_argument("--groups-per-batch", type=int, default=2)
    parser.add_argument("--training-cache-batches", type=int, default=512)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--grid-resolution", type=int, default=300)
    parser.add_argument("--spatial-size", type=int, default=16)
    parser.add_argument("--learned-channels", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--maximum-relative-gain", type=float, default=4.0)
    parser.add_argument("--max-planes", type=int, default=512)
    parser.add_argument("--max-bank-size", type=int, default=24)
    parser.add_argument("--max-axes", type=int, default=96)
    parser.add_argument("--max-sample-count", type=int, default=4096)
    parser.add_argument("--candidates-per-bank", type=int, default=16)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("spatial-grid acquisition self-test passed")
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cfg = UnitSquareConfig(
        min_axes=6,
        max_axes=int(args.max_axes),
        min_planes=1,
        max_planes=int(args.max_planes),
        min_bank_size=1,
        max_bank_size=int(args.max_bank_size),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=int(args.grid_resolution),
        grid_layout=GRID_LAYOUT_REGULAR,
    )
    stress_cfg = UnitSquareConfig(
        min_axes=12,
        max_axes=max(128, int(args.max_axes)),
        min_planes=1,
        max_planes=max(768, int(args.max_planes)),
        min_bank_size=1,
        max_bank_size=max(32, int(args.max_bank_size)),
        candidates_per_bank=int(args.candidates_per_bank),
        grid_resolution=int(args.grid_resolution),
        grid_layout=GRID_LAYOUT_REGULAR,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executor: Executor | None = (
        ProcessPoolExecutor(max_workers=int(args.target_workers))
        if int(args.target_workers) > 1 else None
    )
    started = time.monotonic()

    def prepare_many(
        seed: int,
        count: int,
        config: UnitSquareConfig,
        groups: int,
        sample_count: int,
    ) -> list[GridBatch]:
        rng = np.random.default_rng(seed)
        batches: list[GridBatch] = []
        progress = max(1, count // 10)
        for index in range(count):
            batches.append(make_batch(
                rng, config, groups, int(sample_count), executor
            ))
            if (index + 1) % progress == 0 or index + 1 == count:
                print(f"[GRID-ACQ] prepared={index + 1}/{count} seed={seed}", flush=True)
        return batches

    validation = prepare_many(
        int(args.seed) + 100000,
        int(args.validation_batches),
        cfg,
        int(args.groups_per_batch),
        int(args.max_sample_count),
    )
    stress = prepare_many(
        int(args.seed) + 200000,
        max(16, int(args.validation_batches) // 2),
        stress_cfg,
        int(args.groups_per_batch),
        max(32768, int(args.max_sample_count)),
    )
    training = prepare_many(
        int(args.seed) + 300000,
        int(args.training_cache_batches),
        cfg,
        int(args.groups_per_batch),
        int(args.max_sample_count),
    )
    encoder = SpatialGridEncoder(
        spatial_size=int(args.spatial_size),
        learned_channels=int(args.learned_channels),
        count_scale=float(args.max_sample_count),
    ).to(device)
    acquisition = SpatialGridGainModel(
        spatial_size=int(args.spatial_size),
        latent_channels=2 + int(args.learned_channels),
        hidden_channels=int(args.hidden_channels),
        hidden_dim=int(args.hidden_dim),
        count_scale=float(args.max_sample_count),
        maximum_relative_gain=float(args.maximum_relative_gain),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *acquisition.parameters()],
        lr=float(args.learning_rate),
        weight_decay=1.0e-5,
    )
    order = np.arange(len(training), dtype=np.int64)
    order_rng = np.random.default_rng(int(args.seed) + 400000)
    history: list[dict[str, float | int]] = []
    best = float("inf")
    best_step = 0
    stale = 0
    cap = math.log1p(float(args.maximum_relative_gain))
    actual_steps = 0
    for step in range(1, int(args.max_steps) + 1):
        offset = (step - 1) % len(training)
        if offset == 0:
            order_rng.shuffle(order)
        batch = training[int(order[offset])].to(device)
        encoder.train()
        acquisition.train()
        prediction = predict(encoder, acquisition, batch)
        target = torch.clamp(batch.targets, max=cap)
        loss = balanced_regression_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *acquisition.parameters()], 5.0
        )
        optimizer.step()
        actual_steps = step
        if step == 1 or step % int(args.validation_every) == 0:
            metrics = evaluate(
                encoder,
                acquisition,
                validation,
                device=device,
                maximum_relative_gain=float(args.maximum_relative_gain),
            )
            score = float(metrics["gain_log_rmse_capped"])
            history.append({"step": step, "training_loss": float(loss.detach()), **metrics})
            if score < best - 1.0e-5:
                best = score
                best_step = step
                stale = 0
                save_bundle(output, encoder, acquisition, args, cfg, stress_cfg, step, metrics)
            else:
                stale += 1
            print(
                f"[GRID-ACQ] step={step:05d}/{args.max_steps} loss={float(loss.detach()):.6f} "
                f"rmse={score:.5f} p2={100*float(metrics['threshold_2_precision']):.1f}% "
                f"r2={100*float(metrics['threshold_2_recall']):.1f}% "
                f"p10={100*float(metrics['threshold_10_precision']):.1f}% "
                f"top1={100*float(metrics['top1_accuracy_tie_aware']):.1f}% "
                f"best={best_step} stale={stale}/{args.patience}",
                flush=True,
            )
            if step >= int(args.min_steps) and stale >= int(args.patience):
                break
    payload = torch.load(output / "bundle.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(payload["encoder_state_dict"])
    acquisition.load_state_dict(payload["acquisition_state_dict"])
    holdout = prepare_many(
        int(args.seed) + 500000,
        int(args.validation_batches),
        cfg,
        int(args.groups_per_batch),
        int(args.max_sample_count),
    )
    holdout_metrics = evaluate(
        encoder, acquisition, holdout, device=device,
        maximum_relative_gain=float(args.maximum_relative_gain)
    )
    stress_metrics = evaluate(
        encoder, acquisition, stress, device=device,
        maximum_relative_gain=float(args.maximum_relative_gain)
    )
    if executor is not None:
        executor.shutdown()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    atomic_json(output / "metrics.json", {
        "schema": "synthetic_spatial_grid_gain_pretraining_v3",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "best_step": int(best_step),
        "actual_steps": int(actual_steps),
        "elapsed_seconds": float(time.monotonic() - started),
        "training_unique_examples": int(len(training) * args.groups_per_batch * args.candidates_per_bank),
        "best_validation": payload["validation"],
        "holdout": holdout_metrics,
        "stress_holdout": stress_metrics,
        "bundle": str(output / "bundle.pt"),
        "configuration": {key: value for key, value in vars(args).items() if key != "output"},
    })
    print(
        f"[GRID-ACQ] complete best={best_step} actual={actual_steps} saved={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
