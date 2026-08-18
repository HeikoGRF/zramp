#!/usr/bin/env python3
"""Pretrain scalar acquisition from candidate and bank-union plane sets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
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

from experiments.support_acquisition_pretraining.pretrain import (
    SyntheticConfig,
    _candidate_focus,
    _make_axes,
    _make_expert,
    _sample_queries,
)
from experiments.support_acquisition_pretraining.union_gain_model import (
    PlaneSetEncoder,
    SCALAR_PLANE_FEATURE_SCHEMA,
    UnionGainModel,
    normalize_plane_set,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/support_acquisition_pretraining/synthetic_union_gain_v1"
)


@dataclass
class UnionBatch:
    plane_features: torch.Tensor
    plane_to_set: torch.Tensor
    set_centers: torch.Tensor
    set_scales: torch.Tensor
    candidate_indices: torch.Tensor
    bank_indices: torch.Tensor
    targets: torch.Tensor
    group_indices: torch.Tensor
    plane_counts: torch.Tensor

    def to(self, device: torch.device) -> "UnionBatch":
        return UnionBatch(**{
            name: value.to(device)
            for name, value in self.__dict__.items()
        })


def support_counts(rows: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Return the largest raw sample count of a plane supporting each query."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    if len(values) == 0:
        return np.zeros(len(queries), dtype=np.float64)
    start, end = values[:, 0:2], values[:, 2:4]
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-9)
    axis = vector / length[:, None]
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    midpoint = 0.5 * (start + end)
    relative = queries[:, :, None, :] - midpoint[None, None, :, :]
    along = np.einsum("qepc,pc->qep", relative, axis)
    lateral = np.einsum("qepc,pc->qep", relative, normal)
    fraction = np.clip(
        (along + 0.5 * length[None, None, :])
        / length[None, None, :],
        0.0,
        1.0,
    )
    low = (
        (1.0 - fraction) * values[None, None, :, 4]
        + fraction * values[None, None, :, 6]
    )
    high = (
        (1.0 - fraction) * values[None, None, :, 5]
        + fraction * values[None, None, :, 7]
    )
    query_length = np.linalg.norm(
        queries[:, 1] - queries[:, 0], axis=1
    )
    supported = (
        ((lateral >= low) & (lateral <= high)).all(axis=1)
        & (np.abs(along) <= 0.5 * length[None, None, :]).all(axis=1)
        & (query_length[:, None] <= values[None, :, 9])
    )
    counts = np.maximum(values[:, 8], 0.0)
    return np.max(np.where(supported, counts[None, :], 0.0), axis=1)


def _expert(
    rng: np.random.Generator,
    axes: np.ndarray,
    cfg: SyntheticConfig,
    max_sample_count: int,
    *,
    preferred: np.ndarray | None = None,
) -> np.ndarray:
    rows = _make_expert(rng, axes, cfg, preferred=preferred)
    log_max = math.log(float(max(1, max_sample_count)))
    rows[:, 8] = np.maximum(
        1.0,
        np.floor(np.exp(rng.uniform(0.0, log_max, size=len(rows)))),
    )
    return rows


def make_batch(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    group_count: int,
    max_sample_count: int,
) -> UnionBatch:
    all_features: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scales: list[float] = []
    plane_to_set: list[np.ndarray] = []
    candidate_indices: list[int] = []
    bank_indices: list[int] = []
    targets: list[float] = []
    group_indices: list[int] = []
    plane_counts: list[int] = []
    set_index = 0

    def append_set(rows: np.ndarray) -> int:
        nonlocal set_index
        features, center, scale = normalize_plane_set(rows)
        index = set_index
        all_features.append(features)
        centers.append(center)
        scales.append(scale)
        plane_to_set.append(
            np.full(len(rows), index, dtype=np.int64)
        )
        plane_counts.append(len(rows))
        set_index += 1
        return index

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
            _expert(
                rng,
                axes,
                cfg,
                max_sample_count,
                preferred=bank_axes,
            )
            for _ in range(bank_size)
        ]
        bank_index = append_set(
            np.concatenate(bank_rows, axis=0)
        )
        queries = _sample_queries(rng, axes, cfg.queries_per_world)
        bank_support = support_counts(
            np.concatenate(bank_rows, axis=0), queries
        )
        bank_strength = float(np.mean(bank_support))

        for _candidate in range(cfg.candidates_per_bank):
            preferred = _candidate_focus(rng, len(axes), bank_axes)
            rows = _expert(
                rng,
                axes,
                cfg,
                max_sample_count,
                preferred=preferred,
            )
            candidate_support = support_counts(rows, queries)
            marginal = float(np.mean(
                np.maximum(candidate_support - bank_support, 0.0)
            ))
            relative_gain = marginal / max(bank_strength, 1.0)
            target = math.log1p(relative_gain)
            candidate_indices.append(append_set(rows))
            bank_indices.append(bank_index)
            targets.append(target)
            group_indices.append(group)

    return UnionBatch(
        plane_features=torch.from_numpy(np.concatenate(all_features)),
        plane_to_set=torch.from_numpy(np.concatenate(plane_to_set)),
        set_centers=torch.from_numpy(np.stack(centers)),
        set_scales=torch.from_numpy(
            np.asarray(scales, dtype=np.float32)
        ),
        candidate_indices=torch.from_numpy(
            np.asarray(candidate_indices, dtype=np.int64)
        ),
        bank_indices=torch.from_numpy(
            np.asarray(bank_indices, dtype=np.int64)
        ),
        targets=torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        group_indices=torch.from_numpy(
            np.asarray(group_indices, dtype=np.int64)
        ),
        plane_counts=torch.from_numpy(
            np.asarray(plane_counts, dtype=np.int64)
        ),
    )


def predict(
    encoder: PlaneSetEncoder,
    acquisition: UnionGainModel,
    batch: UnionBatch,
) -> torch.Tensor:
    embeddings = encoder(
        batch.plane_features,
        batch.plane_to_set,
        int(batch.set_centers.shape[0]),
    )
    return acquisition(
        embeddings,
        batch.set_centers,
        batch.set_scales,
        batch.candidate_indices,
        batch.bank_indices,
    )


@torch.no_grad()
def evaluate(
    encoder: PlaneSetEncoder,
    acquisition: UnionGainModel,
    *,
    cfg: SyntheticConfig,
    max_sample_count: int,
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
    top_correct = 0
    group_total = 0
    regret_sum = 0.0
    plane_min = 2**63 - 1
    plane_max = 0
    plane_sum = 0
    plane_total = 0
    for _ in range(int(batches)):
        batch = make_batch(
            rng, cfg, groups_per_batch, max_sample_count
        ).to(device)
        predicted = predict(encoder, acquisition, batch)
        prediction_np = predicted.detach().cpu().numpy()
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
    prediction_nonnegative = np.maximum(prediction, 0.0)
    absolute_error = np.abs(prediction_nonnegative - target)
    square_error = np.square(prediction_nonnegative - target)
    relative_prediction = np.expm1(prediction_nonnegative)
    relative_target = np.expm1(target)
    return {
        "examples": int(len(target)),
        "groups": int(group_total),
        "gain_log_mae": float(np.mean(absolute_error)),
        "gain_log_rmse": float(np.sqrt(np.mean(square_error))),
        "gain_relative_mae": float(np.mean(
            np.abs(relative_prediction - relative_target)
        )),
        "gain_relative_rmse": float(np.sqrt(np.mean(
            np.square(relative_prediction - relative_target)
        ))),
        "gain_prediction_bias_log": float(np.mean(
            prediction_nonnegative - target
        )),
        "top1_accuracy": float(top_correct / max(1, group_total)),
        "mean_selection_regret_log": float(
            regret_sum / max(1, group_total)
        ),
        "positive_target_fraction": float(np.mean(target > 0.0)),
        "negative_prediction_fraction": float(np.mean(prediction < 0.0)),
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


def save_best(
    output: Path,
    *,
    encoder: PlaneSetEncoder,
    acquisition: UnionGainModel,
    cfg: SyntheticConfig,
    validation_cfg: SyntheticConfig,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float | int],
) -> None:
    common = {
        "feature_schema": SCALAR_PLANE_FEATURE_SCHEMA,
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "synthetic_config": asdict(cfg),
        "validation_config": asdict(validation_cfg),
        "max_sample_count": int(args.max_sample_count),
        "target": (
            "log1p(mean(max(candidate_count-bank_count,0))/"
            "max(mean(bank_count),1 sample))"
        ),
        "best_step": int(step),
        "validation": metrics,
    }
    torch.save(
        {
            "format": "union_plane_set_encoder_v1",
            "state_dict": encoder.state_dict(),
            **common,
        },
        output / "encoder.pt",
    )
    torch.save(
        {
            "format": "union_support_gain_model_v1",
            "state_dict": acquisition.state_dict(),
            "output": "one_scalar_log1p_relative_support_count_gain",
            **common,
        },
        output / "acquisition.pt",
    )
    torch.save(
        {
            "format": "synthetic_union_support_gain_bundle_v1",
            "encoder_state_dict": encoder.state_dict(),
            "acquisition_state_dict": acquisition.state_dict(),
            "output": "one_scalar_log1p_relative_support_count_gain",
            **common,
        },
        output / "bundle.pt",
    )


def self_test() -> None:
    cfg = SyntheticConfig(
        min_world_m=30.0,
        max_world_m=100.0,
        max_axes=8,
        max_planes=16,
        max_bank_size=4,
        candidates_per_bank=3,
        queries_per_world=64,
        normalization_version="v2",
    )
    batch = make_batch(np.random.default_rng(7), cfg, 3, 128)
    assert int(batch.plane_counts.min()) < int(batch.plane_counts.max())
    assert bool(torch.isfinite(batch.targets).all())
    assert bool((batch.targets >= 0.0).all())
    encoder = PlaneSetEncoder(hidden_dim=32, latent_dim=16)
    acquisition = UnionGainModel(latent_dim=16, hidden_dim=32)
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
            int(batch.set_centers.shape[0]),
        )
        order = torch.randperm(len(batch.plane_features))
        permuted = encoder(
            batch.plane_features[order],
            batch.plane_to_set[order],
            int(batch.set_centers.shape[0]),
        )
    assert torch.allclose(original, permuted, atol=1.0e-6, rtol=1.0e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--min-steps", type=int, default=5_000)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=24)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-planes", type=int, default=256)
    parser.add_argument("--max-bank-size", type=int, default=20)
    parser.add_argument("--queries-per-world", type=int, default=512)
    parser.add_argument("--candidates-per-bank", type=int, default=8)
    parser.add_argument("--min-world-m", type=float, default=40.0)
    parser.add_argument("--max-world-m", type=float, default=1800.0)
    parser.add_argument("--max-axes", type=int, default=48)
    parser.add_argument("--max-sample-count", type=int, default=4096)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("union acquisition pretraining self-test passed")
        return 0
    if args.min_steps > args.max_steps:
        raise ValueError("minimum steps cannot exceed maximum steps")
    if args.early_stopping_patience <= 0:
        raise ValueError("early-stopping patience must be positive")
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
        normalization_version="v2",
    )
    validation_cfg = replace(
        cfg,
        min_world_m=max(
            float(np.nextafter(2.0 / 0.06, math.inf)), 0.75 * cfg.min_world_m
        ),
        max_world_m=max(2250.0, 1.25 * cfg.max_world_m),
        max_axes=max(56, cfg.max_axes),
        max_planes=max(320, cfg.max_planes),
        max_bank_size=max(24, cfg.max_bank_size),
    )
    encoder = PlaneSetEncoder(
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
    ).to(device)
    acquisition = UnionGainModel(
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
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_step = 0
    stale_validations = 0
    actual_steps = 0
    early_stopped = False
    started = time.monotonic()

    for step in range(1, int(args.max_steps) + 1):
        encoder.train()
        acquisition.train()
        batch = make_batch(
            rng,
            cfg,
            int(args.groups_per_batch),
            int(args.max_sample_count),
        ).to(device)
        predicted = predict(encoder, acquisition, batch)
        loss = F.mse_loss(predicted, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
                groups_per_batch=max(
                    2, int(args.groups_per_batch) // 2
                ),
                device=device,
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
                f"[UNION-ACQ] step={step:05d}/{args.max_steps} "
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
                and stale_validations
                >= int(args.early_stopping_patience)
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
    )
    with (output / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    elapsed = time.monotonic() - started
    atomic_json(output / "metrics.json", {
        "schema": "synthetic_union_support_gain_pretraining_v1",
        "status": "complete",
        "map_data_used": False,
        "radio_measurements_used": False,
        "data_generation": "continuous; no finite training set reused",
        "training_target": (
            "one scalar log1p relative marginal raw support-count gain"
        ),
        "training_loss": "mean squared error only",
        "probability_output": False,
        "intensity_output": False,
        "ranking_loss": False,
        "autoencoder": False,
        "feature_schema": SCALAR_PLANE_FEATURE_SCHEMA,
        "architecture": {
            "plane_encoder": "shared MLP plus mean/max pooling",
            "bank_encoder": "same plane encoder over the union set",
            "acquisition_head": "one scalar",
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
        },
        "synthetic_config": asdict(cfg),
        "validation_config": asdict(validation_cfg),
        "max_sample_count": int(args.max_sample_count),
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": float(args.learning_rate),
            "max_steps": int(args.max_steps),
            "actual_steps": int(actual_steps),
            "groups_per_batch": int(args.groups_per_batch),
            "training_examples_generated": (
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
        f"[UNION-ACQ] saved={output} best_step={best_step} "
        f"actual_steps={actual_steps} early_stopped={int(early_stopped)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
