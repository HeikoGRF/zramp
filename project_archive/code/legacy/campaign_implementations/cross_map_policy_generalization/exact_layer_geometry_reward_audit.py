#!/usr/bin/env python3
"""Post-pull reward from exact layer geometry and model-update history only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cross_map_policy_generalization.analyze_parameter_objective_audit import (
    mlp_predict,
    ridge_predict,
    selection_metrics,
)
from cross_map_policy_generalization.train_cross_map_encoder_audit import load_seed


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = float(first.norm() * second.norm())
    if denominator < 1.0e-12:
        return 1.0 if float((first - second).norm()) < 1.0e-12 else 0.0
    return float(torch.dot(first, second) / denominator)


def tensor_geometry(receiver: torch.Tensor, provider: torch.Tensor, alpha: float) -> list[float]:
    first = receiver.detach().reshape(-1).to(dtype=torch.float64)
    second = provider.detach().reshape(-1).to(dtype=torch.float64)
    delta = second - first
    aggregate = float(alpha) * first + (1.0 - float(alpha)) * second
    result: list[float] = []
    for value in (first, second, delta, aggregate):
        result.extend(
            (
                float(value.norm()),
                float(value.abs().mean()),
                float(value.std(unbiased=False)),
                float(value.abs().max()),
            )
        )
    result.extend(
        (
            cosine(first, second),
            cosine(first, delta),
            cosine(second, delta),
            cosine(aggregate, first),
            cosine(aggregate, second),
            float((aggregate - first).norm()),
            float((aggregate - second).norm()),
            float(second.norm() / max(float(first.norm()), 1.0e-8)),
        )
    )
    return result


def history_geometry(receiver: torch.Tensor, provider: torch.Tensor) -> list[float]:
    # ParameterGeometryRoleSimulation stores only four safe model-history
    # values at the tail: radial distance, training stability, merge
    # stability, and summary marker. Earlier columns are deliberately ignored.
    first = receiver.detach().to(dtype=torch.float64)[:, -4:]
    second = provider.detach().to(dtype=torch.float64)[:, -4:]

    def summary(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                value[-1],
                value.mean(dim=0),
                value.std(dim=0, unbiased=False),
                value[-1] - value[0],
            )
        )

    first_summary = summary(first)
    second_summary = summary(second)
    return torch.cat(
        (
            first_summary,
            second_summary,
            second_summary - first_summary,
            torch.abs(second_summary - first_summary),
            first_summary * second_summary,
        )
    ).cpu().numpy().astype(np.float64).tolist()


def selected_alpha(directory: Path) -> dict[tuple[int, str, int, int], float]:
    result: dict[tuple[int, str, int, int], float] = {}
    with (directory / "parameter_objective_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["step"]),
                str(row["mode"]),
                int(row["receiver_idx"]),
                int(row["provider_idx"]),
            )
            result[key] = float(row["geometry_selected_alpha"])
    return result


def load_case(directory: Path, case_id: int) -> tuple[list[dict[str, float]], np.ndarray]:
    actual_seed = int(directory.parent.name.split("_")[-1]) if directory.name == "audit" else int(directory.name.split("_")[-1])
    dataset = load_seed(directory, case_id)
    alpha = selected_alpha(directory)
    rows: list[dict[str, float]] = []
    features: list[list[float]] = []
    for example in dataset.examples:
        key = (example.step, example.mode, example.receiver, example.provider)
        weight = float(alpha[key])
        receiver = dataset.states[example.receiver_state]
        provider = dataset.states[example.provider_state]
        values: list[float] = [weight, 1.0 - weight]
        all_receiver: list[torch.Tensor] = []
        all_provider: list[torch.Tensor] = []
        for first, second in zip(receiver.model_groups, provider.model_groups):
            values.extend(tensor_geometry(first, second, weight))
            all_receiver.append(first.reshape(-1))
            all_provider.append(second.reshape(-1))
        values.extend(
            tensor_geometry(
                torch.cat(all_receiver), torch.cat(all_provider), weight
            )
        )
        values.extend(history_geometry(receiver.trajectory, provider.trajectory))
        features.append(values)
        rows.append(
            {
                "seed": float(case_id),
                "step": float(example.step),
                "receiver_idx": float(example.receiver),
                "provider_idx": float(example.provider),
                "oracle_gain_db": float(example.geometry_gain),
                "actual_seed": float(actual_seed),
            }
        )
    return rows, np.asarray(features, dtype=np.float64)


def discover(source_root: Path, target_root: Path):
    specs: list[tuple[str, Path]] = []
    specs.extend(("train", p) for p in sorted(source_root.glob("source_train_*/seed_*/audit")))
    specs.extend(("validation", p) for p in sorted(source_root.glob("source_valid_*/seed_*/audit")))
    specs.extend(("additional", p) for p in sorted(source_root.glob("source_test_*/seed_*/audit")))
    specs.extend(("tiny", p) for p in sorted((target_root / "source_tiny").glob("seed_*")))
    specs.extend(("kirchberg", p) for p in sorted((target_root / "target_kirchberg").glob("seed_*")))
    rows: list[dict[str, float]] = []
    matrices: list[np.ndarray] = []
    indices: dict[str, list[int]] = {
        name: []
        for name in ("train", "validation", "additional", "tiny", "kirchberg")
    }
    cases: list[dict[str, object]] = []
    for case_id, (split, directory) in enumerate(specs, start=1):
        case_rows, matrix = load_case(directory, case_id)
        start = len(rows)
        rows.extend(case_rows)
        matrices.append(matrix)
        indices[split].extend(range(start, len(rows)))
        cases.append({"case_id": case_id, "split": split, "path": str(directory), "rows": len(case_rows)})
    return rows, np.concatenate(matrices, axis=0), indices, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    args = parser.parse_args()
    rows, features, indices, cases = discover(args.source_root, args.target_root)
    target = np.asarray([row["oracle_gain_db"] for row in rows], dtype=np.float64)
    train = np.asarray(indices["train"], dtype=np.int64)
    order = tuple(
        name for name in ("validation", "additional", "tiny", "kirchberg")
        if indices[name]
    )
    evaluation = np.concatenate([np.asarray(indices[name], dtype=np.int64) for name in order])
    slices: dict[str, slice] = {}
    offset = 0
    for name in order:
        slices[name] = slice(offset, offset + len(indices[name]))
        offset += len(indices[name])
    predictions = {
        "exact_layer_geometry_ridge": ridge_predict(
            features[train], target[train], features[evaluation], ridge=10.0
        ),
        "exact_layer_geometry_mlp": np.mean(
            np.stack(
                [
                    mlp_predict(
                        features[train],
                        target[train],
                        features[evaluation],
                        epochs=args.mlp_epochs,
                        seed=20260750 + repeat,
                    )
                    for repeat in range(2)
                ]
            ),
            axis=0,
        ),
    }
    results: dict[str, object] = {}
    for method, prediction in predictions.items():
        results[method] = {}
        for name in order:
            split_rows = [rows[index] for index in indices[name]]
            results[method][name] = selection_metrics(
                split_rows, prediction[slices[name]]
            )
    selected = max(
        results,
        key=lambda method: (
            results[method]["validation"]["selected_gain_db"]
            - results[method]["validation"]["random_action_gain_db"]
        ),
    )
    report = {
        "protocol": {
            "fit_and_selection": "artificial maps only",
            "targets": "evaluation only",
            "features": "exact per-layer and global receiver/provider/aggregate parameter geometry plus model-update stability history",
            "positions_measurements_sample_counts_or_validation_data": False,
            "post_pull_only": True,
            "feature_dimensions": int(features.shape[1]),
        },
        "cases": cases,
        "rows": {name: len(value) for name, value in indices.items()},
        "methods": results,
        "selected_on_artificial_validation": selected,
        "selected_result": results[selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for method, result in results.items():
        print(method, end=" ")
        for name in order:
            metric = result[name]
            gain = metric["selected_gain_db"] - metric["random_action_gain_db"]
            print(f"{name}={gain:+.3f}", end=" ")
        print(flush=True)
    print(f"[SELECTED] {selected}", flush=True)
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
