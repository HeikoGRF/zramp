#!/usr/bin/env python3
"""Cross-seed analysis of parameter-only map-improvement predictors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


IDENTITY = ("seed", "step", "mode", "receiver_idx")
EXPLICIT = (
    "alpha",
    "receiver_radial",
    "provider_radial",
    "receiver_training_stability",
    "provider_training_stability",
    "receiver_merge_stability",
    "provider_merge_stability",
    "receiver_maturity",
    "provider_maturity",
    "pair_distance",
    "novelty",
    "cosine",
    "cancellation_ratio",
    "trust_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> tuple[list[dict[str, float]], list[str]]:
    rows: list[dict[str, float]] = []
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fields:
                fields = list(reader.fieldnames or ())
            for raw in reader:
                rows.append(
                    {
                        key: float(value)
                        for key, value in raw.items()
                        if key != "mode" and value not in {"", None}
                    }
                )
    if not rows:
        raise ValueError("no audit rows found")
    return rows, fields


def deduplicate_no_pull(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        if abs(row["alpha"] - 1.0) < 1.0e-9:
            key = (
                int(row["seed"]),
                int(row["step"]),
                int(row["receiver_idx"]),
            )
            if key in seen:
                continue
            seen.add(key)
            row = dict(row)
            row["provider_idx"] = row["receiver_idx"]
            for suffix in (
                "radial",
                "training_stability",
                "merge_stability",
                "maturity",
            ):
                row[f"provider_{suffix}"] = row[f"receiver_{suffix}"]
            row["pair_distance"] = 0.0
            row["novelty"] = 0.0
            row["cosine"] = 1.0
            row["cancellation_ratio"] = 1.0
            row["trust_ratio"] = 0.0
            for name in tuple(row):
                if name.startswith("provider_embedding_"):
                    receiver_name = name.replace(
                        "provider_embedding_", "receiver_embedding_"
                    )
                    row[name] = row[receiver_name]
                elif name.startswith("provider_sketch_"):
                    receiver_name = name.replace(
                        "provider_sketch_", "receiver_sketch_"
                    )
                    row[name] = row[receiver_name]
        result.append(row)
    return result


def informative_rows(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Drop receiver/checkpoint groups where no action changes the objective."""

    groups: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(
        list
    )
    for row in rows:
        groups[
            (
                int(row["seed"]),
                int(row["step"]),
                int(row["receiver_idx"]),
            )
        ].append(row)
    return [
        row
        for group in groups.values()
        if (
            max(item["oracle_gain_db"] for item in group)
            - min(item["oracle_gain_db"] for item in group)
        )
        > 1.0e-8
        for row in group
    ]


def relational(
    rows: list[dict[str, float]],
    receiver_columns: list[str],
    provider_columns: list[str],
) -> np.ndarray:
    receiver = np.asarray(
        [[row[name] for name in receiver_columns] for row in rows],
        dtype=np.float64,
    )
    provider = np.asarray(
        [[row[name] for name in provider_columns] for row in rows],
        dtype=np.float64,
    )
    alpha = np.asarray([[row["alpha"]] for row in rows], dtype=np.float64)
    return np.concatenate(
        (
            receiver,
            provider,
            provider - receiver,
            np.abs(provider - receiver),
            provider * receiver,
            alpha,
            1.0 - alpha,
        ),
        axis=1,
    )


def feature_sets(
    rows: list[dict[str, float]], fields: list[str]
) -> dict[str, np.ndarray]:
    explicit = np.asarray(
        [[row[name] for name in EXPLICIT] for row in rows],
        dtype=np.float64,
    )
    explicit = np.concatenate(
        (
            explicit,
            np.square(explicit),
            (
                explicit[:, 0:1]
                * explicit[:, 1:]
            ),
        ),
        axis=1,
    )
    receiver_embedding = sorted(
        name for name in fields if name.startswith("receiver_embedding_")
    )
    provider_embedding = sorted(
        name for name in fields if name.startswith("provider_embedding_")
    )
    receiver_sketch = sorted(
        name for name in fields if name.startswith("receiver_sketch_")
    )
    provider_sketch = sorted(
        name for name in fields if name.startswith("provider_sketch_")
    )
    result = {
        "explicit_geometry_ridge": explicit,
        "frozen_encoder_ridge": relational(
            rows, receiver_embedding, provider_embedding
        ),
    }
    for dimension in (32, 64, 128, 256):
        if len(receiver_sketch) < dimension:
            continue
        result[f"parameter_sketch_{dimension}_ridge"] = relational(
            rows,
            receiver_sketch[:dimension],
            provider_sketch[:dimension],
        )
    if receiver_sketch:
        result["parameter_sketch_mlp"] = relational(
            rows, receiver_sketch, provider_sketch
        )
    return result


def standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    scale = np.std(train, axis=0, keepdims=True)
    scale[scale < 1.0e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def ridge_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    ridge: float = 10.0,
) -> np.ndarray:
    train_X, test_X = standardize(train_X, test_X)
    train_y_mean = float(np.mean(train_y))
    centered = train_y - train_y_mean
    if train_X.shape[1] <= train_X.shape[0]:
        gram = train_X.T @ train_X
        weights = np.linalg.solve(
            gram + ridge * np.eye(gram.shape[0]),
            train_X.T @ centered,
        )
        return test_X @ weights + train_y_mean
    kernel = train_X @ train_X.T
    dual = np.linalg.solve(
        kernel + ridge * np.eye(kernel.shape[0]), centered
    )
    return test_X @ (train_X.T @ dual) + train_y_mean


class RewardMLP(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        hidden = min(256, max(32, dimensions // 4))
        self.net = nn.Sequential(
            nn.Linear(dimensions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).reshape(-1)


def mlp_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> np.ndarray:
    train_X, test_X = standardize(train_X, test_X)
    target_mean = float(np.mean(train_y))
    target_scale = max(float(np.std(train_y)), 0.1)
    torch.manual_seed(int(seed))
    model = RewardMLP(train_X.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, weight_decay=1.0e-4
    )
    features = torch.as_tensor(train_X, dtype=torch.float32)
    targets = torch.as_tensor(
        (train_y - target_mean) / target_scale, dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(int(seed) + 17)
    for _epoch in range(max(1, int(epochs))):
        order = torch.randperm(features.shape[0], generator=generator)
        model.train()
        for start in range(0, int(order.numel()), 128):
            index = order[start : start + 128]
            prediction = model(features[index])
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets[index]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(
            torch.as_tensor(test_X, dtype=torch.float32)
        ).numpy()
    return prediction * target_scale + target_mean


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    return result


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def selection_metrics(
    rows: list[dict[str, float]], prediction: np.ndarray
) -> dict[str, float | int]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            (
                int(row["seed"]),
                int(row["step"]),
                int(row["receiver_idx"]),
            )
        ].append(index)
    selected: list[float] = []
    oracle: list[float] = []
    random_gain: list[float] = []
    top1: list[float] = []
    within_spearman: list[float] = []
    y = np.asarray([row["oracle_gain_db"] for row in rows])
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        chosen = int(idx[int(np.argmax(prediction[idx]))])
        best = float(np.max(y[idx]))
        selected.append(float(y[chosen]))
        oracle.append(best)
        random_gain.append(float(np.mean(y[idx])))
        top1.append(float(y[chosen] >= best - 1.0e-9))
        within_spearman.append(
            safe_corr(rank(prediction[idx]), rank(y[idx]))
        )
    finite_spearman = [
        value for value in within_spearman if np.isfinite(value)
    ]
    selected_array = np.asarray(selected)
    oracle_array = np.asarray(oracle)
    return {
        "groups": len(groups),
        "selected_gain_db": float(np.mean(selected_array)),
        "random_action_gain_db": float(np.mean(random_gain)),
        "oracle_gain_db": float(np.mean(oracle_array)),
        "regret_db": float(np.mean(oracle_array - selected_array)),
        "top1_fraction": float(np.mean(top1)),
        "positive_selection_fraction": float(
            np.mean(selected_array > 1.0e-9)
        ),
        "within_group_spearman": (
            float(np.mean(finite_spearman))
            if finite_spearman
            else float("nan")
        ),
        "global_pearson": safe_corr(prediction, y),
    }


def geometry_prediction(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [
            row["geometry_baseline_objective"]
            - row["geometry_objective"]
            for row in rows
        ],
        dtype=np.float64,
    )


def main() -> int:
    args = parse_args()
    raw_rows, fields = read_rows(args.csv)
    deduplicated = deduplicate_no_pull(raw_rows)
    rows = informative_rows(deduplicated)
    if not rows:
        raise ValueError(
            "no informative counterfactual groups: every provider/alpha "
            "action produced the same map RMSE"
        )
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 2:
        raise ValueError("cross-seed audit requires at least two seeds")
    features = feature_sets(rows, fields)
    target = np.asarray(
        [row["oracle_gain_db"] for row in rows], dtype=np.float64
    )
    summary: dict[str, object] = {
        "raw_rows": len(raw_rows),
        "deduplicated_rows": len(deduplicated),
        "rows": len(rows),
        "seeds": seeds,
        "test_rows_per_counterfactual": int(rows[0]["test_rows"]),
        "test_feasible_rows": int(rows[0]["test_feasible_rows"]),
        "test_unavailable_rows": int(rows[0]["test_unavailable_rows"]),
        "methods": {},
    }
    geometry_folds = []
    learned: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for held_out_seed in seeds:
        train_index = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if int(row["seed"]) != held_out_seed
            ],
            dtype=np.int64,
        )
        test_index = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if int(row["seed"]) == held_out_seed
            ],
            dtype=np.int64,
        )
        test_rows = [rows[int(index)] for index in test_index]
        geometry_folds.append(
            selection_metrics(
                test_rows, geometry_prediction(test_rows)
            )
        )
        for name, matrix in features.items():
            if name.endswith("_mlp"):
                prediction = mlp_predict(
                    matrix[train_index],
                    target[train_index],
                    matrix[test_index],
                    epochs=args.mlp_epochs,
                    seed=held_out_seed + 20260727,
                )
            else:
                prediction = ridge_predict(
                    matrix[train_index],
                    target[train_index],
                    matrix[test_index],
                )
            learned[name].append(selection_metrics(test_rows, prediction))
    learned["current_geometry_rule"] = geometry_folds
    methods: dict[str, object] = {}
    for name, folds in learned.items():
        numeric_keys = [
            key
            for key, value in folds[0].items()
            if key != "groups" and isinstance(value, (float, int))
        ]
        methods[name] = {
            "folds": folds,
            "mean": {
                key: float(
                    np.nanmean([float(fold[key]) for fold in folds])
                )
                for key in numeric_keys
            },
        }
    summary["methods"] = methods
    best_parameter = max(
        (
            (name, report["mean"]["selected_gain_db"])
            for name, report in methods.items()
            if name != "frozen_encoder_ridge"
        ),
        key=lambda pair: pair[1],
    )
    frozen_gain = methods["frozen_encoder_ridge"]["mean"][
        "selected_gain_db"
    ]
    random_gain = methods["current_geometry_rule"]["mean"][
        "random_action_gain_db"
    ]
    summary["fast_sequence_verdict"] = {
        "best_parameter_method": best_parameter[0],
        "best_parameter_selected_gain_db": best_parameter[1],
        "random_action_gain_db": random_gain,
        "parameter_signal_pass": bool(
            best_parameter[1] > random_gain + 0.25
        ),
        "frozen_encoder_selected_gain_db": frozen_gain,
        "frozen_encoder_pass": bool(frozen_gain > random_gain + 0.25),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["fast_sequence_verdict"], indent=2))
    for name, report in methods.items():
        mean = report["mean"]
        print(
            f"{name:32s} selected={mean['selected_gain_db']:+.3f} "
            f"random={mean['random_action_gain_db']:+.3f} "
            f"oracle={mean['oracle_gain_db']:+.3f} "
            f"top1={mean['top1_fraction']:.3f} "
            f"rho={mean['within_group_spearman']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
