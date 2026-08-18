#!/usr/bin/env python3
"""Compare current, learned exact, and oracle rewards for shared head replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cross_map_policy_generalization.analyze_parameter_objective_audit import ridge_predict
from cross_map_policy_generalization.exact_layer_geometry_reward_audit import load_case
from cross_map_policy_generalization.frozen_encoder_shared_reward_audit import run_fold
from cross_map_policy_generalization.mature_requester_policy_deployment_audit import (
    load_checkpoint,
    transform_target,
)
from cross_map_policy_generalization.train_cross_map_encoder_audit import (
    averaged_metrics,
    load_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    checkpoint = load_checkpoint(args.checkpoint)
    rows = []
    matrices = []
    split_indices = {"train": [], "tiny": [], "kirchberg": []}
    specs = []
    specs.extend(
        ("train", path)
        for path in sorted(args.source_root.glob("source_train_*/seed_*/audit"))
    )
    specs.extend(
        ("tiny", path)
        for path in sorted((args.target_root / "source_tiny").glob("seed_*"))
    )
    specs.extend(
        ("kirchberg", path)
        for path in sorted((args.target_root / "target_kirchberg").glob("seed_*"))
    )
    for case_id, (split, path) in enumerate(specs, start=1):
        case_rows, matrix = load_case(path, case_id)
        start = len(rows)
        rows.extend(case_rows)
        matrices.append(matrix)
        split_indices[split].extend(range(start, len(rows)))
    features = np.concatenate(matrices, axis=0)
    objective = np.asarray(
        [row["oracle_gain_db"] for row in rows], dtype=np.float64
    )
    source_index = np.asarray(split_indices["train"], dtype=np.int64)
    exact_predictions: dict[str, np.ndarray] = {}
    for name in ("tiny", "kirchberg"):
        index = np.asarray(split_indices[name], dtype=np.int64)
        exact_predictions[name] = ridge_predict(
            features[source_index],
            objective[source_index],
            features[index],
            ridge=10.0,
        )
    source_gain_mean = float(np.mean(objective[source_index]))
    source_gain_scale = max(float(np.std(objective[source_index])), 1.0e-3)
    current_values = []
    for audit in sorted(args.source_root.glob("source_train_*/seed_*/audit")):
        dataset = load_seed(audit, len(current_values) + 1)
        current_values.extend(float(row.geometry_reward) for row in dataset.examples)
    current_array = np.asarray(current_values, dtype=np.float64)
    current_mean = float(np.mean(current_array))
    current_scale = max(float(np.std(current_array)), 1.0e-3)
    variants = (
        "frozen",
        "shared_current_consensus",
        "shared_exact_consensus",
        "shared_oracle_consensus",
        "shared_fresh_current_consensus",
        "shared_fresh_exact_consensus",
        "shared_fresh_oracle_consensus",
    )
    report: dict[str, object] = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "encoder": "frozen",
            "head": "private anchored online adaptation",
            "shared_payload": "embedding pair, scalar post-pull reward, id, step",
            "exact_reward_fit": "artificial source maps only",
            "exact_reward_inputs": "exact per-layer parameters, aggregate geometry, alpha, model-update history",
            "target_oracle_reward": "positive control only",
            "minimum_samples": int(args.minimum_samples),
            "replay_capacity": 64,
            "bundle_capacity": 32,
            "learning_rate": float(args.learning_rate),
            "head_epochs": int(args.head_epochs),
            "anchor_weight": float(args.anchor_weight),
            "warmup_steps": int(args.warmup_steps),
        },
        "targets": {},
    }
    for target_name, directory in (
        ("tiny", "source_tiny"),
        ("kirchberg", "target_kirchberg"),
    ):
        first = transform_target(
            load_seed(args.target_root / directory / "seed_01", 1), 1, "delta"
        )
        second = transform_target(
            load_seed(args.target_root / directory / "seed_02", 2), 2, "delta"
        )
        size_first = len(first.examples)
        exact_by_seed = (
            exact_predictions[target_name][:size_first],
            exact_predictions[target_name][size_first:],
        )
        folds: list[dict[str, object]] = []
        for fold, (train, holdout, exact_reward) in enumerate(
            (
                (first, second, exact_by_seed[0]),
                (second, first, exact_by_seed[1]),
            )
        ):
            current_reward = np.asarray(
                [row.geometry_reward for row in train.examples], dtype=np.float64
            )
            oracle_reward = np.asarray(
                [row.geometry_gain for row in train.examples], dtype=np.float64
            )
            for variant in variants:
                if variant == "frozen":
                    reward = None
                    mean, scale = source_gain_mean, source_gain_scale
                elif "current" in variant:
                    reward = current_reward
                    mean, scale = current_mean, current_scale
                elif "exact" in variant:
                    reward = exact_reward
                    mean, scale = source_gain_mean, source_gain_scale
                else:
                    reward = oracle_reward
                    mean, scale = source_gain_mean, source_gain_scale
                row = run_fold(
                    train,
                    holdout,
                    checkpoint,
                    variant=variant,
                    seed=20260820 + fold,
                    device=device,
                    reward_mean=mean,
                    reward_scale=scale,
                    minimum_samples=int(args.minimum_samples),
                    replay_capacity=64,
                    bundle_capacity=32,
                    learning_rate=float(args.learning_rate),
                    head_epochs=int(args.head_epochs),
                    anchor_weight=float(args.anchor_weight),
                    observed_rewards=reward,
                    warmup_steps=int(args.warmup_steps),
                )
                row["fold"] = fold
                folds.append(row)
                metric = row["oracle_provider_audit"]
                print(
                    f"[{target_name}] fold={fold} variant={variant} "
                    f"gain={metric['gain_over_random_db']:.3f} "
                    f"order={metric['pairwise_order_accuracy']:.3f}",
                    flush=True,
                )
        summary: dict[str, object] = {}
        for variant in variants:
            selected = [row for row in folds if row["variant"] == variant]
            summary[variant] = {
                "oracle_provider_audit": averaged_metrics(
                    [row["oracle_provider_audit"] for row in selected]
                ),
                "deployable_geometry_alpha": averaged_metrics(
                    [row["deployable_geometry_alpha"] for row in selected]
                ),
            }
            if variant != "frozen":
                diag = [row["sample_diagnostics"] for row in selected]
                summary[variant]["sample_diagnostics"] = {
                    key: float(np.mean([float(value[key]) for value in diag]))
                    for key in diag[0]
                }
        report["targets"][target_name] = {
            "folds": folds,
            "summary": summary,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
