#!/usr/bin/env python3
"""Pretrain an exact policy to prefer providers more mature than receivers."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from cross_map_policy_generalization.mature_requester_policy_deployment_audit import transform_target
from cross_map_policy_generalization.train_cross_map_encoder_audit import (
    averaged_metrics,
    combine,
    evaluate_scores,
    load_seed,
    make_policy,
    predict,
    train_central,
)


def relative_maturity_dataset(directory: Path, dataset):
    targets: dict[tuple[int, str, int, int], float] = {}
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
            targets.setdefault(
                key,
                float(row["provider_maturity"])
                - float(row["receiver_maturity"]),
            )
    examples = [
        replace(
            row,
            oracle_best_gain=targets[
                (
                    int(row.step),
                    str(row.mode),
                    int(row.receiver),
                    int(row.provider),
                )
            ],
        )
        for row in dataset.examples
    ]
    return type(dataset)(
        dataset.states,
        examples,
        dataset.group_widths,
        dataset.trajectory_dim,
    )


def load_cases(root: Path, split: str, case_start: int):
    pattern = "source_train_*" if split == "train" else "source_valid_*"
    datasets = []
    maps = []
    case_id = int(case_start)
    for audit in sorted(root.glob(f"{pattern}/seed_*/audit")):
        actual_seed = int(audit.parent.name.split("_")[-1])
        dataset = relative_maturity_dataset(audit, load_seed(audit, case_id))
        datasets.append(transform_target(dataset, actual_seed, "delta"))
        maps.append(audit.parent.parent.name)
        case_id += 1
    if not datasets:
        raise FileNotFoundError(f"No {split} cases under {root}")
    return combine(datasets), sorted(set(maps)), case_id


def metrics(policy, dataset, device):
    return averaged_metrics(
        [
            evaluate_scores(
                dataset,
                predict(policy, dataset, device),
                "oracle",
            )
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--onpolicy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onpolicy-repeats", type=int, default=1)
    parser.add_argument("--no-random-training", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)

    random_train, random_train_maps, next_case = load_cases(
        args.random_root, "train", 1
    )
    random_valid, random_valid_maps, next_case = load_cases(
        args.random_root, "validation", next_case
    )
    policy_train, policy_train_maps, next_case = load_cases(
        args.onpolicy_root, "train", next_case
    )
    policy_valid, policy_valid_maps, _next_case = load_cases(
        args.onpolicy_root, "validation", next_case
    )
    repeats = max(1, int(args.onpolicy_repeats))
    training_parts = [policy_train for _ in range(repeats)]
    validation_parts = [policy_valid]
    if not args.no_random_training:
        training_parts.insert(0, random_train)
        validation_parts.insert(0, random_valid)
    training = combine(training_parts)
    validation = combine(validation_parts)
    if training.group_widths != validation.group_widths:
        raise ValueError("predictor architectures differ")

    config = {"hidden": 16, "embedding": 64, "gain_hidden": 128}
    state, validation_metric, best_epoch = train_central(
        training,
        validation,
        config=config,
        target="oracle",
        seed=int(args.seed),
        device=device,
        epochs=int(args.epochs),
    )
    policy = make_policy(training, config, int(args.seed), device)
    policy.load_state_dict(state)
    split_metrics = {
        "random_training": metrics(policy, random_train, device),
        "onpolicy_training": metrics(policy, policy_train, device),
        "random_validation": metrics(policy, random_valid, device),
        "onpolicy_validation": metrics(policy, policy_valid, device),
        "combined_validation": validation_metric,
    }
    architecture = {
        "group_widths": list(policy.group_widths),
        "trajectory_dim": int(policy.trajectory_encoder.input_size),
        "hidden_dim": int(policy.layer_encoder.hidden_size),
        "embedding_dim": int(policy.embedding_dim),
        "gain_hidden_dim": int(policy.gain_head[0].out_features),
        "pair_feature_mode": str(policy.pair_feature_mode),
        "model_input_representation": "delta",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "cross_map_pretrained_exact_policy_v1",
        "policy_state_dict": {
            name: value.detach().cpu()
            for name, value in policy.state_dict().items()
        },
        "architecture": architecture,
        "decisions_seen": len(training.examples),
        "epoch": int(best_epoch),
        "source_maps": sorted(set(random_train_maps + policy_train_maps)),
        "validation_maps": sorted(set(random_valid_maps + policy_valid_maps)),
        "deployment_map_excluded": "single_zone_urban_220,kirchberg_north",
        "training_target": "provider-minus-receiver-maturity",
        "source_state_process": (
            "random plus policy-induced artificial deployment evolution"
        ),
        "onpolicy_repeats": repeats,
        "random_training_included": not bool(args.no_random_training),
        "validation": validation_metric,
    }
    torch.save(checkpoint, output / "policy_best_validation.pt")
    report = {
        "format": "relative_maturity_source_policy_report_v1",
        "protocol": {
            "fit_and_selection": "artificial maps only",
            "deployment_map_used": False,
            "random_training_included": not bool(args.no_random_training),
            "onpolicy_repeats": repeats,
            "training_target": (
                "sample-count-free provider minus receiver parameter maturity"
            ),
        },
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "train_pairs": len(training.examples),
        "validation_pairs": len(validation.examples),
        "best_validation_uplift": float(
            validation_metric["gain_over_random_db"]
        ),
        "metrics": split_metrics,
        "architecture": architecture,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
