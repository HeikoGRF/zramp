#!/usr/bin/env python3
"""Train an initialization-relative policy on exact synthetic deployments."""
from __future__ import annotations
import argparse
import json
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


def load_split(root: Path, split: str):
    pattern = "source_train_*" if split == "train" else "source_valid_*"
    datasets = []
    maps = []
    case_id = 1
    for audit in sorted(root.glob(f"{pattern}/seed_*/audit")):
        actual_seed = int(audit.parent.name.split("_")[-1])
        dataset = load_seed(audit, case_id)
        datasets.append(transform_target(dataset, actual_seed, "delta"))
        maps.append(audit.parent.parent.name)
        case_id += 1
    if not datasets:
        raise FileNotFoundError(f"No {split} exact audits under {root}")
    return combine(datasets), sorted(set(maps)), len(datasets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--gain-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--training-target",
        choices=("oracle", "oracle-geometry-alpha", "geometry"),
        default="oracle",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    training, source_maps, train_cases = load_split(args.source_root, "train")
    validation, validation_maps, validation_cases = load_split(args.source_root, "validation")
    if training.group_widths != validation.group_widths:
        raise ValueError("source predictor architectures differ")
    config = {
        "hidden": int(args.hidden_dim),
        "embedding": int(args.embedding_dim),
        "gain_hidden": int(args.gain_hidden_dim),
    }
    state, metrics, best_epoch = train_central(
        training,
        validation,
        config=config,
        target=str(args.training_target),
        seed=int(args.seed),
        device=device,
        epochs=int(args.epochs),
    )
    policy = make_policy(training, config, int(args.seed), device)
    policy.load_state_dict(state)
    training_metrics = averaged_metrics(
        [
            evaluate_scores(
                training,
                predict(policy, training, device),
                str(args.training_target),
            )
        ]
    )
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
            name: value.detach().cpu() for name, value in policy.state_dict().items()
        },
        "architecture": architecture,
        "decisions_seen": len(training.examples),
        "epoch": int(best_epoch),
        "source_maps": source_maps,
        "validation_maps": validation_maps,
        "deployment_map_excluded": "single_zone_urban_220,kirchberg_north",
        "validation": metrics,
        "training_args": vars(args),
        "source_state_process": "exact decentralized deployment evolution",
        "training_target": str(args.training_target),
    }
    torch.save(checkpoint, output / "policy_best_validation.pt")
    report = {
        "format": "exact_deployment_source_policy_report_v1",
        "source_maps": source_maps,
        "validation_maps": validation_maps,
        "deployment_map_excluded": "single_zone_urban_220,kirchberg_north",
        "train_cases": train_cases,
        "validation_cases": validation_cases,
        "train_pairs": len(training.examples),
        "validation_pairs": len(validation.examples),
        "best_validation_uplift": float(metrics["gain_over_random_db"]),
        "best_epoch": int(best_epoch),
        "training": training_metrics,
        "validation": metrics,
        "architecture": architecture,
        "policy_input": "initialization-relative exact model state plus remotely encoded private history",
        "target_labels_used": False,
        "training_target": str(args.training_target),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
