#!/usr/bin/env python3
"""Evaluate source-pretrained policy candidates on unseen artificial maps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cross_map_policy_generalization.mature_requester_policy_deployment_audit import (
    checkpoint_candidates,
    frozen_audit,
    load_checkpoint,
    transform_target,
)
from cross_map_policy_generalization.train_cross_map_encoder_audit import combine, load_seed


def architecture_key(checkpoint: dict) -> tuple[int, int, int]:
    row = checkpoint["architecture"]
    return (
        int(row["hidden_dim"]),
        int(row["embedding_dim"]),
        int(row["gain_hidden_dim"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    _selected, candidates = checkpoint_candidates(args.policy_root)
    datasets = []
    maps = []
    for case_id, audit in enumerate(
        sorted(args.source_root.glob("source_test_*/seed_*/audit")), start=1
    ):
        actual_seed = int(audit.parent.name.split("_")[-1])
        datasets.append(
            transform_target(load_seed(audit, case_id), actual_seed, "delta")
        )
        maps.append(audit.parent.parent.name)
    if not datasets:
        raise FileNotFoundError(f"No additional holdouts under {args.source_root}")
    holdout = combine(datasets)
    results: dict[str, object] = {}
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        path = Path(str(candidate["checkpoint"]))
        checkpoint = load_checkpoint(path)
        key = architecture_key(checkpoint)
        groups[key].append(candidate)
        results[str(path)] = {
            "architecture": {
                "hidden_dim": key[0],
                "embedding_dim": key[1],
                "gain_hidden_dim": key[2],
            },
            "source_validation_gain_db": float(candidate["best_validation_uplift"]),
            "additional_holdout": frozen_audit(holdout, checkpoint, device),
        }
    architecture_rows = []
    for key, rows in groups.items():
        architecture_rows.append(
            {
                "architecture": key,
                "source_validation_mean_db": float(
                    np.mean([float(row["best_validation_uplift"]) for row in rows])
                ),
                "candidates": [str(row["checkpoint"]) for row in rows],
            }
        )
    best_mean = max(row["source_validation_mean_db"] for row in architecture_rows)
    eligible = [
        row for row in architecture_rows
        if row["source_validation_mean_db"] >= best_mean - 0.02
    ]
    chosen_architecture = min(
        eligible,
        key=lambda row: (
            sum(int(value) for value in row["architecture"]),
            tuple(int(value) for value in row["architecture"]),
        ),
    )
    chosen_checkpoint = max(
        chosen_architecture["candidates"],
        key=lambda path: float(results[path]["source_validation_gain_db"]),
    )
    report = {
        "protocol": {
            "maps": maps,
            "maps_used_for_fitting_or_selection": False,
            "selection_rule": "smallest architecture within 0.02 dB of best two-seed artificial-validation mean; best source-validation seed within architecture",
            "target_labels_used": False,
        },
        "architecture_source_validation": architecture_rows,
        "complexity_selected_architecture": chosen_architecture,
        "complexity_selected_checkpoint": chosen_checkpoint,
        "complexity_selected_result": results[chosen_checkpoint],
        "all_candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    metric = results[chosen_checkpoint]["additional_holdout"]["deployable_geometry_alpha"]
    print(
        f"[SELECTED] {chosen_checkpoint} gain={metric['gain_over_random_db']:.3f} "
        f"order={metric['pairwise_order_accuracy']:.3f}",
        flush=True,
    )
    for path, row in results.items():
        metric = row["additional_holdout"]["deployable_geometry_alpha"]
        print(
            f"{Path(path).parent.name:48s} gain={metric['gain_over_random_db']:+.3f} "
            f"order={metric['pairwise_order_accuracy']:.3f}",
            flush=True,
        )
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
