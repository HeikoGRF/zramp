#!/usr/bin/env python3
"""Select a policy on artificial random/on-policy validation only."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from cross_map_policy_generalization.mature_requester_policy_deployment_audit import (
    frozen_audit,
    load_checkpoint,
    transform_target,
)
from cross_map_policy_generalization.train_cross_map_encoder_audit import combine, load_seed


def load_split(root: Path, pattern: str):
    datasets = []
    maps = []
    for case_id, audit in enumerate(sorted(root.glob(pattern)), start=1):
        actual_seed = int(audit.parent.name.split("_")[-1])
        datasets.append(
            transform_target(load_seed(audit, case_id), actual_seed, "delta")
        )
        maps.append(audit.parent.parent.name)
    if not datasets:
        raise FileNotFoundError(f"No cases for {pattern} under {root}")
    return combine(datasets), sorted(set(maps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--random-source-root", type=Path, required=True)
    parser.add_argument("--onpolicy-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--random-validation-tolerance-db", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    random_validation, random_maps = load_split(
        args.random_source_root, "source_valid_*/seed_*/audit"
    )
    onpolicy_validation, onpolicy_maps = load_split(
        args.onpolicy_source_root, "source_valid_*/seed_*/audit"
    )
    untouched_test, test_maps = load_split(
        args.random_source_root, "source_test_*/seed_*/audit"
    )
    candidates = [args.original_checkpoint]
    candidates.extend(
        sorted(args.candidate_root.glob("*/policy_best_validation.pt"))
    )
    rows: dict[str, object] = {}
    for path in candidates:
        checkpoint = load_checkpoint(path)
        rows[str(path)] = {
            "random_validation": frozen_audit(
                random_validation, checkpoint, device
            ),
            "onpolicy_validation": frozen_audit(
                onpolicy_validation, checkpoint, device
            ),
            "untouched_test": frozen_audit(untouched_test, checkpoint, device),
        }
    original_key = str(args.original_checkpoint)
    original_random = float(
        rows[original_key]["random_validation"]["deployable_geometry_alpha"]
        ["gain_over_random_db"]
    )
    floor = original_random - float(args.random_validation_tolerance_db)
    eligible = [
        path
        for path in candidates
        if float(
            rows[str(path)]["random_validation"]["deployable_geometry_alpha"]
            ["gain_over_random_db"]
        )
        >= floor
    ]
    selected = max(
        eligible,
        key=lambda path: (
            float(
                rows[str(path)]["onpolicy_validation"]
                ["deployable_geometry_alpha"]["gain_over_random_db"]
            ),
            float(
                rows[str(path)]["random_validation"]
                ["deployable_geometry_alpha"]["gain_over_random_db"]
            ),
            -len(str(path)),
        ),
    )
    args.selected_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, args.selected_checkpoint)
    report = {
        "protocol": {
            "fit_and_selection": "artificial maps only",
            "random_validation_maps": random_maps,
            "onpolicy_validation_maps": onpolicy_maps,
            "untouched_test_maps": test_maps,
            "kirchberg_labels_used": False,
            "selection_rule": (
                "maximize policy-induced validation gain subject to random-state "
                f"validation being within {args.random_validation_tolerance_db:.3f} dB "
                "of the original policy"
            ),
            "random_validation_floor_db": floor,
        },
        "original_checkpoint": original_key,
        "selected_source_checkpoint": str(selected),
        "selected_checkpoint": str(args.selected_checkpoint),
        "selected_result": rows[str(selected)],
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for path, row in rows.items():
        random_gain = row["random_validation"]["deployable_geometry_alpha"]["gain_over_random_db"]
        policy_gain = row["onpolicy_validation"]["deployable_geometry_alpha"]["gain_over_random_db"]
        test_gain = row["untouched_test"]["deployable_geometry_alpha"]["gain_over_random_db"]
        print(
            f"{Path(path).parent.name:32s} random_valid={random_gain:+.3f} "
            f"onpolicy_valid={policy_gain:+.3f} test={test_gain:+.3f}",
            flush=True,
        )
    print(f"[SELECTED] {selected}", flush=True)
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
