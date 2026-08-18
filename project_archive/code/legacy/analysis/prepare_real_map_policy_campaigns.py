#!/usr/bin/env python3
"""Create compact archive records for the two real-Luxembourg policy campaigns.

The original campaign directories contain multi-gigabyte, per-decision debug
tables.  This script keeps the exact run settings and final records while
reducing the large validation, policy-training, fidelity, and communication
tables to per-step review histories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(
    "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city"
)

METHOD_NAMES = {
    "policy": "learned_policy",
    "random": "uninformed_selection",
}

EXACT_FILES = (
    "config.json",
    "role_experiment_config.json",
    "communication_overhead_assumptions.json",
    "progress.json",
    "final_fidelity.json",
    "latest_fidelity.json",
    "learning_summary.json",
    "local_policy_summary.json",
    "local_policy_training.csv",
    "kirchberg_penalty_calibration.json",
)

CONFIG_FILES = (
    "config.json",
    "role_experiment_config.json",
    "communication_overhead_assumptions.json",
)

FIDELITY_FIELDS = (
    "step",
    "eval_is_final",
    "eval_n_pairs_per_zone",
    "t2_b0_total",
    "t2_b0_population_rmse",
    "t2_b0_route_weighted_rmse_total",
    "t2_b0_balanced_rmse_total",
    "t2_b0_reachable_rmse_total",
    "t2_b0_censored_rmse_total",
    "t2_b0_unavailable_censored_rmse_total",
    "t2_b0_false_reachable_rate_total",
    "t2_b0_active_models_total",
)

COMMUNICATION_FIELDS = (
    "step",
    "feasible_contact_pairs",
    "feasible_contact_decisions",
    "t2_b0_events",
    "t2_b0_pull_events",
    "t2_b0_valid_pull_events",
    "t2_b0_comm_mb",
    "t2_b0_comm_cumulative_mb",
    "t2_b0_metadata_bytes",
    "t2_b0_model_bytes",
    "t2_b0_policy_bytes",
    "t2_b0_training_sample_bytes",
)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def write_csv(
    path: Path, rows: Iterable[dict[str, object]], fields: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy_named_files(
    source: Path, destination: Path, names: tuple[str, ...]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_path = source / name
        if not source_path.is_file():
            continue
        # A completed run already has a canonical final snapshot.
        if name == "latest_fidelity.json" and (source / "final_fidelity.json").is_file():
            continue
        destination_path = destination / name
        if source_path.suffix != ".json":
            shutil.copy2(source_path, destination_path)
            continue
        payload = read_json(source_path)
        # Drop settings belonging only to an intentionally omitted comparison
        # protocol. They are unused by every run retained by this extractor.
        filtered = {
            key: value
            for key, value in payload.items()
            if not (
                "match" in key.lower()
                and (
                    "schedule" in key.lower()
                    or "pull_opportunit" in key.lower()
                )
            )
        }
        destination_path.write_text(
            json.dumps(filtered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def compact_columns(source: Path, destination: Path, requested: tuple[str, ...]) -> None:
    if not source.is_file():
        return
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        fields = tuple(field for field in requested if field in reader.fieldnames)
        rows = ({field: row.get(field, "") for field in fields} for row in reader)
        write_csv(destination, rows, fields)


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def pearson(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, n: int) -> str:
    if n < 2:
        return ""
    variance_x = n * sum_x2 - sum_x * sum_x
    variance_y = n * sum_y2 - sum_y * sum_y
    if variance_x <= 0 or variance_y <= 0:
        return ""
    return f"{(n * sum_xy - sum_x * sum_y) / math.sqrt(variance_x * variance_y):.9f}"


def compact_validation(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    grouped: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            values = grouped[step]
            values["pulls"] += 1
            values["valid_pulls"] += int(float(row.get("valid", "0") or 0))
            values["adopted_pulls"] += int(float(row.get("adopted", "0") or 0))
            reward = optional_float(row.get("joint_reward"))
            if reward is not None:
                values["reward_count"] += 1
                values["joint_reward_sum"] += reward
                values["positive_joint_rewards"] += int(reward > 0)
            alpha = optional_float(row.get("alpha"))
            if alpha is not None:
                values["alpha_count"] += 1
                values["alpha_sum"] += alpha
            values["model_bytes"] += float(row.get("model_bytes", "0") or 0)
            values["scalar_bytes"] += float(row.get("scalar_bytes", "0") or 0)

    rows: list[dict[str, object]] = []
    cumulative_pulls = 0
    cumulative_adoptions = 0
    for step, values in sorted(grouped.items()):
        pulls = int(values["pulls"])
        adopted = int(values["adopted_pulls"])
        cumulative_pulls += pulls
        cumulative_adoptions += adopted
        reward_count = int(values["reward_count"])
        alpha_count = int(values["alpha_count"])
        rows.append(
            {
                "step": step,
                "pulls": pulls,
                "valid_pulls": int(values["valid_pulls"]),
                "adopted_pulls": adopted,
                "positive_joint_rewards": int(values["positive_joint_rewards"]),
                "mean_joint_reward": (
                    f"{values['joint_reward_sum'] / reward_count:.9f}"
                    if reward_count
                    else ""
                ),
                "mean_alpha": (
                    f"{values['alpha_sum'] / alpha_count:.9f}" if alpha_count else ""
                ),
                "model_bytes": int(values["model_bytes"]),
                "scalar_bytes": int(values["scalar_bytes"]),
                "cumulative_pulls": cumulative_pulls,
                "cumulative_adoptions": cumulative_adoptions,
            }
        )
    write_csv(
        destination,
        rows,
        (
            "step",
            "pulls",
            "valid_pulls",
            "adopted_pulls",
            "positive_joint_rewards",
            "mean_joint_reward",
            "mean_alpha",
            "model_bytes",
            "scalar_bytes",
            "cumulative_pulls",
            "cumulative_adoptions",
        ),
    )


def compact_policy_training(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        return
    grouped: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "target_gain" not in reader.fieldnames:
            return
        for row in reader:
            target = optional_float(row.get("target_gain"))
            prediction = optional_float(row.get("online_prediction"))
            if target is None or prediction is None:
                continue
            step = int(row["step"])
            values = grouped[step]
            error = prediction - target
            values["examples"] += 1
            values["target"] += target
            values["prediction"] += prediction
            values["abs_error"] += abs(error)
            values["squared_error"] += error * error
            values["target2"] += target * target
            values["prediction2"] += prediction * prediction
            values["cross"] += target * prediction

    rows: list[dict[str, object]] = []
    cumulative_examples = 0
    for step, values in sorted(grouped.items()):
        n = int(values["examples"])
        cumulative_examples += n
        rows.append(
            {
                "step": step,
                "examples": n,
                "mean_target_gain": f"{values['target'] / n:.9f}",
                "mean_online_prediction": f"{values['prediction'] / n:.9f}",
                "mean_absolute_error": f"{values['abs_error'] / n:.9f}",
                "root_mean_squared_error": f"{math.sqrt(values['squared_error'] / n):.9f}",
                "target_prediction_pearson": pearson(
                    values["target"],
                    values["prediction"],
                    values["target2"],
                    values["prediction2"],
                    values["cross"],
                    n,
                ),
                "cumulative_examples": cumulative_examples,
            }
        )
    write_csv(
        destination,
        rows,
        (
            "step",
            "examples",
            "mean_target_gain",
            "mean_online_prediction",
            "mean_absolute_error",
            "root_mean_squared_error",
            "target_prediction_pearson",
            "cumulative_examples",
        ),
    )


def compact_run(source: Path, destination: Path, config_destination: Path) -> dict[str, object]:
    copy_named_files(source, destination, EXACT_FILES)
    copy_named_files(source, config_destination, CONFIG_FILES)
    compact_columns(source / "fidelity.csv", destination / "fidelity_history.csv", FIDELITY_FIELDS)
    compact_columns(
        source / "sharing_events.csv",
        destination / "communication_history.csv",
        COMMUNICATION_FIELDS,
    )
    compact_validation(
        source / "cross_validation_pulls.csv",
        destination / "validation_outcomes_by_step.csv",
    )
    compact_policy_training(
        source / "exact_policy_training.csv",
        destination / "policy_training_by_step.csv",
    )

    progress = read_json(source / "progress.json")
    complete = (
        progress.get("reason") == "completed"
        and int(progress.get("step", -1)) == int(progress.get("requested_steps", -2))
        and (source / "final_fidelity.json").is_file()
    )
    final_rmse: object = ""
    if complete:
        final_rmse = read_json(source / "final_fidelity.json").get("t2_b0_total", "")
    return {
        "status": "complete" if complete else "incomplete",
        "completion_reason": progress.get("reason", ""),
        "completed_step": progress.get("step", ""),
        "requested_steps": progress.get("requested_steps", ""),
        "final_rmse_db": final_rmse,
    }


def prepare_kirchberg(data_root: Path, archive_root: Path) -> None:
    source_root = data_root / "kirchberg_north_30min" / "floating_structure_private_reward_v1"
    result_root = (
        archive_root
        / "results"
        / "legacy_experiments"
        / "kirchberg_online_private_validation_policy"
    )
    config_root = (
        archive_root
        / "experiment_configs"
        / "experimental"
        / "kirchberg_online_private_validation_policy"
    )
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        source = source_root / f"seed_{seed:02d}" / "policy"
        destination = result_root / "runs" / f"seed_{seed:02d}" / "learned_policy"
        config_destination = config_root / f"seed_{seed:02d}" / "learned_policy"
        status = compact_run(source, destination, config_destination)
        rows.append(
            {
                "seed": seed,
                "method": "learned_policy",
                **status,
                "original_campaign_id": "floating_structure_private_reward_v1",
                "original_run_suffix": f"seed_{seed:02d}/policy",
            }
        )
    write_csv(
        result_root / "run_inventory.csv",
        rows,
        (
            "seed",
            "method",
            "status",
            "completion_reason",
            "completed_step",
            "requested_steps",
            "final_rmse_db",
            "original_campaign_id",
            "original_run_suffix",
        ),
    )


def prepare_global_sender(data_root: Path, archive_root: Path) -> None:
    source_root = (
        data_root
        / "gare_bonnevoie_30min"
        / "results_1az_global_tiny_policy_rate_sweep_v1"
    )
    result_root = (
        archive_root
        / "results"
        / "legacy_experiments"
        / "receiver_in_zone_global_sender_policy_sweep"
    )
    config_root = (
        archive_root
        / "experiment_configs"
        / "experimental"
        / "receiver_in_zone_global_sender_policy_sweep"
    )
    settings = (
        ("k1_s1", 1, 1),
        ("k1_s2", 1, 2),
        ("k1_s5", 1, 5),
        ("k1_s10", 1, 10),
        ("k1_s25", 1, 25),
        ("k1_s50", 1, 50),
        ("k2_s1", 2, 1),
        ("k4_s1", 4, 1),
    )
    rows: list[dict[str, object]] = []
    for original_setting, pulls, window in settings:
        archive_setting = f"pulls_{pulls}_window_{window:03d}"
        for original_method, archive_method in METHOD_NAMES.items():
            for seed in (1, 2):
                source = (
                    source_root
                    / original_setting
                    / original_method
                    / f"seed_{seed}"
                )
                destination = (
                    result_root
                    / "runs"
                    / archive_setting
                    / archive_method
                    / f"seed_{seed:02d}"
                )
                config_destination = (
                    config_root
                    / archive_setting
                    / archive_method
                    / f"seed_{seed:02d}"
                )
                status = compact_run(source, destination, config_destination)
                rows.append(
                    {
                        "pulls_per_receiver_step": pulls,
                        "token_window_steps": window,
                        "seed": seed,
                        "method": archive_method,
                        **status,
                        "original_campaign_id": "results_1az_global_tiny_policy_rate_sweep_v1",
                        "original_run_suffix": (
                            f"{original_setting}/{original_method}/seed_{seed}"
                        ),
                    }
                )
    write_csv(
        result_root / "run_inventory.csv",
        rows,
        (
            "pulls_per_receiver_step",
            "token_window_steps",
            "seed",
            "method",
            "status",
            "completion_reason",
            "completed_step",
            "requested_steps",
            "final_rmse_db",
            "original_campaign_id",
            "original_run_suffix",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_kirchberg(args.data_root, args.archive_root)
    prepare_global_sender(args.data_root, args.archive_root)
    print("Prepared compact real-map policy campaign records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
