#!/usr/bin/env python3
"""Summarize the matched two-seed N/S policy-versus-random sweep."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


VEHICLES = (10, 20, 30, 40)
WINDOWS = (1, 5, 10, 20)
SEEDS = (1, 2)
METHODS = ("policy", "random")
TAIL_STEPS = (900, 925, 950, 975, 1000)
METRIC = "t2_b0_individual_total"


def _read_tail(path: Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {int(row["step"]): float(row[METRIC]) for row in rows}
    missing = set(TAIL_STEPS).difference(values)
    if missing:
        raise ValueError(f"{path}: missing tail steps {sorted(missing)}")
    return {step: values[step] for step in TAIL_STEPS}


def summarize(root: Path) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    for vehicles in VEHICLES:
        for window in WINDOWS:
            by_method: dict[str, dict[int, dict[int, float]]] = {}
            for method in METHODS:
                by_method[method] = {}
                for seed in SEEDS:
                    path = (
                        root
                        / f"n{vehicles:03d}"
                        / f"s{window:02d}"
                        / method
                        / f"seed_{seed:02d}"
                        / "fidelity.csv"
                    )
                    by_method[method][seed] = _read_tail(path)

            policy_values = [
                by_method["policy"][seed][step]
                for seed in SEEDS
                for step in TAIL_STEPS
            ]
            random_values = [
                by_method["random"][seed][step]
                for seed in SEEDS
                for step in TAIL_STEPS
            ]
            paired_gaps = [
                by_method["random"][seed][step]
                - by_method["policy"][seed][step]
                for seed in SEEDS
                for step in TAIL_STEPS
            ]
            seed_wins = sum(
                statistics.mean(by_method["policy"][seed].values())
                < statistics.mean(by_method["random"][seed].values())
                for seed in SEEDS
            )
            policy_mean = statistics.mean(policy_values)
            random_mean = statistics.mean(random_values)
            summaries.append(
                {
                    "vehicles": vehicles,
                    "window": window,
                    "policy_tail_rmse": policy_mean,
                    "random_tail_rmse": random_mean,
                    "policy_advantage_pct": (
                        100.0 * (random_mean - policy_mean) / random_mean
                    ),
                    "policy_checkpoint_wins": sum(gap > 0.0 for gap in paired_gaps),
                    "policy_seed_wins": seed_wins,
                    "minimum_paired_gap": min(paired_gaps),
                }
            )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root.resolve())
    fields = list(rows[0])
    print(" | ".join(fields))
    for row in rows:
        print(
            " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row.values()
            )
        )
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
