#!/usr/bin/env python3
"""Aggregate the final nine-zone, five-temporal-replicate paper results.

The aggregation unit for confidence intervals is one temporal replicate. Each
replicate first averages the nine factorial zones. Run-level tail metrics are
recomputed from ``temporal_evaluation.values`` so non-finite evaluations are
removed consistently. This handles the shared step-1724 NaNs in B1V1-rep5 and
B3V1-rep5 by averaging their remaining nine valid tail evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ZONES = tuple(
    f"factor_b{building}_v{vehicles}"
    for building in range(1, 4)
    for vehicles in range(1, 4)
)
METHODS = (
    "iso",
    "central",
    "full",
    "top5",
    "every1",
    "every5",
    "every10",
    "every20",
    "every40",
    "every80",
    "ungated",
)
METHOD_LABELS = {
    "iso": "ISO",
    "central": "Central",
    "full": "Full",
    "top5": "Top 5",
    "every1": "Top 1",
    "every5": "Every 5",
    "every10": "Every 10",
    "every20": "Every 20",
    "every40": "Every 40",
    "every80": "Every 80",
    "ungated": "Ungated Greedy",
}
METRIC_KEYS = (
    "greedy_total",
    "greedy_feasible_rmse",
    "greedy_infeasible_rmse",
)
BYTE_COLUMNS = (
    "expert_bank_advertisement_bytes",
    "capsule_payload_bytes",
    "model_payload_bytes",
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    archive_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=archive_root / "results" / "paper",
        help="directory containing the four archived paper result roots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=archive_root / "figures" / "data" / "statistical_aggregation",
        help="directory for all-run, per-timeframe, and confidence-interval CSV files",
    )
    return parser.parse_args()


def run_directory(
    artifacts: Path, zone: str, replicate: int, method: str
) -> Path:
    if method == "ungated":
        return (
            artifacts
            / "luxembourg_cell_grid_ci_temporal5_paper_final_v1"
            / "methods"
            / zone
            / f"rep{replicate}"
            / "ungated_equal_greedy_paper_final_full1800_eval50_tail10x25"
        )
    if replicate >= 2:
        return (
            artifacts
            / "luxembourg_cell_grid_ci_temporal5_paper_final_v1"
            / "methods"
            / zone
            / f"rep{replicate}"
            / f"{method}_paper_final_full1800_eval50_tail10x25"
        )
    if method in {"iso", "central"}:
        run_name = (
            "cell_grid_local_only_eval50_tail10x25"
            if method == "iso"
            else "cell_grid_central_eval50_tail10x25"
        )
        return (
            artifacts
            / "luxembourg_cell_grid_factorial_sweeps_v1"
            / "methods"
            / zone
            / run_name
        )
    if method in {"full", "top5", "every1"}:
        suffix = {"full": "greedy", "top5": "top5", "every1": "top1"}[
            method
        ]
        return (
            artifacts
            / "luxembourg_cell_grid_intensity_budget_sweep_v1"
            / "methods"
            / zone
            / f"cell_grid_intensity_{suffix}_eval50_tail10x25"
        )
    interval = method.removeprefix("every")
    return (
        artifacts
        / "luxembourg_cell_grid_synchronized_9map_paper_final_v1"
        / "methods"
        / f"{zone}_300m"
        / (
            "cell_grid_intensity_top1_global_every"
            f"{interval}_paper_final_full1800_eval50_tail10x25"
        )
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tail_metrics(run_dir: Path) -> dict[str, object]:
    metrics = read_json(run_dir / "metrics.json")
    temporal = metrics.get("temporal_evaluation", {})
    if not temporal.get("complete", False):
        raise RuntimeError(f"incomplete temporal evaluation: {run_dir}")
    steps = [int(step) for step in temporal["observed_steps"]]
    values = temporal["values"]
    valid_indices = [
        index
        for index in range(len(steps))
        if all(
            math.isfinite(float(values[key][index])) for key in METRIC_KEYS
        )
    ]
    if len(valid_indices) not in {9, 10}:
        raise RuntimeError(
            f"expected nine or ten finite tail evaluations in {run_dir}, "
            f"found {len(valid_indices)}"
        )
    excluded_steps = [
        steps[index]
        for index in range(len(steps))
        if index not in valid_indices
    ]
    result: dict[str, object] = {
        key: fmean(float(values[key][index]) for index in valid_indices)
        for key in METRIC_KEYS
    }
    result["valid_tail_evaluations"] = len(valid_indices)
    result["excluded_tail_steps"] = ";".join(map(str, excluded_steps))
    return result


def communication_rate(run_dir: Path) -> float:
    path = run_dir / "sharing_events.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tail = rows[-min(250, len(rows)) :]
    if not tail:
        raise RuntimeError(f"no communication rows in {path}")
    bytes_per_step = [
        sum(float(row.get(column, 0.0) or 0.0) for column in BYTE_COLUMNS)
        for row in tail
    ]
    return fmean(bytes_per_step) / 1_000_000.0


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(artifacts: Path) -> tuple[list[dict], list[dict], list[dict]]:
    per_run: list[dict] = []
    for method in METHODS:
        for zone in ZONES:
            for replicate in range(1, 6):
                run_dir = run_directory(artifacts, zone, replicate, method)
                row = tail_metrics(run_dir)
                try:
                    archived_run_directory = run_dir.relative_to(
                        artifacts.parent.parent
                    )
                except ValueError:
                    archived_run_directory = run_dir
                row.update(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "zone": zone,
                        "replicate": replicate,
                        "communication_mb_per_s": communication_rate(run_dir),
                        "run_directory": str(archived_run_directory),
                    }
                )
                per_run.append(row)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in per_run:
        grouped[(str(row["method"]), int(row["replicate"]))].append(row)
    per_replicate: list[dict] = []
    value_fields = (*METRIC_KEYS, "communication_mb_per_s")
    for method in METHODS:
        for replicate in range(1, 6):
            rows = grouped[(method, replicate)]
            if len(rows) != len(ZONES):
                raise RuntimeError(
                    f"expected {len(ZONES)} zones for {method} replicate "
                    f"{replicate}, found {len(rows)}"
                )
            per_replicate.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "replicate": replicate,
                    "zones": len(rows),
                    **{
                        field: fmean(float(row[field]) for row in rows)
                        for field in value_fields
                    },
                }
            )

    summary: list[dict] = []
    for method in METHODS:
        rows = [row for row in per_replicate if row["method"] == method]
        summary_row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "replicates": len(rows),
            "zones_per_replicate": len(ZONES),
        }
        for field in value_fields:
            values = [float(row[field]) for row in rows]
            mean = fmean(values)
            sample_sd = stdev(values)
            standard_error = sample_sd / math.sqrt(len(values))
            summary_row[f"{field}_mean"] = mean
            summary_row[f"{field}_sample_sd"] = sample_sd
            summary_row[f"{field}_ci95_half_width"] = (
                T_CRITICAL_95_DF4 * standard_error
            )
        summary.append(summary_row)
    return per_run, per_replicate, summary


def main() -> int:
    args = parse_args()
    artifacts = args.artifact_root.resolve()
    output = args.output_dir.resolve()
    per_run, per_replicate, summary = aggregate(artifacts)
    write_csv(
        output / "all_495_runs.csv",
        per_run,
        (
            "method",
            "method_label",
            "zone",
            "replicate",
            "valid_tail_evaluations",
            "excluded_tail_steps",
            *METRIC_KEYS,
            "communication_mb_per_s",
            "run_directory",
        ),
    )
    write_csv(
        output / "per_timeframe_method_averages.csv",
        per_replicate,
        (
            "method",
            "method_label",
            "replicate",
            "zones",
            *METRIC_KEYS,
            "communication_mb_per_s",
        ),
    )
    summary_fields = ["method", "method_label", "replicates", "zones_per_replicate"]
    for field in (*METRIC_KEYS, "communication_mb_per_s"):
        summary_fields.extend(
            (
                f"{field}_mean",
                f"{field}_sample_sd",
                f"{field}_ci95_half_width",
            )
        )
    write_csv(
        output / "five_timeframe_confidence_intervals.csv",
        summary,
        tuple(summary_fields),
    )
    print(f"Wrote five-timeframe statistical tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
