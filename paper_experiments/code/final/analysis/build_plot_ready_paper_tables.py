#!/usr/bin/env python3
"""Validate five-timeframe aggregates and export final paper plot tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


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
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
SWEEP = (
    ("full", 0, "all", 1),
    ("top5", 1, "5", 1),
    ("every1", 2, "1", 1),
    ("every5", 3, "1", 5),
    ("every10", 4, "1", 10),
    ("every20", 5, "1", 20),
    ("every40", 6, "1", 40),
    ("every80", 7, "1", 80),
)
VALUE_FIELDS = (
    "greedy_total",
    "greedy_feasible_rmse",
    "greedy_infeasible_rmse",
    "communication_mb_per_s",
)
PLOT_NAMES = {
    "greedy_total": "rmse",
    "greedy_feasible_rmse": "feasible_rmse",
    "greedy_infeasible_rmse": "nonfeasible_rmse",
    "communication_mb_per_s": "communication_mb_per_s",
}
PLOT_VALUE_FIELDS = (
    "rmse",
    "rmse_ci95",
    "feasible_rmse",
    "feasible_rmse_ci95",
    "nonfeasible_rmse",
    "nonfeasible_rmse_ci95",
    "communication_mb_per_s",
    "communication_mb_per_s_ci95",
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    archive_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation-dir",
        type=Path,
        default=archive_root / "figures" / "data" / "statistical_aggregation",
        help=(
            "directory containing all_495_runs.csv, "
            "per_timeframe_method_averages.csv, and "
            "five_timeframe_confidence_intervals.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=archive_root / "figures" / "data" / "plot_ready_tables",
        help="destination for validated plot-ready tables",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    return rows


def finite_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {field} in {context}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field} in {context}")
    return value


def integer(row: dict[str, str], field: str, context: str) -> int:
    value = finite_float(row, field, context)
    if not value.is_integer():
        raise RuntimeError(f"non-integer {field} in {context}: {value}")
    return int(value)


def index_by_method(
    rows: list[dict[str, str]], source: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        method = row.get("method", "")
        if method in indexed:
            raise RuntimeError(f"duplicate method {method!r} in {source}")
        indexed[method] = row
    expected = set(METHODS)
    actual = set(indexed)
    if actual != expected:
        raise RuntimeError(
            f"method mismatch in {source}; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return indexed


def parse_zone(zone: str) -> tuple[int, int]:
    prefix = "factor_b"
    if not zone.startswith(prefix) or "_v" not in zone:
        raise RuntimeError(f"invalid factorial zone name: {zone!r}")
    building_text, vehicle_text = zone[len(prefix) :].split("_v", 1)
    try:
        building = int(building_text)
        vehicle = int(vehicle_text)
    except ValueError as exc:
        raise RuntimeError(f"invalid factorial zone name: {zone!r}") from exc
    if building not in {1, 2, 3} or vehicle not in {1, 2, 3}:
        raise RuntimeError(f"factorial scores out of range: {zone!r}")
    return building, vehicle


def plot_values_from_summary(
    row: dict[str, str], context: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for source, target in PLOT_NAMES.items():
        values[target] = finite_float(row, f"{source}_mean", context)
        values[f"{target}_ci95"] = finite_float(
            row, f"{source}_ci95_half_width", context
        )
    return values


def summarize_values(rows: list[dict[str, object]]) -> dict[str, float]:
    if len(rows) != 5:
        raise RuntimeError(f"expected five replicate values, found {len(rows)}")
    result: dict[str, float] = {}
    for source, target in PLOT_NAMES.items():
        values = [float(row[source]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite value while summarizing {source}")
        result[target] = fmean(values)
        result[f"{target}_ci95"] = (
            T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(len(values))
        )
    return result


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    aggregation_dir = args.aggregation_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_paths = {
        name: aggregation_dir / name
        for name in (
            "all_495_runs.csv",
            "per_timeframe_method_averages.csv",
            "five_timeframe_confidence_intervals.csv",
        )
    }
    per_run = read_csv(source_paths["all_495_runs.csv"])
    per_replicate = read_csv(source_paths["per_timeframe_method_averages.csv"])
    summary = read_csv(source_paths["five_timeframe_confidence_intervals.csv"])
    summary_by_method = index_by_method(
        summary, "five_timeframe_confidence_intervals.csv"
    )

    method_summary: list[dict] = []
    for method in METHODS:
        source = summary_by_method[method]
        context = f"five_timeframe_confidence_intervals.csv method {method}"
        replicates = integer(source, "replicates", context)
        zones = integer(source, "zones_per_replicate", context)
        if replicates != 5 or zones != 9:
            raise RuntimeError(
                f"expected five replicates and nine zones for {method}, "
                f"found {replicates} and {zones}"
            )
        method_summary.append(
            {
                "method_order": METHOD_ORDER[method],
                "method": method,
                "method_label": source["method_label"],
                "replicates": replicates,
                "zones_per_replicate": zones,
                **plot_values_from_summary(source, context),
            }
        )

    method_fields = (
        "method_order",
        "method",
        "method_label",
        "replicates",
        "zones_per_replicate",
        *PLOT_VALUE_FIELDS,
    )
    write_csv(
        output_dir / "method_performance_with_confidence_intervals.csv",
        method_summary,
        method_fields,
    )

    method_rows = {row["method"]: row for row in method_summary}
    sweep_rows = []
    for method, policy_index, pull_budget, interval in SWEEP:
        sweep_rows.append(
            {
                "policy_index": policy_index,
                "pull_budget": pull_budget,
                "pull_interval_steps": interval,
                **method_rows[method],
            }
        )
    write_csv(
        output_dir / "communication_frequency_tradeoff.csv",
        sweep_rows,
        (
            "policy_index",
            "pull_budget",
            "pull_interval_steps",
            *method_fields,
        ),
    )

    replicate_rows: list[dict] = []
    replicate_groups: dict[str, list[int]] = defaultdict(list)
    for source in per_replicate:
        method = source.get("method", "")
        if method not in METHOD_ORDER:
            raise RuntimeError(
                "unexpected method in per_timeframe_method_averages.csv: "
                f"{method!r}"
            )
        context = f"per_timeframe_method_averages.csv method {method}"
        replicate = integer(source, "replicate", context)
        zones = integer(source, "zones", context)
        if replicate not in range(1, 6) or zones != 9:
            raise RuntimeError(
                f"invalid replicate/zones for {method}: {replicate}/{zones}"
            )
        replicate_groups[method].append(replicate)
        row = {
            "method_order": METHOD_ORDER[method],
            "method": method,
            "method_label": source["method_label"],
            "replicate": replicate,
            "zones": zones,
        }
        for field, target in PLOT_NAMES.items():
            row[target] = finite_float(source, field, context)
        replicate_rows.append(row)
    for method in METHODS:
        if sorted(replicate_groups[method]) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"incomplete replicates for {method}")
    replicate_rows.sort(
        key=lambda row: (row["method_order"], row["replicate"])
    )
    write_csv(
        output_dir / "per_timeframe_method_plot_data.csv",
        replicate_rows,
        (
            "method_order",
            "method",
            "method_label",
            "replicate",
            "zones",
            "rmse",
            "feasible_rmse",
            "nonfeasible_rmse",
            "communication_mb_per_s",
        ),
    )

    run_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    density_replicates: dict[
        tuple[str, str, int, int], list[dict[str, object]]
    ] = defaultdict(list)
    seen_runs: set[tuple[str, str, int]] = set()
    labels: dict[str, str] = {}
    for source in per_run:
        method = source.get("method", "")
        zone = source.get("zone", "")
        if method not in METHOD_ORDER:
            raise RuntimeError(
                f"unexpected method in all_495_runs.csv: {method!r}"
            )
        building, vehicle = parse_zone(zone)
        context = f"all_495_runs.csv method {method}, zone {zone}"
        replicate = integer(source, "replicate", context)
        run_key = (method, zone, replicate)
        if run_key in seen_runs:
            raise RuntimeError(f"duplicate run row: {run_key}")
        seen_runs.add(run_key)
        labels[method] = source["method_label"]
        values: dict[str, object] = {
            field: finite_float(source, field, context)
            for field in VALUE_FIELDS
        }
        values["replicate"] = replicate
        run_groups[(method, zone)].append(values)
        density_replicates[
            (method, "building", building, replicate)
        ].append(values)
        density_replicates[(method, "traffic", vehicle, replicate)].append(
            values
        )
    expected_run_count = len(METHODS) * 9 * 5
    if len(seen_runs) != expected_run_count:
        raise RuntimeError(
            f"expected {expected_run_count} per-run rows, found {len(seen_runs)}"
        )

    zone_rows = []
    for method in METHODS:
        for building in range(1, 4):
            for vehicle in range(1, 4):
                zone = f"factor_b{building}_v{vehicle}"
                values = run_groups[(method, zone)]
                if sorted(int(row["replicate"]) for row in values) != [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]:
                    raise RuntimeError(
                        f"incomplete per-run group for {method}/{zone}"
                    )
                zone_rows.append(
                    {
                        "method_order": METHOD_ORDER[method],
                        "method": method,
                        "method_label": labels[method],
                        "zone": zone,
                        "building_score": building,
                        "vehicle_score": vehicle,
                        "replicates": 5,
                        **summarize_values(values),
                    }
                )
    write_csv(
        output_dir / "per_map_method_summary.csv",
        zone_rows,
        (
            "method_order",
            "method",
            "method_label",
            "zone",
            "building_score",
            "vehicle_score",
            "replicates",
            *PLOT_VALUE_FIELDS,
        ),
    )

    for dimension, filename in (
        ("building", "building_density_plot_data.csv"),
        ("traffic", "traffic_density_plot_data.csv"),
    ):
        density_rows = []
        for method in METHODS:
            for score in range(1, 4):
                replicate_means: list[dict[str, object]] = []
                for replicate in range(1, 6):
                    values = density_replicates[
                        (method, dimension, score, replicate)
                    ]
                    if len(values) != 3:
                        raise RuntimeError(
                            f"expected three zones for {method}/{dimension}"
                            f"{score}/rep{replicate}, found {len(values)}"
                        )
                    replicate_means.append(
                        {
                            field: fmean(float(row[field]) for row in values)
                            for field in VALUE_FIELDS
                        }
                    )
                density_rows.append(
                    {
                        "method_order": METHOD_ORDER[method],
                        "method": method,
                        "method_label": labels[method],
                        "density_score": score,
                        "replicates": 5,
                        "zones_per_replicate": 3,
                        **summarize_values(replicate_means),
                    }
                )
        write_csv(
            output_dir / filename,
            density_rows,
            (
                "method_order",
                "method",
                "method_label",
                "density_score",
                "replicates",
                "zones_per_replicate",
                *PLOT_VALUE_FIELDS,
            ),
        )

    print(f"Wrote final paper plot data to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
