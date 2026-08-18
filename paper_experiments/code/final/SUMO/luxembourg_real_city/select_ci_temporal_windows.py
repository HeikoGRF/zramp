#!/usr/bin/env python3
"""Select matched, separated LuST windows for temporal CI replicates."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import statistics
import xml.etree.ElementTree as ET
from bisect import bisect_right
from pathlib import Path


def open_xml(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def load_targets(path: Path) -> dict[str, dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            b = int(row["building_score"])
            v = int(row["vehicle_score"])
            zone = f"factor_b{b}_v{v}_300m"
            targets[zone] = {
                "mean_active": float(row["mean_active"]),
                "median_active": float(row["median_active"]),
                "p95_active": float(row["p95_active"]),
                "max_active": float(row["max_active"]),
                "active_cv": float(row["active_cv"]),
            }
    return targets


def load_bounds(path: Path, zones: list[str]) -> dict[str, tuple[float, float, float, float]]:
    raw = json.loads(path.read_text())
    return {
        zone: tuple(float(value) for value in raw["crops"][zone]["bounds_sumo_xy_m"])
        for zone in zones
    }


def scan_counts(
    fcd: Path,
    bounds: dict[str, tuple[float, float, float, float]],
) -> tuple[list[int], dict[str, list[int]]]:
    times: list[int] = []
    counts = {zone: [] for zone in bounds}
    with open_xml(fcd) as stream:
        for _event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag.rsplit("}", 1)[-1] != "timestep":
                continue
            time_s = int(round(float(elem.get("time", "0"))))
            frame_counts = {zone: 0 for zone in bounds}
            for vehicle in elem:
                x = float(vehicle.get("x", "nan"))
                y = float(vehicle.get("y", "nan"))
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                for zone, (x0, y0, x1, y1) in bounds.items():
                    if x0 <= x < x1 and y0 <= y < y1:
                        frame_counts[zone] += 1
            times.append(time_s)
            for zone in bounds:
                counts[zone].append(frame_counts[zone])
            elem.clear()
    if len(times) < 2:
        raise RuntimeError(f"FCD trace has too few frames: {fcd}")
    return times, counts


def describe(values: list[int]) -> dict[str, float]:
    mean = statistics.fmean(values)
    ordered = sorted(values)
    n = len(ordered)
    p95_index = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    std = statistics.pstdev(values)
    return {
        "mean_active": mean,
        "median_active": float(statistics.median(values)),
        "p95_active": float(ordered[p95_index]),
        "max_active": float(max(values)),
        "active_cv": std / mean if mean > 0 else math.inf,
    }


def relative_error(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1.0)


def candidate_cost(stats: dict[str, float], target: dict[str, float]) -> float:
    return (
        0.55 * relative_error(stats["mean_active"], target["mean_active"])
        + 0.25 * relative_error(stats["median_active"], target["median_active"])
        + 0.15 * relative_error(stats["p95_active"], target["p95_active"])
        + 0.05 * abs(stats["active_cv"] - target["active_cv"])
    )


def select_four(
    candidates: list[dict[str, object]],
    minimum_start_separation: int,
) -> list[dict[str, object]] | None:
    ordered = sorted(candidates, key=lambda item: int(item["start_s"]))
    starts = [int(item["start_s"]) for item in ordered]
    predecessors = [
        bisect_right(starts, start - minimum_start_separation) - 1
        for start in starts
    ]
    count = len(ordered)
    infinity = float("inf")
    dp = [[infinity] * count for _ in range(5)]
    parent: list[list[int | None]] = [[None] * count for _ in range(5)]
    for index, item in enumerate(ordered):
        dp[1][index] = float(item["cost"])
    for selected_count in range(2, 5):
        for index, item in enumerate(ordered):
            limit = predecessors[index]
            best_previous = None
            best_cost = infinity
            for previous in range(limit + 1):
                if dp[selected_count - 1][previous] < best_cost:
                    best_cost = dp[selected_count - 1][previous]
                    best_previous = previous
            if best_previous is not None:
                dp[selected_count][index] = best_cost + float(item["cost"])
                parent[selected_count][index] = best_previous
    if not dp[4] or min(dp[4], default=infinity) == infinity:
        return None
    index = min(range(count), key=lambda candidate_index: dp[4][candidate_index])
    chosen_indices = [index]
    for selected_count in range(4, 1, -1):
        previous = parent[selected_count][index]
        if previous is None:
            raise RuntimeError("window-selection backtracking failed")
        chosen_indices.append(previous)
        index = previous
    return [ordered[index] for index in sorted(chosen_indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcd", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--window-seconds", type=int, default=1800)
    parser.add_argument("--candidate-stride-seconds", type=int, default=300)
    parser.add_argument("--separation-buffer-seconds", type=int, default=900)
    parser.add_argument("--existing-start-seconds", type=int, default=27900)
    parser.add_argument("--earliest-start-seconds", type=int, default=3600)
    args = parser.parse_args()

    targets = load_targets(args.targets)
    zones = sorted(targets)
    bounds = load_bounds(args.crop_manifest, zones)
    times, counts = scan_counts(args.fcd, bounds)
    sample_period = int(round(statistics.median(b - a for a, b in itertools.pairwise(times))))
    if sample_period <= 0 or args.window_seconds % sample_period:
        raise RuntimeError("window length is incompatible with the FCD sample period")
    frames_per_window = args.window_seconds // sample_period
    time_to_index = {time_s: index for index, time_s in enumerate(times)}
    latest_start = times[-1] - args.window_seconds + sample_period
    minimum_separation = args.window_seconds + args.separation_buffer_seconds
    existing_start = args.existing_start_seconds

    output_rows: list[dict[str, object]] = []
    selection_details: dict[str, object] = {}
    tolerance_levels = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50)

    for zone in zones:
        target = targets[zone]
        all_candidates: list[dict[str, object]] = []
        first_start = int(
            math.ceil(args.earliest_start_seconds / args.candidate_stride_seconds)
            * args.candidate_stride_seconds
        )
        for start_s in range(first_start, latest_start + 1, args.candidate_stride_seconds):
            if start_s not in time_to_index:
                continue
            if abs(start_s - existing_start) < minimum_separation:
                continue
            start_index = time_to_index[start_s]
            values = counts[zone][start_index : start_index + frames_per_window]
            if len(values) != frames_per_window:
                continue
            stats = describe(values)
            all_candidates.append(
                {
                    "start_s": start_s,
                    "end_s": start_s + args.window_seconds,
                    "stats": stats,
                    "cost": candidate_cost(stats, target),
                }
            )

        selected = None
        selected_tolerance = None
        for tolerance in tolerance_levels:
            eligible = [
                item
                for item in all_candidates
                if relative_error(
                    float(item["stats"]["mean_active"]), target["mean_active"]
                )
                <= tolerance
                and relative_error(
                    float(item["stats"]["median_active"]), target["median_active"]
                )
                <= tolerance + 0.05
                and relative_error(
                    float(item["stats"]["p95_active"]), target["p95_active"]
                )
                <= tolerance + 0.10
            ]
            selected = select_four(eligible, minimum_separation)
            if selected is not None:
                selected_tolerance = tolerance
                break
        if selected is None:
            selected = select_four(all_candidates, minimum_separation)
            selected_tolerance = None
        if selected is None:
            raise RuntimeError(f"could not select four separated windows for {zone}")

        output_rows.append(
            {
                "zone": zone,
                "replicate": 1,
                "start_s": existing_start,
                "end_s": existing_start + args.window_seconds,
                "existing": 1,
                "selection_tolerance": 0.0,
                **target,
            }
        )
        for replicate, item in enumerate(selected, start=2):
            stats = item["stats"]
            output_rows.append(
                {
                    "zone": zone,
                    "replicate": replicate,
                    "start_s": int(item["start_s"]),
                    "end_s": int(item["end_s"]),
                    "existing": 0,
                    "selection_tolerance": (
                        selected_tolerance if selected_tolerance is not None else "unbounded"
                    ),
                    **stats,
                }
            )
        selection_details[zone] = {
            "target": target,
            "tolerance": selected_tolerance,
            "selected": selected,
            "candidate_count": len(all_candidates),
        }

    fieldnames = [
        "zone",
        "replicate",
        "start_s",
        "end_s",
        "existing",
        "selection_tolerance",
        "mean_active",
        "median_active",
        "p95_active",
        "max_active",
        "active_cv",
    ]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    provenance = {
        "schema": "luxembourg_temporal_ci_windows_v1",
        "source_fcd": str(args.fcd.resolve()),
        "sample_period_seconds": sample_period,
        "window_seconds": args.window_seconds,
        "candidate_stride_seconds": args.candidate_stride_seconds,
        "separation_buffer_seconds": args.separation_buffer_seconds,
        "existing_start_seconds": existing_start,
        "selection": selection_details,
    }
    args.output_json.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
