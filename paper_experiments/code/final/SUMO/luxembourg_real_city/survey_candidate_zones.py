#!/usr/bin/env python3
"""Survey 300 m Luxembourg zones by traffic and building footprint density.

This is deliberately read-only with respect to simulation inputs.  It parses the
existing morning FCD trace and LuST building polygons, then writes a compact
candidate table and proposal figures.  It does not build scenes or traces.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree


WINDOW_M = 300.0
STRIDE_M = 100.0
BEGIN_S = 27900.0
STEPS = 1800
NETWORK_WIDTH_M = 13613.76
NETWORK_HEIGHT_M = 11455.04

EXISTING = {
    "place_wallis": (7350.0, 5900.0, 7650.0, 6200.0),
    "ville_haute": (6950.0, 7000.0, 7250.0, 7300.0),
    "belair": (5450.0, 6350.0, 5750.0, 6650.0),
    "limpertsberg_s": (6650.0, 7650.0, 6950.0, 7950.0),
    "hollerich_w": (6050.0, 5450.0, 6350.0, 5750.0),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    source = root / "SUMO" / "luxembourg_real_city" / "gare_bonnevoie"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fcd",
        type=Path,
        default=source / "traces" / "lust3d_0745_0815_period1.fcd.xml.gz",
    )
    parser.add_argument(
        "--polygons",
        type=Path,
        default=source / "map" / "sumo" / "lust3d.poly.xml",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=source / "map" / "sumo" / "lust3d.net.xml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "luxembourg_zone_survey_300m_v2",
    )
    return parser.parse_args()


def _shape_points(value: str) -> list[tuple[float, float]]:
    return [tuple(map(float, pair.split(",")[:2])) for pair in value.split()]


def load_buildings(path: Path) -> tuple[list[Polygon], np.ndarray]:
    polygons: list[Polygon] = []
    heights: list[float] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == "poly" and element.get("type") == "building":
            points = _shape_points(element.get("shape", ""))
            if len(points) >= 3:
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty and polygon.area > 1.0:
                    polygons.append(polygon)
                    heights.append(float(element.get("layer", "0") or 0.0))
        element.clear()
    return polygons, np.asarray(heights, dtype=np.float32)


def load_roads(path: Path) -> list[LineString]:
    roads: list[LineString] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == "edge" and element.get("function") != "internal":
            lanes = element.findall("lane")
            if lanes:
                shape = lanes[0].get("shape", "")
                points = _shape_points(shape) if shape else []
                if len(points) >= 2:
                    roads.append(LineString(points))
            element.clear()
    return roads


def _window_sums(grid: np.ndarray, cells: int = 3) -> np.ndarray:
    integral = np.pad(grid, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[cells:, cells:]
        - integral[:-cells, cells:]
        - integral[cells:, :-cells]
        + integral[:-cells, :-cells]
    )


def scan_traffic(path: Path) -> tuple[np.ndarray, int]:
    nx = int(math.ceil(NETWORK_WIDTH_M / STRIDE_M))
    ny = int(math.ceil(NETWORK_HEIGHT_M / STRIDE_M))
    history = np.zeros((STEPS, ny - 2, nx - 2), dtype=np.uint16)
    timestep_index = -1
    with gzip.open(path, "rb") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == "timestep":
                time_s = float(element.get("time", "-inf"))
                timestep_index = int(round(time_s - BEGIN_S))
            elif event == "end" and element.tag == "timestep":
                if 0 <= timestep_index < STEPS:
                    grid = np.zeros((ny, nx), dtype=np.uint16)
                    vehicles = element.findall("vehicle")
                    if vehicles:
                        xs = np.fromiter((float(v.get("x")) for v in vehicles), float)
                        ys = np.fromiter((float(v.get("y")) for v in vehicles), float)
                        ix = np.floor(xs / STRIDE_M).astype(int)
                        iy = np.floor(ys / STRIDE_M).astype(int)
                        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                        np.add.at(grid, (iy[valid], ix[valid]), 1)
                    history[timestep_index] = _window_sums(grid)
                element.clear()
    observed = int(np.count_nonzero(history.sum(axis=(1, 2))))
    if observed != STEPS:
        raise RuntimeError(f"expected {STEPS} populated timesteps, found {observed}")
    return history, nx


def overlaps_existing(window: Polygon) -> bool:
    return any(window.intersection(box(*bounds)).area > 0 for bounds in EXISTING.values())


def building_stats(
    window: Polygon,
    tree: STRtree,
    buildings: list[Polygon],
    heights: np.ndarray,
) -> tuple[int, float, float]:
    indices = tree.query(window, predicate="intersects")
    if len(indices) == 0:
        return 0, 0.0, 0.0
    areas = np.fromiter((buildings[int(i)].intersection(window).area for i in indices), float)
    positive = areas > 0.25
    selected = np.asarray(indices, dtype=int)[positive]
    area = float(areas[positive].sum())
    median_height = float(np.median(heights[selected])) if len(selected) else 0.0
    return int(positive.sum()), area / (WINDOW_M * WINDOW_M), median_height


def build_candidates(
    history: np.ndarray,
    buildings: list[Polygon],
    heights: np.ndarray,
) -> list[dict[str, float | int]]:
    mean = history.mean(axis=0)
    median = np.median(history, axis=0)
    p95 = np.percentile(history, 95, axis=0)
    std = history.std(axis=0)
    maximum = history.max(axis=0)
    tree = STRtree(buildings)
    rows: list[dict[str, float | int]] = []
    for iy in range(mean.shape[0]):
        for ix in range(mean.shape[1]):
            traffic = float(mean[iy, ix])
            if traffic < 5.0 or traffic > 96.0:
                continue
            xmin = ix * STRIDE_M
            ymin = iy * STRIDE_M
            window = box(xmin, ymin, xmin + WINDOW_M, ymin + WINDOW_M)
            if overlaps_existing(window):
                continue
            count, coverage, median_height = building_stats(window, tree, buildings, heights)
            rows.append(
                {
                    "xmin": int(xmin),
                    "ymin": int(ymin),
                    "xmax": int(xmin + WINDOW_M),
                    "ymax": int(ymin + WINDOW_M),
                    "mean_active": traffic,
                    "median_active": float(median[iy, ix]),
                    "p95_active": float(p95[iy, ix]),
                    "max_active": int(maximum[iy, ix]),
                    "active_cv": float(std[iy, ix] / traffic),
                    "building_count": count,
                    "building_coverage": coverage,
                    "median_building_height_m": median_height,
                }
            )
    return rows


def add_road_lengths(rows: list[dict], roads: list[LineString]) -> None:
    tree = STRtree(roads)
    for row in rows:
        window = box(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        indices = tree.query(window, predicate="intersects")
        row["directed_road_length_km"] = sum(
            roads[int(index)].intersection(window).length for index in indices
        ) / 1000.0


def add_unique_vehicle_counts(rows: list[dict], fcd_path: Path) -> None:
    ids = [set() for _ in rows]
    with gzip.open(fcd_path, "rb") as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            if element.tag == "vehicle":
                x = float(element.get("x"))
                y = float(element.get("y"))
                for index, row in enumerate(rows):
                    if row["xmin"] <= x < row["xmax"] and row["ymin"] <= y < row["ymax"]:
                        ids[index].add(element.get("id", ""))
            element.clear()
    for row, vehicle_ids in zip(rows, ids):
        row["unique_vehicles"] = len(vehicle_ids)


def separated(candidate: dict, chosen: list[dict], minimum_center_distance: float = 500.0) -> bool:
    cx = float(candidate["xmin"]) + WINDOW_M / 2
    cy = float(candidate["ymin"]) + WINDOW_M / 2
    return all(
        math.hypot(cx - (float(row["xmin"]) + WINDOW_M / 2), cy - (float(row["ymin"]) + WINDOW_M / 2))
        >= minimum_center_distance
        for row in chosen
    )


def select_proposals(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []

    # First isolate building density around a common traffic level.
    matched = [r for r in rows if 37.0 <= float(r["mean_active"]) <= 43.0]
    if len(matched) < 3:
        raise RuntimeError("not enough matched-traffic candidates")
    coverage_values = np.asarray([float(r["building_coverage"]) for r in matched])
    unique_target = float(np.median([int(r["unique_vehicles"]) for r in matched]))
    density_targets = [
        ("matched_sparse", float(np.quantile(coverage_values, 0.05))),
        ("matched_medium", float(np.quantile(coverage_values, 0.50))),
        ("matched_dense", float(np.quantile(coverage_values, 0.95))),
    ]
    for name, density_target in density_targets:
        ranked = sorted(
            matched,
            key=lambda r: (
                abs(float(r["building_coverage"]) - density_target) / 0.05
                + abs(float(r["mean_active"]) - 40.0) / 4.0
                + abs(math.log((int(r["unique_vehicles"]) + 1) / (unique_target + 1))) * 3.0
                + abs(float(r["directed_road_length_km"]) - 1.8) / 0.5
                + float(r["active_cv"]) / 2.0
            ),
        )
        choice = next(r for r in ranked if separated(r, selected, 450.0))
        choice = dict(choice)
        choice["proposal_group"] = "matched_traffic_buildings"
        choice["proposal_name"] = name
        selected.append(choice)

    # Then make a sparse-built traffic ladder; matched_sparse is its 40-vehicle anchor.
    for target in (10.0, 25.0, 55.0, 75.0):
        ranked = sorted(
            rows,
            key=lambda r: (
                abs(float(r["mean_active"]) - target) / 4.0
                + float(r["building_coverage"]) / 0.03
                + abs(float(r["directed_road_length_km"]) - 2.1) / 0.5
                + float(r["active_cv"]) / 2.0
            ),
        )
        choice = next(
            r for r in ranked
            if float(r["building_coverage"]) <= 0.05 and separated(r, selected, 450.0)
        )
        choice = dict(choice)
        choice["proposal_group"] = "traffic_ladder"
        choice["proposal_name"] = f"traffic_{int(target):02d}"
        selected.append(choice)
    return selected


def enrich_selected(
    selected: list[dict],
    history: np.ndarray,
    roads: list[LineString],
    fcd_path: Path,
) -> None:
    road_tree = STRtree(roads)
    for row in selected:
        window = box(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        indices = road_tree.query(window, predicate="intersects")
        row["directed_road_length_km"] = sum(
            roads[int(i)].intersection(window).length for i in indices
        ) / 1000.0
        ix = int(float(row["xmin"]) / STRIDE_M)
        iy = int(float(row["ymin"]) / STRIDE_M)
        row["active_q05"] = float(np.percentile(history[:, iy, ix], 5))

    missing_unique = [row for row in selected if "unique_vehicles" not in row]
    if missing_unique:
        add_unique_vehicle_counts(missing_unique, fcd_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_proposals(
    path: Path,
    selected: list[dict],
    buildings: list[Polygon],
    roads: list[LineString],
) -> None:
    building_tree = STRtree(buildings)
    road_tree = STRtree(roads)
    columns = 3
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 4.4 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, proposal in zip(axes_array, selected):
        window = box(proposal["xmin"], proposal["ymin"], proposal["xmax"], proposal["ymax"])
        road_indices = road_tree.query(window, predicate="intersects")
        segments = []
        for index in road_indices:
            clipped = roads[int(index)].intersection(window)
            if clipped.geom_type == "LineString":
                segments.append(np.asarray(clipped.coords) - [proposal["xmin"], proposal["ymin"]])
            elif clipped.geom_type == "MultiLineString":
                segments.extend(
                    np.asarray(part.coords) - [proposal["xmin"], proposal["ymin"]]
                    for part in clipped.geoms
                )
        if segments:
            axis.add_collection(LineCollection(segments, colors="#ffffff", linewidths=2.2, zorder=1))
            axis.add_collection(LineCollection(segments, colors="#475569", linewidths=0.75, zorder=2))

        patches = []
        for index in building_tree.query(window, predicate="intersects"):
            clipped = buildings[int(index)].intersection(window)
            parts = [clipped] if clipped.geom_type == "Polygon" else getattr(clipped, "geoms", [])
            for part in parts:
                if part.geom_type == "Polygon" and part.area > 0.25:
                    coords = np.asarray(part.exterior.coords) - [proposal["xmin"], proposal["ymin"]]
                    patches.append(MplPolygon(coords, closed=True))
        if patches:
            axis.add_collection(
                PatchCollection(patches, facecolor="#334155", edgecolor="#0f172a", linewidth=0.25, zorder=3)
            )
        axis.set_facecolor("#dbe8d2")
        axis.set_xlim(0, WINDOW_M)
        axis.set_ylim(0, WINDOW_M)
        axis.set_aspect("equal")
        axis.set_xticks([0, 100, 200, 300])
        axis.set_yticks([0, 100, 200, 300])
        title = str(proposal["proposal_name"]).replace("_", " ").title()
        axis.set_title(
            f"{title}\n"
            f"active {proposal['mean_active']:.1f} · buildings {100*proposal['building_coverage']:.1f}% · "
            f"unique {proposal['unique_vehicles']}",
            fontsize=10,
        )
        axis.set_xlabel(f"SUMO origin ({proposal['xmin']}, {proposal['ymin']}) m", fontsize=8)
    for axis in axes_array[len(selected):]:
        axis.axis("off")
    figure.suptitle(
        "Proposed independent 300×300 m Luxembourg evaluation zones",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading building footprints...", flush=True)
    buildings, heights = load_buildings(args.polygons)
    print(f"Loaded {len(buildings)} buildings", flush=True)
    print("Loading road geometry...", flush=True)
    roads = load_roads(args.network)
    print(f"Loaded {len(roads)} directed road shapes", flush=True)
    print("Scanning 30-minute traffic trace...", flush=True)
    history, _nx = scan_traffic(args.fcd)
    print("Computing candidate statistics...", flush=True)
    candidates = build_candidates(history, buildings, heights)
    add_road_lengths(candidates, roads)
    write_csv(args.output_dir / "all_candidates.csv", candidates)
    matched_candidates = [row for row in candidates if 37.0 <= float(row["mean_active"]) <= 43.0]
    add_unique_vehicle_counts(matched_candidates, args.fcd)
    selected = select_proposals(candidates)
    print("Computing exact statistics for selected zones...", flush=True)
    enrich_selected(selected, history, roads, args.fcd)
    write_csv(args.output_dir / "proposed_zones.csv", selected)
    with (args.output_dir / "proposed_zones.json").open("w", encoding="utf-8") as stream:
        json.dump({"existing_zones": EXISTING, "proposals": selected}, stream, indent=2)
        stream.write("\n")
    plot_proposals(args.output_dir / "proposed_zones.png", selected, buildings, roads)
    print(json.dumps(selected, indent=2), flush=True)
    print(f"Wrote proposal to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
