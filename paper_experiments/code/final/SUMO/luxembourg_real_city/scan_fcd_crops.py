#!/usr/bin/env python3
"""Rank square LuST crops from a coarse FCD trace before ray tracing."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree


class Frame(NamedTuple):
    time: float
    ids: list[str]
    types: list[str]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    angle: np.ndarray
    slope: np.ndarray
    speed: np.ndarray


def _open_xml(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def iter_fcd(path: Path) -> Iterator[Frame]:
    with _open_xml(path) as source:
        for _event, elem in ET.iterparse(source, events=("end",)):
            if elem.tag.rsplit("}", 1)[-1] != "timestep":
                continue
            vehicles = list(elem)
            yield Frame(
                time=float(elem.get("time", "0")),
                ids=[v.get("id", "") for v in vehicles],
                types=[v.get("type", "") for v in vehicles],
                x=np.fromiter((float(v.get("x", "nan")) for v in vehicles), dtype=np.float64),
                y=np.fromiter((float(v.get("y", "nan")) for v in vehicles), dtype=np.float64),
                z=np.fromiter((float(v.get("z", "0")) for v in vehicles), dtype=np.float64),
                angle=np.fromiter(
                    (float(v.get("angle", "nan")) for v in vehicles),
                    dtype=np.float64,
                ),
                slope=np.fromiter(
                    (float(v.get("slope", "0")) for v in vehicles),
                    dtype=np.float64,
                ),
                speed=np.fromiter(
                    (float(v.get("speed", "nan")) for v in vehicles),
                    dtype=np.float64,
                ),
            )
            elem.clear()


def window_sums(hist: np.ndarray, cells: int) -> np.ndarray:
    integral = np.pad(hist, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (
        integral[cells:, cells:]
        - integral[:-cells, cells:]
        - integral[cells:, :-cells]
        + integral[:-cells, :-cells]
    )


def contact_degrees(distance2: np.ndarray, radius2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-vehicle neighbor counts and the upper-triangle contact indices."""
    ii, jj = np.where(np.triu(distance2 <= radius2, k=1))
    degrees = np.zeros(distance2.shape[0], dtype=np.int32)
    # Repeated advanced-index updates do not accumulate with ``degrees[ii] += 1``.
    np.add.at(degrees, ii, 1)
    np.add.at(degrees, jj, 1)
    return degrees, ii, jj


def polygon_index(path: Path) -> tuple[list[Polygon], np.ndarray, STRtree]:
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    polygons: list[Polygon] = []
    heights: list[float] = []
    for elem in ET.parse(path).getroot().findall("poly"):
        if elem.get("type") != "building":
            continue
        points = []
        for raw in elem.get("shape", "").split():
            x, y = raw.split(",")[:2]
            points.append((float(x), float(y)))
        if len(points) < 3:
            continue
        geom = Polygon(points)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        polygons.append(geom)
        heights.append(float(elem.get("layer", "0")))
    return polygons, np.asarray(heights, dtype=np.float64), STRtree(polygons)


def load_route_sources(raw_specs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map persistent physical vehicle IDs to their LuST demand source."""
    sources: dict[str, str] = {}
    paths: dict[str, str] = {}
    for raw in raw_specs:
        try:
            name, raw_path = raw.split(":", 1)
        except ValueError as exc:
            raise ValueError("route-file must be SOURCE:PATH") from exc
        if not name or not raw_path:
            raise ValueError("route-file must be SOURCE:PATH")
        path = Path(raw_path)
        paths[name] = str(path.resolve())
        with _open_xml(path) as stream:
            for _event, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag.rsplit("}", 1)[-1] != "vehicle":
                    continue
                vehicle_id = elem.get("id", "")
                previous = sources.setdefault(vehicle_id, name)
                if previous != name:
                    raise ValueError(
                        f"vehicle ID {vehicle_id!r} occurs in both {previous!r} and {name!r}"
                    )
                elem.clear()
    return sources, paths


def coarse_candidates(
    *,
    fcd: Path,
    boundary: tuple[float, float, float, float],
    polygons: list[Polygon],
    heights: np.ndarray,
    crop_size: float,
    grid_step: float,
    target_active: float,
    preliminary_count: int,
) -> tuple[list[dict[str, float]], int, float]:
    x0, y0, x1, y1 = boundary
    nx = int(math.ceil((x1 - x0) / grid_step))
    ny = int(math.ceil((y1 - y0) / grid_step))
    cells = int(round(crop_size / grid_step))
    if not math.isclose(cells * grid_step, crop_size):
        raise ValueError("crop-size must be an integer multiple of grid-step")
    x_edges = x0 + np.arange(nx + 1, dtype=np.float64) * grid_step
    y_edges = y0 + np.arange(ny + 1, dtype=np.float64) * grid_step

    count_frames: list[np.ndarray] = []
    frame_times: list[float] = []
    for frame in iter_fcd(fcd):
        valid = np.isfinite(frame.x) & np.isfinite(frame.y)
        hist = np.histogram2d(frame.x[valid], frame.y[valid], bins=(x_edges, y_edges))[0]
        count_frames.append(window_sums(hist, cells).astype(np.int16))
        frame_times.append(frame.time)
    if not count_frames:
        raise ValueError(f"FCD trace contains no timesteps: {fcd}")
    counts = np.stack(count_frames, axis=0)

    building_hist = np.zeros((nx, ny), dtype=np.float64)
    known_height_hist = np.zeros((nx, ny), dtype=np.float64)
    for geom, height in zip(polygons, heights):
        cx, cy = geom.centroid.coords[0]
        ix = int((cx - x0) // grid_step)
        iy = int((cy - y0) // grid_step)
        if 0 <= ix < nx and 0 <= iy < ny:
            building_hist[ix, iy] += 1.0
            known_height_hist[ix, iy] += float(height > 0)
    building_counts = window_sums(building_hist, cells)
    known_heights = window_sums(known_height_hist, cells)

    median = np.median(counts, axis=0)
    mean = np.mean(counts, axis=0)
    p95 = np.percentile(counts, 95, axis=0)
    maximum = np.max(counts, axis=0)
    occupancy = np.mean(counts > 0, axis=0)
    activity_score = np.exp(-np.abs(median - target_active) / max(target_active, 1.0))
    urban_score = np.minimum(building_counts / 100.0, 1.0)
    height_score = np.divide(
        known_heights,
        np.maximum(building_counts, 1.0),
        out=np.zeros_like(known_heights),
    )
    score = 0.60 * activity_score + 0.25 * urban_score + 0.10 * occupancy + 0.05 * height_score
    eligible = (median >= 5) & (building_counts >= 20)
    score = np.where(eligible, score, -np.inf)

    ranked = np.argsort(score.ravel())[::-1]
    selected: list[dict[str, float]] = []
    min_separation = 0.5 * crop_size
    shape = score.shape
    for flat in ranked:
        if not np.isfinite(score.ravel()[flat]):
            break
        ix, iy = np.unravel_index(flat, shape)
        cx = x0 + (ix + 0.5 * cells) * grid_step
        cy = y0 + (iy + 0.5 * cells) * grid_step
        if any(math.hypot(cx - c["center_x"], cy - c["center_y"]) < min_separation for c in selected):
            continue
        selected.append(
            {
                "name": f"auto_{len(selected) + 1:02d}",
                "center_x": float(cx),
                "center_y": float(cy),
                "coarse_score": float(score[ix, iy]),
                "active_mean": float(mean[ix, iy]),
                "active_median": float(median[ix, iy]),
                "active_p95": float(p95[ix, iy]),
                "active_max": float(maximum[ix, iy]),
                "building_centroids": int(building_counts[ix, iy]),
                "known_height_centroids": int(known_heights[ix, iy]),
            }
        )
        if len(selected) >= preliminary_count:
            break
    sample_period = float(statistics.median(np.diff(frame_times))) if len(frame_times) > 1 else 0.0
    return selected, len(frame_times), sample_period


def exact_geometry_metrics(
    candidate: dict[str, float],
    *,
    crop_size: float,
    polygons: list[Polygon],
    heights: np.ndarray,
    tree: STRtree,
) -> None:
    from shapely.geometry import box

    half = 0.5 * crop_size
    crop = box(
        candidate["center_x"] - half,
        candidate["center_y"] - half,
        candidate["center_x"] + half,
        candidate["center_y"] + half,
    )
    indices = np.asarray(tree.query(crop), dtype=np.int64)
    indices = np.asarray([i for i in indices if polygons[int(i)].intersects(crop)], dtype=np.int64)
    selected_heights = heights[indices] if indices.size else np.zeros((0,), dtype=np.float64)
    positive = selected_heights[selected_heights > 0]
    candidate.update(
        {
            "building_count": int(indices.size),
            "building_height_coverage": float(np.mean(selected_heights > 0)) if indices.size else 0.0,
            "building_height_median_m": float(np.median(positive)) if positive.size else None,
            "building_height_p95_m": float(np.percentile(positive, 95)) if positive.size else None,
        }
    )


def exact_mobility_metrics(
    candidates: list[dict[str, float]],
    *,
    fcd: Path,
    crop_size: float,
    contact_radius: float,
    sample_period: float,
    route_sources: Mapping[str, str],
) -> None:
    state = []
    for _candidate in candidates:
        state.append(
            {
                "counts": [],
                "bus_frames": 0,
                "vehicle_frames": 0,
                "unique_ids": set(),
                "previous_ids": set(),
                "entries": Counter(),
                "oversubscribed": Counter(),
                "receiver_frames": 0,
                "active_contacts": {},
                "contact_durations": [],
                "source_frames": Counter(),
                "source_unique_ids": defaultdict(set),
            }
        )

    radius2 = contact_radius * contact_radius
    half = 0.5 * crop_size
    for frame in iter_fcd(fcd):
        for candidate, metrics in zip(candidates, state):
            inside = (
                (frame.x >= candidate["center_x"] - half)
                & (frame.x <= candidate["center_x"] + half)
                & (frame.y >= candidate["center_y"] - half)
                & (frame.y <= candidate["center_y"] + half)
            )
            idx = np.flatnonzero(inside)
            ids = [frame.ids[int(i)] for i in idx]
            types = [frame.types[int(i)] for i in idx]
            current_ids = set(ids)
            for vehicle_id in current_ids - metrics["previous_ids"]:
                metrics["entries"][vehicle_id] += 1
            metrics["previous_ids"] = current_ids
            metrics["unique_ids"].update(current_ids)
            metrics["counts"].append(len(idx))
            metrics["vehicle_frames"] += len(idx)
            metrics["bus_frames"] += sum(vehicle_type == "bus" for vehicle_type in types)
            for vehicle_id in ids:
                source = route_sources.get(vehicle_id, "unclassified")
                metrics["source_frames"][source] += 1
                metrics["source_unique_ids"][source].add(vehicle_id)

            contacts: set[tuple[str, str]] = set()
            degrees = np.zeros(len(idx), dtype=np.int32)
            if len(idx) >= 2:
                px = frame.x[idx]
                py = frame.y[idx]
                distance2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
                degrees, ii, jj = contact_degrees(distance2, radius2)
                contacts = {tuple(sorted((ids[int(i)], ids[int(j)]))) for i, j in zip(ii, jj)}
            metrics["receiver_frames"] += len(idx)
            for budget in (1, 2, 4):
                metrics["oversubscribed"][budget] += int(np.sum(degrees > budget))

            active_contacts = metrics["active_contacts"]
            ended = set(active_contacts).difference(contacts)
            for pair in ended:
                metrics["contact_durations"].append(active_contacts.pop(pair) * sample_period)
            for pair in contacts:
                active_contacts[pair] = int(active_contacts.get(pair, 0)) + 1

    for candidate, metrics in zip(candidates, state):
        metrics["contact_durations"].extend(
            steps * sample_period for steps in metrics["active_contacts"].values()
        )
        counts = np.asarray(metrics["counts"], dtype=np.float64)
        durations = np.asarray(metrics["contact_durations"], dtype=np.float64)
        receiver_frames = max(1, int(metrics["receiver_frames"]))
        entry_counts = metrics["entries"]
        candidate.update(
            {
                "active_mean": float(np.mean(counts)),
                "active_median": float(np.median(counts)),
                "active_p95": float(np.percentile(counts, 95)),
                "active_max": int(np.max(counts)),
                "unique_vehicle_count": len(metrics["unique_ids"]),
                "crop_entry_count": int(sum(entry_counts.values())),
                "reentering_vehicle_count": int(sum(value > 1 for value in entry_counts.values())),
                "bus_vehicle_frame_fraction": float(metrics["bus_frames"] / max(1, metrics["vehicle_frames"])),
                "route_source_unique_vehicle_counts": {
                    source: len(vehicle_ids)
                    for source, vehicle_ids in sorted(metrics["source_unique_ids"].items())
                },
                "route_source_vehicle_frame_fractions": {
                    source: float(count / max(1, metrics["vehicle_frames"]))
                    for source, count in sorted(metrics["source_frames"].items())
                },
                "distance_contact_episode_count": int(durations.size),
                "distance_contact_duration_median_s": float(np.median(durations)) if durations.size else 0.0,
                "distance_contact_duration_p95_s": float(np.percentile(durations, 95)) if durations.size else 0.0,
                "receiver_frame_fraction_more_than_1_provider": float(metrics["oversubscribed"][1] / receiver_frames),
                "receiver_frame_fraction_more_than_2_providers": float(metrics["oversubscribed"][2] / receiver_frames),
                "receiver_frame_fraction_more_than_4_providers": float(metrics["oversubscribed"][4] / receiver_frames),
            }
        )


def parse_named_candidate(raw: str, net) -> dict[str, float]:
    try:
        name, lon, lat = raw.split(":")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("named candidate must be NAME:LON:LAT") from exc
    x, y = net.convertLonLat2XY(float(lon), float(lat))
    return {"name": name, "center_x": float(x), "center_y": float(y), "coarse_score": float("nan")}


def main() -> int:
    import sumolib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcd", type=Path, required=True)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop-size", type=float, default=800.0)
    parser.add_argument("--grid-step", type=float, default=100.0)
    parser.add_argument("--target-active", type=float, default=40.0)
    parser.add_argument("--contact-radius", type=float, default=150.0)
    parser.add_argument("--preliminary-count", type=int, default=12)
    parser.add_argument("--named-candidate", action="append", default=[])
    parser.add_argument(
        "--route-file",
        action="append",
        default=[],
        help="Demand source and route file as SOURCE:PATH; repeat for every source",
    )
    args = parser.parse_args()

    net = sumolib.net.readNet(str(args.net))
    boundary = tuple(float(value) for value in net.getBoundary())
    polygons, heights, tree = polygon_index(args.polygons)
    route_sources, route_files = load_route_sources(args.route_file)
    candidates, frames, sample_period = coarse_candidates(
        fcd=args.fcd,
        boundary=boundary,
        polygons=polygons,
        heights=heights,
        crop_size=float(args.crop_size),
        grid_step=float(args.grid_step),
        target_active=float(args.target_active),
        preliminary_count=int(args.preliminary_count),
    )
    candidates.extend(parse_named_candidate(raw, net) for raw in args.named_candidate)

    for candidate in candidates:
        lon, lat = net.convertXY2LonLat(candidate["center_x"], candidate["center_y"])
        candidate["center_lon"] = float(lon)
        candidate["center_lat"] = float(lat)
        exact_geometry_metrics(
            candidate,
            crop_size=float(args.crop_size),
            polygons=polygons,
            heights=heights,
            tree=tree,
        )
    exact_mobility_metrics(
        candidates,
        fcd=args.fcd,
        crop_size=float(args.crop_size),
        contact_radius=float(args.contact_radius),
        sample_period=sample_period,
        route_sources=route_sources,
    )

    report = {
        "format": "lust_crop_scan_v1",
        "fcd": str(args.fcd.resolve()),
        "net": str(args.net.resolve()),
        "polygons": str(args.polygons.resolve()),
        "frames": frames,
        "sample_period_s": sample_period,
        "crop_size_m": float(args.crop_size),
        "contact_radius_m": float(args.contact_radius),
        "route_files": route_files,
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for candidate in candidates:
        print(
            f"{candidate['name']:16s} center=({candidate['center_x']:.1f},{candidate['center_y']:.1f}) "
            f"active={candidate['active_median']:.1f}/{candidate['active_p95']:.1f} "
            f"buildings={candidate['building_count']} unique={candidate['unique_vehicle_count']} "
            f">2providers={candidate['receiver_frame_fraction_more_than_2_providers']:.3f}"
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
