#!/usr/bin/env python3
"""Isolated visual probe for finite-segment radio-support capsules.

This script deliberately does not import or modify the production predictor.
It reads the existing Luxembourg RSSI trace, builds capsule summaries from a
small number of feasible links, and writes diagnostic PNGs.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE = Path(
    "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
    "gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers/"
    "rssi/gare_bonnevoie_vehicles_0745_0815_1s_right_middle_opaque_"
    "no_vehicle_blockers_r20k_d3_llvm.npz"
)
DEFAULT_NET = (
    ROOT
    / "SUMO/luxembourg_real_city/gare_bonnevoie/map/sumo/lust3d.net.xml"
)
DEFAULT_POLY = (
    ROOT
    / "SUMO/luxembourg_real_city/gare_bonnevoie/map/sumo/lust3d.poly.xml"
)
DEFAULT_OUTPUT = ROOT / "artifacts/capsule_probe"
SUMO_OFFSET = np.asarray([7200.0, 5100.0], dtype=np.float64)
REGION = (400.0, 800.0, 200.0, 600.0)


@dataclass(frozen=True)
class CapsuleParams:
    angle_deg: float
    lateral_merge_m: float
    longitudinal_gap_m: float
    sigma_perp_m: float = 6.0
    sigma_parallel_m: float = 10.0
    mass_scale: float = 3.0


@dataclass
class Capsule:
    """A direction-free finite segment with accumulated observation mass."""

    start: np.ndarray
    end: np.ndarray
    mass: float = 1.0

    @classmethod
    def from_segment(cls, segment: np.ndarray, mass: float = 1.0) -> "Capsule":
        points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
        if float(np.linalg.norm(points[1] - points[0])) < 1.0e-6:
            raise ValueError("capsule segment must have nonzero length")
        return cls(points[0].copy(), points[1].copy(), float(mass))

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        value = (self.end - self.start) / self.length
        if value[0] < 0.0 or (abs(value[0]) < 1.0e-12 and value[1] < 0.0):
            value = -value
        return value

    @property
    def segment(self) -> np.ndarray:
        return np.stack((self.start, self.end))


@dataclass(frozen=True)
class Compatibility:
    angle_deg: float
    lateral_m: float
    gap_m: float
    score: float


def _common_axis(first: Capsule, second: Capsule) -> np.ndarray:
    left = first.direction
    right = second.direction
    if float(np.dot(left, right)) < 0.0:
        right = -right
    value = first.mass * left + second.mass * right
    norm = float(np.linalg.norm(value))
    return left if norm < 1.0e-12 else value / norm


def capsule_compatibility(
    first: Capsule, second: Capsule, params: CapsuleParams
) -> Compatibility | None:
    cosine = float(np.clip(abs(np.dot(first.direction, second.direction)), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    axis = _common_axis(first, second)
    normal = np.asarray([-axis[1], axis[0]])
    lateral = abs(float(np.dot(second.midpoint - first.midpoint, normal)))

    first_t = first.segment @ axis
    second_t = second.segment @ axis
    first_interval = (float(first_t.min()), float(first_t.max()))
    second_interval = (float(second_t.min()), float(second_t.max()))
    gap = max(
        0.0,
        max(first_interval[0], second_interval[0])
        - min(first_interval[1], second_interval[1]),
    )
    if (
        angle > params.angle_deg
        or lateral > params.lateral_merge_m
        or gap > params.longitudinal_gap_m
    ):
        return None
    score = (
        angle / max(params.angle_deg, 1.0e-9)
        + lateral / max(params.lateral_merge_m, 1.0e-9)
        + gap / max(params.longitudinal_gap_m, 1.0e-9)
    )
    return Compatibility(angle, lateral, gap, float(score))


def merge_capsules(first: Capsule, second: Capsule) -> Capsule:
    """Conservatively span two already-compatible capsules."""

    axis = _common_axis(first, second)
    normal = np.asarray([-axis[1], axis[0]])
    endpoints = np.concatenate((first.segment, second.segment), axis=0)
    along = endpoints @ axis
    low, high = float(along.min()), float(along.max())
    lateral = float(
        (
            first.mass * np.dot(first.midpoint, normal)
            + second.mass * np.dot(second.midpoint, normal)
        )
        / (first.mass + second.mass)
    )
    start = low * axis + lateral * normal
    end = high * axis + lateral * normal
    return Capsule(start, end, first.mass + second.mass)


def add_capsule(
    capsules: list[Capsule], incoming: Capsule, params: CapsuleParams
) -> None:
    """Insert one capsule and absorb every newly compatible neighbour."""

    current = incoming
    while True:
        candidates: list[tuple[float, int]] = []
        for index, existing in enumerate(capsules):
            result = capsule_compatibility(existing, current, params)
            if result is not None:
                candidates.append((result.score, index))
        if not candidates:
            capsules.append(current)
            return
        _score, index = min(candidates)
        current = merge_capsules(capsules.pop(index), current)


def build_capsules(segments: np.ndarray, params: CapsuleParams) -> list[Capsule]:
    capsules: list[Capsule] = []
    for segment in np.asarray(segments, dtype=np.float64):
        if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0:
            add_capsule(capsules, Capsule.from_segment(segment), params)
    return capsules


def merge_capsule_sets(
    first: list[Capsule], second: list[Capsule], params: CapsuleParams
) -> list[Capsule]:
    """Model-pull support union using only capsule summaries."""

    combined = [
        Capsule(row.start.copy(), row.end.copy(), row.mass)
        for row in sorted(
            [*first, *second],
            key=lambda row: (
                float(row.midpoint[0]),
                float(row.midpoint[1]),
                float(row.length),
            ),
        )
    ]
    result: list[Capsule] = []
    for capsule in combined:
        add_capsule(result, capsule, params)
    return result


def point_confidence(
    points: np.ndarray, capsules: list[Capsule], params: CapsuleParams
) -> np.ndarray:
    """Maximum finite-segment RBF confidence at each 2-D point."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    confidence = np.zeros(len(values), dtype=np.float64)
    for capsule in capsules:
        axis = capsule.direction
        relative = values - capsule.midpoint
        along = relative @ axis
        perpendicular = relative - along[:, None] * axis[None, :]
        d_perp = np.linalg.norm(perpendicular, axis=1)
        d_parallel = np.maximum(np.abs(along) - 0.5 * capsule.length, 0.0)
        spatial = np.exp(
            -0.5 * np.square(d_perp / params.sigma_perp_m)
            -0.5 * np.square(d_parallel / params.sigma_parallel_m)
        )
        maturity = 1.0 - math.exp(-capsule.mass / params.mass_scale)
        confidence = np.maximum(confidence, maturity * spatial)
    return confidence


def link_confidence(
    segments: np.ndarray, capsules: list[Capsule], params: CapsuleParams
) -> np.ndarray:
    """Confidence that the complete link lies inside the union of capsules."""

    rows = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    fractions = np.linspace(0.0, 1.0, 9)
    points = (
        rows[:, None, 0, :] * (1.0 - fractions[None, :, None])
        + rows[:, None, 1, :] * fractions[None, :, None]
    )
    values = point_confidence(points.reshape(-1, 2), capsules, params)
    return values.reshape(len(rows), len(fractions)).min(axis=1)


def shifted_segments(segments: np.ndarray, distance_m: float = 25.0) -> np.ndarray:
    rows = np.asarray(segments, dtype=np.float64).copy()
    direction = rows[:, 1] - rows[:, 0]
    norm = np.linalg.norm(direction, axis=1).clip(min=1.0e-9)
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1) / norm[:, None]
    signs = np.where(np.arange(len(rows)) % 2, -1.0, 1.0)
    return rows + signs[:, None, None] * distance_m * normal[:, None, :]


def select_receiver_links(
    measurements: np.ndarray,
    states: np.ndarray,
    *,
    receiver: int,
    rssi_threshold: float,
    transmitter_count: int,
    stride: int,
    max_length_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    rows = measurements[
        (measurements[:, 3].astype(np.int64) == int(receiver))
        & (measurements[:, 4] >= float(rssi_threshold))
    ]
    if not len(rows):
        raise ValueError(f"receiver {receiver} has no feasible links")
    tx_ids, counts = np.unique(rows[:, 2].astype(np.int64), return_counts=True)
    order = np.lexsort((tx_ids, -counts))
    selected_tx = tuple(int(value) for value in tx_ids[order[:transmitter_count]])

    selected: list[np.ndarray] = []
    metadata: list[np.ndarray] = []
    for tx in selected_tx:
        track = rows[rows[:, 2].astype(np.int64) == tx]
        track = track[np.argsort(track[:, 0])][:: max(1, int(stride))]
        for row in track:
            step, _zone, tx_idx, rx_idx, rssi = row
            endpoints = states[
                int(step), [int(tx_idx), int(rx_idx)], :2
            ].astype(np.float64)
            length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            if 5.0 <= length <= float(max_length_m):
                selected.append(endpoints)
                metadata.append(
                    np.asarray([step, tx_idx, rx_idx, rssi, length])
                )
    if len(selected) < 10:
        raise ValueError(f"receiver {receiver} produced too few selected links")
    return np.stack(selected), np.stack(metadata), selected_tx


def train_holdout_split(
    segments: np.ndarray, metadata: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic interleaving preserves every transmitter trajectory in
    # both sets without using RSSI values in the capsule construction.
    order = np.lexsort((metadata[:, 0], metadata[:, 1]))
    mask = np.zeros(len(segments), dtype=bool)
    mask[order[::5]] = True
    return segments[~mask], segments[mask]


def map_geometry(
    net_path: Path, poly_path: Path
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xmin, xmax, ymin, ymax = REGION
    roads: list[np.ndarray] = []
    for _event, element in ET.iterparse(net_path, events=("end",)):
        if element.tag == "lane":
            raw = element.get("shape")
            if raw:
                points = np.asarray(
                    [
                        [float(value) for value in token.split(",")[:2]]
                        for token in raw.split()
                    ],
                    dtype=np.float64,
                )
                points -= SUMO_OFFSET
                if (
                    len(points) >= 2
                    and points[:, 0].max() >= xmin
                    and points[:, 0].min() <= xmax
                    and points[:, 1].max() >= ymin
                    and points[:, 1].min() <= ymax
                ):
                    roads.append(points)
        element.clear()

    buildings: list[np.ndarray] = []
    for _event, element in ET.iterparse(poly_path, events=("end",)):
        if element.tag == "poly":
            raw = element.get("shape")
            if raw:
                points = np.asarray(
                    [
                        [float(value) for value in token.split(",")[:2]]
                        for token in raw.split()
                    ],
                    dtype=np.float64,
                )
                points -= SUMO_OFFSET
                if (
                    len(points) >= 3
                    and points[:, 0].max() >= xmin
                    and points[:, 0].min() <= xmax
                    and points[:, 1].max() >= ymin
                    and points[:, 1].min() <= ymax
                ):
                    buildings.append(points)
        element.clear()
    return roads, buildings


def draw_background(
    axis: plt.Axes, roads: list[np.ndarray], buildings: list[np.ndarray]
) -> None:
    axis.add_collection(
        PolyCollection(
            buildings,
            facecolors="#d8d5cf",
            edgecolors="#c5c1ba",
            linewidths=0.25,
            zorder=0,
        )
    )
    axis.add_collection(
        LineCollection(
            roads,
            colors="#a6a6a6",
            linewidths=0.45,
            alpha=0.8,
            zorder=1,
        )
    )
    xmin, xmax, ymin, ymax = REGION
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_facecolor("#f7f6f3")
    axis.set_xlabel("local x [m]")
    axis.set_ylabel("local y [m]")


def draw_support(
    axis: plt.Axes,
    roads: list[np.ndarray],
    buildings: list[np.ndarray],
    *,
    segments: np.ndarray,
    capsules: list[Capsule],
    params: CapsuleParams,
    title: str,
    heatmap: bool = True,
) -> None:
    draw_background(axis, roads, buildings)
    if heatmap and capsules:
        xmin, xmax, ymin, ymax = REGION
        xs = np.linspace(xmin, xmax, 220)
        ys = np.linspace(ymin, ymax, 220)
        xx, yy = np.meshgrid(xs, ys)
        confidence = point_confidence(
            np.stack((xx.ravel(), yy.ravel()), axis=1), capsules, params
        ).reshape(xx.shape)
        axis.contourf(
            xx,
            yy,
            confidence,
            levels=[0.15, 0.35, 0.55, 0.75, 1.0],
            colors=["#dff3ff", "#a9ddf5", "#65bce7", "#2389c9"],
            alpha=0.48,
            zorder=2,
        )
    if len(segments):
        axis.add_collection(
            LineCollection(
                segments,
                colors="#4f4f4f",
                linewidths=0.45,
                alpha=0.16,
                zorder=3,
            )
        )
    if capsules:
        colors = plt.get_cmap("turbo")(
            np.linspace(0.03, 0.97, max(2, len(capsules)))
        )
        axis.add_collection(
            LineCollection(
                [capsule.segment for capsule in capsules],
                colors=colors,
                linewidths=[
                    1.1 + 0.45 * math.log1p(capsule.mass)
                    for capsule in capsules
                ],
                alpha=0.92,
                zorder=4,
            )
        )
        endpoints = np.concatenate([capsule.segment for capsule in capsules])
        axis.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=4.0,
            color="#151515",
            alpha=0.5,
            linewidths=0.0,
            zorder=5,
        )
    axis.set_title(title, fontsize=10)


def draw_holdout(
    axis: plt.Axes,
    roads: list[np.ndarray],
    buildings: list[np.ndarray],
    *,
    holdout: np.ndarray,
    capsules: list[Capsule],
    params: CapsuleParams,
) -> None:
    draw_background(axis, roads, buildings)
    confidence = link_confidence(holdout, capsules, params)
    colors = plt.get_cmap("RdYlGn")(confidence)
    axis.add_collection(
        LineCollection(
            holdout,
            colors=colors,
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
        )
    )
    axis.add_collection(
        LineCollection(
            [capsule.segment for capsule in capsules],
            colors="#202020",
            linewidths=1.0,
            alpha=0.4,
            zorder=4,
        )
    )
    axis.set_title(
        "Held-out feasible links: red = unsupported, green = supported\n"
        f"mean confidence={confidence.mean():.2f}, "
        f"fraction ≥0.5={(confidence >= 0.5).mean():.2f}",
        fontsize=10,
    )


def candidate_metrics(
    train_a: np.ndarray,
    holdout_a: np.ndarray,
    train_b: np.ndarray,
    holdout_b: np.ndarray,
    params: CapsuleParams,
) -> tuple[float, dict[str, float], list[Capsule], list[Capsule], list[Capsule]]:
    capsules_a = build_capsules(train_a, params)
    capsules_b = build_capsules(train_b, params)
    merged = merge_capsule_sets(capsules_a, capsules_b, params)
    holdout = np.concatenate((holdout_a, holdout_b), axis=0)
    held_conf = link_confidence(holdout, merged, params)
    off_conf = link_confidence(shifted_segments(holdout), merged, params)
    input_count = len(train_a) + len(train_b)
    metrics = {
        "capsules_a": float(len(capsules_a)),
        "capsules_b": float(len(capsules_b)),
        "capsules_merged": float(len(merged)),
        "compression": float(input_count / max(1, len(merged))),
        "holdout_mean": float(held_conf.mean()),
        "holdout_half": float((held_conf >= 0.5).mean()),
        "off_corridor_mean": float(off_conf.mean()),
        "off_corridor_half": float((off_conf >= 0.5).mean()),
    }
    score = (
        metrics["holdout_half"]
        + 0.35 * metrics["holdout_mean"]
        - 1.5 * metrics["off_corridor_mean"]
        - 0.001 * metrics["capsules_merged"]
    )
    return score, metrics, capsules_a, capsules_b, merged


def self_test() -> None:
    params = CapsuleParams(8.0, 5.0, 10.0)
    first = Capsule.from_segment(np.asarray([[0.0, 0.0], [20.0, 0.0]]))
    nearby = Capsule.from_segment(np.asarray([[18.0, 1.0], [35.0, 1.0]]))
    crossing = Capsule.from_segment(np.asarray([[10.0, -10.0], [10.0, 10.0]]))
    assert capsule_compatibility(first, nearby, params) is not None
    assert capsule_compatibility(first, crossing, params) is None
    merged = merge_capsule_sets([first], [nearby], params)
    assert len(merged) == 1 and merged[0].length >= 34.0
    on_line = point_confidence(np.asarray([[10.0, 0.0]]), merged, params)[0]
    off_line = point_confidence(np.asarray([[10.0, 30.0]]), merged, params)[0]
    assert on_line > off_line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--poly", type=Path, default=DEFAULT_POLY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receiver-a", type=int, default=2652)
    parser.add_argument("--receiver-b", type=int, default=1312)
    parser.add_argument("--rssi-threshold", type=float, default=-100.0)
    parser.add_argument("--transmitters", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-link-length", type=float, default=160.0)
    args = parser.parse_args()

    self_test()
    with np.load(args.trace, allow_pickle=False) as archive:
        trace_metadata = json.loads(str(archive["meta_json"]))
        if not bool(trace_metadata.get("buildings_opaque", False)):
            raise ValueError("capsule probe requires an opaque-building trace")
        if bool(trace_metadata.get("dynamic_vehicle_blockers", True)):
            raise ValueError(
                "capsule probe requires a trace without vehicle blockers"
            )
        measurements = np.asarray(archive["measurements"])
        states = np.asarray(archive["node_states"])
    print(
        "trace verified: buildings_opaque=True "
        "dynamic_vehicle_blockers=False"
    )
    links_a, metadata_a, tx_a = select_receiver_links(
        measurements,
        states,
        receiver=args.receiver_a,
        rssi_threshold=args.rssi_threshold,
        transmitter_count=args.transmitters,
        stride=args.stride,
        max_length_m=args.max_link_length,
    )
    links_b, metadata_b, tx_b = select_receiver_links(
        measurements,
        states,
        receiver=args.receiver_b,
        rssi_threshold=args.rssi_threshold,
        transmitter_count=args.transmitters,
        stride=args.stride,
        max_length_m=args.max_link_length,
    )
    train_a, holdout_a = train_holdout_split(links_a, metadata_a)
    train_b, holdout_b = train_holdout_split(links_b, metadata_b)

    candidates: list[
        tuple[
            float,
            CapsuleParams,
            dict[str, float],
            list[Capsule],
            list[Capsule],
            list[Capsule],
        ]
    ] = []
    for angle in (4.0, 7.0, 10.0):
        for lateral in (3.0, 5.0, 8.0):
            for gap in (4.0, 10.0, 18.0):
                params = CapsuleParams(angle, lateral, gap)
                score, metrics, cap_a, cap_b, merged = candidate_metrics(
                    train_a, holdout_a, train_b, holdout_b, params
                )
                candidates.append(
                    (score, params, metrics, cap_a, cap_b, merged)
                )
    candidates.sort(key=lambda row: row[0], reverse=True)
    score, selected, metrics, capsules_a, capsules_b, merged = candidates[0]

    print(
        f"selected angle={selected.angle_deg:g}deg "
        f"lateral={selected.lateral_merge_m:g}m "
        f"gap={selected.longitudinal_gap_m:g}m score={score:.3f}"
    )
    print(
        f"links A={len(train_a)}+{len(holdout_a)} holdout, "
        f"B={len(train_b)}+{len(holdout_b)} holdout"
    )
    print(f"receivers A={args.receiver_a} tx={tx_a}; B={args.receiver_b} tx={tx_b}")
    print(
        "metrics "
        + " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
    )
    print("top parameter candidates")
    for candidate_score, params, row, *_unused in candidates[:8]:
        print(
            f"  score={candidate_score:.3f} angle={params.angle_deg:g} "
            f"lat={params.lateral_merge_m:g} gap={params.longitudinal_gap_m:g} "
            f"caps={int(row['capsules_merged'])} "
            f"held>=.5={row['holdout_half']:.2f} "
            f"offmean={row['off_corridor_mean']:.2f}"
        )

    roads, buildings = map_geometry(args.net, args.poly)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_train = np.concatenate((train_a, train_b), axis=0)
    all_holdout = np.concatenate((holdout_a, holdout_b), axis=0)

    figure, axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
    draw_support(
        axes[0, 0],
        roads,
        buildings,
        segments=train_a,
        capsules=capsules_a,
        params=selected,
        title=(
            f"Vehicle A (receiver {args.receiver_a}): "
            f"{len(train_a)} links → {len(capsules_a)} capsules"
        ),
    )
    draw_support(
        axes[0, 1],
        roads,
        buildings,
        segments=train_b,
        capsules=capsules_b,
        params=selected,
        title=(
            f"Vehicle B (receiver {args.receiver_b}): "
            f"{len(train_b)} links → {len(capsules_b)} capsules"
        ),
    )
    draw_support(
        axes[1, 0],
        roads,
        buildings,
        segments=all_train,
        capsules=merged,
        params=selected,
        title=(
            f"Pulled support: {len(capsules_a)} + {len(capsules_b)} "
            f"→ {len(merged)} capsules\n"
            f"θ≤{selected.angle_deg:g}°, lateral≤"
            f"{selected.lateral_merge_m:g} m, gap≤"
            f"{selected.longitudinal_gap_m:g} m"
        ),
    )
    draw_holdout(
        axes[1, 1],
        roads,
        buildings,
        holdout=all_holdout,
        capsules=merged,
        params=selected,
    )
    figure.suptitle(
        "Finite-segment capsule support from Luxembourg feasible links\n"
        "Opaque buildings, no vehicle blockers; colored centerlines are "
        "capsules and blue bands are RBF confidence",
        fontsize=15,
    )
    main_path = args.output_dir / "capsule_sets_and_merge.png"
    figure.savefig(main_path, dpi=180)
    plt.close(figure)

    strict = CapsuleParams(4.0, 3.0, 4.0)
    loose = CapsuleParams(14.0, 12.0, 28.0)
    comparisons = [
        ("Strict", strict),
        ("Selected", selected),
        ("Loose", loose),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(19, 7.2), constrained_layout=False)
    for axis, (label, params) in zip(axes, comparisons):
        cap_a = build_capsules(train_a, params)
        cap_b = build_capsules(train_b, params)
        combined = merge_capsule_sets(cap_a, cap_b, params)
        held = link_confidence(all_holdout, combined, params)
        shifted = link_confidence(
            shifted_segments(all_holdout), combined, params
        )
        draw_support(
            axis,
            roads,
            buildings,
            segments=all_train,
            capsules=combined,
            params=params,
            title=(
                f"{label}: {len(combined)} capsules\n"
                f"θ={params.angle_deg:g}°, lateral="
                f"{params.lateral_merge_m:g} m, gap="
                f"{params.longitudinal_gap_m:g} m\n"
                f"held-out≥0.5={(held >= 0.5).mean():.2f}, "
                f"25 m shifted mean={shifted.mean():.2f}"
            ),
        )
    figure.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.78, wspace=0.16)
    figure.suptitle(
        "Capsule merge-parameter sensitivity\nOpaque buildings, no vehicle blockers",
        fontsize=15,
        y=0.96
    )
    comparison_path = args.output_dir / "capsule_parameter_sensitivity.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {main_path}")
    print(f"wrote {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
