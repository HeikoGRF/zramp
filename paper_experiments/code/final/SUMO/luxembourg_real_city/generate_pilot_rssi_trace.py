#!/usr/bin/env python3
"""Ray-trace a persistent-vehicle Luxembourg mobility trace with Sionna RT."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive_paths import activate as activate_shared_runtime  # noqa: E402

activate_shared_runtime()

if TYPE_CHECKING:
    from rl_reward_experiment.measurement import RayTracer


@dataclass(frozen=True)
class VehicleDimensions:
    """Physical blocker dimensions in metres."""

    length: float
    width: float
    height: float
    vclass: str
    gui_shape: str


SUMO_CLASS_DIMENSIONS = {
    "passenger": (5.0, 1.8, 1.5),
    "bus": (12.0, 2.5, 3.4),
}


def load_mobility(path: Path, *, nodes: int, steps: int):
    payload = json.loads(path.read_text())
    if payload.get("format") != "sumo_crop_mobility_trace_v1":
        raise ValueError(f"Unsupported mobility format: {payload.get('format')!r}")
    vehicle_ids = list(payload["vehicle_ids"])
    position_columns = []
    for vehicle_id in vehicle_ids:
        raw_points = payload["traces"][vehicle_id]
        column = np.full((len(raw_points), 3), np.nan, dtype=np.float64)
        for step, point in enumerate(raw_points):
            if point is not None:
                column[step] = np.asarray(point, dtype=np.float64)
        position_columns.append(column)
    positions = np.stack(position_columns, axis=1)
    active = np.stack(
        [np.asarray(payload["active_traces"][vehicle_id], dtype=np.bool_) for vehicle_id in vehicle_ids],
        axis=1,
    )
    headings = _load_optional_vehicle_trace(
        payload,
        key="heading_traces_deg",
        vehicle_ids=vehicle_ids,
        frames=positions.shape[0],
    )
    slopes = _load_optional_vehicle_trace(
        payload,
        key="slope_traces_deg",
        vehicle_ids=vehicle_ids,
        frames=positions.shape[0],
        inactive_default=0.0,
    )
    speeds = _load_optional_vehicle_trace(
        payload,
        key="speed_traces_mps",
        vehicle_ids=vehicle_ids,
        frames=positions.shape[0],
    )
    expected = (steps + 1, nodes)
    if len(vehicle_ids) != nodes:
        raise ValueError(f"Expected {nodes} vehicle IDs, got {len(vehicle_ids)}")
    if positions.shape[1:] != (nodes, 3) or positions.shape[0] < steps + 1:
        raise ValueError(f"Expected at least {expected + (3,)}, got {positions.shape}")
    if active.shape[1:] != (nodes,) or active.shape[0] < steps + 1:
        raise ValueError(f"Expected at least {expected}, got {active.shape}")
    # Sionna still needs a finite location for parked/inactive slots. Retain their
    # latest physical location; node_active prevents those slots from participating.
    for node in range(nodes):
        valid = np.flatnonzero(active[:, node])
        if not len(valid):
            raise ValueError(f"Vehicle slot {node} is never active")
        first = int(valid[0])
        positions[:first, node] = positions[first, node]
        for trace_step in range(first + 1, positions.shape[0]):
            if not active[trace_step, node]:
                positions[trace_step, node] = positions[trace_step - 1, node]
    positions = positions[: steps + 1].copy()
    active = active[: steps + 1].copy()
    headings = headings[: steps + 1].copy() if headings is not None else None
    slopes = slopes[: steps + 1].copy() if slopes is not None else None
    speeds = speeds[: steps + 1].copy() if speeds is not None else None
    return payload, vehicle_ids, positions, active, headings, slopes, speeds


def _load_optional_vehicle_trace(
    payload: dict,
    *,
    key: str,
    vehicle_ids: list[str],
    frames: int,
    inactive_default: float = float("nan"),
) -> np.ndarray | None:
    raw = payload.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must map vehicle IDs to frame values")
    columns: list[np.ndarray] = []
    for vehicle_id in vehicle_ids:
        values = raw.get(vehicle_id)
        if not isinstance(values, list) or len(values) < int(frames):
            raise ValueError(f"{key} is incomplete for vehicle {vehicle_id}")
        columns.append(
            np.asarray(
                [inactive_default if value is None else float(value) for value in values[:frames]],
                dtype=np.float64,
            )
        )
    return np.stack(columns, axis=1)


def load_vehicle_dimensions(
    vehicle_types: dict[str, str],
    vtype_file: Path,
) -> dict[str, VehicleDimensions]:
    """Resolve LuST lengths and SUMO class-default widths/heights."""

    root = ET.parse(vtype_file).getroot()
    by_type: dict[str, VehicleDimensions] = {}
    for element in root.iter("vType"):
        type_id = str(element.get("id", ""))
        if not type_id:
            continue
        vclass = str(element.get("vClass", "passenger"))
        default = SUMO_CLASS_DIMENSIONS.get(vclass, SUMO_CLASS_DIMENSIONS["passenger"])
        by_type[type_id] = VehicleDimensions(
            length=float(element.get("length", default[0])),
            width=float(element.get("width", default[1])),
            height=float(element.get("height", default[2])),
            vclass=vclass,
            gui_shape=str(element.get("guiShape", "")),
        )
    missing = sorted({str(value) for value in vehicle_types.values()} - set(by_type))
    if missing:
        raise ValueError(f"vehicle types missing from {vtype_file}: {missing}")
    return by_type


def save_trace(path: Path, *, meta: dict, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, meta_json=np.asarray(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def _box_template(length: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-length, -0.5 * width, 0.0],
            [0.0, -0.5 * width, 0.0],
            [0.0, 0.5 * width, 0.0],
            [-length, 0.5 * width, 0.0],
            [-length, -0.5 * width, height],
            [0.0, -0.5 * width, height],
            [0.0, 0.5 * width, height],
            [-length, 0.5 * width, height],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _passenger_template(
    length: float, width: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    shoulder = 0.55 * height
    roof_width = 0.43 * width
    vertices = np.asarray(
        [
            [-length, -0.5 * width, 0.0],
            [0.0, -0.5 * width, 0.0],
            [0.0, 0.5 * width, 0.0],
            [-length, 0.5 * width, 0.0],
            [-length, -0.5 * width, shoulder],
            [0.0, -0.5 * width, shoulder],
            [0.0, 0.5 * width, shoulder],
            [-length, 0.5 * width, shoulder],
            [-0.72 * length, -roof_width, height],
            [-0.20 * length, -roof_width, height],
            [-0.20 * length, roof_width, height],
            [-0.72 * length, roof_width, height],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2],
            [5, 9, 10], [5, 10, 6],
            [4, 7, 11], [4, 11, 8],
            [4, 8, 9], [4, 9, 5],
            [7, 6, 10], [7, 10, 11],
            [8, 11, 10], [8, 10, 9],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _vehicle_basis(angle_deg: float, slope_deg: float) -> np.ndarray:
    """Return local-forward/right/up columns from exact SUMO angles."""

    angle = math.radians(float(angle_deg))
    slope = math.radians(float(slope_deg))
    forward = np.asarray(
        [math.sin(angle) * math.cos(slope),
         math.cos(angle) * math.cos(slope),
         math.sin(slope)],
        dtype=np.float64,
    )
    right = np.asarray([math.cos(angle), -math.sin(angle), 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    return np.stack((forward, right, up), axis=1)


def build_dynamic_vehicle_mesh(
    *,
    frame_positions: np.ndarray,
    frame_headings_deg: np.ndarray,
    frame_slopes_deg: np.ndarray,
    active_nodes: np.ndarray,
    vehicle_type_ids: list[str],
    dimensions: dict[str, VehicleDimensions],
    antenna_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Build one combined metal mesh and exact roof-antenna positions."""

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    antennas: dict[int, np.ndarray] = {}
    vertex_offset = 0
    for raw_node in active_nodes:
        node = int(raw_node)
        position = np.asarray(frame_positions[node], dtype=np.float64)
        heading = float(frame_headings_deg[node])
        slope = float(frame_slopes_deg[node])
        if not (np.all(np.isfinite(position)) and math.isfinite(heading) and math.isfinite(slope)):
            raise ValueError(f"non-finite dynamic-vehicle pose for slot {node}")
        type_id = str(vehicle_type_ids[node])
        dims = dimensions[type_id]
        local_vertices, local_faces = (
            _box_template(dims.length, dims.width, dims.height)
            if dims.vclass == "bus"
            else _passenger_template(dims.length, dims.width, dims.height)
        )
        basis = _vehicle_basis(heading, slope)
        transformed = position.reshape(1, 3) + local_vertices @ basis.T
        vertices.append(transformed)
        faces.append(local_faces + int(vertex_offset))
        vertex_offset += int(local_vertices.shape[0])
        antenna_local = np.asarray(
            [-0.5 * dims.length, 0.0, dims.height + float(antenna_clearance_m)],
            dtype=np.float64,
        )
        antennas[node] = position + basis @ antenna_local
    if not vertices:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            antennas,
        )
    return np.concatenate(vertices), np.concatenate(faces), antennas


def write_ascii_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a compact combined blocker mesh accepted by Mitsuba/Sionna."""

    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_rows = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    face_rows = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertex_rows)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(face_rows)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertex_rows:
            stream.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in face_rows:
            stream.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def parse_step_list(raw: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if any(step < 0 for step in steps):
        raise ValueError("fidelity steps must be non-negative")
    return steps


def build_street_candidates_3d(
    *,
    sumo_net: Path,
    scene_manifest: dict,
    spacing_m: float,
    margin_m: float,
    antenna_height_m: float,
) -> np.ndarray:
    """Densify LuST3D passenger-lane polylines into local scene coordinates."""
    import sumolib

    bounds = [float(value) for value in scene_manifest["source_bounds_sumo_xy_m"]]
    xmin, ymin, xmax, ymax = bounds
    z_reference = float(scene_manifest["z_reference_m"])
    if spacing_m <= 0.0:
        raise ValueError("fidelity street spacing must be positive")
    if margin_m < 0.0 or 2.0 * margin_m >= min(xmax - xmin, ymax - ymin):
        raise ValueError("invalid fidelity margin")

    net = sumolib.net.readNet(str(sumo_net), withInternal=True)
    candidates: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for edge in net.getEdges(withInternal=False):
        for lane in edge.getLanes():
            if not lane.allows("passenger"):
                continue
            shape = [tuple(float(value) for value in point) for point in lane.getShape3D()]
            for a, b in zip(shape[:-1], shape[1:]):
                length = float(math.hypot(b[0] - a[0], b[1] - a[1]))
                count = max(1, int(math.ceil(length / float(spacing_m))))
                for alpha in np.linspace(0.0, 1.0, count + 1):
                    x = float(a[0] + alpha * (b[0] - a[0]) - xmin)
                    y = float(a[1] + alpha * (b[1] - a[1]) - ymin)
                    if not (
                        margin_m <= x <= (xmax - xmin) - margin_m
                        and margin_m <= y <= (ymax - ymin) - margin_m
                    ):
                        continue
                    z = float(
                        a[2]
                        + alpha * (b[2] - a[2])
                        - z_reference
                        + antenna_height_m
                    )
                    key = (round(x, 3), round(y, 3), round(z, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((x, y, z))
    if not candidates:
        raise ValueError("no passenger-lane fidelity candidates lie inside the scene crop")
    return np.asarray(candidates, dtype=np.float64)


def sample_fidelity_pairs_3d(
    transmitter_candidates: np.ndarray,
    *,
    receiver_candidates: np.ndarray | None = None,
    transmitter_weights: np.ndarray | None = None,
    receiver_weights: np.ndarray | None = None,
    n_tx: int,
    n_pairs: int,
    min_distance_m: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Sample a deterministic held-out pair set from exact 3D lane geometry."""
    tx_points = np.asarray(transmitter_candidates, dtype=np.float64)
    rx_points = (
        tx_points
        if receiver_candidates is None
        else np.asarray(receiver_candidates, dtype=np.float64)
    )
    if tx_points.ndim != 2 or tx_points.shape[1] != 3:
        raise ValueError("fidelity transmitter candidates must have shape (N, 3)")
    if rx_points.ndim != 2 or rx_points.shape[1] != 3:
        raise ValueError("fidelity receiver candidates must have shape (N, 3)")
    if n_pairs <= 0 or n_tx <= 0:
        raise ValueError("fidelity pair and transmitter counts must be positive")
    if not len(tx_points) or not len(rx_points):
        raise ValueError("fidelity transmitter and receiver candidates cannot be empty")

    def probabilities(raw: np.ndarray | None, count: int, label: str) -> np.ndarray | None:
        if raw is None:
            return None
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(values) != int(count):
            raise ValueError(f"{label} weights must match the candidate count")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{label} weights must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError(f"{label} weights must contain positive mass")
        return values / total

    tx_probabilities = probabilities(
        transmitter_weights, len(tx_points), "fidelity transmitter"
    )
    rx_probabilities = probabilities(
        receiver_weights, len(rx_points), "fidelity receiver"
    )
    tx_count = min(int(n_tx), int(n_pairs))
    tx_replace = len(tx_points) < tx_count
    if tx_probabilities is not None:
        tx_replace = tx_replace or int(np.count_nonzero(tx_probabilities)) < tx_count
    tx_indices = rng.choice(
        len(tx_points), size=tx_count, replace=tx_replace, p=tx_probabilities
    )
    pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    base_count, remainder = divmod(int(n_pairs), tx_count)
    for group_index, raw_tx_index in enumerate(tx_indices):
        tx_index = int(raw_tx_index)
        tx = tx_points[tx_index]
        distances = np.linalg.norm(rx_points[:, :2] - tx[:2], axis=1)
        eligible = np.flatnonzero(distances >= float(min_distance_m))
        if not len(eligible):
            raise ValueError(
                f"no fidelity receiver is at least {float(min_distance_m):g} m from a transmitter"
            )
        group_size = base_count + (1 if group_index < remainder else 0)
        eligible_probabilities = None
        if rx_probabilities is not None:
            eligible_probabilities = rx_probabilities[eligible]
            eligible_total = float(np.sum(eligible_probabilities))
            if eligible_total > 0.0:
                eligible_probabilities = eligible_probabilities / eligible_total
            else:
                eligible_probabilities = None
        rx_indices = rng.choice(
            eligible,
            size=group_size,
            replace=len(eligible) < group_size,
            p=eligible_probabilities,
        )
        tx_tuple = tuple(float(value) for value in tx)
        for raw_rx_index in rx_indices:
            rx = rx_points[int(raw_rx_index)]
            pairs.append((tx_tuple, tuple(float(value) for value in rx)))
    if len(pairs) != int(n_pairs):
        raise AssertionError(f"expected {n_pairs} fidelity pairs, got {len(pairs)}")
    return pairs


def route_density_candidate_weights(
    candidates: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    *,
    bandwidth_m: float,
    floor_fraction: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Estimate lane-point sampling mass from the full observed route density."""
    from scipy.spatial import cKDTree

    points = np.asarray(candidates, dtype=np.float64)
    states = np.asarray(positions, dtype=np.float64)
    present = np.asarray(active, dtype=np.bool_)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("route-density candidates must have shape (N, 3)")
    if states.ndim != 3 or states.shape[2] < 2 or present.shape != states.shape[:2]:
        raise ValueError("route-density positions/active arrays have incompatible shapes")
    if float(bandwidth_m) <= 0.0:
        raise ValueError("route-density bandwidth must be positive")
    if not 0.0 <= float(floor_fraction) <= 1.0:
        raise ValueError("route-density floor fraction must lie in [0, 1]")

    observations = states[present, :2]
    if not len(observations):
        raise ValueError("route-density weighting requires at least one active observation")
    tree = cKDTree(points[:, :2])
    _distances, nearest = tree.query(observations, k=1)
    counts = np.bincount(np.asarray(nearest, dtype=np.int64), minlength=len(points)).astype(
        np.float64
    )

    radius = 3.0 * float(bandwidth_m)
    density = np.zeros((len(points),), dtype=np.float64)
    for index, neighbors in enumerate(tree.query_ball_point(points[:, :2], r=radius)):
        neighbor_indices = np.asarray(neighbors, dtype=np.int64)
        offsets = points[neighbor_indices, :2] - points[index, :2]
        squared_distance = np.sum(offsets * offsets, axis=1)
        density[index] = float(
            np.sum(
                counts[neighbor_indices]
                * np.exp(-0.5 * squared_distance / float(bandwidth_m) ** 2)
            )
        )
    positive = density[density > 0.0]
    if not len(positive):
        raise ValueError("route-density weighting produced no positive candidate mass")
    floor_mass = float(floor_fraction) * float(np.mean(positive))
    weights = density + floor_mass
    weights /= float(np.sum(weights))
    return weights, {
        "active_route_observations": int(len(observations)),
        "directly_visited_candidate_count": int(np.count_nonzero(counts)),
        "positive_smoothed_candidate_count": int(np.count_nonzero(density)),
        "effective_candidate_count": float(1.0 / np.sum(weights * weights)),
        "bandwidth_m": float(bandwidth_m),
        "floor_fraction": float(floor_fraction),
    }

def trace_fidelity_pairs_3d(
    tracer: RayTracer,
    pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    *,
    map_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-trace 3D pairs without the legacy 2D pair-group canonicalization."""
    from rl_reward_experiment.measurement import _gains_to_rssi

    tx_positions = list(dict.fromkeys(tx for tx, _rx in pairs))
    rx_positions = list(dict.fromkeys(rx for _tx, rx in pairs))
    tx_index = {position: index for index, position in enumerate(tx_positions)}
    rx_index = {position: index for index, position in enumerate(rx_positions)}
    gains = tracer._solve_pairs_tx_chunked(tx_positions, rx_positions)
    rssi = _gains_to_rssi(
        gains,
        tx_power_dbm=tracer.tx_power_dbm,
        rssi_min=tracer.rssi_min,
        rssi_max=tracer.rssi_max,
    )
    features = np.asarray(
        [
            [tx[0] / map_size, tx[1] / map_size, rx[0] / map_size, rx[1] / map_size]
            for tx, rx in pairs
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [rssi[rx_index[rx], tx_index[tx]] for tx, rx in pairs], dtype=np.float32
    ).reshape(-1, 1)
    return features, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--radio-net", type=Path, required=True)
    parser.add_argument("--sumo-net-3d", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--num-zones", type=int, default=1)
    parser.add_argument("--zones", type=int, nargs="+")
    parser.add_argument(
        "--region-bounds",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help=("Trace one arbitrary rectangle in local Sionna-scene coordinates; "
              "nodes outside it remain physical blockers but produce no measurements."),
    )
    parser.add_argument(
        "--measurement-receiver-nodes",
        type=int,
        nargs="+",
        help=(
            "Trace directed measurements only into these node indices. "
            "All active same-zone nodes remain candidate transmitters."
        ),
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--num-rays", type=int, default=20_000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--disable-refraction",
        action="store_true",
        help="Make scene surfaces opaque; keep LOS and specular reflections only.",
    )
    parser.add_argument("--tx-batch-size", type=int, default=20)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--rssi-min-dbm", type=float, default=-120.0)
    parser.add_argument("--rssi-max-dbm", type=float, default=0.0)
    parser.add_argument("--antenna-height-m", type=float, default=1.5)
    parser.add_argument(
        "--dynamic-vehicle-meshes",
        action="store_true",
        help="Insert every active LuST vehicle as an oriented metal mesh before tracing each frame.",
    )
    parser.add_argument(
        "--vehicle-roof-antennas",
        action="store_true",
        help="Use exact type-specific roof antennas without inserting vehicle blocker meshes.",
    )
    parser.add_argument(
        "--vehicle-type-file",
        type=Path,
        help="LuST SUMO additional file containing the referenced vType definitions.",
    )
    parser.add_argument(
        "--vehicle-mesh-dir",
        type=Path,
        help="Temporary directory for per-frame combined vehicle PLY meshes.",
    )
    parser.add_argument("--vehicle-antenna-clearance-m", type=float, default=0.1)
    parser.add_argument("--vehicle-metal-thickness-m", type=float, default=0.01)
    parser.add_argument("--fidelity-pairs", type=int, default=0)
    parser.add_argument("--fidelity-n-tx", type=int, default=10)
    parser.add_argument("--fidelity-steps", default="30,35,40")
    parser.add_argument("--fidelity-street-spacing-m", type=float, default=5.0)
    parser.add_argument("--fidelity-margin-m", type=float, default=10.0)
    parser.add_argument("--fidelity-min-distance-m", type=float, default=20.0)
    parser.add_argument(
        "--fidelity-route-density-weighted",
        action="store_true",
        help=(
            "Sample fixed held-out transmitter and receiver lane points in proportion "
            "to the full mobility trace's smoothed route occupancy."
        ),
    )
    parser.add_argument("--fidelity-route-density-bandwidth-m", type=float, default=10.0)
    parser.add_argument("--fidelity-route-density-floor-fraction", type=float, default=0.01)
    parser.add_argument(
        "--fidelity-static-scene",
        action="store_true",
        help=(
            "Exclude transient vehicle blocker meshes from fidelity targets so the "
            "held-out set measures the persistent building/terrain radio map."
        ),
    )
    parser.add_argument(
        "--fidelity-global-senders",
        action="store_true",
        help=(
            "Sample fidelity transmitters across the full map while keeping "
            "receivers inside each requested zone."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_step = int(args.start_step)
    end_step = int(args.steps if args.end_step is None else args.end_step)
    if not 0 <= start_step <= end_step <= int(args.steps):
        raise ValueError(
            "Require 0 <= start-step <= end-step <= steps, where steps is "
            "the complete simulation length"
        )
    is_shard = start_step not in (0, 1) or end_step != int(args.steps)
    import sionna.rt as rt
    from rl_reward_experiment.measurement import RayTracer, _gains_to_rssi
    from rl_reward_experiment.mobility import zone_of

    (
        mobility_meta,
        vehicle_ids,
        positions,
        active,
        headings,
        slopes,
        speeds,
    ) = load_mobility(args.mobility, nodes=args.nodes, steps=args.steps)
    scene_manifest = json.loads(args.scene_manifest.read_text())
    measurement_bounds = tuple(
        float(value) for value in scene_manifest["measurement_bounds_local_xy_m"]
    )
    zone_map_size = max(
        measurement_bounds[2] - measurement_bounds[0],
        measurement_bounds[3] - measurement_bounds[1],
    )
    region_bounds = (
        None
        if args.region_bounds is None
        else tuple(float(value) for value in args.region_bounds)
    )
    if region_bounds is not None:
        rxmin, rymin, rxmax, rymax = region_bounds
        if not (rxmin < rxmax and rymin < rymax):
            raise ValueError("--region-bounds must have positive width and height")
        mxmin, mymin, mxmax, mymax = measurement_bounds
        if not (mxmin <= rxmin and mymin <= rymin and rxmax <= mxmax and rymax <= mymax):
            raise ValueError("--region-bounds must lie inside the scene measurement bounds")
        num_zones = 1
        zones_per_side = 1

        def spatial_zone(x: float, y: float) -> int:
            return int(rxmin <= x <= rxmax and rymin <= y <= rymax) - 1

        measurement_active = active & (
            (positions[:, :, 0] >= rxmin)
            & (positions[:, :, 0] <= rxmax)
            & (positions[:, :, 1] >= rymin)
            & (positions[:, :, 1] <= rymax)
        )
    else:
        num_zones = int(args.num_zones)
        zones_per_side = int(math.isqrt(num_zones))
        if num_zones < 1 or zones_per_side * zones_per_side != num_zones:
            raise ValueError("--num-zones must be a positive perfect square")

        def spatial_zone(x: float, y: float) -> int:
            return int(zone_of(x, y, zone_map_size, num_zones))

        measurement_active = active
    traced_zones = (
        tuple(range(num_zones))
        if args.zones is None
        else tuple(sorted(set(int(zone) for zone in args.zones)))
    )
    if not traced_zones or any(zone < 0 or zone >= num_zones for zone in traced_zones):
        raise ValueError("--zones must contain valid zone indices")
    measurement_receiver_nodes = (
        None
        if args.measurement_receiver_nodes is None
        else set(map(int, args.measurement_receiver_nodes))
    )
    if measurement_receiver_nodes is not None and (
        not measurement_receiver_nodes
        or min(measurement_receiver_nodes) < 0
        or max(measurement_receiver_nodes) >= int(args.nodes)
    ):
        raise ValueError("--measurement-receiver-nodes contains an invalid node")

    vehicle_types_by_id = {
        str(key): str(value)
        for key, value in dict(mobility_meta.get("vehicle_types", {})).items()
    }
    vehicle_type_ids = [vehicle_types_by_id.get(vehicle_id, "") for vehicle_id in vehicle_ids]
    dynamic_vehicle_meshes = bool(args.dynamic_vehicle_meshes)
    vehicle_roof_antennas = dynamic_vehicle_meshes or bool(args.vehicle_roof_antennas)
    dimensions: dict[str, VehicleDimensions] = {}
    mesh_dir: Path | None = None
    if vehicle_roof_antennas:
        if headings is None or slopes is None:
            raise ValueError(
                "vehicle roof antennas require exact heading_traces_deg and "
                "slope_traces_deg in the mobility trace"
            )
        if args.vehicle_type_file is None:
            raise ValueError("--vehicle-type-file is required with vehicle roof antennas")
        if float(args.vehicle_antenna_clearance_m) <= 0.0:
            raise ValueError("vehicle antenna clearance must be positive")
        dimensions = load_vehicle_dimensions(vehicle_types_by_id, args.vehicle_type_file)
    if dynamic_vehicle_meshes:
        mesh_dir = (
            args.vehicle_mesh_dir
            if args.vehicle_mesh_dir is not None
            else Path(tempfile.mkdtemp(prefix="lust-sionna-vehicles-"))
        )
        mesh_dir.mkdir(parents=True, exist_ok=True)
    scene = rt.load_scene(str(args.scene))
    scene.frequency = args.frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    tracer = RayTracer(
        scene,
        num_rays=args.num_rays,
        max_depth=args.max_depth,
        tx_power_dbm=args.tx_power_dbm,
        rssi_min=args.rssi_min_dbm,
        rssi_max=args.rssi_max_dbm,
        tx_batch_size=args.tx_batch_size,
        refraction=not bool(args.disable_refraction),
    )

    fidelity_steps = parse_step_list(args.fidelity_steps)
    fidelity_events: list[dict[str, object]] = []
    fidelity_arrays: dict[str, np.ndarray] = {}
    fidelity_generation: dict[str, object] | None = None
    fidelity_pairs_by_zone: dict[
        int, list[tuple[tuple[float, float, float], tuple[float, float, float]]]
    ] = {}
    fidelity_map_size = 0.0
    fidelity_candidates_by_zone: dict[int, int] = {}
    fidelity_density_by_zone: dict[int, dict[str, float | int]] = {}
    if int(args.fidelity_pairs) > 0:
        if args.sumo_net_3d is None:
            raise ValueError("--sumo-net-3d is required when --fidelity-pairs is positive")
        if not fidelity_steps or fidelity_steps[0] <= 0:
            raise ValueError("scheduled fidelity steps must be positive")
        if fidelity_steps[-1] > int(args.steps):
            raise ValueError("fidelity step exceeds the generated simulation length")
        measurement_bounds = [
            float(value) for value in scene_manifest["measurement_bounds_local_xy_m"]
        ]
        fidelity_map_size = max(
            measurement_bounds[2] - measurement_bounds[0],
            measurement_bounds[3] - measurement_bounds[1],
        )
        candidates = build_street_candidates_3d(
            sumo_net=args.sumo_net_3d,
            scene_manifest=scene_manifest,
            spacing_m=float(args.fidelity_street_spacing_m),
            margin_m=float(args.fidelity_margin_m),
            antenna_height_m=float(args.antenna_height_m),
        )
        candidate_zones = np.asarray(
            [
                spatial_zone(float(point[0]), float(point[1]))
                for point in candidates
            ],
            dtype=np.int32,
        )
        density_weights = None
        density_summary: dict[str, float | int] = {}
        if bool(args.fidelity_route_density_weighted):
            density_weights, density_summary = route_density_candidate_weights(
                candidates,
                positions,
                active,
                bandwidth_m=float(args.fidelity_route_density_bandwidth_m),
                floor_fraction=float(args.fidelity_route_density_floor_fraction),
            )
        for az in traced_zones:
            zone_mask = candidate_zones == int(az)
            zone_candidates = candidates[zone_mask]
            fidelity_candidates_by_zone[int(az)] = int(len(zone_candidates))
            if len(zone_candidates) < 2:
                raise ValueError(f"zone {az} has fewer than two fidelity candidates")
            zone_weights = None if density_weights is None else density_weights[zone_mask]
            if zone_weights is not None:
                zone_probability_mass = float(np.sum(zone_weights))
                zone_weights = zone_weights / zone_probability_mass
                fidelity_density_by_zone[int(az)] = {
                    **density_summary,
                    "zone_probability_mass": zone_probability_mass,
                    "zone_effective_candidate_count": float(
                        1.0 / np.sum(zone_weights * zone_weights)
                    ),
                }
            fidelity_pairs_by_zone[int(az)] = sample_fidelity_pairs_3d(
                candidates if args.fidelity_global_senders else zone_candidates,
                receiver_candidates=zone_candidates,
                transmitter_weights=(
                    density_weights if args.fidelity_global_senders else zone_weights
                ),
                receiver_weights=zone_weights,
                n_tx=int(args.fidelity_n_tx),
                n_pairs=int(args.fidelity_pairs),
                min_distance_m=float(args.fidelity_min_distance_m),
                rng=np.random.default_rng(int(args.seed) + 9_999 + 100_003 * int(az)),
            )

    node_states = np.zeros((args.steps + 1, args.nodes, 3), dtype=np.float32)
    node_states[:, :, :2] = positions[:, :, :2]
    for state_step in range(args.steps + 1):
        for node in range(args.nodes):
            node_states[state_step, node, 2] = float(
                spatial_zone(
                    float(positions[state_step, node, 0]),
                    float(positions[state_step, node, 1]),
                )
            )
    node_generations = np.zeros((args.steps + 1, args.nodes), dtype=np.int32)
    synced = measurement_active.sum(axis=1).astype(np.int32)
    measurement_rows: list[list[float]] = []
    dynamic_object = None
    dynamic_mesh_vertices: list[int] = []
    dynamic_mesh_faces: list[int] = []
    vehicle_material = (
        rt.ITURadioMaterial(
            name="lust-vehicle-metal",
            itu_type="metal",
            thickness=float(args.vehicle_metal_thickness_m),
        )
        if dynamic_vehicle_meshes
        else None
    )

    if vehicle_roof_antennas:
        initial_nodes = np.flatnonzero(active[0])
        _vertices, _faces, initial_antennas = build_dynamic_vehicle_mesh(
            frame_positions=positions[0],
            frame_headings_deg=headings[0],
            frame_slopes_deg=slopes[0],
            active_nodes=initial_nodes,
            vehicle_type_ids=vehicle_type_ids,
            dimensions=dimensions,
            antenna_clearance_m=float(args.vehicle_antenna_clearance_m),
        )
        for node, antenna in initial_antennas.items():
            node_states[0, int(node), :2] = antenna[:2]
            if region_bounds is None:
                node_states[0, int(node), 2] = float(
                    spatial_zone(float(antenna[0]), float(antenna[1]))
                )

    for step in range(start_step, end_step + 1):
        physical_active_nodes = np.flatnonzero(active[step])
        active_nodes = np.flatnonzero(measurement_active[step])
        if vehicle_roof_antennas:
            assert headings is not None and slopes is not None
            vertices, faces, antennas = build_dynamic_vehicle_mesh(
                frame_positions=positions[step],
                frame_headings_deg=headings[step],
                frame_slopes_deg=slopes[step],
                active_nodes=physical_active_nodes,
                vehicle_type_ids=vehicle_type_ids,
                dimensions=dimensions,
                antenna_clearance_m=float(args.vehicle_antenna_clearance_m),
            )
            xyz = (
                np.stack(
                    [antennas[int(node)] for node in active_nodes], axis=0
                )
                if len(active_nodes)
                else np.empty((0, 3), dtype=np.float64)
            )
            for node, antenna in antennas.items():
                node_states[step, int(node), :2] = antenna[:2]
                if region_bounds is None:
                    node_states[step, int(node), 2] = float(
                        spatial_zone(float(antenna[0]), float(antenna[1]))
                    )
            if dynamic_vehicle_meshes:
                assert mesh_dir is not None and vehicle_material is not None
                mesh_path = mesh_dir / f"vehicles_step_{step:03d}.ply"
                write_ascii_ply(mesh_path, vertices, faces)
                new_dynamic_object = rt.SceneObject(
                    fname=str(mesh_path),
                    name=f"lust-moving-vehicles-{step:03d}",
                    radio_material=vehicle_material,
                )
                scene.edit(add=new_dynamic_object, remove=dynamic_object)
                dynamic_object = new_dynamic_object
                dynamic_mesh_vertices.append(int(len(vertices)))
                dynamic_mesh_faces.append(int(len(faces)))
        else:
            xyz = positions[step, active_nodes].copy()
            xyz[:, 2] += args.antenna_height_m
        directed_links = 0
        if len(active_nodes) >= 2:
            # Explicit regions select samples by their supplied SUMO positions.
            # A roof antenna may extend slightly beyond the boundary; that must
            # not remove an otherwise in-region vehicle pair.
            active_zones = (
                np.zeros(len(active_nodes), dtype=np.int32)
                if region_bounds is not None
                else np.asarray(
                    [spatial_zone(float(point[0]), float(point[1])) for point in xyz],
                    dtype=np.int32,
                )
            )
            for az in traced_zones:
                local_indices = np.flatnonzero(active_zones == int(az))
                if len(local_indices) < 2:
                    continue
                receiver_local_indices = (
                    local_indices
                    if measurement_receiver_nodes is None
                    else np.asarray(
                        [
                            index
                            for index in local_indices
                            if int(active_nodes[index])
                            in measurement_receiver_nodes
                        ],
                        dtype=np.int64,
                    )
                )
                if not len(receiver_local_indices):
                    continue
                zone_xyz = xyz[local_indices]
                receiver_xyz = xyz[receiver_local_indices]
                gains = tracer._solve_pairs_tx_chunked(
                    [tuple(point) for point in zone_xyz],
                    [tuple(point) for point in receiver_xyz],
                )
                rssi = _gains_to_rssi(
                    gains,
                    tx_power_dbm=args.tx_power_dbm,
                    rssi_min=args.rssi_min_dbm,
                    rssi_max=args.rssi_max_dbm,
                )
                directed_links += int(
                    len(receiver_local_indices) * (len(local_indices) - 1)
                )
                for local_tx, active_tx in enumerate(local_indices):
                    for output_rx, active_rx in enumerate(receiver_local_indices):
                        if int(active_tx) == int(active_rx):
                            continue
                        measurement_rows.append(
                            [step, int(az), int(active_nodes[active_tx]), int(active_nodes[active_rx]), float(rssi[output_rx, local_tx])]
                        )
        if fidelity_pairs_by_zone and dynamic_vehicle_meshes and step in fidelity_steps:
            event_index = fidelity_steps.index(int(step))
            if args.fidelity_static_scene:
                scene.edit(remove=dynamic_object)
            try:
                for az, fidelity_pairs in fidelity_pairs_by_zone.items():
                    fidelity_X, fidelity_y = trace_fidelity_pairs_3d(
                        tracer, fidelity_pairs, map_size=float(fidelity_map_size)
                    )
                    fidelity_arrays[f"fid_{event_index:04d}_z{az}_X"] = fidelity_X
                    fidelity_arrays[f"fid_{event_index:04d}_z{az}_y"] = fidelity_y
            finally:
                if args.fidelity_static_scene:
                    scene.edit(add=dynamic_object)
        print(
            f"step={step}/{args.steps} active={len(active_nodes)} "
            f"physical_active={len(physical_active_nodes)} "
            f"directed_links={directed_links} zones={num_zones} "
            f"dynamic_vehicle_faces={dynamic_mesh_faces[-1] if dynamic_mesh_faces else 0}",
            flush=True,
        )

    measurements = np.asarray(measurement_rows, dtype=np.float64)
    if measurements.size == 0:
        measurements = np.empty((0, 5), dtype=np.float64)
    else:
        for row in measurements:
            step, az, tx, rx, _ = row
            if not (measurement_active[int(step), int(tx)] and measurement_active[int(step), int(rx)]):
                raise AssertionError("Measurement references an inactive physical vehicle")
            endpoint_zones = {
                int(node_states[int(step), int(tx), 2]),
                int(node_states[int(step), int(rx), 2]),
            }
            if endpoint_zones != {int(az)}:
                raise AssertionError(
                    f"Measurement zone {int(az)} differs from endpoint zones {endpoint_zones}"
                )

    local_fidelity_events = [
        (index, int(step))
        for index, step in enumerate(fidelity_steps)
        if start_step <= int(step) <= end_step
    ]
    if fidelity_pairs_by_zone:
        if not dynamic_vehicle_meshes:
            for az, fidelity_pairs in fidelity_pairs_by_zone.items():
                fidelity_X, fidelity_y = trace_fidelity_pairs_3d(
                    tracer, fidelity_pairs, map_size=float(fidelity_map_size)
                )
                for event_index, _event_step in local_fidelity_events:
                    fidelity_arrays[f"fid_{event_index:04d}_z{az}_X"] = fidelity_X.copy()
                    fidelity_arrays[f"fid_{event_index:04d}_z{az}_y"] = fidelity_y.copy()
        missing_fidelity = [
            (int(step), int(az))
            for index, step in local_fidelity_events
            for az in traced_zones
            if f"fid_{index:04d}_z{az}_y" not in fidelity_arrays
        ]
        if missing_fidelity:
            raise AssertionError(f"missing dynamic fidelity traces at step/zones {missing_fidelity}")
        local_fidelity_arrays: dict[str, np.ndarray] = {}
        for output_index, (event_index, event_step) in enumerate(local_fidelity_events):
            fidelity_events.append(
                {
                    "step": int(event_step),
                    "n_pairs": int(args.fidelity_pairs),
                    "zones": list(traced_zones),
                }
            )
            for az in traced_zones:
                local_fidelity_arrays[f"fid_{output_index:04d}_z{az}_X"] = fidelity_arrays[
                    f"fid_{event_index:04d}_z{az}_X"
                ]
                local_fidelity_arrays[f"fid_{output_index:04d}_z{az}_y"] = fidelity_arrays[
                    f"fid_{event_index:04d}_z{az}_y"
                ]
        fidelity_arrays = local_fidelity_arrays
        fidelity_generation = {
            "mode": (
                "fixed-held-out-lust3d-passenger-lane-pairs-static-map"
                if args.fidelity_static_scene
                else (
                    "fixed-held-out-lust3d-passenger-lane-pairs-with-dynamic-vehicle-blockers"
                    if dynamic_vehicle_meshes
                    else "fixed-held-out-lust3d-passenger-lane-pairs"
                )
            ),
            "sumo_net_3d": str(args.sumo_net_3d),
            "candidate_count_by_zone": fidelity_candidates_by_zone,
            "global_sender_candidates": bool(args.fidelity_global_senders),
            "pair_sampling": (
                "smoothed-full-trace-route-density"
                if args.fidelity_route_density_weighted
                else "uniform-passenger-lane-geometry"
            ),
            "route_density_by_zone": fidelity_density_by_zone,
            "pair_count_per_zone": int(args.fidelity_pairs),
            "total_pair_count": int(args.fidelity_pairs) * len(traced_zones),
            "n_tx_per_zone": int(args.fidelity_n_tx),
            "num_zones": num_zones,
            "steps": list(fidelity_steps),
            "street_spacing_m": float(args.fidelity_street_spacing_m),
            "margin_m": float(args.fidelity_margin_m),
            "min_pair_distance_m": float(args.fidelity_min_distance_m),
            "position_source": (
                "LuST3D lane.getShape3D polylines transformed into the Sionna scene; "
                "only antenna height is added"
            ),
            "targets_used_for_training": False,
            "dynamic_vehicle_geometry_retraced_per_event": bool(
                dynamic_vehicle_meshes and not args.fidelity_static_scene
            ),
            "fidelity_excludes_dynamic_vehicle_blockers": bool(
                args.fidelity_static_scene
            ),
            "no_path_censor_floor_dbm": float(args.rssi_min_dbm),
        }
        print(
            f"fidelity candidates_by_zone={fidelity_candidates_by_zone} "
            f"fixed_pairs_per_zone={int(args.fidelity_pairs)} "
            f"events={list(fidelity_steps)} zones={num_zones}",
            flush=True,
        )

    meta = {
        "format": "sumo_rssi_trace_v3_shard" if is_shard else "sumo_rssi_trace_v3",
        "seed": args.seed,
        "sim_steps": args.steps,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "num_zones": num_zones,
        "traced_zones": list(traced_zones),
        "cars_per_zone": args.nodes,
        "num_nodes": args.nodes,
        "zone_layout": (
            {
                "type": "custom-rectangle",
                "bounds_local_xy_m": list(region_bounds),
                "width_m": region_bounds[2] - region_bounds[0],
                "height_m": region_bounds[3] - region_bounds[1],
                "numbering": "single region labelled zero; outside labelled minus one",
            }
            if region_bounds is not None
            else {
                "type": "equal-square-grid",
                "zones_per_side": zones_per_side,
                "map_size_m": zone_map_size,
                "cell_size_m": zone_map_size / zones_per_side,
                "numbering": "row-major from south-west to north-east",
            }
        ),
        "mitsuba_variant": os.environ.get("MI_DEFAULT_VARIANT", "unknown"),
        "physical_vehicle_ids": vehicle_ids,
        "physical_vehicle_types": vehicle_types_by_id,
        "identity_semantics": (
            "Each node slot is permanently assigned to one LuST vehicle ID. Its model "
            "is retained while node_active is false and reused if that ID re-enters."
        ),
        "mobility_source": str(args.mobility),
        "mobility_start_time_s": mobility_meta.get(
            "start_time_s", mobility_meta.get("source_begin_s")
        ),
        "mobility_step_length_s": mobility_meta.get(
            "step_length_s", mobility_meta.get("sample_period_s")
        ),
        "scene_xml": str(args.scene),
        "scene_manifest": str(args.scene_manifest),
        "radio_net": str(args.radio_net),
        "scene_limitations": (
            "Static LuST3D buildings, terrain and bridge geometry plus exact-frame "
            "active LuST vehicle meshes; tire, window and articulated-body details "
            "are represented by one metal low-poly body per vehicle."
            if dynamic_vehicle_meshes
            else (
                "Static LuST3D buildings, terrain and bridge geometry; active "
                "vehicles retain exact roof antennas but are absent from scene geometry."
                if vehicle_roof_antennas
                else scene_manifest.get("pilot_limitations", "")
            )
        ),
        "frequency_hz": args.frequency_hz,
        "tx_power_dbm": args.tx_power_dbm,
        "num_rays": args.num_rays,
        "max_depth": args.max_depth,
        "tx_batch_size": args.tx_batch_size,
        "propagation_phenomena": {
            "line_of_sight": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": not bool(args.disable_refraction),
            "diffraction": False,
            "edge_diffraction": False,
        },
        "buildings_opaque": bool(args.disable_refraction),
        "antenna_height_m": (
            None if vehicle_roof_antennas else args.antenna_height_m
        ),
        "vehicle_antenna_model": (
            {
                "type": "exact-type-specific-roof",
                "pose_source": "exact SUMO FCD x/y/z/angle/slope at each stored frame",
                "rule": "vehicle roof height plus fixed clearance at longitudinal center",
                "clearance_m": float(args.vehicle_antenna_clearance_m),
                "vehicle_type_file": str(args.vehicle_type_file),
            }
            if vehicle_roof_antennas
            else {
                "type": "fixed-height-above-SUMO-position",
                "height_m": float(args.antenna_height_m),
            }
        ),
        "dynamic_vehicle_blockers": bool(dynamic_vehicle_meshes),
        "dynamic_vehicle_geometry": (
            {
                "pose_source": "exact SUMO FCD x/y/z/angle/slope at each stored frame",
                "position_interpolation": False,
                "material": "ITU metal",
                "material_thickness_m": float(args.vehicle_metal_thickness_m),
                "antenna_rule": "vehicle roof height plus fixed clearance at longitudinal center",
                "antenna_clearance_m": float(args.vehicle_antenna_clearance_m),
                "vehicle_type_file": str(args.vehicle_type_file),
                "mesh_vertices_min_median_max": [
                    int(np.min(dynamic_mesh_vertices)),
                    float(np.median(dynamic_mesh_vertices)),
                    int(np.max(dynamic_mesh_vertices)),
                ],
                "mesh_faces_min_median_max": [
                    int(np.min(dynamic_mesh_faces)),
                    float(np.median(dynamic_mesh_faces)),
                    int(np.max(dynamic_mesh_faces)),
                ],
                "type_dimensions_m": {
                    type_id: {
                        "length": float(value.length),
                        "width": float(value.width),
                        "height": float(value.height),
                        "vclass": value.vclass,
                        "gui_shape": value.gui_shape,
                    }
                    for type_id, value in sorted(dimensions.items())
                    if type_id in set(vehicle_type_ids)
                },
            }
            if dynamic_vehicle_meshes
            else None
        ),
        "rssi_min_dbm": args.rssi_min_dbm,
        "rssi_max_dbm": args.rssi_max_dbm,
        "no_path_semantics": (
            "Sionna links with no resolved path are censored at rssi_min_dbm, "
            "matching the predictor benchmark's supported target range."
        ),
        "fidelity_available": bool(fidelity_events),
        "fidelity_events": fidelity_events,
        "fidelity_generation": fidelity_generation,
        "refresh_zones_by_step": {
            str(int(event["step"])): list(traced_zones) for event in fidelity_events
        },
        "measurement_receiver_nodes": (
            None
            if measurement_receiver_nodes is None
            else sorted(measurement_receiver_nodes)
        ),
    }
    state_start_step = 0 if start_step == 1 else start_step
    meta["state_start_step"] = int(state_start_step)
    trace_arrays: dict[str, np.ndarray] = {
        "node_states": node_states[state_start_step : end_step + 1],
        "node_generations": node_generations[state_start_step : end_step + 1],
        "node_active": measurement_active[state_start_step : end_step + 1],
        "synced": synced[state_start_step : end_step + 1],
        "measurements": measurements,
    }
    if is_shard:
        trace_arrays["dynamic_mesh_vertices"] = np.asarray(
            dynamic_mesh_vertices, dtype=np.int32
        )
        trace_arrays["dynamic_mesh_faces"] = np.asarray(
            dynamic_mesh_faces, dtype=np.int32
        )
    save_trace(args.output, meta=meta, **trace_arrays, **fidelity_arrays)
    print(f"wrote {args.output} ({len(measurements)} directed measurements)", flush=True)


if __name__ == "__main__":
    main()
