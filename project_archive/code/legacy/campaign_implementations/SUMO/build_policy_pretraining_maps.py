#!/usr/bin/env python3
"""Build diverse source-only SUMO maps for policy pretraining.

The generated maps are deliberately separate from ``single_zone_urban_150``.
They share only the normalized coordinate convention and predictor input
schema used at deployment.  Four maps are training sources and two maps are
held out for policy validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "SUMO" / "policy_pretraining_maps"
COORDINATES = (0.0, 36.0, 72.0, 108.0, 144.0, 180.0)
MATERIALS = (
    ("itu_concrete", "126,132,139", 7.0),
    ("itu_brick", "151,111,91", 9.0),
    ("itu_stone", "119,122,128", 11.0),
    ("itu_wood", "159,137,102", 5.0),
    ("itu_glass", "94,139,162", 6.0),
)


@dataclass(frozen=True)
class SourceSpec:
    map_id: str
    split: str
    seed: int
    road_density: float
    building_density: float
    style: str


SPECS = (
    SourceSpec("source_train_00_dense_grid", "train", 1103, 0.92, 0.80, "dense"),
    SourceSpec("source_train_01_two_corridors", "train", 2207, 0.72, 0.64, "corridors"),
    SourceSpec("source_train_02_open_campus", "train", 3313, 0.84, 0.44, "open"),
    SourceSpec("source_train_03_irregular_blocks", "train", 4421, 0.66, 0.72, "irregular"),
    SourceSpec("source_valid_00_ring_spokes", "validation", 5527, 0.76, 0.60, "ring"),
    SourceSpec("source_valid_01_staggered", "validation", 6637, 0.70, 0.68, "staggered"),
    SourceSpec("source_test_00_cross_core", "test", 7741, 0.68, 0.76, "cross"),
    SourceSpec("source_test_01_sparse_campus", "test", 8851, 0.60, 0.36, "open"),
    SourceSpec("source_test_02_dense_irregular", "test", 9967, 0.82, 0.88, "irregular"),
)


@dataclass(frozen=True)
class Building:
    building_id: str
    bounds: tuple[float, float, float, float]
    height: float
    material: str
    color: str
    district: str
    dynamic_event: str | None = None

    @property
    def shape(self) -> str:
        x0, y0, x1, y1 = self.bounds
        return " ".join(
            f"{x:.3f},{y:.3f}"
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        )


GridNode = tuple[int, int]
Road = tuple[GridNode, GridNode]


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _node(col: int, row: int) -> str:
    return f"N{col}_{row}"


def _edge(left: GridNode, right: GridNode) -> str:
    return f"{_node(*left)}__{_node(*right)}"


def _canonical(left: GridNode, right: GridNode) -> Road:
    return (left, right) if left < right else (right, left)


def _full_grid() -> set[Road]:
    roads: set[Road] = set()
    width = len(COORDINATES)
    for col in range(width):
        for row in range(width):
            if col + 1 < width:
                roads.add(_canonical((col, row), (col + 1, row)))
            if row + 1 < width:
                roads.add(_canonical((col, row), (col, row + 1)))
    return roads


def _connected(roads: set[Road]) -> bool:
    adjacency: dict[GridNode, set[GridNode]] = {}
    for left, right in roads:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    if len(adjacency) != len(COORDINATES) ** 2:
        return False
    start = next(iter(adjacency))
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(adjacency)


def _border_nodes() -> set[GridNode]:
    last = len(COORDINATES) - 1
    return {
        (col, row)
        for col in range(last + 1)
        for row in range(last + 1)
        if col in {0, last} or row in {0, last}
    }


def _make_roads(spec: SourceSpec) -> set[Road]:
    """Prune a full grid while preserving connectivity and every border gate."""

    rng = random.Random(spec.seed)
    roads = _full_grid()
    protected: set[Road] = set()
    last = len(COORDINATES) - 1
    # Each border intersection keeps at least one edge pointing toward the
    # interior, so it remains a valid entry and exit gate.
    for col, row in sorted(_border_nodes()):
        inward = (
            (1, row) if col == 0 else
            (last - 1, row) if col == last else
            (col, 1) if row == 0 else
            (col, last - 1)
        )
        protected.add(_canonical((col, row), inward))

    if spec.style == "corridors":
        protected.update(
            road for road in roads
            if (
                road[0][0] == road[1][0] and road[0][0] in {1, 4}
            ) or (
                road[0][1] == road[1][1] and road[0][1] in {1, 4}
            )
        )
    elif spec.style == "ring":
        protected.update(
            road for road in roads
            if all(node[0] in {1, 4} or node[1] in {1, 4} for node in road)
        )
    elif spec.style == "staggered":
        protected.update(
            road for road in roads
            if (sum(road[0]) + sum(road[1])) % 4 == 0
        )
    elif spec.style == "cross":
        protected.update(
            road for road in roads
            if (
                road[0][0] == road[1][0] and road[0][0] in {2, 3}
            ) or (
                road[0][1] == road[1][1] and road[0][1] in {2, 3}
            )
        )

    target = max(len(protected), int(round(len(roads) * spec.road_density)))
    candidates = list(roads - protected)
    rng.shuffle(candidates)
    for road in candidates:
        if len(roads) <= target:
            break
        proposal = set(roads)
        proposal.remove(road)
        if _connected(proposal):
            roads = proposal
    if not _connected(roads):
        raise RuntimeError(f"{spec.map_id}: generated road graph is disconnected")
    return roads


def _buildings(spec: SourceSpec) -> list[Building]:
    rng = random.Random(spec.seed + 901)
    cells = [(col, row) for col in range(5) for row in range(5)]
    rng.shuffle(cells)
    count = max(6, int(round(len(cells) * spec.building_density)))
    occupied = sorted(cells[:count])
    dynamic_cells = max(
        ((a, b) for a in occupied for b in occupied if a < b),
        key=lambda pair: abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]),
    )
    buildings: list[Building] = []
    for index, (col, row) in enumerate(occupied):
        left, right = COORDINATES[col], COORDINATES[col + 1]
        bottom, top = COORDINATES[row], COORDINATES[row + 1]
        margin_x = rng.uniform(5.5, 9.0)
        margin_y = rng.uniform(5.5, 9.0)
        if spec.style == "open":
            margin_x += 2.5
            margin_y += 2.5
        if spec.style == "irregular":
            left_shift = rng.uniform(-2.0, 2.0)
            bottom_shift = rng.uniform(-2.0, 2.0)
        else:
            left_shift = bottom_shift = 0.0
        bounds = (
            left + margin_x + left_shift,
            bottom + margin_y + bottom_shift,
            right - margin_x + left_shift,
            top - margin_y + bottom_shift,
        )
        material, color, base_height = rng.choice(MATERIALS)
        height = base_height * rng.uniform(0.75, 1.75)
        dynamic_index = (
            0 if (col, row) == dynamic_cells[0]
            else 1 if (col, row) == dynamic_cells[1]
            else None
        )
        event_id = (
            f"{spec.map_id}_periodic_{dynamic_index}"
            if dynamic_index is not None else None
        )
        buildings.append(
            Building(
                building_id=f"building_{index:02d}",
                bounds=tuple(round(value, 3) for value in bounds),
                height=round(height, 3),
                material=material,
                color=color,
                district=f"cell_{col}_{row}",
                dynamic_event=event_id,
            )
        )
    return buildings


def _build_nodes(path: Path, roads: set[Road]) -> None:
    used = {node for road in roads for node in road}
    root = ET.Element("nodes")
    for col, row in sorted(used):
        degree = sum((col, row) in road for road in roads)
        ET.SubElement(
            root,
            "node",
            id=_node(col, row),
            x=f"{COORDINATES[col]:.3f}",
            y=f"{COORDINATES[row]:.3f}",
            type="traffic_light" if degree >= 4 else "priority",
        )
    _write_xml(root, path)


def _build_edges(path: Path, roads: set[Road]) -> None:
    root = ET.Element("edges")
    for left, right in sorted(roads):
        arterial = left[0] in {1, 4} and right[0] in {1, 4}
        arterial = arterial or (left[1] in {1, 4} and right[1] in {1, 4})
        for start, stop in ((left, right), (right, left)):
            ET.SubElement(
                root,
                "edge",
                id=_edge(start, stop),
                **{
                    "from": _node(*start),
                    "to": _node(*stop),
                    "numLanes": "1",
                    "speed": "11.11" if arterial else "8.33",
                    "width": "3.2",
                    "priority": "4" if arterial else "2",
                },
            )
    _write_xml(root, path)


def _build_routes(path: Path) -> None:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="passenger",
        vClass="passenger",
        accel="1.5",
        decel="3.5",
        sigma="0.25",
        length="4.8",
        minGap="2.5",
        tau="1.0",
        maxSpeed="11.11",
        speedFactor="0.95",
        speedDev="0.08",
    )
    _write_xml(root, path)


def _build_polygons(path: Path, buildings: list[Building]) -> None:
    root = ET.Element("additional")
    for building in buildings:
        poly = ET.SubElement(
            root,
            "poly",
            id=building.building_id,
            type="building",
            color=building.color,
            fill="1",
            layer="4",
            shape=building.shape,
        )
        ET.SubElement(poly, "param", key="height", value=f"{building.height:.3f}")
        ET.SubElement(poly, "param", key="material", value=building.material)
        ET.SubElement(poly, "param", key="district", value=building.district)
        if building.dynamic_event is not None:
            ET.SubElement(poly, "param", key="dynamic_event", value=building.dynamic_event)
    _write_xml(root, path)


def _dynamic_schedule(spec: SourceSpec, buildings: list[Building]) -> dict[str, object]:
    dynamic = [building for building in buildings if building.dynamic_event]
    if len(dynamic) != 2:
        raise RuntimeError(f"{spec.map_id}: expected two dynamic buildings")
    periods = (37 + spec.seed % 11, 61 + spec.seed % 17)
    events: list[dict[str, object]] = []
    for index, (building, period) in enumerate(zip(dynamic, periods)):
        x0, y0, x1, y1 = building.bounds
        events.append(
            {
                "id": building.dynamic_event,
                "kind": "periodic_building",
                "zone": 0,
                "center": [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
                "size": [x1 - x0, y1 - y0, building.height],
                "material": building.material,
                "placement": "existing_building",
                "replaces_static_block": True,
                "period_steps": int(period),
                "active_steps": int(round(period * (0.35 + 0.1 * index))),
                "phase_steps": int((spec.seed // (index + 1)) % period),
            }
        )
    return {
        "seed": spec.seed,
        "sim_steps": 1000,
        "description": "Source-only periodic buildings for transferable policy pretraining.",
        "events": events,
    }


def _gate_manifest(roads: set[Road]) -> dict[str, dict[str, str]]:
    center = 0.5 * (len(COORDINATES) - 1)
    gates: dict[str, dict[str, str]] = {}
    for col, row in sorted(_border_nodes()):
        node = (col, row)
        neighbors = [right if left == node else left for left, right in roads if node in (left, right)]
        if not neighbors:
            raise RuntimeError(f"border node {node} is disconnected")
        neighbor = min(
            neighbors,
            key=lambda value: (value[0] - center) ** 2 + (value[1] - center) ** 2,
        )
        gate_id = f"G_{col}_{row}"
        gates[gate_id] = {
            "entry": _edge(node, neighbor),
            "exit": _edge(neighbor, node),
            "node": _node(*node),
        }
    return gates


def _build_config(path: Path, map_id: str) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=f"{map_id}.net.xml")
    ET.SubElement(inputs, "route-files", value=f"{map_id}.rou.xml")
    ET.SubElement(inputs, "additional-files", value=f"{map_id}.poly.xml")
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value="1000")
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def _render(path: Path, spec: SourceSpec, roads: set[Road], buildings: list[Building]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_facecolor("#eef0eb")
    for left, right in roads:
        ax.plot(
            [COORDINATES[left[0]], COORDINATES[right[0]]],
            [COORDINATES[left[1]], COORDINATES[right[1]]],
            color="#f7f5ef",
            path_effects=[],
            linewidth=5,
            solid_capstyle="round",
            zorder=2,
        )
    for building in buildings:
        x0, y0, x1, y1 = building.bounds
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#d9903d" if building.dynamic_event else "#777e89",
                edgecolor="#34383d",
                hatch="///" if building.dynamic_event else None,
                linewidth=0.7,
                zorder=3,
            )
        )
    ax.set(xlim=(-5, 185), ylim=(-5, 185), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax.set_title(f"{spec.map_id} ({spec.split})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_map(root: Path, spec: SourceSpec, *, netconvert: str) -> dict[str, object]:
    scenario = root / spec.map_id
    scenario.mkdir(parents=True, exist_ok=True)
    roads = _make_roads(spec)
    buildings = _buildings(spec)
    nodes = scenario / f"{spec.map_id}.nod.xml"
    edges = scenario / f"{spec.map_id}.edg.xml"
    routes = scenario / f"{spec.map_id}.rou.xml"
    polygons = scenario / f"{spec.map_id}.poly.xml"
    network = scenario / f"{spec.map_id}.net.xml"
    config = scenario / f"{spec.map_id}.sumocfg"
    dynamic_path = scenario / f"{spec.map_id}_dynamic.json"
    overview = scenario / f"{spec.map_id}_overview.png"
    _build_nodes(nodes, roads)
    _build_edges(edges, roads)
    _build_routes(routes)
    _build_polygons(polygons, buildings)
    _build_config(config, spec.map_id)
    schedule = _dynamic_schedule(spec, buildings)
    dynamic_path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            netconvert,
            "--node-files", str(nodes),
            "--edge-files", str(edges),
            "--output-file", str(network),
            "--junctions.join", "true",
            "--roundabouts.guess", "true",
        ],
        check=True,
    )
    _render(overview, spec, roads, buildings)
    manifest = {
        **asdict(spec),
        "map_size": 180.0,
        "num_zones": 1,
        "network": str(network.resolve()),
        "polygons": str(polygons.resolve()),
        "dynamic_map": str(dynamic_path.resolve()),
        "overview": str(overview.resolve()),
        "road_count": len(roads),
        "building_count": len(buildings),
        "gates": _gate_manifest(roads),
        "buildings": [asdict(building) for building in buildings],
        "dynamic_schedule": schedule,
    }
    manifest_path = scenario / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--netconvert", default="netconvert")
    parser.add_argument(
        "--map-id",
        action="append",
        default=[],
        help="Build only this map id; repeat for multiple maps.",
    )
    args = parser.parse_args()
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [
        spec for spec in SPECS
        if not args.map_id or spec.map_id in set(args.map_id)
    ]
    if not selected:
        raise ValueError(f"No matching maps for {args.map_id}")
    maps = [build_map(root, spec, netconvert=str(args.netconvert)) for spec in selected]
    manifest = {
        "format": "cross_map_policy_pretraining_sources_v1",
        "deployment_map_excluded": "single_zone_urban_150",
        "coordinate_normalization": "divide x/y by each source map size",
        "sim_steps": 1000,
        "maps": maps,
    }
    path = root / (
        "manifest.json"
        if not args.map_id
        else "manifest_selected.json"
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
