#!/usr/bin/env python3
"""Build the compact 150 m single-zone urban bottleneck map."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "SUMO" / "single_zone_urban_150"
NAME = "single_zone_urban_150"
X = (0.0, 35.0, 75.0, 115.0, 150.0)
Y = (0.0, 45.0, 65.0, 85.0, 105.0, 150.0)
DYNAMIC_BUILDINGS = {
    "b_dynamic_southwest": "southwest_modular_building",
    "b_dynamic_northeast": "northeast_modular_building",
}


@dataclass(frozen=True)
class Building:
    building_id: str
    bounds: tuple[float, float, float, float]
    height: float
    color: str
    district: str

    @property
    def shape(self) -> str:
        x0, y0, x1, y1 = self.bounds
        return " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        )


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _node(col: int, row: int) -> str:
    return f"N{col}_{row}"


def _edge(a: tuple[int, int], b: tuple[int, int]) -> str:
    return f"{_node(*a)}__{_node(*b)}"


def _road_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # Four main east-west streets form two deliberately coarse urban
    # districts. The wide middle band is not another dense street grid.
    for row in (0, 1, 4, 5):
        for col in range(4):
            pairs.append(((col, row), (col + 1, row)))
    for col in range(5):
        pairs.extend([((col, 0), (col, 1)), ((col, 4), (col, 5))])
    # Only these two continuous streets cross the central blocker.
    for col in (1, 3):
        pairs.extend(
            [((col, 1), (col, 2)), ((col, 2), (col, 3)), ((col, 3), (col, 4))]
        )
    return pairs


def _build_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    signals = {(1, 1), (3, 1), (1, 4), (3, 4)}
    used_nodes = {node for pair in _road_pairs() for node in pair}
    for col, row in sorted(used_nodes):
        ET.SubElement(
            root,
            "node",
            id=_node(col, row),
            x=f"{X[col]:.1f}",
            y=f"{Y[row]:.1f}",
            type="traffic_light" if (col, row) in signals else "priority",
        )
    _write_xml(root, path)


def _build_edges(path: Path) -> None:
    root = ET.Element("edges")
    for a, b in _road_pairs():
        arterial = (
            (a[1] == b[1] and a[1] in {1, 4})
            or ({a[1], b[1]} == {2, 3})
        )
        for start, stop in ((a, b), (b, a)):
            ET.SubElement(
                root,
                "edge",
                id=_edge(start, stop),
                **{
                    "from": _node(*start),
                    "to": _node(*stop),
                    "numLanes": "1",
                    "speed": "11.11" if arterial else "8.33",
                    "width": "3.20",
                    "priority": "4" if arterial else "2",
                },
            )
    _write_xml(root, path)


def _buildings() -> list[Building]:
    # Six coarse city blocks rather than shrinking the building density of
    # the former 220 m map. Two blocks are periodic dynamic obstacles.
    buildings = [
        Building("b_dynamic_southwest", (6.0, 7.0, 29.0, 36.0), 9.0, "151,139,125", "south"),
        Building("b_south_central", (44.0, 8.0, 68.0, 37.0), 11.0, "151,139,125", "south"),
        Building("b_south_east", (122.0, 7.0, 144.0, 37.0), 10.0, "151,139,125", "south"),
        Building("b_north_west", (6.0, 113.0, 29.0, 143.0), 12.0, "119,126,137", "north"),
        Building("b_north_central", (82.0, 112.0, 108.0, 142.0), 14.0, "119,126,137", "north"),
        Building("b_dynamic_northeast", (122.0, 114.0, 144.0, 143.0), 11.0, "119,126,137", "north"),
    ]
    # A single logical barrier is represented by three warehouse segments.
    # Its two wide openings align with the only cross-district roads.
    for index, (x0, x1, height) in enumerate(
        ((3.0, 22.0, 11.0), (48.0, 102.0, 16.0), (128.0, 147.0, 11.0))
    ):
        buildings.append(
            Building(
                building_id=f"b_central_blocker_{index}",
                bounds=(x0, 68.0, x1, 82.0),
                height=height,
                color="76,81,87",
                district="barrier",
            )
        )
    return buildings


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
    ET.SubElement(
        root,
        "poly",
        id="central_ground",
        type="infrastructure",
        color="198,188,166",
        fill="1",
        layer="1",
        shape="0,65 150,65 150,85 0,85",
    )
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
        ET.SubElement(poly, "param", key="height", value=f"{building.height:.1f}")
        ET.SubElement(poly, "param", key="material", value="itu_concrete")
        ET.SubElement(poly, "param", key="district", value=building.district)
        if building.building_id in DYNAMIC_BUILDINGS:
            ET.SubElement(poly, "param", key="dynamic_event", value=DYNAMIC_BUILDINGS[building.building_id])
    _write_xml(root, path)


def _build_config(path: Path) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=f"{NAME}.net.xml")
    ET.SubElement(inputs, "route-files", value=f"{NAME}.rou.xml")
    ET.SubElement(inputs, "additional-files", value=f"{NAME}.poly.xml")
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value="2000")
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="true")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def _build_dynamic_schedule(path: Path, buildings: list[Building]) -> None:
    by_id = {building.building_id: building for building in buildings}
    southwest = by_id["b_dynamic_southwest"]
    northeast = by_id["b_dynamic_northeast"]

    def event(
        *, event_id: str, building: Building, period: int, active: int, phase: int,
        description: str,
    ) -> dict[str, object]:
        x0, y0, x1, y1 = building.bounds
        return {
            "id": event_id,
            "kind": "periodic_building",
            "zone": 0,
            "center": [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
            "size": [x1 - x0, y1 - y0, building.height],
            "material": "itu_concrete",
            "placement": "existing_building",
            "replaces_static_block": True,
            "period_steps": period,
            "active_steps": active,
            "phase_steps": phase,
            "description": description,
        }

    schedule = {
        "seed": 1,
        "sim_steps": 1000,
        "description": "Two separated periodic buildings in the 150 m single-AZ bottleneck map.",
        "events": [
            event(
                event_id="southwest_modular_building",
                building=southwest,
                period=43,
                active=19,
                phase=5,
                description="Southwest building; faster appearance cycle.",
            ),
            event(
                event_id="northeast_modular_building",
                building=northeast,
                period=67,
                active=29,
                phase=17,
                description="Northeast building; slower appearance cycle.",
            ),
        ],
    }
    path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")


def _render_overview(path: Path, buildings: list[Building]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    ax.set_facecolor("#eef0eb")
    ax.add_patch(Rectangle((0, 65), 150, 20, color="#c6bca6", zorder=0))
    for a, b in _road_pairs():
        x0, y0 = X[a[0]], Y[a[1]]
        x1, y1 = X[b[0]], Y[b[1]]
        arterial = (
            (a[1] == b[1] and a[1] in {1, 4})
            or ({a[1], b[1]} == {2, 3})
        )
        ax.plot(
            [x0, x1], [y0, y1], color="#555b62",
            lw=8 if arterial else 6, solid_capstyle="round", zorder=2,
        )
        ax.plot(
            [x0, x1], [y0, y1], color="#f7f5ef",
            lw=4.5 if arterial else 3.5, solid_capstyle="round", zorder=3,
        )
    for building in buildings:
        x0, y0, x1, y1 = building.bounds
        dynamic = building.building_id in DYNAMIC_BUILDINGS
        color = (
            "#4c5157" if building.district == "barrier"
            else "#978b7d" if building.district == "south"
            else "#777e89"
        )
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#d9903d" if dynamic else color,
                edgecolor="#8b3f18" if dynamic else "#34383d",
                hatch="///" if dynamic else None,
                lw=1.5 if dynamic else 0.7,
                zorder=4,
            )
        )
    gates = {
        "S1": (35, 0), "S2": (115, 0), "N1": (35, 150), "N2": (115, 150),
        "W1": (0, 45), "W2": (0, 105), "E1": (150, 45), "E2": (150, 105),
    }
    for label, (x, y) in gates.items():
        ax.scatter([x], [y], s=38, color="#e2a33a", edgecolor="#5d431c", zorder=6)
        ax.annotate(
            label, (x, y), xytext=(0, 8 if y < 75 else -14),
            textcoords="offset points", ha="center", fontsize=8,
            weight="bold", zorder=7,
        )
    for x, label in ((35, "West passage"), (115, "East passage")):
        ax.annotate(
            label, (x, 75), ha="center", va="center", rotation=90,
            fontsize=8, weight="bold", color="#183c5a", zorder=7,
        )
    ax.text(75, 75, "CENTRAL RADIO BLOCKER", ha="center", va="center", fontsize=8, weight="bold", color="#f5f5f2", zorder=8)
    ax.text(55, 99, "NORTH DISTRICT", ha="center", fontsize=9, weight="bold", color="#3f484e", zorder=8)
    ax.text(95, 51, "SOUTH DISTRICT", ha="center", fontsize=9, weight="bold", color="#3f484e", zorder=8)
    ax.text(
        15, 16, "DYNAMIC\nperiod 43 / active 19",
        ha="center", va="center", fontsize=7, color="#54250d",
        weight="bold", zorder=9,
    )
    ax.text(
        135, 134, "DYNAMIC\nperiod 67 / active 29",
        ha="center", va="center", fontsize=7, color="#54250d",
        weight="bold", zorder=9,
    )
    ax.set_xlim(-6, 156)
    ax.set_ylim(-6, 156)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Sparse single-zone urban map (150 m × 150 m)", pad=12, weight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#978b7d", edgecolor="#34383d", label="Static urban buildings"),
            Patch(facecolor="#4c5157", edgecolor="#34383d", label="Central radio blocker"),
            Patch(facecolor="#d9903d", edgecolor="#8b3f18", hatch="///", label="Periodic buildings"),
            Line2D([0], [0], color="#f7f5ef", lw=4, marker="o", markerfacecolor="#e2a33a", markeredgecolor="#5d431c", label="Roads and gates"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _validate_design(buildings: list[Building]) -> None:
    crossing_pairs = [
        pair for pair in _road_pairs() if {pair[0][1], pair[1][1]} == {2, 3}
    ]
    if crossing_pairs != [((1, 2), (1, 3)), ((3, 2), (3, 3))]:
        raise ValueError(f"unexpected central crossings: {crossing_pairs}")
    by_id = {building.building_id: building for building in buildings}
    if set(DYNAMIC_BUILDINGS).difference(by_id):
        raise ValueError("dynamic replacement buildings are missing")
    if not any(building.building_id == "b_central_blocker_1" for building in buildings):
        raise ValueError("central blocker is missing")
    if len([building for building in buildings if building.district != "barrier"]) != 6:
        raise ValueError("compact map should contain exactly six district buildings")


def build(out_dir: Path, *, netconvert: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = out_dir / f"{NAME}.nod.xml"
    edges = out_dir / f"{NAME}.edg.xml"
    routes = out_dir / f"{NAME}.rou.xml"
    polygons = out_dir / f"{NAME}.poly.xml"
    dynamic = out_dir / f"{NAME}_dynamic.json"
    config = out_dir / f"{NAME}.sumocfg"
    network = out_dir / f"{NAME}.net.xml"
    overview = out_dir / f"{NAME}_overview.png"
    buildings = _buildings()
    _validate_design(buildings)
    _build_nodes(nodes)
    _build_edges(edges)
    _build_routes(routes)
    _build_polygons(polygons, buildings)
    _build_dynamic_schedule(dynamic, buildings)
    _build_config(config)
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
    _render_overview(overview, buildings)
    print(network)
    print(overview)
    return network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--netconvert", default="netconvert")
    args = parser.parse_args()
    build(args.out_dir.resolve(), netconvert=str(args.netconvert))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
