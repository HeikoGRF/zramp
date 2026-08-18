#!/usr/bin/env python3
"""Build the compact 220 m single-zone urban bottleneck map."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "SUMO" / "single_zone_urban_220"
NAME = "single_zone_urban_220"
X = (0.0, 45.0, 90.0, 130.0, 175.0, 220.0)
Y = (0.0, 50.0, 95.0, 125.0, 170.0, 220.0)


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
    # Complete local grids within the south and north districts.
    for row in (0, 1, 2, 3, 4, 5):
        for col in range(5):
            pairs.append(((col, row), (col + 1, row)))
    for col in range(6):
        for row in (0, 1, 3, 4):
            pairs.append(((col, row), (col, row + 1)))
    # Two crossings are the only connections across the separation band.
    pairs.extend([((1, 2), (1, 3)), ((4, 2), (4, 3))])
    return pairs


def _build_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    signals = {(1, 1), (4, 1), (1, 4), (4, 4)}
    for col, x in enumerate(X):
        for row, y in enumerate(Y):
            ET.SubElement(
                root,
                "node",
                id=_node(col, row),
                x=f"{x:.1f}",
                y=f"{y:.1f}",
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


def _district_buildings(
    district: str, rows: tuple[int, int]
) -> list[Building]:
    buildings: list[Building] = []
    open_cells = {(3, 0), (1, 4)}
    split_cells = {(0, 0), (4, 1), (2, 3), (4, 4)}
    for row in rows:
        for col in range(5):
            if (col, row) in open_cells:
                continue
            x0, x1 = X[col] + 7.0, X[col + 1] - 7.0
            y0, y1 = Y[row] + 7.0, Y[row + 1] - 7.0
            base_height = 9.0 if district == "south" else 12.0
            height = base_height + float((3 * col + 2 * row) % 6)
            color = "151,139,125" if district == "south" else "119,126,137"
            if (col, row) in split_cells:
                gap = 5.0
                middle = 0.5 * (x0 + x1)
                rectangles = (
                    (x0, y0, middle - gap / 2.0, y1),
                    (middle + gap / 2.0, y0, x1, y1),
                )
            else:
                dx = float(((col + row) % 3) - 1)
                dy = 0.75 * float(((2 * col + row) % 3) - 1)
                rectangles = ((x0 + dx, y0 + dy, x1 - dx, y1 - dy),)
            for part, rectangle in enumerate(rectangles):
                buildings.append(
                    Building(
                        building_id=f"b_{district}_{col}_{row}_{part}",
                        bounds=rectangle,
                        height=height + float(part),
                        color=color,
                        district=district,
                    )
                )
    return buildings


def _buildings() -> list[Building]:
    buildings = _district_buildings("south", (0, 1))
    buildings.extend(_district_buildings("north", (3, 4)))
    # Three warehouses block cross-district radio paths except at x=45 and x=175.
    for index, (x0, x1, height) in enumerate(
        ((5.0, 32.0, 11.0), (58.0, 162.0, 14.0), (188.0, 215.0, 11.0))
    ):
        buildings.append(
            Building(
                building_id=f"b_barrier_{index}",
                bounds=(x0, 101.0, x1, 119.0),
                height=height,
                color="83,88,94",
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
        id="railway_ground",
        type="infrastructure",
        color="198,188,166",
        fill="1",
        layer="1",
        shape="0,95 220,95 220,125 0,125",
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


def _build_dynamic_schedule(path: Path) -> None:
    schedule = {
        "seed": 1,
        "sim_steps": 1000,
        "description": "Two separated periodic buildings in the compact single-AZ urban map.",
        "events": [
            {
                "id": "southwest_modular_building",
                "kind": "periodic_building",
                "zone": 0,
                "center": [13.5, 25.0],
                "size": [13.0, 36.0, 9.0],
                "material": "itu_concrete",
                "placement": "existing_building",
                "replaces_static_block": True,
                "period_steps": 47,
                "active_steps": 21,
                "phase_steps": 5,
                "description": "Southwest building b_south_0_0_0; faster appearance cycle.",
            },
            {
                "id": "northeast_modular_building",
                "kind": "periodic_building",
                "zone": 0,
                "center": [206.5, 195.0],
                "size": [13.0, 36.0, 15.0],
                "material": "itu_concrete",
                "placement": "existing_building",
                "replaces_static_block": True,
                "period_steps": 71,
                "active_steps": 31,
                "phase_steps": 17,
                "description": "Northeast building b_north_4_4_1; slower appearance cycle.",
            },
        ],
    }
    path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")


def _render_overview(path: Path, buildings: list[Building]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    ax.set_facecolor("#eef0eb")
    ax.add_patch(Rectangle((0, 95), 220, 30, color="#c6bca6", zorder=0))
    for y in (105, 115):
        ax.plot([0, 220], [y, y], color="#736c61", lw=1.1, ls=(0, (6, 5)), zorder=1)
    for a, b in _road_pairs():
        x0, y0 = X[a[0]], Y[a[1]]
        x1, y1 = X[b[0]], Y[b[1]]
        arterial = (a[1] == b[1] and a[1] in {1, 4}) or ({a[1], b[1]} == {2, 3})
        ax.plot([x0, x1], [y0, y1], color="#555b62", lw=8 if arterial else 6, solid_capstyle="round", zorder=2)
        ax.plot([x0, x1], [y0, y1], color="#f7f5ef", lw=4.5 if arterial else 3.5, solid_capstyle="round", zorder=3)
    for building in buildings:
        x0, y0, x1, y1 = building.bounds
        color = "#53585e" if building.district == "barrier" else (
            "#978b7d" if building.district == "south" else "#777e89"
        )
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="#34383d", lw=0.7, zorder=4))
    gates = {
        "S1": (45, 0), "S2": (175, 0), "N1": (45, 220), "N2": (175, 220),
        "W1": (0, 50), "W2": (0, 170), "E1": (220, 50), "E2": (220, 170),
    }
    for label, (x, y) in gates.items():
        ax.scatter([x], [y], s=38, color="#e2a33a", edgecolor="#5d431c", zorder=6)
        ax.annotate(label, (x, y), xytext=(0, 8 if y < 110 else -14), textcoords="offset points", ha="center", fontsize=8, weight="bold", zorder=7)
    for x, label in ((45, "West crossing"), (175, "East crossing")):
        ax.annotate(label, (x, 110), ha="center", va="center", rotation=90, fontsize=8, weight="bold", zorder=7)
    ax.text(67, 195, "NORTH DISTRICT", ha="center", fontsize=9, weight="bold", color="#3f484e", zorder=8)
    ax.text(152, 25, "SOUTH DISTRICT", ha="center", fontsize=9, weight="bold", color="#3f484e", zorder=8)
    ax.text(110, 110, "RAIL / WAREHOUSE SEPARATION", ha="center", va="center", fontsize=7, weight="bold", color="#f5f5f2", zorder=8)
    ax.set_xlim(-8, 228)
    ax.set_ylim(-8, 228)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Compact single-zone urban map (220 m × 220 m)", pad=12, weight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#978b7d", edgecolor="#34383d", label="Static concrete buildings"),
            Patch(facecolor="#53585e", edgecolor="#34383d", label="Radio-separating warehouses"),
            Line2D([0], [0], color="#f7f5ef", lw=4, marker="o", markerfacecolor="#e2a33a", markeredgecolor="#5d431c", label="Roads and boundary gates"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    _build_nodes(nodes)
    _build_edges(edges)
    _build_routes(routes)
    _build_polygons(polygons, buildings)
    _build_dynamic_schedule(dynamic)
    _build_config(config)
    subprocess.run(
        [netconvert, "--node-files", str(nodes), "--edge-files", str(edges), "--output-file", str(network), "--junctions.join", "true", "--roundabouts.guess", "true"],
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
