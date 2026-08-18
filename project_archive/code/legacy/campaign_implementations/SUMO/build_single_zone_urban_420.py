#!/usr/bin/env python3
"""Build the 420 m single-zone urban bottleneck experiment map."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "SUMO" / "single_zone_urban_420"
NAME = "single_zone_urban_420"
X = (0.0, 65.0, 135.0, 210.0, 285.0, 355.0, 420.0)
Y = (0.0, 70.0, 140.0, 190.0, 230.0, 285.0, 350.0, 420.0)


@dataclass(frozen=True)
class Building:
    building_id: str
    bounds: tuple[float, float, float, float]
    height: float
    material: str
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
    # Two internally well-connected districts.
    for row in (*range(0, 4), *range(4, 8)):
        for col in range(6):
            pairs.append(((col, row), (col + 1, row)))
    for col in range(7):
        for row in (*range(0, 3), *range(4, 7)):
            pairs.append(((col, row), (col, row + 1)))
    # Only two streets cross the railway/warehouse separation band.
    pairs.extend([((1, 3), (1, 4)), ((5, 3), (5, 4))])
    return pairs


def _build_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    for col, x in enumerate(X):
        for row, y in enumerate(Y):
            node_type = "traffic_light" if (col, row) in {
                (1, 2), (5, 2), (1, 5), (5, 5)
            } else "priority"
            ET.SubElement(
                root,
                "node",
                id=_node(col, row),
                x=f"{x:.1f}",
                y=f"{y:.1f}",
                type=node_type,
            )
    _write_xml(root, path)


def _build_edges(path: Path) -> None:
    root = ET.Element("edges")
    arterial_rows = {2, 5}
    bridge_cols = {1, 5}
    for a, b in _road_pairs():
        is_arterial = (
            (a[1] == b[1] and a[1] in arterial_rows)
            or (a[0] == b[0] and a[0] in bridge_cols and {a[1], b[1]} == {3, 4})
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
                    "speed": "11.11" if is_arterial else "8.33",
                    "width": "3.20",
                    "priority": "4" if is_arterial else "2",
                },
            )
    _write_xml(root, path)


def _cell_buildings(
    *, district: str, row_start: int, row_stop: int
) -> list[Building]:
    buildings: list[Building] = []
    colors = {
        "itu_concrete": "119,126,137",
        "itu_stone": "151,139,125",
        "itu_brick": "157,105,89",
        "itu_glass": "102,135,151",
    }
    materials = ("itu_brick", "itu_concrete", "itu_stone", "itu_glass")
    # Open cells form small parks/plazas and keep the scene learnable.
    open_cells = {(4, 0), (2, 1), (0, 5), (3, 6)}
    split_cells = {(0, 0), (3, 0), (5, 1), (1, 2), (2, 4), (4, 4), (0, 6), (5, 6)}
    for row in range(row_start, row_stop):
        for col in range(6):
            if (col, row) in open_cells:
                continue
            x0, x1 = X[col] + 10.0, X[col + 1] - 10.0
            y0, y1 = Y[row] + 10.0, Y[row + 1] - 10.0
            material = materials[(2 * col + row) % len(materials)]
            base_height = 11.0 if district == "south" else 17.0
            height = base_height + float((5 * col + 3 * row) % 9)
            if (col, row) in split_cells:
                gap = 7.0
                middle = 0.5 * (x0 + x1)
                bounds = ((x0, y0, middle - gap / 2, y1), (middle + gap / 2, y0, x1, y1))
            else:
                # Offset several footprints to avoid a perfectly uniform grid.
                dx = float(((col + row) % 3) - 1) * 2.0
                dy = float(((2 * col + row) % 3) - 1) * 1.5
                bounds = ((x0 + dx, y0 + dy, x1 - dx, y1 - dy),)
            for part, rectangle in enumerate(bounds):
                buildings.append(
                    Building(
                        building_id=f"b_{district}_{col}_{row}_{part}",
                        bounds=rectangle,
                        height=height + 2.0 * part,
                        material=material,
                        color=colors[material],
                        district=district,
                    )
                )
    return buildings


def _buildings() -> list[Building]:
    buildings = _cell_buildings(district="south", row_start=0, row_stop=3)
    buildings.extend(_cell_buildings(district="north", row_start=4, row_stop=7))
    # Railway-side warehouses create radio separation, leaving two crossings.
    barrier_color = "83,88,94"
    for idx, (x0, x1, height) in enumerate(
        ((8.0, 49.0, 13.0), (81.0, 339.0, 18.0), (371.0, 412.0, 13.0))
    ):
        buildings.append(
            Building(
                building_id=f"b_barrier_{idx}",
                bounds=(x0, 197.0, x1, 223.0),
                height=height,
                material="itu_concrete",
                color=barrier_color,
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
    # Visual context only; radio geometry comes from the explicit buildings.
    ET.SubElement(
        root,
        "poly",
        id="railway_ground",
        type="infrastructure",
        color="198,188,166",
        fill="1",
        layer="1",
        shape="0,190 420,190 420,230 0,230",
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
        ET.SubElement(poly, "param", key="material", value=building.material)
        ET.SubElement(poly, "param", key="district", value=building.district)
    _write_xml(root, path)


def _build_dynamic(path: Path) -> None:
    schedule = {
        "seed": 1,
        "sim_steps": 1000,
        "description": "Seeded asynchronous courtyard and roadside blockers in one AZ.",
        "events": [
            {
                "id": "south_market_delivery",
                "kind": "temporary_delivery_vehicle",
                "zone": 0,
                "center": [247.0, 111.0],
                "size": [9.0, 4.0, 3.5],
                "material": "itu_metal",
                "placement": "roadside_loading_bay",
                "period_steps": 47,
                "active_steps": 13,
                "phase_steps": 5,
            },
            {
                "id": "north_courtyard_service",
                "kind": "temporary_service_vehicle",
                "zone": 0,
                "center": [101.0, 317.0],
                "size": [8.0, 4.0, 3.2],
                "material": "itu_metal",
                "placement": "courtyard",
                "period_steps": 71,
                "active_steps": 19,
                "phase_steps": 17,
            },
            {
                "id": "east_bridge_maintenance",
                "kind": "temporary_maintenance_equipment",
                "zone": 0,
                "center": [373.0, 242.0],
                "size": [7.0, 5.0, 4.0],
                "material": "itu_metal",
                "placement": "roadside_work_area",
                "period_steps": 109,
                "active_steps": 31,
                "phase_steps": 41,
            },
        ],
    }
    path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")


def _build_config(path: Path) -> None:
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=f"{NAME}.net.xml")
    ET.SubElement(inp, "route-files", value=f"{NAME}.rou.xml")
    ET.SubElement(inp, "additional-files", value=f"{NAME}.poly.xml")
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


def _render_overview(
    path: Path,
    buildings: list[Building],
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.set_facecolor("#eef0eb")
    ax.add_patch(Rectangle((0, 190), 420, 40, color="#c6bca6", zorder=0))
    for y in (204, 216):
        ax.plot([0, 420], [y, y], color="#736c61", lw=1.2, ls=(0, (6, 5)), zorder=1)
    for a, b in pairs:
        x0, y0 = X[a[0]], Y[a[1]]
        x1, y1 = X[b[0]], Y[b[1]]
        arterial = (a[1] == b[1] and a[1] in {2, 5}) or ({a[1], b[1]} == {3, 4})
        ax.plot([x0, x1], [y0, y1], color="#555b62", lw=8 if arterial else 6, solid_capstyle="round", zorder=2)
        ax.plot([x0, x1], [y0, y1], color="#f7f5ef", lw=4.5 if arterial else 3.5, solid_capstyle="round", zorder=3)
    for building in buildings:
        x0, y0, x1, y1 = building.bounds
        color = "#53585e" if building.district == "barrier" else {
            "itu_concrete": "#777e89",
            "itu_stone": "#978b7d",
            "itu_brick": "#9d6959",
            "itu_glass": "#668797",
        }[building.material]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="#34383d", lw=0.7, zorder=4))
    gates = {
        "S1": (65, 0), "S2": (355, 0), "N1": (65, 420), "N2": (355, 420),
        "W1": (0, 70), "W2": (0, 350), "E1": (420, 70), "E2": (420, 350),
    }
    for label, (x, y) in gates.items():
        ax.scatter([x], [y], s=38, color="#e2a33a", edgecolor="#5d431c", zorder=6)
        ax.annotate(label, (x, y), xytext=(0, 8 if y < 210 else -14), textcoords="offset points", ha="center", fontsize=8, weight="bold", zorder=7)
    for x, label in ((65, "West crossing"), (355, "East crossing")):
        ax.annotate(label, (x, 210), xytext=(0, 0), textcoords="offset points", ha="center", va="center", rotation=90, fontsize=8, color="#21252a", weight="bold", zorder=7)
    ax.text(247, 385, "NORTH DISTRICT", ha="center", fontsize=10, weight="bold", color="#3f484e", zorder=8)
    ax.text(320, 35, "SOUTH DISTRICT", ha="center", fontsize=10, weight="bold", color="#3f484e", zorder=8)
    ax.text(210, 211, "RAIL / WAREHOUSE SEPARATION", ha="center", fontsize=8, weight="bold", color="#f5f5f2", zorder=8)
    ax.set_xlim(-15, 435)
    ax.set_ylim(-15, 435)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Single-zone urban bottleneck map (420 m × 420 m)", pad=14, weight="bold")
    ax.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor="#9d6959", edgecolor="#34383d", label="Urban buildings"),
            Patch(facecolor="#53585e", edgecolor="#34383d", label="Radio-separating warehouses"),
            Line2D([0], [0], color="#f7f5ef", lw=4, marker="o", markerfacecolor="#e2a33a", markeredgecolor="#5d431c", label="Roads and boundary gates"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
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
    config = out_dir / f"{NAME}.sumocfg"
    network = out_dir / f"{NAME}.net.xml"
    dynamic = out_dir / f"{NAME}_dynamic.json"
    overview = out_dir / f"{NAME}_overview.png"
    buildings = _buildings()
    pairs = _road_pairs()
    _build_nodes(nodes)
    _build_edges(edges)
    _build_routes(routes)
    _build_polygons(polygons, buildings)
    _build_dynamic(dynamic)
    _build_config(config)
    subprocess.run(
        [netconvert, "--node-files", str(nodes), "--edge-files", str(edges), "--output-file", str(network), "--junctions.join", "true", "--roundabouts.guess", "true"],
        check=True,
    )
    _render_overview(overview, buildings, pairs)
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
