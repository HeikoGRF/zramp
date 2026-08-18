#!/usr/bin/env python3
"""Build a 150 m urban map with two radio communities and two crossings."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from SUMO import build_single_zone_urban_150 as base


ROOT = Path(__file__).resolve().parents[1]
NAME = "single_zone_urban_150_communities"
DEFAULT_OUT = ROOT / "SUMO" / NAME


def _buildings() -> list[base.Building]:
    buildings = [
        base.Building("b_dynamic_southwest", (6.0, 7.0, 29.0, 36.0), 9.0, "151,139,125", "south"),
        base.Building("b_south_central", (44.0, 8.0, 68.0, 37.0), 11.0, "151,139,125", "south"),
        base.Building("b_south_east", (122.0, 7.0, 144.0, 37.0), 10.0, "151,139,125", "south"),
        base.Building("b_north_west", (6.0, 113.0, 29.0, 143.0), 12.0, "119,126,137", "north"),
        base.Building("b_north_central", (82.0, 112.0, 108.0, 142.0), 14.0, "119,126,137", "north"),
        base.Building("b_dynamic_northeast", (122.0, 114.0, 144.0, 143.0), 11.0, "119,126,137", "north"),
    ]
    # A railway/depot corridor is a plausible urban radio separator.  Two
    # 12 m gaps centered at x=35 and x=115 carry the only crossing roads.
    segments = ((0.0, 29.0), (41.0, 109.0), (121.0, 150.0))
    for side, y0, y1 in (("south", 62.0, 66.0), ("north", 84.0, 88.0)):
        for index, (x0, x1) in enumerate(segments):
            buildings.append(
                base.Building(
                    f"b_rail_noise_wall_{side}_{index}",
                    (x0, y0, x1, y1),
                    6.0,
                    "76,81,87",
                    "barrier",
                )
            )
    buildings.append(
        base.Building(
            "b_central_rail_depot",
            (43.0, 67.0, 107.0, 83.0),
            22.0,
            "76,81,87",
            "barrier",
        )
    )
    return buildings


def _build_polygons(path: Path, buildings: list[base.Building]) -> None:
    root = ET.Element("additional")
    ET.SubElement(
        root,
        "poly",
        id="rail_corridor_ground",
        type="infrastructure",
        color="198,188,166",
        fill="1",
        layer="1",
        shape="0,58 150,58 150,92 0,92",
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
        material = (
            "itu_metal"
            if building.building_id.startswith("b_rail_noise_wall")
            else "itu_concrete"
        )
        ET.SubElement(poly, "param", key="material", value=material)
        ET.SubElement(poly, "param", key="district", value=building.district)
        if building.building_id in base.DYNAMIC_BUILDINGS:
            ET.SubElement(
                poly,
                "param",
                key="dynamic_event",
                value=base.DYNAMIC_BUILDINGS[building.building_id],
            )
    base._write_xml(root, path)


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
    base._write_xml(root, path)


def _validate(buildings: list[base.Building]) -> None:
    barrier = [item for item in buildings if item.district == "barrier"]
    if len(barrier) != 7:
        raise ValueError("community map must contain six rail walls and one depot")
    for crossing_x in (35.0, 115.0):
        if any(item.bounds[0] <= crossing_x <= item.bounds[2] for item in barrier):
            raise ValueError(f"crossing x={crossing_x} is obstructed")


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
    _validate(buildings)
    base._build_nodes(nodes)
    base._build_edges(edges)
    base._build_routes(routes)
    _build_polygons(polygons, buildings)
    base._build_dynamic_schedule(dynamic, buildings)
    _build_config(config)
    subprocess.run(
        [
            netconvert,
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(network),
            "--junctions.join",
            "true",
            "--roundabouts.guess",
            "true",
        ],
        check=True,
    )
    base._render_overview(overview, buildings)
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
