#!/usr/bin/env python3
"""Build the controlled 300 m, four-zone SUMO experiment."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "SUMO" / "controlled_4zone_300"
GRID = (0.0, 75.0, 150.0, 225.0, 300.0)


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _node(col: int, row: int) -> str:
    return f"N{col}{row}"


def _edge(a: tuple[int, int], b: tuple[int, int]) -> str:
    return f"{_node(*a)}_{_node(*b)}"


def _build_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    for col, x in enumerate(GRID):
        for row, y in enumerate(GRID):
            ET.SubElement(root, "node", id=_node(col, row), x=f"{x:.1f}", y=f"{y:.1f}", type="priority")
    _write_xml(root, path)


def _build_edges(path: Path) -> None:
    root = ET.Element("edges")
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row in range(5):
        for col in range(4):
            pairs.append(((col, row), (col + 1, row)))
    for col in range(5):
        for row in range(4):
            pairs.append(((col, row), (col, row + 1)))
    for a, b in pairs:
        for start, stop in ((a, b), (b, a)):
            ET.SubElement(
                root,
                "edge",
                id=_edge(start, stop),
                **{"from": _node(*start), "to": _node(*stop), "numLanes": "1", "speed": "8.33", "width": "3.20", "priority": "2"},
            )
    _write_xml(root, path)


def _perimeter_cycle() -> list[tuple[int, int]]:
    return (
        [(col, 0) for col in range(5)]
        + [(4, row) for row in range(1, 5)]
        + [(col, 4) for col in range(3, -1, -1)]
        + [(0, row) for row in range(3, 0, -1)]
    )


def _build_routes(path: Path, num_vehicles: int) -> None:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="passenger",
        vClass="passenger",
        accel="1.3",
        decel="3.5",
        sigma="0.25",
        length="4.8",
        minGap="2.8",
        tau="1.1",
        maxSpeed="7.0",
        speedFactor="0.92",
        speedDev="0.06",
    )
    cycle = _perimeter_cycle()
    for idx in range(int(num_vehicles)):
        offset = idx % len(cycle)
        rotated = cycle[offset:] + cycle[:offset]
        loop_edges = [_edge(rotated[i], rotated[(i + 1) % len(rotated)]) for i in range(len(rotated))]
        vehicle = ET.SubElement(
            root,
            "vehicle",
            id=f"veh_{idx:03d}",
            type="passenger",
            depart="0",
            departLane="best",
            departPos="random_free",
            departSpeed="0",
        )
        ET.SubElement(vehicle, "route", edges=" ".join(loop_edges * 4))
    _write_xml(root, path)


def _zone_profile(cx: float) -> tuple[float, str]:
    return (28.0, "151,165,143") if cx < 150.0 else (54.0, "103,109,116")


def _build_polygons(path: Path) -> None:
    root = ET.Element("additional")
    dynamic_centers = {(112.5, 187.5), (187.5, 187.5)}
    block_idx = 0
    for ix in range(4):
        for iy in range(4):
            cx = 37.5 + 75.0 * ix
            cy = 37.5 + 75.0 * iy
            if (cx, cy) in dynamic_centers:
                block_idx += 1
                continue
            size, color = _zone_profile(cx)
            half = 0.5 * size
            shape = " ".join(
                f"{x:.1f},{y:.1f}"
                for x, y in ((cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half))
            )
            ET.SubElement(root, "poly", id=f"building_{block_idx}", type="building", color=color, fill="1", layer="4", shape=shape)
            block_idx += 1
    _write_xml(root, path)


def _build_config(path: Path) -> None:
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value="controlled_4zone_300.net.xml")
    ET.SubElement(inp, "route-files", value="controlled_4zone_300.rou.xml")
    ET.SubElement(inp, "additional-files", value="controlled_4zone_300.poly.xml")
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


def build(out_dir: Path, *, netconvert: str, num_vehicles: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = out_dir / "controlled_4zone_300.nod.xml"
    edges = out_dir / "controlled_4zone_300.edg.xml"
    routes = out_dir / "controlled_4zone_300.rou.xml"
    polygons = out_dir / "controlled_4zone_300.poly.xml"
    config = out_dir / "controlled_4zone_300.sumocfg"
    network = out_dir / "controlled_4zone_300.net.xml"
    _build_nodes(nodes)
    _build_edges(edges)
    _build_routes(routes, num_vehicles)
    _build_polygons(polygons)
    _build_config(config)
    subprocess.run(
        [netconvert, "--node-files", str(nodes), "--edge-files", str(edges), "--output-file", str(network), "--junctions.join", "true", "--roundabouts.guess", "true"],
        check=True,
    )
    print(network)
    return network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--netconvert", default="netconvert")
    parser.add_argument("--num-vehicles", type=int, default=20)
    args = parser.parse_args()
    build(args.out_dir.resolve(), netconvert=str(args.netconvert), num_vehicles=args.num_vehicles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
