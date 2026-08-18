#!/usr/bin/env python3
"""Build role-based routes using every border intersection of the 150 m map."""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "SUMO" / "single_zone_urban_150"
DEFAULT_NET = SCENARIO / "single_zone_urban_150.net.xml"

GATES = {
    "S000": {"entry": "N0_0__N0_1", "exit": "N0_1__N0_0"},
    "S035": {"entry": "N1_0__N1_1", "exit": "N1_1__N1_0"},
    "S075": {"entry": "N2_0__N2_1", "exit": "N2_1__N2_0"},
    "S115": {"entry": "N3_0__N3_1", "exit": "N3_1__N3_0"},
    "S150": {"entry": "N4_0__N4_1", "exit": "N4_1__N4_0"},
    "N000": {"entry": "N0_5__N0_4", "exit": "N0_4__N0_5"},
    "N035": {"entry": "N1_5__N1_4", "exit": "N1_4__N1_5"},
    "N075": {"entry": "N2_5__N2_4", "exit": "N2_4__N2_5"},
    "N115": {"entry": "N3_5__N3_4", "exit": "N3_4__N3_5"},
    "N150": {"entry": "N4_5__N4_4", "exit": "N4_4__N4_5"},
    "W045": {"entry": "N0_1__N1_1", "exit": "N1_1__N0_1"},
    "W105": {"entry": "N0_4__N1_4", "exit": "N1_4__N0_4"},
    "E045": {"entry": "N4_1__N3_1", "exit": "N3_1__N4_1"},
    "E105": {"entry": "N4_4__N3_4", "exit": "N3_4__N4_4"},
}

REGULAR_GATE_PAIRS = (
    ("S035", "N115"),
    ("W045", "E105"),
    ("S115", "N035"),
    ("E045", "W105"),
    ("S000", "N150"),
    ("S150", "N000"),
    ("W045", "E045"),
    ("W105", "E105"),
)


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _shortest_route(net, entry_gate: str, exit_gate: str) -> list[str]:
    entry = net.getEdge(GATES[entry_gate]["entry"])
    exit_edge = net.getEdge(GATES[exit_gate]["exit"])
    result = net.getShortestPath(entry, exit_edge)
    if result is None or not result[0]:
        raise ValueError(f"no route from {entry_gate} to {exit_gate}")
    return [edge.getID() for edge in result[0]]


def _add_vehicle_type(root: ET.Element) -> None:
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


def _build_config(
    path: Path, *, net: Path, routes: Path, polygons: Path
) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=str(net.resolve()))
    ET.SubElement(inputs, "route-files", value=str(routes.resolve()))
    ET.SubElement(
        inputs, "additional-files", value=str(polygons.resolve())
    )
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value="1200")
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def build(
    *,
    net_path: Path,
    polygons_path: Path,
    routes_path: Path,
    plan_path: Path,
    config_path: Path,
    seed: int,
    num_vehicles: int,
    regular_count: int,
) -> None:
    if not 0 < regular_count <= len(REGULAR_GATE_PAIRS):
        raise ValueError("unsupported regular vehicle count")
    if regular_count >= num_vehicles:
        raise ValueError("regular_count must be smaller than num_vehicles")

    net = sumolib.net.readNet(str(net_path))
    rng = random.Random(int(seed) + 2_204_203)
    root = ET.Element("routes")
    _add_vehicle_type(root)
    regulars: list[dict[str, object]] = []
    visitors: list[dict[str, object]] = []

    for node_idx, (gate_a, gate_b) in enumerate(
        REGULAR_GATE_PAIRS[:regular_count]
    ):
        forward = _shortest_route(net, gate_a, gate_b)
        reverse = _shortest_route(net, gate_b, gate_a)
        forward_id = f"regular_{node_idx:03d}_forward"
        reverse_id = f"regular_{node_idx:03d}_reverse"
        ET.SubElement(root, "route", id=forward_id, edges=" ".join(forward))
        ET.SubElement(root, "route", id=reverse_id, edges=" ".join(reverse))
        physical_id = f"regular_{node_idx:03d}"
        ET.SubElement(
            root,
            "vehicle",
            id=physical_id,
            type="passenger",
            route=forward_id,
            depart="0",
            departLane="best",
            departPos="0",
            departSpeed="max",
        )
        regulars.append(
            {
                "node_idx": node_idx,
                "physical_vehicle_id": physical_id,
                "entry_gate": gate_a,
                "exit_gate": gate_b,
                "forward_route_id": forward_id,
                "reverse_route_id": reverse_id,
                "forward_edges": forward,
                "reverse_edges": reverse,
                "reentry_wait_steps": {
                    "distribution": "uniform_integer",
                    "min": 10,
                    "max": 20,
                },
                "persistent_state_while_absent": True,
            }
        )

    gate_names = sorted(GATES)
    for node_idx in range(regular_count, num_vehicles):
        entry = rng.choice(gate_names)
        exit_gate = rng.choice(
            [gate for gate in gate_names if gate != entry]
        )
        edges = _shortest_route(net, entry, exit_gate)
        route_id = f"visitor_initial_{node_idx:03d}"
        physical_id = f"visitor_{node_idx:03d}_g00000"
        ET.SubElement(root, "route", id=route_id, edges=" ".join(edges))
        ET.SubElement(
            root,
            "vehicle",
            id=physical_id,
            type="passenger",
            route=route_id,
            depart=str((node_idx - regular_count) // 6),
            departLane="best",
            departPos="0",
            departSpeed="max",
        )
        visitors.append(
            {
                "node_idx": node_idx,
                "initial_physical_vehicle_id": physical_id,
                "entry_gate": entry,
                "exit_gate": exit_gate,
                "initial_route_id": route_id,
                "initial_edges": edges,
            }
        )

    _write_xml(root, routes_path)
    plan = {
        "format": "single_zone_role_route_plan_v2",
        "seed": int(seed),
        "map": "single_zone_urban_150",
        "map_size": 150.0,
        "num_vehicle_slots": int(num_vehicles),
        "regular_count": int(regular_count),
        "visitor_count": int(num_vehicles - regular_count),
        "gates": GATES,
        "border_intersection_count": len(GATES),
        "all_border_intersections_enabled": True,
        "regular_vehicles": regulars,
        "one_time_visitor_slots": visitors,
        "visitor_replacement": {
            "delay_steps": 0,
            "entry_gate": "uniform_random_over_all_border_intersections",
            "exit_gate": "uniform_random_over_all_other_border_intersections",
            "routing": "shortest_path",
            "physical_state": "complete_cold_start",
            "generation_increments": True,
            "rng_seed": int(seed) + 9_104_729,
        },
        "regular_reentry": {
            "direction": "alternate_forward_reverse",
            "wait_steps": {
                "distribution": "uniform_integer",
                "min": 10,
                "max": 20,
            },
            "physical_state": "fully_persistent",
            "generation_increments": False,
            "rng_seed": int(seed) + 6_301_337,
        },
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    _build_config(
        config_path,
        net=net_path,
        routes=routes_path,
        polygons=polygons_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument(
        "--polygons",
        type=Path,
        default=SCENARIO / "single_zone_urban_150.poly.xml",
    )
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-vehicles", type=int, default=20)
    parser.add_argument("--regular-count", type=int, default=2)
    args = parser.parse_args()
    build(
        net_path=args.net.resolve(),
        polygons_path=args.polygons.resolve(),
        routes_path=args.routes.resolve(),
        plan_path=args.plan.resolve(),
        config_path=args.config.resolve(),
        seed=args.seed,
        num_vehicles=args.num_vehicles,
        regular_count=args.regular_count,
    )
    print(args.routes.resolve())
    print(args.plan.resolve())
    print(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
