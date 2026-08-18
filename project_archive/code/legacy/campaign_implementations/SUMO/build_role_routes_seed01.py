#!/usr/bin/env python3
"""Build the seed-1 regular/one-time route plan for the compact map."""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "SUMO" / "single_zone_urban_220"
DEFAULT_NET = MAP_DIR / "single_zone_urban_220.net.xml"
DEFAULT_ROUTES = MAP_DIR / "single_zone_urban_220_roles_seed01.rou.xml"
DEFAULT_PLAN = MAP_DIR / "single_zone_urban_220_roles_seed01.json"
DEFAULT_CONFIG = MAP_DIR / "single_zone_urban_220_roles_seed01.sumocfg"

GATES = {
    "S1": {"entry": "N1_0__N1_1", "exit": "N1_1__N1_0"},
    "S2": {"entry": "N4_0__N4_1", "exit": "N4_1__N4_0"},
    "N1": {"entry": "N1_5__N1_4", "exit": "N1_4__N1_5"},
    "N2": {"entry": "N4_5__N4_4", "exit": "N4_4__N4_5"},
    "W1": {"entry": "N0_1__N1_1", "exit": "N1_1__N0_1"},
    "W2": {"entry": "N0_4__N1_4", "exit": "N1_4__N0_4"},
    "E1": {"entry": "N5_1__N4_1", "exit": "N4_1__N5_1"},
    "E2": {"entry": "N5_4__N4_4", "exit": "N4_4__N5_4"},
}

REGULAR_GATE_PAIRS = (
    ("S1", "N2"),
    ("W1", "E2"),
    ("E1", "W2"),
)


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _shortest_route(net, entry_gate: str, exit_gate: str) -> list[str]:
    entry = net.getEdge(GATES[entry_gate]["entry"])
    exit_edge = net.getEdge(GATES[exit_gate]["exit"])
    result = net.getShortestPath(entry, exit_edge)
    if result is None or not result[0]:
        raise ValueError(f"no route from {entry_gate} to {exit_gate}")
    edges = [edge.getID() for edge in result[0]]
    if edges[0] != entry.getID() or edges[-1] != exit_edge.getID():
        raise ValueError(f"incomplete route from {entry_gate} to {exit_gate}")
    return edges


def _vehicle_type(root: ET.Element) -> None:
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


def _build_config(path: Path, route_name: str) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="single_zone_urban_220.net.xml")
    ET.SubElement(inputs, "route-files", value=route_name)
    ET.SubElement(inputs, "additional-files", value="single_zone_urban_220.poly.xml")
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value="3000")
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def build(
    net_path: Path,
    routes_path: Path,
    plan_path: Path,
    config_path: Path,
    *,
    seed: int,
    num_vehicles: int,
    regular_count: int,
) -> None:
    if not 0 < regular_count <= len(REGULAR_GATE_PAIRS):
        raise ValueError(f"regular_count must be in [1, {len(REGULAR_GATE_PAIRS)}]")
    if regular_count >= num_vehicles:
        raise ValueError("regular_count must be between zero and num_vehicles")
    net = sumolib.net.readNet(str(net_path))
    rng = random.Random(int(seed) + 2_204_203)
    root = ET.Element("routes")
    _vehicle_type(root)
    regulars: list[dict[str, object]] = []
    visitors: list[dict[str, object]] = []

    for node_idx, (gate_a, gate_b) in enumerate(REGULAR_GATE_PAIRS[:regular_count]):
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
                "reentry_wait_steps": {"distribution": "uniform_integer", "min": 10, "max": 20},
                "persistent_state_while_absent": True,
            }
        )

    gate_names = sorted(GATES)
    for node_idx in range(regular_count, num_vehicles):
        entry_gate = rng.choice(gate_names)
        exit_gate = rng.choice([gate for gate in gate_names if gate != entry_gate])
        edges = _shortest_route(net, entry_gate, exit_gate)
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
                "entry_gate": entry_gate,
                "exit_gate": exit_gate,
                "initial_route_id": route_id,
                "initial_edges": edges,
            }
        )

    routes_path.parent.mkdir(parents=True, exist_ok=True)
    _write_xml(root, routes_path)
    plan = {
        "format": "single_zone_role_route_plan_v1",
        "seed": int(seed),
        "map": "single_zone_urban_220",
        "num_vehicle_slots": int(num_vehicles),
        "regular_count": int(regular_count),
        "visitor_count": int(num_vehicles - regular_count),
        "regular_fraction": float(regular_count / num_vehicles),
        "gates": GATES,
        "regular_vehicles": regulars,
        "one_time_visitor_slots": visitors,
        "visitor_replacement": {
            "delay_steps": 0,
            "entry_gate": "uniform_random_over_all_gates",
            "exit_gate": "uniform_random_over_all_other_gates",
            "routing": "shortest_path",
            "physical_state": "complete_cold_start",
            "generation_increments": True,
            "rng_seed": int(seed) + 9_104_729,
        },
        "regular_reentry": {
            "direction": "alternate_forward_reverse",
            "wait_steps": {"distribution": "uniform_integer", "min": 10, "max": 20},
            "physical_state": "fully_persistent",
            "generation_increments": False,
            "rng_seed": int(seed) + 6_301_337,
        },
        "policy": {
            "role_label_visible": False,
            "policy_gossip": "every_feasible_contact",
            "exploration_probability": 0.02,
        },
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    _build_config(config_path, routes_path.name)
    print(routes_path)
    print(plan_path)
    print(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-vehicles", type=int, default=20)
    parser.add_argument("--regular-count", type=int, default=2)
    args = parser.parse_args()
    build(
        args.net.resolve(),
        args.routes.resolve(),
        args.plan.resolve(),
        args.config.resolve(),
        seed=args.seed,
        num_vehicles=args.num_vehicles,
        regular_count=args.regular_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
