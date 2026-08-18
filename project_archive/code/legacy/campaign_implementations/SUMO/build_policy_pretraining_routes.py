#!/usr/bin/env python3
"""Build role-based source-map routes for transferable policy pretraining."""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _shortest_route(net, gates: dict[str, dict[str, str]], entry: str, exit_gate: str) -> list[str]:
    result = net.getShortestPath(
        net.getEdge(gates[entry]["entry"]),
        net.getEdge(gates[exit_gate]["exit"]),
    )
    if result is None or not result[0]:
        raise ValueError(f"no route from {entry} to {exit_gate}")
    return [edge.getID() for edge in result[0]]


def _gate_xy(gate: dict[str, str]) -> tuple[int, int]:
    _prefix, col, row = str(gate["node"]).replace("N", "N_").split("_")
    return int(col), int(row)


def _regular_pairs(gates: dict[str, dict[str, str]], count: int) -> list[tuple[str, str]]:
    names = sorted(gates)
    candidates = sorted(
        (
            (
                abs(_gate_xy(gates[left])[0] - _gate_xy(gates[right])[0])
                + abs(_gate_xy(gates[left])[1] - _gate_xy(gates[right])[1]),
                left,
                right,
            )
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    selected: list[tuple[str, str]] = []
    usage = {name: 0 for name in names}
    for _distance, left, right in candidates:
        if usage[left] or usage[right]:
            continue
        selected.append((left, right))
        usage[left] += 1
        usage[right] += 1
        if len(selected) >= int(count):
            break
    if len(selected) != int(count):
        raise ValueError(f"only found {len(selected)} disjoint regular routes")
    return selected


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


def _build_config(path: Path, *, net: Path, routes: Path, polygons: Path, steps: int) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=str(net.resolve()))
    ET.SubElement(inputs, "route-files", value=str(routes.resolve()))
    ET.SubElement(inputs, "additional-files", value=str(polygons.resolve()))
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value=str(int(steps)))
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def build(
    *,
    source_manifest: Path,
    routes_path: Path,
    plan_path: Path,
    config_path: Path,
    seed: int,
    num_vehicles: int,
    regular_count: int,
    steps: int,
) -> None:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    network = Path(source["network"])
    polygons = Path(source["polygons"])
    gates = {
        str(name): {str(k): str(v) for k, v in payload.items()}
        for name, payload in source["gates"].items()
    }
    if not 0 < int(regular_count) < int(num_vehicles):
        raise ValueError("require 0 < regular-count < num-vehicles")
    net = sumolib.net.readNet(str(network))
    rng = random.Random(int(seed) + int(source["seed"]) * 101)
    root = ET.Element("routes")
    _add_vehicle_type(root)
    regulars: list[dict[str, object]] = []
    visitors: list[dict[str, object]] = []

    for node_idx, (gate_a, gate_b) in enumerate(_regular_pairs(gates, regular_count)):
        forward = _shortest_route(net, gates, gate_a, gate_b)
        reverse = _shortest_route(net, gates, gate_b, gate_a)
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
                    "min": 8,
                    "max": 22,
                },
                "persistent_state_while_absent": True,
            }
        )

    gate_names = sorted(gates)
    for node_idx in range(int(regular_count), int(num_vehicles)):
        entry = rng.choice(gate_names)
        exit_gate = rng.choice([gate for gate in gate_names if gate != entry])
        edges = _shortest_route(net, gates, entry, exit_gate)
        route_id = f"visitor_initial_{node_idx:03d}"
        physical_id = f"visitor_{node_idx:03d}_g00000"
        ET.SubElement(root, "route", id=route_id, edges=" ".join(edges))
        ET.SubElement(
            root,
            "vehicle",
            id=physical_id,
            type="passenger",
            route=route_id,
            depart=str((node_idx - int(regular_count)) // 6),
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
        "format": "cross_map_policy_pretraining_route_plan_v1",
        "seed": int(seed),
        "map": str(source["map_id"]),
        "map_split": str(source["split"]),
        "map_size": float(source["map_size"]),
        "num_vehicle_slots": int(num_vehicles),
        "regular_count": int(regular_count),
        "visitor_count": int(num_vehicles - regular_count),
        "gates": gates,
        "border_intersection_count": len(gates),
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
            "rng_seed": int(seed) + int(source["seed"]) + 9_104_729,
        },
        "regular_reentry": {
            "direction": "alternate_forward_reverse",
            "wait_steps": {
                "distribution": "uniform_integer",
                "min": 8,
                "max": 22,
            },
            "physical_state": "fully_persistent",
            "generation_increments": False,
            "rng_seed": int(seed) + int(source["seed"]) + 6_301_337,
        },
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    _build_config(
        config_path,
        net=network,
        routes=routes_path,
        polygons=polygons,
        steps=int(steps),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-vehicles", type=int, default=24)
    parser.add_argument("--regular-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    build(
        source_manifest=args.source_manifest.resolve(),
        routes_path=args.routes.resolve(),
        plan_path=args.plan.resolve(),
        config_path=args.config.resolve(),
        seed=int(args.seed),
        num_vehicles=int(args.num_vehicles),
        regular_count=int(args.regular_count),
        steps=int(args.steps),
    )
    print(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
