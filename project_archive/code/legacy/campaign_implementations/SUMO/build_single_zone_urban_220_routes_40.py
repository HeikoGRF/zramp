#!/usr/bin/env python3
"""Build seed-1 routes for two buses, two regulars, and 36 crossings."""

from __future__ import annotations

import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import sumolib


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "SUMO" / "single_zone_urban_220"
DEFAULT_OUTPUT = SCENARIO / "roles_40_seed01"
DEFAULT_NET = SCENARIO / "single_zone_urban_220.net.xml"
MAP_SIZE = 220.0
BUS_COUNT = 2
REGULAR_COUNT = 2
CROSSING_COUNT = 36
NUM_VEHICLES = BUS_COUNT + REGULAR_COUNT + CROSSING_COUNT

REGULAR_GATES = {
    "S1": {"entry": "N1_0__N1_1", "exit": "N1_1__N1_0"},
    "N2": {"entry": "N4_5__N4_4", "exit": "N4_4__N4_5"},
    "W1": {"entry": "N0_1__N1_1", "exit": "N1_1__N0_1"},
    "E2": {"entry": "N5_4__N4_4", "exit": "N4_4__N5_4"},
}
REGULAR_PAIRS = (("S1", "N2"), ("W1", "E2"))


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _road_edges(net) -> list:
    return sorted(
        [
            edge
            for edge in net.getEdges()
            if not edge.getID().startswith(":")
            and str(edge.getFunction() or "") == ""
        ],
        key=lambda edge: edge.getID(),
    )


def _street_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _edge_lookup(net):
    directed = {}
    streets = set()
    edge_pairs = {}
    for edge in _road_edges(net):
        left = edge.getFromNode().getID()
        right = edge.getToNode().getID()
        directed[(left, right)] = edge.getID()
        edge_pairs[edge.getID()] = (left, right)
        streets.add(_street_key(left, right))
    for left, right in streets:
        if (left, right) not in directed or (right, left) not in directed:
            raise ValueError(f"street {left}<->{right} is not bidirectional")
    return directed, edge_pairs, streets


def _row(node_id: str) -> int:
    return int(str(node_id).split("_")[1])


def _closed_cover_walk(streets, directed, *, start: str) -> list[str]:
    adjacency = defaultdict(list)
    for left, right in streets:
        adjacency[left].append(right)
        adjacency[right].append(left)
    visited = set()
    walk = []

    def visit(node: str) -> None:
        for neighbor in sorted(adjacency[node]):
            street = _street_key(node, neighbor)
            if street in visited:
                continue
            visited.add(street)
            walk.append(directed[(node, neighbor)])
            visit(neighbor)
            walk.append(directed[(neighbor, node)])

    visit(start)
    if visited != streets:
        raise ValueError("bus street partition is disconnected")
    return walk


def _bus_routes(streets, directed, edge_pairs):
    south = {
        street
        for street in streets
        if not (_row(street[0]) >= 3 and _row(street[1]) >= 3)
    }
    north = streets - south
    partitions = (("south", south, "N1_1"), ("north", north, "N1_4"))
    buses = []
    covered = set()
    for index, (name, partition, start) in enumerate(partitions):
        base = _closed_cover_walk(partition, directed, start=start)
        traversed = {_street_key(*edge_pairs[edge]) for edge in base}
        covered.update(traversed)
        if edge_pairs[base[0]][0] != start or edge_pairs[base[-1]][1] != start:
            raise AssertionError("bus route is not closed")
        buses.append(
            {
                "node_idx": index,
                "logical_vehicle_id": f"node_{index:03d}",
                "physical_vehicle_id": f"bus_{index:03d}",
                "route_id": f"bus_{name}_closed_cover",
                "district": name,
                "start_node": start,
                "base_edges": base,
                "route_edges": base * 12,
                "base_street_count": len(traversed),
                "closed": True,
                "persistent_state": True,
            }
        )
    if covered != streets:
        raise AssertionError(f"bus routes miss {len(streets - covered)} streets")
    return buses


def _shortest_edges(net, entry_id: str, exit_id: str) -> list[str]:
    result = net.getShortestPath(net.getEdge(entry_id), net.getEdge(exit_id))
    if result is None or not result[0]:
        raise ValueError(f"no route from {entry_id} to {exit_id}")
    return [edge.getID() for edge in result[0]]


def _reverse_route(forward, directed, edge_pairs) -> list[str]:
    return [
        directed[(edge_pairs[edge][1], edge_pairs[edge][0])]
        for edge in reversed(forward)
    ]


def _outside_position(node, distance: float = 8.0) -> list[float]:
    x, y = (float(value) for value in node.getCoord())
    if math.isclose(x, 0.0):
        x -= distance
    elif math.isclose(x, MAP_SIZE):
        x += distance
    elif math.isclose(y, 0.0):
        y -= distance
    elif math.isclose(y, MAP_SIZE):
        y += distance
    return [round(x, 3), round(y, 3)]


def _regular_routes(net, directed, edge_pairs):
    regulars = []
    for offset, (gate_a, gate_b) in enumerate(REGULAR_PAIRS):
        node_idx = BUS_COUNT + offset
        forward = _shortest_edges(
            net, REGULAR_GATES[gate_a]["entry"], REGULAR_GATES[gate_b]["exit"]
        )
        reverse = _reverse_route(forward, directed, edge_pairs)
        regulars.append(
            {
                "node_idx": node_idx,
                "logical_vehicle_id": f"node_{node_idx:03d}",
                "physical_vehicle_id": f"regular_{offset:03d}_trip00000",
                "gate_a": gate_a,
                "gate_b": gate_b,
                "forward_route_id": f"regular_{offset:03d}_forward",
                "reverse_route_id": f"regular_{offset:03d}_reverse",
                "forward_edges": forward,
                "reverse_edges": reverse,
                "staging_at_a": _outside_position(net.getEdge(forward[0]).getFromNode()),
                "staging_at_b": _outside_position(net.getEdge(forward[-1]).getToNode()),
                "wait_steps": {"distribution": "uniform_integer", "min": 20, "max": 40},
                "persistent_state_while_waiting": True,
            }
        )
    return regulars


def _is_boundary(node) -> bool:
    x, y = (float(value) for value in node.getCoord())
    return any(
        math.isclose(value, boundary)
        for value, boundary in ((x, 0.0), (x, MAP_SIZE), (y, 0.0), (y, MAP_SIZE))
    )


def _random_crossing(net, rng, entry_edges, *, max_decisions: int = 300):
    for _attempt in range(1000):
        first = rng.choice(entry_edges)
        entry_node = first.getFromNode()
        previous = entry_node.getID()
        current = first.getToNode()
        route = [first.getID()]
        for _decision in range(max_decisions):
            outgoing = [
                edge
                for edge in current.getOutgoing()
                if not edge.getID().startswith(":")
                and str(edge.getFunction() or "") == ""
            ]
            non_uturn = [edge for edge in outgoing if edge.getToNode().getID() != previous]
            choices = non_uturn or outgoing
            admissible = []
            for edge in choices:
                destination = edge.getToNode()
                if not _is_boundary(destination) or math.dist(
                    entry_node.getCoord(), destination.getCoord()
                ) >= 110.0:
                    admissible.append(edge)
            if not admissible:
                break
            selected = rng.choice(sorted(admissible, key=lambda edge: edge.getID()))
            route.append(selected.getID())
            previous = current.getID()
            current = selected.getToNode()
            if _is_boundary(current):
                return route
    raise RuntimeError("could not sample a terminating random crossing route")


def _crossing_routes(net, rng):
    entry_edges = [
        edge
        for edge in _road_edges(net)
        if _is_boundary(edge.getFromNode()) and not _is_boundary(edge.getToNode())
    ]
    crossings = []
    for offset in range(CROSSING_COUNT):
        node_idx = BUS_COUNT + REGULAR_COUNT + offset
        edges = _random_crossing(net, rng, entry_edges)
        crossings.append(
            {
                "node_idx": node_idx,
                "logical_vehicle_id": f"node_{node_idx:03d}",
                "physical_vehicle_id": f"crossing_{offset:03d}",
                "route_id": f"crossing_{offset:03d}_random_walk",
                "edges": edges,
                "decision_count": len(edges) - 1,
                "exit_staging": _outside_position(net.getEdge(edges[-1]).getToNode()),
                "one_shot": True,
            }
        )
    return crossings


def _vehicle_types(root: ET.Element) -> None:
    ET.SubElement(root, "vType", id="bus", vClass="bus", accel="1.2", decel="3.0", sigma="0.15", length="10.0", minGap="3.0", tau="1.0", maxSpeed="8.33", speedFactor="0.95", speedDev="0.03")
    ET.SubElement(root, "vType", id="passenger", vClass="passenger", accel="1.5", decel="3.5", sigma="0.25", length="4.8", minGap="2.5", tau="1.0", maxSpeed="11.11", speedFactor="0.95", speedDev="0.08")


def _build_config(path: Path, route_name: str, sim_steps: int) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="../single_zone_urban_220.net.xml")
    ET.SubElement(inputs, "route-files", value=route_name)
    ET.SubElement(inputs, "additional-files", value="../single_zone_urban_220.poly.xml")
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value=str(int(sim_steps) + 1))
    ET.SubElement(timing, "step-length", value="1.0")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(root, path)


def build(net_path: Path, output: Path, *, seed: int, sim_steps: int) -> None:
    net = sumolib.net.readNet(str(net_path))
    directed, edge_pairs, streets = _edge_lookup(net)
    buses = _bus_routes(streets, directed, edge_pairs)
    regulars = _regular_routes(net, directed, edge_pairs)
    crossings = _crossing_routes(net, random.Random(int(seed) + 4_022_031))
    output.mkdir(parents=True, exist_ok=True)
    routes_path = output / "roles_40_seed01.rou.xml"
    plan_path = output / "roles_40_seed01.json"
    config_path = output / "roles_40_seed01.sumocfg"
    root = ET.Element("routes")
    _vehicle_types(root)
    for bus in buses:
        ET.SubElement(root, "route", id=bus["route_id"], edges=" ".join(bus["route_edges"]))
        ET.SubElement(root, "vehicle", id=bus["physical_vehicle_id"], type="bus", route=bus["route_id"], depart="0", departLane="best", departPos="0", departSpeed="max")
    for regular in regulars:
        ET.SubElement(root, "route", id=regular["forward_route_id"], edges=" ".join(regular["forward_edges"]))
        ET.SubElement(root, "route", id=regular["reverse_route_id"], edges=" ".join(regular["reverse_edges"]))
        ET.SubElement(root, "vehicle", id=regular["physical_vehicle_id"], type="passenger", route=regular["forward_route_id"], depart="0", departLane="best", departPos="0", departSpeed="max")
    for offset, crossing in enumerate(crossings):
        ET.SubElement(root, "route", id=crossing["route_id"], edges=" ".join(crossing["edges"]))
        ET.SubElement(root, "vehicle", id=crossing["physical_vehicle_id"], type="passenger", route=crossing["route_id"], depart=str(offset // 12), departLane="best", departPos="0", departSpeed="max")
    _write_xml(root, routes_path)
    _build_config(config_path, routes_path.name, sim_steps)
    plan = {
        "format": "single_zone_urban_220_roles_40_v1",
        "seed": int(seed), "sim_steps": int(sim_steps),
        "map": "single_zone_urban_220", "map_size": MAP_SIZE,
        "num_vehicle_slots": NUM_VEHICLES, "bus_count": BUS_COUNT,
        "regular_count": REGULAR_COUNT, "crossing_count": CROSSING_COUNT,
        "street_count": len(streets), "bus_union_street_coverage": len(streets),
        "bus_vehicles": buses, "regular_vehicles": regulars,
        "crossing_vehicles": crossings, "policy_role_labels_visible": False,
        "crossing_route_rule": "uniform random non-U-turn outgoing edge at every intersection, conditioned on exiting at least 110 m from entry",
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(routes_path)
    print(plan_path)
    print(config_path)
    print(f"vehicles={NUM_VEHICLES} streets={len(streets)} bus_coverage={len(streets)} crossings={len(crossings)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-steps", type=int, default=1000)
    args = parser.parse_args()
    build(args.net.resolve(), args.output.resolve(), seed=args.seed, sim_steps=args.sim_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
