#!/usr/bin/env python3
"""Export a SUMO-only vehicle trace for the seeded random-OD mobility model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections import Counter
from pathlib import Path

import traci

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTE_REASSIGN_EDGE_BUFFER = 4
INVALID_SUMO_POSITION = -1.0e9


def _valid_sumo_position(x: float, y: float) -> bool:
    return float(x) > INVALID_SUMO_POSITION and float(y) > INVALID_SUMO_POSITION

from rl_reward_experiment.mobility import zone_of
from SUMO.sumo_sionna_map import read_net_bounds


def _closed_edges_from_config(sumo_config: Path) -> set[str]:
    closed: set[str] = set()
    root = ET.parse(sumo_config).getroot()
    for elem in root.findall(".//additional-files"):
        for raw in elem.get("value", "").split(","):
            item = raw.strip()
            if not item:
                continue
            path = Path(item)
            path = path if path.is_absolute() else sumo_config.parent / path
            if not path.is_file():
                continue
            try:
                add_root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            for closing in add_root.findall(".//closingReroute"):
                edge_id = closing.get("id")
                if edge_id:
                    closed.add(edge_id)
    return closed


def _routing_index(
    net_path: Path,
    sumo_config: Path,
    *,
    num_zones: int,
    map_size: float,
    boundary_margin: float = 0.12,
):
    bounds = read_net_bounds(str(net_path))
    closed = _closed_edges_from_config(sumo_config)
    by_zone: dict[int, list[str]] = defaultdict(list)
    edge_zone: dict[str, int] = {}
    edge_shape: dict[str, list[tuple[float, float]]] = {}
    edge_midpoint: dict[str, tuple[float, float]] = {}
    zone_capacity: dict[int, float] = defaultdict(float)
    boundary_edges: dict[str, list[str]] = defaultdict(list)
    incoming_by_node: dict[str, list[str]] = defaultdict(list)
    outgoing_by_node: dict[str, list[str]] = defaultdict(list)
    root = ET.parse(net_path).getroot()
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":") or edge.get("function") == "internal":
            continue
        if edge_id in closed:
            continue
        lane = edge.find("lane")
        if lane is None or not lane.get("shape"):
            continue
        pts = []
        for item in lane.get("shape", "").split():
            x, y = item.split(",")[:2]
            pts.append((float(x), float(y)))
        if len(pts) < 2:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        zx = (mx - bounds.x0) / max(1e-9, bounds.width) * map_size
        zy = (my - bounds.y0) / max(1e-9, bounds.height) * map_size
        az = zone_of(zx, zy, map_size, num_zones)
        edge_zone[edge_id] = int(az)
        edge_shape[edge_id] = pts
        edge_midpoint[edge_id] = (float(zx), float(zy))
        by_zone[int(az)].append(edge_id)
        from_node = edge.get("from") or ""
        to_node = edge.get("to") or ""
        if from_node:
            outgoing_by_node[from_node].append(edge_id)
        if to_node:
            incoming_by_node[to_node].append(edge_id)
        try:
            zone_capacity[int(az)] += float(lane.get("length") or 0.0)
        except ValueError:
            zone_capacity[int(az)] += 0.0
        dists = {
            "left": float(zx),
            "right": float(map_size) - float(zx),
            "bottom": float(zy),
            "top": float(map_size) - float(zy),
        }
        side, dist = min(dists.items(), key=lambda item: item[1])
        if dist <= float(boundary_margin) * float(map_size):
            boundary_edges[side].append(edge_id)
    junction_in_edges = {
        node: sorted(edges)
        for node, edges in incoming_by_node.items()
        if len(edges) >= 3 and len(outgoing_by_node.get(node, ())) >= 3
    }
    return (
        dict(by_zone),
        edge_zone,
        edge_shape,
        edge_midpoint,
        dict(zone_capacity),
        {k: list(v) for k, v in boundary_edges.items()},
        junction_in_edges,
    )


def _zone_distance(a: int, b: int, *, num_zones: int) -> int:
    side = int(round(float(num_zones) ** 0.5))
    if side * side != int(num_zones):
        return 0 if a == b else 1
    return abs((a % side) - (b % side)) + abs((a // side) - (b // side))


def _candidate_zones(
    zones: list[int],
    current_zone: int | None,
    *,
    num_zones: int,
    min_distance: int,
    max_distance: int | None,
    ignore_min_distance: bool = False,
) -> list[int]:
    if current_zone is None:
        return list(zones)
    out: list[int] = []
    for z in zones:
        if z == current_zone:
            continue
        dist = _zone_distance(current_zone, z, num_zones=num_zones)
        if not ignore_min_distance and dist < min_distance:
            continue
        if max_distance is not None and dist > max_distance:
            continue
        out.append(z)
    if out:
        return out
    if max_distance is not None:
        close = [
            z
            for z in zones
            if z != current_zone
            and _zone_distance(current_zone, z, num_zones=num_zones) <= max_distance
        ]
        if close:
            return close
    return [z for z in zones if z != current_zone] or list(zones)



def _zone_expected_count(zone: int, zone_capacity, by_zone, *, num_nodes: int) -> float:
    total = sum(max(1.0, float(v)) for v in zone_capacity.values())
    if total <= 0.0:
        return max(1.0, float(num_nodes) / max(1, len(by_zone)))
    return max(1.0, float(num_nodes) * max(1.0, float(zone_capacity.get(int(zone), 1.0))) / total)


def _zone_density(zone: int, live_counts, zone_capacity, by_zone, *, num_nodes: int) -> float:
    return float(live_counts[int(zone)]) / _zone_expected_count(
        int(zone), zone_capacity, by_zone, num_nodes=num_nodes
    )



def _opposite_side(side: str) -> str:
    return {"left": "right", "right": "left", "bottom": "top", "top": "bottom"}[side]


def _nearest_side(edge_id: str, edge_midpoint, *, map_size: float, margin: float) -> str | None:
    point = edge_midpoint.get(edge_id)
    if point is None:
        return None
    x, y = point
    dists = {"left": x, "right": map_size - x, "bottom": y, "top": map_size - y}
    side, dist = min(dists.items(), key=lambda item: item[1])
    return side if dist <= margin * map_size else None


def _rank_boundary_sides(current_edge, boundary_edges, edge_zone, edge_midpoint, live_counts, zone_capacity, by_zone, *, map_size, num_nodes, margin):
    sides = [side for side, edges in boundary_edges.items() if edges]
    nearest = _nearest_side(current_edge, edge_midpoint, map_size=map_size, margin=margin)
    preferred = _opposite_side(nearest) if nearest else None

    def score(side: str):
        zones = [edge_zone[e] for e in boundary_edges.get(side, []) if e in edge_zone]
        density = min(
            (_zone_density(z, live_counts, zone_capacity, by_zone, num_nodes=num_nodes) for z in zones),
            default=0.0,
        )
        return (0 if side == preferred else 1, density)

    return sorted(sides, key=score)


def _rank_boundary_edges(side, boundary_edges, edge_zone, live_counts, zone_capacity, by_zone, blocked, rng, *, num_nodes):
    edges = [e for e in boundary_edges.get(side, []) if e not in blocked]
    edges.sort(
        key=lambda e: (
            _zone_density(edge_zone.get(e, 0), live_counts, zone_capacity, by_zone, num_nodes=num_nodes),
            rng.random(),
        )
    )
    return edges


def _central_route_zones(num_zones: int) -> set[int]:
    side = int(round(float(num_zones) ** 0.5))
    if side * side != int(num_zones) or side <= 2:
        return set()
    if side % 2 == 1:
        mid = side // 2
        return {mid + side * mid}
    lo = side // 2 - 1
    hi = side // 2
    return {x + side * y for x in (lo, hi) for y in (lo, hi)}


def _edge_from_lane_id(lane_id: str) -> str:
    return str(lane_id).rsplit("_", 1)[0]


def _lane_can_reach_edge(lane_id: str, edge_id: str) -> bool:
    if not lane_id or not edge_id:
        return False
    try:
        links = traci.lane.getLinks(lane_id, extended=True)
    except traci.TraCIException:
        return True
    for link in links:
        if not link:
            continue
        target_lane = str(link[0])
        if target_lane and _edge_from_lane_id(target_lane) == edge_id:
            return True
    return False


def _corner_route_zones(num_zones: int) -> set[int]:
    side = int(round(float(num_zones) ** 0.5))
    if side * side != int(num_zones) or side < 2:
        return set()
    return {0, side - 1, side * (side - 1), side * side - 1}


def _score_route(
    edges,
    dest_zone,
    edge_zone,
    target_counts,
    zone_load,
    live_counts,
    zone_capacity,
    by_zone,
    edge_pressure,
    *,
    num_nodes: int,
    num_zones: int,
):
    counts = Counter(edge_zone[e] for e in edges if e in edge_zone)
    edge_jam = sum(float(edge_pressure.get(edge_id, 0.0)) for edge_id in edges)
    crowd = sum(float(n) * float(zone_load[z]) for z, n in counts.items())
    live_crowd = sum(float(n) * float(live_counts[z]) for z, n in counts.items())
    density_crowd = sum(
        float(n) * _zone_density(int(z), live_counts, zone_capacity, by_zone, num_nodes=num_nodes)
        for z, n in counts.items()
    )
    central_zones = _central_route_zones(num_zones)
    central = sum(float(n) for z, n in counts.items() if z in central_zones)
    if int(num_zones) == 9:
        zones_seen = set(counts)
        perimeter_zones = set(edge_zone.values()) - central_zones
        corner_zones = _corner_route_zones(num_zones)
        target_penalty = 70.0 * float(target_counts[dest_zone])
        live_target_penalty = 55.0 * float(live_counts[dest_zone])
        density_target_penalty = 180.0 * _zone_density(
            int(dest_zone), live_counts, zone_capacity, by_zone, num_nodes=num_nodes
        )
        center_penalty = 125.0 * central
        crowd_penalty = 1.6 * crowd + 1.9 * live_crowd + 95.0 * density_crowd + 18.0 * edge_jam
        coverage_bonus = 12.0 * float(len(zones_seen & perimeter_zones))
        corner_bonus = 12.0 * float(len(zones_seen & corner_zones))
        underused_bonus = sum(
            20.0 / (1.0 + float(live_counts[z]))
            for z in zones_seen & perimeter_zones
        )
        score = (
            float(len(edges))
            + crowd_penalty
            + center_penalty
            + target_penalty
            + live_target_penalty
            + density_target_penalty
            - coverage_bonus
            - corner_bonus
            - underused_bonus
        )
        return score, counts
    center_penalty = 6.0 * central
    crowd_penalty = 0.35 * crowd + 0.15 * live_crowd + 20.0 * density_crowd + 8.0 * edge_jam
    score = float(len(edges)) + crowd_penalty + center_penalty + 55.0 * float(target_counts[dest_zone])
    return score, counts


def _cached_route(route_cache, start_edge: str, end_edge: str):
    cache_key = (start_edge, end_edge)
    if cache_key not in route_cache:
        route_cache[cache_key] = list(traci.simulation.findRoute(start_edge, end_edge).edges)
    return route_cache[cache_key]


def _compose_route_edges(
    current_edge: str,
    current_lane: str,
    dest_edge: str,
    blocked,
    route_cache,
    *,
    via_edge: str | None = None,
):
    stops = [current_edge]
    if via_edge is not None:
        stops.append(via_edge)
    stops.append(dest_edge)
    full_edges = []
    for idx, (start_edge, end_edge) in enumerate(zip(stops, stops[1:])):
        if start_edge == end_edge:
            return None
        edges = _cached_route(route_cache, start_edge, end_edge)
        if len(edges) < 2 or edges[0] != start_edge or edges[-1] != end_edge:
            return None
        if idx == 0 and current_lane and not _lane_can_reach_edge(current_lane, edges[1]):
            return None
        if idx == 0:
            full_edges.extend(edges)
        else:
            full_edges.extend(edges[1:])
    if len(full_edges) < 2 or full_edges[0] != current_edge or full_edges[-1] != dest_edge:
        return None
    if blocked.intersection(full_edges[1:]):
        return None
    return full_edges


def _live_zone_counts(bounds, map_size: float, num_zones: int):
    counts = Counter()
    try:
        vehicle_ids = traci.vehicle.getIDList()
    except traci.TraCIException:
        return counts
    for veh_id in vehicle_ids:
        try:
            x, y = traci.vehicle.getPosition(veh_id)
        except traci.TraCIException:
            continue
        if not _valid_sumo_position(x, y):
            continue
        nx = (float(x) - bounds.x0) / max(1e-9, bounds.width)
        ny = (float(y) - bounds.y0) / max(1e-9, bounds.height)
        zx = max(0.0, min(map_size, nx * map_size))
        zy = max(0.0, min(map_size, ny * map_size))
        counts[int(zone_of(zx, zy, map_size, num_zones))] += 1
    return counts


def _live_edge_pressure(edge_zone):
    pressure = Counter()
    try:
        vehicle_ids = traci.vehicle.getIDList()
    except traci.TraCIException:
        return pressure
    for veh_id in vehicle_ids:
        try:
            edge_id = traci.vehicle.getRoadID(veh_id)
        except traci.TraCIException:
            continue
        if edge_id not in edge_zone:
            continue
        try:
            wait = float(traci.vehicle.getWaitingTime(veh_id))
            speed = float(traci.vehicle.getSpeed(veh_id))
        except traci.TraCIException:
            wait = 0.0
            speed = 0.0
        pressure[edge_id] += 1.0 + min(8.0, wait / 8.0)
        if speed < 0.3:
            pressure[edge_id] += 4.0
    return pressure


def _position_on_map(veh_id, bounds, map_size):
    x, y = traci.vehicle.getPosition(veh_id)
    if not _valid_sumo_position(x, y):
        return None
    nx = (float(x) - bounds.x0) / max(1e-9, bounds.width)
    ny = (float(y) - bounds.y0) / max(1e-9, bounds.height)
    return (
        max(0.0, min(float(map_size), nx * float(map_size))),
        max(0.0, min(float(map_size), ny * float(map_size))),
    )


def _vehicle_at_boundary_exit(veh_id, exit_side, boundary_edges, bounds, map_size, exit_margin):
    try:
        edge_id = traci.vehicle.getRoadID(veh_id)
    except traci.TraCIException:
        return False
    if edge_id not in boundary_edges.get(exit_side, ()): 
        return False
    try:
        point = _position_on_map(veh_id, bounds, map_size)
    except traci.TraCIException:
        return False
    if point is None:
        return False
    x, y = point
    side_dist = {
        "left": x,
        "right": float(map_size) - x,
        "bottom": y,
        "top": float(map_size) - y,
    }.get(exit_side)
    return side_dist is not None and side_dist <= float(map_size) * float(exit_margin)


def _distance_to_lane_end(veh_id):
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        lane_pos = float(traci.vehicle.getLanePosition(veh_id))
        lane_len = float(traci.lane.getLength(lane_id))
    except traci.TraCIException:
        return None
    return max(0.0, lane_len - lane_pos)


def _apply_intersection_right_of_way(
    junction_in_edges,
    held_vehicle_ids,
    release_state,
    *,
    step: int,
    wait_seconds: float,
    release_steps: int,
    stop_distance: float,
):
    if not junction_in_edges:
        return
    active_ids = set(traci.vehicle.getIDList())
    for veh_id in list(held_vehicle_ids):
        if veh_id in active_ids:
            try:
                traci.vehicle.setSpeed(veh_id, -1.0)
            except traci.TraCIException:
                pass
    held_vehicle_ids.clear()

    for junction_id, in_edges in junction_in_edges.items():
        queues = {}
        for edge_id in in_edges:
            try:
                veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
            except traci.TraCIException:
                continue
            close = []
            for veh_id in veh_ids:
                dist = _distance_to_lane_end(veh_id)
                if dist is None or dist > 2.5 * float(stop_distance):
                    continue
                try:
                    wait = float(traci.vehicle.getWaitingTime(veh_id))
                    speed = float(traci.vehicle.getSpeed(veh_id))
                except traci.TraCIException:
                    continue
                if wait >= float(wait_seconds) or speed < 0.25:
                    close.append((veh_id, dist, wait))
            if close:
                queues[edge_id] = close
        if len(queues) < 3:
            release_state.pop(junction_id, None)
            continue
        state = release_state.get(junction_id)
        if state and state[0] in queues and int(step) <= int(state[1]):
            release_edge = state[0]
        else:
            release_edge = max(
                queues,
                key=lambda edge: (
                    max(wait for _veh, _dist, wait in queues[edge]),
                    len(queues[edge]),
                ),
            )
            release_state[junction_id] = (release_edge, int(step) + int(release_steps))
        for edge_id, rows in queues.items():
            for veh_id, dist, _wait in rows:
                try:
                    if edge_id == release_edge:
                        traci.vehicle.setSpeed(veh_id, -1.0)
                    elif dist <= float(stop_distance):
                        traci.vehicle.setSpeed(veh_id, 0.0)
                        held_vehicle_ids.add(veh_id)
                except traci.TraCIException:
                    pass


def _candidate_via_zones(zones, current_zone, dest_zone, target_counts, zone_load, live_counts, *, num_zones: int, rng):
    if int(num_zones) != 9:
        return []
    central_zones = _central_route_zones(num_zones)
    corner_zones = _corner_route_zones(num_zones)
    out = [
        int(z)
        for z in zones
        if z not in central_zones and z not in {current_zone, dest_zone}
    ]
    out.sort(
        key=lambda z: (
            float(live_counts[z])
            + 0.35 * float(zone_load[z])
            + 2.5 * float(target_counts[z]),
            0 if z in corner_zones else 1,
            rng.random(),
        )
    )
    return out


def _clear_route_accounting(
    veh_id: str,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
) -> None:
    old_zone = target_by_vehicle.pop(veh_id, None)
    if old_zone is not None:
        target_counts[old_zone] -= 1
        if target_counts[old_zone] <= 0:
            del target_counts[old_zone]
    old_load = zone_load_by_vehicle.pop(veh_id, None)
    if old_load:
        for zone, n in old_load.items():
            zone_load[zone] -= n
            if zone_load[zone] <= 0:
                del zone_load[zone]
    exit_by_vehicle.pop(veh_id, None)


def _remember_route_accounting(
    veh_id: str,
    *,
    dest_zone: int,
    counts,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
    exit_side: str | None = None,
) -> None:
    _clear_route_accounting(
        veh_id,
        target_counts,
        zone_load,
        target_by_vehicle,
        zone_load_by_vehicle,
        exit_by_vehicle,
    )
    target_by_vehicle[veh_id] = int(dest_zone)
    target_counts[int(dest_zone)] += 1
    for zone, n in counts.items():
        zone_load[zone] += n
    zone_load_by_vehicle[veh_id] = Counter(counts)
    if exit_side is not None:
        exit_by_vehicle[veh_id] = str(exit_side)


def _assign_route(
    veh_id: str,
    rng: random.Random,
    by_zone,
    edge_zone,
    edge_midpoint,
    zone_capacity,
    boundary_edges,
    route_cache,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
    *,
    num_nodes: int,
    num_zones: int,
    min_distance: int,
    max_distance: int | None,
    open_boundary: bool,
    open_boundary_probability: float,
    open_boundary_margin: float,
    bounds,
    map_size: float,
) -> bool:
    del zone_capacity, num_nodes
    try:
        current = traci.vehicle.getRoadID(veh_id)
        current_lane = traci.vehicle.getLaneID(veh_id)
    except traci.TraCIException:
        return False
    if not current or current.startswith(":"):
        return False

    current_zone = edge_zone.get(current)
    zones = sorted(z for z, edges in by_zone.items() if edges)
    if not zones:
        return False
    zone_passes = [
        _candidate_zones(
            zones,
            current_zone,
            num_zones=num_zones,
            min_distance=min_distance,
            max_distance=max_distance,
        )
    ]
    relaxed = _candidate_zones(
        zones,
        current_zone,
        num_zones=num_zones,
        min_distance=min_distance,
        max_distance=max_distance,
        ignore_min_distance=True,
    )
    if relaxed and set(relaxed) != set(zone_passes[0]):
        zone_passes.append(relaxed)

    blocked: set[str] = set()

    def install_route(edges: list[str], dest_zone: int, exit_side: str | None = None) -> bool:
        try:
            traci.vehicle.setRoute(veh_id, edges)
        except traci.TraCIException:
            return False
        counts = Counter(edge_zone[e] for e in edges if e in edge_zone)
        _remember_route_accounting(
            veh_id,
            dest_zone=int(dest_zone),
            counts=counts,
            target_counts=target_counts,
            zone_load=zone_load,
            target_by_vehicle=target_by_vehicle,
            zone_load_by_vehicle=zone_load_by_vehicle,
            exit_by_vehicle=exit_by_vehicle,
            exit_side=exit_side,
        )
        return True

    if open_boundary and boundary_edges and rng.random() < open_boundary_probability:
        live_counts = _live_zone_counts(bounds, map_size, num_zones)
        sides = _rank_boundary_sides(
            current,
            boundary_edges,
            edge_zone,
            edge_midpoint,
            live_counts,
            {},
            by_zone,
            map_size=map_size,
            num_nodes=1,
            margin=open_boundary_margin,
        )
        for exit_side in sides[:2]:
            dest_edges = [e for e in boundary_edges.get(exit_side, []) if e != current and e not in blocked]
            rng.shuffle(dest_edges)
            for dest in dest_edges[:6]:
                try:
                    edges = _compose_route_edges(current, current_lane, dest, blocked, route_cache)
                except traci.TraCIException:
                    continue
                if edges and install_route(edges, int(edge_zone.get(dest, current_zone or 0)), exit_side=exit_side):
                    return True

    for pass_candidates in zone_passes:
        candidate_order = list(pass_candidates)
        rng.shuffle(candidate_order)
        for dest_zone in candidate_order:
            dest_edges = [e for e in by_zone.get(dest_zone, []) if e != current and e not in blocked]
            rng.shuffle(dest_edges)
            for dest in dest_edges[:6]:
                try:
                    edges = _compose_route_edges(current, current_lane, dest, blocked, route_cache)
                except traci.TraCIException:
                    continue
                if edges and install_route(edges, int(dest_zone), exit_side=None):
                    return True
    return False


def _respawn_vehicle_opposite_side(
    veh_id: str,
    node_idx: int,
    exit_side: str,
    rng: random.Random,
    by_zone,
    edge_zone,
    zone_capacity,
    boundary_edges,
    route_cache,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
    *,
    num_nodes: int,
    num_zones: int,
    bounds,
    map_size: float,
    respawn_seq: int,
) -> tuple[str | None, int]:
    entry_side = _opposite_side(exit_side)
    blocked: set[str] = set()
    live_counts = _live_zone_counts(bounds, map_size, num_zones)
    entries = _rank_boundary_edges(
        entry_side,
        boundary_edges,
        edge_zone,
        live_counts,
        zone_capacity,
        by_zone,
        blocked,
        rng,
        num_nodes=num_nodes,
    )[:12]
    exits = _rank_boundary_edges(
        exit_side,
        boundary_edges,
        edge_zone,
        live_counts,
        zone_capacity,
        by_zone,
        blocked,
        rng,
        num_nodes=num_nodes,
    )[:12]

    for entry in entries:
        for dest in exits:
            if entry == dest:
                continue
            try:
                edges = _compose_route_edges(entry, "", dest, blocked, route_cache)
            except traci.TraCIException:
                continue
            if not edges:
                continue
            respawn_seq += 1
            route_id = f"open_boundary_trace_{node_idx}_{respawn_seq}"
            new_id = f"veh_ob_{node_idx:03d}_{respawn_seq:05d}"
            try:
                traci.route.add(route_id, edges)
                traci.vehicle.add(
                    new_id,
                    route_id,
                    depart="now",
                    departLane="best",
                    departPos="0",
                    departSpeed="max",
                )
            except traci.TraCIException:
                continue
            try:
                traci.vehicle.remove(veh_id)
            except traci.TraCIException:
                pass
            _clear_route_accounting(
                veh_id,
                target_counts,
                zone_load,
                target_by_vehicle,
                zone_load_by_vehicle,
                exit_by_vehicle,
            )
            dest_zone = int(edge_zone.get(dest, 0))
            counts = Counter(edge_zone[e] for e in edges if e in edge_zone)
            _remember_route_accounting(
                new_id,
                dest_zone=dest_zone,
                counts=counts,
                target_counts=target_counts,
                zone_load=zone_load,
                target_by_vehicle=target_by_vehicle,
                zone_load_by_vehicle=zone_load_by_vehicle,
                exit_by_vehicle=exit_by_vehicle,
                exit_side=exit_side,
            )
            return new_id, respawn_seq
    return None, respawn_seq


def _zone_from_trace_position(pos, bounds, map_size: float, num_zones: int) -> int | None:
    if not pos or len(pos) < 2:
        return None
    x, y = float(pos[0]), float(pos[1])
    if not _valid_sumo_position(x, y):
        return None
    nx = (x - bounds.x0) / max(1e-9, bounds.width)
    ny = (y - bounds.y0) / max(1e-9, bounds.height)
    zx = max(0.0, min(float(map_size), nx * float(map_size)))
    zy = max(0.0, min(float(map_size), ny * float(map_size)))
    return int(zone_of(zx, zy, map_size, num_zones))


def _seed_initial_extra_vehicle(
    node_idx: int,
    rng: random.Random,
    by_zone,
    edge_zone,
    zone_capacity,
    route_cache,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
    *,
    num_nodes: int,
    num_zones: int,
    min_distance: int,
    max_distance: int | None,
    respawn_seq: int,
) -> tuple[str | None, int]:
    zones = sorted(z for z, edges in by_zone.items() if edges)
    if not zones:
        return None, respawn_seq
    all_edges = [e for edges in by_zone.values() for e in edges]
    if not all_edges:
        return None, respawn_seq

    blocked: set[str] = set()
    live_counts = Counter()
    edge_pressure = _live_edge_pressure(edge_zone)
    candidates: list[tuple[float, int, str, str, list[str], Counter[int]]] = []
    for start in rng.sample(all_edges, k=min(48, len(all_edges))):
        current_zone = edge_zone.get(start)
        dest_zones = _candidate_zones(
            zones,
            current_zone,
            num_zones=num_zones,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        if not dest_zones:
            dest_zones = _candidate_zones(
                zones,
                current_zone,
                num_zones=num_zones,
                min_distance=min_distance,
                max_distance=max_distance,
                ignore_min_distance=True,
            )
        for _ in range(4):
            if not dest_zones:
                break
            dest_zone = rng.choice(dest_zones)
            dest = rng.choice(by_zone[dest_zone])
            if dest == start or dest in blocked:
                continue
            try:
                edges = _compose_route_edges(start, "", dest, blocked, route_cache)
            except traci.TraCIException:
                continue
            if not edges:
                continue
            score, counts = _score_route(
                edges,
                dest_zone,
                edge_zone,
                target_counts,
                zone_load,
                live_counts,
                zone_capacity,
                by_zone,
                edge_pressure,
                num_nodes=num_nodes,
                num_zones=num_zones,
            )
            candidates.append((score, int(dest_zone), start, dest, edges, counts))
    if not candidates:
        return None, respawn_seq

    candidates.sort(key=lambda item: item[0])
    for _score, dest_zone, _start, _dest, edges, counts in candidates[: min(32, len(candidates))]:
        respawn_seq += 1
        route_id = f"seed_extra_trace_{node_idx}_{respawn_seq}"
        new_id = f"veh_seed_{node_idx:03d}_{respawn_seq:05d}"
        try:
            traci.route.add(route_id, edges)
            traci.vehicle.add(
                new_id,
                route_id,
                depart="now",
                departLane="best",
                departPos="random_free",
                departSpeed="0",
            )
        except traci.TraCIException:
            continue
        _remember_route_accounting(
            new_id,
            dest_zone=int(dest_zone),
            counts=counts,
            target_counts=target_counts,
            zone_load=zone_load,
            target_by_vehicle=target_by_vehicle,
            zone_load_by_vehicle=zone_load_by_vehicle,
            exit_by_vehicle=exit_by_vehicle,
            exit_side=None,
        )
        return new_id, respawn_seq
    return None, respawn_seq


def _respawn_missing_vehicle(
    veh_id: str,
    node_idx: int,
    last_pos,
    rng: random.Random,
    by_zone,
    edge_zone,
    zone_capacity,
    route_cache,
    target_counts,
    zone_load,
    target_by_vehicle,
    zone_load_by_vehicle,
    exit_by_vehicle,
    *,
    num_nodes: int,
    num_zones: int,
    min_distance: int,
    max_distance: int | None,
    bounds,
    map_size: float,
    respawn_seq: int,
) -> tuple[str | None, int]:
    zones = sorted(z for z, edges in by_zone.items() if edges)
    if not zones:
        return None, respawn_seq
    source_zone = _zone_from_trace_position(last_pos, bounds, map_size, num_zones)
    if source_zone not in by_zone or not by_zone.get(source_zone):
        source_zone = target_by_vehicle.get(veh_id)
    source_edges = list(by_zone.get(source_zone, [])) if source_zone is not None else []
    if not source_edges:
        source_edges = [e for edges in by_zone.values() for e in edges]
    if not source_edges:
        return None, respawn_seq

    blocked: set[str] = set()
    live_counts = _live_zone_counts(bounds, map_size, num_zones)
    edge_pressure = _live_edge_pressure(edge_zone)
    candidates: list[tuple[float, int, str, str, list[str], Counter[int]]] = []
    for start in rng.sample(source_edges, k=min(8, len(source_edges))):
        current_zone = edge_zone.get(start)
        dest_zones = _candidate_zones(
            zones,
            current_zone,
            num_zones=num_zones,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        if not dest_zones:
            dest_zones = _candidate_zones(
                zones,
                current_zone,
                num_zones=num_zones,
                min_distance=min_distance,
                max_distance=max_distance,
                ignore_min_distance=True,
            )
        for _ in range(4):
            if not dest_zones:
                break
            dest_zone = rng.choice(dest_zones)
            dest = rng.choice(by_zone[dest_zone])
            if dest == start or dest in blocked:
                continue
            try:
                edges = _compose_route_edges(start, "", dest, blocked, route_cache)
            except traci.TraCIException:
                continue
            if not edges:
                continue
            score, counts = _score_route(
                edges,
                dest_zone,
                edge_zone,
                target_counts,
                zone_load,
                live_counts,
                zone_capacity,
                by_zone,
                edge_pressure,
                num_nodes=num_nodes,
                num_zones=num_zones,
            )
            candidates.append((score, int(dest_zone), start, dest, edges, counts))
    if not candidates:
        return None, respawn_seq

    candidates.sort(key=lambda item: item[0])
    _score, dest_zone, _start, _dest, edges, counts = rng.choice(candidates[: min(8, len(candidates))])
    respawn_seq += 1
    route_id = f"tracked_respawn_trace_{node_idx}_{respawn_seq}"
    new_id = f"veh_rp_{node_idx:03d}_{respawn_seq:05d}"
    try:
        traci.route.add(route_id, edges)
        traci.vehicle.add(
            new_id,
            route_id,
            depart="now",
            departLane="best",
            departPos="0",
            departSpeed="max",
        )
    except traci.TraCIException:
        return None, respawn_seq

    _clear_route_accounting(
        veh_id,
        target_counts,
        zone_load,
        target_by_vehicle,
        zone_load_by_vehicle,
        exit_by_vehicle,
    )
    _remember_route_accounting(
        new_id,
        dest_zone=int(dest_zone),
        counts=counts,
        target_counts=target_counts,
        zone_load=zone_load,
        target_by_vehicle=target_by_vehicle,
        zone_load_by_vehicle=zone_load_by_vehicle,
        exit_by_vehicle=exit_by_vehicle,
        exit_side=None,
    )
    return new_id, respawn_seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sumo-config", default="SUMO/controlled_4zone_300/controlled_4zone_300.sumocfg")
    ap.add_argument("--sumo-net", default="SUMO/controlled_4zone_300/controlled_4zone_300.net.xml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-nodes", type=int, default=180)
    ap.add_argument("--num-zones", type=int, default=9)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--route-min-zone-distance", type=int, default=1)
    ap.add_argument("--route-max-zone-distance", type=int, default=1)
    ap.add_argument("--open-boundary-routing", action="store_true")
    ap.add_argument("--open-boundary-probability", type=float, default=0.45)
    ap.add_argument("--open-boundary-margin", type=float, default=0.12)
    ap.add_argument("--open-boundary-respawn-buffer", type=int, default=2)
    ap.add_argument("--open-boundary-exit-margin", type=float, default=0.035)
    ap.add_argument("--jam-reroute-wait-seconds", type=float, default=25.0)
    ap.add_argument("--intersection-control", action="store_true")
    ap.add_argument("--intersection-wait-seconds", type=float, default=12.0)
    ap.add_argument("--intersection-release-steps", type=int, default=8)
    ap.add_argument("--intersection-stop-distance", type=float, default=24.0)
    ap.add_argument(
        "--static-routes",
        action="store_true",
        help="Trace the routes from the SUMO route file without random OD reassignment",
    )
    ap.add_argument("--out", default="SUMO/visualizer/random_od_seed01_trace.json")
    args = ap.parse_args()

    sumo_config = (ROOT / args.sumo_config).resolve()
    sumo_net = (ROOT / args.sumo_net).resolve()
    bounds = read_net_bounds(str(sumo_net))
    map_size = max(bounds.width, bounds.height)
    by_zone, edge_zone, _edge_shape, edge_midpoint, zone_capacity, boundary_edges, junction_in_edges = _routing_index(
        sumo_net,
        sumo_config,
        num_zones=args.num_zones,
        map_size=map_size,
        boundary_margin=max(0.01, min(0.45, float(args.open_boundary_margin))),
    )
    rng = random.Random(int(args.seed) + 8_271_913)
    route_cache: dict[tuple[str, str], list[str]] = {}
    target_counts: Counter[int] = Counter()
    zone_load: Counter[int] = Counter()
    target_by_vehicle: dict[str, int] = {}
    zone_load_by_vehicle: dict[str, Counter[int]] = {}
    exit_by_vehicle: dict[str, str] = {}
    held_vehicle_ids: set[str] = set()
    release_state: dict[str, tuple[str, int]] = {}
    respawn_seq = 0
    respawn_events: list[dict[str, int | str]] = []
    node_generations = [0 for _ in range(int(args.num_nodes))]

    def record_respawn(
        *, step: int, node_idx: int, old_id: str, new_id: str, reason: str
    ) -> None:
        node_generations[int(node_idx)] += 1
        respawn_events.append(
            {
                "first_step": int(step) + 1,
                "node_idx": int(node_idx),
                "generation": int(node_generations[int(node_idx)]),
                "old_vehicle_id": str(old_id),
                "new_vehicle_id": str(new_id),
                "reason": str(reason),
            }
        )

    traci.start(["sumo", "-c", str(sumo_config), "--seed", str(int(args.seed)), "--time-to-teleport", "-1"])
    try:
        warmup = 0
        while len(traci.vehicle.getIDList()) < args.num_nodes and warmup < 5000:
            traci.simulationStep()
            warmup += 1
            active_count = len(traci.vehicle.getIDList())
            # If all route-file vehicles that SUMO expects are already active,
            # stop waiting and seed any missing logical cars ourselves. This
            # avoids advancing to the end of a 180-vehicle route file before
            # creating the 220/280-car grid cases.
            if active_count > 0 and traci.simulation.getMinExpectedNumber() <= active_count:
                break

        vehicle_ids = sorted(traci.vehicle.getIDList())[: args.num_nodes]
        if len(vehicle_ids) < int(args.num_nodes) and args.static_routes:
            raise RuntimeError(
                f"SUMO route file produced only {len(vehicle_ids)} active vehicles, "
                f"but --num-nodes={int(args.num_nodes)} was requested for static route export"
            )

        # The persistent route file only contains 180 vehicles. For larger grid
        # points, seed additional logical cars directly into SUMO and then let
        # the normal random-OD/respawn code maintain them.
        if len(vehicle_ids) < int(args.num_nodes):
            base_vehicle_ids = list(vehicle_ids)
            missing = int(args.num_nodes) - len(base_vehicle_ids)
            print(
                f"[trace] seeding {missing} additional vehicles "
                f"({len(base_vehicle_ids)} from route file -> {int(args.num_nodes)} requested)",
                flush=True,
            )
            extra_ids: list[str] = []
            attempts = 0
            max_attempts = max(100, missing * 20)
            # Queue the full batch before advancing SUMO. Adding one vehicle
            # per simulation step lets early vehicles finish short routes
            # before dense (60--100 car) batches are fully initialized.
            while len(extra_ids) < missing and attempts < max_attempts:
                node_idx = len(base_vehicle_ids) + len(extra_ids)
                new_id, respawn_seq = _seed_initial_extra_vehicle(
                    node_idx,
                    rng,
                    by_zone,
                    edge_zone,
                    zone_capacity,
                    route_cache,
                    target_counts,
                    zone_load,
                    target_by_vehicle,
                    zone_load_by_vehicle,
                    exit_by_vehicle,
                    num_nodes=int(args.num_nodes),
                    num_zones=args.num_zones,
                    min_distance=args.route_min_zone_distance,
                    max_distance=args.route_max_zone_distance,
                    respawn_seq=respawn_seq,
                )
                if new_id is not None:
                    extra_ids.append(new_id)
                attempts += 1
            for _ in range(200):
                traci.simulationStep()
                active = set(traci.vehicle.getIDList())
                active_extra = [vid for vid in extra_ids if vid in active]
                if len(base_vehicle_ids) + len(active_extra) >= int(args.num_nodes):
                    break
                # Keep inserted extras alive while SUMO is still waiting for
                # crowded departures. Otherwise early short routes can finish
                # before the final delayed vehicle enters the network.
                for vid in active_extra:
                    try:
                        route = list(traci.vehicle.getRoute(vid))
                        route_idx = int(traci.vehicle.getRouteIndex(vid))
                        remaining = len(route) - route_idx - 1 if route and route_idx >= 0 else 0
                    except traci.TraCIException:
                        continue
                    if remaining > ROUTE_REASSIGN_EDGE_BUFFER:
                        continue
                    _assign_route(
                        vid,
                        rng,
                        by_zone,
                        edge_zone,
                        edge_midpoint,
                        zone_capacity,
                        boundary_edges,
                        route_cache,
                        target_counts,
                        zone_load,
                        target_by_vehicle,
                        zone_load_by_vehicle,
                        exit_by_vehicle,
                        num_nodes=int(args.num_nodes),
                        num_zones=args.num_zones,
                        min_distance=args.route_min_zone_distance,
                        max_distance=args.route_max_zone_distance,
                        open_boundary=False,
                        open_boundary_probability=0.0,
                        open_boundary_margin=max(0.01, min(0.45, float(args.open_boundary_margin))),
                        bounds=bounds,
                        map_size=map_size,
                    )
            active = set(traci.vehicle.getIDList())
            active_extra = [vid for vid in extra_ids if vid in active]
            vehicle_ids = base_vehicle_ids + active_extra[: int(args.num_nodes) - len(base_vehicle_ids)]
            if len(vehicle_ids) < int(args.num_nodes):
                fillers = [vid for vid in sorted(active) if vid not in set(vehicle_ids)]
                vehicle_ids.extend(fillers[: int(args.num_nodes) - len(vehicle_ids)])

        vehicle_ids = vehicle_ids[: int(args.num_nodes)]
        if len(vehicle_ids) != int(args.num_nodes):
            raise RuntimeError(
                f"Mobility exporter initialized {len(vehicle_ids)} vehicles, "
                f"but --num-nodes={int(args.num_nodes)} was requested"
            )

        logical_ids = [f"node_{i:03d}" for i in range(len(vehicle_ids))]
        last_pos = {lid: [0.0, 0.0] for lid in logical_ids}
        if not args.static_routes:
            for vid in vehicle_ids:
                _assign_route(
                    vid,
                    rng,
                    by_zone,
                    edge_zone,
                    edge_midpoint,
                    zone_capacity,
                    boundary_edges,
                    route_cache,
                    target_counts,
                    zone_load,
                    target_by_vehicle,
                    zone_load_by_vehicle,
                    exit_by_vehicle,
                    num_nodes=len(vehicle_ids),
                    num_zones=args.num_zones,
                    min_distance=args.route_min_zone_distance,
                    max_distance=args.route_max_zone_distance,
                    open_boundary=bool(args.open_boundary_routing),
                    open_boundary_probability=max(0.0, min(1.0, float(args.open_boundary_probability))),
                    open_boundary_margin=max(0.01, min(0.45, float(args.open_boundary_margin))),
                    bounds=bounds,
                    map_size=map_size,
                )
        traces = {lid: [] for lid in logical_ids}
        for step in range(args.steps + 1):
            if step and step % 25 == 0:
                print(f"[trace] step {step}/{args.steps}", flush=True)
            active = set(traci.vehicle.getIDList())
            for node_idx, vid in enumerate(list(vehicle_ids)):
                lid = logical_ids[node_idx]
                if vid not in active:
                    exit_side = exit_by_vehicle.get(vid)
                    new_id = None
                    if not args.static_routes and args.open_boundary_routing and exit_side:
                        new_id, respawn_seq = _respawn_vehicle_opposite_side(
                            vid,
                            node_idx,
                            exit_side,
                            rng,
                            by_zone,
                            edge_zone,
                            zone_capacity,
                            boundary_edges,
                            route_cache,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                            num_nodes=len(vehicle_ids),
                            num_zones=args.num_zones,
                            bounds=bounds,
                            map_size=map_size,
                            respawn_seq=respawn_seq,
                        )
                    if new_id is None and not args.static_routes:
                        new_id, respawn_seq = _respawn_missing_vehicle(
                            vid,
                            node_idx,
                            last_pos[lid],
                            rng,
                            by_zone,
                            edge_zone,
                            zone_capacity,
                            route_cache,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                            num_nodes=len(vehicle_ids),
                            num_zones=args.num_zones,
                            min_distance=args.route_min_zone_distance,
                            max_distance=args.route_max_zone_distance,
                            bounds=bounds,
                            map_size=map_size,
                            respawn_seq=respawn_seq,
                        )
                    if new_id is not None:
                        record_respawn(
                            step=step,
                            node_idx=node_idx,
                            old_id=vid,
                            new_id=new_id,
                            reason=(
                                "opposite-boundary" if exit_side else "missing"
                            ),
                        )
                        vehicle_ids[node_idx] = new_id
                        traces[lid].append(None)
                        continue
                    else:
                        _clear_route_accounting(
                            vid,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                        )
                        traces[lid].append(last_pos[lid])
                        continue
                try:
                    respawn_frame_pos = None
                    route = list(traci.vehicle.getRoute(vid))
                    idx = int(traci.vehicle.getRouteIndex(vid))
                    remaining = len(route) - idx - 1 if route and idx >= 0 else 0
                    exit_side = exit_by_vehicle.get(vid)
                    if (
                        not args.static_routes
                        and args.open_boundary_routing
                        and exit_side
                        and _vehicle_at_boundary_exit(
                            vid,
                            exit_side,
                            boundary_edges,
                            bounds,
                            map_size,
                            float(args.open_boundary_exit_margin),
                        )
                    ):
                        try:
                            old_x, old_y = traci.vehicle.getPosition(vid)
                            if _valid_sumo_position(old_x, old_y):
                                respawn_frame_pos = [round(float(old_x), 2), round(float(old_y), 2)]
                        except traci.TraCIException:
                            respawn_frame_pos = None
                        new_id, respawn_seq = _respawn_vehicle_opposite_side(
                            vid,
                            node_idx,
                            exit_side,
                            rng,
                            by_zone,
                            edge_zone,
                            zone_capacity,
                            boundary_edges,
                            route_cache,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                            num_nodes=len(vehicle_ids),
                            num_zones=args.num_zones,
                            bounds=bounds,
                            map_size=map_size,
                            respawn_seq=respawn_seq,
                        )
                        if new_id is not None:
                            record_respawn(
                                step=step,
                                node_idx=node_idx,
                                old_id=vid,
                                new_id=new_id,
                                reason="opposite-boundary",
                            )
                            vehicle_ids[node_idx] = new_id
                            traces[lid].append(
                                respawn_frame_pos
                                if respawn_frame_pos is not None
                                else None
                            )
                            continue
                    try:
                        waiting_time = float(traci.vehicle.getWaitingTime(vid))
                    except traci.TraCIException:
                        waiting_time = 0.0
                    jammed = float(args.jam_reroute_wait_seconds) > 0.0 and waiting_time >= float(args.jam_reroute_wait_seconds)
                    near_exit = bool(
                        exit_side
                        and _vehicle_at_boundary_exit(
                            vid,
                            exit_side,
                            boundary_edges,
                            bounds,
                            map_size,
                            float(args.open_boundary_exit_margin),
                        )
                    )
                    if (
                        not args.static_routes
                        and not near_exit
                        and (
                            vid not in target_by_vehicle
                            or jammed
                            or (not exit_side and remaining <= ROUTE_REASSIGN_EDGE_BUFFER)
                        )
                    ):
                        _assign_route(
                            vid,
                            rng,
                            by_zone,
                            edge_zone,
                            edge_midpoint,
                            zone_capacity,
                            boundary_edges,
                            route_cache,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                            num_nodes=len(vehicle_ids),
                            num_zones=args.num_zones,
                            min_distance=args.route_min_zone_distance,
                            max_distance=args.route_max_zone_distance,
                            open_boundary=bool(args.open_boundary_routing),
                            open_boundary_probability=max(0.0, min(1.0, float(args.open_boundary_probability))),
                            open_boundary_margin=max(0.01, min(0.45, float(args.open_boundary_margin))),
                            bounds=bounds,
                            map_size=map_size,
                        )
                    if respawn_frame_pos is not None:
                        last_pos[lid] = respawn_frame_pos
                        traces[lid].append(last_pos[lid])
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    if _valid_sumo_position(x, y):
                        pos = [round(float(x), 2), round(float(y), 2)]
                        last_pos[lid] = pos
                    traces[lid].append(last_pos[lid])
                except traci.TraCIException:
                    if not args.static_routes:
                        new_id, respawn_seq = _respawn_missing_vehicle(
                            vid,
                            node_idx,
                            last_pos[lid],
                            rng,
                            by_zone,
                            edge_zone,
                            zone_capacity,
                            route_cache,
                            target_counts,
                            zone_load,
                            target_by_vehicle,
                            zone_load_by_vehicle,
                            exit_by_vehicle,
                            num_nodes=len(vehicle_ids),
                            num_zones=args.num_zones,
                            min_distance=args.route_min_zone_distance,
                            max_distance=args.route_max_zone_distance,
                            bounds=bounds,
                            map_size=map_size,
                            respawn_seq=respawn_seq,
                        )
                        if new_id is not None:
                            record_respawn(
                                step=step,
                                node_idx=node_idx,
                                old_id=vid,
                                new_id=new_id,
                                reason="missing-after-traci-error",
                            )
                            vehicle_ids[node_idx] = new_id
                            traces[lid].append(None)
                        else:
                            traces[lid].append(last_pos[lid])
            if step < args.steps:
                if bool(args.intersection_control):
                    _apply_intersection_right_of_way(
                        junction_in_edges,
                        held_vehicle_ids,
                        release_state,
                        step=step,
                        wait_seconds=float(args.intersection_wait_seconds),
                        release_steps=int(args.intersection_release_steps),
                        stop_distance=float(args.intersection_stop_distance),
                    )
                traci.simulationStep()
    finally:
        traci.close()

    if len(logical_ids) != int(args.num_nodes) or len(traces) != int(args.num_nodes):
        raise RuntimeError(
            f"Refusing to write undersized mobility trace: "
            f"{len(logical_ids)} logical IDs/{len(traces)} traces for --num-nodes={int(args.num_nodes)}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "sumo_mobility_trace_v3",
        "seed": int(args.seed),
        "max_step": int(args.steps),
        "num_nodes": int(args.num_nodes),
        "num_zones": int(args.num_zones),
        "map_size": float(map_size),
        "sumo_config": str(sumo_config),
        "sumo_net": str(sumo_net),
        "route_min_zone_distance": int(args.route_min_zone_distance),
        "route_max_zone_distance": int(args.route_max_zone_distance),
        "vehicle_ids": logical_ids,
        "actual_vehicle_ids_final": vehicle_ids,
        "open_boundary_routing": bool(args.open_boundary_routing),
        "open_boundary_probability": float(args.open_boundary_probability),
        "open_boundary_margin": float(args.open_boundary_margin),
        "open_boundary_exit_margin": float(args.open_boundary_exit_margin),
        "open_boundary_respawn_buffer": int(args.open_boundary_respawn_buffer),
        "jam_reroute_wait_seconds": float(args.jam_reroute_wait_seconds),
        "intersection_control": bool(args.intersection_control),
        "intersection_wait_seconds": float(args.intersection_wait_seconds),
        "intersection_release_steps": int(args.intersection_release_steps),
        "intersection_stop_distance": float(args.intersection_stop_distance),
        "respawns": int(len(respawn_events)),
        "respawn_events": respawn_events,
        "replacement_semantics": "complete-cold-start-before-first-new-vehicle-frame",
        "traces": traces,
    }
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, out)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
