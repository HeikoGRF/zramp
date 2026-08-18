#!/usr/bin/env python3
"""Export persistent regular and cold-start visitor mobility for one seed."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import sumolib
import traci


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "SUMO" / "single_zone_urban_220"
DEFAULT_CONFIG = MAP_DIR / "single_zone_urban_220_roles_seed01.sumocfg"
DEFAULT_PLAN = MAP_DIR / "single_zone_urban_220_roles_seed01.json"
DEFAULT_OUTPUT = MAP_DIR / "single_zone_urban_220_roles_seed01_mobility_v4.json"


@dataclass
class Slot:
    node_idx: int
    role: str
    logical_id: str
    sumo_id: str | None
    generation: int = 0
    trip: int = 0
    next_direction: str = "reverse"
    reenter_step: int | None = None
    awaiting_departure: bool = True
    pending_wait_event: dict[str, object] | None = None


def _stage_position(node_idx: int, regular_count: int, num_nodes: int) -> list[float]:
    if node_idx < regular_count:
        x = 75.0 + 35.0 * node_idx
        y = -15.0
    else:
        visitor_idx = node_idx - regular_count
        visitor_count = max(1, num_nodes - regular_count)
        x = 8.0 + 204.0 * visitor_idx / max(1, visitor_count - 1)
        y = -25.0
    return [round(x, 3), y]


def _add_vehicle(vehicle_id: str, route_id: str) -> None:
    traci.vehicle.add(
        vehicle_id,
        route_id,
        typeID="passenger",
        depart="now",
        departLane="best",
        departPos="0",
        departSpeed="max",
    )


def export(config: Path, plan_path: Path, output: Path, *, steps: int) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seed = int(plan["seed"])
    num_nodes = int(plan["num_vehicle_slots"])
    regular_count = int(plan["regular_count"])
    gates = plan["gates"]
    gate_names = sorted(gates)
    regular_rng = random.Random(int(plan["regular_reentry"]["rng_seed"]))
    visitor_rng = random.Random(int(plan["visitor_replacement"]["rng_seed"]))

    slots: list[Slot] = []
    regular_by_idx: dict[int, dict[str, object]] = {}
    for item in plan["regular_vehicles"]:
        node_idx = int(item["node_idx"])
        regular_by_idx[node_idx] = item
        slots.append(
            Slot(
                node_idx=node_idx,
                role="regular",
                logical_id=f"node_{node_idx:03d}",
                sumo_id=str(item["physical_vehicle_id"]),
            )
        )
    for item in plan["one_time_visitor_slots"]:
        node_idx = int(item["node_idx"])
        slots.append(
            Slot(
                node_idx=node_idx,
                role="one_time_visitor",
                logical_id=f"node_{node_idx:03d}",
                sumo_id=str(item["initial_physical_vehicle_id"]),
            )
        )
    slots.sort(key=lambda slot: slot.node_idx)
    if [slot.node_idx for slot in slots] != list(range(num_nodes)):
        raise ValueError("route plan does not define one slot for every node index")

    vehicle_ids = [slot.logical_id for slot in slots]
    traces = {vehicle_id: [] for vehicle_id in vehicle_ids}
    active_traces = {vehicle_id: [] for vehicle_id in vehicle_ids}
    physical_id_traces = {vehicle_id: [] for vehicle_id in vehicle_ids}
    respawn_events: list[dict[str, object]] = []
    regular_wait_events: list[dict[str, object]] = []

    sumo_binary = sumolib.checkBinary("sumo")
    traci.start(
        [
            sumo_binary,
            "-c",
            str(config),
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
    )
    try:
        for step in range(int(steps) + 1):
            active_ids = set(traci.vehicle.getIDList())

            for slot in slots:
                if (
                    slot.sumo_id is not None
                    and slot.sumo_id not in active_ids
                    and not slot.awaiting_departure
                ):
                    if slot.role == "regular":
                        wait_steps = regular_rng.randint(10, 20)
                        slot.reenter_step = step + wait_steps
                        wait_event: dict[str, object] = {
                            "node_idx": slot.node_idx,
                            "logical_vehicle_id": slot.logical_id,
                            "exit_step": step,
                            "reentry_step": slot.reenter_step,
                            "wait_steps": wait_steps,
                            "next_direction": slot.next_direction,
                        }
                        regular_wait_events.append(wait_event)
                        slot.pending_wait_event = wait_event
                        slot.sumo_id = None
                    else:
                        entry_gate = visitor_rng.choice(gate_names)
                        exit_gate = visitor_rng.choice(
                            [gate for gate in gate_names if gate != entry_gate]
                        )
                        route = traci.simulation.findRoute(
                            str(gates[entry_gate]["entry"]),
                            str(gates[exit_gate]["exit"]),
                            vType="passenger",
                        )
                        if not route.edges:
                            raise RuntimeError(
                                f"no visitor route from {entry_gate} to {exit_gate}"
                            )
                        slot.generation += 1
                        slot.trip += 1
                        route_id = f"visitor_runtime_{slot.node_idx:03d}_{slot.trip:05d}"
                        new_id = f"visitor_{slot.node_idx:03d}_g{slot.generation:05d}"
                        traci.route.add(route_id, list(route.edges))
                        _add_vehicle(new_id, route_id)
                        slot.sumo_id = new_id
                        slot.awaiting_departure = True
                        respawn_events.append(
                            {
                                "node_idx": slot.node_idx,
                                "logical_vehicle_id": slot.logical_id,
                                "physical_vehicle_id": new_id,
                                "generation": slot.generation,
                                "exit_step": step,
                                "first_step": step + 1,
                                "entry_gate": entry_gate,
                                "exit_gate": exit_gate,
                            }
                        )

            for slot in slots[:regular_count]:
                if (
                    slot.sumo_id is None
                    and slot.reenter_step is not None
                    and step >= slot.reenter_step
                ):
                    item = regular_by_idx[slot.node_idx]
                    route_key = f"{slot.next_direction}_route_id"
                    route_id = str(item[route_key])
                    slot.trip += 1
                    new_id = f"regular_{slot.node_idx:03d}_trip{slot.trip:05d}"
                    _add_vehicle(new_id, route_id)
                    slot.sumo_id = new_id
                    slot.awaiting_departure = True
                    slot.reenter_step = None
                    slot.next_direction = (
                        "forward" if slot.next_direction == "reverse" else "reverse"
                    )

            active_ids = set(traci.vehicle.getIDList())
            for slot in slots:
                active = slot.sumo_id is not None and slot.sumo_id in active_ids
                if active:
                    if slot.awaiting_departure and slot.pending_wait_event is not None:
                        request_step = int(slot.pending_wait_event["reentry_step"])
                        exit_step = int(slot.pending_wait_event["exit_step"])
                        slot.pending_wait_event["first_active_step"] = step
                        slot.pending_wait_event["inactive_steps"] = step - exit_step
                        slot.pending_wait_event["insertion_delay_steps"] = step - request_step
                        slot.pending_wait_event = None
                    slot.awaiting_departure = False
                    x, y = traci.vehicle.getPosition(str(slot.sumo_id))
                    position = [round(float(x), 3), round(float(y), 3)]
                else:
                    position = _stage_position(slot.node_idx, regular_count, num_nodes)
                traces[slot.logical_id].append(position)
                active_traces[slot.logical_id].append(bool(active))
                physical_id_traces[slot.logical_id].append(
                    str(slot.sumo_id) if active else None
                )

            if step < int(steps):
                traci.simulationStep()
    finally:
        traci.close()

    payload = {
        "format": "sumo_mobility_trace_v4",
        "seed": seed,
        "max_step": int(steps),
        "num_nodes": num_nodes,
        "num_zones": 1,
        "map_size": 220.0,
        "vehicle_ids": vehicle_ids,
        "vehicle_roles": {slot.logical_id: slot.role for slot in slots},
        "traces": traces,
        "active_traces": active_traces,
        "physical_id_traces": physical_id_traces,
        "respawn_events": respawn_events,
        "regular_wait_events": regular_wait_events,
        "open_boundary_routing": True,
        "replacement_semantics": "visitors-cold-start; regulars-persist-while-inactive",
        "staging": {
            "description": "inactive vehicles are parked below the radio map",
            "regular_y": -15.0,
            "visitor_y": -25.0,
        },
        "visualization_bounds": [0.0, -32.0, 220.0, 220.0],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(output)
    print(
        f"steps={steps} nodes={num_nodes} visitor_respawns={len(respawn_events)} "
        f"regular_waits={len(regular_wait_events)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    export(
        args.config.resolve(),
        args.plan.resolve(),
        args.output.resolve(),
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
