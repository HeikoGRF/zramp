#!/usr/bin/env python3
"""Generate one time shard of online scalar Sionna RSSI measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_reward_experiment.measurement import RayTracer
from rl_reward_experiment.mobility import zone_of
from SUMO.sumo_sionna_map import SumoNetSionnaMap, sionna_variant_for_net


def _load_mobility(path: Path, num_nodes: int, end_step: int):
    raw = path.read_bytes()
    data = json.loads(raw)
    traces = data.get("traces", {})
    vehicle_ids = [str(value) for value in data.get("vehicle_ids", [])[:num_nodes]]
    if len(vehicle_ids) != num_nodes:
        raise ValueError(f"mobility trace has {len(vehicle_ids)} nodes, expected {num_nodes}")
    positions = np.stack(
        [np.asarray(traces[vehicle_id], dtype=np.float32)[:, :2] for vehicle_id in vehicle_ids],
        axis=1,
    )
    if positions.shape[0] <= end_step:
        raise ValueError(f"mobility trace ends at {positions.shape[0] - 1}, need {end_step}")
    active_payload = data.get("active_traces", {})
    active = np.stack(
        [
            np.asarray(active_payload.get(vehicle_id, [True] * positions.shape[0]), dtype=np.bool_)
            for vehicle_id in vehicle_ids
        ],
        axis=1,
    )
    if active.shape != positions.shape[:2]:
        raise ValueError(f"active mask shape {active.shape} differs from positions {positions.shape[:2]}")
    return data, hashlib.sha256(raw).hexdigest(), vehicle_ids, positions, active


def _save(path: Path, *, meta: dict[str, object], measurements: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
            measurements=np.asarray(measurements, dtype=np.float32).reshape(-1, 5),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(args: argparse.Namespace) -> None:
    if not 1 <= int(args.start_step) <= int(args.end_step) <= int(args.sim_steps):
        raise ValueError("require 1 <= start-step <= end-step <= sim-steps")
    mobility, mobility_hash, vehicle_ids, positions, active = _load_mobility(
        args.mobility.resolve(), int(args.num_nodes), int(args.end_step)
    )
    if int(mobility.get("num_zones", -1)) != int(args.num_zones):
        raise ValueError("mobility num_zones differs from the requested shard")
    map_size = float(mobility["map_size"])
    engine = SumoNetSionnaMap(
        net_path=str(args.sumo_net.resolve()),
        frequency=float(args.freq_hz),
        sionna_variant=sionna_variant_for_net(str(args.sumo_net)),
        dynamic_schedule_path=str(args.dynamic_map.resolve()),
    )
    scene = engine.build()
    tracer = RayTracer(
        scene,
        num_rays=int(args.num_rays),
        max_depth=int(args.max_depth),
        tx_power_dbm=float(args.tx_power_dbm),
        rssi_min=float(args.rssi_min_dbm),
        rssi_max=float(args.rssi_max_dbm),
        tx_batch_size=int(args.trace_tx_batch_size),
    )
    measurement_rows: list[np.ndarray] = []
    dynamic_by_step: dict[str, str] = {}
    try:
        for step in range(int(args.start_step), int(args.end_step) + 1):
            dynamic_by_step[str(step)] = ";".join(engine.apply_dynamic_step(step))
            nodes = [
                SimpleNamespace(x=float(positions[step, index, 0]), y=float(positions[step, index, 1]))
                for index in range(int(args.num_nodes))
            ]
            zone_nodes: dict[int, list[int]] = defaultdict(list)
            for node_idx in range(int(args.num_nodes)):
                if not bool(active[step, node_idx]):
                    continue
                x, y = positions[step, node_idx]
                zone_nodes[int(zone_of(float(x), float(y), map_size, int(args.num_zones)))].append(node_idx)
            measurements = tracer.step_measurements(nodes, dict(zone_nodes))
            if measurements:
                rows = np.empty((len(measurements), 5), dtype=np.float32)
                rows[:, 0] = float(step)
                for row_idx, (zone, tx_idx, rx_idx, rssi) in enumerate(measurements):
                    rows[row_idx, 1:] = (float(zone), float(tx_idx), float(rx_idx), float(rssi))
                measurement_rows.append(rows)
            print(
                f"[scalar shard] step={step}/{args.end_step} active={int(active[step].sum())} "
                f"links={len(measurements)}",
                flush=True,
            )
    finally:
        engine.cleanup()

    packed = (
        np.concatenate(measurement_rows, axis=0)
        if measurement_rows
        else np.zeros((0, 5), dtype=np.float32)
    )
    meta: dict[str, object] = {
        "format": "sumo_rssi_online_shard_v1",
        "seed": int(args.seed),
        "sim_steps": int(args.sim_steps),
        "num_nodes": int(args.num_nodes),
        "num_zones": int(args.num_zones),
        "map_size": map_size,
        "start_step": int(args.start_step),
        "end_step": int(args.end_step),
        "mobility_sha256": mobility_hash,
        "vehicle_ids": vehicle_ids,
        "sumo_config": str(args.sumo_config.resolve()),
        "sumo_net": str(args.sumo_net.resolve()),
        "dynamic_map": str(args.dynamic_map.resolve()),
        "num_rays": int(args.num_rays),
        "max_depth": int(args.max_depth),
        "trace_tx_batch_size": int(args.trace_tx_batch_size),
        "freq_hz": float(args.freq_hz),
        "tx_power_dbm": float(args.tx_power_dbm),
        "rssi_min_dbm": float(args.rssi_min_dbm),
        "rssi_max_dbm": float(args.rssi_max_dbm),
        "dynamic_by_step": dynamic_by_step,
    }
    _save(args.output.resolve(), meta=meta, measurements=packed)
    print(f"Wrote scalar online shard: {args.output} ({packed.shape[0]} measurements)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sumo-config", type=Path, required=True)
    parser.add_argument("--sumo-net", type=Path, required=True)
    parser.add_argument("--dynamic-map", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=20)
    parser.add_argument("--num-zones", type=int, default=1)
    parser.add_argument("--sim-steps", type=int, default=1000)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--num-rays", type=int, default=100_000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--trace-tx-batch-size", type=int, default=20)
    parser.add_argument("--freq-hz", type=float, default=3.5e9)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--rssi-min-dbm", type=float, default=-120.0)
    parser.add_argument("--rssi-max-dbm", type=float, default=15.0)
    generate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
