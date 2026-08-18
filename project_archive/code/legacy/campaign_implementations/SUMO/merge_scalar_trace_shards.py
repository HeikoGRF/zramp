#!/usr/bin/env python3
"""Merge contiguous online scalar Sionna shards into a replayable base trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_reward_experiment.mobility import zone_of


CONFIG_KEYS = (
    "seed",
    "sim_steps",
    "num_nodes",
    "num_zones",
    "map_size",
    "mobility_sha256",
    "sumo_config",
    "sumo_net",
    "dynamic_map",
    "num_rays",
    "max_depth",
    "trace_tx_batch_size",
    "freq_hz",
    "tx_power_dbm",
    "rssi_min_dbm",
    "rssi_max_dbm",
)


def _load_mobility(path: Path, *, num_nodes: int, sim_steps: int):
    raw = path.read_bytes()
    data = json.loads(raw)
    vehicle_ids = [str(value) for value in data.get("vehicle_ids", [])[:num_nodes]]
    if len(vehicle_ids) != num_nodes:
        raise ValueError("mobility vehicle count differs from shard metadata")
    positions = np.stack(
        [np.asarray(data["traces"][vehicle_id], dtype=np.float32)[: sim_steps + 1, :2] for vehicle_id in vehicle_ids],
        axis=1,
    )
    active = np.stack(
        [
            np.asarray(data.get("active_traces", {}).get(vehicle_id, [True] * (sim_steps + 1)), dtype=np.bool_)[
                : sim_steps + 1
            ]
            for vehicle_id in vehicle_ids
        ],
        axis=1,
    )
    if positions.shape != (sim_steps + 1, num_nodes, 2):
        raise ValueError(f"unexpected mobility shape {positions.shape}")
    if active.shape != positions.shape[:2]:
        raise ValueError(f"unexpected active-mask shape {active.shape}")
    generations = np.zeros((sim_steps + 1, num_nodes), dtype=np.int32)
    for event in data.get("respawn_events", []):
        node_idx = int(event["node_idx"])
        first_step = int(event["first_step"])
        if 0 <= node_idx < num_nodes and first_step <= sim_steps:
            generations[max(0, first_step) :, node_idx] += 1
    return data, hashlib.sha256(raw).hexdigest(), vehicle_ids, positions, active, generations


def _atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge(mobility_path: Path, shard_paths: list[Path], output: Path) -> None:
    if not shard_paths:
        raise ValueError("no shard paths supplied")
    shards: list[tuple[dict[str, object], np.ndarray]] = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
            measurements = np.asarray(data["measurements"], dtype=np.float32).reshape(-1, 5)
        if meta.get("format") != "sumo_rssi_online_shard_v1":
            raise ValueError(f"{path}: unsupported shard format")
        shards.append((meta, measurements))
    shards.sort(key=lambda item: int(item[0]["start_step"]))
    reference = shards[0][0]
    for meta, _measurements in shards[1:]:
        for key in CONFIG_KEYS:
            if meta.get(key) != reference.get(key):
                raise ValueError(f"shard configuration differs for {key}")
    sim_steps = int(reference["sim_steps"])
    expected_start = 1
    dynamic_by_step: dict[str, str] = {}
    rows: list[np.ndarray] = []
    for meta, measurements in shards:
        start = int(meta["start_step"])
        end = int(meta["end_step"])
        if start != expected_start:
            raise ValueError(f"expected shard starting at {expected_start}, got {start}")
        if end < start:
            raise ValueError(f"invalid shard interval {start}-{end}")
        expected_start = end + 1
        if measurements.size:
            measured_steps = measurements[:, 0].astype(np.int32)
            if measured_steps.min() < start or measured_steps.max() > end:
                raise ValueError(f"measurement outside shard interval {start}-{end}")
            rows.append(measurements)
        dynamic_by_step.update({str(key): str(value) for key, value in meta["dynamic_by_step"].items()})
    if expected_start != sim_steps + 1:
        raise ValueError(f"shards end at {expected_start - 1}, expected {sim_steps}")

    num_nodes = int(reference["num_nodes"])
    num_zones = int(reference["num_zones"])
    map_size = float(reference["map_size"])
    mobility, mobility_hash, vehicle_ids, positions, active, generations = _load_mobility(
        mobility_path, num_nodes=num_nodes, sim_steps=sim_steps
    )
    if mobility_hash != reference["mobility_sha256"]:
        raise ValueError("mobility hash differs from online shards")
    if vehicle_ids != list(reference["vehicle_ids"]):
        raise ValueError("vehicle ordering differs from online shards")
    measurements = np.concatenate(rows, axis=0) if rows else np.zeros((0, 5), dtype=np.float32)
    if measurements.size:
        order = np.lexsort(
            (
                measurements[:, 3],
                measurements[:, 2],
                measurements[:, 1],
                measurements[:, 0],
            )
        )
        measurements = measurements[order]
        for row in measurements:
            step, _zone, tx_idx, rx_idx = (int(round(float(value))) for value in row[:4])
            if not active[step, tx_idx] or not active[step, rx_idx]:
                raise ValueError(f"step {step}: shard measurement includes inactive endpoint")

    node_states = np.zeros((sim_steps + 1, num_nodes, 3), dtype=np.float32)
    node_states[:, :, :2] = positions
    for step in range(sim_steps + 1):
        for node_idx in range(num_nodes):
            if active[step, node_idx]:
                node_states[step, node_idx, 2] = float(
                    zone_of(float(positions[step, node_idx, 0]), float(positions[step, node_idx, 1]), map_size, num_zones)
                )
    synced = active.sum(axis=1).astype(np.int32)
    meta: dict[str, object] = {
        "format": "sumo_rssi_trace_v2",
        "seed": int(reference["seed"]),
        "sim_steps": sim_steps,
        "num_nodes": num_nodes,
        "num_zones": num_zones,
        "map_size": map_size,
        "num_rays": int(reference["num_rays"]),
        "max_depth": int(reference["max_depth"]),
        "trace_tx_batch_size": int(reference["trace_tx_batch_size"]),
        "freq_hz": float(reference["freq_hz"]),
        "tx_power_dbm": float(reference["tx_power_dbm"]),
        "rssi_min_dbm": float(reference["rssi_min_dbm"]),
        "rssi_max_dbm": float(reference["rssi_max_dbm"]),
        "sumo_config": str(reference["sumo_config"]),
        "sumo_net": str(reference["sumo_net"]),
        "dynamic_map": str(reference["dynamic_map"]),
        "mobility_trace": str(mobility_path.resolve()),
        "last_step": sim_steps,
        "reason": "sharded_online_complete",
        "replacement_semantics": "complete-cold-start-before-first-new-vehicle-frame",
        "replacement_events": int(np.sum(np.diff(generations, axis=0) > 0)),
        "fidelity_events": [],
        "dynamic_by_step": dynamic_by_step,
        "refresh_zones_by_step": {},
        "sharded_generation": {
            "format": "time-shards-v1",
            "num_shards": len(shards),
            "intervals": [[int(meta["start_step"]), int(meta["end_step"])] for meta, _ in shards],
        },
    }
    arrays = {
        "node_states": node_states,
        "node_generations": generations,
        "synced": synced,
        "measurements": measurements,
        "meta_json": np.asarray(json.dumps(meta, sort_keys=True)),
    }
    _atomic_save(output.resolve(), arrays)
    print(
        f"Merged {len(shards)} online shards into {output} "
        f"({measurements.shape[0]} measurements)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.mobility.resolve(), [path.resolve() for path in args.shards], args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
