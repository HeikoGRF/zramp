#!/usr/bin/env python3
"""Convert a static synthetic source trace into deployment-faithful replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from SUMO.generate_structured_radio_trace import _segment_intersects_rect


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--mobility", type=Path, required=True)
    result.add_argument("--structured", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--contacts", type=Path, required=True)
    result.add_argument("--steps", type=int, default=200)
    result.add_argument("--evaluation-pairs", type=int, default=2048)
    result.add_argument(
        "--evaluation-every",
        type=int,
        default=0,
        help=(
            "Repeat the same fixed, disjoint static-map evaluation set at this "
            "interval; zero records only the final event."
        ),
    )
    result.add_argument(
        "--evaluation-unavailable-fraction", type=float, default=0.25
    )
    result.add_argument("--seed", type=int, default=20260727)
    result.add_argument("--noise-floor-dbm", type=float, default=-105.0)
    result.add_argument("--snr-min-db", type=float, default=5.0)
    return result


def main() -> int:
    args = parser().parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    mobility = json.loads(args.mobility.read_text(encoding="utf-8"))
    vehicle_ids = [str(value) for value in mobility["vehicle_ids"]]
    positions = np.stack(
        [np.asarray(mobility["traces"][vehicle], dtype=np.float32)[:, :2]
         for vehicle in vehicle_ids], axis=1
    )
    active = np.stack(
        [np.asarray(mobility["active_traces"][vehicle], dtype=np.bool_)
         for vehicle in vehicle_ids], axis=1
    )
    steps = min(int(args.steps), int(mobility["max_step"]), int(positions.shape[0]) - 1)
    positions = positions[: steps + 1]
    active = active[: steps + 1]
    node_count = len(vehicle_ids)
    generations = np.zeros((steps + 1, node_count), dtype=np.int32)
    for event in mobility.get("respawn_events", []):
        node = int(event["node_idx"])
        first = int(event["first_step"])
        if 0 <= node < node_count and first <= steps:
            generations[max(0, first):, node] += 1

    with np.load(args.structured, allow_pickle=False) as archive:
        structured_meta = json.loads(str(archive["meta_json"].item()))
        rows = np.asarray(archive["measurements"], dtype=np.float32)
    rows = rows[(rows[:, 0] >= 1) & (rows[:, 0] <= steps)].copy()
    rows[:, 4] = np.maximum(rows[:, 4], float(args.noise_floor_dbm))
    row_steps = rows[:, 0].astype(np.int32)
    tx = rows[:, 2].astype(np.int32)
    rx = rows[:, 3].astype(np.int32)
    unique_direction = np.flatnonzero(tx < rx)
    number = min(int(args.evaluation_pairs), int(unique_direction.size))
    if number < 64:
        raise ValueError("not enough distinct link-time pairs for evaluation")
    threshold = float(args.noise_floor_dbm + args.snr_min_db)
    class_truth = rows[unique_direction, 4]
    reachable = unique_direction[class_truth >= threshold]
    unavailable = unique_direction[class_truth < threshold]
    rng = np.random.default_rng(int(args.seed))
    requested_unavailable = int(
        round(number * float(args.evaluation_unavailable_fraction))
    )
    unavailable_count = min(len(unavailable), requested_unavailable)
    reachable_count = min(len(reachable), number - unavailable_count)
    if reachable_count + unavailable_count < number:
        unavailable_count = min(len(unavailable), number - reachable_count)
    selected = np.concatenate((
        rng.choice(reachable, reachable_count, replace=False),
        rng.choice(unavailable, unavailable_count, replace=False),
    )).astype(np.int64)
    rng.shuffle(selected)
    selected_steps = row_steps[selected]
    selected_tx = tx[selected]
    selected_rx = rx[selected]
    map_size = float(source["map_size"])
    evaluation_X = (
        np.concatenate((positions[selected_steps, selected_tx], positions[selected_steps, selected_rx]), axis=1)
        / map_size
    ).astype(np.float32)
    evaluation_y = rows[selected, 4].reshape(-1, 1).astype(np.float32)
    evaluation_codes = {
        (int(step), min(int(first), int(second)), max(int(first), int(second)))
        for step, first, second in zip(selected_steps, selected_tx, selected_rx)
    }
    training_keep = np.asarray([
        (int(step), min(int(first), int(second)), max(int(first), int(second))) not in evaluation_codes
        for step, first, second in zip(row_steps, tx, rx)
    ], dtype=np.bool_)
    training_rows = rows[training_keep]

    pair_rows = rows[tx < rx]
    pair_steps = pair_rows[:, 0].astype(np.int32)
    pair_tx = pair_rows[:, 2].astype(np.int16)
    pair_rx = pair_rows[:, 3].astype(np.int16)
    reverse_rssi = {
        (int(step), int(first), int(second)): float(value)
        for step, _zone, first, second, value in rows
    }
    pair_feasible = np.asarray([
        float(value) >= threshold
        and reverse_rssi.get((int(step), int(second), int(first)), float(args.noise_floor_dbm)) >= threshold
        for step, _zone, first, second, value in pair_rows
    ], dtype=np.bool_)
    buildings = list(source["buildings"])
    pair_blocked = np.asarray([
        any(
            _segment_intersects_rect(
                tuple(float(value) for value in positions[int(step), int(first)]),
                tuple(float(value) for value in positions[int(step), int(second)]),
                tuple(float(value) for value in building["bounds"]),
            )
            for building in buildings
        )
        for step, first, second in zip(pair_steps, pair_tx, pair_rx)
    ], dtype=np.bool_)

    node_states = np.zeros((steps + 1, node_count, 3), dtype=np.float32)
    node_states[:, :, :2] = positions
    synced = np.count_nonzero(active, axis=1).astype(np.int32)
    if int(args.evaluation_every) > 0:
        fidelity_steps = list(
            range(int(args.evaluation_every), steps + 1, int(args.evaluation_every))
        )
        if not fidelity_steps or fidelity_steps[-1] != steps:
            fidelity_steps.append(steps)
    else:
        fidelity_steps = [steps]
    fidelity_events = [
        {"step": int(step), "n_pairs": len(selected), "zones": [0]}
        for step in fidelity_steps
    ]
    replay_meta = {
        "format": "sumo_rssi_trace_v3",
        "source_format": str(structured_meta["format"]),
        "source_map": str(source["map_id"]),
        "source_split": str(source["split"]),
        "static_environment": True,
        "seed": int(mobility["seed"]),
        "sim_steps": steps,
        "num_nodes": node_count,
        "num_zones": 1,
        "map_size": map_size,
        "tx_power_dbm": float(structured_meta["tx_power_dbm"]),
        "rssi_min_dbm": float(args.noise_floor_dbm),
        "rssi_max_dbm": float(structured_meta["rssi_max_dbm"]),
        "noise_floor_dbm": float(args.noise_floor_dbm),
        "feasible_threshold_dbm": threshold,
        "sumo_config": "",
        "sumo_net": "",
        "mobility_trace": str(args.mobility.resolve()),
        "all_link_trace": str(args.contacts.resolve()),
        "last_step": steps,
        "reason": "completed",
        "replacement_semantics": str(mobility["replacement_semantics"]),
        "replacement_events": int(np.count_nonzero(np.diff(generations, axis=0) > 0)),
        "fidelity_events": fidelity_events,
        "dynamic_by_step": {},
        "refresh_zones_by_step": {},
        "local_training_rows": "receiver-side feasible rows only",
        "measurement_rows": "all directed links excluding held-out link-time pairs",
        "evaluation_split": "held-out undirected link-time pairs removed in both orientations",
        "roles_hidden_from_policy": True,
    }
    replay_arrays = {
        "node_states": node_states,
        "node_active": active,
        "node_generations": generations,
        "synced": synced,
        "measurements": training_rows.astype(np.float32),
        "meta_json": np.asarray(json.dumps(replay_meta, sort_keys=True)),
        "evaluation_natural_X": evaluation_X,
        "evaluation_natural_y": evaluation_y,
        "evaluation_route_weighted_X": evaluation_X,
        "evaluation_route_weighted_y": evaluation_y,
    }
    for event_index, _event_step in enumerate(fidelity_steps):
        replay_arrays[f"fid_{event_index:04d}_z0_X"] = evaluation_X
        replay_arrays[f"fid_{event_index:04d}_z0_y"] = evaluation_y
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **replay_arrays)
    contact_meta = {
        "format": "synthetic_static_building_contact_mask_v1",
        "source_map": str(source["map_id"]),
        "steps": steps,
        "rx_threshold_dbm": threshold,
        "building_count": len(buildings),
        "pair_steps": len(pair_steps),
        "feasible_pair_steps": int(np.count_nonzero(pair_feasible)),
        "clear_feasible_pair_steps": int(np.count_nonzero(pair_feasible & (~pair_blocked))),
        "building_test": "2D direct segment intersects declared footprint",
    }
    args.contacts.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.contacts,
        meta_json=np.asarray(json.dumps(contact_meta, sort_keys=True)),
        step=pair_steps,
        tx_vehicle_index=pair_tx,
        rx_vehicle_index=pair_rx,
        feasible=pair_feasible,
        direct_path_blocked=pair_blocked,
    )
    print(json.dumps({
        "map": source["map_id"],
        "seed": mobility["seed"],
        "steps": steps,
        "training_rows": len(training_rows),
        "evaluation_rows": len(selected),
        "evaluation_unavailable_fraction": float(np.mean(evaluation_y[:, 0] < threshold)),
        "clear_feasible_contacts": int(np.count_nonzero(pair_feasible & (~pair_blocked))),
        "replay": str(args.output),
        "contacts": str(args.contacts),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
