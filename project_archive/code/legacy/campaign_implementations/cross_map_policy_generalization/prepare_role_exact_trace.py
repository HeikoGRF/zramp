#!/usr/bin/env python3
"""Prepare the MergeTestMap role traces for exact sequential replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOBILITY = (
    ROOT
    / "cross_map_policy_generalization"
    / "role_mobility_40_seed01"
    / "mobility_roles_40_seed01.npz"
)
DEFAULT_SIONNA = (
    ROOT
    / "cross_map_policy_generalization"
    / "role_sionna_40_seed01"
    / "role_sionna_all_links_seed01.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "cross_map_policy_generalization"
    / "role_sionna_40_seed01"
    / "role_exact_replay_seed01.npz"
)
DEFAULT_NET = (
    ROOT
    / "cross_map_policy_generalization"
    / "role_mobility_40_seed01"
    / "sumo"
    / "merge_online.net.xml"
)
DEFAULT_CONFIG = DEFAULT_NET.with_name("merge_online.sumocfg")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobility", type=Path, default=DEFAULT_MOBILITY)
    parser.add_argument("--sionna", type=Path, default=DEFAULT_SIONNA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map-size-m", type=float, default=114.0)
    parser.add_argument("--evaluation-pairs", type=int, default=4096)
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800, 1000],
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def _fixed_evaluation_indices(
    *,
    steps: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    roles: np.ndarray,
    number: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(steps > 0)
    if int(number) > len(eligible):
        raise ValueError("evaluation pair count exceeds available all-link rows")
    natural = np.sort(rng.choice(eligible, int(number), replace=False))

    role_groups = ["bus", "regular", "crossing"]
    per_role = int(number) // len(role_groups)
    weighted_parts: list[np.ndarray] = []
    for role in role_groups:
        candidates = eligible[roles[rx[eligible]] == role]
        count = per_role + int(role == role_groups[-1]) * (
            int(number) - per_role * len(role_groups)
        )
        weighted_parts.append(
            rng.choice(candidates, count, replace=len(candidates) < count)
        )
    route_weighted = np.sort(np.concatenate(weighted_parts).astype(np.int64))
    return natural, route_weighted


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.mobility, allow_pickle=False) as mobility:
        positions = np.asarray(mobility["positions"], dtype=np.float32)
        radio_active = np.asarray(mobility["radio_active"], dtype=np.bool_)
        generations = np.asarray(mobility["node_generations"], dtype=np.int32)
        roles = np.asarray(mobility["roles"]).astype(str)
        mobility_meta = json.loads(str(mobility["metadata_json"].item()))
    with np.load(args.sionna, allow_pickle=False) as sionna:
        steps = np.asarray(sionna["step"], dtype=np.int32)
        tx = np.asarray(sionna["tx_vehicle_index"], dtype=np.int16)
        rx = np.asarray(sionna["rx_vehicle_index"], dtype=np.int16)
        rssi = np.asarray(sionna["rssi_dbm"], dtype=np.float32)
        feasible = np.asarray(sionna["feasible"], dtype=np.bool_)
        blocked = np.asarray(sionna["direct_path_blocked"], dtype=np.bool_)
        sionna_meta = json.loads(str(sionna["metadata_json"].item()))

    if positions.shape[:2] != radio_active.shape or radio_active.shape != generations.shape:
        raise ValueError("mobility position, active, and generation shapes differ")
    if positions.shape[1] != len(roles):
        raise ValueError("mobility roles do not match vehicle columns")
    if not (
        len(steps) == len(tx) == len(rx) == len(rssi) == len(feasible) == len(blocked)
    ):
        raise ValueError("Sionna link arrays have different lengths")
    frame_count, node_count = positions.shape[:2]
    sim_steps = frame_count - 1
    if int(np.max(steps)) != sim_steps:
        raise ValueError("Sionna and mobility frame ranges differ")

    node_states = np.zeros((frame_count, node_count, 3), dtype=np.float32)
    node_states[:, :, :2] = positions
    measurements = np.column_stack(
        (
            steps.astype(np.float32),
            np.zeros(len(steps), dtype=np.float32),
            tx.astype(np.float32),
            rx.astype(np.float32),
            rssi,
        )
    ).astype(np.float32, copy=False)
    synced = np.count_nonzero(radio_active, axis=1).astype(np.int32)

    rng = np.random.default_rng(int(args.seed))
    natural_indices, route_indices = _fixed_evaluation_indices(
        steps=steps,
        tx=tx,
        rx=rx,
        roles=roles,
        number=int(args.evaluation_pairs),
        rng=rng,
    )

    def features(indices: np.ndarray) -> np.ndarray:
        row_steps = steps[indices]
        tx_xy = positions[row_steps, tx[indices]]
        rx_xy = positions[row_steps, rx[indices]]
        return (
            np.concatenate((tx_xy, rx_xy), axis=1)
            / float(args.map_size_m)
        ).astype(np.float32)

    natural_X = features(natural_indices)
    natural_y = rssi[natural_indices].reshape(-1, 1).astype(np.float32)
    route_X = features(route_indices)
    route_y = rssi[route_indices].reshape(-1, 1).astype(np.float32)

    evaluation_steps = sorted(
        {
            int(step)
            for step in args.evaluation_steps
            if 1 <= int(step) <= sim_steps
        }
    )
    if sim_steps not in evaluation_steps:
        evaluation_steps.append(sim_steps)
        evaluation_steps.sort()
    fidelity_events = [
        {
            "step": int(step),
            "n_pairs": int(args.evaluation_pairs),
            "zones": [0],
            "name": "fixed-natural-all-link",
        }
        for step in evaluation_steps
    ]
    metadata = {
        "format": "sumo_rssi_trace_v3",
        "source_format": "merge_test_map_role_sionna_all_links_v1",
        "seed": int(args.seed),
        "sim_steps": int(sim_steps),
        "num_nodes": int(node_count),
        "num_zones": 1,
        "map_size": float(args.map_size_m),
        "num_rays": int(sionna_meta["num_rays"]),
        "max_depth": int(sionna_meta["max_depth"]),
        "freq_hz": float(sionna_meta["frequency_hz"]),
        "tx_power_dbm": float(sionna_meta["tx_power_dbm"]),
        "rssi_min_dbm": -120.0,
        "rssi_max_dbm": float(sionna_meta["tx_power_dbm"]),
        "noise_floor_dbm": float(sionna_meta["noise_floor_dbm"]),
        "feasible_threshold_dbm": float(sionna_meta["feasible_threshold_dbm"]),
        "sumo_config": str(DEFAULT_CONFIG),
        "sumo_net": str(DEFAULT_NET),
        "mobility_trace": str(args.mobility.resolve()),
        "all_link_trace": str(args.sionna.resolve()),
        "last_step": int(sim_steps),
        "reason": "completed",
        "replacement_semantics": "complete-cold-start-before-first-new-vehicle-frame",
        "replacement_events": int(np.sum(np.diff(generations, axis=0) > 0)),
        "fidelity_events": fidelity_events,
        "dynamic_by_step": {},
        "refresh_zones_by_step": {},
        "local_training_rows": "receiver-side feasible rows only",
        "measurement_rows": "all directed links including unavailable",
        "node_active_semantics": "radio_active; inactive regulars retain state",
        "natural_all_link_evaluation_pairs": int(len(natural_indices)),
        "route_weighted_evaluation_pairs": int(len(route_indices)),
        "route_weighting": "equal receiver-row count for bus, regular, crossing roles",
        "roles_hidden_from_policy": True,
        "mobility_metadata_format": mobility_meta.get("format"),
    }
    arrays: dict[str, np.ndarray] = {
        "node_states": node_states,
        "node_active": radio_active,
        "node_generations": generations,
        "synced": synced,
        "measurements": measurements,
        "meta_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "evaluation_natural_X": natural_X,
        "evaluation_natural_y": natural_y,
        "evaluation_natural_feasible": feasible[natural_indices],
        "evaluation_natural_blocked": blocked[natural_indices],
        "evaluation_route_weighted_X": route_X,
        "evaluation_route_weighted_y": route_y,
        "evaluation_route_weighted_feasible": feasible[route_indices],
        "evaluation_route_weighted_blocked": blocked[route_indices],
    }
    for event_index, _event in enumerate(fidelity_events):
        arrays[f"fid_{event_index:04d}_z0_X"] = natural_X
        arrays[f"fid_{event_index:04d}_z0_y"] = natural_y

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frames": int(frame_count),
                "nodes": int(node_count),
                "measurements": int(len(measurements)),
                "fidelity_events": evaluation_steps,
                "evaluation_pairs": int(args.evaluation_pairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

