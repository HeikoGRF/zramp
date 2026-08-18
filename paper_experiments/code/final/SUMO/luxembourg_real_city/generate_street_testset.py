#!/usr/bin/env python3
"""Generate a held-out propagation-loss test set from random LuST3D street pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SUMO.luxembourg_real_city.generate_pilot_rssi_trace import (
    build_street_candidates_3d,
    sample_fidelity_pairs_3d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--sumo-net-3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--region-bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--senders", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--street-spacing-m", type=float, default=2.0)
    parser.add_argument("--region-margin-m", type=float, default=0.0)
    parser.add_argument("--min-distance-m", type=float, default=1.0)
    parser.add_argument("--antenna-height-m", type=float, default=1.5)
    parser.add_argument("--num-rays", type=int, default=20_000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--disable-refraction",
        action="store_true",
        help="Make scene surfaces opaque; keep LOS and specular reflections only.",
    )
    parser.add_argument("--tx-batch-size", type=int, default=20)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--rssi-min-dbm", type=float, default=-120.0)
    parser.add_argument("--rssi-max-dbm", type=float, default=0.0)
    return parser.parse_args()


def atomic_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.senders <= 0:
        raise ValueError("--samples and --senders must be positive")
    if args.region_margin_m < 0.0:
        raise ValueError("--region-margin-m cannot be negative")

    manifest = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    measurement_bounds = tuple(
        float(value) for value in manifest["measurement_bounds_local_xy_m"]
    )
    xmin, ymin, xmax, ymax = (float(value) for value in args.region_bounds)
    mxmin, mymin, mxmax, mymax = measurement_bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("--region-bounds must have positive width and height")
    if not (mxmin <= xmin and mymin <= ymin and xmax <= mxmax and ymax <= mymax):
        raise ValueError("--region-bounds must lie inside the scene measurement bounds")
    margin = float(args.region_margin_m)
    if 2.0 * margin >= min(xmax - xmin, ymax - ymin):
        raise ValueError("region margin removes the complete sampling rectangle")

    candidates = build_street_candidates_3d(
        sumo_net=args.sumo_net_3d,
        scene_manifest=manifest,
        spacing_m=float(args.street_spacing_m),
        margin_m=0.0,
        antenna_height_m=float(args.antenna_height_m),
    )
    inside = (
        (candidates[:, 0] >= xmin + margin)
        & (candidates[:, 0] <= xmax - margin)
        & (candidates[:, 1] >= ymin + margin)
        & (candidates[:, 1] <= ymax - margin)
    )
    region_candidates = candidates[inside]
    if len(region_candidates) < 2:
        raise ValueError("the requested region contains fewer than two street candidates")

    pairs = sample_fidelity_pairs_3d(
        region_candidates,
        receiver_candidates=region_candidates,
        n_tx=int(args.senders),
        n_pairs=int(args.samples),
        min_distance_m=float(args.min_distance_m),
        rng=np.random.default_rng(int(args.seed)),
    )

    import sionna.rt as rt
    from rl_reward_experiment.measurement import RayTracer

    scene = rt.load_scene(str(args.scene.resolve()))
    scene.frequency = float(args.frequency_hz)
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    tracer = RayTracer(
        scene,
        num_rays=int(args.num_rays),
        max_depth=int(args.max_depth),
        tx_power_dbm=float(args.tx_power_dbm),
        rssi_min=float(args.rssi_min_dbm),
        rssi_max=float(args.rssi_max_dbm),
        tx_batch_size=int(args.tx_batch_size),
        refraction=not bool(args.disable_refraction),
    )

    grouped: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}
    grouped_indices: dict[tuple[float, float, float], list[int]] = {}
    for index, (tx, rx) in enumerate(pairs):
        grouped.setdefault(tx, []).append(rx)
        grouped_indices.setdefault(tx, []).append(index)
    groups = list(grouped.items())
    traced = tracer.measure_pairs(groups)
    rssi = np.empty((len(pairs), 1), dtype=np.float32)
    for (tx, _receivers), values in zip(groups, traced):
        indices = grouped_indices[tx]
        if len(indices) != len(values):
            raise AssertionError("test-pair tracing returned an unexpected group length")
        rssi[np.asarray(indices, dtype=np.int64), 0] = np.asarray(values, dtype=np.float32)

    positions_xyz_m = np.asarray(
        [[*tx, *rx] for tx, rx in pairs], dtype=np.float32
    )
    positions_xy_m = positions_xyz_m[:, [0, 1, 3, 4]]
    scale = np.asarray(
        [mxmax - mxmin, mymax - mymin, mxmax - mxmin, mymax - mymin],
        dtype=np.float32,
    )
    origin = np.asarray([mxmin, mymin, mxmin, mymin], dtype=np.float32)
    features = (positions_xy_m - origin) / scale
    propagation_loss = float(args.tx_power_dbm) - rssi

    metadata = {
        "format": "sionna_street_propagation_testset_v1",
        "seed": int(args.seed),
        "samples": int(args.samples),
        "candidate_count": int(len(region_candidates)),
        "requested_sender_count": int(args.senders),
        "unique_sender_count": int(len(grouped)),
        "unique_receiver_count": int(len({rx for _tx, rx in pairs})),
        "region_bounds_local_xy_m": [xmin, ymin, xmax, ymax],
        "scene_measurement_bounds_local_xy_m": list(measurement_bounds),
        "street_spacing_m": float(args.street_spacing_m),
        "region_margin_m": margin,
        "min_pair_distance_m": float(args.min_distance_m),
        "position_source": "uniform random points from densified LuST3D passenger lanes",
        "scene": str(args.scene.resolve()),
        "scene_manifest": str(args.scene_manifest.resolve()),
        "sumo_net_3d": str(args.sumo_net_3d.resolve()),
        "frequency_hz": float(args.frequency_hz),
        "tx_power_dbm": float(args.tx_power_dbm),
        "num_rays": int(args.num_rays),
        "max_depth": int(args.max_depth),
        "tx_batch_size": int(args.tx_batch_size),
        "propagation_phenomena": {
            "line_of_sight": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": not bool(args.disable_refraction),
            "diffraction": False,
            "edge_diffraction": False,
        },
        "buildings_opaque": bool(args.disable_refraction),
        "dynamic_vehicle_blockers": False,
        "target": "propagation_loss_db = tx_power_dbm - rssi_dbm",
        "X_columns": ["tx_x_normalized", "tx_y_normalized", "rx_x_normalized", "rx_y_normalized"],
        "positions_xy_m_columns": ["tx_x", "tx_y", "rx_x", "rx_y"],
        "positions_xyz_m_columns": ["tx_x", "tx_y", "tx_z", "rx_x", "rx_y", "rx_z"],
        "rssi_floor_dbm": float(args.rssi_min_dbm),
        "no_path_propagation_loss_ceiling_db": float(args.tx_power_dbm - args.rssi_min_dbm),
    }
    atomic_save(
        args.output.resolve(),
        meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        X=features.astype(np.float32),
        positions_xy_m=positions_xy_m.astype(np.float32),
        positions_xyz_m=positions_xyz_m.astype(np.float32),
        rssi_dbm=rssi,
        propagation_loss_db=propagation_loss.astype(np.float32),
    )
    print(
        f"wrote {args.output} ({len(pairs)} pairs, {len(region_candidates)} street candidates)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
