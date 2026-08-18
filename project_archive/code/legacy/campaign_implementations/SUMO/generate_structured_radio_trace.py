#!/usr/bin/env python3
"""Generate learnable source-map RSSI from actual SUMO vehicle trajectories.

This fast source-only generator is deterministic and geometry based.  It uses
distance, material-dependent building intersections, street orientation, and
the declared periodic buildings.  There is no independent per-sample noise,
so every label is a function of the policy-visible coordinates and time.
Sionna traces can later replace these files without changing the downstream
snapshot/decision format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


MATERIAL_PENALTY_DB = {
    "itu_wood": 4.0,
    "itu_glass": 6.5,
    "itu_brick": 9.0,
    "itu_concrete": 11.0,
    "itu_stone": 13.0,
    "itu_metal": 22.0,
}


def _active(event: dict[str, object], step: int) -> bool:
    period = int(event["period_steps"])
    active_steps = int(event["active_steps"])
    phase = int(event.get("phase_steps", 0))
    return ((int(step) + phase) % period) < active_steps


def _segment_intersects_rect(
    start: tuple[float, float],
    stop: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    """Liang--Barsky line/rectangle test, excluding endpoint-only touches."""

    x0, y0 = start
    x1, y1 = stop
    left, bottom, right, top = bounds
    dx, dy = x1 - x0, y1 - y0
    lower, upper = 0.0, 1.0
    for p, q in (
        (-dx, x0 - left),
        (dx, right - x0),
        (-dy, y0 - bottom),
        (dy, top - y0),
    ):
        if abs(p) < 1.0e-12:
            if q < 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return upper > 1.0e-5 and lower < 1.0 - 1.0e-5


def _load_mobility(path: Path) -> tuple[dict[str, object], bytes, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    data = json.loads(raw)
    vehicle_ids = [str(value) for value in data["vehicle_ids"]]
    positions = np.stack(
        [np.asarray(data["traces"][vehicle], dtype=np.float32)[:, :2] for vehicle in vehicle_ids],
        axis=1,
    )
    active = np.stack(
        [np.asarray(data["active_traces"][vehicle], dtype=np.bool_) for vehicle in vehicle_ids],
        axis=1,
    )
    return data, raw, positions, active


def _structured_rssi(
    *,
    tx: tuple[float, float],
    rx: tuple[float, float],
    step: int,
    map_size: float,
    map_seed: int,
    buildings: list[dict[str, object]],
    active_events: set[str],
    tx_power_dbm: float,
) -> float:
    distance = max(1.0, math.dist(tx, rx))
    x_mid = 0.5 * (tx[0] + rx[0]) / map_size
    y_mid = 0.5 * (tx[1] + rx[1]) / map_size
    angle = math.atan2(rx[1] - tx[1], rx[0] - tx[0])
    loss = 40.1 + 18.5 * math.log10(distance)

    for building in buildings:
        event = building.get("dynamic_event")
        if event is not None and str(event) not in active_events:
            continue
        bounds = tuple(float(value) for value in building["bounds"])
        if _segment_intersects_rect(tx, rx, bounds):
            material = str(building.get("material", "itu_concrete"))
            height = float(building.get("height", 10.0))
            loss += MATERIAL_PENALTY_DB.get(material, 10.0) * (
                0.75 + 0.025 * min(height, 20.0)
            )

    # Smooth map-specific multipath and canyon terms.  They vary between
    # source maps but remain completely determined by positions and time.
    phase = (map_seed % 997) / 997.0 * 2.0 * math.pi
    loss += 4.0 * math.sin(2.0 * math.pi * (1.3 * x_mid + 0.7 * y_mid) + phase)
    loss += 2.2 * math.cos(2.0 * math.pi * (0.4 * x_mid - 1.7 * y_mid) - phase)
    loss += 2.5 * abs(math.sin(2.0 * angle + 0.5 * phase))
    loss += 1.25 * math.sin(
        0.031 * float(step) + 3.0 * tx[0] / map_size - 2.0 * rx[1] / map_size
    )
    return float(np.clip(tx_power_dbm - loss, -120.0, 15.0))


def generate(
    *,
    source_manifest: Path,
    mobility_path: Path,
    output: Path,
    tx_power_dbm: float,
    static: bool = False,
    opaque_buildings: bool = False,
    max_steps: int | None = None,
) -> None:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    mobility, mobility_raw, positions, active = _load_mobility(mobility_path)
    steps = min(int(mobility["max_step"]), int(positions.shape[0]) - 1)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    num_nodes = int(mobility["num_nodes"])
    map_size = float(source["map_size"])
    events = list(source["dynamic_schedule"]["events"])
    rows: list[tuple[float, float, float, float, float]] = []
    dynamic_by_step: dict[str, str] = {}
    for step in range(1, steps + 1):
        active_events = (
            {str(event["id"]) for event in events}
            if static
            else {
                str(event["id"])
                for event in events
                if _active(event, step)
            }
        )
        dynamic_by_step[str(step)] = ";".join(sorted(active_events))
        active_nodes = np.flatnonzero(active[step]).tolist()
        for tx_idx in active_nodes:
            tx = tuple(float(value) for value in positions[step, tx_idx])
            for rx_idx in active_nodes:
                if int(tx_idx) == int(rx_idx):
                    continue
                rx = tuple(float(value) for value in positions[step, rx_idx])
                rssi = _structured_rssi(
                    tx=tx,
                    rx=rx,
                    step=0 if static else step,
                    map_size=map_size,
                    map_seed=int(source["seed"]),
                    buildings=list(source["buildings"]),
                    active_events=active_events,
                    tx_power_dbm=float(tx_power_dbm),
                )
                if opaque_buildings and any(
                    _segment_intersects_rect(
                        tx, rx, tuple(float(value) for value in building["bounds"])
                    )
                    for building in source["buildings"]
                ):
                    rssi = -120.0
                rows.append((float(step), 0.0, float(tx_idx), float(rx_idx), rssi))
    measurements = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    meta = {
        "format": "structured_source_rssi_trace_v1",
        "source_only": True,
        "static_environment": bool(static),
        "opaque_buildings": bool(opaque_buildings),
        "map_id": str(source["map_id"]),
        "map_split": str(source["split"]),
        "map_size": map_size,
        "seed": int(mobility["seed"]),
        "sim_steps": steps,
        "num_nodes": num_nodes,
        "num_zones": 1,
        "vehicle_ids": list(mobility["vehicle_ids"]),
        "mobility_sha256": hashlib.sha256(mobility_raw).hexdigest(),
        "source_manifest": str(source_manifest.resolve()),
        "tx_power_dbm": float(tx_power_dbm),
        "rssi_min_dbm": -120.0,
        "rssi_max_dbm": 15.0,
        "label_function": (
            "distance+material_line_intersections+smooth_map_field;"
            "static_environment;no independent measurement noise"
            if static
            else "distance+material_line_intersections+smooth_map_field+periodic_time;"
            "no independent measurement noise"
        ),
        "dynamic_by_step": dynamic_by_step,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".npz", dir=output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
            measurements=measurements,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": int(measurements.shape[0]),
                "rssi_mean": float(measurements[:, 4].mean()),
                "rssi_std": float(measurements[:, 4].std()),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--opaque-buildings", action="store_true")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    generate(
        source_manifest=args.source_manifest.resolve(),
        mobility_path=args.mobility.resolve(),
        output=args.output.resolve(),
        tx_power_dbm=float(args.tx_power_dbm),
        static=bool(args.static),
        opaque_buildings=bool(args.opaque_buildings),
        max_steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
