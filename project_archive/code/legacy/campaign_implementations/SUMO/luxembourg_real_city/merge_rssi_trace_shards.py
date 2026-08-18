#!/usr/bin/env python3
"""Validate and merge independently generated Sionna RSSI frame shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="step_*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--expected-end", type=int, default=1799)
    parser.add_argument(
        "--min-rssi-dbm",
        type=float,
        default=None,
        help="Retain only measurement rows whose RSSI is at least this value.",
    )
    return parser.parse_args()


def load_meta(archive: np.lib.npyio.NpzFile, path: Path) -> dict:
    if "meta_json" not in archive.files:
        raise ValueError(f"{path}: missing meta_json")
    return json.loads(str(archive["meta_json"].item()))


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


def comparable_config(meta: dict) -> dict:
    keys = (
        "seed",
        "sim_steps",
        "num_zones",
        "traced_zones",
        "cars_per_zone",
        "num_nodes",
        "zone_layout",
        "mitsuba_variant",
        "physical_vehicle_ids",
        "physical_vehicle_types",
        "mobility_source",
        "mobility_start_time_s",
        "mobility_step_length_s",
        "scene_xml",
        "scene_manifest",
        "radio_net",
        "frequency_hz",
        "tx_power_dbm",
        "num_rays",
        "max_depth",
        "tx_batch_size",
        "propagation_phenomena",
        "buildings_opaque",
        "antenna_height_m",
        "vehicle_antenna_model",
        "dynamic_vehicle_blockers",
        "rssi_min_dbm",
        "rssi_max_dbm",
        "measurement_receiver_nodes",
    )
    return {key: meta.get(key) for key in keys}


def main() -> int:
    args = parse_args()
    if args.expected_start < 0 or args.expected_end < args.expected_start:
        raise ValueError("invalid expected frame interval")
    if args.min_rssi_dbm is not None and not np.isfinite(args.min_rssi_dbm):
        raise ValueError("--min-rssi-dbm must be finite")
    paths = sorted(args.shard_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"no shards matching {args.pattern!r} in {args.shard_dir}"
        )

    records: list[tuple[int, int, Path, dict]] = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = load_meta(archive, path)
        except Exception as exc:
            raise ValueError(f"cannot read shard {path}: {exc}") from exc
        start = int(meta.get("start_step", -1))
        end = int(meta.get("end_step", -1))
        if meta.get("format") != "sumo_rssi_trace_v3_shard":
            raise ValueError(f"{path}: expected a v3 shard, got {meta.get('format')!r}")
        if start < 0 or end < start:
            raise ValueError(f"{path}: invalid shard interval {start}..{end}")
        records.append((start, end, path, meta))
    records.sort(key=lambda item: (item[0], item[1], str(item[2])))

    expected = args.expected_start
    reference_meta = records[0][3]
    reference_config = comparable_config(reference_meta)
    for start, end, path, meta in records:
        if start != expected:
            raise ValueError(
                f"frame coverage break before {path}: expected {expected}, got {start}"
            )
        if comparable_config(meta) != reference_config:
            raise ValueError(f"{path}: tracing configuration differs from the first shard")
        expected = end + 1
    if expected != args.expected_end + 1:
        raise ValueError(
            f"frame coverage ends at {expected - 1}, expected {args.expected_end}"
        )

    node_states: list[np.ndarray] = []
    node_generations: list[np.ndarray] = []
    node_active: list[np.ndarray] = []
    synced: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    mesh_vertices: list[np.ndarray] = []
    mesh_faces: list[np.ndarray] = []
    unfiltered_measurement_count = 0
    required = {
        "node_states",
        "node_generations",
        "node_active",
        "synced",
        "measurements",
        "dynamic_mesh_vertices",
        "dynamic_mesh_faces",
    }
    for index, (start, end, path, meta) in enumerate(records, start=1):
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            frame_count = end - start + 1
            state_start = int(meta.get("state_start_step", start))
            if state_start > start:
                raise ValueError(f"{path}: state_start_step is after start_step")
            stored_state_count = end - state_start + 1
            for key in ("node_states", "node_generations", "node_active", "synced"):
                if len(archive[key]) != stored_state_count:
                    raise ValueError(
                        f"{path}: {key} has {len(archive[key])} frames, "
                        f"expected {stored_state_count}"
                    )
            # Legacy shards beginning at step 1 also carry step 0 as bootstrap
            # state. Keep only the shard's declared frame interval when merging.
            state_offset = start - state_start
            state_slice = slice(state_offset, state_offset + frame_count)
            rows = np.asarray(archive["measurements"])
            if (
                int(meta.get("num_zones", 0)) == 1
                and meta.get("measurement_receiver_nodes") is None
            ):
                frame_synced = np.asarray(archive["synced"])[state_slice].astype(np.int64)
                expected_rows = int(np.sum(frame_synced * (frame_synced - 1)))
                if len(rows) != expected_rows:
                    raise ValueError(
                        f"{path}: incomplete one-zone all-pairs shard; "
                        f"found {len(rows)} rows, expected {expected_rows}"
                    )
            unfiltered_measurement_count += len(rows)
            if args.min_rssi_dbm is not None and len(rows):
                rows = rows[
                    rows[:, 4] >= float(args.min_rssi_dbm)
                ]
            if len(rows) and (
                np.min(rows[:, 0]) < start or np.max(rows[:, 0]) > end
            ):
                raise ValueError(f"{path}: measurement step lies outside shard interval")
            node_states.append(np.asarray(archive["node_states"])[state_slice])
            node_generations.append(np.asarray(archive["node_generations"])[state_slice])
            node_active.append(np.asarray(archive["node_active"])[state_slice])
            synced.append(np.asarray(archive["synced"])[state_slice])
            measurements.append(rows)
            mesh_vertices.append(np.asarray(archive["dynamic_mesh_vertices"]))
            mesh_faces.append(np.asarray(archive["dynamic_mesh_faces"]))
        if index % 100 == 0 or index == len(records):
            print(f"loaded {index}/{len(records)} shards", flush=True)

    merged_measurements = np.concatenate(measurements, axis=0)
    if len(merged_measurements):
        order = np.lexsort(
            (
                merged_measurements[:, 3],
                merged_measurements[:, 2],
                merged_measurements[:, 1],
                merged_measurements[:, 0],
            )
        )
        merged_measurements = merged_measurements[order]

    vertices = np.concatenate(mesh_vertices).astype(np.int32, copy=False)
    faces = np.concatenate(mesh_faces).astype(np.int32, copy=False)
    output_meta = dict(reference_meta)
    output_meta.update(
        {
            "format": "sumo_rssi_trace_v3",
            "start_step": int(args.expected_start),
            "end_step": int(args.expected_end),
            "state_start_step": int(args.expected_start),
            "sharded_generation": {
                "shard_count": len(records),
                "pattern": args.pattern,
                "frame_interval": [int(args.expected_start), int(args.expected_end)],
            },
        }
    )
    if args.min_rssi_dbm is not None:
        output_meta["measurement_filter"] = {
            "column": "rssi_dbm",
            "comparison": ">=",
            "minimum_rssi_dbm": float(args.min_rssi_dbm),
            "unfiltered_directed_measurement_count": int(unfiltered_measurement_count),
            "retained_directed_measurement_count": int(len(merged_measurements)),
        }
    geometry = output_meta.get("dynamic_vehicle_geometry")
    if isinstance(geometry, dict) and len(vertices):
        geometry["mesh_vertices_min_median_max"] = [
            int(np.min(vertices)), float(np.median(vertices)), int(np.max(vertices))
        ]
        geometry["mesh_faces_min_median_max"] = [
            int(np.min(faces)), float(np.median(faces)), int(np.max(faces))
        ]

    atomic_save(
        args.output.resolve(),
        meta_json=np.asarray(json.dumps(output_meta, sort_keys=True)),
        node_states=np.concatenate(node_states, axis=0),
        node_generations=np.concatenate(node_generations, axis=0),
        node_active=np.concatenate(node_active, axis=0),
        synced=np.concatenate(synced, axis=0),
        measurements=merged_measurements,
        dynamic_mesh_vertices=vertices,
        dynamic_mesh_faces=faces,
    )
    print(
        f"wrote {args.output} ({args.expected_end - args.expected_start + 1} frames, "
        f"{len(merged_measurements)} directed measurements)",
        flush=True,
    )
    if args.min_rssi_dbm is not None:
        print(
            f"RSSI filter >= {float(args.min_rssi_dbm):g} dBm retained "
            f"{len(merged_measurements)}/{unfiltered_measurement_count} measurements",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
