#!/usr/bin/env python3
"""Create controlled phase-shifted traffic-density variants of a mobility trace."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SEQUENCE_FIELDS = (
    "traces",
    "active_traces",
    "heading_traces_deg",
    "slope_traces_deg",
    "speed_traces_mps",
)


def cyclic_shift(values: list[Any], offset: int) -> list[Any]:
    if not values:
        raise ValueError("cannot shift an empty sequence")
    shift = int(offset) % len(values)
    return list(values[shift:]) + list(values[:shift])


def entry_count(values: list[int]) -> int:
    mask = [bool(value) for value in values]
    return int(mask[0]) + sum(
        int(current and not previous)
        for previous, current in zip(mask[:-1], mask[1:])
    )


def augment(payload: dict[str, Any], factor: int, source: Path) -> dict[str, Any]:
    if payload.get("format") != "sumo_crop_mobility_trace_v1":
        raise ValueError("unsupported mobility format")
    if int(factor) <= 1:
        raise ValueError("density factor must be greater than one")

    source_ids = [str(value) for value in payload["vehicle_ids"]]
    frames = int(payload["max_step"]) + 1
    if frames % int(factor):
        raise ValueError("frame count must be divisible by the density factor")
    offsets = [phase * frames // int(factor) for phase in range(int(factor))]

    for field in ("traces", "active_traces"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"source mobility is missing {field}")
    for field in SEQUENCE_FIELDS:
        values_by_id = payload.get(field)
        if values_by_id is None:
            continue
        for vehicle_id in source_ids:
            values = values_by_id.get(vehicle_id)
            if not isinstance(values, list) or len(values) != frames:
                raise ValueError(f"{field} is incomplete for {vehicle_id}")

    result = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            *SEQUENCE_FIELDS,
            "vehicle_ids",
            "vehicle_types",
            "active_counts_by_step",
            "participant_summary",
            "num_nodes",
        }
    }
    result["vehicle_ids"] = []
    result["vehicle_types"] = {}
    for field in SEQUENCE_FIELDS:
        if payload.get(field) is not None:
            result[field] = {}

    source_types = payload.get("vehicle_types", {})
    participant_summary = []
    active_counts = [0 for _ in range(frames)]
    for phase, offset in enumerate(offsets):
        for vehicle_id in source_ids:
            clone_id = f"density{int(factor)}x_phase{phase}:{vehicle_id}"
            result["vehicle_ids"].append(clone_id)
            result["vehicle_types"][clone_id] = str(source_types.get(vehicle_id, ""))
            for field in SEQUENCE_FIELDS:
                if field in result:
                    result[field][clone_id] = cyclic_shift(
                        payload[field][vehicle_id], offset
                    )

            trace = result["traces"][clone_id]
            active = [int(value) for value in result["active_traces"][clone_id]]
            if any(bool(value) != (point is not None) for value, point in zip(active, trace)):
                raise ValueError(f"activity and positions disagree for {clone_id}")
            for step, value in enumerate(active):
                active_counts[step] += int(value)
            active_steps = [step for step, value in enumerate(active) if value]
            if not active_steps:
                raise ValueError(f"clone {clone_id} is never active")
            entries = entry_count(active)
            participant_summary.append(
                {
                    "slot": len(participant_summary),
                    "vehicle_id": clone_id,
                    "vehicle_type": result["vehicle_types"][clone_id],
                    "active_frames": len(active_steps),
                    "active_fraction": len(active_steps) / float(frames),
                    "entry_count": entries,
                    "reenters": bool(entries >= 2),
                    "first_active_step": active_steps[0],
                    "last_active_step": active_steps[-1],
                    "source_vehicle_id": vehicle_id,
                    "phase_offset_steps": int(offset),
                }
            )

    source_counts = [int(value) for value in payload["active_counts_by_step"]]
    if sum(active_counts) != int(factor) * sum(source_counts):
        raise AssertionError("density augmentation did not preserve total activity")

    result["num_nodes"] = len(result["vehicle_ids"])
    result["active_counts_by_step"] = active_counts
    result["participant_summary"] = participant_summary
    result["density_augmentation"] = {
        "factor": int(factor),
        "source_mobility": str(source.resolve()),
        "phase_offsets_steps": offsets,
        "phase_offsets_seconds": [
            float(value) * float(payload["sample_period_s"]) for value in offsets
        ],
        "method": (
            "Independent persistent copies of every observed trajectory, "
            "cyclically phase-shifted by equal offsets."
        ),
        "purpose": (
            "Controlled open-loop vehicle/contact-density scaling with unchanged "
            "routes, map geometry, propagation, and per-trajectory motion."
        ),
        "vehicle_blocker_assumption": (
            "Valid only for experiments with dynamic vehicle blockers disabled."
        ),
    }
    result["identity_semantics"] = (
        "Each augmented slot is permanently bound to one source trajectory and "
        "one fixed cyclic phase; its predictor persists through inactive periods."
    )
    result["participant_selection"] = (
        f"all source participants replicated {int(factor)} times at equal cyclic phases"
    )
    result["replacement_semantics"] = "none; every augmented slot is persistent"
    return result


def atomic_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    payload = {
        "format": "sumo_crop_mobility_trace_v1",
        "max_step": 3,
        "sample_period_s": 1.0,
        "vehicle_ids": ["a"],
        "vehicle_types": {"a": "passenger"},
        "traces": {"a": [[0, 0, 0], None, [1, 0, 0], None]},
        "active_traces": {"a": [1, 0, 1, 0]},
        "heading_traces_deg": {"a": [0.0, None, 10.0, None]},
        "slope_traces_deg": {"a": [0.0, None, 0.0, None]},
        "speed_traces_mps": {"a": [1.0, None, 1.0, None]},
        "active_counts_by_step": [1, 0, 1, 0],
        "num_nodes": 1,
    }
    result = augment(payload, 2, Path("source.json"))
    assert result["num_nodes"] == 2
    assert result["active_counts_by_step"] == [2, 0, 2, 0]
    assert result["density_augmentation"]["phase_offsets_steps"] == [0, 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--factor", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("density augmentation self-test passed")
        return 0
    if args.input is None or args.output is None or args.factor is None:
        parser.error("--input, --output, and --factor are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    with args.input.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    augmented = augment(payload, int(args.factor), args.input)
    atomic_dump(args.output, augmented)
    counts = sorted(int(value) for value in augmented["active_counts_by_step"])
    print(
        f"{args.output} factor={int(args.factor)} nodes={augmented['num_nodes']} "
        f"active_min={counts[0]} active_median={counts[len(counts) // 2]} "
        f"active_max={counts[-1]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

