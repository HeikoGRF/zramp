#!/usr/bin/env python3
"""Export a fixed-identity participant cohort from a LuST3D crop FCD trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from SUMO.luxembourg_real_city.scan_fcd_crops import Frame, iter_fcd
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from scan_fcd_crops import Frame, iter_fcd


def entry_count(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.size == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def stable_tiebreak(vehicle_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{vehicle_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_participants(
    presence: dict[str, np.ndarray],
    *,
    count: int,
    seed: int,
    min_reentering: int,
) -> list[str]:
    if len(presence) < int(count):
        raise ValueError(f"only {len(presence)} vehicles enter the window; requested {count}")

    def rank(vehicle_id: str) -> tuple[int, int, int]:
        mask = presence[vehicle_id]
        return (
            -int(np.sum(mask)),
            -entry_count(mask),
            stable_tiebreak(vehicle_id, seed),
        )

    vehicle_ids = sorted(presence)
    rank_by_id = {vehicle_id: rank(vehicle_id) for vehicle_id in vehicle_ids}
    reentering = sorted(
        (vehicle_id for vehicle_id, mask in presence.items() if entry_count(mask) >= 2),
        key=rank_by_id.__getitem__,
    )
    forced = reentering[: min(int(min_reentering), int(count))]
    selected = list(forced)
    index_by_id = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    masks = np.stack([presence[vehicle_id] for vehicle_id in vehicle_ids]).astype(np.float64)
    available = np.ones((len(vehicle_ids),), dtype=np.bool_)
    if selected:
        available[[index_by_id[vehicle_id] for vehicle_id in selected]] = False

    # Prefer long-lived participants, while greedily filling frames that are
    # underrepresented by the cohort selected so far.
    coverage = (
        np.sum([presence[vehicle_id] for vehicle_id in selected], axis=0).astype(np.int32)
        if selected
        else np.zeros_like(next(iter(presence.values())), dtype=np.int32)
    )
    while len(selected) < int(count):
        weights = 1.0 / (1.0 + coverage.astype(np.float64))
        scores = masks @ weights
        scores[~available] = -np.inf
        best_score = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1.0e-12))
        candidate_index = min(
            (int(index) for index in tied),
            key=lambda index: rank_by_id[vehicle_ids[index]],
        )
        candidate = vehicle_ids[candidate_index]
        selected.append(candidate)
        available[candidate_index] = False
        coverage += presence[candidate].astype(np.int32)
    return selected


def crop_frames(
    fcd: Path,
    *,
    begin: float,
    steps: int,
    sample_period: float,
) -> list[Frame]:
    end = float(begin) + int(steps) * float(sample_period)
    frames = [
        frame
        for frame in iter_fcd(fcd)
        if float(begin) - 1.0e-6 <= frame.time <= end + 1.0e-6
    ]
    expected = int(steps) + 1
    if len(frames) != expected:
        raise ValueError(f"expected {expected} frames in [{begin}, {end}], found {len(frames)}")
    times = np.asarray([frame.time for frame in frames], dtype=np.float64)
    if not np.allclose(np.diff(times), float(sample_period), atol=1.0e-6):
        raise ValueError("FCD frame spacing differs from --sample-period")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcd", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop", default=None)
    parser.add_argument("--begin", type=float, default=28800.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sample-period", type=float, default=5.0)
    parser.add_argument("--participants", type=int, default=20)
    parser.add_argument(
        "--all-participants",
        action="store_true",
        help="Assign a persistent slot to every physical vehicle observed in the crop window.",
    )
    parser.add_argument(
        "--vehicle-ids-from",
        type=Path,
        help="Preserve the ordered vehicle_ids cohort from an existing mobility JSON.",
    )
    parser.add_argument(
        "--collapse-id-prefix",
        action="append",
        default=[],
        metavar="PHYSICAL_PREFIX=LOGICAL_ID",
        help=(
            "Collapse non-overlapping physical SUMO vehicle generations into one "
            "persistent logical identity. May be repeated."
        ),
    )
    parser.add_argument("--min-reentering", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z-reference", type=float, default=230.0)
    args = parser.parse_args()

    collapse_prefixes: list[tuple[str, str]] = []
    for raw in args.collapse_id_prefix:
        if "=" not in str(raw):
            raise ValueError(
                "--collapse-id-prefix must use PHYSICAL_PREFIX=LOGICAL_ID"
            )
        prefix, logical_id = str(raw).split("=", 1)
        if not prefix or not logical_id:
            raise ValueError("collapse prefixes and logical IDs must be non-empty")
        collapse_prefixes.append((prefix, logical_id))
    if len({logical_id for _prefix, logical_id in collapse_prefixes}) != len(
        collapse_prefixes
    ):
        raise ValueError("collapsed logical IDs must be unique")

    def logical_vehicle_id(physical_vehicle_id: str) -> str:
        matches = [
            logical_id
            for prefix, logical_id in collapse_prefixes
            if physical_vehicle_id.startswith(prefix)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"vehicle {physical_vehicle_id!r} matches multiple collapse prefixes"
            )
        return matches[0] if matches else physical_vehicle_id

    manifest = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
    crop_name = str(args.crop or manifest["selected_crop"])
    crop = manifest["crops"][crop_name]
    xmin, ymin, xmax, ymax = [float(value) for value in crop["bounds_sumo_xy_m"]]
    frames = crop_frames(
        args.fcd,
        begin=float(args.begin),
        steps=int(args.steps),
        sample_period=float(args.sample_period),
    )

    n_frames = len(frames)
    positions: dict[str, list[list[float] | None]] = defaultdict(
        lambda: [None for _ in range(n_frames)]
    )
    headings: dict[str, list[float | None]] = defaultdict(
        lambda: [None for _ in range(n_frames)]
    )
    slopes: dict[str, list[float | None]] = defaultdict(
        lambda: [None for _ in range(n_frames)]
    )
    speeds: dict[str, list[float | None]] = defaultdict(
        lambda: [None for _ in range(n_frames)]
    )
    vehicle_types: dict[str, str] = {}
    for frame_index, frame in enumerate(frames):
        inside = (
            (frame.x >= xmin)
            & (frame.x <= xmax)
            & (frame.y >= ymin)
            & (frame.y <= ymax)
        )
        for raw_index in np.flatnonzero(inside):
            index = int(raw_index)
            physical_vehicle_id = frame.ids[index]
            vehicle_id = logical_vehicle_id(physical_vehicle_id)
            if positions[vehicle_id][frame_index] is not None:
                raise RuntimeError(
                    f"multiple physical generations of {vehicle_id!r} are active "
                    f"at SUMO time {frame.time:g}"
                )
            positions[vehicle_id][frame_index] = [
                float(frame.x[index] - xmin),
                float(frame.y[index] - ymin),
                float(frame.z[index] - float(args.z_reference)),
            ]
            headings[vehicle_id][frame_index] = float(frame.angle[index])
            slopes[vehicle_id][frame_index] = float(frame.slope[index])
            speeds[vehicle_id][frame_index] = float(frame.speed[index])
            vehicle_types[vehicle_id] = frame.types[index]

    presence = {
        vehicle_id: np.asarray([point is not None for point in points], dtype=np.bool_)
        for vehicle_id, points in positions.items()
    }
    if args.vehicle_ids_from is not None and bool(args.all_participants):
        raise ValueError("--vehicle-ids-from and --all-participants are mutually exclusive")
    if args.vehicle_ids_from is not None:
        cohort = json.loads(args.vehicle_ids_from.read_text(encoding="utf-8"))
        selected = [str(vehicle_id) for vehicle_id in cohort["vehicle_ids"]]
        missing = sorted(set(selected) - set(presence))
        if missing:
            raise ValueError(
                f"{len(missing)} cohort vehicles are absent from the requested FCD window"
            )
    elif bool(args.all_participants):
        selected = sorted(presence)
    else:
        selected = select_participants(
            presence,
            count=int(args.participants),
            seed=int(args.seed),
            min_reentering=int(args.min_reentering),
        )
    active_counts = np.sum([presence[vehicle_id] for vehicle_id in selected], axis=0)
    participant_summary = []
    for slot, vehicle_id in enumerate(selected):
        mask = presence[vehicle_id]
        entries = entry_count(mask)
        participant_summary.append(
            {
                "slot": int(slot),
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_types.get(vehicle_id, ""),
                "active_frames": int(np.sum(mask)),
                "active_fraction": float(np.mean(mask)),
                "entry_count": int(entries),
                "reenters": bool(entries >= 2),
                "first_active_step": int(np.flatnonzero(mask)[0]),
                "last_active_step": int(np.flatnonzero(mask)[-1]),
            }
        )

    payload = {
        "format": "sumo_crop_mobility_trace_v1",
        "seed": int(args.seed),
        "max_step": int(args.steps),
        "num_nodes": int(len(selected)),
        "num_zones": 1,
        "map_size": float(xmax - xmin),
        "map_width_m": float(xmax - xmin),
        "map_height_m": float(ymax - ymin),
        "sample_period_s": float(args.sample_period),
        "source_begin_s": float(args.begin),
        "source_end_s": float(args.begin) + int(args.steps) * float(args.sample_period),
        "source_fcd": str(args.fcd.resolve()),
        "crop_manifest": str(args.crop_manifest.resolve()),
        "crop_name": crop_name,
        "crop_bounds_sumo_xy_m": [xmin, ymin, xmax, ymax],
        "z_reference_m": float(args.z_reference),
        "vehicle_ids": selected,
        "vehicle_types": {vehicle_id: vehicle_types.get(vehicle_id, "") for vehicle_id in selected},
        "traces": {vehicle_id: positions[vehicle_id] for vehicle_id in selected},
        "heading_traces_deg": {
            vehicle_id: headings[vehicle_id] for vehicle_id in selected
        },
        "slope_traces_deg": {
            vehicle_id: slopes[vehicle_id] for vehicle_id in selected
        },
        "speed_traces_mps": {
            vehicle_id: speeds[vehicle_id] for vehicle_id in selected
        },
        "orientation_semantics": (
            f"Exact SUMO FCD angle and slope at each stored {float(args.sample_period):g}-second frame; "
            "no heading or position interpolation."
        ),
        "active_traces": {
            vehicle_id: presence[vehicle_id].astype(np.int8).tolist()
            for vehicle_id in selected
        },
        "active_counts_by_step": active_counts.astype(np.int32).tolist(),
        "participant_summary": participant_summary,
        "identity_semantics": (
            "Each slot is permanently bound to one exact LuST3D SUMO vehicle ID. "
            "Inactive frames park that vehicle's model; re-entry resumes the same model."
        ),
        "participant_selection": (
            f"ordered physical vehicle cohort preserved from {args.vehicle_ids_from.resolve()}"
            if args.vehicle_ids_from is not None
            else (
                "all physical vehicle IDs observed in the crop window"
                if bool(args.all_participants)
                else f"deterministic fixed cohort of {int(args.participants)} physical IDs"
            )
        ),
        "replacement_semantics": (
            "none; one permanent slot per selected physical ID for the complete window"
        ),
        "collapsed_physical_id_prefixes": [
            {"physical_prefix": prefix, "logical_vehicle_id": logical_id}
            for prefix, logical_id in collapse_prefixes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.output} participants={len(selected)} "
        f"active_min={int(np.min(active_counts))} active_median={float(np.median(active_counts)):.1f} "
        f"active_max={int(np.max(active_counts))} "
        f"reentering={sum(int(row['reenters']) for row in participant_summary)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
