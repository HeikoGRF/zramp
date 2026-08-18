#!/usr/bin/env python3
"""Evaluate the constant noise-floor predictor on a censored RSSI test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_TESTSET = Path(
    "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
    "place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers/testset/"
    "place_wallis_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/place_wallis_benchmark/methods/noise_floor/metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--noise-floor-dbm", type=float, default=-100.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error)))


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    testset = args.testset.resolve()
    floor = float(args.noise_floor_dbm)
    with np.load(testset, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["meta_json"].item()))
        truth = np.asarray(archive["rssi_dbm"], dtype=np.float64).reshape(-1)

    if not bool(metadata.get("buildings_opaque", False)):
        raise ValueError("test set does not use opaque buildings")
    if bool(metadata.get("dynamic_vehicle_blockers", True)):
        raise ValueError("test set uses dynamic vehicle blockers")
    if float(metadata["rssi_floor_dbm"]) != floor:
        raise ValueError("test-set censor floor and requested noise floor differ")
    if not np.isfinite(truth).all() or np.any(truth < floor):
        raise ValueError("test RSSI contains invalid values below the censor floor")

    # Values exactly at the censor floor represent non-feasible/undetected links.
    feasible = truth > floor
    infeasible = ~feasible
    if not feasible.any() or not infeasible.any():
        raise ValueError("test set must contain feasible and non-feasible links")

    prediction = np.full_like(truth, floor)
    error = prediction - truth
    payload: dict[str, object] = {
        "schema": "place_wallis_benchmark_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "id": "noise_floor",
            "name": "Noise-floor baseline",
            "description": "Predict -100 dBm for every sender-receiver query.",
            "trainable_parameters": 0,
            "training_steps": 0,
        },
        "testset": {
            "path": str(testset),
            "sha256": file_sha256(testset),
            "samples": int(truth.size),
            "feasible_samples": int(feasible.sum()),
            "non_feasible_samples": int(infeasible.sum()),
            "buildings_opaque": True,
            "dynamic_vehicle_blockers": False,
        },
        "evaluation": {
            "noise_floor_dbm": floor,
            "true_feasible_rule": f"rssi_dbm > {floor:g}",
            "true_non_feasible_rule": f"rssi_dbm <= {floor:g}",
            "prediction_dbm": floor,
            "predicted_feasible_samples": 0,
            "predicted_feasible_fraction": 0.0,
        },
        "metrics_db": {
            "overall_rmse": rmse(error),
            "feasible_rmse": rmse(error[feasible]),
            "non_feasible_rmse": rmse(error[infeasible]),
            "overall_mae": mae(error),
            "feasible_mae": mae(error[feasible]),
            "non_feasible_mae": mae(error[infeasible]),
            "overall_bias": float(np.mean(error)),
            "feasible_bias": float(np.mean(error[feasible])),
            "non_feasible_bias": float(np.mean(error[infeasible])),
        },
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
