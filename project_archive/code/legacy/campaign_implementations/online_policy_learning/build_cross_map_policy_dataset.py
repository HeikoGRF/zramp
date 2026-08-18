#!/usr/bin/env python3
"""Create vehicle predictor snapshots and diverse local merge labels.

Each snapshot belongs to one physical vehicle on one source map.  Predictor
models share an initialization within a case, but learn from distinct private
receiver measurements and reach naturally different experience stages.  At
each checkpoint, a bounded set of feasible directed contacts spanning strong,
weak, and randomly varied providers is evaluated with the deployed private-CV
interpolation rule.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from online_policy_learning.local_validation_reward import interpolate_states, validation_quality
from online_policy_learning.online_local_validation_policy import ExactTrajectoryHistory
from model import make_rssi_predictor


TX_POWER_DBM = 23.0
RSSI_MIN_DBM = -120.0
RSSI_MAX_DBM = 15.0
LOSS_MIN_DB = TX_POWER_DBM - RSSI_MAX_DBM
LOSS_MAX_DB = TX_POWER_DBM - RSSI_MIN_DBM


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _make_predictor(
    seed: int,
    device: torch.device,
    *,
    include_time: bool,
) -> nn.Module:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        model = make_rssi_predictor(
            "small",
            input_dim=5 if include_time else 4,
            include_time=include_time,
            time_encoding="learned",
            learned_time_dim=16,
            learned_time_hidden_dim=16,
            learned_time_scale=1000.0,
        ).to(device)
    linear = [module for module in model.modules() if isinstance(module, nn.Linear)]
    with torch.no_grad():
        linear[-1].weight.zero_()
        if linear[-1].bias is None:
            raise ValueError("small predictor has no output bias")
        linear[-1].bias.fill_(1.0)
    return model


def _normalized_loss_target(rssi: np.ndarray) -> np.ndarray:
    loss = TX_POWER_DBM - np.asarray(rssi, dtype=np.float32)
    return ((loss - LOSS_MIN_DB) / (LOSS_MAX_DB - LOSS_MIN_DB)).astype(np.float32)


def _loss_db(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    if int(features.shape[0]) == 0:
        return np.empty((0,), dtype=np.float32)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), 512):
            batch = torch.as_tensor(features[start : start + 512], dtype=torch.float32, device=device)
            values = model(batch).reshape(-1)
            values = torch.clamp(values, 0.0, 1.0)
            outputs.append(values.cpu().numpy())
    normalized = np.concatenate(outputs)
    return normalized * (LOSS_MAX_DB - LOSS_MIN_DB) + LOSS_MIN_DB


def _rmse(model: nn.Module, features: np.ndarray, rssi: np.ndarray, device: torch.device) -> float:
    if not len(features):
        return float("nan")
    truth = TX_POWER_DBM - np.asarray(rssi, dtype=np.float32).reshape(-1)
    prediction = _loss_db(model, features, device)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _fit_snapshot(
    *,
    features: np.ndarray,
    rssi: np.ndarray,
    model_seed: int,
    train_seed: int,
    device: torch.device,
    include_time: bool,
) -> tuple[nn.Module, int]:
    model = _make_predictor(model_seed, device, include_time=include_time)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(_normalized_loss_target(rssi), dtype=torch.float32).reshape(-1, 1)
    count = int(x.shape[0])
    updates = min(160, max(12, int(round(18.0 * math.log2(1.0 + count / 32.0)))))
    batch_size = min(128, count)
    generator = torch.Generator().manual_seed(int(train_seed) & 0x7FFFFFFFFFFFFFFF)
    model.train()
    for _ in range(updates):
        indices = (
            torch.arange(count)
            if count <= batch_size
            else torch.randperm(count, generator=generator)[:batch_size]
        )
        bx = x.index_select(0, indices).to(device)
        by = y.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(bx), by)
        anchor = sum(parameter.square().sum() for parameter in model.parameters())
        (loss + 1.0e-8 * anchor).backward()
        optimizer.step()
    return model, updates


def _bounded_indices(count: int, capacity: int, seed: int) -> np.ndarray:
    if count <= capacity:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return np.sort(rng.choice(count, size=capacity, replace=False))


def _split_indices(
    sample_steps: np.ndarray,
    tx_indices: np.ndarray,
    *,
    identity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets = np.asarray(
        [
            _stable_seed(identity, int(step), int(tx), ordinal) % 10
            for ordinal, (step, tx) in enumerate(zip(sample_steps, tx_indices))
        ],
        dtype=np.int64,
    )
    return np.flatnonzero(buckets <= 6), np.flatnonzero((buckets == 7) | (buckets == 8)), np.flatnonzero(buckets == 9)


def _trajectory(
    features: np.ndarray,
    rssi: np.ndarray,
    *,
    samples_seen: int,
    visits: int,
    predictor_input_dim: int,
    summary_only: bool,
) -> torch.Tensor:
    history = ExactTrajectoryHistory(capacity=256)
    history.append(features, _normalized_loss_target(rssi))
    width = int(predictor_input_dim) + 1
    base = history.tensor(width=width)
    measurement_experience = min(1.0, math.log1p(max(0, samples_seen)) / math.log1p(100_000))
    visit_experience = min(1.0, math.log1p(max(0, visits)) / math.log1p(100))
    if int(base.shape[0]):
        experience = torch.tensor(
            [measurement_experience, visit_experience, 0.0], dtype=torch.float32
        ).repeat(int(base.shape[0]), 1)
        rows = torch.cat((base, experience), dim=1)
    else:
        rows = torch.empty((0, width + 3), dtype=torch.float32)
    summary = torch.zeros((1, width + 3), dtype=torch.float32)
    summary[0, -3:] = torch.tensor(
        [measurement_experience, visit_experience, 1.0], dtype=torch.float32
    )
    return summary if summary_only else torch.cat((summary, rows), dim=0)


def _active_visits(active: np.ndarray, step: int) -> int:
    values = np.asarray(active[: step + 1], dtype=np.bool_)
    if not values.size:
        return 0
    return int(values[0]) + int(np.count_nonzero(values[1:] & ~values[:-1]))


def _quality(features: np.ndarray) -> float:
    return float(validation_quality(features)) if len(features) else 0.0


def _evaluate_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    rssi: np.ndarray,
    device: torch.device,
) -> float:
    if not len(features):
        return 0.0
    model.load_state_dict(state)
    truth = TX_POWER_DBM - np.asarray(rssi, dtype=np.float32).reshape(-1)
    prediction = _loss_db(model, features, device)
    return float(np.mean(np.square(prediction - truth)))


def _label_decision(
    receiver: dict[str, object],
    provider: dict[str, object],
    eval_model: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    receiver_state = receiver["model_state"]
    provider_state = provider["model_state"]
    if not isinstance(receiver_state, dict) or not isinstance(provider_state, dict):
        raise TypeError("snapshot model state is malformed")
    rx_opt_x = np.asarray(receiver["opt_features"], dtype=np.float32)
    rx_opt_y = np.asarray(receiver["opt_rssi"], dtype=np.float32)
    pr_opt_x = np.asarray(provider["opt_features"], dtype=np.float32)
    pr_opt_y = np.asarray(provider["opt_rssi"], dtype=np.float32)
    qa, qb = _quality(rx_opt_x), _quality(pr_opt_x)
    candidates: list[tuple[float, float]] = []
    for alpha in np.linspace(0.0, 1.0, 9):
        state = interpolate_states(receiver_state, provider_state, float(alpha))
        mse_a = _evaluate_state(eval_model, state, rx_opt_x, rx_opt_y, device)
        mse_b = _evaluate_state(eval_model, state, pr_opt_x, pr_opt_y, device)
        objective = (qa * mse_a + qb * mse_b) / max(qa + qb, 1.0e-12)
        candidates.append((float(alpha), float(objective)))
    minimum = min(value for _alpha, value in candidates)
    alpha = max(alpha for alpha, value in candidates if abs(value - minimum) <= 1.0e-9 * max(1.0, abs(minimum)))
    aggregate = interpolate_states(receiver_state, provider_state, alpha)
    reward_x = np.asarray(receiver["reward_features"], dtype=np.float32)
    reward_y = np.asarray(receiver["reward_rssi"], dtype=np.float32)
    before = _evaluate_state(eval_model, receiver_state, reward_x, reward_y, device)
    after = _evaluate_state(eval_model, aggregate, reward_x, reward_y, device)
    target = (before - after) / max(before, 1.0e-12)
    return {
        "target_gain": float(np.clip(target, -1.0, 1.0)),
        "alpha": float(alpha),
        "before_rmse": float(math.sqrt(max(0.0, before))),
        "after_rmse": float(math.sqrt(max(0.0, after))),
        "opt_objective": float(minimum),
    }


def _candidate_providers(
    receiver: int,
    active_nodes: list[int],
    powers: dict[tuple[int, int], float],
    *,
    threshold: float,
    limit: int,
    seed: int,
) -> list[int]:
    feasible = [
        provider for provider in active_nodes
        if provider != receiver
        and powers.get((provider, receiver), -120.0) >= threshold
        and powers.get((receiver, provider), -120.0) >= threshold
    ]
    feasible.sort(key=lambda provider: (-powers[(provider, receiver)], provider))
    if len(feasible) <= limit:
        return feasible
    selected = [feasible[0], feasible[-1]]
    remaining = [provider for provider in feasible if provider not in selected]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    return sorted(set(selected))


def build(args: argparse.Namespace) -> None:
    global RSSI_MIN_DBM, LOSS_MAX_DB
    RSSI_MIN_DBM = float(args.noise_floor_dbm)
    LOSS_MAX_DB = TX_POWER_DBM - RSSI_MIN_DBM
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    mobility = json.loads(args.mobility.read_text(encoding="utf-8"))
    trace = np.load(args.measurements, allow_pickle=False)
    trace_meta = json.loads(str(trace["meta_json"].item()))
    rows = np.asarray(trace["measurements"], dtype=np.float32)
    map_id = str(source["map_id"])
    split = str(source["split"])
    case_seed = int(mobility["seed"])
    num_nodes = int(mobility["num_nodes"])
    map_size = float(mobility["map_size"])
    vehicle_ids = [str(value) for value in mobility["vehicle_ids"]]
    positions = np.stack(
        [np.asarray(mobility["traces"][vehicle], dtype=np.float32)[:, :2] for vehicle in vehicle_ids], axis=1
    )
    active = np.stack(
        [np.asarray(mobility["active_traces"][vehicle], dtype=np.bool_) for vehicle in vehicle_ids], axis=1
    )
    physical = [mobility["physical_id_traces"][vehicle] for vehicle in vehicle_ids]
    by_physical: dict[str, list[tuple[int, int, np.ndarray, float]]] = defaultdict(list)
    powers_by_step: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    for raw in rows:
        step, _zone, tx_idx, rx_idx = (int(round(float(value))) for value in raw[:4])
        rssi = max(float(raw[4]), float(args.noise_floor_dbm))
        powers_by_step[step][(tx_idx, rx_idx)] = rssi
        physical_id = str(physical[rx_idx][step])
        if not physical_id:
            continue
        feature_values = [
            positions[step, tx_idx, 0] / map_size,
            positions[step, tx_idx, 1] / map_size,
            positions[step, rx_idx, 0] / map_size,
            positions[step, rx_idx, 1] / map_size,
        ]
        if args.predictor_time:
            feature_values.append(float(step))
        feature = np.asarray(feature_values, dtype=np.float32)
        by_physical[physical_id].append((step, tx_idx, feature, rssi))
    sample_steps = {
        identity: [row[0] for row in values] for identity, values in by_physical.items()
    }

    device = torch.device(args.device)
    checkpoints = [int(value) for value in str(args.checkpoints).split(",") if value.strip()]
    snapshots: list[dict[str, object]] = []
    snapshot_for_node_step: dict[tuple[int, int], int] = {}
    model_seed = _stable_seed("source-predictor", map_id, case_seed)
    for checkpoint in checkpoints:
        for node_idx in np.flatnonzero(active[checkpoint]).tolist():
            identity = str(physical[node_idx][checkpoint])
            values = by_physical.get(identity, [])
            stop = bisect.bisect_right(sample_steps.get(identity, []), checkpoint)
            observed = values[:stop]
            if len(observed) < int(args.min_observations):
                continue
            steps = np.asarray([row[0] for row in observed], dtype=np.int64)
            tx_indices = np.asarray([row[1] for row in observed], dtype=np.int64)
            features = np.stack([row[2] for row in observed])
            rssi = np.asarray([row[3] for row in observed], dtype=np.float32)
            train_idx, opt_idx, reward_idx = _split_indices(
                steps, tx_indices, identity=f"{map_id}|{case_seed}|{identity}"
            )
            if min(len(train_idx), len(opt_idx), len(reward_idx)) < 8:
                continue
            train_idx = train_idx[_bounded_indices(len(train_idx), int(args.train_capacity), _stable_seed(identity, checkpoint, "train"))]
            opt_idx = opt_idx[_bounded_indices(len(opt_idx), int(args.validation_capacity), _stable_seed(identity, checkpoint, "opt"))]
            reward_idx = reward_idx[_bounded_indices(len(reward_idx), int(args.validation_capacity), _stable_seed(identity, checkpoint, "reward"))]
            model, updates = _fit_snapshot(
                features=features[train_idx],
                rssi=rssi[train_idx],
                model_seed=model_seed,
                train_seed=_stable_seed(map_id, case_seed, identity, checkpoint),
                device=device,
                include_time=bool(args.predictor_time),
            )
            own_rmse = _rmse(model, features[reward_idx], rssi[reward_idx], device)
            snapshot_id = len(snapshots)
            snapshot = {
                "snapshot_id": snapshot_id,
                "map_id": map_id,
                "split": split,
                "case_seed": case_seed,
                "checkpoint": checkpoint,
                "node_idx": int(node_idx),
                "physical_vehicle_id": identity,
                "role": str(mobility["vehicle_roles"][vehicle_ids[node_idx]]),
                "observations_seen": len(observed),
                "train_samples": len(train_idx),
                "optimizer_updates": updates,
                "visits": _active_visits(active[:, node_idx], checkpoint),
                "model_state": {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                },
                "trajectory": _trajectory(
                    features[train_idx], rssi[train_idx],
                    samples_seen=len(train_idx),
                    visits=_active_visits(active[:, node_idx], checkpoint),
                    predictor_input_dim=5 if args.predictor_time else 4,
                    summary_only=bool(args.trajectory_summary_only),
                ),
                "opt_features": features[opt_idx],
                "opt_rssi": rssi[opt_idx],
                "reward_features": features[reward_idx],
                "reward_rssi": rssi[reward_idx],
                "own_reward_rmse": own_rmse,
                "prior_reward_rmse": float(
                    np.sqrt(np.mean(np.square((TX_POWER_DBM - rssi[reward_idx]) - LOSS_MAX_DB)))
                ),
            }
            snapshots.append(snapshot)
            snapshot_for_node_step[(checkpoint, int(node_idx))] = snapshot_id

    eval_model = _make_predictor(
        model_seed, device, include_time=bool(args.predictor_time)
    )
    decisions: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        active_nodes = [
            node for node in np.flatnonzero(active[checkpoint]).tolist()
            if (checkpoint, int(node)) in snapshot_for_node_step
        ]
        powers = powers_by_step.get(checkpoint, {})
        for receiver_idx in active_nodes:
            receiver_snapshot = snapshot_for_node_step[(checkpoint, receiver_idx)]
            providers = _candidate_providers(
                receiver_idx,
                active_nodes,
                powers,
                threshold=float(args.contact_rssi_min),
                limit=int(args.providers_per_receiver),
                seed=_stable_seed(map_id, case_seed, checkpoint, receiver_idx),
            )
            group_id = f"{map_id}|seed{case_seed}|step{checkpoint}|rx{receiver_idx}"
            for provider_idx in providers:
                provider_snapshot = snapshot_for_node_step[(checkpoint, provider_idx)]
                label = _label_decision(
                    snapshots[receiver_snapshot], snapshots[provider_snapshot], eval_model, device
                )
                decisions.append(
                    {
                        "decision_id": len(decisions),
                        "group_id": group_id,
                        "map_id": map_id,
                        "split": split,
                        "case_seed": case_seed,
                        "checkpoint": checkpoint,
                        "receiver_node_idx": receiver_idx,
                        "provider_node_idx": provider_idx,
                        "receiver_snapshot": receiver_snapshot,
                        "provider_snapshot": provider_snapshot,
                        "contact_rssi_dbm": float(powers[(provider_idx, receiver_idx)]),
                        **label,
                    }
                )

    gains = np.asarray([float(row["target_gain"]) for row in decisions], dtype=np.float64)
    snapshot_gain = np.asarray(
        [float(row["prior_reward_rmse"]) - float(row["own_reward_rmse"]) for row in snapshots],
        dtype=np.float64,
    )
    summary = {
        "format": "cross_map_policy_case_v1",
        "map_id": map_id,
        "split": split,
        "case_seed": case_seed,
        "source_trace_format": trace_meta["format"],
        "deployment_map_excluded": str(args.deployment_map_excluded),
        "predictor_time": bool(args.predictor_time),
        "noise_floor_dbm": float(args.noise_floor_dbm),
        "trajectory_summary_only": bool(args.trajectory_summary_only),
        "snapshots": len(snapshots),
        "decisions": len(decisions),
        "decision_groups": len({str(row["group_id"]) for row in decisions}),
        "positive_gain_fraction": float(np.mean(gains > 0.0)) if len(gains) else float("nan"),
        "negative_gain_fraction": float(np.mean(gains < 0.0)) if len(gains) else float("nan"),
        "gain_mean": float(np.mean(gains)) if len(gains) else float("nan"),
        "gain_std": float(np.std(gains)) if len(gains) else float("nan"),
        "predictor_gain_over_prior_mean_db": float(np.mean(snapshot_gain)) if len(snapshot_gain) else float("nan"),
        "policy_inputs": (
            "complete predictor state + shareable experience summary"
            if args.trajectory_summary_only
            else "complete predictor state + chronological private training trajectory + experience summary"
        ),
        "label": "directional normalized receiver reward after joint private-CV alpha selection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cross_map_policy_case_v1",
            "summary": summary,
            "snapshots": snapshots,
            "decisions": decisions,
        },
        args.output,
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", default="50,100,200,350,550,750,1000")
    parser.add_argument("--min-observations", type=int, default=80)
    parser.add_argument("--train-capacity", type=int, default=2048)
    parser.add_argument("--validation-capacity", type=int, default=96)
    parser.add_argument("--providers-per-receiver", type=int, default=6)
    parser.add_argument("--contact-rssi-min", type=float, default=-95.0)
    parser.add_argument(
        "--predictor-time", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--noise-floor-dbm", type=float, default=-120.0)
    parser.add_argument("--trajectory-summary-only", action="store_true")
    parser.add_argument(
        "--deployment-map-excluded",
        default="single_zone_urban_150",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
