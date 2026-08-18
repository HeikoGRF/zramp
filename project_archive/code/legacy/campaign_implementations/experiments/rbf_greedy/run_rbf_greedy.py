#!/usr/bin/env python3
"""Greedy all-neighbour MLP sharing with pair-specific RBF support.

Each support prototype is a mergeable four-dimensional sender/receiver
position summary. Raw measurements remain local; only prototype centres,
masses, model parameters, and experience are communicated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_reward_experiment.config import build_config_from_env  # noqa: E402
from SUMO.sumo_rl import SumoT2Simulation  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
    "gare_bonnevoie_right_middle_30min_opaque_buildings_no_vehicle_blockers"
)
DEFAULT_TRACE = (
    DEFAULT_DATA_ROOT
    / "rssi/gare_bonnevoie_vehicles_0745_0815_1s_right_middle_opaque_"
    "no_vehicle_blockers_r20k_d3_llvm.npz"
)
DEFAULT_TESTSET = (
    DEFAULT_DATA_ROOT
    / "testset/right_middle_street_pairs_10000_opaque_no_vehicle_blockers_static.npz"
)
DEFAULT_NET = (
    ROOT
    / "SUMO/luxembourg_real_city/gare_bonnevoie/map/sionna/"
    "gare_bonnevoie_balanced_radio_bounds.net.xml"
)
DEFAULT_RESULTS = ROOT / "artifacts/rbf_greedy/pair_rbf_sigma2_merge3"

PrototypeRow = tuple[float, float, float, float, float]
OriginSummary = tuple[int, tuple[PrototypeRow, ...]]


@dataclass(frozen=True)
class RBFParams:
    sigma_pair_m: float = 2.0
    merge_radius_m: float = 3.0
    confidence_floor: float = 0.0
    eval_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.sigma_pair_m <= 0.0:
            raise ValueError("sigma_pair_m must be positive")
        if self.merge_radius_m <= 0.0:
            raise ValueError("merge_radius_m must be positive")


@dataclass
class PairPrototype:
    center: np.ndarray
    mass: float = 1.0

    def __post_init__(self) -> None:
        value = np.asarray(self.center, dtype=np.float64).reshape(4)
        first = tuple(value[:2].tolist())
        second = tuple(value[2:].tolist())
        self.center = value if first <= second else value[[2, 3, 0, 1]]
        self.mass = float(self.mass)
        if self.mass <= 0.0:
            raise ValueError("prototype mass must be positive")

    @classmethod
    def from_segment(cls, segment: np.ndarray) -> "PairPrototype":
        return cls(np.asarray(segment, dtype=np.float64).reshape(4), 1.0)


@dataclass(frozen=True)
class TrainingParams:
    replay_capacity: int = 4096
    new_data_epochs: int = 2
    replay_batches: int = 8
    recent_replay_batches: int = 4
    recent_window: int = 512
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.new_data_epochs <= 0:
            raise ValueError("new_data_epochs must be positive")
        if not 0 <= self.recent_replay_batches <= self.replay_batches:
            raise ValueError("recent_replay_batches must be within replay_batches")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


class ReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int = 4) -> None:
        self.capacity = int(capacity)
        self.X = np.empty((self.capacity, feature_dim), dtype=np.float32)
        self.y = np.empty((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.next_index = 0

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        for features, target in zip(X, y):
            self.X[self.next_index] = features
            self.y[self.next_index] = target
            self.next_index = (self.next_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def _recent_indices(self, window: int) -> np.ndarray:
        count = min(self.size, max(1, int(window)))
        offsets = np.arange(count, 0, -1, dtype=np.int64)
        return (self.next_index - offsets) % self.capacity

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        *,
        recent_window: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        pool = (
            np.arange(self.size, dtype=np.int64)
            if recent_window is None
            else self._recent_indices(recent_window)
        )
        indices = rng.choice(pool, size=int(batch_size), replace=True)
        return self.X[indices].copy(), self.y[indices].copy()

    def state_dict(self) -> dict[str, np.ndarray | int]:
        indices = self._recent_indices(self.size)
        return {
            "capacity": self.capacity,
            "X": self.X[indices].copy(),
            "y": self.y[indices].copy(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ReplayBuffer":
        X = np.asarray(state["X"], dtype=np.float32)
        y = np.asarray(state["y"], dtype=np.float32).reshape(-1, 1)
        result = cls(int(state["capacity"]), int(X.shape[1]))
        result.add(X, y)
        return result


def serialize_prototypes(
    prototypes: list[PairPrototype],
) -> tuple[PrototypeRow, ...]:
    return tuple((*map(float, row.center), float(row.mass)) for row in prototypes)


def deserialize_prototypes(
    rows: tuple[PrototypeRow, ...],
) -> list[PairPrototype]:
    return [PairPrototype(np.asarray(row[:4]), row[4]) for row in rows]


def prototype_delta(
    previous: tuple[PrototypeRow, ...],
    current: tuple[PrototypeRow, ...],
) -> tuple[PrototypeRow, ...]:
    previous_rows = set(previous)
    return tuple(row for row in current if row not in previous_rows)


def _pair_distance_sq(
    centers: np.ndarray, incoming: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    direct = 0.5 * (
        np.square(centers[:, :2] - incoming[None, :2]).sum(axis=1)
        + np.square(centers[:, 2:] - incoming[None, 2:]).sum(axis=1)
    )
    swapped = 0.5 * (
        np.square(centers[:, :2] - incoming[None, 2:]).sum(axis=1)
        + np.square(centers[:, 2:] - incoming[None, :2]).sum(axis=1)
    )
    return direct, swapped


def merge_prototypes(
    incoming: list[PairPrototype],
    params: RBFParams,
    *,
    remote: bool,
) -> list[PairPrototype]:
    """Greedy radius merging using a four-dimensional spatial hash."""

    if not incoming:
        return []
    radius_sq = float(params.merge_radius_m) ** 2
    cell_size = math.sqrt(2.0) * float(params.merge_radius_m)
    offsets = tuple(product((-1, 0, 1), repeat=4))
    items: list[PairPrototype | None] = []
    buckets: dict[tuple[int, int, int, int], set[int]] = {}

    def cell(center: np.ndarray) -> tuple[int, int, int, int]:
        return tuple(np.floor(center / cell_size).astype(np.int64).tolist())

    def insert(row: PairPrototype) -> None:
        index = len(items)
        items.append(row)
        buckets.setdefault(cell(row.center), set()).add(index)

    for source in incoming:
        current = PairPrototype(source.center.copy(), source.mass)
        while True:
            key = cell(current.center)
            candidates: list[int] = []
            for offset in offsets:
                neighbour = tuple(key[i] + offset[i] for i in range(4))
                candidates.extend(buckets.get(neighbour, ()))
            best_index: int | None = None
            best_distance = float("inf")
            for index in candidates:
                existing = items[index]
                if existing is None:
                    continue
                distance_sq = 0.5 * float(
                    np.square(existing.center - current.center).sum()
                )
                if distance_sq <= radius_sq and distance_sq < best_distance:
                    best_index = index
                    best_distance = distance_sq
            if best_index is None:
                insert(current)
                break
            existing = items[best_index]
            assert existing is not None
            buckets[cell(existing.center)].remove(best_index)
            items[best_index] = None
            total_mass = float(existing.mass + current.mass)
            center = (
                existing.mass * existing.center + current.mass * current.center
            ) / total_mass
            mass = max(existing.mass, current.mass) if remote else total_mass
            current = PairPrototype(center, mass)

    return [row for row in items if row is not None]


def add_prototype(
    prototypes: list[PairPrototype],
    incoming: PairPrototype,
    params: RBFParams,
    *,
    remote: bool,
) -> None:
    prototypes[:] = merge_prototypes(
        [*prototypes, incoming], params, remote=remote
    )


def prototype_union(
    prototype_sets: list[tuple[PrototypeRow, ...]],
    params: RBFParams,
) -> tuple[PrototypeRow, ...]:
    incoming = [
        prototype
        for rows in prototype_sets
        for prototype in deserialize_prototypes(rows)
    ]
    incoming.sort(key=lambda row: (*row.center.tolist(), float(row.mass)))
    return serialize_prototypes(merge_prototypes(incoming, params, remote=True))


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
    experience: list[int],
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot average an empty model list")
    weights = torch.as_tensor(
        [max(0, int(value)) for value in experience],
        dtype=torch.float64,
    )
    if float(weights.sum()) <= 0.0:
        weights.fill_(1.0)
    weights /= weights.sum()
    result: dict[str, torch.Tensor] = {}
    for name in states[0]:
        first = states[0][name]
        if not (first.is_floating_point() or first.is_complex()):
            result[name] = first.detach().clone()
            continue
        value = torch.zeros_like(first)
        for weight, state in zip(weights.tolist(), states):
            value.add_(state[name], alpha=float(weight))
        result[name] = value
    return result


class PairRBFGatedMLP(nn.Module):
    """Train on feasible links and gate inference by pair-space RBF support."""

    def __init__(
        self,
        base: nn.Module,
        *,
        map_size_m: float,
        floor_prior_norm: float,
        rbf_params: RBFParams,
    ) -> None:
        super().__init__()
        self.base = base
        self.map_size_m = float(map_size_m)
        self.floor_prior_norm = float(floor_prior_norm)
        self.rbf_params = rbf_params
        self._prototype_rows: tuple[PrototypeRow, ...] = ()
        self._prototype_tree: cKDTree | None = None

    @property
    def prototype_rows(self) -> tuple[PrototypeRow, ...]:
        return self._prototype_rows

    def set_prototypes(self, rows: tuple[PrototypeRow, ...]) -> None:
        self._prototype_rows = tuple(rows)
        if not rows:
            self._prototype_tree = None
            return
        values = np.asarray(rows, dtype=np.float64)
        centers = values[:, :4]
        swapped = centers[:, [2, 3, 0, 1]]
        self._prototype_tree = cKDTree(np.concatenate((centers, swapped), axis=0))

    def _confidence_chunk(self, x: torch.Tensor) -> torch.Tensor:
        if self._prototype_tree is None:
            return torch.zeros((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
        query = (
            x[:, :4].detach().cpu().numpy().astype(np.float64, copy=False)
            * self.map_size_m
        )
        euclidean_distance, _index = self._prototype_tree.query(query, k=1)
        # d_pair^2 = (d_tx^2 + d_rx^2) / 2, while cKDTree returns
        # sqrt(d_tx^2 + d_rx^2) in the four-dimensional coordinate space.
        confidence = np.exp(
            -np.square(euclidean_distance)
            / (4.0 * float(self.rbf_params.sigma_pair_m) ** 2)
        )
        return torch.as_tensor(
            confidence.reshape(-1, 1), dtype=x.dtype, device=x.device
        ).clamp(min=float(self.rbf_params.confidence_floor), max=1.0)

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        chunk = max(1, int(self.rbf_params.eval_chunk_size))
        return torch.cat(
            [self._confidence_chunk(x[start : start + chunk])
             for start in range(0, int(x.shape[0]), chunk)],
            dim=0,
        )

    def forward_with_confidence(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base(x)
        confidence = self.confidence(x)
        prior = torch.full_like(raw, self.floor_prior_norm)
        return prior + confidence * (raw - prior), confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.base(x)
        prediction, _confidence = self.forward_with_confidence(x)
        return prediction


class RBFGreedySimulation(SumoT2Simulation):
    checkpoint_format = "rbf_greedy_checkpoint_v1"

    def __init__(
        self,
        cfg,
        *,
        testset: Path,
        reception_floor_dbm: float,
        rbf_params: RBFParams,
        training_params: TrainingParams,
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._testset_path = Path(testset).resolve()
        self.reception_floor_dbm = float(reception_floor_dbm)
        self.rbf_params = rbf_params
        self.training_params = training_params
        self._replay_buffers: dict[int, ReplayBuffer] = {}
        self._staged_measurements: list[tuple[int, int, int, float]] | None = None
        self._support_knowledge: list[dict[int, OriginSummary]] = []
        self._resume_step = 0
        self._resume_payload: dict[str, Any] | None = None
        self._resume_logs_restored = True
        super().__init__(cfg, aux_baselines="greedy", **kwargs)

        floor_loss_db = float(cfg.tx_power_dbm) - float(cfg.noise_floor_dbm)
        floor_prior_norm = float(
            np.clip(
                self._norm_loss_db(
                    np.asarray([floor_loss_db], dtype=np.float32)
                )[0],
                0.0,
                1.0,
            )
        )
        wrapped: list[PairRBFGatedMLP] = []
        wrapped_opts: list[optim.Optimizer] = []
        for base in self.greedy_models:
            model = PairRBFGatedMLP(
                base,
                map_size_m=float(cfg.map_size),
                floor_prior_norm=floor_prior_norm,
                rbf_params=rbf_params,
            ).to(self.aux_device)
            wrapped.append(model)
            wrapped_opts.append(optim.Adam(model.parameters(), lr=cfg.local_lr))
        self.greedy_models = wrapped
        self.greedy_opts = wrapped_opts
        self._aux_template_state = {
            name: value.detach().clone()
            for name, value in wrapped[0].state_dict().items()
        }
        self._support_knowledge = [
            {index: (0, ())} for index in range(int(cfg.num_nodes))
        ]
        self._communication_assumptions.update(
            {
                "rbf_support": "origin-versioned reciprocal 4-D pair prototypes",
                "rbf_raw_samples_shared": False,
                "rbf_params": asdict(rbf_params),
                "local_training": {
                    **asdict(training_params),
                    "optimizer": "Adam",
                    "learning_rate": float(cfg.local_lr),
                    "batch_size": int(cfg.local_batch_size),
                    "optimizer_reset": "after every external model average",
                    "experience_counts_replay": False,
                },
                "prototype_remote_mass_merge": "max (idempotent)",
                "round_order": "synchronous aggregate, then local train",
                "experience_merge": "maximum received, then add local links",
                "reception_floor_dbm": self.reception_floor_dbm,
                "unsupported_prediction_dbm": self.reception_floor_dbm,
                "training_measurements": "RSSI >= reception floor only",
                "evaluation_targets": "censored at reception floor",
            }
        )
        if resume is not None:
            self._load_checkpoint(Path(resume))

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        self._replay_buffers.pop(int(i), None)
        if 0 <= int(i) < len(self.greedy_models):
            model = self.greedy_models[int(i)]
            if isinstance(model, PairRBFGatedMLP):
                model.set_prototypes(())

    def _load_measurement_trace(self, path: Path) -> dict[str, object]:
        replay = super()._load_measurement_trace(path)
        states = replay["node_states"]
        active = replay["node_active"]
        assert isinstance(states, np.ndarray)
        assert isinstance(active, np.ndarray)
        # This crop labels positions outside its radio rectangle as zone -1.
        # Such slots cannot train, communicate, or be evaluated in zone 0.
        replay["node_active"] = active & (states[:, :, 2] >= 0.0)
        measurements = replay["measurements"]
        assert isinstance(measurements, dict)
        replay["measurements"] = {
            int(step): rows[rows[:, 4] >= self.reception_floor_dbm]
            for step, rows in measurements.items()
        }
        return replay

    def _build_fidelity_grid(
        self,
        n_pairs: int | None = None,
        zones=None,
    ) -> None:
        del zones
        if int(n_pairs or 0) <= 0:
            self.fidelity_grid = {}
            return
        with np.load(self._testset_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["meta_json"].item()))
            if not bool(metadata.get("buildings_opaque", False)):
                raise ValueError("fidelity set does not use opaque buildings")
            if bool(metadata.get("dynamic_vehicle_blockers", True)):
                raise ValueError("fidelity set uses vehicle blockers")
            X = np.asarray(archive["X"], dtype=np.float32)
            raw_y = np.asarray(
                archive["rssi_dbm"], dtype=np.float32
            ).reshape(-1, 1)
        limit = min(int(n_pairs or len(X)), int(len(X)))
        self._fidelity_feasible = (
            raw_y[:limit].reshape(-1) >= self.reception_floor_dbm
        )
        y = np.maximum(raw_y[:limit], self.reception_floor_dbm)
        self.fidelity_grid = {0: (X[:limit], y)}

    def _gossip_step(self, *args, **kwargs) -> None:
        # The experiment contains only the greedy auxiliary predictor.
        return None

    def _train_predictors_from_current_measurements(
        self,
        *,
        step: int,
        measurements: list[tuple[int, int, int, float]],
    ) -> None:
        # The base run calls this before sharing.  Stage the samples so the
        # overridden greedy round can aggregate first and train second.
        if int(step) <= int(self._resume_step):
            self._staged_measurements = None
        else:
            self._staged_measurements = list(measurements)

    def _bundle_key(
        self, knowledge: dict[int, OriginSummary]
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(origin), int(summary[0]))
            for origin, summary in sorted(knowledge.items())
            if summary[1]
        )

    def _operational_rows(
        self, knowledge: dict[int, OriginSummary]
    ) -> tuple[PrototypeRow, ...]:
        return prototype_union(
            [
                summary[1]
                for _origin, summary in sorted(knowledge.items())
                if summary[1]
            ],
            self.rbf_params,
        )

    def _restore_logs_before_first_new_step(self) -> None:
        if self._resume_logs_restored or self._resume_payload is None:
            return
        payload = self._resume_payload
        self.sharing_rows = list(payload.get("sharing_rows", []))
        self.local_policy_rows = list(payload.get("local_policy_rows", []))
        self.fidelity_history = list(payload.get("fidelity_history", []))
        self._resume_logs_restored = True

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            # The replay loop cheaply reapplies mobility up to the checkpoint.
            # Discard placeholder rows produced while fast-forwarding.
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs_before_first_new_step()

        links = sorted(
            {
                (int(zone), min(int(a), int(b)), max(int(a), int(b)))
                for zone, a, b in (contact_links or [])
                if int(a) != int(b)
            }
        )
        neighbours: dict[int, list[int]] = {}
        for _zone, left, right in links:
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        participants = sorted(neighbours)

        pre_states = {
            index: {
                name: value.detach().clone()
                for name, value in self.greedy_models[index].state_dict().items()
            }
            for index in participants
        }
        pre_experience = {
            index: int(self.greedy_m_samples[index]) for index in participants
        }
        pre_knowledge = {
            index: dict(self._support_knowledge[index]) for index in participants
        }

        next_states: dict[int, dict[str, torch.Tensor]] = {}
        next_experience: dict[int, int] = {}
        next_knowledge: dict[int, dict[int, OriginSummary]] = {}
        next_deltas: dict[int, list[tuple[PrototypeRow, ...]]] = {}
        prototype_values_sent = 0
        support_origins_sent = 0
        for receiver in participants:
            senders = [receiver, *sorted(neighbours[receiver])]
            next_states[receiver] = average_state_dicts(
                [pre_states[sender] for sender in senders],
                [pre_experience[sender] for sender in senders],
            )
            next_experience[receiver] = max(
                pre_experience[sender] for sender in senders
            )
            combined = dict(pre_knowledge[receiver])
            deltas: list[tuple[PrototypeRow, ...]] = []
            for sender in sorted(neighbours[receiver]):
                bundle = pre_knowledge[sender]
                support_origins_sent += len(bundle)
                prototype_values_sent += 5 * sum(
                    len(summary[1]) for summary in bundle.values()
                )
                for origin, summary in bundle.items():
                    previous = combined.get(origin)
                    if previous is None or int(summary[0]) > int(previous[0]):
                        deltas.append(
                            prototype_delta(
                                () if previous is None else previous[1],
                                summary[1],
                            )
                        )
                        combined[origin] = summary
            next_knowledge[receiver] = combined
            next_deltas[receiver] = deltas

        for receiver in participants:
            self.greedy_models[receiver].load_state_dict(next_states[receiver])
            # Adam moments describe the pre-aggregation parameters and are no
            # longer valid after an external weighted model average.
            self.greedy_opts[receiver].state.clear()
            self.greedy_m_samples[receiver] = int(next_experience[receiver])
            self.greedy_n_samples[receiver] = int(next_experience[receiver])
            self._support_knowledge[receiver] = next_knowledge[receiver]
            rows = prototype_union(
                [
                    self.greedy_models[receiver].prototype_rows,
                    *next_deltas[receiver],
                ],
                self.rbf_params,
            )
            self.greedy_models[receiver].set_prototypes(rows)

        self._network_step_stats.update(
            {
                "prototype_scalar_values_sent": int(prototype_values_sent),
                "prototype_payload_bytes": int(4 * prototype_values_sent),
                "prototype_origin_records_sent": int(support_origins_sent),
                "synchronous_greedy_receivers": int(len(participants)),
            }
        )
        self._train_staged_local_samples(step)
        return int(2 * len(links))

    def _train_staged_local_samples(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows_by_receiver: dict[int, list[tuple[list[float], float, np.ndarray]]] = {}
        self._meas_per_node = {}
        for zone, tx_idx, rx_idx, value in measurements:
            tx_node = self.nodes[int(tx_idx)].node
            rx_node = self.nodes[int(rx_idx)].node
            features = self._pair_model_features(
                (tx_node.x, tx_node.y),
                (rx_node.x, rx_node.y),
                step=step,
                zone=int(zone),
            )
            segment = np.asarray(
                [[tx_node.x, tx_node.y], [rx_node.x, rx_node.y]],
                dtype=np.float64,
            )
            rows_by_receiver.setdefault(int(rx_idx), []).append(
                (features, float(value), segment)
            )

        active = {
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
        }
        train_receivers = sorted(
            set(rows_by_receiver) | (active & set(self._replay_buffers))
        )
        for receiver in train_receivers:
            rows = rows_by_receiver.get(receiver, [])
            if rows:
                local_version, local_rows = self._support_knowledge[receiver].get(
                    receiver, (0, ())
                )
                local_prototypes = deserialize_prototypes(local_rows)
                observations = [
                    PairPrototype.from_segment(segment)
                    for _features, _value, segment in rows
                    if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0
                ]
                local_prototypes = merge_prototypes(
                    [*local_prototypes, *observations],
                    self.rbf_params,
                    remote=False,
                )
                updated_local = serialize_prototypes(local_prototypes)
                self._support_knowledge[receiver][receiver] = (
                    int(local_version) + 1,
                    updated_local,
                )
                updated_operational = prototype_union(
                    [
                        self.greedy_models[receiver].prototype_rows,
                        prototype_delta(local_rows, updated_local),
                    ],
                    self.rbf_params,
                )
                self.greedy_models[receiver].set_prototypes(updated_operational)

            X = np.asarray(
                [row[0] for row in rows], dtype=np.float32
            ).reshape(-1, 4)
            y = np.asarray(
                [row[1] for row in rows], dtype=np.float32
            ).reshape(-1, 1)
            replay = self._replay_buffers.get(receiver)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(self.cfg.seed), int(step), int(receiver)]
                )
            )
            if rows:
                self._train_array_epochs(
                    receiver,
                    X,
                    y,
                    epochs=self.training_params.new_data_epochs,
                    rng=rng,
                )
            if replay is not None and replay.size > 0:
                recent_start = (
                    self.training_params.replay_batches
                    - self.training_params.recent_replay_batches
                )
                for batch_index in range(self.training_params.replay_batches):
                    replay_X, replay_y = replay.sample(
                        rng,
                        int(self.cfg.local_batch_size),
                        recent_window=(
                            self.training_params.recent_window
                            if batch_index >= recent_start
                            else None
                        ),
                    )
                    self._train_array_epochs(
                        receiver, replay_X, replay_y, epochs=1, rng=rng
                    )
            if rows:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity,
                        int(X.shape[1]),
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                increment = int(len(rows))
                self.greedy_m_samples[receiver] += increment
                self.greedy_n_samples[receiver] = self.greedy_m_samples[receiver]
        self._staged_measurements = None

    def _train_array_epochs(
        self,
        receiver: int,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        epochs: int,
        rng: np.random.Generator,
    ) -> None:
        model = self.greedy_models[receiver]
        optimizer = self.greedy_opts[receiver]
        y = self._normalize_target_from_rssi(y_dbm)
        batch_size = int(self.cfg.local_batch_size)
        model.train()
        for _epoch in range(int(epochs)):
            order = rng.permutation(len(X))
            for start in range(0, int(len(X)), batch_size):
                indices = order[start : start + batch_size]
                bx = torch.as_tensor(
                    X[indices], dtype=torch.float32, device=self.aux_device
                )
                by = torch.as_tensor(
                    y[indices], dtype=torch.float32, device=self.aux_device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = self._weighted_regression_loss(model(bx), by, None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_params.gradient_clip_norm,
                )
                optimizer.step()

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        if int(step) <= self._resume_step and self._resume_payload is not None:
            old_rows = self._resume_payload.get("fidelity_history", [])
            for row in reversed(old_rows):
                if int(row.get("step", -1)) == int(step):
                    return dict(row)
            return {"step": int(step)}
        self._build_fidelity_grid(n_pairs=n_pairs)
        X, y = self.fidelity_grid[0]
        truth = y.reshape(-1)
        feasible = self._fidelity_feasible
        active = [
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
            and int(self.greedy_m_samples[index]) > 0
        ]
        total_sq = feasible_sq = infeasible_sq = 0.0
        total_count = feasible_count = infeasible_count = 0
        model_rmse: list[float] = []
        confidence_sum = 0.0
        covered_count = 0
        confidence_count = 0
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.aux_device)
        for index in active:
            model = self.greedy_models[index]
            model.eval()
            with torch.no_grad():
                normalized, confidence = model.forward_with_confidence(xt)
                prediction = self._denorm_dbm(
                    normalized.detach().cpu().numpy().reshape(-1)
                )
                conf = confidence.detach().cpu().numpy().reshape(-1)
            error_sq = np.square(prediction - truth)
            total_sq += float(error_sq.sum())
            total_count += int(error_sq.size)
            feasible_sq += float(error_sq[feasible].sum())
            feasible_count += int(feasible.sum())
            infeasible_sq += float(error_sq[~feasible].sum())
            infeasible_count += int((~feasible).sum())
            model_rmse.append(float(np.sqrt(error_sq.mean())))
            confidence_sum += float(conf.sum())
            covered_count += int((conf >= 0.5).sum())
            confidence_count += int(conf.size)

        def rmse(square_sum: float, count: int) -> float:
            return (
                float(math.sqrt(square_sum / count))
                if count > 0
                else float("nan")
            )

        prototype_counts = [
            len(self.greedy_models[index].prototype_rows) for index in active
        ]
        experiences = [int(self.greedy_m_samples[index]) for index in active]
        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_total": rmse(total_sq, total_count),
            "greedy_mean_model_rmse": (
                float(np.mean(model_rmse)) if model_rmse else float("nan")
            ),
            "greedy_feasible_rmse": rmse(feasible_sq, feasible_count),
            "greedy_infeasible_rmse": rmse(infeasible_sq, infeasible_count),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_mean_confidence": (
                confidence_sum / confidence_count
                if confidence_count
                else float("nan")
            ),
            "greedy_coverage_at_0_5": (
                covered_count / confidence_count
                if confidence_count
                else float("nan")
            ),
            "greedy_mean_prototypes": (
                float(np.mean(prototype_counts))
                if prototype_counts
                else float("nan")
            ),
            "greedy_max_prototypes": (
                int(max(prototype_counts)) if prototype_counts else 0
            ),
            "greedy_mean_experience": (
                float(np.mean(experiences)) if experiences else 0.0
            ),
            "greedy_max_experience": (
                int(max(experiences)) if experiences else 0
            ),
        }
        self.fidelity_history.append(row)
        return row

    @staticmethod
    def _cpu_tree(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: RBFGreedySimulation._cpu_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [RBFGreedySimulation._cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                RBFGreedySimulation._cpu_tree(item) for item in value
            )
        return value

    def _save_checkpoint(self, step: int) -> None:
        experienced = [
            index
            for index, value in enumerate(self.greedy_m_samples)
            if int(value) > 0
        ]
        payload = {
            "format": self.checkpoint_format,
            "step": int(step),
            "trace": str(self.measurement_trace_in),
            "testset": str(self._testset_path),
            "reception_floor_dbm": self.reception_floor_dbm,
            "rbf_params": asdict(self.rbf_params),
            "training_params": asdict(self.training_params),
            "experience": list(self.greedy_m_samples),
            "models": {
                int(index): self._cpu_tree(
                    self.greedy_models[index].state_dict()
                )
                for index in experienced
            },
            "optimizers": {
                int(index): self._cpu_tree(
                    self.greedy_opts[index].state_dict()
                )
                for index in experienced
            },
            "support_knowledge": self._support_knowledge,
            "replay_buffers": {
                int(index): replay.state_dict()
                for index, replay in self._replay_buffers.items()
            },
            "fidelity_history": self.fidelity_history,
            "sharing_rows": self.sharing_rows,
            "local_policy_rows": self.local_policy_rows,
        }
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        target = output / "checkpoint_latest.pt"
        temporary = output / "checkpoint_latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, target)
        status = {
            "format": self.checkpoint_format,
            "step": int(step),
            "path": str(target),
            "experienced_models": int(len(experienced)),
            "latest_fidelity": (
                self.fidelity_history[-1] if self.fidelity_history else None
            ),
        }
        status_tmp = output / "checkpoint_status.json.tmp"
        status_path = output / "checkpoint_status.json"
        with open(status_tmp, "w", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2, sort_keys=True)
        os.replace(status_tmp, status_path)
        print(
            f"[RBF-GREEDY] checkpoint step={step} "
            f"models={len(experienced)} path={target}",
            flush=True,
        )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("format") != self.checkpoint_format:
            raise ValueError(f"unsupported checkpoint format in {path}")
        if float(payload.get("reception_floor_dbm", float("nan"))) != (
            self.reception_floor_dbm
        ):
            raise ValueError("checkpoint reception floor differs")
        if payload.get("rbf_params") != asdict(self.rbf_params):
            raise ValueError("checkpoint RBF parameters differ")
        if payload.get("training_params") != asdict(self.training_params):
            raise ValueError("checkpoint local training parameters differ")
        if str(Path(payload.get("trace", "")).resolve()) != str(
            Path(self.measurement_trace_in).resolve()
        ):
            raise ValueError("checkpoint trace differs from requested trace")
        self._resume_step = int(payload["step"])
        experience = [int(value) for value in payload["experience"]]
        if len(experience) != int(self.cfg.num_nodes):
            raise ValueError("checkpoint vehicle count differs from config")
        self.greedy_m_samples = list(experience)
        self.greedy_n_samples = list(experience)
        for raw_index, state in payload["models"].items():
            index = int(raw_index)
            self.greedy_models[index].load_state_dict(state)
        for raw_index, state in payload["optimizers"].items():
            index = int(raw_index)
            self.greedy_opts[index].load_state_dict(state)
        self._support_knowledge = payload["support_knowledge"]
        self._replay_buffers = {
            int(index): ReplayBuffer.from_state_dict(state)
            for index, state in payload["replay_buffers"].items()
        }
        operational_cache: dict[
            tuple[tuple[int, int], ...], tuple[PrototypeRow, ...]
        ] = {}
        for index, value in enumerate(experience):
            if value <= 0:
                continue
            key = self._bundle_key(self._support_knowledge[index])
            rows = operational_cache.get(key)
            if rows is None:
                rows = self._operational_rows(self._support_knowledge[index])
                operational_cache[key] = rows
            self.greedy_models[index].set_prototypes(rows)
        self._resume_payload = payload
        self._resume_logs_restored = False
        print(
            f"[RBF-GREEDY] resumed step={self._resume_step} from {path}",
            flush=True,
        )

    def _write_partial_outputs(self, **kwargs) -> None:
        step = int(kwargs.get("step", 0))
        if step <= self._resume_step and self._resume_payload is not None:
            return
        super()._write_partial_outputs(**kwargs)
        if step > 0 and (
            step % max(1, int(self.flush_every)) == 0
            or step >= int(self.cfg.sim_steps)
        ):
            self._save_checkpoint(step)


def validate_dataset(trace: Path, testset: Path) -> dict[str, int | float]:
    with np.load(trace, allow_pickle=False) as archive:
        trace_meta = json.loads(str(archive["meta_json"].item()))
        if not bool(trace_meta.get("buildings_opaque", False)):
            raise ValueError("trace does not use opaque buildings")
        if bool(trace_meta.get("dynamic_vehicle_blockers", True)):
            raise ValueError("trace uses vehicle blockers")
    with np.load(testset, allow_pickle=False) as archive:
        test_meta = json.loads(str(archive["meta_json"].item()))
        if not bool(test_meta.get("buildings_opaque", False)):
            raise ValueError("test set does not use opaque buildings")
        if bool(test_meta.get("dynamic_vehicle_blockers", True)):
            raise ValueError("test set uses vehicle blockers")
        test_count = int(archive["X"].shape[0])
    return {
        "num_nodes": int(trace_meta["num_nodes"]),
        "sim_steps": int(trace_meta["sim_steps"]),
        "num_zones": int(trace_meta["num_zones"]),
        "tx_power_dbm": float(trace_meta["tx_power_dbm"]),
        "rssi_min_dbm": float(trace_meta["rssi_min_dbm"]),
        "rssi_max_dbm": float(trace_meta["rssi_max_dbm"]),
        "test_count": test_count,
    }


def self_test() -> None:
    params = RBFParams(sigma_pair_m=2.0, merge_radius_m=3.0)
    prototypes: list[PairPrototype] = []
    add_prototype(
        prototypes,
        PairPrototype.from_segment(np.asarray([[10.0, 10.0], [90.0, 10.0]])),
        params,
        remote=False,
    )
    add_prototype(
        prototypes,
        PairPrototype.from_segment(np.asarray([[11.0, 10.0], [91.0, 10.0]])),
        params,
        remote=False,
    )
    assert len(prototypes) == 1
    assert prototypes[0].mass == 2.0

    base = nn.Sequential(nn.Linear(4, 1))
    with torch.no_grad():
        base[0].weight.zero_()
        base[0].bias.zero_()
    model = PairRBFGatedMLP(
        base,
        map_size_m=100.0,
        floor_prior_norm=0.8,
        rbf_params=params,
    )
    model.set_prototypes(serialize_prototypes(prototypes))
    model.eval()
    exact = torch.tensor([[0.105, 0.1, 0.905, 0.1]])
    swapped = torch.tensor([[0.905, 0.1, 0.105, 0.1]])
    shifted = torch.tensor([[0.105, 0.15, 0.905, 0.15]])
    assert torch.allclose(model.confidence(exact), model.confidence(swapped))
    assert float(model.confidence(shifted)) < 0.05 * float(model.confidence(exact))

    replay = ReplayBuffer(3)
    replay.add(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(4, dtype=np.float32).reshape(-1, 1),
    )
    state = replay.state_dict()
    assert np.array_equal(state["y"].reshape(-1), [1.0, 2.0, 3.0])
    assert ReplayBuffer.from_state_dict(state).size == 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--local-lr", type=float, default=5.0e-4)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--new-data-epochs", type=int, default=2)
    parser.add_argument("--replay-batches", type=int, default=8)
    parser.add_argument("--recent-replay-batches", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--reception-floor-dbm", type=float, default=-100.0
    )
    parser.add_argument("--rbf-sigma-m", type=float, default=2.0)
    parser.add_argument("--prototype-merge-radius-m", type=float, default=3.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("pair RBF greedy self-test passed")
        return 0

    metadata = validate_dataset(args.trace.resolve(), args.testset.resolve())
    sim_steps = int(args.sim_steps or metadata["sim_steps"])
    if sim_steps > int(metadata["sim_steps"]):
        raise ValueError("requested steps exceed the trace")
    checkpoint_every = max(1, int(args.checkpoint_every))
    results_dir = args.results_dir.resolve()
    resume = args.resume
    automatic = results_dir / "checkpoint_latest.pt"
    if resume is None and args.resume_if_exists and automatic.exists():
        resume = automatic

    rbf_params = RBFParams(
        sigma_pair_m=float(args.rbf_sigma_m),
        merge_radius_m=float(args.prototype_merge_radius_m),
    )
    training_params = TrainingParams(
        replay_capacity=int(args.replay_capacity),
        new_data_epochs=int(args.new_data_epochs),
        replay_batches=int(args.replay_batches),
        recent_replay_batches=int(args.recent_replay_batches),
        recent_window=int(args.recent_window),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )
    reception_floor_dbm = float(args.reception_floor_dbm)
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(metadata["num_nodes"]),
        num_zones=int(metadata["num_zones"]),
        sim_steps=sim_steps,
        map_size=800.0,
        active_modes=(),
        results_dir=str(results_dir),
        tx_power_dbm=float(metadata["tx_power_dbm"]),
        rssi_min_dbm=reception_floor_dbm,
        rssi_max_dbm=float(metadata["rssi_max_dbm"]),
        noise_floor_dbm=reception_floor_dbm,
        snr_min_db=0.0,
        model_transfer_snr_min_db=0.0,
        rssi_model="tiny",
        predictor_prior="none",
        predictor_include_time=False,
        local_lr=float(args.local_lr),
        local_batch_size=int(args.local_batch_size),
        local_epochs=1,
        merge_strategy="average",
        fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_eval_every=checkpoint_every,
        final_fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_final_steps=(sim_steps,),
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    simulation = RBFGreedySimulation(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=reception_floor_dbm,
        rbf_params=rbf_params,
        training_params=training_params,
        resume=resume,
        progress_every=int(args.progress_every),
        log_rmse_every=0,
        flush_every=checkpoint_every,
        random_od_routing=False,
        local_policy_share=False,
    )
    simulation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
