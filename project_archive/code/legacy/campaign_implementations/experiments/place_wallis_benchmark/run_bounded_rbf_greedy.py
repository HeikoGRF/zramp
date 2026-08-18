#!/usr/bin/env python3
"""Equal-greedy MLP sharing with bounded reciprocal pair-RBF support."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.place_wallis_benchmark.run_equal_greedy import (  # noqa: E402
    DEFAULT_NET,
    DEFAULT_TESTSET,
    DEFAULT_TRACE,
    EqualGreedySimulation,
    ReplayBuffer,
    TrainingParams,
    atomic_json,
    equal_average,
    validate_dataset,
)
from experiments.place_wallis_benchmark.tail_metrics import (  # noqa: E402
    make_tail_evaluation_steps,
    temporal_metric_summary,
)
from rl_reward_experiment.config import build_config_from_env  # noqa: E402

DEFAULT_RESULTS = (
    ROOT / "artifacts/place_wallis_benchmark/methods/bounded_rbf_greedy_eval50_tail10x25"
)


@dataclass(frozen=True)
class RBFParams:
    support_budget: int = 512
    local_merge_radius_m: float = 2.5
    maturity_scale: float = 3.0
    bandwidths_m: tuple[float, ...] = (3.0, 5.0, 8.0)
    primary_bandwidth_m: float = 5.0
    confidence_threshold: float = 0.5
    tree_neighbours: int = 8

    def __post_init__(self) -> None:
        if self.support_budget <= 0:
            raise ValueError("support budget must be positive")
        if self.local_merge_radius_m <= 0.0:
            raise ValueError("local merge radius must be positive")
        if self.maturity_scale <= 0.0:
            raise ValueError("maturity scale must be positive")
        if not self.bandwidths_m or any(v <= 0.0 for v in self.bandwidths_m):
            raise ValueError("RBF bandwidths must be positive")
        if self.primary_bandwidth_m not in self.bandwidths_m:
            raise ValueError("primary bandwidth must be evaluated")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("confidence threshold must lie in (0, 1)")
        if self.tree_neighbours <= 0:
            raise ValueError("tree neighbour count must be positive")


class Prototype(NamedTuple):
    origin: int
    local_id: int
    version: int
    center: tuple[float, float, float, float]
    mass: float


def canonical_center(values: np.ndarray) -> np.ndarray:
    center = np.asarray(values, dtype=np.float64).reshape(4)
    return (
        center
        if tuple(center[:2]) <= tuple(center[2:])
        else center[[2, 3, 0, 1]]
    )


def pair_distance_sq(centers: np.ndarray, query: np.ndarray) -> np.ndarray:
    direct = 0.5 * np.square(centers - query[None, :]).sum(axis=1)
    swapped_query = query[[2, 3, 0, 1]]
    swapped = 0.5 * np.square(centers - swapped_query[None, :]).sum(axis=1)
    return np.minimum(direct, swapped)


def prototype_key(row: Prototype) -> tuple[int, int]:
    return int(row.origin), int(row.local_id)


def newest_union(groups: list[tuple[Prototype, ...]]) -> list[Prototype]:
    result: dict[tuple[int, int], Prototype] = {}
    for row in (item for group in groups for item in group):
        key = prototype_key(row)
        previous = result.get(key)
        if previous is None or int(row.version) > int(previous.version):
            result[key] = row
        elif int(row.version) == int(previous.version):
            # Deterministic tie breaking keeps repeated gossip idempotent.
            result[key] = min(previous, row)
    return list(result.values())


def compress_prototypes(
    rows: list[Prototype],
    params: RBFParams,
) -> tuple[Prototype, ...]:
    """Deduplicate geometrically, then make a deterministic k-centre coreset."""
    if not rows:
        return ()
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.mass),
            int(row.origin),
            int(row.local_id),
            -int(row.version),
        ),
    )
    radius_sq = float(params.local_merge_radius_m) ** 2
    kept: list[Prototype] = []
    centers: list[np.ndarray] = []
    for row in ordered:
        center = canonical_center(np.asarray(row.center))
        if centers:
            existing = np.stack(centers)
            if float(pair_distance_sq(existing, center).min()) <= radius_sq:
                continue
        kept.append(
            Prototype(
                int(row.origin),
                int(row.local_id),
                int(row.version),
                tuple(float(value) for value in center),
                float(row.mass),
            )
        )
        centers.append(center)
    budget = int(params.support_budget)
    if len(kept) <= budget:
        return tuple(sorted(kept, key=prototype_key))

    values = np.stack(centers)
    masses = np.asarray([row.mass for row in kept], dtype=np.float64)
    first = int(np.argmax(masses))
    selected = [first]
    available = np.ones(len(kept), dtype=np.bool_)
    available[first] = False
    minimum_sq = pair_distance_sq(values, values[first])
    while len(selected) < budget:
        # A small maturity term resolves spatial ties without letting traffic
        # volume overwhelm the geometric coverage objective.
        score = minimum_sq * (1.0 + 0.05 * np.log1p(masses))
        score[~available] = -1.0
        index = int(np.argmax(score))
        selected.append(index)
        available[index] = False
        minimum_sq = np.minimum(
            minimum_sq, pair_distance_sq(values, values[index])
        )
    return tuple(sorted((kept[index] for index in selected), key=prototype_key))


class BoundedRBFGatedMLP(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        map_size_m: float,
        floor_prior_norm: float,
        params: RBFParams,
    ) -> None:
        super().__init__()
        self.base = base
        self.map_size_m = float(map_size_m)
        self.floor_prior_norm = float(floor_prior_norm)
        self.params = params
        self._rows: tuple[Prototype, ...] = ()
        self._tree: cKDTree | None = None
        self._maturity = np.empty((0,), dtype=np.float64)

    @property
    def prototype_rows(self) -> tuple[Prototype, ...]:
        return self._rows

    def set_prototypes(self, rows: tuple[Prototype, ...]) -> None:
        self._rows = tuple(rows)
        if not rows:
            self._tree = None
            self._maturity = np.empty((0,), dtype=np.float64)
            return
        centers = np.asarray([row.center for row in rows], dtype=np.float64)
        self._tree = cKDTree(
            np.concatenate((centers, centers[:, [2, 3, 0, 1]]), axis=0)
        )
        mass = np.asarray([row.mass for row in rows], dtype=np.float64)
        self._maturity = 1.0 - np.exp(-mass / self.params.maturity_scale)

    def confidence_numpy(
        self, X: np.ndarray, sigma_m: float
    ) -> np.ndarray:
        if self._tree is None:
            return np.zeros((len(X),), dtype=np.float32)
        query = np.asarray(X[:, :4], dtype=np.float64) * self.map_size_m
        k = min(
            int(self.params.tree_neighbours),
            2 * len(self._rows),
        )
        distance, indices = self._tree.query(query, k=k)
        if k == 1:
            distance = distance[:, None]
            indices = indices[:, None]
        maturity = self._maturity[np.asarray(indices) % len(self._rows)]
        scores = maturity * np.exp(
            -np.square(distance) / (4.0 * float(sigma_m) ** 2)
        )
        return np.asarray(scores.max(axis=1), dtype=np.float32)

    def forward_for_sigma(
        self, x: torch.Tensor, sigma_m: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base(x)
        confidence = torch.as_tensor(
            self.confidence_numpy(
                x.detach().cpu().numpy(), float(sigma_m)
            ).reshape(-1, 1),
            dtype=x.dtype,
            device=x.device,
        )
        prior = torch.full_like(raw, self.floor_prior_norm)
        return prior + confidence * (raw - prior), confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support is an inference gate, never a target or gradient multiplier.
        return self.base(x)


class BoundedRBFGreedySimulation(EqualGreedySimulation):
    checkpoint_format = "place_wallis_bounded_rbf_greedy_checkpoint_v1"

    def __init__(
        self,
        cfg,
        *,
        rbf_params: RBFParams,
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.rbf_params = rbf_params
        self._supports: list[tuple[Prototype, ...]] = [
            () for _ in range(int(cfg.num_nodes))
        ]
        self._next_local_id = [0 for _ in range(int(cfg.num_nodes))]
        super().__init__(cfg, resume=None, **kwargs)
        floor_loss = float(cfg.tx_power_dbm) - float(cfg.noise_floor_dbm)
        floor_prior_norm = float(
            np.clip(
                self._norm_loss_db(
                    np.asarray([floor_loss], dtype=np.float32)
                )[0],
                0.0,
                1.0,
            )
        )
        wrapped: list[BoundedRBFGatedMLP] = []
        optimizers: list[optim.Optimizer] = []
        for base in self.greedy_models:
            model = BoundedRBFGatedMLP(
                base,
                map_size_m=float(cfg.map_size),
                floor_prior_norm=floor_prior_norm,
                params=rbf_params,
            ).to(self.aux_device)
            wrapped.append(model)
            optimizers.append(optim.Adam(model.parameters(), lr=cfg.local_lr))
        self.greedy_models = wrapped
        self.greedy_opts = optimizers
        self._aux_template_state = {
            name: value.detach().clone()
            for name, value in wrapped[0].state_dict().items()
        }
        self._communication_assumptions.update(
            {
                "method": "equal-greedy MLP with bounded reciprocal pair-RBF support",
                "rbf_support": asdict(rbf_params),
                "support_shared": True,
                "support_payload": "origin, local ID, version, two endpoints, mass",
                "support_merge": "newest-ID union, geometric deduplication, deterministic k-centre reduction",
                "remote_mass_merge": "never summed",
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
        if 0 <= int(i) < len(self._supports):
            self._supports[int(i)] = ()
            self._next_local_id[int(i)] = 0

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs()
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
        pre_supports = {index: self._supports[index] for index in participants}
        next_states: dict[int, dict[str, torch.Tensor]] = {}
        next_supports: dict[int, tuple[Prototype, ...]] = {}
        support_cache: dict[
            tuple[tuple[int, int, int], ...], tuple[Prototype, ...]
        ] = {}
        support_records_sent = 0
        for receiver in participants:
            senders = [receiver, *sorted(neighbours[receiver])]
            next_states[receiver] = equal_average(
                [pre_states[sender] for sender in senders]
            )
            incoming = [pre_supports[sender] for sender in senders]
            support_records_sent += sum(
                len(pre_supports[sender]) for sender in neighbours[receiver]
            )
            union = newest_union(incoming)
            signature = tuple(
                sorted(
                    (
                        int(row.origin),
                        int(row.local_id),
                        int(row.version),
                    )
                    for row in union
                )
            )
            reduced = support_cache.get(signature)
            if reduced is None:
                reduced = compress_prototypes(union, self.rbf_params)
                support_cache[signature] = reduced
            next_supports[receiver] = reduced
        for receiver in participants:
            self.greedy_models[receiver].load_state_dict(next_states[receiver])
            self.greedy_opts[receiver].state.clear()
            self._supports[receiver] = next_supports[receiver]
            self.greedy_models[receiver].set_prototypes(
                next_supports[receiver]
            )
        model_parameters = (
            sum(value.numel() for value in next(iter(pre_states.values())).values())
            if pre_states
            else 0
        )
        transfers = 2 * len(links)
        self._network_step_stats.update(
            {
                "synchronous_greedy_receivers": int(len(participants)),
                "equal_weight_model_transfers": int(transfers),
                "model_payload_bytes": int(4 * model_parameters * transfers),
                "rbf_support_records_sent": int(support_records_sent),
                "rbf_support_payload_bytes": int(32 * support_records_sent),
                "rbf_max_operational_prototypes": int(
                    max((len(rows) for rows in next_supports.values()), default=0)
                ),
            }
        )
        self._update_local_support(step)
        super()._train_local(step)
        return int(transfers)

    def _update_local_support(self, step: int) -> None:
        del step
        measurements = self._staged_measurements or []
        segments: dict[int, list[np.ndarray]] = {}
        for _zone, tx_idx, rx_idx, _value in measurements:
            tx = self.nodes[int(tx_idx)].node
            rx = self.nodes[int(rx_idx)].node
            center = canonical_center(
                np.asarray([tx.x, tx.y, rx.x, rx.y], dtype=np.float64)
            )
            if float(np.linalg.norm(center[:2] - center[2:])) >= 1.0:
                segments.setdefault(int(rx_idx), []).append(center)
        for receiver, observations in segments.items():
            support = {
                prototype_key(row): row for row in self._supports[receiver]
            }
            for center in observations:
                local = [
                    row for row in support.values() if row.origin == receiver
                ]
                match: Prototype | None = None
                if local:
                    local_centers = np.asarray(
                        [row.center for row in local], dtype=np.float64
                    )
                    distances = pair_distance_sq(local_centers, center)
                    index = int(np.argmin(distances))
                    if float(distances[index]) <= (
                        self.rbf_params.local_merge_radius_m**2
                    ):
                        match = local[index]
                if match is None:
                    local_id = int(self._next_local_id[receiver])
                    self._next_local_id[receiver] += 1
                    row = Prototype(
                        receiver,
                        local_id,
                        1,
                        tuple(float(value) for value in center),
                        1.0,
                    )
                else:
                    total = float(match.mass) + 1.0
                    updated = canonical_center(
                        (
                            float(match.mass) * np.asarray(match.center)
                            + center
                        )
                        / total
                    )
                    row = Prototype(
                        receiver,
                        int(match.local_id),
                        int(match.version) + 1,
                        tuple(float(value) for value in updated),
                        total,
                    )
                support[prototype_key(row)] = row
            reduced = compress_prototypes(
                list(support.values()), self.rbf_params
            )
            self._supports[receiver] = reduced
            self.greedy_models[receiver].set_prototypes(reduced)

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        if int(step) <= self._resume_step and self._resume_payload is not None:
            for row in reversed(self._resume_payload.get("fidelity_history", [])):
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
        accumulators = {
            sigma: {
                "total_sq": 0.0,
                "feasible_sq": 0.0,
                "infeasible_sq": 0.0,
                "coverage": 0,
                "feasible_coverage": 0,
                "infeasible_coverage": 0,
                "confidence_sum": 0.0,
            }
            for sigma in self.rbf_params.bandwidths_m
        }
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.aux_device)
        for index in active:
            model = self.greedy_models[index]
            model.eval()
            with torch.no_grad():
                raw = model(xt)
            for sigma in self.rbf_params.bandwidths_m:
                confidence_np = model.confidence_numpy(X, sigma)
                confidence = torch.as_tensor(
                    confidence_np.reshape(-1, 1),
                    dtype=raw.dtype,
                    device=raw.device,
                )
                prior = torch.full_like(raw, model.floor_prior_norm)
                normalized = prior + confidence * (raw - prior)
                prediction = self._denorm_dbm(
                    normalized.detach().cpu().numpy().reshape(-1)
                )
                error_sq = np.square(prediction - truth)
                covered = (
                    confidence_np >= self.rbf_params.confidence_threshold
                )
                stats = accumulators[sigma]
                stats["total_sq"] += float(error_sq.sum())
                stats["feasible_sq"] += float(error_sq[feasible].sum())
                stats["infeasible_sq"] += float(error_sq[~feasible].sum())
                stats["coverage"] += int(covered.sum())
                stats["feasible_coverage"] += int(covered[feasible].sum())
                stats["infeasible_coverage"] += int(covered[~feasible].sum())
                stats["confidence_sum"] += float(confidence_np.sum())
        total_count = len(active) * len(truth)
        feasible_count = len(active) * int(feasible.sum())
        infeasible_count = len(active) * int((~feasible).sum())

        def ratio(value: float, count: int) -> float:
            return float(value / count) if count else float("nan")

        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_mean_prototypes": (
                float(np.mean([len(self._supports[i]) for i in active]))
                if active
                else 0.0
            ),
            "greedy_max_prototypes": int(
                max((len(self._supports[i]) for i in active), default=0)
            ),
            "greedy_mean_local_experience": (
                float(np.mean([self.greedy_m_samples[i] for i in active]))
                if active
                else 0.0
            ),
        }
        for sigma, stats in accumulators.items():
            label = f"sigma_{sigma:g}m"
            row[f"{label}_overall_rmse"] = math.sqrt(
                ratio(stats["total_sq"], total_count)
            )
            row[f"{label}_feasible_rmse"] = math.sqrt(
                ratio(stats["feasible_sq"], feasible_count)
            )
            row[f"{label}_infeasible_rmse"] = math.sqrt(
                ratio(stats["infeasible_sq"], infeasible_count)
            )
            row[f"{label}_mean_confidence"] = ratio(
                stats["confidence_sum"], total_count
            )
            row[f"{label}_coverage_at_0_5"] = ratio(
                stats["coverage"], total_count
            )
            row[f"{label}_feasible_coverage_at_0_5"] = ratio(
                stats["feasible_coverage"], feasible_count
            )
            row[f"{label}_infeasible_leakage_at_0_5"] = ratio(
                stats["infeasible_coverage"], infeasible_count
            )
        primary = f"sigma_{self.rbf_params.primary_bandwidth_m:g}m"
        row["greedy_total"] = row[f"{primary}_overall_rmse"]
        row["greedy_feasible_rmse"] = row[f"{primary}_feasible_rmse"]
        row["greedy_infeasible_rmse"] = row[f"{primary}_infeasible_rmse"]
        row["greedy_coverage_at_0_5"] = row[f"{primary}_coverage_at_0_5"]
        self.fidelity_history.append(row)
        return row

    def _save_checkpoint(self, step: int) -> None:
        experienced = [
            index
            for index, count in enumerate(self.greedy_m_samples)
            if int(count) > 0
        ]
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        latest = self.fidelity_history[-1] if self.fidelity_history else {}
        metric_keys = [
            "greedy_total",
            "greedy_feasible_rmse",
            "greedy_infeasible_rmse",
            "greedy_coverage_at_0_5",
        ]
        for sigma in self.rbf_params.bandwidths_m:
            label = f"sigma_{sigma:g}m"
            metric_keys.extend(
                f"{label}_{suffix}"
                for suffix in (
                    "overall_rmse",
                    "feasible_rmse",
                    "infeasible_rmse",
                    "mean_confidence",
                    "coverage_at_0_5",
                    "feasible_coverage_at_0_5",
                    "infeasible_leakage_at_0_5",
                )
            )
        temporal = temporal_metric_summary(
            self.fidelity_history,
            evaluation_steps=self.tail_evaluation_steps,
            metric_keys=tuple(metric_keys),
        )
        means = temporal["mean"]
        deviations = temporal["standard_deviation"]
        status = (
            "complete"
            if step >= int(self.cfg.sim_steps) and temporal["complete"]
            else "running"
        )
        atomic_json(
            output / "checkpoint_status.json",
            {
                "format": self.checkpoint_format,
                "step": int(step),
                "checkpoint_kind": "metrics-only",
                "path": None,
                "experienced_models": len(experienced),
                "latest_fidelity": latest,
                "temporal_evaluation": temporal,
            },
        )
        atomic_json(
            output / "metrics.json",
            {
                "schema": "place_wallis_benchmark_result_v1",
                "status": status,
                "method": {
                    "id": "bounded_rbf_greedy",
                    "name": "Bounded pair-RBF greedy sharing",
                    "model": "4-64-64-1 MLP",
                },
                "checkpoint": {
                    "step": int(step),
                    "final_step": int(self.cfg.sim_steps),
                },
                "primary_bandwidth_m": self.rbf_params.primary_bandwidth_m,
                "metrics_db": {
                    "overall_rmse": means["greedy_total"],
                    "feasible_rmse": means["greedy_feasible_rmse"],
                    "non_feasible_rmse": means["greedy_infeasible_rmse"],
                },
                "metrics_db_standard_deviation": {
                    "overall_rmse": deviations["greedy_total"],
                    "feasible_rmse": deviations["greedy_feasible_rmse"],
                    "non_feasible_rmse": deviations[
                        "greedy_infeasible_rmse"
                    ],
                },
                "support": {
                    "budget": self.rbf_params.support_budget,
                    "mean_prototypes": latest.get("greedy_mean_prototypes"),
                    "max_prototypes": latest.get("greedy_max_prototypes"),
                    "primary_coverage_at_0_5": means[
                        "greedy_coverage_at_0_5"
                    ],
                },
                "rbf_params": asdict(self.rbf_params),
                "temporal_evaluation": temporal,
                "training": asdict(self.training_params),
                "latest_fidelity": latest,
                "testset": {
                    "path": str(self._testset_path),
                    "samples": int(
                        latest.get("eval_n_pairs_per_zone", 0)
                    ),
                },
            },
        )
        if latest:
            print(
                f"[BOUNDED-RBF] metrics={step} "
                f"overall={float(latest['greedy_total']):.4f}dB "
                f"feasible={float(latest['greedy_feasible_rmse']):.4f}dB "
                f"non-feasible={float(latest['greedy_infeasible_rmse']):.4f}dB "
                f"coverage={100 * float(latest['greedy_coverage_at_0_5']):.1f}% "
                f"prototypes={float(latest['greedy_mean_prototypes']):.1f}",
                flush=True,
            )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("format") != self.checkpoint_format:
            raise ValueError("unsupported checkpoint format")
        if payload.get("rbf_params") != asdict(self.rbf_params):
            raise ValueError("checkpoint RBF parameters differ")
        if payload.get("training_params") != asdict(self.training_params):
            raise ValueError("checkpoint training parameters differ")
        if tuple(payload.get("tail_evaluation_steps", ())) != (
            self.tail_evaluation_steps
        ):
            raise ValueError("checkpoint tail evaluation schedule differs")
        self._resume_step = int(payload["step"])
        experience = [int(value) for value in payload["local_experience"]]
        self.greedy_m_samples = list(experience)
        self.greedy_n_samples = list(experience)
        for index, state in payload["models"].items():
            self.greedy_models[int(index)].load_state_dict(state)
        for index, state in payload["optimizers"].items():
            self.greedy_opts[int(index)].load_state_dict(state)
        self._supports = [tuple(rows) for rows in payload["supports"]]
        self._next_local_id = [
            int(value) for value in payload["next_local_id"]
        ]
        for index, rows in enumerate(self._supports):
            self.greedy_models[index].set_prototypes(rows)
        self._replay_buffers = {
            int(index): ReplayBuffer.from_state_dict(state)
            for index, state in payload["replay_buffers"].items()
        }
        self._resume_payload = payload
        self._resume_logs_restored = False
        print(
            f"[BOUNDED-RBF] resumed step={self._resume_step} from {path}",
            flush=True,
        )


def self_test() -> None:
    params = RBFParams(support_budget=3, local_merge_radius_m=1.0)
    rows = [
        Prototype(i, 0, 1, (float(i * 3), 0.0, 20.0, 0.0), 1.0)
        for i in range(8)
    ]
    compressed = compress_prototypes(rows, params)
    assert len(compressed) == 3
    assert len({prototype_key(row) for row in compressed}) == 3
    newer = Prototype(0, 0, 2, (1.0, 0.0, 20.0, 0.0), 2.0)
    union = newest_union([(rows[0],), (newer,)])
    assert len(union) == 1 and union[0].version == 2
    centers = np.asarray([[0.0, 0.0, 10.0, 0.0]])
    query = np.asarray([10.0, 0.0, 0.0, 0.0])
    assert float(pair_distance_sq(centers, query)[0]) == 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--support-budget", type=int, default=512)
    parser.add_argument("--local-merge-radius-m", type=float, default=2.5)
    parser.add_argument("--maturity-scale", type=float, default=3.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--tail-eval-count", type=int, default=10)
    parser.add_argument("--tail-eval-stride", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("bounded RBF self-test passed")
        return 0
    metadata = validate_dataset(args.trace.resolve(), args.testset.resolve())
    sim_steps = (
        int(metadata["sim_steps"])
        if args.sim_steps is None
        else int(args.sim_steps)
    )
    if sim_steps > int(metadata["sim_steps"]):
        raise ValueError("requested steps exceed trace")
    tail_evaluation_steps = make_tail_evaluation_steps(
        sim_steps,
        count=int(args.tail_eval_count),
        stride=int(args.tail_eval_stride),
    )
    results_dir = args.results_dir.resolve()
    resume = args.resume
    automatic = results_dir / "checkpoint_latest.pt"
    if resume is None and args.resume_if_exists and automatic.exists():
        resume = automatic
    training = TrainingParams(
        replay_capacity=int(args.replay_capacity),
        new_data_epochs=int(args.new_data_epochs),
        replay_batches=int(args.replay_batches),
        recent_replay_batches=int(args.recent_replay_batches),
        recent_window=int(args.recent_window),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )
    rbf_params = RBFParams(
        support_budget=int(args.support_budget),
        local_merge_radius_m=float(args.local_merge_radius_m),
        maturity_scale=float(args.maturity_scale),
    )
    checkpoint_every = max(1, int(args.checkpoint_every))
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(metadata["num_nodes"]),
        num_zones=int(metadata["num_zones"]),
        sim_steps=sim_steps,
        map_size=300.0,
        active_modes=(),
        results_dir=str(results_dir),
        tx_power_dbm=float(metadata["tx_power_dbm"]),
        rssi_min_dbm=-100.0,
        rssi_max_dbm=float(metadata["rssi_max_dbm"]),
        noise_floor_dbm=-100.0,
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
        fidelity_final_steps=tail_evaluation_steps,
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    simulation = BoundedRBFGreedySimulation(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=-100.0,
        training_params=training,
        tail_evaluation_steps=tail_evaluation_steps,
        rbf_params=rbf_params,
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
