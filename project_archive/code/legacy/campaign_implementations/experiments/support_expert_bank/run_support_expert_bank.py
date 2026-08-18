#!/usr/bin/env python3
"""Support-driven bounded expert bank using finite-segment capsules."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.capsule_greedy.run_capsule_greedy import (  # noqa: E402
    DEFAULT_NET,
    DEFAULT_TESTSET,
    DEFAULT_TRACE,
    Capsule,
    CapsuleGatedMLP,
    CapsuleGreedySimulation,
    CapsuleParams,
    CapsuleRow,
    GateParams,
    ReplayBuffer,
    TrainingParams,
    add_capsule_vectorized,
    deserialize_capsules,
    serialize_capsules,
    self_test as capsule_greedy_self_test,
    validate_dataset,
)
from rl_reward_experiment.config import build_config_from_env  # noqa: E402


DEFAULT_RESULTS = (
    ROOT
    / "artifacts/support_expert_bank/replay10_capsule_baseline_k4_cost0"
)
ExpertKey = tuple[int, int, int]


@dataclass(frozen=True)
class ExpertRecord:
    """One immutable model version and the support on which it was trained."""

    key: ExpertKey
    experience: int
    capsules: tuple[CapsuleRow, ...]
    model_state: dict[str, torch.Tensor]

    @property
    def lineage(self) -> tuple[int, int]:
        return self.key[:2]


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _support_weight(
    rows: tuple[CapsuleRow, ...], params: CapsuleParams
) -> float:
    if not rows:
        return 0.0
    values = np.asarray(rows, dtype=np.float64)
    length = np.linalg.norm(values[:, 2:4] - values[:, 0:2], axis=1)
    maturity = 1.0 - np.exp(-values[:, 4] / float(params.mass_scale))
    return float(np.sum(length * maturity))


def capsule_novelty_fraction(
    candidate: tuple[CapsuleRow, ...],
    existing: tuple[CapsuleRow, ...],
    params: CapsuleParams,
) -> float:
    """Mass/length-weighted support not compatible with an existing capsule."""

    if not candidate:
        return 0.0
    if not existing:
        return 1.0
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(existing, dtype=np.float64)
    left_points = left[:, :4].reshape(-1, 2, 2)
    right_points = right[:, :4].reshape(-1, 2, 2)
    left_vector = left_points[:, 1] - left_points[:, 0]
    right_vector = right_points[:, 1] - right_points[:, 0]
    left_length = np.linalg.norm(left_vector, axis=1).clip(min=1.0e-9)
    right_length = np.linalg.norm(right_vector, axis=1).clip(min=1.0e-9)
    left_axis = left_vector / left_length[:, None]
    right_axis = right_vector / right_length[:, None]

    cosine = np.clip(np.abs(left_axis @ right_axis.T), 0.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    delta = (
        right_points.mean(axis=1)[None, :, :]
        - left_points.mean(axis=1)[:, None, :]
    )
    lateral = np.abs(
        delta[..., 0] * left_axis[:, None, 1]
        - delta[..., 1] * left_axis[:, None, 0]
    )
    left_projection = np.einsum("nki,ni->nk", left_points, left_axis)
    right_projection = np.einsum(
        "mki,ni->nmk", right_points, left_axis
    )
    left_low = left_projection.min(axis=1)[:, None]
    left_high = left_projection.max(axis=1)[:, None]
    right_low = right_projection.min(axis=2)
    right_high = right_projection.max(axis=2)
    gap = np.maximum(
        0.0,
        np.maximum(left_low, right_low) - np.minimum(left_high, right_high),
    )
    covered = np.any(
        (angle <= float(params.angle_deg))
        & (lateral <= float(params.lateral_merge_m))
        & (gap <= float(params.longitudinal_gap_m)),
        axis=1,
    )
    maturity = 1.0 - np.exp(-left[:, 4] / float(params.mass_scale))
    weight = left_length * maturity
    total = float(np.sum(weight))
    return (
        float(np.sum(weight[~covered])) / total
        if total > 0.0
        else 0.0
    )


class SupportExpertBankSimulation(CapsuleGreedySimulation):
    checkpoint_format = "support_expert_bank_checkpoint_v1"

    def __init__(
        self,
        cfg,
        *,
        bank_capacity: int,
        transfer_cost: float,
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.bank_capacity = int(bank_capacity)
        self.transfer_cost = float(transfer_cost)
        if self.bank_capacity <= 0:
            raise ValueError("bank capacity must be positive")
        if self.transfer_cost < 0.0:
            raise ValueError("transfer cost cannot be negative")
        self._expert_registry: dict[ExpertKey, ExpertRecord] = {}
        self._expert_banks: list[list[ExpertKey]] = []
        self._local_support: list[tuple[CapsuleRow, ...]] = []
        self._local_versions: list[int] = []
        self._expert_incarnations: list[int] = []
        self._model_transfers = 0
        self._manifest_records = 0
        super().__init__(cfg, resume=None, **kwargs)
        count = int(cfg.num_nodes)
        self._expert_banks = [[] for _ in range(count)]
        self._local_support = [() for _ in range(count)]
        self._local_versions = [0 for _ in range(count)]
        self._expert_incarnations = [0 for _ in range(count)]
        self._communication_assumptions.update(
            {
                "method": "support-driven bounded capsule expert bank",
                "expert_bank_capacity": self.bank_capacity,
                "expert_bank_transfer_cost": self.transfer_cost,
                "expert_bank_acquisition_score": (
                    "novel_capsule_mass_fraction * log1p(experience) "
                    "- transfer_cost"
                ),
                "expert_bank_support_advertisement": (
                    "capsule endpoints, mass, lineage, version, experience"
                ),
                "expert_bank_raw_samples_shared": False,
                "expert_bank_routing": "hard maximum capsule confidence",
                "expert_bank_model_aggregation": False,
                "expert_bank_local_expert_retained": True,
                "local_training": {
                    **asdict(self.training_params),
                    "optimizer": "Adam",
                    "learning_rate": float(cfg.local_lr),
                    "batch_size": int(cfg.local_batch_size),
                    "optimizer_reset": "never; expert parameters are not averaged",
                    "experience_counts_replay": False,
                },
                "round_order": (
                    "synchronous support-driven acquisition, then local replay train"
                ),
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
        index = int(i)
        if index < len(self._expert_banks):
            self._expert_incarnations[index] += 1
            self._local_versions[index] = 0
            self._local_support[index] = ()
            self._expert_banks[index] = []
            self._prune_registry()

    def _rows_for_keys(
        self, keys: list[ExpertKey] | tuple[ExpertKey, ...]
    ) -> tuple[CapsuleRow, ...]:
        return tuple(
            row
            for key in keys
            if key in self._expert_registry
            for row in self._expert_registry[key].capsules
        )

    def _score(
        self, record: ExpertRecord, existing: tuple[CapsuleRow, ...]
    ) -> tuple[float, float, int, int, ExpertKey]:
        novelty = capsule_novelty_fraction(
            record.capsules, existing, self.capsule_params
        )
        value = (
            novelty * math.log1p(max(0, int(record.experience)))
            - self.transfer_cost
        )
        return (
            float(value),
            float(novelty),
            int(record.experience),
            int(record.key[2]),
            record.key,
        )

    def _select_bank(
        self, receiver: int, candidates: list[ExpertKey]
    ) -> list[ExpertKey]:
        newest: dict[tuple[int, int], ExpertKey] = {}
        for key in candidates:
            if key not in self._expert_registry:
                continue
            lineage = key[:2]
            current = newest.get(lineage)
            if current is None or int(key[2]) > int(current[2]):
                newest[lineage] = key
        remaining = list(newest.values())
        selected: list[ExpertKey] = []
        own_lineage = (
            int(receiver),
            int(self._expert_incarnations[int(receiver)]),
        )
        own = newest.get(own_lineage)
        if own is not None:
            selected.append(own)
            remaining.remove(own)
        while remaining and len(selected) < self.bank_capacity:
            existing = self._rows_for_keys(selected)
            chosen = max(
                remaining,
                key=lambda key: self._score(
                    self._expert_registry[key], existing
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _provider_candidate(
        self,
        receiver_keys: list[ExpertKey],
        provider_keys: list[ExpertKey],
    ) -> ExpertKey | None:
        exact = set(receiver_keys)
        receiver_versions = {
            key[:2]: int(key[2]) for key in receiver_keys
        }
        usable = [
            key
            for key in provider_keys
            if key in self._expert_registry
            and key not in exact
            and int(key[2]) > receiver_versions.get(key[:2], -1)
        ]
        if not usable:
            return None
        existing = self._rows_for_keys(receiver_keys)
        return max(
            usable,
            key=lambda key: self._score(self._expert_registry[key], existing),
        )

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
        pre_banks = {
            index: list(self._expert_banks[index])
            for index in neighbours
        }
        next_banks: dict[int, list[ExpertKey]] = {}
        model_messages = 0
        manifest_records = 0
        capsule_values = 0
        for receiver in sorted(neighbours):
            offered = list(pre_banks[receiver])
            for sender in sorted(neighbours[receiver]):
                sender_bank = pre_banks[sender]
                manifest_records += len(sender_bank)
                capsule_values += 5 * sum(
                    len(self._expert_registry[key].capsules)
                    for key in sender_bank
                    if key in self._expert_registry
                )
                candidate = self._provider_candidate(
                    pre_banks[receiver], sender_bank
                )
                if candidate is not None:
                    offered.append(candidate)
                    model_messages += 1
            next_banks[receiver] = self._select_bank(receiver, offered)
        for receiver, bank in next_banks.items():
            self._expert_banks[receiver] = bank
        self._model_transfers += int(model_messages)
        self._manifest_records += int(manifest_records)
        self._network_step_stats.update(
            {
                "expert_bank_model_messages": int(model_messages),
                "expert_bank_manifest_records": int(manifest_records),
                "capsule_scalar_values_sent": int(capsule_values),
                "capsule_payload_bytes": int(4 * capsule_values),
                "expert_bank_receivers": int(len(next_banks)),
            }
        )
        self._train_staged_local_samples(step)
        return int(model_messages)

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        self._local_versions[index] += 1
        key = (
            index,
            int(self._expert_incarnations[index]),
            int(self._local_versions[index]),
        )
        self._expert_registry[key] = ExpertRecord(
            key=key,
            experience=int(self.greedy_m_samples[index]),
            capsules=self._local_support[index],
            model_state=_cpu_state(self.greedy_models[index]),
        )
        own_lineage = key[:2]
        candidates = [
            candidate
            for candidate in self._expert_banks[index]
            if candidate[:2] != own_lineage
        ]
        self._expert_banks[index] = self._select_bank(
            index, [key, *candidates]
        )

    def _train_staged_local_samples(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows_by_receiver: dict[
            int, list[tuple[list[float], float, np.ndarray]]
        ] = {}
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
        receivers = sorted(
            set(rows_by_receiver) | (active & set(self._replay_buffers))
        )
        for receiver in receivers:
            rows = rows_by_receiver.get(receiver, [])
            if rows:
                capsules = deserialize_capsules(
                    self._local_support[receiver]
                )
                for _features, _value, segment in rows:
                    if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0:
                        add_capsule_vectorized(
                            capsules,
                            Capsule.from_segment(segment),
                            self.capsule_params,
                            remote=False,
                        )
                self._local_support[receiver] = serialize_capsules(capsules)
                self.greedy_models[receiver].set_capsules(
                    self._local_support[receiver]
                )
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
            updated = False
            if rows:
                self._train_array_epochs(
                    receiver,
                    X,
                    y,
                    epochs=self.training_params.new_data_epochs,
                    rng=rng,
                )
                updated = True
            if replay is not None and replay.size > 0:
                recent_start = (
                    self.training_params.replay_batches
                    - self.training_params.recent_replay_batches
                )
                for batch_index in range(
                    self.training_params.replay_batches
                ):
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
                updated = True
            if rows:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity, 4
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += int(len(rows))
                self.greedy_n_samples[receiver] = self.greedy_m_samples[
                    receiver
                ]
            if updated:
                self._refresh_local_expert(receiver)
        self._staged_measurements = None
        self._prune_registry()

    def _prune_registry(self) -> None:
        live = {
            key for bank in self._expert_banks for key in bank
        }
        self._expert_registry = {
            key: record
            for key, record in self._expert_registry.items()
            if key in live
        }

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        if int(step) <= self._resume_step and self._resume_payload is not None:
            for row in reversed(
                self._resume_payload.get("fidelity_history", [])
            ):
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
            and bool(self._expert_banks[index])
        ]
        unique = sorted(
            {key for index in active for key in self._expert_banks[index]}
        )
        predictions: dict[ExpertKey, np.ndarray] = {}
        confidences: dict[ExpertKey, np.ndarray] = {}
        if unique:
            template = copy.deepcopy(self.greedy_models[active[0]])
            xt = torch.as_tensor(
                X, dtype=torch.float32, device=self.aux_device
            )
            for key in unique:
                record = self._expert_registry[key]
                template.load_state_dict(record.model_state)
                template.set_capsules(record.capsules)
                template.eval()
                with torch.no_grad():
                    normalized, confidence = template.forward_with_confidence(
                        xt
                    )
                predictions[key] = (
                    normalized.detach().cpu().numpy().reshape(-1)
                )
                confidences[key] = (
                    confidence.detach().cpu().numpy().reshape(-1)
                )
        total_sq = feasible_sq = infeasible_sq = 0.0
        total_count = feasible_count = infeasible_count = 0
        model_rmse: list[float] = []
        confidence_sum = 0.0
        covered_count = 0
        for index in active:
            keys = self._expert_banks[index]
            confidence = np.stack(
                [confidences[key] for key in keys], axis=1
            )
            normalized = np.stack(
                [predictions[key] for key in keys], axis=1
            )
            choice = np.argmax(confidence, axis=1)
            rows = np.arange(len(X))
            selected = normalized[rows, choice]
            conf = confidence[rows, choice]
            prediction = self._denorm_dbm(selected)
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

        def rmse(value: float, count: int) -> float:
            return (
                float(math.sqrt(value / count))
                if count > 0
                else float("nan")
            )

        bank_sizes = [len(self._expert_banks[index]) for index in active]
        capsule_counts = [
            sum(
                len(self._expert_registry[key].capsules)
                for key in self._expert_banks[index]
            )
            for index in active
        ]
        experiences = [
            sum(
                int(self._expert_registry[key].experience)
                for key in self._expert_banks[index]
            )
            for index in active
        ]
        denominator = int(len(active) * len(X))
        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_total": rmse(total_sq, total_count),
            "greedy_mean_model_rmse": (
                float(np.mean(model_rmse)) if model_rmse else float("nan")
            ),
            "greedy_feasible_rmse": rmse(feasible_sq, feasible_count),
            "greedy_infeasible_rmse": rmse(
                infeasible_sq, infeasible_count
            ),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_mean_confidence": (
                confidence_sum / denominator
                if denominator
                else float("nan")
            ),
            "greedy_coverage_at_0_5": (
                covered_count / denominator
                if denominator
                else float("nan")
            ),
            "greedy_mean_capsules": (
                float(np.mean(capsule_counts))
                if capsule_counts
                else float("nan")
            ),
            "greedy_max_capsules": int(max(capsule_counts, default=0)),
            "greedy_mean_experience": (
                float(np.mean(experiences)) if experiences else 0.0
            ),
            "greedy_max_experience": int(max(experiences, default=0)),
            "expert_bank_mean_size": (
                float(np.mean(bank_sizes)) if bank_sizes else 0.0
            ),
            "expert_bank_max_size": int(max(bank_sizes, default=0)),
            "expert_bank_unique_versions": int(len(unique)),
            "expert_bank_model_transfers": int(self._model_transfers),
            "expert_bank_manifest_records": int(self._manifest_records),
        }
        self.fidelity_history.append(row)
        return row

    def _save_checkpoint(self, step: int) -> None:
        experienced = [
            index
            for index, value in enumerate(self.greedy_m_samples)
            if int(value) > 0
        ]
        registry = {
            key: {
                "experience": int(record.experience),
                "capsules": record.capsules,
                "model_state": self._cpu_tree(record.model_state),
            }
            for key, record in self._expert_registry.items()
        }
        payload = {
            "format": self.checkpoint_format,
            "step": int(step),
            "trace": str(self.measurement_trace_in),
            "testset": str(self._testset_path),
            "reception_floor_dbm": self.reception_floor_dbm,
            "capsule_params": asdict(self.capsule_params),
            "gate_params": asdict(self.gate_params),
            "training_params": asdict(self.training_params),
            "bank_capacity": self.bank_capacity,
            "transfer_cost": self.transfer_cost,
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
            "local_support": self._local_support,
            "local_versions": self._local_versions,
            "expert_incarnations": self._expert_incarnations,
            "expert_registry": registry,
            "expert_banks": self._expert_banks,
            "replay_buffers": {
                int(index): replay.state_dict()
                for index, replay in self._replay_buffers.items()
            },
            "model_transfers": self._model_transfers,
            "manifest_records": self._manifest_records,
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
            "live_expert_versions": int(len(self._expert_registry)),
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
            f"[SUPPORT-EXPERT-BANK] checkpoint step={step} "
            f"models={len(experienced)} experts={len(self._expert_registry)} "
            f"path={target}",
            flush=True,
        )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("format") != self.checkpoint_format:
            raise ValueError(f"unsupported checkpoint format in {path}")
        if int(payload.get("bank_capacity", -1)) != self.bank_capacity:
            raise ValueError("checkpoint bank capacity differs")
        if float(payload.get("transfer_cost", float("nan"))) != self.transfer_cost:
            raise ValueError("checkpoint transfer cost differs")
        if payload.get("capsule_params") != asdict(self.capsule_params):
            raise ValueError("checkpoint capsule parameters differ")
        if payload.get("gate_params") != asdict(self.gate_params):
            raise ValueError("checkpoint gate parameters differ")
        if payload.get("training_params") != asdict(self.training_params):
            raise ValueError("checkpoint training parameters differ")
        self._resume_step = int(payload["step"])
        experience = [int(value) for value in payload["experience"]]
        self.greedy_m_samples = list(experience)
        self.greedy_n_samples = list(experience)
        for raw_index, state in payload["models"].items():
            self.greedy_models[int(raw_index)].load_state_dict(state)
        for raw_index, state in payload["optimizers"].items():
            self.greedy_opts[int(raw_index)].load_state_dict(state)
        self._local_support = payload["local_support"]
        self._local_versions = [
            int(value) for value in payload["local_versions"]
        ]
        self._expert_incarnations = [
            int(value) for value in payload["expert_incarnations"]
        ]
        self._expert_registry = {
            tuple(int(value) for value in key): ExpertRecord(
                key=tuple(int(value) for value in key),
                experience=int(data["experience"]),
                capsules=tuple(data["capsules"]),
                model_state=data["model_state"],
            )
            for key, data in payload["expert_registry"].items()
        }
        self._expert_banks = [
            [tuple(int(value) for value in key) for key in bank]
            for bank in payload["expert_banks"]
        ]
        self._replay_buffers = {
            int(index): ReplayBuffer.from_state_dict(state)
            for index, state in payload["replay_buffers"].items()
        }
        for index, support in enumerate(self._local_support):
            self.greedy_models[index].set_capsules(support)
        self._model_transfers = int(payload.get("model_transfers", 0))
        self._manifest_records = int(payload.get("manifest_records", 0))
        self._resume_payload = payload
        self._resume_logs_restored = False
        print(
            f"[SUPPORT-EXPERT-BANK] resumed step={self._resume_step} "
            f"from {path}",
            flush=True,
        )


def self_test() -> None:
    capsule_greedy_self_test()
    params = CapsuleParams(
        angle_deg=15.0,
        lateral_merge_m=12.0,
        longitudinal_gap_m=20.0,
        sigma_perp_m=8.0,
        sigma_parallel_m=15.0,
        mass_scale=3.0,
    )
    first = ((0.0, 0.0, 20.0, 0.0, 4.0),)
    same = ((2.0, 1.0, 22.0, 1.0, 4.0),)
    other = ((0.0, 30.0, 20.0, 30.0, 4.0),)
    assert capsule_novelty_fraction(first, (), params) == 1.0
    assert capsule_novelty_fraction(same, first, params) == 0.0
    assert capsule_novelty_fraction(other, first, params) == 1.0
    assert _support_weight(first, params) > 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank-capacity", type=int, default=4)
    parser.add_argument("--transfer-cost", type=float, default=0.0)
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
    parser.add_argument("--reception-floor-dbm", type=float, default=-100.0)
    parser.add_argument("--angle-deg", type=float, default=15.0)
    parser.add_argument("--lateral-merge-m", type=float, default=12.0)
    parser.add_argument("--longitudinal-gap-m", type=float, default=20.0)
    parser.add_argument("--sigma-perp-m", type=float, default=8.0)
    parser.add_argument("--sigma-parallel-m", type=float, default=15.0)
    parser.add_argument("--sigma-angle-deg", type=float, default=15.0)
    parser.add_argument("--mass-scale", type=float, default=3.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("support expert bank self-test passed")
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
    capsule_params = CapsuleParams(
        angle_deg=float(args.angle_deg),
        lateral_merge_m=float(args.lateral_merge_m),
        longitudinal_gap_m=float(args.longitudinal_gap_m),
        sigma_perp_m=float(args.sigma_perp_m),
        sigma_parallel_m=float(args.sigma_parallel_m),
        mass_scale=float(args.mass_scale),
    )
    gate_params = GateParams(
        sigma_perp_m=float(args.sigma_perp_m),
        sigma_parallel_m=float(args.sigma_parallel_m),
        sigma_angle_deg=float(args.sigma_angle_deg),
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
        # Required by the shared configuration schema; this experiment never
        # invokes model-parameter merging.
        merge_strategy="average",
        fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_eval_every=checkpoint_every,
        final_fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_final_steps=(sim_steps,),
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    simulation = SupportExpertBankSimulation(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=reception_floor_dbm,
        capsule_params=capsule_params,
        gate_params=gate_params,
        training_params=training_params,
        bank_capacity=int(args.bank_capacity),
        transfer_cost=float(args.transfer_cost),
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
