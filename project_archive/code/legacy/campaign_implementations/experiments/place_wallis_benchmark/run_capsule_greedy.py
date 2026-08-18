#!/usr/bin/env python3
"""Greedy MLP sharing with finite variable-width support-plane geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.capsule_greedy.run_capsule_greedy import (  # noqa: E402
    average_state_dicts,
)
from experiments.place_wallis_benchmark.ribbon_support import (  # noqa: E402
    Ribbon as Capsule,
    RibbonGatedMLP as CapsuleGatedMLP,
    RibbonParams as CapsuleParams,
    RibbonRow as CapsuleRow,
    GateParams,
    add_ribbon_vectorized as add_capsule_vectorized,
    deserialize_ribbons as deserialize_capsules,
    remote_union,
    ribbon_delta as capsule_delta,
    serialize_ribbons as serialize_capsules,
    self_test as ribbon_self_test,
)
from experiments.place_wallis_benchmark.run_equal_greedy import (  # noqa: E402
    DEFAULT_NET,
    DEFAULT_TESTSET,
    DEFAULT_TRACE,
    EqualGreedySimulation,
    TrainingParams,
    atomic_json,
    validate_dataset,
)
from experiments.place_wallis_benchmark.tail_metrics import (  # noqa: E402
    make_tail_evaluation_steps,
    temporal_metric_summary,
)
from rl_reward_experiment.config import build_config_from_env  # noqa: E402

DEFAULT_RESULTS = (
    ROOT
    / "artifacts/place_wallis_benchmark/methods/"
    "support_plane_greedy_eval50_tail10x25"
)

SUPPORT_VARIANT = "outer-envelope"
SUPPORT_RECORD_FLOATS = 9
SUPPORT_PAYLOAD_DESCRIPTION = (
    "representative endpoints, four border offsets, and mass"
)
SUPPORT_MERGE_DESCRIPTION = (
    "capsule-like representative clustering plus variable outer borders"
)

OriginSummary = tuple[int, tuple[CapsuleRow, ...]]


def _parallel_support_union(payload: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
    signature, union_function, plane_sets, params = payload
    return signature, union_function([(), *plane_sets], params)


class CapsuleGreedySimulation(EqualGreedySimulation):
    checkpoint_format = "place_wallis_support_plane_greedy_checkpoint_v4"

    def __init__(
        self,
        cfg,
        *,
        capsule_params: CapsuleParams,
        gate_params: GateParams,
        experience_weighted: bool = False,
        binary_support: bool = False,
        lazy_support_rebuild: bool = False,
        support_rebuild_workers: int = 1,
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.capsule_params = capsule_params
        self.gate_params = gate_params
        self.experience_weighted = bool(experience_weighted)
        self.binary_support = bool(binary_support)
        self.lazy_support_rebuild = bool(lazy_support_rebuild)
        self.support_rebuild_workers = max(1, int(support_rebuild_workers))
        self._supports: list[tuple[CapsuleRow, ...]] = [
            () for _ in range(int(cfg.num_nodes))
        ]
        self._support_knowledge: list[dict[int, OriginSummary]] = [
            {index: (0, ())} for index in range(int(cfg.num_nodes))
        ]
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
        wrapped: list[CapsuleGatedMLP] = []
        optimizers: list[optim.Optimizer] = []
        for base in self.greedy_models:
            model = CapsuleGatedMLP(
                base,
                map_size_m=float(cfg.map_size),
                floor_prior_norm=floor_prior_norm,
                ribbon_params=capsule_params,
                gate_params=gate_params,
                binary_support=self.binary_support,
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
                "method": "greedy MLP with finite variable-width support-plane geometry",
                "support_shared": True,
                "support_payload": (
                    f"{SUPPORT_PAYLOAD_DESCRIPTION} "
                    f"({SUPPORT_RECORD_FLOATS} float32 values)"
                ),
                "support_state": "origin-versioned summaries plus merged operational set",
                "support_merge": SUPPORT_MERGE_DESCRIPTION,
                "remote_mass_merge": "maximum (idempotent gossip)",
                "support_plane_merge_params": {
                    "angle_deg": float(capsule_params.angle_deg),
                    "lateral_merge_m": float(capsule_params.lateral_merge_m),
                    "longitudinal_gap_m": float(
                        capsule_params.longitudinal_gap_m
                    ),
                    "initial_half_width_m": float(
                        capsule_params.initial_half_width_m
                    ),
                },
                "support_plane_gate_params": asdict(gate_params),
                "support_gate": (
                    "hard binary geometry"
                    if self.binary_support
                    else "smooth Gaussian geometry"
                ),
                "model_average_weights": (
                    "experience" if self.experience_weighted else "equal"
                ),
                "operational_support_refresh": (
                    "evaluation checkpoints only"
                    if self.lazy_support_rebuild else "every received update"
                ),
                "support_rebuild_workers": self.support_rebuild_workers,
                "experience_shared": self.experience_weighted,
                "experience_merge": (
                    "maximum aggregated experience, then add fresh local samples"
                    if self.experience_weighted
                    else "local fresh samples only"
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
        if 0 <= int(i) < len(self._supports):
            self._supports[int(i)] = ()
            self._support_knowledge[int(i)] = {int(i): (0, ())}
        if 0 <= int(i) < len(self.greedy_models):
            model = self.greedy_models[int(i)]
            if isinstance(model, CapsuleGatedMLP):
                model.set_ribbons(())

    def _rebuild_operational_support(self, receiver: int) -> None:
        """Rebuild support from the newest summary of every known origin."""

        knowledge = self._support_knowledge[int(receiver)]
        rows = remote_union(
            [(), *(knowledge[origin][1] for origin in sorted(knowledge))],
            self.capsule_params,
        )
        self._supports[int(receiver)] = rows
        self.greedy_models[int(receiver)].set_ribbons(rows)

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            return super()._greedy_share_step(zone_nodes, contact_links)
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
        pre_knowledge = {
            index: dict(self._support_knowledge[index]) for index in neighbours
        }
        next_knowledge: dict[int, dict[int, OriginSummary]] = {}
        next_deltas: dict[int, list[tuple[CapsuleRow, ...]]] = {}
        support_records_sent = 0
        support_origin_records_sent = 0
        for receiver in sorted(neighbours):
            combined = dict(pre_knowledge[receiver])
            deltas: list[tuple[CapsuleRow, ...]] = []
            for sender in sorted(neighbours[receiver]):
                bundle = pre_knowledge[sender]
                support_origin_records_sent += len(bundle)
                support_records_sent += sum(
                    len(summary[1]) for summary in bundle.values()
                )
                for origin, summary in bundle.items():
                    previous = combined.get(origin)
                    if previous is None or int(summary[0]) > int(previous[0]):
                        if not self.lazy_support_rebuild:
                            deltas.append(
                                capsule_delta(
                                    () if previous is None else previous[1],
                                    summary[1],
                                )
                            )
                        combined[origin] = summary
            next_knowledge[receiver] = combined
            next_deltas[receiver] = deltas
        for receiver in sorted(neighbours):
            self._support_knowledge[receiver] = next_knowledge[receiver]
            if not self.lazy_support_rebuild:
                rows = remote_union(
                    [self._supports[receiver], *next_deltas[receiver]],
                    self.capsule_params,
                )
                self._supports[receiver] = rows
                self.greedy_models[receiver].set_ribbons(rows)
        self._network_step_stats.update(
            {
                "capsule_support_records_sent": int(support_records_sent),
                "capsule_support_payload_bytes": int(
                    4 * SUPPORT_RECORD_FLOATS * support_records_sent
                ),
                "capsule_support_origin_records_sent": int(
                    support_origin_records_sent
                ),
                "capsule_max_operational_count": int(
                    max(
                        (len(self._supports[index]) for index in neighbours),
                        default=0,
                    )
                ),
            }
        )
        if self.experience_weighted:
            participants = sorted(neighbours)
            pre_states = {
                index: {
                    name: value.detach().clone()
                    for name, value in self.greedy_models[index]
                    .state_dict()
                    .items()
                }
                for index in participants
            }
            pre_experience = {
                index: int(self.greedy_m_samples[index])
                for index in participants
            }
            for receiver in participants:
                members = [receiver, *sorted(neighbours[receiver])]
                state = average_state_dicts(
                    [pre_states[index] for index in members],
                    [pre_experience[index] for index in members],
                )
                self.greedy_models[receiver].load_state_dict(state)
                self.greedy_opts[receiver].state.clear()
                merged_experience = max(
                    pre_experience[index] for index in members
                )
                self.greedy_m_samples[receiver] = merged_experience
                self.greedy_n_samples[receiver] = merged_experience
            parameter_count = (
                sum(
                    tensor.numel()
                    for tensor in next(iter(pre_states.values())).values()
                )
                if pre_states
                else 0
            )
            transfers = 2 * len(links)
            self._network_step_stats.update(
                {
                    "synchronous_greedy_receivers": int(len(participants)),
                    "experience_weighted_model_transfers": int(transfers),
                    "model_payload_bytes": int(
                        4 * parameter_count * transfers
                    ),
                }
            )
            self._train_local(step)
            return int(transfers)
        return super()._greedy_share_step(zone_nodes, contact_links)

    def _update_local_support(self) -> None:
        measurements = self._staged_measurements or []
        segments: dict[int, list[np.ndarray]] = {}
        for _zone, tx_idx, rx_idx, _value in measurements:
            tx = self.nodes[int(tx_idx)].node
            rx = self.nodes[int(rx_idx)].node
            segment = np.asarray(
                [[tx.x, tx.y], [rx.x, rx.y]], dtype=np.float64
            )
            if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0:
                segments.setdefault(int(rx_idx), []).append(segment)
        for receiver, observations in segments.items():
            version, local_rows = self._support_knowledge[receiver].get(
                receiver, (0, ())
            )
            capsules = deserialize_capsules(local_rows)
            for segment in observations:
                add_capsule_vectorized(
                    capsules,
                    Capsule.from_segment(
                        segment,
                        half_width=self.capsule_params.initial_half_width_m,
                    ),
                    self.capsule_params,
                    remote=False,
                )
            updated_local = serialize_capsules(capsules)
            self._support_knowledge[receiver][receiver] = (
                int(version) + 1, updated_local
            )
            if not self.lazy_support_rebuild:
                rows = remote_union(
                    [self._supports[receiver], capsule_delta(local_rows, updated_local)],
                    self.capsule_params,
                )
                self._supports[receiver] = rows
                self.greedy_models[receiver].set_ribbons(rows)

    def _train_local(self, step: int) -> None:
        self._update_local_support()
        super()._train_local(step)

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
        signature_groups: dict[
            tuple[tuple[int, int], ...], list[int]
        ] = {}
        for index in active:
            knowledge = self._support_knowledge[index]
            signature = tuple(
                (origin, int(knowledge[origin][0]))
                for origin in sorted(knowledge)
            )
            signature_groups.setdefault(signature, []).append(index)

        group_payloads = []
        for signature, indices in signature_groups.items():
            knowledge = self._support_knowledge[indices[0]]
            plane_sets = tuple(
                knowledge[origin][1] for origin in sorted(knowledge)
            )
            group_payloads.append((
                signature, remote_union, plane_sets, self.capsule_params
            ))
        worker_count = min(self.support_rebuild_workers, len(group_payloads))
        if worker_count > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                rebuilt_groups = list(executor.map(
                    _parallel_support_union, group_payloads
                ))
        else:
            rebuilt_groups = [
                _parallel_support_union(payload) for payload in group_payloads
            ]
        for signature, rows in rebuilt_groups:
            for index in signature_groups[signature]:
                self._supports[index] = rows
                self.greedy_models[index].set_ribbons(rows)
        total_sq = feasible_sq = infeasible_sq = 0.0
        confidence_sum = 0.0
        predicted_total = predicted_infeasible = 0
        covered_total = covered_feasible = covered_infeasible = 0
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
            covered = conf >= 0.5
            positive = prediction > self.reception_floor_dbm
            total_sq += float(error_sq.sum())
            feasible_sq += float(error_sq[feasible].sum())
            infeasible_sq += float(error_sq[~feasible].sum())
            confidence_sum += float(conf.sum())
            covered_total += int(covered.sum())
            covered_feasible += int(covered[feasible].sum())
            covered_infeasible += int(covered[~feasible].sum())
            predicted_total += int(positive.sum())
            predicted_infeasible += int(positive[~feasible].sum())
        total_count = len(active) * len(truth)
        feasible_count = len(active) * int(feasible.sum())
        infeasible_count = len(active) * int((~feasible).sum())

        def rmse(square_sum: float, count: int) -> float:
            return (
                float(math.sqrt(square_sum / count))
                if count
                else float("nan")
            )

        def ratio(value: float, count: int) -> float:
            return float(value / count) if count else float("nan")

        counts = [len(self._supports[index]) for index in active]
        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_total": rmse(total_sq, total_count),
            "greedy_feasible_rmse": rmse(feasible_sq, feasible_count),
            "greedy_infeasible_rmse": rmse(infeasible_sq, infeasible_count),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_mean_confidence": ratio(confidence_sum, total_count),
            "greedy_coverage_at_0_5": ratio(covered_total, total_count),
            "greedy_feasible_coverage_at_0_5": ratio(
                covered_feasible, feasible_count
            ),
            "greedy_infeasible_leakage_at_0_5": ratio(
                covered_infeasible, infeasible_count
            ),
            "greedy_predicted_feasible_fraction": ratio(
                predicted_total, total_count
            ),
            "greedy_non_feasible_false_positive_rate": ratio(
                predicted_infeasible, infeasible_count
            ),
            "greedy_mean_capsules": (
                float(np.mean(counts)) if counts else float("nan")
            ),
            "greedy_max_capsules": int(max(counts, default=0)),
        }
        self.fidelity_history.append(row)
        return row

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        support_temporal = temporal_metric_summary(
            self.fidelity_history,
            evaluation_steps=self.tail_evaluation_steps,
            metric_keys=(
                "greedy_mean_confidence",
                "greedy_coverage_at_0_5",
                "greedy_feasible_coverage_at_0_5",
                "greedy_infeasible_leakage_at_0_5",
                "greedy_mean_capsules",
                "greedy_max_capsules",
            ),
        )
        latest = self.fidelity_history[-1] if self.fidelity_history else {}
        status_path = output / "checkpoint_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            {
                "mean_capsules": latest.get("greedy_mean_capsules"),
                "max_capsules": latest.get("greedy_max_capsules"),
                "support_temporal_evaluation": support_temporal,
            }
        )
        atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if SUPPORT_VARIANT == "finite-corridor-capsules":
            method_id = (
                "experience_weighted_capsule_greedy"
                if self.experience_weighted
                else "capsule_greedy"
            )
            method_name = (
                "Experience-weighted corridor-capsule greedy sharing"
                if self.experience_weighted
                else "Finite corridor-capsule greedy sharing"
            )
        elif SUPPORT_VARIANT == "overlapping-straight-planes":
            method_id = (
                "experience_weighted_overlapping_plane_greedy"
                if self.experience_weighted
                else "overlapping_plane_greedy"
            )
            method_name = (
                "Experience-weighted overlapping-plane greedy sharing"
                if self.experience_weighted
                else "Equal-weight overlapping-plane greedy sharing"
            )
        elif self.binary_support:
            method_id = (
                "experience_weighted_binary_support_plane_greedy"
                if self.experience_weighted
                else "binary_support_plane_greedy"
            )
            method_name = (
                "Experience-weighted binary support-plane greedy sharing"
                if self.experience_weighted
                else "Equal-weight binary support-plane greedy sharing"
            )
        else:
            method_id = (
                "experience_weighted_support_plane_greedy"
                if self.experience_weighted
                else "support_plane_greedy"
            )
            method_name = (
                "Experience-weighted variable-width support-plane greedy sharing"
                if self.experience_weighted
                else "Equal-weight variable-width support-plane greedy sharing"
            )
        method_tag = os.environ.get("METHOD_TAG", "").strip()
        if method_tag:
            method_id = f"{method_id}_{method_tag}"
            method_name = f"{method_name} ({method_tag.replace('_', ' ')})"
        metrics["method"] = {
            "id": method_id,
            "name": method_name,
            "model": "4-64-64-1 MLP",
        }
        metrics["dissemination"] = {
            "providers": "all feasible neighbours",
            "model_weights": (
                "experience" if self.experience_weighted else "equal"
            ),
            "experience_update": (
                "max of aggregate members, then fresh local sample count"
                if self.experience_weighted
                else "fresh local sample count only"
            ),
        }
        metrics["support_plane_params"] = asdict(self.capsule_params)
        metrics["gate_params"] = asdict(self.gate_params)
        metrics["support"] = {
            "gate": "binary" if self.binary_support else "smooth",
            "mean_support_planes_latest": latest.get("greedy_mean_capsules"),
            "max_support_planes_latest": latest.get("greedy_max_capsules"),
            "mean_confidence_tail": support_temporal["mean"][
                "greedy_mean_confidence"
            ],
            "coverage_at_0_5_tail": support_temporal["mean"][
                "greedy_coverage_at_0_5"
            ],
            "feasible_coverage_at_0_5_tail": support_temporal["mean"][
                "greedy_feasible_coverage_at_0_5"
            ],
            "infeasible_leakage_at_0_5_tail": support_temporal["mean"][
                "greedy_infeasible_leakage_at_0_5"
            ],
        }
        metrics["support_temporal_evaluation"] = support_temporal
        atomic_json(metrics_path, metrics)
        if latest:
            print(
                f"[SUPPORT-PLANE] metrics={step} "
                f"coverage={100 * float(latest['greedy_coverage_at_0_5']):.1f}% "
                f"planes={float(latest['greedy_mean_capsules']):.1f}",
                flush=True,
            )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("support_plane_params") != asdict(self.capsule_params):
            raise ValueError("checkpoint support-plane parameters differ")
        if payload.get("gate_params") != asdict(self.gate_params):
            raise ValueError("checkpoint support-plane gate parameters differ")
        if bool(payload.get("experience_weighted", False)) != (
            self.experience_weighted
        ):
            raise ValueError("checkpoint aggregation mode differs")
        if bool(payload.get("binary_support", False)) != self.binary_support:
            raise ValueError("checkpoint support-gate mode differs")
        if payload.get("support_variant", "outer-envelope") != SUPPORT_VARIANT:
            raise ValueError("checkpoint support representation differs")
        if "supports" not in payload or "support_knowledge" not in payload:
            raise ValueError("checkpoint does not contain support-plane geometry")
        super()._load_checkpoint(path)
        self._supports = [tuple(rows) for rows in payload["supports"]]
        self._support_knowledge = payload["support_knowledge"]
        for index, rows in enumerate(self._supports):
            self.greedy_models[index].set_ribbons(rows)


def self_test() -> None:
    ribbon_self_test()


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
    parser.add_argument("--angle-deg", type=float, default=12.0)
    parser.add_argument("--lateral-merge-m", type=float, default=8.0)
    parser.add_argument("--longitudinal-gap-m", type=float, default=10.0)
    parser.add_argument("--initial-half-width-m", type=float, default=1.5)
    parser.add_argument("--sigma-perp-m", type=float, default=2.5)
    parser.add_argument("--sigma-parallel-m", type=float, default=4.0)
    parser.add_argument("--sigma-angle-deg", type=float, default=10.0)
    parser.add_argument("--mass-scale", type=float, default=3.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--tail-eval-count", type=int, default=10)
    parser.add_argument("--tail-eval-stride", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--experience-weighted", action="store_true")
    parser.add_argument("--binary-support", action="store_true")
    parser.add_argument("--lazy-support-rebuild", action="store_true")
    parser.add_argument("--support-rebuild-workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("Place Wallis support-plane greedy self-test passed")
        return 0
    metadata = validate_dataset(args.trace.resolve(), args.testset.resolve())
    sim_steps = int(metadata["sim_steps"] if args.sim_steps is None else args.sim_steps)
    if sim_steps > int(metadata["sim_steps"]):
        raise ValueError("requested steps exceed trace")
    tail_steps = make_tail_evaluation_steps(
        sim_steps,
        count=int(args.tail_eval_count),
        stride=int(args.tail_eval_stride),
    )
    results_dir = args.results_dir.resolve()
    resume = args.resume
    automatic = results_dir / "checkpoint_latest.pt"
    if resume is None and args.resume_if_exists and automatic.exists():
        resume = automatic
    capsule_params = CapsuleParams(
        angle_deg=float(args.angle_deg),
        lateral_merge_m=float(args.lateral_merge_m),
        longitudinal_gap_m=float(args.longitudinal_gap_m),
        initial_half_width_m=float(args.initial_half_width_m),
        mass_scale=float(args.mass_scale),
    )
    gate_params = GateParams(
        sigma_perp_m=float(args.sigma_perp_m),
        sigma_parallel_m=float(args.sigma_parallel_m),
        sigma_angle_deg=float(args.sigma_angle_deg),
    )
    training = TrainingParams(
        replay_capacity=int(args.replay_capacity),
        new_data_epochs=int(args.new_data_epochs),
        replay_batches=int(args.replay_batches),
        recent_replay_batches=int(args.recent_replay_batches),
        recent_window=int(args.recent_window),
        gradient_clip_norm=float(args.gradient_clip_norm),
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
        fidelity_final_steps=tail_steps,
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    simulation = CapsuleGreedySimulation(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=-100.0,
        training_params=training,
        tail_evaluation_steps=tail_steps,
        capsule_params=capsule_params,
        gate_params=gate_params,
        experience_weighted=bool(args.experience_weighted),
        binary_support=bool(args.binary_support),
        lazy_support_rebuild=bool(args.lazy_support_rebuild),
        support_rebuild_workers=int(args.support_rebuild_workers),
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
