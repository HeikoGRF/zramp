"""Budgeted, receiver-batched bootstrap zRAMP utility selection."""

from __future__ import annotations

import csv
import math
import random
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

import rl_reward_experiment.sim as rre_sim
from rl_reward_experiment.node_state import bound_raw_samples, saturate_n_samples

from .aggregation import (
    experience_weights,
    weighted_average,
)
from .budget import sample_pull_capacity, select_random_subset, select_top_k
from .feedback import ModelSnapshot, PendingPull, advance_pending_pull
from .metadata import (
    CKA_PROBE_COUNT,
    CKA_SIGNATURE_FLOATS,
    CURRENT_OBSERVATION_FEATURES,
    FEATURE_SETS,
    STUDY_OBSERVATION_FEATURES,
    build_cka_signature,
    build_metadata,
    build_prediction_signature,
    build_observation,
)
from .prequential import evaluate_finetuned_pair
from .simulation import BootstrapPolicySharingSimulation
from .utility import UtilityAgent, aggregate_policy_states


POLICY_EXPERIENCE_BYTES = 8


AGGREGATION_LOG_FIELDS = (
    "step",
    "mode",
    "receiver_idx",
    "zone",
    "neighbor_count",
    "capacity",
    "exploratory",
    "selected_provider_ids",
    "policy_source_ids",
    "policy_experiences",
    "policy_weights",
    "policy_bytes",
    "aggregation_weights",
    "metadata_bytes",
    "model_bytes",
    "current_samples",
    "current_rmse_before",
    "current_rmse_aggregated",
    "current_rmse_after_local",
)
FEEDBACK_LOG_FIELDS = (
    "mode",
    "receiver_idx",
    "provider_idx",
    "zone",
    "step_started",
    "terminal_reason",
    "step_finalized",
    "n_samples",
    "feedback_mode",
    "future_steps",
    "local_samples",
    "future_samples",
    "fine_tune_epochs",
    "baseline_rmse",
    "pairwise_rmse",
    "realized_utility",
    *tuple(dict.fromkeys((*CURRENT_OBSERVATION_FEATURES, *STUDY_OBSERVATION_FEATURES))),
)
UTILITY_TRAINING_LOG_FIELDS = (
    "step",
    "mode",
    "receiver_idx",
    "new_rewards",
    "replay_size",
    "train_steps",
    "loss",
)


@dataclass(frozen=True)
class ProviderCandidate:
    provider_idx: int
    observation: torch.Tensor
    snapshot: ModelSnapshot
    score: float


@dataclass
class ReceiverPullPlan:
    step: int
    zone: int
    receiver_idx: int
    mode: str
    receiver_snapshot: ModelSnapshot
    candidates: list[ProviderCandidate]
    capacity: int
    selected_provider_ids: list[int]
    exploratory: bool

    policy_source_ids: list[int]
    policy_experiences: list[int]
    policy_weights: torch.Tensor
    policy_bytes: int


def _state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return int(
        sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in state.values())
    )


def _format_ids(values: list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _format_weights(labels: list[str], weights: torch.Tensor) -> str:
    return ";".join(
        f"{label}:{float(weight):.9g}"
        for label, weight in zip(labels, weights.detach().cpu().tolist())
    )


class UtilitySelectionBootstrapSimulation(BootstrapPolicySharingSimulation):
    """Select up to a stochastic pull budget, then aggregate exactly once."""

    policy_transfer_rule = "all_feasible_experience_weighted_policy_average"

    def __init__(
        self,
        *args,
        pull_budget: float = 1.0,
        utility_exploration_prob: float = 0.1,
        utility_evaluation: bool = False,
        utility_hidden_dim: int = 64,
        utility_train_updates: int = 4,
        aggregation_experience_epsilon: float = 1.0,
        utility_horizon: int = 2,
        utility_feedback_mode: str = "frozen",
        **kwargs,
    ) -> None:
        if not math.isfinite(float(pull_budget)) or float(pull_budget) < 0.0:
            raise ValueError("pull_budget must be finite and non-negative")
        if not 0.0 <= float(utility_exploration_prob) <= 1.0:
            raise ValueError("utility_exploration_prob must be in [0, 1]")
        if int(utility_hidden_dim) <= 0:
            raise ValueError("utility_hidden_dim must be positive")
        if int(utility_train_updates) < 0:
            raise ValueError("utility_train_updates must be non-negative")
        if float(aggregation_experience_epsilon) <= 0.0:
            raise ValueError("aggregation_experience_epsilon must be positive")
        if int(utility_horizon) <= 0:
            raise ValueError("utility_horizon must be positive")
        if str(utility_feedback_mode) not in {"frozen", "finetune-window"}:
            raise ValueError("utility_feedback_mode must be frozen or finetune-window")

        self.pull_budget = float(pull_budget)
        self.utility_exploration_prob = float(utility_exploration_prob)
        self.utility_evaluation = bool(utility_evaluation)
        self.utility_hidden_dim = int(utility_hidden_dim)
        self.utility_train_updates = int(utility_train_updates)
        self.aggregation_experience_epsilon = float(
            aggregation_experience_epsilon
        )
        self.utility_horizon = int(utility_horizon)
        self.utility_feedback_mode = str(utility_feedback_mode)

        self.utility_agents: dict[str, list[UtilityAgent]] = {}
        self._capacity_rngs: list[random.Random] = []
        self._staged_pull_plans: dict[tuple[str, int], ReceiverPullPlan] = {}
        self._utility_pending: dict[tuple[str, int], list[PendingPull]] = defaultdict(list)
        self._utility_exit_dirty: set[tuple[str, int]] = set()
        self._utility_current_samples: dict[
            int, tuple[int, np.ndarray, np.ndarray]
        ] = {}
        self._last_predictor_pull_step: dict[
            tuple[str, int, int, int], int
        ] = {}
        self._utility_step_metadata_bytes: Counter[str] = Counter()
        self._utility_step_model_bytes: Counter[str] = Counter()
        self._utility_step_selected: Counter[str] = Counter()
        self._utility_new_rewards: Counter[tuple[str, int]] = Counter()
        self._utility_step_policy_bytes: Counter[str] = Counter()
        self._utility_csv_files: dict[str, object] = {}
        self._utility_csv_writers: dict[str, csv.DictWriter] = {}
        self._utility_csv_counts: Counter[str] = Counter()

        kwargs["local_policy_share"] = False
        requested_features = kwargs.pop("policy_state_features", "current6")
        feature_key = "current6" if requested_features is None else str(requested_features)
        if feature_key not in FEATURE_SETS:
            raise ValueError(f"unknown utility feature set {feature_key!r}")
        self.feature_set = feature_key
        self.observation_features = FEATURE_SETS[feature_key]
        super().__init__(*args, policy_state_features="current6", **kwargs)
        self.zramp_policy_mode = "utility-top-k"
        self.local_policy_initial_pull = "utility-warmup-random"
        self.local_policy_initial_pull_probability = 1.0
        self._aggregation_log_path = (
            Path(self.cfg.results_dir) / "utility_aggregation.csv"
        )
        self._feedback_log_path = (
            Path(self.cfg.results_dir) / "utility_feedback.csv"
        )
        self._utility_training_log_path = (
            Path(self.cfg.results_dir) / "utility_training.csv"
        )
        self._cka_probe_cache: dict[tuple[int, float | None], torch.Tensor] = {}

    # ---------------------------------------------------------- initialization

    def _init_local_policy_agents(self) -> None:
        """Create one local scalar utility estimator per vehicle and mode."""

        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()
        self.utility_agents.clear()
        self._capacity_rngs = [
            random.Random(int(self.cfg.seed) + 7_700_003 + 65_537 * node_idx)
            for node_idx in range(int(self.cfg.num_nodes))
        ]
        for mode_id in self.agents:
            mode_offset = int(zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000)
            model_seed = int(self.cfg.seed) + 5_900_011 + mode_offset
            agents: list[UtilityAgent] = []
            for node_idx in range(int(self.cfg.num_nodes)):
                seed = (
                    int(self.cfg.seed)
                    + 5_900_011
                    + mode_offset
                    + 104_729 * node_idx
                )
                agents.append(
                    UtilityAgent(
                        observation_dim=len(self.observation_features),
                        hidden_dim=self.utility_hidden_dim,
                        device=self.device,
                        learning_rate=float(self.cfg.rl_lr),
                        batch_size=int(self.cfg.rl_batch_size),
                        replay_capacity=int(self.cfg.replay_capacity),
                        rng_seed=seed,
                        model_seed=model_seed,
                    )
                )
            self.utility_agents[mode_id] = agents
            self._local_policy_pending_transitions[mode_id] = [
                0 for _ in agents
            ]
            self._local_policy_versions[mode_id] = [0 for _ in agents]

    # --------------------------------------------------------------- snapshots

    def _cka_probes(self, zone: int) -> torch.Tensor:
        """Return deterministic same-zone public probes for CKA metadata."""

        time_value = self._predictor_time_feature()
        key = (int(zone), time_value)
        cached = self._cka_probe_cache.get(key)
        if cached is not None:
            return cached
        x_lo, x_hi, y_lo, y_hi = rre_sim.zone_bounds(
            int(zone), self.cfg.map_size, self.cfg.num_zones
        )
        points: list[list[float]] = []
        for idx in range(CKA_PROBE_COUNT):
            col, row = idx % 4, idx // 4
            tx = (
                x_lo + (col + 0.5) * (x_hi - x_lo) / 4.0,
                y_lo + (row + 0.5) * (y_hi - y_lo) / 4.0,
            )
            paired = (idx * 5 + 3) % CKA_PROBE_COUNT
            rx_col, rx_row = paired % 4, paired // 4
            rx = (
                x_lo + (rx_col + 0.5) * (x_hi - x_lo) / 4.0,
                y_lo + (rx_row + 0.5) * (y_hi - y_lo) / 4.0,
            )
            points.append(self._pair_model_features(tx, rx))
        probes = torch.tensor(points, dtype=torch.float32)
        self._cka_probe_cache[key] = probes
        return probes

    def _snapshot_model(self, variant, *, zone: int) -> ModelSnapshot:
        state = {
            name: tensor.detach().to(device="cpu").clone()
            for name, tensor in variant.model.state_dict().items()
        }
        needs_representation = (
            "representation_cka_dissimilarity" in self.observation_features
        )
        needs_prediction = (
            "normalized_prediction_disagreement" in self.observation_features
        )
        probes = (
            self._cka_probes(zone)
            if needs_representation or needs_prediction
            else None
        )
        representation_signature = (
            build_cka_signature(variant.model, probes)
            if needs_representation and probes is not None
            else None
        )
        prediction_signature = (
            build_prediction_signature(variant.model, probes)
            if needs_prediction and probes is not None
            else None
        )
        return ModelSnapshot(
            metadata=build_metadata(
                variant,
                representation_signature=representation_signature,
                prediction_signature=prediction_signature,
                share_model_age=("relative_provider_freshness" in self.observation_features),
            ),
            state=state,
        )

    def _snapshot_variant(self, variant) -> dict[str, object]:
        """Zone memory keeps weights and their associated metadata together."""

        snapshot = super()._snapshot_variant(variant)
        snapshot.update(
            {
                "m_samples": int(variant.m_samples),
                "n_samples": int(variant.n_samples),
                "quality": float(variant.quality),
                "t_wait": int(variant.t_wait),
                "last_rmse": float(variant.last_rmse),
                "last_rmse_available": bool(variant.last_rmse_available),
                "rmse_ema_short": float(variant.rmse_ema_short),
                "rmse_ema_long": float(variant.rmse_ema_long),
                "rmse_batches": int(variant.rmse_batches),
                "model_signature": variant.model_signature.detach().cpu().clone(),
                "step_saved": int(getattr(self, "_current_sumo_step", 0)),
            }
        )
        return snapshot

    def _restore_variant(self, variant, snapshot: dict[str, object]) -> None:
        if "m_samples" not in snapshot:
            super()._restore_variant(variant, snapshot)
            return
        self._load_model_state(variant.model, snapshot["weights"])  # type: ignore[arg-type]
        variant.opt = optim.Adam(variant.model.parameters(), lr=self.cfg.local_lr)
        variant.m_samples = bound_raw_samples(int(snapshot["m_samples"]))
        variant.n_samples = int(snapshot["n_samples"])
        variant.quality = float(snapshot["quality"])
        elapsed = max(
            0,
            int(getattr(self, "_current_sumo_step", 0))
            - int(snapshot.get("step_saved", 0)),
        )
        variant.t_wait = max(0, int(snapshot["t_wait"]) + elapsed)
        variant.last_rmse = float(snapshot["last_rmse"])
        variant.last_rmse_available = bool(snapshot["last_rmse_available"])
        variant.rmse_ema_short = float(snapshot["rmse_ema_short"])
        variant.rmse_ema_long = float(snapshot["rmse_ema_long"])
        variant.rmse_batches = int(snapshot["rmse_batches"])
        variant.model_signature = snapshot["model_signature"].detach().cpu().clone()  # type: ignore[union-attr]
        variant.recovery_steps_left = 0
        variant.recovery_accepts_left = 0
        variant.recovery_cooldown_left = 0

    # ------------------------------------------------------------ pull staging

    def _normalized_links(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None,
    ) -> list[tuple[int, int, int]]:
        if contact_links is None:
            return [
                (int(zone), int(a), int(b))
                for zone, indices in zone_nodes.items()
                for offset, a in enumerate(sorted(indices))
                for b in sorted(indices)[offset + 1 :]
            ]
        unique: set[tuple[int, int, int]] = set()
        for zone, a, b in contact_links:
            ia, ib = sorted((int(a), int(b)))
            if ia == ib or not (0 <= ia < len(self.nodes) and 0 <= ib < len(self.nodes)):
                continue
            if int(self.nodes[ia].current_az) != int(self.nodes[ib].current_az):
                continue
            unique.add((int(zone), ia, ib))
        return sorted(unique)

    def _aggregate_feasible_policies(
        self,
        neighbors: dict[int, list[int]],
    ) -> dict[tuple[str, int], tuple[list[int], list[int], torch.Tensor, int]]:
        """Synchronously average own and all feasible neighbor policies."""

        states = {
            mode: [agent.snapshot() for agent in agents]
            for mode, agents in self.utility_agents.items()
        }
        experiences = {
            mode: [agent.experience for agent in agents]
            for mode, agents in self.utility_agents.items()
        }
        ready = {
            mode: [agent.ready for agent in agents]
            for mode, agents in self.utility_agents.items()
        }
        versions = {
            mode: list(self._local_policy_versions[mode])
            for mode in self.utility_agents
        }
        aggregated: dict[
            tuple[str, int], tuple[list[int], list[int], torch.Tensor, int]
        ] = {}
        for mode, agents in self.utility_agents.items():
            for receiver_idx in sorted(neighbors):
                provider_ids = neighbors[receiver_idx]
                source_ids = [receiver_idx, *provider_ids]
                source_experiences = [
                    int(experiences[mode][source_idx]) for source_idx in source_ids
                ]
                state, weights = aggregate_policy_states(
                    [states[mode][source_idx] for source_idx in source_ids],
                    source_experiences,
                )
                inherited_ready = any(
                    bool(ready[mode][source_idx]) and float(weight) > 0.0
                    for source_idx, weight in zip(source_ids, weights.tolist())
                )
                agents[receiver_idx].load_shared_model(
                    state, inherited_ready=inherited_ready
                )
                self._local_policy_versions[mode][receiver_idx] = (
                    max(versions[mode][source_idx] for source_idx in source_ids) + 1
                )
                transferred = sum(
                    _state_nbytes(states[mode][provider_idx])
                    for provider_idx in provider_ids
                )
                self._utility_step_policy_bytes[mode] += int(transferred)
                self._local_policy_pull_updates[mode] += len(provider_ids)
                self._last_local_policy_pull_updates += len(provider_ids)
                self._last_local_policy_pull_updates_by_mode[mode] += len(provider_ids)
                aggregated[(mode, receiver_idx)] = (
                    source_ids,
                    source_experiences,
                    weights.detach().cpu().clone(),
                    int(transferred),
                )
        return aggregated

    def _steps_since_predictor_pull(
        self, mode: str, receiver_idx: int, provider_idx: int, zone: int, step: int
    ) -> int | None:
        key = (str(mode), int(receiver_idx), int(provider_idx), int(zone))
        last_step = self._last_predictor_pull_step.get(key)
        return None if last_step is None else max(0, int(step) - int(last_step))

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Snapshot globally, batch-score each receiver, and stage downloads."""

        if self._staged_pull_plans:
            raise RuntimeError("unfinalized pull plans leaked into the next step")
        self._utility_current_samples.clear()
        links = self._normalized_links(zone_nodes, contact_links)
        neighbors: dict[int, list[int]] = defaultdict(list)
        encounter_ids: dict[tuple[int, int], int] = {}
        for _zone, a, b in links:
            neighbors[a].append(b)
            neighbors[b].append(a)
            encounter_ids[(a, b)] = int(self._next_enc_id)
            self._next_enc_id += 1
        for receiver_idx in neighbors:
            neighbors[receiver_idx] = sorted(set(neighbors[receiver_idx]))
        policy_aggregations = self._aggregate_feasible_policies(neighbors)

        snapshots: dict[str, dict[int, ModelSnapshot]] = {}
        for mode in self.agents:
            snapshots[mode] = {
                node_idx: self._snapshot_model(
                    ns.variants[mode], zone=int(ns.current_az)
                )
                for node_idx, ns in enumerate(self.nodes)
            }

        capacities = {
            receiver_idx: sample_pull_capacity(
                self.pull_budget,
                len(provider_ids),
                rng=self._capacity_rngs[receiver_idx],
            )
            for receiver_idx, provider_ids in neighbors.items()
        }

        for mode, mode_agents in self.utility_agents.items():
            for receiver_idx in sorted(neighbors):
                provider_ids = neighbors[receiver_idx]
                (
                    policy_source_ids,
                    policy_experiences,
                    policy_weights,
                    policy_bytes,
                ) = policy_aggregations[(mode, receiver_idx)]
                zone = int(self.nodes[receiver_idx].current_az)
                receiver_snapshot = snapshots[mode][receiver_idx]
                observations = torch.stack(
                    [
                        build_observation(
                            receiver_snapshot.metadata,
                            snapshots[mode][provider_idx].metadata,
                            neighbor_count=len(provider_ids),
                            zone_neighbor_count=max(
                                0,
                                len(
                                    zone_nodes[
                                        int(self.nodes[receiver_idx].current_az)
                                    ]
                                )
                                - 1,
                            ),
                            zone_buffer_samples=len(
                                self.nodes[receiver_idx].current_visit_samples_x
                            ),
                            feature_names=self.observation_features,
                            steps_since_provider_pull=self._steps_since_predictor_pull(
                                mode,
                                receiver_idx,
                                provider_idx,
                                zone,
                                step,
                            ),
                        )
                        for provider_idx in provider_ids
                    ],
                    dim=0,
                )
                agent = mode_agents[receiver_idx]
                scores = agent.score(observations)
                capacity = capacities[receiver_idx]
                exploratory = bool(
                    not self.utility_evaluation
                    and (
                        not agent.ready
                        or agent.rng.random() < self.utility_exploration_prob
                    )
                )
                selected_ids = (
                    select_random_subset(provider_ids, capacity, rng=agent.rng)
                    if exploratory
                    else select_top_k(scores, provider_ids, capacity)
                )
                selected_set = set(selected_ids)
                for provider_idx in selected_ids:
                    self._last_predictor_pull_step[
                        (str(mode), int(receiver_idx), int(provider_idx), zone)
                    ] = int(step)

                candidates = [
                    ProviderCandidate(
                        provider_idx=int(provider_idx),
                        observation=observation.detach().cpu().clone(),
                        snapshot=snapshots[mode][provider_idx],
                        score=float(score),
                    )
                    for provider_idx, observation, score in zip(
                        provider_ids, observations, scores.tolist()
                    )
                ]
                plan = ReceiverPullPlan(
                    step=int(step),
                    zone=int(self.nodes[receiver_idx].current_az),
                    receiver_idx=int(receiver_idx),
                    mode=str(mode),
                    receiver_snapshot=receiver_snapshot,
                    candidates=candidates,
                    capacity=int(capacity),
                    selected_provider_ids=list(selected_ids),
                    exploratory=exploratory,
                    policy_source_ids=policy_source_ids,
                    policy_experiences=policy_experiences,
                    policy_weights=policy_weights,
                    policy_bytes=policy_bytes,
                )
                self._staged_pull_plans[(mode, receiver_idx)] = plan

                metadata_bytes = sum(
                    snapshots[mode][provider_idx].metadata.wire_nbytes
                    for provider_idx in provider_ids
                ) + len(provider_ids) * POLICY_EXPERIENCE_BYTES
                model_bytes = sum(
                    _state_nbytes(snapshots[mode][provider_idx].state)
                    for provider_idx in selected_ids
                )
                self._utility_step_metadata_bytes[mode] += int(metadata_bytes)
                self._utility_step_model_bytes[mode] += int(model_bytes)
                self._utility_step_selected[mode] += len(selected_ids)

                score_by_id = {
                    candidate.provider_idx: candidate.score for candidate in candidates
                }
                for provider_idx in provider_ids:
                    pair = tuple(sorted((receiver_idx, provider_idx)))
                    provider_ns = self.nodes[provider_idx]
                    receiver_ns = self.nodes[receiver_idx]
                    self._record_decision_row(
                        {
                            "step": int(step),
                            "enc_id": encounter_ids[pair],
                            "node_i": int(receiver_idx),
                            "node_j": int(provider_idx),
                            "az": int(receiver_ns.current_az),
                            "dist": float(
                                np.hypot(
                                    receiver_ns.node.x - provider_ns.node.x,
                                    receiver_ns.node.y - provider_ns.node.y,
                                )
                            ),
                            "mode": str(mode),
                            "action": int(provider_idx in selected_set),
                            "merge_weight": "",
                            "predicted_gain": float(score_by_id[provider_idx]),
                            "gain_threshold": 0.0,
                            "exploratory": int(exploratory),
                            "reward": float("nan"),
                            "deferred": int(provider_idx in selected_set),
                        }
                    )

    # ------------------------------------------ experience-only aggregation

    def _aggregate_candidates(
        self,
        snapshots: list[ModelSnapshot],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = experience_weights(
            [snapshot.metadata.experience for snapshot in snapshots],
            epsilon=self.aggregation_experience_epsilon,
        )
        state = weighted_average([snapshot.state for snapshot in snapshots], weights)
        return weights, state

    def _finalize_plan(
        self,
        plan: ReceiverPullPlan,
        X: np.ndarray,
        y_dbm: np.ndarray,
        local_X: np.ndarray | None = None,
        local_y_dbm: np.ndarray | None = None,
    ) -> dict[str, object]:
        ns = self.nodes[plan.receiver_idx]
        fine_tune_X = (
            np.asarray(X, dtype=np.float32)
            if local_X is None
            else np.asarray(local_X, dtype=np.float32)
        )
        fine_tune_y = y_dbm if local_y_dbm is None else local_y_dbm
        variant = ns.variants[plan.mode]
        selected_map = {
            candidate.provider_idx: candidate
            for candidate in plan.candidates
            if candidate.provider_idx in set(plan.selected_provider_ids)
        }
        selected = [
            selected_map[provider_idx] for provider_idx in plan.selected_provider_ids
        ]
        snapshots = [plan.receiver_snapshot] + [
            candidate.snapshot for candidate in selected
        ]
        aggregation_weights, aggregated_state = self._aggregate_candidates(
            snapshots
        )

        before_rmse: float | str = ""
        aggregated_rmse: float | str = ""
        if int(X.shape[0]) > 0:
            before_rmse = self.eval_rmse_with_weights(
                plan.mode, plan.receiver_snapshot.state, X, y_dbm
            )
            aggregated_rmse = self.eval_rmse_with_weights(
                plan.mode, aggregated_state, X, y_dbm
            )

        if selected:
            for candidate in selected:
                _, pairwise_state = self._aggregate_candidates(
                    [plan.receiver_snapshot, candidate.snapshot]
                )
                pending = PendingPull(
                    observation=candidate.observation.detach().cpu().clone(),
                    receiver_snapshot=plan.receiver_snapshot,
                    provider_snapshot=candidate.snapshot,
                    reference_pairwise_state=pairwise_state,
                    receiver_idx=plan.receiver_idx,
                    provider_idx=candidate.provider_idx,
                    mode=plan.mode,
                    zone=plan.zone,
                    timestep=plan.step,
                    horizon=self.utility_horizon,
                    initial_samples_x=fine_tune_X.tolist(),
                    initial_samples_y=np.asarray(
                        fine_tune_y, dtype=np.float32
                    ).reshape(-1).tolist(),
                )
                queue = self._utility_pending[(plan.mode, plan.receiver_idx)]
                queue.append(pending)
                cap = max(1, int(self.cfg.pending_slot_cap))
                if len(queue) > cap:
                    del queue[: len(queue) - cap]

            self._load_model_state(variant.model, aggregated_state)
            represented_raw = sum(
                float(weight) * float(snapshot.metadata.raw_experience)
                for weight, snapshot in zip(aggregation_weights.tolist(), snapshots)
            )
            variant.m_samples = bound_raw_samples(int(round(represented_raw)))
            variant.n_samples = saturate_n_samples(variant.m_samples)
            variant.opt = optim.Adam(
                variant.model.parameters(), lr=self.cfg.local_lr
            )
            variant.t_wait = 0
            variant.last_rmse_available = False
            self._refresh_variant_signature(variant)

        labels = [f"self:{plan.receiver_idx}"] + [
            str(candidate.provider_idx) for candidate in selected
        ]
        policy_labels = [f"self:{plan.receiver_idx}"] + [
            str(source_idx) for source_idx in plan.policy_source_ids[1:]
        ]
        return {
            "step": int(plan.step),
            "mode": str(plan.mode),
            "receiver_idx": int(plan.receiver_idx),
            "zone": int(plan.zone),
            "neighbor_count": int(len(plan.candidates)),
            "capacity": int(plan.capacity),
            "exploratory": int(plan.exploratory),
            "selected_provider_ids": _format_ids(plan.selected_provider_ids),
            "policy_source_ids": _format_ids(plan.policy_source_ids),
            "policy_experiences": _format_weights(
                policy_labels, torch.tensor(plan.policy_experiences)
            ),
            "policy_weights": _format_weights(
                policy_labels, plan.policy_weights
            ),
            "policy_bytes": int(plan.policy_bytes),
            "aggregation_weights": _format_weights(labels, aggregation_weights),
            "metadata_bytes": int(
                sum(
                    candidate.snapshot.metadata.wire_nbytes
                    for candidate in plan.candidates
                )
            ) + len(plan.candidates) * POLICY_EXPERIENCE_BYTES,
            "model_bytes": int(
                sum(_state_nbytes(candidate.snapshot.state) for candidate in selected)
            ),
            "current_samples": int(X.shape[0]),
            "current_rmse_before": before_rmse,
            "current_rmse_aggregated": aggregated_rmse,
            "current_rmse_after_local": "",
        }

    def _train_local(
        self,
        ns,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        sample_count_increment: int | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Aggregate by experience, then train on the unweighted zone buffer."""

        del sample_weights
        n_new = (
            int(X.shape[0])
            if sample_count_increment is None
            else max(0, int(sample_count_increment))
        )
        X_step = np.asarray(X[-n_new:], dtype=np.float32) if n_new else np.empty(
            (0, int(X.shape[1])), dtype=np.float32
        )
        y_step = np.asarray(y_dbm[-n_new:], dtype=np.float32) if n_new else np.empty(
            (0, 1), dtype=np.float32
        )
        node_idx = self.node_idx(ns)
        self._utility_current_samples[node_idx] = (
            int(ns.current_az),
            X_step.copy(),
            y_step.copy(),
        )

        rows: list[tuple[str, dict[str, object]]] = []
        for mode in self.agents:
            plan = self._staged_pull_plans.pop((mode, node_idx), None)
            if plan is not None:
                rows.append(
                    (
                        mode,
                        self._finalize_plan(
                            plan,
                            X_step,
                            y_step,
                            local_X=X,
                            local_y_dbm=y_dbm,
                        ),

                    )
                )
        if int(X.shape[0]) > 0:
            super()._train_local(
                ns,
                X,
                y_dbm,
                sample_count_increment=n_new,
                sample_weights=None,
            )
        for mode, row in rows:
            if n_new > 0:
                row["current_rmse_after_local"] = float(
                    self.eval_rmse(ns, mode, X_step, y_step)
                )
            self._write_utility_row("aggregation", row)

    def _sample_recency_weights(self, *args, **kwargs) -> None:
        """The new method never applies age weights to a sample set."""

        del args, kwargs
        return None

    def _train_one_local(
        self,
        model: torch.nn.Module,
        opt: optim.Optimizer,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        seed_key: str = "aux",
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Train auxiliary baselines on the full unweighted zone-visit buffer."""

        del sample_weights
        super()._train_one_local(
            model,
            opt,
            X,
            y_dbm,
            seed_key=seed_key,
            sample_weights=None,
        )

    # ---------------------------------------------------------- delayed reward

    def _utility_reward_from_gain(self, raw_gain: float) -> float:
        """Map a measured RMSE gain to the policy's regression target."""

        return float(raw_gain)

    def _utility_sample_batches(
        self, pull: PendingPull
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not pull.samples_x:
            return []
        steps = (
            list(pull.sample_steps)
            if len(pull.sample_steps) == len(pull.samples_x)
            else [int(pull.maturity_step)] * len(pull.samples_x)
        )
        grouped: list[tuple[torch.Tensor, torch.Tensor]] = []
        start = 0
        while start < len(steps):
            end = start + 1
            while end < len(steps) and steps[end] == steps[start]:
                end += 1
            X = np.asarray(pull.samples_x[start:end], dtype=np.float32)
            y = np.asarray(pull.samples_y[start:end], dtype=np.float32)
            grouped.append(
                (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(
                        self._normalize_target_from_rssi(y), dtype=torch.float32
                    ).reshape(-1),
                )
            )
            start = end
        return grouped

    def _evaluate_utility_pull(
        self, pull: PendingPull
    ) -> dict[str, int | float | str] | None:
        """Fine-tune equally, freeze, and evaluate the common local-plus-future set."""

        if not pull.samples_x:
            return None
        future_X = np.asarray(pull.samples_x, dtype=np.float32)
        future_y = np.asarray(pull.samples_y, dtype=np.float32).reshape(-1, 1)
        future_batches = self._utility_sample_batches(pull)
        future_steps = len(future_batches)
        local_samples = len(pull.initial_samples_x)
        fine_tune_epochs = 0
        if self.utility_feedback_mode == "frozen":
            baseline_rmse = self.eval_rmse_with_weights(
                pull.mode, pull.receiver_snapshot.state, future_X, future_y
            )
            pairwise_rmse = self.eval_rmse_with_weights(
                pull.mode, pull.reference_pairwise_state, future_X, future_y
            )
            n_samples = int(future_X.shape[0])
        else:
            local_batch = None
            if pull.initial_samples_x:
                local_X = np.asarray(pull.initial_samples_x, dtype=np.float32)
                local_y = np.asarray(pull.initial_samples_y, dtype=np.float32)
                local_batch = (
                    torch.tensor(local_X, dtype=torch.float32),
                    torch.tensor(
                        self._normalize_target_from_rssi(local_y),
                        dtype=torch.float32,
                    ).reshape(-1),
                )
            evaluation_batches = (
                ([] if local_batch is None else [local_batch]) + future_batches
            )
            result = evaluate_finetuned_pair(
                model_factory=self._make_predictor,
                baseline_state=pull.receiver_snapshot.state,
                merged_state=pull.reference_pairwise_state,
                local_batch=local_batch,
                evaluation_batches=evaluation_batches,
                device=self.device,
                lr=float(self.cfg.local_lr),
                epochs=int(self.cfg.local_epochs),
                batch_size=int(self.cfg.local_batch_size),
                metric_scale=self._loss_max_db() - self._loss_min_db(),
                random_seed=rre_sim._stable_torch_seed(
                    self.cfg.seed,
                    "utility-finetune-window",
                    pull.mode,
                    pull.receiver_idx,
                    pull.provider_idx,
                    pull.zone,
                    pull.timestep,
                ),
            )
            baseline_rmse = result.baseline_rmse
            pairwise_rmse = result.merged_rmse
            n_samples = result.num_samples
            fine_tune_epochs = int(self.cfg.local_epochs)
        reward = self._utility_reward_from_gain(
            float(baseline_rmse - pairwise_rmse)
        )
        return {
            "feedback_mode": self.utility_feedback_mode,
            "n_samples": int(n_samples),
            "future_steps": int(future_steps),
            "local_samples": int(local_samples),
            "future_samples": int(future_X.shape[0]),
            "fine_tune_epochs": int(fine_tune_epochs),
            "baseline_rmse": float(baseline_rmse),
            "pairwise_rmse": float(pairwise_rmse),
            "realized_utility": float(reward),
        }

    def _finalize_utility_pull(
        self,
        pull: PendingPull,
        *,
        step: int,
        terminal_reason: str,
        baseline_rmse: float | None = None,
    ) -> bool:
        """Evaluate and enqueue one delayed pull using its collected samples."""

        del baseline_rmse
        result = self._evaluate_utility_pull(pull)
        if result is None:
            return False
        reward = float(result["realized_utility"])
        self.utility_agents[pull.mode][pull.receiver_idx].push(
            pull.observation, reward
        )
        self._utility_new_rewards[(pull.mode, pull.receiver_idx)] += 1
        self._last_local_policy_queued_transitions += 1
        values = pull.observation.detach().cpu().reshape(-1).tolist()
        self._write_utility_row(
            "feedback",
            {
                "mode": pull.mode,
                "receiver_idx": pull.receiver_idx,
                "provider_idx": pull.provider_idx,
                "zone": pull.zone,
                "step_started": pull.timestep,
                "step_finalized": int(step),
                "terminal_reason": str(terminal_reason),
                **result,
                **{
                    name: float(value)
                    for name, value in zip(self.observation_features, values)
                },
            },
        )
        return True

    def _finalize_pending_on_zone_exit(
        self,
        *,
        receiver_idx: int,
        old_zone: int,
        step: int,
    ) -> None:
        """Treat leaving a zone as a terminal utility window."""

        for mode in self.utility_agents:
            key = (mode, int(receiver_idx))
            pending = self._utility_pending.get(key)
            if not pending:
                continue
            remaining: list[PendingPull] = []
            for pull in pending:
                if int(pull.zone) != int(old_zone):
                    remaining.append(pull)
                    continue
                if self._finalize_utility_pull(
                    pull,
                    step=int(step),
                    terminal_reason="zone_exit",
                ):
                    self._utility_exit_dirty.add(key)
            if remaining:
                self._utility_pending[key] = remaining
            else:
                self._utility_pending.pop(key, None)

    def _reset_node_for_zone_change(self, ns, new_az: int) -> None:
        """Finalize old-zone utility windows before visit state is reset."""

        old_zone = int(ns.current_az)
        if int(new_az) != old_zone:
            receiver_idx = self.node_idx(ns)
            if receiver_idx >= 0:
                self._finalize_pending_on_zone_exit(
                    receiver_idx=receiver_idx,
                    old_zone=old_zone,
                    step=int(getattr(self, "_current_sumo_step", 0)),
                )
        super()._reset_node_for_zone_change(ns, new_az)

    def _process_delayed_feedback(self, step: int) -> set[tuple[str, int]]:
        dirty = set(self._utility_exit_dirty)
        self._utility_exit_dirty.clear()
        for key in list(self._utility_pending):
            mode, receiver_idx = key
            zone, X, y = self._utility_current_samples.get(
                receiver_idx,
                (
                    int(self.nodes[receiver_idx].current_az),
                    np.empty((0, self._predictor_input_dim()), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32),
                ),
            )
            remaining: list[PendingPull] = []
            for pull in self._utility_pending[key]:
                matured = advance_pending_pull(
                    pull,
                    step=int(step),
                    receiver_zone=int(zone),
                    samples_x=X.tolist(),
                    samples_y=y.reshape(-1).tolist(),
                )
                if not matured:
                    remaining.append(pull)
                    continue
                if self._finalize_utility_pull(
                    pull,
                    step=int(step),
                    terminal_reason="horizon",
                ):
                    dirty.add((mode, receiver_idx))
            if remaining:
                self._utility_pending[key] = remaining
            else:
                self._utility_pending.pop(key, None)
        return dirty

    def _train_rl_agents(self, step: int | None = None) -> dict[str, float]:
        current_step = int(
            getattr(self, "_current_sumo_step", 0) if step is None else step
        )
        # Plans without any receiver samples still use experience initialization.
        for key, plan in list(self._staged_pull_plans.items()):
            empty_X = np.empty((0, self._predictor_input_dim()), dtype=np.float32)
            empty_y = np.empty((0, 1), dtype=np.float32)
            row = self._finalize_plan(plan, empty_X, empty_y)
            self._write_utility_row("aggregation", row)
            self._staged_pull_plans.pop(key, None)

        dirty = self._process_delayed_feedback(current_step)
        losses = {mode: 0.0 for mode in self.agents}
        for mode, receiver_idx in sorted(dirty):
            agent = self.utility_agents[mode][receiver_idx]
            loss = agent.train(self.utility_train_updates)
            if loss is not None:
                losses[mode] = float(loss)
                self._local_policy_train_updates[mode] += self.utility_train_updates
                self._last_local_policy_train_updates_this_step += self.utility_train_updates
                self._local_policy_versions[mode][receiver_idx] += self.utility_train_updates
            self._write_utility_row(
                "training",
                {
                    "step": current_step,
                    "mode": mode,
                    "receiver_idx": receiver_idx,
                    "new_rewards": int(
                        self._utility_new_rewards[(mode, receiver_idx)]
                    ),
                    "replay_size": len(agent.replay),
                    "train_steps": int(agent.train_steps),
                    "loss": "" if loss is None else float(loss),
                },
            )
        self._utility_new_rewards.clear()
        self._utility_current_samples.clear()
        return losses

    def _local_pending_transition_count(self) -> int:
        return int(sum(len(queue) for queue in self._utility_pending.values()))

    # ---------------------------------------------------------- communication

    def _reset_policy_step_counters(self) -> None:
        super()._reset_policy_step_counters()
        self._utility_step_metadata_bytes.clear()
        self._utility_step_model_bytes.clear()
        self._utility_step_policy_bytes.clear()
        self._utility_step_selected.clear()

    def _build_communication_assumptions(self) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        signature_size = int(rre_sim.MODEL_SIGNATURE_FLOATS) * 4
        cka_size = (
            int(CKA_SIGNATURE_FLOATS) * 4
            if "representation_cka_dissimilarity" in self.observation_features
            else 0
        )
        prediction_size = (
            int(CKA_PROBE_COUNT) * 4
            if "normalized_prediction_disagreement" in self.observation_features
            else 0
        )
        provider_age_size = (
            4 if "relative_provider_freshness" in self.observation_features else 0
        )
        metadata_size = (
            2 * 4
            + 1
            + signature_size
            + cka_size
            + prediction_size
            + provider_age_size
            + POLICY_EXPERIENCE_BYTES
        )
        utility_params = 0
        utility_bytes = 0
        if self.utility_agents:
            model = next(iter(self.utility_agents.values()))[0].model
            utility_params = sum(int(p.numel()) for p in model.parameters())
            utility_bytes = sum(
                int(p.numel()) * int(p.element_size()) for p in model.parameters()
            )
        assumptions.update(
            {
                "policy_feature_set": self.feature_set,
                "policy_observation_dim": len(self.observation_features),
                "policy_observation_features": ",".join(self.observation_features),
                "pull_budget_expected_slots": float(self.pull_budget),
                "utility_exploration_probability": float(
                    self.utility_exploration_prob
                ),
                "utility_evaluation": bool(self.utility_evaluation),
                "utility_horizon_steps": int(self.utility_horizon),
                "utility_feedback_mode": self.utility_feedback_mode,
                "utility_fine_tune_epochs": int(self.cfg.local_epochs),
                "pending_slot_capacity": int(self.cfg.pending_slot_cap),
                "zone_exit_reward_finalization": True,
                "predictor_aggregation_weighting": "normalized-effective-experience",
                "aggregation_experience_epsilon": float(
                    self.aggregation_experience_epsilon
                ),
                "B_decision_meta_bytes_per_directed_decision": metadata_size,
                "B_decision_scalar_meta_bytes_per_directed_decision": 9 + POLICY_EXPERIENCE_BYTES,
                "B_model_signature_bytes_per_directed_decision": signature_size,
                "B_cka_signature_bytes_per_directed_decision": cka_size,
                "B_prediction_signature_bytes_per_directed_decision": prediction_size,
                "B_provider_model_age_bytes_per_directed_decision": provider_age_size,
                "B_policy_experience_bytes_per_directed_decision": POLICY_EXPERIENCE_BYTES,
                "cka_probe_count": CKA_PROBE_COUNT,
                "cka_signature": "linear-CKA-centered-penultimate-gram-upper-triangle",
                "B_local_policy_pull_bytes": int(utility_bytes),
                "B_policy_bytes": int(utility_bytes),
                "utility_model_params": int(utility_params),
                "utility_model_bytes": int(utility_bytes),
                "local_policy_share": True,
                "utility_model_architecture": "micro-single-hidden-layer",
                "zramp_policy_mode": "utility-top-k",
                "aggregation_sample_scope": "none-metadata-only",
                "local_training_sample_scope": "current-zone-visit-buffer",
                "policy_transfer_rule": self.policy_transfer_rule,
                "policy_experience": "matured local pull decisions retained in replay",
                "aux_local_training_sample_scope": "current-zone-visit-buffer",
                "local_sample_weighting": "uniform",
                "policy_pull_rule": "top-k-strictly-positive-utility-with-random-exploration",
                "policy_reward_target": (
                    "equal-local-finetune-then-frozen-local-plus-future-rmse-improvement"
                ),
                "metadata_note": (
                    "Every feasible directed provider sends fixed predictor metadata, "
                    "its local policy experience, and its utility-policy weights. Own "
                    "and received policies are averaged synchronously by experience; "
                    "only selected providers send predictor weights. Replay stays local."
                ),
            }
        )
        return assumptions

    def _communication_overhead_row(
        self,
        *,
        feasible_decisions: int,
        greedy_events: int,
        rl_events: Counter,
    ) -> dict[str, int | float]:
        del feasible_decisions
        assumptions = self._communication_assumptions
        model_bytes = int(assumptions.get("B_model_bytes", 0))
        merge_metadata = int(
            assumptions.get("B_accepted_merge_meta_bytes_per_pull", 0)
        )
        greedy_bytes = int(greedy_events) * (model_bytes + merge_metadata)
        self._comm_cumulative_bytes["greedy"] += greedy_bytes
        row: dict[str, int | float] = {
            "greedy_comm_bytes": greedy_bytes,
            "greedy_comm_mb": float(greedy_bytes) / 1_000_000.0,
            "greedy_comm_cumulative_mb": float(
                self._comm_cumulative_bytes["greedy"]
            )
            / 1_000_000.0,
            "local_policy_initial_pull_probability": 1.0,
        }
        for mode in self.agents:
            selected = int(rl_events.get(mode, self._utility_step_selected[mode]))
            metadata = int(self._utility_step_metadata_bytes[mode])
            transferred_models = int(self._utility_step_model_bytes[mode])
            transferred_policies = int(self._utility_step_policy_bytes[mode])
            merge_bytes = selected * merge_metadata
            total = metadata + transferred_models + transferred_policies + merge_bytes
            self._comm_cumulative_bytes[mode] += total
            greedy_cumulative = float(self._comm_cumulative_bytes["greedy"])
            row.update(
                {
                    f"{mode}_metadata_bytes": metadata,
                    f"{mode}_model_bytes": transferred_models,
                    f"{mode}_merge_metadata_bytes": merge_bytes,
                    f"{mode}_policy_model_bytes": transferred_policies,
                    f"{mode}_selected_providers": selected,
                    f"{mode}_comm_bytes": total,
                    f"{mode}_comm_mb": float(total) / 1_000_000.0,
                    f"{mode}_comm_cumulative_mb": float(
                        self._comm_cumulative_bytes[mode]
                    )
                    / 1_000_000.0,
                    f"{mode}_comm_vs_greedy_ratio": (
                        float(total) / float(greedy_bytes)
                        if greedy_bytes > 0
                        else float("nan")
                    ),
                    f"{mode}_comm_cumulative_vs_greedy_ratio": (
                        float(self._comm_cumulative_bytes[mode]) / greedy_cumulative
                        if greedy_cumulative > 0.0
                        else float("nan")
                    ),
                }
            )
        return row

    # ---------------------------------------------------------------- logging

    def _write_utility_row(self, kind: str, row: dict[str, object]) -> None:
        if kind == "aggregation":
            fields = AGGREGATION_LOG_FIELDS
            path = self._aggregation_log_path
        elif kind == "feedback":
            fields = FEEDBACK_LOG_FIELDS
            path = self._feedback_log_path
        elif kind == "training":
            fields = UTILITY_TRAINING_LOG_FIELDS
            path = self._utility_training_log_path
        else:  # pragma: no cover - internal programming error.
            raise ValueError(f"unknown utility log kind {kind!r}")
        writer = self._utility_csv_writers.get(kind)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_obj = open(path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(file_obj, fieldnames=fields)
            writer.writeheader()
            self._utility_csv_files[kind] = file_obj
            self._utility_csv_writers[kind] = writer
        writer.writerow({field: row.get(field, "") for field in fields})
        self._utility_csv_counts[kind] += 1
        if self._utility_csv_counts[kind] % 10000 == 0:
            self._utility_csv_files[kind].flush()  # type: ignore[union-attr]

    def _flush_utility_logs(self) -> None:
        for file_obj in self._utility_csv_files.values():
            file_obj.flush()  # type: ignore[union-attr]

    def _close_utility_logs(self) -> None:
        for file_obj in self._utility_csv_files.values():
            file_obj.flush()  # type: ignore[union-attr]
            file_obj.close()  # type: ignore[union-attr]
        self._utility_csv_files.clear()
        self._utility_csv_writers.clear()

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._flush_utility_logs()

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._close_utility_logs()
