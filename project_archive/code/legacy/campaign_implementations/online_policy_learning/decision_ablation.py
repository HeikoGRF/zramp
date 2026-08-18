"""Sequential decision-policy variants for the controlled four-zone ablation.

The current joint-weight optimizer remains in :mod:`utility_selection`.
This module isolates the old softmax decision and the two proposed bounded
score variants. All three share policies synchronously at every feasible
contact and merge selected predictors sequentially by experience.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim

import rl_reward_experiment.sim as rre_sim
from rl_reward_experiment.node_state import bound_raw_samples, saturate_n_samples

from .budget import select_random_subset
from .feedback import ModelSnapshot, PendingPull
from .metadata import build_observation
from .simulation import BootstrapPolicySharingSimulation
from .utility_selection import (
    POLICY_EXPERIENCE_BYTES,
    UtilitySelectionBootstrapSimulation,
    _state_nbytes,
)


DECISION_RULES = ("old-softmax", "unsigned-top-b", "signed-top-b")


def bounded_policy_scores(raw_scores: torch.Tensor, rule: str) -> torch.Tensor:
    """Map raw RMSE-gain predictions to the requested decision range."""

    if rule in {"old-softmax", "unsigned-top-b"}:
        return torch.sigmoid(raw_scores)
    if rule == "signed-top-b":
        return torch.tanh(raw_scores)
    raise ValueError(f"unknown decision rule {rule!r}")


def select_ablation_providers(
    raw_scores: torch.Tensor,
    provider_ids: Sequence[int],
    *,
    rule: str,
    budget: int,
    ready: bool,
    evaluation: bool,
    exploration_probability: float,
    rng,
) -> tuple[list[int], bool]:
    """Pure selection helper used by the simulation and focused tests."""

    ids = [int(value) for value in provider_ids]
    if len(ids) != int(raw_scores.numel()):
        raise ValueError("provider_ids and raw_scores must have the same length")
    if rule not in DECISION_RULES:
        raise ValueError(f"unknown decision rule {rule!r}")
    if not ids:
        return [], False

    if rule == "old-softmax":
        # Equivalent to softmax([Q_reject=0, Q_pull=raw_score]).
        probabilities = torch.sigmoid(raw_scores).tolist()
        return (
            [
                provider_id
                for provider_id, probability in zip(ids, probabilities)
                if rng.random() < float(probability)
            ],
            bool(not evaluation and not ready),
        )

    capacity = max(0, min(int(budget), len(ids)))
    exploratory = bool(
        not evaluation
        and (not ready or rng.random() < float(exploration_probability))
    )
    if exploratory:
        return select_random_subset(ids, capacity, rng=rng), True
    ranked = sorted(
        zip(ids, raw_scores.tolist()), key=lambda item: (-float(item[1]), item[0])
    )
    if rule == "signed-top-b":
        ranked = [item for item in ranked if float(item[1]) > 0.0]
    return [provider_id for provider_id, _ in ranked[:capacity]], False


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }


class SequentialDecisionAblationSimulation(UtilitySelectionBootstrapSimulation):
    """Old-style sequential pulls with the requested scalar policy rules."""

    def __init__(
        self,
        *args,
        decision_rule: str,
        communication_penalty: float = 0.0,
        compact_decision_logging: bool = True,
        **kwargs,
    ) -> None:
        if decision_rule not in DECISION_RULES:
            raise ValueError(f"decision_rule must be one of {DECISION_RULES}")
        penalty = float(communication_penalty)
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("communication_penalty must be finite and non-negative")
        if decision_rule != "old-softmax" and penalty:
            raise ValueError("communication_penalty is only valid for old-softmax")
        self.decision_rule = str(decision_rule)
        self.communication_penalty = penalty
        self.compact_decision_logging = bool(compact_decision_logging)
        super().__init__(*args, **kwargs)
        self.zramp_policy_mode = self.decision_rule


    def _record_decision_row(self, row: dict) -> None:
        """Keep action totals in memory but suppress the per-contact CSV."""

        if not self.compact_decision_logging:
            super()._record_decision_row(row)
            return
        normalized = dict(row)
        self.decision_log.append(normalized)
        try:
            self._decision_action_counts[str(normalized["mode"])][
                int(normalized["action"])
            ] += 1
        except Exception:
            pass

    def _apply_sequential_pull(
        self,
        *,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        zone: int,
        step: int,
        observation: torch.Tensor,
        provider_snapshot: ModelSnapshot,
    ) -> None:
        variant = self.nodes[receiver_idx].variants[mode]
        pre_snapshot = ModelSnapshot(
            metadata=self._snapshot_model(variant, zone=zone).metadata,
            state=_clone_state(variant.model),
        )
        receiver_raw = int(variant.m_samples)
        rre_sim.weighted_state_pull(
            variant.model,
            float(variant.n_samples),
            provider_snapshot.state,
            float(provider_snapshot.metadata.experience),
            merge_strategy="average",
        )
        post_state = _clone_state(variant.model)
        variant.m_samples = bound_raw_samples(
            receiver_raw + int(provider_snapshot.metadata.raw_experience)
        )
        variant.n_samples = saturate_n_samples(variant.m_samples)
        variant.opt = optim.Adam(variant.model.parameters(), lr=self.cfg.local_lr)
        variant.t_wait = 0
        variant.last_rmse_available = False
        self._refresh_variant_signature(variant)

        queue = self._utility_pending[(mode, receiver_idx)]
        queue.append(
            PendingPull(
                observation=observation.detach().cpu().clone(),
                receiver_snapshot=pre_snapshot,
                provider_snapshot=provider_snapshot,
                reference_pairwise_state=post_state,
                receiver_idx=receiver_idx,
                provider_idx=provider_idx,
                mode=mode,
                zone=zone,
                timestep=step,
                horizon=self.utility_horizon,
            )
        )
        cap = max(1, int(self.cfg.pending_slot_cap))
        if len(queue) > cap:
            del queue[: len(queue) - cap]

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Snapshot and batch-score first, then apply ranked pulls sequentially."""

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
        self._aggregate_feasible_policies(neighbors)

        snapshots = {
            mode: {
                node_idx: self._snapshot_model(
                    ns.variants[mode], zone=int(ns.current_az)
                )
                for node_idx, ns in enumerate(self.nodes)
            }
            for mode in self.agents
        }
        for mode, agents in self.utility_agents.items():
            for receiver_idx in sorted(neighbors):
                provider_ids = neighbors[receiver_idx]
                zone = int(self.nodes[receiver_idx].current_az)
                receiver_snapshot = snapshots[mode][receiver_idx]
                observations = torch.stack(
                    [
                        build_observation(
                            receiver_snapshot.metadata,
                            snapshots[mode][provider_idx].metadata,
                            neighbor_count=len(provider_ids),
                            zone_neighbor_count=max(0, len(zone_nodes[zone]) - 1),
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
                    ]
                )
                agent = agents[receiver_idx]
                raw_scores = agent.score(observations)
                selected_ids, exploratory = select_ablation_providers(
                    raw_scores,
                    provider_ids,
                    rule=self.decision_rule,
                    budget=int(self.pull_budget),
                    ready=agent.ready,
                    evaluation=self.utility_evaluation,
                    exploration_probability=self.utility_exploration_prob,
                    rng=agent.rng,
                )
                selected_set = set(selected_ids)
                observations_by_id = dict(zip(provider_ids, observations))
                raw_by_id = dict(zip(provider_ids, raw_scores.tolist()))
                bounded_by_id = dict(
                    zip(
                        provider_ids,
                        bounded_policy_scores(
                            raw_scores, self.decision_rule
                        ).tolist(),
                    )
                )
                for provider_idx in selected_ids:
                    self._apply_sequential_pull(
                        mode=mode,
                        receiver_idx=receiver_idx,
                        provider_idx=provider_idx,
                        zone=zone,
                        step=int(step),
                        observation=observations_by_id[provider_idx],
                        provider_snapshot=snapshots[mode][provider_idx],
                    )
                    self._last_predictor_pull_step[
                        (mode, receiver_idx, provider_idx, zone)
                    ] = int(step)

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

                # One compact row per receiver/step keeps selected IDs and bytes.
                self._write_utility_row(
                    "aggregation",
                    {
                        "step": int(step),
                        "mode": mode,
                        "receiver_idx": receiver_idx,
                        "zone": zone,
                        "neighbor_count": len(provider_ids),
                        "capacity": (
                            len(provider_ids)
                            if self.decision_rule == "old-softmax"
                            else int(self.pull_budget)
                        ),
                        "exploratory": int(exploratory),
                        "selected_provider_ids": ";".join(map(str, selected_ids)),
                        "metadata_bytes": int(metadata_bytes),
                        "model_bytes": int(model_bytes),
                        "current_samples": 0,
                    },
                )
                for provider_idx in provider_ids:
                    receiver_ns = self.nodes[receiver_idx]
                    provider_ns = self.nodes[provider_idx]
                    pair = tuple(sorted((receiver_idx, provider_idx)))
                    self._record_decision_row(
                        {
                            "step": int(step),
                            "enc_id": encounter_ids[pair],
                            "node_i": receiver_idx,
                            "node_j": provider_idx,
                            "az": zone,
                            "dist": float(
                                np.hypot(
                                    receiver_ns.node.x - provider_ns.node.x,
                                    receiver_ns.node.y - provider_ns.node.y,
                                )
                            ),
                            "mode": mode,
                            "action": int(provider_idx in selected_set),
                            "merge_weight": "",
                            "predicted_gain": float(raw_by_id[provider_idx]),
                            "gain_threshold": (
                                self.communication_penalty
                                if self.decision_rule == "old-softmax"
                                else 0.0
                            ),
                            "exploratory": int(exploratory),
                            "reward": float("nan"),
                            "deferred": int(provider_idx in selected_set),
                            "bounded_score": float(bounded_by_id[provider_idx]),
                        }
                    )

    def _train_local(
        self,
        ns,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        sample_count_increment: int | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Train on the full, unweighted current-zone visit buffer."""

        del sample_weights
        n_new = (
            int(X.shape[0])
            if sample_count_increment is None
            else max(0, int(sample_count_increment))
        )
        X_step = (
            np.asarray(X[-n_new:], dtype=np.float32)
            if n_new
            else np.empty((0, int(X.shape[1])), dtype=np.float32)
        )
        y_step = (
            np.asarray(y_dbm[-n_new:], dtype=np.float32)
            if n_new
            else np.empty((0, 1), dtype=np.float32)
        )
        self._utility_current_samples[self.node_idx(ns)] = (
            int(ns.current_az),
            X_step.copy(),
            y_step.copy(),
        )
        receiver_idx = self.node_idx(ns)
        current_step = int(getattr(self, "_current_sumo_step", 0))
        for mode in self.agents:
            for pull in self._utility_pending.get((mode, receiver_idx), []):
                if int(pull.timestep) != current_step:
                    continue
                pull.initial_samples_x = np.asarray(X, dtype=np.float32).tolist()
                pull.initial_samples_y = np.asarray(
                    y_dbm, dtype=np.float32
                ).reshape(-1).tolist()

        if int(X.shape[0]) > 0:
            BootstrapPolicySharingSimulation._train_local(
                self,
                ns,
                X,
                y_dbm,
                sample_count_increment=n_new,
                sample_weights=None,
            )

    def _process_delayed_feedback(self, step: int) -> set[tuple[str, int]]:
        """Use the common fine-tune-window counterfactual for every strategy."""

        return super()._process_delayed_feedback(step)


    def _utility_reward_from_gain(self, raw_gain: float) -> float:
        """Retain the original beta penalty for old-softmax feedback."""

        penalty = (
            self.communication_penalty
            if self.decision_rule == "old-softmax"
            else 0.0
        )
        return float(raw_gain - penalty)
    def _build_communication_assumptions(self):
        assumptions = super()._build_communication_assumptions()
        assumptions.update(
            {
                "zramp_policy_mode": self.decision_rule,
                "policy_pull_rule": self.decision_rule,
                "communication_penalty_beta": self.communication_penalty,
                "policy_reward_target": (
                    "finetuned_frozen_local_plus_future_rmse_improvement_minus_beta"


                    if self.decision_rule == "old-softmax"
                    else "finetuned_frozen_local_plus_future_rmse_improvement"
                ),
                "predictor_aggregation": (
                    "sequential_experience_weighted; equivalent to one-shot "
                    "experience weighting for fixed provider snapshots"
                ),
                "aggregation_sample_scope": "none-experience-only",
                "pull_budget_expected_slots": (
                    "unbounded-old-softmax"
                    if self.decision_rule == "old-softmax"
                    else float(self.pull_budget)
                ),
                "policy_model_size": "micro-one-hidden-layer-64",
                "utility_loss": "mse",
            }
        )
        return assumptions


class CompactCurrentUtilitySelectionSimulation(UtilitySelectionBootstrapSimulation):
    """Unchanged current algorithm with per-contact CSV streaming disabled."""

    def _record_decision_row(self, row: dict) -> None:
        normalized = dict(row)
        self.decision_log.append(normalized)
        try:
            self._decision_action_counts[str(normalized["mode"])][
                int(normalized["action"])
            ] += 1
        except Exception:
            pass

