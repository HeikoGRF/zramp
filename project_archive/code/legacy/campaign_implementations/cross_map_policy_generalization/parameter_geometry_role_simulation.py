"""Tiny-map role simulation with parameter-only pull reward and aggregation."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import optim

from online_policy_learning.local_validation_reward import PullResult, interpolate_states
from online_policy_learning.online_local_validation_policy import ExactPrivateState, exact_model_groups
from online_policy_learning.parameter_geometry import (
    ParameterGeometryTracker,
    parameter_delta_state,
    select_geometry_aggregation,
)
from cross_map_policy_generalization.role_exact_simulation import RoleExactSequentialSimulation


TensorState = Mapping[str, torch.Tensor]


class ParameterGeometryRoleSimulation(RoleExactSequentialSimulation):
    """Use common-initialization geometry instead of validation for pulls."""

    def __init__(
        self,
        *args,
        geometry_radial_scale: float = 0.10,
        geometry_scale_floor: float = 0.05,
        geometry_ema_decay: float = 0.90,
        geometry_stability_warmup_updates: int = 8,
        geometry_retention_updates: int = 8,
        geometry_cancellation_penalty: float = 1.0,
        geometry_trust_penalty: float = 1.0,
        geometry_trust_radius: float = 1.0,
        geometry_alpha_max: float = 1.0,
        geometry_alpha_grid_size: int = 9,
        geometry_min_local_updates_between_merges: int = 0,
        **kwargs,
    ) -> None:
        self.geometry_radial_scale = float(geometry_radial_scale)
        self.geometry_scale_floor = float(geometry_scale_floor)
        self.geometry_ema_decay = float(geometry_ema_decay)
        self.geometry_stability_warmup_updates = int(
            geometry_stability_warmup_updates
        )
        self.geometry_retention_updates = int(geometry_retention_updates)
        self.geometry_cancellation_penalty = float(
            geometry_cancellation_penalty
        )
        self.geometry_trust_penalty = float(geometry_trust_penalty)
        self.geometry_trust_radius = float(geometry_trust_radius)
        self.geometry_alpha_max = float(geometry_alpha_max)
        self.geometry_alpha_grid_size = int(geometry_alpha_grid_size)
        self.geometry_min_local_updates_between_merges = max(
            0, int(geometry_min_local_updates_between_merges)
        )
        if not 0.0 < self.geometry_alpha_max <= 1.0:
            raise ValueError("geometry_alpha_max must be in (0, 1]")
        if self.geometry_alpha_grid_size < 2:
            raise ValueError("geometry_alpha_grid_size must be at least two")
        self._geometry_trackers: dict[
            str, list[ParameterGeometryTracker]
        ] = {}
        self._geometry_last: dict[str, float] = {}
        self._geometry_alpha_counts: Counter[float] = Counter()
        self._geometry_log_file = None
        self._geometry_log_writer: csv.DictWriter | None = None
        self._geometry_log_rows = 0
        # Parameter geometry does not use private validation. Keep every real
        # observation in predictor training rather than withholding 20%.
        kwargs.setdefault("train_all_observations", True)
        super().__init__(*args, **kwargs)
        if self.policy_training_target != "parameter-geometry":
            raise ValueError(
                "ParameterGeometryRoleSimulation requires "
                "--policy-training-target parameter-geometry"
            )
        self._initialize_geometry_trackers()
        self._communication_assumptions = self._build_communication_assumptions()

    def _new_geometry_tracker(
        self, reference: TensorState
    ) -> ParameterGeometryTracker:
        return ParameterGeometryTracker(
            reference,
            ema_decay=self.geometry_ema_decay,
            stability_warmup_updates=self.geometry_stability_warmup_updates,
            retention_updates=self.geometry_retention_updates,
            scale_floor=self.geometry_scale_floor,
        )

    def _initialize_geometry_trackers(self) -> None:
        self._geometry_trackers = {
            str(mode): [
                self._new_geometry_tracker(
                    self._clone_state(node.variants[str(mode)].model)
                )
                for node in self.nodes
            ]
            for mode in self.agents
        }

    def _geometry_tracker(
        self, node_idx: int, mode: str
    ) -> ParameterGeometryTracker:
        return self._geometry_trackers[str(mode)][int(node_idx)]

    def _write_geometry_log(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        geometry,
        adopted: bool,
    ) -> None:
        if self._geometry_log_writer is None:
            path = Path(self.cfg.results_dir) / "parameter_geometry_pulls.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._geometry_log_file = path.open(
                "w", newline="", encoding="utf-8"
            )
            fields = (
                "step",
                "mode",
                "receiver_idx",
                "provider_idx",
                "alpha",
                "gross_reward",
                "receiver_radial",
                "provider_radial",
                "receiver_training_stability",
                "provider_training_stability",
                "receiver_merge_stability",
                "provider_merge_stability",
                "receiver_maturity",
                "provider_maturity",
                "pair_distance",
                "novelty",
                "cosine",
                "cancellation_ratio",
                "trust_ratio",
                "objective_before",
                "objective_after",
                "adopted",
            )
            self._geometry_log_writer = csv.DictWriter(
                self._geometry_log_file, fieldnames=fields
            )
            self._geometry_log_writer.writeheader()
        self._geometry_log_writer.writerow(
            {
                "step": int(step),
                "mode": str(mode),
                "receiver_idx": int(receiver_idx),
                "provider_idx": int(provider_idx),
                "alpha": float(geometry.alpha),
                "gross_reward": float(geometry.gross_reward),
                "receiver_radial": float(geometry.receiver.radial_distance),
                "provider_radial": float(geometry.provider.radial_distance),
                "receiver_training_stability": float(
                    geometry.receiver.training_stability
                ),
                "provider_training_stability": float(
                    geometry.provider.training_stability
                ),
                "receiver_merge_stability": float(
                    geometry.receiver.merge_stability
                ),
                "provider_merge_stability": float(
                    geometry.provider.merge_stability
                ),
                "receiver_maturity": float(geometry.receiver.maturity),
                "provider_maturity": float(geometry.provider.maturity),
                "pair_distance": float(geometry.pair_distance),
                "novelty": float(geometry.normalized_novelty),
                "cosine": float(geometry.cosine),
                "cancellation_ratio": float(geometry.cancellation_ratio),
                "trust_ratio": float(geometry.trust_ratio),
                "objective_before": float(geometry.objective_before),
                "objective_after": float(geometry.objective_after),
                "adopted": int(adopted),
            }
        )
        self._geometry_log_rows += 1
        if self._geometry_log_rows % 250 == 0:
            self._geometry_log_file.flush()

    def _raw_state(
        self,
        node_idx: int,
        mode: str,
        *,
        model_state: TensorState | None = None,
    ) -> ExactPrivateState:
        if not self._geometry_trackers:
            return super()._raw_state(
                node_idx, mode, model_state=model_state
            )
        node = int(node_idx)
        mode_id = str(mode)
        if model_state is None:
            model_state = self.nodes[node].variants[mode_id].model.state_dict()
        tracker = self._geometry_tracker(node, mode_id)
        summary = tracker.summary(
            model_state, radial_scale=self.geometry_radial_scale
        )
        delta = parameter_delta_state(model_state, tracker.reference)
        trajectory_dim = self._predictor_input_dim() + 4
        trajectory = torch.zeros((1, trajectory_dim), dtype=torch.float32)
        trajectory[0, -4:] = torch.tensor(
            [
                summary.radial_distance
                / (summary.radial_distance + self.geometry_radial_scale),
                summary.training_stability,
                summary.merge_stability,
                1.0,
            ],
            dtype=torch.float32,
        )
        return ExactPrivateState(
            model_groups=exact_model_groups(delta),
            trajectory=trajectory,
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
        node_idx = self.node_idx(ns)
        before = (
            {
                str(mode): self._clone_state(ns.variants[str(mode)].model)
                for mode in self.agents
            }
            if self._geometry_trackers
            else {}
        )
        super()._train_local(
            ns,
            X,
            y_dbm,
            sample_count_increment=sample_count_increment,
            sample_weights=sample_weights,
        )
        for mode, state_before in before.items():
            state_after = self._clone_state(ns.variants[mode].model)
            self._geometry_tracker(node_idx, mode).observe_local_update(
                state_before, state_after
            )

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        super()._reset_respawned_node(node_idx, generation=generation)
        if not self._geometry_trackers:
            return
        node = int(node_idx)
        for mode in self.agents:
            self._geometry_trackers[str(mode)][node] = (
                self._new_geometry_tracker(
                    self._clone_state(
                        self.nodes[node].variants[str(mode)].model
                    )
                )
            )

    def _execute_validation_pull(
        self,
        *,
        step: int,
        mode: str,
        receiver,
        provider,
        zone: int,
        provider_view: object | None,
        diagnostic: bool = False,
    ) -> PullResult:
        mode_id = str(mode)
        receiver_idx = self.node_idx(receiver)
        provider_idx = self.node_idx(provider)
        receiver_variant = receiver.variants[mode_id]
        state_a = self._clone_state(receiver_variant.model)
        state_b = (
            {
                name: value.detach().cpu().clone()
                for name, value in provider_view._model_state.items()  # type: ignore[attr-defined]
            }
            if provider_view is not None
            else self._clone_state(provider.variants[mode_id].model)
        )
        tracker_a = self._geometry_tracker(receiver_idx, mode_id)
        tracker_b = self._geometry_tracker(provider_idx, mode_id)
        alpha_grid = np.linspace(
            0.0, self.geometry_alpha_max, self.geometry_alpha_grid_size
        )
        geometry = select_geometry_aggregation(
            state_a,
            state_b,
            tracker_a,
            tracker_b,
            alpha_grid=alpha_grid,
            radial_scale=self.geometry_radial_scale,
            cancellation_penalty=self.geometry_cancellation_penalty,
            trust_penalty=self.geometry_trust_penalty,
            trust_radius=self.geometry_trust_radius,
        )
        consolidation_ready = (
            tracker_a.local_updates_since_merge
            >= self.geometry_min_local_updates_between_merges
        )
        model_bytes = self._state_nbytes(state_a) + self._state_nbytes(state_b)
        metadata_bytes = int(
            self._metadata_for(provider_idx, mode_id).wire_nbytes
        )
        adopted = bool(
            consolidation_ready
            and geometry.alpha < 1.0
            and geometry.gross_reward > 1.0e-12
        )
        effective_alpha = float(geometry.alpha if consolidation_ready else 1.0)
        effective_reward = float(
            geometry.gross_reward if consolidation_ready else 0.0
        )
        if not diagnostic:
            self._cv_last_provider_pull_step[
                (mode_id, receiver_idx, provider_idx, int(zone))
            ] = int(step)
            self._cv_step_pulls[mode_id] += 1
            self._cv_step_valid_pulls[mode_id] += 1
            self._cv_step_model_messages[mode_id] += 2
            self._cv_step_model_bytes[mode_id] += model_bytes
            # Provider training stability, merge stability, and selected alpha.
            scalar_control_messages = 3
            self._cv_step_scalar_control_messages[mode_id] += (
                scalar_control_messages
            )
            self._cv_step_scalar_messages[mode_id] += scalar_control_messages
            self._cv_step_scalar_bytes[mode_id] += 4 * scalar_control_messages
            if adopted:
                aggregate = interpolate_states(state_a, state_b, geometry.alpha)
                self._load_model_state(receiver_variant.model, aggregate)
                receiver_variant.opt = optim.Adam(
                    receiver_variant.model.parameters(), lr=self.cfg.local_lr
                )
                receiver_variant.t_wait = 0
                receiver_variant.last_rmse_available = False
                self._refresh_variant_signature(receiver_variant)
                self._cv_receiver_aggregations[
                    (mode_id, int(receiver_idx))
                ] += 1
                tracker_a.inherit_merge(
                    tracker_b,
                    alpha=geometry.alpha,
                    before=state_a,
                    after=aggregate,
                )
                self._geometry_alpha_counts[float(geometry.alpha)] += 1
            self._geometry_last = {
                "receiver_maturity": geometry.receiver.maturity,
                "provider_maturity": geometry.provider.maturity,
                "receiver_radial": geometry.receiver.radial_distance,
                "provider_radial": geometry.provider.radial_distance,
                "receiver_training_stability": geometry.receiver.training_stability,
                "provider_training_stability": geometry.provider.training_stability,
                "receiver_merge_stability": geometry.receiver.merge_stability,
                "provider_merge_stability": geometry.provider.merge_stability,
                "pair_distance": geometry.pair_distance,
                "novelty": geometry.normalized_novelty,
                "cosine": geometry.cosine,
                "cancellation_ratio": geometry.cancellation_ratio,
                "trust_ratio": geometry.trust_ratio,
                "alpha": geometry.alpha,
                "gross_reward": geometry.gross_reward,
            }
            self._write_geometry_log(
                step=int(step),
                mode=mode_id,
                receiver_idx=receiver_idx,
                provider_idx=provider_idx,
                geometry=geometry,
                adopted=adopted,
            )
        result = PullResult(
            valid=True,
            reason=(
                "geometry_adopted"
                if adopted
                else "geometry_consolidating"
                if not consolidation_ready
                else "geometry_retained"
            ),
            alpha=effective_alpha,
            objective_evaluations=len(geometry.evaluations),
            reward=effective_reward,
            joint_reward=effective_reward,
            receiver_reward=effective_reward,
            adopted=adopted,
            receiver_adopted=adopted,
            parameter_geometry_reward=effective_reward,
            parameter_geometry_alpha=effective_alpha,
            scalar_control_messages=3,
            scalar_messages=3,
        )
        if not diagnostic:
            self._write_pull_log(
                step,
                mode_id,
                receiver_idx,
                provider_idx,
                zone,
                result,
                0.0,
                0.0,
                0.0,
                0.0,
                metadata_bytes,
                model_bytes,
            )
        return result

    def _build_communication_assumptions(self) -> dict[str, object]:
        assumptions = super()._build_communication_assumptions()
        assumptions.update(
            {
                "predictor_aggregation": "parameter-geometry-grid",
                "aggregation_validation_samples_used": False,
                "parameter_geometry_reference": "common-initialization",
                "parameter_geometry_policy_input": (
                    "delta-from-initialization,local-training-stability,"
                    "post-merge-retention-stability"
                ),
                "parameter_geometry_sample_count_used": False,
                "all_real_observations_train_predictor": True,
                "all_artificial_observations_train_predictor": bool(
                    not self.artificial_validation
                ),
                "parameter_geometry_radial_scale": self.geometry_radial_scale,
                "parameter_geometry_scale_floor": self.geometry_scale_floor,
                "parameter_geometry_ema_decay": self.geometry_ema_decay,
                "parameter_geometry_stability_warmup_updates": (
                    self.geometry_stability_warmup_updates
                ),
                "parameter_geometry_retention_updates": (
                    self.geometry_retention_updates
                ),
                "parameter_geometry_cancellation_penalty": (
                    self.geometry_cancellation_penalty
                ),
                "parameter_geometry_trust_penalty": self.geometry_trust_penalty,
                "parameter_geometry_trust_radius": self.geometry_trust_radius,
                "parameter_geometry_alpha_max": self.geometry_alpha_max,
                "parameter_geometry_alpha_grid_size": (
                    self.geometry_alpha_grid_size
                ),
                "parameter_geometry_min_local_updates_between_merges": (
                    self.geometry_min_local_updates_between_merges
                ),
            }
        )
        return assumptions

    def _role_experiment_metadata(self) -> dict[str, object]:
        metadata = super()._role_experiment_metadata()
        metadata.update(
            {
                "aggregation_validation_samples_used": False,
                "policy_reward_source": "parameter-geometry",
                "policy_decision_rule": (
                    "pull iff predicted parameter-geometry improvement minus "
                    "communication cost exceeds the configured trigger"
                ),
                "parameter_geometry_sample_count_used": False,
                "all_real_observations_train_predictor": True,
                "parameter_geometry_alpha_counts": {
                    f"{alpha:g}": int(count)
                    for alpha, count in sorted(
                        self._geometry_alpha_counts.items()
                    )
                },
                "parameter_geometry_last": dict(self._geometry_last),
            }
        )
        return metadata

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._geometry_log_file is not None:
                self._geometry_log_file.flush()
                self._geometry_log_file.close()
            self._geometry_log_file = None
            self._geometry_log_writer = None
