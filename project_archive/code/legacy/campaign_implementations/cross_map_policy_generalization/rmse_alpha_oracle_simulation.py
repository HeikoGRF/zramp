"""Non-deployable provider/alpha oracle for source-map mechanism tests.

Both oracle-provider and random-provider runs use the same RMSE-optimal alpha
rule.  Fixed-map labels are diagnostic only and never enter a deployable
policy, predictor, or communication payload.
"""

from __future__ import annotations

import numpy as np
from torch import optim

from online_policy_learning.local_validation_reward import PullResult, interpolate_states
from online_policy_learning.parameter_geometry import select_geometry_aggregation
from cross_map_policy_generalization.policy_source_parameter_objective_audit import (
    PolicySourceParameterObjectiveAuditSimulation,
)


class RmseAlphaOracleSimulation(PolicySourceParameterObjectiveAuditSimulation):
    """Use true map RMSE to choose alpha after a provider is selected."""

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
        geometry = select_geometry_aggregation(
            state_a,
            state_b,
            tracker_a,
            tracker_b,
            alpha_grid=np.linspace(
                0.0, self.geometry_alpha_max, self.geometry_alpha_grid_size
            ),
            radial_scale=self.geometry_radial_scale,
            cancellation_penalty=self.geometry_cancellation_penalty,
            trust_penalty=self.geometry_trust_penalty,
            trust_radius=self.geometry_trust_radius,
        )
        consolidation_ready = (
            tracker_a.local_updates_since_merge
            >= self.geometry_min_local_updates_between_merges
        )
        baseline_rmse = float(self._evaluate_state(state_a)[0])
        evaluations: list[tuple[float, float, dict[str, object] | None]] = [
            (1.0, baseline_rmse, None)
        ]
        if consolidation_ready:
            for alpha in np.linspace(
                0.0, self.geometry_alpha_max, self.geometry_alpha_grid_size
            ):
                alpha_value = float(alpha)
                if abs(alpha_value - 1.0) <= 1.0e-12:
                    continue
                aggregate = interpolate_states(state_a, state_b, alpha_value)
                evaluations.append(
                    (
                        alpha_value,
                        float(self._evaluate_state(aggregate)[0]),
                        aggregate,
                    )
                )
        alpha, candidate_rmse, aggregate = min(
            evaluations,
            key=lambda row: (float(row[1]), -float(row[0])),
        )
        gain = float(baseline_rmse - candidate_rmse)
        adopted = bool(
            consolidation_ready
            and aggregate is not None
            and alpha < 1.0
            and gain > 1.0e-12
        )
        model_bytes = self._state_nbytes(state_a) + self._state_nbytes(state_b)
        metadata_bytes = int(
            self._metadata_for(provider_idx, mode_id).wire_nbytes
        )

        result = PullResult(
            valid=True,
            reason=(
                "rmse_alpha_adopted"
                if adopted
                else "rmse_alpha_consolidating"
                if not consolidation_ready
                else "rmse_alpha_retained"
            ),
            alpha=float(alpha),
            objective_evaluations=len(evaluations),
            before_loss=baseline_rmse,
            after_loss=candidate_rmse,
            reward=gain,
            joint_reward=gain,
            receiver_before_loss=baseline_rmse,
            receiver_after_loss=candidate_rmse,
            receiver_reward=gain,
            adopted=adopted,
            receiver_adopted=adopted,
            parameter_geometry_reward=gain,
            parameter_geometry_alpha=float(alpha),
            scalar_control_messages=3,
            scalar_messages=3,
        )
        if diagnostic:
            return result

        self._cv_last_provider_pull_step[
            (mode_id, receiver_idx, provider_idx, int(zone))
        ] = int(step)
        self._cv_step_pulls[mode_id] += 1
        self._cv_step_valid_pulls[mode_id] += 1
        self._cv_step_model_messages[mode_id] += 2
        self._cv_step_model_bytes[mode_id] += model_bytes
        self._cv_step_scalar_control_messages[mode_id] += 3
        self._cv_step_scalar_messages[mode_id] += 3
        self._cv_step_scalar_bytes[mode_id] += 12
        if adopted:
            assert aggregate is not None
            self._load_model_state(receiver_variant.model, aggregate)
            receiver_variant.opt = optim.Adam(
                receiver_variant.model.parameters(), lr=self.cfg.local_lr
            )
            receiver_variant.t_wait = 0
            receiver_variant.last_rmse_available = False
            self._refresh_variant_signature(receiver_variant)
            self._cv_receiver_aggregations[(mode_id, receiver_idx)] += 1
            tracker_a.inherit_merge(
                tracker_b,
                alpha=float(alpha),
                before=state_a,
                after=aggregate,
            )
            self._geometry_alpha_counts[float(alpha)] += 1
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
            "alpha": float(alpha),
            "gross_reward": gain,
        }
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

