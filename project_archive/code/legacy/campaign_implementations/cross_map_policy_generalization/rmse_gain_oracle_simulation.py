"""Diagnostic provider oracle using fixed-map RMSE after the real aggregation.

The fixed evaluation labels are used only while ``selection_mode=oracle`` asks
which already-feasible provider would have produced the best immediate result.
The actual pull still uses the deployable parameter-geometry aggregation rule.
This class is therefore a non-deployable positive control, not a policy.
"""

from __future__ import annotations

from dataclasses import replace

from online_policy_learning.local_validation_reward import PullResult, interpolate_states
from cross_map_policy_generalization.policy_source_parameter_objective_audit import (
    PolicySourceParameterObjectiveAuditSimulation,
)


class RmseGainOracleSimulation(PolicySourceParameterObjectiveAuditSimulation):
    """Rank providers by true immediate RMSE gain for a diagnostic rollout."""

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
        result = super()._execute_validation_pull(
            step=step,
            mode=mode,
            receiver=receiver,
            provider=provider,
            zone=zone,
            provider_view=provider_view,
            diagnostic=diagnostic,
        )
        if not diagnostic or not result.valid:
            return result

        mode_id = str(mode)
        state_a = self._clone_state(receiver.variants[mode_id].model)
        baseline_rmse = float(self._evaluate_state(state_a)[0])
        candidate_rmse = baseline_rmse
        if bool(result.adopted) and result.alpha is not None:
            state_b = (
                {
                    name: value.detach().cpu().clone()
                    for name, value in provider_view._model_state.items()  # type: ignore[attr-defined]
                }
                if provider_view is not None
                else self._clone_state(provider.variants[mode_id].model)
            )
            aggregate = interpolate_states(
                state_a, state_b, float(result.alpha)
            )
            candidate_rmse = float(self._evaluate_state(aggregate)[0])
        gain = float(baseline_rmse - candidate_rmse)
        return replace(
            result,
            before_loss=baseline_rmse,
            after_loss=candidate_rmse,
            receiver_before_loss=baseline_rmse,
            receiver_after_loss=candidate_rmse,
            reward=gain,
            receiver_reward=gain,
            joint_reward=gain,
            parameter_geometry_reward=gain,
        )

