"""Source-map policy simulation rewarded by provider maturity."""

from __future__ import annotations

from dataclasses import replace

import os
from pathlib import Path

from cross_map_policy_generalization.parameter_geometry_role_simulation import (
    ParameterGeometryRoleSimulation,
)
from cross_map_policy_generalization.parameter_objective_audit import (
    ParameterObjectiveAuditMixin,
)


DEFAULT_CONTACTS = Path("/tmp/source_contacts_missing.npz")


class SourceMaturityRewardSimulation(
    ParameterObjectiveAuditMixin,
    ParameterGeometryRoleSimulation,
):
    """Use model-only provider maturity as the post-pull target."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "all_link_trace",
            Path(
                os.environ.get(
                    "SOURCE_CONTACT_MASK", str(DEFAULT_CONTACTS)
                )
            ),
        )
        super().__init__(*args, **kwargs)

    def _execute_validation_pull(self, **kwargs):
        mode = str(kwargs["mode"])
        provider = kwargs["provider"]
        provider_idx = self.node_idx(provider)
        provider_view = kwargs.get("provider_view")
        state = (
            dict(provider_view._model_state)
            if provider_view is not None
            else self._clone_state(provider.variants[mode].model)
        )
        maturity = float(
            self._geometry_tracker(provider_idx, mode).summary(
                state, radial_scale=self.geometry_radial_scale
            ).maturity
        )
        result = super()._execute_validation_pull(**kwargs)
        return replace(
            result,
            reward=maturity,
            joint_reward=maturity,
            receiver_reward=maturity,
            parameter_geometry_reward=maturity,
        )
