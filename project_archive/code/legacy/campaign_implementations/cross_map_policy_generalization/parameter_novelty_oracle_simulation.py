"""Source-map mechanism audit selecting the most parameter-novel provider."""

from __future__ import annotations

import os
from pathlib import Path

from online_policy_learning.parameter_geometry import select_geometry_aggregation
from cross_map_policy_generalization.parameter_geometry_role_simulation import (
    ParameterGeometryRoleSimulation,
)
from cross_map_policy_generalization.parameter_objective_audit import (
    ParameterObjectiveAuditMixin,
)


DEFAULT_CONTACTS = Path("/tmp/source_contacts_missing.npz")


class ParameterNoveltyOracleSimulation(
    ParameterObjectiveAuditMixin,
    ParameterGeometryRoleSimulation,
):
    """Select providers by normalized distance from the receiver model."""

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

    def _evidence_novelty_scores(
        self,
        *,
        receiver_idx: int,
        mode: str,
        candidate_ids: list[int],
        provider_views,
    ) -> dict[int, float]:
        """Return normalized receiver-provider parameter novelty."""

        mode_id = str(mode)
        state_a = self._clone_state(
            self.nodes[int(receiver_idx)].variants[mode_id].model
        )
        tracker_a = self._geometry_tracker(int(receiver_idx), mode_id)
        scores: dict[int, float] = {}
        for provider_idx in candidate_ids:
            state_b = dict(provider_views[int(provider_idx)]._model_state)
            geometry = select_geometry_aggregation(
                state_a,
                state_b,
                tracker_a,
                self._geometry_tracker(int(provider_idx), mode_id),
                alpha_grid=(1.0,),
                radial_scale=self.geometry_radial_scale,
                cancellation_penalty=self.geometry_cancellation_penalty,
                trust_penalty=self.geometry_trust_penalty,
                trust_radius=self.geometry_trust_radius,
            )
            scores[int(provider_idx)] = float(
                geometry.normalized_novelty
            )
        return scores
