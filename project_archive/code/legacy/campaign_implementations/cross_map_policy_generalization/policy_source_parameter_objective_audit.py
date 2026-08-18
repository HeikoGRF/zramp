"""Deployment-faithful exact audit on synthetic source maps."""

from __future__ import annotations

import os
from pathlib import Path

from cross_map_policy_generalization.parameter_geometry_role_simulation import (
    ParameterGeometryRoleSimulation,
)
from cross_map_policy_generalization.parameter_objective_audit import (
    ParameterObjectiveAuditMixin,
)


DEFAULT_CONTACTS = Path("/tmp/source_contacts_missing.npz")


class PolicySourceParameterObjectiveAuditSimulation(
    ParameterObjectiveAuditMixin,
    ParameterGeometryRoleSimulation,
):
    """Use an explicit static-building-clear source contact mask."""

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
