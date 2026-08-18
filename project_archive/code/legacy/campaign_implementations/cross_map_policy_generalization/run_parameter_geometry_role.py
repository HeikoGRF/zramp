#!/usr/bin/env python3
"""Run the tiny-map role experiment with parameter-geometry aggregation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.parameter_geometry_role_simulation import (
    ParameterGeometryRoleSimulation,
)


if __name__ == "__main__":
    raise SystemExit(main(simulation_cls=ParameterGeometryRoleSimulation))
