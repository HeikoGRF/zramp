#!/usr/bin/env python3
"""Run the source-map parameter-novelty mechanism audit."""

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.parameter_novelty_oracle_simulation import (
    ParameterNoveltyOracleSimulation,
)


if __name__ == "__main__":
    raise SystemExit(
        main(simulation_cls=ParameterNoveltyOracleSimulation)
    )
