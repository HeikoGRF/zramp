#!/usr/bin/env python3
"""Run the source-map relative-maturity mechanism audit."""

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.source_relative_maturity_oracle import (
    SourceRelativeMaturityOracleSimulation,
)


if __name__ == "__main__":
    raise SystemExit(
        main(simulation_cls=SourceRelativeMaturityOracleSimulation)
    )
