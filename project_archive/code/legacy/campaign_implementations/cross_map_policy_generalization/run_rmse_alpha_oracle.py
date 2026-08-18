#!/usr/bin/env python3
"""Run the source-only provider/alpha oracle mechanism audit."""

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.rmse_alpha_oracle_simulation import (
    RmseAlphaOracleSimulation,
)


if __name__ == "__main__":
    raise SystemExit(main(simulation_cls=RmseAlphaOracleSimulation))

