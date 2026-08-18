#!/usr/bin/env python3
"""Run the non-deployable source-map RMSE-gain provider oracle."""

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.rmse_gain_oracle_simulation import (
    RmseGainOracleSimulation,
)


if __name__ == "__main__":
    raise SystemExit(main(simulation_cls=RmseGainOracleSimulation))

