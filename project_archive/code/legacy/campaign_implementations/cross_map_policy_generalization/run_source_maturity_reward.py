#!/usr/bin/env python3
"""Run the source-map maturity-reward policy simulation."""

from online_policy_learning.run_online_local_validation_policy import main
from cross_map_policy_generalization.source_maturity_reward_simulation import (
    SourceMaturityRewardSimulation,
)


if __name__ == "__main__":
    raise SystemExit(
        main(simulation_cls=SourceMaturityRewardSimulation)
    )
