#!/usr/bin/env python3
"""Run the bounded decentralized expert-bank policy/random benchmark."""

from __future__ import annotations

from .expert_bank_simulation import ExpertBankSampleSharingSimulation
from .run_online_local_validation_policy import main


if __name__ == "__main__":
    raise SystemExit(main(simulation_cls=ExpertBankSampleSharingSimulation))
