"""
Shared RNG setup for FROMTHEGROUND simulations.

Node placement and mobility use the same `random` / `torch` state for all methods in one run
(iso, greedy, RL share `self.nodes` and a single movement loop). Call `set_simulation_seed`
once before constructing `ZoneSharingSimulation` (or pass `seed=` into `__init__`).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_simulation_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
