"""
Per-mode state carried by every node.

Each node owns an independent model / optimiser / replay statistics per active
reward mode so the variants evolve without interfering with each other.
Mode-specific extras such as pending reward slots live next to the per-variant
state but are only populated for the modes that use them.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.optim as optim


# Soft cap for `n_samples`. The merge rule remains additive on the *raw* sum
# but is then passed through `saturate_n_samples`, which asymptotes smoothly
# to `N_SAMPLES_CAP` instead of growing without bound. Below ~cap/10 the
# saturation is effectively the identity, so existing additive behaviour is
# preserved; above that, increments yield diminishing returns and the stored
# value converges to the cap.
N_SAMPLES_CAP: int = 100_000
# Values this large already map to the cap numerically. Keeping raw counters
# bounded prevents dense repeated model pulls from creating enormous Python ints.
RAW_SAMPLES_BOUND: int = N_SAMPLES_CAP * 1_000


def bound_raw_samples(n_raw: int | float) -> int:
    """Bound the raw additive counter without changing the capped score in practice."""
    if n_raw <= 0:
        return 0
    n_int = int(n_raw)
    if n_int >= RAW_SAMPLES_BOUND:
        return RAW_SAMPLES_BOUND
    return n_int


def saturate_n_samples(n_raw: float) -> int:
    """Smooth soft cap on `n_samples`.

    Uses an exponential saturation `cap * (1 - exp(-n_raw / cap))` so that:

    - `sat(0) = 0`,
    - `sat(n) ≈ n` for `n << cap` (additive semantics preserved),
    - `sat(n) -> cap` monotonically as `n -> infinity` (no overflow possible).

    The result is rounded to int because `n_samples` is stored as an integer
    counter. Negative inputs (which should never occur in practice) are
    clamped to 0.
    """
    n_bounded = bound_raw_samples(n_raw)
    if n_bounded <= 0:
        return 0
    if n_bounded >= RAW_SAMPLES_BOUND:
        return N_SAMPLES_CAP
    cap = float(N_SAMPLES_CAP)
    return int(round(cap * (1.0 - math.exp(-float(n_bounded) / cap))))


@dataclass
class VariantState:
    """Per-mode model / optimiser / bookkeeping for one node.

    `experience` is computed on demand from `n_samples` and is therefore not
    stored directly. `m_samples` keeps the raw additive count from local updates
    and merges; `n_samples` is the capped score used for experience.
    """

    model: torch.nn.Module
    opt: optim.Optimizer
    m_samples: int = 0
    n_samples: int = 0
    quality: float = 0.0
    t_wait: int = 0
    # Legacy placeholder for code paths that still consume `last_rmse`
    # directly. New contact policies must check `last_rmse_available` instead
    # of treating this initial value as a real recent error estimate.
    last_rmse: float = 45.0
    last_rmse_available: bool = False
    rmse_ema_short: float = 0.0
    rmse_ema_long: float = 0.0
    rmse_batches: int = 0
    model_signature: torch.Tensor = field(default_factory=lambda: torch.zeros(16, dtype=torch.float32))
    recovery_steps_left: int = 0
    recovery_accepts_left: int = 0
    recovery_cooldown_left: int = 0
    @property
    def experience(self) -> float:
        """Experience is the capped sample count."""
        return float(self.n_samples)


@dataclass
class PendingSlot:
    """One outstanding reward window for a single node and mode."""

    step_started: int
    action: int
    state: torch.Tensor
    next_state: torch.Tensor
    done: bool
    beta: float
    # Pre-merge weights copy (CPU tensors) so the post-merge model can be
    # scored against a reference trained state later.
    pre_weights: dict[str, torch.Tensor]
    post_weights: dict[str, torch.Tensor]
    # Collected (normalised) samples from this node's future activity.
    samples_x: list[list[float]] = field(default_factory=list)
    samples_y: list[float] = field(default_factory=list)
    # Number of simulation *steps* (not samples) to collect before maturing.
    target_steps: int = 1


@dataclass
class NodeState:
    """Full per-node state across all active reward modes.

    `visited_bitmap` is a packed `int` whose bits flag the K x K grid cells the
    node has visited inside its current zone. It is retained as local movement
    bookkeeping; merge experience is based only on capped sample count.

    `current_visit_samples_*` stores the receiver-side observations collected
    since the node entered its current anchor zone. Zone transitions clear this
    buffer; zone-memory restores model weights only, not old raw samples.
    """

    node: Any  # parent `node.Node` instance
    current_az: int
    variants: dict[str, VariantState]
    pending_slots: dict[str, deque[PendingSlot]] = field(default_factory=dict)
    visited_bitmap: int = 0
    current_visit_samples_x: list[list[float]] = field(default_factory=list)
    current_visit_samples_y: list[float] = field(default_factory=list)
    current_visit_sample_steps: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ utils

    def reset_mode(self, mode: str, template_state: dict[str, torch.Tensor], lr: float) -> None:
        """Reset one variant back to template weights (used on zone transitions)."""
        v = self.variants[mode]
        v.model.load_state_dict(template_state)
        v.opt = optim.Adam(v.model.parameters(), lr=lr)
        v.m_samples = 0
        v.n_samples = 0
        v.quality = 0.0
        v.t_wait = 0
        v.last_rmse = 45.0
        v.last_rmse_available = False
        v.rmse_ema_short = 0.0
        v.rmse_ema_long = 0.0
        v.rmse_batches = 0
        v.model_signature = torch.zeros_like(v.model_signature)
        v.recovery_steps_left = 0
        v.recovery_accepts_left = 0
        v.recovery_cooldown_left = 0

    def reset_all(self, template_state: dict[str, torch.Tensor], lr: float) -> None:
        for m in list(self.variants.keys()):
            self.reset_mode(m, template_state, lr)
        self.pending_slots.clear()
        self.visited_bitmap = 0
        self.clear_current_visit_samples()

    def clear_current_visit_samples(self) -> None:
        self.current_visit_samples_x.clear()
        self.current_visit_samples_y.clear()
        self.current_visit_sample_steps.clear()

    def pending_for(self, mode: str) -> deque[PendingSlot]:
        """Return this node's pending reward queue for one reward mode."""
        return self.pending_slots.setdefault(mode, deque())
