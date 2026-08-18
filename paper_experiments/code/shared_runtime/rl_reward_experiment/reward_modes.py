"""
Reward-mode implementations sharing a common interface.

Each mode computes the per-encounter reward independently and, if necessary,
defers the reward until T future simulation steps have elapsed.

Interface (see `RewardMode`):

- `on_step_start(sim, step)`           - bookkeeping before gossip.
- `on_encounter(...)` -> Transition?   - immediate reward, or None if deferred.
- `on_step_end(sim, step)`             - return list of matured transitions.
- `on_sim_end(sim)`                    - flush any remaining deferred slots.

`on_encounter` receives a `j_view` argument: an immutable view of the
provider variant's *pre-link* state (`n_samples`, `last_rmse`,
plus a scratch `nn.Module` already loaded with the provider's pre-link
weights). When non-`None`, modes MUST forward it to `sim.perform_merge` so
that bidirectional encounters within the same physical link are
order-independent (no `n_samples` double-counting, no leg-2 sees mutated
provider). The first leg of a link is invoked with `j_view=None`, which
means "use the provider's live state" (still pre-link at that point).

Transition type:

    (state: torch.Tensor, action: int, reward: float,
     next_state: torch.Tensor, done: bool, node_index: int)

All transitions are emitted with `done=True` and `next_state == state`
(bandit-style). See `docs/rl_reward_method_comparison.md`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from .config import (
    ExperimentConfig,
    SPIKE_RECOVERY_PROFILES,
    WINDOW_T_BY_MODE,
    WINDOW_T_VALUES,
    split_mode_policy,
    split_spike_profile,
)
from .node_state import NodeState, PendingSlot

if TYPE_CHECKING:
    from .sim import Simulation, _LinkPeerView


Transition = tuple[torch.Tensor, int, float, torch.Tensor, bool, int]


# --- Base class -------------------------------------------------------------

class RewardMode:
    id: str = ""
    name: str = ""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.beta = float(cfg.beta)

    def on_step_start(self, sim: "Simulation", step: int) -> None:
        """Per-step hook fired before gossip. Default no-op."""

    def on_encounter(
        self,
        sim: "Simulation",
        step: int,
        ns_i: NodeState,
        ns_j: NodeState,
        az: int,
        action: int,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
        j_view: Optional["_LinkPeerView"] = None,
    ) -> Optional[Transition]:
        raise NotImplementedError

    def on_step_end(self, sim: "Simulation", step: int) -> list[Transition]:
        return []

    def on_sim_end(self, sim: "Simulation") -> list[Transition]:
        return []


# --- FutureWindow(T) --------------------------------------------------------

class FutureWindowMode(RewardMode):
    """
    At each encounter, store shadow copies of both the pre-merge and
    post-merge weights of this node's model. Over the next T simulation
    *steps* (regardless of how many individual samples arrive per step),
    accumulate all receiver-measurements for this node. Once T steps have
    elapsed since the slot was opened (or the sim ends / slot cap is
    exceeded), compute:

        reward = rmse(pre, samples) - rmse(post, samples) - beta * action

    T=1 means: use all samples collected in the step immediately following
    the merge decision. T=3 means: pool samples from the 3 steps after the
    merge. This gives T a genuine temporal-horizon interpretation rather than
    being a sample-count threshold.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        mode_id: str,
        window_t: int,
        *,
        reward_beta: float | None = None,
        spike_profile: str | None = None,
    ):
        super().__init__(cfg)
        self.id = mode_id
        self.window_t = int(window_t)
        self.spike_profile = spike_profile
        self.spike_recovery_enabled = bool(cfg.spike_recovery_enabled or spike_profile is not None)
        self.spike_recovery_params = {
            "short_alpha": float(cfg.spike_recovery_short_alpha),
            "long_alpha": float(cfg.spike_recovery_long_alpha),
            "ratio": float(cfg.spike_recovery_ratio),
            "abs_db": float(cfg.spike_recovery_abs_db),
            "min_batches": int(cfg.spike_recovery_min_batches),
            "window_steps": int(cfg.spike_recovery_window_steps),
            "accept_budget": int(cfg.spike_recovery_accept_budget),
            "cooldown_steps": int(cfg.spike_recovery_cooldown_steps),
            "beta_scale": float(cfg.spike_recovery_beta_scale),
            "accept_prob": float(cfg.spike_recovery_accept_prob),
        }
        if spike_profile is not None:
            self.spike_recovery_params.update(SPIKE_RECOVERY_PROFILES[spike_profile])
        if reward_beta is not None:
            self.beta = float(reward_beta)
        self.name = f"FutureWindowT{self.window_t}"
        if reward_beta is not None:
            self.name += f" β={self.beta:g}"
        if spike_profile is not None:
            self.name += f" spike={spike_profile}"
        self._overflow: list[Transition] = []

    def _snapshot(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    def on_encounter(self, sim, step, ns_i, ns_j, az, action, state, next_state, done, j_view=None):
        pre = self._snapshot(ns_i.variants[self.id].model)
        v_i = ns_i.variants[self.id]
        beta = float(self.beta)
        in_recovery = bool(
            self.spike_recovery_enabled
            and v_i.recovery_steps_left > 0
            and v_i.recovery_accepts_left > 0
        )
        if in_recovery:
            beta *= float(self.spike_recovery_params["beta_scale"])
        if action == 1:
            sim.perform_merge(ns_i, ns_j, self.id, j_view=j_view)
            ns_i.variants[self.id].t_wait = 0
            if in_recovery:
                v_i.recovery_accepts_left = max(0, int(v_i.recovery_accepts_left) - 1)
                if v_i.recovery_accepts_left <= 0:
                    v_i.recovery_steps_left = 0
                    v_i.recovery_cooldown_left = max(
                        v_i.recovery_cooldown_left,
                        int(self.spike_recovery_params["cooldown_steps"]),
                    )
        post = self._snapshot(ns_i.variants[self.id].model)

        slot = PendingSlot(
            step_started=int(step),
            action=int(action),
            state=state.detach().cpu(),
            next_state=next_state.detach().cpu(),
            done=bool(done),
            beta=beta,
            pre_weights=pre,
            post_weights=post,
            target_steps=self.window_t,
        )
        pending = ns_i.pending_for(self.id)
        pending.append(slot)

        while len(pending) > self.cfg.pending_slot_cap:
            old = pending.popleft()
            t = self._finalize_slot(sim, ns_i, old, step_finalized=step)
            if t is not None:
                self._overflow.append(t)
        return None

    # -- helpers ------------------------------------------------------------

    def _finalize_slot(
        self,
        sim: "Simulation",
        ns: NodeState,
        slot: PendingSlot,
        step_finalized: int | None = None,
    ) -> Optional[Transition]:
        if not slot.samples_x:
            # No samples ever landed in this slot (e.g. the node left the zone
            # before any future-window measurement could be ingested). Skip
            # the transition entirely instead of emitting a misleading
            # `r=0` neutral sample, which would teach the agent that the
            # action it took had a tiny negative reward `-beta * action`
            # regardless of merge quality.
            return None
        X = np.asarray(slot.samples_x, dtype=np.float32)
        y = np.asarray(slot.samples_y, dtype=np.float32).reshape(-1, 1)
        mse_pre = sim.eval_mse_with_weights(self.id, slot.pre_weights, X, y)
        mse_post = sim.eval_mse_with_weights(self.id, slot.post_weights, X, y)
        rmse_pre = float(np.sqrt(mse_pre))
        rmse_post = float(np.sqrt(mse_post))
        reward = float(rmse_pre - rmse_post) - slot.beta * float(slot.action)
        if hasattr(sim, "merge_eval_rows"):
            sim.merge_eval_rows.append(
                {
                    "mode": self.id,
                    "node_idx": int(sim.node_idx(ns)),
                    "step_started": int(slot.step_started),
                    "step_finalized": int(
                        step_finalized if step_finalized is not None else -1
                    ),
                    "action": int(slot.action),
                    "beta": float(slot.beta),
                    "n_samples": int(X.shape[0]),
                    "mse_pre": float(mse_pre),
                    "mse_post": float(mse_post),
                    "mse_improvement": float(mse_pre - mse_post),
                    "rmse_pre": float(rmse_pre),
                    "rmse_post": float(rmse_post),
                    "rmse_improvement": float(rmse_pre - rmse_post),
                    "reward": float(reward),
                }
            )
        return (
            slot.state,
            slot.action,
            reward,
            slot.next_state,
            True,
            sim.node_idx(ns),
        )

    # -- stream integration -------------------------------------------------

    def ingest_sample(
        self,
        ns: NodeState,
        x_norm: list[float],
        y_dbm: float,
    ) -> None:
        """Append one new measurement to all of this node's open pending slots."""
        for slot in ns.pending_for(self.id):
            slot.samples_x.append(list(x_norm))
            slot.samples_y.append(float(y_dbm))

    def on_step_end(self, sim, step):
        ready: list[Transition] = list(self._overflow)
        self._overflow.clear()
        for ns in sim.nodes:
            remaining = []
            pending = ns.pending_for(self.id)
            for slot in pending:
                # The decision precedes measurement generation in step S, so
                # S is the first post-decision measurement round. Exactly T
                # rounds are therefore S through S+T-1, inclusive.
                final_step = slot.step_started + slot.target_steps - 1
                if int(step) >= final_step:
                    t = self._finalize_slot(sim, ns, slot, step_finalized=step)
                    if t is not None:
                        ready.append(t)
                else:
                    remaining.append(slot)
            pending.clear()
            for s in remaining:
                pending.append(s)
        return ready

    def on_sim_end(self, sim):
        ready: list[Transition] = []
        for ns in sim.nodes:
            pending = ns.pending_for(self.id)
            while pending:
                slot = pending.popleft()
                t = self._finalize_slot(sim, ns, slot, step_finalized=-1)
                if t is not None:
                    ready.append(t)
        return ready


# --- v4 Oracle --------------------------------------------------------------

class OracleMode(RewardMode):
    """
    Unrealistic benchmark. Reward is computed on a fresh per-step per-zone
    set of ground-truth ray-traced pairs supplied by the simulation in
    `sim.oracle_sets[az] = (X_norm, y_dbm)`.
    """

    id = "v4"
    name = "Reference"

    def on_encounter(self, sim, step, ns_i, ns_j, az, action, state, next_state, done, j_view=None):
        X, y = sim.oracle_sets.get(az, (np.zeros((0, 4), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)))
        if X.shape[0] == 0:
            # Empty oracle set -> skip; do not feed an empty signal to the agent.
            if action == 1:
                sim.perform_merge(ns_i, ns_j, self.id, j_view=j_view)
                ns_i.variants[self.id].t_wait = 0
            return None
        rmse_before = sim.eval_rmse(ns_i, self.id, X, y)
        if action == 1:
            sim.perform_merge(ns_i, ns_j, self.id, j_view=j_view)
            ns_i.variants[self.id].t_wait = 0
        rmse_after = sim.eval_rmse(ns_i, self.id, X, y)
        ns_i.variants[self.id].last_rmse = float(rmse_after)
        ns_i.variants[self.id].last_rmse_available = True
        reward = float(rmse_before - rmse_after) - self.beta * float(action)
        return (state, int(action), float(reward), next_state, bool(done), sim.node_idx(ns_i))


# --- Factory ----------------------------------------------------------------

_FWB_RE = re.compile(r"^t([0-9]+)_b([-+eE0-9.]+)$")


def make_reward_mode(mode_id: str, cfg: ExperimentConfig) -> RewardMode:
    reward_id, _policy = split_mode_policy(mode_id)
    reward_id, spike_profile = split_spike_profile(reward_id)
    bm = _FWB_RE.fullmatch(reward_id.strip().lower())
    if bm:
        tw = int(bm.group(1))
        bv = float(bm.group(2))
        if tw not in WINDOW_T_VALUES:
            raise ValueError(
                f"Unsupported future-window T={tw} in {mode_id!r}; expected one of {WINDOW_T_VALUES}"
            )
        return FutureWindowMode(cfg, mode_id, tw, reward_beta=bv, spike_profile=spike_profile)
    if spike_profile is not None:
        raise ValueError(
            f"Spike profile suffix in {mode_id!r} is only supported for "
            "compound future-window beta ids like t2_b1_spmild"
        )
    if reward_id in WINDOW_T_BY_MODE:
        return FutureWindowMode(cfg, mode_id, WINDOW_T_BY_MODE[reward_id])
    if reward_id == "v4":
        return OracleMode(cfg)
    raise ValueError(f"Unknown reward mode {mode_id!r}")
