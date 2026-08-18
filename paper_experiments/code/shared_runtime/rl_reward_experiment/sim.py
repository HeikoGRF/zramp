"""
Simulation orchestrator.

Runs one end-to-end experiment that trains independent RL policies
in parallel (one per reward mode). Mobility, ray tracing, and local model
training are shared so every encounter produces one `(s, a, r, s')` per
active mode, which makes the decision-overlap analysis straightforward.

The simulation depends only on the following parent modules:

- `model.TinyRSSIPredictor`
- `build_map.Complex100mMap`
- `node.Node`
- `sim_seeding.set_simulation_seed`
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import _parent_path  # noqa: F401  (ensures parent dir on sys.path)
from .config import ExperimentConfig, split_mode_policy
from .mobility import (
    collides_with_walls,
    get_zone_center_feature,
    group_pairs_by_tx,
    move_annulus_jump,
    sample_oracle_pairs,
    zone_bounds,
    zone_of,
)
from .measurement import RayTracer
from .node_state import NodeState, VariantState, bound_raw_samples, saturate_n_samples
from .reward_modes import make_reward_mode, RewardMode
from .rl_agent import DQNAgent

# Parent-package (fixed) modules:
from build_map import Complex100mMap  # noqa: E402
from model import TinyRSSIPredictor, make_rssi_predictor  # noqa: E402
from node import Node  # noqa: E402
from sim_seeding import set_simulation_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _merge_coefficients(exp_puller: float, exp_provider: float) -> tuple[float, float]:
    exp_puller = float(max(exp_puller, 0.0))
    exp_provider = float(max(exp_provider, 0.0))
    total = exp_puller + exp_provider
    if total <= 0.0:
        return 0.5, 0.5
    return exp_puller / total, exp_provider / total


def _prediction_fidelity_statistics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    """Return individual-model error statistics.

    ``pooled_rmse`` treats every model/pair prediction as one served query.
    ``mean_model_rmse`` averages the independently calculated model RMSEs.
    """

    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != truth.size:
        raise ValueError("fidelity predictions must have shape [models, pairs]")
    squared = (matrix - truth.reshape(1, -1)) ** 2
    model_rmse = np.sqrt(np.mean(squared, axis=1))
    return {
        "individual_sum_sq": float(np.sum(squared)),
        "individual_count": int(squared.size),
        "pooled_rmse": float(np.sqrt(np.mean(squared))),
        "model_rmse": model_rmse,
        "mean_model_rmse": float(np.mean(model_rmse)),
    }


def _censored_fidelity_statistics(
    predictions_rssi: np.ndarray,
    targets_rssi: np.ndarray,
    reachable_threshold_rssi: float,
) -> dict[str, float | int]:
    """Score received links exactly and unavailable links as censored labels.

    An unavailable link only establishes that RSSI is below the decoding
    threshold. Its error is therefore zero for any prediction at or below the
    threshold and grows quadratically only when a model claims reachability.
    """

    matrix = np.asarray(predictions_rssi, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    truth = np.asarray(targets_rssi, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != truth.size:
        raise ValueError("fidelity predictions must have shape [models, pairs]")

    reachable = truth >= float(reachable_threshold_rssi)
    unavailable = ~reachable
    reachable_sq = (
        np.square(matrix[:, reachable] - truth[reachable][None, :])
        if np.any(reachable)
        else np.empty((matrix.shape[0], 0), dtype=np.float64)
    )
    unavailable_excess = (
        np.maximum(
            matrix[:, unavailable] - float(reachable_threshold_rssi), 0.0
        )
        if np.any(unavailable)
        else np.empty((matrix.shape[0], 0), dtype=np.float64)
    )
    unavailable_sq = np.square(unavailable_excess)
    total_sum_sq = float(np.sum(reachable_sq) + np.sum(unavailable_sq))
    total_count = int(matrix.size)
    unavailable_count = int(unavailable_sq.size)

    def _rmse(values: np.ndarray) -> float:
        return (
            float(np.sqrt(np.mean(values)))
            if int(values.size) > 0
            else float("nan")
        )

    false_reachable_count = int(np.count_nonzero(unavailable_excess > 0.0))
    return {
        "censored_sum_sq": total_sum_sq,
        "censored_count": total_count,
        "censored_rmse": (
            float(np.sqrt(total_sum_sq / total_count))
            if total_count > 0
            else float("nan")
        ),
        "reachable_sum_sq": float(np.sum(reachable_sq)),
        "reachable_count": int(reachable_sq.size),
        "reachable_rmse": _rmse(reachable_sq),
        "unavailable_censored_sum_sq": float(np.sum(unavailable_sq)),
        "unavailable_count": unavailable_count,
        "unavailable_censored_rmse": _rmse(unavailable_sq),
        "false_reachable_count": false_reachable_count,
        "false_reachable_rate": (
            float(false_reachable_count / unavailable_count)
            if unavailable_count > 0
            else float("nan")
        ),
    }


def _linear_layer_names(model: nn.Module) -> list[str]:
    cached = getattr(model, "_rre_linear_layer_names", None)
    if cached is None:
        cached = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        setattr(model, "_rre_linear_layer_names", cached)
    return list(cached)


def _stable_torch_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def _stable_int64_key(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _unit_signature(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    norms = weight.norm(dim=1)
    direction = weight / (norms[:, None] + 1e-12)
    coeff = torch.linspace(
        -1.0,
        1.0,
        int(weight.shape[1]),
        device=weight.device,
        dtype=weight.dtype,
    )
    scores = direction @ coeff
    scores = scores + 0.05 * torch.log1p(norms)
    if bias is not None:
        b_scale = bias.std(unbiased=False) + 1e-12
        scores = scores + 0.01 * (bias / b_scale)
    tie_break = torch.arange(
        int(weight.shape[0]),
        device=weight.device,
        dtype=weight.dtype,
    )
    return scores + 1e-8 * tie_break


def _ot_weighted_state_pull(
    model_puller: nn.Module,
    exp_puller: float,
    provider_state: dict[str, torch.Tensor],
    exp_provider: float,
) -> bool:
    """Layer-wise sliced-OT alignment followed by weighted averaging."""
    linear_names = _linear_layer_names(model_puller)
    if len(linear_names) < 2:
        return False
    target_state = model_puller.state_dict()
    if any(f"{name}.weight" not in provider_state for name in linear_names):
        return False

    a, b = _merge_coefficients(exp_puller, exp_provider)
    copied: set[str] = set()
    prev_perm: torch.Tensor | None = None
    with torch.no_grad():
        for idx, name in enumerate(linear_names):
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key not in target_state or w_key not in provider_state:
                return False
            target_w = target_state[w_key]
            source_w = provider_state[w_key].to(device=target_w.device, dtype=target_w.dtype)
            if target_w.shape != source_w.shape or target_w.ndim != 2:
                return False
            if prev_perm is not None:
                if int(source_w.shape[1]) != int(prev_perm.numel()):
                    return False
                source_w = source_w.index_select(1, prev_perm)

            target_b = target_state.get(b_key)
            raw_source_b = provider_state.get(b_key)
            if (target_b is None) != (raw_source_b is None):
                return False
            source_b = None
            if target_b is not None and raw_source_b is not None:
                source_b = raw_source_b.to(device=target_b.device, dtype=target_b.dtype)
                if target_b.shape != source_b.shape:
                    return False

            if idx < len(linear_names) - 1:
                perm = torch.empty(int(source_w.shape[0]), device=source_w.device, dtype=torch.long)
                target_order = torch.argsort(_unit_signature(target_w.detach(), target_b.detach() if target_b is not None else None))
                source_order = torch.argsort(_unit_signature(source_w, source_b))
                perm[target_order] = source_order
                source_w = source_w.index_select(0, perm)
                if source_b is not None:
                    source_b = source_b.index_select(0, perm)
                prev_perm = perm

            target_w.copy_(a * target_w + b * source_w)
            copied.add(w_key)
            if target_b is not None and source_b is not None:
                target_b.copy_(a * target_b + b * source_b)
                copied.add(b_key)

        for name, tensor in target_state.items():
            if name in copied or name not in provider_state:
                continue
            provider_tensor = provider_state[name].to(device=tensor.device, dtype=tensor.dtype)
            if torch.is_floating_point(tensor):
                tensor.copy_(a * tensor + b * provider_tensor)
            else:
                tensor.copy_(provider_tensor)
    return True

def weighted_state_pull(
    model_puller: nn.Module,
    exp_puller: float,
    provider_state: dict[str, torch.Tensor],
    exp_provider: float,
    *,
    merge_strategy: str = "average",
) -> None:
    """Experience-weighted pull from a provider state into the puller model."""
    a, b = _merge_coefficients(exp_puller, exp_provider)
    strategy = str(merge_strategy or "average").strip().lower().replace("_", "-")
    provider_for_merge = provider_state
    if strategy in {"ot", "sliced-ot", "transport"}:
        if _ot_weighted_state_pull(model_puller, exp_puller, provider_state, exp_provider):
            return

    with torch.no_grad():
        for name, tensor in model_puller.state_dict().items():
            if name not in provider_for_merge:
                continue
            provider_tensor = provider_for_merge[name].to(device=tensor.device, dtype=tensor.dtype)
            if torch.is_floating_point(tensor):
                tensor.copy_(a * tensor + b * provider_tensor)
            else:
                tensor.copy_(provider_tensor)


def unidirectional_pull(
    model_puller: nn.Module,
    exp_puller: float,
    model_provider: nn.Module,
    exp_provider: float,
    *,
    merge_strategy: str = "average",
) -> None:
    """Experience-weighted pull of the provider's weights into the puller."""
    weighted_state_pull(
        model_puller,
        exp_puller,
        model_provider.state_dict(),
        exp_provider,
        merge_strategy=merge_strategy,
    )

MODEL_SIGNATURE_FLOATS = 16
DECISION_METADATA_SCALAR_FLOATS = 2


def _model_signature(model: nn.Module, n_floats: int = MODEL_SIGNATURE_FLOATS) -> torch.Tensor:
    """Return the compact fixed-size signature exchanged before a pull decision.

    The signature is a deterministic signed projection of all floating-point
    parameters into ``n_floats`` float32 values. It is deliberately linear, so
    average-merged models can update their cached signature with the same
    coefficients without rescanning the full parameter vector after every pull.
    """
    n = max(1, int(n_floats))
    sig = torch.zeros(n, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            if not torch.is_floating_point(param):
                continue
            flat = param.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            count = int(flat.numel())
            if count <= 0:
                continue
            idx = torch.arange(count, dtype=torch.long) + int(offset)
            bins = torch.remainder(idx, n)
            sign = torch.where(
                torch.remainder(torch.div(idx, n, rounding_mode="floor"), 2) == 0,
                torch.ones(count, dtype=torch.float32),
                -torch.ones(count, dtype=torch.float32),
            )
            sig.index_add_(0, bins, flat * sign)
            offset += count
    if offset > 0:
        sig /= math.sqrt(max(1.0, float(offset) / float(n)))
    return sig


def _signature_diff_norm(sig_a: torch.Tensor, sig_b: torch.Tensor) -> float:
    if sig_a.numel() != sig_b.numel():
        n = min(int(sig_a.numel()), int(sig_b.numel()))
        if n <= 0:
            return 0.0
        sig_a = sig_a.reshape(-1)[:n]
        sig_b = sig_b.reshape(-1)[:n]
    return float(torch.norm(sig_a.to(torch.float32) - sig_b.to(torch.float32)).item())


def _pair_to_features(
    tx_xy: tuple[float, float],
    rx_xy: tuple[float, float],
    map_size: float,
    time_feature: float | None = None,
) -> list[float]:
    features = [
        float(tx_xy[0]) / map_size,
        float(tx_xy[1]) / map_size,
        float(rx_xy[0]) / map_size,
        float(rx_xy[1]) / map_size,
    ]
    if time_feature is not None:
        features.append(float(time_feature))
    return features


# ---------------------------------------------------------------------------
# Per-link peer view (used to make bidirectional gossip order-independent)
# ---------------------------------------------------------------------------

@dataclass
class _LinkPeerView:
    """Immutable, pre-link snapshot of one peer's variant state.

    A `_LinkPeerView` is created once per gossip link for the puller of the
    *first* leg, and consumed by the *second* leg so that:

    - raw/capped sample accumulation is symmetric (no double counting),
    - `w_diff` and `rei` see the same provider metadata both legs saw,

    independently of which physical leg is processed first.

    `model` references the simulation's single scratch `nn.Module`, loaded on
    demand with this peer's pre-link weights only if an accepted pull needs
    to merge full predictor weights.
    A `_LinkPeerView` MUST therefore not be retained across iterations — the
    next leg/mode pair will overwrite the scratch weights.
    """

    m_samples: int
    n_samples: int
    last_rmse: float
    last_rmse_available: bool
    model_signature: torch.Tensor
    model: nn.Module

    @property
    def experience(self) -> float:
        return float(self.n_samples)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

class Simulation:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        set_simulation_seed(cfg.seed)
        self._rng_py = random.Random(cfg.seed + 1_000)
        self._rng_np = np.random.default_rng(cfg.seed + 2_000)
        self._rng_grid = np.random.default_rng(cfg.seed + 9_999)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RRE] Device: {self.device}")

        # Build scene
        self.map_engine = Complex100mMap(frequency=cfg.freq_hz)
        self.scene = self.map_engine.build()
        self.walls = self.map_engine.walls

        # Template model -> shared conservative init for every node / mode.
        self.template = self._make_predictor().to(self.device)
        self._apply_predictor_prior(self.template)
        self.template_state = {k: v.detach().clone() for k, v in self.template.state_dict().items()}

        # Scratch model used to host pre-link snapshots of provider weights
        # during accepted pulls.
        self._scratch_model = self._make_predictor().to(self.device)
        self._scratch_model.load_state_dict(self.template_state)
        self._scratch_model.eval()

        # Ray tracer (Sionna)
        self.tracer = RayTracer(
            self.scene,
            num_rays=cfg.num_rays,
            max_depth=cfg.max_depth,
            tx_power_dbm=cfg.tx_power_dbm,
            rssi_min=cfg.rssi_min_dbm,
            rssi_max=cfg.rssi_max_dbm,
            tx_batch_size=cfg.trace_tx_batch_size,
        )

        # Nodes
        self.nodes: list[NodeState] = []
        self._init_nodes()

        # Reward modes + RL agents (one per active mode)
        self.reward_modes: dict[str, RewardMode] = {
            m: make_reward_mode(m, cfg) for m in cfg.active_modes
        }
        # One RNG stream per mode so parallel heads (e.g. several β) do not
        # consume each other's randomness during action sampling / replay.
        self.agents: dict[str, DQNAgent] = {}
        for m in cfg.active_modes:
            mode_offset = int(zlib.crc32(str(m).encode("utf-8")) % 10_000_000)
            agent_rng_seed = int(cfg.seed) + 501_173 + mode_offset
            _reward_id, policy_suffix = split_mode_policy(m)
            action_policy = policy_suffix or str(cfg.rl_action_policy)
            self.agents[m] = DQNAgent(
                device=self.device,
                gamma=cfg.gamma,
                lr=cfg.rl_lr,
                batch_size=cfg.rl_batch_size,
                tau=cfg.rl_target_tau,
                capacity=cfg.replay_capacity,
                epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end,
                epsilon_decay_steps=cfg.epsilon_decay_steps,
                rng_seed=agent_rng_seed,
                action_policy=action_policy,
            )

        # Per-step oracle eval sets keyed by zone (populated when v4 is active).
        self.oracle_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Rolling held-out fidelity grid; rebuilt periodically during run.
        self.fidelity_grid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._build_fidelity_grid(n_pairs=cfg.fidelity_grid_per_zone)
        self.final_fidelity_snapshot: dict[str, float | int] | None = None

        # Logs
        self.fidelity_history: list[dict] = []
        self.decision_log: list[dict] = []
        self.merge_eval_rows: list[dict] = []
        self._next_enc_id = 0

        # Propagation-loss predictor regression loss.
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------ init

    def _init_nodes(self) -> None:
        cfg = self.cfg
        for _ in range(cfg.num_nodes):
            while True:
                x = self._rng_py.uniform(cfg.xy_margin, cfg.map_size - cfg.xy_margin)
                y = self._rng_py.uniform(cfg.xy_margin, cfg.map_size - cfg.xy_margin)
                if not collides_with_walls(x, y, self.walls):
                    break
            raw = Node(x=x, y=y)
            variants = {}
            for m in cfg.active_modes:
                mdl = self._make_predictor().to(self.device)
                mdl.load_state_dict(self.template_state)
                opt = optim.Adam(mdl.parameters(), lr=cfg.local_lr)
                variant = VariantState(model=mdl, opt=opt)
                self._refresh_variant_signature(variant)
                variants[m] = variant
            ns = NodeState(
                node=raw,
                current_az=zone_of(x, y, cfg.map_size, cfg.num_zones),
                variants=variants,
            )
            self.nodes.append(ns)

    def _fidelity_sampling_walls(self):
        walls = list(self.walls)
        for obs in getattr(self.map_engine, "dynamic_obstacles", ()) or ():
            walls.append(obs.as_wall())
        return walls

    def _build_fidelity_grid(
        self,
        n_pairs: int | None = None,
        zones: set[int] | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Ray-trace a held-out grid per zone used for fidelity reporting."""
        cfg = self.cfg
        n_pairs = int(cfg.fidelity_grid_per_zone if n_pairs is None else n_pairs)
        zone_ids = range(cfg.num_zones) if zones is None else sorted({int(z) for z in zones})
        fidelity_walls = self._fidelity_sampling_walls()
        for az in zone_ids:
            pairs = sample_oracle_pairs(
                zone=az,
                walls=fidelity_walls,
                map_size=cfg.map_size,
                margin=cfg.xy_margin,
                n_tx=cfg.fidelity_grid_n_tx,
                n_pairs=n_pairs,
                rng=self._rng_grid,
                num_zones=cfg.num_zones,
            )
            groups = group_pairs_by_tx(pairs)
            rssi_groups = self.tracer.measure_pairs(groups)
            X = []
            y = []
            for (tx, rxs), rssi_list in zip(groups, rssi_groups):
                for rx, val in zip(rxs, rssi_list):
                    X.append(self._pair_model_features(tx, rx))
                    y.append(val)
            self.fidelity_grid[az] = (
                np.asarray(X, dtype=np.float32),
                np.asarray(y, dtype=np.float32).reshape(-1, 1),
            )
            if cfg.verbose:
                print(f"[RRE] Fidelity grid zone {az}: {len(X)} pairs")

    def _compute_fidelity_row(self, step: int) -> dict[str, float | int]:
        """Evaluate individual maps and expose conservative-prior failure modes."""

        row: dict[str, float | int] = {"step": int(step)}
        cfg = getattr(self, "cfg", None)
        prior_loss = self._predictor_prior_loss_db() if cfg is not None else None
        prior_rssi = (
            float(self._loss_to_rssi_dbm(np.asarray([prior_loss]))[0])
            if prior_loss is not None
            else float("nan")
        )
        reachable_threshold = (
            self._rx_power_threshold_dbm()
            if cfg is not None
            else float("-inf")
        )
        floor_threshold = (
            float(cfg.rssi_min_dbm) + 1.0e-4
            if cfg is not None
            else float("-inf")
        )

        for mode_id in self.reward_modes:
            individual_sum_sq = 0.0
            individual_count = 0
            reachable_sum_sq = 0.0
            reachable_count = 0
            floor_sum_sq = 0.0
            floor_count = 0
            prior_sum_sq = 0.0
            prior_count = 0

            censored_sum_sq = 0.0
            censored_count = 0
            unavailable_censored_sum_sq = 0.0
            unavailable_count = 0
            false_reachable_count = 0
            prior_censored_sum_sq = 0.0
            prior_censored_count = 0
            prior_false_reachable_count = 0
            prior_unavailable_count = 0
            prior_reachable_sum_sq = 0.0
            prior_reachable_count = 0
            prior_floor_sum_sq = 0.0
            prior_floor_count = 0
            prediction_shift_sum = 0.0
            prediction_shift_count = 0
            supported_sum_sq = 0.0
            supported_prior_sum_sq = 0.0
            supported_count = 0
            support_query_count = 0
            support_query_total = 0
            support_value_sum = 0.0
            model_rmse_values: list[float] = []
            active_model_count = 0
            active_mask = getattr(self, "_current_node_active", None)

            for az, (X, y) in self.fidelity_grid.items():
                if X.shape[0] == 0:
                    row[f"{mode_id}_z{az}"] = float("nan")
                    continue
                members = [
                    ns
                    for node_index, ns in enumerate(self.nodes)
                    if ns.current_az == az
                    and (
                        active_mask is None
                        or bool(active_mask[node_index])
                    )
                ]
                row[f"{mode_id}_active_models_z{az}"] = int(len(members))
                active_model_count += len(members)
                if not members:
                    row[f"{mode_id}_z{az}"] = float("nan")
                    continue

                preds: list[np.ndarray] = []
                supports: list[np.ndarray] = []
                all_have_support = True
                for ns in members:
                    p, support = self._predict_variant_fidelity(
                        ns, mode_id, X
                    )
                    if support is None:
                        all_have_support = False
                    else:
                        supports.append(support)
                    preds.append(p)

                pred_matrix = np.stack(preds)
                truth = np.asarray(y, dtype=np.float32).reshape(-1)
                squared_error = np.square(pred_matrix - truth[None, :])
                stats = _prediction_fidelity_statistics(pred_matrix, y)
                individual = float(stats["pooled_rmse"])
                row[f"{mode_id}_z{az}"] = individual
                row[f"{mode_id}_individual_z{az}"] = individual

                censored = _censored_fidelity_statistics(
                    pred_matrix, truth, reachable_threshold
                )
                row[f"{mode_id}_censored_rmse_z{az}"] = float(
                    censored["censored_rmse"]
                )
                row[
                    f"{mode_id}_unavailable_censored_rmse_z{az}"
                ] = float(censored["unavailable_censored_rmse"])
                row[f"{mode_id}_false_reachable_rate_z{az}"] = float(
                    censored["false_reachable_rate"]
                )
                censored_sum_sq += float(censored["censored_sum_sq"])
                censored_count += int(censored["censored_count"])
                unavailable_censored_sum_sq += float(
                    censored["unavailable_censored_sum_sq"]
                )
                unavailable_count += int(censored["unavailable_count"])
                false_reachable_count += int(
                    censored["false_reachable_count"]
                )

                if math.isfinite(prior_rssi):
                    prior_predictions = np.full_like(
                        pred_matrix, prior_rssi, dtype=np.float64
                    )
                    prior_censored = _censored_fidelity_statistics(
                        prior_predictions, truth, reachable_threshold
                    )
                    prior_censored_sum_sq += float(
                        prior_censored["censored_sum_sq"]
                    )
                    prior_censored_count += int(
                        prior_censored["censored_count"]
                    )
                    prior_false_reachable_count += int(
                        prior_censored["false_reachable_count"]
                    )
                    prior_unavailable_count += int(
                        prior_censored["unavailable_count"]
                    )
                row[f"{mode_id}_mean_model_rmse_z{az}"] = float(
                    stats["mean_model_rmse"]
                )
                individual_sum_sq += float(stats["individual_sum_sq"])
                individual_count += int(stats["individual_count"])
                model_rmse_values.extend(
                    float(value) for value in np.asarray(stats["model_rmse"])
                )

                model_count = int(pred_matrix.shape[0])
                reachable = truth >= reachable_threshold
                floor = truth <= floor_threshold
                if np.any(reachable):
                    reachable_sum_sq += float(
                        np.sum(squared_error[:, reachable], dtype=np.float64)
                    )
                    reachable_count += model_count * int(np.count_nonzero(reachable))
                if np.any(floor):
                    floor_sum_sq += float(
                        np.sum(squared_error[:, floor], dtype=np.float64)
                    )
                    floor_count += model_count * int(np.count_nonzero(floor))

                if math.isfinite(prior_rssi):
                    prior_squared = np.square(prior_rssi - truth)
                    prior_sum_sq += model_count * float(
                        np.sum(prior_squared, dtype=np.float64)
                    )
                    prior_count += model_count * int(truth.size)
                    prediction_shift_sum += float(
                        np.sum(np.abs(pred_matrix - prior_rssi), dtype=np.float64)
                    )
                    prediction_shift_count += int(pred_matrix.size)
                    if np.any(reachable):
                        prior_reachable_sum_sq += model_count * float(
                            np.sum(prior_squared[reachable], dtype=np.float64)
                        )
                        prior_reachable_count += (
                            model_count * int(np.count_nonzero(reachable))
                        )
                    if np.any(floor):
                        prior_floor_sum_sq += model_count * float(
                            np.sum(prior_squared[floor], dtype=np.float64)
                        )
                        prior_floor_count += model_count * int(np.count_nonzero(floor))

                if all_have_support and len(supports) == model_count:
                    support_matrix = np.stack(supports)
                    support_mask = support_matrix > 1.0e-8
                    support_query_count += int(np.count_nonzero(support_mask))
                    support_query_total += int(support_mask.size)
                    support_value_sum += float(
                        np.sum(support_matrix, dtype=np.float64)
                    )
                    if np.any(support_mask):
                        supported_sum_sq += float(
                            np.sum(squared_error[support_mask], dtype=np.float64)
                        )
                        supported_count += int(np.count_nonzero(support_mask))
                        if math.isfinite(prior_rssi):
                            tiled_prior_squared = np.broadcast_to(
                                np.square(prior_rssi - truth)[None, :],
                                pred_matrix.shape,
                            )
                            supported_prior_sum_sq += float(
                                np.sum(
                                    tiled_prior_squared[support_mask],
                                    dtype=np.float64,
                                )
                            )

            def _rmse(sum_sq: float, count: int) -> float:
                return (
                    float(np.sqrt(sum_sq / count))
                    if int(count) > 0
                    else float("nan")
                )

            pooled_total = _rmse(individual_sum_sq, individual_count)
            censored_rmse = _rmse(censored_sum_sq, censored_count)
            unavailable_censored_rmse = _rmse(
                unavailable_censored_sum_sq, unavailable_count
            )
            false_reachable_rate = (
                float(false_reachable_count / unavailable_count)
                if unavailable_count > 0
                else float("nan")
            )
            prior_censored_rmse = _rmse(
                prior_censored_sum_sq, prior_censored_count
            )
            prior_false_reachable_rate = (
                float(prior_false_reachable_count / prior_unavailable_count)
                if prior_unavailable_count > 0
                else float("nan")
            )
            reachable_rmse = _rmse(reachable_sum_sq, reachable_count)
            floor_rmse = _rmse(floor_sum_sq, floor_count)
            prior_rmse = _rmse(prior_sum_sq, prior_count)
            prior_reachable_rmse = _rmse(
                prior_reachable_sum_sq, prior_reachable_count
            )
            prior_floor_rmse = _rmse(prior_floor_sum_sq, prior_floor_count)
            supported_rmse = _rmse(supported_sum_sq, supported_count)
            supported_prior_rmse = _rmse(
                supported_prior_sum_sq, supported_count
            )
            balanced_rmse = (
                float(np.sqrt(0.5 * (reachable_rmse**2 + floor_rmse**2)))
                if math.isfinite(reachable_rmse) and math.isfinite(floor_rmse)
                else float("nan")
            )

            row[f"{mode_id}_total"] = pooled_total
            row[f"{mode_id}_individual_total"] = pooled_total
            row[f"{mode_id}_censored_rmse_total"] = censored_rmse
            row[f"{mode_id}_unavailable_censored_rmse_total"] = (
                unavailable_censored_rmse
            )
            row[f"{mode_id}_false_reachable_rate_total"] = (
                false_reachable_rate
            )
            row[f"{mode_id}_reachable_count_total"] = int(reachable_count)
            row[f"{mode_id}_unavailable_count_total"] = int(unavailable_count)
            row[f"{mode_id}_prior_censored_rmse_total"] = prior_censored_rmse
            row[f"{mode_id}_prior_false_reachable_rate_total"] = (
                prior_false_reachable_rate
            )
            row[f"{mode_id}_censored_gain_vs_prior_total"] = (
                prior_censored_rmse - censored_rmse
            )
            row[f"{mode_id}_reachable_rmse_total"] = reachable_rmse
            row[f"{mode_id}_floor_rmse_total"] = floor_rmse
            row[f"{mode_id}_balanced_rmse_total"] = balanced_rmse
            row[f"{mode_id}_prior_rmse_total"] = prior_rmse
            row[f"{mode_id}_prior_reachable_rmse_total"] = prior_reachable_rmse
            row[f"{mode_id}_prior_floor_rmse_total"] = prior_floor_rmse
            row[f"{mode_id}_rmse_gain_vs_prior_total"] = prior_rmse - pooled_total
            row[f"{mode_id}_reachable_gain_vs_prior_total"] = (
                prior_reachable_rmse - reachable_rmse
            )
            row[f"{mode_id}_supported_fraction_total"] = (
                float(support_query_count / support_query_total)
                if support_query_total > 0
                else float("nan")
            )
            row[f"{mode_id}_supported_rmse_total"] = supported_rmse
            row[f"{mode_id}_supported_prior_rmse_total"] = supported_prior_rmse
            row[f"{mode_id}_supported_gain_vs_prior_total"] = (
                supported_prior_rmse - supported_rmse
            )
            row[f"{mode_id}_mean_support_total"] = (
                float(support_value_sum / support_query_total)
                if support_query_total > 0
                else float("nan")
            )
            row[f"{mode_id}_mean_abs_prediction_shift_from_prior_db_total"] = (
                float(prediction_shift_sum / prediction_shift_count)
                if prediction_shift_count > 0
                else float("nan")
            )
            row[f"{mode_id}_mean_model_rmse_total"] = (
                float(np.mean(model_rmse_values))
                if model_rmse_values
                else float("nan")
            )
            row[f"{mode_id}_median_model_rmse_total"] = (
                float(np.median(model_rmse_values))
                if model_rmse_values
                else float("nan")
            )
            row[f"{mode_id}_p10_model_rmse_total"] = (
                float(np.percentile(model_rmse_values, 10.0))
                if model_rmse_values
                else float("nan")
            )
            row[f"{mode_id}_p90_model_rmse_total"] = (
                float(np.percentile(model_rmse_values, 90.0))
                if model_rmse_values
                else float("nan")
            )
            row[f"{mode_id}_active_models_total"] = int(active_model_count)
        return row

    def _predict_variant_fidelity(
        self,
        ns: NodeState,
        mode_id: str,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return one vehicle's operational map and optional support values."""

        model = ns.variants[mode_id].model
        model.eval()
        adapted = self._adapt_predictor_features(X)
        xt = torch.tensor(
            adapted, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            prediction = self._denorm_dbm(
                model(xt).cpu().numpy().flatten()
            )
            support_at = getattr(model, "support_at", None)
            support = (
                support_at(xt).cpu().numpy().flatten()
                if callable(support_at)
                else None
            )
        return prediction, support

    # --------------------------------------------------------------- helpers

    def node_idx(self, ns: NodeState) -> int:
        return self.nodes.index(ns) if ns in self.nodes else -1

    def _rx_power_threshold_dbm(self) -> float:
        return float(self.cfg.noise_floor_dbm) + float(self.cfg.snr_min_db)

    def _snr_from_rx_power_dbm(self, rx_power_dbm: float) -> float:
        return float(rx_power_dbm) - float(self.cfg.noise_floor_dbm)

    def _predictor_prior_loss_db(self) -> float | None:
        prior = str(getattr(self.cfg, "predictor_prior", "none")).strip().lower().replace("_", "-")
        if prior == "none":
            return None
        if prior == "max-loss":
            return float(self._loss_max_db())
        if prior == "snr-threshold":
            threshold_loss = float(self.cfg.tx_power_dbm) - float(self._rx_power_threshold_dbm())
            return float(min(max(threshold_loss, self._loss_min_db()), self._loss_max_db()))
        raise ValueError(f"Unknown predictor_prior={prior!r}")

    def _predictor_prior_normalized_loss(self) -> float | None:
        loss_db = self._predictor_prior_loss_db()
        if loss_db is None:
            return None
        value = self._norm_loss_db(np.asarray([loss_db], dtype=np.float32)).reshape(-1)[0]
        return float(np.clip(value, 0.0, 1.0))

    def _apply_predictor_prior(self, model: nn.Module) -> None:
        prior = self._predictor_prior_normalized_loss()
        if prior is None:
            return
        set_prior = getattr(model, "set_normalized_prior", None)
        if callable(set_prior):
            set_prior(float(prior))
            return
        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("predictor_prior requires at least one Linear layer")
        out = linear_layers[-1]
        if int(out.out_features) != 1 or out.bias is None:
            raise ValueError("predictor_prior requires a final scalar Linear layer with bias")
        with torch.no_grad():
            out.weight.zero_()
            out.bias.fill_(float(prior))

    def _predictor_input_dim(self) -> int:
        # Raw samples retain one scalar global time; Fourier expansion happens
        # inside the predictor so its frequencies train and aggregate normally.
        return 5 if bool(getattr(self.cfg, "predictor_include_time", False)) else 4

    def _make_predictor(self) -> nn.Module:
        cfg = self.cfg
        return make_rssi_predictor(
            cfg.rssi_model,
            input_dim=self._predictor_input_dim(),
            include_time=bool(getattr(cfg, "predictor_include_time", False)),
            num_time_frequencies=int(getattr(cfg, "predictor_time_num_frequencies", 8)),
            min_time_period=float(getattr(cfg, "predictor_time_min_period", 2.0)),
            max_time_period=float(getattr(cfg, "predictor_time_max_period", 1000.0)),
            time_unit=float(getattr(cfg, "predictor_time_unit", 1.0)),
            learned_time_scale=float(
                getattr(cfg, "predictor_learned_time_scale", 1000.0)
            ),
            spatial_grid_points=int(
                getattr(cfg, "local_support_spatial_grid_points", 9)
            ),
            support_prior_strength=float(
                getattr(cfg, "local_support_prior_strength", 0.0)
            ),
            mergeable_basis_dim=int(
                getattr(cfg, "mergeable_basis_dim", 192)
            ),
            mergeable_ridge=float(
                getattr(cfg, "mergeable_ridge", 1.0)
            ),
        )

    def _predictor_time_feature(self, step: int | float | None = None) -> float | None:
        if not bool(getattr(self.cfg, "predictor_include_time", False)):
            return None
        if step is None:
            step = getattr(self, "_current_sumo_step", 0)
        duration = float(getattr(self.cfg, "predictor_time_step_duration", 1.0))
        if duration <= 0.0:
            raise ValueError("predictor_time_step_duration must be positive")
        return float(step) * duration

    def _pair_model_features(
        self,
        tx_xy: tuple[float, float],
        rx_xy: tuple[float, float],
        *,
        step: int | float | None = None,
        zone: int | None = None,
    ) -> list[float]:
        if zone is None:
            zone = zone_of(
                float(tx_xy[0]),
                float(tx_xy[1]),
                float(self.cfg.map_size),
                int(self.cfg.num_zones),
            )
        global_row = np.asarray(
            [
                _pair_to_features(
                    tx_xy,
                    rx_xy,
                    self.cfg.map_size,
                    self._predictor_time_feature(step),
                )
            ],
            dtype=np.float32,
        )
        return self._adapt_predictor_features(
            global_row, step=step, zone=int(zone)
        )[0].tolist()

    def _adapt_predictor_features(
        self,
        X: np.ndarray,
        *,
        step: int | float | None = None,
        zone: int | None = None,
    ) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            return arr
        target_dim = self._predictor_input_dim()
        if arr.shape[1] == target_dim:
            adapted = arr
        elif target_dim == 5 and arr.shape[1] == 4:
            t = self._predictor_time_feature(step)
            if t is None:
                t = 0.0
            col = np.full((arr.shape[0], 1), float(t), dtype=np.float32)
            adapted = np.concatenate([arr, col], axis=1)
        elif target_dim == 4 and arr.shape[1] == 5:
            adapted = arr[:, :4].astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Predictor feature dimension mismatch: got {arr.shape[1]}, "
                f"expected {target_dim}"
            )

        if not bool(
            getattr(self.cfg, "predictor_zone_local_coordinates", False)
        ):
            return adapted
        if zone is None:
            raise ValueError("zone is required for AZ-local predictor coordinates")
        side = int(self.cfg.zones_per_side)
        zone_id = int(zone)
        if zone_id < 0 or zone_id >= int(self.cfg.num_zones):
            raise ValueError(f"invalid predictor zone {zone_id}")
        col = zone_id % side
        row = zone_id // side
        localized = adapted.copy()
        localized[:, (0, 2)] = localized[:, (0, 2)] * side - float(col)
        localized[:, (1, 3)] = localized[:, (1, 3)] * side - float(row)
        localized[:, :4] = np.clip(localized[:, :4], 0.0, 1.0)
        return localized

    def _sample_recency_weights(
        self,
        sample_steps: np.ndarray | list[int] | None,
        *,
        current_step: int | float | None = None,
    ) -> np.ndarray | None:
        if str(
            getattr(self.cfg, "local_sample_weighting", "uniform")
        ) != "exponential-recency":
            return None
        if sample_steps is None:
            return None
        steps = np.asarray(sample_steps, dtype=np.float32).reshape(-1)
        if steps.size == 0:
            return None
        if current_step is None:
            current_step = getattr(self, "_current_sumo_step", 0)
        half_life = float(getattr(self.cfg, "local_sample_recency_half_life_steps", 50.0))
        if half_life <= 0.0:
            raise ValueError("local_sample_recency_half_life_steps must be positive")
        age = np.maximum(0.0, float(current_step) - steps)
        weights = np.exp(-np.log(2.0) * age / half_life).astype(np.float32)
        mean = float(np.mean(weights))
        if mean > 0.0:
            weights = weights / mean
        return weights.astype(np.float32, copy=False)


    def _spatial_balance_weights(
        self, X: np.ndarray, *, zone: int | None = None
    ) -> np.ndarray | None:
        """Give each occupied AZ-local TX/RX coordinate cell equal weight."""

        rows = np.asarray(X, dtype=np.float32)
        if rows.ndim != 2 or int(rows.shape[0]) == 0:
            return None
        if int(rows.shape[1]) < 4:
            raise ValueError(
                "spatial balancing requires four endpoint coordinates"
            )
        coordinates = rows[:, :4].copy()
        if (
            not bool(
                getattr(
                    self.cfg, "predictor_zone_local_coordinates", False
                )
            )
            and zone is not None
        ):
            side = int(self.cfg.zones_per_side)
            zone_id = int(zone)
            col, row = zone_id % side, zone_id // side
            coordinates[:, (0, 2)] = (
                coordinates[:, (0, 2)] * side - float(col)
            )
            coordinates[:, (1, 3)] = (
                coordinates[:, (1, 3)] * side - float(row)
            )
        coordinates = np.clip(coordinates, 0.0, 1.0)
        bins = int(getattr(self.cfg, "local_spatial_balance_bins", 4))
        cells = np.minimum(
            np.floor(coordinates * bins).astype(np.int64), bins - 1
        )
        ids = np.ravel_multi_index(
            cells.T, (bins, bins, bins, bins)
        )
        _unique, inverse, counts = np.unique(
            ids, return_inverse=True, return_counts=True
        )
        weights = 1.0 / counts[inverse].astype(np.float32)
        return (weights / float(np.mean(weights))).astype(
            np.float32, copy=False
        )

    def _configured_sample_weights(
        self,
        X: np.ndarray,
        supplied_weights: np.ndarray | None,
        *,
        zone: int | None = None,
    ) -> np.ndarray | None:
        if (
            str(
                getattr(
                    self.cfg, "local_sample_weighting", "uniform"
                )
            )
            == "spatial-balanced"
        ):
            return self._spatial_balance_weights(X, zone=zone)
        return supplied_weights

    def _initialization_anchor_loss(
        self, model: nn.Module
    ) -> torch.Tensor:
        named = list(model.named_parameters())
        if not named:
            return torch.zeros((), device=self.device)
        total = named[0][1].new_zeros(())
        for name, parameter in named:
            reference = self.template_state.get(name)
            if (
                reference is None
                or tuple(reference.shape) != tuple(parameter.shape)
            ):
                continue
            ref = reference.to(
                device=parameter.device, dtype=parameter.dtype
            )
            total = total + torch.sum((parameter - ref) ** 2)
        return total

    def _training_loss(
        self,
        model: nn.Module,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        loss = self._weighted_regression_loss(
            prediction, target, weights
        )
        strength = float(
            getattr(
                self.cfg,
                "local_initialization_anchor_strength",
                0.0,
            )
        )
        if strength > 0.0:
            loss = loss + strength * self._initialization_anchor_loss(
                model
            )
        return loss

    def _additional_predictor_training_loss(
        self, model: nn.Module
    ) -> torch.Tensor:
        """Optional local regularizer supplied by specialized simulations."""

        parameter = next(model.parameters())
        return parameter.sum() * 0.0

    def _fit_predictor(
        self,
        model: nn.Module,
        opt: optim.Optimizer,
        X: np.ndarray,
        y_scaled: np.ndarray,
        sample_weights: np.ndarray | None,
        *,
        device: torch.device,
        seed_parts: tuple[object, ...],
        n_new_samples: int = 0,
    ) -> int:
        """Fit with either legacy full epochs or a fixed minibatch budget."""

        n_samples = int(X.shape[0])
        if n_samples == 0:
            return 0
        xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(
            y_scaled, dtype=torch.float32
        ).reshape(n_samples, -1)
        wt = (
            None
            if sample_weights is None
            else torch.as_tensor(
                sample_weights, dtype=torch.float32
            ).reshape(n_samples, 1)
        )
        generator = torch.Generator()
        generator.manual_seed(
            _stable_torch_seed(
                self.cfg.seed, *seed_parts, n_samples
            )
        )
        batch_size = min(
            n_samples, int(self.cfg.local_batch_size)
        )
        budget = int(
            getattr(self.cfg, "local_batches_per_step", 0)
        )
        maximum_budget = int(
            getattr(self.cfg, "local_batches_per_step_max", 0)
        )
        maturity_rows = int(
            getattr(self.cfg, "local_batches_maturity_rows", 0)
        )
        if maximum_budget > 0 and maturity_rows > 0:
            maturity = min(
                1.0, float(n_samples) / float(maturity_rows)
            )
            budget += int(
                round(maturity * float(maximum_budget - budget))
            )
        model.train()
        updates = 0

        def update(
            indices: torch.Tensor,
            batch_weights: torch.Tensor | None,
        ) -> None:
            nonlocal updates
            bx = xt.index_select(0, indices).to(device)
            by = yt.index_select(0, indices).to(device)
            bw = (
                None
                if batch_weights is None
                else batch_weights.to(device)
            )
            opt.zero_grad(set_to_none=True)
            supervised_loss = getattr(model, "supervised_loss", None)
            if callable(supervised_loss):
                loss = supervised_loss(bx, by, bw)
                strength = float(
                    getattr(
                        self.cfg,
                        "local_initialization_anchor_strength",
                        0.0,
                    )
                )
                if strength > 0.0:
                    loss = loss + strength * self._initialization_anchor_loss(
                        model
                    )
            else:
                loss = self._training_loss(
                    model, model(bx), by, bw
                )
            loss = loss + self._additional_predictor_training_loss(model)
            loss.backward()
            opt.step()
            updates += 1

        if bool(
            getattr(self.cfg, "local_train_all_new_samples", False)
        ):
            new_count = min(n_samples, max(0, int(n_new_samples)))
            if new_count > 0:
                new_indices = torch.arange(
                    n_samples - new_count, n_samples
                )
                new_indices = new_indices[
                    torch.randperm(new_count, generator=generator)
                ]
                for start in range(0, new_count, batch_size):
                    indices = new_indices[start : start + batch_size]
                    selected_weights = (
                        None
                        if wt is None
                        else wt.index_select(0, indices)
                    )
                    update(indices, selected_weights)

        if budget > 0:
            for _ in range(budget):
                if n_samples <= batch_size:
                    indices = torch.arange(n_samples)
                    selected_weights = wt
                elif wt is not None:
                    probabilities = wt.reshape(-1).clamp_min(0.0)
                    probabilities = probabilities / probabilities.sum().clamp_min(
                        1.0e-12
                    )
                    indices = torch.multinomial(
                        probabilities,
                        batch_size,
                        replacement=True,
                        generator=generator,
                    )
                    # Weighted sampling already balances the expected batch;
                    # applying the weights again would square the correction.
                    selected_weights = None
                else:
                    indices = torch.randperm(
                        n_samples, generator=generator
                    )[:batch_size]
                    selected_weights = None
                update(indices, selected_weights)
        else:
            for _ in range(int(self.cfg.local_epochs)):
                permutation = torch.randperm(
                    n_samples, generator=generator
                )
                for start in range(
                    0, n_samples, batch_size
                ):
                    indices = permutation[
                        start : start + batch_size
                    ]
                    selected_weights = (
                        None
                        if wt is None
                        else wt.index_select(0, indices)
                    )
                    update(indices, selected_weights)

        classifier_budget = int(
            getattr(self.cfg, "local_classifier_batches_per_step", 0)
        )
        forward_components = getattr(model, "forward_components", None)
        boundary = float(getattr(model, "censoring_boundary", 1.0))
        labels_cpu = (
            yt.reshape(-1) < boundary - 1.0e-6
        ).to(dtype=torch.float32)
        feasible_indices = torch.nonzero(
            labels_cpu > 0.5, as_tuple=False
        ).reshape(-1)
        blocked_indices = torch.nonzero(
            labels_cpu <= 0.5, as_tuple=False
        ).reshape(-1)
        if (
            classifier_budget > 0
            and callable(forward_components)
            and int(feasible_indices.numel()) > 0
            and int(blocked_indices.numel()) > 0
        ):
            candidate_multiplier = int(
                getattr(
                    self.cfg,
                    "local_classifier_hard_candidate_multiplier",
                    4,
                )
            )

            def sample_source(
                source: torch.Tensor, count: int
            ) -> torch.Tensor:
                source_count = int(source.numel())
                if source_count <= 0 or count <= 0:
                    return torch.empty((0,), dtype=torch.long)
                if wt is None:
                    chosen = torch.randint(
                        source_count,
                        (int(count),),
                        generator=generator,
                    )
                else:
                    source_weights = wt.reshape(-1).index_select(
                        0, source
                    ).clamp_min(0.0)
                    if float(source_weights.sum().item()) <= 0.0:
                        source_weights = torch.ones_like(source_weights)
                    chosen = torch.multinomial(
                        source_weights,
                        int(count),
                        replacement=True,
                        generator=generator,
                    )
                return source.index_select(0, chosen)

            def balanced_hard_indices(
                source: torch.Tensor,
                count: int,
                *,
                feasible_label: bool,
            ) -> torch.Tensor:
                random_count = int(count) // 2
                hard_count = int(count) - random_count
                random_rows = sample_source(source, random_count)
                candidates = sample_source(
                    source,
                    max(hard_count, candidate_multiplier * hard_count),
                )
                with torch.no_grad():
                    logits, _conditional = forward_components(
                        xt.index_select(0, candidates).to(device)
                    )
                    probability = torch.sigmoid(logits).reshape(-1)
                    hardness = (
                        1.0 - probability
                        if feasible_label
                        else probability
                    )
                    hard_local = torch.topk(
                        hardness,
                        k=min(hard_count, int(hardness.numel())),
                    ).indices.cpu()
                hard_rows = candidates.index_select(0, hard_local)
                return torch.cat((random_rows, hard_rows), dim=0)

            feasible_count = max(1, batch_size // 2)
            blocked_count = max(1, batch_size - feasible_count)
            for _ in range(classifier_budget):
                indices = torch.cat(
                    (
                        balanced_hard_indices(
                            feasible_indices,
                            feasible_count,
                            feasible_label=True,
                        ),
                        balanced_hard_indices(
                            blocked_indices,
                            blocked_count,
                            feasible_label=False,
                        ),
                    ),
                    dim=0,
                )
                bx = xt.index_select(0, indices).to(device)
                labels = labels_cpu.index_select(0, indices).reshape(-1, 1).to(device)
                opt.zero_grad(set_to_none=True)
                logits, _conditional = forward_components(bx)
                classifier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels
                )
                classifier_loss.backward()
                opt.step()
                updates += 1
        return updates

    @staticmethod
    def _weighted_regression_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weights is None:
            return torch.nn.functional.mse_loss(pred, target)
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        weights = weights.to(device=pred.device, dtype=pred.dtype)
        per_sample = torch.nn.functional.mse_loss(pred, target, reduction="none")
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0e-8)

    def _refresh_variant_signature(self, variant) -> None:
        variant.model_signature = _model_signature(variant.model)

    def _variant_signature(self, variant) -> torch.Tensor:
        sig = getattr(variant, "model_signature", None)
        if not isinstance(sig, torch.Tensor) or int(sig.numel()) != MODEL_SIGNATURE_FLOATS:
            self._refresh_variant_signature(variant)
        return variant.model_signature.detach().to(dtype=torch.float32, device="cpu")

    def _refresh_node_signatures(self, ns: NodeState) -> None:
        for variant in ns.variants.values():
            self._refresh_variant_signature(variant)

    def _reset_node_for_zone_change(self, ns: NodeState, new_az: int) -> None:
        ns.current_az = new_az
        ns.reset_all(self.template_state, self.cfg.local_lr)
        self._refresh_node_signatures(ns)

    # ---------------------------------------------------------------- eval
    def _loss_min_db(self) -> float:
        return float(self.cfg.tx_power_dbm) - float(self.cfg.rssi_max_dbm)

    def _loss_max_db(self) -> float:
        return float(self.cfg.tx_power_dbm) - float(self.cfg.rssi_min_dbm)

    def _rssi_to_loss_db(self, y_rssi_dbm: np.ndarray) -> np.ndarray:
        return float(self.cfg.tx_power_dbm) - np.asarray(y_rssi_dbm, dtype=np.float32)

    def _loss_to_rssi_dbm(self, y_loss_db: np.ndarray) -> np.ndarray:
        return float(self.cfg.tx_power_dbm) - np.asarray(y_loss_db, dtype=np.float32)

    def _norm_loss_db(self, y_loss_db: np.ndarray) -> np.ndarray:
        lo = self._loss_min_db()
        hi = self._loss_max_db()
        return (np.asarray(y_loss_db, dtype=np.float32) - lo) / max(hi - lo, 1e-8)

    def _normalize_target_from_rssi(self, y_rssi_dbm: np.ndarray) -> np.ndarray:
        """Convert RSSI measurements to normalized propagation-loss targets."""
        return self._norm_loss_db(self._rssi_to_loss_db(y_rssi_dbm))

    def _denorm_loss_db(self, yp_norm: np.ndarray) -> np.ndarray:
        """De-normalise model output as propagation loss in dB."""
        lo = self._loss_min_db()
        hi = self._loss_max_db()
        yp = np.asarray(yp_norm, dtype=np.float32) * (hi - lo) + lo
        return np.clip(yp, lo, hi)

    def _denorm_dbm(self, yp_norm: np.ndarray) -> np.ndarray:
        """Return RSSI-equivalent predictions for legacy reports and helpers.

        The predictor is trained on propagation loss. Since transmit power is
        constant, converting the prediction back to RSSI preserves MSE/RMSE.
        """
        yp = self._loss_to_rssi_dbm(self._denorm_loss_db(yp_norm))
        return np.clip(yp, self.cfg.rssi_min_dbm, self.cfg.rssi_max_dbm)

    def eval_mse(
        self,
        ns: NodeState,
        mode: str,
        X: np.ndarray,
        y_dbm: np.ndarray,
    ) -> float:
        if X.shape[0] == 0:
            return 0.0
        model = ns.variants[mode].model
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yp = self._denorm_loss_db(model(xt).cpu().numpy())
        y_loss = self._rssi_to_loss_db(y_dbm)
        return float(np.mean((yp.flatten() - y_loss.flatten()) ** 2))

    def eval_rmse(
        self,
        ns: NodeState,
        mode: str,
        X: np.ndarray,
        y_dbm: np.ndarray,
    ) -> float:
        return float(np.sqrt(self.eval_mse(ns, mode, X, y_dbm)))

    def predict_loss_db(
        self,
        ns: NodeState,
        mode: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return propagation-loss predictions in dB for rows of X."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        model = ns.variants[mode].model
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yp = self._denorm_loss_db(model(xt).cpu().numpy().flatten())
        return yp

    def predict_dbm(
        self,
        ns: NodeState,
        mode: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return RSSI-equivalent predictions in dBm for rows of X."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        model = ns.variants[mode].model
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yp = self._denorm_dbm(model(xt).cpu().numpy().flatten())
        return yp

    def eval_mse_with_weights(
        self,
        mode: str,
        weights: dict[str, torch.Tensor],
        X: np.ndarray,
        y_dbm: np.ndarray,
    ) -> float:
        if X.shape[0] == 0:
            return 0.0
        tmp = self._make_predictor().to(self.device)
        tmp.load_state_dict({k: v.to(self.device) for k, v in weights.items()})
        tmp.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yp = self._denorm_loss_db(tmp(xt).cpu().numpy())
        y_loss = self._rssi_to_loss_db(y_dbm)
        return float(np.mean((yp.flatten() - y_loss.flatten()) ** 2))

    def eval_rmse_with_weights(
        self,
        mode: str,
        weights: dict[str, torch.Tensor],
        X: np.ndarray,
        y_dbm: np.ndarray,
    ) -> float:
        return float(np.sqrt(self.eval_mse_with_weights(mode, weights, X, y_dbm)))

    # --------------------------------------------------------------- merges

    def _load_scratch(self, model_state: dict[str, torch.Tensor]) -> nn.Module:
        """Load `model_state` into the shared scratch model and return it.

        Used by `_gossip_step` to make a pre-link snapshot of the provider
        variant available as an `nn.Module` for accepted full-model pulls,
        without having to allocate a fresh module per encounter.
        """
        self._scratch_model.load_state_dict(
            {k: t.to(self.device) for k, t in model_state.items()}
        )
        return self._scratch_model

    def _make_peer_view(self, ns_j: NodeState, mode: str) -> _LinkPeerView:
        """Snapshot `ns_j.variants[mode]`'s pre-link state into a peer view.

        The compact signature is what the receiver uses before deciding.
        The full weights are retained as simulation bookkeeping so an accepted
        pull in the second leg can still merge the provider's pre-link predictor.
        The view's `.model` attribute is the scratch model, loaded lazily by
        `_load_scratch` only after an accepted pull.
        """
        v_j = ns_j.variants[mode]
        snap = {k: t.detach().to("cpu").clone() for k, t in v_j.model.state_dict().items()}
        # The scratch model is loaded on demand (right before use) by the
        # caller — we can't do it here because subsequent peer views would
        # overwrite the scratch. We stash the state dict on the view via a
        # hidden attribute; see `_activate_peer_view`.
        # Keep both the raw additive count and the capped experience score.
        view = _LinkPeerView(
            m_samples=int(v_j.m_samples),
            n_samples=int(v_j.n_samples),
            last_rmse=float(v_j.last_rmse),
            last_rmse_available=bool(getattr(v_j, "last_rmse_available", False)),
            model_signature=self._variant_signature(v_j).clone(),
            model=self._scratch_model,
        )
        view._model_state = snap  # type: ignore[attr-defined]
        return view

    def _activate_peer_view(self, view: _LinkPeerView) -> None:
        """Load the view's snapshotted weights into the scratch model.

        Must be called exactly once before the view is consumed (by
        `_state_features` and `perform_merge`).
        """
        self._load_scratch(view._model_state)  # type: ignore[attr-defined]

    def perform_merge(
        self,
        ns_i: NodeState,
        ns_j: NodeState,
        mode: str,
        j_view: Optional[_LinkPeerView] = None,
    ) -> None:
        """Pull `ns_j`'s weights into `ns_i` and merge per-mode metadata.

        Experience used for the weighted average is the capped sample count
        (`n_samples`). Raw quantity (`m_samples`) is additive on the receiving
        side and then re-capped, so experience cannot grow without bound.

        If `j_view` is provided, the merge uses the snapshotted pre-link
        provider state (model signature, model weights, n_samples)
        instead of the live `ns_j` state. This is what makes the bidirectional
        gossip on a single physical link order-independent. The full provider
        weights are activated only after the policy accepted the pull.
        """
        v_i = ns_i.variants[mode]
        i_e = float(v_i.experience)
        if j_view is None:
            v_j = ns_j.variants[mode]
            j_m = int(v_j.m_samples)
            j_e = float(v_j.experience)
            j_model = v_j.model
        else:
            j_m = int(j_view.m_samples)
            j_e = float(j_view.experience)
            # Full provider weights are loaded only after the action accepted
            # the pull; decision-state construction uses the compact signature.
            self._activate_peer_view(j_view)
            j_model = j_view.model
        merge_strategy = str(self.cfg.merge_strategy)
        unidirectional_pull(
            v_i.model,
            i_e,
            j_model,
            j_e,
            merge_strategy=merge_strategy,
        )
        # Raw counts remain additive; the capped score is recomputed from the
        # raw total so the experiment matches the paper's m -> cap(m) rule.
        v_i.m_samples = bound_raw_samples(int(v_i.m_samples) + j_m)
        v_i.n_samples = saturate_n_samples(v_i.m_samples)
        self._refresh_variant_signature(v_i)
        v_i.t_wait = 0
        v_i.last_rmse_available = False

    # --------------------------------------------------- quality / mobility

    def _zone_cell_index(self, x: float, y: float, az: int) -> int:
        """Cell index in [0, K**2) for (x, y) inside zone `az`'s K x K grid."""
        K = max(1, int(self.cfg.quality_grid_k))
        x_lo, x_hi, y_lo, y_hi = zone_bounds(az, self.cfg.map_size, self.cfg.num_zones)
        # Linear bucket; clamp for points exactly on the high edge.
        cell_w = (x_hi - x_lo) / K
        cell_h = (y_hi - y_lo) / K
        cx = int((float(x) - x_lo) / cell_w) if cell_w > 0 else 0
        cy = int((float(y) - y_lo) / cell_h) if cell_h > 0 else 0
        cx = max(0, min(K - 1, cx))
        cy = max(0, min(K - 1, cy))
        return cy * K + cx

    def _bitmap_quality(self, bitmap: int) -> float:
        """Fraction of K**2 grid cells visited so far."""
        K = max(1, int(self.cfg.quality_grid_k))
        denom = float(K * K)
        return float(bin(int(bitmap)).count("1")) / denom

    def _update_visited(self, ns: NodeState) -> None:
        """Mark current cell as visited for local coverage diagnostics."""
        cell = self._zone_cell_index(ns.node.x, ns.node.y, ns.current_az)
        ns.visited_bitmap |= (1 << cell)
        bq = self._bitmap_quality(ns.visited_bitmap)
        for v in ns.variants.values():
            if bq > v.quality:
                v.quality = bq

    def _spike_recovery_settings(self, mode: str) -> tuple[bool, dict[str, float | int]]:
        cfg = self.cfg
        defaults: dict[str, float | int] = {
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
        rm = self.reward_modes.get(mode)
        if rm is None:
            return bool(cfg.spike_recovery_enabled), defaults
        params = getattr(rm, "spike_recovery_params", defaults)
        enabled = bool(getattr(rm, "spike_recovery_enabled", cfg.spike_recovery_enabled))
        return enabled, params

    def _tick_spike_recovery(self, ns: NodeState) -> None:
        for mode, v in ns.variants.items():
            enabled, params = self._spike_recovery_settings(mode)
            if not enabled and v.recovery_steps_left <= 0 and v.recovery_cooldown_left <= 0:
                continue
            if v.recovery_steps_left > 0:
                v.recovery_steps_left -= 1
                if v.recovery_steps_left <= 0:
                    v.recovery_accepts_left = 0
                    v.recovery_cooldown_left = max(
                        v.recovery_cooldown_left,
                        int(params["cooldown_steps"]),
                    )
            elif v.recovery_cooldown_left > 0:
                v.recovery_cooldown_left -= 1

    def _update_spike_recovery_state(
        self,
        ns: NodeState,
        mode: str,
        pre_train_rmse: float,
    ) -> None:
        enabled, params = self._spike_recovery_settings(mode)
        if not enabled:
            return
        v = ns.variants[mode]
        rmse = float(pre_train_rmse)
        if v.rmse_batches <= 0:
            v.rmse_ema_short = rmse
            v.rmse_ema_long = rmse
            v.rmse_batches = 1
            return
        v.rmse_batches += 1
        sa = max(0.0, min(1.0, float(params["short_alpha"])))
        la = max(0.0, min(1.0, float(params["long_alpha"])))
        v.rmse_ema_short = (1.0 - sa) * v.rmse_ema_short + sa * rmse
        v.rmse_ema_long = (1.0 - la) * v.rmse_ema_long + la * rmse
        if v.rmse_batches < int(params["min_batches"]):
            return
        if v.recovery_steps_left > 0 or v.recovery_cooldown_left > 0:
            return
        diff = float(v.rmse_ema_short - v.rmse_ema_long)
        ratio_hit = v.rmse_ema_short > float(params["ratio"]) * max(
            v.rmse_ema_long, 1e-6
        )
        abs_hit = diff > float(params["abs_db"])
        if ratio_hit and abs_hit:
            v.recovery_steps_left = int(params["window_steps"])
            v.recovery_accepts_left = int(params["accept_budget"])

    # --------------------------------------------------------- state (8d)

    def _state_features(
        self,
        mode: str,
        ns_i: NodeState,
        ns_j: NodeState,
        az: int,
        neighbor_count: int,
        j_view: Optional[_LinkPeerView] = None,
    ) -> torch.Tensor:
        """Build the 8-d state vector for the gossip decision.

        Layout: see `rl_agent.py` docstring. When `j_view` is provided, the
        provider's `n_samples`, `last_rmse`, and compact model
        signature are read from the snapshot so that the second leg of a
        bidirectional link sees the *pre-link* provider metadata, not the
        post-(first-leg) state.
        """
        eps = 1e-8
        cfg = self.cfg
        v_i = ns_i.variants[mode]
        if j_view is None:
            v_j = ns_j.variants[mode]
            j_n = float(v_j.n_samples)
            j_last_rmse = float(v_j.last_rmse)
            j_signature = self._variant_signature(v_j)
        else:
            j_n = float(j_view.n_samples)
            j_last_rmse = float(j_view.last_rmse)
            j_signature = j_view.model_signature.detach().to(dtype=torch.float32, device="cpu")
        noise_floor = float(abs(cfg.rssi_min_dbm))
        li_norm = float(v_i.last_rmse) / max(noise_floor, eps)
        e_i = float(v_i.experience)
        e_j = j_n
        rei = (e_i - e_j) / (e_i + e_j + eps)
        rqi = (v_i.last_rmse - j_last_rmse) / (v_i.last_rmse + j_last_rmse + eps)
        tau = max(float(cfg.t_norm_tau), eps)
        t_norm = 1.0 - math.exp(-float(v_i.t_wait) / tau)
        zone_x_norm, zone_y_norm = get_zone_center_feature(
            int(az),
            num_zones=int(cfg.num_zones),
        )
        w_diff = _signature_diff_norm(self._variant_signature(v_i), j_signature)
        denom_neigh = max(1, int(cfg.num_nodes) - 1)
        neighbor_norm = float(neighbor_count) / float(denom_neigh)
        return torch.tensor(
            [
                li_norm,
                rei,
                rqi,
                t_norm,
                zone_x_norm,
                zone_y_norm,
                w_diff,
                neighbor_norm,
            ],
            dtype=torch.float32,
        )

    # -------------------------------------------------- local model training

    def _local_predictor_training_targets(
        self,
        ns: NodeState,
        X: np.ndarray,
        y_scaled: np.ndarray,
        *,
        n_new_samples: int,
        sample_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Return normalized targets used to train the primary predictor.

        The default is the complete normalized propagation loss. Specialized
        simulations may update an external baseline here and return residual
        targets while retaining the common local replay/training loop.
        """

        del ns, X, n_new_samples, sample_weights
        return y_scaled

    def _train_local(
        self,
        ns: NodeState,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        sample_count_increment: int | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        if X.shape[0] == 0:
            return
        cfg = self.cfg
        y_scaled = self._normalize_target_from_rssi(y_dbm)
        weights = self._configured_sample_weights(
            X, sample_weights, zone=int(ns.current_az)
        )
        if weights is not None:
            weights = np.asarray(
                weights, dtype=np.float32
            ).reshape(-1, 1)
            if int(weights.shape[0]) != int(X.shape[0]):
                raise ValueError(
                    f"sample_weights length {weights.shape[0]} "
                    f"does not match X rows {X.shape[0]}"
                )
        n_samples = int(X.shape[0])
        n_new_samples = (
            n_samples
            if sample_count_increment is None
            else max(0, int(sample_count_increment))
        )
        training_targets = np.asarray(
            self._local_predictor_training_targets(
                ns,
                X,
                y_scaled,
                n_new_samples=n_new_samples,
                sample_weights=weights,
            ),
            dtype=np.float32,
        ).reshape(y_scaled.shape)
        node_idx = self.node_idx(ns)
        step_idx = int(getattr(self, "_current_sumo_step", 0))
        for mode, v in ns.variants.items():
            if n_new_samples > 0:
                record_support = getattr(v.model, "record_support", None)
                if callable(record_support):
                    record_support(
                        torch.as_tensor(
                            X[-n_new_samples:], dtype=torch.float32, device=self.device
                        )
                    )
            pre_rmse = self.eval_rmse(ns, mode, X, y_dbm)
            self._update_spike_recovery_state(ns, mode, pre_rmse)
            add_evidence = getattr(v.model, "add_evidence", None)
            if callable(add_evidence):
                if n_new_samples > 0:
                    generations = getattr(self, "_node_generations", None)
                    generation = (
                        int(generations[node_idx])
                        if generations is not None
                        else 0
                    )
                    origin = _stable_int64_key(
                        "mergeable-evidence",
                        int(cfg.seed),
                        node_idx,
                        generation,
                        int(ns.current_az),
                    )
                    start = n_samples - n_new_samples
                    add_evidence(
                        torch.as_tensor(
                            X[start:],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.as_tensor(
                            training_targets[start:],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        origin=origin,
                        sample_weights=(
                            None
                            if weights is None
                            else torch.as_tensor(
                                weights[start:],
                                dtype=torch.float32,
                                device=self.device,
                            )
                        ),
                    )
            else:
                self._fit_predictor(
                    v.model,
                    v.opt,
                    X,
                    training_targets,
                    weights,
                    device=self.device,
                    seed_parts=(
                        "local",
                        step_idx,
                        node_idx,
                        mode,
                    ),
                    n_new_samples=n_new_samples,
                )
            # Raw sample count is additive over newly observed samples only.
            if n_new_samples > 0:
                v.m_samples = bound_raw_samples(int(v.m_samples) + n_new_samples)
                v.n_samples = saturate_n_samples(v.m_samples)
            # RMSE on own links is valid for the current post-training predictor.
            v.last_rmse = self.eval_rmse(ns, mode, X, y_dbm)
            v.last_rmse_available = True
            self._refresh_variant_signature(v)

    # ------------------------------------------------------------- run loop

    def run(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.results_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.results_dir, "config.json"))
        if cfg.verbose:
            print(f"[RRE] Running {cfg.sim_steps} steps with modes {cfg.active_modes}")

        total_start = time.time()
        for step in range(1, cfg.sim_steps + 1):
            t0 = time.time()
            self._current_sumo_step = int(step)

            # 1. Move + zone bookkeeping (track visited cells for the local
            #    coverage diagnostic; `_reset_node_for_zone_change` clears
            #    the bitmap so coverage starts fresh in each zone).
            for ns in self.nodes:
                for v in ns.variants.values():
                    v.t_wait += 1
                self._tick_spike_recovery(ns)
                move_annulus_jump(
                    ns.node,
                    self.walls,
                    cfg.map_size,
                    cfg.move_annulus_min,
                    cfg.move_annulus_max,
                    cfg.xy_margin,
                    rng=self._rng_py,
                )
                new_az = zone_of(ns.node.x, ns.node.y, cfg.map_size, cfg.num_zones)
                if new_az != ns.current_az:
                    self._reset_node_for_zone_change(ns, new_az)
                self._update_visited(ns)

            # 2. Partition nodes by zone (used both for the gossip topology
            #    and for the subsequent ray-tracing call).
            zone_nodes: dict[int, list[int]] = defaultdict(list)
            for i, ns in enumerate(self.nodes):
                zone_nodes[ns.current_az].append(i)

            # 3. Oracle sets for v4 (ray traced, fresh every `oracle_every_k`
            #    steps). Built BEFORE gossip so the v4 mode can read them in
            #    `on_encounter`.
            if "v4" in self.reward_modes and step % cfg.oracle_every_k == 0:
                self._build_oracle_sets()

            for mode in self.reward_modes.values():
                mode.on_step_start(self, step)

            # 4. Gossip + per-mode reward. The topology is "all unique pairs
            #    of nodes within the same zone": gossip runs BEFORE this
            #    step's measurements so a node that just entered a new zone
            #    can absorb a peer's pre-trained model before it observes
            #    its first link in the new zone (reduces first-step error).
            self._gossip_step(step, zone_nodes)

            # 5. Ray trace this step's measurements (after gossip so the
            #    snapshots in step 4 reflect the post-merge model only).
            meas = self.tracer.step_measurements(
                [ns.node for ns in self.nodes], zone_nodes
            )

            # 6. Index measurements per node. Receiver-only: each
            #    measurement `(tx_idx -> rx_idx, val)` is the RSSI the
            #    receiver OBSERVED, so we attribute the sample only to the
            #    receiver. Each node therefore trains exclusively on links
            #    it itself measured (one-directional measurements).
            self._meas_per_node: dict[int, list[tuple[list[float], float]]] = defaultdict(list)
            for _az, tx_idx, rx_idx, val in meas:
                tx_node = self.nodes[tx_idx].node
                rx_node = self.nodes[rx_idx].node
                feat = self._pair_model_features(
                    (tx_node.x, tx_node.y),
                    (rx_node.x, rx_node.y),
                    step=step,
                )
                self._meas_per_node[rx_idx].append((feat, float(val)))

            # 7. Feed this step's measurements into all OPEN future-window
            #    slots, including the ones just opened in step 4 by the
            #    gossip pass. With T=1 this means the slot can mature within
            #    the same step (drain in step 8); with T>1 the slot keeps
            #    accumulating samples over future steps.
            stream_modes = [
                mode for mode in self.reward_modes.values()
                if hasattr(mode, "ingest_sample")
            ]
            if stream_modes:
                for i, rows in self._meas_per_node.items():
                    ns = self.nodes[i]
                    for feat, val in rows:
                        for mode in stream_modes:
                            mode.ingest_sample(ns, feat, val)  # type: ignore[attr-defined]

            # 8. Drain per-mode committed transitions (T-window slots that
            #    just filled up).
            for mode_id, mode in self.reward_modes.items():
                ready = mode.on_step_end(self, step)
                for t in ready:
                    s, a, r, s2, d, _node_idx = t
                    self.agents[mode_id].push(s, a, r, s2, d)

            # 9. Local per-variant training on own links (done AFTER merges
            #    have been tested this step, so reward reflects pure-merge
            #    effect rather than post-training confounds).
            for i, rows in self._meas_per_node.items():
                if not rows:
                    continue
                X = np.asarray([r[0] for r in rows], dtype=np.float32)
                y = np.asarray([r[1] for r in rows], dtype=np.float32).reshape(-1, 1)
                self._train_local(self.nodes[i], X, y)

            # 10. DQN training
            losses = {m: 0.0 for m in self.agents}
            for _ in range(cfg.rl_train_updates_per_step):
                for m, agent in self.agents.items():
                    losses[m] = agent.train_step()

            # 11. Fidelity logging on a rolling held-out set.
            if step % max(1, int(cfg.fidelity_refresh_every)) == 0:
                self._build_fidelity_grid(n_pairs=cfg.fidelity_grid_per_zone)
            if step % cfg.fidelity_log_every == 0:
                self._log_fidelity(step)

            if cfg.verbose:
                dt = time.time() - t0
                loss_str = " ".join(f"{m}:{losses[m]:.3f}" for m in self.agents)
                # Reuse the latest CSV row if it was just logged this step,
                # otherwise compute a fresh snapshot just for printing.
                fid_row = (
                    self.fidelity_history[-1]
                    if (
                        self.fidelity_history
                        and int(self.fidelity_history[-1].get("step", -1)) == int(step)
                    )
                    else self._compute_fidelity_row(step)
                )
                rmse_str = " ".join(
                    f"{m}:{float(fid_row.get(f'{m}_total', float('nan'))):.2f}"
                    for m in self.agents
                )
                print(
                    f"[RRE] step {step:03d}/{cfg.sim_steps}  dt={dt:.1f}s  "
                    f"rmse {rmse_str}  losses {loss_str}",
                    flush=True,
                )

        # Flush deferred reward windows.
        for mode_id, mode in self.reward_modes.items():
            ready = mode.on_sim_end(self)
            for t in ready:
                s, a, r, s2, d, _ = t
                self.agents[mode_id].push(s, a, r, s2, d)

        # Final high-confidence evaluation on a larger held-out set.
        self._build_fidelity_grid(n_pairs=cfg.final_fidelity_grid_per_zone)
        self.final_fidelity_snapshot = self._compute_fidelity_row(cfg.sim_steps)

        self._save_outputs()
        if cfg.verbose:
            print(f"[RRE] total runtime {time.time() - total_start:.1f}s")

    # ------------------------------------------------------- oracle / gossip

    def _build_oracle_sets(self) -> None:
        cfg = self.cfg
        self.oracle_sets.clear()
        for az in range(cfg.num_zones):
            pairs = sample_oracle_pairs(
                zone=az,
                walls=self.walls,
                map_size=cfg.map_size,
                margin=cfg.xy_margin,
                n_tx=cfg.oracle_n_tx_per_zone,
                n_pairs=cfg.oracle_n_pairs_per_zone,
                rng=self._rng_np,
                num_zones=cfg.num_zones,
            )
            groups = group_pairs_by_tx(pairs)
            rssi_groups = self.tracer.measure_pairs(groups)
            X = []
            y = []
            for (tx, rxs), rssi_list in zip(groups, rssi_groups):
                for rx, val in zip(rxs, rssi_list):
                    X.append(self._pair_model_features(tx, rx))
                    y.append(val)
            self.oracle_sets[az] = (
                np.asarray(X, dtype=np.float32),
                np.asarray(y, dtype=np.float32).reshape(-1, 1),
            )

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Run the gossip / merge pass for this step.

        Topology defaults to all unique same-zone pairs. Callers may pass an
        explicit feasible-contact list, which SUMO uses for bidirectional
        RSSI-thresholded contacts.

        Per link `(a, b)` (with `a < b`) we process BOTH legs `(a, b)` and
        `(b, a)`, but in an order-independent way: a single pre-link
        snapshot of `a`'s variant state is taken at the start of the link.
        Leg 1 `(a, b)` uses the live `b` (which is also pre-link, since `b`
        has not been touched yet in this link). Leg 2 `(b, a)` uses the
        snapshot of `a`. This guarantees:

        - sample accumulation is symmetric: both ends end up with
          `samples_a_pre + samples_b_pre` (no geometric inflation when
          both legs accept the merge),
        - state features `rei`, `rqi`, `w_diff` for leg 2 are computed
          against `a_pre`, not `a_post-leg-1`,
        - the merged weights for `b` are `mix(b_pre, a_pre)` rather than
          `mix(b_pre, a_post-leg-1)`.
        """
        # Build per-zone unique pairs as the gossip topology unless the
        # caller supplies an explicit feasible-contact set. SUMO uses the
        # explicit path for RSSI-thresholded same-zone contacts.
        links: list[tuple[int, int, int]] = []
        if contact_links is None:
            for az, indices in zone_nodes.items():
                ids = sorted(indices)
                for ii in range(len(ids)):
                    for jj in range(ii + 1, len(ids)):
                        links.append((int(az), int(ids[ii]), int(ids[jj])))
        else:
            seen: set[tuple[int, int, int]] = set()
            for az, a, b in contact_links:
                ia = int(a)
                ib = int(b)
                if ia == ib:
                    continue
                if ib < ia:
                    ia, ib = ib, ia
                if ia < 0 or ib < 0 or ia >= len(self.nodes) or ib >= len(self.nodes):
                    continue
                if int(self.nodes[ia].current_az) != int(self.nodes[ib].current_az):
                    continue
                key = (int(az), ia, ib)
                if key not in seen:
                    links.append(key)
                    seen.add(key)
        links.sort(key=lambda row: (row[0], row[1], row[2]))

        # Neighbour density per node (= feasible same-zone contacts).
        neighbour: defaultdict[int, int] = defaultdict(int)
        for _az, a, b in links:
            neighbour[a] += 1
            neighbour[b] += 1

        for az, a, b in links:
            ns_a = self.nodes[a]
            ns_b = self.nodes[b]
            dist = float(
                np.hypot(ns_a.node.x - ns_b.node.x, ns_a.node.y - ns_b.node.y)
            )

            # Snapshot a's pre-link variant state once for this link, for
            # use as the j_view of leg 2 `(b, a)`. We don't need a snapshot
            # for b: leg 1 `(a, b)` runs first and only mutates a, so b is
            # still pre-link when leg 1 reads it live.
            a_pre_link: dict[str, _LinkPeerView] = {
                mode_id: self._make_peer_view(ns_a, mode_id)
                for mode_id in self.reward_modes
            }

            enc_id = self._next_enc_id
            self._next_enc_id += 1

            # ---- Leg 1: (a, b) -- live v_b, no snapshot needed.
            self._run_leg(
                step=step,
                enc_id=enc_id,
                az=az,
                dist=dist,
                ns_i=ns_a,
                ns_j=ns_b,
                i_idx=a,
                j_idx=b,
                neighbor_count=neighbour[a],
                j_views=None,
            )

            # ---- Leg 2: (b, a) -- use a's pre-link snapshot per mode.
            self._run_leg(
                step=step,
                enc_id=enc_id,
                az=az,
                dist=dist,
                ns_i=ns_b,
                ns_j=ns_a,
                i_idx=b,
                j_idx=a,
                neighbor_count=neighbour[b],
                j_views=a_pre_link,
            )

    def _select_action(
        self,
        mode_id: str,
        state: torch.Tensor,
        node_idx: int | None = None,
    ) -> int:
        del node_idx
        return self.agents[mode_id].select_action(state, rng=self._rng_py)

    def _queue_rl_transition(self, mode_id: str, transition) -> None:
        s, a, r, s2, d, _node_idx = transition
        self.agents[mode_id].push(s, a, r, s2, d)

    def _train_rl_agents(self) -> dict[str, float]:
        losses = {m: 0.0 for m in self.agents}
        for _ in range(self.cfg.rl_train_updates_per_step):
            for m, agent in self.agents.items():
                losses[m] = agent.train_step()
        return losses

    def _record_decision_row(self, row: dict) -> None:
        self.decision_log.append(row)

    def _run_leg(
        self,
        *,
        step: int,
        enc_id: int,
        az: int,
        dist: float,
        ns_i: NodeState,
        ns_j: NodeState,
        i_idx: int,
        j_idx: int,
        neighbor_count: int,
        j_views: Optional[dict[str, _LinkPeerView]],
    ) -> None:
        """Process one direction of one link across all reward modes.

        If `j_views` is provided (leg 2), the compact signature in each view
        is used for the decision state. Full snapshotted weights are loaded
        into the scratch model only if that decision accepts the pull.
        """
        for mode_id, mode in self.reward_modes.items():
            j_view = None
            if j_views is not None:
                j_view = j_views[mode_id]

            state = self._state_features(
                mode_id, ns_i, ns_j, az, neighbor_count, j_view=j_view,
            )
            action = self._select_action(mode_id, state, node_idx=i_idx)
            v_i = ns_i.variants[mode_id]
            spike_enabled, spike_params = self._spike_recovery_settings(mode_id)
            if (
                spike_enabled
                and v_i.recovery_steps_left > 0
                and v_i.recovery_accepts_left > 0
                and int(action) == 0
                and self._rng_py.random() < float(spike_params["accept_prob"])
            ):
                action = 1
            # Bandit-style: each encounter is an independent terminal step
            # (`done=True`, `next_state == state`). See
            # docs/rl_reward_method_comparison.md for rationale.
            done = True
            next_state = state.clone()
            trans = mode.on_encounter(
                self, step, ns_i, ns_j, az, action,
                state, next_state, done,
                j_view=j_view,
            )
            if trans is not None:
                self._queue_rl_transition(mode_id, trans)
            self._record_decision_row(
                {
                    "step": step,
                    "enc_id": enc_id,
                    "node_i": i_idx,
                    "node_j": j_idx,
                    "az": az,
                    "dist": dist,
                    "mode": mode_id,
                    "action": int(action),
                    "reward": float(trans[2]) if trans is not None else float("nan"),
                    "deferred": trans is None,
                }
            )

    # --------------------------------------------------------------- logging

    def _log_fidelity(self, step: int) -> None:
        self.fidelity_history.append(self._compute_fidelity_row(step))

    def _save_outputs(self) -> None:
        out = Path(self.cfg.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Fidelity CSV
        if self.fidelity_history:
            fieldnames = sorted({k for row in self.fidelity_history for k in row.keys()})
            fieldnames = ["step"] + [f for f in fieldnames if f != "step"]
            with open(out / "fidelity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in self.fidelity_history:
                    w.writerow(row)
        # Decisions CSV
        if self.decision_log:
            fieldnames = [
                "step", "enc_id", "node_i", "node_j", "az", "dist",
                "mode", "action", "reward", "deferred",
            ]
            with open(out / "decisions.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in self.decision_log:
                    w.writerow(row)
        # Final fidelity snapshot (evaluated on final_fidelity_grid_per_zone).
        if self.final_fidelity_snapshot is not None:
            with open(out / "final_fidelity.json", "w", encoding="utf-8") as f:
                json.dump(self.final_fidelity_snapshot, f, indent=2, sort_keys=True)
        elif self.fidelity_history:
            with open(out / "final_fidelity.json", "w", encoding="utf-8") as f:
                json.dump(self.fidelity_history[-1], f, indent=2, sort_keys=True)
        # Cleanup scene temp files.
        try:
            self.map_engine.cleanup()
        except Exception:
            pass
