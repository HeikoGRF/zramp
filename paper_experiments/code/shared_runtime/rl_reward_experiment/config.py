"""
Configuration for the RL reward comparison experiment.

All knobs are CLI-overridable (see `run.py`) and can also be provided through
`RRE_*` environment variables so that sweep scripts can drive runs without
editing Python.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    out: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(out)


WINDOW_T_VALUES = (1, 2, 3, 4, 5, 7)
WINDOW_MODE_IDS = tuple(f"t{t}" for t in WINDOW_T_VALUES)
WINDOW_T_BY_MODE = {f"t{t}": t for t in WINDOW_T_VALUES}
MODE_IDS = (*WINDOW_MODE_IDS, "v4")
MODE_NAMES = {
    **{f"t{t}": f"FutureWindowT{t}" for t in WINDOW_T_VALUES},
    "v4": "Reference",
}
DEFAULT_ACTIVE_MODES = (*WINDOW_MODE_IDS, "v4")

_ACTION_POLICIES = ("softmax", "argmax", "reject", "accept")
_POLICY_SUFFIX_RE = re.compile(r"^(.+)_(softmax|argmax|reject|accept)$")
_FUTURE_BETA_MODE = re.compile(r"^t([0-9]+)_b([-+eE0-9.]+)$")
_SPIKE_PROFILE_RE = re.compile(r"^(.+)_sp(mild|rec|recommended|strong|aggr|aggressive)$")

SPIKE_RECOVERY_PROFILES = {
    "mild": {
        "ratio": 1.35,
        "abs_db": 3.0,
        "window_steps": 8,
        "accept_budget": 1,
        "cooldown_steps": 25,
        "beta_scale": 0.75,
        "accept_prob": 0.10,
    },
    "recommended": {
        "ratio": 1.25,
        "abs_db": 2.0,
        "window_steps": 12,
        "accept_budget": 3,
        "cooldown_steps": 20,
        "beta_scale": 0.50,
        "accept_prob": 0.25,
    },
    "strong": {
        "ratio": 1.15,
        "abs_db": 1.5,
        "window_steps": 16,
        "accept_budget": 5,
        "cooldown_steps": 15,
        "beta_scale": 0.35,
        "accept_prob": 0.40,
    },
    "aggressive": {
        "ratio": 1.10,
        "abs_db": 1.0,
        "window_steps": 20,
        "accept_budget": 8,
        "cooldown_steps": 10,
        "beta_scale": 0.25,
        "accept_prob": 0.60,
    },
}


def split_spike_profile(mode_id: str) -> tuple[str, str | None]:
    """Return (base_reward_mode_id, spike profile suffix if present)."""
    m = _SPIKE_PROFILE_RE.fullmatch(mode_id.strip().lower())
    if not m:
        return mode_id.strip().lower(), None
    profile = m.group(2)
    if profile == "rec":
        profile = "recommended"
    elif profile == "aggr":
        profile = "aggressive"
    return m.group(1), profile


def split_mode_policy(mode_id: str) -> tuple[str, str | None]:
    """Return (reward_mode_id, per-mode action policy suffix if present)."""
    m = _POLICY_SUFFIX_RE.fullmatch(mode_id.strip().lower())
    if not m:
        return mode_id.strip().lower(), None
    return m.group(1), m.group(2)


def parse_modes(raw: str) -> tuple[str, ...]:
    """Parse a mode list like "1,2,3,5,7,4", "t1,t3", or "reference"/"oracle" (v4).

    Also accepts per-beta future-window RL heads in one run, e.g. ``t2_b1``, ``t2_b1.25``
    (fixed horizon T but distinct reward shaping β).
    """
    out: list[str] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        base_t, policy = split_mode_policy(t)
        suffix = f"_{policy}" if policy is not None else ""
        t, spike_profile = split_spike_profile(base_t)
        spike_suffix = f"_sp{spike_profile}" if spike_profile is not None else ""
        bm = _FUTURE_BETA_MODE.fullmatch(t)
        if bm:
            tw = int(bm.group(1))
            bval = float(bm.group(2))
            if f"t{tw}" not in WINDOW_T_BY_MODE:
                raise ValueError(
                    f"Unknown reward mode {token!r}; future-window T={tw} is not "
                    f"supported (expected one of {WINDOW_T_VALUES})"
                )
            key = f"t{tw}_b{bval:g}{spike_suffix}{suffix}"
            if key not in out:
                out.append(key)
            continue
        if spike_profile is not None:
            raise ValueError(
                f"Spike profile suffix in {token!r} is only supported for "
                "compound future-window beta ids like t2_b1_spmild"
            )
        if t in {"oracle", "o", "reference", "ref"}:
            t = "v4"
        elif t.startswith("w") and t[1:].isdigit():
            t = f"t{int(t[1:])}"
        if t.isdigit():
            n = int(t)
            t = "v4" if n == 4 else f"t{n}"
        if t not in MODE_IDS:
            raise ValueError(
                f"Unknown reward mode {token!r}; expected one of {MODE_IDS}, "
                f"compound ids t<W>_b<β> with W ∈ {WINDOW_T_VALUES}, "
                "or integer windows 1, 2, 3, 5, 7 plus reference mode 4 (v4)"
            )
        key = f"{t}{suffix}"
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one reward mode must be active")
    return tuple(out)


@dataclass
class ExperimentConfig:
    """Frozen set of parameters for a single simulation run."""

    seed: int = 0
    num_nodes: int = 70
    num_zones: int = 4
    sim_steps: int = 200
    map_size: float = 100.0

    # RL
    beta: float = 2.0
    active_modes: tuple[str, ...] = DEFAULT_ACTIVE_MODES
    gamma: float = 0.99
    rl_lr: float = 1e-3
    rl_batch_size: int = 64
    rl_train_updates_per_step: int = 4
    rl_target_tau: float = 0.01
    replay_capacity: int = 10_000
    epsilon_start: float = 1.0
    # End at 0.0 so post-decay decisions are purely greedy: any merge that
    # happens late in the run is the *learned policy* picking action 1, not
    # epsilon-greedy noise. This keeps "always reject" cleanly distinguishable
    # from "reject + ε/2 random accepts".
    epsilon_end: float = 0.0
    epsilon_decay_steps: int = 150
    rl_action_policy: str = "softmax"

    # Local training
    local_lr: float = 1e-3
    local_batch_size: int = 64
    # One fixed pass preserves aggregation gains; never train the local buffer
    # to convergence after receiving shared model parameters.
    local_epochs: int = 1
    # Zero preserves the legacy full-epoch pass. A positive value bounds the
    # online work to exactly this many minibatches per simulation step.
    local_batches_per_step: int = 0
    # When enabled, visit every newly received row exactly once before the
    # configured replay minibatches. This makes received-sample maturity
    # correspond to predictor optimization exposure at acquisition time.
    local_train_all_new_samples: bool = False
    # Optional role-agnostic maturity schedule. When both values are positive,
    # the minibatch budget grows linearly from local_batches_per_step to this
    # maximum as the retained predictor replay reaches maturity_rows.
    local_batches_per_step_max: int = 0
    local_batches_maturity_rows: int = 0
    # Additional classifier-only updates for two-path censored predictors.
    local_classifier_batches_per_step: int = 0
    local_classifier_hard_candidate_multiplier: int = 4
    # L2-SP regularization toward the common initialized predictor.
    local_initialization_anchor_strength: float = 0.0
    # Bins per normalized endpoint coordinate for spatial balancing.
    local_spatial_balance_bins: int = 4
    rssi_model: str = "tiny"
    mergeable_basis_dim: int = 192
    mergeable_ridge: float = 1.0
    merge_strategy: str = "average"
    predictor_include_time: bool = False
    # Re-express global [0,1] coordinates inside the active square AZ. This
    # spends local-grid capacity on the current neighborhood without changing
    # the physical map, zone boundaries, or routes.
    predictor_zone_local_coordinates: bool = False
    local_support_spatial_grid_points: int = 9
    local_support_prior_strength: float = 0.0
    # Global physical time is step * duration; the encoder then uses
    # u = physical_time / unit. Neither quantity depends on run horizon.
    predictor_time_step_duration: float = 1.0
    predictor_time_unit: float = 1.0
    predictor_time_num_frequencies: int = 8
    predictor_time_min_period: float = 2.0
    predictor_time_max_period: float = 1000.0
    # Fixed scale for the learned scalar-time encoder, in the same physical
    # units as predictor_time_step_duration. It must not depend on sim_steps.
    predictor_learned_time_scale: float = 1000.0
    # Conservative prior for unobserved links. With "snr-threshold", freshly
    # initialized predictors output the propagation loss corresponding to the
    # minimum receivable SNR. Received samples then provide the only evidence
    # that particular TX/RX geometries are better than this boundary.
    predictor_prior: str = "snr-threshold"
    # Spatial-balanced gives equal aggregate weight to occupied sparse TX/RX
    # coordinate cells and deliberately ignores age. Exponential-recency is
    # retained for reproducing older runs.
    local_sample_weighting: str = "uniform"
    local_sample_recency_half_life_steps: float = 50.0

    # Mode-specific
    pending_slot_cap: int = 32
    oracle_n_pairs_per_zone: int = 50
    oracle_n_tx_per_zone: int = 10  # groups of RX per TX = n_pairs / n_tx
    oracle_every_k: int = 1

    # State features
    # K x K grid per zone for the quality-of-experience tracker.
    # K=10 -> 100 cells of side ~map_size / (2 * K) = 5 units, comparable to
    # the per-step mobility annulus.
    quality_grid_k: int = 10
    # Saturation timescale for the t_norm feature: t_norm = 1 - exp(-t_wait / tau).
    # tau ~= 10 steps puts the feature at ~0.63 after 10 idle steps and ~0.95
    # after 30, which matches the typical zone-residence timescale.
    t_norm_tau: float = 10.0

    # Opt-in local spike recovery experiment. Disabled by default so the
    # regular simulation semantics stay unchanged.
    spike_recovery_enabled: bool = False
    spike_recovery_short_alpha: float = 0.45
    spike_recovery_long_alpha: float = 0.08
    spike_recovery_ratio: float = 1.25
    spike_recovery_abs_db: float = 2.0
    spike_recovery_min_batches: int = 5
    spike_recovery_window_steps: int = 12
    spike_recovery_accept_budget: int = 3
    spike_recovery_cooldown_steps: int = 20
    spike_recovery_beta_scale: float = 0.5
    spike_recovery_accept_prob: float = 0.25

    # Fidelity
    fidelity_grid_per_zone: int = 100  # rolling held-out eval set used for map-fidelity metric
    fidelity_grid_n_tx: int = 10
    fidelity_refresh_every: int = 10
    final_fidelity_grid_per_zone: int = 500
    fidelity_final_steps: tuple[int, ...] = ()
    fidelity_eval_every: int = 0
    fidelity_log_every: int = 1

    # Mobility
    move_annulus_min: float = 1.0
    move_annulus_max: float = 3.0
    xy_margin: float = 0.5

    # Ray tracing
    num_rays: int = 100_000
    max_depth: int = 2
    trace_tx_batch_size: int = 32
    freq_hz: float = 3.5e9
    tx_power_dbm: float = 15.0
    rssi_min_dbm: float = -120.0
    rssi_max_dbm: float = 15.0
    # Feasible reception follows the paper's SNR condition. The received-power
    # threshold used for cached RSSI measurements is derived as
    # noise_floor_dbm + snr_min_db. The legacy RSSI threshold is accepted only as
    # an input alias and converted to snr_min_db in __post_init__.
    noise_floor_dbm: float = -105.0
    snr_min_db: float = 5.0
    model_transfer_snr_min_db: float | None = None
    legacy_rssi_gossip_threshold_dbm: float | None = None
    rssi_gossip_threshold: float = field(init=False)

    # I/O
    results_dir: str = "results/rl_reward_experiment/seed_00"

    # Logging
    verbose: bool = True

    def __post_init__(self) -> None:
        self.rssi_model = str(self.rssi_model or "tiny").strip().lower().replace("_", "-")
        known_model = self.rssi_model in {"mergeable-rff", "mergeable-evidence", "evidence-ridge", "mergeable-log-distance", "exact-log-distance", "sufficient-statistics-log-distance", "local-support", "conservative-local-support", "conservative-grid", "local-grid", "micro", "single-64", "4-64-1", "tiny", "4-64-64-1", "64", "log-distance-only", "learned-log-distance", "distance-only-learned", "log-distance-residual", "distance-residual", "tiny-distance-residual", "censored-tiny", "two-head-tiny", "hurdle-tiny", "hard-censored-tiny", "hard-two-head-tiny", "hard-hurdle-tiny", "small", "medium-small", "4-128-128-128-1", "128", "censored-small", "two-head-small", "hurdle-small", "hard-censored-small", "hard-two-head-small", "hard-hurdle-small", "hard-ensemble-hurdle", "ensemble-hurdle", "blocked-calibrated-ensemble", "hard-blockage-ensemble", "dual", "large", "4-512-512-512-512-1", "512"}
        rbf_model = self.rssi_model.startswith("rbf-distance-residual-k") and self.rssi_model.removeprefix("rbf-distance-residual-k").isdigit()
        if not (known_model or rbf_model):
            raise ValueError(f"Unknown rssi_model={self.rssi_model!r}")
        self.mergeable_basis_dim = int(self.mergeable_basis_dim)
        self.mergeable_ridge = float(self.mergeable_ridge)
        if self.mergeable_basis_dim < 8:
            raise ValueError("mergeable_basis_dim must be at least 8")
        if not math.isfinite(self.mergeable_ridge) or self.mergeable_ridge <= 0.0:
            raise ValueError("mergeable_ridge must be finite and positive")
        self.local_lr = float(self.local_lr)
        self.local_batch_size = int(self.local_batch_size)
        self.local_epochs = int(self.local_epochs)
        self.local_batches_per_step = int(self.local_batches_per_step)
        self.local_train_all_new_samples = bool(
            self.local_train_all_new_samples
        )
        self.local_batches_per_step_max = int(
            self.local_batches_per_step_max
        )
        self.local_batches_maturity_rows = int(
            self.local_batches_maturity_rows
        )
        self.local_classifier_batches_per_step = int(
            self.local_classifier_batches_per_step
        )
        self.local_classifier_hard_candidate_multiplier = int(
            self.local_classifier_hard_candidate_multiplier
        )
        self.local_initialization_anchor_strength = float(
            self.local_initialization_anchor_strength
        )
        self.local_spatial_balance_bins = int(self.local_spatial_balance_bins)
        if not math.isfinite(self.local_lr) or self.local_lr <= 0.0:
            raise ValueError("local_lr must be finite and positive")
        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.local_batches_per_step < 0:
            raise ValueError("local_batches_per_step must be nonnegative")
        if self.local_batches_per_step_max < 0:
            raise ValueError(
                "local_batches_per_step_max must be nonnegative"
            )
        if (
            self.local_batches_per_step_max > 0
            and self.local_batches_per_step_max
            < self.local_batches_per_step
        ):
            raise ValueError(
                "local_batches_per_step_max must be at least "
                "local_batches_per_step"
            )
        if self.local_batches_maturity_rows < 0:
            raise ValueError(
                "local_batches_maturity_rows must be nonnegative"
            )
        if self.local_classifier_batches_per_step < 0:
            raise ValueError(
                "local_classifier_batches_per_step must be nonnegative"
            )
        if self.local_classifier_hard_candidate_multiplier < 1:
            raise ValueError(
                "local_classifier_hard_candidate_multiplier must be positive"
            )
        if (self.local_batches_per_step_max > 0) != (
            self.local_batches_maturity_rows > 0
        ):
            raise ValueError(
                "local_batches_per_step_max and "
                "local_batches_maturity_rows must be enabled together"
            )
        if (
            not math.isfinite(self.local_initialization_anchor_strength)
            or self.local_initialization_anchor_strength < 0.0
        ):
            raise ValueError(
                "local_initialization_anchor_strength must be finite and nonnegative"
            )
        if self.local_spatial_balance_bins < 1:
            raise ValueError("local_spatial_balance_bins must be positive")
        self.merge_strategy = str(self.merge_strategy or "average").strip().lower().replace("_", "-")
        self.predictor_include_time = bool(self.predictor_include_time)
        self.predictor_zone_local_coordinates = bool(
            self.predictor_zone_local_coordinates
        )
        self.local_support_spatial_grid_points = int(
            self.local_support_spatial_grid_points
        )
        if self.local_support_spatial_grid_points < 2:
            raise ValueError("local_support_spatial_grid_points must be at least 2")
        self.local_support_prior_strength = float(
            self.local_support_prior_strength
        )
        if not math.isfinite(self.local_support_prior_strength) or self.local_support_prior_strength < 0.0:
            raise ValueError("local_support_prior_strength must be finite and nonnegative")
        self.predictor_time_step_duration = float(self.predictor_time_step_duration)
        self.predictor_time_unit = float(self.predictor_time_unit)
        self.predictor_time_num_frequencies = int(self.predictor_time_num_frequencies)
        self.predictor_time_min_period = float(self.predictor_time_min_period)
        self.predictor_time_max_period = float(self.predictor_time_max_period)
        self.predictor_learned_time_scale = float(
            self.predictor_learned_time_scale
        )
        if self.predictor_time_step_duration <= 0.0:
            raise ValueError("predictor_time_step_duration must be positive")
        if self.predictor_time_unit <= 0.0:
            raise ValueError("predictor_time_unit must be positive")
        if self.predictor_time_num_frequencies <= 0:
            raise ValueError("predictor_time_num_frequencies must be positive")
        if self.predictor_time_min_period <= 0.0 or self.predictor_time_max_period <= self.predictor_time_min_period:
            raise ValueError("Require 0 < predictor_time_min_period < predictor_time_max_period")
        if (
            not math.isfinite(self.predictor_learned_time_scale)
            or self.predictor_learned_time_scale <= 0.0
        ):
            raise ValueError("predictor_learned_time_scale must be finite and positive")
        self.predictor_prior = (
            str(self.predictor_prior or "snr-threshold").strip().lower().replace("_", "-")
        )
        if self.predictor_prior in {"bad", "bad-link", "threshold", "snr", "snr-floor"}:
            self.predictor_prior = "snr-threshold"
        elif self.predictor_prior in {"off", "random"}:
            self.predictor_prior = "none"
        elif self.predictor_prior in {"max", "worst", "worst-link"}:
            self.predictor_prior = "max-loss"
        if self.predictor_prior not in {"snr-threshold", "max-loss", "none"}:
            raise ValueError(f"Unknown predictor_prior={self.predictor_prior!r}")
        self.local_sample_weighting = (
            str(self.local_sample_weighting or "uniform").strip().lower().replace("_", "-")
        )
        if self.local_sample_weighting in {"none", "off"}:
            self.local_sample_weighting = "uniform"
        elif self.local_sample_weighting in {"recency", "exp", "exponential"}:
            self.local_sample_weighting = "exponential-recency"
        elif self.local_sample_weighting in {
            "spatial", "balanced", "spatial-balance"
        }:
            self.local_sample_weighting = "spatial-balanced"
        if self.local_sample_weighting not in {
            "uniform", "exponential-recency", "spatial-balanced"
        }:
            raise ValueError(f"Unknown local_sample_weighting={self.local_sample_weighting!r}")
        self.local_sample_recency_half_life_steps = float(
            self.local_sample_recency_half_life_steps
        )
        if self.local_sample_recency_half_life_steps <= 0.0:
            raise ValueError("local_sample_recency_half_life_steps must be positive")
        self.noise_floor_dbm = float(self.noise_floor_dbm)
        self.snr_min_db = float(self.snr_min_db)
        if self.legacy_rssi_gossip_threshold_dbm is not None:
            self.legacy_rssi_gossip_threshold_dbm = float(
                self.legacy_rssi_gossip_threshold_dbm
            )
            self.snr_min_db = (
                self.legacy_rssi_gossip_threshold_dbm - self.noise_floor_dbm
            )
        self.model_transfer_snr_min_db = (
            self.snr_min_db
            if self.model_transfer_snr_min_db is None
            else float(self.model_transfer_snr_min_db)
        )
        if self.model_transfer_snr_min_db < self.snr_min_db:
            raise ValueError(
                "model_transfer_snr_min_db cannot be below snr_min_db"
            )
        self.rssi_gossip_threshold = self.noise_floor_dbm + self.snr_min_db
        if self.merge_strategy in {"mean", "weighted-average"}:
            self.merge_strategy = "average"
        elif self.merge_strategy in {"sliced-ot", "transport"}:
            self.merge_strategy = "ot"
        if self.merge_strategy not in {"average", "ot"}:
            raise ValueError(f"Unknown merge_strategy={self.merge_strategy!r}")
        if int(self.num_zones) < 1:
            raise ValueError("num_zones must be positive")
        side = int(math.isqrt(int(self.num_zones)))
        if side * side != int(self.num_zones):
            raise ValueError(
                f"num_zones={self.num_zones} must be a perfect square "
                "(e.g. 4, 9, 16) for square-grid partitioning"
            )

    @property
    def map_half(self) -> float:
        return self.map_size / 2.0

    @property
    def zones_per_side(self) -> int:
        return int(math.isqrt(int(self.num_zones)))

    @property
    def zone_diag(self) -> float:
        return (self.map_size / float(self.zones_per_side)) * (2.0**0.5)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["active_modes"] = list(self.active_modes)
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


def build_config_from_env(**overrides) -> ExperimentConfig:
    """Construct an `ExperimentConfig` from `RRE_*` env vars with explicit overrides on top."""
    cfg = ExperimentConfig(
        seed=_env_int("RRE_SEED", ExperimentConfig.seed),
        num_nodes=_env_int("RRE_NUM_NODES", ExperimentConfig.num_nodes),
        num_zones=_env_int("RRE_NUM_ZONES", ExperimentConfig.num_zones),
        sim_steps=_env_int("RRE_SIM_STEPS", ExperimentConfig.sim_steps),
        map_size=_env_float("RRE_MAP_SIZE", ExperimentConfig.map_size),
        local_lr=_env_float("RRE_LOCAL_LR", ExperimentConfig.local_lr),
        local_batch_size=_env_int("RRE_LOCAL_BATCH_SIZE", ExperimentConfig.local_batch_size),
        local_epochs=_env_int("RRE_LOCAL_EPOCHS", ExperimentConfig.local_epochs),
        local_batches_per_step=_env_int(
            "RRE_LOCAL_BATCHES_PER_STEP", ExperimentConfig.local_batches_per_step
        ),
        local_train_all_new_samples=_env_bool(
            "RRE_LOCAL_TRAIN_ALL_NEW_SAMPLES",
            ExperimentConfig.local_train_all_new_samples,
        ),
        local_batches_per_step_max=_env_int(
            "RRE_LOCAL_BATCHES_PER_STEP_MAX",
            ExperimentConfig.local_batches_per_step_max,
        ),
        local_batches_maturity_rows=_env_int(
            "RRE_LOCAL_BATCHES_MATURITY_ROWS",
            ExperimentConfig.local_batches_maturity_rows,
        ),
        local_classifier_batches_per_step=_env_int(
            "RRE_LOCAL_CLASSIFIER_BATCHES_PER_STEP",
            ExperimentConfig.local_classifier_batches_per_step,
        ),
        local_classifier_hard_candidate_multiplier=_env_int(
            "RRE_LOCAL_CLASSIFIER_HARD_CANDIDATE_MULTIPLIER",
            ExperimentConfig.local_classifier_hard_candidate_multiplier,
        ),
        local_initialization_anchor_strength=_env_float(
            "RRE_LOCAL_INITIALIZATION_ANCHOR_STRENGTH",
            ExperimentConfig.local_initialization_anchor_strength,
        ),
        local_spatial_balance_bins=_env_int(
            "RRE_LOCAL_SPATIAL_BALANCE_BINS",
            ExperimentConfig.local_spatial_balance_bins,
        ),
        beta=_env_float("RRE_BETA", ExperimentConfig.beta),
        active_modes=parse_modes(_env_str("RRE_MODES", "1,2,3,5,7,4")),
        pending_slot_cap=_env_int(
            "RRE_PENDING_SLOT_CAP", ExperimentConfig.pending_slot_cap
        ),
        oracle_n_pairs_per_zone=_env_int(
            "RRE_ORACLE_N_PAIRS_PER_ZONE",
            ExperimentConfig.oracle_n_pairs_per_zone,
        ),
        oracle_n_tx_per_zone=_env_int(
            "RRE_ORACLE_N_TX_PER_ZONE", ExperimentConfig.oracle_n_tx_per_zone
        ),
        oracle_every_k=_env_int(
            "RRE_ORACLE_EVERY_K", ExperimentConfig.oracle_every_k
        ),
        quality_grid_k=_env_int(
            "RRE_QUALITY_GRID_K", ExperimentConfig.quality_grid_k
        ),
        t_norm_tau=_env_float(
            "RRE_T_NORM_TAU", ExperimentConfig.t_norm_tau
        ),
        spike_recovery_enabled=bool(
            _env_int(
                "RRE_SPIKE_RECOVERY_ENABLED",
                int(ExperimentConfig.spike_recovery_enabled),
            )
        ),
        spike_recovery_short_alpha=_env_float(
            "RRE_SPIKE_RECOVERY_SHORT_ALPHA",
            ExperimentConfig.spike_recovery_short_alpha,
        ),
        spike_recovery_long_alpha=_env_float(
            "RRE_SPIKE_RECOVERY_LONG_ALPHA",
            ExperimentConfig.spike_recovery_long_alpha,
        ),
        spike_recovery_ratio=_env_float(
            "RRE_SPIKE_RECOVERY_RATIO",
            ExperimentConfig.spike_recovery_ratio,
        ),
        spike_recovery_abs_db=_env_float(
            "RRE_SPIKE_RECOVERY_ABS_DB",
            ExperimentConfig.spike_recovery_abs_db,
        ),
        spike_recovery_min_batches=_env_int(
            "RRE_SPIKE_RECOVERY_MIN_BATCHES",
            ExperimentConfig.spike_recovery_min_batches,
        ),
        spike_recovery_window_steps=_env_int(
            "RRE_SPIKE_RECOVERY_WINDOW_STEPS",
            ExperimentConfig.spike_recovery_window_steps,
        ),
        spike_recovery_accept_budget=_env_int(
            "RRE_SPIKE_RECOVERY_ACCEPT_BUDGET",
            ExperimentConfig.spike_recovery_accept_budget,
        ),
        spike_recovery_cooldown_steps=_env_int(
            "RRE_SPIKE_RECOVERY_COOLDOWN_STEPS",
            ExperimentConfig.spike_recovery_cooldown_steps,
        ),
        spike_recovery_beta_scale=_env_float(
            "RRE_SPIKE_RECOVERY_BETA_SCALE",
            ExperimentConfig.spike_recovery_beta_scale,
        ),
        spike_recovery_accept_prob=_env_float(
            "RRE_SPIKE_RECOVERY_ACCEPT_PROB",
            ExperimentConfig.spike_recovery_accept_prob,
        ),
        fidelity_grid_per_zone=_env_int(
            "RRE_FIDELITY_GRID_PER_ZONE",
            ExperimentConfig.fidelity_grid_per_zone,
        ),
        fidelity_grid_n_tx=_env_int(
            "RRE_FIDELITY_GRID_N_TX", ExperimentConfig.fidelity_grid_n_tx
        ),
        fidelity_refresh_every=_env_int(
            "RRE_FIDELITY_REFRESH_EVERY", ExperimentConfig.fidelity_refresh_every
        ),
        final_fidelity_grid_per_zone=_env_int(
            "RRE_FINAL_FIDELITY_GRID_PER_ZONE",
            ExperimentConfig.final_fidelity_grid_per_zone,
        ),
        fidelity_final_steps=_env_int_tuple(
            "RRE_FIDELITY_FINAL_STEPS", ExperimentConfig.fidelity_final_steps
        ),
        fidelity_eval_every=_env_int(
            "RRE_FIDELITY_EVAL_EVERY", ExperimentConfig.fidelity_eval_every
        ),
        fidelity_log_every=_env_int(
            "RRE_FIDELITY_LOG_EVERY", ExperimentConfig.fidelity_log_every
        ),
        epsilon_start=_env_float(
            "RRE_EPSILON_START", ExperimentConfig.epsilon_start
        ),
        epsilon_end=_env_float(
            "RRE_EPSILON_END", ExperimentConfig.epsilon_end
        ),
        epsilon_decay_steps=_env_int(
            "RRE_EPSILON_DECAY_STEPS", ExperimentConfig.epsilon_decay_steps
        ),
        rl_action_policy=_env_str(
            "RRE_RL_ACTION_POLICY", ExperimentConfig.rl_action_policy
        ),
        num_rays=_env_int("RRE_NUM_RAYS", ExperimentConfig.num_rays),
        trace_tx_batch_size=_env_int(
            "RRE_TRACE_TX_BATCH_SIZE", ExperimentConfig.trace_tx_batch_size
        ),
        tx_power_dbm=_env_float(
            "RRE_TX_POWER_DBM", ExperimentConfig.tx_power_dbm
        ),
        noise_floor_dbm=_env_float(
            "RRE_NOISE_FLOOR_DBM", ExperimentConfig.noise_floor_dbm
        ),
        snr_min_db=_env_float(
            "RRE_SNR_MIN_DB", ExperimentConfig.snr_min_db
        ),
        model_transfer_snr_min_db=_env_float_optional(
            "RRE_MODEL_TRANSFER_SNR_MIN_DB"
        ),
        legacy_rssi_gossip_threshold_dbm=(
            None
            if os.environ.get("RRE_SNR_MIN_DB") not in {None, ""}
            else _env_float_optional("RRE_RSSI_GOSSIP_THRESHOLD")
        ),
        rssi_model=_env_str("RRE_RSSI_MODEL", ExperimentConfig.rssi_model),
        mergeable_basis_dim=_env_int(
            "RRE_MERGEABLE_BASIS_DIM",
            ExperimentConfig.mergeable_basis_dim,
        ),
        mergeable_ridge=_env_float(
            "RRE_MERGEABLE_RIDGE", ExperimentConfig.mergeable_ridge
        ),
        merge_strategy=_env_str("RRE_MERGE_STRATEGY", ExperimentConfig.merge_strategy),
        predictor_include_time=bool(
            _env_int(
                "RRE_PREDICTOR_INCLUDE_TIME",
                int(ExperimentConfig.predictor_include_time),
            )
        ),
        predictor_zone_local_coordinates=bool(
            _env_int(
                "RRE_PREDICTOR_ZONE_LOCAL_COORDINATES",
                int(ExperimentConfig.predictor_zone_local_coordinates),
            )
        ),
        local_support_spatial_grid_points=_env_int(
            "RRE_LOCAL_SUPPORT_SPATIAL_GRID_POINTS",
            ExperimentConfig.local_support_spatial_grid_points,
        ),
        local_support_prior_strength=_env_float(
            "RRE_LOCAL_SUPPORT_PRIOR_STRENGTH",
            ExperimentConfig.local_support_prior_strength,
        ),
        predictor_time_step_duration=_env_float(
            "RRE_PREDICTOR_TIME_STEP_DURATION",
            ExperimentConfig.predictor_time_step_duration,
        ),
        predictor_time_unit=_env_float(
            "RRE_PREDICTOR_TIME_UNIT",
            _env_float(
                "RRE_PREDICTOR_TIME_SCALE_STEPS",
                ExperimentConfig.predictor_time_unit,
            ),
        ),
        predictor_time_num_frequencies=_env_int(
            "RRE_PREDICTOR_TIME_NUM_FREQUENCIES",
            ExperimentConfig.predictor_time_num_frequencies,
        ),
        predictor_time_min_period=_env_float(
            "RRE_PREDICTOR_TIME_MIN_PERIOD",
            ExperimentConfig.predictor_time_min_period,
        ),
        predictor_time_max_period=_env_float(
            "RRE_PREDICTOR_TIME_MAX_PERIOD",
            ExperimentConfig.predictor_time_max_period,
        ),
        predictor_learned_time_scale=_env_float(
            "RRE_PREDICTOR_LEARNED_TIME_SCALE",
            ExperimentConfig.predictor_learned_time_scale,
        ),
        predictor_prior=_env_str(
            "RRE_PREDICTOR_PRIOR",
            ExperimentConfig.predictor_prior,
        ),
        local_sample_weighting=_env_str(
            "RRE_LOCAL_SAMPLE_WEIGHTING",
            ExperimentConfig.local_sample_weighting,
        ),
        local_sample_recency_half_life_steps=_env_float(
            "RRE_LOCAL_SAMPLE_RECENCY_HALF_LIFE_STEPS",
            ExperimentConfig.local_sample_recency_half_life_steps,
        ),
        results_dir=_env_str("RRE_RESULTS_DIR", ExperimentConfig.results_dir),
    )
    # Apply explicit overrides last so CLI beats env.
    if overrides:
        normalized_overrides = dict(overrides)
        legacy_threshold = normalized_overrides.pop("rssi_gossip_threshold", None)
        if legacy_threshold is not None:
            normalized_overrides["legacy_rssi_gossip_threshold_dbm"] = legacy_threshold
        elif (
            "noise_floor_dbm" in normalized_overrides
            or "snr_min_db" in normalized_overrides
        ):
            normalized_overrides["legacy_rssi_gossip_threshold_dbm"] = None
        cfg = dataclasses.replace(cfg, **normalized_overrides)
    return cfg
