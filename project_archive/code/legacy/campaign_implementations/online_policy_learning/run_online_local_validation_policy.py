#!/usr/bin/env python3
"""Run exact-input learned-encoder sequential bidirectional aggregation."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_policy_learning.run import (  # noqa: E402
    _cli as _base_cli,
    _parse_steps,
    _snr_config_kwargs,
    _time_config_kwargs,
)


def _cli():
    parser = _base_cli()
    for action in parser._actions:
        if action.dest == "beta":
            action.dest = "pull_budget"
            action.metavar = "K"
            action.help = (
                "Per-receiver, per-timestep model-pull budget. Values below "
                "one activate one pull with probability K; values at least "
                "one must be integers."
            )
        elif "--pull-budget" in action.option_strings:
            action.help = argparse.SUPPRESS
    parser.description = (
        "Exact model-graph/full-trajectory online encoder with learned time "
        "and fixed-budget sequential bidirectional pulls"
    )
    parser.add_argument("--aggregation-tolerance", type=float, default=0.01)
    parser.add_argument("--aggregation-max-iterations", type=int, default=16)
    parser.add_argument("--validation-epsilon", type=float, default=1.0e-12)
    parser.add_argument(
        "--validation-capacity",
        type=int,
        default=1000,
        help="Combined per-vehicle private validation cap, split equally between optimization and reward reservoirs.",
    )
    parser.add_argument("--exact-hidden-dim", type=int, default=8)
    parser.add_argument("--gain-hidden-dim", type=int, default=8)
    parser.add_argument(
        "--pair-feature-mode",
        choices=("concat", "relational"),
        default="concat",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("policy", "random", "oracle", "novelty", "isolated"),
        default="policy",
    )
    parser.add_argument(
        "--aux-only",
        action="store_true",
        help=(
            "Train/evaluate only the requested auxiliary baselines; skip "
            "model-pull selection and exact-policy updates."
        ),
    )
    parser.add_argument("--accumulated-head-epoch", action="store_true")
    parser.add_argument("--head-replay-batches-per-step", type=int, default=0)
    parser.add_argument(
        "--policy-warmup-steps",
        type=int,
        default=0,
        help="Use scheduled uniform-random pulls for policy selection before this step.",
    )
    parser.add_argument(
        "--policy-warmup-pull-probability",
        type=float,
        default=1.0,
        help=(
            "Probability of taking an otherwise available random pull during "
            "--policy-warmup-steps. This controls warm-up data collection "
            "without imposing a post-warm-up pull budget."
        ),
    )
    parser.add_argument(
        "--share-policy-on-scheduled-links",
        action="store_true",
        help="Average policy parameters only across MAC-scheduled model-transfer links.",
    )
    parser.add_argument(
        "--share-training-samples",
        action="store_true",
        help=(
            "Keep private gain heads, train encoders locally, and gossip "
            "bounded deduplicated embedding-pair/gain samples."
        ),
    )

    parser.add_argument(
        "--policy-sample-capacity",
        type=int,
        default=512,
        help="Per-vehicle deterministic bottom-k policy-sample capacity.",
    )
    parser.add_argument(
        "--policy-sample-bundle-capacity",
        type=int,
        default=32,
        help="Maximum policy samples sent per direction and contact.",
    )
    parser.add_argument(
        "--policy-encoder-lr-scale",
        type=float,
        default=0.1,
        help="Encoder learning-rate multiplier relative to the private head.",
    )
    parser.add_argument(
        "--align-policy-encoders",
        action="store_true",
        help=(
            "Experience-average encoder parameters after every scheduled "
            "bilateral sample-gossip exchange; heads and replay remain private."
        ),
    )
    parser.add_argument(
        "--freeze-policy-encoders",
        action="store_true",
        help=(
            "Freeze the identical seeded encoders so gossiped embedding "
            "samples never become stale or incompatible."
        ),
    )
    parser.add_argument(
        "--pretrained-policy",
        type=Path,
        default=None,
        help="Load an exact encoder/gain-head checkpoint trained only on source maps.",
    )
    parser.add_argument(
        "--freeze-pretrained-policy",
        action="store_true",
        help="Disable all online encoder/head updates on the deployment map.",
    )
    parser.add_argument(
        "--normalize-policy-rewards",
        action="store_true",
        help=(
            "Legacy mode: train each private gain head on changing local "
            "standardized gains. Prefer --policy-reward-scale-db."
        ),
    )
    parser.add_argument(
        "--policy-reward-scale-db",
        type=float,
        default=None,
        help=(
            "Divide every observed RMSE-gain target by this fixed dB scale; "
            "predictions are converted back to dB for decisions and logs."
        ),
    )
    parser.add_argument(
        "--policy-reward-scope",
        choices=("directional", "joint"),
        default="directional",
        help=(
            "Train/rank on the initiating endpoint gain or the common "
            "quality-weighted bilateral private-validation gain."
        ),
    )
    parser.add_argument(
        "--policy-training-target",
        choices=(
            "validation-gain",
            "information-gain",
            "parameter-geometry",
        ),
        default="validation-gain",
        help=(
            "Train provider ranking on private-validation gain, delivered "
            "evidence novelty, or parameter geometry."
        ),
    )
    parser.add_argument(
        "--policy-ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight of conditional pairwise provider-ranking loss.",
    )
    parser.add_argument(
        "--policy-ranking-margin-db", type=float, default=0.25
    )
    parser.add_argument(
        "--policy-ranking-temperature-db", type=float, default=1.0
    )
    parser.add_argument(
        "--policy-ranking-receiver-cosine-min", type=float, default=0.8
    )
    parser.add_argument(
        "--policy-min-samples",
        type=int,
        default=0,
        help=(
            "Use uniform provider selection until this many deduplicated "
            "policy samples are available locally."
        ),
    )
    parser.add_argument(
        "--policy-exploration-start",
        type=float,
        default=None,
        help=(
            "Exploration probability when learned ranking starts; decays "
            "toward --exploration-prob as policy samples accumulate."
        ),
    )
    parser.add_argument(
        "--policy-exploration-decay-samples",
        type=int,
        default=0,
        help="Number of post-readiness samples over which exploration decays.",
    )
    parser.add_argument(
        "--trajectory-capacity",
        type=int,
        default=256,
        help="Spatially balanced per-AZ trajectory rows encoded by the policy.",
    )
    parser.add_argument(
        "--symmetric-pulls",
        action="store_true",
        help="Evaluate and conservatively install one aggregate at both endpoints.",
    )
    parser.add_argument(
        "--unconditional-evidence-union",
        action="store_true",
        help=(
            "Validation control: install mergeable evidence unions at both "
            "endpoints even when immediate private validation worsens."
        ),
    )
    parser.add_argument(
        "--mergeable-max-delta-rows",
        type=int,
        default=0,
        help=(
            "Maximum newer provenance rows sent per direction and pull; "
            "zero sends every newer row."
        ),
    )
    parser.add_argument(
        "--policy-reward-metric",
        choices=("normalized-improvement", "rmse-gain"),
        default="normalized-improvement",
    )

    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument(
        "--expert-certificate-grid-points", type=int, default=3
    )
    parser.add_argument(
        "--expert-certificate-min-samples", type=int, default=8
    )
    parser.add_argument(
        "--expert-certificate-epoch-steps",
        type=int,
        default=60,
        help=(
            "Certificate epoch in simulation steps; with the Gare 5-second "
            "trace, 60 steps are five minutes."
        ),
    )
    parser.add_argument(
        "--expert-bandit-exploration", type=float, default=0.35
    )
    parser.add_argument("--exploration-prob", type=float, default=0.02)
    parser.add_argument("--token-window-steps", type=int, default=10)
    parser.add_argument(
        "--visit-pull-budget",
        type=int,
        default=0,
        help=(
            "When positive, replace time windows with this many bilateral "
            "predictor exchanges per physical vehicle and contiguous AZ "
            "visit. Random samples contact frames uniformly from the replay."
        ),
    )
    parser.add_argument(
        "--policy-trigger-quantile",
        type=float,
        default=0.75,
        help=(
            "Policy spends a visit token only when its best predicted gain "
            "exceeds this quantile of its locally observed/gossiped gains."
        ),
    )
    parser.add_argument(
        "--policy-fixed-trigger-db",
        type=float,
        default=None,
        help=(
            "Use this fixed predicted net-gain threshold instead of a local "
            "gain quantile. Zero implements pull iff predicted gain exceeds cost."
        ),
    )
    parser.add_argument(
        "--allow-unused-policy-tokens",
        action="store_true",
        help=(
            "Let a policy token expire when no provider exceeds the learned "
            "or fixed gain threshold instead of forcing a window-end pull."
        ),
    )
    parser.add_argument(
        "--communication-penalty-db",
        type=float,
        default=0.0,
        help="Subtract this dB cost from every realized pull reward.",
    )
    parser.add_argument("--artificial-min-real-samples", type=int, default=128)
    parser.add_argument("--artificial-full-weight-samples", type=int, default=2048)
    parser.add_argument("--artificial-ratio", type=float, default=0.5)
    parser.add_argument("--artificial-support-min-m", type=float, default=7.5)
    parser.add_argument("--artificial-support-high-m", type=float, default=10.0)
    parser.add_argument("--artificial-candidate-pool", type=int, default=512)
    parser.add_argument(
        "--geometry-min-local-updates-between-merges",
        type=int,
        default=0,
        help=(
            "For parameter-geometry simulations, require this many local "
            "training updates after an adopted merge before adopting another."
        ),
    )
    parser.add_argument("--artificial-low-weight-start", type=float, default=0.10)
    parser.add_argument("--artificial-low-weight-end", type=float, default=0.50)
    parser.add_argument("--artificial-high-weight-start", type=float, default=0.25)
    parser.add_argument("--artificial-high-weight-end", type=float, default=1.00)
    parser.add_argument("--artificial-maturity-exponent", type=float, default=1.0)
    parser.add_argument(
        "--artificial-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include disjoint artificial floor rows in FedAvg optimization "
            "and pull-reward validation; disabling sends every artificial "
            "row to predictor training and keeps validation real-only."
        ),
    )
    parser.add_argument(
        "--artificial-max-new-per-vehicle-step", type=int, default=16
    )
    parser.add_argument(
        "--contact-aware-window-timing",
        action="store_true",
        help=(
            "Let random use the first feasible contact at or after its "
            "sampled window offset and force an unspent policy token at the "
            "window deadline. This preserves timing decisions while keeping "
            "pull counts comparable."
        ),
    )
    parser.add_argument("--learned-time-dim", type=int, default=16)
    parser.add_argument(
        "--learned-time-scale",
        type=float,
        default=None,
        help=(
            "Exact learned-time scale; defaults to "
            "--predictor-learned-time-scale."
        ),
    )
    parser.add_argument("--realistic-network", action="store_true")
    parser.add_argument(
        "--network-candidate-top-k",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility control: 0 considers every "
            "bidirectionally decodable contact; positive values impose a "
            "strongest-link shortlist."
        ),
    )
    parser.add_argument("--network-resource-count", type=int, default=4)
    parser.add_argument("--network-bandwidth-hz", type=float, default=10.0e6)
    parser.add_argument("--network-direction-airtime-s", type=float, default=0.125)
    parser.add_argument("--network-efficiency", type=float, default=0.6)
    parser.add_argument(
        "--network-max-spectral-efficiency", type=float, default=6.0
    )
    parser.add_argument("--network-min-sinr-db", type=float, default=5.0)
    parser.add_argument("--network-missing-power-dbm", type=float, default=-120.0)
    parser.add_argument(
        "--network-decentralized-reservation",
        action="store_true",
        help=(
            "Use deterministic local backoff plus request/grant fallback so "
            "each node participates in at most one proposal per 5-second frame."
        ),
    )
    parser.add_argument(
        "--network-reservation-control-bytes", type=int, default=32
    )
    parser.add_argument(
        "--diagnostic-regular-count",
        type=int,
        default=0,
        help="Number of leading node indices treated as regular vehicles for logging only.",
    )
    parser.add_argument(
        "--oracle-reward-pairs",
        type=int,
        default=256,
        help=(
            "For oracle-reward diagnostic simulations, reserve this many "
            "fixed route-density samples for post-pull reward labels."
        ),
    )
    parser.add_argument(
        "--oracle-reward-split-seed",
        type=int,
        default=20260727,
        help=(
            "Fixed seed used to stratify the oracle reward set away from the "
            "checkpoint test set."
        ),
    )
    parser.add_argument(
        "--oracle-router-checkpoint-pairs",
        type=int,
        default=512,
        help=(
            "Maximum checkpoint-test samples per vehicle used to audit expert "
            "router top-1 accuracy."
        ),
    )
    parser.set_defaults(
        aux_baselines="none",
        local_sample_weighting="uniform",
        rl_action_policy="argmax",
    )
    return parser


def _configure_cpu_threads() -> None:
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return
    threads = max(
        1,
        int(
            os.environ.get(
                "TORCH_NUM_THREADS",
                os.environ.get("SLURM_CPUS_PER_TASK", "1"),
            )
        ),
    )
    interop = max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    print(
        f"[EXACT-CPU] intraop_threads={threads} interop_threads={interop}",
        flush=True,
    )


def main(
    argv: list[str] | None = None,
    *,
    simulation_cls=None,
) -> int:
    args = _cli().parse_args(argv)
    _configure_cpu_threads()
    budget = float(args.pull_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("pull budget must be finite and positive")
    if budget >= 1.0 and not budget.is_integer():
        raise ValueError("pull budget must be an integer when it is at least one")
    if int(args.visit_pull_budget) < 0:
        raise ValueError("--visit-pull-budget must be nonnegative")
    if not 0.0 <= float(args.policy_warmup_pull_probability) <= 1.0:
        raise ValueError(
            "--policy-warmup-pull-probability must be in [0, 1]"
        )
    if not 0.0 <= float(args.policy_trigger_quantile) <= 1.0:
        raise ValueError("--policy-trigger-quantile must be in [0, 1]")
    if not math.isfinite(float(args.communication_penalty_db)) or float(
        args.communication_penalty_db
    ) < 0.0:
        raise ValueError("--communication-penalty-db must be finite and nonnegative")
    if int(args.visit_pull_budget) > 0 and bool(
        args.contact_aware_window_timing
    ):
        raise ValueError(
            "--visit-pull-budget replaces both token-window timing modes"
        )
    if str(args.merge_strategy) != "average":
        raise ValueError("exact sequential CV requires --merge-strategy average")
    if bool(args.aux_only) and str(args.aux_baselines).strip().lower() in {
        "",
        "none",
    }:
        raise ValueError("--aux-only requires --aux-baselines")
    if bool(args.share_policy_on_scheduled_links) and not bool(
        args.realistic_network
    ):
        raise ValueError(
            "--share-policy-on-scheduled-links requires --realistic-network"

        )
    if args.pretrained_policy is not None and str(args.selection_mode) != "policy":
        raise ValueError("--pretrained-policy requires --selection-mode policy")
    if bool(args.freeze_pretrained_policy) and args.pretrained_policy is None:
        raise ValueError("--freeze-pretrained-policy requires --pretrained-policy")
    if bool(args.share_training_samples) and str(
        args.selection_mode
    ) != "policy":
        raise ValueError(
            "--share-training-samples is only valid for policy selection"
        )
    if bool(args.align_policy_encoders) and not bool(
        args.share_training_samples
    ):
        raise ValueError(
            "--align-policy-encoders requires --share-training-samples"
        )
    if (
        bool(args.freeze_policy_encoders)
        or bool(args.normalize_policy_rewards)
        or args.policy_reward_scale_db is not None
        or float(args.policy_ranking_loss_weight) > 0.0
        or int(args.policy_min_samples) > 0
    ) and not bool(args.share_training_samples):
        raise ValueError(
            "frozen/normalized/readiness policy options require "
            "--share-training-samples"
        )
    if (
        bool(args.normalize_policy_rewards)
        and args.policy_reward_scale_db is not None
    ):
        raise ValueError(
            "changing local reward normalization and fixed reward scaling "
            "are mutually exclusive"
        )
    if bool(args.freeze_policy_encoders) and bool(
        args.align_policy_encoders
    ):
        raise ValueError(
            "--freeze-policy-encoders and --align-policy-encoders are "
            "mutually exclusive"
        )
    if simulation_cls is None:
        from online_policy_learning.online_local_validation_policy import (  # noqa: PLC0415
            ExactSequentialBidirectionalSimulation,
            SampleSharingExactSequentialSimulation,
        )

        simulation_cls = (
            SampleSharingExactSequentialSimulation
            if bool(args.share_training_samples)
            else ExactSequentialBidirectionalSimulation
        )
    from rl_reward_experiment.config import (  # noqa: PLC0415
        build_config_from_env,
        parse_modes,
    )
    from SUMO.sumo_rl import read_net_bounds  # noqa: PLC0415

    mode = "t2_b0"
    safe_budget = f"{budget:g}".replace(".", "p").replace("-", "m")
    results_dir = args.results_dir or (
        ROOT
        / "online_policy_learning"
        / "results"
        / "exact_sequential"
        / (
            f"seed_{int(args.seed):02d}_cars_{int(args.cars):03d}_"
            f"{str(args.selection_mode)}_budget_{safe_budget}"
        )
    )
    bounds = read_net_bounds(args.sumo_net)
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(args.cars),
        num_zones=int(args.num_zones),
        sim_steps=int(args.sim_steps),
        map_size=float(max(bounds.width, bounds.height)),
        beta=0.0,
        active_modes=parse_modes(mode),
        results_dir=str(results_dir),
        num_rays=int(args.num_rays),
        trace_tx_batch_size=int(args.trace_tx_batch_size),
        tx_power_dbm=float(args.tx_power_dbm),
        **_snr_config_kwargs(args),
        rssi_model=str(args.rssi_model),
        mergeable_basis_dim=int(args.mergeable_basis_dim),
        mergeable_ridge=float(args.mergeable_ridge),
        merge_strategy="average",
        **_time_config_kwargs(args),
        predictor_prior=str(args.predictor_prior),
        predictor_zone_local_coordinates=bool(
            args.predictor_zone_local_coordinates
        ),
        local_support_spatial_grid_points=int(args.local_support_spatial_grid_points),
        local_support_prior_strength=float(args.local_support_prior_strength),
        local_lr=float(args.local_lr),
        local_batch_size=int(args.local_batch_size),
        local_epochs=int(args.local_epochs),
        local_batches_per_step=int(args.local_batches_per_step),
        local_train_all_new_samples=bool(
            args.local_train_all_new_samples
        ),
        local_batches_per_step_max=int(
            args.local_batches_per_step_max
        ),
        local_batches_maturity_rows=int(
            args.local_batches_maturity_rows
        ),
        local_classifier_batches_per_step=int(
            args.local_classifier_batches_per_step
        ),
        local_classifier_hard_candidate_multiplier=int(
            args.local_classifier_hard_candidate_multiplier
        ),
        local_initialization_anchor_strength=float(
            args.local_initialization_anchor_strength
        ),
        local_spatial_balance_bins=int(
            args.local_spatial_balance_bins
        ),
        local_sample_weighting=str(args.local_sample_weighting),
        local_sample_recency_half_life_steps=float(
            args.local_sample_recency_half_life_steps
        ),
        rl_action_policy="argmax",
        fidelity_grid_per_zone=int(args.fidelity_pairs_per_zone),
        fidelity_eval_every=int(args.fidelity_eval_every),
        final_fidelity_grid_per_zone=int(args.final_fidelity_pairs_per_zone),
        fidelity_final_steps=_parse_steps(args.final_steps),
        fidelity_log_every=0,
        verbose=not args.quiet,
        spike_recovery_enabled=False,
    )
    expert_bank_kwargs = {}
    if getattr(simulation_cls, "__name__", "") == (
        "ExpertBankSampleSharingSimulation"
    ):
        expert_bank_kwargs = {
            "expert_bank_certificate_grid_points": int(
                args.expert_certificate_grid_points
            ),
            "expert_bank_certificate_min_samples": int(
                args.expert_certificate_min_samples
            ),
            "expert_bank_certificate_epoch_steps": int(
                args.expert_certificate_epoch_steps
            ),
            "expert_bank_bandit_exploration": float(
                args.expert_bandit_exploration
            ),
        }
    role_kwargs = {}
    simulation_mro_names = {
        getattr(base, "__name__", "")
        for base in getattr(simulation_cls, "__mro__", ())
    }
    if "RoleExactSequentialSimulation" in simulation_mro_names:
        role_kwargs = {
            "artificial_min_real_samples": int(args.artificial_min_real_samples),
            "artificial_full_weight_samples": int(
                args.artificial_full_weight_samples
            ),
            "artificial_ratio": float(args.artificial_ratio),
            "artificial_support_min_m": float(args.artificial_support_min_m),
            "artificial_support_high_m": float(args.artificial_support_high_m),
            "artificial_candidate_pool": int(args.artificial_candidate_pool),
            "artificial_low_weight_start": float(
                args.artificial_low_weight_start
            ),
            "artificial_low_weight_end": float(args.artificial_low_weight_end),
            "artificial_high_weight_start": float(
                args.artificial_high_weight_start
            ),
            "artificial_high_weight_end": float(
                args.artificial_high_weight_end
            ),
            "artificial_maturity_exponent": float(
                args.artificial_maturity_exponent
            ),
            "artificial_validation": bool(args.artificial_validation),
            "artificial_max_new_per_vehicle_step": int(
                args.artificial_max_new_per_vehicle_step
            ),
        }
    if "ParameterGeometryRoleSimulation" in simulation_mro_names:
        role_kwargs["geometry_min_local_updates_between_merges"] = int(
            args.geometry_min_local_updates_between_merges
        )
    if "KirchbergOracleRewardMixin" in simulation_mro_names:
        role_kwargs.update(
            {
                "oracle_reward_pairs": int(args.oracle_reward_pairs),
                "oracle_reward_split_seed": int(
                    args.oracle_reward_split_seed
                ),
                "oracle_router_checkpoint_pairs": int(
                    args.oracle_router_checkpoint_pairs
                ),
            }
        )
    simulation = simulation_cls(
        cfg,
        sumo_config=args.sumo_config,
        sumo_net=args.sumo_net,
        dynamic_map=args.dynamic_map,
        aux_baselines=str(args.aux_baselines),
        central_accumulate_samples=bool(
            args.central_accumulated_training
        ),
        progress_every=int(args.progress_every),
        log_rmse_every=int(args.log_rmse_every),
        flush_every=int(args.flush_every),
        max_wall_seconds=args.max_wall_seconds,
        random_od_routing=True,
        route_min_zone_distance=1,
        route_max_zone_distance=1,
        open_boundary_routing=True,
        open_boundary_probability=0.5,
        open_boundary_margin=0.12,
        open_boundary_exit_margin=0.035,
        open_boundary_respawn_buffer=2,
        jam_reroute_wait_seconds=25.0,
        intersection_control=True,
        intersection_wait_seconds=12.0,
        intersection_release_steps=8,
        intersection_stop_distance=24.0,
        zone_model_memory=True,
        local_policy_share=False,
        share_policy_every_contact=bool(args.share_policy_on_scheduled_links),
        local_policy_updates_per_batch=1,
        policy_temperature=1.0,
        hard_warmup_steps=0,
        hard_warmup_pull_probability=0.0,
        mobility_trace_in=args.mobility_trace_in,
        measurement_trace_in=args.measurement_trace_in,
        measurement_trace_out=args.measurement_trace_out,
        trace_record_only=bool(args.trace_record_only),
        policy_state_features="current6",
        communication_penalty=float(args.communication_penalty_db),
        aggregation_tolerance=float(args.aggregation_tolerance),
        aggregation_max_iterations=int(args.aggregation_max_iterations),
        validation_epsilon=float(args.validation_epsilon),
        validation_capacity=int(args.validation_capacity),
        exact_hidden_dim=int(args.exact_hidden_dim),
        gain_hidden_dim=int(args.gain_hidden_dim),
        pair_feature_mode=str(args.pair_feature_mode),
        pull_budget=budget,
        token_window_steps=int(args.token_window_steps),
        contact_aware_window_timing=bool(
            args.contact_aware_window_timing
        ),
        selection_mode=str(args.selection_mode),
        policy_warmup_steps=int(args.policy_warmup_steps),
        policy_warmup_pull_probability=float(
            args.policy_warmup_pull_probability
        ),
        train_accumulated_head_epoch=bool(args.accumulated_head_epoch),
        head_replay_batches_per_step=int(args.head_replay_batches_per_step),
        embedding_dim=int(args.embedding_dim),
        exploration_probability=float(args.exploration_prob),
        learned_time_dim=int(args.learned_time_dim),
        learned_time_scale=(
            float(args.predictor_learned_time_scale)
            if args.learned_time_scale is None
            else float(args.learned_time_scale)
        ),
        policy_sample_capacity=int(args.policy_sample_capacity),
        policy_sample_bundle_capacity=int(
            args.policy_sample_bundle_capacity
        ),
        encoder_lr_scale=float(args.policy_encoder_lr_scale),
        align_policy_encoders=bool(args.align_policy_encoders),
        freeze_policy_encoders=bool(args.freeze_policy_encoders),
        pretrained_policy_path=args.pretrained_policy,
        freeze_pretrained_policy=bool(args.freeze_pretrained_policy),
        normalize_policy_rewards=bool(args.normalize_policy_rewards),
        policy_reward_scale_db=args.policy_reward_scale_db,
        policy_reward_scope=str(args.policy_reward_scope),
        policy_training_target=str(args.policy_training_target),
        policy_ranking_loss_weight=float(
            args.policy_ranking_loss_weight
        ),
        policy_ranking_margin_db=float(args.policy_ranking_margin_db),
        policy_ranking_temperature_db=float(
            args.policy_ranking_temperature_db
        ),
        policy_ranking_receiver_cosine_min=float(
            args.policy_ranking_receiver_cosine_min
        ),
        policy_min_samples=int(args.policy_min_samples),
        policy_exploration_start=args.policy_exploration_start,
        policy_exploration_decay_samples=int(
            args.policy_exploration_decay_samples
        ),
        visit_pull_budget=int(args.visit_pull_budget),
        policy_trigger_quantile=float(args.policy_trigger_quantile),
        policy_fixed_trigger_db=args.policy_fixed_trigger_db,
        allow_unused_policy_tokens=bool(args.allow_unused_policy_tokens),
        trajectory_capacity=int(args.trajectory_capacity),
        symmetric_pulls=bool(args.symmetric_pulls),
        policy_reward_metric=str(args.policy_reward_metric),
        unconditional_evidence_union=bool(
            args.unconditional_evidence_union
        ),
        mergeable_max_delta_rows=int(args.mergeable_max_delta_rows),
        diagnostic_regular_count=int(args.diagnostic_regular_count),
        aux_only=bool(args.aux_only),
        realistic_network=bool(args.realistic_network),
        network_candidate_top_k=int(args.network_candidate_top_k),
        network_resource_count=int(args.network_resource_count),
        network_bandwidth_hz=float(args.network_bandwidth_hz),
        network_direction_airtime_s=float(args.network_direction_airtime_s),
        network_efficiency=float(args.network_efficiency),
        network_max_spectral_efficiency=float(
            args.network_max_spectral_efficiency
        ),
        network_min_sinr_db=float(args.network_min_sinr_db),
        network_missing_power_dbm=float(args.network_missing_power_dbm),
        network_decentralized_reservation=bool(
            args.network_decentralized_reservation
        ),
        network_reservation_control_bytes=int(
            args.network_reservation_control_bytes
        ),
        # Every valid pull has already paid for private validation and yields
        # a realized gain label, so retain all of them for online training.
        train_all_current_examples=True,
        **role_kwargs,
        **expert_bank_kwargs,
    )
    simulation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
