#!/usr/bin/env python3
"""Run one controlled-map decision-policy ablation configuration."""

from __future__ import annotations

import argparse
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


def _cli() -> argparse.ArgumentParser:
    parser = _base_cli()
    parser.description = "Controlled four-zone zRAMP decision ablation"
    parser.add_argument(
        "--ablation-method",
        required=True,
        choices=[
            "iso",
            "greedy",
            "old-softmax",
            "unsigned-top-b",
            "signed-top-b",
            "current",
        ],
    )
    parser.set_defaults(
        predictor_prior="max-loss",
        policy_feature_set="study13",
        utility_hidden_dim=64,
        utility_horizon=10,
        aux_baselines="none",
    )
    return parser


def _simulation_common(args) -> dict:
    return {
        "sumo_config": args.sumo_config,
        "sumo_net": args.sumo_net,
        "dynamic_map": args.dynamic_map,
        "progress_every": int(args.progress_every),
        "log_rmse_every": int(args.log_rmse_every),
        "flush_every": int(args.flush_every),
        "max_wall_seconds": args.max_wall_seconds,
        "random_od_routing": True,
        "route_min_zone_distance": 1,
        "route_max_zone_distance": 1,
        "open_boundary_routing": True,
        "open_boundary_probability": 0.5,
        "open_boundary_margin": 0.12,
        "open_boundary_exit_margin": 0.035,
        "open_boundary_respawn_buffer": 2,
        "jam_reroute_wait_seconds": 25.0,
        "intersection_control": True,
        "intersection_wait_seconds": 12.0,
        "intersection_release_steps": 8,
        "intersection_stop_distance": 24.0,
        "zone_model_memory": True,
        "mobility_trace_in": args.mobility_trace_in,
        "measurement_trace_in": args.measurement_trace_in,
        "measurement_trace_out": args.measurement_trace_out,
        "trace_record_only": bool(args.trace_record_only),
    }


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    method = str(args.ablation_method)
    beta = float(args.beta)
    if method != "old-softmax" and beta != 0.0:
        raise ValueError("the positional beta must be zero except for old-softmax")
    if method in {"unsigned-top-b", "signed-top-b", "current"}:
        allowed_budgets = {1.0, 2.0, 3.0, 5.0}
        if method == "unsigned-top-b":
            allowed_budgets.update({7.0, 10.0})
        if float(args.pull_budget) not in allowed_budgets:
            raise ValueError(f"unsupported B={args.pull_budget:g} for {method}")
    if method == "old-softmax" and beta not in {0.0, 0.1, 0.25, 1.0}:
        raise ValueError("old-softmax beta must be one of 0, 0.1, 0.25, 1")

    from rl_reward_experiment.config import build_config_from_env
    from SUMO.sumo_rl import SumoT2Simulation, read_net_bounds
    from online_policy_learning.decision_ablation import (
        CompactCurrentUtilitySelectionSimulation,
        SequentialDecisionAblationSimulation,
    )

    results_dir = args.results_dir or (
        ROOT / "online_policy_learning" / "results" / method / f"seed_{args.seed:02d}"
    )
    net_bounds = read_net_bounds(args.sumo_net)
    map_size = max(net_bounds.width, net_bounds.height)
    mode = f"t2_b{beta:g}"
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(args.cars),
        num_zones=int(args.num_zones),
        sim_steps=int(args.sim_steps),
        map_size=float(map_size),
        beta=beta,
        active_modes=() if method in {"iso", "greedy"} else (mode,),
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
        local_sample_weighting="uniform",
        local_sample_recency_half_life_steps=float(
            args.local_sample_recency_half_life_steps
        ),
        rl_action_policy=str(args.rl_action_policy),
        fidelity_grid_per_zone=int(args.fidelity_pairs_per_zone),
        fidelity_eval_every=int(args.fidelity_eval_every),
        final_fidelity_grid_per_zone=int(args.final_fidelity_pairs_per_zone),
        fidelity_final_steps=_parse_steps(args.final_steps),
        fidelity_log_every=0,
        verbose=not args.quiet,
        pending_slot_cap=int(args.pending_slot_cap),
    )
    common = _simulation_common(args)
    if method in {"iso", "greedy"}:
        sim = SumoT2Simulation(
            cfg,
            aux_baselines=method,
            local_policy_share=False,
            **common,
        )
    else:
        utility_common = {
            **common,
            "aux_baselines": "none",
            "local_policy_share": False,
            "local_policy_initial_pull": str(args.local_policy_initial_pull),
            "local_policy_initial_pull_prob": args.local_policy_initial_pull_prob,
            "local_policy_updates_per_batch": int(
                args.local_policy_updates_per_batch
            ),
            "pull_budget": float(args.pull_budget),
            "utility_exploration_prob": float(args.utility_exploration_prob),
            "utility_evaluation": bool(args.utility_evaluation),
            "utility_hidden_dim": int(args.utility_hidden_dim),
            "utility_train_updates": int(args.utility_train_updates),
            "utility_horizon": int(args.utility_horizon),
            "utility_feedback_mode": str(args.utility_feedback_mode),
            "aggregation_experience_epsilon": float(
                args.aggregation_experience_epsilon
            ),
            "policy_state_features": str(args.policy_feature_set),
        }
        if method == "current":
            sim = CompactCurrentUtilitySelectionSimulation(cfg, **utility_common)
        else:
            sim = SequentialDecisionAblationSimulation(
                cfg,
                decision_rule=method,
                communication_penalty=beta,
                compact_decision_logging=True,
                **utility_common,
            )
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
