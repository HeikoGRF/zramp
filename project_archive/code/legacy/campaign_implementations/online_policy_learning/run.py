#!/usr/bin/env python3
"""Run the budgeted utility-selection bootstrap zRAMP variant.

The positional beta is retained in the mode/result identifier for compatibility
with existing sweeps. Pull capacity is configured independently with
``--pull-budget``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_beta(beta: float) -> str:
    return f"{beta:g}".replace(".", "p").replace("-", "m")


def _snr_config_kwargs(args) -> dict[str, float]:
    noise_floor_dbm = float(args.noise_floor_dbm)
    if args.rssi_gossip_threshold is None:
        snr_min_db = float(args.snr_min_db)
    else:
        snr_min_db = float(args.rssi_gossip_threshold) - noise_floor_dbm
    return {
        "noise_floor_dbm": noise_floor_dbm,
        "snr_min_db": snr_min_db,
        "model_transfer_snr_min_db": (
            snr_min_db
            if args.model_transfer_snr_min_db is None
            else float(args.model_transfer_snr_min_db)
        ),
    }


def _time_config_kwargs(args) -> dict[str, float | int | bool]:
    duration = float(args.predictor_time_step_duration)
    unit = float(args.predictor_time_unit)
    minimum = float(args.predictor_time_min_period)
    maximum = (
        float(args.predictor_time_max_period)
        if args.predictor_time_max_period is not None
        else max(float(args.sim_steps) * duration / unit, minimum * 2.0)
    )
    return {
        "predictor_include_time": bool(args.predictor_include_time),
        "predictor_time_step_duration": duration,
        "predictor_time_unit": unit,
        "predictor_time_num_frequencies": int(args.predictor_time_frequencies),
        "predictor_time_min_period": minimum,
        "predictor_time_max_period": maximum,
        "predictor_learned_time_scale": float(
            args.predictor_learned_time_scale
        ),
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Budgeted utility-selection zRAMP SUMO simulation")
    p.add_argument("seed", type=int)
    p.add_argument("cars", type=int)
    p.add_argument("beta", type=float)
    p.add_argument("--sim-steps", type=int, default=1000)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--sumo-config", default="SUMO/controlled_4zone_300/controlled_4zone_300.sumocfg")
    p.add_argument("--sumo-net", default="SUMO/controlled_4zone_300/controlled_4zone_300.net.xml")
    p.add_argument("--dynamic-map", default="SUMO/controlled_4zone_300/controlled_4zone_300_dynamic.json")
    p.add_argument("--num-zones", type=int, default=4)
    p.add_argument("--num-rays", type=int, default=100_000)
    p.add_argument("--trace-tx-batch-size", type=int, default=32)
    p.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=-105.0,
        help="Receiver noise floor used for SNR feasibility, in dBm.",
    )
    p.add_argument(
        "--snr-min-db",
        type=float,
        default=5.0,
        help="Minimum SNR in both directions for a same-zone contact to be feasible.",
    )
    p.add_argument(
        "--model-transfer-snr-min-db",
        type=float,
        default=None,
        help=(
            "Minimum SNR in both directions for same-zone model transfer; "
            "defaults to --snr-min-db."
        ),
    )
    p.add_argument(
        "--rssi-gossip-threshold",
        type=float,
        default=None,
        help="Deprecated compatibility alias: received-power threshold in dBm; converted to SNR.",
    )
    p.add_argument("--rssi-model", default="small")
    p.add_argument(
        "--mergeable-basis-dim",
        type=int,
        default=192,
        help="Frozen random-feature width for mergeable evidence predictors.",
    )
    p.add_argument(
        "--mergeable-ridge",
        type=float,
        default=1.0,
        help="Ridge regularization for mergeable evidence predictors.",
    )
    p.set_defaults(predictor_include_time=False)
    p.add_argument(
        "--predictor-time",
        dest="predictor_include_time",
        action="store_true",
        help="Use learnable Fourier encoding of global absolute simulation time.",
    )
    p.add_argument(
        "--no-predictor-time",
        dest="predictor_include_time",
        action="store_false",
        help="Use only the four normalized transmitter/receiver coordinates.",
    )
    p.add_argument("--predictor-time-step-duration", type=float, default=1.0)
    p.add_argument(
        "--predictor-time-unit",
        "--predictor-time-scale-steps",
        dest="predictor_time_unit",
        type=float,
        default=1.0,
        help="Fixed physical unit conversion for time; never the simulation horizon.",
    )
    p.add_argument("--predictor-time-frequencies", type=int, default=8)
    p.add_argument("--predictor-time-min-period", type=float, default=2.0)
    p.add_argument(
        "--predictor-time-max-period",
        type=float,
        default=None,
        help="Longest initialized period in converted time units; defaults to the observable run duration.",
    )
    p.add_argument(
        "--predictor-learned-time-scale",
        type=float,
        default=1000.0,
        help="Fixed physical scale used by the learned scalar time encoder.",
    )
    p.add_argument(
        "--predictor-prior",
        choices=["snr-threshold", "max-loss", "none"],
        default="snr-threshold",
        help="Initial propagation-loss prior before any received samples are observed.",
    )
    p.add_argument(
        "--predictor-zone-local-coordinates",
        action="store_true",
        help="Normalize TX/RX coordinates inside the active square AZ.",
    )
    p.add_argument(
        "--local-support-spatial-grid-points",
        type=int,
        default=9,
        help="Grid points per endpoint-coordinate dimension for local-support maps.",
    )
    p.add_argument(
        "--local-support-prior-strength",
        type=float,
        default=0.0,
        help="Pseudo-support required before local-grid deltas override the prior.",
    )
    p.add_argument("--local-lr", type=float, default=1.0e-3)
    p.add_argument("--local-batch-size", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument(
        "--local-batches-per-step",
        type=int,
        default=0,
        help="Fixed optimizer updates per online fit; zero keeps full epochs.",
    )
    p.add_argument(
        "--local-train-all-new-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Visit every newly received predictor row once before the "
            "configured random replay minibatches."
        ),
    )
    p.add_argument(
        "--local-batches-per-step-max",
        type=int,
        default=0,
        help=(
            "Maximum online minibatches after replay maturity; zero disables "
            "the maturity schedule."
        ),
    )
    p.add_argument(
        "--local-batches-maturity-rows",
        type=int,
        default=0,
        help=(
            "Retained predictor rows at which the maximum minibatch budget is "
            "reached; zero disables the maturity schedule."
        ),
    )
    p.add_argument(
        "--local-classifier-batches-per-step",
        type=int,
        default=0,
        help=(
            "Additional balanced hard-example classifier-only minibatches per "
            "online fit; zero disables them."
        ),
    )
    p.add_argument(
        "--local-classifier-hard-candidate-multiplier",
        type=int,
        default=4,
        help="Candidate-pool multiplier used for classifier hard-example replay.",
    )
    p.add_argument(
        "--local-initialization-anchor-strength",
        type=float,
        default=0.0,
        help="L2-SP strength toward the common initialized predictor.",
    )
    p.add_argument(
        "--local-spatial-balance-bins",
        type=int,
        default=4,
        help="Bins per normalized TX/RX coordinate for replay balancing.",
    )

    p.add_argument(
        "--local-sample-weighting",
        choices=[
            "uniform",
            "exponential-recency",
            "spatial-balanced",
        ],
        default="uniform",
        help="How retained local predictor samples are weighted.",
    )
    p.add_argument(
        "--local-sample-recency-half-life-steps",
        type=float,
        default=50.0,
        help="Half-life for exponential-recency sample weights, in simulation steps.",
    )
    p.add_argument("--tx-power-dbm", type=float, default=23.0)
    p.add_argument("--merge-strategy", choices=["average", "ot"], default="average")
    p.add_argument("--aux-baselines", default="all")
    p.add_argument(
        "--central-accumulated-training",
        action="store_true",
        help="Train each central zone model on its complete accumulated feasible-sample buffer.",
    )
    p.add_argument("--rl-action-policy", choices=["softmax", "argmax", "reject", "accept"], default="softmax")
    p.add_argument(
        "--policy-feature-set",
        choices=["current6", "study10", "study13", "best5", "probe_free14"],
        default="current6",
        help="Utility-policy observation schema.",
    )
    p.add_argument(
        "--local-policy-initial-pull",
        choices=["greedy", "byte-match", "fixed"],
        default="byte-match",
        help="Compatibility option retained for older runners; continuous gain uses its warm-up pull probability.",
    )
    p.add_argument("--local-policy-initial-pull-prob", type=float, default=None)
    p.add_argument("--local-policy-updates-per-batch", type=int, default=1)
    p.add_argument(
        "--continuous-warmup-pull-prob",
        type=float,
        default=0.5,
        help="Exploratory pull probability before a node has trained its continuous gain policy.",
    )
    p.add_argument(
        "--continuous-weight-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian exploration noise for actor-selected merge weights after warm-up.",
    )
    p.add_argument(
        "--continuous-actor-lr-scale",
        type=float,
        default=0.5,
        help="Actor optimizer learning-rate multiplier relative to RRE_RL_LR.",
    )
    p.add_argument(
        "--pull-budget",
        type=float,
        default=1.0,
        help="Expected full-model download slots per receiver and step.",
    )
    p.add_argument(
        "--utility-exploration-prob",
        type=float,
        default=0.1,
        help="Probability of random feasible selection after utility warm-up.",
    )
    p.add_argument(
        "--utility-evaluation",
        action="store_true",
        help="Disable random and warm-up selection; use positive top-K only.",
    )
    p.add_argument("--utility-hidden-dim", type=int, default=64)
    p.add_argument("--utility-train-updates", type=int, default=4)
    p.add_argument("--utility-horizon", type=int, default=2)
    p.add_argument(
        "--utility-feedback-mode",
        choices=["frozen", "finetune-window"],
        default="frozen",
        help="Compare frozen snapshots or equally fine-tuned initializations on one common window.",
    )
    p.add_argument("--pending-slot-cap", type=int, default=32)
    p.add_argument("--aggregation-steps", type=int, default=8)
    p.add_argument("--aggregation-lr", type=float, default=0.1)
    p.add_argument("--aggregation-kl", type=float, default=0.01)
    p.add_argument("--aggregation-experience-epsilon", type=float, default=1.0)
    p.add_argument("--fidelity-eval-every", type=int, default=50)
    p.add_argument("--fidelity-pairs-per-zone", type=int, default=50)
    p.add_argument("--final-fidelity-pairs-per-zone", type=int, default=200)
    p.add_argument("--final-steps", default="900,925,950,975,1000")
    p.add_argument("--measurement-trace-in", default=None)
    p.add_argument("--mobility-trace-in", default=None)
    p.add_argument("--measurement-trace-out", default=None)
    p.add_argument("--trace-record-only", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--log-rmse-every", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--max-wall-seconds", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _parse_steps(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if text.lower() in {"", "none", "off"}:
        return ()
    return tuple(int(p.strip()) for p in text.replace(";", ",").split(",") if p.strip())


def main(argv: list[str] | None = None) -> int:
    del argv
    raise RuntimeError(
        "The legacy utility-selection runner was removed; use "
        "online_policy_learning.run_online_local_validation_policy or "
        "online_policy_learning.run_expert_bank."
    )


if __name__ == "__main__":
    raise SystemExit(main())
