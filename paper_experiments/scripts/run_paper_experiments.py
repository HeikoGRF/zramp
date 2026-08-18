#!/usr/bin/env python3
"""Validate or run the final nine-map, five-window paper experiment matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ZONES = tuple(
    f"factor_b{building}_v{vehicles}_300m"
    for building in range(1, 4)
    for vehicles in range(1, 4)
)
METHODS = (
    "iso",
    "central",
    "full",
    "top5",
    "every1",
    "every5",
    "every10",
    "every20",
    "every40",
    "every80",
    "ungated",
)
INTERVALS = {
    "every1": 1,
    "every5": 5,
    "every10": 10,
    "every20": 20,
    "every40": 40,
    "every80": 80,
}


def parse_args() -> argparse.Namespace:
    paper_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-input-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paper_root / "results" / "reproduced",
        help="root for the four result trees understood by the paper aggregator",
    )
    parser.add_argument("--zones", nargs="+", choices=ZONES, default=list(ZONES))
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        choices=range(1, 6),
        default=list(range(1, 6)),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sim-steps", type=int, default=1799)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run sequentially; without this flag commands are validated and printed",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def result_directory(root: Path, zone: str, replicate: int, method: str) -> Path:
    short = zone.removesuffix("_300m")
    if method == "ungated":
        return (
            root
            / "luxembourg_cell_grid_ci_temporal5_paper_final_v1"
            / "methods"
            / short
            / f"rep{replicate}"
            / "ungated_equal_greedy_paper_final_full1800_eval50_tail10x25"
        )
    if replicate >= 2:
        return (
            root
            / "luxembourg_cell_grid_ci_temporal5_paper_final_v1"
            / "methods"
            / short
            / f"rep{replicate}"
            / f"{method}_paper_final_full1800_eval50_tail10x25"
        )
    if method in {"iso", "central"}:
        name = (
            "cell_grid_local_only_eval50_tail10x25"
            if method == "iso"
            else "cell_grid_central_eval50_tail10x25"
        )
        return root / "luxembourg_cell_grid_factorial_sweeps_v1" / "methods" / short / name
    if method in {"full", "top5", "every1"}:
        suffix = {"full": "greedy", "top5": "top5", "every1": "top1"}[method]
        return (
            root
            / "luxembourg_cell_grid_intensity_budget_sweep_v1"
            / "methods"
            / short
            / f"cell_grid_intensity_{suffix}_eval50_tail10x25"
        )
    interval = INTERVALS[method]
    return (
        root
        / "luxembourg_cell_grid_synchronized_9map_paper_final_v1"
        / "methods"
        / zone
        / (
            f"cell_grid_intensity_top1_global_every{interval}_"
            "paper_final_full1800_eval50_tail10x25"
        )
    )


def common_paths(paper_root: Path, generated_root: Path, zone: str, replicate: int):
    replicate_root = generated_root / "replicates" / zone / f"rep{replicate}"
    trace = require_file(
        replicate_root
        / "rssi"
        / (
            f"{zone}_rep{replicate}_vehicles_1s_opaque_no_vehicle_blockers_"
            "ge-100dbm_r20k_d3_llvm.npz"
        )
    )
    testset = require_file(
        paper_root
        / "input_data"
        / "prepared_traces"
        / "luxembourg_real_city"
        / f"{zone}_30min_opaque_buildings_no_vehicle_blockers"
        / "testset"
        / f"{zone}_street_pairs_10000_opaque_no_vehicle_blockers_static_floor100.npz"
    )
    net = require_file(
        paper_root
        / "input_data"
        / "maps"
        / "luxembourg_real_city"
        / "factorial_zones"
        / zone
        / "map"
        / "sionna"
        / f"{zone}_radio_bounds.net.xml"
    )
    return trace, testset, net


def build_command(
    args: argparse.Namespace,
    paper_root: Path,
    zone: str,
    replicate: int,
    method: str,
) -> tuple[list[str], Path]:
    trace, testset, net = common_paths(
        paper_root, args.generated_input_root.resolve(), zone, replicate
    )
    destination = result_directory(args.output_root.resolve(), zone, replicate, method)
    experiment_root = paper_root / "code" / "final" / "experiments" / "place_wallis_benchmark"
    base = [
        args.python,
        str(experiment_root / ("run_equal_greedy.py" if method == "ungated" else "run_support_expert_bank.py")),
        "--trace",
        str(trace),
        "--testset",
        str(testset),
        "--net",
        str(net),
        "--results-dir",
        str(destination),
        "--sim-steps",
        str(args.sim_steps),
        "--seed",
        str(args.seed),
        "--replay-capacity",
        "0",
        "--full-dataset-epochs",
        "1",
        "--checkpoint-every",
        "50",
        "--tail-eval-count",
        "10",
        "--tail-eval-stride",
        "25",
        "--progress-every",
        "10",
        "--method-tag",
        f"paper_{method}_rep{replicate}",
        "--resume-if-exists",
        "--quiet",
    ]
    if method == "ungated":
        base.extend(
            [
                "--local-lr",
                "5.0e-4",
                "--local-batch-size",
                "64",
                "--new-data-epochs",
                "2",
                "--replay-batches",
                "8",
                "--recent-replay-batches",
                "4",
                "--recent-window",
                "512",
                "--gradient-clip-norm",
                "1.0",
            ]
        )
        return base, destination

    base.extend(
        [
            "--bank-capacity",
            "1" if method in {"iso", "central"} else "6",
            "--transfer-cost",
            "0",
            "--probe-count",
            "512",
            "--angle-deg",
            "12",
            "--lateral-merge-m",
            "1",
            "--longitudinal-gap-m",
            "3",
            "--initial-half-width-m",
            "0",
            "--max-envelope-inflation",
            "1.2",
            "--max-corridor-width-m",
            "12",
            "--link-length-margin-m",
            "0",
            "--cell-grid-support",
            "--cell-grid-confidence",
            "binary",
            "--cell-grid-min-intensity",
            "1",
        ]
    )
    if method in {"iso", "central"}:
        base.extend(["--baseline-mode", "local-only" if method == "iso" else "central"])
        return base, destination

    bundle = require_file(
        paper_root
        / "trained_models"
        / "paper_runtime"
        / "cell_grid_patch_acquisition_v1_c16_pq4x256"
        / "bundle.pt"
    )
    budget = 0 if method == "full" else 5 if method == "top5" else 1
    interval = INTERVALS.get(method, 1)
    base.extend(
        [
            "--learned-acquisition-bundle",
            str(bundle),
            "--cell-grid-weighted-single",
            "--weighted-selection",
            "grid-intensity",
            "--weighted-pulls-per-receiver-step",
            str(budget),
            "--weighted-pull-interval-steps",
            str(interval),
            "--weighted-pull-schedule-anchor",
            "global",
        ]
    )
    return base, destination


def main() -> int:
    args = parse_args()
    paper_root = Path(__file__).resolve().parents[1]
    commands = []
    for zone in dict.fromkeys(args.zones):
        for replicate in dict.fromkeys(args.replicates):
            for method in dict.fromkeys(args.methods):
                commands.append(build_command(args, paper_root, zone, replicate, method))

    print(f"Validated inputs for {len(commands)} simulation runs.")
    if not args.execute:
        for command, _destination in commands:
            print(shlex.join(command))
        print("Dry run only. Add --execute to run these commands sequentially.")
        return 0

    code_root = paper_root / "code" / "final"
    for index, (command, destination) in enumerate(commands, start=1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(commands)}] {destination}", flush=True)
        subprocess.run(command, cwd=code_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

