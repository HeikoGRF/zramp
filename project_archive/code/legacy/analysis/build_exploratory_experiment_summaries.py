#!/usr/bin/env python3
"""Build compact review tables for the archived experimental campaigns."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = ARCHIVE_ROOT.parent
RESULT_ROOT = ARCHIVE_ROOT / "results" / "legacy_experiments"
OUTPUT_ROOT = (
    ARCHIVE_ROOT / "figures" / "data" / "exploratory_experiment_summaries"
)

FACTORIAL_MAPS = tuple(
    f"factor_b{building}_v{vehicles}"
    for building in range(1, 4)
    for vehicles in range(1, 4)
)

# This development comparison shares the first temporal realization and the
# central/local reference runs with the final paper workflow. Its raw files
# therefore remain under paper_experiments and are indexed here rather than
# duplicated inside project_archive.
SINGLE_MODEL_COMPARISON_METHODS = (
    {
        "method_id": "centralized_reference",
        "method_label": "Centralized reference",
        "method_family": "reference",
        "root": "factorial",
        "run_name": "cell_grid_central_eval50_tail10x25",
        "note": "Idealized accuracy reference; model-transfer count is not a communication estimate.",
    },
    {
        "method_id": "intensity_top1",
        "method_label": "Grid intensity, Top-1",
        "method_family": "deterministic_grid_intensity",
        "root": "intensity",
        "run_name": "cell_grid_intensity_top1_eval50_tail10x25",
        "note": "Final deterministic selection and merge rule at one pull per event.",
    },
    {
        "method_id": "intensity_top5",
        "method_label": "Grid intensity, Top-5",
        "method_family": "deterministic_grid_intensity",
        "root": "intensity",
        "run_name": "cell_grid_intensity_top5_eval50_tail10x25",
        "note": "At most five highest-intensity models per event.",
    },
    {
        "method_id": "intensity_all",
        "method_label": "Grid intensity, all available",
        "method_family": "deterministic_grid_intensity",
        "root": "intensity",
        "run_name": "cell_grid_intensity_greedy_eval50_tail10x25",
        "note": "All available models are pulled at each event.",
    },
    {
        "method_id": "sample_count_top1",
        "method_label": "Sample count, Top-1",
        "method_family": "deterministic_sample_count",
        "root": "factorial",
        "run_name": "cell_grid_weighted_pull1_eval50_tail10x25",
        "note": "At most one model, ranked and merged by cumulative sample count.",
    },
    {
        "method_id": "sample_count_top2",
        "method_label": "Sample count, Top-2",
        "method_family": "deterministic_sample_count",
        "root": "factorial",
        "run_name": "cell_grid_weighted_pull2_eval50_tail10x25",
        "note": "At most two models, ranked and merged by cumulative sample count.",
    },
    {
        "method_id": "sample_count_top4",
        "method_label": "Sample count, Top-4",
        "method_family": "deterministic_sample_count",
        "root": "factorial",
        "run_name": "cell_grid_weighted_pull4_eval50_tail10x25",
        "note": "At most four models, ranked and merged by cumulative sample count.",
    },
    {
        "method_id": "sample_count_all",
        "method_label": "Sample count, all available",
        "method_family": "deterministic_sample_count",
        "root": "factorial",
        "run_name": "cell_grid_weighted_greedy_eval50_tail10x25",
        "note": "All available models are pulled and merged by cumulative sample count.",
    },
    {
        "method_id": "learned_expert_bank_kappa2",
        "method_label": "Learned Expert Bank, kappa=0.02",
        "method_family": "pretrained_learned_acquisition",
        "root": "factorial",
        "run_name": "cell_grid_expert_kappa2_eval50_tail10x25",
        "note": "Only eight of nine maps completed; factor_b2_v3 has no final result.",
    },
    {
        "method_id": "learned_expert_bank_kappa10",
        "method_label": "Learned Expert Bank, kappa=0.10",
        "method_family": "pretrained_learned_acquisition",
        "root": "factorial",
        "run_name": "cell_grid_expert_kappa10_eval50_tail10x25",
        "note": "Pretrained relative-gain acquisition with an uncapped expert bank.",
    },
    {
        "method_id": "learned_expert_bank_kappa50",
        "method_label": "Learned Expert Bank, kappa=0.50",
        "method_family": "pretrained_learned_acquisition",
        "root": "factorial",
        "run_name": "cell_grid_expert_kappa50_eval50_tail10x25",
        "note": "Pretrained relative-gain acquisition with an uncapped expert bank.",
    },
    {
        "method_id": "local_only",
        "method_label": "Local only",
        "method_family": "reference",
        "root": "factorial",
        "run_name": "cell_grid_local_only_eval50_tail10x25",
        "note": "No peer model sharing.",
    },
)


def read_final_rmse(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "t2_b0_total" in payload:
        return float(payload["t2_b0_total"])
    legacy_keys = [
        key for key in payload if re.fullmatch(r"t2_b[^_]+_total", key)
    ]
    if len(legacy_keys) != 1:
        raise RuntimeError(f"cannot identify final RMSE key in {path}")
    return float(payload[legacy_keys[0]])


def write_csv(name: str, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path = OUTPUT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_contact_timing() -> None:
    root = RESULT_ROOT / "contact_timing_sweep"
    grouped: dict[tuple[int, int, int, str], list[float]] = defaultdict(list)
    paths = sorted(root.glob("n*_r*/s*/*/seed_*/final_fidelity.json"))
    for path in paths:
        condition, setting, variant, _seed, _filename = path.relative_to(root).parts
        match = re.fullmatch(r"n(\d+)_r(\d+)", condition)
        if match is None or not setting.startswith("s"):
            raise RuntimeError(f"unexpected contact-timing path: {path}")
        key = (
            int(match.group(1)),
            int(match.group(2)),
            int(setting.removeprefix("s")),
            variant,
        )
        grouped[key].append(read_final_rmse(path))
    if len(paths) != 180 or len(grouped) != 90:
        raise RuntimeError(
            f"expected 180 contact runs and 90 groups; found {len(paths)} and {len(grouped)}"
        )
    rows: list[dict[str, object]] = []
    for (nodes, radius, step_setting, variant), values in sorted(grouped.items()):
        rows.append(
            {
                "nodes": nodes,
                "r_setting": radius,
                "step_setting": step_setting,
                "variant": variant,
                "seeds": len(values),
                "final_rmse_mean_db": f"{fmean(values):.6f}",
                "final_rmse_sample_sd_db": f"{stdev(values):.6f}",
            }
        )
    write_csv(
        "contact_timing_sweep_summary.csv",
        rows,
        (
            "nodes",
            "r_setting",
            "step_setting",
            "variant",
            "seeds",
            "final_rmse_mean_db",
            "final_rmse_sample_sd_db",
        ),
    )


def summarize_four_zone_ablation() -> None:
    root = RESULT_ROOT / "controlled_four_zone_decision_ablation"
    grouped: dict[str, list[float]] = defaultdict(list)
    paths = sorted(root.glob("seed_*/*/final_fidelity.json"))
    for path in paths:
        _seed, variant, _filename = path.relative_to(root).parts
        grouped[variant].append(read_final_rmse(path))
    if len(paths) != 39 or len(grouped) != 13:
        raise RuntimeError(
            f"expected 39 four-zone runs and 13 variants; found {len(paths)} and {len(grouped)}"
        )
    rows = [
        {
            "variant": variant,
            "seeds": len(values),
            "final_rmse_mean_db": f"{fmean(values):.6f}",
            "final_rmse_sample_sd_db": f"{stdev(values):.6f}",
        }
        for variant, values in sorted(grouped.items())
    ]
    write_csv(
        "four_zone_decision_ablation_summary.csv",
        rows,
        (
            "variant",
            "seeds",
            "final_rmse_mean_db",
            "final_rmse_sample_sd_db",
        ),
    )


def summarize_cross_map_holdouts() -> None:
    specifications = (
        ("cross_map_online_policy_generalization", "sequential_unseen", 3),
        ("cross_map_aligned_policy_generalization", "sequential_unseen_holdouts_v2", 6),
    )
    rows: list[dict[str, object]] = []
    for campaign, evaluation, expected_runs in specifications:
        root = RESULT_ROOT / campaign / evaluation
        campaign_values: dict[str, list[float]] = defaultdict(list)
        paths = sorted(root.glob("source_test_*/*/final_fidelity.json"))
        if len(paths) != expected_runs:
            raise RuntimeError(
                f"expected {expected_runs} holdout runs in {root}; found {len(paths)}"
            )
        for path in paths:
            map_name, method, _filename = path.relative_to(root).parts
            value = read_final_rmse(path)
            campaign_values[method].append(value)
            rows.append(
                {
                    "campaign": campaign,
                    "evaluation": evaluation,
                    "map": map_name,
                    "method": method,
                    "final_rmse_db": f"{value:.6f}",
                    "maps_in_summary": "",
                    "across_map_mean_rmse_db": "",
                    "across_map_sample_sd_db": "",
                }
            )
        for method, values in sorted(campaign_values.items()):
            rows.append(
                {
                    "campaign": campaign,
                    "evaluation": evaluation,
                    "map": "ALL_MAPS",
                    "method": method,
                    "final_rmse_db": "",
                    "maps_in_summary": len(values),
                    "across_map_mean_rmse_db": f"{fmean(values):.6f}",
                    "across_map_sample_sd_db": f"{stdev(values):.6f}",
                }
            )
    write_csv(
        "cross_map_policy_holdout_summary.csv",
        rows,
        (
            "campaign",
            "evaluation",
            "map",
            "method",
            "final_rmse_db",
            "maps_in_summary",
            "across_map_mean_rmse_db",
            "across_map_sample_sd_db",
        ),
    )


def summarize_online_local_validation_policy() -> None:
    root = RESULT_ROOT / "online_local_validation_policy" / "runs"
    grouped: dict[
        tuple[str, int, int, int, str], dict[int, tuple[float, dict[str, object]]]
    ] = defaultdict(dict)
    paths = sorted(root.glob("*/n*_r*/s*/*/seed_*/final_fidelity.json"))
    for path in paths:
        campaign, condition, setting, selection_mode, seed_name, _filename = (
            path.relative_to(root).parts
        )
        condition_match = re.fullmatch(r"n(\d+)_r(\d+)", condition)
        setting_match = re.fullmatch(r"s(\d+)", setting)
        seed_match = re.fullmatch(r"seed_(\d+)", seed_name)
        if condition_match is None or setting_match is None or seed_match is None:
            raise RuntimeError(f"unexpected online-policy path: {path}")

        run_dir = path.parent
        with (run_dir / "progress.json").open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if progress.get("reason") != "completed":
            raise RuntimeError(f"non-completed run was archived: {run_dir}")
        with (run_dir / "learning_summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            learning = json.load(handle)

        key = (
            campaign,
            int(condition_match.group(1)),
            int(condition_match.group(2)),
            int(setting_match.group(1)),
            selection_mode,
        )
        seed = int(seed_match.group(1))
        grouped[key][seed] = (read_final_rmse(path), learning)

    if len(paths) != 46 or len(grouped) != 23:
        raise RuntimeError(
            f"expected 46 completed online-policy runs and 23 groups; "
            f"found {len(paths)} and {len(grouped)}"
        )
    if any(len(values) != 2 for values in grouped.values()):
        raise RuntimeError("every archived online-policy group must contain two seeds")

    rows: list[dict[str, object]] = []
    for key, seed_values in sorted(grouped.items()):
        campaign, nodes, radius_setting, token_window_steps, selection_mode = key
        rmse_values = [value[0] for _, value in sorted(seed_values.items())]
        learning_rows = [value[1] for _, value in sorted(seed_values.items())]

        paired_delta = ""
        comparator_key = (
            campaign,
            nodes,
            radius_setting,
            token_window_steps,
            "random",
        )
        if selection_mode != "random" and comparator_key in grouped:
            comparator = grouped[comparator_key]
            if set(comparator) != set(seed_values):
                raise RuntimeError(f"seed mismatch for paired comparison: {key}")
            deltas = [
                seed_values[seed][0] - comparator[seed][0]
                for seed in sorted(seed_values)
            ]
            paired_delta = f"{fmean(deltas):.6f}"

        def mean_optional(field: str) -> str:
            values = [
                float(row[field])
                for row in learning_rows
                if row.get(field) is not None
            ]
            return f"{fmean(values):.6f}" if values else ""

        rows.append(
            {
                "campaign": campaign,
                "nodes": nodes,
                "r_setting": radius_setting,
                "token_window_steps": token_window_steps,
                "selection_mode": selection_mode,
                "seeds": len(rmse_values),
                "final_rmse_mean_db": f"{fmean(rmse_values):.6f}",
                "final_rmse_sample_sd_db": f"{stdev(rmse_values):.6f}",
                "paired_mean_delta_vs_uninformed_db": paired_delta,
                "attempts_mean": mean_optional("attempts"),
                "gain_prediction_rmse_mean_db": mean_optional("gain_prediction_rmse"),
                "gain_prediction_pearson_mean": mean_optional(
                    "gain_prediction_pearson"
                ),
                "gain_sign_accuracy_mean": mean_optional("gain_sign_accuracy"),
            }
        )

    write_csv(
        "online_local_validation_policy_summary.csv",
        rows,
        (
            "campaign",
            "nodes",
            "r_setting",
            "token_window_steps",
            "selection_mode",
            "seeds",
            "final_rmse_mean_db",
            "final_rmse_sample_sd_db",
            "paired_mean_delta_vs_uninformed_db",
            "attempts_mean",
            "gain_prediction_rmse_mean_db",
            "gain_prediction_pearson_mean",
            "gain_sign_accuracy_mean",
        ),
    )



def read_run_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_kirchberg_online_private_validation_policy() -> None:
    root = RESULT_ROOT / "kirchberg_online_private_validation_policy"
    inventory = read_run_inventory(root / "run_inventory.csv")
    if len(inventory) != 3 or any(row["status"] != "complete" for row in inventory):
        raise RuntimeError("expected three completed Kirchberg learned-policy runs")
    if any(row["method"] != "learned_policy" for row in inventory):
        raise RuntimeError("Kirchberg archive must contain learned-policy runs only")

    seed_values = {
        int(row["seed"]): float(row["final_rmse_db"])
        for row in inventory
    }
    if set(seed_values) != {1, 2, 3}:
        raise RuntimeError("expected Kirchberg seeds 1, 2, and 3")
    values = [seed_values[seed] for seed in sorted(seed_values)]
    rows = [
        {
            "campaign": "kirchberg_online_private_validation_policy",
            "map": "Kirchberg North, Luxembourg",
            "method": "learned_policy",
            "seeds": len(values),
            "final_rmse_mean_db": f"{fmean(values):.6f}",
            "final_rmse_sample_sd_db": f"{stdev(values):.6f}",
        }
    ]
    write_csv(
        "kirchberg_online_private_validation_policy_summary.csv",
        rows,
        (
            "campaign",
            "map",
            "method",
            "seeds",
            "final_rmse_mean_db",
            "final_rmse_sample_sd_db",
        ),
    )


def summarize_receiver_in_zone_global_sender_policy_sweep() -> None:
    root = RESULT_ROOT / "receiver_in_zone_global_sender_policy_sweep"
    inventory = read_run_inventory(root / "run_inventory.csv")
    if len(inventory) != 32:
        raise RuntimeError(f"expected 32 real-map sweep attempts; found {len(inventory)}")
    status_counts: dict[str, int] = defaultdict(int)
    grouped: dict[tuple[int, int, str], dict[int, float]] = defaultdict(dict)
    attempted: dict[tuple[int, int, str], int] = defaultdict(int)
    for row in inventory:
        key = (
            int(row["pulls_per_receiver_step"]),
            int(row["token_window_steps"]),
            row["method"],
        )
        attempted[key] += 1
        status_counts[row["status"]] += 1
        if row["status"] == "complete":
            grouped[key][int(row["seed"])] = float(row["final_rmse_db"])
    if status_counts != {"complete": 30, "incomplete": 2}:
        raise RuntimeError(f"unexpected real-map sweep completion counts: {dict(status_counts)}")

    rows: list[dict[str, object]] = []
    for key in sorted(attempted):
        pulls, window, method = key
        seed_values = grouped.get(key, {})
        values = [seed_values[seed] for seed in sorted(seed_values)]
        paired_delta = ""
        better_seeds = ""
        comparator = grouped.get((pulls, window, "uninformed_selection"), {})
        if method == "learned_policy" and seed_values and set(seed_values) == set(comparator):
            deltas = [seed_values[seed] - comparator[seed] for seed in sorted(seed_values)]
            paired_delta = f"{fmean(deltas):.6f}"
            better_seeds = sum(delta < 0 for delta in deltas)
        rows.append(
            {
                "pulls_per_receiver_step": pulls,
                "token_window_steps": window,
                "method": method,
                "attempted_runs": attempted[key],
                "completed_runs": len(values),
                "incomplete_runs": attempted[key] - len(values),
                "final_rmse_mean_db": f"{fmean(values):.6f}" if values else "",
                "final_rmse_sample_sd_db": (
                    f"{stdev(values):.6f}" if len(values) > 1 else ""
                ),
                "paired_mean_delta_vs_uninformed_db": paired_delta,
                "paired_seeds_with_lower_rmse": better_seeds,
            }
        )
    write_csv(
        "receiver_in_zone_global_sender_policy_sweep_summary.csv",
        rows,
        (
            "pulls_per_receiver_step",
            "token_window_steps",
            "method",
            "attempted_runs",
            "completed_runs",
            "incomplete_runs",
            "final_rmse_mean_db",
            "final_rmse_sample_sd_db",
            "paired_mean_delta_vs_uninformed_db",
            "paired_seeds_with_lower_rmse",
        ),
    )


def summarize_deterministic_single_model_comparison() -> None:
    paper_results = REPOSITORY_ROOT / "paper_experiments" / "results" / "paper"
    roots = {
        "factorial": paper_results / "luxembourg_cell_grid_factorial_sweeps_v1",
        "intensity": paper_results / "luxembourg_cell_grid_intensity_budget_sweep_v1",
    }
    per_map_rows: list[dict[str, object]] = []
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for method in SINGLE_MODEL_COMPARISON_METHODS:
        method_id = str(method["method_id"])
        source_root = roots[str(method["root"])]
        for map_id in FACTORIAL_MAPS:
            run_dir = source_root / "methods" / map_id / str(method["run_name"])
            result_path = run_dir / "final_fidelity.json"
            row: dict[str, object] = {
                "map_id": map_id,
                "method_id": method_id,
                "method_label": method["method_label"],
                "method_family": method["method_family"],
                "status": "missing",
                "tail_rmse_db": "",
                "model_transfers": "",
                "final_step": "",
                "source_run_directory": run_dir.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
            }
            if result_path.is_file():
                with result_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                rmse = float(payload["greedy_total"])
                transfers = int(payload["expert_bank_model_transfers"])
                step = int(payload["step"])
                if step != 1799:
                    raise RuntimeError(
                        f"unexpected final step in {result_path}: {step}"
                    )
                row.update(
                    {
                        "status": "complete",
                        "tail_rmse_db": f"{rmse:.6f}",
                        "model_transfers": transfers,
                        "final_step": step,
                    }
                )
                grouped[method_id].append((rmse, transfers))
            per_map_rows.append(row)

    completed = sum(row["status"] == "complete" for row in per_map_rows)
    if len(per_map_rows) != 108 or completed != 107:
        raise RuntimeError(
            f"expected 108 comparison cells and 107 completed runs; "
            f"found {len(per_map_rows)} and {completed}"
        )
    missing = [row for row in per_map_rows if row["status"] != "complete"]
    if len(missing) != 1 or (
        missing[0]["map_id"], missing[0]["method_id"]
    ) != ("factor_b2_v3", "learned_expert_bank_kappa2"):
        raise RuntimeError(f"unexpected missing comparison runs: {missing}")

    write_csv(
        "deterministic_single_model_per_map.csv",
        per_map_rows,
        (
            "map_id",
            "method_id",
            "method_label",
            "method_family",
            "status",
            "tail_rmse_db",
            "model_transfers",
            "final_step",
            "source_run_directory",
        ),
    )

    complete_methods = sorted(
        (
            method_id
            for method_id, values in grouped.items()
            if len(values) == len(FACTORIAL_MAPS)
        ),
        key=lambda method_id: fmean(value[0] for value in grouped[method_id]),
    )
    ranks = {method_id: rank for rank, method_id in enumerate(complete_methods, 1)}
    aggregate_rows: list[dict[str, object]] = []
    for method in SINGLE_MODEL_COMPARISON_METHODS:
        method_id = str(method["method_id"])
        values = grouped[method_id]
        aggregate_rows.append(
            {
                "rank_among_complete_methods": ranks.get(method_id, ""),
                "method_id": method_id,
                "method_label": method["method_label"],
                "method_family": method["method_family"],
                "completed_maps": len(values),
                "expected_maps": len(FACTORIAL_MAPS),
                "mean_tail_rmse_db": f"{fmean(value[0] for value in values):.6f}",
                "mean_model_transfers": f"{fmean(value[1] for value in values):.1f}",
                "note": method["note"],
            }
        )
    aggregate_rows.sort(
        key=lambda row: (
            row["rank_among_complete_methods"] == "",
            row["rank_among_complete_methods"] or 10_000,
        )
    )
    write_csv(
        "deterministic_single_model_aggregate.csv",
        aggregate_rows,
        (
            "rank_among_complete_methods",
            "method_id",
            "method_label",
            "method_family",
            "completed_maps",
            "expected_maps",
            "mean_tail_rmse_db",
            "mean_model_transfers",
            "note",
        ),
    )


def main() -> int:
    summarize_contact_timing()
    summarize_four_zone_ablation()
    summarize_cross_map_holdouts()
    summarize_online_local_validation_policy()
    summarize_kirchberg_online_private_validation_policy()
    summarize_receiver_in_zone_global_sender_policy_sweep()
    summarize_deterministic_single_model_comparison()
    print(f"Wrote experimental summaries to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
