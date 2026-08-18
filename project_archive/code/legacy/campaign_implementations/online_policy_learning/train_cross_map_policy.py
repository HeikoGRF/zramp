#!/usr/bin/env python3
"""Centrally pretrain the exact encoder/gain head on source-map decisions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from online_policy_learning.build_cross_map_policy_dataset import _make_predictor, _stable_seed
from online_policy_learning.online_local_validation_policy import (
    ExactModelTrajectoryPolicy,
    ExactPrivateState,
    exact_model_groups,
)


def _transform_groups(
    groups: tuple[torch.Tensor, ...],
    initial: tuple[torch.Tensor, ...],
    representation: str,
) -> tuple[torch.Tensor, ...]:
    if representation == "raw":
        return groups
    output: list[torch.Tensor] = []
    for current, reference in zip(groups, initial):
        delta = current - reference
        if representation == "delta_unit":
            scale = reference.square().mean().sqrt().clamp_min(1.0e-3)
            delta = delta / scale
        output.append(delta)
    return tuple(output)


def _load_cases(
    paths: list[Path], representation: str
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "cross_map_policy_case_v1":
            raise ValueError(f"{path}: unsupported case format")
        summary = payload["summary"]
        initial_model = _make_predictor(
            _stable_seed(
                "source-predictor", summary["map_id"], summary["case_seed"]
            ),
            torch.device("cpu"),
            include_time=bool(summary.get("predictor_time", True)),
        )
        initial_groups = exact_model_groups(initial_model.state_dict())
        compact: list[ExactPrivateState] = []
        for snapshot in payload["snapshots"]:
            state = snapshot["model_state"]
            trajectory = snapshot["trajectory"]
            if not isinstance(state, dict) or not isinstance(trajectory, torch.Tensor):
                raise TypeError(f"{path}: malformed source snapshot")
            compact.append(
                ExactPrivateState(
                    model_groups=_transform_groups(
                        exact_model_groups(state), initial_groups, representation
                    ),
                    trajectory=trajectory,
                )
            )
        payload["snapshots"] = compact
        payload["path"] = str(path.resolve())
        cases.append(payload)
    return cases


def _groups(cases: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for case_idx, case in enumerate(cases):
        summary = case["summary"]
        if not isinstance(summary, dict) or str(summary["split"]) != split:
            continue
        for row in case["decisions"]:
            grouped[(case_idx, str(row["group_id"]))].append(row)
    output: list[dict[str, object]] = []
    for (case_idx, group_id), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        output.append(
            {
                "case_idx": case_idx,
                "group_id": group_id,
                "rows": sorted(rows, key=lambda row: int(row["provider_snapshot"])),
                "size": len(rows),
            }
        )
    return output


def _build_policy(first_snapshot: ExactPrivateState, args: argparse.Namespace) -> ExactModelTrajectoryPolicy:
    widths = tuple(int(group.shape[1]) for group in first_snapshot.model_groups)
    return ExactModelTrajectoryPolicy(
        group_widths=widths,
        trajectory_dim=int(first_snapshot.trajectory.shape[1]),
        hidden_dim=int(args.hidden_dim),
        embedding_dim=int(args.embedding_dim),
        gain_hidden_dim=int(args.gain_hidden_dim),
        pair_feature_mode=str(args.pair_feature_mode),
    )


def _group_forward(
    policy: ExactModelTrajectoryPolicy,
    case: dict[str, object],
    group: dict[str, object],
    device: torch.device,
    embedding_cache: dict[tuple[int, int], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = group["rows"]
    snapshots = case["snapshots"]
    case_index = int(group["case_idx"])

    def embedding(snapshot_index: int) -> torch.Tensor:
        key = (case_index, snapshot_index)
        if embedding_cache is None:
            return policy.encode(snapshots[snapshot_index])
        if key not in embedding_cache:
            embedding_cache[key] = policy.encode(snapshots[snapshot_index])
        return embedding_cache[key]

    receiver_index = int(rows[0]["receiver_snapshot"])
    receiver = embedding(receiver_index)
    providers = torch.stack(
        [embedding(int(row["provider_snapshot"])) for row in rows]
    )
    repeated = receiver.unsqueeze(0).expand(int(providers.shape[0]), -1)
    scores = policy.score_embeddings(repeated, providers)
    targets = torch.tensor(
        [float(row["target_gain"]) for row in rows], dtype=torch.float32, device=device
    )
    return scores, targets


def _prime_embedding_cache(
    policy: ExactModelTrajectoryPolicy,
    cases: list[dict[str, object]],
    groups: list[dict[str, object]],
    embedding_cache: dict[tuple[int, int], torch.Tensor],
) -> None:
    keys: set[tuple[int, int]] = set()
    for group in groups:
        case_index = int(group["case_idx"])
        rows = group["rows"]
        keys.add((case_index, int(rows[0]["receiver_snapshot"])))
        keys.update(
            (case_index, int(row["provider_snapshot"])) for row in rows
        )
    missing = sorted(key for key in keys if key not in embedding_cache)
    if not missing:
        return
    states = [
        cases[case_index]["snapshots"][snapshot_index]
        for case_index, snapshot_index in missing
    ]
    embeddings = policy.encode_many(states)
    embedding_cache.update(
        {key: embedding for key, embedding in zip(missing, embeddings)}
    )


def _group_loss(scores: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    pointwise = F.smooth_l1_loss(scores, targets)
    target_probability = torch.softmax(targets / float(args.target_temperature), dim=0)
    listwise = -(target_probability * torch.log_softmax(scores / float(args.score_temperature), dim=0)).sum()
    differences = targets[:, None] - targets[None, :]
    score_differences = scores[:, None] - scores[None, :]
    mask = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1)
    mask &= torch.abs(differences) >= float(args.ranking_margin)
    if bool(mask.any()):
        direction = torch.sign(differences[mask])
        ranking = F.softplus(-direction * score_differences[mask] / float(args.score_temperature)).mean()
    else:
        ranking = scores.new_zeros(())
    loss = pointwise + float(args.listwise_weight) * listwise + float(args.ranking_weight) * ranking
    return loss, {
        "pointwise": float(pointwise.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        for index, count in enumerate(counts):
            if count > 1:
                positions = np.flatnonzero(inverse == index)
                ranks[positions] = float(np.mean(ranks[positions]))
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) <= 1.0e-12 or float(np.std(right)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _evaluate(
    policy: ExactModelTrajectoryPolicy,
    cases: list[dict[str, object]],
    groups: list[dict[str, object]],
    device: torch.device,
) -> dict[str, object]:
    predictions: list[float] = []
    targets: list[float] = []
    selected: list[float] = []
    random_expected: list[float] = []
    oracle: list[float] = []
    regrets: list[float] = []
    pair_correct = 0
    pair_total = 0
    by_map: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    policy.eval()
    with torch.inference_mode():
        embedding_cache: dict[tuple[int, int], torch.Tensor] = {}
        _prime_embedding_cache(
            policy, cases, groups, embedding_cache
        )
        for group in groups:
            case = cases[int(group["case_idx"])]
            scores, labels = _group_forward(
                policy, case, group, device, embedding_cache
            )
            score_values = scores.detach().cpu().numpy().astype(np.float64)
            label_values = labels.detach().cpu().numpy().astype(np.float64)
            predictions.extend(score_values.tolist())
            targets.extend(label_values.tolist())
            choice = int(np.argmax(score_values))
            selected.append(float(label_values[choice]))
            random_expected.append(float(np.mean(label_values)))
            oracle.append(float(np.max(label_values)))
            regrets.append(float(np.max(label_values) - label_values[choice]))
            map_id = str(case["summary"]["map_id"])
            by_map[map_id].append((score_values, label_values))
            for left in range(len(label_values)):
                for right in range(left + 1, len(label_values)):
                    if abs(label_values[left] - label_values[right]) < 0.01:
                        continue
                    pair_total += 1
                    pair_correct += int(
                        (score_values[left] - score_values[right])
                        * (label_values[left] - label_values[right]) > 0.0
                    )
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    result: dict[str, object] = {
        "groups": len(groups),
        "decisions": len(truth),
        "rmse": float(np.sqrt(np.mean(np.square(pred - truth)))) if len(truth) else float("nan"),
        "mae": float(np.mean(np.abs(pred - truth))) if len(truth) else float("nan"),
        "pearson": _correlation(pred, truth),
        "spearman": _correlation(_ranks(pred), _ranks(truth)),
        "sign_accuracy": float(np.mean((pred > 0.0) == (truth > 0.0))) if len(truth) else float("nan"),
        "pairwise_accuracy": float(pair_correct / pair_total) if pair_total else float("nan"),
        "selected_gain_mean": float(np.mean(selected)) if selected else float("nan"),
        "random_expected_gain_mean": float(np.mean(random_expected)) if random_expected else float("nan"),
        "oracle_gain_mean": float(np.mean(oracle)) if oracle else float("nan"),
        "gain_uplift_over_random": float(np.mean(selected) - np.mean(random_expected)) if selected else float("nan"),
        "top1_regret_mean": float(np.mean(regrets)) if regrets else float("nan"),
    }
    per_map: dict[str, object] = {}
    for map_id, blocks in sorted(by_map.items()):
        map_pred = np.concatenate([block[0] for block in blocks])
        map_truth = np.concatenate([block[1] for block in blocks])
        map_selected = [float(labels[int(np.argmax(scores))]) for scores, labels in blocks]
        map_random = [float(np.mean(labels)) for _scores, labels in blocks]
        per_map[map_id] = {
            "groups": len(blocks),
            "decisions": len(map_truth),
            "pearson": _correlation(map_pred, map_truth),
            "spearman": _correlation(_ranks(map_pred), _ranks(map_truth)),
            "selected_gain_mean": float(np.mean(map_selected)),
            "random_expected_gain_mean": float(np.mean(map_random)),
            "gain_uplift_over_random": float(np.mean(map_selected) - np.mean(map_random)),
        }
    result["per_map"] = per_map
    return result


def _checkpoint(
    path: Path,
    policy: ExactModelTrajectoryPolicy,
    *,
    args: argparse.Namespace,
    decisions_seen: int,
    epoch: int,
    source_maps: list[str],
    validation_maps: list[str],
    validation: dict[str, object] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cross_map_pretrained_exact_policy_v1",
            "policy_state_dict": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
            "architecture": {
                "group_widths": list(policy.group_widths),
                "trajectory_dim": int(policy.trajectory_encoder.input_size),
                "hidden_dim": int(policy.layer_encoder.hidden_size),
                "embedding_dim": int(policy.embedding_dim),
                "gain_hidden_dim": int(policy.gain_head[0].out_features),
                "pair_feature_mode": str(policy.pair_feature_mode),
                "model_input_representation": str(args.model_input_representation),
            },
            "decisions_seen": int(decisions_seen),
            "epoch": int(epoch),
            "source_maps": source_maps,
            "validation_maps": validation_maps,
            "deployment_map_excluded": str(args.deployment_map_excluded),
            "validation": validation,
            "training_args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    cases = _load_cases(
        [path.resolve() for path in args.cases],
        str(args.model_input_representation),
    )
    train_groups = _groups(cases, "train")
    validation_groups = _groups(cases, "validation")
    if not train_groups or not validation_groups:
        raise ValueError("both training and held-out validation groups are required")
    first_case = cases[int(train_groups[0]["case_idx"])]
    first_snapshot = first_case["snapshots"][int(train_groups[0]["rows"][0]["receiver_snapshot"])]
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    policy = _build_policy(first_snapshot, args).to(device)
    encoder = [parameter for name, parameter in policy.named_parameters() if not name.startswith("gain_head.")]
    head = list(policy.gain_head.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": encoder, "lr": float(args.learning_rate) * float(args.encoder_lr_scale)},
            {"params": head, "lr": float(args.learning_rate)},
        ]
    )
    source_maps = sorted({str(case["summary"]["map_id"]) for case in cases if str(case["summary"]["split"]) == "train"})
    validation_maps = sorted({str(case["summary"]["map_id"]) for case in cases if str(case["summary"]["split"]) == "validation"})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    milestones = sorted({int(value) for value in str(args.experience_checkpoints).split(",") if value.strip()})
    saved: set[int] = set()
    decisions_seen = 0
    best_uplift = -float("inf")
    history: list[dict[str, object]] = []
    rng = random.Random(int(args.seed))
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(train_groups)))
        rng.shuffle(order)
        epoch_losses: list[float] = []
        batch_size = int(args.accumulate_groups)
        for batch_start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            embedding_cache: dict[tuple[int, int], torch.Tensor] = {}
            batch_losses: list[torch.Tensor] = []
            batch_indices = order[batch_start : batch_start + batch_size]
            batch_groups = [train_groups[index] for index in batch_indices]
            _prime_embedding_cache(
                policy, cases, batch_groups, embedding_cache
            )
            for index in batch_indices:
                group = train_groups[index]
                case = cases[int(group["case_idx"])]
                scores, targets = _group_forward(
                    policy, case, group, device, embedding_cache
                )
                loss, _parts = _group_loss(scores, targets, args)
                batch_losses.append(loss)
                epoch_losses.append(float(loss.detach().cpu()))
                decisions_seen += int(group["size"])
            torch.stack(batch_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            for milestone in milestones:
                if milestone not in saved and decisions_seen >= milestone:
                    _checkpoint(
                        output / f"policy_experience_{milestone:07d}.pt",
                        policy,
                        args=args,
                        decisions_seen=decisions_seen,
                        epoch=epoch,
                        source_maps=source_maps,
                        validation_maps=validation_maps,
                        validation=None,
                    )
                    saved.add(milestone)
        validation = _evaluate(policy, cases, validation_groups, device)
        training = _evaluate(policy, cases, train_groups, device)
        row = {
            "epoch": epoch,
            "decisions_seen": decisions_seen,
            "loss_mean": float(np.mean(epoch_losses)),
            "train": training,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        uplift = float(validation["gain_uplift_over_random"])
        if math.isfinite(uplift) and uplift > best_uplift:
            best_uplift = uplift
            _checkpoint(
                output / "policy_best_validation.pt",
                policy,
                args=args,
                decisions_seen=decisions_seen,
                epoch=epoch,
                source_maps=source_maps,
                validation_maps=validation_maps,
                validation=validation,
            )
    final_validation = _evaluate(policy, cases, validation_groups, device)
    _checkpoint(
        output / "policy_final.pt",
        policy,
        args=args,
        decisions_seen=decisions_seen,
        epoch=int(args.epochs),
        source_maps=source_maps,
        validation_maps=validation_maps,
        validation=final_validation,
    )
    report = {
        "format": "cross_map_policy_pretraining_report_v1",
        "source_maps": source_maps,
        "validation_maps": validation_maps,
        "deployment_map_excluded": str(args.deployment_map_excluded),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "best_validation_uplift": best_uplift,
        "final_validation": final_validation,
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--gain-hidden-dim", type=int, default=64)
    parser.add_argument("--pair-feature-mode", choices=("concat", "relational"), default="relational")
    parser.add_argument(
        "--model-input-representation",
        choices=("raw", "delta", "delta_unit"),
        default="raw",
    )
    parser.add_argument("--accumulate-groups", type=int, default=8)
    parser.add_argument("--listwise-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.01)
    parser.add_argument("--target-temperature", type=float, default=0.08)
    parser.add_argument("--score-temperature", type=float, default=0.10)
    parser.add_argument("--experience-checkpoints", default="1000,5000,10000,25000,50000")
    parser.add_argument(
        "--deployment-map-excluded",
        default="single_zone_urban_150",
    )
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
