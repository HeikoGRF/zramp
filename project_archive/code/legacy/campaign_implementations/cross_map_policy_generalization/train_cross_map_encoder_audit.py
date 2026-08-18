#!/usr/bin/env python3
"""Train an exact policy encoder on an artificial map and test zero-shot.

Kirchberg counterfactual labels are loaded only after source-only model and
architecture selection is complete. They are never used for fitting, early
stopping, normalization, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from online_policy_learning.online_local_validation_policy import (
    ExactModelTrajectoryPolicy,
    ExactPrivateState,
)


@dataclass(frozen=True)
class Example:
    seed: int
    step: int
    mode: str
    receiver: int
    provider: int
    receiver_state: int
    provider_state: int
    oracle_best_gain: float
    geometry_gain: float
    geometry_reward: float

    @property
    def group(self) -> tuple[int, int, str, int]:
        return self.seed, self.step, self.mode, self.receiver


@dataclass
class AuditDataset:
    states: list[ExactPrivateState]
    examples: list[Example]
    group_widths: tuple[int, ...]
    trajectory_dim: int


def _state_archive(directory: Path) -> tuple[list[ExactPrivateState], dict[tuple[int, str, int], int]]:
    path = directory / "parameter_objective_states.npz"
    with np.load(path, allow_pickle=False) as payload:
        group_keys = sorted(key for key in payload.files if key.startswith("group_"))
        trajectories = np.asarray(payload["trajectory"], dtype=np.float32)
        lengths = (
            np.asarray(payload["trajectory_length"], dtype=np.int32)
            if "trajectory_length" in payload.files
            else np.full(trajectories.shape[0], trajectories.shape[1], dtype=np.int32)
        )
        steps = np.asarray(payload["step"], dtype=np.int32)
        modes = np.asarray(payload["mode"]).astype(str)
        nodes = np.asarray(payload["node_idx"], dtype=np.int32)
        groups = [np.asarray(payload[key], dtype=np.float32) for key in group_keys]
        states: list[ExactPrivateState] = []
        lookup: dict[tuple[int, str, int], int] = {}
        for index in range(int(steps.size)):
            state = ExactPrivateState(
                model_groups=tuple(torch.from_numpy(group[index].copy()) for group in groups),
                trajectory=torch.from_numpy(
                    trajectories[index, : int(lengths[index])].copy()
                ),
            )
            lookup[(int(steps[index]), str(modes[index]), int(nodes[index]))] = len(states)
            states.append(state)
    return states, lookup


def load_seed(directory: Path, seed: int) -> AuditDataset:
    states, lookup = _state_archive(directory)
    grouped: dict[tuple[int, str, int, int], list[dict[str, str]]] = defaultdict(list)
    with (directory / "parameter_objective_audit.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[
                (
                    int(row["step"]),
                    str(row["mode"]),
                    int(row["receiver_idx"]),
                    int(row["provider_idx"]),
                )
            ].append(row)
    examples: list[Example] = []
    for (step, mode, receiver, provider), rows in sorted(grouped.items()):
        geometry_alpha = float(rows[0]["geometry_selected_alpha"])
        geometry_row = min(rows, key=lambda row: abs(float(row["alpha"]) - geometry_alpha))
        examples.append(
            Example(
                seed=int(seed),
                step=step,
                mode=mode,
                receiver=receiver,
                provider=provider,
                receiver_state=lookup[(step, mode, receiver)],
                provider_state=lookup[(step, mode, provider)],
                oracle_best_gain=max(float(row["oracle_gain_db"]) for row in rows),
                geometry_gain=float(geometry_row["oracle_gain_db"]),
                geometry_reward=float(rows[0]["geometry_gross_reward"]),
            )
        )
    first = states[0]
    return AuditDataset(
        states=states,
        examples=examples,
        group_widths=tuple(int(group.shape[1]) for group in first.model_groups),
        trajectory_dim=int(first.trajectory.shape[1]),
    )


def combine(datasets: list[AuditDataset]) -> AuditDataset:
    states: list[ExactPrivateState] = []
    examples: list[Example] = []
    group_widths = datasets[0].group_widths
    trajectory_dim = datasets[0].trajectory_dim
    for dataset in datasets:
        if dataset.group_widths != group_widths or dataset.trajectory_dim != trajectory_dim:
            raise ValueError("source and target predictor state architectures differ")
        offset = len(states)
        states.extend(dataset.states)
        examples.extend(
            Example(
                seed=row.seed,
                step=row.step,
                mode=row.mode,
                receiver=row.receiver,
                provider=row.provider,
                receiver_state=row.receiver_state + offset,
                provider_state=row.provider_state + offset,
                oracle_best_gain=row.oracle_best_gain,
                geometry_gain=row.geometry_gain,
                geometry_reward=row.geometry_reward,
            )
            for row in dataset.examples
        )
    return AuditDataset(states, examples, group_widths, trajectory_dim)


def labels(dataset: AuditDataset, target: str) -> np.ndarray:
    if target == "oracle":
        return np.asarray([row.oracle_best_gain for row in dataset.examples], dtype=np.float32)
    if target == "oracle-geometry-alpha":
        return np.asarray([row.geometry_gain for row in dataset.examples], dtype=np.float32)
    if target == "geometry":
        return np.asarray([row.geometry_reward for row in dataset.examples], dtype=np.float32)
    raise ValueError(target)


def true_objective(dataset: AuditDataset, target: str) -> np.ndarray:
    if target == "oracle":
        return np.asarray([row.oracle_best_gain for row in dataset.examples], dtype=np.float64)
    if target == "oracle-geometry-alpha":
        return np.asarray([row.geometry_gain for row in dataset.examples], dtype=np.float64)
    return np.asarray([row.geometry_gain for row in dataset.examples], dtype=np.float64)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or float(np.std(first)) < 1.0e-12 or float(np.std(second)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_scores(dataset: AuditDataset, score: np.ndarray, target: str) -> dict[str, float | int]:
    truth = true_objective(dataset, target)
    grouped: dict[tuple[int, int, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.examples):
        grouped[row.group].append(index)
    selected: list[float] = []
    random_expected: list[float] = []
    oracle: list[float] = []
    order_hits = 0
    order_total = 0
    top_hits = 0
    within_rhos: list[float] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        values = truth[idx]
        scores = score[idx]
        chosen = int(np.argmax(scores))
        selected.append(float(values[chosen]))
        random_expected.append(float(np.mean(values)))
        oracle.append(float(np.max(values)))
        top_hits += int(values[chosen] >= float(np.max(values)) - 1.0e-10)
        within_rhos.append(_correlation(_rank(scores), _rank(values)))
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                delta = float(values[left] - values[right])
                if abs(delta) < 1.0e-10:
                    continue
                order_total += 1
                order_hits += int((float(scores[left] - scores[right]) * delta) > 0.0)
    selected_mean = float(np.mean(selected)) if selected else float("nan")
    random_mean = float(np.mean(random_expected)) if random_expected else float("nan")
    oracle_mean = float(np.mean(oracle)) if oracle else float("nan")
    gap = oracle_mean - random_mean
    return {
        "pairs": int(len(dataset.examples)),
        "rankable_opportunities": int(len(selected)),
        "pearson": _correlation(score, truth),
        "spearman": _correlation(_rank(score), _rank(truth)),
        "within_opportunity_spearman": float(np.nanmean(within_rhos)) if within_rhos else float("nan"),
        "pairwise_order_accuracy": float(order_hits / order_total) if order_total else float("nan"),
        "top_provider_accuracy": float(top_hits / len(selected)) if selected else float("nan"),
        "selected_gain_db": selected_mean,
        "random_gain_db": random_mean,
        "oracle_gain_db": oracle_mean,
        "gain_over_random_db": selected_mean - random_mean,
        "oracle_gap_closed": (selected_mean - random_mean) / gap if gap > 1.0e-12 else float("nan"),
    }


def make_policy(dataset: AuditDataset, config: dict[str, int], seed: int, device: torch.device) -> ExactModelTrajectoryPolicy:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        return ExactModelTrajectoryPolicy(
            group_widths=dataset.group_widths,
            trajectory_dim=dataset.trajectory_dim,
            hidden_dim=int(config["hidden"]),
            embedding_dim=int(config["embedding"]),
            gain_hidden_dim=int(config["gain_hidden"]),
            pair_feature_mode="relational",
        ).to(device)


def example_indices(dataset: AuditDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([row.receiver_state for row in dataset.examples], dtype=torch.long, device=device),
        torch.tensor([row.provider_state for row in dataset.examples], dtype=torch.long, device=device),
    )


def ranking_pairs(dataset: AuditDataset, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped: dict[tuple[int, int, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.examples):
        grouped[row.group].append(index)
    left: list[int] = []
    right: list[int] = []
    sign: list[float] = []
    minimum = max(1.0e-4, 0.05 * float(np.std(y)))
    for indices in grouped.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                delta = float(y[indices[a]] - y[indices[b]])
                if abs(delta) < minimum:
                    continue
                left.append(indices[a])
                right.append(indices[b])
                sign.append(1.0 if delta > 0.0 else -1.0)
    return (
        torch.tensor(left, dtype=torch.long, device=device),
        torch.tensor(right, dtype=torch.long, device=device),
        torch.tensor(sign, dtype=torch.float32, device=device),
    )


def predict(policy: ExactModelTrajectoryPolicy, dataset: AuditDataset, device: torch.device) -> np.ndarray:
    policy.eval()
    receiver, provider = example_indices(dataset, device)
    with torch.inference_mode():
        embedding = policy.encode_many(dataset.states)
        result = policy.score_embeddings(embedding[receiver], embedding[provider])
    return result.detach().cpu().numpy().astype(np.float64)


def train_central(
    train: AuditDataset,
    validation: AuditDataset,
    *,
    config: dict[str, int],
    target: str,
    seed: int,
    device: torch.device,
    freeze_encoder: bool = False,
    epochs: int = 300,
) -> tuple[dict[str, torch.Tensor], dict[str, object], int]:
    policy = make_policy(train, config, seed, device)
    if freeze_encoder:
        for name, parameter in policy.named_parameters():
            if not name.startswith("gain_head."):
                parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=2.0e-3,
        weight_decay=1.0e-6,
    )
    raw_y = labels(train, target)
    mean = float(np.mean(raw_y))
    scale = max(float(np.std(raw_y)), 1.0e-3)
    y = torch.tensor((raw_y - mean) / scale, dtype=torch.float32, device=device)
    receiver, provider = example_indices(train, device)
    rank_left, rank_right, rank_sign = ranking_pairs(train, raw_y, device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        embedding = policy.encode_many(train.states)
        score = policy.score_embeddings(embedding[receiver], embedding[provider])
        pointwise = F.smooth_l1_loss(score, y)
        if rank_left.numel() > 0:
            ranking = F.softplus(
                -rank_sign * (score[rank_left] - score[rank_right]) / 0.5
            ).mean()
        else:
            ranking = score.new_zeros(())
        loss = pointwise + 0.35 * ranking
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        if epoch % 10 != 0 and epoch != epochs:
            continue
        metrics = evaluate_scores(validation, predict(policy, validation, device), target)
        metric = float(metrics["pairwise_order_accuracy"]) + 0.15 * float(metrics["spearman"])
        if math.isfinite(metric) and metric > best_metric + 1.0e-5:
            best_metric = metric
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            stale = 0
        else:
            stale += 10
        if stale >= 100 and epoch >= 150:
            break
    if best_state is None:
        best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        best_epoch = epoch
    policy.load_state_dict(best_state)
    return best_state, evaluate_scores(validation, predict(policy, validation, device), target), best_epoch


def train_final(
    dataset: AuditDataset,
    *,
    config: dict[str, int],
    target: str,
    seed: int,
    device: torch.device,
    epochs: int,
    freeze_encoder: bool = False,
) -> ExactModelTrajectoryPolicy:
    policy = make_policy(dataset, config, seed, device)
    if freeze_encoder:
        for name, parameter in policy.named_parameters():
            if not name.startswith("gain_head."):
                parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=2.0e-3,
        weight_decay=1.0e-6,
    )
    raw_y = labels(dataset, target)
    y = torch.tensor(
        (raw_y - float(np.mean(raw_y))) / max(float(np.std(raw_y)), 1.0e-3),
        dtype=torch.float32,
        device=device,
    )
    receiver, provider = example_indices(dataset, device)
    rank_left, rank_right, rank_sign = ranking_pairs(dataset, raw_y, device)
    for _ in range(max(1, int(epochs))):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        embedding = policy.encode_many(dataset.states)
        score = policy.score_embeddings(embedding[receiver], embedding[provider])
        loss = F.smooth_l1_loss(score, y)
        if rank_left.numel() > 0:
            loss = loss + 0.35 * F.softplus(
                -rank_sign * (score[rank_left] - score[rank_right]) / 0.5
            ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
    return policy


@dataclass
class ReplayRow:
    receiver_state: ExactPrivateState
    provider_embedding: torch.Tensor
    target: float


@dataclass
class Client:
    policy: ExactModelTrajectoryPolicy
    optimizer: torch.optim.Optimizer
    replay: list[ReplayRow]
    experience: int = 0


def _parameter_names(policy: ExactModelTrajectoryPolicy, mode: str) -> list[str]:
    if mode == "full_gossip":
        return [name for name, _ in policy.named_parameters()]
    return [name for name, _ in policy.named_parameters() if not name.startswith("gain_head.")]


def _flat(client: Client, names: list[str]) -> torch.Tensor:
    params = dict(client.policy.named_parameters())
    return torch.cat([params[name].detach().reshape(-1).cpu() for name in names])


def _align(first: Client, second: Client, names: list[str]) -> None:
    own = dict(first.policy.named_parameters())
    peer = dict(second.policy.named_parameters())
    total = first.experience + second.experience
    fraction = 0.5 if total <= 0 else float(first.experience) / float(total)
    with torch.no_grad():
        for name in names:
            value = own[name].detach() * fraction + peer[name].detach() * (1.0 - fraction)
            own[name].copy_(value)
            peer[name].copy_(value)
    first.optimizer.state.clear()
    second.optimizer.state.clear()


def _train_client(client: Client, device: torch.device, epochs: int) -> None:
    if not client.replay:
        return
    for _ in range(int(epochs)):
        client.policy.train()
        client.optimizer.zero_grad(set_to_none=True)
        receiver = client.policy.encode_many([row.receiver_state for row in client.replay])
        provider = torch.stack([row.provider_embedding for row in client.replay]).to(device)
        prediction = client.policy.score_embeddings(receiver, provider)
        target = torch.tensor([row.target for row in client.replay], dtype=torch.float32, device=device)
        loss = F.smooth_l1_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(client.policy.parameters(), 5.0)
        client.optimizer.step()


def decentralized_scores(
    clients: list[Client], dataset: AuditDataset, device: torch.device, offset: int
) -> np.ndarray:
    result: list[float] = []
    for row in dataset.examples:
        receiver_client = clients[(row.receiver + offset) % len(clients)]
        provider_client = clients[(row.provider + offset) % len(clients)]
        receiver_client.policy.eval()
        provider_client.policy.eval()
        with torch.inference_mode():
            receiver = receiver_client.policy.encode(dataset.states[row.receiver_state])
            provider = provider_client.policy.encode(dataset.states[row.provider_state])
            score = receiver_client.policy.score_embeddings(
                receiver.unsqueeze(0), provider.unsqueeze(0)
            )
        result.append(float(score.item()))
    return np.asarray(result, dtype=np.float64)


def run_gossip(
    train: AuditDataset,
    evaluation: AuditDataset,
    target_map: AuditDataset,
    *,
    config: dict[str, int],
    mode: str,
    seed: int,
    device: torch.device,
    local_epochs: int = 12,
) -> dict[str, object]:
    torch.manual_seed(seed)
    template = make_policy(train, config, seed, device)
    initial = copy.deepcopy(template.state_dict())
    clients: list[Client] = []
    for _ in range(40):
        policy = make_policy(train, config, seed, device)
        policy.load_state_dict(initial)
        clients.append(
            Client(
                policy=policy,
                optimizer=torch.optim.Adam(policy.parameters(), lr=2.0e-3, weight_decay=1.0e-6),
                replay=[],
            )
        )
    raw = labels(train, "oracle")
    mean = float(np.mean(raw))
    scale = max(float(np.std(raw)), 1.0e-3)
    by_step: dict[int, list[tuple[int, Example]]] = defaultdict(list)
    for index, row in enumerate(train.examples):
        by_step[row.step].append((index, row))
    cancellation: list[float] = []
    dispersion_before: list[float] = []
    dispersion_after: list[float] = []
    for step in sorted(by_step):
        rows = by_step[step]
        for index, row in rows:
            receiver_client = clients[row.receiver % 40]
            provider_client = clients[row.provider % 40]
            provider_client.policy.eval()
            with torch.inference_mode():
                provider_embedding = provider_client.policy.encode(train.states[row.provider_state]).detach().cpu()
            receiver_client.replay.append(
                ReplayRow(
                    receiver_state=train.states[row.receiver_state],
                    provider_embedding=provider_embedding,
                    target=(float(raw[index]) - mean) / scale,
                )
            )
            receiver_client.experience += 1
        names = _parameter_names(clients[0].policy, mode)
        before = [_flat(client, names) for client in clients]
        for client in clients:
            _train_client(client, device, local_epochs)
        after_local = [_flat(client, names) for client in clients]
        deltas = [after - start for after, start in zip(after_local, before)]
        norms = torch.tensor([float(delta.norm()) for delta in deltas])
        mean_delta = torch.stack(deltas).mean(dim=0)
        cancellation.append(float(mean_delta.norm() / norms.mean().clamp_min(1.0e-12)))
        edges = sorted({(min(row.receiver % 40, row.provider % 40), max(row.receiver % 40, row.provider % 40)) for _, row in rows if row.receiver % 40 != row.provider % 40})
        dispersion_before.append(float(np.mean([float((after_local[a] - after_local[b]).norm()) for a, b in edges])) if edges else 0.0)
        if mode != "local_only":
            for first, second in edges:
                _align(clients[first], clients[second], names)
        after_gossip = [_flat(client, names) for client in clients]
        dispersion_after.append(float(np.mean([float((after_gossip[a] - after_gossip[b]).norm()) for a, b in edges])) if edges else 0.0)
    evaluation_metrics = []
    target_metrics = []
    for offset in range(4):
        evaluation_metrics.append(evaluate_scores(evaluation, decentralized_scores(clients, evaluation, device, offset), "oracle"))
        target_metrics.append(evaluate_scores(target_map, decentralized_scores(clients, target_map, device, offset), "oracle"))
    def average(rows: list[dict[str, float | int]]) -> dict[str, float]:
        keys = [key for key, value in rows[0].items() if isinstance(value, (float, int)) and key not in {"pairs", "rankable_opportunities"}]
        return {key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys}
    return {
        "mode": mode,
        "source_holdout": average(evaluation_metrics),
        "kirchberg_zero_shot": average(target_metrics),
        "mean_update_cancellation_ratio": float(np.mean(cancellation)),
        "mean_contact_dispersion_before_gossip": float(np.mean(dispersion_before)),
        "mean_contact_dispersion_after_gossip": float(np.mean(dispersion_after)),
        "local_examples": int(sum(client.experience for client in clients)),
    }


def averaged_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in rows[0].items():
        if not isinstance(value, (float, int)) or key in {"pairs", "rankable_opportunities"}:
            continue
        output[key] = float(np.nanmean([float(row[key]) for row in rows]))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(8)
    source_one = load_seed(args.root / "source_tiny" / "seed_01", 1)
    source_two = load_seed(args.root / "source_tiny" / "seed_02", 2)
    source_all = combine([source_one, source_two])
    if source_one.group_widths != source_two.group_widths:
        raise ValueError("source architectures differ")
    configs = [
        {"hidden": 8, "embedding": 32, "gain_hidden": 64},
        {"hidden": 16, "embedding": 64, "gain_hidden": 96},
        {"hidden": 32, "embedding": 128, "gain_hidden": 128},
    ]
    sweep: list[dict[str, object]] = []
    best: tuple[float, dict[str, int], int] | None = None
    for config in configs:
        fold_metrics: list[dict[str, float | int]] = []
        epochs: list[int] = []
        for fold, (train, validation) in enumerate(((source_one, source_two), (source_two, source_one))):
            _state, metrics, best_epoch = train_central(
                train,
                validation,
                config=config,
                target="oracle",
                seed=20260727 + fold,
                device=device,
            )
            fold_metrics.append(metrics)
            epochs.append(best_epoch)
        mean_metrics = averaged_metrics(fold_metrics)
        score = float(mean_metrics["pairwise_order_accuracy"]) + 0.15 * float(mean_metrics["spearman"])
        sweep.append({"config": config, "folds": fold_metrics, "mean": mean_metrics, "best_epochs": epochs, "source_selection_score": score})
        if best is None or score > best[0]:
            best = (score, config, max(100, int(round(float(np.mean(epochs))))))
        print(f"[ENCODER-SWEEP] config={config} score={score:.4f} order={mean_metrics['pairwise_order_accuracy']:.3f} rho={mean_metrics['spearman']:.3f}", flush=True)
    assert best is not None
    _, best_config, final_epochs = best

    frozen_folds = []
    for fold, (train, validation) in enumerate(((source_one, source_two), (source_two, source_one))):
        _state, metrics, _epoch = train_central(
            train,
            validation,
            config=best_config,
            target="oracle",
            seed=20260727 + fold,
            device=device,
            freeze_encoder=True,
        )
        frozen_folds.append(metrics)
    frozen_mean = averaged_metrics(frozen_folds)
    print(f"[FROZEN-CONTROL] order={frozen_mean['pairwise_order_accuracy']:.3f} rho={frozen_mean['spearman']:.3f}", flush=True)

    # Target labels are deliberately loaded only now, after source-only selection.
    target_one = load_seed(args.root / "target_kirchberg" / "seed_01", 1)
    target_two = load_seed(args.root / "target_kirchberg" / "seed_02", 2)
    target_all = combine([target_one, target_two])
    if target_all.group_widths != source_all.group_widths or target_all.trajectory_dim != source_all.trajectory_dim:
        raise ValueError("target predictor architecture differs from artificial source")

    transfer: dict[str, object] = {}
    for objective in ("oracle", "geometry"):
        target_rows = []
        source_rows = []
        for repeat in range(2):
            policy = train_final(
                source_all,
                config=best_config,
                target=objective,
                seed=20260800 + repeat,
                device=device,
                epochs=final_epochs,
            )
            source_rows.append(evaluate_scores(source_all, predict(policy, source_all, device), objective))
            target_rows.append(evaluate_scores(target_all, predict(policy, target_all, device), objective))
        transfer[objective] = {
            "source_train": averaged_metrics(source_rows),
            "kirchberg_zero_shot": averaged_metrics(target_rows),
            "repeats": target_rows,
        }
        result = transfer[objective]["kirchberg_zero_shot"]
        print(f"[ZERO-SHOT] objective={objective} gain_over_random={result['gain_over_random_db']:.3f} order={result['pairwise_order_accuracy']:.3f} rho={result['spearman']:.3f}", flush=True)

    gossip: list[dict[str, object]] = []
    for fold, (train, evaluation) in enumerate(((source_one, source_two), (source_two, source_one))):
        for mode in ("local_only", "encoder_gossip", "full_gossip"):
            row = run_gossip(
                train,
                evaluation,
                target_all,
                config=best_config,
                mode=mode,
                seed=20260900 + fold,
                device=device,
            )
            row["fold"] = fold
            gossip.append(row)
            target_result = row["kirchberg_zero_shot"]
            print(f"[GOSSIP] fold={fold} mode={mode} target_gain={target_result['gain_over_random_db']:.3f} order={target_result['pairwise_order_accuracy']:.3f} cancel={row['mean_update_cancellation_ratio']:.3f}", flush=True)

    gossip_summary: dict[str, object] = {}
    for mode in ("local_only", "encoder_gossip", "full_gossip"):
        rows = [row for row in gossip if row["mode"] == mode]
        gossip_summary[mode] = {
            "source_holdout": averaged_metrics([row["source_holdout"] for row in rows]),
            "kirchberg_zero_shot": averaged_metrics([row["kirchberg_zero_shot"] for row in rows]),
            "mean_update_cancellation_ratio": float(np.mean([row["mean_update_cancellation_ratio"] for row in rows])),
            "mean_contact_dispersion_before_gossip": float(np.mean([row["mean_contact_dispersion_before_gossip"] for row in rows])),
            "mean_contact_dispersion_after_gossip": float(np.mean([row["mean_contact_dispersion_after_gossip"] for row in rows])),
        }

    report = {
        "protocol": {
            "source": "tiny artificial map only; seed cross-validation",
            "target": "Kirchberg; loaded only after source-only architecture selection",
            "target_used_for_training_or_selection": False,
            "predictor_architecture": "small (identical source and target)",
            "policy_input": "exact predictor tensors and private trajectory summary; no handmade geometry features",
            "aggregation_labels": {
                "oracle": "best true artificial-map counterfactual alpha; source-pretraining capacity test",
                "geometry": "post-pull parameter-geometry reward; target evaluated on true gain at geometry-selected alpha",
            },
            "gossip_isolation": "oracle source labels for every method; only local versus encoder-only versus full-policy averaging changes",
            "device": str(device),
        },
        "source_examples": len(source_all.examples),
        "target_examples": len(target_all.examples),
        "architecture_sweep": sweep,
        "selected_config": best_config,
        "final_epochs": final_epochs,
        "frozen_encoder_control": {"folds": frozen_folds, "mean": frozen_mean},
        "cross_map_transfer": transfer,
        "gossip_folds": gossip,
        "gossip_summary": gossip_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
