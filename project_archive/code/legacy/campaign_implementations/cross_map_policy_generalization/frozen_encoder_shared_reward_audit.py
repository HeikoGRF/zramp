#!/usr/bin/env python3
"""Fast causal audit of shared policy-reward samples with a frozen encoder.

Only the selected provider's deployable post-pull geometry reward is observed.
Oracle RMSE gains on tiny and Kirchberg are used only to score the held-out seed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cross_map_policy_generalization.mature_requester_policy_deployment_audit import (
    checkpoint_candidates,
    load_checkpoint,
    make_policy,
    transform_target,
)
from cross_map_policy_generalization.train_cross_map_encoder_audit import (
    AuditDataset,
    averaged_metrics,
    evaluate_scores,
    load_seed,
)


@dataclass(frozen=True)
class RewardSample:
    sample_id: tuple[int, int, str, int, int]
    feature: torch.Tensor
    reward: float
    step: int


@dataclass
class HeadClient:
    head: torch.nn.Module
    optimizer: torch.optim.Optimizer
    initial: dict[str, torch.Tensor]
    samples: dict[tuple[int, int, str, int, int], RewardSample] = field(
        default_factory=dict
    )
    updates: int = 0
    received: int = 0
    last_trained_count: int = 0


def sample_priority(sample_id: object) -> int:
    return int.from_bytes(
        hashlib.blake2b(repr(sample_id).encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def retain(samples: dict, capacity: int) -> dict:
    if len(samples) <= int(capacity):
        return samples
    # Keep half recent and half deterministic historical samples.
    recent_count = int(capacity) // 2
    recent = sorted(
        samples,
        key=lambda key: (-int(samples[key].step), sample_priority(key)),
    )[:recent_count]
    recent_set = set(recent)
    historical = sorted(
        (key for key in samples if key not in recent_set),
        key=sample_priority,
    )[: int(capacity) - len(recent)]
    keep = recent + historical
    return {key: samples[key] for key in keep}


def newest_bundle(samples: dict, capacity: int) -> list[RewardSample]:
    return [
        samples[key]
        for key in sorted(
            samples,
            key=lambda key: (-int(samples[key].step), sample_priority(key)),
        )[: int(capacity)]
    ]


def fresh_client(
    base,
    device: torch.device,
    learning_rate: float,
    *,
    reset_head: bool = False,
    initialization_seed: int = 20260729,
) -> HeadClient:
    head = copy.deepcopy(base.gain_head).to(device)
    if bool(reset_head):
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(initialization_seed))
            for module in head.modules():
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()
    initial = {
        name: value.detach().clone()
        for name, value in head.state_dict().items()
    }
    return HeadClient(
        head=head,
        optimizer=torch.optim.Adam(
            head.parameters(), lr=float(learning_rate), weight_decay=1.0e-6
        ),
        initial=initial,
    )


def train_head(
    client: HeadClient,
    *,
    device: torch.device,
    reward_mean: float,
    reward_scale: float,
    minimum_samples: int,
    epochs: int,
    anchor_weight: float,
) -> bool:
    rows = list(client.samples.values())
    if len(rows) < int(minimum_samples):
        return False
    rewards = np.asarray([row.reward for row in rows], dtype=np.float64)
    if float(np.std(rewards)) < 1.0e-3:
        return False
    features = torch.stack([row.feature for row in rows]).to(device)
    targets = torch.tensor(
        (rewards - float(reward_mean)) / max(float(reward_scale), 1.0e-3),
        dtype=torch.float32,
        device=device,
    )
    for _ in range(int(epochs)):
        client.head.train()
        client.optimizer.zero_grad(set_to_none=True)
        prediction = client.head(features).squeeze(-1)
        pointwise = F.smooth_l1_loss(prediction, targets)
        anchor = prediction.new_zeros(())
        for name, parameter in client.head.named_parameters():
            anchor = anchor + (parameter - client.initial[name].to(device)).square().mean()
        loss = pointwise + float(anchor_weight) * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(client.head.parameters(), 5.0)
        client.optimizer.step()
    client.updates += 1
    client.last_trained_count = len(rows)
    return True


def encode_features(base, dataset: AuditDataset, device: torch.device) -> list[torch.Tensor]:
    base.eval()
    with torch.inference_mode():
        embedding = base.encode_many(dataset.states)
        return [
            base.pair_features(
                embedding[row.receiver_state], embedding[row.provider_state]
            ).detach().cpu()
            for row in dataset.examples
        ]


def base_scores(base, features: list[torch.Tensor], device: torch.device) -> np.ndarray:
    if not features:
        return np.empty(0, dtype=np.float64)
    with torch.inference_mode():
        return (
            base.gain_head(torch.stack(features).to(device))
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def head_scores(head, rows: list[torch.Tensor], device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        return (
            head(torch.stack(rows).to(device))
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def exchange_samples(
    clients: dict[int, HeadClient],
    neighbors: dict[int, set[int]],
    *,
    bundle_capacity: int,
    replay_capacity: int,
) -> None:
    snapshots = {
        node: dict(client.samples) for node, client in clients.items()
    }
    for node, peers in neighbors.items():
        client = clients[node]
        merged = dict(client.samples)
        before = len(merged)
        for peer in sorted(peers):
            for sample in newest_bundle(snapshots.get(peer, {}), bundle_capacity):
                merged.setdefault(sample.sample_id, sample)
        client.samples = retain(merged, replay_capacity)
        client.received += max(0, len(client.samples) - before)


def consensus_heads(
    clients: dict[int, HeadClient], neighbors: dict[int, set[int]]
) -> None:
    snapshots = {
        node: {
            name: value.detach().clone()
            for name, value in client.head.state_dict().items()
        }
        for node, client in clients.items()
    }
    updates: dict[int, dict[str, torch.Tensor]] = {}
    for node, client in clients.items():
        members = [node] + sorted(neighbors.get(node, set()))
        weights = [max(1, len(clients[member].samples)) for member in members]
        total = float(sum(weights))
        updates[node] = {
            name: sum(
                snapshots[member][name] * (float(weight) / total)
                for member, weight in zip(members, weights)
            )
            for name in snapshots[node]
        }
    for node, state in updates.items():
        clients[node].head.load_state_dict(state)
        clients[node].optimizer.state.clear()


def diagnostics(clients: dict[int, HeadClient], minimum_samples: int) -> dict[str, float | int]:
    counts = np.asarray([len(client.samples) for client in clients.values()], dtype=np.float64)
    reward_std = np.asarray(
        [
            float(np.std([row.reward for row in client.samples.values()]))
            if len(client.samples) > 1
            else 0.0
            for client in clients.values()
        ],
        dtype=np.float64,
    )
    if counts.size == 0:
        return {}
    return {
        "vehicles": int(counts.size),
        "mean_samples": float(np.mean(counts)),
        "median_samples": float(np.median(counts)),
        "minimum_samples": int(np.min(counts)),
        "maximum_samples": int(np.max(counts)),
        "fraction_with_minimum_samples": float(np.mean(counts >= minimum_samples)),
        "fraction_with_reward_variance": float(np.mean(reward_std >= 1.0e-3)),
        "mean_training_updates": float(
            np.mean([client.updates for client in clients.values()])
        ),
        "vehicles_trained": int(sum(client.updates > 0 for client in clients.values())),
        "mean_received_samples": float(
            np.mean([client.received for client in clients.values()])
        ),
    }


def score_holdout(
    clients: dict[int, HeadClient],
    base,
    dataset: AuditDataset,
    features: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(dataset.examples):
        head = clients[row.receiver].head if row.receiver in clients else base.gain_head
        values.append(float(head_scores(head, [features[index]], device)[0]))
    return np.asarray(values, dtype=np.float64)


def run_fold(
    train: AuditDataset,
    holdout: AuditDataset,
    checkpoint: dict,
    *,
    variant: str,
    seed: int,
    device: torch.device,
    reward_mean: float,
    reward_scale: float,
    minimum_samples: int,
    replay_capacity: int,
    bundle_capacity: int,
    learning_rate: float,
    head_epochs: int,
    anchor_weight: float,
    observed_rewards: np.ndarray | None = None,
    warmup_steps: int = 0,
) -> dict[str, object]:
    base = make_policy(checkpoint, train, device, load_weights=True)
    train_features = encode_features(base, train, device)
    holdout_features = encode_features(base, holdout, device)
    if variant == "frozen":
        score = base_scores(base, holdout_features, device)
        return {
            "variant": variant,
            "oracle_provider_audit": evaluate_scores(holdout, score, "oracle"),
            "deployable_geometry_alpha": evaluate_scores(
                holdout, score, "geometry"
            ),
            "sample_diagnostics": {},
        }

    clients: dict[int, HeadClient] = {}
    rng = random.Random(int(seed))
    by_step: dict[int, dict[tuple[str, int], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(train.examples):
        by_step[row.step][(row.mode, row.receiver)].append(index)
    acquired: set[tuple[int, int, str, int, int]] = set()
    online_selected: list[float] = []
    online_random: list[float] = []
    for step in sorted(by_step):
        opportunities = by_step[step]
        neighbors: dict[int, set[int]] = defaultdict(set)
        active: set[int] = set()
        for (_mode, receiver), indices in opportunities.items():
            active.add(int(receiver))
            for index in indices:
                provider = int(train.examples[index].provider)
                active.add(provider)
                neighbors[int(receiver)].add(provider)
                neighbors[provider].add(int(receiver))
        for node in active:
            clients.setdefault(
                node,
                fresh_client(
                    base,
                    device,
                    learning_rate,
                    reset_head="fresh" in variant,
                ),
            )
        if "shared" in variant:
            exchange_samples(
                clients,
                neighbors,
                bundle_capacity=bundle_capacity,
                replay_capacity=replay_capacity,
            )
        for node in sorted(active):
            client = clients[node]
            if len(client.samples) == client.last_trained_count:
                continue
            train_head(
                client,
                device=device,
                reward_mean=reward_mean,
                reward_scale=reward_scale,
                minimum_samples=minimum_samples,
                epochs=head_epochs,
                anchor_weight=anchor_weight,
            )
        if "consensus" in variant:
            consensus_heads(clients, neighbors)

        pending: list[tuple[int, RewardSample]] = []
        for (mode, receiver), indices in sorted(opportunities.items()):
            client = clients[int(receiver)]
            scores = head_scores(
                client.head, [train_features[index] for index in indices], device
            )
            warmup = int(step) <= int(warmup_steps)
            chosen_position = (
                rng.randrange(len(indices))
                if warmup or rng.random() < 0.02
                else int(np.argmax(scores))
            )
            chosen_index = indices[chosen_position]
            row = train.examples[chosen_index]
            gains = [float(train.examples[index].geometry_gain) for index in indices]
            online_selected.append(gains[chosen_position])
            online_random.append(float(np.mean(gains)))
            sample_id = (
                int(row.seed), int(row.step), str(mode), int(receiver), int(row.provider)
            )
            sample = RewardSample(
                sample_id=sample_id,
                feature=train_features[chosen_index],
                reward=float(
                    row.geometry_reward
                    if observed_rewards is None
                    else observed_rewards[chosen_index]
                ),
                step=int(step),
            )
            pending.append((int(receiver), sample))
            acquired.add(sample_id)
        for receiver, sample in pending:
            clients[receiver].samples[sample.sample_id] = sample
            clients[receiver].samples = retain(
                clients[receiver].samples, replay_capacity
            )

    # Make the last acquired rewards available to the final held-out decision.
    for client in clients.values():
        if len(client.samples) != client.last_trained_count:
            train_head(
                client,
                device=device,
                reward_mean=reward_mean,
                reward_scale=reward_scale,
                minimum_samples=minimum_samples,
                epochs=head_epochs,
                anchor_weight=anchor_weight,
            )
    score = score_holdout(clients, base, holdout, holdout_features, device)
    return {
        "variant": variant,
        "oracle_provider_audit": evaluate_scores(holdout, score, "oracle"),
        "deployable_geometry_alpha": evaluate_scores(
            holdout, score, "geometry"
        ),
        "online_geometry_gain_over_random_db": float(
            np.mean(online_selected) - np.mean(online_random)
        ),
        "unique_reward_samples_acquired": int(len(acquired)),
        "warmup_steps": int(warmup_steps),
        "sample_diagnostics": diagnostics(clients, minimum_samples),
    }


def pooled_upper_bound(
    train: AuditDataset,
    holdout: AuditDataset,
    checkpoint: dict,
    *,
    seed: int,
    device: torch.device,
    reward_mean: float,
    reward_scale: float,
    minimum_samples: int,
    learning_rate: float,
    anchor_weight: float,
) -> dict[str, object]:
    base = make_policy(checkpoint, train, device, load_weights=True)
    train_features = encode_features(base, train, device)
    holdout_features = encode_features(base, holdout, device)
    rng = random.Random(int(seed))
    grouped: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(train.examples):
        grouped[(row.step, row.mode, row.receiver)].append(index)
    frozen = base_scores(base, train_features, device)
    samples: dict = {}
    for (_step, mode, receiver), indices in sorted(grouped.items()):
        chosen_position = (
            rng.randrange(len(indices))
            if rng.random() < 0.02
            else int(np.argmax(frozen[np.asarray(indices, dtype=np.int64)]))
        )
        index = indices[chosen_position]
        row = train.examples[index]
        sample_id = (row.seed, row.step, mode, receiver, row.provider)
        samples[sample_id] = RewardSample(
            sample_id, train_features[index], float(row.geometry_reward), int(row.step)
        )
    client = fresh_client(base, device, learning_rate)
    client.samples = samples
    train_head(
        client,
        device=device,
        reward_mean=reward_mean,
        reward_scale=reward_scale,
        minimum_samples=minimum_samples,
        epochs=40,
        anchor_weight=anchor_weight,
    )
    score = head_scores(client.head, holdout_features, device)
    return {
        "variant": "pooled_replay_upper_bound",
        "oracle_provider_audit": evaluate_scores(holdout, score, "oracle"),
        "unique_reward_samples_acquired": len(samples),
        "sample_reward_std": float(
            np.std([sample.reward for sample in samples.values()])
        ),
    }


def source_reward_statistics(root: Path) -> tuple[float, float, int]:
    values: list[float] = []
    for audit in sorted(root.glob("source_train_*/seed_*/audit")):
        case_id = len(values) + 1
        dataset = load_seed(audit, case_id)
        values.extend(float(row.geometry_reward) for row in dataset.examples)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise FileNotFoundError(f"No source audits under {root}")
    return float(np.mean(array)), max(float(np.std(array)), 1.0e-3), int(array.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=64)
    parser.add_argument("--bundle-capacity", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    checkpoint_path, _candidates = checkpoint_candidates(args.policy_root)
    checkpoint = load_checkpoint(checkpoint_path)
    reward_mean, reward_scale, reward_rows = source_reward_statistics(args.source_root)
    variants = (
        "frozen",
        "local_replay",
        "shared_replay",
        "shared_replay_head_consensus",
    )
    report: dict[str, object] = {
        "protocol": {
            "checkpoint": str(checkpoint_path),
            "encoder": "source-pretrained and permanently frozen",
            "online_training_label": "selected pull post-pull parameter-geometry reward",
            "target_oracle_labels": "held-out evaluation only",
            "sample_payload": "receiver embedding, provider embedding, reward, id, step",
            "raw_measurements_or_positions_shared": False,
            "gossip": "one-hop causal bundles at recorded feasible contacts",
            "reward_normalization_source": "artificial source maps only",
            "reward_mean": reward_mean,
            "reward_scale": reward_scale,
            "reward_rows": reward_rows,
            "minimum_samples": int(args.minimum_samples),
            "replay_capacity": int(args.replay_capacity),
            "bundle_capacity": int(args.bundle_capacity),
            "learning_rate": float(args.learning_rate),
            "head_epochs_per_checkpoint": int(args.head_epochs),
            "anchor_weight": float(args.anchor_weight),
        },
        "targets": {},
    }
    for target_name, directory in (
        ("tiny", "source_tiny"),
        ("kirchberg", "target_kirchberg"),
    ):
        first = transform_target(
            load_seed(args.target_root / directory / "seed_01", 1), 1, "delta"
        )
        second = transform_target(
            load_seed(args.target_root / directory / "seed_02", 2), 2, "delta"
        )
        folds: list[dict[str, object]] = []
        for fold, (train, holdout) in enumerate(((first, second), (second, first))):
            for variant in variants:
                row = run_fold(
                    train,
                    holdout,
                    checkpoint,
                    variant=variant,
                    seed=20260728 + fold,
                    device=device,
                    reward_mean=reward_mean,
                    reward_scale=reward_scale,
                    minimum_samples=args.minimum_samples,
                    replay_capacity=args.replay_capacity,
                    bundle_capacity=args.bundle_capacity,
                    learning_rate=args.learning_rate,
                    head_epochs=args.head_epochs,
                    anchor_weight=args.anchor_weight,
                )
                row["fold"] = fold
                folds.append(row)
                metric = row["oracle_provider_audit"]
                print(
                    f"[{target_name}] fold={fold} variant={variant} "
                    f"gain={metric['gain_over_random_db']:.3f} "
                    f"order={metric['pairwise_order_accuracy']:.3f}",
                    flush=True,
                )
            upper = pooled_upper_bound(
                train,
                holdout,
                checkpoint,
                seed=20260728 + fold,
                device=device,
                reward_mean=reward_mean,
                reward_scale=reward_scale,
                minimum_samples=args.minimum_samples,
                learning_rate=args.learning_rate,
                anchor_weight=args.anchor_weight,
            )
            upper["fold"] = fold
            folds.append(upper)
            metric = upper["oracle_provider_audit"]
            print(
                f"[{target_name}] fold={fold} variant=pooled_replay_upper_bound "
                f"gain={metric['gain_over_random_db']:.3f} "
                f"order={metric['pairwise_order_accuracy']:.3f}",
                flush=True,
            )
        summary: dict[str, object] = {}
        for variant in (*variants, "pooled_replay_upper_bound"):
            selected = [row for row in folds if row["variant"] == variant]
            summary[variant] = {
                "oracle_provider_audit": averaged_metrics(
                    [row["oracle_provider_audit"] for row in selected]
                ),
                "deployable_geometry_alpha": averaged_metrics(
                    [row["deployable_geometry_alpha"] for row in selected]
                ),
            }
            if variant not in {"frozen", "pooled_replay_upper_bound"}:
                diagnostics_rows = [row["sample_diagnostics"] for row in selected]
                keys = diagnostics_rows[0].keys()
                summary[variant]["sample_diagnostics"] = {
                    key: float(np.mean([float(row[key]) for row in diagnostics_rows]))
                    for key in keys
                }
        report["targets"][target_name] = {"folds": folds, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
