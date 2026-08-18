#!/usr/bin/env python3
"""Deployment audit for a mature requester-defined encoder and policy capsule.

The complete capsule is trained and selected only on artificial source maps.
Target-map oracle labels are evaluation-only. Online controls may learn only
from the post-pull parameter-geometry reward that is available in deployment.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from online_policy_learning.build_cross_map_policy_dataset import _make_predictor
from online_policy_learning.online_local_validation_policy import (
    ExactModelTrajectoryPolicy,
    ExactPrivateState,
    exact_model_groups,
)
from online_policy_learning.train_cross_map_policy import _transform_groups
from cross_map_policy_generalization.train_cross_map_encoder_audit import (
    AuditDataset,
    averaged_metrics,
    combine,
    evaluate_scores,
    load_seed,
    predict,
)


@dataclass(frozen=True)
class RawSample:
    receiver: ExactPrivateState
    provider: ExactPrivateState
    target: float


@dataclass
class Client:
    policy: ExactModelTrajectoryPolicy
    optimizer: torch.optim.Optimizer | None
    replay: list[RawSample] = field(default_factory=list)
    pending: list[RawSample] = field(default_factory=list)
    pulls: int = 0


def checkpoint_candidates(root: Path) -> tuple[Path, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for directory in sorted(root.glob("trained_policy*_seed_*")):
        report_path = directory / "training_report.json"
        checkpoint = directory / "policy_best_validation.pt"
        if not report_path.exists() or not checkpoint.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "directory": str(directory),
                "checkpoint": str(checkpoint),
                "best_validation_uplift": float(report["best_validation_uplift"]),
                "source_maps": report["source_maps"],
                "validation_maps": report["validation_maps"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed policy training under {root}")
    chosen = max(rows, key=lambda row: float(row["best_validation_uplift"]))
    return Path(str(chosen["checkpoint"])), rows


def load_checkpoint(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "cross_map_pretrained_exact_policy_v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    return payload


def transform_target(
    dataset: AuditDataset,
    seed: int,
    representation: str,
) -> AuditDataset:
    initial_model = _make_predictor(
        int(seed), torch.device("cpu"), include_time=False
    )
    initial_groups = exact_model_groups(initial_model.state_dict())
    states = [
        ExactPrivateState(
            model_groups=_transform_groups(
                state.model_groups, initial_groups, representation
            ),
            trajectory=state.trajectory,
        )
        for state in dataset.states
    ]
    return AuditDataset(
        states=states,
        examples=dataset.examples,
        group_widths=dataset.group_widths,
        trajectory_dim=dataset.trajectory_dim,
    )


def make_policy(
    checkpoint: dict[str, object],
    dataset: AuditDataset,
    device: torch.device,
    *,
    load_weights: bool,
    seed: int = 0,
) -> ExactModelTrajectoryPolicy:
    architecture = checkpoint["architecture"]
    expected_widths = tuple(int(value) for value in architecture["group_widths"])
    if expected_widths != dataset.group_widths:
        raise ValueError(f"Predictor group mismatch: {expected_widths} != {dataset.group_widths}")
    if int(architecture["trajectory_dim"]) != dataset.trajectory_dim:
        raise ValueError("Trajectory summary dimension mismatch")
    torch.manual_seed(int(seed))
    policy = ExactModelTrajectoryPolicy(
        group_widths=expected_widths,
        trajectory_dim=int(architecture["trajectory_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        embedding_dim=int(architecture["embedding_dim"]),
        gain_hidden_dim=int(architecture["gain_hidden_dim"]),
        pair_feature_mode=str(architecture["pair_feature_mode"]),
    ).to(device)
    if load_weights:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy


def requester_score(
    policy: ExactModelTrajectoryPolicy,
    receiver: ExactPrivateState,
    provider: ExactPrivateState,
) -> float:
    policy.eval()
    with torch.inference_mode():
        receiver_embedding = policy.encode(receiver)
        provider_embedding = policy.encode(provider)
        result = policy.score_embeddings(
            receiver_embedding.unsqueeze(0), provider_embedding.unsqueeze(0)
        )
    return float(result.item())


def fresh_client(
    checkpoint: dict[str, object],
    dataset: AuditDataset,
    device: torch.device,
    variant: str,
) -> Client:
    policy = make_policy(checkpoint, dataset, device, load_weights=True)
    if variant == "head_raw_replay":
        for name, parameter in policy.named_parameters():
            if not name.startswith("gain_head."):
                parameter.requires_grad_(False)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = None if variant == "frozen" else torch.optim.Adam(
        parameters, lr=5.0e-4, weight_decay=1.0e-6
    )
    return Client(policy=policy, optimizer=optimizer)


def train_client(client: Client, samples: list[RawSample], epochs: int) -> None:
    if not samples or client.optimizer is None:
        return
    device = next(client.policy.parameters()).device
    for _ in range(int(epochs)):
        client.policy.train()
        client.optimizer.zero_grad(set_to_none=True)
        receiver = client.policy.encode_many([row.receiver for row in samples])
        provider = client.policy.encode_many([row.provider for row in samples])
        score = client.policy.score_embeddings(receiver, provider)
        target = torch.tensor(
            [row.target for row in samples], dtype=torch.float32, device=device
        )
        loss = F.smooth_l1_loss(score, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in client.policy.parameters() if parameter.requires_grad],
            5.0,
        )
        client.optimizer.step()


def score_with_clients(
    clients: dict[int, Client],
    base: ExactModelTrajectoryPolicy,
    dataset: AuditDataset,
) -> np.ndarray:
    values: list[float] = []
    for row in dataset.examples:
        policy = clients[row.receiver].policy if row.receiver in clients else base
        values.append(
            requester_score(
                policy,
                dataset.states[row.receiver_state],
                dataset.states[row.provider_state],
            )
        )
    return np.asarray(values, dtype=np.float64)


def run_online_fold(
    train: AuditDataset,
    holdout: AuditDataset,
    checkpoint: dict[str, object],
    device: torch.device,
    *,
    variant: str,
    seed: int,
) -> dict[str, object]:
    base = make_policy(checkpoint, train, device, load_weights=True)
    clients: dict[int, Client] = {}
    rng = random.Random(int(seed))
    opportunities: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(train.examples):
        opportunities[(row.step, row.mode, row.receiver)].append(index)
    online_selected: list[float] = []
    online_random: list[float] = []
    for (_step, _mode, receiver), indices in sorted(opportunities.items()):
        if receiver not in clients:
            clients[receiver] = fresh_client(checkpoint, train, device, variant)
        client = clients[receiver]
        scores = [
            requester_score(
                client.policy,
                train.states[train.examples[index].receiver_state],
                train.states[train.examples[index].provider_state],
            )
            for index in indices
        ]
        chosen_position = (
            rng.randrange(len(indices))
            if rng.random() < 0.02
            else int(np.argmax(scores))
        )
        chosen_index = indices[chosen_position]
        row = train.examples[chosen_index]
        true_values = [float(train.examples[index].geometry_gain) for index in indices]
        online_selected.append(true_values[chosen_position])
        online_random.append(float(np.mean(true_values)))
        if variant == "frozen":
            continue
        sample = RawSample(
            receiver=train.states[row.receiver_state],
            provider=train.states[row.provider_state],
            target=float(np.clip(row.geometry_reward, -1.0, 1.0)),
        )
        client.pending.append(sample)
        client.pulls += 1
        if variant in {"full_raw_replay", "head_raw_replay"}:
            client.replay.append(sample)
            client.replay = client.replay[-64:]
        if len(client.pending) >= 4:
            if variant == "full_recent_discard":
                train_client(client, list(client.pending), epochs=12)
            else:
                train_client(client, list(client.replay), epochs=6)
            client.pending.clear()
    for client in clients.values():
        if not client.pending or variant == "frozen":
            continue
        if variant == "full_recent_discard":
            train_client(client, list(client.pending), epochs=12)
        else:
            train_client(client, list(client.replay), epochs=6)
        client.pending.clear()
    holdout_scores = score_with_clients(clients, base, holdout)
    return {
        "variant": variant,
        "oracle_provider_audit": evaluate_scores(holdout, holdout_scores, "oracle"),
        "deployable_geometry_alpha": evaluate_scores(holdout, holdout_scores, "geometry"),
        "online_geometry_gain_over_random_db": float(np.mean(online_selected) - np.mean(online_random)),
        "clients": len(clients),
        "pulls": int(sum(client.pulls for client in clients.values())),
    }


def online_audit(
    first: AuditDataset,
    second: AuditDataset,
    checkpoint: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    variants = ("frozen", "full_recent_discard", "full_raw_replay", "head_raw_replay")
    rows: list[dict[str, object]] = []
    for fold, (train, holdout) in enumerate(((first, second), (second, first))):
        for variant in variants:
            row = run_online_fold(
                train,
                holdout,
                checkpoint,
                device,
                variant=variant,
                seed=20260727 + fold,
            )
            row["fold"] = fold
            rows.append(row)
            metric = row["oracle_provider_audit"]
            gain = float(metric["gain_over_random_db"])
            order = float(metric["pairwise_order_accuracy"])
            print(
                f"[ONLINE] fold={fold} variant={variant} "
                f"gain={gain:.3f} order={order:.3f}",
                flush=True,
            )
    summary: dict[str, object] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "oracle_provider_audit": averaged_metrics(
                [row["oracle_provider_audit"] for row in selected]
            ),
            "deployable_geometry_alpha": averaged_metrics(
                [row["deployable_geometry_alpha"] for row in selected]
            ),
            "online_geometry_gain_over_random_db": float(
                np.mean([row["online_geometry_gain_over_random_db"] for row in selected])
            ),
        }
    return {"folds": rows, "summary": summary}


def frozen_audit(
    dataset: AuditDataset,
    checkpoint: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    policy = make_policy(checkpoint, dataset, device, load_weights=True)
    scores = predict(policy, dataset, device)
    return {
        "oracle_provider_audit": evaluate_scores(dataset, scores, "oracle"),
        "deployable_geometry_alpha": evaluate_scores(dataset, scores, "geometry"),
    }


def random_encoder_control(
    dataset: AuditDataset,
    checkpoint: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for seed in range(8):
        policy = make_policy(
            checkpoint, dataset, device, load_weights=False, seed=20260800 + seed
        )
        rows.append(evaluate_scores(dataset, predict(policy, dataset, device), "oracle"))
    return {"mean": averaged_metrics(rows), "repeats": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    checkpoint_path, candidates = checkpoint_candidates(args.policy_root)
    checkpoint = load_checkpoint(checkpoint_path)
    raw_tiny_one = load_seed(args.target_root / "source_tiny" / "seed_01", 1)
    raw_tiny_two = load_seed(args.target_root / "source_tiny" / "seed_02", 2)
    raw_kirch_one = load_seed(
        args.target_root / "target_kirchberg" / "seed_01", 1
    )
    raw_kirch_two = load_seed(
        args.target_root / "target_kirchberg" / "seed_02", 2
    )
    candidate_target_audits: dict[str, object] = {}
    for candidate in candidates:
        candidate_checkpoint = load_checkpoint(Path(str(candidate["checkpoint"])))
        candidate_representation = str(
            candidate_checkpoint["architecture"].get(
                "model_input_representation", "raw"
            )
        )
        candidate_tiny = combine(
            [
                transform_target(raw_tiny_one, 1, candidate_representation),
                transform_target(raw_tiny_two, 2, candidate_representation),
            ]
        )
        candidate_kirchberg = combine(
            [
                transform_target(raw_kirch_one, 1, candidate_representation),
                transform_target(raw_kirch_two, 2, candidate_representation),
            ]
        )
        tiny_result = frozen_audit(
            candidate_tiny, candidate_checkpoint, device
        )
        kirchberg_result = frozen_audit(
            candidate_kirchberg, candidate_checkpoint, device
        )
        candidate_target_audits[str(candidate["checkpoint"])] = {
            "representation": candidate_representation,
            "tiny": tiny_result,
            "kirchberg": kirchberg_result,
        }
        tiny_gain = float(
            tiny_result["oracle_provider_audit"]["gain_over_random_db"]
        )
        kirchberg_gain = float(
            kirchberg_result["oracle_provider_audit"]["gain_over_random_db"]
        )
        print(
            f"[ZERO-SHOT] representation={candidate_representation} "
            f"tiny={tiny_gain:.3f} kirchberg={kirchberg_gain:.3f}",
            flush=True,
        )
    representation = str(
        checkpoint["architecture"].get("model_input_representation", "raw")
    )
    tiny_one = transform_target(raw_tiny_one, 1, representation)
    tiny_two = transform_target(raw_tiny_two, 2, representation)
    kirch_one = transform_target(raw_kirch_one, 1, representation)
    kirch_two = transform_target(raw_kirch_two, 2, representation)
    tiny = combine([tiny_one, tiny_two])
    kirchberg = combine([kirch_one, kirch_two])
    report = {
        "protocol": {
            "policy_selection": "two artificial held-out maps only",
            "deployment_targets_used_for_training_or_selection": False,
            "requester_defined_remote_encoding": True,
            "pre_pull_provider_weights_transferred": False,
            "online_reward": "post-pull parameter geometry only; no validation data",
            "online_replay": "raw pulled model-state pairs are re-encoded; embeddings are never replayed",
            "policy_inheritance": "complete encoder and head capsule copied atomically",
            "source_private_trajectories_shared": False,
            "device": str(device),
        },
        "checkpoint_candidates": candidates,
        "candidate_target_audits_evaluation_only": candidate_target_audits,
        "selected_checkpoint": str(checkpoint_path),
        "checkpoint_metadata": {
            "architecture": checkpoint["architecture"],
            "source_maps": checkpoint["source_maps"],
            "validation_maps": checkpoint["validation_maps"],
            "epoch": checkpoint["epoch"],
            "decisions_seen": checkpoint["decisions_seen"],
        },
        "tiny_frozen_zero_shot": frozen_audit(tiny, checkpoint, device),
        "kirchberg_frozen_zero_shot": frozen_audit(kirchberg, checkpoint, device),
        "tiny_random_encoder_control": random_encoder_control(tiny, checkpoint, device),
        "kirchberg_random_encoder_control": random_encoder_control(kirchberg, checkpoint, device),
        "tiny_online_adaptation": online_audit(tiny_one, tiny_two, checkpoint, device),
        "kirchberg_online_adaptation": online_audit(kirch_one, kirch_two, checkpoint, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DONE] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
