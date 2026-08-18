"""Exact-input online encoder for sequential bidirectional aggregation.

This variant removes fixed parameter and trajectory sketches.  Every floating
predictor value is consumed by learned per-layer neuron encoders, and every
private sample is consumed in chronological order by a causal GRU.  Only the
final quantized embedding leaves the provider.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    mergeable_evidence_delta_state,
    is_mergeable_evidence_state,
    make_rssi_predictor,
    mergeable_evidence_diagonal_information_gain,
    mergeable_evidence_direction_nbytes,
)

from .local_validation_reward import (
    BidirectionalCrossValidationSimulation,
    PullResult,
    ZoneValidationState,
    interpolate_states,
    minimize_bounded,
    quality_weighted_loss,
)
from .token_window import VehicleTokenWindows
from .encoded_sequential import (
    EncodedSequentialBidirectionalSimulation,
    _ExperienceView,
)
from .realistic_network import TransferProposal, schedule_transfers


TensorState = Mapping[str, torch.Tensor]
PROVENANCE_SKETCH_DIM = 128


@dataclass(frozen=True)
class ExactPrivateState:
    """Complete predictor tensors and complete chronological sample history."""

    model_groups: tuple[torch.Tensor, ...]
    trajectory: torch.Tensor
    support: torch.Tensor | None = None

    def clone(self) -> "ExactPrivateState":
        return ExactPrivateState(
            model_groups=tuple(group.detach().cpu().clone() for group in self.model_groups),
            trajectory=self.trajectory.detach().cpu().clone(),
            support=(
                None
                if self.support is None
                else self.support.detach().cpu().clone()
            ),
        )


def exact_model_groups(state: TensorState) -> tuple[torch.Tensor, ...]:
    """Return fixed-shape predictor rows consumed by the policy encoder."""

    if "mergeable_format_version" in state:
        weight = state["weight"].detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(1, -1)
        precision_rows = state["evidence_precision"].detach().to(
            device="cpu", dtype=torch.float32
        )
        information_rows = state["evidence_information"].detach().to(
            device="cpu", dtype=torch.float32
        )
        masses = state["evidence_mass"].detach().to(
            device="cpu", dtype=torch.float32
        )
        versions = state["evidence_versions"].detach().to(
            device="cpu", dtype=torch.int64
        )
        basis_dim = int(weight.shape[1])
        if int(precision_rows.shape[0]) > 0:
            total_precision = precision_rows.sum(dim=0)
            total_information = information_rows.sum(dim=0)
        else:
            total_precision = torch.zeros(
                (basis_dim, basis_dim), dtype=torch.float32
            )
            total_information = torch.zeros(
                basis_dim, dtype=torch.float32
            )
        total_mass = float(masses.sum().item())
        scale = max(total_mass, 1.0)
        origins = int(precision_rows.shape[0])
        version_total = int(versions.sum().item())
        summary = torch.tensor(
            [
                math.log1p(origins),
                math.log1p(version_total),
                math.log1p(max(total_mass, 0.0)),
                math.log1p(
                    float(version_total) / float(max(origins, 1))
                ),
            ],
            dtype=torch.float32,
        ).reshape(1, 4)
        provenance = torch.zeros(
            PROVENANCE_SKETCH_DIM, dtype=torch.float32
        )
        for raw_key, raw_version in zip(
            state["evidence_keys"], state["evidence_versions"]
        ):
            key_bytes = int(raw_key).to_bytes(
                8, byteorder="little", signed=True
            )
            magnitude = math.log1p(
                max(0, int(raw_version))
            ) / math.sqrt(2.0)
            for salt in (b"\x00", b"\x01"):
                digest = hashlib.blake2b(
                    salt + key_bytes, digest_size=8
                ).digest()
                code = int.from_bytes(digest, byteorder="little")
                index = code % PROVENANCE_SKETCH_DIM
                sign = 1.0 if (code >> 63) == 0 else -1.0
                provenance[index] += sign * magnitude
        return (
            weight,
            total_precision / scale,
            (total_information / scale).reshape(1, -1),
            summary,
            provenance.reshape(1, -1),
        )

    consumed: set[str] = set()
    groups: list[torch.Tensor] = []
    for name, value in state.items():
        if name in consumed or not torch.is_tensor(value):
            continue
        # Support precision is transmitted with a predictor and used through
        # explicit model-history scalars. It must not masquerade as learned
        # network parameters in the model encoder.
        if "_support_" in name:
            consumed.add(name)
            continue
        if not torch.is_floating_point(value):
            consumed.add(name)
            continue
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        if name.endswith(".weight") and tensor.ndim == 2:
            bias_name = f"{name[:-7]}.bias"
            bias = state.get(bias_name)
            if (
                torch.is_tensor(bias)
                and torch.is_floating_point(bias)
                and bias.ndim == 1
                and int(bias.numel()) == int(tensor.shape[0])
            ):
                rows = torch.cat(
                    (
                        tensor,
                        bias.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .reshape(-1, 1),
                    ),
                    dim=1,
                )
                consumed.add(bias_name)
            else:
                rows = tensor
            groups.append(rows.clone())
        else:
            groups.append(tensor.reshape(1, -1).clone())
        consumed.add(name)
    if not groups:
        raise ValueError("predictor state contains no floating-point parameters")
    return tuple(groups)


@dataclass
class ExactTrajectoryHistory:
    """Bounded, spatially balanced private training trajectory.

    Rows are retained without a recency preference. The four normalized link
    coordinates define a coarse spatial cell; capacity is shared as evenly as
    possible across occupied cells, and deterministic bottom-k sampling is
    used within a cell. Retained rows are presented to the causal encoder in
    their original chronological order.
    """

    capacity: int = 256
    spatial_bins: int = 4
    rows: list[list[float]] = field(default_factory=list)
    sequences: list[int] = field(default_factory=list)
    samples_seen: int = 0
    _cache: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.capacity = int(self.capacity)
        self.spatial_bins = int(self.spatial_bins)
        if self.capacity <= 0:
            raise ValueError("trajectory capacity must be positive")
        if self.spatial_bins <= 1:
            raise ValueError("trajectory spatial_bins must exceed one")
        if not self.sequences and self.rows:
            self.sequences = list(range(len(self.rows)))
        if len(self.sequences) != len(self.rows):
            raise ValueError("trajectory rows and sequence IDs must align")
        self.samples_seen = max(
            int(self.samples_seen),
            (max(self.sequences) + 1) if self.sequences else 0,
        )
        self._rebalance()

    def _cell(self, row: list[float]) -> tuple[int, int, int, int]:
        if len(row) < 4:
            raise ValueError("trajectory rows need four spatial coordinates")
        bins = int(self.spatial_bins)
        values = tuple(
            min(bins - 1, max(0, int(math.floor(float(value) * bins))))
            for value in row[:4]
        )
        return values  # type: ignore[return-value]

    @staticmethod
    def _priority(row: list[float], sequence: int) -> int:
        values = np.asarray(row, dtype=np.float32)
        payload = values.tobytes() + int(sequence).to_bytes(
            8, byteorder="little", signed=False
        )
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            byteorder="big",
            signed=False,
        )

    def _rebalance(self) -> None:
        grouped: dict[
            tuple[int, int, int, int], list[tuple[int, int, list[float]]]
        ] = defaultdict(list)
        for row, sequence in zip(self.rows, self.sequences):
            values = [float(value) for value in row]
            grouped[self._cell(values)].append(
                (self._priority(values, int(sequence)), int(sequence), values)
            )
        for entries in grouped.values():
            entries.sort(key=lambda item: (item[0], item[1]))

        selected: list[tuple[int, list[float]]] = []
        cells = sorted(grouped)
        depth = 0
        while len(selected) < self.capacity:
            added = False
            for cell in cells:
                entries = grouped[cell]
                if depth < len(entries):
                    _priority, sequence, row = entries[depth]
                    selected.append((sequence, row))
                    added = True
                    if len(selected) >= self.capacity:
                        break
            if not added:
                break
            depth += 1
        selected.sort(key=lambda item: item[0])
        self.sequences = [int(sequence) for sequence, _row in selected]
        self.rows = [list(row) for _sequence, row in selected]
        self._cache = None

    def append(
        self, features: np.ndarray, normalized_targets: np.ndarray
    ) -> None:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(normalized_targets, dtype=np.float32).reshape(-1, 1)
        if values.ndim != 2 or int(values.shape[0]) != int(targets.shape[0]):
            raise ValueError("trajectory features and targets must align")
        if int(values.shape[0]) == 0:
            return
        combined = np.concatenate((values, targets), axis=1)
        for row in combined:
            self.rows.append(row.tolist())
            self.sequences.append(int(self.samples_seen))
            self.samples_seen += 1
        self._rebalance()

    def tensor(self, *, width: int) -> torch.Tensor:
        if self._cache is None:
            self._cache = (
                torch.as_tensor(self.rows, dtype=torch.float32).reshape(-1, int(width))
                if self.rows
                else torch.empty((0, int(width)), dtype=torch.float32)
            )
        return self._cache

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "spatial_bins": int(self.spatial_bins),
            "rows": [list(row) for row in self.rows],
            "sequences": [int(value) for value in self.sequences],
            "samples_seen": int(self.samples_seen),
        }

    @classmethod
    def restore(
        cls, payload: object, *, capacity: int | None = None
    ) -> "ExactTrajectoryHistory":
        if not isinstance(payload, dict):
            return cls(capacity=256 if capacity is None else int(capacity))
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return cls(capacity=256 if capacity is None else int(capacity))
        sequences = payload.get("sequences", [])
        if not isinstance(sequences, list):
            sequences = []
        restored_capacity = (
            int(payload.get("capacity", 256))
            if capacity is None
            else int(capacity)
        )
        return cls(
            capacity=restored_capacity,
            spatial_bins=int(payload.get("spatial_bins", 4)),
            rows=[list(map(float, row)) for row in rows],
            sequences=[int(value) for value in sequences],
            samples_seen=int(payload.get("samples_seen", len(rows))),
        )


class ExactModelTrajectoryPolicy(nn.Module):
    """Learn directly from every model parameter and trajectory sample."""

    def __init__(
        self,
        *,
        group_widths: tuple[int, ...],
        trajectory_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        gain_hidden_dim: int | None = None,
        pair_feature_mode: str = "concat",
    ) -> None:
        super().__init__()
        if not group_widths:
            raise ValueError("at least one parameter group is required")
        hidden = int(hidden_dim)
        gain_hidden = (
            hidden if gain_hidden_dim is None else int(gain_hidden_dim)
        )
        if gain_hidden <= 0:
            raise ValueError("gain_hidden_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.pair_feature_mode = str(pair_feature_mode).strip().lower()
        if self.pair_feature_mode not in {"concat", "relational"}:
            raise ValueError("pair_feature_mode must be concat or relational")
        self.group_widths = tuple(int(width) for width in group_widths)
        self.row_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            for width in self.group_widths
        )
        self.row_attention = nn.ModuleList(
            nn.Linear(hidden, 1) for _ in self.group_widths
        )
        self.layer_encoder = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
        )
        self.trajectory_encoder = nn.GRU(
            input_size=int(trajectory_dim),
            hidden_size=hidden,
            batch_first=True,
        )
        self.empty_trajectory = nn.Parameter(torch.zeros(hidden))
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embedding_dim),
            nn.Tanh(),
        )
        pair_dim = (
            2 * self.embedding_dim
            if self.pair_feature_mode == "concat"
            else 5 * self.embedding_dim
        )
        self.gain_head = nn.Sequential(
            nn.Linear(pair_dim, gain_hidden),
            nn.SiLU(),
            nn.Linear(gain_hidden, 1),
        )

    @staticmethod
    def _quantize(values: torch.Tensor) -> torch.Tensor:
        maximum = values.detach().abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(maximum / 127.0, min=1.0e-8)
        dequantized = torch.clamp(
            torch.round(values / scale), -127.0, 127.0
        ) * scale
        return values + (dequantized - values).detach()

    def encode_model(self, groups: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(groups) != len(self.row_encoders):
            raise ValueError("model group count changed")
        layer_tokens: list[torch.Tensor] = []
        device = next(self.parameters()).device
        for group, expected, encoder, attention in zip(
            groups,
            self.group_widths,
            self.row_encoders,
            self.row_attention,
        ):
            rows = group.to(device=device, dtype=torch.float32)
            if rows.ndim != 2 or int(rows.shape[1]) != int(expected):
                raise ValueError("model parameter group shape changed")
            encoded = encoder(rows)
            weights = torch.softmax(attention(encoded).squeeze(-1), dim=0)
            layer_tokens.append((weights.unsqueeze(-1) * encoded).sum(dim=0))
        sequence = torch.stack(layer_tokens, dim=0).unsqueeze(0)
        _output, hidden = self.layer_encoder(sequence)
        return hidden[-1, 0]

    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        sequence = trajectory.to(device=device, dtype=torch.float32)
        if sequence.ndim != 2:
            raise ValueError("trajectory must have shape [samples, features]")
        if int(sequence.shape[0]) == 0:
            return self.empty_trajectory
        _output, hidden = self.trajectory_encoder(sequence.unsqueeze(0))
        return hidden[-1, 0]

    def fuse(
        self, model_embedding: torch.Tensor, trajectory_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self._quantize(
            self.fusion(torch.cat((model_embedding, trajectory_embedding), dim=-1))
        )

    def encode(self, state: ExactPrivateState) -> torch.Tensor:
        return self.fuse(
            self.encode_model(state.model_groups),
            self.encode_trajectory(state.trajectory),
        )

    def encode_many(
        self, states: list[ExactPrivateState]
    ) -> torch.Tensor:
        """Encode equal-architecture private states in one exact batch."""

        if not states:
            return self.empty_trajectory.new_empty((0, self.embedding_dim))
        device = next(self.parameters()).device
        layer_tokens: list[torch.Tensor] = []
        for group_index, (expected, encoder, attention) in enumerate(
            zip(self.group_widths, self.row_encoders, self.row_attention)
        ):
            rows = torch.stack(
                [state.model_groups[group_index] for state in states]
            ).to(device=device, dtype=torch.float32)
            if rows.ndim != 3 or int(rows.shape[2]) != int(expected):
                raise ValueError("model parameter group shape changed")
            encoded = encoder(rows)
            weights = torch.softmax(attention(encoded).squeeze(-1), dim=1)
            layer_tokens.append(
                (weights.unsqueeze(-1) * encoded).sum(dim=1)
            )
        model_sequence = torch.stack(layer_tokens, dim=1)
        _output, model_hidden = self.layer_encoder(model_sequence)
        model_embedding = model_hidden[-1]

        nonempty = [
            index
            for index, state in enumerate(states)
            if int(state.trajectory.shape[0]) > 0
        ]
        trajectory_by_index: dict[int, torch.Tensor] = {}
        if nonempty:
            sequences = [
                states[index].trajectory.to(
                    device=device, dtype=torch.float32
                )
                for index in nonempty
            ]
            lengths = torch.tensor(
                [int(sequence.shape[0]) for sequence in sequences],
                dtype=torch.int64,
                device="cpu",
            )
            padded = nn.utils.rnn.pad_sequence(
                sequences, batch_first=True
            )
            packed = nn.utils.rnn.pack_padded_sequence(
                padded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _output, trajectory_hidden = self.trajectory_encoder(packed)
            trajectory_by_index = {
                index: trajectory_hidden[-1, position]
                for position, index in enumerate(nonempty)
            }
        trajectory_embedding = torch.stack(
            [
                trajectory_by_index.get(index, self.empty_trajectory)
                for index in range(len(states))
            ],
            dim=0,
        )
        return self._quantize(
            self.fusion(
                torch.cat(
                    (model_embedding, trajectory_embedding), dim=-1
                )
            )
        )

    def pair_features(
        self, receiver: torch.Tensor, provider: torch.Tensor
    ) -> torch.Tensor:
        if self.pair_feature_mode == "concat":
            return torch.cat((receiver, provider), dim=-1)
        difference = provider - receiver
        return torch.cat(
            (
                receiver,
                provider,
                difference,
                torch.abs(difference),
                receiver * provider,
            ),
            dim=-1,
        )

    def score_embeddings(
        self, receiver: torch.Tensor, provider: torch.Tensor
    ) -> torch.Tensor:
        return self.gain_head(
            self.pair_features(receiver, provider)
        ).squeeze(-1)


class OnlineExactUtilityAgent:
    """One local exact-input policy used for decisions and local learning."""

    def __init__(
        self,
        *,
        group_widths: tuple[int, ...],
        trajectory_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        gain_hidden_dim: int | None,
        pair_feature_mode: str,
        device: torch.device,
        learning_rate: float,
        rng_seed: int,
        model_seed: int,
        sample_capacity: int = 512,
        encoder_lr_scale: float = 0.1,
        normalize_rewards: bool = False,
        reward_scale_db: float | None = None,
        ranking_loss_weight: float = 0.0,
        ranking_margin_db: float = 0.25,
        ranking_temperature_db: float = 1.0,
        ranking_receiver_cosine_min: float = 0.8,
        support_dim: int = 0,
    ) -> None:
        self.device = device
        self.learning_rate = float(learning_rate)
        self.sample_capacity = int(sample_capacity)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self.normalize_rewards = bool(normalize_rewards)
        self.reward_scale_db = (
            None if reward_scale_db is None else float(reward_scale_db)
        )
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_margin_db = float(ranking_margin_db)
        self.ranking_temperature_db = float(ranking_temperature_db)
        self.ranking_receiver_cosine_min = float(
            ranking_receiver_cosine_min
        )
        if self.sample_capacity <= 0:
            raise ValueError("policy sample capacity must be positive")
        if not 0.0 < self.encoder_lr_scale <= 1.0:
            raise ValueError("encoder_lr_scale must be in (0, 1]")
        if (
            self.reward_scale_db is not None
            and (
                not math.isfinite(self.reward_scale_db)
                or self.reward_scale_db <= 0.0
            )
        ):
            raise ValueError("reward_scale_db must be finite and positive")
        if not math.isfinite(self.ranking_loss_weight) or self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.ranking_margin_db) or self.ranking_margin_db < 0.0:
            raise ValueError("ranking_margin_db must be finite and nonnegative")
        if (
            not math.isfinite(self.ranking_temperature_db)
            or self.ranking_temperature_db <= 0.0
        ):
            raise ValueError("ranking_temperature_db must be finite and positive")
        if not -1.0 <= self.ranking_receiver_cosine_min <= 1.0:
            raise ValueError("ranking_receiver_cosine_min must be in [-1, 1]")
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(model_seed) & 0x7FFFFFFFFFFFFFFF)
            if int(support_dim) > 0:
                from .expert_bank_policy import SupportAugmentedExactPolicy

                self.policy = SupportAugmentedExactPolicy(
                    group_widths=group_widths,
                    trajectory_dim=int(trajectory_dim),
                    support_dim=int(support_dim),
                    hidden_dim=int(hidden_dim),
                    embedding_dim=int(embedding_dim),
                    gain_hidden_dim=(
                        int(hidden_dim)
                        if gain_hidden_dim is None
                        else int(gain_hidden_dim)
                    ),
                ).to(device)
            else:
                self.policy = ExactModelTrajectoryPolicy(
                    group_widths=group_widths,
                    trajectory_dim=int(trajectory_dim),
                    hidden_dim=int(hidden_dim),
                    embedding_dim=int(embedding_dim),
                    gain_hidden_dim=gain_hidden_dim,
                    pair_feature_mode=pair_feature_mode,
                ).to(device)
        encoder_parameters = [
            parameter
            for name, parameter in self.policy.named_parameters()
            if not name.startswith("gain_head.")
            and not name.startswith("exact.gain_head.")
        ]
        head_parameters = list(self.policy.gain_head.parameters())
        self.opt = torch.optim.Adam(
            [
                {
                    "params": encoder_parameters,
                    "lr": self.learning_rate * self.encoder_lr_scale,
                },
                {"params": head_parameters, "lr": self.learning_rate},
            ]
        )
        self._py_rng = random.Random(int(rng_seed))
        self._replay_rng = random.Random(int(rng_seed) + 1_000_003)
        self.action_policy = "argmax"
        self.batch_size = 1
        self.experience = 0
        self.local_evaluated_pulls = 0
        self.train_steps = 0
        self.head_replay_inputs: list[torch.Tensor] = []
        self.head_replay_targets: list[float] = []
        self.head_replay_ids: list[object | None] = []
        self.head_replay_steps: list[int] = []
        self.head_epoch_steps = 0
        self.shared_samples: dict[
            object, tuple[torch.Tensor, torch.Tensor, float]
        ] = {}
        # Event time is kept beside the wire-compatible sample tuple.  The
        # default policy still uses the historical bottom-k reservoir; the
        # corrected frozen-encoder variant enables a recent+historical split.
        self.shared_sample_steps: dict[object, int] = {}
        self.recent_sample_capacity = 0
        self.reward_balanced_replay_fraction = 0.0
        self.replay = _ExperienceView(self)  # type: ignore[arg-type]

    def configure_hybrid_samples(self, recent_capacity: int) -> None:
        """Keep recent samples plus an order-independent historical reservoir."""

        capacity = int(recent_capacity)
        if not 0 <= capacity <= self.sample_capacity:
            raise ValueError(
                "recent sample capacity must be between zero and sample capacity"
            )
        if self.shared_samples:
            raise ValueError("sample retention must be configured before use")
        self.recent_sample_capacity = capacity

    def configure_reward_balanced_replay(self, fraction: float) -> None:
        """Mix reward-sign-balanced and time-stratified replay batches.

        The remaining batches retain the naturally observed reward frequency,
        so a rare-positive bootstrap does not replace gain calibration.
        """

        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("balanced replay fraction must be in [0, 1]")
        self.reward_balanced_replay_fraction = value

    def encoder_named_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in self.policy.named_parameters()
            if not name.startswith("gain_head.")
            and not name.startswith("exact.gain_head.")
        }

    def encoder_nbytes(self) -> int:
        return int(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.encoder_named_parameters().values()
            )
        )

    def reset_encoder_optimizer_state(self) -> None:
        """Drop only local encoder Adam moments after consensus."""

        for parameter in self.encoder_named_parameters().values():
            self.opt.state.pop(parameter, None)

    def align_encoder_with(self, peer: "OnlineExactUtilityAgent") -> None:
        """Experience-average encoder parameters; never touch private heads."""

        own = self.encoder_named_parameters()
        other = peer.encoder_named_parameters()
        if own.keys() != other.keys():
            raise ValueError("policy encoder structures do not match")
        own_weight = max(0, int(self.local_evaluated_pulls))
        other_weight = max(0, int(peer.local_evaluated_pulls))
        total = own_weight + other_weight
        if total <= 0:
            own_fraction = 0.5
        else:
            own_fraction = float(own_weight) / float(total)
        with torch.no_grad():
            for name in own:
                aggregate = (
                    own[name].detach() * own_fraction
                    + other[name].detach() * (1.0 - own_fraction)
                )
                own[name].copy_(aggregate)
                other[name].copy_(aggregate)
        self.reset_encoder_optimizer_state()
        peer.reset_encoder_optimizer_state()

    def freeze_encoder(self) -> None:
        """Keep every seeded encoder in one stable embedding coordinate system."""

        for parameter in self.encoder_named_parameters().values():
            parameter.requires_grad_(False)

    def freeze_policy(self) -> None:
        """Freeze the complete source-pretrained decision policy."""

        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    def reward_normalization(self) -> tuple[float, float]:
        """Return local replay mean/scale without exposing replay samples."""

        if self.reward_scale_db is not None:
            return 0.0, float(self.reward_scale_db)
        if not self.normalize_rewards or not self.head_replay_targets:
            return 0.0, 1.0
        values = np.asarray(self.head_replay_targets, dtype=np.float64)
        mean = float(np.mean(values))
        scale = float(np.std(values)) if values.size > 1 else 1.0
        if not math.isfinite(scale) or scale < 1.0e-3:
            scale = 1.0
        return mean, scale

    def normalize_gain(self, value: float) -> float:
        mean, scale = self.reward_normalization()
        return (float(value) - mean) / scale

    def denormalize_gain(self, value: float) -> float:
        mean, scale = self.reward_normalization()
        return float(value) * scale + mean

    @staticmethod
    def _sample_key(sample_id: object) -> object:
        if isinstance(sample_id, bytearray):
            return bytes(sample_id)
        if isinstance(sample_id, list):
            return tuple(sample_id)
        try:
            hash(sample_id)
        except TypeError:
            return repr(sample_id)
        return sample_id

    @classmethod
    def sample_priority(cls, sample_id: object) -> int:
        key = cls._sample_key(sample_id)
        payload = key if isinstance(key, bytes) else repr(key).encode("utf-8")
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            byteorder="big",
            signed=False,
        )

    def ordered_shared_samples(
        self,
    ) -> list[tuple[object, tuple[torch.Tensor, torch.Tensor, float]]]:
        return sorted(
            self.shared_samples.items(),
            key=lambda item: (self.sample_priority(item[0]), repr(item[0])),
        )

    def _hybrid_keep_keys(
        self,
        *,
        candidate_key: object | None = None,
        candidate_step: int = 0,
    ) -> set[object]:
        steps = dict(self.shared_sample_steps)
        keys = set(self.shared_samples)
        if candidate_key is not None:
            keys.add(candidate_key)
            steps[candidate_key] = int(candidate_step)
        recent_count = min(int(self.recent_sample_capacity), len(keys))
        recent = sorted(
            keys,
            key=lambda key: (
                -int(steps.get(key, 0)),
                self.sample_priority(key),
                repr(key),
            ),
        )[:recent_count]
        recent_set = set(recent)
        historical_capacity = max(0, self.sample_capacity - recent_count)
        historical = sorted(
            (key for key in keys if key not in recent_set),
            key=lambda key: (self.sample_priority(key), repr(key)),
        )[:historical_capacity]
        return recent_set | set(historical)

    def balanced_shared_samples(
        self, limit: int
    ) -> list[tuple[object, tuple[torch.Tensor, torch.Tensor, float]]]:
        """Return a recent, sign-balanced bootstrap bundle."""

        count = max(0, int(limit))
        if count == 0 or not self.shared_samples:
            return []
        newest = sorted(
            self.shared_samples,
            key=lambda key: (
                -int(self.shared_sample_steps.get(key, 0)),
                self.sample_priority(key),
                repr(key),
            ),
        )
        selected: list[object] = newest[: min(len(newest), count // 2)]
        selected_set = set(selected)
        remaining_slots = count - len(selected)
        negative_slots = remaining_slots // 2
        positive_slots = remaining_slots - negative_slots
        positive = [
            key
            for key in newest
            if key not in selected_set
            and float(self.shared_samples[key][2]) > 0.0
        ][:positive_slots]
        selected.extend(positive)
        selected_set.update(positive)
        nonpositive = [
            key
            for key in newest
            if key not in selected_set
            and float(self.shared_samples[key][2]) <= 0.0
        ][:negative_slots]
        selected.extend(nonpositive)
        selected_set.update(nonpositive)
        if len(selected) < count:
            selected.extend(
                key
                for key in newest
                if key not in selected_set
            )
        return [
            (key, self.shared_samples[key]) for key in selected[:count]
        ]

    def would_retain_sample(
        self, sample_id: object, *, sample_step: int | None = None
    ) -> bool:
        key = self._sample_key(sample_id)
        if key in self.shared_samples:
            return False
        if self.recent_sample_capacity > 0:
            return key in self._hybrid_keep_keys(
                candidate_key=key,
                candidate_step=0 if sample_step is None else int(sample_step),
            )
        if len(self.shared_samples) < self.sample_capacity:
            return True
        worst = max(
            self.shared_samples,
            key=lambda value: (self.sample_priority(value), repr(value)),
        )
        return (self.sample_priority(key), repr(key)) < (
            self.sample_priority(worst),
            repr(worst),
        )

    def _remove_head_sample(self, sample_id: object) -> None:
        for index, stored_id in enumerate(self.head_replay_ids):
            if stored_id == sample_id:
                self.head_replay_ids.pop(index)
                self.head_replay_inputs.pop(index)
                self.head_replay_targets.pop(index)
                self.head_replay_steps.pop(index)
                return

    def remember_shared_sample(
        self,
        sample_id: object,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
        target_gain: float,
        *,
        sample_step: int | None = None,
    ) -> bool:
        """Install one sample in the configured bounded reservoir."""

        key = self._sample_key(sample_id)
        step = 0 if sample_step is None else int(sample_step)
        if not self.would_retain_sample(key, sample_step=step):
            return False
        if self.recent_sample_capacity > 0:
            keep = self._hybrid_keep_keys(
                candidate_key=key, candidate_step=step
            )
            evicted = [
                stored for stored in self.shared_samples if stored not in keep
            ]
            for stored in evicted:
                self.shared_samples.pop(stored)
                self.shared_sample_steps.pop(stored, None)
                self._remove_head_sample(stored)
        elif len(self.shared_samples) >= self.sample_capacity:
            worst = max(
                self.shared_samples,
                key=lambda value: (self.sample_priority(value), repr(value)),
            )
            self.shared_samples.pop(worst)
            self.shared_sample_steps.pop(worst, None)
            self._remove_head_sample(worst)
        receiver = receiver_embedding.detach().to(
            device="cpu", dtype=torch.float32
        ).clone()
        provider = provider_embedding.detach().to(
            device="cpu", dtype=torch.float32
        ).clone()
        target = float(target_gain)
        self.shared_samples[key] = (receiver, provider, target)
        self.shared_sample_steps[key] = step
        self.remember_head_example(
            receiver,
            provider,
            target,
            sample_id=key,
            sample_step=step,
        )
        return True

    def remember_head_example(
        self,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
        target_gain: float,
        *,
        sample_id: object | None = None,
        sample_step: int = 0,
    ) -> None:
        pair = self.policy.pair_features(
            receiver_embedding.detach(), provider_embedding.detach()
        ).to(device="cpu", dtype=torch.float32)
        self.head_replay_inputs.append(pair.clone())
        self.head_replay_targets.append(float(target_gain))
        self.head_replay_ids.append(sample_id)
        self.head_replay_steps.append(int(sample_step))

    def _stratified_batch_indices(self, size: int) -> list[int]:
        """Sample evenly across early, intermediate, and mature events."""

        count = len(self.head_replay_inputs)
        take = min(count, max(1, int(size)))
        if take >= count:
            indices = list(range(count))
            self._replay_rng.shuffle(indices)
            return indices
        ordered = sorted(
            range(count),
            key=lambda index: (self.head_replay_steps[index], index),
        )
        strata = [
            list(map(int, values))
            for values in np.array_split(np.asarray(ordered), 3)
            if len(values) > 0
        ]
        selected: list[int] = []
        while len(selected) < take:
            progress = False
            for stratum in strata:
                remaining = [idx for idx in stratum if idx not in selected]
                if not remaining:
                    continue
                selected.append(
                    remaining[self._replay_rng.randrange(len(remaining))]
                )
                progress = True
                if len(selected) >= take:
                    break
            if not progress:
                break
        self._replay_rng.shuffle(selected)
        return selected

    def _reward_balanced_batch_indices(self, size: int) -> list[int]:
        """Sample both useful and non-useful pulls when both are available."""

        count = len(self.head_replay_inputs)
        take = min(count, max(1, int(size)))
        positive = [
            index
            for index, value in enumerate(self.head_replay_targets)
            if float(value) > 0.0
        ]
        nonpositive = [
            index
            for index, value in enumerate(self.head_replay_targets)
            if float(value) <= 0.0
        ]
        if not positive or not nonpositive:
            return self._stratified_batch_indices(take)
        self._replay_rng.shuffle(positive)
        self._replay_rng.shuffle(nonpositive)
        positive_take = min(len(positive), max(1, take // 2))
        negative_take = min(len(nonpositive), take - positive_take)
        selected = positive[:positive_take] + nonpositive[:negative_take]
        if len(selected) < take:
            selected_set = set(selected)
            remainder = [
                index for index in range(count) if index not in selected_set
            ]
            self._replay_rng.shuffle(remainder)
            selected.extend(remainder[: take - len(selected)])
        self._replay_rng.shuffle(selected)
        return selected

    def _head_training_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        *,
        target_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Combine gain calibration with provider ordering for like receivers."""

        pointwise = F.smooth_l1_loss(predictions, targets)
        zero = predictions.new_zeros(())
        if self.ranking_loss_weight <= 0.0 or int(predictions.numel()) < 2:
            return pointwise, pointwise, zero, 0

        receiver = features[:, : int(self.policy.embedding_dim)]
        receiver = F.normalize(receiver, p=2, dim=-1, eps=1.0e-8)
        cosine = receiver @ receiver.transpose(0, 1)
        upper = torch.triu(
            torch.ones_like(cosine, dtype=torch.bool), diagonal=1
        )
        target_difference = targets[:, None] - targets[None, :]
        margin = float(self.ranking_margin_db) / max(
            float(target_scale), 1.0e-8
        )
        comparable = (
            upper
            & (cosine >= float(self.ranking_receiver_cosine_min))
            & (torch.abs(target_difference) >= margin)
        )
        pair_count = int(comparable.sum().item())
        if pair_count == 0:
            return pointwise, pointwise, zero, 0

        score_difference = predictions[:, None] - predictions[None, :]
        direction = torch.sign(target_difference[comparable])
        temperature = float(self.ranking_temperature_db) / max(
            float(target_scale), 1.0e-8
        )
        ranking = F.softplus(
            -direction * score_difference[comparable] / temperature
        ).mean()
        combined = pointwise + float(self.ranking_loss_weight) * ranking
        return combined, pointwise, ranking, pair_count

    def train_head_epoch(self, batch_size: int = 64) -> tuple[int, float]:
        if not self.head_replay_inputs:
            return 0, 0.0
        if not any(parameter.requires_grad for parameter in self.policy.gain_head.parameters()):
            return 0, 0.0
        order = list(range(len(self.head_replay_inputs)))
        self._replay_rng.shuffle(order)
        losses: list[float] = []
        target_mean, target_scale = self.reward_normalization()
        self.policy.train()
        for start in range(0, len(order), max(1, int(batch_size))):
            indices = order[start : start + max(1, int(batch_size))]
            features = torch.stack(
                [self.head_replay_inputs[index] for index in indices]
            ).to(self.device)
            targets = torch.tensor(
                [
                    (
                        self.head_replay_targets[index] - target_mean
                    )
                    / target_scale
                    for index in indices
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self.opt.zero_grad(set_to_none=True)
            predictions = self.policy.gain_head(features).squeeze(-1)
            loss, _pointwise, _ranking, _pairs = self._head_training_loss(
                predictions,
                targets,
                features,
                target_scale=float(target_scale),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.gain_head.parameters(), 5.0)
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
            self.head_epoch_steps += 1
        return len(losses), float(sum(losses) / len(losses))

    def train_head_batches(
        self, *, num_batches: int, batch_size: int = 64
    ) -> tuple[int, float]:
        batches = max(0, int(num_batches))
        if batches == 0 or not self.head_replay_inputs:
            return 0, 0.0
        if not any(parameter.requires_grad for parameter in self.policy.gain_head.parameters()):
            return 0, 0.0
        count = len(self.head_replay_inputs)
        size = min(count, max(1, int(batch_size)))
        losses: list[float] = []
        target_mean, target_scale = self.reward_normalization()
        self.policy.train()
        balanced_batches = int(
            round(batches * float(self.reward_balanced_replay_fraction))
        )
        for batch_index in range(batches):
            indices = (
                self._reward_balanced_batch_indices(size)
                if batch_index < balanced_batches
                else self._stratified_batch_indices(size)
            )
            features = torch.stack(
                [self.head_replay_inputs[index] for index in indices]
            ).to(self.device)
            targets = torch.tensor(
                [
                    (
                        self.head_replay_targets[index] - target_mean
                    )
                    / target_scale
                    for index in indices
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self.opt.zero_grad(set_to_none=True)
            predictions = self.policy.gain_head(features).squeeze(-1)
            loss, _pointwise, _ranking, _pairs = self._head_training_loss(
                predictions,
                targets,
                features,
                target_scale=float(target_scale),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.gain_head.parameters(), 5.0)
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
            self.head_epoch_steps += 1
        return len(losses), float(sum(losses) / len(losses))

    def policy_embedding(self, state: ExactPrivateState) -> torch.Tensor:
        self.policy.eval()
        with torch.no_grad():
            return self.policy.encode(state).detach().cpu()

    def policy_gain_from_embeddings(
        self, receiver: torch.Tensor, provider: torch.Tensor
    ) -> float:
        self.policy.eval()
        with torch.no_grad():
            value = self.policy.score_embeddings(
                receiver.to(self.device).unsqueeze(0),
                provider.to(self.device).unsqueeze(0),
            )
        return self.denormalize_gain(float(value.item()))

    def policy_trigger_threshold(self, quantile: float) -> float:
        """Return a deployable threshold learned only from observed gains."""

        if not self.head_replay_targets:
            return 0.0
        q = min(1.0, max(0.0, float(quantile)))
        return float(np.quantile(self.head_replay_targets, q))


@dataclass(frozen=True)
class _TrainingExample:
    sample_id: bytes
    provider_idx: int
    receiver_state: ExactPrivateState
    receiver_embedding: torch.Tensor
    provider_state: ExactPrivateState
    provider_embedding: torch.Tensor
    target_gain: float
    propensity: float


def decentralized_reservation_order(
    receivers: Iterable[int], *, step: int, seed: int
) -> list[int]:
    """Deterministic local-backoff order reproducible at every vehicle."""

    def priority(node_idx: int) -> int:
        value = (
            int(node_idx)
            ^ (int(step) * 0x9E3779B9)
            ^ (int(seed) * 0x85EBCA6B)
        ) & 0xFFFFFFFF
        value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        return int((value ^ (value >> 16)) & 0xFFFFFFFF)

    return sorted(
        (int(node_idx) for node_idx in receivers),
        key=lambda node_idx: (priority(node_idx), int(node_idx)),
    )


def available_reservation_candidates(
    receiver_idx: int,
    candidate_ids: Iterable[int],
    reserved_nodes: set[int],
) -> set[int]:
    if int(receiver_idx) in reserved_nodes:
        return set()
    return {
        int(provider)
        for provider in candidate_ids
        if int(provider) not in reserved_nodes
    }


@dataclass
class _RealisticDecisionContext:
    mode: str
    receiver_idx: int
    agent: OnlineExactUtilityAgent
    candidate_ids: list[int]
    zones: dict[int, int]
    enc_ids: dict[int, int]
    current_scores: dict[int, float]
    decision_diag: dict[str, int | float | str | bool]
    receiver_state: ExactPrivateState | None
    receiver_embedding: torch.Tensor | None
    proposal: TransferProposal | None = None
    provider_idx: int | None = None
    propensity: float = 0.0
    exploratory: bool = False
    token_stream: str = ""


@dataclass(frozen=True)
class _VisitPullState:
    """Token-compatible view of one vehicle's current AZ visit budget."""

    window_index: int
    window_offset: int
    phase: int
    random_offset: int
    available: bool
    random_due: bool
    random_ready: bool


class ExactSequentialBidirectionalSimulation(
    EncodedSequentialBidirectionalSimulation
):
    """Exact learned state encoder with fixed-budget sequential pulls."""

    policy_transfer_rule = "all-feasible-exact-single-policy-average"

    def __init__(
        self,
        *args,
        exact_hidden_dim: int = 8,
        embedding_dim: int = 32,
        gain_hidden_dim: int | None = None,
        pair_feature_mode: str = "concat",
        pull_budget: float = 1.0,
        token_window_steps: int = 10,
        contact_aware_window_timing: bool = False,
        selection_mode: str = "policy",
        exploration_probability: float = 0.02,
        policy_warmup_steps: int = 0,
        policy_warmup_pull_probability: float = 1.0,
        learned_time_dim: int = 16,
        learned_time_scale: float | None = None,
        policy_sample_capacity: int = 512,
        policy_sample_bundle_capacity: int = 32,
        encoder_lr_scale: float = 0.1,
        align_policy_encoders: bool = False,
        freeze_policy_encoders: bool = False,
        pretrained_policy_path: str | Path | None = None,
        freeze_pretrained_policy: bool = False,
        normalize_policy_rewards: bool = False,
        policy_reward_scale_db: float | None = None,
        policy_reward_scope: str = "directional",
        policy_training_target: str = "validation-gain",
        policy_ranking_loss_weight: float = 0.0,
        policy_ranking_margin_db: float = 0.25,
        policy_ranking_temperature_db: float = 1.0,
        policy_ranking_receiver_cosine_min: float = 0.8,
        policy_min_samples: int = 0,
        policy_exploration_start: float | None = None,
        policy_exploration_decay_samples: int = 0,
        visit_pull_budget: int = 0,
        policy_trigger_quantile: float = 0.75,
        policy_fixed_trigger_db: float | None = None,
        allow_unused_policy_tokens: bool = False,
        trajectory_capacity: int = 256,
        symmetric_pulls: bool = False,
        unconditional_evidence_union: bool = False,
        mergeable_max_delta_rows: int = 0,
        policy_reward_metric: str = "normalized-improvement",
        policy_support_dim: int = 0,
        inverse_propensity_weighting: bool = False,
        train_all_current_examples: bool = True,
        train_accumulated_head_epoch: bool = False,
        head_replay_batches_per_step: int = 0,
        diagnostic_regular_count: int = 0,
        aux_only: bool = False,
        realistic_network: bool = False,
        network_candidate_top_k: int = 0,
        network_resource_count: int = 4,
        network_bandwidth_hz: float = 10.0e6,
        network_direction_airtime_s: float = 0.125,
        network_efficiency: float = 0.6,
        network_max_spectral_efficiency: float = 6.0,
        network_min_sinr_db: float = 5.0,
        network_missing_power_dbm: float = -120.0,
        network_decentralized_reservation: bool = False,
        network_reservation_control_bytes: int = 32,
        **kwargs,
    ) -> None:
        if int(exact_hidden_dim) <= 0:
            raise ValueError("exact_hidden_dim must be positive")
        if not 0.0 < float(exploration_probability) < 1.0:
            raise ValueError("exploration_probability must be in (0, 1)")
        self.exact_hidden_dim = int(exact_hidden_dim)
        self.gain_hidden_dim = (
            self.exact_hidden_dim
            if gain_hidden_dim is None
            else int(gain_hidden_dim)
        )
        if self.gain_hidden_dim <= 0:
            raise ValueError("gain_hidden_dim must be positive")
        self.pair_feature_mode = str(pair_feature_mode).strip().lower()
        budget = float(pull_budget)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("pull_budget must be finite and positive")
        if budget >= 1.0 and not budget.is_integer():
            raise ValueError("pull_budget must be an integer when it is at least one")
        self.pull_budget = budget
        mode = str(selection_mode).strip().lower()
        if mode not in {"policy", "random", "oracle", "novelty", "isolated"}:
            raise ValueError(
                "selection_mode must be policy, random, oracle, novelty, or isolated"
            )
        self.selection_mode = mode
        if int(token_window_steps) <= 0:
            raise ValueError("token_window_steps must be positive")
        self.token_window_steps = int(token_window_steps)
        self.contact_aware_window_timing = bool(
            contact_aware_window_timing
        )
        self.policy_warmup_steps = max(0, int(policy_warmup_steps))
        self.policy_warmup_pull_probability = float(
            policy_warmup_pull_probability
        )
        if not 0.0 <= self.policy_warmup_pull_probability <= 1.0:
            raise ValueError(
                "policy_warmup_pull_probability must be in [0, 1]"
            )
        self.exploration_probability = float(exploration_probability)
        self.inverse_propensity_weighting = bool(
            inverse_propensity_weighting
        )
        self.train_all_current_examples = bool(train_all_current_examples)
        self.train_accumulated_head_epoch = bool(
            train_accumulated_head_epoch
        )
        self.head_replay_batches_per_step = max(
            0, int(head_replay_batches_per_step)
        )
        self.learned_time_dim = int(learned_time_dim)
        self.learned_time_scale = (
            None if learned_time_scale is None else float(learned_time_scale)
        )
        if self.learned_time_scale is not None and self.learned_time_scale <= 0.0:
            raise ValueError("learned_time_scale must be positive")
        self.policy_sample_capacity = int(policy_sample_capacity)
        self.policy_sample_bundle_capacity = int(policy_sample_bundle_capacity)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self.align_policy_encoders = bool(align_policy_encoders) and mode == "policy"
        self.freeze_policy_encoders = (
            bool(freeze_policy_encoders) and mode == "policy"
        )
        self.pretrained_policy_path = (
            None if pretrained_policy_path in {None, ""}
            else Path(pretrained_policy_path).expanduser().resolve()
        )
        self.freeze_pretrained_policy = bool(freeze_pretrained_policy)
        if self.pretrained_policy_path is not None and mode != "policy":
            raise ValueError("pretrained policies are only valid for policy selection")
        if self.freeze_pretrained_policy and self.pretrained_policy_path is None:
            raise ValueError("freeze_pretrained_policy requires a pretrained policy")
        self.normalize_policy_rewards = (
            bool(normalize_policy_rewards) and mode == "policy"
        )
        self.policy_reward_scale_db = (
            None
            if policy_reward_scale_db is None or mode != "policy"
            else float(policy_reward_scale_db)
        )
        self.policy_reward_scope = str(policy_reward_scope).strip().lower()
        self.policy_training_target = str(
            policy_training_target
        ).strip().lower()
        if self.policy_training_target not in {
            "validation-gain",
            "information-gain",
            "parameter-geometry",
        }:
            raise ValueError("invalid policy_training_target")
        self.policy_ranking_loss_weight = float(policy_ranking_loss_weight)
        self.policy_ranking_margin_db = float(policy_ranking_margin_db)
        self.policy_ranking_temperature_db = float(
            policy_ranking_temperature_db
        )
        self.policy_ranking_receiver_cosine_min = float(
            policy_ranking_receiver_cosine_min
        )
        self.policy_min_samples = max(0, int(policy_min_samples))
        self.policy_exploration_start = (
            self.exploration_probability
            if policy_exploration_start is None
            else float(policy_exploration_start)
        )
        self.policy_exploration_decay_samples = max(
            0, int(policy_exploration_decay_samples)
        )
        self.visit_pull_budget = max(0, int(visit_pull_budget))
        self.policy_trigger_quantile = float(policy_trigger_quantile)
        self.policy_fixed_trigger_db = (
            None
            if policy_fixed_trigger_db is None
            else float(policy_fixed_trigger_db)
        )
        if self.policy_fixed_trigger_db is not None and not math.isfinite(
            self.policy_fixed_trigger_db
        ):
            raise ValueError("policy_fixed_trigger_db must be finite")
        self.allow_unused_policy_tokens = bool(allow_unused_policy_tokens)
        self.trajectory_capacity = int(trajectory_capacity)
        self.symmetric_pulls = bool(symmetric_pulls)
        self.policy_reward_metric = str(policy_reward_metric)
        self.unconditional_evidence_union = bool(
            unconditional_evidence_union
        )
        self.mergeable_max_delta_rows = max(
            0, int(mergeable_max_delta_rows)
        )
        self.policy_support_dim = max(0, int(policy_support_dim))
        if self.policy_sample_capacity <= 0:
            raise ValueError("policy_sample_capacity must be positive")
        if self.policy_sample_bundle_capacity <= 0:
            raise ValueError("policy_sample_bundle_capacity must be positive")
        if not 0.0 < self.encoder_lr_scale <= 1.0:
            raise ValueError("encoder_lr_scale must be in (0, 1]")
        if not self.exploration_probability <= self.policy_exploration_start < 1.0:
            raise ValueError(
                "policy_exploration_start must be at least exploration_probability "
                "and less than one"
            )
        if (
            self.policy_reward_scale_db is not None
            and (
                not math.isfinite(self.policy_reward_scale_db)
                or self.policy_reward_scale_db <= 0.0
            )
        ):
            raise ValueError(
                "policy_reward_scale_db must be finite and positive"
            )
        if self.policy_reward_scope not in {"directional", "joint"}:
            raise ValueError(
                "policy_reward_scope must be directional or joint"
            )
        if (
            not math.isfinite(self.policy_ranking_loss_weight)
            or self.policy_ranking_loss_weight < 0.0
        ):
            raise ValueError(
                "policy_ranking_loss_weight must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.policy_ranking_margin_db)
            or self.policy_ranking_margin_db < 0.0
        ):
            raise ValueError(
                "policy_ranking_margin_db must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.policy_ranking_temperature_db)
            or self.policy_ranking_temperature_db <= 0.0
        ):
            raise ValueError(
                "policy_ranking_temperature_db must be finite and positive"
            )
        if not -1.0 <= self.policy_ranking_receiver_cosine_min <= 1.0:
            raise ValueError(
                "policy_ranking_receiver_cosine_min must be in [-1, 1]"
            )
        if not 0.0 <= self.policy_trigger_quantile <= 1.0:
            raise ValueError("policy_trigger_quantile must be in [0, 1]")
        if self.visit_pull_budget > 0 and self.contact_aware_window_timing:
            raise ValueError(
                "visit pull budgets replace token-window timing options"
            )
        if self.freeze_policy_encoders and self.align_policy_encoders:
            raise ValueError(
                "frozen policy encoders must not be exchanged or averaged"
            )
        if self.trajectory_capacity <= 0:
            raise ValueError("trajectory_capacity must be positive")
        self.diagnostic_regular_count = max(
            0, int(diagnostic_regular_count)
        )
        self.aux_only = bool(aux_only)
        self.realistic_network = bool(realistic_network)
        self.network_candidate_top_k = int(network_candidate_top_k)
        self.network_resource_count = int(network_resource_count)
        self.network_bandwidth_hz = float(network_bandwidth_hz)
        self.network_direction_airtime_s = float(network_direction_airtime_s)
        self.network_efficiency = float(network_efficiency)
        self.network_max_spectral_efficiency = float(
            network_max_spectral_efficiency
        )
        self.network_min_sinr_db = float(network_min_sinr_db)
        self.network_missing_power_dbm = float(network_missing_power_dbm)
        self.network_decentralized_reservation = bool(
            network_decentralized_reservation
        )
        self.network_reservation_control_bytes = int(
            network_reservation_control_bytes
        )
        if self.network_reservation_control_bytes < 0:
            raise ValueError("network reservation bytes cannot be negative")
        if self.realistic_network:
            if self.pull_budget != 1.0:
                raise ValueError("realistic network currently requires pull_budget=1")
            if self.network_candidate_top_k < 0:
                raise ValueError("network_candidate_top_k must be nonnegative")
            if self.network_resource_count <= 0:
                raise ValueError("network_resource_count must be positive")
        self._trajectory_histories: list[ExactTrajectoryHistory] = []
        super().__init__(
            *args,
            symmetric_predictor_pull=self.symmetric_pulls,
            pull_reward_metric=self.policy_reward_metric,
            state_sketch_dim=1,
            embedding_dim=int(embedding_dim),
            # Compatibility values required by the encoded parent.  This
            # subclass overrides policy sharing and never creates a serving
            # policy or uses these snapshot settings.
            serving_interval=1,
            serving_tau=1.0,
            **kwargs,
        )
        self._trajectory_histories = [
            ExactTrajectoryHistory(capacity=self.trajectory_capacity)
            for _ in self.nodes
        ]
        self.token_windows = VehicleTokenWindows(
            window_steps=self.token_window_steps,
            seed=int(getattr(self.cfg, "seed", 0)),
            capacity=(
                int(self.pull_budget) if self.pull_budget >= 1.0 else 1
            ),
        )
        self._active_visit_keys: dict[int, tuple[int, int, int]] = {}
        self._visit_last_steps: dict[int, int] = {}
        self._visit_serials: dict[int, int] = defaultdict(int)
        self._visit_remaining: dict[tuple[str, int], int] = {}
        self._visit_random_targets: dict[tuple[str, int], list[int]] = {}
        self._zone_visit_counts: list[dict[int, int]] = [
            defaultdict(int) for _ in self.nodes
        ]
        self._trace_visit_random_targets = (
            self._precompute_trace_visit_random_targets()
        )
        self._trace_contact_available = (
            self._build_trace_contact_availability()
            if self.contact_aware_window_timing
            else None
        )
        self._token_decision_rows: list[dict[str, int | float | str | bool]] = []
        self._live_diagnostic_rows: list[dict[str, int | float | str]] = []
        self.zramp_policy_mode = (
            "bidirectional-private-cv-exact-single-policy-sequential"
        )
        self._encoded_log_path = (
            Path(self.cfg.results_dir) / "exact_policy_training.csv"
        )
        self._communication_assumptions = self._build_communication_assumptions()

    def _precompute_trace_visit_random_targets(
        self,
    ) -> dict[tuple[int, int, int, int], tuple[int, ...]]:
        """Uniformly select contact frames using only the replay benchmark."""

        if self.visit_pull_budget <= 0:
            return {}
        replay = getattr(self, "_trace_replay", None)
        if not isinstance(replay, dict):
            return {}
        active = replay.get("node_active")
        states = replay.get("node_states")
        generations = replay.get("node_generations")
        measurements = replay.get("measurements")
        if not (
            isinstance(active, np.ndarray)
            and isinstance(states, np.ndarray)
            and isinstance(generations, np.ndarray)
            and isinstance(measurements, dict)
        ):
            return {}
        contact_nodes: dict[int, set[int]] = {}
        for raw_step, rows in measurements.items():
            links = self._contact_links_from_measurements(
                [
                    (
                        int(row[1]),
                        int(row[2]),
                        int(row[3]),
                        float(row[4]),
                    )
                    for row in np.asarray(rows)
                ]
            )
            nodes: set[int] = set()
            for _zone, first, second in links:
                nodes.add(int(first))
                nodes.add(int(second))
            contact_nodes[int(raw_step)] = nodes

        visits: dict[tuple[int, int, int, int], list[int]] = {}
        open_keys: dict[int, tuple[int, int, int, int]] = {}
        last_steps: dict[int, int] = {}
        for step in range(int(active.shape[0])):
            for raw_node in np.flatnonzero(active[step]):
                node = int(raw_node)
                generation = int(generations[step, node])
                zone = int(states[step, node, 2])
                previous = open_keys.get(node)
                if (
                    previous is None
                    or previous[1] != generation
                    or previous[2] != zone
                    or last_steps.get(node, -2) != step - 1
                ):
                    previous = (node, generation, zone, step)
                    open_keys[node] = previous
                    visits[previous] = []
                if node in contact_nodes.get(step, set()):
                    visits[previous].append(step)
                last_steps[node] = step

        selected: dict[tuple[int, int, int, int], tuple[int, ...]] = {}
        for key, contact_steps in visits.items():
            if not contact_steps:
                selected[key] = ()
                continue
            payload = (
                f"{int(self.cfg.seed)}|visit-random|"
                f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"
            ).encode("utf-8")
            seed = int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(),
                byteorder="big",
                signed=False,
            )
            rng = random.Random(seed)
            count = min(self.visit_pull_budget, len(contact_steps))
            selected[key] = tuple(
                sorted(rng.sample(contact_steps, count))
            )
        return selected

    def _sync_visit_budgets(
        self, *, step: int, zone_nodes: dict[int, list[int]]
    ) -> None:
        if self.visit_pull_budget <= 0:
            return
        replay = getattr(self, "_trace_replay", None)
        trace_generations = (
            replay.get("node_generations")
            if isinstance(replay, dict)
            else None
        )
        active = {
            int(node): int(zone)
            for zone, nodes in zone_nodes.items()
            for node in nodes
        }
        for node, zone in active.items():
            if isinstance(trace_generations, np.ndarray):
                generation = int(trace_generations[int(step), node])
            else:
                generations = getattr(self, "_node_generations", [])
                generation = (
                    int(generations[node]) if node < len(generations) else 0
                )
            previous = self._active_visit_keys.get(node)
            is_new = bool(
                previous is None
                or previous[0] != generation
                or previous[1] != zone
                or self._visit_last_steps.get(node, -2) != int(step) - 1
            )
            if is_new:
                key = (generation, zone, int(step))
                self._active_visit_keys[node] = key
                self._visit_serials[node] += 1
                self._zone_visit_counts[node][zone] += 1
                trace_key = (node, generation, zone, int(step))
                targets = list(
                    self._trace_visit_random_targets.get(
                        trace_key, (int(step),)
                    )
                )
                for mode in self.agents:
                    self._visit_remaining[(str(mode), node)] = int(
                        self.visit_pull_budget
                    )
                    self._visit_random_targets[(str(mode), node)] = list(
                        targets
                    )
            self._visit_last_steps[node] = int(step)

    def _pull_state(
        self, *, step: int, node_idx: int, mode: str, stream: str
    ):
        if self.visit_pull_budget <= 0:
            return self.token_windows.state(
                step=int(step), node_idx=int(node_idx), stream=str(stream)
            )
        key = self._active_visit_keys.get(int(node_idx), (0, -1, int(step)))
        targets = self._visit_random_targets.get(
            (str(mode), int(node_idx)), []
        )
        next_target = targets[0] if targets else None
        due = bool(next_target is not None and int(step) >= int(next_target))
        return _VisitPullState(
            window_index=int(self._visit_serials.get(int(node_idx), 0)),
            window_offset=max(0, int(step) - int(key[2])),
            phase=0,
            random_offset=(
                -1 if next_target is None else int(next_target) - int(key[2])
            ),
            available=bool(
                self._visit_remaining.get((str(mode), int(node_idx)), 0) > 0
            ),
            random_due=due,
            random_ready=due,
        )

    def _pull_budget_available(self, *, mode: str, node_idx: int) -> bool:
        if self.visit_pull_budget <= 0:
            return True
        return bool(
            self._visit_remaining.get((str(mode), int(node_idx)), 0) > 0
        )

    def _spend_pull_budget(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int | None,
        stream: str,
    ) -> None:
        if self.visit_pull_budget <= 0:
            self.token_windows.spend(
                step=int(step),
                node_idx=int(receiver_idx),
                stream=str(stream),
            )
            return
        endpoints = [int(receiver_idx)]
        if self.symmetric_pulls and provider_idx is not None:
            endpoints.append(int(provider_idx))
        for node in set(endpoints):
            key = (str(mode), node)
            self._visit_remaining[key] = max(
                0, int(self._visit_remaining.get(key, 0)) - 1
            )
            targets = self._visit_random_targets.get(key, [])
            if targets:
                targets.pop(0)

    def _random_window_ready(
        self, token, *, step: int, node_idx: int
    ) -> bool:
        """Allow retrying at feasible contacts after the random deadline."""

        if self.contact_aware_window_timing:
            return bool(
                token.random_ready
                or self._window_deadline(
                    token, step=int(step), node_idx=int(node_idx)
                )
            )
        return bool(token.random_due)

    def _build_trace_contact_availability(self) -> np.ndarray | None:
        """Precompute only whether each vehicle has a feasible V2V contact."""

        replay = getattr(self, "_trace_replay", None)
        if not isinstance(replay, dict):
            return None
        active = replay.get("node_active")
        by_step = replay.get("measurements")
        if not isinstance(active, np.ndarray) or not isinstance(by_step, dict):
            return None
        available = np.zeros(active.shape, dtype=np.bool_)
        for raw_step, rows in by_step.items():
            step = int(raw_step)
            if step < 0 or step >= int(available.shape[0]) or len(rows) == 0:
                continue
            measurements = [
                (
                    int(round(float(row[1]))),
                    int(round(float(row[2]))),
                    int(round(float(row[3]))),
                    float(row[4]),
                )
                for row in rows
            ]
            for _zone, left, right in self._contact_links_from_measurements(
                measurements
            ):
                available[step, int(left)] = True
                available[step, int(right)] = True
        return available

    def _window_deadline(
        self, token, *, step: int, node_idx: int
    ) -> bool:
        """Return true at the window end or vehicle's final active step."""

        if not self.contact_aware_window_timing or not token.available:
            return False
        if int(token.window_offset) == self.token_window_steps - 1:
            return True
        available = getattr(self, "_trace_contact_available", None)
        replay = getattr(self, "_trace_replay", None)
        if not isinstance(available, np.ndarray) or not isinstance(replay, dict):
            return False
        generations = replay.get("node_generations")
        last_step = min(
            int(available.shape[0]) - 1,
            int(step) + self.token_window_steps - 1 - int(token.window_offset),
        )
        current_generation = (
            int(generations[int(step), int(node_idx)])
            if isinstance(generations, np.ndarray)
            else None
        )
        for future_step in range(int(step) + 1, last_step + 1):
            if (
                current_generation is not None
                and int(generations[future_step, int(node_idx)])
                != current_generation
            ):
                break
            if bool(available[future_step, int(node_idx)]):
                return False
        return True

    def _exact_greedy_direction(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        receiver_state: TensorState,
        provider_state: TensorState,
    ) -> tuple[bool, bool, float]:
        """Apply the policy method's private-CV aggregation to one greedy pull."""

        receiver_validation = self._zone_validation[int(receiver_idx)]
        provider_validation = self._zone_validation[int(provider_idx)]
        qa_opt = self._validation_subset_weight(
            receiver_validation.optimization
        )
        qb_opt = self._validation_subset_weight(
            provider_validation.optimization
        )
        qa_reward = self._validation_subset_weight(
            receiver_validation.reward
        )
        qb_reward = self._validation_subset_weight(
            provider_validation.reward
        )
        if qa_opt + qb_opt <= 0.0 or qa_reward + qb_reward <= 0.0:
            return False, False, float("nan")

        optimization_pair = self._prepare_validation_pair(
            receiver_validation.optimization,
            provider_validation.optimization,
            quality_a=qa_opt,
            quality_b=qb_opt,
        )
        reward_pair = self._prepare_validation_pair(
            receiver_validation.reward,
            provider_validation.reward,
            quality_a=qa_reward,
            quality_b=qb_reward,
        )

        def objective(alpha: float) -> float:
            aggregate = interpolate_states(
                receiver_state, provider_state, alpha
            )
            loss_a, loss_b = self._pair_mses(
                aggregate, optimization_pair
            )
            combined = quality_weighted_loss(
                loss_a,
                qa_opt,
                loss_b,
                qb_opt,
                epsilon=self.validation_epsilon,
            )
            if combined is None:
                raise RuntimeError(
                    "greedy optimization validation quality vanished"
                )
            return combined

        optimum = minimize_bounded(
            objective,
            tolerance=self.aggregation_tolerance,
            max_iterations=self.aggregation_max_iterations,
        )
        aggregate = interpolate_states(
            receiver_state, provider_state, optimum.alpha
        )
        before_a, before_b = self._pair_mses(
            receiver_state, reward_pair
        )
        after_a, after_b = self._pair_mses(aggregate, reward_pair)
        before = quality_weighted_loss(
            before_a,
            qa_reward,
            before_b,
            qb_reward,
            epsilon=self.validation_epsilon,
        )
        after = quality_weighted_loss(
            after_a,
            qa_reward,
            after_b,
            qb_reward,
            epsilon=self.validation_epsilon,
        )
        if before is None or after is None:
            raise RuntimeError("greedy reward validation quality vanished")
        adopted = bool(after < before)
        if adopted:
            model = self.greedy_models[int(receiver_idx)]
            self._load_model_state(model, aggregate)
            self.greedy_opts[int(receiver_idx)] = torch.optim.Adam(
                model.parameters(), lr=self.cfg.local_lr
            )
        return True, adopted, float(optimum.alpha)

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        """Share on every contact using the policy's directional CV weighting."""

        links: list[tuple[int, int, int]] = []
        if contact_links is None:
            for zone, node_indices in zone_nodes.items():
                ids = sorted(int(index) for index in node_indices)
                for left in range(len(ids)):
                    for right in range(left + 1, len(ids)):
                        links.append((int(zone), ids[left], ids[right]))
        else:
            seen: set[tuple[int, int, int]] = set()
            for zone, first, second in contact_links:
                first_idx = int(first)
                second_idx = int(second)
                if first_idx == second_idx:
                    continue
                if second_idx < first_idx:
                    first_idx, second_idx = second_idx, first_idx
                key = (int(zone), first_idx, second_idx)
                if key not in seen:
                    seen.add(key)
                    links.append(key)
        links.sort(key=lambda row: (row[0], row[1], row[2]))

        attempts = 0
        valid = 0
        adopted = 0
        alphas: list[float] = []
        for _zone, first_idx, second_idx in links:
            first_state = self._clone_state(
                self.greedy_models[first_idx]
            )
            second_state = self._clone_state(
                self.greedy_models[second_idx]
            )
            for receiver_idx, provider_idx, receiver_state, provider_state in (
                (
                    first_idx,
                    second_idx,
                    first_state,
                    second_state,
                ),
                (
                    second_idx,
                    first_idx,
                    second_state,
                    first_state,
                ),
            ):
                pull_valid, pull_adopted, alpha = (
                    self._exact_greedy_direction(
                        receiver_idx=receiver_idx,
                        provider_idx=provider_idx,
                        receiver_state=receiver_state,
                        provider_state=provider_state,
                    )
                )
                attempts += 1
                valid += int(pull_valid)
                adopted += int(pull_adopted)
                if math.isfinite(alpha):
                    alphas.append(float(alpha))

        self._last_exact_greedy_valid = int(valid)
        self._last_exact_greedy_adopted = int(adopted)
        self._last_exact_greedy_mean_alpha = (
            float(np.mean(alphas)) if alphas else float("nan")
        )
        return int(attempts)

    def _communication_overhead_row(self, **kwargs):
        row = super()._communication_overhead_row(**kwargs)
        if "greedy" in self.aux_baselines:
            row.update(
                {
                    "greedy_cv_valid_pulls": int(
                        getattr(self, "_last_exact_greedy_valid", 0)
                    ),
                    "greedy_cv_adopted_pulls": int(
                        getattr(self, "_last_exact_greedy_adopted", 0)
                    ),
                    "greedy_cv_mean_self_weight": float(
                        getattr(
                            self,
                            "_last_exact_greedy_mean_alpha",
                            float("nan"),
                        )
                    ),
                }
            )
        return row

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        """Reset every vehicle-owned object before a replacement participates."""

        super()._reset_respawned_node(node_idx, generation=generation)
        node_idx = int(node_idx)
        if len(self._trajectory_histories) == len(self.nodes):
            self._trajectory_histories[node_idx] = ExactTrajectoryHistory(
                capacity=int(getattr(self, "trajectory_capacity", 256))
            )
        if len(self._zone_validation) == len(self.nodes):
            self._zone_validation[node_idx] = ZoneValidationState()

        template_groups = exact_model_groups(self.template_state)
        group_widths = tuple(int(group.shape[1]) for group in template_groups)
        # Predictor inputs + target + measurement experience + AZ-visit
        # experience + summary-row marker.
        trajectory_dim = self._predictor_input_dim() + 4
        for mode_id, agents in self.local_agents.items():
            mode_trajectory_dim = int(
                self.agents[mode_id].policy.trajectory_encoder.input_size
            )
            mode_offset = int(
                zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000
            )
            model_seed = int(self.cfg.seed) + 907_001 + mode_offset
            rng_seed = (
                int(self.cfg.seed)
                + 2_900_003
                + mode_offset
                + 104_729 * node_idx
            )
            replacement = OnlineExactUtilityAgent(
                group_widths=group_widths,
                trajectory_dim=mode_trajectory_dim,
                hidden_dim=self.exact_hidden_dim,
                embedding_dim=self.embedding_dim,
                gain_hidden_dim=self.gain_hidden_dim,
                pair_feature_mode=self.pair_feature_mode,
                device=self.device,
                learning_rate=self.cfg.rl_lr,
                rng_seed=rng_seed,
                model_seed=model_seed,
                sample_capacity=int(
                    getattr(self, "policy_sample_capacity", 512)
                ),
                encoder_lr_scale=float(
                    getattr(self, "encoder_lr_scale", 0.1)
                ),
                normalize_rewards=bool(
                    getattr(self, "normalize_policy_rewards", False)
                ),
                reward_scale_db=getattr(
                    self, "policy_reward_scale_db", None
                ),
                ranking_loss_weight=float(
                    getattr(self, "policy_ranking_loss_weight", 0.0)
                ),
                ranking_margin_db=float(
                    getattr(self, "policy_ranking_margin_db", 0.25)
                ),
                ranking_temperature_db=float(
                    getattr(self, "policy_ranking_temperature_db", 1.0)
                ),
                ranking_receiver_cosine_min=float(
                    getattr(
                        self,
                        "policy_ranking_receiver_cosine_min",
                        0.8,
                    )
                ),
                support_dim=int(getattr(self, "policy_support_dim", 0)),
            )
            replacement.policy.load_state_dict(
                {
                    name: value.detach().clone()
                    for name, value in self.agents[mode_id].policy.state_dict().items()
                }
            )
            if bool(getattr(self, "freeze_policy_encoders", False)):
                replacement.freeze_encoder()
            if bool(getattr(self, "freeze_pretrained_policy", False)):
                replacement.freeze_policy()
            agents[node_idx] = replacement
            self._local_policy_pending_transitions[mode_id][node_idx] = 0
            self._local_policy_versions[mode_id][node_idx] = 0
            self._local_policy_initial_rngs[mode_id][node_idx] = random.Random(
                rng_seed + 57_911
            )
            self._cv_receiver_aggregations.pop((str(mode_id), node_idx), None)
            for table in (
                self._last_predicted_gain,
                self._last_protocol_valid,
                self._last_exploratory,
            ):
                table.pop((str(mode_id), node_idx), None)

        self._cv_last_provider_pull_step = {
            key: value
            for key, value in self._cv_last_provider_pull_step.items()
            if int(key[1]) != node_idx and int(key[2]) != node_idx
        }
        if hasattr(self, "token_windows"):
            self.token_windows.reset_node(node_idx)
        if hasattr(self, "_active_visit_keys"):
            self._active_visit_keys.pop(node_idx, None)
            self._visit_last_steps.pop(node_idx, None)
            self._visit_serials.pop(node_idx, None)
            if node_idx < len(self._zone_visit_counts):
                self._zone_visit_counts[node_idx].clear()
            for mode_id in list(self.agents):
                self._visit_remaining.pop((str(mode_id), node_idx), None)
                self._visit_random_targets.pop(
                    (str(mode_id), node_idx), None
                )

    # ------------------------------------------------------- learned predictor

    def _make_predictor(self) -> nn.Module:
        cfg = self.cfg
        duration = float(getattr(cfg, "predictor_time_step_duration", 1.0))
        unit = float(getattr(cfg, "predictor_time_unit", 1.0))
        configured_scale = getattr(
            self, "learned_time_scale", None
        )
        if configured_scale is None:
            configured_scale = float(
                getattr(cfg, "predictor_learned_time_scale", 1000.0)
            )
            self.learned_time_scale = configured_scale
        scale = float(configured_scale)
        return make_rssi_predictor(
            str(cfg.rssi_model),
            input_dim=self._predictor_input_dim(),
            include_time=bool(getattr(cfg, "predictor_include_time", False)),
            time_encoding="learned",
            learned_time_dim=int(self.learned_time_dim),
            learned_time_hidden_dim=int(self.learned_time_dim),
            learned_time_scale=float(scale),
            spatial_grid_points=int(
                getattr(cfg, "local_support_spatial_grid_points", 9)
            ),
            support_prior_strength=float(
                getattr(cfg, "local_support_prior_strength", 0.0)
            ),
        )

    # ------------------------------------------------------------ policy init

    def _init_local_policy_agents(self) -> None:
        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()
        template_groups = exact_model_groups(self.template_state)
        group_widths = tuple(int(group.shape[1]) for group in template_groups)
        trajectory_dim = self._predictor_input_dim() + 4
        for mode_id in list(self.agents):
            mode_offset = int(
                zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000
            )
            model_seed = int(self.cfg.seed) + 907_001 + mode_offset
            template = OnlineExactUtilityAgent(
                group_widths=group_widths,
                trajectory_dim=trajectory_dim,
                hidden_dim=self.exact_hidden_dim,
                embedding_dim=self.embedding_dim,
                gain_hidden_dim=self.gain_hidden_dim,
                pair_feature_mode=self.pair_feature_mode,
                device=self.device,
                learning_rate=self.cfg.rl_lr,
                rng_seed=model_seed,
                model_seed=model_seed,
                sample_capacity=self.policy_sample_capacity,
                encoder_lr_scale=self.encoder_lr_scale,
                normalize_rewards=self.normalize_policy_rewards,
                reward_scale_db=self.policy_reward_scale_db,
                ranking_loss_weight=self.policy_ranking_loss_weight,
                ranking_margin_db=self.policy_ranking_margin_db,
                ranking_temperature_db=self.policy_ranking_temperature_db,
                ranking_receiver_cosine_min=(
                    self.policy_ranking_receiver_cosine_min
                ),
                support_dim=self.policy_support_dim,
            )
            if self.pretrained_policy_path is not None:
                checkpoint = torch.load(
                    self.pretrained_policy_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if checkpoint.get("format") != "cross_map_pretrained_exact_policy_v1":
                    raise ValueError(
                        f"unsupported pretrained policy format in {self.pretrained_policy_path}"
                    )
                architecture = checkpoint.get("architecture", {})
                expected = {
                    "group_widths": list(group_widths),
                    "trajectory_dim": int(trajectory_dim),
                    "hidden_dim": int(self.exact_hidden_dim),
                    "embedding_dim": int(self.embedding_dim),
                    "gain_hidden_dim": int(self.gain_hidden_dim),
                    "pair_feature_mode": str(self.pair_feature_mode),
                }
                actual = {key: architecture.get(key) for key in expected}
                if actual != expected:
                    raise ValueError(
                        "pretrained policy architecture mismatch: "
                        f"expected={expected}, checkpoint={actual}"
                    )
                template.policy.load_state_dict(
                    checkpoint["policy_state_dict"], strict=True
                )
                self._pretrained_policy_metadata = {
                    "path": str(self.pretrained_policy_path),
                    "source_maps": list(checkpoint.get("source_maps", [])),
                    "validation_maps": list(checkpoint.get("validation_maps", [])),
                    "decisions_seen": int(checkpoint.get("decisions_seen", 0)),
                }
            if self.freeze_pretrained_policy:
                template.freeze_policy()
            template_state = {
                name: value.detach().clone()
                for name, value in template.policy.state_dict().items()
            }
            agents: list[OnlineExactUtilityAgent] = []
            rngs: list[random.Random] = []
            for node_idx in range(int(self.cfg.num_nodes)):
                seed = (
                    int(self.cfg.seed)
                    + 2_900_003
                    + mode_offset
                    + 104_729 * int(node_idx)
                )
                agent = OnlineExactUtilityAgent(
                    group_widths=group_widths,
                    trajectory_dim=trajectory_dim,
                    hidden_dim=self.exact_hidden_dim,
                    embedding_dim=self.embedding_dim,
                    gain_hidden_dim=self.gain_hidden_dim,
                    pair_feature_mode=self.pair_feature_mode,
                    device=self.device,
                    learning_rate=self.cfg.rl_lr,
                    rng_seed=seed,
                    model_seed=model_seed,
                    sample_capacity=self.policy_sample_capacity,
                    encoder_lr_scale=self.encoder_lr_scale,
                    normalize_rewards=self.normalize_policy_rewards,
                    reward_scale_db=self.policy_reward_scale_db,
                    ranking_loss_weight=self.policy_ranking_loss_weight,
                    ranking_margin_db=self.policy_ranking_margin_db,
                    ranking_temperature_db=(
                        self.policy_ranking_temperature_db
                    ),
                    ranking_receiver_cosine_min=(
                        self.policy_ranking_receiver_cosine_min
                    ),
                    support_dim=self.policy_support_dim,
                )
                agent.policy.load_state_dict(template_state)
                if self.freeze_policy_encoders:
                    agent.freeze_encoder()
                if self.freeze_pretrained_policy:
                    agent.freeze_policy()
                agents.append(agent)
                rngs.append(random.Random(seed + 57_911))
            self.agents[mode_id] = template
            self.local_agents[mode_id] = agents
            self._local_policy_pending_transitions[mode_id] = [
                0 for _ in agents
            ]
            self._local_policy_versions[mode_id] = [0 for _ in agents]
            self._local_policy_initial_rngs[mode_id] = rngs

    # --------------------------------------------------------- exact histories

    def _raw_state(
        self,
        node_idx: int,
        mode: str,
        *,
        model_state: TensorState | None = None,
    ) -> ExactPrivateState:
        if model_state is None:
            model_state = self.nodes[int(node_idx)].variants[mode].model.state_dict()
        width = self._predictor_input_dim() + 1
        node = int(node_idx)
        history = self._trajectory_histories[node]
        base = history.tensor(width=width)
        zone = int(getattr(self.nodes[node], "current_az", -1))
        visits = (
            int(self._zone_visit_counts[node].get(zone, 0))
            if node < len(self._zone_visit_counts)
            else 0
        )
        measurement_experience = min(
            1.0,
            math.log1p(max(0, int(history.samples_seen)))
            / math.log1p(100_000),
        )
        visit_experience = min(
            1.0,
            math.log1p(max(0, visits)) / math.log1p(100),
        )
        if int(base.shape[0]) > 0:
            experience = torch.tensor(
                [measurement_experience, visit_experience, 0.0],
                dtype=torch.float32,
            ).repeat(int(base.shape[0]), 1)
            trajectory = torch.cat((base, experience), dim=1)
        else:
            trajectory = torch.empty((0, width + 3), dtype=torch.float32)
        summary = torch.zeros((1, width + 3), dtype=torch.float32)
        summary[0, -3:] = torch.tensor(
            [measurement_experience, visit_experience, 1.0],
            dtype=torch.float32,
        )
        trajectory = torch.cat((summary, trajectory), dim=0)
        return ExactPrivateState(
            model_groups=exact_model_groups(model_state),
            trajectory=trajectory,
        )

    def _provider_policy_observation(
        self, node_idx: int, mode: str, provider_view: object
    ) -> tuple[ExactPrivateState, torch.Tensor]:
        """Return the provider state and advertised embedding for ranking.

        Subclasses may advertise a concrete immutable expert instead of the
        node's locally trainable predictor. The default preserves the legacy
        one-predictor behavior.
        """

        state = self._raw_state(
            int(node_idx),
            str(mode),
            model_state=provider_view._model_state,  # type: ignore[attr-defined]
        )
        embedding = self.local_agents[str(mode)][int(node_idx)].policy_embedding(
            state
        )
        return state, embedding

    def _policy_candidate_score(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        mode: str,
        provider_view: object,
        agent: ExactSequentialAgent,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
    ) -> float:
        """Score one receiver/provider contact.

        Subclasses may rank a receiver-specific capsule advertised in a bank
        manifest. The default preserves the learned embedding-pair policy.
        """

        del receiver_idx, provider_idx, mode, provider_view
        return float(
            agent.policy_gain_from_embeddings(
                receiver_embedding, provider_embedding
            )
        )

    def _selected_provider_policy_observation(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        mode: str,
        provider_view: object,
        fallback_state: ExactPrivateState,
        fallback_embedding: torch.Tensor,
    ) -> tuple[ExactPrivateState, torch.Tensor]:
        """Return the precise provider capsule selected for this receiver.

        The default is the provider's single advertised model. Multi-model
        simulations override this so a reward label is paired with the exact
        immutable expert encoding that was scored and transferred.
        """

        del receiver_idx, provider_idx, mode, provider_view
        return fallback_state, fallback_embedding

    def _train_local(
        self,
        ns,
        X: np.ndarray,
        y_dbm: np.ndarray,
        *,
        sample_count_increment: int | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        n_new = (
            int(X.shape[0])
            if sample_count_increment is None
            else max(0, int(sample_count_increment))
        )
        node_idx = self.node_idx(ns)
        previous_train_count = len(ns.current_visit_samples_x) - n_new
        if previous_train_count < 0:
            raise ValueError("new-sample count exceeds the current zone buffer")
        BidirectionalCrossValidationSimulation._train_local(
            self,
            ns,
            X,
            y_dbm,
            sample_count_increment=sample_count_increment,
            sample_weights=sample_weights,
        )
        # The policy sees the complete model-training trajectory, but never the
        # optimization/reward validation samples used to choose or label pulls.
        # This keeps the state input exact without leaking the private holdouts.
        if (
            self.selection_mode != "random"
            and len(self._trajectory_histories) == len(self.nodes)
            and node_idx >= 0
            and len(ns.current_visit_samples_x) > previous_train_count
        ):
            features = np.asarray(
                ns.current_visit_samples_x[previous_train_count:],
                dtype=np.float32,
            )
            targets = np.asarray(
                ns.current_visit_samples_y[previous_train_count:],
                dtype=np.float32,
            ).reshape(-1, 1)
            self._trajectory_histories[node_idx].append(
                features, self._normalize_target_from_rssi(targets)
            )

    def _select_validation_indices(
        self,
        batch_features: np.ndarray,
        opt_existing: np.ndarray,
        reward_existing: np.ndarray,
        n_opt: int,
        n_reward: int,
        *,
        node_idx: int,
        samples_seen: int,
    ) -> tuple[list[int], list[int], list[int]]:
        return BidirectionalCrossValidationSimulation._select_validation_indices(
            self,
            batch_features,
            opt_existing,
            reward_existing,
            n_opt,
            n_reward,
            node_idx=node_idx,
            samples_seen=samples_seen,
        )

    def _save_node_zone_memory(self, node_idx: int, zone: int) -> None:
        BidirectionalCrossValidationSimulation._save_node_zone_memory(
            self, node_idx, zone
        )
        if (
            self.selection_mode != "random"
            and self.zone_model_memory
            and len(self._trajectory_histories) == len(self.nodes)
        ):
            self._node_zone_memory[int(node_idx)][int(zone)][
                "exact_trajectory"
            ] = self._trajectory_histories[int(node_idx)].snapshot()

    def _restore_node_zone_memory(self, node_idx: int, zone: int) -> bool:
        restored = BidirectionalCrossValidationSimulation._restore_node_zone_memory(
            self, node_idx, zone
        )
        if (
            self.selection_mode != "random"
            and restored
            and len(self._trajectory_histories) == len(self.nodes)
        ):
            payload = self._node_zone_memory[int(node_idx)][int(zone)].get(
                "exact_trajectory", {}
            )
            self._trajectory_histories[int(node_idx)] = (
                ExactTrajectoryHistory.restore(
                    payload, capacity=self.trajectory_capacity
                )
            )
        return restored

    def _reset_node_for_zone_change(self, ns, new_az: int) -> None:
        node_idx = self.node_idx(ns)
        cached = bool(
            node_idx >= 0
            and hasattr(self, "_node_zone_memory")
            and int(new_az) in self._node_zone_memory[node_idx]
        )
        BidirectionalCrossValidationSimulation._reset_node_for_zone_change(
            self, ns, new_az
        )
        if (
            node_idx >= 0
            and not cached
            and len(self._trajectory_histories) == len(self.nodes)
        ):
            self._trajectory_histories[node_idx] = ExactTrajectoryHistory(
                capacity=self.trajectory_capacity
            )

    # ------------------------------------------------------------ online loss

    def _share_policies_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> None:
        """Share the one decision policy without creating a serving copy."""

        BidirectionalCrossValidationSimulation._share_policies_with_all_feasible_neighbors(
            self, links
        )

    def _train_exact_pair(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        example: _TrainingExample,
        sample_multiplier: int,
    ) -> None:
        """Train only the receiver's private encoder/head from deployable data."""

        if self.selection_mode == "random":
            return
        receiver_agent = self.local_agents[mode][int(receiver_idx)]
        if not any(
            parameter.requires_grad for parameter in receiver_agent.policy.parameters()
        ):
            return
        receiver_agent.policy.train()
        receiver_agent.opt.zero_grad(set_to_none=True)
        receiver_embedding = receiver_agent.policy.encode(example.receiver_state)
        provider_embedding = example.provider_embedding.detach().to(self.device)
        normalized_prediction = receiver_agent.policy.score_embeddings(
            receiver_embedding.unsqueeze(0), provider_embedding.unsqueeze(0)
        )
        if (
            not bool(getattr(self, "share_training_samples", False))
            and (
                self.train_accumulated_head_epoch
                or self.head_replay_batches_per_step > 0
            )
        ):
            receiver_agent.remember_head_example(
                example.receiver_embedding,
                example.provider_embedding,
                float(example.target_gain),
            )
        target = torch.tensor(
            [receiver_agent.normalize_gain(float(example.target_gain))],
            dtype=torch.float32,
            device=self.device,
        )
        importance = (
            float(sample_multiplier)
            / max(float(example.propensity), 1.0e-8)
            if self.inverse_propensity_weighting
            else 1.0
        )
        base_loss = F.smooth_l1_loss(normalized_prediction, target)
        loss = base_loss * importance
        loss.backward()
        nn.utils.clip_grad_norm_(receiver_agent.policy.parameters(), 5.0)
        receiver_agent.opt.step()
        receiver_agent.experience += 1
        receiver_agent.local_evaluated_pulls += 1
        receiver_agent.train_steps += 1
        self._local_policy_train_updates[mode] += 1
        self._last_local_policy_train_updates_this_step += 1
        self._last_local_policy_queued_transitions += 1
        self._ensure_encoded_log().writerow(
            {
                "step": int(step),
                "mode": str(mode),
                "receiver_idx": int(receiver_idx),
                "provider_idx": int(example.provider_idx),
                "target_gain": float(example.target_gain),
                "online_prediction": receiver_agent.denormalize_gain(
                    float(normalized_prediction.detach().cpu().item())
                ),
                "base_loss": float(base_loss.detach().cpu()),
                "weighted_loss": float(loss.detach().cpu()),
                "propensity": float(example.propensity),
                "importance_weight": float(importance),
                "receiver_policy_version": int(
                    self._local_policy_versions[mode][int(receiver_idx)]
                ),
                "provider_policy_version": int(
                    self._local_policy_versions[mode][int(example.provider_idx)]
                ),
            }
        )
        self._encoded_log_count += 1
        if self._encoded_log_count % 1000 == 0:
            self._flush_encoded_log()

    def _policy_sample_id(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
    ) -> bytes:
        generations = getattr(self, "_node_generations", [])
        receiver_generation = (
            int(generations[int(receiver_idx)])
            if int(receiver_idx) < len(generations)
            else 0
        )
        provider_generation = (
            int(generations[int(provider_idx)])
            if int(provider_idx) < len(generations)
            else 0
        )
        payload = (
            f"{int(self.cfg.seed)}|{mode}|{int(step)}|"
            f"{int(receiver_idx)}:{receiver_generation}|"
            f"{int(provider_idx)}:{provider_generation}"
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()

    def _exchange_training_examples(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        receiver_state: ExactPrivateState,
        provider_state: ExactPrivateState,
        receiver_embedding: torch.Tensor,
        provider_embedding: torch.Tensor,
        receiver_gain: float | None,
        provider_gain: float | None,
        propensity: float,
    ) -> list[tuple[int, _TrainingExample]]:
        """Create deployable labels from one evaluated bilateral exchange."""

        rows: list[tuple[int, _TrainingExample]] = []
        if receiver_gain is not None:
            forward = _TrainingExample(
                sample_id=self._policy_sample_id(
                    step=step,
                    mode=mode,
                    receiver_idx=receiver_idx,
                    provider_idx=provider_idx,
                ),
                provider_idx=int(provider_idx),
                receiver_state=receiver_state.clone(),
                receiver_embedding=receiver_embedding.detach().cpu().clone(),
                provider_state=provider_state.clone(),
                provider_embedding=provider_embedding.detach().cpu().clone(),
                target_gain=float(receiver_gain),
                propensity=float(propensity),
            )
            rows.append((int(receiver_idx), forward))
        if self.symmetric_pulls and provider_gain is not None:
            reverse = _TrainingExample(
                sample_id=self._policy_sample_id(
                    step=step,
                    mode=mode,
                    receiver_idx=provider_idx,
                    provider_idx=receiver_idx,
                ),
                provider_idx=int(receiver_idx),
                receiver_state=provider_state.clone(),
                receiver_embedding=provider_embedding.detach().cpu().clone(),
                provider_state=receiver_state.clone(),
                provider_embedding=receiver_embedding.detach().cpu().clone(),
                target_gain=float(provider_gain),
                propensity=float(propensity),
            )
            rows.append((int(provider_idx), reverse))
        return rows

    def _evidence_novelty_scores(
        self,
        *,
        receiver_idx: int,
        mode: str,
        candidate_ids: list[int],
        provider_views: Mapping[int, object],
    ) -> dict[int, float]:
        """Score evidence that the provider can deliver in this pull."""

        receiver_state = self._clone_state(
            self.nodes[int(receiver_idx)].variants[str(mode)].model
        )
        scores: dict[int, float] = {}
        for provider_idx in candidate_ids:
            provider_state = dict(
                provider_views[int(provider_idx)]._model_state
            )
            deliverable = mergeable_evidence_delta_state(
                receiver_state,
                provider_state,
                max_rows=self.mergeable_max_delta_rows,
            )
            scores[int(provider_idx)] = (
                mergeable_evidence_diagonal_information_gain(
                    receiver_state, deliverable
                )
            )
        return scores

    def _policy_result_gain(self, result: PullResult) -> float | None:
        """Return the configured deployable objective for one exchange."""

        penalty = float(getattr(self, "communication_penalty", 0.0))
        if getattr(
            self, "policy_training_target", "validation-gain"
        ) == "parameter-geometry":
            value = result.parameter_geometry_reward
            return (
                None
                if value is None
                else float(value) - penalty
            )
        if getattr(
            self, "policy_training_target", "validation-gain"
        ) == "information-gain":
            if self.policy_reward_scope == "joint":
                values = [
                    value
                    for value in (
                        result.receiver_information_gain,
                        result.provider_information_gain,
                    )
                    if value is not None
                ]
                return None if not values else float(np.mean(values))
            return result.receiver_information_gain
        raw = (
            result.joint_reward
            if self.policy_reward_scope == "joint"
            else result.reward
        )
        return (
            None
            if raw is None
            else float(raw) - penalty
        )

    def _policy_training_gains(
        self, result: PullResult
    ) -> tuple[float | None, float | None]:
        """Return forward/reverse labels without exposing either holdout set."""

        penalty = float(getattr(self, "communication_penalty", 0.0))
        if getattr(
            self, "policy_training_target", "validation-gain"
        ) == "parameter-geometry":
            value = result.parameter_geometry_reward
            target = (
                None
                if value is None
                else float(value) - penalty
            )
            return target, target
        if getattr(
            self, "policy_training_target", "validation-gain"
        ) == "information-gain":
            return (
                result.receiver_information_gain,
                result.provider_information_gain,
            )
        if self.policy_reward_scope == "joint":
            values = result.joint_reward, result.joint_reward
        else:
            values = result.receiver_reward, result.provider_reward
        return tuple(
            None
            if value is None
            else float(value) - penalty
            for value in values
        )

    def _policy_decision_threshold(
        self, agent: OnlineExactUtilityAgent
    ) -> float:
        if self.policy_fixed_trigger_db is not None:
            return float(self.policy_fixed_trigger_db)
        return float(
            agent.policy_trigger_threshold(self.policy_trigger_quantile)
        )

    def _align_encoder_pairs(
        self, pairs: list[tuple[str, int, int]]
    ) -> None:
        if not self.align_policy_encoders or self.selection_mode != "policy":
            return
        seen: set[tuple[str, int, int]] = set()
        for mode, first, second in pairs:
            key = (str(mode), min(int(first), int(second)), max(int(first), int(second)))
            if key in seen:
                continue
            seen.add(key)
            first_agent = self.local_agents[str(mode)][int(first)]
            second_agent = self.local_agents[str(mode)][int(second)]
            first_agent.align_encoder_with(second_agent)
            transferred = 2 * int(first_agent.encoder_nbytes())
            self._cv_step_policy_bytes[str(mode)] += transferred
            self._cv_step_policy_messages[str(mode)] += 2
            self._local_policy_versions[str(mode)][int(first)] += 1
            self._local_policy_versions[str(mode)][int(second)] += 1

    # ------------------------------------------------------------ sequential

    def _pull_quota(
        self, agent: OnlineExactUtilityAgent
    ) -> tuple[int, float]:
        """Return this receiver's pull-attempt quota and activation probability."""

        if self.pull_budget < 1.0:
            probability = float(self.pull_budget)
            return int(agent._py_rng.random() < probability), probability
        return int(self.pull_budget), 1.0

    @staticmethod
    def _policy_sample_count(agent: OnlineExactUtilityAgent) -> int:
        return int(len(agent.head_replay_targets))

    def _policy_cold_start(
        self, agent: OnlineExactUtilityAgent
    ) -> bool:
        return (
            self.selection_mode == "policy"
            and self._policy_sample_count(agent) < self.policy_min_samples
        )

    def _policy_warmup_pull_ready(
        self, *, step: int, node_idx: int, mode: str
    ) -> bool:
        """Deterministically thin warm-up opportunities without a later cap."""

        probability = float(self.policy_warmup_pull_probability)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        payload = (
            f"{int(self.cfg.seed)}|policy-warmup-pull|"
            f"{int(step)}|{int(node_idx)}|{str(mode)}"
        ).encode("utf-8")
        draw = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            byteorder="big",
            signed=False,
        ) / float(1 << 64)
        return bool(draw < probability)

    def _effective_exploration_probability(
        self, agent: OnlineExactUtilityAgent
    ) -> float:
        if self.policy_exploration_decay_samples <= 0:
            return float(self.exploration_probability)
        learned_samples = max(
            0, self._policy_sample_count(agent) - self.policy_min_samples
        )
        progress = min(
            1.0,
            float(learned_samples)
            / float(self.policy_exploration_decay_samples),
        )
        return float(
            self.policy_exploration_start
            + progress
            * (
                self.exploration_probability
                - self.policy_exploration_start
            )
        )

    def _choose_next_candidate(
        self,
        *,
        agent: OnlineExactUtilityAgent,
        remaining_all: set[int],
        policy_best: int | None,
        activation_probability: float,
        force_random: bool = False,
        exploration_probability: float | None = None,
    ) -> tuple[int, float, bool, bool] | None:
        """Choose one provider and return its complete action propensity."""

        ordered = sorted(remaining_all)
        if self.selection_mode == "random" or bool(force_random):
            provider_idx = ordered[agent._py_rng.randrange(len(ordered))]
            propensity = float(activation_probability) / float(len(ordered))
            return int(provider_idx), float(propensity), True, False
        if self.selection_mode in {"oracle", "novelty"}:
            if policy_best is None:
                return None
            return int(policy_best), 1.0, False, True


        epsilon = float(
            self.exploration_probability
            if exploration_probability is None
            else exploration_probability
        )
        explore = agent._py_rng.random() < epsilon
        if explore:
            provider_idx = ordered[agent._py_rng.randrange(len(ordered))]
            propensity = epsilon / float(len(ordered))
            if provider_idx == policy_best:
                propensity += 1.0 - epsilon
            return (
                int(provider_idx),
                float(activation_probability) * float(propensity),
                True,
                False,
            )
        if policy_best is None:
            return None
        propensity = (1.0 - epsilon) + epsilon / float(len(remaining_all))
        return (
            int(policy_best),
            float(activation_probability) * float(propensity),
            False,
            False,
        )

    def _training_examples_for_receiver(
        self,
        agent: OnlineExactUtilityAgent,
        examples: list[_TrainingExample],
    ) -> list[tuple[_TrainingExample, int]]:
        if not examples:
            return []
        if self.train_all_current_examples:
            return [(example, 1) for example in examples]
        chosen = examples[agent._py_rng.randrange(len(examples))]
        return [(chosen, len(examples))]

    def _train_pending_examples(
        self,
        *,
        step: int,
        pending_training: list[tuple[str, int, _TrainingExample, int]],
    ) -> None:
        for mode, receiver_idx, example, multiplier in pending_training:
            self._train_exact_pair(
                step=int(step),
                mode=mode,
                receiver_idx=int(receiver_idx),
                example=example,
                sample_multiplier=int(multiplier),
            )

    def _candidate_transfer_eligible(
        self,
        *,
        receiver_idx: int,
        provider_idx: int,
        mode: str,
        provider_view: object,
    ) -> bool:
        """Extension hook for protocols that advertise multiple models."""

        del receiver_idx, provider_idx, mode, provider_view
        return True

    def _continue_sequential_pull_round(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        result: PullResult,
    ) -> bool:
        """Extension hook for state-dependent per-receiver pull limits."""

        del step, mode, receiver_idx, provider_idx, result
        return True

    def _remaining_sequential_candidates_after_pull(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        selected_provider_idx: int,
        remaining_provider_ids: set[int],
        provider_views: dict[int, object],
        result: PullResult,
    ) -> set[int]:
        """Refresh provider opportunities after one sequential pull.

        The default preserves the historical one-model-per-provider behavior.
        Protocols that advertise several independently transferable models may
        override this hook and retain a provider while another eligible model
        remains in its fixed timestep-start view.
        """

        del step, mode, receiver_idx, provider_views, result
        remaining = set(int(value) for value in remaining_provider_ids)
        remaining.discard(int(selected_provider_idx))
        return remaining

    def _prepare_peer_view_batch(
        self,
        *,
        node_indices: list[int],
        modes: tuple[str, ...],
    ) -> None:
        """Extension hook for protocols that can batch advertisements."""

        del node_indices, modes

    def _provider_pull_budget_required(self) -> bool:
        """Whether serving requires the provider receiver-side pull token."""

        return True

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,

    ) -> None:
        if self.aux_only:
            self._network_step_stats = {}
            return
        if self.selection_mode == "isolated":
            self._network_step_stats = {}
            return

        self._sync_visit_budgets(step=int(step), zone_nodes=zone_nodes)
        if self.realistic_network:
            self._gossip_step_realistic(step, zone_nodes, contact_links)
            return
        links = self._normalized_contact_links(zone_nodes, contact_links)
        if self.selection_mode == "policy":
            self._share_policies_with_all_feasible_neighbors(links)
        if not links:
            return
        mutable: dict[int, list[tuple[int, int]]] = {}
        for zone, a, b in links:
            mutable.setdefault(int(a), []).append((int(zone), int(b)))
            mutable.setdefault(int(b), []).append((int(zone), int(a)))
        neighbors = {
            receiver: sorted(set(rows), key=lambda row: (row[0], row[1]))
            for receiver, rows in mutable.items()
        }
        self._prepare_peer_view_batch(
            node_indices=sorted(neighbors),
            modes=tuple(str(mode) for mode in self.agents),
        )

        provider_views: dict[str, dict[int, object]] = {}
        provider_states: dict[str, dict[int, ExactPrivateState]] = {}
        provider_embeddings: dict[str, dict[int, torch.Tensor]] = {}
        pending_training: list[
            tuple[str, int, _TrainingExample, int]
        ] = []
        for mode in self.agents:
            provider_views[mode] = {}
            provider_states[mode] = {}
            provider_embeddings[mode] = {}
            for node_idx in sorted(neighbors):
                view = self._make_peer_view(self.nodes[node_idx], mode)
                provider_views[mode][node_idx] = view
                if self.selection_mode == "policy":
                    state, embedding = self._provider_policy_observation(
                        node_idx, mode, view
                    )
                    provider_states[mode][node_idx] = state
                    provider_embeddings[mode][node_idx] = embedding

        for receiver_idx in sorted(neighbors):
            candidates = neighbors[receiver_idx]
            for mode in self.agents:
                if not self._pull_budget_available(
                    mode=str(mode), node_idx=int(receiver_idx)
                ):
                    continue
                agent = self.local_agents[mode][receiver_idx]
                eligible_candidates = [
                    (zone, provider)
                    for zone, provider in candidates
                    if (
                        not self._provider_pull_budget_required()
                        or self._pull_budget_available(
                            mode=str(mode), node_idx=int(provider)
                        )
                    )
                    and self._candidate_transfer_eligible(
                        receiver_idx=int(receiver_idx),
                        provider_idx=int(provider),
                        mode=str(mode),
                        provider_view=provider_views[mode][provider],
                    )
                ]
                if not eligible_candidates:
                    continue
                candidate_ids = [
                    provider for _zone, provider in eligible_candidates
                ]
                zones = {
                    provider: zone
                    for zone, provider in eligible_candidates
                }
                enc_ids: dict[int, int] = {}
                for provider_idx in candidate_ids:
                    enc_ids[provider_idx] = int(self._next_enc_id)
                    self._next_enc_id += 1

                if self.selection_mode in {"random", "oracle"}:
                    receiver_state = None
                    receiver_embedding = None
                    current_scores = {
                        provider: float("nan") for provider in candidate_ids
                    }
                elif self.selection_mode == "novelty":
                    receiver_state = None
                    receiver_embedding = None
                    current_scores = self._evidence_novelty_scores(
                        receiver_idx=int(receiver_idx),
                        mode=str(mode),
                        candidate_ids=candidate_ids,
                        provider_views=provider_views[mode],
                    )
                else:
                    receiver_state = self._raw_state(receiver_idx, mode)
                    receiver_embedding = agent.policy_embedding(receiver_state)
                    current_scores = {
                        provider: self._policy_candidate_score(
                            receiver_idx=int(receiver_idx),
                            provider_idx=int(provider),
                            mode=str(mode),
                            provider_view=provider_views[mode][provider],
                            agent=agent,
                            receiver_embedding=receiver_embedding,
                            provider_embedding=provider_embeddings[mode][provider],
                        )
                        for provider in candidate_ids
                    }
                remaining_all = set(candidate_ids)
                selected_rows: dict[int, tuple[float, object, bool]] = {}
                training_examples: list[_TrainingExample] = []
                token_stream = f"{mode}:model-pull"
                token = self._pull_state(
                    step=int(step),
                    node_idx=int(receiver_idx),
                    mode=str(mode),
                    stream=token_stream,
                )
                score_values = np.asarray(
                    list(current_scores.values()), dtype=np.float64
                )
                receiver_model_samples = int(
                    getattr(
                        self.nodes[receiver_idx].variants[mode],
                        "n_samples",
                        0,
                    )
                )
                candidate_model_samples = np.asarray(
                    [
                        int(
                            getattr(
                                self.nodes[provider].variants[mode],
                                "n_samples",
                                0,
                            )
                        )
                        for provider in candidate_ids
                    ],
                    dtype=np.float64,
                )
                regular_count = min(
                    int(self.diagnostic_regular_count), len(self.nodes)
                )
                decision_diag: dict[str, int | float | str | bool] = {
                    "step": int(step),
                    "mode": str(mode),
                    "receiver_idx": int(receiver_idx),
                    "selection_mode": self.selection_mode,
                    "window_index": int(token.window_index),
                    "window_offset": int(token.window_offset),
                    "token_phase": int(token.phase),
                    "random_offset": int(token.random_offset),
                    "token_available": bool(token.available),
                    "token_remaining": int(
                        getattr(token, "remaining", int(token.available))
                    ),
                    "candidate_count": len(candidate_ids),
                    "receiver_model_samples": receiver_model_samples,
                    "candidate_model_samples_mean": float(
                        np.mean(candidate_model_samples)
                    ),
                    "candidate_model_samples_max": int(
                        np.max(candidate_model_samples)
                    ),
                    "candidate_more_samples_fraction": float(
                        np.mean(candidate_model_samples > receiver_model_samples)
                    ),
                    "candidate_regular_fraction": float(
                        np.mean([provider < regular_count for provider in candidate_ids])
                    ),
                    "predicted_gain_max": float(np.max(score_values)),
                    "predicted_gain_mean": float(np.mean(score_values)),
                    "predicted_gain_std": float(np.std(score_values)),
                    "predicted_positive_fraction": float(
                        np.mean(score_values > 0.0)
                    ),
                    "attempted": False,
                    "timing_reason": (
                        "token_spent" if not token.available else "wait"
                    ),
                }
                global_warmup_random = (
                    self.selection_mode == "policy"
                    and int(step) < self.policy_warmup_steps
                )
                cold_start_random = self._policy_cold_start(agent)
                warmup_random = bool(
                    global_warmup_random or cold_start_random
                )
                effective_exploration = (
                    self._effective_exploration_probability(agent)
                )
                decision_diag.update(
                    {
                        "policy_warmup": bool(global_warmup_random),
                        "policy_cold_start": bool(cold_start_random),
                        "policy_sample_count": self._policy_sample_count(agent),
                        "effective_exploration_probability": float(
                            effective_exploration
                        ),
                        "policy_trigger_threshold": float(
                            self._policy_decision_threshold(agent)
                        ),
                    }
                )
                policy_opportunity = True
                if (
                    self.selection_mode == "policy"
                    and token.available
                    and not warmup_random
                    and policy_opportunity
                ):
                    self._cv_step_metadata_bytes[mode] += (
                        len(candidate_ids) * int(self.embedding_wire_bytes)
                    )
                activation_probability = 1.0
                token_remaining = int(
                    getattr(token, "remaining", int(token.available))
                )
                if not token.available:
                    pull_quota = 0
                elif self.selection_mode == "random" or warmup_random:
                    random_ready = self._random_window_ready(
                        token, step=int(step), node_idx=int(receiver_idx)
                    )
                    warmup_probability_ready = bool(
                        not global_warmup_random
                        or self._policy_warmup_pull_ready(
                            step=int(step),
                            node_idx=int(receiver_idx),
                            mode=str(mode),
                        )
                    )
                    random_ready = bool(
                        random_ready and warmup_probability_ready
                    )
                    pull_quota = token_remaining if random_ready else 0
                    decision_diag["timing_reason"] = (
                        "warmup_probability_wait"
                        if global_warmup_random
                        and not warmup_probability_ready
                        else "random_scheduled"
                        if random_ready
                        else "random_wait"
                    )
                else:
                    pull_quota = token_remaining
                decision_diag["pull_quota"] = int(pull_quota)
                pull_attempts = 0
                decision_attempt_rows: list[
                    dict[str, int | float | str | bool]
                ] = []

                while remaining_all and pull_attempts < pull_quota:
                    policy_best: int | None = None
                    if self.selection_mode == "oracle":
                        oracle_results = {
                            provider: self._execute_validation_pull(
                                step=int(step),
                                mode=mode,
                                receiver=self.nodes[receiver_idx],
                                provider=self.nodes[provider],
                                zone=int(zones[provider]),
                                provider_view=provider_views[mode][provider],
                                diagnostic=True,
                            )
                            for provider in remaining_all
                        }
                        realized = {
                            provider: float(self._policy_result_gain(result))
                            for provider, result in oracle_results.items()
                            if result.valid
                            and self._policy_result_gain(result) is not None
                        }
                        if realized:
                            realized_values = np.asarray(
                                list(realized.values()), dtype=np.float64
                            )
                            decision_diag.update(
                                {
                                    "oracle_valid_provider_count": int(
                                        len(realized)
                                    ),
                                    "oracle_gain_max": float(
                                        np.max(realized_values)
                                    ),
                                    "oracle_gain_mean": float(
                                        np.mean(realized_values)
                                    ),
                                    "oracle_gain_std": float(
                                        np.std(realized_values)
                                    ),
                                    "oracle_positive_fraction": float(
                                        np.mean(realized_values > 0.0)
                                    ),
                                    "oracle_provider_headroom": float(
                                        np.max(realized_values)
                                        - np.mean(realized_values)
                                    ),
                                }
                            )
                            eligible = {
                                provider
                                for provider, gain in realized.items()
                                if gain > 0.0
                            }
                            if eligible:
                                policy_best = max(
                                    eligible,
                                    key=lambda provider: (
                                        realized[provider],
                                        -int(provider),
                                    ),
                                )
                                decision_diag["timing_reason"] = "oracle_positive"
                            else:
                                decision_diag["timing_reason"] = (
                                    "oracle_no_positive"
                                )
                        else:
                            decision_diag["timing_reason"] = "oracle_no_valid"
                    elif self.selection_mode == "novelty":
                        policy_best = max(
                            remaining_all,
                            key=lambda provider: (
                                float(current_scores[provider]),
                                -int(provider),
                            ),
                        )
                        decision_diag["timing_reason"] = "novelty_ranked"
                    elif self.selection_mode == "policy" and not warmup_random:
                        trigger_threshold = self._policy_decision_threshold(
                            agent
                        )
                        eligible = (
                            set(remaining_all)
                            if (
                                not self.allow_unused_policy_tokens
                                and self._window_deadline(
                                    token,
                                    step=int(step),
                                    node_idx=int(receiver_idx),
                                )
                            )
                            else {
                                provider
                                for provider in remaining_all
                                if float(current_scores[provider])
                                > float(trigger_threshold)
                            }
                        )
                        if eligible:
                            policy_best = max(
                                eligible,
                                key=lambda provider: (
                                    float(current_scores[provider]),
                                    -int(provider),
                                ),
                            )
                    else:
                        eligible = set()
                    choice = self._choose_next_candidate(
                        agent=agent,
                        remaining_all=remaining_all,
                        policy_best=policy_best,
                        activation_probability=activation_probability,
                        force_random=warmup_random,
                        exploration_probability=effective_exploration,
                    )
                    if choice is None:
                        if (
                            token.available
                            and self.selection_mode == "policy"
                            and not warmup_random
                        ):
                            decision_diag["timing_reason"] = "policy_wait"
                        break
                    provider_idx, propensity, explore, _unused = choice
                    self._spend_pull_budget(
                        step=int(step),
                        mode=str(mode),
                        receiver_idx=int(receiver_idx),
                        provider_idx=int(provider_idx),
                        stream=token_stream,
                    )
                    pull_attempts += 1
                    decision_diag["attempted"] = True
                    decision_diag["selected_provider_idx"] = int(provider_idx)
                    selected_provider_samples = int(
                        getattr(
                            self.nodes[provider_idx].variants[mode],
                            "n_samples",
                            0,
                        )
                    )
                    decision_diag["selected_provider_regular"] = bool(
                        provider_idx < regular_count
                    )
                    decision_diag["selected_provider_model_samples"] = (
                        selected_provider_samples
                    )
                    decision_diag["selected_provider_sample_advantage"] = (
                        selected_provider_samples - receiver_model_samples
                    )
                    decision_diag["selected_provider_more_samples"] = bool(
                        selected_provider_samples > receiver_model_samples
                    )
                    decision_diag["exploratory"] = bool(explore)

                    predicted = float(current_scores[provider_idx])
                    if self.selection_mode == "random":
                        decision_diag["selected_predicted_gain"] = float("nan")
                        decision_diag["selected_predicted_rank"] = 0
                        before_state = None
                    else:
                        predicted_order = sorted(
                            candidate_ids,
                            key=lambda provider: (
                                -float(current_scores[provider]),
                                int(provider),
                            ),
                        )
                        decision_diag["selected_predicted_gain"] = predicted
                        decision_diag["selected_predicted_rank"] = int(
                            predicted_order.index(provider_idx) + 1
                        )
                        before_state = (
                            receiver_state.clone()
                            if self.selection_mode == "policy"
                            and receiver_state is not None
                            else None
                        )
                    result = self._execute_validation_pull(
                        step=int(step),
                        mode=mode,
                        receiver=self.nodes[receiver_idx],
                        provider=self.nodes[provider_idx],
                        zone=int(zones[provider_idx]),
                        provider_view=provider_views[mode][provider_idx],
                    )
                    selected_rows[provider_idx] = (
                        predicted,
                        result,
                        bool(explore),
                    )
                    policy_gain = self._policy_result_gain(result)
                    decision_diag.update(
                        {
                            "valid": bool(result.valid),
                            "reason": str(result.reason),
                            "realized_gain": (
                                float(policy_gain)
                                if policy_gain is not None
                                else float("nan")
                            ),
                            "adopted": bool(result.adopted),
                        }
                    )
                    decision_diag["pull_index"] = int(pull_attempts)
                    decision_attempt_rows.append(dict(decision_diag))
                    if (
                        self.selection_mode == "policy"
                        and result.valid
                        and policy_gain is not None
                    ):
                        assert before_state is not None
                        assert receiver_embedding is not None
                        receiver_gain, provider_gain = (
                            self._policy_training_gains(result)
                        )
                        selected_provider_state, selected_provider_embedding = (
                            self._selected_provider_policy_observation(
                                receiver_idx=int(receiver_idx),
                                provider_idx=int(provider_idx),
                                mode=str(mode),
                                provider_view=provider_views[mode][provider_idx],
                                fallback_state=provider_states[mode][provider_idx],
                                fallback_embedding=provider_embeddings[mode][provider_idx],
                            )
                        )
                        examples = self._exchange_training_examples(
                            step=int(step),
                            mode=str(mode),
                            receiver_idx=int(receiver_idx),
                            provider_idx=int(provider_idx),
                            receiver_state=before_state,
                            provider_state=selected_provider_state,
                            receiver_embedding=receiver_embedding,
                            provider_embedding=selected_provider_embedding,
                            receiver_gain=receiver_gain,
                            provider_gain=provider_gain,
                            propensity=float(propensity),
                        )
                        training_examples.extend(
                            example for owner, example in examples
                            if owner == int(receiver_idx)
                        )
                    remaining_all = (
                        self._remaining_sequential_candidates_after_pull(
                            step=int(step),
                            mode=str(mode),
                            receiver_idx=int(receiver_idx),
                            selected_provider_idx=int(provider_idx),
                            remaining_provider_ids=remaining_all,
                            provider_views=provider_views[mode],
                            result=result,
                        )
                    )
                    if not self._continue_sequential_pull_round(
                        step=int(step),
                        mode=str(mode),
                        receiver_idx=int(receiver_idx),
                        provider_idx=int(provider_idx),
                        result=result,
                    ):
                        break
                    if self.selection_mode == "policy":
                        receiver_state = self._raw_state(receiver_idx, mode)
                        receiver_embedding = agent.policy_embedding(receiver_state)
                        for candidate in remaining_all:
                            current_scores[candidate] = self._policy_candidate_score(
                                receiver_idx=int(receiver_idx),
                                provider_idx=int(candidate),
                                mode=str(mode),
                                provider_view=provider_views[mode][candidate],
                                agent=agent,
                                receiver_embedding=receiver_embedding,
                                provider_embedding=provider_embeddings[mode][candidate],
                            )
                    elif self.selection_mode == "novelty":
                        refreshed = self._evidence_novelty_scores(
                            receiver_idx=int(receiver_idx),
                            mode=str(mode),
                            candidate_ids=sorted(remaining_all),
                            provider_views=provider_views[mode],
                        )
                        current_scores.update(refreshed)

                if not decision_attempt_rows:
                    decision_diag["pull_index"] = 0
                    decision_attempt_rows.append(dict(decision_diag))
                self._token_decision_rows.extend(decision_attempt_rows)

                for chosen, multiplier in self._training_examples_for_receiver(
                    agent, training_examples
                ):
                    pending_training.append(
                        (
                            str(mode),
                            int(receiver_idx),
                            chosen,
                            int(multiplier),
                        )
                    )

                for provider_idx in candidate_ids:
                    if provider_idx in selected_rows:
                        predicted, result, exploratory = selected_rows[provider_idx]
                        selected_gain = self._policy_result_gain(result)
                        raw_reward = (
                            0.0
                            if selected_gain is None
                            else float(selected_gain)
                        )
                        self._record_encoded_decision(
                            step=int(step),
                            enc_id=enc_ids[provider_idx],
                            zone=zones[provider_idx],
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                            mode=mode,
                            action=1,
                            predicted_gain=float(predicted),
                            reward=raw_reward,
                            exploratory=bool(exploratory),
                            alpha=result.alpha,
                        )
                    else:
                        self._record_encoded_decision(
                            step=int(step),
                            enc_id=enc_ids[provider_idx],
                            zone=zones[provider_idx],
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                            mode=mode,
                            action=0,
                            predicted_gain=float(current_scores[provider_idx]),
                            reward=0.0,
                            exploratory=False,
                            alpha=None,
                        )

        # All actions in the timestep use exactly the policy shared at its
        # start.  Learning happens only after every receiver has completed its
        # sequential, state-dependent pull round.
        if self.train_all_current_examples and pending_training:
            random.Random(
                int(self.cfg.seed) * 1_000_003 + int(step) * 65_537
            ).shuffle(pending_training)
        self._train_pending_examples(
            step=int(step), pending_training=pending_training
        )
        active_indices = sorted(
            {
                int(node_idx)
                for node_indices in zone_nodes.values()
                for node_idx in node_indices
            }
        )
        if self.selection_mode == "policy" and self.train_accumulated_head_epoch:
            for agents in self.local_agents.values():
                for node_idx in active_indices:
                    agents[node_idx].train_head_epoch()
        elif (
            self.selection_mode == "policy"
            and self.head_replay_batches_per_step > 0
        ):
            for agents in self.local_agents.values():
                for node_idx in active_indices:
                    agents[node_idx].train_head_batches(
                        num_batches=self.head_replay_batches_per_step
                    )

    def _gossip_step_realistic(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None,
    ) -> None:
        """Plan pulls first, then apply shared-channel scheduling and SINR."""

        links = self._normalized_contact_links(zone_nodes, contact_links)
        powers = {
            (int(tx), int(rx)): float(value)
            for (tx, rx), value in dict(
                getattr(self, "_current_link_rssi_dbm", {})
            ).items()
        }
        if not links:
            self._network_step_stats = {
                "network_raw_contact_pairs": 0,
                "network_shortlisted_directed_candidates": 0,
                "network_transfer_proposals": 0,
                "network_scheduled_transfers": 0,
            }
            return

        directed: dict[int, list[tuple[int, int, float]]] = {}
        for zone, a, b in links:
            ab = float(powers.get((a, b), self.network_missing_power_dbm))
            ba = float(powers.get((b, a), self.network_missing_power_dbm))
            strength = min(ab, ba)
            directed.setdefault(a, []).append((zone, b, strength))
            directed.setdefault(b, []).append((zone, a, strength))
        neighbors = {}
        for receiver, rows in directed.items():
            ordered = sorted(rows, key=lambda row: (-row[2], row[1]))
            if self.network_candidate_top_k > 0:
                ordered = ordered[: self.network_candidate_top_k]
            neighbors[receiver] = [
                (zone, provider) for zone, provider, _strength in ordered
            ]

        participating = set(neighbors)
        participating.update(
            provider for rows in neighbors.values() for _zone, provider in rows
        )
        self._prepare_peer_view_batch(
            node_indices=sorted(participating),
            modes=tuple(str(mode) for mode in self.agents),
        )
        provider_views: dict[str, dict[int, object]] = {}
        provider_states: dict[str, dict[int, ExactPrivateState]] = {}
        provider_embeddings: dict[str, dict[int, torch.Tensor]] = {}
        for mode in self.agents:
            provider_views[mode] = {}
            provider_states[mode] = {}
            provider_embeddings[mode] = {}
            for node_idx in sorted(participating):
                view = self._make_peer_view(self.nodes[node_idx], mode)
                provider_views[mode][node_idx] = view
                if self.selection_mode == "policy":
                    state, embedding = self._provider_policy_observation(
                        node_idx, mode, view
                    )
                    provider_states[mode][node_idx] = state
                    provider_embeddings[mode][node_idx] = embedding

        assumptions = self._communication_assumptions
        model_bytes = int(assumptions.get("B_model_bytes", 0))
        policy_bytes = (
            int(assumptions.get("B_policy_bytes", 0))
            if bool(self.local_policy_share)
            else 0
        )
        accepted_bytes = int(
            assumptions.get("B_accepted_pull_bytes", 2 * model_bytes)
        )
        scalar_bytes = max(0, accepted_bytes - 2 * model_bytes)
        encoder_alignment_bytes = 0
        if (
            self.align_policy_encoders
            and self.selection_mode == "policy"
            and not bool(getattr(self, "share_training_samples", False))
        ):
            rows = next(iter(self.local_agents.values()), [])
            if rows:
                encoder_alignment_bytes = int(rows[0].encoder_nbytes())
        # Both current embeddings are advertised before provider ranking and
        # are already included in policy metadata accounting. Do not charge
        # either embedding again to the predictor payload: policy, random,
        # and oracle must schedule the same model/scalar byte count.
        reverse_training_embedding_bytes = 0
        reservation_control_bytes = (
            self.network_reservation_control_bytes
            if self.network_decentralized_reservation
            else 0
        )
        directional_overhead = (
            policy_bytes
            + encoder_alignment_bytes
            + int(math.ceil(scalar_bytes / 2.0))
            + int(math.ceil(reservation_control_bytes / 2.0))
        )

        contexts: list[_RealisticDecisionContext] = []
        proposal_to_context: dict[TransferProposal, _RealisticDecisionContext] = {}
        reservation_nodes: set[int] = set()
        receiver_order = sorted(neighbors)
        if self.network_decentralized_reservation:
            receiver_order = decentralized_reservation_order(
                receiver_order,
                step=int(step),
                seed=int(self.cfg.seed),
            )
        for receiver_idx in receiver_order:
            candidates = neighbors[receiver_idx]
            for mode in self.agents:
                if not self._pull_budget_available(
                    mode=str(mode), node_idx=int(receiver_idx)
                ):
                    continue
                agent = self.local_agents[mode][receiver_idx]
                eligible_candidates = [
                    (zone, provider)
                    for zone, provider in candidates
                    if self._pull_budget_available(
                        mode=str(mode), node_idx=int(provider)
                    )
                ]
                if not eligible_candidates:
                    continue
                candidate_ids = [
                    provider for _zone, provider in eligible_candidates
                ]
                zones = {
                    provider: zone
                    for zone, provider in eligible_candidates
                }
                enc_ids: dict[int, int] = {}
                for provider_idx in candidate_ids:
                    enc_ids[provider_idx] = int(self._next_enc_id)
                    self._next_enc_id += 1

                if self.selection_mode in {"random", "oracle"}:
                    receiver_state = None
                    receiver_embedding = None
                    scores = {
                        provider: float("nan") for provider in candidate_ids
                    }
                elif self.selection_mode == "novelty":
                    receiver_state = None
                    receiver_embedding = None
                    scores = self._evidence_novelty_scores(
                        receiver_idx=int(receiver_idx),
                        mode=str(mode),
                        candidate_ids=candidate_ids,
                        provider_views=provider_views[mode],
                    )
                else:
                    receiver_state = self._raw_state(receiver_idx, mode)
                    receiver_embedding = agent.policy_embedding(receiver_state)
                    scores = {
                        provider: self._policy_candidate_score(
                            receiver_idx=int(receiver_idx),
                            provider_idx=int(provider),
                            mode=str(mode),
                            provider_view=provider_views[mode][provider],
                            agent=agent,
                            receiver_embedding=receiver_embedding,
                            provider_embedding=provider_embeddings[mode][provider],
                        )
                        for provider in candidate_ids
                    }
                score_values = np.asarray(list(scores.values()), dtype=np.float64)
                receiver_samples = int(
                    getattr(self.nodes[receiver_idx].variants[mode], "n_samples", 0)
                )
                candidate_samples = np.asarray(
                    [
                        int(getattr(self.nodes[p].variants[mode], "n_samples", 0))
                        for p in candidate_ids
                    ],
                    dtype=np.float64,
                )
                token_stream = f"{mode}:model-pull"
                token = self._pull_state(
                    step=int(step),
                    node_idx=int(receiver_idx),
                    mode=str(mode),
                    stream=token_stream,
                )
                global_warmup_random = (
                    self.selection_mode == "policy"
                    and int(step) < self.policy_warmup_steps
                )
                cold_start_random = self._policy_cold_start(agent)
                warmup_random = bool(
                    global_warmup_random or cold_start_random
                )
                effective_exploration = (
                    self._effective_exploration_probability(agent)
                )
                policy_opportunity = True
                if (
                    self.selection_mode == "policy"
                    and token.available
                    and not warmup_random
                    and policy_opportunity
                ):
                    self._cv_step_metadata_bytes[mode] += (
                        len(candidate_ids) * int(self.embedding_wire_bytes)
                    )
                diag: dict[str, int | float | str | bool] = {
                    "step": step,
                    "mode": str(mode),
                    "receiver_idx": receiver_idx,
                    "selection_mode": self.selection_mode,
                    "window_index": token.window_index,
                    "window_offset": token.window_offset,
                    "token_phase": token.phase,
                    "random_offset": token.random_offset,
                    "token_available": token.available,
                    "policy_warmup": bool(global_warmup_random),
                    "policy_cold_start": bool(cold_start_random),
                    "policy_sample_count": self._policy_sample_count(agent),
                    "effective_exploration_probability": float(
                        effective_exploration
                    ),
                    "policy_trigger_threshold": float(
                        self._policy_decision_threshold(agent)
                    ),
                    "candidate_count": len(candidate_ids),
                    "raw_candidate_count": len(directed[receiver_idx]),
                    "receiver_model_samples": receiver_samples,
                    "candidate_model_samples_mean": float(np.mean(candidate_samples)),
                    "candidate_model_samples_max": int(np.max(candidate_samples)),
                    "candidate_more_samples_fraction": float(
                        np.mean(candidate_samples > receiver_samples)
                    ),
                    "predicted_gain_max": float(np.max(score_values)),
                    "predicted_gain_mean": float(np.mean(score_values)),
                    "predicted_gain_std": float(np.std(score_values)),
                    "predicted_positive_fraction": float(np.mean(score_values > 0.0)),
                    "network_proposed": False,
                    "network_scheduled": False,
                    "attempted": False,
                    "timing_reason": "token_spent" if not token.available else "wait",
                }
                context = _RealisticDecisionContext(
                    mode=str(mode),
                    receiver_idx=receiver_idx,
                    agent=agent,
                    candidate_ids=candidate_ids,
                    zones=zones,
                    enc_ids=enc_ids,
                    current_scores=scores,
                    decision_diag=diag,
                    receiver_state=receiver_state,
                    receiver_embedding=receiver_embedding,
                    token_stream=token_stream,
                )
                contexts.append(context)
                if not token.available:
                    continue

                reservation_candidates = set(candidate_ids)
                if self.network_decentralized_reservation:
                    reservation_candidates = available_reservation_candidates(
                        receiver_idx,
                        candidate_ids,
                        reservation_nodes,
                    )
                    if not reservation_candidates:
                        diag["timing_reason"] = (
                            "reservation_node_busy"
                            if int(receiver_idx) in reservation_nodes
                            else "reservation_no_free_provider"
                        )
                        continue

                policy_best: int | None = None
                if self.selection_mode == "random" or warmup_random:
                    if not self._random_window_ready(
                        token, step=int(step), node_idx=int(receiver_idx)
                    ):
                        diag["timing_reason"] = "random_wait"
                        continue
                    if (
                        global_warmup_random
                        and not self._policy_warmup_pull_ready(
                            step=int(step),
                            node_idx=int(receiver_idx),
                            mode=str(mode),
                        )
                    ):
                        diag["timing_reason"] = "warmup_probability_wait"
                        continue
                    diag["timing_reason"] = "random_scheduled"
                elif self.selection_mode == "oracle":
                    oracle_results = {
                        provider: self._execute_validation_pull(
                            step=int(step),
                            mode=mode,
                            receiver=self.nodes[receiver_idx],
                            provider=self.nodes[provider],
                            zone=int(zones[provider]),
                            provider_view=provider_views[mode][provider],
                            diagnostic=True,
                        )
                        for provider in reservation_candidates
                    }
                    realized = {
                        provider: float(self._policy_result_gain(result))
                        for provider, result in oracle_results.items()
                        if result.valid
                        and self._policy_result_gain(result) is not None
                    }
                    if realized:
                        values = np.asarray(
                            list(realized.values()), dtype=np.float64
                        )
                        diag.update(
                            {
                                "oracle_valid_provider_count": int(
                                    len(realized)
                                ),
                                "oracle_gain_max": float(np.max(values)),
                                "oracle_gain_mean": float(np.mean(values)),
                                "oracle_gain_std": float(np.std(values)),
                                "oracle_positive_fraction": float(
                                    np.mean(values > 0.0)
                                ),
                                "oracle_provider_headroom": float(
                                    np.max(values) - np.mean(values)
                                ),
                            }
                        )
                        eligible = {
                            provider
                            for provider, gain in realized.items()
                            if gain > 0.0
                        }
                        if eligible:
                            policy_best = max(
                                eligible,
                                key=lambda provider: (
                                    realized[provider],
                                    -int(provider),
                                ),
                            )
                            diag["timing_reason"] = "oracle_positive"
                        else:
                            diag["timing_reason"] = "oracle_no_positive"
                    else:
                        diag["timing_reason"] = "oracle_no_valid"
                elif self.selection_mode == "novelty":
                    policy_best = max(
                        reservation_candidates,
                        key=lambda provider: (
                            float(scores[provider]),
                            -int(provider),
                        ),
                    )
                    diag["timing_reason"] = "novelty_ranked"
                else:
                    trigger_threshold = self._policy_decision_threshold(agent)
                    eligible = (
                        set(reservation_candidates)
                        if (
                            not self.allow_unused_policy_tokens
                            and self._window_deadline(
                                token,
                                step=int(step),
                                node_idx=int(receiver_idx),
                            )
                        )
                        else {
                            p
                            for p in reservation_candidates
                            if float(scores[p]) > float(trigger_threshold)
                        }
                    )
                    if eligible:
                        policy_best = max(
                            eligible, key=lambda p: (float(scores[p]), -int(p))
                        )
                choice = self._choose_next_candidate(
                    agent=agent,
                    remaining_all=set(reservation_candidates),
                    policy_best=policy_best,
                    activation_probability=1.0,
                    force_random=warmup_random,
                    exploration_probability=effective_exploration,
                )
                if choice is None:
                    diag["timing_reason"] = (
                        "oracle_wait"
                        if self.selection_mode == "oracle"
                        else "policy_wait"
                    )
                    continue
                provider_idx, propensity, explore, _unused = choice
                if self.network_decentralized_reservation:
                    reservation_nodes.update(
                        (int(receiver_idx), int(provider_idx))
                    )
                    diag["reservation_committed"] = True
                context.provider_idx = int(provider_idx)
                context.propensity = float(propensity)
                context.exploratory = bool(explore)
                provider_samples = int(
                    getattr(self.nodes[provider_idx].variants[mode], "n_samples", 0)
                )
                ordered = (
                    sorted(
                        candidate_ids,
                        key=lambda p: (-float(scores[p]), int(p)),
                    )
                    if self.selection_mode != "random"
                    else sorted(candidate_ids)
                )
                forward_rssi = float(
                    powers.get((provider_idx, receiver_idx), self.network_missing_power_dbm)
                )
                reverse_rssi = float(
                    powers.get((receiver_idx, provider_idx), self.network_missing_power_dbm)
                )
                min_standalone_snr = (
                    min(forward_rssi, reverse_rssi)
                    - float(self.cfg.noise_floor_dbm)
                )
                diag.update(
                    {
                        "network_proposed": True,
                        "selected_provider_idx": int(provider_idx),
                        "selected_provider_model_samples": provider_samples,
                        "selected_provider_sample_advantage": provider_samples - receiver_samples,
                        "selected_provider_more_samples": provider_samples > receiver_samples,
                        "selected_link_forward_rssi_dbm": forward_rssi,
                        "selected_link_reverse_rssi_dbm": reverse_rssi,
                        "selected_link_standalone_min_snr_db": min_standalone_snr,
                        "selected_predicted_gain": (
                            float(scores[provider_idx])
                            if self.selection_mode != "random"
                            else float("nan")
                        ),
                        "selected_predicted_rank": (
                            ordered.index(provider_idx) + 1
                            if self.selection_mode != "random"
                            else 0
                        ),
                        "exploratory": bool(explore),
                        "timing_reason": "network_proposed",
                    }
                )
                receiver_model_state = self._clone_state(
                    self.nodes[receiver_idx].variants[mode].model
                )
                provider_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in provider_views[mode][
                        provider_idx
                    ]._model_state.items()
                }
                if (
                    is_mergeable_evidence_state(receiver_model_state)
                    and is_mergeable_evidence_state(provider_model_state)
                ):
                    forward_model_bytes = mergeable_evidence_direction_nbytes(
                        receiver_model_state,
                        provider_model_state,
                        max_rows=self.mergeable_max_delta_rows,
                    )
                    reverse_model_bytes = mergeable_evidence_direction_nbytes(
                        provider_model_state,
                        receiver_model_state,
                        max_rows=self.mergeable_max_delta_rows,
                    )
                else:
                    forward_model_bytes = model_bytes
                    reverse_model_bytes = model_bytes
                forward_payload = forward_model_bytes + directional_overhead
                reverse_payload = reverse_model_bytes + directional_overhead
                diag["network_forward_payload_bytes"] = int(forward_payload)
                diag["network_reverse_payload_bytes"] = int(reverse_payload)
                proposal = TransferProposal(
                    receiver=receiver_idx,
                    provider=int(provider_idx),
                    priority=float(min_standalone_snr),
                    provider_to_receiver_bytes=forward_payload,
                    receiver_to_provider_bytes=(
                        reverse_payload + reverse_training_embedding_bytes
                    ),
                )
                if proposal in proposal_to_context:
                    raise AssertionError("duplicate realistic-network proposal")
                context.proposal = proposal
                proposal_to_context[proposal] = context

        schedule = schedule_transfers(
            list(proposal_to_context),
            received_power_dbm=powers,
            noise_floor_dbm=float(self.cfg.noise_floor_dbm),
            min_sinr_db=self.network_min_sinr_db,
            resource_count=self.network_resource_count,
            bandwidth_hz=self.network_bandwidth_hz,
            direction_airtime_s=self.network_direction_airtime_s,
            efficiency=self.network_efficiency,
            max_spectral_efficiency=self.network_max_spectral_efficiency,
            missing_power_dbm=self.network_missing_power_dbm,
        )
        accepted = {row.proposal: row for row in schedule.accepted}
        rejected = {row.proposal: row.reason for row in schedule.rejected}
        accepted_links = sorted(
            {
                (
                    proposal_to_context[row.proposal].zones[row.proposal.provider],
                    min(row.proposal.receiver, row.proposal.provider),
                    max(row.proposal.receiver, row.proposal.provider),
                )
                for row in schedule.accepted
            }
        )
        if self.selection_mode == "policy":
            self._share_policies_with_all_feasible_neighbors(accepted_links)

        pending_training: list[tuple[str, int, _TrainingExample, int]] = []
        alignment_pairs: list[tuple[str, int, int]] = []
        for context in contexts:
            result = None
            proposal = context.proposal
            if proposal is not None and proposal in accepted:
                scheduled = accepted[proposal]
                provider_idx = proposal.provider
                # The encoder exchange already passed MAC/SINR scheduling.
                # Align after the decision round even when private validation
                # rejects the predictor aggregate or cannot label the pull.
                if (
                    self.align_policy_encoders
                    and self.selection_mode == "policy"
                    and not bool(
                        getattr(self, "share_training_samples", False)
                    )
                ):
                    alignment_pairs.append(
                        (
                            str(context.mode),
                            int(context.receiver_idx),
                            int(provider_idx),
                        )
                    )
                self._spend_pull_budget(
                    step=int(step),
                    mode=str(context.mode),
                    receiver_idx=int(context.receiver_idx),
                    provider_idx=int(provider_idx),
                    stream=context.token_stream,
                )
                context.decision_diag.update(
                    {
                        "network_scheduled": True,
                        "attempted": True,
                        "timing_reason": "network_scheduled",
                        "network_resource": scheduled.resource,
                        "network_forward_sinr_db": scheduled.metrics.forward_sinr_db,
                        "network_reverse_sinr_db": scheduled.metrics.reverse_sinr_db,
                        "network_forward_capacity_bytes": scheduled.metrics.forward_capacity_bytes,
                        "network_reverse_capacity_bytes": scheduled.metrics.reverse_capacity_bytes,
                    }
                )
                result = self._execute_validation_pull(
                    step=step,
                    mode=context.mode,
                    receiver=self.nodes[context.receiver_idx],
                    provider=self.nodes[provider_idx],
                    zone=context.zones[provider_idx],
                    provider_view=provider_views[context.mode][provider_idx],
                )
                policy_gain = self._policy_result_gain(result)
                context.decision_diag.update(
                    {
                        "valid": result.valid,
                        "reason": str(result.reason),
                        "realized_gain": (
                            float(policy_gain)
                            if policy_gain is not None
                            else float("nan")
                        ),
                        "adopted": result.adopted,
                    }
                )
                if (
                    self.selection_mode == "policy"
                    and result.valid
                    and policy_gain is not None
                ):
                    assert context.receiver_state is not None
                    assert context.receiver_embedding is not None
                    receiver_gain, provider_gain = (
                        self._policy_training_gains(result)
                    )
                    selected_provider_state, selected_provider_embedding = (
                        self._selected_provider_policy_observation(
                            receiver_idx=int(context.receiver_idx),
                            provider_idx=int(provider_idx),
                            mode=str(context.mode),
                            provider_view=provider_views[context.mode][provider_idx],
                            fallback_state=provider_states[context.mode][provider_idx],
                            fallback_embedding=provider_embeddings[context.mode][provider_idx],
                        )
                    )
                    examples = self._exchange_training_examples(
                        step=int(step),
                        mode=str(context.mode),
                        receiver_idx=int(context.receiver_idx),
                        provider_idx=int(provider_idx),
                        receiver_state=context.receiver_state,
                        provider_state=selected_provider_state,
                        receiver_embedding=context.receiver_embedding,
                        provider_embedding=selected_provider_embedding,
                        receiver_gain=receiver_gain,
                        provider_gain=provider_gain,
                        propensity=float(context.propensity),
                    )
                    pending_training.extend(
                        (str(context.mode), owner, example, 1)
                        for owner, example in examples
                    )
                    if self.symmetric_pulls:
                        self._cv_step_metadata_bytes[context.mode] += int(
                            self.embedding_wire_bytes
                        )
            elif proposal is not None:
                reason = rejected.get(proposal, "network_rejected")
                context.decision_diag.update(
                    {"timing_reason": f"network_{reason}", "reason": reason}
                )

            self._token_decision_rows.append(context.decision_diag)
            for provider_idx in context.candidate_ids:
                selected = result is not None and context.provider_idx == provider_idx
                self._record_encoded_decision(
                    step=step,
                    enc_id=context.enc_ids[provider_idx],
                    zone=context.zones[provider_idx],
                    receiver_idx=context.receiver_idx,
                    provider_idx=provider_idx,
                    mode=context.mode,
                    action=int(selected),
                    predicted_gain=float(context.current_scores[provider_idx]),
                    reward=(
                        float(self._policy_result_gain(result))
                        if selected
                        and self._policy_result_gain(result) is not None
                        else 0.0
                    ),
                    exploratory=bool(selected and context.exploratory),
                    alpha=result.alpha if selected else None,
                )

        if self.train_all_current_examples and pending_training:
            random.Random(
                int(self.cfg.seed) * 1_000_003 + step * 65_537
            ).shuffle(pending_training)
        self._train_pending_examples(step=step, pending_training=pending_training)
        # Consensus happens only after both endpoints made their own local
        # realized-pull update. Heads, replay, and head optimizer moments stay
        # private; encoder Adam moments are reset by align_encoder_with().
        self._align_encoder_pairs(alignment_pairs)
        active_indices = sorted(
            {
                int(node_idx)
                for node_indices in zone_nodes.values()
                for node_idx in node_indices
            }
        )
        if self.selection_mode == "policy" and self.train_accumulated_head_epoch:
            for agents in self.local_agents.values():
                for node_idx in active_indices:
                    agents[node_idx].train_head_epoch()
        elif (
            self.selection_mode == "policy"
            and self.head_replay_batches_per_step > 0
        ):
            for agents in self.local_agents.values():
                for node_idx in active_indices:
                    agents[node_idx].train_head_batches(
                        num_batches=self.head_replay_batches_per_step
                    )

        sinrs = [
            value
            for row in schedule.accepted
            for value in (
                row.metrics.forward_sinr_db,
                row.metrics.reverse_sinr_db,
            )
        ]
        capacities = [
            value
            for row in schedule.accepted
            for value in (
                row.metrics.forward_capacity_bytes,
                row.metrics.reverse_capacity_bytes,
            )
        ]
        rejection_counts = {
            reason: sum(row.reason == reason for row in schedule.rejected)
            for reason in {row.reason for row in schedule.rejected}
        }
        self._network_step_stats = {
            "network_raw_contact_pairs": len(links),
            "network_raw_directed_candidates": 2 * len(links),
            "network_shortlisted_directed_candidates": sum(
                len(rows) for rows in neighbors.values()
            ),
            "network_transfer_proposals": len(proposal_to_context),
            "network_scheduled_transfers": len(schedule.accepted),
            "network_rejected_transfers": len(schedule.rejected),
            "network_half_duplex_rejections": rejection_counts.get(
                "half_duplex_conflict", 0
            ),
            "network_sinr_or_airtime_rejections": rejection_counts.get(
                "sinr_or_airtime", 0
            ),
            "network_mean_sinr_db": float(np.mean(sinrs)) if sinrs else float("nan"),
            "network_min_sinr_db": float(np.min(sinrs)) if sinrs else float("nan"),
            "network_mean_direction_capacity_bytes": (
                float(np.mean(capacities)) if capacities else 0.0
            ),
        }

    # ---------------------------------------------------------- communication

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        assumptions.pop("serving_encoder_interval_steps", None)
        assumptions.pop("serving_encoder_ema_tau", None)
        policy_bytes = int(assumptions.get("B_policy_bytes", 0))
        agents = getattr(self, "local_agents", {})
        if agents:
            rows = next(iter(agents.values()))
            if rows:
                policy_bytes = self._state_nbytes(
                    self._clone_state(rows[0].policy)
                )
        encoder_bytes = 0
        if (
            self.align_policy_encoders
            and self.selection_mode == "policy"
            and agents
            and not bool(getattr(self, "share_training_samples", False))
        ):
            rows = next(iter(agents.values()))
            if rows:
                encoder_bytes = int(rows[0].encoder_nbytes())
        assumptions.update(
            {
                "zramp_policy_mode": (
                    "bidirectional-private-cv-exact-single-policy-sequential"
                ),
                "pretrained_policy_loaded": self.pretrained_policy_path is not None,
                "pretrained_policy_path": (
                    "" if self.pretrained_policy_path is None else str(self.pretrained_policy_path)
                ),
                "pretrained_policy_frozen_at_deployment": bool(self.freeze_pretrained_policy),
                "pretrained_policy_source_maps": ",".join(
                    getattr(self, "_pretrained_policy_metadata", {}).get("source_maps", [])
                ),
                "pretrained_policy_decisions_seen": int(
                    getattr(self, "_pretrained_policy_metadata", {}).get("decisions_seen", 0)
                ),
                "policy_copies_per_vehicle": 1,
                "policy_decision_schedule": (
                    "share-at-timestep-start;hold-fixed-for-all-sequential-"
                    "decisions;train-after-complete-decision-round"
                ),
                "state_encoder_input": (
                    "every-floating-predictor-weight-and-bias-as-learned-"
                    "neuron-rows-plus-spatially-balanced-private-samples-"
                    "plus-locally-observable-measurement-and-AZ-visit-counts"
                ),
                "state_encoder_fixed_sketch": False,
                "state_embedding_dim": int(self.embedding_dim),
                "state_embedding_quantization": "signed-int8-plus-scale",
                "validation_selection": (
                    "deterministic-max-distance-normalized-coordinate-4d-"
                    "disjoint-80-10-10"
                ),
                "validation_capacity_per_vehicle_combined": (
                    -1
                    if self.validation_capacity is None
                    else int(self.validation_capacity)
                ),
                "validation_optimization_capacity": self.validation_optimization_capacity,
                "validation_reward_capacity": self.validation_reward_capacity,
                "validation_retention": (
                    "deterministic-bounded-spatially-diverse-reservoir"
                ),
                "validation_quality": "private-validation-sample-count",
                "policy_reward_target": (
                    "normalized-delivered-feature-information-gain"
                    if self.policy_training_target == "information-gain"
                    else "parameter-geometry-maturity-novelty-gain"
                    if self.policy_training_target == "parameter-geometry"
                    else (
                        f"{self.policy_reward_scope}-bilateral-private-heldout-"
                        + (
                            "RMSE-gain"
                            if self.policy_reward_metric == "rmse-gain"
                            else "normalized-MSE-improvement"
                        )
                    )
                ),
                "policy_training_target": str(self.policy_training_target),
                "policy_reward_scope": str(self.policy_reward_scope),
                "policy_pull_rule": (
                    "bounded-bilateral-exchanges-per-contiguous-AZ-visit;"
                    "random-uniform-replay-contact;policy-learned-local-gain-"
                    "quantile-timing-and-provider-ranking"
                    if self.visit_pull_budget > 0
                    else "one-expiring-local-token-per-phase-shifted-window;"
                    "random-retries-after-deadline;policy-times-positive-pull-"
                    "with-window-end-fallback"
                    if self.contact_aware_window_timing
                    else "one-expiring-local-token-per-phase-shifted-window;"
                    "policy-times-positive-pull;random-has-independent-time"
                ),
                "token_window_steps": int(self.token_window_steps),
                "visit_pull_budget_per_physical_vehicle": int(
                    self.visit_pull_budget
                ),
                "visit_budget_consumption": (
                    "one-token-at-each-endpoint-per-scheduled-bilateral-"
                    "predictor-exchange"
                    if self.visit_pull_budget > 0 and self.symmetric_pulls
                    else "receiver-token-only"
                ),
                "random_visit_timing": (
                    "uniform-without-replacement-over-feasible-replay-contact-"
                    "frames;retry-after-MAC-rejection"
                    if self.visit_pull_budget > 0
                    else "token-window"
                ),
                "random_token_timing": (
                    "seeded-uniform-offset-in-each-local-S-step-window;"
                    "next-feasible-contact-after-offset;window-end-fallback"
                    if self.contact_aware_window_timing
                    else "seeded-uniform-offset-in-each-local-S-step-window"
                ),
                "random_provider_rule": (
                    "uniform-over-all-currently-feasible-providers"
                ),
                "policy_trigger_quantile": float(
                    self.policy_trigger_quantile
                ),
                "policy_fixed_trigger_db": (
                    None
                    if self.policy_fixed_trigger_db is None
                    else float(self.policy_fixed_trigger_db)
                ),
                "allow_unused_policy_tokens": bool(
                    self.allow_unused_policy_tokens
                ),
                "contact_aware_window_timing": bool(
                    self.contact_aware_window_timing
                ),
                "policy_warmup_steps": int(self.policy_warmup_steps),
                "policy_warmup_pull_probability": float(
                    self.policy_warmup_pull_probability
                ),
                "token_rejected_pull_consumes_slot": True,
                "policy_selection_mode": str(self.selection_mode),
                "policy_pull_budget_per_receiver_timestep": float(
                    self.pull_budget
                ),
                "policy_exploration_probability": float(
                    self.exploration_probability
                ),
                "policy_training_correction": (
                    "inverse-action-propensity-per-valid-pull"
                    if self.inverse_propensity_weighting
                    else "none"
                ),
                "policy_training_examples": (
                    "all-valid-pulls-once-per-timestep"
                    if self.train_all_current_examples
                    else "one-uniform-valid-pull-per-receiver-timestep"
                ),
                "policy_accumulated_head_epoch": bool(
                    self.train_accumulated_head_epoch
                ),
                "policy_head_replay_batches_per_step": int(
                    self.head_replay_batches_per_step
                ),
                "policy_replay_batching": (
                    "balanced-across-early-intermediate-mature-event-time"
                ),
                "policy_reward_fixed_scale_db": (
                    -1.0
                    if self.policy_reward_scale_db is None
                    else float(self.policy_reward_scale_db)
                ),
                "policy_ranking_loss_weight": float(
                    self.policy_ranking_loss_weight
                ),
                "policy_ranking_margin_db": float(
                    self.policy_ranking_margin_db
                ),
                "policy_ranking_temperature_db": float(
                    self.policy_ranking_temperature_db
                ),
                "policy_ranking_receiver_cosine_min": float(
                    self.policy_ranking_receiver_cosine_min
                ),
                "policy_training_objective": (
                    "smooth-l1-gain-plus-conditional-pairwise-provider-ranking"
                    if self.policy_ranking_loss_weight > 0.0
                    else "smooth-l1-gain"
                ),
                "bilateral_adoption_rule": (
                    "provenance-evidence-union;both-endpoints-install-"
                    "unconditionally"
                    if self.unconditional_evidence_union
                    and self.symmetric_pulls
                    else "provenance-evidence-union;receiver-installs-"
                    "unconditionally"
                    if self.unconditional_evidence_union
                    else
                    "common-alpha-optimized-on-both-private-optimization-sets;"
                    "each-endpoint-installs-iff-own-private-heldout-gain-positive"
                    if self.symmetric_pulls
                    else "receiver-installs-on-positive-joint-private-gain"
                ),
                "policy_gain_hidden_dim": int(self.gain_hidden_dim),
                "mergeable_max_delta_rows_per_direction": int(
                    self.mergeable_max_delta_rows
                ),
                "policy_pair_feature_mode": str(self.pair_feature_mode),
                "policy_gain_head_initialization": "pytorch-linear-default",
                "policy_action_selection": (
                    "epsilon-greedy-argmax-over-positive-predicted-gain;"
                    "explore-among-all-feasible-providers"
                ),
                "central_training_samples": (
                    "all-accumulated-decodable-directed-link-measurements"
                    if bool(getattr(self, "central_accumulate_samples", False))
                    else "current-step-decodable-directed-link-measurements"
                ),
                "central_accumulate_samples": bool(
                    getattr(self, "central_accumulate_samples", False)
                ),
                "policy_max_pulls_per_receiver_timestep": (
                    1
                    if self.pull_budget < 1.0
                    else int(self.pull_budget)
                ),
                "predictor_time_encoding": "learned-scalar-MLP",
                "predictor_time_embedding_dim": int(self.learned_time_dim),
                "predictor_time_scale": float(self.learned_time_scale),
                "B_policy_bytes": int(policy_bytes),
                "B_policy_model_bytes_per_directed_contact": int(policy_bytes),
                "policy_params": int(policy_bytes // 4),
                "policy_execution_enabled": (
                    self.selection_mode == "policy" and not bool(self.aux_only)
                ),
                "B_embedding_gradient_bytes_per_trained_pull": 0,
                "policy_sample_capacity_per_vehicle": int(
                    self.policy_sample_capacity
                ),
                "policy_sample_bundle_capacity_per_direction": int(
                    self.policy_sample_bundle_capacity
                ),
                "policy_encoder_learning_rate_scale": float(
                    self.encoder_lr_scale
                ),
                "policy_encoder_alignment_enabled": bool(
                    self.align_policy_encoders
                ),
                "policy_encoder_alignment_rule": (
                    "experience-weighted-consensus-after-every-scheduled-"
                    + (
                        "sample-gossip-exchange-before-provider-selection;"
                        if bool(
                            getattr(self, "share_training_samples", False)
                        )
                        else "decoded-bilateral-predictor-exchange;"
                    )
                    + "zero-experience-newcomer-copies-experienced-peer"
                    if self.align_policy_encoders
                    else "none"
                ),
                "trajectory_capacity_per_vehicle": int(
                    self.trajectory_capacity
                ),
                "metadata_note": (
                    "No public probes, fixed model sketches, trajectory "
                    "sketches, Fourier time basis, or hand-designed utility "
                    "features are used. Providers transmit only quantized "
                    "learned embeddings. Exact predictor states and complete "
                    "ordered model-training trajectories are processed "
                    "locally; validation holdouts never enter the encoder. "
                    "Each vehicle trains its private policy after the timestep "
                    "decision round. No embedding gradients are transmitted."
                ),
            }
        )
        if self.realistic_network:
            assumptions.update(
                {
                    "zramp_policy_mode": (
                        "bidirectional-private-cv-exact-realistic-v2v"
                    ),
                    "policy_transfer_rule": (
                        "no-gain-head-transfer;experience-weighted-encoder-"
                        "consensus-on-every-scheduled-bilateral-exchange"
                    ),
                    "policy_decision_schedule": (
                        "local-policy-proposal;global-half-duplex-resource-"
                        "schedule;exchange-predictors;validate;train-local;"
                        "align-encoders-independent-of-adoption"
                    ),
                    "B_policy_model_bytes_per_directed_contact": 0,
                    "B_policy_messages_per_directed_contact": 0,
                    "network_realism": True,
                    "network_decentralized_reservation": bool(
                        self.network_decentralized_reservation
                    ),
                    "network_reservation_control_bytes_per_pair": int(
                        self.network_reservation_control_bytes
                    ),
                    "network_reservation_rule": (
                        "deterministic-local-backoff-request-grant-with-"
                        "next-ranked-provider-fallback"
                        if self.network_decentralized_reservation
                        else "none"
                    ),
                    "network_candidate_rule": (
                        (
                            f"top-{int(self.network_candidate_top_k)}-strongest-"
                            "bidirectional-sionna-links-per-receiver"
                        )
                        if self.network_candidate_top_k > 0
                        else "all-bidirectionally-decodable-sionna-links"
                    ),
                    "network_resource_count": int(self.network_resource_count),
                    "network_half_duplex": True,
                    "network_bandwidth_hz_per_resource": float(
                        self.network_bandwidth_hz
                    ),
                    "network_direction_airtime_s_per_resource": float(
                        self.network_direction_airtime_s
                    ),
                    "network_total_reserved_airtime_s_per_step": float(
                        2.0
                        * self.network_resource_count
                        * self.network_direction_airtime_s
                    ),
                    "network_capacity_model": (
                        "efficiency-times-bandwidth-times-airtime-times-"
                        "min(log2(1+sinr),spectral-efficiency-cap)"
                    ),
                    "network_efficiency": float(self.network_efficiency),
                    "network_max_spectral_efficiency_bps_hz": float(
                        self.network_max_spectral_efficiency
                    ),
                    "network_min_sinr_db": float(self.network_min_sinr_db),
                    "network_interference_model": (
                        "exact-traced-cross-link-powers-for-simultaneous-"
                        "providers-forward-and-receivers-reverse"
                    ),
                    "network_direction_payload_bytes": (
                        "pair-specific-provenance-summary-plus-newer-evidence-"
                        "rows-plus-control"
                    ),
                    "network_mac_rejection_consumes_token": False,
                    "token_rejected_pull_consumes_slot": (
                        "only-after-network-scheduling"
                    ),
                    "metadata_note": (
                        "Providers advertise quantized embeddings only to the "
                        "shortlist. Mergeable predictors exchange provenance "
                        "summaries and capacity-bounded newer rows; other "
                        "predictors and optional encoder parameters use a scheduled "
                        "bidirectional link; private gain heads never move. "
                        "Private trajectories and validation holdouts never "
                        "leave the vehicle."
                    ),
                }
            )
        return assumptions

    # ---------------------------------------------------------------- logging

    def _ensure_encoded_log(self) -> csv.DictWriter:
        if self._encoded_log_writer is None:
            self._encoded_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._encoded_log_file = open(
                self._encoded_log_path, "w", newline="", encoding="utf-8"
            )
            self._encoded_log_writer = csv.DictWriter(
                self._encoded_log_file,
                fieldnames=(
                    "step",
                    "mode",
                    "receiver_idx",
                    "provider_idx",
                    "target_gain",
                    "online_prediction",
                    "base_loss",
                    "weighted_loss",
                    "propensity",
                    "importance_weight",
                    "receiver_policy_version",
                    "provider_policy_version",
                ),
            )
            self._encoded_log_writer.writeheader()
        return self._encoded_log_writer

    def _token_learning_summary(self) -> dict[str, object]:
        attempted = [
            row for row in self._token_decision_rows if bool(row.get("attempted"))
        ]
        paired = [
            row
            for row in attempted
            if np.isfinite(
                float(row.get("selected_predicted_gain", float("nan")))
            )
            and np.isfinite(float(row.get("realized_gain", float("nan"))))
        ]
        reasons = sorted(
            {
                str(row.get("timing_reason", ""))
                for row in self._token_decision_rows
            }
        )
        summary: dict[str, object] = {
            "selection_mode": self.selection_mode,
            "token_window_steps": int(self.token_window_steps),
            "decision_opportunities": len(self._token_decision_rows),
            "attempts": len(attempted),
            "timing_reason_counts": {
                reason: sum(
                    str(row.get("timing_reason", "")) == reason
                    for row in self._token_decision_rows
                )
                for reason in reasons
            },
        }
        if paired:
            predictions = np.asarray(
                [float(row["selected_predicted_gain"]) for row in paired]
            )
            targets = np.asarray(
                [float(row["realized_gain"]) for row in paired]
            )
            error = predictions - targets
            summary.update(
                {
                    "prediction_target_pairs": len(paired),
                    "gain_prediction_mae": float(np.mean(np.abs(error))),
                    "gain_prediction_rmse": float(np.sqrt(np.mean(error**2))),
                    "gain_sign_accuracy": float(
                        np.mean((predictions > 0.0) == (targets > 0.0))
                    ),
                    "gain_prediction_pearson": (
                        float(np.corrcoef(predictions, targets)[0, 1])
                        if len(paired) > 1
                        and float(np.std(predictions)) > 0.0
                        and float(np.std(targets)) > 0.0
                        else float("nan")
                    ),
                    "selected_predicted_top_fraction": float(
                        np.mean(
                            [
                                int(row.get("selected_predicted_rank", 0)) == 1
                                for row in paired
                            ]
                        )
                    ),
                }
            )
        oracle_headroom = [
            float(row["oracle_provider_headroom"])
            for row in self._token_decision_rows
            if np.isfinite(
                float(row.get("oracle_provider_headroom", float("nan")))
            )
        ]
        if oracle_headroom:
            summary["oracle_provider_headroom_mean"] = float(
                np.mean(oracle_headroom)
            )
            summary["oracle_provider_headroom_p90"] = float(
                np.percentile(oracle_headroom, 90)
            )
        return summary

    @staticmethod
    def _decision_metric_block(
        rows: list[dict[str, int | float | str | bool]],
    ) -> dict[str, float | int]:
        attempted = [row for row in rows if bool(row.get("attempted"))]
        labeled = [
            row
            for row in attempted
            if np.isfinite(float(row.get("realized_gain", float("nan"))))
        ]
        predictions = np.asarray(
            [float(row.get("selected_predicted_gain", float("nan"))) for row in labeled],
            dtype=np.float64,
        )
        gains = np.asarray(
            [float(row["realized_gain"]) for row in labeled],
            dtype=np.float64,
        )
        score_stds = np.asarray(
            [
                float(row.get("predicted_gain_std", float("nan")))
                for row in rows
                if np.isfinite(
                    float(row.get("predicted_gain_std", float("nan")))
                )
            ],
            dtype=np.float64,
        )
        correlation = float("nan")
        if (
            len(labeled) > 1
            and float(np.std(predictions)) > 0.0
            and float(np.std(gains)) > 0.0
        ):
            correlation = float(np.corrcoef(predictions, gains)[0, 1])
        candidate_regular = np.asarray(
            [float(row.get("candidate_regular_fraction", float("nan"))) for row in rows],
            dtype=np.float64,
        )
        candidate_regular = candidate_regular[np.isfinite(candidate_regular)]
        selected_regular = np.asarray(
            [float(bool(row.get("selected_provider_regular"))) for row in attempted],
            dtype=np.float64,
        )
        candidate_more = np.asarray(
            [float(row.get("candidate_more_samples_fraction", float("nan"))) for row in rows],
            dtype=np.float64,
        )
        candidate_more = candidate_more[np.isfinite(candidate_more)]
        selected_more = np.asarray(
            [float(bool(row.get("selected_provider_more_samples"))) for row in attempted],
            dtype=np.float64,
        )
        sample_advantages = np.asarray(
            [float(row.get("selected_provider_sample_advantage", float("nan"))) for row in attempted],
            dtype=np.float64,
        )
        sample_advantages = sample_advantages[np.isfinite(sample_advantages)]
        available_regular = (
            float(np.mean(candidate_regular))
            if len(candidate_regular)
            else float("nan")
        )
        available_more = (
            float(np.mean(candidate_more))
            if len(candidate_more)
            else float("nan")
        )
        selected_regular_fraction = (
            float(np.mean(selected_regular))
            if len(selected_regular)
            else float("nan")
        )
        selected_more_fraction = (
            float(np.mean(selected_more))
            if len(selected_more)
            else float("nan")
        )
        return {
            "opportunities": int(len(rows)),
            "attempts": int(len(attempted)),
            "labeled_pulls": int(len(labeled)),
            "attempt_rate": (
                float(len(attempted) / len(rows)) if rows else float("nan")
            ),
            "mean_realized_gain": (
                float(np.mean(gains)) if len(gains) else float("nan")
            ),
            "beneficial_pull_fraction": (
                float(np.mean(gains > 0.0)) if len(gains) else float("nan")
            ),
            "adopted_pull_fraction": (
                float(np.mean([bool(row.get("adopted")) for row in labeled]))
                if labeled
                else float("nan")
            ),
            "gain_prediction_pearson": correlation,
            "selected_predicted_top_fraction": (
                float(
                    np.mean(
                        [
                            int(row.get("selected_predicted_rank", 0)) == 1
                            for row in labeled
                        ]
                    )
                )
                if labeled
                else float("nan")
            ),
            "mean_candidate_score_std": (
                float(np.mean(score_stds))
                if len(score_stds)
                else float("nan")
            ),
            "available_regular_provider_fraction": available_regular,
            "selected_regular_provider_fraction": selected_regular_fraction,
            "regular_provider_selection_enrichment": (
                selected_regular_fraction / available_regular
                if np.isfinite(selected_regular_fraction)
                and np.isfinite(available_regular)
                and available_regular > 0.0
                else float("nan")
            ),
            "available_more_experienced_provider_fraction": available_more,
            "selected_more_experienced_provider_fraction": selected_more_fraction,
            "more_experienced_selection_enrichment": (
                selected_more_fraction / available_more
                if np.isfinite(selected_more_fraction)
                and np.isfinite(available_more)
                and available_more > 0.0
                else float("nan")
            ),
            "selected_provider_sample_advantage_mean": (
                float(np.mean(sample_advantages))
                if len(sample_advantages)
                else float("nan")
            ),
        }

    @staticmethod
    def _mean_for_nodes(values: list[int], indices: list[int]) -> float:
        selected = [float(values[index]) for index in indices]
        return float(np.mean(selected)) if selected else float("nan")

    def _live_diagnostic_row(
        self, *, step: int, window_steps: int = 100
    ) -> dict[str, int | float | str]:
        current_step = int(step)
        start_step = max(1, current_step - int(window_steps) + 1)
        window_rows = [
            row
            for row in self._token_decision_rows
            if start_step <= int(row.get("step", -1)) <= current_step
        ]
        window = self._decision_metric_block(window_rows)
        cumulative = self._decision_metric_block(self._token_decision_rows)

        retained = [
            len(state.optimization.features) + len(state.reward.features)
            for state in self._zone_validation
        ]
        validation_seen = [
            int(state.optimization.samples_seen + state.reward.samples_seen)
            for state in self._zone_validation
        ]
        regular_count = min(self.diagnostic_regular_count, len(retained))
        regular = list(range(regular_count))
        visitors = list(range(regular_count, len(retained)))

        examples: list[int] = []
        if self.local_agents:
            mode_agents = next(iter(self.local_agents.values()))
            if bool(getattr(self, "share_training_samples", False)):
                examples = [
                    len(agent.head_replay_targets) for agent in mode_agents
                ]
            else:
                examples = [
                    int(getattr(agent, "experience", 0))
                    for agent in mode_agents
                ]

        latest_fidelity_step = -1
        latest_tail_rmse = float("nan")
        if self.fidelity_history:
            fidelity = self.fidelity_history[-1]
            mode_id = next(iter(self.agents), "")
            latest_fidelity_step = int(fidelity.get("step", -1))
            latest_tail_rmse = float(
                fidelity.get(f"{mode_id}_total", float("nan"))
            )

        row: dict[str, int | float | str] = {
            "step": current_step,
            "selection_mode": str(self.selection_mode),
            "window_steps": int(window_steps),
            "validation_capacity": (
                -1
                if self.validation_capacity is None
                else int(self.validation_capacity)
            ),
            "regular_vehicle_count": regular_count,
            "visitor_vehicle_count": len(visitors),
            "regular_validation_retained_mean": self._mean_for_nodes(
                retained, regular
            ),
            "visitor_validation_retained_mean": self._mean_for_nodes(
                retained, visitors
            ),
            "regular_validation_seen_mean": self._mean_for_nodes(
                validation_seen, regular
            ),
            "visitor_validation_seen_mean": self._mean_for_nodes(
                validation_seen, visitors
            ),
            "policy_examples_per_vehicle_mean": (
                float(np.mean(examples)) if examples else float("nan")
            ),
            "policy_examples_per_vehicle_min": (
                int(min(examples)) if examples else 0
            ),
            "policy_examples_per_vehicle_max": (
                int(max(examples)) if examples else 0
            ),
            "latest_fidelity_step": latest_fidelity_step,
            "latest_tail_rmse": latest_tail_rmse,
        }
        row.update({f"window_{key}": value for key, value in window.items()})
        row.update(
            {f"cumulative_{key}": value for key, value in cumulative.items()}
        )
        return row

    @staticmethod
    def _format_live_metric(value: object) -> str:
        number = float(value)
        return "na" if not np.isfinite(number) else f"{number:.3f}"

    def _write_live_diagnostics(self, *, step: int) -> None:
        row = self._live_diagnostic_row(step=int(step))
        if self._live_diagnostic_rows and int(self._live_diagnostic_rows[-1]["step"]) == int(step):
            self._live_diagnostic_rows[-1] = row
        else:
            self._live_diagnostic_rows.append(row)
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        fields = list(self._live_diagnostic_rows[0].keys())
        with open(
            output / "live_diagnostics.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._live_diagnostic_rows)
        with open(
            output / "live_diagnostics.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(row, handle, indent=2, sort_keys=True)

        fmt = self._format_live_metric
        print(
            "[EXACT-LIVE] "
            f"method={self.selection_mode} step={int(step)} "
            f"attempts={int(row['window_attempts'])}/"
            f"{int(row['window_opportunities'])} "
            f"gain={fmt(row['window_mean_realized_gain'])} "
            f"beneficial={fmt(row['window_beneficial_pull_fraction'])} "
            f"adopted={fmt(row['window_adopted_pull_fraction'])} "
            f"pred_corr={fmt(row['window_gain_prediction_pearson'])} "
            f"score_std="
            f"{float(row['window_mean_candidate_score_std']):.3e} "
            f"regular_enrich="
            f"{fmt(row['window_regular_provider_selection_enrichment'])} "
            f"experience_enrich="
            f"{fmt(row['window_more_experienced_selection_enrichment'])} "
            f"examples_per_vehicle={fmt(row['policy_examples_per_vehicle_mean'])} "
            f"validation_regular/visitor="
            f"{fmt(row['regular_validation_retained_mean'])}/"
            f"{fmt(row['visitor_validation_retained_mean'])} "
            f"tail_rmse={fmt(row['latest_tail_rmse'])}",
            flush=True,
        )

    def _write_token_diagnostics(self) -> None:
        if self._token_decision_rows:
            fields = [
                "step",
                *sorted(
                    {
                        key
                        for row in self._token_decision_rows
                        for key in row
                        if key != "step"
                    }
                ),
            ]
            with open(
                Path(self.cfg.results_dir) / "decision_diagnostics.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._token_decision_rows)
        with open(
            Path(self.cfg.results_dir) / "learning_summary.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self._token_learning_summary(),
                handle,
                indent=2,
                sort_keys=True,
            )

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._write_token_diagnostics()
        self._write_live_diagnostics(
            step=int(
                kwargs.get("step", getattr(self, "_current_sumo_step", 0))
            )
        )
class SampleSharingExactSequentialSimulation(
    ExactSequentialBidirectionalSimulation
):
    """No policy transfer; gossip embedding-pair/gain training samples only."""

    policy_transfer_rule = "no-policy-transfer-sample-gossip"

    def __init__(self, *args, **kwargs) -> None:
        self.share_training_samples = True
        super().__init__(*args, **kwargs)
        self.share_policy_every_contact = False
        self.local_policy_share = False
        self.zramp_policy_mode = (
            "bidirectional-private-cv-local-encoder-sample-gossip"
        )
        self._communication_assumptions = self._build_communication_assumptions()

    def _reset_respawned_node(
        self, node_idx: int, *, generation: int | None = None
    ) -> None:
        super()._reset_respawned_node(
            node_idx, generation=generation
        )

    def _share_policies_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> None:
        del links

    def _training_sample_wire_bytes(self) -> int:
        return 2 * int(self.embedding_wire_bytes) + 4 + 16

    def _ordered_snapshot_samples(
        self,
        samples: dict[
            object, tuple[torch.Tensor, torch.Tensor, float]
        ],
        steps: dict[object, int],
        limit: int,
    ) -> list[
        tuple[
            object,
            tuple[torch.Tensor, torch.Tensor, float],
            int,
        ]
    ]:
        del limit
        ordered = sorted(
            samples.items(),
            key=lambda item: (
                OnlineExactUtilityAgent.sample_priority(item[0]),
                repr(item[0]),
            ),
        )
        return [
            (sample_id, sample, int(steps.get(sample_id, 0)))
            for sample_id, sample in ordered
        ]

    @staticmethod
    def _would_retain_transferred_sample(
        agent: object, sample_id: object, sample_step: int
    ) -> bool:
        retain = getattr(agent, "would_retain_sample", None)
        if not callable(retain):
            return True
        try:
            return bool(retain(sample_id, sample_step=int(sample_step)))
        except TypeError:
            return bool(retain(sample_id))

    @staticmethod
    def _remember_transferred_sample(
        agent: object,
        sample_id: object,
        sample: tuple[torch.Tensor, torch.Tensor, float],
        sample_step: int,
    ) -> bool:
        remember = getattr(agent, "remember_shared_sample")
        try:
            return bool(
                remember(
                    sample_id,
                    *sample,
                    sample_step=int(sample_step),
                )
            )
        except TypeError:
            return bool(remember(sample_id, *sample))

    def _share_samples_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> dict[str, int]:
        if not links:
            return {}
        if self.realistic_network:
            return self._share_samples_with_realistic_network(links)

        neighbors: dict[int, set[int]] = defaultdict(set)
        for _zone, first, second in links:
            neighbors[int(first)].add(int(second))
            neighbors[int(second)].add(int(first))
        sample_wire_bytes = self._training_sample_wire_bytes()
        for mode, agents in self.local_agents.items():
            snapshots = [dict(agent.shared_samples) for agent in agents]
            snapshot_steps = [
                dict(getattr(agent, "shared_sample_steps", {}))
                for agent in agents
            ]
            for receiver_idx, provider_ids in sorted(neighbors.items()):
                receiver = agents[receiver_idx]
                for provider_idx in sorted(provider_ids):
                    installed = 0
                    ordered = self._ordered_snapshot_samples(
                        snapshots[provider_idx],
                        snapshot_steps[provider_idx],
                        int(
                            getattr(
                                self,
                                "policy_sample_bundle_capacity",
                                32,
                            )
                        ),
                    )
                    for sample_id, sample, sample_step in ordered:
                        if not self._would_retain_transferred_sample(
                            receiver, sample_id, sample_step
                        ):
                            continue
                        if self._remember_transferred_sample(
                            receiver, sample_id, sample, sample_step
                        ):
                            installed += 1
                            self._cv_step_sample_bytes[
                                mode
                            ] += sample_wire_bytes
                            self._cv_step_sample_messages[mode] += 1
                            if (
                                installed
                                >= int(
                                    getattr(
                                        self,
                                        "policy_sample_bundle_capacity",
                                        32,
                                    )
                                )
                            ):
                                break
        if self.align_policy_encoders and self.selection_mode == "policy":
            self._align_encoder_pairs(
                [
                    (str(mode), int(first), int(second))
                    for mode in self.local_agents
                    for _zone, first, second in links
                ]
            )
        return {}

    def _share_samples_with_realistic_network(
        self, links: list[tuple[int, int, int]]
    ) -> dict[str, int]:
        """Deliver causal sample bundles in a separate MAC/SINR phase."""

        sample_wire_bytes = self._training_sample_wire_bytes()
        min_spectral_efficiency = min(
            math.log2(1.0 + 10.0 ** (self.network_min_sinr_db / 10.0)),
            self.network_max_spectral_efficiency,
        )
        bundle_capacity_bytes = int(
            math.floor(
                self.network_bandwidth_hz
                * self.network_direction_airtime_s
                * self.network_efficiency
                * min_spectral_efficiency
                / 8.0
            )
        )
        encoder_bytes = 0
        if self.align_policy_encoders and self.selection_mode == "policy":
            rows = next(iter(self.local_agents.values()), [])
            if rows:
                encoder_bytes = int(rows[0].encoder_nbytes())
        if encoder_bytes > bundle_capacity_bytes:
            return {
                "sample_network_raw_contact_pairs": int(len(links)),
                "sample_network_transfer_proposals": 0,
                "sample_network_scheduled_transfers": 0,
                "sample_network_delivered_samples": 0,
                "sample_network_encoder_capacity_rejected": int(len(links)),
            }
        max_samples = min(
            int(getattr(self, "policy_sample_bundle_capacity", 32)),
            max(0, bundle_capacity_bytes - encoder_bytes)
            // max(1, sample_wire_bytes),
        )
        if max_samples <= 0 and encoder_bytes <= 0:
            return {
                "sample_network_raw_contact_pairs": int(len(links)),
                "sample_network_transfer_proposals": 0,
                "sample_network_scheduled_transfers": 0,
                "sample_network_delivered_samples": 0,
            }

        powers = {
            (int(tx), int(rx)): float(value)
            for (tx, rx), value in dict(
                getattr(self, "_current_link_rssi_dbm", {})
            ).items()
        }
        proposed = 0
        scheduled_count = 0
        delivered = 0
        for mode, agents in self.local_agents.items():
            # One-hop-per-step propagation: no sample received in this phase
            # can be forwarded until a later simulation step.
            snapshots = [dict(agent.shared_samples) for agent in agents]
            snapshot_steps = [
                dict(getattr(agent, "shared_sample_steps", {}))
                for agent in agents
            ]
            proposals: list[TransferProposal] = []
            payloads: dict[
                TransferProposal,
                tuple[
                    int,
                    int,
                    list[
                        tuple[
                            object,
                            tuple[torch.Tensor, torch.Tensor, float],
                            int,
                        ]
                    ],
                    list[
                        tuple[
                            object,
                            tuple[torch.Tensor, torch.Tensor, float],
                            int,
                        ]
                    ],
                ],
            ] = {}
            for _zone, first, second in links:
                receiver_idx, provider_idx = sorted(
                    (int(first), int(second))
                )
                to_receiver: list[
                    tuple[
                        object,
                        tuple[torch.Tensor, torch.Tensor, float],
                        int,
                    ]
                ] = []
                ordered_provider = self._ordered_snapshot_samples(
                    snapshots[provider_idx],
                    snapshot_steps[provider_idx],
                    max_samples,
                )
                for sample_id, sample, sample_step in ordered_provider:
                    if not self._would_retain_transferred_sample(
                        agents[receiver_idx], sample_id, sample_step
                    ):
                        continue
                    to_receiver.append((sample_id, sample, sample_step))
                    if len(to_receiver) >= max_samples:
                        break
                to_provider: list[
                    tuple[
                        object,
                        tuple[torch.Tensor, torch.Tensor, float],
                        int,
                    ]
                ] = []
                ordered_receiver = self._ordered_snapshot_samples(
                    snapshots[receiver_idx],
                    snapshot_steps[receiver_idx],
                    max_samples,
                )
                for sample_id, sample, sample_step in ordered_receiver:
                    if not self._would_retain_transferred_sample(
                        agents[provider_idx], sample_id, sample_step
                    ):
                        continue
                    to_provider.append((sample_id, sample, sample_step))
                    if len(to_provider) >= max_samples:
                        break
                if not to_receiver and not to_provider and encoder_bytes <= 0:
                    continue
                forward_rssi = float(
                    powers.get(
                        (provider_idx, receiver_idx),
                        self.network_missing_power_dbm,
                    )
                )
                reverse_rssi = float(
                    powers.get(
                        (receiver_idx, provider_idx),
                        self.network_missing_power_dbm,
                    )
                )
                proposal = TransferProposal(
                    receiver=receiver_idx,
                    provider=provider_idx,
                    priority=min(forward_rssi, reverse_rssi),
                    provider_to_receiver_bytes=(
                        encoder_bytes + len(to_receiver) * sample_wire_bytes
                    ),
                    receiver_to_provider_bytes=(
                        encoder_bytes + len(to_provider) * sample_wire_bytes
                    ),
                )
                proposals.append(proposal)
                payloads[proposal] = (
                    receiver_idx,
                    provider_idx,
                    to_receiver,
                    to_provider,
                )

            schedule = schedule_transfers(
                proposals,
                received_power_dbm=powers,
                noise_floor_dbm=float(self.cfg.noise_floor_dbm),
                min_sinr_db=self.network_min_sinr_db,
                resource_count=self.network_resource_count,
                bandwidth_hz=self.network_bandwidth_hz,
                direction_airtime_s=self.network_direction_airtime_s,
                efficiency=self.network_efficiency,
                max_spectral_efficiency=self.network_max_spectral_efficiency,
                missing_power_dbm=self.network_missing_power_dbm,
            )
            proposed += len(proposals)
            scheduled_count += len(schedule.accepted)
            alignment_pairs: list[tuple[str, int, int]] = []
            for row in schedule.accepted:
                (
                    receiver_idx,
                    provider_idx,
                    to_receiver,
                    to_provider,
                ) = payloads[row.proposal]
                if encoder_bytes > 0:
                    alignment_pairs.append(
                        (str(mode), int(receiver_idx), int(provider_idx))
                    )
                for sample_id, sample, sample_step in to_receiver:
                    if self._remember_transferred_sample(
                        agents[receiver_idx],
                        sample_id,
                        sample,
                        sample_step,
                    ):
                        delivered += 1
                        self._cv_step_sample_bytes[mode] += sample_wire_bytes
                        self._cv_step_sample_messages[mode] += 1
                for sample_id, sample, sample_step in to_provider:
                    if self._remember_transferred_sample(
                        agents[provider_idx],
                        sample_id,
                        sample,
                        sample_step,
                    ):
                        delivered += 1
                        self._cv_step_sample_bytes[mode] += sample_wire_bytes
                        self._cv_step_sample_messages[mode] += 1
            self._align_encoder_pairs(alignment_pairs)

        return {
            "sample_network_raw_contact_pairs": int(len(links)),
            "sample_network_transfer_proposals": int(proposed),
            "sample_network_scheduled_transfers": int(scheduled_count),
            "sample_network_delivered_samples": int(delivered),
            "sample_network_bundle_capacity_bytes": int(
                bundle_capacity_bytes
            ),
            "sample_network_max_samples_per_direction": int(max_samples),
            "sample_network_encoder_bytes_per_direction": int(
                encoder_bytes
            ),
        }

    def _train_exact_pair(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        example: _TrainingExample,
        sample_multiplier: int,
    ) -> None:
        receiver_agent = self.local_agents[mode][int(receiver_idx)]
        receiver_agent.remember_shared_sample(
            example.sample_id,
            example.receiver_embedding,
            example.provider_embedding,
            float(example.target_gain),
            sample_step=int(step),
        )
        # Only this endpoint's raw state participates in autograd. The peer
        # embedding is detached, so no private trajectory or gradient crosses
        # the link.
        super()._train_exact_pair(
            step=step,
            mode=mode,
            receiver_idx=receiver_idx,
            example=example,
            sample_multiplier=sample_multiplier,
        )

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        if self.selection_mode == "random":
            super()._gossip_step(
                step, zone_nodes, contact_links=contact_links
            )
            return
        links = self._normalized_contact_links(zone_nodes, contact_links)
        sample_network_stats = (
            self._share_samples_with_all_feasible_neighbors(links)
        )
        super()._gossip_step(step, zone_nodes, contact_links=links)
        if sample_network_stats:
            self._network_step_stats.update(sample_network_stats)

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        assumptions.update(
            {
                "zramp_policy_mode": (
                    "bidirectional-private-cv-local-encoder-sample-gossip"
                ),
                "local_policy_share": False,
                "policy_transfer_rule": self.policy_transfer_rule,
                "policy_training_sample_sharing": True,
                "policy_encoder_frozen": bool(
                    self.freeze_policy_encoders
                ),
                "policy_encoder_alignment_enabled": bool(
                    self.align_policy_encoders
                ),
                "policy_reward_normalization": bool(
                    self.normalize_policy_rewards
                ),
                "policy_reward_fixed_scale_db": (
                    -1.0
                    if self.policy_reward_scale_db is None
                    else float(self.policy_reward_scale_db)
                ),
                "policy_min_samples_before_ranking": int(
                    self.policy_min_samples
                ),
                "policy_exploration_start": float(
                    self.policy_exploration_start
                ),
                "policy_exploration_end": float(
                    self.exploration_probability
                ),
                "policy_exploration_decay_samples": int(
                    self.policy_exploration_decay_samples
                ),
                "policy_encoder_alignment_payload_bytes_per_direction": (
                    int(
                        next(iter(self.local_agents.values()))[
                            0
                        ].encoder_nbytes()
                    )
                    if self.align_policy_encoders and self.local_agents
                    else 0
                ),
                "policy_decision_schedule": (
                    "share-one-hop-causal-embedding-gain-sample-bundles-at-"
                    "timestep-start;apply-MAC-SINR-airtime-when-realistic;"
                    "experience-weighted-encoder-consensus-on-every-scheduled-"
                    "sample-exchange;decide;train-local-encoder-and-gain-head"
                ),
                "policy_encoder_exchange_phase": "sample-gossip-before-decision",
                "B_policy_model_bytes_per_directed_contact": 0,
                "B_policy_messages_per_directed_contact": 0,
                "B_embedding_gradient_bytes_per_trained_pull": 0,
                "B_training_sample_bytes": int(
                    2 * self.embedding_wire_bytes + 4 + 16
                ),
                "training_sample_payload": (
                    "two-signed-int8-embeddings-with-scales-plus-float32-"
                    "gain-and-16-byte-id"
                ),
                "training_sample_network_enforcement": (
                    "separate-half-duplex-MAC-SINR-airtime-phase"
                    if self.realistic_network
                    else "unrestricted-contact-gossip"
                ),
                "sample_gossip_direction_airtime_s": float(
                    self.network_direction_airtime_s
                ),
                "metadata_note": (
                    "Gain-head parameters, optimizer state, raw measurements, "
                    "private trajectories, and embedding gradients never "
                    "leave a vehicle. All seeded encoders are frozen and "
                    "identical when policy_encoder_frozen is true, so every "
                    "stored or gossiped embedding remains compatible and no "
                    "encoder payload is transmitted. Vehicles gossip "
                    "bounded, deduplicated embedding-pair/gain samples one hop "
                    "per step. In realistic mode, only MAC-scheduled, "
                    "SINR-decodable bundles are delivered."
                ),
            }
        )
        return assumptions
