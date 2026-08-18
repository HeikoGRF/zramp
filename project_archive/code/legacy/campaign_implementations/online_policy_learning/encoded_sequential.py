"""Online learned state embeddings with sequential best-positive pulls.

Every directed receiver first obtains one quantized embedding from every
feasible provider.  The receiver then repeatedly pulls the provider with the
largest positive predicted net benefit, recomputing its own embedding after
each adopted aggregate and rescoring only providers that were positive in the
initial metadata round.

The embedding is learned online.  Its input is a deterministic signed sketch
of every predictor parameter and a streaming signed sketch of the complete
private sample trajectory.  These sketches are never transmitted; only the
quantized bottleneck is counted as decision metadata.
"""

from __future__ import annotations

import copy
import csv
import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rl_reward_experiment.sim as rre_sim
from SUMO.sumo_rl import SumoT2Simulation

from .local_validation_reward import (
    BidirectionalCrossValidationSimulation,
    PullResult,
    ValidationSubset,
)


TensorState = Mapping[str, torch.Tensor]


def signed_tensor_state_sketch(
    state: TensorState, dimension: int
) -> torch.Tensor:
    """Project every floating-point state value into a fixed signed sketch."""

    size = max(1, int(dimension))
    sketch = torch.zeros(size, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for value in state.values():
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                continue
            flat = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            count = int(flat.numel())
            if count <= 0:
                continue
            indices = torch.arange(count, dtype=torch.long) + int(offset)
            bins = torch.remainder(indices, size)
            signs = torch.where(
                torch.remainder(
                    torch.div(indices, size, rounding_mode="floor"), 2
                )
                == 0,
                torch.ones(count, dtype=torch.float32),
                -torch.ones(count, dtype=torch.float32),
            )
            sketch.index_add_(0, bins, flat * signs)
            offset += count
    if offset > 0:
        sketch /= math.sqrt(max(1.0, float(offset) / float(size)))
    return sketch


@dataclass
class StreamingTrajectorySketch:
    """Streaming signed projection of the complete ordered sample history."""

    dimension: int
    values: torch.Tensor = field(init=False)
    scalar_count: int = 0
    sample_count: int = 0

    def __post_init__(self) -> None:
        self.dimension = max(1, int(self.dimension))
        self.values = torch.zeros(self.dimension, dtype=torch.float32)

    def update(
        self,
        features: np.ndarray,
        normalized_targets: np.ndarray,
    ) -> None:
        rows = np.asarray(features, dtype=np.float32)
        targets = np.asarray(normalized_targets, dtype=np.float32).reshape(-1, 1)
        if rows.ndim != 2 or int(rows.shape[0]) != int(targets.shape[0]):
            raise ValueError("trajectory features and targets must align")
        if int(rows.shape[0]) == 0:
            return
        flat = torch.from_numpy(
            np.concatenate((rows, targets), axis=1).reshape(-1)
        ).to(dtype=torch.float32)
        count = int(flat.numel())
        indices = torch.arange(count, dtype=torch.long) + int(self.scalar_count)
        bins = torch.remainder(indices, self.dimension)
        signs = torch.where(
            torch.remainder(
                torch.div(indices, self.dimension, rounding_mode="floor"), 2
            )
            == 0,
            torch.ones(count, dtype=torch.float32),
            -torch.ones(count, dtype=torch.float32),
        )
        self.values.index_add_(0, bins, flat * signs)
        self.scalar_count += count
        self.sample_count += int(rows.shape[0])

    def tensor(self) -> torch.Tensor:
        if self.scalar_count <= 0:
            return torch.zeros(self.dimension, dtype=torch.float32)
        scale = math.sqrt(
            max(1.0, float(self.scalar_count) / float(self.dimension))
        )
        return self.values / scale

    def snapshot(self) -> dict[str, object]:
        return {
            "dimension": int(self.dimension),
            "values": self.values.detach().cpu().tolist(),
            "scalar_count": int(self.scalar_count),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def restore(
        cls, payload: object, *, dimension: int
    ) -> "StreamingTrajectorySketch":
        restored = cls(dimension=int(dimension))
        if not isinstance(payload, dict):
            return restored
        values = torch.as_tensor(
            payload.get("values", []), dtype=torch.float32
        ).reshape(-1)
        if int(values.numel()) == restored.dimension:
            restored.values.copy_(values)
        restored.scalar_count = max(0, int(payload.get("scalar_count", 0)))
        restored.sample_count = max(0, int(payload.get("sample_count", 0)))
        return restored


class LearnedStateUtilityPolicy(nn.Module):
    """Small encoder plus a scalar raw pull-benefit regressor."""

    def __init__(self, *, raw_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(32, int(embedding_dim))
        self.embedding_dim = int(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(int(raw_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.embedding_dim),
            nn.Tanh(),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        final = self.gain_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def quantize_embedding(values: torch.Tensor) -> torch.Tensor:
        """Fake-quantize to signed int8 while retaining a training gradient."""

        maximum = values.detach().abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(maximum / 127.0, min=1.0e-8)
        dequantized = torch.clamp(
            torch.round(values / scale), -127.0, 127.0
        ) * scale
        return values + (dequantized - values).detach()

    def encode(self, raw_state: torch.Tensor) -> torch.Tensor:
        values = self.encoder(raw_state)
        return self.quantize_embedding(values)

    def score_embeddings(
        self, receiver_embedding: torch.Tensor, provider_embedding: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.cat((receiver_embedding, provider_embedding), dim=-1)
        return self.gain_head(pair).squeeze(-1)

    def forward(
        self, receiver_raw: torch.Tensor, provider_raw: torch.Tensor
    ) -> torch.Tensor:
        return self.score_embeddings(
            self.encode(receiver_raw), self.encode(provider_raw)
        )


class _ExperienceView:
    def __init__(self, owner: "OnlineEncodedUtilityAgent") -> None:
        self.owner = owner
        self.capacity = 1_000_000_000

    def __len__(self) -> int:
        return int(self.owner.experience)


class OnlineEncodedUtilityAgent:
    """Online utility learner with a slowly changing serving snapshot."""

    def __init__(
        self,
        *,
        raw_dim: int,
        embedding_dim: int,
        device: torch.device,
        learning_rate: float,
        rng_seed: int,
        model_seed: int,
    ) -> None:
        self.device = device
        self.learning_rate = float(learning_rate)
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(model_seed) & 0x7FFFFFFFFFFFFFFF)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(model_seed))
            self.policy = LearnedStateUtilityPolicy(
                raw_dim=int(raw_dim), embedding_dim=int(embedding_dim)
            ).to(device)
        self.serving = copy.deepcopy(self.policy).to(device)
        self.serving.eval()
        self.opt = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )
        self._py_rng = random.Random(int(rng_seed))
        self.action_policy = "argmax"
        self.batch_size = 1
        self.experience = 0
        self.train_steps = 0
        self.serving_version = 0
        self.replay = _ExperienceView(self)

    def train_pair(
        self,
        receiver_raw: torch.Tensor,
        provider_raw: torch.Tensor,
        target_gain: float,
    ) -> tuple[float, float]:
        self.policy.train()
        receiver = receiver_raw.to(self.device, dtype=torch.float32).unsqueeze(0)
        provider = provider_raw.to(self.device, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(
            [float(target_gain)], dtype=torch.float32, device=self.device
        )
        self.opt.zero_grad(set_to_none=True)
        prediction = self.policy(receiver, provider)
        loss = F.smooth_l1_loss(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.opt.step()
        self.experience += 1
        self.train_steps += 1
        return float(prediction.detach().cpu().item()), float(loss.detach().cpu())

    def update_serving(self, tau: float) -> None:
        amount = float(tau)
        if not 0.0 < amount <= 1.0:
            raise ValueError("serving tau must be in (0, 1]")
        with torch.no_grad():
            for target, source in zip(
                self.serving.parameters(), self.policy.parameters()
            ):
                target.mul_(1.0 - amount).add_(source, alpha=amount)
            for target, source in zip(
                self.serving.buffers(), self.policy.buffers()
            ):
                target.copy_(source)
        self.serving.eval()
        self.serving_version += 1

    def serving_embeddings(
        self, receiver_raw: torch.Tensor, provider_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.serving.eval()
        with torch.no_grad():
            receiver = self.serving.encode(
                receiver_raw.to(self.device, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
            provider = self.serving.encode(
                provider_raw.to(self.device, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
            gain = self.serving.score_embeddings(
                receiver.unsqueeze(0), provider.unsqueeze(0)
            ).item()
        return receiver.cpu(), provider.cpu(), float(gain)

    def serving_provider_embedding(
        self, provider_raw: torch.Tensor
    ) -> torch.Tensor:
        self.serving.eval()
        with torch.no_grad():
            return self.serving.encode(
                provider_raw.to(self.device, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0).cpu()

    def serving_receiver_embedding(
        self, receiver_raw: torch.Tensor
    ) -> torch.Tensor:
        return self.serving_provider_embedding(receiver_raw)

    def serving_gain_from_embeddings(
        self, receiver: torch.Tensor, provider: torch.Tensor
    ) -> float:
        self.serving.eval()
        with torch.no_grad():
            value = self.serving.score_embeddings(
                receiver.to(self.device).unsqueeze(0),
                provider.to(self.device).unsqueeze(0),
            )
        return float(value.item())


@dataclass(frozen=True)
class SequentialSelection:
    provider: int
    predicted_gain: float
    net_score: float


def sequential_best_positive(
    candidate_ids: list[int],
    *,
    beta: float,
    score: Callable[[int], float],
    select: Callable[[SequentialSelection], None],
) -> tuple[dict[int, float], dict[int, float], list[SequentialSelection]]:
    """Select the highest positive candidate, then rescore the initial set."""

    initial = {int(provider): float(score(int(provider))) for provider in candidate_ids}
    remaining = {
        provider
        for provider, predicted in initial.items()
        if float(predicted) - float(beta) > 0.0
    }
    current = dict(initial)
    selected: list[SequentialSelection] = []
    while remaining:
        best = max(
            remaining,
            key=lambda provider: (
                float(current[provider]) - float(beta), -int(provider)
            ),
        )
        net = float(current[best]) - float(beta)
        if net <= 0.0:
            break
        row = SequentialSelection(
            provider=int(best),
            predicted_gain=float(current[best]),
            net_score=float(net),
        )
        select(row)
        selected.append(row)
        remaining.remove(best)
        current.update(
            {provider: float(score(provider)) for provider in remaining}
        )
    return initial, current, selected


class EncodedSequentialBidirectionalSimulation(
    BidirectionalCrossValidationSimulation
):
    """Learned compact metadata and sequential best-positive aggregation."""

    policy_transfer_rule = "all-feasible-online-encoded-policy-average"

    def __init__(
        self,
        *args,
        state_sketch_dim: int = 32,
        embedding_dim: int = 32,
        serving_interval: int = 25,
        serving_tau: float = 0.5,
        **kwargs,
    ) -> None:
        if int(state_sketch_dim) <= 0:
            raise ValueError("state_sketch_dim must be positive")
        if not 8 <= int(embedding_dim) <= 64:
            raise ValueError("embedding_dim must be in [8, 64]")
        if int(serving_interval) <= 0:
            raise ValueError("serving_interval must be positive")
        if not 0.0 < float(serving_tau) <= 1.0:
            raise ValueError("serving_tau must be in (0, 1]")
        self.state_sketch_dim = int(state_sketch_dim)
        self.embedding_dim = int(embedding_dim)
        self.raw_state_dim = 2 * self.state_sketch_dim
        self.serving_interval = int(serving_interval)
        self.serving_tau = float(serving_tau)
        self.embedding_wire_bytes = self.embedding_dim + 8
        self._trajectory_sketches: list[StreamingTrajectorySketch] = []
        self._encoded_log_file = None
        self._encoded_log_writer: csv.DictWriter | None = None
        self._encoded_log_count = 0
        kwargs["random_pull_probability"] = None
        kwargs.setdefault("share_policy_every_contact", True)
        super().__init__(*args, **kwargs)
        self.policy_state_dim = 2 * self.embedding_dim
        self._trajectory_sketches = [
            StreamingTrajectorySketch(self.state_sketch_dim)
            for _ in self.nodes
        ]
        self.zramp_policy_mode = (
            "bidirectional-private-cv-learned-embedding-sequential-positive"
        )
        self.local_policy_initial_pull = (
            "best-score-random-fallback-during-online-warmup"
        )
        self.local_policy_initial_pull_probability = float(
            self.hard_warmup_pull_probability
        )
        self._encoded_log_path = (
            Path(self.cfg.results_dir) / "encoded_policy_training.csv"
        )
        self._communication_assumptions = self._build_communication_assumptions()

    # ------------------------------------------------------------ policy init

    def _init_local_policy_agents(self) -> None:
        self.local_agents.clear()
        self._local_policy_pending_transitions.clear()
        self._local_policy_versions.clear()
        self._local_policy_initial_rngs.clear()
        for mode_id in list(self.agents):
            mode_offset = int(
                zlib.crc32(str(mode_id).encode("utf-8")) % 10_000_000
            )
            model_seed = int(self.cfg.seed) + 703_001 + mode_offset
            template = OnlineEncodedUtilityAgent(
                raw_dim=self.raw_state_dim,
                embedding_dim=self.embedding_dim,
                device=self.device,
                learning_rate=self.cfg.rl_lr,
                rng_seed=model_seed,
                model_seed=model_seed,
            )
            template_state = {
                name: value.detach().clone()
                for name, value in template.policy.state_dict().items()
            }
            agents: list[OnlineEncodedUtilityAgent] = []
            rngs: list[random.Random] = []
            for node_idx in range(int(self.cfg.num_nodes)):
                seed = (
                    int(self.cfg.seed)
                    + 2_300_003
                    + mode_offset
                    + 104_729 * int(node_idx)
                )
                agent = OnlineEncodedUtilityAgent(
                    raw_dim=self.raw_state_dim,
                    embedding_dim=self.embedding_dim,
                    device=self.device,
                    learning_rate=self.cfg.rl_lr,
                    rng_seed=seed,
                    model_seed=model_seed,
                )
                agent.policy.load_state_dict(template_state)
                agent.serving.load_state_dict(template_state)
                agents.append(agent)
                rngs.append(random.Random(seed + 57_911))
            self.agents[mode_id] = template
            self.local_agents[mode_id] = agents
            self._local_policy_pending_transitions[mode_id] = [
                0 for _ in agents
            ]
            self._local_policy_versions[mode_id] = [0 for _ in agents]
            self._local_policy_initial_rngs[mode_id] = rngs

    def _policy_experience(self, mode_id: str, node_idx: int) -> int:
        agents = self.local_agents.get(mode_id, [])
        if not 0 <= int(node_idx) < len(agents):
            return 0
        return int(agents[int(node_idx)].experience)

    def _policy_replay_capacity(self, mode_id: str, node_idx: int) -> int:
        del mode_id, node_idx
        return 1_000_000_000

    def _train_rl_agents(self, step: int | None = None) -> dict[str, float]:
        del step
        return {mode: 0.0 for mode in self.agents}

    def _queue_rl_transition(self, mode_id: str, transition) -> None:
        del mode_id, transition

    # --------------------------------------------------------- private state

    def _raw_state(
        self,
        node_idx: int,
        mode: str,
        *,
        model_state: TensorState | None = None,
    ) -> torch.Tensor:
        if model_state is None:
            model_state = self.nodes[int(node_idx)].variants[mode].model.state_dict()
        model = signed_tensor_state_sketch(
            model_state, self.state_sketch_dim
        )
        trajectory = self._trajectory_sketches[int(node_idx)].tensor()
        return torch.cat((model, trajectory), dim=0).to(dtype=torch.float32)

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
        if (
            hasattr(self, "_trajectory_sketches")
            and len(self._trajectory_sketches) == len(self.nodes)
            and n_new > 0
            and node_idx >= 0
        ):
            start = len(ns.current_visit_samples_x) - n_new
            new_features = np.asarray(
                ns.current_visit_samples_x[start:], dtype=np.float32
            )
            new_targets = np.asarray(
                ns.current_visit_samples_y[start:], dtype=np.float32
            ).reshape(-1, 1)
            normalized = self._normalize_target_from_rssi(new_targets)
            self._trajectory_sketches[node_idx].update(
                new_features, normalized
            )
        super()._train_local(
            ns,
            X,
            y_dbm,
            sample_count_increment=sample_count_increment,
            sample_weights=sample_weights,
        )

    def _save_node_zone_memory(self, node_idx: int, zone: int) -> None:
        super()._save_node_zone_memory(node_idx, zone)
        if (
            self.zone_model_memory
            and len(self._trajectory_sketches) == len(self.nodes)
        ):
            self._node_zone_memory[int(node_idx)][int(zone)][
                "trajectory_sketch"
            ] = self._trajectory_sketches[int(node_idx)].snapshot()

    def _restore_node_zone_memory(self, node_idx: int, zone: int) -> bool:
        restored = super()._restore_node_zone_memory(node_idx, zone)
        if (
            restored
            and len(self._trajectory_sketches) == len(self.nodes)
        ):
            payload = self._node_zone_memory[int(node_idx)][int(zone)].get(
                "trajectory_sketch", {}
            )
            self._trajectory_sketches[int(node_idx)] = (
                StreamingTrajectorySketch.restore(
                    payload, dimension=self.state_sketch_dim
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
        super()._reset_node_for_zone_change(ns, new_az)
        if (
            node_idx >= 0
            and not cached
            and len(self._trajectory_sketches) == len(self.nodes)
        ):
            self._trajectory_sketches[node_idx] = StreamingTrajectorySketch(
                self.state_sketch_dim
            )

    # ------------------------------------------------------- private CV rules

    @staticmethod
    def _validation_subset_weight(subset: ValidationSubset) -> float:
        """Weight both private validation losses by their sample counts."""

        return float(len(subset.features))

    def _metadata_for(self, node_idx: int, mode: str):
        del node_idx, mode
        return SimpleNamespace(wire_nbytes=int(self.embedding_wire_bytes))

    def _make_peer_view(self, ns_j, mode: str):
        return rre_sim.Simulation._make_peer_view(self, ns_j, mode)

    # ------------------------------------------------------------ sequential

    def _share_policies_with_all_feasible_neighbors(
        self, links: list[tuple[int, int, int]]
    ) -> None:
        super()._share_policies_with_all_feasible_neighbors(links)
        step = int(getattr(self, "_current_sumo_step", 0))
        if step != 1 and step % self.serving_interval != 0:
            return
        for agents in self.local_agents.values():
            for agent in agents:
                agent.update_serving(self.serving_tau)

    def _record_encoded_decision(
        self,
        *,
        step: int,
        enc_id: int,
        zone: int,
        receiver_idx: int,
        provider_idx: int,
        mode: str,
        action: int,
        predicted_gain: float,
        reward: float,
        exploratory: bool,
        alpha: float | None,
    ) -> None:
        receiver = self.nodes[int(receiver_idx)]
        provider = self.nodes[int(provider_idx)]
        distance = float(
            np.hypot(
                receiver.node.x - provider.node.x,
                receiver.node.y - provider.node.y,
            )
        )
        SumoT2Simulation._record_decision_row(
            self,
            {
                "step": int(step),
                "enc_id": int(enc_id),
                "node_i": int(receiver_idx),
                "node_j": int(provider_idx),
                "az": int(zone),
                "dist": distance,
                "mode": str(mode),
                "action": int(action),
                "merge_weight": "" if alpha is None else float(alpha),
                "predicted_gain": float(predicted_gain),
                "gain_threshold": float(self.communication_penalty),
                "exploratory": int(bool(exploratory)),
                "reward": float(reward),
                "deferred": False,
            },
        )

    def _train_selected_pair(
        self,
        *,
        step: int,
        mode: str,
        receiver_idx: int,
        provider_idx: int,
        receiver_raw: torch.Tensor,
        provider_raw: torch.Tensor,
        target_gain: float,
    ) -> None:
        agent = self.local_agents[mode][int(receiver_idx)]
        prediction, loss = agent.train_pair(
            receiver_raw, provider_raw, float(target_gain)
        )
        self._local_policy_train_updates[mode] += 1
        self._last_local_policy_train_updates_this_step += 1
        self._last_local_policy_queued_transitions += 1
        self._ensure_encoded_log().writerow(
            {
                "step": int(step),
                "mode": str(mode),
                "receiver_idx": int(receiver_idx),
                "provider_idx": int(provider_idx),
                "target_gain": float(target_gain),
                "online_prediction": float(prediction),
                "loss": float(loss),
                "serving_version": int(agent.serving_version),
            }
        )
        self._encoded_log_count += 1
        if self._encoded_log_count % 1000 == 0:
            self._flush_encoded_log()

    def _gossip_step(
        self,
        step: int,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> None:
        links = self._normalized_contact_links(zone_nodes, contact_links)
        self._share_policies_with_all_feasible_neighbors(links)
        if not links:
            return

        neighbors: dict[int, list[tuple[int, int]]] = {}
        mutable: dict[int, list[tuple[int, int]]] = {}
        for zone, a, b in links:
            mutable.setdefault(int(a), []).append((int(zone), int(b)))
            mutable.setdefault(int(b), []).append((int(zone), int(a)))
        neighbors = {
            receiver: sorted(set(rows), key=lambda row: (row[0], row[1]))
            for receiver, rows in mutable.items()
        }

        provider_views: dict[str, dict[int, object]] = {}
        provider_raw: dict[str, dict[int, torch.Tensor]] = {}
        for mode in self.agents:
            provider_views[mode] = {}
            provider_raw[mode] = {}
            for node_idx in sorted(neighbors):
                view = self._make_peer_view(self.nodes[node_idx], mode)
                provider_views[mode][node_idx] = view
                provider_raw[mode][node_idx] = self._raw_state(
                    node_idx,
                    mode,
                    model_state=view._model_state,  # type: ignore[attr-defined]
                )

        for receiver_idx in sorted(neighbors):
            candidates = neighbors[receiver_idx]
            for mode in self.agents:
                agent = self.local_agents[mode][receiver_idx]
                candidate_ids = [provider for _zone, provider in candidates]
                zones = {provider: zone for zone, provider in candidates}
                enc_ids = {}
                for provider_idx in candidate_ids:
                    enc_ids[provider_idx] = int(self._next_enc_id)
                    self._next_enc_id += 1
                    self._cv_step_metadata_bytes[mode] += int(
                        self.embedding_wire_bytes
                    )

                current_receiver_raw = self._raw_state(receiver_idx, mode)
                receiver_embedding = agent.serving_receiver_embedding(
                    current_receiver_raw
                )
                provider_embeddings = {
                    provider_idx: agent.serving_provider_embedding(
                        provider_raw[mode][provider_idx]
                    )
                    for provider_idx in candidate_ids
                }

                def score(provider_idx: int) -> float:
                    return agent.serving_gain_from_embeddings(
                        receiver_embedding,
                        provider_embeddings[int(provider_idx)],
                    )

                selected_results: dict[int, tuple[SequentialSelection, PullResult]] = {}

                def select(selection: SequentialSelection) -> None:
                    nonlocal current_receiver_raw, receiver_embedding
                    provider_idx = int(selection.provider)
                    before_raw = current_receiver_raw.clone()
                    result = self._execute_validation_pull(
                        step=int(step),
                        mode=mode,
                        receiver=self.nodes[receiver_idx],
                        provider=self.nodes[provider_idx],
                        zone=int(zones[provider_idx]),
                        provider_view=provider_views[mode][provider_idx],
                    )
                    selected_results[provider_idx] = (selection, result)
                    if result.valid and result.reward is not None:
                        self._train_selected_pair(
                            step=int(step),
                            mode=mode,
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                            receiver_raw=before_raw,
                            provider_raw=provider_raw[mode][provider_idx],
                            target_gain=float(result.reward),
                        )
                    current_receiver_raw = self._raw_state(receiver_idx, mode)
                    receiver_embedding = agent.serving_receiver_embedding(
                        current_receiver_raw
                    )

                initial, final_scores, selected = sequential_best_positive(
                    candidate_ids,
                    beta=float(self.communication_penalty),
                    score=score,
                    select=select,
                )

                exploratory_provider: int | None = None
                if (
                    not selected
                    and int(step) <= int(self.hard_warmup_steps)
                    and candidate_ids
                    and agent._py_rng.random()
                    < float(self.hard_warmup_pull_probability)
                ):
                    exploratory_provider = max(
                        candidate_ids,
                        key=lambda provider: (
                            float(initial[provider]), -int(provider)
                        ),
                    )
                    exploratory_selection = SequentialSelection(
                        provider=int(exploratory_provider),
                        predicted_gain=float(initial[exploratory_provider]),
                        net_score=float(initial[exploratory_provider])
                        - float(self.communication_penalty),
                    )
                    select(exploratory_selection)
                    selected.append(exploratory_selection)

                selected_ids = {row.provider for row in selected}
                for provider_idx in candidate_ids:
                    if provider_idx in selected_ids:
                        selection, result = selected_results[provider_idx]
                        raw_reward = (
                            0.0
                            if result.reward is None
                            else float(result.reward)
                        )
                        self._record_encoded_decision(
                            step=int(step),
                            enc_id=enc_ids[provider_idx],
                            zone=zones[provider_idx],
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                            mode=mode,
                            action=1,
                            predicted_gain=selection.predicted_gain,
                            reward=raw_reward
                            - float(self.communication_penalty),
                            exploratory=(provider_idx == exploratory_provider),
                            alpha=result.alpha,
                        )
                    else:
                        predicted = float(
                            final_scores.get(provider_idx, initial[provider_idx])
                        )
                        self._record_encoded_decision(
                            step=int(step),
                            enc_id=enc_ids[provider_idx],
                            zone=zones[provider_idx],
                            receiver_idx=receiver_idx,
                            provider_idx=provider_idx,
                            mode=mode,
                            action=0,
                            predicted_gain=predicted,
                            reward=0.0,
                            exploratory=False,
                            alpha=None,
                        )

    # ---------------------------------------------------------- communication

    def _build_communication_assumptions(
        self,
    ) -> dict[str, int | float | str | bool]:
        assumptions = super()._build_communication_assumptions()
        policy_bytes = int(assumptions.get("B_policy_bytes", 0))
        agents = getattr(self, "local_agents", {})
        if agents:
            first_mode = next(iter(agents.values()))
            if first_mode:
                policy_bytes = self._state_nbytes(
                    self._clone_state(first_mode[0].policy)
                )
        assumptions.update(
            {
                "zramp_policy_mode": (
                    "bidirectional-private-cv-learned-embedding-"
                    "sequential-positive"
                ),
                "policy_pull_rule": (
                    "repeated-highest-predicted-raw-gain-minus-beta-"
                    "until-no-initially-positive-provider-remains"
                ),
                "policy_reward_target": (
                    "raw-sample-count-weighted-two-sided-normalized-MSE-"
                    "improvement;beta-applied-only-at-decision"
                ),
                "validation_quality": "private-validation-sample-count",
                "policy_observation_dim": int(2 * self.embedding_dim),
                "policy_observation_features": (
                    "learned_receiver_embedding,learned_provider_embedding"
                ),
                "state_encoder_input": (
                    "all-predictor-parameters-plus-complete-ordered-local-"
                    "sample-trajectory-signed-sketches"
                ),
                "state_sketch_dim_per_branch": int(self.state_sketch_dim),
                "state_embedding_dim": int(self.embedding_dim),
                "state_embedding_quantization": "signed-int8-plus-scale",
                "serving_encoder_interval_steps": int(self.serving_interval),
                "serving_encoder_ema_tau": float(self.serving_tau),
                "B_decision_meta_bytes_per_directed_decision": int(
                    self.embedding_wire_bytes
                ),
                "B_model_signature_bytes_per_directed_decision": 0,
                "B_validation_quality_bytes_per_directed_decision": 0,
                "B_policy_bytes": int(policy_bytes),
                "B_policy_model_bytes_per_directed_contact": int(policy_bytes),
                "policy_params": int(policy_bytes // 4),
                "metadata_note": (
                    "Every feasible provider returns one quantized learned "
                    "embedding. Predictor parameters, trajectory samples, "
                    "and validation samples remain local. Provider predictor "
                    "states are frozen at the metadata round; a receiver may "
                    "pull several such providers sequentially. Each pull is "
                    "optimized and evaluated on both vehicles' private "
                    "validation sets with sample-count weighting."
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
                    "loss",
                    "serving_version",
                ),
            )
            self._encoded_log_writer.writeheader()
        return self._encoded_log_writer

    def _flush_encoded_log(self) -> None:
        if self._encoded_log_file is not None:
            self._encoded_log_file.flush()

    def _close_encoded_log(self) -> None:
        if self._encoded_log_file is not None:
            self._encoded_log_file.flush()
            self._encoded_log_file.close()
        self._encoded_log_file = None
        self._encoded_log_writer = None

    def _write_partial_outputs(self, *args, **kwargs) -> None:
        super()._write_partial_outputs(*args, **kwargs)
        self._flush_encoded_log()

    def run(self) -> None:
        self._ensure_encoded_log()
        try:
            super().run()
        finally:
            self._close_encoded_log()
