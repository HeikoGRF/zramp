"""Scalar pull-utility estimator and replay buffer."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import weighted_average


class UtilityNet(nn.Module):
    """Micro scalar policy with one configurable hidden layer."""

    def __init__(self, observation_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(observation_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


def policy_experience_weights(experiences: Sequence[int | float]) -> torch.Tensor:
    """Normalize policy decision counts, using equal weights before learning."""

    if not experiences:
        raise ValueError("at least one policy experience is required")
    values = torch.tensor(experiences, dtype=torch.float64).clamp_min(0.0)
    total = float(values.sum())
    if total <= 0.0:
        values.fill_(1.0 / float(values.numel()))
    else:
        values /= total
    return values.to(dtype=torch.float32)


def aggregate_policy_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    experiences: Sequence[int | float],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Average own and received policy weights by local pull-decision counts."""

    if len(states) != len(experiences):
        raise ValueError("policy states and experiences must have the same length")
    weights = policy_experience_weights(experiences)
    return weighted_average(list(states), weights), weights


class UtilityReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int):
        if int(capacity) <= 0:
            raise ValueError("utility replay capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self._observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._size = 0
        self._next = 0

    def __len__(self) -> int:
        return int(self._size)

    def push(self, observation: torch.Tensor | np.ndarray, reward: float) -> None:
        values = np.asarray(
            observation.detach().cpu().numpy()
            if isinstance(observation, torch.Tensor)
            else observation,
            dtype=np.float32,
        ).reshape(-1)
        if int(values.size) != self.observation_dim:
            raise ValueError("utility observation has the wrong dimension")
        self._observations[self._next] = values
        self._rewards[self._next] = float(reward)
        self._next = (self._next + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(
        self, batch_size: int, *, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray]:
        if int(batch_size) > self._size:
            raise ValueError("not enough utility replay entries")
        indices = rng.sample(range(self._size), int(batch_size))
        return self._observations[indices].copy(), self._rewards[indices].copy()


class UtilityAgent:
    def __init__(
        self,
        *,
        observation_dim: int,
        hidden_dim: int,
        device: torch.device,
        learning_rate: float,
        batch_size: int,
        replay_capacity: int,
        rng_seed: int,
        model_seed: int | None = None,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.rng = random.Random(int(rng_seed))
        torch_seed = (
            int(rng_seed if model_seed is None else model_seed)
            & 0x7FFFFFFFFFFFFFFF
        )
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(torch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(torch_seed)
            self.model = UtilityNet(observation_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.replay = UtilityReplayBuffer(replay_capacity, observation_dim)
        self.train_steps = 0
        self.inherited_ready = False

    @property
    def experience(self) -> int:
        """Number of matured local pull decisions retained for policy learning."""

        return len(self.replay)

    @property
    def ready(self) -> bool:
        return bool(
            self.inherited_ready
            or (len(self.replay) >= self.batch_size and self.train_steps > 0)
        )

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().to(device="cpu").clone()
            for name, tensor in self.model.state_dict().items()
        }

    def load_shared_model(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        inherited_ready: bool,
    ) -> None:
        """Install an aggregated policy and discard stale Adam momentum."""

        self.model.load_state_dict(
            {name: tensor.to(self.device) for name, tensor in state.items()}
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.inherited_ready = bool(inherited_ready)

    def score(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(
                observations.to(device=self.device, dtype=torch.float32)
            ).detach().cpu()

    def push(self, observation: torch.Tensor, reward: float) -> None:
        self.replay.push(observation, reward)

    def train(self, updates: int) -> float | None:
        if len(self.replay) < self.batch_size or int(updates) <= 0:
            return None
        self.model.train()
        last_loss: float | None = None
        for _ in range(int(updates)):
            observations, rewards = self.replay.sample(
                self.batch_size, rng=self.rng
            )
            x = torch.tensor(observations, dtype=torch.float32, device=self.device)
            y = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(x)
            loss = F.mse_loss(predictions, y)
            loss.backward()
            self.optimizer.step()
            self.train_steps += 1
            last_loss = float(loss.detach().cpu())
        return last_loss
