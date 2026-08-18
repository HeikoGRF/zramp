"""Training-state utilities shared by the paper's simulation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingParams:
    replay_capacity: int = 4096
    new_data_epochs: int = 2
    replay_batches: int = 8
    recent_replay_batches: int = 4
    recent_window: int = 512
    full_dataset_epochs: int = 0
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.replay_capacity < 0:
            raise ValueError("replay_capacity must be nonnegative; zero means append-only")
        if self.new_data_epochs <= 0:
            raise ValueError("new_data_epochs must be positive")
        if not 0 <= self.recent_replay_batches <= self.replay_batches:
            raise ValueError("recent_replay_batches must be within replay_batches")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.full_dataset_epochs < 0:
            raise ValueError("full_dataset_epochs must be nonnegative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


class ReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int = 4) -> None:
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError("capacity must be nonnegative")
        storage_capacity = self.capacity if self.capacity > 0 else 1024
        self.X = np.empty((storage_capacity, feature_dim), dtype=np.float32)
        self.y = np.empty((storage_capacity, 1), dtype=np.float32)
        self.size = 0
        self.next_index = 0

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.capacity == 0:
            count = int(len(X))
            required = self.size + count
            if required > len(self.X):
                new_capacity = max(required, 2 * len(self.X))
                expanded_X = np.empty(
                    (new_capacity, self.X.shape[1]), dtype=np.float32
                )
                expanded_y = np.empty((new_capacity, 1), dtype=np.float32)
                expanded_X[: self.size] = self.X[: self.size]
                expanded_y[: self.size] = self.y[: self.size]
                self.X = expanded_X
                self.y = expanded_y
            self.X[self.size : required] = X
            self.y[self.size : required] = y
            self.size = required
            self.next_index = self.size
            return
        for features, target in zip(X, y):
            self.X[self.next_index] = features
            self.y[self.next_index] = target
            self.next_index = (self.next_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def _recent_indices(self, window: int) -> np.ndarray:
        count = min(self.size, max(1, int(window)))
        if self.capacity == 0:
            return np.arange(self.size - count, self.size, dtype=np.int64)
        offsets = np.arange(count, 0, -1, dtype=np.int64)
        return (self.next_index - offsets) % self.capacity

    def all_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise ValueError("cannot read an empty replay dataset")
        indices = self._recent_indices(self.size)
        return self.X[indices].copy(), self.y[indices].copy()

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        *,
        recent_window: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        pool = (
            np.arange(self.size, dtype=np.int64)
            if recent_window is None
            else self._recent_indices(recent_window)
        )
        indices = rng.choice(pool, size=int(batch_size), replace=True)
        return self.X[indices].copy(), self.y[indices].copy()

    def state_dict(self) -> dict[str, np.ndarray | int]:
        indices = self._recent_indices(self.size)
        return {
            "capacity": self.capacity,
            "X": self.X[indices].copy(),
            "y": self.y[indices].copy(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ReplayBuffer":
        X = np.asarray(state["X"], dtype=np.float32)
        y = np.asarray(state["y"], dtype=np.float32).reshape(-1, 1)
        result = cls(int(state["capacity"]), int(X.shape[1]))
        result.add(X, y)
        return result


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
    experience: list[int],
) -> dict[str, torch.Tensor]:
    """Return the experience-weighted average of compatible model states."""
    if not states:
        raise ValueError("cannot average an empty model list")
    weights = torch.as_tensor(
        [max(0, int(value)) for value in experience],
        dtype=torch.float64,
    )
    if float(weights.sum()) <= 0.0:
        weights.fill_(1.0)
    weights /= weights.sum()
    result: dict[str, torch.Tensor] = {}
    for name in states[0]:
        first = states[0][name]
        if not (first.is_floating_point() or first.is_complex()):
            result[name] = first.detach().clone()
            continue
        value = torch.zeros_like(first)
        for weight, state in zip(weights.tolist(), states):
            value.add_(state[name], alpha=float(weight))
        result[name] = value
    return result
