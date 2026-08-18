"""
Minimal DQN agent tailored to the 8-feature / 2-action gossip decision problem.

One agent instance per reward mode. State layout (produced by
`sim._state_features`):

    0  li_norm         local RMSE / |rssi_min_dbm|  (noise-floor normalised)
    1  rei             relative experience index, where experience = capped n_samples
    2  rqi             relative quality index (RMSE-based)
    3  t_norm          1 - exp(-t_wait / tau) with tau = cfg.t_norm_tau
    4  zone_x_norm     current zone center x coordinate in [0, 1]
    5  zone_y_norm     current zone center y coordinate in [0, 1]
    6  w_diff          L2 distance between model weights
    7  neighbor_norm   neighbour density / (num_nodes - 1)

Actions: 0 = reject, 1 = accept.

Exploration: Boltzmann (softmax) sampling on Q-values, NOT epsilon-greedy.
`P(a) = softmax(Q(s, a))`, so a small Q-margin keeps both actions probable
while a large Q-margin commits the policy. The `epsilon_*` config knobs are
kept for backwards compatibility but no longer used by `select_action`.

Bandit reduction: every transition pushed into the replay buffer carries
`done=True` and `next_state == state` (see `_gossip_step` and
`reward_modes.FutureWindowMode._finalize_slot`). The Bellman target therefore
collapses to `target = r`, turning DQN training into per-encounter regression
onto the (possibly delayed) reward. This is intentional — the "next" gossip
encounter for a given node typically involves a different peer in a different
zone, so propagating value through `Q(s', a')` would mix unrelated decisions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


STATE_DIM = 8
ACTION_DIM = 2


class QNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, state_dim: int = STATE_DIM):
        self.capacity = max(0, int(capacity))
        self.state_dim = int(state_dim)
        self._states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.empty((self.capacity,), dtype=np.int64)
        self._rewards = np.empty((self.capacity,), dtype=np.float64)
        self._dones = np.empty((self.capacity,), dtype=np.bool_)
        self._start = 0
        self._size = 0
        self._next = 0
        self._population = _ReplayPopulation(self)

    def __len__(self) -> int:
        return int(self._size)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        if self.capacity <= 0:
            return
        idx = int(self._next)
        state_cpu = state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        next_state_cpu = next_state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if int(state_cpu.numel()) != self.state_dim or int(next_state_cpu.numel()) != self.state_dim:
            raise ValueError(
                f"Replay state has shape {tuple(state_cpu.shape)} / {tuple(next_state_cpu.shape)}, "
                f"expected flat dimension {self.state_dim}"
            )
        self._states[idx, :] = state_cpu.numpy()
        self._next_states[idx, :] = next_state_cpu.numpy()
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._dones[idx] = bool(done)
        if self._size < self.capacity:
            self._size += 1
        else:
            self._start = (self._start + 1) % self.capacity
        self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int, rng: random.Random):
        batch = rng.sample(self._population, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32),
        )

    def _physical_index(self, logical_idx: int) -> int:
        idx = int(logical_idx)
        if idx < 0:
            idx += int(self._size)
        if idx < 0 or idx >= int(self._size):
            raise IndexError(idx)
        return int((self._start + idx) % self.capacity)

    def _get_transition(self, logical_idx: int):
        idx = self._physical_index(logical_idx)
        return (
            torch.from_numpy(self._states[idx]),
            int(self._actions[idx]),
            float(self._rewards[idx]),
            torch.from_numpy(self._next_states[idx]),
            bool(self._dones[idx]),
        )


class _ReplayPopulation(Sequence):
    """Sequence facade so random.sample consumes the same RNG pattern as before."""

    def __init__(self, replay: ReplayBuffer):
        self._replay = replay

    def __len__(self) -> int:
        return len(self._replay)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._replay._get_transition(i) for i in range(*idx.indices(len(self)))]
        return self._replay._get_transition(int(idx))


class DQNAgent:
    """Single-mode DQN agent with soft target updates and linear epsilon decay."""

    def __init__(
        self,
        *,
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        tau: float = 0.01,
        capacity: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2_000,
        rng_seed: int = 0,
        action_policy: str = "softmax",
        softmax_temperature: float = 1.0,
    ) -> None:
        self.device = device
        # Isolated from other agents so parallel RL heads do not share one RNG stream.
        self._py_rng = random.Random(int(rng_seed))
        torch_seed = int(rng_seed) & 0x7FFFFFFFFFFFFFFF
        fork_devices: list[int] = []
        if getattr(device, "type", "") == "cuda":
            fork_devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(torch_seed)
            if getattr(device, "type", "") == "cuda":
                torch.cuda.manual_seed_all(torch_seed)
            self.policy = QNet().to(device)
            self.target = QNet().to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(capacity=capacity)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.action_policy = str(action_policy).strip().lower()
        self.softmax_temperature = float(softmax_temperature)
        if not math.isfinite(self.softmax_temperature) or self.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be finite and positive")
        self._decisions = 0

    def epsilon(self) -> float:
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        frac = min(1.0, self._decisions / float(self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def select_action(self, state: torch.Tensor, rng: random.Random | None = None) -> int:
        """Softmax (Boltzmann) sampling on the Q-values.

        Probability of each action is proportional to exp(Q(s, a)). This
        replaces the previous epsilon-greedy scheme so that exploration is
        intrinsic and Q-margin-aware: small Q-differences keep both actions
        alive (good for noisy reward signals), large Q-differences commit
        confidently. Matches the design of the older `zone_sharing_sim` agent.

        Sampling uses this agent's private RNG (not global torch or ``random``)
        so multiple active modes in one run do not perturb each other's action
        trajectories. ``rng`` is ignored; kept for call-site compatibility.
        """
        del rng
        self._decisions += 1
        with torch.no_grad():
            q = self.policy(state.unsqueeze(0).to(self.device))
        if self.action_policy in {"reject", "always_reject"}:
            return 0
        if self.action_policy in {"accept", "always_accept"}:
            return 1
        if self.action_policy == "argmax":
            return int(torch.argmax(q, dim=1).item())
        if self.action_policy != "softmax":
            raise ValueError(f"Unknown RL action policy {self.action_policy!r}")
        with torch.no_grad():
            probs = torch.softmax(
                q / self.softmax_temperature, dim=1
            ).squeeze(0).detach().cpu().tolist()
        u = self._py_rng.random()
        cut = 0.0
        for a, p in enumerate(probs):
            cut += float(p)
            if u < cut:
                return int(a)
        return int(len(probs) - 1)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0
        s, a, r, s2, d = self.replay.sample(self.batch_size, self._py_rng)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - d)
        loss = self.loss_fn(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.opt.step()
        self._soft_update()
        return float(loss.item())

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p_target, p_policy in zip(
                self.target.parameters(), self.policy.parameters()
            ):
                p_target.data.mul_(1.0 - self.tau).add_(p_policy.data, alpha=self.tau)
