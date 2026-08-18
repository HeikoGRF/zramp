"""Deterministic decentralized per-vehicle model-transfer token windows."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenWindowState:
    window_index: int
    window_offset: int
    phase: int
    random_offset: int
    spent: bool
    spent_count: int = 0
    capacity: int = 1

    @property
    def available(self) -> bool:
        return not self.spent

    @property
    def remaining(self) -> int:
        return max(0, int(self.capacity) - int(self.spent_count))

    @property
    def random_due(self) -> bool:
        return self.available and self.window_offset == self.random_offset

    @property
    def random_ready(self) -> bool:
        """Whether a random deadline has passed while the token is unspent.

        Unlike ``random_due``, this remains true after the sampled offset. A
        vehicle can therefore use the next feasible contact when it was
        inactive, respawned late, or lost MAC scheduling at the exact offset.
        """

        return self.available and self.window_offset >= self.random_offset


class VehicleTokenWindows:
    """One expiring transmission token per local, phase-shifted S-step window."""

    def __init__(
        self, *, window_steps: int, seed: int, capacity: int = 1
    ) -> None:
        if int(window_steps) <= 0:
            raise ValueError("window_steps must be positive")
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.window_steps = int(window_steps)
        self.seed = int(seed)
        self.capacity = int(capacity)
        self._spent: dict[tuple[str, int, int], int] = {}

    @staticmethod
    def _stream_code(stream: str) -> int:
        return int(zlib.crc32(str(stream).encode("utf-8")))

    def _phase(self, node_idx: int) -> int:
        mixed = self.seed * 1_000_003 + int(node_idx) * 104_729 + 17_003
        return int(mixed % self.window_steps)

    def _random_offset(
        self, *, stream: str, node_idx: int, window_index: int
    ) -> int:
        mixed = (
            self.seed * 2_000_033
            + int(node_idx) * 130_363
            + int(window_index) * 97_409
            + self._stream_code(stream)
            + 31_337
        )
        return random.Random(mixed).randrange(self.window_steps)

    def state(
        self, *, step: int, node_idx: int, stream: str = "default"
    ) -> TokenWindowState:
        if int(step) <= 0:
            raise ValueError("step must be positive")
        phase = self._phase(int(node_idx))
        shifted = int(step) - 1 + phase
        window_index, window_offset = divmod(shifted, self.window_steps)
        key = (str(stream), int(node_idx), int(window_index))
        spent_count = int(self._spent.get(key, 0))
        return TokenWindowState(
            window_index=int(window_index),
            window_offset=int(window_offset),
            phase=int(phase),
            random_offset=self._random_offset(
                stream=str(stream),
                node_idx=int(node_idx),
                window_index=int(window_index),
            ),
            spent=spent_count >= self.capacity,
            spent_count=spent_count,
            capacity=self.capacity,
        )

    def spend(
        self, *, step: int, node_idx: int, stream: str = "default"
    ) -> TokenWindowState:
        state = self.state(step=step, node_idx=node_idx, stream=stream)
        if not state.available:
            raise RuntimeError(
                "vehicle token capacity was already spent in this window"
            )
        key = (str(stream), int(node_idx), int(state.window_index))
        self._spent[key] = int(self._spent.get(key, 0)) + 1
        return state

    def reset_node(self, node_idx: int) -> None:
        """Give a replacement vehicle a completely fresh token history."""

        target = int(node_idx)
        self._spent = {
            key: count
            for key, count in self._spent.items()
            if int(key[1]) != target
        }
