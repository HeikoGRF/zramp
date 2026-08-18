"""Compatibility layer for the original finite-segment capsule support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.capsule_greedy import run_capsule_greedy as original


CapsuleRow = original.CapsuleRow


@dataclass(frozen=True)
class CapsuleParams:
    angle_deg: float = 12.0
    lateral_merge_m: float = 8.0
    longitudinal_gap_m: float = 10.0
    initial_half_width_m: float = 1.5
    mass_scale: float = 3.0


class Capsule(original.Capsule):
    @classmethod
    def from_segment(
        cls,
        segment: np.ndarray,
        *,
        half_width: float | None = None,
        mass: float = 1.0,
    ) -> "Capsule":
        del half_width
        points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
        if float(np.linalg.norm(points[1] - points[0])) < 1.0e-6:
            raise ValueError("capsule segment must have nonzero length")
        return cls(points[0].copy(), points[1].copy(), float(mass))


class CapsuleGatedMLP(original.CapsuleGatedMLP):
    def __init__(
        self,
        base,
        *,
        map_size_m: float,
        floor_prior_norm: float,
        ribbon_params: CapsuleParams,
        gate_params,
        binary_support: bool = False,
    ) -> None:
        if binary_support:
            raise ValueError("finite capsules use their smooth confidence gate")
        super().__init__(
            base,
            map_size_m=map_size_m,
            floor_prior_norm=floor_prior_norm,
            capsule_params=ribbon_params,
            gate_params=gate_params,
        )

    def set_ribbons(self, rows: tuple[CapsuleRow, ...]) -> None:
        self.set_capsules(rows)


add_capsule_vectorized = original.add_capsule_vectorized
deserialize_capsules = original.deserialize_capsules
remote_union = original.remote_union
capsule_delta = original.capsule_delta
serialize_capsules = original.serialize_capsules
self_test = original.self_test
