"""Finite variable-width support planes for Place Wallis predictors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


RibbonRow = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


@dataclass(frozen=True)
class RibbonParams:
    angle_deg: float = 12.0
    lateral_merge_m: float = 8.0
    longitudinal_gap_m: float = 10.0
    initial_half_width_m: float = 1.5
    mass_scale: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.angle_deg < 90.0:
            raise ValueError("angle_deg must be in [0, 90)")
        if self.lateral_merge_m < 0.0:
            raise ValueError("lateral_merge_m cannot be negative")
        if self.longitudinal_gap_m < 0.0:
            raise ValueError("longitudinal_gap_m cannot be negative")
        if self.initial_half_width_m < 0.0:
            raise ValueError("initial_half_width_m cannot be negative")
        if self.mass_scale <= 0.0:
            raise ValueError("mass_scale must be positive")


@dataclass(frozen=True)
class GateParams:
    sigma_perp_m: float = 2.5
    sigma_parallel_m: float = 4.0
    sigma_angle_deg: float = 10.0
    confidence_floor: float = 0.0
    eval_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.sigma_perp_m <= 0.0:
            raise ValueError("sigma_perp_m must be positive")
        if self.sigma_parallel_m <= 0.0:
            raise ValueError("sigma_parallel_m must be positive")
        if self.sigma_angle_deg <= 0.0:
            raise ValueError("sigma_angle_deg must be positive")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        if self.eval_chunk_size <= 0:
            raise ValueError("eval_chunk_size must be positive")


@dataclass
class Ribbon:
    """Finite trapezoidal support around a clustering representative line."""

    start: np.ndarray
    end: np.ndarray
    low_start: float
    high_start: float
    low_end: float
    high_end: float
    mass: float = 1.0

    @classmethod
    def from_segment(
        cls,
        segment: np.ndarray,
        *,
        half_width: float = 1.5,
        mass: float = 1.0,
    ) -> "Ribbon":
        points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
        vector = points[1] - points[0]
        if float(np.linalg.norm(vector)) < 1.0e-6:
            raise ValueError("support-plane segment must have nonzero length")
        if vector[0] < 0.0 or (
            abs(float(vector[0])) < 1.0e-12 and vector[1] < 0.0
        ):
            points = points[::-1]
        width = float(half_width)
        return cls(
            points[0].copy(),
            points[1].copy(),
            -width,
            width,
            -width,
            width,
            float(mass),
        )

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / self.length

    @property
    def normal(self) -> np.ndarray:
        axis = self.direction
        return np.asarray([-axis[1], axis[0]], dtype=np.float64)

    @property
    def segment(self) -> np.ndarray:
        return np.stack((self.start, self.end))

    @property
    def corners(self) -> np.ndarray:
        normal = self.normal
        return np.stack(
            (
                self.start + self.low_start * normal,
                self.start + self.high_start * normal,
                self.end + self.low_end * normal,
                self.end + self.high_end * normal,
            )
        )


def _common_axis(first: Ribbon, second: Ribbon) -> np.ndarray:
    left = first.direction
    right = second.direction
    if float(np.dot(left, right)) < 0.0:
        right = -right
    value = first.mass * left + second.mass * right
    norm = float(np.linalg.norm(value))
    axis = left if norm < 1.0e-12 else value / norm
    if axis[0] < 0.0 or (
        abs(float(axis[0])) < 1.0e-12 and axis[1] < 0.0
    ):
        axis = -axis
    return axis


def _line_value(
    first: np.ndarray,
    second: np.ndarray,
    axis: np.ndarray,
    normal: np.ndarray,
    target_u: float,
) -> float:
    u0, u1 = float(first @ axis), float(second @ axis)
    v0, v1 = float(first @ normal), float(second @ normal)
    if abs(u1 - u0) < 1.0e-9:
        return 0.5 * (v0 + v1)
    fraction = (target_u - u0) / (u1 - u0)
    return v0 + fraction * (v1 - v0)


def merge_ribbons(first: Ribbon, second: Ribbon) -> Ribbon:
    """Merge representatives like capsules and span their outer border lines."""

    axis = _common_axis(first, second)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    representative_points = np.concatenate(
        (first.segment, second.segment), axis=0
    )
    along = representative_points @ axis
    low_u, high_u = float(along.min()), float(along.max())
    representative_v = float(
        (
            first.mass * np.dot(first.midpoint, normal)
            + second.mass * np.dot(second.midpoint, normal)
        )
        / (first.mass + second.mass)
    )
    start = low_u * axis + representative_v * normal
    end = high_u * axis + representative_v * normal

    boundary_values_start: list[float] = []
    boundary_values_end: list[float] = []
    for plane in (first, second):
        corners = plane.corners
        for left, right in ((corners[0], corners[2]), (corners[1], corners[3])):
            boundary_values_start.append(
                _line_value(left, right, axis, normal, low_u)
            )
            boundary_values_end.append(
                _line_value(left, right, axis, normal, high_u)
            )

    start_center_v = float(start @ normal)
    end_center_v = float(end @ normal)
    return Ribbon(
        start=start,
        end=end,
        low_start=min(boundary_values_start) - start_center_v,
        high_start=max(boundary_values_start) - start_center_v,
        low_end=min(boundary_values_end) - end_center_v,
        high_end=max(boundary_values_end) - end_center_v,
        mass=float(first.mass + second.mass),
    )


def serialize_ribbons(ribbons: list[Ribbon]) -> tuple[RibbonRow, ...]:
    return tuple(
        (
            float(row.start[0]),
            float(row.start[1]),
            float(row.end[0]),
            float(row.end[1]),
            float(row.low_start),
            float(row.high_start),
            float(row.low_end),
            float(row.high_end),
            float(row.mass),
        )
        for row in ribbons
    )


def deserialize_ribbons(rows: tuple[RibbonRow, ...]) -> list[Ribbon]:
    return [
        Ribbon(
            np.asarray([row[0], row[1]], dtype=np.float64),
            np.asarray([row[2], row[3]], dtype=np.float64),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
            float(row[8]),
        )
        for row in rows
    ]


def ribbon_delta(
    previous: tuple[RibbonRow, ...],
    current: tuple[RibbonRow, ...],
) -> tuple[RibbonRow, ...]:
    previous_rows = set(previous)
    return tuple(row for row in current if row not in previous_rows)


def add_ribbon_vectorized(
    ribbons: list[Ribbon],
    incoming: Ribbon,
    params: RibbonParams,
    *,
    remote: bool,
) -> None:
    """Cluster by representative lines; envelope width never blocks a merge."""

    current = incoming
    while ribbons:
        starts = np.stack([row.start for row in ribbons])
        ends = np.stack([row.end for row in ribbons])
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / lengths[:, None]
        midpoints = 0.5 * (starts + ends)
        masses = np.asarray([row.mass for row in ribbons], dtype=np.float64)

        current_direction = current.direction
        dot = directions @ current_direction
        cosine = np.clip(np.abs(dot), 0.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        aligned_current = np.where(
            (dot < 0.0)[:, None], -current_direction, current_direction
        )
        axes = masses[:, None] * directions + current.mass * aligned_current
        axis_norm = np.linalg.norm(axes, axis=1)
        weak = axis_norm < 1.0e-12
        axes[~weak] /= axis_norm[~weak, None]
        axes[weak] = directions[weak]
        normals = np.stack((-axes[:, 1], axes[:, 0]), axis=1)

        lateral = np.abs(
            np.einsum("nc,nc->n", current.midpoint - midpoints, normals)
        )
        existing_projection = np.stack(
            (
                np.einsum("nc,nc->n", starts, axes),
                np.einsum("nc,nc->n", ends, axes),
            ),
            axis=1,
        )
        current_projection = current.segment @ axes.T
        first_low = existing_projection.min(axis=1)
        first_high = existing_projection.max(axis=1)
        second_low = current_projection.min(axis=0)
        second_high = current_projection.max(axis=0)
        gap = np.maximum(
            0.0,
            np.maximum(first_low, second_low)
            - np.minimum(first_high, second_high),
        )

        compatible = (
            (angle <= params.angle_deg)
            & (lateral <= params.lateral_merge_m)
            & (gap <= params.longitudinal_gap_m)
        )
        if not bool(np.any(compatible)):
            ribbons.append(current)
            return
        score = (
            angle / max(params.angle_deg, 1.0e-9)
            + lateral / max(params.lateral_merge_m, 1.0e-9)
            + gap / max(params.longitudinal_gap_m, 1.0e-9)
        )
        score[~compatible] = np.inf
        index = int(np.argmin(score))
        existing = ribbons.pop(index)
        combined = merge_ribbons(existing, current)
        if remote:
            combined.mass = max(float(existing.mass), float(current.mass))
        current = combined
    ribbons.append(current)


def remote_union(
    ribbon_sets: list[tuple[RibbonRow, ...]],
    params: RibbonParams,
) -> tuple[RibbonRow, ...]:
    result: list[Ribbon] = []
    incoming = [
        ribbon
        for rows in ribbon_sets
        for ribbon in deserialize_ribbons(rows)
    ]
    incoming.sort(
        key=lambda row: (
            float(row.midpoint[0]),
            float(row.midpoint[1]),
            float(row.length),
            float(row.mass),
        )
    )
    for ribbon in incoming:
        add_ribbon_vectorized(result, ribbon, params, remote=True)
    return serialize_ribbons(result)


class RibbonGatedMLP(nn.Module):
    """Train the MLP on samples and gate inference by support-plane borders."""

    def __init__(
        self,
        base: nn.Module,
        *,
        map_size_m: float,
        floor_prior_norm: float,
        ribbon_params: RibbonParams,
        gate_params: GateParams,
        binary_support: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.map_size_m = float(map_size_m)
        self.floor_prior_norm = float(floor_prior_norm)
        self.ribbon_params = ribbon_params
        self.gate_params = gate_params
        self.binary_support = bool(binary_support)
        self._ribbon_rows: tuple[RibbonRow, ...] = ()
        self._ribbon_tensor = torch.empty((0, 9), dtype=torch.float32)

    @property
    def ribbon_rows(self) -> tuple[RibbonRow, ...]:
        return self._ribbon_rows

    def set_ribbons(self, rows: tuple[RibbonRow, ...]) -> None:
        self._ribbon_rows = tuple(rows)
        if rows:
            self._ribbon_tensor = torch.as_tensor(
                rows,
                dtype=torch.float32,
                device=next(self.base.parameters()).device,
            )
        else:
            self._ribbon_tensor = torch.empty(
                (0, 9),
                dtype=torch.float32,
                device=next(self.base.parameters()).device,
            )

    def _confidence_chunk(self, x: torch.Tensor) -> torch.Tensor:
        ribbons = self._ribbon_tensor
        if int(ribbons.shape[0]) == 0:
            return torch.zeros(
                (int(x.shape[0]), 1), dtype=x.dtype, device=x.device
            )
        query = x[:, :4].reshape(-1, 2, 2) * self.map_size_m
        query_vector = query[:, 1] - query[:, 0]
        query_length = torch.linalg.vector_norm(
            query_vector, dim=-1
        ).clamp_min(1.0e-6)
        query_axis = query_vector / query_length[:, None]

        start = ribbons[:, 0:2]
        end = ribbons[:, 2:4]
        low_start = ribbons[:, 4]
        high_start = ribbons[:, 5]
        low_end = ribbons[:, 6]
        high_end = ribbons[:, 7]
        mass = ribbons[:, 8]
        plane_vector = end - start
        plane_length = torch.linalg.vector_norm(
            plane_vector, dim=-1
        ).clamp_min(1.0e-6)
        plane_axis = plane_vector / plane_length[:, None]
        plane_normal = torch.stack(
            (-plane_axis[:, 1], plane_axis[:, 0]), dim=1
        )
        midpoint = 0.5 * (start + end)

        cosine = torch.abs(query_axis @ plane_axis.T).clamp(0.0, 1.0)
        angle = torch.acos(cosine)
        sigma_angle = math.radians(self.gate_params.sigma_angle_deg)
        orientation = torch.exp(
            -0.5 * torch.square(angle / max(sigma_angle, 1.0e-6))
        )

        relative = query[:, :, None, :] - midpoint[None, None, :, :]
        along = torch.einsum("bemc,mc->bem", relative, plane_axis)
        lateral = torch.einsum("bemc,mc->bem", relative, plane_normal)
        fraction = (
            (along + 0.5 * plane_length[None, None, :])
            / plane_length[None, None, :]
        ).clamp(0.0, 1.0)
        low = (
            (1.0 - fraction) * low_start[None, None, :]
            + fraction * low_end[None, None, :]
        )
        high = (
            (1.0 - fraction) * high_start[None, None, :]
            + fraction * high_end[None, None, :]
        )
        if self.binary_support:
            inside_angle = angle <= sigma_angle
            inside_lateral = ((lateral >= low) & (lateral <= high)).all(
                dim=1
            )
            inside_longitudinal = (
                torch.abs(along)
                <= 0.5 * plane_length[None, None, :]
            ).all(dim=1)
            supported = (
                inside_angle & inside_lateral & inside_longitudinal
            ).any(dim=1, keepdim=True)
            return supported.to(dtype=x.dtype)
        outside_lateral = torch.relu(low - lateral) + torch.relu(
            lateral - high
        )
        d_perp = outside_lateral.amax(dim=1)
        d_parallel = torch.relu(
            torch.abs(along) - 0.5 * plane_length[None, None, :]
        ).amax(dim=1)
        spatial = torch.exp(
            -0.5
            * torch.square(d_perp / self.gate_params.sigma_perp_m)
            -0.5
            * torch.square(d_parallel / self.gate_params.sigma_parallel_m)
        )
        maturity = 1.0 - torch.exp(
            -mass / self.ribbon_params.mass_scale
        )
        confidence = (orientation * spatial * maturity[None, :]).amax(
            dim=1, keepdim=True
        )
        return confidence.clamp(
            min=float(self.gate_params.confidence_floor), max=1.0
        )

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        chunk = max(1, int(self.gate_params.eval_chunk_size))
        return torch.cat(
            [
                self._confidence_chunk(x[start : start + chunk])
                for start in range(0, int(x.shape[0]), chunk)
            ],
            dim=0,
        )

    def forward_with_confidence(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base(x)
        confidence = self.confidence(x)
        prior = torch.full_like(raw, self.floor_prior_norm)
        return prior + confidence * (raw - prior), confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.base(x)
        prediction, _confidence = self.forward_with_confidence(x)
        return prediction


def self_test() -> None:
    params = RibbonParams()
    first = Ribbon.from_segment(
        np.asarray([[0.0, 0.0], [100.0, 0.0]]),
        half_width=1.0,
        mass=5.0,
    )
    angled = Ribbon.from_segment(
        np.asarray([[0.0, 0.0], [100.0, 10.0]]),
        half_width=1.0,
        mass=5.0,
    )
    rows = remote_union(
        [serialize_ribbons([first]), serialize_ribbons([angled])], params
    )
    assert len(rows) == 1
    merged = deserialize_ribbons(rows)[0]
    assert merged.length > 99.0
    start_width = merged.high_start - merged.low_start
    end_width = merged.high_end - merged.low_end
    assert end_width > start_width

    extension = Ribbon.from_segment(
        np.asarray([[100.0, 10.0], [200.0, 10.0]]),
        half_width=1.0,
        mass=5.0,
    )
    extended_rows = remote_union(
        [rows, serialize_ribbons([extension])], params
    )
    assert len(extended_rows) == 1
    assert deserialize_ribbons(extended_rows)[0].length > 190.0

    corner = Ribbon.from_segment(
        np.asarray([[10.0, 0.0], [10.0, 20.0]]),
        half_width=1.0,
        mass=5.0,
    )
    assert len(
        remote_union([rows, serialize_ribbons([corner])], params)
    ) == 2

    base = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
    model = RibbonGatedMLP(
        base,
        map_size_m=100.0,
        floor_prior_norm=0.0,
        ribbon_params=params,
        gate_params=GateParams(),
    )
    plane = Ribbon(
        np.asarray([0.0, 0.0]),
        np.asarray([100.0, 0.0]),
        -2.0,
        2.0,
        -2.0,
        10.0,
        20.0,
    )
    model.set_ribbons(serialize_ribbons([plane]))
    inside = torch.tensor([[0.10, 0.01, 0.90, 0.01]])
    outside = torch.tensor([[0.10, 0.20, 0.90, 0.20]])
    assert float(model.confidence(inside)) > 0.95
    assert float(model.confidence(outside)) < 0.25

    binary = RibbonGatedMLP(
        nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()),
        map_size_m=100.0,
        floor_prior_norm=0.0,
        ribbon_params=params,
        gate_params=GateParams(),
        binary_support=True,
    )
    binary.set_ribbons(serialize_ribbons([plane]))
    wrong_angle = torch.tensor([[0.10, 0.01, 0.10, 0.09]])
    beyond_end = torch.tensor([[0.10, 0.01, 1.01, 0.01]])
    assert float(binary.confidence(inside)) == 1.0
    assert float(binary.confidence(outside)) == 0.0
    assert float(binary.confidence(wrong_angle)) == 0.0
    assert float(binary.confidence(beyond_end)) == 0.0
