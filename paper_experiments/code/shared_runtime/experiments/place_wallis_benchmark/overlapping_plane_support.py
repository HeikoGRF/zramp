"""Bounded overlapping straight support planes for positive radio links."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn

from experiments.place_wallis_benchmark.ribbon_support import GateParams


PlaneRow = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float,
]


@dataclass(frozen=True)
class OverlappingPlaneParams:
    """Straight-corridor merge parameters.

    Partly overlapping supports may share one fitted direction while their
    accumulated deviation stays below angle_deg and their physical envelope
    stays below max_corridor_width_m.
    """

    angle_deg: float = 7.0
    lateral_merge_m: float = 1.0
    longitudinal_gap_m: float = 3.0
    initial_half_width_m: float = 0.0
    mass_scale: float = 3.0
    max_envelope_inflation: float = 1.20
    max_corridor_width_m: float = 12.0
    link_length_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.angle_deg < 90.0:
            raise ValueError("angle_deg must be in (0, 90)")
        if self.lateral_merge_m < 0.0:
            raise ValueError("lateral_merge_m cannot be negative")
        if self.longitudinal_gap_m < 0.0:
            raise ValueError("longitudinal_gap_m cannot be negative")
        if self.initial_half_width_m < 0.0:
            raise ValueError("initial_half_width_m cannot be negative")
        if self.mass_scale <= 0.0:
            raise ValueError("mass_scale must be positive")
        if self.max_envelope_inflation < 1.0:
            raise ValueError("max_envelope_inflation must be at least one")
        if self.max_corridor_width_m <= 0.0:
            raise ValueError("max_corridor_width_m must be positive")
        if self.link_length_margin_m < 0.0:
            raise ValueError("link_length_margin_m cannot be negative")


@dataclass
class OverlappingPlane:
    """One fixed-direction finite trapezoid in an overlapping cover."""

    start: np.ndarray
    end: np.ndarray
    low_start: float
    high_start: float
    low_end: float
    high_end: float
    mass: float = 1.0
    max_link_length: float = 0.0
    angle_spread_deg: float = 0.0

    @classmethod
    def from_segment(
        cls,
        segment: np.ndarray,
        *,
        half_width: float = 0.0,
        mass: float = 1.0,
    ) -> "OverlappingPlane":
        points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
        length = float(np.linalg.norm(points[1] - points[0]))
        if length < 1.0e-6:
            raise ValueError("support-plane segment must have nonzero length")
        if points[1, 0] < points[0, 0] or (
            abs(float(points[1, 0] - points[0, 0])) < 1.0e-12
            and points[1, 1] < points[0, 1]
        ):
            points = points[::-1]
        width = float(half_width)
        return cls(
            points[0].copy(), points[1].copy(),
            -width, width, -width, width,
            float(mass), length, 0.0,
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
        return np.stack((
            self.start + self.low_start * normal,
            self.start + self.high_start * normal,
            self.end + self.low_end * normal,
            self.end + self.high_end * normal,
        ))

    @property
    def area(self) -> float:
        start_width = max(0.0, self.high_start - self.low_start)
        end_width = max(0.0, self.high_end - self.low_end)
        return 0.5 * (start_width + end_width) * self.length


def _axial_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(float(left @ right)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

def _fitted_axis_and_spread(
    first: OverlappingPlane,
    second: OverlappingPlane,
) -> tuple[np.ndarray, float]:
    """Fit one axial direction and conservatively carry prior deviation."""

    first_axis = first.direction
    second_axis = second.direction
    if float(first_axis @ second_axis) < 0.0:
        second_axis = -second_axis
    first_weight = max(1.0, float(first.mass)) * max(1.0, float(first.length))
    second_weight = max(1.0, float(second.mass)) * max(1.0, float(second.length))
    vector = first_weight * first_axis + second_weight * second_axis
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        axis = first_axis.copy()
    else:
        axis = vector / norm
    if axis[0] < 0.0 or (
        abs(float(axis[0])) < 1.0e-12 and axis[1] < 0.0
    ):
        axis = -axis
    spread = max(
        float(first.angle_spread_deg)
        + _axial_angle_deg(first.direction, axis),
        float(second.angle_spread_deg)
        + _axial_angle_deg(second.direction, axis),
    )
    return axis, float(spread)



def _anchor_key(plane: OverlappingPlane) -> tuple[float, ...]:
    angle = math.atan2(float(plane.direction[1]), float(plane.direction[0]))
    return (
        float(plane.mass), float(plane.length), -abs(angle),
        -float(plane.midpoint[0]), -float(plane.midpoint[1]),
    )


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


def _projected_interval(
    points: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    values = points @ axis
    return float(values.min()), float(values.max())


def _interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, max(first[0], second[0]) - min(first[1], second[1]))


def _interval_overlaps(
    first: tuple[float, float], second: tuple[float, float]
) -> bool:
    return min(first[1], second[1]) >= max(first[0], second[0]) - 1.0e-8


def _corridor_width(plane: OverlappingPlane) -> float:
    return max(
        float(plane.high_start - plane.low_start),
        float(plane.high_end - plane.low_end),
    )


def _merge_candidate(
    first: OverlappingPlane,
    second: OverlappingPlane,
    params: OverlappingPlaneParams,
    *,
    remote: bool,
) -> OverlappingPlane | None:
    axis, spread = _fitted_axis_and_spread(first, second)
    if spread > float(params.angle_deg) + 1.0e-8:
        return None
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    first_u = _projected_interval(first.segment, axis)
    second_u = _projected_interval(second.segment, axis)
    if not _interval_overlaps(first_u, second_u):
        return None

    support_corners = np.concatenate((first.corners, second.corners), axis=0)
    along = support_corners @ axis
    lateral = support_corners @ normal
    low_u, high_u = float(along.min()), float(along.max())
    low_v, high_v = float(lateral.min()), float(lateral.max())
    center_v = 0.5 * (low_v + high_v)
    start = low_u * axis + center_v * normal
    end = high_u * axis + center_v * normal
    low_offset = low_v - center_v
    high_offset = high_v - center_v
    combined = OverlappingPlane(
        start=start,
        end=end,
        low_start=low_offset,
        high_start=high_offset,
        low_end=low_offset,
        high_end=high_offset,
        mass=(
            max(float(first.mass), float(second.mass))
            if remote
            else float(first.mass + second.mass)
        ),
        max_link_length=max(
            float(first.max_link_length), float(second.max_link_length)
        ),
        angle_spread_deg=float(spread),
    )
    if _corridor_width(combined) > params.max_corridor_width_m + 1.0e-8:
        return None
    return combined


def serialize_planes(planes: list[OverlappingPlane]) -> tuple[PlaneRow, ...]:
    return tuple((
        float(row.start[0]), float(row.start[1]),
        float(row.end[0]), float(row.end[1]),
        float(row.low_start), float(row.high_start),
        float(row.low_end), float(row.high_end),
        float(row.mass), float(row.max_link_length),
        float(row.angle_spread_deg),
    ) for row in planes)


def deserialize_planes(rows: tuple[PlaneRow, ...]) -> list[OverlappingPlane]:
    return [OverlappingPlane(
        np.asarray([row[0], row[1]], dtype=np.float64),
        np.asarray([row[2], row[3]], dtype=np.float64),
        float(row[4]), float(row[5]), float(row[6]), float(row[7]),
        float(row[8]), float(row[9]), float(row[10]),
    ) for row in rows]


def plane_delta(
    previous: tuple[PlaneRow, ...], current: tuple[PlaneRow, ...]
) -> tuple[PlaneRow, ...]:
    previous_rows = set(previous)
    return tuple(row for row in current if row not in previous_rows)


def _point_inside(plane: OverlappingPlane, point: np.ndarray) -> bool:
    relative = point - plane.midpoint
    along = float(relative @ plane.direction)
    if abs(along) > 0.5 * plane.length + 1.0e-8:
        return False
    fraction = (along + 0.5 * plane.length) / plane.length
    low = (1.0 - fraction) * plane.low_start + fraction * plane.low_end
    high = (1.0 - fraction) * plane.high_start + fraction * plane.high_end
    lateral = float(relative @ plane.normal)
    return low - 1.0e-8 <= lateral <= high + 1.0e-8


def _screen_planes(
    planes: list[OverlappingPlane],
    current: OverlappingPlane,
    params: OverlappingPlaneParams,
    *,
    remote: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized containment and corridor-envelope prefilter."""

    starts = np.stack([row.start for row in planes])
    ends = np.stack([row.end for row in planes])
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    axes = vectors / lengths[:, None]
    normals = np.stack((-axes[:, 1], axes[:, 0]), axis=1)
    midpoints = 0.5 * (starts + ends)
    low_start = np.asarray([row.low_start for row in planes])
    high_start = np.asarray([row.high_start for row in planes])
    low_end = np.asarray([row.low_end for row in planes])
    high_end = np.asarray([row.high_end for row in planes])

    contained_points = current.corners if remote else current.segment
    relative = contained_points[:, None, :] - midpoints[None, :, :]
    along = np.einsum("pnc,nc->pn", relative, axes)
    lateral = np.einsum("pnc,nc->pn", relative, normals)
    fraction = np.clip(
        (along + 0.5 * lengths[None, :]) / lengths[None, :], 0.0, 1.0
    )
    low = (
        (1.0 - fraction) * low_start[None, :]
        + fraction * low_end[None, :]
    )
    high = (
        (1.0 - fraction) * high_start[None, :]
        + fraction * high_end[None, :]
    )
    contained = (
        (np.abs(along) <= 0.5 * lengths[None, :] + 1.0e-8)
        & (lateral >= low - 1.0e-8)
        & (lateral <= high + 1.0e-8)
    ).all(axis=0)

    current_relative_segment = current.segment[:, None, :] - midpoints[None, :, :]
    current_u = np.einsum("pnc,nc->pn", current_relative_segment, axes)
    overlap_on_existing = (
        np.minimum(0.5 * lengths, current_u.max(axis=0))
        >= np.maximum(-0.5 * lengths, current_u.min(axis=0)) - 1.0e-8
    )
    current_axis = current.direction
    existing_relative_segment = np.stack((starts, ends), axis=1) - current.midpoint
    existing_u = np.einsum("npc,c->np", existing_relative_segment, current_axis)
    overlap_on_current = (
        np.minimum(0.5 * current.length, existing_u.max(axis=1))
        >= np.maximum(-0.5 * current.length, existing_u.min(axis=1)) - 1.0e-8
    )
    compatible = overlap_on_existing | overlap_on_current
    return np.flatnonzero(contained), np.flatnonzero(compatible)


def add_plane(
    planes: list[OverlappingPlane],
    incoming: OverlappingPlane,
    params: OverlappingPlaneParams,
    *,
    remote: bool,
) -> None:
    current = incoming
    while planes:
        contained, compatible = _screen_planes(
            planes, current, params, remote=remote
        )
        if len(contained):
            index = min(
                (int(value) for value in contained),
                key=lambda value: (planes[value].area, value),
            )
            existing = planes[index]
            existing.mass = (
                max(float(existing.mass), float(current.mass))
                if remote
                else float(existing.mass + current.mass)
            )
            existing.max_link_length = max(
                float(existing.max_link_length),
                float(current.max_link_length),
            )
            planes[index] = existing
            return
        candidates: list[tuple[float, int, OverlappingPlane]] = []
        for index in compatible:
            existing = planes[int(index)]
            combined = _merge_candidate(existing, current, params, remote=remote)
            if combined is None:
                continue
            width = _corridor_width(combined)
            candidates.append((width, int(index), combined))
        if not candidates:
            planes.append(current)
            return
        _score, index, current = min(candidates, key=lambda row: (row[0], row[1]))
        planes.pop(index)
    planes.append(current)


def remote_union(
    plane_sets: list[tuple[PlaneRow, ...]],
    params: OverlappingPlaneParams,
) -> tuple[PlaneRow, ...]:
    if not plane_sets:
        return ()
    result = deserialize_planes(plane_sets[0])
    seen = set(plane_sets[0])
    unique = []
    for rows in plane_sets[1:]:
        for row in rows:
            if row not in seen:
                seen.add(row)
                unique.append(row)
    incoming = deserialize_planes(tuple(unique))
    incoming.sort(key=lambda row: (
        -float(row.mass), float(row.midpoint[0]),
        float(row.midpoint[1]), float(row.length),
    ))
    for plane in incoming:
        add_plane(result, plane, params, remote=True)
    return serialize_planes(result)


def remote_union_with_sources(
    plane_sets: list[tuple[object, tuple[PlaneRow, ...]]],
    params: OverlappingPlaneParams,
) -> tuple[tuple[PlaneRow, ...], tuple[frozenset[object], ...]]:
    """Merge remote plane sets while retaining exact source provenance.

    The geometry and remote mass update are identical to :func:`add_plane`.
    When planes are contained or merged, the resulting plane inherits the
    union of the sources that contributed to either input plane.
    """

    incoming = [
        (plane, frozenset((source,)))
        for source, rows in plane_sets
        for plane in deserialize_planes(rows)
    ]
    incoming.sort(key=lambda item: (
        -float(item[0].mass),
        float(item[0].midpoint[0]),
        float(item[0].midpoint[1]),
        float(item[0].length),
        repr(next(iter(item[1]))),
    ))
    planes: list[OverlappingPlane] = []
    sources: list[frozenset[object]] = []
    for incoming_plane, incoming_sources in incoming:
        current = incoming_plane
        current_sources = incoming_sources
        while planes:
            contained, compatible = _screen_planes(
                planes, current, params, remote=True
            )
            if len(contained):
                index = min(
                    (int(value) for value in contained),
                    key=lambda value: (planes[value].area, value),
                )
                existing = planes[index]
                existing.mass = max(
                    float(existing.mass), float(current.mass)
                )
                existing.max_link_length = max(
                    float(existing.max_link_length),
                    float(current.max_link_length),
                )
                planes[index] = existing
                sources[index] = sources[index] | current_sources
                break
            candidates: list[
                tuple[float, int, OverlappingPlane]
            ] = []
            for value in compatible:
                index = int(value)
                combined = _merge_candidate(
                    planes[index], current, params, remote=True
                )
                if combined is None:
                    continue
                candidates.append(
                    (_corridor_width(combined), index, combined)
                )
            if not candidates:
                planes.append(current)
                sources.append(current_sources)
                break
            _score, index, current = min(
                candidates, key=lambda row: (row[0], row[1])
            )
            current_sources = sources.pop(index) | current_sources
            planes.pop(index)
        else:
            planes.append(current)
            sources.append(current_sources)
    return serialize_planes(planes), tuple(sources)


class OverlappingPlaneGatedMLP(nn.Module):
    """MLP with hard support from one complete local straight plane."""

    def __init__(
        self,
        base: nn.Module,
        *,
        map_size_m: float,
        floor_prior_norm: float,
        ribbon_params: OverlappingPlaneParams,
        gate_params: GateParams,
        binary_support: bool = True,
    ) -> None:
        super().__init__()
        if not binary_support:
            raise ValueError("overlapping planes currently require binary support")
        self.base = base
        self.map_size_m = float(map_size_m)
        self.floor_prior_norm = float(floor_prior_norm)
        self.ribbon_params = ribbon_params
        self.gate_params = gate_params
        self.binary_support = True
        self._ribbon_rows: tuple[PlaneRow, ...] = ()
        self._ribbon_tensor = torch.empty((0, 11), dtype=torch.float32)

    @property
    def ribbon_rows(self) -> tuple[PlaneRow, ...]:
        return self._ribbon_rows

    def set_ribbons(self, rows: tuple[PlaneRow, ...]) -> None:
        self._ribbon_rows = tuple(rows)
        device = next(self.base.parameters()).device
        self._ribbon_tensor = (
            torch.as_tensor(rows, dtype=torch.float32, device=device)
            if rows
            else torch.empty((0, 11), dtype=torch.float32, device=device)
        )

    def _confidence_chunk(self, x: torch.Tensor) -> torch.Tensor:
        planes = self._ribbon_tensor
        if int(planes.shape[0]) == 0:
            return torch.zeros((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
        query = x[:, :4].reshape(-1, 2, 2) * self.map_size_m
        query_length = torch.linalg.vector_norm(query[:, 1] - query[:, 0], dim=1)
        start, end = planes[:, 0:2], planes[:, 2:4]
        vector = end - start
        length = torch.linalg.vector_norm(vector, dim=1).clamp_min(1.0e-6)
        axis = vector / length[:, None]
        normal = torch.stack((-axis[:, 1], axis[:, 0]), dim=1)
        midpoint = 0.5 * (start + end)
        relative = query[:, :, None, :] - midpoint[None, None, :, :]
        along = torch.einsum("bemc,mc->bem", relative, axis)
        lateral = torch.einsum("bemc,mc->bem", relative, normal)
        fraction = ((along + 0.5 * length[None, None, :]) / length[None, None, :]).clamp(0.0, 1.0)
        low = (1.0 - fraction) * planes[None, None, :, 4] + fraction * planes[None, None, :, 6]
        high = (1.0 - fraction) * planes[None, None, :, 5] + fraction * planes[None, None, :, 7]
        inside_lateral = ((lateral >= low) & (lateral <= high)).all(dim=1)
        inside_longitudinal = (torch.abs(along) <= 0.5 * length[None, None, :]).all(dim=1)
        inside_length = query_length[:, None] <= (
            planes[None, :, 9] + self.ribbon_params.link_length_margin_m
        )
        supported = (inside_lateral & inside_longitudinal & inside_length).any(dim=1, keepdim=True)
        return supported.to(dtype=x.dtype)

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        chunk = max(1, int(self.gate_params.eval_chunk_size))
        return torch.cat([
            self._confidence_chunk(x[start:start + chunk])
            for start in range(0, int(x.shape[0]), chunk)
        ], dim=0)

    def forward_with_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    params = OverlappingPlaneParams()
    horizontal = OverlappingPlane.from_segment(np.asarray([[0.0, 0.0], [40.0, 0.0]]))
    assert horizontal.area == 0.0
    assert horizontal.low_start == horizontal.high_start == 0.0
    assert horizontal.low_end == horizontal.high_end == 0.0
    parallel = OverlappingPlane.from_segment(np.asarray([[0.0, 3.5], [40.0, 3.5]]))
    widened = deserialize_planes(remote_union([
        serialize_planes([horizontal]), serialize_planes([parallel])
    ], params))
    assert len(widened) == 1
    assert 3.49 <= widened[0].high_start - widened[0].low_start <= 3.51
    previous_mass = widened[0].mass
    add_plane(
        widened,
        OverlappingPlane.from_segment(
            np.asarray([[20.0, 0.0], [20.0, 3.5]])
        ),
        params,
        remote=False,
    )
    assert len(widened) == 1
    assert widened[0].mass > previous_mass

    straight = []
    for offset in range(0, 100, 20):
        add_plane(straight, OverlappingPlane.from_segment(
            np.asarray([[float(offset), 0.0], [float(offset + 30), 0.0]])
        ), params, remote=False)
    assert len(straight) == 1
    assert straight[0].length >= 100.0
    assert straight[0].max_link_length <= 31.0

    disjoint: list[OverlappingPlane] = []
    for start in (0.0, 40.0):
        add_plane(disjoint, OverlappingPlane.from_segment(
            np.asarray([[start, 0.0], [start + 30.0, 0.0]])
        ), params, remote=False)
    assert len(disjoint) == 2

    curved: list[OverlappingPlane] = []
    for degrees in (0.0, 10.0, 20.0, 30.0):
        radians = math.radians(degrees)
        start = np.asarray([2.0 * degrees, 0.0])
        end = start + 30.0 * np.asarray([math.cos(radians), math.sin(radians)])
        add_plane(curved, OverlappingPlane.from_segment(np.stack((start, end))), params, remote=False)
    assert 2 <= len(curved) <= 4

    model = OverlappingPlaneGatedMLP(
        nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()),
        map_size_m=100.0,
        floor_prior_norm=0.0,
        ribbon_params=params,
        gate_params=GateParams(),
    )
    model.set_ribbons(serialize_planes([horizontal]))
    exact_singleton = torch.tensor([[0.10, 0.0, 0.30, 0.0]])
    offset_singleton = torch.tensor([[0.10, 0.001, 0.30, 0.001]])
    assert float(model.confidence(exact_singleton)) == 1.0
    assert float(model.confidence(offset_singleton)) == 0.0

    model.set_ribbons(serialize_planes(widened))
    orthogonal = torch.tensor([[0.20, 0.01, 0.20, 0.025]])
    too_long = torch.tensor([[0.05, 0.01, 0.95, 0.01]])
    assert float(model.confidence(orthogonal)) == 1.0
    assert float(model.confidence(too_long)) == 0.0
