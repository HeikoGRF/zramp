#!/usr/bin/env python3
"""Support-driven expert banks on the Place Wallis benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = CODE_ROOT / "final"
SHARED_RUNTIME_ROOT = CODE_ROOT / "shared_runtime"
for import_root in (ROOT, SHARED_RUNTIME_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.place_wallis_benchmark import run_capsule_greedy as plane_runner  # noqa: E402
from experiments.place_wallis_benchmark.overlapping_plane_support import (  # noqa: E402
    OverlappingPlane as Capsule,
    OverlappingPlaneGatedMLP as CapsuleGatedMLP,
    OverlappingPlaneParams as CapsuleParams,
    PlaneRow as CapsuleRow,
    add_plane as add_capsule_vectorized,
    deserialize_planes as deserialize_capsules,
    serialize_planes as serialize_capsules,
    remote_union,
    remote_union_with_sources,
    self_test as overlapping_plane_self_test,
)
from experiments.place_wallis_benchmark.ribbon_support import GateParams  # noqa: E402
from experiments.place_wallis_benchmark.run_equal_greedy import (  # noqa: E402
    DEFAULT_NET,
    DEFAULT_TESTSET,
    DEFAULT_TRACE,
    atomic_json,
    validate_dataset,
)
from experiments.place_wallis_benchmark.training_utils import (  # noqa: E402
    ReplayBuffer,
    TrainingParams,
)
from experiments.place_wallis_benchmark.tail_metrics import (  # noqa: E402
    make_tail_evaluation_steps,
    temporal_metric_summary,
)
from experiments.support_acquisition_pretraining.pretrain import (  # noqa: E402
    AcquisitionModel as PretrainedAcquisitionModel,
    PLANE_FEATURE_SCHEMA_V2,
    PlaneSetEncoder as PretrainedPlaneSetEncoder,
    normalize_expert as normalize_planes_for_encoder,
)
from experiments.support_acquisition_pretraining.union_gain_model import (  # noqa: E402
    PlaneSetEncoder as UnionPlaneSetEncoder,
    SCALAR_PLANE_FEATURE_SCHEMA,
    UnionGainModel,
    normalize_plane_set as normalize_planes_for_union_encoder,
)
from experiments.support_acquisition_pretraining.shared_frame_gain_model import (  # noqa: E402
    EncodingOnlyGainModel,
    SHARED_FRAME_PLANE_FEATURE_SCHEMA,
    normalize_plane_set_shared_frame,
)
from experiments.support_acquisition_pretraining.spatial_grid_gain_model import (  # noqa: E402
    SpatialGridEncoder,
    SpatialGridGainModel,
)
from experiments.support_acquisition_pretraining.grid_autoencoder_gain_model import (  # noqa: E402
    GridAutoencoder,
    GridEncodingGainModel,
)
from experiments.support_acquisition_pretraining.patch_grid_codec_model import (  # noqa: E402
    PatchGridCodec,
    PatchGridGainModel,
)
from experiments.support_acquisition_pretraining.grid_gain import (  # noqa: E402
    GRID_LAYOUT_STAGGERED,
    grid_support_counts,
    relative_point_grid_gain,
    self_test as grid_gain_self_test,
    unit_square_point_grid,
)
from rl_reward_experiment.config import build_config_from_env  # noqa: E402


plane_runner.Capsule = Capsule
plane_runner.CapsuleGatedMLP = CapsuleGatedMLP
plane_runner.CapsuleParams = CapsuleParams
plane_runner.CapsuleRow = CapsuleRow
plane_runner.add_capsule_vectorized = add_capsule_vectorized
plane_runner.deserialize_capsules = deserialize_capsules
plane_runner.serialize_capsules = serialize_capsules
plane_runner.SUPPORT_VARIANT = "overlapping-straight-planes"
plane_runner.SUPPORT_RECORD_FLOATS = 11
CapsuleGreedySimulation = plane_runner.CapsuleGreedySimulation


DEFAULT_RESULTS = (
    ROOT
    / "artifacts/place_wallis_benchmark/methods/support_expert_bank_k6_eval50_tail10x25"
)
ExpertKey = tuple[int, int, int]


@dataclass(frozen=True)
class ExpertRecord:
    """One immutable model version and the support on which it was trained."""

    key: ExpertKey
    experience: int
    capsules: tuple[CapsuleRow, ...]
    model_state: dict[str, torch.Tensor]

    @property
    def lineage(self) -> tuple[int, int]:
        return self.key[:2]


@dataclass(frozen=True)
class EncodedExpertAdvertisement:
    """Encoder-ready support storage and the compact transmitted summary."""

    key: ExpertKey
    normalized_planes: np.ndarray
    encoding: np.ndarray
    center_xy_m: np.ndarray | None
    scale_m: float | None
    experience: int


@dataclass(frozen=True)
class MergedBankSupport:
    """One vehicle-level plane union and its contributing experts."""

    capsules: tuple[CapsuleRow, ...]
    contributors: tuple[frozenset[ExpertKey], ...]


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def support_strength_matrix(
    rows: tuple[CapsuleRow, ...],
    probes_m: np.ndarray,
    params: CapsuleParams,
    *,
    raw_counts: bool = False,
) -> np.ndarray:
    """Return per-plane maturity or sample-count support for links."""

    probes = np.asarray(probes_m, dtype=np.float64).reshape(-1, 2, 2)
    if not rows:
        return np.zeros((len(probes), 0), dtype=np.float64)
    values = np.asarray(rows, dtype=np.float64)
    start, end = values[:, 0:2], values[:, 2:4]
    vector = end - start
    length = np.linalg.norm(vector, axis=1).clip(min=1.0e-9)
    axis = vector / length[:, None]
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    midpoint = 0.5 * (start + end)
    relative = probes[:, :, None, :] - midpoint[None, None, :, :]
    along = np.einsum("qepc,pc->qep", relative, axis)
    lateral = np.einsum("qepc,pc->qep", relative, normal)
    fraction = np.clip(
        (along + 0.5 * length[None, None, :])
        / length[None, None, :],
        0.0,
        1.0,
    )
    low = (
        (1.0 - fraction) * values[None, None, :, 4]
        + fraction * values[None, None, :, 6]
    )
    high = (
        (1.0 - fraction) * values[None, None, :, 5]
        + fraction * values[None, None, :, 7]
    )
    query_length = np.linalg.norm(probes[:, 1] - probes[:, 0], axis=1)
    supported = (
        ((lateral >= low) & (lateral <= high)).all(axis=1)
        & (np.abs(along) <= 0.5 * length[None, None, :]).all(axis=1)
        & (
            query_length[:, None]
            <= values[None, :, 9] + float(params.link_length_margin_m)
        )
    )
    strength = (
        np.maximum(values[:, 8], 0.0)
        if raw_counts
        else 1.0 - np.exp(-values[:, 8] / float(params.mass_scale))
    )
    return np.where(supported, strength[None, :], 0.0)


def support_profile(
    rows: tuple[CapsuleRow, ...],
    probes_m: np.ndarray,
    params: CapsuleParams,
    *,
    raw_counts: bool = False,
) -> np.ndarray:
    """Return maximum maturity or sample-count support for supplied links."""

    matrix = support_strength_matrix(
        rows, probes_m, params, raw_counts=raw_counts
    )
    return matrix.max(axis=1) if matrix.shape[1] else np.zeros(len(matrix))


def marginal_support_gain(candidate: np.ndarray, bank: np.ndarray) -> float:
    return float(np.mean(np.maximum(candidate, bank) - bank))


def binary_coverage_mask(
    profile: np.ndarray, *, threshold: float = 0.5
) -> int:
    """Pack probe coverage into one integer for fast set operations."""

    packed = np.packbits(
        np.asarray(profile) >= float(threshold), bitorder="little"
    )
    return int.from_bytes(packed.tobytes(), byteorder="little")


@dataclass(frozen=True)
class PlaneDominanceGeometry:
    """Cached vector geometry used by exact support-dominance checks."""

    corners: np.ndarray
    bbox_low: np.ndarray
    bbox_high: np.ndarray
    midpoint: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    low_start: np.ndarray
    high_start: np.ndarray
    low_end: np.ndarray
    high_end: np.ndarray
    mass: np.ndarray
    max_link_length: np.ndarray

    @property
    def count(self) -> int:
        return int(self.length.size)


def plane_dominance_geometry(
    rows: tuple[CapsuleRow, ...] | np.ndarray,
) -> PlaneDominanceGeometry:
    """Precompute exact convex-plane geometry without deployment probes."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    if len(values) == 0:
        empty_2 = np.empty((0, 2), dtype=np.float64)
        return PlaneDominanceGeometry(
            corners=np.empty((0, 4, 2), dtype=np.float64),
            bbox_low=empty_2.copy(),
            bbox_high=empty_2.copy(),
            midpoint=empty_2.copy(),
            axis=empty_2.copy(),
            normal=empty_2.copy(),
            length=np.empty(0, dtype=np.float64),
            low_start=np.empty(0, dtype=np.float64),
            high_start=np.empty(0, dtype=np.float64),
            low_end=np.empty(0, dtype=np.float64),
            high_end=np.empty(0, dtype=np.float64),
            mass=np.empty(0, dtype=np.float64),
            max_link_length=np.empty(0, dtype=np.float64),
        )
    start, end = values[:, 0:2], values[:, 2:4]
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    if bool(np.any(length < 1.0e-9)):
        raise ValueError("geometric dominance requires nonzero plane lengths")
    axis = vector / length[:, None]
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    corners = np.stack(
        (
            start + values[:, 4, None] * normal,
            start + values[:, 5, None] * normal,
            end + values[:, 6, None] * normal,
            end + values[:, 7, None] * normal,
        ),
        axis=1,
    )
    return PlaneDominanceGeometry(
        corners=corners,
        bbox_low=corners.min(axis=1),
        bbox_high=corners.max(axis=1),
        midpoint=0.5 * (start + end),
        axis=axis,
        normal=normal,
        length=length,
        low_start=values[:, 4],
        high_start=values[:, 5],
        low_end=values[:, 6],
        high_end=values[:, 7],
        mass=values[:, 8],
        max_link_length=values[:, 9],
    )


def geometrically_dominated_plane_mask(
    target: PlaneDominanceGeometry,
    dominator: PlaneDominanceGeometry,
) -> int:
    """Return target-plane bits dominated by individual dominator planes.

    A target plane is dominated only when one dominator plane contains all
    four target corners and has at least its link-length and sample support.
    The fixed chunk size affects memory use only, not the result.
    """

    if target.count == 0:
        return 0
    if dominator.count == 0:
        return 0
    dominated = np.zeros(target.count, dtype=np.uint8)
    tolerance = 1.0e-8
    for start in range(0, target.count, 64):
        stop = min(target.count, start + 64)
        target_corners = target.corners[start:stop]
        eligible = (
            (
                dominator.mass[None, :]
                >= target.mass[start:stop, None] - tolerance
            )
            & (
                dominator.max_link_length[None, :]
                >= target.max_link_length[start:stop, None] - tolerance
            )
            & (
                dominator.bbox_low[None, :, :]
                <= target.bbox_low[start:stop, None, :] + tolerance
            ).all(axis=2)
            & (
                dominator.bbox_high[None, :, :]
                >= target.bbox_high[start:stop, None, :] - tolerance
            ).all(axis=2)
        )
        if not bool(np.any(eligible)):
            continue
        target_indices, dominator_indices = np.nonzero(eligible)
        pair_corners = target_corners[target_indices]
        relative = (
            pair_corners
            - dominator.midpoint[dominator_indices, None, :]
        )
        pair_axis = dominator.axis[dominator_indices]
        pair_normal = dominator.normal[dominator_indices]
        pair_length = dominator.length[dominator_indices, None]
        along = np.einsum("ntc,nc->nt", relative, pair_axis)
        lateral = np.einsum("ntc,nc->nt", relative, pair_normal)
        fraction = np.clip(
            (along + 0.5 * pair_length) / pair_length,
            0.0,
            1.0,
        )
        low = (
            (1.0 - fraction)
            * dominator.low_start[dominator_indices, None]
            + fraction * dominator.low_end[dominator_indices, None]
        )
        high = (
            (1.0 - fraction)
            * dominator.high_start[dominator_indices, None]
            + fraction * dominator.high_end[dominator_indices, None]
        )
        inside = (
            (np.abs(along) <= 0.5 * pair_length + tolerance)
            & (lateral >= low - tolerance)
            & (lateral <= high + tolerance)
        ).all(axis=1)
        if bool(np.any(inside)):
            dominated[
                start + np.unique(target_indices[inside])
            ] = 1
    packed = np.packbits(dominated, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def plane_set_weighted_geometry(
    rows: tuple[CapsuleRow, ...] | np.ndarray,
    *,
    map_size_m: float,
) -> tuple[tuple[float, BaseGeometry], ...]:
    """Group normalized plane polygons by raw sample-count intensity."""

    if map_size_m <= 0.0:
        raise ValueError("map_size_m must be positive")
    geometry = plane_dominance_geometry(rows)
    polygons_by_intensity: dict[float, list[Polygon]] = defaultdict(list)
    for corners, mass in zip(geometry.corners, geometry.mass):
        # plane_dominance_geometry stores start-low, start-high, end-low,
        # end-high. Reorder them around the convex boundary.
        boundary = corners[[0, 2, 3, 1]] / float(map_size_m)
        polygon = _polygonal_geometry(Polygon(boundary))
        if not polygon.is_empty and float(polygon.area) > 1.0e-14:
            polygons_by_intensity[max(0.0, float(mass))].append(polygon)
    unit_square = box(0.0, 0.0, 1.0, 1.0)
    return tuple(
        (
            intensity,
            _polygonal_geometry(
                unary_union(polygons).intersection(unit_square)
            ),
        )
        for intensity, polygons in sorted(
            polygons_by_intensity.items(), reverse=True
        )
        if intensity > 0.0
    )


def _polygonal_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        parts: list[Polygon] = []
        for child in geometry.geoms:
            parts.extend(_polygonal_parts(child))
        return parts
    return []


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Repair topology and retain exactly the positive-area polygonal part."""

    if geometry.is_empty:
        return GeometryCollection()
    if geometry.is_valid and isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    repaired = geometry if geometry.is_valid else make_valid(geometry)
    parts = [
        part for part in _polygonal_parts(repaired)
        if float(part.area) > 1.0e-14
    ]
    if not parts:
        return GeometryCollection()
    combined: BaseGeometry = parts[0] if len(parts) == 1 else unary_union(parts)
    if not combined.is_valid:
        combined = make_valid(combined)
        valid_parts = [
            part for part in _polygonal_parts(combined)
            if float(part.area) > 1.0e-14
        ]
        if not valid_parts:
            return GeometryCollection()
        combined = (
            valid_parts[0]
            if len(valid_parts) == 1
            else unary_union(valid_parts)
        )
    return combined


@dataclass
class WeightedSupportEnvelope:
    """Disjoint exact regions carrying their maximum sample-count intensity."""

    regions: tuple[tuple[float, BaseGeometry], ...]
    score: float
    _tree: STRtree | None = None

    def spatial_index(self) -> STRtree | None:
        if self._tree is None and self.regions:
            self._tree = STRtree([geometry for _weight, geometry in self.regions])
        return self._tree


def weighted_support_envelope(
    plane_groups: list[tuple[tuple[float, BaseGeometry], ...]],
) -> WeightedSupportEnvelope:
    """Build disjoint maximum-intensity regions from grouped plane polygons."""

    geometries_by_intensity: dict[float, list[BaseGeometry]] = defaultdict(list)
    for groups in plane_groups:
        for intensity, geometry in groups:
            if intensity > 0.0 and not geometry.is_empty:
                geometries_by_intensity[float(intensity)].append(geometry)
    covered: BaseGeometry = GeometryCollection()
    regions: list[tuple[float, BaseGeometry]] = []
    score = 0.0
    for intensity in sorted(geometries_by_intensity, reverse=True):
        level = _polygonal_geometry(
            unary_union(geometries_by_intensity[intensity])
        )
        newly_covered = (
            level
            if covered.is_empty
            else _polygonal_geometry(level.difference(covered))
        )
        if not newly_covered.is_empty:
            regions.append((float(intensity), newly_covered))
            score += float(intensity) * float(newly_covered.area)
        covered = (
            level
            if covered.is_empty
            else _polygonal_geometry(unary_union((covered, level)))
        )
    return WeightedSupportEnvelope(tuple(regions), max(0.0, score))


def weighted_geometric_gain(
    bank: WeightedSupportEnvelope,
    candidate: WeightedSupportEnvelope,
) -> tuple[float, float]:
    """Return relative and absolute gain over the exact pointwise bank maximum."""

    if candidate.score <= 1.0e-14:
        return 0.0, 0.0
    if bank.score <= 1.0e-14:
        return float("inf"), float(candidate.score)
    tree = bank.spatial_index()
    increase = 0.0
    for candidate_intensity, candidate_geometry in candidate.regions:
        indices = (
            np.asarray(tree.query(candidate_geometry), dtype=np.int64)
            if tree is not None
            else np.empty(0, dtype=np.int64)
        )
        indices = np.asarray(sorted(
            (int(index) for index in indices),
            key=lambda index: bank.regions[index][0],
            reverse=True,
        ), dtype=np.int64)
        covered: BaseGeometry = GeometryCollection()
        for index in indices:
            bank_intensity, bank_geometry = bank.regions[int(index)]
            overlap = _polygonal_geometry(
                candidate_geometry.intersection(bank_geometry)
            )
            if overlap.is_empty:
                continue
            visible = (
                overlap
                if covered.is_empty
                else _polygonal_geometry(overlap.difference(covered))
            )
            overlap_area = float(visible.area)
            if candidate_intensity > bank_intensity:
                increase += (
                    candidate_intensity - bank_intensity
                ) * overlap_area
            covered = (
                overlap
                if covered.is_empty
                else _polygonal_geometry(unary_union((covered, overlap)))
            )
        uncovered_area = max(
            0.0,
            float(candidate_geometry.area) - float(covered.area),
        )
        increase += candidate_intensity * uncovered_area
    increase = max(0.0, increase)
    return increase / float(bank.score), increase


def _bounds_overlap(left: BaseGeometry, right: BaseGeometry) -> bool:
    left_bounds, right_bounds = left.bounds, right.bounds
    return not (
        left_bounds[2] < right_bounds[0]
        or right_bounds[2] < left_bounds[0]
        or left_bounds[3] < right_bounds[1]
        or right_bounds[3] < left_bounds[1]
    )


def merge_weighted_support_envelopes(
    bank: WeightedSupportEnvelope,
    candidate: WeightedSupportEnvelope,
) -> WeightedSupportEnvelope:
    """Return the exact pointwise maximum of two disjoint envelopes."""

    if not candidate.regions:
        return bank
    if not bank.regions:
        return candidate
    regions = list(bank.regions)
    for candidate_intensity, candidate_geometry in candidate.regions:
        visible = candidate_geometry
        updated: list[tuple[float, BaseGeometry]] = []
        for bank_intensity, bank_geometry in regions:
            if not _bounds_overlap(bank_geometry, candidate_geometry):
                updated.append((bank_intensity, bank_geometry))
                continue
            if bank_intensity >= candidate_intensity:
                if not visible.is_empty and _bounds_overlap(
                    visible, bank_geometry
                ):
                    visible = _polygonal_geometry(
                        visible.difference(bank_geometry)
                    )
                updated.append((bank_intensity, bank_geometry))
            else:
                remaining = _polygonal_geometry(
                    bank_geometry.difference(candidate_geometry)
                )
                if not remaining.is_empty:
                    updated.append((bank_intensity, remaining))
        if not visible.is_empty:
            updated.append((candidate_intensity, visible))
        regions = updated
    compact = tuple(
        (float(intensity), geometry)
        for intensity, geometry in regions
        if not geometry.is_empty
    )
    score = sum(
        intensity * float(geometry.area)
        for intensity, geometry in compact
    )
    return WeightedSupportEnvelope(compact, max(0.0, score))


class SupportExpertBankSimulation(CapsuleGreedySimulation):
    checkpoint_format = "place_wallis_support_expert_bank_checkpoint_v1"

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"support_expert_bank_k{self.bank_capacity}{suffix}"

    def _result_method_name(self) -> str:
        suffix = f", {self.method_tag}" if self.method_tag else ""
        return f"Support-driven expert bank (K={self.bank_capacity}{suffix})"

    def _prediction_choice(
        self,
        *,
        step: int,
        receiver: int,
        keys: list[ExpertKey],
        routing: np.ndarray,
    ) -> np.ndarray:
        del step, receiver, keys
        return np.argmax(routing, axis=1)

    def __init__(
        self,
        cfg,
        *,
        bank_capacity: int,
        transfer_cost: float,
        probe_count: int,
        method_tag: str = "",
        resume: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.bank_capacity = int(bank_capacity)
        self.transfer_cost = float(transfer_cost)
        self.probe_count = int(probe_count)
        self.method_tag = str(method_tag).strip().replace(" ", "_")
        if any(
            not (character.isalnum() or character in "_-")
            for character in self.method_tag
        ):
            raise ValueError("method_tag must contain only letters, digits, _ or -")
        if self.bank_capacity <= 0:
            raise ValueError("bank capacity must be positive")
        if self.transfer_cost < 0.0:
            raise ValueError("transfer cost cannot be negative")
        if self.probe_count <= 0:
            raise ValueError("probe count must be positive")
        self._expert_registry: dict[ExpertKey, ExpertRecord] = {}
        self._expert_banks: list[list[ExpertKey]] = []
        self._local_support: list[tuple[CapsuleRow, ...]] = []
        self._local_versions: list[int] = []
        self._expert_incarnations: list[int] = []
        self._model_transfers = 0
        self._manifest_records = 0
        self._coverage_recalculations = 0
        self._selection_probes_m = np.empty((0, 2, 2), dtype=np.float64)
        self._support_profiles: dict[ExpertKey, np.ndarray] = {}
        super().__init__(cfg, resume=None, **kwargs)
        self._selection_probes_m = self._make_selection_probes()
        count = int(cfg.num_nodes)
        self._expert_banks = [[] for _ in range(count)]
        self._local_support = [() for _ in range(count)]
        self._local_versions = [0 for _ in range(count)]
        self._expert_incarnations = [0 for _ in range(count)]
        self._communication_assumptions.update(
            {
                "method": "support-driven overlapping-plane expert bank",
                "expert_bank_capacity": self.bank_capacity,
                "expert_bank_transfer_cost": self.transfer_cost,
                "expert_bank_acquisition_score": (
                    "mean marginal support-intensity gain on fixed unlabeled "
                    "street-link probes - transfer_cost"
                ),
                "expert_bank_support_advertisement": (
                    "overlapping-plane geometry, mass, maximum link length, "
                    "lineage, version, and experience"
                ),
                "expert_bank_probe_count": self.probe_count,
                "expert_bank_probe_source": (
                    "deterministic unlabeled positions from the mobility trace"
                ),
                "expert_bank_raw_samples_shared": False,
                "expert_bank_routing": "maximum support intensity",
                "expert_bank_model_aggregation": False,
                "expert_bank_local_expert_retained": True,
                "expert_bank_coverage_refresh": (
                    "immediately after every accepted pull"
                ),
                "local_training": {
                    **asdict(self.training_params),
                    "optimizer": "Adam",
                    "learning_rate": float(cfg.local_lr),
                    "batch_size": int(cfg.local_batch_size),
                    "optimizer_reset": "never; expert parameters are not averaged",
                    "experience_counts_replay": False,
                },
                "round_order": (
                    "synchronous support-driven acquisition, then local replay train"
                ),
            }
        )
        if resume is not None:
            self._load_checkpoint(Path(resume))

    def _make_selection_probes(self) -> np.ndarray:
        replay = self._trace_replay
        if replay is None:
            raise ValueError("expert bank requires a replay trace")
        states = replay["node_states"]
        active = replay["node_active"]
        assert isinstance(states, np.ndarray)
        assert isinstance(active, np.ndarray)
        points = np.asarray(states[active][:, :2], dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        points = np.unique(np.round(points, decimals=2), axis=0)
        if len(points) < 2:
            raise ValueError("trace has too few active road positions for probes")
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.cfg.seed), 0x45585052])
        )
        left = rng.integers(0, len(points), size=self.probe_count)
        right = rng.integers(0, len(points), size=self.probe_count)
        for _ in range(8):
            short = np.linalg.norm(points[right] - points[left], axis=1) < 1.0
            if not bool(np.any(short)):
                break
            right[short] = rng.integers(0, len(points), size=int(short.sum()))
        return np.stack((points[left], points[right]), axis=1)

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        index = int(i)
        if index < len(self._expert_banks):
            self._expert_incarnations[index] += 1
            self._local_versions[index] = 0
            self._local_support[index] = ()
            self._expert_banks[index] = []
            self._prune_registry()

    def _profile_for_key(self, key: ExpertKey) -> np.ndarray:
        profile = self._support_profiles.get(key)
        if profile is None:
            record = self._expert_registry[key]
            profile = support_profile(
                record.capsules,
                self._selection_probes_m,
                self.capsule_params,
            )
            self._support_profiles[key] = profile
        return profile

    def _bank_profile(self, keys: list[ExpertKey]) -> np.ndarray:
        if not keys:
            return np.zeros(self.probe_count, dtype=np.float64)
        return np.max(
            np.stack([self._profile_for_key(key) for key in keys]), axis=0
        )

    def _score(
        self, record: ExpertRecord, bank_profile: np.ndarray
    ) -> tuple[float, float, int, int, ExpertKey]:
        gain = marginal_support_gain(
            self._profile_for_key(record.key), bank_profile
        )
        value = gain - self.transfer_cost
        return (
            float(value),
            float(gain),
            int(record.experience),
            int(record.key[2]),
            record.key,
        )

    def _select_bank(
        self, receiver: int, candidates: list[ExpertKey]
    ) -> list[ExpertKey]:
        newest: dict[tuple[int, int], ExpertKey] = {}
        for key in candidates:
            if key not in self._expert_registry:
                continue
            lineage = key[:2]
            current = newest.get(lineage)
            if current is None or int(key[2]) > int(current[2]):
                newest[lineage] = key
        remaining = list(newest.values())
        selected: list[ExpertKey] = []
        own_lineage = (
            int(receiver),
            int(self._expert_incarnations[int(receiver)]),
        )
        own = newest.get(own_lineage)
        if own is not None:
            selected.append(own)
            remaining.remove(own)
        bank_profile = self._bank_profile(selected)
        while remaining and len(selected) < self.bank_capacity:
            chosen = max(
                remaining,
                key=lambda key: self._score(
                    self._expert_registry[key], bank_profile
                ),
            )
            if self._score(
                self._expert_registry[chosen], bank_profile
            )[0] <= 1.0e-12:
                break
            selected.append(chosen)
            remaining.remove(chosen)
            bank_profile = np.maximum(
                bank_profile, self._profile_for_key(chosen)
            )
        return selected

    def _provider_candidate(
        self,
        receiver_keys: list[ExpertKey],
        provider_keys: list[ExpertKey],
    ) -> ExpertKey | None:
        exact = set(receiver_keys)
        receiver_versions = {
            key[:2]: int(key[2]) for key in receiver_keys
        }
        usable = [
            key
            for key in provider_keys
            if key in self._expert_registry
            and key not in exact
            and int(key[2]) > receiver_versions.get(key[:2], -1)
        ]
        if not usable:
            return None
        bank_profile = self._bank_profile(receiver_keys)
        chosen = max(
            usable,
            key=lambda key: self._score(
                self._expert_registry[key], bank_profile
            ),
        )
        return (
            chosen
            if self._score(self._expert_registry[chosen], bank_profile)[0]
            > 1.0e-12
            else None
        )

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs()
        links = sorted(
            {
                (int(zone), min(int(a), int(b)), max(int(a), int(b)))
                for zone, a, b in (contact_links or [])
                if int(a) != int(b)
            }
        )
        neighbours: dict[int, list[int]] = {}
        for _zone, left, right in links:
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        pre_banks = {
            index: list(self._expert_banks[index])
            for index in neighbours
        }
        next_banks: dict[int, list[ExpertKey]] = {}
        model_messages = 0
        manifest_records = 0
        capsule_values = 0
        for receiver in sorted(neighbours):
            working = self._select_bank(receiver, list(pre_banks[receiver]))
            for sender in sorted(neighbours[receiver]):
                sender_bank = pre_banks[sender]
                manifest_records += len(sender_bank)
                capsule_values += 11 * sum(
                    len(self._expert_registry[key].capsules)
                    for key in sender_bank
                    if key in self._expert_registry
                )
                candidate = self._provider_candidate(
                    working, sender_bank
                )
                if candidate is not None:
                    updated = self._select_bank(
                        receiver, [*working, candidate]
                    )
                    if candidate in updated:
                        working = updated
                        model_messages += 1
                        self._coverage_recalculations += 1
            next_banks[receiver] = working
        for receiver, bank in next_banks.items():
            self._expert_banks[receiver] = bank
        self._model_transfers += int(model_messages)
        self._manifest_records += int(manifest_records)
        self._network_step_stats.update(
            {
                "expert_bank_model_messages": int(model_messages),
                "expert_bank_manifest_records": int(manifest_records),
                "capsule_scalar_values_sent": int(capsule_values),
                "capsule_payload_bytes": int(4 * capsule_values),
                "expert_bank_receivers": int(len(next_banks)),
                "expert_bank_coverage_recalculations": int(
                    self._coverage_recalculations
                ),
            }
        )
        self._train_staged_local_samples(step)
        return int(model_messages)

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        self._local_versions[index] += 1
        key = (
            index,
            int(self._expert_incarnations[index]),
            int(self._local_versions[index]),
        )
        self._expert_registry[key] = ExpertRecord(
            key=key,
            experience=int(self.greedy_m_samples[index]),
            capsules=self._local_support[index],
            model_state=_cpu_state(self.greedy_models[index]),
        )
        self._support_profiles[key] = support_profile(
            self._local_support[index],
            self._selection_probes_m,
            self.capsule_params,
        )
        own_lineage = key[:2]
        candidates = [
            candidate
            for candidate in self._expert_banks[index]
            if candidate[:2] != own_lineage
        ]
        self._expert_banks[index] = self._select_bank(
            index, [key, *candidates]
        )

    def _additional_training_receivers(
        self, active: set[int]
    ) -> set[int]:
        del active
        return set()

    def _train_additional_receiver_data(
        self, step: int, receiver: int, rng: np.random.Generator
    ) -> bool:
        del step, receiver, rng
        return False

    def _train_staged_local_samples(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows_by_receiver: dict[
            int, list[tuple[list[float], float, np.ndarray]]
        ] = {}
        self._meas_per_node = {}
        for zone, tx_idx, rx_idx, value in measurements:
            tx_node = self.nodes[int(tx_idx)].node
            rx_node = self.nodes[int(rx_idx)].node
            features = self._pair_model_features(
                (tx_node.x, tx_node.y),
                (rx_node.x, rx_node.y),
                step=step,
                zone=int(zone),
            )
            segment = np.asarray(
                [[tx_node.x, tx_node.y], [rx_node.x, rx_node.y]],
                dtype=np.float64,
            )
            rows_by_receiver.setdefault(int(rx_idx), []).append(
                (features, float(value), segment)
            )
        active = {
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
        }
        receivers = sorted(
            set(rows_by_receiver)
            | (active & set(self._replay_buffers))
            | self._additional_training_receivers(active)
        )
        for receiver in receivers:
            rows = rows_by_receiver.get(receiver, [])
            if rows:
                capsules = deserialize_capsules(
                    self._local_support[receiver]
                )
                for _features, _value, segment in rows:
                    if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0:
                        add_capsule_vectorized(
                            capsules,
                            Capsule.from_segment(
                                segment,
                                half_width=(
                                    self.capsule_params.initial_half_width_m
                                ),
                            ),
                            self.capsule_params,
                            remote=False,
                        )
                self._local_support[receiver] = serialize_capsules(capsules)
                self.greedy_models[receiver].set_ribbons(
                    self._local_support[receiver]
                )
            X = np.asarray(
                [row[0] for row in rows], dtype=np.float32
            ).reshape(-1, 4)
            y = np.asarray(
                [row[1] for row in rows], dtype=np.float32
            ).reshape(-1, 1)
            replay = self._replay_buffers.get(receiver)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(self.cfg.seed), int(step), int(receiver)]
                )
            )
            updated = False
            full_dataset_training = (
                self.training_params.full_dataset_epochs > 0
            )
            if rows:
                self._train_array(
                    receiver,
                    X,
                    y,
                    epochs=self.training_params.new_data_epochs,
                    rng=rng,
                )
                updated = True
            if full_dataset_training and rows:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity, 4
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += int(len(rows))
                self.greedy_n_samples[receiver] = self.greedy_m_samples[
                    receiver
                ]
            if replay is not None and replay.size > 0:
                if full_dataset_training:
                    replay_X, replay_y = replay.all_data()
                    self._train_array(
                        receiver,
                        replay_X,
                        replay_y,
                        epochs=self.training_params.full_dataset_epochs,
                        rng=rng,
                    )
                else:
                    recent_start = (
                        self.training_params.replay_batches
                        - self.training_params.recent_replay_batches
                    )
                    for batch_index in range(
                        self.training_params.replay_batches
                    ):
                        replay_X, replay_y = replay.sample(
                            rng,
                            int(self.cfg.local_batch_size),
                            recent_window=(
                                self.training_params.recent_window
                                if batch_index >= recent_start
                                else None
                            ),
                        )
                        self._train_array(
                            receiver, replay_X, replay_y, epochs=1, rng=rng
                        )
                updated = True
            if rows and not full_dataset_training:
                if replay is None:
                    replay = ReplayBuffer(
                        self.training_params.replay_capacity, 4
                    )
                    self._replay_buffers[receiver] = replay
                replay.add(X, y)
                self.greedy_m_samples[receiver] += int(len(rows))
                self.greedy_n_samples[receiver] = self.greedy_m_samples[
                    receiver
                ]
            if self._train_additional_receiver_data(
                step, receiver, rng
            ):
                updated = True
            if updated:
                self._refresh_local_expert(receiver)
        self._staged_measurements = None
        self._prune_registry()

    def _prune_registry(self) -> None:
        live = {
            key for bank in self._expert_banks for key in bank
        }
        self._expert_registry = {
            key: record
            for key, record in self._expert_registry.items()
            if key in live
        }
        self._support_profiles = {
            key: profile
            for key, profile in self._support_profiles.items()
            if key in live
        }

    def _expert_support_profile(
        self, key: ExpertKey, query_m: np.ndarray
    ) -> np.ndarray:
        record = self._expert_registry[key]
        return support_profile(
            record.capsules, query_m, self.capsule_params
        )

    def _expert_support_profiles(
        self, keys: list[ExpertKey], query_m: np.ndarray
    ) -> dict[ExpertKey, np.ndarray]:
        return {
            key: self._expert_support_profile(key, query_m)
            for key in keys
        }

    def _expert_normalized_predictions(
        self,
        template: torch.nn.Module,
        record: ExpertRecord,
        xt: torch.Tensor,
        routing_profile: np.ndarray,
    ) -> np.ndarray:
        del routing_profile
        template.load_state_dict(record.model_state)
        template.set_ribbons(record.capsules)
        template.eval()
        with torch.no_grad():
            normalized, _confidence = template.forward_with_confidence(xt)
        return normalized.detach().cpu().numpy().reshape(-1)

    def _expert_support_unit_count(self, key: ExpertKey) -> int:
        return int(len(self._expert_registry[key].capsules))

    def _pulled_support_payload_values(self, key: ExpertKey) -> int:
        return int(11 * len(self._expert_registry[key].capsules))

    def _requires_raw_expert_predictions(self) -> bool:
        return False

    def _route_fidelity_predictions(
        self,
        *,
        step: int,
        receiver: int,
        keys: list[ExpertKey],
        query_m: np.ndarray,
        predictions: dict[ExpertKey, np.ndarray],
        raw_predictions: dict[ExpertKey, np.ndarray],
        routing_profiles: dict[ExpertKey, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        del query_m, raw_predictions
        routing = np.stack(
            [routing_profiles[key] for key in keys], axis=1
        )
        normalized = np.stack(
            [predictions[key] for key in keys], axis=1
        )
        choice = self._prediction_choice(
            step=int(step),
            receiver=int(receiver),
            keys=keys,
            routing=routing,
        )
        rows = np.arange(len(normalized))
        return normalized[rows, choice], routing[rows, choice]

    def _fidelity_prediction_keys(
        self, active: list[int], unique: list[ExpertKey]
    ) -> set[ExpertKey]:
        del active
        return set(unique)

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        if int(step) <= self._resume_step and self._resume_payload is not None:
            for row in reversed(
                self._resume_payload.get("fidelity_history", [])
            ):
                if int(row.get("step", -1)) == int(step):
                    return dict(row)
            return {"step": int(step)}
        self._build_fidelity_grid(n_pairs=n_pairs)
        X, y = self.fidelity_grid[0]
        truth = y.reshape(-1)
        feasible = self._fidelity_feasible
        active = [
            index
            for index in range(int(self.cfg.num_nodes))
            if bool(self._current_node_active[index])
            and bool(self._expert_banks[index])
        ]
        unique = sorted(
            {key for index in active for key in self._expert_banks[index]}
        )
        prediction_keys = self._fidelity_prediction_keys(active, unique)
        predictions: dict[ExpertKey, np.ndarray] = {}
        routing_profiles: dict[ExpertKey, np.ndarray] = {}
        raw_predictions: dict[ExpertKey, np.ndarray] = {}
        query_m = (
            np.asarray(X[:, :4], dtype=np.float64).reshape(-1, 2, 2)
            * float(self.cfg.map_size)
        )
        if unique:
            template = copy.deepcopy(self.greedy_models[active[0]])
            xt = torch.as_tensor(
                X, dtype=torch.float32, device=self.aux_device
            )
            routing_profiles.update(
                self._expert_support_profiles(unique, query_m)
            )
            for key in unique:
                record = self._expert_registry[key]
                if key in prediction_keys:
                    predictions[key] = self._expert_normalized_predictions(
                        template, record, xt, routing_profiles[key]
                    )
                    if self._requires_raw_expert_predictions():
                        with torch.no_grad():
                            raw = template.base(xt)
                        raw_predictions[key] = (
                            raw.detach().cpu().numpy().reshape(-1)
                        )
        total_sq = feasible_sq = infeasible_sq = 0.0
        total_count = feasible_count = infeasible_count = 0
        model_rmse: list[float] = []
        confidence_sum = 0.0
        predicted_total = predicted_infeasible = 0
        covered_total = covered_feasible = covered_infeasible = 0
        for index in active:
            keys = self._expert_banks[index]
            selected, conf = self._route_fidelity_predictions(
                step=int(step),
                receiver=int(index),
                keys=keys,
                query_m=query_m,
                predictions=predictions,
                raw_predictions=raw_predictions,
                routing_profiles=routing_profiles,
            )
            prediction = self._denorm_dbm(selected)
            error_sq = np.square(prediction - truth)
            covered = conf >= 0.5
            positive = prediction > self.reception_floor_dbm
            total_sq += float(error_sq.sum())
            total_count += int(error_sq.size)
            feasible_sq += float(error_sq[feasible].sum())
            feasible_count += int(feasible.sum())
            infeasible_sq += float(error_sq[~feasible].sum())
            infeasible_count += int((~feasible).sum())
            model_rmse.append(float(np.sqrt(error_sq.mean())))
            confidence_sum += float(conf.sum())
            covered_total += int(covered.sum())
            covered_feasible += int(covered[feasible].sum())
            covered_infeasible += int(covered[~feasible].sum())
            predicted_total += int(positive.sum())
            predicted_infeasible += int(positive[~feasible].sum())

        def rmse(value: float, count: int) -> float:
            return (
                float(math.sqrt(value / count))
                if count > 0
                else float("nan")
            )

        def ratio(value: float, count: int) -> float:
            return float(value / count) if count else float("nan")

        bank_sizes = [len(self._expert_banks[index]) for index in active]
        capsule_counts = [
            sum(
                self._expert_support_unit_count(key)
                for key in self._expert_banks[index]
            )
            for index in active
        ]
        experiences = [
            sum(
                int(self._expert_registry[key].experience)
                for key in self._expert_banks[index]
            )
            for index in active
        ]
        denominator = int(len(active) * len(X))
        row: dict[str, float | int] = {
            "step": int(step),
            "eval_n_pairs_per_zone": int(len(X)),
            "eval_is_final": int(is_final),
            "greedy_total": rmse(total_sq, total_count),
            "greedy_mean_model_rmse": (
                float(np.mean(model_rmse)) if model_rmse else float("nan")
            ),
            "greedy_feasible_rmse": rmse(feasible_sq, feasible_count),
            "greedy_infeasible_rmse": rmse(
                infeasible_sq, infeasible_count
            ),
            "greedy_active_experienced_models": int(len(active)),
            "greedy_mean_confidence": (
                confidence_sum / denominator
                if denominator
                else float("nan")
            ),
            "greedy_coverage_at_0_5": ratio(covered_total, denominator),
            "greedy_feasible_coverage_at_0_5": ratio(
                covered_feasible, feasible_count
            ),
            "greedy_infeasible_leakage_at_0_5": ratio(
                covered_infeasible, infeasible_count
            ),
            "greedy_predicted_feasible_fraction": ratio(
                predicted_total, denominator
            ),
            "greedy_non_feasible_false_positive_rate": ratio(
                predicted_infeasible, infeasible_count
            ),
            "greedy_mean_capsules": (
                float(np.mean(capsule_counts))
                if capsule_counts
                else float("nan")
            ),
            "greedy_max_capsules": int(max(capsule_counts, default=0)),
            "greedy_mean_experience": (
                float(np.mean(experiences)) if experiences else 0.0
            ),
            "greedy_max_experience": int(max(experiences, default=0)),
            "expert_bank_mean_size": (
                float(np.mean(bank_sizes)) if bank_sizes else 0.0
            ),
            "expert_bank_max_size": int(max(bank_sizes, default=0)),
            "expert_bank_unique_versions": int(len(unique)),
            "expert_bank_model_transfers": int(self._model_transfers),
            "expert_bank_manifest_records": int(self._manifest_records),
            "expert_bank_coverage_recalculations": int(
                self._coverage_recalculations
            ),
        }
        self.fidelity_history.append(row)
        return row

    def _save_checkpoint(self, step: int) -> None:
        experienced = [
            index
            for index, value in enumerate(self.greedy_m_samples)
            if int(value) > 0
        ]
        output = Path(self.cfg.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        latest = self.fidelity_history[-1] if self.fidelity_history else {}
        temporal = temporal_metric_summary(
            self.fidelity_history,
            evaluation_steps=self.tail_evaluation_steps,
            metric_keys=(
                "greedy_total",
                "greedy_feasible_rmse",
                "greedy_infeasible_rmse",
                "greedy_predicted_feasible_fraction",
                "greedy_non_feasible_false_positive_rate",
                "greedy_mean_confidence",
                "greedy_coverage_at_0_5",
                "greedy_feasible_coverage_at_0_5",
                "greedy_infeasible_leakage_at_0_5",
                "greedy_mean_capsules",
                "expert_bank_mean_size",
            ),
        )
        means = temporal["mean"]
        deviations = temporal["standard_deviation"]
        run_status = (
            "complete"
            if int(step) >= int(self.cfg.sim_steps) and bool(temporal["complete"])
            else "running"
        )
        status = {
            "format": self.checkpoint_format,
            "step": int(step),
            "checkpoint_kind": "metrics-only",
            "path": None,
            "experienced_models": int(len(experienced)),
            "live_expert_versions": int(len(self._expert_registry)),
            "latest_fidelity": latest,
            "temporal_evaluation": temporal,
            "bank_capacity": self.bank_capacity,
            "coverage_recalculations": self._coverage_recalculations,
        }
        atomic_json(output / "checkpoint_status.json", status)
        atomic_json(
            output / "metrics.json",
            {
                "schema": "place_wallis_benchmark_result_v1",
                "status": run_status,
                "method": {
                    "id": self._result_method_id(),
                    "name": self._result_method_name(),
                    "model": "bank of 4-64-64-1 MLP experts",
                },
                "checkpoint": {
                    "step": int(step),
                    "final_step": int(self.cfg.sim_steps),
                },
                "metrics_db": {
                    "overall_rmse": means["greedy_total"],
                    "feasible_rmse": means["greedy_feasible_rmse"],
                    "non_feasible_rmse": means["greedy_infeasible_rmse"],
                },
                "metrics_db_standard_deviation": {
                    "overall_rmse": deviations["greedy_total"],
                    "feasible_rmse": deviations["greedy_feasible_rmse"],
                    "non_feasible_rmse": deviations[
                        "greedy_infeasible_rmse"
                    ],
                },
                "evaluation": {
                    "noise_floor_dbm": self.reception_floor_dbm,
                    "true_feasible_rule": "rssi_dbm > -100",
                    "predicted_feasible_fraction": means[
                        "greedy_predicted_feasible_fraction"
                    ],
                    "non_feasible_false_positive_rate": means[
                        "greedy_non_feasible_false_positive_rate"
                    ],
                },
                "temporal_evaluation": temporal,
                "training": asdict(self.training_params),
                "support_plane_params": asdict(self.capsule_params),
                "expert_bank": {
                    "capacity_including_local": self.bank_capacity,
                    "transfer_cost": self.transfer_cost,
                    "selection_probe_count": self.probe_count,
                    "selection": "greedy marginal support-intensity coverage",
                    "routing": "maximum support intensity",
                    "coverage_recalculation": "after every accepted pull",
                    "coverage_recalculations_latest": (
                        self._coverage_recalculations
                    ),
                    "mean_size_tail": means["expert_bank_mean_size"],
                    "model_transfers_latest": self._model_transfers,
                },
                "support": {
                    "gate": "binary overlapping straight planes",
                    "mean_planes_tail": means["greedy_mean_capsules"],
                    "mean_intensity_tail": means["greedy_mean_confidence"],
                    "coverage_at_0_5_tail": means[
                        "greedy_coverage_at_0_5"
                    ],
                    "feasible_coverage_at_0_5_tail": means[
                        "greedy_feasible_coverage_at_0_5"
                    ],
                    "infeasible_leakage_at_0_5_tail": means[
                        "greedy_infeasible_leakage_at_0_5"
                    ],
                },
                "latest_fidelity": latest,
                "testset": {
                    "path": str(self._testset_path),
                    "samples": int(latest.get("eval_n_pairs_per_zone", 0)),
                },
            },
        )
        print(
            f"[SUPPORT-EXPERT-BANK] metrics step={step} "
            f"models={len(experienced)} experts={len(self._expert_registry)} "
            f"bank={float(latest.get('expert_bank_mean_size', 0.0)):.2f} "
            f"overall={float(latest.get('greedy_total', float('nan'))):.4f}dB "
            f"false-positive={100 * float(latest.get('greedy_non_feasible_false_positive_rate', float('nan'))):.1f}%",
            flush=True,
        )

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(
            path.resolve(), map_location=self.aux_device, weights_only=False
        )
        if payload.get("format") != self.checkpoint_format:
            raise ValueError(f"unsupported checkpoint format in {path}")
        if int(payload.get("bank_capacity", -1)) != self.bank_capacity:
            raise ValueError("checkpoint bank capacity differs")
        if float(payload.get("transfer_cost", float("nan"))) != self.transfer_cost:
            raise ValueError("checkpoint transfer cost differs")
        if int(payload.get("probe_count", -1)) != self.probe_count:
            raise ValueError("checkpoint probe count differs")
        if tuple(payload.get("tail_evaluation_steps", ())) != (
            self.tail_evaluation_steps
        ):
            raise ValueError("checkpoint tail evaluation schedule differs")
        if payload.get("capsule_params") != asdict(self.capsule_params):
            raise ValueError("checkpoint capsule parameters differ")
        if payload.get("gate_params") != asdict(self.gate_params):
            raise ValueError("checkpoint gate parameters differ")
        if payload.get("training_params") != asdict(self.training_params):
            raise ValueError("checkpoint training parameters differ")
        self._resume_step = int(payload["step"])
        experience = [int(value) for value in payload["experience"]]
        self.greedy_m_samples = list(experience)
        self.greedy_n_samples = list(experience)
        for raw_index, state in payload["models"].items():
            self.greedy_models[int(raw_index)].load_state_dict(state)
        for raw_index, state in payload["optimizers"].items():
            self.greedy_opts[int(raw_index)].load_state_dict(state)
        self._local_support = payload["local_support"]
        self._local_versions = [
            int(value) for value in payload["local_versions"]
        ]
        self._expert_incarnations = [
            int(value) for value in payload["expert_incarnations"]
        ]
        self._expert_registry = {
            tuple(int(value) for value in key): ExpertRecord(
                key=tuple(int(value) for value in key),
                experience=int(data["experience"]),
                capsules=tuple(data["capsules"]),
                model_state=data["model_state"],
            )
            for key, data in payload["expert_registry"].items()
        }
        self._support_profiles = {}
        self._expert_banks = [
            [tuple(int(value) for value in key) for key in bank]
            for bank in payload["expert_banks"]
        ]
        self._replay_buffers = {
            int(index): ReplayBuffer.from_state_dict(state)
            for index, state in payload["replay_buffers"].items()
        }
        for index, support in enumerate(self._local_support):
            self.greedy_models[index].set_ribbons(support)
        self._model_transfers = int(payload.get("model_transfers", 0))
        self._manifest_records = int(payload.get("manifest_records", 0))
        self._coverage_recalculations = int(
            payload.get("coverage_recalculations", 0)
        )
        self._resume_payload = payload
        self._resume_logs_restored = False
        print(
            f"[SUPPORT-EXPERT-BANK] resumed step={self._resume_step} "
            f"from {path}",
            flush=True,
        )



class LocalOnlySupportSimulation(SupportExpertBankSimulation):
    """One persistent support-gated predictor per vehicle, without sharing."""

    checkpoint_format = "place_wallis_local_only_support_checkpoint_v1"

    def __init__(self, cfg, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._communication_assumptions.update(
            {
                "method": "local-only overlapping-plane support baseline",
                "support_shared": False,
                "raw_samples_shared": False,
                "model_parameters_shared": False,
                "expert_bank_capacity": 1,
                "expert_bank_routing": "own persistent predictor only",
                "round_order": "local support update, then local replay train",
            }
        )

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"local_only_support{suffix}"

    def _result_method_name(self) -> str:
        suffix = f" ({self.method_tag})" if self.method_tag else ""
        return f"Local-only support-gated MLP{suffix}"

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes, contact_links
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs()
        self._network_step_stats.update(
            {
                "expert_bank_model_messages": 0,
                "expert_bank_manifest_records": 0,
                "capsule_scalar_values_sent": 0,
                "capsule_payload_bytes": 0,
                "expert_bank_coverage_recalculations": 0,
            }
        )
        self._train_staged_local_samples(step)
        return 0

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        path = Path(self.cfg.results_dir) / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["method"] = {
            "id": self._result_method_id(),
            "name": self._result_method_name(),
            "model": "one persistent 4-64-64-1 support-gated MLP per vehicle",
        }
        metrics["baseline"] = {
            "type": "lower",
            "communication": "none",
            "prediction": "each active vehicle uses only its own model and support",
        }
        atomic_json(path, metrics)


class CentralSupportSimulation(SupportExpertBankSimulation):
    """One map-wide predictor and support set trained on every observation."""

    checkpoint_format = "place_wallis_central_support_checkpoint_v1"

    def __init__(self, cfg, **kwargs: Any) -> None:
        self._central_state_ready = False
        super().__init__(cfg, **kwargs)
        self._central_model_index = 0
        self._central_support: tuple[CapsuleRow, ...] = ()
        self._central_replay = ReplayBuffer(
            self.training_params.replay_capacity, 4
        )
        self._central_experience = 0
        self._central_version = 0
        self._expert_registry.clear()
        self._support_profiles.clear()
        self._expert_banks = [[] for _ in range(int(self.cfg.num_nodes))]
        self._central_state_ready = True
        self._communication_assumptions.update(
            {
                "method": "central overlapping-plane support baseline",
                "central_predictors": 1,
                "central_training": (
                    "all new feasible measurements, then one shuffled epoch "
                    "over every accumulated measurement"
                ),
                "central_support": "all feasible measurement segments",
                "support_shared": False,
                "raw_samples_shared": "conceptual centralized collection",
                "model_parameters_shared": False,
                "evaluation_computation": "one predictor evaluation per checkpoint",
                "round_order": "central support update, then central replay train",
            }
        )

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"central_support{suffix}"

    def _result_method_name(self) -> str:
        suffix = f" ({self.method_tag})" if self.method_tag else ""
        return f"Central support-gated MLP{suffix}"

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        if (
            getattr(self, "_central_state_ready", False)
            and int(i) == int(self._central_model_index)
        ):
            return
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)

    def _refresh_central_record(self) -> None:
        self._central_version += 1
        key = (0, 0, int(self._central_version))
        self._expert_registry = {
            key: ExpertRecord(
                key=key,
                experience=int(self._central_experience),
                capsules=self._central_support,
                model_state=_cpu_state(
                    self.greedy_models[self._central_model_index]
                ),
            )
        }
        self._support_profiles.clear()

    def _update_central_support(
        self, rows: list[tuple[list[float], float, np.ndarray]]
    ) -> None:
        """Update the baseline support representation from new segments."""
        capsules = deserialize_capsules(self._central_support)
        for _features, _value, segment in rows:
            if float(np.linalg.norm(segment[1] - segment[0])) >= 1.0:
                add_capsule_vectorized(
                    capsules,
                    Capsule.from_segment(
                        segment,
                        half_width=self.capsule_params.initial_half_width_m,
                    ),
                    self.capsule_params,
                    remote=False,
                )
        self._central_support = serialize_capsules(capsules)
        self.greedy_models[self._central_model_index].set_ribbons(
            self._central_support
        )

    def _train_central(self, step: int) -> None:
        measurements = self._staged_measurements or []
        rows: list[tuple[list[float], float, np.ndarray]] = []
        for zone, tx_idx, rx_idx, value in measurements:
            tx_node = self.nodes[int(tx_idx)].node
            rx_node = self.nodes[int(rx_idx)].node
            rows.append(
                (
                    self._pair_model_features(
                        (tx_node.x, tx_node.y),
                        (rx_node.x, rx_node.y),
                        step=step,
                        zone=int(zone),
                    ),
                    float(value),
                    np.asarray(
                        [[tx_node.x, tx_node.y], [rx_node.x, rx_node.y]],
                        dtype=np.float64,
                    ),
                )
            )

        if rows:
            self._update_central_support(rows)

        X = np.asarray(
            [row[0] for row in rows], dtype=np.float32
        ).reshape(-1, 4)
        y = np.asarray(
            [row[1] for row in rows], dtype=np.float32
        ).reshape(-1, 1)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.cfg.seed), int(step), 0x43454E54])
        )
        updated = False
        if rows:
            self._train_array(
                self._central_model_index,
                X,
                y,
                epochs=self.training_params.new_data_epochs,
                rng=rng,
            )
            self._central_replay.add(X, y)
            self._central_experience += int(len(rows))
            updated = True

        if self._central_replay.size > 0:
            if self.training_params.full_dataset_epochs > 0:
                replay_X, replay_y = self._central_replay.all_data()
                self._train_array(
                    self._central_model_index,
                    replay_X,
                    replay_y,
                    epochs=self.training_params.full_dataset_epochs,
                    rng=rng,
                )
            else:
                recent_start = (
                    self.training_params.replay_batches
                    - self.training_params.recent_replay_batches
                )
                for batch_index in range(self.training_params.replay_batches):
                    replay_X, replay_y = self._central_replay.sample(
                        rng,
                        int(self.cfg.local_batch_size),
                        recent_window=(
                            self.training_params.recent_window
                            if batch_index >= recent_start
                            else None
                        ),
                    )
                    self._train_array(
                        self._central_model_index,
                        replay_X,
                        replay_y,
                        epochs=1,
                        rng=rng,
                    )
            updated = True

        self.greedy_m_samples[self._central_model_index] = int(
            self._central_experience
        )
        self.greedy_n_samples[self._central_model_index] = int(
            self._central_experience
        )
        if updated:
            self._refresh_central_record()
        self._staged_measurements = None

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes, contact_links
        step = int(getattr(self, "_current_sumo_step", 0))
        self._restore_logs()
        self._network_step_stats.update(
            {
                "expert_bank_model_messages": 0,
                "expert_bank_manifest_records": 0,
                "capsule_scalar_values_sent": 0,
                "capsule_payload_bytes": 0,
                "expert_bank_coverage_recalculations": 0,
            }
        )
        self._train_central(step)
        return 0

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        self._expert_banks = [[] for _ in range(int(self.cfg.num_nodes))]
        if self._central_experience > 0 and self._expert_registry:
            active = np.flatnonzero(self._current_node_active)
            if len(active):
                self._expert_banks[int(active[0])] = [
                    next(iter(self._expert_registry))
                ]
        return super()._evaluate_fidelity_now(
            step, n_pairs=n_pairs, is_final=is_final
        )

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        path = Path(self.cfg.results_dir) / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["method"] = {
            "id": self._result_method_id(),
            "name": self._result_method_name(),
            "model": "one map-wide 4-64-64-1 support-gated MLP",
        }
        metrics["baseline"] = {
            "type": "upper",
            "communication": "ideal centralized access to every measurement",
            "prediction": "one central model with support from every sample",
            "central_experience": int(self._central_experience),
        }
        atomic_json(path, metrics)


class DominancePrunedExpertBankSimulation(SupportExpertBankSimulation):
    """Uncapped bank pruned only by exact geometric support dominance."""

    checkpoint_format = (
        "place_wallis_geometric_dominance_expert_bank_checkpoint_v2"
    )

    def __init__(
        self,
        cfg,
        *,
        min_unique_coverage: float,
        **kwargs: Any,
    ) -> None:
        # Retained only as a CLI-compatibility argument for earlier runs.  The
        # geometric method has no coverage threshold.
        del min_unique_coverage
        probe_count = int(kwargs.get("probe_count", 512))
        self._plane_geometry_cache: dict[
            ExpertKey, PlaneDominanceGeometry
        ] = {}
        self._geometric_dominance_cache: dict[
            tuple[ExpertKey, ExpertKey], int
        ] = {}
        self._geometric_pair_evaluations = 0
        # The subclass does not use the capacity; a positive value is supplied
        # only to satisfy the base-class constructor contract.
        kwargs["bank_capacity"] = max(1, probe_count)
        super().__init__(cfg, **kwargs)
        for obsolete in (
            "expert_bank_probe_count",
            "expert_bank_probe_source",
            "expert_bank_coverage_refresh",
        ):
            self._communication_assumptions.pop(obsolete, None)
        self._communication_assumptions.update(
            {
                "method": "uncapped geometric-dominance expert bank",
                "expert_bank_capacity": "none",
                "expert_bank_acquisition_score": (
                    "exact undominated support planes and bank compression"
                ),
                "expert_bank_pruning": (
                    "exact convex-plane containment, maximum link length, "
                    "and plane sample count"
                ),
                "expert_bank_local_expert_retained": (
                    "only while it contributes non-redundant coverage"
                ),
                "expert_bank_lineage_replacement": False,
                "dominance_implementation": (
                    "cached exact expert-pair plane-dominance bitsets"
                ),
                "deployment_probe_set": False,
            }
        )

    def _make_selection_probes(self) -> np.ndarray:
        """Geometric banks never inspect unvisited deployment positions."""

        return np.empty((0, 2, 2), dtype=np.float64)

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return f"support_expert_bank_geometric_dominance_unbounded{suffix}"

    def _result_method_name(self) -> str:
        suffix = f", {self.method_tag}" if self.method_tag else ""
        return (
            "Geometric-dominance expert bank "
            f"(uncapped{suffix})"
        )

    def _geometry_for_key(self, key: ExpertKey) -> PlaneDominanceGeometry:
        geometry = self._plane_geometry_cache.get(key)
        if geometry is None:
            geometry = plane_dominance_geometry(
                self._expert_registry[key].capsules
            )
            self._plane_geometry_cache[key] = geometry
        return geometry

    def _dominated_plane_mask(
        self, target: ExpertKey, dominator: ExpertKey
    ) -> int:
        pair = (target, dominator)
        mask = self._geometric_dominance_cache.get(pair)
        if mask is None:
            mask = geometrically_dominated_plane_mask(
                self._geometry_for_key(target),
                self._geometry_for_key(dominator),
            )
            self._geometric_dominance_cache[pair] = mask
            self._geometric_pair_evaluations += 1
        return mask

    def _full_plane_mask(self, key: ExpertKey) -> int:
        count = self._geometry_for_key(key).count
        return (1 << count) - 1 if count else 0

    def _expert_redundant(
        self, target: ExpertKey, retained: list[ExpertKey]
    ) -> bool:
        full = self._full_plane_mask(target)
        if full == 0:
            return bool(retained)
        dominated = 0
        for dominator in retained:
            if dominator == target or dominator not in self._expert_registry:
                continue
            dominated |= self._dominated_plane_mask(target, dominator)
            if dominated == full:
                return True
        return False

    def _dominates(self, candidate: ExpertKey, existing: ExpertKey) -> bool:
        full = self._full_plane_mask(existing)
        return (
            full == 0
            or self._dominated_plane_mask(existing, candidate) == full
        )

    def _select_bank(
        self, receiver: int, candidates: list[ExpertKey]
    ) -> list[ExpertKey]:
        del receiver
        keys = list(
            dict.fromkeys(
                key for key in candidates if key in self._expert_registry
            )
        )
        if len(keys) < 2:
            return keys

        # Remove one redundant expert at a time.  Rechecking against survivors
        # prevents circular deletion; dominance transitivity preserves every
        # plane removed in an earlier iteration.
        while len(keys) > 1:
            removable = [
                key
                for key in keys
                if self._expert_redundant(
                    key, [other for other in keys if other != key]
                )
            ]
            if not removable:
                break
            remove = min(
                removable,
                key=lambda key: (
                    int(self._expert_registry[key].experience), key
                ),
            )
            keys.remove(remove)
        return keys

    def _insert_candidate(
        self,
        receiver: int,
        bank: list[ExpertKey],
        candidate: ExpertKey,
    ) -> list[ExpertKey]:
        """Insert into an already-pruned bank using cached pair relations."""

        del receiver
        keys = list(dict.fromkeys(
            key for key in bank if key in self._expert_registry
        ))
        if candidate not in self._expert_registry or candidate in keys:
            return keys
        keys.append(candidate)
        while len(keys) > 1:
            removable = [
                key
                for key in keys
                if self._expert_redundant(
                    key, [other for other in keys if other != key]
                )
            ]
            if not removable:
                return keys
            remove = min(
                removable,
                key=lambda key: (
                    int(self._expert_registry[key].experience), key
                ),
            )
            keys.remove(remove)
        return keys

    def _provider_candidate(
        self,
        receiver_keys: list[ExpertKey],
        provider_keys: list[ExpertKey],
    ) -> ExpertKey | None:
        exact = set(receiver_keys)
        usable = [
            key
            for key in provider_keys
            if key in self._expert_registry
            and key not in exact
            and self._geometry_for_key(key).count > 0
        ]
        if not usable:
            return None
        scored: list[tuple[int, int, int, ExpertKey]] = []
        for key in usable:
            dominated_planes = 0
            for existing in receiver_keys:
                if existing in self._expert_registry:
                    dominated_planes |= self._dominated_plane_mask(
                        key, existing
                    )
            new_planes = (
                self._geometry_for_key(key).count
                - dominated_planes.bit_count()
            )
            dominated_count = sum(
                self._dominates(key, existing)
                for existing in receiver_keys
                if existing in self._expert_registry
            )
            if new_planes == 0 and dominated_count == 0:
                continue
            scored.append(
                (
                    new_planes,
                    dominated_count,
                    int(self._expert_registry[key].experience),
                    key,
                )
            )
        return max(scored)[-1] if scored else None

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        self._local_versions[index] += 1
        key = (
            index,
            int(self._expert_incarnations[index]),
            int(self._local_versions[index]),
        )
        self._expert_registry[key] = ExpertRecord(
            key=key,
            experience=int(self.greedy_m_samples[index]),
            capsules=self._local_support[index],
            model_state=_cpu_state(self.greedy_models[index]),
        )
        self._expert_banks[index] = self._insert_candidate(
            index, self._expert_banks[index], key
        )

    def _prune_registry(self) -> None:
        super()._prune_registry()
        live = set(self._expert_registry)
        self._plane_geometry_cache = {
            key: geometry
            for key, geometry in self._plane_geometry_cache.items()
            if key in live
        }
        self._geometric_dominance_cache = {
            pair: mask
            for pair, mask in self._geometric_dominance_cache.items()
            if pair[0] in live and pair[1] in live
        }

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["bank_capacity"] = None
        status.pop("coverage_recalculations", None)
        status["post_pull_validation"] = "exact geometric plane dominance"
        status["deployment_probe_set"] = False
        status["geometric_validations"] = self._coverage_recalculations
        status["geometric_pair_evaluations"] = (
            self._geometric_pair_evaluations
        )
        status["geometric_dominance_cache_entries"] = len(
            self._geometric_dominance_cache
        )
        atomic_json(status_path, status)

        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        bank = metrics["expert_bank"]
        bank["capacity_including_local"] = None
        bank.pop("selection_probe_count", None)
        bank.pop("minimum_unique_coverage", None)
        bank.pop("minimum_unique_probes", None)
        bank.pop("coverage_cache", None)
        bank.pop("coverage_recalculation", None)
        bank.pop("coverage_recalculations_latest", None)
        bank["selection"] = "uncapped exact geometric plane dominance"
        bank["pruning"] = (
            "remove an expert only when every plane is geometrically "
            "dominated by an individually containing retained plane"
        )
        bank["plane_dominance"] = (
            "polygon containment, no shorter maximum link length, and no "
            "smaller plane sample count"
        )
        bank["geometric_dominance_cache"] = (
            "one exact plane-index bitset per encountered immutable expert pair"
        )
        bank["deployment_probe_set"] = False
        bank["unique_coverage_threshold"] = None
        bank["geometric_validations_latest"] = (
            self._coverage_recalculations
        )
        bank["geometric_pair_evaluations_latest"] = (
            self._geometric_pair_evaluations
        )
        atomic_json(metrics_path, metrics)


class LearnedAcquisitionExpertBankSimulation(SupportExpertBankSimulation):
    """Uncapped bank with compact pretrained acquisition advertisements."""

    checkpoint_format = (
        "place_wallis_learned_staggered_grid_gain_expert_bank_checkpoint_v5"
    )

    def __init__(
        self,
        cfg,
        *,
        acquisition_bundle: Path,
        acquisition_probability_threshold: float,
        acquisition_relative_gain_penalty: float,
        bank_support_routing: str = "individual",
        teacher_distillation_batches_per_step: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("min_unique_coverage", None)
        self.acquisition_bundle = Path(acquisition_bundle).resolve()
        self.acquisition_probability_threshold = float(
            acquisition_probability_threshold
        )
        self.acquisition_relative_gain_penalty = float(
            acquisition_relative_gain_penalty
        )
        self.bank_support_routing = str(bank_support_routing)
        if self.bank_support_routing not in {
            "individual",
            "primary-only",
            "primary-bank-gate",
            "most-experienced",
            "experience-ensemble",
        }:
            raise ValueError("invalid bank support routing mode")
        self._merged_bank_support_cache: dict[
            frozenset[ExpertKey], MergedBankSupport
        ] = {}
        self.teacher_distillation_batches_per_step = int(
            teacher_distillation_batches_per_step
        )
        if self.teacher_distillation_batches_per_step < 0:
            raise ValueError(
                "teacher distillation batches per step cannot be negative"
            )
        if not self.acquisition_bundle.exists():
            raise FileNotFoundError(self.acquisition_bundle)
        if not 0.0 <= self.acquisition_probability_threshold <= 1.0:
            raise ValueError("acquisition probability threshold must be in [0, 1]")
        if self.acquisition_relative_gain_penalty < 0.0:
            raise ValueError("relative-gain penalty cannot be negative")
        self._encoded_advertisements: dict[
            ExpertKey, EncodedExpertAdvertisement
        ] = {}
        self._unit_square_encoder_pools: dict[
            ExpertKey, tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        self._grid_profile_cache: dict[ExpertKey, np.ndarray] = {}
        self._grid_bank_profile_cache: dict[
            frozenset[ExpertKey], np.ndarray
        ] = {}
        self._grid_bank_encoding_cache: dict[
            frozenset[ExpertKey], np.ndarray
        ] = {}
        self._grid_profile_evaluations = 0
        self._grid_bank_profile_evaluations = 0
        self._grid_redundancy_evaluations = 0
        self._gain_upper_bound_rejections = 0
        self._rejected_candidates: list[set[ExpertKey]] = []
        self._learned_pull_attempts = 0
        self._learned_pull_accepts = 0
        self._learned_pull_rejections = 0
        self._post_pull_gain_rejections = 0
        self._advertisement_scalar_values = 0
        self._pulled_support_scalar_values = 0
        self._pulled_model_parameter_values = 0
        self._teacher_support_merges = 0
        self._teacher_distillation_passes = 0
        self._teacher_distillation_samples = 0
        self._latest_acquisition_prediction: tuple[
            ExpertKey, float, float, float
        ] | None = None
        self._acquisition_checkpoint_step = -1
        self._gain_grid_layout = GRID_LAYOUT_STAGGERED
        self._sparse_patch_advertisements = False
        self._quantized_patch_advertisements = False
        super().__init__(cfg, **kwargs)
        self._rejected_candidates = [
            set() for _ in range(int(cfg.num_nodes))
        ]
        self._receiver_advertisement_cache: list[set[ExpertKey]] = [
            set() for _ in range(int(cfg.num_nodes))
        ]
        payload = torch.load(
            self.acquisition_bundle,
            map_location=self.aux_device,
            weights_only=False,
        )
        bundle_format = str(payload.get("format", ""))
        latent_dim = int(payload["latent_dim"])
        advertisement_dim = int(payload.get("advertisement_dim", latent_dim))
        hidden_dim = int(payload["hidden_dim"])
        encoder_state_strict = True
        if bundle_format in {
            "synthetic_staggered_unit_square_grid_gain_bundle_v1",
            "synthetic_unit_square_grid_gain_bundle_v2",
        }:
            if tuple(payload.get("feature_schema", ())) != tuple(
                SHARED_FRAME_PLANE_FEATURE_SCHEMA
            ):
                raise ValueError(
                    "unit-square feature schema differs from simulation"
                )
            self._acquisition_variant = (
                "unit_square_encoding_only_relative_gain"
            )
            self._gain_grid_layout = str(payload.get(
                "grid_layout", GRID_LAYOUT_STAGGERED
            ))
            self._gain_grid_resolution = int(payload["grid_resolution"])
            self._gain_grid_points = unit_square_point_grid(
                self._gain_grid_resolution,
                layout=self._gain_grid_layout,
            )
            self._support_encoder = UnionPlaneSetEncoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            ).to(self.aux_device)
            self._acquisition_model = EncodingOnlyGainModel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            ).to(self.aux_device)
        elif bundle_format == "synthetic_spatial_grid_gain_bundle_v3":
            self._acquisition_variant = "spatial_grid_encoding_relative_gain"
            self._gain_grid_layout = str(payload["grid_layout"])
            self._gain_grid_resolution = int(payload["grid_resolution"])
            self._gain_grid_points = unit_square_point_grid(
                self._gain_grid_resolution,
                layout=self._gain_grid_layout,
            )
            self._support_encoder = SpatialGridEncoder(
                spatial_size=int(payload["spatial_size"]),
                learned_channels=int(payload["learned_channels"]),
                count_scale=float(payload["count_scale"]),
            ).to(self.aux_device)
            self._acquisition_model = SpatialGridGainModel(
                spatial_size=int(payload["spatial_size"]),
                latent_channels=int(payload["latent_channels"]),
                hidden_channels=int(payload["hidden_channels"]),
                hidden_dim=int(payload["hidden_dim"]),
                count_scale=float(payload["count_scale"]),
                maximum_relative_gain=float(payload["maximum_relative_gain"]),
            ).to(self.aux_device)
        elif bundle_format == "synthetic_grid_autoencoder_gain_bundle_v4":
            self._acquisition_variant = "spatial_grid_encoding_relative_gain"
            self._gain_grid_layout = str(payload["grid_layout"])
            self._gain_grid_resolution = int(payload["grid_resolution"])
            self._gain_grid_points = unit_square_point_grid(
                self._gain_grid_resolution,
                layout=self._gain_grid_layout,
            )
            self._support_encoder = GridAutoencoder(
                grid_resolution=self._gain_grid_resolution,
                latent_dim=latent_dim,
                base_channels=int(payload["base_channels"]),
            ).to(self.aux_device)
            self._acquisition_model = GridEncodingGainModel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            ).to(self.aux_device)
            encoder_state_strict = False
        elif bundle_format in {
            "synthetic_sparse_patch_grid_acquisition_bundle_v6",
            "synthetic_quantized_patch_grid_acquisition_bundle_v7",
            "synthetic_product_quantized_patch_grid_acquisition_bundle_v8",
        }:
            self._acquisition_variant = "spatial_grid_encoding_relative_gain"
            self._gain_grid_layout = str(payload["grid_layout"])
            self._gain_grid_resolution = int(payload["grid_resolution"])
            self._gain_grid_points = unit_square_point_grid(
                self._gain_grid_resolution,
                layout=self._gain_grid_layout,
            )
            self._support_encoder = PatchGridCodec(
                grid_resolution=self._gain_grid_resolution,
                patch_size=int(payload["patch_size"]),
                latent_channels=int(payload["latent_channels"]),
                hidden_dim=hidden_dim,
                codebook_size=int(payload.get("codebook_size", 0)),
                codebook_groups=int(payload.get("codebook_groups", 1)),
            ).to(self.aux_device)
            self._acquisition_model = PatchGridGainModel(
                patch_count=int(payload["patches_per_axis"]) ** 2,
                latent_channels=int(payload["latent_channels"]),
                hidden_dim=int(payload["acquisition_hidden_dim"]),
            ).to(self.aux_device)
            self._sparse_patch_advertisements = True
            self._quantized_patch_advertisements = bool(
                int(payload.get("codebook_size", 0))
            )
            self._patch_latent_channels = int(payload["latent_channels"])
            self._patch_count = int(payload["patches_per_axis"]) ** 2
            self._patch_codebook_groups = int(
                payload.get("codebook_groups", 1)
            )
            encoder_state_strict = False
        else:
            raise ValueError(
                f"acquisition bundle {bundle_format!r} does not use the "
                "fixed point-grid gain target"
            )
        encoder_state = payload.get("encoder_state_dict")
        if encoder_state is None:
            encoder_state = payload["codec_state_dict"]
        if (
            "codebook" in encoder_state
            and encoder_state["codebook"].ndim == 2
            and getattr(self._support_encoder, "codebook", None) is not None
            and self._support_encoder.codebook.ndim == 3
        ):
            encoder_state = dict(encoder_state)
            encoder_state["codebook"] = encoder_state["codebook"].unsqueeze(0)
        self._support_encoder.load_state_dict(
            encoder_state, strict=encoder_state_strict
        )
        self._acquisition_model.load_state_dict(
            payload["acquisition_state_dict"]
        )
        self._support_encoder.eval()
        self._acquisition_model.eval()
        for parameter in self._support_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self._acquisition_model.parameters():
            parameter.requires_grad_(False)
        self._advertisement_latent_dim = advertisement_dim
        self._acquisition_checkpoint_step = int(payload.get("best_step", -1))
        for obsolete in (
            "expert_bank_probe_count",
            "expert_bank_probe_source",
            "expert_bank_coverage_refresh",
        ):
            self._communication_assumptions.pop(obsolete, None)
        self._communication_assumptions.update(
            {
                "method": "pretrained learned-acquisition uncapped expert bank",
                "expert_bank_capacity": "none",
                "expert_bank_pruning": (
                    "remove an expert only when its removal leaves the "
                    "pointwise grid maximum unchanged"
                ),
                "geometric_dominance": False,
                "expert_bank_acquisition_score": (
                    "predicted relative intensity-weighted grid gain minus kappa"
                    if self._acquisition_variant in {
                        "union_scalar_relative_gain",
                        "unit_square_encoding_only_relative_gain",
                        "spatial_grid_encoding_relative_gain",
                    }
                    else
                    "frozen synthetic-v2 encoder and acquisition network"
                ),
                "expert_bank_advertisement": (
                    "one byte-sized code per aligned patch, total intensity, experience, and key"
                    if self._quantized_patch_advertisements
                    else "nonempty patch indices and codes, total intensity, experience, and key"
                    if self._sparse_patch_advertisements
                    else "latent encoding, experience, and key"
                    if self._acquisition_variant in {
                        "unit_square_encoding_only_relative_gain",
                        "spatial_grid_encoding_relative_gain",
                    }
                    else
                    "latent encoding, centroid, scale, experience, and key"
                ),
                "full_support_transfer": "only after a model pull",
                "advertisement_cache": (
                    "receiver retains each immutable model-version encoding; "
                    "later contacts resend only model key and experience"
                ),
                "post_pull_validation": (
                    "same fixed-grid gain plus grid-profile redundancy"
                ),
                "post_pull_gain_validation": (
                    "same-kappa relative increase on the fixed normalized "
                    "point grid weighted by maximum sample count"
                ),
                "post_pull_gain_geometry": (
                    "deterministic normalized unit-square lattice "
                    f"({self._gain_grid_layout})"
                ),
                "post_pull_gain_grid_layout": self._gain_grid_layout,
                "post_pull_gain_grid_resolution": self._gain_grid_resolution,
                "post_pull_gain_grid_points": len(self._gain_grid_points),
                "acquisition_bundle": str(self.acquisition_bundle),
                "acquisition_checkpoint_step": (
                    self._acquisition_checkpoint_step
                ),
                "acquisition_probability_threshold": (
                    None
                    if self._acquisition_variant in {
                        "union_scalar_relative_gain",
                        "unit_square_encoding_only_relative_gain",
                        "spatial_grid_encoding_relative_gain",
                    }
                    else self.acquisition_probability_threshold
                ),
                "acquisition_variant": self._acquisition_variant,
                "acquisition_relative_gain_penalty": self.acquisition_relative_gain_penalty,
                "acquisition_utility": "expm1(predicted_log1p_relative_gain) - kappa",
                "expert_bank_routing": self.bank_support_routing,
                "bank_support_union": (
                    "existing remote plane merge rules with exact contributor provenance"
                    if self.bank_support_routing in {
                        "most-experienced", "experience-ensemble"
                    }
                    else False
                ),
                "teacher_distillation_batches_per_step": (
                    self.teacher_distillation_batches_per_step
                ),
                "teacher_distillation": (
                    (
                        "bounded synthetic teacher pass; accepted support "
                        "merged into the primary"
                        if self.bank_support_routing == "primary-only"
                        else "bounded synthetic teacher pass; retained bank "
                        "support gates primary-only RSSI predictions"
                    )
                    if self.teacher_distillation_batches_per_step > 0
                    else False
                ),
                "encoder_plane_storage": (
                    "cached exact normalized support-intensity grid"
                    if self._acquisition_variant
                    == "spatial_grid_encoding_relative_gain"
                    else (
                        "variable-length unit-square normalized [plane, 12] arrays"
                        if self._acquisition_variant
                        == "unit_square_encoding_only_relative_gain"
                        else "variable-length normalized [plane, 12] arrays"
                    )
                ),
            }
        )

    def _make_selection_probes(self) -> np.ndarray:
        """The learned method uses no trace-derived positions or links."""

        return np.empty((0, 2, 2), dtype=np.float64)

    def _result_method_id(self) -> str:
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        return (
            "support_expert_bank_learned_acquisition_unbounded"
            f"_step{self._acquisition_checkpoint_step}{suffix}"
        )

    def _result_method_name(self) -> str:
        suffix = f", {self.method_tag}" if self.method_tag else ""
        return (
            "Learned-acquisition expert bank "
            f"(uncapped, pretrained step {self._acquisition_checkpoint_step}"
            f"{suffix})"
        )

    def _requires_raw_expert_predictions(self) -> bool:
        return self.bank_support_routing in {
            "primary-bank-gate",
            "most-experienced",
            "experience-ensemble",
        }

    def _fidelity_prediction_keys(
        self, active: list[int], unique: list[ExpertKey]
    ) -> set[ExpertKey]:
        if self.bank_support_routing != "primary-bank-gate":
            return super()._fidelity_prediction_keys(active, unique)
        unique_set = set(unique)
        selected: set[ExpertKey] = set()
        for receiver in active:
            own_lineage = (
                int(receiver),
                int(self._expert_incarnations[int(receiver)]),
            )
            own_keys = [
                key
                for key in self._expert_banks[int(receiver)]
                if key in unique_set and key[:2] == own_lineage
            ]
            if own_keys:
                selected.add(max(own_keys, key=lambda key: key[2]))
            else:
                selected.update(
                    key
                    for key in self._expert_banks[int(receiver)]
                    if key in unique_set
                )
        return selected

    def _merged_support_for_bank(
        self, keys: list[ExpertKey]
    ) -> MergedBankSupport:
        cache_key = frozenset(
            key for key in keys if key in self._expert_registry
        )
        cached = self._merged_bank_support_cache.get(cache_key)
        if cached is not None:
            return cached
        rows, source_groups = remote_union_with_sources(
            [
                (key, self._expert_registry[key].capsules)
                for key in sorted(cache_key)
            ],
            self.capsule_params,
        )
        merged = MergedBankSupport(
            capsules=rows,
            contributors=tuple(
                frozenset(
                    source
                    for source in group
                    if source in cache_key
                )
                for group in source_groups
            ),
        )
        self._merged_bank_support_cache[cache_key] = merged
        return merged

    def _route_fidelity_predictions(
        self,
        *,
        step: int,
        receiver: int,
        keys: list[ExpertKey],
        query_m: np.ndarray,
        predictions: dict[ExpertKey, np.ndarray],
        raw_predictions: dict[ExpertKey, np.ndarray],
        routing_profiles: dict[ExpertKey, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.bank_support_routing == "primary-bank-gate":
            own_lineage = (
                int(receiver),
                int(self._expert_incarnations[int(receiver)]),
            )
            own_keys = [
                key for key in keys if key[:2] == own_lineage
            ]
            if own_keys:
                primary = max(own_keys, key=lambda key: key[2])
                confidence = np.max(
                    np.stack(
                        [routing_profiles[key] for key in keys], axis=1
                    ),
                    axis=1,
                )
                floor = float(self._normalize_target_from_rssi(
                    np.asarray(
                        [self.reception_floor_dbm], dtype=np.float32
                    )
                )[0])
                selected = floor + confidence * (
                    raw_predictions[primary] - floor
                )
                return selected, confidence
        if self.bank_support_routing == "primary-only":
            own_lineage = (
                int(receiver),
                int(self._expert_incarnations[int(receiver)]),
            )
            own_keys = [
                key for key in keys if key[:2] == own_lineage
            ]
            if own_keys:
                primary = max(own_keys, key=lambda key: key[2])
                return predictions[primary], routing_profiles[primary]
        if self.bank_support_routing == "individual":
            return super()._route_fidelity_predictions(
                step=step,
                receiver=receiver,
                keys=keys,
                query_m=query_m,
                predictions=predictions,
                raw_predictions=raw_predictions,
                routing_profiles=routing_profiles,
            )
        merged = self._merged_support_for_bank(keys)
        query_count = len(query_m)
        floor = float(self._normalize_target_from_rssi(
            np.asarray([self.reception_floor_dbm], dtype=np.float32)
        )[0])
        selected = np.full(query_count, floor, dtype=np.float32)
        confidence = np.zeros(query_count, dtype=np.float64)
        chosen_plane = np.full(query_count, -1, dtype=np.int64)
        plane_chunk = 64
        for start in range(0, len(merged.capsules), plane_chunk):
            stop = min(len(merged.capsules), start + plane_chunk)
            matrix = support_strength_matrix(
                merged.capsules[start:stop],
                query_m,
                self.capsule_params,
            )
            local_choice = np.argmax(matrix, axis=1)
            local_strength = matrix[np.arange(query_count), local_choice]
            improves = local_strength > confidence
            confidence[improves] = local_strength[improves]
            chosen_plane[improves] = start + local_choice[improves]
        for plane_index in np.unique(chosen_plane[chosen_plane >= 0]):
            query_indices = np.flatnonzero(
                chosen_plane == int(plane_index)
            )
            contributors = sorted(
                key
                for key in merged.contributors[int(plane_index)]
                if key in raw_predictions
            )
            if not contributors:
                continue
            if self.bank_support_routing == "most-experienced":
                chosen = max(
                    contributors,
                    key=lambda key: (
                        int(self._expert_registry[key].experience),
                        key,
                    ),
                )
                selected[query_indices] = raw_predictions[chosen][
                    query_indices
                ]
                continue
            weights = np.asarray(
                [
                    max(1, int(self._expert_registry[key].experience))
                    for key in contributors
                ],
                dtype=np.float64,
            )
            values = np.stack(
                [
                    raw_predictions[key][query_indices]
                    for key in contributors
                ],
                axis=1,
            )
            selected[query_indices] = (
                values @ weights / float(weights.sum())
            )
        self._merged_routing_stats[int(receiver)] = (
            len(merged.capsules),
            float(np.mean([
                len(group) for group in merged.contributors
            ])) if merged.contributors else 0.0,
        )
        return selected, confidence

    def _evaluate_fidelity_now(
        self, step: int, *, n_pairs: int, is_final: int
    ) -> dict[str, float | int]:
        self._merged_routing_stats = {}
        metrics = super()._evaluate_fidelity_now(
            step, n_pairs=n_pairs, is_final=is_final
        )
        if self.bank_support_routing != "individual":
            values = list(self._merged_routing_stats.values())
            metrics["expert_bank_merged_mean_planes"] = (
                float(np.mean([value[0] for value in values]))
                if values else 0.0
            )
            metrics["expert_bank_merged_mean_contributors_per_plane"] = (
                float(np.mean([value[1] for value in values]))
                if values else 0.0
            )
        return metrics

    def _encode_plane_rows(
        self, rows: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray | None, float | None, np.ndarray
    ]:
        values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
        encoding_only = self._acquisition_variant in {
            "unit_square_encoding_only_relative_gain",
            "spatial_grid_encoding_relative_gain",
        }
        if len(values) == 0:
            return (
                np.empty((0, 12), dtype=np.float32),
                None if encoding_only else np.zeros(2, dtype=np.float32),
                None if encoding_only else 1.0,
                np.zeros(self._advertisement_latent_dim, dtype=np.float32),
            )
        if self._acquisition_variant == "spatial_grid_encoding_relative_gain":
            profile = grid_support_counts(
                values,
                resolution=self._gain_grid_resolution,
                map_size=float(self.cfg.map_size),
                layout=self._gain_grid_layout,
            ).reshape(1, self._gain_grid_resolution, self._gain_grid_resolution)
            with torch.no_grad():
                encoding = self._support_encoder(torch.tensor(
                    profile, dtype=torch.float32, device=self.aux_device
                ))[0]
            return (
                np.empty((0, 12), dtype=np.float32),
                None,
                None,
                encoding.detach().cpu().numpy().astype(np.float32, copy=False),
            )
        if self._acquisition_variant == "unit_square_encoding_only_relative_gain":
            normalized = normalize_plane_set_shared_frame(
                values, map_size=float(self.cfg.map_size)
            )
            center = None
            scale = None
        elif self._acquisition_variant == "union_scalar_relative_gain":
            normalized, center, scale = normalize_planes_for_union_encoder(values)
        else:
            normalized, center, scale = normalize_planes_for_encoder(
                values, float(self.capsule_params.mass_scale), "v2"
            )
        with torch.no_grad():
            features = torch.as_tensor(
                normalized, dtype=torch.float32, device=self.aux_device
            )
            plane_to_set = torch.zeros(
                len(normalized), dtype=torch.long, device=self.aux_device
            )
            encoding = (
                self._support_encoder(features, plane_to_set, 1)[0]
                .detach().cpu().numpy().astype(np.float32, copy=False)
            )
        return (
            normalized,
            None if center is None else np.asarray(center),
            None if scale is None else float(scale),
            encoding,
        )

    def _advertisement_for_key(
        self, key: ExpertKey
    ) -> EncodedExpertAdvertisement:
        advertisement = self._encoded_advertisements.get(key)
        if advertisement is not None:
            return advertisement
        record = self._expert_registry[key]
        rows = np.asarray(record.capsules, dtype=np.float64).reshape(-1, 11)
        if self._acquisition_variant == "spatial_grid_encoding_relative_gain":
            profile = self._grid_profile_for_key(key).reshape(
                1, self._gain_grid_resolution, self._gain_grid_resolution
            )
            with torch.no_grad():
                encoded = self._support_encoder(torch.tensor(
                    profile, dtype=torch.float32, device=self.aux_device
                ))[0]
            normalized = np.empty((0, 12), dtype=np.float32)
            center = None
            scale = None
            encoding = encoded.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        else:
            normalized, center, scale, encoding = self._encode_plane_rows(rows)
        advertisement = EncodedExpertAdvertisement(
            key=key,
            normalized_planes=normalized,
            encoding=encoding,
            center_xy_m=(
                None
                if center is None
                else np.asarray(center, dtype=np.float32)
            ),
            scale_m=None if scale is None else float(scale),
            experience=int(record.experience),
        )
        self._encoded_advertisements[key] = advertisement
        return advertisement

    def _ensure_grid_advertisements(
        self, keys: list[ExpertKey], *, batch_size: int = 32
    ) -> None:
        """Encode missing spatial-grid advertisements in exact batches."""

        if self._acquisition_variant != "spatial_grid_encoding_relative_gain":
            return
        missing = sorted({
            key
            for key in keys
            if key in self._expert_registry
            and key not in self._encoded_advertisements
        })
        for start in range(0, len(missing), max(1, int(batch_size))):
            chunk = missing[start : start + max(1, int(batch_size))]
            profiles = np.stack([
                self._grid_profile_for_key(key).reshape(
                    self._gain_grid_resolution, self._gain_grid_resolution
                )
                for key in chunk
            ])
            with torch.no_grad():
                encoded = self._support_encoder(torch.as_tensor(
                    profiles,
                    dtype=torch.float32,
                    device=self.aux_device,
                )).detach().cpu().numpy().astype(np.float32, copy=False)
            for key, encoding in zip(chunk, encoded, strict=True):
                record = self._expert_registry[key]
                self._encoded_advertisements[key] = EncodedExpertAdvertisement(
                    key=key,
                    normalized_planes=np.empty((0, 12), dtype=np.float32),
                    encoding=encoding,
                    center_xy_m=None,
                    scale_m=None,
                    experience=int(record.experience),
                )

    def _reset_aux_node(
        self,
        i: int,
        *,
        old_az: int | None = None,
        new_az: int | None = None,
    ) -> None:
        super()._reset_aux_node(i, old_az=old_az, new_az=new_az)
        index = int(i)
        if index < len(self._rejected_candidates):
            self._rejected_candidates[index].clear()

    def _grid_profile_for_key(self, key: ExpertKey) -> np.ndarray:
        profile = self._grid_profile_cache.get(key)
        if profile is None:
            profile = grid_support_counts(
                self._expert_registry[key].capsules,
                resolution=self._gain_grid_resolution,
                map_size=float(self.cfg.map_size),
                layout=self._gain_grid_layout,
            )
            profile.setflags(write=False)
            self._grid_profile_cache[key] = profile
            self._grid_profile_evaluations += 1
        return profile

    def _grid_bank_profile(self, keys: list[ExpertKey]) -> np.ndarray:
        cache_key = frozenset(
            key for key in keys if key in self._expert_registry
        )
        profile = self._grid_bank_profile_cache.get(cache_key)
        if profile is None:
            profile = np.zeros(
                len(self._gain_grid_points), dtype=np.float32
            )
            for key in sorted(cache_key):
                np.maximum(
                    profile, self._grid_profile_for_key(key), out=profile
                )
            profile.setflags(write=False)
            self._grid_bank_profile_cache[cache_key] = profile
            self._grid_bank_profile_evaluations += 1
        return profile

    def _cache_grid_bank_profile(
        self,
        keys: list[ExpertKey],
        profile: np.ndarray,
    ) -> None:
        cache_key = frozenset(
            key for key in keys if key in self._expert_registry
        )
        profile.setflags(write=False)
        self._grid_bank_profile_cache[cache_key] = profile

    def _measured_relative_gain(
        self,
        bank: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[float, float]:
        return relative_point_grid_gain(bank, candidate)

    def _select_bank(
        self, receiver: int, candidates: list[ExpertKey]
    ) -> list[ExpertKey]:
        """Remove only experts that do not affect the grid maximum."""

        keys = list(dict.fromkeys(
            key for key in candidates if key in self._expert_registry
        ))
        protected: set[ExpertKey] = set()
        if self.teacher_distillation_batches_per_step > 0:
            own_lineage = (
                int(receiver),
                int(self._expert_incarnations[int(receiver)]),
            )
            own_keys = [key for key in keys if key[:2] == own_lineage]
            if own_keys:
                newest_own = max(own_keys, key=lambda key: key[2])
                keys = [
                    key
                    for key in keys
                    if key[:2] != own_lineage or key == newest_own
                ]
                protected.add(newest_own)
        while len(keys) > 1:
            profiles = np.stack([
                self._grid_profile_for_key(key) for key in keys
            ])
            maximum = np.max(profiles, axis=0)
            at_maximum = profiles == maximum[None, :]
            maximum_count = np.sum(at_maximum, axis=0)
            essential = np.any(
                at_maximum & (maximum_count[None, :] == 1), axis=1
            )
            self._grid_redundancy_evaluations += len(keys)
            removable = [
                index for index, required in enumerate(essential)
                if not bool(required) and keys[index] not in protected
            ]
            if not removable:
                break
            remove_index = min(
                removable,
                key=lambda index: (
                    int(self._expert_registry[keys[index]].experience),
                    keys[index],
                ),
            )
            keys.pop(remove_index)
        return keys

    def _insert_candidate(
        self,
        receiver: int,
        bank: list[ExpertKey],
        candidate: ExpertKey,
    ) -> list[ExpertKey]:
        keys = list(dict.fromkeys(
            key for key in bank if key in self._expert_registry
        ))
        if candidate not in self._expert_registry or candidate in keys:
            return keys
        return self._select_bank(receiver, [*keys, candidate])

    def _unit_square_encoder_pool(
        self, key: ExpertKey
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Cache exact DeepSets sufficient statistics for one expert."""

        cached = self._unit_square_encoder_pools.get(key)
        if cached is not None:
            return cached
        advertisement = self._advertisement_for_key(key)
        features = torch.as_tensor(
            advertisement.normalized_planes,
            dtype=torch.float32,
            device=self.aux_device,
        )
        with torch.no_grad():
            if len(features):
                hidden = self._support_encoder.plane_mlp(features)
                total = hidden.sum(dim=0)
                maximum = hidden.amax(dim=0)
                count = int(hidden.shape[0])
            else:
                hidden_dim = int(self._support_encoder.hidden_dim)
                total = torch.zeros(hidden_dim, device=self.aux_device)
                maximum = torch.zeros(hidden_dim, device=self.aux_device)
                count = 0
        cached = (total, maximum, count)
        self._unit_square_encoder_pools[key] = cached
        return cached

    def _unit_square_bank_encoding(
        self, keys: list[ExpertKey]
    ) -> np.ndarray:
        """Encode the receiver bank in the bundle's shared map frame."""

        if self._acquisition_variant == "spatial_grid_encoding_relative_gain":
            cache_key = frozenset(
                key for key in keys if key in self._expert_registry
            )
            cached = self._grid_bank_encoding_cache.get(cache_key)
            if cached is not None:
                return cached
            if len(cache_key) == 1:
                only_key = next(iter(cache_key))
                result = self._advertisement_for_key(only_key).encoding
                self._grid_bank_encoding_cache[cache_key] = result
                return result
            profile = self._grid_bank_profile(keys).reshape(
                1, self._gain_grid_resolution, self._gain_grid_resolution
            )
            with torch.no_grad():
                encoding = self._support_encoder(torch.tensor(
                    profile, dtype=torch.float32, device=self.aux_device
                ))[0]
            result = encoding.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            result.setflags(write=False)
            self._grid_bank_encoding_cache[cache_key] = result
            return result
        pools = [
            self._unit_square_encoder_pool(key)
            for key in keys
            if key in self._expert_registry
        ]
        if not pools:
            return np.zeros(
                self._advertisement_latent_dim, dtype=np.float32
            )
        total = torch.stack([pool[0] for pool in pools]).sum(dim=0)
        count = sum(pool[2] for pool in pools)
        if count:
            maximum = torch.stack([
                pool[1] for pool in pools if pool[2]
            ]).amax(dim=0)
            mean = total / float(count)
        else:
            maximum = torch.zeros_like(total)
            mean = torch.zeros_like(total)
        with torch.no_grad():
            encoding = self._support_encoder.output_mlp(
                torch.cat((mean, maximum))[None, :]
            )[0]
        return (
            encoding.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    def _union_provider_candidate(
        self,
        receiver_keys: list[ExpertKey],
        usable: list[ExpertKey],
        *,
        minimum_relative_gain: float | None = None,
    ) -> ExpertKey | None:
        if self._acquisition_variant in {
            "unit_square_encoding_only_relative_gain",
            "spatial_grid_encoding_relative_gain",
        }:
            if self._acquisition_variant == "spatial_grid_encoding_relative_gain":
                has_bank_support = any(
                    np.any(self._grid_profile_for_key(key) > 0.0)
                    for key in receiver_keys
                    if key in self._expert_registry
                )
            else:
                has_bank_support = any(
                    self._expert_registry[key].capsules
                    for key in receiver_keys
                    if key in self._expert_registry
                )
            if not has_bank_support:
                return None
            bank_center = None
            bank_scale = None
            bank_encoding = self._unit_square_bank_encoding(receiver_keys)
        else:
            bank_parts = [
                np.asarray(
                    self._expert_registry[key].capsules, dtype=np.float64
                ).reshape(-1, 11)
                for key in receiver_keys
                if key in self._expert_registry
            ]
            nonempty = [rows for rows in bank_parts if len(rows)]
            if not nonempty:
                return None
            bank_rows = np.concatenate(nonempty, axis=0)
            _, bank_center, bank_scale, bank_encoding = (
                self._encode_plane_rows(bank_rows)
            )
        candidates = [
            self._advertisement_for_key(key) for key in usable
        ]
        embeddings = torch.as_tensor(
            np.stack([bank_encoding, *[item.encoding for item in candidates]]),
            dtype=torch.float32,
            device=self.aux_device,
        )
        candidate_indices = torch.arange(
            1,
            len(candidates) + 1,
            dtype=torch.long,
            device=self.aux_device,
        )
        bank_indices = torch.zeros_like(candidate_indices)
        with torch.no_grad():
            if self._acquisition_variant in {
                "unit_square_encoding_only_relative_gain",
                "spatial_grid_encoding_relative_gain",
            }:
                predicted_log_gain = self._acquisition_model(
                    embeddings,
                    candidate_indices,
                    bank_indices,
                )
            else:
                centers = torch.as_tensor(
                    np.stack([
                        bank_center,
                        *[item.center_xy_m for item in candidates],
                    ]),
                    dtype=torch.float32,
                    device=self.aux_device,
                )
                scales = torch.as_tensor(
                    [bank_scale, *[item.scale_m for item in candidates]],
                    dtype=torch.float32,
                    device=self.aux_device,
                )
                predicted_log_gain = self._acquisition_model(
                    embeddings,
                    centers,
                    scales,
                    candidate_indices,
                    bank_indices,
                )
            predicted_relative_gain = torch.expm1(predicted_log_gain)
        relative_values = predicted_relative_gain.detach().cpu().numpy()
        gain_threshold = (
            self.acquisition_relative_gain_penalty
            if minimum_relative_gain is None
            else float(minimum_relative_gain)
        )
        eligible = [
            index
            for index, gain in enumerate(relative_values)
            if float(gain) > gain_threshold
        ]
        if not eligible:
            return None
        chosen_index = max(
            eligible,
            key=lambda index: (
                float(relative_values[index]),
                int(self._expert_registry[usable[index]].experience),
                usable[index],
            ),
        )
        chosen = usable[chosen_index]
        log_gain = float(predicted_log_gain[chosen_index])
        relative_gain = float(relative_values[chosen_index])
        self._latest_acquisition_prediction = (
            chosen,
            log_gain,
            relative_gain,
            relative_gain - gain_threshold,
        )
        return chosen

    def _provider_candidate(
        self,
        receiver_keys: list[ExpertKey],
        provider_keys: list[ExpertKey],
    ) -> ExpertKey | None:
        self._latest_acquisition_prediction = None
        if not receiver_keys:
            return None
        exact = set(receiver_keys)
        usable = [
            key
            for key in provider_keys
            if key in self._expert_registry and key not in exact
        ]
        if not usable:
            return None
        if self._acquisition_variant in {
            "union_scalar_relative_gain",
            "unit_square_encoding_only_relative_gain",
            "spatial_grid_encoding_relative_gain",
        }:
            return self._union_provider_candidate(
                receiver_keys,
                usable,
            )
        bank_advertisements = [
            self._advertisement_for_key(key) for key in receiver_keys
        ]
        candidate_advertisements = [
            self._advertisement_for_key(key) for key in usable
        ]
        all_advertisements = [
            *bank_advertisements, *candidate_advertisements
        ]
        embeddings = torch.as_tensor(
            np.stack([item.encoding for item in all_advertisements]),
            dtype=torch.float32,
            device=self.aux_device,
        )
        centers = torch.as_tensor(
            np.stack([item.center_xy_m for item in all_advertisements]),
            dtype=torch.float32,
            device=self.aux_device,
        )
        scales = torch.as_tensor(
            [item.scale_m for item in all_advertisements],
            dtype=torch.float32,
            device=self.aux_device,
        )
        bank_count = len(bank_advertisements)
        candidate_count = len(candidate_advertisements)
        candidate_indices = torch.arange(
            bank_count,
            bank_count + candidate_count,
            dtype=torch.long,
            device=self.aux_device,
        )
        bank_indices = torch.arange(
            bank_count,
            dtype=torch.long,
            device=self.aux_device,
        )[None, :].expand(candidate_count, -1)
        bank_mask = torch.ones_like(bank_indices, dtype=torch.bool)
        with torch.no_grad():
            gains, logits = self._acquisition_model(
                embeddings,
                centers,
                scales,
                candidate_indices,
                bank_indices,
                bank_mask,
            )
            probabilities = torch.sigmoid(logits)
        gain_values = gains.detach().cpu().numpy()
        probability_values = probabilities.detach().cpu().numpy()
        eligible = [
            index
            for index, probability in enumerate(probability_values)
            if float(probability) >= self.acquisition_probability_threshold
        ]
        if not eligible:
            return None
        chosen_index = max(
            eligible,
            key=lambda index: (
                float(gain_values[index, 1]),
                float(gain_values[index, 0]),
                float(probability_values[index]),
                int(self._expert_registry[usable[index]].experience),
                usable[index],
            ),
        )
        chosen = usable[chosen_index]
        self._latest_acquisition_prediction = (
            chosen,
            float(gain_values[chosen_index, 0]),
            float(gain_values[chosen_index, 1]),
            float(probability_values[chosen_index]),
        )
        return chosen

    def _sample_teacher_supported_links(
        self,
        rows: tuple[CapsuleRow, ...],
        count: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not rows or count <= 0:
            return (
                np.empty((0, 2, 2), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
        plane_count = len(values)
        choices = rng.choice(
            plane_count,
            size=int(count),
            replace=int(count) > plane_count,
        )
        pairs: list[np.ndarray] = []
        selected_planes: list[int] = []
        for plane_index in choices:
            row = values[int(plane_index)]
            start, end = row[0:2], row[2:4]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 1.0:
                continue
            axis = vector / length
            normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            pair: np.ndarray | None = None
            for _attempt in range(32):
                fractions = rng.random(2)
                low = (
                    (1.0 - fractions) * row[4]
                    + fractions * row[6]
                )
                high = (
                    (1.0 - fractions) * row[5]
                    + fractions * row[7]
                )
                lateral = low + rng.random(2) * (high - low)
                points = (
                    start[None, :]
                    + fractions[:, None] * vector[None, :]
                    + lateral[:, None] * normal[None, :]
                )
                link_length = float(np.linalg.norm(points[1] - points[0]))
                if (
                    link_length >= 1.0
                    and link_length
                    <= float(row[9]) + float(
                        self.capsule_params.link_length_margin_m
                    )
                ):
                    pair = points
                    break
            if pair is None:
                continue
            if bool(rng.integers(0, 2)):
                pair = pair[::-1]
            pairs.append(pair)
            selected_planes.append(int(plane_index))
        if not pairs:
            return (
                np.empty((0, 2, 2), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        return (
            np.asarray(pairs, dtype=np.float64),
            np.asarray(selected_planes, dtype=np.int64),
        )

    def _merge_pulled_teacher_support(
        self,
        accepted_by_receiver: dict[int, list[ExpertKey]],
    ) -> None:
        if (
            self.teacher_distillation_batches_per_step <= 0
            or self.bank_support_routing != "primary-only"
        ):
            return
        for receiver, candidates in accepted_by_receiver.items():
            own_lineage = (
                int(receiver),
                int(self._expert_incarnations[int(receiver)]),
            )
            teacher_rows = [
                self._expert_registry[key].capsules
                for key in dict.fromkeys(candidates)
                if key in self._expert_registry
                and key[:2] != own_lineage
            ]
            if not teacher_rows:
                continue
            self._local_support[int(receiver)] = remote_union(
                [self._local_support[int(receiver)], *teacher_rows],
                self.capsule_params,
            )
            self.greedy_models[int(receiver)].set_ribbons(
                self._local_support[int(receiver)]
            )
            self._teacher_support_merges += len(teacher_rows)

    def _teacher_keys_for_receiver(
        self, receiver: int
    ) -> list[ExpertKey]:
        own_lineage = (
            int(receiver),
            int(self._expert_incarnations[int(receiver)]),
        )
        return [
            key
            for key in dict.fromkeys(self._expert_banks[int(receiver)])
            if key in self._expert_registry
            and key[:2] != own_lineage
            and bool(self._expert_registry[key].capsules)
        ]

    def _additional_training_receivers(
        self, active: set[int]
    ) -> set[int]:
        if self.teacher_distillation_batches_per_step <= 0:
            return set()
        return {
            receiver
            for receiver in active
            if self._teacher_keys_for_receiver(receiver)
        }

    def _train_additional_receiver_data(
        self,
        step: int,
        receiver: int,
        rng: np.random.Generator,
    ) -> bool:
        del step
        batches = self.teacher_distillation_batches_per_step
        teachers = self._teacher_keys_for_receiver(receiver)
        if batches <= 0 or not teachers:
            return False
        sample_count = int(self.cfg.local_batch_size) * int(batches)
        assignments = rng.integers(
            0, len(teachers), size=sample_count
        )
        features_parts: list[np.ndarray] = []
        targets_parts: list[np.ndarray] = []
        teacher_model = copy.deepcopy(self.greedy_models[int(receiver)])
        teacher_model.eval()
        for teacher_index in np.unique(assignments):
            key = teachers[int(teacher_index)]
            record = self._expert_registry[key]
            pairs, _plane_indices = self._sample_teacher_supported_links(
                record.capsules,
                int(np.count_nonzero(assignments == teacher_index)),
                rng,
            )
            if len(pairs) == 0:
                continue
            features = (
                pairs.reshape(-1, 4) / float(self.cfg.map_size)
            ).astype(np.float32)
            teacher_model.load_state_dict(record.model_state)
            teacher_model.set_ribbons(record.capsules)
            with torch.no_grad():
                normalized, confidence = (
                    teacher_model.forward_with_confidence(
                        torch.as_tensor(
                            features,
                            dtype=torch.float32,
                            device=self.aux_device,
                        )
                    )
                )
            valid = (
                confidence.detach().cpu().numpy().reshape(-1) >= 0.5
            )
            if not bool(np.any(valid)):
                continue
            features_parts.append(features[valid])
            targets_parts.append(
                self._denorm_dbm(
                    normalized.detach().cpu().numpy().reshape(-1)[valid]
                ).reshape(-1, 1)
            )
        if not features_parts:
            return False
        features = np.concatenate(features_parts, axis=0)
        targets = np.concatenate(targets_parts, axis=0)
        self._train_array(
            int(receiver), features, targets, epochs=1, rng=rng
        )
        self._teacher_distillation_passes += 1
        self._teacher_distillation_samples += int(len(features))
        return True

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        del zone_nodes
        step = int(getattr(self, "_current_sumo_step", 0))
        if step <= self._resume_step:
            self.sharing_rows.clear()
            self.local_policy_rows.clear()
            return 0
        self._restore_logs()
        links = sorted({
            (int(zone), min(int(a), int(b)), max(int(a), int(b)))
            for zone, a, b in (contact_links or [])
            if int(a) != int(b)
        })
        neighbours: dict[int, list[int]] = {}
        for _zone, left, right in links:
            neighbours.setdefault(left, []).append(right)
            neighbours.setdefault(right, []).append(left)
        pre_banks = {
            index: list(self._expert_banks[index]) for index in neighbours
        }
        self._ensure_grid_advertisements([
            key
            for bank in pre_banks.values()
            for key in bank
        ])
        next_banks: dict[int, list[ExpertKey]] = {}
        pull_attempts = 0
        pull_accepts = 0
        pull_rejections = 0
        manifest_records = 0
        advertisement_values = 0
        support_values = 0
        model_parameter_values = 0
        def advertisement_encoding_scalar_values(key: ExpertKey) -> int:
            if self._quantized_patch_advertisements:
                # Code indices use one byte each and total intensity uses one
                # ordinary 32-bit scalar.  Key and experience are the
                # separately counted four-scalar manifest record.
                payload_bytes = (
                    self._patch_count * self._patch_codebook_groups + 4
                )
                return int((payload_bytes + 3) // 4)
            if not self._sparse_patch_advertisements:
                center_and_scale = (
                    0
                    if self._acquisition_variant in {
                        "unit_square_encoding_only_relative_gain",
                        "spatial_grid_encoding_relative_gain",
                    }
                    else 3
                )
                return int(
                    self._advertisement_latent_dim + center_and_scale
                )
            encoding = self._advertisement_for_key(key).encoding
            codes = encoding[:-1].reshape(
                self._patch_count, self._patch_latent_channels
            )
            active_patches = int(np.count_nonzero(
                np.any(codes != 0.0, axis=1)
            ))
            # One patch index per nonempty code and total intensity.  Key and
            # experience are counted in the manifest record.
            return int(
                active_patches * (self._patch_latent_channels + 1) + 1
            )

        def process_receiver(receiver: int) -> tuple[Any, ...]:
            local_attempts = 0
            local_accepts = 0
            local_rejections = 0
            local_manifest_records = 0
            local_advertisement_values = 0
            local_support_values = 0
            local_model_parameter_values = 0
            local_post_pull_gain_rejections = 0
            local_upper_bound_rejections = 0
            local_accepted_candidates: list[ExpertKey] = []
            # Every stored bank is pruned after local training and after each
            # pull, so no quadratic full-bank cleanup is needed here.
            working = list(dict.fromkeys(
                key
                for key in pre_banks[receiver]
                if key in self._expert_registry
            ))
            working_profile = self._grid_bank_profile(working)
            working_score = float(np.sum(
                working_profile, dtype=np.float64
            ))
            known_model_ids = set(working)
            rejected_model_ids = self._rejected_candidates[receiver]
            cached_advertisements = self._receiver_advertisement_cache[
                receiver
            ]
            offers: list[ExpertKey] = []
            offered_model_ids: set[ExpertKey] = set()
            for sender in sorted(neighbours[receiver]):
                sender_bank = pre_banks[sender]
                local_manifest_records += len(sender_bank)
                local_advertisement_values += 4 * len(sender_bank)
                for model_id in sender_bank:
                    if (
                        model_id in self._expert_registry
                        and model_id not in cached_advertisements
                    ):
                        local_advertisement_values += (
                            advertisement_encoding_scalar_values(model_id)
                        )
                        cached_advertisements.add(model_id)
                    if (
                        model_id not in self._expert_registry
                        or model_id in known_model_ids
                        or model_id in rejected_model_ids
                        or model_id in offered_model_ids
                    ):
                        continue
                    offered_model_ids.add(model_id)
                    offers.append(model_id)
            remaining_offers = list(offers)
            while remaining_offers:
                candidate = self._provider_candidate(
                    working, remaining_offers
                )
                if candidate is None:
                    break
                remaining_offers = [
                    model_id
                    for model_id in remaining_offers
                    if model_id != candidate
                ]
                local_attempts += 1
                record = self._expert_registry[candidate]
                local_support_values += self._pulled_support_payload_values(candidate)
                local_model_parameter_values += sum(
                    int(value.numel()) for value in record.model_state.values()
                )
                candidate_profile = self._grid_profile_for_key(candidate)
                candidate_score = float(np.sum(
                    candidate_profile, dtype=np.float64
                ))
                gain_upper_bound = (
                    candidate_score / max(working_score, 1.0)
                )
                measured_gain = 0.0
                gain_amount = 0.0
                if (
                    gain_upper_bound
                    > self.acquisition_relative_gain_penalty
                ):
                    measured_gain, gain_amount = self._measured_relative_gain(
                        working_profile,
                        candidate_profile,
                    )
                else:
                    local_upper_bound_rejections += 1
                gain_passes = (
                    measured_gain
                    > self.acquisition_relative_gain_penalty
                )
                if gain_passes:
                    working.append(candidate)
                    working_profile = np.maximum(
                        working_profile, candidate_profile
                    )
                    working_score += gain_amount
                    known_model_ids.add(candidate)
                    local_accepts += 1
                    local_accepted_candidates.append(candidate)
                else:
                    rejected_model_ids.add(candidate)
                    local_rejections += 1
                    local_post_pull_gain_rejections += 1
            if local_accepts:
                # The support envelope is all acquisition needs.  Prune once
                # after the receiver has processed its offers rather than
                # rebuilding the full expert stack after every accepted pull.
                working = self._select_bank(receiver, working)
                self._cache_grid_bank_profile(
                    working,
                    working_profile,
                )
            return (
                receiver,
                working,
                local_attempts,
                local_accepts,
                local_rejections,
                local_manifest_records,
                local_advertisement_values,
                local_support_values,
                local_model_parameter_values,
                local_post_pull_gain_rejections,
                local_upper_bound_rejections,
                local_accepted_candidates,
            )

        receivers = sorted(neighbours)
        receiver_results = [
            process_receiver(receiver) for receiver in receivers
        ]
        accepted_by_receiver: dict[int, list[ExpertKey]] = {}
        for result in receiver_results:
            (
                receiver,
                bank,
                local_attempts,
                local_accepts,
                local_rejections,
                local_manifest_records,
                local_advertisement_values,
                local_support_values,
                local_model_parameter_values,
                local_post_pull_gain_rejections,
                local_upper_bound_rejections,
                local_accepted_candidates,
            ) = result
            next_banks[int(receiver)] = bank
            if local_accepted_candidates:
                accepted_by_receiver[int(receiver)] = (
                    local_accepted_candidates
                )
            pull_attempts += int(local_attempts)
            pull_accepts += int(local_accepts)
            pull_rejections += int(local_rejections)
            manifest_records += int(local_manifest_records)
            advertisement_values += int(local_advertisement_values)
            support_values += int(local_support_values)
            model_parameter_values += int(local_model_parameter_values)
            self._coverage_recalculations += int(local_attempts)
            self._post_pull_gain_rejections += int(
                local_post_pull_gain_rejections
            )
            self._gain_upper_bound_rejections += int(
                local_upper_bound_rejections
            )
        for receiver, bank in next_banks.items():
            self._expert_banks[receiver] = bank
        self._merge_pulled_teacher_support(
            accepted_by_receiver
        )
        self._learned_pull_attempts += pull_attempts
        self._learned_pull_accepts += pull_accepts
        self._learned_pull_rejections += pull_rejections
        self._model_transfers += pull_attempts
        self._manifest_records += manifest_records
        self._advertisement_scalar_values += advertisement_values
        self._pulled_support_scalar_values += support_values
        self._pulled_model_parameter_values += model_parameter_values
        self._network_step_stats.update(
            {
                "expert_bank_model_messages": int(pull_attempts),
                "expert_bank_accepted_pulls": int(pull_accepts),
                "expert_bank_rejected_pulls": int(pull_rejections),
                "expert_bank_manifest_records": int(manifest_records),
                "expert_advertisement_scalar_values": int(
                    advertisement_values
                ),
                "expert_advertisement_bytes": int(4 * advertisement_values),
                "capsule_scalar_values_sent": int(support_values),
                "capsule_payload_bytes": int(4 * support_values),
                "model_parameter_values_pulled": int(model_parameter_values),
                "model_payload_bytes": int(4 * model_parameter_values),
                "expert_bank_receivers": int(len(next_banks)),
                "expert_bank_coverage_recalculations": int(
                    self._coverage_recalculations
                ),
                "teacher_support_merges": int(self._teacher_support_merges),
                "teacher_distillation_passes": int(self._teacher_distillation_passes),
                "teacher_distillation_samples": int(self._teacher_distillation_samples),
            }
        )
        self._train_staged_local_samples(step)
        return int(pull_attempts)

    def _refresh_local_expert(self, receiver: int) -> None:
        index = int(receiver)
        previous = list(self._expert_banks[index])
        previous_profile = self._grid_bank_profile(previous)
        self._local_versions[index] += 1
        key = (
            index,
            int(self._expert_incarnations[index]),
            int(self._local_versions[index]),
        )
        self._expert_registry[key] = ExpertRecord(
            key=key,
            experience=int(self.greedy_m_samples[index]),
            capsules=self._local_support[index],
            model_state=_cpu_state(self.greedy_models[index]),
        )
        self._expert_banks[index] = self._insert_candidate(
            index, previous, key
        )
        if key in self._expert_banks[index]:
            if key not in previous:
                updated_profile = np.maximum(
                    previous_profile,
                    self._grid_profile_for_key(key),
                )
                self._cache_grid_bank_profile(
                    self._expert_banks[index],
                    updated_profile,
                )
            self._advertisement_for_key(key)

    def _prune_registry(self) -> None:
        super()._prune_registry()
        self._encoded_advertisements = {
            key: advertisement
            for key, advertisement in self._encoded_advertisements.items()
            if key in self._expert_registry
        }
        self._unit_square_encoder_pools = {
            key: pool
            for key, pool in self._unit_square_encoder_pools.items()
            if key in self._expert_registry
        }
        self._grid_profile_cache = {
            key: profile
            for key, profile in self._grid_profile_cache.items()
            if key in self._expert_registry
        }
        live_keys = set(self._expert_registry)
        self._grid_bank_profile_cache = {
            keys: profile
            for keys, profile in self._grid_bank_profile_cache.items()
            if keys.issubset(live_keys)
        }
        self._grid_bank_encoding_cache = {
            keys: encoding
            for keys, encoding in self._grid_bank_encoding_cache.items()
            if keys.issubset(live_keys)
        }
        self._merged_bank_support_cache = {
            keys: merged
            for keys, merged in self._merged_bank_support_cache.items()
            if keys.issubset(live_keys)
        }
        self._rejected_candidates = [
            {key for key in rejected if key in self._expert_registry}
            for rejected in self._rejected_candidates
        ]
        self._receiver_advertisement_cache = [
            {key for key in cached if key in self._expert_registry}
            for cached in self._receiver_advertisement_cache
        ]

    def _save_checkpoint(self, step: int) -> None:
        super()._save_checkpoint(step)
        output = Path(self.cfg.results_dir)
        status_path = output / "checkpoint_status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        status["learned_acquisition"] = {
            "bundle": str(self.acquisition_bundle),
            "pretraining_step": self._acquisition_checkpoint_step,
            "variant": self._acquisition_variant,
            "relative_gain_penalty": self.acquisition_relative_gain_penalty,
            "pull_attempts": self._learned_pull_attempts,
            "accepted_pulls": self._learned_pull_accepts,
            "rejected_pulls": self._learned_pull_rejections,
            "post_pull_gain_rejections": self._post_pull_gain_rejections,
            "teacher_batches_per_step": self.teacher_distillation_batches_per_step,
            "teacher_support_merges": self._teacher_support_merges,
            "teacher_distillation_passes": self._teacher_distillation_passes,
            "teacher_distillation_samples": self._teacher_distillation_samples,
            "post_pull_gain_threshold": (
                self.acquisition_relative_gain_penalty
            ),
            "post_pull_gain_measure": (
                "fixed normalized point-grid support weighted by maximum "
                "sample count"
            ),
            "gain_grid_resolution": self._gain_grid_resolution,
            "gain_grid_points": len(self._gain_grid_points),
            "gain_grid_layout": self._gain_grid_layout,
            "expert_grid_profile_evaluations": (
                self._grid_profile_evaluations
            ),
            "bank_grid_profile_evaluations": (
                self._grid_bank_profile_evaluations
            ),
            "grid_redundancy_evaluations": (
                self._grid_redundancy_evaluations
            ),
            "gain_upper_bound_rejections": (
                self._gain_upper_bound_rejections
            ),
            "sampled_gain_validation": True,
            "gain_validation_points_are_map_independent": True,
            "deployment_positions_used_for_gain_validation": False,
        }
        status["bank_capacity"] = None
        status["post_pull_validation"] = (
            f"{self._gain_grid_layout}-grid gain and grid-profile "
            "redundancy only"
        )
        atomic_json(status_path, status)
        metrics_path = output / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["expert_bank"].update(
            {
                "selection": (
                    "frozen aligned-grid scalar-gain acquisition"
                ),
                "internal_expanded_advertisement_dim": (
                    self._advertisement_latent_dim
                ),
                "advertisement": (
                    "product-quantized aligned patch codes on first sight; "
                    "model key and experience thereafter"
                    if self._quantized_patch_advertisements
                    else "latent encoding on first sight; model key and experience thereafter"
                ),
                "advertisement_cached_per_receiver": True,
                "patch_codebook_groups": getattr(
                    self, "_patch_codebook_groups", None
                ),
                "full_support_shared_only_after_pull": True,
                "post_pull_exact_validation": True,
                "deterministic_model_id_prefilter": True,
                "iterative_pull_until_no_positive_candidate": True,
                "post_pull_grid_validation": "after every pulled model",
                "capacity_including_local": None,
                "pruning": (
                    "remove an expert only when its removal leaves the "
                    "pointwise grid maximum unchanged"
                ),
                "geometric_dominance": False,
                "deployment_probe_set": False,
                "unique_coverage_threshold": None,
                "post_pull_gain_threshold": (
                    "same kappa as predicted acquisition gain"
                ),
                "post_pull_gain_measure": (
                    "fixed normalized point-grid support weighted by maximum "
                    "sample count"
                ),
                "gain_grid_resolution": self._gain_grid_resolution,
                "gain_grid_points": len(self._gain_grid_points),
                "gain_grid_layout": self._gain_grid_layout,
                "sampled_gain_validation": True,
                "gain_validation_points_are_map_independent": True,
                "gain_validation_uses_deployment_positions": False,
            }
        )
        for obsolete in (
            "selection_probe_count",
            "minimum_unique_coverage",
            "minimum_unique_probes",
            "coverage_cache",
            "coverage_recalculation",
        ):
            metrics["expert_bank"].pop(obsolete, None)
        metrics["learned_acquisition"] = {
            "bundle": str(self.acquisition_bundle),
            "pretraining_step": self._acquisition_checkpoint_step,
            "variant": self._acquisition_variant,
            "relative_gain_penalty": self.acquisition_relative_gain_penalty,
            "probability_threshold": (
                None
                if self._acquisition_variant in {
                    "union_scalar_relative_gain",
                    "unit_square_encoding_only_relative_gain",
                    "spatial_grid_encoding_relative_gain",
                }
                else self.acquisition_probability_threshold
            ),
            "pull_attempts": self._learned_pull_attempts,
            "accepted_pulls": self._learned_pull_accepts,
            "rejected_pulls": self._learned_pull_rejections,
            "post_pull_gain_rejections": self._post_pull_gain_rejections,
            "teacher_batches_per_step": self.teacher_distillation_batches_per_step,
            "teacher_support_merges": self._teacher_support_merges,
            "teacher_distillation_passes": self._teacher_distillation_passes,
            "teacher_distillation_samples": self._teacher_distillation_samples,
            "expert_grid_profile_evaluations": (
                self._grid_profile_evaluations
            ),
            "bank_grid_profile_evaluations": (
                self._grid_bank_profile_evaluations
            ),
            "grid_redundancy_evaluations": (
                self._grid_redundancy_evaluations
            ),
            "gain_upper_bound_rejections": (
                self._gain_upper_bound_rejections
            ),
            "advertisement_scalar_values": self._advertisement_scalar_values,
            "advertisement_bytes": 4 * self._advertisement_scalar_values,
            "pulled_support_scalar_values": self._pulled_support_scalar_values,
            "pulled_model_parameter_values": (
                self._pulled_model_parameter_values
            ),
        }
        atomic_json(metrics_path, metrics)


def self_test() -> None:
    overlapping_plane_self_test()
    grid_gain_self_test()
    params = CapsuleParams()
    large = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[0.0, 0.0], [40.0, 0.0]]),
            half_width=4.0,
            mass=5.0,
        )
    ])
    inner = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[5.0, 1.0], [25.0, 1.0]]),
            half_width=1.0,
            mass=3.0,
        )
    ])
    outside = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[5.0, 6.0], [25.0, 6.0]]),
            half_width=1.0,
            mass=3.0,
        )
    ])
    stronger_inner = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[5.0, 1.0], [25.0, 1.0]]),
            half_width=1.0,
            mass=6.0,
        )
    ])
    singleton = serialize_capsules([
        Capsule.from_segment(np.asarray([[5.0, 0.0], [25.0, 0.0]]))
    ])
    large_geometry = plane_dominance_geometry(large)
    assert geometrically_dominated_plane_mask(
        plane_dominance_geometry(inner), large_geometry
    ) == 1
    assert geometrically_dominated_plane_mask(
        plane_dominance_geometry(outside), large_geometry
    ) == 0
    assert geometrically_dominated_plane_mask(
        plane_dominance_geometry(stronger_inner), large_geometry
    ) == 0
    assert geometrically_dominated_plane_mask(
        plane_dominance_geometry(singleton), large_geometry
    ) == 1

    left_half = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[0.0, 0.0], [25.0, 0.0]]),
            half_width=4.0,
            mass=5.0,
        )
    ])
    right_half = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[15.0, 0.0], [40.0, 0.0]]),
            half_width=4.0,
            mass=5.0,
        )
    ])
    assert (
        geometrically_dominated_plane_mask(
            large_geometry, plane_dominance_geometry(left_half)
        )
        | geometrically_dominated_plane_mask(
            large_geometry, plane_dominance_geometry(right_half)
        )
    ) == 0

    weighted_left = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[0.0, 10.0], [10.0, 10.0]]),
            half_width=1.0,
            mass=2.0,
        )
    ])
    weighted_right = serialize_capsules([
        Capsule.from_segment(
            np.asarray([[5.0, 10.0], [15.0, 10.0]]),
            half_width=1.0,
            mass=4.0,
        )
    ])
    left_groups = plane_set_weighted_geometry(
        weighted_left, map_size_m=100.0
    )
    right_groups = plane_set_weighted_geometry(
        weighted_right, map_size_m=100.0
    )
    left_envelope = weighted_support_envelope([left_groups])
    right_envelope = weighted_support_envelope([right_groups])
    combined_envelope = merge_weighted_support_envelopes(
        left_envelope, right_envelope
    )
    rebuilt_envelope = weighted_support_envelope([
        left_groups, right_groups
    ])
    assert math.isclose(left_envelope.score, 0.004, rel_tol=1.0e-9)
    assert math.isclose(right_envelope.score, 0.008, rel_tol=1.0e-9)
    assert math.isclose(combined_envelope.score, 0.010, rel_tol=1.0e-9)
    assert math.isclose(
        combined_envelope.score,
        rebuilt_envelope.score,
        rel_tol=1.0e-12,
    )
    relative_gain, absolute_gain = weighted_geometric_gain(
        left_envelope, right_envelope
    )
    assert math.isclose(relative_gain, 1.5, rel_tol=1.0e-9)
    assert math.isclose(absolute_gain, 0.006, rel_tol=1.0e-9)
    deferred_envelope = WeightedSupportEnvelope(
        (*left_envelope.regions, *right_envelope.regions),
        rebuilt_envelope.score,
    )
    probe_envelope = weighted_support_envelope([
        plane_set_weighted_geometry(
            serialize_capsules([
                Capsule.from_segment(
                    np.asarray([[7.0, 10.0], [17.0, 10.0]]),
                    half_width=1.0,
                    mass=3.0,
                )
            ]),
            map_size_m=100.0,
        )
    ])
    deferred_gain = weighted_geometric_gain(
        deferred_envelope, probe_envelope
    )
    compact_gain = weighted_geometric_gain(
        rebuilt_envelope, probe_envelope
    )
    assert math.isclose(
        deferred_gain[0], compact_gain[0], rel_tol=1.0e-12
    )
    assert math.isclose(
        deferred_gain[1], compact_gain[1], rel_tol=1.0e-12
    )
    assert weighted_support_envelope([
        plane_set_weighted_geometry(singleton, map_size_m=100.0)
    ]).score == 0.0

    first = serialize_capsules([
        Capsule.from_segment(np.asarray([[0.0, 0.0], [20.0, 0.0]]))
    ])
    other = serialize_capsules([
        Capsule.from_segment(np.asarray([[0.0, 10.0], [20.0, 10.0]]))
    ])
    probes = np.asarray([
        [[2.0, 0.0], [18.0, 0.0]],
        [[2.0, 10.0], [18.0, 10.0]],
        [[2.0, 30.0], [18.0, 30.0]],
    ])
    empty = np.zeros(len(probes), dtype=np.float64)
    first_profile = support_profile(first, probes, params)
    other_profile = support_profile(other, probes, params)
    assert first_profile[0] > 0.0 and first_profile[1] == 0.0
    assert other_profile[1] > 0.0 and other_profile[0] == 0.0
    assert marginal_support_gain(first_profile, empty) > 0.0
    assert marginal_support_gain(first_profile, first_profile) == 0.0
    assert marginal_support_gain(other_profile, first_profile) > 0.0
    merged, contributors = remote_union_with_sources(
        [("first", first), ("other", other)], params
    )
    assert len(merged) == len(contributors) == 1
    assert contributors[0] == frozenset(("first", "other"))
    middle = np.asarray([[[2.0, 5.0], [18.0, 5.0]]])
    assert support_profile(merged, middle, params)[0] > 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--transfer-cost", type=float, default=0.0)
    parser.add_argument(
        "--cell-grid-weighted-acquisition-fixed-budget",
        action="store_true",
    )
    parser.add_argument("--probe-count", type=int, default=512)
    parser.add_argument("--dominance-pruned", action="store_true")
    parser.add_argument("--min-unique-coverage", type=float, default=0.005)
    parser.add_argument("--learned-acquisition-bundle", type=Path, default=None)
    parser.add_argument("--cell-grid-support", action="store_true")
    parser.add_argument("--cell-grid-weighted-single", action="store_true")
    parser.add_argument("--cell-grid-weighted-acquisition", action="store_true")
    parser.add_argument(
        "--weighted-selection",
        choices=("experience", "grid-intensity"),
        default="experience",
    )
    parser.add_argument(
        "--weighted-pulls-per-receiver-step",
        type=int,
        default=1,
        help="weighted-single pull budget; zero means every available neighbour",
    )
    parser.add_argument(
        "--weighted-pull-interval-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--weighted-pull-schedule-anchor",
        choices=("entry", "global"),
        default="entry",
        help=(
            "anchor periodic pulls to each vehicle's entry or to global "
            "simulation steps"
        ),
    )
    parser.add_argument(
        "--cell-grid-confidence",
        choices=("binary", "path-ratio", "global-ratio"),
        default="binary",
    )
    parser.add_argument(
        "--cell-grid-min-intensity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acquisition-probability-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--acquisition-relative-gain-penalty", type=float, default=0.0
    )
    parser.add_argument(
        "--bank-support-routing",
        choices=(
            "individual",
            "primary-only",
            "primary-bank-gate",
            "most-experienced",
            "experience-ensemble",
        ),
        default="individual",
    )
    parser.add_argument(
        "--teacher-distillation-batches-per-step", type=int, default=0
    )
    parser.add_argument("--local-lr", type=float, default=5.0e-4)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=0)
    parser.add_argument("--new-data-epochs", type=int, default=2)
    parser.add_argument("--replay-batches", type=int, default=8)
    parser.add_argument("--recent-replay-batches", type=int, default=4)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--full-dataset-epochs", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--tail-eval-count", type=int, default=10)
    parser.add_argument("--tail-eval-stride", type=int, default=25)
    parser.add_argument("--reception-floor-dbm", type=float, default=-100.0)
    parser.add_argument("--method-tag", type=str, default="")
    parser.add_argument(
        "--baseline-mode",
        choices=("expert-bank", "local-only", "central"),
        default="expert-bank",
    )
    parser.add_argument("--angle-deg", type=float, default=7.0)
    parser.add_argument("--lateral-merge-m", type=float, default=1.0)
    parser.add_argument("--longitudinal-gap-m", type=float, default=3.0)
    parser.add_argument("--initial-half-width-m", type=float, default=1.75)
    parser.add_argument("--mass-scale", type=float, default=3.0)
    parser.add_argument("--max-envelope-inflation", type=float, default=1.2)
    parser.add_argument("--max-corridor-width-m", type=float, default=12.0)
    parser.add_argument("--link-length-margin-m", type=float, default=0.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("support expert bank self-test passed")
        return 0
    metadata = validate_dataset(args.trace.resolve(), args.testset.resolve())
    sim_steps = int(args.sim_steps or metadata["sim_steps"])
    if sim_steps > int(metadata["sim_steps"]):
        raise ValueError("requested steps exceed the trace")
    tail_steps = make_tail_evaluation_steps(
        sim_steps,
        count=int(args.tail_eval_count),
        stride=int(args.tail_eval_stride),
    )
    checkpoint_every = max(1, int(args.checkpoint_every))
    results_dir = args.results_dir.resolve()
    resume = args.resume
    automatic = results_dir / "checkpoint_latest.pt"
    if resume is None and args.resume_if_exists and automatic.exists():
        resume = automatic
    capsule_params = CapsuleParams(
        angle_deg=float(args.angle_deg),
        lateral_merge_m=float(args.lateral_merge_m),
        longitudinal_gap_m=float(args.longitudinal_gap_m),
        initial_half_width_m=float(args.initial_half_width_m),
        mass_scale=float(args.mass_scale),
        max_envelope_inflation=float(args.max_envelope_inflation),
        max_corridor_width_m=float(args.max_corridor_width_m),
        link_length_margin_m=float(args.link_length_margin_m),
    )
    gate_params = GateParams()
    training_params = TrainingParams(
        replay_capacity=int(args.replay_capacity),
        new_data_epochs=int(args.new_data_epochs),
        replay_batches=int(args.replay_batches),
        recent_replay_batches=int(args.recent_replay_batches),
        recent_window=int(args.recent_window),
        full_dataset_epochs=int(args.full_dataset_epochs),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )
    reception_floor_dbm = float(args.reception_floor_dbm)
    cfg = build_config_from_env(
        seed=int(args.seed),
        num_nodes=int(metadata["num_nodes"]),
        num_zones=int(metadata["num_zones"]),
        sim_steps=sim_steps,
        map_size=float(metadata["map_size"]),
        active_modes=(),
        results_dir=str(results_dir),
        tx_power_dbm=float(metadata["tx_power_dbm"]),
        rssi_min_dbm=reception_floor_dbm,
        rssi_max_dbm=float(metadata["rssi_max_dbm"]),
        noise_floor_dbm=reception_floor_dbm,
        snr_min_db=0.0,
        model_transfer_snr_min_db=0.0,
        rssi_model="tiny",
        predictor_prior="none",
        predictor_include_time=False,
        local_lr=float(args.local_lr),
        local_batch_size=int(args.local_batch_size),
        local_epochs=1,
        # Required by the shared configuration schema; this experiment never
        # invokes model-parameter merging.
        merge_strategy="average",
        fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_eval_every=checkpoint_every,
        final_fidelity_grid_per_zone=int(metadata["test_count"]),
        fidelity_final_steps=tail_steps,
        fidelity_log_every=0,
        verbose=not bool(args.quiet),
        spike_recovery_enabled=False,
    )
    learned_acquisition = args.learned_acquisition_bundle is not None
    baseline_mode = str(args.baseline_mode)
    if (
        args.cell_grid_weighted_acquisition
        and not args.cell_grid_weighted_single
    ):
        raise ValueError("weighted acquisition requires weighted single-model mode")
    if (
        args.cell_grid_weighted_acquisition_fixed_budget
        and not args.cell_grid_weighted_acquisition
    ):
        raise ValueError(
            "fixed-budget acquisition requires weighted acquisition mode"
        )
    if args.cell_grid_weighted_single and not args.cell_grid_support:
        raise ValueError("weighted single-model mode requires --cell-grid-support")
    if (
        args.cell_grid_support
        and not learned_acquisition
        and baseline_mode not in {"local-only", "central"}
    ):
        raise ValueError("cell-grid support requires a cell-pretrained acquisition bundle")
    if args.cell_grid_support and baseline_mode not in {
        "expert-bank", "local-only", "central"
    }:
        raise ValueError("unknown cell-grid baseline mode")
    if baseline_mode != "expert-bank" and (
        learned_acquisition or args.dominance_pruned
    ):
        raise ValueError(
            "local-only and central baselines cannot use acquisition "
            "or dominance pruning"
        )
    if baseline_mode == "local-only":
        if args.cell_grid_support:
            from experiments.place_wallis_benchmark.cell_grid_methods import (
                CellGridLocalOnlySimulation,
            )

            simulation_class = CellGridLocalOnlySimulation
        else:
            simulation_class = LocalOnlySupportSimulation
    elif baseline_mode == "central":
        if args.cell_grid_support:
            from experiments.place_wallis_benchmark.cell_grid_methods import (
                CellGridCentralSimulation,
            )

            simulation_class = CellGridCentralSimulation
        else:
            simulation_class = CentralSupportSimulation
    elif args.cell_grid_weighted_single:
        from experiments.place_wallis_benchmark.cell_grid_methods import (
            CellGridWeightedSingleSimulation,
        )
        simulation_class = CellGridWeightedSingleSimulation
    elif args.cell_grid_support:
        from experiments.place_wallis_benchmark.cell_grid_methods import (
            CellGridExpertBankSimulation,
        )
        simulation_class = CellGridExpertBankSimulation
    elif learned_acquisition:
        simulation_class = LearnedAcquisitionExpertBankSimulation
    elif args.dominance_pruned:
        simulation_class = DominancePrunedExpertBankSimulation
    else:
        simulation_class = SupportExpertBankSimulation
    method_kwargs: dict[str, Any] = {}
    if args.dominance_pruned or learned_acquisition:
        method_kwargs["min_unique_coverage"] = float(
            args.min_unique_coverage
        )
    if learned_acquisition:
        method_kwargs["acquisition_bundle"] = (
            args.learned_acquisition_bundle.resolve()
        )
        method_kwargs["acquisition_probability_threshold"] = float(
            args.acquisition_probability_threshold
        )
        method_kwargs["acquisition_relative_gain_penalty"] = float(
            args.acquisition_relative_gain_penalty
        )
        method_kwargs["bank_support_routing"] = str(
            args.bank_support_routing
        )
        method_kwargs["teacher_distillation_batches_per_step"] = int(
            args.teacher_distillation_batches_per_step
        )
    if args.cell_grid_support:
        method_kwargs["cell_confidence_mode"] = str(
            args.cell_grid_confidence
        )
        method_kwargs["cell_minimum_intensity"] = float(
            args.cell_grid_min_intensity
        )
    if args.cell_grid_weighted_single:
        method_kwargs["weighted_pulls_per_receiver_step"] = int(
            args.weighted_pulls_per_receiver_step
        )
        method_kwargs["weighted_pull_interval_steps"] = int(
            args.weighted_pull_interval_steps
        )
        method_kwargs["weighted_pull_schedule_anchor"] = str(
            args.weighted_pull_schedule_anchor
        )
        method_kwargs["weighted_acquisition"] = bool(
            args.cell_grid_weighted_acquisition
        )
        method_kwargs["weighted_selection"] = str(args.weighted_selection)
        method_kwargs["weighted_acquisition_fixed_budget"] = bool(
            args.cell_grid_weighted_acquisition_fixed_budget
        )
    simulation = simulation_class(
        cfg,
        sumo_config=str(args.net.resolve()),
        sumo_net=str(args.net.resolve()),
        measurement_trace_in=str(args.trace.resolve()),
        testset=args.testset.resolve(),
        reception_floor_dbm=reception_floor_dbm,
        capsule_params=capsule_params,
        gate_params=gate_params,
        binary_support=True,
        training_params=training_params,
        tail_evaluation_steps=tail_steps,
        bank_capacity=(
            1 if baseline_mode != "expert-bank" else int(args.bank_capacity)
        ),
        transfer_cost=float(args.transfer_cost),
        probe_count=int(args.probe_count),
        method_tag=str(args.method_tag),
        resume=resume,
        progress_every=int(args.progress_every),
        log_rmse_every=0,
        flush_every=checkpoint_every,
        random_od_routing=False,
        local_policy_share=False,
        **method_kwargs,
    )
    simulation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
