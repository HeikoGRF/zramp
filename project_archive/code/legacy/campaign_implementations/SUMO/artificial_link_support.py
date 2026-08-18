"""Mapless continuous-4D artificial floor-link support for online vehicles."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class VehicleEvidence:
    """Only coordinates and successful links locally observed by one vehicle."""

    # A non-positive capacity means unbounded. Persistent route experts must
    # not forget old feasible support merely because they completed more loops.
    coordinate_capacity: int = 0
    link_capacity: int = 0
    coordinates: deque[list[float]] = field(init=False)
    feasible_pairs: deque[list[float]] = field(init=False)
    feasible_rssi_dbm: deque[float] = field(init=False)
    coordinate_keys: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        coordinate_limit = int(self.coordinate_capacity)
        link_limit = int(self.link_capacity)
        self.coordinates = deque(
            maxlen=None if coordinate_limit <= 0 else coordinate_limit
        )
        self.feasible_pairs = deque(
            maxlen=None if link_limit <= 0 else link_limit
        )
        self.feasible_rssi_dbm = deque(
            maxlen=None if link_limit <= 0 else link_limit
        )

    def observe_coordinate(self, point: np.ndarray) -> None:
        """Remember a locally known endpoint without inventing a new link."""

        value = np.asarray(point, dtype=np.float32).reshape(2)
        key = tuple(np.rint(value / 0.5).astype(np.int64).tolist())
        if key in self.coordinate_keys:
            return
        if (
            self.coordinates.maxlen is not None
            and len(self.coordinates) == self.coordinates.maxlen
        ):
            expired = np.asarray(self.coordinates[0], dtype=np.float32)
            expired_key = tuple(
                np.rint(expired / 0.5).astype(np.int64).tolist()
            )
            self.coordinate_keys.discard(expired_key)
        self.coordinates.append(value.tolist())
        self.coordinate_keys.add(key)

    def observe(
        self,
        raw_pair: np.ndarray,
        *,
        rssi_dbm: float | None = None,
    ) -> None:
        raw = np.asarray(raw_pair, dtype=np.float32).reshape(4)
        self.feasible_pairs.append(raw.tolist())
        self.feasible_rssi_dbm.append(
            float("nan") if rssi_dbm is None else float(rssi_dbm)
        )
        for point in (raw[:2], raw[2:]):
            self.observe_coordinate(point)


def temporal_artificial_key(
    raw_pair: np.ndarray,
    step: int,
    *,
    spatial_resolution_m: float = 0.5,
) -> tuple[int, int, int, int, int]:
    """Key a provisional link by local geometry and original raw step."""

    resolution = float(spatial_resolution_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("spatial resolution must be finite and positive")
    spatial = tuple(
        np.rint(np.asarray(raw_pair, dtype=np.float32).reshape(4) / resolution)
        .astype(np.int64)
        .tolist()
    )
    return (*spatial, int(step))


def temporal_contradiction_keep_mask(
    artificial_pairs: np.ndarray,
    artificial_steps: np.ndarray,
    real_pairs: np.ndarray,
    *,
    step: int,
    minimum_distance_m: float,
) -> np.ndarray:
    """Mark artificial rows contradicted by real links at the same time only."""

    artificial = np.asarray(artificial_pairs, dtype=np.float32).reshape(-1, 4)
    steps = np.asarray(artificial_steps, dtype=np.int64).reshape(-1)
    real = np.asarray(real_pairs, dtype=np.float32).reshape(-1, 4)
    if len(artificial) != len(steps):
        raise ValueError("artificial pairs and timestamps must align")
    keep = np.ones((len(artificial),), dtype=np.bool_)
    same_time = steps == int(step)
    if not len(real) or not np.any(same_time):
        return keep
    support = np.concatenate((real, real[:, [2, 3, 0, 1]]), axis=0)
    current = np.flatnonzero(same_time)
    distances, _ = cKDTree(support).query(
        artificial[current], k=1, workers=1
    )
    normalized = np.asarray(distances, dtype=np.float32) / math.sqrt(2.0)
    keep[current] = normalized > float(minimum_distance_m)
    return keep


@dataclass(frozen=True)
class CalibratedSupportCandidate:
    """A provisional unavailable-link constraint, never a true negative."""

    raw_pair: np.ndarray
    training_weight: float
    support_distance_m: float
    calibrated_minimum_m: float
    conformal_p_value: float
    optimistic_rssi_dbm: float | None
    high_confidence: bool


def artificial_split(raw_pair: np.ndarray, seed: int) -> str:
    quantized = np.rint(np.asarray(raw_pair) * 10.0).astype(np.int64)
    code = int(
        np.sum(quantized * np.asarray([73_856_093, 19_349_663, 83_492_791, 2_654_435_761]))
        + int(seed) * 97_531
    ) % 10
    if code == 0:
        return "optimize"
    if code == 1:
        return "reward"
    return "train"


def feature_for_pair(
    raw_pair: np.ndarray,
    *,
    tile_id: int,
    bounds_m: tuple[float, float, float, float],
    tile_size_m: float,
    tiles_per_side: int,
) -> np.ndarray:
    sx, sy, rx, ry = map(float, np.asarray(raw_pair).reshape(4))
    x0, x1, y0, y1 = map(float, bounds_m)
    column = int(tile_id) % int(tiles_per_side)
    row = int(tile_id) // int(tiles_per_side)
    tile_x0 = x0 + column * float(tile_size_m)
    tile_y0 = y0 + row * float(tile_size_m)
    return np.clip(
        np.asarray(
            [
                (sx - x0) / (x1 - x0),
                (sy - y0) / (y1 - y0),
                (rx - tile_x0) / float(tile_size_m),
                (ry - tile_y0) / float(tile_size_m),
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )


def support_distance_candidates(
    evidence: VehicleEvidence,
    *,
    receiver_xy: np.ndarray,
    number: int,
    rng: np.random.Generator,
    candidate_pool: int,
    minimum_distance_m: float,
    high_distance_m: float,
    low_weight: float,
    high_weight: float,
) -> list[tuple[np.ndarray, float, float]]:
    """Return safest unseen sender/receiver combinations, highest distance first."""

    if number <= 0 or not evidence.coordinates or not evidence.feasible_pairs:
        return []
    points = np.asarray(evidence.coordinates, dtype=np.float32)
    if int(candidate_pool) > 0 and len(points) > int(candidate_pool):
        points = points[
            rng.choice(len(points), int(candidate_pool), replace=False)
        ]
    receiver = np.asarray(receiver_xy, dtype=np.float32).reshape(2)
    physical_distance = np.linalg.norm(points - receiver[None, :], axis=1)
    candidates = np.concatenate(
        (points, np.repeat(receiver[None, :], len(points), axis=0)), axis=1
    )
    feasible = np.asarray(evidence.feasible_pairs, dtype=np.float32)
    support = np.concatenate((feasible, feasible[:, [2, 3, 0, 1]]), axis=0)
    best, _ = cKDTree(support).query(candidates, k=1, workers=1)
    best = np.asarray(best, dtype=np.float32) / math.sqrt(2.0)
    valid = (physical_distance >= 1.0) & (best >= float(minimum_distance_m))
    order = np.flatnonzero(valid)
    order = order[np.argsort(-best[order])]
    result: list[tuple[np.ndarray, float, float]] = []
    for index in order[: int(number)]:
        distance = float(best[index])
        weight = (
            float(high_weight)
            if distance >= float(high_distance_m)
            else float(low_weight)
        )
        result.append((candidates[index].copy(), weight, distance))
    return result


def _deduplicated_oriented_support(evidence: VehicleEvidence) -> np.ndarray:
    feasible = np.asarray(evidence.feasible_pairs, dtype=np.float32).reshape(-1, 4)
    oriented = np.concatenate((feasible, feasible[:, [2, 3, 0, 1]]), axis=0)
    keys = np.rint(oriented / 0.5).astype(np.int64)
    _unique, indices = np.unique(keys, axis=0, return_index=True)
    return oriented[np.sort(indices)]


def _angular_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def calibrated_support_candidates(
    evidence: VehicleEvidence,
    *,
    receiver_xy: np.ndarray,
    number: int,
    rng: np.random.Generator,
    candidate_pool: int = 2048,
    minimum_distance_m: float = 7.5,
    conformal_alpha: float = 0.05,
    minimum_unique_links: int = 64,
    minimum_physical_distance_m: float = 15.0,
    feasible_threshold_dbm: float = -100.0,
    pathloss_exponent: float = 2.0,
    anchor_receiver_radius_m: float = 5.0,
    anchor_bearing_degrees: float = 15.0,
    minimum_range_ratio: float = 1.20,
    minimum_anchor_count: int = 2,
    rssi_margin_db: float = 3.0,
    low_weight: float = 0.025,
    high_weight: float = 0.10,
    require_range_frontier: bool = False,
    diversity_distance_m: float = 2.0,
) -> list[CalibratedSupportCandidate]:
    """Generate calibrated, provisional unavailable-link constraints.

    A fixed support radius is unsafe while the successful-link support is still
    sparse.  This routine first calibrates the radius against leave-one-out
    nearest-neighbour distances among the vehicle's own successful links.  A
    candidate is eligible only when it is more unusual than the configured
    upper tail of those known successes.  This is a one-class/conformal-style
    guard, not proof that the candidate is unavailable.

    High-confidence rows pass a second, independent physics gate: successful
    links observed from nearby receiver positions and in the same bearing
    sector are extrapolated with an optimistic free-space path-loss exponent.
    The candidate is high-confidence only if even the strongest extrapolation
    falls below the reception threshold by ``rssi_margin_db``.  Lower-tier
    outliers remain weak constraints unless ``require_range_frontier`` is true.
    No unavailable observation, map geometry, or future sample is consulted.
    """

    if number <= 0 or not evidence.coordinates or not evidence.feasible_pairs:
        return []
    if not 0.0 < float(conformal_alpha) < 1.0:
        raise ValueError("conformal_alpha must lie strictly between zero and one")
    if float(minimum_distance_m) <= 0.0:
        raise ValueError("minimum_distance_m must be positive")
    if float(minimum_physical_distance_m) <= 0.0:
        raise ValueError("minimum_physical_distance_m must be positive")

    support = _deduplicated_oriented_support(evidence)
    if support.shape[0] < max(3, int(minimum_unique_links)):
        return []
    support_tree = cKDTree(support)
    leave_one_out, _ = support_tree.query(support, k=2, workers=1)
    calibration = np.asarray(leave_one_out[:, 1], dtype=np.float64)
    calibration /= math.sqrt(2.0)
    ordered_calibration = np.sort(calibration)
    rank = int(
        math.ceil((ordered_calibration.size + 1) * (1.0 - conformal_alpha))
        - 1
    )
    calibrated_minimum = max(
        float(minimum_distance_m),
        float(ordered_calibration[np.clip(rank, 0, ordered_calibration.size - 1)]),
    )

    points = np.asarray(evidence.coordinates, dtype=np.float32).reshape(-1, 2)
    if int(candidate_pool) > 0 and len(points) > int(candidate_pool):
        points = points[
            rng.choice(len(points), int(candidate_pool), replace=False)
        ]
    receiver = np.asarray(receiver_xy, dtype=np.float32).reshape(2)
    physical_distance = np.linalg.norm(points - receiver[None, :], axis=1)
    candidates = np.concatenate(
        (points, np.repeat(receiver[None, :], len(points), axis=0)), axis=1
    )
    support_distance, _ = support_tree.query(candidates, k=1, workers=1)
    support_distance = np.asarray(support_distance, dtype=np.float64)
    support_distance /= math.sqrt(2.0)
    positions = np.searchsorted(
        ordered_calibration, support_distance, side="left"
    )
    tail_count = ordered_calibration.size - positions
    p_value = (1.0 + tail_count) / (ordered_calibration.size + 1.0)
    eligible = (
        (physical_distance >= float(minimum_physical_distance_m))
        & (support_distance >= calibrated_minimum)
        & (p_value <= float(conformal_alpha))
    )

    raw_support = np.asarray(
        evidence.feasible_pairs, dtype=np.float32
    ).reshape(-1, 4)
    raw_rssi = np.asarray(evidence.feasible_rssi_dbm, dtype=np.float64)
    anchors = np.concatenate(
        (raw_support, raw_support[:, [2, 3, 0, 1]]), axis=0
    )
    anchor_rssi = np.concatenate((raw_rssi, raw_rssi))
    anchor_vector = anchors[:, :2] - anchors[:, 2:]
    anchor_distance = np.linalg.norm(anchor_vector, axis=1)
    anchor_bearing = np.arctan2(anchor_vector[:, 1], anchor_vector[:, 0])
    receiver_close = (
        np.linalg.norm(anchors[:, 2:] - receiver[None, :], axis=1)
        <= float(anchor_receiver_radius_m)
    )
    finite_anchor = (
        np.isfinite(anchor_rssi)
        & receiver_close
        & (anchor_distance >= 1.0)
    )
    candidate_vector = points - receiver[None, :]
    candidate_bearing = np.arctan2(
        candidate_vector[:, 1], candidate_vector[:, 0]
    )
    bearing_limit = math.radians(float(anchor_bearing_degrees))

    scored: list[tuple[tuple[float, ...], CalibratedSupportCandidate]] = []
    for index in np.flatnonzero(eligible):
        distance = float(physical_distance[index])
        ratio = distance / np.maximum(anchor_distance, 1.0e-9)
        aligned = (
            finite_anchor
            & (ratio >= float(minimum_range_ratio))
            & (
                _angular_difference(candidate_bearing[index], anchor_bearing)
                <= bearing_limit
            )
        )
        optimistic: float | None = None
        if int(np.count_nonzero(aligned)) >= int(minimum_anchor_count):
            extrapolated = anchor_rssi[aligned] - (
                10.0
                * float(pathloss_exponent)
                * np.log10(ratio[aligned])
            )
            optimistic = float(np.max(extrapolated))
        high = bool(
            optimistic is not None
            and optimistic
            <= float(feasible_threshold_dbm) - float(rssi_margin_db)
        )
        if bool(require_range_frontier) and not high:
            continue
        weight = float(high_weight if high else low_weight)
        raw = candidates[index].astype(np.float32, copy=True)
        raw.setflags(write=False)
        item = CalibratedSupportCandidate(
            raw_pair=raw,
            training_weight=weight,
            support_distance_m=float(support_distance[index]),
            calibrated_minimum_m=calibrated_minimum,
            conformal_p_value=float(p_value[index]),
            optimistic_rssi_dbm=optimistic,
            high_confidence=high,
        )
        margin = (
            float(feasible_threshold_dbm) - float(rssi_margin_db) - optimistic
            if optimistic is not None
            else -1.0e6
        )
        score = (
            float(high),
            margin,
            -float(p_value[index]),
            float(support_distance[index]),
        )
        scored.append((score, item))

    scored.sort(key=lambda value: value[0], reverse=True)
    result: list[CalibratedSupportCandidate] = []
    for _score, item in scored:
        if result and float(diversity_distance_m) > 0.0:
            selected = np.stack([entry.raw_pair for entry in result])
            nearest = float(
                np.min(np.linalg.norm(selected - item.raw_pair[None, :], axis=1))
                / math.sqrt(2.0)
            )
            if nearest < float(diversity_distance_m):
                continue
        result.append(item)
        if len(result) >= int(number):
            break
    return result
