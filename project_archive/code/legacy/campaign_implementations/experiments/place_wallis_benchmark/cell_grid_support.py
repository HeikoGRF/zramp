"""Cell-native support for positive radio links.

Every positive link activates the grid cells traversed by its straight
transmitter--receiver segment.  Queries use the identical traversal and are
supported only when every visited cell is active.
"""

from __future__ import annotations

import math

import numpy as np


CELL_GRID_SUPPORT_SEMANTICS = "normalized_square_cell_traversal_v1"


def _clipped_grid_coordinate(value: float, resolution: int) -> float:
    upper = np.nextafter(float(resolution), 0.0)
    return float(np.clip(float(value) * float(resolution), 0.0, upper))


def segment_cell_indices(
    segment: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Return cells entered by a normalized segment using deterministic DDA.

    Cells are half-open on their upper boundaries.  A segment crossing an
    exact grid corner enters the diagonal cell; the same convention is used
    for both updates and queries.
    """

    size = int(resolution)
    if size <= 0:
        raise ValueError("resolution must be positive")
    points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    x0 = _clipped_grid_coordinate(points[0, 0], size)
    y0 = _clipped_grid_coordinate(points[0, 1], size)
    x1 = _clipped_grid_coordinate(points[1, 0], size)
    y1 = _clipped_grid_coordinate(points[1, 1], size)
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    end_x, end_y = int(math.floor(x1)), int(math.floor(y1))
    cells = [iy * size + ix]
    if ix == end_x and iy == end_y:
        return np.asarray(cells, dtype=np.int32)

    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_y = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    if step_x:
        boundary_x = float(ix + (1 if step_x > 0 else 0))
        t_max_x = (boundary_x - x0) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = t_delta_x = float("inf")
    if step_y:
        boundary_y = float(iy + (1 if step_y > 0 else 0))
        t_max_y = (boundary_y - y0) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = t_delta_y = float("inf")

    maximum_steps = 2 * size + 2
    for _ in range(maximum_steps):
        if ix == end_x and iy == end_y:
            break
        finite_crossings = [
            abs(value) for value in (t_max_x, t_max_y) if math.isfinite(value)
        ]
        tolerance = 8.0 * np.finfo(np.float64).eps * max(
            [1.0, *finite_crossings]
        )
        if t_max_x < t_max_y - tolerance:
            ix += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x - tolerance:
            iy += step_y
            t_max_y += t_delta_y
        else:
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        if 0 <= ix < size and 0 <= iy < size:
            cells.append(iy * size + ix)
    else:
        raise RuntimeError("cell traversal did not terminate")
    if ix != end_x or iy != end_y:
        raise RuntimeError("cell traversal missed its endpoint")
    return np.asarray(cells, dtype=np.int32)


def add_segments_to_grid(
    profile: np.ndarray,
    segments: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> None:
    """Increment an intensity grid for every cell traversed by each segment."""

    grid = np.asarray(profile)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("profile must be a square two-dimensional grid")
    links = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if weights is None:
        increments = np.ones(len(links), dtype=grid.dtype)
    else:
        increments = np.asarray(weights, dtype=grid.dtype).reshape(-1)
        if len(increments) != len(links):
            raise ValueError("weights must match the number of segments")
    flat = grid.reshape(-1)
    for segment, increment in zip(links, increments, strict=True):
        if float(increment) <= 0.0:
            continue
        indices = segment_cell_indices(segment, int(grid.shape[0]))
        flat[indices] += increment


def cell_grid_from_support_rows(
    rows: np.ndarray,
    *,
    resolution: int,
) -> np.ndarray:
    """Rasterize synthetic support rows as cell-crossing link evidence.

    Each row already summarizes repeated positive links through its ``mass``
    field.  Pointwise maximum matches the idempotent expert-bank envelope.
    Width fields are deliberately ignored: cell support comes only from the
    represented transmitter--receiver segment.
    """

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    profile = np.zeros((int(resolution), int(resolution)), dtype=np.float32)
    flat = profile.reshape(-1)
    for row in values:
        intensity = max(0.0, float(row[8]))
        if intensity <= 0.0:
            continue
        indices = segment_cell_indices(row[:4].reshape(2, 2), int(resolution))
        flat[indices] = np.maximum(flat[indices], intensity)
    return profile


def link_support_profile(
    profile: np.ndarray,
    queries: np.ndarray,
    *,
    binary: bool = True,
    minimum_intensity: float = 1.0,
) -> np.ndarray:
    """Return binary support or bottleneck intensity for normalized links."""

    grid = np.asarray(profile)
    if grid.ndim == 1:
        size = int(round(math.sqrt(len(grid))))
        if size * size != len(grid):
            raise ValueError("flat profile length is not a square")
        grid = grid.reshape(size, size)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("profile must be square")
    links = np.asarray(queries, dtype=np.float64).reshape(-1, 2, 2)
    flat = grid.reshape(-1)
    result = np.zeros(len(links), dtype=np.float64)
    for index, segment in enumerate(links):
        values = flat[segment_cell_indices(segment, int(grid.shape[0]))]
        minimum = float(np.min(values)) if len(values) else 0.0
        result[index] = (
            float(minimum >= minimum_intensity) if binary else max(0.0, minimum)
        )
    return result


def link_support_profiles(
    profiles: np.ndarray,
    queries: np.ndarray,
    *,
    binary: bool = True,
    minimum_intensity: float = 1.0,
) -> np.ndarray:
    """Evaluate many expert grids while traversing every query only once."""

    grids = np.asarray(profiles)
    if grids.ndim == 2:
        size = int(round(math.sqrt(grids.shape[1])))
        if size * size != grids.shape[1]:
            raise ValueError("flat profile length is not a square")
        flat = grids.reshape(len(grids), -1)
    elif grids.ndim == 3 and grids.shape[1] == grids.shape[2]:
        size = int(grids.shape[1])
        flat = grids.reshape(len(grids), -1)
    else:
        raise ValueError("profiles must be a stack of equal square grids")
    links = np.asarray(queries, dtype=np.float64).reshape(-1, 2, 2)
    result = np.zeros((len(flat), len(links)), dtype=np.float64)
    for query_index, segment in enumerate(links):
        indices = segment_cell_indices(segment, size)
        minimum = np.min(flat[:, indices], axis=1)
        if binary:
            result[:, query_index] = minimum >= minimum_intensity
        else:
            result[:, query_index] = np.maximum(minimum, 0.0)
    return result


def link_confidence_profiles(
    profiles: np.ndarray,
    queries: np.ndarray,
    *,
    mode: str,
    minimum_intensity: float = 1.0,
) -> np.ndarray:
    """Return parameter-free confidence from traversed cell intensities."""

    grids = np.asarray(profiles)
    if grids.ndim == 2:
        size = int(round(math.sqrt(grids.shape[1])))
        if size * size != grids.shape[1]:
            raise ValueError("flat profile length is not a square")
        flat = grids.reshape(len(grids), -1)
    elif grids.ndim == 3 and grids.shape[1] == grids.shape[2]:
        size = int(grids.shape[1])
        flat = grids.reshape(len(grids), -1)
    else:
        raise ValueError("profiles must be a stack of equal square grids")
    if mode not in {"binary", "path-ratio", "global-ratio"}:
        raise ValueError(f"unknown cell confidence mode {mode!r}")
    links = np.asarray(queries, dtype=np.float64).reshape(-1, 2, 2)
    result = np.zeros((len(flat), len(links)), dtype=np.float64)
    global_maximum = np.max(flat, axis=1) if mode == "global-ratio" else None
    for query_index, segment in enumerate(links):
        indices = segment_cell_indices(segment, size)
        values = flat[:, indices]
        minimum = np.min(values, axis=1)
        if mode == "binary":
            confidence = minimum >= minimum_intensity
        elif mode == "path-ratio":
            maximum = np.max(values, axis=1)
            confidence = np.divide(
                minimum,
                maximum,
                out=np.zeros_like(minimum, dtype=np.float64),
                where=maximum > 0.0,
            )
        else:
            assert global_maximum is not None
            confidence = np.divide(
                minimum,
                global_maximum,
                out=np.zeros_like(minimum, dtype=np.float64),
                where=global_maximum > 0.0,
            )
        if mode != "binary":
            confidence = np.where(
                minimum >= minimum_intensity, confidence, 0.0)
        result[:, query_index] = confidence
    return result


def sparse_grid_payload_bytes(profile: np.ndarray) -> int:
    """Compact exact payload: uint32 index and uint32 count per active cell."""

    active = int(np.count_nonzero(np.asarray(profile) > 0.0))
    sparse = 8 * active + 8
    dense = 4 * int(np.asarray(profile).size) + 8
    return int(min(sparse, dense))


def self_test() -> None:
    grid = np.zeros((10, 10), dtype=np.float32)
    horizontal = np.asarray([[[0.05, 0.25], [0.95, 0.25]]])
    add_segments_to_grid(grid, horizontal)
    assert int(np.count_nonzero(grid)) == 10
    assert float(link_support_profile(grid, horizontal)[0]) == 1.0
    offset = np.asarray([[[0.05, 0.35], [0.95, 0.35]]])
    assert float(link_support_profile(grid, offset)[0]) == 0.0
    diagonal = np.zeros_like(grid)
    add_segments_to_grid(
        diagonal,
        np.asarray([[[0.05, 0.05], [0.95, 0.95]]]),
        weights=np.asarray([3.0]),
    )
    assert int(np.count_nonzero(diagonal)) == 10
    assert float(link_support_profile(
        diagonal,
        np.asarray([[[0.05, 0.05], [0.95, 0.95]]]),
        binary=False,
    )[0]) == 3.0
    broken = diagonal.copy()
    broken[5, 5] = 0.0
    assert float(link_support_profile(
        broken, np.asarray([[[0.05, 0.05], [0.95, 0.95]]])
    )[0]) == 0.0
    stacked = link_support_profiles(
        np.stack((grid, broken)),
        np.concatenate((horizontal, offset)),
    )
    assert np.array_equal(
        stacked,
        np.stack((
            link_support_profile(grid, np.concatenate((horizontal, offset))),
            link_support_profile(broken, np.concatenate((horizontal, offset))),
        )),
    )
    path_confidence = link_confidence_profiles(
        np.stack((grid, diagonal)), horizontal, mode="path-ratio"
    )
    assert float(path_confidence[0, 0]) == 1.0
    global_confidence = link_confidence_profiles(
        np.stack((grid, diagonal)), horizontal, mode="global-ratio"
    )
    assert 0.0 <= float(global_confidence[0, 0]) <= 1.0


if __name__ == "__main__":
    self_test()
    print("cell-grid support self-test passed")
