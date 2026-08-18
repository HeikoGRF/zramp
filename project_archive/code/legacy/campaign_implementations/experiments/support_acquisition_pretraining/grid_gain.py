#!/usr/bin/env python3
"""Deterministic intensity-weighted support on a normalized point grid."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_GRID_RESOLUTION = 150
GRID_LAYOUT_REGULAR = "regular"
GRID_LAYOUT_STAGGERED = "staggered"
GRID_LAYOUTS = (GRID_LAYOUT_REGULAR, GRID_LAYOUT_STAGGERED)


def unit_square_point_grid(
    resolution: int,
    *,
    layout: str = GRID_LAYOUT_STAGGERED,
) -> np.ndarray:
    """Return a row-major regular or staggered unit-square lattice."""

    size = int(resolution)
    if size <= 0:
        raise ValueError("grid resolution must be positive")
    selected_layout = str(layout)
    if selected_layout not in GRID_LAYOUTS:
        raise ValueError(f"unknown grid layout {selected_layout!r}")
    columns = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    if selected_layout == GRID_LAYOUT_REGULAR:
        x = np.broadcast_to(
            (columns[None, :] + 0.5) / size,
            (size, size),
        )
    else:
        x = np.broadcast_to(
            columns[None, :] + 0.5 * (1.0 - (rows[:, None] % 2.0)),
            (size, size),
        ) / size
    y = np.broadcast_to(
        ((rows[:, None] + 0.5) / size),
        (size, size),
    )
    return np.column_stack((x.ravel(), y.ravel()))


def point_grid_support_counts(
    rows: np.ndarray,
    points: np.ndarray,
    *,
    map_size: float,
    plane_chunk_size: int = 32,
) -> np.ndarray:
    """Return the maximum raw plane count supporting each fixed point.

    Positive-area trapezoidal planes support points inside their boundary.
    Zero-width singleton lines deliberately have zero point-area support.
    """

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    profile = np.zeros(len(queries), dtype=np.float32)
    if len(values) == 0 or len(queries) == 0:
        return profile
    scale = float(map_size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_size must be finite and positive")
    chunk_size = max(1, int(plane_chunk_size))
    start = values[:, 0:2] / scale
    end = values[:, 2:4] / scale
    low_start = values[:, 4] / scale
    high_start = values[:, 5] / scale
    low_end = values[:, 6] / scale
    high_end = values[:, 7] / scale
    intensity = np.maximum(values[:, 8], 0.0)
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    positive_area = (
        length > np.finfo(np.float64).eps
    ) & (
        (high_start - low_start) + (high_end - low_end)
        > np.finfo(np.float64).eps
    ) & (intensity > 0.0)
    valid = np.flatnonzero(positive_area)
    for offset in range(0, len(valid), chunk_size):
        indices = valid[offset:offset + chunk_size]
        chunk_start = start[indices]
        chunk_vector = vector[indices]
        chunk_length = length[indices]
        axis = chunk_vector / chunk_length[:, None]
        normal = np.column_stack((-axis[:, 1], axis[:, 0]))
        delta = queries[:, None, :] - chunk_start[None, :, :]
        along = np.einsum("pci,ci->pc", delta, axis, optimize=True)
        fraction = along / chunk_length[None, :]
        lateral = np.einsum("pci,ci->pc", delta, normal, optimize=True)
        low = (
            (1.0 - fraction) * low_start[indices][None, :]
            + fraction * low_end[indices][None, :]
        )
        high = (
            (1.0 - fraction) * high_start[indices][None, :]
            + fraction * high_end[indices][None, :]
        )
        supported = (
            (fraction >= 0.0)
            & (fraction <= 1.0)
            & (lateral >= low)
            & (lateral <= high)
        )
        profile = np.maximum(
            profile,
            np.max(
                np.where(supported, intensity[indices][None, :], 0.0),
                axis=1,
            ).astype(np.float32, copy=False),
        )
    return profile


def grid_support_counts(
    rows: np.ndarray,
    *,
    resolution: int,
    map_size: float,
    layout: str = GRID_LAYOUT_STAGGERED,
) -> np.ndarray:
    """Rasterize exact point support while visiting only plane bounding boxes."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 11)
    size = int(resolution)
    if size <= 0:
        raise ValueError("grid resolution must be positive")
    selected_layout = str(layout)
    if selected_layout not in GRID_LAYOUTS:
        raise ValueError(f"unknown grid layout {selected_layout!r}")
    scale = float(map_size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_size must be finite and positive")
    profile = np.zeros((size, size), dtype=np.float32)
    if len(values) == 0:
        return profile.ravel()
    y_coordinates = (np.arange(size, dtype=np.float64) + 0.5) / size
    centered_x = (np.arange(size, dtype=np.float64) + 0.5) / size
    if selected_layout == GRID_LAYOUT_REGULAR:
        x_coordinates = (centered_x,)
        row_parities = (0,)
    else:
        x_coordinates = (
            centered_x,
            np.arange(size, dtype=np.float64) / size,
        )
        row_parities = (0, 1)
    for row in values:
        start = row[0:2] / scale
        end = row[2:4] / scale
        vector = end - start
        length = float(np.linalg.norm(vector))
        low_start, high_start, low_end, high_end = row[4:8] / scale
        intensity = max(0.0, float(row[8]))
        if (
            length <= np.finfo(np.float64).eps
            or (high_start - low_start) + (high_end - low_end)
            <= np.finfo(np.float64).eps
            or intensity <= 0.0
        ):
            continue
        axis = vector / length
        normal = np.asarray([-axis[1], axis[0]])
        corners = np.stack((
            start + low_start * normal,
            start + high_start * normal,
            end + low_end * normal,
            end + high_end * normal,
        ))
        low_bound = np.maximum(np.min(corners, axis=0), 0.0)
        high_bound = np.minimum(np.max(corners, axis=0), 1.0)
        if bool(np.any(high_bound < low_bound)):
            continue
        row_start = int(np.searchsorted(
            y_coordinates, low_bound[1], side="left"
        ))
        row_stop = int(np.searchsorted(
            y_coordinates, high_bound[1], side="right"
        ))
        for parity in row_parities:
            if selected_layout == GRID_LAYOUT_REGULAR:
                row_indices = np.arange(
                    row_start, row_stop, dtype=np.int64
                )
            else:
                row_indices = np.arange(
                    row_start + ((parity - row_start) % 2),
                    row_stop,
                    2,
                    dtype=np.int64,
                )
            if len(row_indices) == 0:
                continue
            x_values = x_coordinates[parity]
            column_start = int(np.searchsorted(
                x_values, low_bound[0], side="left"
            ))
            column_stop = int(np.searchsorted(
                x_values, high_bound[0], side="right"
            ))
            if column_start >= column_stop:
                continue
            column_indices = np.arange(
                column_start, column_stop, dtype=np.int64
            )
            delta_x = x_values[column_indices][None, :] - start[0]
            delta_y = y_coordinates[row_indices][:, None] - start[1]
            along = delta_x * axis[0] + delta_y * axis[1]
            fraction = along / length
            lateral = delta_x * normal[0] + delta_y * normal[1]
            low = (1.0 - fraction) * low_start + fraction * low_end
            high = (1.0 - fraction) * high_start + fraction * high_end
            supported = (
                (fraction >= 0.0)
                & (fraction <= 1.0)
                & (lateral >= low)
                & (lateral <= high)
            )
            current = profile[np.ix_(row_indices, column_indices)]
            np.maximum(
                current,
                np.where(supported, intensity, 0.0),
                out=current,
            )
            profile[np.ix_(row_indices, column_indices)] = current
    return profile.ravel()


def staggered_grid_support_counts(
    rows: np.ndarray,
    *,
    resolution: int,
    map_size: float,
) -> np.ndarray:
    """Backward-compatible exact rasterization on the staggered lattice."""

    return grid_support_counts(
        rows,
        resolution=resolution,
        map_size=map_size,
        layout=GRID_LAYOUT_STAGGERED,
    )


def regular_grid_support_counts(
    rows: np.ndarray,
    *,
    resolution: int,
    map_size: float,
) -> np.ndarray:
    """Exact rasterization on a regular cell-centred lattice."""

    return grid_support_counts(
        rows,
        resolution=resolution,
        map_size=map_size,
        layout=GRID_LAYOUT_REGULAR,
    )


def relative_point_grid_gain(
    bank: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    """Return exact relative and absolute gain for the discrete grid score."""

    bank_values = np.asarray(bank, dtype=np.float32).reshape(-1)
    candidate_values = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if bank_values.shape != candidate_values.shape:
        raise ValueError("bank and candidate grid profiles must have equal shape")
    increase = float(np.sum(
        np.maximum(candidate_values - bank_values, 0.0),
        dtype=np.float64,
    ))
    bank_score = float(np.sum(bank_values, dtype=np.float64))
    return increase / max(bank_score, 1.0), increase


def self_test() -> None:
    grid = unit_square_point_grid(8, layout=GRID_LAYOUT_STAGGERED)
    regular_grid = unit_square_point_grid(8, layout=GRID_LAYOUT_REGULAR)
    horizontal = np.asarray([[0.0, 0.5, 1.0, 0.5,
                              -0.1, 0.1, -0.1, 0.1,
                              2.0, 1.0, 0.0]])
    stronger = horizontal.copy()
    stronger[:, 8] = 5.0
    line = horizontal.copy()
    line[:, 4:8] = 0.0
    first = point_grid_support_counts(horizontal, grid, map_size=1.0)
    second = point_grid_support_counts(stronger, grid, map_size=1.0)
    zero = point_grid_support_counts(line, grid, map_size=1.0)
    raster = staggered_grid_support_counts(
        horizontal, resolution=8, map_size=1.0
    )
    regular_direct = point_grid_support_counts(
        horizontal, regular_grid, map_size=1.0
    )
    regular_raster = regular_grid_support_counts(
        horizontal, resolution=8, map_size=1.0
    )
    assert np.count_nonzero(first) == 16
    assert np.all(second[first > 0.0] == 5.0)
    assert not np.any(zero)
    assert np.array_equal(first, raster)
    assert np.array_equal(regular_direct, regular_raster)
    relative, absolute = relative_point_grid_gain(first, second)
    assert math.isclose(relative, 1.5)
    assert math.isclose(absolute, 48.0)


__all__ = [
    "DEFAULT_GRID_RESOLUTION",
    "GRID_LAYOUT_REGULAR",
    "GRID_LAYOUT_STAGGERED",
    "GRID_LAYOUTS",
    "grid_support_counts",
    "point_grid_support_counts",
    "regular_grid_support_counts",
    "relative_point_grid_gain",
    "self_test",
    "staggered_grid_support_counts",
    "unit_square_point_grid",
]
