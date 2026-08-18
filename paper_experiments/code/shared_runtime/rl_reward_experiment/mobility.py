"""
Mobility and geometry helpers.

    Nodes live in a square map `[0, map_size]^2` partitioned into a square grid
    of zones. Walls come from the parent `build_map` module
as `(name, cx, cy, thickness, length, height)` tuples; only the XY footprint
matters here.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


WallTuple = tuple[str, float, float, float, float, float]


# --- Wall collision ---------------------------------------------------------

def collides_with_walls(x: float, y: float, walls: Iterable[WallTuple]) -> bool:
    """True iff (x, y) lies inside any wall's XY footprint (+ small margin)."""
    for _, cx, cy, th, l, _ in walls:
        if (cx - th / 2.0 - 0.1 <= x <= cx + th / 2.0 + 0.1) and (
            cy - l / 2.0 - 0.1 <= y <= cy + l / 2.0 + 0.1
        ):
            return True
    return False


def clamp_xy(x: float, y: float, map_size: float, margin: float) -> tuple[float, float]:
    return (
        float(max(margin, min(map_size - margin, x))),
        float(max(margin, min(map_size - margin, y))),
    )


def snap_out_of_walls(
    x: float,
    y: float,
    walls: Iterable[WallTuple],
    map_size: float,
    margin: float,
) -> tuple[float, float, bool]:
    """Spiral search outward for the closest free spot."""
    if not collides_with_walls(x, y, walls):
        return float(x), float(y), False
    for dist in np.arange(0.5, 20.0, 0.5):
        for angle in np.arange(0.0, 2.0 * math.pi, math.pi / 4.0):
            nx, ny = clamp_xy(
                x + dist * math.cos(angle),
                y + dist * math.sin(angle),
                map_size,
                margin,
            )
            if not collides_with_walls(nx, ny, walls):
                return float(nx), float(ny), True
    return float(x), float(y), False


# --- Zone assignment --------------------------------------------------------

def _zones_per_side(num_zones: int) -> int:
    side = int(math.isqrt(int(num_zones)))
    if side * side != int(num_zones):
        raise ValueError(f"num_zones={num_zones} must be a perfect square")
    return max(1, side)


def zone_of(x: float, y: float, map_size: float, num_zones: int = 4) -> int:
    """Square-grid zone id, row-major from south-west to north-east."""
    side = _zones_per_side(num_zones)
    cell = float(map_size) / float(side)
    col = int(float(x) / max(cell, 1e-9))
    row = int(float(y) / max(cell, 1e-9))
    col = max(0, min(side - 1, col))
    row = max(0, min(side - 1, row))
    return row * side + col


def zone_bounds(
    zone: int,
    map_size: float,
    num_zones: int = 4,
) -> tuple[float, float, float, float]:
    """Return (x_lo, x_hi, y_lo, y_hi) for a square-grid zone."""
    side = _zones_per_side(num_zones)
    z = int(zone)
    if z < 0 or z >= int(num_zones):
        raise ValueError(f"Unknown zone {zone} for num_zones={num_zones}")
    col = z % side
    row = z // side
    cell = float(map_size) / float(side)
    x_lo = float(col) * cell
    y_lo = float(row) * cell
    return x_lo, x_lo + cell, y_lo, y_lo + cell


def get_zone_center_feature(
    zone_id: int,
    zones_per_side: int | None = None,
    *,
    num_zones: int | None = None,
) -> tuple[float, float]:
    """Normalized `(x, y)` center of a row-major square-grid zone."""
    if zones_per_side is None:
        zones_per_side = _zones_per_side(9 if num_zones is None else int(num_zones))
    side = max(1, int(zones_per_side))
    z = int(zone_id)
    if z < 0 or z >= side * side:
        raise ValueError(f"Unknown zone {zone_id} for zones_per_side={side}")
    row = z // side
    col = z % side
    return (float(col) + 0.5) / float(side), (float(row) + 0.5) / float(side)


# --- Mobility ---------------------------------------------------------------

def move_annulus_jump(
    node,
    walls: Iterable[WallTuple],
    map_size: float,
    r_min: float,
    r_max: float,
    margin: float,
    rng: random.Random | None = None,
) -> bool:
    """Teleport the node to a random point in an annulus, snapped out of walls."""
    rng = rng or random
    angle = rng.uniform(0.0, 2.0 * math.pi)
    r = rng.uniform(r_min, r_max)
    x, y = clamp_xy(
        node.x + r * math.cos(angle),
        node.y + r * math.sin(angle),
        map_size,
        margin,
    )
    nx, ny, _ = snap_out_of_walls(x, y, walls, map_size, margin)
    if collides_with_walls(nx, ny, walls):
        return False
    node.x, node.y = float(nx), float(ny)
    return True


# --- Sampling within a zone -------------------------------------------------

def sample_free_point_in_zone(
    zone: int,
    walls: Iterable[WallTuple],
    map_size: float,
    margin: float,
    rng: np.random.Generator,
    num_zones: int = 4,
    max_tries: int = 200,
) -> tuple[float, float]:
    """Uniform sample in a zone, rejecting points inside walls."""
    x_lo, x_hi, y_lo, y_hi = zone_bounds(zone, map_size, num_zones)
    x_lo += margin
    x_hi -= margin
    y_lo += margin
    y_hi -= margin
    for _ in range(max_tries):
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        if not collides_with_walls(x, y, walls):
            return x, y
    # Fallback: snap from the centre of the zone.
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    nx, ny, _ = snap_out_of_walls(cx, cy, walls, map_size, margin)
    return float(nx), float(ny)


def _stratified_grid_points(
    *,
    n: int,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    walls: Iterable[WallTuple],
    map_size: float,
    margin: float,
    rng: np.random.Generator,
    jitter_frac: float = 0.25,
) -> list[tuple[float, float]]:
    """Return `n` stratified points inside (x_lo, x_hi) x (y_lo, y_hi).

    A near-square `rows x cols >= n` lattice is laid down across the rectangle.
    Each cell contributes one point near its centre with `+/- jitter_frac` of
    the cell size. Points landing inside walls fall back to a uniform free-point
    sample so the returned list always has length `n`.
    """
    if n <= 0:
        return []
    rows = max(1, int(math.sqrt(n)))
    cols = max(1, math.ceil(n / rows))
    cell_w = (x_hi - x_lo) / cols
    cell_h = (y_hi - y_lo) / rows
    out: list[tuple[float, float]] = []
    # Use a Halton-ish sequencing of cells to deterministically scatter
    # collisions when `rows*cols > n`.
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)  # type: ignore[arg-type]
    for r, c in cells:
        if len(out) >= n:
            break
        cx = x_lo + (c + 0.5) * cell_w + float(rng.uniform(-jitter_frac, jitter_frac)) * cell_w
        cy = y_lo + (r + 0.5) * cell_h + float(rng.uniform(-jitter_frac, jitter_frac)) * cell_h
        cx = float(min(max(cx, x_lo), x_hi))
        cy = float(min(max(cy, y_lo), y_hi))
        nx, ny, _ = snap_out_of_walls(cx, cy, walls, map_size, margin)
        if collides_with_walls(nx, ny, walls):
            # Fallback: uniform free-point inside the same rectangle. We pass
            # the original zone arg only via bounds; reusing
            # `sample_free_point_in_zone` would need the zone id, so we do a
            # local rejection sample here to keep this helper self-contained.
            for _ in range(50):
                rx = float(rng.uniform(x_lo, x_hi))
                ry = float(rng.uniform(y_lo, y_hi))
                if not collides_with_walls(rx, ry, walls):
                    nx, ny = rx, ry
                    break
            else:
                continue
        out.append((float(nx), float(ny)))
    while len(out) < n:
        for _ in range(50):
            rx = float(rng.uniform(x_lo, x_hi))
            ry = float(rng.uniform(y_lo, y_hi))
            if not collides_with_walls(rx, ry, walls):
                out.append((rx, ry))
                break
        else:
            # Last resort: append duplicate of an existing point so we don't
            # block the caller. Should be unreachable on a real map.
            out.append(out[-1])
    return out


def sample_oracle_pairs(
    zone: int,
    walls: Iterable[WallTuple],
    map_size: float,
    margin: float,
    n_tx: int,
    n_pairs: int,
    rng: np.random.Generator,
    num_zones: int = 4,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Sample `n_pairs` distinct (tx_xy, rx_xy) pairs in a zone, grouped into
    `n_tx` TX positions (each shared by `ceil(n_pairs/n_tx)` RXs at most) so
    we can ray-trace them with `n_tx` solver calls instead of `n_pairs`.

    Both TX and RX positions are stratified across the zone (jittered grid)
    so the returned set covers the (TX, RX) joint space evenly rather than
    clustering RXs randomly around each TX.

    Returns a list of pairs `[((tx_x, tx_y), (rx_x, rx_y)), ...]` of length
    exactly `n_pairs`.
    """
    x_lo, x_hi, y_lo, y_hi = zone_bounds(zone, map_size, num_zones)
    x_lo += margin
    x_hi -= margin
    y_lo += margin
    y_hi -= margin

    rx_per_tx = max(1, math.ceil(n_pairs / max(1, n_tx)))
    tx_positions = _stratified_grid_points(
        n=int(n_tx),
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
        walls=walls, map_size=map_size, margin=margin, rng=rng,
    )

    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for tx in tx_positions:
        if len(pairs) >= n_pairs:
            break
        rx_positions = _stratified_grid_points(
            n=int(rx_per_tx),
            x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
            walls=walls, map_size=map_size, margin=margin, rng=rng,
        )
        for rx in rx_positions:
            if len(pairs) >= n_pairs:
                break
            pairs.append((tx, rx))
    # Trim/pad to exact length (the stratified path is deterministic in size,
    # so the pad branch is a defensive fallback only).
    pairs = pairs[:n_pairs]
    while len(pairs) < n_pairs:
        tx = sample_free_point_in_zone(zone, walls, map_size, margin, rng, num_zones)
        rx = sample_free_point_in_zone(zone, walls, map_size, margin, rng, num_zones)
        pairs.append((tx, rx))
    return pairs


def group_pairs_by_tx(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], list[tuple[float, float]]]]:
    """Group `[(tx, rx), ...]` into `[(tx, [rx, rx, ...]), ...]` preserving tx order."""
    order: list[tuple[float, float]] = []
    groups: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for tx, rx in pairs:
        if tx not in groups:
            groups[tx] = []
            order.append(tx)
        groups[tx].append(rx)
    return [(tx, groups[tx]) for tx in order]
