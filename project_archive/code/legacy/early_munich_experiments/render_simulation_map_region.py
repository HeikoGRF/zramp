"""
Render the exact Munich scene region used by the zone simulations.

This script produces a PNG showing building occupancy inside the simulation
bounds and overlays the 2×2 zone grid used by the benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

import sionna.rt as rt


def _build_grid_safe(scene, bounds_x, bounds_y, res: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(bounds_x[0], bounds_x[1] + res, res)
    ys = np.arange(bounds_y[0], bounds_y[1] + res, res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel()]
    mask = np.zeros(len(points), dtype=bool)

    for name, obj in scene.objects.items():
        if "ground" in name.lower():
            continue
        mesh = obj.mi_mesh
        try:
            vertices = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
            faces = np.array(mesh.faces_buffer()).reshape(-1, 3)
        except Exception:
            continue
        verts2d = vertices[:, :2]
        for face in faces:
            poly = verts2d[face]
            p = MplPath(poly)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            if xmax < bounds_x[0] or xmin > bounds_x[1] or ymax < bounds_y[0] or ymin > bounds_y[1]:
                continue
            in_bbox = (
                (points[:, 0] >= xmin - res)
                & (points[:, 0] <= xmax + res)
                & (points[:, 1] >= ymin - res)
                & (points[:, 1] <= ymax + res)
            )
            if np.any(in_bbox):
                idx = np.where(in_bbox)[0]
                inside = p.contains_points(points[idx])
                mask[idx[inside]] = True

    return xx, yy, mask.reshape(xx.shape)


def render(out_path: Path) -> Path:
    # Must match the benchmark scripts exactly
    bounds_x = (0, 99)
    bounds_y = (-199, -100)
    n_col, n_row = 2, 2

    scene = rt.load_scene(rt.scene.munich)
    xx, yy, grid_mask = _build_grid_safe(scene, bounds_x, bounds_y, res=1.0)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(
        grid_mask.astype(float),
        origin="lower",
        extent=(bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]),
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Overlay zone grid (50×50 m in a 2×2 split)
    zone_w = (bounds_x[1] - bounds_x[0]) / n_col
    zone_h = (bounds_y[1] - bounds_y[0]) / n_row

    for c in range(1, n_col):
        x = bounds_x[0] + c * zone_w
        ax.axvline(x, color="#ff6f00", linewidth=2.0, alpha=0.9)
    for r in range(1, n_row):
        y = bounds_y[0] + r * zone_h
        ax.axhline(y, color="#ff6f00", linewidth=2.0, alpha=0.9)

    # Zone labels (row-major from south to north)
    zone_labels = {
        0: "Zone 0 (SW)",
        1: "Zone 1 (SE)",
        2: "Zone 2 (NW)",
        3: "Zone 3 (NE)",
    }
    for row in range(n_row):
        for col in range(n_col):
            zid = row * n_col + col
            cx = bounds_x[0] + (col + 0.5) * zone_w
            cy = bounds_y[0] + (row + 0.5) * zone_h
            ax.text(
                cx,
                cy,
                zone_labels[zid],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#ff6f00",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.7),
            )

    ax.set_title("Sionna Munich — exact simulation region (x=0..99, y=-199..-100)", fontsize=11, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out = render(Path("benchmark_results") / "simulation_region_map.png")
    print(out)


if __name__ == "__main__":
    main()

