#!/usr/bin/env python3
"""Build an elevation-aligned exact-footprint Sionna scene for a LuST3D crop."""

from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sumolib
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import triangulate


def add_triangle(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    points: list[tuple[float, float, float]],
    *,
    reverse: bool = False,
) -> None:
    start = len(vertices)
    vertices.extend(points)
    face = (start, start + 1, start + 2)
    faces.append(tuple(reversed(face)) if reverse else face)


def add_wall_quad(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    base: float,
    top: float,
) -> None:
    start = len(vertices)
    vertices.extend(
        [
            (float(a[0]), float(a[1]), float(base)),
            (float(b[0]), float(b[1]), float(base)),
            (float(b[0]), float(b[1]), float(top)),
            (float(a[0]), float(a[1]), float(top)),
        ]
    )
    faces.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])


def extrude_polygon(
    polygon: Polygon,
    *,
    base_z: float,
    height: float,
    x_origin: float,
    y_origin: float,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    geometry = orient(polygon, sign=1.0)
    top_z = float(base_z + height)
    for triangle in triangulate(geometry):
        if not geometry.covers(triangle.representative_point()):
            continue
        coords = list(triangle.exterior.coords)[:3]
        bottom = [
            (float(x - x_origin), float(y - y_origin), float(base_z))
            for x, y in coords
        ]
        top = [(x, y, top_z) for x, y, _z in bottom]
        add_triangle(vertices, faces, bottom, reverse=True)
        add_triangle(vertices, faces, top)

    rings = [geometry.exterior, *geometry.interiors]
    for ring in rings:
        coords = list(ring.coords)
        for raw_a, raw_b in zip(coords[:-1], coords[1:]):
            a = (float(raw_a[0] - x_origin), float(raw_a[1] - y_origin))
            b = (float(raw_b[0] - x_origin), float(raw_b[1] - y_origin))
            add_wall_quad(vertices, faces, a, b, base=float(base_z), top=top_z)


def write_binary_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for vertex in vertices:
            stream.write(struct.pack("<fff", *vertex))
        for face in faces:
            stream.write(struct.pack("<B3i", 3, *face))


def load_terrain_xyz(
    path: Path,
    *,
    net_offset: tuple[float, float],
    x_origin: float,
    y_origin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete regular UTM XYZ grid as local X/Y and absolute Z."""
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"terrain XYZ {path} must have exactly three columns")
    utm_x = np.unique(raw[:, 0])
    utm_y = np.unique(raw[:, 1])
    if len(raw) != len(utm_x) * len(utm_y):
        raise ValueError(f"terrain XYZ {path} is not a complete regular grid")
    z_grid = np.full((len(utm_y), len(utm_x)), np.nan, dtype=np.float64)
    z_grid[np.searchsorted(utm_y, raw[:, 1]), np.searchsorted(utm_x, raw[:, 0])] = raw[:, 2]
    if not np.isfinite(z_grid).all():
        raise ValueError(f"terrain XYZ {path} contains missing or non-finite heights")
    local_x = utm_x + float(net_offset[0]) - float(x_origin)
    local_y = utm_y + float(net_offset[1]) - float(y_origin)
    return local_x, local_y, z_grid


def terrain_grid_mesh(
    local_x: np.ndarray,
    local_y: np.ndarray,
    absolute_z: np.ndarray,
    *,
    z_reference: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if absolute_z.shape != (len(local_y), len(local_x)):
        raise ValueError("terrain Z grid shape does not match X/Y axes")
    vertices = [
        (float(x), float(y), float(absolute_z[yi, xi] - z_reference))
        for yi, y in enumerate(local_y)
        for xi, x in enumerate(local_x)
    ]
    faces: list[tuple[int, int, int]] = []
    nx = len(local_x)
    for yi in range(len(local_y) - 1):
        for xi in range(nx - 1):
            v00 = yi * nx + xi
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            faces.extend([(v00, v10, v11), (v00, v11, v01)])
    return vertices, faces


def add_slab_segment(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    *,
    width: float,
    thickness: float,
    z_reference: float,
    x_origin: float,
    y_origin: float,
) -> None:
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    length = float(np.hypot(dx, dy))
    if length <= 1.0e-6:
        return
    nx, ny = -dy / length * width / 2.0, dx / length * width / 2.0
    top_a, top_b = float(a[2] - z_reference - 0.05), float(b[2] - z_reference - 0.05)
    bottom_a, bottom_b = top_a - thickness, top_b - thickness
    ax, ay = float(a[0] - x_origin), float(a[1] - y_origin)
    bx, by = float(b[0] - x_origin), float(b[1] - y_origin)
    start = len(vertices)
    vertices.extend(
        [
            (ax + nx, ay + ny, top_a), (ax - nx, ay - ny, top_a),
            (bx + nx, by + ny, top_b), (bx - nx, by - ny, top_b),
            (ax + nx, ay + ny, bottom_a), (ax - nx, ay - ny, bottom_a),
            (bx + nx, by + ny, bottom_b), (bx - nx, by - ny, bottom_b),
        ]
    )
    faces.extend(
        [
            (start, start + 3, start + 2), (start, start + 1, start + 3),
            (start + 4, start + 6, start + 7), (start + 4, start + 7, start + 5),
            (start, start + 2, start + 6), (start, start + 6, start + 4),
            (start + 1, start + 5, start + 7), (start + 1, start + 7, start + 3),
            (start, start + 4, start + 5), (start, start + 5, start + 1),
            (start + 2, start + 3, start + 7), (start + 2, start + 7, start + 6),
        ]
    )


def bridge_deck_mesh(
    net,
    bounds: tuple[float, float, float, float],
    *,
    z_reference: float,
    x_origin: float,
    y_origin: float,
    thickness: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[str], int]:
    study = box(*bounds)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    edge_ids: list[str] = []
    lane_count = 0
    for edge in net.getEdges(withInternal=False):
        if "bridge" not in str(edge.getType()).lower():
            continue
        used_edge = False
        for lane in edge.getLanes():
            shape = [tuple(float(value) for value in point) for point in lane.getShape3D()]
            used_lane = False
            for a, b in zip(shape[:-1], shape[1:]):
                if not LineString([(a[0], a[1]), (b[0], b[1])]).intersects(study):
                    continue
                add_slab_segment(
                    vertices, faces, a, b,
                    width=float(lane.getWidth()), thickness=float(thickness),
                    z_reference=float(z_reference), x_origin=float(x_origin), y_origin=float(y_origin),
                )
                used_lane = True
            if used_lane:
                lane_count += 1
                used_edge = True
        if used_edge:
            edge_ids.append(str(edge.getID()))
    if not faces:
        raise ValueError("no bridge lane geometry intersects the scene bounds")
    return vertices, faces, sorted(edge_ids), lane_count


def lane_elevation_points(net, bounds: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    points: list[tuple[float, float, float]] = []
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            for x, y, z in lane.getShape3D():
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    points.append((float(x), float(y), float(z)))
    if not points:
        raise ValueError("no 3D lane points intersect the geometry bounds")
    raw = np.asarray(points, dtype=np.float64)
    # Multiple lanes repeat nearly identical elevation samples. Rounded XY
    # makes the nearest-neighbor base-height estimate less lane-count biased.
    rounded = np.round(raw[:, :2], decimals=1)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return raw[np.sort(indices)]


def densified_nonbridge_road_points(
    net,
    bounds: tuple[float, float, float, float],
    *,
    spacing: float = 5.0,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    points: list[tuple[float, float, float]] = []
    for edge in net.getEdges(withInternal=True):
        if "bridge" in str(edge.getType()).lower():
            continue
        for lane in edge.getLanes():
            shape = lane.getShape3D()
            for a, b in zip(shape[:-1], shape[1:]):
                length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                samples = max(2, int(np.ceil(length / float(spacing))) + 1)
                for alpha in np.linspace(0.0, 1.0, samples):
                    x = float(a[0] + alpha * (b[0] - a[0]))
                    y = float(a[1] + alpha * (b[1] - a[1]))
                    z = float(a[2] + alpha * (b[2] - a[2]))
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        points.append((x, y, z))
    if not points:
        raise ValueError("no non-bridge road points intersect the terrain bounds")
    return np.asarray(points, dtype=np.float64)


def condition_terrain_to_roads(
    local_x: np.ndarray,
    local_y: np.ndarray,
    absolute_z: np.ndarray,
    road_points: np.ndarray,
    *,
    x_origin: float,
    y_origin: float,
    radius: float,
    clearance: float,
) -> tuple[np.ndarray, int]:
    """Burn non-bridge road corridors into a coarse DEM without filling valleys."""
    grid_x, grid_y = np.meshgrid(local_x, local_y)
    query = np.column_stack(
        [(grid_x + float(x_origin)).ravel(), (grid_y + float(y_origin)).ravel()]
    )
    tree = cKDTree(road_points[:, :2])
    nearby = tree.query_ball_point(query, r=float(radius))
    mask = np.asarray([bool(indices) for indices in nearby], dtype=np.bool_)
    conditioned = np.asarray(absolute_z, dtype=np.float64).copy().reshape(-1)
    conditioned[mask] = [
        float(np.min(road_points[np.asarray(indices, dtype=np.int64), 2])) - float(clearance)
        for indices in nearby
        if indices
    ]
    return conditioned.reshape(absolute_z.shape), int(np.count_nonzero(mask))


def building_polygons(
    path: Path,
    bounds: tuple[float, float, float, float],
) -> list[tuple[str, Polygon, float]]:
    study = box(*bounds)
    output: list[tuple[str, Polygon, float]] = []
    for index, elem in enumerate(ET.parse(path).getroot().findall("poly")):
        if str(elem.get("type", "")).lower() != "building":
            continue
        points = [
            tuple(float(value) for value in token.split(",")[:2])
            for token in elem.get("shape", "").split()
        ]
        if len(points) < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.intersects(study):
            continue
        if polygon.geom_type != "Polygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area)
        height = float(elem.get("layer", "0"))
        if height <= 0.0:
            height = 14.0
        output.append((str(elem.get("id", index)), polygon, height))
    if not output:
        raise ValueError("no building polygons intersect the geometry bounds")
    return output


def scene_xml(
    building_path: Path,
    *,
    terrain_path: Path | None = None,
    bridge_path: Path | None = None,
) -> str:
    terrain_shape = (
        f"""
    <shape type="ply" id="luxembourg_terrain_ground">
        <string name="filename" value="{terrain_path.resolve()}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-ground" name="bsdf"/>
    </shape>"""
        if terrain_path is not None else ""
    )
    bridge_shape = (
        f"""
    <shape type="ply" id="luxembourg_bridge_deck">
        <string name="filename" value="{bridge_path.resolve()}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-concrete" name="bsdf"/>
    </shape>"""
        if bridge_path is not None else ""
    )
    return f"""<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="mat-concrete">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.3"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-ground">
        <string name="type" value="medium_dry_ground"/>
        <float name="thickness" value="0.5"/>
    </bsdf>
    <shape type="ply" id="luxembourg_buildings">
        <string name="filename" value="{building_path.resolve()}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-concrete" name="bsdf"/>
    </shape>{terrain_shape}{bridge_shape}
</scene>
"""


def radio_bounds_net(path: Path, size: float) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,{size:.2f},{size:.2f}" origBoundary="0.00,0.00,{size:.2f},{size:.2f}" projParameter="!"/>
</net>
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--sumo-net", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", default=None)
    parser.add_argument("--buffer", type=float, default=200.0)
    parser.add_argument("--z-reference", type=float, default=230.0)
    parser.add_argument("--base-neighbors", type=int, default=12)
    parser.add_argument("--terrain-xyz", type=Path, default=None)
    parser.add_argument("--terrain-source-url", default="")
    parser.add_argument("--bridge-deck", action="store_true")
    parser.add_argument("--bridge-deck-thickness", type=float, default=1.0)
    parser.add_argument("--road-condition-radius", type=float, default=10.0)
    parser.add_argument("--road-condition-clearance", type=float, default=0.5)
    args = parser.parse_args()

    manifest = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
    crop_name = str(args.crop or manifest["selected_crop"])
    crop = manifest["crops"][crop_name]
    xmin, ymin, xmax, ymax = [float(value) for value in crop["bounds_sumo_xy_m"]]
    buffer = float(args.buffer)
    geometry_bounds = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)
    net = sumolib.net.readNet(str(args.sumo_net), withInternal=True)
    elevation_points = lane_elevation_points(net, geometry_bounds)
    elevation_tree = cKDTree(elevation_points[:, :2])
    buildings = building_polygons(args.polygons, geometry_bounds)
    terrain_vertices: list[tuple[float, float, float]] = []
    terrain_faces: list[tuple[int, int, int]] = []
    terrain_absolute_z: np.ndarray | None = None
    terrain_local_x: np.ndarray | None = None
    terrain_local_y: np.ndarray | None = None
    terrain_interpolator: RegularGridInterpolator | None = None
    terrain_unconditioned_z: np.ndarray | None = None
    road_conditioned_vertices = 0
    if args.terrain_xyz is not None:
        terrain_local_x, terrain_local_y, terrain_absolute_z = load_terrain_xyz(
            args.terrain_xyz,
            net_offset=tuple(float(value) for value in net.getLocationOffset()),
            x_origin=xmin,
            y_origin=ymin,
        )
        terrain_unconditioned_z = terrain_absolute_z.copy()
        road_points = densified_nonbridge_road_points(net, geometry_bounds)
        terrain_absolute_z, road_conditioned_vertices = condition_terrain_to_roads(
            terrain_local_x,
            terrain_local_y,
            terrain_absolute_z,
            road_points,
            x_origin=xmin,
            y_origin=ymin,
            radius=float(args.road_condition_radius),
            clearance=float(args.road_condition_clearance),
        )
        terrain_vertices, terrain_faces = terrain_grid_mesh(
            terrain_local_x,
            terrain_local_y,
            terrain_absolute_z,
            z_reference=float(args.z_reference),
        )
        terrain_interpolator = RegularGridInterpolator(
            (terrain_local_y, terrain_local_x),
            terrain_absolute_z,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    base_elevations = []
    heights = []
    k = min(max(1, int(args.base_neighbors)), len(elevation_points))
    for _building_id, polygon, height in buildings:
        centroid = polygon.centroid
        if terrain_interpolator is not None:
            absolute_base = float(
                terrain_interpolator(
                    [[float(centroid.y - ymin), float(centroid.x - xmin)]]
                )[0]
            )
        else:
            _distances, indices = elevation_tree.query([centroid.x, centroid.y], k=k)
            nearest = np.atleast_1d(indices).astype(np.int64)
            absolute_base = float(np.median(elevation_points[nearest, 2]))
        local_base = absolute_base - float(args.z_reference)
        extrude_polygon(
            polygon,
            base_z=local_base,
            height=float(height),
            x_origin=xmin,
            y_origin=ymin,
            vertices=vertices,
            faces=faces,
        )
        base_elevations.append(absolute_base)
        heights.append(float(height))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = args.output_dir / f"{crop_name}_buildings_buffer{int(buffer)}m.ply"
    terrain_path = (
        args.output_dir / f"{crop_name}_terrain_ground_buffer{int(buffer)}m.ply"
        if terrain_faces else None
    )
    bridge_path = (
        args.output_dir / f"{crop_name}_bridge_deck_buffer{int(buffer)}m.ply"
        if args.bridge_deck else None
    )
    xml_path = args.output_dir / f"{crop_name}_scene.xml"
    net_path = args.output_dir / f"{crop_name}_radio_bounds.net.xml"
    report_path = args.output_dir / f"{crop_name}_scene_manifest.json"
    write_binary_ply(ply_path, vertices, faces)
    if terrain_path is not None:
        write_binary_ply(terrain_path, terrain_vertices, terrain_faces)
    bridge_vertices: list[tuple[float, float, float]] = []
    bridge_faces: list[tuple[int, int, int]] = []
    bridge_edge_ids: list[str] = []
    bridge_lane_count = 0
    if bridge_path is not None:
        bridge_vertices, bridge_faces, bridge_edge_ids, bridge_lane_count = bridge_deck_mesh(
            net,
            geometry_bounds,
            z_reference=float(args.z_reference),
            x_origin=xmin,
            y_origin=ymin,
            thickness=float(args.bridge_deck_thickness),
        )
        write_binary_ply(bridge_path, bridge_vertices, bridge_faces)
    xml_path.write_text(
        scene_xml(ply_path, terrain_path=terrain_path, bridge_path=bridge_path),
        encoding="utf-8",
    )
    radio_bounds_net(net_path, float(xmax - xmin))

    report = {
        "format": (
            "radiodiff_luxembourg_sionna_scene_v2"
            if terrain_path is not None or bridge_path is not None
            else "radiodiff_luxembourg_sionna_scene_v1"
        ),
        "crop": crop_name,
        "measurement_bounds_local_xy_m": [0.0, 0.0, xmax - xmin, ymax - ymin],
        "geometry_bounds_local_xy_m": [-buffer, -buffer, xmax - xmin + buffer, ymax - ymin + buffer],
        "source_bounds_sumo_xy_m": [xmin, ymin, xmax, ymax],
        "z_reference_m": float(args.z_reference),
        "building_count": len(buildings),
        "mesh_vertices": len(vertices),
        "mesh_triangles": len(faces),
        "building_height_min_m": float(np.min(heights)),
        "building_height_median_m": float(np.median(heights)),
        "building_height_max_m": float(np.max(heights)),
        "building_base_elevation_min_m": float(np.min(base_elevations)),
        "building_base_elevation_median_m": float(np.median(base_elevations)),
        "building_base_elevation_max_m": float(np.max(base_elevations)),
        "lane_elevation_min_m": float(np.min(elevation_points[:, 2])),
        "lane_elevation_median_m": float(np.median(elevation_points[:, 2])),
        "lane_elevation_max_m": float(np.max(elevation_points[:, 2])),
        "building_base_method": (
            "bilinear interpolation of official terrain grid at footprint centroid"
            if terrain_interpolator is not None
            else f"median elevation of {k} nearest 3D lane-shape points"
        ),
        "terrain_mesh_included": terrain_path is not None,
        "terrain_source_xyz": (
            str(args.terrain_xyz.resolve()) if args.terrain_xyz is not None else ""
        ),
        "terrain_source_url": str(args.terrain_source_url),
        "terrain_material": "itu_medium_dry_ground",
        "terrain_grid_shape": (
            [int(len(terrain_local_y)), int(len(terrain_local_x))]
            if terrain_local_x is not None and terrain_local_y is not None else []
        ),
        "terrain_grid_spacing_m": (
            [
                float(np.median(np.diff(terrain_local_x))),
                float(np.median(np.diff(terrain_local_y))),
            ]
            if terrain_local_x is not None and terrain_local_y is not None
            and len(terrain_local_x) > 1 and len(terrain_local_y) > 1 else []
        ),
        "terrain_unconditioned_elevation_min_m": (
            float(np.min(terrain_unconditioned_z))
            if terrain_unconditioned_z is not None else None
        ),
        "terrain_unconditioned_elevation_median_m": (
            float(np.median(terrain_unconditioned_z))
            if terrain_unconditioned_z is not None else None
        ),
        "terrain_unconditioned_elevation_max_m": (
            float(np.max(terrain_unconditioned_z))
            if terrain_unconditioned_z is not None else None
        ),
        "road_conditioning_method": "minimum nearby densified non-bridge LuST3D lane elevation",
        "road_conditioning_radius_m": float(args.road_condition_radius),
        "road_conditioning_clearance_m": float(args.road_condition_clearance),
        "road_conditioned_terrain_vertices": int(road_conditioned_vertices),
        "terrain_elevation_min_m": (
            float(np.min(terrain_absolute_z)) if terrain_absolute_z is not None else None
        ),
        "terrain_elevation_median_m": (
            float(np.median(terrain_absolute_z)) if terrain_absolute_z is not None else None
        ),
        "terrain_elevation_max_m": (
            float(np.max(terrain_absolute_z)) if terrain_absolute_z is not None else None
        ),
        "terrain_mesh_vertices": len(terrain_vertices),
        "terrain_mesh_triangles": len(terrain_faces),
        "bridge_deck_mesh_included": bridge_path is not None,
        "bridge_deck_source": "LuST3D lanes whose edge type contains bridge",
        "bridge_deck_edge_ids": bridge_edge_ids,
        "bridge_deck_lane_count": int(bridge_lane_count),
        "bridge_deck_thickness_m": float(args.bridge_deck_thickness),
        "bridge_deck_mesh_vertices": len(bridge_vertices),
        "bridge_deck_mesh_triangles": len(bridge_faces),
        "pilot_limitations": (
            "Terrain/ground uses the official Luxembourg 2024 DTM and the deck "
            "uses LuST3D bridge-lane elevations. Bridge piers, railings, vegetation, "
            "and spatially varying road-surface materials are not modeled."
        ),
        "scene_xml": str(xml_path.resolve()),
        "building_mesh": str(ply_path.resolve()),
        "terrain_mesh": str(terrain_path.resolve()) if terrain_path is not None else "",
        "bridge_deck_mesh": str(bridge_path.resolve()) if bridge_path is not None else "",
        "radio_bounds_net": str(net_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{report_path} buildings={len(buildings)} "
        f"building_triangles={len(faces)} terrain_triangles={len(terrain_faces)} "
        f"bridge_triangles={len(bridge_faces)} "
        f"base_z={min(base_elevations):.1f}..{max(base_elevations):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
