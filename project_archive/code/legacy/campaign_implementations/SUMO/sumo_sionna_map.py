"""
Build a Sionna scene directly from a SUMO .net.xml urban layout.

The scene is approximated as city blocks (concrete cuboids) between the road
corridors induced by SUMO junction coordinates.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Import Sionna only when a live radio scene is actually built. CPU-only SUMO
# mobility export imports this module for ``read_net_bounds`` and should not pay
# TensorFlow/Mitsuba startup cost or probe a GPU.
rt = None  # type: ignore[assignment]


def _require_sionna_rt():
    global rt
    if rt is None:
        try:
            import sionna.rt as sionna_rt
        except (ModuleNotFoundError, ImportError) as exc:
            raise RuntimeError("sionna.rt is required to build a live Sionna scene") from exc
        rt = sionna_rt
    return rt

from SUMO.dynamic_obstacles import DynamicObstacle, DynamicObstacleSchedule, load_dynamic_obstacle_schedule


SionnaVariant = Literal["standard", "showcase", "controlled-4zone", "single-zone-urban"]


def sionna_variant_for_net(net_path: str | Path) -> SionnaVariant:
    name = Path(net_path).name
    if "single_zone_urban_" in name or "source_train_" in name or "source_valid_" in name:
        return "single-zone-urban"
    if "controlled_4zone_300" in name:
        return "controlled-4zone"
    if "designed_city_center" in name:
        return "showcase"
    return "standard"


@dataclass
class NetBounds:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return float(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return float(self.y1 - self.y0)


def read_net_bounds(net_path: str) -> NetBounds:
    root = ET.parse(net_path).getroot()
    loc = root.find("location")
    if loc is None:
        raise ValueError("SUMO net has no <location> element")
    conv = loc.get("convBoundary")
    if not conv:
        raise ValueError("SUMO net location has no convBoundary")
    x0, y0, x1, y1 = [float(v) for v in conv.split(",")]
    return NetBounds(x0=x0, y0=y0, x1=x1, y1=y1)


class SumoNetSionnaMap:
    def __init__(
        self,
        *,
        net_path: str,
        frequency: float = 3.5e9,
        road_half_width: float | None = None,
        building_height: float | None = None,
        sionna_variant: SionnaVariant = "standard",
        dynamic_schedule_path: str | None = None,
        polygon_path: str | None = None,
    ):
        self.net_path = str(net_path)
        self.frequency = float(frequency)
        self.sionna_variant = sionna_variant
        self.dynamic_schedule_path = str(dynamic_schedule_path) if dynamic_schedule_path else None
        self.polygon_path = str(polygon_path) if polygon_path else self._infer_polygon_path()
        if sionna_variant == "single-zone-urban":
            self.road_half_width = float(7.0 if road_half_width is None else road_half_width)
            self.building_height = float(14.0 if building_height is None else building_height)
        elif sionna_variant == "controlled-4zone":
            self.road_half_width = float(7.0 if road_half_width is None else road_half_width)
            self.building_height = float(8.0 if building_height is None else building_height)
        elif sionna_variant == "showcase":
            # Demonstration layout: stronger cross-quarter contrast (canyon vs open, low vs high rise)
            # so selective model sharing outperforms unconditional greedy averaging.
            self.road_half_width = float(7.0 if road_half_width is None else road_half_width)
            self.building_height = float(24.0 if building_height is None else building_height)
        else:
            self.road_half_width = float(8.0 if road_half_width is None else road_half_width)
            self.building_height = float(12.0 if building_height is None else building_height)

        self.scene: Any | None = None
        self._xml_path: str | None = None
        self._ply_path: str | None = None
        self.bounds = read_net_bounds(self.net_path)
        self._mid_x = 0.5 * (self.bounds.x0 + self.bounds.x1)
        self._mid_y = 0.5 * (self.bounds.y0 + self.bounds.y1)
        # Quarter-specific urban styles (same 4-way split as RL zones in sumo_rl).
        if self.sionna_variant == "single-zone-urban":
            self.quarter_profiles = {
                q: {"h_mul": 1.0, "shrink": 1.0, "material": "itu_concrete"}
                for q in range(4)
            }
        elif self.sionna_variant == "controlled-4zone":
            # Matched static/dynamic pairs: left quarters are sparse 28x28x8 m,
            # right quarters are dense 54x54x24 m. All materials stay fixed.
            self.quarter_profiles = {
                0: {"h_mul": 1.0, "shrink": 28.0 / 61.0, "material": "itu_concrete"},
                1: {"h_mul": 3.0, "shrink": 54.0 / 61.0, "material": "itu_concrete"},
                2: {"h_mul": 1.0, "shrink": 28.0 / 61.0, "material": "itu_concrete"},
                3: {"h_mul": 3.0, "shrink": 54.0 / 61.0, "material": "itu_concrete"},
            }
        elif self.sionna_variant == "showcase":
            self.quarter_profiles = {
                0: {"h_mul": 1.35, "shrink": 0.80, "material": "itu_stone"},
                1: {"h_mul": 1.15, "shrink": 0.92, "material": "itu_concrete"},
                2: {"h_mul": 0.72, "shrink": 1.10, "material": "itu_wood"},
                3: {"h_mul": 2.05, "shrink": 0.72, "material": "itu_glass"},
            }
        else:
            self.quarter_profiles = {
                0: {"h_mul": 1.35, "shrink": 0.85, "material": "itu_stone"},
                1: {"h_mul": 1.10, "shrink": 0.95, "material": "itu_concrete"},
                2: {"h_mul": 0.85, "shrink": 1.05, "material": "itu_wood"},
                3: {"h_mul": 1.55, "shrink": 0.80, "material": "itu_glass"},
            }
        self.block_material: dict[str, str] = {}
        self.dynamic_schedule: DynamicObstacleSchedule | None = (
            load_dynamic_obstacle_schedule(self.dynamic_schedule_path)
            if self.dynamic_schedule_path
            else None
        )
        self.dynamic_obstacles: tuple[DynamicObstacle, ...] = (
            self.dynamic_schedule.obstacles if self.dynamic_schedule else ()
        )
        self.walls = (
            self._build_blocks_from_polygons()
            if self.sionna_variant == "single-zone-urban"
            else self._build_blocks_from_net()
        )
        self._dynamic_active_state: dict[str, bool] = {}
        self._last_dynamic_active: tuple[str, ...] = ()

    def _infer_polygon_path(self) -> str | None:
        network = Path(self.net_path)
        suffix = ".net.xml"
        if not network.name.endswith(suffix):
            return None
        candidate = network.with_name(network.name[: -len(suffix)] + ".poly.xml")
        return str(candidate) if candidate.is_file() else None

    def _build_blocks_from_polygons(
        self,
    ) -> list[tuple[str, float, float, float, float, float]]:
        if not self.polygon_path:
            raise ValueError("single-zone urban maps require a sibling .poly.xml file")
        root = ET.parse(self.polygon_path).getroot()
        walls: list[tuple[str, float, float, float, float, float]] = []
        for index, poly in enumerate(root.findall("poly")):
            if str(poly.get("type", "")).lower() != "building":
                continue
            points: list[tuple[float, float]] = []
            for token in str(poly.get("shape", "")).split():
                raw_x, raw_y = token.split(",")[:2]
                points.append((float(raw_x), float(raw_y)))
            if len(points) < 4:
                raise ValueError(f"{poly.get('id')}: building polygon has fewer than four points")
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if x1 - x0 <= 2.0 or y1 - y0 <= 2.0:
                raise ValueError(f"{poly.get('id')}: building polygon is too small")
            params = {
                str(param.get("key")): str(param.get("value"))
                for param in poly.findall("param")
            }
            height = float(params.get("height", self.building_height))
            name = f"poly_{poly.get('id', index)}"
            replacement = next(
                (
                    obs
                    for obs in self.dynamic_obstacles
                    if obs.replaces_static_block
                    and abs(float(obs.center[0]) - 0.5 * (x0 + x1)) <= 1.0e-6
                    and abs(float(obs.center[1]) - 0.5 * (y0 + y1)) <= 1.0e-6
                ),
                None,
            )
            if replacement is not None:
                expected = (x1 - x0, y1 - y0, height)
                if any(
                    abs(float(actual) - float(wanted)) > 1.0e-5
                    for actual, wanted in zip(replacement.size, expected)
                ):
                    raise ValueError(
                        f"{replacement.event_id}: replacement size {replacement.size} "
                        f"does not match polygon {poly.get('id')} size {expected}"
                    )
                continue
            walls.append(
                (name, 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0, height)
            )
            self.block_material[name] = params.get("material", "itu_concrete")
        if not walls:
            raise ValueError(f"{self.polygon_path}: no building polygons found")
        return walls

    def _build_blocks_from_net(self) -> list[tuple[str, float, float, float, float, float]]:
        root = ET.parse(self.net_path).getroot()
        jx: list[float] = []
        jy: list[float] = []
        for j in root.findall("junction"):
            if j.get("type") == "internal":
                continue
            x = j.get("x")
            y = j.get("y")
            if x is None or y is None:
                continue
            jx.append(float(x))
            jy.append(float(y))

        xs = sorted(set(jx))
        ys = sorted(set(jy))
        if len(xs) < 2 or len(ys) < 2:
            return []

        walls: list[tuple[str, float, float, float, float, float]] = []
        k = 0
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                left = xs[ix] + self.road_half_width
                right = xs[ix + 1] - self.road_half_width
                bot = ys[iy] + self.road_half_width
                top = ys[iy + 1] - self.road_half_width
                if right - left <= 2.0 or top - bot <= 2.0:
                    continue
                cx = 0.5 * (left + right)
                cy = 0.5 * (bot + top)
                qx = 0 if cx < self._mid_x else 1
                qy = 0 if cy < self._mid_y else 1
                q = qx + 2 * qy
                prof = self.quarter_profiles[q]
                th = float(right - left) * float(prof["shrink"])   # x extent
                ln = float(top - bot) * float(prof["shrink"])      # y extent
                h = float(self.building_height) * float(prof["h_mul"])
                replacement = next(
                    (
                        obs
                        for obs in self.dynamic_obstacles
                        if obs.replaces_static_block
                        and abs(float(obs.center[0]) - cx) <= 1.0e-6
                        and abs(float(obs.center[1]) - cy) <= 1.0e-6
                    ),
                    None,
                )
                if replacement is not None:
                    expected = (th, ln, h)
                    if any(
                        abs(float(a) - float(b)) > 1.0e-5
                        for a, b in zip(replacement.size, expected)
                    ):
                        raise ValueError(
                            f"{replacement.event_id}: replacement size {replacement.size} "
                            f"does not match generated block size {expected}"
                        )
                    k += 1
                    continue
                name = f"blk_{k}"
                walls.append((name, cx, cy, th, ln, h))
                self.block_material[name] = str(prof["material"])
                k += 1
        return walls

    def build(self) -> Any:
        _require_sionna_rt()
        from build_map import _build_scene_xml_generic, _write_cube_ply

        tmp_dir = tempfile.gettempdir()
        self._ply_path = os.path.join(tmp_dir, f"sionna_sumo_cube_{id(self)}.ply")
        _write_cube_ply(self._ply_path)

        scene_walls = list(self.walls)
        scene_walls.extend(obs.as_wall() for obs in self.dynamic_obstacles)
        xml = _build_scene_xml_generic(self._ply_path, scene_walls)
        xml_tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
        xml_tmp.write(xml)
        xml_tmp.close()
        self._xml_path = xml_tmp.name

        if self.dynamic_obstacles:
            scene = rt.load_scene(
                self._xml_path,
                merge_shapes=True,
                merge_shapes_exclude_regex=r"^dyn_",
            )
        else:
            scene = rt.load_scene(self._xml_path)
        scene.frequency = self.frequency
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        materials = {
            "itu_concrete": rt.RadioMaterial(
                "itu_concrete", relative_permittivity=5.31, conductivity=0.0326
            ),
            "itu_stone": rt.RadioMaterial(
                "itu_stone", relative_permittivity=6.27, conductivity=0.04
            ),
            "itu_wood": rt.RadioMaterial(
                "itu_wood", relative_permittivity=1.99, conductivity=0.0047
            ),
            "itu_glass": rt.RadioMaterial(
                "itu_glass", relative_permittivity=6.06, conductivity=0.012
            ),
            "itu_brick": rt.RadioMaterial(
                "itu_brick", relative_permittivity=3.91, conductivity=0.0238
            ),
            "itu_metal": rt.RadioMaterial(
                "itu_metal", relative_permittivity=1.0, conductivity=1.0e7
            ),
        }
        for mat in materials.values():
            scene.add(mat)
        for name, *_ in self.walls:
            obj = scene.get(name)
            if obj:
                mat_name = self.block_material.get(name, "itu_concrete")
                obj.radio_material = materials.get(mat_name, materials["itu_concrete"])
        for obs in self.dynamic_obstacles:
            obj = scene.get(obs.scene_id)
            if obj:
                obj.radio_material = materials.get(obs.material, materials["itu_concrete"])
        self.scene = scene
        if self.dynamic_obstacles:
            self.apply_dynamic_step(0)
        return scene

    def _inactive_position(self, idx: int, obs: DynamicObstacle) -> tuple[float, float, float]:
        # Keep inactive blockers out of the radio path without changing SUMO mobility.
        return (obs.center[0], obs.center[1], -100.0 - float(idx))

    def apply_dynamic_step(self, step: int) -> list[str]:
        """Move scheduled radio obstacles into or out of the active scene."""
        if self.scene is None or not self.dynamic_obstacles:
            return []

        active: list[str] = []
        for idx, obs in enumerate(self.dynamic_obstacles):
            is_active = obs.active_at(int(step))
            if is_active:
                active.append(obs.scene_id)

            previous = self._dynamic_active_state.get(obs.scene_id)
            if previous is is_active:
                continue

            obj = self.scene.get(obs.scene_id)
            if obj is None:
                continue
            target = obs.active_center_position if is_active else self._inactive_position(idx, obs)
            obj.position = [float(target[0]), float(target[1]), float(target[2])]
            self._dynamic_active_state[obs.scene_id] = is_active

        self._last_dynamic_active = tuple(active)
        return active

    def cleanup(self) -> None:
        for path_attr in ("_xml_path", "_ply_path"):
            path = getattr(self, path_attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
                setattr(self, path_attr, None)
