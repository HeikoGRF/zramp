"""Dynamic radio-obstacle schedules for SUMO-derived Sionna maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DynamicObstacle:
    """One movable, axis-aligned radio blocker in a Sionna scene."""

    event_id: str
    kind: str
    zone: int
    center: tuple[float, float]
    size: tuple[float, float, float]
    material: str
    active_intervals: tuple[tuple[int, int], ...]
    period_steps: int | None = None
    active_steps: int | None = None
    phase_steps: int = 0
    replaces_static_block: bool = False
    placement: str = "dynamic"
    description: str = ""

    @property
    def scene_id(self) -> str:
        return self.event_id if self.event_id.startswith("dyn_") else f"dyn_{self.event_id}"

    @property
    def active_center_position(self) -> tuple[float, float, float]:
        return (self.center[0], self.center[1], 0.5 * self.size[2])

    def active_at(self, step: int) -> bool:
        s = int(step)
        if self.period_steps is not None:
            assert self.active_steps is not None
            phase_step = (s + int(self.phase_steps)) % int(self.period_steps)
            return phase_step < int(self.active_steps)
        return any(start <= s <= end for start, end in self.active_intervals)

    def as_wall(self) -> tuple[str, float, float, float, float, float]:
        return (
            self.scene_id,
            self.center[0],
            self.center[1],
            self.size[0],
            self.size[1],
            self.size[2],
        )


@dataclass(frozen=True)
class DynamicObstacleSchedule:
    """A reproducible schedule of dynamic map changes."""

    source_path: str
    seed: int | None
    sim_steps: int | None
    obstacles: tuple[DynamicObstacle, ...]

    def active_ids(self, step: int) -> list[str]:
        return [obs.scene_id for obs in self.obstacles if obs.active_at(step)]

    def coverage_gaps(self, *, sim_steps: int | None = None) -> list[tuple[int, int]]:
        n_steps = int(sim_steps if sim_steps is not None else (self.sim_steps or 0))
        if n_steps <= 0:
            return []
        gaps: list[tuple[int, int]] = []
        gap_start: int | None = None
        for step in range(1, n_steps + 1):
            if self.active_ids(step):
                if gap_start is not None:
                    gaps.append((gap_start, step - 1))
                    gap_start = None
            elif gap_start is None:
                gap_start = step
        if gap_start is not None:
            gaps.append((gap_start, n_steps))
        return gaps

    def summary(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "seed": self.seed,
            "sim_steps": self.sim_steps,
            "n_obstacles": len(self.obstacles),
            "coverage_gaps": self.coverage_gaps(),
            "obstacles": [
                {
                    "id": obs.scene_id,
                    "kind": obs.kind,
                    "zone": obs.zone,
                    "center": list(obs.center),
                    "size": list(obs.size),
                    "material": obs.material,
                    "placement": obs.placement,
                    "replaces_static_block": obs.replaces_static_block,
                    "active": [list(iv) for iv in obs.active_intervals],
                    "periodic": (
                        None
                        if obs.period_steps is None
                        else {
                            "period_steps": obs.period_steps,
                            "active_steps": obs.active_steps,
                            "phase_steps": obs.phase_steps,
                        }
                    ),
                }
                for obs in self.obstacles
            ],
        }


def _parse_float_pair(raw: Any, *, field: str, event_id: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{event_id}: {field} must be a 2-item list")
    return (float(raw[0]), float(raw[1]))


def _parse_size(raw: Any, *, event_id: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{event_id}: size must be a 3-item list")
    sx, sy, sz = (float(raw[0]), float(raw[1]), float(raw[2]))
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"{event_id}: size values must be positive")
    return (sx, sy, sz)


def _parse_intervals(raw: Any, *, event_id: str) -> tuple[tuple[int, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{event_id}: active must be a list of [start, end] intervals")
    intervals: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{event_id}: every active interval must be [start, end]")
        start = int(item[0])
        end = int(item[1])
        if start < 0 or end < start:
            raise ValueError(f"{event_id}: invalid active interval [{start}, {end}]")
        intervals.append((start, end))
    return tuple(sorted(intervals))


def load_dynamic_obstacle_schedule(path: str | Path) -> DynamicObstacleSchedule:
    schedule_path = Path(path).expanduser().resolve()
    with open(schedule_path, encoding="utf-8") as f:
        raw = json.load(f)
    events = raw.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{schedule_path}: expected a non-empty events list")

    obstacles: list[DynamicObstacle] = []
    seen: set[str] = set()
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise ValueError(f"{schedule_path}: event #{idx} must be an object")
        event_id = str(ev.get("id", f"event_{idx}"))
        scene_id = event_id if event_id.startswith("dyn_") else f"dyn_{event_id}"
        if scene_id in seen:
            raise ValueError(f"{schedule_path}: duplicate dynamic obstacle id {scene_id}")
        seen.add(scene_id)
        kind = str(ev.get("kind", "temporary_obstacle"))
        placement = str(ev.get("placement", "dynamic")).lower()
        if "roadblock" in kind.lower() or "roadblock" in placement:
            raise ValueError(f"{event_id}: dynamic obstacles must be free-space blockers, not roadblocks")
        if "traffic_active" in ev:
            raise ValueError(f"{event_id}: dynamic obstacles must not define traffic_active/rerouting intervals")
        intervals = _parse_intervals(ev.get("active"), event_id=event_id)
        period_raw = ev.get("period_steps")
        period_steps = int(period_raw) if period_raw is not None else None
        active_steps = int(ev.get("active_steps")) if period_steps is not None else None
        phase_steps = int(ev.get("phase_steps", 0))
        if period_steps is None:
            if not intervals:
                raise ValueError(f"{event_id}: define either active intervals or period_steps")
        else:
            if period_steps <= 1:
                raise ValueError(f"{event_id}: period_steps must exceed one")
            if active_steps is None or not 0 < active_steps < period_steps:
                raise ValueError(
                    f"{event_id}: active_steps must be in [1, period_steps)"
                )
            if intervals:
                raise ValueError(
                    f"{event_id}: do not combine active intervals with a periodic schedule"
                )
        obstacles.append(
            DynamicObstacle(
                event_id=event_id,
                kind=kind,
                zone=int(ev.get("zone", -1)),
                center=_parse_float_pair(ev.get("center"), field="center", event_id=event_id),
                size=_parse_size(ev.get("size"), event_id=event_id),
                material=str(ev.get("material", "itu_concrete")),
                active_intervals=intervals,
                period_steps=period_steps,
                active_steps=active_steps,
                phase_steps=phase_steps,
                replaces_static_block=bool(ev.get("replaces_static_block", False)),
                placement=placement,
                description=str(ev.get("description", "")),
            )
        )

    seed = raw.get("seed")
    sim_steps = raw.get("sim_steps")
    return DynamicObstacleSchedule(
        source_path=str(schedule_path),
        seed=int(seed) if seed is not None else None,
        sim_steps=int(sim_steps) if sim_steps is not None else None,
        obstacles=tuple(obstacles),
    )
