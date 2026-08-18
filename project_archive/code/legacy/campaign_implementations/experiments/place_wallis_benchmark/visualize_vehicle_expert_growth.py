#!/usr/bin/env python3
"""Replay one Place Wallis expert-bank vehicle and render corridor growth."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RENDER_ONLY = "--render-capture" in sys.argv
if RENDER_ONLY:
    class _RenderDefaults:
        SupportExpertBankSimulation = object
        DEFAULT_TRACE = Path(
            "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
            "place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers/"
            "rssi/place_wallis_vehicles_0745_0815_1s_opaque_"
            "no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz"
        )
        DEFAULT_TESTSET = Path(
            "/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/"
            "place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers/"
            "testset/place_wallis_street_pairs_10000_opaque_"
            "no_vehicle_blockers_static_floor100.npz"
        )
        DEFAULT_NET = (
            ROOT / "SUMO/luxembourg_real_city/place_wallis/map/sionna/"
            "place_wallis_300m_radio_bounds.net.xml"
        )

    runner = _RenderDefaults()
else:
    from experiments.place_wallis_benchmark import (
        run_support_expert_bank as runner,
    )


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/place_wallis_benchmark/visualizations/"
    "expert_corridor_growth_k6_vehicle517"
)
DEFAULT_BACKGROUND = (
    ROOT
    / "SUMO/luxembourg_real_city/place_wallis/place_wallis_map_300m.png"
)
CAPTURE: dict[str, Any] = {}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def clock_string(step: int) -> str:
    seconds = (7 * 3600 + 45 * 60 + int(step)) % (24 * 3600)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def plane_polygon(row: list[float], *, scale: float, height: int) -> list[tuple[float, float]]:
    start = np.asarray(row[0:2], dtype=np.float64)
    end = np.asarray(row[2:4], dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        return []
    axis = vector / length
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    corners = (
        start + float(row[4]) * normal,
        start + float(row[5]) * normal,
        end + float(row[7]) * normal,
        end + float(row[6]) * normal,
    )
    return [(float(p[0]) * scale, height - float(p[1]) * scale) for p in corners]


def frame_cutoff(frames: list[dict[str, Any]], *, entry_step: int) -> tuple[int, int | None]:
    """Return inclusive frame index and detected plateau step."""
    running_max: list[float] = []
    current = 0.0
    for row in frames:
        current = max(current, float(row["coverage"]))
        running_max.append(current)
    plateau_step: int | None = None
    cutoff = len(frames) - 1
    for index, row in enumerate(frames):
        step = int(row["step"])
        if step < entry_step + 45:
            continue
        earlier_step = step - 30
        earlier = max(
            (j for j, candidate in enumerate(frames) if int(candidate["step"]) <= earlier_step),
            default=0,
        )
        if running_max[index] - running_max[earlier] < 0.01:
            plateau_step = step
            target_step = step + 5
            cutoff = min(
                len(frames) - 1,
                max(
                    (j for j, candidate in enumerate(frames) if int(candidate["step"]) <= target_step),
                    default=index,
                ),
            )
            break
    return cutoff, plateau_step


class RecordingExpertBankSimulation(runner.SupportExpertBankSimulation):
    """Unmodified K-bank simulation with a lightweight single-node observer."""

    latest_instance: "RecordingExpertBankSimulation | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capture_frames: list[dict[str, Any]] = []
        self.cfg.fidelity_eval_every = 0
        self.cfg.fidelity_final_steps = ()
        self.cfg.fidelity_grid_per_zone = 1
        self.cfg.final_fidelity_grid_per_zone = 1
        RecordingExpertBankSimulation.latest_instance = self

    def _greedy_share_step(
        self,
        zone_nodes: dict[int, list[int]],
        contact_links: list[tuple[int, int, int]] | None = None,
    ) -> int:
        target = int(CAPTURE["target_node"])
        step = int(getattr(self, "_current_sumo_step", 0))
        before_keys = list(self._expert_banks[target]) if self._expert_banks else []
        before_versions = {
            key[:2]: int(key[2])
            for key in before_keys
            if int(key[0]) != target
        }
        before_local = set(self._local_support[target]) if self._local_support else set()
        staged = self._staged_measurements or []
        local_samples = sum(1 for _zone, _tx, rx, _value in staged if int(rx) == target)

        messages = super()._greedy_share_step(zone_nodes, contact_links=contact_links)

        if not int(CAPTURE["capture_start"]) <= step <= int(CAPTURE["capture_end"]):
            return messages
        keys = list(self._expert_banks[target])
        own_lineage = (target, int(self._expert_incarnations[target]))
        provider_updates = sorted({
            int(key[0])
            for key in keys
            if key[:2] != own_lineage
            and int(key[2]) > before_versions.get(key[:2], -1)
        })
        new_local = set(self._local_support[target]) - before_local
        records: dict[tuple[float, ...], dict[str, Any]] = {}
        total_corridors = 0
        own_corridors = 0
        for key in keys:
            expert = self._expert_registry.get(key)
            if expert is None:
                continue
            source = "local" if key[:2] == own_lineage else "pulled"
            if source == "local":
                own_corridors += len(expert.capsules)
            total_corridors += len(expert.capsules)
            provider_is_new = source == "pulled" and int(key[0]) in provider_updates
            for raw in expert.capsules:
                row = [float(value) for value in raw]
                signature = tuple(round(value, 4) for value in row)
                is_new = provider_is_new or (source == "local" and raw in new_local)
                previous = records.get(signature)
                if previous is None or is_new or (
                    source == "local" and previous["source"] != "local"
                ):
                    records[signature] = {
                        "row": row,
                        "source": source,
                        "new": bool(is_new),
                    }
        profile = self._bank_profile(keys)
        coverage = float(np.mean(profile >= 0.5)) if profile.size else 0.0
        active = bool(self._current_node_active[target])
        node = self.nodes[target].node
        self.capture_frames.append({
            "step": step,
            "active": active,
            "position": [float(node.x), float(node.y)],
            "bank_size": len(keys),
            "corridor_count": int(total_corridors),
            "visible_corridor_count": len(records),
            "own_corridor_count": int(own_corridors),
            "coverage": coverage,
            "local_samples": int(local_samples),
            "provider_updates": provider_updates,
            "planes": list(records.values()),
        })
        return messages

    def _save_checkpoint(self, step: int) -> None:
        runner.atomic_json(
            Path(self.cfg.results_dir) / "checkpoint_status.json",
            {
                "format": "place_wallis_vehicle_corridor_visualization_v1",
                "checkpoint_kind": "metrics-only",
                "step": int(step),
                "target_node": int(CAPTURE["target_node"]),
                "captured_frames": len(self.capture_frames),
            },
        )


def draw_video(
    *,
    frames: list[dict[str, Any]],
    trace_path: Path,
    background_path: Path,
    output: Path,
    poster: Path,
    fps: int,
    target_node: int,
    entry_step: int,
) -> dict[str, Any]:
    cutoff, plateau_step = frame_cutoff(frames, entry_step=entry_step)
    frames = frames[: cutoff + 1]
    with np.load(trace_path, allow_pickle=False) as archive:
        node_states = archive["node_states"]
        node_active = archive["node_active"]

    base = Image.open(background_path).convert("RGBA")
    map_width, map_height = base.size
    if map_width != map_height:
        raise ValueError("Place Wallis background must be square")
    scale = map_width / 300.0
    panel_width = 320
    output_size = (map_width + panel_width, map_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".mp4", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    command = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{output_size[0]}x{output_size[1]}",
        "-framerate", "1", "-i", "-", "-an",
        "-vf", f"fps={int(fps)}", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    title_font = font(26, bold=True)
    section_font = font(18, bold=True)
    body_font = font(16)
    small_font = font(14)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    trail: list[tuple[float, float]] = []
    rendered_last: Image.Image | None = None
    coverage_values = [float(row["coverage"]) for row in frames]
    corridor_values = [int(row["visible_corridor_count"]) for row in frames]
    max_corridors = max(corridor_values, default=1)
    try:
        for frame_index, row in enumerate(frames):
            step = int(row["step"])
            canvas = Image.new("RGBA", output_size, "#f7f5ef")
            canvas.paste(base, (0, 0))
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            plane_draw = ImageDraw.Draw(overlay, "RGBA")
            for plane in row["planes"]:
                polygon = plane_polygon(
                    plane["row"], scale=scale, height=map_height
                )
                if len(polygon) != 4:
                    continue
                local = plane["source"] == "local"
                fill = (0, 88, 210, 72) if local else (132, 67, 205, 50)
                outline = (0, 58, 165, 150) if local else (94, 42, 160, 115)
                plane_draw.polygon(polygon, fill=fill, outline=outline)
                if plane["new"]:
                    highlight = (0, 210, 245, 245) if local else (245, 125, 20, 245)
                    plane_draw.line(polygon + [polygon[0]], fill=highlight, width=3)
            canvas.alpha_composite(overlay, (0, 0))
            draw = ImageDraw.Draw(canvas, "RGBA")

            active_indices = np.flatnonzero(node_active[step])
            for node_index in active_indices:
                x, y = node_states[step, node_index, :2]
                px, py = float(x) * scale, map_height - float(y) * scale
                radius = 2.4
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=(70, 74, 78, 150),
                )
            if bool(row["active"]):
                x, y = row["position"]
                point = (float(x) * scale, map_height - float(y) * scale)
                trail.append(point)
                trail = trail[-30:]
                if len(trail) > 1:
                    draw.line(trail, fill=(230, 82, 42, 150), width=3)
                px, py = point
                draw.ellipse((px - 11, py - 11, px + 11, py + 11), fill=(255, 255, 255, 230))
                draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(228, 74, 38, 255))
            else:
                trail.clear()

            panel_x = map_width
            draw.rectangle((panel_x, 0, output_size[0], map_height), fill=(248, 247, 243, 255))
            draw.line((panel_x, 0, panel_x, map_height), fill=(180, 178, 170, 255), width=2)
            x0 = panel_x + 22
            draw.text((x0, 24), "Floating radio map", fill="#22272b", font=title_font)
            draw.text((x0, 60), "K=6 support expert bank", fill="#4d555b", font=body_font)
            draw.text((x0, 96), clock_string(step), fill="#22272b", font=section_font)
            draw.text((x0, 124), f"Simulation step {step}", fill="#596168", font=small_font)

            y0 = 174
            metrics = (
                ("Vehicle", f"slot {target_node}"),
                ("Bank", f"{row['bank_size']} / 6 experts"),
                ("Corridors carried", f"{row['corridor_count']:,}"),
                ("Visible corridors", f"{row['visible_corridor_count']:,}"),
                ("Own corridors", f"{row['own_corridor_count']:,}"),
                ("Street-link coverage", f"{100.0 * float(row['coverage']):.1f}%"),
            )
            for label, value in metrics:
                draw.text((x0, y0), label, fill="#6a7075", font=small_font)
                draw.text((x0 + 150, y0 - 2), value, fill="#22272b", font=body_font)
                y0 += 38

            event_y = 425
            draw.text((x0, event_y), "This second", fill="#22272b", font=section_font)
            event_y += 34
            draw.text(
                (x0, event_y),
                f"{row['local_samples']} local link samples",
                fill="#164f9f", font=body_font,
            )
            event_y += 28
            updates = row["provider_updates"]
            update_text = (
                f"{len(updates)} pulled expert update{'s' if len(updates) != 1 else ''}"
            )
            draw.text((x0, event_y), update_text, fill="#008ca8", font=body_font)

            legend_y = 530
            draw.text((x0, legend_y), "Support", fill="#22272b", font=section_font)
            legend_y += 36
            draw.rectangle((x0, legend_y, x0 + 34, legend_y + 18), fill=(0, 88, 210, 150), outline="#003aa5")
            draw.text((x0 + 48, legend_y - 2), "Own observations", fill="#3f474d", font=body_font)
            legend_y += 38
            draw.rectangle((x0, legend_y, x0 + 34, legend_y + 18), fill=(132, 67, 205, 135), outline="#5e2aa0")
            draw.text((x0 + 48, legend_y - 2), "Pulled experts", fill="#3f474d", font=body_font)
            legend_y += 38
            draw.line((x0, legend_y + 5, x0 + 34, legend_y + 5), fill="#00d2f5", width=3)
            draw.line((x0, legend_y + 13, x0 + 34, legend_y + 13), fill="#f57d14", width=3)
            draw.text((x0 + 48, legend_y - 2), "New local / pulled", fill="#3f474d", font=body_font)

            plot_left, plot_top = x0, 700
            plot_right, plot_bottom = output_size[0] - 22, 820
            draw.rounded_rectangle(
                (plot_left, plot_top, plot_right, plot_bottom), radius=8,
                fill=(255, 255, 255, 230), outline=(205, 203, 197, 255),
            )
            draw.text((plot_left + 10, plot_top + 8), "Coverage growth", fill="#4c545a", font=small_font)
            if frame_index > 0:
                points = []
                for i, value in enumerate(coverage_values[: frame_index + 1]):
                    px = plot_left + 10 + i * (plot_right - plot_left - 20) / max(1, len(frames) - 1)
                    py = plot_bottom - 12 - value * (plot_bottom - plot_top - 42)
                    points.append((px, py))
                if len(points) > 1:
                    draw.line(points, fill="#1e78cf", width=3)
            progress = frame_index / max(1, len(frames) - 1)
            draw.rectangle((x0, 858, output_size[0] - 22, 866), fill="#d8d7d2")
            draw.rectangle((x0, 858, x0 + progress * (panel_width - 44), 866), fill="#2579c8")
            draw.text((x0, 875), "1 simulation second = 1 video second", fill="#747a7e", font=small_font)

            rgb = canvas.convert("RGB")
            process.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())
            rendered_last = rgb
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    os.replace(temporary, output)
    if rendered_last is not None:
        poster.parent.mkdir(parents=True, exist_ok=True)
        rendered_last.save(poster)
    return {
        "rendered_frames": len(frames),
        "video_duration_s": len(frames),
        "output_fps": int(fps),
        "plateau_step": plateau_step,
        "final_step": int(frames[-1]["step"]),
        "final_coverage": float(frames[-1]["coverage"]),
        "final_corridors": int(frames[-1]["corridor_count"]),
        "maximum_visible_corridors": int(max_corridors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--trace", type=Path, default=runner.DEFAULT_TRACE)
    parser.add_argument("--testset", type=Path, default=runner.DEFAULT_TESTSET)
    parser.add_argument("--net", type=Path, default=runner.DEFAULT_NET)
    parser.add_argument("--target-node", type=int, default=517)
    parser.add_argument("--capture-start", type=int, default=190)
    parser.add_argument("--capture-end", type=int, default=383)
    parser.add_argument("--entry-step", type=int, default=195)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--render-capture", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_capture is not None:
        capture_path = args.render_capture.resolve()
        with gzip.open(capture_path, "rt", encoding="utf-8") as stream:
            capture_payload = json.load(stream)
        frames = capture_payload["frames"]
        target_node = int(capture_payload["target_node"])
        entry_step = int(capture_payload["entry_step"])
        video_path = output_dir / "expert_corridor_growth_k6_vehicle517.mp4"
        poster_path = output_dir / "expert_corridor_growth_k6_vehicle517_poster.png"
        rendered = draw_video(
            frames=frames,
            trace_path=args.trace.resolve(),
            background_path=args.background.resolve(),
            output=video_path,
            poster=poster_path,
            fps=int(args.fps),
            target_node=target_node,
            entry_step=entry_step,
        )
        metadata = {
            "format": "place_wallis_expert_corridor_video_v1",
            "method": "updated support-driven expert bank K=6",
            "target_node": target_node,
            "entry_step": entry_step,
            "capture_start": int(capture_payload["capture_start"]),
            "capture_end": int(capture_payload["capture_end"]),
            "plateau_rule": (
                "stop five seconds after running-maximum street-link probe coverage "
                "improves by less than 0.01 over 30 seconds, after 45 seconds"
            ),
            "corridor_style": "transparent blue; dark blue local; cyan newly pulled",
            "video": str(video_path),
            "poster": str(poster_path),
            "capture": str(capture_path),
            **rendered,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[CORRIDOR-VIDEO] wrote {video_path} "
            f"duration={rendered['video_duration_s']}s "
            f"plateau_step={rendered['plateau_step']}",
            flush=True,
        )
        return 0

    if not 0 <= args.capture_start <= args.entry_step <= args.capture_end:
        raise ValueError("capture interval must contain the entry step")
    simulation_dir = output_dir / "simulation"
    CAPTURE.update({
        "target_node": int(args.target_node),
        "capture_start": int(args.capture_start),
        "capture_end": int(args.capture_end),
    })
    runner.SupportExpertBankSimulation = RecordingExpertBankSimulation
    original_argv = sys.argv
    sys.argv = [
        str(Path(__file__).name),
        "--trace", str(args.trace.resolve()),
        "--testset", str(args.testset.resolve()),
        "--net", str(args.net.resolve()),
        "--results-dir", str(simulation_dir),
        "--sim-steps", str(int(args.capture_end)),
        "--bank-capacity", str(int(args.bank_capacity)),
        "--transfer-cost", "0",
        "--probe-count", "512",
        "--checkpoint-every", "10000",
        "--progress-every", "25",
        "--tail-eval-count", "1",
        "--tail-eval-stride", "1",
        "--max-corridor-width-m", "12",
        "--link-length-margin-m", "0",
        "--quiet",
    ]
    try:
        runner.main()
    finally:
        sys.argv = original_argv
    simulation = RecordingExpertBankSimulation.latest_instance
    if simulation is None or not simulation.capture_frames:
        raise RuntimeError("visualization replay produced no captured frames")

    capture_path = output_dir / "vehicle517_corridors.json.gz"
    capture_payload = {
        "format": "place_wallis_vehicle_corridor_capture_v1",
        "target_node": int(args.target_node),
        "bank_capacity": int(args.bank_capacity),
        "capture_start": int(args.capture_start),
        "capture_end": int(args.capture_end),
        "entry_step": int(args.entry_step),
        "frames": simulation.capture_frames,
    }
    with gzip.open(capture_path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(capture_payload, stream, separators=(",", ":"))

    if args.capture_only:
        print(
            f"[CORRIDOR-CAPTURE] wrote {capture_path} frames={len(simulation.capture_frames)}",
            flush=True,
        )
        return 0
    video_path = output_dir / "expert_corridor_growth_k6_vehicle517.mp4"
    poster_path = output_dir / "expert_corridor_growth_k6_vehicle517_poster.png"
    rendered = draw_video(
        frames=simulation.capture_frames,
        trace_path=args.trace.resolve(),
        background_path=args.background.resolve(),
        output=video_path,
        poster=poster_path,
        fps=int(args.fps),
        target_node=int(args.target_node),
        entry_step=int(args.entry_step),
    )
    metadata = {
        "format": "place_wallis_expert_corridor_video_v1",
        "method": "updated support-driven expert bank K=6",
        "target_node": int(args.target_node),
        "entry_step": int(args.entry_step),
        "capture_start": int(args.capture_start),
        "capture_end": int(args.capture_end),
        "plateau_rule": (
            "stop five seconds after running-maximum street-link probe coverage "
            "improves by less than 0.01 over 30 seconds, after 45 seconds"
        ),
        "corridor_style": "transparent blue; dark blue local; cyan newly pulled",
        "video": str(video_path),
        "poster": str(poster_path),
        "capture": str(capture_path),
        **rendered,
    }
    runner.atomic_json(output_dir / "metadata.json", metadata)
    print(
        f"[CORRIDOR-VIDEO] wrote {video_path} "
        f"duration={rendered['video_duration_s']}s "
        f"plateau_step={rendered['plateau_step']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
