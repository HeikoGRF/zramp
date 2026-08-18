#!/usr/bin/env python3
"""Render the 30-minute Bonnevoie right-middle traffic trace as an MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FREE_SPACE_COLOR = "#eeeae2"
BUILDING_FILL_COLOR = "#c9c3b9"
BUILDING_OUTLINE_COLOR = "#a9a39a"
WATER_FILL_COLOR = "#b9d8df"
WATER_OUTLINE_COLOR = "#9abfc8"
ROAD_EDGE_COLOR = "#3e4449"
ROAD_FILL_COLOR = "#656b70"
LEGEND_BACKGROUND_COLOR = "#f8f6f1"
LEGEND_TEXT_COLOR = "#34393d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--vehicle-types", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        default=(400.0, 200.0, 800.0, 600.0),
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--vehicle-color", default="#E4572E")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--poster-step", type=int, default=900)
    parser.add_argument(
        "--poster-legend-width",
        type=int,
        default=220,
        help="width of the right-hand legend panel in poster pixels (0 disables it)",
    )
    return parser.parse_args()


def load_dimensions(path: Path) -> dict[str, tuple[float, float]]:
    dimensions: dict[str, tuple[float, float]] = {}
    root = ET.parse(path).getroot()
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] != "vType":
            continue
        type_id = str(elem.get("id", ""))
        if not type_id:
            continue
        length = float(elem.get("length", "5.0"))
        width = float(elem.get("width", "1.8"))
        dimensions[type_id] = (length, width)
    return dimensions


def clock_string(seconds: float) -> str:
    rounded = int(round(seconds)) % (24 * 3600)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def vehicle_polygon(
    x: float,
    y: float,
    angle_deg: float,
    length_m: float,
    width_m: float,
    *,
    xmin: float,
    ymax: float,
    scale_x: float,
    scale_y: float,
) -> list[tuple[float, float]]:
    center_x = (x - xmin) * scale_x
    center_y = (ymax - y) * scale_y
    theta = math.radians(angle_deg)
    # SUMO headings are clockwise from north. Image y points downward.
    forward = (math.sin(theta), -math.cos(theta))
    right = (math.cos(theta), math.sin(theta))
    half_length = max(2.0, 0.5 * length_m * scale_x)
    half_width = max(1.25, 0.5 * width_m * scale_y)
    return [
        (
            center_x + sign_l * half_length * forward[0] + sign_w * half_width * right[0],
            center_y + sign_l * half_length * forward[1] + sign_w * half_width * right[1],
        )
        for sign_l, sign_w in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]


def render_frame(
    base: Image.Image,
    *,
    step: int,
    vehicle_ids: list[str],
    traces: dict[str, list[list[float] | None]],
    headings: dict[str, list[float | None]],
    vehicle_types_by_id: dict[str, str],
    dimensions: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    vehicle_color: str,
) -> tuple[Image.Image, int]:
    xmin, ymin, xmax, ymax = bounds
    scale_x = base.width / (xmax - xmin)
    scale_y = base.height / (ymax - ymin)
    frame = base.copy()
    draw = ImageDraw.Draw(frame)
    visible = 0
    for vehicle_id in vehicle_ids:
        point = traces[vehicle_id][step]
        if point is None:
            continue
        x, y = float(point[0]), float(point[1])
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        raw_heading = headings[vehicle_id][step]
        heading = 0.0 if raw_heading is None else float(raw_heading)
        type_id = vehicle_types_by_id.get(vehicle_id, "")
        length, width = dimensions.get(type_id, (5.0, 1.8))
        draw.polygon(
            vehicle_polygon(
                x,
                y,
                heading,
                length,
                width,
                xmin=xmin,
                ymax=ymax,
                scale_x=scale_x,
                scale_y=scale_y,
            ),
            fill=vehicle_color,
        )
        visible += 1
    return frame, visible


def load_legend_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")
        if bold
        else ("DejaVuSans.ttf",)
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_poster_legend(
    frame: Image.Image,
    *,
    vehicle_color: str,
    width: int,
) -> Image.Image:
    if width <= 0:
        return frame

    poster = Image.new(
        "RGB",
        (frame.width + width, frame.height),
        LEGEND_BACKGROUND_COLOR,
    )
    poster.paste(frame, (0, 0))
    draw = ImageDraw.Draw(poster)
    draw.line(
        (frame.width, 0, frame.width, frame.height),
        fill="#b8b3aa",
        width=2,
    )

    margin = max(18, width // 10)
    x = frame.width + margin
    title_font = load_legend_font(max(18, width // 10), bold=True)
    label_font = load_legend_font(max(14, width // 14))
    draw.text((x, 32), "Legend", fill=LEGEND_TEXT_COLOR, font=title_font)

    swatch = max(18, width // 10)
    gap = max(18, width // 11)
    y = 86
    items = (
        ("Vehicles", vehicle_color, "#b33d20"),
        ("Buildings", BUILDING_FILL_COLOR, BUILDING_OUTLINE_COLOR),
        ("Roads", ROAD_FILL_COLOR, ROAD_EDGE_COLOR),
        ("Open space", FREE_SPACE_COLOR, "#c9c4ba"),
    )
    for label, fill, outline in items:
        draw.rectangle(
            (x, y, x + swatch, y + swatch),
            fill=fill,
            outline=outline,
            width=2,
        )
        draw.text(
            (x + swatch + 12, y),
            label,
            fill=LEGEND_TEXT_COLOR,
            font=label_font,
        )
        y += swatch + gap
    return poster


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.poster_legend_width < 0:
        raise ValueError("--poster-legend-width must be non-negative")
    bounds = tuple(float(value) for value in args.bounds)
    xmin, ymin, xmax, ymax = bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("--bounds must have positive width and height")

    with args.mobility.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("format") != "sumo_crop_mobility_trace_v1":
        raise ValueError("unsupported mobility format")
    vehicle_ids = [str(value) for value in payload["vehicle_ids"]]
    traces = payload["traces"]
    headings = payload["heading_traces_deg"]
    vehicle_types_by_id = {
        str(key): str(value) for key, value in payload.get("vehicle_types", {}).items()
    }
    available_end = int(payload["max_step"])
    start_step = int(args.start_step)
    end_step = available_end if args.end_step is None else int(args.end_step)
    if not 0 <= start_step <= end_step <= available_end:
        raise ValueError("requested frame interval lies outside the mobility trace")
    frame_count = end_step - start_step + 1

    base = Image.open(args.background).convert("RGB")
    if base.width % 2 or base.height % 2:
        raise ValueError("H.264 yuv420p output requires even frame dimensions")
    dimensions = load_dimensions(args.vehicle_types)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".mp4", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{base.width}x{base.height}",
        "-framerate",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(args.preset),
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    visible_counts: list[int] = []
    poster_step = min(max(int(args.poster_step), start_step), end_step)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for output_index, step in enumerate(range(start_step, end_step + 1)):
            frame, visible = render_frame(
                base,
                step=step,
                vehicle_ids=vehicle_ids,
                traces=traces,
                headings=headings,
                vehicle_types_by_id=vehicle_types_by_id,
                dimensions=dimensions,
                bounds=bounds,
                vehicle_color=str(args.vehicle_color),
            )
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            visible_counts.append(visible)
            if args.poster is not None and step == poster_step:
                args.poster.parent.mkdir(parents=True, exist_ok=True)
                add_poster_legend(
                    frame,
                    vehicle_color=str(args.vehicle_color),
                    width=int(args.poster_legend_width),
                ).save(args.poster)
            if output_index % 300 == 0 or output_index + 1 == frame_count:
                print(
                    f"rendered {output_index + 1}/{frame_count} frames "
                    f"step={step} visible_vehicles={visible}",
                    flush=True,
                )
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        os.replace(temporary, args.output)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)

    metadata = {
        "format": "sumo_traffic_video_v1",
        "output": str(args.output.resolve()),
        "source_mobility": str(args.mobility.resolve()),
        "background": str(args.background.resolve()),
        "bounds_local_xy_m": list(bounds),
        "frame_size_px": [base.width, base.height],
        "source_start_step": start_step,
        "source_end_step": end_step,
        "source_frame_period_s": float(payload["sample_period_s"]),
        "source_start_time_s": float(payload["source_begin_s"]) + start_step,
        "source_end_time_s": float(payload["source_begin_s"]) + end_step,
        "source_start_clock": clock_string(float(payload["source_begin_s"]) + start_step),
        "source_end_clock": clock_string(float(payload["source_begin_s"]) + end_step),
        "video_fps": int(args.fps),
        "video_duration_s": frame_count / float(args.fps),
        "playback_speedup": float(args.fps) * float(payload["sample_period_s"]),
        "vehicle_color": str(args.vehicle_color),
        "vehicle_marker": "oriented SUMO length/width rectangle",
        "visible_vehicle_min_median_max": [
            int(np.min(visible_counts)),
            float(np.median(visible_counts)),
            int(np.max(visible_counts)),
        ],
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.output} ({frame_count} frames, {metadata['video_duration_s']:.1f}s, "
        f"{metadata['source_start_clock']}--{metadata['source_end_clock']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
