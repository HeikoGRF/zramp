#!/usr/bin/env python3
"""Render a square SUMO crop as a clean raster traffic-video background."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw


FREE_SPACE_COLOR = "#eeeae2"
BUILDING_FILL_COLOR = "#c9c3b9"
BUILDING_OUTLINE_COLOR = "#a9a39a"
WATER_FILL_COLOR = "#b9d8df"
WATER_OUTLINE_COLOR = "#9abfc8"
ROAD_EDGE_COLOR = "#3e4449"
ROAD_FILL_COLOR = "#656b70"

def parse_shape(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in raw.split():
        values = token.split(",")
        if len(values) < 2:
            raise ValueError(f"invalid SUMO shape point: {token!r}")
        points.append((float(values[0]), float(values[1])))
    return points


def intersects(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    xmin, ymin, xmax, ymax = bounds
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return not (
        max(x) < xmin - margin
        or min(x) > xmax + margin
        or max(y) < ymin - margin
        or min(y) > ymax + margin
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()

    bounds = tuple(float(value) for value in args.bounds)
    xmin, ymin, xmax, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bounds must have positive area")
    if int(args.size) <= 0 or int(args.size) % 2:
        raise ValueError("--size must be a positive even integer")
    scale_x = int(args.size) / (xmax - xmin)
    scale_y = int(args.size) / (ymax - ymin)

    def pixels(
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return [
            ((x - xmin) * scale_x, (ymax - y) * scale_y)
            for x, y in points
        ]

    image = Image.new(
        "RGB",
        (int(args.size), int(args.size)),
        FREE_SPACE_COLOR,
    )
    draw = ImageDraw.Draw(image)

    for _event, elem in ET.iterparse(args.polygons, events=("end",)):
        if elem.tag.rsplit("}", 1)[-1] != "poly":
            elem.clear()
            continue
        raw_shape = elem.get("shape", "")
        if not raw_shape:
            elem.clear()
            continue
        points = parse_shape(raw_shape)
        if not intersects(points, bounds):
            elem.clear()
            continue
        kind = str(elem.get("type", "")).lower()
        if "building" in kind:
            fill, outline = BUILDING_FILL_COLOR, BUILDING_OUTLINE_COLOR
        elif any(word in kind for word in ("water", "river", "canal")):
            fill, outline = WATER_FILL_COLOR, WATER_OUTLINE_COLOR
        elif any(word in kind for word in ("park", "green", "forest", "grass")):
            # Vegetation is not represented in the propagation geometry, so
            # render it like every other non-road, non-building free space.
            fill = outline = FREE_SPACE_COLOR
        else:
            elem.clear()
            continue
        draw.polygon(pixels(points), fill=fill, outline=outline)
        elem.clear()

    lanes: list[tuple[list[tuple[float, float]], float]] = []
    for _event, elem in ET.iterparse(args.net, events=("end",)):
        if elem.tag.rsplit("}", 1)[-1] != "lane":
            elem.clear()
            continue
        raw_shape = elem.get("shape", "")
        if not raw_shape:
            elem.clear()
            continue
        points = parse_shape(raw_shape)
        if intersects(points, bounds, margin=8.0):
            lanes.append((points, float(elem.get("width", "3.2"))))
        elem.clear()

    for points, width_m in lanes:
        draw.line(
            pixels(points),
            fill=ROAD_EDGE_COLOR,
            width=max(2, int(round(width_m * scale_x + 2.0))),
            joint="curve",
        )
    for points, width_m in lanes:
        draw.line(
            pixels(points),
            fill=ROAD_FILL_COLOR,
            width=max(1, int(round(width_m * scale_x))),
            joint="curve",
        )

    draw.rectangle(
        (0, 0, int(args.size) - 1, int(args.size) - 1),
        outline="#34393d",
        width=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, optimize=True)
    print(
        f"wrote {args.output} ({args.size}x{args.size}, "
        f"{len(lanes)} cropped lanes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
