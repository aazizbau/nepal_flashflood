#!/usr/bin/env python3
"""Create a separate Planet before/after slider with two length measurements.

This script preserves all imagery, alignment, zoom, pan, titles, and credits
from ``create_planet_slider_map.py`` and adds only the two line features stored
in ``assets/nepal_flood_falling_glacier.gpkg``:

* ``width``: direct horizontal line length in metres.
* ``length``: terrain-following 3D length in metres, calculated by sampling
  ``data/external/dem/aw3d30/aw3d30_v4_1_dsm_mosaic.tif`` along the line.

The GeoPackage currently stores both features in one layer and distinguishes
them with its ``name`` field. Separate ``width`` and ``length`` layers are also
supported. Measurements are saved in JSON and rendered as a scalable SVG on
the post-event side only.

PowerShell::

    python scripts/create_planet_slider_map_lengths.py
    start outputs\maps\planet_before_after_lengths\index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import create_planet_slider_map as base
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import transform as transform_geometry

LOG = logging.getLogger("planet-slider-lengths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input", type=Path, default=Path("assets/nepal_flashflood26.tif"))
    parser.add_argument(
        "--post-input",
        type=Path,
        default=Path("data/processed/planet/planetscope_20260828_visual_mosaic.tif"),
    )
    parser.add_argument(
        "--lines", type=Path, default=Path("assets/nepal_flood_falling_glacier.gpkg")
    )
    parser.add_argument(
        "--dsm",
        type=Path,
        default=Path("data/external/dem/aw3d30/aw3d30_v4_1_dsm_mosaic.tif"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/maps/planet_before_after_lengths")
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=16000,
        help="Maximum display raster dimension; higher values preserve more PlanetScope detail.",
    )
    parser.add_argument("--title", default="Nepal-Tibet Flash Flood 26 August 2026")
    parser.add_argument("--pre-label", default="Pre-event")
    parser.add_argument("--post-label", default="Post-event · 28 August 2026")
    return parser.parse_args()


def load_measurement_lines(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Measurement GeoPackage does not exist: {path}")
    layers = gpd.list_layers(path)["name"].tolist()
    if "width" in layers and "length" in layers:
        return gpd.read_file(path, layer="width"), gpd.read_file(path, layer="length")
    frame = gpd.read_file(path)
    category = next(
        (
            column
            for column in frame.columns
            if column != frame.geometry.name
            and {"width", "length"}.issubset(
                set(frame[column].dropna().astype(str).str.strip().str.casefold())
            )
        ),
        None,
    )
    if category is None:
        raise ValueError(f"Could not identify width/length features; available layers: {layers}")
    values = frame[category].fillna("").astype(str).str.strip().str.casefold()
    return frame[values == "width"].copy(), frame[values == "length"].copy()


def line_parts(geometry) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def slope_length(geometry, source_crs, dsm: rasterio.io.DatasetReader) -> float:
    transformer = Transformer.from_crs(source_crs, dsm.crs, always_xy=True)
    projected = transform_geometry(transformer.transform, geometry)
    unit_factor = CRS.from_user_input(dsm.crs).axis_info[0].unit_conversion_factor
    sampling_step = max(abs(dsm.res[0]), abs(dsm.res[1]))
    total = 0.0
    for part in line_parts(projected):
        if part.length <= 0:
            continue
        count = max(2, math.ceil(part.length / sampling_step) + 1)
        distances = np.linspace(0.0, part.length, count)
        points = [part.interpolate(float(distance)) for distance in distances]
        elevations = np.array([sample[0] for sample in dsm.sample([(p.x, p.y) for p in points])])
        valid = np.isfinite(elevations)
        if dsm.nodata is not None:
            valid &= elevations != dsm.nodata
        for index in range(1, count):
            horizontal = float(distances[index] - distances[index - 1]) * unit_factor
            vertical = float(elevations[index] - elevations[index - 1]) if valid[index - 1] and valid[index] else 0.0
            total += math.hypot(horizontal, vertical)
    return total


def svg_path(geometry, transform) -> str:
    inverse = ~transform
    commands = []
    for part in line_parts(geometry):
        pixels = [inverse * (coordinate[0], coordinate[1]) for coordinate in part.coords]
        if pixels:
            commands.append("M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pixels))
    return " ".join(commands)


def write_measurements(
    lines_path: Path, dsm_path: Path, grid: dict, output: Path
) -> tuple[float, float]:
    width_frame, length_frame = load_measurement_lines(lines_path)
    if width_frame.empty or length_frame.empty:
        raise ValueError("Both width and length features are required")
    target_crs, transform = grid["crs"], grid["transform"]
    width_projected = width_frame.to_crs(target_crs)
    length_projected = length_frame.to_crs(target_crs)
    map_unit = CRS.from_user_input(target_crs).axis_info[0].unit_conversion_factor
    records = {"width": [], "slope_length": []}
    display_dimension = max(grid["width"], grid["height"])
    font_size = max(160, round(display_dimension / 65))
    line_width = max(14, round(display_dimension / 800))
    halo_width = line_width + max(12, round(display_dimension / 900))
    text_halo = max(12, round(font_size / 12))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {grid["width"]} {grid["height"]}" preserveAspectRatio="xMidYMid meet">'
    ]
    label_positions: list[tuple[float, float]] = []
    with rasterio.open(dsm_path) as dsm:
        if dsm.crs is None:
            raise ValueError(f"DSM has no CRS: {dsm_path}")
        for kind, projected_frame, original_frame, color in (
            ("width", width_projected, width_frame, "#00ffff"),
            ("length", length_projected, length_frame, "#ff8c00"),
        ):
            svg.append(f'<g fill="none" stroke="{color}" stroke-width="{line_width}">')
            for index, geometry in projected_frame.geometry.items():
                if geometry is None or geometry.is_empty:
                    continue
                measured = (
                    geometry.length * map_unit
                    if kind == "width"
                    else slope_length(original_frame.loc[index].geometry, original_frame.crs, dsm)
                )
                label = f'{"Width" if kind == "width" else "Slope length"}: {measured:,.1f} m'
                midpoint = geometry.interpolate(0.5, normalized=True)
                x, y = (~transform) * (midpoint.x, midpoint.y)
                label_positions.append((float(x), float(y)))
                path = svg_path(geometry, transform)
                svg.append(
                    f'<path d="{path}" stroke="#000" stroke-width="{halo_width}" opacity="0.8"/>'
                )
                svg.append(f'<path d="{path}"/>')
                svg.append(
                    f'<text x="{x:.2f}" y="{y:.2f}" dy="-{font_size * 0.35:.1f}" fill="{color}" '
                    f'stroke="#000" stroke-width="{text_halo}" '
                    f'paint-order="stroke" font-family="Arial,sans-serif" font-size="{font_size}" '
                    f'font-weight="700" text-anchor="middle">{label}</text>'
                )
                records["width" if kind == "width" else "slope_length"].append(
                    {"feature_index": str(index), "length_m": float(measured)}
                )
            svg.append("</g>")
    svg.append("</svg>")
    (output / "measurements.svg").write_text("\n".join(svg), encoding="utf-8")
    records.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "lines": str(lines_path),
            "dsm": str(dsm_path),
            "width_method": "direct projected length",
            "length_method": "terrain-following 3D length sampled at DSM resolution",
        }
    )
    (output / "measurements.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if not label_positions:
        raise ValueError("No valid measurement lines were rendered")
    center_x = sum(position[0] for position in label_positions) / len(label_positions)
    center_y = sum(position[1] for position in label_positions) / len(label_positions)
    return center_x / grid["width"], center_y / grid["height"]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in (args.lines, args.dsm):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    pre_bounds, post_bounds, grid = base.make_aligned_overlays(
        base.discover(args.pre_input),
        base.discover(args.post_input),
        args.output / "pre_event.png",
        args.output / "post_event.png",
        args.max_dimension,
    )
    focus_x, focus_y = write_measurements(args.lines, args.dsm, grid, args.output)
    base.write_html(
        args.output / "index.html",
        args.title,
        args.pre_label,
        args.post_label,
        pre_bounds,
        post_bounds,
    )
    html_path = args.output / "index.html"
    document = html_path.read_text(encoding="utf-8")
    marker = "addRaster('post-layer','post_event.png',cfg.postBounds);"
    if marker not in document:
        raise RuntimeError("Could not find post-event insertion point in generated HTML")
    document = document.replace(marker, marker + "\naddRaster('post-layer','measurements.svg',cfg.postBounds);")
    document = document.replace(
        '<button id="reset" title="Reset view">1:1</button>',
        '<button id="focus-lines" title="Focus measurement lines" style="font-size:11px">Lines</button>'
        '<button id="reset" title="Full overview">1:1</button>',
    )
    focus_script = f"""
function focusMeasurements(){{
  scale=2;
  tx=viewer.clientWidth*0.72-({focus_x:.10f}*viewer.clientWidth*scale);
  ty=viewer.clientHeight/2-({focus_y:.10f}*viewer.clientHeight*scale);
  render();
}}
document.getElementById('focus-lines').onclick=focusMeasurements;
"""
    document = document.replace("</script></body></html>", focus_script + "</script></body></html>")
    html_path.write_text(document, encoding="utf-8")
    LOG.info("Measured slider map ready: %s", html_path.resolve())


if __name__ == "__main__":
    main()
