#!/usr/bin/env python3
"""Create a movable before/after map from georeferenced RGB GeoTIFFs.

Each input may be one GeoTIFF or a directory containing ``*_visual.tif`` tiles.
The script builds both RGB overlays on one shared north-up projected grid and
common extent, then writes a self-contained HTML viewer with a draggable swipe
divider, synchronized pan, and zoom. Source GeoTIFFs are read only and never
modified.

The PNGs are intentionally display products, not analysis rasters. Use the
original analytic surface-reflectance products for quantitative change
detection. Planet imagery licensing may restrict publication or redistribution;
check the applicable disaster-data terms before deploying the generated map.

Default PowerShell usage from the repository root::

    python scripts/create_planet_slider_map.py

Explicit paths and a larger display mosaic::

    python scripts/create_planet_slider_map.py `
      --pre-input assets/nepal_flashflood26.tif `
      --post-input data/processed/planet/planetscope_20260828_visual_mosaic.tif `
      --output outputs/maps/planet_before_after `
      --max-dimension 6000 `
      --title "Nepal flash flood: PlanetScope before/after"

Open ``outputs/maps/planet_before_after/index.html`` in a browser. The viewer
and Planet PNG overlays are local and work without internet access.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from pyproj import datadir as pyproj_datadir
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

# On Windows, rasterio and pyproj wheels can use different packaged PROJ data
# locations. Prefer rasterio's matching database, then fall back to pyproj's,
# before rasterio/GDAL opens a dataset or constructs an EPSG CRS.
_RASTERIO_SPEC = importlib.util.find_spec("rasterio")
_PROJ_CANDIDATES = []
if _RASTERIO_SPEC and _RASTERIO_SPEC.submodule_search_locations:
    _RASTERIO_DIR = Path(next(iter(_RASTERIO_SPEC.submodule_search_locations)))
    _PROJ_CANDIDATES.extend((_RASTERIO_DIR / "proj_data", _RASTERIO_DIR / "data"))
_PROJ_CANDIDATES.append(Path(pyproj_datadir.get_data_dir()))
_PROJ_DATA = next((path for path in _PROJ_CANDIDATES if (path / "proj.db").is_file()), None)
if _PROJ_DATA is None:
    searched = ", ".join(str(path) for path in _PROJ_CANDIDATES)
    raise RuntimeError(f"Could not find proj.db; searched: {searched}")
os.environ["PROJ_DATA"] = str(_PROJ_DATA)
os.environ["PROJ_LIB"] = str(_PROJ_DATA)  # compatibility with older PROJ/GDAL
os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

import geopandas as gpd  # noqa: E402  (PROJ environment must be configured first)
import rasterio  # noqa: E402
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

LOG = logging.getLogger("planet-slider")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-input",
        "--pre-dir",
        dest="pre_input",
        type=Path,
        default=Path("assets/nepal_flashflood26.tif"),
        help="Pre-event GeoTIFF, or directory searched recursively for *_visual.tif files.",
    )
    parser.add_argument(
        "--post-input",
        "--post-dir",
        dest="post_input",
        type=Path,
        default=Path("data/processed/planet/planetscope_20260828_visual_mosaic.tif"),
        help="Post-event GeoTIFF, or directory searched recursively for *_visual.tif files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/maps/planet_before_after"),
        help="Output folder for index.html and the two PNG overlays.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=12000,
        help="Maximum width or height of each display mosaic in pixels.",
    )
    parser.add_argument(
        "--title",
        default="Nepal-Tibet Flash Flood 26 August 2026",
        help="Map title shown in the browser.",
    )
    parser.add_argument("--pre-label", default="Pre-event")
    parser.add_argument("--post-label", default="Post-event · 28 August 2026")
    parser.add_argument(
        "--flood-polygon",
        type=Path,
        default=Path("assets/nepal_flooded_river.gpkg"),
        help="Flood polygon drawn as a thin yellow outline on the post-event side.",
    )
    parser.add_argument(
        "--glacier-lines",
        type=Path,
        default=Path("assets/nepal_flood_falling_glacier.gpkg"),
        help="GeoPackage containing width and length line layers.",
    )
    parser.add_argument(
        "--dsm",
        type=Path,
        default=Path("data/external/dem/aw3d30/aw3d30_v4_1_dsm_mosaic.tif"),
        help="DSM used to calculate terrain-following glacier length.",
    )
    parser.add_argument("--no-annotations", action="store_true", help="Do not add flood/glacier vectors.")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Reuse existing PNG overlays and rebuild only annotations and HTML.",
    )
    return parser.parse_args()


def discover(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Input file must be a GeoTIFF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Imagery input does not exist: {input_path}")
    paths = sorted(input_path.rglob("*_visual.tif"))
    if not paths:
        raise FileNotFoundError(f"No *_visual.tif files found below {input_path}")
    return paths


def make_overlay(paths: list[Path], destination: Path, max_dimension: int) -> list[list[float]]:
    if max_dimension < 256:
        raise ValueError("--max-dimension must be at least 256")

    sources = [rasterio.open(path) for path in paths]
    try:
        crs = sources[0].crs
        if crs is None:
            raise ValueError(f"Missing CRS: {paths[0]}")
        if any(src.crs != crs for src in sources):
            raise ValueError("All scenes for one date must use the same CRS")
        if any(src.count < 3 for src in sources):
            raise ValueError("Visual products must contain at least three RGB bands")

        left = min(src.bounds.left for src in sources)
        bottom = min(src.bounds.bottom for src in sources)
        right = max(src.bounds.right for src in sources)
        top = max(src.bounds.top for src in sources)
        xres = min(abs(src.res[0]) for src in sources)
        yres = min(abs(src.res[1]) for src in sources)
        native_width = (right - left) / xres
        native_height = (top - bottom) / yres
        scale = max(native_width / max_dimension, native_height / max_dimension, 1.0)

        LOG.info("Mosaicking %d scenes at %.2f x native display scale", len(paths), scale)
        mosaic, transform = merge(
            sources,
            indexes=[1, 2, 3],
            res=(xres * scale, yres * scale),
            nodata=0,
            method="first",
        )
    finally:
        for src in sources:
            src.close()

    height, width = mosaic.shape[1:]
    src_left, src_bottom, src_right, src_top = array_bounds(height, width, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs,
        "EPSG:4326",
        width,
        height,
        src_left,
        src_bottom,
        src_right,
        src_top,
    )
    rgb = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=mosaic[band],
            destination=rgb[band],
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )

    alpha = np.where(np.any(rgb != 0, axis=0), 255, 0).astype(np.uint8)
    rgba = np.dstack((np.moveaxis(rgb, 0, -1), alpha))
    Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)
    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    LOG.info("Wrote %s (%dx%d)", destination, dst_width, dst_height)
    return [[south, west], [north, east]]


def make_aligned_overlays(
    pre_paths: list[Path],
    post_paths: list[Path],
    pre_destination: Path,
    post_destination: Path,
    max_dimension: int,
    render_images: bool = True,
) -> tuple[list[list[float]], list[list[float]], dict]:
    """Render both dates on the same north-up projected pixel grid."""
    if max_dimension < 256:
        raise ValueError("--max-dimension must be at least 256")
    all_paths = pre_paths + post_paths
    datasets = [rasterio.open(path) for path in all_paths]
    warped_datasets: list[WarpedVRT] = []
    try:
        target_crs = datasets[0].crs
        if target_crs is None:
            raise ValueError(f"Missing CRS: {all_paths[0]}")
        if not target_crs.is_projected:
            raise ValueError("Inputs must use a projected CRS for undistorted top-down display")
        if any(dataset.crs is None for dataset in datasets):
            missing = all_paths[next(i for i, dataset in enumerate(datasets) if dataset.crs is None)]
            raise ValueError(f"Missing CRS: {missing}")
        if any(dataset.count < 3 for dataset in datasets):
            raise ValueError("All inputs must contain at least three RGB bands")

        aligned_datasets = []
        for path, dataset in zip(all_paths, datasets):
            if dataset.crs == target_crs:
                aligned_datasets.append(dataset)
            else:
                LOG.info("Reprojecting %s from %s to %s for display alignment", path, dataset.crs, target_crs)
                warped = WarpedVRT(
                    dataset,
                    crs=target_crs,
                    resampling=Resampling.bilinear,
                    nodata=0,
                )
                warped_datasets.append(warped)
                aligned_datasets.append(warped)

        pre_count = len(pre_paths)
        pre_sources = aligned_datasets[:pre_count]
        post_sources = aligned_datasets[pre_count:]

        def group_bounds(sources: list[rasterio.io.DatasetReader]) -> tuple[float, float, float, float]:
            return (
                min(source.bounds.left for source in sources),
                min(source.bounds.bottom for source in sources),
                max(source.bounds.right for source in sources),
                max(source.bounds.top for source in sources),
            )

        pre_extent = group_bounds(pre_sources)
        post_extent = group_bounds(post_sources)
        common_bounds = (
            max(pre_extent[0], post_extent[0]),
            max(pre_extent[1], post_extent[1]),
            min(pre_extent[2], post_extent[2]),
            min(pre_extent[3], post_extent[3]),
        )
        left, bottom, right, top = common_bounds
        if left >= right or bottom >= top:
            raise ValueError("Pre- and post-event inputs do not overlap")

        native_x = min(abs(source.res[0]) for source in aligned_datasets)
        native_y = min(abs(source.res[1]) for source in aligned_datasets)
        scale = max(
            (right - left) / native_x / max_dimension,
            (top - bottom) / native_y / max_dimension,
            1.0,
        )
        resolution = (native_x * scale, native_y * scale)
        aligned_left = math.floor(left / resolution[0]) * resolution[0]
        aligned_bottom = math.floor(bottom / resolution[1]) * resolution[1]
        aligned_right = math.ceil(right / resolution[0]) * resolution[0]
        aligned_top = math.ceil(top / resolution[1]) * resolution[1]
        output_width = int(round((aligned_right - aligned_left) / resolution[0]))
        output_height = int(round((aligned_top - aligned_bottom) / resolution[1]))
        output_transform = Affine(
            resolution[0], 0.0, aligned_left, 0.0, -resolution[1], aligned_top
        )
        LOG.info(
            "Rendering shared %s grid at %.3f x %.3f units/pixel (%.2f x native)",
            target_crs,
            resolution[0],
            resolution[1],
            scale,
        )

        if render_images:
            for label, sources, destination in (
                ("pre-event", pre_sources, pre_destination),
                ("post-event", post_sources, post_destination),
            ):
                mosaic, mosaic_transform = merge(
                    sources,
                    bounds=common_bounds,
                    indexes=[1, 2, 3],
                    res=resolution,
                    nodata=0,
                    method="first",
                    target_aligned_pixels=True,
                )
                height, width = mosaic.shape[1:]
                if (height, width) != (output_height, output_width) or not mosaic_transform.almost_equals(
                    output_transform
                ):
                    raise RuntimeError("Rendered grid differs from the planned aligned grid")
                alpha = np.where(np.any(mosaic != 0, axis=0), 255, 0).astype(np.uint8)
                rgba = np.dstack((np.moveaxis(mosaic.astype(np.uint8), 0, -1), alpha))
                Image.fromarray(rgba, mode="RGBA").save(destination, optimize=True)
                LOG.info("Wrote aligned %s overlay %s (%dx%d)", label, destination, width, height)
    finally:
        for dataset in warped_datasets:
            dataset.close()
        for dataset in datasets:
            dataset.close()

    normalized = [[0.0, 0.0], [1.0, 1.0]]
    grid = {
        "crs": target_crs,
        "transform": output_transform,
        "height": output_height,
        "width": output_width,
    }
    return normalized, normalized, grid


def geometry_to_svg_path(geometry, transform: Affine) -> str:
    """Convert projected vector geometry to output-grid SVG path commands."""
    inverse = ~transform

    def point(value) -> tuple[float, float]:
        x, y = inverse * (value[0], value[1])
        return float(x), float(y)

    def line_path(line: LineString, close: bool = False) -> str:
        coordinates = list(line.coords)
        if not coordinates:
            return ""
        pixels = [point(coordinate) for coordinate in coordinates]
        command = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pixels)
        return command + (" Z" if close else "")

    if geometry is None or geometry.is_empty:
        return ""
    if isinstance(geometry, LineString):
        return line_path(geometry)
    if isinstance(geometry, MultiLineString):
        return " ".join(line_path(part) for part in geometry.geoms)
    if isinstance(geometry, Polygon):
        paths = [line_path(LineString(geometry.exterior.coords), close=True)]
        paths.extend(line_path(LineString(ring.coords), close=True) for ring in geometry.interiors)
        return " ".join(paths)
    if isinstance(geometry, MultiPolygon):
        return " ".join(geometry_to_svg_path(part, transform) for part in geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return " ".join(geometry_to_svg_path(part, transform) for part in geometry.geoms)
    raise ValueError(f"Unsupported annotation geometry type: {geometry.geom_type}")


def line_parts(geometry) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def terrain_length_metres(geometry, geometry_crs, dsm: rasterio.io.DatasetReader) -> float:
    """Calculate polyline 3D length after sampling elevations along the DSM."""
    transformer = Transformer.from_crs(geometry_crs, dsm.crs, always_xy=True)
    projected = transform_geometry(transformer.transform, geometry)
    unit_factor = CRS.from_user_input(dsm.crs).axis_info[0].unit_conversion_factor
    step = max(abs(dsm.res[0]), abs(dsm.res[1]))
    total = 0.0
    for part in line_parts(projected):
        horizontal_length = part.length
        if horizontal_length <= 0:
            continue
        sample_count = max(2, math.ceil(horizontal_length / step) + 1)
        distances = np.linspace(0.0, horizontal_length, sample_count)
        points = [part.interpolate(float(distance)) for distance in distances]
        elevations = np.array([value[0] for value in dsm.sample([(p.x, p.y) for p in points])], dtype=float)
        valid = np.isfinite(elevations)
        if dsm.nodata is not None:
            valid &= elevations != dsm.nodata
        for index in range(1, sample_count):
            horizontal = (distances[index] - distances[index - 1]) * unit_factor
            if valid[index - 1] and valid[index]:
                vertical = elevations[index] - elevations[index - 1]
                total += math.hypot(horizontal, vertical)
            else:
                total += horizontal
    return total


def write_post_annotations(
    destination: Path,
    measurements_path: Path,
    grid: dict,
    flood_polygon_path: Path,
    glacier_lines_path: Path,
    dsm_path: Path,
) -> None:
    """Write a transparent SVG overlay and measurement provenance JSON."""
    for path in (flood_polygon_path, glacier_lines_path, dsm_path):
        if not path.is_file():
            raise FileNotFoundError(f"Annotation input does not exist: {path}")
    target_crs = grid["crs"]
    transform = grid["transform"]
    width, height = grid["width"], grid["height"]
    flood = gpd.read_file(flood_polygon_path).to_crs(target_crs)
    available_layers = gpd.list_layers(glacier_lines_path)["name"].tolist()
    if "width" in available_layers and "length" in available_layers:
        width_lines = gpd.read_file(glacier_lines_path, layer="width")
        length_lines = gpd.read_file(glacier_lines_path, layer="length")
    else:
        glacier = gpd.read_file(glacier_lines_path)
        category_column = next(
            (
                column
                for column in glacier.columns
                if column != glacier.geometry.name
                and {"width", "length"}.issubset(
                    set(glacier[column].dropna().astype(str).str.strip().str.casefold())
                )
            ),
            None,
        )
        if category_column is None:
            raise ValueError(
                "Glacier GeoPackage must have width/length layers or a field containing "
                f"width and length values. Available layers: {available_layers}"
            )
        categories = glacier[category_column].fillna("").astype(str).str.strip().str.casefold()
        width_lines = glacier[categories == "width"].copy()
        length_lines = glacier[categories == "length"].copy()
        LOG.info(
            "Using glacier layer %s and category field %s for width/length features",
            available_layers[0],
            category_column,
        )
    width_lines = width_lines.to_crs(target_crs)
    if flood.empty or width_lines.empty or length_lines.empty:
        raise ValueError("Flood polygon, width layer, and length layer must each contain features")

    unit_factor = CRS.from_user_input(target_crs).axis_info[0].unit_conversion_factor
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet">',
        '<g fill="none" stroke="#ffff00" stroke-width="2" vector-effect="non-scaling-stroke">',
    ]
    for geometry in flood.geometry:
        elements.append(f'<path d="{geometry_to_svg_path(geometry, transform)}"/>')
    elements.append("</g>")

    measurements = {"width": [], "terrain_following_length": []}
    label_size = max(22, round(max(width, height) / 260))
    with rasterio.open(dsm_path) as dsm:
        if dsm.crs is None:
            raise ValueError(f"DSM has no CRS: {dsm_path}")
        for layer_name, frame, color in (
            ("width", width_lines, "#00ffff"),
            ("length", length_lines.to_crs(target_crs), "#ff8c00"),
        ):
            elements.append(
                f'<g fill="none" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke">'
            )
            for feature_index, geometry in frame.geometry.items():
                if geometry is None or geometry.is_empty:
                    continue
                if layer_name == "width":
                    measured = geometry.length * unit_factor
                    key = "width"
                    method = "planar projected length"
                else:
                    original_geometry = length_lines.loc[feature_index].geometry
                    measured = terrain_length_metres(original_geometry, length_lines.crs, dsm)
                    key = "terrain_following_length"
                    method = "3D length from AW3D30 DSM samples"
                elements.append(f'<path d="{geometry_to_svg_path(geometry, transform)}"/>')
                midpoint = geometry.interpolate(0.5, normalized=True)
                column, row = (~transform) * (midpoint.x, midpoint.y)
                label = (
                    f'Width: {measured:,.1f} m'
                    if layer_name == "width"
                    else f'Slope length: {measured:,.1f} m'
                )
                elements.append(
                    f'<text x="{column:.2f}" y="{row:.2f}" fill="{color}" stroke="#000" '
                    f'stroke-width="4" paint-order="stroke" font-family="Arial,sans-serif" '
                    f'font-size="{label_size}" font-weight="700" text-anchor="middle">{label}</text>'
                )
                measurements[key].append(
                    {"feature_index": str(feature_index), "length_m": measured, "method": method}
                )
            elements.append("</g>")
    elements.append("</svg>")
    destination.write_text("\n".join(elements), encoding="utf-8")
    measurements.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "flood_polygon": str(flood_polygon_path),
            "glacier_lines": str(glacier_lines_path),
            "dsm": str(dsm_path),
            "display_crs": target_crs.to_string(),
        }
    )
    measurements_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    LOG.info("Wrote post-event vector overlay: %s", destination)
    LOG.info("Wrote vector measurements: %s", measurements_path)


def write_html(
    destination: Path,
    title: str,
    pre_label: str,
    post_label: str,
    pre_bounds: list[list[float]],
    post_bounds: list[list[float]],
    annotations: bool,
) -> None:
    # Start on the shared footprint instead of the union of both orbital
    # swaths. A small inset keeps diagonal acquisition edges outside the
    # initial viewport without rotating or corrupting north-up geometry.
    all_south = max(pre_bounds[0][0], post_bounds[0][0])
    all_west = max(pre_bounds[0][1], post_bounds[0][1])
    all_north = min(pre_bounds[1][0], post_bounds[1][0])
    all_east = min(pre_bounds[1][1], post_bounds[1][1])
    if all_south >= all_north or all_west >= all_east:
        raise ValueError("Pre- and post-event mosaics do not overlap")
    inset = 0.0
    latitude_padding = (all_north - all_south) * inset
    longitude_padding = (all_east - all_west) * inset
    all_south += latitude_padding
    all_north -= latitude_padding
    all_west += longitude_padding
    all_east -= longitude_padding
    values = {
        "title": title,
        "preLabel": pre_label,
        "postLabel": post_label,
        "preBounds": pre_bounds,
        "postBounds": post_bounds,
        "allBounds": [[all_south, all_west], [all_north, all_east]],
        "annotations": annotations,
    }
    config = json.dumps(values, ensure_ascii=False).replace("</", "<\\/")
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}html,body,#viewer{height:100%;margin:0}
body{font-family:system-ui,sans-serif;background:#182028;overflow:hidden}
#viewer{position:relative;background:linear-gradient(135deg,#25313b,#12181e);touch-action:none}
.layer{position:absolute;inset:0;overflow:hidden}.scene{position:absolute;inset:0;
transform-origin:0 0;will-change:transform}.layer img{position:absolute;display:block;
user-select:none;-webkit-user-drag:none;object-fit:contain}
#post-layer{clip-path:inset(0 0 0 50%)}
.title{position:absolute;z-index:5;top:14px;left:50%;transform:translateX(-50%);
background:#fffffff0;padding:8px 14px;border-radius:6px;box-shadow:0 1px 5px #0006;
font-size:clamp(22px,2vw,34px);font-weight:750;text-align:center;pointer-events:none;white-space:nowrap}
.labels{position:absolute;z-index:5;top:82px;left:0;right:0;display:grid;
grid-template-columns:1fr 1fr;pointer-events:none}
.labels span{justify-self:center;background:#111d;color:white;padding:8px 14px;border-radius:5px;
font-size:clamp(17px,1.35vw,23px);font-weight:700;box-shadow:0 1px 5px #0007}
.credits{position:absolute;z-index:5;right:16px;bottom:16px;background:#111d;color:#fff;
padding:9px 12px;border-radius:5px;font-size:14px;line-height:1.45;text-align:left;
box-shadow:0 1px 5px #0007;pointer-events:none}
#divider{position:absolute;z-index:4;top:0;bottom:0;left:50%;width:3px;
background:#fff;box-shadow:0 0 5px #000;pointer-events:none}
#divider::after{content:'↔';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
width:42px;height:42px;border-radius:50%;display:grid;place-items:center;background:#fff;
color:#222;font-size:22px;box-shadow:0 1px 7px #0008}
.hint{position:absolute;z-index:5;right:12px;top:12px;background:#111b;color:#fff;
padding:6px 9px;border-radius:4px;font-size:12px;pointer-events:none}
.controls{position:absolute;z-index:7;left:12px;top:12px;display:flex;flex-direction:column;
box-shadow:0 1px 5px #0007}.controls button{width:36px;height:36px;border:0;border-bottom:1px solid #bbb;
background:#fff;color:#222;font:bold 20px system-ui;cursor:pointer}.controls button:last-child{border:0;
font-size:13px}.controls button:hover{background:#eee}
</style></head><body><div id="viewer">
<div id="pre-layer" class="layer"><div class="scene"></div></div>
<div id="post-layer" class="layer"><div class="scene"></div></div>
<div id="divider"></div>
<div class="controls"><button id="zoom-in" title="Zoom in">+</button>
<button id="zoom-out" title="Zoom out">−</button><button id="reset" title="Reset view">1:1</button></div>
<div class="title"></div><div class="hint">Wheel: zoom · Drag: pan · Drag divider: compare</div>
<div class="labels"><span id="pre-label"></span><span id="post-label"></span></div>
<div class="credits">Developed by: Remote Sensing Nasahara Lab<br>
Data source: PlanetScope, OpenStreetMap</div>
</div><script>
const cfg=__CONFIG__;
document.querySelector('.title').textContent=cfg.title;
document.getElementById('pre-label').textContent=cfg.preLabel;
document.getElementById('post-label').textContent=cfg.postLabel;
function addRaster(layerId,src,bounds){
  const south=cfg.allBounds[0][0],west=cfg.allBounds[0][1];
  const north=cfg.allBounds[1][0],east=cfg.allBounds[1][1];
  const img=document.createElement('img');img.src=src;img.alt='';
  img.style.left=((bounds[0][1]-west)/(east-west)*100)+'%';
  img.style.top=((north-bounds[1][0])/(north-south)*100)+'%';
  img.style.width=((bounds[1][1]-bounds[0][1])/(east-west)*100)+'%';
  img.style.height=((bounds[1][0]-bounds[0][0])/(north-south)*100)+'%';
  document.querySelector(`#${layerId} .scene`).appendChild(img);
}
addRaster('pre-layer','pre_event.png',cfg.preBounds);
addRaster('post-layer','post_event.png',cfg.postBounds);
if(cfg.annotations)addRaster('post-layer','post_annotations.svg',cfg.postBounds);
const viewer=document.getElementById('viewer'),post=document.getElementById('post-layer');
const divider=document.getElementById('divider'),scenes=document.querySelectorAll('.scene');
let split=.5,scale=1,tx=0,ty=0,mode=null,lastX=0,lastY=0;
function render(){post.style.clipPath=`inset(0 0 0 ${split*100}%)`;
divider.style.left=(split*100)+'%';scenes.forEach(s=>s.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`)}
function zoomAt(factor,x=viewer.clientWidth/2,y=viewer.clientHeight/2){const next=Math.max(1,Math.min(64,scale*factor));
const ratio=next/scale;tx=x-(x-tx)*ratio;ty=y-(y-ty)*ratio;scale=next;if(scale===1){tx=0;ty=0}render()}
viewer.addEventListener('wheel',e=>{e.preventDefault();const r=viewer.getBoundingClientRect();
zoomAt(e.deltaY<0?1.25:.8,e.clientX-r.left,e.clientY-r.top)},{passive:false});
viewer.addEventListener('pointerdown',e=>{if(e.target.closest('.controls'))return;viewer.setPointerCapture(e.pointerId);
const r=viewer.getBoundingClientRect();mode=Math.abs(e.clientX-r.left-split*r.width)<30?'split':'pan';lastX=e.clientX;lastY=e.clientY});
viewer.addEventListener('pointermove',e=>{if(!mode)return;const r=viewer.getBoundingClientRect();
if(mode==='split')split=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));else{tx+=e.clientX-lastX;ty+=e.clientY-lastY}
lastX=e.clientX;lastY=e.clientY;render()});
viewer.addEventListener('pointerup',()=>mode=null);viewer.addEventListener('pointercancel',()=>mode=null);
document.getElementById('zoom-in').onclick=()=>zoomAt(1.5);document.getElementById('zoom-out').onclick=()=>zoomAt(2/3);
document.getElementById('reset').onclick=()=>{scale=1;tx=0;ty=0;render()};render();
</script></body></html>"""
    document = document.replace("__TITLE__", html.escape(title)).replace("__CONFIG__", config)
    destination.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Using PROJ data directory: %s", _PROJ_DATA)
    pre_paths = discover(args.pre_input)
    post_paths = discover(args.post_input)
    args.output.mkdir(parents=True, exist_ok=True)
    LOG.info("Found %d pre-event and %d post-event scenes", len(pre_paths), len(post_paths))
    if args.annotations_only:
        missing_overlays = [
            path
            for path in (args.output / "pre_event.png", args.output / "post_event.png")
            if not path.is_file()
        ]
        if missing_overlays:
            raise FileNotFoundError(
                "--annotations-only requires existing overlays: "
                + ", ".join(str(path) for path in missing_overlays)
            )
    pre_bounds, post_bounds, grid = make_aligned_overlays(
        pre_paths,
        post_paths,
        args.output / "pre_event.png",
        args.output / "post_event.png",
        args.max_dimension,
        render_images=not args.annotations_only,
    )
    annotation_path = args.output / "post_annotations.svg"
    measurements_path = args.output / "post_annotation_measurements.json"
    if args.no_annotations:
        annotation_path.unlink(missing_ok=True)
        measurements_path.unlink(missing_ok=True)
    else:
        write_post_annotations(
            annotation_path,
            measurements_path,
            grid,
            args.flood_polygon,
            args.glacier_lines,
            args.dsm,
        )
    write_html(
        args.output / "index.html",
        args.title,
        args.pre_label,
        args.post_label,
        pre_bounds,
        post_bounds,
        not args.no_annotations,
    )
    LOG.info("Slider map ready: %s", (args.output / "index.html").resolve())


if __name__ == "__main__":
    main()
