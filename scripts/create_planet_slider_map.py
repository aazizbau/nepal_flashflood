#!/usr/bin/env python3
"""Create a movable before/after map from PlanetScope visual GeoTIFFs.

The script discovers every ``*_visual.tif`` below the pre- and post-event
directories, builds a reduced-resolution RGB mosaic for each date, reprojects
the mosaics to WGS 84, and writes PNG overlays plus a Leaflet HTML map with a
draggable swipe divider. Source GeoTIFFs are read only and never modified.

The PNGs are intentionally display products, not analysis rasters. Use the
original analytic surface-reflectance products for quantitative change
detection. Planet imagery licensing may restrict publication or redistribution;
check the applicable disaster-data terms before deploying the generated map.

Default PowerShell usage from the repository root::

    python scripts/create_planet_slider_map.py

Explicit paths and a larger display mosaic::

    python scripts/create_planet_slider_map.py `
      --pre-dir planet/pre_event/planetscope-2026-05-27 `
      --post-dir planet/post_event/planetscope-2026-08-26 `
      --output outputs/maps/planet_before_after `
      --max-dimension 6000 `
      --title "Nepal flash flood: PlanetScope before/after"

Open ``outputs/maps/planet_before_after/index.html`` in a browser. The page
needs internet access only for the Leaflet libraries and optional basemap; the
Planet PNG overlays are local files beside the HTML document.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

LOG = logging.getLogger("planet-slider")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-dir",
        type=Path,
        default=Path("planet/pre_event/planetscope-2026-05-27"),
        help="Directory searched recursively for pre-event *_visual.tif files.",
    )
    parser.add_argument(
        "--post-dir",
        type=Path,
        default=Path("planet/post_event/planetscope-2026-08-26"),
        help="Directory searched recursively for post-event *_visual.tif files.",
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
        default=4096,
        help="Maximum width or height of each display mosaic in pixels.",
    )
    parser.add_argument(
        "--title",
        default="Nepal flash flood: PlanetScope before/after",
        help="Map title shown in the browser.",
    )
    parser.add_argument("--pre-label", default="Pre-event · 27 May 2026")
    parser.add_argument("--post-label", default="Post-event · 26 August 2026")
    return parser.parse_args()


def discover(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Imagery directory does not exist: {directory}")
    paths = sorted(directory.rglob("*_visual.tif"))
    if not paths:
        raise FileNotFoundError(f"No *_visual.tif files found below {directory}")
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


def write_html(
    destination: Path,
    title: str,
    pre_label: str,
    post_label: str,
    pre_bounds: list[list[float]],
    post_bounds: list[list[float]],
) -> None:
    all_south = min(pre_bounds[0][0], post_bounds[0][0])
    all_west = min(pre_bounds[0][1], post_bounds[0][1])
    all_north = max(pre_bounds[1][0], post_bounds[1][0])
    all_east = max(pre_bounds[1][1], post_bounds[1][1])
    values = {
        "title": title,
        "preLabel": pre_label,
        "postLabel": post_label,
        "preBounds": pre_bounds,
        "postBounds": post_bounds,
        "allBounds": [[all_south, all_west], [all_north, all_east]],
    }
    config = json.dumps(values, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body,#map{{height:100%;margin:0}} body{{font-family:system-ui,sans-serif}}
.title{{position:absolute;z-index:1000;top:12px;left:50%;transform:translateX(-50%);
background:#fffffff0;padding:8px 14px;border-radius:6px;box-shadow:0 1px 5px #0006;
font-weight:650;text-align:center;pointer-events:none}}
.labels{{position:absolute;z-index:1000;bottom:22px;left:0;right:0;display:flex;
justify-content:space-between;padding:0 24px;pointer-events:none}}
.labels span{{background:#111c;color:white;padding:6px 10px;border-radius:4px}}
</style></head><body>
<div class="title"></div><div id="map"></div>
<div class="labels"><span id="pre-label"></span><span id="post-label"></span></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-side-by-side@2.2.0/leaflet-side-by-side.js"></script>
<script>
const cfg={config};
document.querySelector('.title').textContent=cfg.title;
document.getElementById('pre-label').textContent=cfg.preLabel;
document.getElementById('post-label').textContent=cfg.postLabel;
const map=L.map('map',{{zoomControl:true}});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
  maxZoom:19,attribution:'© OpenStreetMap contributors'
}}).addTo(map);
const pre=L.imageOverlay('pre_event.png',cfg.preBounds,{{opacity:1}}).addTo(map);
const post=L.imageOverlay('post_event.png',cfg.postBounds,{{opacity:1}}).addTo(map);
L.control.sideBySide(pre,post).addTo(map);
map.fitBounds(cfg.allBounds,{{padding:[12,12]}});
L.control.layers(null,{{[cfg.preLabel]:pre,[cfg.postLabel]:post}},{{collapsed:true}}).addTo(map);
</script></body></html>"""
    destination.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pre_paths = discover(args.pre_dir)
    post_paths = discover(args.post_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    LOG.info("Found %d pre-event and %d post-event scenes", len(pre_paths), len(post_paths))
    pre_bounds = make_overlay(pre_paths, args.output / "pre_event.png", args.max_dimension)
    post_bounds = make_overlay(post_paths, args.output / "post_event.png", args.max_dimension)
    write_html(
        args.output / "index.html",
        args.title,
        args.pre_label,
        args.post_label,
        pre_bounds,
        post_bounds,
    )
    LOG.info("Slider map ready: %s", (args.output / "index.html").resolve())


if __name__ == "__main__":
    main()
