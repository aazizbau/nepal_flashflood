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
viewer and Planet PNG overlays are local and work without internet access.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import datadir as pyproj_datadir

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

import rasterio  # noqa: E402  (PROJ environment must be configured first)
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
        default=8192,
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
    # Start on the shared footprint instead of the union of both orbital
    # swaths. A small inset keeps diagonal acquisition edges outside the
    # initial viewport without rotating or corrupting north-up geometry.
    all_south = max(pre_bounds[0][0], post_bounds[0][0])
    all_west = max(pre_bounds[0][1], post_bounds[0][1])
    all_north = min(pre_bounds[1][0], post_bounds[1][0])
    all_east = min(pre_bounds[1][1], post_bounds[1][1])
    if all_south >= all_north or all_west >= all_east:
        raise ValueError("Pre- and post-event mosaics do not overlap")
    inset = 0.15
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
user-select:none;-webkit-user-drag:none}
#post-layer{clip-path:inset(0 50% 0 0)}
.title{position:absolute;z-index:5;top:12px;left:50%;transform:translateX(-50%);
background:#fffffff0;padding:8px 14px;border-radius:6px;box-shadow:0 1px 5px #0006;
font-weight:650;text-align:center;pointer-events:none;white-space:nowrap}
.labels{position:absolute;z-index:5;bottom:22px;left:0;right:0;display:flex;
justify-content:space-between;padding:0 24px;pointer-events:none}
.labels span{background:#111d;color:white;padding:6px 10px;border-radius:4px}
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
const viewer=document.getElementById('viewer'),post=document.getElementById('post-layer');
const divider=document.getElementById('divider'),scenes=document.querySelectorAll('.scene');
let split=.5,scale=1,tx=0,ty=0,mode=null,lastX=0,lastY=0;
function render(){post.style.clipPath=`inset(0 ${(1-split)*100}% 0 0)`;
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
