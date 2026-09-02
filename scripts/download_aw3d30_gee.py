#!/usr/bin/env python3
r"""Download one AW3D30 DSM mosaic for a GeoPackage AOI using Earth Engine.

Dataset
-------
Google Earth Engine collection ``JAXA/ALOS/AW3D30/V4_1`` contains the JAXA
ALOS World 3D 30 m global Digital Surface Model. This script mosaics the
collection's ``DSM`` band, clips it to the AOI, and writes one projected,
losslessly compressed GeoTIFF. Elevations are signed 16-bit integer metres
above sea level (EGM96 geoid), as supplied by AW3D30.

Earth Engine's direct-download endpoint has request-size limits. To support the
expanded flood AOI reliably, the script divides the final projected grid into
exactly aligned chunks, downloads them sequentially, and writes each chunk into
one destination GeoTIFF. Temporary chunk files are deleted automatically.

Requirements
------------
Authenticate Earth Engine once and provide a registered Cloud project::

    earthengine authenticate
    $env:EE_PROJECT = "your-google-cloud-project"

Default PowerShell command::

    python scripts/download_aw3d30_gee.py --download

Explicit example::

    python scripts/download_aw3d30_gee.py `
      --aoi assets/nepal_rasuwa_langtang_gyirong_flood_2026_expanded_AOI_single_layer.gpkg `
      --project YOUR_GOOGLE_CLOUD_PROJECT `
      --crs EPSG:32645 --scale 30 --tile-pixels 2048 `
      --output data/external/dem/aw3d30 `
      --filename aw3d30_v4_1_dsm_mosaic.tif `
      --download --overwrite

Without ``--download``, the command validates the AOI, initializes Earth
Engine, reports the planned mosaic dimensions/chunks, and writes ``run.json``.

Use and limitations
-------------------
AW3D30 is a DSM, so elevations can include vegetation and structures. Optical
stereo matching can retain errors near cloud, snow, and ice. Retain JAXA's
required attribution and review the AW3D30 terms before redistribution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ee
import requests
from dotenv import load_dotenv
from pyproj import Transformer, datadir as pyproj_datadir

# Windows geospatial wheels may not automatically agree on a PROJ data path.
# Prefer Rasterio's matching database, then fall back to pyproj's, before GDAL
# opens the AOI or creates the output CRS.
_RASTERIO_SPEC = importlib.util.find_spec("rasterio")
_PROJ_CANDIDATES: list[Path] = []
if _RASTERIO_SPEC and _RASTERIO_SPEC.submodule_search_locations:
    _RASTERIO_DIR = Path(next(iter(_RASTERIO_SPEC.submodule_search_locations)))
    _PROJ_CANDIDATES.extend((_RASTERIO_DIR / "proj_data", _RASTERIO_DIR / "data"))
_PROJ_CANDIDATES.append(Path(pyproj_datadir.get_data_dir()))
_PROJ_DATA = next((path for path in _PROJ_CANDIDATES if (path / "proj.db").is_file()), None)
if _PROJ_DATA is None:
    raise RuntimeError("Could not locate proj.db in the Rasterio or pyproj installation")
os.environ["PROJ_DATA"] = str(_PROJ_DATA)
os.environ["PROJ_LIB"] = str(_PROJ_DATA)
os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

LOGGER = logging.getLogger("aw3d30-gee")

COLLECTION_ID = "JAXA/ALOS/AW3D30/V4_1"
BAND = "DSM"
NODATA = -32768
DEFAULT_AOI = Path(
    "assets/nepal_rasuwa_langtang_gyirong_flood_2026_expanded_AOI_single_layer.gpkg"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one tiled-and-merged AW3D30 DSM GeoTIFF from Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI, help="Polygon AOI GeoPackage/vector file")
    parser.add_argument(
        "--project",
        default=os.getenv("EE_PROJECT"),
        help="Earth Engine-enabled Google Cloud project; alternatively set EE_PROJECT",
    )
    parser.add_argument("--crs", default="EPSG:32645", help="Projected output CRS")
    parser.add_argument("--scale", type=float, default=30.0, help="Output pixel size in CRS units")
    parser.add_argument(
        "--tile-pixels",
        type=int,
        default=2048,
        help="Maximum width and height of each Earth Engine download chunk",
    )
    parser.add_argument("--output", type=Path, default=Path("data/external/dem/aw3d30"))
    parser.add_argument("--filename", default="aw3d30_v4_1_dsm_mosaic.tif")
    parser.add_argument("--download", action="store_true", help="Download DSM; otherwise preview plan only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-tiles", action="store_true", help="Retain downloaded chunk GeoTIFFs")
    parser.add_argument("--authenticate", action="store_true", help="Launch Earth Engine authentication")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.project:
        parser.error("provide --project or set EE_PROJECT in .env")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if not 256 <= args.tile_pixels <= 8192:
        parser.error("--tile-pixels must be between 256 and 8192")
    if Path(args.filename).name != args.filename or not args.filename.lower().endswith((".tif", ".tiff")):
        parser.error("--filename must be a GeoTIFF filename without directory components")
    return args


def load_aoi(path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"AOI file does not exist: {path}")
    frame = gpd.read_file(path)
    if frame.empty or frame.crs is None:
        raise ValueError("AOI must contain geometry and have a defined CRS")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise ValueError("AOI contains no usable geometry")
    geometry = frame.to_crs("EPSG:4326").geometry.union_all()
    if geometry.is_empty or not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("AOI union is empty or invalid")
    return frame, mapping(geometry)


def initialize_earth_engine(project: str, authenticate: bool) -> None:
    if authenticate:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run 'earthengine authenticate' and "
            "ensure --project is registered for Earth Engine."
        ) from exc


def aligned_grid(frame: gpd.GeoDataFrame, crs: str, scale: float) -> tuple[Affine, int, int]:
    projected = frame.to_crs(crs)
    xmin, ymin, xmax, ymax = projected.total_bounds
    left = math.floor(xmin / scale) * scale
    bottom = math.floor(ymin / scale) * scale
    right = math.ceil(xmax / scale) * scale
    top = math.ceil(ymax / scale) * scale
    width = int(round((right - left) / scale))
    height = int(round((top - bottom) / scale))
    if width <= 0 or height <= 0:
        raise ValueError("AOI produces an empty output grid")
    return Affine(scale, 0, left, 0, -scale, top), width, height


def windows(width: int, height: int, tile_pixels: int):
    for row in range(0, height, tile_pixels):
        for col in range(0, width, tile_pixels):
            yield Window(col, row, min(tile_pixels, width - col), min(tile_pixels, height - row))


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    frame, aoi_wgs84 = load_aoi(args.aoi)
    transform, width, height = aligned_grid(frame, args.crs, args.scale)
    planned_windows = list(windows(width, height, args.tile_pixels))
    destination = args.output / args.filename
    LOGGER.info("Output grid: %d x %d pixels in %s at %.2f units", width, height, args.crs, args.scale)
    LOGGER.info("Planned Earth Engine chunks: %d", len(planned_windows))
    initialize_earth_engine(args.project, args.authenticate)

    collection = ee.ImageCollection(COLLECTION_ID).filterBounds(ee.Geometry(aoi_wgs84)).select(BAND)
    tile_count = int(collection.size().getInfo())
    if tile_count == 0:
        raise RuntimeError("No AW3D30 tiles intersect the AOI")
    source_projection = ee.Image(collection.first()).select(BAND).projection()
    image = (
        collection.mosaic()
        .setDefaultProjection(source_projection)
        .select(BAND)
        .clip(ee.Geometry(aoi_wgs84))
        .unmask(NODATA, False)
        .toInt16()
    )

    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download" if args.download else "preview",
        "dataset": COLLECTION_ID,
        "band": BAND,
        "product_type": "Digital Surface Model (DSM)",
        "elevation_units": "metres",
        "vertical_reference": "EGM96 geoid",
        "aoi_file": str(args.aoi),
        "aoi_wgs84": aoi_wgs84,
        "intersecting_source_tiles": tile_count,
        "output_file": str(destination),
        "output_crs": args.crs,
        "output_scale": args.scale,
        "width": width,
        "height": height,
        "download_chunks": len(planned_windows),
        "nodata": NODATA,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if not args.download:
        print("Preview complete. Review run.json, then rerun with --download.")
        return 0
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; use --overwrite to replace it")

    to_wgs84 = Transformer.from_crs(args.crs, "EPSG:4326", always_xy=True).transform
    tile_folder = args.output / "tiles"
    tile_folder.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "int16",
        "crs": args.crs,
        "transform": transform,
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
        "bigtiff": "IF_SAFER",
    }
    try:
        with rasterio.open(destination, "w", **profile) as output:
            output.update_tags(
                dataset=COLLECTION_ID,
                product="AW3D30 v4.1 DSM",
                source="JAXA ALOS PRISM via Google Earth Engine",
                units="metres",
                vertical_reference="EGM96 geoid",
            )
            for index, window in enumerate(planned_windows, start=1):
                window = window.round_offsets().round_lengths()
                tile_transform = rasterio.windows.transform(window, transform)
                west, south, east, north = rasterio.windows.bounds(window, transform)
                tile_region = mapping(transform_geometry(to_wgs84, box(west, south, east, north)))
                tile_path = tile_folder / f"aw3d30_{index:04d}.tif"
                LOGGER.info("Downloading chunk %d/%d", index, len(planned_windows))
                url = image.getDownloadURL(
                    {
                        "name": tile_path.stem,
                        "bands": [BAND],
                        "region": tile_region,
                        "crs": args.crs,
                        "crs_transform": list(tile_transform)[:6],
                        "dimensions": [int(window.width), int(window.height)],
                        "format": "GEO_TIFF",
                        "filePerBand": False,
                    }
                )
                download_file(url, tile_path)
                with rasterio.open(tile_path) as tile:
                    data = tile.read(1)
                    if data.shape != (int(window.height), int(window.width)):
                        raise RuntimeError(
                            f"Chunk {index} shape {data.shape} does not match planned window "
                            f"{int(window.height), int(window.width)}"
                        )
                    output.write(data.astype(np.int16, copy=False), 1, window=window)
                if not args.keep_tiles:
                    tile_path.unlink(missing_ok=True)

            factors = [factor for factor in (2, 4, 8, 16) if width // factor >= 1 and height // factor >= 1]
            if factors:
                output.build_overviews(factors, Resampling.average)
                output.update_tags(ns="rio_overview", resampling="average")
    except Exception:
        if not args.keep_tiles:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if not args.keep_tiles:
            try:
                tile_folder.rmdir()
            except OSError:
                pass

    LOGGER.info("Completed AW3D30 DSM mosaic: %s", destination)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        sys.exit(130)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)
