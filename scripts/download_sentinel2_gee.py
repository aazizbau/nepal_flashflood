#!/usr/bin/env python
r"""Search and download a cloud-masked Sentinel-2 RGB median from Earth Engine.

This command mirrors ``download_landsat9_gee.py`` for the harmonized Sentinel-2
Level-2A surface-reflectance collection:

    COPERNICUS/S2_SR_HARMONIZED

It joins each surface-reflectance granule to
``COPERNICUS/S2_CLOUD_PROBABILITY`` by ``system:index``, masks pixels whose
cloud probability exceeds the chosen threshold, masks problematic Scene
Classification Layer (SCL) classes, removes invalid image edges, and computes a
per-pixel median. The output is a three-band RGB GeoTIFF:

* band 1 ``red``   = B4
* band 2 ``green`` = B3
* band 3 ``blue``  = B2

Sentinel-2 surface reflectance is stored as scaled integers. Output values use
the source convention:

    surface_reflectance = stored_value * 0.0001

The no-data value is -32768. The end date supplied to the CLI is inclusive.

AOI and download strategy
-------------------------
The embedded default AOI matches the Planet and Landsat downloaders:

    bbox = [85.08, 27.88, 85.62, 28.40]
    # [xmin, ymin, xmax, ymax], EPSG:4326

At 10 m this AOI is too large for one Earth Engine ``getDownloadURL`` request,
which is limited to 32 MB and 10,000 pixels per dimension. The script therefore
requests aligned tiles (2048 pixels by default) and writes them directly into a
single compressed GeoTIFF. Temporary tiles are removed after assembly.

Authentication
--------------
Authenticate Earth Engine once:

    earthengine authenticate

Supply a registered Google Cloud project with ``--project`` or set
``EE_PROJECT`` in the untracked ``.env`` file.

Examples (PowerShell)
---------------------
Search-only preview:

    python scripts/download_sentinel2_gee.py `
      --project YOUR_GOOGLE_CLOUD_PROJECT `
      --aoi configs/aoi_event.example.geojson `
      --start 2026-06-01 --end 2026-08-15 `
      --max-cloud 80 --cloud-probability 40 `
      --output data/raw/sentinel2/pre_event

Download and assemble the 10 m RGB median:

    python scripts/download_sentinel2_gee.py `
      --project YOUR_GOOGLE_CLOUD_PROJECT `
      --aoi configs/aoi_event.example.geojson `
      --start 2026-06-01 --end 2026-08-15 `
      --max-cloud 80 --cloud-probability 40 `
      --scale 10 --output data/raw/sentinel2/pre_event `
      --download

Running without ``--download`` only writes ``scenes.json`` and ``run.json``.
Always review acquisition dates, coverage, cloud, snow, terrain shadow, and
seams before using a median composite for change or damage analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


def configure_bundled_gdal_resources() -> None:
    """Prevent incompatible system PostGIS paths from overriding Rasterio.

    Windows installations of PostgreSQL/PostGIS commonly define global
    ``PROJ_LIB`` and ``GDAL_DATA`` variables. Those resources can be older than
    the GDAL/PROJ libraries bundled with the active Rasterio wheel, producing
    ``Cannot find proj.db`` or database-layout mismatch warnings. Resolve the
    installed Rasterio package without importing it, then select its matching
    resource directories before GDAL is loaded.
    """
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.submodule_search_locations:
        return
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    proj_data = package_dir / "proj_data"
    gdal_data = package_dir / "gdal_data"
    if (proj_data / "proj.db").is_file():
        os.environ["PROJ_DATA"] = str(proj_data)
        # PROJ_LIB is retained for compatibility with software that has not
        # migrated to the newer PROJ_DATA variable name.
        os.environ["PROJ_LIB"] = str(proj_data)
    if gdal_data.is_dir():
        os.environ["GDAL_DATA"] = str(gdal_data)
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")


configure_bundled_gdal_resources()

import ee
import rasterio
import requests
from dotenv import load_dotenv
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

LOGGER = logging.getLogger("sentinel2-gee")

SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
DEFAULT_BBOX = (85.08, 27.88, 85.62, 28.40)
OUTPUT_BANDS = ("red", "green", "blue")
REFLECTANCE_SCALE = 0.0001
NODATA = -32768


def default_aoi() -> dict[str, Any]:
    xmin, ymin, xmax, ymax = DEFAULT_BBOX
    return {
        "type": "Polygon",
        "coordinates": [
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
        ],
    }


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a cloud-masked Sentinel-2 RGB median from Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aoi", type=Path, help="AOI GeoJSON in EPSG:4326")
    parser.add_argument("--start", required=True, type=parse_iso_date)
    parser.add_argument("--end", required=True, type=parse_iso_date, help="Inclusive end date")
    parser.add_argument(
        "--project",
        default=os.getenv("EE_PROJECT"),
        help="Google Cloud project registered for Earth Engine; alternatively set EE_PROJECT",
    )
    parser.add_argument("--max-cloud", type=float, default=80.0, help="Maximum granule cloudy-pixel percent")
    parser.add_argument(
        "--cloud-probability",
        type=float,
        default=40.0,
        help="Mask pixels with s2cloudless probability at or above this percent",
    )
    parser.add_argument("--scale", type=float, default=10.0, help="Output pixel size in metres")
    parser.add_argument("--crs", default="EPSG:32645", help="Projected output CRS")
    parser.add_argument("--tile-size", type=int, default=2048, help="Download tile width/height in pixels")
    parser.add_argument("--output", type=Path, default=Path("data/raw/sentinel2/median"))
    parser.add_argument("--filename", default="sentinel2_rgb_median.tif")
    parser.add_argument("--download", action="store_true", help="Download the median; otherwise preview only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--authenticate", action="store_true", help="Launch interactive authentication first")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if not 0 <= args.max_cloud <= 100 or not 0 <= args.cloud_probability <= 100:
        parser.error("cloud thresholds must be between 0 and 100")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    # Three int16 bands at 2048 square are about 24 MiB before protocol
    # overhead, leaving room below Earth Engine's 32 MB direct-request limit.
    if not 128 <= args.tile_size <= 2048:
        parser.error("--tile-size must be between 128 and 2048")
    if not args.project:
        parser.error("provide --project or set EE_PROJECT in .env")
    if Path(args.filename).name != args.filename or not args.filename.lower().endswith((".tif", ".tiff")):
        parser.error("--filename must be a GeoTIFF filename without directory components")
    return args


def load_aoi(path: Path | None) -> dict[str, Any]:
    raw = default_aoi() if path is None else json.loads(path.read_text(encoding="utf-8"))
    if raw.get("type") == "Feature":
        raw = raw.get("geometry")
    elif raw.get("type") == "FeatureCollection":
        features = raw.get("features", [])
        if len(features) != 1:
            raise ValueError("AOI FeatureCollection must contain exactly one feature")
        raw = features[0].get("geometry")
    if not isinstance(raw, dict) or raw.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("AOI must resolve to a GeoJSON Polygon or MultiPolygon")
    geometry = shape(raw)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("AOI geometry is empty or invalid")
    minx, miny, maxx, maxy = geometry.bounds
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError("AOI coordinates must be longitude/latitude EPSG:4326")
    return mapping(geometry)


def initialize_earth_engine(project: str, authenticate: bool) -> None:
    if authenticate:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run 'earthengine authenticate', ensure "
            "the project is registered for Earth Engine, and pass its ID with --project."
        ) from exc


def build_collection(aoi: dict[str, Any], args: argparse.Namespace) -> ee.ImageCollection:
    region = ee.Geometry(aoi)
    exclusive_end = args.end + timedelta(days=1)
    filters = (
        ee.Filter.bounds(region)
        .And(ee.Filter.date(args.start.isoformat(), exclusive_end.isoformat()))
    )
    sr = (
        ee.ImageCollection(SR_COLLECTION)
        .filter(filters)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", args.max_cloud))
    )
    clouds = ee.ImageCollection(CLOUD_COLLECTION).filter(filters)
    joined = ee.Join.saveFirst("cloud_probability_image").apply(
        primary=sr,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined).filter(ee.Filter.notNull(["cloud_probability_image"]))


def mask_rgb(image: ee.Image, threshold: float) -> ee.Image:
    probability = ee.Image(image.get("cloud_probability_image")).select("probability")
    clear_probability = probability.lt(threshold)
    scl = image.select("SCL")
    # SCL: 0 no data, 1 saturated/defective, 3 shadow, 8/9 cloud,
    # 10 cirrus, and 11 snow/ice. Other classes, including water, are retained.
    clear_scl = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    edge_mask = image.select("B8A").mask().And(image.select("B9").mask())
    return (
        image.select(["B4", "B3", "B2"], list(OUTPUT_BANDS))
        .updateMask(clear_probability.And(clear_scl).And(edge_mask))
        .copyProperties(
            image,
            ["system:time_start", "PRODUCT_ID", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE"],
        )
    )


def scene_catalog(collection: ee.ImageCollection) -> list[dict[str, Any]]:
    info = ee.Dictionary(
        {
            "system_index": collection.aggregate_array("system:index"),
            "acquired_millis": collection.aggregate_array("system:time_start"),
            "product_id": collection.aggregate_array("PRODUCT_ID"),
            "mgrs_tile": collection.aggregate_array("MGRS_TILE"),
            "cloudy_pixel_percent": collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE"),
        }
    ).getInfo()
    scenes: list[dict[str, Any]] = []
    count = len(info.get("system_index", []))
    for index in range(count):
        millis = info["acquired_millis"][index]
        scene = {key: values[index] for key, values in info.items()}
        scene.pop("acquired_millis", None)
        scene["acquired_utc"] = (
            datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat() if millis else None
        )
        scenes.append(scene)
    return scenes


def print_scenes(scenes: list[dict[str, Any]]) -> None:
    print(f"Found {len(scenes)} joined Sentinel-2 scene(s):")
    for scene in scenes:
        print(
            f"  {scene.get('product_id') or scene.get('system_index')}  "
            f"{scene.get('acquired_utc')}  cloud={scene.get('cloudy_pixel_percent')}%  "
            f"tile={scene.get('mgrs_tile')}"
        )


def output_grid(
    aoi: dict[str, Any], crs: CRS, scale: float
) -> tuple[Any, int, int]:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected = shapely_transform(transformer.transform, shape(aoi))
    minx, miny, maxx, maxy = projected.bounds
    minx = math.floor(minx / scale) * scale
    miny = math.floor(miny / scale) * scale
    maxx = math.ceil(maxx / scale) * scale
    maxy = math.ceil(maxy / scale) * scale
    width = math.ceil((maxx - minx) / scale)
    height = math.ceil((maxy - miny) / scale)
    return from_origin(minx, maxy, scale, scale), width, height


def tile_windows(width: int, height: int, tile_size: int) -> Iterable[Window]:
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            yield Window(col, row, min(tile_size, width - col), min(tile_size, height - row))


def fetch_tile(image: ee.Image, crs: CRS, transform: Any, window: Window, path: Path) -> None:
    tile_transform = rasterio.windows.transform(window, transform)
    width, height = int(window.width), int(window.height)
    url = image.getDownloadURL(
        {
            "bands": list(OUTPUT_BANDS),
            "crs": crs.to_string(),
            "crs_transform": [
                tile_transform.a,
                tile_transform.b,
                tile_transform.c,
                tile_transform.d,
                tile_transform.e,
                tile_transform.f,
            ],
            "dimensions": [width, height],
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with path.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)


def download_median(
    collection: ee.ImageCollection,
    aoi: dict[str, Any],
    args: argparse.Namespace,
    destination: Path,
) -> tuple[int, int, int]:
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; use --overwrite to replace it")
    crs = CRS.from_user_input(args.crs)
    if not crs.is_projected:
        raise ValueError("--crs must be projected because --scale is in metres")
    transform, width, height = output_grid(aoi, crs, args.scale)
    median = (
        collection.map(lambda image: mask_rgb(ee.Image(image), args.cloud_probability))
        .median()
        .round()
        .toInt16()
        .clip(ee.Geometry(aoi))
        .unmask(NODATA, False)
    )
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "count": 3,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(destination.suffix + ".part.tif")
    temporary_tile = destination.with_suffix(destination.suffix + ".tile.tif")
    windows = list(tile_windows(width, height, args.tile_size))
    try:
        with rasterio.open(temporary_output, "w", **profile) as output:
            for band, name in enumerate(OUTPUT_BANDS, start=1):
                output.set_band_description(band, name)
            output.update_tags(
                dataset=SR_COLLECTION,
                reflectance_scale=REFLECTANCE_SCALE,
                reflectance_offset=0.0,
                cloud_probability_threshold=args.cloud_probability,
            )
            for number, window in enumerate(windows, start=1):
                LOGGER.info("Downloading tile %s/%s", number, len(windows))
                fetch_tile(median, crs, transform, window, temporary_tile)
                with rasterio.open(temporary_tile) as tile:
                    if tile.count != 3 or tile.width != window.width or tile.height != window.height:
                        raise RuntimeError(
                            f"Unexpected tile layout: bands={tile.count}, size={tile.width}x{tile.height}"
                        )
                    output.write(tile.read(), window=window)
                temporary_tile.unlink(missing_ok=True)
        temporary_output.replace(destination)
    except Exception:
        temporary_tile.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)
        raise
    return width, height, len(windows)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    aoi = load_aoi(args.aoi)
    initialize_earth_engine(args.project, args.authenticate)
    collection = build_collection(aoi, args)
    scenes = scene_catalog(collection)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print_scenes(scenes)
    if not scenes:
        LOGGER.warning("No matching joined Sentinel-2 scenes; no composite created")
        return 2

    run: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download" if args.download else "search-only",
        "surface_reflectance_dataset": SR_COLLECTION,
        "cloud_probability_dataset": CLOUD_COLLECTION,
        "aoi": aoi,
        "start": args.start.isoformat(),
        "end_inclusive": args.end.isoformat(),
        "max_granule_cloud_percent": args.max_cloud,
        "cloud_probability_threshold": args.cloud_probability,
        "scene_count": len(scenes),
        "bands": list(OUTPUT_BANDS),
        "output_crs": args.crs,
        "output_resolution_m": args.scale,
        "stored_value_scale": REFLECTANCE_SCALE,
        "stored_value_offset": 0.0,
        "nodata": NODATA,
    }
    if args.download:
        destination = args.output / args.filename
        width, height, tiles = download_median(collection, aoi, args, destination)
        run.update(output_file=str(destination), width=width, height=height, download_tiles=tiles)
        LOGGER.info("Wrote %s (%sx%s pixels, %s tiles)", destination, width, height, tiles)
    else:
        print("Search preview complete. Review scenes.json, then rerun with --download.")
    (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
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
