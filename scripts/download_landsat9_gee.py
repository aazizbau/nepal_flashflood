#!/usr/bin/env python
r"""Search and download a cloud-masked Landsat 9 RGB median from Earth Engine.

Dataset
-------
Google Earth Engine collection: ``LANDSAT/LC09/C02/T1_L2`` (USGS Landsat 9,
Collection 2, Tier 1, Level 2). The script selects surface-reflectance bands:

* Red: ``SR_B4``
* Green: ``SR_B3``
* Blue: ``SR_B2``

It applies the Collection 2 optical scale factor (0.0000275) and offset (-0.2),
masks fill, dilated cloud, cirrus, cloud, cloud shadow, snow, and radiometrically
saturated pixels, and then calculates a per-pixel median over the requested date
range. The end date is inclusive.

The downloaded GeoTIFF contains three bands named ``red``, ``green``, and
``blue``. Values are signed 16-bit scaled surface reflectance:

    surface_reflectance = stored_value * 0.0001

Masked pixels are retained as masked/no-data pixels by Earth Engine. Use the
generated ``run.json`` for provenance and scaling metadata.

AOI
---
The embedded default AOI matches the Planet downloader:

    bbox = [85.08, 27.88, 85.62, 28.40]
    # [xmin, ymin, xmax, ymax], EPSG:4326

Pass ``--aoi`` to use a GeoJSON Polygon/MultiPolygon, Feature, or a
single-feature FeatureCollection instead.

Authentication
--------------
Earth Engine requires an authorized Google account and a Google Cloud project
registered for Earth Engine. Authenticate once from PowerShell:

    earthengine authenticate

Then supply the Cloud project with ``--project`` or set ``EE_PROJECT`` in the
project's untracked ``.env`` file.

Examples (PowerShell)
---------------------
Search-only preview; no raster is downloaded:

    python scripts/download_landsat9_gee.py `
      --project YOUR_GOOGLE_CLOUD_PROJECT `
      --start 2026-06-01 --end 2026-08-15 `
      --max-cloud 80 `
      --output data/raw/landsat9/pre_event

Download the 30 m RGB median after reviewing the preview:

    python scripts/download_landsat9_gee.py `
      --project YOUR_GOOGLE_CLOUD_PROJECT `
      --start 2026-06-01 --end 2026-08-15 `
      --max-cloud 80 --scale 30 `
      --output data/raw/landsat9/pre_event `
      --download

Earth Engine's direct-download endpoint is intended for small images and is
limited to 32 MB and 10,000 pixels per grid dimension. This AOI's three-band,
30 m, int16 output is designed to remain within those constraints. For a much
larger AOI, use a batch Export workflow instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ee
import requests
from dotenv import load_dotenv
from shapely.geometry import mapping, shape

LOGGER = logging.getLogger("landsat9-gee")

COLLECTION_ID = "LANDSAT/LC09/C02/T1_L2"
DEFAULT_BBOX = (85.08, 27.88, 85.62, 28.40)  # xmin, ymin, xmax, ymax
REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2
OUTPUT_SCALE = 0.0001
OUTPUT_BANDS = ("red", "green", "blue")


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
        description="Download a cloud-masked Landsat 9 RGB median from Earth Engine.",
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
    parser.add_argument("--max-cloud", type=float, default=80.0, help="Maximum scene CLOUD_COVER percent")
    parser.add_argument("--scale", type=float, default=30.0, help="Output pixel size in metres")
    parser.add_argument("--crs", default="EPSG:32645", help="Projected output CRS")
    parser.add_argument("--output", type=Path, default=Path("data/raw/landsat9/median"))
    parser.add_argument("--filename", default="landsat9_rgb_median.tif")
    parser.add_argument("--download", action="store_true", help="Download the median; otherwise preview only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Launch Earth Engine's interactive authentication flow before initializing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if not 0 <= args.max_cloud <= 100:
        parser.error("--max-cloud must be between 0 and 100")
    if args.scale <= 0:
        parser.error("--scale must be positive")
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


def mask_and_scale(image: ee.Image) -> ee.Image:
    """Return scaled RGB reflectance with Collection 2 QA masking."""
    qa_pixel = image.select("QA_PIXEL")
    # Bits 0–5: fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
    clear = qa_pixel.bitwiseAnd(0b111111).eq(0)
    unsaturated = image.select("QA_RADSAT").eq(0)
    rgb = (
        image.select(["SR_B4", "SR_B3", "SR_B2"], list(OUTPUT_BANDS))
        .multiply(REFLECTANCE_SCALE)
        .add(REFLECTANCE_OFFSET)
    )
    return rgb.updateMask(clear.And(unsaturated)).copyProperties(
        image, ["system:time_start", "LANDSAT_PRODUCT_ID", "CLOUD_COVER"]
    )


def build_collection(aoi: dict[str, Any], args: argparse.Namespace) -> ee.ImageCollection:
    region = ee.Geometry(aoi)
    # Earth Engine filterDate uses an exclusive end, so add one day to make the
    # CLI's --end date inclusive.
    exclusive_end = args.end + timedelta(days=1)
    return (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(region)
        .filterDate(args.start.isoformat(), exclusive_end.isoformat())
        .filter(ee.Filter.lte("CLOUD_COVER", args.max_cloud))
        .sort("system:time_start")
    )


def scene_catalog(collection: ee.ImageCollection) -> list[dict[str, Any]]:
    info = ee.Dictionary(
        {
            "system_index": collection.aggregate_array("system:index"),
            "acquired_millis": collection.aggregate_array("system:time_start"),
            "landsat_product_id": collection.aggregate_array("LANDSAT_PRODUCT_ID"),
            "cloud_cover_percent": collection.aggregate_array("CLOUD_COVER"),
            "wrs_path": collection.aggregate_array("WRS_PATH"),
            "wrs_row": collection.aggregate_array("WRS_ROW"),
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
    print(f"Found {len(scenes)} Landsat 9 scene(s):")
    for scene in scenes:
        print(
            f"  {scene.get('landsat_product_id') or scene.get('system_index')}  "
            f"{scene.get('acquired_utc')}  cloud={scene.get('cloud_cover_percent')}%  "
            f"path/row={scene.get('wrs_path')}/{scene.get('wrs_row')}"
        )


def download_median(
    collection: ee.ImageCollection,
    aoi: dict[str, Any],
    args: argparse.Namespace,
    destination: Path,
) -> None:
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} exists; use --overwrite to replace it")
    # Preserve surface reflectance at 1e-4 precision while keeping the direct
    # request under Earth Engine's 32 MB limit for this AOI.
    median = (
        collection.map(mask_and_scale)
        .median()
        .multiply(1 / OUTPUT_SCALE)
        .round()
        .toInt16()
        .clip(ee.Geometry(aoi))
    )
    url = median.getDownloadURL(
        {
            "name": destination.stem,
            "bands": list(OUTPUT_BANDS),
            "region": aoi,
            "scale": args.scale,
            "crs": args.crs,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    LOGGER.info("Downloading RGB median to %s", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
        LOGGER.warning("No matching Landsat 9 scenes; no composite created")
        return 2

    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download" if args.download else "search-only",
        "dataset": COLLECTION_ID,
        "aoi": aoi,
        "start": args.start.isoformat(),
        "end_inclusive": args.end.isoformat(),
        "max_cloud_percent": args.max_cloud,
        "scene_count": len(scenes),
        "bands": list(OUTPUT_BANDS),
        "output_crs": args.crs,
        "output_resolution_m": args.scale,
        "stored_value_scale": OUTPUT_SCALE,
        "stored_value_offset": 0.0,
        "source_reflectance_scale": REFLECTANCE_SCALE,
        "source_reflectance_offset": REFLECTANCE_OFFSET,
    }
    if args.download:
        destination = args.output / args.filename
        download_median(collection, aoi, args, destination)
        run["output_file"] = str(destination)
        LOGGER.info("Download complete")
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
