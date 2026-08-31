#!/usr/bin/env python
r"""Download OpenStreetMap buildings and infrastructure for the project AOI.

This command queries the Overpass API through OSMnx and writes categorized
vector features to one GeoPackage. It uses the same default WGS84 bounding box
as the remote-sensing downloaders:

    bbox = [85.08, 27.88, 85.62, 28.40]
    # [xmin, ymin, xmax, ymax]

The following layers are requested independently so a failed/empty category is
reported clearly and can be retried without constructing one oversized query:

* ``buildings``: all ``building=*`` objects;
* ``transport``: roads/paths, railways, and aeroways;
* ``crossings``: bridges and tunnels;
* ``utilities``: power, pipelines, telecom, water/wastewater works, towers;
* ``water_infrastructure``: waterways, dams, dykes, embankments, reservoirs;
* ``public_services``: healthcare, emergency, education, police, fire,
  shelters, government/community facilities, markets, bus stations, and fuel.

Outputs
-------
``osm_infrastructure.gpkg``
    One layer per non-empty category. Original OSM element type and ID are
    retained as ``element_type`` and ``osmid``.
``osm_summary.csv``
    Feature counts and geometry-type counts by category.
``osm_metadata.json``
    AOI, retrieval time, queries, source, license, attribution, and results.

OpenStreetMap completeness and positional accuracy vary. An absent feature does
not prove that infrastructure is absent on the ground. Validate critical
buildings, roads, and bridges with imagery and authoritative/local sources.

License and attribution
-----------------------
OpenStreetMap data is © OpenStreetMap contributors and available under the Open
Database License (ODbL). Preserve attribution and review share-alike obligations
when publishing or redistributing the downloaded database or derivatives:
https://www.openstreetmap.org/copyright

Examples (PowerShell)
---------------------
Download every category for the embedded bbox:

    python scripts/download_osm_infrastructure.py `
      --output data/external/osm

Use the shared AOI GeoJSON and download selected categories:

    python scripts/download_osm_infrastructure.py `
      --aoi configs/aoi_event.example.geojson `
      --categories buildings transport crossings public_services `
      --output data/external/osm

Use ``--overwrite`` to replace an existing GeoPackage. Public Overpass servers
are shared community infrastructure: avoid rapid repeated requests, retain the
OSMnx cache, and use regional extracts for substantially larger areas.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
import pyogrio
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from shapely.geometry import box, mapping, shape

LOGGER = logging.getLogger("osm-infrastructure")

DEFAULT_BBOX = (85.08, 27.88, 85.62, 28.40)  # xmin, ymin, xmax, ymax
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
OSM_ATTRIBUTION = "© OpenStreetMap contributors"

CATEGORY_TAGS: dict[str, dict[str, bool | str | list[str]]] = {
    "buildings": {"building": True},
    "transport": {
        "highway": True,
        "railway": True,
        "aeroway": True,
    },
    "crossings": {
        "bridge": True,
        "tunnel": True,
    },
    "utilities": {
        "power": True,
        "pipeline": True,
        "telecom": True,
        "man_made": [
            "pipeline",
            "water_works",
            "wastewater_plant",
            "communications_tower",
            "mast",
            "tower",
            "water_tower",
        ],
    },
    "water_infrastructure": {
        "waterway": True,
        "water": ["reservoir", "basin"],
        "man_made": ["dam", "dyke", "embankment", "reservoir_covered"],
    },
    "public_services": {
        "amenity": [
            "hospital",
            "clinic",
            "doctors",
            "pharmacy",
            "school",
            "college",
            "university",
            "police",
            "fire_station",
            "shelter",
            "community_centre",
            "townhall",
            "marketplace",
            "bus_station",
            "fuel",
        ],
        "healthcare": True,
        "emergency": True,
        "government": True,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OSM buildings and infrastructure into a GeoPackage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aoi", type=Path, help="AOI GeoJSON in EPSG:4326")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_TAGS),
        default=list(CATEGORY_TAGS),
        help="Infrastructure categories to download",
    )
    parser.add_argument("--output", type=Path, default=Path("data/external/osm"))
    parser.add_argument("--filename", default="osm_infrastructure.gpkg")
    parser.add_argument("--timeout", type=int, default=300, help="Overpass request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per category after transient failures")
    parser.add_argument("--retry-delay", type=float, default=15.0, help="Initial retry delay in seconds")
    parser.add_argument("--no-clip", action="store_true", help="Keep complete geometries extending beyond AOI")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the existing GeoPackage and skip layers already completed",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries <= 0 or args.retry_delay < 0:
        parser.error("--timeout and --retries must be positive; --retry-delay cannot be negative")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if Path(args.filename).name != args.filename or not args.filename.lower().endswith(".gpkg"):
        parser.error("--filename must be a .gpkg filename without directory components")
    return args


def default_aoi() -> dict[str, Any]:
    xmin, ymin, xmax, ymax = DEFAULT_BBOX
    return mapping(box(xmin, ymin, xmax, ymax))


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


def osm_text(value: Any) -> str | None:
    """Convert heterogeneous OSM tag values into GeoPackage-safe text."""
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
        return None
    return str(value)


def geopackage_field_names(columns: list[str], geometry_name: str) -> dict[str, str]:
    """Return collision-free field names for SQLite/GeoPackage.

    OSM tag keys are case-sensitive, but SQLite identifiers are not. Thus OSM
    columns such as ``FIXME`` and ``fixme`` cannot coexist unchanged in a
    GeoPackage layer. Suffix later collisions deterministically and avoid
    SQLite's conventional feature-ID field name.
    """
    used = {geometry_name.casefold()}
    renames: dict[str, str] = {}
    for original in columns:
        if original == geometry_name:
            continue
        base = "osm_fid" if original.casefold() == "fid" else original
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate.casefold())
        if candidate != original:
            renames[original] = candidate
    return renames


def prepare_layer(gdf: gpd.GeoDataFrame, aoi: dict[str, Any], clip: bool) -> gpd.GeoDataFrame:
    result = gdf.reset_index()
    if clip:
        result = gpd.clip(result, shape(aoi), keep_geom_type=False)
        result = result.loc[~result.geometry.is_empty & result.geometry.notna()].copy()
    renames = geopackage_field_names(list(result.columns), result.geometry.name)
    if renames:
        LOGGER.info("Renaming case-colliding/reserved fields for GeoPackage: %s", renames)
        result = result.rename(columns=renames)
    for column in result.columns:
        if column != result.geometry.name and result[column].dtype == "object":
            # OSM tags can contain strings, numbers, lists, or combinations in
            # the same column. Force one nullable text schema so GDAL does not
            # infer an incompatible field type from the first non-null value.
            result[column] = result[column].map(osm_text).astype("string")
    return result


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "status", "feature_count", "geometry_counts", "message"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download_category(
    bbox: tuple[float, float, float, float],
    tags: dict[str, bool | str | list[str]],
    retries: int,
    initial_delay: float,
) -> gpd.GeoDataFrame:
    """Query Overpass with bounded exponential retries for transient failures."""
    for attempt in range(1, retries + 1):
        try:
            return ox.features_from_bbox(bbox, tags=tags)
        except (requests.RequestException, ResponseStatusCodeError) as exc:
            if attempt == retries:
                raise
            delay = initial_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "Transient Overpass failure (%s/%s): %s. Retrying in %.0f seconds.",
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable retry state")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    aoi = load_aoi(args.aoi)
    bbox = tuple(shape(aoi).bounds)  # left, bottom, right, top
    args.output.mkdir(parents=True, exist_ok=True)
    geopackage = args.output / args.filename
    if geopackage.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"{geopackage} exists; use --resume or --overwrite")
    if geopackage.exists() and args.overwrite:
        geopackage.unlink()
        LOGGER.info("Removed existing output GeoPackage before rebuilding: %s", geopackage)

    ox.settings.requests_timeout = args.timeout
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(args.output / ".cache")
    ox.settings.log_console = args.verbose

    summaries: list[dict[str, Any]] = []
    completed_layers: list[str] = []
    existing_layers = (
        {str(row[0]) for row in pyogrio.list_layers(geopackage)}
        if geopackage.exists()
        else set()
    )
    for category in args.categories:
        if args.resume and category in existing_layers:
            feature_count = int(pyogrio.read_info(geopackage, layer=category)["features"])
            LOGGER.info("Skipping completed layer %s (%s features)", category, feature_count)
            summaries.append(
                {
                    "category": category,
                    "status": "existing",
                    "feature_count": feature_count,
                    "geometry_counts": "{}",
                    "message": "Preserved by --resume",
                }
            )
            completed_layers.append(category)
            continue
        tags = CATEGORY_TAGS[category]
        LOGGER.info("Downloading %s with tags %s", category, tags)
        try:
            raw = download_category(bbox, tags, args.retries, args.retry_delay)
            layer = prepare_layer(raw, aoi, clip=not args.no_clip)
            if layer.empty:
                raise InsufficientResponseError("query returned no features after AOI clipping")
            layer.to_file(geopackage, layer=category, driver="GPKG", index=False)
            geometry_counts = layer.geometry.geom_type.value_counts().sort_index().to_dict()
            summaries.append(
                {
                    "category": category,
                    "status": "success",
                    "feature_count": len(layer),
                    "geometry_counts": json.dumps(geometry_counts, sort_keys=True),
                    "message": "",
                }
            )
            completed_layers.append(category)
            LOGGER.info("Wrote %s features to layer %s", len(layer), category)
        except InsufficientResponseError as exc:
            LOGGER.warning("No %s features: %s", category, exc)
            summaries.append(
                {
                    "category": category,
                    "status": "empty",
                    "feature_count": 0,
                    "geometry_counts": "{}",
                    "message": str(exc),
                }
            )

    retrieved = datetime.now(timezone.utc).isoformat()
    metadata = {
        "retrieved_utc": retrieved,
        "source": "OpenStreetMap via the Overpass API and OSMnx",
        "attribution": OSM_ATTRIBUTION,
        "license": "Open Database License (ODbL)",
        "license_url": OSM_COPYRIGHT_URL,
        "aoi": aoi,
        "bbox_wgs84": list(bbox),
        "clipped_to_aoi": not args.no_clip,
        "requested_categories": args.categories,
        "completed_layers": completed_layers,
        "category_tags": {category: CATEGORY_TAGS[category] for category in args.categories},
        "summary": summaries,
        "limitations": [
            "OSM completeness, currency, classification, and positional accuracy vary.",
            "Absence from OSM does not demonstrate absence on the ground.",
            "Critical assets require imagery or authoritative-source validation.",
        ],
    }
    (args.output / "osm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.output / "osm_summary.csv", summaries)
    print(f"Completed layers: {', '.join(completed_layers) if completed_layers else 'none'}")
    print(f"GeoPackage: {geopackage}")
    print(f"Attribution: {OSM_ATTRIBUTION} ({OSM_COPYRIGHT_URL})")
    return 0 if completed_layers else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        sys.exit(130)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)
