#!/usr/bin/env python
r"""Search, order, download, and median-composite PlanetScope scenes over an AOI.

Purpose
-------
This command adapts the simple ``Planet().data.search(...)`` pattern into a
reproducible workflow for the Nepal–Tibet flash-flood project. It:

1. reads an AOI from GeoJSON (or uses the example polygon below);
2. searches ``PSScene`` imagery acquired within an inclusive UTC date range;
3. filters by cloud cover, downloadable permissions, and the requested asset;
4. saves the search result metadata before any imagery order is submitted;
5. optionally submits a clipped, partial Planet Orders API order;
6. waits with retry-safe polling and downloads the order; and
7. aligns the downloaded rasters to one grid and calculates a block-wise,
   per-band, per-pixel median GeoTIFF.

Important behavior
------------------
* Running without ``--submit`` is a safe search/preview: no order is placed.
* ``--submit`` can consume Planet quota, processing units, and data egress.
* The default bundle is eight-band orthorectified surface reflectance plus
  UDM2 (``analytic_8b_sr_udm2``). Your Planet plan must include access.
* The UDM2 ``clear`` band (band 1) is used by default. Cloud, haze, shadow,
  snow, and unusable pixels therefore do not contribute to the median.
* Planet's Orders API ``composite`` tool is not used because it overlays
  scenes in sequence; it does not calculate a statistical median. Median
  compositing is performed locally after download.
* Search cloud cover describes the whole scene, not necessarily the AOI.
* A median is not a substitute for image co-registration or manual QA. Inspect
  footprints, acquisition times, residual cloud, snow, terrain shadow, seams,
  and radiometry before damage analysis.
* The embedded example AOI is intentionally broad. It can entail substantial
  quota and storage use at 3 m. Prefer a small, verified flood-corridor AOI.

Authentication
--------------
Set ``PL_API_KEY`` in the environment or in the project's untracked ``.env``
file. Planet SDK authentication previously configured on the machine may also
be used. Never put a real key in source code or commit ``.env``.

AOI formats
-----------
``--aoi`` accepts a GeoJSON Polygon/MultiPolygon, Feature, or a FeatureCollection
containing exactly one feature. Coordinates must be longitude/latitude WGS84
(EPSG:4326). If ``--aoi`` is omitted, this example polygon is used:

    84.9,28.0 ───────── 86.0,28.0
         │                    │
    84.9,29.1 ───────── 86.0,29.1

Example commands (PowerShell)
-----------------------------
Preview the supplied example AOI and a pre-event interval without ordering:

    python scripts/download_planet_median.py `
      --start 2026-08-01 --end 2026-08-25 `
      --max-cloud 20 --limit 30

Search a verified AOI file, submit the order, download it, and create a median:

    python scripts/download_planet_median.py `
      --aoi configs/aoi_event.geojson `
      --start 2026-08-01 --end 2026-08-25 `
      --max-cloud 20 --limit 30 `
      --resolution 3 --output data/raw/planet/pre_event `
      --submit

Create only a local median from an already downloaded order directory:

    python scripts/download_planet_median.py `
      --aoi configs/aoi_event.geojson `
      --start 2026-08-01 --end 2026-08-25 `
      --from-download data/raw/planet/pre_event/order `
      --output data/raw/planet/pre_event

Resume an order after a timeout or interrupted terminal (this never creates a
new order):

    python scripts/download_planet_median.py `
      --aoi configs/aoi_event.geojson `
      --start 2026-08-01 --end 2026-08-25 `
      --output data/raw/planet/pre_event `
      --resume-order YOUR_ORDER_ID

Equivalent Bash commands use ``\`` for line continuation. Run ``--help`` for
all options. The script writes ``search_results.geojson``, ``run.json``, the
downloaded order tree, ``median_composite.tif``, and
``median_observation_count.tif`` under the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time as clock
import warnings
from contextlib import ExitStack
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import httpx
from dotenv import load_dotenv
from planet import Planet, data_filter, order_request
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

LOGGER = logging.getLogger("planet-median")

DEFAULT_AOI: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [
            [84.9, 28.0],
            [86.0, 28.0],
            [86.0, 29.1],
            [84.9, 29.1],
            [84.9, 28.0],
        ]
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PlanetScope scenes and build a local median composite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aoi", type=Path, help="AOI GeoJSON in EPSG:4326")
    parser.add_argument("--start", required=True, type=parse_iso_date, help="Start date (YYYY-MM-DD, UTC)")
    parser.add_argument("--end", required=True, type=parse_iso_date, help="End date (YYYY-MM-DD, UTC, inclusive)")
    parser.add_argument("--item-type", default="PSScene")
    parser.add_argument("--asset-type", default="ortho_analytic_8b_sr")
    parser.add_argument("--product-bundle", default="analytic_8b_sr_udm2")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Maximum scene cloud cover, percent")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of scenes/order items")
    parser.add_argument("--output", type=Path, default=Path("data/raw/planet/median"))
    parser.add_argument("--order-name", help="Planet order name; generated if omitted")
    parser.add_argument("--resolution", type=float, default=3.0, help="Output pixel size in metres")
    parser.add_argument("--target-crs", help="Output CRS such as EPSG:32645; auto UTM if omitted")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=250_000_000,
        help="Safety limit on output width multiplied by height",
    )
    parser.add_argument("--block-size", type=int, default=512, help="Processing window size in pixels")
    parser.add_argument("--no-udm2-mask", action="store_true", help="Do not restrict inputs to UDM2 clear pixels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloads/composites where supported")
    parser.add_argument(
        "--from-download",
        type=Path,
        help="Skip API search/order and composite GeoTIFFs already below this directory",
    )
    parser.add_argument(
        "--resume-order",
        help="Resume polling/downloading an existing Planet order ID; never creates a new order",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Seconds between order status checks")
    parser.add_argument("--wait-hours", type=float, default=6.0, help="Maximum time to wait for an order")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit and download an order; without this, only search metadata is saved",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if not 0 <= args.max_cloud <= 100:
        parser.error("--max-cloud must be between 0 and 100")
    if args.limit < 1 or args.limit > 500:
        parser.error("--limit must be between 1 and the Orders API limit of 500")
    if any(value <= 0 for value in (args.resolution, args.block_size, args.max_pixels, args.poll_seconds, args.wait_hours)):
        parser.error("resolution, block size, pixel limit, polling interval, and wait time must be positive")
    selected_modes = sum(bool(value) for value in (args.from_download, args.resume_order, args.submit))
    if selected_modes > 1:
        parser.error("--from-download, --resume-order, and --submit are mutually exclusive")
    return args


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def load_aoi(path: Path | None) -> dict[str, Any]:
    raw = DEFAULT_AOI if path is None else json.loads(path.read_text(encoding="utf-8"))
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


def search_scenes(pl: Planet, args: argparse.Namespace, aoi: dict[str, Any]) -> list[dict[str, Any]]:
    start = datetime.combine(args.start, time.min, tzinfo=timezone.utc)
    end = datetime.combine(args.end, time.max, tzinfo=timezone.utc)
    search_filter = data_filter.and_filter(
        [
            data_filter.date_range_filter("acquired", gte=start, lte=end),
            data_filter.range_filter("cloud_cover", gte=0.0, lte=args.max_cloud / 100.0),
            data_filter.asset_filter([args.asset_type]),
            data_filter.permission_filter(),
        ]
    )
    return list(
        pl.data.search(
            [args.item_type],
            search_filter=search_filter,
            geometry=aoi,
            sort="acquired asc",
            limit=args.limit,
        )
    )


def write_search_results(path: Path, scenes: list[dict[str, Any]]) -> None:
    features = []
    for scene in scenes:
        # Search responses are GeoJSON-like; retaining selected links/properties
        # provides provenance without embedding authentication credentials.
        features.append(
            {
                "type": "Feature",
                "id": scene.get("id"),
                "geometry": scene.get("geometry"),
                "properties": scene.get("properties", {}),
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def print_scene_summary(scenes: list[dict[str, Any]]) -> None:
    print(f"Found {len(scenes)} downloadable scene(s):")
    for scene in scenes:
        properties = scene.get("properties", {})
        cloud = properties.get("cloud_cover")
        cloud_text = "unknown" if cloud is None else f"{100 * float(cloud):.1f}%"
        print(f"  {scene.get('id')}  {properties.get('acquired')}  cloud={cloud_text}")


def create_and_download_order(
    pl: Planet,
    args: argparse.Namespace,
    aoi: dict[str, Any],
    scene_ids: list[str],
    download_dir: Path,
) -> tuple[str, str]:
    name = args.order_name or f"nepal-flashflood-{args.start}-{args.end}"
    product = order_request.product(scene_ids, args.product_bundle, args.item_type)
    request = order_request.build_request(
        name=name,
        products=[product],
        order_type="partial",
        tools=[order_request.clip_tool(aoi)],
    )
    created = pl.orders.create_order(request)
    order_id = created["id"]
    LOGGER.info("Created order %s", order_id)
    state, _ = wait_for_order(pl, order_id, args.poll_seconds, args.wait_hours)
    download_order(pl, order_id, download_dir, args.overwrite)
    return order_id, state


def wait_for_order(
    pl: Planet, order_id: str, poll_seconds: float, wait_hours: float
) -> tuple[str, dict[str, Any]]:
    """Poll an order while tolerating temporary connection/read timeouts."""
    terminal_states = {"success", "partial", "failed", "cancelled"}
    deadline = clock.monotonic() + wait_hours * 3600
    consecutive_errors = 0
    last_state: str | None = None
    while clock.monotonic() < deadline:
        try:
            order = pl.orders.get_order(order_id)
            state = str(order.get("state", "unknown"))
            consecutive_errors = 0
            if state != last_state:
                LOGGER.info("Order %s: %s", order_id, state)
                last_state = state
            if state in terminal_states:
                if state not in {"success", "partial"}:
                    hints = order.get("error_hints") or order.get("last_message") or "no details supplied"
                    raise RuntimeError(f"Planet order {order_id} ended in state {state!r}: {hints}")
                return state, order
        except httpx.TransportError as exc:
            consecutive_errors += 1
            LOGGER.warning(
                "Temporary Planet connection error (%s/12): %s; polling will continue",
                consecutive_errors,
                exc or type(exc).__name__,
            )
            if consecutive_errors >= 12:
                raise RuntimeError("Too many consecutive Planet connection errors") from exc
        clock.sleep(poll_seconds)
    raise TimeoutError(f"Order {order_id} did not finish within {wait_hours:g} hour(s)")


def download_order(
    pl: Planet, order_id: str, download_dir: Path, overwrite: bool
) -> None:
    """Download a completed order into the local order directory."""
    download_dir.mkdir(parents=True, exist_ok=True)
    pl.orders.download_order(
        order_id,
        directory=download_dir,
        overwrite=args.overwrite,
        progress_bar=True,
    )


def order_scene_ids(order: dict[str, Any]) -> list[str]:
    """Extract scene IDs from the products recorded in an Orders API response."""
    ids: list[str] = []
    for product in order.get("products", []):
        ids.extend(str(item_id) for item_id in product.get("item_ids", []))
    if not ids:
        raise ValueError("The order response contains no product item IDs")
    return list(dict.fromkeys(ids))


def choose_target_crs(aoi: dict[str, Any], requested: str | None) -> CRS:
    if requested:
        crs = CRS.from_user_input(requested)
        if not crs.is_projected:
            raise ValueError("--target-crs must be projected because --resolution is in metres")
        return crs
    centroid = shape(aoi).centroid
    zone = int(math.floor((centroid.x + 180) / 6) + 1)
    epsg = (32600 if centroid.y >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def output_grid(
    aoi: dict[str, Any], target_crs: CRS, resolution: float, max_pixels: int
) -> tuple[Any, int, int]:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected = shapely_transform(transformer.transform, shape(aoi))
    minx, miny, maxx, maxy = projected.bounds
    minx = math.floor(minx / resolution) * resolution
    miny = math.floor(miny / resolution) * resolution
    maxx = math.ceil(maxx / resolution) * resolution
    maxy = math.ceil(maxy / resolution) * resolution
    width = math.ceil((maxx - minx) / resolution)
    height = math.ceil((maxy - miny) / resolution)
    pixels = width * height
    if pixels > max_pixels:
        raise ValueError(
            f"Requested grid is {width:,} x {height:,} ({pixels:,} pixels), exceeding "
            f"--max-pixels {max_pixels:,}. Use a smaller AOI, coarser --resolution, or "
            "raise the limit only after checking memory and disk requirements."
        )
    return from_origin(minx, maxy, resolution, resolution), width, height


def find_scene_files(root: Path, scene_ids: Iterable[str]) -> list[tuple[str, Path, Path | None]]:
    tifs = [path for path in root.rglob("*.tif") if path.is_file()]
    images = [path for path in tifs if "udm" not in path.name.lower() and "composite" not in path.name.lower()]
    udms = [path for path in tifs if "udm2" in path.name.lower()]
    pairs: list[tuple[str, Path, Path | None]] = []
    for scene_id in scene_ids:
        scene_images = [path for path in images if scene_id in path.name]
        if not scene_images:
            LOGGER.warning("No analytic GeoTIFF found for scene %s", scene_id)
            continue
        if len(scene_images) > 1:
            raise ValueError(f"Multiple analytic GeoTIFFs found for scene {scene_id}: {scene_images}")
        scene_udms = [path for path in udms if scene_id in path.name]
        pairs.append((scene_id, scene_images[0], scene_udms[0] if scene_udms else None))
    if not pairs:
        raise FileNotFoundError(f"No downloaded analytic GeoTIFFs matched scene IDs below {root}")
    return pairs


def all_downloaded_scene_ids(root: Path) -> list[str]:
    """Infer IDs from Planet filenames when --from-download is used.

    PlanetScope filenames normally begin with ``YYYYMMDD_HHMMSS_<satellite>``.
    Everything before the first known analytic suffix is treated as the ID.
    """
    ids: set[str] = set()
    for path in root.rglob("*.tif"):
        lower = path.name.lower()
        if "udm" in lower or "composite" in lower:
            continue
        marker_positions = [pos for marker in ("_3b_", "_3b-", "_analytic") if (pos := lower.find(marker)) > 0]
        if marker_positions:
            ids.add(path.name[: min(marker_positions)])
    return sorted(ids)


def windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def build_median_composite(
    pairs: list[tuple[str, Path, Path | None]],
    aoi: dict[str, Any],
    output_path: Path,
    count_path: Path,
    target_crs: CRS,
    resolution: float,
    max_pixels: int,
    block_size: int,
    mask_udm2: bool,
    overwrite: bool,
) -> None:
    if (output_path.exists() or count_path.exists()) and not overwrite:
        raise FileExistsError("Composite output exists; use --overwrite to replace it")
    transform, width, height = output_grid(aoi, target_crs, resolution, max_pixels)
    LOGGER.info("Output grid: %s x %s pixels in %s", width, height, target_crs.to_string())

    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(image)) for _, image, _ in pairs]
        band_count = sources[0].count
        if any(source.count != band_count for source in sources):
            raise ValueError("Downloaded imagery has inconsistent band counts")
        image_vrts = [
            stack.enter_context(
                WarpedVRT(
                    source,
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                    dtype="float32",
                )
            )
            for source in sources
        ]

        udm_vrts: list[WarpedVRT | None] = []
        for _, _, udm_path in pairs:
            if mask_udm2 and udm_path:
                udm_source = stack.enter_context(rasterio.open(udm_path))
                udm_vrts.append(
                    stack.enter_context(
                        WarpedVRT(
                            udm_source,
                            crs=target_crs,
                            transform=transform,
                            width=width,
                            height=height,
                            resampling=Resampling.nearest,
                            nodata=0,
                        )
                    )
                )
            else:
                if mask_udm2:
                    LOGGER.warning("Missing UDM2; scene will use its raster validity mask")
                udm_vrts.append(None)

        profile = sources[0].profile.copy()
        for inherited_key in ("blockxsize", "blockysize", "interleave", "photometric"):
            profile.pop(inherited_key, None)
        use_tiling = width >= 256 and height >= 256
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=band_count,
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
            tiled=use_tiling,
        )
        if use_tiling:
            profile.update(blockxsize=256, blockysize=256)
        count_profile = profile.copy()
        count_profile.update(dtype="uint16", count=1, nodata=0, predictor=2)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as destination, rasterio.open(
            count_path, "w", **count_profile
        ) as count_destination:
            for window in windows(width, height, block_size):
                clear_masks: list[np.ndarray | None] = []
                for udm_vrt in udm_vrts:
                    clear_masks.append(None if udm_vrt is None else udm_vrt.read(1, window=window) == 1)

                observation_count: np.ndarray | None = None
                for band in range(1, band_count + 1):
                    observations = []
                    for image_vrt, clear in zip(image_vrts, clear_masks, strict=True):
                        array = image_vrt.read(band, window=window, masked=True).filled(np.nan).astype("float32")
                        if clear is not None:
                            array[~clear] = np.nan
                        observations.append(array)
                    stack_array = np.stack(observations)
                    if observation_count is None:
                        observation_count = np.sum(np.isfinite(stack_array), axis=0).astype("uint16")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                        median = np.nanmedian(stack_array, axis=0).astype("float32")
                    median[~np.isfinite(median)] = -9999.0
                    destination.write(median, band, window=window)
                assert observation_count is not None
                count_destination.write(observation_count, 1, window=window)

        LOGGER.info("Wrote %s", output_path)
        LOGGER.info("Wrote %s", count_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv()
    aoi = load_aoi(args.aoi)
    args.output.mkdir(parents=True, exist_ok=True)
    run_metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aoi": aoi,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "item_type": args.item_type,
        "asset_type": args.asset_type,
        "product_bundle": args.product_bundle,
        "max_cloud_percent": args.max_cloud,
        "resolution_m": args.resolution,
    }

    download_dir = args.from_download or (args.output / "order")
    if args.from_download:
        scene_ids = all_downloaded_scene_ids(download_dir)
        if not scene_ids:
            raise FileNotFoundError(f"Could not infer Planet scene IDs below {download_dir}")
        run_metadata["mode"] = "existing-download"
    elif args.resume_order:
        pl = Planet()
        state, order = wait_for_order(pl, args.resume_order, args.poll_seconds, args.wait_hours)
        scene_ids = order_scene_ids(order)
        run_metadata.update(
            mode="resumed-order",
            scene_ids=scene_ids,
            order_id=args.resume_order,
            order_state=state,
        )
        download_order(pl, args.resume_order, download_dir, args.overwrite)
    else:
        pl = Planet()
        scenes = search_scenes(pl, args, aoi)
        write_search_results(args.output / "search_results.geojson", scenes)
        print_scene_summary(scenes)
        if not scenes:
            LOGGER.warning("No matching scenes; no order or composite created")
            return 2
        scene_ids = [str(scene["id"]) for scene in scenes]
        run_metadata["scene_ids"] = scene_ids
        if not args.submit:
            run_metadata["mode"] = "search-only"
            (args.output / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
            print("Search preview complete. Review results, then rerun with --submit to order imagery.")
            return 0
        run_metadata["mode"] = "ordered"
        order_id, state = create_and_download_order(pl, args, aoi, scene_ids, download_dir)
        run_metadata.update(order_id=order_id, order_state=state)

    pairs = find_scene_files(download_dir, scene_ids)
    run_metadata["composited_scene_ids"] = [scene_id for scene_id, _, _ in pairs]
    target_crs = choose_target_crs(aoi, args.target_crs)
    run_metadata["target_crs"] = target_crs.to_string()
    build_median_composite(
        pairs=pairs,
        aoi=aoi,
        output_path=args.output / "median_composite.tif",
        count_path=args.output / "median_observation_count.tif",
        target_crs=target_crs,
        resolution=args.resolution,
        max_pixels=args.max_pixels,
        block_size=args.block_size,
        mask_udm2=not args.no_udm2_mask,
        overwrite=args.overwrite,
    )
    (args.output / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        sys.exit(130)
    except Exception as exc:  # concise CLI failure; --verbose gives logging context
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)
