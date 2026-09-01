#!/usr/bin/env python3
"""Build a native-resolution mosaic from PlanetScope visual GeoTIFF tiles.

The command searches an input directory recursively for ``*_visual.tif`` and
writes one georeferenced GeoTIFF at the finest native resolution found among
the inputs. It preserves the source CRS and RGB band values; it does not
reproject, rescale, stretch, or downsample the imagery. Source tiles are never
modified.

The output uses internal 512-pixel tiling, lossless DEFLATE compression and
BigTIFF so mosaics larger than 4 GiB are supported. Rasterio writes the mosaic
in memory-limited chunks rather than holding the complete native-resolution
image in RAM. Earlier sorted tiles win where scenes overlap by default.

Default August 28, 2026 PowerShell command::

    python scripts/mosaic_planetscope_visual.py

Equivalent explicit command::

    python scripts/mosaic_planetscope_visual.py `
      --input planet/post_event/planetscope-2026-08-28/items `
      --pattern "20260828_*/*_visual.tif" `
      --output data/processed/planet/planetscope_20260828_visual_mosaic.tif `
      --overwrite

Use ``--method last`` if later sorted scenes should replace earlier scenes in
overlaps. The native mosaic can be very large. Ensure sufficient free disk
space; temporary and final storage requirements depend on footprint overlap
and compression ratio.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pyproj import datadir as pyproj_datadir


def configure_proj() -> Path:
    """Point GDAL/PROJ at a packaged database before importing rasterio."""
    candidates: list[Path] = []
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.submodule_search_locations:
        rasterio_dir = Path(next(iter(spec.submodule_search_locations)))
        candidates.extend((rasterio_dir / "proj_data", rasterio_dir / "data"))
    candidates.append(Path(pyproj_datadir.get_data_dir()))
    proj_data = next((path for path in candidates if (path / "proj.db").is_file()), None)
    if proj_data is None:
        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Could not find proj.db; searched: {searched}")
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
    return proj_data


PROJ_DATA = configure_proj()

import rasterio  # noqa: E402
from rasterio.merge import merge  # noqa: E402

LOG = logging.getLogger("planetscope-mosaic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("planet/post_event/planetscope-2026-08-28/items"),
        help="Root directory searched recursively for visual tiles.",
    )
    parser.add_argument(
        "--pattern",
        default="20260828_*/*_visual.tif",
        help="Glob relative to --input (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/planet/planetscope_20260828_visual_mosaic.tif"),
        help="Destination GeoTIFF.",
    )
    parser.add_argument(
        "--method",
        choices=("first", "last", "min", "max"),
        default="first",
        help="Pixel selection rule for overlapping valid imagery.",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=512,
        metavar="MB",
        help="Approximate Rasterio processing memory limit in MiB.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output mosaic.")
    return parser.parse_args()


def discover(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No input tiles matched {root / pattern}")
    return paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.memory_limit < 64:
        raise ValueError("--memory-limit must be at least 64 MiB")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {args.output}")

    paths = discover(args.input, args.pattern)
    LOG.info("Using PROJ data directory: %s", PROJ_DATA)
    LOG.info("Found %d visual tile(s)", len(paths))
    sources = [rasterio.open(path) for path in paths]
    try:
        first = sources[0]
        if first.crs is None:
            raise ValueError(f"Input has no CRS: {paths[0]}")
        for path, source in zip(paths[1:], sources[1:]):
            if source.crs != first.crs:
                raise ValueError(f"CRS differs from the first tile: {path}")
            if source.count != first.count:
                raise ValueError(f"Band count differs from the first tile: {path}")
            if source.dtypes != first.dtypes:
                raise ValueError(f"Data type differs from the first tile: {path}")

        finest_x = min(abs(source.res[0]) for source in sources)
        finest_y = min(abs(source.res[1]) for source in sources)
        nodata = first.nodata if first.nodata is not None else 0
        predictor = 2 if first.dtypes[0].startswith(("uint", "int")) else 3
        args.output.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("Writing native mosaic at %.6f x %.6f CRS units", finest_x, finest_y)
        merge(
            sources,
            res=(finest_x, finest_y),
            nodata=nodata,
            method=args.method,
            target_aligned_pixels=True,
            mem_limit=args.memory_limit,
            dst_path=args.output,
            dst_kwds={
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "DEFLATE",
                "predictor": predictor,
                "bigtiff": "YES",
                "num_threads": "ALL_CPUS",
                "sparse_ok": True,
            },
        )
    finally:
        for source in sources:
            source.close()

    with rasterio.open(args.output) as result:
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(args.output),
            "source_count": len(paths),
            "sources": [str(path) for path in paths],
            "crs": result.crs.to_string() if result.crs else None,
            "bounds": list(result.bounds),
            "resolution": list(result.res),
            "width": result.width,
            "height": result.height,
            "bands": result.count,
            "dtype": list(result.dtypes),
            "nodata": result.nodata,
            "overlap_method": args.method,
        }
    manifest = args.output.with_suffix(".json")
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote %s (%d x %d pixels)", args.output, summary["width"], summary["height"])
    LOG.info("Wrote provenance manifest: %s", manifest)


if __name__ == "__main__":
    main()
