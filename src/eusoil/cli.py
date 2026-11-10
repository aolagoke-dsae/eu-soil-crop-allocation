"""Command line for Phase 1: ``python -m eusoil <command>``.

Commands, in the order they are normally run:

``points``        clean the tabular sources, write ``data/interim/points.parquet``,
                  the sample-flow table and the audit tables
``worldclim``     WorldClim bio variables at the points (local GeoTIFFs)
``dem``           elevation and slope from Copernicus GLO-90 (needs internet access)
``terraclimate``  TerraClimate 2015-2018 at the points (needs internet access)
``table``         join everything into ``data/processed/modelling_table.parquet``
``precision``     the H1 precision check (simulation on labels and coordinates)

Each command reads the outputs of the earlier ones from ``data/interim``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from eusoil.build import (
    add_projected_coordinates,
    build_modelling_table,
    build_points,
    write_tables,
)
from eusoil.config import load_config, resolve_paths
from eusoil.flow import SampleFlow

LOG = logging.getLogger("eusoil")


def _points(paths) -> pd.DataFrame:
    path = paths.interim / "points.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: run 'python -m eusoil points' first")
    return pd.read_parquet(path)


def cmd_points(args, config, paths) -> None:
    flow = SampleFlow()
    landcover = Path(args.landcover) if args.landcover else None
    points, audit = build_points(paths, config, flow, landcover_path=landcover)
    points.to_parquet(paths.interim / "points.parquet", index=False)
    flow.save(paths.reports / "tables" / "sample_flow.csv")
    write_tables(audit, paths.reports / "tables")
    LOG.info("points: %d rows; sample flow and %d audit tables written", len(points), len(audit))


def cmd_worldclim(args, config, paths) -> None:
    from eusoil.ingest.climate import extract_worldclim

    directory = paths.worldclim_30s if args.resolution == "30s" else paths.worldclim_2_5m
    # 30s is the primary climate block (bio1..bio19); 2.5m is the resolution
    # sensitivity run (wc25m_bio1..wc25m_bio19).
    prefix = "bio" if args.resolution == "30s" else "wc25m_bio"
    result = extract_worldclim(_points(paths), directory, prefix=prefix)
    out = paths.interim / f"worldclim_{args.resolution}.parquet"
    result.to_parquet(out, index=False)
    filled = int((result[f"{prefix}_fill_px"] > 0).sum())
    missing = int(result[f"{prefix}1"].isna().sum())
    LOG.info(
        "worldclim %s: %d points, %d filled from a neighbouring pixel, %d still missing",
        args.resolution,
        len(result),
        filled,
        missing,
    )


def cmd_dem(args, config, paths) -> None:
    from eusoil.ingest.dem import extract_dem

    result = extract_dem(_points(paths), cache=paths.interim / "dem_tiles")
    result.to_parquet(paths.interim / "dem.parquet", index=False)
    LOG.info(
        "dem: %d points, elevation missing %d, slope missing %d",
        len(result),
        int(result["elevation_m"].isna().sum()),
        int(result["slope_deg"].isna().sum()),
    )


def cmd_terraclimate(args, config, paths) -> None:
    from eusoil.ingest.terraclimate import extract_terraclimate

    result = extract_terraclimate(_points(paths))
    result.to_parquet(paths.interim / "terraclimate_2015_2018.parquet", index=False)
    LOG.info(
        "terraclimate: %d points, nearest-cell distance max %.1f km",
        len(result),
        float(result["tc_cell_distance_km"].max()),
    )


def cmd_table(args, config, paths) -> None:
    points = _points(paths)
    required = {"worldclim_30s.parquet": "worldclim", "dem.parquet": "dem"}
    optional = ["worldclim_2.5m.parquet", "terraclimate_2015_2018.parquet"]
    extractions = []
    for name, command in required.items():
        path = paths.interim / name
        if not path.exists():
            sys.exit(f"{path} not found: run 'python -m eusoil {command}' first")
        extractions.append(pd.read_parquet(path))
    for name in optional:
        path = paths.interim / name
        if path.exists():
            extractions.append(pd.read_parquet(path))
        else:
            LOG.warning("%s not found; the table is built without it", name)
    table = build_modelling_table(points, extractions, config)
    table.to_parquet(paths.processed / "modelling_table.parquet", index=False)
    LOG.info("modelling table: %d rows, %d columns", *table.shape)


def cmd_precision(args, config, paths) -> None:
    from eusoil.precision import precision_check

    table_path = paths.processed / "modelling_table.parquet"
    table = (
        pd.read_parquet(table_path)
        if table_path.exists()
        else add_projected_coordinates(_points(paths), config["crs"])
    )
    result = precision_check(table, config["precision"], seed=config["seed"])
    out = paths.reports / "tables" / "precision_check.csv"
    result.to_csv(out, index=False)
    LOG.info("precision check written to %s\n%s", out, result.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eusoil", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="path to analysis.yaml (default: config/analysis.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("points", help="clean the tabular sources and write the audit")
    p.add_argument("--landcover", help="override the harmonised 2018 table (testing)")
    p = sub.add_parser("worldclim", help="WorldClim bio variables at the points")
    p.add_argument("--resolution", choices=["30s", "2.5m"], default="30s")
    sub.add_parser("dem", help="elevation and slope from Copernicus GLO-90 (internet)")
    sub.add_parser("terraclimate", help="TerraClimate 2015-2018 (internet)")
    sub.add_parser("table", help="build the modelling table")
    sub.add_parser("precision", help="H1 precision check")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    paths = resolve_paths(config)
    paths.ensure_output_dirs()
    commands = {
        "points": cmd_points,
        "worldclim": cmd_worldclim,
        "dem": cmd_dem,
        "terraclimate": cmd_terraclimate,
        "table": cmd_table,
        "precision": cmd_precision,
    }
    commands[args.command](args, config, paths)


if __name__ == "__main__":
    main()
