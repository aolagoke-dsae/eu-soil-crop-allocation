"""Command line for Phase 1: ``python -m eusoil <command>``.

Commands, in the order they are normally run:

``points``        clean the tabular sources, write ``data/interim/points.parquet``,
                  the sample-flow table and the audit tables
``dem``           elevation and slope from Copernicus GLO-90 (needs internet access)
``terraclimate``  TerraClimate bioclimatic variables at the points (needs internet
                  access); ``--period 1991-2020`` is the primary climate block,
                  ``--period 2015-2018`` the climate-period sensitivity run
``worldclim``     WorldClim bio variables at the points (local GeoTIFFs); an
                  internal resolution check only, never published (protocol A7)
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
    # WorldClim is an internal resolution check (protocol A7): its licence does
    # not allow redistribution, so no output derived from it is published.
    prefix = "wc30s_bio" if args.resolution == "30s" else "wc25m_bio"
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
    from eusoil.ingest.terraclimate import PERIODS, extract_terraclimate

    result = extract_terraclimate(_points(paths), period=args.period)
    out = paths.interim / f"terraclimate_{args.period.replace('-', '_')}.parquet"
    result.to_parquet(out, index=False)
    tag = PERIODS[args.period]["tag"]
    LOG.info(
        "terraclimate %s: %d points, nearest-cell distance max %.1f km",
        args.period,
        len(result),
        float(result[f"{tag}_cell_km"].max()),
    )


# Interim files joined into the modelling table: file -> (column prefix it
# must carry, command that writes it). An older file without the prefix was
# written under a previous naming scheme and must be regenerated.
REQUIRED_EXTRACTIONS = {
    "terraclimate_1991_2020.parquet": ("tc9120_bio", "terraclimate --period 1991-2020"),
    "dem.parquet": ("elevation_m", "dem"),
}
OPTIONAL_EXTRACTIONS = {
    "terraclimate_2015_2018.parquet": ("tc1518_bio", "terraclimate --period 2015-2018"),
    "worldclim_30s.parquet": ("wc30s_bio", "worldclim --resolution 30s"),
    "worldclim_2.5m.parquet": ("wc25m_bio", "worldclim --resolution 2.5m"),
}


def _read_extraction(path: Path, prefix: str, command: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not any(column.startswith(prefix) for column in frame.columns):
        sys.exit(
            f"{path} has no '{prefix}*' columns: it predates the current naming. "
            f"Re-run 'python -m eusoil {command}'."
        )
    return frame


def cmd_table(args, config, paths) -> None:
    points = _points(paths)
    extractions = []
    for name, (prefix, command) in REQUIRED_EXTRACTIONS.items():
        path = paths.interim / name
        if not path.exists():
            sys.exit(f"{path} not found: run 'python -m eusoil {command}' first")
        extractions.append((name, _read_extraction(path, prefix, command)))
    for name, (prefix, command) in OPTIONAL_EXTRACTIONS.items():
        path = paths.interim / name
        if path.exists():
            extractions.append((name, _read_extraction(path, prefix, command)))
        else:
            LOG.warning("%s not found; the table is built without it", name)
    try:
        table = build_modelling_table(points, extractions, config)
    except ValueError as error:
        sys.exit(str(error))
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
    p = sub.add_parser("worldclim", help="WorldClim bio variables (internal check only)")
    p.add_argument("--resolution", choices=["30s", "2.5m"], default="30s")
    sub.add_parser("dem", help="elevation and slope from Copernicus GLO-90 (internet)")
    p = sub.add_parser("terraclimate", help="TerraClimate bioclimatic variables (internet)")
    p.add_argument("--period", choices=["1991-2020", "2015-2018"], default="1991-2020")
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
