"""Assemble the analysis tables and the data-audit reports (protocol Phase 1).

Two products:

``data/interim/points.parquet``
    One row per agricultural point that passed every filter: cleaned soil,
    land-cover and location-quality fields, texture and the target. Built
    from the tabular sources only.

``data/processed/modelling_table.parquet``
    ``points`` joined with the point extractions (WorldClim, DEM and,
    optionally, TerraClimate), with projected coordinates.

The audit tables written to ``reports/tables/`` contain counts and shares
only, never point-level soil values, so they can be published without
breaching the ESDAC licence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pyproj import Transformer

from eusoil.clean import clean_soil
from eusoil.config import Paths
from eusoil.flow import SampleFlow
from eusoil.ingest.landcover import choose_coordinates, read_landcover_2018
from eusoil.ingest.soil import read_soil
from eusoil.ingest.texture import read_texture
from eusoil.target import assign_target, classes_below_minimum, excluded_code_counts

LOG = logging.getLogger(__name__)

AGRICULTURAL_LAND_USE = "U111"  # Agriculture (excluding fallow land and kitchen gardens)
AUDIT_VARIABLES = [
    "ph_cacl2",
    "oc_g_kg",
    "caco3_g_kg",
    "n_g_kg",
    "p_mg_kg",
    "k_mg_kg",
    "ec_ms_m",
    "clay",
    "silt",
]


def _check_one_to_one(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    missing = set(left["point_id"]) - set(right["point_id"])
    if missing:
        raise ValueError(f"{len(missing)} points have no row in the {name} table")


def build_points(
    paths: Paths, config: dict[str, Any], flow: SampleFlow, landcover_path: Path | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the point table and the audit tables from the tabular sources.

    ``landcover_path`` overrides the configured harmonised table (used in
    tests with a reduced extract).
    """
    soil = read_soil(paths.soil)
    flow.note(soil, step="soil samples", reason="LUCAS 2018 topsoil table, one row per point")
    soil = clean_soil(soil, config["cleaning"], flow)

    landcover = read_landcover_2018(landcover_path or paths.landcover_2018)
    _check_one_to_one(soil, landcover, "harmonised 2018")
    points = soil.merge(landcover, on="point_id", how="left", validate="one_to_one")
    disagree = points["lc1"].ne(points["lc1_harmo"])
    if disagree.any():
        raise ValueError(
            f"{int(disagree.sum())} points have a different LC1 in the soil and harmonised tables"
        )
    points = choose_coordinates(points)

    points = flow.apply(
        points,
        points["lc1"].str[0].isin(["B", "E"]),
        step="agricultural land cover",
        reason="cropland (B) or grassland (E) recorded at the point",
    )
    points = flow.apply(
        points,
        points["lu1"].eq(AGRICULTURAL_LAND_USE),
        step="agricultural land use",
        reason="land use U111: agriculture, excluding fallow and kitchen gardens",
    )

    points = assign_target(points, config["target"])
    excluded = excluded_code_counts(points)
    points = flow.apply(
        points,
        points["crop_group"].notna(),
        step="crop group",
        reason="land-cover code belongs to a configured crop group",
    )
    below = classes_below_minimum(points, config["target"]["min_class_size"])
    if not below.empty:
        raise ValueError(f"Crop groups below the minimum size: {below.to_dict()}")

    texture = read_texture(paths.texture)
    points = points.merge(texture, on="point_id", how="left", validate="one_to_one")

    pure = set(config["cleaning"]["pure_lc1_perc"])
    points["label_pure"] = points["lc1_perc"].isin(pure)
    points["observed_in_situ"] = points["obs_type"].eq("In situ < 100 m")

    audit = audit_tables(points, excluded)
    return points.reset_index(drop=True), audit


def audit_tables(points: pd.DataFrame, excluded: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Counts and shares for the data audit; no point-level values."""
    tables: dict[str, pd.DataFrame] = {}
    tables["class_counts"] = (
        points.groupby(["farming_system", "crop_group"]).size().rename("n_points").reset_index()
    )
    tables["class_by_country"] = pd.crosstab(
        points["nuts0"], points["crop_group"], margins=True, margins_name="total"
    )
    tables["excluded_codes"] = excluded
    tables["missing_by_variable"] = (
        points[AUDIT_VARIABLES]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .rename("percent_missing")
        .rename_axis("variable")
        .reset_index()
    )
    tables["missing_by_country"] = points.groupby("nuts0")[AUDIT_VARIABLES].agg(
        lambda s: round(100 * s.isna().mean(), 1)
    )
    lod_columns = [c for c in points.columns if c.endswith("_below_lod")]
    tables["below_lod_by_group"] = (
        points.groupby("crop_group")[lod_columns].mean().mul(100).round(1)
    )
    tables["below_lod_by_country"] = points.groupby("nuts0")[lod_columns].mean().mul(100).round(1)
    flag_columns = (
        [c for c in points.columns if c.startswith("implausible_")]
        + ["organic_soil"]
        + [c for c in points.columns if c.endswith("_invalid")]
    )
    tables["flags"] = (
        points[flag_columns].sum().astype(int).rename("n_points").rename_axis("flag").reset_index()
    )
    tables["quality"] = pd.DataFrame(
        {
            "item": [
                "coordinates from GPS",
                "observed in situ within 100 m",
                "label pure (lc1_perc >= 50%)",
                "texture measured",
            ],
            "n_points": [
                int(points["coord_source"].eq("gps").sum()),
                int(points["observed_in_situ"].sum()),
                int(points["label_pure"].sum()),
                int(points["clay"].notna().sum()),
            ],
        }
    )
    tables["texture_year"] = (
        points["texture_year"]
        .value_counts(dropna=False)
        .rename("n_points")
        .rename_axis("texture_year")
        .reset_index()
    )
    return tables


def add_projected_coordinates(df: pd.DataFrame, crs: dict[str, str]) -> pd.DataFrame:
    """Add ``easting`` and ``northing`` in the projected CRS (metres)."""
    transformer = Transformer.from_crs(crs["geographic"], crs["projected"], always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["easting"] = x
    out["northing"] = y
    return out


def build_modelling_table(
    points: pd.DataFrame, extractions: list[pd.DataFrame], config: dict[str, Any]
) -> pd.DataFrame:
    """Join the point extractions and add projected coordinates."""
    table = points
    for extraction in extractions:
        table = table.merge(extraction, on="point_id", how="left", validate="one_to_one")
    table = add_projected_coordinates(table, config["crs"])
    table["country"] = table["nuts0"]
    return table


def write_tables(tables: dict[str, pd.DataFrame], directory: Path, prefix: str = "audit") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        keep_index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(directory / f"{prefix}_{name}.csv", index=keep_index)
