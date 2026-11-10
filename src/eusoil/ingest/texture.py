"""Read the LUCAS particle-size (texture) table.

Texture changes over decades, not years, so the values measured at the same
point in an earlier survey (mostly 2009, then 2015 and 2012) are used as a
fixed property of the 2018 point. ``texture_year`` records which survey
supplied each value, for the sensitivity run restricted to points with
measured texture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS: dict[str, str] = {
    "POINTID": "point_id",
    "Clay": "clay",
    "Silt": "silt",
    "Sand": "sand",
    "Coarse": "coarse_fragments",
    "PSDAvailable": "psd_available",
    "TextureFromSurvey": "texture_year",
    "USDA": "usda_texture_class",
}

SUM_TOLERANCE = 2.0  # percentage points; clay + silt + sand should be 100


def read_texture(path: Path) -> pd.DataFrame:
    """Read texture, keeping values only where a particle-size analysis exists.

    Raises ``ValueError`` if ``point_id`` is not unique or if clay, silt and
    sand do not add up to 100 within :data:`SUM_TOLERANCE` where they are
    present.
    """
    df = pd.read_csv(
        path, usecols=list(COLUMNS), dtype=str, keep_default_na=False, na_values=["", "NA"]
    ).rename(columns=COLUMNS)
    df["point_id"] = df["point_id"].astype("int64")
    if not df["point_id"].is_unique:
        raise ValueError("point_id is not unique in the texture table")
    for column in ("clay", "silt", "sand", "coarse_fragments"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    available = df["psd_available"].str.strip().str.lower().eq("yes").fillna(False)
    df.loc[~available, ["clay", "silt", "sand"]] = float("nan")
    df["texture_year"] = pd.to_numeric(df["texture_year"], errors="coerce").astype("Int64")

    total = df[["clay", "silt", "sand"]].sum(axis=1, min_count=3)
    off = total.notna() & (total - 100).abs().gt(SUM_TOLERANCE)
    if off.any():
        raise ValueError(f"{int(off.sum())} texture rows do not sum to 100 +/- {SUM_TOLERANCE}")
    return df.drop(columns=["psd_available"])
