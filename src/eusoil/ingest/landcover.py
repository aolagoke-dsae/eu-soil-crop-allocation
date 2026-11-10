"""Read the fields needed from the harmonised LUCAS 2018 survey table.

The soil file already carries each point's land cover, land use and
theoretical coordinates. The harmonised table (d'Andrimont et al., 2020)
adds what the protocol needs for location and label quality: the GPS
position, how the point was observed, the share of the surroundings under
the recorded land cover, irrigation and grazing.

Only the 2018 table is read, and only these columns, because the file is
large (about 430 MB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS: dict[str, str] = {
    "point_id": "point_id",
    "lc1": "lc1_harmo",
    "lc1_label": "lc1_label",
    "lc1_perc": "lc1_perc",
    "lu1": "lu1_harmo",
    "obs_type": "obs_type",
    "obs_dist": "obs_dist",
    "th_gps_dist": "th_gps_dist_m",
    "gps_lat": "gps_lat",
    "gps_long": "gps_long",
    "gps_ew": "gps_ew",
    "gps_prec": "gps_prec_m",
    "wm": "water_management",
    "grazing": "grazing",
    "parcel_area_ha": "parcel_area_ha",
}

IN_SITU = "In situ < 100 m"


def read_landcover_2018(path: Path) -> pd.DataFrame:
    """Read the 2018 harmonised survey, keeping the columns in :data:`COLUMNS`."""
    df = pd.read_csv(
        path, usecols=list(COLUMNS), dtype=str, keep_default_na=False, na_values=["", "NA"]
    )
    df = df.rename(columns=COLUMNS)
    df["point_id"] = df["point_id"].astype("int64")
    if not df["point_id"].is_unique:
        raise ValueError("point_id is not unique in the harmonised 2018 table")
    df["lc1_harmo"] = df["lc1_harmo"].str.strip().str.upper()
    for column in ("th_gps_dist_m", "gps_lat", "gps_prec_m", "parcel_area_ha"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["gps_lon"] = signed_longitude(df["gps_long"], df["gps_ew"])
    return df.drop(columns=["gps_long", "gps_ew"])


def signed_longitude(longitude: pd.Series, east_west: pd.Series) -> pd.Series:
    """Combine an unsigned GPS longitude and its East/West flag.

    The harmonised table stores the GPS longitude as a magnitude plus a
    separate ``gps_ew`` flag. A value that is already negative is left as
    it is; a positive value flagged West is negated.
    """
    value = pd.to_numeric(longitude, errors="coerce")
    is_west = east_west.astype("string").str.strip().str.lower().eq("west").fillna(False)
    return pd.Series(
        np.where(is_west & (value > 0), -value, value), index=longitude.index, dtype="float64"
    )


EUROPE_LAT = (27.0, 72.0)  # includes the Canary Islands and northern Scandinavia
EUROPE_LON = (-32.0, 45.0)  # includes the Azores and Cyprus
MAX_GPS_OFFSET_M = 100.0


def choose_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the coordinates used for every spatial operation.

    The GPS position is used when the point was observed in the field within
    100 m, the fix lies inside Europe and it is at most 100 m from the
    theoretical position (the GPS columns contain placeholder values such as
    0 and 88.88889). Otherwise the theoretical position is used. The
    ``coord_source`` column records which.
    """
    out = df.copy()
    plausible = (
        out["gps_lat"].between(*EUROPE_LAT)
        & out["gps_lon"].between(*EUROPE_LON)
        & out["th_gps_dist_m"].le(MAX_GPS_OFFSET_M)
    )
    use_gps = out["obs_type"].eq(IN_SITU) & plausible.fillna(False)
    out["lat"] = np.where(use_gps, out["gps_lat"], out["th_lat"])
    out["lon"] = np.where(use_gps, out["gps_lon"], out["th_lon"])
    out["coord_source"] = np.where(use_gps, "gps", "theoretical")
    return out
