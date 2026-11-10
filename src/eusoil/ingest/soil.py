"""Read the LUCAS 2018 topsoil table and parse its measurement codes.

The file stores measurements as text. Besides plain numbers it contains the
codes ``"< LOD"`` (with one or two spaces) and ``"<0.0"``. Reading every
column as text and parsing explicitly keeps those codes from being turned
into missing values without a trace.

Units (JRC 2022 report, Table 3): organic carbon, total N and CaCO3 in g/kg;
extractable P (Olsen) and K in mg/kg; electrical conductivity in mS/m.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Raw column -> analysis column. Columns not listed are dropped on read; the
# reason for each drop is in DROPPED_COLUMNS.
COLUMN_MAP: dict[str, str] = {
    "POINTID": "point_id",
    "Depth": "depth",
    "pH_CaCl2": "ph_cacl2",
    "pH_H2O": "ph_h2o",
    "EC": "ec_ms_m",
    "OC": "oc_g_kg",
    "CaCO3": "caco3_g_kg",
    "P": "p_mg_kg",
    "N": "n_g_kg",
    "K": "k_mg_kg",
    "NUTS_0": "nuts0",
    "NUTS_1": "nuts1",
    "NUTS_2": "nuts2",
    "NUTS_3": "nuts3",
    "TH_LAT": "th_lat",
    "TH_LONG": "th_lon",
    "SURVEY_DATE": "survey_date",
    "LC": "lc1",
    "LU": "lu1",
    "LC0_Desc": "lc0_desc",
    "LC1_Desc": "lc1_desc",
    "LU1_Desc": "lu1_desc",
}

DROPPED_COLUMNS: dict[str, str] = {
    "OC (20-30 cm)": "subsoil layer, outside the 0-20 cm unit of analysis",
    "CaCO3 (20-30 cm)": "subsoil layer, outside the 0-20 cm unit of analysis",
    "Ox_Al": "measured for about 13% of points only",
    "Ox_Fe": "measured for about 13% of points only",
    "Elev": "unusable (code 8888, values above 10,000); elevation comes from the DEM",
}

MEASUREMENTS: tuple[str, ...] = (
    "ph_cacl2",
    "ph_h2o",
    "ec_ms_m",
    "oc_g_kg",
    "caco3_g_kg",
    "p_mg_kg",
    "n_g_kg",
    "k_mg_kg",
)

_LOD_TOKEN = re.compile(r"^<\s*LOD$", flags=re.IGNORECASE)
_NEGATIVE_ZERO_TOKEN = re.compile(r"^<\s*0(\.0+)?$")


def read_soil(path: Path) -> pd.DataFrame:
    """Read the soil CSV with every value as text and rename the columns.

    Raises ``ValueError`` if an expected column is missing or ``point_id``
    is not unique, because every later join relies on it.
    """
    # Only the empty cell and the literal "NA" mean "missing"; every other
    # code (for example "< LOD") is kept as text and parsed explicitly.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA"])
    missing = sorted(set(COLUMN_MAP) - set(raw.columns))
    if missing:
        raise ValueError(f"Soil file is missing expected columns: {missing}")
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["point_id"] = df["point_id"].astype("int64")
    if not df["point_id"].is_unique:
        raise ValueError("point_id is not unique in the soil file")
    # Land-cover codes appear as both 'Bx1' and 'BX1'; the harmonised survey
    # uses upper case.
    df["lc1"] = df["lc1"].str.strip().str.upper()
    df["lu1"] = df["lu1"].str.strip().str.upper()
    df["th_lat"] = pd.to_numeric(df["th_lat"], errors="raise")
    df["th_lon"] = pd.to_numeric(df["th_lon"], errors="raise")
    df["survey_date"] = pd.to_datetime(df["survey_date"], format="%d-%m-%y", errors="raise")
    return df


def parse_measurement(text: pd.Series, lod: float | None) -> pd.DataFrame:
    """Parse one measurement column.

    Returns a frame with three columns, aligned with ``text``:

    ``value``
        The number as reported, or NaN for a code, a missing cell or a value
        flagged below the limit of detection.
    ``below_lod``
        True when the cell is the ``< LOD`` code, or when a number is below
        ``lod`` (the file contains some numbers under the documented limit;
        they are treated as censored too, for consistency).
    ``invalid``
        True for the ``<0.0`` code, which is neither a measurement nor a
        detection-limit code.

    A text that is none of these raises ``ValueError``, so an unexpected
    code is found rather than silently lost.
    """
    stripped = text.astype("string").str.strip()
    is_lod_code = stripped.str.match(_LOD_TOKEN).fillna(False).astype(bool)
    is_invalid = stripped.str.match(_NEGATIVE_ZERO_TOKEN).fillna(False).astype(bool)
    numeric = pd.to_numeric(stripped.where(~(is_lod_code | is_invalid)), errors="coerce")

    unparsed = stripped.notna() & ~is_lod_code & ~is_invalid & numeric.isna()
    if unparsed.any():
        examples = stripped[unparsed].unique()[:5].tolist()
        raise ValueError(f"Unrecognised measurement codes: {examples}")

    below_numeric = numeric.lt(lod) if lod is not None else pd.Series(False, index=text.index)
    below_lod = is_lod_code | below_numeric.fillna(False).astype(bool)
    value = numeric.where(~below_lod).astype("float64")
    return pd.DataFrame(
        {"value": value, "below_lod": below_lod.astype(bool), "invalid": is_invalid.astype(bool)},
        index=text.index,
    )


def caco3_from_codes(text: pd.Series) -> pd.DataFrame:
    """Decode CaCO3 following the 2018 readme.

    An empty cell means "absent, pH < 7"; ``< LOD`` means "absent,
    pH >= 7"; every measured value is at least 1 g/kg. Both codes therefore
    mean 0 g/kg. Returns the value and a ``present`` flag.
    """
    parsed = parse_measurement(text, lod=None)
    absent = text.isna() | parsed["below_lod"]
    value = parsed["value"].where(~absent, 0.0)
    if parsed["invalid"].any():
        raise ValueError("CaCO3 contains '<0.0' codes, which the readme does not define")
    return pd.DataFrame(
        {"value": value.astype("float64"), "present": value.gt(0)}, index=text.index
    )


def log1p_safe(values: pd.Series) -> pd.Series:
    """log(1 + x) that keeps NaN as NaN and rejects negative input."""
    if (values.dropna() < 0).any():
        raise ValueError("log1p_safe received negative values")
    return np.log1p(values)
