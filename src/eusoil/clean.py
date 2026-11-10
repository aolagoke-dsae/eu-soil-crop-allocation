"""Apply the protocol's cleaning rules to the parsed soil table.

Rules (protocol: Data quality and cleaning, and Amendment A1):

* keep the standard 0-20 cm samples;
* drop countries listed in ``cleaning.exclude_countries``;
* every measurement below its limit of detection is left-censored at the
  limit, substituted by ``lod_substitution_factor * LOD`` and flagged;
* CaCO3 codes are decoded as absent (0 g/kg) following the readme;
* ``<0.0`` codes become missing and are flagged;
* values outside the plausible ranges are flagged, never removed;
* derived columns (log transforms, presence flags) are added here, so every
  model reads the same definitions.

Rows are only removed through :class:`eusoil.flow.SampleFlow`, so every
removal is counted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eusoil.flow import SampleFlow
from eusoil.ingest.soil import MEASUREMENTS, caco3_from_codes, log1p_safe, parse_measurement


def parse_soil_measurements(df: pd.DataFrame, cleaning: dict[str, Any]) -> pd.DataFrame:
    """Replace the text measurement columns with parsed numbers and flags.

    For each measurement ``m`` with a detection limit, the output holds
    ``m`` (censored values substituted), ``m_below_lod`` and ``m_invalid``.
    CaCO3 becomes ``caco3_g_kg`` and ``caco3_present``.
    """
    out = df.copy()
    lods: dict[str, float] = cleaning["lod"]
    factor: float = cleaning["lod_substitution_factor"]

    for column in MEASUREMENTS:
        if column == "caco3_g_kg":
            decoded = caco3_from_codes(df[column])
            out["caco3_g_kg"] = decoded["value"]
            out["caco3_present"] = decoded["present"]
            continue
        lod = lods.get(column)
        parsed = parse_measurement(df[column], lod)
        value = parsed["value"]
        if lod is not None:
            value = value.mask(parsed["below_lod"], factor * lod)
            out[f"{column}_below_lod"] = parsed["below_lod"]
        out[column] = value
        out[f"{column}_invalid"] = parsed["invalid"]
    return out


def flag_implausible(df: pd.DataFrame, cleaning: dict[str, Any]) -> pd.DataFrame:
    """Add ``implausible_<column>`` flags for values outside the configured ranges."""
    out = df.copy()
    for column, (low, high) in cleaning["plausible"].items():
        values = out[column]
        out[f"implausible_{column}"] = values.notna() & ((values < low) | (values > high))
    out["organic_soil"] = out["oc_g_kg"] > cleaning["organic_soil_oc_g_kg"]
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the transformed predictors named in ``config['blocks']``."""
    out = df.copy()
    out["oc_log"] = np.log(out["oc_g_kg"])
    out["n_log"] = np.log(out["n_g_kg"])
    out["p_log"] = np.log(out["p_mg_kg"])
    out["k_log"] = np.log(out["k_mg_kg"])
    out["caco3_log1p"] = log1p_safe(out["caco3_g_kg"])
    out["p_below_lod"] = out["p_mg_kg_below_lod"]
    return out


def clean_soil(df: pd.DataFrame, cleaning: dict[str, Any], flow: SampleFlow) -> pd.DataFrame:
    """Run the full cleaning sequence and log every row removal in ``flow``."""
    out = flow.apply(
        df,
        df["depth"].eq(cleaning["keep_depth"]),
        step="standard depth",
        reason=f"keep {cleaning['keep_depth']} samples only",
    )
    excluded = set(cleaning["exclude_countries"])
    out = flow.apply(
        out,
        ~out["nuts0"].isin(excluded),
        step="excluded countries",
        reason=f"too few points to model: {', '.join(sorted(excluded))}",
    )
    out = parse_soil_measurements(out, cleaning)
    flow.note(
        out,
        step="detection limits",
        reason="values below their LOD censored and flagged; CaCO3 codes decoded as 0",
    )
    out = flag_implausible(out, cleaning)
    # Values outside a plausible range are reported in the audit, not removed.
    out = add_derived(out)
    return out
