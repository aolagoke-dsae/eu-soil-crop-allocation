"""Map LUCAS land-cover codes to the protocol's crop groups.

Level 2 (the main target) groups the ``LC1`` codes as configured in
``config['target']['level2']``; Level 1 groups those into farming systems.
Codes not listed in any group are excluded, and :func:`excluded_code_counts`
reports how many points each excluded code had.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _invert(groups: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, members in groups.items():
        for member in members:
            if member in mapping:
                raise ValueError(
                    f"'{member}' is assigned to both '{mapping[member]}' and '{group}'"
                )
            mapping[member] = group
    return mapping


def assign_target(df: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    """Add ``crop_group`` (Level 2), ``farming_system`` (Level 1) and ``input_intensity``."""
    code_to_group = _invert(target["level2"])
    group_to_system = _invert(target["level1"])
    group_to_intensity = _invert(target["input_intensity"])

    unassigned = set(target["level2"]) - set(group_to_system)
    if unassigned:
        raise ValueError(f"Level 2 groups without a Level 1 system: {sorted(unassigned)}")
    no_intensity = set(target["level2"]) - set(group_to_intensity)
    if no_intensity:
        raise ValueError(f"Level 2 groups without an input intensity: {sorted(no_intensity)}")

    out = df.copy()
    out["crop_group"] = out["lc1"].map(code_to_group)
    out["farming_system"] = out["crop_group"].map(group_to_system)
    out["input_intensity"] = out["crop_group"].map(group_to_intensity)
    return out


def excluded_code_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Points per land-cover code that maps to no crop group."""
    excluded = df[df["crop_group"].isna()]
    return (
        excluded.groupby(["lc1", "lc1_desc"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values("n_points", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def classes_below_minimum(df: pd.DataFrame, minimum: int) -> pd.Series:
    """Crop groups with fewer than ``minimum`` points (should be empty)."""
    counts = df["crop_group"].value_counts()
    return counts[counts < minimum]
