"""Prediction domain, the sealed test set and the outer folds (protocol Phase 2).

Order of operations, each a CLI command:

1. ``domain``: the 2018 LUCAS points on farmland in the configured crop
   groups, minus the soil-sample points themselves. This is where the model
   is meant to predict (amendment A10).
2. ``seal``: kNNDM with ``k_seal`` folds on all sample points, matched to the
   distances from the domain to the sample. One fold that contains every
   crop group becomes the sealed test set. A SHA-256 hash of its point IDs is
   written to the reports, so the commit that adds it proves the set was
   fixed before any exploration.
3. ``folds``: kNNDM with ``k_outer`` folds on the training points, matched
   to the distances from the domain to the training points, repeated for
   each outer seed.

Point-level split assignments stay in ``data/processed/splits.parquet``
(git-ignored); only counts, quantiles and the hash are reported.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from eusoil.knndm import KnndmResult, knndm, nearest_distances

AGRICULTURAL_LAND_USE = "U111"


def build_domain(
    survey: pd.DataFrame, sample_ids: pd.Series, config: dict[str, Any]
) -> pd.DataFrame:
    """Prediction locations: unsampled 2018 points under the same rules as the sample.

    ``survey`` is the harmonised 2018 table with ``point_id``, ``nuts0``,
    ``th_lat``, ``th_long``, ``lc1`` and ``lu1``. Points keep their planned
    (theoretical) coordinates, the only ones every point has.
    """
    codes = {c for cs in config["target"]["level2"].values() for c in cs}
    excluded = set(config["cleaning"]["exclude_countries"])
    lc1 = survey["lc1"].astype("string").str.strip().str.upper()
    keep = (
        lc1.isin(codes)
        & survey["lu1"].eq(AGRICULTURAL_LAND_USE)
        & ~survey["nuts0"].isin(excluded)
        & ~survey["point_id"].isin(set(sample_ids))
    )
    domain = survey.loc[keep, ["point_id", "nuts0"]].copy()
    domain["lat"] = pd.to_numeric(survey.loc[keep, "th_lat"], errors="coerce")
    domain["lon"] = pd.to_numeric(survey.loc[keep, "th_long"], errors="coerce")
    domain["lc1"] = lc1[keep]
    domain = domain.dropna(subset=["lat", "lon"])
    return domain.reset_index(drop=True)


def ids_hash(point_ids: pd.Series) -> str:
    """SHA-256 of the sorted point IDs, one per line."""
    text = "\n".join(str(i) for i in sorted(int(v) for v in point_ids)) + "\n"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def choose_test_fold(
    folds: np.ndarray, crop_group: pd.Series, rng: np.random.Generator
) -> int | None:
    """A fold that contains every crop group; among those, the one closest to n / k.

    Returns ``None`` when no fold contains every group.
    """
    k = int(folds.max()) + 1
    groups = set(crop_group.unique())
    target = len(folds) / k
    eligible = [f for f in range(k) if set(crop_group[folds == f].unique()) == groups]
    if not eligible:
        return None
    gaps = np.array([abs((folds == f).sum() - target) for f in eligible])
    best = [f for f, g in zip(eligible, gaps, strict=True) if g == gaps.min()]
    return int(rng.choice(best))


def seal_test_set(
    points: pd.DataFrame, domain_xy: np.ndarray, settings: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, KnndmResult, dict[str, Any]]:
    """Assign the sealed test set; return the split table, the kNNDM result and a record.

    ``points`` needs ``point_id``, ``easting``, ``northing``, ``crop_group``
    and ``country``. Retries with later seeds (at most ``max_seal_attempts``)
    if no fold of the chosen partition contains every crop group.
    """
    xy = points[["easting", "northing"]].to_numpy()
    g_ij = nearest_distances(domain_xy, xy)
    tree = linkage(xy, method="ward")
    for attempt in range(settings["max_seal_attempts"]):
        rng = np.random.default_rng(seed + attempt)
        result = knndm(
            xy,
            g_ij,
            k=settings["k_seal"],
            rng=rng,
            n_candidates=settings["n_candidates"],
            max_fold_share=settings["max_fold_share"],
            tree=tree,
        )
        fold = choose_test_fold(result.folds, points["crop_group"].reset_index(drop=True), rng)
        if fold is not None:
            break
    else:
        raise ValueError("No kNNDM fold contained every crop group; see the protocol")

    sealed = result.folds == fold
    splits = pd.DataFrame({"point_id": points["point_id"].to_numpy(), "sealed_test": sealed})
    test_countries = set(points.loc[sealed, "country"])
    record = {
        "seed": seed + attempt,
        "k": settings["k_seal"],
        "q": result.q,
        "w": result.w,
        "w_random_cv": result.w_random,
        "n_points": len(points),
        "n_test": int(sealed.sum()),
        "test_share": float(sealed.mean()),
        "countries_without_test_points": ";".join(sorted(set(points["country"]) - test_countries)),
        "sha256_test_ids": ids_hash(points.loc[sealed, "point_id"]),
    }
    return splits, result, record


def outer_folds(
    points: pd.DataFrame,
    splits: pd.DataFrame,
    domain_xy: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, list[KnndmResult]]:
    """Add ``outer_fold_s1`` ... columns (training points only; test rows stay missing)."""
    frame = points.merge(splits, on="point_id", how="inner", validate="one_to_one")
    train = frame[~frame["sealed_test"]].reset_index(drop=True)
    xy = train[["easting", "northing"]].to_numpy()
    g_ij = nearest_distances(domain_xy, xy)
    tree = linkage(xy, method="ward")
    out = splits.copy()
    results = []
    for repeat in range(1, settings["outer_repeats"] + 1):
        result = knndm(
            xy,
            g_ij,
            k=settings["k_outer"],
            rng=np.random.default_rng(seed + 100 * repeat),
            n_candidates=settings["n_candidates"],
            max_fold_share=settings["max_fold_share"],
            tree=tree,
        )
        column = pd.Series(result.folds, index=train["point_id"].to_numpy(), dtype="Int64")
        out[f"outer_fold_s{repeat}"] = out["point_id"].map(column).astype("Int64")
        results.append(result)
    return out, results


def fold_counts(frame: pd.DataFrame, fold_column: str, by: str) -> pd.DataFrame:
    """Counts per fold and group (aggregate; publishable)."""
    return pd.crosstab(frame[by], frame[fold_column], margins=True, margins_name="total")
