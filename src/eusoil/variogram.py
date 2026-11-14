"""Indicator variograms of the crop groups and the bootstrap block size.

The protocol (Validation design, "Spatial blocks") sets the block size of
every bootstrap from the spatial dependence of the target: the smallest
multiple of 25 km at or above the median variogram range of the class
indicators, capped at 200 km. The variograms are computed on the training
points only, after the test set is sealed.

For each crop group c, the indicator z = 1 if a point belongs to c and 0
otherwise. The empirical semivariance in a distance bin is half the mean
squared difference over the point pairs in that bin, which for an indicator
is half the share of pairs that disagree. An exponential model with nugget,

    gamma(h) = nugget + sill * (1 - exp(-3 h / range)),

is fitted by weighted least squares with the pair counts as weights (Cressie,
1985). ``range`` is the practical range, at which the model reaches 95% of
its sill.

Each fit carries a flag, because a single number hides two common failures:

* ``no_plateau``: the range reached its upper bound; the semivariance keeps
  rising over the lags used (a regional trend), so the range is a lower bound.
* ``nugget_only``: the spatially structured part is under 10% of the total
  sill; the class shows little spatial dependence and its range is not
  meaningful.
* ``below_first_lag``: the range is shorter than one distance bin, so the
  bins cannot resolve it; the class behaves as spatially independent at the
  sample spacing.

The block size follows the pre-registered rule (median over all classes).
The median over the classes flagged ``ok`` is reported beside it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


def pair_distances(
    xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index pairs (i < j) closer than ``max_distance`` and their distances."""
    pairs = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), max_distance, output_type="ndarray")
    keep = pairs["i"] < pairs["j"]
    return pairs["i"][keep], pairs["j"][keep], pairs["v"][keep]


def indicator_variogram(
    i: np.ndarray, j: np.ndarray, d: np.ndarray, indicator: np.ndarray, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Empirical semivariance of a 0/1 indicator per distance bin."""
    disagree = (indicator[i] != indicator[j]).astype("float64")
    bins = np.digitize(d, bin_edges) - 1
    valid = (bins >= 0) & (bins < len(bin_edges) - 1)
    counts = np.bincount(bins[valid], minlength=len(bin_edges) - 1)
    sums = np.bincount(bins[valid], weights=disagree[valid], minlength=len(bin_edges) - 1)
    mean_d = np.bincount(bins[valid], weights=d[valid], minlength=len(bin_edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sums / counts
        lag = mean_d / counts
    return pd.DataFrame({"lag": lag, "gamma": gamma, "n_pairs": counts})


def exponential_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    return nugget + sill * (1.0 - np.exp(-3.0 * h / range_))


def fit_exponential(empirical: pd.DataFrame, max_range: float) -> dict[str, float]:
    """Fit nugget, partial sill and practical range by pair-count-weighted least squares."""
    data = empirical[empirical["n_pairs"] > 0]
    if len(data) < 4:
        raise ValueError("Too few non-empty distance bins to fit a variogram")
    h, gamma, n = data["lag"].to_numpy(), data["gamma"].to_numpy(), data["n_pairs"].to_numpy()
    top = float(gamma.max())
    start = [float(gamma[0]), max(top - float(gamma[0]), 1e-6), float(h[len(h) // 3])]
    params, _ = curve_fit(
        exponential_model,
        h,
        gamma,
        p0=start,
        sigma=1.0 / np.sqrt(n),
        bounds=([0.0, 0.0, float(h[0]) / 10], [top * 2, top * 2, max_range]),
        maxfev=20000,
    )
    nugget, sill, range_ = (float(v) for v in params)
    return {"nugget": nugget, "partial_sill": sill, "range": range_}


def fit_flag(model: dict[str, float], max_range: float, bin_width: float) -> str:
    """``no_plateau``, ``nugget_only``, ``below_first_lag`` or ``ok`` (module docstring)."""
    total = model["nugget"] + model["partial_sill"]
    if model["range"] >= 0.99 * max_range:
        return "no_plateau"
    if total <= 0 or model["partial_sill"] / total < 0.10:
        return "nugget_only"
    if model["range"] < bin_width:
        return "below_first_lag"
    return "ok"


def class_variograms(
    xy: np.ndarray, classes: pd.Series, max_lag: float, bin_width: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical variograms (long table) and fitted models (one row per class).

    Distances are in the units of ``xy``. The fitted range is capped at
    twice ``max_lag``; a range at the cap means no plateau within the lags.
    """
    i, j, d = pair_distances(xy, max_lag)
    edges = np.arange(0.0, max_lag + bin_width, bin_width)
    empirical, fitted = [], []
    for name in sorted(classes.unique()):
        indicator = classes.to_numpy() == name
        table = indicator_variogram(i, j, d, indicator, edges)
        empirical.append(table.assign(crop_group=name))
        model = fit_exponential(table, max_range=2 * max_lag)
        fitted.append(
            {"crop_group": name, "share": float(indicator.mean())}
            | model
            | {"fit_flag": fit_flag(model, max_range=2 * max_lag, bin_width=bin_width)}
        )
    return pd.concat(empirical, ignore_index=True), pd.DataFrame(fitted)


def bootstrap_block_km(
    median_range_km: float, step_km: float = 25.0, cap_km: float = 200.0
) -> float:
    """Smallest multiple of ``step_km`` at or above the median range, capped."""
    if not np.isfinite(median_range_km) or median_range_km <= 0:
        raise ValueError("The median range must be a positive number")
    return float(min(cap_km, step_km * max(1, math.ceil(median_range_km / step_km))))
