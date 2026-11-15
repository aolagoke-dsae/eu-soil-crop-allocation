"""Exploratory data analysis on the training points (protocol: Exploratory data analysis).

Every function here works on the training points only; :func:`training_only`
is the single way in and refuses any row of the sealed test set.

Outputs are aggregates (medians, quantiles, correlations, variance shares,
gridded medians), never point-level soil values, so they can be published
under the ESDAC licence. Results are exploratory: they can raise questions,
which are recorded as exploratory and never promoted to the confirmatory
hypotheses.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import squareform

LOG = logging.getLogger(__name__)

# Predictors that are 0/1 flags or categories rather than measurements.
BINARY = {"caco3_present", "p_below_lod"}
NOT_CONTINUOUS = BINARY | {"country", "easting", "northing"}


def training_only(table: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Join the split assignment and keep the training points; never the sealed test set."""
    frame = table.merge(splits[["point_id", "sealed_test"]], on="point_id", validate="one_to_one")
    if len(frame) != len(table):
        raise ValueError("Every point in the modelling table needs a split assignment")
    train = frame[~frame["sealed_test"]].drop(columns="sealed_test").reset_index(drop=True)
    assert_no_test_rows(train, splits)
    return train


def assert_no_test_rows(frame: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Raise if any sealed test point is present."""
    sealed = set(splits.loc[splits["sealed_test"], "point_id"])
    leaked = sealed & set(frame["point_id"])
    if leaked:
        raise ValueError(f"{len(leaked)} sealed test points reached the EDA")


def continuous_predictors(config: dict[str, Any]) -> list[str]:
    """Measured predictors of the confirmatory blocks, in block order, plus organic carbon."""
    names = [c for block, cols in config["blocks"].items() if block != "K" for c in cols]
    return [c for c in dict.fromkeys(names) if c not in NOT_CONTINUOUS]


# --- distributions --------------------------------------------------------------


def class_summaries(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median and quartiles of each predictor per crop group (long table)."""
    rows = []
    for group, frame in df.groupby("crop_group"):
        for column in columns:
            values = frame[column].dropna()
            q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else (np.nan,) * 3
            rows.append(
                {
                    "crop_group": group,
                    "variable": column,
                    "n": len(values),
                    "median": q50,
                    "q25": q25,
                    "q75": q75,
                }
            )
    return pd.DataFrame(rows)


# --- pooled vs within-group correlation ----------------------------------------


def _fisher_mean(matrices: list[pd.DataFrame], weights: list[int]) -> pd.DataFrame:
    z = sum(
        w * np.arctanh(m.clip(-0.999999, 0.999999)) for m, w in zip(matrices, weights, strict=True)
    )
    return np.tanh(z / sum(weights))


def within_correlation(
    df: pd.DataFrame, columns: list[str], by: str, min_n: int = 30
) -> pd.DataFrame:
    """Spearman correlation within each level of ``by``, averaged on Fisher's z (weights n - 3)."""
    matrices, weights = [], []
    for _, frame in df.groupby(by):
        if len(frame) < min_n:
            continue
        matrices.append(frame[columns].corr(method="spearman"))
        weights.append(len(frame) - 3)
    return _fisher_mean(matrices, weights)


def correlation_structure(
    df: pd.DataFrame, columns: list[str], simpson_gap: float = 0.3
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Pooled, within-crop-group and within-country matrices, and a pair table.

    A pair is flagged when the pooled correlation and a within-group one
    differ in sign (both at least 0.1 in size) or by more than ``simpson_gap``:
    the pooled association is then partly made by differences between crop
    groups or countries (Simpson's paradox).
    """
    matrices = {
        "pooled": df[columns].corr(method="spearman"),
        "within_crop_group": within_correlation(df, columns, "crop_group"),
        "within_country": within_correlation(df, columns, "country"),
    }
    rows = []
    for a_index, a in enumerate(columns):
        for b in columns[a_index + 1 :]:
            values = {name: float(m.loc[a, b]) for name, m in matrices.items()}
            flags = []
            for name in ("within_crop_group", "within_country"):
                pooled, within = values["pooled"], values[name]
                opposite = (
                    np.sign(pooled) != np.sign(within) and min(abs(pooled), abs(within)) >= 0.1
                )
                if opposite or abs(pooled - within) > simpson_gap:
                    flags.append(name)
            rows.append({"var_a": a, "var_b": b} | values | {"flag": ";".join(flags)})
    return matrices, pd.DataFrame(rows)


# --- variance partitioning ------------------------------------------------------


def icc_random_intercept(values: pd.Series, groups: pd.Series) -> float:
    """Share of variance between groups from a random-intercept model (REML).

    Falls back to the one-way ANOVA estimator if the mixed model does not
    converge; both estimate the same intraclass correlation.
    """
    import statsmodels.formula.api as smf

    data = pd.DataFrame({"y": values, "g": groups}).dropna()
    data["y"] = (data["y"] - data["y"].mean()) / data["y"].std()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = smf.mixedlm("y ~ 1", data, groups=data["g"]).fit(reml=True)
        between = float(fit.cov_re.iloc[0, 0])
        within = float(fit.scale)
    except Exception:  # noqa: BLE001 - any failure falls back to the moment estimator
        return icc_anova(data["y"], data["g"])
    return between / (between + within)


def icc_anova(values: pd.Series, groups: pd.Series) -> float:
    """One-way ANOVA (method-of-moments) intraclass correlation for unbalanced groups."""
    data = pd.DataFrame({"y": values, "g": groups}).dropna()
    n, k = len(data), data["g"].nunique()
    sizes = data.groupby("g").size()
    means = data.groupby("g")["y"].mean()
    grand = data["y"].mean()
    ms_between = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    ms_within = float(((data["y"] - data["g"].map(means)) ** 2).sum() / (n - k))
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    between = max((ms_between - ms_within) / n0, 0.0)
    return between / (between + ms_within)


def variance_partition(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """ICC of each predictor between countries and between NUTS-2 regions."""
    rows = []
    for column in columns:
        rows.append(
            {
                "variable": column,
                "icc_country": icc_random_intercept(df[column], df["country"]),
                "icc_nuts2": icc_random_intercept(df[column], df["nuts2"]),
            }
        )
    return pd.DataFrame(rows)


# --- spatial autocorrelation -----------------------------------------------------


def knn_neighbours(xy: np.ndarray, k: int) -> np.ndarray:
    """Indices of each point's ``k`` nearest neighbours (excluding itself)."""
    _, index = cKDTree(xy).query(xy, k=k + 1)
    return index[:, 1:]


def morans_i(values: np.ndarray, neighbours: np.ndarray) -> float:
    """Global Moran's I with row-standardised k-nearest-neighbour weights.

    With row-standardised weights the sum of weights equals n, so
    I = sum_i z_i * lag_i / sum_i z_i^2, where lag_i is the mean of z over i's
    neighbours.
    """
    z = values - values.mean()
    lag = z[neighbours].mean(axis=1)
    return float((z * lag).sum() / (z**2).sum())


def morans_table(
    df: pd.DataFrame, columns: list[str], k: int, n_permutations: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Moran's I with a permutation p-value for each predictor and class indicator.

    Points with a missing value are dropped for that variable and the
    neighbour graph is rebuilt on the rest.
    """
    xy_all = df[["easting", "northing"]].to_numpy()
    items = [(c, df[c].to_numpy(dtype="float64"), "predictor") for c in columns]
    for group in sorted(df["crop_group"].unique()):
        items.append(
            (group, (df["crop_group"] == group).to_numpy(dtype="float64"), "class indicator")
        )
    rows = []
    for name, values, kind in items:
        keep = np.isfinite(values)
        x, xy = values[keep], xy_all[keep]
        neighbours = knn_neighbours(xy, k)
        observed = morans_i(x, neighbours)
        null = np.array([morans_i(rng.permutation(x), neighbours) for _ in range(n_permutations)])
        p = (1 + (np.abs(null) >= abs(observed)).sum()) / (n_permutations + 1)
        rows.append(
            {
                "variable": name,
                "kind": kind,
                "n": int(keep.sum()),
                "morans_i": observed,
                "expected": -1.0 / (keep.sum() - 1),
                "p_permutation": p,
            }
        )
    return pd.DataFrame(rows)


# --- collinearity and the climate subset -----------------------------------------


def vif_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance inflation factors from the inverse correlation matrix (complete cases)."""
    corr = df[columns].dropna().corr().to_numpy()
    vif = np.diag(np.linalg.inv(corr))
    return pd.DataFrame({"variable": columns, "vif": vif})


def climate_clusters(
    df: pd.DataFrame, columns: list[str], chosen: list[str], threshold: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman clusters of the climate variables and the check of the pre-chosen subset.

    Complete linkage on 1 - |rho| cut at 1 - ``threshold``: inside a cluster
    every pair has |rho| above the threshold. Each cluster's representative is
    a pre-chosen variable if one is in it, otherwise the member with the
    highest mean |rho| to the rest of the cluster.

    Returns the cluster table and a one-row check: whether the pre-chosen
    variables fall in distinct clusters, and the cluster-based set.
    """
    rho = df[columns].corr(method="spearman").abs()
    distance = squareform(1.0 - rho.to_numpy(), checks=False)
    labels = fcluster(linkage(distance, method="complete"), t=1.0 - threshold, criterion="distance")
    table = pd.DataFrame({"variable": columns, "cluster": labels})
    representatives = []
    for _, members in table.groupby("cluster")["variable"]:
        members = list(members)
        pre = [m for m in members if m in chosen]
        if pre:
            rep = pre[0]
        elif len(members) == 1:
            rep = members[0]
        else:
            rep = rho.loc[members, members].mean().idxmax()
        representatives.append(rep)
    table["representative"] = table["variable"].isin(representatives)
    table["pre_chosen"] = table["variable"].isin(chosen)
    chosen_clusters = table.loc[table["pre_chosen"], "cluster"]
    check = pd.DataFrame(
        [
            {
                "n_clusters": int(table["cluster"].nunique()),
                "pre_chosen_in_distinct_clusters": bool(chosen_clusters.is_unique),
                "clusters_without_pre_chosen": int(
                    table["cluster"].nunique() - chosen_clusters.nunique()
                ),
                "cluster_based_set": ";".join(sorted(representatives, key=columns.index)),
                "differs_from_pre_chosen": set(representatives) != set(chosen),
            }
        ]
    )
    return table, check


# --- environmental coverage --------------------------------------------------------


def pca_scores(df: pd.DataFrame, columns: list[str], n_components: int) -> np.ndarray:
    """Scores on the first principal components of the standardised predictors.

    Missing values are replaced by the column median; this is a descriptive
    check, not a model input.
    """
    x = df[columns].astype("float64")
    x = x.fillna(x.median())
    x = (x - x.mean()) / x.std()
    _, _, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
    return x.to_numpy() @ vt[:n_components].T


def coverage_table(
    df: pd.DataFrame, columns: list[str], n_components: int = 3, min_points: int = 10
) -> pd.DataFrame:
    """Share of each country's points inside the convex hull of all other countries.

    The hull is taken in the space of the first ``n_components`` principal
    components of the climate and soil predictors. A low share marks a country
    whose conditions the others do not cover, which leave-one-country-out
    (H3b) will find hard to predict.
    """
    scores = pca_scores(df, columns, n_components)
    country = df["country"].to_numpy()
    rows = []
    for name in sorted(set(country)):
        own = country == name
        if own.sum() < min_points:
            rows.append({"country": name, "n": int(own.sum()), "share_inside_others": np.nan})
            continue
        hull = Delaunay(scores[~own])
        inside = hull.find_simplex(scores[own]) >= 0
        rows.append(
            {"country": name, "n": int(own.sum()), "share_inside_others": float(inside.mean())}
        )
    return pd.DataFrame(rows)


# --- gridded maps -------------------------------------------------------------------


def grid_medians(df: pd.DataFrame, column: str, cell_m: float, min_points: int) -> pd.DataFrame:
    """Median of ``column`` per square cell; cells with fewer than ``min_points`` are dropped.

    The threshold keeps any single point's value from being shown, so the
    map can be published under the ESDAC licence.
    """
    data = df[["easting", "northing", column]].dropna()
    cx = np.floor(data["easting"] / cell_m).astype(int)
    cy = np.floor(data["northing"] / cell_m).astype(int)
    grouped = data.groupby([cx.rename("cx"), cy.rename("cy")])[column]
    out = grouped.agg(["median", "size"]).reset_index()
    out = out[out["size"] >= min_points]
    out["x"] = (out["cx"] + 0.5) * cell_m
    out["y"] = (out["cy"] + 0.5) * cell_m
    return out.rename(columns={"size": "n"})[["x", "y", "median", "n"]]
