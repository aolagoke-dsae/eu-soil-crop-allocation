"""kNNDM, the sealed split, the prediction domain and the indicator variograms."""

import numpy as np
import pandas as pd
import pytest

from eusoil.config import load_config
from eusoil.knndm import (
    candidate_cluster_counts,
    cv_distances,
    knndm,
    merge_clusters,
    nearest_distances,
)
from eusoil.split import build_domain, choose_test_fold, ids_hash, outer_folds, seal_test_set
from eusoil.variogram import (
    bootstrap_block_km,
    class_variograms,
    exponential_model,
    fit_exponential,
    fit_flag,
    indicator_variogram,
    pair_distances,
)


def _grid(n_side: int, spacing: float = 10_000.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2) * spacing
    return g + rng.normal(0, spacing / 10, g.shape)


# --- kNNDM -----------------------------------------------------------------


def test_cv_distances_measure_to_other_folds_only():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    folds = np.array([0, 0, 1, 1])
    assert cv_distances(xy, folds).tolist() == [10.0, 9.0, 9.0, 10.0]


def test_merge_clusters_balances_fold_sizes():
    labels = np.repeat(np.arange(20), np.arange(1, 21))  # cluster sizes 1..20, n = 210
    folds = merge_clusters(labels, 5, np.random.default_rng(1))
    sizes = np.bincount(folds)
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 20
    for cluster in range(20):  # a cluster never splits across folds
        assert len(set(folds[labels == cluster])) == 1


def test_candidate_counts_span_k_to_n():
    q = candidate_cluster_counts(1000, 5, 50)
    assert q[0] == 5 and q[-1] == 1000 and np.all(np.diff(q) > 0)


def test_knndm_picks_random_cv_when_prediction_is_between_samples():
    xy = _grid(30)  # 900 points, 10 km apart
    rng = np.random.default_rng(2)
    # Prediction points scattered among the samples: distances are short.
    targets = xy[rng.choice(len(xy), 400)] + rng.normal(0, 4000, (400, 2))
    g_ij = nearest_distances(targets, xy)
    result = knndm(xy, g_ij, k=5, rng=rng, n_candidates=40)
    assert result.q > 300  # close to one point per cluster, i.e. random CV
    assert result.w <= result.w_random + 1e-9


def test_knndm_picks_clustered_folds_when_prediction_is_far_from_samples():
    xy = _grid(30)
    rng = np.random.default_rng(3)
    # A target distribution of 40-80 km: far beyond the 10 km sample spacing.
    g_ij = rng.uniform(40_000, 80_000, 500)
    result = knndm(xy, g_ij, k=5, rng=rng, n_candidates=40)
    assert result.q < 50
    assert result.w < result.w_random / 2
    assert np.median(result.g_cv) > 3 * np.median(result.g_random)


def test_knndm_respects_the_maximum_fold_share():
    xy = _grid(20)
    g_ij = np.full(100, 150_000.0)  # pushes towards the biggest possible clusters
    result = knndm(xy, g_ij, k=5, rng=np.random.default_rng(4), max_fold_share=0.3)
    assert np.bincount(result.folds).max() / len(xy) <= 0.3


# --- the sealed split and outer folds ---------------------------------------


def _points(n_side=24, seed=5):
    xy = _grid(n_side, seed=seed)
    rng = np.random.default_rng(seed)
    groups = np.array(["a", "b", "c", "d"])
    return pd.DataFrame(
        {
            "point_id": np.arange(1000, 1000 + len(xy)),
            "easting": xy[:, 0],
            "northing": xy[:, 1],
            "crop_group": groups[rng.integers(0, 4, len(xy))],
            "country": np.where(xy[:, 0] < xy[:, 0].mean(), "AA", "BB"),
        }
    )


SETTINGS = {
    "k_seal": 5,
    "max_seal_attempts": 5,
    "k_outer": 5,
    "outer_repeats": 2,
    "n_candidates": 30,
    "max_fold_share": 0.5,
}


def test_seal_is_reproducible_and_records_a_hash():
    points = _points()
    domain = points[["easting", "northing"]].to_numpy() + 3000.0
    first, _, record = seal_test_set(points, domain, SETTINGS, seed=11)
    second, _, again = seal_test_set(points, domain, SETTINGS, seed=11)
    assert first.equals(second) and record["sha256_test_ids"] == again["sha256_test_ids"]
    assert 0.1 < record["test_share"] < 0.3
    test_groups = set(points.loc[first["sealed_test"].to_numpy(), "crop_group"])
    assert test_groups == {"a", "b", "c", "d"}
    assert record["sha256_test_ids"] == ids_hash(points.loc[first["sealed_test"], "point_id"])


def test_outer_folds_cover_training_points_only():
    points = _points()
    domain = points[["easting", "northing"]].to_numpy() + 3000.0
    splits, _, _ = seal_test_set(points, domain, SETTINGS, seed=11)
    out, results = outer_folds(points, splits, domain, SETTINGS, seed=11)
    assert len(results) == 2
    for column in ("outer_fold_s1", "outer_fold_s2"):
        assert out.loc[out["sealed_test"], column].isna().all()
        assert out.loc[~out["sealed_test"], column].between(0, 4).all()


def test_choose_test_fold_requires_every_group():
    folds = np.array([0, 0, 0, 1, 1, 1])
    groups = pd.Series(["a", "b", "a", "a", "a", "a"])
    assert choose_test_fold(folds, groups, np.random.default_rng(0)) == 0
    assert (
        choose_test_fold(folds, pd.Series(["a", "a", "a", "b", "b", "b"]), np.random.default_rng(0))
        is None
    )


def test_ids_hash_ignores_order():
    assert ids_hash(pd.Series([3, 1, 2])) == ids_hash(pd.Series([1, 2, 3]))


def test_domain_applies_the_sample_rules_and_drops_sampled_points():
    config = load_config()
    survey = pd.DataFrame(
        {
            "point_id": [1, 2, 3, 4, 5, 6],
            "nuts0": ["DE", "DE", "MT", "FR", "FR", "FR"],
            "th_lat": ["50.1", "50.2", "35.9", "46.0", "46.1", "46.2"],
            "th_long": ["8.1", "8.2", "14.4", "-1.0", "2.0", "2.1"],
            "lc1": ["B11", "b16", "B11", "E20", "C10", "E30"],
            "lu1": ["U111", "U111", "U111", "U111", "U111", "U111"],
        }
    )
    domain = build_domain(survey, pd.Series([2]), config)
    # 2 is sampled, 3 is in Malta, 5 is woodland, 6 (E30) is in no crop group.
    assert domain["point_id"].tolist() == [1, 4]
    assert domain["lon"].tolist() == [8.1, -1.0]


# --- variograms ---------------------------------------------------------------


def test_indicator_semivariance_is_half_the_share_of_disagreeing_pairs():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    i, j, d = pair_distances(xy, 5.0)
    table = indicator_variogram(i, j, d, np.array([1, 0, 1]), np.array([0.0, 1.5, 2.5]))
    # Lag 1 bin: pairs (0,1) and (1,2), both disagree; lag 2 bin: (0,2) agrees.
    assert table["gamma"].tolist() == [0.5, 0.0]
    assert table["n_pairs"].tolist() == [2, 1]


def test_exponential_fit_recovers_a_known_range():
    lags = np.linspace(5, 300, 30)
    empirical = pd.DataFrame(
        {"lag": lags, "gamma": exponential_model(lags, 0.02, 0.1, 80.0), "n_pairs": 1000}
    )
    fit = fit_exponential(empirical, max_range=600)
    assert fit["range"] == pytest.approx(80.0, rel=0.01)
    assert fit["nugget"] == pytest.approx(0.02, abs=1e-3)


def test_class_variograms_find_longer_ranges_for_broader_patterns():
    xy = _grid(40, spacing=5000.0)  # 1,600 points over 200 km
    broad = np.where(np.sin(xy[:, 0] / 60_000) > 0, "x", "y")
    fine = np.where(np.sin(xy[:, 0] / 8_000) > 0, "x", "y")
    _, broad_fit = class_variograms(xy, pd.Series(broad), max_lag=150_000, bin_width=5000)
    _, fine_fit = class_variograms(xy, pd.Series(fine), max_lag=150_000, bin_width=5000)
    assert broad_fit["range"].median() > 2 * fine_fit["range"].median()


@pytest.mark.parametrize(
    "median, expected", [(10.0, 25.0), (25.0, 25.0), (26.0, 50.0), (140.0, 150.0), (900.0, 200.0)]
)
def test_bootstrap_block_rule(median, expected):
    assert bootstrap_block_km(median) == expected


def test_fit_flags():
    def flag(nugget, sill, range_):
        return fit_flag({"nugget": nugget, "partial_sill": sill, "range": range_}, 600.0, 10.0)

    assert flag(0.0, 0.1, 600.0) == "no_plateau"
    assert flag(0.1, 0.005, 50.0) == "nugget_only"
    assert flag(0.05, 0.01, 2.0) == "below_first_lag"
    assert flag(0.02, 0.1, 80.0) == "ok"
