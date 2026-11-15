"""Exploratory analysis helpers: the training-only guard and each statistic."""

import numpy as np
import pandas as pd
import pytest

from eusoil import eda
from eusoil.config import load_config


def _table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    country = np.repeat(["AA", "BB", "CC", "DD"], n // 4)
    shift = pd.Series(country).map({"AA": -2.0, "BB": -0.5, "CC": 0.5, "DD": 2.0}).to_numpy()
    return pd.DataFrame(
        {
            "point_id": np.arange(n),
            "country": country,
            "nuts2": [f"{c}{i % 5}" for c, i in zip(country, range(n), strict=True)],
            "crop_group": rng.choice(["a", "b", "c"], n),
            "easting": rng.uniform(0, 1e6, n),
            "northing": rng.uniform(0, 1e6, n),
            "x": shift + rng.normal(0, 1, n),
            "noise": rng.normal(0, 1, n),
        }
    )


def test_training_only_drops_and_guards_the_sealed_test_set():
    table = _table()
    splits = pd.DataFrame(
        {"point_id": table["point_id"], "sealed_test": table["point_id"] % 5 == 0}
    )
    train = eda.training_only(table, splits)
    assert len(train) == 320 and not (train["point_id"] % 5 == 0).any()
    with pytest.raises(ValueError, match="sealed test points"):
        eda.assert_no_test_rows(table, splits)


def test_training_only_requires_a_split_for_every_point():
    table = _table()
    splits = pd.DataFrame({"point_id": table["point_id"][:-1], "sealed_test": False})
    with pytest.raises(ValueError, match="split assignment"):
        eda.training_only(table, splits)


def test_continuous_predictors_skip_flags_country_and_coordinates():
    columns = eda.continuous_predictors(load_config())
    assert "ph_cacl2" in columns and "oc_log" in columns and "tc9120_bio1" in columns
    assert not {"country", "easting", "northing", "p_below_lod", "caco3_present"} & set(columns)


def test_class_summaries_give_quartiles_per_group():
    df = pd.DataFrame({"crop_group": ["a"] * 5 + ["b"] * 3, "v": [1, 2, 3, 4, 5, 10, 20, 30.0]})
    out = eda.class_summaries(df, ["v"]).set_index("crop_group")
    assert out.loc["a", "median"] == 3 and out.loc["a", "q25"] == 2 and out.loc["b", "n"] == 3


def test_simpson_pattern_is_flagged():
    # Within each group u and v fall together; between groups they rise together.
    rng = np.random.default_rng(1)
    frames = []
    for level, centre in zip("abcd", [0, 5, 10, 15], strict=True):
        u = rng.normal(0, 1, 100)
        frames.append(
            pd.DataFrame(
                {
                    "crop_group": level,
                    "country": "AA",
                    "u": centre + u,
                    "v": centre - u + rng.normal(0, 0.3, 100),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    matrices, pairs = eda.correlation_structure(df, ["u", "v"])
    assert matrices["pooled"].loc["u", "v"] > 0.5
    assert matrices["within_crop_group"].loc["u", "v"] < -0.5
    assert "within_crop_group" in pairs.loc[0, "flag"]


def test_icc_is_high_for_a_country_effect_and_low_for_noise():
    df = _table(n=800)
    assert eda.icc_random_intercept(df["x"], df["country"]) > 0.5
    assert eda.icc_random_intercept(df["noise"], df["country"]) < 0.05
    # The moment estimator agrees with the mixed model on the same data.
    assert eda.icc_anova(df["x"], df["country"]) == pytest.approx(
        eda.icc_random_intercept(df["x"], df["country"]), abs=0.05
    )


def test_morans_i_detects_a_smooth_field_but_not_noise():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 100, (500, 2))
    neighbours = eda.knn_neighbours(xy, 8)
    smooth = np.sin(xy[:, 0] / 15) + np.cos(xy[:, 1] / 15)
    assert eda.morans_i(smooth, neighbours) > 0.7
    assert abs(eda.morans_i(rng.normal(size=500), neighbours)) < 0.1


def test_morans_table_covers_predictors_and_class_indicators():
    df = _table(n=200)
    out = eda.morans_table(df, ["x", "noise"], k=6, n_permutations=19, rng=np.random.default_rng(0))
    assert set(out["kind"]) == {"predictor", "class indicator"}
    assert len(out) == 2 + 3 and out["p_permutation"].between(0, 1).all()


def test_vif_is_large_for_near_duplicate_columns():
    rng = np.random.default_rng(3)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.05, 300), "c": rng.normal(size=300)})
    vif = eda.vif_table(df, ["a", "b", "c"]).set_index("variable")["vif"]
    assert vif["a"] > 50 and vif["c"] < 1.2


def test_climate_clusters_keep_pre_chosen_representatives():
    rng = np.random.default_rng(4)
    base1, base2 = rng.normal(size=500), rng.normal(size=500)
    df = pd.DataFrame(
        {
            "c1": base1,
            "c2": base1 + rng.normal(0, 0.1, 500),  # same cluster as c1
            "c3": base2,
            "c4": rng.normal(size=500),  # its own cluster, not pre-chosen
        }
    )
    table, check = eda.climate_clusters(df, ["c1", "c2", "c3", "c4"], chosen=["c2", "c3"])
    assert check.loc[0, "n_clusters"] == 3
    assert check.loc[0, "pre_chosen_in_distinct_clusters"]
    assert check.loc[0, "cluster_based_set"] == "c2;c3;c4"
    assert check.loc[0, "differs_from_pre_chosen"]
    assert table.set_index("variable").loc["c1", "representative"] == False  # noqa: E712


def test_coverage_flags_a_country_outside_the_others():
    rng = np.random.default_rng(5)
    inside = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=["p", "q", "r"])
    inside["country"] = rng.choice(["AA", "BB", "CC"], 300)
    outside = pd.DataFrame(rng.normal(0, 0.3, (30, 3)) + 6, columns=["p", "q", "r"])
    outside["country"] = "ZZ"
    out = eda.coverage_table(pd.concat([inside, outside], ignore_index=True), ["p", "q", "r"])
    shares = out.set_index("country")["share_inside_others"]
    assert shares["ZZ"] < 0.1 and shares["AA"] > 0.5


def test_grid_medians_hide_sparse_cells():
    df = pd.DataFrame(
        {
            "easting": [10, 20, 30, 40, 50, 160],
            "northing": [10, 10, 10, 10, 10, 10],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 99.0],
        }
    )
    out = eda.grid_medians(df, "v", cell_m=100, min_points=5)
    assert len(out) == 1 and out.iloc[0]["median"] == 3.0 and out.iloc[0]["n"] == 5
