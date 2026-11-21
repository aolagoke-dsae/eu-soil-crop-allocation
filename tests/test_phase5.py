"""Phase 5: interpretation helpers and the sensitivity runs' data changes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from eusoil.config import load_config
from eusoil.interpret import ale_curve, grouped_importance, kendall_stability, permute_columns
from eusoil.sensitivity import (
    coarser_groups,
    finer_groups,
    impute_below_lod,
    lod_factor,
    permute_within_country,
    registry,
    robustness,
    survey_weights,
    thin,
)

CLASSES = ["a", "b"]


def _fitted(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"signal": rng.normal(size=n), "twin": 0.0, "noise": rng.normal(size=n)})
    x["twin"] = x["signal"] + rng.normal(0, 0.05, n)
    y = np.where(x["signal"] > 0, "b", "a")
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(x, y)
    return model, x, y


def test_permute_columns_shuffles_a_block_jointly():
    x = pd.DataFrame({"u": np.arange(50.0), "v": np.arange(50.0) * 2, "w": np.arange(50.0)})
    out = permute_columns(x, ["u", "v"], np.random.default_rng(1))
    assert (out["v"] == 2 * out["u"]).all()  # rows stay together within the block
    assert not out["u"].equals(x["u"]) and out["w"].equals(x["w"])


def test_grouped_importance_separates_signal_from_noise():
    model, x, y = _fitted()
    groups = {"signal_block": ["signal", "twin"], "noise": ["noise"]}
    out = grouped_importance(model, x, y, groups, CLASSES, 5, np.random.default_rng(2)).set_index(
        "group"
    )
    assert out.loc["signal_block", "drop_macro_f1"] > 0.3
    assert abs(out.loc["noise", "drop_macro_f1"]) < 0.05
    assert {"drop_f1__a", "drop_f1__b"} <= set(out.columns)


def test_ale_rises_for_the_class_the_variable_favours():
    model, x, _ = _fitted()
    grid = np.quantile(x["signal"], np.linspace(0, 1, 11))
    curve = ale_curve(model, x, "signal", CLASSES, grid)
    assert curve["b"].iloc[-1] - curve["b"].iloc[0] > 0.5
    assert np.allclose(curve["a"] + curve["b"], 0, atol=1e-9)  # probabilities sum to one


def test_kendall_stability_is_one_for_identical_rankings():
    rows = [
        {"variable": v, "fold_id": f, "drop_macro_f1": d}
        for f in ("f1", "f2", "f3")
        for v, d in (("p", 0.3), ("q", 0.2), ("r", 0.1))
    ]
    out = kendall_stability(pd.DataFrame(rows))
    assert out.loc[0, "tau_mean"] == pytest.approx(1.0) and out.loc[0, "n_pairs"] == 3


def _soil(n=400, seed=3):
    rng = np.random.default_rng(seed)
    p = np.exp(rng.normal(np.log(20), 1.0, n))
    below = p < 10
    df = pd.DataFrame(
        {
            "p_mg_kg_below_lod": below,
            "p_log": np.where(below, np.log(5.0), np.log(p)),
        }
    )
    for v, col in (("n_g_kg", "n_log"), ("k_mg_kg", "k_log"), ("oc_g_kg", "oc_log")):
        df[f"{v}_below_lod"] = False
        df[col] = 1.0
    return df


def test_lod_factor_moves_only_flagged_values():
    config = load_config()
    df = _soil()
    out = lod_factor(1.0)(df, config)
    assert np.allclose(out.loc[df["p_mg_kg_below_lod"], "p_log"], np.log(10.0))
    assert out.loc[~df["p_mg_kg_below_lod"], "p_log"].equals(
        df.loc[~df["p_mg_kg_below_lod"], "p_log"]
    )


def test_imputation_draws_below_the_limit_and_recovers_the_mean():
    config = load_config()
    df = _soil(n=3000)
    out = impute_below_lod(4)(df, config)
    imputed = out.loc[df["p_mg_kg_below_lod"], "p_log"]
    assert (imputed < np.log(10.0)).all() and imputed.nunique() > 10
    assert out["p_log"].mean() == pytest.approx(np.log(20), abs=0.08)


def test_regroupings():
    df = pd.DataFrame(
        {
            "crop_group": [
                "maize_and_root_crops",
                "maize_and_root_crops",
                "vineyards_and_orchards",
                "vineyards_and_orchards",
                "olive_groves",
                "wheat",
            ],
            "lc1": ["B16", "B22", "B82", "B73", "B81", "B11"],
        }
    )
    assert finer_groups(df).tolist() == [
        "maize",
        "root_crops",
        "vineyards",
        "orchards",
        "olive_groves",
        "wheat",
    ]
    coarse = coarser_groups(df).tolist()
    assert coarse[2:5] == ["permanent_crops"] * 3 and coarse[0] == "maize_and_root_crops"


def test_thinning_keeps_one_point_per_cell():
    df = pd.DataFrame({"point_id": range(6), "easting": [1, 2, 3, 15, 16, 31.0], "northing": 0.0})
    out = thin(10.0, 0)(df, {})
    assert len(out) == 3 and out["point_id"].is_monotonic_increasing


def test_permutation_within_country_keeps_country_label_counts():
    df = pd.DataFrame(
        {"country": ["A"] * 6 + ["B"] * 4, "crop_group": list("aabbcc") + list("ddde")}
    )
    out = permute_within_country(5)(df, {})
    for c in ("A", "B"):
        assert sorted(out.loc[out["country"] == c, "crop_group"]) == sorted(
            df.loc[df["country"] == c, "crop_group"]
        )


def test_survey_weights_match_farmland_shares():
    df = pd.DataFrame({"country": ["A"] * 80 + ["B"] * 20})
    domain = pd.Series({"A": 120, "B": 780})  # area shares 0.2 and 0.8 with the sample
    w = survey_weights(df, domain)
    assert w.mean() == pytest.approx(1.0)
    share_a = w[df["country"] == "A"].sum() / w.sum()
    assert share_a == pytest.approx(0.2)


def test_registry_names_are_unique_and_cover_every_country():
    config = load_config()
    runs = registry(config, ["AT", "DE", "FR"], ["tc9120_bio1", "tc9120_bio12"])
    names = [r.name for r in runs]
    assert len(names) == len(set(names)) and names[0] == "reference"
    assert {"without_AT", "without_DE", "without_FR", "lod_mi", "survey_weights"} <= set(names)


def test_robustness_verdicts():
    summary = pd.DataFrame(
        {
            "run": ["reference", "x", "y", "reference", "x", "y", "z"],
            "area": ["Reference", "A", "A", "Reference", "A", "A", "Negative control"],
            "hypothesis": ["H1", "H1", "H1", "H2", "H2", "H2", "H2"],
            "decision": [
                "supported",
                "supported",
                "supported",
                "reversed",
                "inconclusive",
                "negligible",
                "x",
            ],
        }
    )
    out = robustness(summary).set_index("hypothesis")
    assert out.loc["H1", "verdict"] == "robust"
    assert out.loc["H2", "verdict"] == "fragile" and out.loc["H2", "n_runs"] == 2
