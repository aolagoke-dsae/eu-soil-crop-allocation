"""Stage B: decisions, Holm, the block bootstrap, fixed parameters, block folds, E1 plateau."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from eusoil.config import load_config
from eusoil.precision import block_ids
from eusoil.stage_b import (
    LADDER,
    BlockBootstrap,
    block_folds,
    bootstrap_p,
    corrected_t_test,
    decide,
    estimand_rows,
    final_params,
    final_step,
    holm,
    plateau,
    validation_schemes,
)

CLASSES = ["a", "b", "c"]


def test_ladder_matches_the_protocol():
    assert LADDER[2] == ["G", "C"] and LADDER[5] == ["G", "C", "S_inh", "S_nut"]
    assert LADDER[6][-1] == "K" and set(LADDER) == set(range(1, 7))


@pytest.mark.parametrize(
    "estimate, low, high, two_sided, expected",
    [
        (0.05, 0.03, 0.07, False, "supported"),
        (0.015, 0.001, 0.03, False, "inconclusive"),  # positive but below the SESOI
        (0.005, -0.004, 0.012, False, "negligible"),
        (0.01, -0.01, 0.03, False, "inconclusive"),
        (-0.05, -0.07, -0.03, True, "reversed"),
        (0.002, -0.01, 0.015, True, "negligible"),
        (0.002, -0.01, 0.025, True, "inconclusive"),
    ],
)
def test_decision_rule(estimate, low, high, two_sided, expected):
    assert decide(estimate, low, high, 0.02, two_sided) == expected


def test_holm_is_step_down_and_monotone():
    adjusted = holm({"H1": 0.01, "H2": 0.04, "H3a": 0.03, "H3b": 0.5})
    assert adjusted["H1"] == pytest.approx(0.04)
    assert adjusted["H3a"] == pytest.approx(0.09)
    assert adjusted["H2"] == pytest.approx(0.09)  # 2 x 0.04 = 0.08, raised to keep the order
    assert adjusted["H3b"] == pytest.approx(0.5)


def test_bootstrap_p_is_small_when_all_draws_share_a_sign():
    assert bootstrap_p(np.full(999, 0.03)) == pytest.approx(2 / 1000)
    assert bootstrap_p(np.linspace(-1, 1, 1001)) == pytest.approx(1.0, abs=0.01)


def test_corrected_t_test_is_more_conservative_than_the_naive_one():
    diffs = np.array([0.02, 0.03, 0.025, 0.018, 0.031, 0.022, 0.027, 0.02, 0.024, 0.029])
    t, p = corrected_t_test(diffs, n_train=0.8, n_test=0.2)
    naive = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
    assert 0 < t < naive and 0 < p < 0.05


def _predictions(points, accuracy, seed):
    rng = np.random.default_rng(seed)
    y = points["y"].to_numpy()
    pred = np.where(rng.random(len(y)) < accuracy, y, rng.choice(CLASSES, len(y)))
    p = (pred[:, None] == np.asarray(CLASSES)[None, :]).astype(float)
    table = pd.DataFrame(p, columns=[f"p__{c}" for c in CLASSES])
    table.insert(0, "point_id", points["point_id"].to_numpy())
    table.insert(1, "y_true", y)
    return table


def _points(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "point_id": np.arange(n),
            "easting": rng.uniform(0, 1e6, n),
            "northing": rng.uniform(0, 1e6, n),
            "y": rng.choice(CLASSES, n),
        }
    )


def test_block_bootstrap_observed_score_equals_sklearn_and_resamples_are_paired():
    points = _points()
    good = _predictions(points, 0.8, 1)
    poor = _predictions(points, 0.4, 2)
    boot = BlockBootstrap(points, {"good": [good], "poor": [poor]}, CLASSES, 100_000.0)
    observed = boot.observed()
    expected = f1_score(
        good["y_true"], np.asarray(CLASSES)[good.iloc[:, 2:].to_numpy().argmax(1)], average="macro"
    )
    assert observed["good"] == pytest.approx(expected)
    draws = boot.resample(300, np.random.default_rng(3))
    assert (draws["good"] > draws["poor"]).all()  # same blocks on both sides
    rows = estimand_rows(observed, draws, {"Hx": ("good", "poor", False)}, 0.02, 0.9875)
    assert rows.loc[0, "decision"] == "supported" and rows.loc[0, "decision_ci_low"] > 0


def test_block_bootstrap_averages_repeats():
    points = _points()
    a, b = _predictions(points, 0.9, 4), _predictions(points, 0.5, 5)
    boot = BlockBootstrap(points, {"two": [a, b], "a": [a], "b": [b]}, CLASSES, 100_000.0)
    obs = boot.observed()
    assert obs["two"] == pytest.approx((obs["a"] + obs["b"]) / 2)


def test_final_params_take_medians_and_modes():
    records = pd.DataFrame(
        {
            "best_params": [
                json.dumps({"min_samples_leaf": 3, "max_features": 0.2, "max_depth": None}),
                json.dumps({"min_samples_leaf": 5, "max_features": 0.4, "max_depth": None}),
                json.dumps({"min_samples_leaf": 8, "max_features": 0.3, "max_depth": 20}),
            ]
        }
    )
    params = final_params(records)
    assert params == {"min_samples_leaf": 5, "max_features": 0.3, "max_depth": None}
    assert isinstance(params["min_samples_leaf"], int)


def test_final_step_follows_h3a():
    estimands = pd.DataFrame({"hypothesis": ["H1", "H3a"], "decision": ["supported", "negligible"]})
    assert final_step(estimands) == 5
    estimands.loc[1, "decision"] = "supported"
    assert final_step(estimands) == 6


def test_block_folds_keep_blocks_whole_and_balanced():
    points = _points(2000)
    xy = points[["easting", "northing"]].to_numpy()
    folds = block_folds(xy, 100_000.0, 0.333, 5, np.random.default_rng(6))
    blocks = block_ids(xy[:, 0] + 33_300, xy[:, 1] + 33_300, 100_000.0)
    for block in np.unique(blocks):
        assert len(set(folds[blocks == block])) == 1
    sizes = np.bincount(folds)
    assert len(sizes) == 5 and sizes.min() > 0.6 * sizes.max()


def test_validation_schemes_cover_every_scheme():
    config = load_config()
    data = _points(800).assign(
        country=lambda d: np.where(d["easting"] < 5e5, "AA", "BB"),
        outer_fold_s1=0,
        outer_fold_s2=0,
        outer_fold_s3=0,
    )
    schemes = validation_schemes(data, config)
    assert set(schemes) == {
        "knndm",
        "random",
        "blocks_25km",
        "blocks_50km",
        "blocks_100km",
        "blocks_200km",
        "loco",
    }
    assert len(schemes["blocks_100km"]) == 3 and data["cv_loco"].nunique() == 2
    assert (
        data["cv_random_s1"].value_counts().max() - data["cv_random_s1"].value_counts().min() <= 1
    )


def test_plateau_rule():
    changes = pd.DataFrame(
        {
            "from_km": [25, 50, 100],
            "to_km": [50, 100, 200],
            "estimate": [-0.04, -0.015, -0.005],
            "low": [-0.05, -0.025, -0.012],
            "high": [-0.03, -0.005, 0.003],
        }
    )
    assert plateau(changes, 0.02) == "100 km"  # 50 km fails: its interval reaches -0.025
    changes.loc[2, "low"] = -0.03
    assert plateau(changes, 0.02).startswith("no plateau")
