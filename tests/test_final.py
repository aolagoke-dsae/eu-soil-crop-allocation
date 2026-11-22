"""Phase 6: the seal check, the final fit and the test-set scoring; control (a)'s chance level."""

import numpy as np
import pandas as pd
import pytest

from eusoil.config import load_config
from eusoil.final import SealBroken, check_seal, fit_final, score_test
from eusoil.sensitivity import country_prior_baseline
from eusoil.split import ids_hash

CLASSES = ["a", "b", "c"]


def test_check_seal_accepts_the_committed_set_and_rejects_any_change():
    splits = pd.DataFrame({"point_id": range(10), "sealed_test": [True, False] * 5})
    committed = ids_hash(splits.loc[splits["sealed_test"], "point_id"])
    check_seal(splits, committed)
    splits.loc[1, "sealed_test"] = True  # one more point in the test set
    with pytest.raises(SealBroken):
        check_seal(splits, committed)


def _points(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 600_000, n)
    label = np.where(x < 200_000, "a", np.where(x < 400_000, "b", "c"))
    flip = rng.random(n) < 0.1
    label[flip] = rng.choice(CLASSES, flip.sum())
    return pd.DataFrame(
        {
            "point_id": np.arange(n) + seed * 10_000,
            "easting": x,
            "northing": rng.uniform(0, 600_000, n),
            "ph_cacl2": rng.normal(6, 1, n),
            "country": np.where(x < 300_000, "AA", "BB"),
            "crop_group": label,
        }
    )


def test_fit_and_score_on_synthetic_data():
    config = load_config()
    config["stage_a"]["models"]["random_forest"]["fixed"]["n_estimators"] = 50
    columns = ["easting", "northing", "ph_cacl2"]
    train, test = _points(600, 1), _points(300, 2)
    model = fit_final(
        train, columns, {"min_samples_leaf": 3, "max_features": 0.6, "max_depth": None}, config, 1
    )
    out = score_test(model, test, columns, CLASSES, 100_000.0, 200, 15, np.random.default_rng(3))
    metrics = out["metrics"].set_index("metric")
    assert metrics.loc["macro_f1", "test"] > 0.7
    assert (
        metrics.loc["macro_f1", "ci95_low"]
        <= metrics.loc["macro_f1", "test"]
        <= metrics.loc["macro_f1", "ci95_high"]
    )
    assert np.allclose(out["confusion"].sum(axis=1), 1.0)
    assert set(out["by_country"]["country"]) == {"AA", "BB"}
    assert 0 <= out["calibration"].loc[0, "top_label_ece"] <= 1


def test_country_prior_baseline_reflects_each_countrys_crop_mix():
    config = load_config()
    config["split"]["outer_repeats"] = 1
    rng = np.random.default_rng(4)
    n = 900
    country = rng.choice(["AA", "BB", "CC"], n)
    mix = {"AA": [0.8, 0.1, 0.1], "BB": [0.1, 0.8, 0.1], "CC": [0.1, 0.1, 0.8]}
    labels = np.array([rng.choice(CLASSES, p=mix[c]) for c in country])
    frame = pd.DataFrame(
        {"country": country, "crop_group": labels, "outer_fold_s1": rng.integers(0, 5, n)}
    )
    assert country_prior_baseline(frame, config) > 0.6  # far above 1/3: country alone predicts well


def test_usable_resample_count_matches_the_scoring_bootstrap_without_a_model():
    """The count rebuilt from labels and seed equals the rows the scoring bootstrap keeps."""
    from eusoil.final import bootstrap_metrics, count_usable_resamples

    test = _points(120, 5)  # few points and large blocks: some resamples lack a class
    rng_proba = np.random.default_rng(6)
    proba = rng_proba.dirichlet(np.ones(len(CLASSES)), len(test))
    draws = bootstrap_metrics(test, proba, CLASSES, 200_000.0, 300, np.random.default_rng(9))
    count = count_usable_resamples(test, CLASSES, 200_000.0, 300, np.random.default_rng(9))
    assert count == len(draws)
    assert 0 < count < 300
