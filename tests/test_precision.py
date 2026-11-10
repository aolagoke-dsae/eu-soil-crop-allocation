import numpy as np
import pandas as pd
import pytest

from eusoil.precision import (
    block_ids,
    bootstrap_difference,
    macro_f1_from_confusion,
    precision_check,
    simulate_pair,
)


def test_macro_f1_perfect_and_known_case():
    assert macro_f1_from_confusion(np.diag([5, 3])) == 1.0
    # Class 0: tp 1, fp 1, fn 1 -> F1 0.5; class 1: tp 1, fp 1, fn 1 -> F1 0.5.
    assert macro_f1_from_confusion(np.array([[1, 1], [1, 1]])) == pytest.approx(0.5)


def test_block_ids_group_points_in_the_same_square():
    ids = block_ids(np.array([0.0, 10.0, 60_000.0]), np.array([0.0, 20.0, 0.0]), 50_000.0)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_simulated_models_differ_by_the_sesoi():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 4000)
    blocks = rng.integers(0, 80, 4000)
    pair = simulate_pair(y, blocks, 4, base_accuracy=0.5, sigma=0.5, sesoi=0.02, rng=rng)

    def cm(pred):
        return np.bincount(y * 4 + pred, minlength=16).reshape(4, 4)

    diff = macro_f1_from_confusion(cm(pair.pred_b)) - macro_f1_from_confusion(cm(pair.pred_a))
    assert diff == pytest.approx(0.02, abs=0.002)
    assert np.mean(pair.pred_a == y) == pytest.approx(0.5, abs=0.03)
    boot = bootstrap_difference(pair, blocks, 4, 200, rng)
    assert boot.mean() == pytest.approx(diff, abs=0.01)


def test_precision_check_reports_every_setting():
    rng = np.random.default_rng(2)
    table = pd.DataFrame(
        {
            "crop_group": rng.choice(list("abc"), 1500),
            "easting": rng.uniform(0, 1e6, 1500),
            "northing": rng.uniform(0, 1e6, 1500),
        }
    )
    settings = {
        "block_sizes_km": [100],
        "n_bootstrap": 100,
        "n_replicates": 2,
        "sesoi_macro_f1": 0.02,
        "max_ci_width": 0.04,
    }
    out = precision_check(table, settings, seed=3, sigmas=(0.0, 1.0))
    assert len(out) == 2 and set(out["sigma_block_logit"]) == {0.0, 1.0}
    assert (out["median_ci_width"] > 0).all()
