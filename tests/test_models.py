"""Stage A: pipelines, metrics, calibration helpers, the nested run and the selection rule."""

import json

import numpy as np
import pandas as pd
import pytest

from eusoil.config import load_config
from eusoil.models import (
    FoldTask,
    apply_temperature,
    build_estimator,
    complete_cases,
    evaluate,
    fit_temperature,
    load_records,
    predictor_columns,
    run_stage_a,
    search_space,
    select_model,
    summarise,
    top_label_ece,
)

CLASSES = ["a", "b", "c"]


def _data(n=360, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 300_000, n)
    y = rng.uniform(0, 300_000, n)
    label = np.where(x < 100_000, "a", np.where(x < 200_000, "b", "c"))
    flip = rng.random(n) < 0.15
    label[flip] = rng.choice(CLASSES, flip.sum())
    return pd.DataFrame(
        {
            "point_id": np.arange(n),
            "easting": x,
            "northing": y,
            "elevation_m": rng.normal(200, 50, n),
            "slope_deg": rng.uniform(0, 5, n),
            "ph_cacl2": 5 + (label == "c") * 1.5 + rng.normal(0, 0.5, n),
            "p_below_lod": rng.random(n) < 0.1,
            "country": np.where(y < 150_000, "AA", "BB"),
            "crop_group": label,
            "outer_fold_s1": rng.integers(0, 5, n),
        }
    )


def _config(n_trials=3):
    config = load_config()
    config["blocks"] = {
        "G": ["easting", "northing", "elevation_m", "slope_deg"],
        "S": ["ph_cacl2", "p_below_lod"],
        "K": ["country"],
    }
    config["stage_a"]["blocks"] = ["G", "S", "K"]
    config["stage_a"]["n_trials"] = n_trials
    config["stage_a"]["models"]["random_forest"]["fixed"]["n_estimators"] = 30
    config["stage_a"]["models"]["boosting"]["fixed"]["max_iter"] = 50
    config["split"]["k_inner"] = 3
    config["split"]["n_candidates"] = 10
    config["split"]["outer_repeats"] = 1
    return config


def test_predictor_columns_follow_block_order_without_repeats():
    config = load_config()
    columns = predictor_columns(config, ["G", "C", "S_inh", "S_nut", "K"])
    assert columns[:2] == ["easting", "northing"] and columns[-1] == "country"
    assert "oc_log" not in columns and len(columns) == len(set(columns))


def test_complete_cases_reports_the_drop():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, None], "c": [None, 1, 1]})
    out, dropped = complete_cases(df, ["a", "b"])
    assert len(out) == 1 and dropped == 2


@pytest.mark.parametrize(
    "name", ["majority", "stratified", "knn_coordinates", "logistic", "random_forest", "boosting"]
)
def test_every_candidate_fits_and_predicts_probabilities(name):
    config = _config()
    data = _data()
    columns = predictor_columns(config, config["stage_a"]["blocks"])
    spec = config["stage_a"]["models"].get(name, {})
    model = build_estimator(name, columns, spec.get("fixed", {}), seed=1)
    x = data[list(dict.fromkeys(columns + ["easting", "northing"]))]
    model.fit(x, data["crop_group"])
    proba = model.predict_proba(x)
    assert proba.shape == (len(data), 3) and np.allclose(proba.sum(axis=1), 1)


def test_search_space_parses_every_distribution_kind():
    space = search_space(
        {
            "C": {"loguniform": [0.01, 10]},
            "f": {"uniform": [0.1, 0.5]},
            "k": {"randint": [1, 5]},
            "w": {"choice": ["uniform", None]},
        }
    )
    assert set(space) == {"model__C", "model__f", "model__k", "model__w"}
    assert space["model__w"] == ["uniform", None]
    assert 0.01 <= space["model__C"].rvs(random_state=0) <= 10


def test_evaluate_scores_a_perfect_and_a_uniform_prediction():
    y = np.array(["a", "b", "c", "a"])
    perfect = np.eye(3)[[0, 1, 2, 0]]
    scores = evaluate(y, perfect, CLASSES)
    assert scores["macro_f1"] == 1 and scores["brier"] == 0 and scores["top2_accuracy"] == 1
    uniform = evaluate(y, np.full((4, 3), 1 / 3), CLASSES)
    assert uniform["log_loss"] == pytest.approx(np.log(3))


def test_temperature_scaling_repairs_overconfidence():
    rng = np.random.default_rng(2)
    true = rng.integers(0, 3, 3000)
    logits = np.eye(3)[true] * 1.0 + rng.normal(0, 1, (3000, 3))
    calibrated = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    overconfident = apply_temperature(calibrated, 0.4)  # sharpened
    y = np.asarray(CLASSES)[true]
    t = fit_temperature(y, overconfident, CLASSES)
    assert t == pytest.approx(1 / 0.4, rel=0.15)
    before = top_label_ece(y, overconfident, CLASSES, 15)
    after = top_label_ece(y, apply_temperature(overconfident, t), CLASSES, 15)
    assert after < before / 2


def test_one_se_rule_prefers_the_simpler_model_within_one_se():
    summary = pd.DataFrame(
        {
            "model": ["logistic", "random_forest", "boosting"],
            "macro_f1_mean": [0.50, 0.58, 0.60],
            "macro_f1_se": [0.01, 0.01, 0.025],
        }
    )
    choice = select_model(summary, ["logistic", "random_forest", "boosting"])
    assert choice["best_by_mean"] == "boosting" and choice["selected"] == "random_forest"
    summary.loc[2, "macro_f1_se"] = 0.005
    assert (
        select_model(summary, ["logistic", "random_forest", "boosting"])["selected"] == "boosting"
    )


def test_summarise_gives_mean_se_and_interval():
    records = pd.DataFrame({"model": ["m"] * 4, "macro_f1": [0.5, 0.6, 0.55, 0.65]})
    row = summarise(records, ["macro_f1"]).iloc[0]
    assert row["macro_f1_mean"] == pytest.approx(0.575)
    assert row["macro_f1_lo"] < 0.575 < row["macro_f1_hi"]


def test_run_stage_a_is_resumable_and_scores_every_fold(tmp_path):
    config = _config()
    data = _data()
    columns = predictor_columns(config, config["stage_a"]["blocks"])
    domain = data[["easting", "northing"]].to_numpy() + 2000.0
    run_stage_a(
        data, columns, CLASSES, domain, config, ["majority", "logistic"], [1], tmp_path, n_jobs=1
    )
    records = load_records(tmp_path)
    assert len(records) == 10 and set(records["fold"]) == set(range(5))
    logistic = records[records["model"] == "logistic"]
    assert logistic["macro_f1"].mean() > records[records["model"] == "majority"]["macro_f1"].mean()
    assert "C" in json.loads(logistic.iloc[0]["best_params"])
    stamp = (tmp_path / "logistic_s1_f0.json").stat().st_mtime
    run_stage_a(
        data, columns, CLASSES, domain, config, ["majority", "logistic"], [1], tmp_path, n_jobs=1
    )
    assert (tmp_path / "logistic_s1_f0.json").stat().st_mtime == stamp  # not recomputed
    predictions = pd.read_parquet(tmp_path / "logistic_s1_f0.parquet")
    assert set(predictions["point_id"]) == set(data.loc[data["outer_fold_s1"] == 0, "point_id"])


def test_fold_task_names_its_files():
    assert FoldTask("boosting", 2, 4).stem == "boosting_s2_f4"
