"""The run log, and train/test disjointness under every validation scheme."""

import argparse
import csv

import numpy as np
import pandas as pd
import pytest

from eusoil.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from eusoil.models import FoldTask, run_fold
from eusoil.precision import block_ids
from eusoil.runlog import FIELDS, RunLog, options_of, package_versions
from eusoil.stage_b import validation_schemes

CLASSES = ["a", "b", "c"]


# --- run log -----------------------------------------------------------------


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_log_appends_one_row_per_command_with_its_outcome(tmp_path):
    log_path = tmp_path / "run_log.csv"
    config = load_config()

    def log(command):
        return RunLog(log_path, PROJECT_ROOT, command, {"n_jobs": 2}, DEFAULT_CONFIG, config)

    with log("table"):
        pass
    with pytest.raises(SystemExit), log("final-test"):
        raise SystemExit(f"{PROJECT_ROOT / 'reports' / 'final_test_metrics.csv'} exists")
    with pytest.raises(ValueError), log("eda"):
        raise ValueError("boom")

    rows = _read(log_path)
    assert list(rows[0]) == list(FIELDS)
    status = [r["status"] for r in rows]
    assert status[0] == "ok" and status[2] == "error: ValueError"
    assert status[1].startswith("stopped: reports") and status[1].endswith(
        "final_test_metrics.csv exists"
    )
    assert str(PROJECT_ROOT) not in status[1]
    first = rows[0]
    assert first["options"] == "n_jobs=2" and first["seed"] == str(config["seed"])
    assert len(first["config_sha256"]) == 64
    assert first["git_dirty"] in {"yes", "no", "unknown"}
    assert "numpy==" in first["packages"]


def test_run_log_writes_no_paths_and_skips_the_config_option():
    args = argparse.Namespace(
        command="sensitivity", config="C:/private/analysis.yaml", n_jobs=4, runs=None
    )
    assert options_of(args) == {"n_jobs": 4}
    assert "pandas==" in package_versions(("pandas", "not-a-package"))


# --- disjointness ------------------------------------------------------------


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 1e6, n), rng.uniform(0, 1e6, n)
    return pd.DataFrame(
        {
            "point_id": np.arange(n) + 1000,
            "easting": x,
            "northing": y,
            "country": np.select([x < 3e5, x < 7e5], ["AA", "BB"], "CC"),
            "crop_group": rng.choice(CLASSES, n),
            "outer_fold_s1": rng.integers(0, 5, n),
            "outer_fold_s2": rng.integers(0, 5, n),
            "outer_fold_s3": rng.integers(0, 5, n),
        }
    )


def test_every_scheme_partitions_the_points_and_keeps_groups_on_one_side():
    """No point, block or country is on both the training and the evaluation side."""
    config = load_config()
    data = _data()
    schemes = validation_schemes(data, config)
    for scheme, entries in schemes.items():
        for repeat, column in entries:
            folds = data[column]
            assert folds.notna().all(), (scheme, repeat)
            for fold in folds.unique():
                test_ids = set(data.loc[folds == fold, "point_id"])
                train_ids = set(data.loc[folds != fold, "point_id"])
                assert not test_ids & train_ids
                assert len(test_ids) + len(train_ids) == len(data)
            if scheme.startswith("blocks_"):
                size = float(scheme.removeprefix("blocks_").removesuffix("km")) * 1000.0
                offset = config["stage_b"]["block_offsets"][repeat - 1]
                blocks = block_ids(
                    data["easting"] + offset * size, data["northing"] + offset * size, size
                )
                assert (pd.Series(folds.to_numpy()).groupby(blocks).nunique() == 1).all()
            if scheme == "loco":
                assert (data.groupby("country")[column].nunique() == 1).all()


def test_run_fold_scores_exactly_the_held_out_fold():
    config = load_config()
    data = _data()
    validation_schemes(data, config)
    for column in ("outer_fold_s1", "cv_blocks_100km_s2", "cv_loco"):
        fold = int(data[column].iloc[0])
        record, predictions = run_fold(
            FoldTask("majority", 1, fold),
            data,
            ["easting", "northing"],
            CLASSES,
            data[["easting", "northing"]].to_numpy(),
            config,
            n_jobs=1,
            fold_column=column,
        )
        held_out = set(data.loc[data[column] == fold, "point_id"])
        assert set(predictions["point_id"]) == held_out
        assert record["n_train"] == len(data) - len(held_out)
