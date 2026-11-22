"""Phase 6: the single evaluation on the sealed test set (protocol: Validation design).

The frozen final configuration (Stage B: ladder step 5, fixed
hyperparameters) is refitted once on every training point with complete
predictors and scored once on the sealed test points. Before anything is
fitted, the sealed set is checked against the SHA-256 hash committed in
Phase 2, so the score is provably on the set fixed before exploration.

Reported: macro F1 (primary), balanced accuracy, log loss, Brier score, top-2
accuracy and per-class F1, each with a 95% spatial block-bootstrap interval
over the test points, the cross-validated estimate beside it, the
row-normalised confusion matrix, macro F1 per country, and calibration (ECE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from eusoil.models import (
    TARGET,
    _x_columns,
    aligned_proba,
    build_estimator,
    classwise_ece,
    evaluate,
    top_label_ece,
)
from eusoil.precision import block_ids
from eusoil.split import ids_hash


class SealBroken(ValueError):
    """The sealed test set does not match the hash committed in Phase 2."""


def check_seal(splits: pd.DataFrame, expected_hash: str) -> None:
    observed = ids_hash(splits.loc[splits["sealed_test"], "point_id"])
    if observed != expected_hash:
        raise SealBroken(
            f"Sealed test set hash {observed[:12]}... differs from the committed "
            f"{expected_hash[:12]}..."
        )


def fit_final(
    train: pd.DataFrame,
    columns: list[str],
    params: dict[str, Any],
    config: dict[str, Any],
    n_jobs: int = -1,
):
    """The final configuration fitted on all training points."""
    model_name = config["stage_b"]["model"]
    spec = config["stage_a"]["models"][model_name]
    estimator = build_estimator(model_name, columns, spec.get("fixed", {}), int(config["seed"]))
    estimator.set_params(**{f"model__{k}": v for k, v in params.items()})
    if "model__n_jobs" in estimator.get_params():
        estimator.set_params(model__n_jobs=n_jobs)
    return estimator.fit(train[_x_columns(columns)], train[TARGET].to_numpy())


def resample_indices(
    test: pd.DataFrame,
    classes: list[str],
    block_m: float,
    n_boot: int,
    rng: np.random.Generator,
):
    """Block-bootstrap resamples of the test points, as index arrays.

    Yields one array per resample that holds every class, and ``None`` for one
    that lacks a class (it cannot give per-class scores and is dropped). The
    draws depend only on the coordinates, the labels and ``rng``, never on the
    model, so the resamples can be rebuilt without scoring the test set again.
    """
    blocks = block_ids(test["easting"].to_numpy(), test["northing"].to_numpy(), block_m)
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    y = test[TARGET].to_numpy()
    for _ in range(n_boot):
        draw = rng.integers(0, len(members), len(members))
        index = np.concatenate([members[i] for i in draw])
        yield index if len(set(y[index])) == len(classes) else None


def count_usable_resamples(
    test: pd.DataFrame,
    classes: list[str],
    block_m: float,
    n_boot: int,
    rng: np.random.Generator,
) -> int:
    """How many of ``n_boot`` resamples hold every class (see ``resample_indices``)."""
    return sum(i is not None for i in resample_indices(test, classes, block_m, n_boot, rng))


def bootstrap_metrics(
    test: pd.DataFrame,
    proba: np.ndarray,
    classes: list[str],
    block_m: float,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Metrics on block-bootstrap resamples of the test points (one row per resample)."""
    y = test[TARGET].to_numpy()
    rows = [
        evaluate(y[index], proba[index], classes)
        for index in resample_indices(test, classes, block_m, n_boot, rng)
        if index is not None
    ]
    return pd.DataFrame(rows)


def score_test(
    model,
    test: pd.DataFrame,
    columns: list[str],
    classes: list[str],
    block_m: float,
    n_boot: int,
    ece_bins: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """All Phase 6 tables for the fitted model on the sealed test points."""
    proba = aligned_proba(model, test[_x_columns(columns)], classes)
    y = test[TARGET].to_numpy()
    observed = evaluate(y, proba, classes)
    draws = bootstrap_metrics(test, proba, classes, block_m, n_boot, rng)
    metrics = pd.DataFrame(
        [
            {
                "metric": name,
                "test": value,
                "ci95_low": float(np.quantile(draws[name], 0.025)),
                "ci95_high": float(np.quantile(draws[name], 0.975)),
            }
            for name, value in observed.items()
        ]
    )
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    confusion = pd.crosstab(
        pd.Series(y, name="y_true"), pd.Series(pred, name="y_pred"), normalize="index"
    ).reindex(index=classes, columns=classes, fill_value=0.0)
    by_country = []
    for country, idx in test.groupby("country").groups.items():
        rows = test.index.get_indexer(idx)
        present = sorted(set(y[rows]))
        by_country.append(
            {
                "country": country,
                "n_points": len(rows),
                "macro_f1": f1_score(
                    y[rows], pred[rows], labels=present, average="macro", zero_division=0
                ),
            }
        )
    calibration = {"top_label_ece": top_label_ece(y, proba, classes, ece_bins)}
    calibration |= {f"ece__{k}": v for k, v in classwise_ece(y, proba, classes, ece_bins).items()}
    return {
        "metrics": metrics,
        "confusion": confusion,
        "by_country": pd.DataFrame(by_country),
        "calibration": pd.DataFrame([calibration]),
        "n_bootstrap_used": pd.DataFrame(
            [{"n_bootstrap_used": len(draws), "n_test_points": len(test)}]
        ),
    }
