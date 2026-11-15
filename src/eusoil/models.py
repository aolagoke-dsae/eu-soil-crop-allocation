"""Stage A model selection (protocol: Modelling and model selection).

One algorithm is chosen on the full predictor set before any hypothesis is
tested. Every candidate is tuned with the same budget on inner kNNDM folds
and scored on the outer kNNDM folds made in Phase 2, for every repeat:

* baselines without tuning: majority class, stratified random;
* k-nearest neighbours on coordinates only (the spatial baseline);
* multinomial logistic regression on natural-spline bases, L2-penalised;
* random forest;
* gradient boosting (scikit-learn ``HistGradientBoostingClassifier``).

Every preprocessing step (scaling, spline bases, one-hot encoding, class
weights) sits inside a scikit-learn ``Pipeline`` fitted on the training fold
only. Points with a missing value in the predictors are dropped (the
protocol's primary complete-case rule), and the number dropped is reported.

The selection rule: the best mean outer-fold macro F1 wins, unless a simpler
model is within one standard error of it (logistic < random forest <
boosting). The standard error is that of the mean over all outer folds and
repeats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from eusoil.knndm import knndm, nearest_distances

LOG = logging.getLogger(__name__)

BINARY = {"caco3_present", "p_below_lod"}
CATEGORICAL = {"country"}
COORDINATES = ["easting", "northing"]
TARGET = "crop_group"
UNTUNED = ("majority", "stratified")
TUNED = ("knn_coordinates", "logistic", "random_forest", "boosting")


# --- predictors and preprocessing ----------------------------------------------


def predictor_columns(config: dict[str, Any], blocks: list[str]) -> list[str]:
    """Columns of the named blocks, in block order, without duplicates."""
    names = [c for block in blocks for c in config["blocks"][block]]
    return list(dict.fromkeys(names))


def complete_cases(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Rows with every predictor present, and how many were dropped."""
    keep = df[columns].notna().all(axis=1)
    return df[keep].reset_index(drop=True), int((~keep).sum())


def _x_columns(columns: list[str]) -> list[str]:
    """Predictors plus the coordinates the spatial baseline needs, without repeats."""
    return list(dict.fromkeys(columns + COORDINATES))


def _split_columns(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    categorical = [c for c in columns if c in CATEGORICAL]
    binary = [c for c in columns if c in BINARY]
    continuous = [c for c in columns if c not in CATEGORICAL | BINARY]
    return continuous, binary, categorical


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_estimator(name: str, columns: list[str], fixed: dict[str, Any], seed: int) -> Pipeline:
    """The candidate ``name`` as a pipeline whose preprocessing is fitted per fold."""
    continuous, binary, categorical = _split_columns(columns)
    if name in UNTUNED:
        strategy = "most_frequent" if name == "majority" else "stratified"
        return Pipeline(
            [
                ("prep", ColumnTransformer([("keep", "passthrough", COORDINATES)])),
                ("model", DummyClassifier(strategy=strategy, random_state=seed)),
            ]
        )
    if name == "knn_coordinates":
        return Pipeline(
            [
                ("prep", ColumnTransformer([("xy", "passthrough", COORDINATES)])),
                ("model", KNeighborsClassifier()),
            ]
        )
    if name == "logistic":
        spline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=int(fixed.get("n_knots", 3)), degree=3, extrapolation="linear"
                    ),
                ),
            ]
        )
        prep = ColumnTransformer(
            [("spline", spline, continuous), ("bin", "passthrough", binary)]
            + ([("cat", _one_hot(), categorical)] if categorical else [])
        )
        model = LogisticRegression(class_weight="balanced", max_iter=5000)
        return Pipeline([("prep", prep), ("model", model)])

    prep = ColumnTransformer(
        [("num", "passthrough", continuous + binary)]
        + ([("cat", _one_hot(), categorical)] if categorical else [])
    )
    if name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=int(fixed.get("n_estimators", 400)),
            n_jobs=1,
            random_state=seed,
        )
    elif name == "boosting":
        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            **{k: v for k, v in fixed.items()},
        )
    else:
        raise ValueError(f"Unknown model {name!r}")
    return Pipeline([("prep", prep), ("model", model)])


def search_space(spec: dict[str, Any]) -> dict[str, Any]:
    """scikit-learn distributions from the config, keyed ``model__<parameter>``."""
    space = {}
    for parameter, definition in spec.items():
        ((kind, args),) = definition.items()
        if kind == "loguniform":
            dist = stats.loguniform(*args)
        elif kind == "uniform":
            dist = stats.uniform(*args)
        elif kind == "randint":
            dist = stats.randint(*args)
        elif kind == "choice":
            dist = list(args)
        else:
            raise ValueError(f"Unknown distribution {kind!r} for {parameter}")
        space[f"model__{parameter}"] = dist
    return space


# --- metrics --------------------------------------------------------------------


def aligned_proba(model, x: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Predicted probabilities with columns in ``classes`` order (zeros for unseen)."""
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)))
    for j, label in enumerate(model.classes_):
        out[:, classes.index(label)] = proba[:, j]
    return out


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Primary and secondary metrics, plus per-class F1."""
    y_pred = np.asarray(classes)[proba.argmax(axis=1)]
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    clipped = np.clip(proba, 1e-15, 1)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, clipped, labels=classes),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "top2_accuracy": top_k_accuracy_score(y_true, proba, k=2, labels=classes),
    }
    per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    out |= {f"f1__{c}": float(v) for c, v in zip(classes, per_class, strict=True)}
    return out


# --- calibration ------------------------------------------------------------------


def top_label_ece(y_true: np.ndarray, proba: np.ndarray, classes: list[str], bins: int) -> float:
    """Expected calibration error of the top predicted class (equal-width bins)."""
    confidence = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classwise_ece(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str], bins: int
) -> dict[str, float]:
    """One-vs-rest ECE per class."""
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for j, label in enumerate(classes):
        p = proba[:, j]
        hit = (np.asarray(y_true) == label).astype(float)
        index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        ece = 0.0
        for b in range(bins):
            mask = index == b
            if mask.any():
                ece += mask.mean() * abs(hit[mask].mean() - p[mask].mean())
        out[label] = float(ece)
    return out


def fit_temperature(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    """Temperature T minimising log loss of p^(1/T), renormalised."""
    log_p = np.log(np.clip(proba, 1e-15, 1))

    def loss(t: float) -> float:
        return log_loss(y_true, apply_temperature_log(log_p, t), labels=classes)

    return float(minimize_scalar(loss, bounds=(0.2, 5.0), method="bounded").x)


def apply_temperature_log(log_p: np.ndarray, t: float) -> np.ndarray:
    scaled = log_p / t
    scaled -= scaled.max(axis=1, keepdims=True)
    p = np.exp(scaled)
    return p / p.sum(axis=1, keepdims=True)


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    return apply_temperature_log(np.log(np.clip(proba, 1e-15, 1)), t)


# --- the nested run -----------------------------------------------------------------


@dataclass
class FoldTask:
    model: str
    repeat: int
    fold: int

    @property
    def stem(self) -> str:
        return f"{self.model}_s{self.repeat}_f{self.fold}"


def inner_splits(
    train: pd.DataFrame, domain_xy: np.ndarray, settings: dict[str, Any], seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """kNNDM folds inside an outer training set, matched to the domain's distances."""
    xy = train[COORDINATES].to_numpy()
    result = knndm(
        xy,
        nearest_distances(domain_xy, xy),
        k=settings["k_inner"],
        rng=np.random.default_rng(seed),
        n_candidates=settings["n_candidates"],
        max_fold_share=settings["max_fold_share"],
    )
    folds = result.folds
    return [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in np.unique(folds)]


def run_fold(
    task: FoldTask,
    data: pd.DataFrame,
    columns: list[str],
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    n_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune on inner folds, fit, and score one outer fold of one repeat."""
    settings = config["stage_a"]
    seed = int(config["seed"]) + 1000 * task.repeat + task.fold
    fold_col = f"outer_fold_s{task.repeat}"
    test_mask = data[fold_col].eq(task.fold).to_numpy()
    train, test = data[~test_mask].reset_index(drop=True), data[test_mask].reset_index(drop=True)
    x_train, y_train = train[_x_columns(columns)], train[TARGET].to_numpy()
    x_test, y_test = test[_x_columns(columns)], test[TARGET].to_numpy()

    spec = settings["models"].get(task.model, {})
    estimator = build_estimator(task.model, columns, spec.get("fixed", {}), seed)
    record: dict[str, Any] = {
        "model": task.model,
        "repeat": task.repeat,
        "fold": task.fold,
        "n_train": len(train),
        "n_test": len(test),
    }
    if task.model in UNTUNED:
        fitted = clone(estimator).fit(x_train, y_train)
        record |= {"best_params": "{}", "inner_score": np.nan}
    else:
        search = RandomizedSearchCV(
            estimator,
            search_space(spec["search"]),
            n_iter=settings["n_trials"],
            scoring=settings["tuning_metric"],
            cv=inner_splits(train, domain_xy, config["split"], seed),
            refit=True,
            random_state=seed,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        fitted = search.best_estimator_
        params = {k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()}
        record |= {
            "best_params": json.dumps(params, sort_keys=True),
            "inner_score": search.best_score_,
        }

    proba = aligned_proba(fitted, x_test, classes)
    record |= evaluate(y_test, proba, classes)
    predictions = pd.DataFrame(proba, columns=[f"p__{c}" for c in classes])
    predictions.insert(0, "point_id", test["point_id"].to_numpy())
    predictions.insert(1, "y_true", y_test)
    predictions.insert(2, "repeat", task.repeat)
    predictions.insert(3, "fold", task.fold)
    predictions.insert(4, "model", task.model)
    return record, predictions


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def run_stage_a(
    data: pd.DataFrame,
    columns: list[str],
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    models: list[str],
    repeats: list[int],
    checkpoint: Path,
    n_jobs: int = -1,
) -> None:
    """Run every (model, repeat, fold) not yet in ``checkpoint``; safe to interrupt and resume."""
    checkpoint.mkdir(parents=True, exist_ok=True)
    tasks = [
        FoldTask(model, repeat, fold)
        for model in models
        for repeat in repeats
        for fold in range(config["split"]["k_outer"])
    ]
    for number, task in enumerate(tasks, start=1):
        done = checkpoint / f"{task.stem}.json"
        if done.exists():
            continue
        jobs = 1 if task.model == "boosting" else n_jobs  # boosting threads internally
        record, predictions = run_fold(task, data, columns, classes, domain_xy, config, jobs)
        predictions.to_parquet(checkpoint / f"{task.stem}.parquet", index=False)
        done.write_text(json.dumps(record, default=_plain, indent=1))
        LOG.info(
            "stage A %d/%d %s: macro F1 %.3f, log loss %.3f",
            number,
            len(tasks),
            task.stem,
            record["macro_f1"],
            record["log_loss"],
        )


# --- summary and selection ------------------------------------------------------------


def load_records(checkpoint: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(checkpoint.glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"No Stage A results in {checkpoint}")
    return pd.DataFrame(rows)


def summarise(records: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean, standard error and 95% t-interval of each metric per model over folds and repeats."""
    rows = []
    for model, frame in records.groupby("model"):
        row = {"model": model, "n_folds": len(frame)}
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
            half = stats.t.ppf(0.975, len(values) - 1) * se
            row |= {
                f"{metric}_mean": mean,
                f"{metric}_se": se,
                f"{metric}_lo": mean - half,
                f"{metric}_hi": mean + half,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, order: list[str]) -> dict[str, Any]:
    """The one-SE rule over the eligible candidates, simplest first."""
    eligible = summary[summary["model"].isin(order)].set_index("model")
    missing = [m for m in order if m not in eligible.index]
    if missing:
        raise ValueError(f"Stage A results missing for {missing}")
    best = eligible["macro_f1_mean"].idxmax()
    threshold = eligible.loc[best, "macro_f1_mean"] - eligible.loc[best, "macro_f1_se"]
    chosen = next(m for m in order if eligible.loc[m, "macro_f1_mean"] >= threshold)
    return {
        "best_by_mean": best,
        "best_mean_macro_f1": float(eligible.loc[best, "macro_f1_mean"]),
        "best_se": float(eligible.loc[best, "macro_f1_se"]),
        "one_se_threshold": float(threshold),
        "selected": chosen,
        "selected_mean_macro_f1": float(eligible.loc[chosen, "macro_f1_mean"]),
        "rule": "simplest model within one SE of the best mean outer-fold macro F1",
    }


def temperature_for_fold(
    task: FoldTask,
    params: dict[str, Any],
    data: pd.DataFrame,
    columns: list[str],
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Temperature fitted on inner out-of-fold probabilities of the tuned model.

    The outer test fold is never used: the tuned configuration is refitted on
    each inner training set, the inner validation folds are predicted, and the
    temperature that minimises their log loss is returned.
    """
    seed = int(config["seed"]) + 1000 * task.repeat + task.fold
    fold_col = f"outer_fold_s{task.repeat}"
    train = data[~data[fold_col].eq(task.fold)].reset_index(drop=True)
    x, y = train[_x_columns(columns)], train[TARGET].to_numpy()
    spec = config["stage_a"]["models"][task.model]
    estimator = build_estimator(task.model, columns, spec.get("fixed", {}), seed)
    estimator.set_params(**{f"model__{k}": v for k, v in params.items()})
    oof = np.zeros((len(train), len(classes)))
    for fit_idx, val_idx in inner_splits(train, domain_xy, config["split"], seed):
        model = clone(estimator).fit(x.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = aligned_proba(model, x.iloc[val_idx], classes)
    return fit_temperature(y, oof, classes)


def reliability_table(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str], bins: int
) -> pd.DataFrame:
    """Observed frequency against mean predicted probability, per class and bin."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for j, label in enumerate(classes):
        p = proba[:, j]
        hit = (np.asarray(y_true) == label).astype(float)
        index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = index == b
            if mask.any():
                rows.append(
                    {
                        "crop_group": label,
                        "bin": b,
                        "mean_predicted": float(p[mask].mean()),
                        "observed": float(hit[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)
