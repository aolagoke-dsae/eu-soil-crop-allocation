"""Phase 5 interpretation of the final configuration (protocol: Interpretation).

The final configuration (Stage B: random forest, ladder step 5, fixed
hyperparameters) is refitted on the training part of every outer kNNDM fold
and repeat; every statistic below is computed on that fold's held-out points
and summarised over the 15 folds with a 95% t-interval:

* grouped permutation importance per predictor block (whole blocks shuffled
  together, 30 shuffles), overall and per crop group, as the drop in macro F1
  and in each class's F1;
* permutation importance per variable (10 shuffles) and the stability of its
  ranking across folds (Kendall's tau);
* accumulated local effects (ALE) of pH, P, K, annual precipitation and
  annual mean temperature on each class's predicted probability, on quantile
  bins between the 1st and 99th percentiles (never the minimum or maximum);
* TreeSHAP mean absolute contributions on a subsample (exploratory);
* a pure-noise variable added to a second fit as a negative control: its
  importance must be near zero.

Error analysis uses the Stage B kNNDM predictions of the same configuration:
the row-normalised confusion matrix, macro F1 per country and per NUTS-2
region (regions with at least ``min_points`` points), and Moran's I of the
misclassification indicator.

Everything here describes the fitted model, not causal effects of soil or
climate on farmers' choices.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from eusoil.eda import knn_neighbours, morans_i
from eusoil.models import COORDINATES, TARGET, FoldTask, _x_columns, aligned_proba, run_fold

LOG = logging.getLogger(__name__)

NOISE = "noise_control"


def _macro_and_class_f1(
    y: np.ndarray, proba: np.ndarray, classes: list[str]
) -> tuple[float, np.ndarray]:
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    per_class = f1_score(y, pred, labels=classes, average=None, zero_division=0)
    return float(per_class.mean()), per_class


def permute_columns(x: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle ``columns`` jointly: one row permutation applied to all of them."""
    out = x.copy()
    order = rng.permutation(len(x))
    out[columns] = x[columns].to_numpy()[order]
    return out


def grouped_importance(
    model,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    classes: list[str],
    n_repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Mean drop in macro F1 and in each class's F1 when a group is shuffled."""
    base_macro, base_class = _macro_and_class_f1(y, aligned_proba(model, x, classes), classes)
    rows = []
    for name, columns in groups.items():
        drops_macro, drops_class = [], []
        for _ in range(n_repeats):
            shuffled = permute_columns(x, columns, rng)
            macro, per_class = _macro_and_class_f1(
                y, aligned_proba(model, shuffled, classes), classes
            )
            drops_macro.append(base_macro - macro)
            drops_class.append(base_class - per_class)
        row = {"group": name, "drop_macro_f1": float(np.mean(drops_macro))}
        row |= {
            f"drop_f1__{c}": float(v)
            for c, v in zip(classes, np.mean(drops_class, axis=0), strict=True)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def ale_grid(values: pd.Series, n_bins: int, quantile_range: list[float]) -> np.ndarray:
    """ALE bin edges: quantiles of the training data between two inner quantiles.

    The edges are published with the curves, so they never include the
    minimum or the maximum, each of which is one point's measured value
    (protocol, disclosure control; amendment A21). Points beyond the outer
    edges fall in the first or last bin.
    """
    low, high = quantile_range
    if not 0.0 < low < high < 1.0:
        raise ValueError(
            f"ale_quantile_range must lie strictly inside (0, 1), got {quantile_range}"
        )
    return np.unique(np.quantile(values.dropna(), np.linspace(low, high, n_bins + 1)))


def ale_curve(
    model,
    x: pd.DataFrame,
    column: str,
    classes: list[str],
    grid: np.ndarray,
) -> pd.DataFrame:
    """First-order ALE of every class probability along ``column`` (Apley and Zhu, 2020).

    ``grid`` holds the bin edges (quantiles of the training data). Within each
    bin, the points in it are moved to the lower and the upper edge and the
    mean change in predicted probability is accumulated; the curve is then
    centred to mean zero over the points.
    """
    values = x[column].to_numpy()
    bins = np.clip(np.searchsorted(grid, values, side="left") - 1, 0, len(grid) - 2)
    effects = np.zeros((len(grid) - 1, len(classes)))
    counts = np.zeros(len(grid) - 1)
    for b in range(len(grid) - 1):
        members = bins == b
        if not members.any():
            continue
        low, high = x[members].copy(), x[members].copy()
        low[column], high[column] = grid[b], grid[b + 1]
        diff = aligned_proba(model, high, classes) - aligned_proba(model, low, classes)
        effects[b] = diff.mean(axis=0)
        counts[b] = members.sum()
    accumulated = np.vstack([np.zeros(len(classes)), np.cumsum(effects, axis=0)])
    # Centre: subtract the count-weighted mean of the bin midpoints' values.
    mids = (accumulated[:-1] + accumulated[1:]) / 2
    centre = (mids * counts[:, None]).sum(axis=0) / max(counts.sum(), 1)
    centred = accumulated - centre
    frame = pd.DataFrame(centred, columns=classes)
    frame.insert(0, "value", grid)
    return frame


def kendall_stability(importance: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau between the variable rankings of every pair of folds."""
    wide = importance.pivot_table(index="variable", columns="fold_id", values="drop_macro_f1")
    taus = []
    for a, b in itertools.combinations(wide.columns, 2):
        tau, _ = stats.kendalltau(wide[a], wide[b])
        taus.append(tau)
    taus = np.asarray(taus)
    return pd.DataFrame(
        [
            {
                "n_folds": wide.shape[1],
                "n_pairs": len(taus),
                "tau_mean": float(taus.mean()),
                "tau_min": float(taus.min()),
                "tau_q05": float(np.quantile(taus, 0.05)),
            }
        ]
    )


def summarise_folds(frame: pd.DataFrame, by: list[str], values: list[str]) -> pd.DataFrame:
    """Mean and 95% t-interval over folds for each value column."""
    rows = []
    for key, group in frame.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key, strict=True)) | {"n_folds": len(group)}
        for value in values:
            v = group[value].to_numpy(dtype=float)
            mean = v.mean()
            half = (
                stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                if len(v) > 1
                else np.nan
            )
            row |= {f"{value}": mean, f"{value}_lo": mean - half, f"{value}_hi": mean + half}
        rows.append(row)
    return pd.DataFrame(rows)


def run_interpretation(
    data: pd.DataFrame,
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    columns: list[str],
    blocks: dict[str, list[str]],
    params: dict[str, Any],
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Refit the final configuration per outer fold and compute every interpretation table."""
    settings = config["interpretation"]
    model_name = config["stage_b"]["model"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed + 11_000)
    data = data.copy()
    data[NOISE] = rng.normal(size=len(data))
    ale_vars = settings["ale_variables"]
    grids = {
        v: ale_grid(data[v], settings["ale_bins"], settings["ale_quantile_range"]) for v in ale_vars
    }
    grouped, single, ale, noise, shap_rows = [], [], [], [], []
    for repeat in range(1, config["split"]["outer_repeats"] + 1):
        for fold in range(config["split"]["k_outer"]):
            fold_id = f"s{repeat}_f{fold}"
            task = FoldTask(model_name, repeat, fold, "interpretation")
            _, _, model, test = run_fold(
                task,
                data,
                columns,
                classes,
                domain_xy,
                config,
                n_jobs,
                fixed_params=params,
                return_model=True,
            )
            x, y = test[_x_columns(columns)], test[TARGET].to_numpy()
            g = grouped_importance(model, x, y, blocks, classes, settings["n_repeats_grouped"], rng)
            grouped.append(g.assign(fold_id=fold_id))
            singles = {c: [c] for c in columns if c not in COORDINATES} | {
                "coordinates": COORDINATES
            }
            s = grouped_importance(model, x, y, singles, classes, settings["n_repeats_single"], rng)
            single.append(s.rename(columns={"group": "variable"}).assign(fold_id=fold_id))
            for v in ale_vars:
                ale.append(
                    ale_curve(model, x, v, classes, grids[v]).assign(variable=v, fold_id=fold_id)
                )
            if repeat == 1:
                shap_rows.append(
                    _shap_summary(model, x, columns, classes, settings, rng).assign(fold_id=fold_id)
                )
                # Negative control (b): the same configuration with a pure-noise variable.
                _, _, noisy, test_n = run_fold(
                    FoldTask(model_name, repeat, fold, "noise"),
                    data,
                    columns + [NOISE],
                    classes,
                    domain_xy,
                    config,
                    n_jobs,
                    fixed_params=params,
                    return_model=True,
                )
                xn, yn = test_n[_x_columns(columns + [NOISE])], test_n[TARGET].to_numpy()
                n = grouped_importance(
                    noisy, xn, yn, {NOISE: [NOISE]}, classes, settings["n_repeats_grouped"], rng
                )
                noise.append(n.assign(fold_id=fold_id))
            LOG.info("interpretation %s done", fold_id)

    grouped = pd.concat(grouped, ignore_index=True)
    single = pd.concat(single, ignore_index=True)
    ale = pd.concat(ale, ignore_index=True)
    class_cols = [f"drop_f1__{c}" for c in classes]
    out = {
        "grouped": summarise_folds(grouped, ["group"], ["drop_macro_f1"] + class_cols),
        "single": summarise_folds(single, ["variable"], ["drop_macro_f1"]),
        "stability": kendall_stability(single),
        "ale": _summarise_ale(ale, classes),
        "noise": summarise_folds(pd.concat(noise, ignore_index=True), ["group"], ["drop_macro_f1"]),
        "shap": summarise_folds(
            pd.concat(shap_rows, ignore_index=True), ["variable", "crop_group"], ["mean_abs_shap"]
        ),
    }
    return out


def _summarise_ale(ale: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Mean ALE and the 2.5-97.5% range over folds, per variable, grid point and class."""
    long = ale.melt(
        id_vars=["variable", "fold_id", "value"],
        value_vars=classes,
        var_name="crop_group",
        value_name="ale",
    )
    long["grid_index"] = long.groupby(["variable", "fold_id", "crop_group"]).cumcount()
    agg = long.groupby(["variable", "crop_group", "grid_index"]).agg(
        value=("value", "mean"),
        ale=("ale", "mean"),
        ale_lo=("ale", lambda s: s.quantile(0.025)),
        ale_hi=("ale", lambda s: s.quantile(0.975)),
    )
    return agg.reset_index()


def _shap_summary(model, x, columns, classes, settings, rng) -> pd.DataFrame:
    """Mean |SHAP| per variable and class on a random subsample (TreeSHAP)."""
    import shap

    sample = x.iloc[rng.choice(len(x), size=min(settings["shap_sample"], len(x)), replace=False)]
    prep, forest = model.named_steps["prep"], model.named_steps["model"]
    transformed = prep.transform(sample)
    names = list(prep.get_feature_names_out())
    values = shap.TreeExplainer(forest).shap_values(transformed, check_additivity=False)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == len(sample):  # (n, features, classes)
        values = np.moveaxis(values, 2, 0)
    rows = []
    for j, label in enumerate(forest.classes_):
        mean_abs = np.abs(values[j]).mean(axis=0)
        for name, v in zip(names, mean_abs, strict=True):
            rows.append(
                {
                    "variable": name.split("__", 1)[-1],
                    "crop_group": label,
                    "mean_abs_shap": float(v),
                }
            )
    return pd.DataFrame(rows)


# --- error analysis -----------------------------------------------------------------


def error_analysis(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    classes: list[str],
    min_points: int,
    moran_k: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Confusion matrix, macro F1 by country and NUTS-2, Moran's I of errors."""
    pred = predictions.copy()
    pred["y_pred"] = np.asarray(classes)[
        pred[[f"p__{c}" for c in classes]].to_numpy().argmax(axis=1)
    ]
    pred = pred.merge(data[["point_id", "country", "nuts2"] + COORDINATES], on="point_id")
    confusion = pd.crosstab(pred["y_true"], pred["y_pred"], normalize="index").reindex(
        index=classes, columns=classes, fill_value=0.0
    )

    def regional(by: str) -> pd.DataFrame:
        rows = []
        for region, frame in pred.groupby(by):
            n_points = frame["point_id"].nunique()
            if n_points < min_points:
                continue
            present = sorted(frame["y_true"].unique())
            scores = [
                f1_score(g["y_true"], g["y_pred"], labels=present, average="macro", zero_division=0)
                for _, g in frame.groupby("repeat")
            ]
            rows.append({by: region, "n_points": n_points, "macro_f1": float(np.mean(scores))})
        return pd.DataFrame(rows)

    morans = []
    for repeat, frame in pred.groupby("repeat"):
        wrong = (frame["y_true"] != frame["y_pred"]).to_numpy(dtype=float)
        neighbours = knn_neighbours(frame[COORDINATES].to_numpy(), moran_k)
        observed = morans_i(wrong, neighbours)
        null = np.array(
            [morans_i(rng.permutation(wrong), neighbours) for _ in range(n_permutations)]
        )
        morans.append(
            {
                "repeat": repeat,
                "error_rate": float(wrong.mean()),
                "morans_i": observed,
                "p_permutation": (1 + (np.abs(null) >= abs(observed)).sum()) / (n_permutations + 1),
            }
        )
    return {
        "confusion": confusion,
        "by_country": regional("country"),
        "by_nuts2": regional("nuts2"),
        "morans": pd.DataFrame(morans),
    }
