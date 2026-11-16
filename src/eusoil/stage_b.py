"""Stage B: the hypothesis tests (protocol: Modelling, Stage B; Inferential statistics).

Three runs, each resumable from its checkpoint folder:

1. ``ladder``: the selected algorithm, retuned in every outer fold for each of
   the six predictor sets, on the same complete-case points, folds and seeds,
   so every comparison between steps is paired.
2. ``validation``: the final configuration with fixed hyperparameters (taken
   from the ladder's tuned values) under the validation schemes of E1, and
   leave-one-country-out for H3b.
3. ``report``: the paired spatial block bootstrap of every estimand, the
   corrected resampled t-test as a cross-check, the decisions against the
   0.02 SESOI with 98.75% intervals, Holm-adjusted p-values, and E1.

Bootstrap. The evaluation points are grouped into square blocks of the size
set from the variograms. One resample draws blocks with replacement; every
score in a comparison is recomputed from the same drawn blocks, which keeps
the comparison paired and the spatial dependence inside blocks intact.
Scores averaged over repeats use the same draw for every repeat.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from eusoil.knndm import merge_clusters, nearest_distances
from eusoil.models import (
    COORDINATES,
    FoldTask,
    _plain,
    predictor_columns,
    run_fold,
)
from eusoil.precision import block_ids, macro_f1_from_confusion, per_block_confusion

LOG = logging.getLogger(__name__)

LADDER = {
    1: ["G"],
    2: ["G", "C"],
    3: ["G", "C", "S_inh"],
    4: ["G", "C", "S_nut"],
    5: ["G", "C", "S_inh", "S_nut"],
    6: ["G", "C", "S_inh", "S_nut", "K"],
}
ALL_BLOCKS = LADDER[6]


def step_label(step: int) -> str:
    return f"step{step}"


# --- running ----------------------------------------------------------------------


def _run_tasks(
    tasks: list[tuple[FoldTask, dict[str, Any]]],
    data: pd.DataFrame,
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    checkpoint: Path,
    n_jobs: int,
    title: str,
) -> None:
    """Run (task, options) pairs not yet in ``checkpoint``; resumable."""
    checkpoint.mkdir(parents=True, exist_ok=True)
    for number, (task, options) in enumerate(tasks, start=1):
        done = checkpoint / f"{task.stem}.json"
        if done.exists():
            continue
        record, predictions = run_fold(
            task, data, options["columns"], classes, domain_xy, config, n_jobs,
            fold_column=options.get("fold_column"),
            n_trials=options.get("n_trials"),
            fixed_params=options.get("fixed_params"),
        )  # fmt: skip
        predictions.to_parquet(checkpoint / f"{task.stem}.parquet", index=False)
        done.write_text(json.dumps(record, default=_plain, indent=1))
        LOG.info(
            "%s %d/%d %s: macro F1 %.3f", title, number, len(tasks), task.stem, record["macro_f1"]
        )


def run_ladder(
    data: pd.DataFrame,
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    checkpoint: Path,
    steps: list[int] | None = None,
    n_jobs: int = -1,
) -> None:
    """Each ladder step, retuned per outer fold with the Stage B budget."""
    settings = config["stage_b"]
    tasks = []
    for step in steps or list(LADDER):
        columns = predictor_columns(config, LADDER[step])
        for repeat in range(1, config["split"]["outer_repeats"] + 1):
            for fold in range(config["split"]["k_outer"]):
                task = FoldTask(settings["model"], repeat, fold, step_label(step))
                tasks.append((task, {"columns": columns, "n_trials": settings["n_trials"]}))
    _run_tasks(tasks, data, classes, domain_xy, config, checkpoint, n_jobs, "ladder")


def final_params(records: pd.DataFrame) -> dict[str, Any]:
    """One hyperparameter set from the per-fold tuned values: medians, or the mode.

    Numbers take the median over folds (rounded for integer parameters);
    other values (``None`` included) take the most frequent choice.
    """
    params = [json.loads(p) for p in records["best_params"]]
    out: dict[str, Any] = {}
    for key in params[0]:
        values = [p[key] for p in params]
        numeric = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
        if len(numeric) == len(values):
            median = float(np.median(numeric))
            out[key] = int(round(median)) if all(isinstance(v, int) for v in numeric) else median
        else:
            labels = pd.Series([json.dumps(v) for v in values])
            out[key] = json.loads(labels.value_counts().index[0])
    return out


def block_folds(
    xy: np.ndarray, size_m: float, offset: float, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k folds of whole square blocks; the grid is shifted by ``offset`` x block size."""
    shifted = xy + offset * size_m
    blocks = block_ids(shifted[:, 0], shifted[:, 1], size_m)
    return merge_clusters(blocks, k, rng)


def validation_schemes(
    data: pd.DataFrame, config: dict[str, Any]
) -> dict[str, list[tuple[int, str]]]:
    """Add fold columns for E1 and H3b to ``data``; return {scheme: [(repeat, column)]}.

    kNNDM (the Phase 2 outer folds), random k-fold, square blocks at each size
    with shifted grids, and leave-one-country-out.
    """
    settings = config["stage_b"]
    k = config["split"]["k_outer"]
    seed = int(config["seed"])
    xy = data[COORDINATES].to_numpy()
    schemes: dict[str, list[tuple[int, str]]] = {
        "knndm": [(r, f"outer_fold_s{r}") for r in range(1, config["split"]["outer_repeats"] + 1)]
    }
    schemes["random"] = []
    for r in range(1, settings["random_repeats"] + 1):
        column = f"cv_random_s{r}"
        data[column] = np.random.default_rng(seed + 7000 + r).permutation(np.arange(len(data)) % k)
        schemes["random"].append((r, column))
    offsets = settings["block_offsets"]
    for size in settings["block_sizes_km"]:
        name = f"blocks_{size}km"
        schemes[name] = []
        for r, offset in enumerate(offsets, start=1):
            column = f"cv_{name}_s{r}"
            rng = np.random.default_rng(seed + 8000 + 10 * size + r)
            data[column] = block_folds(xy, size * 1000.0, offset, k, rng)
            schemes[name].append((r, column))
    countries = sorted(data["country"].unique())
    data["cv_loco"] = data["country"].map({c: i for i, c in enumerate(countries)})
    schemes["loco"] = [(1, "cv_loco")]
    return schemes


def run_validation(
    data: pd.DataFrame,
    classes: list[str],
    domain_xy: np.ndarray,
    config: dict[str, Any],
    final_step: int,
    params: dict[str, Any],
    checkpoint: Path,
    n_jobs: int = -1,
) -> dict[str, list[tuple[int, str]]]:
    """The final configuration, fixed hyperparameters, under every validation scheme."""
    model = config["stage_b"]["model"]
    columns = predictor_columns(config, LADDER[final_step])
    schemes = validation_schemes(data, config)
    tasks = []
    for scheme, runs in schemes.items():
        for repeat, column in runs:
            for fold in sorted(data[column].unique()):
                task = FoldTask(model, repeat, int(fold), scheme)
                tasks.append(
                    (task, {"columns": columns, "fold_column": column, "fixed_params": params})
                )
    _run_tasks(tasks, data, classes, domain_xy, config, checkpoint, n_jobs, "validation")
    return schemes


# --- loading predictions -------------------------------------------------------------


def load_predictions(checkpoint: Path, label: str) -> pd.DataFrame:
    files = sorted(checkpoint.glob(f"*_{label}_s*_f*.parquet"))
    if not files:
        raise FileNotFoundError(f"No predictions for {label} in {checkpoint}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def predicted_class_index(predictions: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return predictions[[f"p__{c}" for c in classes]].to_numpy().argmax(axis=1)


# --- the paired spatial block bootstrap ----------------------------------------------


class BlockBootstrap:
    """Macro F1 of several prediction sets on the same resampled blocks.

    ``scores`` maps a name to a list of prediction tables (one per repeat),
    each covering the same points. Every table is reduced to per-block
    confusion matrices once; a resample is then a vector of block counts.
    """

    def __init__(
        self,
        points: pd.DataFrame,
        scores: dict[str, list[pd.DataFrame]],
        classes: list[str],
        block_m: float,
    ):
        self.classes = classes
        xy = points.set_index("point_id")[COORDINATES]
        self.order = xy.index.to_numpy()
        blocks = block_ids(xy["easting"].to_numpy(), xy["northing"].to_numpy(), block_m)
        self.n_blocks = int(blocks.max()) + 1
        block_of = pd.Series(blocks, index=self.order)
        index_of = {c: i for i, c in enumerate(classes)}
        k = len(classes)
        self.confusions: dict[str, list[np.ndarray]] = {}
        for name, tables in scores.items():
            mats = []
            for table in tables:
                y = table["y_true"].map(index_of).to_numpy()
                pred = predicted_class_index(table, classes)
                b = block_of.loc[table["point_id"]].to_numpy()
                cm = per_block_confusion(y, pred, b, k)
                if cm.shape[0] < self.n_blocks:
                    cm = np.concatenate([cm, np.zeros((self.n_blocks - cm.shape[0], k, k), int)])
                mats.append(cm)
            self.confusions[name] = mats

    def scores(self, counts: np.ndarray) -> dict[str, float]:
        """Macro F1 per name (mean over repeats) for block multiplicities ``counts``."""
        out = {}
        for name, mats in self.confusions.items():
            values = [macro_f1_from_confusion(np.tensordot(counts, m, axes=1)) for m in mats]
            out[name] = float(np.mean(values))
        return out

    def observed(self) -> dict[str, float]:
        return self.scores(np.ones(self.n_blocks))

    def resample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Scores for ``n`` block resamples (one row each)."""
        rows = []
        for _ in range(n):
            draw = rng.integers(0, self.n_blocks, self.n_blocks)
            rows.append(self.scores(np.bincount(draw, minlength=self.n_blocks)))
        return pd.DataFrame(rows)


def bootstrap_p(differences: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for a difference of zero."""
    n = len(differences)
    below = (np.sum(differences <= 0) + 1) / (n + 1)
    above = (np.sum(differences >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def corrected_t_test(diffs: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau and Bengio (2003) corrected resampled t-test over fold differences."""
    j = len(diffs)
    variance = np.var(diffs, ddof=1)
    if variance == 0:
        return float("inf") * np.sign(diffs.mean()), 0.0
    t = diffs.mean() / np.sqrt((1 / j + n_test / n_train) * variance)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def decide(estimate: float, low: float, high: float, sesoi: float, two_sided: bool) -> str:
    """The protocol's decision for one estimand (Research questions, decision rule)."""
    if low > 0 and estimate >= sesoi:
        return "supported"
    if two_sided and high < 0 and estimate <= -sesoi:
        return "reversed"
    if two_sided and -sesoi < low and high < sesoi:
        return "negligible"
    if not two_sided and high < sesoi:
        return "negligible"
    return "inconclusive"


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


# --- E1 ------------------------------------------------------------------------------


def scheme_distances(data: pd.DataFrame, column: str) -> np.ndarray:
    """For each point, the distance to the nearest training point of its fold split."""
    xy = data[COORDINATES].to_numpy()
    folds = data[column].to_numpy()
    out = np.empty(len(data))
    for fold in np.unique(folds):
        held = folds == fold
        out[held] = nearest_distances(xy[held], xy[~held])
    return out


def plateau(changes: pd.DataFrame, sesoi: float) -> str:
    """First block size beyond which doubling changes macro F1 by less than ``sesoi``.

    ``changes`` has one row per doubling (``from_km``, ``to_km``, ``estimate``,
    ``low``, ``high``), the interval being the 95% bootstrap interval.
    """
    for row in changes.sort_values("from_km").itertuples(index=False):
        if abs(row.estimate) < sesoi and -sesoi < row.low and row.high < sesoi:
            return f"{row.from_km} km"
    return "no plateau up to the largest block size"


# --- putting it together ---------------------------------------------------------------


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def estimand_rows(
    observed: dict[str, float],
    draws: pd.DataFrame,
    definitions: dict[str, tuple[str, str, bool]],
    sesoi: float,
    decision_level: float,
) -> pd.DataFrame:
    """One row per hypothesis: estimate, 95% and decision intervals, p, decision.

    ``definitions`` maps a hypothesis to (score a, score b, two_sided); the
    estimand is a - b.
    """
    rows = []
    for name, (a, b, two_sided) in definitions.items():
        diffs = (draws[a] - draws[b]).to_numpy()
        estimate = observed[a] - observed[b]
        lo95, hi95 = _interval(diffs, 0.95)
        lo, hi = _interval(diffs, decision_level)
        rows.append(
            {
                "hypothesis": name,
                "score_a": a,
                "score_b": b,
                "macro_f1_a": observed[a],
                "macro_f1_b": observed[b],
                "estimate": estimate,
                "bootstrap_median": float(np.median(diffs)),
                "ci95_low": lo95,
                "ci95_high": hi95,
                "decision_ci_low": lo,
                "decision_ci_high": hi,
                "p_bootstrap": bootstrap_p(diffs),
                "decision": decide(estimate, lo, hi, sesoi, two_sided),
            }
        )
    return pd.DataFrame(rows)


LADDER_HYPOTHESES = {
    "H1": (step_label(5), step_label(2), False),
    "H2": (step_label(3), step_label(4), True),
    "H3a": (step_label(6), step_label(5), False),
}


def ladder_analysis(
    data: pd.DataFrame,
    classes: list[str],
    config: dict[str, Any],
    checkpoint: Path,
    block_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step scores, the H1/H2/H3a estimands and the corrected t-test cross-check."""
    settings = config["stage_b"]
    expected = config["split"]["k_outer"] * config["split"]["outer_repeats"]
    tables, records = {}, []
    for step in LADDER:
        label = step_label(step)
        files = sorted(checkpoint.glob(f"*_{label}_s*_f*.json"))
        if len(files) != expected:
            raise ValueError(f"Ladder {label}: {len(files)} of {expected} folds done")
        records += [json.loads(f.read_text()) for f in files]
        predictions = load_predictions(checkpoint, label)
        tables[label] = [g for _, g in predictions.groupby("repeat")]
    boot = BlockBootstrap(data, tables, classes, block_km * 1000.0)
    observed = boot.observed()
    draws = boot.resample(
        settings["n_bootstrap"], np.random.default_rng(int(config["seed"]) + 9000)
    )
    steps = pd.DataFrame(
        [
            {
                "step": step,
                "blocks": " + ".join(LADDER[step]),
                "macro_f1": observed[step_label(step)],
                "ci95_low": _interval(draws[step_label(step)].to_numpy(), 0.95)[0],
                "ci95_high": _interval(draws[step_label(step)].to_numpy(), 0.95)[1],
            }
            for step in LADDER
        ]
    )
    estimands = estimand_rows(
        observed,
        draws,
        LADDER_HYPOTHESES,
        config["precision"]["sesoi_macro_f1"],
        1 - settings["alpha_decision"],
    )
    folds = pd.DataFrame(records)
    wide = folds.pivot_table(index=["repeat", "fold"], columns="label", values="macro_f1")
    ratio = folds["n_test"].mean() / folds["n_train"].mean()
    checks = []
    for name, (a, b, _) in LADDER_HYPOTHESES.items():
        diffs = (wide[a] - wide[b]).to_numpy()
        t, p = corrected_t_test(diffs, n_train=1.0, n_test=ratio)
        checks.append({"hypothesis": name, "t_corrected": t, "p_corrected_t": p})
    estimands = estimands.merge(pd.DataFrame(checks), on="hypothesis")
    return steps, estimands, folds


def final_step(estimands: pd.DataFrame) -> int:
    """Ladder step 6 if H3a is supported, otherwise step 5 (protocol: Final model)."""
    h3a = estimands.set_index("hypothesis").loc["H3a", "decision"]
    return 6 if h3a == "supported" else 5


def validation_analysis(
    data: pd.DataFrame,
    classes: list[str],
    config: dict[str, Any],
    checkpoint: Path,
    schemes: dict[str, list[tuple[int, str]]],
    block_km: float,
    domain_median_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """H3b, the per-country transfer table, the E1 curve and its plateau check."""
    settings = config["stage_b"]
    tables = {}
    for scheme in schemes:
        predictions = load_predictions(checkpoint, scheme)
        tables[scheme] = [g for _, g in predictions.groupby("repeat")]
    boot = BlockBootstrap(data, tables, classes, block_km * 1000.0)
    observed = boot.observed()
    draws = boot.resample(
        settings["n_bootstrap"], np.random.default_rng(int(config["seed"]) + 9100)
    )

    h3b = estimand_rows(
        observed,
        draws,
        {"H3b": ("knndm", "loco", False)},
        config["precision"]["sesoi_macro_f1"],
        1 - settings["alpha_decision"],
    )

    curve = []
    for scheme, runs in schemes.items():
        medians = [np.median(scheme_distances(data, column)) / 1000.0 for _, column in runs]
        low, high = _interval(draws[scheme].to_numpy(), 0.95)
        curve.append(
            {
                "scheme": scheme,
                "n_repeats": len(runs),
                "median_distance_km": float(np.mean(medians)),
                "macro_f1": observed[scheme],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    curve = pd.DataFrame(curve).sort_values("median_distance_km").reset_index(drop=True)
    curve["domain_median_distance_km"] = domain_median_km

    sizes = settings["block_sizes_km"]
    changes = []
    for small, large in zip(sizes[:-1], sizes[1:], strict=True):
        a, b = f"blocks_{large}km", f"blocks_{small}km"
        diffs = (draws[a] - draws[b]).to_numpy()
        low, high = _interval(diffs, 0.95)
        changes.append(
            {
                "from_km": small,
                "to_km": large,
                "estimate": observed[a] - observed[b],
                "low": low,
                "high": high,
            }
        )
    changes = pd.DataFrame(changes)
    changes["plateau"] = plateau(changes, config["precision"]["sesoi_macro_f1"])

    loco = tables["loco"][0]
    knn = pd.concat(tables["knndm"])
    rows = []
    for country, frame in data.groupby("country"):
        ids = set(frame["point_id"])
        sub_l = loco[loco["point_id"].isin(ids)]
        sub_k = knn[knn["point_id"].isin(ids)]
        rows.append(
            {
                "country": country,
                "n": len(frame),
                "macro_f1_loco": _macro_f1(sub_l, classes),
                "macro_f1_knndm": _macro_f1(sub_k, classes),
                "accuracy_loco": float(
                    (
                        np.asarray(classes)[predicted_class_index(sub_l, classes)]
                        == sub_l["y_true"]
                    ).mean()
                ),
            }
        )
    transfer = pd.DataFrame(rows)
    transfer["loss"] = transfer["macro_f1_knndm"] - transfer["macro_f1_loco"]
    return h3b, transfer, curve, changes


def _macro_f1(table: pd.DataFrame, classes: list[str]) -> float:
    """Macro F1 over the classes present in ``table`` (for small per-country subsets)."""
    from sklearn.metrics import f1_score

    y = table["y_true"].to_numpy()
    pred = np.asarray(classes)[predicted_class_index(table, classes)]
    present = sorted(set(y))
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
