"""Phase 5 sensitivity and robustness runs (protocol: Sensitivity and robustness analyses).

Each run changes one thing (the data, the target, the predictors, the folds
or the model) and repeats the hypothesis tests with **fixed hyperparameters**
(amendment A18): every ladder step uses the per-parameter median of its own
Stage B tuned values; other algorithms use the medians of their Stage A
values. A reference run with the same fixed settings and no change anchors
the comparison.

Per run, depending on what it changes:

* the ladder (steps 2 to 6) on the run's folds -> H1, H2 and H3a;
* leave-one-country-out with step 5 -> H3b;
* square blocks of 200 km with step 5 -> the E1 summary (the drop in macro F1
  from the run's primary folds to 200 km blocks).

Decisions use the same rules as Stage B (98.75% block-bootstrap intervals,
the 0.02 SESOI). A hypothesis is robust when every run reaches the primary
decision, mostly robust with one disagreement, and fragile with two or more.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from eusoil.models import TARGET, FoldTask, _plain, run_fold
from eusoil.stage_b import (
    LADDER,
    BlockBootstrap,
    estimand_rows,
    final_params,
    load_predictions,
    step_label,
    validation_schemes,
)

LOG = logging.getLogger(__name__)

SENSITIVITY_STEPS = [2, 3, 4, 5, 6]
TESTS = {
    "H1": (step_label(5), step_label(2), False),
    "H2": (step_label(3), step_label(4), True),
    "H3a": (step_label(6), step_label(5), False),
}


@dataclass
class Variation:
    """One sensitivity run."""

    name: str
    area: str
    description: str
    prepare: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame] | None = None
    blocks: dict[str, list[str]] | None = None  # overrides of config["blocks"]
    fold_prefix: str | None = None  # e.g. "cv_blocks_50km_s"; default kNNDM outer folds
    model: str | None = None  # default: the Stage B model
    class_weight: str | None = "balanced"
    seed_shift: int = 0
    weighted: bool = False  # survey-design weights in fitting and scoring
    outputs: tuple[str, ...] = ("ladder", "h3b", "e1")
    imputations: int = 0  # > 0: multiple imputation of below-LOD values
    extra: dict[str, Any] = field(default_factory=dict)


# --- data changes -----------------------------------------------------------------


def lod_factor(factor: float) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """Below-LOD values set to ``factor`` x LOD (the primary run uses 0.5)."""

    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        out = df.copy()
        for variable, log_column in (
            ("n_g_kg", "n_log"),
            ("p_mg_kg", "p_log"),
            ("k_mg_kg", "k_log"),
            ("oc_g_kg", "oc_log"),
        ):
            flagged = out[f"{variable}_below_lod"].fillna(False).astype(bool)
            out.loc[flagged, log_column] = np.log(factor * config["cleaning"]["lod"][variable])
        return out

    return prepare


def impute_below_lod(seed: int) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """Draw below-LOD values from a left-censored log-normal fitted by maximum likelihood.

    For each of N, P, K and organic carbon, a normal distribution is fitted
    on the log scale to all points, treating below-LOD points as censored at
    log(LOD) (a Tobit likelihood); each censored value is then drawn from
    that normal truncated above at log(LOD). Uses predictors only, never the
    target.
    """

    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        out = df.copy()
        for variable, log_column in (
            ("n_g_kg", "n_log"),
            ("p_mg_kg", "p_log"),
            ("k_mg_kg", "k_log"),
            ("oc_g_kg", "oc_log"),
        ):
            flagged = out[f"{variable}_below_lod"].fillna(False).astype(bool).to_numpy()
            if not flagged.any():
                continue
            limit = np.log(config["cleaning"]["lod"][variable])
            observed = out.loc[~flagged, log_column].dropna().to_numpy()
            n_censored = int(flagged.sum())

            def negloglik(theta, observed=observed, n_cens=n_censored, limit=limit):
                mu, log_sigma = theta
                sigma = np.exp(log_sigma)
                return -(
                    stats.norm.logpdf(observed, mu, sigma).sum()
                    + n_cens * stats.norm.logcdf(limit, mu, sigma)
                )

            fit = minimize(
                negloglik, [observed.mean(), np.log(observed.std())], method="Nelder-Mead"
            )
            mu, sigma = fit.x[0], np.exp(fit.x[1])
            upper = (limit - mu) / sigma
            draws = stats.truncnorm.rvs(
                -np.inf, upper, loc=mu, scale=sigma, size=flagged.sum(), random_state=rng
            )
            out.loc[flagged, log_column] = draws
        return out

    return prepare


def regroup(
    mapping: Callable[[pd.DataFrame], pd.Series],
) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        out = df.copy()
        out[TARGET] = mapping(out)
        return out

    return prepare


def finer_groups(df: pd.DataFrame) -> pd.Series:
    """Minimum class size 200: root crops apart from maize, vineyards apart from orchards."""
    group = df[TARGET].copy()
    lc1 = df["lc1"].str.upper()
    group[lc1.isin(["B21", "B22", "B23"])] = "root_crops"
    group[lc1.eq("B16")] = "maize"
    group[lc1.eq("B82")] = "vineyards"
    group[lc1.isin(["B71", "B72", "B73", "B74", "B75", "B76", "B77"])] = "orchards"
    return group


def coarser_groups(df: pd.DataFrame) -> pd.Series:
    """Minimum class size 500: olive groves joined with vineyards and orchards."""
    group = df[TARGET].copy()
    group[group.isin(["olive_groves", "vineyards_and_orchards"])] = "permanent_crops"
    return group


def subset(
    mask: Callable[[pd.DataFrame], pd.Series],
) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        return df[mask(df).fillna(False).astype(bool)].reset_index(drop=True)

    return prepare


def thin(cell_m: float, seed: int) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """One randomly chosen point per square cell."""

    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        cells = (
            np.floor(df["easting"] / cell_m).astype(int).astype(str)
            + "_"
            + np.floor(df["northing"] / cell_m).astype(int).astype(str)
        )
        order = np.random.default_rng(seed).permutation(len(df))
        shuffled = df.iloc[order].assign(_cell=cells.iloc[order].to_numpy())
        return (
            shuffled.drop_duplicates("_cell")
            .drop(columns="_cell")
            .sort_values("point_id")
            .reset_index(drop=True)
        )

    return prepare


def permute_within_country(seed: int) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """Negative control (a): labels shuffled within each country."""

    def prepare(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = df.copy()
        for _, index in out.groupby("country").groups.items():
            out.loc[index, TARGET] = rng.permutation(out.loc[index, TARGET].to_numpy())
        return out

    return prepare


def survey_weights(df: pd.DataFrame, domain_counts: pd.Series) -> pd.Series:
    """Weight per point so each country's share matches its share of farmland.

    LUCAS points lie on a regular 2 km grid, so the number of 2018 survey
    points on farmland (prediction domain plus sample) is proportional to a
    country's farmland area. Weights average to 1.
    """
    sample = df["country"].value_counts()
    area = domain_counts.reindex(sample.index).fillna(0) + sample
    w_country = (area / area.sum()) / (sample / sample.sum())
    w = df["country"].map(w_country)
    return w / w.mean()


# --- the registry ---------------------------------------------------------------------


def registry(
    config: dict[str, Any], countries: list[str], climate_cluster: list[str]
) -> list[Variation]:
    """Every sensitivity run in the protocol's table that can be run (see ``not_run``)."""
    seed = int(config["seed"])
    blocks = config["blocks"]
    c_numbers = [c.removeprefix("tc9120_bio") for c in blocks["C"]]

    def swap_climate(prefix: str) -> dict[str, list[str]]:
        return {"C": [f"{prefix}{n}" for n in c_numbers]}

    sizes = config["stage_b"]["block_sizes_km"]
    largest = config["sensitivity"]["largest_countries"]
    runs = [
        Variation("reference", "Reference", "Primary settings, fixed hyperparameters"),
        Variation(
            "lod_sqrt2",
            "Detection limits",
            "Below-LOD values at LOD/sqrt(2)",
            prepare=lod_factor(0.7071),
        ),
        Variation(
            "lod_full", "Detection limits", "Below-LOD values at the LOD", prepare=lod_factor(1.0)
        ),
        Variation(
            "lod_mi",
            "Detection limits",
            "Multiple imputation (5 draws, censored log-normal)",
            imputations=config["sensitivity"]["n_imputations"],
        ),
        Variation(
            "level1",
            "Target definition",
            "Level 1: arable, permanent crops, grassland",
            prepare=regroup(lambda d: d["farming_system"]),
        ),
        Variation(
            "min_class_200",
            "Target definition",
            "Finer groups (minimum class size 200): 10 classes",
            prepare=regroup(finer_groups),
        ),
        Variation(
            "min_class_500",
            "Target definition",
            "Coarser groups (minimum class size 500): 7 classes",
            prepare=regroup(coarser_groups),
        ),
        Variation(
            "pure_labels",
            "Target definition",
            "Uncertain labels removed (land cover share above 50%)",
            prepare=subset(lambda d: d["label_pure"]),
        ),
    ]
    runs += [
        Variation(
            f"blocks_{s}km",
            "Validation",
            f"Square blocks of {s} km (3 grid offsets) instead of kNNDM",
            fold_prefix=f"cv_blocks_{s}km_s",
            outputs=("ladder", "h3b"),
        )
        for s in sizes
    ]
    runs += [
        Variation(
            "in_situ",
            "Sample",
            "Points observed in the field within 100 m only",
            prepare=subset(lambda d: d["observed_in_situ"]),
        ),
        Variation(
            "no_organic",
            "Sample",
            "Organic soils (OC above 120 g/kg) excluded",
            prepare=subset(lambda d: ~d["organic_soil"].fillna(False).astype(bool)),
        ),
        Variation(
            "thinned_10km",
            "Sample",
            "Spatially thinned: one point per 10 km cell",
            prepare=thin(10_000.0, seed + 12_000),
        ),
        Variation(
            "largest_five",
            "Geography",
            f"Largest five countries only ({', '.join(largest)})",
            prepare=subset(lambda d: d["country"].isin(largest)),
            outputs=("ladder", "h3b"),
        ),
        Variation(
            "germany_only",
            "Geography",
            "Germany only",
            prepare=subset(lambda d: d["country"].eq("DE")),
            outputs=("ladder",),
        ),
    ]
    runs += [
        Variation(
            f"without_{c}",
            "Geography",
            f"{c} left out of training and evaluation",
            prepare=subset(lambda d, c=c: d["country"].ne(c)),
            outputs=("ladder",),
        )
        for c in countries
    ]
    runs += [
        Variation(
            "oc_in_inh",
            "Predictors",
            "Organic carbon added to the inherent block",
            blocks={"S_inh": blocks["S_inh"] + ["oc_log"]},
        ),
        Variation(
            "oc_in_nut",
            "Predictors",
            "Organic carbon added to the nutrient block",
            blocks={"S_nut": blocks["S_nut"] + ["oc_log"]},
        ),
        Variation(
            "ph_h2o",
            "Predictors",
            "pH in water instead of CaCl2",
            blocks={"S_inh": [("ph_h2o" if c == "ph_cacl2" else c) for c in blocks["S_inh"]]},
        ),
        Variation(
            "climate_clusters",
            "Predictors",
            f"Cluster-based climate subset ({len(climate_cluster)} variables)",
            blocks={"C": climate_cluster},
        ),
        Variation(
            "terraclimate_2015_2018",
            "Predictors",
            "TerraClimate 2015-2018 instead of 1991-2020 normals",
            blocks=swap_climate("tc1518_bio"),
        ),
        Variation(
            "worldclim_30s",
            "Predictors",
            "WorldClim 2.1, 30 arc-seconds (internal check)",
            blocks=swap_climate("wc30s_bio"),
        ),
        Variation(
            "worldclim_2_5m",
            "Predictors",
            "WorldClim 2.1, 2.5 minutes (internal check)",
            blocks=swap_climate("wc25m_bio"),
        ),
        Variation("boosting", "Model", "Gradient boosting through the ladder", model="boosting"),
        Variation("logistic", "Model", "Logistic regression through the ladder", model="logistic"),
        Variation("unweighted", "Model", "Random forest without class weights", class_weight=None),
    ]
    runs += [
        Variation(
            f"seed_{i}",
            "Model",
            f"Model seed {i} of 5",
            seed_shift=100_000 * i,
            outputs=("ladder",),
        )
        for i in range(1, 6)
    ]
    runs += [
        Variation(
            "permuted_labels",
            "Negative control",
            "Labels permuted within each country",
            prepare=permute_within_country(seed + 13_000),
            outputs=("ladder",),
        ),
        Variation(
            "survey_weights",
            "Survey design",
            "Points reweighted to each country's farmland area",
            weighted=True,
        ),
    ]
    return runs


NOT_RUN = [
    {
        "run": "buffered_loo",
        "area": "Validation",
        "description": "Buffered leave-one-out (training points within the variogram range "
        "excluded)",
        "reason": "With a 160 km range, removing every training point within range of a test "
        "point removes most of Europe's training data for each point; the 200 km block run tests "
        "the same question (amendment A18)",
    },
    {
        "run": "focal_loss",
        "area": "Model",
        "description": "Focal-loss variant",
        "reason": "Not available for scikit-learn forests or histogram boosting; the unweighted "
        "run covers the class-weighting question (amendment A18)",
    },
    {
        "run": "loco_prior_em",
        "area": "Validation",
        "description": "Leave-one-country-out with class priors re-estimated by EM "
        "(Saerens et al., 2002)",
        "reason": "Specified in the protocol but not built; found missing in the audit of "
        "22 November 2026 (amendment A20). H3b is reported with the held-out country's own "
        "class mix",
    },
]


# --- running ---------------------------------------------------------------------------


def variation_columns(config: dict[str, Any], variation: Variation, step: int) -> list[str]:
    merged = dict(config["blocks"]) | (variation.blocks or {})
    return list(dict.fromkeys(c for block in LADDER[step] for c in merged[block]))


def prepare_data(
    data: pd.DataFrame, config: dict[str, Any], variation: Variation, imputation: int
) -> pd.DataFrame:
    """The variation's data: changed, then restricted to complete cases on its predictors."""
    out = data
    if variation.imputations:
        out = impute_below_lod(int(config["seed"]) + 14_000 + imputation)(out, config)
    if variation.prepare is not None:
        out = variation.prepare(out, config)
    columns = variation_columns(config, variation, 6)
    return out[out[columns].notna().all(axis=1)].reset_index(drop=True)


def fixed_parameters(
    config: dict[str, Any], variation: Variation, stage_a_dir: Path, ladder_dir: Path, step: int
) -> dict[str, Any]:
    """Per-step medians of Stage B (Stage B model) or Stage A medians (other algorithms)."""
    model = variation.model or config["stage_b"]["model"]
    if model == config["stage_b"]["model"]:
        files = sorted(ladder_dir.glob(f"*_{step_label(step)}_s*_f*.json"))
    else:
        files = sorted(stage_a_dir.glob(f"{model}_s*_f*.json"))
    if not files:
        raise FileNotFoundError(f"No tuned parameters for {model} (step {step})")
    return final_params(pd.DataFrame([json.loads(f.read_text()) for f in files]))


def run_variation(
    variation: Variation,
    data: pd.DataFrame,
    domain_xy: np.ndarray,
    domain_counts: pd.Series,
    config: dict[str, Any],
    stage_a_dir: Path,
    ladder_dir: Path,
    checkpoint: Path,
    n_jobs: int = -1,
) -> None:
    """Every fit of one variation not yet in its checkpoint folder."""
    model = variation.model or config["stage_b"]["model"]
    repeats = range(1, config["split"]["outer_repeats"] + 1)
    for m in range(max(1, variation.imputations)):
        name = f"{variation.name}_m{m + 1}" if variation.imputations else variation.name
        folder = checkpoint / name
        folder.mkdir(parents=True, exist_ok=True)
        frame = prepare_data(data, config, variation, m)
        validation_schemes(frame, config)  # adds the block and leave-one-country-out folds
        if variation.weighted:
            frame["survey_weight"] = survey_weights(frame, domain_counts)
        classes = sorted(frame[TARGET].unique())
        tasks = []
        primary = variation.fold_prefix or "outer_fold_s"
        if "ladder" in variation.outputs:
            for step in SENSITIVITY_STEPS:
                for r in repeats:
                    for f in sorted(frame[f"{primary}{r}"].unique()):
                        tasks.append(
                            (step, FoldTask(model, r, int(f), step_label(step)), f"{primary}{r}")
                        )
        if "h3b" in variation.outputs and frame["country"].nunique() > 1:
            for f in sorted(frame["cv_loco"].unique()):
                tasks.append((5, FoldTask(model, 1, int(f), "loco"), "cv_loco"))
        if "e1" in variation.outputs:
            for r in repeats:
                column = f"cv_blocks_200km_s{r}"
                for f in sorted(frame[column].unique()):
                    tasks.append((5, FoldTask(model, r, int(f), "blocks_200km"), column))
        for step, task, column in tasks:
            done = folder / f"{task.stem}.json"
            if done.exists():
                continue
            params = fixed_parameters(config, variation, stage_a_dir, ladder_dir, step)
            record, predictions = run_fold(
                task,
                frame,
                variation_columns(config, variation, step),
                classes,
                domain_xy,
                config,
                n_jobs,
                fold_column=column,
                fixed_params=params,
                class_weight=variation.class_weight,
                seed_shift=variation.seed_shift,
                weight_column="survey_weight" if variation.weighted else None,
            )
            predictions.to_parquet(folder / f"{task.stem}.parquet", index=False)
            done.write_text(json.dumps(record, default=_plain, indent=1))
        LOG.info(
            "sensitivity %s: %d fits, %d points, %d classes",
            name,
            len(tasks),
            len(frame),
            len(classes),
        )


# --- analysis -----------------------------------------------------------------------------


def analyse_variation(
    variation: Variation,
    data: pd.DataFrame,
    domain_counts: pd.Series,
    config: dict[str, Any],
    checkpoint: Path,
    block_km: float,
) -> list[dict[str, Any]]:
    """Estimate, intervals and decision for each hypothesis and the E1 summary."""
    settings = config["stage_b"]
    sesoi = config["precision"]["sesoi_macro_f1"]
    names = (
        [f"{variation.name}_m{m + 1}" for m in range(variation.imputations)]
        if variation.imputations
        else [variation.name]
    )
    all_draws, all_observed = [], []
    for m, name in enumerate(names):
        folder = checkpoint / name
        frame = prepare_data(data, config, variation, m)
        classes = sorted(frame[TARGET].unique())
        weights = (
            survey_weights(frame, domain_counts).set_axis(frame["point_id"])
            if variation.weighted
            else None
        )
        tables = {}
        labels = []
        if "ladder" in variation.outputs:
            labels += [step_label(s) for s in SENSITIVITY_STEPS]
        if "h3b" in variation.outputs and frame["country"].nunique() > 1:
            labels.append("loco")
        if "e1" in variation.outputs:
            labels.append("blocks_200km")
        for label in labels:
            predictions = load_predictions(folder, label)
            tables[label] = [g for _, g in predictions.groupby("repeat")]
        boot = BlockBootstrap(frame, tables, classes, block_km * 1000.0, weights=weights)
        all_observed.append(boot.observed())
        all_draws.append(
            boot.resample(
                settings["n_bootstrap"] // len(names),
                np.random.default_rng(int(config["seed"]) + 15_000 + m),
            )
        )
    observed = pd.DataFrame(all_observed).mean().to_dict()
    draws = pd.concat(all_draws, ignore_index=True)  # mixture over imputations

    definitions = {}
    if "ladder" in variation.outputs:
        definitions |= TESTS
    if step_label(5) in observed and "loco" in observed:
        definitions["H3b"] = (step_label(5), "loco", False)
    rows = []
    if definitions:
        table = estimand_rows(observed, draws, definitions, sesoi, 1 - settings["alpha_decision"])
        rows += table.to_dict("records")
    if "blocks_200km" in observed and step_label(5) in observed:
        diffs = (draws[step_label(5)] - draws["blocks_200km"]).to_numpy()
        rows.append(
            {
                "hypothesis": "E1",
                "estimate": observed[step_label(5)] - observed["blocks_200km"],
                "ci95_low": float(np.quantile(diffs, 0.025)),
                "ci95_high": float(np.quantile(diffs, 0.975)),
                "decision": "estimate",
            }
        )
    if step_label(5) in observed:
        rows.append(
            {
                "hypothesis": "macro_f1_step5",
                "estimate": observed[step_label(5)],
                "decision": "score",
            }
        )
    for row in rows:
        row |= {
            "run": variation.name,
            "area": variation.area,
            "description": variation.description,
            "n_points": len(frame),
            "n_classes": len(classes),
        }
    return rows


def robustness(summary: pd.DataFrame, reference: str = "reference") -> pd.DataFrame:
    """Agreement of every run's decision with the reference run's, per hypothesis.

    Negative controls are excluded: they are meant to fail.
    """
    rows = []
    tested = summary[
        summary["hypothesis"].isin(["H1", "H2", "H3a", "H3b"])
        & summary["area"].ne("Negative control")
    ]
    for hypothesis, frame in tested.groupby("hypothesis"):
        primary = frame.loc[frame["run"].eq(reference), "decision"]
        if primary.empty:
            continue
        primary = primary.iloc[0]
        others = frame[frame["run"].ne(reference)]
        disagree = others[others["decision"].ne(primary)]
        n_dis = len(disagree)
        label = "robust" if n_dis == 0 else ("mostly robust" if n_dis == 1 else "fragile")
        rows.append(
            {
                "hypothesis": hypothesis,
                "reference_decision": primary,
                "n_runs": len(others),
                "n_disagree": n_dis,
                "disagreeing_runs": ";".join(
                    f"{r}={d}" for r, d in zip(disagree["run"], disagree["decision"], strict=True)
                ),
                "verdict": label,
            }
        )
    return pd.DataFrame(rows)


def country_prior_baseline(frame: pd.DataFrame, config: dict[str, Any]) -> float:
    """Macro F1 of a model that knows only each point's country (the chance level of control a).

    Labels permuted within each country keep every country's crop mix, and
    the coordinates reveal the country, so this, not 1 / n_classes, is the
    score a leak-free pipeline should reach on permuted labels (amendment A19).
    Same folds and class weighting as the runs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    x = pd.get_dummies(frame["country"]).to_numpy(dtype=float)
    y = frame[TARGET].to_numpy()
    scores = []
    for repeat in range(1, config["split"]["outer_repeats"] + 1):
        folds = frame[f"outer_fold_s{repeat}"].to_numpy()
        pred = np.empty(len(frame), dtype=object)
        for fold in np.unique(folds):
            held = folds == fold
            model = LogisticRegression(class_weight="balanced", max_iter=2000)
            pred[held] = model.fit(x[~held], y[~held]).predict(x[held])
        scores.append(f1_score(y, pred, average="macro"))
    return float(np.mean(scores))
