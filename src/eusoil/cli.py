"""Command line for Phase 1: ``python -m eusoil <command>``.

Commands, in the order they are normally run:

``points``        clean the tabular sources, write ``data/interim/points.parquet``,
                  the sample-flow table and the audit tables
``dem``           elevation and slope from Copernicus GLO-90 (needs internet access)
``terraclimate``  TerraClimate bioclimatic variables at the points (needs internet
                  access); ``--period 1991-2020`` is the primary climate block,
                  ``--period 2015-2018`` the climate-period sensitivity run
``worldclim``     WorldClim bio variables at the points (local GeoTIFFs); an
                  internal resolution check only, never published (protocol A7)
``table``         join everything into ``data/processed/modelling_table.parquet``
``precision``     the H1 precision check (simulation on labels and coordinates)

Phase 2, in this order:

``domain``        the prediction domain: unsampled 2018 farmland points
``seal``          kNNDM split of the sample; seals the test set (refuses to re-seal)
``variogram``     indicator variograms on the training points; bootstrap block size
``folds``         kNNDM outer folds on the training points (3 seeds)
``eda``           exploratory analysis on the training points (never the sealed test set)

Phase 3:

``stage-a``         tune and score every candidate on the outer kNNDM folds (resumable)
``stage-a-report``  summary, the one-SE selection, calibration check
``stage-b-ladder``  the predictor-block ladder, retuned per fold (H1, H2, H3a; resumable)
``stage-b-validation``  final configuration under E1's schemes and leave-one-country-out
``stage-b-report``  bootstrap estimands, decisions, Holm, E1 plateau

Each command reads the outputs of the earlier ones from ``data/interim`` and
``data/processed``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eusoil.build import (
    add_projected_coordinates,
    build_modelling_table,
    build_points,
    write_tables,
)
from eusoil.config import load_config, resolve_paths
from eusoil.flow import SampleFlow

LOG = logging.getLogger("eusoil")


def _points(paths) -> pd.DataFrame:
    path = paths.interim / "points.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: run 'python -m eusoil points' first")
    return pd.read_parquet(path)


def cmd_points(args, config, paths) -> None:
    flow = SampleFlow()
    landcover = Path(args.landcover) if args.landcover else None
    points, audit = build_points(paths, config, flow, landcover_path=landcover)
    points.to_parquet(paths.interim / "points.parquet", index=False)
    flow.save(paths.reports / "tables" / "sample_flow.csv")
    write_tables(audit, paths.reports / "tables")
    LOG.info("points: %d rows; sample flow and %d audit tables written", len(points), len(audit))


def cmd_worldclim(args, config, paths) -> None:
    from eusoil.ingest.climate import extract_worldclim

    directory = paths.worldclim_30s if args.resolution == "30s" else paths.worldclim_2_5m
    # WorldClim is an internal resolution check (protocol A7): its licence does
    # not allow redistribution, so no output derived from it is published.
    prefix = "wc30s_bio" if args.resolution == "30s" else "wc25m_bio"
    result = extract_worldclim(_points(paths), directory, prefix=prefix)
    out = paths.interim / f"worldclim_{args.resolution}.parquet"
    result.to_parquet(out, index=False)
    filled = int((result[f"{prefix}_fill_px"] > 0).sum())
    missing = int(result[f"{prefix}1"].isna().sum())
    LOG.info(
        "worldclim %s: %d points, %d filled from a neighbouring pixel, %d still missing",
        args.resolution,
        len(result),
        filled,
        missing,
    )


def cmd_dem(args, config, paths) -> None:
    from eusoil.ingest.dem import extract_dem

    result = extract_dem(_points(paths), cache=paths.interim / "dem_tiles")
    result.to_parquet(paths.interim / "dem.parquet", index=False)
    LOG.info(
        "dem: %d points, elevation missing %d, slope missing %d",
        len(result),
        int(result["elevation_m"].isna().sum()),
        int(result["slope_deg"].isna().sum()),
    )


def cmd_terraclimate(args, config, paths) -> None:
    from eusoil.ingest.terraclimate import PERIODS, extract_terraclimate

    result = extract_terraclimate(_points(paths), period=args.period)
    out = paths.interim / f"terraclimate_{args.period.replace('-', '_')}.parquet"
    result.to_parquet(out, index=False)
    tag = PERIODS[args.period]["tag"]
    LOG.info(
        "terraclimate %s: %d points, nearest-cell distance max %.1f km",
        args.period,
        len(result),
        float(result[f"{tag}_cell_km"].max()),
    )


# Interim files joined into the modelling table: file -> (column prefix it
# must carry, command that writes it). An older file without the prefix was
# written under a previous naming scheme and must be regenerated.
REQUIRED_EXTRACTIONS = {
    "terraclimate_1991_2020.parquet": ("tc9120_bio", "terraclimate --period 1991-2020"),
    "dem.parquet": ("elevation_m", "dem"),
}
OPTIONAL_EXTRACTIONS = {
    "terraclimate_2015_2018.parquet": ("tc1518_bio", "terraclimate --period 2015-2018"),
    "worldclim_30s.parquet": ("wc30s_bio", "worldclim --resolution 30s"),
    "worldclim_2.5m.parquet": ("wc25m_bio", "worldclim --resolution 2.5m"),
}


def _read_extraction(path: Path, prefix: str, command: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not any(column.startswith(prefix) for column in frame.columns):
        sys.exit(
            f"{path} has no '{prefix}*' columns: it predates the current naming. "
            f"Re-run 'python -m eusoil {command}'."
        )
    return frame


def cmd_table(args, config, paths) -> None:
    points = _points(paths)
    extractions = []
    for name, (prefix, command) in REQUIRED_EXTRACTIONS.items():
        path = paths.interim / name
        if not path.exists():
            sys.exit(f"{path} not found: run 'python -m eusoil {command}' first")
        extractions.append((name, _read_extraction(path, prefix, command)))
    for name, (prefix, command) in OPTIONAL_EXTRACTIONS.items():
        path = paths.interim / name
        if path.exists():
            extractions.append((name, _read_extraction(path, prefix, command)))
        else:
            LOG.warning("%s not found; the table is built without it", name)
    try:
        table = build_modelling_table(points, extractions, config)
    except ValueError as error:
        sys.exit(str(error))
    table.to_parquet(paths.processed / "modelling_table.parquet", index=False)
    LOG.info("modelling table: %d rows, %d columns", *table.shape)


def cmd_precision(args, config, paths) -> None:
    from eusoil.precision import precision_check

    table_path = paths.processed / "modelling_table.parquet"
    table = (
        pd.read_parquet(table_path)
        if table_path.exists()
        else add_projected_coordinates(_points(paths), config["crs"])
    )
    result = precision_check(table, config["precision"], seed=config["seed"])
    out = paths.reports / "tables" / "precision_check.csv"
    result.to_csv(out, index=False)
    LOG.info("precision check written to %s\n%s", out, result.to_string(index=False))


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _projected_points(paths, config) -> pd.DataFrame:
    points = add_projected_coordinates(_points(paths), config["crs"])
    points["country"] = points["nuts0"]
    return points.reset_index(drop=True)


def _domain_xy(paths) -> np.ndarray:
    path = paths.interim / "domain.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: run 'python -m eusoil domain' first")
    return pd.read_parquet(path, columns=["easting", "northing"]).to_numpy()


def _splits_path(paths) -> Path:
    return paths.processed / "splits.parquet"


def cmd_domain(args, config, paths) -> None:
    from eusoil.ingest.landcover import read_survey_locations
    from eusoil.split import build_domain

    survey = read_survey_locations(paths.landcover_2018)
    domain = build_domain(survey, _points(paths)["point_id"], config)
    domain = add_projected_coordinates(domain, config["crs"])
    domain.to_parquet(paths.interim / "domain.parquet", index=False)
    by_country = domain.groupby("nuts0").size().rename("n_domain_points")
    by_country.to_csv(paths.reports / "tables" / "domain_by_country.csv")
    LOG.info("domain: %d unsampled farmland points in %d countries", len(domain), len(by_country))


def cmd_seal(args, config, paths) -> None:
    from eusoil.figures import knndm_ecdf
    from eusoil.knndm import distance_summary
    from eusoil.split import fold_counts, seal_test_set

    out = _splits_path(paths)
    if out.exists() and not args.force:
        sys.exit(
            f"{out} exists: the test set is already sealed. Re-sealing after any "
            "exploration breaks the protocol; pass --force only before EDA has started."
        )
    points = _projected_points(paths, config)
    splits, result, record = seal_test_set(
        points, _domain_xy(paths), config["split"], seed=config["seed"]
    )
    splits.to_parquet(out, index=False)

    tables = paths.reports / "tables"
    pd.DataFrame([record]).to_csv(tables / "split_seal.csv", index=False)
    frame = points.merge(splits, on="point_id")
    frame["set"] = np.where(frame["sealed_test"], "test", "train")
    fold_counts(frame, "set", "crop_group").to_csv(tables / "split_seal_by_crop_group.csv")
    fold_counts(frame, "set", "country").to_csv(tables / "split_seal_by_country.csv")
    distance_summary(result, "seal").to_csv(tables / "knndm_distances_seal.csv", index=False)
    result.candidates.to_csv(tables / "knndm_candidates_seal.csv", index=False)
    knndm_ecdf(
        result,
        "Sealed test split: kNNDM vs random CV",
        paths.reports / "figures" / "knndm_seal.png",
    )
    LOG.info(
        "sealed: %d of %d points (%.1f%%), q = %d clusters, W = %.2f km (random CV %.2f km); "
        "countries without test points: %s; SHA-256 %s",
        record["n_test"],
        record["n_points"],
        100 * record["test_share"],
        record["q"],
        record["w"] / 1000,
        record["w_random_cv"] / 1000,
        record["countries_without_test_points"] or "none",
        record["sha256_test_ids"],
    )


def _training_points(paths, config) -> pd.DataFrame:
    path = _splits_path(paths)
    if not path.exists():
        sys.exit(f"{path} not found: run 'python -m eusoil seal' first")
    splits = pd.read_parquet(path)
    points = _projected_points(paths, config).merge(splits, on="point_id", validate="one_to_one")
    return points[~points["sealed_test"]].reset_index(drop=True)


def cmd_variogram(args, config, paths) -> None:
    from eusoil.figures import variogram_panels
    from eusoil.variogram import bootstrap_block_km, class_variograms

    settings = config["variogram"]
    train = _training_points(paths, config)
    empirical, fits = class_variograms(
        train[["easting", "northing"]].to_numpy(),
        train["crop_group"],
        max_lag=settings["max_lag_km"] * 1000.0,
        bin_width=settings["bin_width_km"] * 1000.0,
    )
    median_km = float(fits["range"].median()) / 1000
    ok = fits["fit_flag"].eq("ok")
    median_ok_km = float(fits.loc[ok, "range"].median()) / 1000 if ok.any() else float("nan")
    block = bootstrap_block_km(median_km, settings["block_step_km"], settings["block_cap_km"])
    tables = paths.reports / "tables"
    empirical.to_csv(tables / "variogram_empirical.csv", index=False)
    fits.assign(range_km=fits["range"] / 1000).to_csv(tables / "variogram_fits.csv", index=False)
    pd.DataFrame(
        [
            {
                "median_range_km": round(median_km, 1),
                "bootstrap_block_km": block,
                "median_range_ok_fits_km": round(median_ok_km, 1),
                "n_classes_flagged": int((~ok).sum()),
                "n_train": len(train),
            }
        ]
    ).to_csv(tables / "bootstrap_block.csv", index=False)
    variogram_panels(empirical, fits, block, paths.reports / "figures" / "variograms.png")
    LOG.info(
        "variograms: median range %.1f km -> bootstrap block %.0f km "
        "(median of the %d unflagged fits: %.1f km; flags: %s)",
        median_km,
        block,
        int(ok.sum()),
        median_ok_km,
        ", ".join(
            f"{g} {f}"
            for g, f in zip(fits["crop_group"], fits["fit_flag"], strict=True)
            if f != "ok"
        )
        or "none",
    )


def cmd_folds(args, config, paths) -> None:
    from eusoil.figures import knndm_ecdf
    from eusoil.knndm import distance_summary
    from eusoil.split import fold_counts, outer_folds

    path = _splits_path(paths)
    if not path.exists():
        sys.exit(f"{path} not found: run 'python -m eusoil seal' first")
    splits = pd.read_parquet(path)
    points = _projected_points(paths, config)
    splits = splits[[c for c in splits.columns if not c.startswith("outer_fold_")]]
    splits, results = outer_folds(
        points, splits, _domain_xy(paths), config["split"], config["seed"]
    )
    splits.to_parquet(path, index=False)

    tables = paths.reports / "tables"
    frame = points.merge(splits, on="point_id")
    frame = frame[~frame["sealed_test"]]
    summaries, records = [], []
    for repeat, result in enumerate(results, start=1):
        column = f"outer_fold_s{repeat}"
        fold_counts(frame, column, "crop_group").to_csv(
            tables / f"folds_s{repeat}_by_crop_group.csv"
        )
        summaries.append(distance_summary(result, f"outer_s{repeat}"))
        records.append(
            {"repeat": repeat, "q": result.q, "w": result.w, "w_random_cv": result.w_random}
        )
    pd.concat(summaries).to_csv(tables / "knndm_distances_outer.csv", index=False)
    pd.DataFrame(records).to_csv(tables / "knndm_outer.csv", index=False)
    knndm_ecdf(
        results[0],
        "Outer folds (seed 1): kNNDM vs random CV",
        paths.reports / "figures" / "knndm_outer.png",
    )
    for record in records:
        LOG.info(
            "outer folds s%d: q = %d, W = %.2f km (random CV %.2f km)",
            record["repeat"],
            record["q"],
            record["w"] / 1000,
            record["w_random_cv"] / 1000,
        )


def cmd_eda(args, config, paths) -> None:
    from eusoil import eda, figures

    settings = config["eda"]
    table_path = paths.processed / "modelling_table.parquet"
    if not table_path.exists():
        sys.exit(f"{table_path} not found: run 'python -m eusoil table' first")
    splits_path = _splits_path(paths)
    if not splits_path.exists():
        sys.exit(f"{splits_path} not found: run 'python -m eusoil seal' first")
    splits = pd.read_parquet(splits_path)
    train = eda.training_only(pd.read_parquet(table_path), splits)
    columns = eda.continuous_predictors(config)
    tables, figs = paths.reports / "tables", paths.reports / "figures"
    LOG.info("eda: %d training points, %d continuous predictors", len(train), len(columns))

    eda.class_summaries(train, columns).to_csv(tables / "eda_class_summaries.csv", index=False)
    figures.histograms(train, columns, figs / "eda_histograms.png")
    figures.class_violins(train, columns, figs / "eda_violins.png")

    matrices, pairs = eda.correlation_structure(train, columns, settings["simpson_gap"])
    pairs.to_csv(tables / "eda_correlation_pairs.csv", index=False)
    figures.correlation_panels(matrices, figs / "eda_correlations.png")
    LOG.info(
        "eda: %d of %d predictor pairs flagged pooled vs within",
        (pairs["flag"] != "").sum(),
        len(pairs),
    )

    icc = eda.variance_partition(train, columns)
    icc.to_csv(tables / "eda_variance_partition.csv", index=False)
    figures.icc_bars(icc, figs / "eda_variance_partition.png")

    moran = eda.morans_table(
        train,
        columns,
        k=settings["moran_k"],
        n_permutations=settings["moran_permutations"],
        rng=np.random.default_rng(config["seed"]),
    )
    moran.to_csv(tables / "eda_morans_i.csv", index=False)

    eda.vif_table(train, columns).to_csv(tables / "eda_vif.csv", index=False)
    climate_all = [f"{settings['climate_all_prefix']}{i}" for i in range(1, 20)]
    clusters, check = eda.climate_clusters(
        train, climate_all, config["blocks"]["C"], settings["climate_cluster_rho"]
    )
    clusters.to_csv(tables / "eda_climate_clusters.csv", index=False)
    check.to_csv(tables / "eda_climate_subset_check.csv", index=False)
    LOG.info(
        "eda: climate subset check: %d clusters; cluster-based set differs from pre-chosen: %s",
        check.loc[0, "n_clusters"],
        check.loc[0, "differs_from_pre_chosen"],
    )

    climate_soil = [c for c in columns if c not in config["blocks"]["G"]]
    coverage = eda.coverage_table(train, climate_soil, n_components=settings["coverage_pcs"])
    coverage.to_csv(tables / "eda_coverage.csv", index=False)
    figures.coverage_bars(coverage, figs / "eda_coverage.png")

    outline = _country_outline(paths)
    figures.crop_group_maps(train, outline, figs / "eda_crop_group_maps.png")
    cell = settings["grid_cell_km"] * 1000.0
    grids = {
        v: eda.grid_medians(train, v, cell, settings["grid_min_points"])
        for v in settings["map_variables"]
    }
    figures.gridded_maps(grids, cell, outline, train, figs / "eda_soil_maps.png")
    LOG.info("eda: tables in %s, figures in %s", tables, figs)


def _country_outline(paths):
    from eusoil.ingest.nuts import read_nuts

    try:
        return read_nuts(paths.nuts, level=0)
    except (FileNotFoundError, OSError) as error:
        LOG.warning("Country outlines not drawn: %s", error)
        return None


def _stage_a_data(paths, config):
    """Training points with outer folds, complete cases on the Stage A predictors."""
    from eusoil.eda import assert_no_test_rows
    from eusoil.models import complete_cases, predictor_columns

    table_path = paths.processed / "modelling_table.parquet"
    splits_path = _splits_path(paths)
    for path, command in ((table_path, "table"), (splits_path, "seal' and 'folds")):
        if not path.exists():
            sys.exit(f"{path} not found: run 'python -m eusoil {command}' first")
    splits = pd.read_parquet(splits_path)
    if "outer_fold_s1" not in splits.columns:
        sys.exit("No outer folds yet: run 'python -m eusoil folds' first")
    table = pd.read_parquet(table_path).merge(splits, on="point_id", validate="one_to_one")
    train = table[~table["sealed_test"]].reset_index(drop=True)
    assert_no_test_rows(train, splits)
    columns = predictor_columns(config, config["stage_a"]["blocks"])
    data, dropped = complete_cases(train, columns)
    classes = sorted(config["target"]["level2"])
    return data, dropped, columns, classes


def _stage_a_dir(paths) -> Path:
    return paths.interim / "stage_a"


def cmd_stage_a(args, config, paths) -> None:
    from eusoil.models import TUNED, UNTUNED, run_stage_a

    data, dropped, columns, classes = _stage_a_data(paths, config)
    models = args.models or list(UNTUNED + TUNED)
    repeats = args.repeats or list(range(1, config["split"]["outer_repeats"] + 1))
    LOG.info(
        "stage A: %d training points (%d dropped for missing predictors), %d predictors, "
        "models %s, repeats %s",
        len(data),
        dropped,
        len(columns),
        ", ".join(models),
        repeats,
    )
    run_stage_a(
        data,
        columns,
        classes,
        _domain_xy(paths),
        config,
        models,
        repeats,
        _stage_a_dir(paths),
        n_jobs=args.n_jobs,
    )


def cmd_stage_a_report(args, config, paths) -> None:
    from eusoil import figures
    from eusoil.models import (
        FoldTask,
        apply_temperature,
        classwise_ece,
        load_records,
        reliability_table,
        select_model,
        summarise,
        temperature_for_fold,
        top_label_ece,
    )

    settings = config["stage_a"]
    data, dropped, columns, classes = _stage_a_data(paths, config)
    checkpoint = _stage_a_dir(paths)
    records = load_records(checkpoint)
    expected = config["split"]["k_outer"] * config["split"]["outer_repeats"]
    incomplete = records.groupby("model").size().loc[lambda s: s < expected]
    if not incomplete.empty:
        sys.exit(f"Stage A is incomplete: {incomplete.to_dict()} of {expected} folds")

    tables, figs = paths.reports / "tables", paths.reports / "figures"
    metrics = ["macro_f1", "balanced_accuracy", "log_loss", "brier", "top2_accuracy"]
    per_class = [c for c in records.columns if c.startswith("f1__")]
    records.drop(columns=per_class).to_csv(tables / "stage_a_folds.csv", index=False)
    summary = summarise(records, metrics + per_class)
    summary[
        ["model", "n_folds"]
        + [
            c
            for c in summary.columns
            if c.split("_")[-1] in {"mean", "se", "lo", "hi"} and not c.startswith("f1__")
        ]
    ].to_csv(tables / "stage_a_summary.csv", index=False)
    (
        summary.set_index("model")[[f"{c}_mean" for c in per_class]]
        .rename(columns=lambda c: c.removeprefix("f1__").removesuffix("_mean"))
        .to_csv(tables / "stage_a_per_class_f1.csv")
    )
    selection = select_model(summary, settings["selection_order"])
    selected = selection["selected"]
    figures.stage_a_scores(summary, selected, figs / "stage_a_scores.png")

    # Calibration of the selected model, on its pooled outer-fold predictions.
    predictions = pd.concat(
        pd.read_parquet(p) for p in sorted(checkpoint.glob(f"{selected}_s*_f*.parquet"))
    )
    proba = predictions[[f"p__{c}" for c in classes]].to_numpy()
    y = predictions["y_true"].to_numpy()
    ece_before = top_label_ece(y, proba, classes, settings["ece_bins"])
    calibration = {"model": selected, "top_label_ece": ece_before}
    calibration |= {
        f"ece__{k}": v for k, v in classwise_ece(y, proba, classes, settings["ece_bins"]).items()
    }
    reliability_table(y, proba, classes, settings["ece_bins"]).to_csv(
        tables / "stage_a_reliability.csv", index=False
    )
    figures.reliability_panels(
        reliability_table(y, proba, classes, settings["ece_bins"]),
        f"Reliability of {selected.replace('_', ' ')}, outer folds, all repeats",
        figs / "stage_a_reliability.png",
    )
    worst_class_ece = max(v for k, v in calibration.items() if k.startswith("ece__"))
    calibration["worst_class_ece"] = worst_class_ece
    if max(ece_before, worst_class_ece) > settings["ece_threshold"]:
        LOG.info(
            "stage A: ECE (top-label %.3f, worst class %.3f) above %.3f; temperature scaling",
            ece_before,
            worst_class_ece,
            settings["ece_threshold"],
        )
        scaled_parts = []
        folds = records[records["model"] == selected]
        temperatures = []
        for row in folds.itertuples(index=False):
            task = FoldTask(selected, int(row.repeat), int(row.fold))
            t = temperature_for_fold(
                task, json.loads(row.best_params), data, columns, classes, _domain_xy(paths), config
            )
            temperatures.append(t)
            part = predictions[
                (predictions["repeat"] == task.repeat) & (predictions["fold"] == task.fold)
            ]
            scaled = apply_temperature(part[[f"p__{c}" for c in classes]].to_numpy(), t)
            scaled_parts.append((part["y_true"].to_numpy(), scaled))
        y_s = np.concatenate([a for a, _ in scaled_parts])
        p_s = np.concatenate([b for _, b in scaled_parts])
        calibration |= {
            "temperature_median": float(np.median(temperatures)),
            "top_label_ece_after": top_label_ece(y_s, p_s, classes, settings["ece_bins"]),
            "worst_class_ece_after": max(
                classwise_ece(y_s, p_s, classes, settings["ece_bins"]).values()
            ),
        }
        figures.reliability_panels(
            reliability_table(y_s, p_s, classes, settings["ece_bins"]),
            f"Reliability of {selected.replace('_', ' ')} after temperature scaling",
            figs / "stage_a_reliability_scaled.png",
        )
    pd.DataFrame([calibration]).to_csv(tables / "stage_a_calibration.csv", index=False)
    pd.DataFrame(
        [
            selection
            | {"n_points": len(data), "n_dropped_missing": dropped, "predictors": ";".join(columns)}
        ]
    ).to_csv(tables / "stage_a_selection.csv", index=False)
    LOG.info(
        "stage A: best by mean %s (macro F1 %.3f, SE %.3f); selected %s (%.3f); top-label ECE %.3f",
        selection["best_by_mean"],
        selection["best_mean_macro_f1"],
        selection["best_se"],
        selected,
        selection["selected_mean_macro_f1"],
        ece_before,
    )


def _stage_b_common(paths, config):
    data, dropped, columns, classes = _stage_a_data(paths, config)
    selection = paths.reports / "tables" / "stage_a_selection.csv"
    if not selection.exists():
        sys.exit("Stage A is not reported yet: run 'python -m eusoil stage-a-report' first")
    selected = pd.read_csv(selection).loc[0, "selected"]
    if selected != config["stage_b"]["model"]:
        sys.exit(
            f"config stage_b.model is {config['stage_b']['model']!r} but Stage A selected "
            f"{selected!r}; the protocol requires the Stage A choice"
        )
    block = paths.reports / "tables" / "bootstrap_block.csv"
    if not block.exists():
        sys.exit("No bootstrap block size: run 'python -m eusoil variogram' first")
    block_km = float(pd.read_csv(block).loc[0, "bootstrap_block_km"])
    return data, classes, block_km


def _stage_b_dir(paths, part: str) -> Path:
    return paths.interim / "stage_b" / part


def cmd_stage_b_ladder(args, config, paths) -> None:
    from eusoil.stage_b import run_ladder

    data, classes, _ = _stage_b_common(paths, config)
    LOG.info("stage B ladder: %d points, model %s", len(data), config["stage_b"]["model"])
    run_ladder(
        data, classes, _domain_xy(paths), config, _stage_b_dir(paths, "ladder"),
        steps=args.steps, n_jobs=args.n_jobs,
    )  # fmt: skip


def _ladder_results(paths, config, data, classes, block_km):
    from eusoil.stage_b import ladder_analysis

    try:
        return ladder_analysis(data, classes, config, _stage_b_dir(paths, "ladder"), block_km)
    except ValueError as error:
        sys.exit(f"{error}: finish 'python -m eusoil stage-b-ladder' first")


def cmd_stage_b_validation(args, config, paths) -> None:
    from eusoil.stage_b import final_params, final_step, run_validation, step_label

    data, classes, block_km = _stage_b_common(paths, config)
    _, estimands, folds = _ladder_results(paths, config, data, classes, block_km)
    step = final_step(estimands)
    params = final_params(folds[folds["label"] == step_label(step)])
    LOG.info(
        "stage B validation: final configuration = ladder step %d, parameters %s", step, params
    )
    run_validation(
        data, classes, _domain_xy(paths), config, step, params,
        _stage_b_dir(paths, "validation"), n_jobs=args.n_jobs,
    )  # fmt: skip


def cmd_stage_b_report(args, config, paths) -> None:
    from eusoil import figures
    from eusoil.knndm import nearest_distances
    from eusoil.stage_b import (
        final_params,
        final_step,
        holm,
        step_label,
        validation_analysis,
        validation_schemes,
    )

    data, classes, block_km = _stage_b_common(paths, config)
    steps, estimands, folds = _ladder_results(paths, config, data, classes, block_km)
    step = final_step(estimands)
    params = final_params(folds[folds["label"] == step_label(step)])
    schemes = validation_schemes(data, config)
    domain_km = (
        float(
            np.median(
                nearest_distances(_domain_xy(paths), data[["easting", "northing"]].to_numpy())
            )
        )
        / 1000
    )
    try:
        h3b, transfer, curve, changes = validation_analysis(
            data, classes, config, _stage_b_dir(paths, "validation"), schemes, block_km, domain_km
        )
    except FileNotFoundError as error:
        sys.exit(f"{error}: run 'python -m eusoil stage-b-validation' first")

    decisions = pd.concat([estimands, h3b], ignore_index=True)
    adjusted = holm(dict(zip(decisions["hypothesis"], decisions["p_bootstrap"], strict=True)))
    decisions["p_holm"] = decisions["hypothesis"].map(adjusted)
    decisions["sesoi"] = config["precision"]["sesoi_macro_f1"]
    decisions["bootstrap_block_km"] = block_km

    tables, figs = paths.reports / "tables", paths.reports / "figures"
    steps.to_csv(tables / "stage_b_ladder.csv", index=False)
    folds.drop(columns=[c for c in folds.columns if c.startswith("f1__")]).to_csv(
        tables / "stage_b_ladder_folds.csv", index=False
    )
    decisions.to_csv(tables / "stage_b_decisions.csv", index=False)
    transfer.to_csv(tables / "stage_b_transfer_by_country.csv", index=False)
    curve.to_csv(tables / "stage_b_e1_curve.csv", index=False)
    changes.to_csv(tables / "stage_b_e1_plateau.csv", index=False)
    pd.DataFrame([{"final_step": step} | params]).to_csv(
        tables / "stage_b_final_configuration.csv", index=False
    )
    figures.ladder_plot(steps, decisions, figs / "stage_b_ladder.png")
    figures.e1_curve(curve, figs / "stage_b_e1.png")
    figures.transfer_bars(transfer, figs / "stage_b_transfer.png")
    for row in decisions.itertuples(index=False):
        LOG.info(
            "%s: %.3f (98.75%% interval %.3f to %.3f), p Holm %.4f -> %s",
            row.hypothesis,
            row.estimate,
            row.decision_ci_low,
            row.decision_ci_high,
            row.p_holm,
            row.decision,
        )
    LOG.info("E1 plateau: %s; final configuration: ladder step %d", changes.loc[0, "plateau"], step)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eusoil", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="path to analysis.yaml (default: config/analysis.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("points", help="clean the tabular sources and write the audit")
    p.add_argument("--landcover", help="override the harmonised 2018 table (testing)")
    p = sub.add_parser("worldclim", help="WorldClim bio variables (internal check only)")
    p.add_argument("--resolution", choices=["30s", "2.5m"], default="30s")
    sub.add_parser("dem", help="elevation and slope from Copernicus GLO-90 (internet)")
    p = sub.add_parser("terraclimate", help="TerraClimate bioclimatic variables (internet)")
    p.add_argument("--period", choices=["1991-2020", "2015-2018"], default="1991-2020")
    sub.add_parser("table", help="build the modelling table")
    sub.add_parser("precision", help="H1 precision check")
    sub.add_parser("domain", help="prediction domain: unsampled 2018 farmland points")
    p = sub.add_parser("seal", help="kNNDM split; seal the test set")
    p.add_argument("--force", action="store_true", help="re-seal (only before any EDA)")
    sub.add_parser("variogram", help="indicator variograms; bootstrap block size")
    sub.add_parser("folds", help="kNNDM outer folds on the training points")
    sub.add_parser("eda", help="exploratory analysis on the training points")
    p = sub.add_parser("stage-a", help="Stage A: tune and score the candidate models")
    p.add_argument("--models", nargs="+", help="subset of models (default: all)")
    p.add_argument("--repeats", nargs="+", type=int, help="subset of repeats (default: all)")
    p.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for the search")
    sub.add_parser("stage-a-report", help="Stage A: summary, selection, calibration")
    p = sub.add_parser("stage-b-ladder", help="Stage B: predictor-block ladder (H1, H2, H3a)")
    p.add_argument("--steps", nargs="+", type=int, help="subset of ladder steps (default: all)")
    p.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for the search")
    p = sub.add_parser("stage-b-validation", help="Stage B: E1 schemes and leave-one-country-out")
    p.add_argument(
        "--n-jobs", type=int, default=-1, help="parallel jobs inside each fit"
    )
    sub.add_parser("stage-b-report", help="Stage B: decisions, Holm, E1")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    paths = resolve_paths(config)
    paths.ensure_output_dirs()
    commands = {
        "points": cmd_points,
        "worldclim": cmd_worldclim,
        "dem": cmd_dem,
        "terraclimate": cmd_terraclimate,
        "table": cmd_table,
        "precision": cmd_precision,
        "domain": cmd_domain,
        "seal": cmd_seal,
        "variogram": cmd_variogram,
        "folds": cmd_folds,
        "eda": cmd_eda,
        "stage-a": cmd_stage_a,
        "stage-a-report": cmd_stage_a_report,
        "stage-b-ladder": cmd_stage_b_ladder,
        "stage-b-validation": cmd_stage_b_validation,
        "stage-b-report": cmd_stage_b_report,
    }
    commands[args.command](args, config, paths)


if __name__ == "__main__":
    main()
