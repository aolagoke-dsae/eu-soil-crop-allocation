"""Figures for the Phase 2 reports. Each shows distributions or fitted curves only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eusoil.knndm import KnndmResult  # noqa: E402
from eusoil.variogram import exponential_model  # noqa: E402

INK = "#1f2933"
TARGET = "#4c6ef5"
CHOSEN = "#e8590c"
RANDOM = "#868e96"


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values) / 1000.0
    return x, np.arange(1, len(x) + 1) / len(x)


def knndm_ecdf(result: KnndmResult, title: str, path: Path) -> None:
    """ECDFs of prediction-to-sample, chosen-fold and random-CV distances (km)."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for values, label, colour, style in (
        (result.g_ij, "Prediction domain to nearest sample", TARGET, "-"),
        (result.g_cv, f"kNNDM folds (W = {result.w / 1000:.2f} km)", CHOSEN, "-"),
        (result.g_random, f"Random k-fold (W = {result.w_random / 1000:.2f} km)", RANDOM, "--"),
    ):
        x, y = _ecdf(values)
        ax.step(x, y, where="post", color=colour, linestyle=style, linewidth=1.6, label=label)
    upper = np.quantile(np.concatenate([result.g_ij, result.g_cv, result.g_random]), 0.99) / 1000
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance to the nearest training point (km)")
    ax.set_ylabel("Cumulative share")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def variogram_panels(
    empirical: pd.DataFrame, fits: pd.DataFrame, block_km: float, path: Path
) -> None:
    """Empirical indicator variograms with fitted models, one panel per crop group."""
    groups = list(fits["crop_group"])
    cols = 4
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows), dpi=150, squeeze=False)
    for ax, (_, fit) in zip(axes.flat, fits.iterrows(), strict=False):
        data = empirical[
            (empirical["crop_group"] == fit["crop_group"]) & (empirical["n_pairs"] > 0)
        ]
        h = data["lag"].to_numpy() / 1000
        ax.plot(h, data["gamma"], "o", ms=2.5, color=INK, alpha=0.7)
        grid = np.linspace(0, h.max(), 200)
        ax.plot(
            grid,
            exponential_model(grid * 1000, fit["nugget"], fit["partial_sill"], fit["range"]),
            color=CHOSEN,
            linewidth=1.5,
        )
        ax.axvline(fit["range"] / 1000, color=TARGET, linestyle=":", linewidth=1)
        ax.set_title(
            f"{fit['crop_group'].replace('_', ' ')}\nrange {fit['range'] / 1000:.0f} km"
            + ("" if fit["fit_flag"] == "ok" else f" ({fit['fit_flag'].replace('_', ' ')})"),
            fontsize=8,
            loc="left",
        )
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(groups) :]:
        ax.axis("off")
    fig.supxlabel("Lag (km)", fontsize=9)
    fig.supylabel("Semivariance of the class indicator", fontsize=9)
    fig.suptitle(
        f"Indicator variograms, training points; bootstrap block {block_km:.0f} km",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- EDA (training points only; distributions and aggregates) -------------------

GROUP_COLOURS = [
    "#4c6ef5",
    "#e8590c",
    "#2b8a3e",
    "#ae3ec9",
    "#f59f00",
    "#0b7285",
    "#c2255c",
    "#5c940d",
]


def _label(name: str) -> str:
    return name.replace("_", " ")


# Distribution figures are drawn between these quantiles: a minimum or maximum
# is one identifiable point's value, which the disclosure rule keeps out (A21).
INNER_QUANTILES = (0.01, 0.99)


def inner_values(values: pd.Series | np.ndarray) -> np.ndarray:
    """The values between the 1st and 99th percentiles (inclusive)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    low, high = np.quantile(values, INNER_QUANTILES)
    return values[(values >= low) & (values <= high)]


def histograms(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """One histogram per predictor, on the modelling (transformed) scale.

    Each panel spans the 1st to 99th percentile, so no single point's extreme
    value is drawn; the title gives the full count.
    """
    cols = 5
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.3 * rows), dpi=150, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        values = df[column].dropna()
        ax.hist(inner_values(values), bins=40, color=TARGET, alpha=0.85)
        ax.set_title(f"{column}  (n = {len(values):,})", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(columns) :]:
        ax.axis("off")
    fig.suptitle(
        "Predictor distributions, training points (1st to 99th percentile)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def class_violins(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Per-crop-group violins for each predictor, groups in alphabetical order.

    Each violin is drawn from its group's 1st to 99th percentile, so its tips
    are not single points' values.
    """
    groups = sorted(df["crop_group"].unique())
    cols = 3
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows), dpi=150, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        data = [inner_values(df.loc[df["crop_group"] == g, column]) for g in groups]
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
        for body, colour in zip(parts["bodies"], GROUP_COLOURS, strict=False):
            body.set_facecolor(colour)
            body.set_alpha(0.6)
        parts["cmedians"].set_color(INK)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([_label(g) for g in groups], rotation=35, ha="right", fontsize=6.5)
        ax.set_title(column, fontsize=9, loc="left")
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(columns) :]:
        ax.axis("off")
    fig.suptitle(
        "Predictors by crop group, training points (1st to 99th percentile of each group)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path)
    plt.close(fig)


def correlation_panels(matrices: dict[str, pd.DataFrame], path: Path) -> None:
    """Pooled, within-crop-group and within-country Spearman matrices side by side."""
    names = list(matrices)
    labels = list(matrices[names[0]].columns)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), dpi=150)
    for ax, name in zip(axes, names, strict=True):
        image = ax.imshow(matrices[name].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(_label(name), fontsize=10, loc="left")
    fig.colorbar(image, ax=axes, shrink=0.6, label="Spearman rho")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def icc_bars(table: pd.DataFrame, path: Path) -> None:
    """Share of each predictor's variance between countries and between NUTS-2 regions."""
    data = table.sort_values("icc_nuts2")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2), dpi=150)
    ax.barh(y + 0.2, data["icc_nuts2"], height=0.38, color=TARGET, label="Between NUTS-2 regions")
    ax.barh(y - 0.2, data["icc_country"], height=0.38, color=CHOSEN, label="Between countries")
    ax.set_yticks(y)
    ax.set_yticklabels(data["variable"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Intraclass correlation (share of variance)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Variance between regions, training points", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def coverage_bars(table: pd.DataFrame, path: Path) -> None:
    """Share of each country's points inside the other countries' climate-and-soil hull."""
    data = table.dropna().sort_values("share_inside_others")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(data) + 1.2), dpi=150)
    colours = [CHOSEN if s < 0.9 else TARGET for s in data["share_inside_others"]]
    ax.barh(data["country"], data["share_inside_others"], color=colours)
    ax.axvline(0.9, color=INK, linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the country's points inside the other countries' hull (3 PCs)")
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Environmental coverage for leave-one-country-out", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def crop_group_maps(df: pd.DataFrame, outline, path: Path) -> None:
    """One small map per crop group: its points over all training points."""
    groups = sorted(df["crop_group"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), dpi=150)
    for ax, group, colour in zip(axes.flat, groups, GROUP_COLOURS, strict=False):
        if outline is not None:
            outline.boundary.plot(ax=ax, color="#adb5bd", linewidth=0.3)
        ax.scatter(df["easting"], df["northing"], s=0.3, color="#dee2e6")
        mine = df[df["crop_group"] == group]
        ax.scatter(mine["easting"], mine["northing"], s=1.2, color=colour)
        ax.set_title(f"{_label(group)} (n = {len(mine):,})", fontsize=8, loc="left")
        _map_frame(ax, df)
    fig.suptitle("Crop groups, training points", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gridded_maps(grids: dict[str, pd.DataFrame], cell_m: float, outline, df, path: Path) -> None:
    """Median of each soil variable per grid cell (cells with too few points left blank)."""
    names = list(grids)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.3 * rows), dpi=150, squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        grid = grids[name]
        if outline is not None:
            outline.boundary.plot(ax=ax, color="#adb5bd", linewidth=0.3)
        image = _cell_mesh(ax, grid, cell_m)
        fig.colorbar(image, ax=ax, shrink=0.7)
        ax.set_title(f"{name}: median per {cell_m / 1000:.0f} km cell", fontsize=8, loc="left")
        _map_frame(ax, df)
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Soil variables, training points: cells with fewer than the minimum points are blank",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _cell_mesh(ax, grid: pd.DataFrame, cell_m: float):
    """Draw grid-cell medians as square cells of their true size."""
    ix = np.round((grid["x"] - cell_m / 2) / cell_m).astype(int)
    iy = np.round((grid["y"] - cell_m / 2) / cell_m).astype(int)
    x0, y0 = ix.min(), iy.min()
    values = np.full((iy.max() - y0 + 1, ix.max() - x0 + 1), np.nan)
    values[iy - y0, ix - x0] = grid["median"]
    xs = (np.arange(values.shape[1] + 1) + x0) * cell_m
    ys = (np.arange(values.shape[0] + 1) + y0) * cell_m
    return ax.pcolormesh(xs, ys, np.ma.masked_invalid(values), cmap="viridis")


def _map_frame(ax, df: pd.DataFrame) -> None:
    pad = 100_000
    ax.set_xlim(df["easting"].min() - pad, df["easting"].max() + pad)
    ax.set_ylim(df["northing"].min() - pad, df["northing"].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def reliability_panels(table: pd.DataFrame, title: str, path: Path) -> None:
    """Reliability diagram per crop group: observed frequency against predicted probability."""
    groups = sorted(table["crop_group"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=150)
    for ax, group, colour in zip(axes.flat, groups, GROUP_COLOURS, strict=False):
        data = table[(table["crop_group"] == group) & (table["n"] >= 10)]
        ax.plot([0, 1], [0, 1], color="#adb5bd", linewidth=1)
        ax.plot(data["mean_predicted"], data["observed"], "o-", color=colour, ms=3, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(_label(group), fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Mean predicted probability (bins with at least 10 points)", fontsize=9)
    fig.supylabel("Observed frequency", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stage_a_scores(summary: pd.DataFrame, selected: str, path: Path) -> None:
    """Mean outer-fold macro F1 with 95% intervals per candidate, selected model highlighted."""
    data = summary.sort_values("macro_f1_mean")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(data) + 1.3), dpi=150)
    colours = [CHOSEN if m == selected else TARGET for m in data["model"]]
    ax.errorbar(
        data["macro_f1_mean"],
        y,
        xerr=[
            data["macro_f1_mean"] - data["macro_f1_lo"],
            data["macro_f1_hi"] - data["macro_f1_mean"],
        ],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    ax.scatter(data["macro_f1_mean"], y, color=colours, zorder=3, s=30)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(m) for m in data["model"]], fontsize=8)
    ax.set_xlabel("Macro F1, mean over outer kNNDM folds and repeats (95% interval)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Stage A: selected model {_label(selected)}", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


DECISION_COLOURS = {
    "supported": "#2b8a3e",
    "negligible": "#868e96",
    "inconclusive": "#f59f00",
    "reversed": "#c2255c",
}


def ladder_plot(steps: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> None:
    """Macro F1 per ladder step (95% block-bootstrap interval) and the decisions beside it."""
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.35, 1]}
    )
    y = np.arange(len(steps))[::-1]
    ax.errorbar(
        steps["macro_f1"],
        y,
        xerr=[steps["macro_f1"] - steps["ci95_low"], steps["ci95_high"] - steps["macro_f1"]],
        fmt="o",
        color=TARGET,
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}. {b}" for s, b in zip(steps["step"], steps["blocks"], strict=True)], fontsize=8
    )
    ax.set_xlabel("Macro F1 (95% spatial block bootstrap interval)")
    ax.set_title("Predictor-block ladder", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    d = decisions.reset_index(drop=True)
    yd = np.arange(len(d))[::-1]
    for yi, row in zip(yd, d.itertuples(index=False), strict=True):
        colour = DECISION_COLOURS.get(row.decision, INK)
        bx.plot([row.decision_ci_low, row.decision_ci_high], [yi, yi], color=colour, linewidth=2)
        bx.plot(row.estimate, yi, "o", color=colour)
        bx.text(row.decision_ci_high, yi + 0.18, f" {row.decision}", fontsize=7, color=colour)
    sesoi = float(d["sesoi"].iloc[0])
    bx.axvline(0, color=INK, linewidth=0.8)
    for x in (-sesoi, sesoi):
        bx.axvline(x, color=INK, linestyle=":", linewidth=0.8)
    bx.set_yticks(yd)
    bx.set_yticklabels(d["hypothesis"], fontsize=8)
    bx.set_xlabel("Difference in macro F1 (98.75% interval; dotted: SESOI)")
    bx.set_title("Hypothesis decisions", loc="left", fontsize=11)
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def e1_curve(curve: pd.DataFrame, path: Path) -> None:
    """E1: macro F1 against median distance to the nearest training point, per scheme."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=150)
    ax.errorbar(
        curve["median_distance_km"],
        curve["macro_f1"],
        yerr=[curve["macro_f1"] - curve["ci95_low"], curve["ci95_high"] - curve["macro_f1"]],
        fmt="o-",
        color=TARGET,
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    for i, row in enumerate(curve.itertuples(index=False)):
        ax.annotate(
            row.scheme.replace("_", " "),
            (row.median_distance_km, row.macro_f1),
            textcoords="offset points",
            xytext=(6, 6 if i % 2 == 0 else -12),  # alternate to keep close labels apart
            fontsize=7,
        )
    domain = float(curve["domain_median_distance_km"].iloc[0])
    ax.axvline(domain, color=CHOSEN, linestyle="--", linewidth=1)
    ax.text(domain, ax.get_ylim()[0], " prediction domain", color=CHOSEN, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Median distance to the nearest training point (km, log scale)")
    ax.set_ylabel("Macro F1 (95% interval)")
    ax.set_title("E1: performance against prediction distance", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def transfer_bars(transfer: pd.DataFrame, path: Path) -> None:
    """H3b by country: macro F1 within the surveyed countries vs leave-one-country-out."""
    data = transfer.sort_values("loss")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(data) + 1.4), dpi=150)
    ax.barh(y + 0.2, data["macro_f1_knndm"], height=0.38, color=TARGET, label="kNNDM CV")
    ax.barh(y - 0.2, data["macro_f1_loco"], height=0.38, color=CHOSEN, label="Country left out")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c} (n = {n})" for c, n in zip(data["country"], data["n"], strict=True)], fontsize=7
    )
    ax.set_xlabel("Macro F1 within the country (classes present there)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("H3b: transfer to an unseen country", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Phase 5 -------------------------------------------------------------------------


def importance_bars(grouped: pd.DataFrame, noise: pd.DataFrame, path: Path) -> None:
    """Grouped permutation importance per block and the noise control."""
    data = pd.concat([grouped, noise], ignore_index=True).sort_values("drop_macro_f1")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(data) + 1.3), dpi=150)
    colours = [RANDOM if g == "noise_control" else TARGET for g in data["group"]]
    ax.barh(y, data["drop_macro_f1"], color=colours)
    ax.errorbar(
        data["drop_macro_f1"],
        y,
        xerr=[
            data["drop_macro_f1"] - data["drop_macro_f1_lo"],
            data["drop_macro_f1_hi"] - data["drop_macro_f1"],
        ],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([_label(g) for g in data["group"]], fontsize=8)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Drop in macro F1 when the block is shuffled (95% interval over folds)")
    ax.set_title("How much the final model relies on each block", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def class_importance_heatmap(grouped: pd.DataFrame, classes: list[str], path: Path) -> None:
    """Drop in each crop group's F1 when a block is shuffled."""
    matrix = grouped.set_index("group")[[f"drop_f1__{c}" for c in classes]]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(matrix) + 2), dpi=150)
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([_label(c) for c in classes], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([_label(g) for g in matrix.index], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix.iat[i, j]:.3f}", ha="center", va="center", fontsize=6.5, color=INK
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Drop in class F1")
    ax.set_title("Per-class reliance on each block (exploratory)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ale_panels(ale: pd.DataFrame, classes: list[str], path: Path) -> None:
    """ALE of each crop group's probability along each variable (band: 2.5-97.5% over folds)."""
    variables = list(dict.fromkeys(ale["variable"]))
    fig, axes = plt.subplots(
        1, len(variables), figsize=(3.4 * len(variables), 3.6), dpi=150, squeeze=False
    )
    for ax, variable in zip(axes.flat, variables, strict=True):
        for c, colour in zip(classes, GROUP_COLOURS, strict=False):
            d = ale[(ale["variable"] == variable) & (ale["crop_group"] == c)]
            ax.plot(d["value"], d["ale"], color=colour, linewidth=1.3, label=_label(c))
            ax.fill_between(
                d["value"], d["ale_lo"], d["ale_hi"], color=colour, alpha=0.12, linewidth=0
            )
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.set_title(variable, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_ylabel("ALE of predicted probability", fontsize=8)
    axes.flat[-1].legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Accumulated local effects, final model", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion_heatmap(confusion: pd.DataFrame, path: Path) -> None:
    """Row-normalised confusion matrix (share of each true group predicted as each group)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    image = ax.imshow(confusion.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    labels = [_label(c) for c in confusion.index]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            v = confusion.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > 0.5 else INK,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Confusion matrix, kNNDM CV (rows sum to 1)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def region_map(regions, values: pd.DataFrame, key: str, outline, path: Path, title: str) -> None:
    """Choropleth of macro F1 per region; regions without enough points are left grey."""
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=150)
    if outline is not None:
        outline.plot(ax=ax, color="#f1f3f5", edgecolor="#adb5bd", linewidth=0.3)
    merged = regions.merge(values, left_on="NUTS_ID", right_on=key, how="inner")
    merged.plot(
        ax=ax,
        column="macro_f1",
        cmap="viridis",
        legend=True,
        edgecolor="white",
        linewidth=0.2,
        legend_kwds={"shrink": 0.6, "label": "Macro F1"},
    )
    minx, miny, maxx, maxy = merged.total_bounds
    ax.set_xlim(minx - 1e5, maxx + 1e5)
    ax.set_ylim(miny - 1e5, maxy + 1e5)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sensitivity_forest(summary: pd.DataFrame, path: Path) -> None:
    """Estimate and 98.75% interval of each hypothesis in every sensitivity run."""
    tests = ["H1", "H2", "H3a", "H3b"]
    data = summary[summary["hypothesis"].isin(tests)]
    runs = list(dict.fromkeys(data["run"]))
    fig, axes = plt.subplots(
        1, len(tests), figsize=(15, 0.22 * len(runs) + 1.8), dpi=150, sharey=True
    )
    y_of = {r: i for i, r in enumerate(runs[::-1])}
    for ax, test in zip(axes, tests, strict=True):
        d = data[data["hypothesis"] == test]
        for row in d.itertuples(index=False):
            colour = DECISION_COLOURS.get(row.decision, INK)
            y = y_of[row.run]
            ax.plot(
                [row.decision_ci_low, row.decision_ci_high], [y, y], color=colour, linewidth=1.5
            )
            ax.plot(row.estimate, y, "o", color=colour, ms=3)
        sesoi = 0.02
        ax.axvline(0, color=INK, linewidth=0.7)
        for x in (-sesoi, sesoi):
            ax.axvline(x, color=INK, linestyle=":", linewidth=0.7)
        ax.set_title(test, loc="left", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(list(y_of.values()))
    axes[0].set_yticklabels(list(y_of.keys()), fontsize=6.5)
    fig.supxlabel("Difference in macro F1 (98.75% interval; colour = decision)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def final_test_plot(metrics: pd.DataFrame, path: Path) -> None:
    """Sealed-test scores (95% block-bootstrap interval) beside the cross-validated ones."""
    names = ["macro_f1", "balanced_accuracy", "top2_accuracy"]
    data = metrics.set_index("metric").loc[names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    ax.errorbar(
        data["test"],
        y + 0.12,
        xerr=[data["test"] - data["ci95_low"], data["ci95_high"] - data["test"]],
        fmt="o",
        color=CHOSEN,
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        label="Sealed test set (95% interval)",
    )
    ax.plot(data["cv_knndm"], y - 0.12, "s", color=TARGET, label="Cross-validated (kNNDM)")
    ax.set_yticks(y)
    ax.set_yticklabels([_label(n) for n in names], fontsize=8)
    ax.set_xlabel("Score")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Final model on the sealed test set", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
