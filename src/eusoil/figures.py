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


def histograms(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """One histogram per predictor, on the modelling (transformed) scale."""
    cols = 5
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.3 * rows), dpi=150, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        values = df[column].dropna()
        ax.hist(values, bins=40, color=TARGET, alpha=0.85)
        ax.set_title(f"{column}  (n = {len(values):,})", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Predictor distributions, training points", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def class_violins(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Per-crop-group violins for each predictor, groups ordered by their median."""
    groups = sorted(df["crop_group"].unique())
    cols = 3
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows), dpi=150, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        data = [df.loc[df["crop_group"] == g, column].dropna().to_numpy() for g in groups]
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
    fig.suptitle("Predictors by crop group, training points", x=0.01, ha="left", fontsize=11)
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
