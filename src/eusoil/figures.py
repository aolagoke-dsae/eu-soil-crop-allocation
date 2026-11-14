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
