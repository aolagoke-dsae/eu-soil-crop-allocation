"""Precision check for H1, run before any model is fitted.

The protocol asks: with the real class counts and spatial layout, how wide
will the 95% interval of a macro-F1 difference be under the paired spatial
block bootstrap? If it is wider than ``config['precision']['max_ci_width']``,
a true gain of 0.02 (the SESOI) cannot be told apart from zero, and H1 must
be reported as underpowered before results are seen.

The simulation uses only the labels and coordinates, never soil values:

1. Each point's probability of being classified correctly is drawn on the
   logit scale with a spatial block effect, ``logit(p) = mu + u_block``,
   ``u ~ N(0, sigma^2)``, so errors cluster in space as they will in real
   models. ``mu`` is set so the mean accuracy is ``base_accuracy``.
2. Model A predicts the true class with probability ``p``, otherwise a
   wrong class drawn in proportion to class frequency.
3. Model B is model A with a share of A's errors corrected, chosen by
   bisection so that B's macro F1 exceeds A's by the SESOI.
4. Whole blocks are resampled with replacement; the 2.5-97.5 percentile
   range of macro-F1(B) - macro-F1(A) is the interval width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro F1 from a K x K confusion matrix (rows true, columns predicted)."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    denom = predicted + actual
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[actual > 0].mean())


def block_ids(easting: np.ndarray, northing: np.ndarray, size_m: float) -> np.ndarray:
    """Integer id of the square block containing each point."""
    bx = np.floor(easting / size_m).astype(np.int64)
    by = np.floor(northing / size_m).astype(np.int64)
    _, ids = np.unique(np.column_stack([bx, by]), axis=0, return_inverse=True)
    return ids.ravel()


def per_block_confusion(y: np.ndarray, pred: np.ndarray, blocks: np.ndarray, k: int) -> np.ndarray:
    """Confusion matrices per block, shape (n_blocks, k, k)."""
    n_blocks = int(blocks.max()) + 1
    flat = blocks * k * k + y * k + pred
    counts = np.bincount(flat, minlength=n_blocks * k * k)
    return counts.reshape(n_blocks, k, k)


@dataclass
class SimulatedPair:
    y: np.ndarray
    pred_a: np.ndarray
    pred_b: np.ndarray


def simulate_pair(
    y: np.ndarray,
    blocks: np.ndarray,
    k: int,
    base_accuracy: float,
    sigma: float,
    sesoi: float,
    rng: np.random.Generator,
) -> SimulatedPair:
    """Simulate predictions of two models whose macro F1 differ by about ``sesoi``."""
    n_blocks = int(blocks.max()) + 1
    effect = rng.normal(0.0, sigma, n_blocks)[blocks] if sigma > 0 else np.zeros(len(y))
    mu = logit(base_accuracy)
    # Shift mu so that the mean accuracy, not the median, equals base_accuracy.
    for _ in range(30):
        mu += logit(base_accuracy) - logit(expit(mu + effect).mean())
    p_correct = expit(mu + effect)
    correct = rng.random(len(y)) < p_correct

    class_freq = np.bincount(y, minlength=k) / len(y)
    pred_a = y.copy()
    # A wrong label drawn in proportion to the frequency of the other classes.
    for cls in range(k):
        idx = np.flatnonzero(~correct & (y == cls))
        if idx.size == 0:
            continue
        others = np.delete(np.arange(k), cls)
        weights = class_freq[others] / class_freq[others].sum()
        pred_a[idx] = rng.choice(others, size=idx.size, p=weights)

    errors = np.flatnonzero(pred_a != y)
    order = rng.permutation(errors)
    f1_a = macro_f1_from_confusion(np.bincount(y * k + pred_a, minlength=k * k).reshape(k, k))

    def f1_with_fixed(share: float) -> tuple[float, np.ndarray]:
        pred_b = pred_a.copy()
        fixed = order[: int(round(share * order.size))]
        pred_b[fixed] = y[fixed]
        cm = np.bincount(y * k + pred_b, minlength=k * k).reshape(k, k)
        return macro_f1_from_confusion(cm), pred_b

    low, high = 0.0, 1.0
    pred_b = pred_a
    for _ in range(40):
        mid = (low + high) / 2
        f1_b, pred_b = f1_with_fixed(mid)
        if f1_b - f1_a < sesoi:
            low = mid
        else:
            high = mid
    return SimulatedPair(y=y, pred_a=pred_a, pred_b=pred_b)


def bootstrap_difference(
    pair: SimulatedPair, blocks: np.ndarray, k: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Paired block-bootstrap distribution of macro-F1(B) - macro-F1(A)."""
    cm_a = per_block_confusion(pair.y, pair.pred_a, blocks, k)
    cm_b = per_block_confusion(pair.y, pair.pred_b, blocks, k)
    n_blocks = cm_a.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        diffs[b] = macro_f1_from_confusion(
            np.tensordot(weights, cm_b, axes=1)
        ) - macro_f1_from_confusion(np.tensordot(weights, cm_a, axes=1))
    return diffs


def precision_check(
    table: pd.DataFrame,
    settings: dict,
    seed: int,
    base_accuracy: float = 0.5,
    sigmas: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    """Expected 95% interval width of a macro-F1 difference, by block size and clustering."""
    rng = np.random.default_rng(seed)
    classes, y = np.unique(table["crop_group"].to_numpy(), return_inverse=True)
    k = len(classes)
    rows = []
    for size_km in settings["block_sizes_km"]:
        blocks = block_ids(
            table["easting"].to_numpy(), table["northing"].to_numpy(), size_km * 1000.0
        )
        for sigma in sigmas:
            widths = []
            for _ in range(settings["n_replicates"]):
                pair = simulate_pair(
                    y, blocks, k, base_accuracy, sigma, settings["sesoi_macro_f1"], rng
                )
                diffs = bootstrap_difference(pair, blocks, k, settings["n_bootstrap"], rng)
                low, high = np.percentile(diffs, [2.5, 97.5])
                widths.append(high - low)
            widths_arr = np.asarray(widths)
            rows.append(
                {
                    "block_km": size_km,
                    "sigma_block_logit": sigma,
                    "n_blocks": int(blocks.max()) + 1,
                    "n_points": len(y),
                    "n_classes": k,
                    "median_ci_width": float(np.median(widths_arr)),
                    "p90_ci_width": float(np.percentile(widths_arr, 90)),
                    "resolvable": bool(np.median(widths_arr) <= settings["max_ci_width"]),
                }
            )
    return pd.DataFrame(rows)
