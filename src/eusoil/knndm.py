"""k-fold nearest-neighbour distance matching (kNNDM) cross-validation.

The protocol's prediction domain is unsampled farmland in the EU (amendment
A10). A cross-validation is representative of that use when the distances
from held-out points to their nearest training point look like the distances
from the prediction domain to its nearest sample. kNNDM (Linnenbrink et al.,
2024) builds k folds to make the two distributions match:

1. ``g_ij``: for every prediction location, the distance to the nearest
   sample point (the target distribution).
2. The sample points are clustered by location (Ward linkage on the
   projected coordinates). Cutting the tree into ``q`` clusters, for many
   values of ``q`` between ``k`` and ``n``, gives candidate partitions from
   strongly clustered (few large clusters) to random (one point per cluster).
3. Each partition's clusters are merged into ``k`` folds of balanced size.
4. ``g_j*``: for every sample point, the distance to the nearest point in a
   different fold (what the cross-validation will face).
5. The candidate whose ``g_j*`` is closest to ``g_ij`` by the Wasserstein
   (earth mover's) distance ``W`` is chosen.

Folds larger than ``max_fold_share`` of the sample are not allowed, as in
CAST's ``knndm`` (argument ``maxp``). This module re-implements the method
in Python following the paper; it is not a port of the R code, so results are
comparable in kind, not identical.

Reference: Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. (2024). kNNDM
CV: k-fold nearest-neighbour distance matching cross-validation for map
accuracy estimation. Geoscientific Model Development, 17, 5897-5912.
https://doi.org/10.5194/gmd-17-5897-2024
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance


def nearest_distances(from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
    """Distance from each row of ``from_xy`` to its nearest row of ``to_xy``."""
    if len(to_xy) == 0:
        raise ValueError("No points to measure distances to")
    distance, _ = cKDTree(to_xy).query(from_xy)
    return distance


def cv_distances(xy: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """``g_j*``: for each point, the distance to the nearest point in another fold."""
    out = np.empty(len(xy))
    for fold in np.unique(folds):
        held_out = folds == fold
        out[held_out] = nearest_distances(xy[held_out], xy[~held_out])
    return out


def merge_clusters(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Merge cluster labels into ``k`` folds of near-equal size.

    Clusters are visited from largest to smallest, ties in random order, and
    each goes to the currently smallest fold (ties broken at random). The
    seed therefore changes which clusters share a fold without changing how
    the sample is clustered.
    """
    clusters, sizes = np.unique(labels, return_counts=True)
    if len(clusters) < k:
        raise ValueError(f"{len(clusters)} clusters cannot fill {k} folds")
    order = np.lexsort((rng.random(len(clusters)), -sizes))
    fold_of_cluster = {}
    fold_sizes = np.zeros(k, dtype=int)
    for index in order:
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(smallest))
        fold_of_cluster[clusters[index]] = fold
        fold_sizes[fold] += sizes[index]
    return np.array([fold_of_cluster[c] for c in labels], dtype=int)


def candidate_cluster_counts(n: int, k: int, n_candidates: int) -> np.ndarray:
    """Unique cluster counts from ``k`` to ``n``, spaced evenly on a log scale."""
    q = np.unique(np.round(np.geomspace(k, n, n_candidates)).astype(int))
    return q[(q >= k) & (q <= n)]


@dataclass
class KnndmResult:
    """Chosen folds and the evidence for the choice."""

    folds: np.ndarray  # fold index 0..k-1 per point, in input order
    q: int  # number of clusters of the chosen partition
    w: float  # Wasserstein distance between g_j* and g_ij for the chosen folds
    w_random: float  # the same for random k-fold CV (reference)
    g_ij: np.ndarray  # prediction-to-sample distances (the target)
    g_cv: np.ndarray  # held-out-to-training distances of the chosen folds
    g_random: np.ndarray  # the same for random k-fold CV
    candidates: pd.DataFrame  # one row per candidate: q, w, largest fold share


def knndm(
    xy: np.ndarray,
    g_ij: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_candidates: int = 100,
    max_fold_share: float = 0.5,
    tree: np.ndarray | None = None,
) -> KnndmResult:
    """Choose ``k`` folds for the points ``xy`` whose CV distances best match ``g_ij``.

    ``xy`` are projected coordinates (any metric unit, the same as
    ``g_ij``). ``tree`` is an optional precomputed Ward linkage of ``xy``,
    for repeated calls with different seeds. Memory for the linkage grows
    with n squared (about 0.4 GB for 10,000 points).
    """
    xy = np.asarray(xy, dtype="float64")
    n = len(xy)
    if n < 2 * k:
        raise ValueError(f"{n} points are too few for {k} folds")
    if tree is None:
        tree = linkage(xy, method="ward")
    q_values = candidate_cluster_counts(n, k, n_candidates)

    rows = []
    best = None
    for q in q_values:
        labels = fcluster(tree, t=q, criterion="maxclust") if q < n else np.arange(n)
        if len(np.unique(labels)) < k:  # ties in the tree can yield fewer clusters
            continue
        folds = merge_clusters(labels, k, rng)
        share = np.bincount(folds, minlength=k).max() / n
        if share > max_fold_share:
            rows.append({"q": int(q), "w": np.nan, "largest_fold_share": share})
            continue
        g_cv = cv_distances(xy, folds)
        w = wasserstein_distance(g_cv, g_ij)
        rows.append({"q": int(q), "w": w, "largest_fold_share": share})
        if best is None or w < best[0]:
            best = (w, int(q), folds, g_cv)
    if best is None:
        raise ValueError("Every candidate partition exceeds the maximum fold share")

    random_folds = rng.permutation(np.arange(n) % k)
    g_random = cv_distances(xy, random_folds)
    w_random = wasserstein_distance(g_random, g_ij)
    w, q, folds, g_cv = best
    return KnndmResult(
        folds=folds,
        q=q,
        w=w,
        w_random=w_random,
        g_ij=np.asarray(g_ij, dtype="float64"),
        g_cv=g_cv,
        g_random=g_random,
        candidates=pd.DataFrame(rows),
    )


def distance_summary(result: KnndmResult, label: str) -> pd.DataFrame:
    """Quantiles of the target and CV distance distributions (km), for the reports."""
    probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    for name, values in (
        ("prediction_to_sample", result.g_ij),
        ("cv_heldout_to_train", result.g_cv),
    ):
        quantiles = np.quantile(values, probs) / 1000.0
        rows.append(
            {"split": label, "distribution": name, "n": len(values)}
            | {
                f"q{int(p * 100):02d}_km": round(v, 2)
                for p, v in zip(probs, quantiles, strict=True)
            }
        )
    return pd.DataFrame(rows)
