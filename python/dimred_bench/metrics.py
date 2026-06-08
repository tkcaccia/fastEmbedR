from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors


def sample_for_metrics(X: np.ndarray, y: np.ndarray | None, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if X.shape[0] <= max_n:
        keep = np.arange(X.shape[0])
        return X, y, keep
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(X.shape[0], size=max_n, replace=False))
    return X[keep], None if y is None else y[keep], keep


def _ranks(dist: np.ndarray) -> np.ndarray:
    order = np.argsort(dist, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(dist.shape[0])[:, None]
    ranks[rows, order] = np.arange(dist.shape[1])[None, :]
    return ranks


def continuity_score(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15) -> float:
    n = X_high.shape[0]
    k = min(n_neighbors, n - 2)
    if k < 1:
        return float("nan")
    d_high = pairwise_distances(X_high)
    d_low = pairwise_distances(X_low)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    high_order = np.argsort(d_high, axis=1)[:, :k]
    low_order = np.argsort(d_low, axis=1)[:, :k]
    low_ranks = _ranks(d_low)
    penalty = 0.0
    for i in range(n):
        missing = set(high_order[i]) - set(low_order[i])
        penalty += sum(low_ranks[i, j] - k for j in missing)
    denom = n * k * (2 * n - 3 * k - 1)
    return float(1.0 - 2.0 * penalty / denom)


def knn_preservation_score(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15) -> float:
    k = min(n_neighbors, X_high.shape[0] - 1)
    if k < 1:
        return float("nan")
    high_idx = NearestNeighbors(n_neighbors=k + 1).fit(X_high).kneighbors(return_distance=False)[:, 1:]
    low_idx = NearestNeighbors(n_neighbors=k + 1).fit(X_low).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([
        len(set(high_idx[i]).intersection(low_idx[i])) / k
        for i in range(X_high.shape[0])
    ]))


def silhouette_or_none(X_low: np.ndarray, y: np.ndarray | None) -> float | None:
    if y is None:
        return None
    labels = np.asarray(y)
    if np.unique(labels).size < 2 or np.unique(labels).size >= labels.size:
        return None
    return float(silhouette_score(X_low, labels))


def embedding_metrics(
    X_high: np.ndarray,
    X_low: np.ndarray,
    y: np.ndarray | None,
    *,
    n_neighbors: int = 15,
) -> dict[str, float | None]:
    k = min(n_neighbors, X_high.shape[0] - 2)
    return {
        "trustworthiness": float(trustworthiness(X_high, X_low, n_neighbors=k)) if k >= 1 else None,
        "continuity": continuity_score(X_high, X_low, n_neighbors=k) if k >= 1 else None,
        "knn_preservation": knn_preservation_score(X_high, X_low, n_neighbors=k) if k >= 1 else None,
        "silhouette": silhouette_or_none(X_low, y),
    }


def procrustes_stability(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference - reference.mean(axis=0, keepdims=True)
    cand = candidate - candidate.mean(axis=0, keepdims=True)
    ref_norm = np.linalg.norm(ref)
    cand_norm = np.linalg.norm(cand)
    if ref_norm == 0 or cand_norm == 0:
        return float("nan")
    ref /= ref_norm
    cand /= cand_norm
    rotation, _ = orthogonal_procrustes(cand, ref)
    aligned = cand @ rotation
    disparity = float(np.mean((ref - aligned) ** 2))
    return float(1.0 / (1.0 + disparity))
