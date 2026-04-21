from __future__ import annotations

import numpy as np


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    return x2 + c2 - 2.0 * x @ centers.T


def kmeans_plus_plus_init(x: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n_samples, n_features = x.shape
    centers = np.empty((n_clusters, n_features), dtype=float)
    first = rng.integers(0, n_samples)
    centers[0] = x[first]

    closest_dist_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, n_clusters):
        total = closest_dist_sq.sum()
        if total <= 0:
            centers[c] = x[rng.integers(0, n_samples)]
        else:
            probs = closest_dist_sq / total
            next_idx = rng.choice(n_samples, p=probs)
            centers[c] = x[next_idx]
        new_dist_sq = np.sum((x - centers[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
    return centers


def run_kmeans(
    x: np.ndarray,
    n_clusters: int,
    random_state: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple from-scratch K-Means used for initialization."""
    rng = np.random.default_rng(random_state)
    centers = kmeans_plus_plus_init(x, n_clusters, rng)

    prev_inertia = np.inf
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(max_iter):
        dist_sq = _squared_distances(x, centers)
        labels = np.argmin(dist_sq, axis=1)
        inertia = np.take_along_axis(dist_sq, labels[:, None], axis=1).sum()

        new_centers = centers.copy()
        for k in range(n_clusters):
            members = x[labels == k]
            if len(members) == 0:
                new_centers[k] = x[rng.integers(0, x.shape[0])]
            else:
                new_centers[k] = members.mean(axis=0)

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if abs(prev_inertia - inertia) <= tol and shift <= tol:
            break
        prev_inertia = inertia

    return labels, centers
