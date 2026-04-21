from __future__ import annotations

import numpy as np


def make_synthetic_gmm(
    n_samples: int = 600,
    n_features: int = 6,
    n_clusters: int = 3,
    cluster_std: float = 0.7,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian-mixture data for demos/tests."""
    rng = np.random.default_rng(random_state)
    priors = rng.random(n_clusters)
    priors /= priors.sum()

    counts = np.floor(priors * n_samples).astype(int)
    counts[0] += n_samples - counts.sum()

    means = rng.normal(0.0, 4.0, size=(n_clusters, n_features))
    x_parts = []
    y_parts = []
    for k, count in enumerate(counts):
        cov_diag = cluster_std * (0.8 + 0.4 * rng.random(n_features))
        part = rng.normal(loc=means[k], scale=cov_diag, size=(count, n_features))
        x_parts.append(part)
        y_parts.append(np.full(count, k, dtype=int))

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


def inject_missing_at_random(
    x: np.ndarray,
    missing_ratio: float,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly set a percentage of entries to NaN."""
    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError("missing_ratio must be in [0, 1).")

    rng = np.random.default_rng(random_state)
    out = np.array(x, dtype=float, copy=True)
    mask = rng.random(out.shape) < missing_ratio
    out[mask] = np.nan
    return out, mask
