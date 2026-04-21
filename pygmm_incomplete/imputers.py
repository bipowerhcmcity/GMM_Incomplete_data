from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numpy(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def mean_impute(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Column-wise mean imputation for NaN values."""
    arr = _to_numpy(x).copy()
    col_means = np.nanmean(arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(arr))
    arr[inds] = col_means[inds[1]]
    return arr


def zero_impute(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Replace NaN values with 0."""
    arr = _to_numpy(x).copy()
    arr[np.isnan(arr)] = 0.0
    return arr
