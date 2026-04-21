from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .imputers import mean_impute
from .kmeans import run_kmeans


@dataclass
class FitResult:
    responsibilities: np.ndarray
    log_likelihood_history: list[float]
    labels: np.ndarray
    completed_data: np.ndarray


class IncompleteGMM:
    """
    Gaussian Mixture Model Clustering with Incomplete Data.

    Implemented from scratch using the alternating optimization in the paper:
    - E-step: posterior responsibilities.
    - M-step: update pi, mu, Sigma.
    - Data update: optimize missing entries with fixed model parameters.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-8,
        learning_rate: float = 1.0,
        random_state: int | None = None,
        posterior_floor: float = 1e-12,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.posterior_floor = float(posterior_floor)

        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.log_likelihood_history_: list[float] = []
        self.completed_data_: np.ndarray | None = None

    @staticmethod
    def _to_numpy(x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=float)
        return np.asarray(x, dtype=float)

    def _initialize(self, x_filled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = x_filled.shape
        labels, centers = run_kmeans(
            x_filled,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=100,
            tol=1e-6,
        )

        means = centers.copy()
        covs = np.zeros((self.n_clusters, n_features, n_features), dtype=float)
        weights = np.zeros(self.n_clusters, dtype=float)

        for k in range(self.n_clusters):
            members = x_filled[labels == k]
            if len(members) == 0:
                weights[k] = 1.0 / self.n_clusters
                covs[k] = np.eye(n_features)
                continue
            weights[k] = len(members) / n_samples
            centered = members - means[k]
            covs[k] = (centered.T @ centered) / max(len(members), 1)
            covs[k] = 0.5 * (covs[k] + covs[k].T) + self.reg_covar * np.eye(n_features)

        weights /= weights.sum()
        return means, covs, weights

    def _log_weighted_density(
        self,
        x: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples, n_features = x.shape
        log_lh = np.zeros((n_samples, self.n_clusters), dtype=float)
        inv_covs = np.zeros_like(covs)

        for k in range(self.n_clusters):
            cov = 0.5 * (covs[k] + covs[k].T) + self.reg_covar * np.eye(n_features)
            covs[k] = cov

            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + 10.0 * self.reg_covar * np.eye(n_features)
                covs[k] = cov
                sign, logdet = np.linalg.slogdet(cov)

            inv_cov = np.linalg.inv(cov)
            inv_covs[k] = inv_cov

            diff = x - means[k]
            maha = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
            log_lh[:, k] = -0.5 * (maha + logdet + n_features * np.log(2.0 * np.pi)) + np.log(
                np.maximum(weights[k], 1e-300)
            )

        return log_lh, inv_covs

    def _e_step(self, log_lh: np.ndarray) -> tuple[float, np.ndarray]:
        maxll = np.max(log_lh, axis=1, keepdims=True)
        post = np.exp(log_lh - maxll)
        density = np.sum(post, axis=1, keepdims=True)
        logpdf = np.log(np.maximum(density, 1e-300)) + maxll
        ll = float(np.sum(logpdf))

        post = post / np.maximum(density, 1e-300)
        post[post < self.posterior_floor] = 0.0
        renorm = np.sum(post, axis=1, keepdims=True)
        renorm[renorm == 0] = 1.0
        post = post / renorm
        return ll, post

    def _m_step(self, x: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = x.shape
        nk = np.sum(gamma, axis=0)
        nk = np.maximum(nk, np.finfo(float).eps)

        means = (gamma.T @ x) / nk[:, None]
        covs = np.zeros((self.n_clusters, n_features, n_features), dtype=float)

        for k in range(self.n_clusters):
            diff = x - means[k]
            weighted = gamma[:, [k]] * diff
            cov = (weighted.T @ diff) / nk[k]
            covs[k] = 0.5 * (cov + cov.T) + self.reg_covar * np.eye(n_features)

        weights = nk / n_samples
        return means, covs, weights

    def _update_missing(
        self,
        x: np.ndarray,
        original_obs_mask: np.ndarray,
        gamma: np.ndarray,
        means: np.ndarray,
        inv_covs: np.ndarray,
    ) -> np.ndarray:
        n_samples, n_features = x.shape
        updated = x.copy()

        for j in range(n_samples):
            miss = np.where(~original_obs_mask[j])[0]
            if miss.size == 0:
                continue
            obs = np.where(original_obs_mask[j])[0]

            x_obs = updated[j, obs] if obs.size else np.empty(0, dtype=float)
            smm_sum = np.zeros((miss.size, miss.size), dtype=float)
            sigu_sum = np.zeros(miss.size, dtype=float)
            smox_sum = np.zeros(miss.size, dtype=float)

            for k in range(self.n_clusters):
                inv_cov = inv_covs[k]
                w = gamma[j, k]
                if w <= 0:
                    continue

                smm = inv_cov[np.ix_(miss, miss)]
                smm_sum += w * smm

                mu_k = means[k]
                mu_m = mu_k[miss]
                if obs.size:
                    mu_o = mu_k[obs]
                    smo = inv_cov[np.ix_(miss, obs)]
                    sigu_sum += w * (smo @ mu_o + smm @ mu_m)
                    smox_sum += w * (smo @ x_obs)
                else:
                    sigu_sum += w * (smm @ mu_m)

            smm_sum = 0.5 * (smm_sum + smm_sum.T) + self.reg_covar * np.eye(miss.size)
            rhs = sigu_sum - smox_sum

            try:
                xm_new = np.linalg.solve(smm_sum, rhs)
            except np.linalg.LinAlgError:
                xm_new = np.linalg.pinv(smm_sum) @ rhs

            xm_old = updated[j, miss]
            updated[j, miss] = self.learning_rate * xm_new + (1.0 - self.learning_rate) * xm_old

            if obs.size:
                updated[j, obs] = x[j, obs]

            if miss.size == n_features:
                updated[j, miss] = np.nan_to_num(updated[j, miss], nan=0.0)

        return updated

    def fit(self, x: np.ndarray | pd.DataFrame) -> "IncompleteGMM":
        arr = self._to_numpy(x)
        if arr.ndim != 2:
            raise ValueError("x must be 2D array-like")

        obs_mask = ~np.isnan(arr)
        x_filled = mean_impute(arr)

        means, covs, weights = self._initialize(x_filled)
        self.log_likelihood_history_ = []

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_lh, inv_covs = self._log_weighted_density(x_filled, means, covs, weights)
            ll, gamma = self._e_step(log_lh)
            self.log_likelihood_history_.append(ll)

            rel_gain = (ll - prev_ll) / max(abs(ll), 1.0)
            if rel_gain < self.tol and np.isfinite(prev_ll):
                break
            prev_ll = ll

            means, covs, weights = self._m_step(x_filled, gamma)

            log_lh2, inv_covs2 = self._log_weighted_density(x_filled, means, covs, weights)
            _, gamma2 = self._e_step(log_lh2)
            x_filled = self._update_missing(x_filled, obs_mask, gamma2, means, inv_covs2)

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.completed_data_ = x_filled
        return self

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        arr = self._to_numpy(x)
        arr = mean_impute(arr)
        log_lh, _ = self._log_weighted_density(arr, self.means_.copy(), self.covariances_.copy(), self.weights_)
        _, gamma = self._e_step(log_lh)
        return gamma

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def fit_predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.fit(x)
        if self.completed_data_ is None:
            raise RuntimeError("Fit failed to produce completed data")
        return np.argmax(self.predict_proba(self.completed_data_), axis=1)

    def fit_with_details(self, x: np.ndarray | pd.DataFrame) -> FitResult:
        self.fit(x)
        if self.completed_data_ is None:
            raise RuntimeError("Fit failed to produce completed data")
        responsibilities = self.predict_proba(self.completed_data_)
        labels = responsibilities.argmax(axis=1)
        return FitResult(
            responsibilities=responsibilities,
            log_likelihood_history=self.log_likelihood_history_.copy(),
            labels=labels,
            completed_data=self.completed_data_.copy(),
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "posterior_floor": self.posterior_floor,
        }
