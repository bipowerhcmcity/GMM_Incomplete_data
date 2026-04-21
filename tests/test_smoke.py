from __future__ import annotations

import unittest

import numpy as np

from pygmm_incomplete import IncompleteGMM
from pygmm_incomplete.metrics import clustering_report
from pygmm_incomplete.synthetic import inject_missing_at_random, make_synthetic_gmm


class TestIncompleteGMM(unittest.TestCase):
    def test_smoke_synthetic(self) -> None:
        x, y = make_synthetic_gmm(
            n_samples=500,
            n_features=8,
            n_clusters=3,
            random_state=11,
        )
        x_missing, _ = inject_missing_at_random(x, missing_ratio=0.35, random_state=12)

        model = IncompleteGMM(n_clusters=3, max_iter=120, random_state=11)
        details = model.fit_with_details(x_missing)

        self.assertFalse(np.isnan(details.completed_data).any())
        self.assertGreaterEqual(len(details.log_likelihood_history), 2)

        report = clustering_report(y, details.labels)
        self.assertGreater(report["acc"], 0.35)


if __name__ == "__main__":
    unittest.main()
