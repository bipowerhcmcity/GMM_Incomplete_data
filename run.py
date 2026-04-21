from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pygmm_incomplete import IncompleteGMM
from pygmm_incomplete.metrics import clustering_report
from pygmm_incomplete.synthetic import inject_missing_at_random, make_synthetic_gmm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian Mixture Model Clustering with Incomplete Data (from scratch)."
    )
    parser.add_argument("--mode", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--label-col", type=str, default="")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--missing-ratio", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-completed", type=str, default="")
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None]:
    if args.mode == "synthetic":
        x, y = make_synthetic_gmm(
            n_samples=1000,
            n_features=10,
            n_clusters=args.clusters,
            random_state=args.seed,
        )
        x_missing, _ = inject_missing_at_random(x, args.missing_ratio, random_state=args.seed + 1)
        return x_missing, y

    if not args.csv_path:
        raise ValueError("--csv-path is required when mode=csv")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    y: np.ndarray | None = None

    if args.label_col:
        if args.label_col not in df.columns:
            raise ValueError(f"label column '{args.label_col}' not found")
        y = df[args.label_col].to_numpy()
        x_df = df.drop(columns=[args.label_col])
    else:
        x_df = df

    x = x_df.to_numpy(dtype=float)
    return x, y


def main() -> None:
    args = parse_args()
    x, y_true = load_data(args)

    model = IncompleteGMM(
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        learning_rate=1.0,
    )
    details = model.fit_with_details(x)

    print("=== Training Summary ===")
    print(f"Samples: {details.completed_data.shape[0]}")
    print(f"Features: {details.completed_data.shape[1]}")
    print(f"Iterations: {len(details.log_likelihood_history)}")
    print(f"Final log-likelihood: {details.log_likelihood_history[-1]:.6f}")

    if y_true is not None:
        report = clustering_report(y_true, details.labels)
        print("=== Clustering Metrics ===")
        for k, v in report.items():
            print(f"{k}: {v:.6f}")

    if args.save_completed:
        out_path = Path(args.save_completed)
        out_df = pd.DataFrame(details.completed_data)
        out_df.to_csv(out_path, index=False)
        print(f"Saved completed data to: {out_path}")


if __name__ == "__main__":
    main()
