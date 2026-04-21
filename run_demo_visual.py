from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from pygmm_incomplete import IncompleteGMM
from pygmm_incomplete.metrics import clustering_report
from pygmm_incomplete.synthetic import inject_missing_at_random, make_synthetic_gmm


FEATURE_COLUMNS_IRIS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _get_plt():
    return importlib.import_module("matplotlib.pyplot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GMM-with-Incomplete-Data demo with visualization."
    )
    parser.add_argument("--mode", choices=["synthetic", "csv", "kaggle_iris"], default="kaggle_iris")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--label-col", type=str, default="")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--missing-ratio", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="demo_outputs")
    parser.add_argument("--save-completed", type=str, default="")
    parser.add_argument("--save-generated-csv", action="store_true")
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--template-path", type=str, default="data/template_input.csv")
    return parser.parse_args()


def make_kaggle_iris_like(random_state: int = 7, n_per_class: int = 50) -> pd.DataFrame:
    """
    Generate an Iris-style dataset (popular Kaggle schema) without external downloads.
    Columns: sepal_length, sepal_width, petal_length, petal_width, label.
    """
    rng = np.random.default_rng(random_state)

    specs = {
        "setosa": {
            "mean": np.array([5.01, 3.43, 1.46, 0.25]),
            "std": np.array([0.35, 0.38, 0.17, 0.10]),
        },
        "versicolor": {
            "mean": np.array([5.94, 2.77, 4.26, 1.33]),
            "std": np.array([0.52, 0.31, 0.47, 0.20]),
        },
        "virginica": {
            "mean": np.array([6.59, 2.97, 5.55, 2.03]),
            "std": np.array([0.64, 0.32, 0.55, 0.27]),
        },
    }

    frames: list[pd.DataFrame] = []
    for label, cfg in specs.items():
        x = rng.normal(cfg["mean"], cfg["std"], size=(n_per_class, 4))
        x[:, 2:] = np.maximum(x[:, 2:], 0.1)
        df = pd.DataFrame(x, columns=FEATURE_COLUMNS_IRIS)
        df["label"] = label
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def maybe_write_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template_df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.2, 5.9, 6.7, 7.2],
            "sepal_width": [3.5, 3.0, np.nan, 3.0, 3.1, 3.6],
            "petal_length": [1.4, 1.4, 4.5, np.nan, 4.7, 6.1],
            "petal_width": [0.2, 0.2, 1.5, 1.8, np.nan, 2.5],
            "label": ["setosa", "setosa", "versicolor", "virginica", "versicolor", "virginica"],
        }
    )
    template_df.to_csv(path, index=False)


def to_numeric_labels(labels: np.ndarray) -> np.ndarray:
    unique = pd.Index(pd.Series(labels).astype(str).unique())
    mapping = {name: idx for idx, name in enumerate(unique)}
    return np.array([mapping[str(v)] for v in labels], dtype=int)


def load_data(args: argparse.Namespace, output_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if args.mode == "synthetic":
        x, y = make_synthetic_gmm(
            n_samples=1000,
            n_features=10,
            n_clusters=args.clusters,
            random_state=args.seed,
        )
        x_missing, mask = inject_missing_at_random(x, args.missing_ratio, random_state=args.seed + 1)
        return x_missing, y, mask

    if args.mode == "kaggle_iris":
        df = make_kaggle_iris_like(random_state=args.seed, n_per_class=50)
        y = to_numeric_labels(df["label"].to_numpy())
        x = df[FEATURE_COLUMNS_IRIS].to_numpy(dtype=float)
        x_missing, mask = inject_missing_at_random(x, args.missing_ratio, random_state=args.seed + 1)

        if args.save_generated_csv:
            complete_path = output_dir / "kaggle_iris_like_complete.csv"
            missing_path = output_dir / "kaggle_iris_like_missing.csv"
            df_complete = df.copy()
            df_missing = df.copy()
            df_missing[FEATURE_COLUMNS_IRIS] = x_missing
            df_complete.to_csv(complete_path, index=False)
            df_missing.to_csv(missing_path, index=False)
            print(f"Saved generated complete CSV: {complete_path}")
            print(f"Saved generated missing CSV: {missing_path}")

        return x_missing, y, mask

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
        y = to_numeric_labels(df[args.label_col].to_numpy())
        x_df = df.drop(columns=[args.label_col])
    else:
        x_df = df

    x = x_df.to_numpy(dtype=float)
    mask = np.isnan(x)
    return x, y, mask


def pca2_projection(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x_norm = (x - mu) / std
    _, _, vt = np.linalg.svd(x_norm, full_matrices=False)
    comp = vt[:2].T
    return x_norm @ comp


def plot_missing_heatmap(mask: np.ndarray, output_path: Path) -> None:
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(mask, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_title("Missingness Map (black = missing)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Sample index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_log_likelihood(history: list[float], output_path: Path) -> None:
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(history) + 1), history, marker="o", linewidth=1.5)
    ax.set_title("Log-likelihood over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_embedding(
    completed_data: np.ndarray,
    pred_labels: np.ndarray,
    y_true: np.ndarray | None,
    output_path: Path,
) -> None:
    plt = _get_plt()

    emb = pca2_projection(completed_data)

    if y_true is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=pred_labels, s=26, cmap="tab10", alpha=0.85)
        ax.set_title("2D projection colored by predicted clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="Cluster")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        sc1 = axes[0].scatter(emb[:, 0], emb[:, 1], c=y_true, s=26, cmap="tab10", alpha=0.85)
        axes[0].set_title("Ground-truth labels")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        fig.colorbar(sc1, ax=axes[0], label="True label")

        sc2 = axes[1].scatter(emb[:, 0], emb[:, 1], c=pred_labels, s=26, cmap="tab10", alpha=0.85)
        axes[1].set_title("Predicted clusters")
        axes[1].set_xlabel("PC1")
        fig.colorbar(sc2, ax=axes[1], label="Pred cluster")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_template:
        maybe_write_template_csv(Path(args.template_path))
        print(f"Template CSV written to: {Path(args.template_path)}")

    x, y_true, missing_mask = load_data(args, output_dir)

    model = IncompleteGMM(
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        learning_rate=1.0,
    )
    details = model.fit_with_details(x)

    print("=== Training Summary ===")
    print(f"Mode: {args.mode}")
    print(f"Samples: {details.completed_data.shape[0]}")
    print(f"Features: {details.completed_data.shape[1]}")
    print(f"Iterations: {len(details.log_likelihood_history)}")
    print(f"Final log-likelihood: {details.log_likelihood_history[-1]:.6f}")

    if y_true is not None:
        report = clustering_report(y_true, details.labels)
        print("=== Clustering Metrics ===")
        for k, v in report.items():
            print(f"{k}: {v:.6f}")

    completed_path = Path(args.save_completed) if args.save_completed else output_dir / "completed_data.csv"
    pd.DataFrame(details.completed_data).to_csv(completed_path, index=False)
    print(f"Saved completed data to: {completed_path}")

    missing_plot = output_dir / "missingness_map.png"
    ll_plot = output_dir / "log_likelihood_curve.png"
    embed_plot = output_dir / "embedding_clusters.png"

    plot_missing_heatmap(missing_mask, missing_plot)
    plot_log_likelihood(details.log_likelihood_history, ll_plot)
    plot_embedding(details.completed_data, details.labels, y_true, embed_plot)

    print("=== Visualization Artifacts ===")
    print(f"Missingness map: {missing_plot}")
    print(f"Log-likelihood curve: {ll_plot}")
    print(f"2D embedding: {embed_plot}")


if __name__ == "__main__":
    main()
