from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pygmm_incomplete.metrics import clustering_report
from pygmm_incomplete.synthetic import inject_missing_at_random, make_synthetic_gmm
from pygmm_incomplete.kmeans import run_kmeans
from pygmm_incomplete.imputers import mean_impute

FEATURE_COLUMNS_IRIS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _get_plt():
    return importlib.import_module("matplotlib.pyplot")


def parse_ratios(text: str) -> list[float]:
    ratios = [float(x.strip()) for x in text.split(",") if x.strip()]
    for r in ratios:
        if not 0.0 <= r < 1.0:
            raise ValueError(f"Invalid ratio {r}.")
    return ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark K-Means baseline across multiple missing-data ratios."
    )
    parser.add_argument("--mode", choices=["synthetic",'seeds', "kaggle_iris", "wine", "csv"], default="kaggle_iris")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--label-col", type=str, default="")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--runs-per-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--save-plot", action="store_true")
    return parser.parse_args()


def to_numeric_labels(labels: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(labels).astype(str))
    return codes.astype(int)


def make_kaggle_iris_like(random_state: int = 7, n_per_class: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    specs = {
        "setosa":     {"mean": np.array([5.01, 3.43, 1.46, 0.25]), "std": np.array([0.35, 0.38, 0.17, 0.10])},
        "versicolor": {"mean": np.array([5.94, 2.77, 4.26, 1.33]), "std": np.array([0.52, 0.31, 0.47, 0.20])},
        "virginica":  {"mean": np.array([6.59, 2.97, 5.55, 2.03]), "std": np.array([0.64, 0.32, 0.55, 0.27])},
    }
    frames = []
    for label, cfg in specs.items():
        x = rng.normal(cfg["mean"], cfg["std"], size=(n_per_class, 4))
        x[:, 2:] = np.maximum(x[:, 2:], 0.1)
        df = pd.DataFrame(x, columns=FEATURE_COLUMNS_IRIS)
        df["label"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_complete_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, str]:
    if args.mode == "synthetic":
        x, y = make_synthetic_gmm(n_samples=1000, n_features=10, n_clusters=args.clusters, random_state=args.seed)
        return x, y, "synthetic"

    if args.mode == "kaggle_iris":
        df = make_kaggle_iris_like(random_state=args.seed)
        x = df[FEATURE_COLUMNS_IRIS].to_numpy(dtype=float)
        y = to_numeric_labels(df["label"].to_numpy())
        return x, y, "kaggle_iris_like"
    elif args.mode == "wine":
        # from sklearn.datasets import load_wine
        
        data = load_wine()
        x = data.data
        y = data.target
        
        return x, y, "wine"
    elif args.mode == "seeds":
        import pandas as pd
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        df = pd.read_csv(url, sep='\t+', header=None, engine='python')
        x = df.iloc[:, :7].to_numpy(dtype=float)
        y = df.iloc[:, 7].to_numpy().astype(int) - 1
        print(x.shape)
        return x, y, "seeds"
        


    if not args.csv_path:
        raise ValueError("--csv-path is required when mode=csv")
    df = pd.read_csv(args.csv_path)
    y = None
    if args.label_col:
        y = to_numeric_labels(df[args.label_col].to_numpy())
        df = df.drop(columns=[args.label_col])
    x = df.to_numpy(dtype=float)
    if np.isnan(x).any():
        col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        idx = np.where(np.isnan(x))
        x[idx] = col_means[idx[1]]
    dataset_name = args.dataset_name or Path(args.csv_path).stem
    return x, y, dataset_name


def run_single_experiment(
    x_complete: np.ndarray,
    y_true: np.ndarray | None,
    ratio: float,
    run_id: int,
    args: argparse.Namespace,
) -> dict:
    ratio_seed = args.seed + int(ratio * 1000) * 100 + run_id
    x_missing, missing_mask = inject_missing_at_random(x_complete, ratio, random_state=ratio_seed)

    x_filled = mean_impute(x_missing)
    labels, _ = run_kmeans(x_filled, n_clusters=args.clusters, random_state=ratio_seed)

    row: dict = {
        "missing_ratio": ratio,
        "run": run_id,
        "missing_fraction_realized": float(missing_mask.mean()),
    }

    if y_true is not None:
        report = clustering_report(y_true, labels)
        row.update({
            "acc":    float(report["acc"]),
            "nmi":    float(report["nmi"]),
            "purity": float(report["purity"]),
            "f1":     float(report["f1"]),
        })

    return row


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in raw_df.columns if c not in {"run", "missing_ratio"} and pd.api.types.is_numeric_dtype(raw_df[c])]
    agg_spec = {c: ["mean", "std"] for c in numeric_cols}
    grouped = raw_df.groupby("missing_ratio", as_index=False).agg(agg_spec)
    grouped.columns = [
        "missing_ratio" if c1 == "missing_ratio" else f"{c1}_{c2}"
        for c1, c2 in grouped.columns.to_flat_index()
    ]
    return grouped.sort_values("missing_ratio").reset_index(drop=True)


def plot_benchmark(summary_df: pd.DataFrame, output_path: Path, has_labels: bool) -> None:
    plt = _get_plt()
    x = summary_df["missing_ratio"].to_numpy()

    if has_labels:
        metric_names = ["acc", "nmi", "purity", "f1"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        axes = axes.ravel()
        for ax, metric in zip(axes, metric_names):
            y = summary_df[f"{metric}_mean"].to_numpy()
            yerr = np.nan_to_num(summary_df[f"{metric}_std"].to_numpy(), nan=0.0)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
            ax.set_title(metric.upper())
            ax.grid(alpha=0.25)
            ax.set_ylim(0.0, 1.02)
        for ax in axes:
            ax.set_xlabel("Missing ratio")
        fig.suptitle("Benchmark across missing ratios", y=1.02)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        ll = summary_df["missing_fraction_realized_mean"].to_numpy()
        ll_err = np.nan_to_num(summary_df["missing_fraction_realized_std"].to_numpy(), nan=0.0)

        axes[0].errorbar(x, ll, yerr=ll_err, marker="o", capsize=3)
        axes[0].set_title("Missing fraction realized")
        axes[0].set_xlabel("Missing ratio")
        axes[0].grid(alpha=0.25)

        axes[1].set_visible(False)
        fig.suptitle("Benchmark across missing ratios", y=1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_complete, y_true, dataset_name = load_complete_data(args)

    rows = []
    for ratio in ratios:
        for run_id in range(args.runs_per_ratio):
            row = run_single_experiment(x_complete, y_true, ratio, run_id, args)
            rows.append(row)

    raw_df = pd.DataFrame(rows).sort_values(["missing_ratio", "run"]).reset_index(drop=True)
    summary_df = summarize_results(raw_df)

    raw_path     = output_dir / "kmeans_benchmark_raw.csv"
    summary_path = output_dir / "kmeans_benchmark_summary.csv"
    config_path  = output_dir / "kmeans_benchmark_config.json"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    config = {
        "model": "kmeans_baseline",
        "dataset": dataset_name,
        "ratios": ratios,
        "runs_per_ratio": args.runs_per_ratio,
        "clusters": args.clusters,
        "seed": args.seed,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=== Benchmark Done ===")
    print(f"Dataset: {dataset_name}")
    print(f"Ratios: {ratios}")
    print(f"Runs per ratio: {args.runs_per_ratio}")
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary results: {summary_path}")
    print(f"Saved config: {config_path}")

    if args.save_plot:
        plot_path = output_dir / "kmeans_benchmark_plot.png"
        plot_benchmark(summary_df, plot_path, has_labels=y_true is not None)
        print(f"Saved plot: {plot_path}")

    print("\nTop rows of summary:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(summary_df.head())


if __name__ == "__main__":
    main()
