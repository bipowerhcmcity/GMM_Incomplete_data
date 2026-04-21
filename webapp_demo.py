from __future__ import annotations

import io
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pygmm_incomplete import IncompleteGMM
from pygmm_incomplete.metrics import clustering_report


st = importlib.import_module("streamlit")


st.set_page_config(page_title="GMM Incomplete Data Demo", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def read_csv(uploaded_file: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_file))


def pca2_projection(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x_norm = (x - mu) / std
    _, _, vt = np.linalg.svd(x_norm, full_matrices=False)
    comp = vt[:2].T
    return x_norm @ comp


def to_numeric_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    series = pd.Series(labels).astype(str)
    codes, uniques = pd.factorize(series)
    mapping = {int(i): str(name) for i, name in enumerate(uniques.tolist())}
    return codes.astype(int), mapping


def pca_plot(embedding: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray | None = None) -> plt.Figure:
    if true_labels is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=pred_labels, s=26, cmap="tab10", alpha=0.85)
        ax.set_title("PCA 2D - Predicted clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="Pred cluster")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    sc1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=true_labels, s=26, cmap="tab10", alpha=0.85)
    axes[0].set_title("Ground truth labels")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[0], label="True label")

    sc2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=pred_labels, s=26, cmap="tab10", alpha=0.85)
    axes[1].set_title("Predicted clusters")
    axes[1].set_xlabel("PC1")
    fig.colorbar(sc2, ax=axes[1], label="Pred cluster")

    fig.tight_layout()
    return fig


def main() -> None:
    st.title("📊 GMM Clustering with Incomplete Data - Web Demo")
    st.caption("Upload CSV ➜ Run model ➜ Visualize PCA clustering ➜ Download completed data + predicted label")

    with st.sidebar:
        st.header("Model parameters")
        n_clusters = st.number_input("Number of clusters", min_value=2, max_value=50, value=3, step=1)
        max_iter = st.number_input("Max iterations", min_value=10, max_value=2000, value=150, step=10)
        tol = st.number_input("Tolerance", min_value=1e-8, max_value=1e-1, value=1e-4, format="%.8f")
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)

    uploaded = st.file_uploader("Upload input CSV", type=["csv"])
    if uploaded is None:
        st.info("Hãy upload một file .csv để bắt đầu.")
        st.stop()

    df = read_csv(uploaded.getvalue())
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

    if df.empty:
        st.error("CSV rỗng. Vui lòng upload file có dữ liệu.")
        st.stop()

    all_cols = df.columns.tolist()
    label_col = st.selectbox("Label column (optional, để tính metric)", options=["<none>"] + all_cols, index=0)

    if label_col == "<none>":
        y_true = None
        x_df = df.copy()
    else:
        y_true, label_mapping = to_numeric_labels(df[label_col].to_numpy())
        x_df = df.drop(columns=[label_col])

    if x_df.shape[1] == 0:
        st.error("Không còn cột feature sau khi bỏ label. Cần ít nhất 1 cột feature số.")
        st.stop()

    non_numeric = [c for c in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[c])]
    if non_numeric:
        st.error(f"Các cột feature phải là numeric. Cột không hợp lệ: {non_numeric}")
        st.stop()

    x = x_df.to_numpy(dtype=float)
    missing_ratio = float(np.isnan(x).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", x.shape[0])
    c2.metric("Features", x.shape[1])
    c3.metric("Missing ratio", f"{missing_ratio:.2%}")

    if st.button("🚀 Run GMM", type="primary"):
        with st.spinner("Running IncompleteGMM..."):
            model = IncompleteGMM(
                n_clusters=int(n_clusters),
                max_iter=int(max_iter),
                tol=float(tol),
                random_state=int(seed),
                learning_rate=1.0,
            )
            details = model.fit_with_details(x)

        st.success("Model chạy xong!")

        m1, m2 = st.columns(2)
        m1.metric("Iterations", len(details.log_likelihood_history))
        m2.metric("Final log-likelihood", f"{details.log_likelihood_history[-1]:.4f}")

        if y_true is not None:
            report = clustering_report(y_true, details.labels)
            st.subheader("Clustering metrics")
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("ACC", f"{report['acc']:.4f}")
            mm2.metric("NMI", f"{report['nmi']:.4f}")
            mm3.metric("Purity", f"{report['purity']:.4f}")
            mm4.metric("F1", f"{report['f1']:.4f}")

            st.caption(f"Label mapping (ground truth): {label_mapping}")

        st.subheader("PCA visualize")
        emb = pca2_projection(details.completed_data)
        fig = pca_plot(emb, details.labels, true_labels=y_true)
        st.pyplot(fig, use_container_width=True)

        st.subheader("Kết quả để download")
        result_df = pd.DataFrame(details.completed_data, columns=x_df.columns)
        result_df["predicted_label"] = details.labels
        if label_col != "<none>":
            result_df[label_col] = df[label_col].to_numpy()

        st.dataframe(result_df.head(20), use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download completed data + predicted label",
            data=csv_bytes,
            file_name="gmm_completed_with_predicted_labels.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
