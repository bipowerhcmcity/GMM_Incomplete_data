# Python GMM Clustering with Incomplete Data


The implementation follows the paper's alternating optimization:
1. E-step: compute posterior responsibilities.
2. M-step: update mixture weights, means, and covariances.
3. Missing-data update: optimize missing entries while keeping observed entries fixed.


## Project structure

- `pygmm_incomplete/core.py` — main algorithm (`IncompleteGMM`)
- `pygmm_incomplete/kmeans.py` — K-Means for initialization
- `pygmm_incomplete/imputers.py` — mean and zero imputation
- `pygmm_incomplete/metrics.py` — ACC, NMI, Purity, F1 (from scratch)
- `pygmm_incomplete/synthetic.py` — synthetic data generation + missingness injection
- `run_demo_visual.py` — demo with visualizations (missing map, convergence curve, 2D projection)

- `benchmark_gmm.py` — gmm benchmark over multiple missing ratios (paper-style)
- `benchmark_kmeans.py` — k-means benchmark over multiple missing ratios (paper-style)

Optional
- `webapp_demo.py` — Streamlit web app (upload CSV, run model, PCA plot, download results)
- `data/template_input.csv` — CSV template for your input data format


## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

#### 1. Run visual demo (Iris-style Kaggle schema, auto-generate plots):

```bash
python run_demo_visual.py --mode kaggle_iris --clusters 3 --missing-ratio 0.25 --save-generated-csv --write-template
```




#### 2. Run benchmark across missing ratios:
##### 2a. GMM

```bash
python benchmark_gmm.py --mode kaggle_iris --clusters 3 --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7 --runs-per-ratio 10 --save-plot --output-dir benchmark_outputs
```
##### 2b. K-means (mean imputation)
```bash
python benchmark_kmeans.py --mode kaggle_iris --clusters 3 --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7 --runs-per-ratio 10 --save-plot --output-dir benchmark_outputs
```

##### custom data
```bash
python benchmark_gmm.py --mode csv --csv-path /path/to/data.csv --label-col label --clusters 4 --ratios 0.1,0.3,0.5 --runs-per-ratio 5 --save-plot
```


#### 3. (Optional) Run web app demo (upload CSV + PCA visualize + download results):

```bash
streamlit run webapp_demo.py
```

Trong web app:
- Upload `.csv` (feature numeric, có thể có `NaN`).
- Chọn label column để tính ACC/NMI/Purity/F1.
- Bấm **Run GMM** để huấn luyện và visualize PCA phân lớp.
- Download file kết quả.
