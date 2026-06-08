# Python UMAP/t-SNE Benchmark Framework

This benchmark suite compares dimensionality-reduction implementations on
public datasets that can be downloaded automatically without account login,
browser interaction, Kaggle credentials, or private access.

## Datasets

| Name | Source | Notes |
| --- | --- | --- |
| `iris` | `sklearn.datasets.load_iris` | Small sanity check |
| `digits` | `sklearn.datasets.load_digits` | Quick handwritten digit test |
| `mnist` | OpenML `mnist_784` | 70,000 samples; use `--subset mnist=10000`, `30000`, or `70000` |
| `fashion_mnist` | Zalando Research GitHub IDX files | 70,000 samples; same subset options |
| `pendigits` | OpenML `pendigits` | UCI PenDigits |
| `shuttle` | OpenML `shuttle` | UCI Shuttle |
| `pbmc3k` | 10x Genomics public PBMC 3k tarball | Sparse matrix normalization, log1p, HVG selection, PCA/SVD |
| `medmnist_pathmnist` | Optional `medmnist` package | Requires `pip install medmnist` |
| `medmnist_organcmnist` | Optional `medmnist` package | Requires `pip install medmnist` |

## Implementations

| Name | Family | Dependency |
| --- | --- | --- |
| `umap_learn` | UMAP | Optional `umap-learn` |
| `sklearn_tsne_bh` | t-SNE | scikit-learn |
| `sklearn_tsne_exact` | t-SNE | scikit-learn, disabled above 5,000 samples |
| `opentsne_fft` | t-SNE | Optional `openTSNE` |
| `pca` | Baseline | scikit-learn |

Missing optional implementations are recorded as `skipped` rows instead of
failing the full benchmark.

## Metrics

Each implementation is measured for:

- Runtime in seconds.
- Peak resident memory in MB.
- Trustworthiness.
- Continuity.
- KNN preservation.
- Silhouette score when labels are available.
- Procrustes stability across repeated runs.

For large datasets, metrics are computed on a reproducible sample controlled by
`--metric-sample`. A shared PCA input reduction is applied by default when
feature dimension is larger than `--pca-input-dims` so image datasets and
t-SNE runs remain practical and comparable.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[all]"
```

List datasets and implementations:

```bash
dimred-bench --list-datasets
dimred-bench --list-implementations
```

Quick local sanity benchmark:

```bash
PYTHONPATH=python python3 scripts/run_python_benchmark.py \
  --datasets iris digits \
  --implementations pca sklearn_tsne_bh \
  --repeats 2 \
  --output-dir benchmark/python_quick
```

Larger public-data benchmark:

```bash
PYTHONPATH=python python3 scripts/run_python_benchmark.py \
  --datasets mnist fashion_mnist pendigits shuttle pbmc3k \
  --subset mnist=10000 \
  --subset fashion_mnist=10000 \
  --subset shuttle=10000 \
  --implementations umap_learn sklearn_tsne_bh opentsne_fft pca \
  --repeats 3 \
  --metric-sample 3000 \
  --pca-input-dims 50 \
  --output-dir benchmark/python_public_10k
```

Full 70k MNIST/Fashion-MNIST runs:

```bash
PYTHONPATH=python python3 scripts/run_python_benchmark.py \
  --datasets mnist fashion_mnist \
  --subset mnist=70000 \
  --subset fashion_mnist=70000 \
  --implementations umap_learn sklearn_tsne_bh opentsne_fft pca \
  --repeats 3 \
  --metric-sample 5000 \
  --pca-input-dims 50 \
  --output-dir benchmark/python_mnist_70k
```

Outputs:

- `manifest.json`: exact benchmark configuration.
- `results.csv`: one row per dataset, implementation, and repeat.
- `summary.csv`: mean/std summary by dataset and implementation.
- `<dataset>_metadata.json`: dataset source and preprocessing details.
- Optional embeddings under `embeddings/` with `--save-embeddings`.

## R Wrappers Later

The Python package keeps dataset loading, implementation runners, metrics, and
orchestration separate. R wrappers can later call the CLI, consume `results.csv`,
or add new implementation runners that shell out to `Rscript` while preserving
the same result schema.
