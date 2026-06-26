# Usage And API

This page gives the main KNN-first workflows and the public API.

## Which Function Should I Use?

| Situation | Use |
| --- | --- |
| You already computed nearest neighbours | `umap_knn()` or `opentsne_knn()` |
| You want one call from a data matrix | `umap()` or `opentsne()` |
| You want to compare UMAP and openTSNE fairly | compute `knn <- faissR::nn(...)` once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set embedding `backend = "cuda"` |
| You want a fast approximation for very large data | use `landmark_umap()` or `landmark_tsne()` and report it as landmarking |
| You want quality metrics | `evaluate_embedding(x, layout)` |

The recommended workflow is KNN first:

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, backend = "auto", n_threads = 4)
layout_umap <- umap_knn(knn, seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbour time separate from embedding time and makes
benchmarks easier to interpret.

The one-call functions intentionally hide the KNN algorithm choice. For
`opentsne()` and `umap()`, `backend` accepts only `"cpu"`, `"metal"`, or
`"cuda"`. Matrix-input KNN is delegated to faissR through fastEmbedR's internal bridge:
CPU and Metal use faissR CPU HNSW with `target_recall = 0.99`, while CUDA uses
faissR's CUDA policy. To benchmark another KNN algorithm, compute it explicitly with
`faissR::nn()` and pass the result to `opentsne_knn()` or `umap_knn()`.

## Distance Metrics In `faissR::nn()`

The default distance is Euclidean:

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, metric = "euclidean", backend = "auto", n_threads = 4)
```

Cosine distance is available through exact CPU KNN:

```r
knn_cosine <- faissR::nn(x, k = 50, exclude_self = TRUE, metric = "cosine", backend = "cpu", n_threads = 4)
layout <- umap_knn(knn_cosine, seed = 1)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | FAISS CPU and optional CUDA/cuVS through `faissR` | Recommended default for large UMAP/openTSNE benchmarks. |
| `cosine` / inner product | FAISS/candidate paths where enabled by `faissR` | Use normalized rows when treating inner product as cosine similarity. |

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- faissR::nn(x, k = 31, exclude_self = TRUE)
layout <- umap_knn(knn)

plot(layout, pch = 21, bg = labels)
```

The one-call interface computes KNN internally:

```r
fit <- umap(
  x,
  n_neighbors = 30,
  seed = 1
)
plot(fit)
```

## openTSNE From The Same KNN

```r
layout_tsne <- opentsne_knn(
  knn,
  init_data = x,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250
)

plot(layout_tsne, pch = 21, bg = labels)
```

`init_data` is used only to compute PCA initialization for KNN-input runs. It
is not used for neighbour search or optimization.

## Explicit GPU Use

GPU use is explicit. A request for Metal or CUDA must run that backend or fail
clearly.

```r
knn <- faissR::nn(x, k = 50, exclude_self = TRUE, backend = "metal")
layout <- opentsne_knn(knn, init_data = x, backend = "metal", seed = 1)
```

For CUDA builds with RAPIDS cuVS available:

```r
fit <- opentsne(x, perplexity = 50, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Landmark Workflow

```r
fit <- landmark_tsne(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250,
  transform_iter = 100,
  seed = 1
)
plot(fit)
```

UMAP has the same landmark pattern:

```r
fit <- landmark_umap(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  backend = "cpu",
  seed = 1
)
plot(fit)
```

For landmark runs, `backend = "metal"` uses a fused native Metal projection
kernel that computes query-to-landmark KNN, interpolation, and projection
confidence in one pass before the fixed-reference transform. CPU runs use
exact multi-threaded projection KNN by default and switch to a native
projection-specific approximation only for large projections where the cheaper
candidate search is worthwhile.

## Automatic Parameters

`opentsne()` and `opentsne_knn()` use `auto_config = TRUE` by default. Missing
t-SNE settings are resolved in native C++ using the opt-SNE strategy:

- `"auto"` learning rate becomes `n / early_exaggeration`.
- Early exaggeration can stop at the local maximum of KLD relative change.
- The normal phase can stop when KLD improvement drops below the opt-SNE
  threshold.

The KLD monitor is enabled only where it is computationally honest: CPU/small
exact runs. Large FFT and GPU runs keep opt-SNE's learning-rate/default-limit
policy but do not perform a hidden CPU O(n^2) KLD poll or report it as GPU
work.

`umap()` and `umap_knn()` also choose internal defaults from the supplied KNN
distance profile in C++. This keeps the public API small while preserving the
supplied neighbour graph.

## Public API

| Function | Purpose |
| --- | --- |
| `faissR::nn()` | Companion-package KNN function for data/query matrices. |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `umap()` | One-call preprocessing, KNN, and UMAP embedding. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `opentsne_knn()` | Direct native openTSNE-style optimizer from KNN. |
| `opentsne()` | One-call preprocessing, KNN, and openTSNE-style embedding. |
| `transform_tsne()` | Fixed-reference openTSNE-style transform for query points. |
| `landmark_tsne()` | Embed landmarks, then transform remaining rows. |
| `landmark_umap()` | Embed landmarks with UMAP, then project/refine remaining rows. |
| `evaluate_embedding()` | Embedding quality metrics. |
| `faissR::backend_info()` | FAISS/cuVS KNN backend detection without silent fallback. |
