# Usage And API

This page gives the main KNN-first workflows and the public API.

## Which Function Should I Use?

| Situation | Use |
| --- | --- |
| You already computed nearest neighbours | `umap_knn()` or `opentsne_knn()` |
| You want one call from a data matrix | `umap()` or `opentsne()` |
| You want to compare UMAP and openTSNE fairly | compute `knn <- nn(...)` once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set `backend = "cuda"` or a CUDA KNN backend |
| You want a fast approximation for very large data | use `landmark_umap()` or `landmark_tsne()` and report it as landmarking |
| You want quality metrics | `evaluate_embedding(x, layout, labels = labels)` |

The recommended workflow is KNN first:

```r
knn <- nn(x, k = 50, backend = "auto", n_threads = 4)
layout_umap <- umap_knn(knn, seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbour time separate from embedding time and makes
benchmarks easier to interpret.

## Graphs For Clustering

`knn_graph()` dispatches by object type. Pass the KNN result when you want the
graph in the original data space:

```r
knn <- nn(x, k = 50, backend = "auto", n_threads = 4)
g_original <- knn_graph(knn)
```

Pass an embedding fit when you want the graph on the visible UMAP/openTSNE
coordinates:

```r
fit <- opentsne(x, labels = labels, n_neighbors = 50, seed = 1)
g_embedding <- knn_graph(fit, k = 80)
clusters <- igraph::cluster_leiden(
  g_embedding,
  objective_function = "modularity",
  weights = igraph::E(g_embedding)$weight,
  resolution = 0.05
)
```

`knn_graph()` returns a plain `igraph` object. Use your preferred clustering
function on that graph.

For original-data KNN graphs, `weight = "auto"` uses `weight = "snn"`, which
builds a full shared-nearest-neighbour Jaccard graph. This connects all pairs
that share at least one neighbour, following the same standard SNN semantics
used by `bluster::makeSNNGraph()`, but implemented in fastEmbedR's C++ graph
builder so a precomputed `nn()` result can be reused directly.

For embedding-layout graphs, `weight = "auto"` uses distance weights. You can
also test `weight = "adaptive"` for local-density-scaled Gaussian weights and
`mutual = TRUE` to keep only reciprocal neighbour edges.

When `knn_graph()` has to compute neighbours itself, `backend = "auto"` uses
the fastest available graph-KNN backend in this order: CUDA cuVS NN-descent,
FAISS NN-descent, RcppHNSW, then exact CPU. If you pass an `nn()` result,
neighbour search is not repeated.

## Distance Metrics In `nn()`

The default distance is Euclidean:

```r
knn <- nn(x, k = 50, metric = "euclidean", backend = "auto", n_threads = 4)
```

Cosine distance is available through exact CPU KNN:

```r
knn_cosine <- nn(x, k = 50, metric = "cosine", backend = "cpu", n_threads = 4)
layout <- umap_knn(knn_cosine, seed = 1)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | CPU exact/approximate, Metal, CUDA/cuVS, FAISS | Recommended default for large UMAP/openTSNE benchmarks. |
| `cosine` | CPU exact | `backend = "auto"` resolves to CPU. Approximate/GPU/FAISS/cuVS cosine requests error until those paths are validated. |

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- nn(x, k = 31)
layout <- umap_knn(knn)

plot(layout, pch = 21, bg = labels)
```

The one-call interface computes KNN internally:

```r
fit <- umap(
  x,
  labels = labels,
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
knn <- nn(x, k = 50, backend = "metal")
layout <- opentsne_knn(knn, init_data = x, backend = "metal", seed = 1)
```

For CUDA builds with RAPIDS cuVS available:

```r
knn <- nn(x, k = 50, backend = "cuda_cuvs_nndescent")
layout <- opentsne_knn(knn, init_data = x, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Landmark Workflow

```r
fit <- landmark_tsne(
  x,
  labels = labels,
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
  labels = labels,
  landmarks = 0.5,
  n_neighbors = 30,
  backend = "auto",
  seed = 1
)
plot(fit)
```

For landmark runs, `backend = "metal"` uses a fused native Metal projection
kernel that computes query-to-landmark KNN, interpolation, and projection
confidence in one pass before the fixed-reference transform. CPU/auto runs use
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
distance profile in C++. This keeps the public API small while letting
Shuttle-like broad-shell data and high-variability data get less brittle
defaults.

## Public API

| Function | Purpose |
| --- | --- |
| `nn()` | Native exact/approximate KNN for data/query matrices. |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `umap()` | One-call preprocessing, KNN, and UMAP embedding. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `opentsne_knn()` | Direct native openTSNE-style optimizer from KNN. |
| `opentsne()` | One-call preprocessing, KNN, and openTSNE-style embedding. |
| `transform_tsne()` | Fixed-reference openTSNE-style transform for query points. |
| `landmark_tsne()` | Embed landmarks, then transform remaining rows. |
| `landmark_umap()` | Embed landmarks with UMAP, then project/refine remaining rows. |
| `knn_graph()` | Convert `nn()`, `opentsne()`, or `umap()` output into an SNN/distance/binary `igraph` graph. |
| `evaluate_embedding()` | Embedding quality metrics. |
| `backend_info()` | CPU/CUDA/Metal/FAISS/cuVS detection without silent fallback. |
