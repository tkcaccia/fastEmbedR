# Usage And API

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[Reproducibility](reproducibility.md) |
[References](references.md)

This page gives the main KNN-first workflows and the public API.

## Which Function Should I Use?

| Situation | Use |
| --- | --- |
| You want to precompute fastEmbedR's native neighbors | `precompute_knn()` |
| You already computed nearest neighbors | `umap_knn()` or `tsne_knn()` |
| You want one call from a data matrix | `umap()` or `tsne()` |
| You want reusable PCA scores or t-SNE initialization | `pca()` |
| You want to compare UMAP and t-SNE fairly | compute one host KNN list once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set embedding `backend = "cuda"` |
| You want a fast approximation for very large data | set `landmarks` in `umap()` or `tsne()` and report it as landmarking |
| You want quality metrics | `evaluate_embedding(x, layout)` |
| You want a clustering graph | `knn_graph()` |
| You want native graph communities | `graph_cluster(graph, method = "leiden")` |


The recommended workflow is KNN first:

```r
knn <- precompute_knn(
  x,
  k = 50L,
  metric = "euclidean",
  backend = "cpu",
  n.cores = 4L
)
layout_umap <- umap_knn(knn, seed = 1)
layout_tsne <- tsne_knn(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbor time separate from embedding time and makes
benchmarks easier to interpret.

The one-call functions and `precompute_knn()` intentionally hide the KNN
algorithm choice. Their `backend` accepts only `"cpu"`, `"metal"`, or
`"cuda"`. CPU KNN uses native HNSW; Metal uses native exact/IVF-Flat; CUDA
uses RAPIDS cuVS exact or IVF-Flat search and keeps its output resident on the
device. A CUDA KNN object should therefore be reused with a CUDA embedding
backend. A host KNN result from another tool may still be
supplied as a plain list containing `indices` and `distances`; fastEmbedR never
calls that tool itself.

## Distance Metrics

The default distance is Euclidean:

```r
fit <- umap(x, n_neighbors = 50, metric = "euclidean", n.cores = 4)
```

Cosine distance is available through exact CPU KNN:

```r
fit_cosine <- umap(x, n_neighbors = 50, metric = "cosine", n.cores = 4)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | native CPU/Metal and optional direct CUDA/cuVS | Recommended default for large UMAP/t-SNE benchmarks. |
| `cosine` | native CPU/Metal and compiled CUDA | Rows are normalized internally. |
| `correlation` | native CPU/Metal and compiled CUDA | Rows are centered and normalized internally. |

## Parameter Philosophy And Scope

fastEmbedR is deliberately more configurable for t-SNE than for UMAP. The
t-SNE API exposes perplexity and affinity support, initialization, iteration
counts, early and normal exaggeration, learning rate, momentum, clipping, and
exact-versus-FFT repulsion. Set `auto_config = FALSE` and supply explicit
values when an automatic iteration or stopping rule is not wanted.

UMAP is an opinionated high-throughput implementation, not a general-purpose
UMAP tuning interface. It exposes
`n_neighbors`, metric, graph mode, preprocessing, backend, seed, and CPU thread
count. The current release selects and records the remaining optimizer policy:

| UMAP setting | Public status | Current policy |
| --- | --- | --- |
| epochs | internal | 500 below 10,000 rows; 200 for larger ordinary profiles; at least 300 for a high-variability distance profile |
| `min_dist` | internal | 0.01 normally; 0.1 for the documented wide-shell profile |
| spread | internal | 1 |
| learning rate | internal | 1 normally; 1.25 for the wide-shell profile |
| repulsion strength | internal | 1 |
| negative-sample rate | internal | 5 |
| initialization | reusable | package-native graph initialization; use `umap_init()` to compute and reuse it |
| update mode | internal | backend-native asynchronous/atomic schedule; no synchronous GPU mode |

These settings materially affect compactness, optimization effort, and the
attraction/repulsion balance. They are fixed to keep one release policy and
one validated update schedule aligned across CPU, Metal, and CUDA; they are not
claimed to be universally optimal for every scientific question.

The one-call nearest-neighbor router also hides index parameters. It targets
recall 0.99 and chooses the validated CPU, Metal, or CUDA route. To control the
search independently, provide an external KNN object to `umap_knn()` or
`tsne_knn()`.

Resolved choices are not hidden from results:

```r
fit$parameters
fit$timings
```

Float32 input, precomputed or GPU-resident KNN, prepared UMAP state, compact
t-SNE support, and landmarking are the explicit memory/speed choices. Compact
support and landmarking are approximations and should be reported as such.
Users needing arbitrary UMAP `min_dist`, spread, epoch, learning-rate,
negative-sampling, repulsion, or optimizer sweeps
should use a general-purpose UMAP implementation; fastEmbedR does not claim
parameter or API interchangeability with those packages.

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

fit <- umap(x, n_neighbors = 30, n.cores = 4)
layout <- fit$layout

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

## t-SNE From The Same KNN

The KNN must contain at least `ceiling(perplexity)` non-self columns. t-SNE
uses exactly that compact support width and ignores additional KNN columns.

```r
pca_fit <- pca(x, ncomp = 2, seed = 1, tsne_init = TRUE)
Y_init <- pca_fit$tsne_init
layout_tsne <- tsne_knn(
  knn,
  Y_init = Y_init,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250
)

plot(layout_tsne, pch = 21, bg = labels)
```

`Y_init` can be computed once and reused across runs. `init_data` is still
available as a convenience; it is used only to compute PCA initialization for
KNN-input runs and is not used for neighbor search or optimization.

## PCA API

`fastEmbedR::pca()` exposes the backend-native truncated PCA used internally
for t-SNE initialization:

```r
pca_fit <- pca(
  x,
  ncomp = 2,
  backend = "cpu",
  n.cores = 4,
  seed = 1,
  tsne_init = TRUE
)
Y_init <- pca_fit$tsne_init
layout <- tsne_knn(knn, Y_init = Y_init, perplexity = 30)
```

The public `pca()` helper is intentionally simple: there is no `irlba` or
ARPACK method menu and no Python bridge. For t-SNE initialization, CUDA uses
package-native rSVD and may select RAPIDS RAFT TSVD when that optional route is
enabled. It fails loudly if the requested CUDA backend is unavailable.
Float32 CUDA input is passed to the native fit without materializing an R
double matrix, and float32 scores/loadings are returned. Metal uses a native
float32 block-subspace rSVD with MPS matrix products and a resident workspace.
CPU uses fastEmbedR's native
float32 blocked rSVD, with a seeded Gaussian sketch, quality-preserving
oversampling, and one or two subspace iterations according to requested rank.

For `backend = "cpu"`, `n.cores` is a positive integer that temporarily
sets the BLAS/OpenMP thread limit. The result records
`n.cores_requested`, `n.cores_effective`, and `core_control`.
Single-threaded BLAS builds cannot use additional cores and report one
effective thread. Metal and CUDA ignore this CPU-only argument.

The ordinary PCA scores are always retained in `pca_fit$scores`.
`tsne_init = TRUE` adds a second matrix, centered and scaled so that its
largest component standard deviation is `1e-4`; no second decomposition is
performed.

Supplying `xtest` adds projected test coordinates in
`pca_fit$scores_test`. fastEmbedR intentionally omits an SVD method selector.

`irlba` is not imported, suggested, linked, or called by fastEmbedR. It is
used only by external benchmark scripts as a CPU reference.

## Explicit GPU Use

GPU use is explicit. A request for Metal or CUDA must run that backend or fail
clearly.

```r
fit <- tsne(x, perplexity = 30, backend = "metal", seed = 1)
layout <- fit$layout
```

For CUDA builds with RAPIDS cuVS available:

```r
fit <- tsne(x, perplexity = 50, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Graph Clustering

Build a graph from raw data, an embedding result, or a reusable KNN object:

```r
graph <- knn_graph(x, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
table(communities$membership)
```

For an embedding-space graph, pass the fit directly:

```r
fit <- tsne(x, perplexity = 15, backend = "cpu", seed = 1)
graph <- knn_graph(fit, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
```

`knn_graph()` uses the selected backend only when it must compute neighbors.
`graph_cluster()` runs native CPU, CUDA, or Metal Louvain/Leiden without
calling igraph, cuGraph, Python, or another clustering routine. Walktrap is
CPU-only. An unavailable or unsupported GPU backend fails explicitly.
Walktrap is intended for small and moderate graphs because its transition
matrix is quadratic; use Leiden or Louvain for large graphs.

## Landmark Workflow

Set `landmarks` in `umap()` or `tsne()` for explicit landmark
approximations. Both functions select a reference subset, fit the ordinary
embedding implementation on that subset, and project the remaining rows.

`precompute_query_knn(reference, query, ...)` exposes the query-only search
used by the projection stage. It searches the fixed reference only and avoids
constructing unnecessary query-to-query neighbors.

The integrated t-SNE call accepts the landmark selection directly:

```r
fit <- tsne(
  x,
  landmarks = 0.5,
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
fit <- umap(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  graph_mode = "fuzzy",
  backend = "cpu",
  seed = 1
)
plot(fit)
```

Both integrated landmark calls return a reusable fixed-reference model in
`fit$model`. New observations are supplied in the original feature space; the
stored standardization and PCA transforms are reused before projection:

```r
new_layout <- project_landmark_model(
  fit$model,
  newdata,
  transform_k = 15
)$layout
```

To reconstruct the original training rows, pass the complementary rows as
`query_indices = fit$model$selection$query_indices`. With no indices, every
supplied row is treated as a genuinely new observation.

CPU projection uses native fixed-parameter HNSW and reports that recall was
not audited at runtime. Metal uses native exact search
for small references and recall-tuned IVF-Flat for larger references, followed
by native fixed-reference transform kernels. CUDA uses native exact search for
smaller references and IVF-Flat for larger references; its KNN result remains
device resident for the projection and refinement stages.

## Automatic Parameters

`tsne()` and `tsne_knn()` use `auto_config = TRUE` by default. Missing
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
supplied neighbor graph.

## Public API

| Function | Purpose |
| --- | --- |
| `precompute_knn()` | Package-native non-self KNN search with CPU, Metal, or CUDA backend. |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `umap()` | One-call preprocessing, KNN, and UMAP embedding. |
| `pca()` | Backend-native truncated PCA scores/loadings. |
| `tsne_knn()` | Direct native interpolation-based t-SNE optimizer from KNN. |
| `tsne()` | One-call preprocessing, KNN, and interpolation-based t-SNE embedding. |
| `transform_tsne()` | Fixed-reference interpolation-based t-SNE transform for query points. |
| `tsne(..., landmarks = ...)` | Embed landmarks, then transform remaining rows. |
| `umap(..., landmarks = ...)` | Embed landmarks, then project/refine remaining rows. |
| `evaluate_embedding()` | Embedding quality metrics. |
| `knn_graph()` | Compact graph from data, an embedding, or supplied KNN. |
| `graph_cluster()` | Native Louvain, Leiden, or Pons-Latapy Walktrap communities. |
