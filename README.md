# fastEmbedR

`fastEmbedR` is a KNN-first R package for dimensionality reduction. It provides
native nearest-neighbour search, UMAP from precomputed neighbours, and a native
openTSNE-style optimizer from precomputed neighbours.

The intended workflow is:

1. compute nearest neighbours once with `nn()`;
2. reuse the same KNN object for UMAP, openTSNE, graph clustering, or quality
   checks;
3. keep CPU/GPU backend reporting explicit.

The public API is deliberately small and avoids silent CPU fallback when an
explicit GPU backend is requested.

## Installation

For the development version:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

The CRAN build does not vendor large optional GPU libraries. FAISS and RAPIDS
cuVS support are enabled only when the package is built on a machine where the
corresponding C++ libraries are available.

## Main Functions

| Function | Purpose |
| --- | --- |
| `nn()` | Nearest-neighbour search. Uses CPU exact/RcppHNSW by default; optional FAISS/cuVS paths are used only when built. |
| `umap_knn()` | UMAP from a supplied KNN object. |
| `umap()` | One-call KNN plus UMAP. |
| `opentsne_knn()` | Native openTSNE-style t-SNE from a supplied KNN object. |
| `opentsne()` | One-call KNN plus openTSNE-style t-SNE. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `knn_graph()` | Convert `nn()`, `opentsne()`, or `umap()` output into an `igraph` graph. |
| `evaluate_embedding()` | Trustworthiness, neighbour preservation, label accuracy, and related diagnostics. |
| `backend_info()` | Report CPU, Metal, CUDA/cuVS, and FAISS availability. |

## KNN-First Example

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- nn(x, k = 15, backend = "auto", n_threads = 2)
layout_umap <- umap_knn(knn, backend = "cpu", seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, backend = "cpu", seed = 1)

plot(layout_umap, pch = 21, bg = labels)
plot(layout_tsne, pch = 21, bg = labels)
```

## Graph Clustering

`knn_graph()` returns an `igraph` graph. The object you pass decides the graph
space:

- `knn_graph(knn)` builds a graph on the original data neighbours from `nn()`.
- `knn_graph(fit)` builds a graph on the visible `opentsne()` / `umap()`
  embedding coordinates.

The default `weight = "auto"` uses shared-nearest-neighbour Jaccard weights for
original-data KNN graphs and distance weights for embedding-layout graphs.
The SNN path builds the full graph between observations that share neighbours,
using a sparse co-occurrence strategy inspired by `bluster::makeSNNGraph()`.
When a graph needs to compute neighbours from a matrix or embedding layout,
`backend = "auto"` chooses cuVS NN-descent when available, then FAISS
NN-descent, then RcppHNSW, then exact CPU.
`weight = "adaptive"` and `mutual = TRUE` are available when you want to test
stricter local-density weighting or reciprocal-neighbour boundaries.

```r
if (requireNamespace("igraph", quietly = TRUE)) {
  graph <- knn_graph(knn, k = 15)
  clusters <- igraph::cluster_louvain(graph, weights = igraph::E(graph)$weight)
  plot(layout_tsne, pch = 21, bg = igraph::membership(clusters))
}
```

Leiden clustering is available when the optional `leidenbase` package is
installed:

```r
if (requireNamespace("igraph", quietly = TRUE) &&
    "cluster_leiden" %in% getNamespaceExports("igraph")) {
  fit <- opentsne(x, labels = labels, n_neighbors = 30, seed = 1)
  graph <- knn_graph(fit, k = 80)
  clusters <- igraph::cluster_leiden(
    graph,
    objective_function = "modularity",
    weights = igraph::E(graph)$weight,
    resolution = 0.05
  )
}
```

## Distance Metrics

`nn()` intentionally exposes a small set of common distances:

| metric | supported backends | use when |
| --- | --- | --- |
| `euclidean` | CPU exact, RcppHNSW, optional FAISS/cuVS | Default for UMAP/openTSNE. |
| `cosine` | CPU exact, RcppHNSW | Angular similarity and normalized features. |
| `correlation` | CPU exact, RcppHNSW | Expression-profile comparisons. |

Explicit FAISS/cuVS requests currently require Euclidean distance. Unsupported
metric/backend combinations fail clearly rather than returning a mislabeled
Euclidean result.

## GPU Backends

Metal and CUDA support is explicit:

- Metal is used for selected native embedding/projection kernels on macOS.
- CUDA/cuVS support is optional at build time.
- If an explicit GPU backend is unavailable, the function errors instead of
  silently running on CPU and reporting a GPU result.

Use `backend_info()` to inspect what is available on the current machine.

Strict optional builds can be requested with environment variables:

```sh
# FAISS CPU KNN
FAISS_HOME=/path/to/faiss FASTEMBEDR_USE_FAISS=1 R CMD INSTALL .

# CUDA kernels plus RAPIDS cuVS KNN
CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1 R CMD INSTALL .

# FAISS and CUDA/cuVS together
FAISS_HOME=/path/to/faiss CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FASTEMBEDR_USE_FAISS=1 FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1 \
R CMD INSTALL .
```

When these variables are set to `1`, installation fails clearly if the
requested headers or libraries cannot be found.

## Scope

Classic `tsne()`, InfoTSNE, PaCMAP, TriMap, LocalMAP, and slow experimental
branches are not part of the public API. The package focuses on reusable KNN,
UMAP, openTSNE-style t-SNE, graph clustering, metrics, and landmark workflows.

## License And Provenance

`fastEmbedR` is licensed as `GPL (>= 3)`. Implementation notes,
acknowledgements, and algorithmic references are included in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.
