# fastEmbedR

`fastEmbedR` is now the embedding package in a two-package workflow:

- `faissR` provides nearest-neighbour search, candidate KNN, kNN graph building,
  kNN classifier/regressor helpers, and k-means.
- `fastEmbedR` provides UMAP and native openTSNE-style embeddings from
  precomputed neighbours, plus embedding metrics and landmark transforms.

The intended workflow is:

1. compute nearest neighbours once with `fastEmbedR::nn()` or `faissR::nn()`;
2. reuse the same KNN object for `fastEmbedR::umap_knn()` or
   `fastEmbedR::opentsne_knn()`;
3. optionally build clustering graphs with `fastEmbedR::knn_graph()` or
   `faissR::knn_graph()`;
4. keep CPU/GPU backend reporting explicit.

The public API is deliberately small and avoids silent CPU fallback when an
explicit GPU backend is requested.

## Installation

For the development version:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
remotes::install_github("tkcaccia/fastEmbedR")
```

The CRAN-oriented build does not vendor large optional GPU libraries. FAISS and
RAPIDS cuVS support live in `faissR` and are enabled only when `faissR` is built
on a machine where the corresponding C++ libraries are available.

## Main Functions

| Function | Purpose |
| --- | --- |
| `umap_knn()` | UMAP from a supplied KNN object. |
| `umap()` | One-call `faissR::nn()` plus UMAP. |
| `opentsne_knn()` | Native openTSNE-style t-SNE from a supplied KNN object. |
| `opentsne()` | One-call `faissR::nn()` plus openTSNE-style t-SNE. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `evaluate_embedding()` | Trustworthiness, neighbour preservation, label accuracy, and related diagnostics. |
| `backend_info()` | Report `fastEmbedR` embedding backends and `faissR` KNN backends. |

Neighbour-search functions are implemented in `faissR` and re-exported by
`fastEmbedR` as thin wrappers:

| Function | Purpose |
| --- | --- |
| `fastEmbedR::nn()` | Nearest-neighbour search through `faissR`, using mandatory FAISS support and optional CUDA/cuVS acceleration when available. |
| `fastEmbedR::candidate_knn()` | Exact top-k ranking within supplied candidate rows. |
| `fastEmbedR::knn_graph()` | Convert `nn()`, `opentsne()`, or `umap()` output into an `igraph` graph. |
| `fastEmbedR::fast_kmeans()` | CPU/FAISS/cuVS k-means helper. |
| `fastEmbedR::knn_fit()` | kNN classifier/regressor helper. |

## Implementation Notes And Acknowledgements

The detailed implementation story is split into GitHub documentation pages:

- [Implementation and library inventory](docs/implementation-inventory.md)
- [Algorithm design, improvements, and acknowledgements](docs/algorithm-provenance.md)
- [Function implementation details and literature](docs/function-implementation-details.md)

In short, `fastEmbedR` implements UMAP and openTSNE-style t-SNE in native
R/C++ with optional Objective-C++/Metal and CUDA kernels. It does not call
Python for public UMAP or openTSNE functions. The nearest-neighbour layer is
delegated to the companion `faissR` package, which owns FAISS/cuVS KNN,
candidate KNN, graph building, kNN prediction, and k-means. The projects and
papers that shaped the implementation are acknowledged in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.

## KNN-First Example

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- fastEmbedR::nn(x, k = 15, backend = "auto", n_threads = 2)
layout_umap <- umap_knn(knn, backend = "cpu", seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, backend = "cpu", seed = 1)

plot(layout_umap, pch = 21, bg = labels)
plot(layout_tsne, pch = 21, bg = labels)
```

## Graph Clustering

`fastEmbedR::knn_graph()` returns an `igraph` graph. The object you pass decides
the graph space:

- `fastEmbedR::knn_graph(knn)` builds a graph on the original data neighbours
  from `fastEmbedR::nn()`.
- `fastEmbedR::knn_graph(fit)` builds a graph on the visible `opentsne()` /
  `umap()` embedding coordinates.

The default `weight = "auto"` uses shared-nearest-neighbour Jaccard weights for
original-data KNN graphs and distance weights for embedding-layout graphs.
The SNN path builds the full graph between observations that share neighbours,
using a sparse co-occurrence strategy inspired by `bluster::makeSNNGraph()`.
When a graph needs to compute neighbours from a matrix or embedding layout,
`backend = "auto"` delegates to `faissR`; accelerated builds choose CUDA/cuVS
or FAISS paths when available, and otherwise fail or use the configured CPU
FAISS path according to `faissR` backend rules.
`weight = "adaptive"` and `mutual = TRUE` are available when you want to test
stricter local-density weighting or reciprocal-neighbour boundaries.

```r
if (requireNamespace("igraph", quietly = TRUE)) {
  graph <- fastEmbedR::knn_graph(knn, k = 15)
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
  graph <- fastEmbedR::knn_graph(fit, k = 80)
  clusters <- igraph::cluster_leiden(
    graph,
    objective_function = "modularity",
    weights = igraph::E(graph)$weight,
    resolution = 0.05
  )
}
```

## Distance Metrics

`fastEmbedR::nn()` delegates to `faissR::nn()` and intentionally exposes a small
set of common distances:

| metric | supported backends | use when |
| --- | --- | --- |
| `euclidean` | FAISS CPU and optional CUDA/cuVS paths through `faissR` | Default for UMAP/openTSNE. |
| `cosine` / inner product | FAISS/candidate paths where enabled by `faissR` | Normalized features and image/embedding descriptors. |

Unsupported metric/backend combinations fail clearly rather than returning a
mislabeled result.

## GPU Backends

Metal and CUDA support is explicit:

- Metal is used for selected native `fastEmbedR` embedding/projection kernels on
  macOS.
- CUDA embedding support is optional at `fastEmbedR` build time.
- FAISS/cuVS KNN support is optional at `faissR` build time.
- If an explicit GPU backend is unavailable, the function errors instead of
  silently running on CPU and reporting a GPU result.

Use `backend_info()` to inspect what is available on the current machine.

Strict optional builds can be requested with environment variables:

```sh
# FAISS CPU KNN in faissR
FAISS_HOME=/path/to/faiss FAISSR_USE_FAISS=1 R CMD INSTALL /path/to/faissR

# CUDA/cuVS KNN in faissR
CUDA_HOME=/path/to/cuda CUVS_HOME=/path/to/cuvs \
FAISSR_USE_CUDA=1 FAISSR_USE_CUVS=1 R CMD INSTALL /path/to/faissR

# CUDA/Metal embedding kernels in fastEmbedR
CUDA_HOME=/path/to/cuda FASTEMBEDR_USE_CUDA=1 R CMD INSTALL /path/to/fastEmbedR
```

When these variables are set to `1`, installation fails clearly if the
requested headers or libraries cannot be found.

## Scope

Classic `tsne()`, InfoTSNE, PaCMAP, TriMap, LocalMAP, and slow experimental
branches are not part of the public API. `fastEmbedR` focuses on UMAP,
openTSNE-style t-SNE, metrics, and landmark workflows. `faissR` focuses on KNN,
graphs, kNN prediction, and k-means.

## License And Provenance

`fastEmbedR` is licensed as `GPL (>= 3)`. Implementation notes,
acknowledgements, and algorithmic references are included in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.
