# fastEmbedR

**Home** |
[Installation](docs/installation.md) |
[Implementation](docs/implementation.md) |
[Performance Engineering](docs/backend-performance-engineering.md) |
[Examples](docs/examples.md) |
[Benchmarks](docs/benchmarks.md) |
[API](docs/usage-api.md) |
[API Map](docs/api-map.md) |
[Reproducibility](docs/reproducibility.md) |
[Development](docs/development.md) |
[References](docs/references.md) |
[Benchmark repository](https://github.com/tkcaccia/fastEmbedR-extra)

`fastEmbedR` is a native R/C++ package for fast dimensionality reduction from
nearest-neighbor graphs. Its UMAP implementation is deliberately opinionated:
it provides one recorded optimizer policy validated across CPU, Metal, and
CUDA, rather than a drop-in interface for arbitrary UMAP parameter sweeps. Its
primary contributions are:

- UMAP from KNN input;
- interpolation-based t-SNE from KNN input;
- native CPU, Apple Metal, and CUDA embedding backends where available;
- float32 input/output support with float32 native optimizer buffers;
- explicit backend reporting, with no silent CPU fallback labelled as GPU;
- native CPU HNSW and Apple Metal exact/IVF-Flat KNN for one-call embeddings;
- optional GPU-resident CUDA KNN through RAPIDS cuVS, with FAISS GPU as an
  explicit optional exact-search provider.

The t-SNE implementation combines sparse perplexity affinities, two-phase
optimization, interpolation/FFT repulsion, and fixed-reference transformation
in native CPU, Metal, and CUDA kernels. The production default uses compact
affinity support with `ceiling(perplexity)` non-self candidate neighbors.

Publication benchmark scripts, dataset manifests, HPC launchers, and data
acquisition instructions are maintained separately in
[`tkcaccia/fastEmbedR-extra`](https://github.com/tkcaccia/fastEmbedR-extra).
The package repository does not distribute benchmark datasets or manuscript
files.

Hardware evidence is reported at three distinct levels: full benchmark
validation, strict real-device smoke/correctness testing, and build-level
architectural compatibility. Metal performance has been benchmarked only on
one Apple M3 system; other Apple Silicon systems are compatibility targets and
do not inherit the M3 performance results. CUDA architecture flags and a
successful build likewise do not constitute runtime or performance evidence
for an untested GPU. See the [hardware evidence contract](docs/backend-validation.md).

The intended workflow is:

1. call `tsne()` or `umap()` and let fastEmbedR select its native KNN path,
   or call `precompute_knn()` explicitly with a CPU, Metal, or CUDA backend;
2. reuse that KNN object in `fastEmbedR::tsne_knn()` or
   `fastEmbedR::umap_knn()`;
3. evaluate or plot the embedding.

For the one-call functions `tsne()` and `umap()`, the embedding backend is
deliberately limited to `backend = "cpu"`, `"metal"`, or `"cuda"`. Internal
CPU one-call embeddings use the package-native float32 HNSW path. Metal uses
native exact search for small inputs and recall-tuned IVF-Flat for larger
inputs. CUDA uses cuVS brute-force exact search below 100,000 rows and cuVS
IVF-Flat above that threshold, then passes package-owned device pointers
into UMAP or t-SNE. It does not call another R package for KNN. No
unavailable GPU backend is silently relabelled as CPU.

## Quick Start

```r
library(fastEmbedR)

iris_data <- datasets::iris
x <- scale(as.matrix(iris_data[, 1:4]))
labels <- iris_data$Species

y_tsne <- fastEmbedR::tsne(
  x,
  perplexity = 10,
  backend = "cpu",
  n.cores = 4,
  seed = 1
)

y_umap <- fastEmbedR::umap(
  x,
  n_neighbors = 15,
  backend = "cpu",
  n.cores = 4,
  seed = 1
)

plot(y_tsne, pch = 21, bg = labels)
plot(y_umap, pch = 21, bg = labels)

# Precompute once and reuse the identical neighbors.
knn <- fastEmbedR::precompute_knn(
  x, k = 15, backend = "cpu", n.cores = 4
)
y_from_knn <- fastEmbedR::umap_knn(knn, backend = "cpu", seed = 1)
```

`umap()` and `umap_knn()` use the standard fuzzy UMAP graph by default. Set
`graph_mode = "binary"` only for the explicit adjacency-only sensitivity mode.
The public UMAP API exposes `n_neighbors`, metric, graph mode, preprocessing,
backend, seed, output dimension, and CPU thread count. Epochs, `min_dist`,
spread, learning rate, repulsion strength, negative-sample rate, KNN index
tuning, and optimizer mode follow the package policy and are recorded in
`fit$parameters`; they are not public sweep arguments. Users whose analysis
depends on varying those controls should use a general-purpose implementation
such as `uwot` or `umap-learn`.

When a one-call function receives a `float::float32` matrix, its returned
`layout` remains float32 to reduce host memory. Plot the embedding object
directly with `plot(fit)`; fastEmbedR decodes the compact payload for graphics
and quality metrics without changing the stored layout.

## Main Functions

The complete class, backend, residency, method, and lifecycle inventory is
available from `fastEmbedR::fastEmbedR_api()` and in the
[public API map](docs/api-map.md).

| Function | Purpose |
| --- | --- |
| `precompute_knn()` | Native non-self KNN search on CPU, Metal, or CUDA, with backend-specific algorithm selection kept internal. |
| `tsne_knn()` | Native interpolation-based t-SNE from a supplied KNN object. |
| `tsne()` | One-call KNN plus interpolation-based t-SNE. |
| `umap_init()` | Build and retain a reusable UMAP graph plus its independent sparse initialization. |
| `umap_knn()` | Native UMAP from a supplied KNN object. |
| `umap()` | One-call KNN plus UMAP. |
| `pca()` | Backend-native truncated PCA; CPU calls expose `n.cores`, and `tsne_init = TRUE` returns a ready-to-use t-SNE initialization. |
| `select_landmarks()` | Select and retain a reusable landmark/reference split. |
| `fit_landmark_model()` | Fit ordinary UMAP or t-SNE on the landmark reference. |
| `project_landmark_model()` | Project held-out or new observations into the fixed reference. |
| `landmark_tsne()` / `landmark_umap()` | One-call landmark embedding and projection workflows. |
| `evaluate_embedding()` | Trustworthiness, neighbor preservation, label accuracy, and related metrics. |

### Optional Downstream Graph Utilities

Clustering is a secondary downstream facility rather than the package's
principal contribution. An embedding or KNN result can be passed to
`knn_graph()`, followed by `graph_cluster()`. Louvain and Leiden have
package-native CPU, CUDA, and Metal backends; Walktrap is CPU-only.

| Function | Purpose |
| --- | --- |
| `knn_graph()` | Compact graph from data, an embedding, or supplied neighbors. |
| `graph_cluster()` | Native Louvain, Leiden, or Pons-Latapy Walktrap communities. |

## Installation

For the development version:

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

For a reproducible analysis, install the versioned source archive recorded in
the analysis manifest rather than the moving development branch:

```sh
FASTEMBEDR_USE_CUDA=0 \
R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

See [Installation](docs/installation.md) for `fastEmbedR` CPU, Metal, and CUDA
embedding builds, including RAPIDS cuVS linkage for CUDA KNN. FAISS GPU is
optional and disabled unless explicitly requested at configuration time.
The portable CPU build requires R, Rcpp, and a C++17 compiler. Accelerator
libraries are optional and are detected during source installation.

## License

`fastEmbedR` is distributed under the MIT license. GPL packages such as `uwot`
are used only as optional external benchmark/reference tools, not as required
runtime dependencies or vendored source. Native KNN derivatives and optional
adapted code and linked libraries retain the FAISS MIT, Faiss-mlx Apache-2.0,
and RAPIDS cuVS
Apache-2.0 notices under `inst/LICENSES/`.

See [Source provenance and licensing](docs/provenance-and-licensing.md) for the
audit structure. Source-level records are maintained in [`inst/NOTICE`](inst/NOTICE),
[`inst/COPYRIGHTS`](inst/COPYRIGHTS), and the machine-readable
[`inst/THIRD_PARTY_DEPENDENCIES.json`](inst/THIRD_PARTY_DEPENDENCIES.json).
These files distinguish adapted or vendored code from optional linked
libraries, Apple system frameworks, design references, and benchmark-only
software. Every adapted/vendored component is pinned to an upstream commit and
mapped to both package and upstream source files.
