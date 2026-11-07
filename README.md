# fastEmbedR

**Home** |
[Installation][installation] |
[Implementation][implementation] |
[Performance Engineering][performance-engineering] |
[Examples][examples] |
[Benchmarks][benchmarks] |
[API][api] |
[Reproducibility][reproducibility] |
[Development][development] |
[References][references] |
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
- native CPU exact, faissR HNSW, and Apple Metal exact/IVF-Flat KNN for one-call
  embeddings;
- optional GPU-resident exact/IVF-Flat CUDA KNN through RAPIDS cuVS.

Resident 2D Metal UMAP currently uses one CPU thread for the first half of
its optimizer epochs, followed by the native Metal optimizer. The fit records
both stage lengths. Resident 3D Metal UMAP runs its optimizer entirely on
Metal. File-backed Metal UMAP follows a separate route.

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
for an untested GPU. See the [hardware evidence contract][validation].

The intended analysis workflow uses `pca()`, `tsne()`, `umap()`, and
`graph_cluster()` for resident and file-backed inputs:

1. call `pca()`, `tsne()`, or `umap()` on a matrix;
2. if both embeddings must share neighbors, call `precompute_knn()` once
   and pass its result to `tsne()` and `umap()`;
3. evaluate or plot the embedding.

`graph_cluster(x)` builds a graph from a data matrix; pass a prepared graph
to reuse its edges. `landmarks`, `k`, and `vote_k` are named controls.
`summary(fit)` reports backend, devices, scope, stages, and recorded memory
without reading file-backed output. `massive_read_rows()` inspects
bounded rows of file-backed matrices and clustering results. Direct KNN
optimizers and file-stage functions remain available for controlled work;
they are grouped under advanced reference topics.

For reusable file-backed neighbors, describe the input with
`massive_matrix()` and call `precompute_knn(x, output = "knn-prefix")`.
Session defaults can be set with `options(storage = "file_backed",
memory_limit = "8GB")` alongside `options(backend = "cpu")`.
The main analysis functions also accept `storage` directly when one call
needs to override the session setting.
The file-backed result is a graph descriptor, not a resident KNN matrix;
`umap()` and `tsne()` can prepare fuzzy weights or affinities from a
file-backed matrix in one experimental full-data call. Supply a new `.f32`
output path and enough SSD space for the intermediate graph and
initialization. Full-data UMAP computes a streamed spectral start from its
fuzzy graph; full-data t-SNE uses PCA. Prepared graphs remain available for
reuse and recovery.

For the one-call functions `tsne()` and `umap()`, the embedding backend is
deliberately limited to `backend = "cpu"`, `"metal"`, or `"cuda"`. Internal
CPU one-call embeddings use package-native exhaustive float32 search below
5,000 observations and HNSW otherwise. The HNSW route uses a metric-, shape-,
and `k`-aware target-0.99 tuning policy. Metal uses native exact search below
4,096 rows and for inputs up to 50,000 rows and 16,384 columns when the
estimated search work is at most 125 billion row-pair feature operations.
It uses recall-tuned IVF-Flat otherwise.
CUDA uses cuVS brute-force exact search below 100,000 rows and cuVS IVF-Flat
above that threshold, then passes package-owned device pointers into UMAP or
t-SNE. It does not call another R package for KNN. No unavailable GPU backend
is silently relabelled as CPU.

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
y_from_knn <- fastEmbedR::umap(knn, backend = "cpu", seed = 1)
```

Set session defaults with base R options when several calls should use the
same backend and CPU limit. Explicit function arguments still take precedence.

```r
old_options <- options(backend = "cpu", n.cores = 4L)
fit <- fastEmbedR::umap(x)
options(old_options)
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

| Function | Purpose |
| --- | --- |
| `pca()` | PCA on a matrix or file-backed source. |
| `tsne()` / `umap()` | 2D or 3D embeddings from a matrix or reusable KNN. Set `landmarks` for approximation. |
| `graph_cluster()` | Resident or file-backed full-graph and landmark Louvain/Leiden. |
| `massive_matrix()` | Describe a file-backed source for the same analysis functions. |
| `massive_read_rows()` | Inspect bounded matrix or cluster-result rows. |
| `precompute_knn()` | Reuse the same neighbors across embeddings. |
| `evaluate_embedding()` | Embedding quality metrics. |

### Optional Downstream Graph Utilities

Clustering is a secondary downstream facility rather than the package's
principal contribution. An embedding or KNN result can be passed to
`knn_graph()`, followed by `graph_cluster()`. Louvain and Leiden have
package-native CPU, CUDA, and Metal backends; Walktrap is CPU-only.

| Function | Purpose |
| --- | --- |
| `knn_graph()` | Compact graph from data, an embedding, or supplied neighbors. |
| `graph_cluster()` | Native Louvain, Leiden, or Pons-Latapy Walktrap communities. |

CPU, CUDA, and Metal Louvain and Leiden support graph consensus and projection
weighting. Both use the supplied graph; projection weighting additionally
uses observation-aligned UMAP or t-SNE coordinates. No original variables
are required. For example, with precomputed neighbors `nn` and coordinates
`xy`:

```r
graph <- knn_graph(nn, k = 30, mutual = TRUE, weight = "snn",
    backend = "cpu")
consensus <- graph_cluster(graph, method = "leiden", backend = "cpu",
    resolution = c(0.05, 0.1, 0.15, 0.2),
    seed = c(1234, 2345, 3456, 4567, 5678))
projected <- graph_cluster(graph, method = "leiden", backend = "cpu",
    projection = list(layout = xy), resolution = 0.03, seed = 1234)
combined <- graph_cluster(graph, method = "leiden", backend = "cpu",
    resolution = c(0.05, 0.1, 0.15, 0.2),
    seed = c(1234, 2345, 3456, 4567, 5678),
    projection = list(layout = xy))
```

Vectors in either `resolution` or `seed` automatically trigger consensus over
all combinations. The first resolution and seed control final clustering.
Projection weighting, when supplied, follows consensus. Scalar controls and
`projection = NULL` retain ordinary clustering. Vector controls use
`n_runs = 1`; the seed vector determines consensus repeats.

Use `method = "louvain"` for Louvain. To enable small-group
postprocessing, add `small_groups = list(knn = nn, min_size = 20)` to any
call. This reassigns small communities to their nearest supplied core
neighbor and reports reassigned and unresolved rows. The procedures do not
request a fixed group count. See `?graph_cluster` for controls and metadata;
these options cannot be combined with `layout_merge`. CUDA and Metal run
clustering partitions on their devices and perform consensus edge weighting,
projection weighting, and small-group assignment on the host. CUDA community
updates are not bitwise deterministic for a
fixed seed; the result reports this in `repeatability`.

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

See [Installation][installation] for `fastEmbedR` CPU, Metal, and CUDA
embedding builds, including RAPIDS cuVS linkage for CUDA KNN.
The CPU build requires R 4.6 or later, Rcpp, faissR, and a C++17 compiler.
Accelerator libraries are optional and detected during source installation.

## License

`fastEmbedR` is distributed under the MIT license. GPL packages such as `uwot`
are used only as optional external benchmark/reference tools, not as required
runtime dependencies or vendored source. Native KNN derivatives and optional
adapted code and linked libraries retain the FAISS MIT, Faiss-mlx Apache-2.0,
and RAPIDS cuVS
Apache-2.0 notices under `inst/LICENSES/`.

See [Source provenance and licensing][provenance] for the
audit structure. Source-level records are maintained in
[`inst/NOTICE`][notice], [`inst/COPYRIGHTS`][copyrights], and the
machine-readable [`inst/THIRD_PARTY_DEPENDENCIES.json`][third-party].
These files distinguish adapted or vendored code from optional linked
libraries, Apple system frameworks, design references, and benchmark-only
software. Every adapted/vendored component is pinned to an upstream commit and
mapped to both package and upstream source files.

[installation]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/installation.md
[implementation]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/implementation.md
[performance-engineering]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/backend-performance-engineering.md
[examples]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/examples.md
[benchmarks]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/benchmarks.md
[api]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/usage-api.md
[reproducibility]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/reproducibility.md
[development]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/development.md
[references]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/references.md
[validation]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/backend-validation.md
[provenance]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/docs/provenance-and-licensing.md
[notice]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/inst/NOTICE
[copyrights]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/inst/COPYRIGHTS
[third-party]:
  https://github.com/tkcaccia/fastEmbedR/blob/main/inst/THIRD_PARTY_DEPENDENCIES.json
