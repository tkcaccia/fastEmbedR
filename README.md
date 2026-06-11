# fastEmbedR

`fastEmbedR` is an opinionated KNN-first package for native UMAP and
openTSNE-style embeddings in R.

The public embedding API is intentionally small:

- `nn()` computes nearest neighbours in native code.
- `umap_knn()` embeds from a precomputed KNN graph with the restored
  uwot-compatible UMAP path.
- `umap()` computes or reuses KNN, then runs UMAP.
- `opentsne_knn()` embeds from a precomputed KNN graph with the native
  openTSNE-style CPU path.
- `embed_knn()` dispatches to UMAP by default, or to openTSNE with
  `method = "opentsne"`.
- `opentsne()` computes or reuses KNN, then runs the native openTSNE-style
  optimizer.
- `transform_tsne()` places new/query points into an existing openTSNE-style
  map.
- `landmark_tsne()` embeds landmarks with `opentsne()` and transforms the
  remaining observations.
- `evaluate_embedding()` reports trustworthiness, neighbour preservation,
  silhouette, label KNN accuracy, and related diagnostics.

The legacy `tsne()` and `infotsne()` package implementations were removed from
the public package. UMAP remains in the public API because it is the main path
being optimized and visually benchmarked against `uwot`.

## Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

Optional native KNN backends are linked only when available at build time.
Explicit unavailable GPU/FAISS/cuVS backends fail clearly rather than silently
running on CPU.

```sh
FASTEMBEDR_USE_FAISS=1 FAISS_HOME=/path/to/faiss R CMD INSTALL .
FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1 CUVS_HOME=/path/to/cuvs R CMD INSTALL .
```

## Backend Rule

All public compute functions that can use parallel CPU work now expose
`n_threads`; use `n_threads = 4` for a fixed four-core CPU run. Functions with
native GPU support also accept `backend = "metal"` on Apple Silicon. The
convenience `backend = "gpu"` requests a real native GPU path where the
function has one; unsupported GPU work fails clearly instead of being labelled
as GPU after running on CPU.

## Native Metal UMAP

On Apple Silicon, `umap_knn(..., backend = "metal")` uses the restored native
Objective-C++/Metal UMAP optimizer from the supplied KNN graph. It does not call
Python, `reticulate`, Torch, or MLX. The graph preparation is still shared with
the CPU CSR path in this restored build; the optimizer itself is Metal-labelled
only when the native Metal path is used.

The Metal UMAP source keeps three optimizer kernels for diagnostics:
`scheduled`, `atomic_delta`, and `torchdr_row_negatives`. The package default is
`scheduled`, and benchmark scripts force `scheduled` so the experimental modes
are not mixed into published comparisons.

Native Metal openTSNE is available when the `knn_tsne_opentsne_metal_cpp`
symbol is compiled. `negative_gradient_method = "auto"` resolves to the native
Metal FFT-grid path, which keeps the interpolation grid, FFT convolution, sparse
attractive forces, gains, and updates inside Objective-C++/Metal kernels. If
the native symbol is unavailable, `opentsne_knn(..., backend = "metal")` fails
clearly instead of falling back to CPU and reporting a GPU result.

## CPU FIt-SNE-Style openTSNE

For larger CPU runs, `opentsne_knn(..., negative_gradient_method = "fft")`
uses a native multi-threaded grid-FFT approximation for the t-SNE repulsive
field. CPU `negative_gradient_method = "auto"` resolves to this FFT-grid path;
the older Barnes-Hut route has been removed because it was not competitive in
the current benchmarks. The implementation follows the FIt-SNE/t-SNE-CUDA idea
of separating sparse KNN attractive forces from interpolated negative forces on
a 2D grid. It is implemented in package C++ and does not call Python.

Native Metal FFT openTSNE is implemented in Objective-C++/Metal when the Metal
backend is compiled. Native CUDA FFT openTSNE is still refused until the CUDA
port exists; unsupported GPU requests fail clearly instead of falling back to
CPU.

## Basic Use

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

For landmark runs, `backend = "metal"` uses a fused native Metal projection
kernel that computes query-to-landmark KNN, interpolation, and projection
confidence in one pass before the fixed-reference transform. CPU/auto runs use
exact multi-threaded projection KNN by default and switch to a native
projection-specific approximation only for large projections where the cheaper
candidate search is worthwhile.

## API

| Function | Purpose |
| --- | --- |
| `nn()` | Native exact/approximate KNN for data/query matrices. |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `opentsne_knn()` | Direct native openTSNE-style optimizer from KNN. |
| `opentsne()` | One-call preprocessing, KNN, and openTSNE-style embedding. |
| `transform_tsne()` | Fixed-reference openTSNE-style transform for query points. |
| `landmark_tsne()` | Embed landmarks, then transform remaining rows. |
| `evaluate_embedding()` | Embedding quality metrics. |
| `backend_info()` | CPU/CUDA/Metal/FAISS/cuVS detection without silent fallback. |

## Benchmark Snapshot

The current local benchmark summary is in
[`BENCHMARK_SUMMARY.md`](BENCHMARK_SUMMARY.md). After the cleanup, benchmarks
should compare:

- `fastEmbedR::umap_knn()` from a supplied KNN graph.
- `uwot::umap()` / `uwot::umap(..., fast_sgd = TRUE)` as the R reference UMAP
  path.
- `fastEmbedR::opentsne_knn()` from a supplied KNN graph.
- `Rtsne::Rtsne_neighbors()` as the R reference t-SNE-from-KNN path.
- `ReductionWrappers::openTSNE()` as the Python openTSNE wrapper when the
  configured `reticulate` Python can import `openTSNE`.

## License And Provenance

The package remains licensed as `GPL (>= 3)` for a conservative free-software
publication path. The current openTSNE-style implementation is native
fastEmbedR C++ code informed by BSD-3-Clause openTSNE. `Rtsne` informed
neighbour-input validation and perplexity defaults, but Barnes-Hut code is not
vendored or exposed. FAISS and RAPIDS cuVS are optional
external KNN backends and are not vendored.

Detailed provenance is recorded in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.
