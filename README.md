# fastEmbedR

`fastEmbedR` is a compact R package for fast UMAP from precomputed nearest
neighbours. The current package is intentionally focused on two public tasks:

- `nn()` computes exact nearest neighbours quickly in native code.
- `embed_knn()` and `umap()` run UMAP from the KNN graph.
- `embed_knn(method = "tsne")` and `tsne()` run an `Rtsne_neighbors()`-style
  t-SNE path from precomputed neighbours.
- `opentsne_knn()`, `embed_knn(method = "opentsne")`, and `opentsne()` run a
  native C++, openTSNE-style two-phase t-SNE optimizer from precomputed
  neighbours without calling Python.
- `embed_knn(method = "infotsne")` and `infotsne()` run a native
  TorchDR-inspired negative-sampling t-SNE objective with better per-iteration
  scaling on large data.
- `transform_tsne()` and `landmark_tsne()` add an openTSNE-inspired
  fixed-reference transform path for landmark t-SNE without calling Python.

The restart baseline is designed to be comparable with
`uwot::umap(..., fast_sgd = TRUE)`: the large-data CPU path uses
`n_epochs = 200`, `negative_sample_rate = 5`, `min_dist = 0.01`, learning rate
`1`, a fuzzy-union graph, and at most four physical CPU cores.

## Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

The package installs without external FAISS. To enable the real FAISS C++
backend, install FAISS separately and rebuild with its prefix visible:

```sh
FASTEMBEDR_USE_FAISS=1 FAISS_HOME=/path/to/faiss R CMD INSTALL .
```

Then use `backend = "faiss"` for FAISS `IndexFlatL2` or
`backend = "faiss_ivf"` for FAISS `IndexIVFFlat`. If FAISS is not linked,
these explicit backends fail clearly. The older `backend = "cpu_faiss_ivf"` is
only a package-native FAISS-style IVF search and is labelled separately.

The RAPIDS cuVS CUDA KNN backend is also optional and external. On a CUDA
machine with cuVS installed, rebuild with:

```sh
FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1 CUVS_HOME=/path/to/cuvs R CMD INSTALL .
```

Then use `backend = "cuda_cuvs"` for a RAPIDS-inspired cuVS policy: exact
cuVS brute force for smaller searches and cuVS NN-descent for large self-KNN.
Use `backend = "cuda_cuvs_cagra"` to force CAGRA, `backend =
"cuda_cuvs_bruteforce"` for exact cuVS brute force, or `backend =
"cuda_cuvs_nndescent"` for cuVS NN-descent self-KNN. These backends never fall
back to CPU; if cuVS is not linked or a CUDA device is not visible, they fail
with a cuVS-specific error.

For t-SNE, cuVS currently accelerates the KNN stage only:

```r
fit <- tsne(x, backend = "cuda_cuvs_nndescent", perplexity = 30)
fit$parameters$backend     # "cpu" t-SNE optimizer
fit$parameters$nn_backend  # "cuda_cuvs_nndescent" KNN stage, if available
```

This is intentional. cuVS does not provide a full t-SNE optimizer; full RAPIDS
t-SNE belongs to cuML, while cuVS provides the neighbour-search building blocks.

The exact t-SNE optimizer also includes an internal experimental
KeOps-inspired blocked map-reduce repulsion path. It keeps the same exact
Student-t repulsive force, but evaluates it as an online row-wise reduction
instead of storing a dense interaction matrix or one full gradient copy per
thread. On the current CPU path it is a memory/prototyping option rather than
a speed win, so the faster pair-symmetric exact loop remains the default. The
blocked path is native C++; it does not depend on KeOps, Python, PyTorch, or
RKeOps at runtime.

## License And Provenance

The package is licensed as `GPL (>= 3)` so that UMAP implementation details can
be compared and adapted from GPL R implementations such as `uwot`.
FAISS is MIT-licensed and is linked only when available at build time; it is
not vendored into this package.
RAPIDS cuVS is Apache-2.0-licensed and is linked only when available at build
time; it is not vendored into this package.
KeOps is MIT-licensed and is used only as a design reference for the native
t-SNE blocked map-reduce repulsion path; no KeOps source is vendored or linked.

The experimental `backend = "metal_nndescent"` and
`backend = "cuda_nndescent"` KNN paths are native C++/Metal and C++/CUDA
implementations informed by the seeded NN-descent design in
[`mlx-vis`](https://github.com/hanxiao/mlx-vis), which is distributed under
Apache-2.0. `fastEmbedR` does not vendor, link to, or call `mlx-vis` or Python
for these backends. If source code from Apache-2.0 projects is copied in the
future, its copyright and license notices must be retained.

[`annembed`](https://github.com/jean-pierreBoth/annembed), distributed under
MIT OR Apache-2.0, is tracked as a design reference for HNSW-layer landmarking,
directed density-aware graph weights, diffusion/spectral initialization, and
graph-neighbour preservation diagnostics. No annembed source code is currently
copied or vendored; see `inst/ALGORITHMIC_REFERENCES.md` and `inst/NOTICE` for
the provenance notes.

The t-SNE-from-KNN API is modelled on
[`Rtsne::Rtsne_neighbors()`](https://cran.r-project.org/package=Rtsne).
Only the R-level neighbour-input behaviour and defaults were adapted. The
classic Rtsne Barnes-Hut C++ files are not vendored; the current optimizer is
native fastEmbedR code so the package keeps a cleaner publication path.

The `infotsne()` path is informed by
[`TorchDR::InfoTSNE`](https://github.com/TorchDR/TorchDR), distributed under
BSD-3-Clause. The current implementation ports the objective structure to
native C++: sparse KNN affinities for attraction and uniformly sampled
negatives for the repulsive `logsumexp(log Q)` term. It does not vendor
TorchDR source, call Python, or depend on PyTorch.

The native `opentsne()` optimizer and the landmark t-SNE transform path are
informed by
[`openTSNE`](https://github.com/pavlin-policar/openTSNE), distributed under
BSD-3-Clause. The full-embedding path follows openTSNE's two-phase optimizer:
early exaggeration followed by normal optimization, `learning_rate = n /
exaggeration` when requested, openTSNE-style momentum and gains, max-step
clipping, sparse positive KNN forces, and native BH/exact negative gradients.
The transform implementation follows the design of `TSNEEmbedding.transform()`:
initialize query points from reference neighbours, build row-wise
query-to-reference affinities, and optimize query points against a fixed
reference embedding. No openTSNE source is vendored, linked, or called.
The `backend = "metal"` transform optimizer is additionally informed by the
device-resident n-body optimization structure described in t-SNE-CUDA
(Chan, Rao, Huang, and Canny, 2018) and the BSD-3
[`CannyLab/tsne-cuda`](https://github.com/CannyLab/tsne-cuda) repository. No
t-SNE-CUDA source is copied or vendored. The current CUDA t-SNE path exposes a
native fastEmbedR exact-from-KNN kernel through
`embed_knn(method = "tsne", backend = "cuda")` when CUDA is compiled in; the
large-data FFT/FIt-SNE repulsive-field port remains a planned native
CUDA/Metal implementation.

## Basic Use

```r
library(fastEmbedR)

set.seed(1)
x <- scale(iris[, 1:4])
labels <- iris$Species

knn <- nn(x, k = 31)
layout <- embed_knn(knn, method = "umap", seed = 1)

plot(layout, pch = 21, bg = labels)
```

The one-call interface computes KNN internally:

```r
fit <- umap(x, labels = labels, n_neighbors = 30, seed = 1)
plot(fit)
```

For large landmark runs, an experimental opt-in refinement policy can update
only rows with low projection confidence:

```r
options(fastEmbedR.selective_landmark_refinement = TRUE)
fit <- umap(x, landmarks = 0.5, seed = 1)
```

This keeps well-projected rows fixed during the short post-projection
refinement. It is disabled by default because it can reduce refinement time but
should be checked on each dataset for local-neighbour quality.

## API

| Function | Purpose |
| --- | --- |
| `nn()` | Exact KNN for a data matrix or query matrix. |
| `embed_knn()` | UMAP from a supplied KNN object or index/distance matrices. |
| `tsne()` | t-SNE from data, using KNN input and Rtsne-style defaults. |
| `opentsne_knn()` | Native openTSNE-style t-SNE directly from a supplied KNN object or KNN matrices. |
| `opentsne()` | Native openTSNE-style two-phase t-SNE from data, or from an `nn()` result. |
| `infotsne()` | InfoTSNE from data, using KNN attraction and sampled negative repulsion. |
| `transform_tsne()` | Place query points into an existing t-SNE embedding with a fixed-reference optimizer. |
| `landmark_tsne()` | Embed landmarks, then transform the remaining points into the reference t-SNE map. |
| `umap()` | One-call preprocessing, KNN, UMAP, and optional scoring. |
| `supervised_umap()` | UMAP with label-adjusted KNN distances. |
| `transform_embedding()` | Project query points into an existing embedding. |
| `evaluate_embedding()` | Trustworthiness, KNN preservation, silhouette, and related metrics. |
| `backend_info()` | Report CPU/CUDA/Metal availability without silently falling back. |

`backend_info()` also reports whether the real FAISS C++ backend and RAPIDS
cuVS CUDA backend were linked.

Direct openTSNE-style t-SNE from precomputed neighbours:

```r
x <- scale(as.matrix(iris[, 1:4]))
knn <- fastEmbedR::nn(x, k = 31)

layout <- fastEmbedR::opentsne_knn(knn, perplexity = 10)
plot(layout, pch = 21, bg = iris$Species)

fit <- fastEmbedR::opentsne(knn, labels = iris$Species, perplexity = 10)
plot(fit)
```

## Comparison With uwot

Use the same KNN width and UMAP parameters when comparing:

```r
library(uwot)

k <- 50
knn <- fastEmbedR::nn(x, k = k + 1)
idx <- knn$indices[, -1, drop = FALSE]
dst <- knn$distances[, -1, drop = FALSE]

fast <- fastEmbedR::embed_knn(list(indices = idx, distances = dst), seed = 4)
ref <- uwot::umap(
  x,
  nn_method = list(idx = idx, dist = dst),
  n_neighbors = k,
  n_epochs = 200,
  min_dist = 0.01,
  negative_sample_rate = 5,
  learning_rate = 1,
  fast_sgd = TRUE,
  n_threads = 4,
  n_sgd_threads = 4,
  ret_model = FALSE
)
```

For large datasets, visual inspection matters. Benchmark scripts outside the
package under `tools/` create side-by-side panels against `uwot_fast_sgd`, but
those scripts are deliberately excluded from the package build.

## Benchmark Snapshot

The current local benchmark summary is in
[`BENCHMARK_SUMMARY.md`](BENCHMARK_SUMMARY.md). It compares fastEmbedR with
available R tools on MNIST, including `uwot`, `umap`, `Rtsne`, `tsne`,
`Rdimtools`, and the `ReductionWrappers` Python openTSNE wrapper where that
wrapper is available through `reticulate`.

The short version is:

- On MNIST 2.5k, fastEmbedR is the fastest group among the tested R/package
  calls; `Rtsne_neighbors()` and the Python openTSNE wrapper retain slightly
  higher t-SNE trustworthiness.
- On full MNIST 70k, fastEmbedR's native openTSNE-style path gives the best
  quality among the current fastEmbedR t-SNE/UMAP paths.
- On the older full MNIST 70k UMAP-only comparison, `uwot_fast_sgd` remains
  faster than current fastEmbedR CPU UMAP at essentially tied quality; this is
  the main remaining UMAP optimization target.

## GPU Backends

CUDA and Metal checks are explicit. If a requested GPU backend is unavailable,
the package reports an error/status rather than silently running on CPU and
calling it GPU. The current restart focuses on the multicore CPU UMAP path
first; GPU UMAP remains a native experimental path.

For landmark t-SNE, `transform_tsne(..., backend = "metal")` and
`landmark_tsne(..., backend = "metal")` run the fixed-reference transform
optimizer in native Metal for two-dimensional maps. `backend = "cuda"` for this
specific transform path is intentionally not enabled yet; it errors rather than
falling back to CPU and pretending to be CUDA.

For full t-SNE from precomputed neighbours, `embed_knn(method = "tsne",
backend = "cuda")` dispatches to a native exact CUDA optimizer when the package
is built with CUDA support. This is a quality/parity path for moderate sizes,
not the final scalable FFT t-SNE implementation.
