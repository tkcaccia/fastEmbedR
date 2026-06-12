# fastEmbedR

`fastEmbedR` is an opinionated KNN-first package for native UMAP and
openTSNE-style embeddings in R.

The package is deliberately narrow now: one KNN API, one UMAP API, one
openTSNE-style API, and explicit CPU/Metal/CUDA backend reporting. Slow or
visually weak experimental branches have been removed from the public surface.

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

### Optional RAPIDS cuVS KNN

`fastEmbedR` includes native C++ bindings for RAPIDS cuVS KNN, including
brute-force search, CAGRA, and NN-descent. The RAPIDS libraries themselves are
not vendored into the R package because the CUDA binary stack is large and must
match the host NVIDIA driver/CUDA runtime. On a CUDA Linux machine, install the
external cuVS SDK with:

```sh
tools/install_cuvs_linux.sh
. ~/.fastEmbedR/cuvs_env.sh
R CMD INSTALL .
```

After installation:

```r
library(fastEmbedR)
cuvs_available()
backend_info()
knn <- nn(x, k = 50, backend = "cuda_cuvs_nndescent")
```

If cuVS is unavailable, explicit cuVS requests fail clearly. They are never
reported as CUDA/cuVS after running on CPU.

## Backend Rule

All public compute functions that can use parallel CPU work now expose
`n_threads`; use `n_threads = 4` for a fixed four-core CPU run. Functions with
native GPU support also accept `backend = "metal"` on Apple Silicon. The
convenience `backend = "gpu"` requests a real native GPU path where the
function has one; unsupported GPU work fails clearly instead of being labelled
as GPU after running on CPU.

The Metal implementation is kept simple on purpose. The package does not opt in
to an MPSGraph FFT backend because the local MNIST benchmark did not show a
speed advantage over the package-native Metal FFT-grid path. There is no user
option for choosing this experimental backend.

## Native Metal UMAP

On Apple Silicon, `umap_knn(..., backend = "metal")` uses the restored native
Objective-C++/Metal UMAP optimizer from the supplied KNN graph. It does not call
Python, `reticulate`, Torch, or MLX. The graph preparation is still shared with
the CPU CSR path in this restored build; the optimizer itself is Metal-labelled
only when the native Metal path is used.

Metal UMAP has a single optimizer path: `atomic_inplace`. This is the
visually validated native Metal edge-update kernel used for published
benchmarks; the slower or distorted experimental Metal UMAP optimizers were
removed to keep results reproducible and the API simple.

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

The command below runs the local MNIST 70k benchmark on the raw flattened
28x28 images, not on PCA features:

```sh
Rscript tools/benchmark_mnist70k_current_backends.R \
  --feature-source=raw \
  --n=70000 \
  --k=50 \
  --seed=6 \
  --threads=4 \
  --backends=cpu,metal \
  --run-uwot=true \
  --run-umap=true \
  --run-opentsne=true \
  --run-rtsne=true \
  --run-landmark=true
```

Latest local raw-MNIST run, using flattened 784-column images and a 5,000-point
quality sample:

| method | backend | NN sec | embed sec | proj+transform sec | trust |
| --- | --- | ---: | ---: | ---: | ---: |
| openTSNE | CPU | 64.752 | 4.263 | NA | 0.319 |
| openTSNE | Metal | 40.787 | 6.696 | NA | 0.305 |
| Rtsne_neighbors | CPU | 64.752 | 34.871 | NA | 0.198 |
| openTSNE landmark50 | CPU | 192.299 | 2.535 | 143.509 | 0.269 |
| openTSNE landmark50 | Metal | 50.390 | 3.709 | 49.807 | 0.270 |
| UMAP | CPU | 64.752 | 6.914 | NA | 0.281 |
| UMAP | Metal | 40.787 | 2.159 | NA | 0.283 |
| UMAP landmark50 | CPU | 197.833 | 3.581 | 159.008 | 0.267 |
| UMAP landmark50 | Metal | 51.452 | 1.236 | 53.215 | 0.262 |
| uwot::umap fast_sgd | CPU | NA | 67.581 | NA | 0.283 |

For `uwot::umap`, `embed sec` is the total exposed call time because `uwot`
does not report KNN, graph construction, and SGD timing separately. For
fastEmbedR rows, KNN and embedding/projection timing are recorded separately.

## License And Provenance

The package remains licensed as `GPL (>= 3)` for a conservative free-software
publication path. The current openTSNE-style implementation is native
fastEmbedR C++ code informed by BSD-3-Clause openTSNE. `Rtsne` informed
neighbour-input validation and perplexity defaults, but Barnes-Hut code is not
vendored or exposed. FAISS and RAPIDS cuVS are optional
external KNN backends and are not vendored.

Detailed provenance is recorded in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.
