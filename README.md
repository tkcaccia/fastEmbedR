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

For the chiamaka CUDA workstation (`chiamaka@137.158.224.178`), use the same
script with an explicit RAPIDS CUDA version because the machine has a new
NVIDIA driver but an older system `nvcc`:

```sh
FASTEMBEDR_CUVS_CUDA_VERSION=12.9 tools/install_cuvs_linux.sh
. ~/.fastEmbedR/cuvs_env.sh
R CMD INSTALL .
```

The activation file sets `CUVS_HOME`, `CUDA_HOME`, `NVCC`,
`FASTEMBEDR_USE_CUDA=1`, and `FASTEMBEDR_USE_CUVS=1`, so fastEmbedR builds
against the micromamba cuVS/CUDA environment instead of `/usr/bin/nvcc`.

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

The main remaining Metal openTSNE speed gap is the FFT itself. CUDA uses
NVIDIA cuFFT, while the Mac path uses package-native Metal FFT kernels. The
512x512 openTSNE/FIt-SNE grid path now uses a validated Stockham FFT kernel
adapted from the MIT-licensed AppleSiliconFFT design; other grid sizes still
use the generic package-native Metal Cooley-Tukey path. To continue developing
the cuFFT-like Mac path without changing embedding math, see
[`docs/metal-fft-roadmap.md`](docs/metal-fft-roadmap.md) and profile the
current FFT stages with:

```r
system("Rscript tools/profile_metal_opentsne_fft.R --n=10000 --k=50")
```

On the local MNIST raw-pixel profile, the FFT stages account for most of the
Metal GPU time, so new Metal FFT kernels must beat that profiler before they
are promoted into openTSNE.

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
backend is compiled. Native CUDA FFT openTSNE is implemented with CUDA kernels
and cuFFT when the CUDA backend is compiled. Unsupported GPU requests fail
clearly instead of falling back to CPU.

## Automatic Parameters

`opentsne()` and `opentsne_knn()` use `auto_config = TRUE` by default. Missing
t-SNE settings are resolved in native C++ using the opt-SNE strategy: `"auto"`
learning rate becomes `n / early_exaggeration`, early exaggeration can stop at
the local maximum of KLD relative change, and the normal phase can stop when the
KLD improvement drops below the opt-SNE threshold. The KLD monitor is enabled
only where it is computationally honest: CPU/small exact runs. Large FFT and
GPU runs keep opt-SNE's learning-rate/default-limit policy but do not perform a
hidden CPU O(n^2) KLD poll or report it as GPU work.

`umap()` and `umap_knn()` also choose internal defaults from the supplied KNN
distance profile in C++. This keeps the public API small while letting Shuttle-
like broad-shell data and high-variability data get less brittle defaults.

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

## Implementation And Library Inventory

This section records what is actually implemented, linked, or only used as a
design reference. The package does not call Python for the public UMAP or
openTSNE paths.

### openTSNE-style t-SNE

| Component | fastEmbedR implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| KNN input API | Native R/C++ wrapper accepts precomputed `indices` and `distances` | `Rtsne::Rtsne_neighbors()` informed input validation and defaults | No | Rtsne source is not vendored; used as behavioural reference for KNN t-SNE compatibility. |
| Perplexity affinities | Native C++ binary search on supplied KNN distances | Rtsne and openTSNE papers/code informed the math | No | Produces sparse high-dimensional probabilities from KNN. |
| CPU openTSNE optimizer | Native C++ two-phase optimizer | openTSNE design reference | No | Early exaggeration, normal phase, momentum/gains, max-step clipping, sparse attractive forces. |
| CPU negative gradient | Native C++ exact or FFT-grid/FIt-SNE-style approximation | openTSNE, FIt-SNE, t-SNE-CUDA design references | No | Barnes-Hut was removed from the public path after MNIST 70k tests favoured FFT-grid. |
| opt-SNE automation | Native C++/R parameter resolver | Multicore-opt-SNE / opt-SNE paper and BSD-3-Clause codebase as design reference | No | Auto learning rate and safe auto iteration policy; large FFT/GPU runs do not hide CPU KLD polling. |
| Fixed-reference transform | Native C++ and native Metal paths | openTSNE transform design, t-SNE-CUDA architecture | No | Used by `transform_tsne()` and `landmark_tsne()`. |
| Metal openTSNE | Native Objective-C++/Metal kernels | Apple Metal framework; AppleSiliconFFT-inspired Stockham FFT; MPSGraph only for diagnostics | Apple Metal on macOS | No Python, Torch, MLX, or reticulate. Current default is package-native Metal FFT-grid, not MPSGraph. |
| Metal FFT diagnostic | Internal diagnostic path behind environment flags/tools | Apple MPSGraph FFT APIs | Apple MetalPerformanceShadersGraph on macOS | Used to compare MPSGraph FFT/convolution against package-native Metal FFT; not a public algorithm option. |
| CUDA openTSNE | Native CUDA kernels plus cuFFT when built with CUDA | CUDA/cuFFT; t-SNE-CUDA and RAPIDS/openTSNE designs informed architecture | CUDA toolkit/cuFFT at build/runtime | Explicit CUDA requests fail if CUDA support is unavailable; no CPU fallback is reported as CUDA. |
| Python openTSNE wrapper comparison | Benchmark-only optional comparison | `ReductionWrappers::openTSNE()` / Python `openTSNE` | Optional, benchmark scripts only | Not used by package functions. |

### UMAP

| Component | fastEmbedR implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| UMAP KNN input API | Native R/C++ wrapper accepts precomputed KNN | `umap`, `uwot`, and KNN-first benchmark practice informed API shape | No | `umap_knn()` and `umap()` keep KNN time separate from embedding time. |
| CPU fuzzy graph | Native C++ CSR graph path | uwot/UMAP fuzzy simplicial set behaviour | No | Uses smooth KNN bandwidths, local connectivity, fuzzy union weights, and compact CSR storage. |
| CPU optimizer | Native C++ uwot-compatible fast-SGD-style path | uwot fast-SGD scheduling and UMAP objective | No | Epoch scheduling, negative sampling, learning-rate decay, and edge sampling are kept close to uwot for visual parity. |
| Metal UMAP | Native Objective-C++/Metal `atomic_inplace` optimizer | mlx-vis design ideas for GPU-resident NN-descent/UMAP style work | Apple Metal on macOS | No Python/MLX runtime. Slow/distorted Metal optimizer variants were removed; `atomic_inplace` is the default and only public Metal optimizer. |
| CUDA UMAP | Native CUDA fused pure-atomic UMAP path when built with CUDA | RAPIDS cuML/cuVS and uwot/UMAP scheduling ideas informed architecture | CUDA toolkit at build/runtime | Explicit CUDA requests fail if unavailable. Old hybrid/deterministic CUDA UMAP variants were removed. |
| Landmark UMAP | Native C++/Metal projection and refinement helpers | UMAP transform/landmark workflow and package benchmarks | No for CPU; Apple Metal for Metal backend | Landmarking is an explicit approximation and is labelled separately in benchmark tables. |
| Spectral/PCA initialization | Native C++ helpers; optional fastPLS-style randomized SVD/PCA ideas | tkcaccia/fastPLS rSVD/PCA design reference | No external fastPLS runtime | Used for stable initialization where appropriate; no fastPLS package call is required. |

### Nearest-neighbour backends used by both methods

| Backend | Implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| `cpu` / exact | Native C++ exact row-distance search | Rnanoflann-style API behaviour informed early design | No | Used for small data and reference checks. |
| `cpu_nndescent` | Native C++ approximate NN-descent | mlx-vis and annembed design references | No | NEW/OLD candidate handling, reverse candidates, and active-row pruning are implemented in package code. |
| `metal_nndescent` | Native Objective-C++/Metal approximate NN-descent | mlx-vis design reference | Apple Metal on macOS | Used for Metal benchmarks; results are reported as Metal only when this native path runs. |
| `faiss` / `faiss_ivf` | Native C++ bridge to external FAISS | FAISS | Optional external FAISS library | FAISS is not vendored and is only linked when explicitly available. |
| `cuda_cuvs*` | Native C++ bridge to external RAPIDS cuVS | RAPIDS cuVS | Optional external RAPIDS cuVS/CUDA install | Includes cuVS brute force, CAGRA, and NN-descent KNN. cuVS is not vendored. |

### Studied but not used as runtime dependencies

The following projects influenced design or benchmarking decisions but are not
called by the package at runtime: Python openTSNE, TorchDR, KeOps, mlx-vis,
annembed, t-SNE-CUDA, RAPIDS cuML, RAPIDS cuVS for non-KNN embedding kernels,
FAISS for embedding kernels, Rtsne Barnes-Hut internals, and older package
experiments for classic `tsne()`, InfoTSNE, PaCMAP, TriMap, and LocalMAP.

## License And Provenance

The package remains licensed as `GPL (>= 3)` for a conservative free-software
publication path. The current openTSNE-style implementation is native
fastEmbedR C++ code informed by BSD-3-Clause openTSNE. `Rtsne` informed
neighbour-input validation and perplexity defaults, but Barnes-Hut code is not
vendored or exposed. FAISS and RAPIDS cuVS are optional
external KNN backends and are not vendored.

Detailed provenance is recorded in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`.
