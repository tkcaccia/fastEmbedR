# Implementation And Library Inventory

This page records what is actually implemented, linked, or only used as a
design reference. The package does not call Python for the public UMAP or
openTSNE paths.

Detailed provenance is also recorded in [../inst/NOTICE](../inst/NOTICE) and
[../inst/ALGORITHMIC_REFERENCES.md](../inst/ALGORITHMIC_REFERENCES.md).
For function-by-function implementation notes and literature, see
[function-implementation-details.md](function-implementation-details.md).

## openTSNE-style t-SNE

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

## UMAP

| Component | fastEmbedR implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| UMAP KNN input API | Native R/C++ wrapper accepts precomputed KNN | `umap`, `uwot`, and KNN-first benchmark practice informed API shape | No | `umap_knn()` and `umap()` keep KNN time separate from embedding time. |
| CPU fuzzy graph | Native C++ CSR graph path | UMAP fuzzy simplicial set formulation; `uwot` only as an external behavioural benchmark | No | Uses smooth KNN bandwidths, local connectivity, fuzzy union weights, and compact CSR storage. |
| CPU optimizer | Native C++ epoch-scheduled stochastic optimizer | UMAP objective and package-local sampler/RNG/update loops | No | Epoch scheduling, negative sampling, learning-rate decay, and edge sampling are implemented locally for visual parity with UMAP references. |
| Metal UMAP | Native Objective-C++/Metal `atomic_inplace` optimizer | mlx-vis design ideas for GPU-resident NN-descent/UMAP style work | Apple Metal on macOS | No Python/MLX runtime. Slow/distorted Metal optimizer variants were removed; `atomic_inplace` is the default and only public Metal optimizer. |
| CUDA UMAP | Native CUDA fused pure-atomic UMAP path when built with CUDA | RAPIDS cuML/cuVS and UMAP scheduling concepts informed architecture | CUDA toolkit at build/runtime | Explicit CUDA requests fail if unavailable. Old hybrid/deterministic CUDA UMAP variants were removed. |
| Landmark UMAP | Native C++/Metal projection and refinement helpers | UMAP transform/landmark workflow and package benchmarks | No for CPU; Apple Metal for Metal backend | Landmarking is an explicit approximation and is labelled separately in benchmark tables. |
| Spectral/PCA initialization | Native C++ helpers; optional fastPLS-style randomized SVD/PCA ideas | tkcaccia/fastPLS rSVD/PCA design reference | No external fastPLS runtime | Used for stable initialization where appropriate; no fastPLS package call is required. |

## Nearest-neighbour Backends Used By Both Methods

| Backend | Implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| `faissR::nn()` | Companion-package KNN provider | FAISS, RAPIDS cuVS | `faissR` | KNN implementation, tuning, and backend selection live in `faissR`. |
| `faissR::candidate_knn()` | Candidate KNN utility | FAISS/candidate KNN work in `faissR` | `faissR` | Useful when candidate sets are produced outside the embedding optimizer. |
| `faissR::fast_kmeans()` | FAISS/cuVS k-means utility | FAISS/cuVS k-means paths in `faissR` | `faissR` | Kept out of the embedding code. |

Earlier native exact KNN, CPU NN-descent, Metal NN-descent, and grid KNN
experiments were removed from `fastEmbedR`. They remain useful historical
benchmarks, but the cleaned package delegates KNN to `faissR` so that FAISS and
cuVS support have one owner.

## Studied But Not Runtime Dependencies

The following projects influenced design or benchmarking decisions but are not
called by the package at runtime:

- Python openTSNE
- TorchDR
- KeOps
- mlx-vis
- annembed
- the removed native CPU/Metal NN-descent prototypes
- t-SNE-CUDA
- RAPIDS cuML
- RAPIDS cuVS for non-KNN embedding kernels
- FAISS for embedding kernels
- Rtsne Barnes-Hut internals
- older package experiments for classic `tsne()`, InfoTSNE, PaCMAP, TriMap,
  and LocalMAP
