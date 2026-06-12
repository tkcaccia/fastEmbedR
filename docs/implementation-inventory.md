# Implementation And Library Inventory

This page records what is actually implemented, linked, or only used as a
design reference. The package does not call Python for the public UMAP or
openTSNE paths.

Detailed provenance is also recorded in [../inst/NOTICE](../inst/NOTICE) and
[../inst/ALGORITHMIC_REFERENCES.md](../inst/ALGORITHMIC_REFERENCES.md).

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
| CPU fuzzy graph | Native C++ CSR graph path | uwot/UMAP fuzzy simplicial set behaviour | No | Uses smooth KNN bandwidths, local connectivity, fuzzy union weights, and compact CSR storage. |
| CPU optimizer | Native C++ uwot-compatible fast-SGD-style path | uwot fast-SGD scheduling and UMAP objective | No | Epoch scheduling, negative sampling, learning-rate decay, and edge sampling are kept close to uwot for visual parity. |
| Metal UMAP | Native Objective-C++/Metal `atomic_inplace` optimizer | mlx-vis design ideas for GPU-resident NN-descent/UMAP style work | Apple Metal on macOS | No Python/MLX runtime. Slow/distorted Metal optimizer variants were removed; `atomic_inplace` is the default and only public Metal optimizer. |
| CUDA UMAP | Native CUDA fused pure-atomic UMAP path when built with CUDA | RAPIDS cuML/cuVS and uwot/UMAP scheduling ideas informed architecture | CUDA toolkit at build/runtime | Explicit CUDA requests fail if unavailable. Old hybrid/deterministic CUDA UMAP variants were removed. |
| Landmark UMAP | Native C++/Metal projection and refinement helpers | UMAP transform/landmark workflow and package benchmarks | No for CPU; Apple Metal for Metal backend | Landmarking is an explicit approximation and is labelled separately in benchmark tables. |
| Spectral/PCA initialization | Native C++ helpers; optional fastPLS-style randomized SVD/PCA ideas | tkcaccia/fastPLS rSVD/PCA design reference | No external fastPLS runtime | Used for stable initialization where appropriate; no fastPLS package call is required. |

## Nearest-neighbour Backends Used By Both Methods

| Backend | Implementation | Library or project used | Runtime dependency | Notes |
| --- | --- | --- | --- | --- |
| `cpu` / exact | Native C++ exact row-distance search | Rnanoflann-style API behaviour informed early design | No | Used for small data and reference checks. |
| `cpu_nndescent` | Native C++ approximate NN-descent | mlx-vis and annembed design references | No | NEW/OLD candidate handling, reverse candidates, and active-row pruning are implemented in package code. |
| `metal_nndescent` | Native Objective-C++/Metal approximate NN-descent | mlx-vis design reference | Apple Metal on macOS | Used for Metal benchmarks; results are reported as Metal only when this native path runs. |
| `faiss` / `faiss_ivf` | Native C++ bridge to external FAISS | FAISS | Optional external FAISS library | FAISS is not vendored and is only linked when explicitly available. |
| `cuda_cuvs*` | Native C++ bridge to external RAPIDS cuVS | RAPIDS cuVS | Optional external RAPIDS cuVS/CUDA install | Includes cuVS brute force, CAGRA, and NN-descent KNN. cuVS is not vendored. |

## Studied But Not Runtime Dependencies

The following projects influenced design or benchmarking decisions but are not
called by the package at runtime:

- Python openTSNE
- TorchDR
- KeOps
- mlx-vis
- annembed
- t-SNE-CUDA
- RAPIDS cuML
- RAPIDS cuVS for non-KNN embedding kernels
- FAISS for embedding kernels
- Rtsne Barnes-Hut internals
- older package experiments for classic `tsne()`, InfoTSNE, PaCMAP, TriMap,
  and LocalMAP

