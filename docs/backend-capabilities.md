# Backend Capabilities

This page states what each backend does and what it does not do. The central
rule is simple: if a function is requested with `backend = "metal"` or
`backend = "cuda"`, it must run a real native GPU path or fail clearly.

## Capability Matrix

| Function | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `nn()` | `faissR` FAISS CPU search | not implemented in `fastEmbedR` | `faissR` optional RAPIDS cuVS / FAISS GPU search | KNN belongs to `faissR`; `fastEmbedR` re-exports a thin wrapper. |
| `umap_knn()` | native C++ CSR graph and optimizer | native Metal `atomic_inplace` optimizer | native CUDA pure-atomic optimizer | Metal/CUDA optimizers use the supplied graph; unavailable GPU backends fail clearly. |
| `umap()` | FAISS CPU IVF-Flat through `faissR`, then `umap_knn()` | FAISS CPU IVF-Flat through `faissR`, then native Metal UMAP where available | FAISS GPU IVF-Flat through `faissR`, then CUDA UMAP where available | The KNN algorithm is fixed inside the one-call API. |
| `opentsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT | PCA initialization is the default for quality/stability. |
| `opentsne()` | FAISS CPU IVF-Flat through `faissR`, then `opentsne_knn()` | FAISS CPU IVF-Flat through `faissR`, then Metal openTSNE where available | FAISS GPU IVF-Flat through `faissR`, then CUDA openTSNE where available | The package does not call Python openTSNE in public functions. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by openTSNE landmarking. |
| `landmark_umap()` | native landmark embed/project/refine | native Metal projection/refinement kernels where available | native CUDA projection/refinement kernels where built | Landmarking is an explicit approximation, not a replacement for full UMAP. |
| `landmark_tsne()` | native landmark embed plus transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Projection quality is tracked separately in benchmark plots. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

## Distance Metrics

| `nn()` metric | CPU | Metal | CUDA/cuVS | FAISS | Notes |
| --- | --- | --- | --- | --- | --- |
| `euclidean` | FAISS CPU where built | not in `fastEmbedR` | cuVS/CUDA KNN where built | FAISS L2 indexes | Validated default for UMAP/openTSNE. |
| `cosine` / inner product | FAISS CPU where enabled by `faissR` | not in `fastEmbedR` | CUDA/cuVS/FAISS where enabled by `faissR` | FAISS IP indexes | Use normalized rows for cosine/IP workflows. |

## Backend Labels

Every benchmark row should record the backend requested and the backend used.
The package avoids ambiguous GPU reporting:

- `backend_requested = "metal"` and `backend_used = "metal"` means a native
  Metal path ran.
- `backend_requested = "cuda"` and `backend_used = "cuda"` means a native CUDA
  path ran.
- If the GPU path is unavailable, status should be `backend_unavailable` or
  `not_supported`, not a hidden CPU result.

## CPU

CPU paths are native C++ and use `n_threads` where the operation is safe to
parallelize. Current CPU priorities are:

- reuse KNN graphs instead of recomputing them;
- keep graph data in compact integer/float arrays;
- use CSR graph storage for UMAP;
- keep openTSNE attractive affinities sparse;
- avoid unnecessary copies between R matrices and C++ buffers.

## Metal

Metal is implemented with Objective-C++ and Metal kernels. Public UMAP and
openTSNE paths do not call Python, Torch, MLX, or `reticulate`.

KNN is no longer implemented in `fastEmbedR`. Use `faissR::nn()` or the
`fastEmbedR::nn()` wrapper. Metal KNN experiments were removed from
`fastEmbedR`; the supported accelerated KNN route is FAISS/cuVS through
`faissR`.

The validated UMAP Metal path is `atomic_inplace`; other slower or distorted
Metal UMAP optimizer experiments were removed from the public API.

The validated openTSNE Metal path is the package-native FFT-grid path with PCA
initialization. MPSGraph FFT remains diagnostic-only because local MNIST tests
did not justify replacing the package-native implementation.

## CUDA

CUDA support is optional at build time. The package can use:

- RAPIDS cuVS NN-descent for CUDA KNN;
- native CUDA UMAP kernels;
- native CUDA FFT-grid openTSNE kernels with cuFFT.

cuVS, CUDA, and cuFFT are not vendored into the package. They must be installed
on the CUDA machine and matched to the driver/toolkit stack. If CUDA is not
available, explicit CUDA requests fail clearly.

## External Libraries

`fastEmbedR` does not vendor large GPU libraries such as RAPIDS cuVS or FAISS.
Those libraries are handled by `faissR`. This keeps the embedding package
smaller while still allowing accelerated builds on CUDA machines.
