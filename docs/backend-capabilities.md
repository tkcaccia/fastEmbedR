# Backend Capabilities

This page states what each backend does and what it does not do. The central
rule is simple: if a function is requested with `backend = "metal"` or
`backend = "cuda"`, it must run a real native GPU path or fail clearly.

## Capability Matrix

| Function | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `nn()` | native exact and native NN-descent | native Metal NN-descent | optional RAPIDS cuVS NN-descent | KNN results are reusable across UMAP, openTSNE, and landmark workflows. |
| `umap_knn()` | native C++ CSR graph and optimizer | native Metal `atomic_inplace` optimizer | native CUDA pure-atomic optimizer | Metal/CUDA optimizers use the supplied graph; unavailable GPU backends fail clearly. |
| `umap()` | native KNN plus `umap_knn()` | native Metal KNN plus Metal UMAP where available | cuVS KNN plus CUDA UMAP where available | The backend actually used is reported. |
| `opentsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT | PCA initialization is the default for quality/stability. |
| `opentsne()` | native KNN plus `opentsne_knn()` | native Metal KNN plus Metal openTSNE where available | cuVS KNN plus CUDA openTSNE where available | The package does not call Python openTSNE in public functions. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by openTSNE landmarking. |
| `landmark_umap()` | native landmark embed/project/refine | native Metal projection/refinement kernels where available | native CUDA projection/refinement kernels where built | Landmarking is an explicit approximation, not a replacement for full UMAP. |
| `landmark_tsne()` | native landmark embed plus transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Projection quality is tracked separately in benchmark plots. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

## Distance Metrics

| `nn()` metric | CPU | Metal | CUDA/cuVS | FAISS | Notes |
| --- | --- | --- | --- | --- | --- |
| `euclidean` | exact and approximate | native Metal KNN | cuVS/CUDA KNN where built | optional FAISS L2 indexes | Validated default for UMAP/openTSNE. |
| `cosine` | exact CPU | not yet enabled | not yet enabled | not yet enabled | `backend = "auto"` resolves to CPU; unsupported explicit backends error clearly. |

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

For KNN, `backend = "auto"` uses exact Metal on moderate self-KNN searches
where local benchmarks show it is faster than the current Metal NN-descent
pipeline. Metal NN-descent remains available explicitly with
`backend = "metal_nndescent"` and is reserved by auto for larger searches where
exact all-pairs KNN is too expensive.

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
It links to them only when they are explicitly available. This keeps the R
package smaller and easier to submit while still allowing accelerated builds on
CUDA machines.
