# Backend Capabilities

This page states what each backend does and what it does not do. The central
rule is simple: if a function is requested with `backend = "metal"` or
`backend = "cuda"`, it must run a real native GPU path or fail clearly.

Inspect the capabilities of the installed build through the stable public API:

```r
capabilities <- fastEmbedR_capabilities()
capabilities[, c("backend", "available", "device", "precision",
                 "knn_engine", "runtime_libraries",
                 "unavailable_reason")]
```

The `cuvs` row describes the CUDA nearest-neighbor component; public backend
arguments continue to accept only `"cpu"`, `"cuda"`, or `"metal"`. Device-name
queries are best effort. The compiled native probes, not `nvidia-smi` or
`system_profiler`, determine whether a backend is available.

## Capability Matrix

| Function | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `precompute_knn()` / internal one-call KNN | native float32 HNSW | native exact or recall-tuned IVF-Flat | cuVS brute-force exact or IVF-Flat; optional FAISS GPU exact | KNN selection remains internal; the public function exposes only `k`, metric, backend, and CPU thread count. CUDA results stay device-resident. |
| `umap_init()` | native sparse graph initialization | native Metal initialization from prepared graph state | native CUDA initialization when compiled | Returns reusable graph and initial coordinates; a raw CUDA diagnostic call may materialize KNN on the host, whereas ordinary one-call CUDA UMAP remains resident. |
| `umap_knn()` | native C++ CSR graph and optimizer | native Metal `atomic_inplace` optimizer | native CUDA pure-atomic optimizer | Metal/CUDA optimizers use the supplied graph; unavailable GPU backends fail clearly. |
| `umap()` | native HNSW, then `umap_knn()` | native exact/IVF-Flat, then native Metal UMAP | native cuVS device KNN, then native CUDA UMAP | Optional FAISS GPU may provide exact search. CUDA KNN is not copied through R. Metal IVF exact-reranks candidates in the original dimensions and records its pilot recall. |
| `tsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT | Use `Y_init` or `init_data` for explicit PCA initialization. |
| `tsne()` | native HNSW, then `tsne_knn()` | native exact/IVF-Flat, then Metal t-SNE | native cuVS device KNN, then CUDA t-SNE | Optional FAISS GPU may provide exact search. The package does not call Python openTSNE in public functions. |
| `pca()` / t-SNE PCA init | native float32 blocked RSVD | native float32 MPS block-subspace TSVD | native RAPIDS RAFT TSVD when compiled | GPU requests do not route through Python or silently fall back to CPU. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by t-SNE landmarking. |
| `select_landmarks()` | native selection | shared selection | shared selection | Selection is independent of the embedding method and can be reused. |
| `fit_landmark_model()` | ordinary CPU UMAP/t-SNE reference fit | ordinary Metal UMAP/t-SNE reference fit | ordinary CUDA UMAP/t-SNE reference fit | UMAP graph mode and optimizer parameters are preserved explicitly. |
| `precompute_query_knn()` | native HNSW reference-query search | native exact/recall-tuned IVF-Flat reference-query search | native exact/IVF-Flat reference-query search | Searches only the fixed reference; CUDA output remains device resident. |
| `project_landmark_model()` | native projection/transform/refinement | native Metal projection/transform/refinement | native CUDA resident projection/transform/refinement | Projects held-out or genuinely new observations while reference coordinates remain fixed. |
| `landmark_umap()` | one-call landmark embed/project/refine | one-call Metal path | one-call CUDA path | Convenience wrapper; landmarking remains an explicit approximation. |
| `landmark_tsne()` | one-call landmark embed plus transform | one-call Metal path | one-call CUDA path | Convenience wrapper; projection quality is tracked separately. |
| `knn_graph()` | native C++ graph construction | Metal KNN followed by native CPU graph construction | CUDA KNN followed by native CPU graph construction | A GPU label applies to neighbor search only; graph conversion is never reported as GPU work. |
| `graph_cluster()` | native C++ Louvain, Leiden, and Pons-Latapy Walktrap | native Metal Louvain and Leiden | native CUDA Louvain and Leiden | GPU local moving/refinement uses float32 CSR; graph compaction/coarsening is package-owned C++. Walktrap is CPU-only. Unsupported requests fail without fallback. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

## Distance Metrics

| Metric | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `euclidean` | native HNSW | native exact/IVF-Flat | cuVS exact/IVF-Flat; optional FAISS GPU exact | Validated default for UMAP/t-SNE. |
| `cosine` | row normalization plus native HNSW | row normalization plus native exact/IVF-Flat | row normalization plus native cuVS | Candidate selection is approximate; returned distances use the transformed full-dimensional vectors. |
| `correlation` | row centering/normalization plus native HNSW | row centering/normalization plus native exact/IVF-Flat | row centering/normalization plus native cuVS | Correlation is represented by Euclidean distance on centered unit rows. |
| `inner_product` | not supported | not supported | native CUDA route | Explicit unsupported requests fail; they do not fall back to another backend. |

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

CPU paths are native C++ and use `n.cores` where the operation is safe to
parallelize. Current CPU priorities are:

- reuse KNN graphs instead of recomputing them;
- keep graph data in compact integer/float arrays;
- use CSR graph storage for UMAP;
- keep t-SNE attractive affinities sparse;
- avoid unnecessary copies between R matrices and C++ buffers.

## Metal

Metal is implemented with Objective-C++ and Metal kernels. Public Metal and CUDA
UMAP/t-SNE paths do not call Python, Torch, MLX, or `reticulate`.

The build-supported target is Apple Silicon with macOS 14 or newer and a full
Xcode 15 or newer toolchain. Full Metal performance benchmarking has been
performed only on an Apple M3 MacBook Pro with macOS 14.5, Xcode 16.2, and
8 GB unified memory. Other Apple Silicon generations are compatibility targets:
they require a strict real-device smoke/correctness artifact before being
called tested, and they do not inherit the M3 performance results. Intel Macs
are not a supported Metal target. The CPU backend remains available on Intel
macOS.

UMAP initialization/optimization, t-SNE FFT-grid optimization, and the
current Metal transformation/refinement routes are two-dimensional. The
capability matrix labels mixed operations explicitly: Metal KNN followed by
CPU graph assembly is not described as a fully Metal graph workflow, and CPU
quality scoring after layout transfer is not reported as Metal computation.

Metal one-call KNN uses exact search below the internal size threshold and
IVF-Flat above it. Large, high-dimensional IVF searches use a 128-dimensional
signed float32 projection, four native centroid passes, a four-SIMD-group
projected shortlist, and full-dimensional exact reranking. The shortlist grows
internally from 288 to 384 or 512 candidates when needed. Four deterministic
pilot strata select shortlist size and `nprobe`; direct Metal reranking is the
safety fallback. Failure to reach the requested pilot recall is reported
instead of silently switching to CPU.

Exact Metal search caches rows in threadgroup memory through 1,024 dimensions.
For small data with up to 16,384 dimensions, a mathematically identical
global-memory kernel avoids that fixed cache; Metal IVF remains limited to
1,024 input dimensions. These routes fail explicitly outside their supported
domain.

The validated UMAP Metal path is `atomic_inplace`; other slower or distorted
Metal UMAP optimizer experiments were removed from the public API.

The validated t-SNE Metal path is the package-native FFT-grid path with PCA
initialization. Its PCA initialization uses a single resident float32 workspace,
MPS matrix products, a small float32 eigensolve, and one final Metal score
projection. Standalone MPSGraph FFT diagnostic helpers were removed after MNIST
tests did not justify exposing them as package features.

## CUDA

CUDA support is optional at build time. The package can use:

- RAPIDS cuVS C API for brute-force exact KNN and IVF-Flat;
- optional FAISS GPU `bfKnn` when explicitly enabled at build time;
- native CUDA UMAP kernels;
- native CUDA FFT-grid t-SNE kernels with cuFFT.

cuVS, CUDA, and cuFFT are not vendored into the package. They must be installed
on the CUDA machine and matched to the driver/toolkit stack. FAISS GPU is
optional. If CUDA is not available, explicit CUDA requests fail clearly.

CUDA evidence is device-specific. A configured architecture or embedded
`sm_*`/PTX target establishes build-level compatibility only. Strict hardware
smoke/correctness testing requires execution on the named GPU, and full
performance validation additionally requires the release-locked scientific
benchmark. Historical broad benchmarking used an NVIDIA L40S (`sm_89`), while
the current numerical diagnostic has also executed on an RTX 5060 Ti
(`sm_120`); neither observation implies performance on an untested CUDA device.

## External Libraries

`fastEmbedR` does not vendor the full FAISS or RAPIDS libraries. It contains a
small FAISS-derived HNSW implementation and a native Metal IVF design informed
by FAISS and Faiss-mlx; their permissive licenses and pinned source commits are
installed under `inst/LICENSES/`. Installed FAISS GPU and cuVS libraries are
linked directly by optional CUDA builds; fastEmbedR does not vendor them.
