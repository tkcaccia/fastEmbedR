# Backend Capabilities

This page states what each backend does and what it does not do. The central
rule is simple: if a function is requested with `backend = "metal"` or
`backend = "cuda"`, it must run a real native GPU path or fail clearly.

Verify an installed accelerator by running a small calculation and inspecting
the backend recorded in its result:

```r
set.seed(1)
x <- matrix(runif(2048 * 16), nrow = 2048)
fit <- fastEmbedR::umap(x, backend = "cuda", n_neighbors = 15)
stopifnot(identical(fit$parameters$backend, "cuda"))
```

Public backend arguments accept only `"cpu"`, `"cuda"`, or `"metal"`. An
unavailable explicit accelerator request raises an error.

## Capability Matrix

| Function | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `precompute_knn()` / internal one-call KNN | native float32 exact below 5,000 rows; faissR HNSW otherwise | native exact or recall-tuned IVF-Flat | cuVS brute-force exact or IVF-Flat | KNN selection remains internal; the public function exposes only `k`, metric, backend, and CPU thread count. CUDA results stay device-resident. |
| `umap_init()` | native sparse graph initialization | native Metal initialization from prepared graph state | native CUDA initialization when compiled | Returns reusable graph and initial coordinates; a raw CUDA diagnostic call may materialize KNN on the host, whereas ordinary one-call CUDA UMAP remains resident. |
| `umap_knn()` | native C++ CSR graph and optimizer | 2D: ordered CPU half followed by native Metal `atomic_inplace` half; 3D: native Metal optimizer | native CUDA pure-atomic optimizer; experimental resident NCCL route on two to four GPUs | The 2D Metal CPU and GPU epoch counts are returned in metadata. Coordinated CUDA builds the CSR graph in host RAM and keeps layouts and updates on GPUs. Small two-A10G 2D/3D device checks passed; performance remains unqualified. |
| `umap()` | native exact/faissR HNSW, then `umap_knn()` | native exact/IVF-Flat KNN, then staged 2D or native 3D Metal UMAP | native cuVS device KNN, then native CUDA UMAP; experimental resident NCCL fit with explicit devices | Coordinated CUDA materializes native KNN in host RAM; no SSD graph or optimizer state is used. Other CUDA KNN fits stay on device. Metal IVF exact-reranks candidates in the original dimensions. |
| `tsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT; experimental resident NCCL route on two to four GPUs | Use `Y_init` or `init_data` for explicit PCA initialization. Coordinated CUDA builds sparse affinities in host RAM and keeps optimizer state on the GPUs. The new resident route needs two-GPU device validation. |
| `tsne()` | native exact/faissR HNSW, then `tsne_knn()` | native exact/IVF-Flat, then Metal t-SNE | native cuVS device KNN, then CUDA t-SNE; experimental resident NCCL fit with explicit devices | CPU uses exact search below 5,000 rows. Coordinated CUDA materializes native KNN in host RAM; no SSD graph or optimizer state is used. The package does not call Python openTSNE in public functions. |
| `pca()` / t-SNE PCA init | float32 blocked rSVD with OpenBLAS or native products | native float32 MPS block-subspace rSVD; resident landmark score projection uses native Metal multiplication | package-native rSVD; optional RAPIDS RAFT TSVD; experimental resident two to four GPU covariance PCA | Resident `landmarks` fits selected rows and projects all input rows. CPU and single-device CUDA projection use CPU; multi-GPU CUDA projects on the requested devices. Resident multi-GPU PCA accepts a double matrix with at most 2,048 columns. GPUs split covariance and score projection; CPU solves the small eigensystem. Small two-A10G PCA and landmark projections passed; Metal resident projection still needs device qualification. GPU requests never silently fall back to CPU. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by t-SNE landmarking. |
| `select_landmarks()` | native selection | shared selection | shared selection | Selection is independent of the embedding method and can be reused. |
| `precompute_query_knn()` | native exact below 5,000 reference rows; faissR HNSW otherwise | native exact/recall-tuned IVF-Flat reference-query search | native exact/IVF-Flat reference-query search | Searches only the fixed reference; CUDA output remains device resident. |
| `project_landmark_model()` | native projection/transform/refinement | native Metal projection/transform/refinement | native CUDA resident projection/transform/refinement | Projects held-out or genuinely new observations while reference coordinates remain fixed. |
| `umap(..., landmarks = ...)` | landmark embed/project/refine | native Metal path | native CUDA path | Landmarking remains an explicit approximation. |
| `tsne(..., landmarks = ...)` | landmark embed plus transform | native Metal path | native CUDA path | Projection quality is tracked separately. |
| `knn_graph()` | native C++ graph construction | Metal KNN followed by native CPU graph construction | CUDA KNN followed by native CPU graph construction | A GPU label applies to neighbor search only; graph conversion is never reported as GPU work. |
| `graph_cluster(..., landmarks = ...)` | resident reference KNN, native graph fit, all-row label vote | Metal reference KNN and native graph fit; CPU assembly and vote | CUDA reference KNN and native graph fit; CPU assembly and vote | Explicit landmark approximation with fixed reference labels and all-row membership/confidence. CPU `parallel.mode = "fit"` supports one shared multicore fit. GPU routes still need device qualification; no SSD files are used. |
| `graph_cluster()` | native C++ Louvain, Leiden, and Pons-Latapy Walktrap; experimental coordinated CPU fit threads | native Metal Louvain and Leiden | native CUDA Louvain and Leiden; experimental coordinated two-to-four-GPU fit | CPU `parallel.mode = "fit"` shares one partition across local-move workers; its coarsening and Leiden refinement remain serial. Landmark clustering uses the same public call with `landmarks = ...`. GPU local moving/refinement uses float32 CSR; graph compaction/coarsening is package-owned C++. Multiple GPUs split graph rows and synchronize one global partition through host RAM after each color phase. The landmark graph fit can use these devices while file-backed query KNN and label voting remain streamed on CPU. Consensus and graph weighting use coordinated fits with explicit host stages. Small two-A10G Louvain/Leiden checks and a 300-row MNIST quality check passed; scaling remains unqualified. Walktrap is CPU-only. Unsupported requests fail without fallback. |
| `umap(file_graph, ...)` | file-backed full-graph 2D/3D | file-backed full-graph 2D/3D | file-backed full-graph 2D/3D | CUDA supports explicit one to four GPUs; large-data Metal quality and speed remain under qualification. |
| `tsne(affinity_graph, ...)` | file-backed full-graph 2D/3D | experimental mapped or bounded-stream graph 2D/3D | experimental file-backed full-graph 2D/3D | Metal streams graph chunks when selected or when auto exceeds one chunk; its layout and optimizer state remain resident. FashionMNIST 5,000 and 70,000-row 2D comparisons and checkpoint resume passed; broader quality and scale remain unqualified. CUDA accepts one to four GPUs in 2D/3D; a 17-row two-A10G 3D parity/resume check passed outside the UCT HPC, while larger-data quality and scale remain unqualified. |
| `graph_cluster(file_graph, ...)` | file-backed multilevel Louvain and Leiden | file-backed multilevel Louvain and Leiden with native Metal local moves and Leiden refinement | experimental file-backed Louvain and Leiden with native CUDA local moves and refinement | Metal and CUDA stream bounded CSR chunks from SSD; CPU handles contraction, modularity, and Leiden connectivity audits. Separate multi-GPU cuGraph/Dask full-graph clustering loads the graph into GPU memory and is not SSD streaming. CUDA streaming clustering needs device qualification. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

Here, "exact" describes nearest-neighbor search, not t-SNE repulsion.
All 2D and 3D t-SNE fits use FFT repulsion.
Resident full t-SNE defaults to compact affinity support. Set
`affinity_k = ceiling(3 * perplexity)` for conventional-width support on the requested
backend. Precomputed KNN routes use `n_neighbors` for the same control.
Resident landmark t-SNE applies this width to its reference fit. File-backed
routes do not accept the explicit width control.

Resident landmark `tsne()` and `umap()` currently use one CUDA device and
reject an explicit multi-device `devices` request. The resident multi-GPU
full-data routes and the experimental file-backed, sharded landmark
projection are separate implementations. Coordinated resident landmark
reference fitting and query projection remain to be implemented.

## Distance Metrics

| Metric | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `euclidean` | native exact/faissR HNSW | native exact/IVF-Flat | cuVS exact/IVF-Flat | Exact CPU search is selected below 5,000 rows. |
| `cosine` | native exact/HNSW | native exact/IVF-Flat | native cuVS | Exact CPU search is selected below 5,000 rows. |
| `correlation` | native exact/HNSW | native exact/IVF-Flat | native cuVS | Correlation uses centered, normalized vectors. |

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

Resident CPU UMAP accepts `n.cores` from one to 32 for 2D and 3D output.
Native stages use a workload-size cap, so a small graph may use fewer
workers than requested. On graphs with at least 10,000 rows, the 2D and
3D optimizers use concurrent atomic coordinate updates. Their exact
coordinates can vary with scheduling even at a fixed seed; assess
quality over repeated runs. The `n.cores_effective` field records the
configured maximum, not a per-stage thread measurement.

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

Resident UMAP and t-SNE fits and their landmark routes support 2D and 3D
Metal output. File-backed full-graph Metal UMAP also supports 2D and 3D.
The experimental Metal t-SNE file-backed route supports 2D and 3D. Mapped mode
exposes complete read-only CSR files as Metal buffers. Streamed mode maps
the files on the host and uploads bounded graph chunks to Metal; its
layout and optimizer state remain in shared memory. A 70,000-row Mac
mapped/streamed comparison and small checkpoint-resume tests passed;
the 2D route also passed paired 5,000-row quality checks on three seeds;
large-data quality and speed remain unqualified. The capability
matrix labels mixed operations explicitly: Metal KNN followed by
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

Resident 2D Metal UMAP uses an ordered, single-thread CPU warm stage followed
by the native `atomic_inplace` Metal optimizer. The public result records both
stage lengths. Resident 3D Metal UMAP uses the native optimizer throughout.
File-backed full-graph UMAP follows a separate Metal route and needs its own
large-data quality qualification.

The validated t-SNE Metal path is the package-native FFT-grid path with PCA
initialization. Its PCA initialization uses a single resident float32 workspace,
MPS matrix products, a small float32 eigensolve, and one final Metal score
projection. Standalone MPSGraph FFT diagnostic helpers were removed after MNIST
tests did not justify exposing them as package features.

## CUDA

CUDA support is optional at build time. The package can use:

- RAPIDS cuVS C API for matrix-input brute-force exact KNN and IVF-Flat;
- native CUDA UMAP kernels;
- native CUDA FFT-grid t-SNE kernels with cuFFT.

CUDA and cuFFT are not vendored into the package. cuVS is an additional
dependency only for CUDA KNN from matrix input. These components must be
installed on the CUDA machine and matched to the driver/toolkit stack. If CUDA
is not available, explicit CUDA requests fail clearly.

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
installed under `inst/LICENSES/`. Installed cuVS libraries are linked directly
by optional CUDA builds; fastEmbedR does not vendor them.
