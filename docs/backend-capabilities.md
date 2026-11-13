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
| `precompute_knn()` / one-call embedding KNN | faissR Flat exact below 5,000 rows; faissR HNSW otherwise | faissR native Metal exact or IVF-Flat | `precompute_knn()` retains native resident CUDA output; one-call `tsne()` and `umap()` use faissR FAISS GPU Flat or IVF-Flat with host output | These CUDA routes have different output residency. Multi-GPU one-call KNN uses experimental FAISS replicas and requires recall and speed audits. |
| `umap_init()` | native sparse graph initialization | native Metal initialization from prepared graph state | native CUDA initialization when compiled | Returns reusable graph and initial coordinates; a raw CUDA diagnostic call may materialize KNN on the host, whereas ordinary one-call CUDA UMAP remains resident. |
| `umap_knn()` | native C++ CSR graph and optimizer | 2D: ordered CPU half followed by native Metal `atomic_inplace` half; 3D: native Metal optimizer | native CUDA pure-atomic optimizer; experimental resident NCCL multi-GPU route | The 2D Metal CPU and GPU epoch counts are returned in metadata. Coordinated CUDA builds the CSR graph in host RAM and keeps layouts and updates on GPUs. Small 5-8-device routing checks passed; large-data quality and speed remain unqualified. |
| `umap()` | faissR Flat/HNSW, then `umap_knn()` | faissR native Metal KNN, then staged 2D or native 3D Metal UMAP | faissR FAISS GPU KNN with host output, then native CUDA UMAP | Multiple-device one-call matrix fits use experimental FAISS GPU replicas. Metal IVF exact-reranks candidates in the original dimensions. |
| `tsne_knn()` | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT; experimental resident NCCL multi-GPU route | Use `Y_init` or `init_data` for explicit PCA initialization. Coordinated CUDA builds sparse affinities in host RAM and keeps optimizer state on the GPUs. Small 5-8-device routing checks passed; larger quality, memory, and speed remain unqualified. |
| `tsne()` | faissR Flat/HNSW, then `tsne_knn()` | faissR native Metal KNN, then Metal t-SNE | faissR FAISS GPU KNN with host output, then CUDA t-SNE | CPU uses exact search below 5,000 rows. Multiple-device one-call matrix fits use experimental FAISS GPU replicas. The package does not call Python openTSNE in public functions. |
| `pca()` / t-SNE PCA init | float32 blocked rSVD with OpenBLAS or native products | native float32 MPS block-subspace rSVD; resident landmark score projection uses native Metal multiplication | package-native rSVD; optional RAPIDS RAFT TSVD; experimental resident multi-GPU covariance PCA | Resident and file-backed `landmarks` fit selected rows and project all input rows. File-backed PCA accepts a count, fraction or `TRUE` through `pca()` with bounded score access; explicit row indices and partial-landmark resume are pending. CPU and single-device CUDA projection use CPU; multi-GPU CUDA projects on requested devices. Resident multi-GPU PCA accepts a double matrix with at most 2,048 columns. GPUs split covariance and score projection; CPU solves the small eigensystem. Small 5-8-device routing checks passed; large-data quality and speed remain unqualified. Metal landmark projection matched CPU on a 100-row device check (maximum difference `5.23e-7`). GPU requests never silently fall back to CPU. |
| `transform_tsne()` | native fixed-reference transform | native Metal projection/transform kernels where available | native CUDA projection/transform kernels where built | Used by t-SNE landmarking. |
| `select_landmarks()` | native selection | shared selection | shared selection | Selection is independent of the embedding method and can be reused. |
| `precompute_query_knn()` | faissR Flat below 5,000 reference rows; faissR HNSW otherwise | native exact/recall-tuned IVF-Flat reference-query search | native exact/IVF-Flat reference-query search | Searches only the fixed reference; CUDA output remains device resident. |
| `project_landmark_model()` | native projection/transform/refinement | native Metal projection/transform/refinement | native CUDA resident projection/transform/refinement | Projects held-out or genuinely new observations while reference coordinates remain fixed. |
| `umap(..., landmarks = ...)` | landmark embed/project/refine | native Metal path | native CUDA path | Landmarking remains an explicit approximation. |
| `tsne(..., landmarks = ...)` | landmark embed plus transform | native Metal path | native CUDA path | Projection quality is tracked separately. |
| `knn_graph()` | native C++ graph construction | Metal KNN followed by native CPU graph construction | CUDA KNN followed by native CPU graph construction | A GPU label applies to neighbor search only; graph conversion is never reported as GPU work. |
| `graph_cluster(..., landmarks = ...)` | resident reference KNN, native graph fit, all-row label vote | Metal reference KNN and native graph fit; CPU assembly and vote | CUDA reference KNN and native graph fit; CPU assembly and vote | Explicit landmark approximation with fixed reference labels and all-row membership/confidence. CPU `parallel.mode = "fit"` supports one shared multicore fit. GPU routes still need device qualification; no SSD files are used. |
| `graph_cluster()` | native C++ Louvain and Leiden; experimental coordinated CPU fit threads | native Metal Louvain and Leiden | native CUDA Louvain and Leiden; experimental coordinated multi-GPU fit | CPU `parallel.mode = "fit"` shares one partition across local-move workers; its coarsening and Leiden refinement remain serial. Landmark clustering uses the same public call with `landmarks = ...`. GPUs split graph rows and synchronize one global partition through host RAM after each color phase. The landmark graph fit can use these devices while file-backed query KNN and label voting remain streamed on CPU. Small 5-8-device Louvain routing checks and two-A10G Louvain/Leiden checks passed; large-data quality and scaling remain unqualified. Unsupported requests fail without fallback. |
| `umap(file_graph, ...)` | file-backed full-graph 2D/3D | file-backed full-graph 2D/3D | file-backed full-graph 2D/3D | CUDA accepts explicit device lists; 5-8-device file-backed execution remains unqualified. Large-data Metal quality and speed remain under qualification. |
| `tsne(affinity_graph, ...)` | file-backed full-graph 2D/3D | experimental mapped or bounded-stream graph 2D/3D | experimental file-backed full-graph 2D/3D | Metal streams graph chunks when selected or when auto exceeds one chunk; its layout and optimizer state remain resident. FashionMNIST 5,000 and 70,000-row 2D comparisons and checkpoint resume passed; broader quality and scale remain unqualified. CUDA accepts explicit device lists in 2D/3D; a 17-row two-A10G 3D parity/resume check passed outside the UCT HPC. A 5-8-device file-backed fit remains unqualified. |
| `graph_cluster(file_graph, ...)` | file-backed multilevel Louvain and Leiden | file-backed multilevel Louvain and Leiden with native Metal local moves and Leiden refinement | experimental file-backed Louvain and Leiden with native CUDA local moves and refinement | Metal and CUDA stream bounded CSR chunks from SSD; CPU handles contraction, modularity, and Leiden connectivity audits. Native resident multi-GPU clustering keeps graph shards on the selected GPUs; the separate native bounded-upload route handles graphs beyond combined VRAM. Large-data CUDA quality and scaling remain under qualification. |
| `evaluate_embedding()` | native/R quality metrics | CPU metrics after final layout transfer | CPU metrics after final layout transfer | Metrics are not labelled as GPU work. |

Here, "exact" describes nearest-neighbor search, not t-SNE repulsion.
All 2D and 3D t-SNE fits use FFT repulsion.
Resident full t-SNE defaults to compact affinity support. Set
`affinity_k = ceiling(3 * perplexity)` for conventional-width support on the requested
backend. Precomputed KNN routes use `n_neighbors` for the same control.
Resident landmark t-SNE applies this width to its reference fit. File-backed
routes do not accept the explicit width control.

Resident partial-landmark `tsne()` and `umap()` accept multiple CUDA
devices for a coordinated reference fit and sharded query projection.
Small eight-A100 routing checks passed for both methods, but large
real-data quality and memory remain unqualified. Neither route has
established useful scaling or the
ability to exceed combined GPU memory. The file-backed landmark routes
are separate implementations and must be tested outside UCT HPC.

## Distance Metrics

| Metric | CPU | Metal | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `euclidean` | faissR Flat/HNSW | native exact/IVF-Flat | cuVS exact/IVF-Flat | Exact CPU search is selected below 5,000 rows. |
| `cosine` | faissR Flat/HNSW | native exact/IVF-Flat | native cuVS | Exact CPU search is selected below 5,000 rows. |
| `correlation` | faissR Flat/HNSW | native exact/IVF-Flat | native cuVS | Correlation uses centered, normalized vectors. |

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

Resident CPU 2D/3D fuzzy and binary UMAP chooses int32 or uint64 CSR
offsets from a checked edge bound. The wide path passed small edge/step
parity and synthetic offsets above 2^31; the forced-wide 1M/10M Deep1B
development fits exercised 2D fuzzy UMAP.
The 10M graph stayed below 2^31 edges; a real overflow-sized graph,
full 50M output, measured peak RAM and quality remain unqualified.
CPU and Metal wide UMAP have passed small graph checks. CUDA wide graph
construction and bounded edge-tile code exist, but the public resident
route still guards unsupported wide combinations. They need a real-device
case and measured large-memory admission before a 50M fit.

Resident Metal and CUDA Louvain/Leiden select int32 or uint64 CSR row
offsets from the checked edge count. The GPU move and refinement kernels
read bounded edge tiles and retain one decision per complete row, even
when a row spans tiles. A forced-wide 71-vertex Metal case passed
partition and modularity parity for both methods; the CUDA source
compiled in an isolated HPC compute job but still needs a real-device
case. Large-graph memory, speed, and quality are unqualified.

## Metal

Metal is implemented with Objective-C++ and Metal kernels. Public Metal and CUDA
UMAP/t-SNE paths do not call Python, Torch, MLX, or `reticulate`.

The build-supported target is Apple Silicon with macOS 14 or newer and a full
Xcode 15 or newer toolchain. The older full Metal performance cohort used an
Apple M3 MacBook Pro with macOS 14.5, Xcode 16.2, and 8 GB unified memory.
Newer M3 Ultra development fits are described below; they do not turn the
older cohort into an M3 Ultra performance benchmark. Other Apple Silicon
generations need their own real-device checks and do not inherit either
machine's results. Intel Macs
are not a supported Metal target. The CPU backend remains available on Intel
macOS.

Resident UMAP and t-SNE fits and their landmark routes support 2D and 3D
Metal output. File-backed full-graph Metal UMAP also supports 2D and 3D.
The experimental Metal t-SNE file-backed route supports 2D and 3D. Mapped mode
exposes complete read-only CSR files as Metal buffers. Streamed mode maps
the files on the host and uploads bounded graph chunks to Metal; its
layout, optimizer state, and FFT workspace remain in memory. This is
bounded graph upload, not a fully streamed fit. Large file-backed Metal
one-call UMAP and t-SNE use CPU HNSW for KNN preparation above 5,000 rows;
the public result records the actual backend, method, and resource estimates
by stage. File-backed UMAP quality must be tested separately from the
resident 2D UMAP repair. A 70,000-row Mac
mapped/streamed comparison and small checkpoint-resume tests passed;
the 2D route also passed paired 5,000-row quality checks on three seeds;
large-data quality and speed remain unqualified. The capability
matrix labels mixed operations explicitly: Metal KNN followed by
CPU graph assembly is not described as a fully Metal graph workflow, and CPU
quality scoring after layout transfer is not reported as Metal computation.

Real Deep1B 25M and 100M resident PCA fits completed on one M3 Ultra
Mac, but this does not qualify 50M t-SNE or UMAP. Metal resident UMAP
has passed small 2D/3D fuzzy and binary wide graph traversal checks,
including rows crossing bounded edge tiles. A 50M traversal and its
memory budget remain unqualified; t-SNE also needs a
measured affinity, FFT and optimizer memory budget. The file-backed
Metal routes require their own large one-call, quality and recovery
checks. Resident wide UMAP has no optimizer checkpoint or mid-fit resume;
the separate file-backed route supports validated recovery. No Metal
multi-GPU path is claimed on this single-GPU Mac.

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
File-backed full-graph UMAP follows a separate Metal snapshot route with a
25-epoch ordered CPU tail in 2D or 50 epochs in 3D. Both stage times are
reported. Fixed-graph 5,000-row development checks passed the paired
silhouette margin; large-data quality still needs qualification.

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
