# Implementation

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
**Implementation** |
[Performance Engineering](backend-performance-engineering.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

`fastEmbedR` implements two nonlinear embedding families: UMAP and an
interpolation-based t-SNE. UMAP follows the fuzzy simplicial-set graph formulation
introduced by McInnes and colleagues [7,13]. The t-SNE path follows the
probabilistic neighbor-embedding objective of van der Maaten and Hinton [1],
with modern interpolation-based optimization ideas from FIt-SNE and openTSNE [3-4].
The package is intentionally KNN-first. fastEmbedR implements its CPU HNSW and
Apple Metal exact/IVF-Flat one-call KNN paths natively. CUDA builds link to the
Apache-2.0 RAPIDS cuVS C API for exact and IVF-Flat search; FAISS GPU is an
optional exact-search provider. They do not call another R package, Python, or
`reticulate`. Graph/affinity construction,
initialization, stochastic optimization, native fixed-reference transforms,
backend reporting, and quality metrics remain inside fastEmbedR.

The public package surface is deliberately small:

- `precompute_knn()` exposes the same native backend policy used by the
  one-call functions, while keeping algorithm and recall tuning internal.
- `tsne_knn()` and `umap_knn()` consume a supplied KNN object.
- `tsne()` and `umap()` select native CPU/Metal KNN or direct cuVS CUDA KNN and
  then call the corresponding KNN entry point.
- `pca()` computes backend-native truncated PCA scores and loadings. CPU uses
  fastEmbedR's native blocked RSVD implementation, Metal uses a resident
  float32 MPS TSVD path, and CUDA uses native RAPIDS RAFT TSVD. Float32 CUDA
  input is passed to the native fit without an intermediate R double matrix;
  unavailable RAFT support is an explicit error rather than a CPU fallback.
- `knn_graph()` builds one compact undirected graph from data, an embedding,
  or supplied neighbors; `graph_cluster()` applies native Louvain, Leiden,
  or Walktrap community detection.
- `backend` is limited to `"cpu"`, `"metal"`, and `"cuda"`.
- GPU requests fail clearly if native GPU code is unavailable. CPU fallback is
  never reported as Metal or CUDA work.

## Compilation And Numerical Contract

The portable numerical core requests C++17 and adds only `-pthread`; it
inherits the compiler command, optimization level, ABI, and linker flags from
the active R installation. This is part of the algorithmic contract, not only
an installation detail. HNSW construction, sparse graph creation, and
stochastic embedding contain floating-point comparisons whose tie decisions
can change under unsafe reassociation. fastEmbedR therefore does not impose
global `-ffast-math`, `-march=native`, or `-O3` flags.

On macOS, `configure` adds the Apple Foundation, Metal, MPS, and MPSGraph
frameworks and compiles the Objective-C++ bridge with the Xcode SDK. Metal
shader source is compiled by the runtime for the active GPU. The native Metal
KNN shader enables Apple's float32 fast arithmetic locally; this is not
propagated to the CPU core or to UMAP/t-SNE host code.

CUDA translation units are compiled by NVCC as C++17 with extended lambdas,
relaxed `constexpr`, and position-independent host code. The deployment
architectures are explicit through `FASTEMBEDR_CUDA_ARCH`. `CUDAHOSTCXX`
selects an R-ABI-compatible host compiler when CUDA and R come from different
toolchain prefixes. cuVS, optional FAISS GPU, RAFT, and their CUDA dependencies
must be built for the same devices; adding an architecture to fastEmbedR cannot
add missing kernels to a linked library.

The complete commands, architecture examples, diagnostics, and failure modes
are documented in
[Installation And Native Compiler Configuration](installation-backends.md).
The benchmark reproducibility bundle records the resolved `R CMD config`
values, generated `src/Makevars`, compiler versions, Xcode/Metal or
CUDA/NVCC versions, and relevant environment flags.

## Native Implementation Validation

Because fastEmbedR implements the embedding path natively rather than calling
Python openTSNE, the package includes a small reference-validation workflow.
The script
[`tools/validate_reference_implementations.R`](https://github.com/tkcaccia/fastEmbedR-benchmark/blob/main/tools/validate_reference_implementations.R)
uses exact KNN on iris, fixes the random seed, and compares package-native
outputs with established R references.

For t-SNE, the validation compares `tsne_knn()` with
`Rtsne::Rtsne_neighbors()` from the same exact KNN matrix and PCA
initialization. It records trustworthiness, nearest-neighbor preservation,
embedding-space KNN label accuracy, the final KL/cost value where exposed, and
Procrustes-aligned similarity between the two embeddings [1-4,11]. On a small
dataset the script uses the exact negative-gradient diagnostic path rather
than the large-data FFT-grid approximation, because this isolates objective
correctness from interpolation-grid engineering.

For UMAP, the validation compares `umap_knn(..., graph_mode = "fuzzy")` with
`uwot::umap()` and compares the sparse graph returned by
`prepare_umap_knn()` with `uwot::similarity_graph()` [7,10,13]. The graph check
reports edge overlap and Spearman correlation of common-edge weights. These
checks are intended to show behavioural agreement, not bitwise identity:
stochastic optimizers, spectral initialization details, and parallel floating
point reductions can legitimately rotate, reflect, scale, or slightly deform
the final layout.

## Nearest-Neighbor Layer

`fastEmbedR` exports a focused precomputation boundary rather than a general
nearest-neighbor algorithm menu. `precompute_knn()` exposes `k`, metric,
backend, and CPU thread count; it applies the same internally selected search
policy as the one-call embedding functions. The KNN-input functions also
accept a plain host list of indices and distances from another implementation.

The CPU implementation distils the HNSW organization in FAISS 1.14.3 [8]:
exponentially sampled hierarchy levels, greedy descent through upper layers,
bounded `efConstruction` expansion at each insertion layer, diversity-aware
neighbor pruning, reciprocal graph links, and independent parallel queries.
All vectors, graph distances, and search buffers are float32; indices are
int32. A large/high-dimensional target-0.99 policy uses `M = 10`,
`efConstruction = 40`, and `efSearch = 40`, selected only after comparison
against the original FAISS HNSW output. Other shapes use a more conservative
graph. The package records the selected values and does not claim bitwise
identity with FAISS.

Construction uses one persistent worker team, reusable visit tables and
bounded heaps, early-exit squared-distance comparisons, and parallel
reciprocal-row updates. Temporary construction distances are released before
query. These changes preserve the selected graph and query output: exact KNN
indices and distances matched the pre-optimization implementation on
MNIST70k, Fashion-MNIST70k, USPS, and MetRef. On the validated four-thread
Linux environment, MNIST70k construction decreased from 23.730 to 17.345
seconds and complete KNN time from 27.653 to 21.288 seconds. A separate
`-O3 -march=native` build was slower and was rejected, which is why compiler
tuning is not presented as an algorithmic improvement.

The Metal implementation has two routes. Exact search assigns one SIMD group
to each candidate distance and merges per-group top-k lists on device. IVF-Flat
first forms a deterministic 128-dimensional signed float32 projection for
coarse routing. Four native centroid-assignment/update passes then pack the
inverted lists. For large, high-dimensional inputs, a four-SIMD-group kernel
scans selected lists in projected space and retains an internal shortlist. It
starts at 288 candidates and can expand to 384 or 512 candidates. A second
kernel exact-reranks only that shortlist using the original full-dimensional
vectors, in bounded query batches. Four deterministic 64-row pilot strata
spread across the dataset jointly select shortlist size and `nprobe`. If no
shortlist reaches the requested recall tier plus a generalization margin, the
same Metal backend uses direct exact reranking of selected lists. Candidate
routing is approximate; every returned neighbor and distance is based on
full-dimensional exact reranking. This fused list-scan/top-k organization was
informed by FAISS [8] and MLXPorts/Faiss-mlx. Their exact commits and
permissive licenses are retained under `inst/LICENSES/`.

For reproducibility, the one-call API fixes only the requested KNN device class
and a small/large data policy:

- CPU one-call embeddings use native HNSW. Metal uses native exact KNN below
  4,096 observations and recall-tuned IVF-Flat for larger inputs.
- CUDA one-call embeddings use cuVS brute force below 100,000 samples and
  recall-tuned cuVS IVF-Flat at or above 100,000 samples. An explicitly enabled
  FAISS GPU build may provide exact search. Exact float32 input is uploaded in
  its existing R column-major layout, removing a full host transpose and its
  temporary buffer without changing distances or neighbors. IVF starts from a
  deterministic shape rule, compares evenly spaced pilot queries against a
  cuVS exact oracle, and expands `nprobe` until it reaches the requested recall
  tier with a small safety margin. It records pilot recall, `nlist`, `nprobe`,
  and the number of tuning attempts.
- cuVS, and optional FAISS GPU, write int64 row-major search output on the
  device. One package CUDA
  kernel removes self-neighbors, converts indices to int32/one-based form,
  transforms squared distances, and packs the result directly into the
  column-major device layout consumed by UMAP and t-SNE. No KNN matrix is
  materialized in R during a one-call CUDA embedding.
- KNN-input functions accept the index and distance matrices as already
  measured data and do not repeat neighbor search.

This separation makes benchmark timing interpretable: KNN time, affinity/graph
construction time, embedding time, and projection/transform time can be
reported separately.

## KNN Graphs And Community Detection

`knn_graph()` deliberately reuses the package's existing nearest-neighbor
boundary. It accepts raw data, a `fastEmbedR_embedding`, or supplied neighbor
indices and distances; it does not expose another KNN algorithm selector.
Graph construction is package-native C++ and produces one compact undirected
edge list. The supported weights are Jaccard shared-neighbor similarity on
observed KNN edges, inverse distance, and binary adjacency. Reciprocal-edge
filtering and a final weight threshold are optional. Directed duplicates are
collapsed once, so clustering does not repeat neighbor search or retain a
dense adjacency matrix.

`graph_cluster()` provides three package-native algorithms:

- multilevel Louvain local moving and graph aggregation [14];
- Leiden local moving, within-community refinement, and aggregation [15];
- the Pons-Latapy Walktrap random-walk distance and adjacent-community
  agglomeration [16].

The Leiden phase organization is informed by the MIT-licensed NetworKit
implementation [17]. The accelerator design is also informed by the parallel
graph-processing principles used by RAPIDS cuGraph, but no cuGraph source is
copied and no cuGraph library, Python module, or runtime symbol is linked.
fastEmbedR owns its graph representation and implementation.

CPU clustering uses the package's double-precision sequential multilevel
optimizer. CUDA and Metal convert the canonical undirected graph once to
float32 compressed sparse row (CSR) storage. Community volumes, counts,
memberships, and move proposals remain on the accelerator during each
local-moving or Leiden-refinement phase. Vertices are processed in seeded
color batches; proposals and updates are separate kernels, and atomic
float32 volume updates maintain the modularity objective. Empty labels are
compacted and the coarse graph is assembled by package-owned C++ between
levels. Leiden communities are checked and disconnected components are split
before aggregation. This shared host orchestration keeps CPU, CUDA, and Metal
semantics aligned without imposing a cuGraph dependency.

Walktrap is the published random-walk method, not a sampled proximity
approximation. Its exact transition storage is quadratic, so the
implementation stops before an unsafe allocation and directs large graphs to
Louvain or Leiden. Walktrap remains CPU-only. A request for CUDA or Metal
Walktrap fails explicitly; it is never reported as GPU work after running on
CPU. `igraph` is used only as a guarded test and benchmark oracle. Fixed seeds
control scheduling, but floating-point atomic update order means GPU results
are not guaranteed to be bitwise identical across devices.

## PCA And Truncated-SVD Initialization

`fastEmbedR::pca()` provides a small PCA API for reusable scores, loadings, and
t-SNE initialization. The API has no method-selection layer and does not
call `irlba`, ARPACK, Python, or `reticulate`.

The backend implementations share the same truncated low-rank PCA target but
use hardware-appropriate execution:

| Backend | PCA implementation |
| --- | --- |
| CPU | Package-native float32 blocked RSVD. Centering/scaling and large matrix products run in C++; Apple builds use Accelerate SGEMM and other platforms use a threaded float32 kernel. `n.cores` controls the temporary numerical-library/thread limit. |
| Metal | Package-native float32 block-subspace TSVD using MPS matrix multiplication and a resident unified-memory workspace. |
| CUDA | Native C++/CUDA PCA through RAPIDS RAFT TSVD compiled into the CUDA backend. Float32 input is read from its payload without constructing an R double matrix; scores and loadings remain float32 until the requested R return boundary. |

The Metal path is deliberately different from the former R-orchestrated RSVD.
It converts and centers the input once into a float32 buffer whose column-major
R layout is viewed as a row-major transposed matrix. A deterministic Gaussian
feature-space block is orthonormalized, two block power iterations are encoded
as MPS products, and only the small projected Gram matrix is returned to the
CPU for a float32 symmetric eigensolve. Final loadings and scores are projected
once on Metal. The input, projected block, basis, Gram matrix, loadings, and
scores remain allocated for the complete call, eliminating the repeated
full-matrix uploads and R intermediate matrices of the earlier Metal RSVD.

The CPU route similarly avoids materializing repeated double-precision
intermediates. Input is centered and optionally scaled once into contiguous
float32 storage. The seeded range sketch, subspace products, projected matrix,
scores, and loadings remain float32; only the skinny QR and small SVD cross the
ordinary R numeric boundary. The default uses 20 oversampling vectors, one
subspace iteration through rank 20, and two above rank 20. If the sketch spans
the complete available feature space, the redundant power iteration is
omitted.

For t-SNE initialization, the input is mean-centered before decomposition
and the resulting scores are centered and scaled to the small t-SNE
initialization scale. CUDA acceleration for this step is native C++/CUDA
through RAPIDS RAFT TSVD compiled in the package CUDA translation unit. If
RAFT TSVD support is not compiled in, CUDA PCA initialization fails loudly
rather than falling back to a different implementation.

The public CPU API exposes `n.cores` directly. fastEmbedR applies the
requested limit for the duration of the PCA call through standard numerical
library environment variables and, when installed, `RhpcBLASctl`, then
restores the previous process settings. The returned fit records the requested
and observable thread counts. A single-threaded BLAS remains single-threaded.
`irlba` is retained only as an external benchmark comparator and is not part of
the package implementation.

Set `tsne_init = TRUE` in `pca()` to retain the ordinary PCA fit and add an
`tsne_init` matrix derived from those same scores. The added matrix is
centered and rescaled so the maximum component standard deviation is `1e-4`,
matching the small-scale initialization expected by t-SNE/t-SNE optimizers
[1,3-4]. No second decomposition is performed. `tsne_pca_init()` remains a
compact helper for users who need only the initialization matrix or an RDS
cache. Either result can be passed as `Y_init` to `tsne_knn()` or
`tsne()`.

## UMAP From KNN

UMAP is implemented as a sparse graph optimization from a supplied KNN matrix.
The implementation follows the UMAP fuzzy simplicial-set formulation [7,13]:
local bandwidths are estimated per observation, neighbor distances are
converted to directed membership strengths, the graph is symmetrized, and a
low-dimensional layout is optimized with attractive edge updates and sampled
repulsive updates.

### CPU Graph Construction

The CPU path is package-native C++:

1. Validate and normalize KNN indices and distances.
2. Estimate row-wise `rho` and `sigma` values by smooth KNN distance search.
3. Convert directed KNN distances to membership strengths.
4. Build the symmetric graph without dense intermediates.
5. Store compact edge arrays and epoch schedules for the optimizer.

The engineering goal is to preserve UMAP's graph mathematics while reducing
memory traffic. Internal distances and weights use `float` where safe; indices
are stored as integer buffers; and the implementation avoids duplicate graph
copies between R and C++.

### UMAP Optimizer

The optimizer uses stochastic edge sampling, negative sampling, a decaying
learning rate, and compact contiguous layout buffers, following the UMAP
objective rather than an exact copy of any R implementation [7,10,13]. The
current permissive implementation uses package-local samplers, random-number
generation, and update kernels rather than vendored `uwot` code. `uwot`
remains an external benchmark and behavioural reference, not a source
dependency [10].

UMAP exposes two graph modes:

| Mode | Meaning | Intended use |
| --- | --- | --- |
| `"fuzzy"` | Standard UMAP fuzzy graph weights. | Scientific comparison with the original UMAP model and `uwot`. |
| `"binary"` | Unit weights on the symmetric KNN union, with the same low-dimensional optimizer. | Adjacency-only sensitivity analysis when neighbor identity is trusted more than distance calibration. It is not standard UMAP or a guaranteed acceleration. |

Binary graph preparation omits smooth-KNN bandwidth estimation, but all unit
edges receive the maximum sampling frequency. Fuzzy UMAP samples weak edges
less often. Binary mode can consequently perform more optimizer updates and be
slower on CPU. It may improve or reduce separation and label-aware quality
depending on the dataset, so benchmark and scientific claims report it
separately from fuzzy UMAP.

### Metal UMAP

The Metal backend is implemented in Objective-C++/Metal and uses the validated
atomic in-place edge-update kernel. It does not call Python, Torch, MLX, or
`reticulate`. The Metal optimizer consumes the same prepared graph as the CPU
path and returns only the final layout and metadata to R. The package-native
Metal FFT work used by t-SNE was informed by permissive Apple GPU FFT
engineering references [12].

### CUDA UMAP

The CUDA backend is compiled when `FASTEMBEDR_USE_CUDA=1` is enabled. The
public CUDA path uses the pure atomic optimizer. When
`FASTEMBEDR_USE_CUVS=1`, exact/IVF-Flat search is invoked through the native
cuVS C API, its KNN buffers remain on the selected CUDA device, and graph or
affinity construction consumes those pointers directly. The owning R object
contains external pointers with a finalizer; copying to ordinary R matrices is
an explicit diagnostic operation. No CPU or companion-package fallback is
used for a CUDA request.

## Interpolation-Based t-SNE From KNN

In this documentation, **interpolation-based t-SNE** identifies mathematical and workflow
lineage, not software compatibility. The shared published components are the
t-SNE KL objective, perplexity-matched sparse affinities, early-exaggeration
and normal phases, adaptive gains and momentum, FIt-SNE interpolation/FFT
repulsion, and fixed-reference transformation. fastEmbedR does not call or
port Python `openTSNE`, and its API, affinity classes, serialized objects,
neighbor-width defaults, optimizer trajectory, and coordinates are not
compatibility targets.

`tsne_knn()` implements the t-SNE optimization structure used by modern
published interpolation-based t-SNE workflows [3-4]:

1. Convert KNN distances to conditional probabilities by binary search on the
   Gaussian bandwidth for a target perplexity.
2. Symmetrize sparse probabilities and normalize the high-dimensional
   affinity matrix.
3. Initialize the embedding from the KNN-native default or an explicit
   user-supplied layout.
4. Run early exaggeration.
5. Run the normal optimization phase with adaptive gains, momentum, learning
   rate, update clipping, and recentering.
6. Return the layout with parameters, timing, and backend metadata.

The production matrix-input policy uses compact support and supplies
`ceiling(perplexity)` non-self candidate neighbors to the bandwidth search.
This is the package's defined sparse-affinity policy across CPU, Metal, and
CUDA. `tsne_knn()` records the exact support width and
support-to-perplexity ratio.

The public `tsne()` function is now only a convenience wrapper around this
KNN implementation. If a KNN object is supplied through `nn`, then
`tsne(data, nn = knn)` calls the same path as `tsne_knn(knn)` and
produces the same layout for the same seed and parameters. This mirrors the
KNN-input validation style used by R t-SNE tooling [11].

### Initialization

PCA initialization is explicit. Use:

```r
Y_init <- tsne_pca_init(x, backend = "cpu")
y <- tsne_knn(knn, Y_init = Y_init)
```

or pass `init_data` when a KNN-input run should compute PCA internally. The
old public `init = c("pca", "random")` argument was removed because hidden
initialization differences made KNN-input and one-call results difficult to
compare.

### Repulsive Force Approximation

The default large-data path uses an FFT-grid approximation inspired by
FIt-SNE [3]. Sparse attractive forces are evaluated from the KNN affinity
graph. The negative force is approximated by placing points on a
two-dimensional grid, convolving with the t-SNE kernel, and interpolating the
resulting force back to points [3-5]. The Barnes-Hut path is not part of the
public benchmark surface because the FFT-grid path is the intended standard
for MNIST70k-scale data; Barnes-Hut remains an important historical reference
for tree-based t-SNE acceleration [2].

Automatic small-data execution uses exact repulsion. If FFT is requested
explicitly, CPU and Metal use a minimum 128-cell grid. A matched long-run
diagnostic showed that the former 64-cell setting could pass first-step force
tests yet exceed the final common-affinity KL gate after a complete trajectory.
The 128-cell floor restored CPU/Metal KL, trustworthiness, and Preserve@30
eligibility without changing KNN input, initialization, optimizer schedule,
momentum, clipping, or stopping. The fitted object's
`fastEmbedR_config$fft_grid_size` field records the resolution actually used.

Numerical correctness is tested below the final-layout level. An independent
dense float64 reference evaluates affinities, exact attractive and repulsive
forces, the KL objective, and central finite differences. The production
float32 kernels are checked against that oracle; FFT-grid repulsion is checked
against exact repulsion over increasing grid sizes; and available accelerator
backends are compared with CPU after one update from identical coordinates and
through common-affinity KL trajectories. The same tests cover compact,
standard, and expanded candidate support plus duplicated observations, zero
distances, disconnected KNN graphs, coincident coordinates, and extreme finite
coordinates through every available public backend. The initial real-CUDA
first-step check exposed a zero-sign discrepancy in adaptive gains; after the
CUDA update adopted the CPU/Metal three-state sign rule, CPU-CUDA first-step
relative error fell from `2.10e-3` to `3.68e-8` without relaxing the
predeclared `2e-3` threshold. Identity-bound real-hardware validation reports
an unavailable backend as unavailable rather than passing it by fallback.
Final-layout eligibility additionally requires common-affinity KL no greater
than 1.05 times the named matched reference, trustworthiness no more than 0.01
below that reference, and Preserve@30 no more than 0.01 below it. Procrustes
and embedding-neighborhood agreement are reported as diagnostics rather than
used alone as gates for the non-convex t-SNE objective.

### CPU, Metal, and CUDA

| Backend | Native implementation |
| --- | --- |
| CPU | C++ sparse affinities, FFT-grid repulsion, gains/momentum optimizer [3-4]. |
| Metal | Objective-C++/Metal scatter, FFT-grid convolution, gather, attractive-force, update, and centering kernels [3,12]. |
| CUDA | CUDA kernels with cuFFT for the FFT-grid convolution and device-side optimizer updates [3,5]. |

The Metal implementation includes package-native FFT kernels. Standalone
MPSGraph FFT diagnostics were tested and then removed because they did not
provide enough benefit to justify another user-facing backend.

The current CPU FFT-grid path reuses a scoped worker team, FFT bit-reversal and
root tables, and per-thread column scratch. The current Metal path encodes 16
unchanged optimization iterations per command buffer. The current CUDA path
captures chunks of 25 unchanged iterations in CUDA Graphs. These are
implementation-only optimizations: they do not reduce perplexity, grid size,
or iteration counts. Their profiling results, output-agreement gates, rejected
experiments, and exact benchmark artifacts are documented in
[Backend Performance Engineering](backend-performance-engineering.md).

## Landmarking

Landmarking is implemented as an explicit approximation. The package embeds a
subset, projects non-landmark observations using fixed-reference KNN
interpolation/transform steps, and records projection/transform time
separately. Landmarking is not used silently inside full `tsne()` or
`umap()` calls.

## Parameter Policy And Autotuning

The configurability is deliberately asymmetric. t-SNE exposes perplexity
and support, initialization, iteration counts, exaggeration, learning rate,
momentum, clipping, and exact-versus-FFT repulsion. Its native helper supplies
opt-SNE-inspired learning-rate and iteration defaults only when values are
omitted [6], and `auto_config = FALSE` disables automatic iteration/stopping
choices.

UMAP preserves one package-owned policy across CPU, Metal, and CUDA. The user
selects neighbors, metric, graph mode, backend, seed, and preprocessing; the
distance profile and data size select epochs, minimum distance, learning rate,
and initialization effort. Spread, repulsion strength, and negative-sample
rate are fixed at 1, 1, and 5. These choices never alter a supplied KNN graph
and are all stored in the returned configuration [7,13]. The package is
therefore an opinionated high-throughput UMAP implementation, not a drop-in API
for arbitrary UMAP hyperparameter sweeps.

## License Boundary

The implementation is intended to remain compatible with the MIT license.
References such as `uwot`, `Rtsne`, openTSNE, FIt-SNE, t-SNE-CUDA, FAISS,
RAPIDS cuVS, and AppleSiliconFFT informed design and benchmarking decisions
[3-5,8-12]. GPL code is not vendored, linked, or required by the package.

See [References](references.md) for AACR-style citations and software
acknowledgements.
