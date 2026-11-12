# Usage And API

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**API** |
[Reproducibility](reproducibility.md) |
[References](references.md)

Start with `pca()`, `tsne()`, `umap()`, and `graph_cluster()`. These are the
four analysis entry points for resident and file-backed data. The same
embedding functions accept either a matrix or a precomputed KNN result.
File-backed data enter through `massive_matrix()`; use
`graph_cluster()` for resident or file-backed clustering. Graph preparation
helpers remain available for full file-backed fits; optimization and
clustering stages are internal.

## Which Function Should I Use?

| Situation | Use |
| --- | --- |
| You want to precompute fastEmbedR's native neighbors | `precompute_knn()` |
| You want neighbors from a file-backed source | `precompute_knn(massive_matrix(...), output = "knn")` |
| You already computed nearest neighbors | `umap(knn)` or `tsne(knn)` |
| You want one call from a data matrix | `umap()` or `tsne()` |
| You want reusable PCA scores or t-SNE initialization | `pca()` |
| You want to compare UMAP and t-SNE fairly | compute one host KNN list once, then reuse it |
| You want Apple GPU | set `backend = "metal"` explicitly |
| You want NVIDIA GPU | build with CUDA/cuVS, then set embedding `backend = "cuda"` |
| You want a fast approximation for very large data | set `landmarks` in `umap()` or `tsne()` and report it as landmarking |
| Your float32 input exceeds RAM | describe it with `massive_matrix()`; choose `landmarks` when approximating |
| You want quality metrics | `evaluate_embedding(x, layout)` |
| You want native graph communities | `graph_cluster(x, method = "leiden")` |
| You want landmark clustering | `graph_cluster(x, landmarks = 0.1)` |
| You want to inspect a file-backed result | `massive_read_rows(result)` |

Every analysis fit supports `summary(fit)`. It returns the recorded backend
and devices, full or landmark scope, stage times, and available memory
figures with their measurement basis. Missing memory measurements remain
`NA`; they are not estimated from the number of rows. The summary contains
no output coordinates or labels. Read only the rows needed from a
file-backed result:

```r
print(summary(fit))
first_rows <- massive_read_rows(fit, first = 1L, n = 20L)
```

`massive_read_rows()` also accepts a file-backed PCA fit and reads its score
file. Matrix reads are limited to 128 MB per call; label readers impose
their own row caps. A file-backed descriptor never loads all rows for
`summary()`.


Precompute KNN only when methods must share exactly the same neighbors:

```r
knn <- precompute_knn(
  x,
  k = 50L,
  metric = "euclidean",
  backend = "cpu",
  n.cores = 4L
)
layout_umap <- umap(knn, seed = 1)
layout_tsne <- tsne(knn, init_data = x, seed = 1)
```

This keeps nearest-neighbor time separate from embedding time and makes
benchmarks easier to interpret.

For a disk-backed source, set session defaults with base R `options()` or
pass an explicit backend and memory budget:

```r
old <- options(backend = "cpu", n.cores = 4L,
               storage = "file_backed",
               memory_limit = "8GB")
source <- massive_matrix("features.f32", nrow = 1000000L, ncol = 128L)
knn_file <- precompute_knn(source, k = 30L, output = "full_knn")
options(old)
```

The file-backed result is a `fastEmbedR_massive_graph` descriptor, not a
resident KNN matrix. It supports Euclidean distance and needs a new output
prefix. Reuse it to prepare multiple embeddings, or let `umap()` and
`tsne()` build their graph stages from the original file-backed matrix in
one full-data call. A storage option never copies a resident matrix to SSD
implicitly. The memory option limits algorithm buffers, not total RSS.
The option names are ordinary session-wide R keys and can be shared with
other packages that adopt the same values. `pca()`, `tsne()`, `umap()`,
`graph_cluster()`, and `precompute_knn()` also accept an explicit `storage`
argument that overrides the session setting.
The storage request is checked before analysis: `storage = "file_backed"`
cannot silently materialize a clustering input. A file-backed matrix needs
explicit `landmarks` for clustering until one-call full graph preparation is
implemented; full file-backed clustering accepts a prepared graph.
Conflicting legacy `massive` settings fail before fitting.

### Updating older scripts

Use `tsne(x, landmarks = 0.2)` and `umap(x, landmarks = 0.2)` in place of
the removed landmark wrappers. For file-backed results, use
`massive_read_rows(result, first, n)` in place of the four separate
clustering-result readers. The same call returns a matrix for matrix
sources, an integer label vector for full-graph clustering, or membership
and confidence for landmark clustering.

The one-call functions and `precompute_knn()` intentionally hide the KNN
algorithm choice. Their `backend` accepts only `"cpu"`, `"metal"`, or
`"cuda"`. CPU KNN uses exhaustive exact search below 5,000 observations and
compiled faissR HNSW for larger Euclidean searches, and native HNSW for
larger cosine or correlation searches. Metal uses native exact/IVF-Flat;
CUDA uses RAPIDS cuVS
exact or IVF-Flat search and keeps its output resident on the device. A CUDA
KNN object should therefore be reused with a CUDA embedding
backend. A host KNN result from another tool may still be
supplied as a plain list containing `indices` and `distances`; fastEmbedR never
calls that tool itself.

## Distance Metrics

The default distance is Euclidean:

```r
fit <- umap(x, n_neighbors = 50, metric = "euclidean", n.cores = 4)
```

Cosine distance is available through exact CPU KNN:

```r
fit_cosine <- umap(x, n_neighbors = 50, metric = "cosine", n.cores = 4)
```

Current metric support is deliberately explicit:

| metric | supported backends | notes |
| --- | --- | --- |
| `euclidean` | native CPU/Metal and optional direct CUDA/cuVS | Recommended default for large UMAP/t-SNE benchmarks. |
| `cosine` | native CPU/Metal and compiled CUDA | Rows are normalized internally. |
| `correlation` | native CPU/Metal and compiled CUDA | Rows are centered and normalized internally. |

## Parameter Philosophy And Scope

fastEmbedR is deliberately more configurable for t-SNE than for UMAP. The
t-SNE API exposes perplexity and affinity support, initialization, iteration
counts, early and normal exaggeration, learning rate, momentum, and clipping.
Resident `tsne()` defaults to compact support with `ceiling(perplexity)`
neighbors. Set `affinity_k = ceiling(3 * perplexity)` for a conventional-width
comparison. `tsne_knn()` and `prepare_tsne_knn()` use `n_neighbors` for the
same control when a precomputed graph has enough columns. Record the chosen
width with benchmark results; changing it changes both KNN work and the
affinity model. In resident landmark t-SNE, `affinity_k` controls the
reference fit; the query projection keeps its separate `transform_k`.
Repulsion uses FFT for both 2D and 3D output. Set `auto_config = FALSE` and
supply explicit values when automatic iteration or stopping is not wanted.

UMAP is an opinionated high-throughput implementation, not a general-purpose
UMAP tuning interface. It exposes
`n_neighbors`, metric, graph mode, preprocessing, backend, seed, and CPU thread
count. The current release selects and records the remaining optimizer policy:

| UMAP setting | Public status | Current policy |
| --- | --- | --- |
| epochs | internal | 500 below 10,000 rows; 200 for larger ordinary profiles; at least 300 for a high-variability distance profile |
| `min_dist` | internal | 0.01 normally; 0.1 for the documented wide-shell profile |
| spread | internal | 1 |
| learning rate | internal | 1 normally; 1.25 for the wide-shell profile |
| repulsion strength | internal | 1 |
| negative-sample rate | internal | 5 |
| initialization | reusable | package-native graph initialization; use `umap_init()` to compute and reuse it |
| update mode | internal | backend-native asynchronous/atomic schedule; no synchronous GPU mode |

These settings materially affect compactness, optimization effort, and the
attraction/repulsion balance. They are fixed to keep one release policy and
one validated update schedule aligned across CPU, Metal, and CUDA; they are not
claimed to be universally optimal for every scientific question.

The one-call nearest-neighbor router also hides index parameters. It targets
recall 0.99 and chooses the validated CPU, Metal, or CUDA route. To control the
search independently, provide an external KNN object to `umap_knn()` or
`tsne_knn()`.

Resolved choices are not hidden from results:

```r
fit$parameters
fit$timings
```

Float32 input, precomputed or GPU-resident KNN, prepared UMAP state, compact
t-SNE support, and landmarking are the explicit memory/speed choices. Compact
support and landmarking are approximations and should be reported as such.
Users needing arbitrary UMAP `min_dist`, spread, epoch, learning-rate,
negative-sampling, repulsion, or optimizer sweeps
should use a general-purpose UMAP implementation; fastEmbedR does not claim
parameter or API interchangeability with those packages.

## Basic KNN-First UMAP

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

fit <- umap(x, n_neighbors = 30, n.cores = 4)
layout <- fit$layout

plot(layout, pch = 21, bg = labels)
```

The one-call interface computes KNN internally:

```r
fit <- umap(
  x,
  n_neighbors = 30,
  seed = 1
)
plot(fit)
```

## t-SNE From The Same KNN

The KNN must contain at least `ceiling(perplexity)` non-self columns. By
default t-SNE uses that compact width. Set `n_neighbors` in `tsne_knn()`
to use more supplied KNN columns.

```r
pca_fit <- pca(x, ncomp = 2, seed = 1, tsne_init = TRUE)
Y_init <- pca_fit$tsne_init
layout_tsne <- tsne_knn(
  knn,
  Y_init = Y_init,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250
)

plot(layout_tsne, pch = 21, bg = labels)
```

`Y_init` can be computed once and reused across runs. `init_data` is still
available as a convenience; it is used only to compute PCA initialization for
KNN-input runs and is not used for neighbor search or optimization.

## PCA API

`fastEmbedR::pca()` exposes the backend-native truncated PCA used internally
for t-SNE initialization:

```r
pca_fit <- pca(
  x,
  ncomp = 2,
  backend = "cpu",
  n.cores = 4,
  seed = 1,
  tsne_init = TRUE
)
Y_init <- pca_fit$tsne_init
layout <- tsne_knn(knn, Y_init = Y_init, perplexity = 30)
```

The public `pca()` helper is intentionally simple: there is no `irlba` or
ARPACK method menu and no Python bridge. For t-SNE initialization, CUDA uses
package-native rSVD and may select RAPIDS RAFT TSVD when that optional route is
enabled. It fails loudly if the requested CUDA backend is unavailable.
Float32 CUDA input is passed to the native fit without materializing an R
double matrix, and float32 scores/loadings are returned. Metal uses a native
float32 block-subspace rSVD with MPS matrix products and a resident workspace.
CPU uses fastEmbedR's native
float32 blocked rSVD, with a seeded Gaussian sketch, quality-preserving
oversampling, and one or two subspace iterations according to requested rank.

For `backend = "cpu"`, `n.cores` is a positive integer that temporarily
sets the BLAS/OpenMP thread limit. The result records
`n.cores_requested`, `n.cores_effective`, and `core_control`.
Single-threaded BLAS builds cannot use additional cores and report one
effective thread. Metal and CUDA ignore this CPU-only argument.

The ordinary PCA scores are always retained in `pca_fit$scores`.
`tsne_init = TRUE` adds a second matrix, centered and scaled so that its
largest component standard deviation is `1e-4`; no second decomposition is
performed.

Supplying `xtest` adds projected test coordinates in
`pca_fit$scores_test`. fastEmbedR intentionally omits an SVD method selector.
With `massive = "out_of_core"`, supply `xtest` as a `massive_matrix()`
descriptor with matching columns. The projection runs on the selected
backend and writes a separate file beside `output`, replacing `.f32` with
`.test.f32`. `massive_open_pca(output)` checks both score files. For a full-data fit, if the test
projection is interrupted, reopen the completed fit and resume that stage
with `massive_project_pca()` using the test score path.

With out-of-core `tsne_init = TRUE` and two or three components, PCA writes
`.init.f32` beside the score file. Its bounded CPU normalization matches the
resident initialization scale, and the result can be passed to
`tsne()` as `Y_init` with a prepared affinity graph. For a full-data fit, after interruption, rerun
`pca(..., tsne_init = TRUE, resume = TRUE)` with the same input and output
to regenerate the initialization from verified scores.

`irlba` is not imported, suggested, linked, or called by fastEmbedR. It is
used only by external benchmark scripts as a CPU reference.

## Explicit GPU Use

GPU use is explicit. A request for Metal or CUDA must run that backend or fail
clearly.

```r
fit <- tsne(x, perplexity = 30, backend = "metal", seed = 1)
layout <- fit$layout
```

For CUDA builds with RAPIDS cuVS available:

```r
fit <- tsne(x, perplexity = 50, backend = "cuda", seed = 1)
```

The package does not silently run these examples on CPU and report them as GPU
results.

## Graph Clustering

Pass a resident data matrix directly to build its KNN graph in the call:

```r
communities <- graph_cluster(x, method = "leiden", k = 20,
                             backend = "cpu", seed = 1)
```

Pass a prepared graph when its exact edges must be reused:

```r
graph <- knn_graph(x, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
table(communities$membership)
```

For an embedding-space graph, pass the fit directly:

```r
fit <- tsne(x, perplexity = 15, backend = "cpu", seed = 1)
graph <- knn_graph(fit, k = 20, weight = "snn", n.cores = 4)
communities <- graph_cluster(graph, method = "leiden", seed = 1)
```

For a resident landmark approximation, fit the graph on selected rows and
assign every remaining row from its nearest reference labels:

```r
communities <- graph_cluster(x, landmarks = 0.1,
                                 method = "leiden", k = 30,
                                 vote_k = 30, backend = "cpu")
table(communities$membership)
```

`landmarks`, `k`, and `vote_k` are explicit controls. `vote_k` applies only
to resident landmark assignments. A prepared graph rejects landmark
controls, since its edges and vertices are already fixed. File-backed
matrices need an explicit landmark count; a full file-backed fit still
needs a prepared graph.

`graph_cluster(..., landmarks = ...)` keeps reference labels fixed and reports the reference
fit, selection, all-row confidence, and stage times. Its graph assembly and
distance-weighted label vote run on CPU even when KNN and graph fitting use
Metal or CUDA. Use `parallel.mode = "fit"` for a shared multicore CPU fit.
The accelerator routes still require device qualification.

`knn_graph()` uses the selected backend only when it must compute neighbors.
`graph_cluster()` runs native CPU, CUDA, or Metal Louvain/Leiden without
calling igraph, cuGraph, Python, or another clustering routine. An
unavailable or unsupported GPU backend fails explicitly.

CPU Louvain and Leiden can run independent seeded restarts concurrently:

```r
communities <- graph_cluster(graph, method = "leiden", backend = "cpu",
                             n_runs = 8, n.cores = 4, seed = 1)
```

The best modularity is selected in the same run order as `n.cores = 1`.
The result records `effective_cpu_workers` and `parameters$n.cores`.
In restart mode, `n.cores` must not exceed `n_runs`. This accelerates
repeated restarts. Vector consensus, CUDA, and Metal reject
`n.cores > 1`.

For one shared CPU partition, request
`graph_cluster(graph, method = "leiden", backend = "cpu",
n_runs = 1, n.cores = 4, parallel.mode = "fit")`.
This experimental mode parallelizes local-move proposals. Graph
coarsening and Leiden refinement still run on one CPU core, and the
result reports `parallel_mode = "fit"`. Measure its speed separately
from independent restarts.
`graph_cluster(x, landmarks = ..., parallel.mode = "fit")` applies the
same shared CPU fit to the resident landmark graph. File-backed landmark
query KNN and label voting remain separate stages.

CUDA Louvain and Leiden have an experimental coordinated resident fit on
two to four devices. For example, `graph_cluster(graph, method = "leiden",
backend = "cuda", n_runs = 1, devices = c(0, 1))` splits graph rows
across two GPUs and synchronizes one global partition through host RAM
after each color phase. No SSD graph or optimizer files are used.
Multi-GPU device validation and performance qualification are pending.
Consensus, projection weighting, and small-group controls use the same
coordinated fit for each clustering step; graph construction and
postprocessing remain explicit host stages.
`graph_cluster(x, landmarks = ..., devices = c(0, 1))`
uses the selected devices for query KNN shards and the shared resident
landmark-graph fit; voting still streams on the CPU. The `"auto"` route
passes the devices to the resident graph fit when it fits in memory.

For a symmetric file-backed fuzzy graph, the isolated experimental
`gpu_storage = "stream"` route keeps CSR edges and partition state on
host storage while the selected CUDA devices process bounded batches:

```r
fit <- graph_cluster(fuzzy_graph, method = "louvain", backend = "cuda",
                       devices = c(0, 1), gpu_storage = "stream",
                       output = "clusters.u32", memory_limit = "16GB")
```

This performs multilevel Louvain with GPU local moves and CPU graph
contraction and modularity audits. `method = "leiden"` uses the same
GPU local moves plus file-backed CPU Leiden refinement and connectivity
audits. It uses native C++ and CUDA without a Python runtime. The option is
designed for graphs larger than combined GPU memory, but its actual
CUDA correctness, quality, and capacity still require device
qualification. Do not run
this file-backed route on the UCT HPC.

## Landmark Workflow

Set `landmarks` in `umap()` or `tsne()` for explicit landmark
approximations. Both functions select a reference subset, fit the ordinary
embedding implementation on that subset, and project the remaining rows.

`precompute_query_knn(reference, query, ...)` exposes the query-only search
used by the projection stage. It searches the fixed reference only and avoids
constructing unnecessary query-to-query neighbors.

Landmark results report the reference-stage memory ratio and saving, plus the
estimated query-to-reference KNN payload. These are float32 working-set
estimates, excluding the input matrix, R overhead, allocator caches and
temporary buffers; they are not peak-RSS measurements.

The integrated t-SNE call accepts the landmark selection directly:

```r
fit <- tsne(
  x,
  landmarks = 0.5,
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
fit <- umap(
  x,
  landmarks = 0.5,
  n_neighbors = 30,
  graph_mode = "fuzzy",
  backend = "cpu",
  refinement_epochs = 50L,
  seed = 1
)
plot(fit)
```

Resident landmark UMAP defaults to 50 fixed-reference refinement epochs.
Set `refinement_epochs = 0L` to use the initial projection alone, or vary the
count to measure the quality and time tradeoff on each dataset.

Both integrated landmark calls return a reusable fixed-reference model in
`fit$model`. New observations are supplied in the original feature space; the
stored standardization and PCA transforms are reused before projection:

```r
new_layout <- project_landmark_model(
  fit$model,
  newdata,
  transform_k = 15
)$layout
```

To reconstruct the original training rows, pass the complementary rows as
`query_indices = fit$model$selection$query_indices`. With no indices, every
supplied row is treated as a genuinely new observation.

CPU projection uses exhaustive exact search below 5,000 reference rows and
compiled faissR HNSW for larger Euclidean searches, and native HNSW
for larger cosine or correlation searches. The HNSW
route is calibrated for target recall 0.99 and reports that recall was not
audited at runtime. Metal uses native exact search
for small references and recall-tuned IVF-Flat for larger references, followed
by native fixed-reference transform kernels. CUDA uses native exact search for
smaller references and IVF-Flat for larger references; its KNN result remains
device resident for the projection and refinement stages.

## File-Backed Workflows

`massive_matrix()` describes a row-major `.f32` or `.fbin` file without
loading it into R. Ordinary matrix calls remain unchanged. File-backed
calls require a persistent output path. A descriptor without landmarks
selects a full-data fit for PCA, UMAP, and t-SNE; explicit landmarks select
an approximation for all three. A prepared graph may be reused:

| Function | Mode | What is fitted |
| --- | --- | --- |
| `pca()` | `massive = "out_of_core"` | Streamed PCA on all rows. |
| `pca()` | File-backed `landmarks` | Fit selected rows and project all source rows to bounded float32 scores. |
| `massive_project_pca()` | File-backed projection | Apply a saved PCA fit to another file-backed source. |
| `umap()`, `tsne()` | `massive = "landmark"` | Reference landmarks, then projection of other rows. |
| `umap()`, `tsne()` | One-call full data | Prepare KNN, graph, spectral UMAP or PCA t-SNE initialization, and optimize all rows. |
| `umap()`, `tsne()` | `massive = "out_of_core_graph"` | All rows of a prepared graph or affinity object. |

The experimental one-call full route needs Euclidean raw data, a new `.f32`
output path, and enough SSD space for KNN, graph, initialization, and final
coordinates. It records separate stage times and intermediate descriptors.
The ordinary route uses a lower-disk bucketed graph sort and releases
intermediate KNN files by default. Set `keep_knn = TRUE` to retain them.
Resident UMAP, including the wide CSR route, has no optimizer checkpoint
or mid-fit resume. An interrupted resident fit restarts from the beginning.
A prepared resident graph can be reused to start a new fit, but it is not
a recovery snapshot. The result reports `checkpoint_scope = "none"` and
`resume_supported = FALSE`.

Only the file-backed route accepts `checkpoint = TRUE`. Its
source-identified stage record uses the
restartable global graph sort and retains KNN files until the fit completes;
this requires more SSD space. A matching call with `resume = TRUE` reuses
completed stages and native checkpoints. Changed inputs or controls fail.
Work interrupted before a native checkpoint may require a new output.

```r
source <- massive_matrix("full.fbin")
layout_u <- umap(source, n_neighbors = 30L, n_components = 2L,
                 backend = "cpu", storage = "file_backed",
                 output = "full-umap.f32", memory_limit = "8GB")
layout_t <- tsne(source, perplexity = 30, n_components = 3L,
                 backend = "cpu", storage = "file_backed",
                 output = "full-tsne.f32", memory_limit = "8GB")
```

Full-data file-backed PCA runs through `pca()`. State the storage mode when
you require bounded reads even if the source could fit in memory:

```r
source <- massive_matrix("full.fbin")
fit <- pca(source, ncomp = 2L, massive = "out_of_core",
    output = "full-scores.f32", backend = "cpu", n.cores = 8L,
    checkpoint = TRUE)
```

File-backed landmark PCA uses the same `pca()` entry point. A positive
integer count, a fraction in `(0, 1)`, or `TRUE` selects reference rows;
all source rows receive scores. The result records the landmark scope,
selection, reference fit, projection stages, backend, and file-backed score
descriptor. `massive_read_rows(fit, first, n)` reads bounded output, and
`massive_open_pca(output)` reopens the saved fit. Explicit row indices and
partial-landmark checkpoint/resume are currently unsupported and fail before
fitting. The lower-level `massive_project_pca()` remains available for a
prepared reference fit.

```r
fit <- pca(source, ncomp = 2L, landmarks = 10000L,
    output = "landmark-scores.f32", backend = "cpu",
    storage = "file_backed", memory_limit = "8GB")
summary(fit)
first_scores <- massive_read_rows(fit, first = 1L, n = 20L)
```

For CPU full-graph UMAP, `cpu_update = "sgd"` uses ordered updates with one
worker. Set `cpu_update = "synchronized"` with `n.cores > 1` to use the
synchronized multicore optimizer. This changes the update rule; the result
records `optimizer`, `parameters$cpu_update`, and
`resources$effective_cpu_workers`. Other modes and backends reject
`"synchronized"` rather than silently using another optimizer.

The prepared-graph mode requires a graph and file-backed initialization
supplied by the caller. The one-call mode builds them from the matrix.
`massive_full_knn_graph()` uses the package's existing
CPU exact/HNSW or CUDA cuVS exact/IVF search routes; approximate search
remains approximate even when every row is optimized. The file-backed
Metal full-graph UMAP supports 2D and 3D with streamed graph edges and a
complete shared-memory layout.
Its optimizer runs on Metal and then uses one CPU thread for 25 final
epochs in 2D or 50 in 3D. The fit reports `optimizer_backend_stages`,
`optimizer_stage_seconds`, and the combined pipeline backend.
Experimental Metal full-graph t-SNE supports 2D and 3D.
`graph_access = "mmap"`
maps the complete read-only CSR graph into Metal buffers;
`graph_access = "stream"` uploads bounded SSD graph chunks of at most
`edge_chunk_capacity` edges. The default `"auto"` streams when the graph
exceeds one chunk. The layout and optimizer state remain resident. A
deterministic integer grid is used in the file-backed Metal FFT routes;
large-data performance remains unqualified.
The coordinated CUDA optimizer is multi-GPU but retains a full layout on
each GPU.
Requests for unsupported backends fail explicitly.

UMAP and t-SNE reject `massive = "auto"`: memory pressure must not
silently switch a full fit to landmark projection. PCA retains an auto
mode because it chooses between resident and streamed execution, not
between fitting all rows and fitting a subset. `memory_limit` budgets
algorithm buffers; it is not a bound on process RSS or filesystem cache.
See the experimental massive-data vignette for file formats, checkpointing,
and stage-specific controls.

## Automatic Parameters

`tsne()` and `tsne_knn()` use `auto_config = TRUE` by default. Missing
t-SNE settings are resolved in native C++ using the opt-SNE strategy:

- `"auto"` learning rate becomes `n / early_exaggeration`.
- Early exaggeration can stop at the local maximum of KLD relative change.
- The normal phase can stop when KLD improvement drops below the opt-SNE
  threshold.

The KLD monitor is enabled only where it is computationally honest: small CPU
FFT runs. Large FFT and GPU runs keep opt-SNE's learning-rate/default-limit
policy but do not perform a hidden CPU O(n^2) KLD poll or report it as GPU
work.

`umap()` and `umap_knn()` also choose internal defaults from the supplied KNN
distance profile in C++. This keeps the public API small while preserving the
supplied neighbor graph.

## Advanced API

The direct `*_knn()` optimizers and `massive_*` stage functions below
support controlled benchmarking and recovery. Most analyses need only
the functions in the table above. The reference manual retains these
advanced controls and their backend restrictions.

| Function | Purpose |
| --- | --- |
| `umap_knn()` | UMAP from a supplied KNN object or matrices. |
| `tsne_knn()` | Direct native interpolation-based t-SNE optimizer from KNN. |
| `transform_tsne()` | Fixed-reference interpolation-based t-SNE transform for query points. |
| `knn_graph()` | Compact graph from data, an embedding, or supplied KNN. |
