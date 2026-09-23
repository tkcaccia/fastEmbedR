# Public API Map

fastEmbedR separates its canonical workflow from reusable-state, diagnostic,
secondary, and compatibility interfaces. The installed package exposes the
same inventory programmatically:

```r
fastEmbedR::fastEmbedR_api()
fastEmbedR::fastEmbedR_capabilities()
```

`fastEmbedR_api()` describes the software contract. Capability availability is
installation-specific, so a supported Metal or CUDA entry does not imply that
the current machine has that backend. Explicit unavailable accelerator requests
fail; they do not silently use CPU.

UMAP entries in this map refer to fastEmbedR's opinionated fixed optimizer
policy. The public interface controls neighborhood, metric, graph mode,
preprocessing, backend, seed, output dimension, and CPU threads. It does not
offer arbitrary epoch, `min_dist`, spread, learning-rate, repulsion,
negative-sampling, index-tuning, or optimizer-mode sweeps. Resolved values are
recorded in each fit; users requiring those sweeps should use a general-purpose
UMAP implementation.

## Canonical Workflow

| Function | Purpose | Accepted input | Returned class | CPU / Metal / CUDA | Exactness and residency | Methods and lifecycle |
| --- | --- | --- | --- | --- | --- | --- |
| `fastEmbedR_backend()` | Get or set the session backend | `NULL` or backend string | Character scalar | Yes / yes / yes | Configuration only | Stable canonical |
| `fastEmbedR_capabilities()` | Inspect compiled/runtime capabilities | None | `data.frame` | Reports all | Diagnostic device, precision, native engine, runtime-library, and failure-reason data | Stable canonical |
| `fastEmbedR_api()` | Inspect the public API contract | None | `data.frame` | Reports all | Diagnostic; no device data | Stable canonical |
| `pca()` | Backend-native truncated PCA | Matrix, data frame, or `float32` | `fastEmbedR_pca` | Yes / yes / yes | Randomized approximation; accelerator intermediates, host R result | Stable canonical; no registered S3 method |
| `precompute_knn()` | Reusable non-self nearest neighbors | Matrix, data frame, or `float32` | `fastEmbedR_knn`; CUDA also `fastEmbedR_gpu_knn` | Yes / yes / yes | Exact/approximate by router; CUDA result remains resident, CPU/Metal result is host | Stable canonical; `print()` |
| `tsne()` | Complete native t-SNE workflow | Matrix, data frame, `float32`, or KNN object | `fastEmbedR_embedding` | Yes / yes / yes | Sparse affinity plus exact/FFT-grid repulsion; final layout is host | Stable canonical; `print()`, `plot()` |
| `tsne_knn()` | t-SNE from reusable neighbors | Index/distance matrices, KNN/GPU-KNN, or prepared t-SNE object | Numeric or `float32` layout matrix | Yes / yes / yes | Resident CUDA KNN can be consumed on device; final layout is host | Stable canonical |
| `umap()` | Opinionated fixed-policy UMAP workflow | Matrix, data frame, `float32`, or KNN object | `fastEmbedR_embedding` | Yes / yes / yes | Stochastic fuzzy/binary embedding under recorded package policy; final layout is host | Stable canonical; `print()`, `plot()` |
| `umap_knn()` | Fixed-policy UMAP from reusable neighbors | Index/distance matrices, KNN/GPU-KNN, prepared graph, or reusable initialization | Numeric or `float32` layout matrix | Yes / yes / yes | Resident CUDA KNN can be consumed on device; recorded optimizer policy; final layout is host | Stable canonical |
| `select_landmarks()` | Reusable landmark subset | Matrix, data frame, or `float32` | `fastEmbedR_landmark_selection` | CPU selection for all workflows | Seeded approximation; host object | Stable canonical |
| `fit_landmark_model()` | Fit the UMAP or t-SNE reference | Data plus landmark selection | `fastEmbedR_landmark_model` | Yes / yes / yes | Reference-subset approximation; host model | Stable canonical |
| `project_landmark_model()` | Project original/new rows | Landmark model plus compatible data | `fastEmbedR_embedding` | Yes / yes / yes | Fixed-reference approximation; accelerator internals, host layout | Stable canonical; `print()`, `plot()` |

The package does not currently register `summary()`, `predict()`, or
`transform()` S3 methods. New-observation workflows use the explicit stable
functions `project_landmark_model()` and `transform_tsne()`. Prepared KNN and
landmark-model classes are reusable state objects; their fields are documented,
but users should prefer the public fitting/projection functions over direct
list manipulation.

Metal and CUDA UMAP/t-SNE currently require two output components; CPU also
supports higher-dimensional layouts. GPU-resident wording refers to internal
buffers or `fastEmbedR_gpu_knn`. Every public embedding ultimately returns an R
object on the host.

## Advanced Reusable-State Interfaces

| Function | Purpose | Input to output | Backend/residency | Lifecycle |
| --- | --- | --- | --- | --- |
| `precompute_query_knn()` | Query-to-reference search | Two matrices to `fastEmbedR_knn` | CPU/Metal/CUDA; CUDA result can remain resident | Stable advanced; `print()` |
| `prepare_tsne_knn()` | Normalize and cache support | Host KNN to `fastEmbedR_tsne_prepared` | Backend-independent host state | Stable advanced |
| `prepare_umap_knn()` | Build CSR graph/schedule once | Host KNN to `fastEmbedR_umap_prepared` | CPU/Metal/CUDA construction; host diagnostic state | Stable advanced |
| `umap_init()` | Reusable graph and coordinates | Host KNN/prepared state to `fastEmbedR_umap_initialization` | CPU/Metal/CUDA; host diagnostic state | Stable advanced; `print()` |
| `transform_tsne()` | Fixed-reference transformation | Reference layout plus KNN or reference/query data to layout matrix | CPU/Metal/CUDA; host result | Stable advanced |
| `tsne_pca_init()` | Cached t-SNE-scaled PCA coordinates | Matrix/data frame/`float32` to matrix | CPU/Metal/CUDA; host result | Stable advanced |

`landmark_umap()` and `landmark_tsne()` are stable one-call conveniences around
selection, fitting, and projection. They return `fastEmbedR_embedding` and
support CPU, Metal, and CUDA. The staged canonical API is preferable when a
reference model will be reused.

## Diagnostics, Secondary Utilities, and Compatibility

| Function | Role | Input and output | Backend/residency | Lifecycle |
| --- | --- | --- | --- | --- |
| `evaluate_embedding()` | Quality diagnostics | Data/layout/optional labels or KNN to one-row `data.frame` | CPU, conditional Metal/CUDA metrics; host result | Diagnostic |
| `knn_graph()` | Secondary graph construction | Data/embedding/KNN/graph to `fastEmbedR_graph` | CPU/Metal/CUDA; host graph; `print()` | Secondary |
| `graph_cluster()` | Secondary Louvain, Leiden, or Walktrap | Weighted graph to `fastEmbedR_graph_cluster` | Louvain/Leiden on CPU/Metal/CUDA; Walktrap CPU only; `print()` | Secondary |
| `embed_knn()` | Method dispatcher retained for compatibility | KNN/prepared state to layout matrix | CPU/Metal/CUDA | Stable compatibility; prefer method-specific `*_knn()` |

No exported function is deprecated in version 0.99.13. `reference_method` in
`landmark_tsne()` is a compatibility argument whose only accepted value is
`"tsne"`.

## Generic Names and Masking

`umap()` and `pca()` are intentionally concise but may be masked by packages
loaded later in the same R session. R resolves unqualified names according to
the search path; fastEmbedR does not alter that behavior and does not install a
startup-time masking workaround. Use qualified calls in scripts, packages, and
published analyses:

```r
fit <- fastEmbedR::umap(x, backend = "cpu")
pc <- fastEmbedR::pca(x, ncomp = 20L, backend = "cpu")
```

Package authors should use `fastEmbedR::function()` or explicit
`importFrom(fastEmbedR, function)` declarations rather than relying on attach
order. `conflicts()` and `find("umap")` can be used to inspect the active search
path in an interactive session.
