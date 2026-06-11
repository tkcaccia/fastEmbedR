# fastEmbedR 0.1.0

- Added one-call embedding wrappers: `umap()`, `tsne()`, and `pacmap()`.
- Added exact native KNN via `nn()` with CPU, Metal, and CUDA backend requests.
- Added an optional `cpu_clustered` KNN backend inspired by RAPIDS cuML's
  batched UMAP graph construction for large self-KNN workflows.
- Added native UMAP from precomputed KNN via `fast_knn_umap()` and `umap_knn()`.
- Added KNN-driven t-SNE-like and PaCMAP-like objectives via `knn_tsne()` and
  `knn_pacmap()`.
- Added an exact moderate-size CPU t-SNE optimizer from precomputed KNN
  affinities so `knn_tsne()` can be compared directly with
  `Rtsne::Rtsne_neighbors()` on the same neighbor matrices.
- Added adaptive exact t-SNE learning-rate scaling for small neighbor-input
  datasets, improving agreement with `Rtsne_neighbors()` quality metrics without
  adding user-facing tuning parameters.
- Parallelized the exact CPU t-SNE normalization and gradient loops for
  moderate-size KNN inputs.
- Matched the hidden exact t-SNE affinity perplexity to the same KNN-width rule
  used by `Rtsne::Rtsne_neighbors()` comparisons (`floor(k / 3)`, capped at 30).
- Made the high-level automatic neighborhood choice objective-aware: UMAP keeps
  the smaller size-aware default, while t-SNE now requests enough neighbors for
  its default affinity perplexity.
- Tuned the hidden t-SNE defaults toward accuracy by using more refinement
  epochs and 15 negative samples while keeping the public API unchanged.
- Added `supervised_umap()` for explicit label-guided UMAP, inspired by cuML's
  supervised `y`/`target_weight` interface.
- Added `transform_embedding()` to project query observations into an existing
  embedding from precomputed query KNN, inspired by the original UMAP
  `transform()` workflow.
- Added `backend_info()` for explicit native CPU/CUDA/Metal availability
  reporting, in the same spirit as `cuda.ml`'s explicit GPU checks but without
  adding a package dependency.
- Added `backend = "gpu"` to high-level and KNN-based embedding functions as a
  clear native-GPU request that resolves to CUDA first, then Metal, and never
  silently reports CPU work as GPU work.
- Simplified the exported API by keeping user-facing wrappers public and making
  generic helpers such as `embed()`, `auto_k()`, and `knn_embed()` internal.
- Extended the native CUDA KNN path with CUDA-runtime device reporting and
  chunked float device buffers for data, query points, and distances to reduce
  GPU memory traffic while preserving double-precision R outputs.
- Added CUDA memory preflight checks for native KNN/embedding and a cooperative
  shared-memory CUDA KNN kernel for common `k <= 64` workflows.
- Added batched query processing in native CUDA KNN so large query sets no
  longer require all query points and output buffers to fit on the GPU at once.
- Added stricter explicit CUDA build detection and a CUDA smoke-test script for
  NVIDIA machines.
- Added an opt-in native Fortran dense Euclidean KNN subroutine for development
  benchmarking; the faster C++ CPU path remains the default.
- Reduced native GPU embedding graph-preparation RAM by replacing per-row
  vector allocations with flat adjacency buffers and top-neighbor selection.
- Reduced native optimizer RAM use by storing objective graph weights and
  per-thread gradient buffers as float internally, and by replacing duplicated
  objective neighbor lookup rows with a compact CSR-style lookup.
- Added embedding quality metrics and explicit backend detection.
- Removed package-level benchmark and dataset-loader APIs so the installable
  package stays focused on embedding functions.
- Froze the retained large-data UMAP default path to the benchmark-selected
  settings: 100 epochs, negative sample rate 2, `min_dist = 0.01`, init scale
  10, graph-aware spectral iterations, and compact CSR/float graph storage.
- Made the large-data spectral initialization rule graph-aware by computing
  native KNN connected-component statistics and increasing spectral iterations
  only when low-k or fragmented graphs need the extra initialization work.
- Added documentation for each exported function, a getting-started article,
  and pkgdown/GitHub Pages scaffolding.
