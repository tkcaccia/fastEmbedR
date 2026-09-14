# fastEmbedR 0.1

- Prepare the package for CRAN submission with a portable CPU build and
  optional Metal and CUDA capabilities.
- Remove the `Biobase` dependency and its vignette-only expression-data
  example. The package accepts ordinary matrices and does not require a
  Bioconductor runtime package.
- Remove Bioconductor-specific classification fields from `DESCRIPTION`.

# fastEmbedR 0.99.17

- Regenerate the complete reference manual from authoritative roxygen source
  and restore the public help topics for `fastEmbedR_api()`, `umap()`,
  `prepare_umap_knn()`, and `tsne_knn()` after the R-source refactor.
- Document S3 print and plot methods, float32 return behavior, landmark
  preprocessing defaults, and stage-specific CPU thread handling.
- Retain `Biobase` as the package's explicit Bioconductor dependency and use
  its bundled expression example in the vignette. Fix vignette navigation and
  backend guidance.

# fastEmbedR 0.99.16

- Add `Biobase` as a genuine Bioconductor dependency and exercise
  PCA, t-SNE, and fuzzy UMAP in the executable vignette using the bundled
  `sample.ExpressionSet` expression dataset. The example uses namespaced
  access and does not add a `SingleCellExperiment` dependency.

# fastEmbedR 0.99.15

- List Stefano Cacciatore as the sole package author and maintainer.
- Clarify the build contract: CPU installation needs only C++17 and Rcpp;
  Apple accelerator frameworks are supplied by the Apple SDK; FAISS GPU and
  cuVS are linked only by CUDA nearest-neighbor builds; and RAFT/RMM are needed
  only when optional CUDA TSVD initialization is enabled. The package-native
  CPU HNSW and Metal exact/IVF-Flat implementations do not link FAISS.

# fastEmbedR 0.99.14

- Clarify in the README and introductory vignette that the Iris example uses
  the dataset supplied by R's `datasets` package; fastEmbedR does not bundle
  that dataset.

# fastEmbedR 0.99.13

- Rename the canonical public interpolation-based t-SNE API to `tsne()`,
  `tsne_knn()`, `landmark_tsne()`, and `transform_tsne()`. Related prepared-KNN
  and PCA-initialization helpers now use `prepare_tsne_knn()`,
  `tsne_pca_init()`, and `tsne_init`; result method tags and benchmark IDs use
  `tsne` consistently. Compiled implementation symbols remain private.
- Apply the Bioconductor four-space source style across R, namespace, manual,
  and vignette sources; wrap all checked lines to 80 columns; and replace
  constructed `paste0()` condition messages with direct `sprintf()` signals.
- Split the embedded Metal shader into adjacent C++ raw-string literals to
  remain within portable compiler literal-size limits without changing the
  compiled Metal source.

# fastEmbedR 0.99.12

- Build one canonical source tarball on Linux and check that identical archive
  on Linux, macOS, and Windows. This preserves the executable mode of the
  package `configure` script and matches the source-package boundary used by
  Bioconductor and r-universe.

# fastEmbedR 0.99.11

- Keep `R CMD check --as-cran` in the cross-platform CI matrix while disabling
  only its remote incoming lookup. This prevents transient Bioconductor index
  outages from being reported as package warnings; package installation,
  examples, tests, and local incoming checks remain enabled.

# fastEmbedR 0.99.10

- Make CUDA and Metal device-name capability probes handle failed system
  queries explicitly without suppressing warnings. Add regression coverage for
  unavailable and failing query commands.
- Record the maintainer's verified ORCID in `Authors@R` and document the
  disposition of BiocCheck's assay-view and dependency recommendations.
- Ensure the canonical installed API example is included in clean source
  archives and exercised by package tests.

# fastEmbedR 0.99.9

- Classify the package under the Bioconductor `Infrastructure` view. The core
  CPU implementation deliberately remains independent of other Bioconductor
  software packages; no artificial runtime dependency is introduced merely to
  suppress a submission check.

# fastEmbedR 0.99.8

- Make `n_components = 3L` operational for CPU UMAP and openTSNE. Non-2D UMAP
  now uses the float32-compatible dimension-generic CSR optimizer, and non-2D
  openTSNE resolves to exact repulsion. Metal and CUDA requests remain
  explicitly limited to two output dimensions and never fall back to CPU.
- Add a source-level provenance and licensing audit: exact upstream commits,
  adapted and vendored file mappings, SPDX source headers, complete required
  license copies, linked-versus-redistributed dependency boundaries, and a
  machine-readable `inst/THIRD_PARTY_DEPENDENCIES.json` inventory validated by
  `tools/check_provenance_inventory.R`. Remove the unused cuML build switch;
  CUDA PCA links RAFT TSVD directly.
- Document the public parameter philosophy explicitly. openTSNE exposes its
  principal scientific controls, whereas UMAP retains one package-owned,
  backend-validated optimizer policy and reports every resolved choice in the
  returned metadata. The manuals and vignettes now distinguish exposed,
  reusable, internal, and approximation controls and state that fastEmbedR is
  not a drop-in API for arbitrary UMAP hyperparameter sweeps.
- Make conventional sparse t-SNE affinity support the production default:
  `opentsne()` now supplies `ceiling(3 * perplexity)` non-self neighbors to
  the Gaussian bandwidth search. The previous `ceiling(perplexity)` policy is
  retained only as the explicit `affinity_support = "compact"` approximation.
  KNN-input and landmark workflows record the actual support width, support
  ratio, and whether the supplied support meets the conventional rule.
- Add an independent float64 t-SNE reference harness and commit-bound CPU,
  Metal, and CUDA numerical gates for exact attractive/repulsive forces,
  finite-difference gradients, FFT-grid convergence, identical-state first
  steps, common-affinity KL trajectories, support-width sweeps, and
  pathological inputs. Equal-distance rows now resolve explicitly to their
  mathematically correct uniform conditional distribution instead of allowing
  the bandwidth precision to become nonfinite.
- Align CUDA openTSNE adaptive-gain sign handling with the CPU and Metal
  implementations, including the zero-update first iteration. Explicit CUDA
  FFT-grid overrides now also accept the 32- and 64-cell diagnostic grids used
  by the cross-backend numerical tests.
- Make custom NVCC compilation inherit `R CMD config --cppflags`, so CUDA builds
  find R headers on distributions where the configured include directory is
  outside `R_HOME/include` (for example, Debian and Ubuntu).

# fastEmbedR 0.99.7

- Split KNN-input UMAP and openTSNE orchestration into dedicated policy,
  initialization, graph or affinity, backend-dispatch, optimizer, and result
  assembly helpers. This is a behavior-preserving maintainability change;
  nonlocal assignments and scattered warning suppression were removed.

# fastEmbedR 0.99.6

- Add commit-bound real-hardware CI for CPU, Metal, and CUDA. The self-hosted
  accelerator jobs reject backend fallback, run the installed-package test
  suite and native smoke benchmarks, and archive hardware metadata plus
  SHA-256 identities for source, binary, layouts, and logs.
- Export `fastEmbedR_capabilities()` as the stable public interface for native
  KNN, embedding, and clustering capability diagnostics. Public documentation
  no longer recommends the internal `backend_info()` helper.
- Route public CUDA PCA and embedding `pca_dims` preprocessing through native
  RAPIDS RAFT TSVD. Float32 input no longer materializes an intermediate R
  double matrix, scores and loadings preserve float32 storage, and unavailable
  CUDA PCA now fails explicitly instead of falling back to CPU.

# fastEmbedR 0.99.5

- Update native Metal compile options for compatibility with current macOS and
  Xcode toolchains.

# fastEmbedR 0.99.4

- Clear R's automatically injected Objective-C runtime library on Windows,
  where fastEmbedR compiles only the portable Metal stub sources.

# fastEmbedR 0.99.3

- Request Metal Shading Language 3.x explicitly for native KNN and graph
  clustering kernels that use floating-point atomics.
- Prevent non-Metal Windows builds from inheriting an Objective-C runtime link
  dependency from packaged Objective-C++ sources.

# fastEmbedR 0.99.2

- Standardizes backend selection across public backend-capable functions.
  Omitted `backend` arguments consult `options(fastEmbedR.backend)`, then
  `FASTEMBEDR_BACKEND`, before defaulting to CPU. Explicit function arguments
  always take precedence, and unavailable GPU backends fail without fallback.
- Adds `fastEmbedR_backend()` as the package-specific session backend
  setter/getter and applies the same configuration contract to
  `transform_tsne()`.

# fastEmbedR 0.99.1

- Adds a session-wide backend selector through `fastEmbedR_backend()`,
  `options(fastEmbedR.backend = ...)`, and `FASTEMBEDR_BACKEND`. Explicit
  function arguments retain precedence and CPU remains the default.
- Moves publication benchmark and validation workflows to the separate
  `fastEmbedR-benchmark` repository, together with dataset acquisition and
  restricted-data instructions. Raw benchmark data and manuscript files are
  not distributed in either GitHub repository.
- Reduces the vignette-enabled source archive to approximately 1 MB by
  excluding benchmark, manuscript, container, and generated-result artifacts.
- Updates GitHub Pages and checkout workflows to Node 24-compatible action
  releases.

# fastEmbedR 0.99.0

- Makes the standard fuzzy UMAP graph the default for `umap()`, `umap_knn()`,
  UMAP initialization, and landmark UMAP. Binary weighting remains available
  only as the explicit `graph_mode = "binary"` sensitivity mode.
- Standardizes CPU parallelism controls on the public argument `n.cores`.
  Low-level `n_threads` names are now private implementation details; existing
  user scripts should replace `n_threads =` with `n.cores =`.
- Provides native UMAP and openTSNE-style optimizers for CPU, Apple Metal, and
  optional CUDA builds, with explicit backend reporting and no silent GPU to
  CPU fallback.
- Adds package-native Louvain and Leiden clustering for CPU, CUDA, and Metal.
  Accelerator local-moving and refinement phases use float32 compressed sparse
  row graphs and atomic community updates; package-owned C++ performs label
  compaction and graph coarsening between levels. No cuGraph source, library,
  Python module, or runtime symbol is required. Exact Pons-Latapy Walktrap
  remains CPU-only, and unsupported accelerator requests fail explicitly.
- Adds package-native float32 CPU HNSW, Apple Metal exact/recall-tuned IVF-Flat,
  direct FAISS GPU exact search, and direct RAPIDS cuVS CUDA IVF-Flat for one-call
  embeddings. CUDA results remain device-resident through graph or affinity
  construction and optimization; no faissR or CPU fallback is used.
- Distils the CUDA KNN adapter from the MIT-licensed faissR implementation into
  fastEmbedR. Exact search calls installed FAISS GPU directly; approximate
  search calls the installed Apache-2.0 cuVS C API. The package strips
  self-neighbours and packs int32/float32 output on device, and bounds IVF raw
  search storage to 32,768-query batches. `umap_knn()` and `opentsne_knn()`
  continue to accept reusable KNN results from any compatible provider.
- Validates CUDA IVF tuning against an exact cuVS pilot of evenly spaced rows
  and expands `nprobe` until the requested recall tier is reached; a failed
  target is never silently reported as tuned.
- Speeds the native Metal IVF path with a two-stage GPU search: a
  128-dimensional projected list scan builds an adaptive 288/384/512-candidate
  shortlist, followed by exact full-dimensional reranking. A four-stratum
  deterministic pilot selects both shortlist size and `nprobe`; direct
  reranking remains an internal safety fallback.
- Adds float32 input and output handling. Native graph weights, affinities,
  layouts, gradients, and optimizer buffers use float32; a double layout is
  returned only when the input was double.
- Replaces the R-orchestrated CPU and Metal RSVD initialization paths with
  package-native float32 implementations. CPU centers once into contiguous
  float32 storage, uses blocked SGEMM/subspace products, and retains only skinny
  QR and projected-SVD work at the ordinary R numerical boundary. Metal keeps
  the centered input, basis, projected block, loadings, and scores in one
  resident MPS workspace. Public CUDA `pca()` uses fastEmbedR's native RSVD;
  resident CUDA openTSNE initialization uses RAPIDS RAFT TSVD.
- Removes the optional fastPLS PCA delegation and `Enhances` dependency.
  CPU, Metal, and CUDA PCA ownership is now explicit and invariant to the
  packages installed in the user's library.
- Batches four openTSNE iterations per Metal command buffer, reducing command
  submission overhead without changing the optimization objective or schedule.
- Avoids allocating and uploading the padded `epochs_per_sample` buffer for
  the default Metal UMAP sampler, which does not read that schedule. The graph,
  edge sampling, random stream, and optimizer updates are unchanged.
- Corrects a race in a legacy multithreaded CPU UMAP optimizer path by using
  atomic coordinate updates.
- Replaces the former approximate structure score with the standard
  trustworthiness and continuity definitions, adds exact sampled neighbour-rank
  metrics, and samples before quadratic work to bound evaluation memory.
- Makes evaluation cache keys data-aware and removes self neighbours by row
  identity rather than assuming that self is always the first KNN column.
- Correctly decodes `float::float32` layouts before plotting and scoring.
- Preserves the caller's R random-number-generator state in seeded sampling,
  initialization, landmark selection, and evaluation helpers.
- Strengthens installed-package tests so optional CUDA tests skip individually
  while CPU, Metal, float32, and backend-contract tests continue to run.
