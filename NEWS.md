# fastEmbedR 0.1

* Report landmark reference-stage memory ratios and query-to-reference KNN
  payload estimates separately from timing. These estimates distinguish
  optimizer memory reduction from total peak RSS.
* Reduce CUDA setup and transfer costs with the CUDA asynchronous default
  memory pool, thread-local CUDA library handles, cached cuFFT plans, and CUDA
  Graph execution. One-call CUDA t-SNE now reuses the feature matrix retained
  for KNN as device-resident PCA input instead of uploading a second copy.
* Use package-native exhaustive float32 CPU KNN below 5,000 observations and
  HNSW otherwise. The exact route supports Euclidean, cosine, and correlation
  distances, deterministic multithreaded top-k selection, and
  query-to-reference search without linking FAISS or importing faissR.
* Remove the generic `embed_knn()` dispatcher, API/capability inventory
  helpers, KNN print method, backend setter/getter, and staged landmark fitter.
  Use `tsne_knn()` or `umap_knn()` explicitly and configure session defaults
  with `options(backend = ..., n.cores = ...)`.
* Remove `tsne_pca_init()`. Use `pca(..., tsne_init = TRUE)$tsne_init` so one
  PCA call returns both the ordinary fit and the t-SNE-ready initialization.
* Remove the `inner_product` KNN metric and its CUDA-specific implementation.
  The supported distance metrics are Euclidean, cosine, and correlation.
* Integrate landmark workflows into `tsne()` and `umap()`. Landmarking is off
  by default and is enabled with a fraction, count, or explicit row indices in
  the `landmarks` argument; the separate landmark wrappers are removed.
* Removed unused internal PCA aliases and the obsolete host-returning RAFT
  TSVD entry point. Repository rules and CI now reject oversized R functions,
  long authored lines, hidden fallbacks, and caller-free routes.
* Clarified the installation tiers in `SystemRequirements`. CPU builds need
  only the R C++17 toolchain; Apple frameworks come from the Apple SDK; CUDA
  embedding and native rSVD use the CUDA toolkit, while cuVS and RAFT/RMM are
  separate optional capabilities.
* Removed the optional FAISS GPU linkage and exact-search branch. CUDA exact
  and IVF-Flat KNN now use one direct RAPIDS cuVS implementation.
* Removed the accidental RAFT compile-time gate from package-native CUDA rSVD
  and device-resident t-SNE PCA initialization. RAFT remains optional and is
  used only when its TSVD route is available and selected.
* CUDA configuration now uses the vendored DLPack C ABI header for cuVS
  probes instead of requiring a duplicate external DLPack installation.

- Prepare the package for CRAN submission with a portable CPU build and
  optional Metal and CUDA capabilities.
- Remove the `Biobase` dependency and its vignette-only expression-data
  example. The package accepts ordinary matrices and does not require a
  Bioconductor runtime package.
- Remove Bioconductor-specific classification fields from `DESCRIPTION`.
- Use the same automatic t-SNE update-norm limit on CPU, Metal, and CUDA.
  This removes a Metal-only clipping policy that could impair long-run
  convergence, and adds an exact-objective regression test for sustained
  FFT-grid optimization.
- Apply `FASTEMBEDR_CUDA_FLAGS` consistently to CUDA toolkit and RAPIDS RAFT
  configure probes as well as package translation units. This supports custom
  CCCL and CUDA toolchain include layouts without weakening strict detection.
- Add native float32 CUDA rSVD adapted from the permissively licensed fastPLS
  implementation. CUDA PCA and t-SNE initialization now select rSVD for
  sufficiently wide, low-rank matrices and RAFT TSVD otherwise.
- Use package-native float32 block-subspace rSVD for Metal PCA and t-SNE
  initialization, with MPS matrix products and rank-aware power iterations.
- Report the PCA method actually selected by GPU-resident CUDA t-SNE while
  retaining the requested automatic policy as separate metadata.

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
  Apple accelerator frameworks are supplied by the Apple SDK; cuVS is linked
  only by CUDA nearest-neighbor builds; and RAFT/RMM are needed only when
  optional CUDA TSVD initialization is enabled. The package-native CPU HNSW
  and Metal exact/IVF-Flat implementations do not link FAISS.

# fastEmbedR 0.99.14

- Clarify in the README and introductory vignette that the Iris example uses
  the dataset supplied by R's `datasets` package; fastEmbedR does not bundle
  that dataset.

# fastEmbedR 0.99.13

- Rename the canonical public interpolation-based t-SNE API to `tsne()`,
  `tsne_knn()`, `landmark_tsne()`, and `transform_tsne()`. Related prepared-KNN
  helpers now use `prepare_tsne_knn()` and `tsne_init`; result method tags and
  benchmark IDs use `tsne` consistently. Compiled implementation symbols
  remain private.
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
- Document the public parameter philosophy explicitly. t-SNE exposes its
  principal scientific controls, whereas UMAP retains one package-owned,
  backend-validated optimizer policy and reports every resolved choice in the
  returned metadata. The manuals and vignettes now distinguish exposed,
  reusable, internal, and approximation controls and state that fastEmbedR is
  not a drop-in API for arbitrary UMAP hyperparameter sweeps.
- Use compact t-SNE affinity support consistently: `tsne()` supplies
  `ceiling(perplexity)` non-self neighbors to the Gaussian bandwidth search.
  Matrix, KNN-input, landmark, and transformation workflows record the actual
  support width and support-to-perplexity ratio.
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
  `fastEmbedR-extra` repository, together with dataset acquisition and
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
- Adds package-native float32 CPU HNSW, Apple Metal exact/recall-tuned
  IVF-Flat, and direct RAPIDS cuVS CUDA exact/IVF-Flat for one-call embeddings.
  CUDA results remain device-resident through graph or affinity construction
  and optimization; no faissR or CPU fallback is used.
- Distils the CUDA KNN adapter from the MIT-licensed faissR implementation into
  fastEmbedR. Exact and approximate search call the installed Apache-2.0 cuVS
  C API. The package strips
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
