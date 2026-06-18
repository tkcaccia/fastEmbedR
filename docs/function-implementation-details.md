# Function Implementation Details And Literature

This page explains how each public `fastEmbedR` function is implemented, which
native paths it can use, and which papers or software projects inspired the
implementation. It complements the shorter
[implementation inventory](implementation-inventory.md) and the formal
[algorithmic references](../inst/ALGORITHMIC_REFERENCES.md).

The guiding rule is simple: public UMAP and openTSNE functions use native
package code or explicit native library links. They do not call Python, and
they do not silently run on CPU while reporting a GPU backend.

## Public Functions

| Function | Main implementation | Native backends | Main inspiration |
| --- | --- | --- | --- |
| `faissR::nn()` | Companion-package KNN provider consumed by fastEmbedR | FAISS CPU and optional CUDA/cuVS through `faissR` | FAISS, RAPIDS cuVS, KNN-first embedding workflow |
| `umap_knn()` | UMAP directly from supplied KNN | CPU, Metal, CUDA when compiled | UMAP algorithm, `uwot` benchmark behaviour, RAPIDS/cuML GPU-resident design |
| `umap()` | `faissR` KNN, then `umap_knn()` | CPU, Metal, CUDA when compiled | Same as `umap_knn()` plus KNN-first workflow |
| `embed_knn()` | Small dispatcher for KNN-input embedding | CPU, Metal, CUDA when available | KNN reuse across UMAP/openTSNE |
| `opentsne_knn()` | openTSNE-style t-SNE directly from KNN | CPU, Metal, CUDA when compiled | openTSNE, FIt-SNE, t-SNE-CUDA, Rtsne_neighbors, opt-SNE |
| `opentsne()` | `faissR` KNN, PCA init, then `opentsne_knn()` | CPU, Metal, CUDA when compiled | openTSNE and opt-SNE workflows |
| `transform_tsne()` | Fixed-reference openTSNE-style query transform | CPU, Metal | openTSNE transform, t-SNE-CUDA GPU residency |
| `landmark_tsne()` | Embed landmarks, project/transform remaining points | CPU, Metal | openTSNE transform, landmark t-SNE workflows |
| `landmark_umap()` | Embed landmarks, project/refine remaining points | CPU, Metal | UMAP transform/landmark workflow |
| `evaluate_embedding()` | Embedding quality metrics | CPU | Trustworthiness, neighbour preservation, silhouette, label KNN accuracy |
| `faissR::backend_info()` | faissR backend detection and reporting | CPU, CUDA, FAISS, cuVS | Explicit backend reporting, no silent fallback |
| `faissR::metal_available()` | faissR Metal KNN probe where supported | Metal | Apple Metal |
| `faissR::cuda_available()` | faissR CUDA KNN probe | CUDA | CUDA runtime/build probes |
| `faissR::faiss_available()` | FAISS availability probe in faissR | FAISS | FAISS |
| `faissR::cuvs_available()` | cuVS availability probe in faissR | RAPIDS cuVS | RAPIDS cuVS |

## `faissR::nn()`

`faissR::nn()` lives in the companion `faissR` package. `fastEmbedR` imports
and calls it internally for one-call embeddings, but does not re-export it.
It accepts a reference matrix, optional query
matrix, `k`, a backend, and `n_threads`.

Implementation:

- Public KNN calls should use `faissR::nn()` directly.
- The returned object is reused by `umap_knn()`, `opentsne_knn()`,
  landmarking, metrics, and graph construction.
- Backend selection, FAISS/cuVS linkage, candidate KNN, and KNN diagnostics are
  documented and implemented in `faissR`.

Native paths:

- FAISS CPU indexes through `faissR`.
- Optional CUDA/cuVS or FAISS GPU indexes through `faissR`.
- Removed from `fastEmbedR`: old native CPU/Metal NN-descent and grid KNN
  experiments. They were useful during development, but the cleaned package
  keeps KNN ownership in `faissR`.

Distance metrics:

- `metric = "euclidean"` is the validated default for UMAP/openTSNE.
- Cosine/inner-product support is available where the installed `faissR`
  backend enables it. Users should normalize rows before treating inner product
  as cosine similarity.
- Unsupported metric/backend combinations fail clearly instead of silently
  returning neighbours computed under a different metric.

Inspiration and literature:

- FAISS: Johnson, Douze, and Jegou, "Billion-scale similarity search with
  GPUs", IEEE Transactions on Big Data 2019, originally arXiv:1702.08734.
- RAPIDS cuVS and cuML: NVIDIA/RAPIDS GPU nearest-neighbour and UMAP
  engineering references.
- KNN-first workflow: the original benchmark requirement to separate KNN time
  from UMAP/openTSNE embedding time.

## `umap_knn()`

`umap_knn()` is the main UMAP entry point when KNN is already available. This
is the most important path for fair benchmarking because the neighbour graph
can be reused across methods.

Implementation:

- Accepts an `faissR::nn()` result or explicit `indices` and `distances`.
- Normalizes KNN input, removes self-neighbour columns where needed, and
  records the input backend.
- Uses native C++ to derive UMAP defaults from the KNN distance profile.
- Builds a compact fuzzy graph, stores graph data in CSR/COO-style native
  buffers, and runs UMAP SGD.
- Implements the documented UMAP mathematics with package-local code: smooth
  KNN distances, fuzzy graph union, `epochs_per_sample`, negative sampling,
  and learning-rate decay.
Backend details:

- CPU: native C++ CSR fuzzy graph and epoch-scheduled stochastic optimizer.
- Metal: native Objective-C++/Metal `atomic_inplace` optimizer. The public
  Metal UMAP path has only this validated optimizer mode.
- CUDA: native CUDA fused pure-atomic UMAP path when compiled. Old hybrid and
  deterministic CUDA variants were removed from the public code.

Autotuning:

- `umap_auto_parameters_cpp()` inspects the KNN distance profile and chooses
  internal values such as epochs, negative sample rate, learning rate,
  spectral iterations, initialization scale, and min-distance policy.
- Connectivity and low-k spectral rules adjust initialization effort when the
  graph is brittle.
- A pilot tuner can run on a small sample when KNN is not supplied, then choose
  a safer KNN/UMAP configuration. Supplied-KNN runs avoid changing the user's
  KNN graph.

Inspiration and literature:

- UMAP: McInnes, Healy, and Melville, "UMAP: Uniform Manifold Approximation
  and Projection for Dimension Reduction", arXiv:1802.03426, 2018.
- `uwot`: Jim Melville's R implementation is used as an external benchmark
  and behavioural reference; its GPL source is not copied or required by the
  package core.
- RAPIDS cuML UMAP engineering notes for GPU-resident graph/optimizer design.
- mlx-vis as Metal design reference for GPU-resident KNN and optimizer
  structure.

## `umap()`

`umap()` is the one-call UMAP interface.

Implementation:

- Validates and optionally standardizes data.
- Computes KNN with a fixed backend policy when KNN is not supplied: CPU and
  Metal use FAISS CPU IVF-Flat through `faissR`; CUDA uses FAISS GPU IVF-Flat.
- Passes the resulting KNN object to `umap_knn()`.
- Stores timing/configuration metadata in the returned fit object.

Autotuning:

- If `n_neighbors` is omitted, `auto_embedding_k()` chooses a size-aware K.
- `umap_auto_parameters_cpp()` derives internal UMAP settings from the KNN
  distance profile. The one-call API no longer accepts labels or scoring
  samples; benchmark quality is computed separately with `evaluate_embedding()`.

Inspiration:

- KNN-first benchmark design from the user's original UMAP workflow.
- UMAP defaults and epoch-scheduled stochastic optimization behaviour.
- opt-SNE-style philosophy: reduce user parameter burden with internal
  data-aware choices.

## `embed_knn()`

`embed_knn()` is a small dispatcher for precomputed KNN inputs.

Implementation:

- `method = "umap"` dispatches to `umap_knn()`.
- `method = "opentsne"` dispatches to `opentsne_knn()`.
- Keeps the user-facing KNN-first workflow compact.

Inspiration:

- Shared KNN graph benchmarking: compute neighbours once, reuse them for UMAP,
  t-SNE/openTSNE, landmarking, and quality checks.

## `opentsne_knn()`

`opentsne_knn()` is the main native openTSNE-style t-SNE path from supplied
KNN.

Implementation:

- Accepts an `faissR::nn()` result or explicit KNN matrices.
- Computes row-wise perplexity affinities from supplied neighbour distances.
- Symmetrizes sparse high-dimensional probabilities.
- Uses a two-phase optimizer: early exaggeration followed by normal
  optimization.
- Supports PCA initialization through `init_data` or explicit `Y_init`.
- Uses gains, momentum, auto learning rate, max-step clipping, centering, and
  sparse attractive forces.

Negative-gradient paths:

- CPU exact: exact pairwise repulsive normalization for small/reference runs.
- CPU FFT: native grid/FIt-SNE-style approximation for large runs.
- Metal FFT: native Objective-C++/Metal grid scatter, FFT convolution, gather,
  sparse attraction, gains, and update kernels.
- CUDA FFT: native CUDA kernels plus cuFFT when compiled.
- Barnes-Hut and sampled full-embedding GPU repulsion are not public options
  because they either lost in the current benchmarks or changed the intended
  openTSNE mathematics.

Autotuning:

- `auto_tsne_perplexity()` keeps KNN-input calls inside safe sample-size and
  available-neighbour limits when perplexity is omitted.
- `resolve_opentsne_auto_parameters()` calls native C++ opt-SNE-style logic.
- `learning_rate = "auto"` resolves to the opt-SNE rule
  `n / early_exaggeration`.
- CPU/small exact runs can use KLD-based auto-stopping. Large FFT/GPU runs do
  not hide an O(n^2) CPU KLD monitor.

Inspiration and literature:

- t-SNE: van der Maaten and Hinton, "Visualizing Data using t-SNE", JMLR 2008.
- Barnes-Hut t-SNE: van der Maaten, "Accelerating t-SNE using Tree-Based
  Algorithms", JMLR 2014.
- openTSNE: Policar, Strazar, and Zupan, "openTSNE: a modular Python library
  for t-SNE dimensionality reduction and embedding", JOSS 2019.
- FIt-SNE: Linderman et al., "Fast interpolation-based t-SNE for improved
  visualization of single-cell RNA-seq data", Nature Methods 2019.
- t-SNE-CUDA: Chan, Rao, Huang, and Canny, "t-SNE-CUDA: GPU-Accelerated t-SNE
  and its Applications to Modern Data", arXiv:1807.11824, 2018.
- Rtsne: `Rtsne_neighbors()` informed KNN input validation and parameter
  compatibility.
- opt-SNE: Belkina et al., "Automated optimized parameters for t-SNE improve
  visualization and analysis of large datasets", Nature Communications 2019.

## `opentsne()`

`opentsne()` is the one-call openTSNE-style interface.

Implementation:

- Validates and optionally standardizes input.
- Computes KNN with the fixed one-call policy unless a KNN object is supplied:
  CPU/Metal use FAISS CPU IVF-Flat and CUDA uses FAISS GPU IVF-Flat.
  The one-call API uses `ceiling(perplexity)` non-self neighbours internally;
  `n_neighbors` is intentionally not a public `opentsne()` argument.
- Computes PCA initialization by default when original data are available.
- Calls `opentsne_knn()` for the native optimizer.
- Stores configuration, timing, and backend metadata.

Autotuning:

- Same opt-SNE-style parameter selection as `opentsne_knn()`.
- Uses PCA initialization by default because the MNIST benchmark showed better
  visual stability without changing the t-SNE objective.

Inspiration:

- openTSNE API structure and transform workflow.
- opt-SNE parameter automation.
- fastPLS-style randomized PCA/SVD ideas for efficient initialization.

## `transform_tsne()`

`transform_tsne()` places new/query points into an existing openTSNE-style
embedding while keeping the reference embedding fixed.

Implementation:

- Finds query-to-reference neighbours.
- Initializes query points from reference-layout neighbours.
- Converts query distances to t-SNE conditional probabilities once.
- Optimizes query coordinates against fixed reference coordinates.
- CPU path is parallelized by query row where safe.
- Metal path keeps query/reference layout, probabilities, gains, and updates
  in device buffers and returns only the final query layout.

Inspiration:

- openTSNE `TSNEEmbedding.transform`.
- t-SNE-CUDA GPU-resident fixed-reference optimization ideas.

## `landmark_tsne()`

`landmark_tsne()` embeds a subset of rows, then projects/transforms the
remaining rows.

Implementation:

- Selects landmarks from a count, fraction, explicit row indices, or automatic
  policy.
- Runs `opentsne()` on landmarks.
- Uses `transform_tsne()` to place non-landmarks into the fixed landmark
  embedding.
- Reports embedding and projection/transform timing separately.

Inspiration:

- openTSNE transform workflow.
- Landmark/subsampling strategies from scalable t-SNE and manifold-learning
  literature.
- annembed and related ANN embedding ideas for landmark/projection workflows.

## `landmark_umap()`

`landmark_umap()` is the UMAP landmark workflow.

Implementation:

- Selects landmarks from a count, fraction, explicit row indices, or automatic
  policy.
- Runs `umap()` on landmarks.
- Projects non-landmarks from high-dimensional neighbours into the UMAP layout.
- Supports native Metal projection/refinement helpers when available.
- Landmark runs are labelled as approximations in benchmarks.

Autotuning:

- Uses the same UMAP policy layer as `umap()`.
- Automatic landmark choice uses size-aware rules and can choose refinement
  strength from pilot scoring.

Inspiration:

- UMAP transform/landmark usage patterns.
- PaCMAP/openTSNE-style idea of separating expensive embedding from cheaper
  projection/refinement for large datasets.

## `evaluate_embedding()`

`evaluate_embedding()` scores an embedding against high-dimensional input and
optional labels/batches.

Implementation:

- Computes trustworthiness.
- Computes neighbour preservation at requested K values.
- Computes silhouette and KNN label accuracy when labels are supplied.
- Computes batch-aware quantities where batch labels are supplied.
- Uses subsampling for expensive global metrics to avoid all-pairs memory
  blowups on large datasets.

Inspiration:

- Trustworthiness and continuity metrics from manifold-learning evaluation.
- kNN preservation and label KNN accuracy used throughout single-cell and
  embedding benchmark literature.
- Silhouette score for label-aware visual cluster separation.

## Backend Detection Functions

`faissR::backend_info()`, `faissR::metal_available()`, `faissR::cuda_available()`,
`faissR::faiss_available()`, and `faissR::cuvs_available()` are lightweight probes.

Implementation:

- Check compiled embedding symbols and optional native library availability.
- Report CPU, Metal, CUDA, FAISS, and cuVS status without crashing when a
  backend is absent. FAISS/cuVS probes are forwarded to `faissR`.
- Keep explicit backend requests honest: unavailable GPU/FAISS/cuVS requests
  fail rather than silently falling back to CPU.

Inspiration:

- Reproducible benchmark practice: every row must include requested backend,
  used backend, status, and error message when unavailable.

## Internal Autotuning Summary

Autotuning is intentionally internal. The user can still override important
parameters, but the default API is small.

UMAP autotuning:

- `auto_embedding_k()` chooses a size-aware K when KNN is not supplied.
- `umap_auto_parameters_cpp()` scores the supplied KNN distance profile and
  chooses epochs, learning rate, negative sample rate, min-distance policy,
  spectral iterations, and initialization scale.
- `auto_umap_pilot_tune()` can run a 2,000 to 5,000 row pilot when data are
  supplied and KNN is not fixed by the user.
- Pilot scoring uses local structure preservation and, when labels are
  available, label-aware quality proxies.

openTSNE autotuning:

- `auto_tsne_perplexity()` chooses a safe perplexity from `n` and KNN width.
- `tsne_auto_parameters_cpp()` and `resolve_opentsne_auto_parameters()` apply
  opt-SNE-style learning-rate and iteration policy.
- PCA initialization is default when the original data are available.

Landmark autotuning:

- `auto_select_landmark_policy()` chooses whether and how to use landmarks
  based on size, mode, and whether KNN is supplied.
- Projection/refinement strength can be selected from pilot scores.

## Literature And Software References

Core manifold learning:

- McInnes L, Healy J, Melville J. "UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction." arXiv:1802.03426, 2018.
- van der Maaten L, Hinton G. "Visualizing Data using t-SNE." JMLR, 2008.
- van der Maaten L. "Accelerating t-SNE using Tree-Based Algorithms." JMLR,
  2014.
- Linderman GC et al. "Fast interpolation-based t-SNE for improved
  visualization of single-cell RNA-seq data." Nature Methods, 2019.
- Belkina AC et al. "Automated optimized parameters for t-SNE improve
  visualization and analysis of large datasets." Nature Communications, 2019.
- Chan DM, Rao R, Huang F, Canny J. "t-SNE-CUDA: GPU-Accelerated t-SNE and
  its Applications to Modern Data." arXiv:1807.11824, 2018.

Nearest-neighbour search:

- Dong W, Moses C, Li K. "Efficient K-Nearest Neighbor Graph Construction for
  Generic Similarity Measures." WWW, 2011.
- Johnson J, Douze M, Jegou H. "Billion-scale similarity search with GPUs."
  IEEE Transactions on Big Data, 2019.

Software projects studied or used:

- uwot: <https://github.com/jlmelville/uwot>
- UMAP reference implementation: <https://github.com/lmcinnes/umap>
- openTSNE: <https://github.com/pavlin-policar/openTSNE>
- Rtsne: <https://github.com/jkrijthe/Rtsne>
- Multicore-opt-SNE: <https://github.com/omiq-ai/Multicore-opt-SNE>
- t-SNE-CUDA: <https://github.com/CannyLab/tsne-cuda>
- FAISS: <https://github.com/facebookresearch/faiss>
- RAPIDS cuVS: <https://github.com/rapidsai/cuvs>
- RAPIDS cuML: <https://github.com/rapidsai/cuml>
- mlx-vis: <https://github.com/hanxiao/mlx-vis>
- annembed: <https://github.com/jean-pierreBoth/annembed>
- AppleSiliconFFT: <https://github.com/aminems/AppleSiliconFFT>
- fastPLS: <https://github.com/tkcaccia/fastPLS>
- KeOps: <https://www.kernel-operations.io/keops/>
- TorchDR: <https://github.com/TorchDR/TorchDR>
