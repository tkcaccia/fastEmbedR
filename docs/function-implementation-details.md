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
| `nn()` | Native exact and approximate KNN dispatcher | CPU, Metal, optional FAISS, optional CUDA/cuVS | NN-descent, FAISS, RAPIDS cuVS, mlx-vis, annembed, Rnanoflann-style API |
| `umap_knn()` | UMAP directly from supplied KNN | CPU, Metal, CUDA when compiled | UMAP, uwot fast-SGD, RAPIDS/cuML GPU-resident design |
| `umap()` | Data preprocessing, KNN, then `umap_knn()` | CPU, Metal, CUDA when compiled | Same as `umap_knn()` plus KNN-first workflow |
| `embed_knn()` | Small dispatcher for KNN-input embedding | CPU, Metal, CUDA when available | KNN reuse across UMAP/openTSNE |
| `opentsne_knn()` | openTSNE-style t-SNE directly from KNN | CPU, Metal, CUDA when compiled | openTSNE, FIt-SNE, t-SNE-CUDA, Rtsne_neighbors, opt-SNE |
| `opentsne()` | Data preprocessing, KNN, PCA init, then `opentsne_knn()` | CPU, Metal, CUDA when compiled | openTSNE and opt-SNE workflows |
| `transform_tsne()` | Fixed-reference openTSNE-style query transform | CPU, Metal | openTSNE transform, t-SNE-CUDA GPU residency |
| `landmark_tsne()` | Embed landmarks, project/transform remaining points | CPU, Metal | openTSNE transform, landmark t-SNE workflows |
| `landmark_umap()` | Embed landmarks, project/refine remaining points | CPU, Metal | UMAP transform/landmark workflow |
| `evaluate_embedding()` | Embedding quality metrics | CPU | Trustworthiness, neighbour preservation, silhouette, label KNN accuracy |
| `knn_graph()` | Native graph builder from KNN or embedding output | CPU | bluster/scran_graph_cluster SNN construction, igraph graph output |
| `backend_info()` | Backend detection and reporting | CPU, Metal, CUDA, FAISS, cuVS | Explicit backend reporting, no silent fallback |
| `metal_available()` | Metal availability probe | Metal | Apple Metal |
| `cuda_available()` | CUDA availability probe | CUDA | CUDA runtime/build probes |
| `faiss_available()` | FAISS bridge availability probe | FAISS | FAISS |
| `cuvs_available()` | cuVS bridge availability probe | RAPIDS cuVS | RAPIDS cuVS |

## `nn()`

`nn()` is the single nearest-neighbour API. It accepts a reference matrix,
optional query matrix, `k`, a backend, and `n_threads`.

Implementation:

- Converts input once to numeric matrix form and validates dimensions, finite
  values, `k`, and self-neighbour exclusion.
- Chooses a backend from explicit user choice or `backend = "auto"`.
- Returns a list with `indices` and `distances`, plus backend/approximation
  attributes used by UMAP, openTSNE, and benchmarks.
- Can attach approximate KNN recall diagnostics when
  `options(fastEmbedR.gpu_approx_recall = TRUE)`.

Native paths:

- `cpu`: exact native C++ row-distance KNN.
- `cpu_nndescent`: native C++ approximate NN-descent for self-KNN.
- `metal_nndescent`: native Objective-C++/Metal approximate NN-descent.
- `metal_grid`: older native Metal grid-candidate KNN path, kept out of the
  default benchmark path.
- `faiss` and `faiss_ivf`: native C++ bridge to an external FAISS install.
- `cuda_cuvs`, `cuda_cuvs_bruteforce`, `cuda_cuvs_cagra`, and
  `cuda_cuvs_nndescent`: native C++ bridge to external RAPIDS cuVS.

Distance metrics:

- `metric = "euclidean"` is the validated default and is used by all high-speed
  CPU/GPU/FAISS/cuVS KNN paths.
- `metric = "cosine"` is implemented in the exact C++ CPU path. It returns
  cosine distance `1 - cos(x, y)`, treats two zero vectors as distance 0, and
  treats a zero vector compared with a non-zero vector as distance 1.
- Approximate, Metal, CUDA, FAISS, and cuVS cosine paths are intentionally not
  enabled yet. They error clearly instead of silently returning Euclidean
  neighbours with a cosine label.

Inspiration and literature:

- NN-descent: Dong, Moses, and Li, "Efficient K-Nearest Neighbor Graph
  Construction for Generic Similarity Measures", WWW 2011.
- FAISS: Johnson, Douze, and Jegou, "Billion-scale similarity search with
  GPUs", IEEE Transactions on Big Data 2019, originally arXiv:1702.08734.
- RAPIDS cuVS and cuML: NVIDIA/RAPIDS GPU nearest-neighbour and UMAP
  engineering references.
- mlx-vis: Apache-2.0 design reference for NN-descent scheduling and
  GPU-resident pipelines.
- annembed: MIT/Apache-2.0 design reference for ANN/embedding graph ideas.
- Rnanoflann: API and exact-KNN benchmark reference, not vendored.

## `umap_knn()`

`umap_knn()` is the main UMAP entry point when KNN is already available. This
is the most important path for fair benchmarking because the neighbour graph
can be reused across methods.

Implementation:

- Accepts an `nn()` result or explicit `indices` and `distances`.
- Normalizes KNN input, removes self-neighbour columns where needed, and
  records the input backend.
- Uses native C++ to derive UMAP defaults from the KNN distance profile.
- Builds a compact fuzzy graph, stores graph data in CSR/COO-style native
  buffers, and runs UMAP SGD.
- Preserves uwot-like UMAP mathematics: smooth KNN distances, fuzzy graph
  union, `epochs_per_sample`, negative sampling, and learning-rate decay.
Backend details:

- CPU: native C++ CSR fuzzy graph and fast-SGD-style optimizer.
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
- uwot: Jim Melville's R implementation, especially fuzzy graph and
  fast-SGD-compatible behaviour.
- RAPIDS cuML UMAP engineering notes for GPU-resident graph/optimizer design.
- mlx-vis as Metal design reference for GPU-resident KNN and optimizer
  structure.

## `umap()`

`umap()` is the one-call UMAP interface.

Implementation:

- Validates and optionally standardizes data.
- Selects or computes KNN with `nn()`.
- Passes the resulting KNN object to `umap_knn()`.
- Stores timing/configuration metadata in the returned fit object.

Autotuning:

- If `n_neighbors` is omitted, `auto_embedding_k()` chooses a size-aware K.
- If the data are large enough and KNN was not supplied, the pilot tuner can
  sample 2,000 to 5,000 rows, test a small candidate set, and choose K/UMAP
  settings using quality proxies.
- If labels are available in benchmark/pilot contexts, label-aware proxies can
  contribute to the pilot score. Without labels, neighbour preservation and
  continuity-like proxies are used.

Inspiration:

- KNN-first benchmark design from the user's original UMAP workflow.
- UMAP/uwot defaults and fast-SGD behaviour.
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

## `knn_graph()`

`knn_graph()` converts either a precomputed `nn()` result or an embedding object
returned by `opentsne()` / `umap()` into a plain `igraph` graph.

Implementation:

- If the input is an `nn()` result, no neighbour search is repeated. The graph
  is built on the original data-space neighbours already stored in the object.
- If the input is an embedding fit, neighbours are computed on the visible
  two-dimensional layout and the graph is built in embedding space.
- `weight = "auto"` uses full shared-nearest-neighbour Jaccard weights for
  original-data KNN objects and distance weights for embedding-layout graphs.
- The full SNN path builds an inverted neighbour index and counts sparse
  shared-neighbour co-occurrences in C++, then creates an undirected `igraph`
  object with numeric `weight` edge attributes.
- `mutual = TRUE` keeps the stricter reciprocal direct-KNN interpretation for
  users who want a smaller boundary-focused graph.

Inspiration and provenance:

- `bluster::makeSNNGraph()` and `neighborsToSNNGraph()` were used as the
  behavioural reference for full SNN graph semantics: connect all pairs of
  observations that share at least one neighbour and use Jaccard weighting.
- `scran_graph_cluster::build_snn_graph()` informed the sparse
  neighbour-incidence construction strategy.
- The implementation in `src/graph.cpp` is native fastEmbedR C++ code. It does
  not link to or call `bluster` at runtime.

## `opentsne_knn()`

`opentsne_knn()` is the main native openTSNE-style t-SNE path from supplied
KNN.

Implementation:

- Accepts an `nn()` result or explicit KNN matrices.
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

- `auto_tsne_perplexity()` uses `min(30, floor((n - 1) / 3), floor(k / 3))`.
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
- Computes KNN using `nn()` unless a KNN object is supplied.
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

`backend_info()`, `metal_available()`, `cuda_available()`,
`faiss_available()`, and `cuvs_available()` are lightweight probes.

Implementation:

- Check compiled symbols and optional native library availability.
- Report CPU, Metal, CUDA, FAISS, and cuVS status without crashing when a
  backend is absent.
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
