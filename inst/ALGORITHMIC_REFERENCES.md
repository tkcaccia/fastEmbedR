# Algorithmic References And Provenance

This file records implementation ideas studied for `fastEmbedR`. It is meant
to keep future code changes traceable, especially when an idea comes from a
permissively licensed project.

## Graph construction and community detection

- Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
  communities in large networks. J Stat Mech. 2008;2008:P10008.
- Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
  well-connected communities. Sci Rep. 2019;9:5233.
- Pons P, Latapy M. Computing communities in large networks using random
  walks. J Graph Algorithms Appl. 2006;10:191-218.
- NetworKit repository: <https://github.com/networkit/networkit>, commit
  `7b74f6af90bc0865c6c0937a206df63df331b712`
- NetworKit license: MIT
- RAPIDS cuGraph repository: <https://github.com/rapidsai/cugraph>, commit
  `528ddde979df2243bf51c116d89a0ecdf85a39ee`
- RAPIDS cuGraph license: Apache-2.0

Current use in `fastEmbedR`:

- `src/graph_clustering.cpp` implements compact graph storage, KNN-edge graph
  construction, multilevel Louvain, and Leiden local-move/refine/aggregate
  phases. The Leiden phase organization is informed by NetworKit's
  MIT-licensed ParallelLeiden implementation; no NetworKit library or source
  file is linked or vendored.
- `src/graph_clustering_accel_common.h`,
  `src/graph_clustering_cuda_kernels.cu`, and
  `src/graph_clustering_metal_impl.mm` implement the shared multilevel
  orchestration and independent CUDA/Metal local-moving and refinement
  kernels. Public cuGraph documentation informed the CSR/parallel-processing
  design, but no cuGraph source, binary, Python module, or runtime symbol is
  copied, linked, or called.
- `src/walktrap.cpp` independently implements the Pons-Latapy transition
  probabilities, degree-scaled random-walk distance, Ward/Lance-Williams
  updates on adjacent communities, and modularity-selected cut.
- `tests/testthat/test-graph-clustering.R` compares canonical and stochastic
  graph results with guarded igraph reference implementations. igraph is not a
  runtime dependency.
- cuGraph was evaluated as a possible GPU dependency and deliberately omitted;
  fastEmbedR retains its own graph representation and device kernels.

## annembed

- Repository: <https://github.com/jean-pierreBoth/annembed>
- Commit inspected: `eceda8368d69e8c8790285312a974a865d063a07`
- Author: Jean-Pierre Both
- License: MIT OR Apache-2.0
- Inspected: 2026-06-11
- Current use in `fastEmbedR`: design reference only. No annembed source files
  are vendored, linked, called, or copied into the package.

Ideas worth testing later:

- HNSW-layer hierarchical landmarking: use upper ANN graph layers as a compact
  landmark subset, embed landmarks first, then initialize the remaining points
  from their nearest embedded landmark.
- Distance-aware landmark projection jitter: initialize non-landmarks near the
  matched landmark with noise scaled by the high-dimensional projection
  distance.
- Directed probability-normalized graph weights: convert neighbor distances
  to shifted exponential weights, normalize per source row, and allow local
  density to modulate the scale.
- Diffusion-map or spectral initialization from the ANN graph.
- Graph-neighbor preservation diagnostics: estimate quality by checking
  whether original graph neighbors remain within comparable radii in the
  embedded KNN graph.
- Possible utility ideas: randomized range/SVD routines, hubness diagnostics,
  and intrinsic-dimension diagnostics.

Source locations studied, not copied:

- `README.md`: overview of HNSW initialization, density-aware graph weights,
  diffusion maps, and quality estimation.
- `src/tools/kdumap.rs::get_scale_from_proba_normalisation`: row-normalized
  shifted exponential neighbor weighting.
- `src/embedder.rs::h_embed`: hierarchical embedding and landmark projection
  initialization.
- `src/embedder.rs::get_quality_estimate_from_edge_length`: neighborhood
  preservation quality diagnostic.

If any future implementation copies or closely adapts annembed source code,
add the exact source file/function beside the implementation and retain the
MIT/Apache-2.0 copyright and license notices.

## Rtsne

- Repository: <https://github.com/jkrijthe/Rtsne>
- Commit inspected: `0e769505ab791fa3c3ac25bd9434ff20e1f0a689`
- CRAN: <https://cran.r-project.org/package=Rtsne>
- Version studied locally: 0.17
- License: BSD-style package license in Rtsne's `LICENSE` file.
- Current use in `fastEmbedR`: R-level neighbor-input behaviour and parameter
  checking are informed by `Rtsne::Rtsne_neighbors()`. Rtsne's Barnes-Hut C++
  source files are not copied or vendored.

Ideas/code behaviour used:

- KNN input validation: index/distance matrices must have identical dimensions
  and valid neighbor indices.
- Translated t-SNE defaults: perplexity, `theta`, `max_iter`, early
  exaggeration, momentum switch, adaptive gains, and learning rate controls.
- Gaussian bandwidth binary search from neighbor distances to match target
  perplexity.
- Sparse symmetrized high-dimensional probabilities from precomputed KNN.

Source locations studied:

- `R/neighbors.R::Rtsne_neighbors`: user-facing KNN wrapper behaviour.
- `R/utils.R::.check_tsne_params`: parameter validation and initialization
  handling.
- `src/tsne.cpp::computeGaussianPerplexity`: KNN-distance perplexity affinity
  construction.
- `src/tsne.cpp::trainIterations`: early exaggeration, momentum, gains, and
  zero-mean update schedule.

License decision:

- Do not copy Rtsne's Barnes-Hut C++ files into fastEmbedR. They include the
  original Delft advertising-clause BSD text. The current native optimizer in
  `src/tsne_neighbors.cpp` is a fresh implementation of the t-SNE-from-KNN math
  tested against the same public default parameters where appropriate.

## opt-SNE / Multicore-opt-SNE

- Paper: Belkina AC, Ciccolella CO, Anno R, Spidlen J, Halpert R, and
  Snyder-Cappione JE. "Automated optimized parameters for T-distributed
  stochastic neighbor embedding improve visualization and analysis of large
  datasets." Nature Communications 10, 5415, 2019.
- Repository: <https://github.com/omiq-ai/Multicore-opt-SNE>
- Commit inspected: `c6d15e3d4c0372ea8e7f652014819c98354a3325`
- License: BSD-3-Clause
- Current use in `fastEmbedR`: algorithmic behaviour reference for automatic
  t-SNE parameter selection. No Multicore-opt-SNE source files are vendored,
  linked, called, or copied into the package.

Ideas implemented:

- `learning_rate = "auto"` resolves to the opt-SNE learning-rate rule
  `n / early_exaggeration`.
- Missing early-exaggeration and normal-phase iteration limits are selected by
  a native C++ helper.
- On CPU/small exact runs, `src/tsne_neighbors.cpp` monitors KLD and can stop
  early exaggeration after the local maximum of relative KLD change, then stop
  the normal phase when KLD improvement drops below `KLD / 5000`.
- On large FFT and GPU-labelled runs, KLD auto-stopping is disabled unless a
  real native monitor is implemented. This avoids hiding an O(n^2) CPU KLD
  poll inside a GPU or FFT benchmark.

Implemented locations:

- `src/tsne_neighbors.cpp::tsne_auto_parameters_cpp`
- `src/tsne_neighbors.cpp::knn_tsne_opentsne_float_cpp`
- `R/fast_knn_tsne.R::resolve_opentsne_auto_parameters`

## openTSNE

- Repository: <https://github.com/pavlin-policar/openTSNE>
- Commit inspected: `12b8b7bec3d96ee0bc8930c3c45449a5209ed3e0`
- License: BSD-3-Clause
- Current use in `fastEmbedR`: mathematical and workflow reference for native
  C++ t-SNE-from-KNN, openTSNE-style two-phase optimization, and transform
  paths. No Python, Cython, or scikit-learn runtime dependency is used by
  fastEmbedR. The term does not claim compatibility with Python openTSNE's API,
  classes, serialized objects, parameter defaults, or coordinate trajectories.

Ideas/code behaviour used:

- Expose a `negative_gradient_method` choice in the t-SNE API. The native C++
  implementation now supports `"exact"` and `"fft"` grid approximation; the
  older Barnes-Hut route was removed from the public path after it lost to
  FFT-grid in the MNIST 70k benchmark. The Metal backend has a package-native
  FFT-grid path. The CUDA backend uses package-native kernels plus cuFFT for
  the FFT-grid convolution.
- Expose `tsne()` and `embed_knn(method = "tsne")` as a separate native C++
  path with two-phase early-exaggeration and normal optimization,
  automatic learning-rate selection, momentum/gain updates, and max-step
  clipping.
- Separate sparse KNN attractive forces from approximate negative forces.
- Barnes-Hut negative-force normalization follows the openTSNE structure:
  compute an approximate negative gradient and normalizer from a quadtree,
  then add sparse positive KNN forces.
- For transforms/landmark projection, initialize query points from reference
  embedding neighbors, compute asymmetric query-to-reference affinities, and
  optimize query points against the fixed reference embedding.
- For the native Metal full-embedding path, compute exact dense symmetric KNN
  affinities from row-wise perplexity probabilities and use a global t-SNE
  `sum_Q` normalizer inside the Metal optimizer. This is a package-native
  Objective-C++/Metal implementation, not a port of openTSNE Python/Cython
  files.

Source locations studied:

- `openTSNE/_tsne.pyx::estimate_positive_gradient_nn`
- `openTSNE/_tsne.pyx::estimate_negative_gradient_bh`
- `openTSNE/quad_tree.pyx::QuadTree`
- `openTSNE/tsne.py::kl_divergence_bh`
- `openTSNE/tsne.py::TSNEEmbedding.transform`

Implemented native extensions:

- FFT/FIt-SNE-style interpolation negative gradients are implemented as
  package-native CPU, Metal, and CUDA paths. The implementation follows the
  openTSNE/FIt-SNE split into grid scatter, FFT convolution, grid gather,
  sparse attractive forces, and momentum/gain updates, but does not call
  Python/Cython at runtime.

## AppleSiliconFFT

- Repository: <https://github.com/aminems/AppleSiliconFFT>
- Commit adapted: `5d0d51dbd983691ee99822ed74bc3f9a47136511`
- Author: Mohamed Amine Bergach
- License: MIT
- Current use in `fastEmbedR`: the native Metal 512-point Stockham FFT
  organization for the openTSNE/FIt-SNE 512x512 grid path is adapted from
  `src/fft_multisize.metal::fft_512_stockham`; the upstream MIT notice is
  retained in `LICENSES/APPLESILICONFFT-LICENSE`.

Ideas/code behaviour used:

- Radix-4 Stockham autosort stages.
- Threadgroup-memory staging for one 512-point transform per threadgroup.
- Separate row and column kernels for 2D FFT grids.

Implemented locations:

- `src/embedding_metal_impl.mm::opentsne_fft_stockham512_core`
- `src/embedding_metal_impl.mm::opentsne_fft_512_rows_stockham`
- `src/embedding_metal_impl.mm::opentsne_fft_512_cols_stockham`
- `src/embedding_metal_impl.mm::metal_fft512_stockham_diagnostic_impl`

Validation:

- The standalone diagnostic compares the Stockham path with fastEmbedR's
  previous generic Metal Cooley-Tukey FFT path.
- On the local Mac, the corrected kernel matched the generic reference with
  relative RMS error around `3.16e-7` for forward and inverse 512x512 FFTs.

License decision:

- AppleSiliconFFT is MIT licensed and compatible with permissive fastEmbedR
  distribution. Keep this notice if the Stockham kernels are modified further.

## t-SNE-CUDA

- Paper: Chan DM, Rao R, Huang F, Canny J. "t-SNE-CUDA: GPU-Accelerated
  t-SNE and its Applications to Modern Data." arXiv:1807.11824v1, 2018.
- Repository: <https://github.com/CannyLab/tsne-cuda>
- Commit inspected: `44249b6895a2eb389b8a13390ed6fb125d2040c8`
- License: BSD-3-Clause
- Current use in `fastEmbedR`: design reference only. No t-SNE-CUDA source
  files are vendored, linked, called, or copied into the package.

Ideas translated now:

- Treat the transform repulsive term as an n-body style loop and keep it inside
  the GPU kernel rather than shuttling intermediate gradients through R.
- Keep query coordinates, reference coordinates, gains, updates, probabilities,
  and query-reference indices in device buffers for all transform iterations.
- Return only the final two-dimensional query layout to R.
- For a future exact CUDA t-SNE-from-KNN parity path, keep the Rtsne-style
  sparse attractive affinities, early exaggeration, momentum/gains, and
  zero-mean update schedule while executing the dense exact repulsive force on
  CUDA.

Implemented location:

- `src/embedding_metal_impl.mm::tsne_transform_epoch`: native Metal kernel for
  the fixed-reference t-SNE transform.
- `src/embedding_metal_impl.mm::tsne_probability_dense_rows`,
  `tsne_global_sum_q`, and `tsne_full_exact_dense_epoch`: native Metal exact
  t-SNE optimizer from KNN for moderate datasets.
- `R/transform_tsne.R::transform_tsne`: explicit `backend = "metal"` dispatch.
- CUDA full-embedding kernels are shipped as package-owned float32 code using
  installed CUDA/cuFFT primitives. t-SNE-CUDA remains an architecture
  reference; no source from that repository is bundled or linked.

Current CUDA boundary:

- Full embedding and fixed-reference transform use package-owned float32 CUDA
  kernels plus installed CUDA/cuFFT primitives.
- t-SNE-CUDA remains a design reference; no repository source or binary is
  copied, linked, or called.
- Explicit CUDA requests fail if the required compiled backend is unavailable;
  they do not silently fall back to CPU.

## uwot / UMAP

- Repository: <https://github.com/jlmelville/uwot>
- Commit inspected: `4c9c9261ad2944e81aede1f02d1ad01b4add344a`
- CRAN: <https://cran.r-project.org/package=uwot>
- License: GPL (>= 3)
- Current use in `fastEmbedR`: benchmark and behavioural reference for public
  UMAP results from precomputed KNN input. The cleaned implementation does not
  vendor uwot source files; it keeps its own C++/Metal/CUDA API and uses
  independently written CSR graph buffers, negative sampling, RNG, and optimizer
  loops.

License boundary:

- `uwot` may appear in benchmark scripts, vignettes, papers, and external
  comparisons.
- `uwot` is not an Import, LinkingTo dependency, vendored source tree, or
  runtime requirement for `fastEmbedR::umap()` or `fastEmbedR::umap_knn()`.
- Do not add GPL-derived optimizer code to `R/`, `src/`, or installed package
  docs while retaining the MIT package license.

Behaviour compared in benchmarks:

- Smooth KNN bandwidth search with UMAP local connectivity.
- Fuzzy simplicial set weights and fuzzy union graph behaviour.
- `epochs_per_sample` edge scheduling for sampled attractive updates.
- UMAP negative sampling, learning-rate decay, and asynchronous SGD behaviour.
- Public benchmarks compare against both supplied-KNN and end-to-end
  `uwot::umap(..., fast_sgd = TRUE)` runs where available.

## Fast power approximation

- Primary algorithmic reference: Nicol N. Schraudolph, "A Fast, Compact
  Approximation of the Exponential Function", Neural Computation, 1999.
- Additional permissive prior art reviewed: Harrison Ainsworth / HXA7241,
  "Fast pow() With Adjustable Accuracy", whose downloadable source is provided
  under the new BSD license.
- Current use in `fastEmbedR`: the CPU, Metal, and CUDA UMAP optimizers use
  local IEEE-754 exponent interpolation helpers for positive powers in the
  attractive and repulsive force calculations. The CPU helper uses `memcpy`;
  the GPU helpers use the corresponding backend bit-cast intrinsics. The code
  is not vendored from the HXA package or from blog snippets; it is a
  package-local implementation of the published bit-level approximation idea,
  with float-specialized variants for hot 2D optimizer paths.

License boundary:

- The constants and bit operations are used as an independently written
  implementation of the published approximation technique.
- The helper names, control flow, and memory representation are package-local;
  the CPU version uses `std::memcpy` rather than union-punning snippets.
- If future review requires the most conservative posture, replace these
  helpers with standard `pow` calls or another clearly permissive
  implementation and rerun the speed/quality gates.

Implemented locations:

- `src/fast_knn_umap.cpp::umap_pow`
- `src/fast_knn_umap.cpp::umap_powf_fast`
- `src/embedding_metal_impl.mm::fast_positive_pow`
- `src/embedding_cuda_kernels.cpp::fast_positive_pow`

## UMAP reference implementation

- Repository: <https://github.com/lmcinnes/umap>
- License: BSD-3-Clause
- Current use in `fastEmbedR`: mathematical reference for UMAP fuzzy graph
  construction and SGD objective. The Python implementation is not called at
  runtime and source files are not vendored.

Ideas used:

- Fuzzy simplicial set formulation from KNN distances.
- `a`/`b` curve parameters for the low-dimensional UMAP attraction curve.
- Negative-sampling repulsive force objective.

## Permissive C++ UMAP / Optimizer References

- `umappp` repository: <https://github.com/libscran/umappp>
- `umappp` license: BSD-2-Clause
- `ensmallen` repository: <https://github.com/mlpack/ensmallen>
- `ensmallen` license: BSD-3-Clause
- Current use in `fastEmbedR`: provenance and design references for a
  permissively licensed, standalone C++ UMAP implementation style and for
  optimizer-implementation patterns. fastEmbedR does not vendor either project,
  does not link to them at runtime, and does not copy their source code.

Ideas reviewed:

- Keeping the public API independent from a specific R implementation.
- Separating neighbor search, graph construction, initialization, and
  optimization into testable modules.
- Maintaining package-local optimizer code rather than adapting GPL-only
  implementation details from benchmark reference packages.

## FAISS And Faiss-mlx Nearest-Neighbor Search

- FAISS repository: <https://github.com/facebookresearch/faiss>
- FAISS source: release 1.14.3, commit
  `0ca9df4792b173d573044ee14ca0704780176e82`
- FAISS license: MIT
- Faiss-mlx repository: <https://github.com/MLXPorts/Faiss-mlx>
- Faiss-mlx source commit: `d092af559375144fc719cd88a10e414f92c625fa`
- Faiss-mlx license: Apache-2.0
- Current use in `fastEmbedR`: package-native float32 CPU HNSW distilled from
  FAISS's HNSW organization, plus native Metal exact and IVF-Flat search. The
  Metal fused list-scan/top-k structure was adapted from FAISS and Faiss-mlx.
  The package does not link FAISS or MLX for CPU/Metal paths; optional CUDA
  builds may link installed FAISS GPU directly.

Implemented locations:

- `src/native_knn_hnsw.cpp`: compact hierarchical graph construction,
  diversity pruning, reciprocal links, and parallel query search.
- `src/native_knn_metal_impl.mm`: 128-dimensional signed coarse projection,
  four-pass Metal centroid assignment/update, adaptive 288/384/512-candidate
  shortlist construction, exact full-dimensional reranking, four-stratum
  pilot validation, and recall-aware `nprobe` selection.
- `inst/LICENSES/`: full upstream licenses, copyright notices, and pinned
  provenance.

## RAPIDS RAFT / cuVS

- cuVS repository: <https://github.com/rapidsai/cuvs>
- RAFT repository: <https://github.com/rapidsai/raft>
- License: Apache-2.0
- Current use in `fastEmbedR`: optional CUDA builds link directly to the cuVS
  C API for exact and IVF-Flat KNN. RAFT TSVD is used as one branch of the
  automatic CUDA PCA selector. fastEmbedR does not vendor RAPIDS source or call
  cuML UMAP/t-SNE at runtime. The package does not link cuML.

Implemented locations:

- `src/native_knn_cuda_impl.cpp` calls the installed cuVS C API and owns the
  resulting device buffers; fastEmbedR does not vendor RAPIDS source.
- `src/embedding_cuda_impl.cpp` and `src/embedding_cuda_kernels.cpp`:
  package-native CUDA UMAP and t-SNE kernels when CUDA support is compiled.

## mlx-vis

- Repository: <https://github.com/hanxiao/mlx-vis>
- License: Apache-2.0
- Current use in `fastEmbedR`: design reference for GPU-resident
  dimensionality-reduction pipelines. No mlx-vis source files are vendored,
  linked, or called, and fastEmbedR does not depend on Python/MLX at runtime.

Ideas used:

- GPU-resident KNN/embedding pipeline structure as a Metal design reference.
- FFT-grid/scatter/gather architecture as a reference while validating the
  native Metal t-SNE path.

Implemented locations:

- `src/embedding_metal_impl.mm`: native Metal UMAP/t-SNE kernels.

## Apple Metal Performance Shaders Matrix

- Documentation: <https://developer.apple.com/documentation/metalperformanceshaders/matrices_and_vectors>
- Current use in `fastEmbedR`: package-native float32 block-subspace rSVD for
  Metal PCA and t-SNE initialization. MPS matrix multiplication executes the
  large forward/back projections while buffers remain resident in Apple unified
  memory. A package-native float32 Jacobi eigensolver handles only the small
  projected Gram matrix.

Implemented location:

- `src/embedding_metal_impl.mm::run_rsvd_pca_metal`

## Apple MPSGraph

- Documentation: <https://developer.apple.com/documentation/metalperformanceshadersgraph>
- Current use in `fastEmbedR`: diagnostic-only FFT and convolution comparison
  for Metal t-SNE. It is not the default public t-SNE backend because the
  MNIST 70k flattened-image benchmark showed only a small speed gain and a
  small quality/plot shift compared with the package-native Metal FFT-grid
  path.

Implemented locations:

- `src/embedding_metal_impl.mm::metal_mpsgraph_fft_diagnostic_impl`
- `src/embedding_metal_impl.mm::metal_mpsgraph_convolution_diagnostic_impl`
- [`tools/diagnose_mpsgraph_fft.R`](https://github.com/tkcaccia/fastEmbedR-benchmark/blob/main/tools/diagnose_mpsgraph_fft.R)
- [`tools/diagnose_mpsgraph_convolution.R`](https://github.com/tkcaccia/fastEmbedR-benchmark/blob/main/tools/diagnose_mpsgraph_convolution.R)
