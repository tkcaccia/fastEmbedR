# Algorithmic References And Provenance

This file records implementation ideas studied for `fastEmbedR`. It is meant
to keep future code changes traceable, especially when an idea comes from a
permissively licensed project.

## annembed

- Repository: <https://github.com/jean-pierreBoth/annembed>
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
- Directed probability-normalized graph weights: convert neighbour distances
  to shifted exponential weights, normalize per source row, and allow local
  density to modulate the scale.
- Diffusion-map or spectral initialization from the ANN graph.
- Graph-neighbour preservation diagnostics: estimate quality by checking
  whether original graph neighbours remain within comparable radii in the
  embedded KNN graph.
- Possible utility ideas: randomized range/SVD routines, hubness diagnostics,
  and intrinsic-dimension diagnostics.

Source locations studied, not copied:

- `README.md`: overview of HNSW initialization, density-aware graph weights,
  diffusion maps, and quality estimation.
- `src/tools/kdumap.rs::get_scale_from_proba_normalisation`: row-normalized
  shifted exponential neighbour weighting.
- `src/embedder.rs::h_embed`: hierarchical embedding and landmark projection
  initialization.
- `src/embedder.rs::get_quality_estimate_from_edge_length`: neighbourhood
  preservation quality diagnostic.

If any future implementation copies or closely adapts annembed source code,
add the exact source file/function beside the implementation and retain the
MIT/Apache-2.0 copyright and license notices.

## Rtsne

- Repository: <https://github.com/jkrijthe/Rtsne>
- CRAN: <https://cran.r-project.org/package=Rtsne>
- Version studied locally: 0.17
- License: BSD-style package license in Rtsne's `LICENSE` file.
- Current use in `fastEmbedR`: R-level neighbour-input behaviour and parameter
  checking are adapted from `Rtsne::Rtsne_neighbors()`. Rtsne's Barnes-Hut C++
  source files are not copied or vendored.

Ideas/code behaviour used:

- KNN input validation: index/distance matrices must have identical dimensions
  and valid neighbour indices.
- Translated t-SNE defaults: perplexity, `theta`, `max_iter`, early
  exaggeration, momentum switch, adaptive gains, and learning rate controls.
- Gaussian bandwidth binary search from neighbour distances to match target
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

- Do not copy Rtsne's Barnes-Hut C++ files into the GPL fastEmbedR package.
  They include the original Delft advertising-clause BSD text. The current
  native optimizer in `src/tsne_neighbors.cpp` is a fresh implementation of
  the t-SNE-from-KNN math with Rtsne-compatible defaults.

## openTSNE

- Repository: <https://github.com/pavlin-policar/openTSNE>
- License: BSD-3-Clause
- Current use in `fastEmbedR`: design reference and API reference for native
  C++ t-SNE-from-KNN, openTSNE-style two-phase optimization, and transform
  paths. No Python, Cython, or scikit-learn runtime dependency is used by
  fastEmbedR.

Ideas/code behaviour used:

- Expose a `negative_gradient_method` choice in the t-SNE API. The native C++
  implementation now supports `"exact"` and `"fft"` grid approximation; the
  older Barnes-Hut route was removed from the public path after it lost to
  FFT-grid in the MNIST 70k benchmark. The Metal backend has a package-native
  FFT-grid path. CUDA FFT remains explicitly unsupported until real CUDA
  kernels are ported.
- Expose `opentsne()` and `embed_knn(method = "opentsne")` as a separate
  native C++ path with openTSNE-style early exaggeration, normal optimization,
  automatic learning-rate selection, momentum/gain updates, and max-step
  clipping.
- Separate sparse KNN attractive forces from approximate negative forces.
- Barnes-Hut negative-force normalization follows the openTSNE structure:
  compute an approximate negative gradient and normalizer from a quadtree,
  then add sparse positive KNN forces.
- For transforms/landmark projection, initialize query points from reference
  embedding neighbours, compute asymmetric query-to-reference affinities, and
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

Not yet ported:

- FFT/FIt-SNE interpolation negative gradients from
  `openTSNE/_tsne.pyx::estimate_negative_gradient_fft_2d` and the matrix
  multiplication backends. These are BSD-3 and suitable for a later native
  C++/Metal/CUDA port, but fastEmbedR currently does not expose them as a
  working path.

## t-SNE-CUDA

- Paper: Chan DM, Rao R, Huang F, Canny J. "t-SNE-CUDA: GPU-Accelerated
  t-SNE and its Applications to Modern Data." arXiv:1807.11824v1, 2018.
- Repository: <https://github.com/CannyLab/tsne-cuda>
- Commit inspected: `44249b6`
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
  the openTSNE-style fixed-reference transform.
- `src/embedding_metal_impl.mm::tsne_probability_dense_rows`,
  `tsne_global_sum_q`, and `tsne_full_exact_dense_epoch`: native Metal exact
  openTSNE-style optimizer from KNN for moderate datasets.
- `R/transform_tsne.R::transform_tsne`: explicit `backend = "metal"` dispatch.
- CUDA full-embedding kernels are intentionally not shipped in the cleaned
  source until the openTSNE-style CUDA path is implemented and validated.

CUDA notes to keep for later:

- Port the fixed-reference transform first, not full Barnes-Hut t-SNE, because
  it has a smaller validation surface and reuses the landmark workflow.
- Reuse the same host contract as the Metal path: row-wise query probabilities,
  zero-based query-reference KNN, float32 reference/query layout buffers,
  float2 gains and updates, and exact or sampled reference repulsion.
- Then port the CannyLab/FIt-SNE large-data split: GPU KNN, GPU perplexity
  search, device-side probability symmetrization, sparse attractive forces,
  FFT/N-body repulsive field, and fused momentum/gains update.
- Do not enable `backend = "cuda"` for `transform_tsne()` until it is tested on
  a CUDA machine; explicit CUDA requests must not fall back to CPU.
