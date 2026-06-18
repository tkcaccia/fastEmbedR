# Algorithm Design, Improvements, And Acknowledgements

This page explains the main implementation choices behind `fastEmbedR` and
acknowledges the papers and open-source projects that shaped the current UMAP
and openTSNE-style algorithms.

The public package is intentionally narrow:

- `fastEmbedR` implements UMAP and openTSNE-style t-SNE embeddings.
- `faissR` implements nearest-neighbour search, candidate KNN, KNN graphs,
  KNN prediction, and k-means.
- Public UMAP and openTSNE paths do not call Python.
- Explicit GPU requests must run real native Metal/CUDA code or fail clearly.

## Development Summary

The package started from a KNN-first requirement: the user should be able to
compute a neighbour graph once, then reuse it for UMAP, openTSNE, landmarking,
metrics, and clustering graphs. This led to the two-package layout:

| Package | Responsibility |
| --- | --- |
| `faissR` | FAISS/cuVS nearest neighbours, KNN graph construction, KNN classifier/regressor, k-means. |
| `fastEmbedR` | UMAP, openTSNE-style t-SNE, landmark transforms, quality metrics, backend reporting. |

This split keeps the embedding package smaller and makes the accelerated KNN
layer independently testable.

## UMAP Implementation

### CPU Path

The CPU UMAP path is native C++ and accepts a precomputed KNN object. The main
speed and memory improvements are:

- KNN-first API, so KNN time is separated from embedding time.
- Compact graph buffers rather than dense graph intermediates.
- CSR/COO-style storage for graph edges, weights, and epoch schedules.
- Parallel smooth KNN bandwidth search where safe.
- Fuzzy graph construction from local `rho` and `sigma` values.
- UMAP-style `epochs_per_sample` edge scheduling.
- Package-local negative sampling and learning-rate decay implementing the
  documented UMAP objective.
- `float` storage internally where it reduces memory traffic without changing
  the public output type.

### Graph Modes

`umap_knn()` exposes two graph modes:

| `graph_mode` | Meaning |
| --- | --- |
| `"fuzzy"` | Standard UMAP fuzzy simplicial-set weights. |
| `"binary"` | A binary neighbour graph that can increase visible separation on some datasets. This is a deliberate approximation and is recorded in benchmark output. |

The benchmark scripts include both modes when requested, so visual differences
are visible rather than hidden behind one score.

### Metal Path

The Metal UMAP path is implemented in Objective-C++/Metal. It does not use
Python, Torch, MLX, or `reticulate`.

The current public Metal UMAP optimizer is the validated `atomic_inplace`
path. Earlier scheduled, deterministic, and hybrid experiments were removed
from the user-facing code after visual checks and MNIST/USPS benchmark runs
showed that they either distorted the embedding or made the API confusing.

The Metal path keeps edge arrays and layout buffers in native memory and uses
Metal kernels for the stochastic layout updates. The goal is parity with the
documented UMAP objective and the package CPU reference first, then speed.

### CUDA Path

The CUDA UMAP path is implemented with native CUDA kernels when the package is
built with CUDA support. The public CUDA mode is the pure atomic optimizer.
Older hybrid and deterministic CUDA variants were removed from the public API
to avoid presenting unvalidated alternatives.

CUDA KNN comes from `faissR`, preferably through RAPIDS cuVS/CAGRA or FAISS GPU
indexes when they are available. The embedding code receives a KNN object and
does not silently recompute neighbours with a different backend.

### UMAP References And Acknowledgements

The implementation is based on the UMAP algorithm and on behaviour observed in
strong existing implementations:

- McInnes, Healy, and Melville, "UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction", 2018.
- `uwot`, by Jim Melville, as an R UMAP benchmark and behavioural reference
  for output from precomputed KNN. `uwot` is GPL (>= 3), so its source code is
  not copied, vendored, linked, or required by the MIT-licensed `fastEmbedR`
  core implementation.
- RAPIDS cuML UMAP engineering material for GPU-resident graph and optimizer
  design ideas.
- `mlx-vis` for high-level Apple GPU/Metal design ideas. No `mlx-vis` source
  is vendored or called by `fastEmbedR`.

## openTSNE-Style t-SNE Implementation

### CPU Path

`opentsne_knn()` implements t-SNE from supplied KNN distances:

- Convert KNN distances into row-wise conditional probabilities by perplexity
  binary search.
- Symmetrize sparse high-dimensional probabilities.
- Run early exaggeration followed by the normal optimization phase.
- Use momentum, adaptive gains, auto learning rate, clipping, and recentering.
- Use PCA initialization by default when original data or a saved PCA
  initialization is available.

The default large-data negative-gradient method is the FFT-grid/FIt-SNE-style
approximation. The older Barnes-Hut path was removed from the public benchmark
surface because the FFT-grid path was the useful standard in the MNIST 70k
tests.

### Metal Path

The Metal openTSNE path is native Objective-C++/Metal. It contains:

- Metal grid scatter kernels.
- Package-native Metal FFT-grid convolution.
- Metal gather kernels.
- Sparse attractive-force evaluation.
- Gain, momentum, update, and centering kernels.

The Metal FFT implementation was tested against the package's CPU/CUDA
behaviour and a diagnostic MPSGraph FFT path. MPSGraph FFT remains diagnostic
only because it did not provide enough quality/speed benefit to justify a
second public backend option.

PCA initialization is the default for openTSNE because visual inspection on
MNIST 70k showed that it prevents unstable cluster splitting without changing
the t-SNE objective.

### CUDA Path

The CUDA openTSNE path uses native CUDA kernels and cuFFT when compiled:

- Device-side grid scatter/gather.
- cuFFT convolution for the repulsive field.
- Sparse attractive forces.
- Fused optimizer updates where possible.

The design follows the same openTSNE/FIt-SNE split as the CPU and Metal paths.
CUDA-specific KNN is still owned by `faissR`; the embedding path consumes a KNN
object and reports embedding time separately from KNN time.

### openTSNE References And Acknowledgements

The implementation was shaped by:

- van der Maaten and Hinton, "Visualizing Data using t-SNE", 2008.
- van der Maaten, "Accelerating t-SNE using Tree-Based Algorithms", 2014.
- Policar, Strazar, and Zupan, "openTSNE: a modular Python library for t-SNE
  dimensionality reduction and embedding", 2019.
- Linderman et al., "Fast interpolation-based t-SNE for improved
  visualization of single-cell RNA-seq data", 2019.
- Chan, Rao, Huang, and Canny, "t-SNE-CUDA: GPU-Accelerated t-SNE and its
  Applications to Modern Data", 2018.
- `Rtsne::Rtsne_neighbors()` for KNN-input validation and compatibility
  checks. `Rtsne` source is not vendored.
- opt-SNE / Multicore-opt-SNE for automatic learning-rate and iteration
  policy. No Multicore-opt-SNE source is vendored.
- AppleSiliconFFT for the Stockham FFT design reference used in the native
  Metal FFT-grid implementation. AppleSiliconFFT is MIT licensed.

## Landmarking

Landmarking is an explicit approximation, not a replacement for the full
embedding. It is useful when full optimization is too expensive:

1. Select landmarks.
2. Embed landmarks with UMAP or openTSNE.
3. Project non-landmarks from neighbour relationships.
4. Optionally refine only the projected/query points against a fixed
   reference layout.

The package records embedding and projection/transform timing separately so
that landmark speedups are not hidden inside one total.

## Autotuning

The public API intentionally asks for few parameters. Internally, the package
can choose:

- t-SNE learning rate and iteration policy from opt-SNE-style rules.
- PCA initialization for openTSNE by default.
- UMAP initialization effort and graph/optimizer defaults from KNN distance
  profiles.
- Landmark policies for large datasets.

Autotuning never changes a supplied KNN object silently. If the user gives a
KNN graph, that graph is the input to the embedding.

## Backend Honesty

Every GPU implementation follows the same rule:

- A Metal result must come from native Metal kernels.
- A CUDA result must come from native CUDA/cuFFT kernels.
- If a requested GPU backend is unavailable, the function fails or reports
  `backend_unavailable`.
- CPU metrics after a GPU layout transfer are labelled as CPU metrics, not GPU
  work.

This rule is more important than a flattering benchmark number.

## What Is Not Included

The cleaned package deliberately does not expose:

- classic `tsne()`;
- InfoTSNE;
- PaCMAP;
- TriMap;
- LocalMAP;
- old Metal UMAP optimizer variants;
- old CUDA hybrid/deterministic UMAP variants;
- experimental CPU-only grid negative-gradient options;
- Python wrappers as public embedding implementations.

Those experiments were useful during development, but keeping them would make
the package harder to trust and harder to submit.

## License Notes

`fastEmbedR` is licensed under the MIT license. The documentation distinguishes
between:

- code that is part of `fastEmbedR`;
- optional external libraries linked at build/runtime;
- projects used only as design references.

FAISS, RAPIDS cuVS, openTSNE, t-SNE-CUDA, AppleSiliconFFT, mlx-vis, annembed,
Rtsne, uwot, and opt-SNE are acknowledged in
[`inst/NOTICE`](../inst/NOTICE) and
[`inst/ALGORITHMIC_REFERENCES.md`](../inst/ALGORITHMIC_REFERENCES.md).

GPL-licensed projects such as `uwot` may be used in external benchmark scripts
and papers as references, but they are not Imports, LinkingTo dependencies, or
vendored source for the core package.
