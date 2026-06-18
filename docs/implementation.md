# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Provenance](algorithm-provenance.md)

This page gives the GitHub-level implementation overview. More detailed
function-by-function notes are in
[function-implementation-details.md](function-implementation-details.md), and
the full reference/provenance log is in
[algorithm-provenance.md](algorithm-provenance.md) and
[`inst/ALGORITHMIC_REFERENCES.md`](../inst/ALGORITHMIC_REFERENCES.md).

## Nearest Neighbours

Neighbour search is delegated to `faissR`.

- CPU: FAISS flat, IVF, HNSW/NSG where available through `faissR`.
- CUDA: cuVS/FAISS GPU backends where available through `faissR`.
- Distances: Euclidean and cosine/inner-product are the main supported
  distances.

`fastEmbedR` keeps this layer separate so KNN can be timed once and reused by
UMAP, openTSNE, graph construction, and classification workflows.

## UMAP

`umap_knn()` embeds from a supplied KNN object. `umap()` is the convenience
function that first calls `faissR::nn()` and then runs `umap_knn()`.

### CPU

The CPU UMAP implementation is native C++:

- KNN input is normalized once.
- Smooth KNN bandwidths are estimated row-wise.
- Fuzzy graph weights are built into compact CSR/COO-style buffers.
- Internal weights/distances use `float` where safe to reduce memory traffic.
- Edge epochs, negative sampling, learning-rate decay, and random updates are
  package-local code.
- Threading is used for row-parallel stages where safe.

### Metal

The Metal UMAP path is native Objective-C++/Metal. It does not call Python,
Torch, MLX, or `reticulate`.

The public Metal path uses the visually validated atomic in-place optimizer.
Older scheduled, hybrid, and deterministic experimental paths were removed
from the public API.

### CUDA

The CUDA UMAP path uses native CUDA kernels when `FASTEMBEDR_USE_CUDA=1` is
enabled at build time. The CUDA optimizer consumes the KNN graph from `faissR`
and reports embedding time separately from KNN time.

### Graph Mode In Benchmarks

UMAP supports `graph_mode = "fuzzy"` and `graph_mode = "binary"`. The GitHub
benchmark examples show only `graph_mode = "fuzzy"` because that is the
standard UMAP fuzzy simplicial-set graph and is the mode requested for public
comparison against `uwot::umap(..., fast_sgd = TRUE)`.

## openTSNE-Style t-SNE

`opentsne_knn()` embeds from supplied KNN distances. `opentsne()` is the
convenience function that first calls `faissR::nn()`.

### Shared Mathematics

The implementation follows the openTSNE/FIt-SNE structure:

- convert KNN distances to conditional probabilities by perplexity search;
- symmetrize sparse high-dimensional affinities;
- use PCA initialization by default when data are available;
- run early exaggeration followed by normal optimization;
- use adaptive gains, momentum, auto learning rate, clipping, and centering;
- approximate the repulsive force with an FFT-grid/FIt-SNE-style method for
  large datasets.

The older Barnes-Hut path is not part of the public benchmark surface because
the FFT-grid path is the standard fast path used in the MNIST 70k comparisons.

### CPU

The CPU openTSNE path is C++ and supports FFT-grid negative-gradient
approximation. It can also run exact repulsion for small diagnostic cases.

### Metal

The Metal openTSNE path is native Objective-C++/Metal:

- grid scatter kernels;
- package-native Metal FFT-grid convolution;
- grid gather kernels;
- sparse attractive-force evaluation;
- gains, momentum, update, and centering kernels.

A diagnostic MPSGraph FFT path was tested, but the package-native Metal FFT
path remains the default because it gave the best balance of quality,
simplicity, and speed in local MNIST checks.

### CUDA

The CUDA openTSNE path uses native CUDA kernels and cuFFT:

- device-side scatter/gather;
- cuFFT convolution for the repulsive field;
- sparse attractive forces;
- optimizer update kernels.

CUDA KNN is still supplied by `faissR`; the embedding code receives a KNN object
and returns the final layout.

## References And Acknowledgements

Main references include:

- McInnes, Healy, and Melville, UMAP, 2018.
- Policar, Strazar, and Zupan, openTSNE, 2019.
- Linderman et al., FIt-SNE, 2019.
- Chan, Rao, Huang, and Canny, t-SNE-CUDA, 2018.
- van der Maaten and Hinton, t-SNE, 2008.
- van der Maaten, Barnes-Hut t-SNE, 2014.
- FAISS and RAPIDS cuVS for KNN backend design.
- AppleSiliconFFT as a permissive design reference for Metal Stockham FFT.
- `uwot` and `Rtsne` as external R benchmark/reference implementations.

GPL packages such as `uwot` are not imported, linked, vendored, or required by
the core MIT package code.
