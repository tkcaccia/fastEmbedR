# Implementation

[Home](../README.md) |
[Installation](installation.md) |
**Implementation** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[References](references.md)

`fastEmbedR` implements two nonlinear embedding families: UMAP and an
openTSNE-style t-SNE. UMAP follows the fuzzy simplicial-set graph formulation
introduced by McInnes and colleagues [7,13]. The t-SNE path follows the
probabilistic neighbour-embedding objective of van der Maaten and Hinton [1],
with modern openTSNE/FIt-SNE-style optimization and interpolation ideas [3-4].
The package is intentionally KNN-first. Nearest-neighbour search is delegated
to the companion `faissR` package, while `fastEmbedR` performs graph/affinity
construction, initialization, stochastic optimization, native fixed-reference
transforms, backend reporting, and quality metrics.

The public package surface is deliberately small:

- `opentsne_knn()` and `umap_knn()` consume a supplied KNN object.
- `opentsne()` and `umap()` compute KNN through `faissR` and then call the KNN
  entry point.
- `backend` is limited to `"cpu"`, `"metal"`, and `"cuda"`.
- GPU requests fail clearly if native GPU code is unavailable. CPU fallback is
  never reported as Metal or CUDA work.

## Nearest-Neighbour Layer

`fastEmbedR` does not re-export neighbour-search functions. Users call
`faissR::nn()` directly when they want explicit control over KNN search, and
the one-call embedding functions call `faissR` internally. The neighbour layer
uses FAISS/cuVS concepts for high-throughput vector search [8-9].

For reproducibility, the one-call API fixes the internal KNN policy:

- CPU and Metal one-call embedding use FAISS CPU IVF-Flat through `faissR`.
- CUDA one-call embedding uses FAISS GPU IVF-Flat through `faissR`.
- KNN-input functions accept the index and distance matrices as already
  measured data and do not repeat neighbour search.

This separation makes benchmark timing interpretable: KNN time, affinity/graph
construction time, embedding time, and projection/transform time can be
reported separately.

## UMAP From KNN

UMAP is implemented as a sparse graph optimization from a supplied KNN matrix.
The implementation follows the UMAP fuzzy simplicial-set formulation [7,13]:
local bandwidths are estimated per observation, neighbour distances are
converted to directed membership strengths, the graph is symmetrized, and a
low-dimensional layout is optimized with attractive edge updates and sampled
repulsive updates.

### CPU Graph Construction

The CPU path is package-native C++:

1. Validate and normalize KNN indices and distances.
2. Estimate row-wise `rho` and `sigma` values by smooth KNN distance search.
3. Convert directed KNN distances to membership strengths.
4. Build the symmetric graph without dense intermediates.
5. Store compact edge arrays and epoch schedules for the optimizer.

The engineering goal is to preserve UMAP's graph mathematics while reducing
memory traffic. Internal distances and weights use `float` where safe; indices
are stored as integer buffers; and the implementation avoids duplicate graph
copies between R and C++.

### UMAP Optimizer

The optimizer uses stochastic edge sampling, negative sampling, a decaying
learning rate, and compact contiguous layout buffers, following the UMAP
objective rather than an exact copy of any R implementation [7,10,13]. The
current permissive implementation uses package-local samplers, random-number
generation, and update kernels rather than vendored `uwot` code. `uwot`
remains an external benchmark and behavioural reference, not a source
dependency [10].

UMAP exposes two graph modes:

| Mode | Meaning | Intended use |
| --- | --- | --- |
| `"fuzzy"` | Standard UMAP fuzzy graph weights. | Scientific comparison with the original UMAP model and `uwot`. |
| `"binary"` | Binary neighbour graph with the same optimizer. | Explicit approximation that may increase visible separation; always recorded in results. |

### Metal UMAP

The Metal backend is implemented in Objective-C++/Metal and uses the validated
atomic in-place edge-update kernel. It does not call Python, Torch, MLX, or
`reticulate`. The Metal optimizer consumes the same prepared graph as the CPU
path and returns only the final layout and metadata to R. The package-native
Metal FFT work used by openTSNE was informed by permissive Apple GPU FFT
engineering references [12].

### CUDA UMAP

The CUDA backend is compiled when `FASTEMBEDR_USE_CUDA=1` is enabled. The
public CUDA path uses the pure atomic optimizer. KNN is supplied by `faissR`,
and CUDA embedding consumes the graph in device-friendly buffers. The CUDA
path is benchmarked separately from CPU and Metal and reports the backend that
actually ran.

## openTSNE-Style t-SNE From KNN

`opentsne_knn()` implements the t-SNE optimization structure used by modern
openTSNE/FIt-SNE workflows [3-4]:

1. Convert KNN distances to conditional probabilities by binary search on the
   Gaussian bandwidth for a target perplexity.
2. Symmetrize sparse probabilities and normalize the high-dimensional
   affinity matrix.
3. Initialize the embedding from the KNN-native default or an explicit
   user-supplied layout.
4. Run early exaggeration.
5. Run the normal optimization phase with adaptive gains, momentum, learning
   rate, update clipping, and recentering.
6. Return the layout with parameters, timing, and backend metadata.

The public `opentsne()` function is now only a convenience wrapper around this
KNN implementation. If a KNN object is supplied through `nn`, then
`opentsne(data, nn = knn)` calls the same path as `opentsne_knn(knn)` and
produces the same layout for the same seed and parameters. This mirrors the
KNN-input validation style used by R t-SNE tooling [11].

### Initialization

PCA initialization is explicit. Use:

```r
Y_init <- opentsne_pca_init(x, backend = "cpu")
y <- opentsne_knn(knn, Y_init = Y_init)
```

or pass `init_data` when a KNN-input run should compute PCA internally. The
old public `init = c("pca", "random")` argument was removed because hidden
initialization differences made KNN-input and one-call results difficult to
compare.

### Repulsive Force Approximation

The default large-data path uses an FFT-grid approximation inspired by
FIt-SNE [3]. Sparse attractive forces are evaluated from the KNN affinity
graph. The negative force is approximated by placing points on a
two-dimensional grid, convolving with the t-SNE kernel, and interpolating the
resulting force back to points [3-5]. The Barnes-Hut path is not part of the
public benchmark surface because the FFT-grid path is the intended standard
for MNIST70k-scale data; Barnes-Hut remains an important historical reference
for tree-based t-SNE acceleration [2].

### CPU, Metal, and CUDA

| Backend | Native implementation |
| --- | --- |
| CPU | C++ sparse affinities, FFT-grid repulsion, gains/momentum optimizer [3-4]. |
| Metal | Objective-C++/Metal scatter, FFT-grid convolution, gather, attractive-force, update, and centering kernels [3,12]. |
| CUDA | CUDA kernels with cuFFT for the FFT-grid convolution and device-side optimizer updates [3,5]. |

The Metal implementation includes package-native FFT kernels. MPSGraph FFT was
tested diagnostically, but it is not a public option because it did not provide
enough benefit to justify another user-facing backend.

## Landmarking

Landmarking is implemented as an explicit approximation. The package embeds a
subset, projects non-landmark observations using fixed-reference KNN
interpolation/transform steps, and records projection/transform time
separately. Landmarking is not used silently inside full `opentsne()` or
`umap()` calls.

## Autotuning

The API asks for few parameters, but selected defaults are saved in the output.
For openTSNE-style t-SNE, the native helper follows opt-SNE-inspired rules for
learning rate and iteration defaults [6]. For UMAP, the code uses the supplied
KNN and data size to choose internal initialization effort and optimizer
defaults without silently changing the supplied neighbour graph [7,13].

## License Boundary

The implementation is intended to remain compatible with the MIT license.
References such as `uwot`, `Rtsne`, openTSNE, FIt-SNE, t-SNE-CUDA, FAISS,
RAPIDS cuVS, and AppleSiliconFFT informed design and benchmarking decisions
[3-5,8-12]. GPL code is not vendored, linked, or required by the package.

See [References](references.md) for AACR-style citations and software
acknowledgements.
