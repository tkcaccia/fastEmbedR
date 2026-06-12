# fastEmbedR

`fastEmbedR` is a KNN-first R package for fast native UMAP and openTSNE-style
embeddings. The public package is deliberately small: one nearest-neighbour
API, one UMAP API, one openTSNE API, explicit CPU/Metal/CUDA backend reporting,
and no silent CPU fallback when a GPU backend is requested.

The package is designed for workflows where the nearest-neighbour graph is a
real object: compute KNN once, reuse it for UMAP, openTSNE, landmarking,
quality checks, and benchmarks.

## Current MNIST 70k Snapshot

`fastEmbedR::opentsne_knn()` on MNIST 70k raw flattened images, using KNN input
and PCA initialization:

![MNIST 70k openTSNE PCA embeddings](docs/assets/mnist70k-opentsne-pca-embeddings-cpu-metal-cuda.png)

Nearest-neighbour time and embedding time are reported separately:

![MNIST 70k openTSNE PCA timing](docs/assets/mnist70k-opentsne-pca-timing-stacked.png)

| backend | machine | NN sec | embedding sec | total sec | trust | label KNN acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | Stefanos-MacBook-Pro.local | 61.642 | 3.896 | 65.538 | 0.324 | 0.958 |
| Metal | Stefanos-MacBook-Pro.local | 40.904 | 3.250 | 44.154 | 0.312 | 0.966 |
| CUDA | icgeb-bioinformatics-unit | 2.046 | 0.410 | 2.456 | 0.327 | 0.972 |

More benchmark notes and the source CSV are in
[docs/benchmarks.md](docs/benchmarks.md) and
[docs/benchmark-gallery.md](docs/benchmark-gallery.md).

## What Is Included

| Function | Purpose |
| --- | --- |
| `nn()` | Native nearest-neighbour search, including CPU/Metal NN-descent and optional CUDA cuVS NN-descent. |
| `umap_knn()` | UMAP from a supplied KNN graph. |
| `umap()` | One-call KNN plus UMAP. |
| `opentsne_knn()` | Native openTSNE-style t-SNE from a supplied KNN graph. |
| `opentsne()` | One-call KNN plus openTSNE-style t-SNE. |
| `embed_knn()` | KNN dispatcher; UMAP by default, openTSNE with `method = "opentsne"`. |
| `landmark_umap()` | UMAP landmark workflow with projection/refinement. |
| `landmark_tsne()` | openTSNE landmark workflow with fixed-reference transform. |
| `transform_tsne()` | Place new/query points into an existing openTSNE-style map. |
| `evaluate_embedding()` | Trustworthiness, neighbour preservation, label accuracy, and related diagnostics. |
| `backend_info()` | CPU/CUDA/Metal/FAISS/cuVS detection without crashing or silent fallback. |

Legacy classic `tsne()`, InfoTSNE, PaCMAP, TriMap, and LocalMAP experiments
were removed from the public API. The active scope is KNN, UMAP,
openTSNE-style t-SNE, GPU backends, and landmarking.

## Backend Summary

| Component | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| KNN | native exact / native NN-descent | native Metal NN-descent | optional RAPIDS cuVS NN-descent |
| UMAP | native C++ CSR graph + uwot-like SGD | native Metal `atomic_inplace` optimizer | native CUDA pure-atomic optimizer |
| openTSNE | native C++ FFT-grid optimizer | native Metal FFT-grid optimizer | native CUDA FFT-grid optimizer using cuFFT |
| Landmarking | native projection/transform | native Metal projection/refinement kernels where available | native CUDA projection/refinement kernels where built |

See [docs/backend-capabilities.md](docs/backend-capabilities.md) for the full
capability matrix and backend rules.

## Quick Install

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

Optional CUDA/cuVS and FAISS builds are documented in
[docs/installation-backends.md](docs/installation-backends.md).

## Minimal KNN-First Workflow

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- nn(x, k = 31, n_threads = 4)
layout_umap <- umap_knn(knn, backend = "cpu", seed = 1)
layout_tsne <- opentsne_knn(knn, init_data = x, backend = "cpu", seed = 1)

plot(layout_umap, pch = 21, bg = labels)
plot(layout_tsne, pch = 21, bg = labels)
```

Use a real GPU explicitly:

```r
knn_metal <- nn(x, k = 31, backend = "metal")
layout <- umap_knn(knn_metal, backend = "metal", seed = 1)
```

If the requested GPU backend is unavailable, the call fails clearly. It does
not run on CPU and label the result as GPU.

## Documentation Map

| Topic | Page |
| --- | --- |
| Installation, optional FAISS/cuVS/CUDA setup, backend rules | [docs/installation-backends.md](docs/installation-backends.md) |
| Basic use, KNN-first workflow, landmark workflow, public API | [docs/usage-api.md](docs/usage-api.md) |
| Backend capability matrix | [docs/backend-capabilities.md](docs/backend-capabilities.md) |
| Benchmark gallery with plots | [docs/benchmark-gallery.md](docs/benchmark-gallery.md) |
| Extended benchmark suite for MNIST, Fashion-MNIST, Shuttle, Covertype, CIFAR, opt-SNE cytometry datasets, and optional local data | [docs/extended-benchmark-suite.md](docs/extended-benchmark-suite.md) |
| Current MNIST benchmark commands and result snapshots | [docs/benchmarks.md](docs/benchmarks.md) |
| Implementation and library inventory for openTSNE, UMAP, and KNN | [docs/implementation-inventory.md](docs/implementation-inventory.md) |
| Function-by-function implementation details and literature | [docs/function-implementation-details.md](docs/function-implementation-details.md) |
| Metal FFT development notes | [docs/metal-fft-roadmap.md](docs/metal-fft-roadmap.md) |
| FFT library evaluation | [docs/fft-library-evaluation.md](docs/fft-library-evaluation.md) |
| Full benchmark summary | [BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md) |
| License implications | [LICENSE-IMPLICATIONS.md](LICENSE-IMPLICATIONS.md) |
| Detailed provenance | [inst/NOTICE](inst/NOTICE) and [inst/ALGORITHMIC_REFERENCES.md](inst/ALGORITHMIC_REFERENCES.md) |

## Design Rules

- KNN is a first-class input. Compute it once and reuse it.
- CPU, Metal, and CUDA results are labelled by the backend that actually ran.
- Public UMAP/openTSNE functions do not call Python.
- Slow or visually weak experimental branches are kept out of the public API.
- Defaults are opinionated: PCA initialization for openTSNE, validated Metal
  UMAP `atomic_inplace`, FFT-grid openTSNE, and reusable KNN graphs.

## License And Provenance

`fastEmbedR` is licensed as `GPL (>= 3)`. Implementation/library inventory,
acknowledgements, and algorithmic references are maintained in
[docs/implementation-inventory.md](docs/implementation-inventory.md),
[docs/function-implementation-details.md](docs/function-implementation-details.md),
[inst/NOTICE](inst/NOTICE), and
[inst/ALGORITHMIC_REFERENCES.md](inst/ALGORITHMIC_REFERENCES.md).
