# fastEmbedR

`fastEmbedR` is an opinionated KNN-first R package for native UMAP and
openTSNE-style embeddings.

The package now keeps the public surface deliberately small: one KNN API, one
UMAP API, one openTSNE-style API, explicit CPU/Metal/CUDA backend reporting,
and no silent CPU fallback when a GPU backend is requested.

## What Is Included

- `nn()` computes nearest neighbours in native code.
- `umap_knn()` runs UMAP from a precomputed KNN graph.
- `umap()` computes or reuses KNN, then runs UMAP.
- `opentsne_knn()` runs the native openTSNE-style optimizer from a KNN graph.
- `opentsne()` computes or reuses KNN, then runs native openTSNE-style t-SNE.
- `embed_knn()` dispatches to UMAP by default, or openTSNE with
  `method = "opentsne"`.
- `transform_tsne()` places new/query points into an existing openTSNE-style
  map.
- `landmark_tsne()` and `landmark_umap()` run landmark workflows.
- `evaluate_embedding()` reports trustworthiness, neighbour preservation,
  silhouette, label KNN accuracy, and related diagnostics.

Legacy `tsne()`, `infotsne()`, PaCMAP, TriMap, and LocalMAP package
implementations were removed from the public API. The active scope is UMAP,
openTSNE-style t-SNE, KNN, GPU backends, and landmarking.

## Documentation

| Topic | Page |
| --- | --- |
| Installation, optional FAISS/cuVS/CUDA setup, backend rules | [docs/installation-backends.md](docs/installation-backends.md) |
| Basic use, KNN-first workflow, landmark workflow, public API | [docs/usage-api.md](docs/usage-api.md) |
| Current MNIST benchmark command and result snapshot | [docs/benchmarks.md](docs/benchmarks.md) |
| Implementation and library inventory for openTSNE, UMAP, and KNN | [docs/implementation-inventory.md](docs/implementation-inventory.md) |
| Function-by-function implementation details and literature | [docs/function-implementation-details.md](docs/function-implementation-details.md) |
| Metal FFT development notes | [docs/metal-fft-roadmap.md](docs/metal-fft-roadmap.md) |
| FFT library evaluation | [docs/fft-library-evaluation.md](docs/fft-library-evaluation.md) |
| Full benchmark summary | [BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md) |
| License implications | [LICENSE-IMPLICATIONS.md](LICENSE-IMPLICATIONS.md) |
| Detailed provenance | [inst/NOTICE](inst/NOTICE) and [inst/ALGORITHMIC_REFERENCES.md](inst/ALGORITHMIC_REFERENCES.md) |

## Quick Install

```r
install.packages("remotes")
remotes::install_github("tkcaccia/fastEmbedR")
```

For optional native FAISS, CUDA, and RAPIDS cuVS builds, see
[docs/installation-backends.md](docs/installation-backends.md).

## Minimal Example

```r
library(fastEmbedR)

set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

knn <- nn(x, k = 31)
layout <- umap_knn(knn)

plot(layout, pch = 21, bg = labels)
```

Request openTSNE from the same KNN graph:

```r
layout_tsne <- opentsne_knn(
  knn,
  init_data = x,
  perplexity = 10,
  early_exaggeration_iter = 100,
  n_iter = 250
)

plot(layout_tsne, pch = 21, bg = labels)
```

## Design Rules

- KNN is a first-class input. Compute it once, reuse it across UMAP,
  openTSNE, landmarking, and benchmarks.
- GPU requests must use a real native GPU backend. If CUDA or Metal is
  unavailable, the package fails clearly instead of reporting CPU work as GPU.
- CPU, Metal, and CUDA implementations are native package code or explicit
  native library links. Public UMAP/openTSNE functions do not call Python.
- Slow or visually weak experimental branches are kept out of the public API.

## License And Provenance

`fastEmbedR` is licensed as `GPL (>= 3)`. The implementation/library inventory
and acknowledgement notes are maintained in
[docs/implementation-inventory.md](docs/implementation-inventory.md),
[inst/NOTICE](inst/NOTICE), and
[inst/ALGORITHMIC_REFERENCES.md](inst/ALGORITHMIC_REFERENCES.md).
