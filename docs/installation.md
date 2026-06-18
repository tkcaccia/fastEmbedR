# Installation

[Home](../README.md) |
**Installation** |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[References](references.md)

`fastEmbedR` is split from the nearest-neighbour package `faissR`.

- `faissR` provides FAISS/cuVS KNN, candidate KNN, graph construction,
  kNN prediction, and k-means.
- `fastEmbedR` provides UMAP, openTSNE-style t-SNE, landmark transforms, and
  embedding metrics.

## R Packages

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
remotes::install_github("tkcaccia/fastEmbedR")
```

Suggested benchmark/reference packages:

```r
install.packages(c("Rtsne", "uwot", "umap", "igraph", "jsonlite", "knitr", "rmarkdown"))
```

`Rtsne`, `uwot`, and `umap` are optional comparison packages. They are not
required by the core `fastEmbedR` embedding functions.

## Core Build Dependencies

`fastEmbedR` needs:

- R;
- a C++17 compiler;
- `Rcpp`;
- the companion package `faissR`;
- macOS Metal framework for native Metal embedding kernels on Apple Silicon;
- CUDA toolkit for optional native CUDA embedding kernels.

`faissR` owns FAISS/cuVS KNN installation. Install and validate `faissR` first
using the instructions in the [`faissR` GitHub project](https://github.com/tkcaccia/faissR),
then install `fastEmbedR`.

## CUDA Embedding Build

CUDA KNN is provided by `faissR`. `fastEmbedR` only needs CUDA when compiling
native CUDA embedding kernels for UMAP/openTSNE.

```sh
CUDA_HOME=/usr/local/cuda \
FASTEMBEDR_USE_CUDA=1 \
R CMD INSTALL /path/to/fastEmbedR
```

If CUDA is requested explicitly and unavailable, the embedding function fails
clearly. It does not run on CPU while reporting CUDA.

## Apple Metal

On Apple Silicon, `fastEmbedR` builds native Objective-C++/Metal embedding
kernels for:

- UMAP layout optimization from KNN;
- openTSNE FFT-grid optimization;
- selected projection/refinement operations.

No Python, Torch, MLX, or `reticulate` call is required for the public Metal
embedding paths.

## Backend Check

After installation:

```r
library(fastEmbedR)
faissR::backend_info()
fastEmbedR::metal_available()
```

## Backend Rule

Backend labels are strict. An explicit GPU request must resolve to a real
native GPU backend. Otherwise the function errors and reports what dependency
is missing.
