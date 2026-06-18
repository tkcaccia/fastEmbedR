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

`faissR` needs:

- FAISS for CPU KNN;
- CUDA and RAPIDS cuVS for optional CUDA KNN;
- a C++17 compiler compatible with the installed FAISS/cuVS libraries.

## CPU And FAISS

`fastEmbedR` calls `faissR::nn()` internally for one-call embeddings. On CPU,
FAISS should be
available through `faissR`.

Example conda-forge FAISS CPU setup:

```sh
conda create -n fastembedr-faiss -c conda-forge faiss-cpu r-base r-rcpp
conda activate fastembedr-faiss

FAISS_HOME="$CONDA_PREFIX" FAISSR_USE_FAISS=1 \
  R CMD INSTALL /path/to/faissR

R CMD INSTALL /path/to/fastEmbedR
```

## CUDA / cuVS

CUDA KNN is owned by `faissR`. The preferred CUDA graph backend is RAPIDS cuVS
where available. `fastEmbedR` consumes the KNN object and runs native CUDA
embedding kernels for UMAP and openTSNE when compiled.

Example:

```sh
CUDA_HOME=/usr/local/cuda \
CUVS_HOME=/path/to/cuvs \
FAISSR_USE_FAISS=1 \
FAISSR_USE_CUDA=1 \
FAISSR_USE_CUVS=1 \
R CMD INSTALL /path/to/faissR

CUDA_HOME=/usr/local/cuda \
FASTEMBEDR_USE_CUDA=1 \
R CMD INSTALL /path/to/fastEmbedR
```

If CUDA/cuVS is requested explicitly and unavailable, the functions fail
clearly. They do not run on CPU while reporting CUDA.

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
faissR::faiss_available()
faissR::cuda_available()
faissR::cuvs_available()
faissR::metal_available()
```

## Backend Rule

Backend labels are strict. An explicit GPU request must resolve to a real
native GPU backend. Otherwise the function errors and reports what dependency
is missing.
