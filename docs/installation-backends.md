# Installation And Backends

This page describes `fastEmbedR` embedding backends. FAISS/cuVS
nearest-neighbour installation belongs to the companion
[`faissR`](https://github.com/tkcaccia/faissR) project.

## Standard Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
remotes::install_github("tkcaccia/fastEmbedR")
```

Install and validate `faissR` first when using matrix-input `opentsne()` or
`umap()`, because those functions call `faissR::nn()` internally.

## Backend Rule

The public embedding backends are:

- `backend = "cpu"`;
- `backend = "metal"` on Apple Silicon when native Metal symbols are compiled;
- `backend = "cuda"` when native CUDA symbols are compiled.

Explicit unavailable GPU requests fail clearly. They are never reported as GPU
after running on CPU.

## Metal

On Apple Silicon, `umap_knn(..., backend = "metal")` uses the native
Objective-C++/Metal UMAP optimizer from the supplied KNN graph. It does not call
Python, `reticulate`, Torch, or MLX.

Native Metal openTSNE uses the package Metal FFT-grid path when compiled. The
validated default uses PCA initialization for matrix-input workflows and the
current Metal FFT-grid implementation for the negative-gradient approximation.

## CUDA

Native CUDA openTSNE is implemented with CUDA kernels and cuFFT when the CUDA
backend is compiled. Native CUDA UMAP uses the package CUDA optimizer when
available.

Compile the CUDA embedding backend with:

```sh
CUDA_HOME=/usr/local/cuda \
FASTEMBEDR_USE_CUDA=1 \
R CMD INSTALL /path/to/fastEmbedR
```

CUDA KNN is still provided by `faissR`; see the `faissR` installation page for
FAISS GPU and RAPIDS cuVS details.

## Diagnostics

```r
library(fastEmbedR)

fastEmbedR::metal_available()
fastEmbedR::cuda_available()
faissR::backend_info()
```
