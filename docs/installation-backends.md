# Installation And Backends

This page describes installation, optional native libraries, and backend rules
for `fastEmbedR`.

## Standard Installation

```r
install.packages("remotes")
remotes::install_github("tkcaccia/faissR")
remotes::install_github("tkcaccia/fastEmbedR")
```

Native KNN backends are implemented in the companion `faissR` package.
FAISS/cuVS KNN availability is controlled by the `faissR` build. Explicit unavailable GPU,
FAISS, or cuVS backends fail clearly rather than silently running on CPU.

For a local macOS FAISS build through conda-forge:

```sh
conda create -n fastembedr-faiss -c conda-forge faiss-cpu r-base r-rcpp
conda activate fastembedr-faiss
FAISSR_USE_FAISS=1 FAISS_HOME="$CONDA_PREFIX" R CMD INSTALL /path/to/faissR
R CMD INSTALL /path/to/fastEmbedR
```

## Optional RAPIDS cuVS KNN

`faissR` includes native C++ bindings for RAPIDS cuVS KNN, including
brute-force search, CAGRA, and NN-descent. The RAPIDS libraries themselves are
not vendored into `fastEmbedR` because the CUDA binary stack is large and must
match the host NVIDIA driver/CUDA runtime.

On a CUDA Linux machine, install the external cuVS SDK with:

```sh
tools/install_cuvs_linux.sh
. ~/.fastEmbedR/cuvs_env.sh
R CMD INSTALL .
```

For the chiamaka CUDA workstation (`chiamaka@137.158.224.178`), use the same
script with an explicit RAPIDS CUDA version because the machine has a new
NVIDIA driver but an older system `nvcc`:

```sh
FASTEMBEDR_CUVS_CUDA_VERSION=12.9 tools/install_cuvs_linux.sh
. ~/.fastEmbedR/cuvs_env.sh
R CMD INSTALL /path/to/faissR
FASTEMBEDR_USE_CUDA=1 R CMD INSTALL /path/to/fastEmbedR
```

The activation file sets `CUVS_HOME`, `CUDA_HOME`, `NVCC`, and CUDA/cuVS build
flags, so `faissR` and `fastEmbedR` build against the micromamba cuVS/CUDA
environment instead of `/usr/bin/nvcc`.

After installation:

```r
library(fastEmbedR)
faissR::cuvs_available()
faissR::backend_info()
knn <- faissR::nn(x, k = 50, backend = "cuda_cuvs_nndescent")
```

If cuVS is unavailable, explicit cuVS requests fail clearly. They are never
reported as CUDA/cuVS after running on CPU.

## Backend Rule

All public compute functions that can use parallel CPU work expose
`n_threads`; use `n_threads = 4` for a fixed four-core CPU run. Functions with
native GPU support also accept `backend = "metal"` on Apple Silicon. The
convenience `backend = "gpu"` requests a real native GPU path where the
function has one; unsupported GPU work fails clearly instead of being labelled
as GPU after running on CPU.

## Metal

On Apple Silicon, `umap_knn(..., backend = "metal")` uses the restored native
Objective-C++/Metal UMAP optimizer from the supplied KNN graph. It does not
call Python, `reticulate`, Torch, or MLX.

Metal UMAP has a single optimizer path: `atomic_inplace`. This is the visually
validated native Metal edge-update kernel used for benchmarks; slower or
distorted experimental Metal UMAP optimizers were removed to keep results
reproducible and the API simple.

Native Metal openTSNE is available when the `knn_tsne_opentsne_metal_cpp`
symbol is compiled. `negative_gradient_method = "auto"` resolves to the native
Metal FFT-grid path, which keeps the interpolation grid, FFT convolution,
sparse attractive forces, gains, and updates inside Objective-C++/Metal
kernels. If the native symbol is unavailable, `opentsne_knn(..., backend =
"metal")` fails clearly instead of falling back to CPU and reporting a GPU
result.

The package does not opt in to an MPSGraph FFT backend by default because the
local MNIST benchmark did not show a sufficient quality/speed advantage over
the package-native Metal FFT-grid path. MPSGraph is kept diagnostic-only.

The main remaining Metal openTSNE speed gap is the FFT itself. CUDA uses NVIDIA
cuFFT, while the Mac path uses package-native Metal FFT kernels. The 512x512
openTSNE/FIt-SNE grid path uses a validated Stockham FFT kernel implemented
using the MIT-licensed AppleSiliconFFT design as a reference; other grid sizes
use the generic package-native Metal Cooley-Tukey path. See
[metal-fft-roadmap.md](metal-fft-roadmap.md) and profile the current FFT
stages with:

```r
system("Rscript tools/profile_metal_opentsne_fft.R --n=10000 --k=50")
```

## CUDA

Native CUDA FFT openTSNE is implemented with CUDA kernels and cuFFT when the
CUDA backend is compiled. Native CUDA UMAP uses the fused pure-atomic UMAP path
when available. Unsupported CUDA requests fail clearly instead of falling back
to CPU.

Optional RAPIDS cuVS is used for CUDA KNN only. The package does not vendor or
call RAPIDS cuML UMAP/openTSNE at runtime.
