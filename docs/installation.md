# Installation

[Home](../README.md) |
**Installation** |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[API](usage-api.md)

## CPU

A CPU build needs R, Rcpp, and the C++17 toolchain configured for that R
installation. It does not need CUDA, cuVS, or a linked FAISS library.

```sh
FASTEMBEDR_USE_CUDA=0 \
R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

The CPU HNSW source is compiled into fastEmbedR. Its retained FAISS notice
describes source provenance; it does not imply a runtime libfaiss dependency.

## Apple Metal

On supported Apple Silicon, the package builds against Foundation, Accelerate,
Metal, Metal Performance Shaders (MPS), and MPSGraph from the Apple SDK. Use
macOS 14 or newer and a full Xcode 15 or newer installation. Intel Macs use the
CPU backend.

```sh
xcode-select -p
xcrun --sdk macosx --show-sdk-path
FASTEMBEDR_USE_CUDA=0 R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

## NVIDIA driver and CUDA toolkit

The NVIDIA driver and CUDA toolkit are separate components. The host driver
controls the GPU and is normally managed by the system administrator. The CUDA
toolkit supplies `nvcc`, headers, and link libraries used to build fastEmbedR.
The package never installs, removes, downgrades, or modifies the host driver.

Do not replace a working vendor driver merely to install a distribution CUDA
metapackage. Install a compatible toolkit in a separate prefix, or use a
controlled container with NVIDIA GPU passthrough. The driver must be new enough
for the toolkit/runtime used in that environment.

Package-native CUDA embedding and rSVD require:

- the CUDA runtime, cuFFT, cuBLAS, cuSOLVER, and cuRAND;
- CUB and Thrust headers from CUDA Core Compute Libraries (CCCL).

Those components are supplied by a normal CUDA toolkit installation. Complete
one-call CUDA workflows starting from a data matrix additionally require the
RAPIDS cuVS C and C++ libraries for exact and IVF-Flat KNN. CUDA embedding from
a precomputed KNN object does not require cuVS.

RAFT/RMM provide an optional TSVD branch of the automatic CUDA PCA selector.
Package-native CUDA rSVD remains available without RAFT and uses cuBLAS,
cuSOLVER, and cuRAND from the CUDA toolkit.

## Strict CUDA installation

Keep the system compiler before CUDA or Conda compiler wrappers in `PATH` and
select the host compiler explicitly. `nvcc` may invoke this compiler.

```sh
export CUDA_HOME=/usr/local/cuda
export CCCL_HOME=/opt/cccl
export CUVS_HOME=/opt/rapids
export CUDAHOSTCXX=/usr/bin/g++
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${PATH}
export FASTEMBEDR_CUDA_ARCH="89"

PACKAGE_REQUIRE_CUDA=1 \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_CUVS=1 \
R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

`PACKAGE_REQUIRE_CUDA=1` and its package-specific alias
`FASTEMBEDR_REQUIRE_CUDA=1` are equivalent. Strict mode compiles and links a
CUDA test and a cuVS test during configuration. Installation fails if either
test fails; it cannot produce a CPU fallback build.

To enable the optional RAFT TSVD branch of CUDA PCA, also set:

```sh
export RAFT_HOME=/opt/rapids
export RAPIDS_HOME=/opt/rapids
export FASTEMBEDR_USE_RAFT=1
```

Without these RAFT settings, CUDA PCA and CUDA t-SNE PCA initialization use
the package-native rSVD implementation; they do not fall back to CPU.

## Discovery variables

The toolkit root is resolved from `CUDA_HOME`, `CUDA_PATH`, or
`CUDAToolkit_ROOT`; `NVCC` may name `nvcc` directly. For every toolkit root,
configuration searches:

```text
$CUDA_HOME/include
$CUDA_HOME/lib
$CUDA_HOME/lib64
$CUDA_HOME/targets/*/include
$CUDA_HOME/targets/*/lib
$CUDA_HOME/targets/*/lib64
```

When CUDA Core Compute Libraries (CCCL) are installed separately, set
`CCCL_HOME`. Configuration searches its root, `include`, `include/cccl`, and
the corresponding `targets/*/include` directories for CUB and Thrust.

Use `CUVS_HOME`, `RAFT_HOME`, and `RAPIDS_HOME` for external prefixes.
Configure records runtime search paths for detected CUDA, cuVS, and RAFT/RMM
libraries. A matching `LD_LIBRARY_PATH` may still be needed for transitive
dependencies:

`FASTEMBEDR_CUDA_CPPFLAGS` supplies CUDA preprocessor and include flags.
`FASTEMBEDR_CUDA_FLAGS` supplies general NVCC flags. Both are applied to CUDA
configure probes and package translation units, so custom toolchain settings
are validated before package compilation begins.

```sh
export LD_LIBRARY_PATH=/opt/rapids/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

## Verify the installed backend

```r
library(fastEmbedR)

set.seed(1)
x <- matrix(runif(2048 * 32), nrow = 2048)
knn <- precompute_knn(x, k = 15, backend = "cuda")
fit <- umap_knn(knn, backend = "cuda")
stopifnot(
    inherits(knn, "fastEmbedR_gpu_knn"),
    identical(knn$result_residency, "cuda"),
    identical(knn$device_to_host_result_copies, 0),
    !isTRUE(knn$cpu_fallback),
    identical(attr(fit, "fastEmbedR_config")$backend, "cuda")
)
```

The strict explicit request and returned backend metadata prevent a CPU result
from being accepted as CUDA evidence.

## Common failures

- `nvcc was not found`: set `CUDA_HOME` or `NVCC` to the toolkit, not the
  driver installation.
- CUDA compile/link probe failure: inspect configure output for missing CUDA
  headers or libraries and mixed target prefixes.
- cuVS probe failure: make `cuvs/core/c_api.h`, `libcuvs_c`, and `libcuvs`
  available under one compatible prefix.
- `cudaErrorInsufficientDriver`: use a toolkit/runtime compatible with the
  installed driver; do not let the package change the driver.
- `cudaErrorNoKernelImageForDevice`: rebuild fastEmbedR and external CUDA
  libraries for the GPU compute capability.
- undefined `__nvJitLink...`: CUDA and RAPIDS libraries from different
  releases are being mixed. Correct their library search order.
- missing `libcuvs_c.so`: restore the runtime path matching the build prefix.

See [Backend installation](installation-backends.md) for the complete build
and validation contract.
