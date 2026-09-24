# Backend Installation and Validation

This page defines the build contract for CPU, Metal, and CUDA installations.
Backend names are strict: an unavailable requested accelerator raises an error
and is never relabelled CPU work.

## Diagnostic-only builds

The diagnostic-only build is intended for unavailable-backend error-path
testing:

```sh
FASTEMBEDR_DIAGNOSTIC_ONLY=1 R CMD INSTALL fastEmbedR_0.1.tar.gz
```

It must not be used as evidence of CUDA or Metal functionality.

## CPU-only build

```sh
FASTEMBEDR_USE_CUDA=0 R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

R supplies the BLAS, LAPACK, Fortran, OpenMP, and C++ runtime configuration.
fastEmbedR appends R's `LAPACK_LIBS`, `BLAS_LIBS`, and `FLIBS` on Linux rather
than replacing them.

## Metal build

Metal is compiled only on macOS with the required Apple SDK frameworks. Verify
the toolchain and installed result:

```sh
xcode-select -p
xcrun --find metal
FASTEMBEDR_USE_CUDA=0 R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

```r
x <- matrix(runif(1024 * 16), nrow = 1024)
fit <- fastEmbedR::umap(x, backend = "metal", n_neighbors = 15)
stopifnot(identical(fit$parameters$backend, "metal"))
```

## CUDA discovery and validation

The host NVIDIA driver is independent of the toolkit. fastEmbedR never manages
the driver. A controlled container should receive the host GPU only through the
container runtime's NVIDIA passthrough, such as `apptainer exec --nv`.

Configuration locates `nvcc` from `NVCC`, `CUDA_HOME`, `CUDA_PATH`,
`CUDAToolkit_ROOT`, or `PATH`. It searches both traditional and target-specific
toolkit layouts:

```text
include                         lib
include                         lib64
targets/<target>/include        targets/<target>/lib
targets/<target>/include        targets/<target>/lib64
```

For a split CCCL installation, set `CCCL_HOME`. Its root, `include`,
`include/cccl`, and target-specific include directories are searched for CUB
and Thrust. `FASTEMBEDR_CUDA_CPPFLAGS` and `FASTEMBEDR_CUDA_FLAGS` are applied
consistently to compile/link probes and package CUDA translation units.

Detection is not based on the presence of `nvcc` alone. Configure compiles and
links a C++17 CUDA program that references the CUDA runtime, cuFFT, cuBLAS,
cuSOLVER, and cuRAND, and verifies CUB and Thrust headers. It separately
compiles and links a cuVS program against `libcuvs_c` and `libcuvs`.

The package's CUDA KNN routes use cuVS brute force and IVF-Flat. RAFT TSVD is
optional and receives its own NVCC compile/link test when requested. CUDA PCA
always has a package-native rSVD route; a RAFT-enabled build may select RAFT
TSVD for matrix shapes where it is preferable.

## Compiler selection

The system compiler should precede CUDA or Conda wrappers in `PATH`. Set
`CUDAHOSTCXX` explicitly when R or RAPIDS lives in a Conda environment:

```sh
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${PATH}
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export CUVS_HOME=/opt/rapids
```

This does not replace R's compiler configuration. It ensures that NVCC uses a
supported host compiler while R retains its configured BLAS, LAPACK, Fortran,
OpenMP, and C++ runtime linkage.

## Strict CUDA build

```sh
PACKAGE_REQUIRE_CUDA=1 \
FASTEMBEDR_USE_CUDA=1 \
FASTEMBEDR_USE_CUVS=1 \
FASTEMBEDR_CUDA_ARCH="80 86 89 90" \
CUDA_HOME=/usr/local/cuda \
CUVS_HOME=/opt/rapids \
CUDAHOSTCXX=/usr/bin/g++ \
R CMD INSTALL --preclean fastEmbedR_0.1.tar.gz
```

Strict mode fails configuration when native CUDA or cuVS cannot be built.
`FASTEMBEDR_REQUIRE_CUDA=1` is an equivalent package-specific spelling.

Optional RAFT TSVD branch for automatic CUDA PCA:

```sh
FASTEMBEDR_USE_RAFT=1 \
RAFT_HOME=/opt/rapids \
RAPIDS_HOME=/opt/rapids
```

The architecture list affects fastEmbedR translation units only. cuVS and
RAFT must also contain compatible kernels or PTX.

## Runtime paths

Configure embeds runtime search paths for discovered library directories.
Transitive RAPIDS dependencies can still require:

```sh
export LD_LIBRARY_PATH=/opt/rapids/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

The runtime path must resolve libraries from the same compatible toolkit and
RAPIDS build used during compilation.

## Strict hardware validation

The maintained validation entry point is:

```sh
bash .github/scripts/run-hardware-validation.sh cuda /path/to/evidence
```

For CUDA it sets strict mode before `R CMD build`, builds one source archive,
installs that archive, loads the installed package, checks capability metadata,
runs real device-resident KNN/UMAP/t-SNE/PCA/clustering operations, compares
exact CUDA KNN with an independent base-R distance oracle, runs numerical t-SNE
validation, times `testthat`, and executes `R CMD check --as-cran`. The test
fails if metadata indicates CPU fallback or host-resident CUDA KNN output.

The default testthat limit is 180 seconds and can be changed only explicitly:

```sh
FASTEMBEDR_TESTTHAT_MAX_SECONDS=180 \
bash .github/scripts/run-hardware-validation.sh cuda /path/to/evidence
```

Evidence includes source and installed-library checksums, hardware information,
build/install/test/check logs, exact-KNN recall, and per-operation backend
metadata. A successful build without these runtime checks is not CUDA evidence.

## Clean containers

Use a uniquely named immutable image for each validation environment. Do not
modify an image associated with earlier evidence. A CUDA image contains a
toolkit and user-space libraries but no host driver; run it with NVIDIA
passthrough. Record the definition, build log, image SHA-256, source archive
SHA-256, compiler versions, toolkit version, cuVS/RAFT versions, and host driver
reported at runtime.
