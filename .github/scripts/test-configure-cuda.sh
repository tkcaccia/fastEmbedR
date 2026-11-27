#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/fastembedr-config-test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
TEST_CXX="$(command -v clang++ || command -v g++)"

TOOLKIT="$TMP/cuda"
INCLUDE="$TOOLKIT/targets/x86_64-linux/include"
LIB="$TOOLKIT/targets/x86_64-linux/lib64"
RAPIDS="$TMP/rapids"
RAPIDS_INCLUDE="$RAPIDS/include/rapids"
RAPIDS_LIB="$RAPIDS/lib"
CUSTOM_CCCL="$TMP/custom-cccl"
CUSTOM_CCCL_INCLUDE="$CUSTOM_CCCL/include"
mkdir -p "$INCLUDE" "$LIB"
mkdir -p "$RAPIDS_INCLUDE/raft/core" "$RAPIDS_INCLUDE/raft/linalg"
mkdir -p "$RAPIDS_LIB" "$CUSTOM_CCCL_INCLUDE/cub"
mkdir -p "$CUSTOM_CCCL_INCLUDE/thrust/iterator"

cat > "$INCLUDE/cuda_runtime.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int cudaError_t;
int cudaRuntimeGetVersion(int* version);
#ifdef __cplusplus
}
#endif
EOF
cat > "$INCLUDE/cublas_v2.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef void* cublasHandle_t;
int cublasCreate_v2(cublasHandle_t* handle);
#ifdef __cplusplus
}
#endif
EOF
cat > "$INCLUDE/cufft.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int cufftHandle;
int cufftPlan1d(cufftHandle* plan, int n, int type, int batch);
#ifdef __cplusplus
}
#endif
EOF
cat > "$INCLUDE/cusolverDn.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef void* cusolverDnHandle_t;
int cusolverDnCreate(cusolverDnHandle_t* handle);
#ifdef __cplusplus
}
#endif
EOF
cat > "$INCLUDE/curand.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef void* curandGenerator_t;
int curandCreateGenerator(curandGenerator_t* generator, int rng_type);
#ifdef __cplusplus
}
#endif
EOF
printf '#pragma once\n' > "$CUSTOM_CCCL_INCLUDE/cub/cub.cuh"
printf '#pragma once\n' \
  > "$CUSTOM_CCCL_INCLUDE/thrust/iterator/counting_iterator.h"
printf '#pragma once\n' \
  > "$CUSTOM_CCCL_INCLUDE/fastembedr_custom_cccl.hpp"
cat > "$RAPIDS_INCLUDE/raft/core/handle.hpp" <<'EOF'
#pragma once
#include <fastembedr_custom_cccl.hpp>
namespace raft {
class handle_t {};
}
EOF
cat > "$RAPIDS_INCLUDE/raft/linalg/tsvd.cuh" <<'EOF'
#pragma once
#ifndef FASTEMBEDR_CUSTOM_CPPFLAG
#error FASTEMBEDR_CUDA_CPPFLAGS did not reach the RAFT probe
#endif
#ifndef FASTEMBEDR_CUSTOM_NVCC_FLAG
#error FASTEMBEDR_CUDA_FLAGS did not reach the RAFT probe
#endif
namespace raft {
namespace linalg {
struct paramsTSVD {
  int n_iterations = 0;
};
}
}
EOF
cat > "$TMP/cudart.c" <<'EOF'
int cudaRuntimeGetVersion(int* version) {
  if (version) *version = 13000;
  return 0;
}
EOF
cat > "$TMP/cublas.c" <<'EOF'
int cublasCreate_v2(void** handle) {
  if (handle) *handle = 0;
  return 0;
}
EOF
cat > "$TMP/cufft.c" <<'EOF'
int cufftPlan1d(int* plan, int n, int type, int batch) {
  (void)n; (void)type; (void)batch;
  if (plan) *plan = 0;
  return 0;
}
EOF
cat > "$TMP/cusolver.c" <<'EOF'
int cusolverDnCreate(void** handle) {
  if (handle) *handle = 0;
  return 0;
}
EOF
cat > "$TMP/curand.c" <<'EOF'
int curandCreateGenerator(void** generator, int rng_type) {
  (void)rng_type;
  if (generator) *generator = 0;
  return 0;
}
EOF
printf 'int fastembedr_fake_library(void) { return 0; }\n' > "$TMP/stub.c"

cc -shared -fPIC "$TMP/cudart.c" -o "$LIB/libcudart.so"
cc -shared -fPIC "$TMP/cublas.c" -o "$LIB/libcublas.so"
cc -shared -fPIC "$TMP/cufft.c" -o "$LIB/libcufft.so"
cc -shared -fPIC "$TMP/cusolver.c" -o "$LIB/libcusolver.so"
cc -shared -fPIC "$TMP/curand.c" -o "$LIB/libcurand.so"
for library in cublasLt cusparse; do
  cc -shared -fPIC "$TMP/stub.c" -o "$LIB/lib${library}.so"
done
for library in raft rmm rapids_logger; do
  cc -shared -fPIC "$TMP/stub.c" -o "$RAPIDS_LIB/lib${library}.so"
done

mkdir -p "$TOOLKIT/bin"
cat > "$TOOLKIT/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${NVCC_LOG:?}"
args=()
for arg in "$@"; do
  case "$arg" in
    -ccbin=*) ;;
    --extended-lambda) ;;
    --expt-relaxed-constexpr) ;;
    -x) ;;
    cu) ;;
    *) args+=("$arg") ;;
  esac
done
exec "${TEST_HOST_CXX:?}" -x c++ "${args[@]}"
EOF
chmod +x "$TOOLKIT/bin/nvcc"

GOOD="$TMP/good"
mkdir -p "$GOOD/src"
GOOD=$(CDPATH= cd -- "$GOOD" && pwd -P)
cp "$ROOT/configure" "$GOOD/configure"
chmod +x "$GOOD/configure"
: > "$TMP/nvcc.log"
if ! (
  cd "$GOOD"
  PATH="/usr/bin:/bin:$PATH" \
  TEST_HOST_CXX="$TEST_CXX" \
  NVCC_LOG="$TMP/nvcc.log" \
  CUDA_HOME="$TOOLKIT" \
  CCCL_HOME="$CUSTOM_CCCL" \
  RAFT_HOME="$RAPIDS" \
  NVCC="$TOOLKIT/bin/nvcc" \
  CUDAHOSTCXX="$TEST_CXX" \
  PACKAGE_REQUIRE_CUDA=1 \
  FASTEMBEDR_USE_RAFT=1 \
  FASTEMBEDR_CUDA_CPPFLAGS="-DFASTEMBEDR_CUSTOM_CPPFLAG=1" \
  FASTEMBEDR_CUDA_FLAGS="-DFASTEMBEDR_CUSTOM_NVCC_FLAG=1" \
  ./configure > "$TMP/good.log" 2>&1
); then
  cat "$TMP/good.log" >&2
  exit 1
fi

grep -F -- "-I$INCLUDE" "$GOOD/src/Makevars"
grep -F -- "-L$LIB" "$GOOD/src/Makevars"
grep -F -- "-Wl,-rpath,$LIB" "$GOOD/src/Makevars"
grep -F -- "-DFASTEMBEDR_HAS_CUDA" "$GOOD/src/Makevars"
if grep -Fq -- "-DFASTEMBEDR_HAS_CUVS" "$GOOD/src/Makevars"; then
  echo "fastEmbedR must obtain cuVS KNN through faissR." >&2
  exit 1
fi
grep -F -- "-DFASTEMBEDR_HAS_RAFT" "$GOOD/src/Makevars"
grep -F -- "-I$CUSTOM_CCCL_INCLUDE" "$GOOD/src/Makevars"
grep -F -- "-DFASTEMBEDR_CUSTOM_CPPFLAG=1" "$GOOD/src/Makevars"
grep -F -- "-DFASTEMBEDR_CUSTOM_NVCC_FLAG=1" "$GOOD/src/Makevars"
grep -F -- "-DNDEBUG" "$GOOD/src/Makevars"
grep -F -- 'all: $(SHLIB)' "$GOOD/src/Makevars"
if grep -Fq -- ".DEFAULT_GOAL" "$GOOD/src/Makevars"; then
  echo "Generated Makevars contains a GNU-only default-goal assignment." >&2
  exit 1
fi
grep -F -- "-ccbin=$TEST_CXX" "$GOOD/src/Makevars"
grep -F -- "-ccbin=$TEST_CXX" "$TMP/nvcc.log"
raft_probe=$(grep 'raft-link.cu' "$TMP/nvcc.log")
printf '%s\n' "$raft_probe" | grep -F -- "-I$CUSTOM_CCCL_INCLUDE"
printf '%s\n' "$raft_probe" | grep -F -- \
  "-DFASTEMBEDR_CUSTOM_CPPFLAG=1"
printf '%s\n' "$raft_probe" | grep -F -- \
  "-DFASTEMBEDR_CUSTOM_NVCC_FLAG=1"
cuda_probe=$(grep 'cuda-link.cu' "$TMP/nvcc.log")
printf '%s\n' "$cuda_probe" | grep -F -- "-I$CUSTOM_CCCL_INCLUDE"
printf '%s\n' "$cuda_probe" | grep -F -- \
  "-DFASTEMBEDR_CUSTOM_CPPFLAG=1"
printf '%s\n' "$cuda_probe" | grep -F -- \
  "-DFASTEMBEDR_CUSTOM_NVCC_FLAG=1"
grep -F -- "CCCL root:              $CUSTOM_CCCL" "$TMP/good.log"
grep '^PKG_CPPFLAGS.*-DFASTEMBEDR_CUSTOM_CPPFLAG=1' \
  "$GOOD/src/Makevars"

BROKEN="$TMP/broken"
mkdir -p "$BROKEN/src" "$BROKEN/cuda/bin" "$BROKEN/cuda/include"
cp "$ROOT/configure" "$BROKEN/configure"
cp "$TOOLKIT/bin/nvcc" "$BROKEN/cuda/bin/nvcc"
chmod +x "$BROKEN/configure" "$BROKEN/cuda/bin/nvcc"
if (
  cd "$BROKEN"
  NVCC_LOG="$TMP/nvcc-broken.log" \
  TEST_HOST_CXX="$TEST_CXX" \
  CUDA_HOME="$BROKEN/cuda" \
  NVCC="$BROKEN/cuda/bin/nvcc" \
  CUDAHOSTCXX="$TEST_CXX" \
  PACKAGE_REQUIRE_CUDA=1 \
  ./configure > "$TMP/broken.log" 2>&1
); then
  echo "Strict CUDA unexpectedly accepted a broken toolkit." >&2
  exit 1
fi
grep -F "toolkit compile/link probe failed" "$TMP/broken.log"

OBSOLETE="$TMP/obsolete"
mkdir -p "$OBSOLETE/src"
cp "$ROOT/configure" "$OBSOLETE/configure"
chmod +x "$OBSOLETE/configure"
if (
  cd "$OBSOLETE"
  FASTEMBEDR_USE_FAISS_GPU=1 ./configure > "$TMP/obsolete-faiss.log" 2>&1
); then
  echo "The obsolete FAISS GPU switch was unexpectedly accepted." >&2
  exit 1
fi
grep -F "FASTEMBEDR_USE_FAISS_GPU is obsolete" "$TMP/obsolete-faiss.log"
(
  cd "$OBSOLETE"
  FASTEMBEDR_USE_CUDA=0 FASTEMBEDR_USE_CUVS=1 ./configure \
    > "$TMP/obsolete-cuvs.log" 2>&1
)
grep -F "FASTEMBEDR_USE_CUVS is obsolete" "$TMP/obsolete-cuvs.log"

DIAGNOSTIC="$TMP/diagnostic"
mkdir -p "$DIAGNOSTIC/src"
cp "$ROOT/configure" "$DIAGNOSTIC/configure"
chmod +x "$DIAGNOSTIC/configure"
(
  cd "$DIAGNOSTIC"
  FASTEMBEDR_DIAGNOSTIC_ONLY=1 ./configure \
    > "$TMP/diagnostic.log" 2>&1
)
grep -F -- "-DFASTEMBEDR_DIAGNOSTIC_ONLY=1" \
  "$DIAGNOSTIC/src/Makevars"
if grep -Fq -- "-DFASTEMBEDR_HAS_CUDA" "$DIAGNOSTIC/src/Makevars"; then
  echo "Diagnostic-only mode unexpectedly enabled CUDA." >&2
  exit 1
fi

echo "CUDA configure regression tests passed."
