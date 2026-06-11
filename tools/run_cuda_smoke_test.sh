#!/bin/sh

set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
nvcc=${NVCC:-}
if [ -z "$nvcc" ]; then
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    nvcc="${CUDA_HOME}/bin/nvcc"
  else
    nvcc=$(command -v nvcc 2>/dev/null || true)
  fi
fi

if [ -z "$nvcc" ]; then
  echo "nvcc not found; skipping CUDA smoke test."
  exit 0
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/cuda_smoke.cu" <<'CU'
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_one(float* x) {
  x[0] += 1.0f;
}

int main() {
  int count = 0;
  cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(code));
    return 2;
  }
  if (count < 1) {
    std::fprintf(stderr, "No CUDA device is visible.\n");
    return 2;
  }

  float host = 1.0f;
  float* device = nullptr;
  code = cudaMalloc(reinterpret_cast<void**>(&device), sizeof(float));
  if (code != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(code));
    return 2;
  }
  cudaMemcpy(device, &host, sizeof(float), cudaMemcpyHostToDevice);
  add_one<<<1, 1>>>(device);
  code = cudaDeviceSynchronize();
  if (code != cudaSuccess) {
    std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(code));
    cudaFree(device);
    return 2;
  }
  cudaMemcpy(&host, device, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (host != 2.0f) {
    std::fprintf(stderr, "unexpected CUDA result: %f\n", host);
    return 2;
  }
  return 0;
}
CU

"$nvcc" -std=c++14 "$tmpdir/cuda_smoke.cu" -o "$tmpdir/cuda_smoke"
"$tmpdir/cuda_smoke"

cd "$repo_root"
FASTEMBEDR_USE_CUDA=${FASTEMBEDR_USE_CUDA:-1} \
FASTEMBEDR_USE_CUVS=${FASTEMBEDR_USE_CUVS:-auto} \
R CMD INSTALL .

Rscript - <<'RS'
library(fastEmbedR)
stopifnot(isTRUE(cuda_available()))
stopifnot(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
print(backend_info())

set.seed(1)
x <- scale(matrix(rnorm(80L * 6L), 80L, 6L))
cpu <- nn(x, k = 8L, backend = "cpu", n_threads = 4L)
gpu <- nn(x, k = 8L, backend = "cuda")
stopifnot(identical(attr(gpu, "backend"), "cuda"))
stopifnot(isTRUE(attr(gpu, "exact")))
stopifnot(identical(cpu$indices, gpu$indices))
stopifnot(max(abs(cpu$distances - gpu$distances)) < 1e-4)

pre <- fastEmbedR:::prepare_embedding_data(x, TRUE, pca_dims = 4L, seed = 1L, backend = "cuda")
stopifnot(identical(pre$preprocess$standardize_backend, "cuda"))
stopifnot(dim(pre$data)[2L] == 4L)

left <- matrix(rnorm(20L * 5L), 20L, 5L)
right <- matrix(rnorm(5L * 3L), 5L, 3L)
mm <- fastEmbedR:::rsvd_multiply_cuda_cpp(left, right, FALSE)
stopifnot(max(abs(mm - left %*% right)) < 1e-10)

x_wide <- scale(matrix(rnorm(90L * 40L), 90L, 40L))
pre_wide <- fastEmbedR:::prepare_embedding_data(x_wide, TRUE, pca_dims = 4L, seed = 1L, backend = "cuda")
stopifnot(identical(pre_wide$preprocess$standardize_backend, "cuda"))
stopifnot(identical(pre_wide$preprocess$pca_backend, "cuda_rsvd"))
stopifnot(dim(pre_wide$data)[2L] == 4L)

layout <- opentsne_knn(
  gpu,
  perplexity = 2,
  early_exaggeration_iter = 2L,
  n_iter = 3L,
  backend = "cuda"
)
cfg <- attr(layout, "fastEmbedR_config")
stopifnot(is.matrix(layout), ncol(layout) == 2L, all(is.finite(layout)))
stopifnot(identical(cfg$backend, "cuda"))
stopifnot(identical(cfg$optimizer, "opentsne_exact_dense_native_cuda"))
stopifnot(identical(cfg$probabilities, "dense_symmetric_knn_cuda_perplexity"))

fit <- opentsne(
  x,
  n_neighbors = 8L,
  perplexity = 2,
  early_exaggeration_iter = 2L,
  n_iter = 3L,
  backend = "cuda",
  silhouette_sample = NULL,
  preserve_sample = NULL,
  keep_knn = TRUE
)
stopifnot(inherits(fit, "fastEmbedR_embedding"))
stopifnot(identical(fit$parameters$backend, "cuda"))
stopifnot(identical(fit$parameters$nn_backend, "cuda"))
stopifnot(all(is.finite(fit$layout)))

query_knn <- nn(x[1:50, , drop = FALSE], x[51:60, , drop = FALSE], k = 8L, backend = "cuda")
projected <- transform_tsne(
  fit$layout[1:50, , drop = FALSE],
  knn = query_knn,
  perplexity = 2,
  n_iter = 3L,
  n_negatives = 20L,
  backend = "cuda",
  seed = 3L
)
tcfg <- attr(projected, "fastEmbedR_config")
stopifnot(is.matrix(projected), dim(projected)[1L] == 10L, dim(projected)[2L] == 2L)
stopifnot(all(is.finite(projected)))
stopifnot(identical(tcfg$backend, "cuda"))
stopifnot(identical(tcfg$optimizer, "opentsne_style_fixed_reference_transform_cuda"))

land <- landmark_tsne(
  x,
  landmarks = 0.5,
  n_neighbors = 8L,
  perplexity = 2,
  early_exaggeration_iter = 2L,
  n_iter = 3L,
  transform_iter = 3L,
  transform_perplexity = 2,
  backend = "cuda",
  standardize = FALSE,
  silhouette_sample = NULL,
  preserve_sample = NULL,
  keep_knn = TRUE
)
stopifnot(inherits(land, "fastEmbedR_embedding"))
stopifnot(identical(land$parameters$backend, "cuda"))
stopifnot(identical(land$parameters$transform_backend, "cuda"))
stopifnot(identical(land$parameters$projection_nn_backend, "cuda_fused_projection"))
stopifnot(all(is.finite(land$layout)))

scores <- evaluate_embedding(
  x,
  fit$layout,
  k = c(5L, 8L),
  reference_nn = fit$knn,
  backend = "cuda",
  n_threads = 4L
)
stopifnot(is.data.frame(scores), nrow(scores) == 1L)
stopifnot(identical(scores$metric_backend, "cuda"))
stopifnot(is.finite(scores$trustworthiness))

cat("CUDA smoke test passed.\n")
RS
