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
x <- matrix(rnorm(240), ncol = 6)
cpu <- nn(x, k = 8, backend = "cpu")
gpu <- nn(x, k = 8, backend = "cuda")
stopifnot(identical(cpu$indices, gpu$indices))
stopifnot(max(abs(cpu$distances - gpu$distances)) < 1e-4)
stopifnot(identical(attr(gpu, "backend"), "cuda"))
stopifnot(isTRUE(attr(gpu, "exact")))

if (cuvs_available()) {
  cuvs_exact <- nn(x, k = 8, backend = "cuda_cuvs_bruteforce")
  stopifnot(identical(attr(cuvs_exact, "backend"), "cuda_cuvs_bruteforce"))
  stopifnot(isTRUE(attr(cuvs_exact, "exact")))
  stopifnot(identical(cpu$indices, cuvs_exact$indices))
  stopifnot(max(abs(cpu$distances - cuvs_exact$distances)) < 1e-4)

  cuvs_auto_exact <- nn(x, k = 8, backend = "cuda_cuvs")
  stopifnot(identical(attr(cuvs_auto_exact, "backend"), "cuda_cuvs_bruteforce"))
  stopifnot(isTRUE(attr(cuvs_auto_exact, "exact")))

  cuvs_cagra <- nn(x, k = 8, backend = "cuda_cuvs_cagra")
  stopifnot(identical(attr(cuvs_cagra, "backend"), "cuda_cuvs_cagra"))
  stopifnot(!isTRUE(attr(cuvs_cagra, "exact")))
  stopifnot(is.matrix(cuvs_cagra$indices), ncol(cuvs_cagra$indices) == 8L)

  cuvs_nnd <- nn(x, k = 8, backend = "cuda_cuvs_nndescent")
  stopifnot(identical(attr(cuvs_nnd, "backend"), "cuda_cuvs_nndescent"))
  stopifnot(!isTRUE(attr(cuvs_nnd, "exact")))
  stopifnot(is.matrix(cuvs_nnd$indices), ncol(cuvs_nnd$indices) == 8L)

  old_cuvs_options <- options(fastEmbedR.cuvs_nndescent_threshold = 2L)
  cuvs_auto_nnd <- nn(x, k = 8, backend = "cuda_cuvs")
  options(old_cuvs_options)
  stopifnot(identical(attr(cuvs_auto_nnd, "backend"), "cuda_cuvs_nndescent"))
  stopifnot(!isTRUE(attr(cuvs_auto_nnd, "exact")))
} else {
  stopifnot(inherits(
    try(nn(x, k = 8, backend = "cuda_cuvs"), silent = TRUE),
    "try-error"
  ))
}

layout <- fastEmbedR:::fast_knn_umap(gpu, backend = "cuda", seed = 1)
stopifnot(is.matrix(layout), ncol(layout) == 2L, all(is.finite(layout)))
stopifnot(identical(attr(layout, "fastEmbedR_config")$backend, "cuda"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$init_backend, "cuda_fused_spectral"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$graph_prep_backend, "cuda_fused_csr"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$graph_storage, "native_cuda_coo_fused"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$gpu_optimizer_mode, "atomic"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$gpu_optimizer_update_rule, "native_cuda_atomic_coo_uwot_schedule"))
stopifnot(identical(attr(layout, "fastEmbedR_config")$gpu_transfer_policy, "single_upload_optimizer"))
stopifnot(isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_knn_uploaded_once))
stopifnot(!isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_init_uploaded_once))
stopifnot(isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_init_computed_on_device))
stopifnot(isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_embedding_returned_only_at_end))
stopifnot(!isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_init_roundtrip))
stopifnot(!isTRUE(attr(layout, "fastEmbedR_config")$gpu_transfer_graph_metadata_roundtrip))

set.seed(2)
large_x <- matrix(rnorm(3600), ncol = 6)
large_gpu <- nn(large_x, k = 8, backend = "cuda")
large_layout <- fastEmbedR:::fast_knn_umap(large_gpu, backend = "cuda", seed = 2)
stopifnot(is.matrix(large_layout), ncol(large_layout) == 2L, all(is.finite(large_layout)))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$backend, "cuda"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$init_backend, "cuda_fused_spectral"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$graph_prep_backend, "cuda_fused_csr"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$graph_storage, "native_cuda_coo_fused"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$gpu_optimizer_mode, "atomic"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$gpu_optimizer_update_rule, "native_cuda_atomic_coo_uwot_schedule"))
stopifnot(identical(attr(large_layout, "fastEmbedR_config")$gpu_transfer_policy, "single_upload_optimizer"))
stopifnot(isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_knn_uploaded_once))
stopifnot(!isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_init_uploaded_once))
stopifnot(isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_init_computed_on_device))
stopifnot(isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_embedding_returned_only_at_end))
stopifnot(!isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_init_roundtrip))
stopifnot(!isTRUE(attr(large_layout, "fastEmbedR_config")$gpu_transfer_graph_metadata_roundtrip))

approx_gpu <- nn(large_x, k = 8, backend = "cuda_approx")
stopifnot(is.matrix(approx_gpu$indices), nrow(approx_gpu$indices) == nrow(large_x), ncol(approx_gpu$indices) == 8L)
stopifnot(identical(attr(approx_gpu, "backend"), "cuda_approx"))
stopifnot(!isTRUE(attr(approx_gpu, "exact")))
stopifnot(identical(attr(approx_gpu, "approximation")$strategy, "anchor_projection_candidate_knn"))
stopifnot(is.data.frame(attr(approx_gpu, "recall")))
stopifnot(is.finite(attr(approx_gpu, "recall")$recall_at_k[1L]))
stopifnot(attr(approx_gpu, "recall")$sample_size[1L] > 0L)

tsne_layout <- fastEmbedR:::knn_tsne(gpu, backend = "cuda", seed = 1)
stopifnot(is.matrix(tsne_layout), ncol(tsne_layout) == 2L, all(is.finite(tsne_layout)))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$backend, "cuda"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$init_backend, "cpu_random"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$tsne_mode, "exact"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$graph_prep_backend, "cuda_exact"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$affinity_backend, "cuda"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$optimizer_backend, "cuda"))
stopifnot(identical(attr(tsne_layout, "fastEmbedR_config")$gpu_transfer_policy, "single_upload_optimizer"))
stopifnot(isTRUE(attr(tsne_layout, "fastEmbedR_config")$gpu_transfer_knn_uploaded_once))
stopifnot(isTRUE(attr(tsne_layout, "fastEmbedR_config")$gpu_transfer_embedding_returned_only_at_end))
stopifnot(!isTRUE(attr(tsne_layout, "fastEmbedR_config")$gpu_transfer_init_roundtrip))

tsne_exact_layout <- fastEmbedR:::knn_tsne(gpu, backend = "cuda", quality = "exact", seed = 1)
stopifnot(is.matrix(tsne_exact_layout), ncol(tsne_exact_layout) == 2L, all(is.finite(tsne_exact_layout)))
stopifnot(identical(attr(tsne_exact_layout, "fastEmbedR_config")$backend, "cuda"))
stopifnot(identical(attr(tsne_exact_layout, "fastEmbedR_config")$tsne_mode, "exact"))
stopifnot(identical(attr(tsne_exact_layout, "fastEmbedR_config")$graph_prep_backend, "cuda_exact"))
stopifnot(identical(attr(tsne_exact_layout, "fastEmbedR_config")$gpu_transfer_policy, "single_upload_optimizer"))

pacmap_layout <- fastEmbedR:::knn_pacmap(gpu, backend = "cuda", seed = 1)
stopifnot(is.matrix(pacmap_layout), ncol(pacmap_layout) == 2L, all(is.finite(pacmap_layout)))
stopifnot(identical(attr(pacmap_layout, "fastEmbedR_config")$backend, "cuda"))
stopifnot(identical(attr(pacmap_layout, "fastEmbedR_config")$init_backend, "cpu"))
stopifnot(identical(attr(pacmap_layout, "fastEmbedR_config")$graph_prep_backend, "cuda"))
stopifnot(identical(attr(pacmap_layout, "fastEmbedR_config")$gpu_transfer_policy, "single_upload_optimizer"))

fit <- umap(
  x,
  n_neighbors = 7,
  backend = "cuda",
  seed = 1,
  silhouette_sample = NULL,
  preserve_sample = NULL
)
stopifnot(inherits(fit, "fastEmbedR_embedding"))
stopifnot(identical(fit$parameters$backend, "cuda"))
stopifnot(identical(fit$parameters$nn_backend, "cuda"))
stopifnot(identical(fit$parameters$init_backend, "cuda_fused_spectral"))
stopifnot(identical(fit$parameters$graph_prep_backend, "cuda_fused_csr"))
stopifnot(identical(fit$parameters$graph_storage, "native_cuda_coo_fused"))
stopifnot(identical(fit$parameters$gpu_optimizer_mode, "atomic"))
stopifnot(identical(fit$parameters$gpu_optimizer_update_rule, "native_cuda_atomic_coo_uwot_schedule"))
stopifnot(identical(fit$parameters$gpu_transfer_policy, "single_upload_optimizer"))
stopifnot(isTRUE(fit$parameters$gpu_transfer_knn_uploaded_once))
stopifnot(!isTRUE(fit$parameters$gpu_transfer_init_uploaded_once))
stopifnot(isTRUE(fit$parameters$gpu_transfer_init_computed_on_device))
stopifnot(isTRUE(fit$parameters$gpu_transfer_embedding_returned_only_at_end))
stopifnot(!isTRUE(fit$parameters$gpu_transfer_init_roundtrip))
stopifnot(!isTRUE(fit$parameters$gpu_transfer_graph_metadata_roundtrip))
stopifnot(all(is.finite(fit$layout)))

projected <- transform_embedding(
  layout,
  reference_data = x,
  new_data = x[1:5, , drop = FALSE],
  k = 5,
  backend = "cuda"
)
stopifnot(is.matrix(projected), nrow(projected) == 5L, ncol(projected) == 2L)
stopifnot(identical(attr(projected, "backend"), "cuda"))
stopifnot(isTRUE(attr(projected, "exact")))

cat("CUDA smoke test passed.\n")
RS
