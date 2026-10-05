/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

#include "massive_pca_cuda.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

thread_local std::string pca_error;

int check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return 0;
  pca_error = std::string(operation) + ": " +
    cudaGetErrorString(status);
  return 1;
}

int check(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) return 0;
  pca_error = std::string(operation) + ": cuBLAS status " +
    std::to_string(static_cast<int>(status));
  return 1;
}

struct Context {
  int rows;
  int columns;
  int rank;
  cublasHandle_t blas = nullptr;
  float* input = nullptr;
  float* output = nullptr;
  float* center = nullptr;
  float* scale = nullptr;
  float* loadings = nullptr;
  float* cross = nullptr;

  Context(int rows, int columns, int rank)
      : rows(rows), columns(columns), rank(rank) {}

  ~Context() {
    if (blas != nullptr) cublasDestroy(blas);
    if (input != nullptr) cudaFree(input);
    if (output != nullptr) cudaFree(output);
    if (center != nullptr) cudaFree(center);
    if (scale != nullptr) cudaFree(scale);
    if (loadings != nullptr) cudaFree(loadings);
    if (cross != nullptr) cudaFree(cross);
  }
};

__global__ void prepare(float* input, const float* center,
                        const float* scale, std::size_t items,
                        int columns) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (index >= items) return;
  const int col = static_cast<int>(index % columns);
  const float value = input[index] - center[col];
  input[index] = scale == nullptr ? value : value / scale[col];
}

int upload(Context* context, const float* input, int rows,
           bool project) {
  if (context == nullptr || input == nullptr || rows < 1 ||
      rows > context->rows || project != (context->rank > 0)) {
    pca_error = "Invalid streamed CUDA PCA chunk.";
    return 1;
  }
  const std::size_t items = static_cast<std::size_t>(rows) *
    context->columns;
  if (check(cudaMemcpy(context->input, input,
      items * sizeof(float), cudaMemcpyHostToDevice),
      "CUDA PCA input transfer")) return 1;
  prepare<<<(items + 255u) / 256u, 256>>>(
    context->input, context->center, context->scale,
    items, context->columns
  );
  return check(cudaGetLastError(), "CUDA PCA preprocessing");
}

int multiply(Context* context, const float* input, int rows) {
  if (upload(context, input, rows, true)) return 1;
  const float one = 1.0f;
  const float zero = 0.0f;
  return check(cublasSgemm(context->blas, CUBLAS_OP_T,
    CUBLAS_OP_N, context->rank, rows, context->columns, &one,
    context->loadings, context->columns, context->input,
    context->columns, &zero, context->output, context->rank),
    "CUDA PCA projection");
}

}  // namespace

extern "C" void* fastembedr_massive_pca_cuda_create(
    int rows, int columns, int rank, const float* center,
    const float* scale, const float* loadings) {
  pca_error.clear();
  if (rows < 1 || columns < 1 || (rank == 0 && columns > 2048) ||
      rank < 0 || rank > columns || center == nullptr ||
      (rank > 0 && (scale == nullptr || loadings == nullptr))) {
    pca_error = "Invalid streamed CUDA PCA configuration.";
    return nullptr;
  }
  const std::size_t input_items = static_cast<std::size_t>(rows) *
    columns;
  const std::size_t output_items = rank == 0 ?
    static_cast<std::size_t>(columns) * columns :
    static_cast<std::size_t>(rows) * rank;
  const std::size_t cross_items = columns > 1024 ?
    static_cast<std::size_t>(columns) * rank : 0;
  const std::size_t required = (input_items + output_items +
    3u * columns + static_cast<std::size_t>(columns) * rank +
    cross_items) * sizeof(float);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (check(cudaMemGetInfo(&free_bytes, &total_bytes),
            "CUDA PCA memory preflight")) return nullptr;
  if (required > 0.65 * static_cast<double>(free_bytes)) {
    pca_error = "Streamed CUDA PCA buffers exceed 65% of free VRAM.";
    return nullptr;
  }
  std::unique_ptr<Context> context(new Context(rows, columns, rank));
  if (check(cudaMalloc(&context->input, input_items * sizeof(float)),
            "CUDA PCA input allocation") ||
      check(cudaMalloc(&context->output, output_items * sizeof(float)),
            "CUDA PCA output allocation") ||
      check(cudaMalloc(&context->center, columns * sizeof(float)),
            "CUDA PCA center allocation")) return nullptr;
  if (check(cudaMemcpy(context->center, center,
      columns * sizeof(float), cudaMemcpyHostToDevice),
      "CUDA PCA center transfer")) return nullptr;
  if (rank > 0) {
    if (check(cudaMalloc(&context->scale,
          columns * sizeof(float)), "CUDA PCA scale allocation") ||
        check(cudaMalloc(&context->loadings,
          static_cast<std::size_t>(columns) * rank * sizeof(float)),
          "CUDA PCA loadings allocation")) return nullptr;
    if (check(cudaMemcpy(context->scale, scale,
          columns * sizeof(float), cudaMemcpyHostToDevice),
          "CUDA PCA scale transfer") ||
        check(cudaMemcpy(context->loadings, loadings,
          static_cast<std::size_t>(columns) * rank * sizeof(float),
          cudaMemcpyHostToDevice),
          "CUDA PCA loadings transfer")) return nullptr;
  }
  if (check(cublasCreate(&context->blas), "CUDA PCA cuBLAS setup") ||
      check(cublasSetMathMode(context->blas, CUBLAS_DEFAULT_MATH),
            "CUDA PCA math mode")) return nullptr;
  return context.release();
}

extern "C" int fastembedr_massive_pca_cuda_cross(
    void* state, const float* input, int rows, float* cross) {
  auto* context = static_cast<Context*>(state);
  if (cross == nullptr || upload(context, input, rows, false)) return 1;
  const float one = 1.0f;
  const float zero = 0.0f;
  const int p = context->columns;
  if (check(cublasSgemm(context->blas, CUBLAS_OP_N,
      CUBLAS_OP_T, p, p, rows, &one, context->input, p,
      context->input, p, &zero, context->output, p),
      "CUDA PCA crossproduct")) return 1;
  return check(cudaMemcpy(cross, context->output,
    static_cast<std::size_t>(p) * p * sizeof(float),
    cudaMemcpyDeviceToHost), "CUDA PCA crossproduct download");
}

extern "C" int fastembedr_massive_pca_cuda_project(
    void* state, const float* input, int rows, float* scores) {
  auto* context = static_cast<Context*>(state);
  if (scores == nullptr || multiply(context, input, rows)) return 1;
  return check(cudaMemcpy(scores, context->output,
    static_cast<std::size_t>(rows) * context->rank * sizeof(float),
    cudaMemcpyDeviceToHost), "CUDA PCA scores download");
}

extern "C" int fastembedr_massive_pca_cuda_action(
    void* state, const float* input, int rows, float* cross) {
  auto* context = static_cast<Context*>(state);
  if (cross == nullptr || multiply(context, input, rows)) return 1;
  if (context->cross == nullptr &&
      check(cudaMalloc(&context->cross,
        static_cast<std::size_t>(context->columns) * context->rank *
        sizeof(float)), "CUDA PCA action allocation")) return 1;
  const float one = 1.0f;
  const float zero = 0.0f;
  if (check(cublasSgemm(context->blas, CUBLAS_OP_N,
      CUBLAS_OP_T, context->columns, context->rank, rows, &one,
      context->input, context->columns, context->output,
      context->rank, &zero, context->cross, context->columns),
      "CUDA PCA covariance action")) return 1;
  return check(cudaMemcpy(cross, context->cross,
    static_cast<std::size_t>(context->columns) * context->rank *
    sizeof(float), cudaMemcpyDeviceToHost),
    "CUDA PCA action download");
}

extern "C" void fastembedr_massive_pca_cuda_destroy(void* state) {
  delete static_cast<Context*>(state);
}

extern "C" const char* fastembedr_massive_pca_cuda_error() {
  return pca_error.c_str();
}

extern "C" int fastembedr_massive_pca_cuda_memory(
    std::size_t* free_bytes, std::size_t* total_bytes, int* device) {
  pca_error.clear();
  if (free_bytes == nullptr || total_bytes == nullptr ||
      device == nullptr) {
    pca_error = "Invalid CUDA memory query output.";
    return 1;
  }
  return check(cudaGetDevice(device), "CUDA PCA device query") ||
    check(cudaMemGetInfo(free_bytes, total_bytes),
          "CUDA PCA memory query");
}

extern "C" int fastembedr_massive_cuda_device_count(int* count) {
  pca_error.clear();
  return check(cudaGetDeviceCount(count), "CUDA device count");
}

extern "C" int fastembedr_massive_cuda_select_device(int device) {
  pca_error.clear();
  return check(cudaSetDevice(device), "CUDA device selection");
}
