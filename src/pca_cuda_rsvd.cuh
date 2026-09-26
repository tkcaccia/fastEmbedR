/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Float32 CUDA randomized SVD adapted from fastPLS commit
 * 82bbc48a0d69e4bd0d7c261fdcc8e636694133b1. The adaptation keeps only the
 * dense PCA decomposition needed by fastEmbedR.
 */

#ifndef FASTEMBEDR_PCA_CUDA_RSVD_CUH
#define FASTEMBEDR_PCA_CUDA_RSVD_CUH

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

namespace fastembedr_pca {

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status)
    );
  }
}

inline void require_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(operation) + " failed with cuBLAS status " +
      std::to_string(static_cast<int>(status))
    );
  }
}

inline void require_solver(cusolverStatus_t status, const char* operation) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(operation) + " failed with cuSOLVER status " +
      std::to_string(static_cast<int>(status))
    );
  }
}

inline void require_random(curandStatus_t status, const char* operation) {
  if (status != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(operation) + " failed with cuRAND status " +
      std::to_string(static_cast<int>(status))
    );
  }
}

__global__ void collect_solver_status(const int* info, int* invalid) {
  if (*info != 0) atomicExch(invalid, 1);
}

class RsvdHandleCache {
 public:
  RsvdHandleCache() {
    try {
      require_blas(cublasCreate(&blas_), "cublasCreate(rSVD cache)");
      require_blas(
        cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH),
        "cublasSetMathMode(rSVD cache)"
      );
      require_solver(
        cusolverDnCreate(&solver_), "cusolverDnCreate(rSVD cache)"
      );
      require_random(
        curandCreateGenerator(&random_, CURAND_RNG_PSEUDO_DEFAULT),
        "curandCreateGenerator(rSVD cache)"
      );
    } catch (...) {
      release();
      throw;
    }
  }

  RsvdHandleCache(const RsvdHandleCache&) = delete;
  RsvdHandleCache& operator=(const RsvdHandleCache&) = delete;

  ~RsvdHandleCache() { release(); }

  void release() noexcept {
    if (random_ != nullptr) curandDestroyGenerator(random_);
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    random_ = nullptr;
    solver_ = nullptr;
    blas_ = nullptr;
  }

  void bind(cudaStream_t stream) {
    require_blas(
      cublasSetStream(blas_, stream), "cublasSetStream(rSVD cache)"
    );
    require_solver(
      cusolverDnSetStream(solver_, stream),
      "cusolverDnSetStream(rSVD cache)"
    );
    require_random(
      curandSetStream(random_, stream), "curandSetStream(rSVD cache)"
    );
  }

  cublasHandle_t blas() const { return blas_; }
  cusolverDnHandle_t solver() const { return solver_; }
  curandGenerator_t random() const { return random_; }

 private:
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  curandGenerator_t random_ = nullptr;
};

inline RsvdHandleCache& rsvd_handle_cache() {
  static thread_local RsvdHandleCache cache;
  return cache;
}

class RsvdWorkspace {
 public:
  RsvdWorkspace(int rows, int columns, int rank, int oversample,
                int power, cudaStream_t stream)
      : rows_(rows), columns_(columns), rank_(rank),
        sketch_(rank + std::min(
          oversample, std::min(rows, columns) - rank
        )), power_(power), stream_(stream) {
    if (rows < 1 || columns < 1 || rank < 1 ||
        rank > std::min(rows, columns) || oversample < 0 || power < 0) {
      throw std::invalid_argument("invalid CUDA rSVD dimensions or controls");
    }
    try {
      create_handles();
      allocate_buffers();
      configure_workspace();
    } catch (...) {
      release();
      throw;
    }
  }

  RsvdWorkspace(const RsvdWorkspace&) = delete;
  RsvdWorkspace& operator=(const RsvdWorkspace&) = delete;

  ~RsvdWorkspace() { release(); }

  void solve(const float* matrix, unsigned long long seed,
             float* left, float* right, float* singular) {
    require_cuda(
      cudaMemsetAsync(invalid_, 0, sizeof(int), stream_),
      "cudaMemsetAsync(rSVD status)"
    );
    require_random(
      curandSetPseudoRandomGeneratorSeed(random_, seed),
      "curandSetPseudoRandomGeneratorSeed"
    );
    require_random(
      curandSetGeneratorOffset(random_, 0),
      "curandSetGeneratorOffset"
    );
    const std::size_t random_count =
      (static_cast<std::size_t>(columns_) * sketch_ + 1u) / 2u * 2u;
    require_random(
      curandGenerateNormal(random_, omega_, random_count, 0.0f, 1.0f),
      "curandGenerateNormal(rSVD sketch)"
    );

    multiply(
      CUBLAS_OP_N, CUBLAS_OP_N, rows_, sketch_, columns_,
      matrix, rows_, omega_, columns_, basis_, rows_
    );
    orthonormalize(basis_, rows_);
    for (int iteration = 0; iteration < power_; ++iteration) {
      multiply(
        CUBLAS_OP_T, CUBLAS_OP_N, columns_, sketch_, rows_,
        matrix, rows_, basis_, rows_, right_basis_, columns_
      );
      orthonormalize(right_basis_, columns_);
      multiply(
        CUBLAS_OP_N, CUBLAS_OP_N, rows_, sketch_, columns_,
        matrix, rows_, right_basis_, columns_, basis_, rows_
      );
      orthonormalize(basis_, rows_);
    }

    multiply(
      CUBLAS_OP_T, CUBLAS_OP_N, columns_, sketch_, rows_,
      matrix, rows_, basis_, rows_, small_, columns_
    );
    require_solver(
      cusolverDnSgesvd(
        solver_, 'S', 'S', columns_, sketch_, small_, columns_,
        singular_all_, small_u_, columns_, small_vt_, sketch_, work_,
        work_size_, rwork_, info_
      ),
      "cusolverDnSgesvd(rSVD reduced matrix)"
    );
    collect_solver_status<<<1, 1, 0, stream_>>>(info_, invalid_);
    require_cuda(
      cudaGetLastError(), "collect_solver_status(rSVD SVD)"
    );
    multiply(
      CUBLAS_OP_N, CUBLAS_OP_T, rows_, rank_, sketch_,
      basis_, rows_, small_vt_, sketch_, left, rows_
    );
    require_cuda(
      cudaMemcpyAsync(
        right, small_u_,
        static_cast<std::size_t>(columns_) * rank_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_
      ),
      "cudaMemcpyAsync(rSVD right vectors)"
    );
    require_cuda(
      cudaMemcpyAsync(
        singular, singular_all_,
        static_cast<std::size_t>(rank_) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_
      ),
      "cudaMemcpyAsync(rSVD singular values)"
    );
    require_cuda(cudaGetLastError(), "CUDA rSVD solve");
  }

  void synchronize_and_validate() {
    require_cuda(
      cudaStreamSynchronize(stream_), "cudaStreamSynchronize(rSVD)"
    );
    int invalid = 0;
    require_cuda(
      cudaMemcpy(
        &invalid, invalid_, sizeof(int), cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(rSVD status)"
    );
    if (invalid != 0) {
      throw std::runtime_error("CUDA rSVD factorization did not converge");
    }
  }

 private:
  void create_handles() {
    RsvdHandleCache& cache = rsvd_handle_cache();
    cache.bind(stream_);
    blas_ = cache.blas();
    solver_ = cache.solver();
    random_ = cache.random();
  }

  void allocate(float*& pointer, std::size_t count) {
#if CUDART_VERSION >= 11020
    require_cuda(
      cudaMallocAsync(
        reinterpret_cast<void**>(&pointer), count * sizeof(float), stream_
      ),
      "cudaMallocAsync(rSVD buffer)"
    );
#else
    require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)),
      "cudaMalloc(rSVD buffer)"
    );
#endif
  }

  void allocate_buffers() {
    const std::size_t padded_random =
      (static_cast<std::size_t>(columns_) * sketch_ + 1u) / 2u * 2u;
    allocate(omega_, padded_random);
    allocate(basis_, static_cast<std::size_t>(rows_) * sketch_);
    allocate(right_basis_, static_cast<std::size_t>(columns_) * sketch_);
    allocate(small_, static_cast<std::size_t>(columns_) * sketch_);
    allocate(small_u_, static_cast<std::size_t>(columns_) * sketch_);
    allocate(small_vt_, static_cast<std::size_t>(sketch_) * sketch_);
    allocate(singular_all_, sketch_);
    allocate(tau_, sketch_);
    allocate(rwork_, static_cast<std::size_t>(std::max(1, sketch_ - 1)));
#if CUDART_VERSION >= 11020
    require_cuda(
      cudaMallocAsync(
        reinterpret_cast<void**>(&info_), sizeof(int), stream_
      ),
      "cudaMallocAsync(rSVD info)"
    );
    require_cuda(
      cudaMallocAsync(
        reinterpret_cast<void**>(&invalid_), sizeof(int), stream_
      ),
      "cudaMallocAsync(rSVD invalid)"
    );
#else
    require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&info_), sizeof(int)),
      "cudaMalloc(rSVD info)"
    );
    require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&invalid_), sizeof(int)),
      "cudaMalloc(rSVD invalid)"
    );
#endif
  }

  void configure_workspace() {
    int requested = 0;
    for (const int rows : {rows_, columns_}) {
      float* matrix = rows == rows_ ? basis_ : right_basis_;
      require_solver(
        cusolverDnSgeqrf_bufferSize(
          solver_, rows, sketch_, matrix, rows, &requested
        ),
        "cusolverDnSgeqrf_bufferSize(rSVD)"
      );
      work_size_ = std::max(work_size_, requested);
      require_solver(
        cusolverDnSorgqr_bufferSize(
          solver_, rows, sketch_, sketch_, matrix, rows, tau_, &requested
        ),
        "cusolverDnSorgqr_bufferSize(rSVD)"
      );
      work_size_ = std::max(work_size_, requested);
    }
    require_solver(
      cusolverDnSgesvd_bufferSize(
        solver_, columns_, sketch_, &requested
      ),
      "cusolverDnSgesvd_bufferSize(rSVD)"
    );
    work_size_ = std::max(work_size_, requested);
    allocate(work_, static_cast<std::size_t>(work_size_));
  }

  void multiply(cublasOperation_t left_operation,
                cublasOperation_t right_operation,
                int rows, int columns, int inner,
                const float* left, int left_leading,
                const float* right, int right_leading,
                float* output, int output_leading) {
    const float one = 1.0f;
    const float zero = 0.0f;
    require_blas(
      cublasSgemm(
        blas_, left_operation, right_operation,
        rows, columns, inner, &one,
        left, left_leading, right, right_leading,
        &zero, output, output_leading
      ),
      "cublasSgemm(rSVD)"
    );
  }

  void orthonormalize(float* matrix, int rows) {
    require_solver(
      cusolverDnSgeqrf(
        solver_, rows, sketch_, matrix, rows, tau_, work_, work_size_, info_
      ),
      "cusolverDnSgeqrf(rSVD)"
    );
    collect_solver_status<<<1, 1, 0, stream_>>>(info_, invalid_);
    require_solver(
      cusolverDnSorgqr(
        solver_, rows, sketch_, sketch_, matrix, rows, tau_, work_,
        work_size_, info_
      ),
      "cusolverDnSorgqr(rSVD)"
    );
    collect_solver_status<<<1, 1, 0, stream_>>>(info_, invalid_);
    require_cuda(
      cudaGetLastError(), "collect_solver_status(rSVD QR)"
    );
  }

  void release() noexcept {
    for (float* pointer : {
           omega_, basis_, right_basis_, small_, small_u_, small_vt_,
           singular_all_, tau_, work_, rwork_
         }) {
      if (pointer == nullptr) continue;
#if CUDART_VERSION >= 11020
      cudaFreeAsync(pointer, stream_);
#else
      cudaFree(pointer);
#endif
    }
    if (info_ != nullptr) {
#if CUDART_VERSION >= 11020
      cudaFreeAsync(info_, stream_);
#else
      cudaFree(info_);
#endif
    }
    if (invalid_ != nullptr) {
#if CUDART_VERSION >= 11020
      cudaFreeAsync(invalid_, stream_);
#else
      cudaFree(invalid_);
#endif
    }
  }

  int rows_;
  int columns_;
  int rank_;
  int sketch_;
  int power_;
  int work_size_ = 0;
  cudaStream_t stream_;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  curandGenerator_t random_ = nullptr;
  float* omega_ = nullptr;
  float* basis_ = nullptr;
  float* right_basis_ = nullptr;
  float* small_ = nullptr;
  float* small_u_ = nullptr;
  float* small_vt_ = nullptr;
  float* singular_all_ = nullptr;
  float* tau_ = nullptr;
  float* work_ = nullptr;
  float* rwork_ = nullptr;
  int* info_ = nullptr;
  int* invalid_ = nullptr;
};

}  // namespace fastembedr_pca

#endif
