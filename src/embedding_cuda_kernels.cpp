/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Package-owned CUDA embedding, graph, and RAFT-TSVD adapter kernels. The
 * build consumes installed CUDA/cuFFT, CCCL (CUB/Thrust), and optional RAFT
 * headers/libraries; none of those dependency sources or binaries is vendored
 * in fastEmbedR. See inst/THIRD_PARTY_DEPENDENCIES.json.
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>

#ifndef CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#define CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
#endif
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include "pca_cuda_rsvd.cuh"

#ifdef FASTEMBEDR_HAS_RAFT
#include <R_ext/Print.h>
#define printf REprintf
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/tsvd.cuh>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#undef printf
#endif

namespace {

#define FASTEMBEDR_CUDA_FLOAT_INF CUDART_INF_F
constexpr double kCudaDoubleInf = 1.7976931348623157e+308;
constexpr double kCudaDoubleMin = 2.2250738585072014e-308;
constexpr int kCudaProjectionMaxK = 128;
constexpr int kCudaScoreMaxK = 64;
constexpr int kCudaSilhouetteMaxLabels = 128;
constexpr int kCudaScoreWidth = 6;

struct EmbedParams {
  int n;
  int k;
  int n_epochs;
  int negative_sample_rate;
  int objective;
  unsigned int seed;
  float learning_rate;
  float a;
  float b;
  float max_weight;
  float repulsion_strength;
};

struct KnnPrepParams {
  int n;
  int k;
  int width;
  int offset;
};

thread_local std::string embedding_last_error;
thread_local bool last_tsne_cuda_graph_capture = false;
thread_local bool last_umap_cuda_graph_capture = false;

void set_embedding_error(const std::string& message) {
  embedding_last_error = message;
}

int fail_cuda(cudaError_t code, const char* where) {
  set_embedding_error(std::string(where) + ": " + cudaGetErrorString(code));
  return 1;
}

int check_cuda(cudaError_t code, const char* where) {
  return code == cudaSuccess ? 0 : fail_cuda(code, where);
}

int check_cuda(cudaError_t code, const std::string& where) {
  return check_cuda(code, where.c_str());
}

std::size_t align_bytes(std::size_t bytes, std::size_t alignment = 256u) {
  return (bytes + alignment - 1u) & ~(alignment - 1u);
}

class CudaWorkspace {
 public:
  CudaWorkspace() = default;
  CudaWorkspace(const CudaWorkspace&) = delete;
  CudaWorkspace& operator=(const CudaWorkspace&) = delete;
  ~CudaWorkspace() {
    release();
  }

  int init(std::size_t bytes, const char* label) {
    release();
    capacity_ = align_bytes(bytes);
    offset_ = 0u;
    if (capacity_ == 0u) return 0;
#if CUDART_VERSION >= 11020
    if (configure_pool()) return 1;
    return check_cuda(
      cudaMallocAsync(&base_, capacity_, nullptr),
      std::string("cudaMallocAsync(workspace ") + label + ")"
    );
#else
    return check_cuda(
      cudaMalloc(&base_, capacity_),
      std::string("cudaMalloc(workspace ") + label + ")"
    );
#endif
  }

  template <typename T>
  T* alloc(std::size_t count, const char* label) {
    if (count == 0u) return nullptr;
    const std::size_t bytes = align_bytes(count * sizeof(T));
    if (base_ == nullptr || offset_ + bytes > capacity_) {
      set_embedding_error(
        std::string("CUDA workspace exhausted while allocating ") + label +
        "; requested " + std::to_string(static_cast<unsigned long long>(bytes)) +
        " bytes, remaining " +
        std::to_string(static_cast<unsigned long long>(capacity_ > offset_ ? capacity_ - offset_ : 0u)) +
        " bytes"
      );
      return nullptr;
    }
    unsigned char* ptr = static_cast<unsigned char*>(base_) + offset_;
    offset_ += bytes;
    return reinterpret_cast<T*>(ptr);
  }

 private:
  void release() noexcept {
    if (base_ == nullptr) return;
#if CUDART_VERSION >= 11020
    cudaFreeAsync(base_, nullptr);
#else
    cudaFree(base_);
#endif
    base_ = nullptr;
    capacity_ = 0u;
    offset_ = 0u;
  }

#if CUDART_VERSION >= 11020
  int configure_pool() {
    int device = 0;
    if (check_cuda(cudaGetDevice(&device), "cudaGetDevice(workspace)")) {
      return 1;
    }
    static thread_local int configured_device = -1;
    if (configured_device == device) return 0;
    cudaMemPool_t pool = nullptr;
    if (check_cuda(
          cudaDeviceGetDefaultMemPool(&pool, device),
          "cudaDeviceGetDefaultMemPool(workspace)"
        )) {
      return 1;
    }
    std::uint64_t threshold =
      std::numeric_limits<std::uint64_t>::max();
    if (check_cuda(
          cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &threshold
          ),
          "cudaMemPoolSetAttribute(workspace release threshold)"
        )) {
      return 1;
    }
    configured_device = device;
    return 0;
  }
#endif

  void* base_ = nullptr;
  std::size_t capacity_ = 0u;
  std::size_t offset_ = 0u;
};

class CudaStreamOwner {
 public:
  CudaStreamOwner() = default;
  CudaStreamOwner(const CudaStreamOwner&) = delete;
  CudaStreamOwner& operator=(const CudaStreamOwner&) = delete;
  ~CudaStreamOwner() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  int init() {
    if (stream_ != nullptr) return 0;
    return check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags(embedding)");
  }

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

CudaStreamOwner& cuda_execution_stream() {
  static thread_local CudaStreamOwner stream;
  return stream;
}

class CudaGraphExecOwner {
 public:
  CudaGraphExecOwner() = default;
  CudaGraphExecOwner(const CudaGraphExecOwner&) = delete;
  CudaGraphExecOwner& operator=(const CudaGraphExecOwner&) = delete;
  ~CudaGraphExecOwner() { reset(); }

  void reset() {
    if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    exec_ = nullptr;
    graph_ = nullptr;
  }

  int instantiate(cudaGraph_t graph) {
    reset();
    graph_ = graph;
    const cudaError_t code = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (code != cudaSuccess) {
      reset();
      return 1;
    }
    return 0;
  }

  cudaGraphExec_t get() const { return exec_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

bool cuda_graph_capture_enabled() {
  const char* env = std::getenv("FASTEMBEDR_CUDA_GRAPH_CAPTURE");
  if (env == nullptr || env[0] == '\0') return true;
  return !(std::strcmp(env, "0") == 0 ||
           std::strcmp(env, "false") == 0 ||
           std::strcmp(env, "FALSE") == 0 ||
           std::strcmp(env, "no") == 0 ||
           std::strcmp(env, "NO") == 0);
}

std::size_t cub_select_flagged_temp_bytes(int length) {
  if (length <= 0) return 0u;
  std::size_t bytes = 0u;
  thrust::counting_iterator<int> counting(0);
  cudaError_t code = cub::DeviceSelect::Flagged(
    nullptr, bytes, counting, static_cast<unsigned char*>(nullptr),
    static_cast<int*>(nullptr), static_cast<int*>(nullptr), length
  );
  if (code != cudaSuccess) {
    cudaGetLastError();
    return 0u;
  }
  return align_bytes(bytes);
}

int cub_select_valid_ids_rank_major(unsigned char* flags,
                                    int* selected_ids,
                                    int* selected_count,
                                    int length,
                                    CudaWorkspace& workspace,
                                    std::size_t temp_bytes,
                                    const char* label) {
  if (length <= 0) return 0;
  void* temp = nullptr;
  if (temp_bytes > 0u) {
    temp = workspace.alloc<unsigned char>(temp_bytes, label);
    if (temp == nullptr) return 1;
  }
  thrust::counting_iterator<int> counting(0);
  return check_cuda(
    cub::DeviceSelect::Flagged(
      temp, temp_bytes, counting, flags, selected_ids, selected_count, length
    ),
    std::string("cub::DeviceSelect::Flagged(") + label + ")"
  );
}

int check_cufft(cufftResult code, const char* where) {
  if (code == CUFFT_SUCCESS) return 0;
  set_embedding_error(std::string(where) + ": cuFFT error " + std::to_string(static_cast<int>(code)));
  return 1;
}

class CudaFftPlanCache {
 public:
  CudaFftPlanCache() = default;
  CudaFftPlanCache(const CudaFftPlanCache&) = delete;
  CudaFftPlanCache& operator=(const CudaFftPlanCache&) = delete;
  ~CudaFftPlanCache() { reset(); }

  int acquire(int fft_size, cudaStream_t stream) {
    int device = 0;
    if (check_cuda(cudaGetDevice(&device), "cudaGetDevice(cuFFT cache)")) {
      return 1;
    }
    if (device_ == device && fft_size_ == fft_size &&
        forward_ != 0 && inverse_ != 0) {
      return bind_stream(stream);
    }
    reset();
    device_ = device;
    fft_size_ = fft_size;
    int dims[2] = {fft_size, fft_size};
    int embed[2] = {fft_size, fft_size};
    if (check_cufft(cufftCreate(&forward_), "cufftCreate(forward cache)") ||
        check_cufft(cufftCreate(&inverse_), "cufftCreate(inverse cache)") ||
        check_cufft(
          cufftSetAutoAllocation(forward_, 0),
          "cufftSetAutoAllocation(forward cache)"
        ) ||
        check_cufft(
          cufftSetAutoAllocation(inverse_, 0),
          "cufftSetAutoAllocation(inverse cache)"
        ) ||
        check_cufft(
          cufftMakePlanMany(
            forward_, 2, dims, embed, 1, fft_size * fft_size,
            embed, 1, fft_size * fft_size, CUFFT_C2C, 5,
            &forward_work_bytes_
          ),
          "cufftMakePlanMany(forward cache)"
        ) ||
        check_cufft(
          cufftMakePlanMany(
            inverse_, 2, dims, embed, 1, fft_size * fft_size,
            embed, 1, fft_size * fft_size, CUFFT_C2C, 4,
            &inverse_work_bytes_
          ),
          "cufftMakePlanMany(inverse cache)"
        ) ||
        bind_stream(stream)) {
      reset();
      return 1;
    }
    return 0;
  }

  cufftHandle forward() const { return forward_; }
  cufftHandle inverse() const { return inverse_; }
  std::size_t work_bytes() const {
    return align_bytes(
      std::max(forward_work_bytes_, inverse_work_bytes_)
    );
  }

 private:
  int bind_stream(cudaStream_t stream) {
    return check_cufft(
      cufftSetStream(forward_, stream),
      "cufftSetStream(forward cache)"
    ) || check_cufft(
      cufftSetStream(inverse_, stream),
      "cufftSetStream(inverse cache)"
    );
  }

  void reset() noexcept {
    if (forward_ != 0) cufftDestroy(forward_);
    if (inverse_ != 0) cufftDestroy(inverse_);
    forward_ = 0;
    inverse_ = 0;
    fft_size_ = 0;
    device_ = -1;
    forward_work_bytes_ = 0u;
    inverse_work_bytes_ = 0u;
  }

  int device_ = -1;
  int fft_size_ = 0;
  cufftHandle forward_ = 0;
  cufftHandle inverse_ = 0;
  std::size_t forward_work_bytes_ = 0u;
  std::size_t inverse_work_bytes_ = 0u;
};

CudaFftPlanCache& cuda_fft_plan_cache() {
  static thread_local CudaFftPlanCache cache;
  return cache;
}

int check_embedding_memory_available(std::size_t required_bytes, const char* where) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t code = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (code != cudaSuccess) return 0;
  const std::size_t reserve = free_bytes / 20u;
  if (required_bytes > free_bytes - reserve) {
    set_embedding_error(
      std::string(where) + ": CUDA memory request exceeds available memory; required " +
      std::to_string(static_cast<unsigned long long>(required_bytes)) +
      " bytes, free " +
      std::to_string(static_cast<unsigned long long>(free_bytes)) +
      " bytes"
    );
    return 1;
  }
  return 0;
}

static __device__ __forceinline__ unsigned int mix_uint(unsigned int x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static __device__ __forceinline__ unsigned int deterministic_vertex(unsigned int n,
                                             unsigned int seed,
                                             unsigned int epoch,
                                             unsigned int i,
                                             unsigned int edge,
                                             unsigned int sample) {
  unsigned int x = seed;
  x ^= epoch * 0x9e3779b9u;
  x ^= (i + 1u) * 0x85ebca6bu;
  x ^= (edge + 1u) * 0xc2b2ae35u;
  x ^= (sample + 1u) * 0x27d4eb2du;
  return mix_uint(x) % n;
}

static __device__ __forceinline__ float deterministic_unit(unsigned int seed,
                                    unsigned int row,
                                    unsigned int col) {
  unsigned int x = seed;
  x ^= (row + 1u) * 0x9e3779b9u;
  x ^= (col + 1u) * 0x85ebca6bu;
  const unsigned int bits = mix_uint(x);
  return static_cast<float>(bits & 0x00ffffffu) / 8388607.5f - 1.0f;
}

__device__ __forceinline__ int tsne_sign_component(float value) {
  return (value > 0.0f) - (value < 0.0f);
}

static __device__ __forceinline__ float clip4(float x) {
  return fminf(4.0f, fmaxf(-4.0f, x));
}

static __device__ __forceinline__ bool pair_less_device(double a_dist,
                                 int a_idx,
                                 double b_dist,
                                 int b_idx) {
  return (a_dist < b_dist) || (a_dist == b_dist && a_idx < b_idx);
}

static __device__ __forceinline__ double layout_d2_2d(
    const double* layout, int n, int i, int j) {
  const double dx = layout[i] - layout[j];
  const double dy = layout[static_cast<std::size_t>(n) + i] -
    layout[static_cast<std::size_t>(n) + j];
  return dx * dx + dy * dy;
}

static __device__ __forceinline__ double median_device(
    double* values, int count) {
  for (int i = 1; i < count; ++i) {
    const double value = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > value) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = value;
  }
  const int mid = count / 2;
  return (count & 1) ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

static __device__ __forceinline__ void insert_top_neighbor(
                                    double* top_dist,
                                    int* top_idx,
                                    int* top_count,
                                    int k,
                                    double dist,
                                    int idx) {
  if (*top_count < k) {
    int pos = *top_count;
    top_dist[pos] = dist;
    top_idx[pos] = idx;
    ++(*top_count);
    while (pos > 0 && pair_less_device(top_dist[pos], top_idx[pos], top_dist[pos - 1], top_idx[pos - 1])) {
      const double tmp_dist = top_dist[pos - 1];
      const int tmp_idx = top_idx[pos - 1];
      top_dist[pos - 1] = top_dist[pos];
      top_idx[pos - 1] = top_idx[pos];
      top_dist[pos] = tmp_dist;
      top_idx[pos] = tmp_idx;
      --pos;
    }
    return;
  }

  if (!pair_less_device(dist, idx, top_dist[k - 1], top_idx[k - 1])) return;
  int pos = k - 1;
  top_dist[pos] = dist;
  top_idx[pos] = idx;
  while (pos > 0 && pair_less_device(top_dist[pos], top_idx[pos], top_dist[pos - 1], top_idx[pos - 1])) {
    const double tmp_dist = top_dist[pos - 1];
    const int tmp_idx = top_idx[pos - 1];
    top_dist[pos - 1] = top_dist[pos];
    top_idx[pos - 1] = top_idx[pos];
    top_dist[pos] = tmp_dist;
    top_idx[pos] = tmp_idx;
    --pos;
  }
}

// Package-local positive-base power approximation for UMAP force kernels.
// It uses IEEE-754 exponent interpolation, following Schraudolph's published
// fast exponential idea, and is mirrored from the CPU implementation without
// vendoring third-party source.
static __device__ __forceinline__ float fast_positive_pow(float x, float b) {
  if (x <= 0.0f) return 0.0f;
  const unsigned int x_bits = __float_as_uint(x);
  constexpr float exponent_bias_word = 1064866805.0f;
  const int whole = static_cast<int>(b);
  const float fractional = b - static_cast<float>(whole);
  const unsigned int interp_bits = static_cast<unsigned int>(
      fractional * (static_cast<float>(x_bits) - exponent_bias_word) +
      exponent_bias_word);
  const float fractional_pow = __uint_as_float(interp_bits);
  float integer_pow = 1.0f;
  float base = x;
  int exponent = whole;
  while (exponent > 0) {
    if ((exponent & 1) != 0) integer_pow *= base;
    base *= base;
    exponent >>= 1;
  }
  return integer_pow * fractional_pow;
}

static __device__ __forceinline__ float attractive_coeff(
    float d2, float weight, const EmbedParams p) {
  if (p.objective == 0) {
    if (d2 <= 0.0f) return 0.0f;
    const float d2b = fast_positive_pow(d2, p.b);
    return -2.0f * p.a * p.b * (d2b / d2) / (p.a * d2b + 1.0f);
  }
  if (p.objective == 1) return -2.0f * weight / (1.0f + d2);
  if (p.objective == 2) return -2.0f * weight / (10.0f + d2);
  if (p.objective == 4) return -2.5f * weight / (0.15f + d2);
  return -2.0f * weight / (1.0f + d2);
}

static __device__ __forceinline__ float repulsive_coeff(
    float d2, const EmbedParams p) {
  if (d2 <= 0.0f) return 0.0f;
  if (p.objective == 0) {
    const float d2b = fast_positive_pow(d2, p.b);
    return p.repulsion_strength * 2.0f * p.b / ((0.001f + d2) * (p.a * d2b + 1.0f));
  }
  if (p.objective == 1) return p.repulsion_strength * 2.0f / ((1.0f + d2) * (1.0f + d2));
  if (p.objective == 2) return p.repulsion_strength * 0.4f / (1.0f + d2);
  if (p.objective == 4) return p.repulsion_strength * 0.8125f / ((0.15f + d2) * (1.0f + d2));
  return p.repulsion_strength * 2.0f / (1.0f + d2);
}

static __device__ __forceinline__ int positive_samples_this_epoch(
    float weight, const EmbedParams p, unsigned int epoch) {
  if (p.objective != 0) return 1;
  if (weight <= 0.0f) return 0;
  const float period = p.max_weight / fmaxf(weight, 1.0e-6f);
  if (period <= 0.0f || !isfinite(period)) return 0;
  const float now = static_cast<float>(epoch + 1u);
  const float previous = static_cast<float>(epoch);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  const int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

static __device__ __forceinline__ int positive_samples_this_epoch_period(
    float period, unsigned int epoch) {
  if (period <= 0.0f || !isfinite(period)) return 0;
  const float now = static_cast<float>(epoch + 1u);
  const float previous = static_cast<float>(epoch);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  const int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

static __device__ __forceinline__ int cumulative_umap_negative_samples(
    float active_epoch, float negative_period) {
  if (negative_period <= 0.0f || !isfinite(negative_period)) return 0;
  const int samples = static_cast<int>(
    floorf(((active_epoch - negative_period) / negative_period) + 1.0e-6f)
  );
  return samples > 0 ? samples : 0;
}

static __device__ __forceinline__ int negative_samples_this_epoch(
    float weight, const EmbedParams p, unsigned int epoch) {
  if (p.objective != 0) return p.negative_sample_rate;
  if (weight <= 0.0f || p.negative_sample_rate <= 0) return 0;
  const float period = p.max_weight / fmaxf(weight, 1.0e-6f);
  if (period <= 0.0f || !isfinite(period)) return 0;
  const float now = static_cast<float>(epoch + 1u);
  const float previous = static_cast<float>(epoch);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  if (current_sample <= previous_sample) return 0;

  const float negative_period = period / static_cast<float>(p.negative_sample_rate);
  const int current_total = cumulative_umap_negative_samples(now, negative_period);
  int previous_total = 0;
  if (previous_sample > 0) {
    const float previous_active_epoch = ceilf(static_cast<float>(previous_sample) * period);
    previous_total = cumulative_umap_negative_samples(previous_active_epoch, negative_period);
  }
  const int samples = current_total - previous_total;
  return samples > 0 ? samples : 0;
}

static __device__ __forceinline__ int negative_samples_this_epoch_period(
    float period, const EmbedParams p, unsigned int epoch) {
  if (p.negative_sample_rate <= 0 || period <= 0.0f || !isfinite(period)) return 0;
  const float now = static_cast<float>(epoch + 1u);
  const float previous = static_cast<float>(epoch);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  if (current_sample <= previous_sample) return 0;

  const float negative_period = period / static_cast<float>(p.negative_sample_rate);
  const int current_total = cumulative_umap_negative_samples(now, negative_period);
  int previous_total = 0;
  if (previous_sample > 0) {
    const float previous_active_epoch = ceilf(static_cast<float>(previous_sample) * period);
    previous_total = cumulative_umap_negative_samples(previous_active_epoch, negative_period);
  }
  const int samples = current_total - previous_total;
  return samples > 0 ? samples : 0;
}

constexpr int kCudaAffineMaxNeighbors = 12;

struct LandmarkTsneParams {
  int n_reference;
  int n_query;
  int k;
  int n_negatives;
  unsigned int seed;
  int exact_repulsion;
  float learning_rate;
  float exaggeration;
  float momentum;
  float max_grad_norm;
  float max_step_norm;
};

static __device__ __forceinline__ float landmark_median_float(
    float* values, int count) {
  for (int i = 1; i < count; ++i) {
    const float value = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > value) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = value;
  }
  const int mid = count / 2;
  return (count & 1) ? values[mid] : 0.5f * (values[mid - 1] + values[mid]);
}

static __device__ __forceinline__ unsigned int landmark_reference_sample(
                                                  unsigned int n,
                                                  unsigned int seed,
                                                  unsigned int epoch,
                                                  unsigned int row,
                                                  unsigned int sample) {
  unsigned int x = seed;
  x ^= (epoch + 1u) * 0x9e3779b9u;
  x ^= (row + 1u) * 0x85ebca6bu;
  x ^= (sample + 1u) * 0xc2b2ae35u;
  return mix_uint(x) % n;
}

static __device__ __forceinline__ void landmark_affine_fallback(
                                         const float* reference_layout,
                                         const int* projection_indices,
                                         const float* projection_distances,
                                         float2* out,
                                         int n_reference,
                                         int n_query,
                                         int k,
                                         int index_offset,
                                         int row,
                                         float rho,
                                         float sigma) {
  constexpr float eps = 1.4901161193847656e-8f;
  float weight_sum = 0.0f;
  float2 value = make_float2(0.0f, 0.0f);
  for (int j = 0; j < k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(j) * n_query + row;
    const int ref = projection_indices[pos] - index_offset;
    const float d = projection_distances[pos];
    if (ref < 0 || ref >= n_reference || !isfinite(d) || d < 0.0f) continue;
    const float w = expf(-fmaxf(0.0f, d - rho) / fmaxf(sigma, eps));
    if (!isfinite(w) || w <= 0.0f) continue;
    weight_sum += w;
    value.x += w * reference_layout[ref];
    value.y += w * reference_layout[static_cast<std::size_t>(n_reference) + ref];
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0f) weight_sum = 1.0f;
  out[row] = make_float2(value.x / weight_sum, value.y / weight_sum);
}

__global__ void landmark_affine_projection_kernel(const float* reference_data,
                                                   const float* query_data,
                                                   const float* reference_layout,
                                                   const int* projection_indices,
                                                   const float* projection_distances,
                                                   float2* out,
                                                   int n_reference,
                                                   int n_query,
                                                   int n_features,
                                                   int k,
                                                   int index_offset,
                                                   int max_neighbors,
                                                   float ridge,
                                                   float max_extrapolation) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;
  constexpr float eps = 1.4901161193847656e-8f;
  const int m = min(max(max_neighbors, 3), min(k, kCudaAffineMaxNeighbors));

  float all_distances[kCudaProjectionMaxK];
  float scratch[kCudaProjectionMaxK];
  float weights[kCudaAffineMaxNeighbors];
  int refs[kCudaAffineMaxNeighbors];
  float centered[kCudaAffineMaxNeighbors];
  float row_norm[kCudaAffineMaxNeighbors];
  float gram[kCudaAffineMaxNeighbors * kCudaAffineMaxNeighbors];
  float rhs[kCudaAffineMaxNeighbors * 2];
  float q[kCudaAffineMaxNeighbors];

  int zero_col = -1;
  float rho = FASTEMBEDR_CUDA_FLOAT_INF;
  for (int j = 0; j < k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(j) * n_query + row;
    const int ref = projection_indices[pos] - index_offset;
    const float d = projection_distances[pos];
    const bool valid = ref >= 0 && ref < n_reference && isfinite(d) && d >= 0.0f;
    all_distances[j] = valid ? d : FASTEMBEDR_CUDA_FLOAT_INF;
    if (valid && d <= eps && zero_col < 0) zero_col = j;
    if (valid) rho = fminf(rho, d);
  }
  if (!isfinite(rho)) rho = 0.0f;
  if (zero_col >= 0) {
    const int ref = projection_indices[static_cast<std::size_t>(zero_col) * n_query + row] - index_offset;
    out[row] = make_float2(
      reference_layout[ref],
      reference_layout[static_cast<std::size_t>(n_reference) + ref]
    );
    return;
  }

  int positive_count = 0;
  for (int j = 0; j < k; ++j) {
    const float adjusted = fmaxf(0.0f, all_distances[j] - rho);
    if (isfinite(adjusted) && adjusted > eps) scratch[positive_count++] = adjusted;
  }
  float sigma = positive_count > 0 ? landmark_median_float(scratch, positive_count) : fmaxf(rho, eps);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  float weight_sum = 0.0f;
  for (int j = 0; j < m; ++j) {
    const int ref = projection_indices[static_cast<std::size_t>(j) * n_query + row] - index_offset;
    refs[j] = ref;
    const float adjusted = fmaxf(0.0f, all_distances[j] - rho);
    const float w = ref >= 0 && ref < n_reference && isfinite(adjusted) ? expf(-adjusted / sigma) : 0.0f;
    weights[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0f) {
    landmark_affine_fallback(
      reference_layout, projection_indices, projection_distances, out,
      n_reference, n_query, k, index_offset, row, rho, sigma
    );
    return;
  }
  for (int j = 0; j < m; ++j) weights[j] /= weight_sum;

  float2 y_center = make_float2(0.0f, 0.0f);
  for (int j = 0; j < m; ++j) {
    const int ref = refs[j];
    if (ref < 0 || ref >= n_reference) continue;
    y_center.x += weights[j] * reference_layout[ref];
    y_center.y += weights[j] * reference_layout[static_cast<std::size_t>(n_reference) + ref];
  }

  float layout_radius_sq = 0.0f;
  for (int a = 0; a < m; ++a) {
    const int ref = refs[a];
    const float sqrt_w = sqrtf(fmaxf(weights[a], 0.0f));
    const float yc0 = ref >= 0 && ref < n_reference ? reference_layout[ref] - y_center.x : 0.0f;
    const float yc1 = ref >= 0 && ref < n_reference ?
      reference_layout[static_cast<std::size_t>(n_reference) + ref] - y_center.y : 0.0f;
    rhs[a * 2] = sqrt_w * yc0;
    rhs[a * 2 + 1] = sqrt_w * yc1;
    layout_radius_sq = fmaxf(layout_radius_sq, yc0 * yc0 + yc1 * yc1);
    row_norm[a] = 0.0f;
    q[a] = 0.0f;
  }
  for (int i = 0; i < m * m; ++i) gram[i] = 0.0f;

  for (int feature = 0; feature < n_features; ++feature) {
    float x_center = 0.0f;
    const std::size_t ref_col = static_cast<std::size_t>(feature) * n_reference;
    for (int j = 0; j < m; ++j) {
      const int ref = refs[j];
      if (ref >= 0 && ref < n_reference) x_center += weights[j] * reference_data[ref_col + ref];
    }
    const float query_centered = query_data[static_cast<std::size_t>(feature) * n_query + row] - x_center;
    for (int a = 0; a < m; ++a) {
      const int ref = refs[a];
      const float value = ref >= 0 && ref < n_reference ? reference_data[ref_col + ref] - x_center : 0.0f;
      centered[a] = value;
      row_norm[a] += value * value;
      q[a] += query_centered * value;
    }
    for (int a = 0; a < m; ++a) {
      for (int b = 0; b <= a; ++b) gram[a * m + b] += centered[a] * centered[b];
    }
  }

  float trace = 0.0f;
  for (int a = 0; a < m; ++a) {
    const float sqrt_wa = sqrtf(fmaxf(weights[a], 0.0f));
    trace += weights[a] * row_norm[a];
    q[a] *= sqrt_wa;
    for (int b = 0; b <= a; ++b) {
      const float value = sqrt_wa * sqrtf(fmaxf(weights[b], 0.0f)) * gram[a * m + b];
      gram[a * m + b] = value;
      gram[b * m + a] = value;
    }
  }
  const float lambda = ridge * fmaxf(trace, eps) + eps;
  for (int a = 0; a < m; ++a) gram[a * m + a] += lambda;

  bool ok = true;
  for (int j = 0; j < m && ok; ++j) {
    float value = gram[j * m + j];
    for (int kk = 0; kk < j; ++kk) value -= gram[j * m + kk] * gram[j * m + kk];
    if (!isfinite(value) || value <= eps) { ok = false; break; }
    const float diag = sqrtf(value);
    gram[j * m + j] = diag;
    for (int i = j + 1; i < m; ++i) {
      float s = gram[i * m + j];
      for (int kk = 0; kk < j; ++kk) s -= gram[i * m + kk] * gram[j * m + kk];
      gram[i * m + j] = s / diag;
    }
  }
  if (!ok) {
    landmark_affine_fallback(
      reference_layout, projection_indices, projection_distances, out,
      n_reference, n_query, k, index_offset, row, rho, sigma
    );
    return;
  }
  for (int component = 0; component < 2; ++component) {
    for (int i = 0; i < m; ++i) {
      float s = rhs[i * 2 + component];
      for (int kk = 0; kk < i; ++kk) s -= gram[i * m + kk] * rhs[kk * 2 + component];
      rhs[i * 2 + component] = s / gram[i * m + i];
    }
    for (int i = m - 1; i >= 0; --i) {
      float s = rhs[i * 2 + component];
      for (int kk = i + 1; kk < m; ++kk) s -= gram[kk * m + i] * rhs[kk * 2 + component];
      rhs[i * 2 + component] = s / gram[i * m + i];
    }
  }

  float2 displacement = make_float2(0.0f, 0.0f);
  for (int a = 0; a < m; ++a) {
    displacement.x += q[a] * rhs[a * 2];
    displacement.y += q[a] * rhs[a * 2 + 1];
  }
  const float max_disp = max_extrapolation * sqrtf(fmaxf(layout_radius_sq, eps));
  const float disp_sq = displacement.x * displacement.x + displacement.y * displacement.y;
  if (disp_sq > max_disp * max_disp) {
    const float scale = max_disp / (sqrtf(disp_sq) + eps);
    displacement.x *= scale;
    displacement.y *= scale;
  }
  float2 value = make_float2(y_center.x + displacement.x, y_center.y + displacement.y);
  if (!isfinite(value.x) || !isfinite(value.y)) {
    landmark_affine_fallback(
      reference_layout, projection_indices, projection_distances, out,
      n_reference, n_query, k, index_offset, row, rho, sigma
    );
    return;
  }
  out[row] = value;
}

__global__ void landmark_tsne_probabilities_kernel(const float* distances,
                                                    float* probabilities,
                                                    int n_query,
                                                    int k,
                                                    float perplexity) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;
  float beta = 1.0f;
  float min_beta = -FASTEMBEDR_CUDA_FLOAT_INF;
  float max_beta = FASTEMBEDR_CUDA_FLOAT_INF;
  float sum_p = 1.1754943508222875e-38f;
  const float target = logf(perplexity);
  for (int iter = 0; iter < 200; ++iter) {
    sum_p = 1.1754943508222875e-38f;
    float entropy_numerator = 0.0f;
    for (int j = 0; j < k; ++j) {
      const std::size_t pos = static_cast<std::size_t>(j) * n_query + row;
      const float d = distances[pos];
      const float d2 = d * d;
      const float p = expf(-beta * d2);
      probabilities[pos] = p;
      sum_p += p;
      entropy_numerator += beta * d2 * p;
    }
    const float diff = entropy_numerator / sum_p + logf(sum_p) - target;
    if (fabsf(diff) < 1.0e-5f) break;
    if (diff > 0.0f) {
      min_beta = beta;
      beta = isinf(max_beta) ? beta * 2.0f : 0.5f * (beta + max_beta);
    } else {
      max_beta = beta;
      beta = isinf(min_beta) ? beta * 0.5f : 0.5f * (beta + min_beta);
    }
  }
  const float inv_sum = 1.0f / sum_p;
  for (int j = 0; j < k; ++j) {
    probabilities[static_cast<std::size_t>(j) * n_query + row] *= inv_sum;
  }
}

__global__ void landmark_tsne_epoch_kernel(const float* reference_layout,
                                           const int* indices,
                                           const float* probabilities,
                                           float2* current,
                                           float2* gains,
                                           float2* updates,
                                           LandmarkTsneParams p,
                                           unsigned int epoch,
                                           int index_offset) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= p.n_query) return;
  constexpr float eps = 1.0e-12f;
  const float2 yi = current[row];
  float2 grad = make_float2(0.0f, 0.0f);
  float sum_q = eps;
  const int samples = p.exact_repulsion ? p.n_reference : p.n_negatives;
  for (int sample = 0; sample < samples; ++sample) {
    const unsigned int ref = p.exact_repulsion ? static_cast<unsigned int>(sample) :
      landmark_reference_sample(p.n_reference, p.seed, epoch, row, sample);
    const float dx = yi.x - reference_layout[ref];
    const float dy = yi.y - reference_layout[static_cast<std::size_t>(p.n_reference) + ref];
    sum_q += 1.0f / (1.0f + dx * dx + dy * dy);
  }
  for (int sample = 0; sample < samples; ++sample) {
    const unsigned int ref = p.exact_repulsion ? static_cast<unsigned int>(sample) :
      landmark_reference_sample(p.n_reference, p.seed, epoch, row, sample);
    const float dx = yi.x - reference_layout[ref];
    const float dy = yi.y - reference_layout[static_cast<std::size_t>(p.n_reference) + ref];
    const float q = 1.0f / (1.0f + dx * dx + dy * dy);
    const float coeff = -(q * q) / sum_q;
    grad.x += coeff * dx;
    grad.y += coeff * dy;
  }
  for (int j = 0; j < p.k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(j) * p.n_query + row;
    const int ref = indices[pos] - index_offset;
    if (ref < 0 || ref >= p.n_reference) continue;
    const float dx = yi.x - reference_layout[ref];
    const float dy = yi.y - reference_layout[static_cast<std::size_t>(p.n_reference) + ref];
    const float q = 1.0f / (1.0f + dx * dx + dy * dy);
    const float coeff = p.exaggeration * probabilities[pos] * q;
    grad.x += coeff * dx;
    grad.y += coeff * dy;
  }
  const float grad_norm2 = grad.x * grad.x + grad.y * grad.y;
  if (isfinite(p.max_grad_norm) && p.max_grad_norm > 0.0f &&
      grad_norm2 > p.max_grad_norm * p.max_grad_norm) {
    const float scale = p.max_grad_norm / (sqrtf(grad_norm2) + eps);
    grad.x *= scale;
    grad.y *= scale;
  }
  float2 gain = gains[row];
  float2 update = updates[row];
  gain.x = tsne_sign_component(update.x) != tsne_sign_component(grad.x) ?
    gain.x + 0.2f : gain.x * 0.8f + 0.01f;
  gain.y = tsne_sign_component(update.y) != tsne_sign_component(grad.y) ?
    gain.y + 0.2f : gain.y * 0.8f + 0.01f;
  gain.x = fmaxf(gain.x, 0.01f);
  gain.y = fmaxf(gain.y, 0.01f);
  update.x = p.momentum * update.x - p.learning_rate * gain.x * grad.x;
  update.y = p.momentum * update.y - p.learning_rate * gain.y * grad.y;
  const float step_norm2 = update.x * update.x + update.y * update.y;
  if (isfinite(p.max_step_norm) && p.max_step_norm > 0.0f &&
      step_norm2 > p.max_step_norm * p.max_step_norm) {
    const float scale = p.max_step_norm / (sqrtf(step_norm2) + eps);
    update.x *= scale;
    update.y *= scale;
  }
  current[row] = make_float2(yi.x + update.x, yi.y + update.y);
  gains[row] = gain;
  updates[row] = update;
}

__global__ void landmark_tsne_state_init_kernel(float2* gains,
                                                float2* updates,
                                                int n_query) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;
  gains[row] = make_float2(1.0f, 1.0f);
  updates[row] = make_float2(0.0f, 0.0f);
}

__global__ void landmark_distance_sum_kernel(const float* distances,
                                             int total,
                                             float* sum,
                                             unsigned int* count) {
  const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (id >= total) return;
  const float value = distances[id];
  if (isfinite(value) && value >= 0.0f) {
    atomicAdd(sum, value);
    atomicAdd(count, 1u);
  }
}

__global__ void landmark_umap_prepare_kernel(const float* distances,
                                             float* weights,
                                             float* epochs_per_sample,
                                             int n_query,
                                             int k,
                                             float global_mean) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;
  float rho = FASTEMBEDR_CUDA_FLOAT_INF;
  float row_sum = 0.0f;
  int row_count = 0;
  for (int j = 0; j < k; ++j) {
    const float d = distances[static_cast<std::size_t>(j) * n_query + row];
    if (!isfinite(d) || d < 0.0f) continue;
    row_sum += d;
    ++row_count;
    if (d > 0.0f) rho = fminf(rho, d);
  }
  if (!isfinite(rho)) rho = 0.0f;
  const float target = log2f(fmaxf(1.0f, static_cast<float>(k)));
  float lo = 0.0f;
  float hi = FASTEMBEDR_CUDA_FLOAT_INF;
  float sigma = 1.0f;
  float best_sigma = sigma;
  float best_diff = FASTEMBEDR_CUDA_FLOAT_INF;
  for (int iter = 0; iter < 64; ++iter) {
    float psum = 0.0f;
    const float safe_sigma = fmaxf(sigma, 1.0e-12f);
    for (int j = 0; j < k; ++j) {
      const float raw = distances[static_cast<std::size_t>(j) * n_query + row];
      if (!isfinite(raw) || raw < 0.0f) continue;
      const float d = raw - rho;
      psum += d <= 0.0f ? 1.0f : expf(-d / safe_sigma);
    }
    const float diff = fabsf(psum - target);
    if (diff < best_diff) { best_diff = diff; best_sigma = sigma; }
    if (psum > target) {
      hi = sigma;
      sigma = 0.5f * (lo + hi);
    } else {
      lo = sigma;
      sigma = isinf(hi) ? sigma * 2.0f : 0.5f * (lo + hi);
    }
    if (diff < 1.0e-5f) break;
  }
  const float row_mean = row_count > 0 ? row_sum / row_count : global_mean;
  const float sigma_floor = 1.0e-3f * (rho > 0.0f ? row_mean : global_mean);
  best_sigma = fmaxf(fmaxf(best_sigma, sigma_floor), 1.0e-12f);
  for (int j = 0; j < k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(j) * n_query + row;
    const float d = distances[pos];
    float w = 0.0f;
    if (isfinite(d) && d >= 0.0f) {
      w = d <= rho ? 1.0f : expf(-(d - rho) / best_sigma);
      if (!isfinite(w) || w <= 0.0f) w = 0.0f;
    }
    weights[pos] = w;
    epochs_per_sample[pos] = w > 0.0f ? 1.0f / fmaxf(w, 1.0e-6f) : 0.0f;
  }
}

__global__ void landmark_umap_assemble_kernel(const float* reference_layout,
                                              const float2* query_layout,
                                              const int* landmark_rows,
                                              const int* query_rows,
                                              int n_reference,
                                              int n_query,
                                              float* layout) {
  const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (id < n_reference) {
    const int row = landmark_rows[id] - 1;
    layout[static_cast<std::size_t>(row) * 2u] = reference_layout[id];
    layout[static_cast<std::size_t>(row) * 2u + 1u] =
      reference_layout[static_cast<std::size_t>(n_reference) + id];
  }
  if (id < n_query) {
    const int row = query_rows[id] - 1;
    layout[static_cast<std::size_t>(row) * 2u] = query_layout[id].x;
    layout[static_cast<std::size_t>(row) * 2u + 1u] = query_layout[id].y;
  }
}

__global__ void landmark_umap_refine_epoch_kernel(float* layout,
                                                  const int* indices,
                                                  const float* weights,
                                                  const float* epochs_per_sample,
                                                  const int* landmark_rows,
                                                  const int* query_rows,
                                                  int n_reference,
                                                  int n_query,
                                                  int k,
                                                  int index_offset,
                                                  EmbedParams p,
                                                  unsigned int epoch) {
  const int local_row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (local_row >= n_query) return;
  const int head = query_rows[local_row] - 1;
  if (head < 0 || head >= p.n) return;
  float2 yi = make_float2(
    layout[static_cast<std::size_t>(head) * 2u],
    layout[static_cast<std::size_t>(head) * 2u + 1u]
  );
  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) / fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  constexpr float eps = 1.1920928955078125e-7f;
  for (int j = 0; j < k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(j) * n_query + local_row;
    const int ref = indices[pos] - index_offset;
    if (ref < 0 || ref >= n_reference) continue;
    const int tail = landmark_rows[ref] - 1;
    if (tail < 0 || tail >= p.n || tail == head) continue;
    const float period = epochs_per_sample[pos];
    const int positive_samples = positive_samples_this_epoch_period(period, epoch);
    if (positive_samples <= 0) continue;
    const float tx = layout[static_cast<std::size_t>(tail) * 2u];
    const float ty = layout[static_cast<std::size_t>(tail) * 2u + 1u];
    for (int sample = 0; sample < positive_samples; ++sample) {
      const float dx = yi.x - tx;
      const float dy = yi.y - ty;
      const float d2 = fmaxf(eps, dx * dx + dy * dy);
      const float coeff = attractive_coeff(d2, weights[pos], p);
      yi.x += alpha * clip4(coeff * dx);
      yi.y += alpha * clip4(coeff * dy);
    }
    const int negative_samples = negative_samples_this_epoch_period(period, p, epoch);
    for (int sample = 0; sample < negative_samples; ++sample) {
      const unsigned int neg = deterministic_vertex(
        p.n, p.seed, epoch, head, j, sample
      );
      if (static_cast<int>(neg) == head || static_cast<int>(neg) == tail) continue;
      const float nx = layout[static_cast<std::size_t>(neg) * 2u];
      const float ny = layout[static_cast<std::size_t>(neg) * 2u + 1u];
      const float dx = yi.x - nx;
      const float dy = yi.y - ny;
      const float d2 = fmaxf(eps, dx * dx + dy * dy);
      const float coeff = repulsive_coeff(d2, p);
      yi.x += alpha * clip4(coeff * dx);
      yi.y += alpha * clip4(coeff * dy);
    }
    layout[static_cast<std::size_t>(head) * 2u] = yi.x;
    layout[static_cast<std::size_t>(head) * 2u + 1u] = yi.y;
  }
}

template <typename DistanceT>
__global__ void prepare_directed_knn_kernel(const int* indices,
                                            const DistanceT* distances,
                                            int* neighbors,
                                            float* weights,
                                            KnnPrepParams p) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= p.n) return;

  float rho = FASTEMBEDR_CUDA_FLOAT_INF;
  for (int j = 0; j < p.k; ++j) {
    const float d = static_cast<float>(distances[static_cast<std::size_t>(j) * p.n + row]);
    if (d > 0.0f && d < rho) rho = d;
  }
  if (!isfinite(rho)) rho = 0.0f;

  const float target = log2f(static_cast<float>(p.k < 2 ? 2 : p.k));
  float lo = 0.0f;
  float hi = FASTEMBEDR_CUDA_FLOAT_INF;
  float sigma = 1.0f;
  for (int iter = 0; iter < 48; ++iter) {
    float psum = 0.0f;
    for (int j = 0; j < p.k; ++j) {
      const float d = static_cast<float>(distances[static_cast<std::size_t>(j) * p.n + row]) - rho;
      psum += d <= 0.0f ? 1.0f : expf(-d / sigma);
    }
    if (fabsf(psum - target) < 1.0e-5f) break;
    if (psum > target) {
      hi = sigma;
      sigma = 0.5f * (lo + hi);
    } else {
      lo = sigma;
      sigma = isinf(hi) ? sigma * 2.0f : 0.5f * (lo + hi);
    }
  }
  sigma = fmaxf(sigma, 1.0e-6f);

  for (int j = 0; j < p.k; ++j) {
    const std::size_t out = static_cast<std::size_t>(row) * p.width + j;
    int nb = indices[static_cast<std::size_t>(j) * p.n + row] - p.offset;
    const float d = static_cast<float>(distances[static_cast<std::size_t>(j) * p.n + row]);
    if (nb < 0 || nb >= p.n || nb == row) {
      neighbors[out] = row;
      weights[out] = 0.0f;
    } else {
      neighbors[out] = nb;
      weights[out] = d <= rho ? 1.0f : expf(-(d - rho) / sigma);
    }
  }
  for (int j = p.k; j < p.width; ++j) {
    const std::size_t out = static_cast<std::size_t>(row) * p.width + j;
    neighbors[out] = row;
    weights[out] = 0.0f;
  }
}

static __device__ __forceinline__ float reverse_directed_weight(
                                         const int* neighbors,
                                         const float* weights,
                                         int width,
                                         int row,
                                         int target,
                                         int k) {
  for (int j = 0; j < k; ++j) {
    const std::size_t pos = static_cast<std::size_t>(row) * width + j;
    if (neighbors[pos] == target) return weights[pos];
  }
  return 0.0f;
}

static __device__ __forceinline__ bool direct_contains(
                                const int* neighbors,
                                int width,
                                int row,
                                int target,
                                int k) {
  for (int j = 0; j < k; ++j) {
    if (neighbors[static_cast<std::size_t>(row) * width + j] == target) return true;
  }
  return false;
}

__global__ void fill_umap_union_kernel(int* neighbors,
                                       float* weights,
                                       int* row_counts,
                                       int n,
                                       int width) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * width;
  if (gid < total) {
    const int row = gid / width;
    neighbors[gid] = row;
    weights[gid] = 0.0f;
  }
  if (gid < n) row_counts[gid] = 0;
}

__global__ void binary_graph_weights_kernel(const int* neighbors,
                                            float* weights,
                                            int n,
                                            int width) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * width;
  if (gid >= total) return;
  const int row = gid / width;
  const int nb = neighbors[gid];
  weights[gid] = (nb >= 0 && nb < n && nb != row) ? 1.0f : 0.0f;
}

__global__ void add_umap_outgoing_kernel(const int* direct_neighbors,
                                         const float* direct_weights,
                                         int* union_neighbors,
                                         float* union_weights,
                                         int* row_counts,
                                         KnnPrepParams p) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = p.n * p.k;
  if (gid >= total) return;
  const int row = gid / p.k;
  const int col = gid - row * p.k;
  const std::size_t direct_pos = static_cast<std::size_t>(row) * p.width + col;
  const int nb = direct_neighbors[direct_pos];
  const float forward = direct_weights[direct_pos];
  if (nb < 0 || nb >= p.n || nb == row || forward <= 0.0f) return;

  const float reverse = reverse_directed_weight(direct_neighbors, direct_weights, p.width, nb, row, p.k);
  const float w = forward + reverse - forward * reverse;
  const int pos = atomicAdd(row_counts + row, 1);
  if (pos < p.width) {
    const std::size_t out = static_cast<std::size_t>(row) * p.width + pos;
    union_neighbors[out] = nb;
    union_weights[out] = w;
  }
}

__global__ void add_umap_incoming_kernel(const int* direct_neighbors,
                                         const float* direct_weights,
                                         int* union_neighbors,
                                         float* union_weights,
                                         int* row_counts,
                                         KnnPrepParams p) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = p.n * p.k;
  if (gid >= total) return;
  const int row = gid / p.k;
  const int col = gid - row * p.k;
  const std::size_t direct_pos = static_cast<std::size_t>(row) * p.width + col;
  const int nb = direct_neighbors[direct_pos];
  const float w = direct_weights[direct_pos];
  if (nb < 0 || nb >= p.n || nb == row || w <= 0.0f) return;
  if (direct_contains(direct_neighbors, p.width, nb, row, p.k)) return;

  const int pos = atomicAdd(row_counts + nb, 1);
  if (pos < p.width) {
    const std::size_t out = static_cast<std::size_t>(nb) * p.width + pos;
    union_neighbors[out] = row;
    union_weights[out] = w;
  }
}

__global__ void random_init_kernel(float* values, int n, unsigned int seed) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  values[static_cast<std::size_t>(row) * 2u] = deterministic_unit(seed, static_cast<unsigned int>(row), 0u);
  values[static_cast<std::size_t>(row) * 2u + 1u] = deterministic_unit(seed, static_cast<unsigned int>(row), 1u);
}

__global__ void diffuse_init_kernel(const int* neighbors,
                                    const float* weights,
                                    const float* current,
                                    float* next,
                                    int n,
                                    int width) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  float sx = 0.0f;
  float sy = 0.0f;
  float sw = 0.0f;
  for (int j = 0; j < width; ++j) {
    const std::size_t pos = static_cast<std::size_t>(row) * width + j;
    const int nb = neighbors[pos];
    const float w = weights[pos];
    if (nb < 0 || nb >= n || nb == row || w <= 0.0f) continue;
    sx += w * current[static_cast<std::size_t>(nb) * 2u];
    sy += w * current[static_cast<std::size_t>(nb) * 2u + 1u];
    sw += w;
  }
  if (sw <= 0.0f) {
    next[static_cast<std::size_t>(row) * 2u] = current[static_cast<std::size_t>(row) * 2u];
    next[static_cast<std::size_t>(row) * 2u + 1u] = current[static_cast<std::size_t>(row) * 2u + 1u];
  } else {
    next[static_cast<std::size_t>(row) * 2u] = sx / sw;
    next[static_cast<std::size_t>(row) * 2u + 1u] = sy / sw;
  }
}

__global__ void reduce_init_stats_kernel(const float* values,
                                         double* partial,
                                         int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  double* sx2 = sy + blockDim.x;
  double* sxy = sx2 + blockDim.x;
  double* sy2 = sxy + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  double ax2 = 0.0;
  double axy = 0.0;
  double ay2 = 0.0;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       i < n;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    const double x = values[static_cast<std::size_t>(i) * 2u];
    const double y = values[static_cast<std::size_t>(i) * 2u + 1u];
    ax += x;
    ay += y;
    ax2 += x * x;
    axy += x * y;
    ay2 += y * y;
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sxy[tid] = axy;
  sy2[tid] = ay2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sxy[tid] += sxy[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 5u;
    partial[base] = sx[0];
    partial[base + 1u] = sy[0];
    partial[base + 2u] = sx2[0];
    partial[base + 3u] = sxy[0];
    partial[base + 4u] = sy2[0];
  }
}

__global__ void finalize_init_stats_kernel(const double* partial,
                                           double* stats,
                                           int n_blocks,
                                           int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  double* sx2 = sy + blockDim.x;
  double* sxy = sx2 + blockDim.x;
  double* sy2 = sxy + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  double ax2 = 0.0;
  double axy = 0.0;
  double ay2 = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 5u;
    ax += partial[base];
    ay += partial[base + 1u];
    ax2 += partial[base + 2u];
    axy += partial[base + 3u];
    ay2 += partial[base + 4u];
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sxy[tid] = axy;
  sy2[tid] = ay2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sxy[tid] += sxy[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double dn = static_cast<double>(max(n, 1));
    const double mean_x = sx[0] / dn;
    const double mean_y = sy[0] / dn;
    const double x_center_ss = fmax(sx2[0] - sx[0] * sx[0] / dn, 1.0e-24);
    const double y_center_ss = fmax(sy2[0] - sy[0] * sy[0] / dn, 1.0e-24);
    const double xy_center = sxy[0] - sx[0] * sy[0] / dn;
    const double norm_x = sqrt(x_center_ss);
    const double proj_y_on_x = xy_center / norm_x;
    const double y_resid_ss = fmax(y_center_ss - proj_y_on_x * proj_y_on_x, 1.0e-24);
    stats[0] = mean_x;
    stats[1] = mean_y;
    stats[2] = 1.0 / norm_x;
    stats[3] = proj_y_on_x;
    stats[4] = 1.0 / sqrt(y_resid_ss);
  }
}

__global__ void normalize_init_kernel(float* values,
                                      const double* stats,
                                      int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  const double x = (static_cast<double>(values[base]) - stats[0]) * stats[2];
  const double y_centered = static_cast<double>(values[base + 1u]) - stats[1];
  const double y = (y_centered - stats[3] * x) * stats[4];
  values[base] = static_cast<float>(x);
  values[base + 1u] = static_cast<float>(y);
}

__global__ void tsvd_scores_to_rowmajor_init_kernel(const float* scores,
                                                    float* values,
                                                    int n,
                                                    int n_components) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const float x = scores[static_cast<std::size_t>(row)];
  const float y = n_components > 1 ?
    scores[static_cast<std::size_t>(n) + row] :
    0.0f;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  values[base] = isfinite(x) ? x : 0.0f;
  values[base + 1u] = isfinite(y) ? y : 0.0f;
}

__global__ void scale_rsvd_scores_kernel(float* scores,
                                         const float* singular_values,
                                         int n,
                                         int n_components) {
  const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * n_components;
  if (item >= total) return;
  const int component = item / n;
  scores[item] *= singular_values[component];
}

__global__ void finalize_tsne_pca_init_stats_kernel(const double* partial,
                                                    double* stats,
                                                    int n_blocks,
                                                    int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  double* sx2 = sy + blockDim.x;
  double* sxy = sx2 + blockDim.x;
  double* sy2 = sxy + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  double ax2 = 0.0;
  double axy = 0.0;
  double ay2 = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 5u;
    ax += partial[base];
    ay += partial[base + 1u];
    ax2 += partial[base + 2u];
    axy += partial[base + 3u];
    ay2 += partial[base + 4u];
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sxy[tid] = axy;
  sy2[tid] = ay2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sxy[tid] += sxy[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double dn = static_cast<double>(max(n, 1));
    const double denom = static_cast<double>(max(n - 1, 1));
    const double mean_x = sx[0] / dn;
    const double mean_y = sy[0] / dn;
    const double ss_x = fmax(sx2[0] - sx[0] * sx[0] / dn, 0.0);
    const double ss_y = fmax(sy2[0] - sy[0] * sy[0] / dn, 0.0);
    const double sd_x = sqrt(ss_x / denom);
    const double sd_y = sqrt(ss_y / denom);
    const double max_sd = fmax(sd_x, sd_y);
    const double scale = (isfinite(max_sd) && max_sd > 0.0) ?
      (1.0e-4 / max_sd) :
      1.0;
    stats[0] = mean_x;
    stats[1] = mean_y;
    stats[2] = scale;
  }
}

__global__ void scale_tsne_pca_init_kernel(float* values,
                                           const double* stats,
                                           int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  values[base] = static_cast<float>((static_cast<double>(values[base]) - stats[0]) * stats[2]);
  values[base + 1u] = static_cast<float>((static_cast<double>(values[base + 1u]) - stats[1]) * stats[2]);
}

__global__ void fill_float_kernel(float* values,
                                  int n,
                                  float value) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid < n) values[gid] = value;
}

__global__ void finalize_cuvs_knn_kernel(const int64_t* input_indices,
                                         const float* input_distances,
                                         int* output_indices,
                                         float* output_distances,
                                         int batch_n,
                                         int search_k,
                                         int output_k,
                                         int exclude_self,
                                         int distance_mode,
                                         int row_offset,
                                         int total_n,
                                         int* invalid_rows) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= batch_n) return;
  const int global_row = row_offset + row;

  const std::size_t input_base = static_cast<std::size_t>(row) * search_k;
  int written = 0;
  for (int column = 0; column < search_k && written < output_k; ++column) {
    const std::size_t input_offset = input_base + column;
    const int64_t candidate64 = input_indices[input_offset];
    if (candidate64 < 0 || candidate64 >= static_cast<int64_t>(total_n)) continue;
    const int candidate = static_cast<int>(candidate64);
    if (exclude_self && candidate == global_row) continue;

    const float raw = input_distances[input_offset];
    float distance = raw;
    if (distance_mode == 0) {
      distance = sqrtf(fmaxf(raw, 0.0f));
    } else if (distance_mode == 1) {
      distance = 0.5f * fmaxf(raw, 0.0f);
    }

    const std::size_t output_offset =
      static_cast<std::size_t>(written) * total_n + global_row;
    output_indices[output_offset] = candidate + 1;
    output_distances[output_offset] = distance;
    ++written;
  }

  if (written < output_k) {
    atomicAdd(invalid_rows, 1);
    for (; written < output_k; ++written) {
      const std::size_t output_offset =
        static_cast<std::size_t>(written) * total_n + global_row;
      output_indices[output_offset] = 0;
      output_distances[output_offset] = FASTEMBEDR_CUDA_FLOAT_INF;
    }
  }
}

__global__ void finalize_cuvs_query_knn_kernel(
    const int64_t* input_indices,
    const float* input_distances,
    int* output_indices,
    float* output_distances,
    int batch_n,
    int k,
    int distance_mode,
    int output_row_offset,
    int output_n,
    int reference_n,
    int* invalid_rows) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= batch_n) return;
  const int output_row = output_row_offset + row;
  const std::size_t input_base = static_cast<std::size_t>(row) * k;
  int written = 0;
  for (int column = 0; column < k; ++column) {
    const std::size_t input_offset = input_base + column;
    const int64_t candidate64 = input_indices[input_offset];
    if (candidate64 < 0 || candidate64 >= static_cast<int64_t>(reference_n)) {
      continue;
    }
    const float raw = input_distances[input_offset];
    float distance = raw;
    if (distance_mode == 0) {
      distance = sqrtf(fmaxf(raw, 0.0f));
    } else if (distance_mode == 1) {
      distance = 0.5f * fmaxf(raw, 0.0f);
    }
    const std::size_t output_offset =
      static_cast<std::size_t>(written) * output_n + output_row;
    output_indices[output_offset] = static_cast<int>(candidate64) + 1;
    output_distances[output_offset] = distance;
    ++written;
  }
  if (written < k) atomicAdd(invalid_rows, 1);
}

__global__ void standardize_stats_kernel(const double* values,
                                         double* stats,
                                         int n,
                                         int p) {
  extern __shared__ double shared[];
  double* sums = shared;
  double* sums2 = sums + blockDim.x;
  const int col = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (col >= p) return;

  double sum = 0.0;
  double sum2 = 0.0;
  const std::size_t base = static_cast<std::size_t>(col) * n;
  for (int row = tid; row < n; row += static_cast<int>(blockDim.x)) {
    const double x = values[base + row];
    sum += x;
    sum2 += x * x;
  }
  sums[tid] = sum;
  sums2[tid] = sum2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
      sums2[tid] += sums2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double dn = static_cast<double>(max(n, 1));
    const double mean = sums[0] / dn;
    const double denom = static_cast<double>(max(n - 1, 1));
    double variance = (sums2[0] - sums[0] * sums[0] / dn) / denom;
    if (!isfinite(variance) || variance <= 0.0) variance = 1.0;
    double scale = sqrt(variance);
    if (!isfinite(scale) || scale <= 0.0) scale = 1.0;
    stats[col] = mean;
    stats[static_cast<std::size_t>(p) + col] = scale;
  }
}

__global__ void standardize_apply_kernel(double* values,
                                         const double* stats,
                                         int n,
                                         int p) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * p;
  if (gid >= total) return;
  const int col = gid / n;
  const double center = stats[col];
  const double scale = stats[static_cast<std::size_t>(p) + col];
  values[gid] = (values[gid] - center) / scale;
}

__global__ void membership_project_kernel(const double* reference_layout,
                                          const int* projection_indices,
                                          const double* projection_distances,
                                          double* out,
                                          int n_reference,
                                          int n_query,
                                          int k,
                                          int n_components,
                                          int average_zeros) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;

  constexpr double eps = 1.4901161193847656e-8;
  double adjusted[kCudaProjectionMaxK];
  double scratch[kCudaProjectionMaxK];
  int zero_count = 0;
  int first_zero = -1;
  double rho = kCudaDoubleInf;

  for (int j = 0; j < k; ++j) {
    const int idx = projection_indices[static_cast<std::size_t>(j) * n_query + row] - 1;
    double d = projection_distances[static_cast<std::size_t>(j) * n_query + row];
    if (!isfinite(d) || d < 0.0) d = 0.0;
    if (idx < 0 || idx >= n_reference) d = kCudaDoubleInf;
    if (d <= eps) {
      ++zero_count;
      if (first_zero < 0) first_zero = j;
    }
    if (d < rho) rho = d;
  }

  if (zero_count > 0) {
    const double inv_zero = average_zeros ? 1.0 / static_cast<double>(zero_count) : 1.0;
    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      for (int j = 0; j < k; ++j) {
        double d = projection_distances[static_cast<std::size_t>(j) * n_query + row];
        if (!isfinite(d) || d < 0.0) d = 0.0;
        if ((average_zeros && d <= eps) || (!average_zeros && j == first_zero)) {
          const int idx = projection_indices[static_cast<std::size_t>(j) * n_query + row] - 1;
          if (idx >= 0 && idx < n_reference) {
            value += inv_zero * reference_layout[static_cast<std::size_t>(c) * n_reference + idx];
          }
        }
      }
      out[static_cast<std::size_t>(c) * n_query + row] = value;
    }
    return;
  }

  int positive_count = 0;
  for (int j = 0; j < k; ++j) {
    double d = projection_distances[static_cast<std::size_t>(j) * n_query + row];
    if (!isfinite(d) || d < 0.0) d = 0.0;
    const double value = fmax(0.0, d - rho);
    adjusted[j] = value;
    if (value > eps) scratch[positive_count++] = value;
  }
  if (positive_count == 0) {
    for (int j = 0; j < k; ++j) {
      double d = projection_distances[static_cast<std::size_t>(j) * n_query + row];
      if (!isfinite(d) || d < 0.0) d = 0.0;
      scratch[j] = d;
    }
    positive_count = k;
  }
  double sigma = median_device(scratch, positive_count);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  double weight_sum = 0.0;
  for (int j = 0; j < k; ++j) {
    const double w = exp(-adjusted[j] / sigma);
    adjusted[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0) {
    weight_sum = static_cast<double>(k);
    for (int j = 0; j < k; ++j) adjusted[j] = 1.0;
  }

  for (int c = 0; c < n_components; ++c) {
    double value = 0.0;
    for (int j = 0; j < k; ++j) {
      const int idx = projection_indices[static_cast<std::size_t>(j) * n_query + row] - 1;
      if (idx >= 0 && idx < n_reference) {
        value += adjusted[j] * reference_layout[static_cast<std::size_t>(c) * n_reference + idx];
      }
    }
    out[static_cast<std::size_t>(c) * n_query + row] = value / weight_sum;
  }
}

__global__ void landmark_project_interpolate_knn_confidence_kernel(const double* landmark_data,
                                                                   const double* query_data,
                                                                   const double* landmark_layout,
                                                                   double* out,
                                                                   int* out_indices,
                                                                   double* out_distances,
                                                                   double* out_confidence,
                                                                   int n_landmarks,
                                                                   int n_query,
                                                                   int n_features,
                                                                   int k,
                                                                   int n_components) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n_query) return;

  constexpr double eps = 1.4901161193847656e-8;
  double best_sq[kCudaProjectionMaxK];
  int best_idx[kCudaProjectionMaxK];
  for (int j = 0; j < k; ++j) {
    best_sq[j] = kCudaDoubleInf;
    best_idx[j] = INT_MAX;
  }

  for (int i = 0; i < n_landmarks; ++i) {
    double dist = 0.0;
    for (int c = 0; c < n_features; ++c) {
      const double diff =
        landmark_data[static_cast<std::size_t>(c) * n_landmarks + i] -
        query_data[static_cast<std::size_t>(c) * n_query + row];
      dist += diff * diff;
    }

    if (dist < best_sq[k - 1] ||
        (dist == best_sq[k - 1] && i < best_idx[k - 1])) {
      int pos = k - 1;
      while (pos > 0 &&
             (dist < best_sq[pos - 1] ||
              (dist == best_sq[pos - 1] && i < best_idx[pos - 1]))) {
        best_sq[pos] = best_sq[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
      }
      best_sq[pos] = dist;
      best_idx[pos] = i;
    }
  }

  double distances[kCudaProjectionMaxK];
  double adjusted[kCudaProjectionMaxK];
  double scratch[kCudaProjectionMaxK];
  int zero_count = 0;
  int first_zero = -1;
  double rho = kCudaDoubleInf;

  for (int j = 0; j < k; ++j) {
    const double d = sqrt(fmax(best_sq[j], 0.0));
    distances[j] = d;
    out_indices[static_cast<std::size_t>(j) * n_query + row] = best_idx[j] + 1;
    out_distances[static_cast<std::size_t>(j) * n_query + row] = d;
    if (d <= eps) {
      ++zero_count;
      if (first_zero < 0) first_zero = j;
    }
    if (d < rho) rho = d;
  }

  if (zero_count > 0) {
    out_confidence[row] = 1.0;
    const int idx = best_idx[first_zero];
    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      if (idx >= 0 && idx < n_landmarks) {
        value = landmark_layout[static_cast<std::size_t>(c) * n_landmarks + idx];
      }
      out[static_cast<std::size_t>(c) * n_query + row] = value;
    }
    return;
  }

  int positive_count = 0;
  for (int j = 0; j < k; ++j) {
    const double value = fmax(0.0, distances[j] - rho);
    adjusted[j] = value;
    if (value > eps) scratch[positive_count++] = value;
  }
  if (positive_count == 0) {
    for (int j = 0; j < k; ++j) scratch[j] = distances[j];
    positive_count = k;
  }
  double sigma = median_device(scratch, positive_count);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  double weight_sum = 0.0;
  for (int j = 0; j < k; ++j) {
    const double w = exp(-adjusted[j] / sigma);
    adjusted[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0) {
    weight_sum = static_cast<double>(k);
    for (int j = 0; j < k; ++j) adjusted[j] = 1.0;
  }

  double max_probability = 0.0;
  double entropy = 0.0;
  for (int j = 0; j < k; ++j) {
    const double probability = adjusted[j] / weight_sum;
    max_probability = fmax(max_probability, probability);
    entropy -= probability * log(fmax(probability, eps));
  }
  const double entropy_score = k > 1 ? 1.0 - fmin(1.0, entropy / log(static_cast<double>(k))) : 1.0;
  const double confidence = 0.65 * entropy_score + 0.35 * max_probability;
  out_confidence[row] = fmin(1.0, fmax(0.0, confidence));

  for (int c = 0; c < n_components; ++c) {
    double value = 0.0;
    for (int j = 0; j < k; ++j) {
      const int idx = best_idx[j];
      if (idx >= 0 && idx < n_landmarks) {
        value += adjusted[j] * landmark_layout[static_cast<std::size_t>(c) * n_landmarks + idx];
      }
    }
    out[static_cast<std::size_t>(c) * n_query + row] = value / weight_sum;
  }
}

__global__ void overwrite_landmark_rows_kernel(double* out,
                                               const double* landmark_layout,
                                               const int* landmark_indices,
                                               int n_landmarks,
                                               int n,
                                               int n_components) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n_landmarks * n_components;
  if (gid >= total) return;
  const int landmark = gid % n_landmarks;
  const int component = gid / n_landmarks;
  const int row = landmark_indices[landmark] - 1;
  if (row < 0 || row >= n) return;
  out[static_cast<std::size_t>(component) * n + row] =
    landmark_layout[static_cast<std::size_t>(component) * n_landmarks + landmark];
}

static __device__ __forceinline__ int high_rank_device(const int* indices,
                                int index_rows,
                                int index_row,
                                int candidate,
                                int high_rank_limit) {
  for (int r = 0; r < high_rank_limit; ++r) {
    if (indices[static_cast<std::size_t>(r) * index_rows + index_row] - 1 == candidate) {
      return r + 1;
    }
  }
  return high_rank_limit + 1;
}

__global__ void structure_score_rows_kernel(const double* layout,
                                            const int* indices,
                                            const int* keep,
                                            const int* labels,
                                            double* row_scores,
                                            int n,
                                            int index_rows,
                                            int high_rank_limit,
                                            int preserve_k,
                                            int keep_n,
                                            int compact_indices,
                                            int n_label_levels) {
  const int kk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (kk >= keep_n) return;

  const int query = keep[kk] - 1;
  if (query < 0 || query >= n) return;
  const int index_row = compact_indices ? kk : query;
  if (index_row < 0 || index_row >= index_rows) return;

  double low_dist[kCudaScoreMaxK];
  int low_idx[kCudaScoreMaxK];
  double high_dist[kCudaScoreMaxK];
  int high_idx[kCudaScoreMaxK];
  int low_count = 0;
  int high_count = 0;

  for (int r = 0; r < preserve_k; ++r) {
    const int high_nb = indices[static_cast<std::size_t>(r) * index_rows + index_row] - 1;
    if (high_nb < 0 || high_nb >= n) continue;
    const double d2 = layout_d2_2d(layout, n, query, high_nb);
    high_dist[high_count] = d2;
    high_idx[high_count] = high_nb;
    ++high_count;
  }
  for (int a = 1; a < high_count; ++a) {
    const double d = high_dist[a];
    const int idx = high_idx[a];
    int b = a - 1;
    while (b >= 0 && pair_less_device(d, idx, high_dist[b], high_idx[b])) {
      high_dist[b + 1] = high_dist[b];
      high_idx[b + 1] = high_idx[b];
      --b;
    }
    high_dist[b + 1] = d;
    high_idx[b + 1] = idx;
  }

  for (int candidate = 0; candidate < n; ++candidate) {
    if (candidate == query) continue;
    const double d2 = layout_d2_2d(layout, n, query, candidate);
    insert_top_neighbor(low_dist, low_idx, &low_count, preserve_k, d2, candidate);
  }
  if (low_count < preserve_k) return;

  int shared = 0;
  double trust_penalty = 0.0;
  for (int r = 0; r < preserve_k; ++r) {
    const int rank = high_rank_device(
      indices, index_rows, index_row, low_idx[r], high_rank_limit
    );
    if (rank <= preserve_k) ++shared;
    trust_penalty += fmax(0.0, static_cast<double>(rank - preserve_k));
  }

  double cont_penalty = 0.0;
  for (int t = 0; t < high_count; ++t) {
    int lower_rank_count = 0;
    for (int candidate = 0; candidate < n; ++candidate) {
      if (candidate == query) continue;
      const double d2 = layout_d2_2d(layout, n, query, candidate);
      if (pair_less_device(d2, candidate, high_dist[t], high_idx[t])) ++lower_rank_count;
    }
    const int low_rank = 1 + lower_rank_count;
    cont_penalty += fmax(0.0, static_cast<double>(low_rank - preserve_k));
  }

  double label_accuracy = 0.0;
  double label_accuracy_n = 0.0;
  if (labels != nullptr && n_label_levels > 0) {
    const int truth = labels[query];
    if (truth >= 1 && truth <= n_label_levels) {
      int best_label = 0;
      int best_count = 0;
      for (int label = 1; label <= n_label_levels; ++label) {
        int count = 0;
        for (int r = 0; r < preserve_k; ++r) {
          const int nb_label = labels[low_idx[r]];
          if (nb_label == label) ++count;
        }
        if (count > best_count) {
          best_count = count;
          best_label = label;
        }
      }
      if (best_count > 0) {
        label_accuracy = best_label == truth ? 1.0 : 0.0;
        label_accuracy_n = 1.0;
      }
    }
  }

  const double trust_denom = static_cast<double>(preserve_k) *
    static_cast<double>(max(1, high_rank_limit + 1 - preserve_k));
  const double cont_denom = static_cast<double>(preserve_k) *
    static_cast<double>(max(1, n - preserve_k));
  const double preservation = static_cast<double>(shared) / static_cast<double>(preserve_k);
  const double trust = fmax(0.0, fmin(1.0, 1.0 - trust_penalty / trust_denom));
  const double continuity = fmax(0.0, fmin(1.0, 1.0 - cont_penalty / cont_denom));
  const std::size_t base = static_cast<std::size_t>(kk) * kCudaScoreWidth;
  row_scores[base] = preservation;
  row_scores[base + 1u] = trust;
  row_scores[base + 2u] = continuity;
  row_scores[base + 3u] = label_accuracy;
  row_scores[base + 4u] = label_accuracy_n;
  row_scores[base + 5u] = 1.0;
}

__global__ void reduce_width_kernel(const double* rows,
                                    double* totals,
                                    int n_rows,
                                    int width) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  for (int metric = 0; metric < width; ++metric) {
    double value = 0.0;
    for (int row = tid; row < n_rows; row += static_cast<int>(blockDim.x)) {
      value += rows[static_cast<std::size_t>(row) * width + metric];
    }
    shared[static_cast<std::size_t>(metric) * blockDim.x + tid] = value;
  }
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int metric = 0; metric < width; ++metric) {
        shared[static_cast<std::size_t>(metric) * blockDim.x + tid] +=
          shared[static_cast<std::size_t>(metric) * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int metric = 0; metric < width; ++metric) {
      totals[metric] = shared[static_cast<std::size_t>(metric) * blockDim.x];
    }
  }
}

__global__ void silhouette_rows_kernel(const double* layout,
                                       const int* labels,
                                       const int* counts,
                                       double* row_scores,
                                       int n,
                                       int n_label_levels) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const int own_label = labels[row];
  if (own_label < 1 || own_label > n_label_levels) return;
  const double x0 = layout[row];
  const double x1 = layout[static_cast<std::size_t>(n) + row];
  if (!isfinite(x0) || !isfinite(x1)) return;

  double sums[kCudaSilhouetteMaxLabels + 1];
  for (int label = 0; label <= n_label_levels; ++label) sums[label] = 0.0;

  for (int j = 0; j < n; ++j) {
    if (j == row) continue;
    const int label = labels[j];
    if (label < 1 || label > n_label_levels) continue;
    const double y0 = layout[j];
    const double y1 = layout[static_cast<std::size_t>(n) + j];
    if (!isfinite(y0) || !isfinite(y1)) continue;
    const double dx = x0 - y0;
    const double dy = x1 - y1;
    sums[label] += sqrt(dx * dx + dy * dy);
  }

  const int own_count = counts[own_label] - 1;
  const double a = own_count > 0 ? sums[own_label] / static_cast<double>(own_count) : 0.0;
  double b = kCudaDoubleInf;
  for (int label = 1; label <= n_label_levels; ++label) {
    if (label == own_label || counts[label] <= 0) continue;
    b = fmin(b, sums[label] / static_cast<double>(counts[label]));
  }
  double value = 0.0;
  if (isfinite(b)) {
    const double denom = fmax(a, b);
    value = denom > 0.0 ? (b - a) / denom : 0.0;
  }
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  row_scores[base] = value;
  row_scores[base + 1u] = 1.0;
}

__global__ void tsne_affinity_from_knn_kernel(const int* indices,
                                              const double* distances,
                                              float* affinities,
                                              int n,
                                              int k,
                                              int offset,
                                              float perplexity) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;

  int valid = 0;
  int single_neighbor = -1;
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    ++valid;
    single_neighbor = nb;
  }
  if (valid == 0) return;

  const double row_scale = 1.0 / (2.0 * static_cast<double>(n));
  if (valid == 1) {
    affinities[static_cast<std::size_t>(row) * n + single_neighbor] =
      static_cast<float>(row_scale);
    return;
  }

  const double row_perplexity = fmax(1.0, fmin(static_cast<double>(perplexity),
                                               static_cast<double>(valid)));
  const double target_entropy = log(row_perplexity);
  double beta = 1.0;
  double beta_min = 0.0;
  double beta_max = 0.0;
  bool has_beta_min = false;
  bool has_beta_max = false;

  for (int iter = 0; iter < 200; ++iter) {
    double sum_p = kCudaDoubleMin;
    double weighted = 0.0;
    for (int j = 0; j < k; ++j) {
      const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
      const double d = distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
      const double d2 = d * d;
      const double p = exp(-d2 * beta);
      sum_p += p;
      weighted += d2 * p;
    }
    if (sum_p <= 0.0 || !isfinite(sum_p)) break;
    const double entropy = log(sum_p) + beta * weighted / sum_p;
    const double diff = entropy - target_entropy;
    if (fabs(diff) < 1.0e-5) break;
    if (diff > 0.0) {
      beta_min = beta;
      has_beta_min = true;
      beta = has_beta_max ? 0.5 * (beta + beta_max) : beta * 2.0;
    } else {
      beta_max = beta;
      has_beta_max = true;
      beta = has_beta_min ? 0.5 * (beta + beta_min) : beta * 0.5;
    }
    beta = fmax(beta, 1.0e-12);
  }

  double sum_p = kCudaDoubleMin;
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    const double d2 = d * d;
    sum_p += exp(-d2 * beta);
  }

  if (sum_p <= 0.0 || !isfinite(sum_p)) {
    const float uniform = static_cast<float>(row_scale / static_cast<double>(valid));
    for (int j = 0; j < k; ++j) {
      const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
      const double d = distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
      affinities[static_cast<std::size_t>(row) * n + nb] += uniform;
    }
    return;
  }

  const double normalizer = row_scale / sum_p;
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    const double d2 = d * d;
    affinities[static_cast<std::size_t>(row) * n + nb] +=
      static_cast<float>(exp(-d2 * beta) * normalizer);
  }
}

__global__ void tsne_init_stats_kernel(const float* values,
                                       double* partial,
                                       int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  double* sx2 = sy + blockDim.x;
  double* sy2 = sx2 + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  double ax2 = 0.0;
  double ay2 = 0.0;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       i < n;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    const double x = values[static_cast<std::size_t>(i) * 2u];
    const double y = values[static_cast<std::size_t>(i) * 2u + 1u];
    ax += x;
    ay += y;
    ax2 += x * x;
    ay2 += y * y;
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sy2[tid] = ay2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 4u;
    partial[base] = sx[0];
    partial[base + 1u] = sy[0];
    partial[base + 2u] = sx2[0];
    partial[base + 3u] = sy2[0];
  }
}

__global__ void tsne_finalize_init_stats_kernel(const double* partial,
                                                double* stats,
                                                int n_blocks,
                                                int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  double* sx2 = sy + blockDim.x;
  double* sy2 = sx2 + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  double ax2 = 0.0;
  double ay2 = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 4u;
    ax += partial[base];
    ay += partial[base + 1u];
    ax2 += partial[base + 2u];
    ay2 += partial[base + 3u];
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sy2[tid] = ay2;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double dn = static_cast<double>(max(n, 1));
    const double denom = static_cast<double>(max(n - 1, 1));
    const double mean_x = sx[0] / dn;
    const double mean_y = sy[0] / dn;
    const double ss_x = fmax(sx2[0] - sx[0] * sx[0] / dn, 1.0e-24);
    const double ss_y = fmax(sy2[0] - sy[0] * sy[0] / dn, 1.0e-24);
    stats[0] = mean_x;
    stats[1] = mean_y;
    stats[2] = 1.0;
    stats[3] = 1.0;
  }
}

__global__ void tsne_scale_init_kernel(float* values,
                                       const double* stats,
                                       int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  values[base] = static_cast<float>((static_cast<double>(values[base]) - stats[0]) * stats[2]);
  values[base + 1u] = static_cast<float>((static_cast<double>(values[base + 1u]) - stats[1]) * stats[3]);
}

__global__ void tsne_sum_q_kernel(const float* current,
                                  double* partial,
                                  int n) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double local = 0.0;
  if (row < n) {
    const double yi_x = current[static_cast<std::size_t>(row) * 2u];
    const double yi_y = current[static_cast<std::size_t>(row) * 2u + 1u];
    for (int j = row + 1; j < n; ++j) {
      const double dx = yi_x - current[static_cast<std::size_t>(j) * 2u];
      const double dy = yi_y - current[static_cast<std::size_t>(j) * 2u + 1u];
      local += 2.0 / (1.0 + dx * dx + dy * dy);
    }
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void tsne_finalize_sum_kernel(const double* partial,
                                         double* stats,
                                         int n_blocks) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  double total = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    total += partial[b];
  }
  shared[tid] = total;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) stats[0] = fmax(shared[0], 1.0e-12);
}

__global__ void tsne_exact_gradient_kernel(const float* current,
                                           const float* affinities,
                                           float* grad,
                                           const double* stats,
                                           int n,
                                           float exaggeration) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const double sum_q = stats[0];
  const double yi_x = current[static_cast<std::size_t>(row) * 2u];
  const double yi_y = current[static_cast<std::size_t>(row) * 2u + 1u];
  double gx = 0.0;
  double gy = 0.0;
  const std::size_t row_base = static_cast<std::size_t>(row) * n;
  for (int j = 0; j < n; ++j) {
    if (j == row) continue;
    const double dx = yi_x - current[static_cast<std::size_t>(j) * 2u];
    const double dy = yi_y - current[static_cast<std::size_t>(j) * 2u + 1u];
    const double q = 1.0 / (1.0 + dx * dx + dy * dy);
    const double q_prob = q / sum_q;
    const double pij =
      static_cast<double>(affinities[row_base + j]) +
      static_cast<double>(affinities[static_cast<std::size_t>(j) * n + row]);
    const double mult = 4.0 * (static_cast<double>(exaggeration) * pij - q_prob) * q;
    gx += mult * dx;
    gy += mult * dy;
  }
  grad[static_cast<std::size_t>(row) * 2u] = static_cast<float>(gx);
  grad[static_cast<std::size_t>(row) * 2u + 1u] = static_cast<float>(gy);
}

__global__ void tsne_update_reduce_kernel(float* current,
                                          const float* grad,
                                          float* gains,
                                          float* inc,
                                          double* partial,
                                          int n,
                                          float eta,
                                          float momentum) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double x = 0.0;
  double y = 0.0;
  if (row < n) {
    const std::size_t base = static_cast<std::size_t>(row) * 2u;
    for (int c = 0; c < 2; ++c) {
      const std::size_t pos = base + static_cast<std::size_t>(c);
      const float g = grad[pos];
      const float old_inc = inc[pos];
      const bool sign_changed =
        tsne_sign_component(g) != tsne_sign_component(old_inc);
      float gain = sign_changed ? gains[pos] + 0.2f : gains[pos] * 0.8f;
      gain = fmaxf(gain, 0.01f);
      const float step = momentum * old_inc - eta * gain * g;
      gains[pos] = gain;
      inc[pos] = step;
      current[pos] += step;
    }
    x = current[base];
    y = current[base + 1u];
  }
  sx[tid] = x;
  sy[tid] = y;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 2u;
    partial[base] = sx[0];
    partial[base + 1u] = sy[0];
  }
}

__global__ void tsne_finalize_mean_kernel(const double* partial,
                                          double* stats,
                                          int n_blocks,
                                          int n) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double ax = 0.0;
  double ay = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 2u;
    ax += partial[base];
    ay += partial[base + 1u];
  }
  sx[tid] = ax;
  sy[tid] = ay;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double dn = static_cast<double>(max(n, 1));
    stats[0] = sx[0] / dn;
    stats[1] = sy[0] / dn;
  }
}

__global__ void tsne_center_kernel(float* current,
                                   const double* stats,
                                   int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  current[base] -= static_cast<float>(stats[0]);
  current[base + 1u] -= static_cast<float>(stats[1]);
}

template <typename DistanceT>
__global__ void opentsne_affinity_sparse_kernel(const int* indices,
                                                const DistanceT* distances,
                                                float* probabilities,
                                                int n,
                                                int k,
                                                int offset,
                                                float perplexity) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;

  const std::size_t row_base = static_cast<std::size_t>(row) * k;
  for (int j = 0; j < k; ++j) probabilities[row_base + j] = 0.0f;

  int valid = 0;
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    ++valid;
  }
  if (valid == 0) return;
  if (valid == 1) {
    for (int j = 0; j < k; ++j) {
      const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
      const double d = distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
      probabilities[row_base + j] = 0.5f / static_cast<float>(n);
      return;
    }
  }

  const double row_perplexity = fmax(1.0, fmin(static_cast<double>(perplexity),
                                               static_cast<double>(valid)));
  const double target_entropy = log(row_perplexity);
  double beta = 1.0;
  double beta_min = 0.0;
  double beta_max = 0.0;
  bool has_beta_min = false;
  bool has_beta_max = false;

  for (int iter = 0; iter < 200; ++iter) {
    double sum_p = kCudaDoubleMin;
    double weighted = 0.0;
    for (int j = 0; j < k; ++j) {
      const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
      const double d = distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
      const double d2 = d * d;
      const double p = exp(-d2 * beta);
      sum_p += p;
      weighted += d2 * p;
    }
    const double entropy = log(sum_p) + beta * weighted / sum_p;
    const double diff = entropy - target_entropy;
    if (fabs(diff) < 1.0e-5) break;
    if (diff > 0.0) {
      beta_min = beta;
      has_beta_min = true;
      beta = has_beta_max ? 0.5 * (beta + beta_max) : beta * 2.0;
    } else {
      beta_max = beta;
      has_beta_max = true;
      beta = has_beta_min ? 0.5 * (beta + beta_min) : beta * 0.5;
    }
    beta = fmax(beta, 1.0e-12);
  }

  double sum_p = kCudaDoubleMin;
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    sum_p += exp(-(d * d) * beta);
  }
  const double normalizer = 0.5 / (static_cast<double>(n) * sum_p);
  for (int j = 0; j < k; ++j) {
    const int nb = indices[static_cast<std::size_t>(j) * n + row] - offset;
    const double d = distances[static_cast<std::size_t>(j) * n + row];
    if (nb < 0 || nb >= n || nb == row || !isfinite(d) || d < 0.0) continue;
    probabilities[row_base + j] = static_cast<float>(exp(-(d * d) * beta) * normalizer);
  }
}

__global__ void opentsne_layout_stats_blocks_kernel(const float* current,
                                                    double* partial,
                                                    int n) {
  extern __shared__ double shared[];
  double* min_x = shared;
  double* max_x = min_x + blockDim.x;
  double* min_y = max_x + blockDim.x;
  double* max_y = min_y + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double mnx = kCudaDoubleInf;
  double mxx = -kCudaDoubleInf;
  double mny = kCudaDoubleInf;
  double mxy = -kCudaDoubleInf;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       i < n;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    const double x = current[static_cast<std::size_t>(i) * 2u];
    const double y = current[static_cast<std::size_t>(i) * 2u + 1u];
    mnx = fmin(mnx, x);
    mxx = fmax(mxx, x);
    mny = fmin(mny, y);
    mxy = fmax(mxy, y);
  }
  min_x[tid] = mnx;
  max_x[tid] = mxx;
  min_y[tid] = mny;
  max_y[tid] = mxy;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      min_x[tid] = fmin(min_x[tid], min_x[tid + stride]);
      max_x[tid] = fmax(max_x[tid], max_x[tid + stride]);
      min_y[tid] = fmin(min_y[tid], min_y[tid + stride]);
      max_y[tid] = fmax(max_y[tid], max_y[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 4u;
    partial[base] = min_x[0];
    partial[base + 1u] = max_x[0];
    partial[base + 2u] = min_y[0];
    partial[base + 3u] = max_y[0];
  }
}

__global__ void opentsne_finalize_layout_stats_kernel(const double* partial,
                                                      double* stats,
                                                      int n_blocks,
                                                      int grid_size) {
  extern __shared__ double shared[];
  double* min_x = shared;
  double* max_x = min_x + blockDim.x;
  double* min_y = max_x + blockDim.x;
  double* max_y = min_y + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double mnx = kCudaDoubleInf;
  double mxx = -kCudaDoubleInf;
  double mny = kCudaDoubleInf;
  double mxy = -kCudaDoubleInf;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 4u;
    mnx = fmin(mnx, partial[base]);
    mxx = fmax(mxx, partial[base + 1u]);
    mny = fmin(mny, partial[base + 2u]);
    mxy = fmax(mxy, partial[base + 3u]);
  }
  min_x[tid] = mnx;
  max_x[tid] = mxx;
  min_y[tid] = mny;
  max_y[tid] = mxy;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      min_x[tid] = fmin(min_x[tid], min_x[tid + stride]);
      max_x[tid] = fmax(max_x[tid], max_x[tid + stride]);
      min_y[tid] = fmin(min_y[tid], min_y[tid + stride]);
      max_y[tid] = fmax(max_y[tid], max_y[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double cx = 0.5 * (min_x[0] + max_x[0]);
    const double cy = 0.5 * (min_y[0] + max_y[0]);
    double span = fmax(max_x[0] - min_x[0], max_y[0] - min_y[0]);
    if (!isfinite(span) || span <= 0.0) span = 1.0;
    const double half = 0.55 * span + 1.0e-3;
    const double spacing = (2.0 * half) / static_cast<double>(grid_size - 1);
    stats[0] = cx - half;
    stats[1] = cy - half;
    stats[2] = spacing;
    stats[3] = 1.0 / spacing;
  }
}

__global__ void opentsne_fft_clear_kernel(cufftComplex* mass,
                                          cufftComplex* mass_x,
                                          cufftComplex* mass_y,
                                          cufftComplex* kernel_q,
                                          cufftComplex* kernel_q2,
                                          const double* stats,
                                          int fft_size,
                                          int grid_size) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = fft_size * fft_size;
  if (gid >= total) return;
  mass[gid].x = mass[gid].y = 0.0f;
  mass_x[gid].x = mass_x[gid].y = 0.0f;
  mass_y[gid].x = mass_y[gid].y = 0.0f;
  kernel_q[gid].x = kernel_q[gid].y = 0.0f;
  kernel_q2[gid].x = kernel_q2[gid].y = 0.0f;

  const int y = gid / fft_size;
  const int x = gid - y * fft_size;
  const int dx = x < grid_size ? x : x - fft_size;
  const int dy = y < grid_size ? y : y - fft_size;
  if (abs(dx) < grid_size && abs(dy) < grid_size) {
    const float spacing = static_cast<float>(stats[2]);
    const float d2 = (static_cast<float>(dx * dx + dy * dy) * spacing * spacing);
    const float q = 1.0f / (1.0f + d2);
    kernel_q[gid].x = q;
    kernel_q2[gid].x = q * q;
  }
}

__global__ void opentsne_fft_scatter_kernel(const float* current,
                                            cufftComplex* mass,
                                            cufftComplex* mass_x,
                                            cufftComplex* mass_y,
                                            float2* grid_pos,
                                            const double* stats,
                                            int n,
                                            int grid_size,
                                            int fft_size) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  const float x = current[base];
  const float y = current[base + 1u];
  const float gx_raw = (x - static_cast<float>(stats[0])) * static_cast<float>(stats[3]);
  const float gy_raw = (y - static_cast<float>(stats[1])) * static_cast<float>(stats[3]);
  const float gx = fminf(static_cast<float>(grid_size - 1), fmaxf(0.0f, gx_raw));
  const float gy = fminf(static_cast<float>(grid_size - 1), fmaxf(0.0f, gy_raw));
  const int x0 = min(grid_size - 2, max(0, static_cast<int>(floorf(gx))));
  const int y0 = min(grid_size - 2, max(0, static_cast<int>(floorf(gy))));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float tx = gx - static_cast<float>(x0);
  const float ty = gy - static_cast<float>(y0);
  const float w00 = (1.0f - tx) * (1.0f - ty);
  const float w10 = tx * (1.0f - ty);
  const float w01 = (1.0f - tx) * ty;
  const float w11 = tx * ty;
  const int p00 = y0 * fft_size + x0;
  const int p10 = y0 * fft_size + x1;
  const int p01 = y1 * fft_size + x0;
  const int p11 = y1 * fft_size + x1;
  atomicAdd(&mass[p00].x, w00);
  atomicAdd(&mass[p10].x, w10);
  atomicAdd(&mass[p01].x, w01);
  atomicAdd(&mass[p11].x, w11);
  atomicAdd(&mass_x[p00].x, w00 * x);
  atomicAdd(&mass_x[p10].x, w10 * x);
  atomicAdd(&mass_x[p01].x, w01 * x);
  atomicAdd(&mass_x[p11].x, w11 * x);
  atomicAdd(&mass_y[p00].x, w00 * y);
  atomicAdd(&mass_y[p10].x, w10 * y);
  atomicAdd(&mass_y[p01].x, w01 * y);
  atomicAdd(&mass_y[p11].x, w11 * y);
  grid_pos[row] = make_float2(gx, gy);
}

__global__ void opentsne_fft_multiply_all_kernel(const cufftComplex* mass,
                                                 const cufftComplex* mass_x,
                                                 const cufftComplex* mass_y,
                                                 const cufftComplex* kernel_q,
                                                 const cufftComplex* kernel_q2,
                                                 cufftComplex* q,
                                                 cufftComplex* q2,
                                                 cufftComplex* xq2,
                                                 cufftComplex* yq2,
                                                 int total) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= total) return;
  const cufftComplex m = mass[gid];
  const cufftComplex mx = mass_x[gid];
  const cufftComplex my = mass_y[gid];
  const cufftComplex kq = kernel_q[gid];
  const cufftComplex kq2 = kernel_q2[gid];
  q[gid].x = m.x * kq.x - m.y * kq.y;
  q[gid].y = m.x * kq.y + m.y * kq.x;
  q2[gid].x = m.x * kq2.x - m.y * kq2.y;
  q2[gid].y = m.x * kq2.y + m.y * kq2.x;
  xq2[gid].x = mx.x * kq2.x - mx.y * kq2.y;
  xq2[gid].y = mx.x * kq2.y + mx.y * kq2.x;
  yq2[gid].x = my.x * kq2.x - my.y * kq2.y;
  yq2[gid].y = my.x * kq2.y + my.y * kq2.x;
}

static __device__ __forceinline__ float bilinear_grid_complex_real(
                                            const cufftComplex* values,
                                            int fft_size,
                                            int grid_size,
                                            float gx,
                                            float gy) {
  const float cx = fminf(static_cast<float>(grid_size - 1), fmaxf(0.0f, gx));
  const float cy = fminf(static_cast<float>(grid_size - 1), fmaxf(0.0f, gy));
  const int x0 = min(grid_size - 2, max(0, static_cast<int>(floorf(cx))));
  const int y0 = min(grid_size - 2, max(0, static_cast<int>(floorf(cy))));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float tx = cx - static_cast<float>(x0);
  const float ty = cy - static_cast<float>(y0);
  const float w00 = (1.0f - tx) * (1.0f - ty);
  const float w10 = tx * (1.0f - ty);
  const float w01 = (1.0f - tx) * ty;
  const float w11 = tx * ty;
  return w00 * values[y0 * fft_size + x0].x +
    w10 * values[y0 * fft_size + x1].x +
    w01 * values[y1 * fft_size + x0].x +
    w11 * values[y1 * fft_size + x1].x;
}

__global__ void opentsne_fft_sum_q_kernel(const cufftComplex* q_grid,
                                          const float2* grid_pos,
                                          double* partial,
                                          int n,
                                          int fft_size,
                                          int grid_size,
                                          float fft_scale) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  double local = 0.0;
  for (int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       row < n;
       row += static_cast<int>(gridDim.x * blockDim.x)) {
    const float2 gp = grid_pos[row];
    local += static_cast<double>(
      fft_scale * bilinear_grid_complex_real(q_grid, fft_size, grid_size, gp.x, gp.y)
    );
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void opentsne_fft_finalize_sum_q_kernel(const double* partial,
                                                   double* stats,
                                                   int n_blocks) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  double total = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    total += partial[b];
  }
  shared[tid] = total;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0) stats[0] = shared[0];
}

__global__ void opentsne_fft_repulsive_gradient_kernel(const float* current,
                                                       float* grad,
                                                       const cufftComplex* q2_grid,
                                                       const cufftComplex* xq2_grid,
                                                       const cufftComplex* yq2_grid,
                                                       const float2* grid_pos,
                                                       const double* stats,
                                                       int n,
                                                       int fft_size,
                                                       int grid_size,
                                                       float fft_scale) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  const float2 gp = grid_pos[row];
  const double sum_q = fmax(stats[0] - static_cast<double>(n), 1.0e-12);
  const float inv_sum_q = static_cast<float>(1.0 / sum_q);
  const float q2 = fft_scale * bilinear_grid_complex_real(q2_grid, fft_size, grid_size, gp.x, gp.y);
  const float xq2 = fft_scale * bilinear_grid_complex_real(xq2_grid, fft_size, grid_size, gp.x, gp.y);
  const float yq2 = fft_scale * bilinear_grid_complex_real(yq2_grid, fft_size, grid_size, gp.x, gp.y);
  grad[base] = -(current[base] * q2 - xq2) * inv_sum_q;
  grad[base + 1u] = -(current[base + 1u] * q2 - yq2) * inv_sum_q;
}

__global__ void opentsne_sparse_attractive_kernel(const int* indices,
                                                  const float* probabilities,
                                                  const float* current,
                                                  float* grad,
                                                  int n,
                                                  int k,
                                                  int offset,
                                                  float exaggeration) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * k;
  if (gid >= total) return;
  const int row = gid / k;
  const int col = gid - row * k;
  const int nb = indices[static_cast<std::size_t>(col) * n + row] - offset;
  if (nb < 0 || nb >= n || nb == row) return;
  const float p = probabilities[static_cast<std::size_t>(row) * k + col];
  if (p <= 0.0f || !isfinite(p)) return;
  const std::size_t rb = static_cast<std::size_t>(row) * 2u;
  const std::size_t nb_base = static_cast<std::size_t>(nb) * 2u;
  const float dx = current[rb] - current[nb_base];
  const float dy = current[rb + 1u] - current[nb_base + 1u];
  const float q = 1.0f / (1.0f + dx * dx + dy * dy);
  const float coeff = exaggeration * p * q;
  const float gx = coeff * dx;
  const float gy = coeff * dy;
  atomicAdd(grad + rb, gx);
  atomicAdd(grad + rb + 1u, gy);
  atomicAdd(grad + nb_base, -gx);
  atomicAdd(grad + nb_base + 1u, -gy);
}

__global__ void opentsne_update_reduce_kernel(float* current,
                                              const float* grad,
                                              float* gains,
                                              float* update,
                                              double* partial,
                                              int n,
                                              float learning_rate,
                                              float momentum,
                                              float min_gain,
                                              float max_step_norm) {
  extern __shared__ double shared[];
  double* sx = shared;
  double* sy = sx + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  double x_sum = 0.0;
  double y_sum = 0.0;
  if (row < n) {
    const std::size_t base = static_cast<std::size_t>(row) * 2u;
    float step_x;
    float step_y;
    for (int d = 0; d < 2; ++d) {
      const std::size_t pos = base + static_cast<std::size_t>(d);
      const float g = grad[pos];
      const float old = update[pos];
      const bool sign_changed =
        tsne_sign_component(old) != tsne_sign_component(g);
      float gain = sign_changed ?
        gains[pos] + 0.2f : gains[pos] * 0.8f + min_gain;
      gain = fmaxf(gain, min_gain);
      const float step = momentum * old - learning_rate * gain * g;
      gains[pos] = gain;
      update[pos] = step;
      if (d == 0) step_x = step; else step_y = step;
    }
    const bool clip = isfinite(max_step_norm) && max_step_norm > 0.0f;
    if (clip) {
      const float norm2 = step_x * step_x + step_y * step_y;
      const float max2 = max_step_norm * max_step_norm;
      if (norm2 > max2) {
        const float scale = max_step_norm / sqrtf(fmaxf(norm2, 1.0e-20f));
        step_x *= scale;
        step_y *= scale;
        update[base] = step_x;
        update[base + 1u] = step_y;
      }
    }
    current[base] += step_x;
    current[base + 1u] += step_y;
    x_sum = current[base];
    y_sum = current[base + 1u];
  }
  sx[tid] = x_sum;
  sy[tid] = y_sum;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 2u;
    partial[base] = sx[0];
    partial[base + 1u] = sy[0];
  }
}

__global__ void matrix_multiply_kernel(const double* left,
                                       const double* right,
                                       double* out,
                                       int left_rows,
                                       int left_cols,
                                       int right_cols,
                                       int transpose_left) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int out_rows = transpose_left ? left_cols : left_rows;
  const int inner = transpose_left ? left_rows : left_cols;
  if (row >= out_rows || col >= right_cols) return;

  double total = 0.0;
  if (transpose_left) {
    for (int t = 0; t < inner; ++t) {
      total += left[static_cast<std::size_t>(t) + static_cast<std::size_t>(row) * left_rows] *
        right[static_cast<std::size_t>(t) + static_cast<std::size_t>(col) * left_rows];
    }
  } else {
    for (int t = 0; t < inner; ++t) {
      total += left[static_cast<std::size_t>(row) + static_cast<std::size_t>(t) * left_rows] *
        right[static_cast<std::size_t>(t) + static_cast<std::size_t>(col) * left_cols];
    }
  }
  out[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * out_rows] = total;
}

__global__ void embed_epoch_kernel(const float* current,
                                   float* next,
                                   const int* neighbors,
                                   const float* weights,
                                   EmbedParams p,
                                   unsigned int epoch) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= p.n) return;

  const float yi_x = current[static_cast<std::size_t>(gid) * 2u];
  const float yi_y = current[static_cast<std::size_t>(gid) * 2u + 1u];
  float delta_x = 0.0f;
  float delta_y = 0.0f;

  for (int e = 0; e < p.k; ++e) {
    const int nb = neighbors[static_cast<std::size_t>(gid) * p.k + e];
    if (nb < 0 || nb >= p.n || nb == gid) continue;
    const float w = weights[static_cast<std::size_t>(gid) * p.k + e];
    const int positive_samples = positive_samples_this_epoch(w, p, epoch);
    if (positive_samples <= 0) continue;
    const float nb_x = current[static_cast<std::size_t>(nb) * 2u];
    const float nb_y = current[static_cast<std::size_t>(nb) * 2u + 1u];
    const float dx = yi_x - nb_x;
    const float dy = yi_y - nb_y;
    const float d2 = dx * dx + dy * dy;

    if (p.objective == 3) {
      const int samples = p.negative_sample_rate > 1 ? p.negative_sample_rate : 1;
      const float triplet_w = w / static_cast<float>(samples);
      const float pos_d2 = d2 + 1.0e-4f;
      for (int s = 0; s < samples; ++s) {
        const unsigned int neg = deterministic_vertex(
          static_cast<unsigned int>(p.n),
          p.seed,
          epoch,
          static_cast<unsigned int>(gid),
          static_cast<unsigned int>(e),
          static_cast<unsigned int>(s)
        );
        if (static_cast<int>(neg) == gid || static_cast<int>(neg) == nb) continue;
        const float neg_x = current[static_cast<std::size_t>(neg) * 2u];
        const float neg_y = current[static_cast<std::size_t>(neg) * 2u + 1u];
        const float ndx = yi_x - neg_x;
        const float ndy = yi_y - neg_y;
        const float neg_d2 = ndx * ndx + ndy * ndy + 1.0e-4f;
        const float denom = pos_d2 + neg_d2 + 1.0e-6f;
        const float scale = triplet_w / (denom * denom);
        const float pos_coeff = -2.0f * scale * neg_d2;
        const float neg_coeff =  2.0f * scale * pos_d2;
        delta_x += clip4(pos_coeff * dx) + clip4(neg_coeff * ndx);
        delta_y += clip4(pos_coeff * dy) + clip4(neg_coeff * ndy);
      }
      continue;
    }

    const float coeff = attractive_coeff(d2, w, p);
    delta_x += clip4(coeff * dx);
    delta_y += clip4(coeff * dy);

    const int neg_samples = negative_samples_this_epoch(w, p, epoch);
    for (int s = 0; s < neg_samples; ++s) {
      const unsigned int neg = deterministic_vertex(
        static_cast<unsigned int>(p.n),
        p.seed,
        epoch,
        static_cast<unsigned int>(gid),
        static_cast<unsigned int>(e),
        static_cast<unsigned int>(s)
      );
      if (static_cast<int>(neg) == gid || static_cast<int>(neg) == nb) continue;
      const float neg_x = current[static_cast<std::size_t>(neg) * 2u];
      const float neg_y = current[static_cast<std::size_t>(neg) * 2u + 1u];
      const float ndx = yi_x - neg_x;
      const float ndy = yi_y - neg_y;
      const float nd2 = ndx * ndx + ndy * ndy;
      const float rcoeff = repulsive_coeff(nd2, p);
      delta_x += clip4(rcoeff * ndx);
      delta_y += clip4(rcoeff * ndy);
    }
  }

  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) / fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  next[static_cast<std::size_t>(gid) * 2u] = yi_x + alpha * delta_x;
  next[static_cast<std::size_t>(gid) * 2u + 1u] = yi_y + alpha * delta_y;
}

__global__ void mark_valid_graph_entries_rank_major_kernel(const int* neighbors,
                                                           const float* weights,
                                                           unsigned char* flags,
                                                           int n,
                                                           int width) {
  const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = n * width;
  if (id >= total) return;
  const int rank = id / n;
  const int row = id - rank * n;
  const int dense_id = row * width + rank;
  const int nb = neighbors[dense_id];
  const float w = weights[dense_id];
  flags[id] = static_cast<unsigned char>(nb >= 0 && nb < n && nb != row && w > 0.0f);
}

__global__ void pack_coo_graph_from_selected_rank_major_kernel(const int* selected_ids,
                                                               const int* dense_neighbors,
                                                               const float* dense_weights,
                                                               int* coo_heads,
                                                               int* coo_tails,
                                                               float* coo_weights,
                                                               float* coo_epochs_per_sample,
                                                               float max_weight,
                                                               int edge_items,
                                                               int n,
                                                               int width) {
  const int out = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (out >= edge_items) return;
  const int id = selected_ids[out];
  const int rank = id / n;
  const int row = id - rank * n;
  const int dense_id = row * width + rank;
  const int nb = dense_neighbors[dense_id];
  const float w = dense_weights[dense_id];
  coo_heads[out] = row;
  coo_tails[out] = nb;
  coo_weights[out] = w;
  coo_epochs_per_sample[out] = max_weight / fmaxf(w, 1.0e-6f);
}

static __device__ __forceinline__ int
positive_samples_this_epoch_umap_schedule(float period, unsigned int epoch) {
  if (period <= 0.0f || !isfinite(period) || epoch == 0u) return 0;
  const float now = static_cast<float>(epoch);
  const float previous = static_cast<float>(epoch - 1u);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  const int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

static __device__ __forceinline__ int
negative_samples_this_epoch_umap_schedule(float period,
                                                const EmbedParams p,
                                                unsigned int epoch) {
  if (p.negative_sample_rate <= 0 ||
      period <= 0.0f ||
      !isfinite(period) ||
      epoch == 0u) {
    return 0;
  }
  const float now = static_cast<float>(epoch);
  const float previous = static_cast<float>(epoch - 1u);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  if (current_sample <= previous_sample) return 0;

  const float negative_period = period / static_cast<float>(p.negative_sample_rate);
  if (negative_period <= 0.0f || !isfinite(negative_period)) return 0;
  const int current_total = static_cast<int>(
    floorf((now - negative_period) / negative_period)
  );
  int previous_total = 0;
  if (current_sample > 1) {
    const float previous_active_epoch =
      ceilf(static_cast<float>(current_sample - 1) * period);
    previous_total = static_cast<int>(
      floorf((previous_active_epoch - negative_period) / negative_period)
    );
  }
  const int samples = current_total - previous_total;
  return samples > 0 ? samples : 0;
}

__global__ void embed_epoch_coo_atomic_kernel(float* layout,
                                             const int* heads,
                                             const int* tails,
                                             const float* weights,
                                             const float* epochs_per_sample,
                                             EmbedParams p,
                                             unsigned int epoch,
                                             int n_edges_capacity) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= n_edges_capacity) return;

  const int head = heads[gid];
  const int tail = tails[gid];
  if (head < 0 || head >= p.n || tail < 0 || tail >= p.n || head == tail) return;

  const float period = epochs_per_sample[gid];
  const int positive_samples = positive_samples_this_epoch_umap_schedule(period, epoch);
  if (positive_samples <= 0) return;

  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) / fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  const float eps = 1.1920928955078125e-7f;

  for (int sample = 0; sample < positive_samples; ++sample) {
    const std::size_t head_base = static_cast<std::size_t>(head) * 2u;
    const std::size_t tail_base = static_cast<std::size_t>(tail) * 2u;
    const float head_x = layout[head_base];
    const float head_y = layout[head_base + 1u];
    const float tail_x = layout[tail_base];
    const float tail_y = layout[tail_base + 1u];
    const float dx = head_x - tail_x;
    const float dy = head_y - tail_y;
    const float d2 = fmaxf(eps, dx * dx + dy * dy);
    const float w = weights[gid];
    const float coeff = attractive_coeff(d2, w, p);
    const float gx = clip4(coeff * dx) * alpha;
    const float gy = clip4(coeff * dy) * alpha;
    atomicAdd(layout + head_base, gx);
    atomicAdd(layout + head_base + 1u, gy);
    atomicAdd(layout + tail_base, -gx);
    atomicAdd(layout + tail_base + 1u, -gy);
  }

  const int neg_samples = negative_samples_this_epoch_umap_schedule(period, p, epoch);
  for (int s = 0; s < neg_samples; ++s) {
    const unsigned int neg = deterministic_vertex(
      static_cast<unsigned int>(p.n),
      p.seed,
      epoch,
      static_cast<unsigned int>(head),
      static_cast<unsigned int>(tail),
      static_cast<unsigned int>(s)
    );
    if (static_cast<int>(neg) == head || static_cast<int>(neg) == tail) continue;

    const std::size_t head_base = static_cast<std::size_t>(head) * 2u;
    const std::size_t neg_base = static_cast<std::size_t>(neg) * 2u;
    const float head_x = layout[head_base];
    const float head_y = layout[head_base + 1u];
    const float neg_x = layout[neg_base];
    const float neg_y = layout[neg_base + 1u];
    const float ndx = head_x - neg_x;
    const float ndy = head_y - neg_y;
    const float nd2 = fmaxf(eps, ndx * ndx + ndy * ndy);
    const float rcoeff = repulsive_coeff(nd2, p);
    atomicAdd(layout + head_base, clip4(rcoeff * ndx) * alpha);
    atomicAdd(layout + head_base + 1u, clip4(rcoeff * ndy) * alpha);
  }
}

__global__ void embed_epoch_coo_delta_kernel(const float* layout,
                                             float* delta,
                                             const int* heads,
                                             const int* tails,
                                             const float* weights,
                                             const float* epochs_per_sample,
                                             EmbedParams p,
                                             unsigned int epoch,
                                             int n_edges_capacity) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= n_edges_capacity) return;

  const int head = heads[gid];
  const int tail = tails[gid];
  if (head < 0 || head >= p.n || tail < 0 || tail >= p.n || head == tail) return;

  const float period = epochs_per_sample[gid];
  const int positive_samples = positive_samples_this_epoch_umap_schedule(period, epoch);
  if (positive_samples <= 0) return;

  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) / fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  const float eps = 1.1920928955078125e-7f;

  for (int sample = 0; sample < positive_samples; ++sample) {
    const std::size_t head_base = static_cast<std::size_t>(head) * 2u;
    const std::size_t tail_base = static_cast<std::size_t>(tail) * 2u;
    const float head_x = layout[head_base];
    const float head_y = layout[head_base + 1u];
    const float tail_x = layout[tail_base];
    const float tail_y = layout[tail_base + 1u];
    const float dx = head_x - tail_x;
    const float dy = head_y - tail_y;
    const float d2 = fmaxf(eps, dx * dx + dy * dy);
    const float w = weights[gid];
    const float coeff = attractive_coeff(d2, w, p);
    const float gx = clip4(coeff * dx) * alpha;
    const float gy = clip4(coeff * dy) * alpha;
    atomicAdd(delta + head_base, gx);
    atomicAdd(delta + head_base + 1u, gy);
    atomicAdd(delta + tail_base, -gx);
    atomicAdd(delta + tail_base + 1u, -gy);
  }

  const int neg_samples = negative_samples_this_epoch_umap_schedule(period, p, epoch);
  for (int s = 0; s < neg_samples; ++s) {
    const unsigned int neg = deterministic_vertex(
      static_cast<unsigned int>(p.n),
      p.seed,
      epoch,
      static_cast<unsigned int>(head),
      static_cast<unsigned int>(tail),
      static_cast<unsigned int>(s)
    );
    if (static_cast<int>(neg) == head || static_cast<int>(neg) == tail) continue;

    const std::size_t head_base = static_cast<std::size_t>(head) * 2u;
    const std::size_t neg_base = static_cast<std::size_t>(neg) * 2u;
    const float head_x = layout[head_base];
    const float head_y = layout[head_base + 1u];
    const float neg_x = layout[neg_base];
    const float neg_y = layout[neg_base + 1u];
    const float ndx = head_x - neg_x;
    const float ndy = head_y - neg_y;
    const float nd2 = fmaxf(eps, ndx * ndx + ndy * ndy);
    const float rcoeff = repulsive_coeff(nd2, p);
    atomicAdd(delta + head_base, clip4(rcoeff * ndx) * alpha);
    atomicAdd(delta + head_base + 1u, clip4(rcoeff * ndy) * alpha);
  }
}

__device__ void update_umap_row_atomic(float* layout,
                                       const int* neighbors,
                                       const float* weights,
                                       EmbedParams p,
                                       unsigned int epoch,
                                       int width,
                                       int head,
                                       int lane) {
  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) /
     fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  const float eps = 1.1920928955078125e-7f;
  const std::size_t head_base = static_cast<std::size_t>(head) * 2u;
  const float head_x = layout[head_base];
  const float head_y = layout[head_base + 1u];
  float head_dx = 0.0f;
  float head_dy = 0.0f;

  for (int rank = lane; rank < width; rank += 32) {
    const std::size_t edge = static_cast<std::size_t>(head) * width +
      rank;
    const int tail = neighbors[edge];
    const float weight = weights[edge];
    if (tail < 0 || tail >= p.n || tail == head || weight <= 0.0f) {
      continue;
    }
    const float period = p.max_weight / fmaxf(weight, 1.0e-6f);
    const int positive_samples =
      positive_samples_this_epoch_umap_schedule(period, epoch);
    if (positive_samples <= 0) continue;

    const std::size_t tail_base = static_cast<std::size_t>(tail) * 2u;
    const float dx = head_x - layout[tail_base];
    const float dy = head_y - layout[tail_base + 1u];
    const float d2 = fmaxf(eps, dx * dx + dy * dy);
    const float coeff = attractive_coeff(d2, weight, p);
    const float gx = clip4(coeff * dx) * alpha * positive_samples;
    const float gy = clip4(coeff * dy) * alpha * positive_samples;
    head_dx += gx;
    head_dy += gy;
    atomicAdd(layout + tail_base, -gx);
    atomicAdd(layout + tail_base + 1u, -gy);

    const int negatives = negative_samples_this_epoch_umap_schedule(
      period, p, epoch
    );
    for (int sample = 0; sample < negatives; ++sample) {
      const unsigned int negative = deterministic_vertex(
        static_cast<unsigned int>(p.n), p.seed, epoch,
        static_cast<unsigned int>(head),
        static_cast<unsigned int>(tail),
        static_cast<unsigned int>(sample)
      );
      if (static_cast<int>(negative) == head ||
          static_cast<int>(negative) == tail) {
        continue;
      }
      const std::size_t negative_base =
        static_cast<std::size_t>(negative) * 2u;
      const float ndx = head_x - layout[negative_base];
      const float ndy = head_y - layout[negative_base + 1u];
      const float nd2 = fmaxf(eps, ndx * ndx + ndy * ndy);
      const float repulsion = repulsive_coeff(nd2, p);
      head_dx += clip4(repulsion * ndx) * alpha;
      head_dy += clip4(repulsion * ndy) * alpha;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    head_dx += __shfl_down_sync(0xffffffffu, head_dx, offset);
    head_dy += __shfl_down_sync(0xffffffffu, head_dy, offset);
  }
  if (lane == 0) {
    atomicAdd(layout + head_base, head_dx);
    atomicAdd(layout + head_base + 1u, head_dy);
  }
}

__global__ void embed_epoch_row_atomic_kernel(float* layout,
                                              const int* neighbors,
                                              const float* weights,
                                              EmbedParams p,
                                              unsigned int epoch,
                                              int width) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int first_row = static_cast<int>(blockIdx.x) * warps_per_block +
    warp_in_block;
  const int row_stride = static_cast<int>(gridDim.x) * warps_per_block;
  for (int head = first_row; head < p.n; head += row_stride) {
    update_umap_row_atomic(
      layout, neighbors, weights, p, epoch, width, head, lane
    );
  }
}

__global__ void apply_delta_sanitize_layout_kernel(float* layout,
                                                   const float* delta,
                                                   int n,
                                                   float max_abs_coord) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  float x = layout[base] + delta[base];
  float y = layout[base + 1u] + delta[base + 1u];
  if (!isfinite(x)) x = 0.0f;
  if (!isfinite(y)) y = 0.0f;
  x = fminf(max_abs_coord, fmaxf(-max_abs_coord, x));
  y = fminf(max_abs_coord, fmaxf(-max_abs_coord, y));
  layout[base] = x;
  layout[base + 1u] = y;
}

__global__ void umap_sanitize_layout_kernel(float* layout,
                                            int n,
                                            float max_abs_coord) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  float x = layout[base];
  float y = layout[base + 1u];
  if (!isfinite(x)) x = 0.0f;
  if (!isfinite(y)) y = 0.0f;
  x = fminf(max_abs_coord, fmaxf(-max_abs_coord, x));
  y = fminf(max_abs_coord, fmaxf(-max_abs_coord, y));
  layout[base] = x;
  layout[base + 1u] = y;
}

template <typename DistanceT>
int prepare_device_knn_graph(const int* d_indices,
                             const DistanceT* d_distances,
                             int* d_neighbors,
                             float* d_weights,
                             int* d_counts,
                             int n,
                             int k,
                             int width,
                             int offset,
                             int objective,
                             CudaWorkspace* workspace = nullptr) {
  const KnnPrepParams params{n, k, width, offset};
  const int threads = 256;
  const int row_blocks = (n + threads - 1) / threads;
  if (objective != 0) {
    prepare_directed_knn_kernel<<<row_blocks, threads>>>(
      d_indices, d_distances, d_neighbors, d_weights, params
    );
    return check_cuda(cudaGetLastError(), "prepare_directed_knn_kernel launch");
  }

  int* d_direct_neighbors = nullptr;
  float* d_direct_weights = nullptr;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const bool use_workspace = workspace != nullptr;
  if (use_workspace) {
    d_direct_neighbors = workspace->alloc<int>(graph_items, "directed graph temp neighbors");
    d_direct_weights = workspace->alloc<float>(graph_items, "directed graph temp weights");
    if (d_direct_neighbors == nullptr || d_direct_weights == nullptr) return 1;
  } else {
    if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_direct_neighbors), graph_items * sizeof(int)), "cudaMalloc(direct_neighbors)")) {
      return 1;
    }
    if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_direct_weights), graph_items * sizeof(float)), "cudaMalloc(direct_weights)")) {
      cudaFree(d_direct_neighbors);
      return 1;
    }
  }
  auto cleanup_direct = [&]() {
    if (!use_workspace) {
      cudaFree(d_direct_neighbors);
      cudaFree(d_direct_weights);
    }
  };

  prepare_directed_knn_kernel<<<row_blocks, threads>>>(
    d_indices, d_distances, d_direct_neighbors, d_direct_weights, params
  );
  if (check_cuda(cudaGetLastError(), "prepare_directed_knn_kernel(umap) launch")) {
    cleanup_direct();
    return 1;
  }

  const int total_blocks = (static_cast<int>(graph_items) + threads - 1) / threads;
  fill_umap_union_kernel<<<total_blocks, threads>>>(
    d_neighbors, d_weights, d_counts, n, width
  );
  if (check_cuda(cudaGetLastError(), "fill_umap_union_kernel launch")) {
    cleanup_direct();
    return 1;
  }

  const int edge_blocks = (n * k + threads - 1) / threads;
  add_umap_outgoing_kernel<<<edge_blocks, threads>>>(
    d_direct_neighbors, d_direct_weights, d_neighbors, d_weights, d_counts, params
  );
  if (check_cuda(cudaGetLastError(), "add_umap_outgoing_kernel launch")) {
    cleanup_direct();
    return 1;
  }
  add_umap_incoming_kernel<<<edge_blocks, threads>>>(
    d_direct_neighbors, d_direct_weights, d_neighbors, d_weights, d_counts, params
  );
  if (check_cuda(cudaGetLastError(), "add_umap_incoming_kernel launch")) {
    cleanup_direct();
    return 1;
  }

  cleanup_direct();
  return 0;
}

int normalize_device_init(float* d_values,
                          double* d_partial,
                          double* d_stats,
                          int n,
                          int stat_blocks,
                          int threads) {
  const std::size_t shared = static_cast<std::size_t>(threads) * 5u * sizeof(double);
  reduce_init_stats_kernel<<<stat_blocks, threads, shared>>>(d_values, d_partial, n);
  if (check_cuda(cudaGetLastError(), "reduce_init_stats_kernel launch")) return 1;
  finalize_init_stats_kernel<<<1, threads, shared>>>(d_partial, d_stats, stat_blocks, n);
  if (check_cuda(cudaGetLastError(), "finalize_init_stats_kernel launch")) return 1;
  const int blocks = (n + threads - 1) / threads;
  normalize_init_kernel<<<blocks, threads>>>(d_values, d_stats, n);
  return check_cuda(cudaGetLastError(), "normalize_init_kernel launch");
}

int scale_tsne_pca_device_init(float* d_values,
                               double* d_partial,
                               double* d_stats,
                               int n,
                               int stat_blocks,
                               int threads) {
  const std::size_t shared = static_cast<std::size_t>(threads) * 5u * sizeof(double);
  reduce_init_stats_kernel<<<stat_blocks, threads, shared>>>(d_values, d_partial, n);
  if (check_cuda(cudaGetLastError(), "reduce_init_stats_kernel(tsne pca init) launch")) return 1;
  finalize_tsne_pca_init_stats_kernel<<<1, threads, shared>>>(
    d_partial, d_stats, stat_blocks, n
  );
  if (check_cuda(cudaGetLastError(), "finalize_tsne_pca_init_stats_kernel launch")) return 1;
  const int blocks = (n + threads - 1) / threads;
  scale_tsne_pca_init_kernel<<<blocks, threads>>>(d_values, d_stats, n);
  return check_cuda(cudaGetLastError(), "scale_tsne_pca_init_kernel launch");
}

__global__ void convert_pca_input_kernel(const double* input,
                                         float* output,
                                         std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (index < count) output[index] = static_cast<float>(input[index]);
}

__global__ void pca_row_major_to_column_major_kernel(
    const float* input,
    float* output,
    int n,
    int p) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
    blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(n) * p;
  if (index >= count) return;
  const int row = static_cast<int>(index / p);
  const int column = static_cast<int>(index % p);
  output[static_cast<std::size_t>(column) * n + row] = input[index];
}

__global__ void preprocess_pca_columns_kernel(float* values,
                                              float* centers,
                                              float* scales,
                                              unsigned int* invalid,
                                              int n,
                                              int p,
                                              int apply_center,
                                              int apply_scale) {
  const int column = blockIdx.x;
  const int tid = threadIdx.x;
  if (column >= p) return;
  extern __shared__ float scratch[];
  const std::size_t offset = static_cast<std::size_t>(column) * n;
  float sum = 0.0f;
  for (int row = tid; row < n; row += blockDim.x) {
    float value = values[offset + row];
    if (!isfinite(value)) {
      atomicExch(invalid, 1u);
      value = 0.0f;
    }
    sum += value;
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    const float mean = scratch[0] / static_cast<float>(n);
    scratch[0] = apply_center != 0 ? mean : 0.0f;
  }
  __syncthreads();
  const float center = scratch[0];
  float sum_squares = 0.0f;
  for (int row = tid; row < n; row += blockDim.x) {
    const float centered = values[offset + row] - center;
    sum_squares += centered * centered;
  }
  scratch[tid] = sum_squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    float column_scale = 1.0f;
    if (apply_scale != 0 && scratch[0] > 0.0f) {
      column_scale = sqrtf(
        scratch[0] / static_cast<float>(max(1, n - 1))
      );
    }
    centers[column] = center;
    scales[column] = isfinite(column_scale) && column_scale > 0.0f ?
      column_scale : 1.0f;
    scratch[0] = scales[column];
  }
  __syncthreads();
  const float inverse_scale = 1.0f / scratch[0];
  for (int row = tid; row < n; row += blockDim.x) {
    values[offset + row] =
      (values[offset + row] - center) * inverse_scale;
  }
}

constexpr std::size_t kPcaUploadChunkItems = 1u << 20;

std::size_t cuda_pca_workspace_bytes(int n,
                                     int p,
                                     int n_components,
                                     bool input_is_double) {
  const std::size_t input_items = static_cast<std::size_t>(n) * p;
  const std::size_t component_items =
    static_cast<std::size_t>(n_components) * p;
  std::size_t bytes = align_bytes(input_items * sizeof(float));
  bytes += align_bytes(component_items * sizeof(float));
  bytes += 3u * align_bytes(
    static_cast<std::size_t>(n_components) * sizeof(float)
  );
  bytes += 2u * align_bytes(static_cast<std::size_t>(p) * sizeof(float));
  bytes += align_bytes(sizeof(unsigned int));
  if (input_is_double) {
    const std::size_t upload_items =
      std::min(input_items, kPcaUploadChunkItems);
    bytes += align_bytes(upload_items * sizeof(double));
  }
  return bytes;
}

int select_cuda_pca_decomposition(int requested_method,
                                  int n,
                                  int p,
                                  int n_components,
                                  int oversample,
                                  int power) {
  if (requested_method != 0) return requested_method;
  const int rank_limit = std::min(n - 1, p);
  const int sketch = std::min(
    rank_limit, n_components + std::max(0, oversample)
  );
  const std::size_t matrix_items = static_cast<std::size_t>(n) * p;
  const std::size_t pass_factor =
    static_cast<std::size_t>(2 * std::max(0, power) + 2);
  const std::size_t crossover = 12u * pass_factor * sketch;
  const bool truncated = sketch < rank_limit;
  const bool enough_work = matrix_items >= 250000u;
  return truncated && enough_work &&
    static_cast<std::size_t>(p) >= crossover ? 1 : 2;
}

#if defined(FASTEMBEDR_HAS_RAFT) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
template <typename HostT>
int cuda_pca_scores_to_device(const HostT* values,
                              int n,
                              int p,
                              int n_components,
                              float* d_scores_out,
                              CudaWorkspace* workspace = nullptr,
                              bool center_input = true,
                              bool scale_input = false,
                              float* host_components_out = nullptr,
                              float* host_singular_out = nullptr,
                              float* host_center_out = nullptr,
                              float* host_scale_out = nullptr,
                              int requested_method = 0,
                              unsigned int seed = 4u,
                              int requested_oversample = 16,
                              int requested_power = 2,
                              int* selected_method_out = nullptr,
                              const float* device_row_major = nullptr) {
  if ((values == nullptr && device_row_major == nullptr) ||
      d_scores_out == nullptr) {
    set_embedding_error("null pointer in CUDA device PCA initialization");
    return 1;
  }
  if (n < 2 || p < 2 || n_components < 1 || n_components > p) {
    set_embedding_error("invalid dimensions in CUDA device PCA initialization");
    return 1;
  }

  const std::size_t input_items = static_cast<std::size_t>(n) * p;
  const std::size_t component_items =
    static_cast<std::size_t>(n_components) * p;
  const bool input_is_double = std::is_same<HostT, double>::value;
  static_assert(
    std::is_same<HostT, float>::value ||
      std::is_same<HostT, double>::value,
    "CUDA PCA input must be float or double"
  );

  float* d_input = nullptr;
  float* d_components = nullptr;
  float* d_explained = nullptr;
  float* d_explained_ratio = nullptr;
  float* d_singular = nullptr;
  float* d_centers = nullptr;
  float* d_scales = nullptr;
  unsigned int* d_invalid = nullptr;
  double* d_upload = nullptr;
  const bool use_workspace = workspace != nullptr;
  auto cleanup = [&]() {
    if (!use_workspace) {
      if (d_input != nullptr) cudaFree(d_input);
      if (d_components != nullptr) cudaFree(d_components);
      if (d_explained != nullptr) cudaFree(d_explained);
      if (d_explained_ratio != nullptr) cudaFree(d_explained_ratio);
      if (d_singular != nullptr) cudaFree(d_singular);
      if (d_centers != nullptr) cudaFree(d_centers);
      if (d_scales != nullptr) cudaFree(d_scales);
      if (d_invalid != nullptr) cudaFree(d_invalid);
      if (d_upload != nullptr) cudaFree(d_upload);
    }
  };

  try {
    const std::size_t required_bytes = cuda_pca_workspace_bytes(
      n, p, n_components, input_is_double
    );
    if (!use_workspace &&
        check_embedding_memory_available(
          required_bytes, "CUDA device PCA initialization allocation preflight"
        )) {
      cleanup();
      return 1;
    }
    if (use_workspace) {
      d_input = workspace->alloc<float>(input_items, "cuda pca input");
      d_components = workspace->alloc<float>(
        component_items, "cuda pca components"
      );
      d_explained = workspace->alloc<float>(
        n_components, "cuda pca explained"
      );
      d_explained_ratio = workspace->alloc<float>(
        n_components, "cuda pca explained_ratio"
      );
      d_singular = workspace->alloc<float>(
        n_components, "cuda pca singular"
      );
      d_centers = workspace->alloc<float>(p, "cuda pca centers");
      d_scales = workspace->alloc<float>(p, "cuda pca scales");
      d_invalid = workspace->alloc<unsigned int>(
        1u, "cuda pca invalid flag"
      );
      if (input_is_double) {
        d_upload = workspace->alloc<double>(
          std::min(input_items, kPcaUploadChunkItems),
          "cuda pca double upload"
        );
      }
      if (d_input == nullptr || d_components == nullptr ||
          d_explained == nullptr || d_explained_ratio == nullptr ||
          d_singular == nullptr || d_centers == nullptr ||
          d_scales == nullptr || d_invalid == nullptr ||
          (input_is_double && d_upload == nullptr)) {
        cleanup();
        return 1;
      }
    } else {
      if (check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_input),
              input_items * sizeof(float)
            ),
            "cudaMalloc(cuda pca input)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_components),
              component_items * sizeof(float)
            ),
            "cudaMalloc(cuda pca components)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_explained),
              n_components * sizeof(float)
            ),
            "cudaMalloc(cuda pca explained)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_explained_ratio),
              n_components * sizeof(float)
            ),
            "cudaMalloc(cuda pca explained_ratio)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_singular),
              n_components * sizeof(float)
            ),
            "cudaMalloc(cuda pca singular)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_centers),
              static_cast<std::size_t>(p) * sizeof(float)
            ),
            "cudaMalloc(cuda pca centers)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_scales),
              static_cast<std::size_t>(p) * sizeof(float)
            ),
            "cudaMalloc(cuda pca scales)"
          ) ||
          check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_invalid),
              sizeof(unsigned int)
            ),
            "cudaMalloc(cuda pca invalid flag)"
          )) {
        cleanup();
        return 1;
      }
      if (input_is_double && check_cuda(
            cudaMalloc(
              reinterpret_cast<void**>(&d_upload),
              std::min(input_items, kPcaUploadChunkItems) * sizeof(double)
            ),
            "cudaMalloc(cuda pca double upload)"
          )) {
        cleanup();
        return 1;
      }
    }

    if (device_row_major != nullptr) {
      const int threads = 256;
      const int blocks = static_cast<int>(
        (input_items + threads - 1u) / threads
      );
      pca_row_major_to_column_major_kernel<<<blocks, threads>>>(
        device_row_major, d_input, n, p
      );
      if (check_cuda(
            cudaGetLastError(),
            "pca_row_major_to_column_major_kernel launch"
          )) {
        cleanup();
        return 1;
      }
    } else if constexpr (std::is_same<HostT, float>::value) {
      if (check_cuda(
            cudaMemcpy(
              d_input, values, input_items * sizeof(float),
              cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(cuda pca float input H2D)"
          )) {
        cleanup();
        return 1;
      }
    } else {
      const int threads = 256;
      for (std::size_t offset = 0u; offset < input_items;
           offset += kPcaUploadChunkItems) {
        const std::size_t count = std::min(
          input_items - offset, kPcaUploadChunkItems
        );
        if (check_cuda(
              cudaMemcpy(
                d_upload, values + offset, count * sizeof(double),
                cudaMemcpyHostToDevice
              ),
              "cudaMemcpy(cuda pca double input H2D)"
            )) {
          cleanup();
          return 1;
        }
        const int blocks = static_cast<int>(
          (count + threads - 1u) / threads
        );
        convert_pca_input_kernel<<<blocks, threads>>>(
          d_upload, d_input + offset, count
        );
        if (check_cuda(
              cudaGetLastError(), "convert_pca_input_kernel launch"
            )) {
          cleanup();
          return 1;
        }
      }
    }

    if (check_cuda(
          cudaMemset(d_invalid, 0, sizeof(unsigned int)),
          "cudaMemset(cuda pca invalid flag)"
        )) {
      cleanup();
      return 1;
    }
    const int preprocess_threads = 256;
    preprocess_pca_columns_kernel<<<
      p, preprocess_threads, preprocess_threads * sizeof(float)
    >>>(
      d_input, d_centers, d_scales, d_invalid, n, p,
      center_input ? 1 : 0, scale_input ? 1 : 0
    );
    if (check_cuda(
          cudaGetLastError(), "preprocess_pca_columns_kernel launch"
        )) {
      cleanup();
      return 1;
    }
    unsigned int invalid = 0u;
    if (check_cuda(
          cudaMemcpy(
            &invalid, d_invalid, sizeof(unsigned int),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(cuda pca invalid flag D2H)"
        )) {
      cleanup();
      return 1;
    }
    if (invalid != 0u) {
      cleanup();
      set_embedding_error("PCA input must contain only finite values");
      return 1;
    }
    if (host_center_out != nullptr && check_cuda(
          cudaMemcpy(
            host_center_out, d_centers,
            static_cast<std::size_t>(p) * sizeof(float),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(cuda pca centers D2H)"
        )) {
      cleanup();
      return 1;
    }
    if (host_scale_out != nullptr && check_cuda(
          cudaMemcpy(
            host_scale_out, d_scales,
            static_cast<std::size_t>(p) * sizeof(float),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(cuda pca scales D2H)"
        )) {
      cleanup();
      return 1;
    }

    const int rank_limit = std::min(n - 1, p);
    int selected_method = select_cuda_pca_decomposition(
      requested_method, n, p, n_components,
      requested_oversample, requested_power
    );
#ifndef FASTEMBEDR_HAS_RAFT
    if (selected_method == 2) {
      if (requested_method == 2) {
        cleanup();
        set_embedding_error(
          "RAPIDS RAFT TSVD was requested but is not available in this build"
        );
        return 1;
      }
      selected_method = 1;
    }
#endif
    if (selected_method != 1 && selected_method != 2) {
      cleanup();
      set_embedding_error("invalid CUDA PCA decomposition selector");
      return 1;
    }
    if (selected_method_out != nullptr) {
      *selected_method_out = selected_method;
    }

    if (selected_method == 1) {
      const int oversample = std::min(
        rank_limit - n_components,
        std::max(0, requested_oversample)
      );
      const int power = std::max(0, requested_power);
      fastembedr_pca::RsvdWorkspace rsvd(
        n, p, n_components, oversample, power, nullptr
      );
      rsvd.solve(
        d_input, static_cast<unsigned long long>(seed),
        d_scores_out, d_components, d_singular
      );
      rsvd.synchronize_and_validate();
      const int score_items = n * n_components;
      scale_rsvd_scores_kernel<<<(score_items + 255) / 256, 256>>>(
        d_scores_out, d_singular, n, n_components
      );
      if (check_cuda(
            cudaGetLastError(),
            "scale_rsvd_scores_kernel launch"
          )) {
        cleanup();
        return 1;
      }
    } else {
#ifdef FASTEMBEDR_HAS_RAFT
      static thread_local raft::handle_t handle;
      auto input_view = raft::make_device_matrix_view<
        float, std::size_t, raft::col_major
      >(d_input, static_cast<std::size_t>(n), static_cast<std::size_t>(p));
      auto scores_view = raft::make_device_matrix_view<
        float, std::size_t, raft::col_major
      >(
        d_scores_out, static_cast<std::size_t>(n),
        static_cast<std::size_t>(n_components)
      );
      auto components_view = raft::make_device_matrix_view<
        float, std::size_t, raft::col_major
      >(
        d_components, static_cast<std::size_t>(n_components),
        static_cast<std::size_t>(p)
      );
      auto explained_view = raft::make_device_vector_view<
        float, std::size_t
      >(d_explained, static_cast<std::size_t>(n_components));
      auto explained_ratio_view = raft::make_device_vector_view<
        float, std::size_t
      >(d_explained_ratio, static_cast<std::size_t>(n_components));
      auto singular_view = raft::make_device_vector_view<
        float, std::size_t
      >(d_singular, static_cast<std::size_t>(n_components));
      raft::linalg::paramsTSVD params;
      params.algorithm = raft::linalg::solver::COV_EIG_DQ;
      params.tol = 0.0f;
      params.n_iterations = 15;
      raft::linalg::tsvd_fit_transform(
        handle,
        params,
        input_view,
        scores_view,
        components_view,
        explained_view,
        explained_ratio_view,
        singular_view,
        true
      );
#else
      cleanup();
      set_embedding_error(
        "RAPIDS RAFT TSVD is not available in this build"
      );
      return 1;
#endif
    }

    if (check_cuda(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize(cuda pca device init)"
        )) {
      cleanup();
      return 1;
    }
    if (host_components_out != nullptr &&
        check_cuda(
          cudaMemcpy(
            host_components_out,
            d_components,
            component_items * sizeof(float),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(cuda pca components D2H)"
        )) {
      cleanup();
      return 1;
    }
    if (host_singular_out != nullptr &&
        check_cuda(
          cudaMemcpy(
            host_singular_out,
            d_singular,
            static_cast<std::size_t>(n_components) * sizeof(float),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(cuda pca singular values D2H)"
        )) {
      cleanup();
      return 1;
    }
    cleanup();
    return 0;
  } catch (const std::exception& e) {
    cleanup();
    set_embedding_error(
      std::string("CUDA device PCA initialization failed: ") + e.what()
    );
    return 1;
  } catch (...) {
    cleanup();
    set_embedding_error(
      "CUDA device PCA initialization failed with an unknown error."
    );
    return 1;
  }
}
#if defined(FASTEMBEDR_HAS_RAFT) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

} // namespace

extern "C" int fastembedr_cuda_transform_tsne_from_host_knn(
    const int* indices,
    const float* distances,
    int index_offset,
    const float* reference_layout,
    const float* initial_layout,
    int n_reference,
    int n_query,
    int k,
    float perplexity,
    int n_iter,
    int early_exaggeration_iter,
    float learning_rate,
    float early_exaggeration,
    float exaggeration,
    float initial_momentum,
    float final_momentum,
    float max_grad_norm,
    float max_step_norm,
    int n_negatives,
    int exact_repulsion_threshold,
    unsigned int seed,
    float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr ||
      reference_layout == nullptr || initial_layout == nullptr ||
      out == nullptr || n_reference < 1 || n_query < 1 || k < 1 ||
      k > kCudaProjectionMaxK || perplexity <= 0.0f) {
    set_embedding_error("invalid CUDA t-SNE transform inputs");
    return 1;
  }
  const std::size_t graph_items = static_cast<std::size_t>(n_query) * k;
  const std::size_t layout_items = static_cast<std::size_t>(n_reference) * 2u;
  const std::size_t query_items = static_cast<std::size_t>(n_query);
  const std::size_t required_bytes =
    align_bytes(graph_items * sizeof(int)) +
    2u * align_bytes(graph_items * sizeof(float)) +
    align_bytes(layout_items * sizeof(float)) +
    3u * align_bytes(query_items * sizeof(float2));
  if (check_embedding_memory_available(
        required_bytes, "CUDA t-SNE transform workspace")) return 1;
  CudaWorkspace workspace;
  if (workspace.init(required_bytes, "t-SNE transform")) return 1;
  int* d_indices = workspace.alloc<int>(graph_items, "transform indices");
  float* d_distances = workspace.alloc<float>(graph_items, "transform distances");
  float* d_probabilities = workspace.alloc<float>(graph_items, "transform probabilities");
  float* d_reference = workspace.alloc<float>(layout_items, "transform reference");
  float2* d_current = workspace.alloc<float2>(query_items, "transform current");
  float2* d_gains = workspace.alloc<float2>(query_items, "transform gains");
  float2* d_updates = workspace.alloc<float2>(query_items, "transform updates");
  if (d_indices == nullptr || d_distances == nullptr ||
      d_probabilities == nullptr || d_reference == nullptr ||
      d_current == nullptr || d_gains == nullptr || d_updates == nullptr) {
    return 1;
  }
  if (check_cuda(cudaMemcpy(
        d_indices, indices, graph_items * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy(transform indices H2D)") ||
      check_cuda(cudaMemcpy(
        d_distances, distances, graph_items * sizeof(float),
        cudaMemcpyHostToDevice), "cudaMemcpy(transform distances H2D)") ||
      check_cuda(cudaMemcpy(
        d_reference, reference_layout, layout_items * sizeof(float),
        cudaMemcpyHostToDevice), "cudaMemcpy(transform reference H2D)") ||
      check_cuda(cudaMemcpy(
        d_current, initial_layout, query_items * sizeof(float2),
        cudaMemcpyHostToDevice), "cudaMemcpy(transform initial H2D)")) {
    return 1;
  }
  constexpr int threads = 128;
  const int blocks = (n_query + threads - 1) / threads;
  landmark_tsne_probabilities_kernel<<<blocks, threads>>>(
    d_distances, d_probabilities, n_query, k, perplexity
  );
  landmark_tsne_state_init_kernel<<<blocks, threads>>>(
    d_gains, d_updates, n_query
  );
  if (check_cuda(cudaGetLastError(), "CUDA t-SNE transform setup")) return 1;
  const int total_iter = n_iter + early_exaggeration_iter;
  for (int iter = 0; iter < total_iter; ++iter) {
    const bool early = iter < early_exaggeration_iter;
    LandmarkTsneParams params{
      n_reference, n_query, k,
      std::max(1, std::min(n_negatives, n_reference)), seed,
      (n_reference <= exact_repulsion_threshold ||
       n_negatives >= n_reference) ? 1 : 0,
      learning_rate, early ? early_exaggeration : exaggeration,
      early ? initial_momentum : final_momentum,
      max_grad_norm, max_step_norm
    };
    landmark_tsne_epoch_kernel<<<blocks, threads>>>(
      d_reference, d_indices, d_probabilities, d_current,
      d_gains, d_updates, params, static_cast<unsigned int>(iter),
      index_offset
    );
  }
  if (check_cuda(cudaGetLastError(), "CUDA t-SNE transform kernels") ||
      check_cuda(cudaMemcpy(
        out, d_current, query_items * sizeof(float2), cudaMemcpyDeviceToHost),
        "cudaMemcpy(transform layout D2H)")) return 1;
  return 0;
}

extern "C" int fastembedr_cuda_landmark_tsne_from_device_knn(
    const int* device_indices,
    const float* device_distances,
    int index_offset,
    const float* reference_data,
    const float* query_data,
    const float* reference_layout,
    int n_reference,
    int n_query,
    int n_features,
    int k,
    float perplexity,
    int n_iter,
    int early_exaggeration_iter,
    float learning_rate,
    float early_exaggeration,
    float exaggeration,
    float initial_momentum,
    float final_momentum,
    float max_grad_norm,
    float max_step_norm,
    int n_negatives,
    int exact_repulsion_threshold,
    unsigned int seed,
    int affine_neighbors,
    float affine_ridge,
    float max_extrapolation,
    float* out) {
  embedding_last_error.clear();
  if (device_indices == nullptr || device_distances == nullptr ||
      reference_data == nullptr || query_data == nullptr ||
      reference_layout == nullptr || out == nullptr) {
    set_embedding_error("null CUDA landmark t-SNE pointer");
    return 1;
  }
  if (n_reference < 1 || n_query < 1 || n_features < 1 || k < 1 ||
      k > kCudaProjectionMaxK || perplexity <= 0.0f ||
      n_iter < 0 || early_exaggeration_iter < 0 ||
      n_iter + early_exaggeration_iter < 0) {
    set_embedding_error("invalid CUDA landmark t-SNE dimensions or parameters");
    return 1;
  }
  n_negatives = std::max(1, std::min(n_negatives, n_reference));
  affine_neighbors = std::max(3, std::min(affine_neighbors, std::min(k, kCudaAffineMaxNeighbors)));
  const std::size_t reference_items = static_cast<std::size_t>(n_reference) * n_features;
  const std::size_t query_items = static_cast<std::size_t>(n_query) * n_features;
  const std::size_t layout_items = static_cast<std::size_t>(n_reference) * 2u;
  const std::size_t graph_items = static_cast<std::size_t>(n_query) * k;
  const std::size_t query_layout_items = static_cast<std::size_t>(n_query);
  const std::size_t required_bytes =
    align_bytes(reference_items * sizeof(float)) +
    align_bytes(query_items * sizeof(float)) +
    align_bytes(layout_items * sizeof(float)) +
    align_bytes(graph_items * sizeof(float)) +
    3u * align_bytes(query_layout_items * sizeof(float2));
  if (check_embedding_memory_available(required_bytes, "CUDA landmark t-SNE workspace")) return 1;
  CudaWorkspace workspace;
  if (workspace.init(required_bytes, "landmark t-SNE")) return 1;
  float* d_reference_data = workspace.alloc<float>(reference_items, "landmark t-SNE reference data");
  float* d_query_data = workspace.alloc<float>(query_items, "landmark t-SNE query data");
  float* d_reference_layout = workspace.alloc<float>(layout_items, "landmark t-SNE reference layout");
  float* d_probabilities = workspace.alloc<float>(graph_items, "landmark t-SNE probabilities");
  float2* d_current = workspace.alloc<float2>(query_layout_items, "landmark t-SNE current");
  float2* d_gains = workspace.alloc<float2>(query_layout_items, "landmark t-SNE gains");
  float2* d_updates = workspace.alloc<float2>(query_layout_items, "landmark t-SNE updates");
  if (d_reference_data == nullptr || d_query_data == nullptr || d_reference_layout == nullptr ||
      d_probabilities == nullptr || d_current == nullptr || d_gains == nullptr || d_updates == nullptr) return 1;
  if (check_cuda(cudaMemcpy(d_reference_data, reference_data, reference_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark t-SNE reference data H2D)") ||
      check_cuda(cudaMemcpy(d_query_data, query_data, query_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark t-SNE query data H2D)") ||
      check_cuda(cudaMemcpy(d_reference_layout, reference_layout, layout_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark t-SNE reference layout H2D)")) return 1;

  constexpr int threads = 128;
  const int blocks = (n_query + threads - 1) / threads;
  landmark_affine_projection_kernel<<<blocks, threads>>>(
    d_reference_data, d_query_data, d_reference_layout,
    device_indices, device_distances, d_current,
    n_reference, n_query, n_features, k, index_offset,
    affine_neighbors, affine_ridge, max_extrapolation
  );
  landmark_tsne_probabilities_kernel<<<blocks, threads>>>(
    device_distances, d_probabilities, n_query, k, perplexity
  );
  landmark_tsne_state_init_kernel<<<blocks, threads>>>(d_gains, d_updates, n_query);
  if (check_cuda(cudaGetLastError(), "CUDA landmark t-SNE setup kernels")) return 1;

  const int total_iter = n_iter + early_exaggeration_iter;
  for (int iter = 0; iter < total_iter; ++iter) {
    const bool early = iter < early_exaggeration_iter;
    LandmarkTsneParams params{
      n_reference,
      n_query,
      k,
      n_negatives,
      seed,
      (n_reference <= exact_repulsion_threshold || n_negatives >= n_reference) ? 1 : 0,
      learning_rate,
      early ? early_exaggeration : exaggeration,
      early ? initial_momentum : final_momentum,
      max_grad_norm,
      max_step_norm
    };
    landmark_tsne_epoch_kernel<<<blocks, threads>>>(
      d_reference_layout, device_indices, d_probabilities,
      d_current, d_gains, d_updates, params,
      static_cast<unsigned int>(iter), index_offset
    );
  }
  if (check_cuda(cudaGetLastError(), "CUDA landmark t-SNE transform kernels") ||
      check_cuda(cudaMemcpy(out, d_current, query_layout_items * sizeof(float2), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(landmark t-SNE layout D2H)")) return 1;
  return 0;
}

extern "C" int fastembedr_cuda_landmark_umap_from_device_knn(
    const int* device_indices,
    const float* device_distances,
    int index_offset,
    const float* reference_data,
    const float* query_data,
    const float* reference_layout,
    const int* landmark_rows,
    const int* query_rows,
    int n_total,
    int n_reference,
    int n_query,
    int n_features,
    int k,
    int n_epochs,
    int negative_sample_rate,
    float learning_rate,
    float a,
    float b,
    float repulsion_strength,
    unsigned int seed,
    int affine_neighbors,
    float affine_ridge,
    float max_extrapolation,
    float* out) {
  embedding_last_error.clear();
  if (device_indices == nullptr || device_distances == nullptr ||
      reference_data == nullptr || query_data == nullptr || reference_layout == nullptr ||
      landmark_rows == nullptr || query_rows == nullptr || out == nullptr) {
    set_embedding_error("null CUDA landmark UMAP pointer");
    return 1;
  }
  if (n_total < n_reference + n_query || n_reference < 1 || n_query < 1 ||
      n_features < 1 || k < 1 || k > kCudaProjectionMaxK || n_epochs < 0) {
    set_embedding_error("invalid CUDA landmark UMAP dimensions or parameters");
    return 1;
  }
  affine_neighbors = std::max(3, std::min(affine_neighbors, std::min(k, kCudaAffineMaxNeighbors)));
  const std::size_t reference_items = static_cast<std::size_t>(n_reference) * n_features;
  const std::size_t query_items = static_cast<std::size_t>(n_query) * n_features;
  const std::size_t reference_layout_items = static_cast<std::size_t>(n_reference) * 2u;
  const std::size_t graph_items = static_cast<std::size_t>(n_query) * k;
  const std::size_t required_bytes =
    align_bytes(reference_items * sizeof(float)) +
    align_bytes(query_items * sizeof(float)) +
    align_bytes(reference_layout_items * sizeof(float)) +
    align_bytes(static_cast<std::size_t>(n_reference) * sizeof(int)) +
    align_bytes(static_cast<std::size_t>(n_query) * sizeof(int)) +
    align_bytes(static_cast<std::size_t>(n_query) * sizeof(float2)) +
    align_bytes(static_cast<std::size_t>(n_total) * 2u * sizeof(float)) +
    2u * align_bytes(graph_items * sizeof(float)) +
    align_bytes(sizeof(float)) + align_bytes(sizeof(unsigned int));
  if (check_embedding_memory_available(required_bytes, "CUDA landmark UMAP workspace")) return 1;
  CudaWorkspace workspace;
  if (workspace.init(required_bytes, "landmark UMAP")) return 1;
  float* d_reference_data = workspace.alloc<float>(reference_items, "landmark UMAP reference data");
  float* d_query_data = workspace.alloc<float>(query_items, "landmark UMAP query data");
  float* d_reference_layout = workspace.alloc<float>(reference_layout_items, "landmark UMAP reference layout");
  int* d_landmark_rows = workspace.alloc<int>(n_reference, "landmark UMAP landmark rows");
  int* d_query_rows = workspace.alloc<int>(n_query, "landmark UMAP query rows");
  float2* d_query_layout = workspace.alloc<float2>(n_query, "landmark UMAP query layout");
  float* d_layout = workspace.alloc<float>(static_cast<std::size_t>(n_total) * 2u, "landmark UMAP full layout");
  float* d_weights = workspace.alloc<float>(graph_items, "landmark UMAP weights");
  float* d_epochs_per_sample = workspace.alloc<float>(graph_items, "landmark UMAP epoch schedule");
  float* d_distance_sum = workspace.alloc<float>(1u, "landmark UMAP distance sum");
  unsigned int* d_distance_count = workspace.alloc<unsigned int>(1u, "landmark UMAP distance count");
  if (d_reference_data == nullptr || d_query_data == nullptr || d_reference_layout == nullptr ||
      d_landmark_rows == nullptr || d_query_rows == nullptr || d_query_layout == nullptr ||
      d_layout == nullptr || d_weights == nullptr || d_epochs_per_sample == nullptr ||
      d_distance_sum == nullptr || d_distance_count == nullptr) return 1;
  if (check_cuda(cudaMemcpy(d_reference_data, reference_data, reference_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark UMAP reference data H2D)") ||
      check_cuda(cudaMemcpy(d_query_data, query_data, query_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark UMAP query data H2D)") ||
      check_cuda(cudaMemcpy(d_reference_layout, reference_layout, reference_layout_items * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark UMAP reference layout H2D)") ||
      check_cuda(cudaMemcpy(d_landmark_rows, landmark_rows, static_cast<std::size_t>(n_reference) * sizeof(int), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark UMAP landmark rows H2D)") ||
      check_cuda(cudaMemcpy(d_query_rows, query_rows, static_cast<std::size_t>(n_query) * sizeof(int), cudaMemcpyHostToDevice),
                 "cudaMemcpy(landmark UMAP query rows H2D)") ||
      check_cuda(cudaMemset(d_layout, 0, static_cast<std::size_t>(n_total) * 2u * sizeof(float)),
                 "cudaMemset(landmark UMAP layout)") ||
      check_cuda(cudaMemset(d_distance_sum, 0, sizeof(float)),
                 "cudaMemset(landmark UMAP distance sum)") ||
      check_cuda(cudaMemset(d_distance_count, 0, sizeof(unsigned int)),
                 "cudaMemset(landmark UMAP distance count)")) return 1;

  constexpr int threads = 128;
  const int query_blocks = (n_query + threads - 1) / threads;
  landmark_affine_projection_kernel<<<query_blocks, threads>>>(
    d_reference_data, d_query_data, d_reference_layout,
    device_indices, device_distances, d_query_layout,
    n_reference, n_query, n_features, k, index_offset,
    affine_neighbors, affine_ridge, max_extrapolation
  );
  const int assemble_items = std::max(n_reference, n_query);
  landmark_umap_assemble_kernel<<<(assemble_items + threads - 1) / threads, threads>>>(
    d_reference_layout, d_query_layout, d_landmark_rows, d_query_rows,
    n_reference, n_query, d_layout
  );
  const int graph_count = n_query * k;
  landmark_distance_sum_kernel<<<(graph_count + threads - 1) / threads, threads>>>(
    device_distances, graph_count, d_distance_sum, d_distance_count
  );
  if (check_cuda(cudaGetLastError(), "CUDA landmark UMAP setup kernels")) return 1;
  float distance_sum = 0.0f;
  unsigned int distance_count = 0u;
  if (check_cuda(cudaMemcpy(&distance_sum, d_distance_sum, sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(landmark UMAP distance sum D2H)") ||
      check_cuda(cudaMemcpy(&distance_count, d_distance_count, sizeof(unsigned int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(landmark UMAP distance count D2H)")) return 1;
  const float global_mean = distance_count > 0u ? distance_sum / distance_count : 1.0f;
  landmark_umap_prepare_kernel<<<query_blocks, threads>>>(
    device_distances, d_weights, d_epochs_per_sample, n_query, k, global_mean
  );
  EmbedParams params{
    n_total,
    k,
    n_epochs,
    negative_sample_rate,
    0,
    seed,
    learning_rate,
    a,
    b,
    1.0f,
    repulsion_strength
  };
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    landmark_umap_refine_epoch_kernel<<<query_blocks, threads>>>(
      d_layout, device_indices, d_weights, d_epochs_per_sample,
      d_landmark_rows, d_query_rows, n_reference, n_query, k,
      index_offset, params, static_cast<unsigned int>(epoch)
    );
  }
  if (check_cuda(cudaGetLastError(), "CUDA landmark UMAP refinement kernels") ||
      check_cuda(cudaMemcpy(out, d_layout, static_cast<std::size_t>(n_total) * 2u * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(landmark UMAP layout D2H)")) return 1;
  return 0;
}

extern "C" const char* fastembedr_cuda_embedding_last_error() {
  return embedding_last_error.c_str();
}

extern "C" const char* fastembedr_cuda_workspace_allocator() {
#if CUDART_VERSION >= 11020
  return "cuda_async_default_mempool";
#else
  return "cuda_legacy_workspace";
#endif
}

extern "C" int fastembedr_cuda_last_tsne_graph_capture() {
  return last_tsne_cuda_graph_capture ? 1 : 0;
}

extern "C" int fastembedr_cuda_last_umap_graph_capture() {
  return last_umap_cuda_graph_capture ? 1 : 0;
}

extern "C" bool fastembedr_cuda_available() {
  int count = 0;
  cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) {
    set_embedding_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(code));
    return false;
  }
  return count > 0;
}

extern "C" int fastembedr_cuda_finalize_cuvs_knn(
    const int64_t* input_indices,
    const float* input_distances,
    int* output_indices,
    float* output_distances,
    int batch_n,
    int search_k,
    int output_k,
    int exclude_self,
    int distance_mode,
    int row_offset,
    int total_n) {
  embedding_last_error.clear();
  if (input_indices == nullptr || input_distances == nullptr ||
      output_indices == nullptr || output_distances == nullptr) {
    set_embedding_error("null CUDA KNN postprocessing pointer");
    return 1;
  }
  if (batch_n < 1 || total_n < 2 || row_offset < 0 ||
      row_offset + batch_n > total_n || output_k < 1 || search_k < output_k ||
      (exclude_self && search_k <= output_k)) {
    set_embedding_error("invalid CUDA KNN postprocessing dimensions");
    return 1;
  }
  int* invalid_rows = nullptr;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&invalid_rows), sizeof(int)),
                 "cudaMalloc(cuVS invalid rows)")) {
    return 1;
  }
  if (check_cuda(cudaMemset(invalid_rows, 0, sizeof(int)),
                 "cudaMemset(cuVS invalid rows)")) {
    cudaFree(invalid_rows);
    return 1;
  }
  const int threads = 256;
  const int blocks = (batch_n + threads - 1) / threads;
  finalize_cuvs_knn_kernel<<<blocks, threads>>>(
    input_indices,
    input_distances,
    output_indices,
    output_distances,
    batch_n,
    search_k,
    output_k,
    exclude_self,
    distance_mode,
    row_offset,
    total_n,
    invalid_rows
  );
  if (check_cuda(cudaGetLastError(), "finalize_cuvs_knn_kernel launch")) {
    cudaFree(invalid_rows);
    return 1;
  }
  int invalid = 0;
  if (check_cuda(cudaMemcpy(&invalid, invalid_rows, sizeof(int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(cuVS invalid rows D2H)")) {
    cudaFree(invalid_rows);
    return 1;
  }
  cudaFree(invalid_rows);
  if (invalid > 0) {
    set_embedding_error(
      "cuVS returned fewer non-self neighbors than requested for " +
      std::to_string(invalid) + " rows"
    );
    return 1;
  }
  return 0;
}

extern "C" int fastembedr_cuda_finalize_cuvs_query_knn(
    const int64_t* input_indices,
    const float* input_distances,
    int* output_indices,
    float* output_distances,
    int batch_n,
    int k,
    int distance_mode,
    int output_row_offset,
    int output_n,
    int reference_n) {
  embedding_last_error.clear();
  if (input_indices == nullptr || input_distances == nullptr ||
      output_indices == nullptr || output_distances == nullptr) {
    set_embedding_error("null CUDA query KNN postprocessing pointer");
    return 1;
  }
  if (batch_n < 1 || k < 1 || output_n < 1 || reference_n < 1 ||
      output_row_offset < 0 || output_row_offset + batch_n > output_n) {
    set_embedding_error("invalid CUDA query KNN postprocessing dimensions");
    return 1;
  }
  int* invalid_rows = nullptr;
  if (check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&invalid_rows), sizeof(int)),
        "cudaMalloc(query KNN invalid rows)")) {
    return 1;
  }
  if (check_cuda(
        cudaMemset(invalid_rows, 0, sizeof(int)),
        "cudaMemset(query KNN invalid rows)")) {
    cudaFree(invalid_rows);
    return 1;
  }
  const int threads = 256;
  const int blocks = (batch_n + threads - 1) / threads;
  finalize_cuvs_query_knn_kernel<<<blocks, threads>>>(
    input_indices, input_distances, output_indices, output_distances,
    batch_n, k, distance_mode, output_row_offset, output_n, reference_n,
    invalid_rows
  );
  if (check_cuda(
        cudaGetLastError(), "finalize_cuvs_query_knn_kernel launch")) {
    cudaFree(invalid_rows);
    return 1;
  }
  int invalid = 0;
  if (check_cuda(
        cudaMemcpy(
          &invalid, invalid_rows, sizeof(int), cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(query KNN invalid rows D2H)")) {
    cudaFree(invalid_rows);
    return 1;
  }
  cudaFree(invalid_rows);
  if (invalid > 0) {
    set_embedding_error(
      "cuVS returned fewer query-to-reference neighbors than requested for " +
      std::to_string(invalid) + " rows"
    );
    return 1;
  }
  return 0;
}

template <typename HostT>
int fastembedr_cuda_pca_fit_impl(const HostT* values,
                                 int n,
                                 int p,
                                 int n_components,
                                 int center,
                                 int scale,
                                 float* scores,
                                 float* components,
                                 float* singular_values,
                                 float* center_values,
                                 float* scale_values,
                                 int requested_method,
                                 unsigned int seed,
                                 int oversample,
                                 int power,
                                 int* selected_method) {
  embedding_last_error.clear();
  if (values == nullptr || scores == nullptr || components == nullptr ||
      singular_values == nullptr || center_values == nullptr ||
      scale_values == nullptr) {
    set_embedding_error("null host pointer in CUDA PCA fit");
    return 1;
  }
  if (n < 2 || p < 2 || n_components < 1 ||
      n_components > std::min(n - 1, p)) {
    set_embedding_error("invalid dimensions in CUDA PCA fit");
    return 1;
  }

  const std::size_t score_items = static_cast<std::size_t>(n) * n_components;
  const bool input_is_double = std::is_same<HostT, double>::value;
  const std::size_t workspace_bytes = align_bytes(
    score_items * sizeof(float)
  ) + cuda_pca_workspace_bytes(
    n, p, n_components, input_is_double
  );
  CudaWorkspace workspace;
  try {
    if (check_embedding_memory_available(
          workspace_bytes,
          "CUDA PCA fit allocation preflight"
        )) {
      return 1;
    }
    if (workspace.init(workspace_bytes, "cuda pca fit")) return 1;
    float* d_scores = workspace.alloc<float>(score_items, "cuda pca scores");
    if (d_scores == nullptr) return 1;
    if (cuda_pca_scores_to_device<HostT>(
          values,
          n,
          p,
          n_components,
          d_scores,
          &workspace,
          center != 0,
          scale != 0,
          components,
          singular_values,
          center_values,
          scale_values,
          requested_method,
          seed,
          oversample,
          power,
          selected_method
        )) {
      return 1;
    }
    return check_cuda(
      cudaMemcpy(
        scores,
        d_scores,
        score_items * sizeof(float),
        cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(cuda pca scores D2H)"
    );
  } catch (const std::exception& e) {
    set_embedding_error(std::string("CUDA PCA fit failed: ") + e.what());
    return 1;
  } catch (...) {
    set_embedding_error("CUDA PCA fit failed with an unknown error.");
    return 1;
  }
}

extern "C" int fastembedr_cuda_pca_fit(const float* values,
                                        int n,
                                        int p,
                                        int n_components,
                                        int center,
                                        int scale,
                                        float* scores,
                                        float* components,
                                        float* singular_values,
                                        float* center_values,
                                        float* scale_values,
                                        int requested_method,
                                        unsigned int seed,
                                        int oversample,
                                        int power,
                                        int* selected_method) {
  return fastembedr_cuda_pca_fit_impl(
    values, n, p, n_components, center, scale, scores, components,
    singular_values, center_values, scale_values, requested_method,
    seed, oversample, power, selected_method
  );
}

extern "C" int fastembedr_cuda_pca_fit_double(
    const double* values,
    int n,
    int p,
    int n_components,
    int center,
    int scale,
    float* scores,
    float* components,
    float* singular_values,
    float* center_values,
    float* scale_values,
    int requested_method,
    unsigned int seed,
    int oversample,
    int power,
    int* selected_method) {
  return fastembedr_cuda_pca_fit_impl(
    values, n, p, n_components, center, scale, scores, components,
    singular_values, center_values, scale_values, requested_method,
    seed, oversample, power, selected_method
  );
}

extern "C" int fastembedr_cuda_standardize_matrix(const double* values,
                                                  int n,
                                                  int p,
                                                  double* out,
                                                  double* center,
                                                  double* scale) {
  embedding_last_error.clear();
  if (values == nullptr || out == nullptr || center == nullptr || scale == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || p < 1) {
    set_embedding_error("invalid standardization dimensions");
    return 1;
  }
  const std::size_t data_items = static_cast<std::size_t>(n) * p;
  if (data_items > static_cast<std::size_t>(2147483647)) {
    set_embedding_error("CUDA standardization input is too large for one kernel launch");
    return 1;
  }
  const std::size_t data_bytes = data_items * sizeof(double);
  const std::size_t stat_bytes = static_cast<std::size_t>(p) * 2u * sizeof(double);

  double* d_values = nullptr;
  double* d_stats = nullptr;
  auto cleanup = [&]() {
    if (d_values != nullptr) cudaFree(d_values);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(data_bytes + stat_bytes, "CUDA standardization allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_values), data_bytes), "cudaMalloc(standardize values)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), stat_bytes), "cudaMalloc(standardize stats)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_values, values, data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(standardize values H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 256;
  standardize_stats_kernel<<<p, threads, 2u * threads * sizeof(double)>>>(
    d_values, d_stats, n, p
  );
  if (check_cuda(cudaGetLastError(), "standardize_stats_kernel launch")) {
    cleanup();
    return 1;
  }
  const int blocks = (static_cast<int>(data_items) + threads - 1) / threads;
  standardize_apply_kernel<<<blocks, threads>>>(d_values, d_stats, n, p);
  if (check_cuda(cudaGetLastError(), "standardize_apply_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(standardization)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_values, data_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(standardized values D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(center, d_stats, static_cast<std::size_t>(p) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(standardize centers D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(scale, d_stats + p, static_cast<std::size_t>(p) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(standardize scales D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_project_embedding(const double* reference_layout,
                                                 const int* projection_indices,
                                                 const double* projection_distances,
                                                 int n_reference,
                                                 int n_query,
                                                 int k,
                                                 int n_components,
                                                 double* out) {
  embedding_last_error.clear();
  if (reference_layout == nullptr || projection_indices == nullptr ||
      projection_distances == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n_reference < 1 || n_query < 1 || k < 1 || k > kCudaProjectionMaxK || n_components < 1) {
    set_embedding_error("invalid CUDA projection dimensions");
    return 1;
  }

  const std::size_t ref_items = static_cast<std::size_t>(n_reference) * n_components;
  const std::size_t proj_items = static_cast<std::size_t>(n_query) * k;
  const std::size_t out_items = static_cast<std::size_t>(n_query) * n_components;
  double* d_reference = nullptr;
  int* d_indices = nullptr;
  double* d_distances = nullptr;
  double* d_out = nullptr;
  auto cleanup = [&]() {
    if (d_reference != nullptr) cudaFree(d_reference);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_out != nullptr) cudaFree(d_out);
  };

  const std::size_t required_bytes =
    ref_items * sizeof(double) +
    proj_items * (sizeof(int) + sizeof(double)) +
    out_items * sizeof(double);
  if (check_embedding_memory_available(required_bytes, "CUDA projection allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_reference), ref_items * sizeof(double)), "cudaMalloc(projection reference)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), proj_items * sizeof(int)), "cudaMalloc(projection indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), proj_items * sizeof(double)), "cudaMalloc(projection distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), out_items * sizeof(double)), "cudaMalloc(projection output)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_reference, reference_layout, ref_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(projection reference H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, projection_indices, proj_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(projection indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_distances, projection_distances, proj_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(projection distances H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 256;
  const int blocks = (n_query + threads - 1) / threads;
  membership_project_kernel<<<blocks, threads>>>(
    d_reference, d_indices, d_distances, d_out,
    n_reference, n_query, k, n_components, 1
  );
  if (check_cuda(cudaGetLastError(), "membership_project_kernel(project) launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(projection)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_out, out_items * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(projection output D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_interpolate_landmark_layout(const double* landmark_layout,
                                                           const int* landmark_indices,
                                                           const int* projection_indices,
                                                           const double* projection_distances,
                                                           int n_landmarks,
                                                           int n,
                                                           int k,
                                                           int n_components,
                                                           double* out) {
  embedding_last_error.clear();
  if (landmark_layout == nullptr || landmark_indices == nullptr ||
      projection_indices == nullptr || projection_distances == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n_landmarks < 1 || n < 1 || k < 1 || k > kCudaProjectionMaxK || n_components < 1) {
    set_embedding_error("invalid CUDA landmark interpolation dimensions");
    return 1;
  }

  const std::size_t landmark_items = static_cast<std::size_t>(n_landmarks) * n_components;
  const std::size_t proj_items = static_cast<std::size_t>(n) * k;
  const std::size_t out_items = static_cast<std::size_t>(n) * n_components;
  double* d_landmark_layout = nullptr;
  int* d_landmark_indices = nullptr;
  int* d_projection_indices = nullptr;
  double* d_projection_distances = nullptr;
  double* d_out = nullptr;
  auto cleanup = [&]() {
    if (d_landmark_layout != nullptr) cudaFree(d_landmark_layout);
    if (d_landmark_indices != nullptr) cudaFree(d_landmark_indices);
    if (d_projection_indices != nullptr) cudaFree(d_projection_indices);
    if (d_projection_distances != nullptr) cudaFree(d_projection_distances);
    if (d_out != nullptr) cudaFree(d_out);
  };

  const std::size_t required_bytes =
    landmark_items * sizeof(double) +
    static_cast<std::size_t>(n_landmarks) * sizeof(int) +
    proj_items * (sizeof(int) + sizeof(double)) +
    out_items * sizeof(double);
  if (check_embedding_memory_available(required_bytes, "CUDA landmark interpolation allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_landmark_layout), landmark_items * sizeof(double)), "cudaMalloc(landmark layout)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_landmark_indices), static_cast<std::size_t>(n_landmarks) * sizeof(int)), "cudaMalloc(landmark indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_projection_indices), proj_items * sizeof(int)), "cudaMalloc(landmark projection indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_projection_distances), proj_items * sizeof(double)), "cudaMalloc(landmark projection distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), out_items * sizeof(double)), "cudaMalloc(landmark output)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_landmark_layout, landmark_layout, landmark_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(landmark layout H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_landmark_indices, landmark_indices, static_cast<std::size_t>(n_landmarks) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(landmark indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_projection_indices, projection_indices, proj_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(landmark projection indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_projection_distances, projection_distances, proj_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(landmark projection distances H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  membership_project_kernel<<<blocks, threads>>>(
    d_landmark_layout, d_projection_indices, d_projection_distances, d_out,
    n_landmarks, n, k, n_components, 0
  );
  if (check_cuda(cudaGetLastError(), "membership_project_kernel(landmark) launch")) {
    cleanup();
    return 1;
  }
  const int overwrite_blocks = (n_landmarks * n_components + threads - 1) / threads;
  overwrite_landmark_rows_kernel<<<overwrite_blocks, threads>>>(
    d_out, d_landmark_layout, d_landmark_indices, n_landmarks, n, n_components
  );
  if (check_cuda(cudaGetLastError(), "overwrite_landmark_rows_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(landmark interpolation)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_out, out_items * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(landmark output D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_landmark_project_interpolate_knn_confidence(const double* landmark_data,
                                                                           const double* query_data,
                                                                           const double* landmark_layout,
                                                                           const int* landmark_indices,
                                                                           int n_landmarks,
                                                                           int n,
                                                                           int n_features,
                                                                           int k,
                                                                           int n_components,
                                                                           double* out,
                                                                           int* projection_indices,
                                                                           double* projection_distances,
                                                                           double* confidence) {
  embedding_last_error.clear();
  if (landmark_data == nullptr || query_data == nullptr || landmark_layout == nullptr ||
      landmark_indices == nullptr || out == nullptr || projection_indices == nullptr ||
      projection_distances == nullptr || confidence == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n_landmarks < 1 || n < 1 || n_features < 1 || k < 1 ||
      k > kCudaProjectionMaxK || k > n_landmarks || n_components < 1) {
    set_embedding_error("invalid CUDA fused landmark projection dimensions");
    return 1;
  }

  const std::size_t landmark_data_items = static_cast<std::size_t>(n_landmarks) * n_features;
  const std::size_t query_data_items = static_cast<std::size_t>(n) * n_features;
  const std::size_t landmark_layout_items = static_cast<std::size_t>(n_landmarks) * n_components;
  const std::size_t projection_items = static_cast<std::size_t>(n) * k;
  const std::size_t out_items = static_cast<std::size_t>(n) * n_components;

  double* d_landmark_data = nullptr;
  double* d_query_data = nullptr;
  double* d_landmark_layout = nullptr;
  int* d_landmark_indices = nullptr;
  double* d_out = nullptr;
  int* d_projection_indices = nullptr;
  double* d_projection_distances = nullptr;
  double* d_confidence = nullptr;
  auto cleanup = [&]() {
    if (d_landmark_data != nullptr) cudaFree(d_landmark_data);
    if (d_query_data != nullptr) cudaFree(d_query_data);
    if (d_landmark_layout != nullptr) cudaFree(d_landmark_layout);
    if (d_landmark_indices != nullptr) cudaFree(d_landmark_indices);
    if (d_out != nullptr) cudaFree(d_out);
    if (d_projection_indices != nullptr) cudaFree(d_projection_indices);
    if (d_projection_distances != nullptr) cudaFree(d_projection_distances);
    if (d_confidence != nullptr) cudaFree(d_confidence);
  };

  const std::size_t required_bytes =
    (landmark_data_items + query_data_items + landmark_layout_items + out_items) * sizeof(double) +
    static_cast<std::size_t>(n_landmarks) * sizeof(int) +
    projection_items * (sizeof(int) + sizeof(double)) +
    static_cast<std::size_t>(n) * sizeof(double);
  if (check_embedding_memory_available(required_bytes, "CUDA fused landmark projection allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_landmark_data), landmark_data_items * sizeof(double)), "cudaMalloc(fused landmark data)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_query_data), query_data_items * sizeof(double)), "cudaMalloc(fused query data)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_landmark_layout), landmark_layout_items * sizeof(double)), "cudaMalloc(fused landmark layout)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_landmark_indices), static_cast<std::size_t>(n_landmarks) * sizeof(int)), "cudaMalloc(fused landmark indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), out_items * sizeof(double)), "cudaMalloc(fused landmark output)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_projection_indices), projection_items * sizeof(int)), "cudaMalloc(fused projection indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_projection_distances), projection_items * sizeof(double)), "cudaMalloc(fused projection distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_confidence), static_cast<std::size_t>(n) * sizeof(double)), "cudaMalloc(fused projection confidence)")) {
    cleanup();
    return 1;
  }

  if (check_cuda(cudaMemcpy(d_landmark_data, landmark_data, landmark_data_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(fused landmark data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_query_data, query_data, query_data_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(fused query data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_landmark_layout, landmark_layout, landmark_layout_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(fused landmark layout H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_landmark_indices, landmark_indices, static_cast<std::size_t>(n_landmarks) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(fused landmark indices H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  landmark_project_interpolate_knn_confidence_kernel<<<blocks, threads>>>(
    d_landmark_data,
    d_query_data,
    d_landmark_layout,
    d_out,
    d_projection_indices,
    d_projection_distances,
    d_confidence,
    n_landmarks,
    n,
    n_features,
    k,
    n_components
  );
  if (check_cuda(cudaGetLastError(), "landmark_project_interpolate_knn_confidence_kernel launch")) {
    cleanup();
    return 1;
  }
  const int overwrite_blocks = (n_landmarks * n_components + threads - 1) / threads;
  overwrite_landmark_rows_kernel<<<overwrite_blocks, threads>>>(
    d_out, d_landmark_layout, d_landmark_indices, n_landmarks, n, n_components
  );
  if (check_cuda(cudaGetLastError(), "overwrite_landmark_rows_kernel(fused projection) launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fused landmark projection)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_out, out_items * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(fused output D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(projection_indices, d_projection_indices, projection_items * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(fused indices D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(projection_distances, d_projection_distances, projection_items * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(fused distances D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(confidence, d_confidence, static_cast<std::size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(fused confidence D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_knn_structure_score(const double* layout,
                                                   const int* indices,
                                                   const int* keep,
                                                   const int* labels,
                                                   int n,
                                                   int index_rows,
                                                   int high_rank_limit,
                                                   int preserve_k,
                                                   int keep_n,
                                                   int compact_indices,
                                                   int n_label_levels,
                                                   double* totals) {
  embedding_last_error.clear();
  if (layout == nullptr || indices == nullptr || keep == nullptr || totals == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || index_rows < 1 || high_rank_limit < 1 || preserve_k < 1 ||
      preserve_k > high_rank_limit || preserve_k > kCudaScoreMaxK || keep_n < 1) {
    set_embedding_error("invalid CUDA structure-score dimensions");
    return 1;
  }
  if (n_label_levels > 0 && labels == nullptr) {
    set_embedding_error("labels pointer is null");
    return 1;
  }

  const std::size_t layout_items = static_cast<std::size_t>(n) * 2u;
  const std::size_t index_items = static_cast<std::size_t>(index_rows) * high_rank_limit;
  const std::size_t row_items = static_cast<std::size_t>(keep_n) * kCudaScoreWidth;
  double* d_layout = nullptr;
  int* d_indices = nullptr;
  int* d_keep = nullptr;
  int* d_labels = nullptr;
  double* d_rows = nullptr;
  double* d_totals = nullptr;
  auto cleanup = [&]() {
    if (d_layout != nullptr) cudaFree(d_layout);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_keep != nullptr) cudaFree(d_keep);
    if (d_labels != nullptr) cudaFree(d_labels);
    if (d_rows != nullptr) cudaFree(d_rows);
    if (d_totals != nullptr) cudaFree(d_totals);
  };

  const std::size_t required_bytes =
    layout_items * sizeof(double) +
    index_items * sizeof(int) +
    static_cast<std::size_t>(keep_n) * sizeof(int) +
    (n_label_levels > 0 ? static_cast<std::size_t>(n) * sizeof(int) : 0u) +
    row_items * sizeof(double) +
    kCudaScoreWidth * sizeof(double);
  if (check_embedding_memory_available(required_bytes, "CUDA structure-score allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_layout), layout_items * sizeof(double)), "cudaMalloc(score layout)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), index_items * sizeof(int)), "cudaMalloc(score indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_keep), static_cast<std::size_t>(keep_n) * sizeof(int)), "cudaMalloc(score keep)")) {
    cleanup();
    return 1;
  }
  if (n_label_levels > 0) {
    if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_labels), static_cast<std::size_t>(n) * sizeof(int)), "cudaMalloc(score labels)")) {
      cleanup();
      return 1;
    }
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_items * sizeof(double)), "cudaMalloc(score rows)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_totals), kCudaScoreWidth * sizeof(double)), "cudaMalloc(score totals)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_layout, layout, layout_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(score layout H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, index_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(score indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_keep, keep, static_cast<std::size_t>(keep_n) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(score keep H2D)")) {
    cleanup();
    return 1;
  }
  if (n_label_levels > 0 && check_cuda(cudaMemcpy(d_labels, labels, static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(score labels H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemset(d_rows, 0, row_items * sizeof(double)), "cudaMemset(score rows)")) {
    cleanup();
    return 1;
  }

  const int threads = 128;
  const int blocks = (keep_n + threads - 1) / threads;
  structure_score_rows_kernel<<<blocks, threads>>>(
    d_layout, d_indices, d_keep, d_labels, d_rows,
    n, index_rows, high_rank_limit, preserve_k, keep_n,
    compact_indices, n_label_levels
  );
  if (check_cuda(cudaGetLastError(), "structure_score_rows_kernel launch")) {
    cleanup();
    return 1;
  }
  reduce_width_kernel<<<1, 256, static_cast<std::size_t>(kCudaScoreWidth) * 256u * sizeof(double)>>>(
    d_rows, d_totals, keep_n, kCudaScoreWidth
  );
  if (check_cuda(cudaGetLastError(), "reduce_width_kernel(score) launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(structure score)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(totals, d_totals, kCudaScoreWidth * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(score totals D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_silhouette_score(const double* layout,
                                                const int* labels,
                                                const int* counts,
                                                int n,
                                                int n_label_levels,
                                                double* score) {
  embedding_last_error.clear();
  if (layout == nullptr || labels == nullptr || counts == nullptr || score == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || n_label_levels < 2 || n_label_levels > kCudaSilhouetteMaxLabels) {
    set_embedding_error("invalid CUDA silhouette dimensions");
    return 1;
  }

  const std::size_t layout_items = static_cast<std::size_t>(n) * 2u;
  const std::size_t row_items = static_cast<std::size_t>(n) * 2u;
  double* d_layout = nullptr;
  int* d_labels = nullptr;
  int* d_counts = nullptr;
  double* d_rows = nullptr;
  double* d_totals = nullptr;
  auto cleanup = [&]() {
    if (d_layout != nullptr) cudaFree(d_layout);
    if (d_labels != nullptr) cudaFree(d_labels);
    if (d_counts != nullptr) cudaFree(d_counts);
    if (d_rows != nullptr) cudaFree(d_rows);
    if (d_totals != nullptr) cudaFree(d_totals);
  };

  const std::size_t required_bytes =
    layout_items * sizeof(double) +
    static_cast<std::size_t>(n) * sizeof(int) +
    static_cast<std::size_t>(n_label_levels + 1) * sizeof(int) +
    row_items * sizeof(double) +
    2u * sizeof(double);
  if (check_embedding_memory_available(required_bytes, "CUDA silhouette allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_layout), layout_items * sizeof(double)), "cudaMalloc(silhouette layout)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_labels), static_cast<std::size_t>(n) * sizeof(int)), "cudaMalloc(silhouette labels)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_counts), static_cast<std::size_t>(n_label_levels + 1) * sizeof(int)), "cudaMalloc(silhouette counts)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_items * sizeof(double)), "cudaMalloc(silhouette rows)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_totals), 2u * sizeof(double)), "cudaMalloc(silhouette totals)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_layout, layout, layout_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(silhouette layout H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_labels, labels, static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(silhouette labels H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_counts, counts, static_cast<std::size_t>(n_label_levels + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(silhouette counts H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemset(d_rows, 0, row_items * sizeof(double)), "cudaMemset(silhouette rows)")) {
    cleanup();
    return 1;
  }

  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  silhouette_rows_kernel<<<blocks, threads>>>(
    d_layout, d_labels, d_counts, d_rows, n, n_label_levels
  );
  if (check_cuda(cudaGetLastError(), "silhouette_rows_kernel launch")) {
    cleanup();
    return 1;
  }
  reduce_width_kernel<<<1, 256, 2u * 256u * sizeof(double)>>>(d_rows, d_totals, n, 2);
  if (check_cuda(cudaGetLastError(), "reduce_width_kernel(silhouette) launch")) {
    cleanup();
    return 1;
  }
  double totals[2] = {0.0, 0.0};
  if (check_cuda(cudaMemcpy(totals, d_totals, 2u * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(silhouette totals D2H)")) {
    cleanup();
    return 1;
  }
  score[0] = totals[1] > 0.0 ? totals[0] / totals[1] : NAN;

  cleanup();
  return 0;
}

template <typename DistanceT>
int fastembedr_cuda_spectral_init_from_knn_impl(const int* indices,
                                                      const DistanceT* distances,
                                                      int n,
                                                       int k,
                                                       int spectral_n_iter,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || spectral_n_iter < 1) {
    set_embedding_error("invalid spectral initialization dimensions or parameters");
    return 1;
  }

  const int width = std::min(256, std::max(k, 2 * k));
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const int stat_blocks = std::max(1, std::min(1024, blocks));
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(DistanceT)) +
    graph_items * (sizeof(int) + sizeof(float)) +
    static_cast<std::size_t>(n) * sizeof(int) +
    2u * embed_bytes +
    static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double) +
    5u * sizeof(double);

  int* d_indices = nullptr;
  DistanceT* d_distances = nullptr;
  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
  float* d_delta = nullptr;
  double* d_partial = nullptr;
  double* d_stats = nullptr;

  auto cleanup = [&]() {
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_neighbors != nullptr) cudaFree(d_neighbors);
    if (d_weights != nullptr) cudaFree(d_weights);
    if (d_counts != nullptr) cudaFree(d_counts);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_next != nullptr) cudaFree(d_next);
    if (d_delta != nullptr) cudaFree(d_delta);
    if (d_partial != nullptr) cudaFree(d_partial);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA spectral initialization allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(spectral indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(DistanceT)), "cudaMalloc(spectral distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_neighbors), graph_items * sizeof(int)), "cudaMalloc(spectral neighbors)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), graph_items * sizeof(float)), "cudaMalloc(spectral weights)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_counts), static_cast<std::size_t>(n) * sizeof(int)), "cudaMalloc(spectral row counts)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(spectral current)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next), embed_bytes), "cudaMalloc(spectral next)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_partial), static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double)), "cudaMalloc(spectral partial)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), 5u * sizeof(double)), "cudaMalloc(spectral stats)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(spectral indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice), "cudaMemcpy(spectral distances H2D)")) {
    cleanup();
    return 1;
  }

  if (prepare_device_knn_graph(
        d_indices, d_distances, d_neighbors, d_weights, d_counts,
        n, k, width, index_offset, 0)) {
    cleanup();
    return 1;
  }

  random_init_kernel<<<blocks, threads>>>(d_current, n, seed);
  if (check_cuda(cudaGetLastError(), "random_init_kernel launch")) {
    cleanup();
    return 1;
  }
  if (normalize_device_init(d_current, d_partial, d_stats, n, stat_blocks, threads)) {
    cleanup();
    return 1;
  }

  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    diffuse_init_kernel<<<blocks, threads>>>(d_neighbors, d_weights, d_current, d_next, n, width);
    if (check_cuda(cudaGetLastError(), "diffuse_init_kernel launch")) {
      cleanup();
      return 1;
    }
    if (normalize_device_init(d_next, d_partial, d_stats, n, stat_blocks, threads)) {
      cleanup();
      return 1;
    }
    std::swap(d_current, d_next);
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(spectral init)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(spectral init D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_spectral_init_from_knn(const int* indices,
                                                      const double* distances,
                                                      int n,
                                                      int k,
                                                      int spectral_n_iter,
                                                      unsigned int seed,
                                                      int index_offset,
                                                      float* out) {
  return fastembedr_cuda_spectral_init_from_knn_impl<double>(
    indices, distances, n, k, spectral_n_iter, seed, index_offset, out
  );
}

template <typename DistanceT>
int fastembedr_cuda_umap_graph_dump_from_knn_impl(const int* indices,
                                                        const DistanceT* distances,
                                                        int n,
                                                        int k,
                                                        int index_offset,
                                                        int* out_heads,
                                                        int* out_tails,
                                                        float* out_weights,
                                                        float* out_epochs_per_sample,
                                                        int* out_width,
                                                        int* out_capacity) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || out_heads == nullptr ||
      out_tails == nullptr || out_weights == nullptr ||
      out_epochs_per_sample == nullptr || out_width == nullptr ||
      out_capacity == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256) {
    set_embedding_error("invalid CUDA UMAP graph dump dimensions");
    return 1;
  }

  const int objective = 0;
  const int width = std::min(256, std::max(k, 2 * k));
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t cub_select_temp_bytes =
    cub_select_flagged_temp_bytes(static_cast<int>(graph_items));
  const int threads = 256;
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(DistanceT)) +
    graph_items * (3u * sizeof(int) + 5u * sizeof(float)) +
    graph_items * (sizeof(unsigned char) + sizeof(int)) +
    sizeof(int) +
    cub_select_temp_bytes +
    static_cast<std::size_t>(n) * sizeof(int) +
    24u * 256u;

  int* d_indices = nullptr;
  DistanceT* d_distances = nullptr;
  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  unsigned char* d_valid_flags = nullptr;
  int* d_selected_ids = nullptr;
  int* d_selected_count = nullptr;
  int* d_coo_heads = nullptr;
  int* d_coo_tails = nullptr;
  float* d_coo_weights = nullptr;
  float* d_coo_epochs_per_sample = nullptr;

  CudaWorkspace workspace;
  auto cleanup = [&]() {};

  if (check_embedding_memory_available(required_bytes, "CUDA UMAP graph dump allocation preflight")) return 1;
  if (workspace.init(required_bytes, "umap graph dump")) {
    cleanup();
    return 1;
  }
  d_indices = workspace.alloc<int>(input_items, "umap dump indices");
  d_distances = workspace.alloc<DistanceT>(input_items, "umap dump distances");
  d_neighbors = workspace.alloc<int>(graph_items, "umap dump neighbors");
  d_weights = workspace.alloc<float>(graph_items, "umap dump weights");
  d_counts = workspace.alloc<int>(static_cast<std::size_t>(n), "umap dump row counts");
  d_valid_flags = workspace.alloc<unsigned char>(graph_items, "umap dump valid flags");
  d_selected_ids = workspace.alloc<int>(graph_items, "umap dump selected ids");
  d_selected_count = workspace.alloc<int>(1u, "umap dump selected count");
  d_coo_heads = workspace.alloc<int>(graph_items, "umap dump coo heads");
  d_coo_tails = workspace.alloc<int>(graph_items, "umap dump coo tails");
  d_coo_weights = workspace.alloc<float>(graph_items, "umap dump coo weights");
  d_coo_epochs_per_sample = workspace.alloc<float>(graph_items, "umap dump coo epochs_per_sample");
  if (d_indices == nullptr || d_distances == nullptr || d_neighbors == nullptr ||
      d_weights == nullptr || d_counts == nullptr || d_valid_flags == nullptr ||
      d_selected_ids == nullptr || d_selected_count == nullptr ||
      d_coo_heads == nullptr || d_coo_tails == nullptr || d_coo_weights == nullptr ||
      d_coo_epochs_per_sample == nullptr) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(umap dump indices H2D)") ||
      check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice), "cudaMemcpy(umap dump distances H2D)")) {
    cleanup();
    return 1;
  }

  if (prepare_device_knn_graph(
        d_indices, d_distances, d_neighbors, d_weights, d_counts,
        n, k, width, index_offset, objective, &workspace)) {
    cleanup();
    return 1;
  }

  const int graph_blocks = (static_cast<int>(graph_items) + threads - 1) / threads;
  mark_valid_graph_entries_rank_major_kernel<<<graph_blocks, threads>>>(
    d_neighbors, d_weights, d_valid_flags, n, width
  );
  if (check_cuda(cudaGetLastError(), "mark_valid_graph_entries_rank_major_kernel(umap dump) launch")) {
    cleanup();
    return 1;
  }
  if (cub_select_valid_ids_rank_major(
        d_valid_flags, d_selected_ids, d_selected_count,
        static_cast<int>(graph_items), workspace, cub_select_temp_bytes,
        "umap dump selected edge ids")) {
    cleanup();
    return 1;
  }
  int edge_items = 0;
  if (check_cuda(cudaMemcpy(&edge_items, d_selected_count, sizeof(int), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(umap dump edge count D2H)")) {
    cleanup();
    return 1;
  }
  if (edge_items < 1 || static_cast<std::size_t>(edge_items) > graph_items) {
    set_embedding_error("invalid CUDA UMAP graph edge count");
    cleanup();
    return 1;
  }

  const float graph_max_weight = 1.0f;
  const int edge_blocks = (edge_items + threads - 1) / threads;
  pack_coo_graph_from_selected_rank_major_kernel<<<edge_blocks, threads>>>(
    d_selected_ids, d_neighbors, d_weights, d_coo_heads, d_coo_tails, d_coo_weights,
    d_coo_epochs_per_sample, graph_max_weight, edge_items, n, width
  );
  if (check_cuda(cudaGetLastError(), "pack_coo_graph_from_selected_rank_major_kernel(umap dump) launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(umap dump)") ||
      check_cuda(cudaMemcpy(out_heads, d_coo_heads, static_cast<std::size_t>(edge_items) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(umap dump heads D2H)") ||
      check_cuda(cudaMemcpy(out_tails, d_coo_tails, static_cast<std::size_t>(edge_items) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(umap dump tails D2H)") ||
      check_cuda(cudaMemcpy(out_weights, d_coo_weights, static_cast<std::size_t>(edge_items) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(umap dump weights D2H)") ||
      check_cuda(cudaMemcpy(out_epochs_per_sample, d_coo_epochs_per_sample, static_cast<std::size_t>(edge_items) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(umap dump epochs D2H)")) {
    cleanup();
    return 1;
  }

  *out_width = width;
  *out_capacity = edge_items;
  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_umap_graph_dump_from_knn(const int* indices,
                                                        const double* distances,
                                                        int n,
                                                        int k,
                                                        int index_offset,
                                                        int* out_heads,
                                                        int* out_tails,
                                                        float* out_weights,
                                                        float* out_epochs_per_sample,
                                                        int* out_width,
                                                        int* out_capacity) {
  return fastembedr_cuda_umap_graph_dump_from_knn_impl<double>(
    indices, distances, n, k, index_offset, out_heads, out_tails, out_weights,
    out_epochs_per_sample, out_width, out_capacity
  );
}

template <typename DistanceT>
int fastembedr_cuda_umap_from_device_knn_spectral_impl(const int* d_indices_src,
                                                       const DistanceT* d_distances_src,
                                                       int n,
                                                       int k,
                                                       int n_epochs,
                                                       int negative_sample_rate,
                                                       float learning_rate,
                                                       float a,
                                                       float b,
                                                       float repulsion_strength,
                                                       int spectral_n_iter,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       int optimizer_mode,
                                                       bool binary_graph,
                                                       float* out) {
  embedding_last_error.clear();
  last_umap_cuda_graph_capture = false;
  if (d_indices_src == nullptr || d_distances_src == nullptr || out == nullptr) {
    set_embedding_error("null device pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 ||
      negative_sample_rate < 0 || optimizer_mode < 0 || optimizer_mode > 1 ||
      learning_rate <= 0.0f || spectral_n_iter < 1) {
    set_embedding_error("invalid fused CUDA UMAP dimensions or parameters");
    return 1;
  }

  const int objective = 0;
  const int width = std::min(256, std::max(k, 2 * k));
  const bool row_optimizer = optimizer_mode == 0;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t cub_select_temp_bytes = row_optimizer ? 0u :
    cub_select_flagged_temp_bytes(static_cast<int>(graph_items));
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const int stat_blocks = std::max(1, std::min(1024, blocks));
  const std::size_t coo_bytes = row_optimizer ? 0u :
    graph_items * (3u * sizeof(int) + 2u * sizeof(float) +
      sizeof(unsigned char)) + sizeof(int) + cub_select_temp_bytes;
  const std::size_t required_bytes =
    graph_items * 2u * (sizeof(int) + sizeof(float)) + coo_bytes +
    (optimizer_mode == 1 ? 3u : 2u) * embed_bytes +
    static_cast<std::size_t>(n) * sizeof(int) +
    static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double) +
    5u * sizeof(double) +
    32u * 256u;

  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  unsigned char* d_valid_flags = nullptr;
  int* d_selected_ids = nullptr;
  int* d_selected_count = nullptr;
  int* d_coo_heads = nullptr;
  int* d_coo_tails = nullptr;
  float* d_coo_weights = nullptr;
  float* d_coo_epochs_per_sample = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
  float* d_delta = nullptr;
  double* d_partial = nullptr;
  double* d_stats = nullptr;

  CudaWorkspace workspace;
  auto cleanup = [&]() {};

  if (check_embedding_memory_available(required_bytes, "CUDA fused UMAP allocation preflight")) return 1;
  if (workspace.init(required_bytes, "fused umap")) {
    cleanup();
    return 1;
  }
  d_neighbors = workspace.alloc<int>(graph_items, "fused umap neighbors");
  d_weights = workspace.alloc<float>(graph_items, "fused umap weights");
  d_counts = workspace.alloc<int>(static_cast<std::size_t>(n), "fused umap row counts");
  if (!row_optimizer) {
    d_valid_flags = workspace.alloc<unsigned char>(
      graph_items, "fused umap valid flags"
    );
    d_selected_ids = workspace.alloc<int>(
      graph_items, "fused umap selected ids"
    );
    d_selected_count = workspace.alloc<int>(
      1u, "fused umap selected count"
    );
    d_coo_heads = workspace.alloc<int>(graph_items, "fused umap coo heads");
    d_coo_tails = workspace.alloc<int>(graph_items, "fused umap coo tails");
    d_coo_weights = workspace.alloc<float>(
      graph_items, "fused umap coo weights"
    );
    d_coo_epochs_per_sample = workspace.alloc<float>(
      graph_items, "fused umap coo epochs_per_sample"
    );
  }
  d_current = workspace.alloc<float>(static_cast<std::size_t>(n) * 2u, "fused umap current");
  d_next = workspace.alloc<float>(static_cast<std::size_t>(n) * 2u, "fused umap next");
  d_delta = optimizer_mode == 1 ?
    workspace.alloc<float>(static_cast<std::size_t>(n) * 2u, "fused umap delta") :
    nullptr;
  d_partial = workspace.alloc<double>(static_cast<std::size_t>(stat_blocks) * 5u, "fused umap partial");
  d_stats = workspace.alloc<double>(5u, "fused umap stats");
  const bool missing_coo = !row_optimizer &&
    (d_valid_flags == nullptr || d_selected_ids == nullptr ||
     d_selected_count == nullptr || d_coo_heads == nullptr ||
     d_coo_tails == nullptr || d_coo_weights == nullptr ||
     d_coo_epochs_per_sample == nullptr);
  if (d_neighbors == nullptr || d_weights == nullptr || d_counts == nullptr ||
      missing_coo || d_current == nullptr ||
      d_next == nullptr || (optimizer_mode == 1 && d_delta == nullptr) ||
      d_partial == nullptr || d_stats == nullptr) {
    cleanup();
    return 1;
  }

  if (prepare_device_knn_graph(
        d_indices_src, d_distances_src, d_neighbors, d_weights, d_counts,
        n, k, width, index_offset, objective, &workspace)) {
    cleanup();
    return 1;
  }
  if (binary_graph) {
    const int total_blocks = (static_cast<int>(graph_items) + threads - 1) / threads;
    binary_graph_weights_kernel<<<total_blocks, threads>>>(d_neighbors, d_weights, n, width);
    if (check_cuda(cudaGetLastError(), "binary_graph_weights_kernel(fused umap) launch")) {
      cleanup();
      return 1;
    }
  }

  random_init_kernel<<<blocks, threads>>>(d_current, n, seed);
  if (check_cuda(cudaGetLastError(), "random_init_kernel(fused umap) launch")) {
    cleanup();
    return 1;
  }
  if (normalize_device_init(d_current, d_partial, d_stats, n, stat_blocks, threads)) {
    cleanup();
    return 1;
  }
  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    diffuse_init_kernel<<<blocks, threads>>>(d_neighbors, d_weights, d_current, d_next, n, width);
    if (check_cuda(cudaGetLastError(), "diffuse_init_kernel(fused umap) launch")) {
      cleanup();
      return 1;
    }
    if (normalize_device_init(d_next, d_partial, d_stats, n, stat_blocks, threads)) {
      cleanup();
      return 1;
    }
    std::swap(d_current, d_next);
  }

  int edge_items = 0;
  const float graph_max_weight = 1.0f;
  const EmbedParams params{
    n, width, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, graph_max_weight, repulsion_strength
  };
  int edge_blocks = 0;
  if (!row_optimizer) {
    const int graph_blocks =
      (static_cast<int>(graph_items) + threads - 1) / threads;
    mark_valid_graph_entries_rank_major_kernel<<<graph_blocks, threads>>>(
      d_neighbors, d_weights, d_valid_flags, n, width
    );
    if (check_cuda(
          cudaGetLastError(),
          "mark_valid_graph_entries_rank_major_kernel(fused umap) launch"
        )) {
      cleanup();
      return 1;
    }
    if (cub_select_valid_ids_rank_major(
          d_valid_flags, d_selected_ids, d_selected_count,
          static_cast<int>(graph_items), workspace, cub_select_temp_bytes,
          "fused umap selected edge ids")) {
      cleanup();
      return 1;
    }
    if (check_cuda(
          cudaMemcpy(
            &edge_items, d_selected_count, sizeof(int),
            cudaMemcpyDeviceToHost
          ),
          "cudaMemcpy(fused umap edge count D2H)"
        )) {
      cleanup();
      return 1;
    }
    if (edge_items < 1 ||
        static_cast<std::size_t>(edge_items) > graph_items) {
      set_embedding_error("invalid fused CUDA UMAP graph edge count");
      cleanup();
      return 1;
    }
    edge_blocks = (edge_items + threads - 1) / threads;
    pack_coo_graph_from_selected_rank_major_kernel<<<
      edge_blocks, threads
    >>>(
      d_selected_ids, d_neighbors, d_weights, d_coo_heads, d_coo_tails,
      d_coo_weights, d_coo_epochs_per_sample, graph_max_weight,
      edge_items, n, width
    );
    if (check_cuda(
          cudaGetLastError(),
          "pack_coo_graph_from_selected_rank_major_kernel(fused umap) launch"
        )) {
      cleanup();
      return 1;
    }
  }

  const float max_abs_coord = 16.0f;
  CudaStreamOwner& execution_stream = cuda_execution_stream();
  if (execution_stream.init()) {
    cleanup();
    return 1;
  }
  cudaStream_t stream = execution_stream.get();
  const int warps_per_block = threads / 32;
  const int row_blocks = std::max(
    1, std::min(4096, (n + warps_per_block - 1) / warps_per_block)
  );
  if (check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize(umap setup before graph capture)"
      )) {
    cleanup();
    return 1;
  }
  auto encode_epoch = [&](int epoch, bool check_launches) -> int {
    if (optimizer_mode == 1) {
      if (check_cuda(
            cudaMemsetAsync(d_delta, 0, embed_bytes, stream),
            "cudaMemsetAsync(fused umap delta)"
          )) {
        return 1;
      }
      embed_epoch_coo_delta_kernel<<<
        edge_blocks, threads, 0, stream
      >>>(
        d_current, d_delta, d_coo_heads, d_coo_tails, d_coo_weights,
        d_coo_epochs_per_sample, params, static_cast<unsigned int>(epoch),
        edge_items
      );
      if (check_launches && check_cuda(
            cudaGetLastError(),
            "embed_epoch_coo_delta_kernel(fused umap) launch"
          )) {
        return 1;
      }
      apply_delta_sanitize_layout_kernel<<<
        blocks, threads, 0, stream
      >>>(d_current, d_delta, n, max_abs_coord);
      if (check_launches && check_cuda(
            cudaGetLastError(),
            "apply_delta_sanitize_layout_kernel(fused umap) launch"
          )) {
        return 1;
      }
    } else {
      embed_epoch_row_atomic_kernel<<<
        row_blocks, threads, 0, stream
      >>>(
        d_current, d_neighbors, d_weights, params,
        static_cast<unsigned int>(epoch), width
      );
      if (check_launches && check_cuda(
            cudaGetLastError(),
            "embed_epoch_row_atomic_kernel(fused umap) launch"
          )) {
        return 1;
      }
      umap_sanitize_layout_kernel<<<blocks, threads, 0, stream>>>(
        d_current, n, max_abs_coord
      );
      if (check_launches && check_cuda(
            cudaGetLastError(),
            "umap_sanitize_layout_kernel(row umap) launch"
          )) {
        return 1;
      }
    }
    return 0;
  };

  bool used_graph_capture = false;
  CudaGraphExecOwner epoch_graph;
  if (cuda_graph_capture_enabled() && n_epochs > 1) {
    cudaGraph_t graph = nullptr;
    cudaError_t code = cudaStreamBeginCapture(
      stream, cudaStreamCaptureModeGlobal
    );
    bool capture_ok = code == cudaSuccess;
    if (capture_ok) {
      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        if (encode_epoch(epoch, false)) {
          capture_ok = false;
          break;
        }
      }
      code = cudaStreamEndCapture(stream, &graph);
      capture_ok = capture_ok && code == cudaSuccess && graph != nullptr;
    }
    if (capture_ok) {
      const int instantiate_status = epoch_graph.instantiate(graph);
      graph = nullptr;
      if (instantiate_status == 0) {
        if (check_cuda(
              cudaGraphLaunch(epoch_graph.get(), stream),
              "cudaGraphLaunch(umap epochs)"
            )) {
          cleanup();
          return 1;
        }
        used_graph_capture = true;
      }
    }
    if (!used_graph_capture) {
      if (graph != nullptr) cudaGraphDestroy(graph);
      cudaGetLastError();
    }
  }

  if (!used_graph_capture) {
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
      if (encode_epoch(epoch, true)) {
        cleanup();
        return 1;
      }
    }
  }
  last_umap_cuda_graph_capture = used_graph_capture;

  if (check_cuda(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(fused umap)"
      ) || check_cuda(
        cudaMemcpy(
          out, d_current, embed_bytes, cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(fused umap D2H)"
      )) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

template <typename DistanceT>
int fastembedr_cuda_umap_from_knn_spectral_impl(const int* indices,
                                                const DistanceT* distances,
                                                int n,
                                                int k,
                                                int n_epochs,
                                                int negative_sample_rate,
                                                float learning_rate,
                                                float a,
                                                float b,
                                                float repulsion_strength,
                                                int spectral_n_iter,
                                                unsigned int seed,
                                                int index_offset,
                                                int optimizer_mode,
                                                bool binary_graph,
                                                float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256) {
    set_embedding_error("invalid host CUDA UMAP dimensions");
    return 1;
  }
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  int* d_indices = nullptr;
  DistanceT* d_distances = nullptr;
  const std::size_t workspace_bytes =
    input_items * (sizeof(int) + sizeof(DistanceT)) + 8u * 256u;
  CudaWorkspace workspace;
  auto cleanup = [&]() {};
  if (workspace.init(workspace_bytes, "fused umap host knn copy")) {
    cleanup();
    return 1;
  }
  d_indices = workspace.alloc<int>(input_items, "fused umap indices");
  d_distances = workspace.alloc<DistanceT>(input_items, "fused umap distances");
  if (d_indices == nullptr || d_distances == nullptr) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(fused umap indices H2D)") ||
      check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice), "cudaMemcpy(fused umap distances H2D)")) {
    cleanup();
    return 1;
  }
  const int status = fastembedr_cuda_umap_from_device_knn_spectral_impl<DistanceT>(
    d_indices, d_distances, n, k, n_epochs, negative_sample_rate, learning_rate,
    a, b, repulsion_strength, spectral_n_iter, seed, index_offset, optimizer_mode,
    binary_graph, out
  );
  cleanup();
  return status;
}

extern "C" int fastembedr_cuda_umap_from_knn_spectral(const int* indices,
                                                       const double* distances,
                                                       int n,
                                                       int k,
                                                       int n_epochs,
                                                       int negative_sample_rate,
                                                       float learning_rate,
                                                       float a,
                                                       float b,
                                                       float repulsion_strength,
                                                       int spectral_n_iter,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       int optimizer_mode,
                                                       float* out) {
  return fastembedr_cuda_umap_from_knn_spectral_impl<double>(
    indices, distances, n, k, n_epochs, negative_sample_rate, learning_rate,
    a, b, repulsion_strength, spectral_n_iter, seed, index_offset,
    optimizer_mode, false, out
  );
}

extern "C" int fastembedr_cuda_umap_from_knn_spectral_float(const int* indices,
                                                             const float* distances,
                                                             int n,
                                                             int k,
                                                             int n_epochs,
                                                             int negative_sample_rate,
                                                             float learning_rate,
                                                             float a,
                                                             float b,
                                                             float repulsion_strength,
                                                             int spectral_n_iter,
                                                             unsigned int seed,
                                                             int index_offset,
                                                             int optimizer_mode,
                                                             float* out) {
  return fastembedr_cuda_umap_from_knn_spectral_impl<float>(
    indices, distances, n, k, n_epochs, negative_sample_rate, learning_rate,
    a, b, repulsion_strength, spectral_n_iter, seed, index_offset,
    optimizer_mode, false, out
  );
}

extern "C" int fastembedr_cuda_umap_from_device_knn_spectral_float(const int* device_indices,
                                                                    const float* device_distances,
                                                                    int n,
                                                                    int k,
                                                                    int n_epochs,
                                                                    int negative_sample_rate,
                                                                    float learning_rate,
                                                                    float a,
                                                                    float b,
                                                                    float repulsion_strength,
                                                                    int spectral_n_iter,
                                                                    unsigned int seed,
                                                                    int index_offset,
                                                                    int optimizer_mode,
                                                                    int binary_graph,
                                                                    float* out) {
  return fastembedr_cuda_umap_from_device_knn_spectral_impl<float>(
    device_indices, device_distances, n, k, n_epochs, negative_sample_rate,
    learning_rate, a, b, repulsion_strength, spectral_n_iter, seed,
    index_offset, optimizer_mode, binary_graph != 0, out
  );
}

extern "C" int fastembedr_cuda_umap_optimize_coo(const int* heads,
                                                  const int* tails,
                                                  const float* weights,
                                                  const float* epochs_per_sample,
                                                  const float* init,
                                                  int n,
                                                  int n_edges_capacity,
                                                  int n_epochs,
                                                  int negative_sample_rate,
                                                  float learning_rate,
                                                  float a,
                                                  float b,
                                                  float repulsion_strength,
                                                  unsigned int seed,
                                                  int optimizer_mode,
                                                  float* out) {
  embedding_last_error.clear();
  if (heads == nullptr || tails == nullptr || weights == nullptr ||
      epochs_per_sample == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || n_edges_capacity < 1 || n_epochs < 1 ||
      negative_sample_rate < 0 || learning_rate <= 0.0f ||
      repulsion_strength <= 0.0f) {
    set_embedding_error("invalid CUDA COO UMAP optimization parameters");
    return 1;
  }

  const std::size_t edge_items = static_cast<std::size_t>(n_edges_capacity);
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const std::size_t required_bytes =
    edge_items * (2u * sizeof(int) + 2u * sizeof(float)) +
    (optimizer_mode == 1 ? 2u : 1u) * embed_bytes;

  int* d_heads = nullptr;
  int* d_tails = nullptr;
  float* d_weights = nullptr;
  float* d_epochs = nullptr;
  float* d_layout = nullptr;
  float* d_delta = nullptr;
  auto cleanup = [&]() {
    if (d_heads != nullptr) cudaFree(d_heads);
    if (d_tails != nullptr) cudaFree(d_tails);
    if (d_weights != nullptr) cudaFree(d_weights);
    if (d_epochs != nullptr) cudaFree(d_epochs);
    if (d_layout != nullptr) cudaFree(d_layout);
    if (d_delta != nullptr) cudaFree(d_delta);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA COO UMAP optimizer allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_heads), edge_items * sizeof(int)), "cudaMalloc(coo umap heads)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_tails), edge_items * sizeof(int)), "cudaMalloc(coo umap tails)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), edge_items * sizeof(float)), "cudaMalloc(coo umap weights)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_epochs), edge_items * sizeof(float)), "cudaMalloc(coo umap epochs)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_layout), embed_bytes), "cudaMalloc(coo umap layout)")) {
    cleanup();
    return 1;
  }
  if (optimizer_mode == 1 &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_delta), embed_bytes), "cudaMalloc(coo umap delta)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_heads, heads, edge_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coo umap heads H2D)") ||
      check_cuda(cudaMemcpy(d_tails, tails, edge_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(coo umap tails H2D)") ||
      check_cuda(cudaMemcpy(d_weights, weights, edge_items * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(coo umap weights H2D)") ||
      check_cuda(cudaMemcpy(d_epochs, epochs_per_sample, edge_items * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(coo umap epochs H2D)") ||
      check_cuda(cudaMemcpy(d_layout, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(coo umap init H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 256;
  const int row_blocks = (n + threads - 1) / threads;
  const int edge_blocks = (n_edges_capacity + threads - 1) / threads;
  const float graph_max_weight = 1.0f;
  const float max_abs_coord = 16.0f;
  const EmbedParams params{
    n, 0, n_epochs, negative_sample_rate, 0, seed, learning_rate, a, b,
    graph_max_weight, repulsion_strength
  };
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    if (optimizer_mode == 1) {
      if (check_cuda(cudaMemset(d_delta, 0, embed_bytes), "cudaMemset(coo umap delta)")) {
        cleanup();
        return 1;
      }
      embed_epoch_coo_delta_kernel<<<edge_blocks, threads>>>(
        d_layout, d_delta, d_heads, d_tails, d_weights, d_epochs,
        params, static_cast<unsigned int>(epoch), n_edges_capacity
      );
      if (check_cuda(cudaGetLastError(), "embed_epoch_coo_delta_kernel(coo umap) launch")) {
        cleanup();
        return 1;
      }
      apply_delta_sanitize_layout_kernel<<<row_blocks, threads>>>(d_layout, d_delta, n, max_abs_coord);
      if (check_cuda(cudaGetLastError(), "apply_delta_sanitize_layout_kernel(coo umap) launch")) {
        cleanup();
        return 1;
      }
    } else {
      embed_epoch_coo_atomic_kernel<<<edge_blocks, threads>>>(
        d_layout, d_heads, d_tails, d_weights, d_epochs,
        params, static_cast<unsigned int>(epoch), n_edges_capacity
      );
      if (check_cuda(cudaGetLastError(), "embed_epoch_coo_atomic_kernel(coo umap) launch")) {
        cleanup();
        return 1;
      }
      umap_sanitize_layout_kernel<<<row_blocks, threads>>>(d_layout, n, max_abs_coord);
      if (check_cuda(cudaGetLastError(), "umap_sanitize_layout_kernel(coo umap) launch")) {
        cleanup();
        return 1;
      }
    }
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(coo umap)") ||
      check_cuda(cudaMemcpy(out, d_layout, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(coo umap D2H)")) {
    cleanup();
    return 1;
  }
  cleanup();
  return 0;
}

int resolve_cuda_tsne_fft_grid_size(const int n) {
  int grid_size = 512;
  if (n < 20000) grid_size = 256;
  if (n >= 100000) grid_size = 1024;
  const char* grid_env = std::getenv("FASTEMBEDR_TSNE_FFT_GRID");
  if (grid_env != nullptr && grid_env[0] != '\0') {
    const int requested = std::atoi(grid_env);
    if (requested == 32 || requested == 64 || requested == 128 ||
        requested == 256 || requested == 512 || requested == 1024) {
      grid_size = requested;
    }
  }
  return grid_size;
}

extern "C" int fastembedr_cuda_opentsne_fft_grid_size(const int n) {
  return resolve_cuda_tsne_fft_grid_size(n);
}

template <typename DistanceT>
int fastembedr_cuda_opentsne_fft_from_knn_impl(const int* indices,
                                                       const DistanceT* distances,
                                                       const float* init,
                                                       int has_init,
                                                       const double* pca_init_double,
                                                       const float* pca_init_float,
                                                       const float* pca_init_device_row_major,
                                                       int pca_init_p,
                                                       int n,
                                                       int k,
                                                       int n_components,
                                                       float perplexity,
                                                       int early_exaggeration_iter,
                                                       int n_iter,
                                                       float early_exaggeration,
                                                       float exaggeration,
                                                       float learning_rate,
                                                       int learning_rate_auto,
                                                       float initial_momentum,
                                                       float final_momentum,
                                                       float min_gain,
                                                       float max_step_norm,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       float* out,
                                                       cudaMemcpyKind input_copy_kind) {
  embedding_last_error.clear();
  last_tsne_cuda_graph_capture = false;
  if (indices == nullptr || distances == nullptr || out == nullptr ||
      (has_init && init == nullptr)) {
    set_embedding_error("null CUDA openTSNE input pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_components != 2 ||
      ((!has_init && (pca_init_double != nullptr ||
                      pca_init_float != nullptr ||
                      pca_init_device_row_major != nullptr)) &&
       pca_init_p < n_components) ||
      perplexity <= 0.0f || early_exaggeration_iter < 0 || n_iter < 0 ||
      early_exaggeration_iter + n_iter < 1 ||
      early_exaggeration <= 0.0f || exaggeration <= 0.0f ||
      (!learning_rate_auto && learning_rate <= 0.0f) ||
      initial_momentum < 0.0f || final_momentum < 0.0f || min_gain <= 0.0f) {
    set_embedding_error("invalid CUDA openTSNE FFT-grid dimensions or parameters");
    return 1;
  }

  const int grid_size = resolve_cuda_tsne_fft_grid_size(n);
  const int fft_size = grid_size << 1;
  const int fft_total = fft_size * fft_size;
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t embed_items = static_cast<std::size_t>(n) * 2u;
  const std::size_t embed_bytes = embed_items * sizeof(float);
  const int threads = 256;
  const int point_blocks = (n + threads - 1) / threads;
  const int edge_blocks = (static_cast<int>(input_items) + threads - 1) / threads;
  const int fft_blocks = (fft_total + threads - 1) / threads;
  const int layout_stat_blocks = std::min(1024, std::max(1, point_blocks));
  const std::size_t partial_items = static_cast<std::size_t>(
    std::max(layout_stat_blocks * 4, std::max(point_blocks * 2, point_blocks))
  );
  const std::size_t complex_items = static_cast<std::size_t>(fft_total);
  const std::size_t required_bytes =
    input_items * sizeof(float) +
    embed_items * 4u * sizeof(float) +
    static_cast<std::size_t>(n) * sizeof(float2) +
    complex_items * 9u * sizeof(cufftComplex) +
    partial_items * sizeof(double) +
    4u * sizeof(double);

  const int* d_indices = nullptr;
  const DistanceT* d_distances = nullptr;
  float* d_probabilities = nullptr;
  float* d_current = nullptr;
  float* d_grad = nullptr;
  float* d_gains = nullptr;
  float* d_update = nullptr;
  float2* d_grid_pos = nullptr;
  cufftComplex* d_fft_slab = nullptr;
  cufftComplex* d_mass = nullptr;
  cufftComplex* d_mass_x = nullptr;
  cufftComplex* d_mass_y = nullptr;
  cufftComplex* d_kernel_q = nullptr;
  cufftComplex* d_kernel_q2 = nullptr;
  cufftComplex* d_q = nullptr;
  cufftComplex* d_q2 = nullptr;
  cufftComplex* d_xq2 = nullptr;
  cufftComplex* d_yq2 = nullptr;
  double* d_partial = nullptr;
  double* d_stats = nullptr;
  cufftHandle plan_forward = 0;
  cufftHandle plan_inverse = 0;
  void* d_cufft_work = nullptr;

  const bool input_on_device = input_copy_kind == cudaMemcpyDeviceToDevice;
  auto cleanup = []() {};
  CudaStreamOwner& execution_stream = cuda_execution_stream();
  if (execution_stream.init()) return 1;
  cudaStream_t stream = execution_stream.get();
  CudaFftPlanCache& fft_cache = cuda_fft_plan_cache();
  if (fft_cache.acquire(fft_size, stream)) {
    return 1;
  }
  plan_forward = fft_cache.forward();
  plan_inverse = fft_cache.inverse();
  const std::size_t cufft_work_bytes = fft_cache.work_bytes();

  const bool use_device_pca_init =
    !has_init && (pca_init_double != nullptr || pca_init_float != nullptr ||
                  pca_init_device_row_major != nullptr);
  const std::size_t pca_score_bytes = use_device_pca_init ?
    static_cast<std::size_t>(n) * static_cast<std::size_t>(n_components) * sizeof(float) :
    0u;
  const std::size_t pca_tsvd_workspace_bytes = use_device_pca_init ?
    cuda_pca_workspace_bytes(
      n, pca_init_p, n_components, pca_init_double != nullptr
    ) :
    0u;
  const std::size_t workspace_bytes = required_bytes +
    cufft_work_bytes +
    pca_score_bytes +
    pca_tsvd_workspace_bytes +
    (input_on_device ? 0u : input_items * (sizeof(int) + sizeof(DistanceT))) +
    32u * 256u;
  CudaWorkspace workspace;
  if (check_embedding_memory_available(workspace_bytes, "CUDA openTSNE FFT-grid allocation preflight")) return 1;
  if (workspace.init(workspace_bytes, "opentsne fft-grid")) {
    cleanup();
    return 1;
  }
  if (cufft_work_bytes > 0u) {
    d_cufft_work = workspace.alloc<unsigned char>(cufft_work_bytes, "opentsne cufft work area");
    if (d_cufft_work == nullptr ||
        check_cufft(cufftSetWorkArea(plan_forward, d_cufft_work), "cufftSetWorkArea(opentsne forward)") ||
        check_cufft(cufftSetWorkArea(plan_inverse, d_cufft_work), "cufftSetWorkArea(opentsne inverse)")) {
      cleanup();
      return 1;
    }
  }
  if (input_on_device) {
    d_indices = indices;
    d_distances = distances;
  } else {
    int* d_indices_owned = workspace.alloc<int>(input_items, "opentsne indices");
    DistanceT* d_distances_owned = workspace.alloc<DistanceT>(input_items, "opentsne distances");
    if (d_indices_owned == nullptr || d_distances_owned == nullptr) {
      cleanup();
      return 1;
    }
    d_indices = d_indices_owned;
    d_distances = d_distances_owned;
    if (check_cuda(cudaMemcpy(d_indices_owned, indices, input_items * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy(opentsne indices H2D)") ||
        check_cuda(cudaMemcpy(d_distances_owned, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice),
                   "cudaMemcpy(opentsne distances H2D)")) {
      cleanup();
      return 1;
    }
  }
  d_probabilities = workspace.alloc<float>(input_items, "opentsne probabilities");
  d_current = workspace.alloc<float>(embed_items, "opentsne current");
  d_grad = workspace.alloc<float>(embed_items, "opentsne gradient");
  d_gains = workspace.alloc<float>(embed_items, "opentsne gains");
  d_update = workspace.alloc<float>(embed_items, "opentsne update");
  d_grid_pos = workspace.alloc<float2>(static_cast<std::size_t>(n), "opentsne grid_pos");
  d_fft_slab = workspace.alloc<cufftComplex>(complex_items * 9u, "opentsne batched fft slab");
  if (d_fft_slab != nullptr) {
    d_mass = d_fft_slab;
    d_mass_x = d_fft_slab + complex_items;
    d_mass_y = d_fft_slab + 2u * complex_items;
    d_kernel_q = d_fft_slab + 3u * complex_items;
    d_kernel_q2 = d_fft_slab + 4u * complex_items;
    d_q = d_fft_slab + 5u * complex_items;
    d_q2 = d_fft_slab + 6u * complex_items;
    d_xq2 = d_fft_slab + 7u * complex_items;
    d_yq2 = d_fft_slab + 8u * complex_items;
  }
  d_partial = workspace.alloc<double>(partial_items, "opentsne partial");
  d_stats = workspace.alloc<double>(4u, "opentsne stats");
  if (d_probabilities == nullptr || d_current == nullptr || d_grad == nullptr ||
      d_gains == nullptr || d_update == nullptr || d_grid_pos == nullptr ||
      d_fft_slab == nullptr || d_mass == nullptr || d_mass_x == nullptr || d_mass_y == nullptr ||
      d_kernel_q == nullptr || d_kernel_q2 == nullptr || d_q == nullptr ||
      d_q2 == nullptr || d_xq2 == nullptr || d_yq2 == nullptr ||
      d_partial == nullptr || d_stats == nullptr) {
    cleanup();
    return 1;
  }

  if (has_init) {
    if (check_cuda(cudaMemcpy(d_current, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(opentsne init H2D)")) {
      cleanup();
      return 1;
    }
  } else if (pca_init_double != nullptr || pca_init_float != nullptr ||
             pca_init_device_row_major != nullptr) {
    float* d_scores = workspace.alloc<float>(
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n_components),
      "opentsne cuda pca scores"
    );
    if (d_scores == nullptr) {
      cleanup();
      return 1;
    }
    int pca_status = 0;
    if (pca_init_device_row_major != nullptr) {
      pca_status = cuda_pca_scores_to_device<float>(
        nullptr, n, pca_init_p, n_components, d_scores, &workspace,
        true, false, nullptr, nullptr, nullptr, nullptr,
        0, seed, 16, 2, nullptr, pca_init_device_row_major
      );
    } else if (pca_init_double != nullptr) {
      pca_status = cuda_pca_scores_to_device<double>(
        pca_init_double, n, pca_init_p, n_components, d_scores, &workspace
      );
    } else {
      pca_status = cuda_pca_scores_to_device<float>(
        pca_init_float, n, pca_init_p, n_components, d_scores, &workspace
      );
    }
    if (pca_status != 0) {
      cleanup();
      return 1;
    }
    tsvd_scores_to_rowmajor_init_kernel<<<point_blocks, threads>>>(
      d_scores, d_current, n, n_components
    );
    if (check_cuda(cudaGetLastError(), "tsvd_scores_to_rowmajor_init_kernel launch")) {
      cleanup();
      return 1;
    }
    if (scale_tsne_pca_device_init(d_current, d_partial, d_stats, n, layout_stat_blocks, threads)) {
      cleanup();
      return 1;
    }
  } else {
    random_init_kernel<<<point_blocks, threads>>>(d_current, n, seed);
    if (check_cuda(cudaGetLastError(), "random_init_kernel(opentsne) launch")) {
      cleanup();
      return 1;
    }
  }
  if (check_cuda(cudaMemset(d_update, 0, embed_bytes), "cudaMemset(opentsne update)")) {
    cleanup();
    return 1;
  }
  fill_float_kernel<<<(static_cast<int>(embed_items) + threads - 1) / threads, threads>>>(
    d_gains, static_cast<int>(embed_items), 1.0f
  );
  if (check_cuda(cudaGetLastError(), "fill_float_kernel(opentsne gains) launch")) {
    cleanup();
    return 1;
  }

  opentsne_affinity_sparse_kernel<<<point_blocks, threads>>>(
    d_indices, d_distances, d_probabilities, n, k, index_offset, perplexity
  );
  if (check_cuda(cudaGetLastError(), "opentsne_affinity_sparse_kernel launch")) {
    cleanup();
    return 1;
  }

  if (check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize(opentsne setup before graph capture)"
      )) {
    cleanup();
    return 1;
  }

  const int total_iter = early_exaggeration_iter + n_iter;
  const float fft_scale = 1.0f / static_cast<float>(fft_total);

  auto encode_iteration = [&](float current_exaggeration,
                              float current_momentum,
                              float current_learning_rate,
                              bool check_launches) -> int {
    opentsne_layout_stats_blocks_kernel<<<layout_stat_blocks, threads, 4u * threads * sizeof(double), stream>>>(
      d_current, d_partial, n
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_layout_stats_blocks_kernel launch")) {
      return 1;
    }
    opentsne_finalize_layout_stats_kernel<<<1, threads, 4u * threads * sizeof(double), stream>>>(
      d_partial, d_stats, layout_stat_blocks, grid_size
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_finalize_layout_stats_kernel launch")) {
      return 1;
    }

    opentsne_fft_clear_kernel<<<fft_blocks, threads, 0, stream>>>(
      d_mass, d_mass_x, d_mass_y, d_kernel_q, d_kernel_q2,
      d_stats, fft_size, grid_size
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_clear_kernel launch")) {
      return 1;
    }
    opentsne_fft_scatter_kernel<<<point_blocks, threads, 0, stream>>>(
      d_current, d_mass, d_mass_x, d_mass_y, d_grid_pos,
      d_stats, n, grid_size, fft_size
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_scatter_kernel launch")) {
      return 1;
    }

    if (check_cufft(cufftExecC2C(plan_forward, d_fft_slab, d_fft_slab, CUFFT_FORWARD),
                    "cufftExecC2C(opentsne forward batch)")) {
      return 1;
    }

    opentsne_fft_multiply_all_kernel<<<fft_blocks, threads, 0, stream>>>(
      d_mass, d_mass_x, d_mass_y, d_kernel_q, d_kernel_q2,
      d_q, d_q2, d_xq2, d_yq2, fft_total
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_multiply_all_kernel launch")) {
      return 1;
    }

    if (check_cufft(cufftExecC2C(plan_inverse, d_q, d_q, CUFFT_INVERSE),
                    "cufftExecC2C(opentsne inverse batch)")) {
      return 1;
    }

    opentsne_fft_sum_q_kernel<<<point_blocks, threads, threads * sizeof(double), stream>>>(
      d_q, d_grid_pos, d_partial, n, fft_size, grid_size, fft_scale
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_sum_q_kernel launch")) {
      return 1;
    }
    opentsne_fft_finalize_sum_q_kernel<<<1, threads, threads * sizeof(double), stream>>>(
      d_partial, d_stats, point_blocks
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_finalize_sum_q_kernel launch")) {
      return 1;
    }
    opentsne_fft_repulsive_gradient_kernel<<<point_blocks, threads, 0, stream>>>(
      d_current, d_grad, d_q2, d_xq2, d_yq2, d_grid_pos, d_stats,
      n, fft_size, grid_size, fft_scale
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_fft_repulsive_gradient_kernel launch")) {
      return 1;
    }
    opentsne_sparse_attractive_kernel<<<edge_blocks, threads, 0, stream>>>(
      d_indices, d_probabilities, d_current, d_grad, n, k, index_offset, current_exaggeration
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_sparse_attractive_kernel launch")) {
      return 1;
    }
    opentsne_update_reduce_kernel<<<point_blocks, threads, 2u * threads * sizeof(double), stream>>>(
      d_current, d_grad, d_gains, d_update, d_partial, n,
      current_learning_rate, current_momentum, min_gain, max_step_norm
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "opentsne_update_reduce_kernel launch")) {
      return 1;
    }
    tsne_finalize_mean_kernel<<<1, threads, 2u * threads * sizeof(double), stream>>>(
      d_partial, d_stats, point_blocks, n
    );
    if (check_launches &&
        check_cuda(cudaGetLastError(), "tsne_finalize_mean_kernel(opentsne) launch")) {
      return 1;
    }
    tsne_center_kernel<<<point_blocks, threads, 0, stream>>>(d_current, d_stats, n);
    if (check_launches &&
        check_cuda(cudaGetLastError(), "tsne_center_kernel(opentsne) launch")) {
      return 1;
    }
    return 0;
  };

  auto capture_iteration_chunk = [&](float current_exaggeration,
                                     float current_momentum,
                                     float current_learning_rate,
                                     int repetitions,
                                     CudaGraphExecOwner& target) -> int {
    if (repetitions < 1) return 1;
    cudaGraph_t graph = nullptr;
    cudaError_t code = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (code != cudaSuccess) return 1;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      if (encode_iteration(current_exaggeration, current_momentum, current_learning_rate, false)) {
        cudaStreamEndCapture(stream, &graph);
        if (graph != nullptr) cudaGraphDestroy(graph);
        return 1;
      }
    }
    code = cudaStreamEndCapture(stream, &graph);
    if (code != cudaSuccess || graph == nullptr) {
      if (graph != nullptr) cudaGraphDestroy(graph);
      cudaGetLastError();
      return 1;
    }
    if (target.instantiate(graph)) {
      cudaGetLastError();
      return 1;
    }
    return 0;
  };

  const float normal_exaggeration = exaggeration;
  const float early_learning_rate = learning_rate_auto ?
    static_cast<float>(n) / early_exaggeration :
    learning_rate;
  const float normal_learning_rate = learning_rate_auto ?
    static_cast<float>(n) / normal_exaggeration :
    learning_rate;
  constexpr int graph_chunk_limit = 25;
  const int early_graph_iterations = std::min(graph_chunk_limit, early_exaggeration_iter);
  const int normal_graph_iterations = std::min(graph_chunk_limit, n_iter);
  const int early_graph_launches = early_graph_iterations > 0 ?
    early_exaggeration_iter / early_graph_iterations : 0;
  const int normal_graph_launches = normal_graph_iterations > 0 ?
    n_iter / normal_graph_iterations : 0;
  const int early_tail_iterations = early_graph_iterations > 0 ?
    early_exaggeration_iter % early_graph_iterations : 0;
  const int normal_tail_iterations = normal_graph_iterations > 0 ?
    n_iter % normal_graph_iterations : 0;
  bool used_graph_capture = false;
  CudaGraphExecOwner early_graph;
  CudaGraphExecOwner early_tail_graph;
  CudaGraphExecOwner normal_graph;
  CudaGraphExecOwner normal_tail_graph;
  if (cuda_graph_capture_enabled() && total_iter > 1) {
    bool capture_ok = true;
    if (early_exaggeration_iter > 0) {
      capture_ok = capture_iteration_chunk(
        early_exaggeration, initial_momentum, early_learning_rate,
        early_graph_iterations, early_graph
      ) == 0;
      if (capture_ok && early_tail_iterations > 0) {
        capture_ok = capture_iteration_chunk(
          early_exaggeration, initial_momentum, early_learning_rate,
          early_tail_iterations, early_tail_graph
        ) == 0;
      }
    }
    if (capture_ok && n_iter > 0) {
      capture_ok = capture_iteration_chunk(
        normal_exaggeration, final_momentum, normal_learning_rate,
        normal_graph_iterations, normal_graph
      ) == 0;
      if (capture_ok && normal_tail_iterations > 0) {
        capture_ok = capture_iteration_chunk(
          normal_exaggeration, final_momentum, normal_learning_rate,
          normal_tail_iterations, normal_tail_graph
        ) == 0;
      }
    }
    if (capture_ok) {
      used_graph_capture = true;
      for (int launch = 0; launch < early_graph_launches; ++launch) {
        if (check_cuda(cudaGraphLaunch(early_graph.get(), stream),
                       "cudaGraphLaunch(opentsne early chunk)")) {
          cleanup();
          return 1;
        }
      }
      if (early_tail_iterations > 0 &&
          check_cuda(cudaGraphLaunch(early_tail_graph.get(), stream),
                     "cudaGraphLaunch(opentsne early tail)")) {
        cleanup();
        return 1;
      }
      for (int launch = 0; launch < normal_graph_launches; ++launch) {
        if (check_cuda(cudaGraphLaunch(normal_graph.get(), stream),
                       "cudaGraphLaunch(opentsne normal chunk)")) {
          cleanup();
          return 1;
        }
      }
      if (normal_tail_iterations > 0 &&
          check_cuda(cudaGraphLaunch(normal_tail_graph.get(), stream),
                     "cudaGraphLaunch(opentsne normal tail)")) {
        cleanup();
        return 1;
      }
    }
  }

  if (!used_graph_capture) {
    for (int iter = 0; iter < total_iter; ++iter) {
      const float current_exaggeration = iter < early_exaggeration_iter ? early_exaggeration : exaggeration;
      const float current_momentum = iter < early_exaggeration_iter ? initial_momentum : final_momentum;
      const float current_learning_rate = learning_rate_auto ?
        static_cast<float>(n) / current_exaggeration :
        learning_rate;
      if (encode_iteration(current_exaggeration, current_momentum, current_learning_rate, true)) {
        cleanup();
        return 1;
      }
    }
  }

  last_tsne_cuda_graph_capture = used_graph_capture;

  if (check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(opentsne fft)") ||
      check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(opentsne fft D2H)")) {
    cleanup();
    return 1;
  }
  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_opentsne_fft_from_knn(const int* indices,
                                                       const double* distances,
                                                       const float* init,
                                                       int has_init,
                                                       int n,
                                                       int k,
                                                       int n_components,
                                                       float perplexity,
                                                       int early_exaggeration_iter,
                                                       int n_iter,
                                                       float early_exaggeration,
                                                       float exaggeration,
                                                       float learning_rate,
                                                       int learning_rate_auto,
                                                       float initial_momentum,
                                                       float final_momentum,
                                                       float min_gain,
                                                       float max_step_norm,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<double>(
    indices, distances, init, has_init, nullptr, nullptr, nullptr,
    0, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out, cudaMemcpyHostToDevice
  );
}

extern "C" int fastembedr_cuda_opentsne_fft_from_knn_float(const int* indices,
                                                            const float* distances,
                                                            const float* init,
                                                            int has_init,
                                                            int n,
                                                            int k,
                                                            int n_components,
                                                            float perplexity,
                                                            int early_exaggeration_iter,
                                                            int n_iter,
                                                            float early_exaggeration,
                                                            float exaggeration,
                                                            float learning_rate,
                                                            int learning_rate_auto,
                                                            float initial_momentum,
                                                            float final_momentum,
                                                            float min_gain,
                                                            float max_step_norm,
                                                            unsigned int seed,
                                                            int index_offset,
                                                            float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<float>(
    indices, distances, init, has_init, nullptr, nullptr, nullptr,
    0, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out, cudaMemcpyHostToDevice
  );
}

extern "C" int fastembedr_cuda_opentsne_fft_from_device_knn_float(const int* indices,
                                                                  const float* distances,
                                                                  const float* init,
                                                                  int has_init,
                                                                  int n,
                                                                  int k,
                                                                  int n_components,
                                                                  float perplexity,
                                                                  int early_exaggeration_iter,
                                                                  int n_iter,
                                                                  float early_exaggeration,
                                                                  float exaggeration,
                                                                  float learning_rate,
                                                                  int learning_rate_auto,
                                                                  float initial_momentum,
                                                                  float final_momentum,
                                                                  float min_gain,
                                                                  float max_step_norm,
                                                                  unsigned int seed,
                                                                  int index_offset,
                                                                  float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<float>(
    indices, distances, init, has_init, nullptr, nullptr, nullptr,
    0, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out, cudaMemcpyDeviceToDevice
  );
}

extern "C" int fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_double(const int* indices,
                                                                              const float* distances,
                                                                              const double* pca_init_values,
                                                                              int pca_init_p,
                                                                              int n,
                                                                              int k,
                                                                              int n_components,
                                                                              float perplexity,
                                                                              int early_exaggeration_iter,
                                                                              int n_iter,
                                                                              float early_exaggeration,
                                                                              float exaggeration,
                                                                              float learning_rate,
                                                                              int learning_rate_auto,
                                                                              float initial_momentum,
                                                                              float final_momentum,
                                                                              float min_gain,
                                                                              float max_step_norm,
                                                                              unsigned int seed,
                                                                              int index_offset,
                                                                              float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<float>(
    indices, distances, nullptr, 0, pca_init_values, nullptr, nullptr,
    pca_init_p, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out, cudaMemcpyDeviceToDevice
  );
}

extern "C" int fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_float(const int* indices,
                                                                             const float* distances,
                                                                             const float* pca_init_values,
                                                                             int pca_init_p,
                                                                             int n,
                                                                             int k,
                                                                             int n_components,
                                                                             float perplexity,
                                                                             int early_exaggeration_iter,
                                                                             int n_iter,
                                                                             float early_exaggeration,
                                                                             float exaggeration,
                                                                             float learning_rate,
                                                                             int learning_rate_auto,
                                                                             float initial_momentum,
                                                                             float final_momentum,
                                                                             float min_gain,
                                                                             float max_step_norm,
                                                                             unsigned int seed,
                                                                             int index_offset,
                                                                             float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<float>(
    indices, distances, nullptr, 0, nullptr, pca_init_values, nullptr,
    pca_init_p, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out, cudaMemcpyDeviceToDevice
  );
}

extern "C" int
fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_device(
    const int* indices,
    const float* distances,
    const float* pca_init_values,
    int pca_init_p,
    int n,
    int k,
    int n_components,
    float perplexity,
    int early_exaggeration_iter,
    int n_iter,
    float early_exaggeration,
    float exaggeration,
    float learning_rate,
    int learning_rate_auto,
    float initial_momentum,
    float final_momentum,
    float min_gain,
    float max_step_norm,
    unsigned int seed,
    int index_offset,
    float* out) {
  return fastembedr_cuda_opentsne_fft_from_knn_impl<float>(
    indices, distances, nullptr, 0, nullptr, nullptr, pca_init_values,
    pca_init_p, n, k, n_components, perplexity,
    early_exaggeration_iter, n_iter, early_exaggeration, exaggeration,
    learning_rate, learning_rate_auto, initial_momentum, final_momentum,
    min_gain, max_step_norm, seed, index_offset, out,
    cudaMemcpyDeviceToDevice
  );
}

template <typename DistanceT>
int fastembedr_cuda_exact_tsne_from_knn_impl(const int* indices,
                                                   const DistanceT* distances,
                                                   const float* init,
                                                   int n,
                                                   int k,
                                                   int n_epochs,
                                                   float perplexity,
                                                   float learning_rate,
                                                   int stop_lying_iter,
                                                   int mom_switch_iter,
                                                   float momentum,
                                                   float final_momentum,
                                                   float exaggeration_factor,
                                                   unsigned int seed,
                                                   int index_offset,
                                                   float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 ||
      perplexity <= 0.0f || learning_rate <= 0.0f ||
      stop_lying_iter < 0 || mom_switch_iter < 0 ||
      momentum < 0.0f || final_momentum < 0.0f ||
      exaggeration_factor <= 0.0f) {
    set_embedding_error("invalid exact CUDA t-SNE dimensions or parameters");
    return 1;
  }

  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t dense_items = static_cast<std::size_t>(n) * n;
  const std::size_t embed_items = static_cast<std::size_t>(n) * 2u;
  const std::size_t embed_bytes = embed_items * sizeof(float);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const std::size_t partial_items = static_cast<std::size_t>(blocks) * 4u;
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(DistanceT)) +
    dense_items * sizeof(float) +
    embed_items * 4u * sizeof(float) +
    partial_items * sizeof(double) +
    4u * sizeof(double);

  int* d_indices = nullptr;
  DistanceT* d_distances = nullptr;
  float* d_affinities = nullptr;
  float* d_current = nullptr;
  float* d_grad = nullptr;
  float* d_gains = nullptr;
  float* d_inc = nullptr;
  double* d_partial = nullptr;
  double* d_stats = nullptr;

  auto cleanup = [&]() {
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_affinities != nullptr) cudaFree(d_affinities);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_grad != nullptr) cudaFree(d_grad);
    if (d_gains != nullptr) cudaFree(d_gains);
    if (d_inc != nullptr) cudaFree(d_inc);
    if (d_partial != nullptr) cudaFree(d_partial);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA exact t-SNE allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(exact tsne indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(DistanceT)), "cudaMalloc(exact tsne distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_affinities), dense_items * sizeof(float)), "cudaMalloc(exact tsne affinities)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(exact tsne current)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_grad), embed_bytes), "cudaMalloc(exact tsne gradient)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_gains), embed_bytes), "cudaMalloc(exact tsne gains)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_inc), embed_bytes), "cudaMalloc(exact tsne increments)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_partial), partial_items * sizeof(double)), "cudaMalloc(exact tsne partials)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), 4u * sizeof(double)), "cudaMalloc(exact tsne stats)")) {
    cleanup();
    return 1;
  }

  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(exact tsne indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice), "cudaMemcpy(exact tsne distances H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_current, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(exact tsne init H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemset(d_affinities, 0, dense_items * sizeof(float)), "cudaMemset(exact tsne affinities)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemset(d_inc, 0, embed_bytes), "cudaMemset(exact tsne increments)")) {
    cleanup();
    return 1;
  }

  tsne_affinity_from_knn_kernel<<<blocks, threads>>>(
    d_indices, d_distances, d_affinities, n, k, index_offset, perplexity
  );
  if (check_cuda(cudaGetLastError(), "tsne_affinity_from_knn_kernel launch")) {
    cleanup();
    return 1;
  }

  tsne_init_stats_kernel<<<blocks, threads, 4u * threads * sizeof(double)>>>(
    d_current, d_partial, n
  );
  if (check_cuda(cudaGetLastError(), "tsne_init_stats_kernel launch")) {
    cleanup();
    return 1;
  }
  tsne_finalize_init_stats_kernel<<<1, threads, 4u * threads * sizeof(double)>>>(
    d_partial, d_stats, blocks, n
  );
  if (check_cuda(cudaGetLastError(), "tsne_finalize_init_stats_kernel launch")) {
    cleanup();
    return 1;
  }
  tsne_scale_init_kernel<<<blocks, threads>>>(d_current, d_stats, n);
  if (check_cuda(cudaGetLastError(), "tsne_scale_init_kernel launch")) {
    cleanup();
    return 1;
  }

  fill_float_kernel<<<(static_cast<int>(embed_items) + threads - 1) / threads, threads>>>(
    d_gains, static_cast<int>(embed_items), 1.0f
  );
  if (check_cuda(cudaGetLastError(), "fill_float_kernel(exact tsne gains) launch")) {
    cleanup();
    return 1;
  }

  const float eta = learning_rate;
  for (int epoch = 1; epoch <= n_epochs; ++epoch) {
    tsne_sum_q_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_current, d_partial, n);
    if (check_cuda(cudaGetLastError(), "tsne_sum_q_kernel launch")) {
      cleanup();
      return 1;
    }
    tsne_finalize_sum_kernel<<<1, threads, threads * sizeof(double)>>>(d_partial, d_stats, blocks);
    if (check_cuda(cudaGetLastError(), "tsne_finalize_sum_kernel launch")) {
      cleanup();
      return 1;
    }
    const float exaggeration = epoch <= stop_lying_iter ? exaggeration_factor : 1.0f;
    tsne_exact_gradient_kernel<<<blocks, threads>>>(
      d_current, d_affinities, d_grad, d_stats, n, exaggeration
    );
    if (check_cuda(cudaGetLastError(), "tsne_exact_gradient_kernel launch")) {
      cleanup();
      return 1;
    }
    const float current_momentum = epoch <= mom_switch_iter ? momentum : final_momentum;
    tsne_update_reduce_kernel<<<blocks, threads, 2u * threads * sizeof(double)>>>(
      d_current, d_grad, d_gains, d_inc, d_partial, n, eta, current_momentum
    );
    if (check_cuda(cudaGetLastError(), "tsne_update_reduce_kernel launch")) {
      cleanup();
      return 1;
    }
    tsne_finalize_mean_kernel<<<1, threads, 2u * threads * sizeof(double)>>>(
      d_partial, d_stats, blocks, n
    );
    if (check_cuda(cudaGetLastError(), "tsne_finalize_mean_kernel launch")) {
      cleanup();
      return 1;
    }
    tsne_center_kernel<<<blocks, threads>>>(d_current, d_stats, n);
    if (check_cuda(cudaGetLastError(), "tsne_center_kernel launch")) {
      cleanup();
      return 1;
    }
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(exact tsne)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(exact tsne D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_exact_tsne_from_knn(const int* indices,
                                                   const double* distances,
                                                   const float* init,
                                                   int n,
                                                   int k,
                                                   int n_epochs,
                                                   float perplexity,
                                                   float learning_rate,
                                                   int stop_lying_iter,
                                                   int mom_switch_iter,
                                                   float momentum,
                                                   float final_momentum,
                                                   float exaggeration_factor,
                                                   unsigned int seed,
                                                   int index_offset,
                                                   float* out) {
  return fastembedr_cuda_exact_tsne_from_knn_impl<double>(
    indices, distances, init, n, k, n_epochs, perplexity, learning_rate,
    stop_lying_iter, mom_switch_iter, momentum, final_momentum,
    exaggeration_factor, seed, index_offset, out
  );
}

extern "C" int fastembedr_cuda_embed(const int* neighbors,
                                      const float* weights,
                                      const float* init,
                                      int n,
                                      int k,
                                      int objective,
                                      int n_epochs,
                                      int negative_sample_rate,
                                      float learning_rate,
                                      float a,
                                      float b,
                                      float max_weight,
                                      unsigned int seed,
                                      float* out) {
  embedding_last_error.clear();
  if (neighbors == nullptr || weights == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 || negative_sample_rate < 0 || learning_rate <= 0.0f) {
    set_embedding_error("invalid embedding dimensions or parameters");
    return 1;
  }

  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
  const std::size_t knn_bytes = static_cast<std::size_t>(n) * k;
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const std::size_t required_bytes =
    knn_bytes * (sizeof(int) + sizeof(float)) + 2u * embed_bytes;

  auto cleanup = [&]() {
    if (d_neighbors != nullptr) cudaFree(d_neighbors);
    if (d_weights != nullptr) cudaFree(d_weights);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_next != nullptr) cudaFree(d_next);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA embedding allocation preflight")) {
    return 1;
  }

  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_neighbors), knn_bytes * sizeof(int)), "cudaMalloc(neighbors)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), knn_bytes * sizeof(float)), "cudaMalloc(weights)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(current)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next), embed_bytes), "cudaMalloc(next)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_neighbors, neighbors, knn_bytes * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(neighbors H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_weights, weights, knn_bytes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(weights H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_current, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(init H2D)")) {
    cleanup();
    return 1;
  }

  const EmbedParams params{
    n, k, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, max_weight, 1.0f
  };
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    embed_epoch_kernel<<<blocks, threads>>>(
      d_current,
      d_next,
      d_neighbors,
      d_weights,
      params,
      static_cast<unsigned int>(epoch)
    );
    if (check_cuda(cudaGetLastError(), "embed_epoch_kernel launch")) {
      cleanup();
      return 1;
    }
    std::swap(d_current, d_next);
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    cleanup();
    return 1;
  }

  if (check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(embedding D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

template <typename DistanceT>
int fastembedr_cuda_embed_from_knn_impl(const int* indices,
                                               const DistanceT* distances,
                                               const float* init,
                                               int n,
                                               int k,
                                               int objective,
                                               int n_epochs,
                                               int negative_sample_rate,
                                               float learning_rate,
                                               float a,
                                               float b,
                                               unsigned int seed,
                                               int index_offset,
                                               float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 || negative_sample_rate < 0 || learning_rate <= 0.0f) {
    set_embedding_error("invalid KNN embedding dimensions or parameters");
    return 1;
  }
  const int width = objective == 0 ? std::min(256, std::max(k, 2 * k)) : k;
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(DistanceT)) +
    graph_items * (sizeof(int) + sizeof(float)) +
    (objective == 0 ? static_cast<std::size_t>(n) * sizeof(int) : 0u) +
    2u * embed_bytes;

  int* d_indices = nullptr;
  DistanceT* d_distances = nullptr;
  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
  const int threads = 256;

  auto cleanup = [&]() {
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_neighbors != nullptr) cudaFree(d_neighbors);
    if (d_weights != nullptr) cudaFree(d_weights);
    if (d_counts != nullptr) cudaFree(d_counts);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_next != nullptr) cudaFree(d_next);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA KNN embedding allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(knn indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(DistanceT)), "cudaMalloc(knn distances)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_neighbors), graph_items * sizeof(int)), "cudaMalloc(knn neighbors)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), graph_items * sizeof(float)), "cudaMalloc(knn weights)")) {
    cleanup();
    return 1;
  }
  if (objective == 0 &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_counts), static_cast<std::size_t>(n) * sizeof(int)), "cudaMalloc(knn row counts)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(current)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next), embed_bytes), "cudaMalloc(next)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(indices H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(DistanceT), cudaMemcpyHostToDevice), "cudaMemcpy(distances H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_current, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(init H2D)")) {
    cleanup();
    return 1;
  }

  if (prepare_device_knn_graph(
        d_indices, d_distances, d_neighbors, d_weights, d_counts,
        n, k, width, index_offset, objective)) {
    cleanup();
    return 1;
  }

  if (d_indices != nullptr) {
    cudaFree(d_indices);
    d_indices = nullptr;
  }
  if (d_distances != nullptr) {
    cudaFree(d_distances);
    d_distances = nullptr;
  }

  const float graph_max_weight = 1.0f;
  const EmbedParams params{
    n, width, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, graph_max_weight, 1.0f
  };
  const int blocks = (n + threads - 1) / threads;
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    embed_epoch_kernel<<<blocks, threads>>>(
      d_current,
      d_next,
      d_neighbors,
      d_weights,
      params,
      static_cast<unsigned int>(epoch)
    );
    if (check_cuda(cudaGetLastError(), "embed_epoch_kernel(from KNN) launch")) {
      cleanup();
      return 1;
    }
    std::swap(d_current, d_next);
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(from KNN)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(embedding D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_embed_from_knn(const int* indices,
                                               const double* distances,
                                               const float* init,
                                               int n,
                                               int k,
                                               int objective,
                                               int n_epochs,
                                               int negative_sample_rate,
                                               float learning_rate,
                                               float a,
                                               float b,
                                               unsigned int seed,
                                               int index_offset,
                                               float* out) {
  return fastembedr_cuda_embed_from_knn_impl<double>(
    indices, distances, init, n, k, objective, n_epochs,
    negative_sample_rate, learning_rate, a, b, seed, index_offset, out
  );
}

extern "C" int fastembedr_cuda_matrix_multiply(const double* left,
                                                const double* right,
                                                int left_rows,
                                                int left_cols,
                                                int right_cols,
                                                int transpose_left,
                                                double* out) {
  embedding_last_error.clear();
  if (left == nullptr || right == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (left_rows < 1 || left_cols < 1 || right_cols < 1) {
    set_embedding_error("invalid matrix multiply dimensions");
    return 1;
  }

  const int out_rows = transpose_left ? left_cols : left_rows;
  const int right_rows = transpose_left ? left_rows : left_cols;
  const std::size_t left_items = static_cast<std::size_t>(left_rows) * left_cols;
  const std::size_t right_items = static_cast<std::size_t>(right_rows) * right_cols;
  const std::size_t out_items = static_cast<std::size_t>(out_rows) * right_cols;
  const std::size_t required_bytes =
    (left_items + right_items + out_items) * sizeof(double);

  double* d_left = nullptr;
  double* d_right = nullptr;
  double* d_out = nullptr;
  auto cleanup = [&]() {
    if (d_left != nullptr) cudaFree(d_left);
    if (d_right != nullptr) cudaFree(d_right);
    if (d_out != nullptr) cudaFree(d_out);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA RSVD matrix multiply allocation preflight")) {
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_left), left_items * sizeof(double)), "cudaMalloc(rsvd left)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_right), right_items * sizeof(double)), "cudaMalloc(rsvd right)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), out_items * sizeof(double)), "cudaMalloc(rsvd out)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_left, left, left_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(rsvd left H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_right, right, right_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(rsvd right H2D)")) {
    cleanup();
    return 1;
  }

  const dim3 threads(16, 16);
  const dim3 blocks(
    static_cast<unsigned int>((right_cols + threads.x - 1) / threads.x),
    static_cast<unsigned int>((out_rows + threads.y - 1) / threads.y)
  );
  matrix_multiply_kernel<<<blocks, threads>>>(
    d_left,
    d_right,
    d_out,
    left_rows,
    left_cols,
    right_cols,
    transpose_left ? 1 : 0
  );
  if (check_cuda(cudaGetLastError(), "matrix_multiply_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(rsvd multiply)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out, d_out, out_items * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(rsvd out D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}
