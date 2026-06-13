#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr float kCudaFloatInf = 3.4028234663852886e+38F;
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
};

struct KnnPrepParams {
  int n;
  int k;
  int width;
  int offset;
};

thread_local std::string embedding_last_error;

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

int check_cufft(cufftResult code, const char* where) {
  if (code == CUFFT_SUCCESS) return 0;
  set_embedding_error(std::string(where) + ": cuFFT error " + std::to_string(static_cast<int>(code)));
  return 1;
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

__device__ unsigned int mix_uint(unsigned int x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ unsigned int deterministic_vertex(unsigned int n,
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

__device__ float deterministic_unit(unsigned int seed,
                                    unsigned int row,
                                    unsigned int col) {
  unsigned int x = seed;
  x ^= (row + 1u) * 0x9e3779b9u;
  x ^= (col + 1u) * 0x85ebca6bu;
  const unsigned int bits = mix_uint(x);
  return static_cast<float>(bits & 0x00ffffffu) / 8388607.5f - 1.0f;
}

__device__ float clip4(float x) {
  return fminf(4.0f, fmaxf(-4.0f, x));
}

__device__ bool pair_less_device(double a_dist,
                                 int a_idx,
                                 double b_dist,
                                 int b_idx) {
  return (a_dist < b_dist) || (a_dist == b_dist && a_idx < b_idx);
}

__device__ double layout_d2_2d(const double* layout, int n, int i, int j) {
  const double dx = layout[i] - layout[j];
  const double dy = layout[static_cast<std::size_t>(n) + i] -
    layout[static_cast<std::size_t>(n) + j];
  return dx * dx + dy * dy;
}

__device__ double median_device(double* values, int count) {
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

__device__ void insert_top_neighbor(double* top_dist,
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

__device__ float attractive_coeff(float d2, float weight, const EmbedParams p) {
  if (p.objective == 0) {
    if (d2 <= 0.0f) return 0.0f;
    const float d2b = powf(d2, p.b);
    return -2.0f * p.a * p.b * (d2b / d2) / (p.a * d2b + 1.0f);
  }
  if (p.objective == 1) return -2.0f * weight / (1.0f + d2);
  if (p.objective == 2) return -2.0f * weight / (10.0f + d2);
  if (p.objective == 4) return -2.5f * weight / (0.15f + d2);
  return -2.0f * weight / (1.0f + d2);
}

__device__ float repulsive_coeff(float d2, const EmbedParams p) {
  if (d2 <= 0.0f) return 0.0f;
  if (p.objective == 0) {
    const float d2b = powf(d2, p.b);
    return 2.0f * p.b / ((0.001f + d2) * (p.a * d2b + 1.0f));
  }
  if (p.objective == 1) return 2.0f / ((1.0f + d2) * (1.0f + d2));
  if (p.objective == 2) return 0.4f / (1.0f + d2);
  if (p.objective == 4) return 0.8125f / ((0.15f + d2) * (1.0f + d2));
  return 2.0f / (1.0f + d2);
}

__device__ int positive_samples_this_epoch(float weight, const EmbedParams p, unsigned int epoch) {
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

__device__ int positive_samples_this_epoch_period(float period, unsigned int epoch) {
  if (period <= 0.0f || !isfinite(period)) return 0;
  const float now = static_cast<float>(epoch + 1u);
  const float previous = static_cast<float>(epoch);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  const int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

__device__ int cumulative_umap_negative_samples(float active_epoch, float negative_period) {
  if (negative_period <= 0.0f || !isfinite(negative_period)) return 0;
  const int samples = static_cast<int>(
    floorf(((active_epoch - negative_period) / negative_period) + 1.0e-6f)
  );
  return samples > 0 ? samples : 0;
}

__device__ int negative_samples_this_epoch(float weight, const EmbedParams p, unsigned int epoch) {
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

__device__ int negative_samples_this_epoch_period(float period, const EmbedParams p, unsigned int epoch) {
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

__global__ void prepare_directed_knn_kernel(const int* indices,
                                            const double* distances,
                                            int* neighbors,
                                            float* weights,
                                            KnnPrepParams p) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= p.n) return;

  float rho = kCudaFloatInf;
  for (int j = 0; j < p.k; ++j) {
    const double d0 = distances[static_cast<std::size_t>(j) * p.n + row];
    const float d = static_cast<float>(d0);
    if (d > 0.0f && d < rho) rho = d;
  }
  if (!isfinite(rho)) rho = 0.0f;

  const float target = log2f(static_cast<float>(p.k < 2 ? 2 : p.k));
  float lo = 0.0f;
  float hi = kCudaFloatInf;
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

__device__ float reverse_directed_weight(const int* neighbors,
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

__device__ bool direct_contains(const int* neighbors,
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

__global__ void fill_float_kernel(float* values,
                                  int n,
                                  float value) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid < n) values[gid] = value;
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

__device__ int high_rank_device(const int* indices,
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
                                               floor(static_cast<double>(valid) / 3.0)));
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
      const bool sign_changed = (g > 0.0f) != (old_inc > 0.0f);
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

__global__ void opentsne_affinity_sparse_kernel(const int* indices,
                                                const double* distances,
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
                                               floor(static_cast<double>(valid) / 3.0)));
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
  double* sum_x = max_y + blockDim.x;
  double* sum_y = sum_x + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double mnx = kCudaDoubleInf;
  double mxx = -kCudaDoubleInf;
  double mny = kCudaDoubleInf;
  double mxy = -kCudaDoubleInf;
  double sx = 0.0;
  double sy = 0.0;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       i < n;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    const double x = current[static_cast<std::size_t>(i) * 2u];
    const double y = current[static_cast<std::size_t>(i) * 2u + 1u];
    mnx = fmin(mnx, x);
    mxx = fmax(mxx, x);
    mny = fmin(mny, y);
    mxy = fmax(mxy, y);
    sx += x;
    sy += y;
  }
  min_x[tid] = mnx;
  max_x[tid] = mxx;
  min_y[tid] = mny;
  max_y[tid] = mxy;
  sum_x[tid] = sx;
  sum_y[tid] = sy;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      min_x[tid] = fmin(min_x[tid], min_x[tid + stride]);
      max_x[tid] = fmax(max_x[tid], max_x[tid + stride]);
      min_y[tid] = fmin(min_y[tid], min_y[tid + stride]);
      max_y[tid] = fmax(max_y[tid], max_y[tid + stride]);
      sum_x[tid] += sum_x[tid + stride];
      sum_y[tid] += sum_y[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 6u;
    partial[base] = min_x[0];
    partial[base + 1u] = max_x[0];
    partial[base + 2u] = min_y[0];
    partial[base + 3u] = max_y[0];
    partial[base + 4u] = sum_x[0];
    partial[base + 5u] = sum_y[0];
  }
}

__global__ void opentsne_finalize_layout_stats_kernel(const double* partial,
                                                      double* stats,
                                                      int n_blocks,
                                                      int n,
                                                      int grid_size) {
  extern __shared__ double shared[];
  double* min_x = shared;
  double* max_x = min_x + blockDim.x;
  double* min_y = max_x + blockDim.x;
  double* max_y = min_y + blockDim.x;
  double* sum_x = max_y + blockDim.x;
  double* sum_y = sum_x + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  double mnx = kCudaDoubleInf;
  double mxx = -kCudaDoubleInf;
  double mny = kCudaDoubleInf;
  double mxy = -kCudaDoubleInf;
  double sx = 0.0;
  double sy = 0.0;
  for (int b = tid; b < n_blocks; b += static_cast<int>(blockDim.x)) {
    const std::size_t base = static_cast<std::size_t>(b) * 6u;
    mnx = fmin(mnx, partial[base]);
    mxx = fmax(mxx, partial[base + 1u]);
    mny = fmin(mny, partial[base + 2u]);
    mxy = fmax(mxy, partial[base + 3u]);
    sx += partial[base + 4u];
    sy += partial[base + 5u];
  }
  min_x[tid] = mnx;
  max_x[tid] = mxx;
  min_y[tid] = mny;
  max_y[tid] = mxy;
  sum_x[tid] = sx;
  sum_y[tid] = sy;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      min_x[tid] = fmin(min_x[tid], min_x[tid + stride]);
      max_x[tid] = fmax(max_x[tid], max_x[tid + stride]);
      min_y[tid] = fmin(min_y[tid], min_y[tid + stride]);
      max_y[tid] = fmax(max_y[tid], max_y[tid + stride]);
      sum_x[tid] += sum_x[tid + stride];
      sum_y[tid] += sum_y[tid + stride];
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
    stats[4] = sum_x[0] / static_cast<double>(max(n, 1));
    stats[5] = sum_y[0] / static_cast<double>(max(n, 1));
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

__global__ void opentsne_fft_multiply_kernel(const cufftComplex* left,
                                             const cufftComplex* right,
                                             cufftComplex* out,
                                             int total) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= total) return;
  const cufftComplex a = left[gid];
  const cufftComplex b = right[gid];
  out[gid].x = a.x * b.x - a.y * b.y;
  out[gid].y = a.x * b.y + a.y * b.x;
}

__global__ void opentsne_fft_scale_kernel(cufftComplex* values,
                                          int total,
                                          float scale) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= total) return;
  values[gid].x *= scale;
  values[gid].y *= scale;
}

__device__ float bilinear_grid_complex_real(const cufftComplex* values,
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
                                          int grid_size) {
  extern __shared__ double shared[];
  const int tid = static_cast<int>(threadIdx.x);
  double local = 0.0;
  for (int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       row < n;
       row += static_cast<int>(gridDim.x * blockDim.x)) {
    const float2 gp = grid_pos[row];
    local += static_cast<double>(
      bilinear_grid_complex_real(q_grid, fft_size, grid_size, gp.x, gp.y)
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
                                                   int n_blocks,
                                                   int n) {
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
  if (tid == 0) {
    stats[0] = fmax(shared[0] - static_cast<double>(n), 1.0e-12);
  }
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
                                                       int grid_size) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  const std::size_t base = static_cast<std::size_t>(row) * 2u;
  const float2 gp = grid_pos[row];
  const float inv_sum_q = static_cast<float>(1.0 / stats[0]);
  const float q2 = bilinear_grid_complex_real(q2_grid, fft_size, grid_size, gp.x, gp.y);
  const float xq2 = bilinear_grid_complex_real(xq2_grid, fft_size, grid_size, gp.x, gp.y);
  const float yq2 = bilinear_grid_complex_real(yq2_grid, fft_size, grid_size, gp.x, gp.y);
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
      float gain = ((old < 0.0f) != (g < 0.0f)) ? gains[pos] + 0.2f : gains[pos] * 0.8f + min_gain;
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

__global__ void count_valid_graph_entries_kernel(const int* neighbors,
                                                 const float* weights,
                                                 int* row_counts,
                                                 int n,
                                                 int width) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  int count = 0;
  for (int j = 0; j < width; ++j) {
    const std::size_t pos = static_cast<std::size_t>(row) * width + j;
    const int nb = neighbors[pos];
    const float w = weights[pos];
    if (nb >= 0 && nb < n && nb != row && w > 0.0f) ++count;
  }
  row_counts[row] = count;
}

__global__ void init_csr_offsets_from_counts_kernel(const int* row_counts,
                                                    int* offsets,
                                                    int n) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid == 0) offsets[0] = 0;
  if (gid < n) offsets[gid + 1] = row_counts[gid];
}

__global__ void scan_offsets_step_kernel(const int* in,
                                         int* out,
                                         int length,
                                         int stride) {
  const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= length) return;
  int value = in[gid];
  if (gid >= stride) value += in[gid - stride];
  out[gid] = value;
}

__global__ void pack_coo_graph_kernel(const int* dense_neighbors,
                                      const float* dense_weights,
                                      const int* offsets,
                                      int* coo_heads,
                                      int* coo_tails,
                                      float* coo_weights,
                                      float* coo_epochs_per_sample,
                                      float max_weight,
                                      int n,
                                      int width) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  int write = offsets[row];
  for (int j = 0; j < width; ++j) {
    const std::size_t dense_pos = static_cast<std::size_t>(row) * width + j;
    const int nb = dense_neighbors[dense_pos];
    const float w = dense_weights[dense_pos];
    if (nb < 0 || nb >= n || nb == row || w <= 0.0f) continue;
    coo_heads[write] = row;
    coo_tails[write] = nb;
    coo_weights[write] = w;
    coo_epochs_per_sample[write] = max_weight / fmaxf(w, 1.0e-6f);
    ++write;
  }
}

__device__ int positive_samples_this_epoch_uwot(float period, unsigned int epoch) {
  if (period <= 0.0f || !isfinite(period) || epoch == 0u) return 0;
  const float now = static_cast<float>(epoch);
  const float previous = static_cast<float>(epoch - 1u);
  const int current_sample = static_cast<int>(floorf(now / period));
  const int previous_sample = static_cast<int>(floorf(previous / period));
  const int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

__device__ int negative_samples_this_epoch_uwot(float period,
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
  const int positive_samples = positive_samples_this_epoch_uwot(period, epoch);
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

  const int neg_samples = negative_samples_this_epoch_uwot(period, p, epoch);
  for (int s = 0; s < neg_samples; ++s) {
    const unsigned int neg = deterministic_vertex(
      static_cast<unsigned int>(p.n),
      p.seed,
      epoch,
      static_cast<unsigned int>(head),
      static_cast<unsigned int>(gid),
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

int prepare_device_knn_graph(const int* d_indices,
                             const double* d_distances,
                             int* d_neighbors,
                             float* d_weights,
                             int* d_counts,
                             int n,
                             int k,
                             int width,
                             int offset,
                             int objective) {
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
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_direct_neighbors), graph_items * sizeof(int)), "cudaMalloc(direct_neighbors)")) {
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_direct_weights), graph_items * sizeof(float)), "cudaMalloc(direct_weights)")) {
    cudaFree(d_direct_neighbors);
    return 1;
  }
  auto cleanup_direct = [&]() {
    cudaFree(d_direct_neighbors);
    cudaFree(d_direct_weights);
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

} // namespace

extern "C" const char* fastembedr_cuda_embedding_last_error() {
  return embedding_last_error.c_str();
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

extern "C" int fastembedr_cuda_spectral_init_from_knn(const int* indices,
                                                      const double* distances,
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
    input_items * (sizeof(int) + sizeof(double)) +
    graph_items * (sizeof(int) + sizeof(float)) +
    static_cast<std::size_t>(n) * sizeof(int) +
    2u * embed_bytes +
    static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double) +
    5u * sizeof(double);

  int* d_indices = nullptr;
  double* d_distances = nullptr;
  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
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
    if (d_partial != nullptr) cudaFree(d_partial);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA spectral initialization allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(spectral indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(double)), "cudaMalloc(spectral distances)")) {
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
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(spectral distances H2D)")) {
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

extern "C" int fastembedr_cuda_umap_from_knn_spectral(const int* indices,
                                                       const double* distances,
                                                       int n,
                                                       int k,
                                                       int n_epochs,
                                                       int negative_sample_rate,
                                                       float learning_rate,
                                                       float a,
                                                       float b,
                                                       int spectral_n_iter,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       float* out) {
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 || negative_sample_rate < 0 ||
      learning_rate <= 0.0f || spectral_n_iter < 1) {
    set_embedding_error("invalid fused CUDA UMAP dimensions or parameters");
    return 1;
  }

  const int objective = 0;
  const int width = std::min(256, std::max(k, 2 * k));
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const int stat_blocks = std::max(1, std::min(1024, blocks));
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(double)) +
    graph_items * (2u * sizeof(int) + 4u * sizeof(float)) +
    2u * (static_cast<std::size_t>(n) + 1u) * sizeof(int) +
    2u * embed_bytes +
    static_cast<std::size_t>(n) * sizeof(int) +
    static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double) +
    5u * sizeof(double);

  int* d_indices = nullptr;
  double* d_distances = nullptr;
  int* d_neighbors = nullptr;
  float* d_weights = nullptr;
  int* d_counts = nullptr;
  int* d_offsets = nullptr;
  int* d_scan_tmp = nullptr;
  int* d_coo_heads = nullptr;
  int* d_coo_tails = nullptr;
  float* d_coo_weights = nullptr;
  float* d_coo_epochs_per_sample = nullptr;
  float* d_current = nullptr;
  float* d_next = nullptr;
  double* d_partial = nullptr;
  double* d_stats = nullptr;

  auto cleanup = [&]() {
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_neighbors != nullptr) cudaFree(d_neighbors);
    if (d_weights != nullptr) cudaFree(d_weights);
    if (d_counts != nullptr) cudaFree(d_counts);
    if (d_offsets != nullptr) cudaFree(d_offsets);
    if (d_scan_tmp != nullptr) cudaFree(d_scan_tmp);
    if (d_coo_heads != nullptr) cudaFree(d_coo_heads);
    if (d_coo_tails != nullptr) cudaFree(d_coo_tails);
    if (d_coo_weights != nullptr) cudaFree(d_coo_weights);
    if (d_coo_epochs_per_sample != nullptr) cudaFree(d_coo_epochs_per_sample);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_next != nullptr) cudaFree(d_next);
    if (d_partial != nullptr) cudaFree(d_partial);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA fused UMAP allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(fused umap indices)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(double)), "cudaMalloc(fused umap distances)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_neighbors), graph_items * sizeof(int)), "cudaMalloc(fused umap neighbors)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_weights), graph_items * sizeof(float)), "cudaMalloc(fused umap weights)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_counts), static_cast<std::size_t>(n) * sizeof(int)), "cudaMalloc(fused umap row counts)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(fused umap current)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next), embed_bytes), "cudaMalloc(fused umap next)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_partial), static_cast<std::size_t>(stat_blocks) * 5u * sizeof(double)), "cudaMalloc(fused umap partial)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), 5u * sizeof(double)), "cudaMalloc(fused umap stats)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(fused umap indices H2D)") ||
      check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(fused umap distances H2D)")) {
    cleanup();
    return 1;
  }

  if (prepare_device_knn_graph(
        d_indices, d_distances, d_neighbors, d_weights, d_counts,
        n, k, width, index_offset, objective)) {
    cleanup();
    return 1;
  }
  cudaFree(d_indices);
  cudaFree(d_distances);
  d_indices = nullptr;
  d_distances = nullptr;

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

  count_valid_graph_entries_kernel<<<blocks, threads>>>(d_neighbors, d_weights, d_counts, n, width);
  if (check_cuda(cudaGetLastError(), "count_valid_graph_entries_kernel(fused umap) launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_offsets), static_cast<std::size_t>(n + 1) * sizeof(int)), "cudaMalloc(fused umap csr offsets)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scan_tmp), static_cast<std::size_t>(n + 1) * sizeof(int)), "cudaMalloc(fused umap csr scan tmp)")) {
    cleanup();
    return 1;
  }
  init_csr_offsets_from_counts_kernel<<<blocks, threads>>>(d_counts, d_offsets, n);
  if (check_cuda(cudaGetLastError(), "init_csr_offsets_from_counts_kernel(fused umap) launch")) {
    cleanup();
    return 1;
  }
  const int scan_length = n + 1;
  const int scan_blocks = (scan_length + threads - 1) / threads;
  int* scan_in = d_offsets;
  int* scan_out = d_scan_tmp;
  for (int stride = 1; stride < scan_length; stride <<= 1) {
    scan_offsets_step_kernel<<<scan_blocks, threads>>>(scan_in, scan_out, scan_length, stride);
    if (check_cuda(cudaGetLastError(), "scan_offsets_step_kernel(fused umap) launch")) {
      cleanup();
      return 1;
    }
    std::swap(scan_in, scan_out);
  }
  if (scan_in != d_offsets &&
      check_cuda(cudaMemcpy(d_offsets, scan_in, static_cast<std::size_t>(scan_length) * sizeof(int), cudaMemcpyDeviceToDevice), "cudaMemcpy(fused umap scanned offsets D2D)")) {
    cleanup();
    return 1;
  }

  const float graph_max_weight = 1.0f;
  const EmbedParams params{
    n, width, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, graph_max_weight
  };
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_coo_heads), graph_items * sizeof(int)), "cudaMalloc(fused umap coo heads)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_coo_tails), graph_items * sizeof(int)), "cudaMalloc(fused umap coo tails)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_coo_weights), graph_items * sizeof(float)), "cudaMalloc(fused umap coo weights)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_coo_epochs_per_sample), graph_items * sizeof(float)), "cudaMalloc(fused umap coo epochs_per_sample)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemset(d_coo_heads, 0xff, graph_items * sizeof(int)), "cudaMemset(fused umap coo heads)") ||
      check_cuda(cudaMemset(d_coo_tails, 0xff, graph_items * sizeof(int)), "cudaMemset(fused umap coo tails)")) {
    cleanup();
    return 1;
  }
  pack_coo_graph_kernel<<<blocks, threads>>>(
    d_neighbors, d_weights, d_offsets, d_coo_heads, d_coo_tails, d_coo_weights,
    d_coo_epochs_per_sample, graph_max_weight, n, width
  );
  if (check_cuda(cudaGetLastError(), "pack_coo_graph_kernel(fused umap) launch")) {
    cleanup();
    return 1;
  }
  cudaFree(d_neighbors);
  cudaFree(d_weights);
  cudaFree(d_counts);
  cudaFree(d_offsets);
  cudaFree(d_scan_tmp);
  d_neighbors = nullptr;
  d_weights = nullptr;
  d_counts = nullptr;
  d_offsets = nullptr;
  d_scan_tmp = nullptr;

  const int edge_blocks =
    (static_cast<int>(graph_items) + threads - 1) / threads;
  const float max_abs_coord = 16.0f;
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    embed_epoch_coo_atomic_kernel<<<edge_blocks, threads>>>(
      d_current, d_coo_heads, d_coo_tails, d_coo_weights,
      d_coo_epochs_per_sample, params, static_cast<unsigned int>(epoch),
      static_cast<int>(graph_items)
    );
    if (check_cuda(cudaGetLastError(), "embed_epoch_coo_atomic_kernel(fused umap) launch")) {
      cleanup();
      return 1;
    }
    umap_sanitize_layout_kernel<<<blocks, threads>>>(d_current, n, max_abs_coord);
    if (check_cuda(cudaGetLastError(), "umap_sanitize_layout_kernel(fused umap) launch")) {
      cleanup();
      return 1;
    }
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fused umap)") ||
      check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(fused umap D2H)")) {
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
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_components != 2 ||
      perplexity <= 0.0f || early_exaggeration_iter < 0 || n_iter < 0 ||
      early_exaggeration_iter + n_iter < 1 ||
      early_exaggeration <= 0.0f || exaggeration <= 0.0f ||
      (!learning_rate_auto && learning_rate <= 0.0f) ||
      initial_momentum < 0.0f || final_momentum < 0.0f || min_gain <= 0.0f) {
    set_embedding_error("invalid CUDA openTSNE FFT-grid dimensions or parameters");
    return 1;
  }

  int grid_size = 512;
  if (n < 20000) grid_size = 256;
  if (n >= 100000) grid_size = 1024;
  const char* grid_env = std::getenv("FASTEMBEDR_TSNE_FFT_GRID");
  if (grid_env != nullptr && grid_env[0] != '\0') {
    const int requested = std::atoi(grid_env);
    if (requested == 128 || requested == 256 || requested == 512 || requested == 1024) {
      grid_size = requested;
    }
  }
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
    std::max(layout_stat_blocks * 6, std::max(point_blocks * 2, point_blocks))
  );
  const std::size_t complex_items = static_cast<std::size_t>(fft_total);
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(double) + sizeof(float)) +
    embed_items * 4u * sizeof(float) +
    static_cast<std::size_t>(n) * sizeof(float2) +
    complex_items * 9u * sizeof(cufftComplex) +
    partial_items * sizeof(double) +
    6u * sizeof(double);

  int* d_indices = nullptr;
  double* d_distances = nullptr;
  float* d_probabilities = nullptr;
  float* d_current = nullptr;
  float* d_grad = nullptr;
  float* d_gains = nullptr;
  float* d_update = nullptr;
  float2* d_grid_pos = nullptr;
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
  cufftHandle plan = 0;

  auto cleanup = [&]() {
    if (plan != 0) cufftDestroy(plan);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
    if (d_probabilities != nullptr) cudaFree(d_probabilities);
    if (d_current != nullptr) cudaFree(d_current);
    if (d_grad != nullptr) cudaFree(d_grad);
    if (d_gains != nullptr) cudaFree(d_gains);
    if (d_update != nullptr) cudaFree(d_update);
    if (d_grid_pos != nullptr) cudaFree(d_grid_pos);
    if (d_mass != nullptr) cudaFree(d_mass);
    if (d_mass_x != nullptr) cudaFree(d_mass_x);
    if (d_mass_y != nullptr) cudaFree(d_mass_y);
    if (d_kernel_q != nullptr) cudaFree(d_kernel_q);
    if (d_kernel_q2 != nullptr) cudaFree(d_kernel_q2);
    if (d_q != nullptr) cudaFree(d_q);
    if (d_q2 != nullptr) cudaFree(d_q2);
    if (d_xq2 != nullptr) cudaFree(d_xq2);
    if (d_yq2 != nullptr) cudaFree(d_yq2);
    if (d_partial != nullptr) cudaFree(d_partial);
    if (d_stats != nullptr) cudaFree(d_stats);
  };

  if (check_embedding_memory_available(required_bytes, "CUDA openTSNE FFT-grid allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), input_items * sizeof(int)), "cudaMalloc(opentsne indices)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(double)), "cudaMalloc(opentsne distances)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_probabilities), input_items * sizeof(float)), "cudaMalloc(opentsne probabilities)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_current), embed_bytes), "cudaMalloc(opentsne current)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_grad), embed_bytes), "cudaMalloc(opentsne gradient)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_gains), embed_bytes), "cudaMalloc(opentsne gains)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_update), embed_bytes), "cudaMalloc(opentsne update)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_grid_pos), static_cast<std::size_t>(n) * sizeof(float2)), "cudaMalloc(opentsne grid_pos)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mass), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne mass)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mass_x), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne mass_x)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mass_y), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne mass_y)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_kernel_q), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne kernel_q)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_kernel_q2), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne kernel_q2)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_q), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne q_grid)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_q2), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne q2_grid)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_xq2), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne xq2_grid)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_yq2), complex_items * sizeof(cufftComplex)), "cudaMalloc(opentsne yq2_grid)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_partial), partial_items * sizeof(double)), "cudaMalloc(opentsne partial)") ||
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_stats), 6u * sizeof(double)), "cudaMalloc(opentsne stats)")) {
    cleanup();
    return 1;
  }

  if (check_cuda(cudaMemcpy(d_indices, indices, input_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(opentsne indices H2D)") ||
      check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(opentsne distances H2D)")) {
    cleanup();
    return 1;
  }

  if (has_init) {
    if (check_cuda(cudaMemcpy(d_current, init, embed_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(opentsne init H2D)")) {
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
  if (check_cufft(cufftPlan2d(&plan, fft_size, fft_size, CUFFT_C2C), "cufftPlan2d(opentsne)")) {
    cleanup();
    return 1;
  }

  const int total_iter = early_exaggeration_iter + n_iter;
  const float fft_scale = 1.0f / static_cast<float>(fft_total);
  for (int iter = 0; iter < total_iter; ++iter) {
    opentsne_layout_stats_blocks_kernel<<<layout_stat_blocks, threads, 6u * threads * sizeof(double)>>>(
      d_current, d_partial, n
    );
    if (check_cuda(cudaGetLastError(), "opentsne_layout_stats_blocks_kernel launch")) {
      cleanup();
      return 1;
    }
    opentsne_finalize_layout_stats_kernel<<<1, threads, 6u * threads * sizeof(double)>>>(
      d_partial, d_stats, layout_stat_blocks, n, grid_size
    );
    if (check_cuda(cudaGetLastError(), "opentsne_finalize_layout_stats_kernel launch")) {
      cleanup();
      return 1;
    }

    opentsne_fft_clear_kernel<<<fft_blocks, threads>>>(
      d_mass, d_mass_x, d_mass_y, d_kernel_q, d_kernel_q2,
      d_stats, fft_size, grid_size
    );
    if (check_cuda(cudaGetLastError(), "opentsne_fft_clear_kernel launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_scatter_kernel<<<point_blocks, threads>>>(
      d_current, d_mass, d_mass_x, d_mass_y, d_grid_pos,
      d_stats, n, grid_size, fft_size
    );
    if (check_cuda(cudaGetLastError(), "opentsne_fft_scatter_kernel launch")) {
      cleanup();
      return 1;
    }

    if (check_cufft(cufftExecC2C(plan, d_mass, d_mass, CUFFT_FORWARD), "cufftExecC2C(mass forward)") ||
        check_cufft(cufftExecC2C(plan, d_mass_x, d_mass_x, CUFFT_FORWARD), "cufftExecC2C(mass_x forward)") ||
        check_cufft(cufftExecC2C(plan, d_mass_y, d_mass_y, CUFFT_FORWARD), "cufftExecC2C(mass_y forward)") ||
        check_cufft(cufftExecC2C(plan, d_kernel_q, d_kernel_q, CUFFT_FORWARD), "cufftExecC2C(kernel_q forward)") ||
        check_cufft(cufftExecC2C(plan, d_kernel_q2, d_kernel_q2, CUFFT_FORWARD), "cufftExecC2C(kernel_q2 forward)")) {
      cleanup();
      return 1;
    }

    opentsne_fft_multiply_kernel<<<fft_blocks, threads>>>(d_mass, d_kernel_q, d_q, fft_total);
    if (check_cuda(cudaGetLastError(), "opentsne_fft_multiply_kernel(q) launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_multiply_kernel<<<fft_blocks, threads>>>(d_mass, d_kernel_q2, d_q2, fft_total);
    if (check_cuda(cudaGetLastError(), "opentsne_fft_multiply_kernel(q2) launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_multiply_kernel<<<fft_blocks, threads>>>(d_mass_x, d_kernel_q2, d_xq2, fft_total);
    if (check_cuda(cudaGetLastError(), "opentsne_fft_multiply_kernel(xq2) launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_multiply_kernel<<<fft_blocks, threads>>>(d_mass_y, d_kernel_q2, d_yq2, fft_total);
    if (check_cuda(cudaGetLastError(), "opentsne_fft_multiply_kernel(yq2) launch")) {
      cleanup();
      return 1;
    }

    if (check_cufft(cufftExecC2C(plan, d_q, d_q, CUFFT_INVERSE), "cufftExecC2C(q inverse)") ||
        check_cufft(cufftExecC2C(plan, d_q2, d_q2, CUFFT_INVERSE), "cufftExecC2C(q2 inverse)") ||
        check_cufft(cufftExecC2C(plan, d_xq2, d_xq2, CUFFT_INVERSE), "cufftExecC2C(xq2 inverse)") ||
        check_cufft(cufftExecC2C(plan, d_yq2, d_yq2, CUFFT_INVERSE), "cufftExecC2C(yq2 inverse)")) {
      cleanup();
      return 1;
    }
    opentsne_fft_scale_kernel<<<fft_blocks, threads>>>(d_q, fft_total, fft_scale);
    opentsne_fft_scale_kernel<<<fft_blocks, threads>>>(d_q2, fft_total, fft_scale);
    opentsne_fft_scale_kernel<<<fft_blocks, threads>>>(d_xq2, fft_total, fft_scale);
    opentsne_fft_scale_kernel<<<fft_blocks, threads>>>(d_yq2, fft_total, fft_scale);
    if (check_cuda(cudaGetLastError(), "opentsne_fft_scale_kernel launch")) {
      cleanup();
      return 1;
    }

    opentsne_fft_sum_q_kernel<<<point_blocks, threads, threads * sizeof(double)>>>(
      d_q, d_grid_pos, d_partial, n, fft_size, grid_size
    );
    if (check_cuda(cudaGetLastError(), "opentsne_fft_sum_q_kernel launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_finalize_sum_q_kernel<<<1, threads, threads * sizeof(double)>>>(
      d_partial, d_stats, point_blocks, n
    );
    if (check_cuda(cudaGetLastError(), "opentsne_fft_finalize_sum_q_kernel launch")) {
      cleanup();
      return 1;
    }
    opentsne_fft_repulsive_gradient_kernel<<<point_blocks, threads>>>(
      d_current, d_grad, d_q2, d_xq2, d_yq2, d_grid_pos, d_stats,
      n, fft_size, grid_size
    );
    if (check_cuda(cudaGetLastError(), "opentsne_fft_repulsive_gradient_kernel launch")) {
      cleanup();
      return 1;
    }
    const float current_exaggeration = iter < early_exaggeration_iter ? early_exaggeration : exaggeration;
    opentsne_sparse_attractive_kernel<<<edge_blocks, threads>>>(
      d_indices, d_probabilities, d_current, d_grad, n, k, index_offset, current_exaggeration
    );
    if (check_cuda(cudaGetLastError(), "opentsne_sparse_attractive_kernel launch")) {
      cleanup();
      return 1;
    }
    const float current_momentum = iter < early_exaggeration_iter ? initial_momentum : final_momentum;
    const float current_learning_rate = learning_rate_auto ?
      static_cast<float>(n) / current_exaggeration :
      learning_rate;
    opentsne_update_reduce_kernel<<<point_blocks, threads, 2u * threads * sizeof(double)>>>(
      d_current, d_grad, d_gains, d_update, d_partial, n,
      current_learning_rate, current_momentum, min_gain, max_step_norm
    );
    if (check_cuda(cudaGetLastError(), "opentsne_update_reduce_kernel launch")) {
      cleanup();
      return 1;
    }
    tsne_finalize_mean_kernel<<<1, threads, 2u * threads * sizeof(double)>>>(
      d_partial, d_stats, point_blocks, n
    );
    if (check_cuda(cudaGetLastError(), "tsne_finalize_mean_kernel(opentsne) launch")) {
      cleanup();
      return 1;
    }
    tsne_center_kernel<<<point_blocks, threads>>>(d_current, d_stats, n);
    if (check_cuda(cudaGetLastError(), "tsne_center_kernel(opentsne) launch")) {
      cleanup();
      return 1;
    }
  }

  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(opentsne fft)") ||
      check_cuda(cudaMemcpy(out, d_current, embed_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(opentsne fft D2H)")) {
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
    input_items * (sizeof(int) + sizeof(double)) +
    dense_items * sizeof(float) +
    embed_items * 4u * sizeof(float) +
    partial_items * sizeof(double) +
    4u * sizeof(double);

  int* d_indices = nullptr;
  double* d_distances = nullptr;
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
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(double)), "cudaMalloc(exact tsne distances)")) {
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
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(exact tsne distances H2D)")) {
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
    n, k, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, max_weight
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
  embedding_last_error.clear();
  if (indices == nullptr || distances == nullptr || init == nullptr || out == nullptr) {
    set_embedding_error("null host pointer");
    return 1;
  }
  if (n < 2 || k < 1 || k > 256 || n_epochs < 1 || negative_sample_rate < 0 || learning_rate <= 0.0f) {
    set_embedding_error("invalid KNN embedding dimensions or parameters");
    return 1;
  }
  if (objective == 0) {
    set_embedding_error("CUDA UMAP is available only through the fused pure atomic path");
    return 1;
  }

  const int width = k;
  const std::size_t input_items = static_cast<std::size_t>(n) * k;
  const std::size_t graph_items = static_cast<std::size_t>(n) * width;
  const std::size_t embed_bytes = static_cast<std::size_t>(n) * 2u * sizeof(float);
  const std::size_t required_bytes =
    input_items * (sizeof(int) + sizeof(double)) +
    graph_items * (sizeof(int) + sizeof(float)) +
    2u * embed_bytes;

  int* d_indices = nullptr;
  double* d_distances = nullptr;
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
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), input_items * sizeof(double)), "cudaMalloc(knn distances)")) {
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
  if (check_cuda(cudaMemcpy(d_distances, distances, input_items * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(distances H2D)")) {
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
    n, width, n_epochs, negative_sample_rate, objective, seed, learning_rate, a, b, graph_max_weight
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
