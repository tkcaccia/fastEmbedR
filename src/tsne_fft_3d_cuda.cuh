/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Cubic-grid three-dimensional CUDA t-SNE repulsion.
 */

#include <cfloat>
#include <cstddef>

struct CudaTsneGrid3d {
  int n;
  int grid;
  int side;
  int bins;
  float lower[3];
  float spacing;
  float inv_spacing;
  float cutoff;
  float cutoff2;
  float qcut;
};

struct CudaTsneBounds3d {
  float lower[3];
  float upper[3];
  float sum[3];
};

__device__ float cuda_tsne_smooth_q_3d(
    float d2, const CudaTsneGrid3d& g) {
  if (d2 >= g.cutoff2) return 1.0f / (1.0f + d2);
  const float delta = d2 - g.cutoff2;
  return g.qcut * (1.0f - g.qcut * delta +
                   g.qcut * g.qcut * delta * delta);
}

__device__ void cuda_tsne_cubic_3d(float t, float* w,
                                   float* d) {
  const float t2 = t * t;
  w[0] = -t * (t - 1.0f) * (t - 2.0f) / 6.0f;
  w[1] = (t + 1.0f) * (t - 1.0f) * (t - 2.0f) / 2.0f;
  w[2] = -(t + 1.0f) * t * (t - 2.0f) / 2.0f;
  w[3] = (t + 1.0f) * t * (t - 1.0f) / 6.0f;
  d[0] = (-3.0f * t2 + 6.0f * t - 2.0f) / 6.0f;
  d[1] = (3.0f * t2 - 4.0f * t - 1.0f) / 2.0f;
  d[2] = (-3.0f * t2 + 2.0f * t + 2.0f) / 2.0f;
  d[3] = (3.0f * t2 - 1.0f) / 6.0f;
}

__device__ void cuda_tsne_point_weights_3d(
    const float* y, int row, const CudaTsneGrid3d& g,
    int* start, float w[3][4], float d[3][4]) {
  for (int axis = 0; axis < 3; ++axis) {
    const float coord =
      (y[static_cast<std::size_t>(row) * 3u + axis] -
       g.lower[axis]) * g.inv_spacing;
    const int cell = max(1, min(g.grid - 3,
      static_cast<int>(floorf(coord))));
    start[axis] = cell - 1;
    cuda_tsne_cubic_3d(coord - cell, w[axis], d[axis]);
  }
}

__global__ void cuda_tsne_bounds_blocks_3d(
    const float* y, CudaTsneBounds3d* blocks, int n,
    int block_size) {
  if (threadIdx.x != 0) return;
  const int begin = static_cast<int>(blockIdx.x) * block_size;
  if (begin >= n) return;
  CudaTsneBounds3d b;
  for (int axis = 0; axis < 3; ++axis) {
    b.lower[axis] = CUDART_INF_F;
    b.upper[axis] = -CUDART_INF_F;
    b.sum[axis] = 0.0f;
  }
  for (int i = begin; i < min(n, begin + block_size);
       ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float value =
        y[static_cast<std::size_t>(i) * 3u + axis];
      b.lower[axis] = fminf(b.lower[axis], value);
      b.upper[axis] = fmaxf(b.upper[axis], value);
      b.sum[axis] += value;
    }
  }
  blocks[blockIdx.x] = b;
}

__global__ void cuda_tsne_finalize_bounds_3d(
    const CudaTsneBounds3d* blocks,
    CudaTsneGrid3d* grid_out, float* center,
    int count, int n, int grid) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  CudaTsneBounds3d b;
  for (int axis = 0; axis < 3; ++axis) {
    b.lower[axis] = CUDART_INF_F;
    b.upper[axis] = -CUDART_INF_F;
    b.sum[axis] = 0.0f;
  }
  for (int i = 0; i < count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      b.lower[axis] = fminf(b.lower[axis],
                            blocks[i].lower[axis]);
      b.upper[axis] = fmaxf(b.upper[axis],
                            blocks[i].upper[axis]);
      b.sum[axis] += blocks[i].sum[axis];
    }
  }
  float span = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    span = fmaxf(span, b.upper[axis] - b.lower[axis]);
    center[axis] = b.sum[axis] / n;
  }
  const float half_span = 0.55f * span + 1.0e-3f;
  const float spacing = 2.0f * half_span / (grid - 5);
  CudaTsneGrid3d g;
  g.n = n;
  g.grid = grid;
  g.side = 2 * grid;
  g.spacing = spacing;
  g.inv_spacing = 1.0f / spacing;
  g.cutoff = (n < 5000 ? 3.0f : 2.0f) * spacing;
  g.cutoff2 = g.cutoff * g.cutoff;
  g.qcut = 1.0f / (1.0f + g.cutoff2);
  g.bins = static_cast<int>(ceilf(grid * spacing /
                                  g.cutoff));
  for (int axis = 0; axis < 3; ++axis) {
    g.lower[axis] = 0.5f *
      (b.lower[axis] + b.upper[axis]) - half_span -
      2.0f * spacing;
  }
  grid_out[0] = g;
}

__global__ void cuda_tsne_clear_fft_3d(
    cufftComplex* mass, cufftComplex* kernel,
    int* bin_head, int side, int bins) {
  const std::size_t pos =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const std::size_t total =
    static_cast<std::size_t>(side) * side * side;
  if (pos < total) {
    mass[pos] = make_cuFloatComplex(0.0f, 0.0f);
    kernel[pos] = make_cuFloatComplex(0.0f, 0.0f);
  }
  if (pos < static_cast<std::size_t>(bins) * bins * bins) {
    bin_head[pos] = -1;
  }
}

__global__ void cuda_tsne_kernel_fft_3d(
    cufftComplex* kernel, const CudaTsneGrid3d* params,
    int total) {
  const int pos = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (pos >= total) return;
  const CudaTsneGrid3d g = params[0];
  const int x = pos % g.side;
  const int y = (pos / g.side) % g.side;
  const int z = pos / (g.side * g.side);
  const int dx = x < g.grid ? x : x - g.side;
  const int dy = y < g.grid ? y : y - g.side;
  const int dz = z < g.grid ? z : z - g.side;
  const float d2 = static_cast<float>(
    dx * dx + dy * dy + dz * dz
  ) * g.spacing * g.spacing;
  kernel[pos].x = cuda_tsne_smooth_q_3d(d2, g);
  kernel[pos].y = 0.0f;
}

__global__ void cuda_tsne_scatter_fft_3d(
    const float* y, cufftComplex* mass, int* bin_head,
    int* bin_next, const CudaTsneGrid3d* params,
    int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  const CudaTsneGrid3d g = params[0];
  int start[3];
  float w[3][4], d[3][4];
  cuda_tsne_point_weights_3d(y, row, g, start, w, d);
  for (int z = 0; z < 4; ++z) {
    for (int yy = 0; yy < 4; ++yy) {
      for (int x = 0; x < 4; ++x) {
        const std::size_t pos =
          (static_cast<std::size_t>(start[2] + z) *
           g.side + start[1] + yy) * g.side +
          start[0] + x;
        atomicAdd(&mass[pos].x,
                  w[0][x] * w[1][yy] * w[2][z]);
      }
    }
  }
  if (g.cutoff2 <= 0.01f) return;
  int cell[3];
  for (int axis = 0; axis < 3; ++axis) {
    cell[axis] = max(0, min(g.bins - 1,
      static_cast<int>((y[static_cast<std::size_t>(row) *
                            3u + axis] - g.lower[axis]) /
                       g.cutoff)));
  }
  const int bin = (cell[2] * g.bins + cell[1]) *
    g.bins + cell[0];
  bin_next[row] = atomicExch(&bin_head[bin], row);
}

__global__ void cuda_tsne_multiply_fft_3d(
    cufftComplex* mass, const cufftComplex* kernel,
    int total) {
  const int pos = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (pos >= total) return;
  const cufftComplex a = mass[pos];
  const cufftComplex b = kernel[pos];
  mass[pos] = make_cuFloatComplex(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

__global__ void cuda_tsne_gather_fft_3d(
    const float* y, const cufftComplex* potential,
    const int* bin_head, const int* bin_next,
    float* repulsive, float* row_q,
    const CudaTsneGrid3d* params, int n,
    float fft_scale) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  const CudaTsneGrid3d g = params[0];
  int start[3];
  float w[3][4], d[3][4];
  cuda_tsne_point_weights_3d(y, row, g, start, w, d);
  float value = 0.0f;
  float slope[3] = {};
  for (int z = 0; z < 4; ++z) {
    for (int yy = 0; yy < 4; ++yy) {
      for (int x = 0; x < 4; ++x) {
        const std::size_t pos =
          (static_cast<std::size_t>(start[2] + z) *
           g.side + start[1] + yy) * g.side +
          start[0] + x;
        const float p = potential[pos].x * fft_scale;
        value += p * w[0][x] * w[1][yy] * w[2][z];
        slope[0] += p * d[0][x] * w[1][yy] * w[2][z];
        slope[1] += p * w[0][x] * d[1][yy] * w[2][z];
        slope[2] += p * w[0][x] * w[1][yy] * d[2][z];
      }
    }
  }
  float auto_w[3][7] = {};
  float cross_w[3][7] = {};
  for (int axis = 0; axis < 3; ++axis) {
    for (int gather = 0; gather < 4; ++gather) {
      for (int scatter = 0; scatter < 4; ++scatter) {
        const int delta = gather - scatter + 3;
        auto_w[axis][delta] += w[axis][gather] *
          w[axis][scatter];
        cross_w[axis][delta] += d[axis][gather] *
          w[axis][scatter];
      }
    }
  }
  float self_value = 0.0f;
  float self_slope[3] = {};
  for (int z = 0; z < 7; ++z) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int x = 0; x < 7; ++x) {
        const float d2 = static_cast<float>(
          (x - 3) * (x - 3) +
          (yy - 3) * (yy - 3) +
          (z - 3) * (z - 3)
        ) * g.spacing * g.spacing;
        const float q = cuda_tsne_smooth_q_3d(d2, g);
        const float yz = auto_w[1][yy] * auto_w[2][z];
        self_value += q * auto_w[0][x] * yz;
        self_slope[0] += q * cross_w[0][x] * yz;
        self_slope[1] += q * auto_w[0][x] *
          cross_w[1][yy] * auto_w[2][z];
        self_slope[2] += q * auto_w[0][x] *
          auto_w[1][yy] * cross_w[2][z];
      }
    }
  }
  value -= self_value;
  float near_slope[3] = {};
  if (g.cutoff2 > 0.01f) {
    int cell[3];
    for (int axis = 0; axis < 3; ++axis) {
      cell[axis] = max(0, min(g.bins - 1,
        static_cast<int>((y[static_cast<std::size_t>(row) *
                              3u + axis] - g.lower[axis]) /
                         g.cutoff)));
    }
    for (int z = max(0, cell[2] - 1);
         z <= min(g.bins - 1, cell[2] + 1); ++z) {
      for (int yy = max(0, cell[1] - 1);
           yy <= min(g.bins - 1, cell[1] + 1); ++yy) {
        for (int x = max(0, cell[0] - 1);
             x <= min(g.bins - 1, cell[0] + 1); ++x) {
          const int bin = (z * g.bins + yy) * g.bins + x;
          for (int j = bin_head[bin]; j >= 0;
               j = bin_next[j]) {
            if (j == row) continue;
            float diff[3];
            float d2 = 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
              diff[axis] =
                y[static_cast<std::size_t>(row) * 3u + axis] -
                y[static_cast<std::size_t>(j) * 3u + axis];
              d2 += diff[axis] * diff[axis];
            }
            if (d2 >= g.cutoff2) continue;
            const float q = 1.0f / (1.0f + d2);
            const float delta = d2 - g.cutoff2;
            const float smooth_derivative =
              -g.qcut * g.qcut + 2.0f * g.qcut *
              g.qcut * g.qcut * delta;
            const float correction = 2.0f *
              (-q * q - smooth_derivative);
            value += q - cuda_tsne_smooth_q_3d(d2, g);
            for (int axis = 0; axis < 3; ++axis) {
              near_slope[axis] += correction * diff[axis];
            }
          }
        }
      }
    }
  }
  row_q[row] = value;
  for (int axis = 0; axis < 3; ++axis) {
    repulsive[static_cast<std::size_t>(row) * 3u +
              axis] = 0.5f *
      ((slope[axis] - self_slope[axis]) * g.inv_spacing +
       near_slope[axis]);
  }
}

__global__ void cuda_tsne_sum_q_blocks_3d(
    const float* row_q, double* partial, int n) {
  __shared__ double sums[256];
  const int lane = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x) * 256 + lane;
  sums[lane] = row < n ? static_cast<double>(row_q[row]) : 0.0;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  if (lane == 0) partial[blockIdx.x] = sums[0];
}

__global__ void cuda_tsne_sum_q_3d(
    const double* partial, float* inv_sum, int blocks) {
  __shared__ double sums[256];
  const int lane = static_cast<int>(threadIdx.x);
  double total = 0.0;
  for (int i = lane; i < blocks; i += 256) total += partial[i];
  sums[lane] = total;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) sums[lane] += sums[lane + stride];
    __syncthreads();
  }
  if (lane == 0) {
    inv_sum[0] = isfinite(sums[0]) && sums[0] > 0.0 ?
      1.0f / static_cast<float>(sums[0]) : CUDART_NAN_F;
  }
}

__global__ void cuda_tsne_scale_repulsion_3d(
    float* grad, const float* inv_sum, int total) {
  const int pos = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (pos < total) grad[pos] *= inv_sum[0];
}

__global__ void cuda_tsne_attractive_3d(
    const int* indices, const float* probabilities,
    const float* y, float* grad, int n, int k,
    int offset, float exaggeration) {
  const int pos = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (pos >= n * k) return;
  const int row = pos / k;
  const int edge = pos - row * k;
  const int other = indices[
    static_cast<std::size_t>(edge) * n + row
  ] - offset;
  if (other < 0 || other >= n || other == row) return;
  const float p = probabilities[pos];
  if (p <= 0.0f || !isfinite(p)) return;
  float diff[3];
  float d2 = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    diff[axis] = y[static_cast<std::size_t>(row) *
                   3u + axis] -
      y[static_cast<std::size_t>(other) * 3u + axis];
    d2 += diff[axis] * diff[axis];
  }
  const float factor = exaggeration * p /
    (1.0f + d2);
  for (int axis = 0; axis < 3; ++axis) {
    const float step = factor * diff[axis];
    atomicAdd(grad + static_cast<std::size_t>(row) *
              3u + axis, step);
    atomicAdd(grad + static_cast<std::size_t>(other) *
              3u + axis, -step);
  }
}

__global__ void cuda_tsne_update_3d(
    float* y, const float* grad, float* gains,
    float* updates, int n, float learning_rate,
    float momentum, float min_gain, float max_step_norm) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  float step[3];
  float norm2 = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t pos =
      static_cast<std::size_t>(row) * 3u + axis;
    float gain = gains[pos];
    const float old = updates[pos];
    const float g = grad[pos];
    gain = (tsne_sign_component(old) !=
            tsne_sign_component(g)) ?
      gain + 0.2f : gain * 0.8f + min_gain;
    gain = fmaxf(gain, min_gain);
    step[axis] = momentum * old -
      learning_rate * gain * g;
    gains[pos] = gain;
    norm2 += step[axis] * step[axis];
  }
  const float scale = norm2 >
    max_step_norm * max_step_norm ?
    max_step_norm / (sqrtf(norm2) + 1.0e-12f) : 1.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t pos =
      static_cast<std::size_t>(row) * 3u + axis;
    updates[pos] = step[axis] * scale;
    y[pos] += updates[pos];
  }
}

__global__ void cuda_tsne_center_3d(
    float* y, const float* center, int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  for (int axis = 0; axis < 3; ++axis) {
    y[static_cast<std::size_t>(row) * 3u + axis] -=
      center[axis];
  }
}

__global__ void cuda_tsne_random_init_3d(
    float* y, int n, unsigned int seed) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  unsigned int value = seed ^
    (static_cast<unsigned int>(row) * 747796405u);
  for (int axis = 0; axis < 3; ++axis) {
    value ^= value >> 16;
    value *= 2246822519u;
    value ^= value >> 13;
    y[static_cast<std::size_t>(row) * 3u + axis] =
      (static_cast<float>(value & 0xffffu) /
       65535.0f - 0.5f) * 2.0e-4f;
  }
}

__global__ void cuda_tsne_scores_init_3d(
    const float* scores, float* y, int n) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x +
                                   threadIdx.x);
  if (row >= n) return;
  for (int axis = 0; axis < 3; ++axis) {
    const float value = scores[
      static_cast<std::size_t>(axis) * n + row
    ];
    y[static_cast<std::size_t>(row) * 3u + axis] =
      isfinite(value) ? value : 0.0f;
  }
}

__global__ void cuda_tsne_normalize_init_3d(
    float* y, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float mean[3] = {};
  float max_abs = 0.0f;
  for (int i = 0; i < n; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      mean[axis] += y[static_cast<std::size_t>(i) *
                      3u + axis];
    }
  }
  for (int axis = 0; axis < 3; ++axis) mean[axis] /= n;
  for (int i = 0; i < n; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float centered = y[static_cast<std::size_t>(i) *
                               3u + axis] - mean[axis];
      max_abs = fmaxf(max_abs, fabsf(centered));
    }
  }
  const float scale = 1.0e-4f / fmaxf(max_abs, 1.0e-12f);
  for (int i = 0; i < n; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t pos =
        static_cast<std::size_t>(i) * 3u + axis;
      y[pos] = (y[pos] - mean[axis]) * scale;
    }
  }
}

template <typename DistanceT>
int cuda_tsne_fft_3d_from_knn(
    const int* indices, const DistanceT* distances,
    const float* init, int has_init,
    const double* pca_init_double,
    const float* pca_init_float,
    const float* pca_init_device_row_major,
    int pca_init_p, int n, int k, float perplexity,
    int early_iter, int normal_iter,
    float early_exaggeration, float exaggeration,
    float learning_rate, int learning_rate_auto,
    float initial_momentum, float final_momentum,
    float min_gain, float max_step_norm,
    unsigned int seed, int index_offset, float* out,
    cudaMemcpyKind input_copy_kind) {
  CudaStreamOwner& owner = cuda_execution_stream();
  if (owner.init()) return 1;
  cudaStream_t stream = owner.get();
  const int grid = n < 5000 ? 16 : 64;
  const int side = 2 * grid;
  const int bins = n < 5000 ?
    (grid + 2) / 3 : (grid + 1) / 2;
  const std::size_t cube =
    static_cast<std::size_t>(side) * side * side;
  const std::size_t input_items =
    static_cast<std::size_t>(n) * k;
  const std::size_t layout_items =
    static_cast<std::size_t>(n) * 3u;
  const int threads = 256;
  const int point_blocks = (n + threads - 1) / threads;
  const int edge_blocks =
    (static_cast<int>(input_items) + threads - 1) / threads;
  const int fft_blocks =
    (static_cast<int>(cube) + threads - 1) / threads;
  const int bound_blocks = point_blocks;
  const bool input_on_device =
    input_copy_kind == cudaMemcpyDeviceToDevice;
  const bool use_pca = !has_init &&
    (pca_init_double != nullptr ||
     pca_init_float != nullptr ||
     pca_init_device_row_major != nullptr);
  const std::size_t pca_bytes = use_pca ?
    cuda_pca_workspace_bytes(
      n, pca_init_p, 3, pca_init_double != nullptr
    ) : 0u;
  const std::size_t bytes =
    input_items * sizeof(float) +
    layout_items * 4u * sizeof(float) +
    cube * 2u * sizeof(cufftComplex) +
    static_cast<std::size_t>(n) *
      (sizeof(float) + sizeof(int)) +
    static_cast<std::size_t>(bins) * bins * bins *
      sizeof(int) +
    static_cast<std::size_t>(bound_blocks) *
      sizeof(CudaTsneBounds3d) +
    static_cast<std::size_t>(bound_blocks) * sizeof(double) +
    sizeof(CudaTsneGrid3d) + 3u * sizeof(float) +
    sizeof(float) +
    (use_pca ? layout_items * sizeof(float) + pca_bytes : 0u) +
    (input_on_device ? 0u : input_items *
      (sizeof(int) + sizeof(DistanceT))) +
    64u * 256u;
  if (check_embedding_memory_available(
        bytes, "CUDA 3D t-SNE workspace preflight")) {
    return 1;
  }
  CudaWorkspace workspace;
  if (workspace.init(bytes, "3d tsne")) return 1;
  const int* d_indices = indices;
  const DistanceT* d_distances = distances;
  if (!input_on_device) {
    int* owned_indices = workspace.alloc<int>(
      input_items, "3d tsne indices"
    );
    DistanceT* owned_distances = workspace.alloc<DistanceT>(
      input_items, "3d tsne distances"
    );
    if (owned_indices == nullptr ||
        owned_distances == nullptr) return 1;
    d_indices = owned_indices;
    d_distances = owned_distances;
    if (check_cuda(cudaMemcpyAsync(
          owned_indices, indices,
          input_items * sizeof(int),
          cudaMemcpyHostToDevice, stream
        ), "3D t-SNE indices upload") ||
        check_cuda(cudaMemcpyAsync(
          owned_distances, distances,
          input_items * sizeof(DistanceT),
          cudaMemcpyHostToDevice, stream
        ), "3D t-SNE distances upload")) {
      return 1;
    }
  }
  float* probabilities = workspace.alloc<float>(
    input_items, "3d tsne probabilities"
  );
  float* current = workspace.alloc<float>(
    layout_items, "3d tsne layout"
  );
  float* gradient = workspace.alloc<float>(
    layout_items, "3d tsne gradient"
  );
  float* gains = workspace.alloc<float>(
    layout_items, "3d tsne gains"
  );
  float* update = workspace.alloc<float>(
    layout_items, "3d tsne update"
  );
  cufftComplex* mass = workspace.alloc<cufftComplex>(
    cube, "3d tsne mass FFT"
  );
  cufftComplex* kernel = workspace.alloc<cufftComplex>(
    cube, "3d tsne kernel FFT"
  );
  int* bin_head = workspace.alloc<int>(
    static_cast<std::size_t>(bins) * bins * bins,
    "3d tsne bin heads"
  );
  int* bin_next = workspace.alloc<int>(
    static_cast<std::size_t>(n), "3d tsne bin links"
  );
  float* row_q = workspace.alloc<float>(
    static_cast<std::size_t>(n), "3d tsne row q"
  );
  float* inv_sum = workspace.alloc<float>(
    1u, "3d tsne inverse sum q"
  );
  double* q_partial = workspace.alloc<double>(
    static_cast<std::size_t>(bound_blocks),
    "3d tsne q partial sums"
  );
  auto* bounds = workspace.alloc<CudaTsneBounds3d>(
    static_cast<std::size_t>(bound_blocks),
    "3d tsne bounds"
  );
  auto* params = workspace.alloc<CudaTsneGrid3d>(
    1u, "3d tsne grid params"
  );
  float* center = workspace.alloc<float>(
    3u, "3d tsne center"
  );
  if (probabilities == nullptr || current == nullptr ||
      gradient == nullptr || gains == nullptr ||
      update == nullptr || mass == nullptr ||
      kernel == nullptr || bin_head == nullptr ||
      bin_next == nullptr || row_q == nullptr ||
      inv_sum == nullptr || q_partial == nullptr ||
      bounds == nullptr ||
      params == nullptr || center == nullptr) return 1;

  if (has_init) {
    if (check_cuda(cudaMemcpyAsync(
          current, init, layout_items * sizeof(float),
          cudaMemcpyHostToDevice, stream
        ), "3D t-SNE initialization upload")) return 1;
  } else if (use_pca) {
    float* scores = workspace.alloc<float>(
      layout_items, "3d tsne pca scores"
    );
    if (scores == nullptr) return 1;
    int status = 0;
    if (pca_init_device_row_major != nullptr) {
      status = cuda_pca_scores_to_device<float>(
        nullptr, n, pca_init_p, 3, scores,
        &workspace, true, false, nullptr, nullptr,
        nullptr, nullptr, 0, seed, 16, 2,
        nullptr, pca_init_device_row_major
      );
    } else if (pca_init_double != nullptr) {
      status = cuda_pca_scores_to_device<double>(
        pca_init_double, n, pca_init_p, 3,
        scores, &workspace
      );
    } else {
      status = cuda_pca_scores_to_device<float>(
        pca_init_float, n, pca_init_p, 3,
        scores, &workspace
      );
    }
    if (status != 0) return 1;
    if (check_cuda(cudaDeviceSynchronize(),
                   "3D t-SNE PCA initialization completion")) {
      return 1;
    }
    cuda_tsne_scores_init_3d<<<point_blocks, threads,
                               0, stream>>>(scores, current, n);
    cuda_tsne_normalize_init_3d<<<1, 1, 0, stream>>>(
      current, n
    );
  } else {
    cuda_tsne_random_init_3d<<<point_blocks, threads,
                               0, stream>>>(current, n, seed);
  }
  if (check_cuda(cudaGetLastError(),
                 "3D t-SNE initialization kernel") ||
      check_cuda(cudaMemsetAsync(
        update, 0, layout_items * sizeof(float),
        stream
      ), "3D t-SNE update reset")) return 1;
  fill_float_kernel<<<(static_cast<int>(layout_items) +
                       threads - 1) / threads, threads,
                      0, stream>>>(
    gains, static_cast<int>(layout_items), 1.0f
  );
  opentsne_affinity_sparse_kernel<<<point_blocks, threads,
                                    0, stream>>>(
    d_indices, d_distances, probabilities, n, k,
    index_offset, perplexity
  );
  if (check_cuda(cudaGetLastError(),
                 "3D t-SNE affinity kernel")) return 1;

  cufftHandle plan = 0;
  if (check_cufft(cufftPlan3d(
        &plan, side, side, side, CUFFT_C2C
      ), "cufftPlan3d(3D t-SNE)")) return 1;
  if (check_cufft(cufftSetStream(plan, stream),
                  "cufftSetStream(3D t-SNE)")) {
    cufftDestroy(plan);
    return 1;
  }
  const float fft_scale = 1.0f /
    static_cast<float>(cube);
  int status = 0;
  for (int iteration = 0;
       iteration < early_iter + normal_iter; ++iteration) {
    cuda_tsne_bounds_blocks_3d<<<bound_blocks, 1,
                                  0, stream>>>(
      current, bounds, n, threads
    );
    cuda_tsne_finalize_bounds_3d<<<1, 1,
                                    0, stream>>>(
      bounds, params, center, bound_blocks, n, grid
    );
    cuda_tsne_clear_fft_3d<<<fft_blocks, threads,
                              0, stream>>>(
      mass, kernel, bin_head, side, bins
    );
    cuda_tsne_kernel_fft_3d<<<fft_blocks, threads,
                               0, stream>>>(
      kernel, params, static_cast<int>(cube)
    );
    cuda_tsne_scatter_fft_3d<<<point_blocks, threads,
                                0, stream>>>(
      current, mass, bin_head, bin_next, params, n
    );
    if (check_cuda(cudaGetLastError(),
                   "3D t-SNE FFT preparation")) {
      status = 1;
      break;
    }
    if (check_cufft(cufftExecC2C(
          plan, mass, mass, CUFFT_FORWARD
        ), "3D t-SNE mass forward FFT") ||
        check_cufft(cufftExecC2C(
          plan, kernel, kernel, CUFFT_FORWARD
        ), "3D t-SNE kernel forward FFT")) {
      status = 1;
      break;
    }
    cuda_tsne_multiply_fft_3d<<<fft_blocks, threads,
                                 0, stream>>>(
      mass, kernel, static_cast<int>(cube)
    );
    if (check_cufft(cufftExecC2C(
          plan, mass, mass, CUFFT_INVERSE
        ), "3D t-SNE inverse FFT")) {
      status = 1;
      break;
    }
    cuda_tsne_gather_fft_3d<<<point_blocks, threads,
                               0, stream>>>(
      current, mass, bin_head, bin_next,
      gradient, row_q, params, n, fft_scale
    );
    cuda_tsne_sum_q_blocks_3d<<<bound_blocks, threads,
                                  0, stream>>>(
      row_q, q_partial, n
    );
    cuda_tsne_sum_q_3d<<<1, threads, 0, stream>>>(
      q_partial, inv_sum, bound_blocks
    );
    cuda_tsne_scale_repulsion_3d<<<(
      static_cast<int>(layout_items) + threads - 1) /
      threads, threads, 0, stream>>>(
      gradient, inv_sum, static_cast<int>(layout_items)
    );
    const bool early = iteration < early_iter;
    const float factor = early ?
      early_exaggeration : exaggeration;
    cuda_tsne_attractive_3d<<<edge_blocks, threads,
                               0, stream>>>(
      d_indices, probabilities, current, gradient,
      n, k, index_offset, factor
    );
    const float rate = learning_rate_auto ?
      n / factor : learning_rate;
    cuda_tsne_update_3d<<<point_blocks, threads,
                           0, stream>>>(
      current, gradient, gains, update, n, rate,
      early ? initial_momentum : final_momentum,
      min_gain, max_step_norm > 0.0f ?
        max_step_norm : FLT_MAX
    );
    cuda_tsne_bounds_blocks_3d<<<bound_blocks, 1,
                                  0, stream>>>(
      current, bounds, n, threads
    );
    cuda_tsne_finalize_bounds_3d<<<1, 1,
                                    0, stream>>>(
      bounds, params, center, bound_blocks, n, grid
    );
    cuda_tsne_center_3d<<<point_blocks, threads,
                           0, stream>>>(
      current, center, n
    );
    if (check_cuda(cudaGetLastError(),
                   "3D t-SNE optimizer kernels")) {
      status = 1;
      break;
    }
  }
  if (status == 0) {
    status = check_cuda(cudaMemcpyAsync(
      out, current, layout_items * sizeof(float),
      cudaMemcpyDeviceToHost, stream
    ), "3D t-SNE layout download");
  }
  if (status == 0) {
    status = check_cuda(cudaStreamSynchronize(stream),
                        "3D t-SNE GPU completion");
  }
  cufftDestroy(plan);
  return status;
}
