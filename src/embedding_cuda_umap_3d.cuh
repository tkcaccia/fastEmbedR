/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

__global__ void umap_random_init_3d(float* values, int n,
                                    unsigned int seed) {
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  for (int axis = 0; axis < 3; ++axis) {
    values[static_cast<std::size_t>(row) * 3u + axis] =
      deterministic_unit(seed, static_cast<unsigned int>(row), axis);
  }
}

__global__ void umap_diffuse_init_3d(
    const int* neighbors, const float* weights,
    const float* current, float* next, int n, int width) {
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  float sum[3] = {0.0f, 0.0f, 0.0f};
  float weight_sum = 0.0f;
  for (int rank = 0; rank < width; ++rank) {
    std::size_t pos = static_cast<std::size_t>(row) * width + rank;
    int neighbor = neighbors[pos];
    float weight = weights[pos];
    if (neighbor < 0 || neighbor >= n || neighbor == row ||
        weight <= 0.0f) continue;
    std::size_t base = static_cast<std::size_t>(neighbor) * 3u;
    for (int axis = 0; axis < 3; ++axis) {
      sum[axis] += weight * current[base + axis];
    }
    weight_sum += weight;
  }
  std::size_t base = static_cast<std::size_t>(row) * 3u;
  for (int axis = 0; axis < 3; ++axis) {
    next[base + axis] = weight_sum > 0.0f ?
      sum[axis] / weight_sum : current[base + axis];
  }
}

__global__ void umap_init_stats_3d(const float* values,
                                   double* stats, int n) {
  extern __shared__ double shared[];
  int lane = static_cast<int>(threadIdx.x);
  double moments[9] = {};
  for (int row = lane; row < n; row += static_cast<int>(blockDim.x)) {
    std::size_t base = static_cast<std::size_t>(row) * 3u;
    double x = values[base];
    double y = values[base + 1u];
    double z = values[base + 2u];
    moments[0] += x;
    moments[1] += y;
    moments[2] += z;
    moments[3] += x * x;
    moments[4] += x * y;
    moments[5] += x * z;
    moments[6] += y * y;
    moments[7] += y * z;
    moments[8] += z * z;
  }
  for (int moment = 0; moment < 9; ++moment) {
    shared[moment * blockDim.x + lane] = moments[moment];
  }
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2;
       stride > 0; stride >>= 1) {
    if (lane < stride) {
      for (int moment = 0; moment < 9; ++moment) {
        int base = moment * blockDim.x + lane;
        shared[base] += shared[base + stride];
      }
    }
    __syncthreads();
  }
  if (lane != 0) return;
  double inv_n = 1.0 / n;
  double sx = shared[0];
  double sy = shared[blockDim.x];
  double sz = shared[2 * blockDim.x];
  double xx = fmax(1e-24, shared[3 * blockDim.x] - sx * sx * inv_n);
  double xy = shared[4 * blockDim.x] - sx * sy * inv_n;
  double xz = shared[5 * blockDim.x] - sx * sz * inv_n;
  double yy = shared[6 * blockDim.x] - sy * sy * inv_n;
  double yz = shared[7 * blockDim.x] - sy * sz * inv_n;
  double zz = shared[8 * blockDim.x] - sz * sz * inv_n;
  double nx = sqrt(xx);
  double yx = xy / nx;
  double ny = sqrt(fmax(1e-24, yy - yx * yx));
  double zx = xz / nx;
  double zy = (yz - yx * zx) / ny;
  double nz = sqrt(fmax(1e-24, zz - zx * zx - zy * zy));
  stats[0] = sx * inv_n;
  stats[1] = sy * inv_n;
  stats[2] = sz * inv_n;
  stats[3] = 1.0 / nx;
  stats[4] = yx;
  stats[5] = 1.0 / ny;
  stats[6] = zx;
  stats[7] = zy;
  stats[8] = 1.0 / nz;
}

__global__ void umap_normalize_init_3d(float* values,
                                       const double* stats, int n) {
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  std::size_t base = static_cast<std::size_t>(row) * 3u;
  double x = (values[base] - stats[0]) * stats[3];
  double y = ((values[base + 1u] - stats[1]) - stats[4] * x) *
    stats[5];
  double z = ((values[base + 2u] - stats[2]) - stats[6] * x -
              stats[7] * y) * stats[8];
  values[base] = static_cast<float>(x);
  values[base + 1u] = static_cast<float>(y);
  values[base + 2u] = static_cast<float>(z);
}

int normalize_device_init_3d(float* values, double* stats,
                             int n, int blocks, int threads) {
  umap_init_stats_3d<<<1, threads,
    static_cast<std::size_t>(9 * threads) * sizeof(double)>>>(
      values, stats, n
    );
  if (check_cuda(cudaGetLastError(), "umap_init_stats_3d launch")) {
    return 1;
  }
  umap_normalize_init_3d<<<blocks, threads>>>(values, stats, n);
  return check_cuda(cudaGetLastError(),
                    "umap_normalize_init_3d launch");
}

__device__ void update_umap_row_atomic_3d(
    float* layout, const int* neighbors, const float* weights,
    EmbedParams p, unsigned int epoch, int width, int head, int lane) {
  float alpha = p.learning_rate * (1.0f -
    static_cast<float>(epoch) /
    fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  std::size_t head_base = static_cast<std::size_t>(head) * 3u;
  float head_coord[3];
  float head_delta[3] = {0.0f, 0.0f, 0.0f};
  for (int axis = 0; axis < 3; ++axis) {
    head_coord[axis] = layout[head_base + axis];
  }
  for (int rank = lane; rank < width; rank += 32) {
    std::size_t edge = static_cast<std::size_t>(head) * width + rank;
    int tail = neighbors[edge];
    float weight = weights[edge];
    if (tail < 0 || tail >= p.n || tail == head || weight <= 0.0f) {
      continue;
    }
    float period = p.max_weight / fmaxf(weight, 1.0e-6f);
    int positive = positive_samples_this_epoch_umap_schedule(
      period, epoch
    );
    if (positive <= 0) continue;
    std::size_t tail_base = static_cast<std::size_t>(tail) * 3u;
    float diff[3];
    float d2 = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
      diff[axis] = head_coord[axis] - layout[tail_base + axis];
      d2 += diff[axis] * diff[axis];
    }
    float coeff = attractive_coeff(
      fmaxf(1.1920928955078125e-7f, d2), weight, p
    );
    for (int axis = 0; axis < 3; ++axis) {
      float delta = clip4(coeff * diff[axis]) * alpha * positive;
      head_delta[axis] += delta;
      atomicAdd(layout + tail_base + axis, -delta);
    }
    int negatives = negative_samples_this_epoch_umap_schedule(
      period, p, epoch
    );
    for (int sample = 0; sample < negatives; ++sample) {
      unsigned int negative = deterministic_vertex(
        static_cast<unsigned int>(p.n), p.seed, epoch,
        static_cast<unsigned int>(head),
        static_cast<unsigned int>(tail),
        static_cast<unsigned int>(sample)
      );
      if (static_cast<int>(negative) == head ||
          static_cast<int>(negative) == tail) continue;
      std::size_t neg_base = static_cast<std::size_t>(negative) * 3u;
      float ndiff[3];
      float nd2 = 0.0f;
      for (int axis = 0; axis < 3; ++axis) {
        ndiff[axis] = head_coord[axis] - layout[neg_base + axis];
        nd2 += ndiff[axis] * ndiff[axis];
      }
      float repulsion = repulsive_coeff(
        fmaxf(1.1920928955078125e-7f, nd2), p
      );
      for (int axis = 0; axis < 3; ++axis) {
        head_delta[axis] +=
          clip4(repulsion * ndiff[axis]) * alpha;
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    for (int axis = 0; axis < 3; ++axis) {
      head_delta[axis] += __shfl_down_sync(
        0xffffffffu, head_delta[axis], offset
      );
    }
  }
  if (lane == 0) {
    for (int axis = 0; axis < 3; ++axis) {
      atomicAdd(layout + head_base + axis, head_delta[axis]);
    }
  }
}

__global__ void embed_epoch_row_atomic_3d_kernel(
    float* layout, const int* neighbors, const float* weights,
    EmbedParams p, unsigned int epoch, int width) {
  int lane = static_cast<int>(threadIdx.x) & 31;
  int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  int first_row = static_cast<int>(blockIdx.x) * warps_per_block +
    warp_in_block;
  int stride = static_cast<int>(gridDim.x) * warps_per_block;
  for (int head = first_row; head < p.n; head += stride) {
    update_umap_row_atomic_3d(
      layout, neighbors, weights, p, epoch, width, head, lane
    );
  }
}

__global__ void umap_sanitize_layout_3d_kernel(
    float* layout, int n, float limit) {
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  std::size_t base = static_cast<std::size_t>(row) * 3u;
  for (int axis = 0; axis < 3; ++axis) {
    float value = layout[base + axis];
    if (isfinite(value)) {
      layout[base + axis] = fminf(limit, fmaxf(-limit, value));
    }
  }
}

__global__ void embed_epoch_coo_atomic_3d_kernel(
    float* layout, const int* heads, const int* tails,
    const float* weights, const float* periods,
    EmbedParams p, unsigned int epoch, int edges) {
  const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (id >= edges) return;
  const int head = heads[id];
  const int tail = tails[id];
  if (head < 0 || head >= p.n || tail < 0 || tail >= p.n ||
      head == tail) return;
  const float period = periods[id];
  const int positives = positive_samples_this_epoch_umap_schedule(
    period, epoch);
  if (positives <= 0) return;
  const float alpha = p.learning_rate *
    (1.0f - static_cast<float>(epoch) /
      fmaxf(1.0f, static_cast<float>(p.n_epochs)));
  constexpr float eps = 1.1920928955078125e-7f;
  const std::size_t head_base = static_cast<std::size_t>(head) * 3u;
  const std::size_t tail_base = static_cast<std::size_t>(tail) * 3u;
  for (int sample = 0; sample < positives; ++sample) {
    float diff[3];
    float distance = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
      diff[axis] = layout[head_base + axis] -
        layout[tail_base + axis];
      distance += diff[axis] * diff[axis];
    }
    const float coeff = attractive_coeff(
      fmaxf(eps, distance), weights[id], p);
    for (int axis = 0; axis < 3; ++axis) {
      const float delta = clip4(coeff * diff[axis]) * alpha;
      atomicAdd(layout + head_base + axis, delta);
      atomicAdd(layout + tail_base + axis, -delta);
    }
  }
  const int negatives = negative_samples_this_epoch_umap_schedule(
    period, p, epoch);
  for (int sample = 0; sample < negatives; ++sample) {
    const unsigned int neg = deterministic_vertex(
      static_cast<unsigned int>(p.n), p.seed, epoch,
      static_cast<unsigned int>(head),
      static_cast<unsigned int>(tail),
      static_cast<unsigned int>(sample));
    if (static_cast<int>(neg) == head ||
        static_cast<int>(neg) == tail) continue;
    const std::size_t neg_base = static_cast<std::size_t>(neg) * 3u;
    float diff[3];
    float distance = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
      diff[axis] = layout[head_base + axis] -
        layout[neg_base + axis];
      distance += diff[axis] * diff[axis];
    }
    const float coeff = repulsive_coeff(fmaxf(eps, distance), p);
    for (int axis = 0; axis < 3; ++axis) {
      atomicAdd(layout + head_base + axis,
        clip4(coeff * diff[axis]) * alpha);
    }
  }
}
