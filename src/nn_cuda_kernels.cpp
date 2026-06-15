#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxCudaK = 256;
constexpr int kFastCudaK = 64;
constexpr int kFastCudaCandidateK = 64;
constexpr float kCudaLargeDistance = 3.4028234663852886e+38F;

struct KnnParams {
  int n_data;
  int n_points;
  int n_features;
  int k;
  int square;
};

struct CudaGridParams {
  int n;
  int n_features;
  int k;
  int bins;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
};

thread_local std::string last_error;
thread_local std::string device_info_json;

void set_error(const std::string& message) {
  last_error = message;
}

int fail_cuda(cudaError_t code, const char* where) {
  set_error(std::string(where) + ": " + cudaGetErrorString(code));
  return 1;
}

int check_cuda(cudaError_t code, const char* where) {
  return code == cudaSuccess ? 0 : fail_cuda(code, where);
}

int check_memory_available(std::size_t required_bytes, const char* where) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t code = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (code != cudaSuccess) return 0;
  const std::size_t reserve = free_bytes / 20u;
  if (required_bytes > free_bytes - reserve) {
    set_error(
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

int choose_query_batch_size(int n_points,
                            int n_features,
                            int k,
                            std::size_t data_bytes) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t code = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (code != cudaSuccess) return n_points;
  const std::size_t reserve = free_bytes / 20u;
  if (free_bytes <= reserve + data_bytes) return 0;
  const std::size_t budget = free_bytes - reserve - data_bytes;
  const std::size_t per_query =
    static_cast<std::size_t>(n_features) * sizeof(float) +
    static_cast<std::size_t>(k) * (sizeof(int) + sizeof(float));
  if (per_query == 0) return 0;
  const std::size_t by_memory = budget / per_query;
  if (by_memory == 0) return 0;
  return static_cast<int>(std::min<std::size_t>(
    static_cast<std::size_t>(n_points),
    by_memory
  ));
}

std::string json_escape_cuda(const char* text) {
  if (text == nullptr) return std::string();
  std::string out;
  for (const char* p = text; *p != '\0'; ++p) {
    switch (*p) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += *p; break;
    }
  }
  return out;
}

__device__ void insert_candidate(float dist,
                                 int idx,
                                 float* best_dist,
                                 int* best_idx,
                                 int k) {
  if (dist > best_dist[k - 1] ||
      (dist == best_dist[k - 1] && idx >= best_idx[k - 1])) {
    return;
  }
  int pos = k - 1;
  while (pos > 0 &&
         (dist < best_dist[pos - 1] ||
          (dist == best_dist[pos - 1] && idx < best_idx[pos - 1]))) {
    best_dist[pos] = best_dist[pos - 1];
    best_idx[pos] = best_idx[pos - 1];
    --pos;
  }
  best_dist[pos] = dist;
  best_idx[pos] = idx;
}

__device__ bool contains_candidate(int idx,
                                   const int* best_idx,
                                   int k) {
  for (int j = 0; j < k; ++j) {
    if (best_idx[j] == idx) return true;
  }
  return false;
}

__device__ void insert_unique_candidate(float dist,
                                        int idx,
                                        float* best_dist,
                                        int* best_idx,
                                        int k) {
  if (contains_candidate(idx, best_idx, k)) return;
  insert_candidate(dist, idx, best_dist, best_idx, k);
}

__device__ int grid_coord_device(float value, float min_value, float cell_size, int bins) {
  int out = static_cast<int>((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= bins) out = bins - 1;
  return out;
}

__device__ int grid2d_cell_device(int ix, int iy, int bins) {
  return iy * bins + ix;
}

__device__ int grid3d_cell_device(int ix, int iy, int iz, int bins) {
  return (iz * bins + iy) * bins + ix;
}

__device__ float grid2d_lower_outside_device(float x,
                                             float y,
                                             const CudaGridParams params,
                                             int x0,
                                             int x1,
                                             int y0,
                                             int y1) {
  float best = kCudaLargeDistance;
  if (x0 > 0) {
    const float border = params.min_x + static_cast<float>(x0) * params.cell_x;
    const float dx = fmaxf(0.0f, x - border);
    best = fminf(best, dx * dx);
  }
  if (x1 + 1 < params.bins) {
    const float border = params.min_x + static_cast<float>(x1 + 1) * params.cell_x;
    const float dx = fmaxf(0.0f, border - x);
    best = fminf(best, dx * dx);
  }
  if (y0 > 0) {
    const float border = params.min_y + static_cast<float>(y0) * params.cell_y;
    const float dy = fmaxf(0.0f, y - border);
    best = fminf(best, dy * dy);
  }
  if (y1 + 1 < params.bins) {
    const float border = params.min_y + static_cast<float>(y1 + 1) * params.cell_y;
    const float dy = fmaxf(0.0f, border - y);
    best = fminf(best, dy * dy);
  }
  return best;
}

__device__ float grid3d_lower_outside_device(float x,
                                             float y,
                                             float z,
                                             const CudaGridParams params,
                                             int x0,
                                             int x1,
                                             int y0,
                                             int y1,
                                             int z0,
                                             int z1) {
  float best = grid2d_lower_outside_device(x, y, params, x0, x1, y0, y1);
  if (z0 > 0) {
    const float border = params.min_z + static_cast<float>(z0) * params.cell_z;
    const float dz = fmaxf(0.0f, z - border);
    best = fminf(best, dz * dz);
  }
  if (z1 + 1 < params.bins) {
    const float border = params.min_z + static_cast<float>(z1 + 1) * params.cell_z;
    const float dz = fmaxf(0.0f, border - z);
    best = fminf(best, dz * dz);
  }
  return best;
}

__device__ void add_grid2d_cell_device(const float* data,
                                       const int* offsets,
                                       const int* rows,
                                       const CudaGridParams params,
                                       int query,
                                       int ix,
                                       int iy,
                                       float* best_dist,
                                       int* best_idx) {
  if (ix < 0 || iy < 0 || ix >= params.bins || iy >= params.bins) return;
  const int cell = grid2d_cell_device(ix, iy, params.bins);
  const int start = offsets[cell];
  const int end = offsets[cell + 1];
  const float qx = data[query];
  const float qy = data[static_cast<std::size_t>(params.n) + query];
  for (int pos = start; pos < end; ++pos) {
    const int candidate = rows[pos];
    if (candidate == query) continue;
    const float dx = qx - data[candidate];
    const float dy = qy - data[static_cast<std::size_t>(params.n) + candidate];
    insert_candidate(dx * dx + dy * dy, candidate, best_dist, best_idx, params.k);
  }
}

__device__ void add_grid3d_cell_device(const float* data,
                                       const int* offsets,
                                       const int* rows,
                                       const CudaGridParams params,
                                       int query,
                                       int ix,
                                       int iy,
                                       int iz,
                                       float* best_dist,
                                       int* best_idx) {
  if (ix < 0 || iy < 0 || iz < 0 ||
      ix >= params.bins || iy >= params.bins || iz >= params.bins) return;
  const int cell = grid3d_cell_device(ix, iy, iz, params.bins);
  const int start = offsets[cell];
  const int end = offsets[cell + 1];
  const float qx = data[query];
  const float qy = data[static_cast<std::size_t>(params.n) + query];
  const float qz = data[static_cast<std::size_t>(2) * params.n + query];
  for (int pos = start; pos < end; ++pos) {
    const int candidate = rows[pos];
    if (candidate == query) continue;
    const float dx = qx - data[candidate];
    const float dy = qy - data[static_cast<std::size_t>(params.n) + candidate];
    const float dz = qz - data[static_cast<std::size_t>(2) * params.n + candidate];
    insert_candidate(dx * dx + dy * dy + dz * dz, candidate, best_dist, best_idx, params.k);
  }
}

__global__ void grid_self_knn_kernel(const float* data,
                                     const int* offsets,
                                     const int* rows,
                                     int* out_idx,
                                     float* out_dist,
                                     CudaGridParams params) {
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= params.n) return;

  float best_dist[kMaxCudaK];
  int best_idx[kMaxCudaK];
  for (int j = 0; j < params.k; ++j) {
    best_dist[j] = kCudaLargeDistance;
    best_idx[j] = INT_MAX;
  }

  const float qx = data[q];
  const float qy = data[static_cast<std::size_t>(params.n) + q];
  const int cx = grid_coord_device(qx, params.min_x, params.cell_x, params.bins);
  const int cy = grid_coord_device(qy, params.min_y, params.cell_y, params.bins);

  if (params.n_features == 2) {
    for (int radius = 0; radius <= params.bins; ++radius) {
      const int raw_x0 = cx - radius;
      const int raw_x1 = cx + radius;
      const int raw_y0 = cy - radius;
      const int raw_y1 = cy + radius;
      const int x0 = raw_x0 < 0 ? 0 : raw_x0;
      const int x1 = raw_x1 >= params.bins ? params.bins - 1 : raw_x1;
      const int y0 = raw_y0 < 0 ? 0 : raw_y0;
      const int y1 = raw_y1 >= params.bins ? params.bins - 1 : raw_y1;
      if (radius == 0) {
        add_grid2d_cell_device(data, offsets, rows, params, q, cx, cy, best_dist, best_idx);
      } else {
        for (int ix = raw_x0; ix <= raw_x1; ++ix) {
          if (ix < 0 || ix >= params.bins) continue;
          if (raw_y0 >= 0 && raw_y0 < params.bins) {
            add_grid2d_cell_device(data, offsets, rows, params, q, ix, raw_y0, best_dist, best_idx);
          }
          if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < params.bins) {
            add_grid2d_cell_device(data, offsets, rows, params, q, ix, raw_y1, best_dist, best_idx);
          }
        }
        for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
          if (iy < 0 || iy >= params.bins) continue;
          if (raw_x0 >= 0 && raw_x0 < params.bins) {
            add_grid2d_cell_device(data, offsets, rows, params, q, raw_x0, iy, best_dist, best_idx);
          }
          if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < params.bins) {
            add_grid2d_cell_device(data, offsets, rows, params, q, raw_x1, iy, best_dist, best_idx);
          }
        }
      }
      if (best_idx[params.k - 1] != INT_MAX) {
        const float lower = grid2d_lower_outside_device(qx, qy, params, x0, x1, y0, y1);
        if (lower > best_dist[params.k - 1]) break;
      }
    }
  } else {
    const float qz = data[static_cast<std::size_t>(2) * params.n + q];
    const int cz = grid_coord_device(qz, params.min_z, params.cell_z, params.bins);
    for (int radius = 0; radius <= params.bins; ++radius) {
      const int raw_x0 = cx - radius;
      const int raw_x1 = cx + radius;
      const int raw_y0 = cy - radius;
      const int raw_y1 = cy + radius;
      const int raw_z0 = cz - radius;
      const int raw_z1 = cz + radius;
      const int x0 = raw_x0 < 0 ? 0 : raw_x0;
      const int x1 = raw_x1 >= params.bins ? params.bins - 1 : raw_x1;
      const int y0 = raw_y0 < 0 ? 0 : raw_y0;
      const int y1 = raw_y1 >= params.bins ? params.bins - 1 : raw_y1;
      const int z0 = raw_z0 < 0 ? 0 : raw_z0;
      const int z1 = raw_z1 >= params.bins ? params.bins - 1 : raw_z1;
      if (radius == 0) {
        add_grid3d_cell_device(data, offsets, rows, params, q, cx, cy, cz, best_dist, best_idx);
      } else {
        for (int iz = raw_z0; iz <= raw_z1; ++iz) {
          if (iz < 0 || iz >= params.bins) continue;
          for (int iy = raw_y0; iy <= raw_y1; ++iy) {
            if (iy < 0 || iy >= params.bins) continue;
            for (int ix = raw_x0; ix <= raw_x1; ++ix) {
              if (ix < 0 || ix >= params.bins) continue;
              if (ix != raw_x0 && ix != raw_x1 &&
                  iy != raw_y0 && iy != raw_y1 &&
                  iz != raw_z0 && iz != raw_z1) {
                continue;
              }
              add_grid3d_cell_device(data, offsets, rows, params, q, ix, iy, iz, best_dist, best_idx);
            }
          }
        }
      }
      if (best_idx[params.k - 1] != INT_MAX) {
        const float lower = grid3d_lower_outside_device(qx, qy, qz, params, x0, x1, y0, y1, z0, z1);
        if (lower > best_dist[params.k - 1]) break;
      }
    }
  }

  for (int j = 0; j < params.k; ++j) {
    const std::size_t offset = static_cast<std::size_t>(j) * params.n + q;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrtf(best_dist[j]);
  }
}

__global__ void knn_serial_query_kernel(const float* data,
                                        const float* points,
                                        int* out_idx,
                                        float* out_dist,
                                        KnnParams params) {
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= params.n_points) return;

  float best_dist[kMaxCudaK];
  int best_idx[kMaxCudaK];
  for (int j = 0; j < params.k; ++j) {
    best_dist[j] = kCudaLargeDistance;
    best_idx[j] = INT_MAX;
  }

  for (int i = 0; i < params.n_data; ++i) {
    float dist = 0.0f;
    for (int c = 0; c < params.n_features; ++c) {
      const float diff =
        data[static_cast<std::size_t>(c) * params.n_data + i] -
        points[static_cast<std::size_t>(c) * params.n_points + q];
      dist += diff * diff;
    }

    insert_candidate(dist, i, best_dist, best_idx, params.k);
  }

  for (int j = 0; j < params.k; ++j) {
    const std::size_t offset = static_cast<std::size_t>(j) * params.n_points + q;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = params.square ? best_dist[j] : sqrtf(best_dist[j]);
  }
}

__global__ void knn_cooperative_query_kernel(const float* data,
                                             const float* points,
                                             int* out_idx,
                                             float* out_dist,
                                             KnnParams params) {
  const int q = static_cast<int>(blockIdx.x);
  if (q >= params.n_points || params.k > kFastCudaK) return;

  float local_dist[kFastCudaK];
  int local_idx[kFastCudaK];
  for (int j = 0; j < params.k; ++j) {
    local_dist[j] = kCudaLargeDistance;
    local_idx[j] = INT_MAX;
  }

  for (int i = static_cast<int>(threadIdx.x); i < params.n_data; i += static_cast<int>(blockDim.x)) {
    float dist = 0.0f;
    for (int c = 0; c < params.n_features; ++c) {
      const float diff =
        data[static_cast<std::size_t>(c) * params.n_data + i] -
        points[static_cast<std::size_t>(c) * params.n_points + q];
      dist += diff * diff;
    }
    insert_candidate(dist, i, local_dist, local_idx, params.k);
  }

  extern __shared__ unsigned char shared[];
  float* shared_dist = reinterpret_cast<float*>(shared);
  int* shared_idx = reinterpret_cast<int*>(
    shared_dist + static_cast<std::size_t>(blockDim.x) * params.k
  );
  const std::size_t base = static_cast<std::size_t>(threadIdx.x) * params.k;
  for (int j = 0; j < params.k; ++j) {
    shared_dist[base + j] = local_dist[j];
    shared_idx[base + j] = local_idx[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float best_dist[kFastCudaK];
    int best_idx[kFastCudaK];
    for (int j = 0; j < params.k; ++j) {
      best_dist[j] = kCudaLargeDistance;
      best_idx[j] = INT_MAX;
    }
    for (int t = 0; t < static_cast<int>(blockDim.x); ++t) {
      const std::size_t tbase = static_cast<std::size_t>(t) * params.k;
      for (int j = 0; j < params.k; ++j) {
        insert_candidate(shared_dist[tbase + j], shared_idx[tbase + j], best_dist, best_idx, params.k);
      }
    }
    for (int j = 0; j < params.k; ++j) {
      const std::size_t offset = static_cast<std::size_t>(j) * params.n_points + q;
      out_idx[offset] = best_idx[j] + 1;
      out_dist[offset] = params.square ? best_dist[j] : sqrtf(best_dist[j]);
    }
  }
}

__device__ bool shares_projection_bucket(const int* projection_indices,
                                         int n,
                                         int query,
                                         int candidate,
                                         int bucket_cols,
                                         int query_cols) {
  for (int qc = 0; qc < query_cols; ++qc) {
    const int q_landmark = projection_indices[static_cast<std::size_t>(qc) * n + query];
    if (q_landmark < 1) continue;
    for (int bc = 0; bc < bucket_cols; ++bc) {
      if (projection_indices[static_cast<std::size_t>(bc) * n + candidate] == q_landmark) {
        return true;
      }
    }
  }
  return false;
}

__device__ float self_distance_sq(const float* data,
                                  int a,
                                  int b,
                                  int n,
                                  int n_features) {
  float dist = 0.0f;
  for (int c = 0; c < n_features; ++c) {
    const float diff =
      data[static_cast<std::size_t>(c) * n + a] -
      data[static_cast<std::size_t>(c) * n + b];
    dist += diff * diff;
  }
  return dist;
}

__global__ void landmark_candidate_knn_serial_kernel(const float* data,
                                                     const int* projection_indices,
                                                     int* out_idx,
                                                     float* out_dist,
                                                     int n,
                                                     int n_features,
                                                     int k,
                                                     int bucket_cols,
                                                     int query_cols) {
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= n) return;

  float best_dist[kMaxCudaK];
  int best_idx[kMaxCudaK];
  for (int j = 0; j < k; ++j) {
    best_dist[j] = kCudaLargeDistance;
    best_idx[j] = INT_MAX;
  }

  int candidate_count = 0;
  for (int candidate = 0; candidate < n; ++candidate) {
    if (candidate == q) continue;
    if (!shares_projection_bucket(
          projection_indices, n, q, candidate, bucket_cols, query_cols
        )) {
      continue;
    }
    ++candidate_count;
    insert_candidate(
      self_distance_sq(data, q, candidate, n, n_features),
      candidate,
      best_dist,
      best_idx,
      k
    );
  }

  if (candidate_count < k) {
    for (int j = 0; j < k; ++j) {
      best_dist[j] = kCudaLargeDistance;
      best_idx[j] = INT_MAX;
    }
    for (int candidate = 0; candidate < n; ++candidate) {
      if (candidate == q) continue;
      insert_candidate(
        self_distance_sq(data, q, candidate, n, n_features),
        candidate,
        best_dist,
        best_idx,
        k
      );
    }
  }

  for (int j = 0; j < k; ++j) {
    const std::size_t offset = static_cast<std::size_t>(j) * n + q;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrtf(fmaxf(best_dist[j], 0.0f));
  }
}

__global__ void landmark_candidate_knn_cooperative_kernel(const float* data,
                                                          const int* projection_indices,
                                                          int* out_idx,
                                                          float* out_dist,
                                                          int n,
                                                          int n_features,
                                                          int k,
                                                          int bucket_cols,
                                                          int query_cols) {
  const int q = static_cast<int>(blockIdx.x);
  if (q >= n || k > kFastCudaCandidateK) return;

  float local_dist[kFastCudaCandidateK];
  int local_idx[kFastCudaCandidateK];
  int local_count = 0;
  for (int j = 0; j < k; ++j) {
    local_dist[j] = kCudaLargeDistance;
    local_idx[j] = INT_MAX;
  }

  for (int candidate = static_cast<int>(threadIdx.x); candidate < n; candidate += static_cast<int>(blockDim.x)) {
    if (candidate == q) continue;
    if (!shares_projection_bucket(
          projection_indices, n, q, candidate, bucket_cols, query_cols
        )) {
      continue;
    }
    ++local_count;
    insert_candidate(
      self_distance_sq(data, q, candidate, n, n_features),
      candidate,
      local_dist,
      local_idx,
      k
    );
  }

  extern __shared__ unsigned char shared[];
  float* shared_dist = reinterpret_cast<float*>(shared);
  int* shared_idx = reinterpret_cast<int*>(
    shared_dist + static_cast<std::size_t>(blockDim.x) * k
  );
  int* shared_count = reinterpret_cast<int*>(
    shared_idx + static_cast<std::size_t>(blockDim.x) * k
  );
  const std::size_t base = static_cast<std::size_t>(threadIdx.x) * k;
  for (int j = 0; j < k; ++j) {
    shared_dist[base + j] = local_dist[j];
    shared_idx[base + j] = local_idx[j];
  }
  shared_count[threadIdx.x] = local_count;
  __syncthreads();

  if (threadIdx.x == 0) {
    float best_dist[kFastCudaCandidateK];
    int best_idx[kFastCudaCandidateK];
    int total_candidates = 0;
    for (int j = 0; j < k; ++j) {
      best_dist[j] = kCudaLargeDistance;
      best_idx[j] = INT_MAX;
    }
    for (int t = 0; t < static_cast<int>(blockDim.x); ++t) {
      total_candidates += shared_count[t];
      const std::size_t tbase = static_cast<std::size_t>(t) * k;
      for (int j = 0; j < k; ++j) {
        insert_candidate(shared_dist[tbase + j], shared_idx[tbase + j], best_dist, best_idx, k);
      }
    }

    if (total_candidates < k) {
      for (int j = 0; j < k; ++j) {
        best_dist[j] = kCudaLargeDistance;
        best_idx[j] = INT_MAX;
      }
      for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == q) continue;
        insert_candidate(
          self_distance_sq(data, q, candidate, n, n_features),
          candidate,
          best_dist,
          best_idx,
          k
        );
      }
    }

    for (int j = 0; j < k; ++j) {
      const std::size_t offset = static_cast<std::size_t>(j) * n + q;
      out_idx[offset] = best_idx[j] + 1;
      out_dist[offset] = sqrtf(fmaxf(best_dist[j], 0.0f));
    }
  }
}

__global__ void row_candidate_knn_kernel(const float* data,
                                         const int* candidate_indices,
                                         int* out_idx,
                                         float* out_dist,
                                         int n,
                                         int n_features,
                                         int k,
                                         int n_candidates) {
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= n) return;

  float best_dist[kMaxCudaK];
  int best_idx[kMaxCudaK];
  for (int j = 0; j < k; ++j) {
    best_dist[j] = kCudaLargeDistance;
    best_idx[j] = INT_MAX;
  }

  for (int c = 0; c < n_candidates; ++c) {
    const int candidate = candidate_indices[static_cast<std::size_t>(c) * n + q] - 1;
    if (candidate < 0 || candidate >= n || candidate == q) continue;
    insert_unique_candidate(
      self_distance_sq(data, q, candidate, n, n_features),
      candidate,
      best_dist,
      best_idx,
      k
    );
  }

  if (best_idx[k - 1] == INT_MAX) {
    for (int candidate = 0; candidate < n && best_idx[k - 1] == INT_MAX; ++candidate) {
      if (candidate == q) continue;
      insert_unique_candidate(
        self_distance_sq(data, q, candidate, n, n_features),
        candidate,
        best_dist,
        best_idx,
        k
      );
    }
  }

  for (int j = 0; j < k; ++j) {
    const std::size_t offset = static_cast<std::size_t>(j) * n + q;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrtf(fmaxf(best_dist[j], 0.0f));
  }
}

int copy_double_to_float_device(const double* host,
                                float* device,
                                std::size_t n,
                                const char* where) {
  constexpr std::size_t chunk_size = 1u << 20;
  std::vector<float> buffer(std::min(chunk_size, n));
  for (std::size_t offset = 0; offset < n; offset += chunk_size) {
    const std::size_t count = std::min(chunk_size, n - offset);
    for (std::size_t i = 0; i < count; ++i) {
      buffer[i] = static_cast<float>(host[offset + i]);
    }
    if (check_cuda(
      cudaMemcpy(device + offset, buffer.data(), count * sizeof(float), cudaMemcpyHostToDevice),
      where
    )) {
      return 1;
    }
  }
  return 0;
}

int copy_query_batch_to_float_device(const double* host_points,
                                     float* device_points,
                                     int n_points_total,
                                     int n_features,
                                     int start,
                                     int batch_n,
                                     std::vector<float>& buffer) {
  if (static_cast<int>(buffer.size()) < batch_n) buffer.resize(batch_n);
  for (int c = 0; c < n_features; ++c) {
    const double* column = host_points + static_cast<std::size_t>(c) * n_points_total + start;
    for (int i = 0; i < batch_n; ++i) {
      buffer[static_cast<std::size_t>(i)] = static_cast<float>(column[i]);
    }
    if (check_cuda(
      cudaMemcpy(
        device_points + static_cast<std::size_t>(c) * batch_n,
        buffer.data(),
        static_cast<std::size_t>(batch_n) * sizeof(float),
        cudaMemcpyHostToDevice
      ),
      "cudaMemcpy(query batch H2D)"
    )) {
      return 1;
    }
  }
  return 0;
}

int copy_batch_results_to_host(const int* d_indices,
                               const float* d_distances,
                               int batch_n,
                               int total_points,
                               int start,
                               int k,
                               int* out_indices,
                               double* out_distances,
                               std::vector<int>& h_indices,
                               std::vector<float>& h_distances) {
  const std::size_t batch_out_size = static_cast<std::size_t>(batch_n) * k;
  if (h_indices.size() < batch_out_size) h_indices.resize(batch_out_size);
  if (h_distances.size() < batch_out_size) h_distances.resize(batch_out_size);
  if (check_cuda(
    cudaMemcpy(h_indices.data(), d_indices, batch_out_size * sizeof(int), cudaMemcpyDeviceToHost),
    "cudaMemcpy(indices batch D2H)"
  )) {
    return 1;
  }
  if (check_cuda(
    cudaMemcpy(h_distances.data(), d_distances, batch_out_size * sizeof(float), cudaMemcpyDeviceToHost),
    "cudaMemcpy(distances batch D2H)"
  )) {
    return 1;
  }
  for (int j = 0; j < k; ++j) {
    const std::size_t src_col = static_cast<std::size_t>(j) * batch_n;
    const std::size_t dst_col = static_cast<std::size_t>(j) * total_points + start;
    for (int i = 0; i < batch_n; ++i) {
      out_indices[dst_col + i] = h_indices[src_col + i];
      out_distances[dst_col + i] = static_cast<double>(h_distances[src_col + i]);
    }
  }
  return 0;
}

struct HostCudaGrid {
  int bins = 1;
  int n_features = 2;
  int n_cells = 1;
  float min_x = 0.0f;
  float min_y = 0.0f;
  float min_z = 0.0f;
  float cell_x = 1.0f;
  float cell_y = 1.0f;
  float cell_z = 1.0f;
  std::vector<int> offsets;
  std::vector<int> rows;
};

int host_grid_coord(float value, float min_value, float cell_size, int bins) {
  int out = static_cast<int>((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= bins) out = bins - 1;
  return out;
}

int host_grid_cell(int ix, int iy, int iz, int bins, int n_features) {
  return n_features == 3 ? (iz * bins + iy) * bins + ix : iy * bins + ix;
}

HostCudaGrid build_host_cuda_grid(const double* data,
                                  int n,
                                  int n_features,
                                  int bins) {
  HostCudaGrid grid;
  grid.bins = bins;
  grid.n_features = n_features;
  grid.n_cells = n_features == 3 ? bins * bins * bins : bins * bins;

  float min_x = static_cast<float>(data[0]);
  float max_x = min_x;
  float min_y = static_cast<float>(data[static_cast<std::size_t>(n)]);
  float max_y = min_y;
  float min_z = 0.0f;
  float max_z = 0.0f;
  if (n_features == 3) {
    min_z = static_cast<float>(data[static_cast<std::size_t>(2) * n]);
    max_z = min_z;
  }
  for (int i = 1; i < n; ++i) {
    const float x = static_cast<float>(data[i]);
    const float y = static_cast<float>(data[static_cast<std::size_t>(n) + i]);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    if (n_features == 3) {
      const float z = static_cast<float>(data[static_cast<std::size_t>(2) * n + i]);
      min_z = std::min(min_z, z);
      max_z = std::max(max_z, z);
    }
  }
  grid.min_x = min_x;
  grid.min_y = min_y;
  grid.min_z = min_z;
  grid.cell_x = nextafterf(std::max(max_x - min_x, FLT_EPSILON), kCudaLargeDistance) /
    static_cast<float>(bins);
  grid.cell_y = nextafterf(std::max(max_y - min_y, FLT_EPSILON), kCudaLargeDistance) /
    static_cast<float>(bins);
  grid.cell_z = n_features == 3 ?
    nextafterf(std::max(max_z - min_z, FLT_EPSILON), kCudaLargeDistance) / static_cast<float>(bins) :
    1.0f;

  grid.offsets.assign(static_cast<std::size_t>(grid.n_cells + 1), 0);
  std::vector<int> cell_ids(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int ix = host_grid_coord(static_cast<float>(data[i]), grid.min_x, grid.cell_x, bins);
    const int iy = host_grid_coord(static_cast<float>(data[static_cast<std::size_t>(n) + i]), grid.min_y, grid.cell_y, bins);
    const int iz = n_features == 3 ?
      host_grid_coord(static_cast<float>(data[static_cast<std::size_t>(2) * n + i]), grid.min_z, grid.cell_z, bins) :
      0;
    const int cell = host_grid_cell(ix, iy, iz, bins, n_features);
    cell_ids[static_cast<std::size_t>(i)] = cell;
    ++grid.offsets[static_cast<std::size_t>(cell + 1)];
  }
  for (int c = 1; c <= grid.n_cells; ++c) {
    grid.offsets[static_cast<std::size_t>(c)] += grid.offsets[static_cast<std::size_t>(c - 1)];
  }
  grid.rows.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> cursor = grid.offsets;
  for (int i = 0; i < n; ++i) {
    const int cell = cell_ids[static_cast<std::size_t>(i)];
    grid.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cell)]++)] = i;
  }
  return grid;
}

} // namespace

extern "C" bool fastembedr_cuda_available() {
  int count = 0;
  const cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) {
    set_error(cudaGetErrorString(code));
    return false;
  }
  return count > 0;
}

extern "C" const char* fastembedr_cuda_last_error() {
  return last_error.c_str();
}

extern "C" const char* fastembedr_cuda_device_info_json() {
  int count = 0;
  cudaError_t code = cudaGetDeviceCount(&count);
  if (code != cudaSuccess) {
    const char* error = cudaGetErrorString(code);
    set_error(error == nullptr ? "cudaGetDeviceCount failed" : error);
    std::ostringstream os;
    os << "{\"available\":false,\"device_count\":0,\"reason\":\""
       << json_escape_cuda(error) << "\"}";
    device_info_json = os.str();
    return device_info_json.c_str();
  }
  if (count <= 0) {
    device_info_json = "{\"available\":false,\"device_count\":0,\"reason\":\"no_cuda_device\"}";
    return device_info_json.c_str();
  }

  int device = 0;
  code = cudaGetDevice(&device);
  if (code != cudaSuccess) {
    device = 0;
  }

  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  code = cudaGetDeviceProperties(&prop, device);
  if (code != cudaSuccess) {
    const char* error = cudaGetErrorString(code);
    set_error(error == nullptr ? "cudaGetDeviceProperties failed" : error);
    std::ostringstream os;
    os << "{\"available\":false,\"device_count\":" << count
       << ",\"device\":" << device
       << ",\"reason\":\"" << json_escape_cuda(error) << "\"}";
    device_info_json = os.str();
    return device_info_json.c_str();
  }

  std::size_t free_memory = 0;
  std::size_t total_memory = 0;
  code = cudaMemGetInfo(&free_memory, &total_memory);
  if (code != cudaSuccess) {
    free_memory = 0;
    total_memory = prop.totalGlobalMem;
  }

  std::ostringstream os;
  os << "{\"available\":true"
     << ",\"device_count\":" << count
     << ",\"device\":" << device
     << ",\"name\":\"" << json_escape_cuda(prop.name) << "\""
     << ",\"compute_capability\":\"" << prop.major << "." << prop.minor << "\""
     << ",\"total_memory\":" << static_cast<unsigned long long>(total_memory)
     << ",\"free_memory\":" << static_cast<unsigned long long>(free_memory)
     << ",\"warp_size\":" << prop.warpSize
     << ",\"max_threads_per_block\":" << prop.maxThreadsPerBlock
     << "}";
  device_info_json = os.str();
  return device_info_json.c_str();
}

extern "C" int fastembedr_cuda_knn(const double* data,
                                   const double* points,
                                   int n_data,
                                   int n_points,
                                   int n_features,
                                   int k,
                                   int square,
                                   int* out_indices,
                                   double* out_distances) {
  last_error.clear();
  if (data == nullptr || points == nullptr || out_indices == nullptr || out_distances == nullptr) {
    set_error("null host pointer");
    return 1;
  }
  if (n_data < 1 || n_points < 1 || n_features < 1 || k < 1 || k > n_data || k > kMaxCudaK) {
    set_error("invalid KNN dimensions");
    return 1;
  }

  float* d_data = nullptr;
  float* d_points = nullptr;
  int* d_indices = nullptr;
  float* d_distances = nullptr;
  const bool self_query = data == points && n_data == n_points;
  const std::size_t data_size = static_cast<std::size_t>(n_data) * n_features;
  const std::size_t data_bytes = data_size * sizeof(float);
  const int max_batch = self_query ?
    n_points :
    choose_query_batch_size(n_points, n_features, k, data_bytes);
  if (max_batch < 1) {
    set_error("CUDA KNN allocation preflight: insufficient free device memory for reference data and one query batch");
    return 1;
  }
  const std::size_t batch_points_size = static_cast<std::size_t>(max_batch) * n_features;
  const std::size_t batch_out_size = static_cast<std::size_t>(max_batch) * k;
  const std::size_t points_bytes = self_query ? 0u : batch_points_size * sizeof(float);
  const std::size_t out_i_bytes = batch_out_size * sizeof(int);
  const std::size_t out_d_bytes = batch_out_size * sizeof(float);
  const std::size_t required_bytes = data_bytes + points_bytes + out_i_bytes + out_d_bytes;

  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_points != nullptr && d_points != d_data) cudaFree(d_points);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
  };

  if (check_memory_available(required_bytes, "CUDA KNN allocation preflight")) {
    return 1;
  }

  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), data_bytes), "cudaMalloc(data)")) {
    cleanup();
    return 1;
  }
  if (self_query) {
    d_points = d_data;
  } else {
    if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_points), points_bytes), "cudaMalloc(points)")) {
      cleanup();
      return 1;
    }
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), out_i_bytes), "cudaMalloc(indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), out_d_bytes), "cudaMalloc(distances)")) {
    cleanup();
    return 1;
  }
  if (copy_double_to_float_device(data, d_data, data_size, "cudaMemcpy(data chunk H2D)")) {
    cleanup();
    return 1;
  }

  std::vector<float> query_buffer(static_cast<std::size_t>(max_batch));
  std::vector<int> host_indices(batch_out_size);
  std::vector<float> host_distances(batch_out_size);

  for (int start = 0; start < n_points; start += max_batch) {
    const int batch_n = std::min(max_batch, n_points - start);
    if (!self_query) {
      if (copy_query_batch_to_float_device(
        points,
        d_points,
        n_points,
        n_features,
        start,
        batch_n,
        query_buffer
      )) {
        cleanup();
        return 1;
      }
    }

    const KnnParams params{n_data, batch_n, n_features, k, square ? 1 : 0};
    if (k <= kFastCudaK) {
      const int threads = 64;
      const std::size_t shared_bytes =
        static_cast<std::size_t>(threads) * k * (sizeof(float) + sizeof(int));
      knn_cooperative_query_kernel<<<batch_n, threads, shared_bytes>>>(
        d_data,
        d_points,
        d_indices,
        d_distances,
        params
      );
      if (check_cuda(cudaGetLastError(), "knn_cooperative_query_kernel launch")) {
        cleanup();
        return 1;
      }
    } else {
      const int threads = 128;
      const int blocks = (batch_n + threads - 1) / threads;
      knn_serial_query_kernel<<<blocks, threads>>>(d_data, d_points, d_indices, d_distances, params);
      if (check_cuda(cudaGetLastError(), "knn_serial_query_kernel launch")) {
        cleanup();
        return 1;
      }
    }
    if (copy_batch_results_to_host(
      d_indices,
      d_distances,
      batch_n,
      n_points,
      start,
      k,
      out_indices,
      out_distances,
      host_indices,
      host_distances
    )) {
      cleanup();
      return 1;
    }
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_grid_self_knn(const double* data,
                                             int n,
                                             int n_features,
                                             int k,
                                             int bins_per_dim,
                                             int* out_indices,
                                             double* out_distances,
                                             int* out_n_cells) {
  last_error.clear();
  if (data == nullptr || out_indices == nullptr || out_distances == nullptr) {
    set_error("null host pointer");
    return 1;
  }
  if (n < 2 || (n_features != 2 && n_features != 3) ||
      k < 1 || k >= n || k > kMaxCudaK || bins_per_dim < 1) {
    set_error("invalid CUDA grid KNN dimensions");
    return 1;
  }
  const long long n_cells_ll = n_features == 3 ?
    static_cast<long long>(bins_per_dim) * bins_per_dim * bins_per_dim :
    static_cast<long long>(bins_per_dim) * bins_per_dim;
  if (n_cells_ll > static_cast<long long>(INT_MAX - 1)) {
    set_error("CUDA grid KNN requested too many grid cells");
    return 1;
  }

  HostCudaGrid grid = build_host_cuda_grid(data, n, n_features, bins_per_dim);
  if (out_n_cells != nullptr) *out_n_cells = grid.n_cells;

  float* d_data = nullptr;
  int* d_offsets = nullptr;
  int* d_rows = nullptr;
  int* d_indices = nullptr;
  float* d_distances = nullptr;
  const std::size_t data_size = static_cast<std::size_t>(n) * n_features;
  const std::size_t out_size = static_cast<std::size_t>(n) * k;
  const std::size_t data_bytes = data_size * sizeof(float);
  const std::size_t offsets_bytes = static_cast<std::size_t>(grid.n_cells + 1) * sizeof(int);
  const std::size_t rows_bytes = static_cast<std::size_t>(n) * sizeof(int);
  const std::size_t out_i_bytes = out_size * sizeof(int);
  const std::size_t out_d_bytes = out_size * sizeof(float);
  const std::size_t required_bytes = data_bytes + offsets_bytes + rows_bytes + out_i_bytes + out_d_bytes;

  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_offsets != nullptr) cudaFree(d_offsets);
    if (d_rows != nullptr) cudaFree(d_rows);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
  };

  if (check_memory_available(required_bytes, "CUDA grid KNN allocation preflight")) return 1;
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), data_bytes), "cudaMalloc(grid data)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_offsets), offsets_bytes), "cudaMalloc(grid offsets)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rows), rows_bytes), "cudaMalloc(grid rows)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), out_i_bytes), "cudaMalloc(grid indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), out_d_bytes), "cudaMalloc(grid distances)")) {
    cleanup();
    return 1;
  }
  if (copy_double_to_float_device(data, d_data, data_size, "cudaMemcpy(grid data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_offsets, grid.offsets.data(), offsets_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(grid offsets H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_rows, grid.rows.data(), rows_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(grid rows H2D)")) {
    cleanup();
    return 1;
  }

  const CudaGridParams params{
    n,
    n_features,
    k,
    bins_per_dim,
    grid.min_x,
    grid.min_y,
    grid.min_z,
    grid.cell_x,
    grid.cell_y,
    grid.cell_z
  };
  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  grid_self_knn_kernel<<<blocks, threads>>>(d_data, d_offsets, d_rows, d_indices, d_distances, params);
  if (check_cuda(cudaGetLastError(), "grid_self_knn_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "grid_self_knn_kernel synchronize")) {
    cleanup();
    return 1;
  }

  std::vector<int> host_indices(out_size);
  std::vector<float> host_distances(out_size);
  if (check_cuda(cudaMemcpy(host_indices.data(), d_indices, out_i_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(grid indices D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(host_distances.data(), d_distances, out_d_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(grid distances D2H)")) {
    cleanup();
    return 1;
  }
  for (std::size_t i = 0; i < out_size; ++i) {
    out_indices[i] = host_indices[i];
    out_distances[i] = static_cast<double>(host_distances[i]);
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_landmark_candidate_knn(const double* data,
                                                      const int* projection_indices,
                                                      int n,
                                                      int n_features,
                                                      int projection_k,
                                                      int k,
                                                      int bucket_cols,
                                                      int query_cols,
                                                      int* out_indices,
                                                      double* out_distances) {
  last_error.clear();
  if (data == nullptr || projection_indices == nullptr ||
      out_indices == nullptr || out_distances == nullptr) {
    set_error("null host pointer");
    return 1;
  }
  if (n < 2 || n_features < 1 || projection_k < 1 ||
      k < 1 || k >= n || k > kMaxCudaK) {
    set_error("invalid landmark candidate KNN dimensions");
    return 1;
  }
  bucket_cols = std::max(1, std::min(bucket_cols, projection_k));
  query_cols = std::max(1, std::min(query_cols, projection_k));

  const std::size_t data_size = static_cast<std::size_t>(n) * n_features;
  const std::size_t projection_items = static_cast<std::size_t>(n) * projection_k;
  const std::size_t out_items = static_cast<std::size_t>(n) * k;
  const std::size_t required_bytes =
    data_size * sizeof(float) +
    projection_items * sizeof(int) +
    out_items * (sizeof(int) + sizeof(float));

  float* d_data = nullptr;
  int* d_projection = nullptr;
  int* d_indices = nullptr;
  float* d_distances = nullptr;
  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_projection != nullptr) cudaFree(d_projection);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
  };

  if (check_memory_available(required_bytes, "CUDA landmark candidate KNN allocation preflight")) {
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), data_size * sizeof(float)), "cudaMalloc(landmark data)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_projection), projection_items * sizeof(int)), "cudaMalloc(landmark projection indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), out_items * sizeof(int)), "cudaMalloc(landmark output indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), out_items * sizeof(float)), "cudaMalloc(landmark output distances)")) {
    cleanup();
    return 1;
  }
  if (copy_double_to_float_device(data, d_data, data_size, "cudaMemcpy(landmark data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_projection, projection_indices, projection_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(landmark projection H2D)")) {
    cleanup();
    return 1;
  }

  if (k <= kFastCudaCandidateK) {
    int threads = 128;
    std::size_t shared_bytes =
      static_cast<std::size_t>(threads) * k * (sizeof(float) + sizeof(int)) +
      static_cast<std::size_t>(threads) * sizeof(int);
    std::size_t shared_limit = 48u * 1024u;
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
        prop.sharedMemPerBlock > 0) {
      shared_limit = static_cast<std::size_t>(prop.sharedMemPerBlock);
    }
    while (threads > 32 && shared_bytes > shared_limit) {
      threads /= 2;
      shared_bytes =
        static_cast<std::size_t>(threads) * k * (sizeof(float) + sizeof(int)) +
        static_cast<std::size_t>(threads) * sizeof(int);
    }
    if (shared_bytes <= shared_limit) {
      landmark_candidate_knn_cooperative_kernel<<<n, threads, shared_bytes>>>(
        d_data,
        d_projection,
        d_indices,
        d_distances,
        n,
        n_features,
        k,
        bucket_cols,
        query_cols
      );
      if (check_cuda(cudaGetLastError(), "landmark_candidate_knn_cooperative_kernel launch")) {
        cleanup();
        return 1;
      }
    } else {
      const int serial_threads = 128;
      const int blocks = (n + serial_threads - 1) / serial_threads;
      landmark_candidate_knn_serial_kernel<<<blocks, serial_threads>>>(
        d_data,
        d_projection,
        d_indices,
        d_distances,
        n,
        n_features,
        k,
        bucket_cols,
        query_cols
      );
      if (check_cuda(cudaGetLastError(), "landmark_candidate_knn_serial_kernel launch")) {
        cleanup();
        return 1;
      }
    }
  } else {
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    landmark_candidate_knn_serial_kernel<<<blocks, threads>>>(
      d_data,
      d_projection,
      d_indices,
      d_distances,
      n,
      n_features,
      k,
      bucket_cols,
      query_cols
    );
    if (check_cuda(cudaGetLastError(), "landmark_candidate_knn_serial_kernel launch")) {
      cleanup();
      return 1;
    }
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(landmark candidate KNN)")) {
    cleanup();
    return 1;
  }

  std::vector<int> h_indices(out_items);
  std::vector<float> h_distances(out_items);
  if (check_cuda(cudaMemcpy(h_indices.data(), d_indices, out_items * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(landmark indices D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(h_distances.data(), d_distances, out_items * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(landmark distances D2H)")) {
    cleanup();
    return 1;
  }
  for (std::size_t i = 0; i < out_items; ++i) {
    out_indices[i] = h_indices[i];
    out_distances[i] = static_cast<double>(h_distances[i]);
  }

  cleanup();
  return 0;
}

extern "C" int fastembedr_cuda_row_candidate_knn(const double* data,
                                                 const int* candidate_indices,
                                                 int n,
                                                 int n_features,
                                                 int n_candidates,
                                                 int k,
                                                 int* out_indices,
                                                 double* out_distances) {
  last_error.clear();
  if (data == nullptr || candidate_indices == nullptr ||
      out_indices == nullptr || out_distances == nullptr) {
    set_error("null host pointer");
    return 1;
  }
  if (n < 2 || n_features < 1 || n_candidates < 1 ||
      k < 1 || k >= n || k > kMaxCudaK) {
    set_error("invalid row candidate KNN dimensions");
    return 1;
  }

  const std::size_t data_size = static_cast<std::size_t>(n) * n_features;
  const std::size_t candidate_items = static_cast<std::size_t>(n) * n_candidates;
  const std::size_t out_items = static_cast<std::size_t>(n) * k;
  const std::size_t required_bytes =
    data_size * sizeof(float) +
    candidate_items * sizeof(int) +
    out_items * (sizeof(int) + sizeof(float));

  float* d_data = nullptr;
  int* d_candidates = nullptr;
  int* d_indices = nullptr;
  float* d_distances = nullptr;
  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_candidates != nullptr) cudaFree(d_candidates);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
  };

  if (check_memory_available(required_bytes, "CUDA row candidate KNN allocation preflight")) {
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), data_size * sizeof(float)), "cudaMalloc(row candidate data)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidates), candidate_items * sizeof(int)), "cudaMalloc(row candidate indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), out_items * sizeof(int)), "cudaMalloc(row candidate output indices)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances), out_items * sizeof(float)), "cudaMalloc(row candidate output distances)")) {
    cleanup();
    return 1;
  }
  if (copy_double_to_float_device(data, d_data, data_size, "cudaMemcpy(row candidate data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_candidates, candidate_indices, candidate_items * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(row candidates H2D)")) {
    cleanup();
    return 1;
  }

  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  row_candidate_knn_kernel<<<blocks, threads>>>(
    d_data,
    d_candidates,
    d_indices,
    d_distances,
    n,
    n_features,
    k,
    n_candidates
  );
  if (check_cuda(cudaGetLastError(), "row_candidate_knn_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(row candidate KNN)")) {
    cleanup();
    return 1;
  }

  std::vector<int> h_indices(out_items);
  std::vector<float> h_distances(out_items);
  if (check_cuda(cudaMemcpy(h_indices.data(), d_indices, out_items * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(row candidate indices D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(h_distances.data(), d_distances, out_items * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(row candidate distances D2H)")) {
    cleanup();
    return 1;
  }
  for (std::size_t i = 0; i < out_items; ++i) {
    out_indices[i] = h_indices[i];
    out_distances[i] = static_cast<double>(h_distances[i]);
  }

  cleanup();
  return 0;
}
