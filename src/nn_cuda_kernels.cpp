#include <cuda_runtime.h>

#include <algorithm>
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
  const std::size_t data_size = static_cast<std::size_t>(n_data) * n_features;
  const std::size_t data_bytes = data_size * sizeof(float);
  const int max_batch = choose_query_batch_size(n_points, n_features, k, data_bytes);
  if (max_batch < 1) {
    set_error("CUDA KNN allocation preflight: insufficient free device memory for reference data and one query batch");
    return 1;
  }
  const std::size_t batch_points_size = static_cast<std::size_t>(max_batch) * n_features;
  const std::size_t batch_out_size = static_cast<std::size_t>(max_batch) * k;
  const std::size_t points_bytes = batch_points_size * sizeof(float);
  const std::size_t out_i_bytes = batch_out_size * sizeof(int);
  const std::size_t out_d_bytes = batch_out_size * sizeof(float);
  const std::size_t required_bytes = data_bytes + points_bytes + out_i_bytes + out_d_bytes;

  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_points != nullptr) cudaFree(d_points);
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
  if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_points), points_bytes), "cudaMalloc(points)")) {
    cleanup();
    return 1;
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
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
      cleanup();
      return 1;
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
    const int threads = 128;
    const std::size_t shared_bytes =
      static_cast<std::size_t>(threads) * k * (sizeof(float) + sizeof(int)) +
      static_cast<std::size_t>(threads) * sizeof(int);
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
