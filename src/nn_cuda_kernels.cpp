#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <string>

namespace {

constexpr int kMaxCudaK = 256;

struct KnnParams {
  int n_data;
  int n_points;
  int n_features;
  int k;
  int square;
};

thread_local std::string last_error;

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

__global__ void knn_exact_euclidean_kernel(const double* data,
                                           const double* points,
                                           int* out_idx,
                                           double* out_dist,
                                           KnnParams params) {
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= params.n_points) return;

  double best_dist[kMaxCudaK];
  int best_idx[kMaxCudaK];
  for (int j = 0; j < params.k; ++j) {
    best_dist[j] = CUDART_INF;
    best_idx[j] = INT_MAX;
  }

  for (int i = 0; i < params.n_data; ++i) {
    double dist = 0.0;
    for (int c = 0; c < params.n_features; ++c) {
      const double diff =
        data[static_cast<std::size_t>(c) * params.n_data + i] -
        points[static_cast<std::size_t>(c) * params.n_points + q];
      dist += diff * diff;
    }

    if (dist < best_dist[params.k - 1] ||
        (dist == best_dist[params.k - 1] && i < best_idx[params.k - 1])) {
      int pos = params.k - 1;
      while (pos > 0 &&
             (dist < best_dist[pos - 1] ||
              (dist == best_dist[pos - 1] && i < best_idx[pos - 1]))) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
      }
      best_dist[pos] = dist;
      best_idx[pos] = i;
    }
  }

  for (int j = 0; j < params.k; ++j) {
    const std::size_t offset = static_cast<std::size_t>(j) * params.n_points + q;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = params.square ? best_dist[j] : sqrt(best_dist[j]);
  }
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

  double* d_data = nullptr;
  double* d_points = nullptr;
  int* d_indices = nullptr;
  double* d_distances = nullptr;
  const std::size_t data_bytes = static_cast<std::size_t>(n_data) * n_features * sizeof(double);
  const std::size_t points_bytes = static_cast<std::size_t>(n_points) * n_features * sizeof(double);
  const std::size_t out_i_bytes = static_cast<std::size_t>(n_points) * k * sizeof(int);
  const std::size_t out_d_bytes = static_cast<std::size_t>(n_points) * k * sizeof(double);

  auto cleanup = [&]() {
    if (d_data != nullptr) cudaFree(d_data);
    if (d_points != nullptr) cudaFree(d_points);
    if (d_indices != nullptr) cudaFree(d_indices);
    if (d_distances != nullptr) cudaFree(d_distances);
  };

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
  if (check_cuda(cudaMemcpy(d_data, data, data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(data H2D)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(d_points, points, points_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(points H2D)")) {
    cleanup();
    return 1;
  }

  const KnnParams params{n_data, n_points, n_features, k, square ? 1 : 0};
  const int threads = 128;
  const int blocks = (n_points + threads - 1) / threads;
  knn_exact_euclidean_kernel<<<blocks, threads>>>(d_data, d_points, d_indices, d_distances, params);
  if (check_cuda(cudaGetLastError(), "knn_exact_euclidean_kernel launch")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out_indices, d_indices, out_i_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(indices D2H)")) {
    cleanup();
    return 1;
  }
  if (check_cuda(cudaMemcpy(out_distances, d_distances, out_d_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(distances D2H)")) {
    cleanup();
    return 1;
  }

  cleanup();
  return 0;
}
