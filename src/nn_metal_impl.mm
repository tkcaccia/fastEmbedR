#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

struct KnnParams {
  std::uint32_t n_data;
  std::uint32_t n_points;
  std::uint32_t n_features;
  std::uint32_t k;
  std::uint32_t square;
};

struct CandidateKnnParams {
  std::uint32_t n;
  std::uint32_t n_features;
  std::uint32_t projection_k;
  std::uint32_t k;
  std::uint32_t bucket_cols;
  std::uint32_t query_cols;
};

struct CandidateCsrKnnParams {
  std::uint32_t n;
  std::uint32_t n_features;
  std::uint32_t projection_k;
  std::uint32_t k;
  std::uint32_t query_cols;
};

struct GridKnnParams {
  std::uint32_t n;
  std::uint32_t n_features;
  std::uint32_t k;
  std::uint32_t grid_dims;
  std::uint32_t bins_per_dim;
  std::uint32_t radius;
  std::uint32_t total_bins;
  std::uint32_t max_cells;
};

struct RowCandidateKnnParams {
  std::uint32_t n;
  std::uint32_t n_features;
  std::uint32_t k;
  std::uint32_t n_candidates;
};

struct MetalKnnState {
  id<MTLDevice> device;
  id<MTLComputePipelineState> pipeline;
  id<MTLComputePipelineState> row_major_pipeline;
  id<MTLComputePipelineState> candidate_pipeline;
  id<MTLComputePipelineState> candidate_csr_pipeline;
  id<MTLComputePipelineState> grid_pipeline;
  id<MTLComputePipelineState> row_candidate_pipeline;
  id<MTLCommandQueue> queue;
};

constexpr int kMaxMetalK = 256;
constexpr int kMaxGridDims = 5;
constexpr int kMaxGridBins = 2000000;

const char* metal_kernel_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

#define MAX_K 256

struct KnnParams {
  uint n_data;
  uint n_points;
  uint n_features;
  uint k;
  uint square;
};

struct CandidateKnnParams {
  uint n;
  uint n_features;
  uint projection_k;
  uint k;
  uint bucket_cols;
  uint query_cols;
};

struct CandidateCsrKnnParams {
  uint n;
  uint n_features;
  uint projection_k;
  uint k;
  uint query_cols;
};

struct GridKnnParams {
  uint n;
  uint n_features;
  uint k;
  uint grid_dims;
  uint bins_per_dim;
  uint radius;
  uint total_bins;
  uint max_cells;
};

struct RowCandidateKnnParams {
  uint n;
  uint n_features;
  uint k;
  uint n_candidates;
};

inline bool shares_projection_bucket(
  device const int* projection_indices,
  constant CandidateKnnParams& params,
  uint query,
  uint candidate
) {
  for (uint qc = 0; qc < params.query_cols; ++qc) {
    int q_anchor = projection_indices[qc * params.n + query];
    if (q_anchor < 1) continue;
    for (uint bc = 0; bc < params.bucket_cols; ++bc) {
      if (projection_indices[bc * params.n + candidate] == q_anchor) {
        return true;
      }
    }
  }
  return false;
}

inline float self_distance_sq(
  device const float* data,
  constant CandidateKnnParams& params,
  uint a,
  uint b
) {
  float dist = 0.0f;
  for (uint c = 0; c < params.n_features; ++c) {
    float diff = data[c * params.n + a] - data[c * params.n + b];
    dist += diff * diff;
  }
  return dist;
}

inline float self_distance_sq_raw(
  device const float* data,
  uint n,
  uint n_features,
  uint a,
  uint b
) {
  float dist = 0.0f;
  for (uint c = 0; c < n_features; ++c) {
    float diff = data[c * n + a] - data[c * n + b];
    dist += diff * diff;
  }
  return dist;
}

inline float self_distance_sq_row_major_raw(
  device const float* data,
  uint n_features,
  uint a,
  uint b
) {
  const uint a_offset = a * n_features;
  const uint b_offset = b * n_features;
  float dist = 0.0f;
  for (uint c = 0; c < n_features; ++c) {
    float diff = data[a_offset + c] - data[b_offset + c];
    dist += diff * diff;
  }
  return dist;
}

inline void insert_candidate_metal(
  float dist,
  int idx,
  thread float* best_dist,
  thread int* best_idx,
  uint k
) {
  if (dist > best_dist[k - 1] ||
      (dist == best_dist[k - 1] && idx >= best_idx[k - 1])) {
    return;
  }
  uint pos = k - 1;
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

inline void insert_unique_candidate_metal(
  float dist,
  int idx,
  thread float* best_dist,
  thread int* best_idx,
  uint k
) {
  for (uint j = 0; j < k; ++j) {
    if (best_idx[j] == idx) return;
  }
  insert_candidate_metal(dist, idx, best_dist, best_idx, k);
}

inline bool best_contains_candidate_metal(
  int idx,
  thread int* best_idx,
  uint k
) {
  for (uint j = 0; j < k; ++j) {
    if (best_idx[j] == idx) return true;
  }
  return false;
}

kernel void landmark_candidate_knn_metal(
  device const float* data [[buffer(0)]],
  device const int* projection_indices [[buffer(1)]],
  device int* out_idx [[buffer(2)]],
  device float* out_dist [[buffer(3)]],
  constant CandidateKnnParams& params [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  uint candidate_count = 0;
  for (uint candidate = 0; candidate < params.n; ++candidate) {
    if (candidate == gid) continue;
    if (!shares_projection_bucket(projection_indices, params, gid, candidate)) {
      continue;
    }
    ++candidate_count;
    insert_candidate_metal(
      self_distance_sq(data, params, gid, candidate),
      int(candidate),
      best_dist,
      best_idx,
      params.k
    );
  }

  if (candidate_count < params.k) {
    for (uint j = 0; j < params.k; ++j) {
      best_dist[j] = INFINITY;
      best_idx[j] = INT_MAX;
    }
    for (uint candidate = 0; candidate < params.n; ++candidate) {
      if (candidate == gid) continue;
      insert_candidate_metal(
        self_distance_sq(data, params, gid, candidate),
        int(candidate),
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    const uint offset = j * params.n + gid;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrt(max(best_dist[j], 0.0f));
  }
}

kernel void landmark_candidate_knn_csr_metal(
  device const float* data [[buffer(0)]],
  device const int* projection_indices [[buffer(1)]],
  device const int* bucket_offsets [[buffer(2)]],
  device const int* bucket_members [[buffer(3)]],
  device int* out_idx [[buffer(4)]],
  device float* out_dist [[buffer(5)]],
  constant CandidateCsrKnnParams& params [[buffer(6)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  for (uint qc = 0; qc < params.query_cols; ++qc) {
    int anchor = projection_indices[qc * params.n + gid] - 1;
    if (anchor < 0) continue;
    int begin = bucket_offsets[anchor];
    int end = bucket_offsets[anchor + 1];
    for (int pos = begin; pos < end; ++pos) {
      int candidate = bucket_members[pos];
      if (candidate == int(gid)) continue;
      insert_unique_candidate_metal(
        self_distance_sq_row_major_raw(data, params.n_features, gid, uint(candidate)),
        candidate,
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  if (best_idx[params.k - 1] == INT_MAX) {
    for (uint j = 0; j < params.k; ++j) {
      best_dist[j] = INFINITY;
      best_idx[j] = INT_MAX;
    }
    for (uint candidate = 0; candidate < params.n; ++candidate) {
      if (candidate == gid) continue;
      insert_candidate_metal(
        self_distance_sq_row_major_raw(data, params.n_features, gid, candidate),
        int(candidate),
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    const uint offset = j * params.n + gid;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrt(max(best_dist[j], 0.0f));
  }
}

kernel void grid_candidate_knn_metal(
  device const float* data [[buffer(0)]],
  device const int* bin_coords [[buffer(1)]],
  device const int* bin_offsets [[buffer(2)]],
  device const int* bin_members [[buffer(3)]],
  device int* out_idx [[buffer(4)]],
  device float* out_dist [[buffer(5)]],
  constant GridKnnParams& params [[buffer(6)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  int query_coords[5];
  for (uint d = 0; d < params.grid_dims; ++d) {
    query_coords[d] = bin_coords[gid * params.grid_dims + d];
  }

  const uint side = 2 * params.radius + 1;
  for (uint ordinal = 0; ordinal < params.max_cells; ++ordinal) {
    uint tmp = ordinal;
    uint flat = 0;
    uint stride = 1;
    bool valid = true;

    for (uint d = 0; d < params.grid_dims; ++d) {
      const int delta = int(tmp % side) - int(params.radius);
      tmp /= side;
      const int coord = query_coords[d] + delta;
      if (coord < 0 || coord >= int(params.bins_per_dim)) {
        valid = false;
      }
      flat += uint(max(coord, 0)) * stride;
      stride *= params.bins_per_dim;
    }
    if (!valid || flat >= params.total_bins) continue;

    const int begin = bin_offsets[flat];
    const int end = bin_offsets[flat + 1];
    for (int pos = begin; pos < end; ++pos) {
      const int candidate = bin_members[pos];
      if (candidate == int(gid)) continue;
      insert_candidate_metal(
        self_distance_sq_row_major_raw(data, params.n_features, gid, uint(candidate)),
        candidate,
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  if (best_idx[params.k - 1] == INT_MAX) {
    for (uint j = 0; j < params.k; ++j) {
      best_dist[j] = INFINITY;
      best_idx[j] = INT_MAX;
    }
    for (uint candidate = 0; candidate < params.n; ++candidate) {
      if (candidate == gid) continue;
      insert_candidate_metal(
        self_distance_sq_row_major_raw(data, params.n_features, gid, candidate),
        int(candidate),
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    const uint offset = j * params.n + gid;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrt(max(best_dist[j], 0.0f));
  }
}

kernel void row_candidate_knn_metal(
  device const float* data [[buffer(0)]],
  device const int* candidate_indices [[buffer(1)]],
  device int* out_idx [[buffer(2)]],
  device float* out_dist [[buffer(3)]],
  constant RowCandidateKnnParams& params [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  for (uint c = 0; c < params.n_candidates; ++c) {
    const int candidate = candidate_indices[c * params.n + gid] - 1;
    if (candidate < 0 || candidate >= int(params.n) || candidate == int(gid)) {
      continue;
    }
    if (best_contains_candidate_metal(candidate, best_idx, params.k)) {
      continue;
    }
    insert_candidate_metal(
      self_distance_sq_row_major_raw(data, params.n_features, gid, uint(candidate)),
      candidate,
      best_dist,
      best_idx,
      params.k
    );
  }

  if (best_idx[params.k - 1] == INT_MAX) {
    for (uint candidate = 0; candidate < params.n && best_idx[params.k - 1] == INT_MAX; ++candidate) {
      if (candidate == gid) continue;
      insert_unique_candidate_metal(
        self_distance_sq_row_major_raw(data, params.n_features, gid, candidate),
        int(candidate),
        best_dist,
        best_idx,
        params.k
      );
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    const uint offset = j * params.n + gid;
    out_idx[offset] = best_idx[j] + 1;
    out_dist[offset] = sqrt(max(best_dist[j], 0.0f));
  }
}

kernel void knn_exact_euclidean(
  device const float* data [[buffer(0)]],
  device const float* points [[buffer(1)]],
  device int* out_idx [[buffer(2)]],
  device float* out_dist [[buffer(3)]],
  constant KnnParams& params [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n_points) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  for (uint i = 0; i < params.n_data; ++i) {
    float dist = 0.0f;
    for (uint c = 0; c < params.n_features; ++c) {
      float diff = data[c * params.n_data + i] - points[c * params.n_points + gid];
      dist += diff * diff;
    }

    if (dist < best_dist[params.k - 1] ||
        (dist == best_dist[params.k - 1] && int(i) < best_idx[params.k - 1])) {
      uint pos = params.k - 1;
      while (pos > 0 &&
             (dist < best_dist[pos - 1] ||
              (dist == best_dist[pos - 1] && int(i) < best_idx[pos - 1]))) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
      }
      best_dist[pos] = dist;
      best_idx[pos] = int(i);
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    out_idx[j * params.n_points + gid] = best_idx[j] + 1;
    out_dist[j * params.n_points + gid] = params.square ? best_dist[j] : sqrt(best_dist[j]);
  }
}

kernel void knn_exact_euclidean_row_major(
  device const float* data [[buffer(0)]],
  device const float* points [[buffer(1)]],
  device int* out_idx [[buffer(2)]],
  device float* out_dist [[buffer(3)]],
  constant KnnParams& params [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= params.n_points) return;

  float best_dist[MAX_K];
  int best_idx[MAX_K];
  for (uint j = 0; j < params.k; ++j) {
    best_dist[j] = INFINITY;
    best_idx[j] = INT_MAX;
  }

  const uint point_offset = gid * params.n_features;
  for (uint i = 0; i < params.n_data; ++i) {
    const uint data_offset = i * params.n_features;
    float dist = 0.0f;
    for (uint c = 0; c < params.n_features; ++c) {
      float diff = data[data_offset + c] - points[point_offset + c];
      dist += diff * diff;
    }

    if (dist < best_dist[params.k - 1] ||
        (dist == best_dist[params.k - 1] && int(i) < best_idx[params.k - 1])) {
      uint pos = params.k - 1;
      while (pos > 0 &&
             (dist < best_dist[pos - 1] ||
              (dist == best_dist[pos - 1] && int(i) < best_idx[pos - 1]))) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
      }
      best_dist[pos] = dist;
      best_idx[pos] = int(i);
    }
  }

  for (uint j = 0; j < params.k; ++j) {
    out_idx[j * params.n_points + gid] = best_idx[j] + 1;
    out_dist[j * params.n_points + gid] = params.square ? best_dist[j] : sqrt(best_dist[j]);
  }
}
)METAL";
}

std::vector<float> matrix_to_float(const NumericMatrix& x) {
  const double* ptr = x.begin();
  std::vector<float> out(static_cast<std::size_t>(x.nrow()) * x.ncol());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(ptr[i]);
  return out;
}

std::vector<float> matrix_to_row_major_float(const NumericMatrix& x) {
  std::vector<float> out(static_cast<std::size_t>(x.nrow()) * x.ncol());
  for (int c = 0; c < x.ncol(); ++c) {
    for (int r = 0; r < x.nrow(); ++r) {
      out[static_cast<std::size_t>(r) * x.ncol() + c] = static_cast<float>(x(r, c));
    }
  }
  return out;
}

std::string ns_error_message(NSError* error) {
  if (error == nil) return "";
  NSString* description = [error localizedDescription];
  if (description == nil) return "unknown Metal error";
  return std::string([description UTF8String]);
}

int bounded_grid_bins(int n, int k, int grid_dims, int requested_bins) {
  int bins = requested_bins;
  if (bins <= 0) {
    const double target = 32.0 * static_cast<double>(n) / std::max(1, k);
    bins = static_cast<int>(std::ceil(std::pow(target, 1.0 / std::max(1, grid_dims))));
  }
  bins = std::max(2, std::min(30, bins));
  auto total_for = [grid_dims](int b) {
    double total = 1.0;
    for (int d = 0; d < grid_dims; ++d) total *= static_cast<double>(b);
    return total;
  };
  while (bins > 2 && total_for(bins) > static_cast<double>(kMaxGridBins)) {
    --bins;
  }
  return bins;
}

std::uint32_t pow_uint32(std::uint32_t base, std::uint32_t exp) {
  std::uint32_t out = 1;
  for (std::uint32_t i = 0; i < exp; ++i) out *= base;
  return out;
}

MetalKnnState& metal_knn_state() {
  static MetalKnnState state{nil, nil, nil, nil, nil, nil, nil, nil};
  if (state.device != nil &&
      state.pipeline != nil &&
      state.row_major_pipeline != nil &&
      state.candidate_pipeline != nil &&
      state.candidate_csr_pipeline != nil &&
      state.grid_pipeline != nil &&
      state.row_candidate_pipeline != nil &&
      state.queue != nil) {
    return state;
  }

  state.device = MTLCreateSystemDefaultDevice();
  if (state.device == nil) {
    Rcpp::stop("No Metal device is available.");
  }

  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:metal_kernel_source()];
  id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
  if (library == nil) {
    Rcpp::stop("Failed to compile Metal KNN kernel: %s", ns_error_message(error).c_str());
  }

  id<MTLFunction> function = [library newFunctionWithName:@"knn_exact_euclidean"];
  if (function == nil) Rcpp::stop("Failed to load Metal KNN function.");
  state.pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
  if (state.pipeline == nil) {
    Rcpp::stop("Failed to create Metal KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [function release];

  id<MTLFunction> row_major_function = [library newFunctionWithName:@"knn_exact_euclidean_row_major"];
  if (row_major_function == nil) Rcpp::stop("Failed to load row-major Metal KNN function.");
  state.row_major_pipeline = [state.device newComputePipelineStateWithFunction:row_major_function error:&error];
  if (state.row_major_pipeline == nil) {
    Rcpp::stop("Failed to create row-major Metal KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [row_major_function release];

  id<MTLFunction> candidate_function = [library newFunctionWithName:@"landmark_candidate_knn_metal"];
  if (candidate_function == nil) Rcpp::stop("Failed to load Metal candidate KNN function.");
  state.candidate_pipeline = [state.device newComputePipelineStateWithFunction:candidate_function error:&error];
  if (state.candidate_pipeline == nil) {
    Rcpp::stop("Failed to create Metal candidate KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [candidate_function release];

  id<MTLFunction> candidate_csr_function = [library newFunctionWithName:@"landmark_candidate_knn_csr_metal"];
  if (candidate_csr_function == nil) Rcpp::stop("Failed to load Metal CSR candidate KNN function.");
  state.candidate_csr_pipeline = [state.device newComputePipelineStateWithFunction:candidate_csr_function error:&error];
  if (state.candidate_csr_pipeline == nil) {
    Rcpp::stop("Failed to create Metal CSR candidate KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [candidate_csr_function release];

  id<MTLFunction> grid_function = [library newFunctionWithName:@"grid_candidate_knn_metal"];
  if (grid_function == nil) Rcpp::stop("Failed to load Metal grid candidate KNN function.");
  state.grid_pipeline = [state.device newComputePipelineStateWithFunction:grid_function error:&error];
  if (state.grid_pipeline == nil) {
    Rcpp::stop("Failed to create Metal grid candidate KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [grid_function release];

  id<MTLFunction> row_candidate_function = [library newFunctionWithName:@"row_candidate_knn_metal"];
  if (row_candidate_function == nil) Rcpp::stop("Failed to load Metal row-candidate KNN function.");
  state.row_candidate_pipeline = [state.device newComputePipelineStateWithFunction:row_candidate_function error:&error];
  if (state.row_candidate_pipeline == nil) {
    Rcpp::stop("Failed to create Metal row-candidate KNN pipeline: %s", ns_error_message(error).c_str());
  }
  [row_candidate_function release];

  [library release];

  state.queue = [state.device newCommandQueue];
  if (state.queue == nil) {
    Rcpp::stop("Failed to create Metal command queue.");
  }

  return state;
}

List metal_landmark_candidate_knn_impl_internal(NumericMatrix data,
                                                IntegerMatrix projection_indices,
                                                int k,
                                                int bucket_cols,
                                                int query_cols) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int projection_k = projection_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must match data");
  }
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxMetalK) Rcpp::stop("Metal backend currently supports k <= %d", kMaxMetalK);
  bucket_cols = std::max(1, std::min(bucket_cols, projection_k));
  query_cols = std::max(1, std::min(query_cols, projection_k));

  int n_landmarks = 0;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < projection_k; ++c) {
      const int idx = projection_indices(i, c);
      if (idx < 1) Rcpp::stop("projection_indices must be 1-based positive integers");
      n_landmarks = std::max(n_landmarks, idx);
    }
  }

  std::vector<std::int32_t> bucket_offsets(static_cast<std::size_t>(n_landmarks) + 1, 0);
  std::vector<int> used_landmarks;
  used_landmarks.reserve(bucket_cols);
  for (int i = 0; i < n; ++i) {
    used_landmarks.clear();
    for (int c = 0; c < bucket_cols; ++c) {
      const int landmark = projection_indices(i, c) - 1;
      if (landmark < 0 || landmark >= n_landmarks) continue;
      if (std::find(used_landmarks.begin(), used_landmarks.end(), landmark) != used_landmarks.end()) {
        continue;
      }
      used_landmarks.push_back(landmark);
      ++bucket_offsets[static_cast<std::size_t>(landmark) + 1];
    }
  }
  for (int i = 0; i < n_landmarks; ++i) {
    bucket_offsets[static_cast<std::size_t>(i) + 1] += bucket_offsets[static_cast<std::size_t>(i)];
  }
  std::vector<std::int32_t> bucket_members(static_cast<std::size_t>(bucket_offsets.back()));
  std::vector<std::int32_t> cursor = bucket_offsets;
  for (int i = 0; i < n; ++i) {
    used_landmarks.clear();
    for (int c = 0; c < bucket_cols; ++c) {
      const int landmark = projection_indices(i, c) - 1;
      if (landmark < 0 || landmark >= n_landmarks) continue;
      if (std::find(used_landmarks.begin(), used_landmarks.end(), landmark) != used_landmarks.end()) {
        continue;
      }
      used_landmarks.push_back(landmark);
      const std::size_t write = static_cast<std::size_t>(cursor[static_cast<std::size_t>(landmark)]++);
      bucket_members[write] = static_cast<std::int32_t>(i);
    }
  }

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();
    std::vector<float> data_f = matrix_to_row_major_float(data);
    std::vector<std::int32_t> out_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> out_dist(static_cast<std::size_t>(n) * k);
    CandidateCsrKnnParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(n_features),
      static_cast<std::uint32_t>(projection_k),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(query_cols)
    };

    id<MTLBuffer> data_buffer = [state.device newBufferWithBytesNoCopy:data_f.data()
                                                                 length:data_f.size() * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (data_buffer == nil) {
      data_buffer = [state.device newBufferWithBytes:data_f.data()
                                             length:data_f.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> projection_buffer = [state.device newBufferWithBytes:projection_indices.begin()
                                                                length:static_cast<std::size_t>(n) * projection_k * sizeof(std::int32_t)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsets_buffer = [state.device newBufferWithBytes:bucket_offsets.data()
                                                             length:bucket_offsets.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> members_buffer = [state.device newBufferWithBytes:bucket_members.data()
                                                             length:bucket_members.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx_buffer = [state.device newBufferWithLength:out_idx.size() * sizeof(std::int32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist_buffer = [state.device newBufferWithLength:out_dist.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(CandidateCsrKnnParams)
                                                           options:MTLResourceStorageModeShared];
    if (data_buffer == nil || projection_buffer == nil || offsets_buffer == nil ||
        members_buffer == nil || idx_buffer == nil || dist_buffer == nil ||
        params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal candidate KNN buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.candidate_csr_pipeline];
    [encoder setBuffer:data_buffer offset:0 atIndex:0];
    [encoder setBuffer:projection_buffer offset:0 atIndex:1];
    [encoder setBuffer:offsets_buffer offset:0 atIndex:2];
    [encoder setBuffer:members_buffer offset:0 atIndex:3];
    [encoder setBuffer:idx_buffer offset:0 atIndex:4];
    [encoder setBuffer:dist_buffer offset:0 atIndex:5];
    [encoder setBuffer:params_buffer offset:0 atIndex:6];

    const NSUInteger threads_per_group = std::min<NSUInteger>(
      state.candidate_csr_pipeline.maxTotalThreadsPerThreadgroup,
      256
    );
    MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal candidate KNN command failed: %s", ns_error_message(command_buffer.error).c_str());
    }

    std::memcpy(out_idx.data(), [idx_buffer contents], out_idx.size() * sizeof(std::int32_t));
    std::memcpy(out_dist.data(), [dist_buffer contents], out_dist.size() * sizeof(float));

    IntegerMatrix indices(n, k);
    NumericMatrix distances(n, k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t offset = static_cast<std::size_t>(j) * n + i;
        indices(i, j) = out_idx[offset];
        distances(i, j) = static_cast<double>(out_dist[offset]);
      }
    }

    List result = List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
    result.attr("metal_kernel") = "anchor_candidate_csr";
    result.attr("candidate_entries") = static_cast<double>(bucket_members.size());
    result.attr("candidate_landmarks") = n_landmarks;
    [data_buffer release];
    [projection_buffer release];
    [offsets_buffer release];
    [members_buffer release];
    [idx_buffer release];
    [dist_buffer release];
    [params_buffer release];
    return result;
  }
}

List metal_grid_knn_impl_internal(NumericMatrix data,
                                  int k,
                                  int grid_dims,
                                  int bins_per_dim,
                                  int radius) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxMetalK) Rcpp::stop("Metal backend currently supports k <= %d", kMaxMetalK);
  grid_dims = grid_dims <= 0 ? std::min(kMaxGridDims, n_features) : grid_dims;
  grid_dims = std::max(1, std::min({kMaxGridDims, n_features, grid_dims}));
  bins_per_dim = bounded_grid_bins(n, k, grid_dims, bins_per_dim);
  radius = radius <= 0 ? 1 : radius;
  radius = std::max(1, std::min(4, radius));

  const std::uint32_t total_bins = pow_uint32(
    static_cast<std::uint32_t>(bins_per_dim),
    static_cast<std::uint32_t>(grid_dims)
  );
  if (total_bins > static_cast<std::uint32_t>(kMaxGridBins)) {
    Rcpp::stop("Metal grid KNN generated too many bins; lower `fastEmbedR.grid_dims` or `fastEmbedR.grid_bins`.");
  }
  const std::uint32_t max_cells = pow_uint32(
    static_cast<std::uint32_t>(2 * radius + 1),
    static_cast<std::uint32_t>(grid_dims)
  );

  std::vector<double> min_values(grid_dims, R_PosInf);
  std::vector<double> max_values(grid_dims, R_NegInf);
  for (int d = 0; d < grid_dims; ++d) {
    for (int i = 0; i < n; ++i) {
      const double value = data(i, d);
      min_values[d] = std::min(min_values[d], value);
      max_values[d] = std::max(max_values[d], value);
    }
  }

  std::vector<std::int32_t> bin_coords(static_cast<std::size_t>(n) * grid_dims);
  std::vector<std::int32_t> flat_bins(n);
  std::vector<std::int32_t> bin_offsets(static_cast<std::size_t>(total_bins) + 1, 0);
  for (int i = 0; i < n; ++i) {
    int flat = 0;
    int stride = 1;
    for (int d = 0; d < grid_dims; ++d) {
      const double range = max_values[d] - min_values[d];
      int coord = 0;
      if (range > 0.0) {
        const double scaled = (data(i, d) - min_values[d]) / range;
        coord = static_cast<int>(std::floor(scaled * bins_per_dim));
        coord = std::max(0, std::min(bins_per_dim - 1, coord));
      }
      bin_coords[static_cast<std::size_t>(i) * grid_dims + d] = coord;
      flat += coord * stride;
      stride *= bins_per_dim;
    }
    flat_bins[i] = flat;
    ++bin_offsets[static_cast<std::size_t>(flat) + 1];
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(total_bins); ++i) {
    bin_offsets[i + 1] += bin_offsets[i];
  }
  std::vector<std::int32_t> bin_members(static_cast<std::size_t>(n));
  std::vector<std::int32_t> cursor = bin_offsets;
  for (int i = 0; i < n; ++i) {
    const std::size_t write = static_cast<std::size_t>(cursor[static_cast<std::size_t>(flat_bins[i])]++);
    bin_members[write] = static_cast<std::int32_t>(i);
  }

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();
    std::vector<float> data_f = matrix_to_row_major_float(data);
    std::vector<std::int32_t> out_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> out_dist(static_cast<std::size_t>(n) * k);
    GridKnnParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(n_features),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(grid_dims),
      static_cast<std::uint32_t>(bins_per_dim),
      static_cast<std::uint32_t>(radius),
      total_bins,
      max_cells
    };

    id<MTLBuffer> data_buffer = [state.device newBufferWithBytesNoCopy:data_f.data()
                                                                 length:data_f.size() * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (data_buffer == nil) {
      data_buffer = [state.device newBufferWithBytes:data_f.data()
                                             length:data_f.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> coords_buffer = [state.device newBufferWithBytes:bin_coords.data()
                                                            length:bin_coords.size() * sizeof(std::int32_t)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsets_buffer = [state.device newBufferWithBytes:bin_offsets.data()
                                                             length:bin_offsets.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> members_buffer = [state.device newBufferWithBytes:bin_members.data()
                                                             length:bin_members.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx_buffer = [state.device newBufferWithLength:out_idx.size() * sizeof(std::int32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist_buffer = [state.device newBufferWithLength:out_dist.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(GridKnnParams)
                                                           options:MTLResourceStorageModeShared];
    if (data_buffer == nil || coords_buffer == nil || offsets_buffer == nil ||
        members_buffer == nil || idx_buffer == nil || dist_buffer == nil ||
        params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal grid KNN buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.grid_pipeline];
    [encoder setBuffer:data_buffer offset:0 atIndex:0];
    [encoder setBuffer:coords_buffer offset:0 atIndex:1];
    [encoder setBuffer:offsets_buffer offset:0 atIndex:2];
    [encoder setBuffer:members_buffer offset:0 atIndex:3];
    [encoder setBuffer:idx_buffer offset:0 atIndex:4];
    [encoder setBuffer:dist_buffer offset:0 atIndex:5];
    [encoder setBuffer:params_buffer offset:0 atIndex:6];

    const NSUInteger threads_per_group = std::min<NSUInteger>(
      state.grid_pipeline.maxTotalThreadsPerThreadgroup,
      256
    );
    MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal grid KNN command failed: %s", ns_error_message(command_buffer.error).c_str());
    }

    std::memcpy(out_idx.data(), [idx_buffer contents], out_idx.size() * sizeof(std::int32_t));
    std::memcpy(out_dist.data(), [dist_buffer contents], out_dist.size() * sizeof(float));

    IntegerMatrix indices(n, k);
    NumericMatrix distances(n, k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t offset = static_cast<std::size_t>(j) * n + i;
        indices(i, j) = out_idx[offset];
        distances(i, j) = static_cast<double>(out_dist[offset]);
      }
    }

    List result = List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
    result.attr("metal_kernel") = "grid_bin_candidate";
    result.attr("grid_dims") = grid_dims;
    result.attr("grid_bins") = bins_per_dim;
    result.attr("grid_radius") = radius;
    result.attr("grid_total_bins") = static_cast<double>(total_bins);
    result.attr("grid_max_cells") = static_cast<double>(max_cells);
    [data_buffer release];
    [coords_buffer release];
    [offsets_buffer release];
    [members_buffer release];
    [idx_buffer release];
    [dist_buffer release];
    [params_buffer release];
    return result;
  }
}

List metal_row_candidate_knn_impl_internal(NumericMatrix data,
                                           IntegerMatrix candidate_indices,
                                           int k) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int n_candidates = candidate_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (candidate_indices.nrow() != n) {
    Rcpp::stop("candidate_indices row count must match data");
  }
  if (n_candidates < 1) Rcpp::stop("candidate_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxMetalK) Rcpp::stop("Metal backend currently supports k <= %d", kMaxMetalK);

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();
    std::vector<float> data_f = matrix_to_row_major_float(data);
    std::vector<std::int32_t> out_idx(static_cast<std::size_t>(n) * k);
    std::vector<float> out_dist(static_cast<std::size_t>(n) * k);
    RowCandidateKnnParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(n_features),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(n_candidates)
    };

    id<MTLBuffer> data_buffer = [state.device newBufferWithBytesNoCopy:data_f.data()
                                                                 length:data_f.size() * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (data_buffer == nil) {
      data_buffer = [state.device newBufferWithBytes:data_f.data()
                                             length:data_f.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> candidates_buffer = [state.device newBufferWithBytes:candidate_indices.begin()
                                                                length:static_cast<std::size_t>(n) * n_candidates * sizeof(std::int32_t)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx_buffer = [state.device newBufferWithLength:out_idx.size() * sizeof(std::int32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist_buffer = [state.device newBufferWithLength:out_dist.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(RowCandidateKnnParams)
                                                           options:MTLResourceStorageModeShared];
    if (data_buffer == nil || candidates_buffer == nil || idx_buffer == nil ||
        dist_buffer == nil || params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal row-candidate KNN buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.row_candidate_pipeline];
    [encoder setBuffer:data_buffer offset:0 atIndex:0];
    [encoder setBuffer:candidates_buffer offset:0 atIndex:1];
    [encoder setBuffer:idx_buffer offset:0 atIndex:2];
    [encoder setBuffer:dist_buffer offset:0 atIndex:3];
    [encoder setBuffer:params_buffer offset:0 atIndex:4];

    const NSUInteger threads_per_group = std::min<NSUInteger>(
      state.row_candidate_pipeline.maxTotalThreadsPerThreadgroup,
      256
    );
    MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal row-candidate KNN command failed: %s", ns_error_message(command_buffer.error).c_str());
    }

    std::memcpy(out_idx.data(), [idx_buffer contents], out_idx.size() * sizeof(std::int32_t));
    std::memcpy(out_dist.data(), [dist_buffer contents], out_dist.size() * sizeof(float));

    IntegerMatrix indices(n, k);
    NumericMatrix distances(n, k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t offset = static_cast<std::size_t>(j) * n + i;
        indices(i, j) = out_idx[offset];
        distances(i, j) = static_cast<double>(out_dist[offset]);
      }
    }

    List result = List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
    result.attr("metal_kernel") = "row_candidate_knn";
    result.attr("candidate_columns") = n_candidates;
    [data_buffer release];
    [candidates_buffer release];
    [idx_buffer release];
    [dist_buffer release];
    [params_buffer release];
    return result;
  }
}

} // namespace

bool metal_is_available_impl() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

List metal_landmark_candidate_knn_impl(NumericMatrix data,
                                       IntegerMatrix projection_indices,
                                       int k,
                                       int bucket_cols,
                                       int query_cols) {
  return metal_landmark_candidate_knn_impl_internal(
    data, projection_indices, k, bucket_cols, query_cols
  );
}

List metal_grid_knn_impl(NumericMatrix data,
                         int k,
                         int grid_dims,
                         int bins_per_dim,
                         int radius) {
  return metal_grid_knn_impl_internal(data, k, grid_dims, bins_per_dim, radius);
}

List metal_row_candidate_knn_impl(NumericMatrix data,
                                  IntegerMatrix candidate_indices,
                                  int k) {
  return metal_row_candidate_knn_impl_internal(data, candidate_indices, k);
}

List metal_nn_impl(NumericMatrix data,
                   NumericMatrix points,
                   int k,
                   bool square) {
  if (data.ncol() != points.ncol()) Rcpp::stop("data and points must have the same number of columns");
  if (k < 1 || k > data.nrow()) Rcpp::stop("k must be in [1, nrow(data)]");
  if (k > kMaxMetalK) Rcpp::stop("Metal backend currently supports k <= %d", kMaxMetalK);

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();

    const int n_data = data.nrow();
    const int n_points = points.nrow();
    const int n_features = data.ncol();
    const double work_size =
      static_cast<double>(n_data) * static_cast<double>(n_points) * static_cast<double>(n_features);
    const bool use_row_major_kernel =
      n_features >= 4 &&
      work_size >= 1e7 &&
      state.row_major_pipeline.maxTotalThreadsPerThreadgroup >= 1;
    std::vector<float> data_f = use_row_major_kernel ?
      matrix_to_row_major_float(data) :
      matrix_to_float(data);
    std::vector<float> points_f = use_row_major_kernel ?
      matrix_to_row_major_float(points) :
      matrix_to_float(points);
    std::vector<std::int32_t> out_idx(static_cast<std::size_t>(n_points) * k);
    std::vector<float> out_dist(static_cast<std::size_t>(n_points) * k);
    KnnParams params{
      static_cast<std::uint32_t>(n_data),
      static_cast<std::uint32_t>(n_points),
      static_cast<std::uint32_t>(n_features),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(square ? 1 : 0)
    };

    id<MTLBuffer> data_buffer = [state.device newBufferWithBytesNoCopy:data_f.data()
                                                                 length:data_f.size() * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (data_buffer == nil) {
      data_buffer = [state.device newBufferWithBytes:data_f.data()
                                             length:data_f.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> points_buffer = [state.device newBufferWithBytesNoCopy:points_f.data()
                                                                   length:points_f.size() * sizeof(float)
                                                                  options:MTLResourceStorageModeShared
                                                              deallocator:nil];
    if (points_buffer == nil) {
      points_buffer = [state.device newBufferWithBytes:points_f.data()
                                               length:points_f.size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> idx_buffer = [state.device newBufferWithLength:out_idx.size() * sizeof(std::int32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist_buffer = [state.device newBufferWithLength:out_dist.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(KnnParams)
                                                           options:MTLResourceStorageModeShared];
    if (data_buffer == nil || points_buffer == nil || idx_buffer == nil ||
        dist_buffer == nil || params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = use_row_major_kernel ? state.row_major_pipeline : state.pipeline;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data_buffer offset:0 atIndex:0];
    [encoder setBuffer:points_buffer offset:0 atIndex:1];
    [encoder setBuffer:idx_buffer offset:0 atIndex:2];
    [encoder setBuffer:dist_buffer offset:0 atIndex:3];
    [encoder setBuffer:params_buffer offset:0 atIndex:4];

    const NSUInteger threads_per_group = std::min<NSUInteger>(
      pipeline.maxTotalThreadsPerThreadgroup,
      256
    );
    MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n_points), 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal KNN command failed: %s", ns_error_message(command_buffer.error).c_str());
    }

    std::memcpy(out_idx.data(), [idx_buffer contents], out_idx.size() * sizeof(std::int32_t));
    std::memcpy(out_dist.data(), [dist_buffer contents], out_dist.size() * sizeof(float));

    IntegerMatrix indices(n_points, k);
    NumericMatrix distances(n_points, k);
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n_points; ++i) {
        const std::size_t offset = static_cast<std::size_t>(j) * n_points + i;
        indices(i, j) = out_idx[offset];
        distances(i, j) = static_cast<double>(out_dist[offset]);
      }
    }

    List result = List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
    result.attr("metal_kernel") = use_row_major_kernel ? "row_major_exact" : "scalar_exact";
    [data_buffer release];
    [points_buffer release];
    [idx_buffer release];
    [dist_buffer release];
    [params_buffer release];
    return result;
  }
}
