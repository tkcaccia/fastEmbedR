#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <Rcpp.h>
#include <algorithm>
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

constexpr int kMaxMetalK = 256;

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
)METAL";
}

std::vector<float> matrix_to_float(const NumericMatrix& x) {
  const double* ptr = x.begin();
  std::vector<float> out(static_cast<std::size_t>(x.nrow()) * x.ncol());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(ptr[i]);
  return out;
}

std::string ns_error_message(NSError* error) {
  if (error == nil) return "";
  NSString* description = [error localizedDescription];
  if (description == nil) return "unknown Metal error";
  return std::string([description UTF8String]);
}

} // namespace

bool metal_is_available_impl() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

List metal_nn_impl(NumericMatrix data,
                   NumericMatrix points,
                   int k,
                   bool square) {
  if (data.ncol() != points.ncol()) Rcpp::stop("data and points must have the same number of columns");
  if (k < 1 || k > data.nrow()) Rcpp::stop("k must be in [1, nrow(data)]");
  if (k > kMaxMetalK) Rcpp::stop("Metal backend currently supports k <= %d", kMaxMetalK);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      Rcpp::stop("No Metal device is available.");
    }

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_kernel_source()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      Rcpp::stop("Failed to compile Metal KNN kernel: %s", ns_error_message(error).c_str());
    }

    id<MTLFunction> function = [library newFunctionWithName:@"knn_exact_euclidean"];
    if (function == nil) {
      Rcpp::stop("Failed to load Metal KNN function.");
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      Rcpp::stop("Failed to create Metal KNN pipeline: %s", ns_error_message(error).c_str());
    }

    const int n_data = data.nrow();
    const int n_points = points.nrow();
    const int n_features = data.ncol();
    std::vector<float> data_f = matrix_to_float(data);
    std::vector<float> points_f = matrix_to_float(points);
    std::vector<std::int32_t> out_idx(static_cast<std::size_t>(n_points) * k);
    std::vector<float> out_dist(static_cast<std::size_t>(n_points) * k);
    KnnParams params{
      static_cast<std::uint32_t>(n_data),
      static_cast<std::uint32_t>(n_points),
      static_cast<std::uint32_t>(n_features),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(square ? 1 : 0)
    };

    id<MTLBuffer> data_buffer = [device newBufferWithBytes:data_f.data()
                                                    length:data_f.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> points_buffer = [device newBufferWithBytes:points_f.data()
                                                      length:points_f.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx_buffer = [device newBufferWithLength:out_idx.size() * sizeof(std::int32_t)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> dist_buffer = [device newBufferWithLength:out_dist.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params
                                                      length:sizeof(KnnParams)
                                                     options:MTLResourceStorageModeShared];
    if (data_buffer == nil || points_buffer == nil || idx_buffer == nil ||
        dist_buffer == nil || params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal buffers.");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
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

    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
  }
}
