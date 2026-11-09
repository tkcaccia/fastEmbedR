#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "graph_clustering_metal_api.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kColorCount = 8;

struct MetalParams {
  std::uint32_t n;
  std::uint32_t color;
  std::uint32_t seed;
  std::uint32_t refinement;
  std::uint32_t leiden_cuts;
  float graph_volume;
  float resolution;
  float tolerance;
  std::uint32_t first;
  std::uint32_t rows;
};

struct MetalGraphState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> proposal_pipeline = nil;
  id<MTLComputePipelineState> apply_pipeline = nil;
  std::string error;
  bool initialized = false;
};

MetalGraphState& state() {
  static MetalGraphState value;
  return value;
}

std::mutex& state_mutex() {
  static std::mutex value;
  return value;
}

const char* metal_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint color_count = 8;
constant int local_candidate_count = 64;

struct Params {
  uint n;
  uint color;
  uint seed;
  uint refinement;
  uint leiden_cuts;
  float graph_volume;
  float resolution;
  float tolerance;
  uint first;
  uint rows;
};

inline uint mix32(uint value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

inline uint vertex_color(uint vertex_id, uint seed) {
  return mix32(vertex_id ^ seed) & (color_count - 1);
}

kernel void graph_propose(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_idx [[buffer(1)]],
    device const float* edge_weight [[buffer(2)]],
    device const float* degree [[buffer(3)]],
    device const int* labels [[buffer(4)]],
    device const int* parent [[buffer(5)]],
    device const atomic_float* community_volume [[buffer(6)]],
    device const atomic_int* community_count [[buffer(7)]],
    device int* proposal [[buffer(8)]],
    constant Params& params [[buffer(9)]],
    device const atomic_float* cuts [[buffer(10)]],
    device const float* parent_volume [[buffer(11)]],
    device float* proposal_weight [[buffer(12)]],
    uint local [[thread_position_in_grid]]) {
  if (local >= params.rows) return;
  const uint u = params.first + local;
  if (u >= params.n || vertex_color(u, params.seed) != params.color) return;

  const int old = labels[u];
  proposal[u] = old;
  const float node_degree = degree[u];
  if (!(node_degree > 0.0f)) return;
  if (params.refinement &&
      atomic_load_explicit(&community_count[old], memory_order_relaxed) != 1) {
    return;
  }
  if (params.leiden_cuts) {
    const float volume = parent_volume[parent[u]];
    const float threshold = params.resolution * node_degree *
      (volume - node_degree) / params.graph_volume;
    if (atomic_load_explicit(&cuts[old], memory_order_relaxed) +
        params.tolerance < threshold) return;
  }

  const int begin = row_ptr[local];
  const int end = row_ptr[local + 1];
  int candidate_labels[local_candidate_count];
  float candidate_weights[local_candidate_count];
  int unique_count = 0;
  bool overflow = false;
  float old_inside = 0.0f;
  for (int position = begin; position < end; ++position) {
    const int v = col_idx[position];
    if (v == int(u) ||
        (params.refinement && parent[v] != parent[u])) {
      continue;
    }
    const int candidate = labels[v];
    if (candidate == old) old_inside += edge_weight[position];
    int slot = -1;
    for (int index = 0; index < unique_count; ++index) {
      if (candidate_labels[index] == candidate) {
        slot = index;
        break;
      }
    }
    if (slot >= 0) {
      candidate_weights[slot] += edge_weight[position];
    } else if (unique_count < local_candidate_count) {
      candidate_labels[unique_count] = candidate;
      candidate_weights[unique_count] = edge_weight[position];
      ++unique_count;
    } else {
      overflow = true;
    }
  }
  const float old_total = atomic_load_explicit(
    &community_volume[old], memory_order_relaxed
  );
  const float old_volume = max(0.0f, old_total - node_degree);
  const float old_score = old_inside - params.resolution * node_degree *
    old_volume / params.graph_volume;
  float best_score = params.leiden_cuts ? -INFINITY : old_score;
  int best = old;
  float best_weight = 0.0f;

  if (!overflow) {
    for (int index = 0; index < unique_count; ++index) {
      const int candidate = candidate_labels[index];
      if (candidate == old ||
          (params.leiden_cuts &&
           vertex_color(candidate, params.seed) >= params.color) ||
          atomic_load_explicit(
            &community_count[candidate], memory_order_relaxed
          ) <= 0) {
        continue;
      }
      const float candidate_volume = atomic_load_explicit(
        &community_volume[candidate], memory_order_relaxed
      );
      if (params.leiden_cuts) {
        const float parent_total = parent_volume[parent[u]];
        const float threshold = params.resolution * candidate_volume *
          (parent_total - candidate_volume) / params.graph_volume;
        if (atomic_load_explicit(&cuts[candidate],
            memory_order_relaxed) + params.tolerance < threshold) continue;
      }
      const float score = candidate_weights[index] -
        params.resolution * node_degree *
          candidate_volume / params.graph_volume;
      if (params.leiden_cuts && score < -params.tolerance) continue;
      if (score > best_score + params.tolerance ||
          (fabs(score - best_score) <= params.tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
        best_weight = candidate_weights[index];
      }
    }
  } else {
    for (int candidate_position = begin;
         candidate_position < end; ++candidate_position) {
      const int neighbor = col_idx[candidate_position];
      if (neighbor == int(u)) continue;
      if (params.refinement && parent[neighbor] != parent[u]) continue;
      const int candidate = labels[neighbor];
      if (candidate == old ||
          (params.leiden_cuts &&
           vertex_color(candidate, params.seed) >= params.color) ||
          atomic_load_explicit(
            &community_count[candidate], memory_order_relaxed
          ) <= 0) {
        continue;
      }
      bool seen = false;
      for (int previous = begin; previous < candidate_position; ++previous) {
        const int previous_neighbor = col_idx[previous];
        if (previous_neighbor != int(u) &&
            (!params.refinement || parent[previous_neighbor] == parent[u]) &&
            labels[previous_neighbor] == candidate) {
          seen = true;
          break;
        }
      }
      if (seen) continue;
      float inside = 0.0f;
      for (int position = begin; position < end; ++position) {
        const int v = col_idx[position];
        if (v != int(u) &&
            (!params.refinement || parent[v] == parent[u]) &&
            labels[v] == candidate) {
          inside += edge_weight[position];
        }
      }
      const float candidate_volume = atomic_load_explicit(
        &community_volume[candidate], memory_order_relaxed
      );
      if (params.leiden_cuts) {
        const float parent_total = parent_volume[parent[u]];
        const float threshold = params.resolution * candidate_volume *
          (parent_total - candidate_volume) / params.graph_volume;
        if (atomic_load_explicit(&cuts[candidate],
            memory_order_relaxed) + params.tolerance < threshold) continue;
      }
      const float score = inside - params.resolution * node_degree *
        candidate_volume / params.graph_volume;
      if (params.leiden_cuts && score < -params.tolerance) continue;
      if (score > best_score + params.tolerance ||
          (fabs(score - best_score) <= params.tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
        best_weight = inside;
      }
    }
  }
  if (best != old && (params.leiden_cuts ||
      best_score > old_score + params.tolerance)) {
    proposal[u] = best;
    if (params.leiden_cuts) proposal_weight[u] = best_weight;
  }
}

kernel void graph_apply(
    device const float* degree [[buffer(0)]],
    device int* labels [[buffer(1)]],
    device const int* proposal [[buffer(2)]],
    device atomic_float* community_volume [[buffer(3)]],
    device atomic_int* community_count [[buffer(4)]],
    device atomic_uint* move_count [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    device atomic_float* cuts [[buffer(7)]],
    device const float* proposal_weight [[buffer(8)]],
    device const int* row_ptr [[buffer(9)]],
    device const int* col_idx [[buffer(10)]],
    device const float* edge_weight [[buffer(11)]],
    uint local [[thread_position_in_grid]]) {
  if (local >= params.rows) return;
  const uint u = params.first + local;
  if (u >= params.n || vertex_color(u, params.seed) != params.color) return;
  const int old = labels[u];
  const int next = proposal[u];
  if (next == old) return;
  if (params.refinement &&
      atomic_load_explicit(&community_count[old], memory_order_relaxed) != 1) {
    return;
  }
  float old_cut = 0.0f;
  if (params.leiden_cuts) {
    float simultaneous = 0.0f;
    for (int position = row_ptr[local];
         position < row_ptr[local + 1]; ++position) {
      const int v = col_idx[position];
      if (v != int(u) && v >= int(params.first) &&
          v < int(params.first + params.rows) &&
          vertex_color(v, params.seed) == params.color &&
          proposal[v] == next) {
        simultaneous += edge_weight[position];
      }
    }
    old_cut = atomic_exchange_explicit(&cuts[old], 0.0f,
      memory_order_relaxed);
    atomic_fetch_add_explicit(&cuts[next],
      old_cut - 2.0f * proposal_weight[u] - simultaneous,
      memory_order_relaxed);
  }
  labels[u] = next;
  atomic_fetch_add_explicit(
    &community_volume[old], -degree[u], memory_order_relaxed
  );
  atomic_fetch_add_explicit(
    &community_volume[next], degree[u], memory_order_relaxed
  );
  atomic_fetch_sub_explicit(&community_count[old], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&community_count[next], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(move_count, 1U, memory_order_relaxed);
}
)METAL";
}

MTLCompileOptions* metal_graph_compile_options() {
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeFast;
  }
#else
  options.fastMathEnabled = YES;
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
  if (@available(macOS 14.0, *)) {
    options.languageVersion = MTLLanguageVersion3_1;
    return options;
  }
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
  if (@available(macOS 13.0, *)) {
    options.languageVersion = MTLLanguageVersion3_0;
    return options;
  }
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
  if (@available(macOS 12.0, *)) {
    options.languageVersion = MTLLanguageVersion2_4;
    return options;
  }
#endif
  options.languageVersion = MTLLanguageVersion2_3;
  return options;
}

void initialize_state() {
  std::lock_guard<std::mutex> lock(state_mutex());
  MetalGraphState& current = state();
  if (current.initialized) return;
  current.initialized = true;
  current.device = MTLCreateSystemDefaultDevice();
  if (current.device == nil) {
    current.error = "No Metal device is available.";
    return;
  }
  current.queue = [current.device newCommandQueue];
  if (current.queue == nil) {
    current.error = "Could not create a Metal command queue.";
    return;
  }
  NSError* error = nil;
  MTLCompileOptions* options = metal_graph_compile_options();
  current.library = [current.device
    newLibraryWithSource:[NSString stringWithUTF8String:metal_source()]
    options:options
    error:&error];
  [options release];
  if (current.library == nil) {
    current.error = error == nil ? "Metal source compilation failed." :
      std::string([[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> proposal =
    [current.library newFunctionWithName:@"graph_propose"];
  id<MTLFunction> apply =
    [current.library newFunctionWithName:@"graph_apply"];
  current.proposal_pipeline = [current.device
    newComputePipelineStateWithFunction:proposal error:&error];
  if (current.proposal_pipeline == nil) {
    current.error = error == nil ? "Could not create Metal proposal pipeline." :
      std::string([[error localizedDescription] UTF8String]);
    return;
  }
  current.apply_pipeline = [current.device
    newComputePipelineStateWithFunction:apply error:&error];
  if (current.apply_pipeline == nil) {
    current.error = error == nil ? "Could not create Metal apply pipeline." :
      std::string([[error localizedDescription] UTF8String]);
  }
}

template <typename T>
id<MTLBuffer> shared_buffer(id<MTLDevice> device,
                            const std::vector<T>& values) {
  if (values.empty()) {
    return [device newBufferWithLength:sizeof(T)
      options:MTLResourceStorageModeShared];
  }
  return [device
    newBufferWithBytes:values.data()
    length:values.size() * sizeof(T)
    options:MTLResourceStorageModeShared];
}

NSUInteger thread_count(id<MTLComputePipelineState> pipeline) {
  return std::max<NSUInteger>(
    1, std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)
  );
}

void encode_pipeline(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> pipeline,
                     NSUInteger n) {
  [encoder setComputePipelineState:pipeline];
  const NSUInteger threads = thread_count(pipeline);
  [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

struct MetalStreamSession {
  id<MTLBuffer> degree = nil;
  id<MTLBuffer> labels = nil;
  id<MTLBuffer> parent = nil;
  id<MTLBuffer> volume = nil;
  id<MTLBuffer> count = nil;
  id<MTLBuffer> cuts = nil;
  id<MTLBuffer> parent_volume = nil;
  id<MTLBuffer> proposal = nil;
  id<MTLBuffer> proposal_weight = nil;
  id<MTLBuffer> moves = nil;
  MetalParams params{};
  std::uint64_t seed = 0;

  ~MetalStreamSession() {
    [degree release];
    [labels release];
    [parent release];
    [volume release];
    [count release];
    [cuts release];
    [parent_volume release];
    [proposal release];
    [proposal_weight release];
    [moves release];
  }
};

MetalStreamSession& stream_session(void* handle) {
  if (handle == nullptr) Rcpp::stop("Metal graph session is missing.");
  return *static_cast<MetalStreamSession*>(handle);
}

void encode_graph_color(id<MTLCommandBuffer> command,
                        id<MTLBuffer> row, id<MTLBuffer> col,
                        id<MTLBuffer> weight,
                        MetalStreamSession& session) {
  const auto& current = state();
  id<MTLComputeCommandEncoder> proposal = [command computeCommandEncoder];
  [proposal setBuffer:row offset:0 atIndex:0];
  [proposal setBuffer:col offset:0 atIndex:1];
  [proposal setBuffer:weight offset:0 atIndex:2];
  [proposal setBuffer:session.degree offset:0 atIndex:3];
  [proposal setBuffer:session.labels offset:0 atIndex:4];
  [proposal setBuffer:session.parent == nil ? session.labels :
    session.parent offset:0 atIndex:5];
  [proposal setBuffer:session.volume offset:0 atIndex:6];
  [proposal setBuffer:session.count offset:0 atIndex:7];
  [proposal setBuffer:session.proposal offset:0 atIndex:8];
  [proposal setBytes:&session.params length:sizeof(session.params) atIndex:9];
  [proposal setBuffer:session.cuts == nil ? session.volume :
    session.cuts offset:0 atIndex:10];
  [proposal setBuffer:session.parent_volume == nil ? session.volume :
    session.parent_volume offset:0 atIndex:11];
  [proposal setBuffer:session.proposal_weight == nil ? session.volume :
    session.proposal_weight offset:0 atIndex:12];
  encode_pipeline(proposal, current.proposal_pipeline, session.params.rows);
  [proposal endEncoding];

  id<MTLComputeCommandEncoder> apply = [command computeCommandEncoder];
  [apply setBuffer:session.degree offset:0 atIndex:0];
  [apply setBuffer:session.labels offset:0 atIndex:1];
  [apply setBuffer:session.proposal offset:0 atIndex:2];
  [apply setBuffer:session.volume offset:0 atIndex:3];
  [apply setBuffer:session.count offset:0 atIndex:4];
  [apply setBuffer:session.moves offset:0 atIndex:5];
  [apply setBytes:&session.params length:sizeof(session.params) atIndex:6];
  [apply setBuffer:session.cuts == nil ? session.volume :
    session.cuts offset:0 atIndex:7];
  [apply setBuffer:session.proposal_weight == nil ? session.volume :
    session.proposal_weight offset:0 atIndex:8];
  [apply setBuffer:row offset:0 atIndex:9];
  [apply setBuffer:col offset:0 atIndex:10];
  [apply setBuffer:weight offset:0 atIndex:11];
  encode_pipeline(apply, current.apply_pipeline, session.params.rows);
  [apply endEncoding];
}

}  // namespace

bool graph_clustering_metal_available_impl() {
  @autoreleasepool {
    initialize_state();
    const MetalGraphState& current = state();
    return current.device != nil &&
      current.proposal_pipeline != nil &&
      current.apply_pipeline != nil;
  }
}

std::string graph_clustering_metal_error_impl() {
  @autoreleasepool {
    initialize_state();
    return state().error;
  }
}

fastembedr_graph_accel::MoveResult graph_local_move_metal_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    bool refinement) {
  using fastembedr_graph_accel::MoveResult;
  @autoreleasepool {
    initialize_state();
    MetalGraphState& current = state();
    if (!graph_clustering_metal_available_impl()) {
      Rcpp::stop(
        current.error.empty() ? "Metal graph clustering is unavailable." :
        current.error
      );
    }
    if (initial.size() != static_cast<std::size_t>(graph.n)) {
      Rcpp::stop("Internal Metal clustering membership size mismatch.");
    }
    if (refinement &&
        (parent == nullptr || parent->size() != initial.size())) {
      Rcpp::stop("Internal Metal Leiden parent-partition size mismatch.");
    }

    MoveResult result;
    result.membership = initial;
    if (graph.n < 2 || !(graph.volume > 0.0f)) return result;

    std::vector<float> volume(static_cast<std::size_t>(graph.n), 0.0f);
    std::vector<int> count(static_cast<std::size_t>(graph.n), 0);
    for (int u = 0; u < graph.n; ++u) {
      const int community = initial[static_cast<std::size_t>(u)];
      volume[static_cast<std::size_t>(community)] +=
        graph.degree[static_cast<std::size_t>(u)];
      ++count[static_cast<std::size_t>(community)];
    }
    std::vector<int> parent_storage;
    if (parent == nullptr) {
      parent_storage.assign(static_cast<std::size_t>(graph.n), 0);
      parent = &parent_storage;
    }
    std::vector<int> proposal(initial.size(), 0);
    std::vector<std::uint32_t> moves(1, 0U);

    id<MTLBuffer> row_buffer = shared_buffer(current.device, graph.row_ptr);
    id<MTLBuffer> col_buffer = shared_buffer(current.device, graph.col_idx);
    id<MTLBuffer> weight_buffer = shared_buffer(current.device, graph.weight);
    id<MTLBuffer> degree_buffer = shared_buffer(current.device, graph.degree);
    id<MTLBuffer> label_buffer = shared_buffer(current.device, initial);
    id<MTLBuffer> parent_buffer = shared_buffer(current.device, *parent);
    id<MTLBuffer> volume_buffer = shared_buffer(current.device, volume);
    id<MTLBuffer> count_buffer = shared_buffer(current.device, count);
    id<MTLBuffer> proposal_buffer = shared_buffer(current.device, proposal);
    id<MTLBuffer> move_buffer = shared_buffer(current.device, moves);
    if (row_buffer == nil || col_buffer == nil || weight_buffer == nil ||
        degree_buffer == nil || label_buffer == nil || parent_buffer == nil ||
        volume_buffer == nil || count_buffer == nil ||
        proposal_buffer == nil || move_buffer == nil) {
      Rcpp::stop("Metal graph clustering buffer allocation failed.");
    }

    const float tolerance = 1e-7f * std::max(
      1.0f,
      static_cast<float>(graph.total_edge_weight) /
        static_cast<float>(std::max(1, graph.n))
    );
    MetalParams params{
      static_cast<std::uint32_t>(graph.n),
      0U,
      static_cast<std::uint32_t>(seed ^ (seed >> 32U)),
      refinement ? 1U : 0U,
      0U,
      graph.volume,
      resolution,
      tolerance,
      0U,
      static_cast<std::uint32_t>(graph.n)
    };

    for (int pass = 0; pass < max_passes; ++pass) {
      params.seed = static_cast<std::uint32_t>(seed ^ (seed >> 32U)) ^
        (0x9e3779b9U * static_cast<std::uint32_t>(pass + 1));
      *static_cast<std::uint32_t*>(move_buffer.contents) = 0U;
      id<MTLCommandBuffer> command = [current.queue commandBuffer];
      if (command == nil) Rcpp::stop("Could not create Metal command buffer.");
      for (int color = 0; color < kColorCount; ++color) {
        params.color = static_cast<std::uint32_t>(color);
        id<MTLComputeCommandEncoder> proposal_encoder =
          [command computeCommandEncoder];
        [proposal_encoder setBuffer:row_buffer offset:0 atIndex:0];
        [proposal_encoder setBuffer:col_buffer offset:0 atIndex:1];
        [proposal_encoder setBuffer:weight_buffer offset:0 atIndex:2];
        [proposal_encoder setBuffer:degree_buffer offset:0 atIndex:3];
        [proposal_encoder setBuffer:label_buffer offset:0 atIndex:4];
        [proposal_encoder setBuffer:parent_buffer offset:0 atIndex:5];
        [proposal_encoder setBuffer:volume_buffer offset:0 atIndex:6];
        [proposal_encoder setBuffer:count_buffer offset:0 atIndex:7];
        [proposal_encoder setBuffer:proposal_buffer offset:0 atIndex:8];
        [proposal_encoder setBytes:&params length:sizeof(params) atIndex:9];
        [proposal_encoder setBuffer:volume_buffer offset:0 atIndex:10];
        [proposal_encoder setBuffer:volume_buffer offset:0 atIndex:11];
        [proposal_encoder setBuffer:volume_buffer offset:0 atIndex:12];
        encode_pipeline(
          proposal_encoder, current.proposal_pipeline,
          static_cast<NSUInteger>(graph.n)
        );
        [proposal_encoder endEncoding];

        id<MTLComputeCommandEncoder> apply_encoder =
          [command computeCommandEncoder];
        [apply_encoder setBuffer:degree_buffer offset:0 atIndex:0];
        [apply_encoder setBuffer:label_buffer offset:0 atIndex:1];
        [apply_encoder setBuffer:proposal_buffer offset:0 atIndex:2];
        [apply_encoder setBuffer:volume_buffer offset:0 atIndex:3];
        [apply_encoder setBuffer:count_buffer offset:0 atIndex:4];
        [apply_encoder setBuffer:move_buffer offset:0 atIndex:5];
        [apply_encoder setBytes:&params length:sizeof(params) atIndex:6];
        [apply_encoder setBuffer:volume_buffer offset:0 atIndex:7];
        [apply_encoder setBuffer:volume_buffer offset:0 atIndex:8];
        [apply_encoder setBuffer:row_buffer offset:0 atIndex:9];
        [apply_encoder setBuffer:col_buffer offset:0 atIndex:10];
        [apply_encoder setBuffer:weight_buffer offset:0 atIndex:11];
        encode_pipeline(
          apply_encoder, current.apply_pipeline,
          static_cast<NSUInteger>(graph.n)
        );
        [apply_encoder endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        NSString* description = command.error.localizedDescription;
        Rcpp::stop(
          description == nil ? "Metal graph clustering command failed." :
          std::string([description UTF8String])
        );
      }
      const std::uint32_t pass_moves =
        *static_cast<std::uint32_t*>(move_buffer.contents);
      result.moves += static_cast<int>(pass_moves);
      ++result.passes;
      if (pass_moves == 0U) break;
    }

    const int* labels = static_cast<const int*>(label_buffer.contents);
    std::copy(labels, labels + graph.n, result.membership.begin());
    return result;
  }
}

void* graph_stream_metal_create(const std::vector<float>& degree,
                                const std::vector<int>& initial,
                                float graph_volume, float resolution,
                                std::uint64_t seed,
                                const std::vector<int>* parent,
                                const std::vector<float>* cuts,
                                const std::vector<float>* parent_volume) {
  @autoreleasepool {
    initialize_state();
    if (!graph_clustering_metal_available_impl()) {
      Rcpp::stop(graph_clustering_metal_error_impl());
    }
    if (degree.size() != initial.size() || degree.size() < 2 ||
        degree.size() > static_cast<std::size_t>(INT_MAX) ||
        !(graph_volume > 0.0f) || !(resolution > 0.0f)) {
      Rcpp::stop("Invalid Metal streaming graph state.");
    }
    const bool refinement = parent != nullptr;
    if (refinement && (cuts == nullptr || parent_volume == nullptr ||
        parent->size() != degree.size() || cuts->size() != degree.size() ||
        parent_volume->empty())) {
      Rcpp::stop("Invalid Metal streaming Leiden state.");
    }
    const auto n = degree.size();
    const auto bytes = n * sizeof(int);
    if (bytes > state().device.maxBufferLength) {
      Rcpp::stop("Metal graph state exceeds the device buffer limit.");
    }
    const auto working_set = state().device.recommendedMaxWorkingSetSize;
    const double buffers = refinement ? 9.0 : 5.0;
    if (working_set > 0 && buffers * bytes + 4.0 >
        0.7 * working_set) {
      Rcpp::stop("Metal graph state exceeds its working-set budget.");
    }
    std::vector<float> volume(n, 0.0f);
    std::vector<int> count(n, 0);
    for (std::size_t u = 0; u < n; ++u) {
      const int label = initial[u];
      if (label < 0 || static_cast<std::size_t>(label) >= n) {
        Rcpp::stop("Invalid initial Metal community label.");
      }
      volume[static_cast<std::size_t>(label)] += degree[u];
      ++count[static_cast<std::size_t>(label)];
    }
    std::unique_ptr<MetalStreamSession> session(new MetalStreamSession);
    auto device = state().device;
    session->degree = shared_buffer(device, degree);
    session->labels = shared_buffer(device, initial);
    if (refinement) session->parent = shared_buffer(device, *parent);
    session->volume = shared_buffer(device, volume);
    session->count = shared_buffer(device, count);
    if (refinement) {
      session->cuts = shared_buffer(device, *cuts);
      session->parent_volume = shared_buffer(device, *parent_volume);
      session->proposal_weight = [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    }
    session->proposal = [device newBufferWithLength:bytes
      options:MTLResourceStorageModeShared];
    session->moves = [device newBufferWithLength:sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    if (session->degree == nil || session->labels == nil ||
        session->volume == nil || session->count == nil ||
        session->proposal == nil || session->moves == nil ||
        (refinement && (session->parent == nil || session->cuts == nil ||
          session->parent_volume == nil ||
          session->proposal_weight == nil))) {
      Rcpp::stop("Metal streaming graph state allocation failed.");
    }
    session->params = {
      static_cast<std::uint32_t>(n), 0U, 0U,
      refinement ? 1U : 0U, refinement ? 1U : 0U,
      graph_volume, resolution,
      1e-7f * std::max(1.0f, graph_volume / (2.0f * n)),
      0U, 0U
    };
    session->seed = seed;
    return session.release();
  }
}

void graph_stream_metal_begin_pass(void* handle, int pass) {
  auto& session = stream_session(handle);
  session.params.seed = static_cast<std::uint32_t>(
    session.seed ^ (session.seed >> 32U)
  ) ^ (0x9e3779b9U * static_cast<std::uint32_t>(pass + 1));
  *static_cast<std::uint32_t*>(session.moves.contents) = 0U;
}

void graph_stream_metal_set_color(void* handle, int color) {
  auto& session = stream_session(handle);
  if (!session.params.leiden_cuts || color < 0 || color >= kColorCount) {
    Rcpp::stop("Invalid Metal Leiden color phase.");
  }
  session.params.color = static_cast<std::uint32_t>(color);
}

void graph_stream_metal_chunk(void* handle,
                              const std::vector<int>& row_ptr,
                              const std::vector<int>& col_idx,
                              const std::vector<float>& weight,
                              std::uint32_t first) {
  @autoreleasepool {
    auto& session = stream_session(handle);
    if (row_ptr.size() < 2 || col_idx.size() != weight.size() ||
        col_idx.size() > static_cast<std::size_t>(INT_MAX) ||
        row_ptr.back() != static_cast<int>(col_idx.size()) ||
        first + row_ptr.size() - 1 > session.params.n) {
      Rcpp::stop("Invalid Metal streaming graph chunk.");
    }
    auto device = state().device;
    id<MTLBuffer> row = shared_buffer(device, row_ptr);
    id<MTLBuffer> col = shared_buffer(device, col_idx);
    id<MTLBuffer> weights = shared_buffer(device, weight);
    if (row == nil || col == nil || weights == nil) {
      [row release]; [col release]; [weights release];
      Rcpp::stop("Metal graph chunk allocation failed.");
    }
    session.params.first = first;
    session.params.rows = static_cast<std::uint32_t>(row_ptr.size() - 1);
    id<MTLCommandBuffer> command = [state().queue commandBuffer];
    if (command == nil) Rcpp::stop("Metal graph command allocation failed.");
    if (session.params.leiden_cuts) {
      encode_graph_color(command, row, col, weights, session);
    } else {
      for (int color = 0; color < kColorCount; ++color) {
        session.params.color = static_cast<std::uint32_t>(color);
        encode_graph_color(command, row, col, weights, session);
      }
    }
    [command commit];
    [command waitUntilCompleted];
    [row release]; [col release]; [weights release];
    if (command.status == MTLCommandBufferStatusError) {
      Rcpp::stop("Metal streaming graph command failed: %s",
        [[command.error localizedDescription] UTF8String]);
    }
  }
}

std::uint32_t graph_stream_metal_end_pass(void* handle) {
  auto& session = stream_session(handle);
  return *static_cast<std::uint32_t*>(session.moves.contents);
}

std::vector<int> graph_stream_metal_labels(void* handle) {
  auto& session = stream_session(handle);
  const auto n = session.params.n;
  const int* values = static_cast<const int*>(session.labels.contents);
  return std::vector<int>(values, values + n);
}

std::vector<int> graph_stream_metal_counts(void* handle) {
  auto& session = stream_session(handle);
  const auto n = session.params.n;
  const int* values = static_cast<const int*>(session.count.contents);
  return std::vector<int>(values, values + n);
}

void graph_stream_metal_destroy(void* handle) {
  delete static_cast<MetalStreamSession*>(handle);
}
