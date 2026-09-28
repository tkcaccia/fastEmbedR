/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Metal kernels for cubic-grid three-dimensional t-SNE repulsion.
 */

const char* metal_tsne_3d_kernel_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Grid3d {
  uint n;
  uint grid;
  uint side;
  uint bins;
  float lower_x;
  float lower_y;
  float lower_z;
  float spacing;
  float inv_spacing;
  float cutoff;
  float cutoff2;
  float qcut;
};

struct Bounds3d {
  float min_x, max_x, min_y, max_y, min_z, max_z;
  float sum_x, sum_y, sum_z;
};

float smooth_q_3d(float d2, constant Grid3d& g) {
  if (d2 >= g.cutoff2) return 1.0f / (1.0f + d2);
  float delta = d2 - g.cutoff2;
  return g.qcut * (1.0f - g.qcut * delta +
                   g.qcut * g.qcut * delta * delta);
}

void cubic_3d(float t, thread float* w, thread float* d) {
  float t2 = t * t;
  w[0] = -t * (t - 1.0f) * (t - 2.0f) / 6.0f;
  w[1] = (t + 1.0f) * (t - 1.0f) * (t - 2.0f) / 2.0f;
  w[2] = -(t + 1.0f) * t * (t - 2.0f) / 2.0f;
  w[3] = (t + 1.0f) * t * (t - 1.0f) / 6.0f;
  d[0] = (-3.0f * t2 + 6.0f * t - 2.0f) / 6.0f;
  d[1] = (3.0f * t2 - 4.0f * t - 1.0f) / 2.0f;
  d[2] = (-3.0f * t2 + 2.0f * t + 2.0f) / 2.0f;
  d[3] = (3.0f * t2 - 1.0f) / 6.0f;
}

void point_weights_3d(device const float* y, uint row,
                      constant Grid3d& g, thread int* start,
                      thread float w[3][4],
                      thread float d[3][4]) {
  float lower[3] = {g.lower_x, g.lower_y, g.lower_z};
  for (uint axis = 0; axis < 3; ++axis) {
    float coord = (y[row * 3u + axis] - lower[axis]) *
      g.inv_spacing;
    int cell = clamp(int(floor(coord)), 1, int(g.grid) - 3);
    start[axis] = cell - 1;
    cubic_3d(coord - float(cell), w[axis], d[axis]);
  }
}

kernel void bounds_blocks_3d(
  device const float* y [[buffer(0)]],
  device Bounds3d* out [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  constant uint& block_size [[buffer(3)]],
  uint block [[thread_position_in_grid]]
) {
  uint begin = block * block_size;
  if (begin >= n) return;
  Bounds3d b = {INFINITY, -INFINITY, INFINITY, -INFINITY,
                INFINITY, -INFINITY, 0.0f, 0.0f, 0.0f};
  for (uint i = begin; i < min(n, begin + block_size); ++i) {
    float x = y[i * 3u];
    float yy = y[i * 3u + 1u];
    float z = y[i * 3u + 2u];
    b.min_x = min(b.min_x, x);
    b.max_x = max(b.max_x, x);
    b.min_y = min(b.min_y, yy);
    b.max_y = max(b.max_y, yy);
    b.min_z = min(b.min_z, z);
    b.max_z = max(b.max_z, z);
    b.sum_x += x;
    b.sum_y += yy;
    b.sum_z += z;
  }
  out[block] = b;
}

kernel void finalize_bounds_3d(
  device const Bounds3d* blocks [[buffer(0)]],
  device Grid3d* grid_out [[buffer(1)]],
  device float* center_out [[buffer(2)]],
  constant uint& count [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& grid [[buffer(5)]],
  uint id [[thread_position_in_grid]]
) {
  if (id != 0u) return;
  Bounds3d b = {INFINITY, -INFINITY, INFINITY, -INFINITY,
                INFINITY, -INFINITY, 0.0f, 0.0f, 0.0f};
  for (uint i = 0u; i < count; ++i) {
    Bounds3d next = blocks[i];
    b.min_x = min(b.min_x, next.min_x);
    b.max_x = max(b.max_x, next.max_x);
    b.min_y = min(b.min_y, next.min_y);
    b.max_y = max(b.max_y, next.max_y);
    b.min_z = min(b.min_z, next.min_z);
    b.max_z = max(b.max_z, next.max_z);
    b.sum_x += next.sum_x;
    b.sum_y += next.sum_y;
    b.sum_z += next.sum_z;
  }
  float span = max(max(b.max_x - b.min_x,
                       b.max_y - b.min_y), b.max_z - b.min_z);
  float half_span = 0.55f * span + 1.0e-3f;
  float spacing = 2.0f * half_span / float(grid - 5u);
  float cutoff = (n < 5000u ? 3.0f : 2.0f) * spacing;
  Grid3d g;
  g.n = n;
  g.grid = grid;
  g.side = 2u * grid;
  g.bins = uint(ceil(float(grid) * spacing / cutoff));
  g.lower_x = 0.5f * (b.min_x + b.max_x) - half_span -
    2.0f * spacing;
  g.lower_y = 0.5f * (b.min_y + b.max_y) - half_span -
    2.0f * spacing;
  g.lower_z = 0.5f * (b.min_z + b.max_z) - half_span -
    2.0f * spacing;
  g.spacing = spacing;
  g.inv_spacing = 1.0f / spacing;
  g.cutoff = cutoff;
  g.cutoff2 = cutoff * cutoff;
  g.qcut = 1.0f / (1.0f + g.cutoff2);
  grid_out[0] = g;
  center_out[0] = b.sum_x / float(n);
  center_out[1] = b.sum_y / float(n);
  center_out[2] = b.sum_z / float(n);
}

kernel void clear_fft_3d(
  device atomic_float* mass [[buffer(0)]],
  device float* kernel_values [[buffer(1)]],
  device atomic_int* bin_head [[buffer(2)]],
  constant uint& side [[buffer(3)]],
  constant uint& bins [[buffer(4)]],
  uint pos [[thread_position_in_grid]]
) {
  uint total = side * side * side;
  if (pos < total) {
    atomic_store_explicit(&mass[pos], 0.0f,
                          memory_order_relaxed);
    kernel_values[pos] = 0.0f;
  }
  if (pos < bins * bins * bins) {
    atomic_store_explicit(&bin_head[pos], -1,
                          memory_order_relaxed);
  }
}

kernel void kernel_fft_3d(
  device float* kernel_values [[buffer(0)]],
  constant Grid3d& g [[buffer(1)]],
  uint pos [[thread_position_in_grid]]
) {
  uint total = g.side * g.side * g.side;
  if (pos >= total) return;
  uint x = pos % g.side;
  uint yy = (pos / g.side) % g.side;
  uint z = pos / (g.side * g.side);
  int dx = x < g.grid ? int(x) : int(x) - int(g.side);
  int dy = yy < g.grid ? int(yy) : int(yy) - int(g.side);
  int dz = z < g.grid ? int(z) : int(z) - int(g.side);
  float d2 = float(dx * dx + dy * dy + dz * dz) *
    g.spacing * g.spacing;
  kernel_values[pos] = smooth_q_3d(d2, g);
}

kernel void scatter_fft_3d(
  device const float* y [[buffer(0)]],
  device atomic_float* mass [[buffer(1)]],
  device atomic_int* bin_head [[buffer(2)]],
  device int* bin_next [[buffer(3)]],
  constant Grid3d& g [[buffer(4)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  int start[3];
  float w[3][4], d[3][4];
  point_weights_3d(y, row, g, start, w, d);
  for (int z = 0; z < 4; ++z)
    for (int yy = 0; yy < 4; ++yy)
      for (int x = 0; x < 4; ++x) {
        uint pos = (uint(start[2] + z) * g.side +
                    uint(start[1] + yy)) * g.side +
                   uint(start[0] + x);
        atomic_fetch_add_explicit(
          &mass[pos], w[0][x] * w[1][yy] * w[2][z],
          memory_order_relaxed
        );
      }
  if (g.cutoff2 <= 0.01f) return;
  int bx = clamp(int((y[row * 3u] - g.lower_x) / g.cutoff),
                 0, int(g.bins) - 1);
  int by = clamp(int((y[row * 3u + 1u] - g.lower_y) /
                     g.cutoff), 0, int(g.bins) - 1);
  int bz = clamp(int((y[row * 3u + 2u] - g.lower_z) /
                     g.cutoff), 0, int(g.bins) - 1);
  uint bin = (uint(bz) * g.bins + uint(by)) * g.bins +
    uint(bx);
  bin_next[row] = atomic_exchange_explicit(
    &bin_head[bin], int(row), memory_order_relaxed
  );
}

kernel void gather_fft_3d(
  device const float* y [[buffer(0)]],
  device const float* potential [[buffer(1)]],
  device const atomic_int* bin_head [[buffer(2)]],
  device const int* bin_next [[buffer(3)]],
  device float* repulsive [[buffer(4)]],
  device float* row_q [[buffer(5)]],
  constant Grid3d& g [[buffer(6)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  int start[3];
  float w[3][4], d[3][4];
  point_weights_3d(y, row, g, start, w, d);
  float value = 0.0f;
  float slope[3] = {};
  for (int z = 0; z < 4; ++z)
    for (int yy = 0; yy < 4; ++yy)
      for (int x = 0; x < 4; ++x) {
        uint pos = (uint(start[2] + z) * g.side +
                    uint(start[1] + yy)) * g.side +
                   uint(start[0] + x);
        float p = potential[pos];
        value += p * w[0][x] * w[1][yy] * w[2][z];
        slope[0] += p * d[0][x] * w[1][yy] * w[2][z];
        slope[1] += p * w[0][x] * d[1][yy] * w[2][z];
        slope[2] += p * w[0][x] * w[1][yy] * d[2][z];
      }
  float auto_w[3][7] = {};
  float cross_w[3][7] = {};
  for (int axis = 0; axis < 3; ++axis)
    for (int gather = 0; gather < 4; ++gather)
      for (int scatter = 0; scatter < 4; ++scatter) {
        int delta = gather - scatter + 3;
        auto_w[axis][delta] += w[axis][gather] *
          w[axis][scatter];
        cross_w[axis][delta] += d[axis][gather] *
          w[axis][scatter];
      }
  float self_value = 0.0f;
  float self_slope[3] = {};
  for (int z = 0; z < 7; ++z)
    for (int yy = 0; yy < 7; ++yy)
      for (int x = 0; x < 7; ++x) {
        float3 shift = float3(x - 3, yy - 3, z - 3) *
          g.spacing;
        float q = smooth_q_3d(dot(shift, shift), g);
        float yz = auto_w[1][yy] * auto_w[2][z];
        self_value += q * auto_w[0][x] * yz;
        self_slope[0] += q * cross_w[0][x] * yz;
        self_slope[1] += q * auto_w[0][x] *
          cross_w[1][yy] * auto_w[2][z];
        self_slope[2] += q * auto_w[0][x] *
          auto_w[1][yy] * cross_w[2][z];
      }
  value -= self_value;
  float near_slope[3] = {};
  if (g.cutoff2 > 0.01f) {
    float3 yi = float3(y[row * 3u], y[row * 3u + 1u],
                       y[row * 3u + 2u]);
    int3 cell = clamp(int3((yi - float3(g.lower_x,
                                      g.lower_y,
                                      g.lower_z)) / g.cutoff),
                      int3(0), int3(int(g.bins) - 1));
    for (int z = max(0, cell.z - 1);
         z <= min(int(g.bins) - 1, cell.z + 1); ++z)
      for (int yy = max(0, cell.y - 1);
           yy <= min(int(g.bins) - 1, cell.y + 1); ++yy)
        for (int x = max(0, cell.x - 1);
             x <= min(int(g.bins) - 1, cell.x + 1); ++x) {
          uint bin = (uint(z) * g.bins + uint(yy)) *
            g.bins + uint(x);
          for (int j = atomic_load_explicit(
                 &bin_head[bin], memory_order_relaxed);
               j >= 0; j = bin_next[uint(j)]) {
            if (uint(j) == row) continue;
            float3 diff = yi - float3(
              y[uint(j) * 3u], y[uint(j) * 3u + 1u],
              y[uint(j) * 3u + 2u]
            );
            float d2 = dot(diff, diff);
            if (d2 >= g.cutoff2) continue;
            float q = 1.0f / (1.0f + d2);
            float delta = d2 - g.cutoff2;
            float smooth_derivative = -g.qcut * g.qcut +
              2.0f * g.qcut * g.qcut * g.qcut * delta;
            float correction = 2.0f *
              (-q * q - smooth_derivative);
            value += q - smooth_q_3d(d2, g);
            near_slope[0] += correction * diff.x;
            near_slope[1] += correction * diff.y;
            near_slope[2] += correction * diff.z;
          }
        }
  }
  row_q[row] = value;
  for (uint axis = 0u; axis < 3u; ++axis) {
    repulsive[row * 3u + axis] = 0.5f *
      ((slope[axis] - self_slope[axis]) * g.inv_spacing +
       near_slope[axis]);
  }
}

kernel void sum_q_3d(
  device const float* row_q [[buffer(0)]],
  device float* inv_sum_q [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  uint id [[thread_position_in_grid]]
) {
  if (id != 0u) return;
  float total = 0.0f;
  for (uint i = 0u; i < n; ++i) total += row_q[i];
  inv_sum_q[0] = isfinite(total) && total > 0.0f ?
    1.0f / total : NAN;
}

struct Update3d {
  uint n;
  float exaggeration;
  float learning_rate;
  float momentum;
  float min_gain;
  float max_step_norm;
};

kernel void update_tsne_3d(
  device const int* row_ptr [[buffer(0)]],
  device const int* col [[buffer(1)]],
  device const float* p [[buffer(2)]],
  device const float* y [[buffer(3)]],
  device const float* repulsive [[buffer(4)]],
  device const float* inv_sum_q [[buffer(5)]],
  device float* next [[buffer(6)]],
  device float* gains [[buffer(7)]],
  device float* updates [[buffer(8)]],
  constant Update3d& params [[buffer(9)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= params.n) return;
  float grad[3];
  for (uint axis = 0u; axis < 3u; ++axis) {
    grad[axis] = repulsive[row * 3u + axis] *
      inv_sum_q[0];
  }
  for (int edge = row_ptr[row]; edge < row_ptr[row + 1u];
       ++edge) {
    uint neighbor = uint(col[edge]);
    float3 diff = float3(
      y[row * 3u] - y[neighbor * 3u],
      y[row * 3u + 1u] - y[neighbor * 3u + 1u],
      y[row * 3u + 2u] - y[neighbor * 3u + 2u]
    );
    float factor = params.exaggeration * p[edge] /
      (1.0f + dot(diff, diff));
    grad[0] += factor * diff.x;
    grad[1] += factor * diff.y;
    grad[2] += factor * diff.z;
  }
  float step[3];
  float step2 = 0.0f;
  for (uint axis = 0u; axis < 3u; ++axis) {
    uint pos = row * 3u + axis;
    float old = updates[pos];
    float gain = gains[pos];
    gain = sign(old) != sign(grad[axis]) ?
      gain + 0.2f : gain * 0.8f + params.min_gain;
    gain = max(gain, params.min_gain);
    step[axis] = params.momentum * old -
      params.learning_rate * gain * grad[axis];
    gains[pos] = gain;
    step2 += step[axis] * step[axis];
  }
  float scale = step2 > params.max_step_norm *
    params.max_step_norm ?
    params.max_step_norm / (sqrt(step2) + 1.0e-12f) : 1.0f;
  for (uint axis = 0u; axis < 3u; ++axis) {
    uint pos = row * 3u + axis;
    updates[pos] = step[axis] * scale;
    next[pos] = y[pos] + updates[pos];
  }
}

kernel void center_tsne_3d(
  device float* y [[buffer(0)]],
  device const float* center [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  for (uint axis = 0u; axis < 3u; ++axis) {
    y[row * 3u + axis] -= center[axis];
  }
}
)METAL";
}

struct MetalTsne3dPipelines {
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> bounds = nil;
  id<MTLComputePipelineState> finalize = nil;
  id<MTLComputePipelineState> clear = nil;
  id<MTLComputePipelineState> kernel = nil;
  id<MTLComputePipelineState> scatter = nil;
  id<MTLComputePipelineState> gather = nil;
  id<MTLComputePipelineState> sum = nil;
  id<MTLComputePipelineState> update = nil;
  id<MTLComputePipelineState> center = nil;

  ~MetalTsne3dPipelines() {
    [center release];
    [update release];
    [sum release];
    [gather release];
    [scatter release];
    [kernel release];
    [clear release];
    [finalize release];
    [bounds release];
    [library release];
  }
};

MetalTsne3dPipelines& metal_tsne_3d_pipelines(
    MetalEmbeddingState& state) {
  static MetalTsne3dPipelines kernels;
  if (kernels.center != nil) return kernels;
  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:
    metal_tsne_3d_kernel_source()];
  MTLCompileOptions* options = metal_embedding_compile_options();
  kernels.library = [state.device newLibraryWithSource:source
    options:options error:&error];
  [options release];
  if (kernels.library == nil) {
    Rcpp::stop("Metal 3D t-SNE kernels failed to compile: %s",
               ns_error_message(error).c_str());
  }
  auto load = [&](const char* name) {
    id<MTLFunction> function = [kernels.library
      newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (function == nil) {
      Rcpp::stop("Metal 3D t-SNE kernel `%s` is missing.", name);
    }
    id<MTLComputePipelineState> pipeline =
      [state.device newComputePipelineStateWithFunction:function
                                                error:&error];
    [function release];
    if (pipeline == nil) {
      Rcpp::stop("Metal 3D t-SNE pipeline `%s` failed: %s",
                 name, ns_error_message(error).c_str());
    }
    return pipeline;
  };
  kernels.bounds = load("bounds_blocks_3d");
  kernels.finalize = load("finalize_bounds_3d");
  kernels.clear = load("clear_fft_3d");
  kernels.kernel = load("kernel_fft_3d");
  kernels.scatter = load("scatter_fft_3d");
  kernels.gather = load("gather_fft_3d");
  kernels.sum = load("sum_q_3d");
  kernels.update = load("update_tsne_3d");
  kernels.center = load("center_tsne_3d");
  return kernels;
}

struct MetalTsne3dBuffers {
  id<MTLDevice> device;
  std::vector<id<MTLBuffer>> owned;

  explicit MetalTsne3dBuffers(id<MTLDevice> device) : device(device) {}
  ~MetalTsne3dBuffers() {
    for (id<MTLBuffer> buffer : owned) [buffer release];
  }
  id<MTLBuffer> make(std::size_t bytes, const void* data = nullptr) {
    id<MTLBuffer> buffer = data == nullptr ?
      [device newBufferWithLength:bytes
                         options:MTLResourceStorageModeShared] :
      [device newBufferWithBytes:data length:bytes
                        options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      Rcpp::stop("Metal 3D t-SNE buffer allocation failed.");
    }
    owned.push_back(buffer);
    return buffer;
  }
};

void metal_tsne_3d_encode(
    id<MTLCommandBuffer> command,
    id<MTLComputePipelineState> pipeline,
    NSUInteger items,
    std::initializer_list<id<MTLBuffer>> buffers,
    std::initializer_list<std::pair<const void*, std::size_t>> values = {}) {
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  NSUInteger slot = 0;
  for (id<MTLBuffer> buffer : buffers) {
    [encoder setBuffer:buffer offset:0 atIndex:slot++];
  }
  for (const auto& value : values) {
    [encoder setBytes:value.first length:value.second atIndex:slot++];
  }
  [encoder dispatchThreads:MTLSizeMake(items, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(bounded_threads(pipeline), 1, 1)];
  [encoder endEncoding];
}

void metal_tsne_3d_wait(id<MTLCommandBuffer> command,
                        const char* stage) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    Rcpp::stop("Metal 3D t-SNE %s failed: %s", stage,
               ns_error_message(command.error).c_str());
  }
}

struct MetalTsne3dUpdate {
  std::uint32_t n;
  float exaggeration;
  float learning_rate;
  float momentum;
  float min_gain;
  float max_step_norm;
};

List run_metal_tsne_3d(
    MetalEmbeddingState& state,
    const TsneSparseMetalGraph& graph,
    const std::vector<float>& initial,
    int n,
    int early_iter,
    int normal_iter,
    double early_exaggeration,
    double exaggeration,
    double learning_rate,
    bool learning_rate_auto,
    double initial_momentum,
    double final_momentum,
    double min_gain,
    double max_step_norm,
    bool auto_config) {
  MetalTsne3dPipelines& pipe = metal_tsne_3d_pipelines(state);
  MetalTsne3dBuffers memory(state.device);
  const std::uint32_t rows = static_cast<std::uint32_t>(n);
  const std::uint32_t block_size = 256u;
  const std::uint32_t blocks = (rows + block_size - 1u) /
    block_size;
  const std::uint32_t grid = n < 5000 ? 16u : 64u;
  const std::uint32_t side = 2u * grid;
  const std::uint32_t bins = n < 5000 ?
    (grid + 2u) / 3u : (grid + 1u) / 2u;
  const std::size_t cube = static_cast<std::size_t>(side) *
    side * side;
  const std::size_t layout_bytes = initial.size() *
    sizeof(float);
  id<MTLBuffer> row_ptr = memory.make(
    graph.row_ptr.size() * sizeof(std::int32_t),
    graph.row_ptr.data()
  );
  id<MTLBuffer> col = memory.make(
    graph.col.size() * sizeof(std::int32_t), graph.col.data()
  );
  id<MTLBuffer> prob = memory.make(
    graph.val.size() * sizeof(float), graph.val.data()
  );
  id<MTLBuffer> current = memory.make(layout_bytes,
                                       initial.data());
  id<MTLBuffer> next = memory.make(layout_bytes);
  id<MTLBuffer> gradient = memory.make(layout_bytes);
  id<MTLBuffer> gains = memory.make(layout_bytes);
  id<MTLBuffer> updates = memory.make(layout_bytes);
  std::vector<float> ones(initial.size(), 1.0f);
  std::memcpy([gains contents], ones.data(), layout_bytes);
  std::memset([updates contents], 0, layout_bytes);
  id<MTLBuffer> mass = memory.make(cube * sizeof(float));
  id<MTLBuffer> kernel = memory.make(cube * sizeof(float));
  id<MTLBuffer> potential = memory.make(cube * sizeof(float));
  id<MTLBuffer> head = memory.make(
    static_cast<std::size_t>(bins) * bins * bins * sizeof(int)
  );
  id<MTLBuffer> links = memory.make(
    static_cast<std::size_t>(n) * sizeof(int)
  );
  id<MTLBuffer> row_q = memory.make(
    static_cast<std::size_t>(n) * sizeof(float)
  );
  id<MTLBuffer> inv_sum_q = memory.make(sizeof(float));
  id<MTLBuffer> block_bounds = memory.make(
    static_cast<std::size_t>(blocks) * 9u * sizeof(float)
  );
  id<MTLBuffer> params = memory.make(48u);
  id<MTLBuffer> center = memory.make(3u * sizeof(float));

  MPSShape* shape = @[@(side), @(side), @(side)];
  MPSGraph* fft_graph = [[MPSGraph alloc] init];
  MPSGraphTensor* mass_in = [fft_graph
    placeholderWithShape:shape dataType:MPSDataTypeFloat32
                    name:@"mass_3d"];
  MPSGraphTensor* kernel_in = [fft_graph
    placeholderWithShape:shape dataType:MPSDataTypeFloat32
                    name:@"kernel_3d"];
  MPSGraphFFTDescriptor* forward =
    [MPSGraphFFTDescriptor descriptor];
  MPSGraphFFTDescriptor* inverse =
    [MPSGraphFFTDescriptor descriptor];
  inverse.inverse = YES;
  inverse.scalingMode = MPSGraphFFTScalingModeSize;
  NSArray* axes = @[@0, @1, @2];
  MPSGraphTensor* mass_fft = [fft_graph
    realToHermiteanFFTWithTensor:mass_in axes:axes
                     descriptor:forward name:@"mass_fft_3d"];
  MPSGraphTensor* kernel_fft = [fft_graph
    realToHermiteanFFTWithTensor:kernel_in axes:axes
                     descriptor:forward name:@"kernel_fft_3d"];
  MPSGraphTensor* product = [fft_graph
    multiplicationWithPrimaryTensor:mass_fft
                    secondaryTensor:kernel_fft name:@"product_3d"];
  MPSGraphTensor* result = [fft_graph
    HermiteanToRealFFTWithTensor:product axes:axes
                     descriptor:inverse name:@"potential_3d"];
  MPSGraphTensorData* mass_data = [[MPSGraphTensorData alloc]
    initWithMTLBuffer:mass shape:shape
             dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* kernel_data = [[MPSGraphTensorData alloc]
    initWithMTLBuffer:kernel shape:shape
             dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* potential_data = [[MPSGraphTensorData alloc]
    initWithMTLBuffer:potential shape:shape
             dataType:MPSDataTypeFloat32];
  NSDictionary* feeds = @{mass_in: mass_data,
                          kernel_in: kernel_data};
  NSDictionary* results = @{result: potential_data};
  const auto scalar = [](const auto& value) {
    return std::make_pair(
      static_cast<const void*>(&value), sizeof(value)
    );
  };
  for (int iteration = 0;
       iteration < early_iter + normal_iter; ++iteration) {
    id<MTLCommandBuffer> prepare = [state.queue commandBuffer];
    metal_tsne_3d_encode(prepare, pipe.bounds, blocks,
                         {current, block_bounds},
                         {scalar(rows), scalar(block_size)});
    metal_tsne_3d_encode(prepare, pipe.finalize, 1u,
                         {block_bounds, params, center},
                         {scalar(blocks), scalar(rows),
                          scalar(grid)});
    metal_tsne_3d_encode(prepare, pipe.clear, cube,
                         {mass, kernel, head},
                         {scalar(side), scalar(bins)});
    metal_tsne_3d_encode(prepare, pipe.kernel, cube,
                         {kernel, params});
    metal_tsne_3d_encode(prepare, pipe.scatter, rows,
                         {current, mass, head, links, params});
    metal_tsne_3d_wait(prepare, "FFT preparation");
    [fft_graph runWithMTLCommandQueue:state.queue feeds:feeds
                      targetOperations:nil
                    resultsDictionary:results];
    id<MTLCommandBuffer> optimize = [state.queue commandBuffer];
    metal_tsne_3d_encode(optimize, pipe.gather, rows,
                         {current, potential, head, links,
                          gradient, row_q, params});
    metal_tsne_3d_encode(optimize, pipe.sum, 1u,
                         {row_q, inv_sum_q}, {scalar(rows)});
    const bool early = iteration < early_iter;
    const double phase_exaggeration = early ?
      early_exaggeration : exaggeration;
    const float rate = learning_rate_auto ?
      static_cast<float>(n / phase_exaggeration) :
      static_cast<float>(learning_rate);
    const MetalTsne3dUpdate update_params = {
      rows, static_cast<float>(phase_exaggeration), rate,
      static_cast<float>(early ? initial_momentum :
                         final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm > 0.0 ?
                         max_step_norm : FLT_MAX)
    };
    metal_tsne_3d_encode(optimize, pipe.update, rows,
                         {row_ptr, col, prob, current, gradient,
                          inv_sum_q, next, gains, updates},
                         {scalar(update_params)});
    metal_tsne_3d_encode(optimize, pipe.bounds, blocks,
                         {next, block_bounds},
                         {scalar(rows), scalar(block_size)});
    metal_tsne_3d_encode(optimize, pipe.finalize, 1u,
                         {block_bounds, params, center},
                         {scalar(blocks), scalar(rows),
                          scalar(grid)});
    metal_tsne_3d_encode(optimize, pipe.center, rows,
                         {next, center}, {scalar(rows)});
    metal_tsne_3d_wait(optimize, "optimizer update");
    std::swap(current, next);
  }
  const float* final = static_cast<const float*>(
    [current contents]
  );
  NumericMatrix layout(n, 3);
  for (int i = 0; i < n; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float value = final[
        static_cast<std::size_t>(i) * 3u + axis
      ];
      if (!std::isfinite(value)) {
        Rcpp::stop("Metal 3D t-SNE produced nonfinite coordinates.");
      }
      layout(i, axis) = value;
    }
  }
  [mass_data release];
  [kernel_data release];
  [potential_data release];
  [fft_graph release];
  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("costs") = NumericVector(0),
    Rcpp::Named("itercosts") = NumericVector(0),
    Rcpp::Named("optimizer") = "tsne_fft_grid_3d_native_metal",
    Rcpp::Named("repulsion") = "fft_grid_3d_metal",
    Rcpp::Named("fft_grid_size") = static_cast<int>(grid),
    Rcpp::Named("probabilities") =
      "symmetric_sparse_knn_cpu_prepared_for_metal",
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("n_threads") = NA_INTEGER,
    Rcpp::Named("learning_rate") = learning_rate_auto ?
      NA_REAL : learning_rate,
    Rcpp::Named("auto_config") = auto_config,
    Rcpp::Named("auto_kld_stop") = false,
    Rcpp::Named("auto_stop_reason") =
      "three_dimensional_fixed_iteration_schedule",
    Rcpp::Named("early_exaggeration_iter_actual") = early_iter,
    Rcpp::Named("n_iter_actual") = normal_iter,
    Rcpp::Named("max_iter_actual") = early_iter + normal_iter
  );
}
