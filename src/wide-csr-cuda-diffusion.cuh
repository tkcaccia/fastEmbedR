// Resident 2D UMAP diffusion on sharded wide CSR graphs.
#ifdef FASTEMBEDR_HAS_NCCL
namespace {

using DiffusionClock = std::chrono::steady_clock;

double diffusion_seconds(DiffusionClock::time_point start) {
  return std::chrono::duration<double>(DiffusionClock::now() - start).count();
}

__global__ void wide_diffusion_rows(
    int first, int rows, const std::uint64_t* offsets, const int* neighbors,
    const float* weights, const float* current, float* next) {
  const auto local = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  if (local >= static_cast<std::size_t>(rows)) return;
  float sx = 0, sy = 0, sw = 0;
  for (auto pos = offsets[local]; pos < offsets[local + 1]; ++pos) {
    const auto at = static_cast<std::size_t>(neighbors[pos]) * 2;
    const float weight = weights[pos];
    sx += weight * current[at];
    sy += weight * current[at + 1];
    sw += weight;
  }
  const auto out = (static_cast<std::size_t>(first) + local) * 2;
  next[out] = sw > 0 ? sx / sw : current[out];
  next[out + 1] = sw > 0 ? sy / sw : current[out + 1];
}

struct WideDiffusionShard {
  explicit WideDiffusionShard(int id) : device(id) {}
  ~WideDiffusionShard() {
    cudaSetDevice(device);
    if (retained_layout) cudaFree(retained_layout);
  }
  CudaWorkspace graph{false};
  CudaWorkspace scratch{false};
  const int device;
  int first = 0;
  int rows = 0;
  std::uint64_t edges = 0;
  std::uint64_t* offsets = nullptr;
  int* neighbors = nullptr;
  float* weights = nullptr;
  float* retained_layout = nullptr;
  std::size_t layout_items = 0;
  float* current = nullptr;
  float* next = nullptr;
  double* partial = nullptr;
  double* stats = nullptr;
  std::size_t graph_bytes = 0;
  std::size_t scratch_bytes = 0;
};

struct WideDiffusionComms {
  explicit WideDiffusionComms(int devices) : values(devices, nullptr) {}
  ~WideDiffusionComms() {
    for (auto comm : values) {
      if (!comm) continue;
      if (complete) ncclCommDestroy(comm);
      else ncclCommAbort(comm);
    }
  }
  std::vector<ncclComm_t> values;
  bool complete = false;
};

struct WideDiffusionDevice {
  int previous = 0;
  ~WideDiffusionDevice() { cudaSetDevice(previous); }
};

using WideDeviceGraph = std::vector<std::unique_ptr<WideDiffusionShard>>;

float** wide_layout_slot(void* retained, int rank, int device,
    std::size_t items) {
  auto& graph = *static_cast<WideDeviceGraph*>(retained);
  if (static_cast<std::size_t>(rank) < graph.size()) {
    auto& shard = *graph[rank];
    if (shard.device == device && shard.layout_items == items &&
        shard.retained_layout) return &shard.retained_layout;
  }
  set_embedding_error("wide UMAP layout does not match optimizer ownership");
  return nullptr;
}

std::size_t wide_graph_bytes(int rows, std::uint64_t edges) {
  const auto storage = std::max<std::uint64_t>(edges, 1);
  return align_bytes((static_cast<std::size_t>(rows) + 1) *
      sizeof(std::uint64_t)) + align_bytes(storage * sizeof(int)) +
    align_bytes(storage * sizeof(float));
}

int allocate_wide_graph(WideDiffusionShard& shard) {
  const auto storage = std::max<std::uint64_t>(shard.edges, 1);
  if (shard.graph.init(shard.graph_bytes, "wide CSR graph")) return 1;
  shard.offsets = shard.graph.alloc<std::uint64_t>(shard.rows + 1, "offsets");
  shard.neighbors = shard.graph.alloc<int>(storage, "neighbors");
  shard.weights = shard.graph.alloc<float>(storage, "weights");
  return !shard.offsets || !shard.neighbors || !shard.weights;
}

std::size_t wide_diffusion_bytes(std::size_t items, int stat_blocks) {
  return 2 * align_bytes(items * sizeof(float)) +
    align_bytes(static_cast<std::size_t>(stat_blocks) * 5 * sizeof(double)) +
    align_bytes(5 * sizeof(double));
}

int wide_graph_upload(WideDiffusionShard& shard, const double* offsets,
    const int* neighbors, const int* weight_bits) {
  std::vector<std::uint64_t> local_offsets(shard.rows + 1);
  const auto begin = static_cast<std::uint64_t>(offsets[shard.first]);
  for (int i = 0; i <= shard.rows; ++i) {
    local_offsets[i] = static_cast<std::uint64_t>(offsets[shard.first + i]) -
      begin;
  }
  static_assert(sizeof(int) == sizeof(float));
  const auto edges = local_offsets.back();
  if (allocate_wide_graph(shard)) return 1;
  if (check_cuda(cudaMemcpy(shard.offsets, local_offsets.data(),
        local_offsets.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
        "wide CSR offsets upload") ||
      check_cuda(cudaMemcpy(shard.neighbors, neighbors + begin,
        edges * sizeof(int), cudaMemcpyHostToDevice),
        "wide CSR neighbors upload") ||
      check_cuda(cudaMemcpy(shard.weights, weight_bits + begin,
        edges * sizeof(float), cudaMemcpyHostToDevice),
        "wide CSR weights upload")) return 1;
  return 0;
}

int allocate_wide_diffusion(WideDiffusionShard& shard, int block,
    int stat_blocks, std::size_t items, int iterations, bool retain_layout) {
  const auto stats = static_cast<std::size_t>(stat_blocks) * 5;
  const auto retained_bytes = retain_layout ?
    align_bytes(items * sizeof(float)) : 0;
  if (shard.scratch.init(shard.scratch_bytes - retained_bytes,
        "wide CSR diffusion")) return 1;
  shard.current = shard.scratch.alloc<float>(items, "current layout");
  if (retain_layout) {
    if (check_cuda(cudaMalloc(reinterpret_cast<void**>(&shard.retained_layout),
          items * sizeof(float)), "wide UMAP retained layout")) return 1;
    shard.layout_items = items;
    shard.next = shard.retained_layout;
    // After the last swap, current refers to the transferable allocation.
    if (iterations % 2 == 0) std::swap(shard.current, shard.next);
  } else {
    shard.next = shard.scratch.alloc<float>(items, "next layout");
  }
  shard.partial = shard.scratch.alloc<double>(stats, "partial layout norms");
  shard.stats = shard.scratch.alloc<double>(5, "layout norms");
  if (!shard.current || !shard.next || !shard.partial || !shard.stats) return 1;
  const auto padding = static_cast<std::size_t>(block - shard.rows) * 2;
  if (!padding) return 0;
  const auto suffix = (static_cast<std::size_t>(shard.first) + shard.rows) * 2;
  if (check_cuda(cudaMemset(shard.current + suffix, 0, padding * sizeof(float)),
        "wide CSR current padding")) return 1;
  return check_cuda(cudaMemset(shard.next + suffix, 0, padding * sizeof(float)),
    "wide CSR next padding");
}

__global__ void finalize_wide_layout_scale(const double* partial,
    double* stats, int blocks, int n, double target) {
  const int axis = threadIdx.x;
  double sum = 0, squares = 0;
  for (int block = 0; block < blocks; ++block) {
    const auto offset = static_cast<std::size_t>(block) * 5;
    sum += partial[offset + axis];
    squares += partial[offset + 2 + 2 * axis];
  }
  const double mean = sum / n;
  const double sd = sqrt(fmax(0.0, squares - sum * mean) / (n - 1));
  stats[axis] = mean;
  stats[2 + 2 * axis] = target / (sd > 0 ? sd : 1.0);
  if (axis == 0) stats[3] = 0;
}

int scale_wide_layout(WideDiffusionShard& shard, int n, int stat_blocks,
    int threads, double target) {
  const auto shared = static_cast<std::size_t>(threads) * 5 * sizeof(double);
  reduce_init_stats_kernel<<<stat_blocks, threads, shared>>>(
    shard.current, shard.partial, n);
  if (check_cuda(cudaGetLastError(), "wide UMAP scale reduction")) return 1;
  finalize_wide_layout_scale<<<1, 2>>>(shard.partial, shard.stats,
    stat_blocks, n, target);
  if (check_cuda(cudaGetLastError(), "wide UMAP scale statistics")) return 1;
  normalize_init_kernel<<<(n - 1) / threads + 1, threads>>>(
    shard.current, shard.stats, n);
  return check_cuda(cudaGetLastError(), "wide UMAP layout scale");
}

}  // namespace

#include "wide-csr-cuda-build.cuh"

// CSR structure and device uniqueness are checked by the Rcpp bridge.
extern "C" int fastembedr_cuda_umap_wide_init(
    int n, const double* offsets, const int* neighbors, const int* weight_bits,
    int iterations, unsigned int seed, const int* devices, int cards,
    double device_limit, float* output, int* shard_rows,
    double* edges, double* clocks, void** retained, double layout_scale) try {
  const bool retain_layout = !output && retained;
  auto* existing = retained ? static_cast<WideDeviceGraph*>(*retained) : nullptr;
  embedding_last_error.clear();
  if (n <= 1 || n > INT_MAX - 256 || cards < 1 || cards > n ||
      iterations < 1 || !(device_limit > 0) || !devices ||
      (!output && !retained) ||
      !shard_rows || !edges || !clocks ||
      (existing && existing->size() != static_cast<std::size_t>(cards)) ||
      (!existing && (!offsets || !neighbors || !weight_bits))) {
    set_embedding_error("invalid wide CSR diffusion controls");
    return 1;
  }
  constexpr int threads = 256;
  const int block = (n - 1) / cards + 1;
  if (static_cast<std::int64_t>(block) * (cards - 1) >= n) {
    set_embedding_error("wide CSR diffusion needs rows on every device");
    return 1;
  }
  const auto send_count = static_cast<std::size_t>(block) * 2;
  const auto items = send_count * cards;
  const int blocks = (n - 1) / threads + 1;
  const int stat_blocks = std::min(blocks, 1024);
  const auto started = DiffusionClock::now();
  WideDiffusionDevice restore;
  if (check_cuda(cudaGetDevice(&restore.previous),
      "cudaGetDevice(wide CSR)")) return 1;
  WideDeviceGraph uploaded;
  auto& shards = existing ? *existing : uploaded;
  clocks[3] = 0;
  for (int rank = 0; rank < cards; ++rank) {
    if (check_cuda(cudaSetDevice(devices[rank]), "wide CSR device")) return 1;
    const int first = block * rank;
    const int rows = std::min(block, n - first);
    if (!existing) {
      auto shard = std::make_unique<WideDiffusionShard>(devices[rank]);
      shard->first = first;
      shard->rows = rows;
      shard->edges = offsets[first + rows] - offsets[first];
      shard->graph_bytes = wide_graph_bytes(rows, shard->edges);
      shards.push_back(std::move(shard));
    }
    auto& shard = *shards[rank];
    if (shard.device != devices[rank] || shard.first != first ||
        shard.rows != rows || shard.retained_layout) {
      set_embedding_error("wide diffusion CSR owners do not match");
      return 1;
    }
    shard.scratch_bytes = wide_diffusion_bytes(items, stat_blocks);
    const auto bytes = shard.graph_bytes + shard.scratch_bytes;
    const auto layout_bytes = items * sizeof(float);
    const auto fit_scratch = align_bytes(layout_bytes) + 2 * layout_bytes +
      3 * 16;
    const auto scratch = retained ?
      std::max(shard.scratch_bytes, fit_scratch) : shard.scratch_bytes;
    std::size_t free = 0, total = 0;
    if (check_cuda(cudaMemGetInfo(&free, &total),
        "wide CSR workspace preflight")) return 1;
    if (scratch + (existing ? 0 : shard.graph_bytes) > free - free / 20 ||
        static_cast<double>(scratch + shard.graph_bytes) > device_limit) {
      return 2;
    }
    shard_rows[rank] = rows;
    edges[rank] = static_cast<double>(shard.edges);
    clocks[3] = std::max(clocks[3], static_cast<double>(bytes));
  }
  WideDiffusionComms comms(cards);
  if (cards > 1 && massive_umap_nccl_check(
        ncclCommInitAll(comms.values.data(), cards, devices),
        "wide CSR NCCL setup")) return 1;
  for (const auto& shard : shards) {
    if (check_cuda(cudaSetDevice(shard->device), "wide CSR upload device") ||
        (!existing && wide_graph_upload(*shard, offsets, neighbors,
          weight_bits)) ||
        allocate_wide_diffusion(*shard, block, stat_blocks, items,
          iterations, retain_layout)) return 1;
  }
  clocks[0] = diffusion_seconds(started);
  const auto iteration_start = DiffusionClock::now();
  for (const auto& shard : shards) {
    if (check_cuda(cudaSetDevice(shard->device),
        "wide CSR initial layout device")) return 1;
    random_init_kernel<<<blocks, threads>>>(shard->current, n, seed);
    if (check_cuda(cudaGetLastError(), "wide CSR random layout") ||
        normalize_device_init(shard->current, shard->partial, shard->stats,
          n, stat_blocks, threads)) return 1;
  }
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (const auto& shard : shards) {
      if (check_cuda(cudaSetDevice(shard->device),
          "wide CSR diffusion device")) return 1;
      wide_diffusion_rows<<<(shard->rows - 1) / threads + 1, threads>>>(
        shard->first, shard->rows, shard->offsets, shard->neighbors,
        shard->weights, shard->current, shard->next);
      if (check_cuda(cudaGetLastError(), "wide CSR diffusion rows")) return 1;
    }
    if (cards > 1) {
      if (massive_umap_nccl_check(ncclGroupStart(),
          "wide CSR gather group")) return 1;
      int status = 0;
      for (int rank = 0; rank < cards; ++rank) {
        const auto& shard = shards[rank];
        status |= check_cuda(cudaSetDevice(shard->device),
          "wide CSR gather device");
        status |= massive_umap_nccl_check(ncclAllGather(
          shard->next + rank * send_count, shard->next, send_count,
          ncclFloat, comms.values[rank], nullptr), "wide CSR all-gather");
      }
      status |= massive_umap_nccl_check(ncclGroupEnd(),
        "wide CSR gather group end");
      if (status) return 1;
    }
    for (const auto& shard : shards) {
      if (check_cuda(cudaSetDevice(shard->device),
          "wide CSR normalization device") ||
          normalize_device_init(shard->next, shard->partial, shard->stats,
            n, stat_blocks, threads)) return 1;
      std::swap(shard->current, shard->next);
    }
  }
  for (const auto& shard : shards) {
    if (check_cuda(cudaSetDevice(shard->device), "wide CSR sync device") ||
        check_cuda(cudaDeviceSynchronize(), "wide CSR diffusion sync")) {
      return 1;
    }
  }
  clocks[1] = diffusion_seconds(iteration_start);
  const auto export_start = DiffusionClock::now();
  if (retain_layout) {
    if (std::isfinite(layout_scale) && layout_scale > 0) {
      for (const auto& shard : shards) {
        if (check_cuda(cudaSetDevice(shard->device),
            "wide UMAP scale device") ||
            scale_wide_layout(*shard, n, stat_blocks, threads, layout_scale)) {
          return 1;
        }
      }
      for (const auto& shard : shards) {
        if (check_cuda(cudaSetDevice(shard->device),
            "wide UMAP scale device") ||
            check_cuda(cudaDeviceSynchronize(), "wide UMAP scale sync")) {
          return 1;
        }
      }
    }
  } else if (check_cuda(cudaSetDevice(devices[0]),
             "wide CSR download device") ||
             check_cuda(cudaMemcpy(output, shards[0]->current,
               static_cast<std::size_t>(n) * 2 * sizeof(float),
               cudaMemcpyDeviceToHost), "wide CSR layout download")) {
    return 1;
  }
  clocks[2] = diffusion_seconds(export_start);
  comms.complete = true;
  if (retained) {
    for (const auto& shard : shards) {
      if (check_cuda(cudaSetDevice(shard->device),
          "wide CSR release diffusion device") ||
          shard->scratch.init(0, "release diffusion workspace")) return 1;
    }
    if (!existing) *retained = new WideDeviceGraph(std::move(uploaded));
  }
  return 0;
} catch (const std::exception& error) {
  set_embedding_error(error.what());
  return 1;
}

extern "C" double fastembedr_cuda_umap_wide_graph_allocated_bytes(
    const void* graph) {
  double bytes = 0.0;
  for (const auto& shard : *static_cast<const WideDeviceGraph*>(graph)) {
    bytes += shard->graph_bytes;
  }
  return bytes;
}

extern "C" void fastembedr_cuda_umap_wide_graph_destroy(void* graph) {
  WideDiffusionDevice restore;
  cudaGetDevice(&restore.previous);
  delete static_cast<WideDeviceGraph*>(graph);
}

namespace {

__global__ void wide_csr_delta_kernel(const float* layout,
    unsigned long long* delta, int first, int rows,
    const std::uint64_t* offsets, const int* neighbors, const float* weights,
    EmbedParams params, unsigned int epoch, double scale) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  const auto last = offsets[row + 1];
  const float min_weight = params.max_weight / params.n_epochs;
  for (auto edge = offsets[row] + threadIdx.x % 32; edge < last; edge += 32) {
    const float weight = weights[edge];
    if (weight < min_weight) continue;
    embed_epoch_delta_edge<2>(layout, delta, first + row, neighbors[edge],
      weight, params.max_weight / fmaxf(weight, 1e-6f), params, epoch, scale);
  }
}

}  // namespace

extern "C" int fastembedr_cuda_umap_wide_graph_step(
    void* raw, void* csr, int epoch) {
  auto* context = static_cast<MassiveUmapMultiCudaContext*>(raw);
  auto* graph = static_cast<WideDeviceGraph*>(csr);
  if (!context || !graph || context->dimensions != 2 || epoch < 0 ||
      graph->size() != context->devices.size()) {
    set_embedding_error("invalid wide CSR optimizer request");
    return 1;
  }
  WideDiffusionDevice restore;
  if (check_cuda(cudaGetDevice(&restore.previous),
      "wide CSR optimizer current device")) return 1;
  constexpr int threads = 256;
  for (std::size_t rank = 0; rank < graph->size(); ++rank) {
    const auto& shard = *(*graph)[rank];
    auto* worker = context->workers[rank];
    if (shard.device != context->devices[rank] ||
        shard.first != static_cast<int>(rank) * context->block_rows ||
        shard.rows != std::min(context->block_rows,
          context->vertices - shard.first) || epoch >= worker->params.n_epochs) {
      set_embedding_error("wide CSR optimizer does not match graph owners");
      return 1;
    }
    if (check_cuda(cudaSetDevice(shard.device),
        "wide CSR optimizer device")) return 1;
    const int blocks = (shard.rows - 1) / (threads / 32) + 1;
    wide_csr_delta_kernel<<<blocks, threads, 0, context->upload_streams[rank]>>>(
      worker->layout, context->deltas[rank], shard.first, shard.rows,
      shard.offsets, shard.neighbors, shard.weights, worker->params,
      static_cast<unsigned int>(epoch), context->delta_scale);
    if (check_cuda(cudaGetLastError(), "wide CSR optimizer launch")) return 1;
  }
  return 0;
}
#endif
