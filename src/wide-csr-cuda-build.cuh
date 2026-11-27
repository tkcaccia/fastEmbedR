// Included after the wide CUDA graph storage definitions.
namespace {

using WideEdgeKey = unsigned long long;
using WideRecordCount = std::int64_t;
static_assert(sizeof(WideRecordCount) <= sizeof(WideEdgeKey));

__device__ __forceinline__ WideEdgeKey wide_edge_key(int row, int peer) {
  return (static_cast<WideEdgeKey>(row) << 32) | static_cast<unsigned>(peer);
}

__global__ void wide_directed_records(int begin, int end, int n, int k,
    const int* ids, const float* distances, double target, double global_mean,
    WideEdgeKey* keys, float* weights, int index_base, bool incoming) {
  const auto row = static_cast<std::int64_t>(begin) +
    static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= end) return;
  const auto at = static_cast<std::size_t>(row - begin);
  double rho = DBL_MAX, row_sum = 0.0;
  int row_count = 0;
  for (int col = 0; col < k; ++col) {
    const auto item = at + static_cast<std::size_t>(col) * (end - begin);
    const float d = distances[item];
    if (!isfinite(d)) continue;
    if (d >= 0) {
      row_sum += d;
      ++row_count;
    }
    if (d > 0 && d < rho) rho = d;
  }
  if (rho == DBL_MAX) rho = 0.0;
  double sigma = 1.0, lo = 0.0, hi = DBL_MAX;
  double best = sigma, best_diff = DBL_MAX;
  for (int iteration = 0; iteration < 64; ++iteration) {
    double sum = 0.0;
    const double safe = fmax(sigma, 1.0e-12);
    for (int col = 0; col < k; ++col) {
      const auto item = at + static_cast<std::size_t>(col) * (end - begin);
      const float raw = distances[item];
      if (!isfinite(raw)) continue;
      const double d = static_cast<double>(raw) - rho;
      sum += d <= 0 ? 1.0 : exp(-d / safe);
    }
    const double diff = fabs(sum - target);
    if (diff < best_diff) {
      best_diff = diff;
      best = sigma;
    }
    if (sum > target) {
      hi = sigma;
      sigma = 0.5 * (lo + hi);
    } else {
      lo = sigma;
      sigma = hi == DBL_MAX ? sigma * 2.0 : 0.5 * (lo + hi);
    }
    if (diff < 1.0e-5) break;
  }
  const double row_mean = row_count ? row_sum / row_count : global_mean;
  const double floor = 1.0e-3 * (rho > 0 ? row_mean : global_mean);
  const float scale = static_cast<float>(fmax(fmax(best, floor), 1.0e-12));
  const float near = static_cast<float>(rho);
  for (int col = 0; col < k; ++col) {
    const auto pos = at + static_cast<std::size_t>(col) * (end - begin);
    const auto output = at * k + col;
    const int peer_id = ids[pos];
    const float d = distances[pos];
    const bool ok = peer_id >= index_base && peer_id - index_base < n &&
      peer_id - index_base != row &&
      isfinite(d);
    const float w = ok ? (d <= near ? 1.0f : expf(-(d - near) / scale)) : 0.0f;
    const int peer = ok ? peer_id - index_base : 0;
    const auto key = incoming ? wide_edge_key(peer, row) :
      wide_edge_key(row, peer);
    keys[output] = w > 0 ? key : ULLONG_MAX;
    weights[output] = incoming ? -w : w;
  }
}

template <typename Index>
__device__ Index edge_lower_bound(const WideEdgeKey* keys, Index count,
    WideEdgeKey target, const Index* selection = nullptr) {
  Index lo = 0, hi = count;
  while (lo < hi) {
    const Index mid = lo + (hi - lo) / 2;
    if (keys[selection ? selection[mid] : mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

template <typename Index>
__global__ void wide_fuzzy_union(Index count, const WideEdgeKey* keys,
    float* weights) {
  const auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  if (pos >= count || (pos && keys[pos] == keys[pos - 1])) return;
  float forward = 0.0f, reverse = 0.0f;
  // Only the first thread reads or writes the weights for this key.
  for (auto j = pos; j < count && keys[j] == keys[pos]; ++j) {
    forward = fmaxf(forward, weights[j]);
    reverse = fmaxf(reverse, -weights[j]);
    if (j != pos) weights[j] = 0.0f;
  }
  weights[pos] = __fsub_rn(__fadd_rn(forward, reverse),
    __fmul_rn(forward, reverse));
}

struct WideKeptWeight {
  const float* values;
  template <typename Index>
  __device__ bool operator()(Index position) const {
    return values[position] > 1e-6f;
  }
};

template <typename Index>
__global__ void wide_compact_edges(Index count, const Index* select,
    const WideEdgeKey* keys, const float* values, int* ids, float* weights) {
  const auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  if (pos >= count) return;
  const Index raw = select[pos];
  ids[pos] = static_cast<int>(keys[raw] & 0xffffffffu);
  weights[pos] = values[raw];
}

template <typename Index>
__global__ void wide_compact_offsets(int n, Index count, const Index* select,
    const WideEdgeKey* keys, std::uint64_t* offsets, int first,
    std::uint64_t base = 0) {
  const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  if (row <= static_cast<std::size_t>(n)) {
    offsets[row] = base + edge_lower_bound(keys, count,
      wide_edge_key(first + static_cast<int>(row), 0), select);
  }
}

constexpr int wide_build_threads = 256;

int wide_build_blocks(WideRecordCount items) {
  const auto blocks = (items - 1) / wide_build_threads + 1;
  if (items <= 0 || blocks > INT_MAX) {
    throw std::runtime_error("CUDA fuzzy graph launch exceeds its grid limit.");
  }
  return static_cast<int>(blocks);
}

template <typename Callback>
auto wide_build_dispatch(WideRecordCount count, Callback callback) {
  if (count <= INT_MAX) return callback(static_cast<int>(count));
#if CUB_VERSION >= 200800
  return callback(count);
#else
  throw std::runtime_error(
    "Wide CUDA fuzzy graph records require CUB 2.8 or newer.");
#endif
}

void wide_build_check(cudaError_t status, const char* label) {
  if (check_cuda(status, label)) throw std::runtime_error(embedding_last_error);
}

void wide_build_require(std::size_t owned, std::size_t requested,
    double limit) {
  std::size_t free = 0, total = 0;
  wide_build_check(cudaMemGetInfo(&free, &total), "wide graph device memory");
  if (requested > free - free / 20 ||
      static_cast<double>(owned) + requested > limit) {
    throw std::runtime_error("CUDA fuzzy graph exceeds the device budget.");
  }
}

std::size_t wide_build_record_bytes(WideRecordCount items) {
  constexpr std::size_t largest_buffer =
    std::numeric_limits<std::size_t>::max() / 32;
  if (items < 0 || static_cast<std::uint64_t>(items) > largest_buffer) {
    throw std::runtime_error("CUDA fuzzy graph record allocation overflow.");
  }
  wide_build_dispatch(items, [](auto) {});
  const auto count = static_cast<std::size_t>(
    std::max<WideRecordCount>(items, 1));
  return 2 * align_bytes(count * sizeof(WideEdgeKey)) +
    2 * align_bytes(count * sizeof(float));
}

struct WideBuildRecords {
  WideBuildRecords(int owner, WideRecordCount count)
    : device(owner), items(count) {}
  ~WideBuildRecords() { cudaSetDevice(device); }

  void allocate(std::size_t owned, double limit) {
    const auto count = static_cast<std::size_t>(
      std::max<WideRecordCount>(items, 1));
    bytes = wide_build_record_bytes(items);
    wide_build_require(owned, bytes, limit);
    weight_bytes = align_bytes(count * sizeof(float));
    if (storage.init(bytes - 2 * weight_bytes, "wide graph keys") ||
        first_weights.init(weight_bytes, "wide graph weights") ||
        second_weights.init(weight_bytes, "wide graph scratch weights")) {
      throw std::runtime_error(embedding_last_error);
    }
    auto* k1 = storage.alloc<WideEdgeKey>(count, "record keys");
    auto* k2 = storage.alloc<WideEdgeKey>(count, "record keys scratch");
    auto* w1 = first_weights.alloc<float>(count, "record weights");
    auto* w2 = second_weights.alloc<float>(count, "record weights scratch");
    keys = cub::DoubleBuffer<WideEdgeKey>(k1, k2);
    weights = cub::DoubleBuffer<float>(w1, w2);
  }

  void release_scratch_weights() {
    auto& scratch = weights.selector == 0 ? second_weights : first_weights;
    scratch.init(0, "unused graph weights");
    bytes -= weight_bytes;
  }

  void sort(std::size_t owned, double limit) {
    if (items < 2) return;
    wide_build_dispatch(items, [&](auto count) {
      std::size_t size = 0;
      wide_build_check(cub::DeviceRadixSort::SortPairs(nullptr, size,
        keys, weights, count), "wide graph radix-sort workspace");
      wide_build_require(owned + bytes, align_bytes(size), limit);
      CudaWorkspace scratch{false};
      if (scratch.init(size, "wide graph radix sort")) {
        throw std::runtime_error(embedding_last_error);
      }
      auto* tmp = scratch.alloc<unsigned char>(size, "radix scratch");
      wide_build_check(cub::DeviceRadixSort::SortPairs(tmp, size,
        keys, weights, count), "wide graph radix sort");
      wide_build_check(cudaDeviceSynchronize(), "wide graph sort completion");
    });
  }

  const int device;
  WideRecordCount items;
  std::size_t bytes = 0;
  std::size_t weight_bytes = 0;
  CudaWorkspace storage{false};
  CudaWorkspace first_weights{false};
  CudaWorkspace second_weights{false};
  cub::DoubleBuffer<WideEdgeKey> keys;
  cub::DoubleBuffer<float> weights;
};

template <typename Task>
auto wide_build_ranks(int cards, Task task) {
  using Result = decltype(task(0));
  std::vector<Result> result;
  result.reserve(cards);
  if (cards == 1) {
    result.push_back(task(0));
    return result;
  }
  std::vector<std::future<Result>> pending;
  pending.reserve(cards);
  for (int rank = 0; rank < cards; ++rank) {
    pending.emplace_back(std::async(std::launch::async, task, rank));
  }
  for (auto& submitted : pending) result.push_back(submitted.get());
  return result;
}

struct WideNeighborSource {
  const int* indices;
  const int* bits;
  int n;
  int k;
  int index_base;
  double mean;
};

int wide_knn_upload_rows(int rows, int k) {
  constexpr std::size_t tile_bytes = 16 * 1024 * 1024;
  const auto row_bytes = 2 * static_cast<std::size_t>(k) * sizeof(int);
  return std::min(rows,
    static_cast<int>(std::max<std::size_t>(1, tile_bytes / row_bytes)));
}

void pack_wide_records(WideBuildRecords& record,
    const WideNeighborSource& source, int first, int rows,
    bool incoming, double limit) {
  const int k = source.k;
  const int batch = wide_knn_upload_rows(rows, k);
  const auto cells = static_cast<std::size_t>(batch) * k;
  const auto input_bytes = 2 * align_bytes(cells * sizeof(int));
  wide_build_require(record.bytes, input_bytes, limit);
  CudaWorkspace input{false};
  if (input.init(input_bytes, "wide graph KNN tile")) {
    throw std::runtime_error(embedding_last_error);
  }
  auto* ids = input.alloc<int>(cells, "neighbor IDs");
  auto* distances = input.alloc<float>(cells, "neighbor distances");
  const auto host_stride = static_cast<std::size_t>(source.n) * sizeof(int);
  const double target = std::log2(k);
  for (int offset = 0; offset < rows;) {
    const int active = std::min(batch, rows - offset);
    const auto stride = static_cast<std::size_t>(active) * sizeof(int);
    const auto begin = first + offset;
    const auto output = static_cast<std::size_t>(offset) * k;
    wide_build_check(cudaMemcpy2D(ids, stride, source.indices + begin,
      host_stride, stride, k, cudaMemcpyHostToDevice),
      "wide graph neighbor upload");
    wide_build_check(cudaMemcpy2D(distances, stride, source.bits + begin,
      host_stride, stride, k, cudaMemcpyHostToDevice),
      "wide graph distance upload");
    wide_directed_records<<<wide_build_blocks(active), wide_build_threads>>>(
      begin, begin + active, source.n, k, ids, distances, target, source.mean,
      record.keys.Current() + output, record.weights.Current() + output,
      source.index_base, incoming);
    wide_build_check(cudaGetLastError(), "wide graph membership");
    offset += active;
  }
  wide_build_check(cudaDeviceSynchronize(), "wide graph membership finished");
}

template <typename Index>
float compact_wide_records_typed(WideDiffusionShard& shard,
    WideBuildRecords& records, Index count, double limit,
    int row_begin, int rows) {
  const auto base = shard.edges;
  const auto row_offset = row_begin - shard.first;
  if (!count) {
    wide_compact_offsets<<<wide_build_blocks(rows + 1), wide_build_threads>>>(
      rows, Index{0}, static_cast<const Index*>(nullptr),
      static_cast<const WideEdgeKey*>(nullptr), shard.offsets + row_offset,
      row_begin, base);
    wide_build_check(cudaDeviceSynchronize(), "wide graph empty CSR");
    return 0;
  }
  const auto* keys = records.keys.Current();
  records.release_scratch_weights();
  auto* fuzzy = records.weights.Current();
  wide_fuzzy_union<<<wide_build_blocks(count), wide_build_threads>>>(
    count, keys, fuzzy);
  wide_build_check(cudaGetLastError(), "wide graph fuzzy union");
  std::size_t size = 0;
  thrust::counting_iterator<Index> positions(0);
  wide_build_check(cub::DeviceSelect::If(nullptr, size, positions,
    static_cast<Index*>(nullptr), static_cast<Index*>(nullptr), count,
    WideKeptWeight{fuzzy}), "wide graph compact workspace");
  const auto capacity = align_bytes(sizeof(Index)) + align_bytes(size);
  wide_build_require(shard.graph_bytes + records.bytes, capacity, limit);
  CudaWorkspace select{false};
  if (select.init(capacity, "wide graph compact")) {
    throw std::runtime_error(embedding_last_error);
  }
  // The radix-sort alternate key array is free after the union.
  auto* keep = reinterpret_cast<Index*>(records.keys.Alternate());
  auto* value = select.alloc<Index>(1, "compact edge count");
  auto* tmp = select.alloc<unsigned char>(size, "compact workspace");
  wide_build_check(cub::DeviceSelect::If(tmp, size, positions, keep, value,
    count, WideKeptWeight{fuzzy}), "wide graph compact");
  Index nnz = 0;
  wide_build_check(cudaMemcpy(&nnz, value, sizeof(nnz), cudaMemcpyDeviceToHost),
    "wide graph compact edge count read");
  shard.edges = base + nnz;
  if (nnz) {
    wide_compact_edges<<<wide_build_blocks(nnz), wide_build_threads>>>(
      nnz, keep, keys, fuzzy, shard.neighbors + base, shard.weights + base);
    wide_build_check(cudaGetLastError(), "wide graph compact edges");
  }
  wide_compact_offsets<<<wide_build_blocks(rows + 1), wide_build_threads>>>(
    rows, nnz, keep, keys, shard.offsets + row_offset, row_begin, base);
  wide_build_check(cudaGetLastError(), "wide graph compact offsets");
  wide_build_check(cudaDeviceSynchronize(), "wide graph CSR completion");
  if (!nnz) return 0;
  size = 0;
  const auto* result_weights = shard.weights + base;
  wide_build_check(cub::DeviceReduce::Max(nullptr, size, result_weights,
    static_cast<float*>(nullptr), nnz), "wide graph max workspace");
  wide_build_require(records.bytes + capacity + shard.graph_bytes,
    align_bytes(size) + align_bytes(sizeof(float)), limit);
  CudaWorkspace maximum{false};
  if (maximum.init(align_bytes(size) + align_bytes(sizeof(float)),
      "wide graph max weight")) throw std::runtime_error(embedding_last_error);
  auto* scratch = maximum.alloc<unsigned char>(size, "max workspace");
  auto* max_weight = maximum.alloc<float>(1, "max weight");
  wide_build_check(cub::DeviceReduce::Max(scratch, size, result_weights,
    max_weight, nnz), "wide graph max weight");
  float result = 0;
  wide_build_check(cudaMemcpy(&result, max_weight, sizeof(float),
    cudaMemcpyDeviceToHost), "wide graph max weight read");
  return result;
}

#include "wide-csr-cuda-tiled.cuh"

}  // namespace

extern "C" int fastembedr_cuda_umap_wide_graph_build(
    const int* indices, const int* bits, int n, int k, int index_base,
    double mean, const int* devices, int cards, double limit,
    void** retained, double* edges, double* routed, double* times,
    float* max_weight) try {
  embedding_last_error.clear();
  WideDiffusionDevice restore;
  wide_build_check(cudaGetDevice(&restore.previous), "wide graph current device");
  const int block = (n - 1) / cards + 1;
  if (static_cast<WideRecordCount>(block) * (cards - 1) >= n) {
    throw std::runtime_error("CUDA fuzzy graph needs rows on every device.");
  }
  const WideNeighborSource source{indices, bits, n, k, index_base, mean};
  const auto source_start = DiffusionClock::now();
  auto send = wide_build_ranks(cards, [&](int rank) {
    wide_build_check(cudaSetDevice(devices[rank]), "wide graph source device");
    const auto first = rank * block;
    return stage_wide_tiles(source, first, std::min(block, n - first),
      devices[rank], limit);
  });
  times[0] = diffusion_seconds(source_start);
  times[1] = 0;
  const auto compact_start = DiffusionClock::now();
  std::vector<float> maxima(cards);
  auto graph = wide_build_ranks(cards, [&](int rank) {
    wide_build_check(cudaSetDevice(devices[rank]), "wide graph owner device");
    auto shard = std::make_unique<WideDiffusionShard>(devices[rank]);
    shard->first = rank * block;
    shard->rows = std::min(block, n - shard->first);
    maxima[rank] = build_staged_owner(*shard, source, send, rank, limit,
      routed[rank]);
    edges[rank] = static_cast<double>(shard->edges);
    return shard;
  });
  *max_weight = *std::max_element(maxima.begin(), maxima.end());
  if (!(*max_weight > 0)) throw std::runtime_error("CUDA fuzzy graph is empty.");
  send.clear();
  times[2] = diffusion_seconds(compact_start);
  *retained = new WideDeviceGraph(std::move(graph));
  return 0;
} catch (const std::exception& error) {
  set_embedding_error(error.what());
  return 1;
}

namespace {
struct WideScheduledEdge {
  float minimum;
  __device__ unsigned long long operator()(float weight) const {
    return weight >= minimum;
  }
};
}  // namespace

extern "C" int fastembedr_cuda_umap_wide_graph_counts(
    void* raw, float minimum, double limit, double* scheduled) try {
  WideDiffusionDevice restore;
  wide_build_check(cudaGetDevice(&restore.previous), "wide graph count device");
  const auto& graph = *static_cast<WideDeviceGraph*>(raw);
  for (std::size_t rank = 0; rank < graph.size(); ++rank) {
    const auto& shard = *graph[rank];
    scheduled[rank] = 0.0;
    if (!shard.edges) continue;
    wide_build_check(cudaSetDevice(shard.device), "wide graph count owner");
    wide_build_dispatch(static_cast<WideRecordCount>(shard.edges),
        [&](auto count) {
      thrust::transform_iterator<WideScheduledEdge, const float*> weights(
        shard.weights, WideScheduledEdge{minimum});
      std::size_t size = 0;
      using Count = unsigned long long;
      wide_build_check(cub::DeviceReduce::Sum(nullptr, size, weights,
        static_cast<Count*>(nullptr), count), "wide graph count workspace");
      const auto bytes = align_bytes(size) + align_bytes(sizeof(Count));
      wide_build_require(shard.graph_bytes, bytes, limit);
      CudaWorkspace workspace{false};
      if (workspace.init(bytes,
          "wide graph edge counts")) throw std::runtime_error(embedding_last_error);
      auto* scratch = workspace.alloc<unsigned char>(size, "count workspace");
      auto* output = workspace.alloc<Count>(1, "count output");
      wide_build_check(cub::DeviceReduce::Sum(scratch, size, weights,
        output, count), "wide graph eligible count");
      Count total = 0;
      wide_build_check(cudaMemcpy(&total, output, sizeof(total),
        cudaMemcpyDeviceToHost), "wide graph count read");
      scheduled[rank] = static_cast<double>(total);
    });
  }
  return 0;
} catch (const std::exception& error) {
  set_embedding_error(error.what());
  return 1;
}
