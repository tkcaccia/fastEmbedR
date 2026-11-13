#include "graph_clustering_cuda_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int kColorCount = 8;
constexpr int kLocalCandidateCount = 64;
constexpr std::uint32_t kEdgeTileSize = 64U;

struct EdgeTileCursor {
  std::uint64_t base;
  std::uint64_t end;
  std::uint32_t local = 0;
  std::uint32_t count = 0;

  __device__ EdgeTileCursor(std::uint64_t begin,
                            std::uint64_t stop) : base(begin), end(stop) {
    refill();
  }
  __device__ void refill() {
    const auto remaining = end - base;
    count = static_cast<std::uint32_t>(
      remaining < kEdgeTileSize ? remaining : kEdgeTileSize);
    local = 0;
  }
  __device__ bool valid() const { return base < end; }
  __device__ std::uint64_t next() {
    const auto position = base + local++;
    if (local == count) {
      base += count;
      if (base < end) refill();
    }
    return position;
  }
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  Rcpp::stop(message.str());
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
        "cudaMalloc"
      );
    }
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

__device__ __forceinline__ unsigned int mix32(unsigned int value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

__device__ __forceinline__ int vertex_color(
    int vertex, unsigned int seed) {
  return static_cast<int>(mix32(
    static_cast<unsigned int>(vertex) ^ seed
  ) & (kColorCount - 1));
}

template <typename Offset>
__global__ void propose_moves_kernel(
    const Offset* row_ptr,
    const int* col_idx,
    const float* edge_weight,
    const float* degree,
    const int* labels,
    const int* parent,
    const float* community_volume,
    const int* community_count,
    int* proposal,
    int n,
    int first,
    int rows,
    float graph_volume,
    float resolution,
    float tolerance,
    int color,
    unsigned int seed,
    int refinement,
    int leiden_cuts,
    float* cuts,
    const float* parent_volume,
    float* proposal_weight) {
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= rows) return;
  const int u = first + local;
  if (u >= n || vertex_color(u, seed) != color) return;

  const int old = labels[u];
  proposal[u] = old;
  const float node_degree = degree[u];
  if (!(node_degree > 0.0f)) return;
  if (refinement && community_count[old] != 1) return;
  if (leiden_cuts) {
    const float volume = parent_volume[parent[u]];
    const float threshold = resolution * node_degree *
      (volume - node_degree) / graph_volume;
    if (cuts[old] + tolerance < threshold) return;
  }

  const std::uint64_t begin = row_ptr[local];
  const std::uint64_t end = row_ptr[local + 1];
  int candidate_labels[kLocalCandidateCount];
  float candidate_weights[kLocalCandidateCount];
  int unique_count = 0;
  bool overflow = false;
  float old_inside = 0.0f;
  for (EdgeTileCursor tile(begin, end); tile.valid();) {
    const auto position = tile.next();
    const int v = col_idx[position];
    if (v == u || (refinement && parent[v] != parent[u])) continue;
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
    } else if (unique_count < kLocalCandidateCount) {
      candidate_labels[unique_count] = candidate;
      candidate_weights[unique_count] = edge_weight[position];
      ++unique_count;
    } else {
      overflow = true;
    }
  }
  const float old_volume = fmaxf(0.0f, community_volume[old] - node_degree);
  const float old_score =
    old_inside - resolution * node_degree * old_volume / graph_volume;
  float best_score = leiden_cuts ? -INFINITY : old_score;
  int best = old;
  float best_weight = 0.0f;

  if (!overflow) {
    for (int index = 0; index < unique_count; ++index) {
      const int candidate = candidate_labels[index];
      if (candidate == old || community_count[candidate] <= 0 ||
          (leiden_cuts &&
           vertex_color(candidate, seed) >= color)) continue;
      const float candidate_volume = community_volume[candidate];
      if (leiden_cuts) {
        const float parent_total = parent_volume[parent[u]];
        const float threshold = resolution * candidate_volume *
          (parent_total - candidate_volume) / graph_volume;
        if (cuts[candidate] + tolerance < threshold) continue;
      }
      const float score = candidate_weights[index] -
        resolution * node_degree * candidate_volume / graph_volume;
      if (leiden_cuts && score < -tolerance) continue;
      if (score > best_score + tolerance ||
          (fabsf(score - best_score) <= tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
        best_weight = candidate_weights[index];
      }
    }
  } else {
    for (EdgeTileCursor candidates(begin, end);
         candidates.valid();) {
      const auto candidate_position = candidates.next();
      const int neighbor = col_idx[candidate_position];
      if (neighbor == u) continue;
      if (refinement && parent[neighbor] != parent[u]) continue;
      const int candidate = labels[neighbor];
      if (candidate == old || community_count[candidate] <= 0 ||
          (leiden_cuts &&
           vertex_color(candidate, seed) >= color)) continue;
      bool seen = false;
      for (EdgeTileCursor prior(begin, candidate_position);
           prior.valid();) {
        const auto previous = prior.next();
        const int previous_neighbor = col_idx[previous];
        if (previous_neighbor != u &&
            (!refinement || parent[previous_neighbor] == parent[u]) &&
            labels[previous_neighbor] == candidate) {
          seen = true;
          break;
        }
      }
      if (seen) continue;
      float inside = 0.0f;
      for (EdgeTileCursor tile(begin, end); tile.valid();) {
        const auto position = tile.next();
        const int v = col_idx[position];
        if (v != u &&
            (!refinement || parent[v] == parent[u]) &&
            labels[v] == candidate) {
          inside += edge_weight[position];
        }
      }
      const float score = inside - resolution * node_degree *
        community_volume[candidate] / graph_volume;
      if (leiden_cuts) {
        const float candidate_volume = community_volume[candidate];
        const float parent_total = parent_volume[parent[u]];
        const float threshold = resolution * candidate_volume *
          (parent_total - candidate_volume) / graph_volume;
        if (cuts[candidate] + tolerance < threshold ||
            score < -tolerance) continue;
      }
      if (score > best_score + tolerance ||
          (fabsf(score - best_score) <= tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
        best_weight = inside;
      }
    }
  }
  if (best != old &&
      (leiden_cuts || best_score > old_score + tolerance)) {
    proposal[u] = best;
    if (leiden_cuts) proposal_weight[u] = best_weight;
  }
}

template <typename Offset>
__global__ void apply_moves_kernel(
    const float* degree,
    int* labels,
    const int* proposal,
    float* community_volume,
    int* community_count,
    unsigned int* move_count,
    int n,
    int first,
    int rows,
    int color,
    unsigned int seed,
    int leiden_cuts,
    const Offset* row_ptr,
    const int* col_idx,
    const float* edge_weight,
    float* cuts,
    const float* proposal_weight) {
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= rows) return;
  const int u = first + local;
  if (u >= n || vertex_color(u, seed) != color) return;
  const int old = labels[u];
  const int next = proposal[u];
  if (next == old) return;
  if (leiden_cuts) {
    if (community_count[old] != 1) return;
    float simultaneous = 0.0f;
    for (EdgeTileCursor tile(row_ptr[local], row_ptr[local + 1]);
         tile.valid();) {
      const auto position = tile.next();
      const int v = col_idx[position];
      if (v != u && v >= first && v < first + rows &&
          vertex_color(v, seed) == color && proposal[v] == next) {
        simultaneous += edge_weight[position];
      }
    }
    const float old_cut = atomicExch(&cuts[old], 0.0f);
    atomicAdd(&cuts[next],
      old_cut - 2.0f * proposal_weight[u] - simultaneous);
  }
  labels[u] = next;
  atomicAdd(&community_volume[old], -degree[u]);
  atomicAdd(&community_volume[next], degree[u]);
  atomicSub(&community_count[old], 1);
  atomicAdd(&community_count[next], 1);
  atomicAdd(move_count, 1U);
}

__global__ void accumulate_fragment_kernel(
    const int* col, const float* weight, int edges, int row,
    const int* labels, const int* parent, int refinement,
    int* mark, int* touched, float* total, int* count) {
  if (blockIdx.x || threadIdx.x) return;
  for (int edge = 0; edge < edges; ++edge) {
    const int neighbor = col[edge];
    if (neighbor == row ||
        (refinement && parent[neighbor] != parent[row])) continue;
    const int label = labels[neighbor];
    if (!mark[label]) {
      mark[label] = 1;
      touched[(*count)++] = label;
      total[label] = 0.0f;
    }
    total[label] += weight[edge];
  }
}

__global__ void finalize_fragment_kernel(
    int row, const float* degree, const int* labels,
    const int* parent, const float* volume,
    const int* community_count, int* proposal,
    float graph_volume, float resolution, float tolerance,
    int color, unsigned int seed, int refinement,
    int enforce_color,
    const float* cuts, const float* parent_volume,
    float* proposal_weight, int* mark, const int* touched,
    const float* total, int* count) {
  if (blockIdx.x || threadIdx.x) return;
  const int old = labels[row];
  proposal[row] = old;
  if (enforce_color && vertex_color(row, seed) != color) {
    for (int index = 0; index < *count; ++index) {
      mark[touched[index]] = 0;
    }
    *count = 0;
    return;
  }
  const float node_degree = degree[row];
  bool eligible = node_degree > 0.0f;
  if (refinement && community_count[old] != 1) eligible = false;
  if (refinement) {
    const float parent_total = parent_volume[parent[row]];
    const float threshold = resolution * node_degree *
      (parent_total - node_degree) / graph_volume;
    if (cuts[old] + tolerance < threshold) eligible = false;
  }
  const float old_inside = mark[old] ? total[old] : 0.0f;
  const float old_volume = fmaxf(0.0f, volume[old] - node_degree);
  const float old_score = old_inside - resolution * node_degree *
    old_volume / graph_volume;
  float best_score = refinement ? -INFINITY : old_score;
  int best = old;
  float best_weight = 0.0f;
  if (eligible) {
    for (int index = 0; index < *count; ++index) {
      const int candidate = touched[index];
      if (candidate == old || community_count[candidate] <= 0 ||
          (refinement && vertex_color(candidate, seed) >= color)) {
        continue;
      }
      const float candidate_volume = volume[candidate];
      if (refinement) {
        const float parent_total = parent_volume[parent[row]];
        const float threshold = resolution * candidate_volume *
          (parent_total - candidate_volume) / graph_volume;
        if (cuts[candidate] + tolerance < threshold) continue;
      }
      const float score = total[candidate] - resolution * node_degree *
        candidate_volume / graph_volume;
      if (refinement && score < -tolerance) continue;
      if (score > best_score + tolerance ||
          (fabsf(score - best_score) <= tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
        best_weight = total[candidate];
      }
    }
    if (best != old &&
        (refinement || best_score > old_score + tolerance)) {
      proposal[row] = best;
      if (refinement) proposal_weight[row] = best_weight;
    }
  }
  for (int index = 0; index < *count; ++index) {
    mark[touched[index]] = 0;
  }
  *count = 0;
}

struct StreamSession {
  int device_id;
  int n;
  int chunk_rows;
  int edge_capacity;
  float graph_volume;
  float resolution;
  float tolerance;
  std::uint64_t seed;
  unsigned int pass_seed = 0;
  bool leiden;
  int color = 0;
  DeviceBuffer<int> row;
  DeviceBuffer<int> col;
  DeviceBuffer<float> weight;
  DeviceBuffer<float> degree;
  DeviceBuffer<int> labels;
  DeviceBuffer<int> parent;
  DeviceBuffer<float> volume;
  DeviceBuffer<int> count;
  DeviceBuffer<int> proposal;
  DeviceBuffer<unsigned int> moves;
  DeviceBuffer<float> cuts;
  DeviceBuffer<float> parent_volume;
  DeviceBuffer<float> proposal_weight;
  std::unique_ptr<DeviceBuffer<int>> fragment_mark;
  std::unique_ptr<DeviceBuffer<int>> fragment_touched;
  std::unique_ptr<DeviceBuffer<float>> fragment_total;
  std::unique_ptr<DeviceBuffer<int>> fragment_count;
  int fragment_row = -1;

  StreamSession(int device, int vertices, int rows, int edges, float total,
      float gamma, std::uint64_t run_seed, bool refinement)
      : device_id(device), n(vertices), chunk_rows(rows),
        edge_capacity(edges),
        graph_volume(total), resolution(gamma),
        tolerance(1e-7f * std::max(1.0f,
          total / (2.0f * vertices))), seed(run_seed),
        leiden(refinement),
        row(static_cast<std::size_t>(rows) + 1), col(edges),
        weight(edges), degree(vertices), labels(vertices),
        parent(refinement ? vertices : 1), volume(vertices),
        count(vertices), proposal(vertices), moves(1),
        cuts(refinement ? vertices : 1),
        parent_volume(refinement ? vertices : 1),
        proposal_weight(refinement ? vertices : 1) {}
};

StreamSession& stream_session(void* handle) {
  if (handle == nullptr) Rcpp::stop("CUDA graph session is missing.");
  return *static_cast<StreamSession*>(handle);
}

}  // namespace

void* graph_stream_cuda_create(const std::vector<float>& degree,
    const std::vector<int>& initial, float graph_volume,
    float resolution, std::uint64_t seed, int chunk_rows,
    int edge_capacity, const std::vector<int>* parent,
    const std::vector<float>* cuts,
    const std::vector<float>* parent_volume) {
  const bool leiden = parent != nullptr;
  if (!graph_clustering_cuda_available_impl() ||
      degree.size() != initial.size() || degree.size() < 2 ||
      degree.size() > static_cast<std::size_t>(INT_MAX) ||
      !(graph_volume > 0.0f) || !std::isfinite(graph_volume) ||
      !(resolution > 0.0f) || !std::isfinite(resolution) ||
      chunk_rows < 1 || edge_capacity < 1 ||
      (leiden && (cuts == nullptr || parent_volume == nullptr ||
        parent->size() != degree.size() ||
        cuts->size() != degree.size() ||
        parent_volume->empty()))) {
    Rcpp::stop("Invalid CUDA streaming Louvain state.");
  }
  const int n = static_cast<int>(degree.size());
  const double required = (leiden ? 9.0 : 5.0) * n * sizeof(int) +
    (static_cast<double>(chunk_rows) + 1) * sizeof(int) +
    2.0 * edge_capacity * sizeof(int) + 1024 * 1024;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
    "CUDA Louvain VRAM preflight");
  if (required > 0.65 * free_bytes) {
    Rcpp::stop("CUDA streaming Louvain state exceeds VRAM budget.");
  }
  std::vector<float> volume(n, 0.0f);
  std::vector<int> count(n, 0);
  for (int u = 0; u < n; ++u) {
    if (!std::isfinite(degree[u]) || degree[u] < 0.0f ||
        initial[u] < 0 || initial[u] >= n) {
      Rcpp::stop("Invalid CUDA streaming Louvain vertex state.");
    }
    if (leiden && (parent->at(u) < 0 ||
        parent->at(u) >= static_cast<int>(parent_volume->size()) ||
        !std::isfinite(cuts->at(u)))) {
      Rcpp::stop("Invalid CUDA streaming Leiden vertex state.");
    }
    volume[initial[u]] += degree[u];
    ++count[initial[u]];
  }
  if (leiden && (parent_volume->size() > degree.size() ||
      std::any_of(parent_volume->begin(), parent_volume->end(),
        [](float value) { return !std::isfinite(value); }))) {
    Rcpp::stop("Invalid CUDA streaming Leiden parent volume.");
  }
  int device = -1;
  check_cuda(cudaGetDevice(&device),
    "read CUDA clustering device identity");
  std::unique_ptr<StreamSession> session(new StreamSession(
    device, n, chunk_rows, edge_capacity,
    graph_volume, resolution, seed, leiden));
  check_cuda(cudaMemcpy(session->degree.data(), degree.data(),
    degree.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload CUDA Louvain degree");
  check_cuda(cudaMemcpy(session->labels.data(), initial.data(),
    initial.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload CUDA Louvain labels");
  check_cuda(cudaMemcpy(session->volume.data(), volume.data(),
    volume.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload CUDA Louvain volumes");
  check_cuda(cudaMemcpy(session->count.data(), count.data(),
    count.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload CUDA Louvain counts");
  if (leiden) {
    check_cuda(cudaMemcpy(session->parent.data(), parent->data(),
      parent->size() * sizeof(int), cudaMemcpyHostToDevice),
      "upload CUDA Leiden parent labels");
    check_cuda(cudaMemcpy(session->cuts.data(), cuts->data(),
      cuts->size() * sizeof(float), cudaMemcpyHostToDevice),
      "upload CUDA Leiden cuts");
    check_cuda(cudaMemcpy(session->parent_volume.data(),
      parent_volume->data(), parent_volume->size() * sizeof(float),
      cudaMemcpyHostToDevice), "upload CUDA Leiden parent volumes");
  }
  return session.release();
}

void graph_stream_cuda_begin_pass(void* handle, int pass) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0) {
    Rcpp::stop("CUDA clustering row fragment was not finalized.");
  }
  session.pass_seed = static_cast<unsigned int>(
    session.seed ^ (session.seed >> 32U)) ^
    (0x9e3779b9U * static_cast<unsigned int>(pass + 1));
  check_cuda(cudaMemset(session.moves.data(), 0,
    sizeof(unsigned int)), "clear CUDA streaming moves");
}

int graph_stream_cuda_device_id(void* handle) {
  return stream_session(handle).device_id;
}

void graph_stream_cuda_set_color(void* handle, int color) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0 ||
      color < 0 || color >= kColorCount) {
    Rcpp::stop("Invalid CUDA clustering color phase.");
  }
  session.color = color;
}

void graph_stream_cuda_set_state(void* handle,
    const std::vector<int>& labels,
    const std::vector<float>& volume,
    const std::vector<int>& count) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0 ||
      labels.size() != static_cast<std::size_t>(session.n) ||
      volume.size() != labels.size() || count.size() != labels.size()) {
    Rcpp::stop("Invalid coordinated CUDA clustering state size.");
  }
  for (int u = 0; u < session.n; ++u) {
    if (labels[u] < 0 || labels[u] >= session.n ||
        !std::isfinite(volume[u]) || volume[u] < 0.0f ||
        count[u] < 0) {
      Rcpp::stop("Invalid coordinated CUDA clustering state.");
    }
  }
  check_cuda(cudaMemcpy(session.labels.data(), labels.data(),
    labels.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload coordinated CUDA labels");
  check_cuda(cudaMemcpy(session.volume.data(), volume.data(),
    volume.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload coordinated CUDA volumes");
  check_cuda(cudaMemcpy(session.count.data(), count.data(),
    count.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload coordinated CUDA counts");
}

std::vector<int> graph_stream_cuda_propose_chunk(void* handle,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& weight, int first) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0 || row_ptr.size() < 2 ||
      row_ptr.size() - 1 > static_cast<std::size_t>(session.chunk_rows) ||
      first < 0 || first > session.n -
        static_cast<int>(row_ptr.size() - 1) ||
      col_idx.size() != weight.size() ||
      col_idx.size() > static_cast<std::size_t>(session.edge_capacity) ||
      row_ptr.front() != 0 ||
      row_ptr.back() != static_cast<int>(col_idx.size())) {
    Rcpp::stop("Invalid coordinated CUDA graph chunk.");
  }
  const int rows = static_cast<int>(row_ptr.size() - 1);
  check_cuda(cudaMemcpy(session.row.data(), row_ptr.data(),
    row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload coordinated CUDA row offsets");
  check_cuda(cudaMemcpy(session.col.data(), col_idx.data(),
    col_idx.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload coordinated CUDA columns");
  check_cuda(cudaMemcpy(session.weight.data(), weight.data(),
    weight.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload coordinated CUDA weights");
  check_cuda(cudaMemcpy(session.proposal.data() + first,
    session.labels.data() + first, rows * sizeof(int),
    cudaMemcpyDeviceToDevice),
    "initialize coordinated CUDA proposals");
  constexpr int threads = 256;
  const int blocks = (rows + threads - 1) / threads;
  propose_moves_kernel<<<blocks, threads>>>(
    session.row.data(), session.col.data(), session.weight.data(),
    session.degree.data(), session.labels.data(),
    session.parent.data(), session.volume.data(),
    session.count.data(), session.proposal.data(), session.n,
    first, rows, session.graph_volume, session.resolution,
    session.tolerance, session.color, session.pass_seed,
    session.leiden ? 1 : 0, session.leiden ? 1 : 0,
    session.cuts.data(), session.parent_volume.data(),
    session.proposal_weight.data());
  check_cuda(cudaGetLastError(), "coordinated CUDA proposal");
  check_cuda(cudaDeviceSynchronize(),
    "synchronize coordinated CUDA proposal");
  std::vector<int> proposals(rows);
  check_cuda(cudaMemcpy(proposals.data(),
    session.proposal.data() + first, rows * sizeof(int),
    cudaMemcpyDeviceToHost), "read coordinated CUDA proposal");
  return proposals;
}

void graph_stream_cuda_chunk(void* handle,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& weight, int first) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0) {
    Rcpp::stop("CUDA clustering row fragment was not finalized.");
  }
  if (row_ptr.size() < 2 ||
      row_ptr.size() - 1 > static_cast<std::size_t>(INT_MAX)) {
    Rcpp::stop("Invalid CUDA streaming Louvain row count.");
  }
  const int rows = static_cast<int>(row_ptr.size()) - 1;
  if (rows < 1 || rows > session.chunk_rows || first < 0 ||
      first > session.n - rows ||
      col_idx.size() != weight.size() ||
      col_idx.size() > static_cast<std::size_t>(session.edge_capacity) ||
      row_ptr.front() != 0 ||
      row_ptr.back() != static_cast<int>(col_idx.size())) {
    Rcpp::stop("Invalid CUDA streaming Louvain graph chunk.");
  }
  check_cuda(cudaMemcpy(session.row.data(), row_ptr.data(),
    row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload CUDA Louvain row offsets");
  check_cuda(cudaMemcpy(session.col.data(), col_idx.data(),
    col_idx.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload CUDA Louvain columns");
  check_cuda(cudaMemcpy(session.weight.data(), weight.data(),
    weight.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload CUDA Louvain weights");
  constexpr int threads = 256;
  const int blocks = (rows + threads - 1) / threads;
  const int first_color = session.leiden ? session.color : 0;
  const int last_color = session.leiden ? session.color + 1 : kColorCount;
  for (int color = first_color; color < last_color; ++color) {
    propose_moves_kernel<<<blocks, threads>>>(
      session.row.data(), session.col.data(), session.weight.data(),
      session.degree.data(), session.labels.data(),
      session.parent.data(), session.volume.data(),
      session.count.data(), session.proposal.data(), session.n,
      first, rows, session.graph_volume, session.resolution,
      session.tolerance, color, session.pass_seed,
      session.leiden ? 1 : 0, session.leiden ? 1 : 0,
      session.cuts.data(), session.parent_volume.data(),
      session.proposal_weight.data());
    check_cuda(cudaGetLastError(), "CUDA streaming Louvain proposal");
    apply_moves_kernel<<<blocks, threads>>>(
      session.degree.data(), session.labels.data(),
      session.proposal.data(), session.volume.data(),
      session.count.data(), session.moves.data(), session.n,
      first, rows, color, session.pass_seed,
      session.leiden ? 1 : 0, session.row.data(),
      session.col.data(), session.weight.data(),
      session.cuts.data(), session.proposal_weight.data());
    check_cuda(cudaGetLastError(), "CUDA streaming Louvain apply");
  }
  check_cuda(cudaDeviceSynchronize(),
    "synchronize CUDA streaming Louvain chunk");
}

int graph_stream_cuda_fragment_impl(void* handle,
    const std::vector<int>& col_idx,
    const std::vector<float>& weight, int row, bool last,
    bool apply) {
  auto& session = stream_session(handle);
  if (row < 0 || row >= session.n || col_idx.empty() ||
      col_idx.size() != weight.size() ||
      col_idx.size() > static_cast<std::size_t>(session.edge_capacity) ||
      (session.fragment_row >= 0 && session.fragment_row != row)) {
    Rcpp::stop("Invalid CUDA clustering row fragment.");
  }
  if (!session.fragment_mark) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
      "CUDA clustering fragment memory check");
    const double required = 12.0 * session.n + sizeof(int);
    if (required > 0.65 * free_bytes) {
      Rcpp::stop("CUDA clustering fragment state exceeds VRAM budget.");
    }
    session.fragment_mark.reset(new DeviceBuffer<int>(session.n));
    session.fragment_touched.reset(new DeviceBuffer<int>(session.n));
    session.fragment_total.reset(new DeviceBuffer<float>(session.n));
    session.fragment_count.reset(new DeviceBuffer<int>(1));
    check_cuda(cudaMemset(session.fragment_mark->data(), 0,
      session.n * sizeof(int)), "clear CUDA fragment markers");
    check_cuda(cudaMemset(session.fragment_count->data(), 0,
      sizeof(int)), "clear CUDA fragment count");
  }
  session.fragment_row = row;
  check_cuda(cudaMemcpy(session.col.data(), col_idx.data(),
    col_idx.size() * sizeof(int), cudaMemcpyHostToDevice),
    "upload CUDA clustering fragment IDs");
  check_cuda(cudaMemcpy(session.weight.data(), weight.data(),
    weight.size() * sizeof(float), cudaMemcpyHostToDevice),
    "upload CUDA clustering fragment weights");
  accumulate_fragment_kernel<<<1, 1>>>(session.col.data(),
    session.weight.data(), static_cast<int>(col_idx.size()), row,
    session.labels.data(), session.parent.data(), session.leiden,
    session.fragment_mark->data(), session.fragment_touched->data(),
    session.fragment_total->data(), session.fragment_count->data());
  check_cuda(cudaGetLastError(), "CUDA clustering fragment accumulation");
  if (last) {
    finalize_fragment_kernel<<<1, 1>>>(row, session.degree.data(),
      session.labels.data(), session.parent.data(),
      session.volume.data(), session.count.data(),
      session.proposal.data(), session.graph_volume,
      session.resolution, session.tolerance, session.color,
      session.pass_seed, session.leiden, apply ? 0 : 1,
      session.cuts.data(),
      session.parent_volume.data(), session.proposal_weight.data(),
      session.fragment_mark->data(), session.fragment_touched->data(),
      session.fragment_total->data(), session.fragment_count->data());
    check_cuda(cudaGetLastError(), "CUDA clustering fragment proposal");
    if (apply) {
      check_cuda(cudaMemset(session.row.data(), 0, 2 * sizeof(int)),
        "clear CUDA clustering one-row offsets");
      const int first_color = session.leiden ? session.color : 0;
      const int last_color = session.leiden ? session.color + 1 :
        kColorCount;
      for (int color = first_color; color < last_color; ++color) {
        apply_moves_kernel<<<1, 1>>>(session.degree.data(),
          session.labels.data(), session.proposal.data(),
          session.volume.data(), session.count.data(),
          session.moves.data(), session.n, row, 1, color,
          session.pass_seed, session.leiden, session.row.data(),
          session.col.data(), session.weight.data(),
          session.cuts.data(), session.proposal_weight.data());
        check_cuda(cudaGetLastError(),
          "CUDA clustering fragment apply");
      }
    }
    session.fragment_row = -1;
  }
  check_cuda(cudaDeviceSynchronize(),
    "synchronize CUDA clustering row fragment");
  if (!last || apply) return -1;
  int proposal = -1;
  check_cuda(cudaMemcpy(&proposal, session.proposal.data() + row,
    sizeof(int), cudaMemcpyDeviceToHost),
    "read coordinated CUDA fragment proposal");
  return proposal;
}

void graph_stream_cuda_fragment(void* handle,
    const std::vector<int>& col_idx,
    const std::vector<float>& weight, int row, bool last) {
  graph_stream_cuda_fragment_impl(handle, col_idx,
    weight, row, last, true);
}

int graph_stream_cuda_propose_fragment(void* handle,
    const std::vector<int>& col_idx,
    const std::vector<float>& weight, int row, bool last) {
  return graph_stream_cuda_fragment_impl(handle, col_idx,
    weight, row, last, false);
}

std::uint32_t graph_stream_cuda_end_pass(void* handle) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0) {
    Rcpp::stop("CUDA clustering row fragment was not finalized.");
  }
  std::uint32_t moves = 0;
  check_cuda(cudaMemcpy(&moves, session.moves.data(), sizeof(moves),
    cudaMemcpyDeviceToHost), "read CUDA streaming moves");
  return moves;
}

std::vector<int> graph_stream_cuda_labels(void* handle) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0) {
    Rcpp::stop("CUDA clustering row fragment was not finalized.");
  }
  std::vector<int> labels(session.n);
  check_cuda(cudaMemcpy(labels.data(), session.labels.data(),
    labels.size() * sizeof(int), cudaMemcpyDeviceToHost),
    "read CUDA streaming labels");
  return labels;
}

std::vector<int> graph_stream_cuda_counts(void* handle) {
  auto& session = stream_session(handle);
  if (session.fragment_row >= 0) {
    Rcpp::stop("CUDA clustering row fragment was not finalized.");
  }
  std::vector<int> count(session.n);
  check_cuda(cudaMemcpy(count.data(), session.count.data(),
    count.size() * sizeof(int), cudaMemcpyDeviceToHost),
    "read CUDA streaming counts");
  return count;
}

void graph_stream_cuda_destroy(void* handle) {
  delete static_cast<StreamSession*>(handle);
}

bool graph_clustering_cuda_available_impl() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename Offset>
fastembedr_graph_accel::MoveResult graph_local_move_cuda_typed_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    bool refinement) {
  using fastembedr_graph_accel::MoveResult;
  if (!graph_clustering_cuda_available_impl()) {
    Rcpp::stop("No CUDA device is available for graph clustering.");
  }
  if (initial.size() != static_cast<std::size_t>(graph.n)) {
    Rcpp::stop("Internal CUDA clustering membership size mismatch.");
  }
  const std::vector<Offset>& host_row = [&]() ->
      const std::vector<Offset>& {
    if constexpr (std::is_same<Offset, int>::value)
      return graph.row_ptr;
    else return graph.row_ptr_wide;
  }();
  if (refinement &&
      (parent == nullptr || parent->size() != initial.size())) {
    Rcpp::stop("Internal CUDA Leiden parent-partition size mismatch.");
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

  DeviceBuffer<Offset> d_row_ptr(host_row.size());
  DeviceBuffer<int> d_col_idx(graph.col_idx.size());
  DeviceBuffer<float> d_weight(graph.weight.size());
  DeviceBuffer<float> d_degree(graph.degree.size());
  DeviceBuffer<int> d_labels(initial.size());
  DeviceBuffer<int> d_parent(parent->size());
  DeviceBuffer<float> d_volume(volume.size());
  DeviceBuffer<int> d_count(count.size());
  DeviceBuffer<int> d_proposal(initial.size());
  DeviceBuffer<unsigned int> d_moves(1);

  check_cuda(cudaMemcpy(
    d_row_ptr.data(), host_row.data(),
    host_row.size() * sizeof(Offset), cudaMemcpyHostToDevice
  ), "copy CSR row offsets");
  check_cuda(cudaMemcpy(
    d_col_idx.data(), graph.col_idx.data(),
    graph.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy CSR column indices");
  check_cuda(cudaMemcpy(
    d_weight.data(), graph.weight.data(),
    graph.weight.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy CSR weights");
  check_cuda(cudaMemcpy(
    d_degree.data(), graph.degree.data(),
    graph.degree.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy graph degrees");
  check_cuda(cudaMemcpy(
    d_labels.data(), initial.data(),
    initial.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy graph labels");
  check_cuda(cudaMemcpy(
    d_parent.data(), parent->data(),
    parent->size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy Leiden parent labels");
  check_cuda(cudaMemcpy(
    d_volume.data(), volume.data(),
    volume.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy community volumes");
  check_cuda(cudaMemcpy(
    d_count.data(), count.data(),
    count.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy community counts");

  constexpr int threads = 256;
  const int blocks = (graph.n + threads - 1) / threads;
  const float tolerance = 1e-7f * std::max(
    1.0f,
    static_cast<float>(graph.total_edge_weight) /
      static_cast<float>(std::max(1, graph.n))
  );
  const unsigned int base_device_seed =
    static_cast<unsigned int>(seed ^ (seed >> 32U));

  for (int pass = 0; pass < max_passes; ++pass) {
    const unsigned int device_seed = base_device_seed ^
      (0x9e3779b9U * static_cast<unsigned int>(pass + 1));
    check_cuda(cudaMemset(d_moves.data(), 0, sizeof(unsigned int)),
               "clear CUDA graph move counter");
    for (int color = 0; color < kColorCount; ++color) {
      propose_moves_kernel<<<blocks, threads>>>(
        d_row_ptr.data(), d_col_idx.data(), d_weight.data(), d_degree.data(),
        d_labels.data(), d_parent.data(), d_volume.data(), d_count.data(),
        d_proposal.data(), graph.n, 0, graph.n, graph.volume,
        resolution, tolerance,
        color, device_seed, refinement ? 1 : 0, 0,
        nullptr, nullptr, nullptr
      );
      check_cuda(cudaGetLastError(), "launch CUDA graph proposal kernel");
      apply_moves_kernel<Offset><<<blocks, threads>>>(
        d_degree.data(), d_labels.data(), d_proposal.data(), d_volume.data(),
        d_count.data(), d_moves.data(), graph.n, 0, graph.n,
        color, device_seed, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr
      );
      check_cuda(cudaGetLastError(), "launch CUDA graph apply kernel");
    }
    unsigned int moves = 0;
    check_cuda(cudaMemcpy(
      &moves, d_moves.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost
    ), "read CUDA graph move counter");
    result.moves += static_cast<int>(moves);
    ++result.passes;
    if (moves == 0U) break;
  }

  check_cuda(cudaMemcpy(
    result.membership.data(), d_labels.data(),
    result.membership.size() * sizeof(int), cudaMemcpyDeviceToHost
  ), "copy CUDA graph membership");
  return result;
}

fastembedr_graph_accel::MoveResult graph_local_move_cuda_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent, int max_passes,
    float resolution, std::uint64_t seed, bool refinement) {
  if (graph.wide_offsets())
    return graph_local_move_cuda_typed_impl<std::uint64_t>(
      graph, initial, parent, max_passes, resolution, seed, refinement);
  return graph_local_move_cuda_typed_impl<int>(
    graph, initial, parent, max_passes, resolution, seed, refinement);
}

namespace {

struct CoordinatedDevice {
  int id;
  int first;
  int rows;
  std::uint64_t edges;
  bool wide;
  DeviceBuffer<int> row;
  DeviceBuffer<std::uint64_t> row_wide;
  DeviceBuffer<int> col;
  DeviceBuffer<float> weight;
  DeviceBuffer<float> degree;
  DeviceBuffer<int> labels;
  DeviceBuffer<int> parent;
  DeviceBuffer<float> volume;
  DeviceBuffer<int> count;
  DeviceBuffer<int> proposal;
  DeviceBuffer<unsigned int> moves;

  CoordinatedDevice(int device, const fastembedr_graph_accel::Graph& graph,
      int begin, int end, const std::vector<int>& parents)
      : id(device), first(begin), rows(end - begin),
        edges(graph.offset(end) - graph.offset(begin)),
        wide(edges > static_cast<std::uint64_t>(INT_MAX)),
        row(wide ? 0 : static_cast<std::size_t>(rows) + 1),
        row_wide(wide ? static_cast<std::size_t>(rows) + 1 : 0),
        col(static_cast<std::size_t>(edges)),
        weight(edges), degree(graph.n), labels(graph.n),
        parent(graph.n), volume(graph.n), count(graph.n),
        proposal(graph.n), moves(1) {
    const auto offset = graph.offset(begin);
    if (wide) {
      std::vector<std::uint64_t> local_row(
        static_cast<std::size_t>(rows) + 1);
      for (int i = 0; i <= rows; ++i)
        local_row[static_cast<std::size_t>(i)] =
          graph.offset(begin + i) - offset;
      check_cuda(cudaMemcpy(row_wide.data(), local_row.data(),
        local_row.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
        "upload coordinated wide CSR rows");
    } else {
      std::vector<int> local_row(static_cast<std::size_t>(rows) + 1);
      for (int i = 0; i <= rows; ++i)
        local_row[static_cast<std::size_t>(i)] =
          static_cast<int>(graph.offset(begin + i) - offset);
      check_cuda(cudaMemcpy(row.data(), local_row.data(),
        local_row.size() * sizeof(int), cudaMemcpyHostToDevice),
        "upload coordinated CSR rows");
    }
    if (edges > 0) {
      check_cuda(cudaMemcpy(col.data(), graph.col_idx.data() + offset,
        static_cast<std::size_t>(edges) * sizeof(int),
        cudaMemcpyHostToDevice), "upload coordinated CSR columns");
      check_cuda(cudaMemcpy(weight.data(), graph.weight.data() + offset,
        static_cast<std::size_t>(edges) * sizeof(float),
        cudaMemcpyHostToDevice), "upload coordinated CSR weights");
    }
    check_cuda(cudaMemcpy(degree.data(), graph.degree.data(),
      graph.degree.size() * sizeof(float), cudaMemcpyHostToDevice),
      "upload coordinated graph degrees");
    check_cuda(cudaMemcpy(parent.data(), parents.data(),
      parents.size() * sizeof(int), cudaMemcpyHostToDevice),
      "upload coordinated Leiden parents");
  }

  ~CoordinatedDevice() { cudaSetDevice(id); }
};

void coordinated_totals(const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& labels, std::vector<float>& volume,
    std::vector<int>& count) {
  std::fill(volume.begin(), volume.end(), 0.0f);
  std::fill(count.begin(), count.end(), 0);
  for (int u = 0; u < graph.n; ++u) {
    const int label = labels[static_cast<std::size_t>(u)];
    if (label < 0 || label >= graph.n) {
      Rcpp::stop("Invalid coordinated CUDA graph label.");
    }
    volume[static_cast<std::size_t>(label)] +=
      graph.degree[static_cast<std::size_t>(u)];
    ++count[static_cast<std::size_t>(label)];
  }
}

}  // namespace

fastembedr_graph_accel::MoveResult graph_local_move_cuda_devices_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    bool refinement,
    const std::vector<int>& devices) {
  using fastembedr_graph_accel::MoveResult;
  if (devices.size() < 1 ||
      devices.size() > static_cast<std::size_t>(graph.n) ||
      initial.size() != static_cast<std::size_t>(graph.n) ||
      (refinement &&
       (parent == nullptr || parent->size() != initial.size()))) {
    Rcpp::stop("Invalid coordinated CUDA clustering state.");
  }
  MoveResult result;
  result.membership = initial;
  if (graph.n < 2 || !(graph.volume > 0.0f)) return result;
  using Clock = std::chrono::steady_clock;
  const auto setup_started = Clock::now();
  std::vector<int> empty_parent;
  if (parent == nullptr) {
    empty_parent.assign(static_cast<std::size_t>(graph.n), 0);
    parent = &empty_parent;
  }
  std::vector<std::unique_ptr<CoordinatedDevice>> workers;
  workers.reserve(devices.size());
  const int original_device = []() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "read original CUDA device");
    return device;
  }();
  try {
    for (std::size_t i = 0; i < devices.size(); ++i) {
      const int begin = static_cast<int>(
        static_cast<std::int64_t>(graph.n) * i / devices.size());
      const int end = static_cast<int>(
        static_cast<std::int64_t>(graph.n) * (i + 1) / devices.size());
      check_cuda(cudaSetDevice(devices[i]),
        "select coordinated CUDA device");
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
        "coordinated graph VRAM preflight");
      const double required = 28.0 * graph.n +
        8.0 * (graph.offset(end) - graph.offset(begin)) +
        (graph.offset(end) - graph.offset(begin) > INT_MAX ? 8.0 : 4.0) *
          (end - begin + 1);
      if (required > 0.7 * static_cast<double>(free_bytes)) {
        Rcpp::stop("Coordinated graph exceeds a CUDA device VRAM budget.");
      }
      workers.emplace_back(new CoordinatedDevice(
        devices[i], graph, begin, end, *parent));
    }
    result.setup_seconds = std::chrono::duration<double>(
      Clock::now() - setup_started).count();
    std::vector<float> volume(static_cast<std::size_t>(graph.n));
    std::vector<int> count(static_cast<std::size_t>(graph.n));
    const float tolerance = 1e-7f * std::max(
      1.0f, static_cast<float>(graph.total_edge_weight) /
        static_cast<float>(graph.n));
    const unsigned int base_seed =
      static_cast<unsigned int>(seed ^ (seed >> 32U));
    for (int pass = 0; pass < max_passes; ++pass) {
      const std::vector<int> previous = result.membership;
      const auto previous_quality_started = Clock::now();
      const double previous_quality =
        fastembedr_graph_accel::modularity(graph, previous, resolution);
      result.quality_seconds += std::chrono::duration<double>(
        Clock::now() - previous_quality_started).count();
      const unsigned int pass_seed = base_seed ^
        (0x9e3779b9U * static_cast<unsigned int>(pass + 1));
      unsigned int pass_moves = 0;
      for (int color = 0; color < kColorCount; ++color) {
        const auto totals_started = Clock::now();
        coordinated_totals(graph, result.membership, volume, count);
        result.totals_seconds += std::chrono::duration<double>(
          Clock::now() - totals_started).count();
        const auto broadcast_started = Clock::now();
        for (const auto& worker : workers) {
          check_cuda(cudaSetDevice(worker->id),
            "select coordinated CUDA worker");
          check_cuda(cudaMemcpy(worker->labels.data(),
            result.membership.data(),
            result.membership.size() * sizeof(int),
            cudaMemcpyHostToDevice), "broadcast coordinated labels");
          check_cuda(cudaMemcpy(worker->volume.data(), volume.data(),
            volume.size() * sizeof(float), cudaMemcpyHostToDevice),
            "broadcast coordinated volumes");
          check_cuda(cudaMemcpy(worker->count.data(), count.data(),
            count.size() * sizeof(int), cudaMemcpyHostToDevice),
            "broadcast coordinated counts");
          check_cuda(cudaMemset(worker->moves.data(), 0,
            sizeof(unsigned int)), "clear coordinated move count");
          constexpr int threads = 256;
          const int blocks = (worker->rows + threads - 1) / threads;
          if (blocks == 0) continue;
          const auto propose = [&](auto* row_ptr) {
            propose_moves_kernel<<<blocks, threads>>>(
            row_ptr, worker->col.data(),
            worker->weight.data(), worker->degree.data(),
            worker->labels.data(), worker->parent.data(),
            worker->volume.data(), worker->count.data(),
            worker->proposal.data(), graph.n, worker->first,
            worker->rows, graph.volume, resolution, tolerance,
            color, pass_seed, refinement ? 1 : 0, 0,
            nullptr, nullptr, nullptr);
          };
          if (worker->wide) propose(worker->row_wide.data());
          else propose(worker->row.data());
          check_cuda(cudaGetLastError(), "coordinated CUDA proposal");
          const auto apply = [&](auto* row_ptr) {
            apply_moves_kernel<<<blocks, threads>>>(
            worker->degree.data(), worker->labels.data(),
            worker->proposal.data(), worker->volume.data(),
            worker->count.data(), worker->moves.data(),
            graph.n, worker->first, worker->rows, color,
            pass_seed, 0, row_ptr, nullptr, nullptr,
            nullptr, nullptr);
          };
          if (worker->wide) apply(worker->row_wide.data());
          else apply(worker->row.data());
          check_cuda(cudaGetLastError(), "coordinated CUDA apply");
        }
        result.broadcast_launch_seconds += std::chrono::duration<double>(
          Clock::now() - broadcast_started).count();
        const auto merge_started = Clock::now();
        for (const auto& worker : workers) {
          check_cuda(cudaSetDevice(worker->id),
            "select coordinated CUDA worker");
          check_cuda(cudaDeviceSynchronize(),
            "synchronize coordinated CUDA phase");
          if (worker->rows == 0) continue;
          check_cuda(cudaMemcpy(
            result.membership.data() + worker->first,
            worker->labels.data() + worker->first,
            static_cast<std::size_t>(worker->rows) * sizeof(int),
            cudaMemcpyDeviceToHost), "merge coordinated labels");
          unsigned int moves = 0;
          check_cuda(cudaMemcpy(&moves, worker->moves.data(),
            sizeof(moves), cudaMemcpyDeviceToHost),
            "read coordinated moves");
          pass_moves += moves;
        }
        result.sync_merge_seconds += std::chrono::duration<double>(
          Clock::now() - merge_started).count();
      }
      ++result.passes;
      const auto next_quality_started = Clock::now();
      const double next_quality = fastembedr_graph_accel::modularity(
        graph, result.membership, resolution);
      result.quality_seconds += std::chrono::duration<double>(
        Clock::now() - next_quality_started).count();
      if (!std::isfinite(next_quality) ||
          next_quality + 1e-9 < previous_quality) {
        result.membership = previous;
        break;
      }
      result.moves += static_cast<int>(pass_moves);
      if (pass_moves == 0U) break;
    }
  } catch (...) {
    for (auto& worker : workers) worker.reset();
    cudaSetDevice(original_device);
    throw;
  }
  const auto teardown_started = Clock::now();
  for (auto& worker : workers) worker.reset();
  check_cuda(cudaSetDevice(original_device),
    "restore original CUDA device");
  result.teardown_seconds = std::chrono::duration<double>(
    Clock::now() - teardown_started).count();
  return result;
}
