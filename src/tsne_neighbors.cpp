/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Package-owned sparse-affinity and openTSNE-style optimization code. The
 * implementation follows published t-SNE/FIt-SNE mathematics and documented
 * openTSNE workflow behavior; no openTSNE, FIt-SNE, t-SNE-CUDA, or Rtsne
 * source is copied, translated, linked, or vendored.
 */

#include <Rcpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

struct SparseProbabilitiesF {
  std::vector<int> row_ptr;
  std::vector<int> col;
  std::vector<float> val;
};

struct PackedEdgeF {
  std::uint64_t key;
  float value;
};

std::uint64_t pair_key(int a, int b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32u) |
    static_cast<std::uint32_t>(b);
}

int key_first(std::uint64_t key) {
  return static_cast<int>(key >> 32u);
}

int key_second(std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffu);
}

std::uint32_t mix_uint32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value == 0u ? 0x6d2b79f5u : value;
}

std::uint32_t xorshift32(std::uint32_t& state) {
  state ^= state << 13u;
  state ^= state >> 17u;
  state ^= state << 5u;
  return state;
}

int uniform_index(std::uint32_t& state, const int n) {
  return static_cast<int>(
    (static_cast<std::uint64_t>(xorshift32(state)) *
      static_cast<std::uint64_t>(n)) >> 32u
  );
}

int resolve_index_offset(const IntegerMatrix& indices) {
  const int n = indices.nrow();
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < indices.nrow(); ++i) {
    for (int j = 0; j < indices.ncol(); ++j) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  return (min_idx >= 1 && max_idx <= n) ? 1 : 0;
}

int resolve_threads(int n_threads, int n) {
  if (n_threads == NA_INTEGER) n_threads = 1;
  if (n_threads < 0) Rcpp::stop("`n_threads` must be non-negative.");
  if (n_threads == 0) {
    const unsigned int hw = std::thread::hardware_concurrency();
    n_threads = hw == 0u ? 1 : static_cast<int>(hw);
  }
  return std::max(1, std::min(n_threads, std::max(1, n)));
}

class ParallelExecutor {
 public:
  explicit ParallelExecutor(const int max_threads) :
    max_threads_(std::max(1, max_threads)) {
    workers_.reserve(static_cast<std::size_t>(max_threads_ - 1));
    for (int thread_id = 1; thread_id < max_threads_; ++thread_id) {
      workers_.emplace_back([this, thread_id]() { worker_loop(thread_id); });
    }
  }

  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;

  ~ParallelExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  int max_threads() const {
    return max_threads_;
  }

  template <typename Function>
  void run(const int n, const int requested_threads, Function& fn) {
    const int active_threads = std::max(
      1,
      std::min(std::min(requested_threads, max_threads_), std::max(1, n))
    );
    if (active_threads <= 1 || n < 2) {
      fn(0, n, 0);
      return;
    }

    const int chunk = (n + active_threads - 1) / active_threads;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = [&fn](const int begin, const int end, const int thread_id) {
        fn(begin, end, thread_id);
      };
      n_ = n;
      active_threads_ = active_threads;
      chunk_ = chunk;
      completed_workers_ = 0;
      worker_error_ = nullptr;
      ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr main_error;
    try {
      fn(0, std::min(n, chunk), 0);
    } catch (...) {
      main_error = std::current_exception();
    }

    std::exception_ptr worker_error;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this]() {
        return completed_workers_ == static_cast<int>(workers_.size());
      });
      worker_error = worker_error_;
      task_ = nullptr;
    }
    if (main_error != nullptr) std::rethrow_exception(main_error);
    if (worker_error != nullptr) std::rethrow_exception(worker_error);
  }

 private:
  void worker_loop(const int thread_id) {
    std::uint64_t observed_generation = 0;
    while (true) {
      std::function<void(int, int, int)> task;
      int begin = 0;
      int end = 0;
      bool active = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [this, observed_generation]() {
          return stop_ || generation_ != observed_generation;
        });
        if (stop_) return;
        observed_generation = generation_;
        task = task_;
        active = thread_id < active_threads_;
        if (active) {
          begin = thread_id * chunk_;
          end = std::min(n_, begin + chunk_);
          active = begin < end;
        }
      }

      if (active) {
        try {
          task(begin, end, thread_id);
        } catch (...) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (worker_error_ == nullptr) worker_error_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_workers_;
        if (completed_workers_ == static_cast<int>(workers_.size())) {
          done_cv_.notify_one();
        }
      }
    }
  }

  int max_threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::function<void(int, int, int)> task_;
  std::uint64_t generation_ = 0;
  int n_ = 0;
  int active_threads_ = 1;
  int chunk_ = 0;
  int completed_workers_ = 0;
  std::exception_ptr worker_error_;
  bool stop_ = false;
};

thread_local ParallelExecutor* active_parallel_executor = nullptr;

class ParallelExecutorScope {
 public:
  explicit ParallelExecutorScope(ParallelExecutor* executor) :
    previous_(active_parallel_executor) {
    active_parallel_executor = executor;
  }

  ~ParallelExecutorScope() {
    active_parallel_executor = previous_;
  }

 private:
  ParallelExecutor* previous_;
};

template <typename Function>
void parallel_for(const int n, const int n_threads, Function fn) {
  if (n_threads <= 1 || n < 2) {
    fn(0, n, 0);
    return;
  }
  if (active_parallel_executor != nullptr &&
      n_threads <= active_parallel_executor->max_threads()) {
    active_parallel_executor->run(n, n_threads, fn);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(n_threads - 1));
  const int chunk = (n + n_threads - 1) / n_threads;
  for (int t = 1; t < n_threads; ++t) {
    const int begin = t * chunk;
    const int end = std::min(n, begin + chunk);
    if (begin < end) {
      workers.emplace_back([=, &fn]() { fn(begin, end, t); });
    }
  }
  fn(0, std::min(n, chunk), 0);
  for (auto& worker : workers) worker.join();
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

int env_positive_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 0L || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

bool is_float32_s4(SEXP x) {
  if (!Rf_isS4(x)) return false;
  Rcpp::S4 obj(x);
  return obj.is("float32");
}

IntegerMatrix float32_data_slot(SEXP x) {
  Rcpp::S4 obj(x);
  IntegerMatrix payload = obj.slot("Data");
  if (payload.nrow() < 1 || payload.ncol() < 1) {
    Rcpp::stop("float32 distance object has an invalid payload");
  }
  return payload;
}

float int_bits_to_float(const int value) {
  float out = 0.0f;
  static_assert(sizeof(out) == sizeof(value), "float32 payload must use 32-bit storage");
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::pair<int, int> distance_sexp_dims(SEXP distances) {
  if (is_float32_s4(distances)) {
    IntegerMatrix payload = float32_data_slot(distances);
    return std::make_pair(payload.nrow(), payload.ncol());
  }
  if (Rf_isMatrix(distances) && Rf_isReal(distances)) {
    NumericMatrix mat(distances);
    return std::make_pair(mat.nrow(), mat.ncol());
  }
  Rcpp::stop("KNN distances must be a numeric matrix or float::float32 matrix");
}

std::vector<float> copy_distances_float_sexp(SEXP distances, const int n_threads) {
  const std::pair<int, int> dims = distance_sexp_dims(distances);
  const int n = dims.first;
  const int k = dims.second;
  std::vector<float> out(static_cast<std::size_t>(n) * k, 0.0f);
  if (is_float32_s4(distances)) {
    IntegerMatrix payload = float32_data_slot(distances);
    parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        for (int j = 0; j < k; ++j) {
          out[static_cast<std::size_t>(i) * k + j] = int_bits_to_float(payload(i, j));
        }
      }
    });
    return out;
  }
  NumericMatrix mat(distances);
  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      for (int j = 0; j < k; ++j) {
        out[static_cast<std::size_t>(i) * k + j] = static_cast<float>(mat(i, j));
      }
    }
  });
  return out;
}

std::string tsne_repulsion_mode(const int n,
                                const double theta,
                                const std::string& requested_method) {
  const std::string requested = lowercase(requested_method);
  if (requested == "bh" || requested == "barnes_hut" || requested == "barnes-hut") {
    Rcpp::stop(
      "Barnes-Hut t-SNE has been removed from fastEmbedR. "
      "Use `negative_gradient_method = \"fft\"` for the standard CPU path "
      "or `\"exact\"` for small reference runs."
    );
  }
  if (requested == "exact" || requested == "pair" || requested == "pair_symmetric") {
    return "pair_symmetric";
  }
  if (requested == "fft" || requested == "interpolation" || requested == "fitsne") {
    return "fft_grid";
  }

  const char* raw = std::getenv("FASTEMBEDR_TSNE_REPULSION");
  if (raw != nullptr && raw[0] != '\0') {
    const std::string value = lowercase(std::string(raw));
    if (value == "barnes_hut" || value == "barnes-hut" || value == "bh" ||
        value == "rtsne") {
      Rcpp::stop(
        "FASTEMBEDR_TSNE_REPULSION requests Barnes-Hut, which has been "
        "removed from fastEmbedR. Use `fft` or `exact`."
      );
    }
    if (value == "pair" || value == "pair_symmetric" || value == "legacy" ||
        value == "exact") {
      return "pair_symmetric";
    }
  }

  (void)n;
  return theta > 0.0 ? "fft_grid" : "pair_symmetric";
}

int tsne_fft_grid_size(const int n) {
  // The automatic route uses exact repulsion for n <= 3000. When FFT is
  // requested explicitly on a small problem, 64 cells can satisfy force-level
  // tolerances yet converge to a measurably worse long-run KL objective. A
  // minimum of 128 passed the matched common-affinity production trajectory.
  const int fallback = n >= 50000 ? 256 : 128;
  const int requested = env_positive_int("FASTEMBEDR_TSNE_FFT_GRID", fallback);
  int grid = 32;
  while (grid < requested && grid < 512) grid <<= 1;
  return std::max(32, std::min(512, grid));
}

int tsne_transform_batch_size(const int n_query,
                              const int k,
                              const int dims) {
  const int requested = env_positive_int("FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE", 0);
  if (requested > 0) return std::max(1, std::min(n_query, requested));
  if (n_query < 25000) return n_query;

  const double bytes_per_row =
    static_cast<double>(k) * sizeof(double) +
    static_cast<double>(dims) * 4.0 * sizeof(double);
  const double target_bytes = 64.0 * 1024.0 * 1024.0;
  const int auto_batch = static_cast<int>(std::floor(target_bytes / std::max(1.0, bytes_per_row)));
  return std::max(1024, std::min(n_query, std::max(1, auto_batch)));
}

struct TsneTransformGradientWorkspace {
  int n_threads = 0;
  int q_size = 0;
  int n_negatives = 0;
  std::vector<std::vector<int>> sampled;
  std::vector<std::vector<double>> q_values;

  void ensure(const int requested_threads,
              const int requested_q_size,
              const int requested_n_negatives) {
    n_threads = std::max(1, requested_threads);
    q_size = std::max(1, requested_q_size);
    n_negatives = std::max(1, requested_n_negatives);
    sampled.resize(static_cast<std::size_t>(n_threads));
    q_values.resize(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      sampled[static_cast<std::size_t>(t)].resize(static_cast<std::size_t>(n_negatives));
      q_values[static_cast<std::size_t>(t)].resize(static_cast<std::size_t>(q_size));
    }
  }
};

void compute_row_probabilities_flat(const NumericMatrix& distances,
                                    const int row,
                                    const double perplexity,
                                    double* row_p) {
  const int k = distances.ncol();
  std::fill(row_p, row_p + k, 0.0);

  double min_d2 = DBL_MAX;
  double max_d2 = 0.0;
  for (int j = 0; j < k; ++j) {
    const double d2 = distances(row, j) * distances(row, j);
    min_d2 = std::min(min_d2, d2);
    max_d2 = std::max(max_d2, d2);
  }
  const double spread = max_d2 - min_d2;
  if (spread <= DBL_EPSILON * std::max(1.0, max_d2)) {
    const double uniform = 1.0 / static_cast<double>(k);
    std::fill(row_p, row_p + k, uniform);
    return;
  }

  bool found = false;
  double beta = 1.0;
  double min_beta = -DBL_MAX;
  double max_beta = DBL_MAX;
  const double tol = 1e-5;
  double sum_p = DBL_MIN;

  for (int iter = 0; !found && iter < 200; ++iter) {
    sum_p = DBL_MIN;
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      const double d2 = d * d - min_d2;
      const double p = std::exp(-beta * d2);
      row_p[j] = p;
      sum_p += p;
    }

    double entropy = 0.0;
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      entropy += beta * ((d * d - min_d2) * row_p[j]);
    }
    entropy = entropy / sum_p + std::log(sum_p);
    const double diff = entropy - std::log(perplexity);

    if (std::abs(diff) < tol) {
      found = true;
    } else if (diff > 0.0) {
      min_beta = beta;
      beta = (max_beta == DBL_MAX || max_beta == -DBL_MAX) ?
        beta * 2.0 :
        (beta + max_beta) / 2.0;
    } else {
      max_beta = beta;
      beta = (min_beta == -DBL_MAX || min_beta == DBL_MAX) ?
        beta / 2.0 :
        (beta + min_beta) / 2.0;
    }
    if (!std::isfinite(beta)) break;
  }

  if (!std::isfinite(sum_p) || sum_p <= DBL_MIN) {
    int tied = 0;
    for (int j = 0; j < k; ++j) {
      const double d2 = distances(row, j) * distances(row, j);
      if (std::abs(d2 - min_d2) <= DBL_EPSILON * std::max(1.0, min_d2)) ++tied;
    }
    const double tied_mass = 1.0 / static_cast<double>(std::max(1, tied));
    for (int j = 0; j < k; ++j) {
      const double d2 = distances(row, j) * distances(row, j);
      row_p[j] = std::abs(d2 - min_d2) <= DBL_EPSILON * std::max(1.0, min_d2) ?
        tied_mass : 0.0;
    }
    return;
  }
  const double inv_sum_p = 1.0 / sum_p;
  for (int j = 0; j < k; ++j) row_p[j] *= inv_sum_p;
}

void compute_row_probabilities_float(const std::vector<float>& distances,
                                     const int row,
                                     const int k,
                                     const double perplexity,
                                     float* row_p) {
  std::fill(row_p, row_p + k, 0.0f);

  const std::size_t row_base = static_cast<std::size_t>(row) * k;
  float min_d2 = FLT_MAX;
  float max_d2 = 0.0f;
  for (int j = 0; j < k; ++j) {
    const float d = distances[row_base + j];
    const float d2 = d * d;
    min_d2 = std::min(min_d2, d2);
    max_d2 = std::max(max_d2, d2);
  }
  const float spread = max_d2 - min_d2;
  if (spread <= FLT_EPSILON * std::max(1.0f, max_d2)) {
    const float uniform = 1.0f / static_cast<float>(k);
    std::fill(row_p, row_p + k, uniform);
    return;
  }

  bool found = false;
  float beta = 1.0f;
  float min_beta = -FLT_MAX;
  float max_beta = FLT_MAX;
  const float tol = 1e-5f;
  float sum_p = FLT_MIN;

  for (int iter = 0; !found && iter < 200; ++iter) {
    sum_p = FLT_MIN;
    for (int j = 0; j < k; ++j) {
      const float d = distances[row_base + j];
      const float d2 = d * d - min_d2;
      const float p = std::exp(-beta * d2);
      row_p[j] = p;
      sum_p += p;
    }

    float entropy = 0.0f;
    for (int j = 0; j < k; ++j) {
      const float d = distances[row_base + j];
      entropy += beta * ((d * d - min_d2) * row_p[j]);
    }
    entropy = entropy / sum_p + std::log(sum_p);
    const float diff = entropy - static_cast<float>(std::log(perplexity));

    if (std::abs(diff) < tol) {
      found = true;
    } else if (diff > 0.0f) {
      min_beta = beta;
      beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ?
        beta * 2.0f :
        (beta + max_beta) * 0.5f;
    } else {
      max_beta = beta;
      beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ?
        beta * 0.5f :
        (beta + min_beta) * 0.5f;
    }
    if (!std::isfinite(beta)) break;
  }

  if (!std::isfinite(sum_p) || sum_p <= FLT_MIN) {
    int tied = 0;
    for (int j = 0; j < k; ++j) {
      const float d = distances[row_base + j];
      const float d2 = d * d;
      if (std::abs(d2 - min_d2) <= FLT_EPSILON * std::max(1.0f, min_d2)) ++tied;
    }
    const float tied_mass = 1.0f / static_cast<float>(std::max(1, tied));
    for (int j = 0; j < k; ++j) {
      const float d = distances[row_base + j];
      const float d2 = d * d;
      row_p[j] = std::abs(d2 - min_d2) <= FLT_EPSILON * std::max(1.0f, min_d2) ?
        tied_mass : 0.0f;
    }
    return;
  }
  const float inv_sum_p = 1.0f / sum_p;
  for (int j = 0; j < k; ++j) row_p[j] *= inv_sum_p;
}

SparseProbabilitiesF build_tsne_probabilities_float(const IntegerMatrix& indices,
                                                    const std::vector<float>& distances,
                                                    const double perplexity,
                                                    const int n_threads) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (static_cast<std::size_t>(n) * k != distances.size()) {
    Rcpp::stop("float32 KNN distance payload has incompatible dimensions.");
  }
  const int offset = resolve_index_offset(indices);

  if (perplexity > static_cast<double>(k)) {
    Rcpp::warning("Perplexity is larger than the supplied KNN width; results may be unstable.");
  }
  for (int i = 0; i < n; ++i) {
    const std::size_t row_base = static_cast<std::size_t>(i) * k;
    for (int j = 0; j < k; ++j) {
      const int nb = indices(i, j) - offset;
      if (nb < 0 || nb >= n) Rcpp::stop("KNN indices are out of range.");
      const float d = distances[row_base + j];
      if (!std::isfinite(d) || d < 0.0f) {
        Rcpp::stop("KNN distances must be finite and non-negative.");
      }
    }
  }

  std::vector<std::vector<PackedEdgeF>> local_edges(static_cast<std::size_t>(n_threads));
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<float> row_p(static_cast<std::size_t>(k), 0.0f);
    std::vector<PackedEdgeF>& edges = local_edges[static_cast<std::size_t>(thread_id)];
    edges.reserve(edges.size() + static_cast<std::size_t>(std::max(0, end - begin)) * k);

    for (int i = begin; i < end; ++i) {
      compute_row_probabilities_float(distances, i, k, perplexity, row_p.data());
      for (int j = 0; j < k; ++j) {
        const int nb = indices(i, j) - offset;
        if (nb == i) continue;
        const int a = std::min(i, nb);
        const int b = std::max(i, nb);
        edges.push_back({pair_key(a, b), row_p[static_cast<std::size_t>(j)]});
      }
    }
  });

  std::size_t edge_count = 0;
  for (const auto& edges : local_edges) edge_count += edges.size();
  std::vector<PackedEdgeF> edges;
  edges.reserve(edge_count);
  for (auto& local : local_edges) {
    edges.insert(edges.end(), local.begin(), local.end());
    std::vector<PackedEdgeF>().swap(local);
  }

  std::sort(edges.begin(), edges.end(), [](const PackedEdgeF& a, const PackedEdgeF& b) {
    return a.key < b.key;
  });

  SparseProbabilitiesF p;
  p.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
  if (edges.empty()) {
    Rcpp::stop("KNN graph produced no non-self t-SNE edges.");
  }

  std::size_t write = 0;
  double total_directed_mass = 0.0;
  for (std::size_t read = 0; read < edges.size();) {
    const std::uint64_t key = edges[read].key;
    double sum = 0.0;
    while (read < edges.size() && edges[read].key == key) {
      sum += static_cast<double>(edges[read].value);
      ++read;
    }
    edges[write++] = {key, static_cast<float>(sum)};
    total_directed_mass += sum;
    const int a = key_first(key);
    const int b = key_second(key);
    ++p.row_ptr[static_cast<std::size_t>(a + 1)];
    ++p.row_ptr[static_cast<std::size_t>(b + 1)];
  }
  edges.resize(write);

  if (!std::isfinite(total_directed_mass) || total_directed_mass <= 0.0) {
    Rcpp::stop("t-SNE probability normalization failed.");
  }

  for (int i = 0; i < n; ++i) {
    p.row_ptr[static_cast<std::size_t>(i + 1)] += p.row_ptr[static_cast<std::size_t>(i)];
  }
  p.col.assign(static_cast<std::size_t>(p.row_ptr[static_cast<std::size_t>(n)]), 0);
  p.val.assign(p.col.size(), 0.0f);
  std::vector<int> fill = p.row_ptr;

  const float norm = static_cast<float>(0.5 / total_directed_mass);
  for (const PackedEdgeF& edge : edges) {
    const int a = key_first(edge.key);
    const int b = key_second(edge.key);
    const float value = edge.value * norm;

    int pos = fill[static_cast<std::size_t>(a)]++;
    p.col[static_cast<std::size_t>(pos)] = b;
    p.val[static_cast<std::size_t>(pos)] = value;

    pos = fill[static_cast<std::size_t>(b)]++;
    p.col[static_cast<std::size_t>(pos)] = a;
    p.val[static_cast<std::size_t>(pos)] = value;
  }

  return p;
}

template <typename T>
struct FftPlanT {
  int size = 0;
  int thread_capacity = 0;
  std::vector<int> bit_reverse;
  std::vector<std::complex<T>> forward_roots;
  std::vector<std::complex<T>> inverse_roots;
  std::vector<std::complex<T>> column_scratch;

  void ensure(const int requested_size, const int requested_threads) {
    const int threads = std::max(1, requested_threads);
    if (size == requested_size) {
      if (thread_capacity < threads) {
        thread_capacity = threads;
        column_scratch.resize(static_cast<std::size_t>(size) * thread_capacity);
      }
      return;
    }
    size = requested_size;
    thread_capacity = threads;
    bit_reverse.resize(static_cast<std::size_t>(size));
    int bits = 0;
    for (int value = size; value > 1; value >>= 1) ++bits;
    for (int i = 0; i < size; ++i) {
      unsigned int value = static_cast<unsigned int>(i);
      unsigned int reversed = 0u;
      for (int bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1u) | (value & 1u);
        value >>= 1u;
      }
      bit_reverse[static_cast<std::size_t>(i)] = static_cast<int>(reversed);
    }

    forward_roots.clear();
    inverse_roots.clear();
    forward_roots.reserve(static_cast<std::size_t>(bits));
    inverse_roots.reserve(static_cast<std::size_t>(bits));
    for (int len = 2; len <= size; len <<= 1) {
      const T angle = static_cast<T>(6.283185307179586476925286766559) /
        static_cast<T>(len);
      forward_roots.emplace_back(std::cos(angle), std::sin(angle));
      inverse_roots.emplace_back(std::cos(-angle), std::sin(-angle));
    }
    column_scratch.resize(static_cast<std::size_t>(size) * thread_capacity);
  }
};

template <typename T>
void fft_1d_t(std::complex<T>* a,
              const int n,
              const bool inverse,
              const FftPlanT<T>& plan) {
  for (int i = 1; i < n; ++i) {
    const int j = plan.bit_reverse[static_cast<std::size_t>(i)];
    if (i < j) std::swap(a[i], a[j]);
  }

  const std::vector<std::complex<T>>& roots = inverse ?
    plan.inverse_roots : plan.forward_roots;
  int stage = 0;
  for (int len = 2; len <= n; len <<= 1, ++stage) {
    const std::complex<T> root = roots[static_cast<std::size_t>(stage)];
    for (int i = 0; i < n; i += len) {
      std::complex<T> w(static_cast<T>(1.0), static_cast<T>(0.0));
      const int half = len >> 1;
      for (int j = 0; j < half; ++j) {
        const std::complex<T> u = a[i + j];
        const std::complex<T> v = a[i + j + half] * w;
        a[i + j] = u + v;
        a[i + j + half] = u - v;
        w *= root;
      }
    }
  }

  if (inverse) {
    const T scale = static_cast<T>(1.0) / static_cast<T>(n);
    for (int i = 0; i < n; ++i) a[i] *= scale;
  }
}

template <typename T>
void fft_2d_t(std::vector<std::complex<T>>& values,
              const int size,
              const bool inverse,
              const int n_threads,
              FftPlanT<T>& plan) {
  parallel_for(size, n_threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      fft_1d_t<T>(values.data() + static_cast<std::size_t>(row) * size, size, inverse, plan);
    }
  });

  parallel_for(size, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::complex<T>* column = plan.column_scratch.data() +
      static_cast<std::size_t>(thread_id) * size;
    for (int col = begin; col < end; ++col) {
      for (int row = 0; row < size; ++row) {
        column[row] =
          values[static_cast<std::size_t>(row) * size + col];
      }
      fft_1d_t<T>(column, size, inverse, plan);
      for (int row = 0; row < size; ++row) {
        values[static_cast<std::size_t>(row) * size + col] =
          column[row];
      }
    }
  });
}

template <typename T>
T bilinear_grid_value_t(const std::vector<T>& grid,
                        const int grid_size,
                        const T gx,
                        const T gy) {
  const T cx = std::max(static_cast<T>(0.0), std::min(static_cast<T>(grid_size - 1), gx));
  const T cy = std::max(static_cast<T>(0.0), std::min(static_cast<T>(grid_size - 1), gy));
  const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cx))));
  const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cy))));
  const T tx = cx - static_cast<T>(x0);
  const T ty = cy - static_cast<T>(y0);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const T v00 = grid[static_cast<std::size_t>(y0) * grid_size + x0];
  const T v10 = grid[static_cast<std::size_t>(y0) * grid_size + x1];
  const T v01 = grid[static_cast<std::size_t>(y1) * grid_size + x0];
  const T v11 = grid[static_cast<std::size_t>(y1) * grid_size + x1];
  return (static_cast<T>(1.0) - tx) * (static_cast<T>(1.0) - ty) * v00 +
    tx * (static_cast<T>(1.0) - ty) * v10 +
    (static_cast<T>(1.0) - tx) * ty * v01 +
    tx * ty * v11;
}

template <typename T>
struct FftGridWorkspaceT {
  int grid_size = 0;
  int fft_size = 0;
  int n = 0;
  int n_threads = 0;
  std::vector<T> mass;
  std::vector<T> mass_x;
  std::vector<T> mass_y;
  std::vector<T> gx;
  std::vector<T> gy;
  std::vector<T> q_grid;
  std::vector<T> q2_grid;
  std::vector<T> xq2_grid;
  std::vector<T> yq2_grid;
  std::vector<double> partial_sum_q;
  std::vector<std::complex<T>> mass_fft;
  std::vector<std::complex<T>> mass_x_fft;
  std::vector<std::complex<T>> mass_y_fft;
  std::vector<std::complex<T>> kernel_q;
  std::vector<std::complex<T>> kernel_q2;
  std::vector<std::complex<T>> work;
  FftPlanT<T> fft_plan;

  void ensure(const int requested_grid_size,
              const int requested_n,
              const int requested_threads) {
    grid_size = requested_grid_size;
    fft_size = grid_size << 1;
    n = requested_n;
    n_threads = std::max(1, requested_threads);
    const std::size_t grid_total = static_cast<std::size_t>(grid_size) * grid_size;
    const std::size_t fft_total = static_cast<std::size_t>(fft_size) * fft_size;
    mass.resize(grid_total);
    mass_x.resize(grid_total);
    mass_y.resize(grid_total);
    gx.resize(static_cast<std::size_t>(n));
    gy.resize(static_cast<std::size_t>(n));
    q_grid.resize(grid_total);
    q2_grid.resize(grid_total);
    xq2_grid.resize(grid_total);
    yq2_grid.resize(grid_total);
    partial_sum_q.resize(static_cast<std::size_t>(n_threads));
    mass_fft.resize(fft_total);
    mass_x_fft.resize(fft_total);
    mass_y_fft.resize(fft_total);
    kernel_q.resize(fft_total);
    kernel_q2.resize(fft_total);
    work.resize(fft_total);
    fft_plan.ensure(fft_size, n_threads);
  }

  void clear_grid_mass() {
    std::fill(mass.begin(), mass.end(), static_cast<T>(0.0));
    std::fill(mass_x.begin(), mass_x.end(), static_cast<T>(0.0));
    std::fill(mass_y.begin(), mass_y.end(), static_cast<T>(0.0));
  }
};

template <typename T>
void copy_grid_to_fft_t(const std::vector<T>& grid,
                        const int grid_size,
                        const int fft_size,
                        std::vector<std::complex<T>>& out,
                        const int n_threads) {
  parallel_for(fft_size, n_threads, [&](const int begin, const int end, const int) {
    const std::complex<T> zero(static_cast<T>(0.0), static_cast<T>(0.0));
    for (int y_cell = begin; y_cell < end; ++y_cell) {
      std::complex<T>* output_row = out.data() + static_cast<std::size_t>(y_cell) * fft_size;
      std::fill(output_row, output_row + fft_size, zero);
      if (y_cell < grid_size) {
        const T* input_row = grid.data() + static_cast<std::size_t>(y_cell) * grid_size;
        for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
          output_row[x_cell] = input_row[x_cell];
        }
      }
    }
  });
}

template <typename T>
void copy_fft_to_grid_t(const std::vector<std::complex<T>>& values,
                        const int grid_size,
                        const int fft_size,
                        std::vector<T>& out,
                        const int n_threads) {
  out.resize(static_cast<std::size_t>(grid_size) * grid_size);
  parallel_for(grid_size, n_threads, [&](const int begin, const int end, const int) {
    for (int y_cell = begin; y_cell < end; ++y_cell) {
      for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
        out[static_cast<std::size_t>(y_cell) * grid_size + x_cell] =
          values[static_cast<std::size_t>(y_cell) * fft_size + x_cell].real();
      }
    }
  });
}

template <typename T>
void compute_fft_grid_convolution_workspace_t(const std::vector<T>& mass,
                                              const std::vector<T>& mass_x,
                                              const std::vector<T>& mass_y,
                                              const int grid_size,
                                              const T spacing,
                                              const int n_threads,
                                              FftGridWorkspaceT<T>& ws) {
  const int fft_size = grid_size << 1;
  const std::size_t fft_total = static_cast<std::size_t>(fft_size) * fft_size;

  copy_grid_to_fft_t<T>(mass, grid_size, fft_size, ws.mass_fft, n_threads);
  copy_grid_to_fft_t<T>(mass_x, grid_size, fft_size, ws.mass_x_fft, n_threads);
  copy_grid_to_fft_t<T>(mass_y, grid_size, fft_size, ws.mass_y_fft, n_threads);
  parallel_for(static_cast<int>(fft_total), n_threads, [&](const int begin, const int end, const int) {
    const std::complex<T> zero(static_cast<T>(0.0), static_cast<T>(0.0));
    std::fill(ws.kernel_q.begin() + begin, ws.kernel_q.begin() + end, zero);
    std::fill(ws.kernel_q2.begin() + begin, ws.kernel_q2.begin() + end, zero);
  });

  const int kernel_rows = 2 * grid_size - 1;
  parallel_for(kernel_rows, n_threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      const int dy = row - (grid_size - 1);
      const int yy = dy < 0 ? dy + fft_size : dy;
      const T y_offset = static_cast<T>(dy) * spacing;
      for (int dx = -(grid_size - 1); dx <= grid_size - 1; ++dx) {
        const int xx = dx < 0 ? dx + fft_size : dx;
        const T x_offset = static_cast<T>(dx) * spacing;
        const T d2 = x_offset * x_offset + y_offset * y_offset;
        const T q = static_cast<T>(1.0) / (static_cast<T>(1.0) + d2);
        const T q2 = q * q;
        const std::size_t pos = static_cast<std::size_t>(yy) * fft_size + xx;
        ws.kernel_q[pos] = q;
        ws.kernel_q2[pos] = q2;
      }
    }
  });

  fft_2d_t<T>(ws.mass_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.mass_x_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.mass_y_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.kernel_q, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.kernel_q2, fft_size, false, n_threads, ws.fft_plan);

  auto convolve = [&](const std::vector<std::complex<T>>& mass_values,
                      const std::vector<std::complex<T>>& kernel_values,
                      std::vector<T>& out) {
    parallel_for(static_cast<int>(fft_total), n_threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        ws.work[static_cast<std::size_t>(i)] =
          mass_values[static_cast<std::size_t>(i)] * kernel_values[static_cast<std::size_t>(i)];
      }
    });
    fft_2d_t<T>(ws.work, fft_size, true, n_threads, ws.fft_plan);
    copy_fft_to_grid_t<T>(ws.work, grid_size, fft_size, out, n_threads);
  };

  convolve(ws.mass_fft, ws.kernel_q, ws.q_grid);
  convolve(ws.mass_fft, ws.kernel_q2, ws.q2_grid);
  convolve(ws.mass_x_fft, ws.kernel_q2, ws.xq2_grid);
  convolve(ws.mass_y_fft, ws.kernel_q2, ws.yq2_grid);
}

float squared_distance_f(const std::vector<float>& y,
                         const int i,
                         const int j,
                         const int dims) {
  const std::size_t ib = static_cast<std::size_t>(i) * dims;
  const std::size_t jb = static_cast<std::size_t>(j) * dims;
  float out = 0.0f;
  for (int d = 0; d < dims; ++d) {
    const float diff = y[ib + d] - y[jb + d];
    out += diff * diff;
  }
  return out;
}

float compute_sum_q_f(const std::vector<float>& y,
                      const int n,
                      const int dims,
                      const int n_threads) {
  std::vector<double> partial(static_cast<std::size_t>(n_threads), 0.0);
  auto worker = [&](const int thread_id) {
    double local = 0.0;
    for (int i = thread_id; i < n - 1; i += n_threads) {
      for (int j = i + 1; j < n; ++j) {
        local += 2.0 / (1.0 + static_cast<double>(squared_distance_f(y, i, j, dims)));
      }
    }
    partial[static_cast<std::size_t>(thread_id)] = local;
  };
  if (n_threads <= 1) {
    worker(0);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int t = 1; t < n_threads; ++t) workers.emplace_back(worker, t);
    worker(0);
    for (auto& thread : workers) thread.join();
  }
  const double sum_q = std::accumulate(partial.begin(), partial.end(), 0.0);
  return static_cast<float>(std::max(sum_q, static_cast<double>(FLT_MIN)));
}

void add_sparse_attractive_gradient_f(const SparseProbabilitiesF& p,
                                      const std::vector<float>& y,
                                      const int n,
                                      const int dims,
                                      const float exaggeration,
                                      const int n_threads,
                                      std::vector<float>& grad) {
  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      const int row_begin = p.row_ptr[static_cast<std::size_t>(i)];
      const int row_end = p.row_ptr[static_cast<std::size_t>(i + 1)];
      for (int pos = row_begin; pos < row_end; ++pos) {
        const int j = p.col[static_cast<std::size_t>(pos)];
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        float diff[3] = {0.0f, 0.0f, 0.0f};
        float d2 = 0.0f;
        for (int d = 0; d < dims; ++d) {
          diff[d] = y[ib + d] - y[jb + d];
          d2 += diff[d] * diff[d];
        }
        const float q = 1.0f / (1.0f + d2);
        const float coeff = exaggeration * p.val[static_cast<std::size_t>(pos)] * q;
        for (int d = 0; d < dims; ++d) grad[ib + d] += coeff * diff[d];
      }
    }
  });
}

void compute_gradient_pair_symmetric_f(const SparseProbabilitiesF& p,
                                       const std::vector<float>& y,
                                       const int n,
                                       const int dims,
                                       const float exaggeration,
                                       const int n_threads,
                                       std::vector<float>& grad) {
  std::fill(grad.begin(), grad.end(), 0.0f);
  std::vector<std::vector<float>> local_grad(
    static_cast<std::size_t>(n_threads),
    std::vector<float>(grad.size(), 0.0f)
  );
  std::vector<double> partial_sum_q(static_cast<std::size_t>(n_threads), 0.0);
  auto repulsive_worker = [&](const int thread_id) {
    std::vector<float>& g = local_grad[static_cast<std::size_t>(thread_id)];
    double local_sum_q = 0.0;
    for (int i = thread_id; i < n - 1; i += n_threads) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      for (int j = i + 1; j < n; ++j) {
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        float d2 = 0.0f;
        for (int d = 0; d < dims; ++d) {
          const float diff = y[ib + d] - y[jb + d];
          d2 += diff * diff;
        }
        const float q = 1.0f / (1.0f + d2);
        local_sum_q += 2.0 * static_cast<double>(q);
        const float coeff = -(q * q);
        for (int d = 0; d < dims; ++d) {
          const float step = coeff * (y[ib + d] - y[jb + d]);
          g[ib + d] += step;
          g[jb + d] -= step;
        }
      }
    }
    partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  };
  if (n_threads <= 1) {
    repulsive_worker(0);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int t = 1; t < n_threads; ++t) workers.emplace_back(repulsive_worker, t);
    repulsive_worker(0);
    for (auto& thread : workers) thread.join();
  }
  const float inv_sum_q = static_cast<float>(1.0 / std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0),
    static_cast<double>(FLT_MIN)
  ));
  parallel_for(static_cast<int>(grad.size()), n_threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      float value = 0.0f;
      for (int t = 0; t < n_threads; ++t) {
        value += local_grad[static_cast<std::size_t>(t)][static_cast<std::size_t>(index)];
      }
      grad[static_cast<std::size_t>(index)] = value * inv_sum_q;
    }
  });
  add_sparse_attractive_gradient_f(p, y, n, dims, exaggeration, n_threads, grad);
}

void compute_gradient_fft_grid_f(const SparseProbabilitiesF& p,
                                 const std::vector<float>& y,
                                 const int n,
                                 const int dims,
                                 const float exaggeration,
                                 const int n_threads,
                                 FftGridWorkspaceT<float>* workspace,
                                 std::vector<float>& grad,
                                 const int grid_size_override) {
  if (dims != 2) {
    compute_gradient_pair_symmetric_f(p, y, n, dims, exaggeration, n_threads, grad);
    return;
  }
  std::fill(grad.begin(), grad.end(), 0.0f);
  const int grid_size = grid_size_override > 0 ?
    grid_size_override : tsne_fft_grid_size(n);
  float min_x = y[0], max_x = y[0], min_y = y[1], max_y = y[1];
  for (int i = 1; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    min_x = std::min(min_x, y[base]);
    max_x = std::max(max_x, y[base]);
    min_y = std::min(min_y, y[base + 1u]);
    max_y = std::max(max_y, y[base + 1u]);
  }
  const float cx = 0.5f * (min_x + max_x);
  const float cy = 0.5f * (min_y + max_y);
  float span = std::max(max_x - min_x, max_y - min_y);
  if (!std::isfinite(span) || span <= 0.0f) span = 1.0f;
  const float half = 0.55f * span + 1.0e-3f;
  const float lower_x = cx - half;
  const float lower_y = cy - half;
  const float spacing = (2.0f * half) / static_cast<float>(grid_size - 1);
  const float inv_spacing = 1.0f / spacing;

  FftGridWorkspaceT<float> local_workspace;
  FftGridWorkspaceT<float>& ws = workspace == nullptr ? local_workspace : *workspace;
  ws.ensure(grid_size, n, n_threads);
  ws.clear_grid_mass();
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const float x_coord = y[base];
    const float y_coord = y[base + 1u];
    const float raw_x = (x_coord - lower_x) * inv_spacing;
    const float raw_y = (y_coord - lower_y) * inv_spacing;
    const float clamped_x = std::max(0.0f, std::min(static_cast<float>(grid_size - 1), raw_x));
    const float clamped_y = std::max(0.0f, std::min(static_cast<float>(grid_size - 1), raw_y));
    const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_x))));
    const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_y))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = clamped_x - static_cast<float>(x0);
    const float ty = clamped_y - static_cast<float>(y0);
    ws.gx[static_cast<std::size_t>(i)] = clamped_x;
    ws.gy[static_cast<std::size_t>(i)] = clamped_y;
    const float w00 = (1.0f - tx) * (1.0f - ty);
    const float w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty;
    const float w11 = tx * ty;
    const std::size_t p00 = static_cast<std::size_t>(y0) * grid_size + x0;
    const std::size_t p10 = static_cast<std::size_t>(y0) * grid_size + x1;
    const std::size_t p01 = static_cast<std::size_t>(y1) * grid_size + x0;
    const std::size_t p11 = static_cast<std::size_t>(y1) * grid_size + x1;
    ws.mass[p00] += w00;
    ws.mass[p10] += w10;
    ws.mass[p01] += w01;
    ws.mass[p11] += w11;
    ws.mass_x[p00] += w00 * x_coord;
    ws.mass_x[p10] += w10 * x_coord;
    ws.mass_x[p01] += w01 * x_coord;
    ws.mass_x[p11] += w11 * x_coord;
    ws.mass_y[p00] += w00 * y_coord;
    ws.mass_y[p10] += w10 * y_coord;
    ws.mass_y[p01] += w01 * y_coord;
    ws.mass_y[p11] += w11 * y_coord;
  }

  compute_fft_grid_convolution_workspace_t<float>(
    ws.mass, ws.mass_x, ws.mass_y, grid_size, spacing, n_threads, ws
  );

  std::fill(ws.partial_sum_q.begin(), ws.partial_sum_q.end(), 0.0);
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local_sum_q = 0.0;
    for (int i = begin; i < end; ++i) {
      local_sum_q += static_cast<double>(bilinear_grid_value_t<float>(
        ws.q_grid, grid_size, ws.gx[static_cast<std::size_t>(i)], ws.gy[static_cast<std::size_t>(i)]
      ));
    }
    ws.partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  });
  const float inv_sum_q = static_cast<float>(1.0 / std::max(
    std::accumulate(ws.partial_sum_q.begin(), ws.partial_sum_q.end(), 0.0) - static_cast<double>(n),
    static_cast<double>(FLT_MIN)
  ));

  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      const float px = ws.gx[static_cast<std::size_t>(i)];
      const float py = ws.gy[static_cast<std::size_t>(i)];
      const float q2_value = bilinear_grid_value_t<float>(ws.q2_grid, grid_size, px, py);
      const float xq2_value = bilinear_grid_value_t<float>(ws.xq2_grid, grid_size, px, py);
      const float yq2_value = bilinear_grid_value_t<float>(ws.yq2_grid, grid_size, px, py);
      grad[base] = -(y[base] * q2_value - xq2_value) * inv_sum_q;
      grad[base + 1u] = -(y[base + 1u] * q2_value - yq2_value) * inv_sum_q;
    }
  });
  add_sparse_attractive_gradient_f(p, y, n, dims, exaggeration, n_threads, grad);
}

void compute_gradient_f(const SparseProbabilitiesF& p,
                        const std::vector<float>& y,
                        const int n,
                        const int dims,
                        const float exaggeration,
                        const int n_threads,
                        const std::string& repulsion_mode,
                        FftGridWorkspaceT<float>* fft_workspace,
                        std::vector<float>& grad) {
  if (repulsion_mode == "fft_grid") {
    compute_gradient_fft_grid_f(
      p, y, n, dims, exaggeration, n_threads, fft_workspace, grad, 0
    );
  } else {
    compute_gradient_pair_symmetric_f(p, y, n, dims, exaggeration, n_threads, grad);
  }
}

void zero_mean_f(std::vector<float>& y, const int n, const int dims) {
  std::array<double, 3> mean{{0.0, 0.0, 0.0}};
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] += y[base + d];
  }
  for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) y[base + d] -= static_cast<float>(mean[static_cast<std::size_t>(d)]);
  }
}

float sign_tsne_f(float x) {
  return x == 0.0f ? 0.0f : (x < 0.0f ? -1.0f : 1.0f);
}

void apply_open_tsne_update_f(std::vector<float>& y,
                              std::vector<float>& update,
                              std::vector<float>& gains,
                              const std::vector<float>& grad,
                              const int n,
                              const int dims,
                              const float learning_rate,
                              const float momentum,
                              const float min_gain,
                              const float max_step_norm,
                              const int n_threads) {
  const bool clip_steps = std::isfinite(max_step_norm) && max_step_norm > 0.0f;
  const float max_step_norm_sq = max_step_norm * max_step_norm;
  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * dims;
      float step_norm_sq = 0.0f;
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        if (sign_tsne_f(update[index]) != sign_tsne_f(grad[index])) {
          gains[index] += 0.2f;
        } else {
          gains[index] = gains[index] * 0.8f + min_gain;
        }
        if (gains[index] < min_gain) gains[index] = min_gain;
        update[index] = momentum * update[index] - learning_rate * gains[index] * grad[index];
        step_norm_sq += update[index] * update[index];
      }
      float scale = 1.0f;
      if (clip_steps && step_norm_sq > max_step_norm_sq) {
        scale = max_step_norm / std::sqrt(std::max(step_norm_sq, FLT_MIN));
      }
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        update[index] *= scale;
        y[index] += update[index];
      }
    }
  });
}

double evaluate_kl_f(const SparseProbabilitiesF& p,
                     const std::vector<float>& y,
                     const int n,
                     const int dims,
                     const int n_threads,
                     std::vector<double>* row_costs = nullptr) {
  const double sum_q = static_cast<double>(compute_sum_q_f(y, n, dims, n_threads));
  std::vector<double> partial(static_cast<std::size_t>(n_threads), 0.0);
  if (row_costs != nullptr) row_costs->assign(static_cast<std::size_t>(n), 0.0);
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local = 0.0;
    for (int i = begin; i < end; ++i) {
      double row_total = 0.0;
      const int row_begin = p.row_ptr[static_cast<std::size_t>(i)];
      const int row_end = p.row_ptr[static_cast<std::size_t>(i + 1)];
      for (int pos = row_begin; pos < row_end; ++pos) {
        const int j = p.col[static_cast<std::size_t>(pos)];
        const double p_ij = static_cast<double>(p.val[static_cast<std::size_t>(pos)]);
        const double q_ij = (1.0 / (1.0 + static_cast<double>(squared_distance_f(y, i, j, dims)))) / sum_q;
        row_total += p_ij * std::log((p_ij + 1e-9) / (q_ij + 1e-9));
      }
      if (row_costs != nullptr) (*row_costs)[static_cast<std::size_t>(i)] = row_total;
      local += row_total;
    }
    partial[static_cast<std::size_t>(thread_id)] = local;
  });
  return std::accumulate(partial.begin(), partial.end(), 0.0);
}

double sign_tsne(double x) {
  return x == 0.0 ? 0.0 : (x < 0.0 ? -1.0 : 1.0);
}

int resolve_reference_index_offset(const IntegerMatrix& indices,
                                   const int n_reference) {
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < indices.nrow(); ++i) {
    for (int j = 0; j < indices.ncol(); ++j) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  return (min_idx >= 1 && max_idx <= n_reference) ? 1 : 0;
}

void initialize_tsne_transform(const NumericMatrix& reference_layout,
                               const IntegerMatrix& indices,
                               const NumericMatrix& distances,
                               const int offset,
                               const std::string& initialization,
                               const int seed,
                               const int query_begin,
                               const int batch_n,
                               const int n_threads,
                               std::vector<double>& y) {
  const int k = indices.ncol();
  const int dims = reference_layout.ncol();
  const std::size_t active_size = static_cast<std::size_t>(batch_n) * dims;

  if (initialization == "random") {
    const unsigned int resolved_seed = seed == NA_INTEGER ?
      5489u :
      static_cast<unsigned int>(seed);
    parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int) {
      std::normal_distribution<double> normal(0.0, 1.0e-4);
      for (int i = begin; i < end; ++i) {
        const int global_i = query_begin + i;
        std::mt19937 rng(
          mix_uint32(resolved_seed ^ (static_cast<std::uint32_t>(global_i + 1) * 0x9e3779b9u))
        );
        for (int d = 0; d < dims; ++d) {
          y[static_cast<std::size_t>(i) * dims + d] = normal(rng);
        }
      }
    });
    return;
  }

  std::fill(y.begin(), y.begin() + active_size, 0.0);
  parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int) {
    std::vector<double> values(static_cast<std::size_t>(k), 0.0);
    for (int i = begin; i < end; ++i) {
      const int global_i = query_begin + i;
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      for (int d = 0; d < dims; ++d) {
        if (initialization == "weighted") {
          double numerator = 0.0;
          double denominator = DBL_MIN;
          for (int j = 0; j < k; ++j) {
            const int ref = indices(global_i, j) - offset;
            const double distance = std::max(0.0, distances(global_i, j));
            const double weight = 1.0 / (distance + 1e-6);
            numerator += weight * reference_layout(ref, d);
            denominator += weight;
          }
          y[ib + d] = numerator / denominator;
        } else {
          for (int j = 0; j < k; ++j) {
            const int ref = indices(global_i, j) - offset;
            values[static_cast<std::size_t>(j)] = reference_layout(ref, d);
          }
          const int mid = k / 2;
          std::nth_element(values.begin(), values.begin() + mid, values.end());
          double median = values[static_cast<std::size_t>(mid)];
          if ((k & 1) == 0) {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.begin() + mid);
            median = 0.5 * (median + values[static_cast<std::size_t>(mid - 1)]);
          }
          y[ib + d] = median;
        }
      }
    }
  });
}

void compute_tsne_transform_gradient(const NumericMatrix& reference_layout,
                                     const IntegerMatrix& indices,
                                     const std::vector<double>& probabilities,
                                     const std::vector<double>& y,
                                     const int offset,
                                     const int query_begin,
                                     const int batch_n,
                                     const double exaggeration,
                                     const int n_negatives,
                                     const int exact_repulsion_threshold,
                                     const int n_threads,
                                     const int seed,
                                     TsneTransformGradientWorkspace& workspace,
                                     std::vector<double>& grad) {
  const int k = indices.ncol();
  const int n_reference = reference_layout.nrow();
  const int dims = reference_layout.ncol();
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  std::fill(grad.begin(), grad.begin() + static_cast<std::size_t>(batch_n) * dims, 0.0);
  workspace.ensure(
    n_threads,
    exact_repulsion ? n_reference : std::max(1, n_negatives),
    std::max(1, n_negatives)
  );

  parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<int>& sampled = workspace.sampled[static_cast<std::size_t>(thread_id)];
    std::vector<double>& q_values = workspace.q_values[static_cast<std::size_t>(thread_id)];

    for (int i = begin; i < end; ++i) {
      const int global_i = query_begin + i;
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      double sum_q = DBL_MIN;
      std::uint32_t rng_state = mix_uint32(
        static_cast<std::uint32_t>(seed) ^
          (static_cast<std::uint32_t>(global_i + 1) * 0x9e3779b9u)
      );

      if (exact_repulsion) {
        for (int ref = 0; ref < n_reference; ++ref) {
          double d2 = 0.0;
          for (int d = 0; d < dims; ++d) {
            const double diff = y[ib + d] - reference_layout(ref, d);
            d2 += diff * diff;
          }
          const double q = 1.0 / (1.0 + d2);
          q_values[static_cast<std::size_t>(ref)] = q;
          sum_q += q;
        }
        for (int ref = 0; ref < n_reference; ++ref) {
          const double q = q_values[static_cast<std::size_t>(ref)];
          const double coeff = -(q * q) / sum_q;
          for (int d = 0; d < dims; ++d) {
            grad[ib + d] += coeff * (y[ib + d] - reference_layout(ref, d));
          }
        }
      } else {
        for (int m = 0; m < n_negatives; ++m) {
          const int ref = uniform_index(rng_state, n_reference);
          sampled[static_cast<std::size_t>(m)] = ref;
          double d2 = 0.0;
          for (int d = 0; d < dims; ++d) {
            const double diff = y[ib + d] - reference_layout(ref, d);
            d2 += diff * diff;
          }
          const double q = 1.0 / (1.0 + d2);
          q_values[static_cast<std::size_t>(m)] = q;
          sum_q += q;
        }
        for (int m = 0; m < n_negatives; ++m) {
          const int ref = sampled[static_cast<std::size_t>(m)];
          const double q = q_values[static_cast<std::size_t>(m)];
          const double coeff = -(q * q) / sum_q;
          for (int d = 0; d < dims; ++d) {
            grad[ib + d] += coeff * (y[ib + d] - reference_layout(ref, d));
          }
        }
      }

      const std::size_t p_base = static_cast<std::size_t>(i) * k;
      for (int j = 0; j < k; ++j) {
        const int ref = indices(global_i, j) - offset;
        double d2 = 0.0;
        for (int d = 0; d < dims; ++d) {
          const double diff = y[ib + d] - reference_layout(ref, d);
          d2 += diff * diff;
        }
        const double q = 1.0 / (1.0 + d2);
        const double coeff = exaggeration * probabilities[p_base + j] * q;
        for (int d = 0; d < dims; ++d) {
          grad[ib + d] += coeff * (y[ib + d] - reference_layout(ref, d));
        }
      }
    }
  });
}

void compute_tsne_transform_gradient_2d_flat(const std::vector<double>& reference_layout,
                                             const IntegerMatrix& indices,
                                             const std::vector<double>& probabilities,
                                             const std::vector<double>& y,
                                             const int offset,
                                             const int query_begin,
                                             const int batch_n,
                                             const double exaggeration,
                                             const int n_negatives,
                                             const int exact_repulsion_threshold,
                                             const int n_threads,
                                             const int seed,
                                             TsneTransformGradientWorkspace& workspace,
                                             std::vector<double>& grad) {
  const int k = indices.ncol();
  const int n_reference = static_cast<int>(reference_layout.size() / 2u);
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  std::fill(grad.begin(), grad.begin() + static_cast<std::size_t>(batch_n) * 2u, 0.0);
  workspace.ensure(
    n_threads,
    exact_repulsion ? n_reference : std::max(1, n_negatives),
    std::max(1, n_negatives)
  );

  parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<int>& sampled = workspace.sampled[static_cast<std::size_t>(thread_id)];
    std::vector<double>& q_values = workspace.q_values[static_cast<std::size_t>(thread_id)];

    for (int i = begin; i < end; ++i) {
      const int global_i = query_begin + i;
      const std::size_t ib = static_cast<std::size_t>(i) * 2u;
      const double yi0 = y[ib];
      const double yi1 = y[ib + 1u];
      double g0 = 0.0;
      double g1 = 0.0;
      double sum_q = DBL_MIN;
      std::uint32_t rng_state = mix_uint32(
        static_cast<std::uint32_t>(seed) ^
          (static_cast<std::uint32_t>(global_i + 1) * 0x9e3779b9u)
      );

      if (exact_repulsion) {
        for (int ref = 0; ref < n_reference; ++ref) {
          const std::size_t rb = static_cast<std::size_t>(ref) * 2u;
          const double dx = yi0 - reference_layout[rb];
          const double dy = yi1 - reference_layout[rb + 1u];
          const double q = 1.0 / (1.0 + dx * dx + dy * dy);
          q_values[static_cast<std::size_t>(ref)] = q;
          sum_q += q;
        }
        const double inv_sum_q = 1.0 / sum_q;
        for (int ref = 0; ref < n_reference; ++ref) {
          const std::size_t rb = static_cast<std::size_t>(ref) * 2u;
          const double q = q_values[static_cast<std::size_t>(ref)];
          const double coeff = -(q * q) * inv_sum_q;
          g0 += coeff * (yi0 - reference_layout[rb]);
          g1 += coeff * (yi1 - reference_layout[rb + 1u]);
        }
      } else {
        for (int m = 0; m < n_negatives; ++m) {
          const int ref = uniform_index(rng_state, n_reference);
          sampled[static_cast<std::size_t>(m)] = ref;
          const std::size_t rb = static_cast<std::size_t>(ref) * 2u;
          const double dx = yi0 - reference_layout[rb];
          const double dy = yi1 - reference_layout[rb + 1u];
          const double q = 1.0 / (1.0 + dx * dx + dy * dy);
          q_values[static_cast<std::size_t>(m)] = q;
          sum_q += q;
        }
        const double inv_sum_q = 1.0 / sum_q;
        for (int m = 0; m < n_negatives; ++m) {
          const int ref = sampled[static_cast<std::size_t>(m)];
          const std::size_t rb = static_cast<std::size_t>(ref) * 2u;
          const double q = q_values[static_cast<std::size_t>(m)];
          const double coeff = -(q * q) * inv_sum_q;
          g0 += coeff * (yi0 - reference_layout[rb]);
          g1 += coeff * (yi1 - reference_layout[rb + 1u]);
        }
      }

      const std::size_t p_base = static_cast<std::size_t>(i) * k;
      for (int j = 0; j < k; ++j) {
        const int ref = indices(global_i, j) - offset;
        const std::size_t rb = static_cast<std::size_t>(ref) * 2u;
        const double dx = yi0 - reference_layout[rb];
        const double dy = yi1 - reference_layout[rb + 1u];
        const double q = 1.0 / (1.0 + dx * dx + dy * dy);
        const double coeff = exaggeration * probabilities[p_base + j] * q;
        g0 += coeff * dx;
        g1 += coeff * dy;
      }
      grad[ib] = g0;
      grad[ib + 1u] = g1;
    }
  });
}

} // namespace

// Exact post-fit KL diagnostic. This is intentionally separate from the
// production optimizer timing because evaluating the dense low-dimensional
// normalization is O(n^2).
// [[Rcpp::export]]
double opentsne_kl_diagnostic_cpp(IntegerMatrix indices,
                                  SEXP distances,
                                  SEXP layout,
                                  double perplexity,
                                  int n_threads) {
  const std::pair<int, int> knn_dims = distance_sexp_dims(distances);
  const std::pair<int, int> layout_dims = distance_sexp_dims(layout);
  if (indices.nrow() != knn_dims.first || indices.ncol() != knn_dims.second) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  if (layout_dims.first != indices.nrow() || layout_dims.second < 1 ||
      layout_dims.second > 3) {
    Rcpp::stop("`layout` must have one row per KNN row and one to three columns.");
  }
  if (!std::isfinite(perplexity) || perplexity <= 0.0 ||
      perplexity > static_cast<double>(indices.ncol())) {
    Rcpp::stop("`perplexity` must be positive and no larger than the KNN width.");
  }
  const int threads = resolve_threads(n_threads, indices.nrow());
  ParallelExecutor parallel_executor(threads);
  ParallelExecutorScope parallel_scope(&parallel_executor);
  std::vector<float> distance_values = copy_distances_float_sexp(distances, threads);
  SparseProbabilitiesF probabilities = build_tsne_probabilities_float(
    indices,
    distance_values,
    perplexity,
    threads
  );
  std::vector<float> y = copy_distances_float_sexp(layout, threads);
  return evaluate_kl_f(
    probabilities,
    y,
    indices.nrow(),
    layout_dims.second,
    threads
  );
}

// Lower-level validation surface for the native float32 t-SNE force kernels.
// This is intentionally internal: it exposes implementation components for
// regression tests and release validation, not a second embedding API.
// [[Rcpp::export]]
List opentsne_force_diagnostic_cpp(IntegerMatrix indices,
                                   SEXP distances,
                                   SEXP layout,
                                   double perplexity,
                                   double exaggeration,
                                   int grid_size,
                                   int n_threads) {
  const std::pair<int, int> knn_dims = distance_sexp_dims(distances);
  const std::pair<int, int> layout_dims = distance_sexp_dims(layout);
  if (indices.nrow() != knn_dims.first || indices.ncol() != knn_dims.second) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  if (layout_dims.first != indices.nrow() || layout_dims.second != 2) {
    Rcpp::stop("`layout` must have one row per KNN row and exactly two columns.");
  }
  if (!std::isfinite(perplexity) || perplexity <= 0.0 ||
      perplexity > static_cast<double>(indices.ncol())) {
    Rcpp::stop("`perplexity` must be positive and no larger than the KNN width.");
  }
  if (!std::isfinite(exaggeration) || exaggeration <= 0.0) {
    Rcpp::stop("`exaggeration` must be positive and finite.");
  }
  if (grid_size < 32 || grid_size > 512 ||
      (grid_size & (grid_size - 1)) != 0) {
    Rcpp::stop("`grid_size` must be a power of two between 32 and 512.");
  }

  const int n = indices.nrow();
  const int dims = 2;
  const int threads = resolve_threads(n_threads, n);
  ParallelExecutor parallel_executor(threads);
  ParallelExecutorScope parallel_scope(&parallel_executor);
  std::vector<float> distance_values = copy_distances_float_sexp(distances, threads);
  SparseProbabilitiesF probabilities = build_tsne_probabilities_float(
    indices, distance_values, perplexity, threads
  );
  std::vector<float> y = copy_distances_float_sexp(layout, threads);
  if (!std::all_of(y.begin(), y.end(), [](const float value) {
        return std::isfinite(value);
      })) {
    Rcpp::stop("`layout` must contain only finite coordinates.");
  }

  std::vector<float> attractive(y.size(), 0.0f);
  add_sparse_attractive_gradient_f(
    probabilities, y, n, dims, static_cast<float>(exaggeration), threads,
    attractive
  );

  std::vector<float> exact_total(y.size(), 0.0f);
  compute_gradient_pair_symmetric_f(
    probabilities, y, n, dims, static_cast<float>(exaggeration), threads,
    exact_total
  );
  std::vector<float> exact_repulsive(y.size(), 0.0f);
  for (std::size_t i = 0; i < y.size(); ++i) {
    exact_repulsive[i] = exact_total[i] - attractive[i];
  }

  FftGridWorkspaceT<float> fft_workspace;
  std::vector<float> fft_total(y.size(), 0.0f);
  compute_gradient_fft_grid_f(
    probabilities, y, n, dims, static_cast<float>(exaggeration), threads,
    &fft_workspace, fft_total, grid_size
  );
  std::vector<float> fft_repulsive(y.size(), 0.0f);
  for (std::size_t i = 0; i < y.size(); ++i) {
    fft_repulsive[i] = fft_total[i] - attractive[i];
  }

  auto as_matrix = [&](const std::vector<float>& values, const float scale) {
    NumericMatrix out(n, dims);
    for (int row = 0; row < n; ++row) {
      for (int column = 0; column < dims; ++column) {
        out(row, column) = static_cast<double>(
          values[static_cast<std::size_t>(row) * dims + column] * scale
        );
      }
    }
    return out;
  };
  auto l2_norm = [](const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) {
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::sqrt(sum);
  };

  IntegerVector row_ptr(probabilities.row_ptr.begin(), probabilities.row_ptr.end());
  IntegerVector col(probabilities.col.size());
  NumericVector weight(probabilities.val.size());
  for (std::size_t i = 0; i < probabilities.col.size(); ++i) {
    col[static_cast<R_xlen_t>(i)] = probabilities.col[i] + 1;
    weight[static_cast<R_xlen_t>(i)] = static_cast<double>(probabilities.val[i]);
  }

  return List::create(
    Rcpp::Named("attractive_force") = as_matrix(attractive, 1.0f),
    Rcpp::Named("repulsive_force_exact") = as_matrix(exact_repulsive, 1.0f),
    Rcpp::Named("optimizer_gradient_exact") = as_matrix(exact_total, 1.0f),
    Rcpp::Named("objective_gradient_exact") = as_matrix(exact_total, 4.0f),
    Rcpp::Named("repulsive_force_fft") = as_matrix(fft_repulsive, 1.0f),
    Rcpp::Named("optimizer_gradient_fft") = as_matrix(fft_total, 1.0f),
    Rcpp::Named("objective_gradient_fft") = as_matrix(fft_total, 4.0f),
    Rcpp::Named("sum_q") = static_cast<double>(compute_sum_q_f(y, n, dims, threads)),
    Rcpp::Named("kl") = evaluate_kl_f(probabilities, y, n, dims, threads),
    Rcpp::Named("attractive_norm") = l2_norm(attractive),
    Rcpp::Named("repulsive_exact_norm") = l2_norm(exact_repulsive),
    Rcpp::Named("repulsive_fft_norm") = l2_norm(fft_repulsive),
    Rcpp::Named("affinity_row_ptr0") = row_ptr,
    Rcpp::Named("affinity_col1") = col,
    Rcpp::Named("affinity_weight") = weight,
    Rcpp::Named("grid_size") = grid_size,
    Rcpp::Named("perplexity") = perplexity,
    Rcpp::Named("exaggeration") = exaggeration,
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("objective_gradient_scale") = 4.0
  );
}

// [[Rcpp::export]]
List tsne_auto_parameters_cpp(const int n,
                              const int k,
                              const double perplexity,
                              const bool perplexity_missing,
                              const std::string backend,
                              const std::string negative_gradient_method) {
  if (n < 2) Rcpp::stop("`n` must be at least 2.");
  if (k < 1) Rcpp::stop("`k` must be positive.");

  const int max_perplexity_n = std::max(1, (n - 1) / 3);
  const int max_perplexity_k = std::max(1, k);
  double resolved_perplexity = perplexity_missing || !std::isfinite(perplexity) || perplexity <= 0.0 ?
    static_cast<double>(std::min(30, std::min(max_perplexity_n, max_perplexity_k))) :
    perplexity;
  const int max_resolved_perplexity = perplexity_missing ?
    std::min(max_perplexity_n, max_perplexity_k) :
    max_perplexity_n;
  resolved_perplexity = std::max(
    1.0,
    std::min(resolved_perplexity, static_cast<double>(max_resolved_perplexity))
  );

  const double early_exaggeration = 12.0;
  const int needed_k = std::max(
    1,
    std::min(n - 1, static_cast<int>(std::ceil(resolved_perplexity)))
  );
  const bool kld_auto_stop = false;

  return List::create(
    Rcpp::Named("perplexity") = resolved_perplexity,
    Rcpp::Named("n_neighbors") = needed_k,
    Rcpp::Named("early_exaggeration") = early_exaggeration,
    Rcpp::Named("exaggeration") = 1.0,
    Rcpp::Named("learning_rate") = static_cast<double>(n) / early_exaggeration,
    Rcpp::Named("early_exaggeration_iter") = 250,
    Rcpp::Named("n_iter") = 500,
    Rcpp::Named("auto_kld_stop") = kld_auto_stop,
    Rcpp::Named("auto_iter_end") = 5000.0,
    Rcpp::Named("auto_iter_buffer_ee") = 15L,
    Rcpp::Named("auto_iter_buffer_run") = 15L,
    Rcpp::Named("auto_iter_pollrate_ee") = 3L,
    Rcpp::Named("auto_iter_pollrate_run") = 5L,
    Rcpp::Named("auto_iter_ee_switch_buffer") = 2L,
    Rcpp::Named("rule") = kld_auto_stop ?
      "opt_sne_kld_sensor" :
      "opt_sne_learning_rate_fixed_iterations_no_expensive_kld_polling"
  );
}

// [[Rcpp::export]]
List knn_tsne_opentsne_float_cpp(IntegerMatrix indices,
                                 SEXP distances,
                                 NumericMatrix y_init,
                                 bool init,
                                 int n_components,
                                 double perplexity,
                                 double theta,
                                 int early_exaggeration_iter,
                                 int n_iter,
                                 double early_exaggeration,
                                 double exaggeration,
                                 double learning_rate,
                                 bool learning_rate_auto,
                                 double initial_momentum,
                                 double final_momentum,
                                 double min_gain,
                                 double max_step_norm,
                                 std::string negative_gradient_method,
                                 int n_threads,
                                 int seed,
                                 bool verbose,
                                 bool record_costs,
                                 bool auto_config,
                                 double auto_iter_end) {
  const std::pair<int, int> dims_in = distance_sexp_dims(distances);
  if (indices.nrow() != dims_in.first || indices.ncol() != dims_in.second) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 2 || k < 1) Rcpp::stop("KNN input must have at least two rows and one neighbor column.");
  if (n - 1 < 3.0 * perplexity) Rcpp::stop("perplexity is too large for the number of samples.");
  if (n_components < 1 || n_components > 3) Rcpp::stop("`n_components` must be 1, 2, or 3 for t-SNE.");
  if (early_exaggeration_iter < 0 || n_iter < 0) Rcpp::stop("iteration counts must be non-negative.");
  if (early_exaggeration_iter + n_iter < 1) Rcpp::stop("at least one optimization iteration is required.");
  if (learning_rate <= 0.0 && !learning_rate_auto) Rcpp::stop("`learning_rate` must be positive or automatic.");
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0) Rcpp::stop("exaggeration values must be positive.");
  if (initial_momentum < 0.0 || final_momentum < 0.0) Rcpp::stop("momentum values must be non-negative.");
  if (min_gain <= 0.0) Rcpp::stop("`min_gain` must be positive.");
  if (theta < 0.0 || theta > 1.0) Rcpp::stop("`theta` must lie in [0, 1].");

  const int threads = resolve_threads(n_threads, n);
  const int requested_threads = n_threads;
  ParallelExecutor parallel_executor(threads);
  ParallelExecutorScope parallel_scope(&parallel_executor);
  if (verbose) {
    Rcpp::Rcout << "fastEmbedR float32 t-SNE from KNN: n=" << n
                << ", k=" << k
                << ", perplexity=" << perplexity
                << ", threads=" << threads << "\n";
  }

  const auto native_start = std::chrono::steady_clock::now();
  std::vector<float> distance_values = copy_distances_float_sexp(distances, threads);
  SparseProbabilitiesF p = build_tsne_probabilities_float(indices, distance_values, perplexity, threads);
  std::vector<float>().swap(distance_values);
  const auto affinity_end = std::chrono::steady_clock::now();

  const std::string repulsion_mode = tsne_repulsion_mode(n, theta, negative_gradient_method);
  std::string optimizer_name = repulsion_mode == "fft_grid" ?
    "opentsne_fitsne_fft_grid_sparse_knn_float32" :
    "opentsne_exact_sparse_knn_float32";

  std::vector<float> y(static_cast<std::size_t>(n) * n_components);
  if (init) {
    if (y_init.nrow() != n || y_init.ncol() != n_components) {
      Rcpp::stop("`Y_init` has the wrong shape.");
    }
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < n_components; ++d) {
        y[static_cast<std::size_t>(i) * n_components + d] =
          static_cast<float>(y_init(i, d));
      }
    }
  } else {
    const unsigned int resolved_seed = seed == NA_INTEGER ?
      5489u :
      static_cast<unsigned int>(seed);
    std::mt19937 rng(resolved_seed);
    std::normal_distribution<float> normal(0.0f, 1.0e-4f);
    for (float& value : y) value = normal(rng);
  }
  zero_mean_f(y, n, n_components);

  std::vector<float> grad(y.size(), 0.0f);
  std::vector<float> update(y.size(), 0.0f);
  std::vector<float> gains(y.size(), 1.0f);
  FftGridWorkspaceT<float> fft_workspace;
  const int requested_total_iter = early_exaggeration_iter + n_iter;
  const bool auto_kld_stop = auto_config && n <= 5000;
  const bool should_record_costs = record_costs || verbose;
  NumericVector iter_costs(should_record_costs ?
    static_cast<int>(std::ceil(static_cast<double>(requested_total_iter) / 50.0)) :
    0);
  IntegerVector itercost_iterations(should_record_costs ? iter_costs.size() : 0);
  int cost_index = 0;
  double auto_prev_error = std::numeric_limits<double>::infinity();
  double auto_prev_rc = std::numeric_limits<double>::infinity();
  bool auto_prev_valid = false;
  int auto_ee_switch_buffer = 2;
  std::string auto_stop_reason = auto_kld_stop ?
    "max_iterations_without_kld_plateau" :
    "disabled";
  if (!std::isfinite(auto_iter_end) || auto_iter_end <= 0.0) auto_iter_end = 5000.0;

  auto run_phase = [&](const int phase_iter,
                       const double phase_exaggeration,
                       const double phase_momentum,
                       const char* phase_name,
                       int& completed_iter) -> int {
    if (phase_iter <= 0) return 0;
    const bool early_phase = std::string(phase_name) == "early_exaggeration";
    const int auto_pollrate = early_phase ? 3 : 5;
    const int auto_buffer = early_phase ? 15 : 15;
    const float phase_lr = static_cast<float>(learning_rate_auto ?
      static_cast<double>(n) / std::max(phase_exaggeration, DBL_MIN) :
      learning_rate);
    const float phase_exag_f = static_cast<float>(phase_exaggeration);
    const float phase_momentum_f = static_cast<float>(phase_momentum);
    const float min_gain_f = static_cast<float>(min_gain);
    const float max_step_norm_f = std::isfinite(max_step_norm) ?
      static_cast<float>(max_step_norm) :
      std::numeric_limits<float>::quiet_NaN();
    if (verbose) {
      Rcpp::Rcout << "t-SNE float32 phase " << phase_name
                  << ": iterations=" << phase_iter
                  << ", exaggeration=" << phase_exaggeration
                  << ", learning_rate=" << phase_lr
                  << ", momentum=" << phase_momentum << "\n";
    }
    int phase_completed = 0;
    for (int iter = 0; iter < phase_iter; ++iter) {
      if (((completed_iter + iter) & 7) == 0) Rcpp::checkUserInterrupt();
      compute_gradient_f(
        p, y, n, n_components, phase_exag_f, threads, repulsion_mode, &fft_workspace, grad
      );
      apply_open_tsne_update_f(
        y, update, gains, grad, n, n_components, phase_lr, phase_momentum_f,
        min_gain_f, max_step_norm_f, threads
      );
      zero_mean_f(y, n, n_components);

      const int global_iter = completed_iter + iter + 1;
      const bool need_auto_error = auto_kld_stop && ((iter + 1) % auto_pollrate == 0);
      const bool need_record_error = should_record_costs &&
        ((global_iter % 50 == 0) || global_iter == requested_total_iter);
      if (need_auto_error || need_record_error) {
        const double kl = evaluate_kl_f(p, y, n, n_components, threads);
        if (need_record_error && cost_index < iter_costs.size()) {
          iter_costs[cost_index] = kl;
          itercost_iterations[cost_index] = global_iter;
          ++cost_index;
        }
        if (verbose) Rcpp::Rcout << "Iteration " << global_iter << ": error is " << kl << "\n";
        if (need_auto_error) {
          const double error_diff = auto_prev_error - kl;
          const double error_rc = std::isfinite(auto_prev_error) && auto_prev_error > 0.0 ?
            100.0 * error_diff / auto_prev_error :
            std::numeric_limits<double>::infinity();
          if (auto_prev_valid) {
            if (early_phase) {
              if (error_rc < auto_prev_rc && iter + 1 > auto_buffer) {
                if (auto_ee_switch_buffer < 1) {
                  auto_stop_reason = "early_exaggeration_stopped_at_local_max_kld_relative_change";
                  ++phase_completed;
                  completed_iter += phase_completed;
                  return phase_completed;
                }
                --auto_ee_switch_buffer;
              }
            } else if (iter + 1 > auto_buffer &&
                       std::fabs(error_diff) / static_cast<double>(auto_pollrate) < kl / auto_iter_end) {
              auto_stop_reason = "normal_phase_stopped_at_kld_improvement_threshold";
              ++phase_completed;
              completed_iter += phase_completed;
              return phase_completed;
            }
          }
          auto_prev_error = kl;
          auto_prev_rc = error_rc;
          auto_prev_valid = true;
        }
      }
      ++phase_completed;
    }
    completed_iter += phase_iter;
    return phase_completed;
  };

  int completed_iter = 0;
  const int actual_early_iter = run_phase(
    early_exaggeration_iter, early_exaggeration, initial_momentum,
    "early_exaggeration", completed_iter
  );
  const int actual_normal_iter = run_phase(
    n_iter, exaggeration, final_momentum, "normal", completed_iter
  );

  NumericVector row_costs;
  if (should_record_costs) {
    std::vector<double> costs(static_cast<std::size_t>(n), 0.0);
    evaluate_kl_f(p, y, n, n_components, threads, &costs);
    row_costs = NumericVector(costs.begin(), costs.end());
  } else {
    row_costs = NumericVector(0);
  }

  NumericMatrix layout(n, n_components);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_components; ++d) {
      layout(i, d) = static_cast<double>(y[static_cast<std::size_t>(i) * n_components + d]);
    }
  }
  const auto native_end = std::chrono::steady_clock::now();
  const double affinity_sec =
    std::chrono::duration<double>(affinity_end - native_start).count();
  const double optimization_sec =
    std::chrono::duration<double>(native_end - affinity_end).count();

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("costs") = row_costs,
    Rcpp::Named("itercosts") = iter_costs,
    Rcpp::Named("itercost_iterations") = itercost_iterations,
    Rcpp::Named("optimizer") = optimizer_name,
    Rcpp::Named("repulsion") = repulsion_mode,
    Rcpp::Named("fft_grid_size") = repulsion_mode == "fft_grid" ?
      tsne_fft_grid_size(n) : NA_INTEGER,
    Rcpp::Named("theta_requested") = theta,
    Rcpp::Named("n_threads") = threads,
    Rcpp::Named("n_threads_requested") = requested_threads,
    Rcpp::Named("affinity_elapsed_sec") = affinity_sec,
    Rcpp::Named("optimization_elapsed_sec") = optimization_sec,
    Rcpp::Named("native_total_elapsed_sec") = affinity_sec + optimization_sec,
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("learning_rate") = learning_rate_auto ? NA_REAL : learning_rate,
    Rcpp::Named("learning_rate_early") = learning_rate_auto ?
      static_cast<double>(n) / std::max(early_exaggeration, DBL_MIN) :
      learning_rate,
    Rcpp::Named("learning_rate_normal") = learning_rate_auto ?
      static_cast<double>(n) / std::max(exaggeration, DBL_MIN) :
      learning_rate,
    Rcpp::Named("auto_config") = auto_config,
    Rcpp::Named("auto_kld_stop") = auto_kld_stop,
    Rcpp::Named("auto_stop_reason") = auto_stop_reason,
    Rcpp::Named("early_exaggeration_iter_actual") = actual_early_iter,
    Rcpp::Named("n_iter_actual") = actual_normal_iter,
    Rcpp::Named("max_iter_actual") = completed_iter,
    Rcpp::Named("auto_iter_end") = auto_iter_end
  );
}

// [[Rcpp::export]]
List transform_tsne_cpp(NumericMatrix reference_layout,
                        IntegerMatrix indices,
                        NumericMatrix distances,
                        NumericMatrix y_init,
                        bool init,
                        std::string initialization,
                        double perplexity,
                        int n_iter,
                        int early_exaggeration_iter,
                        double learning_rate,
                        double early_exaggeration,
                        double exaggeration,
                        double initial_momentum,
                        double final_momentum,
                        double max_grad_norm,
                        double max_step_norm,
                        int n_negatives,
                        int exact_repulsion_threshold,
                        int n_threads,
                        int seed,
                        bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  const int n_reference = reference_layout.nrow();
  const int dims = reference_layout.ncol();
  if (n_reference < 1 || dims < 1) Rcpp::stop("`reference_layout` must be a non-empty matrix.");
  if (n_query < 1 || k < 1) Rcpp::stop("KNN input must have at least one query row and one neighbor column.");
  if (perplexity <= 0.0) Rcpp::stop("`perplexity` must be positive.");
  if (n_iter < 0 || early_exaggeration_iter < 0) Rcpp::stop("iteration counts must be non-negative.");
  if (n_iter + early_exaggeration_iter < 1) Rcpp::stop("at least one transform iteration is required.");
  if (learning_rate <= 0.0) Rcpp::stop("`learning_rate` must be positive.");
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0) Rcpp::stop("exaggeration values must be positive.");
  if (initial_momentum < 0.0 || final_momentum < 0.0) Rcpp::stop("momentum values must be non-negative.");
  if (n_negatives < 1) Rcpp::stop("`n_negatives` must be positive.");
  if (exact_repulsion_threshold < 1) exact_repulsion_threshold = 1;
  if (max_grad_norm <= 0.0 || !std::isfinite(max_grad_norm)) {
    max_grad_norm = DBL_MAX;
  }
  if (max_step_norm <= 0.0 || !std::isfinite(max_step_norm)) {
    max_step_norm = DBL_MAX;
  }
  if (initialization != "median" && initialization != "weighted" && initialization != "random") {
    Rcpp::stop("`initialization` must be 'median', 'weighted', or 'random'.");
  }
  if (perplexity > static_cast<double>(k)) {
    Rcpp::warning("Transform perplexity exceeds the supplied KNN width.");
  }

  const int offset = resolve_reference_index_offset(indices, n_reference);
  for (int i = 0; i < n_query; ++i) {
    for (int j = 0; j < k; ++j) {
      const int ref = indices(i, j) - offset;
      if (ref < 0 || ref >= n_reference) Rcpp::stop("KNN indices are out of range for `reference_layout`.");
      const double d = distances(i, j);
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative.");
      }
    }
  }

  const int threads = resolve_threads(n_threads, n_query);
  ParallelExecutor parallel_executor(threads);
  ParallelExecutorScope parallel_scope(&parallel_executor);
  if (n_negatives > n_reference) n_negatives = n_reference;
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  if (verbose) {
    Rcpp::Rcout << "fastEmbedR t-SNE transform: queries=" << n_query
                << ", reference=" << n_reference
                << ", k=" << k
                << ", perplexity=" << perplexity
                << ", repulsion=" << (exact_repulsion ? "exact" : "sampled")
                << ", threads=" << threads << "\n";
  }

  if (init && (y_init.nrow() != n_query || y_init.ncol() != dims)) {
    Rcpp::stop("`Y_init` has the wrong shape.");
  }

  const int batch_size = tsne_transform_batch_size(n_query, k, dims);
  const int n_batches = (n_query + batch_size - 1) / batch_size;
  std::vector<double> probabilities(static_cast<std::size_t>(batch_size) * k, 0.0);
  std::vector<double> y(static_cast<std::size_t>(batch_size) * dims, 0.0);
  std::vector<double> grad(y.size(), 0.0);
  std::vector<double> update(y.size(), 0.0);
  std::vector<double> gains(y.size(), 1.0);
  TsneTransformGradientWorkspace gradient_workspace;
  std::vector<double> reference_layout_flat;
  if (dims == 2) {
    reference_layout_flat.resize(static_cast<std::size_t>(n_reference) * 2u);
    for (int ref = 0; ref < n_reference; ++ref) {
      reference_layout_flat[static_cast<std::size_t>(ref) * 2u] = reference_layout(ref, 0);
      reference_layout_flat[static_cast<std::size_t>(ref) * 2u + 1u] = reference_layout(ref, 1);
    }
  }
  const int total_iter = early_exaggeration_iter + n_iter;
  NumericMatrix layout(n_query, dims);

  for (int query_begin = 0; query_begin < n_query; query_begin += batch_size) {
    const int batch_n = std::min(batch_size, n_query - query_begin);
    const int batch_threads = resolve_threads(threads, batch_n);
    const std::size_t active_points = static_cast<std::size_t>(batch_n);
    const std::size_t active_layout = active_points * dims;
    const std::size_t active_graph = active_points * k;

    parallel_for(batch_n, batch_threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        compute_row_probabilities_flat(
          distances,
          query_begin + i,
          perplexity,
          probabilities.data() + static_cast<std::size_t>(i) * k
        );
      }
    });

    if (init) {
      parallel_for(batch_n, batch_threads, [&](const int begin, const int end, const int) {
        for (int i = begin; i < end; ++i) {
          const int global_i = query_begin + i;
          for (int d = 0; d < dims; ++d) {
            y[static_cast<std::size_t>(i) * dims + d] = y_init(global_i, d);
          }
        }
      });
    } else {
      initialize_tsne_transform(
        reference_layout,
        indices,
        distances,
        offset,
        initialization,
        seed,
        query_begin,
        batch_n,
        batch_threads,
        y
      );
    }
    std::fill(update.begin(), update.begin() + active_layout, 0.0);
    std::fill(gains.begin(), gains.begin() + active_layout, 1.0);
    std::fill(grad.begin(), grad.begin() + active_layout, 0.0);
    if (active_graph < probabilities.size()) {
      std::fill(probabilities.begin() + active_graph, probabilities.end(), 0.0);
    }

    for (int iter = 0; iter < total_iter; ++iter) {
      if ((iter & 7) == 0) Rcpp::checkUserInterrupt();
      const bool in_early = iter < early_exaggeration_iter;
      const double current_exaggeration = in_early ? early_exaggeration : exaggeration;
      const double current_momentum = in_early ? initial_momentum : final_momentum;
      const int iter_seed = (seed == NA_INTEGER ? 5489 : seed) + 65537 * (iter + 1);

      if (dims == 2) {
        compute_tsne_transform_gradient_2d_flat(
          reference_layout_flat,
          indices,
          probabilities,
          y,
          offset,
          query_begin,
          batch_n,
          current_exaggeration,
          n_negatives,
          exact_repulsion_threshold,
          batch_threads,
          iter_seed,
          gradient_workspace,
          grad
        );
      } else {
        compute_tsne_transform_gradient(
          reference_layout,
          indices,
          probabilities,
          y,
          offset,
          query_begin,
          batch_n,
          current_exaggeration,
          n_negatives,
          exact_repulsion_threshold,
          batch_threads,
          iter_seed,
          gradient_workspace,
          grad
        );
      }

      parallel_for(batch_n, batch_threads, [&](const int begin, const int end, const int) {
        for (int i = begin; i < end; ++i) {
          const std::size_t ib = static_cast<std::size_t>(i) * dims;
          double grad_norm_sq = 0.0;
          for (int d = 0; d < dims; ++d) {
            grad_norm_sq += grad[ib + d] * grad[ib + d];
          }
          if (grad_norm_sq > max_grad_norm * max_grad_norm) {
            const double scale = max_grad_norm / (std::sqrt(grad_norm_sq) + 1e-12);
            for (int d = 0; d < dims; ++d) grad[ib + d] *= scale;
          }

          for (int d = 0; d < dims; ++d) {
            const std::size_t index = ib + d;
            if (sign_tsne(update[index]) != sign_tsne(grad[index])) {
              gains[index] += 0.2;
            } else {
              gains[index] = gains[index] * 0.8 + 0.01;
            }
            if (gains[index] < 0.01) gains[index] = 0.01;
            update[index] = current_momentum * update[index] -
              learning_rate * gains[index] * grad[index];
          }

          double step_norm_sq = 0.0;
          for (int d = 0; d < dims; ++d) step_norm_sq += update[ib + d] * update[ib + d];
          if (step_norm_sq > max_step_norm * max_step_norm) {
            const double scale = max_step_norm / (std::sqrt(step_norm_sq) + 1e-12);
            for (int d = 0; d < dims; ++d) update[ib + d] *= scale;
          }
          for (int d = 0; d < dims; ++d) y[ib + d] += update[ib + d];
        }
      });
    }

    parallel_for(batch_n, batch_threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        const int global_i = query_begin + i;
        for (int d = 0; d < dims; ++d) {
          layout(global_i, d) = y[static_cast<std::size_t>(i) * dims + d];
        }
      }
    });
  }

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("optimizer") = "opentsne_style_fixed_reference_transform",
    Rcpp::Named("initialization") = initialization,
    Rcpp::Named("repulsion") = exact_repulsion ? "exact_reference" : "sampled_reference",
    Rcpp::Named("affinities") = "precomputed_query_conditional",
    Rcpp::Named("affinity_storage") = "flat_row_major_double",
    Rcpp::Named("transform_batch_size") = batch_size,
    Rcpp::Named("transform_batches") = n_batches,
    Rcpp::Named("n_negatives") = n_negatives,
    Rcpp::Named("n_threads") = threads
  );
}
