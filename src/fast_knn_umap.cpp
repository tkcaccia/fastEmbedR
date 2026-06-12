#include <Rcpp.h>
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
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
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

struct Edge {
  int head;
  int tail;
  float weight;
};

struct NeighborLookup {
  std::vector<int> offsets;
  std::vector<int> neighbors;

  bool empty() const {
    return neighbors.empty();
  }
};

struct Graph {
  std::vector<Edge> edges;
  NeighborLookup neighbor_lookup;
};

struct WeightedEdge {
  std::uint64_t key;
  float weight;
  std::uint8_t direction;
};

class ReusableBarrier {
 public:
  explicit ReusableBarrier(const int participants)
      : threshold_(participants),
        count_(participants),
        generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int generation = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&]() { return generation != generation_; });
  }

 private:
  const int threshold_;
  int count_;
  int generation_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

int effective_cpu_threads(const int n_threads, const int n_items) {
  if (n_items <= 1) return 1;
  return std::max(1, std::min(std::min(n_threads, 4), n_items));
}

template <typename Worker>
void parallel_for_chunks(const int n_items, const int n_threads, Worker worker) {
  const int threads = effective_cpu_threads(n_threads, n_items);
  if (threads == 1) {
    worker(0, n_items, 0);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads - 1));
  for (int t = 1; t < threads; ++t) {
    const int begin = static_cast<int>(
      static_cast<std::int64_t>(n_items) * static_cast<std::int64_t>(t) / threads
    );
    const int end = static_cast<int>(
      static_cast<std::int64_t>(n_items) * static_cast<std::int64_t>(t + 1) / threads
    );
    workers.emplace_back([&, begin, end, t]() {
      worker(begin, end, t);
    });
  }
  const int first_end = static_cast<int>(
    static_cast<std::int64_t>(n_items) / threads
  );
  worker(0, first_end, 0);
  for (auto& thread : workers) thread.join();
}

template <typename T>
void release_vector(std::vector<T>& values) {
  std::vector<T>().swap(values);
}

template <typename T>
void release_nested_vector(std::vector<std::vector<T>>& values) {
  std::vector<std::vector<T>>().swap(values);
}

std::uint64_t edge_key(const int a, const int b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
         static_cast<std::uint32_t>(b);
}

int key_head(const std::uint64_t key) {
  return static_cast<int>(key >> 32);
}

int key_tail(const std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffu);
}

double clip_value(const double x, const double lo, const double hi) {
  return std::max(lo, std::min(hi, x));
}

float clip4f(const float x) {
  return x < -4.0f ? -4.0f : (x > 4.0f ? 4.0f : x);
}

// Approximate pow used by uwot's fast_sgd path. uwot is GPL (>= 3), so any
// adapted optimizer details stay inside this GPL-compatible package.
double fast_precise_pow(double a, double b) {
  if (a <= 0.0) return 0.0;
  int e = static_cast<int>(b);
  union {
    double d;
    int x[2];
  } u = {a};
  u.x[1] = static_cast<int>(
    (b - static_cast<double>(e)) *
      static_cast<double>(u.x[1] - 1072632447) +
    1072632447.0
  );
  u.x[0] = 0;

  double r = 1.0;
  while (e) {
    if (e & 1) r *= a;
    a *= a;
    e >>= 1;
  }
  return r * u.d;
}

double umap_pow(const double x, const double b) {
  return fast_precise_pow(x, b);
}

bool weighted_edge_less(const WeightedEdge& a, const WeightedEdge& b) {
  return a.key < b.key;
}

struct FloatDistanceView {
  const float* values;
  int nrow;
  int ncol;
  int stride;

  const float* row_data(const int row) const {
    return values + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
  }

  float operator()(const int row, const int col) const {
    return row_data(row)[col];
  }
};

struct CsrGraphNative {
  std::vector<int> offsets;
  std::vector<int> neighbors;
  std::vector<float> weights;
  std::vector<float> epochs_per_sample;
  float max_weight = 0.0f;
};

struct CsrCandidateEdge {
  int head;
  int tail;
  float weight;
};

class DisjointSet {
 public:
  explicit DisjointSet(const int n)
      : parent_(static_cast<std::size_t>(n)),
        size_(static_cast<std::size_t>(n), 1) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(const int x) {
    int root = x;
    while (parent_[static_cast<std::size_t>(root)] != root) {
      root = parent_[static_cast<std::size_t>(root)];
    }
    int current = x;
    while (parent_[static_cast<std::size_t>(current)] != current) {
      const int next = parent_[static_cast<std::size_t>(current)];
      parent_[static_cast<std::size_t>(current)] = root;
      current = next;
    }
    return root;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (size_[static_cast<std::size_t>(a)] < size_[static_cast<std::size_t>(b)]) {
      std::swap(a, b);
    }
    parent_[static_cast<std::size_t>(b)] = a;
    size_[static_cast<std::size_t>(a)] += size_[static_cast<std::size_t>(b)];
  }

 private:
  std::vector<int> parent_;
  std::vector<int> size_;
};

std::vector<float> copy_distances_float(const NumericMatrix& distances,
                                        const int n_threads,
                                        const int col_start = 0,
                                        int n_cols = -1) {
  const int n = distances.nrow();
  const int matrix_k = distances.ncol();
  if (n_cols < 0) n_cols = matrix_k - col_start;
  if (col_start < 0 || n_cols < 0 || col_start + n_cols > matrix_k) {
    Rcpp::stop("invalid KNN distance column range");
  }
  const int k = n_cols;
  const std::size_t size = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
  std::vector<float> out(size);
  const double* src = distances.begin();

  const int threads = effective_cpu_threads(n_threads, n);
  auto worker = [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      float* dst = out.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
      for (int col = 0; col < k; ++col) {
        const int src_col = col + col_start;
        dst[col] = static_cast<float>(
          src[static_cast<std::size_t>(src_col) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(row)]
        );
      }
    }
  };

  if (threads == 1 || n < 2048) {
    worker(0, n, 0);
    return out;
  }
  parallel_for_chunks(n, threads, worker);
  return out;
}

std::pair<double, double> find_ab_params(const double spread, const double min_dist) {
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.01) < 1e-12) {
    return {1.895605865596314, 0.8006377738365004};
  }
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.1) < 1e-12) {
    return {1.5769434601962196, 0.8950608781227859};
  }
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.5) < 1e-12) {
    return {0.5830300199018228, 1.3341669931033755};
  }

  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(300);
  ys.reserve(300);
  for (int i = 0; i < 300; ++i) {
    const double x = (spread * 3.0) * static_cast<double>(i) / 299.0;
    xs.push_back(x);
    ys.push_back(x < min_dist ? 1.0 : std::exp(-(x - min_dist) / spread));
  }

  double best_a = 1.5769434601962196;
  double best_b = 0.8950608781227859;
  double best_loss = std::numeric_limits<double>::infinity();

  for (double loga = -4.0; loga <= 4.0001; loga += 0.2) {
    for (double b = 0.25; b <= 2.0001; b += 0.05) {
      const double a = std::exp(loga);
      double loss = 0.0;
      for (std::size_t i = 0; i < xs.size(); ++i) {
        const double x2b = std::pow(xs[i], 2.0 * b);
        const double yhat = 1.0 / (1.0 + a * x2b);
        const double e = yhat - ys[i];
        loss += e * e;
      }
      if (loss < best_loss) {
        best_loss = loss;
        best_a = a;
        best_b = b;
      }
    }
  }

  for (int iter = 0; iter < 80; ++iter) {
    double ga = 0.0;
    double gb = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      const double x = std::max(xs[i], 1e-6);
      const double x2b = std::pow(x, 2.0 * best_b);
      const double denom = 1.0 + best_a * x2b;
      const double yhat = 1.0 / denom;
      const double e = yhat - ys[i];
      ga += e * (-x2b / (denom * denom));
      gb += e * (-(best_a * x2b * 2.0 * std::log(x)) / (denom * denom));
    }
    best_a = std::max(1e-4, best_a - 0.01 * ga);
    best_b = std::max(0.1, best_b - 0.01 * gb);
  }

  return {best_a, best_b};
}

bool lookup_contains(const NeighborLookup& lookup, const int row, const int value) {
  const int begin = lookup.offsets[static_cast<std::size_t>(row)];
  const int end = lookup.offsets[static_cast<std::size_t>(row + 1)];
  return std::binary_search(
    lookup.neighbors.begin() + begin,
    lookup.neighbors.begin() + end,
    value
  );
}

bool lookup_linked(const NeighborLookup& lookup, const int i, const int j) {
  return lookup_contains(lookup, i, j) || lookup_contains(lookup, j, i);
}

std::uint32_t mix_seed(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

int deterministic_vertex(const int n,
                         const int seed,
                         const int epoch,
                         const std::size_t edge,
                         const int sample) {
  std::uint32_t x = static_cast<std::uint32_t>(seed);
  x ^= static_cast<std::uint32_t>(epoch * 0x9e3779b9u);
  x ^= static_cast<std::uint32_t>((edge + 1u) * 0x85ebca6bu);
  x ^= static_cast<std::uint32_t>((sample + 1) * 0xc2b2ae35u);
  return static_cast<int>(mix_seed(x) % static_cast<std::uint32_t>(n));
}

// Three-component combined Tausworthe "taus88" generator used by uwot's
// non-PCG fast SGD path. The exact initial states in uwot come from R-side RNG
// factories; here we seed the same generator deterministically per epoch/window.
struct TauPrng {
  std::uint64_t state0;
  std::uint64_t state1;
  std::uint64_t state2;

  TauPrng(const std::uint64_t s0,
          const std::uint64_t s1,
          const std::uint64_t s2)
      : state0(s0),
        state1(s1 > 7u ? s1 : 8u),
        state2(s2 > 15u ? s2 : 16u) {}

  std::uint32_t next() {
    constexpr std::uint64_t magic0 = 4294967294ull;
    constexpr std::uint64_t magic1 = 4294967288ull;
    constexpr std::uint64_t magic2 = 4294967280ull;
    state0 = (((state0 & magic0) << 12) & 0xffffffffull) ^
             ((((state0 << 13) & 0xffffffffull) ^ state0) >> 19);
    state1 = (((state1 & magic1) << 4) & 0xffffffffull) ^
             ((((state1 << 2) & 0xffffffffull) ^ state1) >> 25);
    state2 = (((state2 & magic2) << 17) & 0xffffffffull) ^
             ((((state2 << 3) & 0xffffffffull) ^ state2) >> 11);
    return static_cast<std::uint32_t>((state0 ^ state1 ^ state2) & 0xffffffffull);
  }

  int vertex(const int n) {
    return static_cast<int>(next() % static_cast<std::uint32_t>(n));
  }
};

TauPrng make_tau_prng(const int seed,
                      const int epoch,
                      const std::size_t window_end,
                      const int thread_id) {
  const std::uint32_t base = mix_seed(
    static_cast<std::uint32_t>(seed) ^
    static_cast<std::uint32_t>((epoch + 1) * 0x9e3779b9u) ^
    static_cast<std::uint32_t>((thread_id + 1) * 0x85ebca6bu) ^
    static_cast<std::uint32_t>((window_end + 1u) * 0xc2b2ae35u)
  );
  const std::uint32_t s0 = mix_seed(base ^ 0xa341316cu);
  const std::uint32_t s1 = mix_seed(base ^ 0xc8013ea4u);
  const std::uint32_t s2 = mix_seed(base ^ 0xad90777du);
  return TauPrng(s0, s1, s2);
}

int deterministic_non_neighbor(const int n,
                               const NeighborLookup& lookup,
                               const int anchor,
                               const int avoid,
                               const int seed,
                               const int epoch,
                               const std::size_t edge,
                               const int sample) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    const int candidate = deterministic_vertex(
      n, seed + attempt, epoch, edge, sample + attempt * 17
    );
    if (candidate != anchor && candidate != avoid && !lookup_linked(lookup, anchor, candidate)) {
      return candidate;
    }
  }

  const int start = deterministic_vertex(n, seed + 7919, epoch, edge, sample + 131);
  for (int step = 0; step < n; ++step) {
    const int candidate = (start + step) % n;
    if (candidate != anchor && candidate != avoid && !lookup_linked(lookup, anchor, candidate)) {
      return candidate;
    }
  }

  for (int candidate = 0; candidate < n; ++candidate) {
    if (candidate != anchor && candidate != avoid) return candidate;
  }
  return anchor == 0 ? 1 % n : 0;
}

int infer_csr_index_offset(const int n, const IntegerVector& neighbors) {
  const int nnz = neighbors.size();
  if (nnz == 0) return 0;
  int min_neighbor = std::numeric_limits<int>::max();
  int max_neighbor = std::numeric_limits<int>::min();
  for (int i = 0; i < nnz; ++i) {
    min_neighbor = std::min(min_neighbor, neighbors[i]);
    max_neighbor = std::max(max_neighbor, neighbors[i]);
  }
  return (min_neighbor >= 1 && max_neighbor <= n) ? 1 : 0;
}

int validate_csr_inputs(const IntegerVector& offsets,
                        const IntegerVector& neighbors,
                        const NumericVector& weights) {
  if (offsets.size() < 2) Rcpp::stop("CSR offsets must have length at least two");
  if (neighbors.size() != weights.size()) {
    Rcpp::stop("CSR neighbors and weights must have the same length");
  }

  const int n = offsets.size() - 1;
  int previous = offsets[0];
  if (previous != 0) Rcpp::stop("First CSR offset must be zero");
  for (int i = 1; i <= n; ++i) {
    const int value = offsets[i];
    if (value < previous) Rcpp::stop("CSR offsets must be non-decreasing");
    previous = value;
  }
  const int nnz = offsets[n];
  if (nnz != neighbors.size()) Rcpp::stop("Last CSR offset must equal length of neighbors");

  const int index_offset = infer_csr_index_offset(n, neighbors);
  for (int row = 0; row < n; ++row) {
    const int begin = offsets[row];
    const int end = offsets[row + 1];
    int last = -1;
    for (int pos = begin; pos < end; ++pos) {
      const int nb = neighbors[pos] - index_offset;
      if (nb < 0 || nb >= n) Rcpp::stop("CSR neighbor index out of range");
      if (nb == row) Rcpp::stop("CSR graph must not contain self-neighbors");
      if (nb < last) Rcpp::stop("CSR neighbors must be sorted within each row");
      last = nb;
      const double w = weights[pos];
      if (!std::isfinite(w) || w <= 0.0) {
        Rcpp::stop("CSR weights must be positive and finite");
      }
    }
  }
  return index_offset;
}

template <typename OffsetVec, typename NeighborVec>
bool csr_contains(const OffsetVec& offsets,
                  const NeighborVec& neighbors,
                  const int index_offset,
                  const int row,
                  const int value) {
  const int target = value + index_offset;
  int lo = offsets[row];
  int hi = offsets[row + 1];
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int current = neighbors[mid];
    if (current < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < offsets[row + 1] && neighbors[lo] == target;
}

template <typename OffsetVec, typename NeighborVec>
bool csr_linked(const OffsetVec& offsets,
                const NeighborVec& neighbors,
                const int index_offset,
                const int i,
                const int j) {
  return csr_contains(offsets, neighbors, index_offset, i, j) ||
         csr_contains(offsets, neighbors, index_offset, j, i);
}

template <typename OffsetVec, typename NeighborVec, typename WeightVec>
float csr_lookup_weight(const OffsetVec& offsets,
                        const NeighborVec& neighbors,
                        const WeightVec& weights,
                        const int row,
                        const int value) {
  int lo = offsets[row];
  int hi = offsets[row + 1];
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int current = neighbors[mid];
    if (current < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < offsets[row + 1] && neighbors[lo] == value) {
    return static_cast<float>(weights[lo]);
  }
  return 0.0f;
}

template <typename OffsetVec, typename NeighborVec>
int deterministic_non_neighbor_csr(const int n,
                                   const OffsetVec& offsets,
                                   const NeighborVec& neighbors,
                                   const int index_offset,
                                   const int anchor,
                                   const int avoid,
                                   const int seed,
                                   const int epoch,
                                   const std::size_t edge,
                                   const int sample) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    const int candidate = deterministic_vertex(
      n, seed + attempt, epoch, edge, sample + attempt * 17
    );
    if (candidate != anchor && candidate != avoid &&
        !csr_linked(offsets, neighbors, index_offset, anchor, candidate)) {
      return candidate;
    }
  }

  const int start = deterministic_vertex(n, seed + 7919, epoch, edge, sample + 131);
  for (int step = 0; step < n; ++step) {
    const int candidate = (start + step) % n;
    if (candidate != anchor && candidate != avoid &&
        !csr_linked(offsets, neighbors, index_offset, anchor, candidate)) {
      return candidate;
    }
  }

  for (int candidate = 0; candidate < n; ++candidate) {
    if (candidate != anchor && candidate != avoid) return candidate;
  }
  return anchor == 0 ? 1 % n : 0;
}

void smooth_knn_dist(const FloatDistanceView& distances,
                     std::vector<float>& sigmas,
                     std::vector<float>& rhos,
                     const int n_threads) {
  const int n = distances.nrow;
  const int k = distances.ncol;
  const double target = std::log2(static_cast<double>(k));
  const double tol = 1.0e-5;
  const double min_k_dist_scale = 1.0e-3;
  sigmas.assign(static_cast<std::size_t>(n), 1.0f);
  rhos.assign(static_cast<std::size_t>(n), 0.0f);

  const int threads = effective_cpu_threads(n_threads, n);
  std::vector<long double> thread_sums(static_cast<std::size_t>(threads), 0.0L);
  std::vector<std::size_t> thread_counts(static_cast<std::size_t>(threads), 0u);

  auto global_worker = [&](const int t, const int begin, const int end) {
    long double local_sum = 0.0L;
    std::size_t local_count = 0u;
    for (int i = begin; i < end; ++i) {
      const float* row_values = distances.row_data(i);
      for (int j = 0; j < k; ++j) {
        const float d = row_values[j];
        if (std::isfinite(d) && d >= 0.0f) {
          local_sum += static_cast<long double>(d);
          ++local_count;
        }
      }
    }
    thread_sums[static_cast<std::size_t>(t)] = local_sum;
    thread_counts[static_cast<std::size_t>(t)] = local_count;
  };

  if (threads == 1 || n < 2048) {
    global_worker(0, 0, n);
  } else {
    parallel_for_chunks(n, threads, [&](const int begin, const int end, const int t) {
      global_worker(t, begin, end);
    });
  }

  long double global_sum = 0.0L;
  std::size_t global_count = 0u;
  for (int t = 0; t < threads; ++t) {
    global_sum += thread_sums[static_cast<std::size_t>(t)];
    global_count += thread_counts[static_cast<std::size_t>(t)];
  }
  const double global_mean = global_count > 0u ?
    static_cast<double>(global_sum / static_cast<long double>(global_count)) :
    1.0;

  auto membership_sum = [&](const float* row_values, const double rho, const double sigma) {
    double psum = 0.0;
    const double safe_sigma = std::max(sigma, 1.0e-12);
    for (int j = 0; j < k; ++j) {
      const float raw = row_values[j];
      if (!std::isfinite(raw)) continue;
      const double d = static_cast<double>(raw) - rho;
      psum += d <= 0.0 ? 1.0 : std::exp(-d / safe_sigma);
    }
    return psum;
  };

  auto worker = [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const float* row_values = distances.row_data(i);
      double rho = std::numeric_limits<double>::infinity();
      double row_sum = 0.0;
      int row_count = 0;
      for (int j = 0; j < k; ++j) {
        const float d = row_values[j];
        if (!std::isfinite(d)) continue;
        if (d >= 0.0f) {
          row_sum += static_cast<double>(d);
          ++row_count;
        }
        if (d > 0.0f && static_cast<double>(d) < rho) {
          rho = static_cast<double>(d);
        }
      }

      if (!std::isfinite(rho)) rho = 0.0;
      rhos[static_cast<std::size_t>(i)] = static_cast<float>(rho);

      constexpr double sigma_max = (std::numeric_limits<double>::max)();
      double sigma = 1.0;
      double sigma_best = sigma;
      double best_diff = sigma_max;
      double lo = 0.0;
      double hi = sigma_max;

      for (int iter = 0; iter < 64; ++iter) {
        const double psum = membership_sum(row_values, rho, sigma);
        const double diff = std::abs(psum - target);
        if (diff < best_diff) {
          best_diff = diff;
          sigma_best = sigma;
        }
        if (psum > target) {
          hi = sigma;
          sigma = 0.5 * (lo + hi);
        } else {
          lo = sigma;
          if (hi == sigma_max) {
            sigma *= 2.0;
          } else {
            sigma = 0.5 * (lo + hi);
          }
        }
        if (diff < tol) break;
      }

      const double row_mean = row_count > 0 ?
        row_sum / static_cast<double>(row_count) :
        global_mean;
      const double sigma_floor = min_k_dist_scale * (rho > 0.0 ? row_mean : global_mean);
      sigma_best = std::max(sigma_best, sigma_floor);
      sigmas[static_cast<std::size_t>(i)] =
        static_cast<float>(std::max(sigma_best, 1.0e-12));
    }
  };

  if (threads == 1 || n < 2048) {
    worker(0, n, 0);
    return;
  }
  parallel_for_chunks(n, threads, worker);
}

void choose_budget_columns(const IntegerMatrix& indices,
                           const FloatDistanceView& distances,
                           const int row,
                           const int budget,
                           const int col_start,
                           std::vector<int>& order,
                           std::vector<char>& selected,
                           std::vector<int>& chosen) {
  const int k = distances.ncol;
  bool already_sorted = true;
  for (int col = 1; col < k; ++col) {
    const float prev_d = distances(row, col - 1);
    const float curr_d = distances(row, col);
    if (curr_d < prev_d ||
        (curr_d == prev_d &&
         indices(row, col + col_start) < indices(row, col - 1 + col_start))) {
      already_sorted = false;
      break;
    }
  }

  if (already_sorted) {
    chosen.clear();
    const int near_count = std::min(
      budget,
      std::max(2, static_cast<int>(std::floor(0.65 * budget)))
    );
    for (int col = 0; col < near_count; ++col) {
      chosen.push_back(col);
    }
    const int far_count = budget - near_count;
    for (int col = k - far_count; col < k; ++col) {
      if (col >= near_count) chosen.push_back(col);
    }
    for (int col = near_count; static_cast<int>(chosen.size()) < budget && col < k; ++col) {
      chosen.push_back(col);
    }
    return;
  }

  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](const int a, const int b) {
    const float da = distances(row, a);
    const float db = distances(row, b);
    if (da == db) return indices(row, a + col_start) < indices(row, b + col_start);
    return da < db;
  });

  std::fill(selected.begin(), selected.end(), 0);
  chosen.clear();
  const int near_count = std::min(
    budget,
    std::max(2, static_cast<int>(std::floor(0.65 * budget)))
  );
  for (int pos = 0; pos < near_count; ++pos) {
    const int col = order[static_cast<std::size_t>(pos)];
    selected[static_cast<std::size_t>(col)] = 1;
    chosen.push_back(col);
  }

  for (int pos = k - 1; pos >= 0 && static_cast<int>(chosen.size()) < budget; --pos) {
    const int col = order[static_cast<std::size_t>(pos)];
    if (!selected[static_cast<std::size_t>(col)]) {
      selected[static_cast<std::size_t>(col)] = 1;
      chosen.push_back(col);
    }
  }

  for (int pos = 0; pos < k && static_cast<int>(chosen.size()) < budget; ++pos) {
    const int col = order[static_cast<std::size_t>(pos)];
    if (!selected[static_cast<std::size_t>(col)]) {
      selected[static_cast<std::size_t>(col)] = 1;
      chosen.push_back(col);
    }
  }

  std::sort(chosen.begin(), chosen.end(), [&](const int a, const int b) {
    const float da = distances(row, a);
    const float db = distances(row, b);
    if (da == db) return indices(row, a + col_start) < indices(row, b + col_start);
    return da < db;
  });
}

std::vector<int> multiscale_graph_scales(const int kept_k) {
  if (kept_k < 1) return {1};
  return {kept_k};
}

int mid_near_edge_count(const int kept_k) {
  (void) kept_k;
  return 0;
}

float mid_near_edge_weight(const int kept_k) {
  (void) kept_k;
  return 0.0f;
}

std::vector<int> mid_near_columns(const int kept_k) {
  const int count = mid_near_edge_count(kept_k);
  if (count <= 0) return {};
  const int local_cutoff = std::min(15, kept_k - 1);
  const int span = kept_k - local_cutoff;
  if (span <= 0) return {};
  std::vector<int> cols;
  cols.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    int col = local_cutoff +
      static_cast<int>(
        (static_cast<std::int64_t>(i + 1) * static_cast<std::int64_t>(span)) /
        static_cast<std::int64_t>(count + 1)
      );
    col = std::max(local_cutoff, std::min(kept_k - 1, col));
    cols.push_back(col);
  }
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
  return cols;
}

double adaptive_prune_fraction(const int kept_k) {
  (void) kept_k;
  return 0.0;
}

int adaptive_prune_min_degree(const int kept_k) {
  if (kept_k < 30) return std::max(2, kept_k);
  if (kept_k < 50) return 15;
  if (kept_k < 150) return 20;
  return 24;
}

void update_top_weights(std::vector<float>& top,
                        const float weight,
                        const int limit) {
  if (limit <= 0) return;
  if (static_cast<int>(top.size()) < limit) {
    top.push_back(weight);
    return;
  }
  auto min_it = std::min_element(top.begin(), top.end());
  if (min_it != top.end() && weight > *min_it) {
    *min_it = weight;
  }
}

float sampled_weight_quantile(const std::vector<CsrCandidateEdge>& edges,
                              const double fraction) {
  if (edges.empty() || fraction <= 0.0) return -std::numeric_limits<float>::infinity();
  const std::size_t max_sample = 1000000u;
  const std::size_t stride = std::max<std::size_t>(1u, edges.size() / max_sample);
  std::vector<float> sample;
  sample.reserve((edges.size() + stride - 1u) / stride);
  for (std::size_t i = 0; i < edges.size(); i += stride) {
    sample.push_back(edges[i].weight);
  }
  if (sample.empty()) return -std::numeric_limits<float>::infinity();
  const double clamped = std::min(0.95, std::max(0.0, fraction));
  std::size_t nth = static_cast<std::size_t>(
    std::floor(clamped * static_cast<double>(sample.size() - 1u))
  );
  nth = std::min(nth, sample.size() - 1u);
  std::nth_element(sample.begin(), sample.begin() + nth, sample.end());
  return sample[nth];
}

std::vector<char> adaptive_prune_edges_with_rescue(const int n,
                                                   const int kept_k,
                                                   const std::vector<CsrCandidateEdge>& edges) {
  std::vector<char> keep(edges.size(), 1);
  const double prune_fraction = adaptive_prune_fraction(kept_k);
  if (edges.empty() || prune_fraction <= 0.0) return keep;

  const int min_degree = adaptive_prune_min_degree(kept_k);
  const float global_threshold = sampled_weight_quantile(edges, prune_fraction);
  std::vector<std::vector<float>> top_weights(static_cast<std::size_t>(n));
  for (auto& values : top_weights) {
    values.reserve(static_cast<std::size_t>(std::min(min_degree, 32)));
  }
  for (const auto& edge : edges) {
    update_top_weights(top_weights[static_cast<std::size_t>(edge.head)], edge.weight, min_degree);
    update_top_weights(top_weights[static_cast<std::size_t>(edge.tail)], edge.weight, min_degree);
  }

  std::vector<float> row_threshold(static_cast<std::size_t>(n), -std::numeric_limits<float>::infinity());
  for (int i = 0; i < n; ++i) {
    const auto& values = top_weights[static_cast<std::size_t>(i)];
    if (static_cast<int>(values.size()) >= min_degree && !values.empty()) {
      row_threshold[static_cast<std::size_t>(i)] =
        *std::min_element(values.begin(), values.end());
    }
  }

  DisjointSet dsu(n);
  for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
    const auto& edge = edges[edge_id];
    const bool globally_strong = edge.weight >= global_threshold;
    const bool row_strong =
      edge.weight >= row_threshold[static_cast<std::size_t>(edge.head)] ||
      edge.weight >= row_threshold[static_cast<std::size_t>(edge.tail)];
    keep[edge_id] = globally_strong || row_strong;
    if (keep[edge_id]) dsu.unite(edge.head, edge.tail);
  }

  int component_count = 0;
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int root = dsu.find(i);
    if (!seen[static_cast<std::size_t>(root)]) {
      seen[static_cast<std::size_t>(root)] = 1;
      ++component_count;
    }
  }
  if (component_count <= 1) return keep;

  std::vector<std::size_t> order(edges.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
    return edges[a].weight > edges[b].weight;
  });
  for (const std::size_t edge_id : order) {
    if (component_count <= 1) break;
    const auto& edge = edges[edge_id];
    const int root_a = dsu.find(edge.head);
    const int root_b = dsu.find(edge.tail);
    if (root_a == root_b) continue;
    keep[edge_id] = 1;
    dsu.unite(root_a, root_b);
    --component_count;
  }
  return keep;
}

void sort_csr_graph_rows(CsrGraphNative& graph) {
  const int n = static_cast<int>(graph.offsets.size()) - 1;
  std::vector<std::pair<int, float>> row;
  for (int i = 0; i < n; ++i) {
    const int begin = graph.offsets[static_cast<std::size_t>(i)];
    const int end = graph.offsets[static_cast<std::size_t>(i + 1)];
    if (end - begin < 2) continue;
    row.clear();
    row.reserve(static_cast<std::size_t>(end - begin));
    for (int pos = begin; pos < end; ++pos) {
      row.emplace_back(
        graph.neighbors[static_cast<std::size_t>(pos)],
        graph.weights[static_cast<std::size_t>(pos)]
      );
    }
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (int pos = begin; pos < end; ++pos) {
      const auto& item = row[static_cast<std::size_t>(pos - begin)];
      graph.neighbors[static_cast<std::size_t>(pos)] = item.first;
      graph.weights[static_cast<std::size_t>(pos)] = item.second;
    }
  }
}

void attach_csr_epoch_schedule(CsrGraphNative& graph) {
  graph.max_weight = 0.0f;
  for (const float w : graph.weights) {
    if (std::isfinite(w) && w > graph.max_weight) graph.max_weight = w;
  }
  graph.epochs_per_sample.resize(graph.weights.size());
  if (graph.max_weight <= 0.0f) {
    std::fill(
      graph.epochs_per_sample.begin(),
      graph.epochs_per_sample.end(),
      std::numeric_limits<float>::infinity()
    );
    return;
  }
  for (std::size_t i = 0; i < graph.weights.size(); ++i) {
    const float w = std::max(graph.weights[i], 1.0e-6f);
    graph.epochs_per_sample[i] = graph.max_weight / w;
  }
}

int compact_directed_rows_inplace(const std::vector<int>& raw_offsets,
                                  const std::vector<int>& row_ends,
                                  std::vector<int>& neighbors,
                                  std::vector<float>& weights,
                                  std::vector<int>& offsets,
                                  const int n_threads) {
  const int n = static_cast<int>(raw_offsets.size()) - 1;
  offsets.assign(static_cast<std::size_t>(n) + 1u, 0);
  std::vector<int> counts(static_cast<std::size_t>(n), 0);

  auto sort_count_worker = [&](const int begin_row, const int end_row, const int) {
    for (int row_id = begin_row; row_id < end_row; ++row_id) {
      const int begin = raw_offsets[static_cast<std::size_t>(row_id)];
      const int end = row_ends[static_cast<std::size_t>(row_id)];

      for (int pos = begin + 1; pos < end; ++pos) {
        const int nb = neighbors[static_cast<std::size_t>(pos)];
        const float w = weights[static_cast<std::size_t>(pos)];
        int cursor = pos;
        while (cursor > begin &&
               neighbors[static_cast<std::size_t>(cursor - 1)] > nb) {
          neighbors[static_cast<std::size_t>(cursor)] =
            neighbors[static_cast<std::size_t>(cursor - 1)];
          weights[static_cast<std::size_t>(cursor)] =
            weights[static_cast<std::size_t>(cursor - 1)];
          --cursor;
        }
        neighbors[static_cast<std::size_t>(cursor)] = nb;
        weights[static_cast<std::size_t>(cursor)] = w;
      }

      int count = 0;
      int read = begin;
      while (read < end) {
        const int nb = neighbors[static_cast<std::size_t>(read)];
        ++read;
        while (read < end && neighbors[static_cast<std::size_t>(read)] == nb) {
          ++read;
        }
        ++count;
      }
      counts[static_cast<std::size_t>(row_id)] = count;
    }
  };

  parallel_for_chunks(n, n_threads, sort_count_worker);

  for (int row_id = 0; row_id < n; ++row_id) {
    offsets[static_cast<std::size_t>(row_id + 1)] =
      offsets[static_cast<std::size_t>(row_id)] + counts[static_cast<std::size_t>(row_id)];
  }
  const int write = offsets[static_cast<std::size_t>(n)];
  std::vector<int> compact_neighbors(static_cast<std::size_t>(write));
  std::vector<float> compact_weights(static_cast<std::size_t>(write));

  auto fill_worker = [&](const int begin_row, const int end_row, const int) {
    for (int row_id = begin_row; row_id < end_row; ++row_id) {
      int write_pos = offsets[static_cast<std::size_t>(row_id)];
      const int begin = raw_offsets[static_cast<std::size_t>(row_id)];
      const int end = row_ends[static_cast<std::size_t>(row_id)];

      int read = begin;
      while (read < end) {
        const int nb = neighbors[static_cast<std::size_t>(read)];
        float w = weights[static_cast<std::size_t>(read)];
        ++read;
        while (read < end && neighbors[static_cast<std::size_t>(read)] == nb) {
          w = std::max(w, weights[static_cast<std::size_t>(read)]);
          ++read;
        }
        compact_neighbors[static_cast<std::size_t>(write_pos)] = nb;
        compact_weights[static_cast<std::size_t>(write_pos)] = w;
        ++write_pos;
      }
    }
  };

  parallel_for_chunks(n, n_threads, fill_worker);
  neighbors.swap(compact_neighbors);
  weights.swap(compact_weights);
  return write;
}

[[maybe_unused]] Graph build_graph(const IntegerMatrix& indices,
                                   const NumericMatrix& distances,
                                   const int edge_budget,
                                   const bool build_neighbor_lookup,
                                   const int n_threads) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  const int kept_k = std::min(k, std::max(1, edge_budget));
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  const int offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  std::vector<float> distance_values = copy_distances_float(distances, n_threads);
  const FloatDistanceView distance_view{distance_values.data(), n, k, k};
  std::vector<float> sigmas;
  std::vector<float> rhos;
  smooth_knn_dist(distance_view, sigmas, rhos, n_threads);

  std::vector<WeightedEdge> edges;
  edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(kept_k));
  NeighborLookup neighbor_lookup;
  if (build_neighbor_lookup) {
    neighbor_lookup.offsets.resize(static_cast<std::size_t>(n) + 1u);
    neighbor_lookup.neighbors.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  }
  std::vector<int> lookup_row;
  if (build_neighbor_lookup) lookup_row.reserve(k);
  std::vector<int> order(static_cast<std::size_t>(k));
  std::vector<char> selected(static_cast<std::size_t>(k), 0);
  std::vector<int> chosen;
  chosen.reserve(kept_k);

  for (int i = 0; i < n; ++i) {
    if (build_neighbor_lookup) {
      neighbor_lookup.offsets[static_cast<std::size_t>(i)] =
        static_cast<int>(neighbor_lookup.neighbors.size());
      lookup_row.clear();
      for (int j = 0; j < k; ++j) {
        const int neighbor = indices(i, j) - offset;
        if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
        lookup_row.push_back(neighbor);
      }
      std::sort(lookup_row.begin(), lookup_row.end());
      lookup_row.erase(std::unique(lookup_row.begin(), lookup_row.end()), lookup_row.end());
      neighbor_lookup.neighbors.insert(
        neighbor_lookup.neighbors.end(), lookup_row.begin(), lookup_row.end()
      );
    }

    if (kept_k < k) {
      choose_budget_columns(indices, distance_view, i, kept_k, 0, order, selected, chosen);
    }

    const int edge_cols = kept_k < k ? static_cast<int>(chosen.size()) : k;
    for (int pos = 0; pos < edge_cols; ++pos) {
      const int j = kept_k < k ? chosen[static_cast<std::size_t>(pos)] : pos;
      const int neighbor = indices(i, j) - offset;
      if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
      const float d = distance_view(i, j);
      const float rho = rhos[static_cast<std::size_t>(i)];
      const float sigma = sigmas[static_cast<std::size_t>(i)];
      const float val = d <= rho ? 1.0f : std::exp(-(d - rho) / sigma);
      if (val > 0.0f) {
        edges.push_back({edge_key(i, neighbor), val, 0u});
      }
    }
  }
  if (build_neighbor_lookup) {
    neighbor_lookup.offsets[static_cast<std::size_t>(n)] =
      static_cast<int>(neighbor_lookup.neighbors.size());
  }

  std::sort(edges.begin(), edges.end(), weighted_edge_less);
  std::size_t write = 0;
  for (std::size_t read = 0; read < edges.size(); ++read) {
    if (write > 0 && edges[write - 1].key == edges[read].key) {
      edges[write - 1].weight = std::max(edges[write - 1].weight, edges[read].weight);
    } else {
      if (write != read) edges[write] = edges[read];
      ++write;
    }
  }
  edges.resize(write);

  for (auto& edge : edges) {
    const int head = key_head(edge.key);
    const int tail = key_tail(edge.key);
    edge.direction = head <= tail ? 1u : 0u;
    edge.key = edge_key(std::min(head, tail), std::max(head, tail));
  }
  std::sort(edges.begin(), edges.end(), weighted_edge_less);

  Graph graph;
  graph.neighbor_lookup = std::move(neighbor_lookup);
  graph.edges.reserve(edges.size());
  for (std::size_t pos = 0; pos < edges.size();) {
    const std::uint64_t key = edges[pos].key;
    const int a = key_head(key);
    const int b = key_tail(key);
    float forward = 0.0f;
    float reverse = 0.0f;
    while (pos < edges.size() && edges[pos].key == key) {
      if (edges[pos].direction == 1u) {
        forward = std::max(forward, edges[pos].weight);
      } else {
        reverse = std::max(reverse, edges[pos].weight);
      }
      ++pos;
    }
    const double w = static_cast<double>(forward) + reverse -
                     static_cast<double>(forward) * reverse;
    if (w > 1e-6) graph.edges.push_back({a, b, static_cast<float>(w)});
  }
  return graph;
}

CsrGraphNative build_graph_csr_native_direct(const IntegerMatrix& indices,
                                             const NumericMatrix& distances,
                                             const int edge_budget,
                                             const int n_threads,
                                             const int index_offset,
                                             const int col_start,
                                             const int n_cols) {
  const int n = indices.nrow();
  const int k = n_cols;
  const int kept_k = std::min(k, std::max(1, edge_budget));

  std::vector<float> distance_values = copy_distances_float(distances, n_threads, col_start, k);
  const FloatDistanceView distance_view{distance_values.data(), n, k, k};
  std::vector<float> sigmas;
  std::vector<float> rhos;
  smooth_knn_dist(distance_view, sigmas, rhos, n_threads);

  std::vector<int> raw_counts(static_cast<std::size_t>(n), 0);
  std::vector<int> raw_offsets(static_cast<std::size_t>(n) + 1u, 0);
  auto count_directed_worker = [&](const int begin_row, const int end_row, const int) {
    std::vector<int> order(static_cast<std::size_t>(k));
    std::vector<char> selected(static_cast<std::size_t>(k), 0);
    std::vector<int> chosen;
    chosen.reserve(static_cast<std::size_t>(kept_k));
    for (int i = begin_row; i < end_row; ++i) {
      if (kept_k < k) {
        choose_budget_columns(indices, distance_view, i, kept_k, col_start, order, selected, chosen);
      }
      const int edge_cols = kept_k < k ? static_cast<int>(chosen.size()) : k;
      int count = 0;
      for (int pos = 0; pos < edge_cols; ++pos) {
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(pos)] : pos;
        const int neighbor = indices(i, col + col_start) - index_offset;
        const float d = distance_view(i, col);
        if (neighbor < 0 || neighbor >= n || neighbor == i || !std::isfinite(d)) continue;
        ++count;
      }
      raw_counts[static_cast<std::size_t>(i)] = count;
    }
  };
  parallel_for_chunks(n, n_threads, count_directed_worker);
  for (int i = 0; i < n; ++i) {
    raw_offsets[static_cast<std::size_t>(i + 1)] =
      raw_offsets[static_cast<std::size_t>(i)] + raw_counts[static_cast<std::size_t>(i)];
  }

  std::vector<int> directed_neighbors(static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)]));
  std::vector<float> directed_weights(static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)]));
  std::vector<int> cursor = raw_offsets;
  auto fill_directed_worker = [&](const int begin_row, const int end_row, const int) {
    std::vector<int> order(static_cast<std::size_t>(k));
    std::vector<char> selected(static_cast<std::size_t>(k), 0);
    std::vector<int> chosen;
    chosen.reserve(static_cast<std::size_t>(kept_k));
    for (int i = begin_row; i < end_row; ++i) {
      if (kept_k < k) {
        choose_budget_columns(indices, distance_view, i, kept_k, col_start, order, selected, chosen);
      }
      const int edge_cols = kept_k < k ? static_cast<int>(chosen.size()) : k;
      const float rho = rhos[static_cast<std::size_t>(i)];
      const float sigma = sigmas[static_cast<std::size_t>(i)];
      for (int pos = 0; pos < edge_cols; ++pos) {
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(pos)] : pos;
        const int neighbor = indices(i, col + col_start) - index_offset;
        const float d = distance_view(i, col);
        if (neighbor < 0 || neighbor >= n || neighbor == i || !std::isfinite(d)) continue;
        const float weight = d <= rho ? 1.0f : std::exp(-(d - rho) / sigma);
        if (!std::isfinite(weight) || weight <= 0.0f) continue;
        const int out_pos = cursor[static_cast<std::size_t>(i)]++;
        directed_neighbors[static_cast<std::size_t>(out_pos)] = neighbor;
        directed_weights[static_cast<std::size_t>(out_pos)] = weight;
      }
    }
  };
  parallel_for_chunks(n, n_threads, fill_directed_worker);

  std::vector<int> directed_offsets(static_cast<std::size_t>(n) + 1u, 0);
  const int write = compact_directed_rows_inplace(
    raw_offsets, cursor, directed_neighbors, directed_weights, directed_offsets, n_threads
  );
  release_vector(distance_values);
  release_vector(sigmas);
  release_vector(rhos);
  release_vector(cursor);
  release_vector(raw_offsets);

  const int graph_threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<int>> incoming_counts(
    static_cast<std::size_t>(graph_threads),
    std::vector<int>(static_cast<std::size_t>(n), 0)
  );
  auto count_incoming_worker = [&](const int begin_row, const int end_row, const int thread_id) {
    auto& local_counts = incoming_counts[static_cast<std::size_t>(thread_id)];
    for (int i = begin_row; i < end_row; ++i) {
      const int begin = directed_offsets[static_cast<std::size_t>(i)];
      const int end = directed_offsets[static_cast<std::size_t>(i + 1)];
      for (int pos = begin; pos < end; ++pos) {
        const int target = directed_neighbors[static_cast<std::size_t>(pos)];
        ++local_counts[static_cast<std::size_t>(target)];
      }
    }
  };
  parallel_for_chunks(n, graph_threads, count_incoming_worker);

  std::vector<int> incoming_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    int row_count = 0;
    for (int t = 0; t < graph_threads; ++t) {
      row_count += incoming_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)];
    }
    incoming_offsets[static_cast<std::size_t>(i + 1)] =
      incoming_offsets[static_cast<std::size_t>(i)] + row_count;
  }

  for (int i = 0; i < n; ++i) {
    int start = incoming_offsets[static_cast<std::size_t>(i)];
    for (int t = 0; t < graph_threads; ++t) {
      const int count = incoming_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)];
      incoming_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)] = start;
      start += count;
    }
  }

  std::vector<int> incoming_neighbors(static_cast<std::size_t>(write));
  std::vector<float> incoming_weights(static_cast<std::size_t>(write));
  auto fill_incoming_worker = [&](const int begin_row, const int end_row, const int thread_id) {
    auto& local_cursor = incoming_counts[static_cast<std::size_t>(thread_id)];
    for (int i = begin_row; i < end_row; ++i) {
      const int begin = directed_offsets[static_cast<std::size_t>(i)];
      const int end = directed_offsets[static_cast<std::size_t>(i + 1)];
      for (int pos = begin; pos < end; ++pos) {
        const int target = directed_neighbors[static_cast<std::size_t>(pos)];
        const int out_pos = local_cursor[static_cast<std::size_t>(target)]++;
        incoming_neighbors[static_cast<std::size_t>(out_pos)] = i;
        incoming_weights[static_cast<std::size_t>(out_pos)] =
          directed_weights[static_cast<std::size_t>(pos)];
      }
    }
  };
  parallel_for_chunks(n, graph_threads, fill_incoming_worker);
  release_nested_vector(incoming_counts);

  auto merge_union_row = [&](const int i,
                             const bool write_output,
                             int& output_pos,
                             CsrGraphNative& graph) {
    int out_pos = directed_offsets[static_cast<std::size_t>(i)];
    const int out_end = directed_offsets[static_cast<std::size_t>(i + 1)];
    int in_pos = incoming_offsets[static_cast<std::size_t>(i)];
    const int in_end = incoming_offsets[static_cast<std::size_t>(i + 1)];
    int count = 0;
    while (out_pos < out_end || in_pos < in_end) {
      const int out_nb = out_pos < out_end ?
        directed_neighbors[static_cast<std::size_t>(out_pos)] :
        std::numeric_limits<int>::max();
      const int in_nb = in_pos < in_end ?
        incoming_neighbors[static_cast<std::size_t>(in_pos)] :
        std::numeric_limits<int>::max();

      int neighbor = 0;
      float forward = 0.0f;
      float reverse = 0.0f;
      if (out_nb == in_nb) {
        neighbor = out_nb;
        forward = directed_weights[static_cast<std::size_t>(out_pos)];
        reverse = incoming_weights[static_cast<std::size_t>(in_pos)];
        ++out_pos;
        ++in_pos;
      } else if (out_nb < in_nb) {
        neighbor = out_nb;
        forward = directed_weights[static_cast<std::size_t>(out_pos)];
        ++out_pos;
      } else {
        neighbor = in_nb;
        reverse = incoming_weights[static_cast<std::size_t>(in_pos)];
        ++in_pos;
      }

      if (neighbor == i) continue;
      const float weight = forward + reverse - forward * reverse;
      if (!std::isfinite(weight) || weight <= 1.0e-6f) continue;
      if (write_output) {
        graph.neighbors[static_cast<std::size_t>(output_pos)] = neighbor;
        graph.weights[static_cast<std::size_t>(output_pos)] = weight;
        ++output_pos;
      }
      ++count;
    }
    return count;
  };

  std::vector<int>& final_counts = raw_counts;
  std::fill(final_counts.begin(), final_counts.end(), 0);
  CsrGraphNative dummy_graph;
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int i = begin_row; i < end_row; ++i) {
      int ignored_output_pos = 0;
      final_counts[static_cast<std::size_t>(i)] =
        merge_union_row(i, false, ignored_output_pos, dummy_graph);
    }
  });

  CsrGraphNative graph;
  graph.offsets.resize(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    graph.offsets[static_cast<std::size_t>(i + 1)] =
      graph.offsets[static_cast<std::size_t>(i)] + final_counts[static_cast<std::size_t>(i)];
  }
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  graph.neighbors.resize(static_cast<std::size_t>(nnz));
  graph.weights.resize(static_cast<std::size_t>(nnz));
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int i = begin_row; i < end_row; ++i) {
      int output_pos = graph.offsets[static_cast<std::size_t>(i)];
      merge_union_row(i, true, output_pos, graph);
    }
  });

  attach_csr_epoch_schedule(graph);
  return graph;
}

CsrGraphNative build_graph_csr_native(const IntegerMatrix& indices,
                                      const NumericMatrix& distances,
                                      const int edge_budget,
                                      const int n_threads,
                                      const int col_start = 0,
                                      int n_cols = -1) {
  const int n = indices.nrow();
  const int matrix_k = indices.ncol();
  if (n_cols < 0) n_cols = matrix_k - col_start;
  if (col_start < 0 || n_cols < 1 || col_start + n_cols > matrix_k ||
      distances.ncol() < col_start + n_cols) {
    Rcpp::stop("invalid KNN column range");
  }
  const int k = n_cols;
  const int kept_k = std::min(k, std::max(1, edge_budget));
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int idx = indices(i, j + col_start);
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const int index_offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  const std::vector<int> graph_scales = multiscale_graph_scales(kept_k);
  const std::vector<int> mid_cols = mid_near_columns(kept_k);
  const bool pure_default_graph =
    graph_scales.size() == 1u &&
    graph_scales[0] == kept_k &&
    mid_cols.empty() &&
    adaptive_prune_fraction(kept_k) <= 0.0;
  if (pure_default_graph) {
    return build_graph_csr_native_direct(
      indices, distances, edge_budget, n_threads, index_offset, col_start, k
    );
  }

  std::vector<float> distance_values = copy_distances_float(distances, n_threads, col_start, k);
  const FloatDistanceView distance_view{distance_values.data(), n, k, k};
  const float mid_weight = mid_near_edge_weight(kept_k);
  std::vector<std::vector<float>> scale_sigmas(graph_scales.size());
  std::vector<std::vector<float>> scale_rhos(graph_scales.size());
  for (std::size_t scale_id = 0; scale_id < graph_scales.size(); ++scale_id) {
    const FloatDistanceView scale_view{
      distance_values.data(), n, graph_scales[scale_id], k
    };
    smooth_knn_dist(
      scale_view,
      scale_sigmas[scale_id],
      scale_rhos[scale_id],
      n_threads
    );
  }

  std::vector<int> order(static_cast<std::size_t>(k));
  std::vector<char> selected(static_cast<std::size_t>(k), 0);
  std::vector<int> chosen;
  chosen.reserve(kept_k);
  int multiscale_edge_budget = 0;
  for (const int scale : graph_scales) multiscale_edge_budget += scale;
  multiscale_edge_budget += static_cast<int>(mid_cols.size());

  std::vector<int> raw_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    if (kept_k < k) {
      choose_budget_columns(indices, distance_view, i, kept_k, col_start, order, selected, chosen);
    }
    int count = 0;
    for (const int scale : graph_scales) {
      const int edge_cols = kept_k < k ?
        std::min(scale, static_cast<int>(chosen.size())) :
        scale;
      for (int pos = 0; pos < edge_cols; ++pos) {
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(pos)] : pos;
        const int neighbor = indices(i, col + col_start) - index_offset;
        if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
        ++count;
      }
    }
    if (!mid_cols.empty() && mid_weight > 0.0f) {
      const int available_cols = kept_k < k ?
        static_cast<int>(chosen.size()) :
        kept_k;
      for (const int rank : mid_cols) {
        if (rank < 0 || rank >= available_cols) continue;
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(rank)] : rank;
        const int neighbor = indices(i, col + col_start) - index_offset;
        if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
        ++count;
      }
    }
    raw_offsets[static_cast<std::size_t>(i + 1)] =
      raw_offsets[static_cast<std::size_t>(i)] + count;
  }

  std::vector<int> directed_neighbors(static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)]));
  std::vector<float> directed_weights(static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)]));
  std::vector<int> cursor = raw_offsets;
  for (int i = 0; i < n; ++i) {
    if (kept_k < k) {
      choose_budget_columns(indices, distance_view, i, kept_k, col_start, order, selected, chosen);
    }
    for (std::size_t scale_id = 0; scale_id < graph_scales.size(); ++scale_id) {
      const int scale = graph_scales[scale_id];
      const int edge_cols = kept_k < k ?
        std::min(scale, static_cast<int>(chosen.size())) :
        scale;
      const float rho = scale_rhos[scale_id][static_cast<std::size_t>(i)];
      const float sigma = scale_sigmas[scale_id][static_cast<std::size_t>(i)];
      for (int pos = 0; pos < edge_cols; ++pos) {
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(pos)] : pos;
        const int neighbor = indices(i, col + col_start) - index_offset;
        if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
        const float d = distance_view(i, col);
        const float weight = d <= rho ? 1.0f : std::exp(-(d - rho) / sigma);
        if (weight <= 0.0f) continue;
        const int out_pos = cursor[static_cast<std::size_t>(i)]++;
        directed_neighbors[static_cast<std::size_t>(out_pos)] = neighbor;
        directed_weights[static_cast<std::size_t>(out_pos)] = weight;
      }
    }
    if (!mid_cols.empty() && mid_weight > 0.0f) {
      const int available_cols = kept_k < k ?
        static_cast<int>(chosen.size()) :
        kept_k;
      for (const int rank : mid_cols) {
        if (rank < 0 || rank >= available_cols) continue;
        const int col = kept_k < k ? chosen[static_cast<std::size_t>(rank)] : rank;
        const int neighbor = indices(i, col + col_start) - index_offset;
        if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
        const int out_pos = cursor[static_cast<std::size_t>(i)]++;
        directed_neighbors[static_cast<std::size_t>(out_pos)] = neighbor;
        directed_weights[static_cast<std::size_t>(out_pos)] = mid_weight;
      }
    }
  }

  std::vector<std::pair<int, float>> row;
  row.reserve(static_cast<std::size_t>(std::max(k, multiscale_edge_budget)));
  std::vector<int> directed_offsets(static_cast<std::size_t>(n) + 1u, 0);
  int write = 0;
  for (int i = 0; i < n; ++i) {
    directed_offsets[static_cast<std::size_t>(i)] = write;
    row.clear();
    const int begin = raw_offsets[static_cast<std::size_t>(i)];
    const int end = cursor[static_cast<std::size_t>(i)];
    for (int pos = begin; pos < end; ++pos) {
      row.emplace_back(
        directed_neighbors[static_cast<std::size_t>(pos)],
        directed_weights[static_cast<std::size_t>(pos)]
      );
    }
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (std::size_t pos = 0; pos < row.size();) {
      const int neighbor = row[pos].first;
      float weight = row[pos].second;
      ++pos;
      while (pos < row.size() && row[pos].first == neighbor) {
        weight = std::max(weight, row[pos].second);
        ++pos;
      }
      directed_neighbors[static_cast<std::size_t>(write)] = neighbor;
      directed_weights[static_cast<std::size_t>(write)] = weight;
      ++write;
    }
  }
  directed_offsets[static_cast<std::size_t>(n)] = write;
  directed_neighbors.resize(static_cast<std::size_t>(write));
  directed_weights.resize(static_cast<std::size_t>(write));
  release_vector(distance_values);
  release_nested_vector(scale_sigmas);
  release_nested_vector(scale_rhos);
  release_vector(raw_offsets);
  release_vector(cursor);

  std::vector<CsrCandidateEdge> candidate_edges;
  candidate_edges.reserve(static_cast<std::size_t>(std::max(1, write)));
  auto collect_undirected_edge = [&](const int i, const int j, const float forward) {
    const float reverse = csr_lookup_weight(directed_offsets, directed_neighbors, directed_weights, j, i);
    if (reverse > 0.0f && i > j) return;
    const float weight = forward + reverse - forward * reverse;
    if (weight <= 1.0e-6f) return;
    candidate_edges.push_back({i, j, weight});
  };

  for (int i = 0; i < n; ++i) {
    for (int pos = directed_offsets[static_cast<std::size_t>(i)];
         pos < directed_offsets[static_cast<std::size_t>(i + 1)];
         ++pos) {
      collect_undirected_edge(i, directed_neighbors[static_cast<std::size_t>(pos)],
                              directed_weights[static_cast<std::size_t>(pos)]);
    }
  }

  const std::vector<char> keep_edges =
    adaptive_prune_edges_with_rescue(n, kept_k, candidate_edges);

  std::vector<int> final_counts(static_cast<std::size_t>(n), 0);
  for (std::size_t edge_id = 0; edge_id < candidate_edges.size(); ++edge_id) {
    if (!keep_edges[edge_id]) continue;
    const auto& edge = candidate_edges[edge_id];
    ++final_counts[static_cast<std::size_t>(edge.head)];
    ++final_counts[static_cast<std::size_t>(edge.tail)];
  }

  CsrGraphNative graph;
  graph.offsets.resize(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    graph.offsets[static_cast<std::size_t>(i + 1)] =
      graph.offsets[static_cast<std::size_t>(i)] + final_counts[static_cast<std::size_t>(i)];
  }
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  graph.neighbors.resize(static_cast<std::size_t>(nnz));
  graph.weights.resize(static_cast<std::size_t>(nnz));
  cursor = graph.offsets;

  for (std::size_t edge_id = 0; edge_id < candidate_edges.size(); ++edge_id) {
    if (!keep_edges[edge_id]) continue;
    const auto& edge = candidate_edges[edge_id];
    const int pos_i = cursor[static_cast<std::size_t>(edge.head)]++;
    graph.neighbors[static_cast<std::size_t>(pos_i)] = edge.tail;
    graph.weights[static_cast<std::size_t>(pos_i)] = edge.weight;
    const int pos_j = cursor[static_cast<std::size_t>(edge.tail)]++;
    graph.neighbors[static_cast<std::size_t>(pos_j)] = edge.head;
    graph.weights[static_cast<std::size_t>(pos_j)] = edge.weight;
  }

  sort_csr_graph_rows(graph);
  attach_csr_epoch_schedule(graph);

  return graph;
}

void scale_embedding_max_abs_and_jitter(NumericMatrix& embedding,
                                        double max_coord,
                                        double jitter_sd,
                                        std::mt19937& rng);

NumericMatrix initialize_layout(const int n,
                                const int n_components,
                                const std::vector<Edge>& edges,
                                const int spectral_n_iter,
                                std::mt19937& rng) {
  NumericMatrix embedding(n, n_components);
  std::normal_distribution<double> normal(0.0, 0.0001);
  std::uniform_real_distribution<double> uniform(-10.0, 10.0);

  if (edges.empty()) {
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < n_components; ++c) embedding(i, c) = uniform(rng);
    }
    return embedding;
  }

  std::vector<double> degree(n, 0.0);
  for (const auto& e : edges) {
    degree[e.head] += e.weight;
    degree[e.tail] += e.weight;
  }

  std::vector<float> spectral_weight(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const Edge& e = edges[i];
    const double denom = std::sqrt(std::max(degree[e.head] * degree[e.tail], 1e-24));
    spectral_weight[i] = static_cast<float>(e.weight / denom);
  }

  std::vector<double> trivial(n, 0.0);
  double trivial_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    trivial[i] = std::sqrt(std::max(degree[i], 0.0));
    trivial_norm += trivial[i] * trivial[i];
  }
  trivial_norm = std::sqrt(std::max(trivial_norm, 1e-12));
  for (double& v : trivial) v /= trivial_norm;

  const int block_size = std::min(n - 1, std::max(n_components + 6, 8));
  if (block_size < n_components) {
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < n_components; ++c) embedding(i, c) = uniform(rng);
    }
    return embedding;
  }

  auto at = [n](std::vector<double>& x, const int row, const int col) -> double& {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto cat = [n](const std::vector<double>& x, const int row, const int col) -> double {
    return x[static_cast<std::size_t>(col) * n + row];
  };

  auto apply_normalized_adjacency = [&](const std::vector<double>& x,
                                        std::vector<double>& y,
                                        const int cols) {
    std::fill(y.begin(), y.end(), 0.0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      const Edge& e = edges[i];
      const double weight = spectral_weight[i];
      for (int c = 0; c < cols; ++c) {
        y[static_cast<std::size_t>(c) * n + e.head] += weight * cat(x, e.tail, c);
        y[static_cast<std::size_t>(c) * n + e.tail] += weight * cat(x, e.head, c);
      }
    }
  };

  auto orthonormalize_block = [&](std::vector<double>& q, const int cols) {
    for (int c = 0; c < cols; ++c) {
      double dot0 = 0.0;
      for (int i = 0; i < n; ++i) dot0 += cat(q, i, c) * trivial[i];
      for (int i = 0; i < n; ++i) at(q, i, c) -= dot0 * trivial[i];

      for (int prev = 0; prev < c; ++prev) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i) dot += cat(q, i, c) * cat(q, i, prev);
        for (int i = 0; i < n; ++i) at(q, i, c) -= dot * cat(q, i, prev);
      }

      double norm = 0.0;
      for (int i = 0; i < n; ++i) norm += cat(q, i, c) * cat(q, i, c);
      norm = std::sqrt(norm);
      if (norm < 1e-10) {
        for (int i = 0; i < n; ++i) at(q, i, c) = normal(rng) * 10000.0;
        --c;
      } else {
        for (int i = 0; i < n; ++i) at(q, i, c) /= norm;
      }
    }
  };

  std::vector<double> q(static_cast<std::size_t>(n) * block_size);
  std::vector<double> z(static_cast<std::size_t>(n) * block_size);
  for (double& v : q) v = normal(rng) * 10000.0;
  orthonormalize_block(q, block_size);

  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    apply_normalized_adjacency(q, z, block_size);
    orthonormalize_block(z, block_size);
    q.swap(z);
  }

  apply_normalized_adjacency(q, z, block_size);
  std::vector<double> projected(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int a_col = 0; a_col < block_size; ++a_col) {
    for (int b_col = a_col; b_col < block_size; ++b_col) {
      double dot = 0.0;
      for (int i = 0; i < n; ++i) dot += cat(q, i, a_col) * cat(z, i, b_col);
      projected[static_cast<std::size_t>(a_col) * block_size + b_col] = dot;
      projected[static_cast<std::size_t>(b_col) * block_size + a_col] = dot;
    }
  }

  std::vector<double> eigvec(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int i = 0; i < block_size; ++i) eigvec[static_cast<std::size_t>(i) * block_size + i] = 1.0;
  for (int sweep = 0; sweep < 80; ++sweep) {
    int p = 0;
    int qidx = 1;
    double max_offdiag = 0.0;
    for (int i = 0; i < block_size; ++i) {
      for (int j = i + 1; j < block_size; ++j) {
        const double v = std::abs(projected[static_cast<std::size_t>(i) * block_size + j]);
        if (v > max_offdiag) {
          max_offdiag = v;
          p = i;
          qidx = j;
        }
      }
    }
    if (max_offdiag < 1e-10) break;

    const double app = projected[static_cast<std::size_t>(p) * block_size + p];
    const double aqq = projected[static_cast<std::size_t>(qidx) * block_size + qidx];
    const double apq = projected[static_cast<std::size_t>(p) * block_size + qidx];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cs = 1.0 / std::sqrt(1.0 + t * t);
    const double sn = t * cs;

    for (int k = 0; k < block_size; ++k) {
      if (k == p || k == qidx) continue;
      const double akp = projected[static_cast<std::size_t>(k) * block_size + p];
      const double akq = projected[static_cast<std::size_t>(k) * block_size + qidx];
      projected[static_cast<std::size_t>(k) * block_size + p] = cs * akp - sn * akq;
      projected[static_cast<std::size_t>(p) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + p];
      projected[static_cast<std::size_t>(k) * block_size + qidx] = sn * akp + cs * akq;
      projected[static_cast<std::size_t>(qidx) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + qidx];
    }
    projected[static_cast<std::size_t>(p) * block_size + p] = cs * cs * app - 2.0 * sn * cs * apq + sn * sn * aqq;
    projected[static_cast<std::size_t>(qidx) * block_size + qidx] = sn * sn * app + 2.0 * sn * cs * apq + cs * cs * aqq;
    projected[static_cast<std::size_t>(p) * block_size + qidx] = 0.0;
    projected[static_cast<std::size_t>(qidx) * block_size + p] = 0.0;

    for (int k = 0; k < block_size; ++k) {
      const double vip = eigvec[static_cast<std::size_t>(k) * block_size + p];
      const double viq = eigvec[static_cast<std::size_t>(k) * block_size + qidx];
      eigvec[static_cast<std::size_t>(k) * block_size + p] = cs * vip - sn * viq;
      eigvec[static_cast<std::size_t>(k) * block_size + qidx] = sn * vip + cs * viq;
    }
  }

  std::vector<int> order(block_size);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](const int a, const int b) {
    return projected[static_cast<std::size_t>(a) * block_size + a] >
           projected[static_cast<std::size_t>(b) * block_size + b];
  });

  std::vector<std::vector<double>> vectors;
  vectors.reserve(n_components);
  for (int c = 0; c < n_components; ++c) {
    const int ritz_col = order[c];
    std::vector<double> values(n, 0.0);
    for (int basis = 0; basis < block_size; ++basis) {
      const double coeff = eigvec[static_cast<std::size_t>(basis) * block_size + ritz_col];
      for (int i = 0; i < n; ++i) values[i] += cat(q, i, basis) * coeff;
    }
    vectors.push_back(std::move(values));
  }

  for (int c = 0; c < n_components; ++c) {
    std::vector<double> values = vectors[c];
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
    for (double& v : values) {
      v -= mean;
    }
    for (int i = 0; i < n; ++i) embedding(i, c) = values[i];
  }
  scale_embedding_max_abs_and_jitter(embedding, 10.0, 1.0e-4, rng);

  return embedding;
}

template <typename OffsetVec, typename NeighborVec, typename WeightVec>
NumericMatrix initialize_layout_csr(const int n,
                                    const int n_components,
                                    const OffsetVec& offsets,
                                    const NeighborVec& neighbors,
                                    const WeightVec& weights,
                                    const int index_offset,
                                    const std::vector<int>& active_rows,
                                    const std::vector<int>& active_pos,
                                    const int spectral_n_iter,
                                    const int n_threads,
                                    std::mt19937& rng) {
  NumericMatrix embedding(n, n_components);
  std::normal_distribution<double> normal(0.0, 0.0001);
  std::uniform_real_distribution<double> uniform(-10.0, 10.0);

  const int nnz = offsets[n];
  const int spectral_edge_count = static_cast<int>(active_pos.size());
  if (nnz == 0 || spectral_edge_count == 0) {
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < n_components; ++c) embedding(i, c) = uniform(rng);
    }
    return embedding;
  }

  std::vector<float> degree(static_cast<std::size_t>(n), 0.0f);
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      double row_degree = 0.0;
      const int begin = offsets[row];
      const int end = offsets[row + 1];
      for (int pos = begin; pos < end; ++pos) {
        row_degree += weights[pos];
      }
      degree[static_cast<std::size_t>(row)] = static_cast<float>(row_degree);
    }
  });

  const int spectral_threads = effective_cpu_threads(n_threads, spectral_edge_count);
  std::vector<std::vector<int>> spectral_thread_counts(
    static_cast<std::size_t>(spectral_threads),
    std::vector<int>(static_cast<std::size_t>(n), 0)
  );
  parallel_for_chunks(
    spectral_edge_count,
    spectral_threads,
    [&](const int begin_edge, const int end_edge, const int thread_id) {
      auto& local_counts = spectral_thread_counts[static_cast<std::size_t>(thread_id)];
      for (int e = begin_edge; e < end_edge; ++e) {
        const int head = active_rows[static_cast<std::size_t>(e)];
        const int tail = neighbors[active_pos[static_cast<std::size_t>(e)]] - index_offset;
        if (head < 0 || head >= n || tail < 0 || tail >= n || head == tail) continue;
        ++local_counts[static_cast<std::size_t>(head)];
        ++local_counts[static_cast<std::size_t>(tail)];
      }
    }
  );

  std::vector<int> spectral_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    int row_count = 0;
    for (int t = 0; t < spectral_threads; ++t) {
      row_count += spectral_thread_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)];
    }
    spectral_offsets[static_cast<std::size_t>(i + 1)] =
      spectral_offsets[static_cast<std::size_t>(i)] + row_count;
  }

  for (int i = 0; i < n; ++i) {
    int start = spectral_offsets[static_cast<std::size_t>(i)];
    for (int t = 0; t < spectral_threads; ++t) {
      const int count = spectral_thread_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)];
      spectral_thread_counts[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)] = start;
      start += count;
    }
  }

  const int spectral_nnz = spectral_offsets[static_cast<std::size_t>(n)];
  std::vector<int> spectral_neighbors(static_cast<std::size_t>(spectral_nnz));
  std::vector<float> spectral_weights(static_cast<std::size_t>(spectral_nnz));
  parallel_for_chunks(
    spectral_edge_count,
    spectral_threads,
    [&](const int begin_edge, const int end_edge, const int thread_id) {
      auto& local_cursor = spectral_thread_counts[static_cast<std::size_t>(thread_id)];
      for (int e = begin_edge; e < end_edge; ++e) {
        const int head = active_rows[static_cast<std::size_t>(e)];
        const int pos = active_pos[static_cast<std::size_t>(e)];
        const int tail = neighbors[pos] - index_offset;
        if (head < 0 || head >= n || tail < 0 || tail >= n || head == tail) continue;
        const double denom = std::sqrt(std::max(
          static_cast<double>(degree[static_cast<std::size_t>(head)]) *
            static_cast<double>(degree[static_cast<std::size_t>(tail)]),
          1e-24
        ));
        const float normalized_weight = static_cast<float>(weights[pos] / denom);
        const int head_pos = local_cursor[static_cast<std::size_t>(head)]++;
        spectral_neighbors[static_cast<std::size_t>(head_pos)] = tail;
        spectral_weights[static_cast<std::size_t>(head_pos)] = normalized_weight;
        const int tail_pos = local_cursor[static_cast<std::size_t>(tail)]++;
        spectral_neighbors[static_cast<std::size_t>(tail_pos)] = head;
        spectral_weights[static_cast<std::size_t>(tail_pos)] = normalized_weight;
      }
    }
  );
  release_nested_vector(spectral_thread_counts);

  std::vector<double> trivial(static_cast<std::size_t>(n), 0.0);
  double trivial_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    trivial[static_cast<std::size_t>(i)] =
      std::sqrt(std::max(static_cast<double>(degree[static_cast<std::size_t>(i)]), 0.0));
    trivial_norm += trivial[static_cast<std::size_t>(i)] * trivial[static_cast<std::size_t>(i)];
  }
  trivial_norm = std::sqrt(std::max(trivial_norm, 1e-12));
  for (double& v : trivial) v /= trivial_norm;

  const int block_size = std::min(n - 1, std::max(n_components + 6, 8));
  if (block_size < n_components) {
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < n_components; ++c) embedding(i, c) = uniform(rng);
    }
    return embedding;
  }

  auto at = [n](std::vector<double>& x, const int row, const int col) -> double& {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto cat = [n](const std::vector<double>& x, const int row, const int col) -> double {
    return x[static_cast<std::size_t>(col) * n + row];
  };

  auto apply_normalized_adjacency = [&](const std::vector<double>& x,
                                        std::vector<double>& y,
                                        const int cols) {
    parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
      std::vector<double> sums(static_cast<std::size_t>(cols), 0.0);
      for (int row = begin_row; row < end_row; ++row) {
        const int begin = spectral_offsets[static_cast<std::size_t>(row)];
        const int end = spectral_offsets[static_cast<std::size_t>(row + 1)];
        std::fill(sums.begin(), sums.end(), 0.0);
        for (int pos = begin; pos < end; ++pos) {
          const int tail = spectral_neighbors[static_cast<std::size_t>(pos)];
          const double weight = static_cast<double>(spectral_weights[static_cast<std::size_t>(pos)]);
          for (int c = 0; c < cols; ++c) {
            sums[static_cast<std::size_t>(c)] += weight * cat(x, tail, c);
          }
        }
        for (int c = 0; c < cols; ++c) {
          y[static_cast<std::size_t>(c) * n + row] = sums[static_cast<std::size_t>(c)];
        }
      }
    });
  };

  auto orthonormalize_block = [&](std::vector<double>& q, const int cols) {
    for (int c = 0; c < cols; ++c) {
      double dot0 = 0.0;
      for (int i = 0; i < n; ++i) dot0 += cat(q, i, c) * trivial[static_cast<std::size_t>(i)];
      for (int i = 0; i < n; ++i) at(q, i, c) -= dot0 * trivial[static_cast<std::size_t>(i)];

      for (int prev = 0; prev < c; ++prev) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i) dot += cat(q, i, c) * cat(q, i, prev);
        for (int i = 0; i < n; ++i) at(q, i, c) -= dot * cat(q, i, prev);
      }

      double norm = 0.0;
      for (int i = 0; i < n; ++i) norm += cat(q, i, c) * cat(q, i, c);
      norm = std::sqrt(norm);
      if (norm < 1e-10) {
        for (int i = 0; i < n; ++i) at(q, i, c) = normal(rng) * 10000.0;
        --c;
      } else {
        for (int i = 0; i < n; ++i) at(q, i, c) /= norm;
      }
    }
  };

  std::vector<double> q(static_cast<std::size_t>(n) * block_size);
  std::vector<double> z(static_cast<std::size_t>(n) * block_size);
  for (double& v : q) v = normal(rng) * 10000.0;
  orthonormalize_block(q, block_size);

  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    apply_normalized_adjacency(q, z, block_size);
    orthonormalize_block(z, block_size);
    q.swap(z);
  }

  apply_normalized_adjacency(q, z, block_size);
  std::vector<double> projected(static_cast<std::size_t>(block_size) * block_size, 0.0);
  const int projection_threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<double>> projected_thread(
    static_cast<std::size_t>(projection_threads),
    std::vector<double>(static_cast<std::size_t>(block_size) * block_size, 0.0)
  );
  parallel_for_chunks(n, projection_threads, [&](const int begin_row, const int end_row, const int thread_id) {
    auto& local = projected_thread[static_cast<std::size_t>(thread_id)];
    for (int row = begin_row; row < end_row; ++row) {
      for (int a_col = 0; a_col < block_size; ++a_col) {
        const double qv = cat(q, row, a_col);
        for (int b_col = a_col; b_col < block_size; ++b_col) {
          local[static_cast<std::size_t>(a_col) * block_size + b_col] +=
            qv * cat(z, row, b_col);
        }
      }
    }
  });
  for (int a_col = 0; a_col < block_size; ++a_col) {
    for (int b_col = a_col; b_col < block_size; ++b_col) {
      double dot = 0.0;
      const std::size_t idx = static_cast<std::size_t>(a_col) * block_size + b_col;
      for (int t = 0; t < projection_threads; ++t) {
        dot += projected_thread[static_cast<std::size_t>(t)][idx];
      }
      projected[idx] = dot;
      projected[static_cast<std::size_t>(b_col) * block_size + a_col] = dot;
    }
  }

  std::vector<double> eigvec(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int i = 0; i < block_size; ++i) eigvec[static_cast<std::size_t>(i) * block_size + i] = 1.0;
  for (int sweep = 0; sweep < 80; ++sweep) {
    int p = 0;
    int qidx = 1;
    double max_offdiag = 0.0;
    for (int i = 0; i < block_size; ++i) {
      for (int j = i + 1; j < block_size; ++j) {
        const double v = std::abs(projected[static_cast<std::size_t>(i) * block_size + j]);
        if (v > max_offdiag) {
          max_offdiag = v;
          p = i;
          qidx = j;
        }
      }
    }
    if (max_offdiag < 1e-10) break;

    const double app = projected[static_cast<std::size_t>(p) * block_size + p];
    const double aqq = projected[static_cast<std::size_t>(qidx) * block_size + qidx];
    const double apq = projected[static_cast<std::size_t>(p) * block_size + qidx];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cs = 1.0 / std::sqrt(1.0 + t * t);
    const double sn = t * cs;

    for (int k = 0; k < block_size; ++k) {
      if (k == p || k == qidx) continue;
      const double akp = projected[static_cast<std::size_t>(k) * block_size + p];
      const double akq = projected[static_cast<std::size_t>(k) * block_size + qidx];
      projected[static_cast<std::size_t>(k) * block_size + p] = cs * akp - sn * akq;
      projected[static_cast<std::size_t>(p) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + p];
      projected[static_cast<std::size_t>(k) * block_size + qidx] = sn * akp + cs * akq;
      projected[static_cast<std::size_t>(qidx) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + qidx];
    }
    projected[static_cast<std::size_t>(p) * block_size + p] = cs * cs * app - 2.0 * sn * cs * apq + sn * sn * aqq;
    projected[static_cast<std::size_t>(qidx) * block_size + qidx] = sn * sn * app + 2.0 * sn * cs * apq + cs * cs * aqq;
    projected[static_cast<std::size_t>(p) * block_size + qidx] = 0.0;
    projected[static_cast<std::size_t>(qidx) * block_size + p] = 0.0;

    for (int k = 0; k < block_size; ++k) {
      const double vip = eigvec[static_cast<std::size_t>(k) * block_size + p];
      const double viq = eigvec[static_cast<std::size_t>(k) * block_size + qidx];
      eigvec[static_cast<std::size_t>(k) * block_size + p] = cs * vip - sn * viq;
      eigvec[static_cast<std::size_t>(k) * block_size + qidx] = sn * vip + cs * viq;
    }
  }

  std::vector<int> order(block_size);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](const int a, const int b) {
    return projected[static_cast<std::size_t>(a) * block_size + a] >
           projected[static_cast<std::size_t>(b) * block_size + b];
  });

  for (int c = 0; c < n_components; ++c) {
    const int ritz_col = order[c];
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
      for (int i = begin_row; i < end_row; ++i) {
        double value = 0.0;
        for (int basis = 0; basis < block_size; ++basis) {
          const double coeff = eigvec[static_cast<std::size_t>(basis) * block_size + ritz_col];
          value += cat(q, i, basis) * coeff;
        }
        values[static_cast<std::size_t>(i)] = value;
      }
    });
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
    for (double& v : values) {
      v -= mean;
    }
    for (int i = 0; i < n; ++i) embedding(i, c) = values[static_cast<std::size_t>(i)];
  }
  scale_embedding_max_abs_and_jitter(embedding, 10.0, 1.0e-4, rng);

  return embedding;
}

void add_delta_2d(std::vector<float>& delta,
                  const int n,
                  const int i,
                  const double dx,
                  const double dy,
                  const double scale,
                  const bool update_other,
                  const int other) {
  const double gx = clip_value(scale * dx, -4.0, 4.0);
  const double gy = clip_value(scale * dy, -4.0, 4.0);
  delta[static_cast<std::size_t>(i)] += static_cast<float>(gx);
  delta[static_cast<std::size_t>(n) + i] += static_cast<float>(gy);
  if (update_other) {
    delta[static_cast<std::size_t>(other)] -= static_cast<float>(gx);
    delta[static_cast<std::size_t>(n) + other] -= static_cast<float>(gy);
  }
}

void scale_embedding_sdev(NumericMatrix& embedding, const double target_sdev) {
  const int n = embedding.nrow();
  const int n_components = embedding.ncol();
  if (n < 2 || target_sdev <= 0.0) return;

  for (int c = 0; c < n_components; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += embedding(i, c);
    mean /= static_cast<double>(n);

    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double centered = embedding(i, c) - mean;
      sumsq += centered * centered;
    }
    const double sdev = std::sqrt(sumsq / static_cast<double>(n - 1));
    if (!std::isfinite(sdev) || sdev <= 0.0) continue;
    const double scale = target_sdev / sdev;
    for (int i = 0; i < n; ++i) {
      embedding(i, c) = (embedding(i, c) - mean) * scale;
    }
  }
}

void scale_embedding_max_abs_and_jitter(NumericMatrix& embedding,
                                        const double max_coord,
                                        const double jitter_sd,
                                        std::mt19937& rng) {
  const int n = embedding.nrow();
  const int n_components = embedding.ncol();
  if (n == 0 || n_components == 0 || max_coord <= 0.0) return;

  double max_abs = 0.0;
  for (int c = 0; c < n_components; ++c) {
    for (int i = 0; i < n; ++i) {
      const double value = std::abs(embedding(i, c));
      if (std::isfinite(value)) max_abs = std::max(max_abs, value);
    }
  }
  if (!std::isfinite(max_abs) || max_abs <= 0.0) return;

  const double expansion = max_coord / max_abs;
  std::normal_distribution<double> jitter(0.0, jitter_sd);
  for (int c = 0; c < n_components; ++c) {
    for (int i = 0; i < n; ++i) {
      embedding(i, c) = embedding(i, c) * expansion + jitter(rng);
    }
  }
}

[[maybe_unused]] NumericMatrix optimize_layout(const int n,
                                               const int n_components,
                                               std::vector<Edge> edges,
                                               const NeighborLookup& neighbor_lookup,
                                               const int n_epochs,
                                               const double min_dist,
                                               const int negative_sample_rate,
                                               const double learning_rate,
                                               const double repulsion_strength,
                                               const int spectral_n_iter,
                                               const int n_threads,
                                               const double init_scale,
                                               const int seed,
                                               const bool verbose,
                                               NumericMatrix init_embedding,
                                               const bool use_init) {
  if (edges.empty()) Rcpp::stop("The KNN graph has no usable edges.");

  const double spread = 1.0;
  const auto ab = find_ab_params(spread, min_dist);
  const double a = ab.first;
  const double b = ab.second;
  const double gamma = repulsion_strength;
  // Match the fast UMAP/uwot-style negative sampling path: negatives are drawn
  // from all vertices instead of checking the high-dimensional graph. Avoiding
  // neighbor negatives costs a binary lookup for every negative sample and was
  // the dominant large-data optimizer overhead in profiling.
  const bool avoid_neighbor_negatives = false;
  const double max_weight = std::max_element(
    edges.begin(), edges.end(),
    [](const Edge& x, const Edge& y) { return x.weight < y.weight; }
  )->weight;

  if (n_epochs > 0) {
    const double min_sample_weight = max_weight / static_cast<double>(n_epochs);
    edges.erase(
      std::remove_if(
        edges.begin(), edges.end(),
        [&](const Edge& e) { return e.weight < min_sample_weight; }
      ),
      edges.end()
    );
    if (edges.empty()) Rcpp::stop("The KNN graph has no edges sampled by n_epochs.");
  }

  std::vector<float> epochs_per_sample(edges.size());
  std::vector<float> epoch_of_next_sample(edges.size());
  std::vector<float> epochs_per_negative_sample(edges.size());
  std::vector<float> epoch_of_next_negative_sample(edges.size());

  for (std::size_t i = 0; i < edges.size(); ++i) {
    epochs_per_sample[i] = static_cast<float>(
      max_weight / std::max(static_cast<double>(edges[i].weight), 1e-6)
    );
    epoch_of_next_sample[i] = epochs_per_sample[i];
    if (negative_sample_rate > 0) {
      epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
      epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
    } else {
      epochs_per_negative_sample[i] = std::numeric_limits<float>::infinity();
      epoch_of_next_negative_sample[i] = std::numeric_limits<float>::infinity();
    }
  }

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  NumericMatrix embedding = use_init ?
    Rcpp::clone(init_embedding) :
    initialize_layout(n, n_components, edges, spectral_n_iter, rng);
  if (!use_init && n_components == 2 &&
      std::isfinite(init_scale) && init_scale > 0.0) {
    scale_embedding_sdev(embedding, init_scale);
  }
  if (n_epochs == 0) {
    return embedding;
  }

  double* emb = embedding.begin();
  double* emb_x = emb;
  double* emb_y = emb + n;

  const int threads = effective_cpu_threads(n_threads, static_cast<int>(edges.size()));
  if (n_components == 2 && threads > 1 && n >= 10000) {
    auto run_worker = [&](const int t) {
      const std::size_t begin = edges.size() * static_cast<std::size_t>(t) / threads;
      const std::size_t end = edges.size() * static_cast<std::size_t>(t + 1) / threads;
      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
        for (std::size_t i = begin; i < end; ++i) {
          if (epoch_of_next_sample[i] > epoch) continue;
          const int j = edges[i].head;
          const int k = edges[i].tail;

          const double dx = emb_x[j] - emb_x[k];
          const double dy = emb_y[j] - emb_y[k];
          const double dist_sq = dx * dx + dy * dy;

          double grad_coeff = 0.0;
          if (dist_sq > 0.0) {
            const double dist_pow = umap_pow(dist_sq, b);
            grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                         (a * dist_pow + 1.0);
          }
          const double gx = clip_value(grad_coeff * dx, -4.0, 4.0) * alpha;
          const double gy = clip_value(grad_coeff * dy, -4.0, 4.0) * alpha;
          emb_x[j] += gx;
          emb_y[j] += gy;
          emb_x[k] -= gx;
          emb_y[k] -= gy;
          epoch_of_next_sample[i] += epochs_per_sample[i];

          int n_neg_samples = 0;
          if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
            n_neg_samples = static_cast<int>(
              std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
            );
            n_neg_samples = std::max(0, n_neg_samples);
          }
          for (int p = 0; p < n_neg_samples; ++p) {
            const int neg = deterministic_vertex(n, seed, epoch, i, p);
            if (neg == j) continue;

            const double ndx = emb_x[j] - emb_x[neg];
            const double ndy = emb_y[j] - emb_y[neg];
            const double neg_dist_sq = ndx * ndx + ndy * ndy;
            double repulse = 0.0;
            if (neg_dist_sq > 0.0) {
              repulse = 2.0 * gamma * b /
                        ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
            }
            emb_x[j] += clip_value(repulse * ndx, -4.0, 4.0) * alpha;
            emb_y[j] += clip_value(repulse * ndy, -4.0, 4.0) * alpha;
          }
          if (n_neg_samples > 0) {
            epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
          }
        }

        if (t == 0 && verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
          Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
      workers.emplace_back(run_worker, t);
    }
    run_worker(0);
    for (auto& worker : workers) worker.join();

    return embedding;
  }

  if (n_components == 2 && threads > 1) {
    const int sync_batches = n >= 3000 ?
      std::min(8, std::max(2, 2 * threads)) :
      std::min(4, threads);
    std::vector<std::vector<float>> deltas(
      static_cast<std::size_t>(threads),
      std::vector<float>(static_cast<std::size_t>(n) * 2u, 0.0f)
    );

    ReusableBarrier barrier(threads);
    auto run_worker = [&](const int t) {
      auto& delta = deltas[static_cast<std::size_t>(t)];
      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
        for (int batch = 0; batch < sync_batches; ++batch) {
          std::fill(delta.begin(), delta.end(), 0.0f);

          const std::size_t batch_begin =
            edges.size() * static_cast<std::size_t>(batch) / sync_batches;
          const std::size_t batch_end =
            edges.size() * static_cast<std::size_t>(batch + 1) / sync_batches;
          const std::size_t batch_size = batch_end - batch_begin;
          const std::size_t begin =
            batch_begin + batch_size * static_cast<std::size_t>(t) / threads;
          const std::size_t end =
            batch_begin + batch_size * static_cast<std::size_t>(t + 1) / threads;

          for (std::size_t i = begin; i < end; ++i) {
            if (epoch_of_next_sample[i] > epoch) continue;
            const int j = edges[i].head;
            const int k = edges[i].tail;

            const double dx = emb_x[j] - emb_x[k];
            const double dy = emb_y[j] - emb_y[k];
            const double dist_sq = dx * dx + dy * dy;

            double grad_coeff = 0.0;
            if (dist_sq > 0.0) {
              const double dist_pow = umap_pow(dist_sq, b);
              grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                           (a * dist_pow + 1.0);
            }
            add_delta_2d(delta, n, j, dx, dy, grad_coeff, true, k);
            epoch_of_next_sample[i] += epochs_per_sample[i];

            int n_neg_samples = 0;
            if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
              n_neg_samples = static_cast<int>(
                std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
              );
              n_neg_samples = std::max(0, n_neg_samples);
            }
            for (int p = 0; p < n_neg_samples; ++p) {
              const int neg = avoid_neighbor_negatives ?
                deterministic_non_neighbor(n, neighbor_lookup, j, k, seed, epoch, i, p) :
                deterministic_vertex(n, seed, epoch, i, p);
              if (neg == j) continue;

              const double ndx = emb_x[j] - emb_x[neg];
              const double ndy = emb_y[j] - emb_y[neg];
              const double neg_dist_sq = ndx * ndx + ndy * ndy;
              double repulse = 0.0;
              if (neg_dist_sq > 0.0) {
                repulse = 2.0 * gamma * b /
                          ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
              }
              add_delta_2d(delta, n, j, ndx, ndy, repulse, false, neg);
            }
            if (n_neg_samples > 0) {
              epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
            }
          }

          barrier.wait();
          if (t == 0) {
            for (int worker_id = 0; worker_id < threads; ++worker_id) {
              const auto& worker_delta = deltas[static_cast<std::size_t>(worker_id)];
              for (int i = 0; i < n; ++i) {
                emb_x[i] += alpha * worker_delta[static_cast<std::size_t>(i)];
                emb_y[i] += alpha * worker_delta[static_cast<std::size_t>(n) + i];
              }
            }
          }
          barrier.wait();
        }

        if (t == 0 && verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
          Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
      workers.emplace_back(run_worker, t);
    }
    run_worker(0);
    for (auto& worker : workers) worker.join();

    return embedding;
  }

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      if (epoch_of_next_sample[i] > epoch) continue;
      const int j = edges[i].head;
      const int k = edges[i].tail;

      if (n_components == 2) {
        const double dx = emb_x[j] - emb_x[k];
        const double dy = emb_y[j] - emb_y[k];
        const double dist_sq = dx * dx + dy * dy;

        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        const double gx = clip_value(grad_coeff * dx, -4.0, 4.0);
        const double gy = clip_value(grad_coeff * dy, -4.0, 4.0);
        emb_x[j] += gx * alpha;
        emb_y[j] += gy * alpha;
        emb_x[k] -= gx * alpha;
        emb_y[k] -= gy * alpha;
      } else {
        double dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = embedding(j, c) - embedding(k, c);
          dist_sq += diff * diff;
        }

        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        for (int c = 0; c < n_components; ++c) {
          const double grad = clip_value(grad_coeff * (embedding(j, c) - embedding(k, c)), -4.0, 4.0);
          embedding(j, c) += grad * alpha;
          embedding(k, c) -= grad * alpha;
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      int n_neg_samples = 0;
      if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
        n_neg_samples = static_cast<int>(
          std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
        );
        n_neg_samples = std::max(0, n_neg_samples);
      }
      for (int p = 0; p < n_neg_samples; ++p) {
        const int neg = avoid_neighbor_negatives ?
          deterministic_non_neighbor(n, neighbor_lookup, j, k, seed, epoch, i, p) :
          deterministic_vertex(n, seed, epoch, i, p);
        if (neg == j) continue;

        if (n_components == 2) {
          const double dx = emb_x[j] - emb_x[neg];
          const double dy = emb_y[j] - emb_y[neg];
          const double neg_dist_sq = dx * dx + dy * dy;
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          emb_x[j] += clip_value(repulse * dx, -4.0, 4.0) * alpha;
          emb_y[j] += clip_value(repulse * dy, -4.0, 4.0) * alpha;
        } else {
          double neg_dist_sq = 0.0;
          for (int c = 0; c < n_components; ++c) {
            const double diff = embedding(j, c) - embedding(neg, c);
            neg_dist_sq += diff * diff;
          }
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          for (int c = 0; c < n_components; ++c) {
            const double grad = clip_value(repulse * (embedding(j, c) - embedding(neg, c)), -4.0, 4.0);
            embedding(j, c) += grad * alpha;
          }
        }
      }
      if (n_neg_samples > 0) {
        epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
      }
    }

    if (verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
      Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
    }
  }

  return embedding;
}

template <typename OffsetVec, typename NeighborVec, typename WeightVec>
NumericMatrix optimize_layout_csr(const int n,
                                  const int n_components,
                                  const OffsetVec& offsets,
                                  const NeighborVec& neighbors,
                                  const WeightVec& weights,
                                  const int index_offset,
                                  const int n_epochs,
                                  const double min_dist,
                                  const int negative_sample_rate,
                                  const double learning_rate,
                                  const double repulsion_strength,
                                  const int spectral_n_iter,
                                  const int n_threads,
                                  const double init_scale,
                                  const int seed,
                                  const bool verbose,
                                  const NumericMatrix& init_embedding = NumericMatrix(),
                                  const bool use_init_embedding = false,
                                  const std::vector<float>* precomputed_epochs_per_sample = nullptr,
                                  const float precomputed_max_weight = 0.0f) {
  const int nnz = offsets[n];
  if (nnz == 0) Rcpp::stop("The CSR graph has no usable UMAP edges.");

  const bool has_precomputed_schedule =
    precomputed_epochs_per_sample != nullptr &&
    precomputed_epochs_per_sample->size() == static_cast<std::size_t>(nnz) &&
    std::isfinite(precomputed_max_weight) &&
    precomputed_max_weight > 0.0f;

  double max_weight = has_precomputed_schedule ?
    static_cast<double>(precomputed_max_weight) :
    0.0;
  if (!has_precomputed_schedule) {
    for (int row = 0; row < n; ++row) {
      const int begin = offsets[row];
      const int end = offsets[row + 1];
      for (int pos = begin; pos < end; ++pos) {
        const int nb = neighbors[pos] - index_offset;
        if (nb >= 0 && nb < n && nb != row) {
          max_weight = std::max(max_weight, static_cast<double>(weights[pos]));
        }
      }
    }
  }
  if (max_weight <= 0.0) Rcpp::stop("The CSR graph has no usable UMAP edges.");

  const double min_sample_weight = n_epochs > 0 ?
    max_weight / static_cast<double>(n_epochs) :
    0.0;
  std::vector<int> spectral_rows;
  std::vector<int> spectral_pos;
  std::vector<int> active_rows;
  std::vector<int> active_pos;
  spectral_rows.reserve(static_cast<std::size_t>(std::max(1, nnz / 2)));
  spectral_pos.reserve(static_cast<std::size_t>(std::max(1, nnz / 2)));
  active_rows.reserve(static_cast<std::size_t>(std::max(1, nnz)));
  active_pos.reserve(static_cast<std::size_t>(std::max(1, nnz)));
  for (int row = 0; row < n; ++row) {
    const int begin = offsets[row];
    const int end = offsets[row + 1];
    for (int pos = begin; pos < end; ++pos) {
      const int nb = neighbors[pos] - index_offset;
      if (nb >= 0 && nb < n && nb != row) {
        if (row < nb) {
          spectral_rows.push_back(row);
          spectral_pos.push_back(pos);
        }
        if (weights[pos] >= min_sample_weight) {
          active_rows.push_back(row);
          active_pos.push_back(pos);
        }
      }
    }
  }
  if (active_pos.empty()) Rcpp::stop("The CSR graph has no edges sampled by n_epochs.");
  if (spectral_pos.empty()) {
    spectral_rows = active_rows;
    spectral_pos = active_pos;
  }

  std::vector<int> active_tails(active_pos.size());
  for (std::size_t i = 0; i < active_pos.size(); ++i) {
    active_tails[i] = neighbors[active_pos[i]] - index_offset;
  }

  std::vector<float> epochs_per_sample(active_pos.size());
  std::vector<float> epoch_of_next_sample(active_pos.size());
  std::vector<float> epochs_per_negative_sample(active_pos.size());
  std::vector<float> epoch_of_next_negative_sample(active_pos.size());

  for (std::size_t i = 0; i < active_pos.size(); ++i) {
    if (has_precomputed_schedule) {
      epochs_per_sample[i] =
        (*precomputed_epochs_per_sample)[static_cast<std::size_t>(active_pos[i])];
    } else {
      const double w = std::max(static_cast<double>(weights[active_pos[i]]), 1e-6);
      epochs_per_sample[i] = static_cast<float>(max_weight / w);
    }
    epoch_of_next_sample[i] = epochs_per_sample[i];
    if (negative_sample_rate > 0) {
      epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
      epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
    } else {
      epochs_per_negative_sample[i] = std::numeric_limits<float>::infinity();
      epoch_of_next_negative_sample[i] = std::numeric_limits<float>::infinity();
    }
  }

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  NumericMatrix embedding = use_init_embedding ?
    Rcpp::clone(init_embedding) :
    initialize_layout_csr(
      n, n_components, offsets, neighbors, weights, index_offset,
      spectral_rows, spectral_pos, spectral_n_iter, n_threads, rng
    );
  if (use_init_embedding &&
      (embedding.nrow() != n || embedding.ncol() != n_components)) {
    Rcpp::stop("init_embedding dimensions do not match the CSR graph");
  }
  if (!use_init_embedding && n_components == 2) {
    if (std::isfinite(init_scale) && init_scale > 0.0) {
      scale_embedding_sdev(embedding, init_scale);
    }
  }
  if (n_epochs == 0) {
    return embedding;
  }

  const double spread = 1.0;
  const auto ab = find_ab_params(spread, min_dist);
  const double a = ab.first;
  const double b = ab.second;
  const double gamma = repulsion_strength;
  const bool avoid_neighbor_negatives = false;

  double* emb = embedding.begin();
  double* emb_x = emb;
  double* emb_y = emb + n;

  const int threads = effective_cpu_threads(n_threads, static_cast<int>(active_pos.size()));
  if (n_components == 2 && threads > 1 && n >= 10000) {
    const int* active_rows_ptr = active_rows.data();
    const int* active_tails_ptr = active_tails.data();
    const float* epochs_per_sample_ptr = epochs_per_sample.data();
    const float* epochs_per_negative_sample_ptr = epochs_per_negative_sample.data();
    const float af = static_cast<float>(a);
    const float bf = static_cast<float>(b);
    const float attraction_const = static_cast<float>(-2.0 * a * b);
    const float repulsion_const = static_cast<float>(2.0 * gamma * b);
    const float eps = std::numeric_limits<float>::epsilon();

    std::vector<float> emb_xf(static_cast<std::size_t>(n));
    std::vector<float> emb_yf(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      emb_xf[static_cast<std::size_t>(i)] = static_cast<float>(emb_x[i]);
      emb_yf[static_cast<std::size_t>(i)] = static_cast<float>(emb_y[i]);
    }
    float* x = emb_xf.data();
    float* y = emb_yf.data();

    ReusableBarrier barrier(threads);

    auto run_worker = [&](const int t) {
      const std::size_t begin = active_pos.size() * static_cast<std::size_t>(t) / threads;
      const std::size_t end = active_pos.size() * static_cast<std::size_t>(t + 1) / threads;

      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        TauPrng prng = make_tau_prng(seed, epoch, end, t);
        const float alpha = static_cast<float>(
          learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs)
        );
        for (std::size_t i = begin; i < end; ++i) {
          if (epoch_of_next_sample[i] > epoch) continue;
          const int j = active_rows_ptr[i];
          const int k = active_tails_ptr[i];

          const float dx = x[j] - x[k];
          const float dy = y[j] - y[k];
          const float dist_sq = std::max(eps, dx * dx + dy * dy);

          const float dist_pow = static_cast<float>(umap_pow(dist_sq, bf));
          const float grad_coeff =
            attraction_const * dist_pow / (dist_sq * (af * dist_pow + 1.0f));
          const float gx = clip4f(grad_coeff * dx) * alpha;
          const float gy = clip4f(grad_coeff * dy) * alpha;
          x[j] += gx;
          y[j] += gy;
          x[k] -= gx;
          y[k] -= gy;
          epoch_of_next_sample[i] += epochs_per_sample_ptr[i];

          int n_neg_samples = 0;
          if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
            n_neg_samples = static_cast<int>(
              std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample_ptr[i])
            );
            n_neg_samples = std::max(0, n_neg_samples);
          }
          for (int p = 0; p < n_neg_samples; ++p) {
            const int neg = prng.vertex(n);
            if (neg == j) continue;

            const float ndx = x[j] - x[neg];
            const float ndy = y[j] - y[neg];
            const float neg_dist_sq = std::max(eps, ndx * ndx + ndy * ndy);
            const float neg_pow = static_cast<float>(umap_pow(neg_dist_sq, bf));
            const float repulse =
              repulsion_const / ((0.001f + neg_dist_sq) * (af * neg_pow + 1.0f));
            x[j] += clip4f(repulse * ndx) * alpha;
            y[j] += clip4f(repulse * ndy) * alpha;
          }
          if (n_neg_samples > 0) {
            epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample_ptr[i];
          }
        }

        barrier.wait();
        if (t == 0 && verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
          Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
        }
        barrier.wait();
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
      workers.emplace_back(run_worker, t);
    }
    run_worker(0);
    for (auto& worker : workers) worker.join();

    for (int i = 0; i < n; ++i) {
      emb_x[i] = static_cast<double>(x[i]);
      emb_y[i] = static_cast<double>(y[i]);
    }

    return embedding;
  }

  if (n_components == 2 && threads > 1) {
    const int sync_batches = n >= 3000 ?
      std::min(8, std::max(2, 2 * threads)) :
      std::min(4, threads);
    std::vector<std::vector<float>> deltas(
      static_cast<std::size_t>(threads),
      std::vector<float>(static_cast<std::size_t>(n) * 2u, 0.0f)
    );

    ReusableBarrier barrier(threads);
    auto run_worker = [&](const int t) {
      auto& delta = deltas[static_cast<std::size_t>(t)];
      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
        for (int batch = 0; batch < sync_batches; ++batch) {
          std::fill(delta.begin(), delta.end(), 0.0f);

          const std::size_t batch_begin =
            active_pos.size() * static_cast<std::size_t>(batch) / sync_batches;
          const std::size_t batch_end =
            active_pos.size() * static_cast<std::size_t>(batch + 1) / sync_batches;
          const std::size_t batch_size = batch_end - batch_begin;
          const std::size_t begin =
            batch_begin + batch_size * static_cast<std::size_t>(t) / threads;
          const std::size_t end =
            batch_begin + batch_size * static_cast<std::size_t>(t + 1) / threads;

          for (std::size_t i = begin; i < end; ++i) {
            if (epoch_of_next_sample[i] > epoch) continue;
            const int j = active_rows[i];
            const int k = active_tails[i];

            const double dx = emb_x[j] - emb_x[k];
            const double dy = emb_y[j] - emb_y[k];
            const double dist_sq = dx * dx + dy * dy;

            double grad_coeff = 0.0;
            if (dist_sq > 0.0) {
              const double dist_pow = umap_pow(dist_sq, b);
              grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                           (a * dist_pow + 1.0);
            }
            add_delta_2d(delta, n, j, dx, dy, grad_coeff, true, k);
            epoch_of_next_sample[i] += epochs_per_sample[i];

            int n_neg_samples = 0;
            if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
              n_neg_samples = static_cast<int>(
                std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
              );
              n_neg_samples = std::max(0, n_neg_samples);
            }
            for (int p = 0; p < n_neg_samples; ++p) {
              const int neg = avoid_neighbor_negatives ?
                deterministic_non_neighbor_csr(
                  n, offsets, neighbors, index_offset, j, k, seed, epoch, i, p
                ) :
                deterministic_vertex(n, seed, epoch, i, p);
              if (neg == j) continue;

              const double ndx = emb_x[j] - emb_x[neg];
              const double ndy = emb_y[j] - emb_y[neg];
              const double neg_dist_sq = ndx * ndx + ndy * ndy;
              double repulse = 0.0;
              if (neg_dist_sq > 0.0) {
                repulse = 2.0 * gamma * b /
                          ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
              }
              add_delta_2d(delta, n, j, ndx, ndy, repulse, false, neg);
            }
            if (n_neg_samples > 0) {
              epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
            }
          }

          barrier.wait();
          if (t == 0) {
            for (int worker_id = 0; worker_id < threads; ++worker_id) {
              const auto& worker_delta = deltas[static_cast<std::size_t>(worker_id)];
              for (int i = 0; i < n; ++i) {
                emb_x[i] += alpha * worker_delta[static_cast<std::size_t>(i)];
                emb_y[i] += alpha * worker_delta[static_cast<std::size_t>(n) + i];
              }
            }
          }
          barrier.wait();
        }

        if (t == 0 && verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
          Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
      workers.emplace_back(run_worker, t);
    }
    run_worker(0);
    for (auto& worker : workers) worker.join();

    return embedding;
  }

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
    for (std::size_t i = 0; i < active_pos.size(); ++i) {
      if (epoch_of_next_sample[i] > epoch) continue;
      const int j = active_rows[i];
      const int k = active_tails[i];

      if (n_components == 2) {
        const double dx = emb_x[j] - emb_x[k];
        const double dy = emb_y[j] - emb_y[k];
        const double dist_sq = dx * dx + dy * dy;

        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        const double gx = clip_value(grad_coeff * dx, -4.0, 4.0);
        const double gy = clip_value(grad_coeff * dy, -4.0, 4.0);
        emb_x[j] += gx * alpha;
        emb_y[j] += gy * alpha;
        emb_x[k] -= gx * alpha;
        emb_y[k] -= gy * alpha;
      } else {
        double dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = embedding(j, c) - embedding(k, c);
          dist_sq += diff * diff;
        }

        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        for (int c = 0; c < n_components; ++c) {
          const double grad = clip_value(grad_coeff * (embedding(j, c) - embedding(k, c)), -4.0, 4.0);
          embedding(j, c) += grad * alpha;
          embedding(k, c) -= grad * alpha;
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      int n_neg_samples = 0;
      if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
        n_neg_samples = static_cast<int>(
          std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
        );
        n_neg_samples = std::max(0, n_neg_samples);
      }
      for (int p = 0; p < n_neg_samples; ++p) {
        const int neg = avoid_neighbor_negatives ?
          deterministic_non_neighbor_csr(
            n, offsets, neighbors, index_offset, j, k, seed, epoch, i, p
          ) :
          deterministic_vertex(n, seed, epoch, i, p);
        if (neg == j) continue;

        if (n_components == 2) {
          const double dx = emb_x[j] - emb_x[neg];
          const double dy = emb_y[j] - emb_y[neg];
          const double neg_dist_sq = dx * dx + dy * dy;
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          emb_x[j] += clip_value(repulse * dx, -4.0, 4.0) * alpha;
          emb_y[j] += clip_value(repulse * dy, -4.0, 4.0) * alpha;
        } else {
          double neg_dist_sq = 0.0;
          for (int c = 0; c < n_components; ++c) {
            const double diff = embedding(j, c) - embedding(neg, c);
            neg_dist_sq += diff * diff;
          }
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          for (int c = 0; c < n_components; ++c) {
            const double grad = clip_value(repulse * (embedding(j, c) - embedding(neg, c)), -4.0, 4.0);
            embedding(j, c) += grad * alpha;
          }
        }
      }
      if (n_neg_samples > 0) {
        epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
      }
    }

    if (verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
      Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
    }
  }

  return embedding;
}

std::vector<unsigned char> update_rows_to_mask(const int n,
                                               const IntegerVector& update_rows) {
  std::vector<unsigned char> mask(static_cast<std::size_t>(n), 0u);
  for (int i = 0; i < update_rows.size(); ++i) {
    const int row = update_rows[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("update_rows must contain 1-based row indices");
    mask[static_cast<std::size_t>(row)] = 1u;
  }
  return mask;
}

int update_mask_count(const std::vector<unsigned char>& mask) {
  int count = 0;
  for (const unsigned char value : mask) {
    if (value != 0u) ++count;
  }
  return count;
}

NumericMatrix optimize_layout_csr_masked(const int n,
                                         const int n_components,
                                         const std::vector<int>& offsets,
                                         const std::vector<int>& neighbors,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& graph_epochs_per_sample,
                                         const float graph_max_weight,
                                         const std::vector<unsigned char>& update_mask,
                                         const int n_epochs,
                                         const double min_dist,
                                         const int negative_sample_rate,
                                         const double learning_rate,
                                         const double repulsion_strength,
                                         const int n_threads,
                                         const int seed,
                                         const bool verbose,
                                         const NumericMatrix& init_embedding) {
  if (init_embedding.nrow() != n || init_embedding.ncol() != n_components) {
    Rcpp::stop("init_embedding dimensions do not match the graph");
  }
  if (update_mask.size() != static_cast<std::size_t>(n)) {
    Rcpp::stop("update mask length does not match the graph");
  }
  if (n_epochs < 1 || update_mask_count(update_mask) == 0) {
    return Rcpp::clone(init_embedding);
  }

  const int nnz = offsets[static_cast<std::size_t>(n)];
  if (nnz == 0) return Rcpp::clone(init_embedding);

  double max_weight = graph_max_weight;
  if (!(std::isfinite(max_weight) && max_weight > 0.0)) {
    max_weight = 0.0;
    for (int row = 0; row < n; ++row) {
      const int begin = offsets[static_cast<std::size_t>(row)];
      const int end = offsets[static_cast<std::size_t>(row + 1)];
      for (int pos = begin; pos < end; ++pos) {
        const int nb = neighbors[static_cast<std::size_t>(pos)];
        if (nb >= 0 && nb < n && nb != row) {
          max_weight = std::max(max_weight, static_cast<double>(weights[static_cast<std::size_t>(pos)]));
        }
      }
    }
  }
  if (max_weight <= 0.0) return Rcpp::clone(init_embedding);

  const double min_sample_weight = max_weight / static_cast<double>(n_epochs);
  const bool has_precomputed_schedule =
    graph_epochs_per_sample.size() == static_cast<std::size_t>(nnz);

  std::vector<int> active_rows;
  std::vector<int> active_pos;
  active_rows.reserve(static_cast<std::size_t>(std::max(1, nnz / 4)));
  active_pos.reserve(static_cast<std::size_t>(std::max(1, nnz / 4)));
  for (int row = 0; row < n; ++row) {
    if (update_mask[static_cast<std::size_t>(row)] == 0u) continue;
    const int begin = offsets[static_cast<std::size_t>(row)];
    const int end = offsets[static_cast<std::size_t>(row + 1)];
    for (int pos = begin; pos < end; ++pos) {
      const int nb = neighbors[static_cast<std::size_t>(pos)];
      const float weight = weights[static_cast<std::size_t>(pos)];
      if (nb >= 0 && nb < n && nb != row && weight >= min_sample_weight) {
        active_rows.push_back(row);
        active_pos.push_back(pos);
      }
    }
  }
  if (active_pos.empty()) {
    return Rcpp::clone(init_embedding);
  }

  std::vector<int> active_tails(active_pos.size());
  std::vector<float> epochs_per_sample(active_pos.size());
  std::vector<float> epoch_of_next_sample(active_pos.size());
  std::vector<float> epochs_per_negative_sample(active_pos.size());
  std::vector<float> epoch_of_next_negative_sample(active_pos.size());
  for (std::size_t i = 0; i < active_pos.size(); ++i) {
    const int pos = active_pos[i];
    active_tails[i] = neighbors[static_cast<std::size_t>(pos)];
    if (has_precomputed_schedule) {
      epochs_per_sample[i] = graph_epochs_per_sample[static_cast<std::size_t>(pos)];
    } else {
      const double w = std::max(static_cast<double>(weights[static_cast<std::size_t>(pos)]), 1e-6);
      epochs_per_sample[i] = static_cast<float>(max_weight / w);
    }
    epoch_of_next_sample[i] = epochs_per_sample[i];
    if (negative_sample_rate > 0) {
      epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
      epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
    } else {
      epochs_per_negative_sample[i] = std::numeric_limits<float>::infinity();
      epoch_of_next_negative_sample[i] = std::numeric_limits<float>::infinity();
    }
  }

  NumericMatrix embedding = Rcpp::clone(init_embedding);
  const double spread = 1.0;
  const auto ab = find_ab_params(spread, min_dist);
  const double a = ab.first;
  const double b = ab.second;
  const double gamma = repulsion_strength;
  double* emb = embedding.begin();
  double* emb_x = emb;
  double* emb_y = emb + n;
  const int threads = effective_cpu_threads(n_threads, static_cast<int>(active_pos.size()));

  if (n_components == 2 && threads > 1 && n >= 10000) {
    const int* active_rows_ptr = active_rows.data();
    const int* active_tails_ptr = active_tails.data();
    const float* epochs_per_sample_ptr = epochs_per_sample.data();
    const float* epochs_per_negative_sample_ptr = epochs_per_negative_sample.data();
    const unsigned char* mask = update_mask.data();
    const float af = static_cast<float>(a);
    const float bf = static_cast<float>(b);
    const float attraction_const = static_cast<float>(-2.0 * a * b);
    const float repulsion_const = static_cast<float>(2.0 * gamma * b);
    const float eps = std::numeric_limits<float>::epsilon();

    std::vector<float> emb_xf(static_cast<std::size_t>(n));
    std::vector<float> emb_yf(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      emb_xf[static_cast<std::size_t>(i)] = static_cast<float>(emb_x[i]);
      emb_yf[static_cast<std::size_t>(i)] = static_cast<float>(emb_y[i]);
    }
    float* x = emb_xf.data();
    float* y = emb_yf.data();

    ReusableBarrier barrier(threads);
    auto run_worker = [&](const int t) {
      const std::size_t begin = active_pos.size() * static_cast<std::size_t>(t) / threads;
      const std::size_t end = active_pos.size() * static_cast<std::size_t>(t + 1) / threads;
      for (int epoch = 0; epoch < n_epochs; ++epoch) {
        TauPrng prng = make_tau_prng(seed, epoch, end, t);
        const float alpha = static_cast<float>(
          learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs)
        );
        for (std::size_t i = begin; i < end; ++i) {
          if (epoch_of_next_sample[i] > epoch) continue;
          const int j = active_rows_ptr[i];
          const int k = active_tails_ptr[i];

          const float dx = x[j] - x[k];
          const float dy = y[j] - y[k];
          const float dist_sq = std::max(eps, dx * dx + dy * dy);
          const float dist_pow = static_cast<float>(umap_pow(dist_sq, bf));
          const float grad_coeff =
            attraction_const * dist_pow / (dist_sq * (af * dist_pow + 1.0f));
          const float gx = clip4f(grad_coeff * dx) * alpha;
          const float gy = clip4f(grad_coeff * dy) * alpha;
          x[j] += gx;
          y[j] += gy;
          if (mask[static_cast<std::size_t>(k)] != 0u) {
            x[k] -= gx;
            y[k] -= gy;
          }
          epoch_of_next_sample[i] += epochs_per_sample_ptr[i];

          int n_neg_samples = 0;
          if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
            n_neg_samples = static_cast<int>(
              std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample_ptr[i])
            );
            n_neg_samples = std::max(0, n_neg_samples);
          }
          for (int p = 0; p < n_neg_samples; ++p) {
            const int neg = prng.vertex(n);
            if (neg == j) continue;
            const float ndx = x[j] - x[neg];
            const float ndy = y[j] - y[neg];
            const float neg_dist_sq = std::max(eps, ndx * ndx + ndy * ndy);
            const float neg_pow = static_cast<float>(umap_pow(neg_dist_sq, bf));
            const float repulse =
              repulsion_const / ((0.001f + neg_dist_sq) * (af * neg_pow + 1.0f));
            x[j] += clip4f(repulse * ndx) * alpha;
            y[j] += clip4f(repulse * ndy) * alpha;
          }
          if (n_neg_samples > 0) {
            epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample_ptr[i];
          }
        }
        barrier.wait();
        if (t == 0 && verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
          Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
        }
        barrier.wait();
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
      workers.emplace_back(run_worker, t);
    }
    run_worker(0);
    for (auto& worker : workers) worker.join();

    for (int i = 0; i < n; ++i) {
      emb_x[i] = static_cast<double>(x[i]);
      emb_y[i] = static_cast<double>(y[i]);
    }
    return embedding;
  }

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    const double alpha = learning_rate * (1.0 - static_cast<double>(epoch) / n_epochs);
    for (std::size_t i = 0; i < active_pos.size(); ++i) {
      if (epoch_of_next_sample[i] > epoch) continue;
      const int j = active_rows[i];
      const int k = active_tails[i];
      const bool update_tail = update_mask[static_cast<std::size_t>(k)] != 0u;

      if (n_components == 2) {
        const double dx = emb_x[j] - emb_x[k];
        const double dy = emb_y[j] - emb_y[k];
        const double dist_sq = dx * dx + dy * dy;
        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        const double gx = clip_value(grad_coeff * dx, -4.0, 4.0) * alpha;
        const double gy = clip_value(grad_coeff * dy, -4.0, 4.0) * alpha;
        emb_x[j] += gx;
        emb_y[j] += gy;
        if (update_tail) {
          emb_x[k] -= gx;
          emb_y[k] -= gy;
        }
      } else {
        double dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = embedding(j, c) - embedding(k, c);
          dist_sq += diff * diff;
        }
        double grad_coeff = 0.0;
        if (dist_sq > 0.0) {
          const double dist_pow = umap_pow(dist_sq, b);
          grad_coeff = -2.0 * a * b * (dist_pow / dist_sq) /
                       (a * dist_pow + 1.0);
        }
        for (int c = 0; c < n_components; ++c) {
          const double grad = clip_value(
            grad_coeff * (embedding(j, c) - embedding(k, c)), -4.0, 4.0
          ) * alpha;
          embedding(j, c) += grad;
          if (update_tail) embedding(k, c) -= grad;
        }
      }
      epoch_of_next_sample[i] += epochs_per_sample[i];

      int n_neg_samples = 0;
      if (negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[i]) {
        n_neg_samples = static_cast<int>(
          std::floor((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
        );
        n_neg_samples = std::max(0, n_neg_samples);
      }
      for (int p = 0; p < n_neg_samples; ++p) {
        const int neg = deterministic_vertex(n, seed, epoch, i, p);
        if (neg == j) continue;
        if (n_components == 2) {
          const double dx = emb_x[j] - emb_x[neg];
          const double dy = emb_y[j] - emb_y[neg];
          const double neg_dist_sq = dx * dx + dy * dy;
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          emb_x[j] += clip_value(repulse * dx, -4.0, 4.0) * alpha;
          emb_y[j] += clip_value(repulse * dy, -4.0, 4.0) * alpha;
        } else {
          double neg_dist_sq = 0.0;
          for (int c = 0; c < n_components; ++c) {
            const double diff = embedding(j, c) - embedding(neg, c);
            neg_dist_sq += diff * diff;
          }
          double repulse = 0.0;
          if (neg_dist_sq > 0.0) {
            repulse = 2.0 * gamma * b /
                      ((0.001 + neg_dist_sq) * (a * umap_pow(neg_dist_sq, b) + 1.0));
          }
          for (int c = 0; c < n_components; ++c) {
            const double grad = clip_value(
              repulse * (embedding(j, c) - embedding(neg, c)), -4.0, 4.0
            );
            embedding(j, c) += grad * alpha;
          }
        }
      }
      if (n_neg_samples > 0) {
        epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
      }
    }
    if (verbose && (epoch == 0 || epoch == n_epochs - 1 || (epoch + 1) % 50 == 0)) {
      Rcpp::Rcout << "epoch " << (epoch + 1) << "/" << n_epochs << "\n";
    }
  }

  return embedding;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List knn_connectivity_range_cpp(IntegerMatrix indices,
                                      int col_start,
                                      int n_cols) {
  const int n = indices.nrow();
  const int matrix_k = indices.ncol();
  if (col_start < 0 || n_cols < 1 || col_start + n_cols > matrix_k) {
    Rcpp::stop("invalid KNN column range");
  }
  const int k = n_cols;
  if (n < 2) Rcpp::stop("indices must have at least two rows");
  if (k < 1) Rcpp::stop("indices must have at least one neighbor column");

  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int idx = indices(i, j + col_start);
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const int index_offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  DisjointSet dsu(n);
  int valid_edges = 0;
  int self_edges = 0;
  int invalid_edges = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int neighbor = indices(i, j + col_start) - index_offset;
      if (neighbor < 0 || neighbor >= n) {
        ++invalid_edges;
      } else if (neighbor == i) {
        ++self_edges;
      } else {
        dsu.unite(i, neighbor);
        ++valid_edges;
      }
    }
  }

  std::vector<int> component_sizes(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    ++component_sizes[static_cast<std::size_t>(dsu.find(i))];
  }

  int component_count = 0;
  int singleton_count = 0;
  int largest_component_size = 0;
  for (int size : component_sizes) {
    if (size <= 0) continue;
    ++component_count;
    if (size == 1) ++singleton_count;
    largest_component_size = std::max(largest_component_size, size);
  }

  const double largest_fraction =
    static_cast<double>(largest_component_size) / static_cast<double>(n);

  return Rcpp::List::create(
    Rcpp::Named("n") = n,
    Rcpp::Named("k") = k,
    Rcpp::Named("connected") = component_count == 1,
    Rcpp::Named("component_count") = component_count,
    Rcpp::Named("largest_component_size") = largest_component_size,
    Rcpp::Named("largest_component_fraction") = largest_fraction,
    Rcpp::Named("singleton_count") = singleton_count,
    Rcpp::Named("valid_edge_count") = valid_edges,
    Rcpp::Named("self_edge_count") = self_edges,
    Rcpp::Named("invalid_edge_count") = invalid_edges,
    Rcpp::Named("index_offset") = index_offset
  );
}

// [[Rcpp::export]]
Rcpp::List knn_connectivity_cpp(IntegerMatrix indices) {
  return knn_connectivity_range_cpp(indices, 0, indices.ncol());
}

// [[Rcpp::export]]
Rcpp::List umap_graph_csr_cpp(IntegerMatrix indices,
                              NumericMatrix distances,
                              int col_start,
                              int n_cols,
                              int edge_budget,
                              int n_threads) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (col_start < 0 || n_cols < 1 || col_start + n_cols > indices.ncol()) {
    Rcpp::stop("invalid KNN column range");
  }
  if (edge_budget < 1) Rcpp::stop("edge_budget must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  CsrGraphNative graph = build_graph_csr_native(
    indices, distances, edge_budget, n_threads, col_start, n_cols
  );

  return Rcpp::List::create(
    Rcpp::Named("offsets") = Rcpp::wrap(graph.offsets),
    Rcpp::Named("neighbors") = Rcpp::wrap(graph.neighbors),
    Rcpp::Named("weights") = Rcpp::wrap(graph.weights),
    Rcpp::Named("epochs_per_sample") = Rcpp::wrap(graph.epochs_per_sample),
    Rcpp::Named("max_weight") = graph.max_weight,
    Rcpp::Named("n") = indices.nrow(),
    Rcpp::Named("nnz") = static_cast<int>(graph.neighbors.size())
  );
}

// [[Rcpp::export]]
NumericMatrix fast_knn_umap_range_cpp(IntegerMatrix indices,
                                      NumericMatrix distances,
                                      int col_start,
                                      int n_cols,
                                      int n_components,
                                      int n_epochs,
                                      double min_dist,
                                      int negative_sample_rate,
                                      double learning_rate,
                                      double repulsion_strength,
                                      int spectral_n_iter,
                                      int n_threads,
                                      double init_scale,
                                      int seed,
                                      bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (col_start < 0 || n_cols < 1 || col_start + n_cols > indices.ncol()) {
    Rcpp::stop("invalid KNN column range");
  }
  if (n_components < 1) Rcpp::stop("n_components must be positive");
  if (n_epochs < 0) Rcpp::stop("n_epochs must be non-negative");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  const int n = indices.nrow();
  CsrGraphNative graph = build_graph_csr_native(
    indices, distances, n_cols, n_threads, col_start, n_cols
  );
  return optimize_layout_csr(
    n, n_components, graph.offsets, graph.neighbors, graph.weights, 0, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, spectral_n_iter,
    n_threads, init_scale, seed, verbose, NumericMatrix(), false,
    &graph.epochs_per_sample, graph.max_weight
  );
}

// [[Rcpp::export]]
NumericMatrix fast_knn_umap_cpp(IntegerMatrix indices,
                                NumericMatrix distances,
                                int n_components,
                                int n_epochs,
                                double min_dist,
                                int negative_sample_rate,
                                double learning_rate,
                                double repulsion_strength,
                                int spectral_n_iter,
                                int n_threads,
                                double init_scale,
                                int seed,
                                bool verbose) {
  return fast_knn_umap_range_cpp(
    indices, distances, 0, indices.ncol(), n_components, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, spectral_n_iter,
    n_threads, init_scale, seed, verbose
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_refine_range_cpp(IntegerMatrix indices,
                                        NumericMatrix distances,
                                        NumericMatrix init_embedding,
                                        int col_start,
                                        int n_cols,
                                        int n_epochs,
                                        double min_dist,
                                        int negative_sample_rate,
                                        double learning_rate,
                                        double repulsion_strength,
                                        int n_threads,
                                        int seed,
                                        bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (col_start < 0 || n_cols < 1 || col_start + n_cols > indices.ncol()) {
    Rcpp::stop("invalid KNN column range");
  }
  if (init_embedding.nrow() != indices.nrow()) {
    Rcpp::stop("init_embedding row count must match indices");
  }
  if (init_embedding.ncol() < 1) Rcpp::stop("init_embedding must have at least one column");
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  const int n = indices.nrow();
  const int n_components = init_embedding.ncol();
  CsrGraphNative graph = build_graph_csr_native(
    indices, distances, n_cols, n_threads, col_start, n_cols
  );
  return optimize_layout_csr(
    n, n_components, graph.offsets, graph.neighbors, graph.weights, 0, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, 1,
    n_threads, R_NaReal, seed, verbose, init_embedding, true,
    &graph.epochs_per_sample, graph.max_weight
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_refine_cpp(IntegerMatrix indices,
                                  NumericMatrix distances,
                                  NumericMatrix init_embedding,
                                  int n_epochs,
                                  double min_dist,
                                  int negative_sample_rate,
                                  double learning_rate,
                                  double repulsion_strength,
                                  int n_threads,
                                  int seed,
                                  bool verbose) {
  return knn_umap_refine_range_cpp(
    indices, distances, init_embedding, 0, indices.ncol(), n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, n_threads, seed, verbose
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_refine_masked_cpp(IntegerMatrix indices,
                                         NumericMatrix distances,
                                         NumericMatrix init_embedding,
                                         IntegerVector update_rows,
                                         int n_epochs,
                                         double min_dist,
                                         int negative_sample_rate,
                                         double learning_rate,
                                         double repulsion_strength,
                                         int n_threads,
                                         int seed,
                                         bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (init_embedding.nrow() != indices.nrow()) {
    Rcpp::stop("init_embedding row count must match indices");
  }
  if (init_embedding.ncol() < 1) Rcpp::stop("init_embedding must have at least one column");
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  const int n = indices.nrow();
  if (update_rows.size() == 0) return Rcpp::clone(init_embedding);
  std::vector<unsigned char> update_mask = update_rows_to_mask(n, update_rows);
  if (update_mask_count(update_mask) >= n) {
    return knn_umap_refine_cpp(
      indices, distances, init_embedding, n_epochs, min_dist, negative_sample_rate,
      learning_rate, repulsion_strength, n_threads, seed, verbose
    );
  }

  CsrGraphNative graph = build_graph_csr_native(
    indices, distances, indices.ncol(), n_threads, 0, indices.ncol()
  );
  return optimize_layout_csr_masked(
    n, init_embedding.ncol(), graph.offsets, graph.neighbors, graph.weights,
    graph.epochs_per_sample, graph.max_weight, update_mask, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, n_threads, seed,
    verbose, init_embedding
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_refine_rows_cpp(IntegerMatrix indices,
                                       NumericMatrix distances,
                                       IntegerVector row_ids,
                                       NumericMatrix init_embedding,
                                       int n_epochs,
                                       double min_dist,
                                       int negative_sample_rate,
                                       double learning_rate,
                                       double repulsion_strength,
                                       int n_threads,
                                       int seed,
                                       bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (row_ids.size() != indices.nrow()) {
    Rcpp::stop("row_ids length must match the number of KNN rows");
  }
  if (init_embedding.ncol() < 1) Rcpp::stop("init_embedding must have at least one column");
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  const int n = init_embedding.nrow();
  const int m = indices.nrow();
  const int k = indices.ncol();
  if (n < 2) Rcpp::stop("init_embedding must have at least two rows");
  if (m < 1) return Rcpp::clone(init_embedding);
  if (k < 1) Rcpp::stop("indices must have at least one neighbor column");

  std::vector<int> rows(static_cast<std::size_t>(m));
  std::vector<unsigned char> update_mask(static_cast<std::size_t>(n), 0u);
  for (int i = 0; i < m; ++i) {
    const int row = row_ids[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("row_ids must contain 1-based row indices");
    rows[static_cast<std::size_t>(i)] = row;
    update_mask[static_cast<std::size_t>(row)] = 1u;
  }

  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      const int idx = indices(i, j);
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const int index_offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  std::vector<float> distance_values = copy_distances_float(distances, n_threads, 0, k);
  const FloatDistanceView distance_view{distance_values.data(), m, k, k};
  std::vector<float> sigmas;
  std::vector<float> rhos;
  smooth_knn_dist(distance_view, sigmas, rhos, n_threads);

  CsrGraphNative graph;
  graph.offsets.assign(static_cast<std::size_t>(n) + 1u, 0);
  std::vector<int> counts(static_cast<std::size_t>(n), 0);
  for (int local = 0; local < m; ++local) {
    const int row = rows[static_cast<std::size_t>(local)];
    int count = 0;
    for (int j = 0; j < k; ++j) {
      const int nb = indices(local, j) - index_offset;
      const float d = distance_view(local, j);
      if (nb >= 0 && nb < n && nb != row && std::isfinite(d) && d >= 0.0f) {
        ++count;
      }
    }
    counts[static_cast<std::size_t>(row)] += count;
  }
  for (int row = 0; row < n; ++row) {
    graph.offsets[static_cast<std::size_t>(row + 1)] =
      graph.offsets[static_cast<std::size_t>(row)] + counts[static_cast<std::size_t>(row)];
  }
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  if (nnz == 0) return Rcpp::clone(init_embedding);
  graph.neighbors.resize(static_cast<std::size_t>(nnz));
  graph.weights.resize(static_cast<std::size_t>(nnz));
  std::vector<int> cursor = graph.offsets;

  for (int local = 0; local < m; ++local) {
    const int row = rows[static_cast<std::size_t>(local)];
    const float rho = rhos[static_cast<std::size_t>(local)];
    const float sigma = sigmas[static_cast<std::size_t>(local)];
    for (int j = 0; j < k; ++j) {
      const int nb = indices(local, j) - index_offset;
      const float d = distance_view(local, j);
      if (nb < 0 || nb >= n || nb == row || !std::isfinite(d) || d < 0.0f) continue;
      const float weight = d <= rho ? 1.0f : std::exp(-(d - rho) / sigma);
      if (!std::isfinite(weight) || weight <= 0.0f) continue;
      const int pos = cursor[static_cast<std::size_t>(row)]++;
      graph.neighbors[static_cast<std::size_t>(pos)] = nb;
      graph.weights[static_cast<std::size_t>(pos)] = weight;
    }
  }
  attach_csr_epoch_schedule(graph);

  return optimize_layout_csr_masked(
    n, init_embedding.ncol(), graph.offsets, graph.neighbors, graph.weights,
    graph.epochs_per_sample, graph.max_weight, update_mask, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, n_threads, seed,
    verbose, init_embedding
  );
}

// [[Rcpp::export]]
NumericMatrix fast_knn_umap_csr_cpp(IntegerVector offsets,
                                    IntegerVector neighbors,
                                    NumericVector weights,
                                    int n_components,
                                    int n_epochs,
                                    double min_dist,
                                    int negative_sample_rate,
                                    double learning_rate,
                                    double repulsion_strength,
                                    int spectral_n_iter,
                                    int n_threads,
                                    double init_scale,
                                    int seed,
                                    bool verbose) {
  if (offsets.size() < 2) Rcpp::stop("CSR offsets must have length at least two");
  if (neighbors.size() != weights.size()) {
    Rcpp::stop("CSR neighbors and weights must have the same length");
  }
  if (n_components < 1) Rcpp::stop("n_components must be positive");
  if (n_epochs < 0) Rcpp::stop("n_epochs must be non-negative");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (n_threads < 1) Rcpp::stop("n_threads must be positive");

  const int n = offsets.size() - 1;
  const int index_offset = validate_csr_inputs(offsets, neighbors, weights);
  return optimize_layout_csr(
    n, n_components, offsets, neighbors, weights, index_offset, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, spectral_n_iter,
    n_threads, init_scale, seed, verbose
  );
}

// [[Rcpp::export]]
Rcpp::List umap_auto_parameters_cpp(NumericMatrix distances,
                                    int n_neighbors,
                                    std::string backend) {
  const int n = distances.nrow();
  const int k = std::max(1, std::min(n_neighbors, distances.ncol()));
  if (n < 2 || k < 1) Rcpp::stop("KNN distances must have at least two rows and one neighbor column.");

  double sum = 0.0;
  double sum_sq = 0.0;
  int count = 0;
  std::vector<double> d15;
  std::vector<double> d30;
  std::vector<double> d50;
  d15.reserve(static_cast<std::size_t>(n));
  d30.reserve(static_cast<std::size_t>(n));
  d50.reserve(static_cast<std::size_t>(n));

  const int col15 = std::min(k, 15) - 1;
  const int col30 = std::min(k, 30) - 1;
  const int col50 = std::min(k, 50) - 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      if (std::isfinite(d)) {
        sum += d;
        sum_sq += d * d;
        ++count;
      }
    }
    const double v15 = distances(i, col15);
    const double v30 = distances(i, col30);
    const double v50 = distances(i, col50);
    if (std::isfinite(v15)) d15.push_back(v15);
    if (std::isfinite(v30)) d30.push_back(v30);
    if (std::isfinite(v50)) d50.push_back(v50);
  }

  auto median_or_na = [](std::vector<double>& values) -> double {
    if (values.empty()) return NA_REAL;
    const std::size_t mid = values.size() / 2u;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double med = values[mid];
    if ((values.size() & 1u) == 0u) {
      const auto lower = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
      med = 0.5 * (med + *lower);
    }
    return med;
  };

  const double mean = count > 0 ? sum / static_cast<double>(count) : NA_REAL;
  double cv = NA_REAL;
  if (count > 1 && std::isfinite(mean) && mean > 0.0) {
    const double var = std::max(0.0, (sum_sq / static_cast<double>(count)) - mean * mean);
    cv = std::sqrt(var) / mean;
  }
  const double med15 = median_or_na(d15);
  const double med30 = median_or_na(d30);
  const double med50 = median_or_na(d50);
  const double denom = std::isfinite(med15) && med15 > 0.0 ?
    med15 :
    std::numeric_limits<double>::epsilon();
  const double ratio30 = std::isfinite(med30) ? med30 / denom : NA_REAL;
  const double ratio50 = std::isfinite(med50) ? med50 / denom : NA_REAL;

  const bool very_large = n >= 10000;
  int n_epochs = very_large ? 200 : 500;
  double min_dist = 0.01;
  int negative_sample_rate = 5;
  int spectral_n_iter = very_large ? (k <= 15 ? 30 : 20) : (n >= 500 ? 60 : 50);
  double init_scale = NA_REAL;
  double learning_rate = 1.0;
  std::string rule = very_large ? "uwot_fast_sgd_compatible_cpp_profile" : "uwot_default_cpp_profile";

  if (very_large && std::isfinite(ratio50) && std::isfinite(cv) && ratio50 >= 1.25 && cv >= 1.0) {
    min_dist = 0.1;
    init_scale = 5.0;
    learning_rate = 1.25;
    rule = "wide_shell_balanced_quality_speed_cpp_profile";
  } else if (very_large && std::isfinite(cv) && cv >= 0.60) {
    n_epochs = std::max(n_epochs, 300);
    rule = "high_variability_more_epochs_cpp_profile";
  }

  int thread_cap = very_large || (k >= 15 && n >= 500) ? 4 : (k >= 15 && n >= 200 ? 3 : 1);
  if (backend == "metal" || backend == "cuda" || backend == "gpu") thread_cap = 4;

  return Rcpp::List::create(
    Rcpp::Named("n_epochs") = n_epochs,
    Rcpp::Named("min_dist") = min_dist,
    Rcpp::Named("negative_sample_rate") = negative_sample_rate,
    Rcpp::Named("learning_rate") = learning_rate,
    Rcpp::Named("spectral_n_iter") = spectral_n_iter,
    Rcpp::Named("init_scale") = init_scale,
    Rcpp::Named("n_threads_cap") = thread_cap,
    Rcpp::Named("knn_distance_cv") = cv,
    Rcpp::Named("knn_distance_ratio_30_15") = ratio30,
    Rcpp::Named("knn_distance_ratio_50_15") = ratio50,
    Rcpp::Named("rule") = rule
  );
}
