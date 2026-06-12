#include <Rcpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <limits>
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

struct SparseProbabilities {
  std::vector<int> row_ptr;
  std::vector<int> col;
  std::vector<double> val;
};

struct PackedEdge {
  std::uint64_t key;
  double value;
};

struct TsneTraceMetrics {
  double sum_q = NA_REAL;
  double repulsive_norm = NA_REAL;
  double attractive_norm = NA_REAL;
  double gradient_norm = NA_REAL;
  double update_norm = NA_REAL;
  double embedding_norm = NA_REAL;
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

template <typename Function>
void parallel_for(const int n, const int n_threads, Function fn) {
  if (n_threads <= 1 || n < 2) {
    fn(0, n, 0);
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

std::string tsne_repulsion_mode(const int n,
                                const double theta,
                                const std::string& requested_method) {
  const std::string requested = lowercase(requested_method);
  if (requested == "bh" || requested == "barnes_hut" || requested == "barnes-hut") {
    Rcpp::stop(
      "Barnes-Hut openTSNE has been removed from fastEmbedR. "
      "Use `negative_gradient_method = \"fft\"` for the standard CPU path "
      "or `\"exact\"` for small reference runs."
    );
  }
  if (requested == "exact" || requested == "pair" || requested == "pair_symmetric") {
    return "pair_symmetric";
  }
  if (requested == "keops_blocked" || requested == "blocked") {
    return "keops_blocked";
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
    if (value == "keops" || value == "keops_blocked" || value == "blocked" ||
        value == "lazy") {
      return "keops_blocked";
    }
    if (value == "pair" || value == "pair_symmetric" || value == "legacy" ||
        value == "exact") {
      return "pair_symmetric";
    }
  }

  (void)n;
  return theta > 0.0 ? "fft_grid" : "pair_symmetric";
}

int tsne_repulsion_block_size(const int n) {
  const int requested = env_positive_int("FASTEMBEDR_TSNE_BLOCK_SIZE", 1024);
  return std::max(32, std::min(n, requested));
}

int tsne_fft_grid_size(const int n) {
  const int fallback = n >= 50000 ? 128 : (n >= 10000 ? 96 : 64);
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

void compute_row_probabilities(const NumericMatrix& distances,
                               const int row,
                               const double perplexity,
                               std::vector<double>& row_p) {
  const int k = distances.ncol();
  row_p.assign(static_cast<std::size_t>(k), 0.0);

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
      const double d2 = d * d;
      const double p = std::exp(-beta * d2);
      row_p[static_cast<std::size_t>(j)] = p;
      sum_p += p;
    }

    double entropy = 0.0;
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      entropy += beta * (d * d * row_p[static_cast<std::size_t>(j)]);
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
  }

  for (double& value : row_p) value /= sum_p;
}

void compute_row_probabilities_flat(const NumericMatrix& distances,
                                    const int row,
                                    const double perplexity,
                                    double* row_p) {
  const int k = distances.ncol();
  std::fill(row_p, row_p + k, 0.0);

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
      const double d2 = d * d;
      const double p = std::exp(-beta * d2);
      row_p[j] = p;
      sum_p += p;
    }

    double entropy = 0.0;
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      entropy += beta * (d * d * row_p[j]);
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
  }

  const double inv_sum_p = 1.0 / sum_p;
  for (int j = 0; j < k; ++j) row_p[j] *= inv_sum_p;
}

SparseProbabilities build_tsne_probabilities(const IntegerMatrix& indices,
                                             const NumericMatrix& distances,
                                             const double perplexity,
                                             const int n_threads) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  const int offset = resolve_index_offset(indices);

  if (perplexity > static_cast<double>(k)) {
    Rcpp::warning("Perplexity is larger than the supplied KNN width; results may be unstable.");
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int nb = indices(i, j) - offset;
      if (nb < 0 || nb >= n) Rcpp::stop("KNN indices are out of range.");
      const double d = distances(i, j);
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative.");
      }
    }
  }

  std::vector<std::vector<PackedEdge>> local_edges(static_cast<std::size_t>(n_threads));
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<double> row_p;
    row_p.reserve(static_cast<std::size_t>(k));
    std::vector<PackedEdge>& edges = local_edges[static_cast<std::size_t>(thread_id)];
    edges.reserve(edges.size() + static_cast<std::size_t>(std::max(0, end - begin)) * k);

    for (int i = begin; i < end; ++i) {
      compute_row_probabilities(distances, i, perplexity, row_p);
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
  std::vector<PackedEdge> edges;
  edges.reserve(edge_count);
  for (auto& local : local_edges) {
    edges.insert(edges.end(), local.begin(), local.end());
    std::vector<PackedEdge>().swap(local);
  }

  std::sort(edges.begin(), edges.end(), [](const PackedEdge& a, const PackedEdge& b) {
    return a.key < b.key;
  });

  SparseProbabilities p;
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
      sum += edges[read].value;
      ++read;
    }
    edges[write++] = {key, sum};
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
  p.val.assign(p.col.size(), 0.0);
  std::vector<int> fill = p.row_ptr;

  for (const PackedEdge& edge : edges) {
    const int a = key_first(edge.key);
    const int b = key_second(edge.key);
    const double value = 0.5 * edge.value / total_directed_mass;

    int pos = fill[static_cast<std::size_t>(a)]++;
    p.col[static_cast<std::size_t>(pos)] = b;
    p.val[static_cast<std::size_t>(pos)] = value;

    pos = fill[static_cast<std::size_t>(b)]++;
    p.col[static_cast<std::size_t>(pos)] = a;
    p.val[static_cast<std::size_t>(pos)] = value;
  }

  return p;
}

double squared_distance(const std::vector<double>& y,
                        const int i,
                        const int j,
                        const int dims) {
  const std::size_t ib = static_cast<std::size_t>(i) * dims;
  const std::size_t jb = static_cast<std::size_t>(j) * dims;
  double out = 0.0;
  for (int d = 0; d < dims; ++d) {
    const double diff = y[ib + d] - y[jb + d];
    out += diff * diff;
  }
  return out;
}

double compute_sum_q(const std::vector<double>& y,
                     const int n,
                     const int dims,
                     const int n_threads) {
  std::vector<double> partial(static_cast<std::size_t>(n_threads), 0.0);
  auto worker = [&](const int thread_id) {
    double local = 0.0;
    for (int i = thread_id; i < n - 1; i += n_threads) {
      for (int j = i + 1; j < n; ++j) {
        local += 2.0 / (1.0 + squared_distance(y, i, j, dims));
      }
    }
    partial[static_cast<std::size_t>(thread_id)] = local;
  };

  if (n_threads <= 1) {
    worker(0);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads - 1));
    for (int t = 1; t < n_threads; ++t) {
      workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto& thread : workers) thread.join();
  }
  double sum_q = std::accumulate(partial.begin(), partial.end(), 0.0);
  return std::max(sum_q, DBL_MIN);
}

void add_sparse_attractive_gradient(const SparseProbabilities& p,
                                    const std::vector<double>& y,
                                    const int n,
                                    const int dims,
                                    const double exaggeration,
                                    const int n_threads,
                                    std::vector<double>& grad) {
  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      const int row_begin = p.row_ptr[static_cast<std::size_t>(i)];
      const int row_end = p.row_ptr[static_cast<std::size_t>(i + 1)];
      for (int pos = row_begin; pos < row_end; ++pos) {
        const int j = p.col[static_cast<std::size_t>(pos)];
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        double diff[3] = {0.0, 0.0, 0.0};
        double d2 = 0.0;
        for (int d = 0; d < dims; ++d) {
          diff[d] = y[ib + d] - y[jb + d];
          d2 += diff[d] * diff[d];
        }
        const double q = 1.0 / (1.0 + d2);
        const double coeff = exaggeration * p.val[static_cast<std::size_t>(pos)] * q;
        for (int d = 0; d < dims; ++d) {
          grad[ib + d] += coeff * diff[d];
        }
      }
    }
  });
}

void compute_gradient_pair_symmetric(const SparseProbabilities& p,
                                     const std::vector<double>& y,
                                     const int n,
                                     const int dims,
                                     const double exaggeration,
                                     const int n_threads,
                                     std::vector<double>& grad) {
  std::fill(grad.begin(), grad.end(), 0.0);
  std::vector<std::vector<double>> local_grad(
    static_cast<std::size_t>(n_threads),
    std::vector<double>(grad.size(), 0.0)
  );
  std::vector<double> partial_sum_q(static_cast<std::size_t>(n_threads), 0.0);

  auto repulsive_worker = [&](const int thread_id) {
    std::vector<double>& g = local_grad[static_cast<std::size_t>(thread_id)];
    double local_sum_q = 0.0;
    for (int i = thread_id; i < n - 1; i += n_threads) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      for (int j = i + 1; j < n; ++j) {
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        double d2 = 0.0;
        for (int d = 0; d < dims; ++d) {
          const double diff = y[ib + d] - y[jb + d];
          d2 += diff * diff;
        }
        const double q = 1.0 / (1.0 + d2);
        local_sum_q += 2.0 * q;
        const double coeff = -(q * q);
        for (int d = 0; d < dims; ++d) {
          const double step = coeff * (y[ib + d] - y[jb + d]);
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
    for (int t = 1; t < n_threads; ++t) {
      workers.emplace_back(repulsive_worker, t);
    }
    repulsive_worker(0);
    for (auto& thread : workers) thread.join();
  }

  const double inv_sum_q = 1.0 / std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0),
    DBL_MIN
  );
  parallel_for(static_cast<int>(grad.size()), n_threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      double value = 0.0;
      for (int t = 0; t < n_threads; ++t) {
        value += local_grad[static_cast<std::size_t>(t)][static_cast<std::size_t>(index)];
      }
      grad[static_cast<std::size_t>(index)] = value * inv_sum_q;
    }
  });

  add_sparse_attractive_gradient(p, y, n, dims, exaggeration, n_threads, grad);
}

void compute_gradient_keops_blocked(const SparseProbabilities& p,
                                    const std::vector<double>& y,
                                    const int n,
                                    const int dims,
                                    const double exaggeration,
                                    const int n_threads,
                                    const int block_size,
                                    std::vector<double>& grad) {
  std::fill(grad.begin(), grad.end(), 0.0);
  const double inv_sum_q = 1.0 / compute_sum_q(y, n, dims, n_threads);

  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      std::array<double, 3> accum{{0.0, 0.0, 0.0}};

      for (int block_begin = 0; block_begin < n; block_begin += block_size) {
        const int block_end = std::min(n, block_begin + block_size);
        for (int j = block_begin; j < block_end; ++j) {
          if (j == i) continue;
          const std::size_t jb = static_cast<std::size_t>(j) * dims;
          std::array<double, 3> diff{{0.0, 0.0, 0.0}};
          double d2 = 0.0;
          for (int d = 0; d < dims; ++d) {
            diff[static_cast<std::size_t>(d)] = y[ib + d] - y[jb + d];
            d2 += diff[static_cast<std::size_t>(d)] * diff[static_cast<std::size_t>(d)];
          }
          const double q = 1.0 / (1.0 + d2);
          const double coeff = -(q * q) * inv_sum_q;
          for (int d = 0; d < dims; ++d) {
            accum[static_cast<std::size_t>(d)] += coeff * diff[static_cast<std::size_t>(d)];
          }
        }
      }

      for (int d = 0; d < dims; ++d) {
        grad[ib + d] = accum[static_cast<std::size_t>(d)];
      }
    }
  });

  add_sparse_attractive_gradient(p, y, n, dims, exaggeration, n_threads, grad);
}

void fft_1d(std::complex<double>* a, const int n, const bool inverse) {
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }

  const double direction = inverse ? -1.0 : 1.0;
  for (int len = 2; len <= n; len <<= 1) {
    const double angle = direction * 6.283185307179586476925286766559 / static_cast<double>(len);
    const std::complex<double> root(std::cos(angle), std::sin(angle));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      const int half = len >> 1;
      for (int j = 0; j < half; ++j) {
        const std::complex<double> u = a[i + j];
        const std::complex<double> v = a[i + j + half] * w;
        a[i + j] = u + v;
        a[i + j + half] = u - v;
        w *= root;
      }
    }
  }

  if (inverse) {
    const double scale = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i) a[i] *= scale;
  }
}

void fft_2d(std::vector<std::complex<double>>& values,
            const int size,
            const bool inverse,
            const int n_threads) {
  parallel_for(size, n_threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      fft_1d(values.data() + static_cast<std::size_t>(row) * size, size, inverse);
    }
  });

  parallel_for(size, n_threads, [&](const int begin, const int end, const int) {
    std::vector<std::complex<double>> column(static_cast<std::size_t>(size));
    for (int col = begin; col < end; ++col) {
      for (int row = 0; row < size; ++row) {
        column[static_cast<std::size_t>(row)] =
          values[static_cast<std::size_t>(row) * size + col];
      }
      fft_1d(column.data(), size, inverse);
      for (int row = 0; row < size; ++row) {
        values[static_cast<std::size_t>(row) * size + col] =
          column[static_cast<std::size_t>(row)];
      }
    }
  });
}

double bilinear_grid_value(const std::vector<double>& grid,
                           const int grid_size,
                           const double gx,
                           const double gy) {
  const double cx = std::max(0.0, std::min(static_cast<double>(grid_size - 1), gx));
  const double cy = std::max(0.0, std::min(static_cast<double>(grid_size - 1), gy));
  const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cx))));
  const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cy))));
  const double tx = cx - static_cast<double>(x0);
  const double ty = cy - static_cast<double>(y0);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const double v00 = grid[static_cast<std::size_t>(y0) * grid_size + x0];
  const double v10 = grid[static_cast<std::size_t>(y0) * grid_size + x1];
  const double v01 = grid[static_cast<std::size_t>(y1) * grid_size + x0];
  const double v11 = grid[static_cast<std::size_t>(y1) * grid_size + x1];
  return (1.0 - tx) * (1.0 - ty) * v00 +
    tx * (1.0 - ty) * v10 +
    (1.0 - tx) * ty * v01 +
    tx * ty * v11;
}

void copy_grid_to_fft(const std::vector<double>& grid,
                      const int grid_size,
                      const int fft_size,
                      std::vector<std::complex<double>>& out) {
  std::fill(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
  for (int y_cell = 0; y_cell < grid_size; ++y_cell) {
    for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
      out[static_cast<std::size_t>(y_cell) * fft_size + x_cell] =
        grid[static_cast<std::size_t>(y_cell) * grid_size + x_cell];
    }
  }
}

void copy_fft_to_grid(const std::vector<std::complex<double>>& values,
                      const int grid_size,
                      const int fft_size,
                      std::vector<double>& out) {
  out.assign(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  for (int y_cell = 0; y_cell < grid_size; ++y_cell) {
    for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
      out[static_cast<std::size_t>(y_cell) * grid_size + x_cell] =
        values[static_cast<std::size_t>(y_cell) * fft_size + x_cell].real();
    }
  }
}

void compute_fft_grid_convolution(const std::vector<double>& mass,
                                  const std::vector<double>& mass_x,
                                  const std::vector<double>& mass_y,
                                  const int grid_size,
                                  const double spacing,
                                  const int n_threads,
                                  std::vector<double>& q_grid,
                                  std::vector<double>& q2_grid,
                                  std::vector<double>& xq2_grid,
                                  std::vector<double>& yq2_grid) {
  const int fft_size = grid_size << 1;
  const std::size_t fft_total = static_cast<std::size_t>(fft_size) * fft_size;
  std::vector<std::complex<double>> mass_fft(fft_total, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mass_x_fft(fft_total, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mass_y_fft(fft_total, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> kernel_q(fft_total, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> kernel_q2(fft_total, std::complex<double>(0.0, 0.0));

  copy_grid_to_fft(mass, grid_size, fft_size, mass_fft);
  copy_grid_to_fft(mass_x, grid_size, fft_size, mass_x_fft);
  copy_grid_to_fft(mass_y, grid_size, fft_size, mass_y_fft);

  for (int dy = -(grid_size - 1); dy <= grid_size - 1; ++dy) {
    const int yy = dy < 0 ? dy + fft_size : dy;
    const double y_offset = static_cast<double>(dy) * spacing;
    for (int dx = -(grid_size - 1); dx <= grid_size - 1; ++dx) {
      const int xx = dx < 0 ? dx + fft_size : dx;
      const double x_offset = static_cast<double>(dx) * spacing;
      const double d2 = x_offset * x_offset + y_offset * y_offset;
      const double q = 1.0 / (1.0 + d2);
      const double q2 = q * q;
      const std::size_t pos = static_cast<std::size_t>(yy) * fft_size + xx;
      kernel_q[pos] = q;
      kernel_q2[pos] = q2;
    }
  }

  fft_2d(mass_fft, fft_size, false, n_threads);
  fft_2d(mass_x_fft, fft_size, false, n_threads);
  fft_2d(mass_y_fft, fft_size, false, n_threads);
  fft_2d(kernel_q, fft_size, false, n_threads);
  fft_2d(kernel_q2, fft_size, false, n_threads);

  auto convolve = [&](const std::vector<std::complex<double>>& mass_values,
                      const std::vector<std::complex<double>>& kernel_values,
                      std::vector<double>& out) {
    std::vector<std::complex<double>> work(fft_total);
    for (std::size_t i = 0; i < fft_total; ++i) work[i] = mass_values[i] * kernel_values[i];
    fft_2d(work, fft_size, true, n_threads);
    copy_fft_to_grid(work, grid_size, fft_size, out);
  };

  convolve(mass_fft, kernel_q, q_grid);
  convolve(mass_fft, kernel_q2, q2_grid);
  convolve(mass_x_fft, kernel_q2, xq2_grid);
  convolve(mass_y_fft, kernel_q2, yq2_grid);
}

void compute_gradient_fft_grid(const SparseProbabilities& p,
                               const std::vector<double>& y,
                               const int n,
                               const int dims,
                               const double exaggeration,
                               const int n_threads,
                               std::vector<double>& grad) {
  if (dims != 2) {
    compute_gradient_pair_symmetric(p, y, n, dims, exaggeration, n_threads, grad);
    return;
  }

  std::fill(grad.begin(), grad.end(), 0.0);
  const int grid_size = tsne_fft_grid_size(n);
  double min_x = y[0], max_x = y[0], min_y = y[1], max_y = y[1];
  for (int i = 1; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    min_x = std::min(min_x, y[base]);
    max_x = std::max(max_x, y[base]);
    min_y = std::min(min_y, y[base + 1u]);
    max_y = std::max(max_y, y[base + 1u]);
  }
  const double cx = 0.5 * (min_x + max_x);
  const double cy = 0.5 * (min_y + max_y);
  double span = std::max(max_x - min_x, max_y - min_y);
  if (!std::isfinite(span) || span <= 0.0) span = 1.0;
  const double half = 0.55 * span + 1.0e-3;
  const double lower_x = cx - half;
  const double lower_y = cy - half;
  const double spacing = (2.0 * half) / static_cast<double>(grid_size - 1);
  const double inv_spacing = 1.0 / spacing;

  std::vector<double> mass(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> mass_x(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> mass_y(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> gx(static_cast<std::size_t>(n), 0.0);
  std::vector<double> gy(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const double x_coord = y[base];
    const double y_coord = y[base + 1u];
    const double raw_x = (x_coord - lower_x) * inv_spacing;
    const double raw_y = (y_coord - lower_y) * inv_spacing;
    const double clamped_x = std::max(0.0, std::min(static_cast<double>(grid_size - 1), raw_x));
    const double clamped_y = std::max(0.0, std::min(static_cast<double>(grid_size - 1), raw_y));
    const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_x))));
    const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_y))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double tx = clamped_x - static_cast<double>(x0);
    const double ty = clamped_y - static_cast<double>(y0);
    gx[static_cast<std::size_t>(i)] = clamped_x;
    gy[static_cast<std::size_t>(i)] = clamped_y;
    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;
    const std::size_t p00 = static_cast<std::size_t>(y0) * grid_size + x0;
    const std::size_t p10 = static_cast<std::size_t>(y0) * grid_size + x1;
    const std::size_t p01 = static_cast<std::size_t>(y1) * grid_size + x0;
    const std::size_t p11 = static_cast<std::size_t>(y1) * grid_size + x1;
    mass[p00] += w00;
    mass[p10] += w10;
    mass[p01] += w01;
    mass[p11] += w11;
    mass_x[p00] += w00 * x_coord;
    mass_x[p10] += w10 * x_coord;
    mass_x[p01] += w01 * x_coord;
    mass_x[p11] += w11 * x_coord;
    mass_y[p00] += w00 * y_coord;
    mass_y[p10] += w10 * y_coord;
    mass_y[p01] += w01 * y_coord;
    mass_y[p11] += w11 * y_coord;
  }

  std::vector<double> q_grid;
  std::vector<double> q2_grid;
  std::vector<double> xq2_grid;
  std::vector<double> yq2_grid;
  compute_fft_grid_convolution(
    mass,
    mass_x,
    mass_y,
    grid_size,
    spacing,
    n_threads,
    q_grid,
    q2_grid,
    xq2_grid,
    yq2_grid
  );

  std::vector<double> partial_sum_q(static_cast<std::size_t>(n_threads), 0.0);
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local_sum_q = 0.0;
    for (int i = begin; i < end; ++i) {
      const double q_value = bilinear_grid_value(
        q_grid,
        grid_size,
        gx[static_cast<std::size_t>(i)],
        gy[static_cast<std::size_t>(i)]
      );
      local_sum_q += q_value;
    }
    partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  });
  const double sum_q = std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0) -
      static_cast<double>(n),
    DBL_MIN
  );
  const double inv_sum_q = 1.0 / sum_q;

  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      const double px = gx[static_cast<std::size_t>(i)];
      const double py = gy[static_cast<std::size_t>(i)];
      const double q2_value = bilinear_grid_value(q2_grid, grid_size, px, py);
      const double xq2_value = bilinear_grid_value(xq2_grid, grid_size, px, py);
      const double yq2_value = bilinear_grid_value(yq2_grid, grid_size, px, py);
      grad[base] = -(y[base] * q2_value - xq2_value) * inv_sum_q;
      grad[base + 1u] = -(y[base + 1u] * q2_value - yq2_value) * inv_sum_q;
    }
  });

  add_sparse_attractive_gradient(p, y, n, dims, exaggeration, n_threads, grad);
}

TsneTraceMetrics compute_gradient_fft_grid_trace(const SparseProbabilities& p,
                                                 const std::vector<double>& y,
                                                 const int n,
                                                 const int dims,
                                                 const double exaggeration,
                                                 const int n_threads,
                                                 std::vector<double>& grad,
                                                 std::vector<double>& attractive) {
  if (dims != 2) {
    Rcpp::stop("CPU openTSNE parity trace currently supports two dimensions.");
  }

  std::fill(grad.begin(), grad.end(), 0.0);
  std::fill(attractive.begin(), attractive.end(), 0.0);
  const int grid_size = tsne_fft_grid_size(n);
  double min_x = y[0], max_x = y[0], min_y = y[1], max_y = y[1];
  for (int i = 1; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    min_x = std::min(min_x, y[base]);
    max_x = std::max(max_x, y[base]);
    min_y = std::min(min_y, y[base + 1u]);
    max_y = std::max(max_y, y[base + 1u]);
  }
  const double cx = 0.5 * (min_x + max_x);
  const double cy = 0.5 * (min_y + max_y);
  double span = std::max(max_x - min_x, max_y - min_y);
  if (!std::isfinite(span) || span <= 0.0) span = 1.0;
  const double half = 0.55 * span + 1.0e-3;
  const double lower_x = cx - half;
  const double lower_y = cy - half;
  const double spacing = (2.0 * half) / static_cast<double>(grid_size - 1);
  const double inv_spacing = 1.0 / spacing;

  std::vector<double> mass(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> mass_x(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> mass_y(static_cast<std::size_t>(grid_size) * grid_size, 0.0);
  std::vector<double> gx(static_cast<std::size_t>(n), 0.0);
  std::vector<double> gy(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const double x_coord = y[base];
    const double y_coord = y[base + 1u];
    const double raw_x = (x_coord - lower_x) * inv_spacing;
    const double raw_y = (y_coord - lower_y) * inv_spacing;
    const double clamped_x = std::max(0.0, std::min(static_cast<double>(grid_size - 1), raw_x));
    const double clamped_y = std::max(0.0, std::min(static_cast<double>(grid_size - 1), raw_y));
    const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_x))));
    const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_y))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const double tx = clamped_x - static_cast<double>(x0);
    const double ty = clamped_y - static_cast<double>(y0);
    gx[static_cast<std::size_t>(i)] = clamped_x;
    gy[static_cast<std::size_t>(i)] = clamped_y;
    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;
    const std::size_t p00 = static_cast<std::size_t>(y0) * grid_size + x0;
    const std::size_t p10 = static_cast<std::size_t>(y0) * grid_size + x1;
    const std::size_t p01 = static_cast<std::size_t>(y1) * grid_size + x0;
    const std::size_t p11 = static_cast<std::size_t>(y1) * grid_size + x1;
    mass[p00] += w00;
    mass[p10] += w10;
    mass[p01] += w01;
    mass[p11] += w11;
    mass_x[p00] += w00 * x_coord;
    mass_x[p10] += w10 * x_coord;
    mass_x[p01] += w01 * x_coord;
    mass_x[p11] += w11 * x_coord;
    mass_y[p00] += w00 * y_coord;
    mass_y[p10] += w10 * y_coord;
    mass_y[p01] += w01 * y_coord;
    mass_y[p11] += w11 * y_coord;
  }

  std::vector<double> q_grid;
  std::vector<double> q2_grid;
  std::vector<double> xq2_grid;
  std::vector<double> yq2_grid;
  compute_fft_grid_convolution(
    mass, mass_x, mass_y, grid_size, spacing, n_threads,
    q_grid, q2_grid, xq2_grid, yq2_grid
  );

  std::vector<double> partial_sum_q(static_cast<std::size_t>(n_threads), 0.0);
  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local_sum_q = 0.0;
    for (int i = begin; i < end; ++i) {
      local_sum_q += bilinear_grid_value(
        q_grid, grid_size, gx[static_cast<std::size_t>(i)], gy[static_cast<std::size_t>(i)]
      );
    }
    partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  });
  const double sum_q = std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0) -
      static_cast<double>(n),
    DBL_MIN
  );
  const double inv_sum_q = 1.0 / sum_q;

  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      const double px = gx[static_cast<std::size_t>(i)];
      const double py = gy[static_cast<std::size_t>(i)];
      const double q2_value = bilinear_grid_value(q2_grid, grid_size, px, py);
      const double xq2_value = bilinear_grid_value(xq2_grid, grid_size, px, py);
      const double yq2_value = bilinear_grid_value(yq2_grid, grid_size, px, py);
      grad[base] = -(y[base] * q2_value - xq2_value) * inv_sum_q;
      grad[base + 1u] = -(y[base + 1u] * q2_value - yq2_value) * inv_sum_q;
    }
  });

  add_sparse_attractive_gradient(p, y, n, dims, exaggeration, n_threads, attractive);
  TsneTraceMetrics metrics;
  metrics.sum_q = sum_q;
  double rep2 = 0.0, att2 = 0.0, grad2 = 0.0;
  for (std::size_t i = 0; i < grad.size(); ++i) {
    const double rep = grad[i];
    const double att = attractive[i];
    const double total = rep + att;
    rep2 += rep * rep;
    att2 += att * att;
    grad2 += total * total;
    grad[i] = total;
  }
  metrics.repulsive_norm = std::sqrt(rep2);
  metrics.attractive_norm = std::sqrt(att2);
  metrics.gradient_norm = std::sqrt(grad2);
  return metrics;
}

void compute_gradient(const SparseProbabilities& p,
                      const std::vector<double>& y,
                      const int n,
                      const int dims,
                      const double exaggeration,
                      const int n_threads,
                      const std::string& repulsion_mode,
                      const int block_size,
                      const double theta,
                      std::vector<double>& grad) {
  if (repulsion_mode == "keops_blocked") {
    compute_gradient_keops_blocked(p, y, n, dims, exaggeration, n_threads, block_size, grad);
  } else if (repulsion_mode == "fft_grid") {
    compute_gradient_fft_grid(p, y, n, dims, exaggeration, n_threads, grad);
  } else {
    compute_gradient_pair_symmetric(p, y, n, dims, exaggeration, n_threads, grad);
  }
}

void zero_mean(std::vector<double>& y, const int n, const int dims) {
  std::array<double, 3> mean{{0.0, 0.0, 0.0}};
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] += y[base + d];
  }
  for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) y[base + d] -= mean[static_cast<std::size_t>(d)];
  }
}

double sign_tsne(double x) {
  return x == 0.0 ? 0.0 : (x < 0.0 ? -1.0 : 1.0);
}

void apply_open_tsne_update(std::vector<double>& y,
                            std::vector<double>& update,
                            std::vector<double>& gains,
                            const std::vector<double>& grad,
                            const int n,
                            const int dims,
                            const double learning_rate,
                            const double momentum,
                            const double min_gain,
	                            const double max_step_norm,
	                            const int n_threads) {
	  const bool clip_steps = std::isfinite(max_step_norm) && max_step_norm > 0.0;
	  const double max_step_norm_sq = max_step_norm * max_step_norm;
	  parallel_for(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * dims;
      double step_norm_sq = 0.0;
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        if (sign_tsne(update[index]) != sign_tsne(grad[index])) {
          gains[index] += 0.2;
        } else {
          gains[index] = gains[index] * 0.8 + min_gain;
        }
        if (gains[index] < min_gain) gains[index] = min_gain;
        update[index] = momentum * update[index] - learning_rate * gains[index] * grad[index];
        step_norm_sq += update[index] * update[index];
      }
      double scale = 1.0;
	      if (clip_steps && step_norm_sq > max_step_norm_sq) {
        scale = max_step_norm / std::sqrt(std::max(step_norm_sq, DBL_MIN));
      }
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        update[index] *= scale;
        y[index] += update[index];
      }
    }
  });
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
                                     std::vector<double>& grad) {
  const int k = indices.ncol();
  const int n_reference = reference_layout.nrow();
  const int dims = reference_layout.ncol();
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  std::fill(grad.begin(), grad.begin() + static_cast<std::size_t>(batch_n) * dims, 0.0);

  parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int) {
    std::vector<int> sampled(static_cast<std::size_t>(std::max(1, n_negatives)), 0);
    std::vector<double> q_values;
    if (exact_repulsion) {
      q_values.resize(static_cast<std::size_t>(n_reference), 0.0);
    } else {
      q_values.resize(static_cast<std::size_t>(std::max(1, n_negatives)), 0.0);
    }

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
                                             std::vector<double>& grad) {
  const int k = indices.ncol();
  const int n_reference = static_cast<int>(reference_layout.size() / 2u);
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  std::fill(grad.begin(), grad.begin() + static_cast<std::size_t>(batch_n) * 2u, 0.0);

  parallel_for(batch_n, n_threads, [&](const int begin, const int end, const int) {
    std::vector<int> sampled(static_cast<std::size_t>(std::max(1, n_negatives)), 0);
    std::vector<double> q_values(static_cast<std::size_t>(
      exact_repulsion ? n_reference : std::max(1, n_negatives)
    ), 0.0);

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

double evaluate_kl(const SparseProbabilities& p,
                   const std::vector<double>& y,
                   const int n,
                   const int dims,
                   const int n_threads,
                   std::vector<double>* row_costs = nullptr) {
  const double sum_q = compute_sum_q(y, n, dims, n_threads);
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
        const double p_ij = p.val[static_cast<std::size_t>(pos)];
        const double q_ij = (1.0 / (1.0 + squared_distance(y, i, j, dims))) / sum_q;
        row_total += p_ij * std::log((p_ij + 1e-9) / (q_ij + 1e-9));
      }
      if (row_costs != nullptr) (*row_costs)[static_cast<std::size_t>(i)] = row_total;
      local += row_total;
    }
    partial[static_cast<std::size_t>(thread_id)] = local;
  });

  return std::accumulate(partial.begin(), partial.end(), 0.0);
}

} // namespace

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
  const int max_perplexity_k = std::max(1, k / 3);
  double resolved_perplexity = perplexity_missing || !std::isfinite(perplexity) || perplexity <= 0.0 ?
    static_cast<double>(std::min(30, std::min(max_perplexity_n, max_perplexity_k))) :
    perplexity;
  resolved_perplexity = std::max(
    1.0,
    std::min(resolved_perplexity, static_cast<double>(std::min(max_perplexity_n, max_perplexity_k)))
  );

  const double early_exaggeration = 12.0;
  const int needed_k = std::max(1, std::min(n - 1, static_cast<int>(std::ceil(3.0 * resolved_perplexity))));
  const int early_max = n >= 10000 ? 750 : 500;
  const int normal_max = n >= 10000 ? 750 : 500;

  const bool cpu_backend = backend == "cpu" || backend == "auto";
  const bool exact_or_small = negative_gradient_method == "exact" || n <= 5000;
  const bool kld_auto_stop = cpu_backend && exact_or_small;

  return List::create(
    Rcpp::Named("perplexity") = resolved_perplexity,
    Rcpp::Named("n_neighbors") = needed_k,
    Rcpp::Named("early_exaggeration") = early_exaggeration,
    Rcpp::Named("exaggeration") = 1.0,
    Rcpp::Named("learning_rate") = static_cast<double>(n) / early_exaggeration,
    Rcpp::Named("early_exaggeration_iter") = kld_auto_stop ? early_max : 250,
    Rcpp::Named("n_iter") = kld_auto_stop ? normal_max : 500,
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

// Native openTSNE-style optimizer from precomputed KNN probabilities. This
// follows openTSNE's two-phase optimization contract while keeping the
// implementation in fastEmbedR C++ and using the same sparse KNN affinity
// builder as the Rtsne-compatible path above.
// [[Rcpp::export]]
List knn_tsne_opentsne_cpp(IntegerMatrix indices,
                           NumericMatrix distances,
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
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
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
  if (verbose) {
    Rcpp::Rcout << "fastEmbedR openTSNE-style t-SNE from KNN: n=" << n
                << ", k=" << k
                << ", perplexity=" << perplexity
                << ", threads=" << threads << "\n";
  }

  SparseProbabilities p = build_tsne_probabilities(indices, distances, perplexity, threads);
  const std::string repulsion_mode = tsne_repulsion_mode(n, theta, negative_gradient_method);
  if (repulsion_mode == "keops_blocked") {
    Rcpp::stop("`negative_gradient_method = \"keops_blocked\"` is not part of the native openTSNE-style path.");
  }
  const int repulsion_block_size = tsne_repulsion_block_size(n);
  std::string optimizer_name = repulsion_mode == "fft_grid" ?
    "opentsne_fitsne_fft_grid_sparse_knn" :
    "opentsne_exact_sparse_knn";
  if (verbose) {
    Rcpp::Rcout << "openTSNE-style repulsion: " << repulsion_mode
                << ", block_size=" << repulsion_block_size << "\n";
  }

  std::vector<double> y(static_cast<std::size_t>(n) * n_components);
  if (init) {
    if (y_init.nrow() != n || y_init.ncol() != n_components) {
      Rcpp::stop("`Y_init` has the wrong shape.");
    }
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < n_components; ++d) {
        y[static_cast<std::size_t>(i) * n_components + d] = y_init(i, d);
      }
    }
  } else {
    const unsigned int resolved_seed = seed == NA_INTEGER ?
      5489u :
      static_cast<unsigned int>(seed);
    std::mt19937 rng(resolved_seed);
    std::normal_distribution<double> normal(0.0, 1.0e-4);
    for (double& value : y) value = normal(rng);
  }
  zero_mean(y, n, n_components);

  std::vector<double> grad(y.size(), 0.0);
  std::vector<double> update(y.size(), 0.0);
  std::vector<double> gains(y.size(), 1.0);
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
    const double phase_lr = learning_rate_auto ?
      static_cast<double>(n) / std::max(phase_exaggeration, DBL_MIN) :
      learning_rate;
    if (verbose) {
      Rcpp::Rcout << "openTSNE-style phase " << phase_name
                  << ": iterations=" << phase_iter
                  << ", exaggeration=" << phase_exaggeration
                  << ", learning_rate=" << phase_lr
                  << ", momentum=" << phase_momentum << "\n";
    }
    int phase_completed = 0;
    for (int iter = 0; iter < phase_iter; ++iter) {
      if (((completed_iter + iter) & 7) == 0) Rcpp::checkUserInterrupt();
      compute_gradient(
        p,
        y,
        n,
        n_components,
        phase_exaggeration,
        threads,
        repulsion_mode,
        repulsion_block_size,
        theta,
        grad
      );
      apply_open_tsne_update(
        y,
        update,
        gains,
        grad,
        n,
        n_components,
        phase_lr,
        phase_momentum,
        min_gain,
        max_step_norm,
        threads
      );
      zero_mean(y, n, n_components);

      const int global_iter = completed_iter + iter + 1;
      const bool need_auto_error = auto_kld_stop && ((iter + 1) % auto_pollrate == 0);
      const bool need_record_error = should_record_costs &&
        ((global_iter % 50 == 0) || global_iter == requested_total_iter);
      if (need_auto_error || need_record_error) {
        const double kl = evaluate_kl(p, y, n, n_components, threads);
        if (need_record_error && cost_index < iter_costs.size()) {
          iter_costs[cost_index] = kl;
          itercost_iterations[cost_index] = global_iter;
          ++cost_index;
        }
        if (verbose) {
          Rcpp::Rcout << "Iteration " << global_iter
                      << ": error is " << kl << "\n";
        }
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
    early_exaggeration_iter,
    early_exaggeration,
    initial_momentum,
    "early_exaggeration",
    completed_iter
  );
  const int actual_normal_iter = run_phase(
    n_iter,
    exaggeration,
    final_momentum,
    "normal",
    completed_iter
  );

  NumericVector row_costs;
  if (should_record_costs) {
    std::vector<double> costs(static_cast<std::size_t>(n), 0.0);
    evaluate_kl(p, y, n, n_components, threads, &costs);
    row_costs = NumericVector(costs.begin(), costs.end());
  } else {
    row_costs = NumericVector(0);
  }

  NumericMatrix layout(n, n_components);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_components; ++d) {
      layout(i, d) = y[static_cast<std::size_t>(i) * n_components + d];
    }
  }

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("costs") = row_costs,
	    Rcpp::Named("itercosts") = iter_costs,
	    Rcpp::Named("itercost_iterations") = itercost_iterations,
	    Rcpp::Named("optimizer") = optimizer_name,
    Rcpp::Named("repulsion") = repulsion_mode,
    Rcpp::Named("repulsion_block_size") = repulsion_block_size,
    Rcpp::Named("theta_requested") = theta,
    Rcpp::Named("n_threads") = threads,
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
List opentsne_cpu_trace_cpp(IntegerMatrix indices,
                            NumericMatrix distances,
                            NumericMatrix y_init,
                            double perplexity,
                            int n_iter,
                            double early_exaggeration,
                            double learning_rate,
                            bool learning_rate_auto,
                            double momentum,
                            double min_gain,
                            double max_step_norm,
                            int n_threads) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  const int n = indices.nrow();
  if (y_init.nrow() != n || y_init.ncol() != 2) {
    Rcpp::stop("`y_init` must have one row per point and two columns.");
  }
  if (n_iter < 1) Rcpp::stop("`n_iter` must be positive.");
  const int threads = resolve_threads(n_threads, n);
  SparseProbabilities p = build_tsne_probabilities(indices, distances, perplexity, threads);
  std::vector<double> y(static_cast<std::size_t>(n) * 2u, 0.0);
  for (int i = 0; i < n; ++i) {
    y[static_cast<std::size_t>(i) * 2u] = y_init(i, 0);
    y[static_cast<std::size_t>(i) * 2u + 1u] = y_init(i, 1);
  }
  zero_mean(y, n, 2);

  std::vector<double> grad(y.size(), 0.0);
  std::vector<double> attractive(y.size(), 0.0);
  std::vector<double> update(y.size(), 0.0);
  std::vector<double> gains(y.size(), 1.0);
  NumericVector iter(n_iter), sum_q(n_iter), repulsive_norm(n_iter),
    attractive_norm(n_iter), gradient_norm(n_iter), update_norm(n_iter),
    embedding_norm(n_iter);
  const double lr = learning_rate_auto ?
    static_cast<double>(n) / std::max(early_exaggeration, DBL_MIN) :
    learning_rate;

  for (int it = 0; it < n_iter; ++it) {
    TsneTraceMetrics metrics = compute_gradient_fft_grid_trace(
      p, y, n, 2, early_exaggeration, threads, grad, attractive
    );
    apply_open_tsne_update(
      y, update, gains, grad, n, 2, lr, momentum, min_gain, max_step_norm, threads
    );
    zero_mean(y, n, 2);
    double update2 = 0.0;
    double layout2 = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      update2 += update[i] * update[i];
      layout2 += y[i] * y[i];
    }
    iter[it] = it + 1;
    sum_q[it] = metrics.sum_q;
    repulsive_norm[it] = metrics.repulsive_norm;
    attractive_norm[it] = metrics.attractive_norm;
    gradient_norm[it] = metrics.gradient_norm;
    update_norm[it] = std::sqrt(update2);
    embedding_norm[it] = std::sqrt(layout2);
  }

  NumericMatrix layout(n, 2);
  for (int i = 0; i < n; ++i) {
    layout(i, 0) = y[static_cast<std::size_t>(i) * 2u];
    layout(i, 1) = y[static_cast<std::size_t>(i) * 2u + 1u];
  }

  Rcpp::DataFrame trace = Rcpp::DataFrame::create(
    Rcpp::Named("iter") = iter,
    Rcpp::Named("sum_q") = sum_q,
    Rcpp::Named("repulsive_norm") = repulsive_norm,
    Rcpp::Named("attractive_norm") = attractive_norm,
    Rcpp::Named("gradient_norm") = gradient_norm,
    Rcpp::Named("update_norm") = update_norm,
    Rcpp::Named("embedding_norm") = embedding_norm
  );
  return List::create(
    Rcpp::Named("trace") = trace,
    Rcpp::Named("Y") = layout,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("negative_gradient_method") = "fft"
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
  if (3.0 * perplexity > static_cast<double>(k)) {
    Rcpp::warning("Transform perplexity is close to or larger than the supplied KNN width; consider a wider query KNN.");
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
