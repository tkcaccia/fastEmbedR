#include <Rcpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
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
    return "barnes_hut";
  }
  if (requested == "exact" || requested == "pair" || requested == "pair_symmetric") {
    return "pair_symmetric";
  }
  if (requested == "keops_blocked" || requested == "blocked") {
    return "keops_blocked";
  }
  if (requested == "fft" || requested == "interpolation" || requested == "fitsne") {
    Rcpp::stop("openTSNE-style FFT interpolation is not yet available in native fastEmbedR C++.");
  }

  const char* raw = std::getenv("FASTEMBEDR_TSNE_REPULSION");
  if (raw != nullptr && raw[0] != '\0') {
    const std::string value = lowercase(std::string(raw));
    if (value == "barnes_hut" || value == "barnes-hut" || value == "bh" ||
        value == "rtsne") {
      return "barnes_hut";
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
  return theta > 0.0 ? "barnes_hut" : "pair_symmetric";
}

int tsne_repulsion_block_size(const int n) {
  const int requested = env_positive_int("FASTEMBEDR_TSNE_BLOCK_SIZE", 1024);
  return std::max(32, std::min(n, requested));
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

struct BarnesHutNode2D {
  double center_x = 0.0;
  double center_y = 0.0;
  double half_x = 0.0;
  double half_y = 0.0;
  double mass_x = 0.0;
  double mass_y = 0.0;
  int count = 0;
  int point = -1;
  int children[4] = {-1, -1, -1, -1};
  bool leaf = true;
};

class BarnesHutTree2D {
 public:
  BarnesHutTree2D(const std::vector<double>& y_in, const int n_in) :
    y(y_in), n(n_in) {
    double min_x = DBL_MAX;
    double max_x = -DBL_MAX;
    double min_y = DBL_MAX;
    double max_y = -DBL_MAX;
    for (int i = 0; i < n; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      min_x = std::min(min_x, y[base]);
      max_x = std::max(max_x, y[base]);
      min_y = std::min(min_y, y[base + 1u]);
      max_y = std::max(max_y, y[base + 1u]);
    }

    BarnesHutNode2D root;
    root.center_x = 0.5 * (min_x + max_x);
    root.center_y = 0.5 * (min_y + max_y);
    const double side = std::max(max_x - min_x, max_y - min_y) + 1e-5;
    root.half_x = std::max(0.5 * side, 1e-5);
    root.half_y = std::max(0.5 * side, 1e-5);
    nodes.reserve(static_cast<std::size_t>(std::max(1, 2 * n)));
    nodes.push_back(root);
    for (int i = 0; i < n; ++i) insert(0, i, 0);
  }

  double non_edge_force(const int point_index,
                        const double theta,
                        double& force_x,
                        double& force_y) const {
    force_x = 0.0;
    force_y = 0.0;
    return non_edge_force_node(0, point_index, theta, force_x, force_y);
  }

 private:
  const std::vector<double>& y;
  int n;
  std::vector<BarnesHutNode2D> nodes;

  int child_slot(const BarnesHutNode2D& node, const int point_index) const {
    const std::size_t base = static_cast<std::size_t>(point_index) * 2u;
    const int right = y[base] > node.center_x ? 1 : 0;
    const int upper = y[base + 1u] > node.center_y ? 2 : 0;
    return right + upper;
  }

  int make_child(const int node_index, const int slot) {
    BarnesHutNode2D& parent = nodes[static_cast<std::size_t>(node_index)];
    BarnesHutNode2D child;
    child.half_x = std::max(0.5 * parent.half_x, 1e-12);
    child.half_y = std::max(0.5 * parent.half_y, 1e-12);
    child.center_x = parent.center_x + ((slot & 1) ? child.half_x : -child.half_x);
    child.center_y = parent.center_y + ((slot & 2) ? child.half_y : -child.half_y);
    const int child_index = static_cast<int>(nodes.size());
    nodes.push_back(child);
    nodes[static_cast<std::size_t>(node_index)].children[slot] = child_index;
    return child_index;
  }

  void subdivide(const int node_index) {
    BarnesHutNode2D& node = nodes[static_cast<std::size_t>(node_index)];
    if (!node.leaf) return;
    node.leaf = false;
    const int existing = node.point;
    node.point = -1;
    if (existing >= 0) {
      const int slot = child_slot(node, existing);
      const int child = node.children[slot] >= 0 ? node.children[slot] : make_child(node_index, slot);
      insert(child, existing, 0, false);
    }
  }

  void update_mass(BarnesHutNode2D& node, const int point_index) {
    const std::size_t base = static_cast<std::size_t>(point_index) * 2u;
    const double old_count = static_cast<double>(node.count);
    const double new_count = old_count + 1.0;
    node.mass_x = (node.mass_x * old_count + y[base]) / new_count;
    node.mass_y = (node.mass_y * old_count + y[base + 1u]) / new_count;
    ++node.count;
  }

  void insert(const int node_index,
              const int point_index,
              const int depth,
              const bool update_current = true) {
    BarnesHutNode2D& node = nodes[static_cast<std::size_t>(node_index)];
    if (update_current) update_mass(node, point_index);

    if (node.leaf) {
      if (node.point < 0) {
        node.point = point_index;
        return;
      }

      const std::size_t old_base = static_cast<std::size_t>(node.point) * 2u;
      const std::size_t new_base = static_cast<std::size_t>(point_index) * 2u;
      const double dx = y[old_base] - y[new_base];
      const double dy = y[old_base + 1u] - y[new_base + 1u];
      if (depth >= 64 || dx * dx + dy * dy <= 1e-24) {
        return;
      }
      subdivide(node_index);
    }

    BarnesHutNode2D& parent = nodes[static_cast<std::size_t>(node_index)];
    const int slot = child_slot(parent, point_index);
    const int child = parent.children[slot] >= 0 ? parent.children[slot] : make_child(node_index, slot);
    insert(child, point_index, depth + 1);
  }

  double non_edge_force_node(const int node_index,
                             const int point_index,
                             const double theta,
                             double& force_x,
                             double& force_y) const {
    const BarnesHutNode2D& node = nodes[static_cast<std::size_t>(node_index)];
    if (node.count == 0) return 0.0;
    if (node.leaf && node.count == 1 && node.point == point_index) return 0.0;

    const std::size_t base = static_cast<std::size_t>(point_index) * 2u;
    const double dx = y[base] - node.mass_x;
    const double dy = y[base + 1u] - node.mass_y;
    const double dist_sq = dx * dx + dy * dy;
    const double width = 2.0 * std::max(node.half_x, node.half_y);

    if (node.leaf || width / std::sqrt(std::max(dist_sq, 1e-24)) < theta) {
      if (dist_sq <= 1e-24) return 0.0;
      const double q = 1.0 / (1.0 + dist_sq);
      const double mass = static_cast<double>(node.count);
      const double coeff = mass * q * q;
      force_x += coeff * dx;
      force_y += coeff * dy;
      return mass * q;
    }

    double sum_q = 0.0;
    for (int slot = 0; slot < 4; ++slot) {
      const int child = node.children[slot];
      if (child >= 0) sum_q += non_edge_force_node(child, point_index, theta, force_x, force_y);
    }
    return sum_q;
  }
};

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
        const double q = 1.0 / (1.0 + squared_distance(y, i, j, dims));
        const double coeff = exaggeration * p.val[static_cast<std::size_t>(pos)] * q;
        for (int d = 0; d < dims; ++d) {
          grad[ib + d] += coeff * (y[ib + d] - y[jb + d]);
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

void compute_gradient_barnes_hut(const SparseProbabilities& p,
                                 const std::vector<double>& y,
                                 const int n,
                                 const int dims,
                                 const double exaggeration,
                                 const double theta,
                                 const int n_threads,
                                 std::vector<double>& grad) {
  if (dims != 2) {
    compute_gradient_pair_symmetric(p, y, n, dims, exaggeration, n_threads, grad);
    return;
  }

  std::fill(grad.begin(), grad.end(), 0.0);
  BarnesHutTree2D tree(y, n);
  std::vector<double> repulsive(grad.size(), 0.0);
  std::vector<double> partial_sum_q(static_cast<std::size_t>(n_threads), 0.0);

  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local_sum_q = 0.0;
    for (int i = begin; i < end; ++i) {
      double fx = 0.0;
      double fy = 0.0;
      local_sum_q += tree.non_edge_force(i, theta, fx, fy);
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      repulsive[base] = fx;
      repulsive[base + 1u] = fy;
    }
    partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  });

  const double inv_sum_q = 1.0 / std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0),
    DBL_MIN
  );
  parallel_for(static_cast<int>(grad.size()), n_threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      grad[static_cast<std::size_t>(index)] = -repulsive[static_cast<std::size_t>(index)] * inv_sum_q;
    }
  });

  add_sparse_attractive_gradient(p, y, n, dims, exaggeration, n_threads, grad);
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
  } else if (repulsion_mode == "barnes_hut") {
    compute_gradient_barnes_hut(p, y, n, dims, exaggeration, theta, n_threads, grad);
  } else {
    compute_gradient_pair_symmetric(p, y, n, dims, exaggeration, n_threads, grad);
  }
}

void zero_mean(std::vector<double>& y, const int n, const int dims) {
  std::vector<double> mean(static_cast<std::size_t>(dims), 0.0);
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

int sample_nonself(std::mt19937& rng, const int n, const int self) {
  std::uniform_int_distribution<int> uniform(0, n - 2);
  const int draw = uniform(rng);
  return draw >= self ? draw + 1 : draw;
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

void compute_infotsne_gradient(const SparseProbabilities& p,
                               const std::vector<double>& y,
                               const int n,
                               const int dims,
                               const double exaggeration,
                               const double repulsion_strength,
                               const int n_negatives,
                               const int n_threads,
                               const int seed,
                               std::vector<double>& grad) {
  std::fill(grad.begin(), grad.end(), 0.0);
  std::vector<std::vector<double>> local_grad(
    static_cast<std::size_t>(n_threads),
    std::vector<double>(grad.size(), 0.0)
  );

  parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<double>& g = local_grad[static_cast<std::size_t>(thread_id)];
    std::vector<int> negatives(static_cast<std::size_t>(n_negatives), 0);
    std::vector<double> neg_q(static_cast<std::size_t>(n_negatives), 0.0);
    const std::uint32_t stream_seed =
      static_cast<std::uint32_t>(seed) ^
      (static_cast<std::uint32_t>(thread_id + 1) * 0x9e3779b9u);
    std::mt19937 rng(stream_seed);

    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;

      const int row_begin = p.row_ptr[static_cast<std::size_t>(i)];
      const int row_end = p.row_ptr[static_cast<std::size_t>(i + 1)];
      for (int pos = row_begin; pos < row_end; ++pos) {
        const int j = p.col[static_cast<std::size_t>(pos)];
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        double d2 = 0.0;
        for (int d = 0; d < dims; ++d) {
          const double diff = y[ib + d] - y[jb + d];
          d2 += diff * diff;
        }
        const double q = 1.0 / (1.0 + d2);
        const double coeff = 2.0 * exaggeration * p.val[static_cast<std::size_t>(pos)] * q;
        for (int d = 0; d < dims; ++d) {
          g[ib + d] += coeff * (y[ib + d] - y[jb + d]);
        }
      }

      double sum_q = DBL_MIN;
      for (int m = 0; m < n_negatives; ++m) {
        const int j = sample_nonself(rng, n, i);
        negatives[static_cast<std::size_t>(m)] = j;
        const double q = 1.0 / (1.0 + squared_distance(y, i, j, dims));
        neg_q[static_cast<std::size_t>(m)] = q;
        sum_q += q;
      }

      const double normalizer = repulsion_strength / (static_cast<double>(n) * sum_q);
      for (int m = 0; m < n_negatives; ++m) {
        const int j = negatives[static_cast<std::size_t>(m)];
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        const double coeff = -2.0 * normalizer * neg_q[static_cast<std::size_t>(m)] *
          neg_q[static_cast<std::size_t>(m)];
        for (int d = 0; d < dims; ++d) {
          const double diff = y[ib + d] - y[jb + d];
          const double step = coeff * diff;
          g[ib + d] += step;
          g[jb + d] -= step;
        }
      }
    }
  });

  parallel_for(static_cast<int>(grad.size()), n_threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      double value = 0.0;
      for (int t = 0; t < n_threads; ++t) {
        value += local_grad[static_cast<std::size_t>(t)][static_cast<std::size_t>(index)];
      }
      grad[static_cast<std::size_t>(index)] = value;
    }
  });
}

void initialize_tsne_transform(const NumericMatrix& reference_layout,
                               const IntegerMatrix& indices,
                               const NumericMatrix& distances,
                               const int offset,
                               const std::string& initialization,
                               const int seed,
                               std::vector<double>& y) {
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  const int dims = reference_layout.ncol();

  if (initialization == "random") {
    const unsigned int resolved_seed = seed == NA_INTEGER ?
      5489u :
      static_cast<unsigned int>(seed);
    std::mt19937 rng(resolved_seed);
    std::normal_distribution<double> normal(0.0, 1.0e-4);
    for (double& value : y) value = normal(rng);
    return;
  }

  std::vector<double> values(static_cast<std::size_t>(k), 0.0);
  for (int i = 0; i < n_query; ++i) {
    const std::size_t ib = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) {
      if (initialization == "weighted") {
        double numerator = 0.0;
        double denominator = DBL_MIN;
        for (int j = 0; j < k; ++j) {
          const int ref = indices(i, j) - offset;
          const double distance = std::max(0.0, distances(i, j));
          const double weight = 1.0 / (distance + 1e-6);
          numerator += weight * reference_layout(ref, d);
          denominator += weight;
        }
        y[ib + d] = numerator / denominator;
      } else {
        for (int j = 0; j < k; ++j) {
          const int ref = indices(i, j) - offset;
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
}

void compute_tsne_transform_gradient(const NumericMatrix& reference_layout,
                                     const IntegerMatrix& indices,
                                     const std::vector<std::vector<double>>& probabilities,
                                     const std::vector<double>& y,
                                     const int offset,
                                     const double exaggeration,
                                     const int n_negatives,
                                     const int exact_repulsion_threshold,
                                     const int n_threads,
                                     const int seed,
                                     std::vector<double>& grad) {
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  const int n_reference = reference_layout.nrow();
  const int dims = reference_layout.ncol();
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  std::fill(grad.begin(), grad.end(), 0.0);

  parallel_for(n_query, n_threads, [&](const int begin, const int end, const int thread_id) {
    const std::uint32_t stream_seed =
      static_cast<std::uint32_t>(seed) ^
      (static_cast<std::uint32_t>(thread_id + 1) * 0x85ebca6bu);
    std::mt19937 rng(stream_seed);
    std::uniform_int_distribution<int> uniform_ref(0, std::max(0, n_reference - 1));
    std::vector<int> sampled(static_cast<std::size_t>(std::max(1, n_negatives)), 0);
    std::vector<double> q_values;
    if (exact_repulsion) {
      q_values.resize(static_cast<std::size_t>(n_reference), 0.0);
    } else {
      q_values.resize(static_cast<std::size_t>(std::max(1, n_negatives)), 0.0);
    }

    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      double sum_q = DBL_MIN;

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
          const int ref = uniform_ref(rng);
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

      const std::vector<double>& row_p = probabilities[static_cast<std::size_t>(i)];
      for (int j = 0; j < k; ++j) {
        const int ref = indices(i, j) - offset;
        double d2 = 0.0;
        for (int d = 0; d < dims; ++d) {
          const double diff = y[ib + d] - reference_layout(ref, d);
          d2 += diff * diff;
        }
        const double q = 1.0 / (1.0 + d2);
        const double coeff = exaggeration * row_p[static_cast<std::size_t>(j)] * q;
        for (int d = 0; d < dims; ++d) {
          grad[ib + d] += coeff * (y[ib + d] - reference_layout(ref, d));
        }
      }
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

// The public R wrapper follows the behaviour of Rtsne::Rtsne_neighbors.
// This C++ optimizer is an independent fastEmbedR implementation: it does not
// copy Rtsne's Barnes-Hut source files, whose original Delft license includes
// an advertising clause.
// [[Rcpp::export]]
List knn_tsne_rtsne_cpp(IntegerMatrix indices,
                        NumericMatrix distances,
                        NumericMatrix y_init,
                        bool init,
                        int n_components,
                        double perplexity,
                        double theta,
                        int max_iter,
                        int stop_lying_iter,
                        int mom_switch_iter,
                        double momentum,
                        double final_momentum,
                        double eta,
                        double exaggeration_factor,
                        std::string negative_gradient_method,
                        int n_threads,
                        int seed,
                        bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 2 || k < 1) Rcpp::stop("KNN input must have at least two rows and one neighbor column.");
  if (n - 1 < 3.0 * perplexity) Rcpp::stop("perplexity is too large for the number of samples.");
  if (n_components < 1 || n_components > 3) Rcpp::stop("`n_components` must be 1, 2, or 3 for t-SNE.");
  if (max_iter < 1) Rcpp::stop("`max_iter` must be positive.");
  if (eta <= 0.0) Rcpp::stop("`eta` must be positive.");
  if (momentum < 0.0 || final_momentum < 0.0) Rcpp::stop("momentum values must be non-negative.");
  if (exaggeration_factor <= 0.0) Rcpp::stop("`exaggeration_factor` must be positive.");
  if (theta < 0.0 || theta > 1.0) Rcpp::stop("`theta` must lie in [0, 1].");

  const int threads = resolve_threads(n_threads, n);
  if (verbose) {
    Rcpp::Rcout << "fastEmbedR t-SNE from KNN: n=" << n
                << ", k=" << k
                << ", perplexity=" << perplexity
                << ", threads=" << threads << "\n";
  }

  SparseProbabilities p = build_tsne_probabilities(indices, distances, perplexity, threads);
  const std::string repulsion_mode = tsne_repulsion_mode(n, theta, negative_gradient_method);
  const int repulsion_block_size = tsne_repulsion_block_size(n);
  std::string optimizer_name = "exact_sparse_knn";
  if (repulsion_mode == "keops_blocked") optimizer_name = "exact_sparse_knn_keops_blocked";
  if (repulsion_mode == "barnes_hut") optimizer_name = "barnes_hut_sparse_knn";
  if (verbose) {
    Rcpp::Rcout << "t-SNE repulsion: " << repulsion_mode
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
  std::vector<double> costs(static_cast<std::size_t>(n), 0.0);
  NumericVector iter_costs(static_cast<int>(std::ceil(static_cast<double>(max_iter) / 50.0)));
  int cost_index = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    if ((iter & 7) == 0) Rcpp::checkUserInterrupt();
    const double current_exaggeration = iter < stop_lying_iter ? exaggeration_factor : 1.0;
    const double current_momentum = iter >= mom_switch_iter ? final_momentum : momentum;
    compute_gradient(
      p,
      y,
      n,
      n_components,
      current_exaggeration,
      threads,
      repulsion_mode,
      repulsion_block_size,
      theta,
      grad
    );

    parallel_for(static_cast<int>(y.size()), threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        gains[static_cast<std::size_t>(i)] =
          sign_tsne(grad[static_cast<std::size_t>(i)]) != sign_tsne(update[static_cast<std::size_t>(i)]) ?
            gains[static_cast<std::size_t>(i)] + 0.2 :
            gains[static_cast<std::size_t>(i)] * 0.8;
        if (gains[static_cast<std::size_t>(i)] < 0.01) gains[static_cast<std::size_t>(i)] = 0.01;
        update[static_cast<std::size_t>(i)] =
          current_momentum * update[static_cast<std::size_t>(i)] -
          eta * gains[static_cast<std::size_t>(i)] * grad[static_cast<std::size_t>(i)];
        y[static_cast<std::size_t>(i)] += update[static_cast<std::size_t>(i)];
      }
    });
    zero_mean(y, n, n_components);

    if ((iter > 0 && (iter + 1) % 50 == 0) || iter == max_iter - 1) {
      const double kl = evaluate_kl(p, y, n, n_components, threads);
      if (cost_index < iter_costs.size()) iter_costs[cost_index++] = kl;
      if (verbose) Rcpp::Rcout << "Iteration " << (iter + 1) << ": error is " << kl << "\n";
    }
  }

  evaluate_kl(p, y, n, n_components, threads, &costs);

  NumericMatrix layout(n, n_components);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_components; ++d) {
      layout(i, d) = y[static_cast<std::size_t>(i) * n_components + d];
    }
  }

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("costs") = NumericVector(costs.begin(), costs.end()),
    Rcpp::Named("itercosts") = iter_costs,
    Rcpp::Named("optimizer") = optimizer_name,
    Rcpp::Named("repulsion") = repulsion_mode,
    Rcpp::Named("repulsion_block_size") = repulsion_block_size,
    Rcpp::Named("theta_requested") = theta,
    Rcpp::Named("n_threads") = threads
  );
}

// [[Rcpp::export]]
List knn_infotsne_cpp(IntegerMatrix indices,
                      NumericMatrix distances,
                      NumericMatrix y_init,
                      bool init,
                      int n_components,
                      double perplexity,
                      int max_iter,
                      int early_exaggeration_iter,
                      double momentum,
                      double final_momentum,
                      double learning_rate,
                      double early_exaggeration_coeff,
                      double repulsion_strength,
                      int n_negatives,
                      int n_threads,
                      int seed,
                      bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 2 || k < 1) Rcpp::stop("KNN input must have at least two rows and one neighbor column.");
  if (n - 1 < 3.0 * perplexity) Rcpp::stop("perplexity is too large for the number of samples.");
  if (n_components < 1 || n_components > 3) Rcpp::stop("`n_components` must be 1, 2, or 3 for InfoTSNE.");
  if (max_iter < 1) Rcpp::stop("`max_iter` must be positive.");
  if (early_exaggeration_iter < 0) Rcpp::stop("`early_exaggeration_iter` must be non-negative.");
  if (momentum < 0.0 || final_momentum < 0.0) Rcpp::stop("momentum values must be non-negative.");
  if (learning_rate <= 0.0) Rcpp::stop("`learning_rate` must be positive.");
  if (early_exaggeration_coeff <= 0.0) Rcpp::stop("`early_exaggeration_coeff` must be positive.");
  if (repulsion_strength <= 0.0) Rcpp::stop("`repulsion_strength` must be positive.");
  if (n_negatives < 1) Rcpp::stop("`n_negatives` must be positive.");
  if (n_negatives > n - 1) {
    Rcpp::warning("`n_negatives` exceeds n - 1; capping to n - 1.");
    n_negatives = n - 1;
  }

  const int threads = resolve_threads(n_threads, n);
  if (verbose) {
    Rcpp::Rcout << "fastEmbedR InfoTSNE from KNN: n=" << n
                << ", k=" << k
                << ", perplexity=" << perplexity
                << ", negatives=" << n_negatives
                << ", threads=" << threads << "\n";
  }

  SparseProbabilities p = build_tsne_probabilities(indices, distances, perplexity, threads);

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
  std::vector<double> velocity(y.size(), 0.0);

  for (int iter = 0; iter < max_iter; ++iter) {
    if ((iter & 7) == 0) Rcpp::checkUserInterrupt();
    const double current_exaggeration = iter < early_exaggeration_iter ?
      early_exaggeration_coeff :
      1.0;
    const double current_momentum = iter >= early_exaggeration_iter ?
      final_momentum :
      momentum;
    const double decay = 1.0 - (static_cast<double>(iter) / std::max(1.0, static_cast<double>(max_iter)));
    const double current_lr = learning_rate * std::max(0.01, decay);
    const int iter_seed = (seed == NA_INTEGER ? 5489 : seed) +
      104729 * (iter + 1);

    compute_infotsne_gradient(
      p,
      y,
      n,
      n_components,
      current_exaggeration,
      repulsion_strength,
      n_negatives,
      threads,
      iter_seed,
      grad
    );

    parallel_for(static_cast<int>(y.size()), threads, [&](const int begin, const int end, const int) {
      for (int i = begin; i < end; ++i) {
        velocity[static_cast<std::size_t>(i)] =
          current_momentum * velocity[static_cast<std::size_t>(i)] -
          current_lr * grad[static_cast<std::size_t>(i)];
        y[static_cast<std::size_t>(i)] += velocity[static_cast<std::size_t>(i)];
      }
    });
    zero_mean(y, n, n_components);
  }

  NumericMatrix layout(n, n_components);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_components; ++d) {
      layout(i, d) = y[static_cast<std::size_t>(i) * n_components + d];
    }
  }

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("optimizer") = "infotsne_negative_sampling",
    Rcpp::Named("objective") = "InfoTSNE",
    Rcpp::Named("n_negatives") = n_negatives,
    Rcpp::Named("n_threads") = threads
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

  std::vector<std::vector<double>> probabilities(static_cast<std::size_t>(n_query));
  parallel_for(n_query, threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      compute_row_probabilities(distances, i, perplexity, probabilities[static_cast<std::size_t>(i)]);
    }
  });

  std::vector<double> y(static_cast<std::size_t>(n_query) * dims, 0.0);
  if (init) {
    if (y_init.nrow() != n_query || y_init.ncol() != dims) {
      Rcpp::stop("`Y_init` has the wrong shape.");
    }
    for (int i = 0; i < n_query; ++i) {
      for (int d = 0; d < dims; ++d) {
        y[static_cast<std::size_t>(i) * dims + d] = y_init(i, d);
      }
    }
  } else {
    initialize_tsne_transform(
      reference_layout,
      indices,
      distances,
      offset,
      initialization,
      seed,
      y
    );
  }

  std::vector<double> grad(y.size(), 0.0);
  std::vector<double> update(y.size(), 0.0);
  std::vector<double> gains(y.size(), 1.0);
  const int total_iter = early_exaggeration_iter + n_iter;

  for (int iter = 0; iter < total_iter; ++iter) {
    if ((iter & 7) == 0) Rcpp::checkUserInterrupt();
    const bool in_early = iter < early_exaggeration_iter;
    const double current_exaggeration = in_early ? early_exaggeration : exaggeration;
    const double current_momentum = in_early ? initial_momentum : final_momentum;
    const int iter_seed = (seed == NA_INTEGER ? 5489 : seed) + 65537 * (iter + 1);

    compute_tsne_transform_gradient(
      reference_layout,
      indices,
      probabilities,
      y,
      offset,
      current_exaggeration,
      n_negatives,
      exact_repulsion_threshold,
      threads,
      iter_seed,
      grad
    );

    parallel_for(n_query, threads, [&](const int begin, const int end, const int) {
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

  NumericMatrix layout(n_query, dims);
  for (int i = 0; i < n_query; ++i) {
    for (int d = 0; d < dims; ++d) {
      layout(i, d) = y[static_cast<std::size_t>(i) * dims + d];
    }
  }

  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("optimizer") = "opentsne_style_fixed_reference_transform",
    Rcpp::Named("initialization") = initialization,
    Rcpp::Named("repulsion") = exact_repulsion ? "exact_reference" : "sampled_reference",
    Rcpp::Named("n_negatives") = n_negatives,
    Rcpp::Named("n_threads") = threads
  );
}
