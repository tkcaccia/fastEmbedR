#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

namespace {

struct Edge {
  int head;
  int tail;
  double weight;
};

struct PairHash {
  std::size_t operator()(const std::uint64_t x) const noexcept {
    return static_cast<std::size_t>(x ^ (x >> 32));
  }
};

std::uint64_t edge_key(const int a, const int b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
         static_cast<std::uint32_t>(b);
}

double clip_value(const double x, const double lo, const double hi) {
  return std::max(lo, std::min(hi, x));
}

std::pair<double, double> find_ab_params(const double spread, const double min_dist) {
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

void smooth_knn_dist(const NumericMatrix& distances,
                     const double local_connectivity,
                     std::vector<double>& sigmas,
                     std::vector<double>& rhos) {
  const int n = distances.nrow();
  const int k = distances.ncol();
  const double target = std::log2(static_cast<double>(k));
  sigmas.assign(n, 1.0);
  rhos.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    std::vector<double> positive;
    positive.reserve(k);
    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      if (d > 0.0) positive.push_back(d);
    }
    std::sort(positive.begin(), positive.end());

    if (!positive.empty()) {
      const int index = static_cast<int>(std::floor(local_connectivity));
      const double interpolation = local_connectivity - index;
      if (index > 0) {
        if (index < static_cast<int>(positive.size())) {
          rhos[i] = positive[index - 1] +
                    interpolation * (positive[index] - positive[index - 1]);
        } else {
          rhos[i] = positive.back();
        }
      } else {
        rhos[i] = interpolation * positive.front();
      }
    }

    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double mid = 1.0;
    for (int iter = 0; iter < 64; ++iter) {
      double psum = 0.0;
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j) - rhos[i];
        psum += d <= 0.0 ? 1.0 : std::exp(-d / mid);
      }

      if (std::abs(psum - target) < 1e-5) break;
      if (psum > target) {
        hi = mid;
        mid = (lo + hi) / 2.0;
      } else {
        lo = mid;
        mid = std::isinf(hi) ? mid * 2.0 : (lo + hi) / 2.0;
      }
    }
    sigmas[i] = std::max(mid, 1e-6);
  }
}

std::vector<Edge> build_graph(const IntegerMatrix& indices,
                              const NumericMatrix& distances,
                              const double local_connectivity,
                              const double set_op_mix_ratio) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  const int offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  std::vector<double> sigmas;
  std::vector<double> rhos;
  smooth_knn_dist(distances, local_connectivity, sigmas, rhos);

  std::unordered_map<std::uint64_t, double, PairHash> directed;
  directed.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int neighbor = indices(i, j) - offset;
      if (neighbor < 0 || neighbor >= n || neighbor == i) continue;
      const double d = distances(i, j);
      const double val = d <= rhos[i] ? 1.0 : std::exp(-(d - rhos[i]) / sigmas[i]);
      if (val > 0.0) directed[edge_key(i, neighbor)] = std::max(directed[edge_key(i, neighbor)], val);
    }
  }

  std::unordered_map<std::uint64_t, double, PairHash> undirected;
  undirected.reserve(directed.size());
  for (const auto& kv : directed) {
    const int i = static_cast<int>(kv.first >> 32);
    const int j = static_cast<int>(kv.first & 0xffffffffu);
    if (i >= j) continue;
    const auto it_forward = directed.find(edge_key(i, j));
    const auto it_reverse = directed.find(edge_key(j, i));
    const double a = it_forward == directed.end() ? 0.0 : it_forward->second;
    const double b = it_reverse == directed.end() ? 0.0 : it_reverse->second;
    const double prod = a * b;
    const double fuzzy_union = a + b - prod;
    const double fuzzy_intersection = prod;
    const double w = set_op_mix_ratio * fuzzy_union +
                     (1.0 - set_op_mix_ratio) * fuzzy_intersection;
    if (w > 1e-6) undirected[edge_key(i, j)] = w;
  }

  for (const auto& kv : directed) {
    const int i = static_cast<int>(kv.first >> 32);
    const int j = static_cast<int>(kv.first & 0xffffffffu);
    if (i < j) continue;
    const auto key = edge_key(j, i);
    if (undirected.find(key) != undirected.end()) continue;
    const auto it_reverse = directed.find(edge_key(j, i));
    const double a = kv.second;
    const double b = it_reverse == directed.end() ? 0.0 : it_reverse->second;
    const double prod = a * b;
    const double fuzzy_union = a + b - prod;
    const double fuzzy_intersection = prod;
    const double w = set_op_mix_ratio * fuzzy_union +
                     (1.0 - set_op_mix_ratio) * fuzzy_intersection;
    if (w > 1e-6) undirected[key] = w;
  }

  std::vector<Edge> edges;
  edges.reserve(undirected.size());
  for (const auto& kv : undirected) {
    edges.push_back({
      static_cast<int>(kv.first >> 32),
      static_cast<int>(kv.first & 0xffffffffu),
      kv.second
    });
  }
  return edges;
}

NumericMatrix initialize_layout(const int n,
                                const int n_components,
                                const std::vector<Edge>& edges,
                                const std::string& init,
                                const std::string& init_sdev_mode,
                                const double init_sdev_value,
                                const int spectral_n_iter,
                                std::mt19937& rng) {
  NumericMatrix embedding(n, n_components);
  std::normal_distribution<double> normal(0.0, 0.0001);
  std::uniform_real_distribution<double> uniform(-10.0, 10.0);

  if (init == "random" || edges.empty()) {
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
    for (const auto& e : edges) {
      const double denom = std::sqrt(std::max(degree[e.head] * degree[e.tail], 1e-24));
      const double scaled = e.weight / denom;
      for (int c = 0; c < cols; ++c) {
        y[static_cast<std::size_t>(c) * n + e.head] += scaled * cat(x, e.tail, c);
        y[static_cast<std::size_t>(c) * n + e.tail] += scaled * cat(x, e.head, c);
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
    for (int i = 0; i < n; ++i) embedding(i, c) = values[i] + normal(rng);
  }

  if (init_sdev_mode == "range") {
    for (int c = 0; c < n_components; ++c) {
      double lo = embedding(0, c);
      double hi = embedding(0, c);
      for (int i = 1; i < n; ++i) {
        lo = std::min(lo, embedding(i, c));
        hi = std::max(hi, embedding(i, c));
      }
      const double span = hi - lo;
      if (span > 0.0) {
        for (int i = 0; i < n; ++i) embedding(i, c) = 10.0 * (embedding(i, c) - lo) / span;
      }
    }
  } else if (init_sdev_mode == "sd") {
    for (int c = 0; c < n_components; ++c) {
      double mean = 0.0;
      for (int i = 0; i < n; ++i) mean += embedding(i, c);
      mean /= n;
      double ss = 0.0;
      for (int i = 0; i < n; ++i) {
        const double centered = embedding(i, c) - mean;
        ss += centered * centered;
      }
      const double sd = std::sqrt(std::max(ss / n, 0.0));
      if (sd > 0.0) {
        const double scale = init_sdev_value / sd;
        for (int i = 0; i < n; ++i) embedding(i, c) = (embedding(i, c) - mean) * scale;
      }
    }
  }
  return embedding;
}

NumericMatrix optimize_layout(const int n,
                              const int n_components,
                              const std::vector<Edge>& input_edges,
                              const int n_epochs,
                              const double min_dist,
                              const double spread,
                              const int negative_sample_rate,
                              const double learning_rate,
                              const double curve_a,
                              const double curve_b,
                              const double repulsion_strength,
                              const std::string& mode,
                              const std::string& init,
                              const std::string& init_sdev_mode,
                              const double init_sdev_value,
                              const bool prune_epochs,
                              const int spectral_n_iter,
                              const int seed,
                              const bool verbose) {
  std::vector<Edge> edges = input_edges;
  if (edges.empty()) Rcpp::stop("The KNN graph has no usable edges.");

  const auto ab = (std::isfinite(curve_a) && std::isfinite(curve_b))
    ? std::make_pair(curve_a, curve_b)
    : find_ab_params(spread, min_dist);
  const double a = ab.first;
  const double b = ab.second;
  const double gamma = repulsion_strength;
  const double max_weight = std::max_element(
    edges.begin(), edges.end(),
    [](const Edge& x, const Edge& y) { return x.weight < y.weight; }
  )->weight;

  if (prune_epochs && n_epochs > 0) {
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

  std::vector<double> epochs_per_sample(edges.size());
  std::vector<double> epoch_of_next_sample(edges.size());
  std::vector<double> epochs_per_negative_sample(edges.size());
  std::vector<double> epoch_of_next_negative_sample(edges.size());

  for (std::size_t i = 0; i < edges.size(); ++i) {
    epochs_per_sample[i] = max_weight / std::max(edges[i].weight, 1e-6);
    epoch_of_next_sample[i] = epochs_per_sample[i];
    if (negative_sample_rate > 0) {
      epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
      epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
    } else {
      epochs_per_negative_sample[i] = std::numeric_limits<double>::infinity();
      epoch_of_next_negative_sample[i] = std::numeric_limits<double>::infinity();
    }
  }

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::uniform_int_distribution<int> vertex_dist(0, n - 1);
  NumericMatrix embedding = initialize_layout(
    n, n_components, edges, init, init_sdev_mode, init_sdev_value, spectral_n_iter, rng
  );
  if (mode == "spectral" || n_epochs == 0) {
    return embedding;
  }

  for (int epoch = 1; epoch <= n_epochs; ++epoch) {
    const double alpha = learning_rate * (1.0 - static_cast<double>(epoch - 1) / n_epochs);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      if (epoch_of_next_sample[i] > epoch) continue;
      const int j = edges[i].head;
      const int k = edges[i].tail;

      double dist_sq = 0.0;
      for (int c = 0; c < n_components; ++c) {
        const double diff = embedding(j, c) - embedding(k, c);
        dist_sq += diff * diff;
      }

      double grad_coeff = 0.0;
      if (dist_sq > 0.0) {
        grad_coeff = -2.0 * a * b * std::pow(dist_sq, b - 1.0) /
                     (a * std::pow(dist_sq, b) + 1.0);
      }
      for (int c = 0; c < n_components; ++c) {
        const double grad = clip_value(grad_coeff * (embedding(j, c) - embedding(k, c)), -4.0, 4.0);
        embedding(j, c) += grad * alpha;
        embedding(k, c) -= grad * alpha;
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
        const int neg = vertex_dist(rng);
        if (neg == j) continue;

        double neg_dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = embedding(j, c) - embedding(neg, c);
          neg_dist_sq += diff * diff;
        }
        double repulse = 0.0;
        if (neg_dist_sq > 0.0) {
          repulse = 2.0 * gamma * b /
                    ((0.001 + neg_dist_sq) * (a * std::pow(neg_dist_sq, b) + 1.0));
        }
        for (int c = 0; c < n_components; ++c) {
          const double grad = clip_value(repulse * (embedding(j, c) - embedding(neg, c)), -4.0, 4.0);
          embedding(j, c) += grad * alpha;
        }
      }
      epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i];
    }

    if (verbose && (epoch == 1 || epoch == n_epochs || epoch % 50 == 0)) {
      Rcpp::Rcout << "epoch " << epoch << "/" << n_epochs << "\n";
    }
  }

  return embedding;
}

} // namespace

// [[Rcpp::export]]
NumericMatrix fast_knn_umap_cpp(IntegerMatrix indices,
                                NumericMatrix distances,
                                int n_components,
                                int n_epochs,
                                double min_dist,
                                double spread,
                                double local_connectivity,
                                double set_op_mix_ratio,
                                int negative_sample_rate,
                                double learning_rate,
                                double curve_a,
                                double curve_b,
                                double repulsion_strength,
                                std::string mode,
                                std::string init,
                                std::string init_sdev_mode,
                                double init_sdev_value,
                                bool prune_epochs,
                                int spectral_n_iter,
                                int seed,
                                bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (n_components < 1) Rcpp::stop("n_components must be positive");
  if (n_epochs < 0) Rcpp::stop("n_epochs must be non-negative");
  if (mode != "sgd" && mode != "spectral" && mode != "hybrid") {
    Rcpp::stop("mode must be 'sgd', 'spectral', or 'hybrid'");
  }
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (spread <= 0.0) Rcpp::stop("spread must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (std::isfinite(curve_a) && curve_a <= 0.0) Rcpp::stop("a must be positive");
  if (std::isfinite(curve_b) && curve_b <= 0.0) Rcpp::stop("b must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (init_sdev_mode != "none" && init_sdev_mode != "range" && init_sdev_mode != "sd") {
    Rcpp::stop("init_sdev_mode must be 'none', 'range', or 'sd'");
  }
  if (init_sdev_mode == "sd" && (!std::isfinite(init_sdev_value) || init_sdev_value <= 0.0)) {
    Rcpp::stop("init_sdev_value must be positive when init_sdev_mode is 'sd'");
  }
  if (set_op_mix_ratio < 0.0 || set_op_mix_ratio > 1.0) {
    Rcpp::stop("set_op_mix_ratio must be in [0, 1]");
  }

  const int n = indices.nrow();
  const std::vector<Edge> edges = build_graph(
    indices, distances, local_connectivity, set_op_mix_ratio
  );
  return optimize_layout(
    n, n_components, edges, n_epochs, min_dist, spread,
    negative_sample_rate, learning_rate, curve_a, curve_b, repulsion_strength,
    mode, init, init_sdev_mode, init_sdev_value, prune_epochs,
    spectral_n_iter, seed, verbose
  );
}
