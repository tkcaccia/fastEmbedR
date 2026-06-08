#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

namespace {

struct ObjEdge {
  int i;
  int j;
  double w;
};

struct ObjPairHash {
  std::size_t operator()(const std::uint64_t x) const noexcept {
    return static_cast<std::size_t>(x ^ (x >> 32));
  }
};

std::uint64_t obj_key(const int a, const int b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
         static_cast<std::uint32_t>(b);
}

double obj_clip(const double x, const double lo, const double hi) {
  return std::max(lo, std::min(hi, x));
}

std::uint32_t mix_seed(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

int deterministic_vertex(const int n, const int seed, const int epoch, const std::size_t edge, const int sample) {
  std::uint32_t x = static_cast<std::uint32_t>(seed);
  x ^= static_cast<std::uint32_t>(epoch * 0x9e3779b9u);
  x ^= static_cast<std::uint32_t>((edge + 1u) * 0x85ebca6bu);
  x ^= static_cast<std::uint32_t>((sample + 1) * 0xc2b2ae35u);
  return static_cast<int>(mix_seed(x) % static_cast<std::uint32_t>(n));
}

void smooth_obj_knn(const NumericMatrix& distances,
                    std::vector<double>& sigmas,
                    std::vector<double>& rhos) {
  const int n = distances.nrow();
  const int k = distances.ncol();
  const double target = std::log2(static_cast<double>(std::max(2, k)));
  sigmas.assign(n, 1.0);
  rhos.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      if (d > 0.0 && d < rho) rho = d;
    }
    if (!std::isfinite(rho)) rho = 0.0;
    rhos[i] = rho;

    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double mid = 1.0;
    for (int iter = 0; iter < 48; ++iter) {
      double psum = 0.0;
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j) - rho;
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

std::vector<ObjEdge> objective_edges(const IntegerMatrix& indices,
                                     const NumericMatrix& distances) {
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
  smooth_obj_knn(distances, sigmas, rhos);

  std::unordered_map<std::uint64_t, double, ObjPairHash> weights;
  weights.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int nb = indices(i, j) - offset;
      if (nb < 0 || nb >= n || nb == i) continue;
      const double d = distances(i, j);
      const double wij = d <= rhos[i] ? 1.0 : std::exp(-(d - rhos[i]) / sigmas[i]);
      const int a = std::min(i, nb);
      const int b = std::max(i, nb);
      const auto key = obj_key(a, b);
      const auto it = weights.find(key);
      if (it == weights.end()) {
        weights.emplace(key, wij);
      } else {
        const double old = it->second;
        it->second = old + wij - old * wij;
      }
    }
  }

  std::vector<ObjEdge> edges;
  edges.reserve(weights.size());
  for (const auto& kv : weights) {
    edges.push_back({
      static_cast<int>(kv.first >> 32),
      static_cast<int>(kv.first & 0xffffffffu),
      std::max(kv.second, 1e-6)
    });
  }
  return edges;
}

NumericMatrix random_init(const int n, const int n_components, const int seed) {
  NumericMatrix y(n, n_components);
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<double> normal(0.0, 0.0001);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < n_components; ++c) y(i, c) = normal(rng);
  }
  return y;
}

void add_pair_force(std::vector<double>& delta,
                    const NumericMatrix& y,
                    const int n_components,
                    const int i,
                    const int j,
                    const double coeff,
                    const bool update_j) {
  for (int c = 0; c < n_components; ++c) {
    const double grad = obj_clip(coeff * (y(i, c) - y(j, c)), -4.0, 4.0);
    delta[static_cast<std::size_t>(i) * n_components + c] += grad;
    if (update_j) delta[static_cast<std::size_t>(j) * n_components + c] -= grad;
  }
}

double attractive_coeff(const std::string& objective, const double d2, const double w) {
  if (objective == "tsne") return -2.0 * w / (1.0 + d2);
  if (objective == "pacmap") return -2.0 * w / (10.0 + d2);
  if (objective == "trimap") return -2.0 * w / (1.0 + std::sqrt(d2 + 1e-6));
  if (objective == "localmap") return -2.0 * w / (0.25 + d2);
  return -2.0 * w / (1.0 + d2);
}

double repulsive_coeff(const std::string& objective, const double d2, const double weight) {
  if (objective == "tsne") return 2.0 * weight / ((1.0 + d2) * (1.0 + d2));
  if (objective == "pacmap") return 2.0 * weight / (1.0 + d2);
  if (objective == "trimap") return 2.0 * weight / (1.0 + std::sqrt(d2 + 1e-6));
  if (objective == "localmap") return 2.0 * weight / (0.1 + d2);
  return 2.0 * weight / ((0.001 + d2) * (1.0 + d2));
}

NumericMatrix optimize_objective(const int n,
                                 const int n_components,
                                 const std::vector<ObjEdge>& edges,
                                 const std::string& objective,
                                 NumericMatrix init_embedding,
                                 const bool use_init,
                                 const int n_epochs,
                                 const int negative_sample_rate,
                                 const double learning_rate,
                                 const int n_threads,
                                 const int seed,
                                 const bool verbose) {
  NumericMatrix y = use_init ? Rcpp::clone(init_embedding) : random_init(n, n_components, seed);
  const int threads = std::max(1, std::min(n_threads, static_cast<int>(edges.size())));
  const double repulse_weight =
    objective == "pacmap" ? 0.2 :
    objective == "trimap" ? 0.5 :
    objective == "localmap" ? 0.35 : 1.0;

  for (int epoch = 1; epoch <= n_epochs; ++epoch) {
    const double alpha = learning_rate * (1.0 - static_cast<double>(epoch - 1) / std::max(1, n_epochs));
    std::vector<std::vector<double>> deltas(
      threads, std::vector<double>(static_cast<std::size_t>(n) * n_components, 0.0)
    );
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        const std::size_t begin = edges.size() * static_cast<std::size_t>(t) / threads;
        const std::size_t end = edges.size() * static_cast<std::size_t>(t + 1) / threads;
        auto& delta = deltas[t];
        for (std::size_t e = begin; e < end; ++e) {
          const int i = edges[e].i;
          const int j = edges[e].j;
          double d2 = 0.0;
          for (int c = 0; c < n_components; ++c) {
            const double diff = y(i, c) - y(j, c);
            d2 += diff * diff;
          }
          add_pair_force(delta, y, n_components, i, j, attractive_coeff(objective, d2, edges[e].w), true);

          for (int s = 0; s < negative_sample_rate; ++s) {
            int neg = deterministic_vertex(n, seed, epoch, e, s);
            if (neg == i || neg == j) neg = (neg + 1) % n;
            double nd2 = 0.0;
            for (int c = 0; c < n_components; ++c) {
              const double diff = y(i, c) - y(neg, c);
              nd2 += diff * diff;
            }
            add_pair_force(delta, y, n_components, i, neg, repulsive_coeff(objective, nd2, repulse_weight), false);

            if (objective == "trimap") {
              int far = deterministic_vertex(n, seed + 17, epoch, e, s);
              if (far == j || far == i) far = (far + 3) % n;
              double fd2 = 0.0;
              for (int c = 0; c < n_components; ++c) {
                const double diff = y(j, c) - y(far, c);
                fd2 += diff * diff;
              }
              add_pair_force(delta, y, n_components, j, far, repulsive_coeff(objective, fd2, 0.25), false);
            }
          }
        }
      });
    }

    for (auto& worker : workers) worker.join();
    for (int t = 0; t < threads; ++t) {
      const auto& delta = deltas[t];
      for (int i = 0; i < n; ++i) {
        for (int c = 0; c < n_components; ++c) {
          y(i, c) += alpha * delta[static_cast<std::size_t>(i) * n_components + c];
        }
      }
    }

    if (verbose && (epoch == 1 || epoch == n_epochs || epoch % 50 == 0)) {
      Rcpp::Rcout << objective << " epoch " << epoch << "/" << n_epochs << "\n";
    }
  }

  return y;
}

} // namespace

// [[Rcpp::export]]
NumericMatrix knn_objective_embed_cpp(IntegerMatrix indices,
                                      NumericMatrix distances,
                                      std::string objective,
                                      NumericMatrix init_embedding,
                                      bool use_init,
                                      int n_components,
                                      int n_epochs,
                                      int negative_sample_rate,
                                      double learning_rate,
                                      int n_threads,
                                      int seed,
                                      bool verbose) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (objective != "tsne" && objective != "pacmap" && objective != "trimap" && objective != "localmap") {
    Rcpp::stop("unknown objective");
  }
  if (n_components < 1) Rcpp::stop("n_components must be positive");
  if (use_init && (init_embedding.nrow() != indices.nrow() || init_embedding.ncol() != n_components)) {
    Rcpp::stop("init_embedding dimensions do not match");
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  const std::vector<ObjEdge> edges = objective_edges(indices, distances);
  if (edges.empty()) Rcpp::stop("The KNN graph has no usable edges.");
  return optimize_objective(
    indices.nrow(), n_components, edges, objective, init_embedding, use_init, n_epochs,
    negative_sample_rate, learning_rate, n_threads, seed, verbose
  );
}
