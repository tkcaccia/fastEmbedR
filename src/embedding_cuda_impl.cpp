#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

extern "C" {
bool fastembedr_cuda_available();
const char* fastembedr_cuda_embedding_last_error();
int fastembedr_cuda_embed(const int* neighbors,
                          const float* weights,
                          const float* init,
                          int n,
                          int k,
                          int objective,
                          int n_epochs,
                          int negative_sample_rate,
                          float learning_rate,
                          float a,
                          float b,
                          float max_weight,
                          unsigned int seed,
                          float* out);
int fastembedr_cuda_embed_from_knn(const int* indices,
                                   const double* distances,
                                   const float* init,
                                   int n,
                                   int k,
                                   int objective,
                                   int n_epochs,
                                   int negative_sample_rate,
                                   float learning_rate,
                                   float a,
                                   float b,
                                   unsigned int seed,
                                   int index_offset,
                                   float* out);
int fastembedr_cuda_spectral_init_from_knn(const int* indices,
                                           const double* distances,
                                           int n,
                                           int k,
                                           int spectral_n_iter,
                                           unsigned int seed,
                                           int index_offset,
                                           float* out);
int fastembedr_cuda_exact_tsne_from_knn(const int* indices,
                                        const double* distances,
                                        const float* init,
                                        int n,
                                        int k,
                                        int n_epochs,
                                        float perplexity,
                                        float learning_rate,
                                        int stop_lying_iter,
                                        int mom_switch_iter,
                                        float momentum,
                                        float final_momentum,
                                        float exaggeration_factor,
                                        unsigned int seed,
                                        int index_offset,
                                        float* out);
int fastembedr_cuda_umap_from_knn_spectral(const int* indices,
                                           const double* distances,
                                           int n,
                                           int k,
                                           int n_epochs,
                                           int negative_sample_rate,
                                           float learning_rate,
                                           float a,
                                           float b,
                                           int spectral_n_iter,
                                           unsigned int seed,
                                           int index_offset,
                                           int optimizer_mode,
                                           float* out);
int fastembedr_cuda_standardize_matrix(const double* values,
                                       int n,
                                       int p,
                                       double* out,
                                       double* center,
                                       double* scale);
int fastembedr_cuda_project_embedding(const double* reference_layout,
                                      const int* projection_indices,
                                      const double* projection_distances,
                                      int n_reference,
                                      int n_query,
                                      int k,
                                      int n_components,
                                      double* out);
int fastembedr_cuda_interpolate_landmark_layout(const double* landmark_layout,
                                                const int* landmark_indices,
                                                const int* projection_indices,
                                                const double* projection_distances,
                                                int n_landmarks,
                                                int n,
                                                int k,
                                                int n_components,
                                                double* out);
int fastembedr_cuda_landmark_project_interpolate_knn_confidence(const double* landmark_data,
                                                                const double* query_data,
                                                                const double* landmark_layout,
                                                                const int* landmark_indices,
                                                                int n_landmarks,
                                                                int n,
                                                                int n_features,
                                                                int k,
                                                                int n_components,
                                                                double* out,
                                                                int* projection_indices,
                                                                double* projection_distances,
                                                                double* confidence);
int fastembedr_cuda_knn_structure_score(const double* layout,
                                        const int* indices,
                                        const int* keep,
                                        const int* labels,
                                        int n,
                                        int index_rows,
                                        int high_rank_limit,
                                        int preserve_k,
                                        int keep_n,
                                        int compact_indices,
                                        int n_label_levels,
                                        double* totals);
int fastembedr_cuda_silhouette_score(const double* layout,
                                     const int* labels,
                                     const int* counts,
                                     int n,
                                     int n_label_levels,
                                     double* score);
int fastembedr_cuda_matrix_multiply(const double* left,
                                    const double* right,
                                    int left_rows,
                                    int left_cols,
                                    int right_cols,
                                    int transpose_left,
                                    double* out);
}

namespace {

constexpr int kMaxCudaNeighbors = 256;
constexpr int kNativeCudaUmapPrepMinN = 500;
constexpr int kMaxCudaProjectionNeighbors = 128;
constexpr int kMaxCudaScoreNeighbors = 64;
constexpr int kMaxCudaSilhouetteLabels = 128;

enum ObjectiveId {
  kObjectiveUmap = 0,
  kObjectiveTsne = 1,
  kObjectivePacmap = 2,
  kObjectiveTrimap = 3,
  kObjectiveLocalmap = 4
};

struct WeightedEdge {
  std::uint64_t key;
  float weight;
  std::uint8_t direction;
};

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

bool weighted_edge_less(const WeightedEdge& a, const WeightedEdge& b) {
  return a.key < b.key;
}

int objective_id(const std::string& objective) {
  if (objective == "umap") return kObjectiveUmap;
  if (objective == "tsne") return kObjectiveTsne;
  if (objective == "pacmap") return kObjectivePacmap;
  if (objective == "trimap") return kObjectiveTrimap;
  if (objective == "localmap") return kObjectiveLocalmap;
  Rcpp::stop("Unknown CUDA embedding objective: %s", objective.c_str());
}

const char* cuda_embedding_error_message() {
  const char* msg = fastembedr_cuda_embedding_last_error();
  return msg == nullptr ? "unknown CUDA embedding error" : msg;
}

void smooth_knn_weights(const NumericMatrix& distances,
                        std::vector<float>& weights) {
  const int n = distances.nrow();
  const int k = distances.ncol();
  const double target = std::log2(static_cast<double>(std::max(2, k)));
  weights.assign(static_cast<std::size_t>(n) * k, 0.0f);

  for (int i = 0; i < n; ++i) {
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      if (d > 0.0 && d < rho) rho = d;
    }
    if (!std::isfinite(rho)) rho = 0.0;

    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double sigma = 1.0;
    for (int iter = 0; iter < 48; ++iter) {
      double psum = 0.0;
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j) - rho;
        psum += d <= 0.0 ? 1.0 : std::exp(-d / sigma);
      }
      if (std::abs(psum - target) < 1e-5) break;
      if (psum > target) {
        hi = sigma;
        sigma = (lo + hi) / 2.0;
      } else {
        lo = sigma;
        sigma = std::isinf(hi) ? sigma * 2.0 : (lo + hi) / 2.0;
      }
    }
    sigma = std::max(sigma, 1e-6);

    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      const double value = d <= rho ? 1.0 : std::exp(-(d - rho) / sigma);
      weights[static_cast<std::size_t>(i) * k + j] = static_cast<float>(value);
    }
  }
}

std::pair<double, double> find_ab_params(const double spread, const double min_dist) {
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.1) < 1e-12) {
    return {1.5769434601962196, 0.8950608781227859};
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

void prepare_knn(const IntegerMatrix& indices,
                 const NumericMatrix& distances,
                 std::vector<int>& neighbors,
                 std::vector<float>& weights) {
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

  neighbors.resize(static_cast<std::size_t>(n) * k);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      int nb = indices(i, j) - offset;
      if (nb < 0 || nb >= n || nb == i) nb = i;
      neighbors[static_cast<std::size_t>(i) * k + j] = nb;
    }
  }

  smooth_knn_weights(distances, weights);
}

void prepare_umap_graph_adjacency(const IntegerMatrix& indices,
                                  const NumericMatrix& distances,
                                  const int n_epochs,
                                  std::vector<int>& neighbors,
                                  std::vector<float>& weights) {
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

  const double target = std::log2(static_cast<double>(std::max(2, k)));
  std::vector<double> sigmas(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rhos(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
      const double d = distances(i, j);
      if (d > 0.0 && d < rho) rho = d;
    }
    if (!std::isfinite(rho)) rho = 0.0;
    rhos[static_cast<std::size_t>(i)] = rho;

    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double sigma = 1.0;
    for (int iter = 0; iter < 48; ++iter) {
      double psum = 0.0;
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j) - rho;
        psum += d <= 0.0 ? 1.0 : std::exp(-d / sigma);
      }
      if (std::abs(psum - target) < 1e-5) break;
      if (psum > target) {
        hi = sigma;
        sigma = (lo + hi) / 2.0;
      } else {
        lo = sigma;
        sigma = std::isinf(hi) ? sigma * 2.0 : (lo + hi) / 2.0;
      }
    }
    sigmas[static_cast<std::size_t>(i)] = std::max(sigma, 1e-6);
  }

  std::vector<WeightedEdge> directed;
  directed.reserve(static_cast<std::size_t>(n) * k);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const int nb = indices(i, j) - offset;
      if (nb < 0 || nb >= n || nb == i) continue;
      const double d = distances(i, j);
      const double rho = rhos[static_cast<std::size_t>(i)];
      const double sigma = sigmas[static_cast<std::size_t>(i)];
      const float w = static_cast<float>(d <= rho ? 1.0 : std::exp(-(d - rho) / sigma));
      if (w > 0.0f) directed.push_back({edge_key(i, nb), w, 0u});
    }
  }

  std::sort(directed.begin(), directed.end(), weighted_edge_less);
  std::size_t write = 0;
  for (std::size_t read = 0; read < directed.size(); ++read) {
    if (write > 0 && directed[write - 1].key == directed[read].key) {
      directed[write - 1].weight = std::max(directed[write - 1].weight, directed[read].weight);
    } else {
      if (write != read) directed[write] = directed[read];
      ++write;
    }
  }
  directed.resize(write);
  std::vector<double>().swap(sigmas);
  std::vector<double>().swap(rhos);

  for (auto& edge : directed) {
    const int head = key_head(edge.key);
    const int tail = key_tail(edge.key);
    edge.direction = head <= tail ? 1u : 0u;
    edge.key = edge_key(std::min(head, tail), std::max(head, tail));
  }
  std::sort(directed.begin(), directed.end(), weighted_edge_less);

  write = 0;
  for (std::size_t pos = 0; pos < directed.size();) {
    const std::uint64_t key = directed[pos].key;
    const int a = key_head(key);
    const int b = key_tail(key);
    float forward = 0.0f;
    float reverse = 0.0f;
    while (pos < directed.size() && directed[pos].key == key) {
      if (directed[pos].direction == 1u) {
        forward = std::max(forward, directed[pos].weight);
      } else {
        reverse = std::max(reverse, directed[pos].weight);
      }
      ++pos;
    }
    const float w = forward + reverse - forward * reverse;
    if (w > 1.0e-6f) {
      directed[write++] = {key, w, 0u};
    }
  }
  directed.resize(write);

  float max_weight = 0.0f;
  for (const auto& edge : directed) {
    max_weight = std::max(max_weight, edge.weight);
  }
  const float min_sample_weight = n_epochs > 0 ?
    max_weight / static_cast<float>(n_epochs) :
    0.0f;

  std::vector<int> row_counts(static_cast<std::size_t>(n), 0);
  for (const auto& edge : directed) {
    if (edge.weight < min_sample_weight) continue;
    const int head = key_head(edge.key);
    const int tail = key_tail(edge.key);
    ++row_counts[static_cast<std::size_t>(head)];
    ++row_counts[static_cast<std::size_t>(tail)];
  }

  std::vector<int> offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    offsets[static_cast<std::size_t>(i + 1)] =
      offsets[static_cast<std::size_t>(i)] + row_counts[static_cast<std::size_t>(i)];
  }
  const int flat_size = offsets[static_cast<std::size_t>(n)];
  std::vector<int> flat_neighbors(static_cast<std::size_t>(flat_size));
  std::vector<float> flat_weights(static_cast<std::size_t>(flat_size));
  std::vector<int> fill = offsets;

  for (const auto& edge : directed) {
    if (edge.weight < min_sample_weight) continue;
    const int head = key_head(edge.key);
    const int tail = key_tail(edge.key);
    int pos = fill[static_cast<std::size_t>(head)]++;
    flat_neighbors[static_cast<std::size_t>(pos)] = tail;
    flat_weights[static_cast<std::size_t>(pos)] = edge.weight;
    pos = fill[static_cast<std::size_t>(tail)]++;
    flat_neighbors[static_cast<std::size_t>(pos)] = head;
    flat_weights[static_cast<std::size_t>(pos)] = edge.weight;
  }
  std::vector<WeightedEdge>().swap(directed);

  int width = 1;
  std::vector<std::pair<int, float>> row;
  for (int i = 0; i < n; ++i) {
    const int begin = offsets[static_cast<std::size_t>(i)];
    const int end = offsets[static_cast<std::size_t>(i + 1)];
    row.clear();
    row.reserve(static_cast<std::size_t>(end - begin));
    for (int pos = begin; pos < end; ++pos) {
      row.push_back({
        flat_neighbors[static_cast<std::size_t>(pos)],
        flat_weights[static_cast<std::size_t>(pos)]
      });
    }
    auto row_less = [](const auto& a, const auto& b) {
      if (a.second == b.second) return a.first < b.first;
      return a.second > b.second;
    };
    if (static_cast<int>(row.size()) > kMaxCudaNeighbors) {
      std::nth_element(
        row.begin(),
        row.begin() + kMaxCudaNeighbors,
        row.end(),
        row_less
      );
      row.resize(kMaxCudaNeighbors);
    }
    std::sort(row.begin(), row.end(), row_less);
    const int row_size = static_cast<int>(row.size());
    row_counts[static_cast<std::size_t>(i)] = row_size;
    width = std::max(width, row_size);
    for (int j = 0; j < row_size; ++j) {
      flat_neighbors[static_cast<std::size_t>(begin + j)] = row[static_cast<std::size_t>(j)].first;
      flat_weights[static_cast<std::size_t>(begin + j)] = row[static_cast<std::size_t>(j)].second;
    }
  }

  neighbors.assign(static_cast<std::size_t>(n) * width, 0);
  weights.assign(static_cast<std::size_t>(n) * width, 0.0f);
  for (int i = 0; i < n; ++i) {
    const int begin = offsets[static_cast<std::size_t>(i)];
    const int row_size = row_counts[static_cast<std::size_t>(i)];
    for (int j = 0; j < width; ++j) {
      const std::size_t out = static_cast<std::size_t>(i) * width + j;
      if (j < row_size) {
        neighbors[out] = flat_neighbors[static_cast<std::size_t>(begin + j)];
        weights[out] = flat_weights[static_cast<std::size_t>(begin + j)];
      } else {
        neighbors[out] = i;
      }
    }
  }
}

void prepare_embedding_neighbors(const IntegerMatrix& indices,
                                 const NumericMatrix& distances,
                                 const int objective,
                                 const int n_epochs,
                                 std::vector<int>& neighbors,
                                 std::vector<float>& weights) {
  if (objective == kObjectiveUmap) {
    prepare_umap_graph_adjacency(indices, distances, n_epochs, neighbors, weights);
  } else {
    prepare_knn(indices, distances, neighbors, weights);
  }
}

std::vector<float> init_to_float_2d(const NumericMatrix& init) {
  const int n = init.nrow();
  std::vector<float> out(static_cast<std::size_t>(n) * 2u);
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i) * 2u] = static_cast<float>(init(i, 0));
    out[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(init(i, 1));
  }
  return out;
}

int knn_index_offset(const IntegerMatrix& indices) {
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
  return (min_idx >= 1 && max_idx <= n) ? 1 : 0;
}

void validate_projection_inputs(const NumericMatrix& reference_layout,
                                const IntegerMatrix& projection_indices,
                                const NumericMatrix& projection_distances) {
  const int n_reference = reference_layout.nrow();
  const int n_components = reference_layout.ncol();
  const int n_query = projection_indices.nrow();
  const int k = projection_indices.ncol();
  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (n_components < 1) Rcpp::stop("reference_layout must have at least one column");
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (k > kMaxCudaProjectionNeighbors) {
    Rcpp::stop("CUDA projection currently supports at most %d neighbors.", kMaxCudaProjectionNeighbors);
  }
  if (projection_distances.nrow() != n_query || projection_distances.ncol() != k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  for (int i = 0; i < n_query; ++i) {
    for (int j = 0; j < k; ++j) {
      const int idx = projection_indices(i, j);
      const double d = projection_distances(i, j);
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
    }
  }
}

NumericVector structure_score_na() {
  return NumericVector::create(
    Rcpp::Named("knn_preservation") = NA_REAL,
    Rcpp::Named("local_trustworthiness") = NA_REAL,
    Rcpp::Named("local_continuity") = NA_REAL,
    Rcpp::Named("structure_score") = NA_REAL,
    Rcpp::Named("embedding_knn_accuracy") = NA_REAL
  );
}

} // namespace

bool embedding_cuda_available_impl() {
  return fastembedr_cuda_available();
}

List standardize_cuda_impl(NumericMatrix data) {
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 2 || p < 1) Rcpp::stop("data must have at least two rows and one column");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  NumericMatrix out(n, p);
  NumericVector center(p);
  NumericVector scale(p);
  const int status = fastembedr_cuda_standardize_matrix(
    data.begin(),
    n,
    p,
    out.begin(),
    center.begin(),
    scale.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA standardization failed: %s", cuda_embedding_error_message());
  }
  return List::create(
    Rcpp::Named("data") = out,
    Rcpp::Named("center") = center,
    Rcpp::Named("scale") = scale
  );
}

NumericMatrix project_embedding_knn_cuda_impl(NumericMatrix reference_layout,
                                              IntegerMatrix projection_indices,
                                              NumericMatrix projection_distances) {
  validate_projection_inputs(reference_layout, projection_indices, projection_distances);
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  NumericMatrix out(projection_indices.nrow(), reference_layout.ncol());
  const int status = fastembedr_cuda_project_embedding(
    reference_layout.begin(),
    projection_indices.begin(),
    projection_distances.begin(),
    reference_layout.nrow(),
    projection_indices.nrow(),
    projection_indices.ncol(),
    reference_layout.ncol(),
    out.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA projection failed: %s", cuda_embedding_error_message());
  }
  return out;
}

NumericMatrix interpolate_landmark_layout_cuda_impl(NumericMatrix landmark_layout,
                                                    IntegerVector landmark_indices,
                                                    IntegerMatrix projection_indices,
                                                    NumericMatrix projection_distances,
                                                    int n) {
  validate_projection_inputs(landmark_layout, projection_indices, projection_distances);
  if (landmark_indices.size() != landmark_layout.nrow()) {
    Rcpp::stop("landmark_indices length must match landmark_layout rows");
  }
  if (n < 1 || projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must equal n");
  }
  for (int i = 0; i < landmark_indices.size(); ++i) {
    if (landmark_indices[i] < 1 || landmark_indices[i] > n) {
      Rcpp::stop("landmark indices out of range");
    }
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  NumericMatrix out(n, landmark_layout.ncol());
  const int status = fastembedr_cuda_interpolate_landmark_layout(
    landmark_layout.begin(),
    landmark_indices.begin(),
    projection_indices.begin(),
    projection_distances.begin(),
    landmark_layout.nrow(),
    n,
    projection_indices.ncol(),
    landmark_layout.ncol(),
    out.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA landmark interpolation failed: %s", cuda_embedding_error_message());
  }
  return out;
}

List landmark_project_interpolate_knn_confidence_cuda_impl(NumericMatrix landmark_data,
                                                           NumericMatrix query_data,
                                                           NumericMatrix landmark_layout,
                                                           IntegerVector landmark_indices,
                                                           int k) {
  const int n_landmarks = landmark_data.nrow();
  const int n = query_data.nrow();
  const int n_features = landmark_data.ncol();
  const int n_components = landmark_layout.ncol();
  if (n_landmarks < 1) Rcpp::stop("landmark_data must have at least one row");
  if (n < 1) Rcpp::stop("query_data must have at least one row");
  if (n_features < 1) Rcpp::stop("landmark_data must have at least one column");
  if (query_data.ncol() != n_features) {
    Rcpp::stop("landmark_data and query_data must have the same number of columns");
  }
  if (landmark_layout.nrow() != n_landmarks) {
    Rcpp::stop("landmark_layout rows must match landmark_data rows");
  }
  if (n_components < 1) Rcpp::stop("landmark_layout must have at least one column");
  if (landmark_indices.size() != n_landmarks) {
    Rcpp::stop("landmark_indices length must match landmark_data rows");
  }
  if (k < 1) Rcpp::stop("k must be positive");
  if (k > n_landmarks) Rcpp::stop("k cannot exceed the number of landmarks");
  if (k > kMaxCudaProjectionNeighbors) {
    Rcpp::stop("CUDA fused landmark projection currently supports at most %d neighbors.", kMaxCudaProjectionNeighbors);
  }
  for (int i = 0; i < landmark_indices.size(); ++i) {
    if (landmark_indices[i] < 1 || landmark_indices[i] > n) {
      Rcpp::stop("landmark indices out of range");
    }
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  NumericMatrix layout(n, n_components);
  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  NumericVector confidence(n);
  const int status = fastembedr_cuda_landmark_project_interpolate_knn_confidence(
    landmark_data.begin(),
    query_data.begin(),
    landmark_layout.begin(),
    landmark_indices.begin(),
    n_landmarks,
    n,
    n_features,
    k,
    n_components,
    layout.begin(),
    indices.begin(),
    distances.begin(),
    confidence.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA fused landmark projection failed: %s", cuda_embedding_error_message());
  }
  return List::create(
    Rcpp::Named("layout") = layout,
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("confidence") = confidence
  );
}

NumericVector knn_structure_score_cuda_impl(NumericMatrix layout,
                                            IntegerMatrix indices,
                                            IntegerVector keep,
                                            int preserve_k,
                                            IntegerVector labels,
                                            int n_label_levels) {
  const int n = layout.nrow();
  const bool compact_indices = indices.nrow() == keep.size();
  if (layout.ncol() != 2) {
    Rcpp::stop("CUDA structure scoring currently supports two-dimensional embeddings.");
  }
  if (indices.nrow() != n && !compact_indices) {
    Rcpp::stop("indices row count must match layout row count or keep length");
  }
  if (preserve_k < 1 || preserve_k > indices.ncol()) Rcpp::stop("invalid preserve_k");
  if (preserve_k > kMaxCudaScoreNeighbors) {
    Rcpp::stop("CUDA structure scoring currently supports at most %d neighbors.", kMaxCudaScoreNeighbors);
  }
  if (labels.size() != 0 && labels.size() != n) Rcpp::stop("labels length must match layout row count");
  if (keep.size() == 0) return structure_score_na();
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  double totals[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const int* label_ptr = labels.size() == n && n_label_levels > 0 ? labels.begin() : nullptr;
  const int status = fastembedr_cuda_knn_structure_score(
    layout.begin(),
    indices.begin(),
    keep.begin(),
    label_ptr,
    n,
    indices.nrow(),
    indices.ncol(),
    preserve_k,
    keep.size(),
    compact_indices ? 1 : 0,
    label_ptr == nullptr ? 0 : n_label_levels,
    totals
  );
  if (status != 0) {
    Rcpp::stop("CUDA structure scoring failed: %s", cuda_embedding_error_message());
  }
  const int scored = static_cast<int>(totals[5]);
  if (scored == 0) return structure_score_na();
  const double preservation = totals[0] / scored;
  const double trustworthiness = totals[1] / scored;
  const double continuity = totals[2] / scored;
  const double structure = (preservation + trustworthiness + continuity) / 3.0;
  const double label_accuracy = totals[4] > 0.0 ? totals[3] / totals[4] : R_NaN;

  return NumericVector::create(
    Rcpp::Named("knn_preservation") = preservation,
    Rcpp::Named("local_trustworthiness") = trustworthiness,
    Rcpp::Named("local_continuity") = continuity,
    Rcpp::Named("structure_score") = structure,
    Rcpp::Named("embedding_knn_accuracy") = label_accuracy
  );
}

double silhouette_score_cuda_impl(NumericMatrix layout,
                                  IntegerVector labels,
                                  int n_label_levels) {
  const int n = layout.nrow();
  if (layout.ncol() != 2) {
    Rcpp::stop("CUDA silhouette scoring currently supports two-dimensional embeddings.");
  }
  if (labels.size() != n) Rcpp::stop("labels length must match layout row count");
  if (n_label_levels < 2 || n_label_levels > kMaxCudaSilhouetteLabels) {
    Rcpp::stop("CUDA silhouette scoring supports between 2 and %d label levels.", kMaxCudaSilhouetteLabels);
  }
  std::vector<int> counts(static_cast<std::size_t>(n_label_levels) + 1u, 0);
  for (int i = 0; i < n; ++i) {
    const int label = labels[i];
    if (label >= 1 && label <= n_label_levels) {
      ++counts[static_cast<std::size_t>(label)];
    }
  }
  int non_empty = 0;
  for (int label = 1; label <= n_label_levels; ++label) {
    if (counts[static_cast<std::size_t>(label)] > 0) ++non_empty;
  }
  if (non_empty < 2) return NA_REAL;
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  double score = NA_REAL;
  const int status = fastembedr_cuda_silhouette_score(
    layout.begin(),
    labels.begin(),
    counts.data(),
    n,
    n_label_levels,
    &score
  );
  if (status != 0) {
    Rcpp::stop("CUDA silhouette scoring failed: %s", cuda_embedding_error_message());
  }
  return score;
}

NumericMatrix rsvd_multiply_cuda_impl(NumericMatrix left,
                                      NumericMatrix right,
                                      bool transpose_left) {
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  if (left.nrow() < 1 || left.ncol() < 1 || right.nrow() < 1 || right.ncol() < 1) {
    Rcpp::stop("CUDA RSVD matrix multiply requires non-empty matrices.");
  }
  if (transpose_left) {
    if (left.nrow() != right.nrow()) {
      Rcpp::stop("CUDA RSVD cross-product received non-conformable matrices.");
    }
  } else if (left.ncol() != right.nrow()) {
    Rcpp::stop("CUDA RSVD matrix multiply received non-conformable matrices.");
  }

  const int out_rows = transpose_left ? left.ncol() : left.nrow();
  NumericMatrix out(out_rows, right.ncol());
  const int status = fastembedr_cuda_matrix_multiply(
    left.begin(),
    right.begin(),
    left.nrow(),
    left.ncol(),
    right.ncol(),
    transpose_left ? 1 : 0,
    out.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA RSVD matrix multiply failed: %s", cuda_embedding_error_message());
  }
  return out;
}

NumericMatrix spectral_knn_init_cuda_impl(IntegerMatrix indices,
                                          NumericMatrix distances,
                                          int n_components,
                                          int spectral_n_iter,
                                          int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (n_components != 2) {
    Rcpp::stop("CUDA spectral initialization currently supports exactly two components.");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA spectral initialization currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  std::vector<float> out(static_cast<std::size_t>(n) * 2u);
  const int status = fastembedr_cuda_spectral_init_from_knn(
    indices.begin(),
    distances.begin(),
    n,
    indices.ncol(),
    spectral_n_iter,
    static_cast<unsigned int>(seed),
    knn_index_offset(indices),
    out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA spectral initialization failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return result;
}

NumericMatrix knn_embed_cuda_impl(IntegerMatrix indices,
                                  NumericMatrix distances,
                                  NumericMatrix init,
                                  std::string objective,
                                  int n_epochs,
                                  int negative_sample_rate,
                                  double learning_rate,
                                  double min_dist,
                                  int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (init.nrow() != indices.nrow() || init.ncol() != 2) {
    Rcpp::stop("CUDA embedding currently requires a two-dimensional initialization.");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA embedding backend currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  const int objective_code = objective_id(objective);
  std::vector<float> init_float = init_to_float_2d(init);
  std::vector<float> out(init_float.size());
  const auto ab = find_ab_params(1.0, min_dist);

  const bool use_native_knn_prep =
    objective_code != kObjectiveUmap || n >= kNativeCudaUmapPrepMinN;
  int status = 0;
  if (use_native_knn_prep) {
    status = fastembedr_cuda_embed_from_knn(
      indices.begin(),
      distances.begin(),
      init_float.data(),
      n,
      indices.ncol(),
      objective_code,
      n_epochs,
      negative_sample_rate,
      static_cast<float>(learning_rate),
      static_cast<float>(ab.first),
      static_cast<float>(ab.second),
      static_cast<unsigned int>(seed),
      knn_index_offset(indices),
      out.data()
    );
  } else {
    std::vector<int> neighbors;
    std::vector<float> weights;
    prepare_embedding_neighbors(indices, distances, objective_code, n_epochs, neighbors, weights);
    const int prepared_k = static_cast<int>(neighbors.size() / static_cast<std::size_t>(n));
    const float max_weight = weights.empty() ? 1.0f :
      std::max(*std::max_element(weights.begin(), weights.end()), 1.0e-6f);
    status = fastembedr_cuda_embed(
      neighbors.data(),
      weights.data(),
      init_float.data(),
      n,
      prepared_k,
      objective_code,
      n_epochs,
      negative_sample_rate,
      static_cast<float>(learning_rate),
      static_cast<float>(ab.first),
      static_cast<float>(ab.second),
      max_weight,
      static_cast<unsigned int>(seed),
      out.data()
    );
  }
  if (status != 0) {
    Rcpp::stop("CUDA embedding failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return result;
}

NumericMatrix knn_umap_cuda_fused_impl(IntegerMatrix indices,
                                       NumericMatrix distances,
                                       int n_epochs,
                                       int negative_sample_rate,
                                       double learning_rate,
                                       double min_dist,
                                       int spectral_n_iter,
                                       int optimizer_mode,
                                       int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA embedding backend currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (optimizer_mode != 0 && optimizer_mode != 1) {
    Rcpp::stop("optimizer_mode must be 0 (deterministic CSR) or 1 (atomic COO).");
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  std::vector<float> out(static_cast<std::size_t>(n) * 2u);
  const auto ab = find_ab_params(1.0, min_dist);
  const int status = fastembedr_cuda_umap_from_knn_spectral(
    indices.begin(),
    distances.begin(),
    n,
    indices.ncol(),
    n_epochs,
    negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
      spectral_n_iter,
      static_cast<unsigned int>(seed),
      knn_index_offset(indices),
      optimizer_mode,
      out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA fused UMAP failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return result;
}

NumericMatrix knn_tsne_exact_cuda_impl(IntegerMatrix indices,
                                       NumericMatrix distances,
                                       NumericMatrix init,
                                       int n_epochs,
                                       double perplexity,
                                       double learning_rate,
                                       int stop_lying_iter,
                                       int mom_switch_iter,
                                       double momentum,
                                       double final_momentum,
                                       double exaggeration_factor,
                                       int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (init.nrow() != indices.nrow() || init.ncol() != 2) {
    Rcpp::stop("CUDA exact t-SNE currently requires a two-dimensional initialization.");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA exact t-SNE currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (perplexity <= 0.0) Rcpp::stop("perplexity must be positive");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (stop_lying_iter < 0 || mom_switch_iter < 0) {
    Rcpp::stop("t-SNE switch iterations must be non-negative.");
  }
  if (momentum < 0.0 || final_momentum < 0.0) {
    Rcpp::stop("t-SNE momentum values must be non-negative.");
  }
  if (exaggeration_factor <= 0.0) {
    Rcpp::stop("t-SNE exaggeration factor must be positive.");
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  std::vector<float> init_float = init_to_float_2d(init);
  std::vector<float> out(init_float.size());
  const int status = fastembedr_cuda_exact_tsne_from_knn(
    indices.begin(),
    distances.begin(),
    init_float.data(),
    n,
    indices.ncol(),
    n_epochs,
    static_cast<float>(perplexity),
    static_cast<float>(learning_rate),
    stop_lying_iter,
    mom_switch_iter,
    static_cast<float>(momentum),
    static_cast<float>(final_momentum),
    static_cast<float>(exaggeration_factor),
    static_cast<unsigned int>(seed),
    knn_index_offset(indices),
    out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA exact t-SNE failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return result;
}
