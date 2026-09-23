#include <Rcpp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef FASTEMBEDR_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

extern "C" {
bool fastembedr_cuda_available();
const char* fastembedr_cuda_embedding_last_error();
int fastembedr_cuda_opentsne_fft_grid_size(int n);
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
int fastembedr_cuda_opentsne_fft_from_knn(const int* indices,
                                          const double* distances,
                                          const float* init,
                                          int has_init,
                                          int n,
                                          int k,
                                          int n_components,
                                          float perplexity,
                                          int early_exaggeration_iter,
                                          int n_iter,
                                          float early_exaggeration,
                                          float exaggeration,
                                          float learning_rate,
                                          int learning_rate_auto,
                                          float initial_momentum,
                                          float final_momentum,
                                          float min_gain,
                                          float max_step_norm,
                                          unsigned int seed,
                                          int index_offset,
                                          float* out);
int fastembedr_cuda_opentsne_fft_from_knn_float(const int* indices,
                                                const float* distances,
                                                const float* init,
                                                int has_init,
                                                int n,
                                                int k,
                                                int n_components,
                                                float perplexity,
                                                int early_exaggeration_iter,
                                                int n_iter,
                                                float early_exaggeration,
                                                float exaggeration,
                                                float learning_rate,
                                                int learning_rate_auto,
                                                float initial_momentum,
                                                float final_momentum,
                                                float min_gain,
                                                float max_step_norm,
                                                unsigned int seed,
                                                int index_offset,
                                                float* out);
int fastembedr_cuda_opentsne_fft_from_device_knn_float(const int* indices,
                                                       const float* distances,
                                                       const float* init,
                                                       int has_init,
                                                       int n,
                                                       int k,
                                                       int n_components,
                                                       float perplexity,
                                                       int early_exaggeration_iter,
                                                       int n_iter,
                                                       float early_exaggeration,
                                                       float exaggeration,
                                                       float learning_rate,
                                                       int learning_rate_auto,
                                                       float initial_momentum,
                                                       float final_momentum,
                                                       float min_gain,
                                                       float max_step_norm,
                                                       unsigned int seed,
                                                       int index_offset,
                                                       float* out);
int fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_double(const int* indices,
                                                                  const float* distances,
                                                                  const double* pca_init_values,
                                                                  int pca_init_p,
                                                                  int n,
                                                                  int k,
                                                                  int n_components,
                                                                  float perplexity,
                                                                  int early_exaggeration_iter,
                                                                  int n_iter,
                                                                  float early_exaggeration,
                                                                  float exaggeration,
                                                                  float learning_rate,
                                                                  int learning_rate_auto,
                                                                  float initial_momentum,
                                                                  float final_momentum,
                                                                  float min_gain,
                                                                  float max_step_norm,
                                                                  unsigned int seed,
                                                                  int index_offset,
                                                                  float* out);
int fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_float(const int* indices,
                                                                 const float* distances,
                                                                 const float* pca_init_values,
                                                                 int pca_init_p,
                                                                 int n,
                                                                 int k,
                                                                 int n_components,
                                                                 float perplexity,
                                                                 int early_exaggeration_iter,
                                                                 int n_iter,
                                                                 float early_exaggeration,
                                                                 float exaggeration,
                                                                 float learning_rate,
                                                                 int learning_rate_auto,
                                                                 float initial_momentum,
                                                                 float final_momentum,
                                                                 float min_gain,
                                                                 float max_step_norm,
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
                               float repulsion_strength,
                                           int spectral_n_iter,
                                           unsigned int seed,
                                           int index_offset,
                                           int optimizer_mode,
                                           float* out);
int fastembedr_cuda_umap_from_knn_spectral_float(const int* indices,
                                                 const float* distances,
                                                 int n,
                                                 int k,
                                                 int n_epochs,
                                                 int negative_sample_rate,
                                                 float learning_rate,
                                                 float a,
                                                 float b,
                                                 float repulsion_strength,
                                                 int spectral_n_iter,
                                                 unsigned int seed,
                                                 int index_offset,
                                                 int optimizer_mode,
                                                 float* out);
int fastembedr_cuda_umap_from_device_knn_spectral_float(const int* device_indices,
                                                        const float* device_distances,
                                                        int n,
                                                        int k,
                                                        int n_epochs,
                                                        int negative_sample_rate,
                                                        float learning_rate,
                                                        float a,
                                                        float b,
                                                        float repulsion_strength,
                                                        int spectral_n_iter,
                                                        unsigned int seed,
                                                        int index_offset,
                                                        int optimizer_mode,
                                                        int binary_graph,
                                                        float* out);
const int* cuda_int_device_ptr_from_external(SEXP ptr, const char* name);
const float* cuda_float_device_ptr_from_external(SEXP ptr, const char* name);
int fastembedr_cuda_umap_graph_dump_from_knn(const int* indices,
                                             const double* distances,
                                             int n,
                                             int k,
                                             int index_offset,
                                             int* out_heads,
                                             int* out_tails,
                                             float* out_weights,
                                             float* out_epochs_per_sample,
                                             int* out_width,
                                             int* out_capacity);
int fastembedr_cuda_umap_optimize_coo(const int* heads,
                                      const int* tails,
                                      const float* weights,
                                      const float* epochs_per_sample,
                                      const float* init,
                                      int n,
                                      int n_edges_capacity,
                                      int n_epochs,
                                      int negative_sample_rate,
                                      float learning_rate,
                                      float a,
                                      float b,
                                      float repulsion_strength,
                                      unsigned int seed,
                                      int optimizer_mode,
                                      float* out);
int fastembedr_cuda_landmark_tsne_from_device_knn(
  const int* device_indices,
  const float* device_distances,
  int index_offset,
  const float* reference_data,
  const float* query_data,
  const float* reference_layout,
  int n_reference,
  int n_query,
  int n_features,
  int k,
  float perplexity,
  int n_iter,
  int early_exaggeration_iter,
  float learning_rate,
  float early_exaggeration,
  float exaggeration,
  float initial_momentum,
  float final_momentum,
  float max_grad_norm,
  float max_step_norm,
  int n_negatives,
  int exact_repulsion_threshold,
  unsigned int seed,
  int affine_neighbors,
  float affine_ridge,
  float max_extrapolation,
  float* out
);
int fastembedr_cuda_transform_tsne_from_host_knn(
  const int* indices,
  const float* distances,
  int index_offset,
  const float* reference_layout,
  const float* initial_layout,
  int n_reference,
  int n_query,
  int k,
  float perplexity,
  int n_iter,
  int early_exaggeration_iter,
  float learning_rate,
  float early_exaggeration,
  float exaggeration,
  float initial_momentum,
  float final_momentum,
  float max_grad_norm,
  float max_step_norm,
  int n_negatives,
  int exact_repulsion_threshold,
  unsigned int seed,
  float* out
);
int fastembedr_cuda_landmark_umap_from_device_knn(
  const int* device_indices,
  const float* device_distances,
  int index_offset,
  const float* reference_data,
  const float* query_data,
  const float* reference_layout,
  const int* landmark_rows,
  const int* query_rows,
  int n_total,
  int n_reference,
  int n_query,
  int n_features,
  int k,
  int n_epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float repulsion_strength,
  unsigned int seed,
  int affine_neighbors,
  float affine_ridge,
  float max_extrapolation,
  float* out
);
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
int fastembedr_cuda_raft_tsvd_pca_init(const double* values,
                                       int n,
                                       int p,
                                       int n_components,
                                       float* out);
int fastembedr_cuda_raft_tsvd_pca_fit(const float* values,
                                      int n,
                                      int p,
                                      int n_components,
                                      int center,
                                      int scale,
                                      float* scores,
                                      float* components,
                                      float* singular_values,
                                      float* center_values,
                                      float* scale_values,
                                      int requested_method,
                                      unsigned int seed,
                                      int oversample,
                                      int power,
                                      int* selected_method);
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

bool cuda_is_float32_s4(SEXP x) {
  if (!Rf_isS4(x)) return false;
  Rcpp::S4 obj(x);
  return obj.is("float32");
}

IntegerMatrix cuda_float32_data_slot(SEXP x) {
  Rcpp::S4 obj(x);
  SEXP data = obj.slot("Data");
  if (TYPEOF(data) != INTSXP || Rf_isNull(Rf_getAttrib(data, R_DimSymbol))) {
    Rcpp::stop("float32 distance object has an invalid payload");
  }
  return IntegerMatrix(data);
}

IntegerVector cuda_float32_payload_slot(SEXP x) {
  Rcpp::S4 obj(x);
  SEXP data = obj.slot("Data");
  if (TYPEOF(data) != INTSXP) {
    Rcpp::stop("float32 object has an invalid payload");
  }
  return IntegerVector(data);
}

float cuda_int_bits_to_float(const int value) {
  float out;
  static_assert(sizeof(out) == sizeof(value), "float32 payload must use 32-bit storage");
  std::memcpy(&out, &value, sizeof(float));
  return out;
}

int cuda_float_to_int_bits(const float value) {
  int out;
  static_assert(sizeof(out) == sizeof(value), "float32 payload must use 32-bit storage");
  std::memcpy(&out, &value, sizeof(float));
  return out;
}

SEXP cuda_float32_matrix(const std::vector<float>& values,
                         const int nrow,
                         const int ncol) {
  IntegerMatrix payload(nrow, ncol);
  int* destination = INTEGER(payload);
  for (std::size_t i = 0; i < values.size(); ++i) {
    destination[i] = cuda_float_to_int_bits(values[i]);
  }
  Rcpp::S4 output("float32");
  output.slot("Data") = payload;
  return output;
}

SEXP cuda_pca_output_matrix(const std::vector<float>& values,
                            const int nrow,
                            const int ncol,
                            const bool float32_output) {
  if (float32_output) return cuda_float32_matrix(values, nrow, ncol);
  NumericMatrix output(nrow, ncol);
  std::copy(values.begin(), values.end(), output.begin());
  return output;
}

std::vector<float> cuda_copy_float32_payload(SEXP distances,
                                             const int expected_n,
                                             const int expected_k) {
  if (!cuda_is_float32_s4(distances)) {
    Rcpp::stop("expected a float::float32 distance matrix");
  }
  IntegerMatrix payload = cuda_float32_data_slot(distances);
  if (payload.nrow() != expected_n || payload.ncol() != expected_k) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  const std::size_t total =
    static_cast<std::size_t>(expected_n) * static_cast<std::size_t>(expected_k);
  std::vector<float> out(total);
  const int* src = INTEGER(payload);
  for (std::size_t i = 0; i < total; ++i) {
    out[i] = cuda_int_bits_to_float(src[i]);
  }
  return out;
}

std::pair<int, int> cuda_matrix_dims(SEXP values, const char* name) {
  if (cuda_is_float32_s4(values)) {
    IntegerMatrix payload = cuda_float32_data_slot(values);
    return std::make_pair(payload.nrow(), payload.ncol());
  }
  if (!Rf_isMatrix(values) || TYPEOF(values) != REALSXP) {
    Rcpp::stop("%s must be a numeric matrix or float::float32 matrix", name);
  }
  NumericMatrix matrix(values);
  return std::make_pair(matrix.nrow(), matrix.ncol());
}

std::vector<float> cuda_copy_matrix_float(SEXP values,
                                          int expected_n,
                                          int expected_p,
                                          const char* name) {
  const std::pair<int, int> dims = cuda_matrix_dims(values, name);
  if (dims.first != expected_n || dims.second != expected_p) {
    Rcpp::stop("%s has incompatible dimensions", name);
  }
  if (cuda_is_float32_s4(values)) {
    return cuda_copy_float32_payload(values, expected_n, expected_p);
  }
  NumericMatrix matrix(values);
  const std::size_t total = static_cast<std::size_t>(expected_n) * expected_p;
  std::vector<float> out(total);
  const double* src = matrix.begin();
  for (std::size_t i = 0; i < total; ++i) out[i] = static_cast<float>(src[i]);
  return out;
}

std::vector<float> cuda_copy_float_vector(SEXP values,
                                          const char* name) {
  if (cuda_is_float32_s4(values)) {
    IntegerVector payload = cuda_float32_payload_slot(values);
    std::vector<float> out(static_cast<std::size_t>(payload.size()));
    const int* src = INTEGER(payload);
    for (R_xlen_t i = 0; i < payload.size(); ++i) {
      out[static_cast<std::size_t>(i)] = cuda_int_bits_to_float(src[i]);
    }
    return out;
  }
  if (TYPEOF(values) != REALSXP) {
    Rcpp::stop("%s must be a numeric vector or float::float32 vector", name);
  }
  NumericVector numeric(values);
  std::vector<float> out(static_cast<std::size_t>(numeric.size()));
  for (R_xlen_t i = 0; i < numeric.size(); ++i) {
    out[static_cast<std::size_t>(i)] = static_cast<float>(numeric[i]);
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

int transform_reference_index_offset(const IntegerMatrix& indices,
                                     int n_reference) {
  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int value : indices) {
    min_index = std::min(min_index, value);
    max_index = std::max(max_index, value);
  }
  if (min_index >= 1 && max_index <= n_reference) return 1;
  if (min_index >= 0 && max_index < n_reference) return 0;
  Rcpp::stop("KNN indices are out of range for `reference_layout`.");
}

std::vector<float> initialize_tsne_transform_cuda(
    const NumericMatrix& reference_layout,
    const IntegerMatrix& indices,
    const NumericMatrix& distances,
    const NumericMatrix& y_init,
    bool init,
    const std::string& initialization,
    int index_offset,
    int seed) {
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  std::vector<float> out(static_cast<std::size_t>(n_query) * 2u, 0.0f);
  if (init) {
    if (y_init.nrow() != n_query || y_init.ncol() != 2) {
      Rcpp::stop("`Y_init` must have one row per query and two columns.");
    }
    for (int i = 0; i < n_query; ++i) {
      out[static_cast<std::size_t>(i) * 2u] = y_init(i, 0);
      out[static_cast<std::size_t>(i) * 2u + 1u] = y_init(i, 1);
    }
    return out;
  }
  if (initialization == "random") {
    std::mt19937 rng(seed == NA_INTEGER ? 5489u : seed);
    std::normal_distribution<float> normal(0.0f, 1.0e-4f);
    for (float& value : out) value = normal(rng);
    return out;
  }
  std::vector<float> values(static_cast<std::size_t>(k));
  for (int i = 0; i < n_query; ++i) {
    for (int dim = 0; dim < 2; ++dim) {
      if (initialization == "weighted") {
        double numerator = 0.0;
        double denominator = std::numeric_limits<double>::min();
        for (int j = 0; j < k; ++j) {
          const int ref = indices(i, j) - index_offset;
          const double weight = 1.0 / (distances(i, j) + 1.0e-6);
          numerator += weight * reference_layout(ref, dim);
          denominator += weight;
        }
        out[static_cast<std::size_t>(i) * 2u + dim] =
          static_cast<float>(numerator / denominator);
        continue;
      }
      for (int j = 0; j < k; ++j) {
        const int ref = indices(i, j) - index_offset;
        values[static_cast<std::size_t>(j)] = reference_layout(ref, dim);
      }
      const int middle = k / 2;
      std::nth_element(values.begin(), values.begin() + middle, values.end());
      float median = values[static_cast<std::size_t>(middle)];
      if ((k & 1) == 0) {
        std::nth_element(
          values.begin(), values.begin() + middle - 1,
          values.begin() + middle
        );
        median = 0.5f * (median + values[static_cast<std::size_t>(middle - 1)]);
      }
      out[static_cast<std::size_t>(i) * 2u + dim] = median;
    }
  }
  return out;
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

NumericMatrix cuda_pca_init_cuda_impl(NumericMatrix data,
                                      int n_components) {
#ifndef FASTEMBEDR_HAS_CUDA
  Rcpp::stop("fastEmbedR was not built with CUDA support.");
#else
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 2 || p < 1) {
    Rcpp::stop("CUDA PCA initialization requires at least two rows and one column.");
  }
  n_components = std::max(1, std::min(n_components, std::min(n, p)));

  std::vector<double> means(p, 0.0);
  for (int j = 0; j < p; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += data(i, j);
    means[j] = s / static_cast<double>(n);
  }

  std::vector<float> h_x(static_cast<std::size_t>(n) * p);
  for (int j = 0; j < p; ++j) {
    const std::size_t col = static_cast<std::size_t>(j) * n;
    for (int i = 0; i < n; ++i) {
      h_x[col + i] = static_cast<float>(data(i, j) - means[j]);
    }
  }

  float* d_x = nullptr;
  float* d_cov = nullptr;
  float* d_scores = nullptr;
  float* d_work = nullptr;
  int* d_info = nullptr;
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  auto cleanup = [&]() {
    if (d_x != nullptr) cudaFree(d_x);
    if (d_cov != nullptr) cudaFree(d_cov);
    if (d_scores != nullptr) cudaFree(d_scores);
    if (d_work != nullptr) cudaFree(d_work);
    if (d_info != nullptr) cudaFree(d_info);
    if (blas != nullptr) cublasDestroy(blas);
    if (solver != nullptr) cusolverDnDestroy(solver);
  };

  const std::size_t x_items = static_cast<std::size_t>(n) * p;
  const std::size_t cov_items = static_cast<std::size_t>(p) * p;
  const std::size_t score_items = static_cast<std::size_t>(n) * n_components;
  try {
    if (cudaMalloc(reinterpret_cast<void**>(&d_x), x_items * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_cov), cov_items * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_scores), score_items * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)) != cudaSuccess) {
      cleanup();
      Rcpp::stop("CUDA allocation failed in PCA initialization.");
    }
    if (cudaMemcpy(d_x, h_x.data(), x_items * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      Rcpp::stop("CUDA upload failed in PCA initialization.");
    }
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
        cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
      cleanup();
      Rcpp::stop("Could not create cuBLAS/cuSOLVER handles for PCA initialization.");
    }

    const float alpha_cov = 1.0f / static_cast<float>(std::max(1, n - 1));
    const float beta_zero = 0.0f;
    if (cublasSgemm(
          blas,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          p,
          p,
          n,
          &alpha_cov,
          d_x,
          n,
          d_x,
          n,
          &beta_zero,
          d_cov,
          p
        ) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      Rcpp::stop("cuBLAS covariance multiply failed in PCA initialization.");
    }

    int lwork = 0;
    if (cusolverDnSsyevd_bufferSize(
          solver,
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER,
          p,
          d_cov,
          p,
          d_scores,
          &lwork
        ) != CUSOLVER_STATUS_SUCCESS || lwork <= 0) {
      cleanup();
      Rcpp::stop("cuSOLVER workspace query failed in PCA initialization.");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_work), static_cast<std::size_t>(lwork) * sizeof(float)) != cudaSuccess) {
      cleanup();
      Rcpp::stop("CUDA workspace allocation failed in PCA initialization.");
    }
    if (cusolverDnSsyevd(
          solver,
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER,
          p,
          d_cov,
          p,
          d_scores,
          d_work,
          lwork,
          d_info
        ) != CUSOLVER_STATUS_SUCCESS) {
      cleanup();
      Rcpp::stop("cuSOLVER eigen decomposition failed in PCA initialization.");
    }
    int info = 0;
    if (cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || info != 0) {
      cleanup();
      Rcpp::stop("cuSOLVER PCA initialization returned info=%d.", info);
    }

    const float alpha_score = 1.0f;
    const float beta_score = 0.0f;
    const float* d_top_vectors = d_cov + static_cast<std::size_t>(p - n_components) * p;
    if (cublasSgemm(
          blas,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          n_components,
          p,
          &alpha_score,
          d_x,
          n,
          d_top_vectors,
          p,
          &beta_score,
          d_scores,
          n
        ) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      Rcpp::stop("cuBLAS score multiply failed in PCA initialization.");
    }

    std::vector<float> h_scores(score_items);
    if (cudaMemcpy(h_scores.data(), d_scores, score_items * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      Rcpp::stop("CUDA score download failed in PCA initialization.");
    }
    cleanup();

    NumericMatrix out(n, n_components);
    for (int j = 0; j < n_components; ++j) {
      const std::size_t col = static_cast<std::size_t>(j) * n;
      for (int i = 0; i < n; ++i) {
        out(i, j) = static_cast<double>(h_scores[col + i]);
      }
    }
    return out;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

NumericMatrix raft_tsvd_init_cuda_impl(NumericMatrix data,
                                       int n_components) {
#ifndef FASTEMBEDR_HAS_RAFT
  Rcpp::stop("fastEmbedR was not built with native RAPIDS RAFT TSVD support.");
#else
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 2 || p < 2) {
    Rcpp::stop("RAPIDS RAFT TSVD initialization requires at least two rows and two columns.");
  }
  n_components = std::max(1, std::min(n_components, std::min(n, p)));

  std::vector<float> h_scores(static_cast<std::size_t>(n) * n_components);
  const int status = fastembedr_cuda_raft_tsvd_pca_init(
    data.begin(), n, p, n_components, h_scores.data()
  );
  if (status != 0) {
    Rcpp::stop("RAPIDS RAFT TSVD initialization failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix out(n, n_components);
  for (int j = 0; j < n_components; ++j) {
    const std::size_t col = static_cast<std::size_t>(j) * n;
    for (int i = 0; i < n; ++i) {
      out(i, j) = static_cast<double>(h_scores[col + i]);
    }
  }
  return out;
#endif
}

List pca_tsvd_cuda_impl(SEXP data,
                        int n_components,
                        bool center,
                        bool scale,
                        int seed,
                        int requested_method,
                        int oversample,
                        int power) {
#ifndef FASTEMBEDR_HAS_RAFT
  Rcpp::stop("fastEmbedR was not built with native RAPIDS RAFT TSVD support.");
#else
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const std::pair<int, int> dims = cuda_matrix_dims(data, "data");
  const int n = dims.first;
  const int p = dims.second;
  const int max_rank = std::min(n - 1, p);
  n_components = std::min(n_components, max_rank);
  if (n < 2 || p < 2 || n_components < 1) {
    Rcpp::stop("RAPIDS RAFT TSVD requires at least two rows, two columns, and one usable component.");
  }

  const bool float32_output = cuda_is_float32_s4(data);
  std::vector<float> input = cuda_copy_matrix_float(data, n, p, "data");
  std::vector<float> scores(static_cast<std::size_t>(n) * n_components);
  std::vector<float> components(static_cast<std::size_t>(n_components) * p);
  std::vector<float> singular_values(n_components);
  std::vector<float> center_values(p);
  std::vector<float> scale_values(p);

  const auto started = std::chrono::steady_clock::now();
  int selected_method = 0;
  const int status = fastembedr_cuda_raft_tsvd_pca_fit(
    input.data(),
    n,
    p,
    n_components,
    center ? 1 : 0,
    scale ? 1 : 0,
    scores.data(),
    components.data(),
    singular_values.data(),
    center_values.data(),
    scale_values.data(),
    requested_method,
    static_cast<unsigned int>(seed),
    oversample,
    power,
    &selected_method
  );
  if (status != 0) {
    Rcpp::stop("Native CUDA PCA failed: %s", cuda_embedding_error_message());
  }

  std::vector<float> loadings(static_cast<std::size_t>(p) * n_components);
  for (int component = 0; component < n_components; ++component) {
    int pivot = 0;
    float largest = -1.0f;
    for (int row = 0; row < p; ++row) {
      const std::size_t source = selected_method == 1 ?
        static_cast<std::size_t>(row) +
          static_cast<std::size_t>(component) * p :
        static_cast<std::size_t>(component) +
          static_cast<std::size_t>(row) * n_components;
      const float value = components[source];
      loadings[
        static_cast<std::size_t>(row) +
        static_cast<std::size_t>(component) * p
      ] = value;
      const float magnitude = std::abs(value);
      if (magnitude > largest) {
        largest = magnitude;
        pivot = row;
      }
    }
    if (loadings[
          static_cast<std::size_t>(pivot) +
          static_cast<std::size_t>(component) * p
        ] < 0.0f) {
      for (int row = 0; row < p; ++row) {
        loadings[
          static_cast<std::size_t>(row) +
          static_cast<std::size_t>(component) * p
        ] *= -1.0f;
      }
      for (int row = 0; row < n; ++row) {
        scores[
          static_cast<std::size_t>(row) +
          static_cast<std::size_t>(component) * n
        ] *= -1.0f;
      }
    }
  }
  const auto finished = std::chrono::steady_clock::now();

  NumericVector center_out(p);
  NumericVector scale_out(p);
  NumericVector singular_out(n_components);
  std::copy(center_values.begin(), center_values.end(), center_out.begin());
  std::copy(scale_values.begin(), scale_values.end(), scale_out.begin());
  std::copy(singular_values.begin(), singular_values.end(), singular_out.begin());

  return List::create(
    Rcpp::Named("scores") = cuda_pca_output_matrix(
      scores, n, n_components, float32_output
    ),
    Rcpp::Named("loadings") = cuda_pca_output_matrix(
      loadings, p, n_components, float32_output
    ),
    Rcpp::Named("singular_values") = singular_out,
    Rcpp::Named("center") = center_out,
    Rcpp::Named("scale") = scale_out,
    Rcpp::Named("backend") = selected_method == 1 ?
      "cuda_native_rsvd" : "cuda_raft_tsvd",
    Rcpp::Named("method") = selected_method == 1 ?
      "rsvd" : "raft_tsvd",
    Rcpp::Named("selection") = requested_method == 0 ?
      "auto" : "explicit",
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("oversample") = selected_method == 1 ?
      oversample : NA_INTEGER,
    Rcpp::Named("power") = selected_method == 1 ? power : NA_INTEGER,
    Rcpp::Named("timing") = NumericVector::create(
      Rcpp::Named("total") =
        std::chrono::duration<double>(finished - started).count()
    )
  );
#endif
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
                                       double repulsion_strength,
                                       int spectral_n_iter,
                                       int seed,
                                       int optimizer_mode) {
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
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
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
      static_cast<float>(repulsion_strength),
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

NumericMatrix knn_umap_cuda_fused_float_impl(IntegerMatrix indices,
                                             SEXP distances,
                                             int n_epochs,
                                             int negative_sample_rate,
                                             double learning_rate,
                                             double min_dist,
                                             double repulsion_strength,
                                             int spectral_n_iter,
                                             int seed,
                                             int optimizer_mode) {
  if (!cuda_is_float32_s4(distances)) {
    Rcpp::stop("CUDA float32 UMAP requires float::float32 KNN distances.");
  }
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (k > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA fused UMAP currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  std::vector<float> distance_float = cuda_copy_float32_payload(distances, n, k);
  std::vector<float> out(static_cast<std::size_t>(n) * 2u);
  const auto ab = find_ab_params(1.0, min_dist);
  const int status = fastembedr_cuda_umap_from_knn_spectral_float(
    indices.begin(),
    distance_float.data(),
    n,
    k,
    n_epochs,
    negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(repulsion_strength),
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

NumericMatrix knn_umap_cuda_fused_gpu_impl(SEXP gpu_knn,
                                           int requested_k,
                                           int n_epochs,
                                           int negative_sample_rate,
                                           double learning_rate,
                                           double min_dist,
                                           double repulsion_strength,
                                           int spectral_n_iter,
                                           int seed,
                                           int optimizer_mode,
                                           bool binary_graph) {
  if (!Rf_isNewList(gpu_knn)) {
    Rcpp::stop("CUDA GPU-resident UMAP requires a GPU KNN list contract.");
  }
  Rcpp::List src(gpu_knn);
  if (!src.containsElementNamed("indices_ptr") ||
      !src.containsElementNamed("distances_ptr")) {
    Rcpp::stop("GPU KNN object is missing CUDA KNN device pointers.");
  }
  const std::string distance_type = src.containsElementNamed("distance_type") ?
    Rcpp::as<std::string>(src["distance_type"]) : "float32";
  if (distance_type != "float32") {
    Rcpp::stop("CUDA GPU-resident UMAP requires float32 KNN distances.");
  }
  const std::string layout = src.containsElementNamed("layout") ?
    Rcpp::as<std::string>(src["layout"]) : "column_major_query_by_k";
  if (layout != "column_major_query_by_k") {
    Rcpp::stop("Unsupported GPU KNN layout for CUDA UMAP.");
  }
  const bool exclude_self = src.containsElementNamed("exclude_self") ?
    Rcpp::as<bool>(src["exclude_self"]) : true;
  if (!exclude_self) {
    Rcpp::stop(
      "CUDA GPU-resident UMAP requires non-self KNN. ",
      "Use a native fastEmbedR GPU KNN object with non-self neighbors or use the host KNN path."
    );
  }
  const int n = src.containsElementNamed("n_query") ?
    Rcpp::as<int>(src["n_query"]) : Rcpp::as<int>(src["n"]);
  const int available_k = Rcpp::as<int>(src["k"]);
  const int k = requested_k > 0 ? requested_k : available_k;
  if (k < 1 || k > available_k) {
    Rcpp::stop("Requested GPU KNN width exceeds the GPU KNN object width.");
  }
  if (k > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA fused UMAP currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int* device_indices = cuda_int_device_ptr_from_external(src["indices_ptr"], "indices_ptr");
  const float* device_distances = cuda_float_device_ptr_from_external(src["distances_ptr"], "distances_ptr");
  const int index_offset = src.containsElementNamed("index_base") ?
    Rcpp::as<int>(src["index_base"]) : 1;
  std::vector<float> out(static_cast<std::size_t>(n) * 2u);
  const auto ab = find_ab_params(1.0, min_dist);
  const int status = fastembedr_cuda_umap_from_device_knn_spectral_float(
    device_indices,
    device_distances,
    n,
    k,
    n_epochs,
    negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(repulsion_strength),
    spectral_n_iter,
    static_cast<unsigned int>(seed),
    index_offset,
    optimizer_mode,
    binary_graph ? 1 : 0,
    out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA GPU-resident UMAP failed: %s", cuda_embedding_error_message());
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

List umap_cuda_graph_dump_impl(IntegerMatrix indices,
                               NumericMatrix distances) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA graph dump currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  const int k = indices.ncol();
  const int width = std::min(kMaxCudaNeighbors, std::max(k, 2 * k));
  const int capacity = n * width;
  std::vector<int> heads(static_cast<std::size_t>(capacity), -1);
  std::vector<int> tails(static_cast<std::size_t>(capacity), -1);
  std::vector<float> weights(static_cast<std::size_t>(capacity), 0.0f);
  std::vector<float> epochs(static_cast<std::size_t>(capacity), 0.0f);
  int reported_width = 0;
  int reported_capacity = 0;
  const int status = fastembedr_cuda_umap_graph_dump_from_knn(
    indices.begin(),
    distances.begin(),
    n,
    k,
    knn_index_offset(indices),
    heads.data(),
    tails.data(),
    weights.data(),
    epochs.data(),
    &reported_width,
    &reported_capacity
  );
  if (status != 0) {
    Rcpp::stop("CUDA UMAP graph dump failed: %s", cuda_embedding_error_message());
  }
  if (reported_capacity < 0 || reported_capacity > capacity) {
    Rcpp::stop("CUDA UMAP graph dump returned an invalid capacity");
  }

  IntegerVector r_heads(reported_capacity);
  IntegerVector r_tails(reported_capacity);
  NumericVector r_weights(reported_capacity);
  NumericVector r_epochs(reported_capacity);
  for (int i = 0; i < reported_capacity; ++i) {
    r_heads[i] = heads[static_cast<std::size_t>(i)];
    r_tails[i] = tails[static_cast<std::size_t>(i)];
    r_weights[i] = static_cast<double>(weights[static_cast<std::size_t>(i)]);
    r_epochs[i] = static_cast<double>(epochs[static_cast<std::size_t>(i)]);
  }
  return List::create(
    Rcpp::Named("heads") = r_heads,
    Rcpp::Named("tails") = r_tails,
    Rcpp::Named("weights") = r_weights,
    Rcpp::Named("epochs_per_sample") = r_epochs,
    Rcpp::Named("width") = reported_width,
    Rcpp::Named("capacity") = reported_capacity
  );
}

NumericMatrix umap_cuda_optimize_coo_impl(IntegerVector heads,
                                          IntegerVector tails,
                                          SEXP weights,
                                          SEXP epochs_per_sample,
                                          NumericMatrix init,
                                          int n_epochs,
                                          int negative_sample_rate,
                                          double learning_rate,
                                          double min_dist,
                                          double repulsion_strength,
                                          int seed,
                                          int optimizer_mode) {
  std::vector<float> w = cuda_copy_float_vector(weights, "COO weights");
  std::vector<float> eps = cuda_copy_float_vector(epochs_per_sample, "COO epochs_per_sample");
  if (heads.size() != tails.size() ||
      static_cast<std::size_t>(heads.size()) != w.size() ||
      static_cast<std::size_t>(heads.size()) != eps.size()) {
    Rcpp::stop("COO heads, tails, weights, and epochs_per_sample must have the same length");
  }
  if (init.ncol() != 2) Rcpp::stop("CUDA COO UMAP optimizer requires a two-dimensional initialization");
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = init.nrow();
  const int n_edges = heads.size();
  std::vector<float> init_float = init_to_float_2d(init);
  std::vector<float> out(init_float.size());
  const auto ab = find_ab_params(1.0, min_dist);
  const int status = fastembedr_cuda_umap_optimize_coo(
    heads.begin(),
    tails.begin(),
    w.data(),
    eps.data(),
    init_float.data(),
    n,
    n_edges,
    n_epochs,
    negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(repulsion_strength),
    static_cast<unsigned int>(seed),
    optimizer_mode,
    out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA COO UMAP optimizer failed: %s", cuda_embedding_error_message());
  }
  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return result;
}

List knn_tsne_opentsne_cuda_impl(IntegerMatrix indices,
                                 NumericMatrix distances,
                                 NumericMatrix y_init,
                                 bool init,
                                 int n_components,
                                 double perplexity,
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
                                 int seed,
                                 bool record_costs) {
  (void)record_costs;
  std::transform(
    negative_gradient_method.begin(),
    negative_gradient_method.end(),
    negative_gradient_method.begin(),
    [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
  );
  if (negative_gradient_method == "fft" ||
      negative_gradient_method == "fitsne" ||
      negative_gradient_method == "fit_sne" ||
      negative_gradient_method == "interpolation" ||
      negative_gradient_method == "auto") {
    if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
      Rcpp::stop("indices and distances must have the same dimensions");
    }
    if (n_components != 2) {
      Rcpp::stop("CUDA openTSNE FFT-grid currently supports exactly two output components.");
    }
    if (init && (y_init.nrow() != indices.nrow() || y_init.ncol() != 2)) {
      Rcpp::stop("CUDA openTSNE FFT-grid requires a two-dimensional initialization.");
    }
    if (indices.ncol() > kMaxCudaNeighbors) {
      Rcpp::stop("CUDA openTSNE FFT-grid currently supports at most %d neighbors.", kMaxCudaNeighbors);
    }
    if (perplexity <= 0.0) Rcpp::stop("perplexity must be positive");
    if (early_exaggeration_iter < 0 || n_iter < 0 || early_exaggeration_iter + n_iter < 1) {
      Rcpp::stop("CUDA openTSNE FFT-grid requires at least one optimization iteration.");
    }
    if (early_exaggeration <= 0.0 || exaggeration <= 0.0) {
      Rcpp::stop("CUDA openTSNE FFT-grid exaggeration values must be positive.");
    }
    if (learning_rate <= 0.0 && !learning_rate_auto) {
      Rcpp::stop("CUDA openTSNE FFT-grid learning rate must be positive.");
    }
    if (initial_momentum < 0.0 || final_momentum < 0.0) {
      Rcpp::stop("CUDA openTSNE FFT-grid momentum values must be non-negative.");
    }
    if (min_gain <= 0.0) Rcpp::stop("CUDA openTSNE FFT-grid min_gain must be positive.");
    if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

    const int n = indices.nrow();
    std::vector<float> init_float;
    if (init) {
      init_float = init_to_float_2d(y_init);
    } else {
      init_float.assign(static_cast<std::size_t>(n) * 2u, 0.0f);
    }
    std::vector<float> out(init_float.size());
    const int status = fastembedr_cuda_opentsne_fft_from_knn(
      indices.begin(),
      distances.begin(),
      init_float.data(),
      init ? 1 : 0,
      n,
      indices.ncol(),
      n_components,
      static_cast<float>(perplexity),
      early_exaggeration_iter,
      n_iter,
      static_cast<float>(early_exaggeration),
      static_cast<float>(exaggeration),
      static_cast<float>(learning_rate),
      learning_rate_auto ? 1 : 0,
      static_cast<float>(initial_momentum),
      static_cast<float>(final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm),
      static_cast<unsigned int>(seed),
      knn_index_offset(indices),
      out.data()
    );
    if (status != 0) {
      Rcpp::stop("CUDA openTSNE FFT-grid failed: %s", cuda_embedding_error_message());
    }

    NumericMatrix result(n, 2);
    for (int i = 0; i < n; ++i) {
      result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
      result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
    }
    return List::create(
      Rcpp::Named("Y") = result,
      Rcpp::Named("costs") = NumericVector::create(),
      Rcpp::Named("itercosts") = NumericVector::create(),
      Rcpp::Named("itercost_iterations") = IntegerVector::create(),
      Rcpp::Named("optimizer") = "opentsne_fitsne_fft_grid_native_cuda",
      Rcpp::Named("repulsion") = "fft_grid_cuda_cufft",
      Rcpp::Named("fft_grid_size") = fastembedr_cuda_opentsne_fft_grid_size(n),
      Rcpp::Named("probabilities") = "symmetric_sparse_knn_cuda",
      Rcpp::Named("n_negatives") = NA_INTEGER,
      Rcpp::Named("n_threads") = NA_INTEGER,
      Rcpp::Named("learning_rate_early") = learning_rate_auto ?
        static_cast<double>(n) / static_cast<double>(early_exaggeration) :
        static_cast<double>(learning_rate),
      Rcpp::Named("learning_rate_normal") = learning_rate_auto ?
        static_cast<double>(n) / static_cast<double>(std::max(exaggeration, 1.0)) :
        static_cast<double>(learning_rate),
      Rcpp::Named("early_exaggeration_iter_actual") = early_exaggeration_iter,
      Rcpp::Named("n_iter_actual") = n_iter,
      Rcpp::Named("max_iter_actual") = early_exaggeration_iter + n_iter,
      Rcpp::Named("auto_kld_stop") = false,
      Rcpp::Named("auto_stop_reason") = "cuda_fft_no_host_kld_monitor",
      Rcpp::Named("auto_iter_end") = NA_REAL
    );
  }
  Rcpp::stop(
    "CUDA openTSNE supports `negative_gradient_method = \"fft\"` only. "
    "Use the FFT/FIt-SNE path for CUDA; exact CUDA t-SNE is kept separate and is not labelled as openTSNE."
  );
}

List knn_tsne_opentsne_cuda_float_impl(IntegerMatrix indices,
                                       SEXP distances,
                                       NumericMatrix y_init,
                                       bool init,
                                       int n_components,
                                       double perplexity,
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
                                       int seed,
                                       bool record_costs) {
  (void)record_costs;
  std::transform(
    negative_gradient_method.begin(),
    negative_gradient_method.end(),
    negative_gradient_method.begin(),
    [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
  );
  if (!(negative_gradient_method == "fft" ||
        negative_gradient_method == "fitsne" ||
        negative_gradient_method == "fit_sne" ||
        negative_gradient_method == "interpolation" ||
        negative_gradient_method == "auto")) {
    Rcpp::stop(
      "CUDA openTSNE supports `negative_gradient_method = \"fft\"` only. "
      "Use the FFT/FIt-SNE path for CUDA; exact CUDA t-SNE is kept separate and is not labelled as openTSNE."
    );
  }
  if (!cuda_is_float32_s4(distances)) {
    Rcpp::stop("CUDA float32 openTSNE requires float::float32 KNN distances.");
  }
  if (n_components != 2) {
    Rcpp::stop("CUDA openTSNE FFT-grid currently supports exactly two output components.");
  }
  if (init && (y_init.nrow() != indices.nrow() || y_init.ncol() != 2)) {
    Rcpp::stop("CUDA openTSNE FFT-grid requires a two-dimensional initialization.");
  }
  if (indices.ncol() > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA openTSNE FFT-grid currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (perplexity <= 0.0) Rcpp::stop("perplexity must be positive");
  if (early_exaggeration_iter < 0 || n_iter < 0 || early_exaggeration_iter + n_iter < 1) {
    Rcpp::stop("CUDA openTSNE FFT-grid requires at least one optimization iteration.");
  }
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0) {
    Rcpp::stop("CUDA openTSNE FFT-grid exaggeration values must be positive.");
  }
  if (learning_rate <= 0.0 && !learning_rate_auto) {
    Rcpp::stop("CUDA openTSNE FFT-grid learning rate must be positive.");
  }
  if (initial_momentum < 0.0 || final_momentum < 0.0) {
    Rcpp::stop("CUDA openTSNE FFT-grid momentum values must be non-negative.");
  }
  if (min_gain <= 0.0) Rcpp::stop("CUDA openTSNE FFT-grid min_gain must be positive.");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n = indices.nrow();
  const int k = indices.ncol();
  std::vector<float> distance_float = cuda_copy_float32_payload(distances, n, k);
  std::vector<float> init_float;
  if (init) {
    init_float = init_to_float_2d(y_init);
  } else {
    init_float.assign(static_cast<std::size_t>(n) * 2u, 0.0f);
  }
  std::vector<float> out(init_float.size());
  const int status = fastembedr_cuda_opentsne_fft_from_knn_float(
    indices.begin(),
    distance_float.data(),
    init_float.data(),
    init ? 1 : 0,
    n,
    k,
    n_components,
    static_cast<float>(perplexity),
    early_exaggeration_iter,
    n_iter,
    static_cast<float>(early_exaggeration),
    static_cast<float>(exaggeration),
    static_cast<float>(learning_rate),
    learning_rate_auto ? 1 : 0,
    static_cast<float>(initial_momentum),
    static_cast<float>(final_momentum),
    static_cast<float>(min_gain),
    static_cast<float>(max_step_norm),
    static_cast<unsigned int>(seed),
    knn_index_offset(indices),
    out.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA openTSNE FFT-grid failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  return List::create(
    Rcpp::Named("Y") = result,
    Rcpp::Named("costs") = NumericVector::create(),
    Rcpp::Named("itercosts") = NumericVector::create(),
    Rcpp::Named("itercost_iterations") = IntegerVector::create(),
    Rcpp::Named("optimizer") = "opentsne_fitsne_fft_grid_native_cuda",
    Rcpp::Named("repulsion") = "fft_grid_cuda_cufft",
    Rcpp::Named("fft_grid_size") = fastembedr_cuda_opentsne_fft_grid_size(n),
    Rcpp::Named("probabilities") = "symmetric_sparse_knn_cuda_float32",
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("n_negatives") = NA_INTEGER,
    Rcpp::Named("n_threads") = NA_INTEGER,
    Rcpp::Named("learning_rate_early") = learning_rate_auto ?
      static_cast<double>(n) / static_cast<double>(early_exaggeration) :
      static_cast<double>(learning_rate),
    Rcpp::Named("learning_rate_normal") = learning_rate_auto ?
      static_cast<double>(n) / static_cast<double>(std::max(exaggeration, 1.0)) :
      static_cast<double>(learning_rate),
    Rcpp::Named("early_exaggeration_iter_actual") = early_exaggeration_iter,
    Rcpp::Named("n_iter_actual") = n_iter,
    Rcpp::Named("max_iter_actual") = early_exaggeration_iter + n_iter,
    Rcpp::Named("auto_kld_stop") = false,
    Rcpp::Named("auto_stop_reason") = "cuda_fft_no_host_kld_monitor",
    Rcpp::Named("auto_iter_end") = NA_REAL
  );
}

const int* cuda_int_device_ptr_from_external(SEXP ptr, const char* name) {
  if (TYPEOF(ptr) != EXTPTRSXP) {
    Rcpp::stop("%s must be an external pointer", name);
  }
  void* address = R_ExternalPtrAddr(ptr);
  if (address == nullptr) {
    Rcpp::stop("%s points to a released CUDA allocation", name);
  }
  return static_cast<const int*>(address);
}

const float* cuda_float_device_ptr_from_external(SEXP ptr, const char* name) {
  if (TYPEOF(ptr) != EXTPTRSXP) {
    Rcpp::stop("%s must be an external pointer", name);
  }
  void* address = R_ExternalPtrAddr(ptr);
  if (address == nullptr) {
    Rcpp::stop("%s points to a released CUDA allocation", name);
  }
  return static_cast<const float*>(address);
}

List knn_tsne_opentsne_cuda_gpu_impl(SEXP gpu_knn,
                                     int requested_k,
                                     NumericMatrix y_init,
                                     bool init,
                                     SEXP pca_init_data,
                                     int n_components,
                                     double perplexity,
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
                                     int seed,
                                     bool record_costs) {
  (void)record_costs;
  if (!Rf_isNewList(gpu_knn)) {
    Rcpp::stop("CUDA GPU-resident openTSNE requires a GPU KNN list contract.");
  }
  Rcpp::List src(gpu_knn);
  std::transform(
    negative_gradient_method.begin(),
    negative_gradient_method.end(),
    negative_gradient_method.begin(),
    [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
  );
  if (!(negative_gradient_method == "fft" ||
        negative_gradient_method == "fitsne" ||
        negative_gradient_method == "fit_sne" ||
        negative_gradient_method == "interpolation" ||
        negative_gradient_method == "auto")) {
    Rcpp::stop(
      "CUDA openTSNE supports `negative_gradient_method = \"fft\"` only. "
      "Use the FFT/FIt-SNE path for CUDA; exact CUDA t-SNE is kept separate and is not labelled as openTSNE."
    );
  }
  if (!src.containsElementNamed("indices_ptr") ||
      !src.containsElementNamed("distances_ptr")) {
    Rcpp::stop("GPU KNN object is missing CUDA KNN device pointers.");
  }
  const std::string distance_type = src.containsElementNamed("distance_type") ?
    Rcpp::as<std::string>(src["distance_type"]) : "float32";
  if (distance_type != "float32") {
    Rcpp::stop("CUDA GPU-resident openTSNE requires float32 KNN distances.");
  }
  const std::string layout = src.containsElementNamed("layout") ?
    Rcpp::as<std::string>(src["layout"]) : "column_major_query_by_k";
  if (layout != "column_major_query_by_k") {
    Rcpp::stop("Unsupported GPU KNN layout for CUDA openTSNE.");
  }
  const bool exclude_self = src.containsElementNamed("exclude_self") ?
    Rcpp::as<bool>(src["exclude_self"]) : true;
  if (!exclude_self) {
    Rcpp::stop(
      "CUDA GPU-resident openTSNE requires non-self KNN. ",
      "Create the GPU KNN with `exclude_self = TRUE` or use the host KNN path."
    );
  }
  const int n = src.containsElementNamed("n_query") ?
    Rcpp::as<int>(src["n_query"]) : Rcpp::as<int>(src["n"]);
  const int available_k = Rcpp::as<int>(src["k"]);
  const int k = requested_k > 0 ? requested_k : available_k;
  if (k < 1 || k > available_k) {
    Rcpp::stop("Requested GPU KNN width exceeds the GPU KNN object width.");
  }
  if (n_components != 2) {
    Rcpp::stop("CUDA openTSNE FFT-grid currently supports exactly two output components.");
  }
  if (init && (y_init.nrow() != n || y_init.ncol() != 2)) {
    Rcpp::stop("CUDA openTSNE FFT-grid requires a two-dimensional initialization.");
  }
  if (k > kMaxCudaNeighbors) {
    Rcpp::stop("CUDA openTSNE FFT-grid currently supports at most %d neighbors.", kMaxCudaNeighbors);
  }
  if (perplexity <= 0.0) Rcpp::stop("perplexity must be positive");
  if (early_exaggeration_iter < 0 || n_iter < 0 || early_exaggeration_iter + n_iter < 1) {
    Rcpp::stop("CUDA openTSNE FFT-grid requires at least one optimization iteration.");
  }
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0) {
    Rcpp::stop("CUDA openTSNE FFT-grid exaggeration values must be positive.");
  }
  if (learning_rate <= 0.0 && !learning_rate_auto) {
    Rcpp::stop("CUDA openTSNE FFT-grid learning rate must be positive.");
  }
  if (initial_momentum < 0.0 || final_momentum < 0.0) {
    Rcpp::stop("CUDA openTSNE FFT-grid momentum values must be non-negative.");
  }
  if (min_gain <= 0.0) Rcpp::stop("CUDA openTSNE FFT-grid min_gain must be positive.");
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int* device_indices = cuda_int_device_ptr_from_external(src["indices_ptr"], "indices_ptr");
  const float* device_distances = cuda_float_device_ptr_from_external(src["distances_ptr"], "distances_ptr");
  const bool use_device_pca = !init && pca_init_data != R_NilValue;
  std::vector<float> init_float;
  std::vector<float> pca_float;
  NumericMatrix pca_double;
  const double* pca_double_ptr = nullptr;
  const float* pca_float_ptr = nullptr;
  int pca_p = 0;
  if (init) {
    init_float = init_to_float_2d(y_init);
  } else if (use_device_pca) {
    if (cuda_is_float32_s4(pca_init_data)) {
      IntegerMatrix payload = cuda_float32_data_slot(pca_init_data);
      if (payload.nrow() != n || payload.ncol() < n_components) {
        Rcpp::stop("CUDA device PCA initialization data has incompatible dimensions.");
      }
      pca_p = payload.ncol();
      pca_float = cuda_copy_float32_payload(pca_init_data, n, pca_p);
      pca_float_ptr = pca_float.data();
    } else {
      pca_double = Rcpp::as<NumericMatrix>(pca_init_data);
      if (pca_double.nrow() != n || pca_double.ncol() < n_components) {
        Rcpp::stop("CUDA device PCA initialization data has incompatible dimensions.");
      }
      pca_p = pca_double.ncol();
      pca_double_ptr = pca_double.begin();
    }
  } else {
    init_float.assign(static_cast<std::size_t>(n) * 2u, 0.0f);
  }
  std::vector<float> out(init_float.size());
  if (out.empty()) out.assign(static_cast<std::size_t>(n) * 2u, 0.0f);
  const int index_offset = src.containsElementNamed("index_base") ?
    Rcpp::as<int>(src["index_base"]) : 1;
  int status = 0;
  if (pca_double_ptr != nullptr) {
    status = fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_double(
      device_indices,
      device_distances,
      pca_double_ptr,
      pca_p,
      n,
      k,
      n_components,
      static_cast<float>(perplexity),
      early_exaggeration_iter,
      n_iter,
      static_cast<float>(early_exaggeration),
      static_cast<float>(exaggeration),
      static_cast<float>(learning_rate),
      learning_rate_auto ? 1 : 0,
      static_cast<float>(initial_momentum),
      static_cast<float>(final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm),
      static_cast<unsigned int>(seed),
      index_offset,
      out.data()
    );
  } else if (pca_float_ptr != nullptr) {
    status = fastembedr_cuda_opentsne_fft_from_device_knn_float_pca_float(
      device_indices,
      device_distances,
      pca_float_ptr,
      pca_p,
      n,
      k,
      n_components,
      static_cast<float>(perplexity),
      early_exaggeration_iter,
      n_iter,
      static_cast<float>(early_exaggeration),
      static_cast<float>(exaggeration),
      static_cast<float>(learning_rate),
      learning_rate_auto ? 1 : 0,
      static_cast<float>(initial_momentum),
      static_cast<float>(final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm),
      static_cast<unsigned int>(seed),
      index_offset,
      out.data()
    );
  } else {
    status = fastembedr_cuda_opentsne_fft_from_device_knn_float(
      device_indices,
      device_distances,
      init_float.data(),
      init ? 1 : 0,
      n,
      k,
      n_components,
      static_cast<float>(perplexity),
      early_exaggeration_iter,
      n_iter,
      static_cast<float>(early_exaggeration),
      static_cast<float>(exaggeration),
      static_cast<float>(learning_rate),
      learning_rate_auto ? 1 : 0,
      static_cast<float>(initial_momentum),
      static_cast<float>(final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm),
      static_cast<unsigned int>(seed),
      index_offset,
      out.data()
    );
  }
  if (status != 0) {
    Rcpp::stop("CUDA openTSNE FFT-grid failed: %s", cuda_embedding_error_message());
  }

  NumericMatrix result(n, 2);
  for (int i = 0; i < n; ++i) {
    result(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u]);
    result(i, 1) = static_cast<double>(out[static_cast<std::size_t>(i) * 2u + 1u]);
  }
  const int pca_rank_limit = use_device_pca ?
    std::min(n - 1, pca_p) : 0;
  const int pca_sketch = use_device_pca ?
    std::min(pca_rank_limit, n_components + 16) : 0;
  const std::size_t pca_crossover =
    12u * 6u * static_cast<std::size_t>(pca_sketch);
  const bool pca_selected_rsvd = use_device_pca &&
    pca_sketch < pca_rank_limit &&
    static_cast<std::size_t>(n) * pca_p >= 250000u &&
    static_cast<std::size_t>(pca_p) >= pca_crossover;
  return List::create(
    Rcpp::Named("Y") = result,
    Rcpp::Named("costs") = NumericVector::create(),
    Rcpp::Named("itercosts") = NumericVector::create(),
    Rcpp::Named("itercost_iterations") = IntegerVector::create(),
    Rcpp::Named("optimizer") = "opentsne_fitsne_fft_grid_native_cuda_gpu_knn",
    Rcpp::Named("repulsion") = "fft_grid_cuda_cufft",
    Rcpp::Named("fft_grid_size") = fastembedr_cuda_opentsne_fft_grid_size(n),
    Rcpp::Named("probabilities") = "symmetric_sparse_knn_cuda_float32_gpu_resident",
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("n_negatives") = NA_INTEGER,
    Rcpp::Named("n_threads") = NA_INTEGER,
    Rcpp::Named("learning_rate_early") = learning_rate_auto ?
      static_cast<double>(n) / static_cast<double>(early_exaggeration) :
      static_cast<double>(learning_rate),
    Rcpp::Named("learning_rate_normal") = learning_rate_auto ?
      static_cast<double>(n) / static_cast<double>(std::max(exaggeration, 1.0)) :
      static_cast<double>(learning_rate),
    Rcpp::Named("early_exaggeration_iter_actual") = early_exaggeration_iter,
    Rcpp::Named("n_iter_actual") = n_iter,
    Rcpp::Named("max_iter_actual") = early_exaggeration_iter + n_iter,
    Rcpp::Named("initialization") = use_device_pca ?
      (pca_selected_rsvd ?
        "pca_cuda_native_rsvd_device" :
        "pca_cuda_raft_tsvd_device") :
      (init ? "host_supplied" : "cuda_random"),
    Rcpp::Named("init_residency") = use_device_pca ? "cuda_device" : (init ? "host_to_device" : "cuda_device"),
    Rcpp::Named("auto_kld_stop") = false,
    Rcpp::Named("auto_stop_reason") = "not_available_cuda_gpu_resident_fft",
    Rcpp::Named("auto_iter_end") = early_exaggeration_iter + n_iter,
    Rcpp::Named("knn_residency") = "cuda_device"
  );
}

List transform_tsne_cuda_impl(NumericMatrix reference_layout,
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
                              int seed) {
  const int n_reference = reference_layout.nrow();
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  if (n_reference < 1 || reference_layout.ncol() != 2 ||
      n_query < 1 || k < 1 || distances.nrow() != n_query ||
      distances.ncol() != k) {
    Rcpp::stop("CUDA t-SNE transform inputs have incompatible dimensions.");
  }
  if (k > kMaxCudaProjectionNeighbors || perplexity <= 0.0 ||
      n_iter < 0 || early_exaggeration_iter < 0 ||
      n_iter + early_exaggeration_iter < 1 || learning_rate <= 0.0 ||
      early_exaggeration <= 0.0 || exaggeration <= 0.0 ||
      initial_momentum < 0.0 || final_momentum < 0.0) {
    Rcpp::stop("Invalid CUDA t-SNE transform parameters.");
  }
  if (initialization != "median" && initialization != "weighted" &&
      initialization != "random") {
    Rcpp::stop("`initialization` must be 'median', 'weighted', or 'random'.");
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const int index_offset = transform_reference_index_offset(
    indices, n_reference
  );
  std::vector<float> distance_values(
    static_cast<std::size_t>(n_query) * k
  );
  for (R_xlen_t i = 0; i < distances.length(); ++i) {
    const double value = distances[i];
    if (!std::isfinite(value) || value < 0.0) {
      Rcpp::stop("KNN distances must be finite and non-negative.");
    }
    distance_values[static_cast<std::size_t>(i)] = value;
  }
  std::vector<float> reference = cuda_copy_matrix_float(
    reference_layout, n_reference, 2, "reference_layout"
  );
  std::vector<float> initial = initialize_tsne_transform_cuda(
    reference_layout, indices, distances, y_init, init, initialization,
    index_offset, seed
  );
  std::vector<float> output(static_cast<std::size_t>(n_query) * 2u);
  n_negatives = std::max(1, std::min(n_negatives, n_reference));
  exact_repulsion_threshold = std::max(1, exact_repulsion_threshold);
  const int status = fastembedr_cuda_transform_tsne_from_host_knn(
    indices.begin(), distance_values.data(), index_offset, reference.data(),
    initial.data(), n_reference, n_query, k, perplexity, n_iter,
    early_exaggeration_iter, learning_rate, early_exaggeration, exaggeration,
    initial_momentum, final_momentum, max_grad_norm, max_step_norm,
    n_negatives, exact_repulsion_threshold,
    static_cast<unsigned int>(seed == NA_INTEGER ? 5489 : seed),
    output.data()
  );
  if (status != 0) {
    Rcpp::stop(
      "CUDA t-SNE transform failed: %s", cuda_embedding_error_message()
    );
  }
  NumericMatrix layout(n_query, 2);
  for (int row = 0; row < n_query; ++row) {
    layout(row, 0) = output[static_cast<std::size_t>(row) * 2u];
    layout(row, 1) = output[static_cast<std::size_t>(row) * 2u + 1u];
  }
  const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
    n_negatives >= n_reference;
  return List::create(
    Rcpp::Named("Y") = layout,
    Rcpp::Named("optimizer") =
      "opentsne_style_fixed_reference_transform_cuda",
    Rcpp::Named("initialization") = initialization,
    Rcpp::Named("repulsion") = exact_repulsion ?
      "exact_reference_cuda" : "sampled_reference_cuda",
    Rcpp::Named("affinities") = "precomputed_query_conditional_cuda_float32",
    Rcpp::Named("affinity_storage") = "cuda_device_rank_major_float32",
    Rcpp::Named("transform_batch_size") = n_query,
    Rcpp::Named("transform_batches") = 1,
    Rcpp::Named("n_negatives") = n_negatives,
    Rcpp::Named("n_threads") = NA_INTEGER
  );
}

List landmark_tsne_transform_cuda_gpu_impl(SEXP gpu_knn,
                                           SEXP reference_data,
                                           SEXP query_data,
                                           SEXP reference_layout,
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
                                           int seed,
                                           int affine_neighbors,
                                           double affine_ridge,
                                           double max_extrapolation) {
  if (!Rf_isNewList(gpu_knn)) {
    Rcpp::stop("CUDA landmark t-SNE requires a GPU-resident KNN object.");
  }
  List src(gpu_knn);
  const int n_query = src.containsElementNamed("n_query") ?
    Rcpp::as<int>(src["n_query"]) : Rcpp::as<int>(src["n"]);
  const int k = Rcpp::as<int>(src["k"]);
  const int index_offset = src.containsElementNamed("index_base") ?
    Rcpp::as<int>(src["index_base"]) : 1;
  const std::pair<int, int> reference_dims = cuda_matrix_dims(reference_data, "reference_data");
  const std::pair<int, int> query_dims = cuda_matrix_dims(query_data, "query_data");
  const std::pair<int, int> layout_dims = cuda_matrix_dims(reference_layout, "reference_layout");
  const int n_reference = reference_dims.first;
  const int n_features = reference_dims.second;
  if (query_dims.first != n_query || query_dims.second != n_features) {
    Rcpp::stop("query_data dimensions do not match the GPU KNN object and reference_data.");
  }
  if (layout_dims.first != n_reference || layout_dims.second != 2) {
    Rcpp::stop("CUDA landmark t-SNE requires a two-dimensional reference layout.");
  }
  if (k < 1 || k > kMaxCudaProjectionNeighbors) {
    Rcpp::stop("CUDA landmark t-SNE supports at most %d projection neighbors.", kMaxCudaProjectionNeighbors);
  }
  if (perplexity <= 0.0 || n_iter < 0 || early_exaggeration_iter < 0 ||
      learning_rate <= 0.0 || early_exaggeration <= 0.0 || exaggeration <= 0.0) {
    Rcpp::stop("Invalid CUDA landmark t-SNE optimization parameters.");
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const int* device_indices = cuda_int_device_ptr_from_external(src["indices_ptr"], "indices_ptr");
  const float* device_distances = cuda_float_device_ptr_from_external(src["distances_ptr"], "distances_ptr");
  std::vector<float> reference_values = cuda_copy_matrix_float(
    reference_data, n_reference, n_features, "reference_data"
  );
  std::vector<float> query_values = cuda_copy_matrix_float(
    query_data, n_query, n_features, "query_data"
  );
  std::vector<float> reference_layout_values = cuda_copy_matrix_float(
    reference_layout, n_reference, 2, "reference_layout"
  );
  std::vector<float> output(static_cast<std::size_t>(n_query) * 2u);
  const int status = fastembedr_cuda_landmark_tsne_from_device_knn(
    device_indices,
    device_distances,
    index_offset,
    reference_values.data(),
    query_values.data(),
    reference_layout_values.data(),
    n_reference,
    n_query,
    n_features,
    k,
    static_cast<float>(perplexity),
    n_iter,
    early_exaggeration_iter,
    static_cast<float>(learning_rate),
    static_cast<float>(early_exaggeration),
    static_cast<float>(exaggeration),
    static_cast<float>(initial_momentum),
    static_cast<float>(final_momentum),
    static_cast<float>(max_grad_norm),
    static_cast<float>(max_step_norm),
    n_negatives,
    exact_repulsion_threshold,
    static_cast<unsigned int>(seed),
    affine_neighbors,
    static_cast<float>(affine_ridge),
    static_cast<float>(max_extrapolation),
    output.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA landmark t-SNE failed: %s", cuda_embedding_error_message());
  }
  NumericMatrix layout(n_query, 2);
  for (int row = 0; row < n_query; ++row) {
    layout(row, 0) = static_cast<double>(output[static_cast<std::size_t>(row) * 2u]);
    layout(row, 1) = static_cast<double>(output[static_cast<std::size_t>(row) * 2u + 1u]);
  }
  List config = List::create(
    Rcpp::Named("optimizer") = n_iter + early_exaggeration_iter > 0 ?
      "opentsne_style_fixed_reference_transform_cuda_resident" : "projection_only_cuda_resident",
    Rcpp::Named("initialization") = "local_affine_knn_projection_cuda_resident",
    Rcpp::Named("repulsion") =
      (n_reference <= exact_repulsion_threshold || n_negatives >= n_reference) ?
      "exact_reference_cuda" : "sampled_reference_cuda",
    Rcpp::Named("affinities") = "query_conditional_cuda_float32",
    Rcpp::Named("affinity_storage") = "cuda_device_rank_major_float32",
    Rcpp::Named("n_negatives") = n_negatives,
    Rcpp::Named("knn_residency") = "cuda_device",
    Rcpp::Named("precision") = "float32"
  );
  return List::create(Rcpp::Named("Y") = layout, Rcpp::Named("config") = config);
}

List landmark_umap_project_refine_cuda_gpu_impl(SEXP gpu_knn,
                                                SEXP reference_data,
                                                SEXP query_data,
                                                SEXP reference_layout,
                                                IntegerVector landmark_rows,
                                                IntegerVector query_rows,
                                                int n_total,
                                                int n_epochs,
                                                double min_dist,
                                                int negative_sample_rate,
                                                double learning_rate,
                                                double repulsion_strength,
                                                int seed,
                                                int affine_neighbors,
                                                double affine_ridge,
                                                double max_extrapolation) {
  if (!Rf_isNewList(gpu_knn)) {
    Rcpp::stop("CUDA landmark UMAP requires a GPU-resident KNN object.");
  }
  List src(gpu_knn);
  const int n_query = src.containsElementNamed("n_query") ?
    Rcpp::as<int>(src["n_query"]) : Rcpp::as<int>(src["n"]);
  const int k = Rcpp::as<int>(src["k"]);
  const int index_offset = src.containsElementNamed("index_base") ?
    Rcpp::as<int>(src["index_base"]) : 1;
  const std::pair<int, int> reference_dims = cuda_matrix_dims(reference_data, "reference_data");
  const std::pair<int, int> query_dims = cuda_matrix_dims(query_data, "query_data");
  const std::pair<int, int> layout_dims = cuda_matrix_dims(reference_layout, "reference_layout");
  const int n_reference = reference_dims.first;
  const int n_features = reference_dims.second;
  if (query_dims.first != n_query || query_dims.second != n_features ||
      layout_dims.first != n_reference || layout_dims.second != 2 ||
      landmark_rows.size() != n_reference || query_rows.size() != n_query) {
    Rcpp::stop("CUDA landmark UMAP inputs have incompatible dimensions.");
  }
  if (n_total < n_reference + n_query || k < 1 || k > kMaxCudaProjectionNeighbors ||
      n_epochs < 0 || min_dist < 0.0 || negative_sample_rate < 0 ||
      learning_rate <= 0.0 || repulsion_strength <= 0.0) {
    Rcpp::stop("Invalid CUDA landmark UMAP parameters.");
  }
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");
  const int* device_indices = cuda_int_device_ptr_from_external(src["indices_ptr"], "indices_ptr");
  const float* device_distances = cuda_float_device_ptr_from_external(src["distances_ptr"], "distances_ptr");
  std::vector<float> reference_values = cuda_copy_matrix_float(
    reference_data, n_reference, n_features, "reference_data"
  );
  std::vector<float> query_values = cuda_copy_matrix_float(
    query_data, n_query, n_features, "query_data"
  );
  std::vector<float> reference_layout_values = cuda_copy_matrix_float(
    reference_layout, n_reference, 2, "reference_layout"
  );
  std::vector<float> output(static_cast<std::size_t>(n_total) * 2u);
  const std::pair<double, double> ab = find_ab_params(1.0, min_dist);
  const int status = fastembedr_cuda_landmark_umap_from_device_knn(
    device_indices,
    device_distances,
    index_offset,
    reference_values.data(),
    query_values.data(),
    reference_layout_values.data(),
    landmark_rows.begin(),
    query_rows.begin(),
    n_total,
    n_reference,
    n_query,
    n_features,
    k,
    n_epochs,
    negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(repulsion_strength),
    static_cast<unsigned int>(seed),
    affine_neighbors,
    static_cast<float>(affine_ridge),
    static_cast<float>(max_extrapolation),
    output.data()
  );
  if (status != 0) {
    Rcpp::stop("CUDA landmark UMAP failed: %s", cuda_embedding_error_message());
  }
  NumericMatrix layout(n_total, 2);
  for (int row = 0; row < n_total; ++row) {
    layout(row, 0) = static_cast<double>(output[static_cast<std::size_t>(row) * 2u]);
    layout(row, 1) = static_cast<double>(output[static_cast<std::size_t>(row) * 2u + 1u]);
  }
  return List::create(
    Rcpp::Named("layout") = layout,
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("projection") = "local_affine_knn_projection_cuda_resident",
    Rcpp::Named("refinement") = "fixed_landmark_umap_cuda_resident",
    Rcpp::Named("knn_residency") = "cuda_device",
    Rcpp::Named("precision") = "float32"
  );
}
