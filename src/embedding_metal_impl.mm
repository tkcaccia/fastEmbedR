#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <Rcpp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
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

struct EmbedParams {
  std::uint32_t n;
  std::uint32_t k;
  std::uint32_t n_epochs;
  std::uint32_t negative_sample_rate;
  std::uint32_t objective;
  std::uint32_t seed;
  float learning_rate;
  float a;
  float b;
  float max_weight;
};

struct RefinePrepareParams {
  std::uint32_t n_total;
  std::uint32_t n_rows;
  std::uint32_t k;
  std::uint32_t n_epochs;
  float global_mean;
};

struct MetalEmbeddingState {
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> embed_pipeline;
  id<MTLComputePipelineState> embed_atomic_inplace_pipeline;
  id<MTLComputePipelineState> refine_prepare_pipeline;
  id<MTLComputePipelineState> refine_rows_pipeline;
  id<MTLComputePipelineState> standardize_stats_pipeline;
  id<MTLComputePipelineState> standardize_apply_pipeline;
  id<MTLComputePipelineState> project_pipeline;
  id<MTLComputePipelineState> affine_project_pipeline;
  id<MTLComputePipelineState> landmark_project_interpolate_pipeline;
  id<MTLComputePipelineState> landmark_project_interpolate_knn_confidence_pipeline;
  id<MTLComputePipelineState> overwrite_landmarks_pipeline;
  id<MTLComputePipelineState> structure_score_pipeline;
  id<MTLComputePipelineState> silhouette_pipeline;
  id<MTLComputePipelineState> matrix_multiply_pipeline;
  id<MTLComputePipelineState> spectral_random_pipeline;
  id<MTLComputePipelineState> spectral_diffuse_pipeline;
  id<MTLComputePipelineState> spectral_stats_pipeline;
  id<MTLComputePipelineState> spectral_normalize_pipeline;
  id<MTLComputePipelineState> tsne_transform_pipeline;
  id<MTLComputePipelineState> opentsne_sum_q_pipeline;
  id<MTLComputePipelineState> opentsne_epoch_pipeline;
  id<MTLComputePipelineState> opentsne_center_pipeline;
  id<MTLComputePipelineState> opentsne_fft_clear_pipeline;
  id<MTLComputePipelineState> opentsne_fft_scatter_pipeline;
  id<MTLComputePipelineState> opentsne_fft_load_pipeline;
  id<MTLComputePipelineState> opentsne_mpsgraph_load_real_pipeline;
  id<MTLComputePipelineState> opentsne_fft_pack_real4_pipeline;
  id<MTLComputePipelineState> opentsne_fft_bit_reverse_rows_pipeline;
  id<MTLComputePipelineState> opentsne_fft_bit_reverse_cols_pipeline;
  id<MTLComputePipelineState> opentsne_fft_butterfly_rows_pipeline;
  id<MTLComputePipelineState> opentsne_fft_butterfly_cols_pipeline;
  id<MTLComputePipelineState> opentsne_fft_512_rows_stockham_pipeline;
  id<MTLComputePipelineState> opentsne_fft_512_cols_stockham_pipeline;
  id<MTLComputePipelineState> opentsne_fft_multiply_pipeline;
  id<MTLComputePipelineState> opentsne_fft_scale_pipeline;
  id<MTLComputePipelineState> opentsne_fft_sum_q_pipeline;
  id<MTLComputePipelineState> opentsne_fft_sum_q_blocks_pipeline;
  id<MTLComputePipelineState> opentsne_fft_finalize_sum_q_pipeline;
  id<MTLComputePipelineState> opentsne_fft_layout_stats_blocks_pipeline;
  id<MTLComputePipelineState> opentsne_fft_finalize_layout_stats_pipeline;
  id<MTLComputePipelineState> opentsne_fft_epoch_pipeline;
  id<MTLComputePipelineState> opentsne_fft_epoch_debug_pipeline;
  id<MTLCommandQueue> queue;
};

struct MatrixMultiplyParams {
  std::uint32_t left_rows;
  std::uint32_t left_cols;
  std::uint32_t right_cols;
  std::uint32_t transpose_left;
};

struct TsneTransformParams {
  std::uint32_t n_reference;
  std::uint32_t n_query;
  std::uint32_t k;
  std::uint32_t n_negatives;
  std::uint32_t seed;
  std::uint32_t exact_repulsion;
  float learning_rate;
  float exaggeration;
  float momentum;
  float max_grad_norm;
  float max_step_norm;
};

struct OpenTsneMetalParams {
  std::uint32_t n;
  std::uint32_t seed;
  float learning_rate;
  float exaggeration;
  float momentum;
  float min_gain;
  float max_step_norm;
  float inv_sum_q;
};

struct OpenTsneFFTGridParams {
  std::uint32_t n;
  std::uint32_t grid_size;
  std::uint32_t fft_size;
  float lower_x;
  float lower_y;
  float inv_spacing;
  float spacing;
  float inv_sum_q;
};

struct Center2 {
  float x;
  float y;
};

struct OpenTsneLayoutStats {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float sum_x;
  float sum_y;
};

constexpr int kMetalOpenTsneExactDenseThreshold = 6000;

struct WeightedEdge {
  std::uint64_t key;
  float weight;
  std::uint8_t direction;
};

enum ObjectiveId : std::uint32_t {
  kObjectiveUmap = 0,
  kObjectiveTsne = 1,
  kObjectivePacmap = 2,
  kObjectiveTrimap = 3,
  kObjectiveLocalmap = 4
};

constexpr int kMaxMetalNeighbors = 256;
constexpr int kMaxMetalProjectionNeighbors = 128;
constexpr int kMaxMetalTsneTransformNeighbors = 256;
constexpr int kMaxMetalScoreNeighbors = 64;
constexpr int kMaxMetalSilhouetteLabels = 128;
constexpr int kMetalScoreWidth = 6;
constexpr std::uint32_t kMetalEmbeddingEpochsPerCommand = 64u;

const char* metal_embed_kernel_source();

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

std::string ns_error_message(NSError* error) {
  if (error == nil) return "";
  NSString* description = [error localizedDescription];
  if (description == nil) return "unknown Metal error";
  return std::string([description UTF8String]);
}

id<MTLComputePipelineState> make_pipeline(MetalEmbeddingState& state,
                                          const char* function_name) {
  NSError* error = nil;
  NSString* name = [NSString stringWithUTF8String:function_name];
  id<MTLFunction> function = [state.library newFunctionWithName:name];
  if (function == nil) {
    Rcpp::stop("Failed to load Metal function `%s`.", function_name);
  }
  id<MTLComputePipelineState> pipeline =
    [state.device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (pipeline == nil) {
    Rcpp::stop(
      "Failed to create Metal pipeline `%s`: %s",
      function_name,
      ns_error_message(error).c_str()
    );
  }
  return pipeline;
}

MetalEmbeddingState& metal_embedding_state() {
  static MetalEmbeddingState state{};
  if (state.device != nil && state.embed_pipeline != nil &&
      state.embed_atomic_inplace_pipeline != nil &&
      state.refine_prepare_pipeline != nil &&
      state.refine_rows_pipeline != nil &&
      state.affine_project_pipeline != nil &&
      state.opentsne_sum_q_pipeline != nil &&
      state.opentsne_epoch_pipeline != nil &&
      state.opentsne_center_pipeline != nil &&
      state.opentsne_fft_clear_pipeline != nil &&
      state.opentsne_fft_scatter_pipeline != nil &&
      state.opentsne_fft_load_pipeline != nil &&
      state.opentsne_mpsgraph_load_real_pipeline != nil &&
      state.opentsne_fft_pack_real4_pipeline != nil &&
      state.opentsne_fft_bit_reverse_rows_pipeline != nil &&
      state.opentsne_fft_bit_reverse_cols_pipeline != nil &&
      state.opentsne_fft_butterfly_rows_pipeline != nil &&
      state.opentsne_fft_butterfly_cols_pipeline != nil &&
      state.opentsne_fft_512_rows_stockham_pipeline != nil &&
      state.opentsne_fft_512_cols_stockham_pipeline != nil &&
      state.opentsne_fft_multiply_pipeline != nil &&
      state.opentsne_fft_scale_pipeline != nil &&
      state.opentsne_fft_sum_q_pipeline != nil &&
      state.opentsne_fft_sum_q_blocks_pipeline != nil &&
      state.opentsne_fft_finalize_sum_q_pipeline != nil &&
      state.opentsne_fft_layout_stats_blocks_pipeline != nil &&
      state.opentsne_fft_finalize_layout_stats_pipeline != nil &&
      state.opentsne_fft_epoch_pipeline != nil &&
      state.opentsne_fft_epoch_debug_pipeline != nil &&
      state.queue != nil) {
    return state;
  }

  state.device = MTLCreateSystemDefaultDevice();
  if (state.device == nil) {
    Rcpp::stop("No Metal device is available.");
  }

  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:metal_embed_kernel_source()];
  state.library = [state.device newLibraryWithSource:source options:nil error:&error];
  if (state.library == nil) {
    Rcpp::stop("Failed to compile Metal embedding kernel: %s", ns_error_message(error).c_str());
  }

  state.embed_pipeline = make_pipeline(state, "embed_epoch");
  state.embed_atomic_inplace_pipeline = make_pipeline(state, "embed_epoch_atomic_inplace");
  state.refine_prepare_pipeline = make_pipeline(state, "umap_refine_prepare_rows");
  state.refine_rows_pipeline = make_pipeline(state, "umap_refine_rows_atomic_inplace");
  state.standardize_stats_pipeline = make_pipeline(state, "standardize_stats");
  state.standardize_apply_pipeline = make_pipeline(state, "standardize_apply");
  state.project_pipeline = make_pipeline(state, "project_membership");
  state.affine_project_pipeline = make_pipeline(state, "project_embedding_affine_rows");
  state.landmark_project_interpolate_pipeline = make_pipeline(state, "landmark_project_interpolate");
  state.landmark_project_interpolate_knn_confidence_pipeline =
    make_pipeline(state, "landmark_project_interpolate_knn_confidence");
  state.overwrite_landmarks_pipeline = make_pipeline(state, "overwrite_landmark_rows");
  state.structure_score_pipeline = make_pipeline(state, "structure_score_rows");
  state.silhouette_pipeline = make_pipeline(state, "silhouette_rows");
  state.matrix_multiply_pipeline = make_pipeline(state, "matrix_multiply");
  state.spectral_random_pipeline = make_pipeline(state, "spectral_random_init");
  state.spectral_diffuse_pipeline = make_pipeline(state, "spectral_diffuse");
  state.spectral_stats_pipeline = make_pipeline(state, "spectral_init_stats");
  state.spectral_normalize_pipeline = make_pipeline(state, "spectral_normalize");
  state.tsne_transform_pipeline = make_pipeline(state, "tsne_transform_epoch");
  state.opentsne_sum_q_pipeline = make_pipeline(state, "opentsne_sum_q_rows");
  state.opentsne_epoch_pipeline = make_pipeline(state, "opentsne_epoch_exact");
  state.opentsne_center_pipeline = make_pipeline(state, "opentsne_apply_center");
  state.opentsne_fft_clear_pipeline = make_pipeline(state, "opentsne_fft_clear_grids");
  state.opentsne_fft_scatter_pipeline = make_pipeline(state, "opentsne_fft_scatter_bilinear");
  state.opentsne_fft_load_pipeline = make_pipeline(state, "opentsne_fft_load_inputs");
  state.opentsne_mpsgraph_load_real_pipeline = make_pipeline(state, "opentsne_mpsgraph_load_real_inputs");
  state.opentsne_fft_pack_real4_pipeline = make_pipeline(state, "opentsne_fft_pack_real_to_complex4");
  state.opentsne_fft_bit_reverse_rows_pipeline = make_pipeline(state, "opentsne_fft_bit_reverse_rows");
  state.opentsne_fft_bit_reverse_cols_pipeline = make_pipeline(state, "opentsne_fft_bit_reverse_cols");
  state.opentsne_fft_butterfly_rows_pipeline = make_pipeline(state, "opentsne_fft_butterfly_rows");
  state.opentsne_fft_butterfly_cols_pipeline = make_pipeline(state, "opentsne_fft_butterfly_cols");
  state.opentsne_fft_512_rows_stockham_pipeline = make_pipeline(state, "opentsne_fft_512_rows_stockham");
  state.opentsne_fft_512_cols_stockham_pipeline = make_pipeline(state, "opentsne_fft_512_cols_stockham");
  state.opentsne_fft_multiply_pipeline = make_pipeline(state, "opentsne_fft_multiply");
  state.opentsne_fft_scale_pipeline = make_pipeline(state, "opentsne_fft_scale");
  state.opentsne_fft_sum_q_pipeline = make_pipeline(state, "opentsne_fft_sum_q_rows");
  state.opentsne_fft_sum_q_blocks_pipeline = make_pipeline(state, "opentsne_fft_sum_q_blocks");
  state.opentsne_fft_finalize_sum_q_pipeline = make_pipeline(state, "opentsne_fft_finalize_sum_q");
  state.opentsne_fft_layout_stats_blocks_pipeline = make_pipeline(state, "opentsne_fft_layout_stats_blocks");
  state.opentsne_fft_finalize_layout_stats_pipeline = make_pipeline(state, "opentsne_fft_finalize_layout_stats");
  state.opentsne_fft_epoch_pipeline = make_pipeline(state, "opentsne_epoch_fft_grid");
  state.opentsne_fft_epoch_debug_pipeline = make_pipeline(state, "opentsne_epoch_fft_grid_debug");

  state.queue = [state.device newCommandQueue];
  if (state.queue == nil) {
    Rcpp::stop("Failed to create Metal embedding command queue.");
  }

  return state;
}

std::uint32_t objective_id(const std::string& objective) {
  if (objective == "umap") return kObjectiveUmap;
  if (objective == "tsne") return kObjectiveTsne;
  if (objective == "pacmap") return kObjectivePacmap;
  if (objective == "trimap") return kObjectiveTrimap;
  if (objective == "localmap") return kObjectiveLocalmap;
  Rcpp::stop("Unknown Metal embedding objective: %s", objective.c_str());
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
                 std::vector<std::int32_t>& neighbors,
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
                                  std::vector<std::int32_t>& neighbors,
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
    if (static_cast<int>(row.size()) > kMaxMetalNeighbors) {
      std::nth_element(
        row.begin(),
        row.begin() + kMaxMetalNeighbors,
        row.end(),
        row_less
      );
      row.resize(kMaxMetalNeighbors);
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
                                 const std::uint32_t objective,
                                 const int n_epochs,
                                 std::vector<std::int32_t>& neighbors,
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

struct TsneSparseMetalGraph {
  std::vector<std::int32_t> row_ptr;
  std::vector<std::int32_t> col;
  std::vector<float> val;
};

struct TsnePackedEdge {
  std::uint64_t key;
  double value;
};

int metal_cpu_prep_threads(const int n) {
  const char* raw = std::getenv("FASTEMBEDR_N_THREADS");
  int requested = 1;
  if (raw != nullptr && raw[0] != '\0') {
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end != raw && parsed > 0L &&
        parsed <= static_cast<long>(std::numeric_limits<int>::max())) {
      requested = static_cast<int>(parsed);
    }
  }
  const unsigned int hw = std::thread::hardware_concurrency();
  const int available = hw == 0u ? requested : static_cast<int>(hw);
  return std::max(1, std::min(std::max(1, n), std::min(requested, available)));
}

template <typename Function>
void metal_parallel_for(const int n, const int n_threads, Function fn) {
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

void compute_tsne_row_probabilities_metal(const NumericMatrix& distances,
                                          const int row,
                                          const double perplexity,
                                          std::vector<double>& row_p) {
  const int k = distances.ncol();
  row_p.assign(static_cast<std::size_t>(k), 0.0);
  bool found = false;
  double beta = 1.0;
  double min_beta = -std::numeric_limits<double>::max();
  double max_beta = std::numeric_limits<double>::max();
  const double target_entropy = std::log(perplexity);
  const double tol = 1e-5;
  double sum_p = std::numeric_limits<double>::min();

  for (int iter = 0; !found && iter < 200; ++iter) {
    sum_p = std::numeric_limits<double>::min();
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      const double p = std::exp(-beta * d * d);
      row_p[static_cast<std::size_t>(j)] = p;
      sum_p += p;
    }
    double entropy = 0.0;
    for (int j = 0; j < k; ++j) {
      const double d = distances(row, j);
      entropy += beta * d * d * row_p[static_cast<std::size_t>(j)];
    }
    entropy = entropy / sum_p + std::log(sum_p);
    const double diff = entropy - target_entropy;
    if (std::abs(diff) < tol) {
      found = true;
    } else if (diff > 0.0) {
      min_beta = beta;
      beta = max_beta == std::numeric_limits<double>::max() ?
        beta * 2.0 :
        (beta + max_beta) / 2.0;
    } else {
      max_beta = beta;
      beta = min_beta == -std::numeric_limits<double>::max() ?
        beta / 2.0 :
        (beta + min_beta) / 2.0;
    }
  }
  const double inv_sum = 1.0 / sum_p;
  for (double& value : row_p) value *= inv_sum;
}

TsneSparseMetalGraph build_tsne_sparse_graph_metal(const IntegerMatrix& indices,
                                                   const NumericMatrix& distances,
                                                   const double perplexity) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  const int offset = (min_idx >= 1 && max_idx <= n) ? 1 : 0;

  const int n_threads = metal_cpu_prep_threads(n);
  std::vector<std::vector<TsnePackedEdge>> local_edges(static_cast<std::size_t>(n_threads));
  metal_parallel_for(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<double> row_p;
    row_p.reserve(static_cast<std::size_t>(k));
    std::vector<TsnePackedEdge>& edges = local_edges[static_cast<std::size_t>(thread_id)];
    edges.reserve(edges.size() + static_cast<std::size_t>(std::max(0, end - begin)) * k);
    for (int i = begin; i < end; ++i) {
      compute_tsne_row_probabilities_metal(distances, i, perplexity, row_p);
      for (int j = 0; j < k; ++j) {
        const int nb = indices(i, j) - offset;
        if (nb < 0 || nb >= n) Rcpp::stop("KNN indices are out of range.");
        const double d = distances(i, j);
        if (!std::isfinite(d) || d < 0.0) {
          Rcpp::stop("KNN distances must be finite and non-negative.");
        }
        if (nb == i) continue;
        const int a = std::min(i, nb);
        const int b = std::max(i, nb);
        edges.push_back({edge_key(a, b), row_p[static_cast<std::size_t>(j)]});
      }
    }
  });
  std::size_t edge_count = 0;
  for (const auto& edges : local_edges) edge_count += edges.size();
  std::vector<TsnePackedEdge> edges;
  edges.reserve(edge_count);
  for (auto& local : local_edges) {
    edges.insert(edges.end(), local.begin(), local.end());
    std::vector<TsnePackedEdge>().swap(local);
  }
  if (edges.empty()) Rcpp::stop("KNN graph produced no non-self t-SNE edges.");
  std::sort(edges.begin(), edges.end(), [](const TsnePackedEdge& a, const TsnePackedEdge& b) {
    return a.key < b.key;
  });

  std::size_t write = 0;
  double total_directed_mass = 0.0;
  TsneSparseMetalGraph graph;
  graph.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
  for (std::size_t read = 0; read < edges.size();) {
    const std::uint64_t key = edges[read].key;
    double sum = 0.0;
    while (read < edges.size() && edges[read].key == key) {
      sum += edges[read].value;
      ++read;
    }
    edges[write++] = {key, sum};
    total_directed_mass += sum;
    ++graph.row_ptr[static_cast<std::size_t>(key_head(key) + 1)];
    ++graph.row_ptr[static_cast<std::size_t>(key_tail(key) + 1)];
  }
  edges.resize(write);
  if (!std::isfinite(total_directed_mass) || total_directed_mass <= 0.0) {
    Rcpp::stop("t-SNE probability normalization failed.");
  }
  for (int i = 0; i < n; ++i) {
    graph.row_ptr[static_cast<std::size_t>(i + 1)] += graph.row_ptr[static_cast<std::size_t>(i)];
  }
  graph.col.assign(static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(n)]), 0);
  graph.val.assign(graph.col.size(), 0.0f);
  std::vector<std::int32_t> fill = graph.row_ptr;
  for (const auto& edge : edges) {
    const int a = key_head(edge.key);
    const int b = key_tail(edge.key);
    const float value = static_cast<float>(0.5 * edge.value / total_directed_mass);
    int pos = fill[static_cast<std::size_t>(a)]++;
    graph.col[static_cast<std::size_t>(pos)] = b;
    graph.val[static_cast<std::size_t>(pos)] = value;
    pos = fill[static_cast<std::size_t>(b)]++;
    graph.col[static_cast<std::size_t>(pos)] = a;
    graph.val[static_cast<std::size_t>(pos)] = value;
  }
  return graph;
}

std::vector<float> initialize_opentsne_metal_layout(const NumericMatrix& y_init,
                                                    const bool init,
                                                    const int n,
                                                    const int seed) {
  std::vector<float> out(static_cast<std::size_t>(n) * 2u, 0.0f);
  if (init) {
    if (y_init.nrow() != n || y_init.ncol() != 2) {
      Rcpp::stop("`Y_init` must have one row per point and two columns for Metal openTSNE.");
    }
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i) * 2u] = static_cast<float>(y_init(i, 0));
      out[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(y_init(i, 1));
    }
  } else {
    const unsigned int resolved_seed = seed == NA_INTEGER ? 5489u : static_cast<unsigned int>(seed);
    std::mt19937 rng(resolved_seed);
    std::normal_distribution<float> normal(0.0f, 1.0e-4f);
    for (float& value : out) value = normal(rng);
  }
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (int i = 0; i < n; ++i) {
    mean_x += out[static_cast<std::size_t>(i) * 2u];
    mean_y += out[static_cast<std::size_t>(i) * 2u + 1u];
  }
  mean_x /= static_cast<double>(n);
  mean_y /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i) * 2u] -= static_cast<float>(mean_x);
    out[static_cast<std::size_t>(i) * 2u + 1u] -= static_cast<float>(mean_y);
  }
  return out;
}

const char* metal_embed_kernel_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

struct EmbedParams {
  uint n;
  uint k;
  uint n_epochs;
  uint negative_sample_rate;
  uint objective;
  uint seed;
  float learning_rate;
  float a;
  float b;
  float max_weight;
};

struct RefinePrepareParams {
  uint n_total;
  uint n_rows;
  uint k;
  uint n_epochs;
  float global_mean;
};

struct MatrixMultiplyParams {
  uint left_rows;
  uint left_cols;
  uint right_cols;
  uint transpose_left;
};

struct TsneTransformParams {
  uint n_reference;
  uint n_query;
  uint k;
  uint n_negatives;
  uint seed;
  uint exact_repulsion;
  float learning_rate;
  float exaggeration;
  float momentum;
  float max_grad_norm;
  float max_step_norm;
};

struct OpenTsneMetalParams {
  uint n;
  uint seed;
  float learning_rate;
  float exaggeration;
  float momentum;
  float min_gain;
  float max_step_norm;
  float inv_sum_q;
};

struct OpenTsneFFTGridParams {
  uint n;
  uint grid_size;
  uint fft_size;
  float lower_x;
  float lower_y;
  float inv_spacing;
  float spacing;
  float inv_sum_q;
};

struct OpenTsneLayoutStats {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float sum_x;
  float sum_y;
};

uint mix_uint(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

uint deterministic_vertex(uint n, uint seed, uint epoch, uint i, uint edge, uint sample) {
  uint x = seed;
  x ^= epoch * 0x9e3779b9u;
  x ^= (i + 1u) * 0x85ebca6bu;
  x ^= (edge + 1u) * 0xc2b2ae35u;
  x ^= (sample + 1u) * 0x27d4eb2du;
  return mix_uint(x) % n;
}

uint deterministic_reference(uint n, uint seed, uint epoch, uint row, uint sample) {
  uint x = seed;
  x ^= (epoch + 1u) * 0x9e3779b9u;
  x ^= (row + 1u) * 0x85ebca6bu;
  x ^= (sample + 1u) * 0xc2b2ae35u;
  return mix_uint(x) % n;
}

float sign_component(float x) {
  if (x > 0.0f) return 1.0f;
  if (x < 0.0f) return -1.0f;
  return 0.0f;
}

float deterministic_unit_signed(uint seed, uint row, uint component) {
  uint x = seed;
  x ^= (row + 1u) * 0x9e3779b9u;
  x ^= (component + 1u) * 0x85ebca6bu;
  x = mix_uint(x);
  return (float(x & 0x00ffffffu) / 8388608.0f) - 1.0f;
}

float clip4(float x) {
  return clamp(x, -4.0f, 4.0f);
}

float attractive_coeff(float d2, float weight, constant EmbedParams& p) {
  if (p.objective == 0u) {
    if (d2 <= 0.0f) return 0.0f;
    float d2b = pow(d2, p.b);
    return -2.0f * p.a * p.b * (d2b / d2) / (p.a * d2b + 1.0f);
  }
  if (p.objective == 1u) return -2.0f * weight / (1.0f + d2);
  if (p.objective == 2u) return -2.0f * weight / (10.0f + d2);
  if (p.objective == 4u) return -2.5f * weight / (0.15f + d2);
  return -2.0f * weight / (1.0f + d2);
}

float repulsive_coeff(float d2, constant EmbedParams& p) {
  if (d2 <= 0.0f) return 0.0f;
  if (p.objective == 0u) {
    float d2b = pow(d2, p.b);
    return 2.0f * p.b / ((0.001f + d2) * (p.a * d2b + 1.0f));
  }
  if (p.objective == 1u) return 2.0f / ((1.0f + d2) * (1.0f + d2));
  if (p.objective == 2u) return 0.2f * 2.0f / (1.0f + d2);
  if (p.objective == 4u) return 0.8125f / ((0.15f + d2) * (1.0f + d2));
  return 2.0f / (1.0f + d2);
}

int positive_samples_this_epoch(float weight, constant EmbedParams& p, uint epoch) {
  if (p.objective != 0u) return 1;
  if (weight <= 0.0f) return 0;
  float period = p.max_weight / max(weight, 1.0e-6f);
  float now = float(epoch + 1u);
  float previous = float(epoch);
  int current_sample = int(floor(now / period));
  int previous_sample = int(floor(previous / period));
  int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

int cumulative_umap_negative_samples(float active_epoch, float negative_period) {
  if (negative_period <= 0.0f || !isfinite(negative_period)) return 0;
  int samples = int(floor(((active_epoch - negative_period) / negative_period) + 1.0e-6f));
  return samples > 0 ? samples : 0;
}

int negative_samples_this_epoch(float weight, constant EmbedParams& p, uint epoch) {
  if (p.objective != 0u) return int(p.negative_sample_rate);
  if (weight <= 0.0f || p.negative_sample_rate == 0u) return 0;
  float period = p.max_weight / max(weight, 1.0e-6f);
  float now = float(epoch + 1u);
  float previous = float(epoch);
  int current_sample = int(floor(now / period));
  int previous_sample = int(floor(previous / period));
  if (current_sample <= previous_sample) return 0;

  float negative_period = period / float(p.negative_sample_rate);
  int current_total = cumulative_umap_negative_samples(now, negative_period);
  int previous_total = 0;
  if (previous_sample > 0) {
    float previous_active_epoch = ceil(float(previous_sample) * period);
    previous_total = cumulative_umap_negative_samples(previous_active_epoch, negative_period);
  }
  int samples = current_total - previous_total;
  return samples > 0 ? samples : 0;
}

int positive_samples_this_epoch_period(float period, constant EmbedParams& p, uint epoch) {
  if (p.objective != 0u) return 1;
  if (period <= 0.0f || !isfinite(period)) return 0;
  float now = float(epoch + 1u);
  float previous = float(epoch);
  int current_sample = int(floor(now / period));
  int previous_sample = int(floor(previous / period));
  int samples = current_sample - previous_sample;
  return samples > 0 ? samples : 0;
}

int negative_samples_this_epoch_period(float period, constant EmbedParams& p, uint epoch) {
  if (p.objective != 0u) return int(p.negative_sample_rate);
  if (period <= 0.0f || !isfinite(period) || p.negative_sample_rate == 0u) return 0;
  float now = float(epoch + 1u);
  float previous = float(epoch);
  int current_sample = int(floor(now / period));
  int previous_sample = int(floor(previous / period));
  if (current_sample <= previous_sample) return 0;

  float negative_period = period / float(p.negative_sample_rate);
  int current_total = cumulative_umap_negative_samples(now, negative_period);
  int previous_total = 0;
  if (previous_sample > 0) {
    float previous_active_epoch = ceil(float(previous_sample) * period);
    previous_total = cumulative_umap_negative_samples(previous_active_epoch, negative_period);
  }
  int samples = current_total - previous_total;
  return samples > 0 ? samples : 0;
}

kernel void embed_epoch(
  device const float2* current [[buffer(0)]],
  device float2* next [[buffer(1)]],
  device const int* neighbors [[buffer(2)]],
  device const float* weights [[buffer(3)]],
  constant EmbedParams& p [[buffer(4)]],
  constant uint& epoch [[buffer(5)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= p.n) return;

  float2 yi = current[gid];
  float2 delta = float2(0.0f, 0.0f);

  for (uint e = 0; e < p.k; ++e) {
    int nb = neighbors[gid * p.k + e];
    if (nb < 0 || uint(nb) >= p.n || uint(nb) == gid) continue;
    float w = weights[gid * p.k + e];
    int positive_samples = positive_samples_this_epoch(w, p, epoch);
    if (positive_samples <= 0) continue;
    float2 diff = yi - current[uint(nb)];
    float d2 = dot(diff, diff);

    if (p.objective == 3u) {
      uint samples = max(1u, p.negative_sample_rate);
      float triplet_w = w / float(samples);
      float pos_d2 = d2 + 1.0e-4f;
      for (uint s = 0; s < samples; ++s) {
        uint neg = deterministic_vertex(p.n, p.seed, epoch, gid, e, s);
        if (neg == gid || neg == uint(nb)) continue;
        float2 ndiff = yi - current[neg];
        float neg_d2 = dot(ndiff, ndiff) + 1.0e-4f;
        float denom = pos_d2 + neg_d2 + 1.0e-6f;
        float scale = triplet_w / (denom * denom);
        float pos_coeff = -2.0f * scale * neg_d2;
        float neg_coeff =  2.0f * scale * pos_d2;
        delta += float2(
          clip4(pos_coeff * diff.x) + clip4(neg_coeff * ndiff.x),
          clip4(pos_coeff * diff.y) + clip4(neg_coeff * ndiff.y)
        );
      }
      continue;
    }

    float coeff = attractive_coeff(d2, w, p);
    delta += float2(clip4(coeff * diff.x), clip4(coeff * diff.y));

    uint neg_samples = uint(negative_samples_this_epoch(w, p, epoch));
    for (uint s = 0; s < neg_samples; ++s) {
      uint neg = deterministic_vertex(p.n, p.seed, epoch, gid, e, s);
      if (neg == gid || neg == uint(nb)) continue;
      float2 ndiff = yi - current[neg];
      float nd2 = dot(ndiff, ndiff);
      float rcoeff = repulsive_coeff(nd2, p);
      delta += float2(clip4(rcoeff * ndiff.x), clip4(rcoeff * ndiff.y));
    }
  }

  float alpha = p.learning_rate * (1.0f - float(epoch) / max(1.0f, float(p.n_epochs)));
  next[gid] = yi + alpha * delta;
}

int fixed_delta(float value) {
  constexpr float scale = 65536.0f;
  return int(clamp(value * scale, -2140000000.0f, 2140000000.0f));
}

kernel void embed_epoch_atomic_inplace(
  device atomic_int* layout [[buffer(0)]],
  device const int* neighbors [[buffer(1)]],
  device const float* weights [[buffer(2)]],
  device const float* epochs_per_sample [[buffer(3)]],
  constant EmbedParams& p [[buffer(4)]],
  constant uint& epoch [[buffer(5)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= p.n) return;

  constexpr float inv_scale = 1.0f / 65536.0f;
  float alpha = p.learning_rate * (1.0f - float(epoch) / max(1.0f, float(p.n_epochs)));

  for (uint e = 0; e < p.k; ++e) {
    uint pos = gid * p.k + e;
    int nb_i = neighbors[pos];
    if (nb_i < 0 || uint(nb_i) >= p.n || uint(nb_i) == gid) continue;
    float period = epochs_per_sample[pos];
    int positive_samples = positive_samples_this_epoch_period(period, p, epoch);
    if (positive_samples <= 0) continue;

    uint nb = uint(nb_i);
    uint head_base = gid * 2u;
    uint tail_base = nb * 2u;
    float head_x = float(atomic_load_explicit(&layout[head_base], memory_order_relaxed)) * inv_scale;
    float head_y = float(atomic_load_explicit(&layout[head_base + 1u], memory_order_relaxed)) * inv_scale;
    float tail_x = float(atomic_load_explicit(&layout[tail_base], memory_order_relaxed)) * inv_scale;
    float tail_y = float(atomic_load_explicit(&layout[tail_base + 1u], memory_order_relaxed)) * inv_scale;

    float2 diff = float2(head_x - tail_x, head_y - tail_y);
    float d2 = max(1.1920928955078125e-7f, dot(diff, diff));
    float w = weights[pos];
    float coeff = attractive_coeff(d2, w, p);
    float2 attractive = alpha * float2(clip4(coeff * diff.x), clip4(coeff * diff.y));

    atomic_fetch_add_explicit(&layout[head_base], fixed_delta(attractive.x), memory_order_relaxed);
    atomic_fetch_add_explicit(&layout[head_base + 1u], fixed_delta(attractive.y), memory_order_relaxed);
    atomic_fetch_add_explicit(&layout[tail_base], fixed_delta(-attractive.x), memory_order_relaxed);
    atomic_fetch_add_explicit(&layout[tail_base + 1u], fixed_delta(-attractive.y), memory_order_relaxed);

    uint neg_samples = uint(negative_samples_this_epoch_period(period, p, epoch));
    for (uint s = 0; s < neg_samples; ++s) {
      uint neg = deterministic_vertex(p.n, p.seed, epoch, gid, e, s);
      if (neg == gid || neg == nb) continue;
      uint neg_base = neg * 2u;
      head_x = float(atomic_load_explicit(&layout[head_base], memory_order_relaxed)) * inv_scale;
      head_y = float(atomic_load_explicit(&layout[head_base + 1u], memory_order_relaxed)) * inv_scale;
      float neg_x = float(atomic_load_explicit(&layout[neg_base], memory_order_relaxed)) * inv_scale;
      float neg_y = float(atomic_load_explicit(&layout[neg_base + 1u], memory_order_relaxed)) * inv_scale;
      float2 ndiff = float2(head_x - neg_x, head_y - neg_y);
      float nd2 = max(1.1920928955078125e-7f, dot(ndiff, ndiff));
      float rcoeff = repulsive_coeff(nd2, p);
      float2 repulsive = alpha * float2(clip4(rcoeff * ndiff.x), clip4(rcoeff * ndiff.y));
      atomic_fetch_add_explicit(&layout[head_base], fixed_delta(repulsive.x), memory_order_relaxed);
      atomic_fetch_add_explicit(&layout[head_base + 1u], fixed_delta(repulsive.y), memory_order_relaxed);
    }
  }
}

kernel void umap_refine_prepare_rows(
  device const int* row_ids [[buffer(0)]],
  device const int* neighbors [[buffer(1)]],
  device const float* distances [[buffer(2)]],
  device float* weights [[buffer(3)]],
  device float* epochs_per_sample [[buffer(4)]],
  constant RefinePrepareParams& p [[buffer(5)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= p.n_rows) return;

  uint row = uint(row_ids[gid]);
  uint base = gid * p.k;
  float rho = INFINITY;
  float row_sum = 0.0f;
  uint row_count = 0u;

  for (uint e = 0; e < p.k; ++e) {
    float d = distances[base + e];
    if (!isfinite(d)) continue;
    if (d >= 0.0f) {
      row_sum += d;
      row_count += 1u;
    }
    if (d > 0.0f && d < rho) rho = d;
  }
  if (!isfinite(rho)) rho = 0.0f;

  float target = log2(max(1.0f, float(p.k)));
  float lo = 0.0f;
  float hi = INFINITY;
  float sigma = 1.0f;
  float best_sigma = sigma;
  float best_diff = INFINITY;

  for (uint iter = 0u; iter < 64u; ++iter) {
    float psum = 0.0f;
    float safe_sigma = max(sigma, 1.0e-12f);
    for (uint e = 0; e < p.k; ++e) {
      float raw = distances[base + e];
      if (!isfinite(raw)) continue;
      float d = raw - rho;
      psum += d <= 0.0f ? 1.0f : exp(-d / safe_sigma);
    }
    float diff = abs(psum - target);
    if (diff < best_diff) {
      best_diff = diff;
      best_sigma = sigma;
    }
    if (psum > target) {
      hi = sigma;
      sigma = 0.5f * (lo + hi);
    } else {
      lo = sigma;
      sigma = isinf(hi) ? sigma * 2.0f : 0.5f * (lo + hi);
    }
    if (diff < 1.0e-5f) break;
  }

  float row_mean = row_count > 0u ? row_sum / float(row_count) : p.global_mean;
  float sigma_floor = 1.0e-3f * (rho > 0.0f ? row_mean : p.global_mean);
  best_sigma = max(max(best_sigma, sigma_floor), 1.0e-12f);

  for (uint e = 0; e < p.k; ++e) {
    uint pos = base + e;
    int nb_i = neighbors[pos];
    float d = distances[pos];
    float w = 0.0f;
    if (nb_i >= 0 && uint(nb_i) < p.n_total && uint(nb_i) != row &&
        isfinite(d) && d >= 0.0f) {
      w = d <= rho ? 1.0f : exp(-(d - rho) / best_sigma);
      if (!isfinite(w) || w <= 0.0f) w = 0.0f;
    }
    weights[pos] = w;
    epochs_per_sample[pos] = w > 0.0f ? 1.0f / max(w, 1.0e-6f) : 0.0f;
  }
}

kernel void umap_refine_rows_atomic_inplace(
  device atomic_int* layout [[buffer(0)]],
  device const int* row_ids [[buffer(1)]],
  device const int* neighbors [[buffer(2)]],
  device const float* weights [[buffer(3)]],
  device const float* epochs_per_sample [[buffer(4)]],
  constant EmbedParams& p [[buffer(5)]],
  constant uint& n_rows [[buffer(6)]],
  constant uint& epoch [[buffer(7)]],
  device const uchar* update_mask [[buffer(8)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= n_rows) return;

  constexpr float inv_scale = 1.0f / 65536.0f;
  float alpha = p.learning_rate * (1.0f - float(epoch) / max(1.0f, float(p.n_epochs)));
  uint row = uint(row_ids[gid]);
  if (row >= p.n) return;
  uint head_base = row * 2u;

  for (uint e = 0; e < p.k; ++e) {
    uint pos = gid * p.k + e;
    int nb_i = neighbors[pos];
    if (nb_i < 0 || uint(nb_i) >= p.n || uint(nb_i) == row) continue;
    float period = epochs_per_sample[pos];
    int positive_samples = positive_samples_this_epoch_period(period, p, epoch);
    if (positive_samples <= 0) continue;

    uint nb = uint(nb_i);
    uint tail_base = nb * 2u;
    float head_x = float(atomic_load_explicit(&layout[head_base], memory_order_relaxed)) * inv_scale;
    float head_y = float(atomic_load_explicit(&layout[head_base + 1u], memory_order_relaxed)) * inv_scale;
    float tail_x = float(atomic_load_explicit(&layout[tail_base], memory_order_relaxed)) * inv_scale;
    float tail_y = float(atomic_load_explicit(&layout[tail_base + 1u], memory_order_relaxed)) * inv_scale;

    float2 diff = float2(head_x - tail_x, head_y - tail_y);
    float d2 = max(1.1920928955078125e-7f, dot(diff, diff));
    float coeff = attractive_coeff(d2, weights[pos], p);
    float2 attractive = alpha * float2(clip4(coeff * diff.x), clip4(coeff * diff.y));

    atomic_fetch_add_explicit(&layout[head_base], fixed_delta(attractive.x), memory_order_relaxed);
    atomic_fetch_add_explicit(&layout[head_base + 1u], fixed_delta(attractive.y), memory_order_relaxed);
    if (update_mask[nb] != uchar(0)) {
      atomic_fetch_add_explicit(&layout[tail_base], fixed_delta(-attractive.x), memory_order_relaxed);
      atomic_fetch_add_explicit(&layout[tail_base + 1u], fixed_delta(-attractive.y), memory_order_relaxed);
    }

    uint neg_samples = uint(negative_samples_this_epoch_period(period, p, epoch));
    for (uint s = 0; s < neg_samples; ++s) {
      uint neg = deterministic_vertex(p.n, p.seed, epoch, row, e, s);
      if (neg == row || neg == nb) continue;
      uint neg_base = neg * 2u;
      head_x = float(atomic_load_explicit(&layout[head_base], memory_order_relaxed)) * inv_scale;
      head_y = float(atomic_load_explicit(&layout[head_base + 1u], memory_order_relaxed)) * inv_scale;
      float neg_x = float(atomic_load_explicit(&layout[neg_base], memory_order_relaxed)) * inv_scale;
      float neg_y = float(atomic_load_explicit(&layout[neg_base + 1u], memory_order_relaxed)) * inv_scale;
      float2 ndiff = float2(head_x - neg_x, head_y - neg_y);
      float nd2 = max(1.1920928955078125e-7f, dot(ndiff, ndiff));
      float rcoeff = repulsive_coeff(nd2, p);
      float2 repulsive = alpha * float2(clip4(rcoeff * ndiff.x), clip4(rcoeff * ndiff.y));
      atomic_fetch_add_explicit(&layout[head_base], fixed_delta(repulsive.x), memory_order_relaxed);
      atomic_fetch_add_explicit(&layout[head_base + 1u], fixed_delta(repulsive.y), memory_order_relaxed);
    }
  }
}

kernel void matrix_multiply(
  device const float* left [[buffer(0)]],
  device const float* right [[buffer(1)]],
  device float* out [[buffer(2)]],
  constant MatrixMultiplyParams& p [[buffer(3)]],
  uint2 gid [[thread_position_in_grid]]
) {
  uint row = gid.y;
  uint col = gid.x;
  uint out_rows = p.transpose_left == 0u ? p.left_rows : p.left_cols;
  uint inner = p.transpose_left == 0u ? p.left_cols : p.left_rows;
  if (row >= out_rows || col >= p.right_cols) return;

  float total = 0.0f;
  if (p.transpose_left != 0u) {
    for (uint t = 0u; t < inner; ++t) {
      total += left[t + row * p.left_rows] * right[t + col * p.left_rows];
    }
  } else {
    for (uint t = 0u; t < inner; ++t) {
      total += left[row + t * p.left_rows] * right[t + col * p.left_cols];
    }
  }
  out[row + col * out_rows] = total;
}

kernel void spectral_random_init(
  device float2* values [[buffer(0)]],
  constant uint& n [[buffer(1)]],
  constant uint& seed [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  values[row] = float2(
    deterministic_unit_signed(seed, row, 0u),
    deterministic_unit_signed(seed, row, 1u)
  );
}

kernel void spectral_diffuse(
  device const int* neighbors [[buffer(0)]],
  device const float* weights [[buffer(1)]],
  device const float2* current [[buffer(2)]],
  device float2* next [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& width [[buffer(5)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  float2 total = float2(0.0f, 0.0f);
  float weight_sum = 0.0f;
  for (uint j = 0u; j < width; ++j) {
    uint pos = row * width + j;
    int nb = neighbors[pos];
    float w = weights[pos];
    if (nb < 0 || uint(nb) >= n || uint(nb) == row || w <= 0.0f) continue;
    total += w * current[uint(nb)];
    weight_sum += w;
  }
  next[row] = weight_sum > 0.0f ? total / weight_sum : current[row];
}

kernel void spectral_init_stats(
  device const float2* values [[buffer(0)]],
  device float* stats [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  constant uint& threads [[buffer(3)]],
  uint tid [[thread_index_in_threadgroup]]
) {
  threadgroup float sx[256];
  threadgroup float sy[256];
  threadgroup float sx2[256];
  threadgroup float sxy[256];
  threadgroup float sy2[256];

  float ax = 0.0f;
  float ay = 0.0f;
  float ax2 = 0.0f;
  float axy = 0.0f;
  float ay2 = 0.0f;
  for (uint row = tid; row < n; row += threads) {
    float2 v = values[row];
    ax += v.x;
    ay += v.y;
    ax2 += v.x * v.x;
    axy += v.x * v.y;
    ay2 += v.y * v.y;
  }
  sx[tid] = ax;
  sy[tid] = ay;
  sx2[tid] = ax2;
  sxy[tid] = axy;
  sy2[tid] = ay2;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
      sx2[tid] += sx2[tid + stride];
      sxy[tid] += sxy[tid + stride];
      sy2[tid] += sy2[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    float dn = max(float(n), 1.0f);
    float mean_x = sx[0] / dn;
    float mean_y = sy[0] / dn;
    float x_center_ss = max(sx2[0] - sx[0] * sx[0] / dn, 1.0e-24f);
    float y_center_ss = max(sy2[0] - sy[0] * sy[0] / dn, 1.0e-24f);
    float xy_center = sxy[0] - sx[0] * sy[0] / dn;
    float norm_x = sqrt(x_center_ss);
    if (!isfinite(norm_x) || norm_x <= 0.0f) norm_x = 1.0f;
    float proj_y_on_x = xy_center / norm_x;
    float y_resid_ss = max(y_center_ss - proj_y_on_x * proj_y_on_x, 1.0e-24f);
    float norm_y = sqrt(y_resid_ss);
    if (!isfinite(norm_y) || norm_y <= 0.0f) norm_y = 1.0f;
    stats[0] = mean_x;
    stats[1] = mean_y;
    stats[2] = 1.0f / norm_x;
    stats[3] = proj_y_on_x;
    stats[4] = 1.0f / norm_y;
  }
}

kernel void spectral_normalize(
  device float2* values [[buffer(0)]],
  device const float* stats [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  float2 v = values[row];
  float x = (v.x - stats[0]) * stats[2];
  float y_centered = v.y - stats[1];
  float y = (y_centered - stats[3] * x) * stats[4];
  values[row] = float2(x, y);
}

#define FASTEMBEDR_METAL_PROJECTION_MAX_K 128
#define FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS 12
#define FASTEMBEDR_METAL_SCORE_MAX_K 64
#define FASTEMBEDR_METAL_SCORE_WIDTH 6
#define FASTEMBEDR_METAL_MAX_LABELS 128

float median_local(thread float* values, int count) {
  if (count <= 0) return 0.0f;
  for (int i = 1; i < count; ++i) {
    float v = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > v) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = v;
  }
  int mid = count / 2;
  if ((count & 1) != 0) return values[mid];
  return 0.5f * (values[mid - 1] + values[mid]);
}

bool pair_less_metal(float d, int idx, float other_d, int other_idx) {
  if (d < other_d) return true;
  if (d > other_d) return false;
  return idx < other_idx;
}

void insert_top_neighbor_metal(thread float* distances,
                               thread int* indices,
                               thread int& count,
                               int k,
                               float distance,
                               int index) {
  if (count < k) {
    int pos = count;
    ++count;
    while (pos > 0 && pair_less_metal(distance, index, distances[pos - 1], indices[pos - 1])) {
      distances[pos] = distances[pos - 1];
      indices[pos] = indices[pos - 1];
      --pos;
    }
    distances[pos] = distance;
    indices[pos] = index;
    return;
  }
  if (!pair_less_metal(distance, index, distances[k - 1], indices[k - 1])) return;
  int pos = k - 1;
  while (pos > 0 && pair_less_metal(distance, index, distances[pos - 1], indices[pos - 1])) {
    distances[pos] = distances[pos - 1];
    indices[pos] = indices[pos - 1];
    --pos;
  }
  distances[pos] = distance;
  indices[pos] = index;
}

float layout_d2_2d_metal(device const float* layout, uint n, int a, int b) {
  float dx = layout[uint(a)] - layout[uint(b)];
  float dy = layout[n + uint(a)] - layout[n + uint(b)];
  return dx * dx + dy * dy;
}

int high_rank_metal(device const int* indices,
                    uint index_rows,
                    int index_row,
                    int candidate,
                    int high_rank_limit) {
  for (int r = 0; r < high_rank_limit; ++r) {
    if (indices[uint(r) * index_rows + uint(index_row)] - 1 == candidate) return r + 1;
  }
  return high_rank_limit + 1;
}

kernel void standardize_stats(
  device const float* values [[buffer(0)]],
  device float* centers [[buffer(1)]],
  device float* scales [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  constant uint& threads [[buffer(4)]],
  uint col [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  threadgroup float sums[256];
  threadgroup float sums2[256];

  float sum = 0.0f;
  float sum2 = 0.0f;
  uint base = col * n;
  for (uint row = tid; row < n; row += threads) {
    float x = values[base + row];
    sum += x;
    sum2 += x * x;
  }
  sums[tid] = sum;
  sums2[tid] = sum2;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
      sums2[tid] += sums2[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    float dn = max(float(n), 1.0f);
    float mean = sums[0] / dn;
    float denom = max(float(n > 0u ? n - 1u : 1u), 1.0f);
    float variance = (sums2[0] - sums[0] * sums[0] / dn) / denom;
    if (!isfinite(variance) || variance <= 0.0f) variance = 1.0f;
    float scale = sqrt(variance);
    if (!isfinite(scale) || scale <= 0.0f) scale = 1.0f;
    centers[col] = mean;
    scales[col] = scale;
  }
}

kernel void standardize_apply(
  device float* values [[buffer(0)]],
  device const float* centers [[buffer(1)]],
  device const float* scales [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  constant uint& total [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= total) return;
  uint col = gid / n;
  values[gid] = (values[gid] - centers[col]) / scales[col];
}

kernel void project_membership(
  device const float* reference_layout [[buffer(0)]],
  device const int* projection_indices [[buffer(1)]],
  device const float* projection_distances [[buffer(2)]],
  device float* out [[buffer(3)]],
  constant uint& n_reference [[buffer(4)]],
  constant uint& n_query [[buffer(5)]],
  constant uint& k [[buffer(6)]],
  constant uint& n_components [[buffer(7)]],
  constant uint& average_zeros [[buffer(8)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_query) return;

  constexpr float eps = 1.4901161193847656e-8f;
  constexpr float inf = 3.4028234663852886e+38f;
  float adjusted[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float scratch[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  int zero_count = 0;
  int first_zero = -1;
  float rho = inf;

  for (uint j = 0u; j < k; ++j) {
    int idx = projection_indices[j * n_query + row] - 1;
    float d = projection_distances[j * n_query + row];
    if (!isfinite(d) || d < 0.0f) d = 0.0f;
    if (idx < 0 || uint(idx) >= n_reference) d = inf;
    if (d <= eps) {
      ++zero_count;
      if (first_zero < 0) first_zero = int(j);
    }
    if (d < rho) rho = d;
  }

  if (zero_count > 0) {
    float inv_zero = average_zeros != 0u ? 1.0f / float(zero_count) : 1.0f;
    for (uint c = 0u; c < n_components; ++c) {
      float value = 0.0f;
      for (uint j = 0u; j < k; ++j) {
        float d = projection_distances[j * n_query + row];
        if (!isfinite(d) || d < 0.0f) d = 0.0f;
        if ((average_zeros != 0u && d <= eps) || (average_zeros == 0u && int(j) == first_zero)) {
          int idx = projection_indices[j * n_query + row] - 1;
          if (idx >= 0 && uint(idx) < n_reference) {
            value += inv_zero * reference_layout[c * n_reference + uint(idx)];
          }
        }
      }
      out[c * n_query + row] = value;
    }
    return;
  }

  int positive_count = 0;
  for (uint j = 0u; j < k; ++j) {
    float d = projection_distances[j * n_query + row];
    if (!isfinite(d) || d < 0.0f) d = 0.0f;
    float value = max(0.0f, d - rho);
    adjusted[j] = value;
    if (value > eps) scratch[positive_count++] = value;
  }
  if (positive_count == 0) {
    for (uint j = 0u; j < k; ++j) {
      float d = projection_distances[j * n_query + row];
      if (!isfinite(d) || d < 0.0f) d = 0.0f;
      scratch[j] = d;
    }
    positive_count = int(k);
  }

  float sigma = median_local(scratch, positive_count);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  float weight_sum = 0.0f;
  for (uint j = 0u; j < k; ++j) {
    float w = exp(-adjusted[j] / sigma);
    adjusted[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0f) {
    weight_sum = float(k);
    for (uint j = 0u; j < k; ++j) adjusted[j] = 1.0f;
  }

  for (uint c = 0u; c < n_components; ++c) {
    float value = 0.0f;
    for (uint j = 0u; j < k; ++j) {
      int idx = projection_indices[j * n_query + row] - 1;
      if (idx >= 0 && uint(idx) < n_reference) {
        value += adjusted[j] * reference_layout[c * n_reference + uint(idx)];
      }
    }
    out[c * n_query + row] = value / weight_sum;
  }
}

kernel void tsne_transform_epoch(
  device const float* reference_layout [[buffer(0)]],
  device const int* indices [[buffer(1)]],
  device const float* probabilities [[buffer(2)]],
  device float2* current [[buffer(3)]],
  device float2* gains [[buffer(4)]],
  device float2* updates [[buffer(5)]],
  constant TsneTransformParams& p [[buffer(6)]],
  constant uint& epoch [[buffer(7)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n_query) return;

  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  float2 grad = float2(0.0f, 0.0f);
  float sum_q = eps;

  if (p.exact_repulsion != 0u || p.n_negatives >= p.n_reference) {
    for (uint ref = 0u; ref < p.n_reference; ++ref) {
      float2 yr = float2(reference_layout[ref], reference_layout[p.n_reference + ref]);
      float2 diff = yi - yr;
      float d2 = dot(diff, diff);
      float q = 1.0f / (1.0f + d2);
      sum_q += q;
    }
    for (uint ref = 0u; ref < p.n_reference; ++ref) {
      float2 yr = float2(reference_layout[ref], reference_layout[p.n_reference + ref]);
      float2 diff = yi - yr;
      float d2 = dot(diff, diff);
      float q = 1.0f / (1.0f + d2);
      float coeff = -(q * q) / sum_q;
      grad += coeff * diff;
    }
  } else {
    for (uint sample = 0u; sample < p.n_negatives; ++sample) {
      uint ref = deterministic_reference(p.n_reference, p.seed, epoch, row, sample);
      float2 yr = float2(reference_layout[ref], reference_layout[p.n_reference + ref]);
      float2 diff = yi - yr;
      float d2 = dot(diff, diff);
      float q = 1.0f / (1.0f + d2);
      sum_q += q;
    }
    for (uint sample = 0u; sample < p.n_negatives; ++sample) {
      uint ref = deterministic_reference(p.n_reference, p.seed, epoch, row, sample);
      float2 yr = float2(reference_layout[ref], reference_layout[p.n_reference + ref]);
      float2 diff = yi - yr;
      float d2 = dot(diff, diff);
      float q = 1.0f / (1.0f + d2);
      float coeff = -(q * q) / sum_q;
      grad += coeff * diff;
    }
  }

  for (uint j = 0u; j < p.k; ++j) {
    uint pos = j * p.n_query + row;
    int ref_i = indices[pos];
    if (ref_i < 0 || uint(ref_i) >= p.n_reference) continue;
    uint ref = uint(ref_i);
    float2 yr = float2(reference_layout[ref], reference_layout[p.n_reference + ref]);
    float2 diff = yi - yr;
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    float coeff = p.exaggeration * probabilities[pos] * q;
    grad += coeff * diff;
  }

  float grad_norm2 = dot(grad, grad);
  float max_grad2 = p.max_grad_norm * p.max_grad_norm;
  if (isfinite(max_grad2) && max_grad2 > 0.0f && grad_norm2 > max_grad2) {
    grad *= p.max_grad_norm / (sqrt(grad_norm2) + eps);
  }

  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + 0.01f;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + 0.01f;
  gain = max(gain, float2(0.01f, 0.01f));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
  }

  current[row] = yi + update;
  gains[row] = gain;
  updates[row] = update;
}

kernel void opentsne_sum_q_rows(
  device const float2* current [[buffer(0)]],
  device float* row_sums [[buffer(1)]],
  constant OpenTsneMetalParams& p [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  float2 yi = current[row];
  float sum_q = 0.0f;
  for (uint j = 0u; j < p.n; ++j) {
    if (j == row) continue;
    float2 diff = yi - current[j];
    float d2 = dot(diff, diff);
    sum_q += 1.0f / (1.0f + d2);
  }
  row_sums[row] = sum_q;
}

kernel void opentsne_epoch_exact(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device float2* current [[buffer(3)]],
  device float2* gains [[buffer(4)]],
  device float2* updates [[buffer(5)]],
  constant OpenTsneMetalParams& p [[buffer(6)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  float2 grad = float2(0.0f, 0.0f);

  for (uint j = 0u; j < p.n; ++j) {
    if (j == row) continue;
    float2 diff = yi - current[j];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (-(q * q) * p.inv_sum_q) * diff;
  }

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
  }

  current[row] = yi + update;
  gains[row] = gain;
  updates[row] = update;
}

kernel void opentsne_apply_center(
  device float2* current [[buffer(0)]],
  constant float2& center [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  current[row] -= center;
}

float2 complex_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint reverse_bits_limited(uint value, uint n_bits) {
  uint out = 0u;
  for (uint bit = 0u; bit < n_bits; ++bit) {
    out = (out << 1u) | (value & 1u);
    value >>= 1u;
  }
  return out;
}

kernel void opentsne_fft_clear_grids(
  device atomic_float* mass [[buffer(0)]],
  device atomic_float* mass_x [[buffer(1)]],
  device atomic_float* mass_y [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  uint cell [[thread_position_in_grid]]
) {
  uint total = g.grid_size * g.grid_size;
  if (cell >= total) return;
  atomic_store_explicit(&mass[cell], 0.0f, memory_order_relaxed);
  atomic_store_explicit(&mass_x[cell], 0.0f, memory_order_relaxed);
  atomic_store_explicit(&mass_y[cell], 0.0f, memory_order_relaxed);
}

kernel void opentsne_fft_scatter_bilinear(
  device const float2* current [[buffer(0)]],
  device atomic_float* mass [[buffer(1)]],
  device atomic_float* mass_x [[buffer(2)]],
  device atomic_float* mass_y [[buffer(3)]],
  constant OpenTsneFFTGridParams& g [[buffer(4)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  float raw_x = (yi.x - g.lower_x) * g.inv_spacing;
  float raw_y = (yi.y - g.lower_y) * g.inv_spacing;
  float max_cell = float(g.grid_size - 1u);
  raw_x = clamp(raw_x, 0.0f, max_cell);
  raw_y = clamp(raw_y, 0.0f, max_cell);
  uint x0 = min(uint(floor(raw_x)), g.grid_size - 2u);
  uint y0 = min(uint(floor(raw_y)), g.grid_size - 2u);
  uint x1 = x0 + 1u;
  uint y1 = y0 + 1u;
  float tx = raw_x - float(x0);
  float ty = raw_y - float(y0);
  float w00 = (1.0f - tx) * (1.0f - ty);
  float w10 = tx * (1.0f - ty);
  float w01 = (1.0f - tx) * ty;
  float w11 = tx * ty;
  uint p00 = y0 * g.grid_size + x0;
  uint p10 = y0 * g.grid_size + x1;
  uint p01 = y1 * g.grid_size + x0;
  uint p11 = y1 * g.grid_size + x1;
  atomic_fetch_add_explicit(&mass[p00], w00, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass[p10], w10, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass[p01], w01, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass[p11], w11, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_x[p00], w00 * yi.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_x[p10], w10 * yi.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_x[p01], w01 * yi.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_x[p11], w11 * yi.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_y[p00], w00 * yi.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_y[p10], w10 * yi.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_y[p01], w01 * yi.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&mass_y[p11], w11 * yi.y, memory_order_relaxed);
}

kernel void opentsne_fft_load_inputs(
  device const atomic_float* mass [[buffer(0)]],
  device const atomic_float* mass_x [[buffer(1)]],
  device const atomic_float* mass_y [[buffer(2)]],
  device float2* mass_fft [[buffer(3)]],
  device float2* mass_x_fft [[buffer(4)]],
  device float2* mass_y_fft [[buffer(5)]],
  device float2* kernel_q [[buffer(6)]],
  device float2* kernel_q2 [[buffer(7)]],
  constant OpenTsneFFTGridParams& g [[buffer(8)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= g.fft_size || gid.y >= g.fft_size) return;
  uint fft_pos = gid.y * g.fft_size + gid.x;
  float m = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;
  if (gid.x < g.grid_size && gid.y < g.grid_size) {
    uint grid_pos = gid.y * g.grid_size + gid.x;
    m = atomic_load_explicit(&mass[grid_pos], memory_order_relaxed);
    mx = atomic_load_explicit(&mass_x[grid_pos], memory_order_relaxed);
    my = atomic_load_explicit(&mass_y[grid_pos], memory_order_relaxed);
  }
  mass_fft[fft_pos] = float2(m, 0.0f);
  mass_x_fft[fft_pos] = float2(mx, 0.0f);
  mass_y_fft[fft_pos] = float2(my, 0.0f);

  bool x_ok = gid.x < g.grid_size || gid.x > g.grid_size;
  bool y_ok = gid.y < g.grid_size || gid.y > g.grid_size;
  float q = 0.0f;
  float q2 = 0.0f;
  if (x_ok && y_ok) {
    int dx = gid.x < g.grid_size ? int(gid.x) : int(gid.x) - int(g.fft_size);
    int dy = gid.y < g.grid_size ? int(gid.y) : int(gid.y) - int(g.fft_size);
    if (abs(dx) < int(g.grid_size) && abs(dy) < int(g.grid_size)) {
      float x_offset = float(dx) * g.spacing;
      float y_offset = float(dy) * g.spacing;
      float d2 = x_offset * x_offset + y_offset * y_offset;
      q = 1.0f / (1.0f + d2);
      q2 = q * q;
    }
  }
  kernel_q[fft_pos] = float2(q, 0.0f);
  kernel_q2[fft_pos] = float2(q2, 0.0f);
}

kernel void opentsne_mpsgraph_load_real_inputs(
  device const atomic_float* mass [[buffer(0)]],
  device const atomic_float* mass_x [[buffer(1)]],
  device const atomic_float* mass_y [[buffer(2)]],
  device float* mass_real [[buffer(3)]],
  device float* mass_x_real [[buffer(4)]],
  device float* mass_y_real [[buffer(5)]],
  device float* kernel_q_real [[buffer(6)]],
  device float* kernel_q2_real [[buffer(7)]],
  constant OpenTsneFFTGridParams& g [[buffer(8)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= g.fft_size || gid.y >= g.fft_size) return;
  uint fft_pos = gid.y * g.fft_size + gid.x;
  float m = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;
  if (gid.x < g.grid_size && gid.y < g.grid_size) {
    uint grid_pos = gid.y * g.grid_size + gid.x;
    m = atomic_load_explicit(&mass[grid_pos], memory_order_relaxed);
    mx = atomic_load_explicit(&mass_x[grid_pos], memory_order_relaxed);
    my = atomic_load_explicit(&mass_y[grid_pos], memory_order_relaxed);
  }
  mass_real[fft_pos] = m;
  mass_x_real[fft_pos] = mx;
  mass_y_real[fft_pos] = my;

  bool x_ok = gid.x < g.grid_size || gid.x > g.grid_size;
  bool y_ok = gid.y < g.grid_size || gid.y > g.grid_size;
  float q = 0.0f;
  float q2 = 0.0f;
  if (x_ok && y_ok) {
    int dx = gid.x < g.grid_size ? int(gid.x) : int(gid.x) - int(g.fft_size);
    int dy = gid.y < g.grid_size ? int(gid.y) : int(gid.y) - int(g.fft_size);
    if (abs(dx) < int(g.grid_size) && abs(dy) < int(g.grid_size)) {
      float x_offset = float(dx) * g.spacing;
      float y_offset = float(dy) * g.spacing;
      float d2 = x_offset * x_offset + y_offset * y_offset;
      q = 1.0f / (1.0f + d2);
      q2 = q * q;
    }
  }
  kernel_q_real[fft_pos] = q;
  kernel_q2_real[fft_pos] = q2;
}

kernel void opentsne_fft_pack_real_to_complex4(
  device const float* q_real [[buffer(0)]],
  device const float* q2_real [[buffer(1)]],
  device const float* xq2_real [[buffer(2)]],
  device const float* yq2_real [[buffer(3)]],
  device float2* q_complex [[buffer(4)]],
  device float2* q2_complex [[buffer(5)]],
  device float2* xq2_complex [[buffer(6)]],
  device float2* yq2_complex [[buffer(7)]],
  constant uint& n_total [[buffer(8)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_total) return;
  q_complex[row] = float2(q_real[row], 0.0f);
  q2_complex[row] = float2(q2_real[row], 0.0f);
  xq2_complex[row] = float2(xq2_real[row], 0.0f);
  yq2_complex[row] = float2(yq2_real[row], 0.0f);
}

kernel void opentsne_fft_bit_reverse_rows(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& n_fft [[buffer(2)]],
  constant uint& log_n [[buffer(3)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n_fft || gid.y >= n_fft) return;
  uint rev = reverse_bits_limited(gid.x, log_n);
  output[gid.y * n_fft + rev] = input[gid.y * n_fft + gid.x];
}

kernel void opentsne_fft_bit_reverse_cols(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& n_fft [[buffer(2)]],
  constant uint& log_n [[buffer(3)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n_fft || gid.y >= n_fft) return;
  uint rev = reverse_bits_limited(gid.y, log_n);
  output[rev * n_fft + gid.x] = input[gid.y * n_fft + gid.x];
}

kernel void opentsne_fft_butterfly_rows(
  device float2* values [[buffer(0)]],
  constant uint& n_fft [[buffer(1)]],
  constant uint& stage [[buffer(2)]],
  constant uint& inverse [[buffer(3)]],
  device const float2* twiddles [[buffer(4)]],
  uint2 gid [[thread_position_in_grid]]
) {
  uint half_count = n_fft >> 1u;
  if (gid.x >= half_count || gid.y >= n_fft) return;
  uint span_half = 1u << (stage - 1u);
  uint width = span_half << 1u;
  uint group = gid.x / span_half;
  uint j = gid.x - group * span_half;
  uint base = gid.y * n_fft + group * width + j;
  float2 w = twiddles[(stage - 1u) * half_count + j];
  if (inverse != 0u) w.y = -w.y;
  float2 u = values[base];
  float2 v = complex_mul(values[base + span_half], w);
  values[base] = u + v;
  values[base + span_half] = u - v;
}

kernel void opentsne_fft_butterfly_cols(
  device float2* values [[buffer(0)]],
  constant uint& n_fft [[buffer(1)]],
  constant uint& stage [[buffer(2)]],
  constant uint& inverse [[buffer(3)]],
  device const float2* twiddles [[buffer(4)]],
  uint2 gid [[thread_position_in_grid]]
) {
  uint half_count = n_fft >> 1u;
  if (gid.x >= n_fft || gid.y >= half_count) return;
  uint span_half = 1u << (stage - 1u);
  uint width = span_half << 1u;
  uint group = gid.y / span_half;
  uint j = gid.y - group * span_half;
  uint row0 = group * width + j;
  uint idx0 = row0 * n_fft + gid.x;
  uint idx1 = (row0 + span_half) * n_fft + gid.x;
  float2 w = twiddles[(stage - 1u) * half_count + j];
  if (inverse != 0u) w.y = -w.y;
  float2 u = values[idx0];
  float2 v = complex_mul(values[idx1], w);
  values[idx0] = u + v;
  values[idx1] = u - v;
}

kernel void opentsne_fft_multiply(
  device const float2* a [[buffer(0)]],
  device const float2* b [[buffer(1)]],
  device float2* out [[buffer(2)]],
  constant uint& total [[buffer(3)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= total) return;
  out[gid] = complex_mul(a[gid], b[gid]);
}

kernel void opentsne_fft_scale(
  device float2* values [[buffer(0)]],
  constant uint& total [[buffer(1)]],
  constant float& scale [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= total) return;
  values[gid] *= scale;
}

// Stockham 512 kernels adapted for fastEmbedR from the MIT-licensed
// AppleSiliconFFT radix-4 Stockham design by Mohamed Amine Bergach. They are
// used only for validated 512x512 openTSNE FFT grids; other sizes stay on the
// generic Cooley-Tukey Metal path.
inline void opentsne_fft_radix4(thread float2& x0, thread float2& x1,
                                thread float2& x2, thread float2& x3,
                                bool inverse) {
  float2 t0 = x0 + x2;
  float2 t1 = x1 + x3;
  float2 t2 = x0 - x2;
  float2 t3 = x1 - x3;
  float2 t3r = inverse ? float2(t3.y, -t3.x) : float2(-t3.y, t3.x);
  x0 = t0 + t1;
  x1 = t2 + t3r;
  x2 = t0 - t1;
  x3 = t2 - t3r;
}

inline void opentsne_fft_radix2(thread float2& x0, thread float2& x1) {
  float2 t = x0;
  x0 = t + x1;
  x1 = t - x1;
}

inline void opentsne_fft_apply_twiddle3(thread float2& x1,
                                        thread float2& x2,
                                        thread float2& x3,
                                        float2 w1) {
  float2 w2 = complex_mul(w1, w1);
  float2 w3 = complex_mul(w2, w1);
  x1 = complex_mul(x1, w1);
  x2 = complex_mul(x2, w2);
  x3 = complex_mul(x3, w3);
}

inline void opentsne_fft_stockham512_core(device const float2* input,
                                          device float2* output,
                                          threadgroup float2* buf,
                                          uint tid,
                                          uint lane,
                                          bool column_major,
                                          bool inverse) {
  constexpr uint N = 512u;
  constexpr uint T = 128u;
  float sign = inverse ? -2.0f : 2.0f;
  float two_pi_over_n = sign * M_PI_F / float(N);

  {
    uint off0 = tid;
    uint off1 = tid + T;
    uint off2 = tid + 2u * T;
    uint off3 = tid + 3u * T;
    float2 x0 = column_major ? input[off0 * N + lane] : input[lane * N + off0];
    float2 x1 = column_major ? input[off1 * N + lane] : input[lane * N + off1];
    float2 x2 = column_major ? input[off2 * N + lane] : input[lane * N + off2];
    float2 x3 = column_major ? input[off3 * N + lane] : input[lane * N + off3];
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = tid << 2u;
    buf[wr] = x0;
    buf[wr + 1u] = x1;
    buf[wr + 2u] = x2;
    buf[wr + 3u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 3u;
    uint grp = tid >> 2u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 32u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 4u] = x1;
    buf[wr + 8u] = x2;
    buf[wr + 12u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 15u;
    uint grp = tid >> 4u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 8u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 16u] = x1;
    buf[wr + 32u] = x2;
    buf[wr + 48u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 63u;
    uint grp = tid >> 6u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 2u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 64u] = x1;
    buf[wr + 128u] = x2;
    buf[wr + 192u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint b = 0u; b < 2u; ++b) {
    uint j = tid + b * T;
    float2 x0 = buf[j];
    float2 x1 = buf[j + 256u];
    float a1 = two_pi_over_n * float(j);
    float c1;
    float s1 = sincos(a1, c1);
    x1 = complex_mul(x1, float2(c1, s1));
    opentsne_fft_radix2(x0, x1);
    uint off0 = j;
    uint off1 = j + 256u;
    if (column_major) {
      output[off0 * N + lane] = x0;
      output[off1 * N + lane] = x1;
    } else {
      output[lane * N + off0] = x0;
      output[lane * N + off1] = x1;
    }
  }
}

kernel void opentsne_fft_512_rows_stockham(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& inverse_u [[buffer(2)]],
  uint tid [[thread_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]
) {
  threadgroup float2 buf[512];
  opentsne_fft_stockham512_core(input, output, buf, tid, row, false, inverse_u != 0u);
}

kernel void opentsne_fft_512_cols_stockham(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& inverse_u [[buffer(2)]],
  uint tid [[thread_index_in_threadgroup]],
  uint col [[threadgroup_position_in_grid]]
) {
  threadgroup float2 buf[512];
  opentsne_fft_stockham512_core(input, output, buf, tid, col, true, inverse_u != 0u);
}

float opentsne_sample_grid_value(device const float2* grid,
                                 constant OpenTsneFFTGridParams& g,
                                 float2 yi) {
  float raw_x = (yi.x - g.lower_x) * g.inv_spacing;
  float raw_y = (yi.y - g.lower_y) * g.inv_spacing;
  float max_cell = float(g.grid_size - 1u);
  raw_x = clamp(raw_x, 0.0f, max_cell);
  raw_y = clamp(raw_y, 0.0f, max_cell);
  uint x0 = min(uint(floor(raw_x)), g.grid_size - 2u);
  uint y0 = min(uint(floor(raw_y)), g.grid_size - 2u);
  uint x1 = x0 + 1u;
  uint y1 = y0 + 1u;
  float tx = raw_x - float(x0);
  float ty = raw_y - float(y0);
  float v00 = grid[y0 * g.fft_size + x0].x;
  float v10 = grid[y0 * g.fft_size + x1].x;
  float v01 = grid[y1 * g.fft_size + x0].x;
  float v11 = grid[y1 * g.fft_size + x1].x;
  return (1.0f - tx) * (1.0f - ty) * v00 +
    tx * (1.0f - ty) * v10 +
    (1.0f - tx) * ty * v01 +
    tx * ty * v11;
}

kernel void opentsne_fft_sum_q_rows(
  device const float2* current [[buffer(0)]],
  device const float2* q_grid [[buffer(1)]],
  device float* row_sums [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  row_sums[row] = opentsne_sample_grid_value(q_grid, g, yi);
}

kernel void opentsne_fft_sum_q_blocks(
  device const float2* current [[buffer(0)]],
  device const float2* q_grid [[buffer(1)]],
  device float* block_sums [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  constant uint& block_size [[buffer(4)]],
  uint block_id [[thread_position_in_grid]]
) {
  uint begin = block_id * block_size;
  if (begin >= g.n) return;
  uint end = min(g.n, begin + block_size);
  float sum = 0.0f;
  for (uint row = begin; row < end; ++row) {
    float2 yi = current[row];
    if (!isfinite(yi.x)) yi.x = 0.0f;
    if (!isfinite(yi.y)) yi.y = 0.0f;
    sum += opentsne_sample_grid_value(q_grid, g, yi);
  }
  block_sums[block_id] = sum;
}

kernel void opentsne_fft_finalize_sum_q(
  device const float* block_sums [[buffer(0)]],
  device float* inv_sum_q [[buffer(1)]],
  constant uint& block_count [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid > 0u) return;
  float sum_q = -float(n);
  for (uint i = 0; i < block_count; ++i) {
    sum_q += block_sums[i];
  }
  if (!isfinite(sum_q) || sum_q <= 0.0f) {
    sum_q = 1.1754943508222875e-38f;
  }
  inv_sum_q[0] = 1.0f / sum_q;
}

kernel void opentsne_fft_layout_stats_blocks(
  device const float2* current [[buffer(0)]],
  device OpenTsneLayoutStats* block_stats [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  constant uint& block_size [[buffer(3)]],
  uint block_id [[thread_position_in_grid]]
) {
  uint begin = block_id * block_size;
  if (begin >= n) return;
  uint end = min(n, begin + block_size);
  float min_x = INFINITY;
  float max_x = -INFINITY;
  float min_y = INFINITY;
  float max_y = -INFINITY;
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  for (uint row = begin; row < end; ++row) {
    float2 yi = current[row];
    if (!isfinite(yi.x)) yi.x = 0.0f;
    if (!isfinite(yi.y)) yi.y = 0.0f;
    min_x = min(min_x, yi.x);
    max_x = max(max_x, yi.x);
    min_y = min(min_y, yi.y);
    max_y = max(max_y, yi.y);
    sum_x += yi.x;
    sum_y += yi.y;
  }
  block_stats[block_id] = OpenTsneLayoutStats{min_x, max_x, min_y, max_y, sum_x, sum_y};
}

kernel void opentsne_fft_finalize_layout_stats(
  device const OpenTsneLayoutStats* block_stats [[buffer(0)]],
  device OpenTsneFFTGridParams* grid_params_out [[buffer(1)]],
  device float2* center_out [[buffer(2)]],
  constant uint& block_count [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& grid_size [[buffer(5)]],
  constant uint& fft_size [[buffer(6)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid > 0u) return;
  float min_x = INFINITY;
  float max_x = -INFINITY;
  float min_y = INFINITY;
  float max_y = -INFINITY;
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  for (uint i = 0u; i < block_count; ++i) {
    OpenTsneLayoutStats s = block_stats[i];
    min_x = min(min_x, s.min_x);
    max_x = max(max_x, s.max_x);
    min_y = min(min_y, s.min_y);
    max_y = max(max_y, s.max_y);
    sum_x += s.sum_x;
    sum_y += s.sum_y;
  }
  float cx = 0.5f * (min_x + max_x);
  float cy = 0.5f * (min_y + max_y);
  float span = max(max_x - min_x, max_y - min_y);
  if (!isfinite(span) || span <= 0.0f) span = 1.0f;
  float half_span = 0.55f * span + 1.0e-3f;
  float spacing = (2.0f * half_span) / float(max(2u, grid_size) - 1u);
  if (!isfinite(spacing) || spacing <= 0.0f) spacing = 1.0f;
  grid_params_out[0] = OpenTsneFFTGridParams{
    n,
    grid_size,
    fft_size,
    cx - half_span,
    cy - half_span,
    1.0f / spacing,
    spacing,
    1.0f
  };
  float inv_n = n > 0u ? 1.0f / float(n) : 0.0f;
  center_out[0] = float2(sum_x * inv_n, sum_y * inv_n);
}

kernel void opentsne_epoch_fft_grid(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device float2* current [[buffer(3)]],
  device float2* gains [[buffer(4)]],
  device float2* updates [[buffer(5)]],
  device const float2* q2_grid [[buffer(6)]],
  device const float2* xq2_grid [[buffer(7)]],
  device const float2* yq2_grid [[buffer(8)]],
  constant OpenTsneMetalParams& p [[buffer(9)]],
  constant OpenTsneFFTGridParams& g [[buffer(10)]],
  device const float* inv_sum_q_device [[buffer(11)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  float q2_value = opentsne_sample_grid_value(q2_grid, g, yi);
  float xq2_value = opentsne_sample_grid_value(xq2_grid, g, yi);
  float yq2_value = opentsne_sample_grid_value(yq2_grid, g, yi);
  float inv_sum_q = inv_sum_q_device[0];
  float2 grad = float2(
    -(yi.x * q2_value - xq2_value) * inv_sum_q,
    -(yi.y * q2_value - yq2_value) * inv_sum_q
  );

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
  }

  current[row] = yi + update;
  gains[row] = gain;
  updates[row] = update;
}

kernel void opentsne_epoch_fft_grid_debug(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device float2* current [[buffer(3)]],
  device float2* gains [[buffer(4)]],
  device float2* updates [[buffer(5)]],
  device const float2* q2_grid [[buffer(6)]],
  device const float2* xq2_grid [[buffer(7)]],
  device const float2* yq2_grid [[buffer(8)]],
  constant OpenTsneMetalParams& p [[buffer(9)]],
  constant OpenTsneFFTGridParams& g [[buffer(10)]],
  device const float* inv_sum_q_device [[buffer(11)]],
  device float* repulsive_norm2 [[buffer(12)]],
  device float* attractive_norm2 [[buffer(13)]],
  device float* gradient_norm2 [[buffer(14)]],
  device float* update_norm2 [[buffer(15)]],
  device float* layout_norm2 [[buffer(16)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;

  float q2_value = opentsne_sample_grid_value(q2_grid, g, yi);
  float xq2_value = opentsne_sample_grid_value(xq2_grid, g, yi);
  float yq2_value = opentsne_sample_grid_value(yq2_grid, g, yi);
  float inv_sum_q = inv_sum_q_device[0];
  float2 repulsive = float2(
    -(yi.x * q2_value - xq2_value) * inv_sum_q,
    -(yi.y * q2_value - yq2_value) * inv_sum_q
  );
  float2 attractive = float2(0.0f, 0.0f);

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    attractive += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 grad = repulsive + attractive;
  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
    step_norm2 = dot(update, update);
  }

  float2 next = yi + update;
  current[row] = next;
  gains[row] = gain;
  updates[row] = update;

  repulsive_norm2[row] = dot(repulsive, repulsive);
  attractive_norm2[row] = dot(attractive, attractive);
  gradient_norm2[row] = dot(grad, grad);
  update_norm2[row] = step_norm2;
  layout_norm2[row] = dot(next, next);
}

void affine_weighted_fallback(
  device const float* reference_layout,
  device const int* projection_indices,
  device const float* projection_distances,
  device float* out,
  device float* confidence_out,
  device int* used_neighbors_out,
  device int* fallback_out,
  uint n_reference,
  uint n_query,
  uint projection_k,
  uint used_neighbors,
  uint row,
  float rho,
  float sigma
) {
  float all_weight_sum = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
  for (uint j = 0u; j < projection_k; ++j) {
    int idx = projection_indices[j * n_query + row] - 1;
    if (idx < 0 || uint(idx) >= n_reference) continue;
    float d = projection_distances[j * n_query + row];
    if (!isfinite(d) || d < 0.0f) continue;
    float adjusted = max(0.0f, d - rho);
    float w = exp(-adjusted / max(sigma, 1.4901161193847656e-8f));
    if (!isfinite(w) || w <= 0.0f) continue;
    all_weight_sum += w;
    y0 += w * reference_layout[uint(idx)];
    y1 += w * reference_layout[n_reference + uint(idx)];
  }
  if (!isfinite(all_weight_sum) || all_weight_sum <= 0.0f) {
    all_weight_sum = 1.0f;
  }
  out[row] = y0 / all_weight_sum;
  out[n_query + row] = y1 / all_weight_sum;
  confidence_out[row] = 0.0f;
  used_neighbors_out[row] = int(used_neighbors);
  fallback_out[row] = 1;
}

kernel void project_embedding_affine_rows(
  device const float* reference_data [[buffer(0)]],
  device const float* query_data [[buffer(1)]],
  device const float* reference_layout [[buffer(2)]],
  device const int* projection_indices [[buffer(3)]],
  device const float* projection_distances [[buffer(4)]],
  device float* out [[buffer(5)]],
  device float* confidence_out [[buffer(6)]],
  device int* used_neighbors_out [[buffer(7)]],
  device int* fallback_out [[buffer(8)]],
  constant uint& n_reference [[buffer(9)]],
  constant uint& n_query [[buffer(10)]],
  constant uint& n_features [[buffer(11)]],
  constant uint& projection_k [[buffer(12)]],
  constant uint& max_neighbors [[buffer(13)]],
  constant float& ridge [[buffer(14)]],
  constant float& max_extrapolation [[buffer(15)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_query) return;

  constexpr float eps = 1.4901161193847656e-8f;
  constexpr float inf = 3.4028234663852886e+38f;
  uint m = min(max(max_neighbors, 3u), min(projection_k, uint(FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS)));

  float distances[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float scratch[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float weights[FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS];
  int refs[FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS];
  float y_center[2];
  float gram[FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS * FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS];
  float rhs[FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS * 2];
  float q[FASTEMBEDR_METAL_AFFINE_MAX_NEIGHBORS];

  int zero_col = -1;
  float rho = inf;
  for (uint j = 0u; j < projection_k; ++j) {
    uint pos = j * n_query + row;
    int idx = projection_indices[pos] - 1;
    float d = projection_distances[pos];
    bool valid = idx >= 0 && uint(idx) < n_reference && isfinite(d) && d >= 0.0f;
    distances[j] = valid ? d : inf;
    if (valid && d <= eps && zero_col < 0) zero_col = int(j);
    if (valid) rho = min(rho, d);
  }
  if (!isfinite(rho)) rho = 0.0f;

  if (zero_col >= 0) {
    int ref = projection_indices[uint(zero_col) * n_query + row] - 1;
    out[row] = reference_layout[uint(ref)];
    out[n_query + row] = reference_layout[n_reference + uint(ref)];
    confidence_out[row] = 1.0f;
    used_neighbors_out[row] = 1;
    fallback_out[row] = 0;
    return;
  }

  int positive_count = 0;
  for (uint j = 0u; j < projection_k; ++j) {
    float adjusted = max(0.0f, distances[j] - rho);
    if (isfinite(adjusted) && adjusted > eps) scratch[positive_count++] = adjusted;
  }
  float sigma = positive_count > 0 ? median_local(scratch, positive_count) : max(rho, eps);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  float weight_sum = 0.0f;
  float weight_sq_sum = 0.0f;
  for (uint j = 0u; j < m; ++j) {
    int idx = projection_indices[j * n_query + row] - 1;
    refs[j] = idx;
    float adjusted = max(0.0f, distances[j] - rho);
    float w = (idx >= 0 && uint(idx) < n_reference && isfinite(adjusted)) ? exp(-adjusted / sigma) : 0.0f;
    weights[j] = w;
    weight_sum += w;
    weight_sq_sum += w * w;
  }

  if (!isfinite(weight_sum) || weight_sum <= 0.0f) {
    affine_weighted_fallback(
      reference_layout, projection_indices, projection_distances, out,
      confidence_out, used_neighbors_out, fallback_out, n_reference, n_query,
      projection_k, m, row, rho, sigma
    );
    return;
  }
  for (uint j = 0u; j < m; ++j) weights[j] /= weight_sum;

  y_center[0] = 0.0f;
  y_center[1] = 0.0f;
  for (uint j = 0u; j < m; ++j) {
    int ref = refs[j];
    if (ref < 0 || uint(ref) >= n_reference) continue;
    float w = weights[j];
    y_center[0] += w * reference_layout[uint(ref)];
    y_center[1] += w * reference_layout[n_reference + uint(ref)];
  }

  float layout_radius_sq = 0.0f;
  float trace = 0.0f;
  for (uint a = 0u; a < m; ++a) {
    int ref_a = refs[a];
    float sqrt_wa = sqrt(max(weights[a], 0.0f));
    float row_norm = 0.0f;
    for (uint f = 0u; f < n_features; ++f) {
      float x_center_f = 0.0f;
      for (uint j = 0u; j < m; ++j) {
        int ref_j = refs[j];
        if (ref_j >= 0 && uint(ref_j) < n_reference) {
          x_center_f += weights[j] * reference_data[uint(ref_j) + f * n_reference];
        }
      }
      float xc = (ref_a >= 0 && uint(ref_a) < n_reference) ?
        reference_data[uint(ref_a) + f * n_reference] - x_center_f :
        0.0f;
      row_norm += xc * xc;
    }
    trace += weights[a] * row_norm;
    float yc0 = (ref_a >= 0 && uint(ref_a) < n_reference) ?
      reference_layout[uint(ref_a)] - y_center[0] :
      0.0f;
    float yc1 = (ref_a >= 0 && uint(ref_a) < n_reference) ?
      reference_layout[n_reference + uint(ref_a)] - y_center[1] :
      0.0f;
    rhs[a * 2u] = sqrt_wa * yc0;
    rhs[a * 2u + 1u] = sqrt_wa * yc1;
    layout_radius_sq = max(layout_radius_sq, yc0 * yc0 + yc1 * yc1);
  }

  for (uint a = 0u; a < m; ++a) {
    int ref_a = refs[a];
    float sqrt_wa = sqrt(max(weights[a], 0.0f));
    for (uint b = 0u; b <= a; ++b) {
      int ref_b = refs[b];
      float sqrt_wb = sqrt(max(weights[b], 0.0f));
      float dot_value = 0.0f;
      for (uint f = 0u; f < n_features; ++f) {
        float x_center_f = 0.0f;
        for (uint j = 0u; j < m; ++j) {
          int ref_j = refs[j];
          if (ref_j >= 0 && uint(ref_j) < n_reference) {
            x_center_f += weights[j] * reference_data[uint(ref_j) + f * n_reference];
          }
        }
        float xa = (ref_a >= 0 && uint(ref_a) < n_reference) ?
          reference_data[uint(ref_a) + f * n_reference] - x_center_f :
          0.0f;
        float xb = (ref_b >= 0 && uint(ref_b) < n_reference) ?
          reference_data[uint(ref_b) + f * n_reference] - x_center_f :
          0.0f;
        dot_value += xa * xb;
      }
      float value = sqrt_wa * sqrt_wb * dot_value;
      gram[a * m + b] = value;
      gram[b * m + a] = value;
    }
  }
  float lambda = ridge * max(trace, eps) + eps;
  for (uint a = 0u; a < m; ++a) gram[a * m + a] += lambda;

  bool ok = true;
  for (uint j = 0u; j < m && ok; ++j) {
    float sum_value = gram[j * m + j];
    for (uint kk = 0u; kk < j; ++kk) {
      float l = gram[j * m + kk];
      sum_value -= l * l;
    }
    if (!isfinite(sum_value) || sum_value <= eps) {
      ok = false;
      break;
    }
    float diag = sqrt(sum_value);
    gram[j * m + j] = diag;
    for (uint i = j + 1u; i < m; ++i) {
      float s = gram[i * m + j];
      for (uint kk = 0u; kk < j; ++kk) s -= gram[i * m + kk] * gram[j * m + kk];
      gram[i * m + j] = s / diag;
    }
  }
  if (!ok) {
    affine_weighted_fallback(
      reference_layout, projection_indices, projection_distances, out,
      confidence_out, used_neighbors_out, fallback_out, n_reference, n_query,
      projection_k, m, row, rho, sigma
    );
    return;
  }

  for (uint c = 0u; c < 2u; ++c) {
    for (uint i = 0u; i < m; ++i) {
      float s = rhs[i * 2u + c];
      for (uint kk = 0u; kk < i; ++kk) s -= gram[i * m + kk] * rhs[kk * 2u + c];
      rhs[i * 2u + c] = s / gram[i * m + i];
    }
    for (int ii = int(m) - 1; ii >= 0; --ii) {
      uint i = uint(ii);
      float s = rhs[i * 2u + c];
      for (uint kk = i + 1u; kk < m; ++kk) s -= gram[kk * m + i] * rhs[kk * 2u + c];
      rhs[i * 2u + c] = s / gram[i * m + i];
    }
  }

  for (uint a = 0u; a < m; ++a) {
    int ref_a = refs[a];
    float sqrt_wa = sqrt(max(weights[a], 0.0f));
    float dot_value = 0.0f;
    for (uint f = 0u; f < n_features; ++f) {
      float x_center_f = 0.0f;
      for (uint j = 0u; j < m; ++j) {
        int ref_j = refs[j];
        if (ref_j >= 0 && uint(ref_j) < n_reference) {
          x_center_f += weights[j] * reference_data[uint(ref_j) + f * n_reference];
        }
      }
      float qa = query_data[row + f * n_query] - x_center_f;
      float xa = (ref_a >= 0 && uint(ref_a) < n_reference) ?
        reference_data[uint(ref_a) + f * n_reference] - x_center_f :
        0.0f;
      dot_value += qa * xa;
    }
    q[a] = sqrt_wa * dot_value;
  }

  float displacement0 = 0.0f;
  float displacement1 = 0.0f;
  for (uint a = 0u; a < m; ++a) {
    displacement0 += q[a] * rhs[a * 2u];
    displacement1 += q[a] * rhs[a * 2u + 1u];
  }
  float y0 = y_center[0] + displacement0;
  float y1 = y_center[1] + displacement1;

  float disp_sq = displacement0 * displacement0 + displacement1 * displacement1;
  float max_disp = max_extrapolation * sqrt(max(layout_radius_sq, eps));
  if (disp_sq > max_disp * max_disp) {
    float scale = max_disp / (sqrt(disp_sq) + eps);
    y0 = y_center[0] + displacement0 * scale;
    y1 = y_center[1] + displacement1 * scale;
  }

  if (!isfinite(y0) || !isfinite(y1)) {
    affine_weighted_fallback(
      reference_layout, projection_indices, projection_distances, out,
      confidence_out, used_neighbors_out, fallback_out, n_reference, n_query,
      projection_k, m, row, rho, sigma
    );
    return;
  }

  out[row] = y0;
  out[n_query + row] = y1;
  float effective_n = weight_sq_sum > 0.0f ? (weight_sum * weight_sum) / weight_sq_sum : 1.0f;
  confidence_out[row] = clamp(effective_n / float(m), 0.0f, 1.0f);
  used_neighbors_out[row] = int(m);
  fallback_out[row] = 0;
}

kernel void landmark_project_interpolate(
  device const float* landmark_data [[buffer(0)]],
  device const float* query_data [[buffer(1)]],
  device const float* landmark_layout [[buffer(2)]],
  device float* out [[buffer(3)]],
  constant uint& n_landmarks [[buffer(4)]],
  constant uint& n_query [[buffer(5)]],
  constant uint& n_features [[buffer(6)]],
  constant uint& k [[buffer(7)]],
  constant uint& n_components [[buffer(8)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_query) return;

  constexpr float eps = 1.4901161193847656e-8f;
  constexpr float inf = 3.4028234663852886e+38f;
  float best_sq[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  int best_idx[FASTEMBEDR_METAL_PROJECTION_MAX_K];

  for (uint j = 0u; j < k; ++j) {
    best_sq[j] = inf;
    best_idx[j] = INT_MAX;
  }

  const uint query_offset = row * n_features;
  for (uint i = 0u; i < n_landmarks; ++i) {
    const uint landmark_offset = i * n_features;
    float dist = 0.0f;
    for (uint c = 0u; c < n_features; ++c) {
      float diff = landmark_data[landmark_offset + c] - query_data[query_offset + c];
      dist += diff * diff;
    }

    if (dist < best_sq[k - 1u] ||
        (dist == best_sq[k - 1u] && int(i) < best_idx[k - 1u])) {
      uint pos = k - 1u;
      while (pos > 0u &&
             (dist < best_sq[pos - 1u] ||
              (dist == best_sq[pos - 1u] && int(i) < best_idx[pos - 1u]))) {
        best_sq[pos] = best_sq[pos - 1u];
        best_idx[pos] = best_idx[pos - 1u];
        --pos;
      }
      best_sq[pos] = dist;
      best_idx[pos] = int(i);
    }
  }

  float distances[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float adjusted[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float scratch[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  int zero_count = 0;
  int first_zero = -1;
  float rho = inf;

  for (uint j = 0u; j < k; ++j) {
    float d = sqrt(max(best_sq[j], 0.0f));
    distances[j] = d;
    if (d <= eps) {
      ++zero_count;
      if (first_zero < 0) first_zero = int(j);
    }
    if (d < rho) rho = d;
  }

  if (zero_count > 0) {
    for (uint c = 0u; c < n_components; ++c) {
      float value = 0.0f;
      int idx = best_idx[uint(first_zero)];
      if (idx >= 0 && uint(idx) < n_landmarks) {
        value = landmark_layout[c * n_landmarks + uint(idx)];
      }
      out[c * n_query + row] = value;
    }
    return;
  }

  int positive_count = 0;
  for (uint j = 0u; j < k; ++j) {
    float value = max(0.0f, distances[j] - rho);
    adjusted[j] = value;
    if (value > eps) scratch[positive_count++] = value;
  }
  if (positive_count == 0) {
    for (uint j = 0u; j < k; ++j) scratch[j] = distances[j];
    positive_count = int(k);
  }

  float sigma = median_local(scratch, positive_count);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  float weight_sum = 0.0f;
  for (uint j = 0u; j < k; ++j) {
    float w = exp(-adjusted[j] / sigma);
    adjusted[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0f) {
    weight_sum = float(k);
    for (uint j = 0u; j < k; ++j) adjusted[j] = 1.0f;
  }

  for (uint c = 0u; c < n_components; ++c) {
    float value = 0.0f;
    for (uint j = 0u; j < k; ++j) {
      int idx = best_idx[j];
      if (idx >= 0 && uint(idx) < n_landmarks) {
        value += adjusted[j] * landmark_layout[c * n_landmarks + uint(idx)];
      }
    }
    out[c * n_query + row] = value / weight_sum;
  }
}

kernel void landmark_project_interpolate_knn_confidence(
  device const float* landmark_data [[buffer(0)]],
  device const float* query_data [[buffer(1)]],
  device const float* landmark_layout [[buffer(2)]],
  device float* out [[buffer(3)]],
  device int* out_indices [[buffer(4)]],
  device float* out_distances [[buffer(5)]],
  device float* out_confidence [[buffer(6)]],
  constant uint& n_landmarks [[buffer(7)]],
  constant uint& n_query [[buffer(8)]],
  constant uint& n_features [[buffer(9)]],
  constant uint& k [[buffer(10)]],
  constant uint& n_components [[buffer(11)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_query) return;

  constexpr float eps = 1.4901161193847656e-8f;
  constexpr float inf = 3.4028234663852886e+38f;
  float best_sq[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  int best_idx[FASTEMBEDR_METAL_PROJECTION_MAX_K];

  for (uint j = 0u; j < k; ++j) {
    best_sq[j] = inf;
    best_idx[j] = INT_MAX;
  }

  const uint query_offset = row * n_features;
  for (uint i = 0u; i < n_landmarks; ++i) {
    const uint landmark_offset = i * n_features;
    float dist = 0.0f;
    for (uint c = 0u; c < n_features; ++c) {
      float diff = landmark_data[landmark_offset + c] - query_data[query_offset + c];
      dist += diff * diff;
    }

    if (dist < best_sq[k - 1u] ||
        (dist == best_sq[k - 1u] && int(i) < best_idx[k - 1u])) {
      uint pos = k - 1u;
      while (pos > 0u &&
             (dist < best_sq[pos - 1u] ||
              (dist == best_sq[pos - 1u] && int(i) < best_idx[pos - 1u]))) {
        best_sq[pos] = best_sq[pos - 1u];
        best_idx[pos] = best_idx[pos - 1u];
        --pos;
      }
      best_sq[pos] = dist;
      best_idx[pos] = int(i);
    }
  }

  float distances[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float adjusted[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  float scratch[FASTEMBEDR_METAL_PROJECTION_MAX_K];
  int zero_count = 0;
  int first_zero = -1;
  float rho = inf;

  for (uint j = 0u; j < k; ++j) {
    float d = sqrt(max(best_sq[j], 0.0f));
    distances[j] = d;
    out_indices[j * n_query + row] = best_idx[j] + 1;
    out_distances[j * n_query + row] = d;
    if (d <= eps) {
      ++zero_count;
      if (first_zero < 0) first_zero = int(j);
    }
    if (d < rho) rho = d;
  }

  if (zero_count > 0) {
    out_confidence[row] = 1.0f;
    for (uint c = 0u; c < n_components; ++c) {
      float value = 0.0f;
      int idx = best_idx[uint(first_zero)];
      if (idx >= 0 && uint(idx) < n_landmarks) {
        value = landmark_layout[c * n_landmarks + uint(idx)];
      }
      out[c * n_query + row] = value;
    }
    return;
  }

  int positive_count = 0;
  for (uint j = 0u; j < k; ++j) {
    float value = max(0.0f, distances[j] - rho);
    adjusted[j] = value;
    if (value > eps) scratch[positive_count++] = value;
  }
  if (positive_count == 0) {
    for (uint j = 0u; j < k; ++j) scratch[j] = distances[j];
    positive_count = int(k);
  }

  float sigma = median_local(scratch, positive_count);
  if (!isfinite(sigma) || sigma < eps) sigma = eps;

  float weight_sum = 0.0f;
  for (uint j = 0u; j < k; ++j) {
    float w = exp(-adjusted[j] / sigma);
    adjusted[j] = w;
    weight_sum += w;
  }
  if (!isfinite(weight_sum) || weight_sum <= 0.0f) {
    weight_sum = float(k);
    for (uint j = 0u; j < k; ++j) adjusted[j] = 1.0f;
  }

  float max_probability = 0.0f;
  float entropy = 0.0f;
  for (uint j = 0u; j < k; ++j) {
    float probability = adjusted[j] / weight_sum;
    max_probability = max(max_probability, probability);
    entropy -= probability * log(max(probability, eps));
  }
  float entropy_score = k > 1u ? 1.0f - min(1.0f, entropy / log(float(k))) : 1.0f;
  float confidence = 0.65f * entropy_score + 0.35f * max_probability;
  out_confidence[row] = clamp(confidence, 0.0f, 1.0f);

  for (uint c = 0u; c < n_components; ++c) {
    float value = 0.0f;
    for (uint j = 0u; j < k; ++j) {
      int idx = best_idx[j];
      if (idx >= 0 && uint(idx) < n_landmarks) {
        value += adjusted[j] * landmark_layout[c * n_landmarks + uint(idx)];
      }
    }
    out[c * n_query + row] = value / weight_sum;
  }
}

kernel void overwrite_landmark_rows(
  device float* out [[buffer(0)]],
  device const float* landmark_layout [[buffer(1)]],
  device const int* landmark_indices [[buffer(2)]],
  constant uint& n_landmarks [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& n_components [[buffer(5)]],
  uint gid [[thread_position_in_grid]]
) {
  uint total = n_landmarks * n_components;
  if (gid >= total) return;
  uint landmark = gid % n_landmarks;
  uint component = gid / n_landmarks;
  int row = landmark_indices[landmark] - 1;
  if (row < 0 || uint(row) >= n) return;
  out[component * n + uint(row)] = landmark_layout[component * n_landmarks + landmark];
}

kernel void structure_score_rows(
  device const float* layout [[buffer(0)]],
  device const int* indices [[buffer(1)]],
  device const int* keep [[buffer(2)]],
  device const int* labels [[buffer(3)]],
  device float* row_scores [[buffer(4)]],
  constant uint& n [[buffer(5)]],
  constant uint& index_rows [[buffer(6)]],
  constant uint& high_rank_limit [[buffer(7)]],
  constant uint& preserve_k [[buffer(8)]],
  constant uint& keep_n [[buffer(9)]],
  constant uint& compact_indices [[buffer(10)]],
  constant uint& n_label_levels [[buffer(11)]],
  uint kk [[thread_position_in_grid]]
) {
  if (kk >= keep_n) return;
  int query = keep[kk] - 1;
  if (query < 0 || uint(query) >= n) return;
  int index_row = compact_indices != 0u ? int(kk) : query;
  if (index_row < 0 || uint(index_row) >= index_rows) return;

  float low_dist[FASTEMBEDR_METAL_SCORE_MAX_K];
  int low_idx[FASTEMBEDR_METAL_SCORE_MAX_K];
  float high_dist[FASTEMBEDR_METAL_SCORE_MAX_K];
  int high_idx[FASTEMBEDR_METAL_SCORE_MAX_K];
  int low_count = 0;
  int high_count = 0;

  for (uint r = 0u; r < preserve_k; ++r) {
    int high_nb = indices[r * index_rows + uint(index_row)] - 1;
    if (high_nb < 0 || uint(high_nb) >= n) continue;
    float d2 = layout_d2_2d_metal(layout, n, query, high_nb);
    int pos = high_count;
    ++high_count;
    while (pos > 0 && pair_less_metal(d2, high_nb, high_dist[pos - 1], high_idx[pos - 1])) {
      high_dist[pos] = high_dist[pos - 1];
      high_idx[pos] = high_idx[pos - 1];
      --pos;
    }
    high_dist[pos] = d2;
    high_idx[pos] = high_nb;
  }

  for (uint candidate = 0u; candidate < n; ++candidate) {
    if (int(candidate) == query) continue;
    float d2 = layout_d2_2d_metal(layout, n, query, int(candidate));
    insert_top_neighbor_metal(low_dist, low_idx, low_count, int(preserve_k), d2, int(candidate));
  }
  if (low_count < int(preserve_k)) return;

  int shared = 0;
  float trust_penalty = 0.0f;
  for (uint r = 0u; r < preserve_k; ++r) {
    int rank = high_rank_metal(indices, index_rows, index_row, low_idx[r], int(high_rank_limit));
    if (rank <= int(preserve_k)) ++shared;
    trust_penalty += max(0.0f, float(rank - int(preserve_k)));
  }

  float cont_penalty = 0.0f;
  for (int t = 0; t < high_count; ++t) {
    int lower_rank_count = 0;
    for (uint candidate = 0u; candidate < n; ++candidate) {
      if (int(candidate) == query) continue;
      float d2 = layout_d2_2d_metal(layout, n, query, int(candidate));
      if (pair_less_metal(d2, int(candidate), high_dist[t], high_idx[t])) ++lower_rank_count;
    }
    int low_rank = 1 + lower_rank_count;
    cont_penalty += max(0.0f, float(low_rank - int(preserve_k)));
  }

  float label_accuracy = 0.0f;
  float label_accuracy_n = 0.0f;
  if (n_label_levels > 0u) {
    int truth = labels[query];
    if (truth >= 1 && uint(truth) <= n_label_levels) {
      int best_label = 0;
      int best_count = 0;
      for (uint label = 1u; label <= n_label_levels; ++label) {
        int count = 0;
        for (uint r = 0u; r < preserve_k; ++r) {
          if (labels[low_idx[r]] == int(label)) ++count;
        }
        if (count > best_count) {
          best_count = count;
          best_label = int(label);
        }
      }
      if (best_count > 0) {
        label_accuracy = best_label == truth ? 1.0f : 0.0f;
        label_accuracy_n = 1.0f;
      }
    }
  }

  float trust_denom = float(preserve_k) * max(1.0f, float(high_rank_limit + 1u - preserve_k));
  float cont_denom = float(preserve_k) * max(1.0f, float(n - preserve_k));
  float preservation = float(shared) / float(preserve_k);
  float trust = clamp(1.0f - trust_penalty / trust_denom, 0.0f, 1.0f);
  float continuity = clamp(1.0f - cont_penalty / cont_denom, 0.0f, 1.0f);
  uint base = kk * FASTEMBEDR_METAL_SCORE_WIDTH;
  row_scores[base] = preservation;
  row_scores[base + 1u] = trust;
  row_scores[base + 2u] = continuity;
  row_scores[base + 3u] = label_accuracy;
  row_scores[base + 4u] = label_accuracy_n;
  row_scores[base + 5u] = 1.0f;
}

kernel void silhouette_rows(
  device const float* layout [[buffer(0)]],
  device const int* labels [[buffer(1)]],
  device const int* counts [[buffer(2)]],
  device float* row_scores [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& n_label_levels [[buffer(5)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  int own_label = labels[row];
  if (own_label < 1 || uint(own_label) > n_label_levels) return;
  float x0 = layout[row];
  float x1 = layout[n + row];
  if (!isfinite(x0) || !isfinite(x1)) return;

  float sums[FASTEMBEDR_METAL_MAX_LABELS + 1];
  for (uint label = 0u; label <= n_label_levels; ++label) sums[label] = 0.0f;

  for (uint j = 0u; j < n; ++j) {
    if (j == row) continue;
    int label = labels[j];
    if (label < 1 || uint(label) > n_label_levels) continue;
    float y0 = layout[j];
    float y1 = layout[n + j];
    if (!isfinite(y0) || !isfinite(y1)) continue;
    float dx = x0 - y0;
    float dy = x1 - y1;
    sums[label] += sqrt(dx * dx + dy * dy);
  }

  int own_count = counts[own_label] - 1;
  float a = own_count > 0 ? sums[own_label] / float(own_count) : 0.0f;
  float b = 3.4028234663852886e+38f;
  for (uint label = 1u; label <= n_label_levels; ++label) {
    if (int(label) == own_label || counts[label] <= 0) continue;
    b = min(b, sums[label] / float(counts[label]));
  }
  float value = 0.0f;
  if (isfinite(b)) {
    float denom = max(a, b);
    value = denom > 0.0f ? (b - a) / denom : 0.0f;
  }
  uint base = row * 2u;
  row_scores[base] = value;
  row_scores[base + 1u] = 1.0f;
}
)METAL";
}

NSUInteger bounded_threads(id<MTLComputePipelineState> pipeline,
                           const NSUInteger cap = 256) {
  const NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
  if (max_threads < 1) Rcpp::stop("Metal pipeline reports no available threads.");
  NSUInteger threads = std::min<NSUInteger>(max_threads, cap);
  NSUInteger power = 1;
  while ((power << 1u) <= threads) power <<= 1u;
  return power;
}

void wait_for_command(id<MTLCommandBuffer> command_buffer,
                      const char* stage) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    Rcpp::stop("Metal %s command failed: %s", stage, ns_error_message(command_buffer.error).c_str());
  }
}

struct MetalStageTimingEntry {
  std::string stage;
  int command_count = 0;
  double wall_sec = 0.0;
  double gpu_sec = 0.0;
  int gpu_timestamp_count = 0;
};

bool metal_stage_timing_enabled() {
  const char* raw = std::getenv("FASTEMBEDR_METAL_STAGE_TIMING");
  if (raw == nullptr || raw[0] == '\0') return false;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value != "0" && value != "false" && value != "no" && value != "off";
}

class MetalStageTimer {
 public:
  explicit MetalStageTimer(const bool enabled) : enabled_(enabled) {}

  bool enabled() const { return enabled_; }

  void record(id<MTLCommandBuffer> command_buffer,
              const char* stage,
              const std::chrono::steady_clock::time_point start) {
    if (!enabled_) return;
    const auto stop = std::chrono::steady_clock::now();
    const double wall_sec =
      std::chrono::duration<double>(stop - start).count();
    MetalStageTimingEntry& entry = entry_for(stage);
    entry.command_count += 1;
    entry.wall_sec += wall_sec;

    bool has_gpu_timestamp = false;
    double gpu_sec = 0.0;
    if (command_buffer != nil &&
        [command_buffer respondsToSelector:@selector(GPUStartTime)] &&
        [command_buffer respondsToSelector:@selector(GPUEndTime)]) {
      const double gpu_start = [command_buffer GPUStartTime];
      const double gpu_end = [command_buffer GPUEndTime];
      if (std::isfinite(gpu_start) && std::isfinite(gpu_end) && gpu_end > gpu_start) {
        has_gpu_timestamp = true;
        gpu_sec = gpu_end - gpu_start;
      }
    }
    if (has_gpu_timestamp) {
      entry.gpu_sec += gpu_sec;
      entry.gpu_timestamp_count += 1;
    }
  }

  Rcpp::DataFrame to_data_frame() const {
    const R_xlen_t n = static_cast<R_xlen_t>(entries_.size());
    Rcpp::CharacterVector stage(n);
    Rcpp::IntegerVector command_count(n);
    Rcpp::NumericVector wall_sec(n);
    Rcpp::NumericVector gpu_sec(n);
    Rcpp::LogicalVector gpu_timestamps_available(n);
    for (R_xlen_t i = 0; i < n; ++i) {
      const MetalStageTimingEntry& entry = entries_[static_cast<std::size_t>(i)];
      stage[i] = entry.stage;
      command_count[i] = entry.command_count;
      wall_sec[i] = entry.wall_sec;
      if (entry.gpu_timestamp_count > 0) {
        gpu_sec[i] = entry.gpu_sec;
        gpu_timestamps_available[i] = true;
      } else {
        gpu_sec[i] = NA_REAL;
        gpu_timestamps_available[i] = false;
      }
    }
    return Rcpp::DataFrame::create(
      Rcpp::Named("stage") = stage,
      Rcpp::Named("command_count") = command_count,
      Rcpp::Named("wall_sec") = wall_sec,
      Rcpp::Named("gpu_sec") = gpu_sec,
      Rcpp::Named("gpu_timestamps_available") = gpu_timestamps_available,
      Rcpp::Named("stringsAsFactors") = false
    );
  }

 private:
  MetalStageTimingEntry& entry_for(const char* stage) {
    for (MetalStageTimingEntry& entry : entries_) {
      if (entry.stage == stage) return entry;
    }
    entries_.push_back(MetalStageTimingEntry{std::string(stage), 0, 0.0, 0.0, 0});
    return entries_.back();
  }

  bool enabled_;
  std::vector<MetalStageTimingEntry> entries_;
};

template <typename EncodeFn>
void run_timed_metal_stage(MetalEmbeddingState& state,
                           MetalStageTimer& timer,
                           const char* stage,
                           EncodeFn encode) {
  id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
  const auto start = std::chrono::steady_clock::now();
  encode(command_buffer);
  wait_for_command(command_buffer, stage);
  timer.record(command_buffer, stage, start);
}

void dispatch_rows(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> pipeline,
                   const int n) {
  const NSUInteger threads = bounded_threads(pipeline);
  [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

MTLSize metal_threadgroup_2d(id<MTLComputePipelineState> pipeline) {
  const NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
  if (max_threads >= 256) return MTLSizeMake(16, 16, 1);
  if (max_threads >= 64) return MTLSizeMake(8, 8, 1);
  return MTLSizeMake(std::max<NSUInteger>(1, max_threads), 1, 1);
}

int metal_env_positive_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 0L || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

bool metal_env_flag(const char* name, const bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

int metal_tsne_fft_grid_size(const int n) {
  // The native Metal path uses the same FFT-grid objective as the CPU path. On
  // MNIST-scale data, 128 cells is too coarse, while 512 costs too much without
  // improving the plot once Metal uses stable step clipping. Use 256 for large
  // runs by default; FASTEMBEDR_TSNE_FFT_GRID remains an explicit override.
  const int fallback = n >= 50000 ? 256 : (n >= 10000 ? 256 : 64);
  const int requested = metal_env_positive_int("FASTEMBEDR_TSNE_FFT_GRID", fallback);
  int grid = 32;
  while (grid < requested && grid < 512) grid <<= 1;
  return std::max(32, std::min(512, grid));
}

std::uint32_t log2_power_of_two(const std::uint32_t value) {
  std::uint32_t out = 0;
  std::uint32_t current = value;
  while (current > 1u) {
    current >>= 1u;
    ++out;
  }
  return out;
}

std::vector<float> make_fft_twiddles(const std::uint32_t fft_size,
                                     const std::uint32_t log_fft) {
  const std::uint32_t half_count = fft_size >> 1u;
  std::vector<float> twiddles(static_cast<std::size_t>(log_fft) * half_count * 2u, 0.0f);
  constexpr double two_pi = 6.283185307179586476925286766559;
  for (std::uint32_t stage = 1u; stage <= log_fft; ++stage) {
    const std::uint32_t span_half = 1u << (stage - 1u);
    const std::uint32_t width = span_half << 1u;
    const std::size_t base = static_cast<std::size_t>(stage - 1u) * half_count * 2u;
    for (std::uint32_t j = 0u; j < span_half; ++j) {
      const double angle = two_pi * static_cast<double>(j) / static_cast<double>(width);
      const std::size_t pos = base + static_cast<std::size_t>(j) * 2u;
      twiddles[pos] = static_cast<float>(std::cos(angle));
      twiddles[pos + 1u] = static_cast<float>(std::sin(angle));
    }
  }
  return twiddles;
}

void encode_fft_512_stockham_metal(MetalEmbeddingState& state,
                                   id<MTLCommandBuffer> command_buffer,
                                   id<MTLBuffer> values,
                                   id<MTLBuffer> scratch,
                                   const bool inverse);

void encode_fft_2d_metal_generic(MetalEmbeddingState& state,
                                 id<MTLCommandBuffer> command_buffer,
                                 id<MTLBuffer> values,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> twiddles,
                                 const std::uint32_t fft_size,
                                 const std::uint32_t log_fft,
                                 const bool inverse);

void encode_fft_2d_metal(MetalEmbeddingState& state,
                         id<MTLCommandBuffer> command_buffer,
                         id<MTLBuffer> values,
                         id<MTLBuffer> scratch,
                         id<MTLBuffer> twiddles,
                         const std::uint32_t fft_size,
                         const std::uint32_t log_fft,
                         const bool inverse) {
  if (fft_size == 512u &&
      state.opentsne_fft_512_rows_stockham_pipeline != nil &&
      state.opentsne_fft_512_cols_stockham_pipeline != nil) {
    encode_fft_512_stockham_metal(state, command_buffer, values, scratch, inverse);
    return;
  }

  encode_fft_2d_metal_generic(
    state, command_buffer, values, scratch, twiddles, fft_size, log_fft, inverse
  );
}

void encode_fft_2d_metal_generic(MetalEmbeddingState& state,
                                 id<MTLCommandBuffer> command_buffer,
                                 id<MTLBuffer> values,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> twiddles,
                                 const std::uint32_t fft_size,
                                 const std::uint32_t log_fft,
                                 const bool inverse) {
  const std::uint32_t inverse_u = inverse ? 1u : 0u;
  const MTLSize full_grid = MTLSizeMake(static_cast<NSUInteger>(fft_size),
                                       static_cast<NSUInteger>(fft_size),
                                       1);
  const MTLSize half_row_grid = MTLSizeMake(static_cast<NSUInteger>(fft_size / 2u),
                                           static_cast<NSUInteger>(fft_size),
                                           1);
  const MTLSize half_col_grid = MTLSizeMake(static_cast<NSUInteger>(fft_size),
                                           static_cast<NSUInteger>(fft_size / 2u),
                                           1);

  {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_bit_reverse_rows_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBuffer:scratch offset:0 atIndex:1];
    [encoder setBytes:&fft_size length:sizeof(std::uint32_t) atIndex:2];
    [encoder setBytes:&log_fft length:sizeof(std::uint32_t) atIndex:3];
    [encoder dispatchThreads:full_grid
       threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_bit_reverse_rows_pipeline)];
    [encoder endEncoding];
  }

  for (std::uint32_t stage = 1u; stage <= log_fft; ++stage) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_butterfly_rows_pipeline];
    [encoder setBuffer:scratch offset:0 atIndex:0];
    [encoder setBytes:&fft_size length:sizeof(std::uint32_t) atIndex:1];
    [encoder setBytes:&stage length:sizeof(std::uint32_t) atIndex:2];
    [encoder setBytes:&inverse_u length:sizeof(std::uint32_t) atIndex:3];
    [encoder setBuffer:twiddles offset:0 atIndex:4];
    [encoder dispatchThreads:half_row_grid
       threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_butterfly_rows_pipeline)];
    [encoder endEncoding];
  }

  {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_bit_reverse_cols_pipeline];
    [encoder setBuffer:scratch offset:0 atIndex:0];
    [encoder setBuffer:values offset:0 atIndex:1];
    [encoder setBytes:&fft_size length:sizeof(std::uint32_t) atIndex:2];
    [encoder setBytes:&log_fft length:sizeof(std::uint32_t) atIndex:3];
    [encoder dispatchThreads:full_grid
       threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_bit_reverse_cols_pipeline)];
    [encoder endEncoding];
  }

  for (std::uint32_t stage = 1u; stage <= log_fft; ++stage) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_butterfly_cols_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&fft_size length:sizeof(std::uint32_t) atIndex:1];
    [encoder setBytes:&stage length:sizeof(std::uint32_t) atIndex:2];
    [encoder setBytes:&inverse_u length:sizeof(std::uint32_t) atIndex:3];
    [encoder setBuffer:twiddles offset:0 atIndex:4];
    [encoder dispatchThreads:half_col_grid
       threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_butterfly_cols_pipeline)];
    [encoder endEncoding];
  }

  if (inverse) {
    const std::uint32_t total = fft_size * fft_size;
    const float scale = 1.0f / static_cast<float>(total);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_scale_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&total length:sizeof(std::uint32_t) atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(bounded_threads(state.opentsne_fft_scale_pipeline), 1, 1)];
    [encoder endEncoding];
  }
}

void encode_fft_convolution_metal(MetalEmbeddingState& state,
                                  id<MTLCommandBuffer> command_buffer,
                                  id<MTLBuffer> transformed_mass,
                                  id<MTLBuffer> transformed_kernel,
                                  id<MTLBuffer> out,
                                  id<MTLBuffer> scratch,
                                  id<MTLBuffer> twiddles,
                                  const std::uint32_t fft_size,
                                  const std::uint32_t log_fft,
                                  const std::uint32_t total) {
  {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_multiply_pipeline];
    [encoder setBuffer:transformed_mass offset:0 atIndex:0];
    [encoder setBuffer:transformed_kernel offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&total length:sizeof(std::uint32_t) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(bounded_threads(state.opentsne_fft_multiply_pipeline), 1, 1)];
    [encoder endEncoding];
  }
  encode_fft_2d_metal(state, command_buffer, out, scratch, twiddles, fft_size, log_fft, true);
}

void encode_fft_512_stockham_metal(MetalEmbeddingState& state,
                                   id<MTLCommandBuffer> command_buffer,
                                   id<MTLBuffer> values,
                                   id<MTLBuffer> scratch,
                                   const bool inverse) {
  const std::uint32_t inverse_u = inverse ? 1u : 0u;
  const MTLSize groups = MTLSizeMake(512, 1, 1);
  const MTLSize threads = MTLSizeMake(128, 1, 1);
  {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_512_rows_stockham_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBuffer:scratch offset:0 atIndex:1];
    [encoder setBytes:&inverse_u length:sizeof(std::uint32_t) atIndex:2];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
  }
  {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_512_cols_stockham_pipeline];
    [encoder setBuffer:scratch offset:0 atIndex:0];
    [encoder setBuffer:values offset:0 atIndex:1];
    [encoder setBytes:&inverse_u length:sizeof(std::uint32_t) atIndex:2];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
  }
  if (inverse) {
    const std::uint32_t total = 512u * 512u;
    const float scale = 1.0f / static_cast<float>(total);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_scale_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&total length:sizeof(std::uint32_t) atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(bounded_threads(state.opentsne_fft_scale_pipeline), 1, 1)];
    [encoder endEncoding];
  }
}

std::vector<float> numeric_matrix_to_float(const NumericMatrix& x) {
  const std::size_t size = static_cast<std::size_t>(x.nrow()) * x.ncol();
  std::vector<float> out(size);
  const double* begin = x.begin();
  for (std::size_t i = 0; i < size; ++i) {
    out[i] = static_cast<float>(begin[i]);
  }
  return out;
}

std::vector<float> numeric_matrix_to_row_major_float(const NumericMatrix& x) {
  std::vector<float> out(static_cast<std::size_t>(x.nrow()) * x.ncol());
  for (int c = 0; c < x.ncol(); ++c) {
    for (int r = 0; r < x.nrow(); ++r) {
      out[static_cast<std::size_t>(r) * x.ncol() + c] = static_cast<float>(x(r, c));
    }
  }
  return out;
}

NumericMatrix float_to_numeric_matrix(const std::vector<float>& values,
                                      const int nrow,
                                      const int ncol) {
  NumericMatrix out(nrow, ncol);
  const std::size_t size = static_cast<std::size_t>(nrow) * ncol;
  for (std::size_t i = 0; i < size; ++i) {
    out.begin()[i] = static_cast<double>(values[i]);
  }
  return out;
}

NumericMatrix run_rsvd_multiply_metal(NumericMatrix left,
                                      NumericMatrix right,
                                      bool transpose_left) {
  if (left.nrow() < 1 || left.ncol() < 1 || right.nrow() < 1 || right.ncol() < 1) {
    Rcpp::stop("Metal RSVD matrix multiply requires non-empty matrices.");
  }
  if (transpose_left) {
    if (left.nrow() != right.nrow()) {
      Rcpp::stop("Metal RSVD cross-product received non-conformable matrices.");
    }
  } else if (left.ncol() != right.nrow()) {
    Rcpp::stop("Metal RSVD matrix multiply received non-conformable matrices.");
  }

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    const int out_rows = transpose_left ? left.ncol() : left.nrow();
    const int out_cols = right.ncol();
    std::vector<float> left_values = numeric_matrix_to_float(left);
    std::vector<float> right_values = numeric_matrix_to_float(right);
    std::vector<float> out_values(static_cast<std::size_t>(out_rows) * out_cols, 0.0f);
    MatrixMultiplyParams params{
      static_cast<std::uint32_t>(left.nrow()),
      static_cast<std::uint32_t>(left.ncol()),
      static_cast<std::uint32_t>(right.ncol()),
      transpose_left ? 1u : 0u
    };

    id<MTLBuffer> left_buffer = [state.device newBufferWithBytes:left_values.data()
                                                          length:left_values.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> right_buffer = [state.device newBufferWithBytes:right_values.data()
                                                           length:right_values.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [state.device newBufferWithBytes:out_values.data()
                                                         length:out_values.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(MatrixMultiplyParams)
                                                           options:MTLResourceStorageModeShared];
    if (left_buffer == nil || right_buffer == nil || out_buffer == nil || params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal RSVD matrix multiply buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.matrix_multiply_pipeline];
    [encoder setBuffer:left_buffer offset:0 atIndex:0];
    [encoder setBuffer:right_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBuffer:params_buffer offset:0 atIndex:3];
    const NSUInteger width = std::min<NSUInteger>(16, [state.matrix_multiply_pipeline threadExecutionWidth]);
    const NSUInteger height = std::max<NSUInteger>(
      1,
      std::min<NSUInteger>(16, [state.matrix_multiply_pipeline maxTotalThreadsPerThreadgroup] / width)
    );
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(out_cols), static_cast<NSUInteger>(out_rows), 1)
       threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
    [encoder endEncoding];
    wait_for_command(command_buffer, "RSVD matrix multiply");

    std::memcpy(out_values.data(), [out_buffer contents], out_values.size() * sizeof(float));
    [left_buffer release];
    [right_buffer release];
    [out_buffer release];
    [params_buffer release];
    return float_to_numeric_matrix(out_values, out_rows, out_cols);
  }
}

int knn_index_offset(const IntegerMatrix& indices, const int n) {
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int j = 0; j < indices.ncol(); ++j) {
    for (int i = 0; i < indices.nrow(); ++i) {
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
  if (k > kMaxMetalProjectionNeighbors) {
    Rcpp::stop("Metal projection currently supports at most %d neighbors.", kMaxMetalProjectionNeighbors);
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

std::vector<float> run_projection_metal(MetalEmbeddingState& state,
                                        const NumericMatrix& reference_layout,
                                        const IntegerMatrix& projection_indices,
                                        const NumericMatrix& projection_distances,
                                        const bool average_zeros) {
  const int n_reference = reference_layout.nrow();
  const int n_query = projection_indices.nrow();
  const int k = projection_indices.ncol();
  const int n_components = reference_layout.ncol();

  std::vector<float> reference = numeric_matrix_to_float(reference_layout);
  std::vector<float> distances = numeric_matrix_to_float(projection_distances);
  std::vector<float> out(static_cast<std::size_t>(n_query) * n_components, 0.0f);

  id<MTLBuffer> reference_buffer = [state.device newBufferWithBytes:reference.data()
                                                             length:reference.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> index_buffer = [state.device newBufferWithBytes:projection_indices.begin()
                                                         length:static_cast<std::size_t>(n_query) * k * sizeof(int)
                                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> distance_buffer = [state.device newBufferWithBytes:distances.data()
                                                            length:distances.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_buffer = [state.device newBufferWithLength:out.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
  if (reference_buffer == nil || index_buffer == nil || distance_buffer == nil || out_buffer == nil) {
    Rcpp::stop("Failed to allocate Metal projection buffers.");
  }

  const std::uint32_t n_reference_u = static_cast<std::uint32_t>(n_reference);
  const std::uint32_t n_query_u = static_cast<std::uint32_t>(n_query);
  const std::uint32_t k_u = static_cast<std::uint32_t>(k);
  const std::uint32_t n_components_u = static_cast<std::uint32_t>(n_components);
  const std::uint32_t average_zeros_u = average_zeros ? 1u : 0u;

  id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:state.project_pipeline];
  [encoder setBuffer:reference_buffer offset:0 atIndex:0];
  [encoder setBuffer:index_buffer offset:0 atIndex:1];
  [encoder setBuffer:distance_buffer offset:0 atIndex:2];
  [encoder setBuffer:out_buffer offset:0 atIndex:3];
  [encoder setBytes:&n_reference_u length:sizeof(std::uint32_t) atIndex:4];
  [encoder setBytes:&n_query_u length:sizeof(std::uint32_t) atIndex:5];
  [encoder setBytes:&k_u length:sizeof(std::uint32_t) atIndex:6];
  [encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:7];
  [encoder setBytes:&average_zeros_u length:sizeof(std::uint32_t) atIndex:8];
  dispatch_rows(encoder, state.project_pipeline, n_query);
  [encoder endEncoding];
  wait_for_command(command_buffer, "projection");

  std::memcpy(out.data(), [out_buffer contents], out.size() * sizeof(float));
  [reference_buffer release];
  [index_buffer release];
  [distance_buffer release];
  [out_buffer release];
  return out;
}

void encode_spectral_normalize(MetalEmbeddingState& state,
                               id<MTLCommandBuffer> command_buffer,
                               id<MTLBuffer> value_buffer,
                               id<MTLBuffer> stats_buffer,
                               const std::uint32_t n_u) {
  const NSUInteger stats_threads = bounded_threads(state.spectral_stats_pipeline);
  const std::uint32_t stats_threads_u = static_cast<std::uint32_t>(stats_threads);

  id<MTLComputeCommandEncoder> stats_encoder = [command_buffer computeCommandEncoder];
  [stats_encoder setComputePipelineState:state.spectral_stats_pipeline];
  [stats_encoder setBuffer:value_buffer offset:0 atIndex:0];
  [stats_encoder setBuffer:stats_buffer offset:0 atIndex:1];
  [stats_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
  [stats_encoder setBytes:&stats_threads_u length:sizeof(std::uint32_t) atIndex:3];
  [stats_encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(stats_threads, 1, 1)];
  [stats_encoder endEncoding];

  id<MTLComputeCommandEncoder> normalize_encoder = [command_buffer computeCommandEncoder];
  [normalize_encoder setComputePipelineState:state.spectral_normalize_pipeline];
  [normalize_encoder setBuffer:value_buffer offset:0 atIndex:0];
  [normalize_encoder setBuffer:stats_buffer offset:0 atIndex:1];
  [normalize_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
  dispatch_rows(normalize_encoder, state.spectral_normalize_pipeline, static_cast<int>(n_u));
  [normalize_encoder endEncoding];
}

std::vector<int> zero_based_reference_indices_metal(const IntegerMatrix& indices,
                                                    const NumericMatrix& distances,
                                                    const int n_reference) {
  const int n_query = indices.nrow();
  const int k = indices.ncol();
  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < n_query; ++i) {
      min_idx = std::min(min_idx, indices(i, j));
      max_idx = std::max(max_idx, indices(i, j));
    }
  }
  const int offset = (min_idx >= 1 && max_idx <= n_reference) ? 1 : 0;
  std::vector<int> out(static_cast<std::size_t>(n_query) * k);
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < n_query; ++i) {
      const int ref = indices(i, j) - offset;
      if (ref < 0 || ref >= n_reference) {
        Rcpp::stop("KNN indices are out of range for `reference_layout`.");
      }
      const double d = distances(i, j);
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative.");
      }
      out[static_cast<std::size_t>(j) * n_query + i] = ref;
    }
  }
  return out;
}

std::vector<float> tsne_transform_probabilities_metal(const NumericMatrix& distances,
                                                      const double perplexity) {
  const int n_query = distances.nrow();
  const int k = distances.ncol();
  std::vector<float> out(static_cast<std::size_t>(n_query) * k, 0.0f);
  std::vector<double> row(static_cast<std::size_t>(k), 0.0);
  const double target_entropy = std::log(perplexity);
  const double tol = 1e-5;

  for (int i = 0; i < n_query; ++i) {
    bool found = false;
    double beta = 1.0;
    double min_beta = -std::numeric_limits<double>::max();
    double max_beta = std::numeric_limits<double>::max();
    double sum_p = std::numeric_limits<double>::min();

    for (int iter = 0; !found && iter < 200; ++iter) {
      sum_p = std::numeric_limits<double>::min();
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j);
        const double p = std::exp(-beta * d * d);
        row[static_cast<std::size_t>(j)] = p;
        sum_p += p;
      }
      double entropy = 0.0;
      for (int j = 0; j < k; ++j) {
        const double d = distances(i, j);
        entropy += beta * (d * d * row[static_cast<std::size_t>(j)]);
      }
      entropy = entropy / sum_p + std::log(sum_p);
      const double diff = entropy - target_entropy;
      if (std::abs(diff) < tol) {
        found = true;
      } else if (diff > 0.0) {
        min_beta = beta;
        beta = max_beta == std::numeric_limits<double>::max() ?
          beta * 2.0 :
          (beta + max_beta) / 2.0;
      } else {
        max_beta = beta;
        beta = min_beta == -std::numeric_limits<double>::max() ?
          beta / 2.0 :
          (beta + min_beta) / 2.0;
      }
    }
    for (int j = 0; j < k; ++j) {
      out[static_cast<std::size_t>(j) * n_query + i] =
        static_cast<float>(row[static_cast<std::size_t>(j)] / sum_p);
    }
  }
  return out;
}

std::vector<float> initialize_tsne_transform_metal(const NumericMatrix& reference_layout,
                                                   const std::vector<int>& indices,
                                                   const NumericMatrix& distances,
                                                   const NumericMatrix& y_init,
                                                   const bool init,
                                                   const std::string& initialization,
                                                   const int seed) {
  const int n_reference = reference_layout.nrow();
  const int n_query = distances.nrow();
  const int k = distances.ncol();
  std::vector<float> out(static_cast<std::size_t>(n_query) * 2u, 0.0f);
  if (init) {
    if (y_init.nrow() != n_query || y_init.ncol() != 2) {
      Rcpp::stop("`Y_init` must have one row per query and two columns for Metal transform.");
    }
    for (int i = 0; i < n_query; ++i) {
      out[static_cast<std::size_t>(i) * 2u] = static_cast<float>(y_init(i, 0));
      out[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(y_init(i, 1));
    }
    return out;
  }

  if (initialization == "random") {
    const unsigned int resolved_seed = seed == NA_INTEGER ?
      5489u :
      static_cast<unsigned int>(seed);
    std::mt19937 rng(resolved_seed);
    std::normal_distribution<float> normal(0.0f, 1.0e-4f);
    for (float& value : out) value = normal(rng);
    return out;
  }

  std::vector<float> values(static_cast<std::size_t>(k), 0.0f);
  for (int i = 0; i < n_query; ++i) {
    for (int dim = 0; dim < 2; ++dim) {
      if (initialization == "weighted") {
        double numerator = 0.0;
        double denominator = std::numeric_limits<double>::min();
        for (int j = 0; j < k; ++j) {
          const int ref = indices[static_cast<std::size_t>(j) * n_query + i];
          const double d = std::max(0.0, distances(i, j));
          const double w = 1.0 / (d + 1e-6);
          numerator += w * reference_layout(ref, dim);
          denominator += w;
        }
        out[static_cast<std::size_t>(i) * 2u + dim] = static_cast<float>(numerator / denominator);
      } else {
        for (int j = 0; j < k; ++j) {
          const int ref = indices[static_cast<std::size_t>(j) * n_query + i];
          if (ref < 0 || ref >= n_reference) Rcpp::stop("KNN indices are out of range.");
          values[static_cast<std::size_t>(j)] = static_cast<float>(reference_layout(ref, dim));
        }
        const int mid = k / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        float median = values[static_cast<std::size_t>(mid)];
        if ((k & 1) == 0) {
          std::nth_element(values.begin(), values.begin() + mid - 1, values.begin() + mid);
          median = 0.5f * (median + values[static_cast<std::size_t>(mid - 1)]);
        }
        out[static_cast<std::size_t>(i) * 2u + dim] = median;
      }
    }
  }
  return out;
}

} // namespace

NumericMatrix spectral_knn_init_metal_impl(IntegerMatrix indices,
                                          NumericMatrix distances,
                                          int n_components,
                                          int spectral_n_iter,
                                          int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (n_components != 2) {
    Rcpp::stop("Metal spectral initialization currently supports exactly two components.");
  }
  if (indices.ncol() > kMaxMetalNeighbors) {
    Rcpp::stop("Metal spectral initialization currently supports at most %d neighbors.", kMaxMetalNeighbors);
  }
  if (spectral_n_iter < 1) Rcpp::stop("spectral_n_iter must be positive");

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    const int n = indices.nrow();
    std::vector<std::int32_t> neighbors;
    std::vector<float> weights;
    prepare_umap_graph_adjacency(indices, distances, 0, neighbors, weights);
    const int width = static_cast<int>(neighbors.size() / static_cast<std::size_t>(n));
    if (width < 1) Rcpp::stop("Metal spectral initialization produced an empty graph.");

    std::vector<float> current(static_cast<std::size_t>(n) * 2u, 0.0f);
    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t width_u = static_cast<std::uint32_t>(width);
    const std::uint32_t seed_u = static_cast<std::uint32_t>(seed);

    id<MTLBuffer> neighbor_buffer = [state.device newBufferWithBytes:neighbors.data()
                                                              length:neighbors.size() * sizeof(std::int32_t)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer = [state.device newBufferWithBytes:weights.data()
                                                            length:weights.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> current_buffer = [state.device newBufferWithLength:current.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> next_buffer = [state.device newBufferWithLength:current.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> stats_buffer = [state.device newBufferWithLength:5u * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    if (neighbor_buffer == nil || weight_buffer == nil || current_buffer == nil ||
        next_buffer == nil || stats_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal spectral initialization buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];

    id<MTLComputeCommandEncoder> random_encoder = [command_buffer computeCommandEncoder];
    [random_encoder setComputePipelineState:state.spectral_random_pipeline];
    [random_encoder setBuffer:current_buffer offset:0 atIndex:0];
    [random_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:1];
    [random_encoder setBytes:&seed_u length:sizeof(std::uint32_t) atIndex:2];
    dispatch_rows(random_encoder, state.spectral_random_pipeline, n);
    [random_encoder endEncoding];
    encode_spectral_normalize(state, command_buffer, current_buffer, stats_buffer, n_u);

    for (int iter = 0; iter < spectral_n_iter; ++iter) {
      id<MTLComputeCommandEncoder> diffuse_encoder = [command_buffer computeCommandEncoder];
      [diffuse_encoder setComputePipelineState:state.spectral_diffuse_pipeline];
      [diffuse_encoder setBuffer:neighbor_buffer offset:0 atIndex:0];
      [diffuse_encoder setBuffer:weight_buffer offset:0 atIndex:1];
      [diffuse_encoder setBuffer:current_buffer offset:0 atIndex:2];
      [diffuse_encoder setBuffer:next_buffer offset:0 atIndex:3];
      [diffuse_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
      [diffuse_encoder setBytes:&width_u length:sizeof(std::uint32_t) atIndex:5];
      dispatch_rows(diffuse_encoder, state.spectral_diffuse_pipeline, n);
      [diffuse_encoder endEncoding];
      encode_spectral_normalize(state, command_buffer, next_buffer, stats_buffer, n_u);
      std::swap(current_buffer, next_buffer);
    }

    wait_for_command(command_buffer, "spectral initialization");
    std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));

    NumericMatrix out(n, 2);
    for (int i = 0; i < n; ++i) {
      out(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
      out(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
    }
    [neighbor_buffer release];
    [weight_buffer release];
    [current_buffer release];
    [next_buffer release];
    [stats_buffer release];
    return out;
  }
}

NumericMatrix rsvd_multiply_metal_impl(NumericMatrix left,
                                       NumericMatrix right,
                                       bool transpose_left) {
  return run_rsvd_multiply_metal(left, right, transpose_left);
}

bool embedding_metal_available_impl() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
  }
}

List standardize_metal_impl(NumericMatrix data) {
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 2 || p < 1) Rcpp::stop("data must have at least two rows and one column");

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> values = numeric_matrix_to_float(data);
    std::vector<float> centers(static_cast<std::size_t>(p), 0.0f);
    std::vector<float> scales(static_cast<std::size_t>(p), 1.0f);

    id<MTLBuffer> value_buffer = [state.device newBufferWithBytes:values.data()
                                                           length:values.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> center_buffer = [state.device newBufferWithBytes:centers.data()
                                                            length:centers.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale_buffer = [state.device newBufferWithBytes:scales.data()
                                                           length:scales.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    if (value_buffer == nil || center_buffer == nil || scale_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal standardization buffers.");
    }

    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t total_u = static_cast<std::uint32_t>(values.size());
    const NSUInteger stats_threads = bounded_threads(state.standardize_stats_pipeline);
    const std::uint32_t stats_threads_u = static_cast<std::uint32_t>(stats_threads);

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> stats_encoder = [command_buffer computeCommandEncoder];
    [stats_encoder setComputePipelineState:state.standardize_stats_pipeline];
    [stats_encoder setBuffer:value_buffer offset:0 atIndex:0];
    [stats_encoder setBuffer:center_buffer offset:0 atIndex:1];
    [stats_encoder setBuffer:scale_buffer offset:0 atIndex:2];
    [stats_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:3];
    [stats_encoder setBytes:&stats_threads_u length:sizeof(std::uint32_t) atIndex:4];
    [stats_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(p), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(stats_threads, 1, 1)];
    [stats_encoder endEncoding];

    id<MTLComputeCommandEncoder> apply_encoder = [command_buffer computeCommandEncoder];
    [apply_encoder setComputePipelineState:state.standardize_apply_pipeline];
    [apply_encoder setBuffer:value_buffer offset:0 atIndex:0];
    [apply_encoder setBuffer:center_buffer offset:0 atIndex:1];
    [apply_encoder setBuffer:scale_buffer offset:0 atIndex:2];
    [apply_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:3];
    [apply_encoder setBytes:&total_u length:sizeof(std::uint32_t) atIndex:4];
    dispatch_rows(apply_encoder, state.standardize_apply_pipeline, static_cast<int>(values.size()));
    [apply_encoder endEncoding];
    wait_for_command(command_buffer, "standardization");

    std::memcpy(values.data(), [value_buffer contents], values.size() * sizeof(float));
    std::memcpy(centers.data(), [center_buffer contents], centers.size() * sizeof(float));
    std::memcpy(scales.data(), [scale_buffer contents], scales.size() * sizeof(float));
    [value_buffer release];
    [center_buffer release];
    [scale_buffer release];

    NumericVector center_out(p);
    NumericVector scale_out(p);
    for (int j = 0; j < p; ++j) {
      center_out[j] = static_cast<double>(centers[static_cast<std::size_t>(j)]);
      scale_out[j] = static_cast<double>(scales[static_cast<std::size_t>(j)]);
    }
    return List::create(
      Rcpp::Named("data") = float_to_numeric_matrix(values, n, p),
      Rcpp::Named("center") = center_out,
      Rcpp::Named("scale") = scale_out
    );
  }
}

List transform_tsne_metal_impl(NumericMatrix reference_layout,
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
  if (n_reference < 1 || reference_layout.ncol() != 2) {
    Rcpp::stop("Metal t-SNE transform requires a two-dimensional non-empty reference layout.");
  }
  if (n_query < 1 || k < 1) {
    Rcpp::stop("KNN input must have at least one query row and one neighbor column.");
  }
  if (k > kMaxMetalTsneTransformNeighbors) {
    Rcpp::stop("Metal t-SNE transform currently supports at most %d neighbors.", kMaxMetalTsneTransformNeighbors);
  }
  if (distances.nrow() != n_query || distances.ncol() != k) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  if (perplexity <= 0.0 || !std::isfinite(perplexity)) {
    Rcpp::stop("`perplexity` must be positive.");
  }
  if (n_iter < 0 || early_exaggeration_iter < 0 || n_iter + early_exaggeration_iter < 1) {
    Rcpp::stop("Metal t-SNE transform iteration counts must sum to at least one.");
  }
  if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
    Rcpp::stop("`learning_rate` must be positive.");
  }
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0 ||
      !std::isfinite(early_exaggeration) || !std::isfinite(exaggeration)) {
    Rcpp::stop("exaggeration values must be positive.");
  }
  if (initial_momentum < 0.0 || final_momentum < 0.0 ||
      !std::isfinite(initial_momentum) || !std::isfinite(final_momentum)) {
    Rcpp::stop("momentum values must be non-negative.");
  }
  if (initialization != "median" && initialization != "weighted" && initialization != "random") {
    Rcpp::stop("`initialization` must be 'median', 'weighted', or 'random'.");
  }
  if (exact_repulsion_threshold < 1) exact_repulsion_threshold = 1;
  if (n_negatives < 1) n_negatives = 1;
  n_negatives = std::min(n_negatives, n_reference);

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<int> ref_indices = zero_based_reference_indices_metal(
      indices,
      distances,
      n_reference
    );
    std::vector<float> probabilities = tsne_transform_probabilities_metal(
      distances,
      perplexity
    );
    std::vector<float> reference = numeric_matrix_to_float(reference_layout);
    std::vector<float> current = initialize_tsne_transform_metal(
      reference_layout,
      ref_indices,
      distances,
      y_init,
      init,
      initialization,
      seed
    );
    std::vector<float> gains(current.size(), 1.0f);
    std::vector<float> updates(current.size(), 0.0f);

    id<MTLBuffer> reference_buffer = [state.device newBufferWithBytes:reference.data()
                                                               length:reference.size() * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [state.device newBufferWithBytes:ref_indices.data()
                                                           length:ref_indices.size() * sizeof(int)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> probability_buffer = [state.device newBufferWithBytes:probabilities.data()
                                                                 length:probabilities.size() * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> current_buffer = [state.device newBufferWithBytes:current.data()
                                                             length:current.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gain_buffer = [state.device newBufferWithBytes:gains.data()
                                                          length:gains.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> update_buffer = [state.device newBufferWithBytes:updates.data()
                                                            length:updates.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    if (reference_buffer == nil || index_buffer == nil || probability_buffer == nil ||
        current_buffer == nil || gain_buffer == nil || update_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal t-SNE transform buffers.");
    }

    const bool exact_repulsion = n_reference <= exact_repulsion_threshold ||
      n_negatives >= n_reference;
    const NSUInteger threads_per_group = bounded_threads(state.tsne_transform_pipeline);
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n_query), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    const std::uint32_t epochs_per_command = kMetalEmbeddingEpochsPerCommand;
    const int total_iter = n_iter + early_exaggeration_iter;
    const float grad_clip = (max_grad_norm > 0.0 && std::isfinite(max_grad_norm)) ?
      static_cast<float>(max_grad_norm) :
      std::numeric_limits<float>::max();
    const float step_clip = (max_step_norm > 0.0 && std::isfinite(max_step_norm)) ?
      static_cast<float>(max_step_norm) :
      std::numeric_limits<float>::max();

    for (std::uint32_t epoch0 = 0; epoch0 < static_cast<std::uint32_t>(total_iter); epoch0 += epochs_per_command) {
      id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
      const std::uint32_t epoch_end = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(total_iter),
        epoch0 + epochs_per_command
      );
      for (std::uint32_t epoch = epoch0; epoch < epoch_end; ++epoch) {
        const bool in_early = epoch < static_cast<std::uint32_t>(early_exaggeration_iter);
        TsneTransformParams params{
          static_cast<std::uint32_t>(n_reference),
          static_cast<std::uint32_t>(n_query),
          static_cast<std::uint32_t>(k),
          static_cast<std::uint32_t>(n_negatives),
          static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed),
          exact_repulsion ? 1u : 0u,
          static_cast<float>(learning_rate),
          static_cast<float>(in_early ? early_exaggeration : exaggeration),
          static_cast<float>(in_early ? initial_momentum : final_momentum),
          grad_clip,
          step_clip
        };

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state.tsne_transform_pipeline];
        [encoder setBuffer:reference_buffer offset:0 atIndex:0];
        [encoder setBuffer:index_buffer offset:0 atIndex:1];
        [encoder setBuffer:probability_buffer offset:0 atIndex:2];
        [encoder setBuffer:current_buffer offset:0 atIndex:3];
        [encoder setBuffer:gain_buffer offset:0 atIndex:4];
        [encoder setBuffer:update_buffer offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(TsneTransformParams) atIndex:6];
        [encoder setBytes:&epoch length:sizeof(std::uint32_t) atIndex:7];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal t-SNE transform command failed: %s", ns_error_message(command_buffer.error).c_str());
      }
    }

    std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
    NumericMatrix layout(n_query, 2);
    for (int i = 0; i < n_query; ++i) {
      layout(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
      layout(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
    }

    [reference_buffer release];
    [index_buffer release];
    [probability_buffer release];
    [current_buffer release];
    [gain_buffer release];
    [update_buffer release];

    return List::create(
      Rcpp::Named("Y") = layout,
      Rcpp::Named("optimizer") = "opentsne_style_fixed_reference_transform_metal",
      Rcpp::Named("initialization") = initialization,
      Rcpp::Named("repulsion") = exact_repulsion ? "exact_reference_metal" : "sampled_reference_metal",
      Rcpp::Named("n_negatives") = n_negatives,
      Rcpp::Named("backend") = "metal"
    );
  }
}

List knn_tsne_opentsne_metal_impl(IntegerMatrix indices,
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
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 2 || k < 1) Rcpp::stop("KNN input must have at least two rows and one neighbor column.");
  if (n_components != 2) Rcpp::stop("Metal openTSNE currently supports exactly two output components.");
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN `indices` and `distances` must have the same dimensions.");
  }
  if (perplexity <= 0.0 || !std::isfinite(perplexity) || n - 1 < 3.0 * perplexity) {
    Rcpp::stop("perplexity is too large for the number of samples.");
  }
  if (early_exaggeration_iter < 0 || n_iter < 0 || early_exaggeration_iter + n_iter < 1) {
    Rcpp::stop("Metal openTSNE iteration counts must be non-negative and sum to at least one.");
  }
  if (learning_rate <= 0.0 && !learning_rate_auto) Rcpp::stop("`learning_rate` must be positive or automatic.");
  if (early_exaggeration <= 0.0 || exaggeration <= 0.0) Rcpp::stop("exaggeration values must be positive.");
  if (initial_momentum < 0.0 || final_momentum < 0.0) Rcpp::stop("momentum values must be non-negative.");
  if (min_gain <= 0.0) Rcpp::stop("`min_gain` must be positive.");

  std::string method = negative_gradient_method;
  std::transform(method.begin(), method.end(), method.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (method == "auto") method = "fft";
  const bool use_fft_grid =
    method == "fft" || method == "fitsne" || method == "fit_sne" || method == "interpolation";
  const bool use_exact =
    method == "exact" || method == "pair" || method == "pair_symmetric";
  if (!use_fft_grid && !use_exact) {
    Rcpp::stop("Metal openTSNE supports `negative_gradient_method = \"fft\"` or `\"exact\"`.");
  }
  if (use_exact && n > kMetalOpenTsneExactDenseThreshold) {
    Rcpp::stop(
      "Native Metal openTSNE exact optimization is limited to n <= %d until "
      "using `negative_gradient_method = \"fft\"`.",
      kMetalOpenTsneExactDenseThreshold
    );
  }

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    TsneSparseMetalGraph graph = build_tsne_sparse_graph_metal(indices, distances, perplexity);
    std::vector<float> current = initialize_opentsne_metal_layout(y_init, init, n, seed);
    std::vector<float> gains(current.size(), 1.0f);
    std::vector<float> updates(current.size(), 0.0f);
    std::vector<float> row_sums(static_cast<std::size_t>(n), 0.0f);
    float inv_sum_q_initial = 1.0f;

    id<MTLBuffer> row_ptr_buffer = [state.device newBufferWithBytes:graph.row_ptr.data()
                                                             length:graph.row_ptr.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> col_buffer = [state.device newBufferWithBytes:graph.col.data()
                                                         length:graph.col.size() * sizeof(std::int32_t)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> val_buffer = [state.device newBufferWithBytes:graph.val.data()
                                                         length:graph.val.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> current_buffer = [state.device newBufferWithBytes:current.data()
                                                             length:current.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gains_buffer = [state.device newBufferWithBytes:gains.data()
                                                           length:gains.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> updates_buffer = [state.device newBufferWithBytes:updates.data()
                                                             length:updates.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> row_sums_buffer = [state.device newBufferWithBytes:row_sums.data()
                                                              length:row_sums.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> inv_sum_q_buffer = [state.device newBufferWithBytes:&inv_sum_q_initial
                                                               length:sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    if (row_ptr_buffer == nil || col_buffer == nil || val_buffer == nil ||
        current_buffer == nil || gains_buffer == nil || updates_buffer == nil ||
        row_sums_buffer == nil || inv_sum_q_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal openTSNE buffers.");
    }

    const int total_iter = early_exaggeration_iter + n_iter;
    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const float max_step = (max_step_norm > 0.0 && std::isfinite(max_step_norm)) ?
      static_cast<float>(max_step_norm) :
      std::numeric_limits<float>::max();

    if (use_fft_grid) {
      const std::uint32_t grid_n = static_cast<std::uint32_t>(metal_tsne_fft_grid_size(n));
      const std::uint32_t fft_n = grid_n << 1u;
      const std::uint32_t log_fft = log2_power_of_two(fft_n);
      const std::uint32_t grid_total = grid_n * grid_n;
      const std::uint32_t fft_total = fft_n * fft_n;
      const std::size_t grid_bytes = static_cast<std::size_t>(grid_total) * sizeof(float);
      const std::size_t real_fft_bytes = static_cast<std::size_t>(fft_total) * sizeof(float);
      const std::size_t complex_bytes = static_cast<std::size_t>(fft_total) * sizeof(float) * 2u;
      std::vector<float> fft_twiddles = make_fft_twiddles(fft_n, log_fft);
      const std::uint32_t stats_block_size = 256u;
      const std::uint32_t stats_block_count =
        (static_cast<std::uint32_t>(n) + stats_block_size - 1u) / stats_block_size;
      bool use_mpsgraph_convolution = false;
      if (!record_costs) {
        if (@available(macOS 14.0, *)) {
          use_mpsgraph_convolution =
            metal_env_flag("FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH", false);
        }
      }

      id<MTLBuffer> mass_buffer = [state.device newBufferWithLength:grid_bytes
                                                            options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_x_buffer = [state.device newBufferWithLength:grid_bytes
                                                              options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_y_buffer = [state.device newBufferWithLength:grid_bytes
                                                              options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_fft_buffer = [state.device newBufferWithLength:complex_bytes
                                                                 options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_x_fft_buffer = [state.device newBufferWithLength:complex_bytes
                                                                   options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_y_fft_buffer = [state.device newBufferWithLength:complex_bytes
                                                                   options:MTLResourceStorageModePrivate];
      id<MTLBuffer> kernel_q_buffer = [state.device newBufferWithLength:complex_bytes
                                                                 options:MTLResourceStorageModePrivate];
      id<MTLBuffer> kernel_q2_buffer = [state.device newBufferWithLength:complex_bytes
                                                                  options:MTLResourceStorageModePrivate];
      id<MTLBuffer> q_grid_buffer = [state.device newBufferWithLength:complex_bytes
                                                              options:MTLResourceStorageModePrivate];
      id<MTLBuffer> q2_grid_buffer = [state.device newBufferWithLength:complex_bytes
                                                               options:MTLResourceStorageModePrivate];
      id<MTLBuffer> xq2_grid_buffer = [state.device newBufferWithLength:complex_bytes
                                                                options:MTLResourceStorageModePrivate];
      id<MTLBuffer> yq2_grid_buffer = [state.device newBufferWithLength:complex_bytes
                                                                options:MTLResourceStorageModePrivate];
      id<MTLBuffer> mass_real_buffer = nil;
      id<MTLBuffer> mass_x_real_buffer = nil;
      id<MTLBuffer> mass_y_real_buffer = nil;
      id<MTLBuffer> kernel_q_real_buffer = nil;
      id<MTLBuffer> kernel_q2_real_buffer = nil;
      id<MTLBuffer> q_grid_real_buffer = nil;
      id<MTLBuffer> q2_grid_real_buffer = nil;
      id<MTLBuffer> xq2_grid_real_buffer = nil;
      id<MTLBuffer> yq2_grid_real_buffer = nil;
      if (use_mpsgraph_convolution) {
        mass_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                     options:MTLResourceStorageModePrivate];
        mass_x_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                       options:MTLResourceStorageModePrivate];
        mass_y_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                       options:MTLResourceStorageModePrivate];
        kernel_q_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                         options:MTLResourceStorageModePrivate];
        kernel_q2_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                          options:MTLResourceStorageModePrivate];
        q_grid_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                       options:MTLResourceStorageModePrivate];
        q2_grid_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                        options:MTLResourceStorageModePrivate];
        xq2_grid_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                         options:MTLResourceStorageModePrivate];
        yq2_grid_real_buffer = [state.device newBufferWithLength:real_fft_bytes
                                                         options:MTLResourceStorageModePrivate];
      }
      id<MTLBuffer> fft_scratch_buffer = [state.device newBufferWithLength:complex_bytes
                                                                   options:MTLResourceStorageModePrivate];
      id<MTLBuffer> fft_twiddle_buffer = [state.device newBufferWithBytes:fft_twiddles.data()
                                                                   length:fft_twiddles.size() * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
      id<MTLBuffer> layout_stats_buffer = [state.device newBufferWithLength:static_cast<std::size_t>(stats_block_count) * sizeof(OpenTsneLayoutStats)
                                                                     options:MTLResourceStorageModePrivate];
      id<MTLBuffer> fft_grid_params_buffer = [state.device newBufferWithLength:sizeof(OpenTsneFFTGridParams)
                                                                       options:MTLResourceStorageModePrivate];
      id<MTLBuffer> center_buffer = [state.device newBufferWithLength:sizeof(Center2)
                                                              options:MTLResourceStorageModePrivate];
      id<MTLBuffer> repulsive_norm_buffer = nil;
      id<MTLBuffer> attractive_norm_buffer = nil;
      id<MTLBuffer> gradient_norm_buffer = nil;
      id<MTLBuffer> update_norm_buffer = nil;
      id<MTLBuffer> layout_norm_buffer = nil;
      if (record_costs) {
        const std::size_t norm_bytes = static_cast<std::size_t>(n) * sizeof(float);
        repulsive_norm_buffer = [state.device newBufferWithLength:norm_bytes options:MTLResourceStorageModeShared];
        attractive_norm_buffer = [state.device newBufferWithLength:norm_bytes options:MTLResourceStorageModeShared];
        gradient_norm_buffer = [state.device newBufferWithLength:norm_bytes options:MTLResourceStorageModeShared];
        update_norm_buffer = [state.device newBufferWithLength:norm_bytes options:MTLResourceStorageModeShared];
        layout_norm_buffer = [state.device newBufferWithLength:norm_bytes options:MTLResourceStorageModeShared];
      }
      if (mass_buffer == nil || mass_x_buffer == nil || mass_y_buffer == nil ||
          mass_fft_buffer == nil || mass_x_fft_buffer == nil || mass_y_fft_buffer == nil ||
          kernel_q_buffer == nil || kernel_q2_buffer == nil || q_grid_buffer == nil ||
          q2_grid_buffer == nil || xq2_grid_buffer == nil || yq2_grid_buffer == nil ||
          fft_scratch_buffer == nil || fft_twiddle_buffer == nil || layout_stats_buffer == nil ||
          fft_grid_params_buffer == nil || center_buffer == nil ||
          (use_mpsgraph_convolution &&
           (mass_real_buffer == nil || mass_x_real_buffer == nil || mass_y_real_buffer == nil ||
            kernel_q_real_buffer == nil || kernel_q2_real_buffer == nil ||
            q_grid_real_buffer == nil || q2_grid_real_buffer == nil ||
            xq2_grid_real_buffer == nil || yq2_grid_real_buffer == nil)) ||
          (record_costs && (repulsive_norm_buffer == nil || attractive_norm_buffer == nil ||
                            gradient_norm_buffer == nil || update_norm_buffer == nil ||
                            layout_norm_buffer == nil))) {
        Rcpp::stop("Failed to allocate Metal openTSNE FFT-grid buffers.");
      }

      const NSUInteger threads_clear = bounded_threads(state.opentsne_fft_clear_pipeline);
      const NSUInteger threads_scatter = bounded_threads(state.opentsne_fft_scatter_pipeline);
      const NSUInteger threads_epoch = bounded_threads(state.opentsne_fft_epoch_pipeline);
      const NSUInteger threads_center = bounded_threads(state.opentsne_center_pipeline);
      const MTLSize point_grid = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
      const MTLSize grid_cells = MTLSizeMake(static_cast<NSUInteger>(grid_total), 1, 1);
      const MTLSize fft_grid = MTLSizeMake(static_cast<NSUInteger>(fft_n),
                                          static_cast<NSUInteger>(fft_n),
                                          1);
      const std::uint32_t sum_block_size = 256u;
      const std::uint32_t sum_block_count =
        (static_cast<std::uint32_t>(n) + sum_block_size - 1u) / sum_block_size;
      const MTLSize sum_blocks = MTLSizeMake(static_cast<NSUInteger>(sum_block_count), 1, 1);
      const MTLSize stats_blocks = MTLSizeMake(static_cast<NSUInteger>(stats_block_count), 1, 1);
      MetalStageTimer stage_timer(metal_stage_timing_enabled() && !record_costs);
      NumericVector trace_iter(record_costs ? total_iter : 0);
      NumericVector trace_sum_q(record_costs ? total_iter : 0);
      NumericVector trace_repulsive_norm(record_costs ? total_iter : 0);
      NumericVector trace_attractive_norm(record_costs ? total_iter : 0);
      NumericVector trace_gradient_norm(record_costs ? total_iter : 0);
      NumericVector trace_update_norm(record_costs ? total_iter : 0);
      NumericVector trace_embedding_norm(record_costs ? total_iter : 0);

      MPSGraph* mpsgraph = nil;
      MPSGraphTensorDataDictionary* mpsgraph_feeds = nil;
      MPSGraphTensorDataDictionary* mpsgraph_results = nil;
      if (use_mpsgraph_convolution) {
        if (@available(macOS 14.0, *)) {
          mpsgraph = [[MPSGraph alloc] init];
          MPSShape* real_shape = @[ @(static_cast<NSUInteger>(fft_n)),
                                    @(static_cast<NSUInteger>(fft_n)) ];
          MPSGraphTensor* mass_tensor = [mpsgraph placeholderWithShape:real_shape
                                                               dataType:MPSDataTypeFloat32
                                                                   name:@"mass"];
          MPSGraphTensor* mass_x_tensor = [mpsgraph placeholderWithShape:real_shape
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"mass_x"];
          MPSGraphTensor* mass_y_tensor = [mpsgraph placeholderWithShape:real_shape
                                                                 dataType:MPSDataTypeFloat32
                                                                     name:@"mass_y"];
          MPSGraphTensor* kernel_q_tensor = [mpsgraph placeholderWithShape:real_shape
                                                                  dataType:MPSDataTypeFloat32
                                                                      name:@"kernel_q"];
          MPSGraphTensor* kernel_q2_tensor = [mpsgraph placeholderWithShape:real_shape
                                                                   dataType:MPSDataTypeFloat32
                                                                       name:@"kernel_q2"];
          MPSGraphFFTDescriptor* forward_desc = [MPSGraphFFTDescriptor descriptor];
          forward_desc.inverse = NO;
          forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
          MPSGraphFFTDescriptor* inverse_desc = [MPSGraphFFTDescriptor descriptor];
          inverse_desc.inverse = YES;
          inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
          MPSGraphTensor* mass_fft = [mpsgraph realToHermiteanFFTWithTensor:mass_tensor
                                                                       axes:@[@0, @1]
                                                                 descriptor:forward_desc
                                                                       name:@"mass_rfft2"];
          MPSGraphTensor* mass_x_fft = [mpsgraph realToHermiteanFFTWithTensor:mass_x_tensor
                                                                         axes:@[@0, @1]
                                                                   descriptor:forward_desc
                                                                         name:@"mass_x_rfft2"];
          MPSGraphTensor* mass_y_fft = [mpsgraph realToHermiteanFFTWithTensor:mass_y_tensor
                                                                         axes:@[@0, @1]
                                                                   descriptor:forward_desc
                                                                         name:@"mass_y_rfft2"];
          MPSGraphTensor* kernel_q_fft = [mpsgraph realToHermiteanFFTWithTensor:kernel_q_tensor
                                                                           axes:@[@0, @1]
                                                                     descriptor:forward_desc
                                                                           name:@"kernel_q_rfft2"];
          MPSGraphTensor* kernel_q2_fft = [mpsgraph realToHermiteanFFTWithTensor:kernel_q2_tensor
                                                                            axes:@[@0, @1]
                                                                      descriptor:forward_desc
                                                                            name:@"kernel_q2_rfft2"];
          MPSGraphTensor* q_spectrum = [mpsgraph multiplicationWithPrimaryTensor:mass_fft
                                                                 secondaryTensor:kernel_q_fft
                                                                            name:@"q_spectrum"];
          MPSGraphTensor* q2_spectrum = [mpsgraph multiplicationWithPrimaryTensor:mass_fft
                                                                  secondaryTensor:kernel_q2_fft
                                                                             name:@"q2_spectrum"];
          MPSGraphTensor* xq2_spectrum = [mpsgraph multiplicationWithPrimaryTensor:mass_x_fft
                                                                   secondaryTensor:kernel_q2_fft
                                                                              name:@"xq2_spectrum"];
          MPSGraphTensor* yq2_spectrum = [mpsgraph multiplicationWithPrimaryTensor:mass_y_fft
                                                                   secondaryTensor:kernel_q2_fft
                                                                              name:@"yq2_spectrum"];
          MPSGraphTensor* q_grid_tensor = [mpsgraph HermiteanToRealFFTWithTensor:q_spectrum
                                                                            axes:@[@0, @1]
                                                                      descriptor:inverse_desc
                                                                            name:@"q_grid"];
          MPSGraphTensor* q2_grid_tensor = [mpsgraph HermiteanToRealFFTWithTensor:q2_spectrum
                                                                             axes:@[@0, @1]
                                                                       descriptor:inverse_desc
                                                                             name:@"q2_grid"];
          MPSGraphTensor* xq2_grid_tensor = [mpsgraph HermiteanToRealFFTWithTensor:xq2_spectrum
                                                                              axes:@[@0, @1]
                                                                        descriptor:inverse_desc
                                                                              name:@"xq2_grid"];
          MPSGraphTensor* yq2_grid_tensor = [mpsgraph HermiteanToRealFFTWithTensor:yq2_spectrum
                                                                              axes:@[@0, @1]
                                                                        descriptor:inverse_desc
                                                                              name:@"yq2_grid"];

          MPSGraphTensorData* mass_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:mass_real_buffer
                                                                                   shape:real_shape
                                                                                dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* mass_x_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:mass_x_real_buffer
                                                                                     shape:real_shape
                                                                                  dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* mass_y_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:mass_y_real_buffer
                                                                                     shape:real_shape
                                                                                  dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* kernel_q_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:kernel_q_real_buffer
                                                                                       shape:real_shape
                                                                                    dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* kernel_q2_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:kernel_q2_real_buffer
                                                                                        shape:real_shape
                                                                                     dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* q_grid_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:q_grid_real_buffer
                                                                                     shape:real_shape
                                                                                  dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* q2_grid_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:q2_grid_real_buffer
                                                                                      shape:real_shape
                                                                                   dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* xq2_grid_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:xq2_grid_real_buffer
                                                                                       shape:real_shape
                                                                                    dataType:MPSDataTypeFloat32] autorelease];
          MPSGraphTensorData* yq2_grid_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:yq2_grid_real_buffer
                                                                                       shape:real_shape
                                                                                    dataType:MPSDataTypeFloat32] autorelease];
          mpsgraph_feeds = [@{
            mass_tensor: mass_data,
            mass_x_tensor: mass_x_data,
            mass_y_tensor: mass_y_data,
            kernel_q_tensor: kernel_q_data,
            kernel_q2_tensor: kernel_q2_data
          } retain];
          mpsgraph_results = [@{
            q_grid_tensor: q_grid_data,
            q2_grid_tensor: q2_grid_data,
            xq2_grid_tensor: xq2_grid_data,
            yq2_grid_tensor: yq2_grid_data
          } retain];
        }
      }

      for (int iter = 0; iter < total_iter; ++iter) {
        const bool in_early = iter < early_exaggeration_iter;
        const double phase_exaggeration = in_early ? early_exaggeration : exaggeration;
        const double phase_lr = learning_rate_auto ?
          static_cast<double>(n) / std::max(phase_exaggeration, std::numeric_limits<double>::min()) :
          learning_rate;
        OpenTsneMetalParams params{
          static_cast<std::uint32_t>(n),
          static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed),
          static_cast<float>(phase_lr),
          static_cast<float>(phase_exaggeration),
          static_cast<float>(in_early ? initial_momentum : final_momentum),
          static_cast<float>(min_gain),
          max_step,
          1.0f
        };

        if (use_mpsgraph_convolution) {
          id<MTLCommandBuffer> load_command = [state.queue commandBuffer];
          {
            id<MTLComputeCommandEncoder> encoder = [load_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_layout_stats_blocks_pipeline];
            [encoder setBuffer:current_buffer offset:0 atIndex:0];
            [encoder setBuffer:layout_stats_buffer offset:0 atIndex:1];
            [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
            [encoder setBytes:&stats_block_size length:sizeof(std::uint32_t) atIndex:3];
            [encoder dispatchThreads:stats_blocks
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [load_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_finalize_layout_stats_pipeline];
            [encoder setBuffer:layout_stats_buffer offset:0 atIndex:0];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:1];
            [encoder setBuffer:center_buffer offset:0 atIndex:2];
            [encoder setBytes:&stats_block_count length:sizeof(std::uint32_t) atIndex:3];
            [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
            [encoder setBytes:&grid_n length:sizeof(std::uint32_t) atIndex:5];
            [encoder setBytes:&fft_n length:sizeof(std::uint32_t) atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [load_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_clear_pipeline];
            [encoder setBuffer:mass_buffer offset:0 atIndex:0];
            [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
            [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
            [encoder dispatchThreads:grid_cells
               threadsPerThreadgroup:MTLSizeMake(threads_clear, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [load_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_scatter_pipeline];
            [encoder setBuffer:current_buffer offset:0 atIndex:0];
            [encoder setBuffer:mass_buffer offset:0 atIndex:1];
            [encoder setBuffer:mass_x_buffer offset:0 atIndex:2];
            [encoder setBuffer:mass_y_buffer offset:0 atIndex:3];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:4];
            [encoder dispatchThreads:point_grid
               threadsPerThreadgroup:MTLSizeMake(threads_scatter, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [load_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_mpsgraph_load_real_pipeline];
            [encoder setBuffer:mass_buffer offset:0 atIndex:0];
            [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
            [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
            [encoder setBuffer:mass_real_buffer offset:0 atIndex:3];
            [encoder setBuffer:mass_x_real_buffer offset:0 atIndex:4];
            [encoder setBuffer:mass_y_real_buffer offset:0 atIndex:5];
            [encoder setBuffer:kernel_q_real_buffer offset:0 atIndex:6];
            [encoder setBuffer:kernel_q2_real_buffer offset:0 atIndex:7];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:8];
            [encoder dispatchThreads:fft_grid
               threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_mpsgraph_load_real_pipeline)];
            [encoder endEncoding];
          }
          [load_command commit];
          [load_command waitUntilCompleted];
          if (load_command.status == MTLCommandBufferStatusError) {
            Rcpp::stop("Metal openTSNE MPSGraph preparation failed: %s",
                       ns_error_message(load_command.error).c_str());
          }
          if (@available(macOS 14.0, *)) {
            [mpsgraph runWithMTLCommandQueue:state.queue
                                       feeds:mpsgraph_feeds
                            targetOperations:nil
                           resultsDictionary:mpsgraph_results];
          }

          id<MTLCommandBuffer> update_command = [state.queue commandBuffer];
          {
            id<MTLComputeCommandEncoder> encoder = [update_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_pack_real4_pipeline];
            [encoder setBuffer:q_grid_real_buffer offset:0 atIndex:0];
            [encoder setBuffer:q2_grid_real_buffer offset:0 atIndex:1];
            [encoder setBuffer:xq2_grid_real_buffer offset:0 atIndex:2];
            [encoder setBuffer:yq2_grid_real_buffer offset:0 atIndex:3];
            [encoder setBuffer:q_grid_buffer offset:0 atIndex:4];
            [encoder setBuffer:q2_grid_buffer offset:0 atIndex:5];
            [encoder setBuffer:xq2_grid_buffer offset:0 atIndex:6];
            [encoder setBuffer:yq2_grid_buffer offset:0 atIndex:7];
            [encoder setBytes:&fft_total length:sizeof(std::uint32_t) atIndex:8];
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(fft_total), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(bounded_threads(state.opentsne_fft_pack_real4_pipeline), 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [update_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_sum_q_blocks_pipeline];
            [encoder setBuffer:current_buffer offset:0 atIndex:0];
            [encoder setBuffer:q_grid_buffer offset:0 atIndex:1];
            [encoder setBuffer:row_sums_buffer offset:0 atIndex:2];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
            [encoder setBytes:&sum_block_size length:sizeof(std::uint32_t) atIndex:4];
            [encoder dispatchThreads:sum_blocks
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [update_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_finalize_sum_q_pipeline];
            [encoder setBuffer:row_sums_buffer offset:0 atIndex:0];
            [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:1];
            [encoder setBytes:&sum_block_count length:sizeof(std::uint32_t) atIndex:2];
            [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> encoder = [update_command computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_epoch_pipeline];
            [encoder setBuffer:row_ptr_buffer offset:0 atIndex:0];
            [encoder setBuffer:col_buffer offset:0 atIndex:1];
            [encoder setBuffer:val_buffer offset:0 atIndex:2];
            [encoder setBuffer:current_buffer offset:0 atIndex:3];
            [encoder setBuffer:gains_buffer offset:0 atIndex:4];
            [encoder setBuffer:updates_buffer offset:0 atIndex:5];
            [encoder setBuffer:q2_grid_buffer offset:0 atIndex:6];
            [encoder setBuffer:xq2_grid_buffer offset:0 atIndex:7];
            [encoder setBuffer:yq2_grid_buffer offset:0 atIndex:8];
            [encoder setBytes:&params length:sizeof(OpenTsneMetalParams) atIndex:9];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:10];
            [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:11];
            [encoder dispatchThreads:point_grid
               threadsPerThreadgroup:MTLSizeMake(threads_epoch, 1, 1)];
            [encoder endEncoding];
          }
          {
            id<MTLComputeCommandEncoder> center_encoder = [update_command computeCommandEncoder];
            [center_encoder setComputePipelineState:state.opentsne_center_pipeline];
            [center_encoder setBuffer:current_buffer offset:0 atIndex:0];
            [center_encoder setBuffer:center_buffer offset:0 atIndex:1];
            [center_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
            [center_encoder dispatchThreads:point_grid
              threadsPerThreadgroup:MTLSizeMake(threads_center, 1, 1)];
            [center_encoder endEncoding];
          }
          [update_command commit];
          [update_command waitUntilCompleted];
          if (update_command.status == MTLCommandBufferStatusError) {
            Rcpp::stop("Metal openTSNE MPSGraph update failed: %s",
                       ns_error_message(update_command.error).c_str());
          }
          continue;
        }

        if (stage_timer.enabled()) {
          auto encode_layout_stats = [&](id<MTLCommandBuffer> command_buffer) {
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_layout_stats_blocks_pipeline];
              [encoder setBuffer:current_buffer offset:0 atIndex:0];
              [encoder setBuffer:layout_stats_buffer offset:0 atIndex:1];
              [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
              [encoder setBytes:&stats_block_size length:sizeof(std::uint32_t) atIndex:3];
              [encoder dispatchThreads:stats_blocks
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
              [encoder endEncoding];
            }
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_finalize_layout_stats_pipeline];
              [encoder setBuffer:layout_stats_buffer offset:0 atIndex:0];
              [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:1];
              [encoder setBuffer:center_buffer offset:0 atIndex:2];
              [encoder setBytes:&stats_block_count length:sizeof(std::uint32_t) atIndex:3];
              [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
              [encoder setBytes:&grid_n length:sizeof(std::uint32_t) atIndex:5];
              [encoder setBytes:&fft_n length:sizeof(std::uint32_t) atIndex:6];
              [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
              [encoder endEncoding];
            }
          };

          auto encode_clear_scatter_load = [&](id<MTLCommandBuffer> command_buffer) {
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_clear_pipeline];
              [encoder setBuffer:mass_buffer offset:0 atIndex:0];
              [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
              [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
              [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
              [encoder dispatchThreads:grid_cells
                 threadsPerThreadgroup:MTLSizeMake(threads_clear, 1, 1)];
              [encoder endEncoding];
            }
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_scatter_pipeline];
              [encoder setBuffer:current_buffer offset:0 atIndex:0];
              [encoder setBuffer:mass_buffer offset:0 atIndex:1];
              [encoder setBuffer:mass_x_buffer offset:0 atIndex:2];
              [encoder setBuffer:mass_y_buffer offset:0 atIndex:3];
              [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:4];
              [encoder dispatchThreads:point_grid
                 threadsPerThreadgroup:MTLSizeMake(threads_scatter, 1, 1)];
              [encoder endEncoding];
            }
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_load_pipeline];
              [encoder setBuffer:mass_buffer offset:0 atIndex:0];
              [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
              [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
              [encoder setBuffer:mass_fft_buffer offset:0 atIndex:3];
              [encoder setBuffer:mass_x_fft_buffer offset:0 atIndex:4];
              [encoder setBuffer:mass_y_fft_buffer offset:0 atIndex:5];
              [encoder setBuffer:kernel_q_buffer offset:0 atIndex:6];
              [encoder setBuffer:kernel_q2_buffer offset:0 atIndex:7];
              [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:8];
              [encoder dispatchThreads:fft_grid
                 threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_load_pipeline)];
              [encoder endEncoding];
            }
          };

          auto encode_fft_forward = [&](id<MTLCommandBuffer> command_buffer) {
            encode_fft_2d_metal(state, command_buffer, mass_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
            encode_fft_2d_metal(state, command_buffer, mass_x_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
            encode_fft_2d_metal(state, command_buffer, mass_y_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
            encode_fft_2d_metal(state, command_buffer, kernel_q_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
            encode_fft_2d_metal(state, command_buffer, kernel_q2_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
          };

          auto encode_fft_convolution = [&](id<MTLCommandBuffer> command_buffer) {
            encode_fft_convolution_metal(state, command_buffer, mass_fft_buffer, kernel_q_buffer,
                                         q_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
            encode_fft_convolution_metal(state, command_buffer, mass_fft_buffer, kernel_q2_buffer,
                                         q2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
            encode_fft_convolution_metal(state, command_buffer, mass_x_fft_buffer, kernel_q2_buffer,
                                         xq2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
            encode_fft_convolution_metal(state, command_buffer, mass_y_fft_buffer, kernel_q2_buffer,
                                         yq2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
          };

          auto encode_sum_q = [&](id<MTLCommandBuffer> command_buffer) {
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_sum_q_blocks_pipeline];
              [encoder setBuffer:current_buffer offset:0 atIndex:0];
              [encoder setBuffer:q_grid_buffer offset:0 atIndex:1];
              [encoder setBuffer:row_sums_buffer offset:0 atIndex:2];
              [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
              [encoder setBytes:&sum_block_size length:sizeof(std::uint32_t) atIndex:4];
              [encoder dispatchThreads:sum_blocks
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
              [encoder endEncoding];
            }
            {
              id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
              [encoder setComputePipelineState:state.opentsne_fft_finalize_sum_q_pipeline];
              [encoder setBuffer:row_sums_buffer offset:0 atIndex:0];
              [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:1];
              [encoder setBytes:&sum_block_count length:sizeof(std::uint32_t) atIndex:2];
              [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:3];
              [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
              [encoder endEncoding];
            }
          };

          auto encode_epoch_update = [&](id<MTLCommandBuffer> command_buffer) {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:state.opentsne_fft_epoch_pipeline];
            [encoder setBuffer:row_ptr_buffer offset:0 atIndex:0];
            [encoder setBuffer:col_buffer offset:0 atIndex:1];
            [encoder setBuffer:val_buffer offset:0 atIndex:2];
            [encoder setBuffer:current_buffer offset:0 atIndex:3];
            [encoder setBuffer:gains_buffer offset:0 atIndex:4];
            [encoder setBuffer:updates_buffer offset:0 atIndex:5];
            [encoder setBuffer:q2_grid_buffer offset:0 atIndex:6];
            [encoder setBuffer:xq2_grid_buffer offset:0 atIndex:7];
            [encoder setBuffer:yq2_grid_buffer offset:0 atIndex:8];
            [encoder setBytes:&params length:sizeof(OpenTsneMetalParams) atIndex:9];
            [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:10];
            [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:11];
            [encoder dispatchThreads:point_grid
               threadsPerThreadgroup:MTLSizeMake(threads_epoch, 1, 1)];
            [encoder endEncoding];
          };

          auto encode_center = [&](id<MTLCommandBuffer> command_buffer) {
            id<MTLComputeCommandEncoder> center_encoder = [command_buffer computeCommandEncoder];
            [center_encoder setComputePipelineState:state.opentsne_center_pipeline];
            [center_encoder setBuffer:current_buffer offset:0 atIndex:0];
            [center_encoder setBuffer:center_buffer offset:0 atIndex:1];
            [center_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
            [center_encoder dispatchThreads:point_grid
              threadsPerThreadgroup:MTLSizeMake(threads_center, 1, 1)];
            [center_encoder endEncoding];
          };

          run_timed_metal_stage(state, stage_timer, "layout_stats", encode_layout_stats);
          run_timed_metal_stage(state, stage_timer, "clear_scatter_load", encode_clear_scatter_load);
          run_timed_metal_stage(state, stage_timer, "fft_forward", encode_fft_forward);
          run_timed_metal_stage(state, stage_timer, "fft_convolution", encode_fft_convolution);
          run_timed_metal_stage(state, stage_timer, "sum_q", encode_sum_q);
          run_timed_metal_stage(state, stage_timer, "epoch_update", encode_epoch_update);
          run_timed_metal_stage(state, stage_timer, "center", encode_center);
          continue;
        }

        id<MTLCommandBuffer> fft_command = [state.queue commandBuffer];
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_layout_stats_blocks_pipeline];
          [encoder setBuffer:current_buffer offset:0 atIndex:0];
          [encoder setBuffer:layout_stats_buffer offset:0 atIndex:1];
          [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
          [encoder setBytes:&stats_block_size length:sizeof(std::uint32_t) atIndex:3];
          [encoder dispatchThreads:stats_blocks
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_finalize_layout_stats_pipeline];
          [encoder setBuffer:layout_stats_buffer offset:0 atIndex:0];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:1];
          [encoder setBuffer:center_buffer offset:0 atIndex:2];
          [encoder setBytes:&stats_block_count length:sizeof(std::uint32_t) atIndex:3];
          [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
          [encoder setBytes:&grid_n length:sizeof(std::uint32_t) atIndex:5];
          [encoder setBytes:&fft_n length:sizeof(std::uint32_t) atIndex:6];
          [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_clear_pipeline];
          [encoder setBuffer:mass_buffer offset:0 atIndex:0];
          [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
          [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
          [encoder dispatchThreads:grid_cells
             threadsPerThreadgroup:MTLSizeMake(threads_clear, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_scatter_pipeline];
          [encoder setBuffer:current_buffer offset:0 atIndex:0];
          [encoder setBuffer:mass_buffer offset:0 atIndex:1];
          [encoder setBuffer:mass_x_buffer offset:0 atIndex:2];
          [encoder setBuffer:mass_y_buffer offset:0 atIndex:3];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:4];
          [encoder dispatchThreads:point_grid
             threadsPerThreadgroup:MTLSizeMake(threads_scatter, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_load_pipeline];
          [encoder setBuffer:mass_buffer offset:0 atIndex:0];
          [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
          [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
          [encoder setBuffer:mass_fft_buffer offset:0 atIndex:3];
          [encoder setBuffer:mass_x_fft_buffer offset:0 atIndex:4];
          [encoder setBuffer:mass_y_fft_buffer offset:0 atIndex:5];
          [encoder setBuffer:kernel_q_buffer offset:0 atIndex:6];
          [encoder setBuffer:kernel_q2_buffer offset:0 atIndex:7];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:8];
          [encoder dispatchThreads:fft_grid
             threadsPerThreadgroup:metal_threadgroup_2d(state.opentsne_fft_load_pipeline)];
          [encoder endEncoding];
        }

        encode_fft_2d_metal(state, fft_command, mass_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
        encode_fft_2d_metal(state, fft_command, mass_x_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
        encode_fft_2d_metal(state, fft_command, mass_y_fft_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
        encode_fft_2d_metal(state, fft_command, kernel_q_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
        encode_fft_2d_metal(state, fft_command, kernel_q2_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, false);
        encode_fft_convolution_metal(state, fft_command, mass_fft_buffer, kernel_q_buffer,
                                     q_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
        encode_fft_convolution_metal(state, fft_command, mass_fft_buffer, kernel_q2_buffer,
                                     q2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
        encode_fft_convolution_metal(state, fft_command, mass_x_fft_buffer, kernel_q2_buffer,
                                     xq2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
        encode_fft_convolution_metal(state, fft_command, mass_y_fft_buffer, kernel_q2_buffer,
                                     yq2_grid_buffer, fft_scratch_buffer, fft_twiddle_buffer, fft_n, log_fft, fft_total);
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_sum_q_blocks_pipeline];
          [encoder setBuffer:current_buffer offset:0 atIndex:0];
          [encoder setBuffer:q_grid_buffer offset:0 atIndex:1];
          [encoder setBuffer:row_sums_buffer offset:0 atIndex:2];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:3];
          [encoder setBytes:&sum_block_size length:sizeof(std::uint32_t) atIndex:4];
          [encoder dispatchThreads:sum_blocks
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:state.opentsne_fft_finalize_sum_q_pipeline];
          [encoder setBuffer:row_sums_buffer offset:0 atIndex:0];
          [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:1];
          [encoder setBytes:&sum_block_count length:sizeof(std::uint32_t) atIndex:2];
          [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:3];
          [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> encoder = [fft_command computeCommandEncoder];
          [encoder setComputePipelineState:record_costs ?
             state.opentsne_fft_epoch_debug_pipeline :
             state.opentsne_fft_epoch_pipeline];
          [encoder setBuffer:row_ptr_buffer offset:0 atIndex:0];
          [encoder setBuffer:col_buffer offset:0 atIndex:1];
          [encoder setBuffer:val_buffer offset:0 atIndex:2];
          [encoder setBuffer:current_buffer offset:0 atIndex:3];
          [encoder setBuffer:gains_buffer offset:0 atIndex:4];
          [encoder setBuffer:updates_buffer offset:0 atIndex:5];
          [encoder setBuffer:q2_grid_buffer offset:0 atIndex:6];
          [encoder setBuffer:xq2_grid_buffer offset:0 atIndex:7];
          [encoder setBuffer:yq2_grid_buffer offset:0 atIndex:8];
          [encoder setBytes:&params length:sizeof(OpenTsneMetalParams) atIndex:9];
          [encoder setBuffer:fft_grid_params_buffer offset:0 atIndex:10];
          [encoder setBuffer:inv_sum_q_buffer offset:0 atIndex:11];
          if (record_costs) {
            [encoder setBuffer:repulsive_norm_buffer offset:0 atIndex:12];
            [encoder setBuffer:attractive_norm_buffer offset:0 atIndex:13];
            [encoder setBuffer:gradient_norm_buffer offset:0 atIndex:14];
            [encoder setBuffer:update_norm_buffer offset:0 atIndex:15];
            [encoder setBuffer:layout_norm_buffer offset:0 atIndex:16];
          }
          [encoder dispatchThreads:point_grid
             threadsPerThreadgroup:MTLSizeMake(threads_epoch, 1, 1)];
          [encoder endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> center_encoder = [fft_command computeCommandEncoder];
          [center_encoder setComputePipelineState:state.opentsne_center_pipeline];
          [center_encoder setBuffer:current_buffer offset:0 atIndex:0];
          [center_encoder setBuffer:center_buffer offset:0 atIndex:1];
          [center_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
          [center_encoder dispatchThreads:point_grid
            threadsPerThreadgroup:MTLSizeMake(threads_center, 1, 1)];
          [center_encoder endEncoding];
        }
        [fft_command commit];
        [fft_command waitUntilCompleted];
        if (fft_command.status == MTLCommandBufferStatusError) {
          Rcpp::stop("Metal openTSNE FFT-grid command failed: %s", ns_error_message(fft_command.error).c_str());
        }
        if (record_costs) {
          auto sum_norm_buffer = [&](id<MTLBuffer> buffer) -> double {
            const float* values = static_cast<const float*>([buffer contents]);
            double total = 0.0;
            for (int row = 0; row < n; ++row) {
              total += static_cast<double>(values[row]);
            }
            return std::sqrt(std::max(0.0, total));
          };
          const float inv_sum_q_value = *static_cast<const float*>([inv_sum_q_buffer contents]);
          std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
          double layout2 = 0.0;
          for (float value : current) {
            layout2 += static_cast<double>(value) * static_cast<double>(value);
          }
          trace_iter[iter] = iter + 1;
          trace_sum_q[iter] = inv_sum_q_value > 0.0f ?
            1.0 / static_cast<double>(inv_sum_q_value) :
            NA_REAL;
          trace_repulsive_norm[iter] = sum_norm_buffer(repulsive_norm_buffer);
          trace_attractive_norm[iter] = sum_norm_buffer(attractive_norm_buffer);
          trace_gradient_norm[iter] = sum_norm_buffer(gradient_norm_buffer);
          trace_update_norm[iter] = sum_norm_buffer(update_norm_buffer);
          trace_embedding_norm[iter] = std::sqrt(std::max(0.0, layout2));
        }
      }

      std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
      NumericMatrix layout(n, 2);
      for (int i = 0; i < n; ++i) {
        layout(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
        layout(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
      }

      [row_ptr_buffer release];
      [col_buffer release];
      [val_buffer release];
      [current_buffer release];
      [gains_buffer release];
      [updates_buffer release];
      [row_sums_buffer release];
      [mass_buffer release];
      [mass_x_buffer release];
      [mass_y_buffer release];
      [mass_fft_buffer release];
      [mass_x_fft_buffer release];
      [mass_y_fft_buffer release];
      [kernel_q_buffer release];
      [kernel_q2_buffer release];
      [q_grid_buffer release];
      [q2_grid_buffer release];
      [xq2_grid_buffer release];
      [yq2_grid_buffer release];
      if (mass_real_buffer != nil) [mass_real_buffer release];
      if (mass_x_real_buffer != nil) [mass_x_real_buffer release];
      if (mass_y_real_buffer != nil) [mass_y_real_buffer release];
      if (kernel_q_real_buffer != nil) [kernel_q_real_buffer release];
      if (kernel_q2_real_buffer != nil) [kernel_q2_real_buffer release];
      if (q_grid_real_buffer != nil) [q_grid_real_buffer release];
      if (q2_grid_real_buffer != nil) [q2_grid_real_buffer release];
      if (xq2_grid_real_buffer != nil) [xq2_grid_real_buffer release];
      if (yq2_grid_real_buffer != nil) [yq2_grid_real_buffer release];
      [fft_scratch_buffer release];
      [fft_twiddle_buffer release];
      [layout_stats_buffer release];
      [fft_grid_params_buffer release];
      [center_buffer release];
      [inv_sum_q_buffer release];
      if (mpsgraph_feeds != nil) [mpsgraph_feeds release];
      if (mpsgraph_results != nil) [mpsgraph_results release];
      if (mpsgraph != nil) [mpsgraph release];
      if (repulsive_norm_buffer != nil) [repulsive_norm_buffer release];
      if (attractive_norm_buffer != nil) [attractive_norm_buffer release];
      if (gradient_norm_buffer != nil) [gradient_norm_buffer release];
      if (update_norm_buffer != nil) [update_norm_buffer release];
      if (layout_norm_buffer != nil) [layout_norm_buffer release];

      Rcpp::DataFrame metal_trace = Rcpp::DataFrame::create(
        Rcpp::Named("iter") = trace_iter,
        Rcpp::Named("sum_q") = trace_sum_q,
        Rcpp::Named("repulsive_norm") = trace_repulsive_norm,
        Rcpp::Named("attractive_norm") = trace_attractive_norm,
        Rcpp::Named("gradient_norm") = trace_gradient_norm,
        Rcpp::Named("update_norm") = trace_update_norm,
        Rcpp::Named("embedding_norm") = trace_embedding_norm
      );

      return List::create(
        Rcpp::Named("Y") = layout,
        Rcpp::Named("costs") = NumericVector(0),
        Rcpp::Named("itercosts") = NumericVector(0),
        Rcpp::Named("metal_trace") = metal_trace,
        Rcpp::Named("optimizer") = "opentsne_fitsne_fft_grid_native_metal",
        Rcpp::Named("repulsion") = use_mpsgraph_convolution ?
          "fft_grid_mpsgraph_metal" : "fft_grid_metal",
        Rcpp::Named("probabilities") = "symmetric_sparse_knn_cpu_prepared_for_metal",
        Rcpp::Named("repulsion_block_size") = static_cast<int>(grid_n),
        Rcpp::Named("n_threads") = NA_INTEGER,
        Rcpp::Named("learning_rate") = learning_rate_auto ? NA_REAL : learning_rate,
        Rcpp::Named("learning_rate_early") = static_cast<double>(n) / std::max(early_exaggeration, std::numeric_limits<double>::min()),
        Rcpp::Named("learning_rate_normal") = static_cast<double>(n) / std::max(exaggeration, std::numeric_limits<double>::min()),
        Rcpp::Named("metal_stage_timing") = stage_timer.to_data_frame()
      );
    }

    const NSUInteger threads_sum = bounded_threads(state.opentsne_sum_q_pipeline);
    const NSUInteger threads_epoch = bounded_threads(state.opentsne_epoch_pipeline);
    const NSUInteger threads_center = bounded_threads(state.opentsne_center_pipeline);
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    const MTLSize sum_threadgroup = MTLSizeMake(threads_sum, 1, 1);
    const MTLSize epoch_threadgroup = MTLSizeMake(threads_epoch, 1, 1);
    const MTLSize center_threadgroup = MTLSizeMake(threads_center, 1, 1);

    for (int iter = 0; iter < total_iter; ++iter) {
      const bool in_early = iter < early_exaggeration_iter;
      const double phase_exaggeration = in_early ? early_exaggeration : exaggeration;
      const double phase_lr = learning_rate_auto ?
        static_cast<double>(n) / std::max(phase_exaggeration, std::numeric_limits<double>::min()) :
        learning_rate;

      OpenTsneMetalParams sum_params{
        static_cast<std::uint32_t>(n),
        static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed),
        static_cast<float>(phase_lr),
        static_cast<float>(phase_exaggeration),
        static_cast<float>(in_early ? initial_momentum : final_momentum),
        static_cast<float>(min_gain),
        max_step,
        1.0f
      };

      id<MTLCommandBuffer> sum_command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> sum_encoder = [sum_command computeCommandEncoder];
      [sum_encoder setComputePipelineState:state.opentsne_sum_q_pipeline];
      [sum_encoder setBuffer:current_buffer offset:0 atIndex:0];
      [sum_encoder setBuffer:row_sums_buffer offset:0 atIndex:1];
      [sum_encoder setBytes:&sum_params length:sizeof(OpenTsneMetalParams) atIndex:2];
      [sum_encoder dispatchThreads:grid_size threadsPerThreadgroup:sum_threadgroup];
      [sum_encoder endEncoding];
      [sum_command commit];
      [sum_command waitUntilCompleted];
      if (sum_command.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal openTSNE normalization command failed: %s", ns_error_message(sum_command.error).c_str());
      }

      std::memcpy(row_sums.data(), [row_sums_buffer contents], row_sums.size() * sizeof(float));
      double sum_q = 0.0;
      for (float value : row_sums) sum_q += static_cast<double>(value);
      if (!std::isfinite(sum_q) || sum_q <= 0.0) sum_q = std::numeric_limits<double>::min();

      OpenTsneMetalParams params = sum_params;
      params.inv_sum_q = static_cast<float>(1.0 / sum_q);

      id<MTLCommandBuffer> epoch_command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> epoch_encoder = [epoch_command computeCommandEncoder];
      [epoch_encoder setComputePipelineState:state.opentsne_epoch_pipeline];
      [epoch_encoder setBuffer:row_ptr_buffer offset:0 atIndex:0];
      [epoch_encoder setBuffer:col_buffer offset:0 atIndex:1];
      [epoch_encoder setBuffer:val_buffer offset:0 atIndex:2];
      [epoch_encoder setBuffer:current_buffer offset:0 atIndex:3];
      [epoch_encoder setBuffer:gains_buffer offset:0 atIndex:4];
      [epoch_encoder setBuffer:updates_buffer offset:0 atIndex:5];
      [epoch_encoder setBytes:&params length:sizeof(OpenTsneMetalParams) atIndex:6];
      [epoch_encoder dispatchThreads:grid_size threadsPerThreadgroup:epoch_threadgroup];
      [epoch_encoder endEncoding];
      [epoch_command commit];
      [epoch_command waitUntilCompleted];
      if (epoch_command.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal openTSNE epoch command failed: %s", ns_error_message(epoch_command.error).c_str());
      }

      std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
      double mean_x = 0.0;
      double mean_y = 0.0;
      for (int i = 0; i < n; ++i) {
        mean_x += current[static_cast<std::size_t>(i) * 2u];
        mean_y += current[static_cast<std::size_t>(i) * 2u + 1u];
      }
      mean_x /= static_cast<double>(n);
      mean_y /= static_cast<double>(n);
      const Center2 center{static_cast<float>(mean_x), static_cast<float>(mean_y)};
      id<MTLCommandBuffer> center_command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> center_encoder = [center_command computeCommandEncoder];
      [center_encoder setComputePipelineState:state.opentsne_center_pipeline];
      [center_encoder setBuffer:current_buffer offset:0 atIndex:0];
      [center_encoder setBytes:&center length:sizeof(Center2) atIndex:1];
      [center_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:2];
      [center_encoder dispatchThreads:grid_size threadsPerThreadgroup:center_threadgroup];
      [center_encoder endEncoding];
      [center_command commit];
      [center_command waitUntilCompleted];
      if (center_command.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal openTSNE centering command failed: %s", ns_error_message(center_command.error).c_str());
      }
    }

    std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
    NumericMatrix layout(n, 2);
    for (int i = 0; i < n; ++i) {
      layout(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
      layout(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
    }

    [row_ptr_buffer release];
    [col_buffer release];
    [val_buffer release];
    [current_buffer release];
    [gains_buffer release];
    [updates_buffer release];
    [row_sums_buffer release];
    [inv_sum_q_buffer release];

    return List::create(
      Rcpp::Named("Y") = layout,
      Rcpp::Named("costs") = NumericVector(0),
      Rcpp::Named("itercosts") = NumericVector(0),
      Rcpp::Named("optimizer") = "opentsne_exact_sparse_native_metal",
      Rcpp::Named("repulsion") = "exact_metal",
      Rcpp::Named("probabilities") = "symmetric_sparse_knn_cpu_prepared_for_metal",
      Rcpp::Named("repulsion_block_size") = NA_INTEGER,
      Rcpp::Named("n_threads") = NA_INTEGER,
      Rcpp::Named("learning_rate") = learning_rate_auto ? NA_REAL : learning_rate,
      Rcpp::Named("learning_rate_early") = static_cast<double>(n) / std::max(early_exaggeration, std::numeric_limits<double>::min()),
      Rcpp::Named("learning_rate_normal") = static_cast<double>(n) / std::max(exaggeration, std::numeric_limits<double>::min())
    );
  }
}

NumericMatrix project_embedding_knn_metal_impl(NumericMatrix reference_layout,
                                               IntegerMatrix projection_indices,
                                               NumericMatrix projection_distances) {
  validate_projection_inputs(reference_layout, projection_indices, projection_distances);
  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> out = run_projection_metal(
      state,
      reference_layout,
      projection_indices,
      projection_distances,
      true
    );
    return float_to_numeric_matrix(
      out,
      projection_indices.nrow(),
      reference_layout.ncol()
    );
  }
}

List project_embedding_affine_metal_impl(NumericMatrix reference_data,
                                         NumericMatrix query_data,
                                         NumericMatrix reference_layout,
                                         IntegerMatrix projection_indices,
                                         NumericMatrix projection_distances,
                                         int max_neighbors,
                                         double ridge,
                                         double max_extrapolation) {
  const int n_reference = reference_layout.nrow();
  const int n_components = reference_layout.ncol();
  const int n_query = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();
  const int n_features = reference_data.ncol();

  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (reference_data.nrow() != n_reference) {
    Rcpp::stop("reference_data and reference_layout must have the same number of rows");
  }
  if (query_data.nrow() != n_query) {
    Rcpp::stop("query_data and projection_indices must have the same number of rows");
  }
  if (query_data.ncol() != n_features) {
    Rcpp::stop("reference_data and query_data must have the same number of columns");
  }
  if (n_components != 2) {
    Rcpp::stop("Metal affine landmark projection currently supports two-dimensional layouts.");
  }
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (projection_k > kMaxMetalProjectionNeighbors) {
    Rcpp::stop("Metal affine landmark projection supports at most %d projection neighbors.", kMaxMetalProjectionNeighbors);
  }
  if (projection_distances.nrow() != n_query ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  if (max_neighbors < 3) max_neighbors = 3;
  max_neighbors = std::min(max_neighbors, projection_k);
  max_neighbors = std::min(max_neighbors, 12);
  if (!std::isfinite(ridge) || ridge <= 0.0) ridge = 1e-3;
  if (!std::isfinite(max_extrapolation) || max_extrapolation <= 0.0) {
    max_extrapolation = 2.5;
  }
  for (int i = 0; i < n_query; ++i) {
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      const double d = projection_distances(i, j);
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
    }
  }

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> reference_values = numeric_matrix_to_float(reference_data);
    std::vector<float> query_values = numeric_matrix_to_float(query_data);
    std::vector<float> layout_values = numeric_matrix_to_float(reference_layout);
    std::vector<float> distance_values = numeric_matrix_to_float(projection_distances);
    std::vector<float> out(static_cast<std::size_t>(n_query) * 2u, 0.0f);
    std::vector<float> confidence(static_cast<std::size_t>(n_query), 0.0f);
    std::vector<std::int32_t> used(static_cast<std::size_t>(n_query), 0);
    std::vector<std::int32_t> fallback(static_cast<std::size_t>(n_query), 0);

    const std::uint32_t n_reference_u = static_cast<std::uint32_t>(n_reference);
    const std::uint32_t n_query_u = static_cast<std::uint32_t>(n_query);
    const std::uint32_t n_features_u = static_cast<std::uint32_t>(n_features);
    const std::uint32_t projection_k_u = static_cast<std::uint32_t>(projection_k);
    const std::uint32_t max_neighbors_u = static_cast<std::uint32_t>(max_neighbors);
    const float ridge_f = static_cast<float>(ridge);
    const float max_extrapolation_f = static_cast<float>(max_extrapolation);

    id<MTLBuffer> reference_buffer = [state.device newBufferWithBytes:reference_values.data()
                                                               length:reference_values.size() * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [state.device newBufferWithBytes:query_values.data()
                                                           length:query_values.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:layout_values.data()
                                                            length:layout_values.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [state.device newBufferWithBytes:projection_indices.begin()
                                                           length:static_cast<std::size_t>(n_query) * projection_k * sizeof(int)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> distance_buffer = [state.device newBufferWithBytes:distance_values.data()
                                                              length:distance_values.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [state.device newBufferWithBytes:out.data()
                                                        length:out.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> confidence_buffer = [state.device newBufferWithBytes:confidence.data()
                                                                length:confidence.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> used_buffer = [state.device newBufferWithBytes:used.data()
                                                         length:used.size() * sizeof(std::int32_t)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> fallback_buffer = [state.device newBufferWithBytes:fallback.data()
                                                             length:fallback.size() * sizeof(std::int32_t)
                                                            options:MTLResourceStorageModeShared];
    if (reference_buffer == nil || query_buffer == nil || layout_buffer == nil ||
        index_buffer == nil || distance_buffer == nil || out_buffer == nil ||
        confidence_buffer == nil || used_buffer == nil || fallback_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal affine projection buffers.");
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.affine_project_pipeline];
    [encoder setBuffer:reference_buffer offset:0 atIndex:0];
    [encoder setBuffer:query_buffer offset:0 atIndex:1];
    [encoder setBuffer:layout_buffer offset:0 atIndex:2];
    [encoder setBuffer:index_buffer offset:0 atIndex:3];
    [encoder setBuffer:distance_buffer offset:0 atIndex:4];
    [encoder setBuffer:out_buffer offset:0 atIndex:5];
    [encoder setBuffer:confidence_buffer offset:0 atIndex:6];
    [encoder setBuffer:used_buffer offset:0 atIndex:7];
    [encoder setBuffer:fallback_buffer offset:0 atIndex:8];
    [encoder setBytes:&n_reference_u length:sizeof(std::uint32_t) atIndex:9];
    [encoder setBytes:&n_query_u length:sizeof(std::uint32_t) atIndex:10];
    [encoder setBytes:&n_features_u length:sizeof(std::uint32_t) atIndex:11];
    [encoder setBytes:&projection_k_u length:sizeof(std::uint32_t) atIndex:12];
    [encoder setBytes:&max_neighbors_u length:sizeof(std::uint32_t) atIndex:13];
    [encoder setBytes:&ridge_f length:sizeof(float) atIndex:14];
    [encoder setBytes:&max_extrapolation_f length:sizeof(float) atIndex:15];
    dispatch_rows(encoder, state.affine_project_pipeline, n_query);
    [encoder endEncoding];
    wait_for_command(command_buffer, "Metal affine landmark projection");

    std::memcpy(out.data(), [out_buffer contents], out.size() * sizeof(float));
    std::memcpy(confidence.data(), [confidence_buffer contents], confidence.size() * sizeof(float));
    std::memcpy(used.data(), [used_buffer contents], used.size() * sizeof(std::int32_t));
    std::memcpy(fallback.data(), [fallback_buffer contents], fallback.size() * sizeof(std::int32_t));

    NumericMatrix layout = float_to_numeric_matrix(out, n_query, 2);
    NumericVector confidence_out(n_query);
    IntegerVector used_out(n_query);
    IntegerVector fallback_out(n_query);
    for (int i = 0; i < n_query; ++i) {
      confidence_out[i] = static_cast<double>(confidence[static_cast<std::size_t>(i)]);
      used_out[i] = used[static_cast<std::size_t>(i)];
      fallback_out[i] = fallback[static_cast<std::size_t>(i)];
    }
    layout.attr("projection_method") = "local_affine_knn_projection_metal";
    layout.attr("projection_backend") = "metal";

    [reference_buffer release];
    [query_buffer release];
    [layout_buffer release];
    [index_buffer release];
    [distance_buffer release];
    [out_buffer release];
    [confidence_buffer release];
    [used_buffer release];
    [fallback_buffer release];

    return List::create(
      Rcpp::Named("layout") = layout,
      Rcpp::Named("confidence") = confidence_out,
      Rcpp::Named("used_neighbors") = used_out,
      Rcpp::Named("fallback") = fallback_out,
      Rcpp::Named("method") = "local_affine_knn_projection_metal",
      Rcpp::Named("backend") = "metal",
      Rcpp::Named("max_neighbors") = max_neighbors,
      Rcpp::Named("ridge") = ridge,
      Rcpp::Named("max_extrapolation") = max_extrapolation
    );
  }
}

NumericMatrix interpolate_landmark_layout_metal_impl(NumericMatrix landmark_layout,
                                                     IntegerVector landmark_indices,
                                                     IntegerMatrix projection_indices,
                                                     NumericMatrix projection_distances,
                                                     int n) {
  validate_projection_inputs(landmark_layout, projection_indices, projection_distances);
  const int n_landmarks = landmark_layout.nrow();
  const int n_components = landmark_layout.ncol();
  const int k = projection_indices.ncol();
  if (landmark_indices.size() != n_landmarks) {
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

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> reference = numeric_matrix_to_float(landmark_layout);
    std::vector<float> distances = numeric_matrix_to_float(projection_distances);
    std::vector<float> out(static_cast<std::size_t>(n) * n_components, 0.0f);

    id<MTLBuffer> reference_buffer = [state.device newBufferWithBytes:reference.data()
                                                               length:reference.size() * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [state.device newBufferWithBytes:projection_indices.begin()
                                                           length:static_cast<std::size_t>(n) * k * sizeof(int)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> distance_buffer = [state.device newBufferWithBytes:distances.data()
                                                              length:distances.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> landmark_index_buffer = [state.device newBufferWithBytes:landmark_indices.begin()
                                                                    length:static_cast<std::size_t>(n_landmarks) * sizeof(int)
                                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [state.device newBufferWithLength:out.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (reference_buffer == nil || index_buffer == nil || distance_buffer == nil ||
        landmark_index_buffer == nil || out_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal landmark interpolation buffers.");
    }

    const std::uint32_t n_landmarks_u = static_cast<std::uint32_t>(n_landmarks);
    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t k_u = static_cast<std::uint32_t>(k);
    const std::uint32_t n_components_u = static_cast<std::uint32_t>(n_components);
    const std::uint32_t average_zeros_u = 0u;

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> project_encoder = [command_buffer computeCommandEncoder];
    [project_encoder setComputePipelineState:state.project_pipeline];
    [project_encoder setBuffer:reference_buffer offset:0 atIndex:0];
    [project_encoder setBuffer:index_buffer offset:0 atIndex:1];
    [project_encoder setBuffer:distance_buffer offset:0 atIndex:2];
    [project_encoder setBuffer:out_buffer offset:0 atIndex:3];
    [project_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:4];
    [project_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:5];
    [project_encoder setBytes:&k_u length:sizeof(std::uint32_t) atIndex:6];
    [project_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:7];
    [project_encoder setBytes:&average_zeros_u length:sizeof(std::uint32_t) atIndex:8];
    dispatch_rows(project_encoder, state.project_pipeline, n);
    [project_encoder endEncoding];

    id<MTLComputeCommandEncoder> overwrite_encoder = [command_buffer computeCommandEncoder];
    [overwrite_encoder setComputePipelineState:state.overwrite_landmarks_pipeline];
    [overwrite_encoder setBuffer:out_buffer offset:0 atIndex:0];
    [overwrite_encoder setBuffer:reference_buffer offset:0 atIndex:1];
    [overwrite_encoder setBuffer:landmark_index_buffer offset:0 atIndex:2];
    [overwrite_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:3];
    [overwrite_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
    [overwrite_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:5];
    dispatch_rows(overwrite_encoder, state.overwrite_landmarks_pipeline, n_landmarks * n_components);
    [overwrite_encoder endEncoding];
    wait_for_command(command_buffer, "landmark interpolation");

    std::memcpy(out.data(), [out_buffer contents], out.size() * sizeof(float));
    [reference_buffer release];
    [index_buffer release];
    [distance_buffer release];
    [landmark_index_buffer release];
    [out_buffer release];
    return float_to_numeric_matrix(out, n, n_components);
  }
}

NumericMatrix landmark_project_interpolate_metal_impl(NumericMatrix landmark_data,
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
  if (k > kMaxMetalProjectionNeighbors) {
    Rcpp::stop("Metal fused landmark projection currently supports at most %d neighbors.", kMaxMetalProjectionNeighbors);
  }
  for (int i = 0; i < landmark_indices.size(); ++i) {
    if (landmark_indices[i] < 1 || landmark_indices[i] > n) {
      Rcpp::stop("landmark indices out of range");
    }
  }

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> landmark_values = numeric_matrix_to_row_major_float(landmark_data);
    std::vector<float> query_values = numeric_matrix_to_row_major_float(query_data);
    std::vector<float> layout_values = numeric_matrix_to_float(landmark_layout);
    std::vector<float> out(static_cast<std::size_t>(n) * n_components, 0.0f);

    id<MTLBuffer> landmark_data_buffer = [state.device newBufferWithBytes:landmark_values.data()
                                                                length:landmark_values.size() * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_data_buffer = [state.device newBufferWithBytes:query_values.data()
                                                             length:query_values.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:layout_values.data()
                                                        length:layout_values.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [state.device newBufferWithLength:out.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> landmark_index_buffer = [state.device newBufferWithBytes:landmark_indices.begin()
                                                                 length:static_cast<std::size_t>(n_landmarks) * sizeof(int)
                                                                options:MTLResourceStorageModeShared];
    if (landmark_data_buffer == nil || query_data_buffer == nil || layout_buffer == nil ||
        out_buffer == nil || landmark_index_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal fused landmark projection buffers.");
    }

    const std::uint32_t n_landmarks_u = static_cast<std::uint32_t>(n_landmarks);
    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t n_features_u = static_cast<std::uint32_t>(n_features);
    const std::uint32_t k_u = static_cast<std::uint32_t>(k);
    const std::uint32_t n_components_u = static_cast<std::uint32_t>(n_components);

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> project_encoder = [command_buffer computeCommandEncoder];
    [project_encoder setComputePipelineState:state.landmark_project_interpolate_pipeline];
    [project_encoder setBuffer:landmark_data_buffer offset:0 atIndex:0];
    [project_encoder setBuffer:query_data_buffer offset:0 atIndex:1];
    [project_encoder setBuffer:layout_buffer offset:0 atIndex:2];
    [project_encoder setBuffer:out_buffer offset:0 atIndex:3];
    [project_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:4];
    [project_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:5];
    [project_encoder setBytes:&n_features_u length:sizeof(std::uint32_t) atIndex:6];
    [project_encoder setBytes:&k_u length:sizeof(std::uint32_t) atIndex:7];
    [project_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:8];
    dispatch_rows(project_encoder, state.landmark_project_interpolate_pipeline, n);
    [project_encoder endEncoding];

    id<MTLComputeCommandEncoder> overwrite_encoder = [command_buffer computeCommandEncoder];
    [overwrite_encoder setComputePipelineState:state.overwrite_landmarks_pipeline];
    [overwrite_encoder setBuffer:out_buffer offset:0 atIndex:0];
    [overwrite_encoder setBuffer:layout_buffer offset:0 atIndex:1];
    [overwrite_encoder setBuffer:landmark_index_buffer offset:0 atIndex:2];
    [overwrite_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:3];
    [overwrite_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
    [overwrite_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:5];
    dispatch_rows(overwrite_encoder, state.overwrite_landmarks_pipeline, n_landmarks * n_components);
    [overwrite_encoder endEncoding];
    wait_for_command(command_buffer, "fused landmark projection");

    std::memcpy(out.data(), [out_buffer contents], out.size() * sizeof(float));
    [landmark_data_buffer release];
    [query_data_buffer release];
    [layout_buffer release];
    [out_buffer release];
    [landmark_index_buffer release];
    return float_to_numeric_matrix(out, n, n_components);
  }
}

List landmark_project_interpolate_knn_confidence_metal_impl(NumericMatrix landmark_data,
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
  if (k > kMaxMetalProjectionNeighbors) {
    Rcpp::stop("Metal fused landmark projection currently supports at most %d neighbors.", kMaxMetalProjectionNeighbors);
  }
  for (int i = 0; i < landmark_indices.size(); ++i) {
    if (landmark_indices[i] < 1 || landmark_indices[i] > n) {
      Rcpp::stop("landmark indices out of range");
    }
  }

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> landmark_values = numeric_matrix_to_row_major_float(landmark_data);
    std::vector<float> query_values = numeric_matrix_to_row_major_float(query_data);
    std::vector<float> layout_values = numeric_matrix_to_float(landmark_layout);
    std::vector<float> out(static_cast<std::size_t>(n) * n_components, 0.0f);
    std::vector<int> projection_indices(static_cast<std::size_t>(n) * k, 1);
    std::vector<float> projection_distances(static_cast<std::size_t>(n) * k, 0.0f);
    std::vector<float> confidence(static_cast<std::size_t>(n), 0.0f);

    id<MTLBuffer> landmark_data_buffer = [state.device newBufferWithBytes:landmark_values.data()
                                                              length:landmark_values.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_data_buffer = [state.device newBufferWithBytes:query_values.data()
                                                           length:query_values.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:layout_values.data()
                                                      length:layout_values.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [state.device newBufferWithLength:out.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> projection_index_buffer =
      [state.device newBufferWithLength:projection_indices.size() * sizeof(int)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> projection_distance_buffer =
      [state.device newBufferWithLength:projection_distances.size() * sizeof(float)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> confidence_buffer =
      [state.device newBufferWithLength:confidence.size() * sizeof(float)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> landmark_index_buffer = [state.device newBufferWithBytes:landmark_indices.begin()
                                                               length:static_cast<std::size_t>(n_landmarks) * sizeof(int)
                                                              options:MTLResourceStorageModeShared];
    if (landmark_data_buffer == nil || query_data_buffer == nil || layout_buffer == nil ||
        out_buffer == nil || projection_index_buffer == nil || projection_distance_buffer == nil ||
        confidence_buffer == nil || landmark_index_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal fused landmark projection/confidence buffers.");
    }

    const std::uint32_t n_landmarks_u = static_cast<std::uint32_t>(n_landmarks);
    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t n_features_u = static_cast<std::uint32_t>(n_features);
    const std::uint32_t k_u = static_cast<std::uint32_t>(k);
    const std::uint32_t n_components_u = static_cast<std::uint32_t>(n_components);

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> project_encoder = [command_buffer computeCommandEncoder];
    [project_encoder setComputePipelineState:state.landmark_project_interpolate_knn_confidence_pipeline];
    [project_encoder setBuffer:landmark_data_buffer offset:0 atIndex:0];
    [project_encoder setBuffer:query_data_buffer offset:0 atIndex:1];
    [project_encoder setBuffer:layout_buffer offset:0 atIndex:2];
    [project_encoder setBuffer:out_buffer offset:0 atIndex:3];
    [project_encoder setBuffer:projection_index_buffer offset:0 atIndex:4];
    [project_encoder setBuffer:projection_distance_buffer offset:0 atIndex:5];
    [project_encoder setBuffer:confidence_buffer offset:0 atIndex:6];
    [project_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:7];
    [project_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:8];
    [project_encoder setBytes:&n_features_u length:sizeof(std::uint32_t) atIndex:9];
    [project_encoder setBytes:&k_u length:sizeof(std::uint32_t) atIndex:10];
    [project_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:11];
    dispatch_rows(project_encoder, state.landmark_project_interpolate_knn_confidence_pipeline, n);
    [project_encoder endEncoding];

    id<MTLComputeCommandEncoder> overwrite_encoder = [command_buffer computeCommandEncoder];
    [overwrite_encoder setComputePipelineState:state.overwrite_landmarks_pipeline];
    [overwrite_encoder setBuffer:out_buffer offset:0 atIndex:0];
    [overwrite_encoder setBuffer:layout_buffer offset:0 atIndex:1];
    [overwrite_encoder setBuffer:landmark_index_buffer offset:0 atIndex:2];
    [overwrite_encoder setBytes:&n_landmarks_u length:sizeof(std::uint32_t) atIndex:3];
    [overwrite_encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
    [overwrite_encoder setBytes:&n_components_u length:sizeof(std::uint32_t) atIndex:5];
    dispatch_rows(overwrite_encoder, state.overwrite_landmarks_pipeline, n_landmarks * n_components);
    [overwrite_encoder endEncoding];
    wait_for_command(command_buffer, "fused landmark projection with confidence");

    std::memcpy(out.data(), [out_buffer contents], out.size() * sizeof(float));
    std::memcpy(projection_indices.data(), [projection_index_buffer contents],
                projection_indices.size() * sizeof(int));
    std::memcpy(projection_distances.data(), [projection_distance_buffer contents],
                projection_distances.size() * sizeof(float));
    std::memcpy(confidence.data(), [confidence_buffer contents],
                confidence.size() * sizeof(float));

    [landmark_data_buffer release];
    [query_data_buffer release];
    [layout_buffer release];
    [out_buffer release];
    [projection_index_buffer release];
    [projection_distance_buffer release];
    [confidence_buffer release];
    [landmark_index_buffer release];

    NumericMatrix layout = float_to_numeric_matrix(out, n, n_components);
    IntegerMatrix indices(n, k);
    std::memcpy(indices.begin(), projection_indices.data(), projection_indices.size() * sizeof(int));
    NumericMatrix distances(n, k);
    for (std::size_t i = 0; i < projection_distances.size(); ++i) {
      distances.begin()[i] = static_cast<double>(projection_distances[i]);
    }
    NumericVector confidence_out(n);
    for (int i = 0; i < n; ++i) {
      confidence_out[i] = static_cast<double>(confidence[static_cast<std::size_t>(i)]);
    }

    return List::create(
      Rcpp::Named("layout") = layout,
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("confidence") = confidence_out
    );
  }
}

NumericVector knn_structure_score_metal_impl(NumericMatrix layout,
                                             IntegerMatrix indices,
                                             IntegerVector keep,
                                             int preserve_k,
                                             IntegerVector labels,
                                             int n_label_levels) {
  const int n = layout.nrow();
  const bool compact_indices = indices.nrow() == keep.size();
  if (layout.ncol() != 2) {
    Rcpp::stop("Metal structure scoring currently supports two-dimensional embeddings.");
  }
  if (indices.nrow() != n && !compact_indices) {
    Rcpp::stop("indices row count must match layout row count or keep length");
  }
  if (preserve_k < 1 || preserve_k > indices.ncol()) Rcpp::stop("invalid preserve_k");
  if (preserve_k > kMaxMetalScoreNeighbors) {
    Rcpp::stop("Metal structure scoring currently supports at most %d neighbors.", kMaxMetalScoreNeighbors);
  }
  if (labels.size() != 0 && labels.size() != n) Rcpp::stop("labels length must match layout row count");
  if (keep.size() == 0) return structure_score_na();

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    const int keep_n = keep.size();
    const int index_rows = indices.nrow();
    const int high_rank_limit = indices.ncol();
    std::vector<float> layout_values = numeric_matrix_to_float(layout);
    std::vector<int> labels_values(static_cast<std::size_t>(n), 0);
    if (labels.size() == n && n_label_levels > 0) {
      for (int i = 0; i < n; ++i) labels_values[static_cast<std::size_t>(i)] = labels[i];
    }
    std::vector<float> row_scores(static_cast<std::size_t>(keep_n) * kMetalScoreWidth, 0.0f);

    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:layout_values.data()
                                                            length:layout_values.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [state.device newBufferWithBytes:indices.begin()
                                                           length:static_cast<std::size_t>(index_rows) * high_rank_limit * sizeof(int)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> keep_buffer = [state.device newBufferWithBytes:keep.begin()
                                                         length:static_cast<std::size_t>(keep_n) * sizeof(int)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> label_buffer = [state.device newBufferWithBytes:labels_values.data()
                                                          length:labels_values.size() * sizeof(int)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> row_buffer = [state.device newBufferWithBytes:row_scores.data()
                                                        length:row_scores.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    if (layout_buffer == nil || index_buffer == nil || keep_buffer == nil ||
        label_buffer == nil || row_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal structure scoring buffers.");
    }

    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t index_rows_u = static_cast<std::uint32_t>(index_rows);
    const std::uint32_t high_rank_limit_u = static_cast<std::uint32_t>(high_rank_limit);
    const std::uint32_t preserve_k_u = static_cast<std::uint32_t>(preserve_k);
    const std::uint32_t keep_n_u = static_cast<std::uint32_t>(keep_n);
    const std::uint32_t compact_u = compact_indices ? 1u : 0u;
    const std::uint32_t n_label_levels_u = labels.size() == n ? static_cast<std::uint32_t>(n_label_levels) : 0u;

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.structure_score_pipeline];
    [encoder setBuffer:layout_buffer offset:0 atIndex:0];
    [encoder setBuffer:index_buffer offset:0 atIndex:1];
    [encoder setBuffer:keep_buffer offset:0 atIndex:2];
    [encoder setBuffer:label_buffer offset:0 atIndex:3];
    [encoder setBuffer:row_buffer offset:0 atIndex:4];
    [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:5];
    [encoder setBytes:&index_rows_u length:sizeof(std::uint32_t) atIndex:6];
    [encoder setBytes:&high_rank_limit_u length:sizeof(std::uint32_t) atIndex:7];
    [encoder setBytes:&preserve_k_u length:sizeof(std::uint32_t) atIndex:8];
    [encoder setBytes:&keep_n_u length:sizeof(std::uint32_t) atIndex:9];
    [encoder setBytes:&compact_u length:sizeof(std::uint32_t) atIndex:10];
    [encoder setBytes:&n_label_levels_u length:sizeof(std::uint32_t) atIndex:11];
    dispatch_rows(encoder, state.structure_score_pipeline, keep_n);
    [encoder endEncoding];
    wait_for_command(command_buffer, "structure scoring");

    std::memcpy(row_scores.data(), [row_buffer contents], row_scores.size() * sizeof(float));
    [layout_buffer release];
    [index_buffer release];
    [keep_buffer release];
    [label_buffer release];
    [row_buffer release];

    double totals[kMetalScoreWidth] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int row = 0; row < keep_n; ++row) {
      const std::size_t base = static_cast<std::size_t>(row) * kMetalScoreWidth;
      for (int c = 0; c < kMetalScoreWidth; ++c) {
        totals[c] += static_cast<double>(row_scores[base + static_cast<std::size_t>(c)]);
      }
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
}

double silhouette_score_metal_impl(NumericMatrix layout,
                                   IntegerVector labels,
                                   int n_label_levels) {
  const int n = layout.nrow();
  if (layout.ncol() != 2) {
    Rcpp::stop("Metal silhouette scoring currently supports two-dimensional embeddings.");
  }
  if (labels.size() != n) Rcpp::stop("labels length must match layout row count");
  if (n_label_levels < 2 || n_label_levels > kMaxMetalSilhouetteLabels) {
    Rcpp::stop("Metal silhouette scoring supports between 2 and %d label levels.", kMaxMetalSilhouetteLabels);
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

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    std::vector<float> layout_values = numeric_matrix_to_float(layout);
    std::vector<float> row_scores(static_cast<std::size_t>(n) * 2u, 0.0f);

    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:layout_values.data()
                                                            length:layout_values.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> label_buffer = [state.device newBufferWithBytes:labels.begin()
                                                          length:static_cast<std::size_t>(n) * sizeof(int)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> count_buffer = [state.device newBufferWithBytes:counts.data()
                                                          length:counts.size() * sizeof(int)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> row_buffer = [state.device newBufferWithBytes:row_scores.data()
                                                        length:row_scores.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    if (layout_buffer == nil || label_buffer == nil || count_buffer == nil || row_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal silhouette scoring buffers.");
    }

    const std::uint32_t n_u = static_cast<std::uint32_t>(n);
    const std::uint32_t n_label_levels_u = static_cast<std::uint32_t>(n_label_levels);

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state.silhouette_pipeline];
    [encoder setBuffer:layout_buffer offset:0 atIndex:0];
    [encoder setBuffer:label_buffer offset:0 atIndex:1];
    [encoder setBuffer:count_buffer offset:0 atIndex:2];
    [encoder setBuffer:row_buffer offset:0 atIndex:3];
    [encoder setBytes:&n_u length:sizeof(std::uint32_t) atIndex:4];
    [encoder setBytes:&n_label_levels_u length:sizeof(std::uint32_t) atIndex:5];
    dispatch_rows(encoder, state.silhouette_pipeline, n);
    [encoder endEncoding];
    wait_for_command(command_buffer, "silhouette scoring");

    std::memcpy(row_scores.data(), [row_buffer contents], row_scores.size() * sizeof(float));
    [layout_buffer release];
    [label_buffer release];
    [count_buffer release];
    [row_buffer release];

    double total = 0.0;
    double scored = 0.0;
    for (int i = 0; i < n; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      total += static_cast<double>(row_scores[base]);
      scored += static_cast<double>(row_scores[base + 1u]);
    }
    return scored > 0.0 ? total / scored : NA_REAL;
  }
}

void pack_umap_csr_for_metal(const IntegerVector& offsets,
                             const IntegerVector& csr_neighbors,
                             const NumericVector& csr_weights,
                             const int n,
                             const int n_epochs,
                             const double max_weight_input,
                             std::vector<std::int32_t>& neighbors,
                             std::vector<float>& weights,
                             float& max_weight,
                             int& truncated_edges) {
  if (offsets.size() != n + 1) {
    Rcpp::stop("CSR offsets length must be n + 1.");
  }
  if (csr_neighbors.size() != csr_weights.size()) {
    Rcpp::stop("CSR neighbors and weights must have the same length.");
  }
  if (offsets[0] != 0) {
    Rcpp::stop("CSR offsets must be zero-based.");
  }
  const int nnz = csr_neighbors.size();
  for (int i = 0; i < n; ++i) {
    const int begin = offsets[i];
    const int end = offsets[i + 1];
    if (begin < 0 || end < begin || end > nnz) {
      Rcpp::stop("CSR offsets are not monotone or are out of range.");
    }
  }

  max_weight = std::isfinite(max_weight_input) && max_weight_input > 0.0 ?
    static_cast<float>(max_weight_input) :
    0.0f;
  if (max_weight <= 0.0f) {
    for (int pos = 0; pos < nnz; ++pos) {
      const double w = csr_weights[pos];
      if (std::isfinite(w) && w > 0.0) {
        max_weight = std::max(max_weight, static_cast<float>(w));
      }
    }
  }
  if (max_weight <= 0.0f) Rcpp::stop("The CSR graph has no positive UMAP weights.");

  const float min_sample_weight = n_epochs > 0 ?
    max_weight / static_cast<float>(n_epochs) :
    0.0f;
  std::vector<int> row_counts(static_cast<std::size_t>(n), 0);
  int width = 1;
  int active_edges = 0;
  for (int i = 0; i < n; ++i) {
    int count = 0;
    for (int pos = offsets[i]; pos < offsets[i + 1]; ++pos) {
      const int nb = csr_neighbors[pos];
      const double w = csr_weights[pos];
      if (nb < 0 || nb >= n || nb == i || !std::isfinite(w) || w <= 0.0) continue;
      if (w < min_sample_weight) continue;
      ++count;
    }
    row_counts[static_cast<std::size_t>(i)] = count;
    active_edges += count;
    width = std::max(width, std::min(count, kMaxMetalNeighbors));
  }
  if (active_edges == 0) {
    Rcpp::stop("The CSR graph has no edges sampled by n_epochs.");
  }

  neighbors.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(width), 0);
  weights.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(width), 0.0f);
  truncated_edges = 0;

  std::vector<std::pair<int, float>> row;
  for (int i = 0; i < n; ++i) {
    row.clear();
    const int count = row_counts[static_cast<std::size_t>(i)];
    row.reserve(static_cast<std::size_t>(std::min(count, kMaxMetalNeighbors)));
    for (int pos = offsets[i]; pos < offsets[i + 1]; ++pos) {
      const int nb = csr_neighbors[pos];
      const double wd = csr_weights[pos];
      if (nb < 0 || nb >= n || nb == i || !std::isfinite(wd) || wd <= 0.0) continue;
      if (wd < min_sample_weight) continue;
      row.push_back({nb, static_cast<float>(wd)});
    }
    if (static_cast<int>(row.size()) > kMaxMetalNeighbors) {
      auto row_less = [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
      };
      std::nth_element(
        row.begin(),
        row.begin() + kMaxMetalNeighbors,
        row.end(),
        row_less
      );
      truncated_edges += static_cast<int>(row.size()) - kMaxMetalNeighbors;
      row.resize(kMaxMetalNeighbors);
      std::sort(row.begin(), row.end(), row_less);
    }
    const int row_size = static_cast<int>(row.size());
    for (int j = 0; j < width; ++j) {
      const std::size_t out =
        static_cast<std::size_t>(i) * static_cast<std::size_t>(width) +
        static_cast<std::size_t>(j);
      if (j < row_size) {
        neighbors[out] = row[static_cast<std::size_t>(j)].first;
        weights[out] = row[static_cast<std::size_t>(j)].second;
      } else {
        neighbors[out] = i;
      }
    }
  }
}

NumericMatrix knn_embed_metal_csr_impl(IntegerVector offsets,
                                       IntegerVector csr_neighbors,
                                       NumericVector csr_weights,
                                       NumericMatrix init,
                                       int n_epochs,
                                       int negative_sample_rate,
                                       double learning_rate,
                                       double min_dist,
                                       double max_weight_input,
                                       int seed) {
  const int n = init.nrow();
  if (n < 1 || init.ncol() != 2) {
    Rcpp::stop("Metal CSR embedding currently requires a two-dimensional initialization.");
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();

    std::vector<std::int32_t> neighbors;
    std::vector<float> weights;
    float max_weight = 0.0f;
    int truncated_edges = 0;
    pack_umap_csr_for_metal(
      offsets,
      csr_neighbors,
      csr_weights,
      n,
      n_epochs,
      max_weight_input,
      neighbors,
      weights,
      max_weight,
      truncated_edges
    );

    const int k = static_cast<int>(neighbors.size() / static_cast<std::size_t>(n));
    std::vector<float> current = init_to_float_2d(init);
    const auto ab = find_ab_params(1.0, min_dist);
    std::vector<float> epochs_per_sample(weights.size(), 0.0f);
    for (std::size_t i = 0; i < weights.size(); ++i) {
      const float w = weights[i];
      epochs_per_sample[i] = w > 0.0f ? max_weight / std::max(w, 1.0e-6f) : 0.0f;
    }

    EmbedParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(n_epochs),
      static_cast<std::uint32_t>(negative_sample_rate),
      kObjectiveUmap,
      static_cast<std::uint32_t>(seed),
      static_cast<float>(learning_rate),
      static_cast<float>(ab.first),
      static_cast<float>(ab.second),
      max_weight
    };

    std::vector<std::int32_t> fixed_layout(current.size());
    constexpr float fixed_scale = 65536.0f;
    for (std::size_t i = 0; i < current.size(); ++i) {
      const float value = std::max(-2140000000.0f, std::min(2140000000.0f, current[i] * fixed_scale));
      fixed_layout[i] = static_cast<std::int32_t>(value);
    }

    id<MTLBuffer> fixed_layout_buffer = [state.device newBufferWithBytes:fixed_layout.data()
                                                                  length:fixed_layout.size() * sizeof(std::int32_t)
                                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> neighbors_buffer = [state.device newBufferWithBytes:neighbors.data()
                                                               length:neighbors.size() * sizeof(std::int32_t)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights_buffer = [state.device newBufferWithBytes:weights.data()
                                                             length:weights.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> epochs_buffer = [state.device newBufferWithBytes:epochs_per_sample.data()
                                                            length:epochs_per_sample.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(EmbedParams)
                                                           options:MTLResourceStorageModeShared];
    if (fixed_layout_buffer == nil || neighbors_buffer == nil ||
        weights_buffer == nil || epochs_buffer == nil || params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal CSR embedding buffers.");
    }

    const char* metal_optimizer = "atomic_inplace";
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    const std::uint32_t epochs_per_command = kMetalEmbeddingEpochsPerCommand;

    const NSUInteger embed_threads = bounded_threads(state.embed_atomic_inplace_pipeline);
    const MTLSize embed_threadgroup_size = MTLSizeMake(embed_threads, 1, 1);
    for (std::uint32_t epoch0 = 0; epoch0 < static_cast<std::uint32_t>(n_epochs); epoch0 += epochs_per_command) {
      id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
      const std::uint32_t epoch_end = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(n_epochs),
        epoch0 + epochs_per_command
      );
      for (std::uint32_t epoch = epoch0; epoch < epoch_end; ++epoch) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state.embed_atomic_inplace_pipeline];
        [encoder setBuffer:fixed_layout_buffer offset:0 atIndex:0];
        [encoder setBuffer:neighbors_buffer offset:0 atIndex:1];
        [encoder setBuffer:weights_buffer offset:0 atIndex:2];
        [encoder setBuffer:epochs_buffer offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        [encoder setBytes:&epoch length:sizeof(std::uint32_t) atIndex:5];
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:embed_threadgroup_size];
        [encoder endEncoding];
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal CSR atomic in-place embedding command failed: %s", ns_error_message(command_buffer.error).c_str());
      }
    }

    std::memcpy(fixed_layout.data(), [fixed_layout_buffer contents], fixed_layout.size() * sizeof(std::int32_t));
    constexpr float inv_fixed_scale = 1.0f / 65536.0f;
    for (std::size_t i = 0; i < current.size(); ++i) {
      current[i] = static_cast<float>(fixed_layout[i]) * inv_fixed_scale;
    }

    NumericMatrix out(n, 2);
    for (int i = 0; i < n; ++i) {
      out(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
      out(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
    }
    out.attr("metal_epoch_schedule") = "precomputed_epochs_per_sample";
    out.attr("metal_optimizer") = metal_optimizer;
    out.attr("metal_graph_input") = "cpu_csr_fuzzy_graph";
    out.attr("metal_csr_width") = k;
    out.attr("metal_truncated_edges") = truncated_edges;
    [fixed_layout_buffer release];
    [neighbors_buffer release];
    [weights_buffer release];
    [epochs_buffer release];
    [params_buffer release];
    return out;
  }
}

List metal_fft512_stockham_diagnostic_impl(int seed,
                                           bool inverse,
                                           int n_checks) {
  constexpr std::uint32_t fft_size = 512u;
  constexpr std::uint32_t log_fft = 9u;
  constexpr std::uint32_t total = fft_size * fft_size;
  const std::size_t floats = static_cast<std::size_t>(total) * 2u;
  const std::size_t bytes = floats * sizeof(float);
  n_checks = std::max(1, std::min<int>(n_checks, static_cast<int>(total)));

  std::vector<float> input(floats);
  std::mt19937 rng(static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : input) value = dist(rng);
  std::vector<float> generic(input);
  std::vector<float> stockham(input);
  std::vector<float> scratch(floats, 0.0f);
  std::vector<float> twiddles = make_fft_twiddles(fft_size, log_fft);
  double generic_time_sec = NA_REAL;
  double stockham_time_sec = NA_REAL;

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();
    id<MTLBuffer> generic_buffer = [state.device newBufferWithBytes:generic.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> stockham_buffer = [state.device newBufferWithBytes:stockham.data()
                                                              length:bytes
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> generic_scratch_buffer = [state.device newBufferWithBytes:scratch.data()
                                                                      length:bytes
                                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> stockham_scratch_buffer = [state.device newBufferWithBytes:scratch.data()
                                                                       length:bytes
                                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> twiddle_buffer = [state.device newBufferWithBytes:twiddles.data()
                                                            length:twiddles.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    if (generic_buffer == nil || stockham_buffer == nil ||
        generic_scratch_buffer == nil || stockham_scratch_buffer == nil ||
        twiddle_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal FFT diagnostic buffers.");
    }

    auto generic_start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    encode_fft_2d_metal_generic(
      state,
      command_buffer,
      generic_buffer,
      generic_scratch_buffer,
      twiddle_buffer,
      fft_size,
      log_fft,
      inverse
    );
    wait_for_command(command_buffer, "FFT512 generic diagnostic");
    auto generic_end = std::chrono::steady_clock::now();

    auto stockham_start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> stockham_command_buffer = [state.queue commandBuffer];
    encode_fft_512_stockham_metal(
      state,
      stockham_command_buffer,
      stockham_buffer,
      stockham_scratch_buffer,
      inverse
    );
    wait_for_command(stockham_command_buffer, "FFT512 Stockham diagnostic");
    auto stockham_end = std::chrono::steady_clock::now();

    generic_time_sec =
      std::chrono::duration<double>(generic_end - generic_start).count();
    stockham_time_sec =
      std::chrono::duration<double>(stockham_end - stockham_start).count();

    std::memcpy(generic.data(), [generic_buffer contents], bytes);
    std::memcpy(stockham.data(), [stockham_buffer contents], bytes);
    [generic_buffer release];
    [stockham_buffer release];
    [generic_scratch_buffer release];
    [stockham_scratch_buffer release];
    [twiddle_buffer release];
  }

  double max_abs = 0.0;
  double sum_sq = 0.0;
  double ref_sq = 0.0;
  int max_index = 0;
  for (std::uint32_t i = 0; i < total; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const double gr = generic[base];
    const double gi = generic[base + 1u];
    const double dr = static_cast<double>(stockham[base]) - gr;
    const double di = static_cast<double>(stockham[base + 1u]) - gi;
    const double err = std::sqrt(dr * dr + di * di);
    sum_sq += err * err;
    ref_sq += gr * gr + gi * gi;
    if (err > max_abs) {
      max_abs = err;
      max_index = static_cast<int>(i);
    }
  }
  const double rms_abs = std::sqrt(sum_sq / static_cast<double>(total));
  const double rms_rel = std::sqrt(sum_sq / std::max(ref_sq, std::numeric_limits<double>::min()));

  const int check_count = std::min<int>(n_checks, 16);
  NumericMatrix sample(check_count, 5);
  Rcpp::colnames(sample) = Rcpp::CharacterVector::create(
    "index", "generic_re", "generic_im", "stockham_re", "stockham_im"
  );
  for (int i = 0; i < check_count; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    sample(i, 0) = i;
    sample(i, 1) = generic[base];
    sample(i, 2) = generic[base + 1u];
    sample(i, 3) = stockham[base];
    sample(i, 4) = stockham[base + 1u];
  }

  return List::create(
    Rcpp::Named("fft_size") = static_cast<int>(fft_size),
    Rcpp::Named("inverse") = inverse,
    Rcpp::Named("seed") = seed,
    Rcpp::Named("max_abs_error") = max_abs,
    Rcpp::Named("rms_abs_error") = rms_abs,
    Rcpp::Named("rms_relative_error") = rms_rel,
    Rcpp::Named("max_error_index") = max_index,
    Rcpp::Named("sample") = sample,
    Rcpp::Named("reference") = "generic_metal_cooley_tukey",
    Rcpp::Named("candidate") = "diagnostic_stockham512",
    Rcpp::Named("generic_time_sec") = generic_time_sec,
    Rcpp::Named("stockham_time_sec") = stockham_time_sec
  );
}

List metal_mpsgraph_fft_diagnostic_impl(int fft_size,
                                        int seed,
                                        int n_repeats) {
  if (@available(macOS 14.0, *)) {
    if (fft_size < 16 || fft_size > 2048) {
      Rcpp::stop("MPSGraph FFT diagnostic requires fft_size between 16 and 2048.");
    }
    if ((fft_size & (fft_size - 1)) != 0) {
      Rcpp::stop("MPSGraph FFT diagnostic requires a power-of-two fft_size.");
    }
    n_repeats = std::max(1, std::min(n_repeats, 20));
    const std::uint32_t n = static_cast<std::uint32_t>(fft_size);
    const std::size_t total = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = total * sizeof(float);

    std::vector<float> input(total);
    std::vector<float> output(total, 0.0f);
    std::mt19937 rng(static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : input) value = dist(rng);

    NumericVector run_times(n_repeats);
    @autoreleasepool {
      MetalEmbeddingState& state = metal_embedding_state();
      MPSGraph* graph = [[[MPSGraph alloc] init] autorelease];
      MPSShape* real_shape = @[ @(fft_size), @(fft_size) ];
      MPSGraphTensor* input_tensor = [graph placeholderWithShape:real_shape
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"input"];
      MPSGraphFFTDescriptor* forward_desc = [MPSGraphFFTDescriptor descriptor];
      forward_desc.inverse = NO;
      forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
      MPSGraphTensor* spectrum = [graph realToHermiteanFFTWithTensor:input_tensor
                                                                axes:@[@0, @1]
                                                          descriptor:forward_desc
                                                                name:@"rfft2"];
      MPSGraphFFTDescriptor* inverse_desc = [MPSGraphFFTDescriptor descriptor];
      inverse_desc.inverse = YES;
      inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
      MPSGraphTensor* recovered = [graph HermiteanToRealFFTWithTensor:spectrum
                                                                 axes:@[@0, @1]
                                                           descriptor:inverse_desc
                                                                 name:@"irfft2"];

      id<MTLBuffer> input_buffer = [state.device newBufferWithBytes:input.data()
                                                             length:bytes
                                                            options:MTLResourceStorageModeShared];
      if (input_buffer == nil) Rcpp::stop("Failed to allocate MPSGraph FFT input buffer.");
      MPSGraphTensorData* input_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                shape:real_shape
                                                                             dataType:MPSDataTypeFloat32] autorelease];
      MPSGraphTensorDataDictionary* feeds = @{ input_tensor: input_data };
      for (int r = 0; r < n_repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        MPSGraphTensorDataDictionary* result =
          [graph runWithMTLCommandQueue:state.queue
                                  feeds:feeds
                          targetTensors:@[ recovered ]
                       targetOperations:nil];
        auto end = std::chrono::steady_clock::now();
        run_times[r] = std::chrono::duration<double>(end - start).count();
        if (r == n_repeats - 1) {
          MPSGraphTensorData* recovered_data = [result objectForKey:recovered];
          if (recovered_data == nil) {
            [input_buffer release];
            Rcpp::stop("MPSGraph FFT diagnostic did not return recovered tensor.");
          }
          MPSNDArray* array = [recovered_data mpsndarray];
          if (array == nil) {
            [input_buffer release];
            Rcpp::stop("MPSGraph FFT diagnostic did not expose an MPSNDArray.");
          }
          [array readBytes:output.data() strideBytes:nil];
        }
      }
      [input_buffer release];
    }

    double max_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    int max_index = 0;
    for (std::size_t i = 0; i < total; ++i) {
      const double err = static_cast<double>(output[i]) - static_cast<double>(input[i]);
      const double abs_err = std::abs(err);
      sum_sq += err * err;
      ref_sq += static_cast<double>(input[i]) * static_cast<double>(input[i]);
      if (abs_err > max_abs) {
        max_abs = abs_err;
        max_index = static_cast<int>(i);
      }
    }
    const double rms_abs = std::sqrt(sum_sq / static_cast<double>(total));
    const double rms_rel = std::sqrt(sum_sq / std::max(ref_sq, std::numeric_limits<double>::min()));
    return List::create(
      Rcpp::Named("available") = true,
      Rcpp::Named("fft_size") = fft_size,
      Rcpp::Named("seed") = seed,
      Rcpp::Named("n_repeats") = n_repeats,
      Rcpp::Named("run_times_sec") = run_times,
      Rcpp::Named("median_time_sec") = Rcpp::median(run_times),
      Rcpp::Named("first_time_sec") = run_times[0],
      Rcpp::Named("max_abs_error") = max_abs,
      Rcpp::Named("rms_abs_error") = rms_abs,
      Rcpp::Named("rms_relative_error") = rms_rel,
      Rcpp::Named("max_error_index") = max_index,
      Rcpp::Named("method") = "MPSGraph realToHermiteanFFT + HermiteanToRealFFT roundtrip"
    );
  } else {
    return List::create(
      Rcpp::Named("available") = false,
      Rcpp::Named("status") = "not_supported",
      Rcpp::Named("error_message") = "MPSGraph FFT requires macOS 14.0 or newer."
    );
  }
}

List metal_mpsgraph_convolution_diagnostic_impl(int fft_size,
                                                int seed,
                                                int n_repeats) {
  if (@available(macOS 14.0, *)) {
    if (fft_size < 16 || fft_size > 2048) {
      Rcpp::stop("MPSGraph convolution diagnostic requires fft_size between 16 and 2048.");
    }
    if ((fft_size & (fft_size - 1)) != 0) {
      Rcpp::stop("MPSGraph convolution diagnostic requires a power-of-two fft_size.");
    }
    n_repeats = std::max(1, std::min(n_repeats, 20));
    const std::uint32_t n = static_cast<std::uint32_t>(fft_size);
    const std::uint32_t log_fft = log2_power_of_two(n);
    const std::uint32_t total_u = n * n;
    const std::size_t total = static_cast<std::size_t>(total_u);
    const std::size_t real_bytes = total * sizeof(float);
    const std::size_t complex_floats = total * 2u;
    const std::size_t complex_bytes = complex_floats * sizeof(float);

    std::vector<float> mass(total);
    std::vector<float> kernel(total);
    std::vector<float> current_complex(complex_floats, 0.0f);
    std::vector<float> kernel_complex(complex_floats, 0.0f);
    std::vector<float> current_out_complex(complex_floats, 0.0f);
    std::vector<float> mpsgraph_out(total, 0.0f);
    std::mt19937 rng(static_cast<std::uint32_t>(seed == NA_INTEGER ? 5489 : seed));
    std::uniform_real_distribution<float> mass_dist(0.0f, 1.0f);
    std::normal_distribution<float> kernel_dist(0.0f, 0.25f);
    for (std::size_t i = 0; i < total; ++i) {
      mass[i] = mass_dist(rng);
      kernel[i] = kernel_dist(rng);
      current_complex[i * 2u] = mass[i];
      kernel_complex[i * 2u] = kernel[i];
    }

    double current_time_sec = NA_REAL;
    NumericVector mpsgraph_times(n_repeats);
    @autoreleasepool {
      MetalEmbeddingState& state = metal_embedding_state();
      std::vector<float> twiddles = make_fft_twiddles(n, log_fft);
      std::vector<float> scratch(complex_floats, 0.0f);
      id<MTLBuffer> mass_buffer = [state.device newBufferWithBytes:current_complex.data()
                                                            length:complex_bytes
                                                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> kernel_buffer = [state.device newBufferWithBytes:kernel_complex.data()
                                                              length:complex_bytes
                                                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> out_buffer = [state.device newBufferWithBytes:current_out_complex.data()
                                                           length:complex_bytes
                                                          options:MTLResourceStorageModeShared];
      id<MTLBuffer> scratch_buffer = [state.device newBufferWithBytes:scratch.data()
                                                               length:complex_bytes
                                                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> twiddle_buffer = [state.device newBufferWithBytes:twiddles.data()
                                                              length:twiddles.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
      if (mass_buffer == nil || kernel_buffer == nil || out_buffer == nil ||
          scratch_buffer == nil || twiddle_buffer == nil) {
        Rcpp::stop("Failed to allocate Metal convolution diagnostic buffers.");
      }

      auto current_start = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> current_command = [state.queue commandBuffer];
      encode_fft_2d_metal(state, current_command, mass_buffer, scratch_buffer, twiddle_buffer, n, log_fft, false);
      encode_fft_2d_metal(state, current_command, kernel_buffer, scratch_buffer, twiddle_buffer, n, log_fft, false);
      {
        id<MTLComputeCommandEncoder> encoder = [current_command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_multiply_pipeline];
        [encoder setBuffer:mass_buffer offset:0 atIndex:0];
        [encoder setBuffer:kernel_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBytes:&total_u length:sizeof(std::uint32_t) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_u), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(bounded_threads(state.opentsne_fft_multiply_pipeline), 1, 1)];
        [encoder endEncoding];
      }
      encode_fft_2d_metal(state, current_command, out_buffer, scratch_buffer, twiddle_buffer, n, log_fft, true);
      wait_for_command(current_command, "Metal FFT convolution diagnostic");
      auto current_end = std::chrono::steady_clock::now();
      current_time_sec = std::chrono::duration<double>(current_end - current_start).count();
      std::memcpy(current_out_complex.data(), [out_buffer contents], complex_bytes);

      MPSGraph* graph = [[[MPSGraph alloc] init] autorelease];
      MPSShape* real_shape = @[ @(fft_size), @(fft_size) ];
      MPSGraphTensor* mass_tensor = [graph placeholderWithShape:real_shape
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"mass"];
      MPSGraphTensor* kernel_tensor = [graph placeholderWithShape:real_shape
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"kernel"];
      MPSGraphFFTDescriptor* forward_desc = [MPSGraphFFTDescriptor descriptor];
      forward_desc.inverse = NO;
      forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
      MPSGraphTensor* mass_spectrum = [graph realToHermiteanFFTWithTensor:mass_tensor
                                                                     axes:@[@0, @1]
                                                               descriptor:forward_desc
                                                                     name:@"mass_rfft2"];
      MPSGraphTensor* kernel_spectrum = [graph realToHermiteanFFTWithTensor:kernel_tensor
                                                                       axes:@[@0, @1]
                                                                 descriptor:forward_desc
                                                                       name:@"kernel_rfft2"];
      MPSGraphTensor* product = [graph multiplicationWithPrimaryTensor:mass_spectrum
                                                       secondaryTensor:kernel_spectrum
                                                                  name:@"spectral_product"];
      MPSGraphFFTDescriptor* inverse_desc = [MPSGraphFFTDescriptor descriptor];
      inverse_desc.inverse = YES;
      inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
      MPSGraphTensor* convolved = [graph HermiteanToRealFFTWithTensor:product
                                                                 axes:@[@0, @1]
                                                           descriptor:inverse_desc
                                                                 name:@"irfft2_convolution"];

      id<MTLBuffer> mass_real_buffer = [state.device newBufferWithBytes:mass.data()
                                                                 length:real_bytes
                                                                options:MTLResourceStorageModeShared];
      id<MTLBuffer> kernel_real_buffer = [state.device newBufferWithBytes:kernel.data()
                                                                   length:real_bytes
                                                                  options:MTLResourceStorageModeShared];
      if (mass_real_buffer == nil || kernel_real_buffer == nil) {
        Rcpp::stop("Failed to allocate MPSGraph convolution diagnostic input buffers.");
      }
      MPSGraphTensorData* mass_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:mass_real_buffer
                                                                               shape:real_shape
                                                                            dataType:MPSDataTypeFloat32] autorelease];
      MPSGraphTensorData* kernel_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:kernel_real_buffer
                                                                                 shape:real_shape
                                                                              dataType:MPSDataTypeFloat32] autorelease];
      MPSGraphTensorDataDictionary* feeds = @{ mass_tensor: mass_data, kernel_tensor: kernel_data };
      for (int r = 0; r < n_repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        MPSGraphTensorDataDictionary* result =
          [graph runWithMTLCommandQueue:state.queue
                                  feeds:feeds
                          targetTensors:@[ convolved ]
                       targetOperations:nil];
        auto end = std::chrono::steady_clock::now();
        mpsgraph_times[r] = std::chrono::duration<double>(end - start).count();
        if (r == n_repeats - 1) {
          MPSGraphTensorData* convolved_data = [result objectForKey:convolved];
          if (convolved_data == nil) {
            Rcpp::stop("MPSGraph convolution diagnostic did not return output tensor.");
          }
          MPSNDArray* array = [convolved_data mpsndarray];
          if (array == nil) {
            Rcpp::stop("MPSGraph convolution diagnostic did not expose an MPSNDArray.");
          }
          [array readBytes:mpsgraph_out.data() strideBytes:nil];
        }
      }

      [mass_buffer release];
      [kernel_buffer release];
      [out_buffer release];
      [scratch_buffer release];
      [twiddle_buffer release];
      [mass_real_buffer release];
      [kernel_real_buffer release];
    }

    double max_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    int max_index = 0;
    for (std::size_t i = 0; i < total; ++i) {
      const double current_value = current_out_complex[i * 2u];
      const double mps_value = mpsgraph_out[i];
      const double err = mps_value - current_value;
      const double abs_err = std::abs(err);
      sum_sq += err * err;
      ref_sq += current_value * current_value;
      if (abs_err > max_abs) {
        max_abs = abs_err;
        max_index = static_cast<int>(i);
      }
    }
    const double rms_abs = std::sqrt(sum_sq / static_cast<double>(total));
    const double rms_rel = std::sqrt(sum_sq / std::max(ref_sq, std::numeric_limits<double>::min()));

    return List::create(
      Rcpp::Named("available") = true,
      Rcpp::Named("fft_size") = fft_size,
      Rcpp::Named("seed") = seed,
      Rcpp::Named("n_repeats") = n_repeats,
      Rcpp::Named("current_metal_time_sec") = current_time_sec,
      Rcpp::Named("mpsgraph_run_times_sec") = mpsgraph_times,
      Rcpp::Named("mpsgraph_median_time_sec") = Rcpp::median(mpsgraph_times),
      Rcpp::Named("mpsgraph_first_time_sec") = mpsgraph_times[0],
      Rcpp::Named("max_abs_error") = max_abs,
      Rcpp::Named("rms_abs_error") = rms_abs,
      Rcpp::Named("rms_relative_error") = rms_rel,
      Rcpp::Named("max_error_index") = max_index,
      Rcpp::Named("reference") = "current_metal_complex_fft_convolution",
      Rcpp::Named("candidate") = "mpsgraph_real_fft_convolution"
    );
  } else {
    return List::create(
      Rcpp::Named("available") = false,
      Rcpp::Named("status") = "not_supported",
      Rcpp::Named("error_message") = "MPSGraph FFT requires macOS 14.0 or newer."
    );
  }
}

NumericMatrix knn_umap_refine_rows_metal_impl(IntegerMatrix indices,
                                              NumericMatrix distances,
                                              IntegerVector row_ids,
                                              NumericMatrix init_embedding,
                                              int n_epochs,
                                              double min_dist,
                                              int negative_sample_rate,
                                              double learning_rate,
                                              double repulsion_strength,
                                              int seed) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions");
  }
  if (row_ids.size() != indices.nrow()) {
    Rcpp::stop("row_ids length must match the number of KNN rows");
  }
  if (init_embedding.ncol() != 2) {
    Rcpp::stop("Metal UMAP landmark refinement currently requires a two-dimensional layout.");
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (repulsion_strength <= 0.0) Rcpp::stop("repulsion_strength must be positive");

  const int n = init_embedding.nrow();
  const int m = indices.nrow();
  const int k = indices.ncol();
  if (n < 2) Rcpp::stop("init_embedding must have at least two rows");
  if (m < 1) return Rcpp::clone(init_embedding);
  if (k < 1) Rcpp::stop("indices must have at least one neighbor column");
  if (k > kMaxMetalNeighbors) {
    Rcpp::stop("Metal UMAP landmark refinement currently supports at most %d neighbors.", kMaxMetalNeighbors);
  }

  std::vector<std::int32_t> rows(static_cast<std::size_t>(m));
  std::vector<std::uint8_t> update_mask(static_cast<std::size_t>(n), 0u);
  for (int i = 0; i < m; ++i) {
    const int row = row_ids[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("row_ids must contain 1-based row indices");
    rows[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(row);
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

  std::vector<std::int32_t> neighbors(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
  std::vector<float> distance_values(neighbors.size(), std::numeric_limits<float>::infinity());
  long double distance_sum = 0.0L;
  std::size_t distance_count = 0u;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      const std::size_t pos =
        static_cast<std::size_t>(i) * static_cast<std::size_t>(k) +
        static_cast<std::size_t>(j);
      const int nb = indices(i, j) - index_offset;
      neighbors[pos] = (nb >= 0 && nb < n) ? static_cast<std::int32_t>(nb) : -1;
      const double d = distances(i, j);
      if (std::isfinite(d)) {
        distance_values[pos] = static_cast<float>(d);
        if (d >= 0.0) {
          distance_sum += static_cast<long double>(d);
          ++distance_count;
        }
      }
    }
  }
  const float global_mean = distance_count > 0u ?
    static_cast<float>(distance_sum / static_cast<long double>(distance_count)) :
    1.0f;

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();

    std::vector<float> current = init_to_float_2d(init_embedding);
    std::vector<std::int32_t> fixed_layout(current.size());
    constexpr float fixed_scale = 65536.0f;
    for (std::size_t i = 0; i < current.size(); ++i) {
      const float value = std::max(-2140000000.0f, std::min(2140000000.0f, current[i] * fixed_scale));
      fixed_layout[i] = static_cast<std::int32_t>(value);
    }

    std::vector<float> weights(neighbors.size(), 0.0f);
    std::vector<float> epochs_per_sample(neighbors.size(), 0.0f);
    const auto ab = find_ab_params(1.0, min_dist);
    EmbedParams embed_params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(n_epochs),
      static_cast<std::uint32_t>(negative_sample_rate),
      kObjectiveUmap,
      static_cast<std::uint32_t>(seed),
      static_cast<float>(learning_rate),
      static_cast<float>(ab.first),
      static_cast<float>(ab.second),
      1.0f
    };
    RefinePrepareParams prepare_params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(m),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(n_epochs),
      global_mean
    };
    const std::uint32_t n_rows_u = static_cast<std::uint32_t>(m);

    id<MTLBuffer> layout_buffer = [state.device newBufferWithBytes:fixed_layout.data()
                                                            length:fixed_layout.size() * sizeof(std::int32_t)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> row_buffer = [state.device newBufferWithBytes:rows.data()
                                                         length:rows.size() * sizeof(std::int32_t)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> neighbor_buffer = [state.device newBufferWithBytes:neighbors.data()
                                                              length:neighbors.size() * sizeof(std::int32_t)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> distance_buffer = [state.device newBufferWithBytes:distance_values.data()
                                                              length:distance_values.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer = [state.device newBufferWithBytes:weights.data()
                                                            length:weights.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> epoch_buffer = [state.device newBufferWithBytes:epochs_per_sample.data()
                                                           length:epochs_per_sample.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> mask_buffer = [state.device newBufferWithBytes:update_mask.data()
                                                          length:update_mask.size() * sizeof(std::uint8_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> embed_params_buffer = [state.device newBufferWithBytes:&embed_params
                                                                  length:sizeof(EmbedParams)
                                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> prepare_params_buffer = [state.device newBufferWithBytes:&prepare_params
                                                                    length:sizeof(RefinePrepareParams)
                                                                   options:MTLResourceStorageModeShared];
    if (layout_buffer == nil || row_buffer == nil || neighbor_buffer == nil ||
        distance_buffer == nil || weight_buffer == nil || epoch_buffer == nil ||
        mask_buffer == nil || embed_params_buffer == nil || prepare_params_buffer == nil) {
      Rcpp::stop("Failed to allocate Metal UMAP landmark refinement buffers.");
    }

    {
      id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:state.refine_prepare_pipeline];
      [encoder setBuffer:row_buffer offset:0 atIndex:0];
      [encoder setBuffer:neighbor_buffer offset:0 atIndex:1];
      [encoder setBuffer:distance_buffer offset:0 atIndex:2];
      [encoder setBuffer:weight_buffer offset:0 atIndex:3];
      [encoder setBuffer:epoch_buffer offset:0 atIndex:4];
      [encoder setBuffer:prepare_params_buffer offset:0 atIndex:5];
      dispatch_rows(encoder, state.refine_prepare_pipeline, m);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal UMAP landmark refinement preparation");
    }

    const NSUInteger refine_threads = bounded_threads(state.refine_rows_pipeline);
    const MTLSize refine_grid = MTLSizeMake(static_cast<NSUInteger>(m), 1, 1);
    const MTLSize refine_threadgroup = MTLSizeMake(refine_threads, 1, 1);
    const std::uint32_t epochs_per_command = kMetalEmbeddingEpochsPerCommand;
    for (std::uint32_t epoch0 = 0; epoch0 < static_cast<std::uint32_t>(n_epochs); epoch0 += epochs_per_command) {
      id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
      const std::uint32_t epoch_end = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(n_epochs),
        epoch0 + epochs_per_command
      );
      for (std::uint32_t epoch = epoch0; epoch < epoch_end; ++epoch) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state.refine_rows_pipeline];
        [encoder setBuffer:layout_buffer offset:0 atIndex:0];
        [encoder setBuffer:row_buffer offset:0 atIndex:1];
        [encoder setBuffer:neighbor_buffer offset:0 atIndex:2];
        [encoder setBuffer:weight_buffer offset:0 atIndex:3];
        [encoder setBuffer:epoch_buffer offset:0 atIndex:4];
        [encoder setBuffer:embed_params_buffer offset:0 atIndex:5];
        [encoder setBytes:&n_rows_u length:sizeof(std::uint32_t) atIndex:6];
        [encoder setBytes:&epoch length:sizeof(std::uint32_t) atIndex:7];
        [encoder setBuffer:mask_buffer offset:0 atIndex:8];
        [encoder dispatchThreads:refine_grid threadsPerThreadgroup:refine_threadgroup];
        [encoder endEncoding];
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        Rcpp::stop(
          "Metal UMAP landmark refinement command failed: %s",
          ns_error_message(command_buffer.error).c_str()
        );
      }
    }

    std::memcpy(fixed_layout.data(), [layout_buffer contents], fixed_layout.size() * sizeof(std::int32_t));
    constexpr float inv_fixed_scale = 1.0f / 65536.0f;
    NumericMatrix out(n, 2);
    for (int i = 0; i < n; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      out(i, 0) = static_cast<double>(fixed_layout[base]) * inv_fixed_scale;
      out(i, 1) = static_cast<double>(fixed_layout[base + 1u]) * inv_fixed_scale;
    }
    out.attr("metal_refinement") = "landmark_rows_atomic_inplace";
    out.attr("metal_refinement_weight_prep") = "smooth_knn_dist_rows";
    out.attr("metal_refinement_rows") = m;
    out.attr("metal_refinement_neighbors") = k;

    [layout_buffer release];
    [row_buffer release];
    [neighbor_buffer release];
    [distance_buffer release];
    [weight_buffer release];
    [epoch_buffer release];
    [mask_buffer release];
    [embed_params_buffer release];
    [prepare_params_buffer release];
    return out;
  }
}

NumericMatrix knn_embed_metal_impl(IntegerMatrix indices,
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
    Rcpp::stop("Metal embedding currently requires a two-dimensional initialization.");
  }
  if (indices.ncol() > kMaxMetalNeighbors) {
    Rcpp::stop("Metal embedding backend currently supports at most %d neighbors.", kMaxMetalNeighbors);
  }
  if (n_epochs < 1) Rcpp::stop("n_epochs must be positive");
  if (negative_sample_rate < 0) Rcpp::stop("negative_sample_rate must be non-negative");
  if (learning_rate <= 0.0) Rcpp::stop("learning_rate must be positive");
  if (min_dist < 0.0) Rcpp::stop("min_dist must be non-negative");

  @autoreleasepool {
    MetalEmbeddingState& state = metal_embedding_state();

    const int n = indices.nrow();
    const std::uint32_t objective_code = objective_id(objective);
    if (objective_code == kObjectiveUmap) {
      Rcpp::stop("Internal Metal UMAP uses the CSR atomic-inplace path; call knn_embed_metal_csr_cpp.");
    }
    std::vector<std::int32_t> neighbors;
    std::vector<float> weights;
    prepare_embedding_neighbors(indices, distances, objective_code, n_epochs, neighbors, weights);
    const int k = static_cast<int>(neighbors.size() / static_cast<std::size_t>(n));
    std::vector<float> current = init_to_float_2d(init);
    std::vector<float> next(current.size());
    const auto ab = find_ab_params(1.0, min_dist);
    const float max_weight = weights.empty() ? 1.0f :
      std::max(*std::max_element(weights.begin(), weights.end()), 1.0e-6f);
    const bool use_precomputed_schedule = false;
    std::vector<float> epochs_per_sample;
    if (use_precomputed_schedule) {
      epochs_per_sample.resize(weights.size(), 0.0f);
      for (std::size_t i = 0; i < weights.size(); ++i) {
        const float w = weights[i];
        epochs_per_sample[i] = w > 0.0f ? max_weight / std::max(w, 1.0e-6f) : 0.0f;
      }
    }

    EmbedParams params{
      static_cast<std::uint32_t>(n),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(n_epochs),
      static_cast<std::uint32_t>(negative_sample_rate),
      objective_code,
      static_cast<std::uint32_t>(seed),
      static_cast<float>(learning_rate),
      static_cast<float>(ab.first),
      static_cast<float>(ab.second),
      max_weight
    };

    id<MTLBuffer> current_buffer = [state.device newBufferWithBytes:current.data()
                                                             length:current.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> next_buffer = [state.device newBufferWithLength:next.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> neighbors_buffer = [state.device newBufferWithBytes:neighbors.data()
                                                               length:neighbors.size() * sizeof(std::int32_t)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights_buffer = [state.device newBufferWithBytes:weights.data()
                                                             length:weights.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> epochs_buffer = nil;
    if (use_precomputed_schedule) {
      epochs_buffer = [state.device newBufferWithBytes:epochs_per_sample.data()
                                                length:epochs_per_sample.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                            length:sizeof(EmbedParams)
                                                           options:MTLResourceStorageModeShared];
    if (current_buffer == nil || next_buffer == nil || neighbors_buffer == nil ||
        weights_buffer == nil || params_buffer == nil ||
        (use_precomputed_schedule && epochs_buffer == nil)) {
      Rcpp::stop("Failed to allocate Metal embedding buffers.");
    }

    id<MTLComputePipelineState> embed_pipeline = state.embed_pipeline;
    const NSUInteger threads_per_group = bounded_threads(embed_pipeline);
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
    const std::uint32_t epochs_per_command = kMetalEmbeddingEpochsPerCommand;

    for (std::uint32_t epoch0 = 0; epoch0 < static_cast<std::uint32_t>(n_epochs); epoch0 += epochs_per_command) {
      id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
      const std::uint32_t epoch_end = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(n_epochs),
        epoch0 + epochs_per_command
      );
      for (std::uint32_t epoch = epoch0; epoch < epoch_end; ++epoch) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:embed_pipeline];
        [encoder setBuffer:current_buffer offset:0 atIndex:0];
        [encoder setBuffer:next_buffer offset:0 atIndex:1];
        [encoder setBuffer:neighbors_buffer offset:0 atIndex:2];
        [encoder setBuffer:weights_buffer offset:0 atIndex:3];
        if (use_precomputed_schedule) {
          [encoder setBuffer:epochs_buffer offset:0 atIndex:4];
          [encoder setBuffer:params_buffer offset:0 atIndex:5];
          [encoder setBytes:&epoch length:sizeof(std::uint32_t) atIndex:6];
        } else {
          [encoder setBuffer:params_buffer offset:0 atIndex:4];
          [encoder setBytes:&epoch length:sizeof(std::uint32_t) atIndex:5];
        }
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        std::swap(current_buffer, next_buffer);
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        Rcpp::stop("Metal embedding command failed: %s", ns_error_message(command_buffer.error).c_str());
      }
    }

    std::memcpy(current.data(), [current_buffer contents], current.size() * sizeof(float));
    NumericMatrix out(n, 2);
    for (int i = 0; i < n; ++i) {
      out(i, 0) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u]);
      out(i, 1) = static_cast<double>(current[static_cast<std::size_t>(i) * 2u + 1u]);
    }
    if (use_precomputed_schedule) {
      out.attr("metal_epoch_schedule") = "precomputed_epochs_per_sample";
    }
    [current_buffer release];
    [next_buffer release];
    [neighbors_buffer release];
    [weights_buffer release];
    if (epochs_buffer != nil) [epochs_buffer release];
    [params_buffer release];
    return out;
  }
}
