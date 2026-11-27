#include <Rcpp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "massive_umap_cuda.h"
#include "csr_edge_tiles.h"
#include "resident_csr_view.h"
#include "umap_optimizer_common.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

bool embedding_cuda_available_impl();
NumericMatrix spectral_knn_init_cuda_impl(IntegerMatrix indices,
                                          NumericMatrix distances,
                                          int n_components,
                                          int spectral_n_iter,
                                          int seed);
NumericMatrix knn_embed_cuda_impl(IntegerMatrix indices,
                                  NumericMatrix distances,
                                  NumericMatrix init,
                                  std::string objective,
                                  int n_epochs,
                                  int negative_sample_rate,
                                  double learning_rate,
                                  double min_dist,
                                  int seed);
NumericMatrix knn_umap_cuda_fused_impl(IntegerMatrix indices,
                                       NumericMatrix distances,
                                       int n_epochs,
                                       int negative_sample_rate,
                                       double learning_rate,
                                       double min_dist,
                                       double repulsion_strength,
                                       int spectral_n_iter,
                                       int seed,
                                       int optimizer_mode,
                                       int n_components,
                                       bool binary_graph);
NumericMatrix knn_umap_cuda_fused_float_impl(IntegerMatrix indices,
                                             SEXP distances,
                                             int n_epochs,
                                             int negative_sample_rate,
                                             double learning_rate,
                                             double min_dist,
                                             double repulsion_strength,
                                             int spectral_n_iter,
                                             int seed,
                                             int optimizer_mode,
                                             int n_components,
                                             bool binary_graph);
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
                                           bool binary_graph,
                                           int n_components);
List umap_cuda_graph_dump_impl(IntegerMatrix indices,
                               NumericMatrix distances);
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
                                          int optimizer_mode);
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
                                 bool record_costs);
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
                                       bool record_costs);
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
                                     bool record_costs);
List standardize_cuda_impl(NumericMatrix data);
NumericMatrix project_embedding_knn_cuda_impl(NumericMatrix reference_layout,
                                              IntegerMatrix projection_indices,
                                              NumericMatrix projection_distances);
SEXP massive_cuda_projector_create_impl(NumericMatrix layout,
                                        int k, int capacity);
NumericMatrix massive_cuda_projector_batch_impl(
    SEXP pointer, IntegerMatrix indices, NumericMatrix distances);
void massive_cuda_projector_release_impl(SEXP pointer);
NumericMatrix interpolate_landmark_layout_cuda_impl(NumericMatrix landmark_layout,
                                                    IntegerVector landmark_indices,
                                                    IntegerMatrix projection_indices,
                                                    NumericMatrix projection_distances,
                                                    int n);
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
                              int seed);
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
                                           double max_extrapolation);
List landmark_tsne_transform_cuda_devices_impl(
    SEXP reference_data, SEXP query_data, SEXP reference_layout,
    IntegerMatrix indices, NumericMatrix distances,
    double perplexity, int n_iter, int early_exaggeration_iter,
    double learning_rate, double early_exaggeration,
    double exaggeration, double initial_momentum,
    double final_momentum, double max_grad_norm,
    double max_step_norm, int n_negatives,
    int exact_repulsion_threshold, int seed, IntegerVector devices);
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
                                                double max_extrapolation);
List landmark_umap_project_refine_cuda_devices_impl(
    SEXP reference_data, SEXP query_data, SEXP reference_layout,
    IntegerMatrix indices, NumericMatrix distances,
    int n_epochs, double min_dist, int negative_sample_rate,
    double learning_rate, double repulsion_strength, int seed,
    IntegerVector devices);
NumericVector knn_structure_score_cuda_impl(NumericMatrix layout,
                                            IntegerMatrix indices,
                                            IntegerVector keep,
                                            int preserve_k,
                                            IntegerVector labels,
                                            int n_label_levels);
double silhouette_score_cuda_impl(NumericMatrix layout,
                                  IntegerVector labels,
                                  int n_label_levels);
NumericMatrix rsvd_multiply_cuda_impl(NumericMatrix left,
                                      NumericMatrix right,
                                      bool transpose_left);
List pca_tsvd_cuda_impl(SEXP data,
                        int n_components,
                        bool center,
                        bool scale,
                        int seed,
                        int requested_method,
                        int oversample,
                        int power);

// [[Rcpp::export]]
List fastembedr_build_config_cpp() {
#ifdef FASTEMBEDR_HAS_CUDA
  const bool cuda_compiled = true;
#else
  const bool cuda_compiled = false;
#endif
#ifdef FASTEMBEDR_HAS_CUVS
  const bool cuvs_compiled = true;
#else
  const bool cuvs_compiled = false;
#endif
#ifdef FASTEMBEDR_HAS_RAFT
  const bool raft_compiled = true;
#else
  const bool raft_compiled = false;
#endif
#ifdef HAVE_METAL
  const bool metal_compiled = true;
#else
  const bool metal_compiled = false;
#endif
#ifdef FASTEMBEDR_DIAGNOSTIC_ONLY
  const bool diagnostic_only = true;
#else
  const bool diagnostic_only = false;
#endif
  return List::create(
    Rcpp::Named("cuda_compiled") = cuda_compiled,
    Rcpp::Named("cuvs_compiled") = cuvs_compiled,
    Rcpp::Named("raft_compiled") = raft_compiled,
    Rcpp::Named("metal_compiled") = metal_compiled,
    Rcpp::Named("diagnostic_only") = diagnostic_only
  );
}

// [[Rcpp::export]]
bool embedding_cuda_available_cpp() {
  return embedding_cuda_available_impl();
}

// [[Rcpp::export]]
NumericMatrix spectral_knn_init_cuda_cpp(IntegerMatrix indices,
                                         NumericMatrix distances,
                                         int n_components,
                                         int spectral_n_iter,
                                         int seed) {
  return spectral_knn_init_cuda_impl(
    indices, distances, n_components, spectral_n_iter, seed
  );
}

// [[Rcpp::export]]
NumericMatrix knn_embed_cuda_cpp(IntegerMatrix indices,
                                 NumericMatrix distances,
                                 NumericMatrix init,
                                 std::string objective,
                                 int n_epochs,
                                 int negative_sample_rate,
                                 double learning_rate,
                                 double min_dist,
                                 int seed) {
  return knn_embed_cuda_impl(
    indices, distances, init, objective, n_epochs,
    negative_sample_rate, learning_rate, min_dist, seed
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_cuda_fused_cpp(IntegerMatrix indices,
                                      NumericMatrix distances,
                                      int n_epochs,
                                      int negative_sample_rate,
                                      double learning_rate,
                                      double min_dist,
                                      double repulsion_strength,
                                      int spectral_n_iter,
                                      int seed,
                                      int optimizer_mode,
                                      int n_components = 2,
                                      bool binary_graph = false) {
  return knn_umap_cuda_fused_impl(
    indices,
    distances,
    n_epochs,
    negative_sample_rate,
    learning_rate,
    min_dist,
    repulsion_strength,
    spectral_n_iter,
    seed,
    optimizer_mode,
    n_components,
    binary_graph
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_cuda_fused_float_cpp(IntegerMatrix indices,
                                            SEXP distances,
                                            int n_epochs,
                                            int negative_sample_rate,
                                            double learning_rate,
                                            double min_dist,
                                            double repulsion_strength,
                                            int spectral_n_iter,
                                            int seed,
                                            int optimizer_mode,
                                            int n_components = 2,
                                            bool binary_graph = false) {
  return knn_umap_cuda_fused_float_impl(
    indices,
    distances,
    n_epochs,
    negative_sample_rate,
    learning_rate,
    min_dist,
    repulsion_strength,
    spectral_n_iter,
    seed,
    optimizer_mode,
    n_components,
    binary_graph
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_cuda_fused_gpu_cpp(SEXP gpu_knn,
                                          int requested_k,
                                          int n_epochs,
                                          int negative_sample_rate,
                                          double learning_rate,
                                          double min_dist,
                                          double repulsion_strength,
                                          int spectral_n_iter,
                                          int seed,
                                          int optimizer_mode,
                                          bool binary_graph,
                                          int n_components = 2) {
  return knn_umap_cuda_fused_gpu_impl(
    gpu_knn,
    requested_k,
    n_epochs,
    negative_sample_rate,
    learning_rate,
    min_dist,
    repulsion_strength,
    spectral_n_iter,
    seed,
    optimizer_mode,
    binary_graph,
    n_components
  );
}

// [[Rcpp::export]]
List umap_cuda_graph_dump_cpp(IntegerMatrix indices,
                              NumericMatrix distances) {
  return umap_cuda_graph_dump_impl(indices, distances);
}

// [[Rcpp::export]]
NumericMatrix umap_cuda_optimize_coo_cpp(IntegerVector heads,
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
  return umap_cuda_optimize_coo_impl(
    heads, tails, weights, epochs_per_sample, init, n_epochs,
    negative_sample_rate, learning_rate, min_dist, repulsion_strength,
    seed, optimizer_mode
  );
}

// [[Rcpp::export]]
NumericMatrix umap_cuda_optimize_csr_cpp(IntegerVector offsets,
                                         IntegerVector neighbors,
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
  if (offsets.size() < 2) Rcpp::stop("CSR offsets must have length at least two");
  const int n = offsets.size() - 1;
  if (init.nrow() != n) Rcpp::stop("init row count must match CSR graph");
  const int nnz = neighbors.size();
  IntegerVector heads(nnz);
  for (int row = 0; row < n; ++row) {
    const int begin = offsets[row];
    const int end = offsets[row + 1];
    if (begin < 0 || end < begin || end > nnz) {
      Rcpp::stop("CSR offsets are not monotone or exceed edge count");
    }
    for (int pos = begin; pos < end; ++pos) {
      heads[pos] = row;
    }
  }
  return umap_cuda_optimize_coo_impl(
    heads, neighbors, weights, epochs_per_sample, init, n_epochs,
    negative_sample_rate, learning_rate, min_dist, repulsion_strength,
    seed, optimizer_mode
  );
}

#if defined(FASTEMBEDR_HAS_CUDA) && defined(FASTEMBEDR_HAS_NCCL)
namespace {

void resident_umap_check(int status) {
  if (status != 0) Rcpp::stop(std::string(
    "Resident multi-GPU UMAP failed: ") +
    fastembedr_cuda_embedding_last_error());
}

std::uint64_t resident_umap_rank_epoch(
    void* context, const ResidentCsrView& graph, int rank,
    int block_rows, int capacity, int epoch,
    float min_weight, float max_weight) {
  std::vector<int> heads, tails;
  std::vector<float> active_weights, periods;
  heads.reserve(capacity);
  tails.reserve(capacity);
  active_weights.reserve(capacity);
  periods.reserve(capacity);
  std::uint64_t visits = 0;
  const auto flush = [&]() {
    if (heads.empty()) return;
    const int status = fastembedr_massive_umap_multigpu_step(
      context, rank, heads.data(), tails.data(),
      active_weights.data(), periods.data(),
      static_cast<int>(heads.size()), epoch);
    if (status != 0) throw std::runtime_error(
      fastembedr_cuda_embedding_last_error());
    visits += static_cast<int>(heads.size());
    heads.clear();
    tails.clear();
    active_weights.clear();
    periods.clear();
  };
  const int n = graph.rows;
  const int first = rank * block_rows;
  const int last = std::min(n, first + block_rows);
  for (int row = first; row < last; ++row) {
    fastembedr_csr_edge_tiles(graph.offset(row),
      graph.offset(row + 1),
      static_cast<std::uint32_t>(capacity),
      [&](std::uint64_t base, std::uint32_t count) {
      for (std::uint32_t local = 0; local < count; ++local) {
        const auto edge = base + local;
        const float weight = graph.weight(edge);
        if (weight < min_weight) continue;
        heads.push_back(row);
        tails.push_back(graph.neighbors[edge]);
        active_weights.push_back(weight);
        periods.push_back(max_weight /
          std::max(weight, 1.0e-6f));
        if (static_cast<int>(heads.size()) == capacity) flush();
      }
    });
  }
  flush();
  return visits;
}

}  // namespace
#endif

// [[Rcpp::export]]
SEXP umap_cuda_init_csr_wide_cpp(
    NumericVector offsets, IntegerVector neighbors, SEXP weights,
    int iterations, int seed, IntegerVector devices, double device_limit) {
#if defined(FASTEMBEDR_HAS_CUDA) && defined(FASTEMBEDR_HAS_NCCL)
  if (offsets.size() < 3 || offsets.size() > INT_MAX - 255 ||
      devices.size() < 1 || devices.size() >= offsets.size() ||
      iterations < 1 || seed < 0 || !(device_limit > 0)) {
    Rcpp::stop("Invalid resident CUDA wide UMAP initialization request.");
  }
  std::vector<int> ids(devices.begin(), devices.end());
  std::sort(ids.begin(), ids.end());
  if (ids.front() < 0 || std::adjacent_find(ids.begin(), ids.end()) !=
      ids.end()) Rcpp::stop("Expected distinct CUDA initialization devices.");
  const int n = static_cast<int>(offsets.size() - 1);
  const auto graph = resident_csr_view(offsets, neighbors, weights, n);
  if (!graph.float_weights) Rcpp::stop("Expected float32 CSR weights.");
  resident_csr_max_weight(graph);
  std::vector<float> layout(static_cast<std::size_t>(n) * 2);
  IntegerVector rows(devices.size());
  NumericVector edges(devices.size());
  NumericVector clocks = NumericVector::create(
    Rcpp::Named("setup_upload") = 0.0,
    Rcpp::Named("diffusion") = 0.0,
    Rcpp::Named("download") = 0.0,
    Rcpp::Named("max_device_workspace_bytes") = 0.0);
  const int status = fastembedr_cuda_umap_wide_init(n, graph.wide_offsets,
    graph.neighbors, graph.float_weights, iterations,
    static_cast<unsigned int>(seed), devices.begin(), devices.size(),
    device_limit, layout.data(), rows.begin(), edges.begin(), clocks.begin());
  if (status == 2) return R_NilValue;
  resident_umap_check(status);
  NumericMatrix result(n, 2);
  for (int row = 0; row < n; ++row) {
    const auto start = static_cast<std::size_t>(row) * 2;
    result(row, 0) = layout[start];
    result(row, 1) = layout[start + 1];
  }
  return List::create(Rcpp::Named("layout") = result,
    Rcpp::Named("device_rows") = rows, Rcpp::Named("device_edges") = edges,
    Rcpp::Named("resources") = clocks);
#else
  Rcpp::stop("Resident CUDA wide UMAP initialization requires CUDA and NCCL.");
#endif
}

// [[Rcpp::export]]
List umap_cuda_optimize_csr_multigpu_cpp(
    SEXP offsets, IntegerVector neighbors,
    SEXP weights, SEXP initial,
    int n_epochs, int negative_sample_rate,
    double learning_rate, double min_dist,
    double repulsion_strength, int seed,
    IntegerVector devices, int diffusion_iterations = 0,
    double diffusion_scale = 0.0, double device_limit = 0.0,
    int knn_col_start = -1, int knn_width = 0) {
#if defined(FASTEMBEDR_HAS_CUDA) && defined(FASTEMBEDR_HAS_NCCL)
  const bool from_knn = knn_col_start >= 0;
  const bool reuse_csr = diffusion_iterations > 0;
  const bool wide = reuse_csr || TYPEOF(offsets) == REALSXP;
  IntegerMatrix indices;
  if (from_knn) indices = Rcpp::as<IntegerMatrix>(neighbors);
  NumericMatrix init;
  if (!reuse_csr) init = Rcpp::as<NumericMatrix>(initial);
  if ((from_knn && !reuse_csr) || (reuse_csr && (!Rf_isNull(initial) ||
      !(device_limit > 0) || (!from_knn &&
      (XLENGTH(offsets) < 3 || XLENGTH(offsets) > INT_MAX - 255))))) {
    Rcpp::stop("Invalid retained CUDA wide UMAP initialization request.");
  }
  const int n = from_knn ? indices.nrow() :
    reuse_csr ? static_cast<int>(XLENGTH(offsets) - 1) : init.nrow();
  const int dimensions = reuse_csr ? 2 : init.ncol();
  const auto graph = from_knn ? ResidentCsrView{} :
    resident_csr_view(offsets, neighbors, weights, n);
  std::uint64_t nnz = graph.edges;
  if (n < 2 || (from_knn && n > INT_MAX - 256) || (dimensions != 2 && dimensions != 3) ||
      (!from_knn && nnz < 1) || devices.size() < 1 || devices.size() > n ||
      n_epochs < 1 || negative_sample_rate < 0 ||
      !std::isfinite(learning_rate) || learning_rate <= 0 ||
      !std::isfinite(min_dist) || min_dist < 0 ||
      !std::isfinite(repulsion_strength) ||
      repulsion_strength <= 0 || seed < 0) {
    Rcpp::stop("Invalid resident multi-GPU UMAP request.");
  }
  float max_weight = from_knn ? 0.0f : resident_csr_max_weight(graph);
  Rcpp::List graph_resources;
  using RetainedGraph = std::unique_ptr<void,
    decltype(&fastembedr_cuda_umap_wide_graph_destroy)>;
  RetainedGraph retained(nullptr, fastembedr_cuda_umap_wide_graph_destroy);
  double initialization_seconds = 0;
  Rcpp::List initialization_resources;
  std::vector<float> layout;
  if (reuse_csr) {
    if (!from_knn && (!graph.wide_offsets || !graph.float_weights)) {
      Rcpp::stop("Retained CUDA UMAP requires wide float32 CSR.");
    }
    std::vector<int> sorted(devices.begin(), devices.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) !=
        sorted.end()) Rcpp::stop("Expected distinct CUDA UMAP devices.");
    if (from_knn) {
      const auto started = std::chrono::steady_clock::now();
      if (!Rf_isS4(weights) || !Rcpp::S4(weights).is("float32")) {
        Rcpp::stop("CUDA fuzzy graph requires float32 KNN distances.");
      }
      const IntegerMatrix bits = Rcpp::S4(weights).slot("Data");
      if (bits.nrow() != n || bits.ncol() != indices.ncol() ||
          knn_width < 1 || knn_col_start >= indices.ncol() ||
          knn_width > indices.ncol() - knn_col_start) {
        Rcpp::stop("Invalid CUDA fuzzy graph KNN shape.");
      }
      const auto first = static_cast<std::size_t>(knn_col_start) * n;
      const auto count = static_cast<std::size_t>(knn_width) * n;
      const auto* ids = indices.begin() + first;
      const auto* values = bits.begin() + first;
      const auto bounds = std::minmax_element(ids, ids + count);
      const int base = *bounds.first >= 1 && *bounds.second <= n ? 1 : 0;
      long double sum = 0;
      std::uint64_t valid = 0;
      for (std::size_t i = 0; i < count; ++i) {
        float distance;
        std::memcpy(&distance, values + i, sizeof(distance));
        if (std::isfinite(distance) && distance >= 0.0f) {
          sum += distance;
          ++valid;
        }
      }
      NumericVector graph_edges(devices.size()), routed(devices.size());
      NumericVector graph_times(3);
      graph_times.names() = Rcpp::CharacterVector::create(
        "source_membership", "peer_route", "owner_sort_and_union");
      void* built = nullptr;
      resident_umap_check(fastembedr_cuda_umap_wide_graph_build(
        ids, values, n, knn_width, base,
        valid ? static_cast<double>(sum / valid) : 1.0,
        devices.begin(), devices.size(), device_limit, &built,
        graph_edges.begin(), routed.begin(), graph_times.begin(), &max_weight));
      retained.reset(built);
      for (double rank_edges : graph_edges) {
        nnz += static_cast<std::uint64_t>(rank_edges);
      }
      graph_resources = List::create(
        Rcpp::Named("edges") = static_cast<double>(nnz),
        Rcpp::Named("max_weight") = max_weight,
        Rcpp::Named("device_edges") = graph_edges,
        Rcpp::Named("cross_owner_records_via_host") = routed,
        Rcpp::Named("host_record_upper_bytes") = 12.0 * count,
        Rcpp::Named("outgoing_records_regenerated") = true,
        Rcpp::Named("stages") = graph_times,
        Rcpp::Named("seconds") = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count());
    }
    IntegerVector init_rows(devices.size());
    NumericVector init_edges(devices.size()), times(4);
    times.names() = Rcpp::CharacterVector::create(
      "setup_upload", "diffusion", "scale_on_device",
      "max_device_workspace_bytes");
    void* raw_graph = retained.get();
    const auto start = std::chrono::steady_clock::now();
    const int status = fastembedr_cuda_umap_wide_init(n, graph.wide_offsets,
      graph.neighbors, graph.float_weights, diffusion_iterations, seed,
      devices.begin(), devices.size(), device_limit, nullptr,
      init_rows.begin(), init_edges.begin(), times.begin(), &raw_graph,
      diffusion_scale);
    if (status == 2) Rcpp::stop("Retained CUDA CSR exceeds the device budget.");
    resident_umap_check(status);
    if (!from_knn) retained.reset(raw_graph);
    initialization_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
    initialization_resources = List::create(
      Rcpp::Named("device_rows") = init_rows,
      Rcpp::Named("device_edges") = init_edges,
      Rcpp::Named("layout_reused_on_device") = true,
      Rcpp::Named("resources") = times);
  } else {
    layout.resize(static_cast<std::size_t>(n) * dimensions);
    for (int row = 0; row < n; ++row) {
      for (int axis = 0; axis < dimensions; ++axis) {
        const double value = init(row, axis);
        if (!std::isfinite(value)) {
          Rcpp::stop("Resident UMAP initialization is non-finite.");
        }
        layout[static_cast<std::size_t>(row) * dimensions + axis] =
          static_cast<float>(value);
      }
    }
  }
  const double contributions = static_cast<double>(nnz) *
    (1.0 + static_cast<double>(n_epochs) *
      negative_sample_rate);
  const double scale_limit =
    (std::ldexp(1.0, 62) / contributions - 0.5) /
    (4.0 * learning_rate);
  if (!std::isfinite(scale_limit) || scale_limit < 256.0) {
    Rcpp::stop("Resident UMAP delta exceeds the precision budget.");
  }
  const double delta_scale = std::exp2(std::floor(std::log2(
    std::min(scale_limit, std::ldexp(1.0, 24)))));
  const float min_weight = max_weight / n_epochs;
  NumericVector visits(devices.size());
  if (from_knn) {
    resident_umap_check(fastembedr_cuda_umap_wide_graph_counts(
      retained.get(), min_weight, device_limit, visits.begin()));
    for (auto& visits_rank : visits) visits_rank *= n_epochs;
  }
  const auto optimizer_started = std::chrono::steady_clock::now();
  const auto curve = fastembedr_umap_curve(min_dist);
  const int capacity = reuse_csr ? 1 : static_cast<int>(std::min(
    std::uint64_t{1000000}, nnz));
  void* raw = fastembedr_massive_umap_multigpu_create(
    n, dimensions, capacity, n_epochs, negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(curve.first),
    static_cast<float>(curve.second),
    static_cast<float>(repulsion_strength), max_weight,
    static_cast<unsigned int>(seed), devices.begin(),
    devices.size(), delta_scale, retained.get(), device_limit);
  if (raw == nullptr) resident_umap_check(1);
  std::unique_ptr<void, decltype(
      &fastembedr_massive_umap_multigpu_destroy)> context(
    raw, fastembedr_massive_umap_multigpu_destroy);
  constexpr int transfer_rows = 1000000;
  for (int first = 0; !reuse_csr && first < n; first += transfer_rows) {
    const int rows = std::min(transfer_rows, n - first);
    resident_umap_check(fastembedr_massive_umap_multigpu_upload(
      context.get(), first, rows, layout.data() +
        static_cast<std::size_t>(first) * dimensions));
  }
  const int block_rows = (n - 1) / devices.size() + 1;
  std::vector<unsigned long long> rank_edges(devices.size());
  for (int rank = 0; !from_knn && rank < devices.size(); ++rank) {
    const int first = rank * block_rows;
    const int last = std::min(n, first + block_rows);
    rank_edges[rank] = graph.offset(last) - graph.offset(first);
  }
  const int cache_state = !reuse_csr && n_epochs > 1 ?
    fastembedr_massive_umap_multigpu_cache_begin(
      context.get(), rank_edges.data()) : 0;
  if (cache_state < 0) resident_umap_check(1);
  const bool cached = cache_state == 1;
  std::vector<std::uint64_t> cached_visits(devices.size(), 0);
  if (reuse_csr && !from_knn) {
    for (int rank = 0; rank < devices.size(); ++rank) {
      const int first = rank * block_rows;
      const int last = std::min(n, first + block_rows);
      std::uint64_t count = 0;
      for (auto i = graph.offset(first); i < graph.offset(last); ++i) {
        if (graph.weight(i) >= min_weight) ++count;
      }
      visits[rank] = static_cast<double>(count) * n_epochs;
    }
  }
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    resident_umap_check(
      fastembedr_massive_umap_multigpu_start_epoch(
        context.get()));
    if (reuse_csr) {
      resident_umap_check(fastembedr_cuda_umap_wide_graph_step(
        context.get(), retained.get(), epoch));
    } else if (cached && epoch > 0) {
      for (int rank = 0; rank < devices.size(); ++rank) {
        resident_umap_check(
          fastembedr_massive_umap_multigpu_cache_replay(
            context.get(), rank, epoch));
        visits[rank] += cached_visits[rank];
      }
    } else {
      std::vector<std::future<std::uint64_t>> pending;
      for (int rank = 0; rank < devices.size(); ++rank) {
        pending.emplace_back(std::async(std::launch::async,
          [&, rank, epoch]() {
            return resident_umap_rank_epoch(context.get(),
              graph, rank, block_rows,
              capacity, epoch, min_weight, max_weight);
          }));
      }
      for (int rank = 0; rank < devices.size(); ++rank) {
        const auto count = pending[rank].get();
        visits[rank] += count;
        if (cached) cached_visits[rank] = count;
      }
    }
    resident_umap_check(
      fastembedr_massive_umap_multigpu_finish_epoch(
        context.get()));
    Rcpp::checkUserInterrupt();
  }
  layout.resize(static_cast<std::size_t>(n) * dimensions);
  for (int first = 0; first < n; first += transfer_rows) {
    const int rows = std::min(transfer_rows, n - first);
    resident_umap_check(
      fastembedr_massive_umap_multigpu_download(
        context.get(), first, rows, layout.data() +
          static_cast<std::size_t>(first) * dimensions));
  }
  Rcpp::RObject result;
  if (wide) {
    Rcpp::IntegerMatrix payload(n, dimensions);
    for (int row = 0; row < n; ++row) {
      for (int axis = 0; axis < dimensions; ++axis) {
        const float value = layout[
          static_cast<std::size_t>(row) * dimensions + axis];
        int bits;
        std::memcpy(&bits, &value, sizeof(bits));
        payload(row, axis) = bits;
      }
    }
    Rcpp::S4 packed("float32");
    packed.slot("Data") = payload;
    result = packed;
  } else {
    Rcpp::NumericMatrix dense(n, dimensions);
    for (int row = 0; row < n; ++row) {
      for (int axis = 0; axis < dimensions; ++axis) {
        dense(row, axis) = layout[
          static_cast<std::size_t>(row) * dimensions + axis];
      }
    }
    result = dense;
  }
  return List::create(
    Rcpp::Named("Y") = result,
    Rcpp::Named("optimizer") =
      "cuda_resident_umap_coordinated_nccl",
    Rcpp::Named("gpu_devices") = devices,
    Rcpp::Named("per_device_edge_visits") = visits,
    Rcpp::Named("delta_scale") = delta_scale,
    Rcpp::Named("graph_cache_mode") = reuse_csr ? "retained_device_csr" :
      cached ? "resident_device_shards" : "staged_batches",
    Rcpp::Named("graph_cache_bytes") = reuse_csr ?
      fastembedr_cuda_umap_wide_graph_allocated_bytes(retained.get()) :
      cached ? static_cast<double>(nnz) * 16.0 : 0.0,
    Rcpp::Named("graph_build") = graph_resources,
    Rcpp::Named("initialization_elapsed_sec") = initialization_seconds,
    Rcpp::Named("initialization_resources") = initialization_resources,
    Rcpp::Named("optimization_elapsed_sec") = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - optimizer_started).count(),
    Rcpp::Named("synchronization_seconds") =
      fastembedr_massive_umap_multigpu_collective_seconds(
        context.get()));
#else
  Rcpp::stop("Resident multi-GPU UMAP requires CUDA and NCCL.");
#endif
}

// [[Rcpp::export]]
List knn_tsne_opentsne_cuda_cpp(IntegerMatrix indices,
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
  return knn_tsne_opentsne_cuda_impl(
    indices,
    distances,
    y_init,
    init,
    n_components,
    perplexity,
    early_exaggeration_iter,
    n_iter,
    early_exaggeration,
    exaggeration,
    learning_rate,
    learning_rate_auto,
    initial_momentum,
    final_momentum,
    min_gain,
    max_step_norm,
    negative_gradient_method,
    seed,
    record_costs
  );
}

// [[Rcpp::export]]
List knn_tsne_opentsne_cuda_float_cpp(IntegerMatrix indices,
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
  return knn_tsne_opentsne_cuda_float_impl(
    indices,
    distances,
    y_init,
    init,
    n_components,
    perplexity,
    early_exaggeration_iter,
    n_iter,
    early_exaggeration,
    exaggeration,
    learning_rate,
    learning_rate_auto,
    initial_momentum,
    final_momentum,
    min_gain,
    max_step_norm,
    negative_gradient_method,
    seed,
    record_costs
  );
}

// [[Rcpp::export]]
List knn_tsne_opentsne_cuda_gpu_cpp(SEXP gpu_knn,
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
  return knn_tsne_opentsne_cuda_gpu_impl(
    gpu_knn,
    requested_k,
    y_init,
    init,
    pca_init_data,
    n_components,
    perplexity,
    early_exaggeration_iter,
    n_iter,
    early_exaggeration,
    exaggeration,
    learning_rate,
    learning_rate_auto,
    initial_momentum,
    final_momentum,
    min_gain,
    max_step_norm,
    negative_gradient_method,
    seed,
    record_costs
  );
}

// [[Rcpp::export]]
List standardize_cuda_cpp(NumericMatrix data) {
  return standardize_cuda_impl(data);
}

// [[Rcpp::export]]
NumericMatrix project_embedding_knn_cuda_cpp(NumericMatrix reference_layout,
                                             IntegerMatrix projection_indices,
                                             NumericMatrix projection_distances) {
  return project_embedding_knn_cuda_impl(
    reference_layout, projection_indices, projection_distances
  );
}

// [[Rcpp::export]]
SEXP massive_cuda_projector_create_cpp(NumericMatrix layout,
                                       int k, int capacity) {
  return massive_cuda_projector_create_impl(layout, k, capacity);
}

// [[Rcpp::export]]
NumericMatrix massive_cuda_projector_batch_cpp(
    SEXP pointer, IntegerMatrix indices, NumericMatrix distances) {
  return massive_cuda_projector_batch_impl(pointer, indices, distances);
}

// [[Rcpp::export]]
void massive_cuda_projector_release_cpp(SEXP pointer) {
  massive_cuda_projector_release_impl(pointer);
}

// [[Rcpp::export]]
NumericMatrix interpolate_landmark_layout_cuda_cpp(NumericMatrix landmark_layout,
                                                   IntegerVector landmark_indices,
                                                   IntegerMatrix projection_indices,
                                                   NumericMatrix projection_distances,
                                                   int n) {
  return interpolate_landmark_layout_cuda_impl(
    landmark_layout,
    landmark_indices,
    projection_indices,
    projection_distances,
    n
  );
}

// [[Rcpp::export]]
List transform_tsne_cuda_cpp(NumericMatrix reference_layout,
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
  return transform_tsne_cuda_impl(
    reference_layout, indices, distances, y_init, init, initialization,
    perplexity, n_iter, early_exaggeration_iter, learning_rate,
    early_exaggeration, exaggeration, initial_momentum, final_momentum,
    max_grad_norm, max_step_norm, n_negatives,
    exact_repulsion_threshold, seed
  );
}

// [[Rcpp::export]]
List landmark_tsne_transform_cuda_gpu_cpp(SEXP gpu_knn,
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
  return landmark_tsne_transform_cuda_gpu_impl(
    gpu_knn, reference_data, query_data, reference_layout,
    perplexity, n_iter, early_exaggeration_iter, learning_rate,
    early_exaggeration, exaggeration, initial_momentum, final_momentum,
    max_grad_norm, max_step_norm, n_negatives, exact_repulsion_threshold,
    seed, affine_neighbors, affine_ridge, max_extrapolation
  );
}

// [[Rcpp::export]]
List landmark_tsne_transform_cuda_devices_cpp(
    SEXP reference_data, SEXP query_data, SEXP reference_layout,
    IntegerMatrix indices, NumericMatrix distances,
    double perplexity, int n_iter, int early_exaggeration_iter,
    double learning_rate, double early_exaggeration,
    double exaggeration, double initial_momentum,
    double final_momentum, double max_grad_norm,
    double max_step_norm, int n_negatives,
    int exact_repulsion_threshold, int seed, IntegerVector devices) {
  return landmark_tsne_transform_cuda_devices_impl(
    reference_data, query_data, reference_layout, indices, distances,
    perplexity, n_iter, early_exaggeration_iter, learning_rate,
    early_exaggeration, exaggeration, initial_momentum,
    final_momentum, max_grad_norm, max_step_norm, n_negatives,
    exact_repulsion_threshold, seed, devices);
}

// [[Rcpp::export]]
List landmark_umap_project_refine_cuda_gpu_cpp(SEXP gpu_knn,
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
  return landmark_umap_project_refine_cuda_gpu_impl(
    gpu_knn, reference_data, query_data, reference_layout,
    landmark_rows, query_rows, n_total, n_epochs, min_dist,
    negative_sample_rate, learning_rate, repulsion_strength, seed,
    affine_neighbors, affine_ridge, max_extrapolation
  );
}

// [[Rcpp::export]]
List landmark_umap_project_refine_cuda_devices_cpp(
    SEXP reference_data, SEXP query_data, SEXP reference_layout,
    IntegerMatrix indices, NumericMatrix distances,
    int n_epochs, double min_dist, int negative_sample_rate,
    double learning_rate, double repulsion_strength, int seed,
    IntegerVector devices) {
  return landmark_umap_project_refine_cuda_devices_impl(
    reference_data, query_data, reference_layout, indices, distances,
    n_epochs, min_dist, negative_sample_rate, learning_rate,
    repulsion_strength, seed, devices);
}

// [[Rcpp::export]]
NumericVector knn_structure_score_cuda_cpp(NumericMatrix layout,
                                           IntegerMatrix indices,
                                           IntegerVector keep,
                                           int preserve_k,
                                           IntegerVector labels,
                                           int n_label_levels) {
  return knn_structure_score_cuda_impl(
    layout, indices, keep, preserve_k, labels, n_label_levels
  );
}

// [[Rcpp::export]]
double silhouette_score_cuda_cpp(NumericMatrix layout,
                                 IntegerVector labels,
                                 int n_label_levels) {
  return silhouette_score_cuda_impl(layout, labels, n_label_levels);
}

// [[Rcpp::export]]
NumericMatrix rsvd_multiply_cuda_cpp(NumericMatrix left,
                                     NumericMatrix right,
                                     bool transpose_left) {
  return rsvd_multiply_cuda_impl(left, right, transpose_left);
}

// [[Rcpp::export]]
List pca_tsvd_cuda_cpp(SEXP data,
                       int n_components,
                       bool center,
                       bool scale,
                       int seed,
                       int requested_method,
                       int oversample,
                       int power) {
  return pca_tsvd_cuda_impl(
    data, n_components, center, scale, seed,
    requested_method, oversample, power
  );
}
