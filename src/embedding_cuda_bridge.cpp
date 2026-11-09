#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "massive_umap_cuda.h"
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
List landmark_project_interpolate_knn_confidence_cuda_impl(NumericMatrix landmark_data,
                                                           NumericMatrix query_data,
                                                           NumericMatrix landmark_layout,
                                                           IntegerVector landmark_indices,
                                                           int k);
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

int resident_umap_rank_epoch(
    void* context, const std::vector<int>& offsets,
    const std::vector<int>& neighbors,
    const std::vector<float>& weights, int rank,
    int block_rows, int capacity, int epoch,
    float min_weight, float max_weight) {
  std::vector<int> heads, tails;
  std::vector<float> active_weights, periods;
  heads.reserve(capacity);
  tails.reserve(capacity);
  active_weights.reserve(capacity);
  periods.reserve(capacity);
  int visits = 0;
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
  const int n = static_cast<int>(offsets.size()) - 1;
  const int first = rank * block_rows;
  const int last = std::min(n, first + block_rows);
  for (int row = first; row < last; ++row) {
    for (int edge = offsets[row]; edge < offsets[row + 1]; ++edge) {
      const float weight = weights[edge];
      if (weight < min_weight) continue;
      heads.push_back(row);
      tails.push_back(neighbors[edge]);
      active_weights.push_back(weight);
      periods.push_back(max_weight /
        std::max(weight, 1.0e-6f));
      if (static_cast<int>(heads.size()) == capacity) flush();
    }
  }
  flush();
  return visits;
}

}  // namespace
#endif

// [[Rcpp::export]]
List umap_cuda_optimize_csr_multigpu_cpp(
    IntegerVector offsets, IntegerVector neighbors,
    NumericVector weights, NumericMatrix init,
    int n_epochs, int negative_sample_rate,
    double learning_rate, double min_dist,
    double repulsion_strength, int seed,
    IntegerVector devices) {
#if defined(FASTEMBEDR_HAS_CUDA) && defined(FASTEMBEDR_HAS_NCCL)
  const int n = init.nrow();
  const int dimensions = init.ncol();
  const int nnz = neighbors.size();
  if (n < 2 || (dimensions != 2 && dimensions != 3) ||
      offsets.size() != n + 1 || weights.size() != nnz ||
      nnz < 1 || devices.size() < 2 || devices.size() > n ||
      n_epochs < 1 || negative_sample_rate < 0 ||
      !std::isfinite(learning_rate) || learning_rate <= 0 ||
      !std::isfinite(min_dist) || min_dist < 0 ||
      !std::isfinite(repulsion_strength) ||
      repulsion_strength <= 0 || seed < 0) {
    Rcpp::stop("Invalid resident multi-GPU UMAP request.");
  }
  if (offsets[0] != 0 || offsets[n] != nnz) {
    Rcpp::stop("Resident UMAP CSR offsets are inconsistent.");
  }
  float max_weight = 0.0f;
  std::vector<int> row_ptr(offsets.begin(), offsets.end());
  std::vector<int> col(neighbors.begin(), neighbors.end());
  std::vector<float> edge_weights(nnz);
  for (int row = 0; row < n; ++row) {
    if (row_ptr[row] < 0 || row_ptr[row + 1] < row_ptr[row] ||
        row_ptr[row + 1] > nnz) {
      Rcpp::stop("Resident UMAP CSR offsets are not monotone.");
    }
  }
  for (int edge = 0; edge < nnz; ++edge) {
    if (col[edge] < 0 || col[edge] >= n ||
        !std::isfinite(weights[edge]) || weights[edge] < 0) {
      Rcpp::stop("Resident UMAP graph has invalid edges.");
    }
    edge_weights[edge] = static_cast<float>(weights[edge]);
    if (!std::isfinite(edge_weights[edge])) {
      Rcpp::stop("Resident UMAP weights exceed float32 range.");
    }
    max_weight = std::max(max_weight, edge_weights[edge]);
  }
  if (max_weight <= 0) {
    Rcpp::stop("Resident UMAP graph has no positive edges.");
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
  const auto curve = fastembedr_umap_curve(min_dist);
  const int capacity = std::min(1000000, nnz);
  void* raw = fastembedr_massive_umap_multigpu_create(
    n, dimensions, capacity, n_epochs, negative_sample_rate,
    static_cast<float>(learning_rate),
    static_cast<float>(curve.first),
    static_cast<float>(curve.second),
    static_cast<float>(repulsion_strength), max_weight,
    static_cast<unsigned int>(seed), devices.begin(),
    devices.size(), delta_scale);
  if (raw == nullptr) resident_umap_check(1);
  std::unique_ptr<void, decltype(
      &fastembedr_massive_umap_multigpu_destroy)> context(
    raw, fastembedr_massive_umap_multigpu_destroy);
  std::vector<float> layout(
    static_cast<std::size_t>(n) * dimensions);
  for (int row = 0; row < n; ++row) {
    for (int axis = 0; axis < dimensions; ++axis) {
      const double value = init(row, axis);
      if (!std::isfinite(value)) {
        Rcpp::stop("Resident UMAP initialization is non-finite.");
      }
      layout[static_cast<std::size_t>(row) * dimensions +
        axis] = static_cast<float>(value);
    }
  }
  constexpr int transfer_rows = 1000000;
  for (int first = 0; first < n; first += transfer_rows) {
    const int rows = std::min(transfer_rows, n - first);
    resident_umap_check(fastembedr_massive_umap_multigpu_upload(
      context.get(), first, rows, layout.data() +
        static_cast<std::size_t>(first) * dimensions));
  }
  const int block_rows = (n - 1) / devices.size() + 1;
  const float min_weight = max_weight / n_epochs;
  NumericVector visits(devices.size());
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    resident_umap_check(
      fastembedr_massive_umap_multigpu_start_epoch(
        context.get()));
    std::vector<std::future<int>> pending;
    for (int rank = 0; rank < devices.size(); ++rank) {
      pending.emplace_back(std::async(std::launch::async,
        [&, rank, epoch]() {
          return resident_umap_rank_epoch(context.get(),
            row_ptr, col, edge_weights, rank, block_rows,
            capacity, epoch, min_weight, max_weight);
        }));
    }
    for (int rank = 0; rank < devices.size(); ++rank) {
      visits[rank] += pending[rank].get();
    }
    resident_umap_check(
      fastembedr_massive_umap_multigpu_finish_epoch(
        context.get()));
    Rcpp::checkUserInterrupt();
  }
  for (int first = 0; first < n; first += transfer_rows) {
    const int rows = std::min(transfer_rows, n - first);
    resident_umap_check(
      fastembedr_massive_umap_multigpu_download(
        context.get(), first, rows, layout.data() +
          static_cast<std::size_t>(first) * dimensions));
  }
  NumericMatrix result(n, dimensions);
  for (int row = 0; row < n; ++row) {
    for (int axis = 0; axis < dimensions; ++axis) {
      result(row, axis) = layout[
        static_cast<std::size_t>(row) * dimensions + axis];
    }
  }
  return List::create(
    Rcpp::Named("Y") = result,
    Rcpp::Named("optimizer") =
      "cuda_resident_umap_coordinated_nccl",
    Rcpp::Named("gpu_devices") = devices,
    Rcpp::Named("per_device_edge_visits") = visits,
    Rcpp::Named("delta_scale") = delta_scale,
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
List landmark_project_interpolate_knn_confidence_cuda_cpp(NumericMatrix landmark_data,
                                                          NumericMatrix query_data,
                                                          NumericMatrix landmark_layout,
                                                          IntegerVector landmark_indices,
                                                          int k) {
  return landmark_project_interpolate_knn_confidence_cuda_impl(
    landmark_data,
    query_data,
    landmark_layout,
    landmark_indices,
    k
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
