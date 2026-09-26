#include <Rcpp.h>

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
                                       int optimizer_mode);
NumericMatrix knn_umap_cuda_fused_float_impl(IntegerMatrix indices,
                                             SEXP distances,
                                             int n_epochs,
                                             int negative_sample_rate,
                                             double learning_rate,
                                             double min_dist,
                                             double repulsion_strength,
                                             int spectral_n_iter,
                                             int seed,
                                             int optimizer_mode);
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
                                           bool binary_graph);
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
                                       int seed);
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
                                      int optimizer_mode) {
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
    optimizer_mode
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
                                            int optimizer_mode) {
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
    optimizer_mode
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
                                          bool binary_graph) {
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
    binary_graph
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

// [[Rcpp::export]]
NumericMatrix knn_tsne_exact_cuda_cpp(IntegerMatrix indices,
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
  return knn_tsne_exact_cuda_impl(
    indices,
    distances,
    init,
    n_epochs,
    perplexity,
    learning_rate,
    stop_lying_iter,
    mom_switch_iter,
    momentum,
    final_momentum,
    exaggeration_factor,
    seed
  );
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
