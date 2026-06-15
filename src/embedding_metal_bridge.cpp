#include <Rcpp.h>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

bool embedding_metal_available_impl();
NumericMatrix spectral_knn_init_metal_impl(IntegerMatrix indices,
                                           NumericMatrix distances,
                                           int n_components,
                                           int spectral_n_iter,
                                           int seed);
List standardize_metal_impl(NumericMatrix data);
NumericMatrix project_embedding_knn_metal_impl(NumericMatrix reference_layout,
                                               IntegerMatrix projection_indices,
                                               NumericMatrix projection_distances);
List project_embedding_affine_metal_impl(NumericMatrix reference_data,
                                         NumericMatrix query_data,
                                         NumericMatrix reference_layout,
                                         IntegerMatrix projection_indices,
                                         NumericMatrix projection_distances,
                                         int max_neighbors,
                                         double ridge,
                                         double max_extrapolation);
NumericMatrix interpolate_landmark_layout_metal_impl(NumericMatrix landmark_layout,
                                                     IntegerVector landmark_indices,
                                                     IntegerMatrix projection_indices,
                                                     NumericMatrix projection_distances,
                                                     int n);
NumericMatrix landmark_project_interpolate_metal_impl(NumericMatrix landmark_data,
                                                      NumericMatrix query_data,
                                                      NumericMatrix landmark_layout,
                                                      IntegerVector landmark_indices,
                                                      int k);
List landmark_project_interpolate_knn_confidence_metal_impl(NumericMatrix landmark_data,
                                                            NumericMatrix query_data,
                                                            NumericMatrix landmark_layout,
                                                            IntegerVector landmark_indices,
                                                            int k);
NumericVector knn_structure_score_metal_impl(NumericMatrix layout,
                                             IntegerMatrix indices,
                                             IntegerVector keep,
                                             int preserve_k,
                                             IntegerVector labels,
                                             int n_label_levels);
double silhouette_score_metal_impl(NumericMatrix layout,
                                   IntegerVector labels,
                                   int n_label_levels);
NumericMatrix knn_embed_metal_impl(IntegerMatrix indices,
                                   NumericMatrix distances,
                                   NumericMatrix init,
                                   std::string objective,
                                   int n_epochs,
                                   int negative_sample_rate,
                                   double learning_rate,
                                   double min_dist,
                                   int seed);
NumericMatrix knn_embed_metal_csr_impl(IntegerVector offsets,
                                       IntegerVector neighbors,
                                       NumericVector weights,
                                       NumericMatrix init,
                                       int n_epochs,
                                       int negative_sample_rate,
                                       double learning_rate,
                                       double min_dist,
                                       double max_weight,
                                       double repulsion_strength,
                                       int seed);
NumericMatrix knn_umap_refine_rows_metal_impl(IntegerMatrix indices,
                                              NumericMatrix distances,
                                              IntegerVector row_ids,
                                              NumericMatrix init_embedding,
                                              int n_epochs,
                                              double min_dist,
                                              int negative_sample_rate,
                                              double learning_rate,
                                              double repulsion_strength,
                                              int seed);
NumericMatrix rsvd_multiply_metal_impl(NumericMatrix left,
                                       NumericMatrix right,
                                       bool transpose_left);
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
                               int seed);
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
                                  bool record_costs);
List metal_fft512_stockham_diagnostic_impl(int seed,
                                           bool inverse,
                                           int n_checks);
List metal_mpsgraph_fft_diagnostic_impl(int fft_size,
                                        int seed,
                                        int n_repeats);
List metal_mpsgraph_convolution_diagnostic_impl(int fft_size,
                                                int seed,
                                                int n_repeats);

// [[Rcpp::export]]
bool embedding_metal_available_cpp() {
  return embedding_metal_available_impl();
}

// [[Rcpp::export]]
NumericMatrix spectral_knn_init_metal_cpp(IntegerMatrix indices,
                                          NumericMatrix distances,
                                          int n_components,
                                          int spectral_n_iter,
                                          int seed) {
  return spectral_knn_init_metal_impl(
    indices, distances, n_components, spectral_n_iter, seed
  );
}

// [[Rcpp::export]]
List standardize_metal_cpp(NumericMatrix data) {
  return standardize_metal_impl(data);
}

// [[Rcpp::export]]
NumericMatrix project_embedding_knn_metal_cpp(NumericMatrix reference_layout,
                                              IntegerMatrix projection_indices,
                                              NumericMatrix projection_distances) {
  return project_embedding_knn_metal_impl(
    reference_layout, projection_indices, projection_distances
  );
}

// [[Rcpp::export]]
List project_embedding_affine_metal_cpp(NumericMatrix reference_data,
                                        NumericMatrix query_data,
                                        NumericMatrix reference_layout,
                                        IntegerMatrix projection_indices,
                                        NumericMatrix projection_distances,
                                        int max_neighbors = 12,
                                        double ridge = 1e-3,
                                        double max_extrapolation = 2.5) {
  return project_embedding_affine_metal_impl(
    reference_data,
    query_data,
    reference_layout,
    projection_indices,
    projection_distances,
    max_neighbors,
    ridge,
    max_extrapolation
  );
}

// [[Rcpp::export]]
NumericMatrix interpolate_landmark_layout_metal_cpp(NumericMatrix landmark_layout,
                                                    IntegerVector landmark_indices,
                                                    IntegerMatrix projection_indices,
                                                    NumericMatrix projection_distances,
                                                    int n) {
  return interpolate_landmark_layout_metal_impl(
    landmark_layout,
    landmark_indices,
    projection_indices,
    projection_distances,
    n
  );
}

// [[Rcpp::export]]
NumericMatrix landmark_project_interpolate_metal_cpp(NumericMatrix landmark_data,
                                                     NumericMatrix query_data,
                                                     NumericMatrix landmark_layout,
                                                     IntegerVector landmark_indices,
                                                     int k) {
  return landmark_project_interpolate_metal_impl(
    landmark_data,
    query_data,
    landmark_layout,
    landmark_indices,
    k
  );
}

// [[Rcpp::export]]
List landmark_project_interpolate_knn_confidence_metal_cpp(NumericMatrix landmark_data,
                                                           NumericMatrix query_data,
                                                           NumericMatrix landmark_layout,
                                                           IntegerVector landmark_indices,
                                                           int k) {
  return landmark_project_interpolate_knn_confidence_metal_impl(
    landmark_data,
    query_data,
    landmark_layout,
    landmark_indices,
    k
  );
}

// [[Rcpp::export]]
NumericVector knn_structure_score_metal_cpp(NumericMatrix layout,
                                            IntegerMatrix indices,
                                            IntegerVector keep,
                                            int preserve_k,
                                            IntegerVector labels,
                                            int n_label_levels) {
  return knn_structure_score_metal_impl(
    layout, indices, keep, preserve_k, labels, n_label_levels
  );
}

// [[Rcpp::export]]
double silhouette_score_metal_cpp(NumericMatrix layout,
                                  IntegerVector labels,
                                  int n_label_levels) {
  return silhouette_score_metal_impl(layout, labels, n_label_levels);
}

// [[Rcpp::export]]
NumericMatrix knn_embed_metal_cpp(IntegerMatrix indices,
                                  NumericMatrix distances,
                                  NumericMatrix init,
                                  std::string objective,
                                  int n_epochs,
                                  int negative_sample_rate,
                                  double learning_rate,
                                  double min_dist,
                                  int seed) {
  return knn_embed_metal_impl(
    indices, distances, init, objective, n_epochs,
    negative_sample_rate, learning_rate, min_dist, seed
  );
}

// [[Rcpp::export]]
NumericMatrix knn_embed_metal_csr_cpp(IntegerVector offsets,
                                      IntegerVector neighbors,
                                      NumericVector weights,
                                      NumericMatrix init,
                                      int n_epochs,
                                      int negative_sample_rate,
                                      double learning_rate,
                                      double min_dist,
                                      double max_weight,
                                      double repulsion_strength,
                                      int seed) {
  return knn_embed_metal_csr_impl(
    offsets, neighbors, weights, init, n_epochs,
    negative_sample_rate, learning_rate, min_dist, max_weight,
    repulsion_strength, seed
  );
}

// [[Rcpp::export]]
NumericMatrix knn_umap_refine_rows_metal_cpp(IntegerMatrix indices,
                                             NumericMatrix distances,
                                             IntegerVector row_ids,
                                             NumericMatrix init_embedding,
                                             int n_epochs,
                                             double min_dist,
                                             int negative_sample_rate,
                                             double learning_rate,
                                             double repulsion_strength,
                                             int seed) {
  return knn_umap_refine_rows_metal_impl(
    indices,
    distances,
    row_ids,
    init_embedding,
    n_epochs,
    min_dist,
    negative_sample_rate,
    learning_rate,
    repulsion_strength,
    seed
  );
}

// [[Rcpp::export]]
NumericMatrix rsvd_multiply_metal_cpp(NumericMatrix left,
                                      NumericMatrix right,
                                      bool transpose_left) {
  return rsvd_multiply_metal_impl(left, right, transpose_left);
}

// [[Rcpp::export]]
List transform_tsne_metal_cpp(NumericMatrix reference_layout,
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
  return transform_tsne_metal_impl(
    reference_layout,
    indices,
    distances,
    y_init,
    init,
    initialization,
    perplexity,
    n_iter,
    early_exaggeration_iter,
    learning_rate,
    early_exaggeration,
    exaggeration,
    initial_momentum,
    final_momentum,
    max_grad_norm,
    max_step_norm,
    n_negatives,
    exact_repulsion_threshold,
    seed
  );
}

// [[Rcpp::export]]
List knn_tsne_opentsne_metal_cpp(IntegerMatrix indices,
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
  return knn_tsne_opentsne_metal_impl(
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
List metal_fft512_stockham_diagnostic_cpp(int seed = 1,
                                          bool inverse = false,
                                          int n_checks = 8) {
  return metal_fft512_stockham_diagnostic_impl(seed, inverse, n_checks);
}

// [[Rcpp::export]]
List metal_mpsgraph_fft_diagnostic_cpp(int fft_size = 512,
                                       int seed = 1,
                                       int n_repeats = 5) {
  return metal_mpsgraph_fft_diagnostic_impl(fft_size, seed, n_repeats);
}

// [[Rcpp::export]]
List metal_mpsgraph_convolution_diagnostic_cpp(int fft_size = 512,
                                               int seed = 1,
                                               int n_repeats = 5) {
  return metal_mpsgraph_convolution_diagnostic_impl(fft_size, seed, n_repeats);
}
