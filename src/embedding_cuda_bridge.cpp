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
                                       int spectral_n_iter,
                                       int seed);
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
                                      int spectral_n_iter,
                                      int seed) {
  return knn_umap_cuda_fused_impl(
    indices,
    distances,
    n_epochs,
    negative_sample_rate,
    learning_rate,
    min_dist,
    spectral_n_iter,
    seed
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
