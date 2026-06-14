#include <Rcpp.h>

using Rcpp::List;
using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::NumericMatrix;

bool metal_is_available_impl() {
  return false;
}

List metal_nn_impl(NumericMatrix,
                   NumericMatrix,
                   int,
                   bool) {
  Rcpp::stop("Metal GPU backend is only available on macOS with Metal support.");
}

List metal_landmark_candidate_knn_impl(NumericMatrix,
                                       IntegerMatrix,
                                       int,
                                       int,
                                       int) {
  Rcpp::stop("Metal approximate candidate KNN is only available on macOS with Metal support.");
}

List metal_grid_knn_impl(NumericMatrix,
                         int,
                         int,
                         int,
                         int) {
  Rcpp::stop("Metal grid KNN is only available on macOS with Metal support.");
}

List metal_row_candidate_knn_impl(NumericMatrix,
                                  IntegerMatrix,
                                  int) {
  Rcpp::stop("Metal row-candidate KNN is only available on macOS with Metal support.");
}

List metal_candidate_topk_l2_batched_impl(NumericMatrix,
                                          IntegerMatrix,
                                          int) {
  Rcpp::stop("Metal batched candidate top-k KNN is only available on macOS with Metal support.");
}

SEXP metal_knn_data_handle_impl(NumericMatrix) {
  Rcpp::stop("Metal persistent KNN data buffers are only available on macOS with Metal support.");
}

List metal_row_candidate_knn_handle_impl(SEXP,
                                         IntegerMatrix,
                                         int,
                                         bool) {
  Rcpp::stop("Metal persistent row-candidate KNN is only available on macOS with Metal support.");
}

List metal_row_candidate_knn_subset_handle_impl(SEXP,
                                                IntegerMatrix,
                                                IntegerVector,
                                                int,
                                                bool) {
  Rcpp::stop("Metal persistent subset row-candidate KNN is only available on macOS with Metal support.");
}
