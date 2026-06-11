#include <Rcpp.h>

using Rcpp::List;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

bool metal_is_available_impl();
List metal_nn_impl(NumericMatrix data,
                   NumericMatrix points,
                   int k,
                   bool square);
List metal_landmark_candidate_knn_impl(NumericMatrix data,
                                       IntegerMatrix projection_indices,
                                       int k,
                                       int bucket_cols,
                                       int query_cols);
List metal_grid_knn_impl(NumericMatrix data,
                         int k,
                         int grid_dims,
                         int bins_per_dim,
                         int radius);
List metal_row_candidate_knn_impl(NumericMatrix data,
                                  IntegerMatrix candidate_indices,
                                  int k);

// [[Rcpp::export]]
bool metal_available_cpp() {
  return metal_is_available_impl();
}

// [[Rcpp::export]]
List nn_metal_cpp(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square) {
  return metal_nn_impl(data, points, k, square);
}

// [[Rcpp::export]]
List landmark_candidate_knn_metal_cpp(NumericMatrix data,
                                      IntegerMatrix projection_indices,
                                      int k,
                                      int bucket_cols,
                                      int query_cols) {
  return metal_landmark_candidate_knn_impl(
    data, projection_indices, k, bucket_cols, query_cols
  );
}

// [[Rcpp::export]]
List grid_knn_metal_cpp(NumericMatrix data,
                        int k,
                        int grid_dims,
                        int bins_per_dim,
                        int radius) {
  return metal_grid_knn_impl(data, k, grid_dims, bins_per_dim, radius);
}

// [[Rcpp::export]]
List row_candidate_knn_metal_cpp(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k) {
  return metal_row_candidate_knn_impl(data, candidate_indices, k);
}
