#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

bool cuda_is_available_impl();
std::string cuda_device_info_json_impl();
List cuda_nn_impl(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square);
List cuda_landmark_candidate_knn_impl(NumericMatrix data,
                                      IntegerMatrix projection_indices,
                                      int k,
                                      int bucket_cols,
                                      int query_cols);
List cuda_row_candidate_knn_impl(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k);
List cuda_grid_self_knn_impl(NumericMatrix data,
                             int k,
                             int bins_per_dim);

// [[Rcpp::export]]
bool cuda_available_cpp() {
  return cuda_is_available_impl();
}

// [[Rcpp::export]]
std::string cuda_device_info_json_cpp() {
  return cuda_device_info_json_impl();
}

// [[Rcpp::export]]
List nn_cuda_cpp(NumericMatrix data,
                 NumericMatrix points,
                 int k,
                 bool square) {
  return cuda_nn_impl(data, points, k, square);
}

// [[Rcpp::export]]
List landmark_candidate_knn_cuda_cpp(NumericMatrix data,
                                     IntegerMatrix projection_indices,
                                     int k,
                                     int bucket_cols,
                                     int query_cols) {
  return cuda_landmark_candidate_knn_impl(
    data, projection_indices, k, bucket_cols, query_cols
  );
}

// [[Rcpp::export]]
List row_candidate_knn_cuda_cpp(NumericMatrix data,
                                IntegerMatrix candidate_indices,
                                int k) {
  return cuda_row_candidate_knn_impl(data, candidate_indices, k);
}

// [[Rcpp::export]]
List cuda_grid_self_knn_cpp(NumericMatrix data,
                            int k,
                            int bins_per_dim) {
  return cuda_grid_self_knn_impl(data, k, bins_per_dim);
}
