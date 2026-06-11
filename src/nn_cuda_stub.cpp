#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::IntegerMatrix;
using Rcpp::NumericMatrix;

bool cuda_is_available_impl() {
  return false;
}

std::string cuda_device_info_json_impl() {
  return "{\"available\":false,\"device_count\":0,\"reason\":\"package_not_built_with_cuda\"}";
}

List cuda_nn_impl(NumericMatrix,
                  NumericMatrix,
                  int,
                  bool) {
  Rcpp::stop("CUDA GPU backend is available only when the package is built with CUDA support.");
}

List cuda_landmark_candidate_knn_impl(NumericMatrix,
                                      IntegerMatrix,
                                      int,
                                      int,
                                      int) {
  Rcpp::stop("CUDA landmark candidate KNN is available only when the package is built with CUDA support.");
}

List cuda_row_candidate_knn_impl(NumericMatrix,
                                 IntegerMatrix,
                                 int) {
  Rcpp::stop("CUDA row-candidate KNN is available only when the package is built with CUDA support.");
}
