#include <Rcpp.h>

using Rcpp::List;
using Rcpp::IntegerMatrix;
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
