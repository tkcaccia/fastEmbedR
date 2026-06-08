#include <Rcpp.h>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuda_is_available_impl() {
  return false;
}

List cuda_nn_impl(NumericMatrix,
                  NumericMatrix,
                  int,
                  bool) {
  Rcpp::stop("CUDA GPU backend is available only when the package is built with CUDA support.");
}
