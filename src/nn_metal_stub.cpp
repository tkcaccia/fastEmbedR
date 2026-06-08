#include <Rcpp.h>

using Rcpp::List;
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
