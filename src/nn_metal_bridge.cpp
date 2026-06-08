#include <Rcpp.h>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool metal_is_available_impl();
List metal_nn_impl(NumericMatrix data,
                   NumericMatrix points,
                   int k,
                   bool square);

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
