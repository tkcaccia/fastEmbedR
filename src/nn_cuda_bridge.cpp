#include <Rcpp.h>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuda_is_available_impl();
List cuda_nn_impl(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square);

// [[Rcpp::export]]
bool cuda_available_cpp() {
  return cuda_is_available_impl();
}

// [[Rcpp::export]]
List nn_cuda_cpp(NumericMatrix data,
                 NumericMatrix points,
                 int k,
                 bool square) {
  return cuda_nn_impl(data, points, k, square);
}
