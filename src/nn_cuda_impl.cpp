#include <Rcpp.h>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

extern "C" {
bool fastembedr_cuda_available();
const char* fastembedr_cuda_last_error();
int fastembedr_cuda_knn(const double* data,
                        const double* points,
                        int n_data,
                        int n_points,
                        int n_features,
                        int k,
                        int square,
                        int* out_indices,
                        double* out_distances);
}

namespace {

constexpr int kMaxCudaK = 256;

const char* cuda_error_message() {
  const char* msg = fastembedr_cuda_last_error();
  return msg == nullptr ? "unknown CUDA error" : msg;
}

} // namespace

bool cuda_is_available_impl() {
  return fastembedr_cuda_available();
}

List cuda_nn_impl(NumericMatrix data,
                  NumericMatrix points,
                  int k,
                  bool square) {
  if (data.ncol() != points.ncol()) Rcpp::stop("data and points must have the same number of columns");
  if (k < 1 || k > data.nrow()) Rcpp::stop("k must be in [1, nrow(data)]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);

  const int status = fastembedr_cuda_knn(
    data.begin(),
    points.begin(),
    n_data,
    n_points,
    n_features,
    k,
    square ? 1 : 0,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA KNN failed: %s", cuda_error_message());
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}
