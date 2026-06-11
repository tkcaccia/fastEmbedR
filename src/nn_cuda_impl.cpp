#include <Rcpp.h>

#include <string>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

extern "C" {
bool fastembedr_cuda_available();
const char* fastembedr_cuda_last_error();
const char* fastembedr_cuda_device_info_json();
int fastembedr_cuda_knn(const double* data,
                        const double* points,
                        int n_data,
                        int n_points,
                        int n_features,
                        int k,
                        int square,
                        int* out_indices,
                        double* out_distances);
int fastembedr_cuda_landmark_candidate_knn(const double* data,
                                           const int* projection_indices,
                                           int n,
                                           int n_features,
                                           int projection_k,
                                           int k,
                                           int bucket_cols,
                                           int query_cols,
                                           int* out_indices,
                                           double* out_distances);
int fastembedr_cuda_row_candidate_knn(const double* data,
                                      const int* candidate_indices,
                                      int n,
                                      int n_features,
                                      int n_candidates,
                                      int k,
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

std::string cuda_device_info_json_impl() {
  const char* info = fastembedr_cuda_device_info_json();
  return info == nullptr ? std::string("{}") : std::string(info);
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

List cuda_landmark_candidate_knn_impl(NumericMatrix data,
                                      IntegerMatrix projection_indices,
                                      int k,
                                      int bucket_cols,
                                      int query_cols) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int projection_k = projection_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must match data");
  }
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < projection_k; ++c) {
      if (projection_indices(i, c) < 1) {
        Rcpp::stop("projection_indices must be 1-based positive integers");
      }
    }
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  const int status = fastembedr_cuda_landmark_candidate_knn(
    data.begin(),
    projection_indices.begin(),
    n,
    n_features,
    projection_k,
    k,
    bucket_cols,
    query_cols,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA landmark candidate KNN failed: %s", cuda_error_message());
  }
  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

List cuda_row_candidate_knn_impl(NumericMatrix data,
                                 IntegerMatrix candidate_indices,
                                 int k) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int n_candidates = candidate_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (candidate_indices.nrow() != n) {
    Rcpp::stop("candidate_indices row count must match data");
  }
  if (n_candidates < 1) Rcpp::stop("candidate_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (k > kMaxCudaK) Rcpp::stop("CUDA backend currently supports k <= %d", kMaxCudaK);
  if (!fastembedr_cuda_available()) Rcpp::stop("No CUDA device is available.");

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  const int status = fastembedr_cuda_row_candidate_knn(
    data.begin(),
    candidate_indices.begin(),
    n,
    n_features,
    n_candidates,
    k,
    indices.begin(),
    distances.begin()
  );
  if (status != 0) {
    Rcpp::stop("CUDA row candidate KNN failed: %s", cuda_error_message());
  }
  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
  result.attr("cuda_kernel") = "row_candidate_knn";
  result.attr("candidate_columns") = n_candidates;
  return result;
}
