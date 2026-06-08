#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

double matrix_value(const double* x, const int row, const int col, const int nrow) {
  return x[static_cast<std::size_t>(col) * nrow + row];
}

double distance_value(const double* data,
                      const double* points,
                      const int data_row,
                      const int point_row,
                      const int n_data,
                      const int n_points,
                      const int n_features,
                      const std::string& method,
                      const double p) {
  double acc = 0.0;
  if (method == "euclidean") {
    for (int c = 0; c < n_features; ++c) {
      const double diff = matrix_value(data, data_row, c, n_data) -
                          matrix_value(points, point_row, c, n_points);
      acc += diff * diff;
    }
    return acc;
  }
  if (method == "manhattan") {
    for (int c = 0; c < n_features; ++c) {
      acc += std::abs(matrix_value(data, data_row, c, n_data) -
                      matrix_value(points, point_row, c, n_points));
    }
    return acc;
  }

  for (int c = 0; c < n_features; ++c) {
    acc += std::pow(
      std::abs(matrix_value(data, data_row, c, n_data) -
               matrix_value(points, point_row, c, n_points)),
      p
    );
  }
  return acc;
}

bool neighbor_less(const std::pair<double, int>& a, const std::pair<double, int>& b) {
  if (a.first == b.first) return a.second < b.second;
  return a.first < b.first;
}

int requested_threads(const bool parallel, const int cores, const int n_points) {
  if (!parallel || n_points < 2) return 1;
  int n_threads = cores;
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = 1;
  }
  return std::max(1, std::min(n_threads, n_points));
}

} // namespace

// [[Rcpp::export]]
List nn_cpp(NumericMatrix data,
            NumericMatrix points,
            int k,
            std::string method,
            bool square,
            bool sorted,
            double p,
            bool parallel,
            int cores) {
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  if (points.ncol() != n_features) Rcpp::stop("data and points must have the same number of columns");
  if (k < 1 || k > n_data) Rcpp::stop("k must be in [1, nrow(data)]");
  if (method != "euclidean" && method != "manhattan" && method != "minkowski") {
    Rcpp::stop("unsupported method");
  }
  if (method == "minkowski" && (!std::isfinite(p) || p <= 0.0)) {
    Rcpp::stop("p must be positive for minkowski distance");
  }

  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);
  const double* data_ptr = data.begin();
  const double* points_ptr = points.begin();
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();

  const auto write_result = [&](const int query_start, const int query_end) {
    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(n_data);
    for (int q = query_start; q < query_end; ++q) {
      candidates.clear();
      for (int i = 0; i < n_data; ++i) {
        candidates.emplace_back(
          distance_value(data_ptr, points_ptr, i, q, n_data, n_points, n_features, method, p),
          i
        );
      }

      std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), neighbor_less);
      if (sorted) {
        std::sort(candidates.begin(), candidates.begin() + k, neighbor_less);
      }

      for (int j = 0; j < k; ++j) {
        double dist = candidates[j].first;
        if (method == "euclidean" && !square) {
          dist = std::sqrt(std::max(dist, 0.0));
        } else if (method == "minkowski") {
          dist = std::pow(std::max(dist, 0.0), 1.0 / p);
        }
        indices_ptr[static_cast<std::size_t>(j) * n_points + q] = candidates[j].second + 1;
        distances_ptr[static_cast<std::size_t>(j) * n_points + q] = dist;
      }
    }
  };

  const int n_threads = requested_threads(parallel, cores, n_points);
  if (n_threads == 1) {
    write_result(0, n_points);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_points * t) / n_threads;
      const int end = (n_points * (t + 1)) / n_threads;
      workers.emplace_back(write_result, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}
