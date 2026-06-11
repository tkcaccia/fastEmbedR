#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

void validate_inputs(const NumericMatrix& data,
                     const NumericMatrix& points,
                     const int k,
                     const bool exclude_self) {
  if (data.nrow() < 1 || points.nrow() < 1) {
    Rcpp::stop("data and points must have at least one row");
  }
  if (data.ncol() != points.ncol()) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (data.ncol() < 1) {
    Rcpp::stop("data and points must have at least one column");
  }
  if (k < 1 || k > data.nrow()) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && data.nrow() != points.nrow()) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }
  if (data.nrow() > std::numeric_limits<int>::max() ||
      points.nrow() > std::numeric_limits<int>::max() ||
      data.ncol() > std::numeric_limits<int>::max()) {
    Rcpp::stop("FAISS backend currently supports dimensions that fit in int");
  }
}

void copy_row_major_float(const NumericMatrix& src, std::vector<float>& dest) {
  const int nrow = src.nrow();
  const int ncol = src.ncol();
  dest.assign(static_cast<std::size_t>(nrow) * ncol, 0.0f);
  for (int c = 0; c < ncol; ++c) {
    for (int r = 0; r < nrow; ++r) {
      const double value = src(r, c);
      if (!std::isfinite(value)) {
        Rcpp::stop("FAISS backend requires finite numeric input");
      }
      dest[static_cast<std::size_t>(r) * ncol + c] =
        static_cast<float>(value);
    }
  }
}

bool same_matrix_storage(const NumericMatrix& data,
                         const NumericMatrix& points) {
  return data.nrow() == points.nrow() &&
    data.ncol() == points.ncol() &&
    data.begin() == points.begin();
}

class OmpThreadScope {
 public:
  explicit OmpThreadScope(const int n_threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (n_threads > 0) {
      omp_set_num_threads(std::max(1, n_threads));
    }
#else
    (void)n_threads;
#endif
  }

  ~OmpThreadScope() {
#ifdef _OPENMP
    if (previous_ > 0) {
      omp_set_num_threads(previous_);
    }
#endif
  }

 private:
  int previous_ = 0;
};

List format_faiss_result(const std::vector<faiss::idx_t>& labels,
                         const std::vector<float>& distances,
                         const int n_points,
                         const int search_k,
                         const int out_k,
                         const bool self_query,
                         const bool exclude_self,
                         const std::string& index_type,
                         const bool exact,
                         const int nlist = NA_INTEGER,
                         const int nprobe = NA_INTEGER) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();

  for (int i = 0; i < n_points; ++i) {
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const faiss::idx_t label = labels[static_cast<std::size_t>(i) * search_k + j];
      if (label < 0) continue;
      if (exclude_self && self_query && label == i) continue;
      indices_ptr[static_cast<std::size_t>(written) * n_points + i] =
        static_cast<int>(label) + 1;
      const float sq = distances[static_cast<std::size_t>(i) * search_k + j];
      dists_ptr[static_cast<std::size_t>(written) * n_points + i] =
        std::sqrt(std::max(static_cast<double>(sq), 0.0));
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("FAISS returned fewer neighbors than requested");
    }
  }

  List out = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
  if (nlist != NA_INTEGER) out["nlist"] = nlist;
  if (nprobe != NA_INTEGER) out["nprobe"] = nprobe;
  return out;
}

} // namespace

bool faiss_is_available_impl() {
  return true;
}

std::string faiss_info_json_impl() {
  return "{\"available\":true,\"library\":\"faiss\",\"interface\":\"c++\"}";
}

List faiss_flat_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int n_threads) {
  validate_inputs(data, points, k, exclude_self);
  const bool self_query = exclude_self || same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (same_matrix_storage(data, points)) {
    xq.clear();
  } else {
    copy_row_major_float(points, xq);
  }
  const float* query_ptr = same_matrix_storage(data, points) ? xb.data() : xq.data();

  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);

  try {
    OmpThreadScope threads(n_threads);
    faiss::IndexFlatL2 index(n_features);
    index.add(n_data, xb.data());
    index.search(n_points, query_ptr, search_k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS IndexFlatL2 search failed: %s", e.what());
  }

  return format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    "IndexFlatL2", true
  );
}

List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        bool exclude_self,
                        int n_threads) {
  validate_inputs(data, points, k, exclude_self);
  const bool self_query = exclude_self || same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (same_matrix_storage(data, points)) {
    xq.clear();
  } else {
    copy_row_major_float(points, xq);
  }
  const float* query_ptr = same_matrix_storage(data, points) ? xb.data() : xq.data();

  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(n_points) * search_k);

  try {
    OmpThreadScope threads(n_threads);
    faiss::IndexFlatL2 quantizer(n_features);
    faiss::IndexIVFFlat index(&quantizer, n_features, nlist, faiss::METRIC_L2);
    index.nprobe = nprobe;
    index.train(n_data, xb.data());
    index.add(n_data, xb.data());
    index.search(n_points, query_ptr, search_k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS IndexIVFFlat search failed: %s", e.what());
  }

  return format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    "IndexIVFFlat", false, nlist, nprobe
  );
}
