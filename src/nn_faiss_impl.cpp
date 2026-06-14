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
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

enum class DistanceOutput {
  L2Squared,
  InnerProduct
};

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
#ifdef _OPENMP
  int previous_ = 0;
#endif
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
                         const DistanceOutput distance_output = DistanceOutput::L2Squared,
                         const int nlist = NA_INTEGER,
                         const int nprobe = NA_INTEGER,
                         const int graph_degree = NA_INTEGER,
                         const int search_width = NA_INTEGER) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();

  for (int i = 0; i < n_points; ++i) {
    double row_best_ip = -std::numeric_limits<double>::infinity();
    if (distance_output == DistanceOutput::InnerProduct) {
      for (int j = 0; j < search_k; ++j) {
        const faiss::idx_t label = labels[static_cast<std::size_t>(i) * search_k + j];
        if (label < 0) continue;
        if (exclude_self && self_query && label == i) continue;
        row_best_ip = std::max(
          row_best_ip,
          static_cast<double>(distances[static_cast<std::size_t>(i) * search_k + j])
        );
      }
      if (!std::isfinite(row_best_ip)) row_best_ip = 0.0;
    }
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const faiss::idx_t label = labels[static_cast<std::size_t>(i) * search_k + j];
      if (label < 0) continue;
      if (exclude_self && self_query && label == i) continue;
      indices_ptr[static_cast<std::size_t>(written) * n_points + i] =
        static_cast<int>(label) + 1;
      const float sq = distances[static_cast<std::size_t>(i) * search_k + j];
      const double value = distance_output == DistanceOutput::InnerProduct ?
        std::max(row_best_ip - static_cast<double>(sq), 0.0) :
        std::sqrt(std::max(static_cast<double>(sq), 0.0));
      dists_ptr[static_cast<std::size_t>(written) * n_points + i] = value;
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
  if (graph_degree != NA_INTEGER) out["graph_degree"] = graph_degree;
  if (search_width != NA_INTEGER) out["search_width"] = search_width;
  out["metric"] = distance_output == DistanceOutput::InnerProduct ?
    "inner_product_similarity_shifted_to_distance" : "euclidean";
  return out;
}

List search_faiss_index(faiss::Index& index,
                        NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        bool exclude_self,
                        int n_threads,
                        const std::string& index_type,
                        bool exact,
                        DistanceOutput distance_output,
                        int nlist = NA_INTEGER,
                        int nprobe = NA_INTEGER,
                        int graph_degree = NA_INTEGER,
                        int search_width = NA_INTEGER) {
  validate_inputs(data, points, k, exclude_self);
  const bool self_query = exclude_self || same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
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
    if (!index.is_trained) {
      index.train(n_data, xb.data());
    }
    index.add(n_data, xb.data());
    index.search(n_points, query_ptr, search_k, distances.data(), labels.data());
  } catch (const std::exception& e) {
    Rcpp::stop("FAISS %s search failed: %s", index_type.c_str(), e.what());
  }

  return format_faiss_result(
    labels, distances, n_points, search_k, k, self_query, exclude_self,
    index_type, exact, distance_output, nlist, nprobe, graph_degree, search_width
  );
}

int clamp_positive(const int value, const int fallback, const int upper) {
  int out = value > 0 ? value : fallback;
  if (upper > 0) out = std::min(out, upper);
  return std::max(1, out);
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
  const int n_features = data.ncol();
  faiss::IndexFlatL2 index(n_features);
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexFlatL2", true, DistanceOutput::L2Squared
  );
}

List faiss_flat_ip_knn_impl(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            bool exclude_self,
                            int n_threads) {
  const int n_features = data.ncol();
  faiss::IndexFlatIP index(n_features);
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexFlatIP", true, DistanceOutput::InnerProduct
  );
}

List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        bool exclude_self,
                        int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  faiss::IndexFlatL2 quantizer(n_features);
  faiss::IndexIVFFlat index(&quantizer, n_features, nlist, faiss::METRIC_L2);
  index.nprobe = nprobe;
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexIVFFlat", false, DistanceOutput::L2Squared, nlist, nprobe
  );
}

List faiss_ivfpq_knn_impl(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          int nlist,
                          int nprobe,
                          int pq_m,
                          int pq_nbits,
                          bool exclude_self,
                          int n_threads) {
  const int n_data = data.nrow();
  const int n_features = data.ncol();
  nlist = std::max(1, std::min(nlist, n_data));
  nprobe = std::max(1, std::min(nprobe, nlist));
  pq_m = clamp_positive(pq_m, 8, n_features);
  while (pq_m > 1 && (n_features % pq_m) != 0) --pq_m;
  pq_nbits = std::max(4, std::min(pq_nbits, 12));
  while (pq_nbits > 4 && (1 << pq_nbits) > n_data) {
    --pq_nbits;
  }
  faiss::IndexFlatL2 quantizer(n_features);
  faiss::IndexIVFPQ index(&quantizer, n_features, nlist, pq_m, pq_nbits, faiss::METRIC_L2);
  index.nprobe = nprobe;
  List out = search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexIVFPQ", false, DistanceOutput::L2Squared, nlist, nprobe
  );
  out["pq_m"] = pq_m;
  out["pq_nbits"] = pq_nbits;
  return out;
}

List faiss_hnsw_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         int m,
                         int ef_construction,
                         int ef_search,
                         bool exclude_self,
                         int n_threads) {
  const int n_features = data.ncol();
  m = clamp_positive(m, 32, data.nrow());
  ef_construction = std::max(ef_construction, m);
  ef_search = std::max(ef_search, k);
  faiss::IndexHNSWFlat index(n_features, m, faiss::METRIC_L2);
  index.hnsw.efConstruction = ef_construction;
  index.hnsw.efSearch = ef_search;
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexHNSWFlat", false, DistanceOutput::L2Squared,
    NA_INTEGER, NA_INTEGER, m, ef_search
  );
}

List faiss_nsg_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int r,
                        int search_l,
                        int build_type,
                        bool exclude_self,
                        int n_threads) {
  const int n_features = data.ncol();
  r = clamp_positive(r, 32, data.nrow());
  search_l = std::max(search_l, k);
  faiss::IndexNSGFlat index(n_features, r, faiss::METRIC_L2);
  index.nsg.search_L = search_l;
  index.build_type = static_cast<char>(build_type == 1 ? 1 : 0);
  index.GK = std::max(64, std::max(2 * k, 2 * r));
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexNSGFlat", false, DistanceOutput::L2Squared,
    NA_INTEGER, NA_INTEGER, r, search_l
  );
}

List faiss_nndescent_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              int graph_k,
                              int n_iter,
                              int search_l,
                              bool exclude_self,
                              int n_threads) {
  const int n_features = data.ncol();
  graph_k = std::max(graph_k, k);
  n_iter = std::max(1, n_iter);
  search_l = std::max(search_l, k);
  faiss::IndexNNDescentFlat index(n_features, graph_k, faiss::METRIC_L2);
  index.nndescent.iter = n_iter;
  index.nndescent.search_L = search_l;
  return search_faiss_index(
    index, data, points, k, exclude_self, n_threads,
    "IndexNNDescentFlat", false, DistanceOutput::L2Squared,
    NA_INTEGER, NA_INTEGER, graph_k, search_l
  );
}
