/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Exhaustive float32 KNN behavior distilled from faissR 0.99.48, commit
 * 09f4c88fe8af431053a35a945db809a1da22033e. This implementation does not
 * include, link, or call FAISS or faissR.
 * Upstream behavior reference: src/nn_faiss_impl.cpp.
 */

#include <Rcpp.h>

#include "native_knn_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ExactNeighbor {
  float distance;
  int id;
};

struct FartherFirst {
  bool operator()(const ExactNeighbor& lhs,
                  const ExactNeighbor& rhs) const {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.id < rhs.id;
  }
};

bool closer(const ExactNeighbor& lhs, const ExactNeighbor& rhs) {
  return lhs.distance < rhs.distance ||
    (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

std::vector<char> zero_rows(const fastembedr::FloatMatrix& matrix,
                            fastembedr::KnnMetric metric) {
  std::vector<char> result;
  if (metric == fastembedr::KnnMetric::Euclidean) return result;
  result.assign(static_cast<std::size_t>(matrix.nrow), 0);
  for (int row = 0; row < matrix.nrow; ++row) {
    const float* values = matrix.values.data() +
      static_cast<std::size_t>(row) * matrix.ncol;
    bool zero = true;
    for (int column = 0; column < matrix.ncol; ++column) {
      zero = zero && values[column] == 0.0f;
    }
    result[static_cast<std::size_t>(row)] = zero ? 1 : 0;
  }
  return result;
}

float exact_output_distance(float squared_distance,
                            fastembedr::KnnMetric metric,
                            bool query_zero,
                            bool reference_zero) {
  if (metric == fastembedr::KnnMetric::Euclidean) {
    return std::sqrt(std::max(0.0f, squared_distance));
  }
  if (query_zero || reference_zero) {
    return query_zero && reference_zero ? 0.0f : 1.0f;
  }
  return 0.5f * std::max(0.0f, squared_distance);
}

void update_heap(std::vector<ExactNeighbor>& heap,
                 const ExactNeighbor& candidate,
                 int k) {
  if (static_cast<int>(heap.size()) < k) {
    heap.push_back(candidate);
    std::push_heap(heap.begin(), heap.end(), FartherFirst{});
  } else if (closer(candidate, heap.front())) {
    std::pop_heap(heap.begin(), heap.end(), FartherFirst{});
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end(), FartherFirst{});
  }
}

struct ExactOutput {
  std::vector<int> indices;
  std::vector<float> distances;
};

ExactOutput exact_search(const fastembedr::FloatMatrix& reference,
                         const fastembedr::FloatMatrix& query,
                         int k,
                         int n_threads,
                         bool exclude_self) {
  const int n_query = query.nrow;
  const int n_reference = reference.nrow;
  ExactOutput output;
  output.indices.resize(static_cast<std::size_t>(n_query) * k);
  output.distances.resize(static_cast<std::size_t>(n_query) * k);
  std::atomic<int> next(0);
  n_threads = fastembedr::adaptive_worker_count(n_threads, n_query);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(n_threads));
  for (int worker = 0; worker < n_threads; ++worker) {
    workers.emplace_back([&]() {
      std::vector<ExactNeighbor> heap;
      heap.reserve(static_cast<std::size_t>(k));
      while (true) {
        const int query_id = next.fetch_add(1, std::memory_order_relaxed);
        if (query_id >= n_query) break;
        heap.clear();
        const float* query_row = query.values.data() +
          static_cast<std::size_t>(query_id) * query.ncol;
        for (int reference_id = 0;
             reference_id < n_reference;
             ++reference_id) {
          if (exclude_self && query_id == reference_id) continue;
          const float* reference_row = reference.values.data() +
            static_cast<std::size_t>(reference_id) * reference.ncol;
          update_heap(
            heap,
            {
              fastembedr::squared_l2_distance(
                query_row, reference_row, reference.ncol
              ),
              reference_id
            },
            k
          );
        }
        std::sort(heap.begin(), heap.end(), closer);
        for (int rank = 0; rank < k; ++rank) {
          const std::size_t position =
            static_cast<std::size_t>(query_id) * k + rank;
          output.indices[position] = heap[static_cast<std::size_t>(rank)].id;
          output.distances[position] =
            heap[static_cast<std::size_t>(rank)].distance;
        }
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  return output;
}

Rcpp::List format_exact_result(
    const ExactOutput& output,
    const fastembedr::FloatMatrix& reference,
    const fastembedr::FloatMatrix& query,
    fastembedr::KnnMetric metric,
    const std::string& metric_name,
    int k,
    int n_threads,
    bool exclude_self,
    double target_recall,
    double convert_seconds,
    double search_seconds) {
  const std::vector<char> reference_zero = zero_rows(reference, metric);
  const std::vector<char> query_zero = exclude_self ?
    reference_zero : zero_rows(query, metric);
  Rcpp::IntegerMatrix indices(query.nrow, k);
  Rcpp::NumericMatrix distances(query.nrow, k);
  for (int row = 0; row < query.nrow; ++row) {
    for (int rank = 0; rank < k; ++rank) {
      const std::size_t position =
        static_cast<std::size_t>(row) * k + rank;
      const int reference_id = output.indices[position];
      const bool query_is_zero = !query_zero.empty() &&
        query_zero[static_cast<std::size_t>(row)] != 0;
      const bool reference_is_zero = !reference_zero.empty() &&
        reference_zero[static_cast<std::size_t>(reference_id)] != 0;
      indices(row, rank) = reference_id + 1;
      distances(row, rank) = exact_output_distance(
        output.distances[position],
        metric,
        query_is_zero,
        reference_is_zero
      );
    }
  }
  const std::string method = exclude_self ?
    "native_exact" : "native_exact_query";
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("backend_used") = "native_cpu_exact",
    Rcpp::Named("method") = method,
    Rcpp::Named("metric") = metric_name,
    Rcpp::Named("exact") = true,
    Rcpp::Named("exact_recall_by_construction") = true,
    Rcpp::Named("expected_recall_at_k") = 1.0,
    Rcpp::Named("target_recall") = target_recall,
    Rcpp::Named("target_met") = true,
    Rcpp::Named("recall_audited") = true,
    Rcpp::Named("recall_status") = "exact_by_construction",
    Rcpp::Named("tuning_policy") = "exact_below_5000",
    Rcpp::Named("tuning_rule") = "native_cpu_exact_n_lt_5000",
    Rcpp::Named("n_threads") = n_threads,
    Rcpp::Named("input_type") = reference.input_float32 &&
      query.input_float32 ? "float32" : "numeric_to_float32",
    Rcpp::Named("distance_computation") = "float32",
    Rcpp::Named("cpu_fallback") = false,
    Rcpp::Named("timing") = Rcpp::NumericVector::create(
      Rcpp::Named("convert") = convert_seconds,
      Rcpp::Named("search") = search_seconds
    )
  );
}

}  // namespace

Rcpp::List native_exact_knn_impl(SEXP data_sexp,
                                 int k,
                                 int n_threads,
                                 const std::string& metric_name,
                                 double target_recall) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const fastembedr::KnnMetric metric =
    fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix data =
    fastembedr::matrix_to_row_major_float(data_sexp, metric);
  fastembedr::require_finite_matrix(data);
  if (data.nrow < 2 || k < 1 || k >= data.nrow) {
    Rcpp::stop("Invalid native exact KNN input.");
  }
  const auto converted = Clock::now();
  const int threads = std::max(1, std::min(n_threads, data.nrow));
  const ExactOutput output = exact_search(
    data, data, k, threads, true
  );
  const auto searched = Clock::now();
  Rcpp::List result = format_exact_result(
    output, data, data, metric, metric_name, k, threads, true,
    target_recall,
    std::chrono::duration<double>(converted - start).count(),
    std::chrono::duration<double>(searched - converted).count()
  );
  result.attr("backend") = "cpu";
  result.attr("method") = "native_exact";
  result.attr("exclude_self") = true;
  return result;
}

Rcpp::List native_exact_query_impl(SEXP data_sexp,
                                   SEXP query_sexp,
                                   int k,
                                   int n_threads,
                                   const std::string& metric_name,
                                   double target_recall) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const fastembedr::KnnMetric metric =
    fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix data =
    fastembedr::matrix_to_row_major_float(data_sexp, metric);
  fastembedr::FloatMatrix query =
    fastembedr::matrix_to_row_major_float(query_sexp, metric);
  fastembedr::require_finite_matrix(data);
  fastembedr::require_finite_matrix(query);
  if (data.nrow < 1 || query.nrow < 1 || data.ncol != query.ncol ||
      k < 1 || k > data.nrow) {
    Rcpp::stop("Invalid native exact query KNN input.");
  }
  const auto converted = Clock::now();
  const int threads = std::max(1, std::min(n_threads, query.nrow));
  const ExactOutput output = exact_search(
    data, query, k, threads, false
  );
  const auto searched = Clock::now();
  Rcpp::List result = format_exact_result(
    output, data, query, metric, metric_name, k, threads, false,
    target_recall,
    std::chrono::duration<double>(converted - start).count(),
    std::chrono::duration<double>(searched - converted).count()
  );
  result.attr("backend") = "cpu";
  result.attr("method") = "native_exact_query";
  result.attr("exclude_self") = false;
  return result;
}
