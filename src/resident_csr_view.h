#ifndef FASTEMBEDR_RESIDENT_CSR_VIEW_H
#define FASTEMBEDR_RESIDENT_CSR_VIEW_H

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

struct ResidentCsrView {
  const int* narrow_offsets = nullptr;
  const double* wide_offsets = nullptr;
  const int* neighbors = nullptr;
  const double* numeric_weights = nullptr;
  const int* float_weights = nullptr;
  int rows = 0;
  std::uint64_t edges = 0;

  std::uint64_t offset(int row) const {
    return wide_offsets ?
      static_cast<std::uint64_t>(wide_offsets[row]) :
      static_cast<std::uint64_t>(narrow_offsets[row]);
  }

  float weight(std::uint64_t edge) const {
    if (float_weights) {
      float value;
      const int bits = float_weights[edge];
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }
    return static_cast<float>(numeric_weights[edge]);
  }
};

inline ResidentCsrView resident_csr_view(SEXP offsets,
    Rcpp::IntegerVector neighbors, SEXP weights, int rows) {
  ResidentCsrView view;
  view.rows = rows;
  view.edges = static_cast<std::uint64_t>(neighbors.size());
  view.neighbors = INTEGER(neighbors);
  if (TYPEOF(offsets) == INTSXP) {
    Rcpp::IntegerVector values(offsets);
    if (values.size() != static_cast<R_xlen_t>(rows) + 1) {
      Rcpp::stop("Resident UMAP CSR row count is inconsistent.");
    }
    view.narrow_offsets = INTEGER(values);
  } else if (TYPEOF(offsets) == REALSXP) {
    Rcpp::NumericVector values(offsets);
    if (values.size() != static_cast<R_xlen_t>(rows) + 1) {
      Rcpp::stop("Resident UMAP CSR row count is inconsistent.");
    }
    view.wide_offsets = REAL(values);
  } else {
    Rcpp::stop("Resident UMAP CSR offsets must be int32 or exact uint64.");
  }
  if (Rf_isS4(weights) && Rcpp::S4(weights).is("float32")) {
    Rcpp::S4 object(weights);
    SEXP data = object.slot("Data");
    if (TYPEOF(data) != INTSXP ||
        XLENGTH(data) != static_cast<R_xlen_t>(view.edges)) {
      Rcpp::stop("Resident UMAP float32 weights are invalid.");
    }
    view.float_weights = INTEGER(data);
  } else if (TYPEOF(weights) == REALSXP &&
             XLENGTH(weights) == static_cast<R_xlen_t>(view.edges)) {
    view.numeric_weights = REAL(weights);
  } else {
    Rcpp::stop("Resident UMAP weights do not match CSR edges.");
  }
  return view;
}

inline float resident_csr_max_weight(const ResidentCsrView& graph) {
  if (graph.rows < 2) Rcpp::stop(
    "Resident UMAP CSR dimensions are inconsistent.");
  double previous = 0.0;
  for (int row = 0; row <= graph.rows; ++row) {
    const double raw = graph.wide_offsets ?
      graph.wide_offsets[row] : graph.narrow_offsets[row];
    if (!std::isfinite(raw) || raw < 0.0 ||
        raw != std::floor(raw) ||
        raw > 9007199254740992.0 ||
        raw < previous || raw > static_cast<double>(graph.edges)) {
      Rcpp::stop("Resident UMAP CSR offsets are not monotone.");
    }
    previous = raw;
  }
  if (graph.offset(0) != 0 ||
      graph.offset(graph.rows) != graph.edges) {
    Rcpp::stop("Resident UMAP CSR dimensions are inconsistent.");
  }
  float maximum = 0.0f;
  for (std::uint64_t edge = 0; edge < graph.edges; ++edge) {
    const float weight = graph.weight(edge);
    if (graph.neighbors[edge] < 0 ||
        graph.neighbors[edge] >= graph.rows ||
        !std::isfinite(weight) || weight < 0.0f) {
      Rcpp::stop("Resident UMAP graph has invalid edges.");
    }
    maximum = std::max(maximum, weight);
  }
  if (!(maximum > 0.0f)) {
    Rcpp::stop("Resident UMAP graph has no positive edges.");
  }
  return maximum;
}

#endif
