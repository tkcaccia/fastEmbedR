#ifndef FASTEMBEDR_NATIVE_KNN_COMMON_H
#define FASTEMBEDR_NATIVE_KNN_COMMON_H

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace fastembedr {

enum class KnnMetric {
  Euclidean,
  Cosine,
  Correlation
};

inline KnnMetric parse_knn_metric(const std::string& metric) {
  if (metric == "euclidean") return KnnMetric::Euclidean;
  if (metric == "cosine") return KnnMetric::Cosine;
  if (metric == "correlation") return KnnMetric::Correlation;
  Rcpp::stop("Unsupported native KNN metric `%s`.", metric.c_str());
}

inline bool is_float32_matrix(SEXP data) {
  if (!Rf_isS4(data)) return false;
  Rcpp::S4 object(data);
  return object.is("float32");
}

inline float int_bits_to_float(int bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits), "float32 payload must use 32-bit storage");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct FloatMatrix {
  std::vector<float> values;
  int nrow = 0;
  int ncol = 0;
  bool input_float32 = false;
};

inline void normalize_rows(FloatMatrix& matrix, bool center) {
  for (int i = 0; i < matrix.nrow; ++i) {
    float* row = matrix.values.data() + static_cast<std::size_t>(i) * matrix.ncol;
    double mean = 0.0;
    if (center) {
      for (int j = 0; j < matrix.ncol; ++j) mean += row[j];
      mean /= std::max(1, matrix.ncol);
    }
    double squared_norm = 0.0;
    for (int j = 0; j < matrix.ncol; ++j) {
      row[j] = static_cast<float>(row[j] - mean);
      squared_norm += static_cast<double>(row[j]) * row[j];
    }
    if (squared_norm <= 0.0) continue;
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (int j = 0; j < matrix.ncol; ++j) row[j] *= inverse_norm;
  }
}

inline FloatMatrix matrix_to_row_major_float(SEXP data, KnnMetric metric) {
  FloatMatrix result;
  result.input_float32 = is_float32_matrix(data);
  if (result.input_float32) {
    Rcpp::S4 object(data);
    SEXP payload_sexp = object.slot("Data");
    if (TYPEOF(payload_sexp) != INTSXP || !Rf_isMatrix(payload_sexp)) {
      Rcpp::stop("Invalid float::float32 matrix payload.");
    }
    Rcpp::IntegerMatrix payload(payload_sexp);
    result.nrow = payload.nrow();
    result.ncol = payload.ncol();
    result.values.resize(static_cast<std::size_t>(result.nrow) * result.ncol);
    const int* source = INTEGER(payload);
    for (int j = 0; j < result.ncol; ++j) {
      for (int i = 0; i < result.nrow; ++i) {
        result.values[static_cast<std::size_t>(i) * result.ncol + j] =
          int_bits_to_float(source[i + static_cast<std::size_t>(j) * result.nrow]);
      }
    }
  } else {
    SEXP dims = Rf_getAttrib(data, R_DimSymbol);
    if (TYPEOF(dims) != INTSXP || Rf_length(dims) != 2) {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
    result.nrow = INTEGER(dims)[0];
    result.ncol = INTEGER(dims)[1];
    result.values.resize(static_cast<std::size_t>(result.nrow) * result.ncol);
    if (TYPEOF(data) == INTSXP) {
      const int* source = INTEGER(data);
      for (int j = 0; j < result.ncol; ++j) for (int i = 0; i < result.nrow; ++i) {
        result.values[static_cast<std::size_t>(i) * result.ncol + j] =
          static_cast<float>(source[i + static_cast<std::size_t>(j) * result.nrow]);
      }
    } else if (TYPEOF(data) == REALSXP) {
      const double* source = REAL(data);
      for (int j = 0; j < result.ncol; ++j) for (int i = 0; i < result.nrow; ++i) {
        result.values[static_cast<std::size_t>(i) * result.ncol + j] =
          static_cast<float>(source[i + static_cast<std::size_t>(j) * result.nrow]);
      }
    } else {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
  }
  if (metric == KnnMetric::Cosine) normalize_rows(result, false);
  if (metric == KnnMetric::Correlation) normalize_rows(result, true);
  return result;
}

inline float output_distance(float internal_distance, KnnMetric metric) {
  if (metric == KnnMetric::Euclidean) {
    return std::sqrt(std::max(0.0f, internal_distance));
  }
  if (metric == KnnMetric::Cosine || metric == KnnMetric::Correlation) {
    return 0.5f * std::max(0.0f, internal_distance);
  }
  Rcpp::stop("Unsupported native KNN metric.");
}

} // namespace fastembedr

#endif
