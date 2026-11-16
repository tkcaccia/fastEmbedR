#ifndef FASTEMBEDR_FLOAT_MATRIX_COMMON_H
#define FASTEMBEDR_FLOAT_MATRIX_COMMON_H

#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace fastembedr {

inline bool is_float32_matrix(SEXP data) {
  if (!Rf_isS4(data)) return false;
  Rcpp::S4 object(data);
  return object.is("float32");
}

inline float int_bits_to_float(int bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits),
                "float32 payload must use 32-bit storage");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct FloatMatrix {
  std::vector<float> values;
  int nrow = 0;
  int ncol = 0;
  bool input_float32 = false;
};

template <typename Source, typename Convert>
inline void transpose_to_float(const Source* source,
                               FloatMatrix& result,
                               Convert convert) {
  constexpr int tile = 32;
  for (int row = 0; row < result.nrow; row += tile) {
    const int row_end = std::min(row + tile, result.nrow);
    for (int col = 0; col < result.ncol; col += tile) {
      const int col_end = std::min(col + tile, result.ncol);
      for (int j = col; j < col_end; ++j) {
        for (int i = row; i < row_end; ++i) {
          result.values[static_cast<std::size_t>(i) * result.ncol + j] =
            convert(source[i + static_cast<std::size_t>(j) * result.nrow]);
        }
      }
    }
  }
}

inline FloatMatrix matrix_to_row_major_float(SEXP data) {
  FloatMatrix result;
  if (TYPEOF(data) == EXTPTRSXP &&
      R_ExternalPtrTag(data) ==
        Rf_install("fastEmbedR_massive_float_buffer")) {
    Rcpp::XPtr<FloatMatrix> buffer(data);
    if (buffer.get() == nullptr || buffer->nrow < 1 ||
        buffer->values.empty()) {
      Rcpp::stop("Massive float32 reference buffer was already consumed.");
    }
    result = std::move(*buffer);
    buffer->nrow = 0;
    buffer->ncol = 0;
    return result;
  }
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
    transpose_to_float(source, result, int_bits_to_float);
  } else {
    SEXP dims = Rf_getAttrib(data, R_DimSymbol);
    if (TYPEOF(dims) != INTSXP || Rf_length(dims) != 2) {
      Rcpp::stop(
        "`data` must be an integer, numeric, or float::float32 matrix.");
    }
    result.nrow = INTEGER(dims)[0];
    result.ncol = INTEGER(dims)[1];
    result.values.resize(static_cast<std::size_t>(result.nrow) * result.ncol);
    if (TYPEOF(data) == INTSXP) {
      const int* source = INTEGER(data);
      transpose_to_float(source, result, [](int value) {
        return static_cast<float>(value);
      });
    } else if (TYPEOF(data) == REALSXP) {
      const double* source = REAL(data);
      transpose_to_float(source, result, [](double value) {
        return static_cast<float>(value);
      });
    } else {
      Rcpp::stop(
        "`data` must be an integer, numeric, or float::float32 matrix.");
    }
  }
  return result;
}

} // namespace fastembedr

#endif
