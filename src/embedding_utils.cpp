#include <Rcpp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#define COMPLEX FASTEMBEDR_ACCELERATE_COMPLEX
#include <Accelerate/Accelerate.h>
#undef COMPLEX
#endif

#include "native_knn_common.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

bool is_float32_s4(SEXP x) {
  if (!Rf_isS4(x)) return false;
  Rcpp::S4 obj(x);
  return obj.is("float32");
}

IntegerMatrix float32_data_slot(SEXP x) {
  Rcpp::S4 obj(x);
  SEXP data = obj.slot("Data");
  if (TYPEOF(data) != INTSXP || Rf_isNull(Rf_getAttrib(data, R_DimSymbol))) {
    Rcpp::stop("float32 distance object has an invalid payload");
  }
  return IntegerMatrix(data);
}

float int_bits_to_float(const int value) {
  float out;
  std::memcpy(&out, &value, sizeof(float));
  return out;
}

int float_to_int_bits(const float value) {
  int out;
  std::memcpy(&out, &value, sizeof(float));
  return out;
}

double layout_distance_sq(const NumericMatrix& layout, const int i, const int j) {
  if (layout.ncol() == 2) {
    const double dx = layout(i, 0) - layout(j, 0);
    const double dy = layout(i, 1) - layout(j, 1);
    return dx * dx + dy * dy;
  }
  double d2 = 0.0;
  for (int c = 0; c < layout.ncol(); ++c) {
    const double diff = layout(i, c) - layout(j, c);
    d2 += diff * diff;
  }
  return d2;
}

int find_high_rank(const IntegerMatrix& indices,
                   const int query,
                   const int candidate,
                   const int high_rank_limit) {
  for (int r = 0; r < high_rank_limit; ++r) {
    if (indices(query, r) - 1 == candidate) return r + 1;
  }
  return high_rank_limit + 1;
}

template <typename T>
double median_inplace(std::vector<T>& values) {
  if (values.empty()) return R_NaReal;
  const std::size_t mid = values.size() / 2u;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double upper = static_cast<double>(values[mid]);
  if (values.size() % 2u == 1u) return upper;
  std::nth_element(values.begin(), values.begin() + mid - 1u, values.begin() + mid);
  return 0.5 * (static_cast<double>(values[mid - 1u]) + upper);
}

bool cholesky_decompose_inplace(std::vector<double>& a, const int n) {
  for (int j = 0; j < n; ++j) {
    double diag = a[static_cast<std::size_t>(j) * n + j];
    for (int k = 0; k < j; ++k) {
      const double v = a[static_cast<std::size_t>(j) * n + k];
      diag -= v * v;
    }
    if (!std::isfinite(diag) || diag <= 1e-14) return false;
    a[static_cast<std::size_t>(j) * n + j] = std::sqrt(diag);
    const double inv_diag = 1.0 / a[static_cast<std::size_t>(j) * n + j];
    for (int i = j + 1; i < n; ++i) {
      double value = a[static_cast<std::size_t>(i) * n + j];
      for (int k = 0; k < j; ++k) {
        value -= a[static_cast<std::size_t>(i) * n + k] *
          a[static_cast<std::size_t>(j) * n + k];
      }
      a[static_cast<std::size_t>(i) * n + j] = value * inv_diag;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      a[static_cast<std::size_t>(i) * n + j] = 0.0;
    }
  }
  return true;
}

bool cholesky_solve_inplace(const std::vector<double>& chol,
                            std::vector<double>& b,
                            const int n,
                            const int nrhs) {
  for (int rhs = 0; rhs < nrhs; ++rhs) {
    for (int i = 0; i < n; ++i) {
      double value = b[static_cast<std::size_t>(i) * nrhs + rhs];
      for (int k = 0; k < i; ++k) {
        value -= chol[static_cast<std::size_t>(i) * n + k] *
          b[static_cast<std::size_t>(k) * nrhs + rhs];
      }
      const double diag = chol[static_cast<std::size_t>(i) * n + i];
      if (!std::isfinite(diag) || diag == 0.0) return false;
      b[static_cast<std::size_t>(i) * nrhs + rhs] = value / diag;
    }
    for (int i = n - 1; i >= 0; --i) {
      double value = b[static_cast<std::size_t>(i) * nrhs + rhs];
      for (int k = i + 1; k < n; ++k) {
        value -= chol[static_cast<std::size_t>(k) * n + i] *
          b[static_cast<std::size_t>(k) * nrhs + rhs];
      }
      const double diag = chol[static_cast<std::size_t>(i) * n + i];
      if (!std::isfinite(diag) || diag == 0.0) return false;
      b[static_cast<std::size_t>(i) * nrhs + rhs] = value / diag;
    }
  }
  return true;
}

int resolve_projection_threads(int n_threads, const int n) {
  if (n <= 1) return 1;
  if (n_threads < 1) {
    const unsigned int hw = std::thread::hardware_concurrency();
    n_threads = hw == 0 ? 1 : static_cast<int>(hw);
  }
  const int cap = n < 256 ? 1 : n < 2048 ? 2 : n_threads;
  return std::max(1, std::min(std::min(n_threads, cap), n));
}

template <typename Function>
void parallel_for_rows(const int n, const int n_threads, Function fn) {
  const int threads = resolve_projection_threads(n_threads, n);
  if (threads <= 1 || n < 2) {
    fn(0, n, 0);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    const int begin = (n * t) / threads;
    const int end = (n * (t + 1)) / threads;
    workers.emplace_back([&, begin, end, t]() {
      fn(begin, end, t);
    });
  }
  for (auto& worker : workers) worker.join();
}

struct PcaFloatMatrix {
  std::vector<float> values;
  int nrow = 0;
  int ncol = 0;
  bool input_float32 = false;
};

PcaFloatMatrix pca_center_scale_float(SEXP data,
                                      const bool center,
                                      const bool scale,
                                      const int n_threads,
                                      NumericVector& center_values,
                                      NumericVector& scale_values) {
  PcaFloatMatrix result;
  result.input_float32 = fastembedr::is_float32_matrix(data);

  const int* float_source = nullptr;
  const double* double_source = nullptr;
  const int* integer_source = nullptr;
  if (result.input_float32) {
    Rcpp::S4 object(data);
    SEXP payload_sexp = object.slot("Data");
    if (TYPEOF(payload_sexp) != INTSXP || !Rf_isMatrix(payload_sexp)) {
      Rcpp::stop("Invalid float::float32 PCA matrix payload.");
    }
    IntegerMatrix payload(payload_sexp);
    result.nrow = payload.nrow();
    result.ncol = payload.ncol();
    float_source = INTEGER(payload);
  } else {
    SEXP dims = Rf_getAttrib(data, R_DimSymbol);
    if (TYPEOF(dims) != INTSXP || Rf_length(dims) != 2) {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
    result.nrow = INTEGER(dims)[0];
    result.ncol = INTEGER(dims)[1];
    if (TYPEOF(data) == REALSXP) {
      double_source = REAL(data);
    } else if (TYPEOF(data) == INTSXP) {
      integer_source = INTEGER(data);
    } else {
      Rcpp::stop("`data` must be an integer, numeric, or float::float32 matrix.");
    }
  }
  if (result.nrow < 2 || result.ncol < 1) {
    Rcpp::stop("`data` must have at least two rows and one column.");
  }

  const std::size_t size =
    static_cast<std::size_t>(result.nrow) * result.ncol;
  result.values.resize(size);
  center_values = NumericVector(result.ncol);
  scale_values = NumericVector(result.ncol, 1.0);
  double* center_ptr = REAL(center_values);
  double* scale_ptr = REAL(scale_values);
  std::atomic<bool> finite(true);
  const double scale_denom = static_cast<double>(std::max(1, result.nrow - 1));

  auto input_value = [&](const std::size_t index) -> double {
    if (float_source != nullptr) {
      return static_cast<double>(fastembedr::int_bits_to_float(float_source[index]));
    }
    if (double_source != nullptr) return double_source[index];
    return static_cast<double>(integer_source[index]);
  };

  parallel_for_rows(result.ncol, n_threads, [&](const int begin, const int end, const int) {
    for (int col = begin; col < end; ++col) {
      const std::size_t offset = static_cast<std::size_t>(col) * result.nrow;
      double sum = 0.0;
      for (int row = 0; row < result.nrow; ++row) {
        const double value = input_value(offset + row);
        if (!std::isfinite(value)) {
          finite.store(false, std::memory_order_relaxed);
          break;
        }
        sum += value;
      }
      if (!finite.load(std::memory_order_relaxed)) continue;

      const double mean = center ? sum / static_cast<double>(result.nrow) : 0.0;
      center_ptr[col] = mean;
      double sd = 1.0;
      if (scale) {
        double sum_squares = 0.0;
        for (int row = 0; row < result.nrow; ++row) {
          const double value = input_value(offset + row) - mean;
          sum_squares += value * value;
        }
        sd = std::sqrt(sum_squares / scale_denom);
        if (!std::isfinite(sd) || sd == 0.0) sd = 1.0;
      }
      scale_ptr[col] = sd;
      const double inverse_sd = 1.0 / sd;
      for (int row = 0; row < result.nrow; ++row) {
        result.values[offset + row] = static_cast<float>(
          (input_value(offset + row) - mean) * inverse_sd
        );
      }
    }
  });
  if (!finite.load(std::memory_order_relaxed)) {
    Rcpp::stop("`data` must contain only finite values.");
  }
  return result;
}

void pca_gemm_nn(const float* left,
                 const float* right,
                 float* output,
                 const int nrow,
                 const int shared,
                 const int ncol,
                 const int n_threads) {
#ifdef __APPLE__
  cblas_sgemm(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    nrow,
    ncol,
    shared,
    1.0f,
    left,
    nrow,
    right,
    shared,
    0.0f,
    output,
    nrow
  );
#else
  parallel_for_rows(ncol, n_threads, [&](const int begin, const int end, const int) {
    for (int col = begin; col < end; ++col) {
      float* destination = output + static_cast<std::size_t>(col) * nrow;
      std::fill(destination, destination + nrow, 0.0f);
      for (int inner = 0; inner < shared; ++inner) {
        const float coefficient = right[inner + static_cast<std::size_t>(col) * shared];
        const float* source = left + static_cast<std::size_t>(inner) * nrow;
        for (int row = 0; row < nrow; ++row) {
          destination[row] += source[row] * coefficient;
        }
      }
    }
  });
#endif
}

void pca_gemm_tn(const float* left,
                 const float* right,
                 float* output,
                 const int shared,
                 const int left_cols,
                 const int right_cols,
                 const int n_threads) {
#ifdef __APPLE__
  cblas_sgemm(
    CblasColMajor,
    CblasTrans,
    CblasNoTrans,
    left_cols,
    right_cols,
    shared,
    1.0f,
    left,
    shared,
    right,
    shared,
    0.0f,
    output,
    left_cols
  );
#else
  const int total = left_cols * right_cols;
  parallel_for_rows(total, n_threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      const int left_col = index % left_cols;
      const int right_col = index / left_cols;
      const float* left_data = left + static_cast<std::size_t>(left_col) * shared;
      const float* right_data = right + static_cast<std::size_t>(right_col) * shared;
      double value = 0.0;
      for (int row = 0; row < shared; ++row) {
        value += static_cast<double>(left_data[row]) * right_data[row];
      }
      output[left_col + static_cast<std::size_t>(right_col) * left_cols] =
        static_cast<float>(value);
    }
  });
#endif
}

NumericMatrix pca_float_to_numeric(const std::vector<float>& values,
                                   const int nrow,
                                   const int ncol) {
  NumericMatrix output(nrow, ncol);
  double* destination = REAL(output);
  for (std::size_t i = 0; i < values.size(); ++i) {
    destination[i] = static_cast<double>(values[i]);
  }
  return output;
}

std::vector<float> pca_numeric_to_float(const NumericMatrix& values) {
  const std::size_t size =
    static_cast<std::size_t>(values.nrow()) * values.ncol();
  std::vector<float> output(size);
  const double* source = REAL(values);
  for (std::size_t i = 0; i < size; ++i) {
    output[i] = static_cast<float>(source[i]);
  }
  return output;
}

SEXP pca_output_matrix(const std::vector<float>& values,
                       const int nrow,
                       const int ncol,
                       const bool float32_output) {
  if (!float32_output) return pca_float_to_numeric(values, nrow, ncol);
  IntegerMatrix payload(nrow, ncol);
  int* destination = INTEGER(payload);
  for (std::size_t i = 0; i < values.size(); ++i) {
    destination[i] = float_to_int_bits(values[i]);
  }
  Rcpp::S4 output("float32");
  output.slot("Data") = payload;
  return output;
}

} // namespace

// [[Rcpp::export]]
List pca_rsvd_cpu_cpp(SEXP data,
                      int n_components,
                      bool center,
                      bool scale,
                      NumericMatrix omega,
                      int power,
                      int n_threads = 1) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  n_threads = std::max(1, n_threads);
  if (n_components < 1) {
    Rcpp::stop("`n_components` must be positive.");
  }

  NumericVector center_values;
  NumericVector scale_values;
  PcaFloatMatrix x = pca_center_scale_float(
    data,
    center,
    scale,
    n_threads,
    center_values,
    scale_values
  );
  const int max_rank = std::min(x.nrow, x.ncol);
  n_threads = resolve_projection_threads(n_threads, x.ncol);
  const int rank = std::min(n_components, max_rank);
  if (rank < 1) Rcpp::stop("PCA input has no usable rank.");
  if (omega.nrow() != x.ncol || omega.ncol() < rank ||
      omega.ncol() > max_rank) {
    Rcpp::stop("The RSVD sketch matrix has incompatible dimensions.");
  }
  const int sketch_rank = omega.ncol();
  const auto converted = Clock::now();

  std::vector<float> omega_float = pca_numeric_to_float(omega);
  std::vector<float> y(
    static_cast<std::size_t>(x.nrow) * sketch_rank
  );
  pca_gemm_nn(
    x.values.data(),
    omega_float.data(),
    y.data(),
    x.nrow,
    x.ncol,
    sketch_rank,
    n_threads
  );

  power = std::max(0, power);
  Rcpp::Environment base = Rcpp::Environment::base_env();
  Rcpp::Function qr_function = base["qr"];
  Rcpp::Function qr_q_function = base["qr.Q"];
  auto qr_basis = [&](const std::vector<float>& values,
                      const int nrow,
                      const int ncol) -> NumericMatrix {
    NumericMatrix matrix = pca_float_to_numeric(values, nrow, ncol);
    List decomposition = qr_function(
      matrix,
      Rcpp::Named("LAPACK") = true
    );
    return Rcpp::as<NumericMatrix>(
      qr_q_function(
        decomposition,
        Rcpp::Named("complete") = false
      )
    );
  };

  for (int iteration = 0; iteration < power; ++iteration) {
    std::vector<float> z(
      static_cast<std::size_t>(x.ncol) * sketch_rank
    );
    pca_gemm_tn(
      x.values.data(),
      y.data(),
      z.data(),
      x.nrow,
      x.ncol,
      sketch_rank,
      n_threads
    );
    if (power > 1) {
      NumericMatrix qz = qr_basis(z, x.ncol, sketch_rank);
      z = pca_numeric_to_float(qz);
    }
    pca_gemm_nn(
      x.values.data(),
      z.data(),
      y.data(),
      x.nrow,
      x.ncol,
      sketch_rank,
      n_threads
    );
    Rcpp::checkUserInterrupt();
  }
  const auto sketched = Clock::now();

  NumericMatrix q = qr_basis(y, x.nrow, sketch_rank);
  std::vector<float> q_float = pca_numeric_to_float(q);
  std::vector<float> b(
    static_cast<std::size_t>(sketch_rank) * x.ncol
  );
  pca_gemm_tn(
    q_float.data(),
    x.values.data(),
    b.data(),
    x.nrow,
    sketch_rank,
    x.ncol,
    n_threads
  );
  const auto projected = Clock::now();

  Rcpp::Function svd_function = base["svd"];
  List small = svd_function(
    pca_float_to_numeric(b, sketch_rank, x.ncol),
    Rcpp::Named("nu") = rank,
    Rcpp::Named("nv") = rank
  );
  NumericVector singular_values = small["d"];
  NumericMatrix left_vectors = small["u"];
  NumericMatrix right_vectors = small["v"];
  const int usable = std::min(
    rank,
    std::min(
      static_cast<int>(singular_values.size()),
      std::min(left_vectors.ncol(), right_vectors.ncol())
    )
  );
  if (usable < 1) {
    Rcpp::stop("RSVD PCA produced no usable singular vectors.");
  }

  std::vector<float> score_coefficients(
    static_cast<std::size_t>(sketch_rank) * usable
  );
  for (int component = 0; component < usable; ++component) {
    const float singular = static_cast<float>(singular_values[component]);
    for (int row = 0; row < sketch_rank; ++row) {
      score_coefficients[row + static_cast<std::size_t>(component) * sketch_rank] =
        static_cast<float>(left_vectors(row, component)) * singular;
    }
  }
  std::vector<float> scores(
    static_cast<std::size_t>(x.nrow) * usable
  );
  pca_gemm_nn(
    q_float.data(),
    score_coefficients.data(),
    scores.data(),
    x.nrow,
    sketch_rank,
    usable,
    n_threads
  );

  std::vector<float> loadings(
    static_cast<std::size_t>(x.ncol) * usable
  );
  for (int component = 0; component < usable; ++component) {
    for (int row = 0; row < x.ncol; ++row) {
      loadings[row + static_cast<std::size_t>(component) * x.ncol] =
        static_cast<float>(right_vectors(row, component));
    }
  }
  for (int component = 0; component < usable; ++component) {
    int pivot = 0;
    float maximum = 0.0f;
    for (int row = 0; row < x.ncol; ++row) {
      const float magnitude = std::abs(
        loadings[row + static_cast<std::size_t>(component) * x.ncol]
      );
      if (magnitude > maximum) {
        maximum = magnitude;
        pivot = row;
      }
    }
    if (loadings[pivot + static_cast<std::size_t>(component) * x.ncol] < 0.0f) {
      for (int row = 0; row < x.ncol; ++row) {
        loadings[row + static_cast<std::size_t>(component) * x.ncol] *= -1.0f;
      }
      for (int row = 0; row < x.nrow; ++row) {
        scores[row + static_cast<std::size_t>(component) * x.nrow] *= -1.0f;
      }
    }
  }
  const auto completed = Clock::now();

  NumericVector retained_singular_values(usable);
  for (int component = 0; component < usable; ++component) {
    retained_singular_values[component] = singular_values[component];
  }
  const bool return_float32 = x.input_float32;
  return List::create(
    Rcpp::Named("scores") = pca_output_matrix(
      scores, x.nrow, usable, return_float32
    ),
    Rcpp::Named("loadings") = pca_output_matrix(
      loadings, x.ncol, usable, return_float32
    ),
    Rcpp::Named("singular_values") = retained_singular_values,
    Rcpp::Named("center") = center_values,
    Rcpp::Named("scale") = scale_values,
    Rcpp::Named("precision") = "float32",
    Rcpp::Named("input_float32") = x.input_float32,
    Rcpp::Named("oversample") = sketch_rank - rank,
    Rcpp::Named("power") = power,
    Rcpp::Named("n_threads") = n_threads,
    Rcpp::Named("timing") = List::create(
      Rcpp::Named("prepare") =
        std::chrono::duration<double>(converted - started).count(),
      Rcpp::Named("sketch") =
        std::chrono::duration<double>(sketched - converted).count(),
      Rcpp::Named("project") =
        std::chrono::duration<double>(projected - sketched).count(),
      Rcpp::Named("small_svd_and_scores") =
        std::chrono::duration<double>(completed - projected).count(),
      Rcpp::Named("total") =
        std::chrono::duration<double>(completed - started).count()
    )
  );
}

// [[Rcpp::export]]
bool float32_all_finite_cpp(SEXP data, int n_threads = 0) {
  if (!is_float32_s4(data)) {
    Rcpp::stop("`data` must be a float::float32 object");
  }
  Rcpp::S4 object(data);
  SEXP payload = object.slot("Data");
  if (TYPEOF(payload) != INTSXP) {
    Rcpp::stop("float32 data has an invalid payload");
  }
  const R_xlen_t length = Rf_xlength(payload);
  const int* source = INTEGER(payload);
  std::atomic<bool> finite(true);
  const int threads = std::min(
    4,
    resolve_projection_threads(n_threads, static_cast<int>(std::min<R_xlen_t>(
      length, static_cast<R_xlen_t>(std::numeric_limits<int>::max())
    )))
  );
  if (threads <= 1 || length < 1000000) {
    for (R_xlen_t i = 0; i < length; ++i) {
      if (!std::isfinite(int_bits_to_float(source[i]))) return false;
    }
    return true;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int thread = 0; thread < threads; ++thread) {
    const R_xlen_t begin = length * thread / threads;
    const R_xlen_t end = length * (thread + 1) / threads;
    workers.emplace_back([&, begin, end]() {
      for (R_xlen_t i = begin; i < end && finite.load(std::memory_order_relaxed); ++i) {
        if (!std::isfinite(int_bits_to_float(source[i]))) {
          finite.store(false, std::memory_order_relaxed);
          break;
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
  return finite.load(std::memory_order_relaxed);
}

// [[Rcpp::export]]
List standardize_cpu_cpp(NumericMatrix data) {
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 1 || p < 1) Rcpp::stop("data must have at least one row and one column");

  NumericMatrix out(n, p);
  NumericVector center(p);
  NumericVector scale(p);
  const double denom = static_cast<double>(std::max(1, n - 1));

  for (int col = 0; col < p; ++col) {
    double sum = 0.0;
    for (int row = 0; row < n; ++row) {
      const double value = data(row, col);
      if (!std::isfinite(value)) Rcpp::stop("data must contain only finite values");
      sum += value;
    }
    const double mean = sum / static_cast<double>(n);
    center[col] = mean;

    double ss = 0.0;
    for (int row = 0; row < n; ++row) {
      const double centered = data(row, col) - mean;
      out(row, col) = centered;
      ss += centered * centered;
    }
    double sd = std::sqrt(ss / denom);
    if (!std::isfinite(sd) || sd == 0.0) sd = 1.0;
    scale[col] = sd;
    const double inv_sd = 1.0 / sd;
    for (int row = 0; row < n; ++row) {
      out(row, col) *= inv_sd;
    }
  }

  return List::create(
    Rcpp::Named("data") = out,
    Rcpp::Named("center") = center,
    Rcpp::Named("scale") = scale
  );
}

// [[Rcpp::export]]
List standardize_float32_cpp(SEXP data, int n_threads = 0) {
  if (!is_float32_s4(data)) {
    Rcpp::stop("`data` must be a float::float32 matrix");
  }
  IntegerMatrix source = float32_data_slot(data);
  const int n = source.nrow();
  const int p = source.ncol();
  if (n < 1 || p < 1) Rcpp::stop("data must have at least one row and one column");

  IntegerMatrix payload(n, p);
  NumericVector center(p);
  NumericVector scale(p);
  const int* source_ptr = INTEGER(source);
  int* destination_ptr = INTEGER(payload);
  double* center_ptr = REAL(center);
  double* scale_ptr = REAL(scale);
  const double denom = static_cast<double>(std::max(1, n - 1));
  const int threads = resolve_projection_threads(n_threads, p);
  std::atomic<bool> finite(true);
  parallel_for_rows(p, threads, [&](const int begin, const int end, const int) {
    for (int col = begin; col < end; ++col) {
      const std::size_t base = static_cast<std::size_t>(col) * n;
      double sum = 0.0;
      bool column_finite = true;
      for (int row = 0; row < n; ++row) {
        const float value = int_bits_to_float(source_ptr[base + row]);
        if (!std::isfinite(value)) {
          column_finite = false;
          finite.store(false, std::memory_order_relaxed);
          break;
        }
        sum += static_cast<double>(value);
      }
      if (!column_finite) continue;
      const double mean = sum / static_cast<double>(n);
      center_ptr[col] = mean;
      double ss = 0.0;
      for (int row = 0; row < n; ++row) {
        const double centered =
          static_cast<double>(int_bits_to_float(source_ptr[base + row])) - mean;
        ss += centered * centered;
      }
      double sd = std::sqrt(ss / denom);
      if (!std::isfinite(sd) || sd == 0.0) sd = 1.0;
      scale_ptr[col] = sd;
      const double inv_sd = 1.0 / sd;
      for (int row = 0; row < n; ++row) {
        const double centered =
          static_cast<double>(int_bits_to_float(source_ptr[base + row])) - mean;
        destination_ptr[base + row] = float_to_int_bits(
          static_cast<float>(centered * inv_sd)
        );
      }
    }
  });
  if (!finite.load(std::memory_order_relaxed)) {
    Rcpp::stop("data must contain only finite values");
  }

  Rcpp::S4 out("float32");
  out.slot("Data") = payload;
  return List::create(
    Rcpp::Named("data") = out,
    Rcpp::Named("center") = center,
    Rcpp::Named("scale") = scale
  );
}

// [[Rcpp::export]]
List split_float32_rows_cpp(SEXP data,
                            IntegerVector landmark_rows,
                            IntegerVector query_rows,
                            int n_threads = 0) {
  if (!is_float32_s4(data)) {
    Rcpp::stop("`data` must be a float::float32 matrix");
  }
  IntegerMatrix source = float32_data_slot(data);
  const int n = source.nrow();
  const int p = source.ncol();
  const int n_landmarks = landmark_rows.size();
  const int n_query = query_rows.size();
  if (n_landmarks < 1 || n_query < 1) {
    Rcpp::stop("landmark and query row sets must both be non-empty");
  }

  std::vector<int> landmark_zero(static_cast<std::size_t>(n_landmarks));
  std::vector<int> query_zero(static_cast<std::size_t>(n_query));
  std::vector<unsigned char> seen(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n_landmarks; ++i) {
    const int row = landmark_rows[i] - 1;
    if (row < 0 || row >= n || seen[static_cast<std::size_t>(row)] != 0) {
      Rcpp::stop("`landmark_rows` must contain unique, valid row indices");
    }
    landmark_zero[static_cast<std::size_t>(i)] = row;
    seen[static_cast<std::size_t>(row)] = 1;
  }
  for (int i = 0; i < n_query; ++i) {
    const int row = query_rows[i] - 1;
    if (row < 0 || row >= n || seen[static_cast<std::size_t>(row)] != 0) {
      Rcpp::stop("`query_rows` must be valid, unique, and disjoint from `landmark_rows`");
    }
    query_zero[static_cast<std::size_t>(i)] = row;
    seen[static_cast<std::size_t>(row)] = 1;
  }
  if (n_landmarks + n_query != n) {
    Rcpp::stop("landmark and query rows must partition every input row");
  }

  IntegerMatrix landmark_payload(n_landmarks, p);
  IntegerMatrix query_payload(n_query, p);
  const int* source_ptr = INTEGER(source);
  int* landmark_ptr = INTEGER(landmark_payload);
  int* query_ptr = INTEGER(query_payload);
  const int threads = resolve_projection_threads(n_threads, p);
  parallel_for_rows(p, threads, [&](const int begin, const int end, const int) {
    for (int col = begin; col < end; ++col) {
      const std::size_t source_base = static_cast<std::size_t>(col) * n;
      const std::size_t landmark_base = static_cast<std::size_t>(col) * n_landmarks;
      const std::size_t query_base = static_cast<std::size_t>(col) * n_query;
      for (int row = 0; row < n_landmarks; ++row) {
        landmark_ptr[landmark_base + row] = source_ptr[
          source_base + landmark_zero[static_cast<std::size_t>(row)]
        ];
      }
      for (int row = 0; row < n_query; ++row) {
        query_ptr[query_base + row] = source_ptr[
          source_base + query_zero[static_cast<std::size_t>(row)]
        ];
      }
    }
  });

  Rcpp::S4 landmarks("float32");
  landmarks.slot("Data") = landmark_payload;
  Rcpp::S4 query("float32");
  query.slot("Data") = query_payload;
  return List::create(
    Rcpp::Named("landmarks") = landmarks,
    Rcpp::Named("query") = query
  );
}

// [[Rcpp::export]]
NumericMatrix landmark_projection_float32_cpp(SEXP data,
                                               NumericMatrix directions,
                                               int n_direct = 4,
                                               int n_threads = 0) {
  if (!is_float32_s4(data)) {
    Rcpp::stop("`data` must be a float::float32 matrix");
  }
  IntegerMatrix source = float32_data_slot(data);
  const int n = source.nrow();
  const int p = source.ncol();
  const int n_random = directions.ncol();
  n_direct = std::max(0, std::min(n_direct, p));
  if (directions.nrow() != p || n_random < 1) {
    Rcpp::stop("`directions` must have ncol(data) rows and at least one column");
  }

  NumericMatrix projected(n, n_direct + n_random);
  const int* source_ptr = INTEGER(source);
  const double* direction_ptr = REAL(directions);
  double* projected_ptr = REAL(projected);
  const int threads = resolve_projection_threads(n_threads, n);
  parallel_for_rows(n, threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      for (int col = 0; col < n_direct; ++col) {
        projected_ptr[static_cast<std::size_t>(col) * n + row] =
          static_cast<double>(int_bits_to_float(
            source_ptr[static_cast<std::size_t>(col) * n + row]
          ));
      }
      for (int component = 0; component < n_random; ++component) {
        double value = 0.0;
        const std::size_t direction_base = static_cast<std::size_t>(component) * p;
        for (int feature = 0; feature < p; ++feature) {
          value += static_cast<double>(int_bits_to_float(
            source_ptr[static_cast<std::size_t>(feature) * n + row]
          )) * direction_ptr[direction_base + feature];
        }
        projected_ptr[
          static_cast<std::size_t>(n_direct + component) * n + row
        ] = value;
      }
    }
  });
  return projected;
}

// [[Rcpp::export]]
List strip_self_neighbors_cpp(IntegerMatrix indices, NumericMatrix distances) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (k < 1) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = 0,
      Rcpp::Named("materialized") = false
    );
  }

  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  const double tolerance = std::max(std::sqrt(std::numeric_limits<double>::epsilon()), 1e-12);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      const int idx = indices(row, col);
      const double dist = distances(row, col);
      if (idx == NA_INTEGER) Rcpp::stop("KNN indices must not contain NA values");
      if (!std::isfinite(dist) || dist < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative");
      }
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const bool one_based = min_idx >= 1 && max_idx <= n;
  const int offset = one_based ? 1 : 0;

  bool first_self = true;
  for (int row = 0; row < n; ++row) {
    const int expected = row + offset;
    if (indices(row, 0) != expected || distances(row, 0) > tolerance) {
      first_self = false;
      break;
    }
  }
  if (first_self) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = true,
      Rcpp::Named("col_start") = 1,
      Rcpp::Named("n_neighbors") = k - 1,
      Rcpp::Named("materialized") = false
    );
  }

  std::vector<int> self_pos(static_cast<std::size_t>(n), -1);
  bool all_rows_have_self = true;
  for (int row = 0; row < n; ++row) {
    const int expected = row + offset;
    for (int col = 0; col < k; ++col) {
      if (indices(row, col) == expected && distances(row, col) <= tolerance) {
        self_pos[static_cast<std::size_t>(row)] = col;
        break;
      }
    }
    if (self_pos[static_cast<std::size_t>(row)] < 0) {
      all_rows_have_self = false;
      break;
    }
  }

  if (!all_rows_have_self) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = k,
      Rcpp::Named("materialized") = false
    );
  }

  if (k == 1) {
    IntegerMatrix out_indices(n, 0);
    NumericMatrix out_distances(n, 0);
    return List::create(
      Rcpp::Named("indices") = out_indices,
      Rcpp::Named("distances") = out_distances,
      Rcpp::Named("has_self") = true,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = 0,
      Rcpp::Named("materialized") = true
    );
  }

  IntegerMatrix out_indices(n, k - 1);
  NumericMatrix out_distances(n, k - 1);
  for (int row = 0; row < n; ++row) {
    int out_col = 0;
    const int skip = self_pos[static_cast<std::size_t>(row)];
    for (int col = 0; col < k; ++col) {
      if (col == skip) continue;
      out_indices(row, out_col) = indices(row, col);
      out_distances(row, out_col) = distances(row, col);
      ++out_col;
    }
  }

  return List::create(
    Rcpp::Named("indices") = out_indices,
    Rcpp::Named("distances") = out_distances,
    Rcpp::Named("has_self") = true,
    Rcpp::Named("col_start") = 0,
    Rcpp::Named("n_neighbors") = k - 1,
    Rcpp::Named("materialized") = true
  );
}

// [[Rcpp::export]]
List strip_self_neighbors_float_cpp(IntegerMatrix indices, SEXP distances) {
  if (!is_float32_s4(distances)) {
    NumericMatrix numeric_distances(distances);
    return strip_self_neighbors_cpp(indices, numeric_distances);
  }
  IntegerMatrix payload = float32_data_slot(distances);
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (payload.nrow() != n || payload.ncol() != k) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (k < 1) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = 0,
      Rcpp::Named("materialized") = false,
      Rcpp::Named("distance_type") = "float32"
    );
  }

  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  const float tolerance = std::max(std::sqrt(std::numeric_limits<float>::epsilon()), 1.0e-6f);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      const int idx = indices(row, col);
      const float dist = int_bits_to_float(payload(row, col));
      if (idx == NA_INTEGER) Rcpp::stop("KNN indices must not contain NA values");
      if (!std::isfinite(dist) || dist < 0.0f) {
        Rcpp::stop("KNN distances must be finite and non-negative");
      }
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const bool one_based = min_idx >= 1 && max_idx <= n;
  const int offset = one_based ? 1 : 0;

  bool first_self = true;
  for (int row = 0; row < n; ++row) {
    const int expected = row + offset;
    if (indices(row, 0) != expected ||
        int_bits_to_float(payload(row, 0)) > tolerance) {
      first_self = false;
      break;
    }
  }
  if (first_self) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = true,
      Rcpp::Named("col_start") = 1,
      Rcpp::Named("n_neighbors") = k - 1,
      Rcpp::Named("materialized") = false,
      Rcpp::Named("distance_type") = "float32"
    );
  }

  bool has_self_elsewhere = false;
  for (int row = 0; row < n && !has_self_elsewhere; ++row) {
    const int expected = row + offset;
    for (int col = 1; col < k; ++col) {
      if (indices(row, col) == expected &&
          int_bits_to_float(payload(row, col)) <= tolerance) {
        has_self_elsewhere = true;
        break;
      }
    }
  }
  if (!has_self_elsewhere) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = k,
      Rcpp::Named("materialized") = false,
      Rcpp::Named("distance_type") = "float32"
    );
  }

  NumericMatrix numeric_distances(n, k);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      numeric_distances(row, col) = static_cast<double>(int_bits_to_float(payload(row, col)));
    }
  }
  List out = strip_self_neighbors_cpp(indices, numeric_distances);
  out["distance_type"] = "double_materialized_from_float32";
  return out;
}

// [[Rcpp::export]]
List validate_projection_knn_cpp(IntegerMatrix indices,
                                 NumericMatrix distances,
                                 int n_reference,
                                 int k) {
  const int n = indices.nrow();
  const int width = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != width) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (n < 1 || width < 1) {
    Rcpp::stop("KNN must have at least one row and one neighbor column");
  }
  if (n_reference < 1) Rcpp::stop("n_reference must be positive");
  if (k < 1 || k > width) Rcpp::stop("k must be in the available neighbor range");

  for (int col = 0; col < width; ++col) {
    for (int row = 0; row < n; ++row) {
      const int idx = indices(row, col);
      const double dist = distances(row, col);
      if (idx == NA_INTEGER || idx < 1 || idx > n_reference) {
        Rcpp::stop("KNN indices must be 1-based row numbers into the reference layout");
      }
      if (!std::isfinite(dist) || dist < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative");
      }
    }
  }

  if (k == width) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
  }

  IntegerMatrix out_indices(n, k);
  NumericMatrix out_distances(n, k);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      out_indices(row, col) = indices(row, col);
      out_distances(row, col) = distances(row, col);
    }
  }
  return List::create(
    Rcpp::Named("indices") = out_indices,
    Rcpp::Named("distances") = out_distances
  );
}

// [[Rcpp::export]]
double mean_neighbor_rank_error_cpp(IntegerMatrix high_indices,
                                    IntegerMatrix embed_indices,
                                    int k) {
  const int n = high_indices.nrow();
  if (embed_indices.nrow() != n) {
    Rcpp::stop("high_indices and embed_indices must have the same row count");
  }
  k = std::min(k, std::min(high_indices.ncol(), embed_indices.ncol()));
  if (k < 1) return R_NaReal;

  double sum_error = 0.0;
  double count = 0.0;
  std::vector<int> ranks;
  ranks.reserve(static_cast<std::size_t>(k));
  for (int row = 0; row < n; ++row) {
    ranks.clear();
    for (int col = 0; col < k; ++col) ranks.push_back(high_indices(row, col));
    for (int col = 0; col < k; ++col) {
      const int target = embed_indices(row, col);
      int high_rank = k + 1;
      for (int r = 0; r < k; ++r) {
        if (ranks[static_cast<std::size_t>(r)] == target) {
          high_rank = r + 1;
          break;
        }
      }
      sum_error += std::abs(high_rank - (col + 1));
      count += 1.0;
    }
  }
  return count > 0.0 ? sum_error / count : R_NaReal;
}

// [[Rcpp::export]]
NumericVector knn_recall_cpp(IntegerMatrix approx_indices,
                             IntegerMatrix exact_indices,
                             int k) {
  const int n = approx_indices.nrow();
  if (exact_indices.nrow() != n) {
    Rcpp::stop("Approximate and exact KNN must have the same number of rows");
  }
  k = std::min(k, std::min(approx_indices.ncol(), exact_indices.ncol()));
  if (k < 1) Rcpp::stop("k must be positive");

  std::vector<double> recalls(static_cast<std::size_t>(n), 0.0);
  for (int row = 0; row < n; ++row) {
    int shared = 0;
    for (int a = 0; a < k; ++a) {
      const int candidate = approx_indices(row, a);
      for (int e = 0; e < k; ++e) {
        if (candidate == exact_indices(row, e)) {
          ++shared;
          break;
        }
      }
    }
    recalls[static_cast<std::size_t>(row)] =
      static_cast<double>(shared) / static_cast<double>(k);
  }

  double sum = 0.0;
  double min_value = std::numeric_limits<double>::infinity();
  for (const double value : recalls) {
    sum += value;
    min_value = std::min(min_value, value);
  }
  std::vector<double> sorted = recalls;
  const std::size_t mid = sorted.size() / 2u;
  std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
  double median = sorted[mid];
  if (sorted.size() % 2u == 0u) {
    std::nth_element(sorted.begin(), sorted.begin() + mid - 1u, sorted.begin() + mid);
    median = 0.5 * (median + sorted[mid - 1u]);
  }

  NumericVector out = NumericVector::create(
    Rcpp::Named("recall_at_k") = sum / static_cast<double>(n),
    Rcpp::Named("median_recall_at_k") = median,
    Rcpp::Named("min_recall_at_k") = min_value
  );
  return out;
}

// [[Rcpp::export]]
IntegerVector majority_vote_knn_labels_cpp(IntegerMatrix embed_indices,
                                           IntegerVector labels,
                                           int k,
                                           int n_label_levels) {
  const int n = embed_indices.nrow();
  if (labels.size() != n) Rcpp::stop("labels length must match KNN row count");
  k = std::min(k, embed_indices.ncol());
  if (k < 1) Rcpp::stop("k must be positive");

  IntegerVector out(n, NA_INTEGER);
  std::vector<int> counts(static_cast<std::size_t>(std::max(0, n_label_levels)) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    std::fill(counts.begin(), counts.end(), 0);
    for (int col = 0; col < k; ++col) {
      const int idx = embed_indices(row, col) - 1;
      if (idx < 0 || idx >= n) continue;
      const int label = labels[idx];
      if (label != NA_INTEGER && label >= 1 && label <= n_label_levels) {
        ++counts[static_cast<std::size_t>(label)];
      }
    }
    int best_label = NA_INTEGER;
    int best_count = 0;
    for (int label = 1; label <= n_label_levels; ++label) {
      const int count = counts[static_cast<std::size_t>(label)];
      if (count > best_count) {
        best_count = count;
        best_label = label;
      }
    }
    out[row] = best_label;
  }
  return out;
}

// [[Rcpp::export]]
NumericVector batch_entropy_cpp(IntegerMatrix embed_indices,
                                IntegerVector batch,
                                int k,
                                int n_batch_levels) {
  const int n = embed_indices.nrow();
  if (batch.size() != n) Rcpp::stop("batch length must match KNN row count");
  k = std::min(k, embed_indices.ncol());
  if (k < 1 || n_batch_levels < 2) {
    return NumericVector::create(
      Rcpp::Named("batch_entropy") = R_NaReal,
      Rcpp::Named("batch_mixing_score") = R_NaReal
    );
  }

  const double denom = std::log(static_cast<double>(n_batch_levels));
  std::vector<int> counts(static_cast<std::size_t>(n_batch_levels) + 1u, 0);
  double entropy_sum = 0.0;
  int scored = 0;
  for (int row = 0; row < n; ++row) {
    std::fill(counts.begin(), counts.end(), 0);
    int valid = 0;
    for (int col = 0; col < k; ++col) {
      const int idx = embed_indices(row, col) - 1;
      if (idx < 0 || idx >= n) continue;
      const int label = batch[idx];
      if (label != NA_INTEGER && label >= 1 && label <= n_batch_levels) {
        ++counts[static_cast<std::size_t>(label)];
        ++valid;
      }
    }
    if (valid == 0) continue;
    double entropy = 0.0;
    for (int label = 1; label <= n_batch_levels; ++label) {
      const int count = counts[static_cast<std::size_t>(label)];
      if (count == 0) continue;
      const double prob = static_cast<double>(count) / static_cast<double>(valid);
      entropy -= prob * std::log(prob);
    }
    entropy_sum += entropy / denom;
    ++scored;
  }
  const double value = scored > 0 ? entropy_sum / static_cast<double>(scored) : R_NaReal;
  return NumericVector::create(
    Rcpp::Named("batch_entropy") = value,
    Rcpp::Named("batch_mixing_score") = value
  );
}

// [[Rcpp::export]]
NumericVector sampled_pair_distances_cpp(NumericMatrix x,
                                         IntegerVector a,
                                         IntegerVector b,
                                         int n_threads) {
  const int n_pairs = a.size();
  if (b.size() != n_pairs) Rcpp::stop("a and b must have the same length");
  const int n = x.nrow();
  const int p = x.ncol();
  NumericVector out(n_pairs);
  if (n_pairs == 0) return out;
  if (n_threads < 1) n_threads = 1;
  n_threads = std::min(n_threads, n_pairs);

  auto write_range = [&](const int begin, const int end) {
    for (int i = begin; i < end; ++i) {
      const int ai = a[i] - 1;
      const int bi = b[i] - 1;
      if (ai < 0 || ai >= n || bi < 0 || bi >= n) {
        Rcpp::stop("pair indices are out of range");
      }
      double d2 = 0.0;
      for (int col = 0; col < p; ++col) {
        const double diff = x(ai, col) - x(bi, col);
        d2 += diff * diff;
      }
      out[i] = std::sqrt(std::max(0.0, d2));
    }
  };

  if (n_threads == 1) {
    write_range(0, n_pairs);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_pairs * t) / n_threads;
      const int end = (n_pairs * (t + 1)) / n_threads;
      workers.emplace_back(write_range, start, end);
    }
    for (auto& worker : workers) worker.join();
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector knn_structure_score_cpp(NumericMatrix layout,
                                            IntegerMatrix indices,
                                            Rcpp::IntegerVector keep,
                                            int preserve_k,
                                            Rcpp::IntegerVector labels,
                                            int n_label_levels) {
  const int n = layout.nrow();
  const bool compact_indices = indices.nrow() == keep.size();
  if (indices.nrow() != n && !compact_indices) {
    Rcpp::stop("indices row count must match layout row count or keep length");
  }
  if (preserve_k < 1 || preserve_k > indices.ncol()) Rcpp::stop("invalid preserve_k");
  if (labels.size() != 0 && labels.size() != n) Rcpp::stop("labels length must match layout row count");

  const int high_rank_limit = indices.ncol();
  struct ScoreAccum {
    double preservation_sum = 0.0;
    double trust_sum = 0.0;
    double continuity_sum = 0.0;
    double label_accuracy_sum = 0.0;
    int label_accuracy_n = 0;
    int scored = 0;
  };

  const int keep_n = keep.size();
  int score_threads = 1;
  if (keep_n >= 128) {
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 1u) score_threads = std::min<int>(keep_n, std::min<unsigned int>(hw, 4u));
  }
  std::vector<ScoreAccum> accumulators(static_cast<std::size_t>(score_threads));

  auto score_range = [&](const int thread_id, const int begin, const int end) {
    ScoreAccum local;
    std::vector<int> label_counts(static_cast<std::size_t>(std::max(0, n_label_levels)) + 1u);
    std::vector<std::pair<double, int>> low_order;
    std::vector<std::pair<double, int>> continuity_targets;
    std::vector<int> continuity_diff;
    low_order.reserve(static_cast<std::size_t>(std::max(0, preserve_k)));
    continuity_targets.reserve(static_cast<std::size_t>(std::max(0, preserve_k)));
    continuity_diff.reserve(static_cast<std::size_t>(std::max(0, preserve_k)) + 1u);

    for (int kk = begin; kk < end; ++kk) {
      const int query = keep[kk] - 1;
      if (query < 0 || query >= n) continue;
      const int index_row = compact_indices ? kk : query;

      low_order.clear();
      continuity_targets.clear();
      for (int r = 0; r < preserve_k; ++r) {
        const int high_nb = indices(index_row, r) - 1;
        if (high_nb < 0 || high_nb >= n) continue;
        continuity_targets.emplace_back(layout_distance_sq(layout, query, high_nb), high_nb);
      }
      std::sort(continuity_targets.begin(), continuity_targets.end());
      continuity_diff.assign(continuity_targets.size() + 1u, 0);
      for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == query) continue;
        const std::pair<double, int> candidate_rank_key(
          layout_distance_sq(layout, query, candidate),
          candidate
        );
        if (static_cast<int>(low_order.size()) < preserve_k) {
          low_order.push_back(candidate_rank_key);
          std::push_heap(low_order.begin(), low_order.end());
        } else if (candidate_rank_key < low_order.front()) {
          std::pop_heap(low_order.begin(), low_order.end());
          low_order.back() = candidate_rank_key;
          std::push_heap(low_order.begin(), low_order.end());
        }
        const auto first_greater = std::upper_bound(
          continuity_targets.begin(),
          continuity_targets.end(),
          candidate_rank_key
        );
        const std::size_t pos = static_cast<std::size_t>(
          first_greater - continuity_targets.begin()
        );
        if (pos < continuity_targets.size()) {
          ++continuity_diff[pos];
          --continuity_diff[continuity_targets.size()];
        }
      }
      if (static_cast<int>(low_order.size()) < preserve_k) continue;
      std::sort(low_order.begin(), low_order.end());

      int shared = 0;
      double trust_penalty = 0.0;
      for (int r = 0; r < preserve_k; ++r) {
        const int low_nb = low_order[static_cast<std::size_t>(r)].second;
        const int high_rank = find_high_rank(indices, index_row, low_nb, high_rank_limit);
        if (high_rank <= preserve_k) ++shared;
        trust_penalty += std::max(0, high_rank - preserve_k);
      }
      const double trust_denom = static_cast<double>(preserve_k) *
        static_cast<double>(std::max(1, high_rank_limit + 1 - preserve_k));

      double cont_penalty = 0.0;
      int lower_rank_count = 0;
      for (std::size_t t = 0; t < continuity_targets.size(); ++t) {
        lower_rank_count += continuity_diff[t];
        const int low_rank = 1 + lower_rank_count;
        cont_penalty += std::max(0, low_rank - preserve_k);
      }
      const double cont_denom = static_cast<double>(preserve_k) *
        static_cast<double>(std::max(1, n - preserve_k));

      local.preservation_sum += static_cast<double>(shared) / static_cast<double>(preserve_k);
      local.trust_sum += std::max(0.0, std::min(1.0, 1.0 - trust_penalty / trust_denom));
      local.continuity_sum += std::max(0.0, std::min(1.0, 1.0 - cont_penalty / cont_denom));

      if (labels.size() == n && n_label_levels > 0 && labels[query] != NA_INTEGER) {
        std::fill(label_counts.begin(), label_counts.end(), 0);
        for (int r = 0; r < preserve_k; ++r) {
          const int label = labels[low_order[static_cast<std::size_t>(r)].second];
          if (label != NA_INTEGER && label >= 1 && label <= n_label_levels) {
            ++label_counts[static_cast<std::size_t>(label)];
          }
        }
        int best_label = 0;
        int best_count = 0;
        for (int label = 1; label <= n_label_levels; ++label) {
          if (label_counts[static_cast<std::size_t>(label)] > best_count) {
            best_count = label_counts[static_cast<std::size_t>(label)];
            best_label = label;
          }
        }
        if (best_count > 0) {
          local.label_accuracy_sum += best_label == labels[query] ? 1.0 : 0.0;
          ++local.label_accuracy_n;
        }
      }

      ++local.scored;
    }
    accumulators[static_cast<std::size_t>(thread_id)] = local;
  };

  if (score_threads == 1) {
    score_range(0, 0, keep_n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(score_threads - 1));
    const int chunk = (keep_n + score_threads - 1) / score_threads;
    for (int t = 1; t < score_threads; ++t) {
      const int begin = t * chunk;
      const int end = std::min(keep_n, begin + chunk);
      workers.emplace_back(score_range, t, begin, end);
    }
    score_range(0, 0, std::min(keep_n, chunk));
    for (auto& worker : workers) worker.join();
  }

  double preservation_sum = 0.0;
  double trust_sum = 0.0;
  double continuity_sum = 0.0;
  double label_accuracy_sum = 0.0;
  int label_accuracy_n = 0;
  int scored = 0;
  for (const ScoreAccum& acc : accumulators) {
    preservation_sum += acc.preservation_sum;
    trust_sum += acc.trust_sum;
    continuity_sum += acc.continuity_sum;
    label_accuracy_sum += acc.label_accuracy_sum;
    label_accuracy_n += acc.label_accuracy_n;
    scored += acc.scored;
  }

  if (scored == 0) {
    return Rcpp::NumericVector::create(
      Rcpp::Named("knn_preservation") = NA_REAL,
      Rcpp::Named("local_trustworthiness") = NA_REAL,
      Rcpp::Named("local_continuity") = NA_REAL,
      Rcpp::Named("structure_score") = NA_REAL,
      Rcpp::Named("embedding_knn_accuracy") = NA_REAL
    );
  }

  const double preservation = preservation_sum / scored;
  const double trustworthiness = trust_sum / scored;
  const double continuity = continuity_sum / scored;
  const double structure = (preservation + trustworthiness + continuity) / 3.0;
  const double label_accuracy = label_accuracy_n > 0 ?
    label_accuracy_sum / label_accuracy_n :
    R_NaN;

  return Rcpp::NumericVector::create(
    Rcpp::Named("knn_preservation") = preservation,
    Rcpp::Named("local_trustworthiness") = trustworthiness,
    Rcpp::Named("local_continuity") = continuity,
    Rcpp::Named("structure_score") = structure,
    Rcpp::Named("embedding_knn_accuracy") = label_accuracy
  );
}

// [[Rcpp::export]]
Rcpp::NumericMatrix exact_structure_metrics_cpp(NumericMatrix high,
                                                NumericMatrix low,
                                                IntegerVector requested_k,
                                                int n_threads) {
  const int n = high.nrow();
  if (low.nrow() != n) Rcpp::stop("high and low must have the same row count");
  if (n < 3) Rcpp::stop("at least three observations are required");
  if (high.ncol() < 1 || low.ncol() < 1) Rcpp::stop("input matrices must have columns");
  if (n > 5000) {
    Rcpp::stop("exact rank metrics support at most 5000 sampled observations");
  }

  const int nk = requested_k.size();
  if (nk < 1) Rcpp::stop("requested_k must not be empty");
  std::vector<int> ks(static_cast<std::size_t>(nk));
  for (int q = 0; q < nk; ++q) {
    const int k = requested_k[q];
    if (k < 1 || k >= n || 2 * n - 3 * k - 1 <= 0) {
      Rcpp::stop("each k must satisfy 1 <= k < (2 * n - 1) / 3");
    }
    ks[static_cast<std::size_t>(q)] = k;
  }

  const std::size_t matrix_size = static_cast<std::size_t>(n) * n;
  std::vector<double> high_dist(matrix_size, 0.0);
  std::vector<double> low_dist(matrix_size, 0.0);
  const int high_p = high.ncol();
  const int low_p = low.ncol();
  const double* const high_data = REAL(high);
  const double* const low_data = REAL(low);
  n_threads = std::max(1, std::min(n_threads, n));

  auto distance_range = [&](const int begin, const int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = i + 1; j < n; ++j) {
        double high_d2 = 0.0;
        for (int c = 0; c < high_p; ++c) {
          const std::size_t offset = static_cast<std::size_t>(c) * n;
          const double delta = high_data[offset + i] - high_data[offset + j];
          high_d2 += delta * delta;
        }
        double low_d2 = 0.0;
        for (int c = 0; c < low_p; ++c) {
          const std::size_t offset = static_cast<std::size_t>(c) * n;
          const double delta = low_data[offset + i] - low_data[offset + j];
          low_d2 += delta * delta;
        }
        const std::size_t ij = static_cast<std::size_t>(i) * n + j;
        const std::size_t ji = static_cast<std::size_t>(j) * n + i;
        high_dist[ij] = high_dist[ji] = high_d2;
        low_dist[ij] = low_dist[ji] = low_d2;
      }
    }
  };

  if (n_threads == 1) {
    distance_range(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      workers.emplace_back(distance_range, n * t / n_threads, n * (t + 1) / n_threads);
    }
    for (auto& worker : workers) worker.join();
  }

  struct RankAccum {
    std::vector<double> shared;
    std::vector<double> trust_penalty;
    std::vector<double> continuity_penalty;
    std::vector<double> rank_error;

    explicit RankAccum(const int size) :
      shared(static_cast<std::size_t>(size), 0.0),
      trust_penalty(static_cast<std::size_t>(size), 0.0),
      continuity_penalty(static_cast<std::size_t>(size), 0.0),
      rank_error(static_cast<std::size_t>(size), 0.0) {}
  };

  std::vector<RankAccum> accumulators;
  accumulators.reserve(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t) accumulators.emplace_back(nk);

  auto rank_range = [&](const int thread_id, const int begin, const int end) {
    RankAccum local(nk);
    std::vector<int> high_order(static_cast<std::size_t>(n - 1));
    std::vector<int> low_order(static_cast<std::size_t>(n - 1));
    std::vector<int> high_rank(static_cast<std::size_t>(n));
    std::vector<int> low_rank(static_cast<std::size_t>(n));

    for (int i = begin; i < end; ++i) {
      int pos = 0;
      for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        high_order[static_cast<std::size_t>(pos)] = j;
        low_order[static_cast<std::size_t>(pos)] = j;
        ++pos;
      }
      const double* high_row = high_dist.data() + static_cast<std::size_t>(i) * n;
      const double* low_row = low_dist.data() + static_cast<std::size_t>(i) * n;
      std::sort(high_order.begin(), high_order.end(), [&](const int lhs, const int rhs) {
        return high_row[lhs] < high_row[rhs] ||
          (high_row[lhs] == high_row[rhs] && lhs < rhs);
      });
      std::sort(low_order.begin(), low_order.end(), [&](const int lhs, const int rhs) {
        return low_row[lhs] < low_row[rhs] ||
          (low_row[lhs] == low_row[rhs] && lhs < rhs);
      });
      for (int r = 0; r < n - 1; ++r) {
        high_rank[static_cast<std::size_t>(high_order[static_cast<std::size_t>(r)])] = r + 1;
        low_rank[static_cast<std::size_t>(low_order[static_cast<std::size_t>(r)])] = r + 1;
      }

      for (int q = 0; q < nk; ++q) {
        const int k = ks[static_cast<std::size_t>(q)];
        for (int r = 0; r < k; ++r) {
          const int low_nb = low_order[static_cast<std::size_t>(r)];
          const int high_nb = high_order[static_cast<std::size_t>(r)];
          const int rank_in_high = high_rank[static_cast<std::size_t>(low_nb)];
          const int rank_in_low = low_rank[static_cast<std::size_t>(high_nb)];
          if (rank_in_high <= k) local.shared[static_cast<std::size_t>(q)] += 1.0;
          if (rank_in_high > k) {
            local.trust_penalty[static_cast<std::size_t>(q)] += rank_in_high - k;
          }
          if (rank_in_low > k) {
            local.continuity_penalty[static_cast<std::size_t>(q)] += rank_in_low - k;
          }
          local.rank_error[static_cast<std::size_t>(q)] +=
            std::abs(rank_in_high - (r + 1));
        }
      }
    }
    accumulators[static_cast<std::size_t>(thread_id)] = std::move(local);
  };

  if (n_threads == 1) {
    rank_range(0, 0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      workers.emplace_back(rank_range, t, n * t / n_threads, n * (t + 1) / n_threads);
    }
    for (auto& worker : workers) worker.join();
  }

  NumericMatrix out(nk, 4);
  for (int q = 0; q < nk; ++q) {
    double shared = 0.0;
    double trust_penalty = 0.0;
    double continuity_penalty = 0.0;
    double rank_error = 0.0;
    for (const auto& acc : accumulators) {
      shared += acc.shared[static_cast<std::size_t>(q)];
      trust_penalty += acc.trust_penalty[static_cast<std::size_t>(q)];
      continuity_penalty += acc.continuity_penalty[static_cast<std::size_t>(q)];
      rank_error += acc.rank_error[static_cast<std::size_t>(q)];
    }
    const double k = static_cast<double>(ks[static_cast<std::size_t>(q)]);
    const double normalizer = 2.0 /
      (static_cast<double>(n) * k * (2.0 * n - 3.0 * k - 1.0));
    out(q, 0) = std::max(0.0, std::min(1.0, 1.0 - normalizer * trust_penalty));
    out(q, 1) = std::max(0.0, std::min(1.0, 1.0 - normalizer * continuity_penalty));
    out(q, 2) = shared / (static_cast<double>(n) * k);
    out(q, 3) = rank_error / (static_cast<double>(n) * k);
  }
  Rcpp::colnames(out) = Rcpp::CharacterVector::create(
    "trustworthiness", "continuity", "knn_preservation", "mean_neighbor_rank_error"
  );
  return out;
}

// [[Rcpp::export]]
double silhouette_score_cpp(NumericMatrix layout, Rcpp::IntegerVector labels) {
  const int n = layout.nrow();
  const int n_components = layout.ncol();
  if (labels.size() != n) Rcpp::stop("labels length must match layout row count");

  int max_label = 0;
  std::vector<int> valid;
  valid.reserve(n);
  for (int i = 0; i < n; ++i) {
    const int label = labels[i];
    if (label == NA_INTEGER || label < 1) continue;
    bool finite = true;
    for (int c = 0; c < n_components; ++c) {
      if (!std::isfinite(layout(i, c))) {
        finite = false;
        break;
      }
    }
    if (!finite) continue;
    max_label = std::max(max_label, label);
    valid.push_back(i);
  }

  const int n_valid = static_cast<int>(valid.size());
  if (n_valid < 2 || max_label < 2) return NA_REAL;

  std::vector<int> counts(static_cast<std::size_t>(max_label) + 1u, 0);
  for (const int i : valid) ++counts[static_cast<std::size_t>(labels[i])];

  int n_nonempty_classes = 0;
  for (int label = 1; label <= max_label; ++label) {
    if (counts[static_cast<std::size_t>(label)] > 0) ++n_nonempty_classes;
  }
  if (n_nonempty_classes < 2) return NA_REAL;

  std::vector<double> class_sums(static_cast<std::size_t>(max_label) + 1u, 0.0);
  double total = 0.0;
  int scored = 0;
  for (const int i : valid) {
    std::fill(class_sums.begin(), class_sums.end(), 0.0);
    const int own_label = labels[i];

    if (n_components == 2) {
      const double xi0 = layout(i, 0);
      const double xi1 = layout(i, 1);
      for (const int j : valid) {
        if (j == i) continue;
        const double dx = xi0 - layout(j, 0);
        const double dy = xi1 - layout(j, 1);
        class_sums[static_cast<std::size_t>(labels[j])] += std::sqrt(dx * dx + dy * dy);
      }
    } else {
      for (const int j : valid) {
        if (j == i) continue;
        double dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = layout(i, c) - layout(j, c);
          dist_sq += diff * diff;
        }
        class_sums[static_cast<std::size_t>(labels[j])] += std::sqrt(std::max(0.0, dist_sq));
      }
    }

    const int own_count = counts[static_cast<std::size_t>(own_label)] - 1;
    const double a = own_count > 0 ?
      class_sums[static_cast<std::size_t>(own_label)] / static_cast<double>(own_count) :
      0.0;

    double b = std::numeric_limits<double>::infinity();
    for (int label = 1; label <= max_label; ++label) {
      if (label == own_label || counts[static_cast<std::size_t>(label)] == 0) continue;
      b = std::min(
        b,
        class_sums[static_cast<std::size_t>(label)] /
          static_cast<double>(counts[static_cast<std::size_t>(label)])
      );
    }

    if (!std::isfinite(b)) {
      total += 0.0;
    } else {
      const double denom = std::max(a, b);
      total += denom > 0.0 ? (b - a) / denom : 0.0;
    }
    ++scored;
  }

  return scored > 0 ? total / static_cast<double>(scored) : NA_REAL;
}

// [[Rcpp::export]]
NumericMatrix interpolate_landmark_layout_cpp(NumericMatrix landmark_layout,
                                              Rcpp::IntegerVector landmark_indices,
                                              IntegerMatrix projection_indices,
                                              NumericMatrix projection_distances,
                                              int n) {
  const int n_landmarks = landmark_layout.nrow();
  const int n_components = landmark_layout.ncol();
  const int projection_n = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();

  if (n < 1) Rcpp::stop("n must be positive");
  if (projection_n != n) Rcpp::stop("projection_indices row count must equal n");
  if (projection_distances.nrow() != projection_n ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  if (landmark_indices.size() != n_landmarks) {
    Rcpp::stop("landmark_indices length must match landmark_layout rows");
  }

  NumericMatrix layout(n, n_components);
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<float> adjusted;
  std::vector<float> positive;
  adjusted.reserve(static_cast<std::size_t>(projection_k));
  positive.reserve(static_cast<std::size_t>(projection_k));

  for (int i = 0; i < n; ++i) {
    int zero_col = -1;
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      if (idx < 1 || idx > n_landmarks) Rcpp::stop("projection indices out of range");
      const double d = std::max(0.0, projection_distances(i, j));
      if (d <= eps && zero_col < 0) zero_col = j;
      if (d < rho) rho = d;
    }

    if (zero_col >= 0) {
      const int landmark = projection_indices(i, zero_col) - 1;
      for (int c = 0; c < n_components; ++c) layout(i, c) = landmark_layout(landmark, c);
      continue;
    }

    adjusted.clear();
    positive.clear();
    for (int j = 0; j < projection_k; ++j) {
      const double d = std::max(0.0, projection_distances(i, j));
      const double value = std::max(0.0, d - rho);
      adjusted.push_back(static_cast<float>(value));
      if (value > eps) positive.push_back(static_cast<float>(value));
    }

    double sigma = R_NaReal;
    if (positive.empty()) {
      std::vector<float> distances;
      distances.reserve(static_cast<std::size_t>(projection_k));
      for (int j = 0; j < projection_k; ++j) {
        distances.push_back(static_cast<float>(std::max(0.0, projection_distances(i, j))));
      }
      sigma = median_inplace(distances);
    } else {
      sigma = median_inplace(positive);
    }
    if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

    double weight_sum = 0.0;
    for (int j = 0; j < projection_k; ++j) {
      const double w = std::exp(-adjusted[static_cast<std::size_t>(j)] / sigma);
      adjusted[static_cast<std::size_t>(j)] = static_cast<float>(w);
      weight_sum += w;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      weight_sum = static_cast<double>(projection_k);
      std::fill(adjusted.begin(), adjusted.end(), 1.0f);
    }

    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      for (int j = 0; j < projection_k; ++j) {
        const int landmark = projection_indices(i, j) - 1;
        value += static_cast<double>(adjusted[static_cast<std::size_t>(j)]) *
          landmark_layout(landmark, c);
      }
      layout(i, c) = value / weight_sum;
    }
  }

  for (int i = 0; i < n_landmarks; ++i) {
    const int row = landmark_indices[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("landmark indices out of range");
    for (int c = 0; c < n_components; ++c) layout(row, c) = landmark_layout(i, c);
  }
  return layout;
}

// [[Rcpp::export]]
List select_low_confidence_rows_cpp(NumericVector confidence,
                                    IntegerVector landmark_indices,
                                    double fraction) {
  const int n = confidence.size();
  if (n < 1) Rcpp::stop("confidence must be non-empty");
  if (!std::isfinite(fraction) || fraction <= 0.0) {
    Rcpp::stop("fraction must be a positive finite value");
  }
  fraction = std::min(1.0, fraction);

  std::vector<unsigned char> is_landmark(static_cast<std::size_t>(n), 0u);
  for (int i = 0; i < landmark_indices.size(); ++i) {
    const int row = landmark_indices[i] - 1;
    if (row >= 0 && row < n) {
      is_landmark[static_cast<std::size_t>(row)] = 1u;
    }
  }

  std::vector<std::pair<double, int>> eligible;
  eligible.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double score = confidence[i];
    if (!std::isfinite(score)) Rcpp::stop("confidence scores must be finite");
    if (is_landmark[static_cast<std::size_t>(i)] == 0u) {
      eligible.emplace_back(score, i);
    }
  }

  if (eligible.empty()) {
    return List::create(
      Rcpp::Named("rows") = IntegerVector(0),
      Rcpp::Named("policy") = "low_confidence",
      Rcpp::Named("selected") = 0,
      Rcpp::Named("selected_fraction") = 0.0,
      Rcpp::Named("confidence_threshold") = R_NaReal,
      Rcpp::Named("selection_backend") = "cpp_confidence_mask"
    );
  }

  int count = static_cast<int>(std::ceil(static_cast<double>(eligible.size()) * fraction));
  count = std::max(1, std::min(static_cast<int>(eligible.size()), count));
  if (count >= static_cast<int>(eligible.size())) {
    return List::create(
      Rcpp::Named("rows") = R_NilValue,
      Rcpp::Named("policy") = "all",
      Rcpp::Named("selected") = static_cast<int>(eligible.size()),
      Rcpp::Named("selected_fraction") =
        static_cast<double>(eligible.size()) / static_cast<double>(n),
      Rcpp::Named("confidence_threshold") = R_NaReal,
      Rcpp::Named("selection_backend") = "cpp_confidence_mask"
    );
  }

  auto row_less = [](const std::pair<double, int>& a,
                     const std::pair<double, int>& b) {
    if (a.first == b.first) return a.second < b.second;
    return a.first < b.first;
  };
  std::nth_element(eligible.begin(), eligible.begin() + count, eligible.end(), row_less);
  eligible.resize(static_cast<std::size_t>(count));
  std::sort(eligible.begin(), eligible.end(), row_less);

  IntegerVector rows(count);
  double threshold = eligible.front().first;
  for (int i = 0; i < count; ++i) {
    rows[i] = eligible[static_cast<std::size_t>(i)].second + 1;
    threshold = std::max(threshold, eligible[static_cast<std::size_t>(i)].first);
  }
  std::sort(rows.begin(), rows.end());

  return List::create(
    Rcpp::Named("rows") = rows,
    Rcpp::Named("policy") = "low_confidence",
    Rcpp::Named("selected") = count,
    Rcpp::Named("selected_fraction") = static_cast<double>(count) / static_cast<double>(n),
    Rcpp::Named("confidence_threshold") = threshold,
    Rcpp::Named("selection_backend") = "cpp_confidence_mask"
  );
}

// [[Rcpp::export]]
NumericMatrix project_embedding_knn_cpp(NumericMatrix reference_layout,
                                        IntegerMatrix projection_indices,
                                        NumericMatrix projection_distances) {
  const int n_reference = reference_layout.nrow();
  const int n_components = reference_layout.ncol();
  const int n_query = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();

  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (n_components < 1) Rcpp::stop("reference_layout must have at least one column");
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (projection_distances.nrow() != n_query ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }

  NumericMatrix layout(n_query, n_components);
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<float> adjusted;
  std::vector<float> positive;
  std::vector<float> row_distances;
  adjusted.reserve(static_cast<std::size_t>(projection_k));
  positive.reserve(static_cast<std::size_t>(projection_k));
  row_distances.reserve(static_cast<std::size_t>(projection_k));

  for (int i = 0; i < n_query; ++i) {
    int zero_count = 0;
    double rho = std::numeric_limits<double>::infinity();
    row_distances.clear();
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      const double d = projection_distances(i, j);
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
      row_distances.push_back(static_cast<float>(d));
      if (d <= eps) ++zero_count;
      if (d < rho) rho = d;
    }

    if (zero_count > 0) {
      const double inv_zero_count = 1.0 / static_cast<double>(zero_count);
      for (int c = 0; c < n_components; ++c) {
        double value = 0.0;
        for (int j = 0; j < projection_k; ++j) {
          if (static_cast<double>(row_distances[static_cast<std::size_t>(j)]) <= eps) {
            const int reference_row = projection_indices(i, j) - 1;
            value += inv_zero_count * reference_layout(reference_row, c);
          }
        }
        layout(i, c) = value;
      }
      continue;
    }

    adjusted.clear();
    positive.clear();
    for (int j = 0; j < projection_k; ++j) {
      const double value = std::max(
        0.0,
        static_cast<double>(row_distances[static_cast<std::size_t>(j)]) - rho
      );
      adjusted.push_back(static_cast<float>(value));
      if (value > eps) positive.push_back(static_cast<float>(value));
    }

    double sigma = positive.empty() ? median_inplace(row_distances) : median_inplace(positive);
    if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

    double weight_sum = 0.0;
    for (int j = 0; j < projection_k; ++j) {
      const double w = std::exp(-adjusted[static_cast<std::size_t>(j)] / sigma);
      adjusted[static_cast<std::size_t>(j)] = static_cast<float>(w);
      weight_sum += w;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      weight_sum = static_cast<double>(projection_k);
      std::fill(adjusted.begin(), adjusted.end(), 1.0f);
    }

    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      for (int j = 0; j < projection_k; ++j) {
        const int reference_row = projection_indices(i, j) - 1;
        value += static_cast<double>(adjusted[static_cast<std::size_t>(j)]) *
          reference_layout(reference_row, c);
      }
      layout(i, c) = value / weight_sum;
    }
  }
  return layout;
}

// [[Rcpp::export]]
List project_embedding_affine_cpp(NumericMatrix reference_data,
                                  NumericMatrix query_data,
                                  NumericMatrix reference_layout,
                                  IntegerMatrix projection_indices,
                                  NumericMatrix projection_distances,
                                  int max_neighbors = 12,
                                  double ridge = 1e-3,
                                  double max_extrapolation = 2.5) {
  const int n_reference = reference_layout.nrow();
  const int n_components = reference_layout.ncol();
  const int n_query = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();
  const int n_features = reference_data.ncol();

  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (reference_data.nrow() != n_reference) {
    Rcpp::stop("reference_data and reference_layout must have the same number of rows");
  }
  if (query_data.nrow() != n_query) {
    Rcpp::stop("query_data and projection_indices must have the same number of rows");
  }
  if (query_data.ncol() != n_features) {
    Rcpp::stop("reference_data and query_data must have the same number of columns");
  }
  if (n_components < 1) Rcpp::stop("reference_layout must have at least one column");
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (projection_distances.nrow() != n_query ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  if (max_neighbors < 3) max_neighbors = 3;
  max_neighbors = std::min(max_neighbors, projection_k);
  if (!std::isfinite(ridge) || ridge <= 0.0) ridge = 1e-3;
  if (!std::isfinite(max_extrapolation) || max_extrapolation <= 0.0) {
    max_extrapolation = 2.5;
  }

  NumericMatrix weighted = project_embedding_knn_cpp(
    reference_layout,
    projection_indices,
    projection_distances
  );
  NumericMatrix layout(n_query, n_components);
  NumericVector confidence(n_query);
  IntegerVector used_neighbors(n_query);
  IntegerVector fallback(n_query);

  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<double> weights(static_cast<std::size_t>(max_neighbors), 0.0);
  std::vector<double> x_center(static_cast<std::size_t>(n_features), 0.0);
  std::vector<double> y_center(static_cast<std::size_t>(n_components), 0.0);
  std::vector<double> x_centered(static_cast<std::size_t>(max_neighbors) * n_features, 0.0);
  std::vector<double> y_rhs(static_cast<std::size_t>(max_neighbors) * n_components, 0.0);
  std::vector<double> kernel(static_cast<std::size_t>(max_neighbors) * max_neighbors, 0.0);
  std::vector<double> q(static_cast<std::size_t>(max_neighbors), 0.0);
  std::vector<float> positive;
  positive.reserve(static_cast<std::size_t>(projection_k));

  for (int i = 0; i < n_query; ++i) {
    int zero_col = -1;
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      const double d = projection_distances(i, j);
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
      if (d <= eps && zero_col < 0) zero_col = j;
      rho = std::min(rho, d);
    }
    if (zero_col >= 0) {
      const int ref = projection_indices(i, zero_col) - 1;
      for (int c = 0; c < n_components; ++c) layout(i, c) = reference_layout(ref, c);
      confidence[i] = 1.0;
      used_neighbors[i] = 1;
      fallback[i] = 0;
      continue;
    }

    const int m = std::min(max_neighbors, projection_k);
    positive.clear();
    for (int j = 0; j < projection_k; ++j) {
      const double adjusted = std::max(0.0, projection_distances(i, j) - rho);
      if (adjusted > eps) positive.push_back(static_cast<float>(adjusted));
    }
    double sigma = positive.empty() ? std::max(rho, eps) : median_inplace(positive);
    if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

    double weight_sum = 0.0;
    double weight_sq_sum = 0.0;
    for (int j = 0; j < m; ++j) {
      const double adjusted = std::max(0.0, projection_distances(i, j) - rho);
      const double w = std::exp(-adjusted / sigma);
      weights[static_cast<std::size_t>(j)] = w;
      weight_sum += w;
      weight_sq_sum += w * w;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      for (int c = 0; c < n_components; ++c) layout(i, c) = weighted(i, c);
      confidence[i] = 0.0;
      used_neighbors[i] = m;
      fallback[i] = 1;
      continue;
    }
    for (int j = 0; j < m; ++j) weights[static_cast<std::size_t>(j)] /= weight_sum;

    std::fill(x_center.begin(), x_center.end(), 0.0);
    std::fill(y_center.begin(), y_center.end(), 0.0);
    for (int j = 0; j < m; ++j) {
      const int ref = projection_indices(i, j) - 1;
      const double w = weights[static_cast<std::size_t>(j)];
      for (int f = 0; f < n_features; ++f) x_center[static_cast<std::size_t>(f)] += w * reference_data(ref, f);
      for (int c = 0; c < n_components; ++c) y_center[static_cast<std::size_t>(c)] += w * reference_layout(ref, c);
    }

    double layout_radius_sq = 0.0;
    double trace = 0.0;
    for (int a = 0; a < m; ++a) {
      const int ref_a = projection_indices(i, a) - 1;
      const double sqrt_wa = std::sqrt(weights[static_cast<std::size_t>(a)]);
      double row_norm = 0.0;
      double y_radius = 0.0;
      for (int f = 0; f < n_features; ++f) {
        const double xc = reference_data(ref_a, f) - x_center[static_cast<std::size_t>(f)];
        x_centered[static_cast<std::size_t>(a) * n_features + f] = xc;
        row_norm += xc * xc;
      }
      trace += weights[static_cast<std::size_t>(a)] * row_norm;
      for (int c = 0; c < n_components; ++c) {
        const double yc = reference_layout(ref_a, c) - y_center[static_cast<std::size_t>(c)];
        y_rhs[static_cast<std::size_t>(a) * n_components + c] = sqrt_wa * yc;
        y_radius += yc * yc;
      }
      layout_radius_sq = std::max(layout_radius_sq, y_radius);
    }

    std::fill(kernel.begin(), kernel.end(), 0.0);
    for (int a = 0; a < m; ++a) {
      const double sqrt_wa = std::sqrt(weights[static_cast<std::size_t>(a)]);
      for (int b = 0; b <= a; ++b) {
        const double sqrt_wb = std::sqrt(weights[static_cast<std::size_t>(b)]);
        double dot = 0.0;
        for (int f = 0; f < n_features; ++f) {
          dot += x_centered[static_cast<std::size_t>(a) * n_features + f] *
            x_centered[static_cast<std::size_t>(b) * n_features + f];
        }
        const double value = sqrt_wa * sqrt_wb * dot;
        kernel[static_cast<std::size_t>(a) * m + b] = value;
        kernel[static_cast<std::size_t>(b) * m + a] = value;
      }
    }
    const double lambda = ridge * std::max(trace, eps) + eps;
    for (int a = 0; a < m; ++a) {
      kernel[static_cast<std::size_t>(a) * m + a] += lambda;
    }

    bool ok = cholesky_decompose_inplace(kernel, m) &&
      cholesky_solve_inplace(kernel, y_rhs, m, n_components);
    if (!ok) {
      for (int c = 0; c < n_components; ++c) layout(i, c) = weighted(i, c);
      confidence[i] = 0.0;
      used_neighbors[i] = m;
      fallback[i] = 1;
      continue;
    }

    for (int a = 0; a < m; ++a) {
      const double sqrt_wa = std::sqrt(weights[static_cast<std::size_t>(a)]);
      double dot = 0.0;
      for (int f = 0; f < n_features; ++f) {
        dot += (query_data(i, f) - x_center[static_cast<std::size_t>(f)]) *
          x_centered[static_cast<std::size_t>(a) * n_features + f];
      }
      q[static_cast<std::size_t>(a)] = sqrt_wa * dot;
    }

    double disp_sq = 0.0;
    for (int c = 0; c < n_components; ++c) {
      double displacement = 0.0;
      for (int a = 0; a < m; ++a) {
        displacement += q[static_cast<std::size_t>(a)] *
          y_rhs[static_cast<std::size_t>(a) * n_components + c];
      }
      layout(i, c) = y_center[static_cast<std::size_t>(c)] + displacement;
      disp_sq += displacement * displacement;
    }

    const double max_disp = max_extrapolation * std::sqrt(std::max(layout_radius_sq, eps));
    if (disp_sq > max_disp * max_disp) {
      const double scale = max_disp / (std::sqrt(disp_sq) + eps);
      for (int c = 0; c < n_components; ++c) {
        layout(i, c) = y_center[static_cast<std::size_t>(c)] +
          (layout(i, c) - y_center[static_cast<std::size_t>(c)]) * scale;
      }
    }

    const double effective_n = weight_sq_sum > 0.0 ? (weight_sum * weight_sum) / weight_sq_sum : 1.0;
    confidence[i] = std::max(0.0, std::min(1.0, effective_n / static_cast<double>(m)));
    used_neighbors[i] = m;
    fallback[i] = 0;
  }

  return List::create(
    Rcpp::Named("layout") = layout,
    Rcpp::Named("confidence") = confidence,
    Rcpp::Named("used_neighbors") = used_neighbors,
    Rcpp::Named("fallback") = fallback,
    Rcpp::Named("method") = "local_affine_knn_projection",
    Rcpp::Named("max_neighbors") = max_neighbors,
    Rcpp::Named("ridge") = ridge,
    Rcpp::Named("max_extrapolation") = max_extrapolation
  );
}

// [[Rcpp::export]]
List project_embedding_affine_parallel_cpp(SEXP reference_data_sexp,
                                           SEXP query_data_sexp,
                                           SEXP reference_layout_sexp,
                                           IntegerMatrix projection_indices,
                                           SEXP projection_distances_sexp,
                                           int max_neighbors = 12,
                                           double ridge = 1e-3,
                                           double max_extrapolation = 2.5,
                                           int n_threads = 1) {
  fastembedr::FloatMatrix reference_data =
    fastembedr::matrix_to_row_major_float(
      reference_data_sexp, fastembedr::KnnMetric::Euclidean
    );
  fastembedr::FloatMatrix query_data =
    fastembedr::matrix_to_row_major_float(
      query_data_sexp, fastembedr::KnnMetric::Euclidean
    );
  fastembedr::FloatMatrix reference_layout =
    fastembedr::matrix_to_row_major_float(
      reference_layout_sexp, fastembedr::KnnMetric::Euclidean
    );
  fastembedr::FloatMatrix projection_distances =
    fastembedr::matrix_to_row_major_float(
      projection_distances_sexp, fastembedr::KnnMetric::Euclidean
    );
  const int n_reference = reference_layout.nrow;
  const int n_components = reference_layout.ncol;
  const int n_query = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();
  const int n_features = reference_data.ncol;

  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (reference_data.nrow != n_reference) {
    Rcpp::stop("reference_data and reference_layout must have the same number of rows");
  }
  if (query_data.nrow != n_query) {
    Rcpp::stop("query_data and projection_indices must have the same number of rows");
  }
  if (query_data.ncol != n_features) {
    Rcpp::stop("reference_data and query_data must have the same number of columns");
  }
  if (n_components < 1) Rcpp::stop("reference_layout must have at least one column");
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (projection_distances.nrow != n_query ||
      projection_distances.ncol != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  if (max_neighbors < 3) max_neighbors = 3;
  max_neighbors = std::min(max_neighbors, projection_k);
  if (!std::isfinite(ridge) || ridge <= 0.0) ridge = 1e-3;
  if (!std::isfinite(max_extrapolation) || max_extrapolation <= 0.0) {
    max_extrapolation = 2.5;
  }
  for (int i = 0; i < n_query; ++i) {
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      const double d = projection_distances.values[
        static_cast<std::size_t>(i) * projection_k + j
      ];
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
    }
  }

  NumericMatrix layout(n_query, n_components);
  NumericVector confidence(n_query);
  IntegerVector used_neighbors(n_query);
  IntegerVector fallback(n_query);
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  const int requested_threads = n_threads < 1 ? 1 : n_threads;
  const int threads = resolve_projection_threads(requested_threads, n_query);
  const auto reference_value = [&](const int row, const int col) -> double {
    return static_cast<double>(reference_data.values[
      static_cast<std::size_t>(row) * n_features + col
    ]);
  };
  const auto query_value = [&](const int row, const int col) -> double {
    return static_cast<double>(query_data.values[
      static_cast<std::size_t>(row) * n_features + col
    ]);
  };
  const auto layout_value = [&](const int row, const int col) -> double {
    return static_cast<double>(reference_layout.values[
      static_cast<std::size_t>(row) * n_components + col
    ]);
  };
  const auto distance_value = [&](const int row, const int col) -> double {
    return static_cast<double>(projection_distances.values[
      static_cast<std::size_t>(row) * projection_k + col
    ]);
  };

  parallel_for_rows(n_query, threads, [&](const int begin, const int end, const int) {
    std::vector<double> weights(static_cast<std::size_t>(max_neighbors), 0.0);
    std::vector<double> x_center(static_cast<std::size_t>(n_features), 0.0);
    std::vector<double> y_center(static_cast<std::size_t>(n_components), 0.0);
    std::vector<double> centered(static_cast<std::size_t>(max_neighbors), 0.0);
    std::vector<double> row_norm(static_cast<std::size_t>(max_neighbors), 0.0);
    std::vector<double> y_rhs(static_cast<std::size_t>(max_neighbors) * n_components, 0.0);
    std::vector<double> kernel(static_cast<std::size_t>(max_neighbors) * max_neighbors, 0.0);
    std::vector<double> q(static_cast<std::size_t>(max_neighbors), 0.0);
    std::vector<float> positive;
    std::vector<float> weighted_adjusted;
    std::vector<float> weighted_positive;
    std::vector<float> weighted_distances;
    positive.reserve(static_cast<std::size_t>(projection_k));
    weighted_adjusted.reserve(static_cast<std::size_t>(projection_k));
    weighted_positive.reserve(static_cast<std::size_t>(projection_k));
    weighted_distances.reserve(static_cast<std::size_t>(projection_k));

    auto weighted_fallback = [&](const int row) {
      weighted_adjusted.clear();
      weighted_positive.clear();
      weighted_distances.clear();
      double rho = std::numeric_limits<double>::infinity();
      int zero_count = 0;
      for (int j = 0; j < projection_k; ++j) {
        const double d = distance_value(row, j);
        weighted_distances.push_back(static_cast<float>(d));
        rho = std::min(rho, d);
        if (d <= eps) ++zero_count;
      }
      if (zero_count > 0) {
        for (int c = 0; c < n_components; ++c) {
          double value = 0.0;
          for (int j = 0; j < projection_k; ++j) {
            if (distance_value(row, j) <= eps) {
              value += layout_value(projection_indices(row, j) - 1, c);
            }
          }
          layout(row, c) = value / static_cast<double>(zero_count);
        }
        return;
      }
      for (int j = 0; j < projection_k; ++j) {
        const double adjusted = std::max(0.0, distance_value(row, j) - rho);
        weighted_adjusted.push_back(static_cast<float>(adjusted));
        if (adjusted > eps) weighted_positive.push_back(static_cast<float>(adjusted));
      }
      double sigma = weighted_positive.empty() ?
        median_inplace(weighted_distances) : median_inplace(weighted_positive);
      if (!std::isfinite(sigma) || sigma < eps) sigma = eps;
      double weight_sum = 0.0;
      for (int j = 0; j < projection_k; ++j) {
        const double weight = std::exp(
          -static_cast<double>(weighted_adjusted[static_cast<std::size_t>(j)]) / sigma
        );
        weighted_adjusted[static_cast<std::size_t>(j)] = static_cast<float>(weight);
        weight_sum += weight;
      }
      if (!std::isfinite(weight_sum) || weight_sum <= 0.0) weight_sum = projection_k;
      for (int c = 0; c < n_components; ++c) {
        double value = 0.0;
        for (int j = 0; j < projection_k; ++j) {
          value += static_cast<double>(weighted_adjusted[static_cast<std::size_t>(j)]) *
            layout_value(projection_indices(row, j) - 1, c);
        }
        layout(row, c) = value / weight_sum;
      }
    };

    for (int i = begin; i < end; ++i) {
      int zero_col = -1;
      double rho = std::numeric_limits<double>::infinity();
      for (int j = 0; j < projection_k; ++j) {
        const double d = distance_value(i, j);
        if (d <= eps && zero_col < 0) zero_col = j;
        rho = std::min(rho, d);
      }
      if (zero_col >= 0) {
        const int ref = projection_indices(i, zero_col) - 1;
        for (int c = 0; c < n_components; ++c) layout(i, c) = layout_value(ref, c);
        confidence[i] = 1.0;
        used_neighbors[i] = 1;
        fallback[i] = 0;
        continue;
      }

      const int m = std::min(max_neighbors, projection_k);
      positive.clear();
      for (int j = 0; j < projection_k; ++j) {
        const double adjusted = std::max(0.0, distance_value(i, j) - rho);
        if (adjusted > eps) positive.push_back(static_cast<float>(adjusted));
      }
      double sigma = positive.empty() ? std::max(rho, eps) : median_inplace(positive);
      if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

      double weight_sum = 0.0;
      double weight_sq_sum = 0.0;
      for (int j = 0; j < m; ++j) {
        const double adjusted = std::max(0.0, distance_value(i, j) - rho);
        const double w = std::exp(-adjusted / sigma);
        weights[static_cast<std::size_t>(j)] = w;
        weight_sum += w;
        weight_sq_sum += w * w;
      }
      if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
        weighted_fallback(i);
        confidence[i] = 0.0;
        used_neighbors[i] = m;
        fallback[i] = 1;
        continue;
      }
      for (int j = 0; j < m; ++j) weights[static_cast<std::size_t>(j)] /= weight_sum;

      std::fill(x_center.begin(), x_center.end(), 0.0);
      std::fill(y_center.begin(), y_center.end(), 0.0);
      for (int j = 0; j < m; ++j) {
        const int ref = projection_indices(i, j) - 1;
        const double w = weights[static_cast<std::size_t>(j)];
        for (int f = 0; f < n_features; ++f) x_center[static_cast<std::size_t>(f)] += w * reference_value(ref, f);
        for (int c = 0; c < n_components; ++c) y_center[static_cast<std::size_t>(c)] += w * layout_value(ref, c);
      }

      double layout_radius_sq = 0.0;
      for (int a = 0; a < m; ++a) {
        const int ref_a = projection_indices(i, a) - 1;
        const double sqrt_wa = std::sqrt(weights[static_cast<std::size_t>(a)]);
        double y_radius = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double yc = layout_value(ref_a, c) - y_center[static_cast<std::size_t>(c)];
          y_rhs[static_cast<std::size_t>(a) * n_components + c] = sqrt_wa * yc;
          y_radius += yc * yc;
        }
        layout_radius_sq = std::max(layout_radius_sq, y_radius);
      }

      std::fill(kernel.begin(), kernel.end(), 0.0);
      std::fill(q.begin(), q.end(), 0.0);
      std::fill(row_norm.begin(), row_norm.end(), 0.0);
      for (int f = 0; f < n_features; ++f) {
        const double center_f = x_center[static_cast<std::size_t>(f)];
        const double query_centered = query_value(i, f) - center_f;
        for (int a = 0; a < m; ++a) {
          const int ref_a = projection_indices(i, a) - 1;
          const double value = reference_value(ref_a, f) - center_f;
          centered[static_cast<std::size_t>(a)] = value;
          row_norm[static_cast<std::size_t>(a)] += value * value;
          q[static_cast<std::size_t>(a)] += query_centered * value;
        }
        for (int a = 0; a < m; ++a) {
          const double xa = centered[static_cast<std::size_t>(a)];
          for (int b = 0; b <= a; ++b) {
            kernel[static_cast<std::size_t>(a) * m + b] +=
              xa * centered[static_cast<std::size_t>(b)];
          }
        }
      }
      double trace = 0.0;
      for (int a = 0; a < m; ++a) {
        const double sqrt_wa = std::sqrt(weights[static_cast<std::size_t>(a)]);
        trace += weights[static_cast<std::size_t>(a)] * row_norm[static_cast<std::size_t>(a)];
        q[static_cast<std::size_t>(a)] *= sqrt_wa;
        for (int b = 0; b <= a; ++b) {
          const double value = sqrt_wa *
            std::sqrt(weights[static_cast<std::size_t>(b)]) *
            kernel[static_cast<std::size_t>(a) * m + b];
          kernel[static_cast<std::size_t>(a) * m + b] = value;
          kernel[static_cast<std::size_t>(b) * m + a] = value;
        }
      }
      const double lambda = ridge * std::max(trace, eps) + eps;
      for (int a = 0; a < m; ++a) kernel[static_cast<std::size_t>(a) * m + a] += lambda;

      bool ok = cholesky_decompose_inplace(kernel, m) &&
        cholesky_solve_inplace(kernel, y_rhs, m, n_components);
      if (!ok) {
        weighted_fallback(i);
        confidence[i] = 0.0;
        used_neighbors[i] = m;
        fallback[i] = 1;
        continue;
      }

      double disp_sq = 0.0;
      for (int c = 0; c < n_components; ++c) {
        double displacement = 0.0;
        for (int a = 0; a < m; ++a) {
          displacement += q[static_cast<std::size_t>(a)] *
            y_rhs[static_cast<std::size_t>(a) * n_components + c];
        }
        layout(i, c) = y_center[static_cast<std::size_t>(c)] + displacement;
        disp_sq += displacement * displacement;
      }

      const double max_disp = max_extrapolation * std::sqrt(std::max(layout_radius_sq, eps));
      if (disp_sq > max_disp * max_disp) {
        const double scale = max_disp / (std::sqrt(disp_sq) + eps);
        for (int c = 0; c < n_components; ++c) {
          layout(i, c) = y_center[static_cast<std::size_t>(c)] +
            (layout(i, c) - y_center[static_cast<std::size_t>(c)]) * scale;
        }
      }

      const double effective_n = weight_sq_sum > 0.0 ? (weight_sum * weight_sum) / weight_sq_sum : 1.0;
      confidence[i] = std::max(0.0, std::min(1.0, effective_n / static_cast<double>(m)));
      used_neighbors[i] = m;
      fallback[i] = 0;
    }
  });

  return List::create(
    Rcpp::Named("layout") = layout,
    Rcpp::Named("confidence") = confidence,
    Rcpp::Named("used_neighbors") = used_neighbors,
    Rcpp::Named("fallback") = fallback,
    Rcpp::Named("method") = "local_affine_knn_projection_parallel",
    Rcpp::Named("max_neighbors") = max_neighbors,
    Rcpp::Named("ridge") = ridge,
    Rcpp::Named("max_extrapolation") = max_extrapolation,
    Rcpp::Named("n_threads") = requested_threads,
    Rcpp::Named("n_threads_effective") = threads
  );
}
