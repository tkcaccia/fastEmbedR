#' Exact nearest neighbors from row-wise matrices
#'
#' `nn()` provides a package-native nearest-neighbor entry point compatible with
#' the common `Rnanoflann::nn(data, points, k)` use case. It currently performs
#' exact brute-force search in C++ with optional multi-CPU parallelism over query
#' points. The API is intentionally shaped so faster tree, SIMD, or GPU backends
#' can be added later without changing benchmark code.
#'
#' @param data Numeric matrix of reference observations in rows.
#' @param points Numeric matrix of query observations in rows. Defaults to
#'   `data`.
#' @param k Number of neighbors to return.
#' @param method Distance metric. Supported values are `"euclidean"`,
#'   `"manhattan"`, and `"minkowski"`.
#' @param search Accepted for compatibility. Only exact `"standard"` search is
#'   currently implemented.
#' @param eps Accepted for compatibility; approximate epsilon search is not yet
#'   implemented.
#' @param square If `TRUE`, return squared Euclidean distances.
#' @param sorted If `TRUE`, return neighbors sorted by distance. The exact
#'   backend currently returns sorted neighbors in all cases.
#' @param radius Accepted for compatibility; radius search is not yet
#'   implemented.
#' @param trans Accepted for compatibility. Observations are always interpreted
#'   as rows in this implementation.
#' @param leafs Accepted for compatibility with tree-based APIs.
#' @param p Minkowski exponent when `method = "minkowski"`.
#' @param parallel Use multiple CPU threads over query points.
#' @param cores Number of CPU threads when `parallel = TRUE`. Values `<= 0`
#'   use the hardware concurrency reported by C++.
#' @return A list with integer matrix `indices` and numeric matrix `distances`.
#'   Indices are 1-based.
#' @export
nn <- function(data,
               points = data,
               k = nrow(data),
               method = "euclidean",
               search = "standard",
               eps = 0,
               square = FALSE,
               sorted = FALSE,
               radius = 0,
               trans = TRUE,
               leafs = 10L,
               p = 0,
               parallel = FALSE,
               cores = 0L) {
  data <- as.matrix(data)
  points <- as.matrix(points)
  storage.mode(data) <- "double"
  storage.mode(points) <- "double"

  method <- match.arg(method, c("euclidean", "manhattan", "minkowski"))
  if (!identical(ncol(data), ncol(points))) {
    stop("`data` and `points` must have the same number of columns.", call. = FALSE)
  }
  if (nrow(data) < 1L || nrow(points) < 1L) {
    stop("`data` and `points` must have at least one row.", call. = FALSE)
  }
  if (!is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  if (k > nrow(data)) {
    stop("`k` cannot be larger than `nrow(data)`.", call. = FALSE)
  }
  if (any(!is.finite(data)) || any(!is.finite(points))) {
    stop("`data` and `points` must contain only finite values.", call. = FALSE)
  }
  if (search != "standard") {
    stop("Only `search = \"standard\"` is currently implemented.", call. = FALSE)
  }
  if (eps != 0) {
    stop("Approximate epsilon search is not currently implemented.", call. = FALSE)
  }
  if (radius != 0) {
    stop("Radius search is not currently implemented.", call. = FALSE)
  }
  if (method == "minkowski" && (!is.finite(p) || p <= 0)) {
    stop("`p` must be positive when `method = \"minkowski\"`.", call. = FALSE)
  }

  nn_cpp(
    data,
    points,
    as.integer(k),
    method,
    isTRUE(square),
    isTRUE(sorted),
    as.numeric(p),
    isTRUE(parallel),
    as.integer(cores)
  )
}
