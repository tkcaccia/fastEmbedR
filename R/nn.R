#' Exact nearest neighbors from row-wise matrices
#'
#' `nn()` provides a package-native nearest-neighbor entry point compatible with
#' the common exact `nn(data, points, k)` use case. It currently performs
#' exact brute-force search in C++ with optional multi-CPU parallelism over query
#' points. Exact Euclidean search can also use native CUDA or Metal GPU
#' backends when available.
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
#' @param backend Execution backend. `"auto"` uses an available native GPU for
#'   sufficiently large Euclidean searches and otherwise uses CPU. `"gpu"`
#'   requests CUDA when available and otherwise Metal. `"cuda"` and `"metal"`
#'   request those GPU backends explicitly. `"cpu"` always uses the C++ CPU path.
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
               cores = 0L,
               backend = c("auto", "cpu", "gpu", "cuda", "metal")) {
  data <- as.matrix(data)
  points <- as.matrix(points)
  storage.mode(data) <- "double"
  storage.mode(points) <- "double"

  method <- match.arg(method, c("euclidean", "manhattan", "minkowski"))
  backend <- match.arg(backend)
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

  work_size <- as.double(nrow(data)) * as.double(nrow(points)) * as.double(ncol(data))
  choose_gpu_backend <- function() {
    if (isTRUE(cuda_available())) return("cuda")
    if (isTRUE(metal_available())) return("metal")
    "none"
  }

  selected_gpu <- "none"
  if (backend == "cuda") {
    selected_gpu <- "cuda"
  } else if (backend == "metal") {
    selected_gpu <- "metal"
  } else if (backend == "gpu") {
    selected_gpu <- choose_gpu_backend()
  } else if (backend == "auto" && method == "euclidean" && k <= 256L && work_size >= 2e7) {
    selected_gpu <- choose_gpu_backend()
  }

  if (backend == "gpu" && selected_gpu == "none") {
    stop("No native GPU backend is available on this machine.", call. = FALSE)
  }

  if (selected_gpu != "none") {
    if (method != "euclidean") {
      stop("Native GPU backends currently support only `method = \"euclidean\"`.", call. = FALSE)
    }
    if (k > 256L) {
      stop("Native GPU backends currently support `k <= 256`.", call. = FALSE)
    }
    if (selected_gpu == "cuda") {
      if (!isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
      }
      out <- nn_cuda_cpp(data, points, as.integer(k), isTRUE(square))
      attr(out, "backend") <- "cuda"
      return(out)
    }
    if (selected_gpu == "metal") {
      if (!isTRUE(metal_available())) {
        stop("No Metal GPU backend is available on this machine.", call. = FALSE)
      }
      out <- nn_metal_cpp(data, points, as.integer(k), isTRUE(square))
      attr(out, "backend") <- "metal"
      return(out)
    }
    stop("No native GPU backend is available on this machine.", call. = FALSE)
  }

  out <- nn_cpp(
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
  attr(out, "backend") <- "cpu"
  out
}

#' Check whether the native Metal backend is available
#'
#' @return `TRUE` when a Metal device is available to the package.
#' @export
metal_available <- function() {
  isTRUE(metal_available_cpp())
}

#' Check whether the native CUDA backend is available
#'
#' @return `TRUE` when the package was built with CUDA support and the CUDA
#'   runtime reports at least one available device.
#' @export
cuda_available <- function() {
  isTRUE(cuda_available_cpp())
}
