#' Fast UMAP from precomputed nearest neighbors
#'
#' @param indices Integer matrix of nearest-neighbor indices, one row per point.
#'   Indices may be 1-based, as returned by R packages, or 0-based.
#' @param distances Numeric matrix matching `indices`.
#' @param n_components Output dimensionality.
#' @param n_epochs Number of optimization epochs. Defaults to a size-aware value.
#' @param min_dist UMAP minimum distance.
#' @param spread UMAP spread parameter.
#' @param local_connectivity Local connectivity used while smoothing KNN distances.
#' @param set_op_mix_ratio Mix ratio for fuzzy union/intersection.
#' @param negative_sample_rate Number of negative samples per positive sample.
#' @param learning_rate Initial learning rate.
#' @param a,b Optional UMAP curve parameters. If `NULL`, they are estimated from
#'   `min_dist` and `spread`.
#' @param repulsion_strength Weight applied to negative samples.
#' @param init Initialization method: `"spectral"` uses a randomized eigensolver
#'   on the normalized graph; `"random"` uses uniform random coordinates.
#' @param init_sdev Optional initialization scaling. Use `NULL` for uwot's
#'   default, `"range"` for Python UMAP-compatible 0-10 range scaling, or a
#'   positive number to scale each initialized dimension to that standard
#'   deviation.
#' @param prune_epochs Remove graph edges too weak to be sampled at least once
#'   over `n_epochs`, matching UMAP/uwot's epoch scheduling more closely.
#' @param seed Integer random seed.
#' @param verbose Print progress from C++.
#' @return A numeric matrix with `nrow(indices)` rows and `n_components` columns.
#' @export
fast_knn_umap <- function(indices,
                          distances,
                          n_components = 2L,
                          n_epochs = NULL,
                          min_dist = 0.1,
                          spread = 1,
                          local_connectivity = 1,
                          set_op_mix_ratio = 1,
                          negative_sample_rate = 5L,
                          learning_rate = 1,
                          a = NULL,
                          b = NULL,
                          repulsion_strength = 1,
                          mode = c("sgd", "spectral", "hybrid"),
                          init = c("spectral", "random"),
                          init_sdev = NULL,
                          prune_epochs = TRUE,
                          spectral_n_iter = 50L,
                          seed = 42L,
                          verbose = FALSE) {
  mode <- match.arg(mode)
  init <- match.arg(init)
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)

  if (!is.integer(indices)) {
    storage.mode(indices) <- "integer"
  }
  storage.mode(distances) <- "double"

  if (!identical(dim(indices), dim(distances))) {
    stop("`indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(indices) < 2L || ncol(indices) < 1L) {
    stop("`indices` must have at least two rows and one neighbor column.", call. = FALSE)
  }
  if (any(!is.finite(distances))) {
    stop("`distances` must contain only finite values.", call. = FALSE)
  }
  if (any(distances < 0)) {
    stop("`distances` must be non-negative.", call. = FALSE)
  }

  if (is.null(n_epochs)) {
    n_epochs <- if (mode == "spectral") 0L else if (mode == "hybrid") 50L else if (nrow(indices) <= 10000L) 500L else 200L
  }

  curve_a <- if (is.null(a)) NA_real_ else as.numeric(a)
  curve_b <- if (is.null(b)) NA_real_ else as.numeric(b)
  if (length(curve_a) != 1L || length(curve_b) != 1L) {
    stop("`a` and `b` must be scalar values or NULL.", call. = FALSE)
  }

  init_sdev_mode <- "none"
  init_sdev_value <- NA_real_
  if (!is.null(init_sdev)) {
    if (is.character(init_sdev)) {
      init_sdev_mode <- match.arg(init_sdev, c("range"))
    } else {
      init_sdev_value <- as.numeric(init_sdev)
      if (length(init_sdev_value) != 1L || !is.finite(init_sdev_value) || init_sdev_value <= 0) {
        stop("`init_sdev` must be NULL, \"range\", or a positive scalar.", call. = FALSE)
      }
      init_sdev_mode <- "sd"
    }
  }

  fast_knn_umap_cpp(
    indices, distances, as.integer(n_components), as.integer(n_epochs),
    min_dist, spread, local_connectivity, set_op_mix_ratio,
    as.integer(negative_sample_rate), learning_rate, curve_a, curve_b,
    repulsion_strength, mode, init, init_sdev_mode, init_sdev_value,
    isTRUE(prune_epochs),
    as.integer(spectral_n_iter),
    as.integer(seed), isTRUE(verbose)
  )
}

#' @rdname fast_knn_umap
#' @export
umap_knn <- fast_knn_umap
