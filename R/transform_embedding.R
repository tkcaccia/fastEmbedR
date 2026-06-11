transform_embedding_matrix <- function(x, name, min_rows = 1L) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) < min_rows || ncol(x) < 1L) {
    stop(
      "`", name, "` must have at least ", min_rows,
      " row(s) and one column.",
      call. = FALSE
    )
  }
  if (any(!is.finite(x))) {
    stop("`", name, "` must contain only finite values.", call. = FALSE)
  }
  x
}

transform_embedding_k <- function(k, max_k) {
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be NULL or a positive integer.", call. = FALSE)
  }
  if (k > max_k) {
    stop("`k` cannot be larger than the available reference count.", call. = FALSE)
  }
  k
}

transform_projection_knn <- function(knn, n_reference, k = NULL) {
  if (!is.list(knn) || !all(c("indices", "distances") %in% names(knn))) {
    stop("`knn` must contain `indices` and `distances`.", call. = FALSE)
  }

  indices <- knn$indices
  distances <- knn$distances
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"

  if (!identical(dim(indices), dim(distances))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(indices) < 1L || ncol(indices) < 1L) {
    stop("`knn` must have at least one row and one neighbor column.", call. = FALSE)
  }

  k <- if (is.null(k)) ncol(indices) else transform_embedding_k(k, max_k = ncol(indices))
  out <- validate_projection_knn_cpp(
    indices,
    distances,
    as.integer(n_reference),
    as.integer(k)
  )

  list(indices = out$indices, distances = out$distances)
}
