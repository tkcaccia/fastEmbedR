#' Project new points into an existing embedding
#'
#' `transform_embedding()` places query observations into an existing layout
#' from their nearest reference observations. It is inspired by UMAP's
#' transform workflow, but keeps the API KNN-first: pass a query KNN object
#' when you already have one, or pass reference/query matrices and let
#' `fastEmbedR::nn()` compute it.
#'
#' The reference and query matrices must be in the same preprocessing space used
#' to fit the reference layout. For example, if the layout was fit on
#' `scale(data)`, pass scaled reference and query matrices here too.
#'
#' @param reference_layout Numeric embedding matrix for the reference
#'   observations. Rows are reference observations and columns are embedding
#'   dimensions.
#' @param knn Optional KNN list with `indices` and `distances`, usually returned
#'   by `nn(reference_data, new_data, k)`. Indices must be 1-based row numbers
#'   into `reference_layout`.
#' @param reference_data Numeric matrix/data frame of reference observations.
#'   Required only when `knn` is `NULL`.
#' @param new_data Numeric matrix/data frame of query observations. Required
#'   only when `knn` is `NULL`.
#' @param k Number of reference neighbors to use. With a supplied `knn`, this
#'   keeps the first `k` columns. With `knn = NULL`, this is passed to `nn()`.
#'   `NULL` chooses the package's automatic neighborhood size.
#' @param backend KNN backend used only when `knn` is `NULL`. Explicit GPU
#'   requests fail clearly when unavailable; they are not silently relabelled as
#'   CPU work.
#' @return A numeric matrix with one row per query observation and the same
#'   number of columns as `reference_layout`.
#' @export
transform_embedding <- function(reference_layout,
                                knn = NULL,
                                reference_data = NULL,
                                new_data = NULL,
                                k = NULL,
                                backend = c("auto", "cpu", "gpu", "cuda", "metal")) {
  backend <- match.arg(backend)
  reference_layout <- transform_embedding_matrix(
    reference_layout,
    "reference_layout",
    min_rows = 1L
  )

  if (is.null(knn)) {
    if (is.null(reference_data) || is.null(new_data)) {
      stop(
        "Supply either `knn`, or both `reference_data` and `new_data`.",
        call. = FALSE
      )
    }
    reference_data <- transform_embedding_matrix(
      reference_data,
      "reference_data",
      min_rows = 1L
    )
    new_data <- transform_embedding_matrix(new_data, "new_data", min_rows = 1L)

    if (nrow(reference_data) != nrow(reference_layout)) {
      stop(
        "`reference_data` and `reference_layout` must have the same number of rows.",
        call. = FALSE
      )
    }
    if (ncol(reference_data) != ncol(new_data)) {
      stop(
        "`reference_data` and `new_data` must have the same number of columns.",
        call. = FALSE
      )
    }

    if (is.null(k)) {
      k <- if (nrow(reference_data) == 1L) 1L else auto_k(nrow(reference_data))
    }
    k <- transform_embedding_k(k, max_k = nrow(reference_data))
    batch_size <- transform_query_batch_size(nrow(new_data), k)
    if (nrow(new_data) > batch_size) {
      layout <- matrix(NA_real_, nrow = nrow(new_data), ncol = ncol(reference_layout))
      backend_values <- character(0L)
      exact_values <- logical(0L)
      projection_backend_values <- character(0L)
      projection_backend_reasons <- character(0L)

      starts <- seq.int(1L, nrow(new_data), by = batch_size)
      for (start in starts) {
        end <- min(nrow(new_data), start + batch_size - 1L)
        rows <- start:end
        raw_knn <- nn(
          reference_data,
          new_data[rows, , drop = FALSE],
          k = k,
          backend = backend
        )
        projection <- transform_projection_knn(
          raw_knn,
          n_reference = nrow(reference_layout),
          k = k
        )
        projected <- project_transform_projection(
          reference_layout,
          projection,
          backend = backend,
          backend_used = attr(raw_knn, "backend")
        )
        layout[rows, ] <- projected$layout
        backend_values <- c(backend_values, attr(raw_knn, "backend"))
        exact_attr <- attr(raw_knn, "exact")
        if (!is.null(exact_attr)) exact_values <- c(exact_values, isTRUE(exact_attr))
        projection_backend_values <- c(projection_backend_values, projected$projection_backend)
        projection_backend_reasons <- c(projection_backend_reasons, projected$projection_backend_reason)
        rm(raw_knn, projection, projected)
        if (length(starts) > 1L) gc(FALSE)
      }

      colnames(layout) <- colnames(reference_layout)
      attr(layout, "backend") <- collapse_unique_character(backend_values, "unknown")
      attr(layout, "projection_backend") <- collapse_unique_character(projection_backend_values, "cpu")
      attr(layout, "projection_backend_reason") <- collapse_unique_character(
        projection_backend_reasons[!is.na(projection_backend_reasons) & nzchar(projection_backend_reasons)],
        NA_character_
      )
      attr(layout, "k") <- as.integer(k)
      if (length(exact_values) > 0L) attr(layout, "exact") <- all(exact_values)
      attr(layout, "transform") <- "local_membership_knn"
      return(layout)
    }
    raw_knn <- nn(reference_data, new_data, k = k, backend = backend)
    projection <- transform_projection_knn(
      raw_knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    backend_used <- attr(raw_knn, "backend")
    exact <- attr(raw_knn, "exact")
  } else {
    projection <- transform_projection_knn(
      knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    backend_used <- attr(knn, "backend")
    if (is.null(backend_used) || length(backend_used) == 0L || is.na(backend_used)) {
      backend_used <- "precomputed"
    }
    exact <- attr(knn, "exact")
  }

  projected <- project_transform_projection(
    reference_layout,
    projection,
    backend = backend,
    backend_used = backend_used
  )
  layout <- projected$layout
  colnames(layout) <- colnames(reference_layout)
  attr(layout, "backend") <- backend_used
  attr(layout, "projection_backend") <- projected$projection_backend
  attr(layout, "projection_backend_reason") <- projected$projection_backend_reason
  attr(layout, "k") <- as.integer(ncol(projection$indices))
  if (!is.null(exact)) attr(layout, "exact") <- isTRUE(exact)
  attr(layout, "transform") <- "local_membership_knn"
  layout
}

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
  if (any(!is.finite(indices)) || any(indices < 1L) || any(indices > n_reference)) {
    stop(
      "KNN `indices` must be 1-based row numbers into `reference_layout`.",
      call. = FALSE
    )
  }
  if (any(!is.finite(distances)) || any(distances < 0)) {
    stop("KNN `distances` must be finite and non-negative.", call. = FALSE)
  }

  if (!is.null(k)) {
    k <- transform_embedding_k(k, max_k = ncol(indices))
    indices <- indices[, seq_len(k), drop = FALSE]
    distances <- distances[, seq_len(k), drop = FALSE]
  }

  list(indices = indices, distances = distances)
}

transform_query_batch_size <- function(n_query, k) {
  n_query <- as.integer(n_query)
  k <- as.integer(max(1L, k))
  max_rows <- as.integer(max(512L, floor(2e6 / k)))
  min(n_query, max_rows)
}

collapse_unique_character <- function(x, default = NA_character_) {
  x <- unique(as.character(x))
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0L) return(default)
  paste(x, collapse = ";")
}

project_transform_projection <- function(reference_layout,
                                         projection,
                                         backend,
                                         backend_used) {
  projection_backend <- "cpu"
  projection_backend_reason <- NA_character_
  layout <- NULL
  request_cuda_projection <- identical(backend, "cuda") ||
    identical(backend_used, "cuda") ||
    (identical(backend, "gpu") && cuda_metric_available())
  request_metal_projection <- identical(backend, "metal") ||
    identical(backend_used, "metal")
  if (isTRUE(request_cuda_projection) && cuda_metric_available()) {
    layout <- tryCatch(
      project_embedding_knn_cuda_cpp(
        reference_layout,
        projection$indices,
        projection$distances
      ),
      error = function(e) {
        projection_backend_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(layout)) {
      projection_backend <- "cuda"
    }
  } else if (isTRUE(request_cuda_projection)) {
    projection_backend_reason <- "cuda_projection_unavailable"
  }
  if (is.null(layout) && isTRUE(request_metal_projection) && metal_metric_available()) {
    layout <- tryCatch(
      project_embedding_knn_metal_cpp(
        reference_layout,
        projection$indices,
        projection$distances
      ),
      error = function(e) {
        projection_backend_reason <<- append_metric_backend_reason(
          projection_backend_reason,
          conditionMessage(e)
        )
        NULL
      }
    )
    if (!is.null(layout)) {
      projection_backend <- "metal"
    }
  } else if (is.null(layout) && isTRUE(request_metal_projection)) {
    projection_backend_reason <- append_metric_backend_reason(
      projection_backend_reason,
      "metal_projection_unavailable"
    )
  }
  if (is.null(layout)) {
    layout <- project_embedding_knn_cpp(
      reference_layout,
      projection$indices,
      projection$distances
    )
  }
  list(
    layout = layout,
    projection_backend = projection_backend,
    projection_backend_reason = projection_backend_reason
  )
}
