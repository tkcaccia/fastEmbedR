#' Choose a default neighborhood size
#'
#' @param x A data matrix/data frame, or an integer row count.
#' @param include_self If `TRUE`, return the value to request from `nn()` when
#'   the query points are the data itself. If `FALSE`, return the number of
#'   non-self neighbors used by embedding functions.
#' @return An integer neighborhood size.
auto_k <- function(x, include_self = FALSE) {
  n <- if (length(x) == 1L && is.numeric(x)) {
    as.integer(x)
  } else {
    nrow(x)
  }
  if (length(n) != 1L || is.na(n) || !is.finite(n) || n < 2L) {
    stop("`x` must describe at least two observations.", call. = FALSE)
  }

  k <- if (n < 500L) {
    15L
  } else if (n < 10000L) {
    30L
  } else {
    50L
  }
  k <- max(1L, min(k, n - 1L))
  if (isTRUE(include_self)) k + 1L else k
}

auto_embedding_k <- function(x, method = "umap", include_self = FALSE) {
  n <- if (length(x) == 1L && is.numeric(x)) {
    as.integer(x)
  } else {
    nrow(x)
  }
  auto_k(n, include_self = include_self)
}

prepare_embedding_data <- function(data,
                                   standardize,
                                   pca_dims,
                                   seed,
                                   backend = "cpu") {
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  if (nrow(x) < 2L || ncol(x) < 1L) {
    stop("`data` must have at least two rows and one column.", call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }

  preprocess <- list(
    standardize = isTRUE(standardize),
    pca_dims = NA_integer_,
    standardize_backend = if (isTRUE(standardize)) "cpu" else "none",
    pca_backend = "none",
    pca_method = "none",
    pca_oversample = NA_integer_,
    pca_power = NA_integer_,
    pca_backend_reason = NA_character_,
    preprocess_backend = if (isTRUE(standardize)) "cpu" else "none",
    preprocess_backend_reason = NA_character_
  )
  if (isTRUE(standardize)) {
    used_native <- FALSE
    if (identical(backend, "cuda") && cuda_metric_available()) {
      standardized <- tryCatch(
        standardize_cuda_cpp(x),
        error = function(e) {
          preprocess$preprocess_backend_reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(standardized)) {
        x <- standardized$data
        used_native <- TRUE
        preprocess$standardize_backend <- "cuda"
        preprocess$preprocess_backend <- "cuda"
      }
    } else if (identical(backend, "cuda")) {
      preprocess$preprocess_backend_reason <- "cuda_preprocessing_unavailable"
    } else if (identical(backend, "metal") && metal_metric_available()) {
      standardized <- tryCatch(
        standardize_metal_cpp(x),
        error = function(e) {
          preprocess$preprocess_backend_reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(standardized)) {
        x <- standardized$data
        used_native <- TRUE
        preprocess$standardize_backend <- "metal"
        preprocess$preprocess_backend <- "metal"
      }
    } else if (identical(backend, "metal")) {
      preprocess$preprocess_backend_reason <- "metal_preprocessing_unavailable"
    }
    if (!used_native) {
      center <- colMeans(x)
      x <- sweep(x, 2L, center, "-")
      scale <- sqrt(colSums(x * x) / max(1L, nrow(x) - 1L))
      scale[!is.finite(scale) | scale == 0] <- 1
      x <- sweep(x, 2L, scale, "/")
      preprocess$standardize_backend <- "cpu"
      preprocess$preprocess_backend <- "cpu"
    }
  }

  if (!is.null(pca_dims)) {
    pca_dims <- as.integer(pca_dims)
    if (length(pca_dims) != 1L || is.na(pca_dims) || !is.finite(pca_dims) || pca_dims < 1L) {
      stop("`pca_dims` must be NULL or a positive integer.", call. = FALSE)
    }
    rank <- min(pca_dims, nrow(x) - 1L, ncol(x))
    if (rank >= 1L && rank < ncol(x)) {
      pca <- fastpls_rsvd_pca_scores(
        x,
        rank = rank,
        seed = seed,
        backend = backend
      )
      x <- pca$scores
      preprocess$pca_dims <- as.integer(ncol(x))
      preprocess$pca_backend <- pca$backend
      preprocess$pca_method <- pca$method
      preprocess$pca_oversample <- pca$oversample
      preprocess$pca_power <- pca$power
      preprocess$pca_backend_reason <- pca$backend_reason
      preprocess$preprocess_backend <- combine_preprocess_backends(
        preprocess$standardize_backend,
        pca$backend
      )
    }
  }
  list(data = x, preprocess = preprocess)
}

combine_preprocess_backends <- function(standardize_backend, pca_backend) {
  standardize_backend <- if (is.null(standardize_backend)) "none" else standardize_backend
  pca_backend <- if (is.null(pca_backend)) "none" else pca_backend
  if (identical(standardize_backend, "none")) {
    return(pca_backend)
  }
  if (identical(pca_backend, "none")) {
    return(standardize_backend)
  }
  if (identical(standardize_backend, pca_backend)) {
    return(standardize_backend)
  }
  paste(standardize_backend, pca_backend, sep = "_")
}

fastpls_rsvd_tuning <- function(n, p, rank, backend) {
  backend <- if (is.null(backend)) "cpu" else backend
  if (identical(backend, "cuda")) {
    oversample <- if (n * p >= 5e6) 16L else 10L
    power <- if (rank <= 20L || p <= 128L) 2L else 1L
  } else if (identical(backend, "metal")) {
    oversample <- if (n * p >= 5e6) 16L else 10L
    power <- if (rank <= 30L) 1L else 0L
  } else {
    oversample <- if (rank <= 10L) 10L else min(20L, max(10L, ceiling(rank / 2)))
    power <- if (rank <= 30L) 1L else 0L
  }
  list(
    oversample = as.integer(max(0L, oversample)),
    power = as.integer(max(0L, power))
  )
}

fastpls_rsvd_pca_scores <- function(x,
                                    rank,
                                    seed,
                                    backend = "cpu") {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  n <- nrow(x)
  p <- ncol(x)
  max_rank <- min(n, p)
  rank <- max(1L, min(as.integer(rank), max_rank))
  backend_requested <- if (identical(backend, "gpu")) {
    resolve_backend_request(backend, need_embedding = TRUE)
  } else if (identical(backend, "auto")) {
    "cpu"
  } else {
    backend
  }
  tuning <- fastpls_rsvd_tuning(n, p, rank, backend_requested)
  sketch_rank <- min(max_rank, rank + tuning$oversample)

  exact_reason <- NA_character_
  if (sketch_rank >= max_rank) {
    exact <- svd(x, nu = rank, nv = rank)
    scores <- exact$u[, seq_len(rank), drop = FALSE]
    scores <- sweep(scores, 2L, exact$d[seq_len(rank)], "*")
    loadings <- exact$v[, seq_len(rank), drop = FALSE]
    scores <- orient_pca_scores(scores, loadings)
    colnames(scores) <- paste0("PC", seq_len(ncol(scores)))
    return(list(
      scores = scores,
      loadings = attr(scores, "loadings"),
      singular_values = exact$d[seq_len(rank)],
      backend = "cpu_exact",
      method = "exact",
      oversample = as.integer(tuning$oversample),
      power = as.integer(tuning$power),
      backend_reason = "sketch_rank_reaches_matrix_rank"
    ))
  }

  run_backend <- function(selected_backend) {
    fastpls_rsvd_pca_scores_backend(
      x,
      rank = rank,
      sketch_rank = sketch_rank,
      seed = seed,
      power = tuning$power,
      backend = selected_backend
    )
  }

  native_reason <- NA_character_
  if (backend_requested %in% c("cuda", "metal")) {
    native <- tryCatch(
      run_backend(backend_requested),
      error = function(e) {
        native_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(native)) {
      return(c(native, list(backend_reason = NA_character_)))
    }
  }

  cpu <- run_backend("cpu")
  if (!is.na(native_reason)) {
    cpu$backend_reason <- paste0(backend_requested, "_rsvd_unavailable: ", native_reason)
  } else if (!identical(backend_requested, "cpu")) {
    cpu$backend_reason <- paste0(backend_requested, "_rsvd_not_requested")
  } else {
    cpu$backend_reason <- NA_character_
  }
  cpu
}

fastpls_rsvd_pca_scores_backend <- function(x,
                                            rank,
                                            sketch_rank,
                                            seed,
                                            power,
                                            backend) {
  set.seed(as.integer(seed))
  omega <- matrix(stats::rnorm(ncol(x) * sketch_rank), nrow = ncol(x), ncol = sketch_rank)
  multiply <- function(left, right, transpose_left = FALSE) {
    rsvd_matrix_multiply(left, right, transpose_left = transpose_left, backend = backend)
  }

  y <- multiply(x, omega)
  power <- max(0L, as.integer(power))
  if (power == 1L) {
    y <- multiply(x, multiply(x, y, transpose_left = TRUE))
  } else if (power > 1L) {
    for (i in seq_len(power)) {
      z <- multiply(x, y, transpose_left = TRUE)
      qz <- qr.Q(qr(z))
      y <- multiply(x, qz)
    }
  }

  q <- qr.Q(qr(y))
  b <- multiply(q, x, transpose_left = TRUE)
  small <- svd(b, nu = rank, nv = rank)
  usable <- min(rank, length(small$d), ncol(small$u), ncol(small$v))
  if (usable < 1L) {
    stop("RSVD PCA produced no usable singular vectors.", call. = FALSE)
  }

  u <- q %*% small$u[, seq_len(usable), drop = FALSE]
  scores <- sweep(u, 2L, small$d[seq_len(usable)], "*")
  loadings <- small$v[, seq_len(usable), drop = FALSE]
  scores <- orient_pca_scores(scores, loadings)
  colnames(scores) <- paste0("PC", seq_len(ncol(scores)))

  list(
    scores = scores,
    loadings = attr(scores, "loadings"),
    singular_values = small$d[seq_len(usable)],
    backend = paste0(backend, "_rsvd"),
    method = "rsvd",
    oversample = as.integer(sketch_rank - rank),
    power = as.integer(power)
  )
}

rsvd_matrix_multiply <- function(left,
                                 right,
                                 transpose_left = FALSE,
                                 backend = "cpu") {
  left <- as.matrix(left)
  right <- as.matrix(right)
  storage.mode(left) <- "double"
  storage.mode(right) <- "double"
  if (isTRUE(transpose_left)) {
    if (nrow(left) != nrow(right)) {
      stop("Non-conformable RSVD cross-product.", call. = FALSE)
    }
  } else if (ncol(left) != nrow(right)) {
    stop("Non-conformable RSVD matrix multiply.", call. = FALSE)
  }

  if (identical(backend, "cuda")) {
    if (!cuda_metric_available()) {
      stop("CUDA RSVD matrix multiply is not available.", call. = FALSE)
    }
    return(rsvd_multiply_cuda_cpp(left, right, isTRUE(transpose_left)))
  }
  if (identical(backend, "metal")) {
    if (!metal_metric_available()) {
      stop("Metal RSVD matrix multiply is not available.", call. = FALSE)
    }
    return(rsvd_multiply_metal_cpp(left, right, isTRUE(transpose_left)))
  }

  if (isTRUE(transpose_left)) {
    crossprod(left, right)
  } else {
    left %*% right
  }
}

orient_pca_scores <- function(scores, loadings) {
  for (j in seq_len(ncol(scores))) {
    pivot <- which.max(abs(loadings[, j]))
    if (length(pivot) == 1L && is.finite(loadings[pivot, j]) && loadings[pivot, j] < 0) {
      scores[, j] <- -scores[, j]
      loadings[, j] <- -loadings[, j]
    }
  }
  attr(scores, "loadings") <- loadings
  scores
}

normalize_supplied_knn <- function(nn, n, n_neighbors = NULL, keep_self = FALSE) {
  if (!all(c("indices", "distances") %in% names(nn))) {
    stop("`nn` must contain `indices` and `distances`.", call. = FALSE)
  }
  indices <- nn$indices
  distances <- nn$distances
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  if (!identical(dim(indices), dim(distances))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(indices) != n) {
    stop("KNN matrix row count must match `nrow(data)`.", call. = FALSE)
  }
  if (ncol(indices) < 1L) {
    stop("KNN matrices must have at least one neighbor column.", call. = FALSE)
  }
  if (any(!is.finite(distances)) || any(distances < 0)) {
    stop("KNN `distances` must be finite and non-negative.", call. = FALSE)
  }

  stripped <- strip_self_neighbors(indices, distances)
  has_self <- stripped$has_self
  knn_with_self <- if (isTRUE(keep_self) && has_self) {
    list(indices = indices, distances = distances)
  } else {
    NULL
  }
  if (has_self) {
    indices <- stripped$indices
    distances <- stripped$distances
  }
  if (ncol(indices) < 1L) {
    stop("KNN matrices must contain at least one non-self neighbor.", call. = FALSE)
  }

  if (is.null(n_neighbors)) {
    n_neighbors <- ncol(indices)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || !is.finite(n_neighbors) || n_neighbors < 1L) {
      stop("`n_neighbors` must be NULL or a positive integer.", call. = FALSE)
    }
    if (n_neighbors > ncol(indices)) {
      stop("`n_neighbors` is larger than the supplied KNN width.", call. = FALSE)
    }
    indices <- indices[, seq_len(n_neighbors), drop = FALSE]
    distances <- distances[, seq_len(n_neighbors), drop = FALSE]
  }
  list(
    indices = indices,
    distances = distances,
    n_neighbors = as.integer(n_neighbors),
    has_self = isTRUE(has_self),
    knn_with_self = knn_with_self
  )
}

embedding_scores <- function(layout,
                             labels,
                             indices,
                             silhouette_sample,
                             preserve_sample,
                             preserve_k,
                             seed,
                             preserve_keep = NULL,
                             backend = "cpu") {
  n <- nrow(layout)
  labels_factor <- if (is.null(labels)) NULL else as.factor(labels)
  labels_int <- if (is.null(labels_factor)) NULL else as.integer(labels_factor)
  n_label_levels <- if (is.null(labels_factor)) 0L else length(levels(labels_factor))
  silhouette_keep <- sample_indices(n, silhouette_sample, seed)
  if (is.null(preserve_keep)) {
    preserve_keep <- sample_indices(n, preserve_sample, seed)
  }
  preserve_k <- if (is.null(preserve_k)) ncol(indices) else min(as.integer(preserve_k), ncol(indices))

  silhouette_result <- if (is.null(labels_int) || length(silhouette_keep) == 0L) {
    list(value = NA_real_, backend = if (is.null(labels_int)) "none" else "skipped", reason = NA_character_)
  } else {
    silhouette_score_with_backend(
      labels_int[silhouette_keep],
      layout[silhouette_keep, , drop = FALSE],
      n_label_levels,
      backend = backend
    )
  }
  structure_result <- structure_score_with_backend(
    layout,
    indices,
    preserve_keep,
    preserve_k,
    if (is.null(labels_int)) integer(0L) else labels_int,
    n_label_levels,
    backend = backend
  )
  structure <- structure_result$values
  out <- data.frame(
    silhouette = silhouette_result$value,
    knn_preservation = unname(structure["knn_preservation"]),
    local_trustworthiness = unname(structure["local_trustworthiness"]),
    local_continuity = unname(structure["local_continuity"]),
    structure_score = unname(structure["structure_score"]),
    embedding_knn_accuracy = unname(structure["embedding_knn_accuracy"]),
    stringsAsFactors = FALSE
  )
  attr(out, "structure_backend") <- structure_result$backend
  attr(out, "structure_backend_reason") <- structure_result$reason
  attr(out, "silhouette_backend") <- silhouette_result$backend
  attr(out, "silhouette_backend_reason") <- silhouette_result$reason
  attr(out, "backend") <- paste(
    paste0("structure:", structure_result$backend),
    paste0("silhouette:", silhouette_result$backend),
    sep = ";"
  )
  out
}

auto_landmark_count <- function(n) {
  if (n <= 3L) {
    return(n)
  }
  if (n <= 80L) {
    return(as.integer(min(n - 1L, max(2L, ceiling(n * 0.5)))))
  }
  if (n <= 1000L) {
    return(as.integer(min(n - 1L, max(80L, ceiling(sqrt(n) * 7)))))
  }
  as.integer(min(n - 1L, max(300L, ceiling(sqrt(n) * 7))))
}

resolve_landmarks <- function(landmarks, x, seed) {
  n <- nrow(x)
  if (is.null(landmarks) || identical(landmarks, FALSE)) {
    return(NULL)
  }

  if (identical(landmarks, TRUE)) {
    count <- auto_landmark_count(n)
    return(select_landmark_rows(x, count, seed))
  }

  if (length(landmarks) == 1L && is.numeric(landmarks)) {
    value <- as.numeric(landmarks)
    if (!is.finite(value) || value <= 0) {
      stop("`landmarks` must be NULL, TRUE, a positive count, a fraction in (0, 1), or row indices.", call. = FALSE)
    }
    count <- if (value > 0 && value < 1) {
      ceiling(n * value)
    } else {
      as.integer(round(value))
    }
    if (count < 2L) {
      stop("Landmark mode requires at least two landmarks.", call. = FALSE)
    }
    if (count >= n) {
      return(NULL)
    }
    return(select_landmark_rows(x, count, seed))
  }

  if (!is.numeric(landmarks)) {
    stop("`landmarks` must be NULL, TRUE, a positive count, a fraction in (0, 1), or row indices.", call. = FALSE)
  }
  idx <- sort(unique(as.integer(landmarks)))
  if (length(idx) < 2L || any(is.na(idx)) || any(idx < 1L) || any(idx > n)) {
    stop("Landmark row indices must contain at least two valid row numbers.", call. = FALSE)
  }
  if (length(idx) >= n) {
    return(NULL)
  }
  idx
}

select_landmark_rows <- function(x, count, seed) {
  n <- nrow(x)
  count <- as.integer(min(max(2L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z <- landmark_selection_features(x, seed)
  if (count <= 2000L) {
    candidate_count <- min(n, max(count, min(12000L, max(500L, 4L * count))))
    candidates <- projection_quantile_rows(z, candidate_count, seed)
    picked <- farthest_landmark_subset(z[candidates, , drop = FALSE], count)
    selected <- candidates[picked]
    method <- "projected_farthest"
  } else {
    selected <- projection_quantile_rows(z, count, seed)
    method <- "multi_projection_quantiles"
  }
  selected <- sort(unique(as.integer(selected)))
  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  selected <- sort(selected[seq_len(count)])
  attr(selected, "selection_method") <- method
  selected
}

landmark_selection_features <- function(x, seed) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  direct <- x[, seq_len(min(4L, p)), drop = FALSE]
  n_random <- min(4L, p)

  set.seed(seed)
  directions <- matrix(stats::rnorm(p * n_random), nrow = p, ncol = n_random)
  norms <- sqrt(colSums(directions * directions))
  norms[!is.finite(norms) | norms == 0] <- 1
  directions <- sweep(directions, 2L, norms, "/")
  z <- cbind(direct, x %*% directions)
  z <- as.matrix(z)
  storage.mode(z) <- "double"

  center <- colMeans(z)
  z <- sweep(z, 2L, center, "-")
  scale <- sqrt(colSums(z * z) / max(1L, n - 1L))
  keep <- is.finite(scale) & scale > 0
  if (!any(keep)) {
    return(matrix(seq_len(n), ncol = 1L))
  }
  sweep(z[, keep, drop = FALSE], 2L, scale[keep], "/")
}

projection_quantile_rows <- function(z, count, seed) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  n_axes <- ncol(z)
  per_axis <- max(2L, ceiling(1.25 * count / max(1L, n_axes)))
  selected <- integer(0)

  center_order <- order(rowSums(z * z), seq_len(n))
  selected <- c(selected, center_order[1L])
  for (axis in seq_len(n_axes)) {
    ordered <- order(z[, axis], seq_len(n))
    positions <- unique(round(seq(1, n, length.out = per_axis)))
    selected <- c(selected, ordered[positions])
  }
  selected <- unique(as.integer(selected))

  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  if (length(selected) > count) {
    ordered <- order(z[selected, 1L], selected)
    positions <- unique(round(seq(1, length(selected), length.out = count)))
    selected <- selected[ordered[positions]]
  }
  sort(unique(as.integer(selected)))[seq_len(count)]
}

farthest_landmark_subset <- function(z, count) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z_norm <- rowSums(z * z)
  selected <- integer(count)
  selected[1L] <- which.min(z_norm)
  min_dist <- rep(Inf, n)

  for (i in seq_len(count)) {
    if (i > 1L) {
      selected[i] <- which.max(min_dist)
    }
    center <- z[selected[i], , drop = FALSE]
    dist <- z_norm + z_norm[selected[i]] - 2 * drop(z %*% t(center))
    min_dist <- pmin(min_dist, pmax(0, dist))
    min_dist[selected[seq_len(i)]] <- -Inf
  }
  selected
}

fill_landmark_rows <- function(selected, n, count, seed) {
  selected <- unique(as.integer(selected))
  if (length(selected) >= count) {
    return(selected[seq_len(count)])
  }
  set.seed(seed + 1009L)
  remaining <- setdiff(seq_len(n), selected)
  need <- count - length(selected)
  c(selected, sort(sample(remaining, need)))
}

landmark_projection_k <- function(n_landmarks, n_neighbors) {
  as.integer(min(n_landmarks, max(2L, min(50L, ceiling(1.5 * max(5L, as.integer(n_neighbors)))))))
}

landmark_refinement_mode <- function(mode, use_landmarks) {
  if (!isTRUE(use_landmarks)) {
    return("none")
  }
  "bucketed"
}

landmark_refinement_epoch_count <- function(n, refinement, method) {
  if (identical(refinement, "none")) {
    return(0L)
  }
  base <- if (n < 10000L) 60L else 40L
  as.integer(base)
}

landmark_bucketed_graph_columns <- function(projection_k, n_neighbors) {
  projection_k <- as.integer(max(1L, projection_k))
  n_neighbors <- as.integer(max(1L, n_neighbors))
  bucket_cols <- min(projection_k, max(3L, min(6L, ceiling(n_neighbors / 2.5))))
  query_cols <- min(projection_k, max(bucket_cols, min(10L, ceiling(n_neighbors / 2))))
  list(bucket_cols = bucket_cols, query_cols = query_cols)
}

landmark_bucketed_knn_graph <- function(x,
                                        projection_nn,
                                        n_neighbors,
                                        backend = "cpu",
                                        update_rows = NULL) {
  cols <- landmark_bucketed_graph_columns(ncol(projection_nn$indices), n_neighbors)
  backend <- if (identical(backend, "cuda")) "cuda" else "cpu"
  backend_reason <- NA_character_
  update_rows <- if (is.null(update_rows)) {
    NULL
  } else {
    sort(unique(as.integer(update_rows)))
  }
  if (!is.null(update_rows)) {
    update_rows <- update_rows[update_rows >= 1L & update_rows <= nrow(x)]
    if (!length(update_rows)) {
      return(list(
        indices = matrix(integer(0L), nrow = 0L, ncol = as.integer(n_neighbors)),
        distances = matrix(numeric(0L), nrow = 0L, ncol = as.integer(n_neighbors)),
        row_ids = integer(0L),
        backend = "skipped_landmark_buckets_subset",
        backend_reason = NA_character_
      ))
    }
  }
  out <- NULL
  if (identical(backend, "cuda") && is.null(update_rows) && cuda_available()) {
    out <- tryCatch(
      landmark_candidate_knn_cuda_cpp(
        x,
        projection_nn$indices,
        as.integer(n_neighbors),
        as.integer(cols$bucket_cols),
        as.integer(cols$query_cols)
      ),
      error = function(e) {
        backend_reason <<- conditionMessage(e)
        NULL
      }
    )
  } else if (identical(backend, "cuda")) {
    backend_reason <- "cuda_landmark_candidate_knn_unavailable"
    if (!is.null(update_rows)) {
      backend_reason <- "cuda_landmark_candidate_knn_subset_unavailable"
    }
  }
  if (!is.null(out)) {
    storage.mode(out$indices) <- "integer"
    if (!identical(typeof(out$distances), "double")) storage.mode(out$distances) <- "double"
    return(list(
      indices = out$indices,
      distances = out$distances,
      backend = "cuda_landmark_buckets",
      backend_reason = NA_character_
    ))
  }

  cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    cores <- 1L
  }
  if (is.null(update_rows)) {
    out <- landmark_candidate_knn_cpp(
      x,
      projection_nn$indices,
      as.integer(n_neighbors),
      as.integer(cols$bucket_cols),
      as.integer(cols$query_cols),
      TRUE,
      as.integer(max(1L, min(4L, cores)))
    )
  } else {
    out <- landmark_candidate_knn_subset_cpp(
      x,
      projection_nn$indices,
      as.integer(update_rows),
      as.integer(n_neighbors),
      as.integer(cols$bucket_cols),
      as.integer(cols$query_cols),
      TRUE,
      as.integer(max(1L, min(4L, cores)))
    )
  }
  storage.mode(out$indices) <- "integer"
  if (!identical(typeof(out$distances), "double")) storage.mode(out$distances) <- "double"
  list(
    indices = out$indices,
    distances = out$distances,
    row_ids = out$row_ids,
    backend = if (is.null(update_rows)) "cpu_landmark_buckets" else "cpu_landmark_buckets_subset",
    backend_reason = backend_reason
  )
}

refine_embedding_from_knn <- function(method,
                                      indices,
                                      distances,
                                      init_layout,
                                      n_epochs,
                                      refinement,
                                      seed,
                                      backend,
                                      verbose,
                                      update_rows = NULL,
                                      row_ids = NULL) {
  n_components <- ncol(init_layout)
  if (n_epochs < 1L) {
    return(init_layout)
  }
  if (!identical(method, "umap")) {
    stop("Only UMAP refinement is supported.", call. = FALSE)
  }
  cfg <- fast_knn_umap_config(nrow(init_layout), ncol(indices), backend)
  cfg$n_epochs <- as.integer(n_epochs)
  cfg$negative_sample_rate <- min(cfg$negative_sample_rate, 3L)
  cfg$learning_rate <- min(cfg$learning_rate, 0.12)
  if (!is.null(row_ids)) {
    row_ids <- as.integer(row_ids)
    if (!length(row_ids)) {
      layout <- init_layout
      layout <- set_embedding_colnames(layout, "UMAP")
      attr(layout, "fastEmbedR_config") <- c(
        cfg,
        list(
          refinement_backend = "skipped_rows",
          refinement_update_rows = 0L,
          refinement_update_fraction = 0
        )
      )
      return(layout)
    }
    layout <- knn_umap_refine_rows_cpp(
      indices,
      distances,
      row_ids,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(seed),
      isTRUE(verbose)
    )
    refinement_backend <- "cpu_masked_rows"
    update_rows <- row_ids
  } else if (!is.null(update_rows)) {
    update_rows <- sort(unique(as.integer(update_rows)))
    update_rows <- update_rows[update_rows >= 1L & update_rows <= nrow(init_layout)]
    if (!length(update_rows)) {
      layout <- init_layout
      layout <- set_embedding_colnames(layout, "UMAP")
      attr(layout, "fastEmbedR_config") <- c(
        cfg,
        list(
          refinement_backend = "skipped_masked",
          refinement_update_rows = 0L,
          refinement_update_fraction = 0
        )
      )
      return(layout)
    }
    layout <- knn_umap_refine_masked_cpp(
      indices,
      distances,
      init_layout,
      as.integer(update_rows),
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(seed),
      isTRUE(verbose)
    )
    refinement_backend <- "cpu_masked"
  } else {
    layout <- run_native_knn_optimizer(
      cfg$backend,
      indices,
      distances,
      init_layout,
      "umap",
      cfg$n_epochs,
      cfg$negative_sample_rate,
      cfg$learning_rate,
      cfg$min_dist,
      seed
    )
    refinement_backend <- cfg$backend
    if (is.null(layout)) {
      layout <- knn_umap_refine_cpp(
        indices,
        distances,
        init_layout,
        as.integer(cfg$n_epochs),
        cfg$min_dist,
        as.integer(cfg$negative_sample_rate),
        cfg$learning_rate,
        cfg$repulsion_strength,
        as.integer(cfg$n_threads),
        as.integer(seed),
        isTRUE(verbose)
      )
      refinement_backend <- "cpu"
    }
  }
  layout <- set_embedding_colnames(layout, "UMAP")
  attr(layout, "fastEmbedR_config") <- c(
    cfg,
    list(
      refinement_backend = refinement_backend,
      refinement_update_rows = if (is.null(update_rows)) nrow(init_layout) else length(update_rows),
      refinement_update_fraction = if (is.null(update_rows)) 1 else length(update_rows) / nrow(init_layout)
    )
  )
  layout
}

interpolate_landmark_layout <- function(landmark_layout,
                                        landmark_indices,
                                        projection_nn,
                                        n,
                                        backend = "cpu") {
  backend_used <- "cpu"
  backend_reason <- NA_character_
  layout <- NULL
  if (identical(backend, "cuda") && cuda_metric_available()) {
    layout <- tryCatch(
      interpolate_landmark_layout_cuda_cpp(
        landmark_layout,
        as.integer(landmark_indices),
        projection_nn$indices,
        projection_nn$distances,
        as.integer(n)
      ),
      error = function(e) {
        backend_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(layout)) {
      backend_used <- "cuda"
    }
  } else if (identical(backend, "cuda")) {
    backend_reason <- "cuda_interpolation_unavailable"
  } else if (identical(backend, "metal") && metal_metric_available()) {
    layout <- tryCatch(
      interpolate_landmark_layout_metal_cpp(
        landmark_layout,
        as.integer(landmark_indices),
        projection_nn$indices,
        projection_nn$distances,
        as.integer(n)
      ),
      error = function(e) {
        backend_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(layout)) {
      backend_used <- "metal"
    }
  } else if (identical(backend, "metal")) {
    backend_reason <- "metal_interpolation_unavailable"
  }
  if (is.null(layout)) {
    layout <- interpolate_landmark_layout_cpp(
      landmark_layout,
      as.integer(landmark_indices),
      projection_nn$indices,
      projection_nn$distances,
      as.integer(n)
    )
  }
  colnames(layout) <- colnames(landmark_layout)
  attr(layout, "interpolation_backend") <- backend_used
  attr(layout, "interpolation_backend_reason") <- backend_reason
  layout
}

fused_landmark_project_layout <- function(x_landmarks,
                                          x,
                                          landmark_layout,
                                          landmark_indices,
                                          projection_k,
                                          backend = "cpu") {
  if (!identical(backend, "metal") || !metal_metric_available()) {
    return(NULL)
  }
  backend_reason <- NA_character_
  layout <- tryCatch(
    landmark_project_interpolate_metal_cpp(
      x_landmarks,
      x,
      landmark_layout,
      as.integer(landmark_indices),
      as.integer(projection_k)
    ),
    error = function(e) {
      backend_reason <<- conditionMessage(e)
      NULL
    }
  )
  if (is.null(layout)) {
    return(NULL)
  }
  colnames(layout) <- colnames(landmark_layout)
  attr(layout, "projection_backend") <- "metal_fused"
  attr(layout, "interpolation_backend") <- "metal_fused"
  attr(layout, "interpolation_backend_reason") <- backend_reason
  layout
}

fused_landmark_project_knn_confidence <- function(x_landmarks,
                                                  x,
                                                  landmark_layout,
                                                  landmark_indices,
                                                  projection_k,
                                                  backend = "cpu") {
  if (!(identical(backend, "cuda") && cuda_metric_available()) &&
      !(identical(backend, "metal") && metal_metric_available())) {
    return(NULL)
  }
  backend_reason <- NA_character_
  out <- if (identical(backend, "cuda")) {
    tryCatch(
      landmark_project_interpolate_knn_confidence_cuda_cpp(
        x_landmarks,
        x,
        landmark_layout,
        as.integer(landmark_indices),
        as.integer(projection_k)
      ),
      error = function(e) {
        backend_reason <<- conditionMessage(e)
        NULL
      }
    )
  } else {
    tryCatch(
      landmark_project_interpolate_knn_confidence_metal_cpp(
        x_landmarks,
        x,
        landmark_layout,
        as.integer(landmark_indices),
        as.integer(projection_k)
      ),
      error = function(e) {
        backend_reason <<- conditionMessage(e)
        NULL
      }
    )
  }
  if (is.null(out)) {
    return(NULL)
  }
  layout <- out$layout
  colnames(layout) <- colnames(landmark_layout)
  projection_nn <- list(indices = out$indices, distances = out$distances)
  storage.mode(projection_nn$indices) <- "integer"
  if (!identical(typeof(projection_nn$distances), "double")) {
    storage.mode(projection_nn$distances) <- "double"
  }
  attr(projection_nn, "backend") <- paste0(backend, "_fused_knn_confidence")
  attr(projection_nn, "exact") <- TRUE
  attr(projection_nn, "confidence") <- as.numeric(out$confidence)
  attr(layout, "projection_backend") <- paste0(backend, "_fused_knn_confidence")
  attr(layout, "interpolation_backend") <- paste0(backend, "_fused")
  attr(layout, "interpolation_backend_reason") <- backend_reason
  list(layout = layout, projection_nn = projection_nn)
}

landmark_projection_approx_params <- function(n_landmarks, n, p, k) {
  n_projections <- as.integer(min(24L, max(12L, ceiling(log2(max(2L, n_landmarks))))))
  window <- as.integer(min(n_landmarks, max(128L, min(512L, ceiling(5 * k)))))
  if (n_landmarks >= 20000L) {
    n_projections <- max(n_projections, 24L)
    window <- max(window, min(n_landmarks, 512L))
  } else if (n_landmarks >= 10000L) {
    n_projections <- max(n_projections, 18L)
    window <- max(window, min(n_landmarks, 384L))
  }
  env_projections <- suppressWarnings(as.integer(Sys.getenv(
    "FASTEMBEDR_LANDMARK_PROJECTION_PROJECTIONS",
    ""
  )))
  env_window <- suppressWarnings(as.integer(Sys.getenv(
    "FASTEMBEDR_LANDMARK_PROJECTION_WINDOW",
    ""
  )))
  if (length(env_projections) == 1L && is.finite(env_projections) && env_projections > 0L) {
    n_projections <- as.integer(min(64L, max(1L, env_projections)))
  }
  if (length(env_window) == 1L && is.finite(env_window) && env_window > 0L) {
    window <- as.integer(min(n_landmarks, max(1L, env_window)))
  }
  list(n_projections = n_projections, window = window)
}

should_use_approx_landmark_projection <- function(x_landmarks, x, projection_k) {
  n_landmarks <- nrow(x_landmarks)
  n <- nrow(x)
  p <- ncol(x)
  work_size <- as.double(n_landmarks) * as.double(n) * as.double(p)
  n_landmarks >= 5000L &&
    n >= 10000L &&
    p >= 2L &&
    projection_k <= 128L &&
    work_size >= 2e9
}

approx_landmark_projection_knn <- function(x_landmarks,
                                           x,
                                           projection_k,
                                           seed) {
  if (!should_use_approx_landmark_projection(x_landmarks, x, projection_k)) {
    return(NULL)
  }
  params <- landmark_projection_approx_params(
    nrow(x_landmarks),
    nrow(x),
    ncol(x),
    projection_k
  )
  cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    cores <- 1L
  }
  out <- tryCatch(
    landmark_projection_knn_approx_cpp(
      x_landmarks,
      x,
      as.integer(projection_k),
      as.integer(params$n_projections),
      as.integer(params$window),
      as.integer(seed),
      TRUE,
      as.integer(max(1L, min(4L, cores)))
    ),
    error = function(e) NULL
  )
  if (is.null(out)) {
    return(NULL)
  }
  storage.mode(out$indices) <- "integer"
  if (!identical(typeof(out$distances), "double")) {
    storage.mode(out$distances) <- "double"
  }
  result <- list(indices = out$indices, distances = out$distances)
  attr(result, "backend") <- "cpu_projection_approx"
  attr(result, "exact") <- FALSE
  attr(result, "approximation") <- list(
    method = "projection_window",
    n_projections = out$n_projections,
    window = out$window
  )
  result
}

use_approx_landmark_projection <- function(knn_backend) {
  if (identical(knn_backend, "cpu")) {
    return(TRUE)
  }
  if (identical(knn_backend, "auto")) {
    return(is.na(available_native_gpu_backend(need_knn = TRUE)))
  }
  FALSE
}

selective_landmark_refinement_enabled <- function() {
  opt <- getOption("fastEmbedR.selective_landmark_refinement", NULL)
  if (!is.null(opt)) {
    return(isTRUE(opt))
  }
  value <- tolower(Sys.getenv("FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT", ""))
  if (!nzchar(value)) {
    return(FALSE)
  }
  !value %in% c("0", "false", "no", "off", "all", "full")
}

selective_landmark_refinement_fraction <- function(n) {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    "FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT_FRACTION",
    ""
  )))
  if (length(value) == 1L && is.finite(value) && value > 0) {
    return(max(0.02, min(1, value)))
  }
  if (n < 20000L) {
    return(1)
  }
  0.35
}

projection_confidence_scores <- function(projection_nn) {
  d <- projection_nn$distances
  if (is.null(d) || !is.matrix(d) || nrow(d) < 1L || ncol(d) < 1L) {
    stop("projection_nn must contain a non-empty distance matrix.", call. = FALSE)
  }
  storage.mode(d) <- "double"
  if (any(!is.finite(d)) || any(d < 0)) {
    stop("projection distances must be finite and non-negative.", call. = FALSE)
  }

  n <- nrow(d)
  k <- ncol(d)
  eps <- sqrt(.Machine$double.eps)
  nearest <- d[, 1L]
  nearest_scale <- stats::median(nearest[is.finite(nearest) & nearest > eps])
  if (!is.finite(nearest_scale) || nearest_scale <= eps) {
    positive <- d[is.finite(d) & d > eps]
    nearest_scale <- if (length(positive)) stats::median(positive) else 1
  }
  if (!is.finite(nearest_scale) || nearest_scale <= eps) nearest_scale <- 1

  scores <- numeric(n)
  log_k <- if (k > 1L) log(k) else 1
  for (i in seq_len(n)) {
    row <- d[i, ]
    if (any(row <= eps)) {
      scores[i] <- 1
      next
    }
    rho <- min(row)
    adjusted <- pmax(0, row - rho)
    positive <- adjusted[adjusted > eps]
    sigma <- if (length(positive)) stats::median(positive) else stats::median(row)
    if (!is.finite(sigma) || sigma <= eps) sigma <- eps
    weights <- exp(-adjusted / sigma)
    weight_sum <- sum(weights)
    if (!is.finite(weight_sum) || weight_sum <= 0) {
      scores[i] <- 0
      next
    }
    probabilities <- weights / weight_sum
    entropy <- -sum(probabilities * log(pmax(probabilities, eps)))
    entropy_score <- if (k > 1L) 1 - min(1, entropy / log_k) else 1
    focus_score <- max(probabilities)
    distance_score <- nearest_scale / (nearest_scale + max(row[1L], 0))
    score <- 0.45 * distance_score + 0.35 * entropy_score + 0.20 * focus_score
    scores[i] <- max(0, min(1, score))
  }
  scores
}

select_landmark_refinement_rows <- function(projection_nn, landmark_indices) {
  n <- nrow(projection_nn$indices)
  if (!selective_landmark_refinement_enabled()) {
    return(list(
      rows = NULL,
      policy = "all",
      selected = n,
      selected_fraction = 1,
      confidence_threshold = NA_real_,
      selection_backend = NA_character_
    ))
  }

  fraction <- selective_landmark_refinement_fraction(n)
  if (!is.finite(fraction) || fraction >= 0.999) {
    return(list(
      rows = NULL,
      policy = "all",
      selected = n,
      selected_fraction = 1,
      confidence_threshold = NA_real_,
      selection_backend = NA_character_
    ))
  }

  scores <- attr(projection_nn, "confidence")
  has_native_confidence <- !is.null(scores) &&
    length(scores) == n &&
    all(is.finite(scores))
  if (isTRUE(has_native_confidence)) {
    return(select_low_confidence_rows_cpp(
      as.numeric(scores),
      as.integer(landmark_indices),
      fraction
    ))
  }
  if (is.null(scores) || length(scores) != n || any(!is.finite(scores))) {
    scores <- projection_confidence_scores(projection_nn)
  } else {
    scores <- as.numeric(scores)
  }
  landmark_mask <- rep(FALSE, n)
  landmark_indices <- as.integer(landmark_indices)
  landmark_indices <- landmark_indices[landmark_indices >= 1L & landmark_indices <= n]
  landmark_mask[landmark_indices] <- TRUE
  eligible <- which(!landmark_mask)
  if (!length(eligible)) {
    return(list(
      rows = integer(0),
      policy = "low_confidence",
      selected = 0L,
      selected_fraction = 0,
      confidence_threshold = NA_real_,
      selection_backend = "r_confidence_mask"
    ))
  }

  count <- as.integer(ceiling(length(eligible) * fraction))
  count <- max(1L, min(length(eligible), count))
  if (count >= length(eligible)) {
    return(list(
      rows = NULL,
      policy = "all",
      selected = length(eligible),
      selected_fraction = length(eligible) / n,
      confidence_threshold = NA_real_,
      selection_backend = "r_confidence_mask"
    ))
  }

  ordered <- eligible[order(scores[eligible], eligible)]
  rows <- sort(ordered[seq_len(count)])
  list(
    rows = rows,
    policy = "low_confidence",
    selected = length(rows),
    selected_fraction = length(rows) / n,
    confidence_threshold = max(scores[rows]),
    selection_backend = "r_confidence_mask"
  )
}

sampled_score_indices <- function(x,
                                  keep,
                                  preserve_k,
                                  backend) {
  n <- nrow(x)
  preserve_k <- as.integer(max(1L, min(preserve_k, n - 1L)))
  keep <- as.integer(keep)
  if (length(keep) == 0L) {
    return(matrix(integer(0L), nrow = 0L, ncol = preserve_k))
  }
  out <- matrix(1L, nrow = length(keep), ncol = preserve_k)
  query_k <- min(n, preserve_k + 1L)
  batch_size <- max(256L, min(length(keep), as.integer(floor(2e6 / max(1L, query_k)))))
  starts <- seq.int(1L, length(keep), by = batch_size)
  for (start in starts) {
    end <- min(length(keep), start + batch_size - 1L)
    rows <- start:end
    batch_keep <- keep[rows]
    raw <- fastEmbedR::nn(
      x,
      x[batch_keep, , drop = FALSE],
      k = query_k,
      backend = backend
    )
    for (local_i in seq_along(batch_keep)) {
      row <- unique(raw$indices[local_i, raw$indices[local_i, ] != batch_keep[local_i]])
      if (length(row) < preserve_k) {
        fill <- setdiff(seq_len(n), c(batch_keep[local_i], row))
        row <- c(row, fill)
      }
      out[rows[local_i], ] <- row[seq_len(preserve_k)]
    }
    rm(raw)
    if (length(starts) > 1L) gc(FALSE)
  }
  out
}

embed_from_knn <- function(method,
                           indices,
                           distances,
                           n_components,
                           seed,
                           backend,
                           verbose,
                           n_epochs = NULL,
                           umap_config_override = NULL) {
  if (!identical(method, "umap")) {
    stop("Only `method = \"umap\"` is supported.", call. = FALSE)
  }
  fast_knn_umap_core(
    indices,
    distances,
    n_components = n_components,
    seed = seed,
    backend = backend,
    verbose = verbose,
    n_epochs = n_epochs,
    config_override = umap_config_override
  )
}

#' Embed data with automatic KNN and layout settings
#'
#' @param data Numeric matrix or data frame with observations in rows.
#' @param labels Optional labels used only for quality scoring and plotting.
#' @param method Embedding objective. Only `"umap"` is supported.
#' @param mode Deprecated compatibility argument. Only `"auto"` is accepted.
#'   The package now uses the benchmark-selected default directly: full
#'   embeddings below the landmark threshold and landmark interpolation plus
#'   bucketed refinement for larger data.
#' @param n_neighbors Number of non-self neighbors. `NULL` chooses the
#'   package's automatic neighborhood size.
#' @param n_components Output dimensionality.
#' @param standardize If `TRUE`, center and scale columns before KNN.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param nn Optional precomputed KNN output with `indices` and `distances`.
#'   It may include a self-neighbor first column, as returned by `nn()`.
#' @param landmarks Optional landmark mode. Use `NULL` for a full embedding,
#'   `TRUE` for an automatic landmark count, a fraction such as `0.2`, a
#'   positive landmark count such as `1000`, or explicit row indices.
#' @param seed Random seed.
#' @param backend Execution backend for KNN and embedding. `"auto"` keeps the
#'   embedding on the accuracy-tested CPU path while allowing large exact KNN
#'   searches to use a native GPU when available. `"gpu"` explicitly requests a
#'   native GPU for both KNN and embedding, preferring CUDA and then Metal, and
#'   fails clearly if no usable native GPU backend is available.
#' @param silhouette_sample Optional sample size for silhouette scoring. Use
#'   `NULL` to skip silhouette scoring.
#' @param preserve_sample Optional sample size for neighborhood scoring. Use
#'   `NULL` to skip neighborhood scoring.
#' @param preserve_k Number of neighbors used for neighborhood scoring.
#' @param keep_knn If `TRUE`, retain the non-self KNN matrices in the returned
#'   object. The default `FALSE` keeps large embeddings compact.
#' @param verbose Print progress.
#' @return A `fastEmbedR_embedding` object containing the layout, metrics,
#'   timings, resolved parameters, and optional KNN matrices.
embed <- function(data,
                  labels = NULL,
                  method = "umap",
                  mode = "auto",
                  n_neighbors = NULL,
                  n_components = 2L,
                  standardize = TRUE,
                  pca_dims = NULL,
                  nn = NULL,
                  landmarks = NULL,
                  seed = 4L,
                  backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                  silhouette_sample = NULL,
                  preserve_sample = NULL,
                  preserve_k = NULL,
                  keep_knn = FALSE,
                  verbose = FALSE) {
  method <- match.arg(method, "umap")
  mode <- match.arg(mode)
  backend <- match.arg(backend)
  quality <- "auto"
  knn_backend <- resolve_backend_request(
    backend,
    need_knn = TRUE,
    need_embedding = identical(backend, "gpu")
  )
  embedding_backend <- resolve_backend_request(backend, need_embedding = TRUE)
  keep_knn <- isTRUE(keep_knn)

  prepared <- NULL
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = knn_backend
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  quality_requested <- quality
  auto_policy <- auto_tune_embedding_policy(
    n,
    method = method,
    mode = mode,
    n_neighbors = n_neighbors,
    landmarks = landmarks,
    quality = quality,
    embedding_backend = embedding_backend,
    nn_supplied = !is.null(nn),
    x = x,
    labels = labels,
    seed = seed,
    knn_backend = knn_backend
  )
  n_neighbors <- auto_policy$n_neighbors
  landmarks <- auto_policy[["landmarks"]]
  quality <- auto_policy$quality
  if (isTRUE(verbose) && isTRUE(auto_policy$auto_tuned)) {
    message(
      "Automatic defaults selected k=", if (is.null(n_neighbors)) "supplied" else n_neighbors,
      ", quality=", quality,
      ", landmarks=", if (is.null(landmarks)) "none" else "yes"
    )
  }
  landmark_indices <- resolve_landmarks(landmarks, x, seed)
  use_landmarks <- !is.null(landmark_indices) && length(landmark_indices) < n
  if (use_landmarks && !is.null(nn)) {
    stop("Landmark mode needs the data matrix to build landmark KNN; do not pass `nn` with `landmarks`.", call. = FALSE)
  }

  if (is.null(n_neighbors)) {
    n_neighbors <- auto_embedding_k(n, method = method, include_self = FALSE)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || !is.finite(n_neighbors) || n_neighbors < 1L) {
      stop("`n_neighbors` must be NULL or a positive integer.", call. = FALSE)
    }
    if (n_neighbors >= n) {
      stop("`n_neighbors` must be smaller than `nrow(data)`.", call. = FALSE)
    }
  }

  knn_result <- NULL
  landmark_info <- NULL
  projection_time <- structure(rep(0, 5), names = names(system.time({})))
  refinement_knn_time <- structure(rep(0, 5), names = names(system.time({})))
  refinement_time <- structure(rep(0, 5), names = names(system.time({})))
  scoring_knn_time <- structure(rep(0, 5), names = names(system.time({})))
  x_landmarks <- NULL
  landmark_refinement <- if (use_landmarks && !is.null(auto_policy$landmark_refinement)) {
    auto_policy$landmark_refinement
  } else {
    landmark_refinement_mode(mode, use_landmarks)
  }
  landmark_refinement_epochs <- auto_refinement_epoch_count(
    n,
    landmark_refinement,
    method
  )
  landmark_refinement_strength <- if (is.null(auto_policy$landmark_refinement_strength)) {
    1
  } else {
    as.numeric(auto_policy$landmark_refinement_strength)
  }
  if (!isTRUE(use_landmarks)) {
    landmark_refinement_strength <- 1
  }
  if (is.finite(landmark_refinement_strength) && landmark_refinement_strength > 0) {
    landmark_refinement_epochs <- as.integer(round(landmark_refinement_epochs * landmark_refinement_strength))
  }
  embedding_n <- if (use_landmarks) length(landmark_indices) else n
  embedding_k <- max(1L, min(n_neighbors, embedding_n - 1L))
  umap_config_override <- auto_policy$embedding_config_override
  embedding_n_epochs <- if (identical(method, "umap") &&
                            !is.null(umap_config_override) &&
                            !is.null(umap_config_override$n_epochs)) {
    validate_epoch_count(umap_config_override$n_epochs)
  } else {
    auto_embedding_epoch_count(
      method,
      embedding_n,
      embedding_k,
      embedding_backend,
      quality
    )
  }
  landmark_refinement_backend <- NA_character_
  landmark_refinement_knn_backend <- NA_character_
  landmark_refinement_knn_backend_reason <- NA_character_
  landmark_refinement_policy <- NA_character_
  landmark_refinement_selection_backend <- NA_character_
  landmark_refinement_selected <- NA_integer_
  landmark_refinement_selected_fraction <- NA_real_
  landmark_refinement_confidence_threshold <- NA_real_
  landmark_interpolation_backend <- NA_character_
  landmark_interpolation_backend_reason <- NA_character_
  landmark_projection_backend <- NA_character_
  knn_time <- system.time({
    if (use_landmarks) {
      n_landmarks <- length(landmark_indices)
      landmark_neighbors <- min(n_neighbors, n_landmarks - 1L)
      if (isTRUE(verbose)) {
        message("Computing landmark KNN with ", n_landmarks, " landmarks")
      }
      x_landmarks <- x[landmark_indices, , drop = FALSE]
      raw_knn <- nn_without_self(
        x_landmarks,
        k = landmark_neighbors,
        backend = knn_backend
      )
      knn_result <- normalize_supplied_knn(raw_knn, n_landmarks, landmark_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
    } else if (is.null(nn)) {
      if (isTRUE(verbose)) message("Computing KNN with ", n_neighbors, " neighbors")
      raw_knn <- nn_without_self(
        x,
        k = n_neighbors,
        backend = knn_backend
      )
      knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
      knn_result$nn_backend <- attr(raw_knn, "backend")
    } else {
      knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
      knn_result$nn_backend <- attr(nn, "backend")
      if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
    }
  })
  indices <- knn_result$indices
  distances <- knn_result$distances
  n_neighbors <- knn_result$n_neighbors

  layout <- NULL
  embedding_time <- system.time({
    if (isTRUE(verbose)) message("Running ", method)
    if (use_landmarks) {
      landmark_layout <- embed_from_knn(
        method,
        indices,
        distances,
        n_components,
        seed,
        embedding_backend,
        verbose,
        n_epochs = embedding_n_epochs,
        umap_config_override = umap_config_override
      )
    } else {
      layout <- embed_from_knn(
        method,
        indices,
        distances,
        n_components,
        seed,
        embedding_backend,
        verbose,
        n_epochs = embedding_n_epochs,
        umap_config_override = umap_config_override
      )
    }
  })

  if (use_landmarks) {
    landmark_selection_method <- attr(landmark_indices, "selection_method")
    if (is.null(landmark_selection_method)) {
      landmark_selection_method <- "indices"
    }
    projection_k <- landmark_projection_k(length(landmark_indices), n_neighbors)
    projection_nn <- NULL
    projection_time <- system.time({
      if (isTRUE(verbose)) message("Interpolating all points from nearest landmarks")
      fused_layout <- NULL
      projection_nn <- if (use_approx_landmark_projection(knn_backend)) {
        approx_landmark_projection_knn(
          x_landmarks,
          x,
          projection_k,
          seed
        )
      } else {
        NULL
      }
      if (!is.null(projection_nn)) {
        layout <- interpolate_landmark_layout(
          landmark_layout,
          landmark_indices,
          projection_nn,
          n,
          backend = embedding_backend
        )
        landmark_interpolation_backend <- attr(layout, "interpolation_backend")
        landmark_interpolation_backend_reason <- attr(layout, "interpolation_backend_reason")
        landmark_projection_backend <- attr(projection_nn, "backend")
      } else {
        fused_projection <- fused_landmark_project_knn_confidence(
          x_landmarks,
          x,
          landmark_layout,
          landmark_indices,
          projection_k,
          backend = embedding_backend
        )
        if (!is.null(fused_projection)) {
          layout <- fused_projection$layout
          projection_nn <- fused_projection$projection_nn
          landmark_interpolation_backend <- attr(layout, "interpolation_backend")
          landmark_interpolation_backend_reason <- attr(layout, "interpolation_backend_reason")
          landmark_projection_backend <- attr(layout, "projection_backend")
        } else if (identical(landmark_refinement, "none")) {
          fused_layout <- fused_landmark_project_layout(
            x_landmarks,
            x,
            landmark_layout,
            landmark_indices,
            projection_k,
            backend = embedding_backend
          )
          if (!is.null(fused_layout)) {
            layout <- fused_layout
            landmark_interpolation_backend <- attr(layout, "interpolation_backend")
            landmark_interpolation_backend_reason <- attr(layout, "interpolation_backend_reason")
            landmark_projection_backend <- attr(layout, "projection_backend")
          }
        }
      }
      if (is.null(layout)) {
        projection_nn <- fastEmbedR::nn(
          x_landmarks,
          x,
          k = projection_k,
          backend = knn_backend
        )
        layout <- interpolate_landmark_layout(
          landmark_layout,
          landmark_indices,
          projection_nn,
          n,
          backend = embedding_backend
        )
        landmark_interpolation_backend <- attr(layout, "interpolation_backend")
        landmark_interpolation_backend_reason <- attr(layout, "interpolation_backend_reason")
        landmark_projection_backend <- attr(projection_nn, "backend")
      }
    })
    refinement_graph <- NULL
    landmark_refinement_rows <- NULL
    if (identical(landmark_refinement, "bucketed")) {
      selection <- select_landmark_refinement_rows(projection_nn, landmark_indices)
      landmark_refinement_rows <- selection$rows
      landmark_refinement_policy <- selection$policy
      landmark_refinement_selection_backend <- selection$selection_backend
      landmark_refinement_selected <- as.integer(selection$selected)
      landmark_refinement_selected_fraction <- as.numeric(selection$selected_fraction)
      landmark_refinement_confidence_threshold <- as.numeric(selection$confidence_threshold)
      refinement_knn_time <- system.time({
        if (isTRUE(verbose)) {
          message(
            "Computing bucketed local KNN for landmark refinement",
            if (identical(landmark_refinement_policy, "low_confidence")) {
              paste0(" (", landmark_refinement_selected, " low-confidence rows)")
            } else {
              ""
            }
          )
        }
        refinement_graph <- landmark_bucketed_knn_graph(
          x,
          projection_nn,
          n_neighbors,
          backend = knn_backend,
          update_rows = landmark_refinement_rows
        )
        landmark_refinement_knn_backend <- refinement_graph$backend
        landmark_refinement_knn_backend_reason <- refinement_graph$backend_reason
      })
    }
    projection_nn <- NULL
    if (n > 5000L) gc(FALSE)
    if (!identical(landmark_refinement, "none") && landmark_refinement_epochs > 0L) {
      refinement_time <- system.time({
        if (isTRUE(verbose)) {
          message(
            "Refining landmark embedding with ",
            landmark_refinement,
            " graph for ",
            landmark_refinement_epochs,
            " epochs"
          )
        }
        layout <- refine_embedding_from_knn(
          method,
          refinement_graph$indices,
          refinement_graph$distances,
          layout,
          landmark_refinement_epochs,
          landmark_refinement,
          seed,
          embedding_backend,
          verbose,
          update_rows = landmark_refinement_rows,
          row_ids = refinement_graph$row_ids
        )
      })
      refinement_config <- attr(layout, "fastEmbedR_config")
      landmark_refinement_backend <- refinement_config$refinement_backend
    }
    refinement_graph <- NULL
    if (n > 5000L) gc(FALSE)
    attr(layout, "fastEmbedR_config") <- c(
      attr(landmark_layout, "fastEmbedR_config"),
      list(
        landmark = TRUE,
        n_landmarks = length(landmark_indices),
        landmark_fraction = length(landmark_indices) / n,
        landmark_projection_k = projection_k,
        landmark_selection = landmark_selection_method,
        landmark_interpolation = "local_membership",
        landmark_refinement = landmark_refinement,
        landmark_refinement_epochs = landmark_refinement_epochs,
        landmark_refinement_strength = landmark_refinement_strength,
        landmark_refinement_backend = landmark_refinement_backend,
        landmark_refinement_knn_backend = landmark_refinement_knn_backend,
        landmark_refinement_knn_backend_reason = landmark_refinement_knn_backend_reason,
        landmark_refinement_policy = landmark_refinement_policy,
        landmark_refinement_selection_backend = landmark_refinement_selection_backend,
        landmark_refinement_selected = landmark_refinement_selected,
        landmark_refinement_selected_fraction = landmark_refinement_selected_fraction,
        landmark_refinement_confidence_threshold = landmark_refinement_confidence_threshold,
        landmark_projection_backend = landmark_projection_backend,
        landmark_interpolation_backend = landmark_interpolation_backend,
        landmark_interpolation_backend_reason = landmark_interpolation_backend_reason
      )
    )
    landmark_info <- list(
      indices = landmark_indices,
      layout = landmark_layout,
      selection = landmark_selection_method,
      interpolation = "local_membership",
      refinement = landmark_refinement,
      refinement_epochs = landmark_refinement_epochs,
      refinement_strength = landmark_refinement_strength,
      refinement_backend = landmark_refinement_backend,
      refinement_knn_backend = landmark_refinement_knn_backend,
      refinement_knn_backend_reason = landmark_refinement_knn_backend_reason,
      refinement_policy = landmark_refinement_policy,
      refinement_selection_backend = landmark_refinement_selection_backend,
      refinement_selected = landmark_refinement_selected,
      refinement_selected_fraction = landmark_refinement_selected_fraction,
      refinement_confidence_threshold = landmark_refinement_confidence_threshold,
	      interpolation_backend = landmark_interpolation_backend,
	      interpolation_backend_reason = landmark_interpolation_backend_reason,
	      projection_k = projection_k,
	      projection_backend = landmark_projection_backend
	    )
  }

  score_indices <- indices
  score_preserve_k <- if (is.null(preserve_k)) ncol(score_indices) else min(as.integer(preserve_k), ncol(score_indices))
  preserve_keep <- NULL
  if (use_landmarks) {
    score_preserve_k <- if (is.null(preserve_k)) min(auto_k(n), n - 1L) else min(as.integer(preserve_k), n - 1L)
    preserve_keep <- sample_indices(n, preserve_sample, seed)
    if (length(preserve_keep) == 0L) {
      score_indices <- matrix(integer(0L), nrow = 0L, ncol = score_preserve_k)
    } else {
      scoring_knn_time <- system.time({
        score_indices <- sampled_score_indices(
          x,
          preserve_keep,
          score_preserve_k,
          knn_backend
        )
      })
    }
  }

  scores <- embedding_scores(
    layout,
    labels,
    score_indices,
    silhouette_sample,
    preserve_sample,
    score_preserve_k,
    seed,
    preserve_keep = preserve_keep,
    backend = embedding_backend
  )
  embedding_config <- attr(layout, "fastEmbedR_config")
  scoring_structure_backend <- attr(scores, "structure_backend")
  scoring_silhouette_backend <- attr(scores, "silhouette_backend")
  scoring_structure_backend_reason <- attr(scores, "structure_backend_reason")
  scoring_silhouette_backend_reason <- attr(scores, "silhouette_backend_reason")
  parameters <- c(
    list(
      method = method,
      mode = mode,
      n = n,
      p = ncol(x),
      n_neighbors = n_neighbors,
      k = n_neighbors + 1L,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      quality_requested = quality_requested,
      quality = quality,
      auto_tuned = isTRUE(auto_policy$auto_tuned),
      auto_k_selected_by = auto_policy$k_selected_by,
      auto_quality_selected_by = auto_policy$quality_selected_by,
      auto_landmarks_selected_by = auto_policy$landmarks_selected_by,
      auto_refinement_selected_by = auto_policy$refinement_selected_by,
      auto_epoch_selected_by = auto_policy$epoch_selected_by,
      auto_pilot_score = auto_policy$selected_pilot_score,
      auto_refinement_pilot_score = auto_policy$selected_refinement_pilot_score,
      auto_pilot_sample_n = auto_policy$pilot_sample_n,
      auto_pilot_backend = auto_policy$pilot_backend,
      auto_pilot_spectral_n_iter = auto_policy$pilot_spectral_n_iter,
      auto_pilot_n_epochs = auto_policy$pilot_n_epochs,
      auto_pilot_init_scale = auto_policy$pilot_init_scale,
      auto_pilot_refinement_strength = auto_policy$pilot_refinement_strength,
      auto_tune_reason = auto_policy$reason,
      embedding_n_epochs = embedding_n_epochs,
      nn_backend = knn_result$nn_backend,
      landmark = isTRUE(use_landmarks),
      n_landmarks = if (use_landmarks) length(landmark_indices) else NA_integer_,
      landmark_fraction = if (use_landmarks) length(landmark_indices) / n else NA_real_,
      landmark_refinement = if (use_landmarks) landmark_refinement else NA_character_,
      landmark_refinement_epochs = if (use_landmarks) landmark_refinement_epochs else NA_integer_,
      landmark_refinement_strength = if (use_landmarks) landmark_refinement_strength else NA_real_,
      landmark_refinement_backend = if (use_landmarks) landmark_refinement_backend else NA_character_,
      landmark_refinement_knn_backend = if (use_landmarks) landmark_refinement_knn_backend else NA_character_,
      landmark_refinement_knn_backend_reason = if (use_landmarks) landmark_refinement_knn_backend_reason else NA_character_,
      landmark_refinement_policy = if (use_landmarks) landmark_refinement_policy else NA_character_,
      landmark_refinement_selected = if (use_landmarks) landmark_refinement_selected else NA_integer_,
      landmark_refinement_selected_fraction = if (use_landmarks) landmark_refinement_selected_fraction else NA_real_,
      landmark_refinement_confidence_threshold = if (use_landmarks) landmark_refinement_confidence_threshold else NA_real_,
      landmark_projection_backend = if (use_landmarks) landmark_projection_backend else NA_character_,
      landmark_interpolation_backend = if (use_landmarks) landmark_interpolation_backend else NA_character_,
      scoring_structure_backend = scoring_structure_backend,
      scoring_silhouette_backend = scoring_silhouette_backend,
      scoring_structure_backend_reason = scoring_structure_backend_reason,
      scoring_silhouette_backend_reason = scoring_silhouette_backend_reason,
      keep_knn = keep_knn
    ),
    embedding_config,
    prepared$preprocess
  )
  timings <- rbind(
    preprocess = preprocess_time,
    knn = knn_time,
    embedding = embedding_time,
    landmark_projection = projection_time,
    landmark_refinement_knn = refinement_knn_time,
    landmark_refinement = refinement_time,
    scoring_knn = scoring_knn_time
  )
  metrics <- data.frame(
    method = method,
    n = n,
    p = ncol(x),
    n_neighbors = n_neighbors,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    knn_elapsed = knn_time["elapsed"],
    embedding_elapsed = embedding_time["elapsed"],
    landmark_projection_elapsed = projection_time["elapsed"],
    landmark_refinement_knn_elapsed = refinement_knn_time["elapsed"],
    landmark_refinement_elapsed = refinement_time["elapsed"],
    scoring_knn_elapsed = scoring_knn_time["elapsed"],
    landmark = isTRUE(use_landmarks),
    n_landmarks = if (use_landmarks) length(landmark_indices) else NA_integer_,
    landmark_refinement = if (use_landmarks) landmark_refinement else NA_character_,
    landmark_refinement_epochs = if (use_landmarks) landmark_refinement_epochs else NA_integer_,
    landmark_refinement_strength = if (use_landmarks) landmark_refinement_strength else NA_real_,
    landmark_refinement_policy = if (use_landmarks) landmark_refinement_policy else NA_character_,
    landmark_refinement_selected = if (use_landmarks) landmark_refinement_selected else NA_integer_,
    landmark_refinement_selected_fraction = if (use_landmarks) landmark_refinement_selected_fraction else NA_real_,
    auto_tuned = isTRUE(auto_policy$auto_tuned),
    auto_pilot_score = auto_policy$selected_pilot_score,
    auto_pilot_sample_n = auto_policy$pilot_sample_n,
    scoring_structure_backend = scoring_structure_backend,
    scoring_silhouette_backend = scoring_silhouette_backend,
    scores,
    stringsAsFactors = FALSE
  )

  out <- list(
    layout = layout,
    labels = labels,
    method = method,
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    auto_tuning = auto_policy,
    knn = if (keep_knn) list(indices = indices, distances = distances) else NULL,
    knn_with_self = if (keep_knn) knn_result$knn_with_self else NULL,
    landmarks = landmark_info,
    preprocess = prepared$preprocess
  )
  class(out) <- "fastEmbedR_embedding"
  out
}

#' @rdname embed
#' @export
umap <- function(data, labels = NULL, ...) {
  embed(data = data, labels = labels, method = "umap", ...)
}

supervised_knn_distances <- function(indices,
                                     distances,
                                     labels,
                                     target_weight = 0.5,
                                     target_metric = "categorical") {
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  if (!identical(dim(indices), dim(distances))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  n <- nrow(indices)
  if (length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  target_weight <- as.numeric(target_weight)
  if (length(target_weight) != 1L || is.na(target_weight) || !is.finite(target_weight) ||
      target_weight < 0 || target_weight > 1) {
    stop("`target_weight` must be a number in [0, 1].", call. = FALSE)
  }
  target_metric <- match.arg(target_metric, c("categorical"))
  if (target_weight == 0) {
    return(distances)
  }

  labs <- as.factor(labels)
  lab_int <- as.integer(labs)
  same_scale <- max(0.25, 1 - 0.5 * target_weight)
  different_scale <- 1 / max(1e-6, 1 - target_weight)

  for (i in seq_len(n)) {
    row_idx <- indices[i, ]
    row_idx <- pmin(pmax(row_idx, 1L), n)
    same <- !is.na(lab_int[i]) & !is.na(lab_int[row_idx]) & lab_int[row_idx] == lab_int[i]
    known_different <- !is.na(lab_int[i]) & !is.na(lab_int[row_idx]) & lab_int[row_idx] != lab_int[i]
    distances[i, same] <- distances[i, same] * same_scale
    distances[i, known_different] <- distances[i, known_different] * different_scale
  }
  distances
}

#' Supervised UMAP from labels
#'
#' Runs UMAP with an explicit supervised graph adjustment inspired by cuML's
#' `y`/`target_weight` supervised UMAP interface. The regular `umap()` function
#' still treats labels only as scoring/plotting metadata; use
#' `supervised_umap()` when labels should guide the embedding itself.
#'
#' @param data Numeric matrix or data frame with observations in rows.
#' @param labels Labels used to adjust the UMAP KNN graph and to score/plot.
#' @param target_weight Strength of label supervision in `[0, 1]`.
#' @param target_metric Currently only `"categorical"` is implemented.
#' @param n_neighbors Number of non-self neighbors. `NULL` chooses the
#'   package's automatic neighborhood size.
#' @param n_components Output dimensionality.
#' @param standardize If `TRUE`, center and scale columns before KNN.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param seed Random seed.
#' @param backend KNN and embedding backend. Use `"gpu"` to explicitly request
#'   a native GPU for both stages. `"auto"` may select the internal clustered
#'   CPU KNN path for very large self-KNN searches and records that backend in
#'   the result metadata.
#' @param silhouette_sample Optional sample size for silhouette scoring. Use
#'   `NULL` to skip silhouette scoring.
#' @param preserve_sample Optional sample size for neighborhood scoring. Use
#'   `NULL` to skip neighborhood scoring.
#' @param preserve_k Number of neighbors used for neighborhood scoring.
#' @param keep_knn If `TRUE`, retain the supervised non-self KNN matrices.
#' @param verbose Print progress.
#' @return A `fastEmbedR_embedding` object.
#' @export
supervised_umap <- function(data,
                            labels,
                            target_weight = 0.5,
                            target_metric = c("categorical"),
                            n_neighbors = NULL,
                            n_components = 2L,
                            standardize = TRUE,
                            pca_dims = NULL,
                            seed = 4L,
                            backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                            silhouette_sample = NULL,
                            preserve_sample = NULL,
                            preserve_k = NULL,
                            keep_knn = FALSE,
                            verbose = FALSE) {
  if (missing(labels) || is.null(labels)) {
    stop("`labels` are required for supervised UMAP.", call. = FALSE)
  }
  target_metric <- match.arg(target_metric)
  backend <- match.arg(backend)
  knn_backend <- if (identical(backend, "gpu")) {
    resolve_backend_request(backend, need_knn = TRUE, need_embedding = TRUE)
  } else {
    backend
  }

  prepared <- NULL
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = knn_backend
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }

  if (is.null(n_neighbors)) {
    n_neighbors <- auto_k(n, include_self = FALSE)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || !is.finite(n_neighbors) || n_neighbors < 1L) {
      stop("`n_neighbors` must be NULL or a positive integer.", call. = FALSE)
    }
    if (n_neighbors >= n) {
      stop("`n_neighbors` must be smaller than `nrow(data)`.", call. = FALSE)
    }
  }

  raw_knn <- NULL
  supervised_knn <- NULL
  knn_time <- system.time({
    raw_knn <- nn_without_self(x, k = n_neighbors, backend = knn_backend)
    knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
    supervised_distances <- supervised_knn_distances(
      knn_result$indices,
      knn_result$distances,
      labels,
      target_weight = target_weight,
      target_metric = target_metric
    )
    supervised_knn <- list(
      indices = knn_result$indices,
      distances = supervised_distances
    )
    attr(supervised_knn, "backend") <- attr(raw_knn, "backend")
  })

  embedding_backend <- if (identical(backend, "gpu")) {
    knn_backend
  } else {
    backend
  }
  fit <- embed(
    x,
    labels = labels,
    method = "umap",
    mode = "auto",
    n_neighbors = n_neighbors,
    n_components = n_components,
    standardize = FALSE,
    pca_dims = NULL,
    nn = supervised_knn,
    landmarks = NULL,
    seed = seed,
    backend = embedding_backend,
    silhouette_sample = silhouette_sample,
    preserve_sample = preserve_sample,
    preserve_k = preserve_k,
    keep_knn = keep_knn,
    verbose = verbose
  )

  fit$timings["preprocess", ] <- fit$timings["preprocess", ] + preprocess_time
  fit$timings["knn", ] <- fit$timings["knn", ] + knn_time
  fit$metrics$preprocess_elapsed <- fit$timings["preprocess", "elapsed"]
  fit$metrics$knn_elapsed <- fit$timings["knn", "elapsed"]
  fit$metrics$elapsed <- sum(fit$timings[, "elapsed"])
  fit$parameters$supervised <- TRUE
  fit$parameters$target_weight <- as.numeric(target_weight)
  fit$parameters$target_metric <- target_metric
  fit$parameters$nn_backend <- attr(raw_knn, "backend")
  for (field in names(prepared$preprocess)) {
    fit$parameters[[field]] <- prepared$preprocess[[field]]
  }
  fit$preprocess <- prepared$preprocess
  fit$metrics$supervised <- TRUE
  fit$metrics$target_weight <- as.numeric(target_weight)
  fit$supervision <- list(
    labels = labels,
    target_weight = as.numeric(target_weight),
    target_metric = target_metric,
    knn_backend = attr(raw_knn, "backend"),
    exact_knn = attr(raw_knn, "exact")
  )
  fit
}

#' @export
print.fastEmbedR_embedding <- function(x, ...) {
  cat("fastEmbedR embedding\n")
  cat("  method: ", x$parameters$method, "\n", sep = "")
  cat("  observations: ", x$parameters$n, "\n", sep = "")
  cat("  dimensions: ", ncol(x$layout), "\n", sep = "")
  cat("  neighbors: ", x$parameters$n_neighbors, "\n", sep = "")
  if (isTRUE(x$parameters$landmark)) {
    cat("  landmarks: ", x$parameters$n_landmarks, "\n", sep = "")
    cat("  landmark refinement: ", x$parameters$landmark_refinement, "\n", sep = "")
  }
  cat("  embedding backend: ", x$parameters$backend, "\n", sep = "")
  cat("  KNN backend: ", x$parameters$nn_backend, "\n", sep = "")
  cat("  elapsed: ", format(round(x$metrics$elapsed, 3L), nsmall = 3L), " sec\n", sep = "")
  if (is.finite(x$metrics$silhouette)) {
    cat("  silhouette: ", format(round(x$metrics$silhouette, 4L), nsmall = 4L), "\n", sep = "")
  }
  if (is.finite(x$metrics$knn_preservation)) {
    cat("  KNN preservation: ", format(round(x$metrics$knn_preservation, 4L), nsmall = 4L), "\n", sep = "")
  }
  invisible(x)
}

#' @export
plot.fastEmbedR_embedding <- function(x,
                                       labels = x$labels,
                                       pch = 21,
                                       bg = NULL,
                                       col = "black",
                                       xlab = "Component 1",
                                       ylab = "Component 2",
                                       main = NULL,
                                       ...) {
  layout <- as.matrix(x$layout)
  if (ncol(layout) < 2L) {
    stop("Plotting requires at least two embedding dimensions.", call. = FALSE)
  }
  if (is.null(main)) {
    main <- paste0("fastEmbedR ", x$parameters$method)
  }
  if (is.null(bg)) {
    bg <- if (is.null(labels)) {
      "white"
    } else {
      as.integer(as.factor(labels))
    }
  }
  graphics::plot(
    layout[, 1L],
    layout[, 2L],
    pch = pch,
    bg = bg,
    col = col,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )
  invisible(x)
}
