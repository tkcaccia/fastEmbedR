#' Run UMAP from a data matrix or precomputed KNN
#'
#' `umap()` is a small convenience wrapper. If `data` is already a KNN object
#' returned by [nn()], it calls [umap_knn()] directly. Otherwise it preprocesses
#' the data, computes KNN once, and embeds from that KNN graph.
#'
#' @param data Numeric matrix/data frame, or a KNN object returned by [nn()].
#' @param labels Optional labels stored in the returned object and used for
#'   optional scoring.
#' @param n_neighbors Number of non-self neighbours. `NULL` chooses the package
#'   default for the data size.
#' @param n_components Output dimensionality.
#' @param standardize Center and scale columns before KNN when `data` is a
#'   matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param nn Optional precomputed KNN result when `data` is a matrix.
#' @param seed Random seed.
#' @param backend Execution backend. GPU requests must resolve to a real native
#'   backend; the package does not relabel CPU work as GPU.
#' @param n_threads Number of CPU worker threads for KNN and CPU UMAP.
#' @param silhouette_sample Optional sample size for silhouette scoring.
#' @param preserve_sample Optional sample size for local structure scoring.
#' @param preserve_k Number of neighbours used for local structure scoring.
#' @param keep_knn Keep KNN matrices in the returned object.
#' @param verbose Print progress.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' fit <- umap(x, labels = iris$Species, n_neighbors = 15, seed = 1)
#' plot(fit)
#' @export
umap <- function(data,
                 labels = NULL,
                 n_neighbors = NULL,
                 n_components = 2L,
                 standardize = TRUE,
                 pca_dims = NULL,
                 nn = NULL,
                 seed = 4L,
                 backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                 n_threads = NULL,
                 silhouette_sample = NULL,
                 preserve_sample = NULL,
                 preserve_k = NULL,
                 keep_knn = FALSE,
                 verbose = FALSE) {
  backend <- match.arg(backend)
  n_components <- validate_n_components(n_components)
  keep_knn <- isTRUE(keep_knn)

  if (is_knn_input(data)) {
    if (!is.null(nn)) {
      stop("When `data` is a KNN object, do not also pass `nn`.", call. = FALSE)
    }
    layout_time <- system.time({
      layout <- fast_knn_umap(
        data,
        n_components = n_components,
        seed = seed,
        verbose = verbose,
        backend = backend,
        n_threads = n_threads
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
    scores <- embedding_scores(
      layout,
      labels,
      coerce_knn_input(data)$indices,
      silhouette_sample,
      preserve_sample,
      preserve_k,
      seed,
      backend = cfg$backend
    )
    resolved_k <- cfg$knn_n_neighbors
    if (is.null(resolved_k) || length(resolved_k) == 0L || is.na(resolved_k)) {
      resolved_k <- cfg$n_neighbors
    }
    if (is.null(resolved_k) || length(resolved_k) == 0L || is.na(resolved_k)) {
      resolved_k <- ncol(coerce_knn_input(data)$indices)
    }
    metrics <- data.frame(
      method = "umap",
      n = nrow(layout),
      p = NA_integer_,
      n_neighbors = as.integer(resolved_k),
      elapsed = layout_time[["elapsed"]],
      preprocess_elapsed = 0,
      knn_elapsed = 0,
      embedding_elapsed = layout_time[["elapsed"]],
      scores,
      stringsAsFactors = FALSE
    )
    out <- list(
      layout = layout,
      labels = labels,
      method = "umap",
      metrics = metrics,
      parameters = cfg,
      timings = rbind(
        preprocess = layout_time * 0,
        knn = layout_time * 0,
        embedding = layout_time
      ),
      knn = if (keep_knn) data else NULL
    )
    class(out) <- "fastEmbedR_embedding"
    return(out)
  }

  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = resolve_backend_request(backend, need_knn = TRUE)
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  if (is.null(n_neighbors)) {
    n_neighbors <- auto_embedding_k(n, method = "umap", include_self = FALSE)
  }

  knn_time <- system.time({
    knn_result <- if (is.null(nn)) {
      nn_without_self(
        x,
        k = as.integer(n_neighbors),
        backend = resolve_backend_request(backend, need_knn = TRUE),
        n_threads = n_threads
      )
    } else {
      coerce_knn_input(nn)
    }
  })

  embedding_backend <- resolve_backend_request(
    backend,
    need_embedding = identical(backend, "gpu")
  )
  embedding_time <- system.time({
    layout <- fast_knn_umap(
      knn_result,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = embedding_backend,
      n_threads = n_threads
    )
  })
  cfg <- attr(layout, "fastEmbedR_config")
  scores <- embedding_scores(
    layout,
    labels,
    coerce_knn_input(knn_result)$indices,
    silhouette_sample,
    preserve_sample,
    preserve_k,
    seed,
    backend = cfg$backend
  )

  elapsed <- preprocess_time[["elapsed"]] + knn_time[["elapsed"]] + embedding_time[["elapsed"]]
  metrics <- data.frame(
    method = "umap",
    n = nrow(layout),
    p = ncol(x),
    n_neighbors = as.integer(n_neighbors),
    elapsed = elapsed,
    preprocess_elapsed = preprocess_time[["elapsed"]],
    knn_elapsed = knn_time[["elapsed"]],
    embedding_elapsed = embedding_time[["elapsed"]],
    scores,
    stringsAsFactors = FALSE
  )
  parameters <- c(
    cfg,
    list(
      standardize = isTRUE(standardize),
      pca_dims = prepared$preprocess$pca_dims,
      preprocess = prepared$preprocess,
      nn_backend = if (is.null(attr(knn_result, "backend"))) "supplied" else attr(knn_result, "backend")
    )
  )
  out <- list(
    layout = layout,
    labels = labels,
    method = "umap",
    metrics = metrics,
    parameters = parameters,
    timings = rbind(
      preprocess = preprocess_time,
      knn = knn_time,
      embedding = embedding_time
    ),
    knn = if (keep_knn) knn_result else NULL
  )
  class(out) <- "fastEmbedR_embedding"
  out
}

#' Run landmark UMAP from a data matrix
#'
#' `landmark_umap()` embeds a landmark subset with [umap()] and projects the
#' remaining observations by KNN interpolation against the fixed landmark
#' embedding. It is an explicit landmark approximation: the UMAP objective and
#' parameters for the landmark subset are unchanged.
#'
#' @inheritParams umap
#' @param landmarks `TRUE` for an automatic subset, a fraction such as `0.5`, a
#'   landmark count, or explicit row indices.
#' @param transform_k Number of landmark neighbours used to project
#'   non-landmark observations. Defaults to `n_neighbors`.
#' @export
landmark_umap <- function(data,
                          labels = NULL,
                          landmarks = 0.5,
                          n_neighbors = NULL,
                          n_components = 2L,
                          standardize = TRUE,
                          pca_dims = NULL,
                          seed = 4L,
                          backend = c("auto", "cpu", "gpu", "metal", "cuda"),
                          transform_k = NULL,
                          n_threads = NULL,
                          silhouette_sample = NULL,
                          preserve_sample = NULL,
                          preserve_k = NULL,
                          keep_knn = FALSE,
                          verbose = FALSE) {
  backend <- match.arg(backend)
  n_components <- validate_n_components(n_components)
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = resolve_backend_request(backend, need_knn = TRUE)
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  if (is.null(n_neighbors)) {
    n_neighbors <- auto_embedding_k(n, method = "umap", include_self = FALSE)
  }
  n_neighbors <- as.integer(n_neighbors)
  if (length(n_neighbors) != 1L || is.na(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
    stop("`n_neighbors` must be a positive integer smaller than `nrow(data)`.", call. = FALSE)
  }

  landmark_indices <- resolve_landmarks(landmarks, x, seed)
  if (is.null(landmark_indices)) {
    return(umap(
      x,
      labels = labels,
      n_neighbors = n_neighbors,
      n_components = n_components,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      n_threads = n_threads,
      silhouette_sample = silhouette_sample,
      preserve_sample = preserve_sample,
      preserve_k = preserve_k,
      keep_knn = keep_knn,
      verbose = verbose
    ))
  }

  x_landmarks <- x[landmark_indices, , drop = FALSE]
  n_landmarks <- nrow(x_landmarks)
  landmark_neighbors <- min(n_neighbors, n_landmarks - 1L)
  landmark_labels <- if (is.null(labels)) NULL else labels[landmark_indices]
  reference_time <- system.time({
    reference_fit <- umap(
      x_landmarks,
      labels = landmark_labels,
      n_neighbors = landmark_neighbors,
      n_components = n_components,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      n_threads = n_threads,
      silhouette_sample = NULL,
      preserve_sample = NULL,
      keep_knn = keep_knn,
      verbose = verbose
    )
  })

  non_landmarks <- setdiff(seq_len(n), landmark_indices)
  if (is.null(transform_k)) transform_k <- min(n_landmarks, n_neighbors)
  transform_k <- transform_embedding_k(transform_k, n_landmarks)
  projection_time <- system.time({
    projection_knn <- landmark_projection_knn(
      x_landmarks,
      x[non_landmarks, , drop = FALSE],
      k = transform_k,
      backend = backend,
      seed = seed + 503L,
      n_threads = n_threads,
      landmark_layout = reference_fit$layout,
      all_data = x,
      landmark_indices = landmark_indices,
      query_rows = non_landmarks
    )
    projected <- attr(projection_knn, "projected_layout", exact = TRUE)
    if (is.null(projected)) {
      project_backend <- if (backend %in% c("metal", "gpu") && isTRUE(embedding_metal_available_cpp())) {
        "metal"
      } else if (identical(backend, "cuda") && isTRUE(embedding_cuda_available_cpp())) {
        "cuda"
      } else {
        "cpu"
      }
      projected <- switch(
        project_backend,
        metal = project_embedding_knn_metal_cpp(
          reference_fit$layout,
          projection_knn$indices,
          projection_knn$distances
        ),
        cuda = project_embedding_knn_cuda_cpp(
          reference_fit$layout,
          projection_knn$indices,
          projection_knn$distances
        ),
        project_embedding_knn_cpp(
          reference_fit$layout,
          projection_knn$indices,
          projection_knn$distances
        )
      )
    }
  })

  layout <- matrix(NA_real_, nrow = n, ncol = n_components)
  layout[landmark_indices, ] <- reference_fit$layout
  layout[non_landmarks, ] <- projected
  colnames(layout) <- colnames(reference_fit$layout)

  zero <- zero_proc_time()
  timings <- rbind(
    preprocess = preprocess_time,
    reference_embedding = reference_time,
    landmark_projection_knn = projection_time,
    transform = zero
  )
  reference_metrics <- reference_fit$metrics
  reference_knn_elapsed <- if ("knn_elapsed" %in% names(reference_metrics)) {
    reference_metrics$knn_elapsed[[1L]]
  } else {
    NA_real_
  }
  reference_optimizer_elapsed <- if ("embedding_elapsed" %in% names(reference_metrics)) {
    reference_metrics$embedding_elapsed[[1L]]
  } else {
    NA_real_
  }
  metrics <- data.frame(
    method = "landmark_umap",
    n = n,
    p = ncol(x),
    n_neighbors = n_neighbors,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time[["elapsed"]],
    reference_embedding_elapsed = reference_time[["elapsed"]],
    reference_knn_elapsed = reference_knn_elapsed,
    reference_optimizer_elapsed = reference_optimizer_elapsed,
    landmark_projection_knn_elapsed = projection_time[["elapsed"]],
    transform_elapsed = 0,
    landmark = TRUE,
    n_landmarks = n_landmarks,
    landmark_fraction = n_landmarks / n,
    transform_k = transform_k,
    silhouette = NA_real_,
    knn_preservation = NA_real_,
    local_trustworthiness = NA_real_,
    local_continuity = NA_real_,
    structure_score = NA_real_,
    embedding_knn_accuracy = NA_real_,
    stringsAsFactors = FALSE
  )
  selection_method <- attr(landmark_indices, "selection_method")
  if (is.null(selection_method)) selection_method <- "indices"
  projection_approximation <- attr(projection_knn, "approximation", exact = TRUE)
  projection_strategy <- if (is.null(projection_approximation$strategy)) {
    if (isTRUE(attr(projection_knn, "exact"))) "exact" else NA_character_
  } else {
    as.character(projection_approximation$strategy)
  }
  parameters <- c(
    list(
      method = "landmark_umap",
      n = n,
      p = ncol(x),
      n_neighbors = n_neighbors,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      nn_backend = reference_fit$parameters$nn_backend,
      projection_nn_backend = attr(projection_knn, "backend"),
      projection_strategy = projection_strategy,
      backend = reference_fit$parameters$backend %||% backend,
      n_threads = normalize_nn_threads(n_threads),
      landmark = TRUE,
      n_landmarks = n_landmarks,
      landmark_fraction = n_landmarks / n,
      landmark_selection = selection_method,
      transform_k = transform_k,
      keep_knn = keep_knn,
      provenance = "UMAP_landmark_projection_native_cpp"
    ),
    prepared$preprocess
  )
  out <- list(
    layout = layout,
    labels = labels,
    method = "landmark_umap",
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = NULL,
    landmarks = list(
      indices = landmark_indices,
      layout = reference_fit$layout,
      reference_fit = reference_fit,
      projection_knn = if (isTRUE(keep_knn)) projection_knn else NULL
    ),
    preprocess = prepared$preprocess
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
