#' Transform query points into an existing t-SNE embedding
#'
#' `transform_tsne()` places query observations into a fixed reference t-SNE
#' embedding. It follows the openTSNE transform design at the algorithm level:
#' query-to-reference affinities are computed from KNN distances, query points
#' are initialized from nearby reference points, and optimization moves only the
#' query points.
#'
#' @param reference_layout Numeric reference embedding matrix, or a
#'   `fastEmbedR_embedding` object.
#' @param knn Optional query-to-reference KNN list with `indices` and
#'   `distances`, usually returned by `nn(reference_data, new_data, k)`.
#' @param reference_data Reference observations in the same preprocessing space
#'   used to fit `reference_layout`. Required only when `knn` is `NULL`.
#' @param new_data Query observations in the same preprocessing space as
#'   `reference_data`. Required only when `knn` is `NULL`.
#' @param k Number of reference neighbors used for transform KNN. If `NULL`,
#'   uses at least `3 * perplexity`, matching the usual t-SNE affinity width.
#' @param perplexity Transform perplexity. The openTSNE default is 5.
#' @param initialization Initial query placement: `"median"` from reference
#'   neighbors, inverse-distance `"weighted"`, or `"random"`.
#' @param Y_init Optional initial query embedding matrix. When supplied, it
#'   overrides `initialization`.
#' @param n_iter Number of normal transform optimization iterations.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param learning_rate Transform learning rate.
#' @param early_exaggeration Early exaggeration multiplier.
#' @param exaggeration Normal transform exaggeration multiplier.
#' @param initial_momentum Momentum used during early exaggeration.
#' @param final_momentum Momentum used during normal optimization.
#' @param max_grad_norm Maximum per-query gradient norm. Use `Inf` to disable.
#' @param max_step_norm Maximum per-query step norm. Use `Inf` to disable.
#' @param n_negatives Number of sampled reference points for repulsion when the
#'   reference set is larger than `exact_repulsion_threshold`.
#' @param exact_repulsion_threshold Use exact query-reference repulsion at or
#'   below this reference count.
#' @param n_threads Number of CPU worker threads for the native optimizer.
#' @param seed Random seed.
#' @param backend Backend used for query KNN when `knn` is `NULL`; `"metal"`
#'   also runs the fixed-reference transform optimizer in native Metal. CUDA
#'   transform is planned but intentionally errors until implemented, so CPU
#'   work is never reported as CUDA.
#' @param verbose Print native optimizer progress.
#' @return A numeric matrix with one row per query observation.
#' @export
transform_tsne <- function(reference_layout,
                           knn = NULL,
                           reference_data = NULL,
                           new_data = NULL,
                           k = NULL,
                           perplexity = 5,
                           initialization = c("median", "weighted", "random"),
                           Y_init = NULL,
                           n_iter = 250L,
                           early_exaggeration_iter = 0L,
                           learning_rate = 0.1,
                           early_exaggeration = 4,
                           exaggeration = 1.5,
                           initial_momentum = 0.8,
                           final_momentum = 0.8,
                           max_grad_norm = 0.25,
                           max_step_norm = Inf,
                           n_negatives = NULL,
                           exact_repulsion_threshold = 4096L,
                           n_threads = NULL,
                           seed = 4L,
                           backend = "auto",
                           verbose = FALSE) {
  if (inherits(reference_layout, "fastEmbedR_embedding")) {
    reference_layout <- reference_layout$layout
  }
  initialization <- match.arg(initialization)
  backend <- as.character(backend)[1L]
  optimizer_backend <- resolve_tsne_transform_backend(backend)
  reference_layout <- transform_embedding_matrix(
    reference_layout,
    "reference_layout",
    min_rows = 1L
  )
  perplexity <- as.numeric(perplexity)
  if (length(perplexity) != 1L || is.na(perplexity) || !is.finite(perplexity) || perplexity <= 0) {
    stop("`perplexity` must be a positive number.", call. = FALSE)
  }

  raw_backend <- "precomputed"
  exact <- NA
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
      k <- min(nrow(reference_data), max(25L, ceiling(3 * perplexity)))
    }
    k <- transform_embedding_k(k, nrow(reference_data))
    raw_knn <- fastEmbedR::nn(
      reference_data,
      new_data,
      k = k,
      backend = backend,
      n_threads = n_threads
    )
    projection <- transform_projection_knn(
      raw_knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    raw_backend <- attr(raw_knn, "backend")
    exact <- attr(raw_knn, "exact")
  } else {
    projection <- transform_projection_knn(
      knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    raw_backend <- attr(knn, "backend")
    exact <- attr(knn, "exact")
  }
  if (is.null(raw_backend) || length(raw_backend) == 0L || is.na(raw_backend)) {
    raw_backend <- "precomputed"
  }

  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  n_threads <- as.integer(n_threads)
  if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 0L) {
    stop("`n_threads` must be NULL or a non-negative integer.", call. = FALSE)
  }
  n_iter <- as.integer(n_iter)
  early_exaggeration_iter <- as.integer(early_exaggeration_iter)
  exact_repulsion_threshold <- as.integer(exact_repulsion_threshold)
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 0L ||
      length(early_exaggeration_iter) != 1L || is.na(early_exaggeration_iter) ||
      early_exaggeration_iter < 0L ||
      n_iter + early_exaggeration_iter < 1L) {
    stop("Transform iteration counts must be non-negative and sum to at least one.", call. = FALSE)
  }
  if (length(exact_repulsion_threshold) != 1L || is.na(exact_repulsion_threshold)) {
    exact_repulsion_threshold <- 4096L
  }
  if (is.null(n_negatives)) {
    n_negatives <- if (nrow(reference_layout) <= exact_repulsion_threshold) {
      nrow(reference_layout)
    } else {
      min(256L, nrow(reference_layout))
    }
  }
  n_negatives <- as.integer(n_negatives)
  if (length(n_negatives) != 1L || is.na(n_negatives) || n_negatives < 1L) {
    stop("`n_negatives` must be NULL or a positive integer.", call. = FALSE)
  }

  init <- !is.null(Y_init)
  y_init <- if (init) {
    y_init <- transform_embedding_matrix(Y_init, "Y_init", min_rows = nrow(projection$indices))
    if (nrow(y_init) != nrow(projection$indices) || ncol(y_init) != ncol(reference_layout)) {
      stop("`Y_init` must have one row per query and the same columns as `reference_layout`.", call. = FALSE)
    }
    y_init
  } else {
    matrix(0, 0L, 0L)
  }

  out <- if (identical(optimizer_backend, "metal")) {
    if (ncol(reference_layout) != 2L) {
      stop("Metal t-SNE transform currently supports only two-dimensional reference layouts.", call. = FALSE)
    }
    transform_tsne_metal_cpp(
      reference_layout,
      projection$indices,
      projection$distances,
      y_init,
      init,
      initialization,
      perplexity,
      n_iter,
      early_exaggeration_iter,
      as.numeric(learning_rate),
      as.numeric(early_exaggeration),
      as.numeric(exaggeration),
      as.numeric(initial_momentum),
      as.numeric(final_momentum),
      as.numeric(max_grad_norm),
      as.numeric(max_step_norm),
      n_negatives,
      exact_repulsion_threshold,
      as.integer(seed)
    )
  } else {
    transform_tsne_cpp(
      reference_layout,
      projection$indices,
      projection$distances,
      y_init,
      init,
      initialization,
      perplexity,
      n_iter,
      early_exaggeration_iter,
      as.numeric(learning_rate),
      as.numeric(early_exaggeration),
      as.numeric(exaggeration),
      as.numeric(initial_momentum),
      as.numeric(final_momentum),
      as.numeric(max_grad_norm),
      as.numeric(max_step_norm),
      n_negatives,
      exact_repulsion_threshold,
      n_threads,
      as.integer(seed),
      isTRUE(verbose)
    )
  }
  layout <- out$Y
  colnames(layout) <- colnames(reference_layout)
  attr(layout, "backend") <- optimizer_backend
  attr(layout, "nn_backend") <- raw_backend
  attr(layout, "exact") <- if (is.null(exact)) NA else isTRUE(exact)
  attr(layout, "k") <- as.integer(ncol(projection$indices))
  attr(layout, "transform") <- "opentsne_style_fixed_reference"
  attr(layout, "fastEmbedR_config") <- list(
    method = "transform_tsne",
    backend = optimizer_backend,
    backend_requested = backend,
    nn_backend = raw_backend,
    n_reference = nrow(reference_layout),
    n_query = nrow(layout),
    k = as.integer(ncol(projection$indices)),
    perplexity = perplexity,
    n_iter = n_iter,
    early_exaggeration_iter = early_exaggeration_iter,
    learning_rate = as.numeric(learning_rate),
    early_exaggeration = as.numeric(early_exaggeration),
    exaggeration = as.numeric(exaggeration),
    initial_momentum = as.numeric(initial_momentum),
    final_momentum = as.numeric(final_momentum),
    max_grad_norm = as.numeric(max_grad_norm),
    max_step_norm = as.numeric(max_step_norm),
    n_negatives = out$n_negatives,
    exact_repulsion_threshold = exact_repulsion_threshold,
    initialization = out$initialization,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    n_threads = if (is.null(out$n_threads)) NA_integer_ else out$n_threads,
    seed = as.integer(seed),
    provenance = "openTSNE_transform_design_native_cpp"
  )
  layout
}

resolve_tsne_transform_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    backend <- "auto"
  }
  if (backend %in% c("auto", "cpu")) {
    return("cpu")
  }
  if (identical(backend, "metal")) {
    if (!metal_metric_available()) {
      stop("Metal t-SNE transform backend is not available on this system.", call. = FALSE)
    }
    return("metal")
  }
  if (identical(backend, "gpu")) {
    if (metal_metric_available()) {
      return("metal")
    }
    stop(
      "No native GPU t-SNE transform backend is available. CUDA transform is planned but not implemented in this build.",
      call. = FALSE
    )
  }
  if (identical(backend, "cuda")) {
    stop(
      "CUDA t-SNE transform backend is planned but not implemented yet; fastEmbedR will not run CPU code and report it as CUDA.",
      call. = FALSE
    )
  }
  "cpu"
}

#' Landmark t-SNE with openTSNE-style transform
#'
#' `landmark_tsne()` embeds a subset of observations with `tsne()` or
#' `infotsne()`, then places the remaining observations with `transform_tsne()`.
#' This mirrors the practical landmark workflow exposed by openTSNE while
#' keeping the implementation native to this package.
#'
#' @param data Numeric matrix/data frame with observations in rows.
#' @param labels Optional labels used only for metadata and optional scoring.
#' @param landmarks `TRUE` for an automatic subset, a fraction such as `0.5`, a
#'   landmark count, or explicit row indices.
#' @param reference_method `"infotsne"` for the scalable reference optimizer or
#'   `"tsne"` for the Rtsne-neighbors-style exact reference optimizer.
#' @inheritParams tsne
#' @param transform_k Number of landmark neighbors used to place non-landmarks.
#' @param transform_perplexity Perplexity used by `transform_tsne()`.
#' @param transform_iter Number of normal transform iterations.
#' @param transform_early_exaggeration_iter Number of transform early
#'   exaggeration iterations.
#' @param transform_n_negatives Number of sampled reference negatives used by
#'   `transform_tsne()` on large landmark sets.
#' @param initialization Initial placement for transformed observations.
#' @param backend KNN backend. `"metal"` also runs the non-landmark transform
#'   optimizer in native Metal. CUDA transform is planned but unavailable and
#'   is not silently replaced by CPU.
#' @return A `fastEmbedR_embedding` object.
#' @export
landmark_tsne <- function(data,
                          labels = NULL,
                          landmarks = TRUE,
                          reference_method = c("infotsne", "tsne"),
                          n_neighbors = NULL,
                          perplexity = NULL,
                          n_components = 2L,
                          standardize = TRUE,
                          pca_dims = NULL,
                          seed = 4L,
                          backend = "auto",
                          transform_k = NULL,
                          transform_perplexity = 5,
                          transform_iter = 250L,
                          transform_early_exaggeration_iter = 0L,
                          transform_n_negatives = NULL,
                          initialization = c("median", "weighted", "random"),
                          silhouette_sample = NULL,
                          preserve_sample = NULL,
                          preserve_k = NULL,
                          keep_knn = FALSE,
                          verbose = FALSE,
                          ...) {
  reference_method <- match.arg(reference_method)
  initialization <- match.arg(initialization)
  backend <- as.character(backend)[1L]
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = if (backend %in% c("cuda", "metal")) backend else "cpu"
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (!is.null(labels) && length(labels) != n) {
    stop("`labels` must have one entry per row of `data`.", call. = FALSE)
  }
  if (is.null(n_neighbors)) {
    n_neighbors <- auto_tsne_k(n, perplexity)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
      stop("`n_neighbors` must be a positive integer smaller than `nrow(data)`.", call. = FALSE)
    }
  }

  landmark_indices <- resolve_landmarks(landmarks, x, seed)
  if (is.null(landmark_indices)) {
    return(if (identical(reference_method, "infotsne")) {
      infotsne(
        x,
        labels = labels,
        n_neighbors = n_neighbors,
        perplexity = perplexity,
        n_components = n_components,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = backend,
        silhouette_sample = silhouette_sample,
        preserve_sample = preserve_sample,
        preserve_k = preserve_k,
        keep_knn = keep_knn,
        verbose = verbose,
        ...
      )
    } else {
      tsne(
        x,
        labels = labels,
        n_neighbors = n_neighbors,
        perplexity = perplexity,
        n_components = n_components,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = backend,
        silhouette_sample = silhouette_sample,
        preserve_sample = preserve_sample,
        preserve_k = preserve_k,
        keep_knn = keep_knn,
        verbose = verbose,
        ...
      )
    })
  }

  x_landmarks <- x[landmark_indices, , drop = FALSE]
  n_landmarks <- nrow(x_landmarks)
  landmark_neighbors <- min(n_neighbors, n_landmarks - 1L)
  landmark_labels <- if (is.null(labels)) NULL else labels[landmark_indices]
  reference_time <- system.time({
    reference_fit <- if (identical(reference_method, "infotsne")) {
      infotsne(
        x_landmarks,
        labels = landmark_labels,
        n_neighbors = landmark_neighbors,
        perplexity = perplexity,
        n_components = n_components,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = backend,
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = keep_knn,
        verbose = verbose,
        ...
      )
    } else {
      tsne(
        x_landmarks,
        labels = landmark_labels,
        n_neighbors = landmark_neighbors,
        perplexity = perplexity,
        n_components = n_components,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = backend,
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = keep_knn,
        verbose = verbose,
        ...
      )
    }
  })

  non_landmarks <- setdiff(seq_len(n), landmark_indices)
  if (is.null(transform_k)) {
    transform_k <- min(n_landmarks, max(25L, ceiling(3 * transform_perplexity)))
  }
  transform_k <- transform_embedding_k(transform_k, n_landmarks)
  projection_knn <- NULL
  projection_time <- system.time({
    projection_knn <- fastEmbedR::nn(
      x_landmarks,
      x[non_landmarks, , drop = FALSE],
      k = transform_k,
      backend = backend
    )
  })
  transform_time <- system.time({
    projected <- transform_tsne(
      reference_fit$layout,
      knn = projection_knn,
      perplexity = transform_perplexity,
      initialization = initialization,
      n_iter = transform_iter,
      early_exaggeration_iter = transform_early_exaggeration_iter,
      n_negatives = transform_n_negatives,
      n_threads = NULL,
      seed = seed + 1009L,
      backend = backend,
      verbose = verbose
    )
  })

  layout <- matrix(NA_real_, nrow = n, ncol = n_components)
  layout[landmark_indices, ] <- reference_fit$layout
  layout[non_landmarks, ] <- projected
  colnames(layout) <- colnames(reference_fit$layout)

  scoring_time <- structure(rep(0, 5), names = names(system.time({})))
  score_preserve_k <- if (is.null(preserve_k)) min(auto_k(n), n - 1L) else {
    min(as.integer(preserve_k), n - 1L)
  }
  preserve_keep <- sample_indices(n, preserve_sample, seed)
  score_indices <- matrix(integer(0L), nrow = 0L, ncol = score_preserve_k)
  if (length(preserve_keep) > 0L) {
    scoring_time <- system.time({
      score_indices <- sampled_score_indices(
        x,
        preserve_keep,
        score_preserve_k,
        backend
      )
    })
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
    backend = "cpu"
  )

  timings <- rbind(
    preprocess = preprocess_time,
    reference_embedding = reference_time,
    landmark_projection_knn = projection_time,
    transform = transform_time,
    scoring_knn = scoring_time
  )
  transform_cfg <- attr(projected, "fastEmbedR_config")
  reference_params <- reference_fit$parameters
  selection_method <- attr(landmark_indices, "selection_method")
  if (is.null(selection_method)) selection_method <- "indices"
  metrics <- data.frame(
    method = "landmark_tsne",
    reference_method = reference_method,
    n = n,
    p = ncol(x),
    n_neighbors = n_neighbors,
    perplexity = if (is.null(perplexity)) reference_params$perplexity else perplexity,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    reference_embedding_elapsed = reference_time["elapsed"],
    landmark_projection_knn_elapsed = projection_time["elapsed"],
    transform_elapsed = transform_time["elapsed"],
    scoring_knn_elapsed = scoring_time["elapsed"],
    landmark = TRUE,
    n_landmarks = n_landmarks,
    landmark_fraction = n_landmarks / n,
    transform_k = transform_k,
    scores,
    stringsAsFactors = FALSE
  )
  parameters <- c(
    list(
      method = "landmark_tsne",
      reference_method = reference_method,
      n = n,
      p = ncol(x),
      n_neighbors = n_neighbors,
      k = n_neighbors + 1L,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      nn_backend = reference_params$nn_backend,
      projection_nn_backend = attr(projection_knn, "backend"),
      backend = attr(projected, "backend"),
      transform_backend = attr(projected, "backend"),
      landmark = TRUE,
      n_landmarks = n_landmarks,
      landmark_fraction = n_landmarks / n,
      landmark_selection = selection_method,
      transform_k = transform_k,
      transform_perplexity = transform_perplexity,
      transform_iter = as.integer(transform_iter),
      transform_early_exaggeration_iter = as.integer(transform_early_exaggeration_iter),
      transform_optimizer = transform_cfg$optimizer,
      transform_repulsion = transform_cfg$repulsion,
      transform_n_negatives = transform_cfg$n_negatives,
      transform_initialization = transform_cfg$initialization,
      keep_knn = keep_knn,
      provenance = "openTSNE_landmark_transform_design_native_cpp"
    ),
    prepared$preprocess
  )
  out <- list(
    layout = layout,
    labels = labels,
    method = "landmark_tsne",
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = NULL,
    landmarks = list(
      indices = landmark_indices,
      layout = reference_fit$layout,
      reference_fit = reference_fit,
      projection_knn = if (isTRUE(keep_knn)) projection_knn else NULL,
      transform = attr(projected, "fastEmbedR_config")
    ),
    preprocess = prepared$preprocess
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
