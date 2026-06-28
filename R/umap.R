#' Run UMAP from a data matrix or precomputed KNN
#'
#' `umap()` is a small convenience wrapper. If `data` is already a KNN object
#' returned by [faissR::nn()], it calls [umap_knn()] directly. Otherwise it preprocesses
#' the data, computes KNN once, and embeds from that KNN graph.
#'
#' @param data Numeric matrix/data frame, or a KNN object returned by [faissR::nn()].
#' @param n_neighbors Number of non-self neighbours. `NULL` chooses the package
#'   default for the data size.
#' @param n_components Output dimensionality.
#' @param standardize Center and scale columns before KNN when `data` is a
#'   matrix. Defaults to `FALSE` so one-call results match a KNN object computed
#'   from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input.
#' @param nn Optional precomputed KNN result when `data` is a matrix.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. KNN is
#'   delegated to faissR with automatic method/tuning selection through an
#'   internal bridge: CPU and Metal request faissR CPU HNSW with
#'   `target_recall = 0.99`, while CUDA requests faissR CUDA
#'   `method = "auto"` with `target_recall = 0.99`.
#'   GPU requests must resolve to a real native backend; the package does not
#'   relabel CPU work as GPU.
#' @param n_threads Number of CPU worker threads for KNN and CPU UMAP.
#' @param keep_knn Keep KNN matrices in the returned object.
#' @param graph_mode Graph weighting mode. `"binary"` uses a symmetric
#'   unit-weight graph. `"fuzzy"` uses standard UMAP fuzzy graph weights.
#' @param verbose Print progress.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' fit <- umap(x, n_neighbors = 15, seed = 1)
#' plot(fit)
#' @export
umap <- function(data,
                 n_neighbors = NULL,
                 n_components = 2L,
                 standardize = FALSE,
                 pca_dims = NULL,
                 metric = c("euclidean", "cosine"),
                 nn = NULL,
                 seed = 4L,
                 backend = c("cpu", "cuda", "metal"),
                 n_threads = NULL,
                 keep_knn = FALSE,
                 graph_mode = c("binary", "fuzzy"),
                 verbose = FALSE) {
  backend <- resolve_embedding_backend(backend)
  graph_mode <- match.arg(graph_mode)
  n_components <- validate_n_components(n_components)
  keep_knn <- isTRUE(keep_knn)
  input_is_float32 <- is_float32_matrix(data)

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
        n_threads = n_threads,
        graph_mode = graph_mode
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
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
      stringsAsFactors = FALSE
    )
    out <- list(
      layout = layout,
      labels = NULL,
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
      backend = backend
    )
  })
  x <- prepared$data
  metric <- resolve_embedding_metric(metric, x)
  n <- nrow(x)
  if (is.null(n_neighbors)) {
    n_neighbors <- auto_embedding_k(n, method = "umap", include_self = FALSE)
  }

  knn_time <- system.time({
    knn_result <- if (is.null(nn)) {
      knn_policy <- fastembedr_embedding_nn_policy(backend)
      fastembedr_nn_without_self(
        x,
        k = as.integer(n_neighbors),
        backend = knn_policy$backend,
        method = knn_policy$method,
        metric = metric,
        output = fastembedr_faiss_float_output(x, knn_policy$backend),
        n_threads = n_threads,
        tuning = knn_policy$tuning,
        target_recall = knn_policy$target_recall
      )
    } else {
      coerce_knn_input(nn)
    }
  })

  embedding_time <- system.time({
    layout <- fast_knn_umap(
      knn_result,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads,
      graph_mode = graph_mode
    )
  })
  layout <- finalize_embedding_layout(
    layout,
    "UMAP",
    return_float32 = input_is_float32 && is_float32_matrix(x)
  )
  cfg <- attr(layout, "fastEmbedR_config")
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
    stringsAsFactors = FALSE
  )
  parameters <- c(
    cfg,
    list(
      standardize = isTRUE(standardize),
      pca_dims = prepared$preprocess$pca_dims,
      preprocess = prepared$preprocess,
      graph_mode = graph_mode,
      metric = metric,
      nn_backend = if (is.null(attr(knn_result, "backend"))) "supplied" else attr(knn_result, "backend")
    )
  )
  out <- list(
    layout = layout,
    labels = NULL,
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
                          landmarks = 0.5,
                          n_neighbors = NULL,
                          n_components = 2L,
                          standardize = TRUE,
                          pca_dims = NULL,
                          seed = 4L,
                          backend = c("cpu", "cuda", "metal"),
                          transform_k = NULL,
                          n_threads = NULL,
                          keep_knn = FALSE,
                          verbose = FALSE) {
  backend <- resolve_embedding_backend(backend)
  n_components <- validate_n_components(n_components)
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = backend
    )
  })
  x <- prepared$data
  n <- nrow(x)
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
      n_neighbors = n_neighbors,
      n_components = n_components,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      n_threads = n_threads,
      keep_knn = keep_knn,
      verbose = verbose
    ))
  }

  x_landmarks <- x[landmark_indices, , drop = FALSE]
  n_landmarks <- nrow(x_landmarks)
  landmark_neighbors <- min(n_neighbors, n_landmarks - 1L)
  reference_time <- system.time({
    reference_fit <- umap(
      x_landmarks,
      n_neighbors = landmark_neighbors,
      n_components = n_components,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      n_threads = n_threads,
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
      backend = embedding_knn_backend(backend),
      seed = seed + 503L,
      n_threads = n_threads,
      landmark_layout = reference_fit$layout,
      all_data = x,
      landmark_indices = landmark_indices,
      query_rows = non_landmarks
    )
    projected <- attr(projection_knn, "projected_layout", exact = TRUE)
    affine_projected <- landmark_affine_projection(
      x_landmarks,
      x[non_landmarks, , drop = FALSE],
      reference_fit$layout,
      projection_knn,
      n_threads = n_threads
    )
    if (is.matrix(affine_projected) &&
        nrow(affine_projected) == length(non_landmarks) &&
        ncol(affine_projected) == n_components) {
      projected <- affine_projected
    }
    if (is.null(projected)) {
      project_backend <- if (identical(backend, "metal") && isTRUE(embedding_metal_available_cpp())) {
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

  refinement_time <- zero_proc_time()
  refinement_backend <- NA_character_
  refinement_epochs <- getOption("fastEmbedR.landmark_umap_refine_epochs", 50L)
  refinement_epochs <- suppressWarnings(as.integer(refinement_epochs))
  if (length(refinement_epochs) != 1L || is.na(refinement_epochs) || refinement_epochs < 0L) {
    refinement_epochs <- 50L
  }
  if (refinement_epochs > 0L && length(non_landmarks) > 0L) {
    refinement_time <- system.time({
      projection_global_indices <- matrix(
        as.integer(landmark_indices[as.integer(projection_knn$indices)]),
        nrow = nrow(projection_knn$indices),
        ncol = ncol(projection_knn$indices)
      )
      ref_params <- reference_fit$parameters
      ref_min_dist <- as.numeric(ref_params$min_dist %||% 0.01)
      ref_negative_sample_rate <- as.integer(ref_params$negative_sample_rate %||% 5L)
      ref_learning_rate <- as.numeric(ref_params$learning_rate %||% 1)
      ref_repulsion_strength <- as.numeric(ref_params$repulsion_strength %||% 1)
      if (!is.finite(ref_min_dist) || ref_min_dist < 0) ref_min_dist <- 0.01
      if (is.na(ref_negative_sample_rate) || ref_negative_sample_rate < 0L) ref_negative_sample_rate <- 5L
      if (!is.finite(ref_learning_rate) || ref_learning_rate <= 0) ref_learning_rate <- 1
      if (!is.finite(ref_repulsion_strength) || ref_repulsion_strength <= 0) ref_repulsion_strength <- 1
      use_metal_refinement <- n_components == 2L &&
        identical(backend, "metal") &&
        isTRUE(embedding_metal_available_cpp())
      if (use_metal_refinement) {
        refined_result <- tryCatch(
          {
            refined <- knn_umap_refine_rows_metal_cpp(
              projection_global_indices,
              projection_knn$distances,
              as.integer(non_landmarks),
              layout,
              as.integer(refinement_epochs),
              ref_min_dist,
              ref_negative_sample_rate,
              ref_learning_rate,
              ref_repulsion_strength,
              as.integer(seed + 2003L)
            )
            list(layout = refined, backend = "metal")
          },
          error = function(e) {
            msg <- conditionMessage(e)
            if (identical(backend, "metal")) {
              stop("Metal UMAP landmark refinement failed: ", msg, call. = FALSE)
            }
            warning(
              "Metal UMAP landmark refinement failed; using CPU refinement and reporting backend='cpu': ",
              msg,
              call. = FALSE
            )
            refined <- knn_umap_refine_rows_cpp(
              projection_global_indices,
              projection_knn$distances,
              as.integer(non_landmarks),
              layout,
              as.integer(refinement_epochs),
              ref_min_dist,
              ref_negative_sample_rate,
              ref_learning_rate,
              ref_repulsion_strength,
              as.integer(normalize_nn_threads(n_threads)),
              as.integer(seed + 2003L),
              isTRUE(verbose)
            )
            list(layout = refined, backend = "cpu")
          }
        )
        layout <- refined_result$layout
        refinement_backend <- refined_result$backend
      } else {
        refinement_backend <- "cpu"
        layout <- knn_umap_refine_rows_cpp(
          projection_global_indices,
          projection_knn$distances,
          as.integer(non_landmarks),
          layout,
          as.integer(refinement_epochs),
          ref_min_dist,
          ref_negative_sample_rate,
          ref_learning_rate,
          ref_repulsion_strength,
          as.integer(normalize_nn_threads(n_threads)),
          as.integer(seed + 2003L),
          isTRUE(verbose)
        )
      }
      layout[landmark_indices, ] <- reference_fit$layout
      colnames(layout) <- colnames(reference_fit$layout)
    })
  }

  zero <- zero_proc_time()
  timings <- rbind(
    preprocess = preprocess_time,
    reference_embedding = reference_time,
    landmark_projection_knn = projection_time,
    landmark_refinement = refinement_time,
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
    landmark_refinement_elapsed = refinement_time[["elapsed"]],
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
      landmark_refinement = if (refinement_epochs > 0L) "fixed_landmark_umap_rows" else "none",
      landmark_refinement_epochs = as.integer(refinement_epochs),
      landmark_refinement_backend = if (refinement_epochs > 0L) refinement_backend else NA_character_,
      keep_knn = keep_knn,
      provenance = "UMAP_landmark_projection_native_cpp"
    ),
    prepared$preprocess
  )
  out <- list(
    layout = layout,
    labels = NULL,
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
