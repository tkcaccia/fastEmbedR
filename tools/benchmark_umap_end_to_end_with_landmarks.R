#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

source_benchmark_helpers <- function() {
  path <- file.path("tools", "benchmark_large_best_methods.R")
  lines <- readLines(path, warn = FALSE)
  main_start <- grep("^methods_arg <-", lines)[1L]
  if (length(main_start) != 1L || is.na(main_start)) {
    stop("Could not find benchmark helper boundary in ", path, call. = FALSE)
  }
  eval(parse(text = paste(lines[seq_len(main_start - 1L)], collapse = "\n")), envir = .GlobalEnv)
}

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) default else sub(prefix, "", value[[1L]], fixed = TRUE)
}

parse_csv <- function(name, default) {
  value <- parse_scalar(name, default)
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

int_arg <- function(name, default) as.integer(as.numeric(parse_scalar(name, as.character(default))))
num_arg <- function(name, default) as.numeric(parse_scalar(name, as.character(default)))

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_.-]+", "_", x)
}

bind_rows_base <- function(...) {
  dfs <- list(...)
  if (length(dfs) == 1L && is.list(dfs[[1L]]) && !is.data.frame(dfs[[1L]])) {
    dfs <- dfs[[1L]]
  }
  dfs <- dfs[vapply(dfs, function(x) is.data.frame(x) && nrow(x) > 0L, logical(1))]
  if (length(dfs) == 0L) return(data.frame())
  cols <- unique(unlist(lapply(dfs, names), use.names = FALSE))
  dfs <- lapply(dfs, function(df) {
    missing <- setdiff(cols, names(df))
    for (name in missing) df[[name]] <- NA
    df[, cols, drop = FALSE]
  })
  do.call(rbind, dfs)
}

timed_value <- function(expr, timeout_sec = 0) {
  invisible(gc())
  rss_before <- current_rss_mb()
  value <- NULL
  elapsed <- system.time({
    value <- with_timeout(force(expr), timeout_sec)
  })[["elapsed"]]
  rss_after <- current_rss_mb()
  list(
    value = value,
    elapsed = as.numeric(elapsed),
    rss_before_mb = rss_before,
    rss_after_mb = rss_after,
    rss_delta_mb = if (is.finite(rss_before) && is.finite(rss_after)) rss_after - rss_before else NA_real_
  )
}

label_colors <- function(labels, alpha = 0.45) {
  if (is.null(labels)) return(rep(grDevices::adjustcolor("black", alpha.f = alpha), 1L))
  labels <- factor(labels)
  n <- nlevels(labels)
  base <- if (n <= 8L) grDevices::hcl.colors(n, "Dark 3") else grDevices::hcl.colors(n, "Spectral")
  stats::setNames(grDevices::adjustcolor(base, alpha.f = alpha), levels(labels))[as.character(labels)]
}

backend_matches_request <- function(requested, used) {
  if (is.null(used) || length(used) == 0L || is.na(used[[1L]])) return(FALSE)
  used <- as.character(used[[1L]])
  if (requested %in% c("cuda", "metal")) {
    identical(used, requested) || startsWith(used, paste0(requested, "+"))
  } else {
    identical(used, requested)
  }
}

run_label <- function(method, backend_used, mode = "full") {
  if (identical(method, "uwot_umap_fast_sgd_own_nn")) return("uwot own NN")
  if (identical(mode, "landmark_projection")) return(paste0("landmark50 projection ", backend_used))
  if (identical(mode, "landmark_refined_selective")) return(paste0("landmark50 selective ", backend_used))
  paste0("full ", backend_used)
}

umap_config_for <- function(n, k, backend = "cpu") {
  fastEmbedR:::fast_knn_umap_config(as.integer(n), as.integer(k), backend)
}

score_embedding_layout <- function(dataset,
                                   layout,
                                   q,
                                   seed,
                                   method,
                                   backend,
                                   global_sample_size) {
  fastEmbedR::evaluate_embedding(
    dataset$x[q, , drop = FALSE],
    layout[q, , drop = FALSE],
    labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
    k = c(15L, 30L, 50L),
    sample_size_for_global_metrics = min(global_sample_size, length(q)),
    seed = seed,
    method = method,
    backend = backend,
    dataset = dataset$name
  )
}

result_row <- function(dataset,
                       method,
                       package,
                       backend_requested,
                       backend_used,
                       status,
                       error_message,
                       seed,
                       k,
                       preprocess_time_sec,
                       knn_time_sec,
                       knn_backend,
                       embedding_time_sec,
                       projection_time_sec,
                       refinement_knn_time_sec,
                       refinement_time_sec,
                       total_time_sec,
                       metrics,
                       layout_path,
                       mode,
                       extra = list()) {
  components <- c(
    knn_time_sec,
    embedding_time_sec,
    projection_time_sec,
    refinement_knn_time_sec,
    refinement_time_sec
  )
  post_preprocess_total_sec <- if (all(is.na(components))) NA_real_ else sum(components, na.rm = TRUE)
  base <- data.frame(
    dataset = dataset$name,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    method = method,
    package = package,
    mode = mode,
    run_label = run_label(method, backend_used, mode),
    backend_requested = backend_requested,
    backend_used = backend_used,
    status = status,
    error_message = error_message,
    seed = seed,
    k = k,
    preprocess_time_sec = preprocess_time_sec,
    knn_time_sec = knn_time_sec,
    knn_backend = knn_backend,
    embedding_time_sec = embedding_time_sec,
    projection_time_sec = projection_time_sec,
    refinement_knn_time_sec = refinement_knn_time_sec,
    refinement_time_sec = refinement_time_sec,
    post_preprocess_total_sec = post_preprocess_total_sec,
    total_time_sec = total_time_sec,
    trustworthiness = if (is.null(metrics)) NA_real_ else metrics$trustworthiness[[1L]],
    continuity = if (is.null(metrics)) NA_real_ else metrics$continuity[[1L]],
    knn_preservation_15 = if (is.null(metrics)) NA_real_ else metrics$knn_preservation_15[[1L]],
    knn_preservation_30 = if (is.null(metrics)) NA_real_ else metrics$knn_preservation_30[[1L]],
    knn_preservation_50 = if (is.null(metrics)) NA_real_ else metrics$knn_preservation_50[[1L]],
    label_knn_accuracy = if (is.null(metrics)) NA_real_ else metrics$label_knn_accuracy[[1L]],
    silhouette = if (is.null(metrics)) NA_real_ else metrics$silhouette[[1L]],
    distance_spearman = if (is.null(metrics)) NA_real_ else metrics$distance_spearman[[1L]],
    layout_path = layout_path,
    internal_nn_in_embedding_time = FALSE,
    timing_note = NA_character_,
    stringsAsFactors = FALSE
  )
  for (name in names(extra)) base[[name]] <- extra[[name]]
  base
}

save_embedding_layout <- function(layout, dataset_name, label, seed, layout_dir) {
  path <- file.path(layout_dir, paste0(safe_name(paste(dataset_name, label, seed, sep = "_")), ".rds"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(as.matrix(layout), path, version = 2)
  path
}

run_fastembedr_full <- function(dataset,
                                backend,
                                k,
                                seed,
                                q,
                                cfg,
                                timeout_sec,
                                global_sample_size,
                                layout_dir) {
  tryCatch({
    message("  fastEmbedR full backend=", backend, " with fastEmbedR nn")
    knn_measured <- timed_value({
      raw <- fastEmbedR:::nn_without_self(dataset$x, k = k, backend = backend)
      out <- fastEmbedR:::normalize_supplied_knn(raw, nrow(dataset$x), k)
      attr(out, "backend") <- attr(raw, "backend")
      out
    }, timeout_sec)
    knn_backend_used <- attr(knn_measured$value, "backend")
    if (is.null(knn_backend_used)) knn_backend_used <- backend
    layout_measured <- measure_layout(
      fastEmbedR::embed_knn(knn_measured$value, method = "umap", seed = seed, backend = backend),
      nrow(dataset$x),
      timeout_sec
    )
    layout <- layout_measured$layout
    config <- attr(layout, "fastEmbedR_config")
    backend_used <- if (is.list(config) && !is.null(config$backend)) as.character(config$backend) else backend
    if (backend %in% c("cuda", "metal") && !backend_matches_request(backend, backend_used)) {
      stop("Requested backend ", backend, " but backend_used was ", backend_used, call. = FALSE)
    }
    metrics <- score_embedding_layout(dataset, layout, q, seed, "umap", backend_used, global_sample_size)
    layout_path <- save_embedding_layout(layout, dataset$name, paste0("full_", backend_used), seed, layout_dir)
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_own_nn",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = backend_used,
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = knn_measured$elapsed,
      knn_backend = as.character(knn_backend_used),
      embedding_time_sec = layout_measured$elapsed,
      projection_time_sec = 0,
      refinement_knn_time_sec = 0,
      refinement_time_sec = 0,
      total_time_sec = dataset$preprocess_time_sec + knn_measured$elapsed + layout_measured$elapsed,
      metrics = metrics,
      layout_path = layout_path,
      mode = "full",
      extra = list(
        internal_nn_in_embedding_time = TRUE,
        timing_note = "uwot does not expose the KNN split here; embedding_time_sec includes uwot internal NN, graph construction, and optimization",
        n_epochs = cfg$n_epochs,
        min_dist = cfg$min_dist,
        negative_sample_rate = cfg$negative_sample_rate
      )
    )
  }, error = function(e) {
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_own_nn",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = NA_character_,
      status = if (backend %in% c("cuda", "metal")) "backend_unavailable" else "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = NA_real_,
      knn_backend = backend,
      embedding_time_sec = NA_real_,
      projection_time_sec = NA_real_,
      refinement_knn_time_sec = NA_real_,
      refinement_time_sec = NA_real_,
      total_time_sec = NA_real_,
      metrics = NULL,
      layout_path = NA_character_,
      mode = "full"
    )
  })
}

run_uwot_own_nn <- function(dataset,
                            k,
                            seed,
                            q,
                            cfg,
                            timeout_sec,
                            global_sample_size,
                            layout_dir) {
  tryCatch({
    if (!requireNamespace("uwot", quietly = TRUE)) {
      stop("Package `uwot` is not installed.", call. = FALSE)
    }
    message("  uwot fast_sgd with uwot internal NN")
    measured <- measure_layout(
      uwot::umap(
        X = dataset$x,
        n_neighbors = as.integer(k),
        n_components = 2L,
        nn_method = "annoy",
        n_epochs = as.integer(cfg$n_epochs),
        init = "spectral",
        min_dist = cfg$min_dist,
        metric = "euclidean",
        learning_rate = 1,
        repulsion_strength = cfg$repulsion_strength,
        negative_sample_rate = as.integer(cfg$negative_sample_rate),
        fast_sgd = TRUE,
        n_threads = as.integer(cfg$n_threads),
        n_sgd_threads = as.integer(cfg$n_threads),
        ret_model = FALSE,
        verbose = FALSE,
        seed = as.integer(seed)
      ),
      nrow(dataset$x),
      timeout_sec
    )
    metrics <- score_embedding_layout(dataset, measured$layout, q, seed, "uwot_umap_fast_sgd", "uwot_internal_annoy", global_sample_size)
    layout_path <- save_embedding_layout(measured$layout, dataset$name, "uwot_own_nn", seed, layout_dir)
    result_row(
      dataset = dataset,
      method = "uwot_umap_fast_sgd_own_nn",
      package = "uwot",
      backend_requested = "uwot_fast_sgd_own_nn",
      backend_used = "uwot_internal_annoy",
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = 0,
      knn_backend = "uwot_internal_annoy",
      embedding_time_sec = measured$elapsed,
      projection_time_sec = 0,
      refinement_knn_time_sec = 0,
      refinement_time_sec = 0,
      total_time_sec = dataset$preprocess_time_sec + measured$elapsed,
      metrics = metrics,
      layout_path = layout_path,
      mode = "full",
      extra = list(
        n_epochs = cfg$n_epochs,
        min_dist = cfg$min_dist,
        negative_sample_rate = cfg$negative_sample_rate
      )
    )
  }, error = function(e) {
    result_row(
      dataset = dataset,
      method = "uwot_umap_fast_sgd_own_nn",
      package = "uwot",
      backend_requested = "uwot_fast_sgd_own_nn",
      backend_used = NA_character_,
      status = "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = NA_real_,
      knn_backend = "uwot_internal_annoy",
      embedding_time_sec = NA_real_,
      projection_time_sec = NA_real_,
      refinement_knn_time_sec = NA_real_,
      refinement_time_sec = NA_real_,
      total_time_sec = NA_real_,
      metrics = NULL,
      layout_path = NA_character_,
      mode = "full"
    )
  })
}

run_landmark_projection <- function(dataset,
                                    backend,
                                    k,
                                    landmark_fraction,
                                    seed,
                                    q,
                                    timeout_sec,
                                    global_sample_size,
                                    layout_dir) {
  tryCatch({
    n <- nrow(dataset$x)
    n_landmarks <- min(max(2L, as.integer(ceiling(n * landmark_fraction))), n - 1L)
    message("  landmark projection backend=", backend, " with fastEmbedR nn, landmarks=", n_landmarks)
    selection <- timed_value({
      fastEmbedR:::select_landmark_rows(dataset$x, n_landmarks, seed)
    }, timeout_sec)
    landmark_indices <- selection$value
    n_landmarks <- length(landmark_indices)
    x_landmarks <- dataset$x[landmark_indices, , drop = FALSE]
    landmark_neighbors <- min(as.integer(k), n_landmarks - 1L)
    projection_k <- fastEmbedR:::landmark_projection_k(n_landmarks, landmark_neighbors)

    landmark_knn <- timed_value({
      raw <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = backend)
      out <- fastEmbedR:::normalize_supplied_knn(raw, n_landmarks, landmark_neighbors)
      attr(out, "backend") <- attr(raw, "backend")
      out
    }, timeout_sec)
    landmark_knn_backend <- attr(landmark_knn$value, "backend")
    if (is.null(landmark_knn_backend)) landmark_knn_backend <- backend

    landmark_layout <- measure_layout(
      fastEmbedR:::fast_knn_umap_core(
        landmark_knn$value$indices,
        landmark_knn$value$distances,
        seed = seed,
        backend = backend
      ),
      n_landmarks,
      timeout_sec
    )
    config <- attr(landmark_layout$layout, "fastEmbedR_config")
    umap_backend <- if (is.list(config) && !is.null(config$backend)) as.character(config$backend) else backend
    if (backend %in% c("cuda", "metal") && !backend_matches_request(backend, umap_backend)) {
      stop("Requested backend ", backend, " but landmark UMAP backend_used was ", umap_backend, call. = FALSE)
    }

    interpolation <- NULL
    projection_time <- NA_real_
    interpolation_time <- NA_real_
    projection_backend <- NA_character_
    if (identical(backend, "metal")) {
      interpolation <- timed_value({
        fastEmbedR:::fused_landmark_project_layout(
          x_landmarks,
          dataset$x,
          landmark_layout$layout,
          landmark_indices,
          projection_k,
          backend = backend
        )
      }, timeout_sec)
      if (!is.null(interpolation$value)) {
        projection_time <- interpolation$elapsed
        interpolation_time <- 0
        projection_backend <- attr(interpolation$value, "projection_backend")
      }
    }
    if (is.null(interpolation) || is.null(interpolation$value)) {
      projection_knn <- timed_value({
        fastEmbedR::nn(x_landmarks, dataset$x, k = projection_k, backend = backend)
      }, timeout_sec)
      projection_time <- projection_knn$elapsed
      projection_backend <- attr(projection_knn$value, "backend")
      if (is.null(projection_backend)) projection_backend <- backend
      interpolation <- timed_value({
        fastEmbedR:::interpolate_landmark_layout(
          landmark_layout$layout,
          landmark_indices,
          projection_knn$value,
          n,
          backend = backend
        )
      }, timeout_sec)
      interpolation_time <- interpolation$elapsed
    }

    layout <- as.matrix(interpolation$value)
    if (nrow(layout) != n && ncol(layout) == n) layout <- t(layout)
    layout <- layout[, 1:2, drop = FALSE]
    interpolation_backend <- attr(interpolation$value, "interpolation_backend")
    if (is.null(interpolation_backend)) interpolation_backend <- "cpu"
    backend_used <- paste0(umap_backend, "+projection_", projection_backend, "+interp_", interpolation_backend)
    metrics <- score_embedding_layout(dataset, layout, q, seed, "umap_landmark50_projection", backend_used, global_sample_size)
    layout_path <- save_embedding_layout(layout, dataset$name, paste0("landmark50_projection_", backend_used), seed, layout_dir)
    post_pre <- selection$elapsed + landmark_knn$elapsed + landmark_layout$elapsed + projection_time + interpolation_time
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_landmark50_projection",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = backend_used,
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = landmark_knn$elapsed,
      knn_backend = as.character(landmark_knn_backend),
      embedding_time_sec = landmark_layout$elapsed,
      projection_time_sec = selection$elapsed + projection_time + interpolation_time,
      refinement_knn_time_sec = 0,
      refinement_time_sec = 0,
      total_time_sec = dataset$preprocess_time_sec + post_pre,
      metrics = metrics,
      layout_path = layout_path,
      mode = "landmark_projection",
      extra = list(
        landmark_fraction = n_landmarks / n,
        n_landmarks = n_landmarks,
        projection_k = projection_k,
        projection_backend = as.character(projection_backend),
        landmark_selection_time_sec = selection$elapsed,
        landmark_projection_knn_time_sec = projection_time,
        interpolation_time_sec = interpolation_time
      )
    )
  }, error = function(e) {
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_landmark50_projection",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = NA_character_,
      status = if (backend %in% c("cuda", "metal")) "backend_unavailable" else "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = NA_real_,
      knn_backend = backend,
      embedding_time_sec = NA_real_,
      projection_time_sec = NA_real_,
      refinement_knn_time_sec = NA_real_,
      refinement_time_sec = NA_real_,
      total_time_sec = NA_real_,
      metrics = NULL,
      layout_path = NA_character_,
      mode = "landmark_projection"
    )
  })
}

run_landmark_refined_selective <- function(dataset,
                                           backend,
                                           k,
                                           landmark_fraction,
                                           seed,
                                           q,
                                           timeout_sec,
                                           global_sample_size,
                                           layout_dir) {
  tryCatch({
    message("  landmark selective refinement backend=", backend, " with fastEmbedR nn")
    old_opt <- getOption("fastEmbedR.selective_landmark_refinement", NULL)
    options(fastEmbedR.selective_landmark_refinement = TRUE)
    on.exit(options(fastEmbedR.selective_landmark_refinement = old_opt), add = TRUE)
    fit_measured <- timed_value({
      fastEmbedR::umap(
        dataset$x,
        labels = dataset$labels,
        n_neighbors = as.integer(k),
        standardize = FALSE,
        pca_dims = NULL,
        landmarks = landmark_fraction,
        seed = seed,
        backend = backend,
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = FALSE,
        verbose = FALSE
      )
    }, timeout_sec)
    fit <- fit_measured$value
    layout <- fit$layout
    params <- fit$parameters
    backend_used <- params$backend
    if (is.null(backend_used) || is.na(backend_used)) {
      cfg <- attr(layout, "fastEmbedR_config")
      backend_used <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else backend
    }
    if (backend %in% c("cuda", "metal") && !backend_matches_request(backend, backend_used)) {
      stop("Requested backend ", backend, " but backend_used was ", backend_used, call. = FALSE)
    }
    metrics <- score_embedding_layout(dataset, layout, q, seed, "umap_landmark50_selective_refine", backend_used, global_sample_size)
    layout_path <- save_embedding_layout(layout, dataset$name, paste0("landmark50_selective_", backend_used), seed, layout_dir)
    timings <- fit$metrics
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_landmark50_selective",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = as.character(backend_used),
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = as.numeric(timings$knn_elapsed[[1L]]),
      knn_backend = as.character(params$nn_backend),
      embedding_time_sec = as.numeric(timings$embedding_elapsed[[1L]]),
      projection_time_sec = as.numeric(timings$landmark_projection_elapsed[[1L]]),
      refinement_knn_time_sec = as.numeric(timings$landmark_refinement_knn_elapsed[[1L]]),
      refinement_time_sec = as.numeric(timings$landmark_refinement_elapsed[[1L]]),
      total_time_sec = dataset$preprocess_time_sec + as.numeric(timings$elapsed[[1L]]),
      metrics = metrics,
      layout_path = layout_path,
      mode = "landmark_refined_selective",
      extra = list(
        landmark_fraction = as.numeric(params$landmark_fraction),
        n_landmarks = as.integer(params$n_landmarks),
        landmark_refinement = as.character(params$landmark_refinement),
        landmark_refinement_selected = as.integer(params$landmark_refinement_selected),
        landmark_refinement_selected_fraction = as.numeric(params$landmark_refinement_selected_fraction),
        landmark_projection_backend = as.character(params$landmark_projection_backend),
        landmark_interpolation_backend = as.character(params$landmark_interpolation_backend),
        landmark_refinement_backend = as.character(params$landmark_refinement_backend)
      )
    )
  }, error = function(e) {
    result_row(
      dataset = dataset,
      method = "fastEmbedR_umap_landmark50_selective",
      package = "fastEmbedR",
      backend_requested = backend,
      backend_used = NA_character_,
      status = if (backend %in% c("cuda", "metal")) "backend_unavailable" else "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_time_sec = dataset$preprocess_time_sec,
      knn_time_sec = NA_real_,
      knn_backend = backend,
      embedding_time_sec = NA_real_,
      projection_time_sec = NA_real_,
      refinement_knn_time_sec = NA_real_,
      refinement_time_sec = NA_real_,
      total_time_sec = NA_real_,
      metrics = NULL,
      layout_path = NA_character_,
      mode = "landmark_refined_selective"
    )
  })
}

plot_dataset_layouts <- function(dataset, rows, plot_dir, max_points, seed) {
  ok <- rows$status == "success" & file.exists(rows$layout_path)
  rows <- rows[ok, , drop = FALSE]
  if (nrow(rows) == 0L) return(NA_character_)
  preferred <- c(
    "full cpu",
    "full metal",
    "uwot own NN",
    "landmark50 projection cpu",
    "landmark50 projection metal",
    "landmark50 selective cpu",
    "landmark50 selective metal"
  )
  rows$order <- match(rows$run_label, preferred)
  rows$order[is.na(rows$order)] <- seq_len(sum(is.na(rows$order))) + length(preferred)
  rows <- rows[order(rows$order), , drop = FALSE]

  n <- nrow(dataset$x)
  set.seed(seed)
  keep <- if (is.finite(max_points) && max_points > 0L && n > max_points) {
    sort(sample.int(n, max_points))
  } else {
    seq_len(n)
  }
  labels <- if (is.null(dataset$labels)) NULL else dataset$labels[keep]
  cols <- label_colors(labels, alpha = if (length(keep) > 50000L) 0.28 else 0.55)
  order <- sample.int(length(keep))

  path <- file.path(plot_dir, paste0(safe_name(dataset$name), "_end_to_end_landmarks.png"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = max(1900, 430 * nrow(rows)), height = 900, res = 150)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1L, nrow(rows)), mar = c(2.5, 2.5, 4, 0.8), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(rows))) {
    layout <- as.matrix(readRDS(rows$layout_path[[i]]))[keep, 1:2, drop = FALSE]
    plot(
      layout[order, 1L], layout[order, 2L],
      col = cols[order],
      pch = 16,
      cex = if (length(keep) > 50000L) 0.15 else 0.28,
      xlab = "",
      ylab = "",
      axes = FALSE,
      main = sprintf(
        "%s\nfull %.2fs, post %.2fs\ntrust %.4f, label %.3f",
        rows$run_label[[i]],
        rows$total_time_sec[[i]],
        rows$post_preprocess_total_sec[[i]],
        rows$trustworthiness[[i]],
        rows$label_knn_accuracy[[i]]
      )
    )
    box(col = "grey85")
  }
  title(paste0(dataset$name, " end-to-end UMAP and landmarking"), outer = TRUE, line = 0.2, cex.main = 1.1)
  path
}

plot_speed_quality <- function(results, plot_dir) {
  ok <- results$status == "success" & is.finite(results$total_time_sec)
  results <- results[ok, , drop = FALSE]
  if (nrow(results) == 0L) return(NA_character_)
  path <- file.path(plot_dir, "end_to_end_speed_quality_summary.png")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 1700, height = 950, res = 150)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1L, 2L), mar = c(9, 4, 3, 1))
  labels <- paste(results$dataset, results$run_label, sep = "\n")
  col <- ifelse(grepl("uwot", results$run_label), "#d95f02",
                ifelse(grepl("landmark50", results$run_label), "#1b9e77",
                       ifelse(grepl("metal", results$run_label), "#2c7fb8", "#7570b3")))
  barplot(
    results$total_time_sec,
    names.arg = labels,
    las = 2,
    col = col,
    ylab = "Total time (sec, preprocessing included)",
    main = "End-to-end runtime"
  )
  plot(
    results$total_time_sec,
    results$trustworthiness,
    pch = 19,
    col = col,
    xlab = "Total time (sec, preprocessing included)",
    ylab = "Trustworthiness / kNN@15",
    main = "Speed vs local quality"
  )
  text(
    results$total_time_sec,
    results$trustworthiness,
    labels = paste(results$dataset, results$run_label),
    pos = 4,
    cex = 0.55
  )
  path
}

source_benchmark_helpers()

datasets <- parse_csv("datasets", "fashion_mnist,mnist")
backends <- parse_csv("backends", "cpu,metal")
landmark_modes <- parse_csv("landmark-modes", "projection,refined_selective")
seed <- int_arg("seed", 6L)
k <- int_arg("k", 50L)
landmark_fraction <- num_arg("landmark-fraction", 0.5)
min_n <- int_arg("min-n", 50000L)
max_n <- int_arg("max-n", 70000L)
pca_dims <- int_arg("pca-dims", 50L)
quality_sample_size <- int_arg("quality-sample-size", 3000L)
global_sample_size <- int_arg("global-sample-size", 1500L)
max_plot_points <- int_arg("max-plot-points", 70000L)
preprocess_backend <- parse_scalar("preprocess-backend", "cpu")
results_root <- parse_scalar("results-root", "/Users/stefano/Documents/fastEmbedR-results")
results_dir <- parse_scalar(
  "results-dir",
  file.path(results_root, paste0("end_to_end_own_nn_landmarks_seed", seed))
)
cache_dir <- parse_scalar("cache-dir", file.path(results_root, "cache"))
timeout_sec <- as.numeric(parse_scalar("timeout-sec", "0"))

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
layout_dir <- file.path(results_dir, "layouts")
plot_dir <- file.path(results_dir, "plots")

all_rows <- list()
plot_paths <- character()

for (dataset_id in datasets) {
  message("Loading ", dataset_id)
  dataset <- load_named_dataset(dataset_id, cache_dir = cache_dir, min_n = min_n, max_n = max_n, seed = seed)
  if (is.null(dataset)) next
  message("Preparing ", dataset$name, " (n=", nrow(dataset$x), ", p=", ncol(dataset$x), ")")
  dataset <- prepare_dataset(dataset, pca_dims = pca_dims, seed = seed, preprocess_backend = preprocess_backend)
  q <- stratified_sample(dataset$labels, nrow(dataset$x), quality_sample_size, seed + 100L)
  cfg <- umap_config_for(nrow(dataset$x), k, "cpu")

  dataset_rows <- list()
  for (backend in backends) {
    dataset_rows[[length(dataset_rows) + 1L]] <- run_fastembedr_full(
      dataset, backend, k, seed, q, cfg, timeout_sec, global_sample_size, layout_dir
    )
  }
  dataset_rows[[length(dataset_rows) + 1L]] <- run_uwot_own_nn(
    dataset, k, seed, q, cfg, timeout_sec, global_sample_size, layout_dir
  )
  if ("projection" %in% landmark_modes) {
    for (backend in backends) {
      dataset_rows[[length(dataset_rows) + 1L]] <- run_landmark_projection(
        dataset, backend, k, landmark_fraction, seed, q, timeout_sec, global_sample_size, layout_dir
      )
    }
  }
  if ("refined_selective" %in% landmark_modes) {
    for (backend in backends) {
      dataset_rows[[length(dataset_rows) + 1L]] <- run_landmark_refined_selective(
        dataset, backend, k, landmark_fraction, seed, q, timeout_sec, global_sample_size, layout_dir
      )
    }
  }

  rows <- bind_rows_base(dataset_rows)
  if (nrow(rows) > 0L) {
    full_cpu <- rows$status == "success" & rows$run_label == "full cpu"
    if (any(full_cpu)) {
      ref <- rows[which(full_cpu)[[1L]], , drop = FALSE]
      ok <- rows$status == "success"
      rows$total_speedup_vs_full_cpu[ok] <- ref$total_time_sec[[1L]] / rows$total_time_sec[ok]
      rows$post_preprocess_speedup_vs_full_cpu[ok] <- ref$post_preprocess_total_sec[[1L]] / rows$post_preprocess_total_sec[ok]
      rows$trust_delta_vs_full_cpu[ok] <- rows$trustworthiness[ok] - ref$trustworthiness[[1L]]
      rows$label_accuracy_delta_vs_full_cpu[ok] <- rows$label_knn_accuracy[ok] - ref$label_knn_accuracy[[1L]]
    }
    plot_paths <- c(plot_paths, plot_dataset_layouts(dataset, rows, plot_dir, max_plot_points, seed))
  }
  all_rows[[length(all_rows) + 1L]] <- rows
  write.csv(bind_rows_base(all_rows), file.path(results_dir, "end_to_end_results_latest.csv"), row.names = FALSE)
}

results <- bind_rows_base(all_rows)
if (nrow(results) > 0L) {
  summary_path <- file.path(results_dir, "end_to_end_results_summary.csv")
  write.csv(results, summary_path, row.names = FALSE)
  summary_plot <- plot_speed_quality(results, plot_dir)
  writeLines(c(capture.output(fastEmbedR::backend_info()), ""), file.path(results_dir, "backend_info.txt"))
  message("\nSaved results:")
  message("  ", normalizePath(summary_path, winslash = "/", mustWork = FALSE))
  message("  ", normalizePath(summary_plot, winslash = "/", mustWork = FALSE))
  for (path in plot_paths[!is.na(plot_paths)]) {
    message("  ", normalizePath(path, winslash = "/", mustWork = FALSE))
  }
  cols <- intersect(c(
    "dataset", "run_label", "status", "backend_used", "knn_backend",
    "preprocess_time_sec", "knn_time_sec", "embedding_time_sec",
    "projection_time_sec", "refinement_knn_time_sec", "refinement_time_sec",
    "post_preprocess_total_sec", "total_time_sec", "total_speedup_vs_full_cpu",
    "trustworthiness", "knn_preservation_50", "label_knn_accuracy",
    "trust_delta_vs_full_cpu", "label_accuracy_delta_vs_full_cpu"
  ), names(results))
  print(results[, cols, drop = FALSE])
}
