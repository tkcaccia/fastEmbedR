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

source_benchmark_helpers()

bench_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) default else sub(prefix, "", value[[1L]], fixed = TRUE)
}

bench_csv <- function(name, default) {
  value <- bench_scalar(name, default)
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

bench_int <- function(name, default) as.integer(as.numeric(bench_scalar(name, as.character(default))))
bench_num <- function(name, default) as.numeric(bench_scalar(name, as.character(default)))

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_.-]+", "_", x)
}

bind_rows_base <- function(rows) {
  rows <- rows[vapply(rows, function(x) is.data.frame(x) && nrow(x) > 0L, logical(1))]
  if (length(rows) == 0L) return(data.frame())
  cols <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(df) {
    missing <- setdiff(cols, names(df))
    for (name in missing) df[[name]] <- NA
    df[, cols, drop = FALSE]
  })
  do.call(rbind, rows)
}

label_colors <- function(labels, alpha = 0.45) {
  if (is.null(labels)) {
    return(rep(grDevices::adjustcolor("black", alpha.f = alpha), 1L))
  }
  labels <- factor(labels)
  n <- nlevels(labels)
  base <- if (n <= 8L) grDevices::hcl.colors(n, "Dark 3") else grDevices::hcl.colors(n, "Spectral")
  stats::setNames(grDevices::adjustcolor(base, alpha.f = alpha), levels(labels))[as.character(labels)]
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

backend_matches_request <- function(requested, used) {
  if (is.null(used) || length(used) == 0L || is.na(used[[1L]])) return(FALSE)
  used <- as.character(used[[1L]])
  if (requested %in% c("cuda", "metal")) {
    identical(used, requested) || startsWith(used, paste0(requested, "+"))
  } else {
    identical(used, requested)
  }
}

cache_preprocessed_dataset <- function(dataset, pca_dims, seed, backend, cache_dir, force_recompute = FALSE) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  cache_path <- file.path(
    cache_dir,
    paste0(safe_name(dataset$name), "_pca", as.integer(pca_dims), "_seed", as.integer(seed), ".rds")
  )
  if (!isTRUE(force_recompute) && file.exists(cache_path)) {
    out <- readRDS(cache_path)
    out$preprocess_cache_hit <- TRUE
    return(out)
  }
  out <- prepare_dataset(dataset, pca_dims = pca_dims, seed = seed, preprocess_backend = backend)
  out$preprocess_cache_hit <- FALSE
  saveRDS(out, cache_path, version = 2)
  out
}

drop_self_with_attrs <- function(raw, k) {
  out <- drop_self_knn(raw, k)
  attr(out, "backend") <- attr(raw, "backend")
  attr(out, "exact") <- attr(raw, "exact")
  attr(out, "recall") <- attr(raw, "recall")
  attr(out, "approximation") <- attr(raw, "approximation")
  out
}

recall_on_subset <- function(x, knn, k, sample_size, seed) {
  if (isTRUE(attr(knn, "exact"))) {
    return(data.frame(
      knn_recall_at_k = 1,
      knn_median_recall_at_k = 1,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = 0,
      stringsAsFactors = FALSE
    ))
  }

  attached <- attr(knn, "recall")
  if (is.data.frame(attached) && nrow(attached) > 0L && is.finite(attached$recall_at_k[[1L]])) {
    return(data.frame(
      knn_recall_at_k = attached$recall_at_k[[1L]],
      knn_median_recall_at_k = attached$median_recall_at_k[[1L]],
      knn_recall_sample_n = attached$sample_size[[1L]],
      knn_recall_time_sec = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  n <- nrow(x)
  sample_size <- min(as.integer(sample_size), n)
  if (sample_size <= 0L) {
    return(data.frame(
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = 0L,
      knn_recall_time_sec = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  set.seed(as.integer(seed) + 1009L)
  rows <- sort(sample.int(n, sample_size))
  measured <- timed_value({
    exact_raw <- fastEmbedR::nn(x, x[rows, , drop = FALSE], k = min(n, k + 1L), backend = "cpu")
    exact_idx <- matrix(NA_integer_, sample_size, k)
    for (i in seq_along(rows)) {
      keep <- exact_raw$indices[i, ] != rows[[i]]
      idx <- exact_raw$indices[i, keep]
      if (length(idx) < k) idx <- exact_raw$indices[i, seq_len(ncol(exact_raw$indices))]
      exact_idx[i, ] <- idx[seq_len(k)]
    }
    approx_idx <- knn$indices[rows, seq_len(k), drop = FALSE]
    fastEmbedR:::knn_recall(list(indices = approx_idx), list(indices = exact_idx), k = k)
  })

  data.frame(
    knn_recall_at_k = measured$value$recall_at_k[[1L]],
    knn_median_recall_at_k = measured$value$median_recall_at_k[[1L]],
    knn_recall_sample_n = sample_size,
    knn_recall_time_sec = measured$elapsed,
    stringsAsFactors = FALSE
  )
}

score_layout <- function(dataset, layout, q, seed, method, backend, global_sample_size) {
  fastEmbedR::evaluate_embedding(
    dataset$x[q, , drop = FALSE],
    layout[q, , drop = FALSE],
    labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
    k = c(15L, 30L, 50L),
    sample_size_for_global_metrics = min(global_sample_size, length(q)),
    sample_size_for_local_metrics = length(q),
    seed = seed,
    method = method,
    backend = backend,
    dataset = dataset$name
  )
}

umap_reference_config <- function(indices, distances) {
  cfg <- fastEmbedR:::fast_knn_umap_config(nrow(indices), ncol(indices), "cpu")
  cfg <- fastEmbedR:::apply_umap_connectivity_spectral_rule(
    cfg,
    indices,
    col_start = 1L,
    n_neighbors = ncol(indices)
  )
  cfg <- fastEmbedR:::apply_fast_knn_umap_distance_profile_rule(cfg, distances)
  cfg
}

run_fastembedr_full <- function(dataset,
                                spec,
                                k,
                                seed,
                                q,
                                global_sample_size,
                                recall_sample_size,
                                layout_dir,
                                timeout_sec) {
  tryCatch({
    nn_measured <- timed_value({
      raw <- fastEmbedR::nn(dataset$x, dataset$x, k = k + 1L, backend = spec$knn_backend)
      list(raw = raw, knn = drop_self_with_attrs(raw, k))
    }, timeout_sec)
    knn <- nn_measured$value$knn
    knn_backend_used <- attr(nn_measured$value$raw, "backend")
    if (is.null(knn_backend_used)) knn_backend_used <- spec$knn_backend

    recall <- recall_on_subset(dataset$x, knn, k, recall_sample_size, seed)
    umap_measured <- measure_layout(
      fastEmbedR::embed_knn(knn, method = "umap", seed = seed, backend = spec$umap_backend),
      nrow(dataset$x),
      timeout_sec
    )
    cfg <- attr(umap_measured$layout, "fastEmbedR_config")
    umap_backend_used <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else spec$umap_backend
    if (spec$umap_backend %in% c("cuda", "metal") && !backend_matches_request(spec$umap_backend, umap_backend_used)) {
      stop("Requested UMAP backend ", spec$umap_backend, " but backend_used was ", umap_backend_used, call. = FALSE)
    }

    metrics <- score_layout(dataset, umap_measured$layout, q, seed, spec$id, umap_backend_used, global_sample_size)
    layout_path <- file.path(layout_dir, paste0(safe_name(paste(dataset$name, spec$id, seed, sep = "_")), ".rds"))
    dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(umap_measured$layout, layout_path, version = 2)
    data.frame(
      dataset = dataset$name,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = spec$id,
      package = "fastEmbedR",
      mode = "full",
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = spec$knn_backend,
      knn_backend_used = knn_backend_used,
      umap_backend_requested = spec$umap_backend,
      umap_backend_used = umap_backend_used,
      nn_sec = nn_measured$elapsed,
      umap_sec = umap_measured$elapsed,
      projection_sec = 0,
      landmark_selection_sec = 0,
      landmark_nn_sec = 0,
      landmark_umap_sec = 0,
      nn_plus_umap_sec = nn_measured$elapsed + umap_measured$elapsed,
      total_timed_sec = nn_measured$elapsed + umap_measured$elapsed,
      rss_delta_mb = nn_measured$rss_delta_mb + umap_measured$rss_delta_mb,
      knn_recall_at_k = recall$knn_recall_at_k,
      knn_median_recall_at_k = recall$knn_median_recall_at_k,
      knn_recall_sample_n = recall$knn_recall_sample_n,
      knn_recall_time_sec = recall$knn_recall_time_sec,
      trustworthiness = metrics$trustworthiness,
      continuity = metrics$continuity,
      knn_preservation_15 = metrics$knn_preservation_15,
      knn_preservation_30 = metrics$knn_preservation_30,
      knn_preservation_50 = metrics$knn_preservation_50,
      distance_spearman = metrics$distance_spearman,
      distance_pearson = metrics$distance_pearson,
      stress = metrics$stress,
      silhouette = metrics$silhouette,
      label_knn_accuracy = metrics$label_knn_accuracy,
      rare_class_recall = metrics$rare_class_recall,
      layout_path = layout_path,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset$name,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = spec$id,
      package = "fastEmbedR",
      mode = "full",
      status = "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = spec$knn_backend,
      knn_backend_used = NA_character_,
      umap_backend_requested = spec$umap_backend,
      umap_backend_used = NA_character_,
      nn_sec = NA_real_,
      umap_sec = NA_real_,
      projection_sec = NA_real_,
      landmark_selection_sec = NA_real_,
      landmark_nn_sec = NA_real_,
      landmark_umap_sec = NA_real_,
      nn_plus_umap_sec = NA_real_,
      total_timed_sec = NA_real_,
      rss_delta_mb = NA_real_,
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = NA_real_,
      trustworthiness = NA_real_,
      continuity = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_,
      distance_spearman = NA_real_,
      distance_pearson = NA_real_,
      stress = NA_real_,
      silhouette = NA_real_,
      label_knn_accuracy = NA_real_,
      rare_class_recall = NA_real_,
      layout_path = NA_character_,
      stringsAsFactors = FALSE
    )
  })
}

run_uwot_split <- function(dataset,
                           k,
                           seed,
                           q,
                           global_sample_size,
                           layout_dir,
                           timeout_sec) {
  tryCatch({
    if (!requireNamespace("uwot", quietly = TRUE)) {
      stop("Package `uwot` is not installed.", call. = FALSE)
    }
    cfg <- fastEmbedR:::fast_knn_umap_config(nrow(dataset$x), k, "cpu")
    nn_measured <- timed_value({
      out <- uwot::umap(
        X = dataset$x,
        n_neighbors = as.integer(k),
        n_components = 2L,
        metric = "euclidean",
        nn_method = "annoy",
        n_epochs = 0L,
        init = "spectral",
        min_dist = cfg$min_dist,
        learning_rate = 1,
        repulsion_strength = cfg$repulsion_strength,
        negative_sample_rate = as.integer(cfg$negative_sample_rate),
        fast_sgd = TRUE,
        n_threads = as.integer(cfg$n_threads),
        n_sgd_threads = as.integer(cfg$n_threads),
        ret_nn = TRUE,
        ret_model = FALSE,
        verbose = FALSE,
        seed = as.integer(seed)
      )
      out$nn$euclidean
    }, timeout_sec)
    knn <- list(
      indices = as.matrix(nn_measured$value$idx),
      distances = as.matrix(nn_measured$value$dist)
    )
    cfg <- umap_reference_config(knn$indices, knn$distances)
    umap_measured <- measure_layout(
      uwot::umap(
        X = dataset$x,
        n_neighbors = as.integer(k),
        n_components = 2L,
        metric = "euclidean",
        nn_method = list(idx = knn$indices, dist = knn$distances),
        n_epochs = as.integer(cfg$n_epochs),
        init = "spectral",
        min_dist = cfg$min_dist,
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
    metrics <- score_layout(dataset, umap_measured$layout, q, seed, "uwot_annoy_fast_sgd", "uwot_fast_sgd", global_sample_size)
    layout_path <- file.path(layout_dir, paste0(safe_name(paste(dataset$name, "uwot_annoy_fast_sgd", seed, sep = "_")), ".rds"))
    dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(umap_measured$layout, layout_path, version = 2)
    data.frame(
      dataset = dataset$name,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = "uwot_annoy_fast_sgd",
      package = "uwot",
      mode = "full",
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = "uwot_annoy",
      knn_backend_used = "uwot_annoy",
      umap_backend_requested = "uwot_fast_sgd",
      umap_backend_used = "uwot_fast_sgd",
      nn_sec = nn_measured$elapsed,
      umap_sec = umap_measured$elapsed,
      projection_sec = 0,
      landmark_selection_sec = 0,
      landmark_nn_sec = 0,
      landmark_umap_sec = 0,
      nn_plus_umap_sec = nn_measured$elapsed + umap_measured$elapsed,
      total_timed_sec = nn_measured$elapsed + umap_measured$elapsed,
      rss_delta_mb = nn_measured$rss_delta_mb + umap_measured$rss_delta_mb,
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = NA_real_,
      trustworthiness = metrics$trustworthiness,
      continuity = metrics$continuity,
      knn_preservation_15 = metrics$knn_preservation_15,
      knn_preservation_30 = metrics$knn_preservation_30,
      knn_preservation_50 = metrics$knn_preservation_50,
      distance_spearman = metrics$distance_spearman,
      distance_pearson = metrics$distance_pearson,
      stress = metrics$stress,
      silhouette = metrics$silhouette,
      label_knn_accuracy = metrics$label_knn_accuracy,
      rare_class_recall = metrics$rare_class_recall,
      layout_path = layout_path,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset$name,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = "uwot_annoy_fast_sgd",
      package = "uwot",
      mode = "full",
      status = "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = "uwot_annoy",
      knn_backend_used = NA_character_,
      umap_backend_requested = "uwot_fast_sgd",
      umap_backend_used = NA_character_,
      nn_sec = NA_real_,
      umap_sec = NA_real_,
      projection_sec = NA_real_,
      landmark_selection_sec = NA_real_,
      landmark_nn_sec = NA_real_,
      landmark_umap_sec = NA_real_,
      nn_plus_umap_sec = NA_real_,
      total_timed_sec = NA_real_,
      rss_delta_mb = NA_real_,
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = NA_real_,
      trustworthiness = NA_real_,
      continuity = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_,
      distance_spearman = NA_real_,
      distance_pearson = NA_real_,
      stress = NA_real_,
      silhouette = NA_real_,
      label_knn_accuracy = NA_real_,
      rare_class_recall = NA_real_,
      layout_path = NA_character_,
      stringsAsFactors = FALSE
    )
  })
}

run_landmark50 <- function(dataset,
                           spec,
                           k,
                           seed,
                           q,
                           global_sample_size,
                           layout_dir,
                           timeout_sec) {
  tryCatch({
    n <- nrow(dataset$x)
    n_landmarks <- min(max(2L, as.integer(ceiling(0.5 * n))), n - 1L)
    selection <- timed_value({
      fastEmbedR:::select_landmark_rows(dataset$x, n_landmarks, seed)
    }, timeout_sec)
    landmark_idx <- selection$value
    x_land <- dataset$x[landmark_idx, , drop = FALSE]
    landmark_k <- min(as.integer(k), nrow(x_land) - 1L)
    projection_k <- fastEmbedR:::landmark_projection_k(nrow(x_land), landmark_k)

    landmark_nn <- timed_value({
      raw <- fastEmbedR:::nn_without_self(x_land, k = landmark_k, backend = spec$knn_backend)
      out <- fastEmbedR:::normalize_supplied_knn(raw, nrow(x_land), landmark_k)
      attr(out, "backend") <- attr(raw, "backend")
      out
    }, timeout_sec)
    landmark_nn_backend <- attr(landmark_nn$value, "backend")
    if (is.null(landmark_nn_backend)) landmark_nn_backend <- spec$knn_backend

    landmark_umap <- measure_layout(
      fastEmbedR::embed_knn(landmark_nn$value, method = "umap", seed = seed, backend = spec$umap_backend),
      nrow(x_land),
      timeout_sec
    )
    cfg <- attr(landmark_umap$layout, "fastEmbedR_config")
    umap_backend_used <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else spec$umap_backend
    if (spec$umap_backend %in% c("cuda", "metal") && !backend_matches_request(spec$umap_backend, umap_backend_used)) {
      stop("Requested landmark UMAP backend ", spec$umap_backend, " but backend_used was ", umap_backend_used, call. = FALSE)
    }

    projection <- timed_value({
      if (identical(spec$projection, "fused_metal")) {
        fastEmbedR:::fused_landmark_project_layout(
          x_land,
          dataset$x,
          landmark_umap$layout,
          landmark_idx,
          projection_k,
          backend = "metal"
        )
      } else if (identical(spec$projection, "approx_cpu")) {
        projection_knn <- fastEmbedR:::approx_landmark_projection_knn(
          x_land,
          dataset$x,
          projection_k,
          seed
        )
        fastEmbedR:::interpolate_landmark_layout(
          landmark_umap$layout,
          landmark_idx,
          projection_knn,
          n,
          backend = spec$umap_backend
        )
      } else {
        projection_knn <- fastEmbedR::nn(x_land, dataset$x, k = projection_k, backend = spec$projection_backend)
        fastEmbedR:::interpolate_landmark_layout(
          landmark_umap$layout,
          landmark_idx,
          projection_knn,
          n,
          backend = spec$umap_backend
        )
      }
    }, timeout_sec)
    layout <- as.matrix(projection$value)[, 1:2, drop = FALSE]
    metrics <- score_layout(dataset, layout, q, seed, spec$id, umap_backend_used, global_sample_size)
    layout_path <- file.path(layout_dir, paste0(safe_name(paste(dataset$name, spec$id, seed, sep = "_")), ".rds"))
    dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(layout, layout_path, version = 2)
    data.frame(
      dataset = dataset$name,
      n = n,
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = spec$id,
      package = "fastEmbedR",
      mode = "landmark50_project",
      status = "success",
      error_message = NA_character_,
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = spec$knn_backend,
      knn_backend_used = landmark_nn_backend,
      umap_backend_requested = spec$umap_backend,
      umap_backend_used = umap_backend_used,
      nn_sec = landmark_nn$elapsed,
      umap_sec = landmark_umap$elapsed,
      projection_sec = projection$elapsed,
      landmark_selection_sec = selection$elapsed,
      landmark_nn_sec = landmark_nn$elapsed,
      landmark_umap_sec = landmark_umap$elapsed,
      nn_plus_umap_sec = landmark_nn$elapsed + landmark_umap$elapsed,
      total_timed_sec = selection$elapsed + landmark_nn$elapsed + landmark_umap$elapsed + projection$elapsed,
      rss_delta_mb = selection$rss_delta_mb + landmark_nn$rss_delta_mb + landmark_umap$rss_delta_mb + projection$rss_delta_mb,
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = NA_real_,
      trustworthiness = metrics$trustworthiness,
      continuity = metrics$continuity,
      knn_preservation_15 = metrics$knn_preservation_15,
      knn_preservation_30 = metrics$knn_preservation_30,
      knn_preservation_50 = metrics$knn_preservation_50,
      distance_spearman = metrics$distance_spearman,
      distance_pearson = metrics$distance_pearson,
      stress = metrics$stress,
      silhouette = metrics$silhouette,
      label_knn_accuracy = metrics$label_knn_accuracy,
      rare_class_recall = metrics$rare_class_recall,
      layout_path = layout_path,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset$name,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      raw_p = dataset$raw_p,
      method = spec$id,
      package = "fastEmbedR",
      mode = "landmark50_project",
      status = "failed",
      error_message = conditionMessage(e),
      seed = seed,
      k = k,
      preprocess_sec_once = dataset$preprocess_time_sec,
      preprocess_excluded_from_total = TRUE,
      knn_backend_requested = spec$knn_backend,
      knn_backend_used = NA_character_,
      umap_backend_requested = spec$umap_backend,
      umap_backend_used = NA_character_,
      nn_sec = NA_real_,
      umap_sec = NA_real_,
      projection_sec = NA_real_,
      landmark_selection_sec = NA_real_,
      landmark_nn_sec = NA_real_,
      landmark_umap_sec = NA_real_,
      nn_plus_umap_sec = NA_real_,
      total_timed_sec = NA_real_,
      rss_delta_mb = NA_real_,
      knn_recall_at_k = NA_real_,
      knn_median_recall_at_k = NA_real_,
      knn_recall_sample_n = NA_integer_,
      knn_recall_time_sec = NA_real_,
      trustworthiness = NA_real_,
      continuity = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_,
      distance_spearman = NA_real_,
      distance_pearson = NA_real_,
      stress = NA_real_,
      silhouette = NA_real_,
      label_knn_accuracy = NA_real_,
      rare_class_recall = NA_real_,
      layout_path = NA_character_,
      stringsAsFactors = FALSE
    )
  })
}

plot_dataset_layouts <- function(dataset, rows, plot_dir, max_points, seed) {
  ok <- rows$status == "success" & file.exists(rows$layout_path)
  rows <- rows[ok, , drop = FALSE]
  if (nrow(rows) == 0L) return(NA_character_)
  n <- nrow(dataset$x)
  set.seed(seed)
  keep <- if (is.finite(max_points) && max_points > 0L && n > max_points) sort(sample.int(n, max_points)) else seq_len(n)
  cols <- label_colors(if (is.null(dataset$labels)) NULL else dataset$labels[keep], alpha = if (length(keep) > 50000L) 0.25 else 0.5)
  draw_order <- sample.int(length(keep))
  n_cols <- min(4L, nrow(rows))
  n_rows <- ceiling(nrow(rows) / n_cols)
  path <- file.path(plot_dir, paste0(safe_name(dataset$name), "_layouts.png"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 520 * n_cols, height = 430 * n_rows, res = 120)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(n_rows, n_cols), mar = c(1.4, 1.4, 3.2, 0.7), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(rows))) {
    layout <- as.matrix(readRDS(rows$layout_path[[i]]))[keep, 1:2, drop = FALSE]
    title <- sprintf(
      "%s\nNN %.2fs | UMAP %.2fs | trust %.3f",
      rows$method[[i]],
      rows$nn_sec[[i]],
      rows$umap_sec[[i]],
      rows$trustworthiness[[i]]
    )
    plot(
      layout[draw_order, 1L],
      layout[draw_order, 2L],
      col = cols[draw_order],
      pch = 16,
      cex = if (length(keep) > 50000L) 0.13 else 0.28,
      axes = FALSE,
      xlab = "",
      ylab = "",
      main = title
    )
    box(col = "grey85")
  }
  if (nrow(rows) < n_rows * n_cols) {
    for (i in seq_len(n_rows * n_cols - nrow(rows))) plot.new()
  }
  title(dataset$name, outer = TRUE, line = 0.1, cex.main = 1.1)
  path
}

plot_summary <- function(results, plot_dir) {
  ok <- results$status == "success"
  results <- results[ok, , drop = FALSE]
  if (nrow(results) == 0L) return(NA_character_)
  path <- file.path(plot_dir, "nn_umap_speed_quality_summary.png")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 1600, height = 950, res = 140)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1, 2), mar = c(9, 5, 3, 1))
  labels <- paste(results$dataset, results$method, sep = "\n")
  barplot(
    rbind(results$nn_sec, results$umap_sec, results$projection_sec),
    names.arg = labels,
    las = 2,
    col = c("#4c78a8", "#59a14f", "#f28e2b"),
    ylab = "Seconds; preprocessing excluded",
    main = "Timed stages",
    legend.text = c("NN", "UMAP", "projection")
  )
  plot(
    results$total_timed_sec,
    results$trustworthiness,
    pch = 19,
    col = ifelse(results$package == "uwot", "#d95f02", ifelse(grepl("metal", results$umap_backend_used), "#2c7fb8", "#4c4c4c")),
    xlab = "Timed total seconds, preprocessing excluded",
    ylab = "Trustworthiness",
    main = "Speed vs quality"
  )
  text(
    results$total_timed_sec,
    results$trustworthiness,
    labels = paste(results$dataset, results$method, sep = " "),
    pos = 4,
    cex = 0.58
  )
  path
}

method_specs <- function(include_landmarks) {
  full <- list(
    list(id = "fastEmbedR_auto_cpu_umap", knn_backend = "auto", umap_backend = "cpu"),
    list(id = "fastEmbedR_auto_metal_umap", knn_backend = "auto", umap_backend = "metal"),
    list(id = "fastEmbedR_cpu_exact_cpu_umap", knn_backend = "cpu", umap_backend = "cpu"),
    list(id = "fastEmbedR_cpu_approx_cpu_umap", knn_backend = "cpu_approx", umap_backend = "cpu"),
    list(id = "fastEmbedR_nndescent_cpu_umap", knn_backend = "cpu_nndescent", umap_backend = "cpu"),
    list(id = "fastEmbedR_clustered_cpu_umap", knn_backend = "cpu_clustered", umap_backend = "cpu"),
    list(id = "fastEmbedR_metal_exact_metal_umap", knn_backend = "metal", umap_backend = "metal"),
    list(id = "fastEmbedR_metal_approx_metal_umap", knn_backend = "metal_approx", umap_backend = "metal")
  )
  if (!isTRUE(include_landmarks)) return(full)
  c(full, list(
    list(id = "fastEmbedR_landmark50_cpuapprox_cpu", knn_backend = "cpu_approx", umap_backend = "cpu", projection = "approx_cpu", projection_backend = "cpu"),
    list(id = "fastEmbedR_landmark50_metal_fused", knn_backend = "metal", umap_backend = "metal", projection = "fused_metal", projection_backend = "metal"),
    list(id = "fastEmbedR_landmark50_metalapprox_fused", knn_backend = "metal_approx", umap_backend = "metal", projection = "fused_metal", projection_backend = "metal")
  ))
}

datasets_arg <- bench_csv("datasets", "fashion_mnist,mnist")
seed <- bench_int("seed", 6L)
k <- bench_int("k", 50L)
pca_dims <- bench_int("pca-dims", 50L)
min_n <- bench_int("min-n", 50000L)
max_n <- bench_num("max-n", 70000)
quality_sample_size <- bench_int("quality-sample-size", 3000L)
global_sample_size <- bench_int("global-sample-size", 1500L)
recall_sample_size <- bench_int("recall-sample-size", 128L)
max_plot_points <- bench_int("max-plot-points", 70000L)
timeout_sec <- bench_num("timeout-sec", 0)
results_dir <- bench_scalar("results-dir", file.path(path.expand("~/Documents/fastEmbedR-results"), paste0("nn_umap_preprocessed_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
data_cache_dir <- bench_scalar("cache-dir", file.path(path.expand("~/Documents/fastEmbedR-results"), "cache"))
preprocess_cache_dir <- bench_scalar("preprocess-cache-dir", file.path(results_dir, "preprocessed"))
preprocess_backend <- bench_scalar("preprocess-backend", "cpu")
include_landmarks <- tolower(bench_scalar("include-landmarks", "true")) %in% c("1", "true", "yes")
force_preprocess <- tolower(bench_scalar("force-preprocess", "false")) %in% c("1", "true", "yes")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
layout_dir <- file.path(results_dir, "layouts")
plot_dir <- file.path(results_dir, "plots")

all_rows <- list()
plot_paths <- character()

for (dataset_id in datasets_arg) {
  dataset <- load_named_dataset(dataset_id, cache_dir = data_cache_dir, min_n = min_n, max_n = max_n, seed = seed)
  if (is.null(dataset)) next
  message("Preprocessing once for ", dataset$name, " (n=", nrow(dataset$x), ", p=", ncol(dataset$x), ")")
  dataset <- cache_preprocessed_dataset(
    dataset,
    pca_dims = pca_dims,
    seed = seed,
    backend = preprocess_backend,
    cache_dir = preprocess_cache_dir,
    force_recompute = force_preprocess
  )
  q <- stratified_sample(dataset$labels, nrow(dataset$x), quality_sample_size, seed + 100L)

  dataset_rows <- list()
  specs <- method_specs(include_landmarks = include_landmarks)
  for (spec in specs) {
    message("  ", spec$id, " on preprocessed data")
    row <- if (startsWith(spec$id, "fastEmbedR_landmark50")) {
      run_landmark50(dataset, spec, k, seed, q, global_sample_size, layout_dir, timeout_sec)
    } else {
      run_fastembedr_full(dataset, spec, k, seed, q, global_sample_size, recall_sample_size, layout_dir, timeout_sec)
    }
    dataset_rows[[length(dataset_rows) + 1L]] <- row
    utils::write.csv(bind_rows_base(c(all_rows, dataset_rows)), file.path(results_dir, "latest_partial_results.csv"), row.names = FALSE)
  }

  message("  uwot_annoy_fast_sgd on preprocessed data")
  dataset_rows[[length(dataset_rows) + 1L]] <- run_uwot_split(
    dataset,
    k = k,
    seed = seed,
    q = q,
    global_sample_size = global_sample_size,
    layout_dir = layout_dir,
    timeout_sec = timeout_sec
  )
  dataset_df <- bind_rows_base(dataset_rows)
  all_rows[[length(all_rows) + 1L]] <- dataset_df
  plot_paths[[length(plot_paths) + 1L]] <- plot_dataset_layouts(dataset, dataset_df, plot_dir, max_plot_points, seed)
  invisible(gc())
}

results <- bind_rows_base(all_rows)
results <- results[order(results$dataset, results$mode, results$total_timed_sec), , drop = FALSE]
summary <- results[results$status == "success", , drop = FALSE]
summary <- summary[, intersect(c(
  "dataset", "method", "mode", "n", "p", "k", "knn_backend_used", "umap_backend_used",
  "preprocess_sec_once", "preprocess_excluded_from_total", "nn_sec", "umap_sec",
  "projection_sec", "total_timed_sec", "knn_recall_at_k", "trustworthiness",
  "knn_preservation_15", "label_knn_accuracy", "silhouette", "layout_path"
), names(summary)), drop = FALSE]

results_file <- file.path(results_dir, "nn_umap_preprocessed_results.csv")
summary_file <- file.path(results_dir, "nn_umap_preprocessed_summary.csv")
plot_manifest <- file.path(results_dir, "plots.txt")
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(summary, summary_file, row.names = FALSE)
writeLines(plot_paths[nzchar(plot_paths) & !is.na(plot_paths)], plot_manifest)
writeLines(capture.output(fastEmbedR::backend_info()), file.path(results_dir, "backend_info.txt"))
writeLines(capture.output(sessionInfo()), file.path(results_dir, "session_info.txt"))
summary_plot <- plot_summary(results, plot_dir)

print(summary[, intersect(c(
  "dataset", "method", "mode", "knn_backend_used", "umap_backend_used",
  "nn_sec", "umap_sec", "projection_sec", "total_timed_sec",
  "knn_recall_at_k", "trustworthiness", "knn_preservation_15", "label_knn_accuracy"
), names(summary)), drop = FALSE], row.names = FALSE)

cat("\nPreprocessing/PCA was performed once per dataset and excluded from total_timed_sec.\n")
cat("Saved results: ", normalizePath(results_file, mustWork = FALSE), "\n", sep = "")
cat("Saved summary: ", normalizePath(summary_file, mustWork = FALSE), "\n", sep = "")
cat("Saved summary plot: ", normalizePath(summary_plot, mustWork = FALSE), "\n", sep = "")
cat("Saved layout plots:\n")
for (path in plot_paths[nzchar(plot_paths) & !is.na(plot_paths)]) {
  cat("  ", normalizePath(path, mustWork = FALSE), "\n", sep = "")
}
