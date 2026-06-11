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

label_colors <- function(labels, alpha = 0.45) {
  if (is.null(labels)) {
    return(rep(grDevices::adjustcolor("black", alpha.f = alpha), 1L))
  }
  labels <- factor(labels)
  n <- nlevels(labels)
  base <- if (n <= 8L) {
    grDevices::hcl.colors(n, "Dark 3")
  } else {
    grDevices::hcl.colors(n, "Spectral")
  }
  stats::setNames(grDevices::adjustcolor(base, alpha.f = alpha), levels(labels))[as.character(labels)]
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

backend_matches_request <- function(requested, used) {
  if (is.null(used) || length(used) == 0L || is.na(used[[1L]])) return(FALSE)
  used <- as.character(used[[1L]])
  if (requested %in% c("cuda", "metal")) {
    identical(used, requested) || startsWith(used, paste0(requested, "+"))
  } else {
    identical(used, requested)
  }
}

run_label <- function(row) {
  if (!is.null(row$run_label) && !is.na(row$run_label[[1L]]) && nzchar(row$run_label[[1L]])) {
    return(row$run_label[[1L]])
  }
  if (identical(row$method[[1L]], "uwot_umap_fast_sgd")) {
    return("uwot fast_sgd")
  }
  if (identical(row$method[[1L]], "fastEmbedR_umap")) {
    return(paste("fastEmbedR", row$backend_used[[1L]]))
  }
  paste(row$method[[1L]], row$backend_used[[1L]])
}

read_baseline_rows <- function(dataset_name, results_root, baseline_dirs) {
  rows <- list()
  for (dir in baseline_dirs) {
    path <- file.path(results_root, dir, "backend_results_summary.csv")
    if (!file.exists(path)) next
    df <- tryCatch(utils::read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
    if (is.null(df) || !("dataset" %in% names(df))) next
    df <- df[df$dataset == dataset_name & df$status == "success", , drop = FALSE]
    if (nrow(df) == 0L) next
    df$source_results_dir <- dir
    df$landmark_fraction <- NA_real_
    df$n_landmarks <- NA_integer_
    df$projection_only <- FALSE
    df$selection_time_sec <- NA_real_
    df$landmark_knn_time_sec <- NA_real_
    df$landmark_embedding_time_sec <- NA_real_
    df$projection_knn_time_sec <- NA_real_
    df$interpolation_time_sec <- NA_real_
    df$post_preprocess_total_sec <- df$knn_time_sec + df$embedding_time_sec
    df$run_label <- ifelse(
      df$method == "uwot_umap_fast_sgd",
      "uwot fast_sgd",
      paste0("full ", df$backend_used)
    )
    rows[[length(rows) + 1L]] <- df
  }
  if (length(rows) == 0L) return(data.frame())
  out <- do.call(rbind, rows)
  keep_order <- c("full cpu", "full metal", "uwot fast_sgd")
  out$plot_order <- match(out$run_label, keep_order)
  out$plot_order[is.na(out$plot_order)] <- seq_len(sum(is.na(out$plot_order))) + length(keep_order)
  out[order(out$plot_order), , drop = FALSE]
}

benchmark_landmark_projection <- function(dataset,
                                          backends,
                                          k,
                                          landmark_fraction,
                                          seed,
                                          knn_backend,
                                          projection_mode,
                                          timeout_sec,
                                          quality_sample_size,
                                          global_sample_size,
                                          layout_dir) {
  n <- nrow(dataset$x)
  n_landmarks <- as.integer(ceiling(n * landmark_fraction))
  n_landmarks <- min(max(2L, n_landmarks), n - 1L)
  selection <- timed_value({
    fastEmbedR:::select_landmark_rows(dataset$x, n_landmarks, seed)
  }, timeout_sec)
  landmark_indices <- selection$value
  n_landmarks <- length(landmark_indices)
  selection_method <- attr(landmark_indices, "selection_method")
  if (is.null(selection_method)) selection_method <- "unknown"
  x_landmarks <- dataset$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- min(as.integer(k), n_landmarks - 1L)
  projection_k <- fastEmbedR:::landmark_projection_k(n_landmarks, landmark_neighbors)

  message(
    "  landmark selection: ", n_landmarks, " / ", n,
    " rows (", selection_method, "), projection_k=", projection_k
  )
  landmark_knn <- timed_value({
    raw <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = knn_backend)
    out <- fastEmbedR:::normalize_supplied_knn(raw, n_landmarks, landmark_neighbors)
    attr(out, "backend") <- attr(raw, "backend")
    out
  }, timeout_sec)
  landmark_knn_backend <- attr(landmark_knn$value, "backend")
  if (is.null(landmark_knn_backend)) landmark_knn_backend <- knn_backend

  q <- stratified_sample(dataset$labels, n, quality_sample_size, seed + 100L)
  rows <- list()
  for (backend in backends) {
    message("  landmark projection-only backend=", backend)
    projection_elapsed <- NA_real_
    interpolation_elapsed <- NA_real_
    projection_backend <- NA_character_
    projection_fused <- FALSE
    projection_materialized <- NA
    row <- tryCatch({
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
      cfg <- attr(landmark_layout$layout, "fastEmbedR_config")
      umap_backend <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else backend
      if (backend %in% c("cuda", "metal") && !backend_matches_request(backend, umap_backend)) {
        stop("Requested backend ", backend, " but UMAP backend_used was ", umap_backend, call. = FALSE)
      }

      interpolation <- NULL
      projection_knn <- NULL
      use_approx_projection <- identical(projection_mode, "approx") ||
        (identical(projection_mode, "auto") &&
           fastEmbedR:::use_approx_landmark_projection(knn_backend))
      if (isTRUE(use_approx_projection)) {
        projection_knn <- timed_value({
          fastEmbedR:::approx_landmark_projection_knn(
            x_landmarks,
            dataset$x,
            projection_k,
            seed
          )
        }, timeout_sec)
      }
      if (!is.null(projection_knn) && !is.null(projection_knn$value)) {
        projection_elapsed <- projection_knn$elapsed
        projection_backend <- attr(projection_knn$value, "backend")
        if (is.null(projection_backend)) projection_backend <- "cpu_projection_approx"
        projection_materialized <- TRUE
        interpolation <- timed_value({
          fastEmbedR:::interpolate_landmark_layout(
            landmark_layout$layout,
            landmark_indices,
            projection_knn$value,
            n,
            backend = backend
          )
        }, timeout_sec)
        interpolation_elapsed <- interpolation$elapsed
      } else if (identical(backend, "metal")) {
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
      }
      if (!is.null(interpolation) && !is.null(interpolation$value) && is.na(projection_materialized)) {
        projection_elapsed <- interpolation$elapsed
        interpolation_elapsed <- 0
        projection_backend <- attr(interpolation$value, "projection_backend")
        projection_fused <- TRUE
        projection_materialized <- FALSE
      } else if (is.null(interpolation) || is.null(interpolation$value)) {
        projection_knn <- timed_value({
          fastEmbedR::nn(x_landmarks, dataset$x, k = projection_k, backend = knn_backend)
        }, timeout_sec)
        projection_elapsed <- projection_knn$elapsed
        projection_backend <- attr(projection_knn$value, "backend")
        if (is.null(projection_backend)) projection_backend <- knn_backend
        projection_materialized <- TRUE
        interpolation <- timed_value({
          fastEmbedR:::interpolate_landmark_layout(
            landmark_layout$layout,
            landmark_indices,
            projection_knn$value,
            n,
            backend = backend
          )
        }, timeout_sec)
        interpolation_elapsed <- interpolation$elapsed
      }
      interpolation_layout <- as.matrix(interpolation$value)
      if (nrow(interpolation_layout) != n && ncol(interpolation_layout) == n) {
        interpolation_layout <- t(interpolation_layout)
      }
      if (nrow(interpolation_layout) != n || ncol(interpolation_layout) < 2L) {
        stop("Landmark interpolation did not return an n x 2 layout.", call. = FALSE)
      }
      interpolation_layout <- interpolation_layout[, 1:2, drop = FALSE]
      interpolation_backend <- attr(interpolation$value, "interpolation_backend")
      if (is.null(interpolation_backend)) interpolation_backend <- "cpu"
      backend_used <- paste0(umap_backend, "+interp_", interpolation_backend)

      metrics <- fastEmbedR::evaluate_embedding(
        dataset$x[q, , drop = FALSE],
        interpolation_layout[q, , drop = FALSE],
        labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
        k = c(15L, 30L, 50L),
        sample_size_for_global_metrics = min(global_sample_size, length(q)),
        seed = seed,
        method = "umap_landmark50_projection",
        backend = backend_used,
        dataset = dataset$name
      )

      layout_path <- file.path(
        layout_dir,
        paste0(safe_name(paste(dataset$name, "landmark50_projection", backend_used, seed, sep = "_")), ".rds")
      )
      dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(interpolation_layout, layout_path, version = 2)

      post_preprocess_total <- selection$elapsed +
        landmark_knn$elapsed +
        landmark_layout$elapsed +
        projection_elapsed +
        interpolation_elapsed

      data.frame(
        dataset = dataset$name,
        n = n,
        p = ncol(dataset$x),
        method = "fastEmbedR_umap_landmark50_projection",
        package = "fastEmbedR",
        backend_requested = backend,
        backend_used = backend_used,
        status = "success",
        error_message = NA_character_,
        seed = seed,
        k = k,
        preprocess_time_sec = dataset$preprocess_time_sec,
        knn_time_sec = NA_real_,
        knn_backend = as.character(landmark_knn_backend),
        embedding_time_sec = landmark_layout$elapsed + interpolation_elapsed,
        total_time_sec = dataset$preprocess_time_sec + post_preprocess_total,
        trustworthiness = metrics$trustworthiness[[1L]],
        continuity = metrics$continuity[[1L]],
        knn_preservation_15 = metrics$knn_preservation_15[[1L]],
        knn_preservation_30 = metrics$knn_preservation_30[[1L]],
        knn_preservation_50 = metrics$knn_preservation_50[[1L]],
        label_knn_accuracy = metrics$label_knn_accuracy[[1L]],
        silhouette = metrics$silhouette[[1L]],
        distance_spearman = metrics$distance_spearman[[1L]],
        layout_path = layout_path,
        embedding_speedup_vs_cpu = NA_real_,
        total_speedup_vs_cpu = NA_real_,
        trust_delta_vs_cpu = NA_real_,
        label_accuracy_delta_vs_cpu = NA_real_,
        source_results_dir = "landmark50_projection",
        landmark_fraction = n_landmarks / n,
        n_landmarks = n_landmarks,
        projection_only = TRUE,
        projection_mode = projection_mode,
        projection_backend = as.character(projection_backend),
        projection_fused = projection_fused,
        projection_materialized = projection_materialized,
        selection_time_sec = selection$elapsed,
        landmark_knn_time_sec = landmark_knn$elapsed,
        landmark_embedding_time_sec = landmark_layout$elapsed,
        projection_knn_time_sec = projection_elapsed,
        interpolation_time_sec = interpolation_elapsed,
        post_preprocess_total_sec = post_preprocess_total,
        run_label = paste0("landmark50 ", backend),
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      data.frame(
        dataset = dataset$name,
        n = n,
        p = ncol(dataset$x),
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
        knn_backend = as.character(landmark_knn_backend),
        embedding_time_sec = NA_real_,
        total_time_sec = NA_real_,
        trustworthiness = NA_real_,
        continuity = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        label_knn_accuracy = NA_real_,
        silhouette = NA_real_,
        distance_spearman = NA_real_,
        layout_path = NA_character_,
        embedding_speedup_vs_cpu = NA_real_,
        total_speedup_vs_cpu = NA_real_,
        trust_delta_vs_cpu = NA_real_,
        label_accuracy_delta_vs_cpu = NA_real_,
        source_results_dir = "landmark50_projection",
        landmark_fraction = n_landmarks / n,
        n_landmarks = n_landmarks,
        projection_only = TRUE,
        projection_mode = projection_mode,
        projection_backend = as.character(projection_backend),
        projection_fused = projection_fused,
        projection_materialized = projection_materialized,
        selection_time_sec = selection$elapsed,
        landmark_knn_time_sec = landmark_knn$elapsed,
        landmark_embedding_time_sec = NA_real_,
        projection_knn_time_sec = projection_elapsed,
        interpolation_time_sec = interpolation_elapsed,
        post_preprocess_total_sec = NA_real_,
        run_label = paste0("landmark50 ", backend),
        stringsAsFactors = FALSE
      )
    })
    rows[[length(rows) + 1L]] <- row
  }
  do.call(rbind, rows)
}

plot_dataset_layouts <- function(dataset, rows, plot_dir, max_points, seed) {
  ok <- rows$status == "success" & file.exists(rows$layout_path)
  rows <- rows[ok, , drop = FALSE]
  if (nrow(rows) == 0L) return(NA_character_)

  preferred <- c("full cpu", "full metal", "uwot fast_sgd", "landmark50 cpu", "landmark50 metal")
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

  path <- file.path(plot_dir, paste0(safe_name(dataset$name), "_landmark50_comparison.png"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = max(1800, 480 * nrow(rows)), height = 900, res = 150)
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
        "%s\npost-pre %.2fs, trust %.4f\nlabel %.3f",
        rows$run_label[[i]],
        rows$post_preprocess_total_sec[[i]],
        rows$trustworthiness[[i]],
        rows$label_knn_accuracy[[i]]
      )
    )
    box(col = "grey85")
  }
  title(paste0(dataset$name, " landmark 50% projection comparison"), outer = TRUE, line = 0.2, cex.main = 1.1)
  path
}

plot_speed_quality <- function(results, plot_dir) {
  ok <- results$status == "success" & is.finite(results$post_preprocess_total_sec)
  results <- results[ok, , drop = FALSE]
  if (nrow(results) == 0L) return(NA_character_)
  path <- file.path(plot_dir, "landmark50_speed_quality_summary.png")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 1500, height = 900, res = 150)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1L, 2L), mar = c(8, 4, 3, 1))
  labels <- paste(results$dataset, results$run_label, sep = "\n")
  col <- ifelse(grepl("landmark50", results$run_label), "#1b9e77",
                ifelse(grepl("uwot", results$run_label), "#d95f02", "#7570b3"))
  barplot(
    results$post_preprocess_total_sec,
    names.arg = labels,
    las = 2,
    col = col,
    ylab = "Post-preprocessing time (sec)",
    main = "KNN/projection + embedding"
  )
  plot(
    results$post_preprocess_total_sec,
    results$trustworthiness,
    pch = 19,
    col = col,
    xlab = "Post-preprocessing time (sec)",
    ylab = "Trustworthiness / kNN@15",
    main = "Speed vs local quality"
  )
  text(
    results$post_preprocess_total_sec,
    results$trustworthiness,
    labels = paste(results$dataset, results$run_label),
    pos = 4,
    cex = 0.58
  )
  path
}

source_benchmark_helpers()

datasets <- parse_csv("datasets", "fashion_mnist,mnist")
backends <- parse_csv("backends", "cpu,metal")
seed <- int_arg("seed", 6L)
k <- int_arg("k", 50L)
landmark_fraction <- num_arg("landmark-fraction", 0.5)
min_n <- int_arg("min-n", 50000L)
max_n <- int_arg("max-n", 70000L)
pca_dims <- int_arg("pca-dims", 50L)
quality_sample_size <- int_arg("quality-sample-size", 3000L)
global_sample_size <- int_arg("global-sample-size", 1500L)
max_plot_points <- int_arg("max-plot-points", 70000L)
knn_backend <- parse_scalar("knn-backend", "auto")
projection_mode <- parse_scalar("projection-mode", "auto")
preprocess_backend <- parse_scalar("preprocess-backend", knn_backend)
results_root <- parse_scalar("results-root", "/Users/stefano/Documents/fastEmbedR-results")
results_dir <- parse_scalar("results-dir", file.path(results_root, paste0("landmark50_projection_seed", seed)))
baseline_dirs <- parse_csv("baseline-dirs", "fashion_threaded4_seed6,mnist_threaded4_seed6")
cache_dir <- parse_scalar("cache-dir", file.path(results_dir, "cache"))
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

  baseline <- read_baseline_rows(dataset$name, results_root, baseline_dirs)
  landmark <- benchmark_landmark_projection(
    dataset,
    backends = backends,
    k = k,
    landmark_fraction = landmark_fraction,
    seed = seed,
    knn_backend = knn_backend,
    projection_mode = projection_mode,
    timeout_sec = timeout_sec,
    quality_sample_size = quality_sample_size,
    global_sample_size = global_sample_size,
    layout_dir = layout_dir
  )

  rows <- bind_rows_base(baseline, landmark)
  rows <- rows[rows$status == "success", , drop = FALSE]
  if (nrow(rows) > 0L) {
    full_cpu <- rows$run_label == "full cpu"
    if (any(full_cpu)) {
      ref <- rows[which(full_cpu)[[1L]], , drop = FALSE]
      ok <- is.finite(rows$post_preprocess_total_sec)
      rows$post_preprocess_speedup_vs_full_cpu[ok] <- ref$post_preprocess_total_sec[[1L]] / rows$post_preprocess_total_sec[ok]
      rows$trust_delta_vs_full_cpu <- rows$trustworthiness - ref$trustworthiness[[1L]]
      rows$label_accuracy_delta_vs_full_cpu <- rows$label_knn_accuracy - ref$label_knn_accuracy[[1L]]
    } else {
      rows$post_preprocess_speedup_vs_full_cpu <- NA_real_
      rows$trust_delta_vs_full_cpu <- NA_real_
      rows$label_accuracy_delta_vs_full_cpu <- NA_real_
    }
    plot_paths <- c(plot_paths, plot_dataset_layouts(dataset, rows, plot_dir, max_plot_points, seed))
  }
  all_rows[[length(all_rows) + 1L]] <- rows
  write.csv(bind_rows_base(all_rows), file.path(results_dir, "landmark50_results_latest.csv"), row.names = FALSE)
}

results <- bind_rows_base(all_rows)
if (nrow(results) > 0L) {
  summary_path <- file.path(results_dir, "landmark50_results_summary.csv")
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
	    "dataset", "run_label", "backend_used", "post_preprocess_total_sec",
	    "embedding_time_sec", "landmark_embedding_time_sec", "projection_knn_time_sec",
	    "projection_mode", "projection_backend", "projection_fused", "trustworthiness", "knn_preservation_50", "label_knn_accuracy",
	    "post_preprocess_speedup_vs_full_cpu", "trust_delta_vs_full_cpu",
	    "label_accuracy_delta_vs_full_cpu"
	  ), names(results))
  print(results[, cols, drop = FALSE])
}
