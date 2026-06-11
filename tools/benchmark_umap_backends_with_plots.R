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

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_.-]+", "_", x)
}

numeric_csv_arg <- function(name, default) {
  values <- parse_csv(name, default)
  values <- suppressWarnings(as.integer(as.numeric(values)))
  values <- values[is.finite(values) & values > 0L]
  if (length(values) == 0L) {
    values <- suppressWarnings(as.integer(as.numeric(parse_csv(name, default))))
    values <- values[is.finite(values) & values > 0L]
  }
  values
}

metric_or_na <- function(metrics, name) {
  if (name %in% names(metrics)) metrics[[name]][[1L]] else NA_real_
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

backend_matches_request <- function(requested, used) {
  if (is.null(used) || length(used) == 0L || is.na(used[[1L]])) return(FALSE)
  used <- as.character(used[[1L]])
  if (requested %in% c("cuda", "metal")) {
    identical(used, requested) || startsWith(used, paste0(requested, "+"))
  } else {
    identical(used, requested)
  }
}

backend_plot_color <- function(backend_used) {
  gpu_assisted <- grepl("^(cuda|metal)(\\+|$)", backend_used)
  external_ref <- grepl("^(uwot|cuml|mlx)", backend_used)
  ifelse(gpu_assisted, "#2c7fb8", ifelse(external_ref, "#d95f02", "#7f7f7f"))
}

plot_row_label <- function(rows, i) {
  method <- if ("method" %in% names(rows)) rows$method[[i]] else rows$backend_used[[i]]
  backend <- rows$backend_used[[i]]
  if (identical(method, "fastEmbedR_umap")) {
    paste0("fastEmbedR ", backend)
  } else {
    backend
  }
}

plot_dataset_layouts <- function(dataset, rows, layout_dir, plot_dir, max_points, seed) {
  ok <- rows$status == "success" & file.exists(rows$layout_path)
  rows <- rows[ok, , drop = FALSE]
  if (nrow(rows) == 0L) return(NA_character_)

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

  path <- file.path(plot_dir, paste0(safe_name(dataset$name), "_umap_backends.png"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 1800, height = 900, res = 160)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1L, nrow(rows)), mar = c(2.5, 2.5, 3.5, 0.8), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(rows))) {
    layout <- readRDS(rows$layout_path[[i]])
    layout <- as.matrix(layout)[keep, 1:2, drop = FALSE]
    plot(
      layout[order, 1L], layout[order, 2L],
      col = cols[order], pch = 16, cex = if (length(keep) > 50000L) 0.16 else 0.28,
      xlab = "", ylab = "", axes = FALSE,
      main = sprintf(
        "%s\n%.2fs, trust %.4f",
        plot_row_label(rows, i),
        rows$embedding_time_sec[[i]],
        rows$trustworthiness[[i]]
      )
    )
    box(col = "grey85")
  }
  title(dataset$name, outer = TRUE, line = 0.2, cex.main = 1.2)
  path
}

plot_summary <- function(results, plot_dir) {
  ok <- results$status == "success"
  results <- results[ok, , drop = FALSE]
  if (nrow(results) == 0L) return(NA_character_)
  path <- file.path(plot_dir, "backend_speed_quality_summary.png")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = 1600, height = 900, res = 160)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1, 2), mar = c(8, 4, 3, 1))
  labels <- paste(results$dataset, results$backend_used, sep = "\n")
  barplot(
    results$embedding_time_sec,
    names.arg = labels,
    las = 2,
    col = backend_plot_color(results$backend_used),
    ylab = "UMAP embedding time (sec)",
    main = "Speed"
  )
  plot(
    results$embedding_time_sec,
    results$trustworthiness,
    pch = 19,
    col = backend_plot_color(results$backend_used),
    xlab = "UMAP embedding time (sec)",
    ylab = "Trustworthiness / kNN@15",
    main = "Speed vs quality"
  )
  text(
    results$embedding_time_sec,
    results$trustworthiness,
    labels = paste(results$dataset, results$backend_used, sep = " "),
    pos = 4,
    cex = 0.62
  )
  path
}

umap_reference_config <- function(indices, distances) {
  cfg <- fastEmbedR:::fast_knn_umap_config(nrow(indices), ncol(indices), "cpu")
  cfg <- fastEmbedR:::apply_umap_connectivity_spectral_rule(cfg, indices)
  cfg
}

run_uwot_reference <- function(dataset, knn, seed, cfg) {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package `uwot` is not installed.", call. = FALSE)
  }
  uwot::umap(
    X = dataset$x,
    n_neighbors = ncol(knn$indices),
    n_components = 2L,
    nn_method = list(idx = knn$indices, dist = knn$distances),
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
  )
}

cuml_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) && tryCatch({
    manifold <- reticulate::import("cuml.manifold", delay_load = FALSE)
    np <- reticulate::import("numpy", delay_load = FALSE)
    reticulate::py_has_attr(manifold, "UMAP") && reticulate::py_has_attr(np, "array")
  }, error = function(e) FALSE)
}

run_cuml_reference <- function(dataset, knn, seed, cfg) {
  if (!cuml_available()) {
    stop("Python RAPIDS cuML UMAP is not available through reticulate.", call. = FALSE)
  }
  np <- reticulate::import("numpy", delay_load = FALSE, convert = FALSE)
  manifold <- reticulate::import("cuml.manifold", delay_load = FALSE, convert = FALSE)
  x <- np$array(dataset$x, dtype = "float32")
  idx <- np$array(knn$indices - 1L, dtype = "int64")
  dst <- np$array(knn$distances, dtype = "float32")
  model <- manifold$UMAP(
    n_neighbors = as.integer(ncol(knn$indices)),
    n_components = 2L,
    n_epochs = as.integer(cfg$n_epochs),
    min_dist = cfg$min_dist,
    metric = "euclidean",
    init = "spectral",
    random_state = as.integer(seed),
    verbose = FALSE
  )
  out <- model$fit_transform(x, knn_graph = reticulate::tuple(idx, dst))
  as.matrix(reticulate::py_to_r(out))
}

run_reference <- function(reference, dataset, knn, seed, cfg) {
  switch(
    reference,
    uwot_fast_sgd = list(
      method = "uwot_umap_fast_sgd",
      backend_used = "uwot_fast_sgd",
      package = "uwot",
      layout = run_uwot_reference(dataset, knn, seed, cfg)
    ),
    cuml_umap = list(
      method = "cuml_umap",
      backend_used = "cuml_umap",
      package = "cuml",
      layout = run_cuml_reference(dataset, knn, seed, cfg)
    ),
    stop("Unknown reference method: ", reference, call. = FALSE)
  )
}

source_benchmark_helpers()

datasets <- parse_csv("datasets", "cifar10_rgb8x8,fashion_mnist,mnist,covtype")
backends <- parse_csv("backends", "cpu,metal")
references <- parse_csv("references", "uwot_fast_sgd")
seed <- int_arg("seed", 4L)
k <- int_arg("k", 50L)
min_n <- int_arg("min-n", 50000L)
max_n <- int_arg("max-n", 70000L)
pca_dims <- int_arg("pca-dims", 50L)
quality_sample_size <- int_arg("quality-sample-size", 3000L)
global_sample_size <- int_arg("global-sample-size", 1500L)
quality_k <- numeric_csv_arg("quality-k", "15,30,50")
primary_quality_k <- int_arg("primary-quality-k", min(15L, max(quality_k)))
max_plot_points <- int_arg("max-plot-points", 70000L)
knn_backend <- parse_scalar("knn-backend", "auto")
preprocess_backend <- parse_scalar("preprocess-backend", knn_backend)
results_dir <- parse_scalar("results-dir", file.path("results", "umap_backend_plots"))
timeout_sec <- as.numeric(parse_scalar("timeout-sec", "0"))

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
layout_dir <- file.path(results_dir, "layouts")
plot_dir <- file.path(results_dir, "plots")
cache_dir <- parse_scalar("cache-dir", file.path(results_dir, "cache"))

rows <- list()
plot_paths <- character()

for (dataset_id in datasets) {
  dataset <- load_named_dataset(dataset_id, cache_dir = cache_dir, min_n = min_n, max_n = max_n, seed = seed)
  if (is.null(dataset)) next
  message("Preparing ", dataset$name, " (n=", nrow(dataset$x), ", p=", ncol(dataset$x), ")")
  dataset <- prepare_dataset(dataset, pca_dims = pca_dims, seed = seed, preprocess_backend = preprocess_backend)
  q <- stratified_sample(dataset$labels, nrow(dataset$x), quality_sample_size, seed + 100L)
  message("Building shared KNN for ", dataset$name, ", k=", k, ", backend=", knn_backend)
  knn_time <- system.time({
    knn_self <- fastEmbedR::nn(dataset$x, dataset$x, k = k + 1L, backend = knn_backend)
    knn <- drop_self_knn(knn_self, k)
  })[["elapsed"]]
  knn_backend_used <- attr(knn_self, "backend")
  if (is.null(knn_backend_used)) knn_backend_used <- knn_backend
  ref_cfg <- umap_reference_config(knn$indices, knn$distances)

  dataset_rows <- list()
  for (backend in backends) {
    message("  UMAP backend=", backend)
    row <- tryCatch({
      measured <- measure_layout(
        fastEmbedR::embed_knn(knn, method = "umap", seed = seed, backend = backend),
        nrow(dataset$x),
        timeout_sec
      )
      cfg <- attr(measured$layout, "fastEmbedR_config")
      backend_used <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else backend
      if (backend %in% c("cuda", "metal") && !backend_matches_request(backend, backend_used)) {
        stop("Requested backend ", backend, " but backend_used was ", backend_used, call. = FALSE)
      }
      metrics <- fastEmbedR::evaluate_embedding(
        dataset$x[q, , drop = FALSE],
        measured$layout[q, , drop = FALSE],
        labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
        k = quality_k,
        primary_k = primary_quality_k,
        sample_size_for_global_metrics = min(global_sample_size, length(q)),
        seed = seed,
        method = "umap",
        backend = backend_used,
        dataset = dataset$name
      )
      layout_path <- file.path(
        layout_dir,
        paste0(safe_name(paste(dataset$name, backend_used, seed, sep = "_")), ".rds")
      )
      dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(measured$layout, layout_path, version = 2)
      data.frame(
        dataset = dataset$name,
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        method = "fastEmbedR_umap",
        package = "fastEmbedR",
        backend_requested = backend,
        backend_used = backend_used,
        status = "success",
        error_message = NA_character_,
        seed = seed,
        k = k,
        quality_primary_k = metrics$primary_k[[1L]],
        preprocess_time_sec = dataset$preprocess_time_sec,
        knn_time_sec = as.numeric(knn_time),
        knn_backend = knn_backend_used,
        embedding_time_sec = measured$elapsed,
        total_time_sec = dataset$preprocess_time_sec + as.numeric(knn_time) + measured$elapsed,
        trustworthiness = metrics$trustworthiness[[1L]],
        continuity = metrics$continuity[[1L]],
        knn_preservation_15 = metrics$knn_preservation_15[[1L]],
        knn_preservation_30 = metrics$knn_preservation_30[[1L]],
        knn_preservation_50 = metrics$knn_preservation_50[[1L]],
        knn_preservation_100 = metric_or_na(metrics, "knn_preservation_100"),
        knn_preservation_150 = metric_or_na(metrics, "knn_preservation_150"),
        label_knn_accuracy = metrics$label_knn_accuracy[[1L]],
        rare_class_recall = metrics$rare_class_recall[[1L]],
        silhouette = metrics$silhouette[[1L]],
        distance_spearman = metrics$distance_spearman[[1L]],
        layout_path = layout_path,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      data.frame(
        dataset = dataset$name,
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        method = "fastEmbedR_umap",
        package = "fastEmbedR",
        backend_requested = backend,
        backend_used = NA_character_,
        status = if (backend %in% c("cuda", "metal")) "backend_unavailable" else "failed",
        error_message = conditionMessage(e),
        seed = seed,
        k = k,
        quality_primary_k = primary_quality_k,
        preprocess_time_sec = dataset$preprocess_time_sec,
        knn_time_sec = as.numeric(knn_time),
        knn_backend = knn_backend_used,
        embedding_time_sec = NA_real_,
        total_time_sec = NA_real_,
        trustworthiness = NA_real_,
        continuity = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        knn_preservation_100 = NA_real_,
        knn_preservation_150 = NA_real_,
        label_knn_accuracy = NA_real_,
        rare_class_recall = NA_real_,
        silhouette = NA_real_,
        distance_spearman = NA_real_,
        layout_path = NA_character_,
        stringsAsFactors = FALSE
      )
    })
    dataset_rows[[length(dataset_rows) + 1L]] <- row
    rows[[length(rows) + 1L]] <- row
    current <- do.call(rbind, rows)
    write.csv(current, file.path(results_dir, "backend_results_latest.csv"), row.names = FALSE)
  }

  for (reference in references) {
    message("  UMAP reference=", reference)
    row <- tryCatch({
      reference_out <- NULL
      measured <- measure_layout({
        reference_out <<- run_reference(reference, dataset, knn, seed, ref_cfg)
        reference_out$layout
      }, nrow(dataset$x), timeout_sec)
      backend_used <- reference_out$backend_used
      metrics <- fastEmbedR::evaluate_embedding(
        dataset$x[q, , drop = FALSE],
        measured$layout[q, , drop = FALSE],
        labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
        k = quality_k,
        primary_k = primary_quality_k,
        sample_size_for_global_metrics = min(global_sample_size, length(q)),
        seed = seed,
        method = reference_out$method,
        backend = backend_used,
        dataset = dataset$name
      )
      layout_path <- file.path(
        layout_dir,
        paste0(safe_name(paste(dataset$name, backend_used, seed, sep = "_")), ".rds")
      )
      dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(measured$layout, layout_path, version = 2)
      data.frame(
        dataset = dataset$name,
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        method = reference_out$method,
        package = reference_out$package,
        backend_requested = reference,
        backend_used = backend_used,
        status = "success",
        error_message = NA_character_,
        seed = seed,
        k = k,
        quality_primary_k = metrics$primary_k[[1L]],
        preprocess_time_sec = dataset$preprocess_time_sec,
        knn_time_sec = as.numeric(knn_time),
        knn_backend = knn_backend_used,
        embedding_time_sec = measured$elapsed,
        total_time_sec = dataset$preprocess_time_sec + as.numeric(knn_time) + measured$elapsed,
        trustworthiness = metrics$trustworthiness[[1L]],
        continuity = metrics$continuity[[1L]],
        knn_preservation_15 = metrics$knn_preservation_15[[1L]],
        knn_preservation_30 = metrics$knn_preservation_30[[1L]],
        knn_preservation_50 = metrics$knn_preservation_50[[1L]],
        knn_preservation_100 = metric_or_na(metrics, "knn_preservation_100"),
        knn_preservation_150 = metric_or_na(metrics, "knn_preservation_150"),
        label_knn_accuracy = metrics$label_knn_accuracy[[1L]],
        rare_class_recall = metrics$rare_class_recall[[1L]],
        silhouette = metrics$silhouette[[1L]],
        distance_spearman = metrics$distance_spearman[[1L]],
        layout_path = layout_path,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      data.frame(
        dataset = dataset$name,
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        method = paste0(reference, "_umap"),
        package = sub("_.*$", "", reference),
        backend_requested = reference,
        backend_used = NA_character_,
        status = "not_installed_or_failed",
        error_message = conditionMessage(e),
        seed = seed,
        k = k,
        quality_primary_k = primary_quality_k,
        preprocess_time_sec = dataset$preprocess_time_sec,
        knn_time_sec = as.numeric(knn_time),
        knn_backend = knn_backend_used,
        embedding_time_sec = NA_real_,
        total_time_sec = NA_real_,
        trustworthiness = NA_real_,
        continuity = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        knn_preservation_100 = NA_real_,
        knn_preservation_150 = NA_real_,
        label_knn_accuracy = NA_real_,
        rare_class_recall = NA_real_,
        silhouette = NA_real_,
        distance_spearman = NA_real_,
        layout_path = NA_character_,
        stringsAsFactors = FALSE
      )
    })
    dataset_rows[[length(dataset_rows) + 1L]] <- row
    rows[[length(rows) + 1L]] <- row
    current <- do.call(rbind, rows)
    write.csv(current, file.path(results_dir, "backend_results_latest.csv"), row.names = FALSE)
  }
  dataset_df <- do.call(rbind, dataset_rows)
  plot_paths <- c(plot_paths, plot_dataset_layouts(dataset, dataset_df, layout_dir, plot_dir, max_plot_points, seed))
}

results <- if (length(rows)) do.call(rbind, rows) else data.frame()
if (nrow(results) > 0L) {
  for (dataset in unique(results$dataset)) {
    cpu <- results[results$dataset == dataset & results$backend_used == "cpu" & results$status == "success", , drop = FALSE]
    if (nrow(cpu) == 0L) next
    idx <- results$dataset == dataset & results$status == "success"
    results$embedding_speedup_vs_cpu[idx] <- cpu$embedding_time_sec[[1L]] / results$embedding_time_sec[idx]
    results$total_speedup_vs_cpu[idx] <- cpu$total_time_sec[[1L]] / results$total_time_sec[idx]
    results$trust_delta_vs_cpu[idx] <- results$trustworthiness[idx] - cpu$trustworthiness[[1L]]
    results$label_accuracy_delta_vs_cpu[idx] <- results$label_knn_accuracy[idx] - cpu$label_knn_accuracy[[1L]]
  }
  write.csv(results, file.path(results_dir, "backend_results_latest.csv"), row.names = FALSE)
  summary_path <- file.path(results_dir, "backend_results_summary.csv")
  write.csv(results, summary_path, row.names = FALSE)
  summary_plot <- plot_summary(results, plot_dir)
  writeLines(c(capture.output(fastEmbedR::backend_info()), ""), file.path(results_dir, "backend_info.txt"))
  message("\nSaved results:")
  message("  ", normalizePath(summary_path, winslash = "/", mustWork = FALSE))
  message("  ", normalizePath(summary_plot, winslash = "/", mustWork = FALSE))
  for (path in plot_paths[!is.na(plot_paths)]) {
    message("  ", normalizePath(path, winslash = "/", mustWork = FALSE))
  }
  print(results[, c(
    "dataset", "method", "backend_requested", "backend_used", "status",
    "knn_backend", "embedding_time_sec", "embedding_speedup_vs_cpu",
    "trustworthiness", "trust_delta_vs_cpu", "label_knn_accuracy"
  )])
}
