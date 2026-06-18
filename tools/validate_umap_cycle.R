#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else file.path("results", paste0("umap_cycle_validation_", format(Sys.time(), "%Y%m%d_%H%M%S")))
data_root <- if (length(args) >= 2L) args[[2L]] else "/Users/stefano/Documents/fastEmbedR/Data"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
cache_dir <- file.path(dirname(normalizePath(out_dir, mustWork = FALSE)), "_umap_cycle_cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

load_dataset <- function(name) {
  path <- file.path(data_root, name, paste0(name, ".RData"))
  if (name == "FlowRepository_FR-FCM-ZYRM_files") {
    path <- file.path(data_root, name, "van_unen_FR-FCM-ZYRM.RData")
  }
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- if (exists("dataset", envir = env, inherits = FALSE)) {
    get("dataset", envir = env)
  } else {
    vals <- mget(ls(env), envir = env)
    vals[[which(vapply(vals, function(x) is.list(x) && !is.null(x$data), logical(1)))[[1L]]]]
  }
  x <- as.matrix(obj$data)
  storage.mode(x) <- "double"
  keep <- apply(x, 2L, function(v) all(is.finite(v)) && stats::sd(v) > 0)
  x <- x[, keep, drop = FALSE]
  x <- scale(x, center = TRUE, scale = FALSE)
  x[!is.finite(x)] <- 0
  list(data = x, labels = if (is.null(obj$labels)) NULL else as.factor(obj$labels))
}

label_cols <- function(labels) {
  if (is.null(labels)) return("#333333")
  labs <- as.integer(as.factor(labels))
  grDevices::hcl.colors(max(labs), "Dark 3")[labs]
}

plot_grid <- function(dataset, layouts, labels, path) {
  png(path, width = 2400, height = 1800, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mfrow = c(2, 2), mar = c(2, 2, 3, 1), bg = "white")
  cols <- label_cols(labels)
  cex <- if (length(cols) > 50000L) 0.28 else 0.55
  for (nm in names(layouts)) {
    y <- layouts[[nm]]
    plot(y[, 1], y[, 2], pch = 16, cex = cex, col = cols,
         axes = FALSE, xlab = "", ylab = "", main = paste(dataset, nm))
    box()
  }
}

metric_row <- function(dataset, method, backend, graph_mode, nn_sec, embed_sec, x, y, labels) {
  set.seed(123)
  keep <- if (nrow(x) > 5000L) sort(sample.int(nrow(x), 5000L)) else seq_len(nrow(x))
  x_metric <- x[keep, , drop = FALSE]
  y_metric <- y[keep, , drop = FALSE]
  labels_metric <- if (is.null(labels)) NULL else labels[keep]
  metrics <- tryCatch(
    fastEmbedR::evaluate_embedding(x_metric, y_metric, labels = labels_metric, k = c(15, 30, 50), sample_size_for_global_metrics = min(5000L, nrow(x_metric))),
    error = function(e) data.frame(trustworthiness = NA_real_, knn_preservation_15 = NA_real_, label_knn_accuracy = NA_real_)
  )
  data.frame(
    dataset = dataset,
    method = method,
    backend = backend,
    graph_mode = graph_mode,
    n = nrow(x),
    p = ncol(x),
    nn_sec = nn_sec,
    embed_sec = embed_sec,
    trustworthiness = metrics$trustworthiness[[1L]],
    knn_preservation_15 = metrics$knn_preservation_15[[1L]],
    label_knn_accuracy = metrics$label_knn_accuracy[[1L]],
    graph_mode_request = NA_character_,
    graph_mode_auto_rule = NA_character_,
    stringsAsFactors = FALSE
  )
}

suppressPackageStartupMessages({
  library(fastEmbedR)
  library(faissR)
})

if (!requireNamespace("uwot", quietly = TRUE)) {
  stop("uwot is required for the validation reference.", call. = FALSE)
}

datasets <- c("USPS", "MNIST", "FashionMNIST")
all_rows <- list()
for (dataset in datasets) {
  message("==== ", dataset, " ====")
  ds <- load_dataset(dataset)
  x <- ds$data
  labels <- ds$labels
  k <- 15L
  gc()
  cache_file <- file.path(cache_dir, paste0(dataset, "_centered_k", k, "_faissR_auto_nn.rds"))
  if (file.exists(cache_file)) {
    cached <- readRDS(cache_file)
    knn <- cached$knn
    nn_time <- cached$nn_sec
    message("using cached KNN: ", cache_file)
  } else {
    nn_time <- system.time({
      knn <- faissR::nn_without_self(x, k = k, metric = "euclidean", backend = "auto", n_threads = 4L)
    })[["elapsed"]]
    saveRDS(list(knn = knn, nn_sec = nn_time), cache_file)
  }
  layouts <- list()

  run_fe <- function(name, graph_mode) {
    gc()
    t <- system.time({
      y <- fastEmbedR::umap_knn(knn, backend = "cpu", graph_mode = graph_mode, n_threads = 4L, seed = 4L)
    })[["elapsed"]]
    cfg <- attr(y, "fastEmbedR_config")
    layouts[[name]] <<- y
    row <- metric_row(dataset, name, "cpu", cfg$graph_mode %||% graph_mode, nn_time, t, x, y, labels)
    row$graph_mode_request <- cfg$graph_mode_request %||% graph_mode
    row$graph_mode_auto_rule <- cfg$graph_mode_auto_rule %||% NA_character_
    row
  }

  rows <- list(
    run_fe("fastEmbedR_binary", "binary"),
    run_fe("fastEmbedR_fuzzy", "fuzzy")
  )

  gc()
  uwot_time <- system.time({
    y_uwot <- uwot::umap(
      x,
      n_neighbors = k,
      nn_method = list(idx = as.matrix(knn$indices), dist = as.matrix(knn$distances)),
      n_threads = 4L,
      n_sgd_threads = 4L,
      fast_sgd = TRUE,
      init = "spectral",
      min_dist = 0.1,
      ret_model = FALSE,
      verbose = FALSE,
      seed = 4L
    )
  })[["elapsed"]]
  layouts[["uwot_fast_sgd"]] <- y_uwot
  rows[[length(rows) + 1L]] <- metric_row(dataset, "uwot_fast_sgd", "cpu", "uwot", nn_time, uwot_time, x, y_uwot, labels)

  plot_path <- file.path(out_dir, "plots", paste0(dataset, "_umap_cycle_comparison.png"))
  plot_grid(dataset, layouts, labels, plot_path)
  all_rows <- c(all_rows, rows)
  utils::write.csv(do.call(rbind, all_rows), file.path(out_dir, "umap_cycle_results_partial.csv"), row.names = FALSE)
}

results <- do.call(rbind, all_rows)
utils::write.csv(results, file.path(out_dir, "umap_cycle_results.csv"), row.names = FALSE)
message("Saved: ", normalizePath(out_dir))
