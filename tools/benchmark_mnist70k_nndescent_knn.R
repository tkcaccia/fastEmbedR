#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

arg_int <- function(name, default) as.integer(as.numeric(arg_value(name, as.character(default))))

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

repo_root <- normalizePath(getwd(), mustWork = TRUE)
if (identical(Sys.getenv("FASTEMBEDR_LOAD_SOURCE"), "1") &&
    requireNamespace("devtools", quietly = TRUE)) {
  suppressPackageStartupMessages(devtools::load_all(repo_root, quiet = TRUE))
} else {
  suppressPackageStartupMessages(library(fastEmbedR))
}

timed <- function(expr) {
  invisible(gc())
  value <- NULL
  sec <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(sec))
}

stratified_rows <- function(labels, n, seed) {
  labels <- factor(labels)
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  levs <- levels(labels)
  base <- floor(n / length(levs))
  remainder <- n - base * length(levs)
  rows <- integer(0L)
  for (lev in levs) {
    idx <- which(labels == lev)
    take <- min(length(idx), base + as.integer(remainder > 0L))
    remainder <- max(0L, remainder - 1L)
    rows <- c(rows, sample(idx, take))
  }
  sort(rows)
}

drop_self_exact <- function(raw, query_rows, k) {
  idx <- as.matrix(raw$indices)
  dist <- as.matrix(raw$distances)
  out_idx <- matrix(0L, nrow(idx), k)
  out_dist <- matrix(0, nrow(idx), k)
  for (i in seq_len(nrow(idx))) {
    keep <- idx[i, ] != query_rows[i]
    kept_idx <- idx[i, keep]
    kept_dist <- dist[i, keep]
    if (length(kept_idx) < k) {
      kept_idx <- idx[i, seq_len(ncol(idx))]
      kept_dist <- dist[i, seq_len(ncol(dist))]
    }
    out_idx[i, ] <- kept_idx[seq_len(k)]
    out_dist[i, ] <- kept_dist[seq_len(k)]
  }
  list(indices = out_idx, distances = out_dist)
}

mean_distance_ratio <- function(approx, exact, rows, k) {
  approx_dist <- approx$distances[rows, seq_len(k), drop = FALSE]
  exact_dist <- exact$distances[, seq_len(k), drop = FALSE]
  ok <- is.finite(approx_dist) & is.finite(exact_dist) & exact_dist > 0
  if (!any(ok)) return(NA_real_)
  mean(approx_dist[ok] / exact_dist[ok], na.rm = TRUE)
}

as_jsonish <- function(x) {
  if (is.null(x)) return(NA_character_)
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(capture.output(str(x, max.level = 2L)), collapse = " ")
}

run_backend <- function(x, labels, backend, k, n_threads, seed, recall_rows, exact_subset) {
  tryCatch({
    measured <- timed(fastEmbedR:::nn_without_self(
      x,
      k = k,
      backend = backend,
      n_threads = n_threads
    ))
    result <- measured$value
    recall <- fastEmbedR:::knn_recall(
      list(indices = result$indices[recall_rows, seq_len(k), drop = FALSE]),
      exact_subset,
      k = k
    )
    approx <- attr(result, "approximation", exact = TRUE)
    data.frame(
      backend_requested = backend,
      backend_used = as.character(attr(result, "backend")),
      status = "success",
      error_message = NA_character_,
      elapsed_sec = measured$sec,
      n = nrow(x),
      p = ncol(x),
      k = k,
      recall_sample = length(recall_rows),
      recall_at_k = recall$recall_at_k[[1L]],
      median_recall_at_k = recall$median_recall_at_k[[1L]],
      min_recall_at_k = recall$min_recall_at_k[[1L]],
      mean_distance_ratio = mean_distance_ratio(result, exact_subset, recall_rows, k),
      exact = isTRUE(attr(result, "exact")),
      approximation_json = as_jsonish(approx),
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      backend_requested = backend,
      backend_used = NA_character_,
      status = "failed",
      error_message = conditionMessage(e),
      elapsed_sec = NA_real_,
      n = nrow(x),
      p = ncol(x),
      k = k,
      recall_sample = length(recall_rows),
      recall_at_k = NA_real_,
      median_recall_at_k = NA_real_,
      min_recall_at_k = NA_real_,
      mean_distance_ratio = NA_real_,
      exact = NA,
      approximation_json = NA_character_,
      stringsAsFactors = FALSE
    )
  })
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
recall_sample <- arg_int("recall-sample", 512L)
host <- arg_value("host", Sys.info()[["nodename"]])
cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
backends <- arg_csv("backends", "cpu_nndescent,metal_nndescent")
out_dir <- arg_value("out-dir", file.path("results", "mnist70k_nndescent_knn", host))

if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- readRDS(cache)
take <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- as.matrix(mnist$x[take, , drop = FALSE])
storage.mode(x) <- "double"
labels <- droplevels(mnist$labels[take])
recall_rows <- stratified_rows(labels, min(recall_sample, nrow(x)), seed + 409L)

message(
  "MNIST NN-descent KNN benchmark: n=", nrow(x),
  " p=", ncol(x),
  " k=", k,
  " recall_sample=", length(recall_rows),
  " backends=", paste(backends, collapse = ",")
)
message("Computing exact subset reference")
exact_time <- timed(fastEmbedR::nn(
  x,
  x[recall_rows, , drop = FALSE],
  k = k + 1L,
  backend = "cpu",
  n_threads = n_threads
))
exact_subset <- drop_self_exact(exact_time$value, recall_rows, k)

rows <- lapply(backends, function(backend) {
  message("Running backend=", backend)
  run_backend(x, labels, backend, k, n_threads, seed, recall_rows, exact_subset)
})
results <- do.call(rbind, rows)
results$host <- host
results$exact_subset_sec <- exact_time$sec
results$device <- paste(capture.output(fastEmbedR::backend_info()), collapse = " | ")

path <- file.path(out_dir, "mnist70k_nndescent_knn_results.csv")
latest <- file.path(out_dir, "latest_mnist70k_nndescent_knn_results.csv")
write.csv(results, path, row.names = FALSE)
write.csv(results, latest, row.names = FALSE)
writeLines(capture.output(fastEmbedR::backend_info()), file.path(out_dir, "backend_info.txt"))

message("Results CSV: ", normalizePath(path, winslash = "/", mustWork = FALSE))
print(results[, c(
  "host", "backend_requested", "backend_used", "status", "elapsed_sec",
  "exact_subset_sec", "recall_at_k", "median_recall_at_k",
  "mean_distance_ratio", "error_message"
), drop = FALSE])
