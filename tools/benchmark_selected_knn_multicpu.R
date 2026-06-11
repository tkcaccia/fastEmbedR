#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastEmbedR)
})

arg_value <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) default else sub(prefix, "", hit[[1L]], fixed = TRUE)
}

as_int <- function(name, default) as.integer(as.numeric(arg_value(name, as.character(default))))

safe_chr <- function(x) {
  if (is.null(x) || length(x) == 0L) NA_character_ else as.character(x[[1L]])
}

load_cached_dataset <- function(path, n = NULL) {
  obj <- readRDS(path)
  x <- if (is.list(obj) && !is.null(obj$x)) obj$x else obj
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (!is.null(n) && is.finite(n) && nrow(x) > n) {
    x <- x[seq_len(n), , drop = FALSE]
  }
  x
}

strip_self <- function(indices, query_rows, k) {
  out <- matrix(NA_integer_, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    idx <- indices[i, indices[i, ] != query_rows[[i]]]
    out[i, ] <- idx[seq_len(k)]
  }
  out
}

recall_against_exact_subset <- function(x, knn, k, sample_size, seed) {
  set.seed(seed + 1009L)
  rows <- sort(sample.int(nrow(x), min(sample_size, nrow(x))))
  exact <- fastEmbedR::nn(
    x,
    x[rows, , drop = FALSE],
    k = k + 1L,
    backend = if (fastEmbedR::metal_available()) "metal" else "cpu",
    n_threads = 4L
  )
  exact_idx <- strip_self(exact$indices, rows, k)
  idx <- knn$indices[rows, , drop = FALSE]
  if (all(idx[, 1L] == rows)) idx <- idx[, -1L, drop = FALSE]
  idx <- idx[, seq_len(k), drop = FALSE]
  fastEmbedR:::knn_recall(list(indices = idx), list(indices = exact_idx), k = k)
}

run_fastembedr_nn <- function(x, backend, k, n_threads) {
  gc(FALSE)
  elapsed <- system.time({
    value <- fastEmbedR::nn(
      x,
      k = k + 1L,
      backend = backend,
      n_threads = n_threads
    )
  })[["elapsed"]]
  list(value = value, elapsed = as.numeric(elapsed))
}

run_uwot_annoy_nn <- function(x, k, n_threads) {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package `uwot` is not installed.", call. = FALSE)
  }
  gc(FALSE)
  elapsed <- system.time({
    value <- uwot:::annoy_nn(
      x,
      k = k + 1L,
      metric = "euclidean",
      n_trees = 50L,
      search_k = 2L * (k + 1L) * 50L,
      n_threads = n_threads,
      verbose = FALSE
    )
  })[["elapsed"]]
  list(
    value = list(indices = value$idx, distances = value$dist),
    elapsed = as.numeric(elapsed)
  )
}

main <- function() {
  n_threads <- as_int("n-threads", 4L)
  k <- as_int("k", 30L)
  sample_size <- as_int("recall-sample", 512L)
  n_limit <- as_int("n", NA_integer_)
  out_dir <- arg_value("out-dir", "results/selected_knn_multicpu")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  default_paths <- c(
    fashion = "/Users/stefano/Documents/fastEmbedR-results/nn_umap_preprocessed_seed6/preprocessed/fashion_mnist_70000_pca50_seed6.rds",
    mnist = "/Users/stefano/Documents/fastEmbedR-results/nn_umap_preprocessed_seed6/preprocessed/mnist_idx_70000_pca50_seed6.rds"
  )
  paths <- default_paths[file.exists(default_paths)]
  if (length(paths) == 0L) {
    stop("No cached PCA benchmark datasets were found.", call. = FALSE)
  }

  backends <- c("cpu_ivf", "cpu_annoy", "metal", "metal_ivf")
  rows <- list()
  for (dataset in names(paths)) {
    x <- load_cached_dataset(paths[[dataset]], n = n_limit)
    message("Dataset ", dataset, ": ", nrow(x), " x ", ncol(x))
    for (backend in backends) {
      if (startsWith(backend, "metal") && !fastEmbedR::metal_available()) next
      result <- tryCatch(run_fastembedr_nn(x, backend, k, n_threads), error = identity)
      if (inherits(result, "error")) {
        row <- data.frame(
          dataset = dataset,
          n = nrow(x),
          p = ncol(x),
          method = paste0("fastEmbedR_", backend),
          backend = backend,
          n_threads = if (startsWith(backend, "cpu")) n_threads else NA_integer_,
          sec = NA_real_,
          recall_at_k = NA_real_,
          median_recall_at_k = NA_real_,
          status = "failed",
          error_message = conditionMessage(result),
          stringsAsFactors = FALSE
        )
      } else {
        recall <- recall_against_exact_subset(x, result$value, k, sample_size, 7L)
        row <- data.frame(
          dataset = dataset,
          n = nrow(x),
          p = ncol(x),
          method = paste0("fastEmbedR_", backend),
          backend = safe_chr(attr(result$value, "backend")),
          n_threads = if (startsWith(backend, "cpu")) n_threads else NA_integer_,
          sec = result$elapsed,
          recall_at_k = recall$recall_at_k[[1L]],
          median_recall_at_k = recall$median_recall_at_k[[1L]],
          status = "success",
          error_message = NA_character_,
          stringsAsFactors = FALSE
        )
      }
      print(row)
      rows[[length(rows) + 1L]] <- row
    }

    uwot_result <- tryCatch(run_uwot_annoy_nn(x, k, n_threads), error = identity)
    if (inherits(uwot_result, "error")) {
      row <- data.frame(
        dataset = dataset,
        n = nrow(x),
        p = ncol(x),
        method = "uwot_internal_annoy_nn",
        backend = "uwot_annoy",
        n_threads = n_threads,
        sec = NA_real_,
        recall_at_k = NA_real_,
        median_recall_at_k = NA_real_,
        status = "failed",
        error_message = conditionMessage(uwot_result),
        stringsAsFactors = FALSE
      )
    } else {
      recall <- recall_against_exact_subset(x, uwot_result$value, k, sample_size, 7L)
      row <- data.frame(
        dataset = dataset,
        n = nrow(x),
        p = ncol(x),
        method = "uwot_internal_annoy_nn",
        backend = "uwot_annoy",
        n_threads = n_threads,
        sec = uwot_result$elapsed,
        recall_at_k = recall$recall_at_k[[1L]],
        median_recall_at_k = recall$median_recall_at_k[[1L]],
        status = "success",
        error_message = NA_character_,
        stringsAsFactors = FALSE
      )
    }
    print(row)
    rows[[length(rows) + 1L]] <- row
  }

  out <- do.call(rbind, rows)
  stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  write.csv(out, file.path(out_dir, paste0("selected_knn_multicpu_", stamp, ".csv")), row.names = FALSE)
  write.csv(out, file.path(out_dir, "latest_selected_knn_multicpu.csv"), row.names = FALSE)
  invisible(out)
}

main()
