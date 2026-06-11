#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

parse_int_csv <- function(name, default) {
  as.integer(strsplit(parse_scalar(name, default), ",", fixed = TRUE)[[1L]])
}

parse_num_csv <- function(name, default) {
  as.numeric(strsplit(parse_scalar(name, default), ",", fixed = TRUE)[[1L]])
}

make_blobs <- function(n, p, classes, seed) {
  set.seed(seed)
  labels <- factor(sample(rep(seq_len(classes), length.out = n)))
  centers <- matrix(rnorm(classes * p, sd = 3), classes, p)
  x <- matrix(rnorm(n * p, sd = 0.9), n, p) + centers[as.integer(labels), , drop = FALSE]
  list(name = paste0("blobs_", n, "x", p, "_c", classes), x = scale(x), labels = labels)
}

make_rare_blobs <- function(n, p, classes, seed) {
  set.seed(seed)
  props <- exp(seq(0, -2.2, length.out = classes))
  props <- props / sum(props)
  counts <- pmax(20L, as.integer(round(n * props)))
  counts[[which.max(counts)]] <- counts[[which.max(counts)]] + n - sum(counts)
  labels <- factor(rep(seq_len(classes), counts))
  labels <- factor(sample(labels))
  centers <- matrix(rnorm(classes * p, sd = 3.2), classes, p)
  spread <- seq(0.55, 1.25, length.out = classes)
  x <- matrix(0, n, p)
  for (cl in seq_len(classes)) {
    rows <- which(as.integer(labels) == cl)
    x[rows, ] <- matrix(rnorm(length(rows) * p, sd = spread[[cl]]), length(rows), p) +
      centers[cl, , drop = TRUE]
  }
  list(name = paste0("rare_blobs_", n, "x", p, "_c", classes), x = scale(x), labels = labels)
}

drop_self <- function(knn, k) {
  idx <- knn$indices
  dst <- knn$distances
  if (ncol(idx) >= k + 1L) {
    self <- row(idx) == idx
    first_self <- all(self[, 1L])
    if (isTRUE(first_self)) {
      idx <- idx[, -1L, drop = FALSE]
      dst <- dst[, -1L, drop = FALSE]
    }
  }
  list(indices = idx[, seq_len(k), drop = FALSE], distances = dst[, seq_len(k), drop = FALSE])
}

sample_keep <- function(labels, n, sample_size, seed) {
  sample_size <- min(as.integer(sample_size), n)
  if (sample_size >= n) return(seq_len(n))
  set.seed(seed)
  if (is.null(labels)) return(sort(sample.int(n, sample_size)))
  labels <- factor(labels)
  counts <- table(labels)
  per <- pmax(1L, floor(sample_size * as.numeric(counts) / n))
  names(per) <- names(counts)
  keep <- integer(0L)
  for (level in levels(labels)) {
    rows <- which(labels == level)
    take <- min(length(rows), per[[level]])
    keep <- c(keep, sample(rows, take))
  }
  if (length(keep) < sample_size) {
    keep <- c(keep, sample(setdiff(seq_len(n), keep), sample_size - length(keep)))
  }
  sort(unique(keep))[seq_len(min(sample_size, length(unique(keep))))]
}

score_layout <- function(layout, idx, labels, keep, preserve_k) {
  labels_int <- if (is.null(labels)) integer(0L) else as.integer(factor(labels))
  n_levels <- if (is.null(labels)) 0L else length(levels(factor(labels)))
  structure <- fastEmbedR:::knn_structure_score_cpp(
    as.matrix(layout),
    as.matrix(idx),
    as.integer(keep),
    as.integer(preserve_k),
    labels_int,
    as.integer(n_levels)
  )
  sil <- if (is.null(labels) || n_levels < 2L) {
    NA_real_
  } else {
    fastEmbedR:::silhouette_score_cpp(as.matrix(layout[keep, , drop = FALSE]), as.integer(factor(labels[keep])))
  }
  c(
    trustworthiness = unname(structure[["local_trustworthiness"]]),
    knn_preservation = unname(structure[["knn_preservation"]]),
    continuity = unname(structure[["local_continuity"]]),
    label_knn_accuracy = unname(structure[["embedding_knn_accuracy"]]),
    silhouette = sil
  )
}

run_measured <- function(expr) {
  gc()
  before <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - before)
}

quality_score <- function(row) {
  vals <- as.numeric(c(row$trustworthiness, row$knn_preservation, row$continuity, row$label_knn_accuracy, row$silhouette))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) NA_real_ else mean(vals)
}

run_dataset_grid <- function(dataset, grid, max_k, threads, seed, score_n) {
  message("KNN: ", dataset$name, " max_k=", max_k)
  knn_time <- system.time({
    knn_full <- fastEmbedR::nn(dataset$x, dataset$x, k = max_k + 1L, backend = "cpu")
  })[["elapsed"]]
  rows <- list()
  row_id <- 0L

  for (g in seq_len(nrow(grid))) {
    k <- grid$k[[g]]
    n_epochs <- grid$n_epochs[[g]]
    min_dist <- grid$min_dist[[g]]
    neg <- grid$negative_sample_rate[[g]]
    spectral <- grid$spectral_n_iter[[g]]
    knn <- drop_self(knn_full, k)
    idx <- knn$indices
    dst <- knn$distances
    preserve_k <- min(15L, k)
    keep <- sample_keep(dataset$labels, nrow(dataset$x), score_n, seed + k + n_epochs)

    message("  k=", k, " epochs=", n_epochs, " min_dist=", min_dist,
            " neg=", neg, " spectral=", spectral)

    fast <- tryCatch({
      run_measured(fastEmbedR:::fast_knn_umap_cpp(
        as.matrix(idx),
        as.matrix(dst),
        2L,
        as.integer(n_epochs),
        as.numeric(min_dist),
        as.integer(neg),
        1.0,
        1.0,
        as.integer(spectral),
        as.integer(threads),
        as.integer(seed),
        FALSE
      ))
    }, error = function(e) structure(list(error = conditionMessage(e)), class = "bench_error"))

    if (inherits(fast, "bench_error")) {
      fast_metrics <- rep(NA_real_, 5L)
      names(fast_metrics) <- c("trustworthiness", "knn_preservation", "continuity", "label_knn_accuracy", "silhouette")
      fast_time <- NA_real_
      fast_status <- "failed"
      fast_error <- fast$error
    } else {
      fast_metrics <- score_layout(fast$value, idx, dataset$labels, keep, preserve_k)
      fast_time <- fast$seconds
      fast_status <- "success"
      fast_error <- NA_character_
    }
    row_id <- row_id + 1L
    rows[[row_id]] <- data.frame(
      dataset = dataset$name,
      method = "fastEmbedR_umap_cpp",
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      k = k,
      n_epochs = n_epochs,
      min_dist = min_dist,
      negative_sample_rate = neg,
      spectral_n_iter = spectral,
      seed = seed,
      threads = threads,
      knn_time_sec = as.numeric(knn_time),
      embedding_time_sec = fast_time,
      status = fast_status,
      error_message = fast_error,
      as.data.frame(as.list(fast_metrics)),
      row.names = NULL
    )

    uw <- tryCatch({
      run_measured(uwot::umap(
        X = dataset$x,
        n_neighbors = k,
        n_components = 2L,
        nn_method = list(idx = idx, dist = dst),
        n_epochs = n_epochs,
        init = "spectral",
        min_dist = min_dist,
        metric = "euclidean",
        learning_rate = 1,
        repulsion_strength = 1,
        negative_sample_rate = neg,
        fast_sgd = TRUE,
        n_threads = threads,
        n_sgd_threads = threads,
        ret_model = FALSE,
        verbose = FALSE,
        seed = seed
      ))
    }, error = function(e) structure(list(error = conditionMessage(e)), class = "bench_error"))

    if (inherits(uw, "bench_error")) {
      uw_metrics <- rep(NA_real_, 5L)
      names(uw_metrics) <- c("trustworthiness", "knn_preservation", "continuity", "label_knn_accuracy", "silhouette")
      uw_time <- NA_real_
      uw_status <- "failed"
      uw_error <- uw$error
    } else {
      uw_metrics <- score_layout(uw$value, idx, dataset$labels, keep, preserve_k)
      uw_time <- uw$seconds
      uw_status <- "success"
      uw_error <- NA_character_
    }
    row_id <- row_id + 1L
    rows[[row_id]] <- data.frame(
      dataset = dataset$name,
      method = "uwot_fast_sgd",
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      k = k,
      n_epochs = n_epochs,
      min_dist = min_dist,
      negative_sample_rate = neg,
      spectral_n_iter = spectral,
      seed = seed,
      threads = threads,
      knn_time_sec = as.numeric(knn_time),
      embedding_time_sec = uw_time,
      status = uw_status,
      error_message = uw_error,
      as.data.frame(as.list(uw_metrics)),
      row.names = NULL
    )
  }
  do.call(rbind, rows)
}

main <- function() {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package `uwot` is required for this comparison.", call. = FALSE)
  }

  seed <- as.integer(parse_scalar("seed", "4"))
  sizes <- parse_int_csv("sizes", "12000")
  ks <- parse_int_csv("k", "15,30,50")
  epochs <- parse_int_csv("epochs", "100,200")
  min_dists <- parse_num_csv("min-dist", "0.01,0.1")
  neg_rates <- parse_int_csv("negative-sample-rate", "2,5")
  spectral_iters <- parse_int_csv("spectral-iters", "10,30")
  p <- as.integer(parse_scalar("p", "8"))
  classes <- as.integer(parse_scalar("classes", "8"))
  score_n <- as.integer(parse_scalar("score-n", "800"))
  threads <- as.integer(parse_scalar("threads", as.character(max(1L, parallel::detectCores(logical = FALSE)))))
  results_dir <- parse_scalar("results-dir", file.path("results", "uwot_fast_sgd_grid"))
  datasets_arg <- strsplit(parse_scalar("datasets", "blobs"), ",", fixed = TRUE)[[1L]]

  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  grid <- expand.grid(
    k = ks,
    n_epochs = epochs,
    min_dist = min_dists,
    negative_sample_rate = neg_rates,
    spectral_n_iter = spectral_iters,
    KEEP.OUT.ATTRS = FALSE
  )
  max_k <- max(grid$k)

  out <- list()
  pos <- 0L
  for (n in sizes) {
    for (kind in datasets_arg) {
      dataset <- switch(
        trimws(kind),
        blobs = make_blobs(n, p, classes, seed),
        rare_blobs = make_rare_blobs(n, p, classes, seed),
        stop("Unknown dataset kind: ", kind, call. = FALSE)
      )
      pos <- pos + 1L
      out[[pos]] <- run_dataset_grid(dataset, grid, max_k, threads, seed, score_n)
    }
  }

  results <- do.call(rbind, out)
  results$quality_score <- vapply(seq_len(nrow(results)), function(i) quality_score(results[i, , drop = FALSE]), numeric(1))
  results$total_time_sec <- results$knn_time_sec + results$embedding_time_sec

  split_key <- paste(
    results$dataset, results$k, results$n_epochs, results$min_dist,
    results$negative_sample_rate, results$spectral_n_iter
  )
  results$fastembedr_speedup_vs_uwot <- NA_real_
  results$fastembedr_quality_delta_vs_uwot <- NA_real_
  for (key in unique(split_key)) {
    rows <- which(split_key == key)
    f <- rows[results$method[rows] == "fastEmbedR_umap_cpp"]
    u <- rows[results$method[rows] == "uwot_fast_sgd"]
    if (length(f) == 1L && length(u) == 1L) {
      results$fastembedr_speedup_vs_uwot[f] <- results$embedding_time_sec[u] / results$embedding_time_sec[f]
      results$fastembedr_quality_delta_vs_uwot[f] <- results$quality_score[f] - results$quality_score[u]
      results$fastembedr_speedup_vs_uwot[u] <- results$embedding_time_sec[u] / results$embedding_time_sec[f]
      results$fastembedr_quality_delta_vs_uwot[u] <- results$quality_score[f] - results$quality_score[u]
    }
  }

  fast_rows <- results[results$method == "fastEmbedR_umap_cpp", , drop = FALSE]
  pass <- subset(fast_rows, fastembedr_speedup_vs_uwot > 1 & fastembedr_quality_delta_vs_uwot >= 0)
  best <- fast_rows[order(
    fast_rows$dataset,
    -as.numeric(fast_rows$fastembedr_speedup_vs_uwot),
    -as.numeric(fast_rows$fastembedr_quality_delta_vs_uwot)
  ), , drop = FALSE]

  stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  result_file <- file.path(results_dir, paste0("uwot_fast_sgd_grid_", stamp, ".csv"))
  latest_file <- file.path(results_dir, "latest_uwot_fast_sgd_grid.csv")
  pass_file <- file.path(results_dir, "latest_fastembedr_wins.csv")
  best_file <- file.path(results_dir, "latest_fastembedr_ranked.csv")
  utils::write.csv(results, result_file, row.names = FALSE)
  utils::write.csv(results, latest_file, row.names = FALSE)
  utils::write.csv(pass, pass_file, row.names = FALSE)
  utils::write.csv(best, best_file, row.names = FALSE)

  print(best[, intersect(c(
    "dataset", "k", "n_epochs", "min_dist", "negative_sample_rate", "spectral_n_iter",
    "embedding_time_sec", "quality_score", "fastembedr_speedup_vs_uwot",
    "fastembedr_quality_delta_vs_uwot", "trustworthiness", "knn_preservation",
    "label_knn_accuracy", "silhouette"
  ), names(best))], row.names = FALSE)
  cat("\nResults:\n", normalizePath(latest_file, mustWork = FALSE), "\n", sep = "")
  cat("FastEmbedR strict wins:\n", normalizePath(pass_file, mustWork = FALSE), "\n", sep = "")
}

main()
