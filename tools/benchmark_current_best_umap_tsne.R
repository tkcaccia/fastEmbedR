#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all(".", quiet = TRUE)
  } else {
    library(fastEmbedR)
  }
})

source_large_helpers <- function() {
  path <- file.path("tools", "benchmark_large_best_methods.R")
  lines <- readLines(path, warn = FALSE)
  main_start <- grep("^datasets_arg <-", lines)[1L]
  if (length(main_start) != 1L || is.na(main_start)) {
    stop("Could not find helper boundary in ", path, call. = FALSE)
  }
  eval(parse(text = paste(lines[seq_len(main_start - 1L)], collapse = "\n")), envir = .GlobalEnv)
}

source_large_helpers()

arg_value <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else {
    Sys.getenv(paste0("FASTEMBEDR_CURRENT_BENCH_", toupper(gsub("-", "_", name))), default)
  }
}

arg_csv <- function(name, default) {
  x <- trimws(strsplit(arg_value(name, default), ",", fixed = TRUE)[[1L]])
  x[nzchar(x)]
}

arg_int <- function(name, default) as.integer(as.numeric(arg_value(name, as.character(default))))
arg_num <- function(name, default) as.numeric(arg_value(name, as.character(default)))

safe_id <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)

bind_rows <- function(rows) {
  rows <- rows[vapply(rows, is.data.frame, logical(1))]
  if (!length(rows)) return(data.frame())
  cols <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(x) {
    missing <- setdiff(cols, names(x))
    for (name in missing) x[[name]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, rows)
}

cache_dataset <- function(name, cache_dir, max_n, pca_dims, seed) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  key <- paste(safe_id(name), "max", if (is.finite(max_n)) as.integer(max_n) else "all",
               "pca", as.integer(pca_dims), "seed", as.integer(seed), sep = "_")
  path <- file.path(cache_dir, paste0(key, ".rds"))
  if (file.exists(path)) {
    out <- readRDS(path)
    out$preprocess_cache_hit <- TRUE
    return(out)
  }
  raw <- load_named_dataset(name, cache_dir, min_n = 1L, max_n = max_n, seed = seed)
  if (is.null(raw)) return(NULL)
  out <- prepare_dataset(raw, pca_dims = pca_dims, seed = seed, preprocess_backend = "cpu")
  out$preprocess_cache_hit <- FALSE
  saveRDS(out, path, version = 2)
  out
}

label_cols <- function(labels, alpha = 0.55) {
  if (is.null(labels)) return(grDevices::adjustcolor("black", alpha.f = alpha))
  labels <- factor(labels)
  pal <- if (nlevels(labels) <= 12L) {
    grDevices::hcl.colors(nlevels(labels), "Dark 3")
  } else {
    grDevices::hcl.colors(nlevels(labels), "Spectral")
  }
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

measure_expr <- function(expr) {
  invisible(gc())
  rss_before <- current_rss_mb()
  value <- NULL
  elapsed <- system.time({ value <- force(expr) })[["elapsed"]]
  rss_after <- current_rss_mb()
  list(
    value = value,
    elapsed = as.numeric(elapsed),
    rss_delta_mb = if (is.finite(rss_before) && is.finite(rss_after)) rss_after - rss_before else NA_real_
  )
}

coerce_layout2 <- function(layout, n) {
  if (inherits(layout, "fastEmbedR_embedding")) layout <- layout$layout
  if (is.list(layout) && !is.null(layout$layout)) layout <- layout$layout
  if (is.list(layout) && !is.null(layout$Y)) layout <- layout$Y
  layout <- as.matrix(layout)
  storage.mode(layout) <- "double"
  if (nrow(layout) != n && ncol(layout) == n) layout <- t(layout)
  if (nrow(layout) != n || ncol(layout) < 2L) {
    stop("Method did not return an n x 2 layout.", call. = FALSE)
  }
  layout[, 1:2, drop = FALSE]
}

score_layout_current <- function(dataset, layout, method, seed, sample_size) {
  q <- stratified_sample(dataset$labels, nrow(dataset$x), sample_size, seed + 91L)
  metrics <- fastEmbedR::evaluate_embedding(
    dataset$x[q, , drop = FALSE],
    layout[q, , drop = FALSE],
    labels = if (is.null(dataset$labels)) NULL else dataset$labels[q],
    k = c(15L, 30L, 50L),
    sample_size_for_global_metrics = min(2000L, length(q)),
    sample_size_for_local_metrics = length(q),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = paste0(dataset$name, "_quality")
  )
  metrics$quality_sample_n <- length(q)
  metrics
}

save_layout_current <- function(layout, dataset, method, seed, out_dir) {
  dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, "layouts", paste0(safe_id(paste(dataset$name, method, seed, sep = "_")), ".rds"))
  saveRDS(layout, path, version = 2)
  path
}

row_from_run <- function(dataset, method, family, package, seed, k, perplexity,
                         n_epochs, max_iter, preprocess_sec, nn_sec, embedding_sec,
                         total_sec, rss_delta_mb, layout, status, error_message,
                         backend_used, knn_backend, layout_path, quality_sample_size) {
  metrics <- if (identical(status, "success")) {
    score_layout_current(dataset, layout, method, seed, quality_sample_size)
  } else {
    data.frame(
      trustworthiness = NA_real_, continuity = NA_real_,
      knn_preservation_15 = NA_real_, knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_, distance_spearman = NA_real_,
      distance_pearson = NA_real_, stress = NA_real_, silhouette = NA_real_,
      label_knn_accuracy = NA_real_, ari = NA_real_, nmi = NA_real_,
      rare_class_recall = NA_real_, quality_sample_n = NA_integer_
    )
  }
  data.frame(
    dataset = dataset$name,
    family = family,
    method = method,
    package = package,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    raw_p = dataset$raw_p,
    seed = seed,
    k = k,
    perplexity = perplexity,
    n_epochs = n_epochs,
    max_iter = max_iter,
    preprocess_sec = preprocess_sec,
    nn_sec = nn_sec,
    embedding_sec = embedding_sec,
    total_sec = total_sec,
    rss_delta_mb = rss_delta_mb,
    backend_used = backend_used,
    knn_backend = knn_backend,
    status = status,
    error_message = error_message,
    layout_path = layout_path,
    metrics,
    stringsAsFactors = FALSE
  )
}

run_safe_layout <- function(dataset, method, family, package, seed, k, perplexity,
                            n_epochs, max_iter, preprocess_sec, nn_sec,
                            backend_used, knn_backend, out_dir, quality_sample_size, expr) {
  tryCatch({
    measured <- measure_expr(expr)
    layout <- coerce_layout2(measured$value, nrow(dataset$x))
    layout_path <- save_layout_current(layout, dataset, method, seed, out_dir)
    nn_for_total <- if (is.finite(nn_sec)) nn_sec else 0
    row <- row_from_run(
      dataset, method, family, package, seed, k, perplexity, n_epochs, max_iter,
      preprocess_sec, nn_sec, measured$elapsed, preprocess_sec + nn_for_total + measured$elapsed,
      measured$rss_delta_mb, layout, "success", NA_character_, backend_used,
      knn_backend, layout_path, quality_sample_size
    )
    list(row = row, layout = layout)
  }, error = function(e) {
    row <- row_from_run(
      dataset, method, family, package, seed, k, perplexity, n_epochs, max_iter,
      preprocess_sec, nn_sec, NA_real_, NA_real_, NA_real_, NULL, "failed",
      conditionMessage(e), backend_used, knn_backend, NA_character_, quality_sample_size
    )
    list(row = row, layout = NULL)
  })
}

run_umap_dataset <- function(dataset, seed, k, n_epochs, quality_sample_size,
                             n_threads, out_dir, include_landmarks) {
  message("UMAP shared KNN: ", dataset$name)
  nn_measured <- measure_expr({
    raw <- fastEmbedR::nn(dataset$x, dataset$x, k = k + 1L, backend = "auto", n_threads = n_threads)
    out <- drop_self_knn(raw, k)
    attr(out, "backend") <- attr(raw, "backend")
    out
  })
  knn <- nn_measured$value
  knn_backend <- attr(knn, "backend")
  if (is.null(knn_backend)) knn_backend <- "auto"

  rows <- list()
  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "fastEmbedR_umap_from_fastEmbedR_nn", "umap", "fastEmbedR",
    seed, k, NA_real_, n_epochs, NA_integer_, dataset$preprocess_time_sec,
    nn_measured$elapsed, "cpu", knn_backend, out_dir, quality_sample_size,
    fastEmbedR::embed_knn(knn, method = "umap", seed = seed, backend = "cpu")
  )$row

  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "uwot_fast_sgd_from_fastEmbedR_nn", "umap", "uwot",
    seed, k, NA_real_, n_epochs, NA_integer_, dataset$preprocess_time_sec,
    nn_measured$elapsed, "cpu", paste0("shared_", knn_backend), out_dir,
    quality_sample_size,
    uwot::umap(
      X = dataset$x,
      n_neighbors = k,
      n_components = 2L,
      nn_method = list(idx = knn$indices, dist = knn$distances),
      n_epochs = n_epochs,
      init = "spectral",
      min_dist = 0.01,
      metric = "euclidean",
      learning_rate = 1,
      negative_sample_rate = 5,
      repulsion_strength = 1,
      fast_sgd = TRUE,
      n_threads = n_threads,
      n_sgd_threads = n_threads,
      ret_model = FALSE,
      verbose = FALSE,
      seed = seed
    )
  )$row

  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "uwot_fast_sgd_end_to_end", "umap", "uwot",
    seed, k, NA_real_, n_epochs, NA_integer_, dataset$preprocess_time_sec,
    NA_real_, "cpu", "uwot_internal", out_dir, quality_sample_size,
    uwot::umap(
      X = dataset$x,
      n_neighbors = k,
      n_components = 2L,
      n_epochs = n_epochs,
      init = "spectral",
      min_dist = 0.01,
      metric = "euclidean",
      learning_rate = 1,
      negative_sample_rate = 5,
      repulsion_strength = 1,
      fast_sgd = TRUE,
      n_threads = n_threads,
      n_sgd_threads = n_threads,
      ret_model = FALSE,
      verbose = FALSE,
      seed = seed
    )
  )$row

  if (isTRUE(include_landmarks)) {
    rows[[length(rows) + 1L]] <- run_safe_layout(
      dataset, "fastEmbedR_umap_landmark50", "umap", "fastEmbedR",
      seed, k, NA_real_, n_epochs, NA_integer_, dataset$preprocess_time_sec,
      NA_real_, "cpu", "internal_landmark", out_dir, quality_sample_size,
      fastEmbedR::umap(
        dataset$x,
        labels = dataset$labels,
        n_neighbors = k,
        standardize = FALSE,
        pca_dims = NULL,
        landmarks = 0.5,
        seed = seed,
        backend = "cpu",
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = FALSE,
        verbose = FALSE
      )
    )$row
  }
  bind_rows(rows)
}

run_tsne_dataset <- function(dataset, seed, k, perplexity, max_iter,
                             quality_sample_size, n_threads, out_dir) {
  message("t-SNE shared KNN: ", dataset$name)
  nn_measured <- measure_expr({
    raw <- fastEmbedR::nn(dataset$x, dataset$x, k = k + 1L, backend = "cpu", n_threads = n_threads)
    out <- drop_self_knn(raw, k)
    attr(out, "backend") <- attr(raw, "backend")
    out
  })
  knn <- nn_measured$value
  knn_backend <- attr(knn, "backend")
  if (is.null(knn_backend)) knn_backend <- "cpu"
  perplexity <- max(2L, min(perplexity, floor(k / 3L), floor((nrow(dataset$x) - 1L) / 3L)))

  rows <- list()
  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "fastEmbedR_tsne_exact_from_knn", "tsne", "fastEmbedR",
    seed, k, perplexity, NA_integer_, max_iter, dataset$preprocess_time_sec,
    nn_measured$elapsed, "cpu", knn_backend, out_dir, quality_sample_size,
    fastEmbedR::embed_knn(
      knn,
      method = "tsne",
      perplexity = perplexity,
      max_iter = max_iter,
      n_threads = n_threads,
      seed = seed,
      backend = "cpu"
    )
  )$row

  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "fastEmbedR_infotsne_from_knn", "tsne", "fastEmbedR",
    seed, k, perplexity, NA_integer_, max_iter, dataset$preprocess_time_sec,
    nn_measured$elapsed, "cpu", knn_backend, out_dir, quality_sample_size,
    fastEmbedR::embed_knn(
      knn,
      method = "infotsne",
      perplexity = perplexity,
      max_iter = max_iter,
      n_threads = n_threads,
      seed = seed,
      backend = "cpu"
    )
  )$row

  rows[[length(rows) + 1L]] <- run_safe_layout(
    dataset, "Rtsne_neighbors", "tsne", "Rtsne",
    seed, k, perplexity, NA_integer_, max_iter, dataset$preprocess_time_sec,
    nn_measured$elapsed, "cpu", paste0("shared_", knn_backend), out_dir,
    quality_sample_size,
    Rtsne::Rtsne_neighbors(
      knn$indices,
      knn$distances,
      dims = 2L,
      perplexity = perplexity,
      max_iter = max_iter,
      theta = 0.5,
      eta = 200,
      stop_lying_iter = min(250L, max_iter),
      mom_switch_iter = min(250L, max_iter),
      momentum = 0.5,
      final_momentum = 0.8,
      exaggeration_factor = 12,
      num_threads = n_threads,
      verbose = FALSE
    )$Y
  )$row
  bind_rows(rows)
}

plot_dataset_panels <- function(results, out_dir) {
  ok <- results[results$status == "success" & file.exists(results$layout_path), , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  for (dataset_name in unique(ok$dataset)) {
    rows <- ok[ok$dataset == dataset_name, , drop = FALSE]
    png_path <- file.path(out_dir, paste0(safe_id(dataset_name), "_embedding_panel.png"))
    n_panels <- nrow(rows)
    png(png_path, width = 420 * min(4L, n_panels), height = 380 * ceiling(n_panels / 4), res = 110)
    old <- par(no.readonly = TRUE)
    on.exit(par(old), add = TRUE)
    par(mfrow = c(ceiling(n_panels / 4), min(4L, n_panels)), mar = c(2.6, 2.6, 3.0, 0.8))
    for (i in seq_len(nrow(rows))) {
      layout <- readRDS(rows$layout_path[[i]])
      labels <- attr(layout, "labels_for_plot")
      if (is.null(labels)) {
        labels <- readRDS(file.path(out_dir, "datasets", paste0(safe_id(dataset_name), "_labels.rds")))
      }
      plot(layout[, 1], layout[, 2], pch = 16, cex = 0.18,
           col = label_cols(labels, alpha = 0.55),
           xlab = "", ylab = "",
           main = paste0(rows$method[[i]], "\n",
                         "sec=", round(rows$total_sec[[i]], 1),
                         " trust=", round(rows$trustworthiness[[i]], 3)))
    }
    dev.off()
  }
}

plot_speed_quality <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  png(file.path(out_dir, "speed_quality_summary.png"), width = 1200, height = 850, res = 130)
  par(mar = c(5, 5, 4, 9), xpd = TRUE)
  cols <- as.integer(factor(ok$method))
  plot(ok$total_sec, ok$trustworthiness, log = "x", pch = 19, col = cols,
       xlab = "Total timed seconds (log scale)",
       ylab = "Trustworthiness",
       main = "fastEmbedR vs uwot/Rtsne: speed and quality")
  legend("right", inset = c(-0.33, 0), legend = levels(factor(ok$method)),
         col = seq_along(levels(factor(ok$method))), pch = 19, cex = 0.75)
  dev.off()
}

summarize_best <- function(results) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(ok)
  groups <- split(ok, paste(ok$dataset, ok$family, ok$method, sep = "\r"))
  out <- bind_rows(lapply(groups, function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      family = x$family[[1L]],
      method = x$method[[1L]],
      package = x$package[[1L]],
      n = x$n[[1L]],
      p = x$p[[1L]],
      runs = nrow(x),
      mean_preprocess_sec = mean(x$preprocess_sec, na.rm = TRUE),
      mean_nn_sec = if (all(is.na(x$nn_sec))) NA_real_ else mean(x$nn_sec, na.rm = TRUE),
      mean_embedding_sec = mean(x$embedding_sec, na.rm = TRUE),
      mean_timed_no_preprocess_sec = mean(x$total_sec - x$preprocess_sec, na.rm = TRUE),
      mean_total_sec = mean(x$total_sec, na.rm = TRUE),
      mean_trustworthiness = mean(x$trustworthiness, na.rm = TRUE),
      mean_knn_preservation_15 = mean(x$knn_preservation_15, na.rm = TRUE),
      mean_knn_preservation_30 = mean(x$knn_preservation_30, na.rm = TRUE),
      mean_knn_preservation_50 = mean(x$knn_preservation_50, na.rm = TRUE),
      mean_silhouette = mean(x$silhouette, na.rm = TRUE),
      mean_label_knn_accuracy = mean(x$label_knn_accuracy, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
  out[order(out$dataset, out$family, out$mean_total_sec), , drop = FALSE]
}

umap_datasets <- arg_csv("umap-datasets", "mnist,fashion_mnist,shuttle")
tsne_datasets <- arg_csv("tsne-datasets", "mnist,fashion_mnist,shuttle")
seed <- arg_int("seed", 6L)
k_umap <- arg_int("k-umap", 50L)
k_tsne <- arg_int("k-tsne", 90L)
perplexity <- arg_int("perplexity", 30L)
n_epochs <- arg_int("n-epochs", 200L)
max_iter <- arg_int("max-iter", 500L)
pca_dims <- arg_int("pca-dims", 50L)
umap_max_n <- arg_num("umap-max-n", Inf)
tsne_max_n <- arg_num("tsne-max-n", 2500L)
quality_sample_size <- arg_int("quality-sample-size", 5000L)
n_threads <- arg_int("n-threads", 4L)
include_landmarks <- tolower(arg_value("include-landmarks", "true")) %in% c("1", "true", "yes")
out_dir <- arg_value(
  "out-dir",
  file.path(path.expand("~/Documents/fastEmbedR-results"),
            paste0("current_best_umap_tsne_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
cache_dir <- file.path(out_dir, "cache")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "datasets"), recursive = TRUE, showWarnings = FALSE)

message("Output: ", normalizePath(out_dir, mustWork = FALSE))

rows <- list()
for (name in umap_datasets) {
  dataset <- cache_dataset(name, cache_dir, umap_max_n, pca_dims, seed)
  if (is.null(dataset)) next
  saveRDS(dataset$labels, file.path(out_dir, "datasets", paste0(safe_id(dataset$name), "_labels.rds")))
  rows[[length(rows) + 1L]] <- run_umap_dataset(
    dataset, seed, k_umap, n_epochs, quality_sample_size,
    n_threads, out_dir, include_landmarks
  )
}

for (name in tsne_datasets) {
  dataset <- cache_dataset(name, cache_dir, tsne_max_n, pca_dims, seed)
  if (is.null(dataset)) next
  saveRDS(dataset$labels, file.path(out_dir, "datasets", paste0(safe_id(dataset$name), "_labels.rds")))
  rows[[length(rows) + 1L]] <- run_tsne_dataset(
    dataset, seed, k_tsne, perplexity, max_iter,
    min(quality_sample_size, nrow(dataset$x)), n_threads, out_dir
  )
}

results <- bind_rows(rows)
summary <- summarize_best(results)
utils::write.csv(results, file.path(out_dir, "current_best_results.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(out_dir, "current_best_summary.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
writeLines(capture.output(fastEmbedR::backend_info()), file.path(out_dir, "backend_info.txt"))
plot_speed_quality(results, out_dir)
plot_dataset_panels(results, out_dir)

print(summary[, intersect(c(
  "dataset", "family", "method", "package", "n", "p", "mean_nn_sec",
  "mean_embedding_sec", "mean_total_sec", "mean_trustworthiness",
  "mean_knn_preservation_15", "mean_silhouette", "mean_label_knn_accuracy"
), names(summary))], row.names = FALSE)

cat("\nSaved:\n")
cat("  ", normalizePath(file.path(out_dir, "current_best_results.csv"), mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(file.path(out_dir, "current_best_summary.csv"), mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(file.path(out_dir, "speed_quality_summary.png"), mustWork = FALSE), "\n", sep = "")
cat("  ", normalizePath(out_dir, mustWork = FALSE), "\n", sep = "")
