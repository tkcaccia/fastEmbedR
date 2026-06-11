#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

standardize_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  sds <- apply(x, 2L, stats::sd)
  keep <- is.finite(sds) & sds > 0
  x <- x[, keep, drop = FALSE]
  x <- scale(x)
  storage.mode(x) <- "double"
  x
}

make_synthetic <- function(n = 900L, p = 20L, classes = 6L, seed = 11L) {
  set.seed(seed)
  labels <- factor(rep(seq_len(classes), length.out = n))
  centers <- matrix(rnorm(classes * p, sd = 1.5), classes, p)
  x <- matrix(rnorm(n * p, sd = 0.7), n, p) + centers[as.integer(labels), , drop = FALSE]
  list(name = paste0("synthetic_", n), x = standardize_matrix(x), labels = labels)
}

load_digits <- function(max_n = 1000L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sklearn <- reticulate::import("sklearn.datasets")
    digits <- sklearn$load_digits()
    n <- min(max_n, nrow(digits$data))
    keep <- seq_len(n)
    list(
      name = paste0("sklearn_digits_", n),
      x = standardize_matrix(digits$data[keep, , drop = FALSE]),
      labels = factor(as.integer(digits$target[keep]))
    )
  }, error = function(e) NULL)
}

as_layout <- function(fit) {
  if (inherits(fit, "fastEmbedR_embedding")) return(as.matrix(fit$layout))
  as.matrix(fit)
}

safe_colors <- function(labels, alpha = 0.55) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), if (nlevels(labels) <= 12L) "Dark 3" else "Spectral")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

timed_fit <- function(expr) {
  invisible(gc())
  value <- NULL
  wall <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, wall_sec = as.numeric(wall))
}

score_embedding <- function(dataset, layout, knn, method, seed) {
  fastEmbedR::evaluate_embedding(
    dataset$x,
    layout,
    labels = dataset$labels,
    k = c(15L, 30L, 50L),
    primary_k = min(30L, nrow(dataset$x) - 1L),
    reference_nn = knn,
    sample_size_for_global_metrics = min(1000L, nrow(dataset$x)),
    sample_size_for_local_metrics = min(1000L, nrow(dataset$x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = dataset$name
  )
}

run_method <- function(dataset, knn, method, k, perplexity, seed, n_threads) {
  early_iter <- 100L
  normal_iter <- 150L
  max_iter <- early_iter + normal_iter
  out <- tryCatch({
    timed <- switch(
      method,
      umap = timed_fit(fastEmbedR::umap(
        dataset$x,
        labels = dataset$labels,
        n_neighbors = k,
        standardize = FALSE,
        pca_dims = NULL,
        nn = knn,
        seed = seed,
        backend = "cpu",
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = FALSE,
        verbose = FALSE
      )),
      tsne = timed_fit(fastEmbedR::tsne(
        dataset$x,
        labels = dataset$labels,
        n_neighbors = k,
        perplexity = perplexity,
        standardize = FALSE,
        pca_dims = NULL,
        nn = knn,
        seed = seed,
        backend = "cpu",
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = FALSE,
        verbose = FALSE,
        max_iter = max_iter,
        stop_lying_iter = early_iter,
        mom_switch_iter = early_iter,
        eta = 200,
        exaggeration_factor = 12,
        negative_gradient_method = "fft",
        n_threads = n_threads
      )),
      opentsne = timed_fit(fastEmbedR::opentsne(
        dataset$x,
        labels = dataset$labels,
        n_neighbors = k,
        perplexity = perplexity,
        standardize = FALSE,
        pca_dims = NULL,
        nn = knn,
        seed = seed,
        backend = "cpu",
        silhouette_sample = NULL,
        preserve_sample = NULL,
        keep_knn = FALSE,
        verbose = FALSE,
        early_exaggeration_iter = early_iter,
        n_iter = normal_iter,
        learning_rate = "auto",
        early_exaggeration = "auto",
        negative_gradient_method = "fft",
        n_threads = n_threads
      )),
      stop("Unknown method: ", method, call. = FALSE)
    )
    fit <- timed$value
    layout <- as_layout(fit)
    metrics <- score_embedding(dataset, layout, knn, method, seed)
    timing <- if (inherits(fit, "fastEmbedR_embedding")) fit$timings else NULL
    preprocess_sec <- if (!is.null(timing) && "preprocess" %in% rownames(timing)) timing["preprocess", "elapsed"] else NA_real_
    knn_sec <- if (!is.null(timing) && "knn" %in% rownames(timing)) timing["knn", "elapsed"] else NA_real_
    embedding_sec <- if (!is.null(timing) && "embedding" %in% rownames(timing)) timing["embedding", "elapsed"] else NA_real_
    total_sec <- if (!is.null(timing)) sum(timing[, "elapsed"]) else timed$wall_sec
    list(
      row = data.frame(
        dataset = dataset$name,
        method = method,
        status = "success",
        error_message = NA_character_,
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        k = k,
        perplexity = perplexity,
        preprocess_sec = as.numeric(preprocess_sec),
        knn_sec = as.numeric(knn_sec),
        embedding_sec = as.numeric(embedding_sec),
        total_timed_sec = as.numeric(total_sec),
        wall_sec = timed$wall_sec,
        trustworthiness = metrics$trustworthiness,
        knn_preservation_15 = metrics$knn_preservation_15,
        knn_preservation_30 = metrics$knn_preservation_30,
        knn_preservation_50 = metrics$knn_preservation_50,
        distance_spearman = metrics$distance_spearman,
        silhouette = metrics$silhouette,
        label_knn_accuracy = metrics$label_knn_accuracy,
        stringsAsFactors = FALSE
      ),
      layout = layout
    )
  }, error = function(e) {
    list(
      row = data.frame(
        dataset = dataset$name,
        method = method,
        status = "failed",
        error_message = conditionMessage(e),
        n = nrow(dataset$x),
        p = ncol(dataset$x),
        k = k,
        perplexity = perplexity,
        preprocess_sec = NA_real_,
        knn_sec = NA_real_,
        embedding_sec = NA_real_,
        total_timed_sec = NA_real_,
        wall_sec = NA_real_,
        trustworthiness = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        distance_spearman = NA_real_,
        silhouette = NA_real_,
        label_knn_accuracy = NA_real_,
        stringsAsFactors = FALSE
      ),
      layout = NULL
    )
  })
  out
}

plot_dataset <- function(dataset, layouts, rows, out_dir) {
  png_path <- file.path(out_dir, paste0(dataset$name, "_umap_tsne_opentsne.png"))
  grDevices::png(png_path, width = 1500, height = 500, res = 130)
  old <- graphics::par(mfrow = c(1L, 3L), mar = c(3.2, 3.2, 3.2, 1.0))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- safe_colors(dataset$labels)
  for (method in c("umap", "tsne", "opentsne")) {
    layout <- layouts[[method]]
    row <- rows[rows$method == method, , drop = FALSE]
    if (is.null(layout)) {
      graphics::plot.new()
      graphics::title(main = paste(dataset$name, method, "failed"))
      next
    }
    graphics::plot(
      layout[, 1L], layout[, 2L],
      pch = 16, cex = 0.45, col = cols,
      xlab = "Dim 1", ylab = "Dim 2",
      main = sprintf("%s | %s\n%.3fs trust %.3f acc %.3f",
                     dataset$name, method, row$embedding_sec,
                     row$trustworthiness, row$label_knn_accuracy)
    )
  }
  png_path
}

run_dataset <- function(dataset, out_dir, seed = 4L, n_threads = 4L) {
  k <- min(50L, nrow(dataset$x) - 1L)
  perplexity <- min(10L, floor((nrow(dataset$x) - 1L) / 3L), floor(k / 3L))
  message("Dataset: ", dataset$name, " n=", nrow(dataset$x), " p=", ncol(dataset$x),
          " k=", k, " perplexity=", perplexity)
  knn_time <- timed_fit(fastEmbedR::nn(dataset$x, k = k + 1L, backend = "cpu", n_threads = n_threads))
  knn <- knn_time$value
  results <- lapply(c("umap", "tsne", "opentsne"), function(method) {
    run_method(dataset, knn, method, k, perplexity, seed, n_threads)
  })
  rows <- do.call(rbind, lapply(results, `[[`, "row"))
  rows$shared_knn_build_sec <- knn_time$wall_sec
  rows$shared_knn_backend <- attr(knn, "backend")
  layouts <- stats::setNames(lapply(results, `[[`, "layout"), c("umap", "tsne", "opentsne"))
  plot_path <- plot_dataset(dataset, layouts, rows, out_dir)
  rows$plot_path <- plot_path
  rows
}

out_dir <- file.path("results", "umap_tsne_opentsne_benchmark")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- list(
  list(name = "iris", x = standardize_matrix(iris[, 1:4]), labels = iris$Species),
  make_synthetic()
)
digits <- load_digits()
if (!is.null(digits)) datasets[[length(datasets) + 1L]] <- digits

rows <- do.call(rbind, lapply(datasets, run_dataset, out_dir = out_dir))
csv_path <- file.path(out_dir, "latest_umap_tsne_opentsne_benchmark.csv")
write.csv(rows, csv_path, row.names = FALSE)

print(rows[, c(
  "dataset", "method", "status", "n", "p", "shared_knn_build_sec",
  "knn_sec", "embedding_sec", "total_timed_sec", "wall_sec",
  "trustworthiness", "knn_preservation_30", "silhouette",
  "label_knn_accuracy", "plot_path"
)], row.names = FALSE)
cat("\nSaved CSV:", normalizePath(csv_path), "\n")
cat("Saved plots:\n")
cat(paste(unique(normalizePath(rows$plot_path)), collapse = "\n"), "\n")
