#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

arg_value <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

arg_int <- function(name, default) as.integer(as.numeric(arg_value(name, as.character(default))))

stratified_rows <- function(labels, n, seed) {
  labels <- factor(labels)
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  levels <- levels(labels)
  base <- floor(n / length(levels))
  remainder <- n - base * length(levels)
  rows <- integer(0L)
  for (lev in levels) {
    idx <- which(labels == lev)
    take <- min(length(idx), base + as.integer(remainder > 0L))
    remainder <- max(0L, remainder - 1L)
    rows <- c(rows, sample(idx, take))
  }
  sort(rows)
}

timed <- function(expr) {
  invisible(gc())
  value <- NULL
  sec <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(sec))
}

as_layout <- function(fit) {
  if (inherits(fit, "fastEmbedR_embedding")) return(as.matrix(fit$layout))
  as.matrix(fit)
}

label_colors <- function(labels, alpha = 0.55) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

score_layout <- function(x, layout, labels, knn, method, seed) {
  fastEmbedR::evaluate_embedding(
    x,
    layout,
    labels = labels,
    k = c(15L, 30L, 50L),
    primary_k = 30L,
    reference_nn = knn,
    sample_size_for_global_metrics = min(5000L, nrow(x)),
    sample_size_for_local_metrics = min(5000L, nrow(x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = "mnist_idx"
  )
}

run_one <- function(method, x, labels, knn, k, perplexity, seed, n_threads) {
  early_iter <- 100L
  normal_iter <- 150L
  max_iter <- early_iter + normal_iter
  tryCatch({
    measured <- switch(
      method,
      umap = timed(fastEmbedR::umap(
        x,
        labels = labels,
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
      tsne = timed(fastEmbedR::tsne(
        x,
        labels = labels,
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
      opentsne = timed(fastEmbedR::opentsne(
        x,
        labels = labels,
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
    fit <- measured$value
    layout <- as_layout(fit)
    metrics <- score_layout(x, layout, labels, knn, method, seed)
    timings <- if (inherits(fit, "fastEmbedR_embedding")) fit$timings else NULL
    row <- data.frame(
      method = method,
      status = "success",
      error_message = NA_character_,
      n = nrow(x),
      p = ncol(x),
      k = k,
      perplexity = perplexity,
      embedding_sec = if (!is.null(timings)) timings["embedding", "elapsed"] else measured$sec,
      function_wall_sec = measured$sec,
      trustworthiness = metrics$trustworthiness,
      knn_preservation_15 = metrics$knn_preservation_15,
      knn_preservation_30 = metrics$knn_preservation_30,
      knn_preservation_50 = metrics$knn_preservation_50,
      distance_spearman = metrics$distance_spearman,
      silhouette = metrics$silhouette,
      label_knn_accuracy = metrics$label_knn_accuracy,
      stringsAsFactors = FALSE
    )
    list(row = row, layout = layout)
  }, error = function(e) {
    list(
      row = data.frame(
        method = method,
        status = "failed",
        error_message = conditionMessage(e),
        n = nrow(x),
        p = ncol(x),
        k = k,
        perplexity = perplexity,
        embedding_sec = NA_real_,
        function_wall_sec = NA_real_,
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
}

plot_layouts <- function(layouts, rows, labels, out_path) {
  grDevices::png(out_path, width = 1500, height = 500, res = 130)
  old <- graphics::par(mfrow = c(1L, 3L), mar = c(3.0, 3.0, 3.5, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- label_colors(labels)
  for (method in c("umap", "tsne", "opentsne")) {
    layout <- layouts[[method]]
    row <- rows[rows$method == method, , drop = FALSE]
    if (is.null(layout)) {
      graphics::plot.new()
      graphics::title(main = paste("MNIST", method, "failed"))
      next
    }
    graphics::plot(
      layout[, 1L], layout[, 2L],
      col = cols, pch = 16, cex = 0.24,
      xlab = "Dim 1", ylab = "Dim 2",
      main = sprintf(
        "MNIST %s\n%.2fs trust %.3f acc %.3f",
        method, row$embedding_sec, row$trustworthiness, row$label_knn_accuracy
      )
    )
  }
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 12000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
knn_backend <- arg_value("knn-backend", "cpu_nndescent")

cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
mnist <- readRDS(cache)
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
labels <- droplevels(mnist$labels[rows])
storage.mode(x) <- "double"

out_dir <- file.path("results", "mnist_umap_tsne_opentsne")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("MNIST benchmark: n=", nrow(x), " p=", ncol(x), " k=", k,
        " knn_backend=", knn_backend, " threads=", n_threads)
knn_time <- timed(fastEmbedR::nn(x, k = k + 1L, backend = knn_backend, n_threads = n_threads))
knn <- knn_time$value
message("KNN: ", round(knn_time$sec, 3), " sec, backend=", attr(knn, "backend"),
        ", exact=", attr(knn, "exact"))

results <- lapply(c("umap", "tsne", "opentsne"), run_one,
                  x = x, labels = labels, knn = knn, k = k,
                  perplexity = min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L)),
                  seed = seed, n_threads = n_threads)
bench <- do.call(rbind, lapply(results, `[[`, "row"))
bench$dataset <- paste0("mnist_idx_", nrow(x), "_pca50")
bench$seed <- seed
bench$shared_knn_sec <- knn_time$sec
bench$shared_knn_backend <- attr(knn, "backend")
bench$shared_knn_exact <- isTRUE(attr(knn, "exact"))
bench$total_embedding_plus_knn_sec <- bench$embedding_sec + knn_time$sec
bench <- bench[, c(
  "dataset", "method", "status", "error_message", "n", "p", "seed", "k",
  "perplexity", "shared_knn_backend", "shared_knn_exact", "shared_knn_sec",
  "embedding_sec", "function_wall_sec", "total_embedding_plus_knn_sec",
  "trustworthiness", "knn_preservation_15", "knn_preservation_30",
  "knn_preservation_50", "distance_spearman", "silhouette",
  "label_knn_accuracy"
)]

layouts <- stats::setNames(lapply(results, `[[`, "layout"), c("umap", "tsne", "opentsne"))
layout_dir <- file.path(out_dir, "layouts")
dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
for (method in names(layouts)) {
  if (!is.null(layouts[[method]])) {
    saveRDS(layouts[[method]], file.path(layout_dir, paste0("mnist_", nrow(x), "_", method, "_seed", seed, ".rds")), version = 2)
  }
}
plot_path <- file.path(out_dir, paste0("mnist_", nrow(x), "_umap_tsne_opentsne_seed", seed, ".png"))
plot_layouts(layouts, bench, labels, plot_path)
bench$plot_path <- plot_path

csv_path <- file.path(out_dir, paste0("mnist_", nrow(x), "_umap_tsne_opentsne_seed", seed, ".csv"))
write.csv(bench, csv_path, row.names = FALSE)
write.csv(bench, file.path(out_dir, "latest_mnist_umap_tsne_opentsne.csv"), row.names = FALSE)

print(bench[, c(
  "dataset", "method", "status", "shared_knn_sec", "embedding_sec",
  "total_embedding_plus_knn_sec", "trustworthiness", "knn_preservation_30",
  "silhouette", "label_knn_accuracy"
)], row.names = FALSE)
cat("\nSaved CSV:", normalizePath(csv_path), "\n")
cat("Saved plot:", normalizePath(plot_path), "\n")
