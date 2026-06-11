#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

if (!requireNamespace("Rtsne", quietly = TRUE)) {
  stop("The Rtsne package is required for this comparison.", call. = FALSE)
}

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

load_digits_dataset <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sklearn <- reticulate::import("sklearn.datasets")
    digits <- sklearn$load_digits()
    list(
      name = "sklearn_digits",
      x = standardize_matrix(digits$data),
      labels = factor(as.integer(digits$target))
    )
  }, error = function(e) NULL)
}

make_synthetic_dataset <- function(seed = 11L) {
  set.seed(seed)
  n_per_class <- 300L
  p <- 20L
  labels <- factor(rep(seq_len(3L), each = n_per_class))
  centers <- matrix(0, 3L, p)
  centers[2L, 1L:6L] <- 2.0
  centers[3L, 7L:12L] <- 2.0
  x <- matrix(rnorm(length(labels) * p, sd = 0.75), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  list(name = "synthetic_900", x = standardize_matrix(x), labels = labels)
}

benchmark_row <- function(dataset, method, layout, elapsed, knn, seed, k_eval, params) {
  metrics <- fastEmbedR::evaluate_embedding(
    dataset$x,
    layout,
    labels = dataset$labels,
    k = k_eval,
    reference_nn = knn,
    sample_size_for_global_metrics = min(500L, nrow(dataset$x)),
    sample_size_for_local_metrics = min(1000L, nrow(dataset$x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = dataset$name
  )
  data.frame(
    dataset = dataset$name,
    method = method,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    input_knn_without_self = params$k_without_self,
    perplexity = params$perplexity,
    max_iter = params$max_iter,
    elapsed_sec = as.numeric(elapsed),
    trustworthiness = metrics$trustworthiness,
    knn_preservation_15 = metrics$knn_preservation_15,
    knn_preservation_30 = metrics$knn_preservation_30,
    knn_preservation_50 = metrics$knn_preservation_50,
    silhouette = metrics$silhouette,
    label_knn_accuracy = metrics$label_knn_accuracy,
    stringsAsFactors = FALSE
  )
}

run_one_dataset <- function(dataset, seed = 4L) {
  k_without_self <- min(90L, nrow(dataset$x) - 1L)
  if (k_without_self < 3L) stop("Dataset is too small for t-SNE neighbor comparison.", call. = FALSE)
  perplexity <- min(30L, floor(k_without_self / 3L))
  k_eval <- c(15L, 30L, 50L)
  k_eval <- k_eval[k_eval < nrow(dataset$x)]

  knn <- fastEmbedR::nn(dataset$x, k = k_without_self + 1L, backend = "cpu")
  index <- knn$indices[, -1L, drop = FALSE]
  distance <- knn$distances[, -1L, drop = FALSE]

  rows <- list()
  gc()
  fast_time <- system.time({
    fast_layout <- fastEmbedR::embed_knn(
      knn,
      method = "tsne",
      seed = seed,
      backend = "cpu"
    )
  })[["elapsed"]]
  rows[[length(rows) + 1L]] <- benchmark_row(
    dataset,
    "fastEmbedR::embed_knn_tsne",
    fast_layout,
    fast_time,
    knn,
    seed,
    k_eval,
    list(k_without_self = k_without_self, perplexity = perplexity, quality = "auto")
  )

  for (max_iter in c(500L, 1000L)) {
    set.seed(seed)
    gc()
    rtsne_time <- system.time({
      rtsne_layout <- Rtsne::Rtsne_neighbors(
        index,
        distance,
        dims = 2L,
        perplexity = perplexity,
	        max_iter = max_iter,
	        theta = 0.5,
	        eta = 200,
	        stop_lying_iter = 250L,
	        mom_switch_iter = 250L,
	        momentum = 0.5,
	        final_momentum = 0.8,
	        exaggeration_factor = 12,
	        num_threads = 1L,
	        verbose = FALSE
	      )$Y
    })[["elapsed"]]
    rows[[length(rows) + 1L]] <- benchmark_row(
      dataset,
      paste0("Rtsne::Rtsne_neighbors_", max_iter),
      rtsne_layout,
      rtsne_time,
      knn,
      seed,
      k_eval,
      list(k_without_self = k_without_self, perplexity = perplexity, max_iter = max_iter)
    )
  }

  do.call(rbind, rows)
}

datasets <- list(
  list(name = "iris", x = standardize_matrix(iris[, 1L:4L]), labels = iris$Species),
  make_synthetic_dataset()
)
digits <- load_digits_dataset()
if (!is.null(digits)) datasets[[length(datasets) + 1L]] <- digits

out <- do.call(rbind, lapply(datasets, run_one_dataset))
out_dir <- file.path("results", "performance")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, "latest_rtsne_neighbors_comparison.csv")
utils::write.csv(out, out_file, row.names = FALSE)

print(out[, c(
  "dataset",
  "method",
  "n",
  "input_knn_without_self",
  "perplexity",
  "max_iter",
  "elapsed_sec",
  "trustworthiness",
  "knn_preservation_15",
  "silhouette",
  "label_knn_accuracy"
)], row.names = FALSE)
cat("\nSaved: ", normalizePath(out_file), "\n", sep = "")
