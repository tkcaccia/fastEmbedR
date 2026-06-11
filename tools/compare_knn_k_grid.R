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

knn_k_grid <- function(n) {
  requested <- c(10L, 15L, 30L, 50L, 90L, 120L)
  unique(requested[requested < n])
}

tsne_perplexity_for_k <- function(k, n) {
  max_valid <- floor((n - 1L) / 3L)
  max(2L, min(30L, max_valid, floor(k / 3L)))
}

metric_k <- function(n) {
  out <- c(15L, 30L, 50L)
  out[out < n]
}

evaluate_row <- function(dataset,
                         objective,
                         method,
                         layout,
                         eval_reference,
                         seed,
                         knn_k,
                         knn_time,
                         embedding_time,
                         params) {
  k_eval <- metric_k(nrow(dataset$x))
  metrics <- fastEmbedR::evaluate_embedding(
    dataset$x,
    layout,
    labels = dataset$labels,
    k = k_eval,
    reference_nn = eval_reference,
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
    objective = objective,
    method = method,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    knn_k_without_self = as.integer(knn_k),
    knn_time_sec = as.numeric(knn_time),
    embedding_time_sec = as.numeric(embedding_time),
    total_time_sec = as.numeric(knn_time) + as.numeric(embedding_time),
    perplexity = params$perplexity,
    max_iter = params$max_iter,
    n_epochs = params$n_epochs,
    status = "success",
    error_message = NA_character_,
    trustworthiness = metrics$trustworthiness,
    knn_preservation_15 = metrics$knn_preservation_15,
    knn_preservation_30 = metrics$knn_preservation_30,
    knn_preservation_50 = metrics$knn_preservation_50,
    silhouette = metrics$silhouette,
    label_knn_accuracy = metrics$label_knn_accuracy,
    distance_spearman = metrics$distance_spearman,
    stress = metrics$stress,
    stringsAsFactors = FALSE
  )
}

failure_row <- function(dataset,
                        objective,
                        method,
                        knn_k,
                        knn_time,
                        params,
                        err) {
  data.frame(
    dataset = dataset$name,
    objective = objective,
    method = method,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    knn_k_without_self = as.integer(knn_k),
    knn_time_sec = as.numeric(knn_time),
    embedding_time_sec = NA_real_,
    total_time_sec = NA_real_,
    perplexity = params$perplexity,
    max_iter = params$max_iter,
    n_epochs = params$n_epochs,
    status = "failed",
    error_message = conditionMessage(err),
    trustworthiness = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    distance_spearman = NA_real_,
    stress = NA_real_,
    stringsAsFactors = FALSE
  )
}

safe_run <- function(dataset,
                     objective,
                     method,
                     knn_k,
                     knn_time,
                     eval_reference,
                     seed,
                     params,
                     expr) {
  tryCatch({
    gc()
    timing <- system.time({
      layout <- force(expr)
    })[["elapsed"]]
    evaluate_row(
      dataset,
      objective,
      method,
      layout,
      eval_reference,
      seed,
      knn_k,
      knn_time,
      timing,
      params
    )
  }, error = function(e) {
    failure_row(dataset, objective, method, knn_k, knn_time, params, e)
  })
}

run_umap_reference <- function(dataset, index, distance, knn_k, seed, n_epochs) {
  if (!requireNamespace("umap", quietly = TRUE)) {
    stop("The umap package is not installed.", call. = FALSE)
  }
  uknn <- umap::umap.knn(index, distance)
  config <- umap::umap.defaults
  config$knn <- uknn
  config$n_neighbors <- knn_k
  config$n_epochs <- n_epochs
  config$min_dist <- 0.1
  set.seed(seed)
  umap::umap(dataset$x, knn = uknn, config = config)$layout
}

run_one_k <- function(dataset, knn_k, seed, eval_reference) {
  gc()
  knn_time <- system.time({
    knn <- fastEmbedR::nn(dataset$x, k = knn_k + 1L, backend = "cpu")
  })[["elapsed"]]
  index <- knn$indices[, -1L, drop = FALSE]
  distance <- knn$distances[, -1L, drop = FALSE]

  rows <- list()
  umap_cfg <- fastEmbedR:::fast_knn_umap_config(nrow(dataset$x), knn_k, "cpu")
  umap_params <- list(perplexity = NA_real_, max_iter = NA_integer_, n_epochs = umap_cfg$n_epochs)
  rows[[length(rows) + 1L]] <- safe_run(
    dataset,
    "umap",
    "fastEmbedR::fast_knn_umap",
    knn_k,
    knn_time,
    eval_reference,
    seed,
    umap_params,
    fastEmbedR::fast_knn_umap(knn, seed = seed, backend = "cpu")
  )
  rows[[length(rows) + 1L]] <- safe_run(
    dataset,
    "umap",
    "umap::umap_precomputed_knn",
    knn_k,
    knn_time,
    eval_reference,
    seed,
    umap_params,
    run_umap_reference(dataset, index, distance, knn_k, seed, umap_cfg$n_epochs)
  )

  perplexity <- tsne_perplexity_for_k(knn_k, nrow(dataset$x))
  tsne_params <- list(perplexity = perplexity, max_iter = 500L, n_epochs = 500L)
  rows[[length(rows) + 1L]] <- safe_run(
    dataset,
    "tsne",
    "fastEmbedR::knn_tsne",
    knn_k,
    knn_time,
    eval_reference,
    seed,
    tsne_params,
    fastEmbedR::knn_tsne(knn, seed = seed, backend = "cpu")
  )
  rows[[length(rows) + 1L]] <- safe_run(
    dataset,
    "tsne",
    "Rtsne::Rtsne_neighbors",
    knn_k,
    knn_time,
    eval_reference,
    seed,
    tsne_params,
    {
      if (!requireNamespace("Rtsne", quietly = TRUE)) {
        stop("The Rtsne package is not installed.", call. = FALSE)
      }
      set.seed(seed)
      Rtsne::Rtsne_neighbors(
        index,
        distance,
        dims = 2L,
        perplexity = perplexity,
        max_iter = 500L,
        theta = 0.5,
        eta = 200,
        num_threads = 1L,
        verbose = FALSE
      )$Y
    }
  )

  do.call(rbind, rows)
}

run_one_dataset <- function(dataset, seed = 4L) {
  k_values <- knn_k_grid(nrow(dataset$x))
  max_eval_k <- max(metric_k(nrow(dataset$x)))
  eval_reference <- fastEmbedR::nn(dataset$x, k = max_eval_k + 1L, backend = "cpu")
  out <- do.call(rbind, lapply(k_values, function(k) {
    message("Dataset ", dataset$name, ": KNN k = ", k)
    run_one_k(dataset, k, seed, eval_reference)
  }))
  out
}

rank_best_rows <- function(results) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok) == 0L) return(ok)
  split_key <- paste(ok$dataset, ok$objective, ok$method, sep = "\r")
  do.call(rbind, lapply(split(ok, split_key), function(x) {
    x <- x[order(
      -x$trustworthiness,
      -x$label_knn_accuracy,
      x$total_time_sec,
      x$knn_k_without_self
    ), , drop = FALSE]
    x[1L, , drop = FALSE]
  }))
}

datasets <- list(
  list(name = "iris", x = standardize_matrix(iris[, 1L:4L]), labels = iris$Species),
  make_synthetic_dataset()
)
digits <- load_digits_dataset()
if (!is.null(digits)) datasets[[length(datasets) + 1L]] <- digits

results <- do.call(rbind, lapply(datasets, run_one_dataset))
best <- rank_best_rows(results)

out_dir <- file.path("results", "performance")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
results_file <- file.path(out_dir, "latest_knn_k_grid.csv")
best_file <- file.path(out_dir, "latest_knn_k_grid_best.csv")
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(best, best_file, row.names = FALSE)

print(results[, c(
  "dataset",
  "objective",
  "method",
  "knn_k_without_self",
  "perplexity",
  "embedding_time_sec",
  "total_time_sec",
  "trustworthiness",
  "knn_preservation_15",
  "silhouette",
  "label_knn_accuracy",
  "status"
)], row.names = FALSE)

cat("\nBest rows by dataset/objective/method:\n")
print(best[, c(
  "dataset",
  "objective",
  "method",
  "knn_k_without_self",
  "perplexity",
  "total_time_sec",
  "trustworthiness",
  "silhouette",
  "label_knn_accuracy"
)], row.names = FALSE)

cat("\nSaved: ", normalizePath(results_file), "\n", sep = "")
cat("Saved: ", normalizePath(best_file), "\n", sep = "")
