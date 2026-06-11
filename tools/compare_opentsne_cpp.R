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

make_synthetic_dataset <- function(seed = 11L) {
  set.seed(seed)
  n_per_class <- 100L
  p <- 12L
  labels <- factor(rep(seq_len(3L), each = n_per_class))
  centers <- matrix(0, 3L, p)
  centers[2L, 1L:4L] <- 2.0
  centers[3L, 5L:8L] <- 2.0
  x <- matrix(rnorm(length(labels) * p, sd = 0.75), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  list(name = "synthetic_300", x = standardize_matrix(x), labels = labels)
}

load_digits_dataset <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sklearn <- reticulate::import("sklearn.datasets")
    digits <- sklearn$load_digits()
    keep <- seq_len(min(600L, nrow(digits$data)))
    list(
      name = "sklearn_digits_600",
      x = standardize_matrix(digits$data[keep, , drop = FALSE]),
      labels = factor(as.integer(digits$target[keep]))
    )
  }, error = function(e) NULL)
}

metric_row <- function(dataset, method, status, elapsed, layout, knn, seed,
                       parameters, error_message = NA_character_) {
  if (!identical(status, "success")) {
    return(data.frame(
      dataset = dataset$name,
      method = method,
      status = status,
      error_message = error_message,
      n = nrow(dataset$x),
      p = ncol(dataset$x),
      elapsed_sec = NA_real_,
      trustworthiness = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      silhouette = NA_real_,
      label_knn_accuracy = NA_real_,
      parameters = parameters,
      stringsAsFactors = FALSE
    ))
  }

  metrics <- fastEmbedR::evaluate_embedding(
    dataset$x,
    layout,
    labels = dataset$labels,
    k = c(15L, 30L),
    primary_k = min(30L, nrow(dataset$x) - 1L),
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
    status = status,
    error_message = NA_character_,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    elapsed_sec = as.numeric(elapsed),
    trustworthiness = metrics$trustworthiness,
    knn_preservation_15 = metrics$knn_preservation_15,
    knn_preservation_30 = metrics$knn_preservation_30,
    silhouette = metrics$silhouette,
    label_knn_accuracy = metrics$label_knn_accuracy,
    parameters = parameters,
    stringsAsFactors = FALSE
  )
}

run_timed <- function(expr) {
  value <- NULL
  elapsed <- system.time({
    value <- force(expr)
  })[["elapsed"]]
  list(value = value, elapsed = elapsed)
}

run_external_wrapper <- function(dataset, seed, common) {
  if (requireNamespace("ReductionWrappers", quietly = TRUE)) {
    wrapper_fun <- get("openTSNE", envir = asNamespace("ReductionWrappers"))
  } else if (requireNamespace("dim.reduction.wrappers", quietly = TRUE)) {
    wrapper_fun <- get("openTSNE", envir = asNamespace("dim.reduction.wrappers"))
  } else {
    stop("ReductionWrappers/dim.reduction.wrappers is not installed.", call. = FALSE)
  }
  if (!requireNamespace("reticulate", quietly = TRUE) ||
      !reticulate::py_module_available("openTSNE")) {
    stop("Python openTSNE module is not available through reticulate.", call. = FALSE)
  }
  out <- wrapper_fun(
    as.data.frame(dataset$x),
    n_components = 2L,
    perplexity = common$perplexity,
    learning_rate = common$learning_rate,
    early_exaggeration_iter = common$early_exaggeration_iter,
    early_exaggeration = common$early_exaggeration,
    n_iter = common$n_iter,
    exaggeration = NULL,
    theta = common$theta,
    initialization = "random",
    metric = "euclidean",
    initial_momentum = common$initial_momentum,
    final_momentum = common$final_momentum,
    n_jobs = common$n_threads,
    neighbors = "exact",
    negative_gradient_method = "bh",
    random_state = seed
  )
  as.matrix(out[, seq_len(2L), drop = FALSE])
}

run_dataset <- function(dataset, seed = 4L) {
  common <- list(
    k_without_self = min(90L, nrow(dataset$x) - 1L),
    perplexity = min(30L, floor((nrow(dataset$x) - 1L) / 3L), 30L),
    learning_rate = 100,
    early_exaggeration_iter = 100L,
    n_iter = 150L,
    early_exaggeration = 12,
    theta = 0.5,
    initial_momentum = 0.8,
    final_momentum = 0.8,
    n_threads = 4L
  )
  common$perplexity <- min(common$perplexity, floor(common$k_without_self / 3L))
  common$parameters <- paste(
    paste(names(common), unlist(common), sep = "="),
    collapse = ";"
  )

  knn <- fastEmbedR::nn(dataset$x, k = common$k_without_self + 1L,
                        backend = "cpu", n_threads = common$n_threads)
  index <- knn$indices[, -1L, drop = FALSE]
  distance <- knn$distances[, -1L, drop = FALSE]

  rows <- list()

  native_open <- tryCatch({
    timed <- run_timed(fastEmbedR::embed_knn(
      knn,
      method = "opentsne",
      perplexity = common$perplexity,
      learning_rate = common$learning_rate,
      early_exaggeration_iter = common$early_exaggeration_iter,
      n_iter = common$n_iter,
      early_exaggeration = common$early_exaggeration,
      initial_momentum = common$initial_momentum,
      final_momentum = common$final_momentum,
      negative_gradient_method = "bh",
      theta = common$theta,
      n_threads = common$n_threads,
      seed = seed
    ))
    metric_row(dataset, "fastEmbedR::opentsne_cpp_bh", "success",
               timed$elapsed, timed$value, knn, seed, common$parameters)
  }, error = function(e) {
    metric_row(dataset, "fastEmbedR::opentsne_cpp_bh", "failed", NA, NULL,
               knn, seed, common$parameters, conditionMessage(e))
  })
  rows[[length(rows) + 1L]] <- native_open

  native_tsne <- tryCatch({
    timed <- run_timed(fastEmbedR::embed_knn(
      knn,
      method = "tsne",
      perplexity = common$perplexity,
      max_iter = common$early_exaggeration_iter + common$n_iter,
      stop_lying_iter = common$early_exaggeration_iter,
      mom_switch_iter = common$early_exaggeration_iter,
      eta = common$learning_rate,
      momentum = common$initial_momentum,
      final_momentum = common$final_momentum,
      exaggeration_factor = common$early_exaggeration,
      negative_gradient_method = "bh",
      theta = common$theta,
      n_threads = common$n_threads,
      seed = seed
    ))
    metric_row(dataset, "fastEmbedR::tsne_rtsne_style_bh", "success",
               timed$elapsed, timed$value, knn, seed, common$parameters)
  }, error = function(e) {
    metric_row(dataset, "fastEmbedR::tsne_rtsne_style_bh", "failed", NA, NULL,
               knn, seed, common$parameters, conditionMessage(e))
  })
  rows[[length(rows) + 1L]] <- native_tsne

  rtsne <- tryCatch({
    if (!requireNamespace("Rtsne", quietly = TRUE)) {
      stop("Rtsne is not installed.", call. = FALSE)
    }
    timed <- run_timed(Rtsne::Rtsne_neighbors(
      index,
      distance,
      dims = 2L,
      perplexity = common$perplexity,
      max_iter = common$early_exaggeration_iter + common$n_iter,
      theta = common$theta,
      eta = common$learning_rate,
      stop_lying_iter = common$early_exaggeration_iter,
      mom_switch_iter = common$early_exaggeration_iter,
      momentum = common$initial_momentum,
      final_momentum = common$final_momentum,
      exaggeration_factor = common$early_exaggeration,
      num_threads = common$n_threads,
      verbose = FALSE
    )$Y)
    metric_row(dataset, "Rtsne::Rtsne_neighbors", "success",
               timed$elapsed, timed$value, knn, seed, common$parameters)
  }, error = function(e) {
    metric_row(dataset, "Rtsne::Rtsne_neighbors", "failed", NA, NULL, knn,
               seed, common$parameters, conditionMessage(e))
  })
  rows[[length(rows) + 1L]] <- rtsne

  wrapper <- tryCatch({
    timed <- run_timed(run_external_wrapper(dataset, seed, common))
    metric_row(dataset, "dim.reduction.wrappers::openTSNE", "success",
               timed$elapsed, timed$value, knn, seed, common$parameters)
  }, error = function(e) {
    metric_row(dataset, "dim.reduction.wrappers::openTSNE",
               "backend_unavailable", NA, NULL, knn, seed,
               common$parameters, conditionMessage(e))
  })
  rows[[length(rows) + 1L]] <- wrapper

  do.call(rbind, rows)
}

datasets <- list(
  list(name = "iris", x = standardize_matrix(iris[, 1L:4L]), labels = iris$Species),
  make_synthetic_dataset()
)
digits <- load_digits_dataset()
if (!is.null(digits)) datasets[[length(datasets) + 1L]] <- digits

results <- do.call(rbind, lapply(datasets, run_dataset))
out_dir <- file.path("results", "opentsne_cpp_comparison")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
csv_path <- file.path(out_dir, "latest_opentsne_cpp_comparison.csv")
utils::write.csv(results, csv_path, row.names = FALSE)

png_path <- file.path(out_dir, "latest_opentsne_cpp_comparison.png")
success <- results[results$status == "success", , drop = FALSE]
if (nrow(success) > 0L) {
  grDevices::png(png_path, width = 1400, height = 800, res = 140)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(1L, 2L), mar = c(8, 4, 3, 1))
  labels <- paste(success$dataset, success$method, sep = "\n")
  barplot(success$elapsed_sec, names.arg = labels, las = 2,
          ylab = "Embedding seconds", main = "Runtime")
  plot(success$elapsed_sec, success$trustworthiness, pch = 19,
       xlab = "Embedding seconds", ylab = "Trustworthiness",
       main = "Speed vs local quality")
  text(success$elapsed_sec, success$trustworthiness,
       labels = paste(success$dataset, success$method, sep = " "),
       pos = 4, cex = 0.6)
}

print(results[, c(
  "dataset", "method", "status", "elapsed_sec", "trustworthiness",
  "knn_preservation_15", "knn_preservation_30", "label_knn_accuracy",
  "error_message"
)], row.names = FALSE)
cat("\nSaved CSV: ", normalizePath(csv_path, mustWork = FALSE), "\n", sep = "")
if (file.exists(png_path)) {
  cat("Saved plot: ", normalizePath(png_path, mustWork = FALSE), "\n", sep = "")
}
