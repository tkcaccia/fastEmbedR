#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

arg_value <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

arg_int <- function(name, default) {
  as.integer(as.numeric(arg_value(name, as.character(default))))
}

stratified_rows <- function(labels, n, seed) {
  labels <- factor(labels)
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  levels <- levels(labels)
  base <- floor(n / length(levels))
  remainder <- n - base * length(levels)
  rows <- integer(0L)
  for (level in levels) {
    candidates <- which(labels == level)
    take <- min(length(candidates), base + as.integer(remainder > 0L))
    remainder <- max(0L, remainder - 1L)
    rows <- c(rows, sample(candidates, take))
  }
  sort(rows)
}

timed <- function(expr) {
  invisible(gc())
  value <- NULL
  sec <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(sec))
}

coerce_layout <- function(x, n) {
  if (inherits(x, "fastEmbedR_embedding")) x <- x$layout
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (is.list(x) && !is.null(x$embedding)) x <- x$embedding
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) {
    stop("No n x 2 layout returned.", call. = FALSE)
  }
  x[, 1:2, drop = FALSE]
}

score_layout <- function(x, layout, labels, knn, method, seed) {
  fastEmbedR::evaluate_embedding(
    x,
    layout,
    labels = labels,
    k = c(15L, 30L, 50L),
    primary_k = 30L,
    reference_nn = knn,
    sample_size_for_global_metrics = min(3000L, nrow(x)),
    sample_size_for_local_metrics = min(3000L, nrow(x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = "mnist_clean_opentsne"
  )
}

label_cols <- function(labels, alpha = 0.55) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

plot_methods <- function(layouts, rows, labels, out_path) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1))]
  n_panels <- max(1L, length(ok))
  n_col <- min(3L, n_panels)
  n_row <- ceiling(n_panels / n_col)
  grDevices::png(out_path, width = 560 * n_col, height = 480 * n_row, res = 120)
  old <- graphics::par(mfrow = c(n_row, n_col), mar = c(3, 3, 3.4, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- label_cols(labels)
  for (id in ok) {
    row <- rows[rows$method == id, , drop = FALSE]
    layout <- layouts[[id]]
    graphics::plot(
      layout[, 1L],
      layout[, 2L],
      pch = 16,
      cex = if (nrow(layout) > 30000L) 0.06 else 0.18,
      col = cols,
      xlab = "Dim 1",
      ylab = "Dim 2",
      main = sprintf(
        "%s\nembed %.2fs total %.2fs trust %.3f acc %.3f",
        id,
        row$embedding_sec,
        row$total_sec,
        row$trustworthiness,
        row$label_knn_accuracy
      )
    )
  }
}

run_fastembedr_opentsne <- function(knn, perplexity, early_iter, normal_iter,
                                    n_threads, seed) {
  timed(fastEmbedR::opentsne_knn(
    knn,
    perplexity = perplexity,
    early_exaggeration_iter = early_iter,
    n_iter = normal_iter,
    learning_rate = "auto",
    negative_gradient_method = "fft",
    theta = 0.5,
    n_threads = n_threads,
    seed = seed
  ))
}

run_rtsne_neighbors <- function(knn, perplexity, early_iter, normal_iter,
                                n_threads, seed) {
  if (!requireNamespace("Rtsne", quietly = TRUE)) {
    stop("Rtsne is not installed.", call. = FALSE)
  }
  timed(Rtsne::Rtsne_neighbors(
    knn$indices,
    knn$distances,
    dims = 2L,
    perplexity = perplexity,
    max_iter = early_iter + normal_iter,
    theta = 0.5,
    eta = 200,
    stop_lying_iter = early_iter,
    mom_switch_iter = early_iter,
    momentum = 0.5,
    final_momentum = 0.8,
    exaggeration_factor = 12,
    num_threads = n_threads,
    verbose = FALSE
  )$Y)
}

run_reductionwrappers_opentsne <- function(x, perplexity, early_iter, normal_iter,
                                           n_threads, seed) {
  if (!requireNamespace("ReductionWrappers", quietly = TRUE)) {
    stop("ReductionWrappers is not installed.", call. = FALSE)
  }
  if (!requireNamespace("reticulate", quietly = TRUE) ||
      !reticulate::py_module_available("openTSNE")) {
    stop("Python openTSNE is unavailable in the active reticulate Python.", call. = FALSE)
  }
  timed(ReductionWrappers::openTSNE(
    as.data.frame(x),
    n_components = 2L,
    perplexity = perplexity,
    learning_rate = "auto",
    early_exaggeration_iter = early_iter,
    early_exaggeration = 12,
    n_iter = normal_iter,
    theta = 0.5,
    initialization = "random",
    metric = "euclidean",
    initial_momentum = 0.8,
    final_momentum = 0.8,
    n_jobs = n_threads,
    neighbors = "exact",
    negative_gradient_method = "fft",
    random_state = seed
  ))
}

run_one <- function(id, package, runner, x, labels, knn, shared_knn_sec,
                    seed, include_shared_knn) {
  tryCatch({
    measured <- runner()
    layout <- coerce_layout(measured$value, nrow(x))
    metrics <- score_layout(x, layout, labels, knn, id, seed)
    embedding_sec <- measured$sec
    nn_sec <- if (isTRUE(include_shared_knn)) shared_knn_sec else NA_real_
    data.frame(
      method = id,
      package = package,
      status = "success",
      error_message = NA_character_,
      preprocessing_sec = 0,
      nn_sec = nn_sec,
      embedding_sec = embedding_sec,
      total_sec = embedding_sec + ifelse(is.na(nn_sec), 0, nn_sec),
      trustworthiness = metrics$trustworthiness,
      knn_preservation_30 = metrics$knn_preservation_30,
      silhouette = metrics$silhouette,
      label_knn_accuracy = metrics$label_knn_accuracy,
      stringsAsFactors = FALSE
    ) |>
      structure(layout = layout)
  }, error = function(e) {
    data.frame(
      method = id,
      package = package,
      status = "failed",
      error_message = conditionMessage(e),
      preprocessing_sec = 0,
      nn_sec = if (isTRUE(include_shared_knn)) shared_knn_sec else NA_real_,
      embedding_sec = NA_real_,
      total_sec = NA_real_,
      trustworthiness = NA_real_,
      knn_preservation_30 = NA_real_,
      silhouette = NA_real_,
      label_knn_accuracy = NA_real_,
      stringsAsFactors = FALSE
    ) |>
      structure(layout = NULL)
  })
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 12000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early_iter", 100L)
normal_iter <- arg_int("normal_iter", 150L)
cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
out_dir <- arg_value("out_dir", file.path("results", "mnist_opentsne_clean"))

if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- readRDS(cache)
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
labels <- droplevels(mnist$labels[rows])
storage.mode(x) <- "double"
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))

message(
  "MNIST clean openTSNE benchmark: n=", nrow(x),
  " p=", ncol(x),
  " k=", k,
  " perplexity=", perplexity,
  " threads=", n_threads
)

knn_measured <- timed(fastEmbedR::nn(
  x,
  k = k + 1L,
  backend = "cpu",
  n_threads = n_threads
))
raw_knn <- knn_measured$value
knn <- list(
  indices = raw_knn$indices[, -1L, drop = FALSE],
  distances = raw_knn$distances[, -1L, drop = FALSE]
)
attr(knn, "backend") <- attr(raw_knn, "backend")
attr(knn, "exact") <- attr(raw_knn, "exact")

specs <- list(
  fastEmbedR_opentsne_knn = list(
    package = "fastEmbedR",
    include_shared_knn = TRUE,
    runner = function() run_fastembedr_opentsne(
      knn, perplexity, early_iter, normal_iter, n_threads, seed
    )
  ),
  Rtsne_neighbors = list(
    package = "Rtsne",
    include_shared_knn = TRUE,
    runner = function() run_rtsne_neighbors(
      knn, perplexity, early_iter, normal_iter, n_threads, seed
    )
  ),
  ReductionWrappers_openTSNE = list(
    package = "ReductionWrappers/Python openTSNE",
    include_shared_knn = FALSE,
    runner = function() run_reductionwrappers_opentsne(
      x, perplexity, early_iter, normal_iter, n_threads, seed
    )
  )
)

rows_out <- list()
layouts <- list()
for (id in names(specs)) {
  message("Running ", id)
  result <- run_one(
    id,
    specs[[id]]$package,
    specs[[id]]$runner,
    x,
    labels,
    knn,
    knn_measured$sec,
    seed,
    specs[[id]]$include_shared_knn
  )
  rows_out[[id]] <- result
  layouts[[id]] <- attr(result, "layout")
}

bench <- do.call(rbind, rows_out)
bench$dataset <- paste0("mnist_idx_", nrow(x), "_pca50")
bench$n <- nrow(x)
bench$p <- ncol(x)
bench$k <- k
bench$perplexity <- perplexity
bench$early_iter <- early_iter
bench$normal_iter <- normal_iter
bench$seed <- seed
bench$n_threads <- n_threads
bench <- bench[, c(
  "dataset", "method", "package", "status", "error_message",
  "n", "p", "k", "perplexity", "early_iter", "normal_iter",
  "seed", "n_threads", "preprocessing_sec", "nn_sec",
  "embedding_sec", "total_sec", "trustworthiness",
  "knn_preservation_30", "silhouette", "label_knn_accuracy"
)]

layout_dir <- file.path(out_dir, "layouts")
dir.create(layout_dir, showWarnings = FALSE)
for (id in names(layouts)) {
  if (!is.null(layouts[[id]])) {
    saveRDS(
      layouts[[id]],
      file.path(layout_dir, paste0("MNIST-CLEAN-", nrow(x), "-", id, "-seed", seed, ".rds")),
      version = 2
    )
  }
}

plot_id <- paste0("MNIST-CLEAN-", nrow(x), "-seed", seed)
plot_path <- file.path(out_dir, paste0(plot_id, ".png"))
plot_methods(layouts, bench, labels, plot_path)

csv_path <- file.path(out_dir, paste0("mnist_", nrow(x), "_opentsne_clean_seed", seed, ".csv"))
write.csv(bench, csv_path, row.names = FALSE)
write.csv(bench, file.path(out_dir, "latest_mnist_opentsne_clean.csv"), row.names = FALSE)

message("Plot ID: ", plot_id)
message("Plot: ", normalizePath(plot_path, mustWork = FALSE))
message("CSV: ", normalizePath(csv_path, mustWork = FALSE))
print(bench, row.names = FALSE)
