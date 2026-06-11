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

arg_logical <- function(name, default = FALSE) {
  value <- tolower(arg_value(name, if (isTRUE(default)) "true" else "false"))
  value %in% c("1", "true", "yes", "y")
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

timed <- function(expr) {
  invisible(gc())
  value <- NULL
  sec <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(sec))
}

as_layout <- function(x, n) {
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

label_colors <- function(labels, alpha = 0.48) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

score_layout <- function(x, layout, labels, knn, method, backend, seed) {
  metric_k <- c(15L, 30L, 50L)
  metric_k <- metric_k[metric_k <= ncol(knn$indices)]
  if (!length(metric_k)) metric_k <- min(15L, ncol(knn$indices))
  fastEmbedR::evaluate_embedding(
    x,
    layout,
    labels = labels,
    k = metric_k,
    primary_k = min(30L, max(metric_k)),
    reference_nn = knn,
    sample_size_for_global_metrics = min(3000L, nrow(x)),
    sample_size_for_local_metrics = min(3000L, nrow(x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = backend,
    dataset = "mnist70k_opentsne_fft"
  )
}

empty_row <- function(id, package, backend, negative_method, status, message,
                      n, p, k, perplexity, early_iter, normal_iter, seed,
                      n_threads, nn_backend, nn_sec) {
  data.frame(
    dataset = paste0("mnist_idx_", n, "_pca", p),
    method = id,
    package = package,
    backend = backend,
    negative_gradient_method = negative_method,
    status = status,
    error_message = message,
    n = as.integer(n),
    p = as.integer(p),
    k = as.integer(k),
    perplexity = as.numeric(perplexity),
    early_iter = as.integer(early_iter),
    normal_iter = as.integer(normal_iter),
    seed = as.integer(seed),
    n_threads = as.integer(n_threads),
    nn_backend = nn_backend,
    preprocessing_sec = 0,
    nn_sec = as.numeric(nn_sec),
    embedding_sec = NA_real_,
    total_sec = NA_real_,
    trustworthiness = NA_real_,
    knn_preservation_30 = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    optimizer = NA_character_,
    repulsion = NA_character_,
    plot_id = NA_character_,
    plot_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

run_one <- function(id, package, backend, negative_method, runner,
                    x, labels, knn, seed, n_threads, nn_backend,
                    nn_sec, k, perplexity, early_iter, normal_iter) {
  n <- nrow(x)
  p <- ncol(x)
  tryCatch({
    measured <- runner()
    layout <- as_layout(measured$value, n)
    cfg <- attr(measured$value, "fastEmbedR_config")
    metrics <- score_layout(x, layout, labels, knn, id, backend, seed)
    row <- empty_row(
      id, package, backend, negative_method, "success", NA_character_,
      n, p, k, perplexity, early_iter, normal_iter, seed, n_threads,
      nn_backend, nn_sec
    )
    row$embedding_sec <- measured$sec
    row$total_sec <- measured$sec + nn_sec
    row$trustworthiness <- metrics$trustworthiness
    row$knn_preservation_30 <- metrics$knn_preservation_30
    row$silhouette <- metrics$silhouette
    row$label_knn_accuracy <- metrics$label_knn_accuracy
    row$optimizer <- cfg$optimizer %||% NA_character_
    row$repulsion <- cfg$repulsion %||% NA_character_
    structure(row, layout = layout)
  }, error = function(e) {
    empty_row(
      id, package, backend, negative_method, "failed", conditionMessage(e),
      n, p, k, perplexity, early_iter, normal_iter, seed, n_threads,
      nn_backend, nn_sec
    ) |>
      structure(layout = NULL)
  })
}

plot_layouts <- function(layouts, rows, labels, out_path) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1L))]
  if (!length(ok)) return(invisible(FALSE))
  n_col <- min(3L, length(ok))
  n_row <- ceiling(length(ok) / n_col)
  grDevices::png(out_path, width = 650 * n_col, height = 560 * n_row, res = 130)
  old <- graphics::par(mfrow = c(n_row, n_col), mar = c(3, 3, 4.2, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- label_colors(labels)
  robust_limits <- function(values) {
    values <- values[is.finite(values)]
    if (!length(values)) return(c(-1, 1))
    limits <- as.numeric(stats::quantile(values, c(0.002, 0.998), names = FALSE, na.rm = TRUE))
    if (!all(is.finite(limits)) || limits[1L] >= limits[2L]) limits <- range(values, finite = TRUE)
    if (!all(is.finite(limits)) || limits[1L] >= limits[2L]) limits <- c(-1, 1)
    limits + c(-0.04, 0.04) * diff(limits)
  }
  for (id in ok) {
    row <- rows[rows$method == id, , drop = FALSE]
    layout <- layouts[[id]]
    graphics::plot(
      layout[, 1L],
      layout[, 2L],
      col = cols,
      pch = 16,
      cex = if (nrow(layout) > 30000L) 0.28 else 0.28,
      xlim = robust_limits(layout[, 1L]),
      ylim = robust_limits(layout[, 2L]),
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
  invisible(TRUE)
}

plot_layouts_inspection <- function(layouts, rows, labels, out_path,
                                    sample_per_class = 2500L,
                                    seed = 6L) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1L))]
  if (!length(ok)) return(invisible(FALSE))
  set.seed(seed)
  labels <- factor(labels)
  sample_rows <- unlist(tapply(seq_along(labels), labels, function(idx) {
    sample(idx, min(length(idx), sample_per_class))
  }))
  sample_rows <- sort(sample_rows)
  n_col <- min(3L, length(ok))
  n_row <- ceiling(length(ok) / n_col)
  grDevices::png(out_path, width = 700 * n_col, height = 620 * n_row, res = 135)
  old <- graphics::par(mfrow = c(n_row, n_col), mar = c(3, 3, 4.5, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  cols <- stats::setNames(
    grDevices::adjustcolor(pal, alpha.f = 0.82),
    levels(labels)
  )[as.character(labels[sample_rows])]
  for (id in ok) {
    row <- rows[rows$method == id, , drop = FALSE]
    layout <- layouts[[id]][sample_rows, , drop = FALSE]
    graphics::plot(
      layout[, 1L],
      layout[, 2L],
      col = cols,
      pch = 16,
      cex = 0.12,
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
  invisible(TRUE)
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 90L)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early-iter", 20L)
normal_iter <- arg_int("normal-iter", 30L)
include_rtsne <- arg_logical("rtsne", TRUE)
include_rtsne_scaled <- arg_logical("rtsne-scaled-eta", TRUE)
cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
out_dir <- arg_value("out-dir", file.path("results", "mnist70k_opentsne_fft"))
nn_backend <- arg_value("nn-backend", "cpu_nndescent")

if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), showWarnings = FALSE)

mnist <- readRDS(cache)
if (!all(c("x", "labels") %in% names(mnist))) {
  stop("MNIST cache must contain `x` and `labels`.", call. = FALSE)
}
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
labels <- droplevels(factor(mnist$labels[rows]))
storage.mode(x) <- "double"
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))

message(
  "MNIST openTSNE FFT benchmark: n=", nrow(x),
  " p=", ncol(x),
  " k=", k,
  " perplexity=", perplexity,
  " early_iter=", early_iter,
  " normal_iter=", normal_iter,
  " threads=", n_threads,
  " nn_backend=", nn_backend
)

knn_time <- timed(fastEmbedR::nn(
  x,
  k = k + 1L,
  backend = nn_backend,
  n_threads = n_threads
))
raw_knn <- knn_time$value
clean_knn <- fastEmbedR:::coerce_knn_input(raw_knn)
clean_mat <- fastEmbedR:::materialize_knn_range(
  clean_knn$indices,
  clean_knn$distances,
  clean_knn$col_start,
  clean_knn$n_neighbors
)
knn <- list(
  indices = clean_mat$indices,
  distances = clean_mat$distances
)
attr(knn, "backend") <- attr(raw_knn, "backend")
attr(knn, "exact") <- attr(raw_knn, "exact")

specs <- list(
  fastEmbedR_opentsne_fft_grid = list(
    package = "fastEmbedR",
    backend = "cpu",
    negative = "fft",
    runner = function() timed(fastEmbedR::opentsne_knn(
      knn,
      n_neighbors = ncol(knn$indices),
      perplexity = perplexity,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      negative_gradient_method = "fft",
      n_threads = n_threads,
      seed = seed
    ))
  ),
  fastEmbedR_opentsne_metal_fft = list(
    package = "fastEmbedR",
    backend = "metal",
    negative = "fft",
    runner = function() timed(fastEmbedR::opentsne_knn(
      knn,
      n_neighbors = ncol(knn$indices),
      perplexity = perplexity,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      negative_gradient_method = "fft",
      backend = "metal",
      seed = seed
    ))
  ),
  fastEmbedR_opentsne_cuda_fft = list(
    package = "fastEmbedR",
    backend = "cuda",
    negative = "fft",
    runner = function() timed(fastEmbedR::opentsne_knn(
      knn,
      n_neighbors = ncol(knn$indices),
      perplexity = perplexity,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      negative_gradient_method = "fft",
      backend = "cuda",
      seed = seed
    ))
  )
)

if (isTRUE(include_rtsne)) {
  specs$Rtsne_neighbors <- list(
    package = "Rtsne",
    backend = "cpu",
    negative = "bh",
    runner = function() {
      if (!requireNamespace("Rtsne", quietly = TRUE)) {
        stop("Rtsne is not installed.", call. = FALSE)
      }
      set.seed(seed)
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
  )
}

if (isTRUE(include_rtsne_scaled)) {
  specs$Rtsne_neighbors_scaled_eta <- list(
    package = "Rtsne",
    backend = "cpu",
    negative = "bh",
    runner = function() {
      if (!requireNamespace("Rtsne", quietly = TRUE)) {
        stop("Rtsne is not installed.", call. = FALSE)
      }
      set.seed(seed)
      timed(Rtsne::Rtsne_neighbors(
        knn$indices,
        knn$distances,
        dims = 2L,
        perplexity = perplexity,
        max_iter = early_iter + normal_iter,
        theta = 0.5,
        eta = nrow(x) / 12,
        stop_lying_iter = early_iter,
        mom_switch_iter = early_iter,
        momentum = 0.5,
        final_momentum = 0.8,
        exaggeration_factor = 12,
        num_threads = n_threads,
        verbose = FALSE
      )$Y)
    }
  )
}

result_rows <- list()
layouts <- list()
for (id in names(specs)) {
  message("Running ", id)
  row <- run_one(
    id = id,
    package = specs[[id]]$package,
    backend = specs[[id]]$backend,
    negative_method = specs[[id]]$negative,
    runner = specs[[id]]$runner,
    x = x,
    labels = labels,
    knn = knn,
    seed = seed,
    n_threads = n_threads,
    nn_backend = nn_backend,
    nn_sec = knn_time$sec,
    k = k,
    perplexity = perplexity,
    early_iter = early_iter,
    normal_iter = normal_iter
  )
  result_rows[[id]] <- row
  layouts[[id]] <- attr(row, "layout")
}

bench <- do.call(rbind, result_rows)
plot_id <- paste0("MNIST70K-OPENTSNE-FFT-", nrow(x), "-seed", seed, "-e", early_iter, "-n", normal_iter)
plot_path <- file.path(out_dir, paste0(plot_id, ".png"))
plot_layouts(layouts, bench, labels, plot_path)
inspect_plot_path <- file.path(out_dir, paste0(plot_id, "-INSPECT.png"))
plot_layouts_inspection(layouts, bench, labels, inspect_plot_path, seed = seed)
bench$plot_id <- plot_id
bench$plot_path <- normalizePath(plot_path, mustWork = FALSE)
bench$inspection_plot_path <- normalizePath(inspect_plot_path, mustWork = FALSE)

for (id in names(layouts)) {
  if (!is.null(layouts[[id]])) {
    saveRDS(
      layouts[[id]],
      file.path(out_dir, "layouts", paste0(plot_id, "-", id, ".rds")),
      version = 2
    )
  }
}

csv_path <- file.path(out_dir, paste0(plot_id, ".csv"))
write.csv(bench, csv_path, row.names = FALSE)
write.csv(bench, file.path(out_dir, "latest_mnist70k_opentsne_fft.csv"), row.names = FALSE)

message("Plot ID: ", plot_id)
message("Plot: ", normalizePath(plot_path, mustWork = FALSE))
message("Inspection plot: ", normalizePath(inspect_plot_path, mustWork = FALSE))
message("CSV: ", normalizePath(csv_path, mustWork = FALSE))
print(bench, row.names = FALSE)
