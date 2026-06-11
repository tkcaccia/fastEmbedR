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
  levs <- levels(labels)
  base <- floor(n / length(levs))
  rem <- n - base * length(levs)
  rows <- integer(0L)
  for (lev in levs) {
    idx <- which(labels == lev)
    take <- min(length(idx), base + as.integer(rem > 0L))
    rem <- max(0L, rem - 1L)
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

coerce_layout <- function(x, n) {
  if (inherits(x, "fastEmbedR_embedding")) x <- x$layout
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (is.list(x) && !is.null(x$embedding)) x <- x$embedding
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) stop("No n x 2 layout returned.", call. = FALSE)
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
    sample_size_for_global_metrics = min(2500L, nrow(x)),
    sample_size_for_local_metrics = min(2500L, nrow(x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = "cpu",
    dataset = "mnist_available_r_tools"
  )
}

label_cols <- function(labels, alpha = 0.55) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

run_method <- function(id, package, x, labels, knn, seed, k, perplexity, n_threads) {
  idx <- knn$indices
  dst <- knn$distances
  early_iter <- 100L
  normal_iter <- 150L
  max_iter <- early_iter + normal_iter

  tryCatch({
    measured <- switch(
      id,
      fastEmbedR_umap = timed(fastEmbedR::embed_knn(knn, method = "umap", seed = seed, backend = "cpu")),
      fastEmbedR_tsne = timed(fastEmbedR::embed_knn(
        knn, method = "tsne", perplexity = perplexity, max_iter = max_iter,
        stop_lying_iter = early_iter, mom_switch_iter = early_iter,
        eta = 200, exaggeration_factor = 12, negative_gradient_method = "bh",
        theta = 0.5, n_threads = n_threads, seed = seed
      )),
      fastEmbedR_opentsne = timed(fastEmbedR::opentsne_knn(
        knn, perplexity = perplexity, early_exaggeration_iter = early_iter,
        n_iter = normal_iter, learning_rate = "auto",
        negative_gradient_method = "bh", theta = 0.5,
        n_threads = n_threads, seed = seed
      )),
      uwot_fast_sgd = {
        if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot not installed.", call. = FALSE)
        timed(uwot::umap(
          X = x, n_neighbors = k, n_components = 2L,
          nn_method = list(idx = idx, dist = dst),
          n_epochs = 200L, init = "spectral", min_dist = 0.01,
          metric = "euclidean", learning_rate = 1,
          negative_sample_rate = 5, repulsion_strength = 1,
          fast_sgd = TRUE, n_threads = n_threads, n_sgd_threads = n_threads,
          ret_model = FALSE, verbose = FALSE, seed = seed
        ))
      },
      umap_package = {
        if (!requireNamespace("umap", quietly = TRUE)) stop("umap package not installed.", call. = FALSE)
        timed({
          uknn <- umap::umap.knn(idx, dst)
          cfg <- umap::umap.defaults
          cfg$n_neighbors <- k
          cfg$random_state <- seed
          cfg$knn <- uknn
          umap::umap(x, knn = uknn, config = cfg)$layout
        })
      },
      Rtsne_neighbors = {
        if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne not installed.", call. = FALSE)
        timed(Rtsne::Rtsne_neighbors(
          idx, dst, dims = 2L, perplexity = perplexity,
          max_iter = max_iter, theta = 0.5, eta = 200,
          stop_lying_iter = early_iter, mom_switch_iter = early_iter,
          momentum = 0.5, final_momentum = 0.8,
          exaggeration_factor = 12, num_threads = n_threads, verbose = FALSE
        )$Y)
      },
      tsne_package = {
        if (!requireNamespace("tsne", quietly = TRUE)) stop("tsne package not installed.", call. = FALSE)
        timed(tsne::tsne(
          x, k = 2L, initial_dims = min(30L, ncol(x)), perplexity = perplexity,
          max_iter = max_iter, epoch = max_iter + 1L, whiten = FALSE
        ))
      },
      Rdimtools_tsne = {
        if (!requireNamespace("Rdimtools", quietly = TRUE)) stop("Rdimtools not installed.", call. = FALSE)
        timed(Rdimtools::do.tsne(
          x, ndim = 2L, perplexity = perplexity, maxiter = max_iter,
          pca = FALSE, BHuse = TRUE, BHtheta = 0.5
        )$Y)
      },
      ReductionWrappers_openTSNE = {
        if (!requireNamespace("ReductionWrappers", quietly = TRUE)) stop("ReductionWrappers not installed.", call. = FALSE)
        if (!requireNamespace("reticulate", quietly = TRUE) || !reticulate::py_module_available("openTSNE")) {
          stop("Python openTSNE is unavailable in the active reticulate Python.", call. = FALSE)
        }
        timed(ReductionWrappers::openTSNE(
          as.data.frame(x), n_components = 2L, perplexity = perplexity,
          learning_rate = 100, early_exaggeration_iter = early_iter,
          early_exaggeration = 12, n_iter = normal_iter,
          theta = 0.5, initialization = "random", metric = "euclidean",
          initial_momentum = 0.8, final_momentum = 0.8, n_jobs = n_threads,
          neighbors = "exact", negative_gradient_method = "bh",
          random_state = seed
        ))
      },
      stop("Unknown method id: ", id, call. = FALSE)
    )
    layout <- coerce_layout(measured$value, nrow(x))
    metrics <- score_layout(x, layout, labels, knn, id, seed)
    list(
      row = data.frame(
        method = id,
        package = package,
        status = "success",
        error_message = NA_character_,
        wall_sec = measured$sec,
        trustworthiness = metrics$trustworthiness,
        knn_preservation_30 = metrics$knn_preservation_30,
        silhouette = metrics$silhouette,
        label_knn_accuracy = metrics$label_knn_accuracy,
        stringsAsFactors = FALSE
      ),
      layout = layout
    )
  }, error = function(e) {
    list(
      row = data.frame(
        method = id,
        package = package,
        status = "failed",
        error_message = conditionMessage(e),
        wall_sec = NA_real_,
        trustworthiness = NA_real_,
        knn_preservation_30 = NA_real_,
        silhouette = NA_real_,
        label_knn_accuracy = NA_real_,
        stringsAsFactors = FALSE
      ),
      layout = NULL
    )
  })
}

plot_methods <- function(layouts, rows, labels, out_path) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1))]
  n_panels <- max(1L, length(ok))
  n_col <- min(3L, n_panels)
  n_row <- ceiling(n_panels / n_col)
  grDevices::png(out_path, width = 520 * n_col, height = 450 * n_row, res = 120)
  old <- graphics::par(mfrow = c(n_row, n_col), mar = c(3, 3, 3.2, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- label_cols(labels)
  for (id in ok) {
    row <- rows[rows$method == id, , drop = FALSE]
    layout <- layouts[[id]]
    graphics::plot(
      layout[, 1L], layout[, 2L], pch = 16, cex = 0.24, col = cols,
      xlab = "Dim 1", ylab = "Dim 2",
      main = sprintf("%s\n%.2fs trust %.3f acc %.3f", id, row$wall_sec, row$trustworthiness, row$label_knn_accuracy)
    )
  }
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 2500L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
cache <- arg_value("cache", "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds")
if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
mnist <- readRDS(cache)
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
labels <- droplevels(mnist$labels[rows])
storage.mode(x) <- "double"

out_dir <- file.path("results", "mnist_available_r_tools")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("MNIST available-tools benchmark: n=", nrow(x), " p=", ncol(x), " k=", k)
knn_time <- timed(fastEmbedR::nn(x, k = k + 1L, backend = "cpu", n_threads = n_threads))
knn <- knn_time$value
knn <- list(indices = knn$indices[, -1L, drop = FALSE], distances = knn$distances[, -1L, drop = FALSE])
attr(knn, "backend") <- "fastEmbedR_cpu_exact"

specs <- list(
  fastEmbedR_umap = "fastEmbedR",
  fastEmbedR_tsne = "fastEmbedR",
  fastEmbedR_opentsne = "fastEmbedR",
  uwot_fast_sgd = "uwot",
  umap_package = "umap",
  Rtsne_neighbors = "Rtsne",
  tsne_package = "tsne",
  Rdimtools_tsne = "Rdimtools",
  ReductionWrappers_openTSNE = "ReductionWrappers/Python openTSNE"
)
res <- lapply(names(specs), function(id) {
  message("Running ", id)
  run_method(id, specs[[id]], x, labels, knn, seed, k,
             min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L)),
             n_threads)
})
bench <- do.call(rbind, lapply(res, `[[`, "row"))
bench$dataset <- paste0("mnist_idx_", nrow(x), "_pca50")
bench$n <- nrow(x)
bench$p <- ncol(x)
bench$seed <- seed
bench$k <- k
bench$shared_knn_sec <- knn_time$sec
layouts <- stats::setNames(lapply(res, `[[`, "layout"), names(specs))
plot_path <- file.path(out_dir, paste0("mnist_", nrow(x), "_available_r_tools_seed", seed, ".png"))
plot_methods(layouts, bench, labels, plot_path)
bench$plot_path <- plot_path

csv_path <- file.path(out_dir, paste0("mnist_", nrow(x), "_available_r_tools_seed", seed, ".csv"))
write.csv(bench, csv_path, row.names = FALSE)
write.csv(bench, file.path(out_dir, "latest_mnist_available_r_tools.csv"), row.names = FALSE)
print(bench[, c("dataset", "method", "package", "status", "wall_sec", "trustworthiness", "knn_preservation_30", "silhouette", "label_knn_accuracy", "error_message")], row.names = FALSE)
cat("\nSaved CSV:", normalizePath(csv_path), "\n")
cat("Saved plot:", normalizePath(plot_path), "\n")
