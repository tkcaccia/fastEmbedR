#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

arg_flag <- function(name, default = FALSE) {
  value <- arg_value(name, NA_character_)
  if (is.na(value)) return(isTRUE(default))
  tolower(value) %in% c("1", "true", "yes", "y")
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

json_params <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x, use.names = FALSE), sep = "="), collapse = ";")
}

download_file <- function(url, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path) && file.info(path)$size > 0) return(path)
  utils::download.file(url, path, mode = "wb", quiet = TRUE)
  path
}

read_idx_images <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 4L, size = 4L, endian = "big")
  if (length(header) != 4L || header[[1L]] != 2051L) {
    stop("Invalid IDX image file: ", path, call. = FALSE)
  }
  n <- header[[2L]]
  rows <- header[[3L]]
  cols <- header[[4L]]
  values <- readBin(con, "integer", n = n * rows * cols, size = 1L, signed = FALSE)
  x <- matrix(as.numeric(values) / 255, nrow = n, byrow = TRUE)
  colnames(x) <- paste0("px", seq_len(ncol(x)))
  x
}

read_idx_labels <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  header <- readBin(con, "integer", n = 2L, size = 4L, endian = "big")
  if (length(header) != 2L || header[[1L]] != 2049L) {
    stop("Invalid IDX label file: ", path, call. = FALSE)
  }
  factor(readBin(con, "integer", n = header[[2L]], size = 1L, signed = FALSE))
}

load_raw_mnist <- function(cache_dir, raw_cache) {
  if (file.exists(raw_cache)) return(readRDS(raw_cache))
  base <- "https://storage.googleapis.com/cvdf-datasets/mnist"
  files <- c(
    train_images = "train-images-idx3-ubyte.gz",
    train_labels = "train-labels-idx1-ubyte.gz",
    test_images = "t10k-images-idx3-ubyte.gz",
    test_labels = "t10k-labels-idx1-ubyte.gz"
  )
  paths <- vapply(
    files,
    function(file) download_file(file.path(base, file), file.path(cache_dir, "mnist", file)),
    character(1L)
  )
  out <- list(
    x = rbind(read_idx_images(paths[["train_images"]]), read_idx_images(paths[["test_images"]])),
    labels = factor(c(
      as.character(read_idx_labels(paths[["train_labels"]])),
      as.character(read_idx_labels(paths[["test_labels"]]))
    )),
    source = "MNIST IDX public files, raw flattened 28x28 pixels"
  )
  dir.create(dirname(raw_cache), recursive = TRUE, showWarnings = FALSE)
  saveRDS(out, raw_cache, version = 2)
  out
}

stratified_rows <- function(labels, n, seed) {
  n <- min(as.integer(n), length(labels))
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  labels <- as.factor(labels)
  base <- split(seq_along(labels), labels)
  take <- lapply(base, function(idx) ceiling(length(idx) / length(labels) * n))
  rows <- unlist(Map(function(idx, m) sample(idx, min(length(idx), m)), base, take), use.names = FALSE)
  if (length(rows) > n) rows <- sample(rows, n)
  if (length(rows) < n) rows <- c(rows, sample(setdiff(seq_along(labels), rows), n - length(rows)))
  sort(rows)
}

coerce_layout <- function(x, n) {
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (is.list(x) && !is.null(x$embedding)) x <- x$embedding
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) stop("No n x 2 layout returned.", call. = FALSE)
  x[, 1:2, drop = FALSE]
}

package_version_or_na <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(package))
}

score_layout <- function(x, labels, layout, reference_nn, seed, n_threads, metric_n) {
  rows <- stratified_rows(labels, min(metric_n, nrow(x)), seed + 17L)
  fastEmbedR::evaluate_embedding(
    x[rows, , drop = FALSE],
    layout[rows, , drop = FALSE],
    labels = labels[rows],
    k = c(15L, 30L, 50L),
    reference_nn = NULL,
    sample_size_for_global_metrics = min(3000L, length(rows)),
    sample_size_for_local_metrics = min(3000L, length(rows)),
    seed = seed,
    method = "mnist70k_r_package_comparison",
    backend = "cpu",
    n_threads = n_threads,
    dataset = "mnist70k_raw_flattened"
  )
}

result_row <- function(method, package, family, input_type, status, error_message,
                       nn_sec = NA_real_, init_sec = NA_real_, embed_sec = NA_real_,
                       total_sec = NA_real_, metrics = NULL, parameters = list()) {
  out <- data.frame(
    method = method,
    package = package,
    family = family,
    input_type = input_type,
    status = status,
    error_message = error_message,
    package_version = package_version_or_na(package),
    nn_sec = as.numeric(nn_sec),
    init_sec = as.numeric(init_sec),
    embed_sec = as.numeric(embed_sec),
    total_sec = as.numeric(total_sec),
    trustworthiness = NA_real_,
    knn_preservation_30 = NA_real_,
    label_knn_accuracy = NA_real_,
    silhouette = NA_real_,
    parameters_json = json_params(parameters),
    stringsAsFactors = FALSE
  )
  if (!is.null(metrics)) {
    metric_cols <- c(
      "trustworthiness", "knn_preservation_30",
      "label_knn_accuracy", "silhouette"
    )
    for (nm in intersect(metric_cols, names(metrics))) out[[nm]] <- metrics[[nm]][[1L]]
  }
  out
}

run_method <- function(method, package, family, input_type, expr, x, labels,
                       reference_nn, seed, n_threads, metric_n,
                       nn_sec = NA_real_, init_sec = NA_real_, parameters = list()) {
  message("Running ", method)
  tryCatch({
    measured <- timed(expr())
    layout <- coerce_layout(measured$value, nrow(x))
    metrics <- score_layout(x, labels, layout, reference_nn, seed, n_threads, metric_n)
    list(
      row = result_row(
        method, package, family, input_type, "success", NA_character_,
        nn_sec = nn_sec, init_sec = init_sec, embed_sec = measured$sec,
        total_sec = sum(c(nn_sec, init_sec, measured$sec), na.rm = TRUE),
        metrics = metrics, parameters = parameters
      ),
      layout = layout
    )
  }, error = function(e) {
    list(
      row = result_row(
        method, package, family, input_type, "failed", conditionMessage(e),
        nn_sec = nn_sec, init_sec = init_sec, parameters = parameters
      ),
      layout = NULL
    )
  })
}

plot_layouts <- function(layouts, labels, rows, path, seed, max_points = 70000L, point_cex = 0.28) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1))]
  if (!length(ok)) return(invisible(FALSE))
  keep <- stratified_rows(labels, min(max_points, length(labels)), seed + 29L)
  ncol_plot <- min(3L, length(ok))
  nrow_plot <- ceiling(length(ok) / ncol_plot)
  png(path, width = 900 * ncol_plot, height = 720 * nrow_plot, res = 150)
  old <- par(no.readonly = TRUE)
  on.exit({ par(old); dev.off() }, add = TRUE)
  par(mfrow = c(nrow_plot, ncol_plot), mar = c(2.0, 2.0, 3.2, 0.6), oma = c(0, 0, 2, 0))
  pal <- grDevices::hcl.colors(length(levels(factor(labels))), "Dark 3")
  cols <- pal[as.integer(factor(labels))]
  for (id in ok) {
    row <- rows[rows$method == id, , drop = FALSE]
    layout <- layouts[[id]]
    main <- sprintf("%s\n%.1fs trust %.3f", id, row$embed_sec[[1L]], row$trustworthiness[[1L]])
    plot(layout[keep, 1L], layout[keep, 2L], pch = 16, cex = point_cex,
         col = cols[keep], axes = FALSE, xlab = "", ylab = "", main = main)
    box(col = "grey70")
  }
  mtext("MNIST 70k raw flattened 28x28 package comparison", outer = TRUE, cex = 1.2)
  invisible(TRUE)
}

suppressPackageStartupMessages(library(fastEmbedR))

seed <- arg_int("seed", 6L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
metric_n <- arg_int("metric-n", 5000L)
plot_n <- arg_int("plot-n", 70000L)
run_slow <- arg_flag("run-slow", FALSE)
run_umap_package <- arg_flag("run-umap-package", FALSE)
cache_dir <- arg_value("raw-cache-dir", file.path("results", "dataset_cache"))
raw_cache <- arg_value("raw-cache", file.path(cache_dir, "mnist_idx_70000_flattened.rds"))
knn_cache_dir <- arg_value("knn-cache-dir", file.path("results", "knn_cache"))
out_dir <- arg_value("out-dir", file.path("results", paste0("mnist70k_r_package_comparison_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- load_raw_mnist(cache_dir, raw_cache)
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
storage.mode(x) <- "double"
labels <- droplevels(factor(mnist$labels[rows]))
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))

message("MNIST70k R-package comparison: raw flattened input only")
message("  n=", nrow(x), " p=", ncol(x), " k=", k, " perplexity=", perplexity)

knn_cache <- file.path(knn_cache_dir, sprintf("mnist70k_flat_k%d_seed%d_metal_nndescent.rds", k + 1L, seed))
if (file.exists(knn_cache)) {
  cached <- readRDS(knn_cache)
  knn <- cached$knn
  nn_sec <- cached$nn_sec
  message("Loaded shared KNN cache: ", knn_cache)
} else {
  message("Building shared KNN with fastEmbedR metal_nndescent")
  nn_run <- timed(fastEmbedR::nn(x, k = k + 1L, backend = "metal_nndescent", n_threads = n_threads))
  knn <- nn_run$value
  nn_sec <- nn_run$sec
  dir.create(dirname(knn_cache), recursive = TRUE, showWarnings = FALSE)
  saveRDS(list(knn = knn, nn_sec = nn_sec, backend = attr(knn, "backend"), created = Sys.time()),
          knn_cache, version = 2)
}

knn_clean <- fastEmbedR:::coerce_knn_input(knn)
knn_no_self <- list(
  indices = knn_clean$indices[, -1L, drop = FALSE],
  distances = knn_clean$distances[, -1L, drop = FALSE]
)
attr(knn_no_self, "backend") <- attr(knn, "backend") %||% "metal_nndescent"

message("Computing shared PCA initialization")
init_run <- timed(fastEmbedR:::make_opentsne_pca_init(x, n_components = 2L, seed = seed, backend = "cpu"))
y_init <- init_run$value
init_sec <- init_run$sec

results <- list()
layouts <- list()
add <- function(res) {
  results[[length(results) + 1L]] <<- res$row
  if (!is.null(res$layout)) layouts[[res$row$method[[1L]]]] <<- res$layout
}

add(run_method(
  "fastEmbedR openTSNE Metal", "fastEmbedR", "tsne", "precomputed_knn",
  function() fastEmbedR::opentsne_knn(
    knn, n_neighbors = k, perplexity = perplexity, Y_init = y_init,
    early_exaggeration_iter = 100L, n_iter = 150L, learning_rate = "auto",
    negative_gradient_method = "fft", backend = "metal", n_threads = n_threads, seed = seed
  ),
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec, init_sec = init_sec,
  parameters = list(k = k, perplexity = perplexity, init = "shared_pca", backend = "metal")
))

add(run_method(
  "fastEmbedR openTSNE CPU", "fastEmbedR", "tsne", "precomputed_knn",
  function() fastEmbedR::opentsne_knn(
    knn, n_neighbors = k, perplexity = perplexity, Y_init = y_init,
    early_exaggeration_iter = 100L, n_iter = 150L, learning_rate = "auto",
    negative_gradient_method = "fft", backend = "cpu", n_threads = n_threads, seed = seed
  ),
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec, init_sec = init_sec,
  parameters = list(k = k, perplexity = perplexity, init = "shared_pca", backend = "cpu")
))

add(run_method(
  "Rtsne_neighbors PCA init", "Rtsne", "tsne", "precomputed_knn",
  function() {
    if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne not installed.", call. = FALSE)
    Rtsne::Rtsne_neighbors(
      index = knn_no_self$indices, distance = knn_no_self$distances,
      dims = 2L, perplexity = perplexity, theta = 0.5,
      max_iter = 250L, Y_init = y_init, stop_lying_iter = 100L,
      mom_switch_iter = 100L, momentum = 0.5, final_momentum = 0.8,
      eta = 200, exaggeration_factor = 12, num_threads = n_threads,
      verbose = FALSE
    )$Y
  },
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec, init_sec = init_sec,
  parameters = list(k = k, perplexity = perplexity, init = "shared_pca", theta = 0.5)
))

add(run_method(
  "fastEmbedR UMAP Metal", "fastEmbedR", "umap", "precomputed_knn",
  function() fastEmbedR::umap_knn(knn, backend = "metal", seed = seed),
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec,
  parameters = list(k = k, init = "package_default", backend = "metal")
))

add(run_method(
  "fastEmbedR UMAP CPU", "fastEmbedR", "umap", "precomputed_knn",
  function() fastEmbedR::umap_knn(knn, backend = "cpu", seed = seed),
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec,
  parameters = list(k = k, init = "package_default", backend = "cpu")
))

add(run_method(
  "uwot UMAP fast_sgd precomputed KNN", "uwot", "umap", "precomputed_knn",
  function() {
    if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot not installed.", call. = FALSE)
    set.seed(seed)
    uwot::umap(
      X = x, n_neighbors = k, n_components = 2L,
      nn_method = list(idx = knn_no_self$indices, dist = knn_no_self$distances),
      n_epochs = 200L, init = "spectral", min_dist = 0.01,
      metric = "euclidean", learning_rate = 1, negative_sample_rate = 5,
      repulsion_strength = 1, fast_sgd = TRUE, n_threads = n_threads,
      n_sgd_threads = n_threads, ret_model = FALSE, verbose = FALSE,
      seed = seed
    )
  },
  x, labels, knn_no_self, seed, n_threads, metric_n,
  nn_sec = nn_sec,
  parameters = list(k = k, init = "spectral", fast_sgd = TRUE, nn_method = "shared_precomputed")
))

if (isTRUE(run_umap_package)) {
  add(run_method(
    "umap package precomputed KNN", "umap", "umap", "precomputed_knn",
    function() {
      if (!requireNamespace("umap", quietly = TRUE)) stop("umap not installed.", call. = FALSE)
      uknn <- umap::umap.knn(knn_no_self$indices, knn_no_self$distances)
      cfg <- umap::umap.defaults
      cfg$n_neighbors <- k
      cfg$random_state <- seed
      cfg$knn <- uknn
      umap::umap(x, knn = uknn, config = cfg)$layout
    },
    x, labels, knn_no_self, seed, n_threads, metric_n,
    nn_sec = nn_sec,
    parameters = list(k = k, init = "umap_default", nn_method = "shared_precomputed")
  ))
} else {
  results[[length(results) + 1L]] <- result_row(
    "umap package precomputed KNN", "umap", "umap", "precomputed_knn",
    "skipped", "Skipped by default on MNIST70k because umap::umap is usually much slower; rerun with --run-umap-package=true.",
    nn_sec = nn_sec, parameters = list(k = k, run_umap_package = FALSE)
  )
}

if (isTRUE(run_slow)) {
  add(run_method(
    "tsne package PCA init", "tsne", "tsne", "raw_x_exact",
    function() {
      if (!requireNamespace("tsne", quietly = TRUE)) stop("tsne not installed.", call. = FALSE)
      tsne::tsne(
        x, initial_config = y_init, k = 2L, initial_dims = ncol(x),
        perplexity = perplexity, max_iter = 250L, whiten = FALSE,
        epoch = 251L
      )
    },
    x, labels, knn_no_self, seed, n_threads, metric_n,
    init_sec = init_sec,
    parameters = list(perplexity = perplexity, init = "shared_pca", note = "exact raw-X t-SNE")
  ))
} else {
  for (method in c("tsne package PCA init", "mmtsne", "Rdimtools do.tsne")) {
    pkg <- if (startsWith(method, "tsne")) "tsne" else if (method == "mmtsne") "mmtsne" else "Rdimtools"
    results[[length(results) + 1L]] <- result_row(
      method, pkg, "tsne", "raw_x_exact",
      "skipped", "Skipped by default on MNIST70k flattened because this path does not use precomputed KNN and is exact/raw-X or lacks PCA initialization; rerun with --run-slow=true.",
      init_sec = if (startsWith(method, "tsne")) init_sec else NA_real_,
      parameters = list(n = nrow(x), p = ncol(x), run_slow = FALSE)
    )
  }
}

bench <- do.call(rbind, results)
bench <- bench[order(bench$family, bench$status != "success", bench$method), , drop = FALSE]
csv <- file.path(out_dir, "mnist70k_r_package_comparison.csv")
write.csv(bench, csv, row.names = FALSE)
plot_path <- file.path(out_dir, "mnist70k_r_package_comparison.png")
plot_layouts(layouts, labels, bench, plot_path, seed, max_points = plot_n)

message("Results CSV: ", normalizePath(csv, winslash = "/", mustWork = FALSE))
message("Plot PNG: ", normalizePath(plot_path, winslash = "/", mustWork = FALSE))
print(bench[, c(
  "method", "package", "family", "input_type", "status",
  "nn_sec", "init_sec", "embed_sec", "total_sec",
  "trustworthiness", "label_knn_accuracy", "error_message"
)], row.names = FALSE)
