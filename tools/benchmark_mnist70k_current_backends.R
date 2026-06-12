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

arg_num <- function(name, default) {
  value <- suppressWarnings(as.numeric(arg_value(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]), proc = t)
}

json_params <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x, use.names = FALSE), sep = "="), collapse = ";")
}

stratified_rows <- function(labels, n, seed) {
  n <- min(as.integer(n), length(labels))
  if (n >= length(labels)) return(seq_along(labels))
  set.seed(seed)
  labels <- as.factor(labels)
  base <- split(seq_along(labels), labels)
  take <- lapply(base, function(idx) {
    ceiling(length(idx) / length(labels) * n)
  })
  rows <- unlist(Map(function(idx, m) sample(idx, min(length(idx), m)), base, take), use.names = FALSE)
  if (length(rows) > n) rows <- sample(rows, n)
  if (length(rows) < n) rows <- c(rows, sample(setdiff(seq_along(labels), rows), n - length(rows)))
  sort(rows)
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

safe_status <- function(expr) {
  tryCatch(
    list(status = "success", value = force(expr), error = NA_character_),
    error = function(e) list(status = "failed", value = NULL, error = conditionMessage(e))
  )
}

as_layout <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.data.frame(x)) return(as.matrix(x))
  if (is.list(x) && !is.null(x$layout)) return(as.matrix(x$layout))
  as.matrix(x)
}

score_layout <- function(x, labels, layout, method, backend, seed, n_threads, metric_n, dataset_name) {
  rows <- stratified_rows(labels, min(metric_n, nrow(x)), seed + 17L)
  out <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x[rows, , drop = FALSE],
      layout[rows, , drop = FALSE],
      labels = labels[rows],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(rows)),
      sample_size_for_local_metrics = min(3000L, length(rows)),
      seed = seed,
      method = method,
      backend = "cpu",
      n_threads = n_threads,
      dataset = dataset_name
    ),
    error = function(e) {
      data.frame(
        trustworthiness = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        label_knn_accuracy = NA_real_,
        silhouette = NA_real_,
        metric_error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )
  out$metric_sample_n <- length(rows)
  out
}

make_row <- function(dataset,
                     method,
                     variant,
                     backend_requested,
                     backend_used,
                     status,
                     error_message,
                     n,
                     p,
                     seed,
                     nn_backend = NA_character_,
                     nn_sec = NA_real_,
                     embedding_sec = NA_real_,
                     projection_sec = NA_real_,
                     transform_sec = NA_real_,
                     total_sec = NA_real_,
                     negative_gradient_method = NA_character_,
                     parameters = list(),
                     metrics = NULL) {
  row <- data.frame(
    machine = Sys.info()[["nodename"]],
    dataset = dataset,
    method = method,
    variant = variant,
    backend_requested = backend_requested,
    backend_used = backend_used,
    status = status,
    error_message = error_message,
    n = as.integer(n),
    p = as.integer(p),
    seed = as.integer(seed),
    nn_backend = nn_backend,
    nn_sec = as.numeric(nn_sec),
    embedding_sec = as.numeric(embedding_sec),
    projection_sec = as.numeric(projection_sec),
    transform_sec = as.numeric(transform_sec),
    total_sec = as.numeric(total_sec),
    negative_gradient_method = negative_gradient_method,
    parameters_json = json_params(parameters),
    trustworthiness = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    label_knn_accuracy = NA_real_,
    silhouette = NA_real_,
    metric_sample_n = NA_integer_,
    stringsAsFactors = FALSE
  )
  if (!is.null(metrics)) {
    copy <- intersect(names(row), names(metrics))
    copy <- intersect(copy, c(
      "trustworthiness",
      "knn_preservation_15",
      "knn_preservation_30",
      "knn_preservation_50",
      "label_knn_accuracy",
      "silhouette",
      "metric_sample_n"
    ))
    for (nm in copy) row[[nm]] <- metrics[[nm]][[1L]]
  }
  row
}

method_report_rank <- function(method, variant = "", backend = "") {
  method <- as.character(method)
  variant <- as.character(variant)
  backend <- as.character(backend)
  family <- ifelse(method %in% c("opentsne", "Rtsne"), 1L,
                   ifelse(method %in% c("umap", "uwot::umap"), 2L, 3L))
  subtype <- ifelse(method == "opentsne" & variant == "full", 10L,
                    ifelse(method == "Rtsne", 20L,
                           ifelse(method == "opentsne", 30L,
                                  ifelse(method == "umap" & variant %in% c("fastEmbedR", "full"), 10L,
                                         ifelse(method == "umap", 20L,
                                                ifelse(method == "uwot::umap", 30L, 99L))))))
  backend_rank <- ifelse(backend == "cpu", 1L,
                         ifelse(backend == "metal", 2L,
                                ifelse(backend == "cuda", 3L, 9L)))
  family * 1000L + subtype * 10L + backend_rank
}

layout_report_rank <- function(names) {
  is_tsne <- grepl("OPENTSNE|RTSNE", names, ignore.case = TRUE)
  is_umap <- grepl("UMAP|UWOT", names, ignore.case = TRUE)
  family <- ifelse(is_tsne, 1L, ifelse(is_umap, 2L, 3L))
  subtype <- ifelse(grepl("OPENTSNE_FULL", names, ignore.case = TRUE), 10L,
                    ifelse(grepl("RTSNE", names, ignore.case = TRUE), 20L,
                           ifelse(grepl("OPENTSNE", names, ignore.case = TRUE), 30L,
                                  ifelse(grepl("FASTEMBEDR_UMAP_[A-Z]+$", names), 10L,
                                         ifelse(grepl("FASTEMBEDR_UMAP_LANDMARK", names), 20L,
                                                ifelse(grepl("UWOT", names, ignore.case = TRUE), 30L, 99L))))))
  backend <- ifelse(grepl("CPU", names), 1L,
                    ifelse(grepl("METAL", names), 2L,
                           ifelse(grepl("CUDA", names), 3L, 9L)))
  family * 1000L + subtype * 10L + backend
}

order_results_for_report <- function(results) {
  if (is.null(results) || nrow(results) == 0L) return(results)
  key <- method_report_rank(results$method, results$variant, results$backend_requested)
  results[order(key, results$method, results$variant, results$backend_requested), , drop = FALSE]
}

order_layouts_for_report <- function(layouts) {
  if (length(layouts) < 2L) return(layouts)
  layouts[order(layout_report_rank(names(layouts)), names(layouts))]
}

robust_plot_limits <- function(values, probs = c(0.002, 0.998)) {
  values <- values[is.finite(values)]
  if (!length(values)) return(c(-1, 1))
  limits <- as.numeric(stats::quantile(values, probs = probs, names = FALSE, na.rm = TRUE))
  if (!all(is.finite(limits)) || limits[1L] >= limits[2L]) {
    limits <- range(values, finite = TRUE)
  }
  if (!all(is.finite(limits)) || limits[1L] >= limits[2L]) {
    center <- if (is.finite(values[1L])) values[1L] else 0
    limits <- center + c(-1, 1)
  }
  pad <- 0.04 * diff(limits)
  limits + c(-pad, pad)
}

plot_layouts <- function(layouts, labels, path, seed, max_points = 20000L, point_cex = 0.28, dataset_label = "MNIST 70k") {
  if (length(layouts) == 0L) return(invisible(FALSE))
  layouts <- order_layouts_for_report(layouts)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  keep <- stratified_rows(labels, min(max_points, length(labels)), seed + 29L)
  png(path, width = 2400, height = 1800, res = 180)
  old <- par(no.readonly = TRUE)
  on.exit({ par(old); dev.off() }, add = TRUE)
  nr <- ceiling(sqrt(length(layouts)))
  nc <- ceiling(length(layouts) / nr)
  par(mfrow = c(nr, nc), mar = c(2.1, 2.1, 2.8, 0.6), oma = c(0, 0, 2, 0))
  pal <- grDevices::hcl.colors(length(levels(as.factor(labels))), "Dark 3")
  label_int <- as.integer(as.factor(labels))
  for (nm in names(layouts)) {
    y <- layouts[[nm]]
    if (is.null(y) || any(dim(y) < c(2L, 2L))) {
      plot.new()
      title(main = paste(nm, "failed"))
      next
    }
    plot(
      y[keep, 1L],
      y[keep, 2L],
      col = pal[label_int[keep]],
      pch = 16,
      cex = point_cex,
      xlim = robust_plot_limits(y[keep, 1L]),
      ylim = robust_plot_limits(y[keep, 2L]),
      xlab = "",
      ylab = "",
      axes = FALSE,
      main = nm
    )
    box(col = "grey70")
  }
  plot_title <- if (length(keep) < length(labels)) {
    paste0(dataset_label, " embeddings (", length(keep), "-point stratified plotting sample)")
  } else {
    paste0(dataset_label, " embeddings (all ", length(labels), " points)")
  }
  mtext(plot_title, outer = TRUE, cex = 1.2)
  invisible(TRUE)
}

suppressPackageStartupMessages(library(fastEmbedR))

cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
feature_source <- arg_value("feature-source", "pca50")
raw_cache_dir <- arg_value("raw-cache-dir", file.path("results", "dataset_cache"))
raw_cache <- arg_value("raw-cache", file.path(raw_cache_dir, "mnist_idx_70000_flattened.rds"))
out_dir <- arg_value("out-dir", file.path("results", "mnist70k_current_backends", Sys.info()[["nodename"]]))
n <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
seed <- arg_int("seed", 6L)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early-iter", 100L)
normal_iter <- arg_int("normal-iter", 150L)
umap_epochs <- arg_int("umap-epochs", 200L)
metric_n <- arg_int("metric-n", 5000L)
plot_n <- arg_int("plot-n", 20000L)
point_cex <- arg_num("point-cex", 0.28)
backends <- arg_csv("backends", "cpu,metal")
run_uwot <- arg_flag("run-uwot", TRUE)
run_umap <- arg_flag("run-umap", TRUE)
run_opentsne <- arg_flag("run-opentsne", TRUE)
run_rtsne <- arg_flag("run-rtsne", TRUE)
run_rtsne_internal <- arg_flag("run-rtsne-internal", FALSE)
run_landmark <- arg_flag("run-landmark", TRUE)
landmark_fraction <- arg_num("landmark-fraction", 0.5)
report_knn_only <- arg_flag("report-knn-only", FALSE)
report_unsupported <- arg_flag("report-unsupported", TRUE)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
feature_source <- tolower(feature_source)
if (feature_source %in% c("raw", "flattened", "pixels", "idx")) {
  mnist <- load_raw_mnist(raw_cache_dir, raw_cache)
  dataset <- "mnist70k_raw_flattened"
  dataset_label <- "MNIST 70k raw flattened 28x28"
} else if (feature_source %in% c("pca", "pca50")) {
  if (!file.exists(cache)) stop("MNIST PCA cache not found: ", cache, call. = FALSE)
  mnist <- readRDS(cache)
  dataset <- "mnist70k_pca50"
  dataset_label <- "MNIST 70k PCA50"
} else {
  stop("`feature-source` must be `raw` or `pca50`.", call. = FALSE)
}
if (!all(c("x", "labels") %in% names(mnist))) {
  stop("MNIST cache must contain `x` and `labels`.", call. = FALSE)
}
rows <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- mnist$x[rows, , drop = FALSE]
storage.mode(x) <- "double"
labels <- droplevels(factor(mnist$labels[rows]))
perplexity <- min(30L, floor(k / 3L), floor((nrow(x) - 1L) / 3L))

message("MNIST current backend benchmark")
message("  feature_source=", feature_source, " dataset=", dataset)
message("  n=", nrow(x), " p=", ncol(x), " k=", k, " perplexity=", perplexity)
message("  backends=", paste(backends, collapse = ","), " out_dir=", out_dir)

knn_backend_for <- function(backend) {
  switch(
    backend,
    cpu = "cpu_nndescent",
    metal = "metal_nndescent",
    cuda = "cuda_cuvs_nndescent",
    stop("Unsupported benchmark backend: ", backend, call. = FALSE)
  )
}

negative_for <- function(backend) {
  switch(
    backend,
    cpu = "fft",
    metal = "fft",
    cuda = "fft",
    "fft"
  )
}

resolved_negative_label <- function(requested, backend, n) {
  if (!identical(requested, "auto")) return(requested)
  "fft"
}

with_default_metal_umap <- function(expr) {
  force(expr)
}

rows_out <- list()
layouts <- list()
knn_cache <- list()

for (backend in backends) {
  knn_backend <- knn_backend_for(backend)
  message("KNN ", backend, " via ", knn_backend)
  knn_run <- safe_status(timed(fastEmbedR::nn(
    x,
    k = k + 1L,
    backend = knn_backend,
    n_threads = n_threads
  )))
  if (!identical(knn_run$status, "success")) {
    if (isTRUE(report_knn_only)) {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "knn", "knn_only", backend, backend, "failed", knn_run$error,
        nrow(x), ncol(x), seed, nn_backend = knn_backend,
        parameters = list(k = k, backend = backend, knn_backend = knn_backend)
      )
    }
    next
  }
  knn <- knn_run$value$value
  knn_sec <- knn_run$value$sec
  knn_cache[[backend]] <- knn
  if (isTRUE(report_knn_only)) {
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "knn", "knn_only", backend, attr(knn, "backend") %||% knn_backend,
      "success", NA_character_, nrow(x), ncol(x), seed,
      nn_backend = attr(knn, "backend") %||% knn_backend,
      nn_sec = knn_sec,
      total_sec = knn_sec,
      parameters = list(k = k, backend = backend, knn_backend = knn_backend)
    )
  }

  if (isTRUE(run_umap) && backend %in% c("cpu", "metal")) {
    message("fastEmbedR UMAP ", backend)
    umap_run <- safe_status(timed(with_default_metal_umap(
      fastEmbedR::umap_knn(
        knn,
        backend = backend,
        seed = seed
      )
    )))
    if (identical(umap_run$status, "success")) {
      layout <- as_layout(umap_run$value$value)
      cfg <- attr(layout, "fastEmbedR_config")
      layout_id <- paste0("MNIST70K_FASTEMBEDR_UMAP_", toupper(backend))
      layouts[[layout_id]] <- layout
      saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
      metrics <- score_layout(x, labels, layout, "fastembedr_umap", backend, seed, n_threads, metric_n, dataset)
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "umap", "fastEmbedR", backend, cfg$backend %||% backend,
        "success", NA_character_, nrow(x), ncol(x), seed,
        nn_backend = attr(knn, "backend") %||% knn_backend,
        nn_sec = knn_sec,
        embedding_sec = umap_run$value$sec,
        total_sec = knn_sec + umap_run$value$sec,
        parameters = list(k = k,
                          n_epochs = cfg$n_epochs %||% NA_integer_,
                          min_dist = cfg$min_dist %||% NA_real_,
                          negative_sample_rate = cfg$negative_sample_rate %||% NA_integer_,
                          learning_rate = cfg$learning_rate %||% NA_real_,
                          spectral_n_iter = cfg$spectral_n_iter %||% NA_integer_,
                          init_scale = cfg$init_scale %||% NA_real_,
                          backend = backend,
                          knn_backend = knn_backend,
                          optimizer = cfg$gpu_optimizer_mode %||% cfg$optimizer %||% NA_character_,
                          graph = cfg$graph_storage %||% cfg$graph %||% NA_character_,
                          graph_prep_backend = cfg$graph_prep_backend %||% NA_character_,
                          metal_optimizer_policy = "package_default",
                          gpu_residency = cfg$gpu_transfer_policy %||% NA_character_),
        metrics = metrics
      )
    } else {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "umap", "fastEmbedR", backend, backend, "failed", umap_run$error,
        nrow(x), ncol(x), seed,
        nn_backend = attr(knn, "backend") %||% knn_backend,
        nn_sec = knn_sec,
        parameters = list(k = k, backend = backend, knn_backend = knn_backend)
      )
    }
  }

  if (isTRUE(run_landmark) && isTRUE(run_umap) && backend %in% c("cpu", "metal")) {
    message("fastEmbedR UMAP landmark50 ", backend)
    umap_landmark_run <- safe_status(timed(with_default_metal_umap(
      fastEmbedR::landmark_umap(
        x,
        labels = labels,
        landmarks = landmark_fraction,
        n_neighbors = k,
        n_components = 2L,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = backend,
        transform_k = k,
        n_threads = n_threads,
        keep_knn = FALSE,
        verbose = FALSE
      )
    )))
    if (identical(umap_landmark_run$status, "success")) {
      fit <- umap_landmark_run$value$value
      layout <- fit$layout
      params <- fit$parameters
      layout_id <- paste0("MNIST70K_FASTEMBEDR_UMAP_LANDMARK50_", toupper(backend))
      layouts[[layout_id]] <- layout
      saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
      metrics <- score_layout(x, labels, layout, "fastembedr_umap_landmark50", backend, seed, n_threads, metric_n, dataset)
      m <- fit$metrics
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "umap", "landmark50", backend, params$backend %||% backend,
        "success", NA_character_, nrow(x), ncol(x), seed,
        nn_backend = params$nn_backend %||% attr(knn, "backend") %||% knn_backend,
        nn_sec = m$reference_knn_elapsed %||% NA_real_,
        embedding_sec = m$reference_optimizer_elapsed %||% NA_real_,
        projection_sec = m$landmark_projection_knn_elapsed %||% NA_real_,
        transform_sec = 0,
        total_sec = umap_landmark_run$value$sec,
        parameters = params,
        metrics = metrics
      )
    } else {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "umap", "landmark50", backend, backend, "failed", umap_landmark_run$error,
        nrow(x), ncol(x), seed,
        nn_backend = knn_backend,
        parameters = list(k = k, backend = backend, landmark_fraction = landmark_fraction)
      )
    }
  }

  negative <- negative_for(backend)
  if (isTRUE(run_opentsne)) {
    message("openTSNE full ", backend, " negative=", negative)
    full_run <- safe_status(timed(fastEmbedR::opentsne_knn(
      knn,
      n_neighbors = k,
      perplexity = perplexity,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      negative_gradient_method = negative,
      backend = backend,
      n_threads = n_threads,
      seed = seed
    )))
    if (identical(full_run$status, "success")) {
      layout <- as_layout(full_run$value$value)
      cfg <- attr(layout, "fastEmbedR_config")
      resolved_negative <- cfg$negative_gradient_method %||% negative
      layout_id <- paste0("MNIST70K_OPENTSNE_FULL_", toupper(backend), "_", toupper(resolved_negative))
      layouts[[layout_id]] <- layout
      saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
      metrics <- score_layout(x, labels, layout, "opentsne", backend, seed, n_threads, metric_n, dataset)
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "opentsne", "full", backend, cfg$backend %||% backend,
        "success", NA_character_, nrow(x), ncol(x), seed,
        nn_backend = attr(knn, "backend") %||% knn_backend,
        nn_sec = knn_sec,
        embedding_sec = full_run$value$sec,
        total_sec = knn_sec + full_run$value$sec,
        negative_gradient_method = cfg$negative_gradient_method %||% negative,
        parameters = list(k = k, perplexity = perplexity, early_iter = early_iter,
                          normal_iter = normal_iter, backend = backend,
                          knn_backend = knn_backend),
        metrics = metrics
      )
    } else {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "opentsne", "full", backend, backend, "failed", full_run$error,
        nrow(x), ncol(x), seed, nn_backend = attr(knn, "backend") %||% knn_backend,
        nn_sec = knn_sec, negative_gradient_method = negative,
        parameters = list(k = k, perplexity = perplexity, backend = backend,
                          knn_backend = knn_backend)
      )
    }
  }

  if (isTRUE(run_rtsne)) {
    if (identical(backend, "cpu")) {
      message("Rtsne_neighbors CPU")
      rtsne_run <- safe_status(timed({
        if (!requireNamespace("Rtsne", quietly = TRUE)) {
          stop("Package `Rtsne` is not installed.", call. = FALSE)
        }
        clean_knn <- fastEmbedR:::coerce_knn_input(knn)
        Rtsne::Rtsne_neighbors(
          index = clean_knn$indices,
          distance = clean_knn$distances,
          dims = 2L,
          perplexity = perplexity,
          theta = 0.5,
          max_iter = early_iter + normal_iter,
          stop_lying_iter = early_iter,
          mom_switch_iter = early_iter,
          num_threads = n_threads,
          verbose = FALSE
        )$Y
      }))
      if (identical(rtsne_run$status, "success")) {
        layout <- as_layout(rtsne_run$value$value)
        layout_id <- "MNIST70K_RTSNE_NEIGHBORS_CPU"
        layouts[[layout_id]] <- layout
        saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
        metrics <- score_layout(x, labels, layout, "Rtsne_neighbors", "cpu", seed, n_threads, metric_n, dataset)
        rows_out[[length(rows_out) + 1L]] <- make_row(
          dataset, "Rtsne", "Rtsne_neighbors", "cpu", "cpu",
          "success", NA_character_, nrow(x), ncol(x), seed,
          nn_backend = attr(knn, "backend") %||% knn_backend,
          nn_sec = knn_sec,
          embedding_sec = rtsne_run$value$sec,
          total_sec = knn_sec + rtsne_run$value$sec,
          parameters = list(k = k, perplexity = perplexity, theta = 0.5,
                            max_iter = early_iter + normal_iter,
                            stop_lying_iter = early_iter,
                            mom_switch_iter = early_iter,
                            n_threads = n_threads),
          metrics = metrics
        )
      } else {
        rows_out[[length(rows_out) + 1L]] <- make_row(
          dataset, "Rtsne", "Rtsne_neighbors", "cpu", "cpu", "failed",
          rtsne_run$error, nrow(x), ncol(x), seed,
          nn_backend = attr(knn, "backend") %||% knn_backend,
          nn_sec = knn_sec,
          parameters = list(k = k, perplexity = perplexity)
        )
      }
    } else if (identical(backend, "metal")) {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "Rtsne", "Rtsne_neighbors", "metal", "none",
        "not_supported",
        "Rtsne::Rtsne_neighbors has no native Metal backend. fastEmbedR does not run it on CPU and report it as Metal.",
        nrow(x), ncol(x), seed,
        parameters = list(k = k, perplexity = perplexity, backend = "metal")
      )
    }
  }

  if (isTRUE(run_landmark) && isTRUE(run_opentsne) && backend %in% c("cpu", "metal")) {
    message("openTSNE landmark50 ", backend, " negative=", negative)
    landmark_run <- safe_status(timed(fastEmbedR::landmark_tsne(
      x,
      labels = labels,
      landmarks = landmark_fraction,
      n_neighbors = k,
      perplexity = perplexity,
      standardize = FALSE,
      pca_dims = NULL,
      backend = backend,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      negative_gradient_method = negative,
      transform_k = k,
      transform_perplexity = min(15, perplexity),
      transform_iter = 0L,
      silhouette_sample = NULL,
      preserve_sample = NULL,
      keep_knn = FALSE,
      n_threads = n_threads,
      seed = seed
    )))
    if (identical(landmark_run$status, "success")) {
      fit <- landmark_run$value$value
      layout <- fit$layout
      params <- fit$parameters
      resolved_negative <- resolved_negative_label(
        params$negative_gradient_method %||% negative,
        backend,
        nrow(x)
      )
      layout_id <- paste0("MNIST70K_OPENTSNE_LANDMARK50_", toupper(backend), "_", toupper(resolved_negative))
      layouts[[layout_id]] <- layout
      saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
      metrics <- score_layout(x, labels, layout, "opentsne_landmark50", backend, seed, n_threads, metric_n, dataset)
      m <- fit$metrics
      projection_elapsed <- m$landmark_projection_knn_elapsed %||% NA_real_
      if (identical(params$projection_strategy, "gpu_resident_landmark_projection_transform")) {
        projection_elapsed <- NA_real_
      }
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "opentsne", "landmark50", backend, params$backend %||% backend,
        "success", NA_character_, nrow(x), ncol(x), seed,
        nn_backend = params$nn_backend %||% NA_character_,
        nn_sec = m$reference_knn_elapsed %||% NA_real_,
        embedding_sec = m$reference_optimizer_elapsed %||% NA_real_,
        projection_sec = projection_elapsed,
        transform_sec = m$transform_elapsed %||% NA_real_,
        total_sec = landmark_run$value$sec,
        negative_gradient_method = resolved_negative,
        parameters = params,
        metrics = metrics
      )
    } else {
      rows_out[[length(rows_out) + 1L]] <- make_row(
        dataset, "opentsne", "landmark50", backend, backend, "failed",
        landmark_run$error, nrow(x), ncol(x), seed,
        negative_gradient_method = negative,
        parameters = list(k = k, perplexity = perplexity, backend = backend,
                          landmark_fraction = landmark_fraction)
      )
    }
  } else if (isTRUE(run_landmark) && isTRUE(run_opentsne) && identical(backend, "cuda") && isTRUE(report_unsupported)) {
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "opentsne", "landmark50", "cuda", "none", "not_supported",
      "The one-call landmark helper cannot yet combine cuVS NN-descent KNN with the native CUDA optimizer without relabelling KNN; skipped to avoid reporting native CUDA NN-descent.",
      nrow(x), ncol(x), seed,
      nn_backend = "cuda_cuvs_nndescent",
      negative_gradient_method = "auto",
      parameters = list(k = k, backend = "cuda", knn_backend = "cuda_cuvs_nndescent")
    )
  }

}

if (isTRUE(run_uwot)) {
  message("uwot UMAP fast_sgd CPU reference")
  uwot_run <- safe_status(timed({
    if (!requireNamespace("uwot", quietly = TRUE)) {
      stop("Package `uwot` is not installed.", call. = FALSE)
    }
    set.seed(seed)
    uwot::umap(
      x,
      n_neighbors = k,
      min_dist = 0.01,
      metric = "euclidean",
      init = "spectral",
      fast_sgd = TRUE,
      n_threads = n_threads,
      n_sgd_threads = n_threads,
      ret_model = FALSE,
      verbose = FALSE
    )
  }))
  if (identical(uwot_run$status, "success")) {
    layout <- as_layout(uwot_run$value$value)
    layout_id <- "MNIST70K_UWOT_UMAP_FAST_SGD_CPU"
    layouts[[layout_id]] <- layout
    saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
    metrics <- score_layout(x, labels, layout, "uwot_umap_fast_sgd", "cpu", seed, n_threads, metric_n, dataset)
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "uwot::umap", "fast_sgd", "cpu", "cpu", "success", NA_character_,
      nrow(x), ncol(x), seed,
      nn_backend = "uwot_internal_annoy",
      embedding_sec = uwot_run$value$sec,
      total_sec = uwot_run$value$sec,
      parameters = list(k = k, min_dist = 0.01, fast_sgd = TRUE,
                        timing_note = "total_sec includes uwot internal NN, graph construction, and optimization; uwot::umap() does not expose a separate NN time",
                        n_threads = n_threads, n_sgd_threads = n_threads),
      metrics = metrics
    )
  } else {
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "uwot::umap", "fast_sgd", "cpu", "cpu", "failed", uwot_run$error,
      nrow(x), ncol(x), seed,
      nn_backend = "uwot_internal_annoy",
      parameters = list(k = k, min_dist = 0.01, fast_sgd = TRUE)
    )
  }
}

if (isTRUE(run_rtsne_internal)) {
  message("Rtsne CPU reference with Rtsne internal NN")
  rtsne_internal_run <- safe_status(timed({
    if (!requireNamespace("Rtsne", quietly = TRUE)) {
      stop("Package `Rtsne` is not installed.", call. = FALSE)
    }
    set.seed(seed)
    Rtsne::Rtsne(
      x,
      dims = 2L,
      initial_dims = ncol(x),
      perplexity = perplexity,
      theta = 0.5,
      check_duplicates = FALSE,
      pca = FALSE,
      normalize = FALSE,
      max_iter = early_iter + normal_iter,
      stop_lying_iter = early_iter,
      mom_switch_iter = early_iter,
      num_threads = n_threads,
      verbose = FALSE
    )$Y
  }))
  if (identical(rtsne_internal_run$status, "success")) {
    layout <- as_layout(rtsne_internal_run$value$value)
    layout_id <- "MNIST70K_RTSNE_INTERNAL_NN_CPU"
    layouts[[layout_id]] <- layout
    saveRDS(layout, file.path(out_dir, paste0(layout_id, ".rds")))
    metrics <- score_layout(x, labels, layout, "Rtsne_internal_nn", "cpu", seed, n_threads, metric_n, dataset)
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "Rtsne", "Rtsne_internal_nn", "cpu", "cpu",
      "success", NA_character_, nrow(x), ncol(x), seed,
      nn_backend = "Rtsne_internal_vptree",
      embedding_sec = rtsne_internal_run$value$sec,
      total_sec = rtsne_internal_run$value$sec,
      parameters = list(k = k, perplexity = perplexity, theta = 0.5,
                        max_iter = early_iter + normal_iter,
                        stop_lying_iter = early_iter,
                        mom_switch_iter = early_iter,
                        n_threads = n_threads,
                        timing_note = "total_sec includes Rtsne internal neighbour search and optimization; Rtsne::Rtsne() does not expose a separate NN time"),
      metrics = metrics
    )
  } else {
    rows_out[[length(rows_out) + 1L]] <- make_row(
      dataset, "Rtsne", "Rtsne_internal_nn", "cpu", "cpu", "failed",
      rtsne_internal_run$error, nrow(x), ncol(x), seed,
      nn_backend = "Rtsne_internal_vptree",
      parameters = list(k = k, perplexity = perplexity, theta = 0.5,
                        max_iter = early_iter + normal_iter)
    )
  }
}

results <- order_results_for_report(do.call(rbind, rows_out))
csv <- file.path(out_dir, "mnist70k_current_backends_results.csv")
latest <- file.path(out_dir, "latest_mnist70k_current_backends_results.csv")
write.csv(results, csv, row.names = FALSE)
write.csv(results, latest, row.names = FALSE)

plot_path <- file.path(out_dir, paste0("MNIST70K_CURRENT_BACKENDS_", Sys.info()[["nodename"]], "_seed", seed, ".png"))
plot_layouts(
  layouts,
  labels,
  plot_path,
  seed,
  max_points = plot_n,
  point_cex = point_cex,
  dataset_label = dataset_label
)

message("Results CSV: ", normalizePath(csv, winslash = "/", mustWork = FALSE))
message("Plot PNG: ", normalizePath(plot_path, winslash = "/", mustWork = FALSE))
table_out <- results[results$method != "knn", , drop = FALSE]
table_out <- table_out[!(
  table_out$method == "Rtsne" &
    table_out$backend_requested == "metal"
), , drop = FALSE]
table_out <- order_results_for_report(table_out)
table_out$method_label <- ifelse(
  table_out$variant %in% c("full", "fastEmbedR", "Rtsne_neighbors", "fast_sgd"),
  table_out$method,
  paste(table_out$method, table_out$variant)
)
contract <- data.frame(
  machine = table_out$machine,
  method = table_out$method_label,
  backend = table_out$backend_requested,
  `NN sec` = table_out$nn_sec,
  `embed sec` = table_out$embedding_sec,
  `proj+transform sec` = rowSums(
    cbind(table_out$projection_sec, table_out$transform_sec),
    na.rm = TRUE
  ),
  trust = table_out$trustworthiness,
  status = table_out$status,
  error_message = table_out$error_message,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
contract$`proj+transform sec`[is.na(table_out$projection_sec) & is.na(table_out$transform_sec)] <- NA_real_
contract_csv <- file.path(out_dir, "mnist70k_contract_table.csv")
write.csv(contract, contract_csv, row.names = FALSE)
message("Contract table CSV: ", normalizePath(contract_csv, winslash = "/", mustWork = FALSE))
print(contract[, c("machine", "method", "backend", "NN sec", "embed sec",
                   "proj+transform sec", "trust", "status")],
      row.names = FALSE)
