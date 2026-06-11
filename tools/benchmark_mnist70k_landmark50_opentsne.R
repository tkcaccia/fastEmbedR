#!/usr/bin/env Rscript

arg_value <- function(name, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

arg_int <- function(name, default) {
  as.integer(as.numeric(arg_value(name, as.character(default))))
}

arg_num <- function(name, default) {
  as.numeric(arg_value(name, as.character(default)))
}

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

`%||%` <- function(x, y) if (is.null(x)) y else x

repo_root <- normalizePath(getwd(), mustWork = TRUE)
load_source <- identical(Sys.getenv("FASTEMBEDR_LOAD_SOURCE"), "1")
if (load_source && requireNamespace("devtools", quietly = TRUE)) {
  suppressPackageStartupMessages(devtools::load_all(repo_root, quiet = TRUE))
} else {
  suppressPackageStartupMessages(library(fastEmbedR))
}

timed <- function(expr) {
  invisible(gc())
  value <- NULL
  sec <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(sec))
}

as_layout <- function(x) {
  if (inherits(x, "fastEmbedR_embedding")) x <- x$layout
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x[, 1:2, drop = FALSE]
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

label_colors <- function(labels, alpha = 0.50) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

empty_row <- function(dataset,
                      option_id,
                      backend_requested,
                      backend_used,
                      negative_gradient_method,
                      status,
                      error_message,
                      n,
                      p,
                      k,
                      perplexity,
                      seed,
                      host,
                      device,
                      n_landmarks,
                      landmark_fraction,
                      selection_sec,
                      selection_method) {
  data.frame(
    dataset = dataset,
    method = "landmark_opentsne",
    option_id = option_id,
    backend_requested = backend_requested,
    backend_used = backend_used,
    negative_gradient_method = negative_gradient_method,
    status = status,
    error_message = error_message,
    n = as.integer(n),
    p = as.integer(p),
    k = as.integer(k),
    perplexity = as.numeric(perplexity),
    seed = as.integer(seed),
    host = host,
    device = device,
    n_landmarks = as.integer(n_landmarks),
    landmark_fraction = as.numeric(landmark_fraction),
    selection_sec = as.numeric(selection_sec),
    selection_method = selection_method,
    landmark_reference_total_sec = NA_real_,
    landmark_knn_sec = NA_real_,
    landmark_embedding_sec = NA_real_,
    projection_knn_sec = NA_real_,
    projection_transform_sec = NA_real_,
    projection_total_sec = NA_real_,
    scoring_knn_sec = NA_real_,
    algorithm_total_sec = NA_real_,
    timed_total_without_scoring_sec = NA_real_,
    trustworthiness = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    projection_nn_backend = NA_character_,
    projection_strategy = NA_character_,
    transform_backend = NA_character_,
    transform_repulsion = NA_character_,
    layout_path = NA_character_,
    plot_id = NA_character_,
    plot_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

elapsed_row <- function(timings, row_name) {
  if (is.null(timings) || !row_name %in% rownames(timings)) return(NA_real_)
  as.numeric(timings[row_name, "elapsed"])
}

score_layout_sample <- function(x, layout, labels, seed, option_id, backend, n_threads) {
  keep <- stratified_rows(labels, min(3000L, nrow(x)), seed + 203L)
  fastEmbedR::evaluate_embedding(
    x[keep, , drop = FALSE],
    layout[keep, , drop = FALSE],
    labels = labels[keep],
    k = c(15L, 30L, 50L),
    primary_k = 30L,
    sample_size_for_global_metrics = min(3000L, length(keep)),
    sample_size_for_local_metrics = min(3000L, length(keep)),
    use_cache = FALSE,
    seed = seed,
    method = option_id,
    backend = backend,
    n_threads = n_threads,
    dataset = "mnist70k_landmark50"
  )
}

run_landmark_option <- function(x,
                                labels,
                                landmark_indices,
                                selection_sec,
                                selection_method,
                                backend,
                                negative_gradient_method,
                                k,
                                perplexity,
                                early_iter,
                                normal_iter,
                                transform_iter,
                                n_threads,
                                seed,
                                host,
                                device,
                                out_dir,
                                plot_id,
                                plot_path) {
  option_id <- paste("landmark50", backend, negative_gradient_method, sep = "_")
  n <- nrow(x)
  n_landmarks <- length(landmark_indices)
  landmark_fraction <- n_landmarks / n
  tryCatch({
    measured <- timed(fastEmbedR::landmark_tsne(
      x,
      labels = labels,
      landmarks = landmark_indices,
      n_neighbors = k,
      perplexity = perplexity,
      n_components = 2L,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      transform_k = k,
      transform_perplexity = min(perplexity, max(1, floor(k / 3))),
      transform_iter = transform_iter,
      transform_early_exaggeration_iter = 0L,
      initialization = "median",
      silhouette_sample = NULL,
      preserve_sample = NULL,
      preserve_k = NULL,
      keep_knn = FALSE,
      verbose = FALSE,
      n_threads = n_threads,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      early_exaggeration = "auto",
      negative_gradient_method = negative_gradient_method
    ))
    fit <- measured$value
    layout <- as_layout(fit)
    reference_fit <- fit$landmarks$reference_fit
    reference_timings <- reference_fit$timings
    timings <- fit$timings
    params <- fit$parameters
    backend_used <- params$backend %||% attr(layout, "backend") %||% backend
    if (backend %in% c("metal", "cuda") && !identical(as.character(backend_used), backend)) {
      stop(
        "Requested backend ", backend,
        " but landmark transform backend_used was ", backend_used,
        call. = FALSE
      )
    }

    metric_backend <- if (backend_used %in% c("metal", "cuda")) backend_used else "cpu"
    metrics <- score_layout_sample(
      x,
      layout,
      labels,
      seed = seed,
      option_id = option_id,
      backend = metric_backend,
      n_threads = n_threads
    )

    layout_dir <- file.path(out_dir, "layouts")
    dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
    layout_path <- file.path(layout_dir, paste0(option_id, ".rds"))
    saveRDS(layout, layout_path, version = 2)

    row <- empty_row(
      dataset = "mnist70k_pca50",
      option_id = option_id,
      backend_requested = backend,
      backend_used = as.character(backend_used),
      negative_gradient_method = negative_gradient_method,
      status = "success",
      error_message = NA_character_,
      n = n,
      p = ncol(x),
      k = k,
      perplexity = perplexity,
      seed = seed,
      host = host,
      device = device,
      n_landmarks = n_landmarks,
      landmark_fraction = landmark_fraction,
      selection_sec = selection_sec,
      selection_method = selection_method
    )
    row$landmark_reference_total_sec <- elapsed_row(timings, "reference_embedding")
    row$landmark_knn_sec <- elapsed_row(reference_timings, "knn")
    row$landmark_embedding_sec <- elapsed_row(reference_timings, "embedding")
    row$projection_knn_sec <- elapsed_row(timings, "landmark_projection_knn")
    row$projection_transform_sec <- elapsed_row(timings, "transform")
    row$projection_total_sec <- row$projection_knn_sec + row$projection_transform_sec
    row$scoring_knn_sec <- elapsed_row(timings, "scoring_knn")
    row$algorithm_total_sec <- measured$sec + selection_sec
    row$timed_total_without_scoring_sec <- selection_sec +
      row$landmark_reference_total_sec +
      row$projection_knn_sec +
      row$projection_transform_sec
    row$trustworthiness <- metrics$trustworthiness[[1L]]
    row$knn_preservation_15 <- metrics$knn_preservation_15[[1L]]
    row$knn_preservation_30 <- metrics$knn_preservation_30[[1L]]
    row$knn_preservation_50 <- metrics$knn_preservation_50[[1L]]
    row$silhouette <- metrics$silhouette[[1L]]
    row$label_knn_accuracy <- metrics$label_knn_accuracy[[1L]]
    row$projection_nn_backend <- params$projection_nn_backend %||% NA_character_
    row$projection_strategy <- params$projection_strategy %||% NA_character_
    row$transform_backend <- params$transform_backend %||% NA_character_
    row$transform_repulsion <- params$transform_repulsion %||% NA_character_
    row$layout_path <- layout_path
    row$plot_id <- plot_id
    row$plot_path <- plot_path
    list(row = row, layout = layout)
  }, error = function(e) {
    row <- empty_row(
      dataset = "mnist70k_pca50",
      option_id = option_id,
      backend_requested = backend,
      backend_used = NA_character_,
      negative_gradient_method = negative_gradient_method,
      status = if (backend %in% c("metal", "cuda")) "backend_unavailable_or_failed" else "failed",
      error_message = conditionMessage(e),
      n = n,
      p = ncol(x),
      k = k,
      perplexity = perplexity,
      seed = seed,
      host = host,
      device = device,
      n_landmarks = n_landmarks,
      landmark_fraction = landmark_fraction,
      selection_sec = selection_sec,
      selection_method = selection_method
    )
    list(row = row, layout = NULL)
  })
}

plot_layouts <- function(layouts, rows, labels, out_path) {
  ok <- names(layouts)[vapply(layouts, function(x) !is.null(x), logical(1L))]
  if (!length(ok)) return(invisible(FALSE))
  n_col <- min(3L, length(ok))
  n_row <- ceiling(length(ok) / n_col)
  grDevices::png(out_path, width = 620 * n_col, height = 540 * n_row, res = 130)
  old <- graphics::par(mfrow = c(n_row, n_col), mar = c(3, 3, 4.0, 0.8))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  cols <- label_colors(labels)
  for (id in ok) {
    layout <- layouts[[id]]
    row <- rows[rows$option_id == id, , drop = FALSE]
    graphics::plot(
      layout[, 1L],
      layout[, 2L],
      col = cols,
      pch = 16,
      cex = 0.055,
      xlab = "Dim 1",
      ylab = "Dim 2",
      main = sprintf(
        "%s\nlandmark %.2fs proj %.2fs trust %.3f acc %.3f",
        id,
        row$landmark_embedding_sec,
        row$projection_total_sec,
        row$trustworthiness,
        row$label_knn_accuracy
      )
    )
  }
  invisible(TRUE)
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early-iter", 100L)
normal_iter <- arg_int("normal-iter", 150L)
transform_iter <- arg_int("transform-iter", early_iter + normal_iter)
landmark_fraction <- arg_num("landmark-fraction", 0.5)
backends <- arg_csv("backends", "cpu,metal")
negative_methods <- arg_csv("negative-methods", "fft")
host <- arg_value("host", Sys.info()[["nodename"]])
cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
out_dir <- arg_value("out-dir", file.path("results", "mnist70k_landmark50", host))

if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mnist <- readRDS(cache)
take <- stratified_rows(mnist$labels, min(n, nrow(mnist$x)), seed)
x <- as.matrix(mnist$x[take, , drop = FALSE])
storage.mode(x) <- "double"
labels <- droplevels(mnist$labels[take])
perplexity <- arg_num("perplexity", min(30, floor((nrow(x) - 1L) / 3), floor(k / 3)))

device <- paste(capture.output(fastEmbedR::backend_info()), collapse = " | ")
plot_id <- paste0("MNIST70K_LANDMARK50_", host, "_seed", seed)
plot_path <- file.path(out_dir, paste0(plot_id, ".png"))

message(
  "MNIST landmark50 benchmark: n=", nrow(x),
  " p=", ncol(x),
  " landmarks=", landmark_fraction,
  " k=", k,
  " perplexity=", perplexity,
  " backends=", paste(backends, collapse = ","),
  " negative_methods=", paste(negative_methods, collapse = ",")
)

selection <- timed(fastEmbedR:::resolve_landmarks(landmark_fraction, x, seed))
landmark_indices <- selection$value
selection_method <- attr(landmark_indices, "selection_method")
if (is.null(selection_method)) selection_method <- "indices"
message("Selected ", length(landmark_indices), " landmarks with ", selection_method, " in ", round(selection$sec, 3), " sec")

rows <- list()
layouts <- list()
for (backend in backends) {
  for (negative_method in negative_methods) {
    if (identical(backend, "cpu") && !negative_method %in% c("auto", "fft")) next
    if (backend %in% c("metal", "cuda")) next
    message("Running landmark option backend=", backend, " negative_gradient_method=", negative_method)
    out <- run_landmark_option(
      x = x,
      labels = labels,
      landmark_indices = landmark_indices,
      selection_sec = selection$sec,
      selection_method = selection_method,
      backend = backend,
      negative_gradient_method = negative_method,
      k = k,
      perplexity = perplexity,
      early_iter = early_iter,
      normal_iter = normal_iter,
      transform_iter = transform_iter,
      n_threads = n_threads,
      seed = seed,
      host = host,
      device = device,
      out_dir = out_dir,
      plot_id = plot_id,
      plot_path = plot_path
    )
    rows[[length(rows) + 1L]] <- out$row
    if (!is.null(out$layout)) layouts[[out$row$option_id[[1L]]]] <- out$layout
    write.csv(do.call(rbind, rows), file.path(out_dir, "mnist70k_landmark50_latest.csv"), row.names = FALSE)
  }
}

results <- do.call(rbind, rows)
write.csv(results, file.path(out_dir, "mnist70k_landmark50_results.csv"), row.names = FALSE)
plot_layouts(layouts, results, labels, plot_path)
writeLines(capture.output(fastEmbedR::backend_info()), file.path(out_dir, "backend_info.txt"))

message("Results CSV: ", normalizePath(file.path(out_dir, "mnist70k_landmark50_results.csv"), winslash = "/", mustWork = FALSE))
message("Plot ID: ", plot_id)
message("Plot: ", normalizePath(plot_path, winslash = "/", mustWork = FALSE))

print(results[, c(
  "option_id", "backend_requested", "backend_used", "negative_gradient_method",
  "status", "selection_sec", "landmark_knn_sec", "landmark_embedding_sec",
  "projection_knn_sec", "projection_transform_sec", "projection_total_sec",
  "timed_total_without_scoring_sec", "trustworthiness", "knn_preservation_30",
  "label_knn_accuracy", "projection_nn_backend", "projection_strategy",
  "error_message"
), drop = FALSE])
