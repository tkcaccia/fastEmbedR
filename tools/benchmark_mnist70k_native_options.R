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

arg_csv <- function(name, default) {
  value <- arg_value(name, default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
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
  elapsed <- system.time({ value <- force(expr) })[["elapsed"]]
  list(value = value, sec = as.numeric(elapsed))
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

as_layout <- function(x) {
  if (inherits(x, "fastEmbedR_embedding")) x <- x$layout
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x[, 1:2, drop = FALSE]
}

label_colors <- function(labels, alpha = 0.48) {
  labels <- factor(labels)
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  stats::setNames(grDevices::adjustcolor(pal, alpha.f = alpha), levels(labels))[as.character(labels)]
}

empty_row <- function(dataset,
                      method,
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
                      nn_backend = NA_character_,
                      nn_status = NA_character_,
                      nn_sec = NA_real_,
                      embedding_sec = NA_real_,
                      total_sec = NA_real_) {
  data.frame(
    dataset = dataset,
    method = method,
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
    nn_backend = nn_backend,
    nn_status = nn_status,
    nn_sec = as.numeric(nn_sec),
    embedding_sec = as.numeric(embedding_sec),
    total_sec = as.numeric(total_sec),
    trustworthiness = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    optimizer = NA_character_,
    probabilities = NA_character_,
    repulsion = NA_character_,
    plot_id = NA_character_,
    plot_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

score_layout <- function(x, layout, labels, reference_nn, seed, method, backend) {
  fastEmbedR::evaluate_embedding(
    x,
    layout,
    labels = labels,
    k = c(15L, 30L, 50L),
    primary_k = 30L,
    reference_nn = reference_nn,
    sample_size_for_global_metrics = min(3000L, nrow(x)),
    sample_size_for_local_metrics = min(3000L, nrow(x)),
    use_cache = FALSE,
    seed = seed,
    method = method,
    backend = backend,
    dataset = "mnist70k_native_options"
  )
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

known_skip <- function(n, backend, negative_gradient_method) {
  if (identical(negative_gradient_method, "exact") && n > 6000L) {
    return("exact dense openTSNE is intentionally skipped for MNIST 70k because it is O(n^2).")
  }
  NULL
}

run_opentsne_option <- function(x,
                                labels,
                                knn,
                                nn_row,
                                backend,
                                negative_gradient_method,
                                perplexity,
                                early_iter,
                                normal_iter,
                                n_threads,
                                seed,
                                host,
                                device,
                                out_dir) {
  option_id <- paste("opentsne", backend, negative_gradient_method, sep = "_")
  skip <- known_skip(nrow(x), backend, negative_gradient_method)
  if (!is.null(skip)) {
    return(list(
      row = empty_row(
        dataset = "mnist70k_pca50",
        method = "opentsne",
        option_id = option_id,
        backend_requested = backend,
        backend_used = NA_character_,
        negative_gradient_method = negative_gradient_method,
        status = "skipped_infeasible",
        error_message = skip,
        n = nrow(x),
        p = ncol(x),
        k = ncol(knn$indices),
        perplexity = perplexity,
        seed = seed,
        host = host,
        device = device,
        nn_backend = nn_row$backend_used,
        nn_status = nn_row$status,
        nn_sec = nn_row$elapsed_sec
      ),
      layout = NULL
    ))
  }
  tryCatch({
    measured <- timed(fastEmbedR::opentsne_knn(
      knn,
      perplexity = perplexity,
      early_exaggeration_iter = early_iter,
      n_iter = normal_iter,
      learning_rate = "auto",
      early_exaggeration = "auto",
      negative_gradient_method = negative_gradient_method,
      theta = 0.5,
      n_threads = n_threads,
      backend = backend,
      seed = seed
    ))
    layout <- as_layout(measured$value)
    cfg <- attr(layout, "fastEmbedR_config")
    backend_used <- cfg$backend %||% backend
    score_backend <- if (backend_used %in% c("metal", "cuda")) backend_used else "cpu"
    metrics <- score_layout(
      x,
      layout,
      labels,
      reference_nn = knn,
      seed = seed,
      method = option_id,
      backend = score_backend
    )
    row <- empty_row(
      dataset = "mnist70k_pca50",
      method = "opentsne",
      option_id = option_id,
      backend_requested = backend,
      backend_used = backend_used,
      negative_gradient_method = cfg$negative_gradient_method %||% negative_gradient_method,
      status = "success",
      error_message = NA_character_,
      n = nrow(x),
      p = ncol(x),
      k = ncol(knn$indices),
      perplexity = perplexity,
      seed = seed,
      host = host,
      device = device,
      nn_backend = nn_row$backend_used,
      nn_status = nn_row$status,
      nn_sec = nn_row$elapsed_sec,
      embedding_sec = measured$sec,
      total_sec = measured$sec + nn_row$elapsed_sec
    )
    row$trustworthiness <- metrics$trustworthiness
    row$knn_preservation_15 <- metrics$knn_preservation_15
    row$knn_preservation_30 <- metrics$knn_preservation_30
    row$knn_preservation_50 <- metrics$knn_preservation_50
    row$silhouette <- metrics$silhouette
    row$label_knn_accuracy <- metrics$label_knn_accuracy
    row$optimizer <- cfg$optimizer %||% NA_character_
    row$probabilities <- cfg$probabilities %||% NA_character_
    row$repulsion <- cfg$repulsion %||% NA_character_
    list(row = row, layout = layout)
  }, error = function(e) {
    list(
      row = empty_row(
        dataset = "mnist70k_pca50",
        method = "opentsne",
        option_id = option_id,
        backend_requested = backend,
        backend_used = NA_character_,
        negative_gradient_method = negative_gradient_method,
        status = "failed",
        error_message = conditionMessage(e),
        n = nrow(x),
        p = ncol(x),
        k = ncol(knn$indices),
        perplexity = perplexity,
        seed = seed,
        host = host,
        device = device,
        nn_backend = nn_row$backend_used,
        nn_status = nn_row$status,
        nn_sec = nn_row$elapsed_sec
      ),
      layout = NULL
    )
  })
}

run_knn <- function(x, backend, k, n_threads, host, device) {
  tryCatch({
    measured <- timed(fastEmbedR::nn(x, k = k + 1L, backend = backend, n_threads = n_threads))
    raw <- measured$value
    knn <- list(
      indices = raw$indices[, -1L, drop = FALSE],
      distances = raw$distances[, -1L, drop = FALSE]
    )
    attr(knn, "backend") <- attr(raw, "backend")
    attr(knn, "exact") <- attr(raw, "exact")
    row <- data.frame(
      host = host,
      device = device,
      backend_requested = backend,
      backend_used = attr(raw, "backend") %||% backend,
      status = "success",
      error_message = NA_character_,
      n = nrow(x),
      p = ncol(x),
      k = k,
      elapsed_sec = measured$sec,
      exact = isTRUE(attr(raw, "exact")),
      stringsAsFactors = FALSE
    )
    list(row = row, knn = knn)
  }, error = function(e) {
    list(
      row = data.frame(
        host = host,
        device = device,
        backend_requested = backend,
        backend_used = NA_character_,
        status = "failed",
        error_message = conditionMessage(e),
        n = nrow(x),
        p = ncol(x),
        k = k,
        elapsed_sec = NA_real_,
        exact = NA,
        stringsAsFactors = FALSE
      ),
      knn = NULL
    )
  })
}

select_knn_for_backend <- function(backend, knn_results) {
  preferred <- switch(
    backend,
    cpu = c("cpu_nndescent", "cpu_ivf", "cpu_annoy", "cpu"),
    metal = c("metal_nndescent", "metal_ivf", "metal"),
      cuda = c("cuda_cuvs_nndescent", "cuda_ivf", "cuda"),
    character(0L)
  )
  all_names <- names(knn_results)
  candidates <- unique(c(preferred, all_names))
  for (id in candidates) {
    if (id %in% all_names && !is.null(knn_results[[id]]$knn)) {
      return(knn_results[[id]])
    }
  }
  NULL
}

seed <- arg_int("seed", 6L)
n <- arg_int("n", 70000L)
k <- arg_int("k", 50L)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early-iter", 100L)
normal_iter <- arg_int("normal-iter", 150L)
cache <- arg_value(
  "cache",
  "/Users/stefano/Documents/fastEmbedR-results/current_best_umap_tsne_full/cache/mnist_max_all_pca_50_seed_6.rds"
)
out_dir <- arg_value("out-dir", file.path("results", "mnist70k_native_options"))
backends <- arg_csv("backends", "cpu,metal")
knn_backends <- arg_csv("knn-backends", "cpu_nndescent")
negative_methods <- arg_csv("negative-methods", "auto,exact,fft")
host <- arg_value("host", Sys.info()[["nodename"]])
device <- arg_value("device", paste(capture.output(print(fastEmbedR::backend_info())), collapse = " | "))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(cache)) stop("MNIST cache not found: ", cache, call. = FALSE)
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
  "MNIST native option benchmark: n=", nrow(x),
  " p=", ncol(x),
  " k=", k,
  " perplexity=", perplexity,
  " backends=", paste(backends, collapse = ","),
  " knn_backends=", paste(knn_backends, collapse = ",")
)

knn_results <- lapply(knn_backends, function(backend) {
  message("Running KNN backend=", backend)
  run_knn(x, backend, k = k, n_threads = n_threads, host = host, device = device)
})
knn_rows <- do.call(rbind, lapply(knn_results, `[[`, "row"))
names(knn_results) <- knn_backends
write.csv(knn_rows, file.path(out_dir, "mnist70k_knn_options.csv"), row.names = FALSE)

primary_idx <- which(vapply(knn_results, function(z) !is.null(z$knn), logical(1L)))[1L]
if (is.na(primary_idx)) stop("No KNN backend succeeded; cannot run embeddings.", call. = FALSE)
primary_knn_row <- knn_results[[primary_idx]]$row

umap_rows <- lapply(backends, function(backend) {
  selected_knn <- select_knn_for_backend(backend, knn_results)
  selected_row <- if (is.null(selected_knn)) primary_knn_row else selected_knn$row
  empty_row(
    dataset = "mnist70k_pca50",
    method = "umap",
    option_id = paste("umap", backend, sep = "_"),
    backend_requested = backend,
    backend_used = NA_character_,
    negative_gradient_method = NA_character_,
    status = "not_available",
    error_message = "fastEmbedR currently does not export UMAP; the cleaned package is KNN + native openTSNE focused.",
    n = nrow(x),
    p = ncol(x),
    k = k,
    perplexity = NA_real_,
    seed = seed,
    host = host,
    device = device,
    nn_backend = selected_row$backend_used,
    nn_status = selected_row$status,
    nn_sec = selected_row$elapsed_sec
  )
})

embedding_results <- list()
layouts <- list()
for (backend in backends) {
  selected_knn <- select_knn_for_backend(backend, knn_results)
  if (is.null(selected_knn)) {
    selected_knn <- knn_results[[primary_idx]]
  }
  for (neg in negative_methods) {
    message("Running opentsne backend=", backend, " negative_gradient_method=", neg)
    result <- run_opentsne_option(
      x,
      labels,
      selected_knn$knn,
      selected_knn$row,
      backend = backend,
      negative_gradient_method = neg,
      perplexity = perplexity,
      early_iter = early_iter,
      normal_iter = normal_iter,
      n_threads = n_threads,
      seed = seed,
      host = host,
      device = device,
      out_dir = out_dir
    )
    embedding_results[[result$row$option_id]] <- result$row
    layouts[[result$row$option_id]] <- result$layout
  }
}

all_rows <- do.call(rbind, c(umap_rows, embedding_results))
plot_id <- paste0("MNIST70K_NATIVE_OPTIONS_", gsub("[^A-Za-z0-9]+", "_", host), "_seed", seed)
plot_path <- file.path(out_dir, paste0(plot_id, ".png"))
plot_layouts(layouts, all_rows, labels, plot_path)
if (file.exists(plot_path)) {
  all_rows$plot_id[all_rows$status == "success"] <- plot_id
  all_rows$plot_path[all_rows$status == "success"] <- normalizePath(plot_path, mustWork = FALSE)
}

layout_dir <- file.path(out_dir, "layouts")
dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
for (id in names(layouts)) {
  if (!is.null(layouts[[id]])) {
    saveRDS(layouts[[id]], file.path(layout_dir, paste0(id, ".rds")), version = 2)
  }
}

csv_path <- file.path(out_dir, "mnist70k_embedding_options.csv")
write.csv(all_rows, csv_path, row.names = FALSE)
write.csv(all_rows, file.path(out_dir, "latest_mnist70k_embedding_options.csv"), row.names = FALSE)

message("KNN CSV: ", normalizePath(file.path(out_dir, "mnist70k_knn_options.csv"), mustWork = FALSE))
message("Embedding CSV: ", normalizePath(csv_path, mustWork = FALSE))
if (file.exists(plot_path)) message("Plot ID: ", plot_id, "\nPlot: ", normalizePath(plot_path, mustWork = FALSE))
print(all_rows[, c(
  "method", "option_id", "backend_requested", "backend_used",
  "negative_gradient_method", "status", "nn_backend", "nn_sec",
  "embedding_sec", "total_sec", "trustworthiness",
  "knn_preservation_30", "label_knn_accuracy", "error_message"
)], row.names = FALSE)
