#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[kv[[1L]]]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- normalizePath(args$data_root %||% "/Users/stefano/Documents/fastEmbedR/Data", mustWork = FALSE)
out_dir <- normalizePath(args$out_dir %||% file.path("results", "mit_umap_validation", format(Sys.time(), "%Y%m%d_%H%M%S")), mustWork = FALSE)
datasets <- strsplit(args$datasets %||% "USPS,COIL20,MetRef,MNIST,FashionMNIST", ",", fixed = TRUE)[[1L]]
datasets <- trimws(datasets[nzchar(datasets)])
max_n_arg <- args$max_n %||% "10000"
max_n <- if (tolower(as.character(max_n_arg)) %in% c("all", "full", "none", "0")) {
  Inf
} else {
  as.integer(max_n_arg)
}
k <- as.integer(args$k %||% "15")
seed <- as.integer(args$seed %||% "4")
n_threads <- as.integer(args$n_threads %||% "4")
nn_backend <- args$nn_backend %||% "faiss_hnsw"
include_reference <- isTRUE(as.logical(args$include_reference %||% TRUE))
if (length(max_n) != 1L || is.na(max_n) || (!is.infinite(max_n) && max_n < 50L)) max_n <- 10000L
if (is.na(k) || k < 2L) k <- 15L
if (is.na(seed)) seed <- 4L
if (is.na(n_threads) || n_threads < 1L) n_threads <- 4L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(fastEmbedR)
  library(faissR)
})

dataset_path <- function(dataset) {
  candidates <- if (identical(dataset, "FlowRepository_FR-FCM-ZYRM_files")) {
    file.path(data_root, dataset, "van_unen_FR-FCM-ZYRM.RData")
  } else {
    file.path(data_root, dataset, paste0(dataset, ".RData"))
  }
  if (!file.exists(candidates)) {
    hits <- list.files(file.path(data_root, dataset), pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
    hits <- hits[!grepl("pca|nn|manifest|summary|backup", basename(hits), ignore.case = TRUE)]
    if (length(hits)) return(hits[[1L]])
  }
  candidates
}

load_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  names <- load(path, envir = env)
  for (nm in names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.list(obj) && !is.null(obj$data)) {
      return(list(data = obj$data, labels = obj$labels %||% NULL))
    }
  }
  stop("No list with `$data` found in ", path, call. = FALSE)
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

prepare_x <- function(x) {
  x <- as_numeric_matrix(x)
  finite_cols <- vapply(seq_len(ncol(x)), function(j) all(is.finite(x[, j])), logical(1L))
  x <- x[, finite_cols, drop = FALSE]
  sds <- apply(x, 2L, stats::sd)
  keep <- is.finite(sds) & sds > 0
  x <- x[, keep, drop = FALSE]
  x <- scale(x, center = TRUE, scale = TRUE)
  x[!is.finite(x)] <- 0
  storage.mode(x) <- "double"
  x
}

subset_rows <- function(x, labels, max_n, seed) {
  n <- nrow(x)
  if (is.infinite(max_n)) return(list(x = x, labels = labels))
  if (n <= max_n) return(list(x = x, labels = labels))
  set.seed(seed)
  if (!is.null(labels)) {
    labels_f <- as.factor(labels)
    per_class <- split(seq_len(n), labels_f)
    take <- unlist(lapply(per_class, function(idx) {
      quota <- max(1L, floor(max_n * length(idx) / n))
      sample(idx, min(length(idx), quota))
    }), use.names = FALSE)
    if (length(take) < max_n) {
      rest <- setdiff(seq_len(n), take)
      take <- c(take, sample(rest, min(length(rest), max_n - length(take))))
    }
    take <- sort(take[seq_len(min(length(take), max_n))])
  } else {
    take <- sort(sample.int(n, max_n))
  }
  list(x = x[take, , drop = FALSE], labels = if (is.null(labels)) NULL else labels[take])
}

plot_layout <- function(layout, labels, path, title) {
  png(path, width = 1800, height = 1400, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 3, 1), bg = "white")
  if (is.null(labels)) {
    plot(layout, pch = 20, cex = 0.32, col = "#1f77b4", xlab = "", ylab = "", axes = FALSE, main = title)
  } else {
    labs <- as.factor(labels)
    cols <- grDevices::hcl.colors(nlevels(labs), "Dark 3")[as.integer(labs)]
    plot(layout, pch = 20, cex = 0.32, col = cols, xlab = "", ylab = "", axes = FALSE, main = title)
  }
  box()
}

plot_layout_panel <- function(layouts, labels, path, title) {
  png(path, width = 2400, height = 850, res = 150)
  on.exit(dev.off(), add = TRUE)
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mfrow = c(1, length(layouts)), mar = c(1.5, 1.5, 3, 1), bg = "white")
  cols <- if (is.null(labels)) {
    rep("#1f77b4", nrow(layouts[[1L]]$layout))
  } else {
    labs <- as.factor(labels)
    grDevices::hcl.colors(nlevels(labs), "Dark 3")[as.integer(labs)]
  }
  for (item in layouts) {
    plot(item$layout, pch = 20, cex = 0.30, col = cols, xlab = "", ylab = "", axes = FALSE, main = item$title)
    box()
  }
  mtext(title, outer = TRUE, line = -1.2, cex = 1.1)
}

as_uwot_knn <- function(knn, k) {
  idx <- as.matrix(knn$indices)
  dist <- as.matrix(knn$distances)
  k <- min(as.integer(k), ncol(idx), ncol(dist))
  list(idx = idx[, seq_len(k), drop = FALSE], dist = dist[, seq_len(k), drop = FALSE])
}

run_one <- function(dataset) {
  path <- dataset_path(dataset)
  if (!file.exists(path)) {
    return(data.frame(dataset = dataset, status = "missing_dataset", error = path))
  }
  obj <- load_dataset(path)
  x <- prepare_x(obj$data)
  labels <- if (is.null(obj$labels)) NULL else as.factor(obj$labels)
  sub <- subset_rows(x, labels, max_n, seed + nchar(dataset))
  x <- sub$x
  labels <- sub$labels
  used_k <- min(k, nrow(x) - 1L)

  knn_time <- system.time({
    knn <- faissR::nn_without_self(x, k = used_k, backend = nn_backend, metric = "euclidean", n_threads = n_threads)
  })[["elapsed"]]

  rows <- list()
  panel_layouts <- list()
  for (graph_mode in c("binary", "fuzzy")) {
    gc()
    err <- NA_character_
    status <- "success"
    layout <- NULL
    embed_time <- system.time({
      layout <- tryCatch(
        fastEmbedR::umap_knn(
          knn,
          backend = "cpu",
          graph_mode = graph_mode,
          n_threads = n_threads,
          seed = seed,
          verbose = FALSE
        ),
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
    })[["elapsed"]]
    if (is.null(layout)) {
      status <- "failed"
      rows[[graph_mode]] <- data.frame(
        dataset = dataset, graph_mode = graph_mode, status = status,
        n = nrow(x), p = ncol(x), k = used_k, nn_sec = knn_time,
        embed_sec = embed_time, trustworthiness = NA_real_,
        knn_preservation_15 = NA_real_, label_knn_accuracy = NA_real_,
        error = err, stringsAsFactors = FALSE
      )
      next
    }
    scores <- fastEmbedR::evaluate_embedding(
      x,
      layout,
      labels = labels,
      k = c(15, 30, 50),
      primary_k = min(15L, nrow(x) - 1L),
      sample_size_for_global_metrics = min(2500L, nrow(x)),
      sample_size_for_local_metrics = min(2500L, nrow(x)),
      seed = seed,
      method = "umap",
      backend = "cpu",
      n_threads = n_threads,
      dataset = dataset
    )
    layout_path <- file.path(out_dir, "layouts", paste0(dataset, "_", graph_mode, ".RData"))
    plot_path <- file.path(out_dir, "plots", paste0(dataset, "_", graph_mode, ".png"))
    save(layout, labels, scores, knn, file = layout_path, compress = "xz")
    plot_layout(layout, labels, plot_path, paste(dataset, "UMAP", graph_mode))
    panel_layouts[[graph_mode]] <- list(layout = layout, title = paste("fastEmbedR", graph_mode))
    rows[[graph_mode]] <- data.frame(
      dataset = dataset,
      method = paste0("fastEmbedR_", graph_mode),
      graph_mode = graph_mode,
      status = status,
      n = nrow(x),
      p = ncol(x),
      k = used_k,
      nn_sec = knn_time,
      embed_sec = embed_time,
      trustworthiness = as.numeric(scores$trustworthiness[[1L]]),
      knn_preservation_15 = as.numeric(scores$knn_preservation_15[[1L]]),
      label_knn_accuracy = as.numeric(scores$label_knn_accuracy[[1L]] %||% NA_real_),
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  if (include_reference) {
    method_name <- "uwot_fast_sgd"
    err <- NA_character_
    status <- "success"
    layout <- NULL
    embed_time <- system.time({
      layout <- tryCatch(
        {
          if (!requireNamespace("uwot", quietly = TRUE)) {
            stop("Package `uwot` is not installed.", call. = FALSE)
          }
          uwot::umap(
            x,
            n_neighbors = used_k,
            nn_method = as_uwot_knn(knn, used_k),
            n_threads = n_threads,
            n_sgd_threads = n_threads,
            fast_sgd = TRUE,
            init = "spectral",
            min_dist = 0.01,
            ret_model = FALSE,
            verbose = FALSE,
            seed = seed
          )
        },
        error = function(e) {
          err <<- conditionMessage(e)
          NULL
        }
      )
    })[["elapsed"]]
    if (is.null(layout)) {
      status <- "failed"
      rows[[method_name]] <- data.frame(
        dataset = dataset, method = method_name, graph_mode = "reference",
        status = status, n = nrow(x), p = ncol(x), k = used_k,
        nn_sec = knn_time, embed_sec = embed_time, trustworthiness = NA_real_,
        knn_preservation_15 = NA_real_, label_knn_accuracy = NA_real_,
        error = err, stringsAsFactors = FALSE
      )
    } else {
      scores <- fastEmbedR::evaluate_embedding(
        x,
        layout,
        labels = labels,
        k = c(15, 30, 50),
        primary_k = min(15L, nrow(x) - 1L),
        sample_size_for_global_metrics = min(2500L, nrow(x)),
        sample_size_for_local_metrics = min(2500L, nrow(x)),
        seed = seed,
        method = "uwot_fast_sgd",
        backend = "cpu",
        n_threads = n_threads,
        dataset = dataset
      )
      layout_path <- file.path(out_dir, "layouts", paste0(dataset, "_uwot_fast_sgd.RData"))
      plot_path <- file.path(out_dir, "plots", paste0(dataset, "_uwot_fast_sgd.png"))
      save(layout, labels, scores, knn, file = layout_path, compress = "xz")
      plot_layout(layout, labels, plot_path, paste(dataset, "uwot fast_sgd"))
      panel_layouts[[method_name]] <- list(layout = layout, title = "uwot fast_sgd")
      rows[[method_name]] <- data.frame(
        dataset = dataset,
        method = method_name,
        graph_mode = "reference",
        status = status,
        n = nrow(x),
        p = ncol(x),
        k = used_k,
        nn_sec = knn_time,
        embed_sec = embed_time,
        trustworthiness = as.numeric(scores$trustworthiness[[1L]]),
        knn_preservation_15 = as.numeric(scores$knn_preservation_15[[1L]]),
        label_knn_accuracy = as.numeric(scores$label_knn_accuracy[[1L]] %||% NA_real_),
        error = NA_character_,
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(panel_layouts) > 1L) {
    panel_path <- file.path(out_dir, "plots", paste0(dataset, "_comparison.png"))
    plot_layout_panel(panel_layouts, labels, panel_path, paste(dataset, "UMAP visual comparison"))
  }
  do.call(rbind, rows)
}

results <- do.call(rbind, lapply(datasets, function(dataset) {
  message("Running ", dataset)
  tryCatch(run_one(dataset), error = function(e) {
    data.frame(dataset = dataset, status = "failed", error = conditionMessage(e), stringsAsFactors = FALSE)
  })
}))

utils::write.csv(results, file.path(out_dir, "umap_mit_validation_results.csv"), row.names = FALSE)
print(results)
message("Results written to: ", out_dir)
