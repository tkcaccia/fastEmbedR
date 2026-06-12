#!/usr/bin/env Rscript

# Extended benchmark runner for GitHub/manuscript datasets.
# Kept in tools/ because benchmarks are not part of the public package API.

user_lib <- Sys.getenv("R_LIBS_USER", "")
if (nzchar(user_lib)) {
  .libPaths(unique(c(user_lib, .libPaths())))
}
suppressPackageStartupMessages(library(fastEmbedR))

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  args <- commandArgs(trailingOnly = TRUE)
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

arg_flag <- function(name, default = FALSE) {
  value <- arg_value(name, if (default) "true" else "false")
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
  out <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

source_large_helpers <- function() {
  path <- file.path("tools", "benchmark_large_best_methods.R")
  lines <- readLines(path, warn = FALSE)
  boundary <- grep("^datasets_arg <-", lines)
  if (length(boundary) != 1L) {
    stop("Could not find helper boundary in ", path, call. = FALSE)
  }
  eval(parse(text = lines[seq_len(boundary[[1L]] - 1L)]), envir = parent.frame())
}

source_large_helpers()

timed <- function(expr) {
  invisible(gc())
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

safe_run <- function(expr) {
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

json_params <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x, use.names = FALSE), sep = "="), collapse = ";")
}

knn_backend_for <- function(backend) {
  switch(
    backend,
    cpu = "cpu_nndescent",
    metal = "metal_nndescent",
    cuda = "cuda_cuvs_nndescent",
    stop("Unsupported backend: ", backend, call. = FALSE)
  )
}

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", as.character(x))

cache_path <- function(cache_dir, prefix, dataset, backend, n, p, k, seed) {
  file.path(
    cache_dir,
    sprintf(
      "%s_%s_backend-%s_n%d_p%d_k%d_seed%d.rds",
      prefix, safe_name(dataset), safe_name(backend),
      as.integer(n), as.integer(p), as.integer(k), as.integer(seed)
    )
  )
}

read_or_prepare <- function(dataset_key, cache_dir, min_n, max_n, pca_dims, seed) {
  raw <- load_named_dataset(
    dataset_key,
    cache_dir = file.path(cache_dir, "downloads"),
    min_n = min_n,
    max_n = max_n,
    seed = seed
  )
  if (is.null(raw)) return(NULL)
  prep_path <- cache_path(
    file.path(cache_dir, "prepared"),
    "prepared",
    raw$name,
    paste0("pca", pca_dims),
    nrow(raw$x),
    ncol(raw$x),
    pca_dims,
    seed
  )
  if (file.exists(prep_path)) {
    return(readRDS(prep_path))
  }
  message("Preprocessing ", raw$name, " with PCA dims=", pca_dims)
  prepared <- prepare_dataset(raw, pca_dims = pca_dims, seed = seed, preprocess_backend = "cpu")
  dir.create(dirname(prep_path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(prepared, prep_path, version = 2)
  prepared
}

read_or_compute_knn <- function(dataset, backend, cache_dir, k, seed, n_threads, force = FALSE) {
  knn_backend <- knn_backend_for(backend)
  path <- cache_path(
    file.path(cache_dir, "knn"),
    "knn",
    dataset$name,
    knn_backend,
    nrow(dataset$x),
    ncol(dataset$x),
    k + 1L,
    seed
  )
  if (!isTRUE(force) && file.exists(path)) {
    cached <- readRDS(path)
    cached$cache_hit <- TRUE
    return(cached)
  }
  message("KNN ", dataset$name, " backend=", backend, " knn_backend=", knn_backend)
  run <- timed(fastEmbedR::nn(
    dataset$x,
    k = k + 1L,
    backend = knn_backend,
    n_threads = n_threads
  ))
  out <- list(
    knn = run$value,
    nn_sec = run$sec,
    backend = backend,
    knn_backend = attr(run$value, "backend") %||% knn_backend,
    cache_hit = FALSE,
    created = as.character(Sys.time())
  )
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(out, path, version = 2)
  out
}

score_layout <- function(dataset, layout, method, backend, seed, metric_n, n_threads) {
  labels <- dataset$labels
  rows <- stratified_sample(labels, nrow(dataset$x), metric_n, seed + 101L)
  tryCatch(
    fastEmbedR::evaluate_embedding(
      dataset$x[rows, , drop = FALSE],
      layout[rows, , drop = FALSE],
      labels = if (is.null(labels)) NULL else labels[rows],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(rows)),
      sample_size_for_local_metrics = min(3000L, length(rows)),
      seed = seed,
      method = method,
      backend = "cpu",
      n_threads = n_threads,
      dataset = dataset$name
    ),
    error = function(e) {
      data.frame(
        trustworthiness = NA_real_,
        continuity = NA_real_,
        knn_preservation_15 = NA_real_,
        knn_preservation_30 = NA_real_,
        knn_preservation_50 = NA_real_,
        distance_spearman = NA_real_,
        distance_pearson = NA_real_,
        stress = NA_real_,
        silhouette = NA_real_,
        label_knn_accuracy = NA_real_,
        ari = NA_real_,
        nmi = NA_real_,
        rare_class_recall = NA_real_,
        metric_error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )
}

empty_metrics <- function() {
  data.frame(
    trustworthiness = NA_real_,
    continuity = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    distance_spearman = NA_real_,
    distance_pearson = NA_real_,
    stress = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    ari = NA_real_,
    nmi = NA_real_,
    rare_class_recall = NA_real_,
    stringsAsFactors = FALSE
  )
}

result_row <- function(dataset,
                       method,
                       backend_requested,
                       backend_used,
                       status,
                       error_message,
                       seed,
                       k,
                       perplexity,
                       knn_info = NULL,
                       embedding_sec = NA_real_,
                       init_sec = NA_real_,
                       total_sec = NA_real_,
                       layout_path = NA_character_,
                       parameters = list(),
                       metrics = empty_metrics()) {
  data.frame(
    machine = Sys.info()[["nodename"]],
    dataset = dataset$name,
    dataset_source = dataset$source,
    raw_n = dataset$raw_n,
    raw_p = dataset$raw_p,
    n = nrow(dataset$x),
    p = ncol(dataset$x),
    method = method,
    backend_requested = backend_requested,
    backend_used = backend_used,
    status = status,
    error_message = error_message,
    seed = seed,
    k = k,
    perplexity = perplexity,
    nn_backend = knn_info$knn_backend %||% NA_character_,
    nn_sec = knn_info$nn_sec %||% NA_real_,
    nn_cache_hit = knn_info$cache_hit %||% NA,
    init_sec = init_sec,
    embedding_sec = embedding_sec,
    total_sec = total_sec,
    trustworthiness = metrics$trustworthiness[[1L]],
    continuity = metrics$continuity[[1L]],
    knn_preservation_15 = metrics$knn_preservation_15[[1L]],
    knn_preservation_30 = metrics$knn_preservation_30[[1L]],
    knn_preservation_50 = metrics$knn_preservation_50[[1L]],
    distance_spearman = metrics$distance_spearman[[1L]],
    distance_pearson = metrics$distance_pearson[[1L]],
    stress = metrics$stress[[1L]],
    silhouette = metrics$silhouette[[1L]],
    label_knn_accuracy = metrics$label_knn_accuracy[[1L]],
    ari = metrics$ari[[1L]],
    nmi = metrics$nmi[[1L]],
    rare_class_recall = metrics$rare_class_recall[[1L]],
    parameters_json = json_params(parameters),
    layout_path = layout_path,
    stringsAsFactors = FALSE
  )
}

run_opentsne_variant <- function(dataset,
                                 knn_info,
                                 backend,
                                 mpsgraph,
                                 init,
                                 init_sec,
                                 k,
                                 perplexity,
                                 early_iter,
                                 normal_iter,
                                 seed,
                                 n_threads) {
  old <- Sys.getenv("FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH")
    else Sys.setenv(FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH = old)
  }, add = TRUE)
  if (isTRUE(mpsgraph)) {
    Sys.setenv(FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH = "1")
  } else {
    Sys.unsetenv("FASTEMBEDR_METAL_OPENTSNE_MPSGRAPH")
  }
  timed(fastEmbedR::opentsne_knn(
    knn_info$knn,
    n_neighbors = k,
    perplexity = perplexity,
    Y_init = init,
    early_exaggeration_iter = early_iter,
    n_iter = normal_iter,
    learning_rate = "auto",
    negative_gradient_method = "fft",
    backend = backend,
    n_threads = n_threads,
    seed = seed
  ))
}

run_one_method <- function(method,
                           dataset,
                           knn_cpu,
                           knn_backend_info,
                           init,
                           init_sec,
                           k,
                           perplexity,
                           seed,
                           n_threads,
                           early_iter,
                           normal_iter,
                           umap_epochs,
                           metric_n,
                           layout_dir,
                           save_layouts) {
  backend <- if (grepl("_metal", method)) "metal" else if (grepl("_cuda", method)) "cuda" else "cpu"
  knn_info <- if (method %in% c("uwot_fast_sgd", "rtsne_neighbors")) knn_cpu else knn_backend_info[[backend]]
  if (is.null(knn_info) && !method %in% c("uwot_fast_sgd")) {
    return(result_row(
      dataset, method, backend, "none", "backend_unavailable",
      paste0("No KNN information for backend ", backend),
      seed, k, perplexity,
      parameters = list(k = k, backend = backend)
    ))
  }
  layout_path <- file.path(layout_dir, paste0(safe_name(dataset$name), "_", method, "_seed", seed, ".rds"))
  runner <- switch(
    method,
    opentsne_cpu = function() run_opentsne_variant(dataset, knn_info, "cpu", FALSE, init, init_sec, k, perplexity, early_iter, normal_iter, seed, n_threads),
    opentsne_metal = function() run_opentsne_variant(dataset, knn_info, "metal", FALSE, init, init_sec, k, perplexity, early_iter, normal_iter, seed, n_threads),
    opentsne_metal_mpsgraph = function() run_opentsne_variant(dataset, knn_info, "metal", TRUE, init, init_sec, k, perplexity, early_iter, normal_iter, seed, n_threads),
    opentsne_cuda = function() run_opentsne_variant(dataset, knn_info, "cuda", FALSE, init, init_sec, k, perplexity, early_iter, normal_iter, seed, n_threads),
    umap_cpu = function() timed(fastEmbedR:::fast_knn_umap_core(knn_info$knn, n_epochs = umap_epochs, backend = "cpu", seed = seed)),
    umap_metal = function() timed(fastEmbedR:::fast_knn_umap_core(knn_info$knn, n_epochs = umap_epochs, backend = "metal", seed = seed)),
    umap_cuda = function() timed(fastEmbedR:::fast_knn_umap_core(knn_info$knn, n_epochs = umap_epochs, backend = "cuda", seed = seed)),
    rtsne_neighbors = function() timed({
      if (!requireNamespace("Rtsne", quietly = TRUE)) {
        stop("Package `Rtsne` is not installed.", call. = FALSE)
      }
      clean <- fastEmbedR:::coerce_knn_input(knn_info$knn)
      Rtsne::Rtsne_neighbors(
        index = clean$indices,
        distance = clean$distances,
        dims = 2L,
        perplexity = perplexity,
        theta = 0.5,
        max_iter = early_iter + normal_iter,
        stop_lying_iter = early_iter,
        mom_switch_iter = early_iter,
        num_threads = n_threads,
        verbose = FALSE
      )$Y
    }),
    uwot_fast_sgd = function() timed({
      if (!requireNamespace("uwot", quietly = TRUE)) {
        stop("Package `uwot` is not installed.", call. = FALSE)
      }
      clean <- fastEmbedR:::coerce_knn_input(knn_info$knn)
      uwot::umap(
        dataset$x,
        n_neighbors = k,
        nn_method = list(idx = clean$indices, dist = clean$distances),
        n_epochs = umap_epochs,
        min_dist = 0.01,
        metric = "euclidean",
        init = "spectral",
        fast_sgd = TRUE,
        n_threads = n_threads,
        n_sgd_threads = n_threads,
        ret_model = FALSE,
        verbose = FALSE
      )
    }),
    stop("Unknown method: ", method, call. = FALSE)
  )
  requested_backend <- backend
  if (identical(method, "uwot_fast_sgd") || identical(method, "rtsne_neighbors")) {
    requested_backend <- "cpu"
  }
  out <- safe_run(runner())
  if (!identical(out$status, "success")) {
    return(result_row(
      dataset,
      method,
      requested_backend,
      "none",
      "failed",
      out$error,
      seed,
      k,
      perplexity,
      knn_info = knn_info,
      init_sec = if (grepl("^opentsne", method)) init_sec else NA_real_,
      parameters = list(k = k, perplexity = perplexity, method = method)
    ))
  }
  layout <- as_layout(out$value$value)
  cfg <- attr(layout, "fastEmbedR_config")
  backend_used <- cfg$backend %||% requested_backend
  if (identical(method, "opentsne_metal_mpsgraph")) {
    backend_used <- "metal_mpsgraph_diagnostic"
  }
  if (isTRUE(save_layouts)) {
    dir.create(layout_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(layout, layout_path, version = 2)
  } else {
    layout_path <- NA_character_
  }
  metrics <- score_layout(dataset, layout, method, backend_used, seed, metric_n, n_threads)
  result_row(
    dataset,
    method,
    requested_backend,
    backend_used,
    "success",
    NA_character_,
    seed,
    k,
    perplexity,
    knn_info = knn_info,
    embedding_sec = out$value$sec,
    init_sec = if (grepl("^opentsne", method)) init_sec else NA_real_,
    total_sec = knn_info$nn_sec + out$value$sec,
    layout_path = layout_path,
    parameters = list(
      k = k,
      perplexity = perplexity,
      early_iter = if (grepl("^opentsne|rtsne", method)) early_iter else NA_integer_,
      normal_iter = if (grepl("^opentsne|rtsne", method)) normal_iter else NA_integer_,
      n_epochs = if (grepl("^umap|uwot", method)) umap_epochs else NA_integer_,
      pca_init = grepl("^opentsne", method),
      mpsgraph = identical(method, "opentsne_metal_mpsgraph"),
      backend = requested_backend,
      backend_used = backend_used
    ),
    metrics = metrics
  )
}

plot_dataset_panels <- function(results, out_dir, plot_n, seed, point_cex) {
  ok <- results[results$status == "success" & nzchar(results$layout_path) & file.exists(results$layout_path), , drop = FALSE]
  if (nrow(ok) == 0L) return(invisible(NULL))
  for (dataset_name in unique(ok$dataset)) {
    rows <- ok[ok$dataset == dataset_name, , drop = FALSE]
    first_layout <- readRDS(rows$layout_path[[1L]])
    n <- nrow(first_layout)
    labels <- get0(dataset_name, envir = .fastembedr_extended_labels, ifnotfound = NULL)
    keep <- seq_len(min(n, plot_n))
    if (!is.null(labels) && length(labels) == n) {
      keep <- stratified_sample(labels, n, plot_n, seed + 91L)
    } else if (n > plot_n) {
      set.seed(seed + 91L)
      keep <- sort(sample.int(n, plot_n))
    }
    methods <- rows$method
    if (!is.null(labels) && length(labels) == n) {
      labels <- droplevels(factor(labels))
      pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
      point_col <- pal[as.integer(labels[keep])]
    } else {
      point_col <- "#2F6F9F"
    }
    png(file.path(out_dir, paste0(safe_name(dataset_name), "_embedding_gallery.png")),
        width = 1800, height = max(700, 620 * ceiling(length(methods) / 3)), res = 130)
    par(mfrow = c(ceiling(length(methods) / 3), min(3, length(methods))), mar = c(1, 1, 3.6, 1))
    for (i in seq_along(methods)) {
      y <- readRDS(rows$layout_path[[i]])
      plot(
        y[keep, 1L],
        y[keep, 2L],
        pch = 16,
        cex = point_cex,
        col = point_col,
        axes = FALSE,
        xlab = "",
        ylab = "",
        main = sprintf(
          "%s\nNN %.2fs | embed %.2fs | trust %.3f",
          rows$method[[i]], rows$nn_sec[[i]], rows$embedding_sec[[i]],
          rows$trustworthiness[[i]]
        )
      )
      box(col = "grey75")
    }
    dev.off()
  }
}

plot_stacked_timing <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok) == 0L) return(invisible(NULL))
  short_label <- function(x) {
    out <- x
    out <- sub("^opentsne_", "oTSNE_", out)
    out <- sub("^umap_", "UMAP_", out)
    out <- sub("metal_mpsgraph", "Metal_MPS", out)
    out <- sub("rtsne_neighbors", "Rtsne", out)
    out <- sub("uwot_fast_sgd", "uwot", out)
    out
  }
  for (dataset_name in unique(ok$dataset)) {
    rows <- ok[ok$dataset == dataset_name, , drop = FALSE]
    png(file.path(out_dir, paste0(safe_name(dataset_name), "_timing_stacked.png")),
        width = 1800, height = 1050, res = 140)
    par(mar = c(9.5, 5, 4, 1), xpd = NA)
    values <- rbind(
      `NN` = rows$nn_sec,
      `Init` = ifelse(is.finite(rows$init_sec), rows$init_sec, 0),
      `Embedding` = rows$embedding_sec
    )
    totals <- colSums(values, na.rm = TRUE)
    bp <- barplot(
      values,
      names.arg = short_label(rows$method),
      las = 2,
      cex.names = 0.8,
      col = c("#4C78A8", "#54A24B", "#F58518"),
      border = NA,
      ylab = "Seconds",
      main = paste(dataset_name, "timing"),
      ylim = c(0, max(totals, na.rm = TRUE) * 1.18)
    )
    legend("topleft", fill = c("#4C78A8", "#54A24B", "#F58518"),
           legend = rownames(values), bty = "n")
    text(bp, totals, labels = sprintf("%.1fs", totals), pos = 3, cex = 0.75)
    dev.off()
  }
}

datasets <- arg_csv("datasets", "mnist,fashion_mnist,shuttle,covertype,cifar10")
methods <- arg_csv(
  "methods",
  "opentsne_cpu,opentsne_metal,opentsne_metal_mpsgraph,umap_cpu,umap_metal,rtsne_neighbors,uwot_fast_sgd"
)
backends <- unique(c(
  if (any(grepl("_cpu$|rtsne|uwot", methods))) "cpu" else character(),
  if (any(grepl("_metal", methods))) "metal" else character(),
  if (any(grepl("_cuda", methods))) "cuda" else character()
))
seed <- arg_int("seed", 6L)
k <- arg_int("k", 50L)
min_n <- arg_int("min-n", 1L)
max_n <- arg_num("max-n", 70000)
pca_dims <- arg_int("pca-dims", 50L)
metric_n <- arg_int("metric-n", 5000L)
plot_n <- arg_int("plot-n", 20000L)
point_cex <- arg_num("point-cex", 0.24)
n_threads <- arg_int("threads", 4L)
early_iter <- arg_int("early-iter", 100L)
normal_iter <- arg_int("normal-iter", 150L)
umap_epochs <- arg_int("umap-epochs", 200L)
cache_dir <- arg_value("cache-dir", file.path("results", "extended_dr_cache"))
out_dir <- arg_value(
  "out-dir",
  file.path("results", paste0("extended_dr_benchmark_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
save_layouts <- !arg_flag("no-save-layouts", FALSE)
force_knn <- arg_flag("force-knn", FALSE)

.fastembedr_extended_labels <- new.env(parent = emptyenv())

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
layout_dir <- file.path(out_dir, "layouts")

message("Extended DR benchmark")
message("  datasets=", paste(datasets, collapse = ","))
message("  methods=", paste(methods, collapse = ","))
message("  max_n=", max_n, " pca_dims=", pca_dims, " k=", k)
message("  out_dir=", out_dir)

all_rows <- list()
for (dataset_key in datasets) {
  dataset <- read_or_prepare(dataset_key, cache_dir, min_n, max_n, pca_dims, seed)
  if (is.null(dataset)) next
  assign(dataset$name, dataset$labels, envir = .fastembedr_extended_labels)
  perplexity <- min(30L, floor(k / 3L), floor((nrow(dataset$x) - 1L) / 3L))
  message("Dataset ", dataset$name, ": n=", nrow(dataset$x), " p=", ncol(dataset$x),
          " perplexity=", perplexity)

  init_run <- timed(fastEmbedR:::make_opentsne_pca_init(
    dataset$x,
    n_components = 2L,
    seed = seed,
    backend = "cpu"
  ))
  opentsne_init <- init_run$value
  init_sec <- init_run$sec

  knn_infos <- list()
  for (backend in backends) {
    knn_infos[[backend]] <- safe_run(read_or_compute_knn(
      dataset,
      backend = backend,
      cache_dir = cache_dir,
      k = k,
      seed = seed,
      n_threads = n_threads,
      force = force_knn
    ))
    if (identical(knn_infos[[backend]]$status, "success")) {
      knn_infos[[backend]] <- knn_infos[[backend]]$value
    } else {
      warning("KNN failed for ", dataset$name, " backend=", backend, ": ",
              knn_infos[[backend]]$error, call. = FALSE)
      knn_infos[[backend]] <- NULL
    }
  }
  cpu_knn <- knn_infos[["cpu"]]

  for (method in methods) {
    message("Running ", dataset$name, " method=", method)
    row <- run_one_method(
      method = method,
      dataset = dataset,
      knn_cpu = cpu_knn,
      knn_backend_info = knn_infos,
      init = opentsne_init,
      init_sec = init_sec,
      k = k,
      perplexity = perplexity,
      seed = seed,
      n_threads = n_threads,
      early_iter = early_iter,
      normal_iter = normal_iter,
      umap_epochs = umap_epochs,
      metric_n = metric_n,
      layout_dir = layout_dir,
      save_layouts = save_layouts
    )
    all_rows[[length(all_rows) + 1L]] <- row
    write.csv(do.call(rbind, all_rows), file.path(out_dir, "extended_dr_results_partial.csv"), row.names = FALSE)
  }
}

results <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
csv <- file.path(out_dir, "extended_dr_results.csv")
write.csv(results, csv, row.names = FALSE)
write.csv(results, file.path(out_dir, "latest_extended_dr_results.csv"), row.names = FALSE)
plot_dataset_panels(results, out_dir, plot_n, seed, point_cex)
plot_stacked_timing(results, out_dir)

summary <- results[results$status == "success", , drop = FALSE]
if (nrow(summary)) {
  summary <- summary[order(summary$dataset, summary$method), c(
    "dataset", "method", "backend_used", "n", "p", "nn_sec", "init_sec",
    "embedding_sec", "total_sec", "trustworthiness", "knn_preservation_50",
    "label_knn_accuracy", "layout_path"
  ), drop = FALSE]
}
write.csv(summary, file.path(out_dir, "extended_dr_summary.csv"), row.names = FALSE)

message("Results: ", normalizePath(csv, winslash = "/", mustWork = FALSE))
message("Summary: ", normalizePath(file.path(out_dir, "extended_dr_summary.csv"), winslash = "/", mustWork = FALSE))
print(summary)
