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
base_dir <- normalizePath(args$base_dir %||% "/scratch/firenze/NN", mustWork = FALSE)
data_root <- normalizePath(args$data_root %||% file.path(base_dir, "Data"), mustWork = FALSE)
out_dir <- normalizePath(args$out_dir %||% file.path(base_dir, "benchmark_embeddings"), mustWork = FALSE)
threads_cpu <- as.integer(args$threads_cpu %||% "2")
seed <- as.integer(args$seed %||% "4")
saved_knn_k <- as.integer(args$saved_knn_k %||% "100")
embed_k <- as.integer(args$embed_k %||% "15")
perplexity <- as.numeric(args$perplexity %||% as.character(embed_k))
datasets_arg <- args$datasets %||% "COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris"
methods_arg <- args$methods %||% "opentsne_cpu,opentsne_cuda,umap_cpu_binary,umap_cpu_fuzzy,umap_cuda_binary,umap_cuda_fuzzy"
force <- isTRUE(as.logical(args$force %||% FALSE))

if (is.na(threads_cpu) || threads_cpu < 1L) threads_cpu <- 2L
if (is.na(seed)) seed <- 4L
if (is.na(saved_knn_k) || saved_knn_k < 1L) saved_knn_k <- 100L
if (is.na(embed_k) || embed_k < 1L) embed_k <- 15L
if (is.na(perplexity) || perplexity <= 0) perplexity <- min(15, embed_k)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(Sys.getenv("XDG_CACHE_HOME", file.path(out_dir, ".cache")), recursive = TRUE, showWarnings = FALSE)
dir.create(Sys.getenv("FONTCONFIG_CACHE", file.path(out_dir, ".cache", "fontconfig")), recursive = TRUE, showWarnings = FALSE)
dir.create(Sys.getenv("TMPDIR", file.path(out_dir, "tmp")), recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  OMP_NUM_THREADS = as.character(threads_cpu),
  OPENBLAS_NUM_THREADS = as.character(threads_cpu),
  MKL_NUM_THREADS = as.character(threads_cpu),
  VECLIB_MAXIMUM_THREADS = as.character(threads_cpu),
  XDG_CACHE_HOME = Sys.getenv("XDG_CACHE_HOME", file.path(out_dir, ".cache")),
  FONTCONFIG_CACHE = Sys.getenv("FONTCONFIG_CACHE", file.path(out_dir, ".cache", "fontconfig")),
  TMPDIR = Sys.getenv("TMPDIR", file.path(out_dir, "tmp"))
)
set.seed(seed)

log_file <- file.path(out_dir, "logs", paste0("benchmark_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))
log_msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
  flush.console()
}

need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Required package not available: ", pkg, call. = FALSE)
  }
}
need_pkg("fastEmbedR")

datasets <- trimws(strsplit(datasets_arg, ",", fixed = TRUE)[[1L]])
datasets <- datasets[nzchar(datasets)]
methods <- trimws(strsplit(methods_arg, ",", fixed = TRUE)[[1L]])
methods <- methods[nzchar(methods)]

source_rdata_path <- function(dataset) {
  hits <- list.files(file.path(data_root, dataset), pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
  hits <- hits[!grepl("(_nn|pca|manifest|summary)", basename(hits), ignore.case = TRUE)]
  if (!length(hits)) stop("No source RData found for ", dataset, call. = FALSE)
  hits[[1L]]
}

pick_dataset_object <- function(path) {
  env <- new.env(parent = emptyenv())
  names <- load(path, envir = env)
  for (nm in names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.list(obj) && !is.null(obj$data)) {
      return(list(data = obj$data, labels = obj$labels %||% NULL, object_name = nm))
    }
  }
  for (nm in names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "Matrix")) {
      labels <- NULL
      for (candidate in c("labels", "label", "Y", "y", "classes", "class")) {
        if (exists(candidate, envir = env, inherits = FALSE)) {
          lab <- get(candidate, envir = env, inherits = FALSE)
          if (length(lab) == nrow(obj)) labels <- lab
        }
      }
      return(list(data = obj, labels = labels, object_name = nm))
    }
  }
  stop("Could not find data object in ", path, call. = FALSE)
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

center_matrix <- function(x) {
  x <- as_numeric_matrix(x)
  finite_col <- vapply(seq_len(ncol(x)), function(j) all(is.finite(x[, j])), logical(1L))
  if (!all(finite_col)) x <- x[, finite_col, drop = FALSE]
  sds <- apply(x, 2L, stats::sd)
  keep <- is.finite(sds) & sds > 0
  x <- x[, keep, drop = FALSE]
  centers <- colMeans(x)
  x <- sweep(x, 2L, centers, "-")
  x[!is.finite(x)] <- 0
  storage.mode(x) <- "double"
  x
}

load_rdata_named_or_first <- function(path, preferred) {
  env <- new.env(parent = emptyenv())
  names <- load(path, envir = env)
  for (nm in preferred) {
    if (exists(nm, envir = env, inherits = FALSE)) return(get(nm, envir = env, inherits = FALSE))
  }
  get(names[[1L]], envir = env, inherits = FALSE)
}

subset_knn <- function(knn, k) {
  if (is.null(knn$indices) || is.null(knn$distances)) stop("KNN object lacks indices/distances.", call. = FALSE)
  use_k <- min(k, ncol(knn$indices), ncol(knn$distances))
  list(
    indices = knn$indices[, seq_len(use_k), drop = FALSE],
    distances = knn$distances[, seq_len(use_k), drop = FALSE]
  )
}

plot_layout <- function(layout, labels, path, title) {
  png(path, width = 1800, height = 1400, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 3, 1))
  if (is.null(labels)) {
    plot(layout, pch = 20, cex = 0.28, col = "#1f77b4", xlab = "", ylab = "", axes = FALSE, main = title)
  } else {
    cols <- grDevices::rainbow(length(unique(labels)))[as.integer(as.factor(labels))]
    plot(layout, pch = 20, cex = 0.28, col = cols, xlab = "", ylab = "", axes = FALSE, main = title)
  }
  box()
}

quick_scores <- function(x_high, layout, labels, k = 15L) {
  out <- list(trust = NA_real_, knn_preservation = NA_real_, label_acc = NA_real_)
  score <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x_high,
      layout,
      labels = labels,
      k = min(k, nrow(layout) - 1L),
      sample_size_for_global_metrics = min(2500L, nrow(layout)),
      sample_size_for_local_metrics = min(2500L, nrow(layout))
    ),
    error = function(e) e
  )
  if (!inherits(score, "error")) {
    nms <- names(score)
    out$trust <- as.numeric(score[[intersect(c("trustworthiness", "trust"), nms)[1L]]] %||% NA_real_)
    kp <- intersect(c("knn_preservation", "knn_preservation_15"), nms)
    out$knn_preservation <- as.numeric(score[[kp[1L]]] %||% NA_real_)
    la <- intersect(c("nn_accuracy", "label_knn_accuracy"), nms)
    out$label_acc <- as.numeric(score[[la[1L]]] %||% NA_real_)
  }
  out
}

cuda_ok <- function() {
  isTRUE(tryCatch(fastEmbedR::cuda_available(), error = function(e) FALSE))
}

method_spec <- function(name) {
  switch(
    name,
    opentsne_cpu = list(family = "opentsne", backend = "cpu"),
    opentsne_cuda = list(family = "opentsne", backend = "cuda"),
    umap_cpu_binary = list(family = "umap", backend = "cpu", graph_mode = "binary"),
    umap_cpu_fuzzy = list(family = "umap", backend = "cpu", graph_mode = "fuzzy"),
    umap_cuda_binary = list(family = "umap", backend = "cuda", graph_mode = "binary"),
    umap_cuda_fuzzy = list(family = "umap", backend = "cuda", graph_mode = "fuzzy"),
    stop("Unknown method: ", name, call. = FALSE)
  )
}

results <- list()
for (dataset in datasets) {
  log_msg("==== %s ====", dataset)
  source_path <- source_rdata_path(dataset)
  pca_path <- file.path(data_root, dataset, paste0(dataset, "_centered_raw_opentsne_pca2_init.RData"))
  nn_path <- file.path(data_root, dataset, paste0(dataset, "_centered_raw_euclidean_k", saved_knn_k, "_nn.RData"))
  if (!file.exists(pca_path)) stop("Missing PCA init: ", pca_path, call. = FALSE)
  if (!file.exists(nn_path)) stop("Missing NN file: ", nn_path, call. = FALSE)

  obj <- pick_dataset_object(source_path)
  labels <- if (is.null(obj$labels)) NULL else as.factor(obj$labels)
  x_centered <- center_matrix(obj$data)
  Y_init <- load_rdata_named_or_first(pca_path, c("Y_init", "pca_init"))
  knn_full <- load_rdata_named_or_first(nn_path, c("nn_centered_euclidean_k100", "knn", "nn_obj"))
  knn_embed <- subset_knn(knn_full, embed_k)
  n <- nrow(x_centered)
  p <- ncol(x_centered)

  for (method in methods) {
    spec <- method_spec(method)
    if (identical(spec$backend, "cuda") && !cuda_ok()) {
      log_msg("%s/%s: CUDA unavailable, skipping", dataset, method)
      results[[length(results) + 1L]] <- data.frame(
        dataset = dataset, method = method, backend = spec$backend,
        status = "backend_unavailable", n = n, p = p, embed_k = embed_k,
        perplexity = if (spec$family == "opentsne") perplexity else NA_real_,
        elapsed_sec = NA_real_, trust = NA_real_, knn_preservation = NA_real_,
        label_acc = NA_real_, layout_file = NA_character_, plot_file = NA_character_,
        error = "CUDA unavailable", stringsAsFactors = FALSE
      )
      next
    }

    layout_file <- file.path(out_dir, "layouts", paste0(dataset, "_", method, "_seed", seed, ".rds"))
    plot_file <- file.path(out_dir, "plots", paste0(dataset, "_", method, "_seed", seed, ".png"))
    if (!force && file.exists(layout_file) && file.exists(plot_file)) {
      log_msg("%s/%s: existing output, skipping", dataset, method)
      next
    }

    log_msg("%s/%s: running backend=%s", dataset, method, spec$backend)
    err <- NA_character_
    status <- "success"
    elapsed <- NA_real_
    layout <- NULL
    t <- system.time({
      layout <- tryCatch({
        if (identical(spec$family, "opentsne")) {
          fastEmbedR::opentsne_knn(
            knn_embed,
            Y_init = Y_init,
            perplexity = perplexity,
            backend = spec$backend,
            n_threads = threads_cpu,
            seed = seed,
            negative_gradient_method = "fft",
            verbose = FALSE
          )
        } else {
          fastEmbedR::umap_knn(
            knn_embed,
            backend = spec$backend,
            graph_mode = spec$graph_mode,
            n_threads = threads_cpu,
            seed = seed,
            verbose = FALSE
          )
        }
      }, error = function(e) e)
    })
    elapsed <- unname(t[["elapsed"]])

    if (inherits(layout, "error")) {
      status <- "failed"
      err <- conditionMessage(layout)
      log_msg("%s/%s: FAILED: %s", dataset, method, err)
      layout <- NULL
      scores <- list(trust = NA_real_, knn_preservation = NA_real_, label_acc = NA_real_)
    } else {
      saveRDS(layout, layout_file)
      plot_layout(layout, labels, plot_file, paste(dataset, method))
      scores <- quick_scores(x_centered, layout, labels, k = min(embed_k, 15L))
      log_msg("%s/%s: success sec=%.3f trust=%s", dataset, method, elapsed, format(scores$trust, digits = 4))
    }

    results[[length(results) + 1L]] <- data.frame(
      dataset = dataset,
      method = method,
      backend = spec$backend,
      status = status,
      n = n,
      p = p,
      embed_k = embed_k,
      perplexity = if (spec$family == "opentsne") perplexity else NA_real_,
      elapsed_sec = elapsed,
      trust = scores$trust,
      knn_preservation = scores$knn_preservation,
      label_acc = scores$label_acc,
      layout_file = if (is.null(layout)) NA_character_ else layout_file,
      plot_file = if (is.null(layout)) NA_character_ else plot_file,
      error = err,
      stringsAsFactors = FALSE
    )
    utils::write.csv(do.call(rbind, results), file.path(out_dir, "embedding_benchmark_results.csv"), row.names = FALSE)
    gc()
  }
  rm(x_centered, obj, knn_full, knn_embed, Y_init)
  gc()
}

summary <- do.call(rbind, results)
utils::write.csv(summary, file.path(out_dir, "embedding_benchmark_results.csv"), row.names = FALSE)
print(summary)
if (any(summary$status == "failed")) quit(status = 2L)
