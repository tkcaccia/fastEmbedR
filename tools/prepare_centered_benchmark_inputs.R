#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (grepl("^--", arg)) {
      kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
      out[[kv[[1L]]]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
    }
  }
  out
}
`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
args <- parse_args(commandArgs(trailingOnly = TRUE))

data_root <- args$data_root %||% "/mnt/sata_ssd/fastEmbedR/Data"
k <- as.integer(args$k %||% "100")
threads <- as.integer(args$threads %||% "4")
seed <- as.integer(args$seed %||% "4")
force <- isTRUE(as.logical(args$force %||% FALSE))
backend_pca_preference <- args$pca_backend %||% "cuda"
nn_backend_preference <- args$nn_backend %||% "faiss_gpu_ivf_flat"
method_tag <- "centered_raw"

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)
if (!available_pkg("fastEmbedR")) stop("fastEmbedR is required", call. = FALSE)
if (!available_pkg("faissR")) stop("faissR is required", call. = FALSE)

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_rdata_object <- function(path) {
  env <- new.env(parent = emptyenv())
  objects <- load(path, envir = env)
  for (object_name in objects) {
    obj <- get(object_name, envir = env, inherits = FALSE)
    if (is.list(obj) && !is.null(obj$data)) return(obj)
  }
  stop("No list object with `$data` found in ", path, call. = FALSE)
}

prepare_centered_data <- function(x) {
  if (is.data.frame(x) && as.numeric(object.size(x)) > 2 * 1024^3) {
    n <- nrow(x)
    p <- ncol(x)
    log_msg("large data.frame detected; building centered matrix column-by-column n=%d p=%d", n, p)
    out <- matrix(0, nrow = n, ncol = p)
    keep <- logical(p)
    centers <- numeric(p)
    out_j <- 0L
    kept_original <- integer(p)
    for (jj in seq_len(p)) {
      v <- as.numeric(x[[jj]])
      if (!all(is.finite(v))) next
      s <- stats::sd(v)
      if (!is.finite(s) || s <= 0) next
      out_j <- out_j + 1L
      centers[[out_j]] <- mean(v)
      out[, out_j] <- v - centers[[out_j]]
      kept_original[[out_j]] <- jj
      keep[[jj]] <- TRUE
      if (jj %% 128L == 0L) gc()
    }
    if (!out_j) stop("Dataset has no finite variable numeric columns after filtering.", call. = FALSE)
    if (out_j < p) {
      out <- out[, seq_len(out_j), drop = FALSE]
    }
    centers <- centers[seq_len(out_j)]
    out[!is.finite(out)] <- 0
    storage.mode(out) <- "double"
    attr(out, "column_centers") <- centers
    attr(out, "kept_columns") <- kept_original[seq_len(out_j)]
    return(out)
  }
  x <- coerce_matrix(x)
  finite_cols <- logical(ncol(x))
  sdv <- numeric(ncol(x))
  centers_all <- numeric(ncol(x))
  for (jj in seq_len(ncol(x))) {
    v <- x[, jj]
    finite_cols[[jj]] <- all(is.finite(v))
    if (finite_cols[[jj]]) {
      centers_all[[jj]] <- mean(v)
      sdv[[jj]] <- stats::sd(v)
    } else {
      centers_all[[jj]] <- NA_real_
      sdv[[jj]] <- NA_real_
    }
  }
  if (any(!finite_cols)) x <- x[, finite_cols, drop = FALSE]
  if (!ncol(x)) stop("Dataset has no finite numeric columns after filtering.", call. = FALSE)
  sdv <- sdv[finite_cols]
  centers <- centers_all[finite_cols]
  keep <- is.finite(sdv) & sdv > 0
  x <- x[, keep, drop = FALSE]
  if (!ncol(x)) stop("Dataset has no variable numeric columns after filtering.", call. = FALSE)
  centers <- centers[keep]
  for (jj in seq_len(ncol(x))) {
    x[, jj] <- x[, jj] - centers[[jj]]
  }
  x[!is.finite(x)] <- 0
  storage.mode(x) <- "double"
  attr(x, "column_centers") <- centers
  attr(x, "kept_columns") <- which(finite_cols)[keep]
  x
}

resolve_dataset_path <- function(row) {
  rel <- as.character(row$relative_path %||% "")
  candidates <- unique(c(
    file.path(data_root, rel),
    file.path(data_root, as.character(row$dataset), basename(rel)),
    as.character(row$path %||% "")
  ))
  candidates <- candidates[nzchar(candidates)]
  hit <- candidates[file.exists(candidates)]
  if (length(hit)) hit[[1L]] else stop("Missing dataset file for ", row$dataset, ": ", paste(candidates, collapse = ", "), call. = FALSE)
}

standardize_knn <- function(obj) {
  if (!is.null(obj$indices) && !is.null(obj$distances)) return(list(indices = obj$indices, distances = obj$distances))
  if (!is.null(obj$idx) && !is.null(obj$dist)) return(list(indices = obj$idx, distances = obj$dist))
  if (!is.null(obj$nn.idx) && !is.null(obj$nn.dists)) return(list(indices = obj$nn.idx, distances = obj$nn.dists))
  stop("Cannot identify KNN indices/distances in object.", call. = FALSE)
}

choose_knn_backends <- function(n, p) {
  if (nzchar(nn_backend_preference)) {
    return(trimws(strsplit(nn_backend_preference, ",", fixed = TRUE)[[1L]]))
  }
  # Prefer exact GPU Flat for medium/small data, approximate CUDA graph indexes for large data.
  if (n <= 120000 && p <= 2048) {
    c("faiss_gpu_flat_l2", "cuda_cuvs_bruteforce", "cuda_cuvs_cagra", "faiss_gpu_ivf_flat", "faiss_hnsw", "faiss")
  } else {
    c("cuda_cuvs_cagra", "faiss_gpu_ivf_flat", "faiss_hnsw", "faiss_nndescent", "faiss")
  }
}

compute_nn <- function(x, dataset_name) {
  last_error <- NULL
  for (backend in choose_knn_backends(nrow(x), ncol(x))) {
    log_msg("%s: trying NN backend=%s k=%d", dataset_name, backend, k)
    t <- system.time({
      out <- tryCatch(
        faissR::nn_without_self(x, k = k, backend = backend, metric = "euclidean", n_threads = threads),
        error = function(e) e
      )
    })[["elapsed"]]
    if (!inherits(out, "error")) {
      sx <- standardize_knn(out)
      if (ncol(sx$indices) >= k) {
        attr(out, "benchmark_backend") <- backend
        attr(out, "benchmark_sec") <- as.numeric(t)
        log_msg("%s: NN success backend=%s sec=%.3f", dataset_name, backend, as.numeric(t))
        return(out)
      }
      last_error <- paste0(backend, ": returned only ", ncol(sx$indices), " neighbours")
    } else {
      last_error <- paste0(backend, ": ", conditionMessage(out))
      log_msg("%s: NN failed %s", dataset_name, last_error)
    }
    gc()
  }
  stop("All NN backends failed for ", dataset_name, ". Last error: ", last_error, call. = FALSE)
}

compute_pca_init <- function(x, dataset_name) {
  if (nrow(x) >= 500000L && ncol(x) <= 2048L) {
    log_msg("%s: using memory-stable covariance PCA init for n=%d p=%d", dataset_name, nrow(x), ncol(x))
    t <- system.time({
      cov_x <- crossprod(x) / max(1, nrow(x) - 1L)
      eig <- eigen(cov_x, symmetric = TRUE)
      scores <- x %*% eig$vectors[, seq_len(2L), drop = FALSE]
      scores <- scale(scores, center = TRUE, scale = FALSE)
      sds <- apply(scores, 2L, stats::sd)
      scale_factor <- max(sds[is.finite(sds)], na.rm = TRUE)
      if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1
      init <- as.matrix(scores) * (1e-4 / scale_factor)
      storage.mode(init) <- "double"
    })[["elapsed"]]
    attr(init, "benchmark_backend") <- "covariance_cpu"
    attr(init, "benchmark_sec") <- as.numeric(t)
    log_msg("%s: covariance PCA init success sec=%.3f max_sd=%.6g", dataset_name, as.numeric(t), max(apply(init, 2L, stats::sd)))
    return(init)
  }
  backends <- unique(c(backend_pca_preference, "cpu"))
  last_error <- NULL
  for (backend in backends) {
    log_msg("%s: trying PCA init backend=%s", dataset_name, backend)
    t <- system.time({
      init <- tryCatch(
        fastEmbedR::opentsne_pca_init(x, n_components = 2L, seed = seed, backend = backend, force_recompute = TRUE),
        error = function(e) e
      )
    })[["elapsed"]]
    if (!inherits(init, "error")) {
      init <- as.matrix(init)
      storage.mode(init) <- "double"
      attr(init, "benchmark_backend") <- backend
      attr(init, "benchmark_sec") <- as.numeric(t)
      log_msg("%s: PCA init success backend=%s sec=%.3f max_sd=%.6g", dataset_name, backend, as.numeric(t), max(apply(init, 2L, stats::sd)))
      return(init)
    }
    last_error <- paste0(backend, ": ", conditionMessage(init))
    log_msg("%s: PCA init failed %s", dataset_name, last_error)
    gc()
  }
  stop("PCA init failed for ", dataset_name, ". Last error: ", last_error, call. = FALSE)
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
if (!file.exists(manifest_path)) stop("Missing dataset manifest: ", manifest_path, call. = FALSE)
manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE)
if (!"relative_path" %in% names(manifest)) stop("dataset_manifest.csv needs `relative_path`", call. = FALSE)

dataset_filter <- args$datasets %||% paste(manifest$dataset, collapse = ",")
dataset_filter <- trimws(strsplit(dataset_filter, ",", fixed = TRUE)[[1L]])
manifest <- manifest[manifest$dataset %in% dataset_filter, , drop = FALSE]
if (!nrow(manifest)) stop("No requested datasets found", call. = FALSE)

pca_rows <- list()
knn_rows <- list()
summary_rows <- list()

for (ii in seq_len(nrow(manifest))) {
  dataset_name <- as.character(manifest$dataset[[ii]])
  data_path <- resolve_dataset_path(manifest[ii, , drop = FALSE])
  dataset_dir <- dirname(data_path)
  pca_file <- file.path(dataset_dir, paste0(dataset_name, "_", method_tag, "_opentsne_pca2_init.RData"))
  nn_file <- file.path(dataset_dir, paste0(dataset_name, "_", method_tag, "_euclidean_k", k, "_nn.RData"))
  log_msg("==== %s ====", dataset_name)
  log_msg("Data: %s", data_path)
  if (!force && file.exists(pca_file) && file.exists(nn_file)) {
    log_msg("%s: existing PCA and NN files found; skipping (use --force=TRUE to overwrite)", dataset_name)
    pca_rows[[length(pca_rows) + 1L]] <- data.frame(dataset = dataset_name, pca_init_path = pca_file, preprocessing = "mean_center_no_scale", stringsAsFactors = FALSE)
    knn_rows[[length(knn_rows) + 1L]] <- data.frame(dataset = dataset_name, knn_path = nn_file, k = k, metric = "euclidean", preprocessing = "mean_center_no_scale", stringsAsFactors = FALSE)
    next
  }
  result <- tryCatch({
    obj <- load_rdata_object(data_path)
    labels <- if (is.null(obj$labels)) NULL else as.factor(obj$labels)
    data_raw <- obj$data
    rm(obj)
    gc()
    log_msg("%s: loaded raw object", dataset_name)
    x <- prepare_centered_data(data_raw)
    rm(data_raw)
    gc()
    n <- nrow(x); p <- ncol(x)
    log_msg("%s: centered matrix n=%d p=%d", dataset_name, n, p)

    Y_init <- if (!force && file.exists(pca_file)) {
      log_msg("%s: loading existing PCA file %s", dataset_name, pca_file)
      env <- new.env(parent = emptyenv()); load(pca_file, envir = env); get("Y_init", env)
    } else {
      compute_pca_init(x, dataset_name)
    }
    if (nrow(Y_init) != n || ncol(Y_init) != 2L) stop("PCA init shape mismatch", call. = FALSE)
    pca_metadata <- list(
      dataset = dataset_name,
      source_data = data_path,
      preprocessing = "non-finite columns removed; zero-variance columns removed; mean-centered only; no variance scaling",
      pca = "fastEmbedR::opentsne_pca_init on centered data",
      tsne_init_transform = "PCA scores centered and multiplied so max component SD is 1e-4",
      seed = seed,
      backend = attr(Y_init, "benchmark_backend") %||% attr(Y_init, "fastEmbedR_init_backend") %||% NA_character_,
      seconds = attr(Y_init, "benchmark_sec") %||% NA_real_,
      n = n,
      p = p,
      created = as.character(Sys.time())
    )
    save(Y_init, labels, pca_metadata, file = pca_file, compress = "gzip")
    log_msg("%s: saved PCA init %s", dataset_name, pca_file)

    nn_centered_euclidean_k100 <- if (!force && file.exists(nn_file)) {
      log_msg("%s: loading existing NN file %s", dataset_name, nn_file)
      env <- new.env(parent = emptyenv()); load(nn_file, envir = env); get("nn_centered_euclidean_k100", env)
    } else {
      compute_nn(x, dataset_name)
    }
    nn_metadata <- list(
      dataset = dataset_name,
      source_data = data_path,
      preprocessing = "same centered matrix used for PCA; no variance scaling",
      k = k,
      metric = "euclidean",
      backend = attr(nn_centered_euclidean_k100, "benchmark_backend") %||% attr(nn_centered_euclidean_k100, "backend") %||% NA_character_,
      seconds = attr(nn_centered_euclidean_k100, "benchmark_sec") %||% NA_real_,
      n = n,
      p = p,
      created = as.character(Sys.time())
    )
    save(nn_centered_euclidean_k100, labels, nn_metadata, file = nn_file, compress = FALSE)
    log_msg("%s: saved NN %s", dataset_name, nn_file)

    list(ok = TRUE, n = n, p = p, pca_file = pca_file, nn_file = nn_file, pca_metadata = pca_metadata, nn_metadata = nn_metadata, error = NA_character_)
  }, error = function(e) {
    log_msg("%s: FAILED: %s", dataset_name, conditionMessage(e))
    list(ok = FALSE, n = NA_integer_, p = NA_integer_, pca_file = pca_file, nn_file = nn_file, pca_metadata = list(), nn_metadata = list(), error = conditionMessage(e))
  })

  pca_rows[[length(pca_rows) + 1L]] <- data.frame(dataset = dataset_name, pca_init_path = pca_file, preprocessing = "mean_center_no_scale", stringsAsFactors = FALSE)
  knn_rows[[length(knn_rows) + 1L]] <- data.frame(dataset = dataset_name, knn_path = nn_file, k = k, metric = "euclidean", preprocessing = "mean_center_no_scale", stringsAsFactors = FALSE)
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    dataset = dataset_name,
    status = if (result$ok) "success" else "failed",
    n = result$n,
    p = result$p,
    pca_file = result$pca_file,
    pca_backend = result$pca_metadata$backend %||% NA_character_,
    pca_sec = result$pca_metadata$seconds %||% NA_real_,
    nn_file = result$nn_file,
    nn_backend = result$nn_metadata$backend %||% NA_character_,
    nn_sec = result$nn_metadata$seconds %||% NA_real_,
    error = result$error,
    stringsAsFactors = FALSE
  )
  gc()
}

pca_manifest <- do.call(rbind, pca_rows)
knn_manifest <- do.call(rbind, knn_rows)
summary <- do.call(rbind, summary_rows)
utils::write.csv(pca_manifest, file.path(data_root, "pca_init_manifest_centered_raw.csv"), row.names = FALSE)
utils::write.csv(knn_manifest, file.path(data_root, paste0("knn_manifest_centered_raw_k", k, ".csv")), row.names = FALSE)
utils::write.csv(summary, file.path(data_root, paste0("centered_raw_precompute_summary_k", k, ".csv")), row.names = FALSE)
log_msg("Wrote manifests and summary under %s", data_root)
print(summary)
