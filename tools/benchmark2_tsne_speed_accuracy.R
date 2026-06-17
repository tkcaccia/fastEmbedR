#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (grepl("^--", arg)) {
      kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
      key <- kv[[1L]]
      value <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
      out[[key]] <- value
    }
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

data_root <- args$data_root %||% "/mnt/sata_ssd/fastEmbedR/Data"
out_dir <- args$out_dir %||% file.path("/mnt/sata_ssd", paste0("fastEmbedR_BENCHMARK2_", format(Sys.time(), "%Y%m%d_%H%M%S")))
benchmark1_dir <- args$benchmark1_dir %||% ""
parse_positive_number_or_auto <- function(value, name) {
  value <- value %||% "auto"
  if (length(value) == 1L && tolower(as.character(value)) %in% c("auto", "dataset", "")) return(NA_real_)
  value <- suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
    stop("`", name, "` must be one positive finite value, or `auto`.", call. = FALSE)
  }
  value
}

benchmark_perplexity <- parse_positive_number_or_auto(args$perplexity %||% Sys.getenv("FASTEMBEDR_BENCHMARK_PERPLEXITY", "auto"), "--perplexity")
k_override <- parse_positive_number_or_auto(args$k %||% Sys.getenv("FASTEMBEDR_BENCHMARK_K", "auto"), "--k")
knn_cache_k_override <- parse_positive_number_or_auto(args$knn_k %||% Sys.getenv("FASTEMBEDR_BENCHMARK_KNN_K", "auto"), "--knn_k")
k <- if (is.finite(k_override)) as.integer(k_override) else NA_integer_
knn_cache_k <- if (is.finite(knn_cache_k_override)) as.integer(knn_cache_k_override) else NA_integer_
rtsne_required_k <- if (is.finite(benchmark_perplexity)) as.integer(ceiling(3 * benchmark_perplexity) + 1L) else NA_integer_
tsne_required_k <- rtsne_required_k
n_threads <- as.integer(args$threads %||% "4")
timeout_sec <- as.integer(args$timeout %||% "600")
seed <- as.integer(args$seed %||% "4")
metric_n <- as.integer(args$metric_n %||% "5000")
worker <- isTRUE(as.logical(args$worker %||% FALSE))
include_raw_x <- isTRUE(as.logical(args$include_raw_x %||% TRUE))
include_fitsne <- isTRUE(as.logical(args$include_fitsne %||% TRUE))
backend_filter <- tolower(trimws(strsplit(args$backends %||% Sys.getenv("FASTEMBEDR_BENCHMARK_BACKENDS", "cpu,cuda"), ",", fixed = TRUE)[[1L]]))
backend_filter <- unique(backend_filter[nzchar(backend_filter)])
if (!length(backend_filter)) backend_filter <- c("cpu", "cuda")
if ("cpu" %in% backend_filter && include_fitsne && !"cpu_fft" %in% backend_filter) {
  backend_filter <- unique(c(backend_filter, "cpu_fft"))
}

benchmark_datasets <- c(
  "USPS",
  "FashionMNIST",
  "FlowRepository_FR-FCM-ZYRM_files",
  "flow18",
  "MNIST",
  "imagenet",
  "MetRef",
  "mass41",
  "COIL20"
)
default_dataset_paths <- data.frame(
  dataset = benchmark_datasets,
  relative_path = c(
    "USPS/USPS.RData",
    "FashionMNIST/FashionMNIST.RData",
    "FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData",
    "flow18/flow18.RData",
    "MNIST/MNIST.RData",
    "imagenet/imagenet.RData",
    "MetRef/MetRef.RData",
    "mass41/mass41.RData",
    "COIL20/COIL20.RData"
  ),
  stringsAsFactors = FALSE
)

dataset_parameter_defaults <- data.frame(
  dataset = benchmark_datasets,
  perplexity = c(15, 15, 30, 30, 15, 50, 15, 30, 15),
  k = c(15, 15, 30, 30, 15, 50, 15, 30, 15),
  reason = c(
    "small digit benchmark; low perplexity preserves digit subclasses",
    "70k fashion images; perplexity 15 matched the visually validated MNIST/Fashion-MNIST setting",
    "very large cytometry benchmark; moderate perplexity balances local and population structure",
    "flow cytometry benchmark; moderate perplexity balances local and population structure",
    "70k flattened digit images; user-validated perplexity 15",
    "large feature benchmark; higher k/perplexity is used for broader semantic neighbourhoods",
    "small metabolomics benchmark; low perplexity is safer",
    "mass cytometry benchmark; moderate perplexity balances local and population structure",
    "small image benchmark; low perplexity avoids over-smoothing"
  ),
  stringsAsFactors = FALSE
)

dataset_parameters <- function(dataset_name, n) {
  row <- dataset_parameter_defaults[dataset_parameter_defaults$dataset == dataset_name, , drop = FALSE]
  p_auto <- if (nrow(row)) row$perplexity[[1L]] else {
    if (n < 5000L) 15 else if (n < 100000L) 30 else 50
  }
  k_auto <- if (nrow(row)) row$k[[1L]] else as.integer(p_auto)
  p <- if (is.finite(benchmark_perplexity)) benchmark_perplexity else p_auto
  p <- min(as.numeric(p), max(1, floor((n - 1L) / 3L)))
  k_embed <- if (is.finite(k_override)) as.integer(k_override) else as.integer(round(k_auto))
  k_embed <- max(1L, min(k_embed, n - 1L))
  rtsne_k <- max(k_embed, as.integer(ceiling(3 * p) + 1L))
  cache_k <- if (is.finite(knn_cache_k_override)) as.integer(knn_cache_k_override) else max(100L, k_embed, rtsne_k)
  cache_k <- max(cache_k, k_embed, rtsne_k)
  cache_k <- min(cache_k, n - 1L)
  list(
    perplexity = p,
    k = k_embed,
    rtsne_required_k = min(rtsne_k, n - 1L),
    knn_cache_k = cache_k,
    policy = if (is.finite(benchmark_perplexity) || is.finite(k_override)) "user_override" else "dataset_auto",
    reason = if (nrow(row)) row$reason[[1L]] else "fallback rule based on sample size"
  )
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

json_or_text <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", digits = NA)
  } else {
    paste(capture.output(str(x)), collapse = " ")
  }
}

read_peak_rss_gb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  x <- readLines(status, warn = FALSE)
  v <- x[grepl("^VmHWM:", x)]
  if (!length(v)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", v[[1L]])))
  if (!is.finite(kb)) NA_real_ else kb / 1024^2
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

scale_dataset_matrix <- function(x) {
  x <- coerce_matrix(x)
  finite_cols <- apply(x, 2L, function(v) all(is.finite(v)))
  if (any(!finite_cols)) x <- x[, finite_cols, drop = FALSE]
  if (!ncol(x)) stop("Dataset has no finite numeric columns after filtering.", call. = FALSE)
  sdv <- apply(x, 2L, stats::sd)
  keep <- is.finite(sdv) & sdv > 0
  x <- x[, keep, drop = FALSE]
  if (!ncol(x)) stop("Dataset has no variable numeric columns after filtering.", call. = FALSE)
  x <- scale(x, center = TRUE, scale = FALSE)
  x[!is.finite(x)] <- 0
  storage.mode(x) <- "double"
  x
}

load_rdata_object <- function(path, name = NULL) {
  env <- new.env(parent = emptyenv())
  objects <- load(path, envir = env)
  if (!is.null(name) && exists(name, envir = env, inherits = FALSE)) {
    return(get(name, envir = env, inherits = FALSE))
  }
  if (!is.null(name)) {
    stop("No object named `", name, "` in ", path, call. = FALSE)
  }
  for (object_name in objects) {
    obj <- get(object_name, envir = env, inherits = FALSE)
    if (is.list(obj) && !is.null(obj$data)) return(obj)
  }
  stop("No list object with `$data` found in ", path, call. = FALSE)
}

load_dataset <- function(dataset, data_path) {
  obj <- load_rdata_object(data_path)
  list(
    data = scale_dataset_matrix(obj$data),
    labels = if (is.null(obj$labels)) NULL else as.factor(obj$labels),
    metadata = obj$metadata %||% list(),
    source = data_path
  )
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
if (!file.exists(manifest_path)) stop("Missing dataset manifest: ", manifest_path, call. = FALSE)
manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE)
if (!all(c("dataset", "path") %in% names(manifest))) {
  stop("dataset_manifest.csv must contain columns `dataset` and `path`.", call. = FALSE)
}
if (!"relative_path" %in% names(manifest)) {
  manifest <- merge(manifest, default_dataset_paths, by = "dataset", all.x = TRUE, sort = FALSE)
}
resolve_dataset_path <- function(row) {
  candidates <- unique(c(
    if ("path" %in% names(row)) as.character(row$path) else character(),
    file.path(data_root, as.character(row$relative_path))
  ))
  candidates <- candidates[nzchar(candidates)]
  hit <- candidates[file.exists(candidates)]
  if (length(hit)) hit[[1L]] else candidates[[length(candidates)]]
}
manifest$path <- vapply(seq_len(nrow(manifest)), function(i) resolve_dataset_path(manifest[i, , drop = FALSE]), character(1L))

pca_manifest_candidates <- c(
  file.path(data_root, "pca_init_manifest_centered_raw.csv"),
  file.path(data_root, "pca_init_manifest.csv")
)
pca_manifest_path <- pca_manifest_candidates[file.exists(pca_manifest_candidates)][1L] %||% pca_manifest_candidates[[1L]]
pca_manifest <- if (file.exists(pca_manifest_path)) {
  utils::read.csv(pca_manifest_path, stringsAsFactors = FALSE)
} else {
  data.frame(dataset = character(), pca_init_path = character(), stringsAsFactors = FALSE)
}

dataset_filter <- strsplit(args$datasets %||% paste(benchmark_datasets, collapse = ","), ",", fixed = TRUE)[[1L]]
dataset_filter <- trimws(dataset_filter)
manifest <- manifest[manifest$dataset %in% dataset_filter, , drop = FALSE]
manifest$dataset <- factor(manifest$dataset, levels = benchmark_datasets)
manifest <- manifest[order(manifest$dataset), , drop = FALSE]
manifest$dataset <- as.character(manifest$dataset)
if (nrow(manifest) == 0L) stop("No requested datasets found in manifest.", call. = FALSE)

standardize_knn <- function(obj) {
  if (!is.null(obj$indices) && !is.null(obj$distances)) {
    return(list(indices = obj$indices, distances = obj$distances))
  }
  if (!is.null(obj$idx) && !is.null(obj$dist)) {
    return(list(indices = obj$idx, distances = obj$dist))
  }
  if (!is.null(obj$nn.idx) && !is.null(obj$nn.dists)) {
    return(list(indices = obj$nn.idx, distances = obj$nn.dists))
  }
  if (!is.null(obj$index) && !is.null(obj$distance)) {
    return(list(indices = obj$index, distances = obj$distance))
  }
  stop("Cannot identify KNN indices/distances in object.", call. = FALSE)
}

drop_self_if_first <- function(indices, distances, target_k) {
  if (ncol(indices) > target_k) {
    self_first <- all(indices[, 1L] == seq_len(nrow(indices)))
    zero_first <- all(abs(distances[, 1L]) < 1e-12)
    if (isTRUE(self_first) || isTRUE(zero_first)) {
      indices <- indices[, -1L, drop = FALSE]
      distances <- distances[, -1L, drop = FALSE]
    }
  }
  if (ncol(indices) > target_k) {
    indices <- indices[, seq_len(target_k), drop = FALSE]
    distances <- distances[, seq_len(target_k), drop = FALSE]
  }
  list(indices = indices, distances = distances)
}

find_existing_cuvs_knn <- function(dataset, k) {
  candidates <- character()
  if (nzchar(benchmark1_dir)) {
    candidates <- c(
      candidates,
      file.path(benchmark1_dir, "knn_cuvs_nndescent", paste0(dataset, "_cuvs_nndescent_k", k, ".RData"))
    )
  }
  b1_dirs <- list.dirs("/mnt/sata_ssd", recursive = FALSE, full.names = TRUE)
  b1_dirs <- b1_dirs[grepl("(fastEmbedR|faissR)_BENCHMARK1_", basename(b1_dirs))]
  b1_dirs <- b1_dirs[order(file.info(b1_dirs)$mtime, decreasing = TRUE)]
  candidates <- c(
    candidates,
    file.path(b1_dirs, "knn_cuvs_nndescent", paste0(dataset, "_cuvs_nndescent_k", k, ".RData"))
  )
  candidates <- candidates[file.exists(candidates)]
  if (length(candidates)) candidates[[1L]] else ""
}

select_reliable_fast_nn_backends <- function() {
  if (!available_pkg("faissR")) return(character())
  cuda_ready <- isTRUE(tryCatch(faissR::cuda_available() && faissR::cuvs_available(), error = function(e) FALSE))
  faiss_cuda_ready <- isTRUE(tryCatch(faissR::cuda_available() && faissR::faiss_available(), error = function(e) FALSE))
  faiss_ready <- isTRUE(tryCatch(faissR::faiss_available(), error = function(e) FALSE))
  backends <- character()
  if (cuda_ready) {
    backends <- c(backends, "cuda_cuvs_cagra", "cuda_cuvs_nndescent", "cuda_cuvs_ivf_flat", "cuda_cuvs_bruteforce")
  }
  if (faiss_cuda_ready) {
    backends <- c("faiss_gpu_flat_l2", backends, "faiss_gpu_ivf_flat")
  }
  if (faiss_ready) {
    backends <- c(backends, "faiss_hnsw", "faiss_nndescent", "faiss_ivf", "faiss")
  }
  unique(backends)
}

precompute_dataset_inputs <- function(dataset_name) {
  row <- manifest[manifest$dataset == dataset_name, , drop = FALSE]
  if (nrow(row) != 1L) stop("Dataset not found in manifest: ", dataset_name, call. = FALSE)
  ds <- load_dataset(dataset_name, row$path)
  params <- dataset_parameters(dataset_name, nrow(ds$data))
  knn_info <- load_or_compute_knn(dataset_name, ds$data, target_k = max(params$k, params$rtsne_required_k), cache_k = params$knn_cache_k)
  pca_info <- load_or_compute_pca_init(dataset_name, ds$data)
  log_msg(
    "Prepared %s: perplexity=%s k=%d Rtsne_support_k=%d KNN cache=%s PCA cache=%s",
    dataset_name,
    format(params$perplexity, digits = 4),
    params$k,
    params$rtsne_required_k,
    knn_info$knn_cache_file,
    pca_info$pca_cache_file
  )
  invisible(list(knn = knn_info, pca = pca_info))
}

load_or_compute_knn <- function(dataset_name, x, target_k = NULL, cache_k = NULL) {
  if (is.null(target_k) || !is.finite(target_k)) target_k <- dataset_parameters(dataset_name, nrow(x))$k
  if (is.null(cache_k) || !is.finite(cache_k)) cache_k <- dataset_parameters(dataset_name, nrow(x))$knn_cache_k
  target_k <- as.integer(target_k)
  cache_k <- as.integer(cache_k)
  cache_dir <- file.path(out_dir, "knn_cache")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  local_cache <- file.path(cache_dir, paste0(dataset_name, "_scaled_euclidean_best_k", cache_k, ".RData"))
  data_cache <- file.path(data_root, dataset_name, paste0(dataset_name, "_centered_raw_euclidean_k", cache_k, "_nn.RData"))
  source <- "benchmark2_cache"
  if (file.exists(data_cache)) {
    obj <- load_rdata_object(data_cache, paste0("nn_centered_euclidean_k", cache_k))
    sx <- standardize_knn(obj)
    if (ncol(sx$indices) >= min(target_k, cache_k)) {
      return(list(knn = drop_self_if_first(sx$indices, sx$distances, min(target_k, cache_k)), source = "precomputed_centered_raw_data_cache", knn_sec = 0, knn_cache_file = data_cache, knn_cache_k = cache_k))
    }
  }
  if (file.exists(local_cache)) {
    obj <- load_rdata_object(local_cache, "nn_scaled_euclidean_k100")
    sx <- standardize_knn(obj)
    if (ncol(sx$indices) >= min(target_k, cache_k)) {
      return(list(knn = drop_self_if_first(sx$indices, sx$distances, min(target_k, cache_k)), source = source, knn_sec = 0, knn_cache_file = local_cache, knn_cache_k = cache_k))
    }
  }
  if (!available_pkg("faissR")) stop("faissR is required to compute KNN.", call. = FALSE)
  knn_backends <- select_reliable_fast_nn_backends()
  nn_scaled_euclidean_k100 <- NULL
  last_error <- NULL
  used_backend <- NA_character_
  t <- system.time({
    for (candidate_backend in knn_backends) {
      attempt <- tryCatch(
        faissR::nn_without_self(
          x,
          k = cache_k,
          backend = candidate_backend,
          n_threads = n_threads,
          metric = "euclidean"
        ),
        error = function(e) {
          last_error <<- paste(candidate_backend, conditionMessage(e), sep = ": ")
          NULL
        }
      )
      if (!is.null(attempt)) {
        nn_scaled_euclidean_k100 <- attempt
        used_backend <- candidate_backend
        break
      }
    }
  })[["elapsed"]]
  if (is.null(nn_scaled_euclidean_k100)) {
    stop("Could not compute fallback KNN for BENCHMARK #2. Last error: ", last_error, call. = FALSE)
  }
  attr(nn_scaled_euclidean_k100, "benchmark_scaled") <- TRUE
  attr(nn_scaled_euclidean_k100, "benchmark_metric") <- "euclidean"
  attr(nn_scaled_euclidean_k100, "benchmark_k") <- as.integer(cache_k)
  attr(nn_scaled_euclidean_k100, "benchmark_backend") <- used_backend
  attr(nn_scaled_euclidean_k100, "benchmark_dataset") <- dataset_name
  save(nn_scaled_euclidean_k100, file = local_cache, compress = "gzip")
  sx <- standardize_knn(nn_scaled_euclidean_k100)
  list(
    knn = drop_self_if_first(sx$indices, sx$distances, min(target_k, cache_k)),
    source = paste0("computed_benchmark2_scaled_euclidean_k", cache_k, "_", used_backend),
    knn_sec = as.numeric(t),
    knn_cache_file = local_cache,
    knn_cache_k = cache_k
  )
}

validate_ready_pca_init <- function(y, dataset_name, pca_file, n_expected) {
  y <- as.matrix(y[, 1:2, drop = FALSE])
  storage.mode(y) <- "double"
  if (nrow(y) != n_expected) {
    stop(
      "Precomputed PCA initialization row mismatch for ", dataset_name,
      ": file has ", nrow(y), " rows but dataset has ", n_expected,
      ". File: ", pca_file,
      call. = FALSE
    )
  }
  if (ncol(y) < 2L || any(!is.finite(y))) {
    stop("Invalid precomputed PCA initialization for ", dataset_name, ": ", pca_file, call. = FALSE)
  }
  y
}

extract_ready_pca_init <- function(pca_file, dataset_name, n_expected) {
  env <- new.env(parent = emptyenv())
  load(pca_file, envir = env)
  object_names <- ls(env, all.names = TRUE)

  if (exists("Y_init", envir = env, inherits = FALSE)) {
    return(validate_ready_pca_init(get("Y_init", envir = env, inherits = FALSE), dataset_name, pca_file, n_expected))
  }
  if (exists("pca_init", envir = env, inherits = FALSE)) {
    obj <- get("pca_init", envir = env, inherits = FALSE)
    if (!is.null(obj$Y_init)) {
      return(validate_ready_pca_init(obj$Y_init, dataset_name, pca_file, n_expected))
    }
  }

  for (nm in object_names) {
    obj <- get(nm, envir = env, inherits = FALSE)
    if (is.matrix(obj) || is.data.frame(obj)) {
      if (nrow(obj) == n_expected && ncol(obj) >= 2L) {
        return(validate_ready_pca_init(obj, dataset_name, pca_file, n_expected))
      }
    }
    if (is.list(obj)) {
      for (field in c("Y_init", "init", "layout", "scores", "pca", "x")) {
        candidate <- obj[[field]]
        if ((is.matrix(candidate) || is.data.frame(candidate)) &&
            nrow(candidate) == n_expected && ncol(candidate) >= 2L) {
          return(validate_ready_pca_init(candidate, dataset_name, pca_file, n_expected))
        }
      }
    }
  }

  stop("Could not find a ready t-SNE PCA initialization matrix in ", pca_file, call. = FALSE)
}

resolve_pca_manifest_path <- function(row) {
  path_cols <- intersect(c("pca_init_path", "path", "file", "pca_file"), names(row))
  candidates <- unique(unlist(lapply(path_cols, function(col) as.character(row[[col]])), use.names = FALSE))
  candidates <- candidates[nzchar(candidates)]
  candidates <- unique(c(
    candidates,
    file.path(data_root, candidates)
  ))
  hits <- candidates[file.exists(candidates)]
  if (length(hits)) normalizePath(hits[[1L]], mustWork = FALSE) else candidates[[1L]] %||% ""
}

load_or_compute_pca_init <- function(dataset_name, x) {
  ready_candidates <- Sys.glob(file.path(data_root, dataset_name, "*centered_raw*opentsne*pca*init*.RData"))
  ready_candidates <- ready_candidates[file.exists(ready_candidates)]
  pca_file <- if (length(ready_candidates)) {
    normalizePath(ready_candidates[[1L]], mustWork = FALSE)
  } else {
    row <- pca_manifest[pca_manifest$dataset == dataset_name, , drop = FALSE]
    if (nrow(row) < 1L) ""
    else resolve_pca_manifest_path(row[1L, , drop = FALSE])
  }
  if (!nzchar(pca_file)) {
    stop(
      "Missing precomputed PCA initialization for ", dataset_name,
      " in ", pca_manifest_path,
      ". Benchmark #2 now requires the saved t-SNE-ready PCA initialization and does not recompute or readjust it.",
      call. = FALSE
    )
  }
  if (!nzchar(pca_file) || !file.exists(pca_file)) {
    stop(
      "Precomputed PCA initialization file is missing for ", dataset_name,
      ". Expected from manifest: ", pca_file,
      call. = FALSE
    )
  }
  Y_init <- extract_ready_pca_init(pca_file, dataset_name, nrow(x))
  list(
    Y_init = Y_init,
    pca = NULL,
    pca_sec = 0,
    pca_source = "precomputed_ready_pca_init",
    pca_cache_file = pca_file
  )
}

extract_layout <- function(x) {
  if (length(dim(x)) == 3L) return(as.matrix(x[, , 1L, drop = TRUE]))
  if (is.matrix(x) || is.data.frame(x)) return(as.matrix(x))
  if (is.list(x)) {
    for (name in c("layout", "Y", "embedding", "embeddings", "data")) {
      if (!is.null(x[[name]])) return(extract_layout(x[[name]]))
    }
  }
  as.matrix(x)
}

coerce_layout <- function(x, n) {
  x <- extract_layout(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) {
    stop("The returned object is not an n x 2 embedding matrix.", call. = FALSE)
  }
  x[, 1:2, drop = FALSE]
}

metric_columns <- c(
  "trustworthiness", "continuity", "knn_preservation_15", "knn_preservation_30",
  "knn_preservation_50", "distance_spearman", "distance_pearson", "stress",
  "silhouette", "label_knn_accuracy", "ari", "nmi", "rare_class_recall"
)

empty_metrics <- function() {
  out <- as.data.frame(as.list(rep(NA_real_, length(metric_columns))), stringsAsFactors = FALSE)
  names(out) <- metric_columns
  out
}

evaluate_layout <- function(dataset_name, x, labels, layout, method, backend) {
  if (!available_pkg("fastEmbedR")) return(empty_metrics())
  n <- nrow(x)
  keep <- seq_len(n)
  if (n > metric_n) {
    set.seed(seed)
    keep <- sort(sample.int(n, metric_n))
  }
  metrics <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x[keep, , drop = FALSE],
      layout[keep, , drop = FALSE],
      labels = if (is.null(labels)) NULL else labels[keep],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(keep)),
      sample_size_for_local_metrics = min(3000L, length(keep)),
      seed = seed,
      method = method,
      backend = "cpu",
      n_threads = n_threads,
      dataset = dataset_name
    ),
    error = function(e) {
      warning("Evaluation failed for ", dataset_name, " / ", method, ": ", conditionMessage(e))
      empty_metrics()
    }
  )
  out <- empty_metrics()
  for (nm in intersect(names(out), names(metrics))) out[[nm]] <- metrics[[nm]][[1L]]
  out
}

plot_layout <- function(layout, labels, path, title) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  png(path, width = 1800, height = 1500, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(3, 3, 3, 1), bg = "white")
  if (is.null(labels)) {
    cols <- "#1f77b4"
  } else {
    f <- as.factor(labels)
    pal <- grDevices::hcl.colors(max(3L, nlevels(f)), "Dark 3")
    cols <- pal[as.integer(f)]
  }
  plot(layout[, 1], layout[, 2], pch = 16, cex = 0.35, col = cols,
       xlab = "dim 1", ylab = "dim 2", main = title)
}

sanitize <- function(x) gsub("[^A-Za-z0-9_+-]+", "_", x)

fast_tsne_path <- function() {
  candidates <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_PATH", ""),
    Sys.getenv("FAST_TSNE_PATH", ""),
    "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne",
    file.path(Sys.getenv("HOME"), ".local", "bin", "fast_tsne"),
    "/mnt/sata_ssd/FIt-SNE/fast_tsne",
    "/mnt/sata_ssd/FIt-SNE/build/bin/fast_tsne",
    Sys.which("fast_tsne")
  )
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1L) == 0L]
  if (length(candidates) == 0L) "" else normalizePath(candidates[[1L]], mustWork = FALSE)
}

fft_rtsne_info <- function() {
  for (package in c("fftRtsne", "Spectre")) {
    if (available_pkg(package) && exists("fftRtsne", envir = asNamespace(package), inherits = FALSE)) {
      fun <- get("fftRtsne", envir = asNamespace(package), inherits = FALSE)
      path <- fast_tsne_path()
      if ("fast_tsne_path" %in% names(formals(fun)) && !nzchar(path)) return(NULL)
      return(list(package = package, fun = fun, fast_tsne_path = path))
    }
  }
  wrapper <- "/mnt/sata_ssd/FIt-SNE/fast_tsne.R"
  path <- fast_tsne_path()
  if (file.exists(wrapper) && nzchar(path)) {
    env <- new.env(parent = .GlobalEnv)
    old_wd <- getwd()
    on.exit(setwd(old_wd), add = TRUE)
    setwd(dirname(wrapper))
    source(wrapper, local = env, chdir = FALSE)
    setwd(old_wd)
    env$FAST_TSNE_SCRIPT_DIR <- dirname(wrapper)
    assign("FAST_TSNE_SCRIPT_DIR", dirname(wrapper), envir = .GlobalEnv)
    if (exists("fftRtsne", envir = env, inherits = FALSE)) {
      fun <- get("fftRtsne", envir = env, inherits = FALSE)
      assign("FAST_TSNE_SCRIPT_DIR", dirname(wrapper), envir = environment(fun))
      return(list(
        package = "KlugerLab_FItSNE_source",
        fun = fun,
        fast_tsne_path = path
      ))
    }
  }
  NULL
}

run_with_timing <- function(expr, n) {
  gc()
  before <- read_peak_rss_gb()
  t <- system.time({
    value <- force(expr)
  })[["elapsed"]]
  after <- read_peak_rss_gb()
  peak <- suppressWarnings(max(before, after, na.rm = TRUE))
  if (!is.finite(peak)) peak <- NA_real_
  list(
    layout = coerce_layout(value, n),
    embedding_sec = as.numeric(t),
    peak_ram_gb = peak
  )
}

with_tsne_fft_grid <- function(grid, expr) {
  old <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
  if (is.null(grid) || is.na(grid)) {
    Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
  } else {
    Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = as.character(as.integer(grid)))
  }
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
    } else {
      Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old)
    }
  }, add = TRUE)
  force(expr)
}

method_specs <- function() {
  specs <- list(
    list(
      method = "fastEmbedR_opentsne_cpu_grid128",
      package = "fastEmbedR",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        with_tsne_fft_grid(
          128L,
          fastEmbedR::opentsne_knn(
            ctx$knn$indices,
            ctx$knn$distances,
            perplexity = ctx$perplexity,
            Y_init = ctx$Y_init,
            backend = "cpu",
            n_threads = n_threads,
            seed = seed,
            verbose = FALSE
          )
        )
      }
    ),
    list(
      method = "fastEmbedR_opentsne_cpu_grid256",
      package = "fastEmbedR",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        with_tsne_fft_grid(
          256L,
          fastEmbedR::opentsne_knn(
            ctx$knn$indices,
            ctx$knn$distances,
            perplexity = ctx$perplexity,
            Y_init = ctx$Y_init,
            backend = "cpu",
            n_threads = n_threads,
            seed = seed,
            verbose = FALSE
          )
        )
      }
    ),
    list(
      method = "fastEmbedR_opentsne_cpu_grid512",
      package = "fastEmbedR",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        with_tsne_fft_grid(
          512L,
          fastEmbedR::opentsne_knn(
            ctx$knn$indices,
            ctx$knn$distances,
            perplexity = ctx$perplexity,
            Y_init = ctx$Y_init,
            backend = "cpu",
            n_threads = n_threads,
            seed = seed,
            verbose = FALSE
          )
        )
      }
    ),
    list(
      method = "fastEmbedR_opentsne_cuda",
      package = "fastEmbedR",
      backend = "cuda",
      uses_precomputed_nn = TRUE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        with_tsne_fft_grid(
          512L,
          fastEmbedR::opentsne_knn(
            ctx$knn$indices,
            ctx$knn$distances,
            perplexity = ctx$perplexity,
            Y_init = ctx$Y_init,
            backend = "cuda",
            n_threads = n_threads,
            seed = seed,
            verbose = FALSE
          )
        )
      }
    ),
    list(
      method = "Rtsne_neighbors",
      package = "Rtsne",
      backend = "cpu",
      uses_precomputed_nn = TRUE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        Rtsne::Rtsne_neighbors(
          ctx$knn$indices,
          ctx$knn$distances,
          dims = 2L,
          perplexity = ctx$perplexity,
          theta = 0.5,
          max_iter = 1000L,
          Y_init = ctx$Y_init,
          num_threads = n_threads,
          verbose = FALSE
        )$Y
      }
    ),
    list(
      method = "Rtsne_Rtsne",
      package = "Rtsne",
      backend = "cpu",
      uses_precomputed_nn = FALSE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        Rtsne::Rtsne(
          ctx$x,
          dims = 2L,
          perplexity = ctx$perplexity,
          theta = 0.5,
          max_iter = 1000L,
          Y_init = ctx$Y_init,
          pca = FALSE,
          check_duplicates = FALSE,
          num_threads = n_threads,
          verbose = FALSE
        )$Y
      }
    ),
    list(
      method = "tsne_package",
      package = "tsne",
      backend = "cpu",
      uses_precomputed_nn = FALSE,
      uses_pca_init = TRUE,
      runner = function(ctx) {
        tsne::tsne(
          ctx$x,
          initial_config = ctx$Y_init,
          k = 2L,
          perplexity = ctx$perplexity,
          max_iter = 1000L
        )
      }
    )
  )
  if (include_fitsne) {
    specs[[length(specs) + 1L]] <- list(
      method = "KlugerLab_FItSNE",
      package = "KlugerLab_FItSNE",
      backend = "cpu_fft",
      uses_precomputed_nn = FALSE,
      uses_pca_init = TRUE,
      availability = function() !is.null(fft_rtsne_info()),
      runner = function(ctx) {
        info <- fft_rtsne_info()
        if (is.null(info)) {
          stop("No usable fftRtsne/Spectre wrapper and fast_tsne executable found.", call. = FALSE)
        }
        fun <- info$fun
        f <- names(formals(fun))
        call_args <- list(
          X = ctx$x,
          dims = 2L,
          perplexity = ctx$perplexity,
          max_iter = 1000L,
          rand_seed = seed,
          theta = 0.5,
          nthreads = n_threads,
          fast_tsne_path = info$fast_tsne_path,
          verbose = FALSE
        )
        if ("Y_init" %in% f) call_args$Y_init <- ctx$Y_init
        if ("initial_config" %in% f) call_args$initial_config <- ctx$Y_init
        if ("init" %in% f) call_args$init <- ctx$Y_init
        if ("initialization" %in% f) call_args$initialization <- ctx$Y_init
        call_args <- call_args[names(call_args) %in% f]
        do.call(fun, call_args)
      }
    )
  }
  specs
}

result_template <- function(dataset_name, method, package, backend, status, error_message = NA_character_) {
  data.frame(
    dataset = dataset_name,
    method = method,
    package = package,
    backend_requested = backend,
    backend_used = if (status == "success") backend else NA_character_,
    status = status,
    error_message = error_message,
    n = NA_integer_,
    p = NA_integer_,
    seed = seed,
    k = NA_integer_,
    perplexity = NA_real_,
    uses_precomputed_nn = NA,
    timing_scope = NA_character_,
    uses_pca_init = NA,
    pca_source = NA_character_,
    pca_sec = NA_real_,
    pca_cache_file = NA_character_,
    knn_source = NA_character_,
    knn_sec = NA_real_,
    knn_cache_k = NA_integer_,
    knn_cache_file = NA_character_,
    embedding_sec = NA_real_,
    total_sec = NA_real_,
    peak_ram_gb = NA_real_,
    layout_rdata = NA_character_,
    plot_png = NA_character_,
    parameters_json = NA_character_,
    empty_metrics(),
    stringsAsFactors = FALSE
  )
}

run_one <- function(dataset_name, method_name, row_out) {
  specs <- method_specs()
  spec <- specs[vapply(specs, function(z) identical(z$method, method_name), logical(1))][[1L]]
  ds_row <- manifest[manifest$dataset == dataset_name, , drop = FALSE]
  if (nrow(ds_row) != 1L) stop("Dataset not found in manifest: ", dataset_name, call. = FALSE)
  ds <- load_dataset(dataset_name, ds_row$path[[1L]])
  x <- ds$data
  labels <- ds$labels
  n <- nrow(x)
  p <- ncol(x)
  params <- dataset_parameters(dataset_name, n)
  perplexity <- params$perplexity
  max_dataset_perplexity <- floor((n - 1L) / 3L)
  if (!is.finite(perplexity) || perplexity < 1L || perplexity > max_dataset_perplexity) {
    row <- result_template(
      dataset_name,
      spec$method,
      spec$package,
      spec$backend,
      "invalid_parameter",
      sprintf("Shared perplexity %.3f is too large for n = %d; maximum allowed is %d.", perplexity, n, max_dataset_perplexity)
    )
    row$n <- n
    row$p <- p
    row$k <- params$k
    row$perplexity <- perplexity
    row$uses_precomputed_nn <- isTRUE(spec$uses_precomputed_nn)
    row$timing_scope <- if (isTRUE(spec$uses_precomputed_nn)) "saved_knn_embedding_only" else "raw_x_internal_affinity_included"
    row$uses_pca_init <- isTRUE(spec$uses_pca_init)
    row$parameters_json <- as.character(json_or_text(list(
      perplexity = perplexity,
      shared_perplexity = TRUE,
      k = params$k,
      dataset_k = params$k,
      rtsne_required_k = params$rtsne_required_k,
      knn_cache_k = params$knn_cache_k,
      parameter_policy = params$policy
    )))
    utils::write.csv(row, row_out, row.names = FALSE)
    return(invisible(row))
  }
  pca_info <- load_or_compute_pca_init(dataset_name, x)
  Y_init <- pca_info$Y_init
  knn_info <- list(
    knn = NULL,
    source = if (isTRUE(spec$uses_precomputed_nn)) NA_character_ else "not_used_raw_x_method",
    knn_sec = NA_real_,
    knn_cache_file = NA_character_,
    knn_cache_k = params$knn_cache_k
  )
  if (isTRUE(spec$uses_precomputed_nn)) {
    target_k <- if (identical(spec$method, "Rtsne_neighbors")) params$rtsne_required_k else params$k
    knn_info <- load_or_compute_knn(dataset_name, x, target_k = target_k, cache_k = params$knn_cache_k)
  }
  spec_available <- if (!is.null(spec$availability) && is.function(spec$availability)) {
    isTRUE(spec$availability())
  } else {
    available_pkg(spec$package)
  }
  if (!isTRUE(spec_available)) {
    row <- result_template(dataset_name, spec$method, spec$package, spec$backend, "not_installed",
                           paste0("Required package/executable for ", spec$method, " is not installed or not discoverable."))
    row$n <- n
    row$p <- p
    row$k <- params$k
    row$perplexity <- perplexity
    row$uses_pca_init <- isTRUE(spec$uses_pca_init)
    row$pca_source <- pca_info$pca_source
    row$pca_sec <- pca_info$pca_sec
    row$pca_cache_file <- pca_info$pca_cache_file
    row$uses_precomputed_nn <- isTRUE(spec$uses_precomputed_nn)
    row$timing_scope <- if (isTRUE(spec$uses_precomputed_nn)) "saved_knn_embedding_only" else "raw_x_internal_affinity_included"
    row$knn_source <- knn_info$source
    row$knn_sec <- knn_info$knn_sec
    row$knn_cache_k <- knn_info$knn_cache_k %||% NA_integer_
    row$knn_cache_file <- knn_info$knn_cache_file %||% NA_character_
    row$parameters_json <- as.character(json_or_text(list(
      timing_scope = row$timing_scope,
      saved_knn_required = isTRUE(spec$uses_precomputed_nn),
      perplexity = perplexity,
      shared_perplexity = TRUE,
      dataset_k = params$k,
      rtsne_required_k = params$rtsne_required_k,
      knn_cache_k = params$knn_cache_k,
      parameter_policy = params$policy,
      parameter_reason = params$reason,
      knn_metric = "euclidean",
      knn_input_scaled = FALSE,
      pca_cache_file = pca_info$pca_cache_file
    )))
    utils::write.csv(row, row_out, row.names = FALSE)
    return(invisible(row))
  }
  ctx <- list(
    dataset = dataset_name,
    x = x,
    labels = labels,
    knn = knn_info$knn,
    Y_init = Y_init,
    pca_source = pca_info$pca_source,
    pca_cache_file = pca_info$pca_cache_file,
    perplexity = perplexity
  )
  status <- "success"
  error <- NA_character_
  measured <- NULL
  tryCatch({
    set.seed(seed)
    measured <- run_with_timing(spec$runner(ctx), n)
  }, error = function(e) {
    status <<- "failed"
    error <<- conditionMessage(e)
  })
  row <- result_template(dataset_name, spec$method, spec$package, spec$backend, status, error)
  row$n <- n
  row$p <- p
  row$k <- params$k
  row$perplexity <- perplexity
  row$uses_precomputed_nn <- isTRUE(spec$uses_precomputed_nn)
  row$timing_scope <- if (isTRUE(spec$uses_precomputed_nn)) "saved_knn_embedding_only" else "raw_x_internal_affinity_included"
  row$uses_pca_init <- isTRUE(spec$uses_pca_init)
  row$pca_source <- pca_info$pca_source
  row$pca_sec <- pca_info$pca_sec
  row$pca_cache_file <- pca_info$pca_cache_file
  row$knn_source <- knn_info$source
  row$knn_sec <- knn_info$knn_sec
  row$knn_cache_k <- knn_info$knn_cache_k %||% NA_integer_
  row$knn_cache_file <- knn_info$knn_cache_file %||% NA_character_
  row$parameters_json <- as.character(json_or_text(list(
    k = params$k,
    knn_cache_k = params$knn_cache_k,
    rtsne_required_k = params$rtsne_required_k,
    knn_metric = "euclidean",
    knn_input_scaled = FALSE,
    perplexity = perplexity,
    shared_perplexity = TRUE,
    parameter_policy = params$policy,
    parameter_reason = params$reason,
    max_iter = 1000L,
    seed = seed,
    n_threads = n_threads,
    pca_sec = pca_info$pca_sec,
    pca_cache_file = pca_info$pca_cache_file,
    pca_adjustment = "centered scores scaled to RMS 1e-4",
    init = if (isTRUE(spec$uses_pca_init)) ctx$pca_source else "method_default",
    uses_precomputed_knn = isTRUE(spec$uses_precomputed_nn),
    timing_scope = row$timing_scope
  )))
  if (!is.null(measured)) {
    layout <- measured$layout
    layout_file <- file.path(out_dir, "layouts", paste0(sanitize(dataset_name), "__", sanitize(spec$method), ".RData"))
    plot_file <- file.path(out_dir, "plots", paste0(sanitize(dataset_name), "__", sanitize(spec$method), ".png"))
    dir.create(dirname(layout_file), recursive = TRUE, showWarnings = FALSE)
    metrics <- evaluate_layout(dataset_name, x, labels, layout, spec$method, spec$backend)
    layout_result <- list(
      dataset = dataset_name,
      method = spec$method,
      backend = spec$backend,
      layout = layout,
      labels = labels,
      metrics = metrics,
      parameters = list(
        k = params$k,
        perplexity = perplexity,
        seed = seed,
        n_threads = n_threads,
        pca_source = ctx$pca_source,
        pca_cache_file = ctx$pca_cache_file,
        uses_precomputed_knn = isTRUE(spec$uses_precomputed_nn),
        timing_scope = row$timing_scope
      )
    )
    save(layout_result, file = layout_file, compress = "gzip")
    plot_layout(layout, labels, plot_file, paste(dataset_name, spec$method))
    row$embedding_sec <- measured$embedding_sec
    row$total_sec <- sum(c(
      measured$embedding_sec,
      if (is.finite(knn_info$knn_sec)) knn_info$knn_sec else 0,
      if (is.finite(pca_info$pca_sec)) pca_info$pca_sec else 0
    ), na.rm = TRUE)
    row$peak_ram_gb <- measured$peak_ram_gb
    row$layout_rdata <- layout_file
    row$plot_png <- plot_file
    for (nm in names(metrics)) {
      if (nm %in% names(row)) row[[nm]] <- metrics[[nm]][[1L]]
    }
  }
  utils::write.csv(row, row_out, row.names = FALSE)
  invisible(row)
}

combine_worker_rows <- function(worker_dir) {
  files <- list.files(worker_dir, pattern = "\\.csv$", full.names = TRUE)
  if (!length(files)) return(data.frame())
  rows <- lapply(files, function(f) utils::read.csv(f, stringsAsFactors = FALSE))
  common <- Reduce(union, lapply(rows, names))
  rows <- lapply(rows, function(x) {
    miss <- setdiff(common, names(x))
    for (m in miss) x[[m]] <- NA
    x[, common, drop = FALSE]
  })
  do.call(rbind, rows)
}

score01 <- function(x, higher_is_better = TRUE) {
  x <- suppressWarnings(as.numeric(x))
  out <- rep(NA_real_, length(x))
  ok <- is.finite(x)
  if (!any(ok)) return(out)
  rng <- range(x[ok], na.rm = TRUE)
  if (!is.finite(diff(rng)) || diff(rng) < .Machine$double.eps) {
    out[ok] <- 1
    return(out)
  }
  val <- (x - rng[[1L]]) / diff(rng)
  if (!higher_is_better) val <- 1 - val
  out[ok] <- pmax(0, pmin(1, val[ok]))
  out
}

safe_row_mean <- function(x) {
  out <- rowMeans(x, na.rm = TRUE)
  out[!is.finite(out)] <- NA_real_
  out
}

add_benchmark_scores <- function(results) {
  if (!nrow(results)) return(results)
  score_cols <- c("runtime_score", "memory_score", "structure_score", "label_score", "combined_score")
  for (nm in score_cols) results[[nm]] <- NA_real_
  ok <- which(results$status == "success")
  if (!length(ok)) return(results)
  for (idx in split(ok, results$dataset[ok])) {
    runtime_raw <- suppressWarnings(as.numeric(results$total_sec[idx]))
    fallback_runtime <- suppressWarnings(as.numeric(results$embedding_sec[idx]))
    runtime_raw[!is.finite(runtime_raw)] <- fallback_runtime[!is.finite(runtime_raw)]
    memory_raw <- suppressWarnings(as.numeric(results$peak_ram_gb[idx]))
    structure <- safe_row_mean(cbind(
      score01(results$trustworthiness[idx], TRUE),
      score01(results$continuity[idx], TRUE),
      score01(results$knn_preservation_15[idx], TRUE),
      score01(results$knn_preservation_30[idx], TRUE),
      score01(results$knn_preservation_50[idx], TRUE),
      score01(results$distance_spearman[idx], TRUE),
      score01(results$distance_pearson[idx], TRUE),
      score01(results$stress[idx], FALSE)
    ))
    labels <- safe_row_mean(cbind(
      score01(results$silhouette[idx], TRUE),
      score01(results$label_knn_accuracy[idx], TRUE),
      score01(results$ari[idx], TRUE),
      score01(results$nmi[idx], TRUE),
      score01(results$rare_class_recall[idx], TRUE)
    ))
    runtime <- score01(runtime_raw, FALSE)
    memory <- score01(memory_raw, FALSE)
    quality <- safe_row_mean(cbind(structure, labels))
    combined <- 0.40 * structure + 0.25 * labels + 0.20 * runtime + 0.15 * memory
    combined[!is.finite(combined)] <- safe_row_mean(cbind(quality, runtime, memory))[!is.finite(combined)]
    results$runtime_score[idx] <- runtime
    results$memory_score[idx] <- memory
    results$structure_score[idx] <- structure
    results$label_score[idx] <- labels
    results$combined_score[idx] <- combined
  }
  results
}

write_ranked_results <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  ranked <- ok[order(ok$dataset, -ok$combined_score, ok$total_sec, ok$peak_ram_gb), , drop = FALSE]
  utils::write.csv(ranked, file.path(out_dir, "benchmark2_ranked_speed_memory_quality.csv"), row.names = FALSE)
  best_combined <- do.call(rbind, lapply(split(ranked, ranked$dataset), head, 1L))
  utils::write.csv(best_combined, file.path(out_dir, "benchmark2_best_by_dataset_combined.csv"), row.names = FALSE)
  best_quality <- ok[order(ok$dataset, -ok$structure_score, -ok$label_score), , drop = FALSE]
  best_quality <- do.call(rbind, lapply(split(best_quality, best_quality$dataset), head, 1L))
  utils::write.csv(best_quality, file.path(out_dir, "benchmark2_best_by_dataset_quality.csv"), row.names = FALSE)
  invisible(NULL)
}

make_barplots <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  png(file.path(out_dir, "benchmark2_runtime_barplot.png"), width = 2200, height = 1500, res = 180)
  par(mar = c(10, 5, 4, 1), bg = "white")
  labels <- paste(ok$dataset, ok$method, sep = "\n")
  vals <- ok$embedding_sec
  cols <- ifelse(grepl("cuda", ok$backend_requested, ignore.case = TRUE), "#D55E00", "#0072B2")
  barplot(vals, names.arg = labels, las = 2, cex.names = 0.55, col = cols,
          ylab = "Embedding seconds", main = "BENCHMARK #2 t-SNE embedding runtime")
  legend("topright", legend = c("CPU/raw", "CUDA"), fill = c("#0072B2", "#D55E00"), bty = "n")
  dev.off()

  if ("trustworthiness" %in% names(ok)) {
    png(file.path(out_dir, "benchmark2_trust_barplot.png"), width = 2200, height = 1500, res = 180)
    par(mar = c(10, 5, 4, 1), bg = "white")
    barplot(ok$trustworthiness, names.arg = labels, las = 2, cex.names = 0.55, col = cols,
            ylab = "Trustworthiness", main = "BENCHMARK #2 t-SNE embedding quality")
    legend("topright", legend = c("CPU/raw", "CUDA"), fill = c("#0072B2", "#D55E00"), bty = "n")
    dev.off()
  }

  if ("combined_score" %in% names(ok)) {
    png(file.path(out_dir, "benchmark2_combined_score_barplot.png"), width = 2200, height = 1500, res = 180)
    par(mar = c(10, 5, 4, 1), bg = "white")
    barplot(ok$combined_score, names.arg = labels, las = 2, cex.names = 0.55, col = cols,
            ylab = "Combined score", main = "BENCHMARK #2 speed, memory, and quality score")
    legend("topright", legend = c("CPU/raw", "CUDA"), fill = c("#0072B2", "#D55E00"), bty = "n")
    dev.off()
  }
}

write_methods_file <- function(out_dir) {
  txt <- c(
    "# BENCHMARK #2 Material and Methods",
    "",
    "This benchmark compares t-SNE-family implementations across the curated fastEmbedR datasets saved under `/mnt/sata_ssd/fastEmbedR/Data` on the chiamaka GPU workstation.",
    "",
    "Datasets are loaded from `dataset_manifest.csv`. Before KNN construction or embedding, non-finite and zero-variance columns are removed and the remaining variables are mean-centered without variance scaling. Labels are used only for post-hoc quality metrics and plots, not for fitting.",
    "",
    "Nearest-neighbour input for saved-KNN methods: after scaling each dataset, the script computes or reuses a saved Euclidean non-self `faissR::nn_without_self()` cache. The cache k is dataset-specific and is at least the dataset embedding k and, for `Rtsne_neighbors`, at least `ceiling(3 * perplexity) + 1`. The backend is chosen in order from fast/reliable CUDA/cuVS methods when available, then FAISS CPU methods.",
    "",
    "Methods that cannot consume a saved KNN object are still run from the centered raw matrix. Their result rows have `uses_precomputed_nn = FALSE` and `timing_scope = \"raw_x_internal_affinity_included\"`; therefore their `embedding_sec` includes any neighbour, affinity, PCA/whitening, and optimizer work performed internally by the package. Saved-KNN rows have `timing_scope = \"saved_knn_embedding_only\"`.",
    "",
    sprintf("Initialization: Benchmark #2 loads the precomputed t-SNE-ready PCA initialization from `%s`. The saved matrix is used as-is: the benchmark does not recenter, rescale, recompute, or otherwise modify the PCA initialization. These files are expected to contain a two-column `Y_init` matrix already centered and scaled for t-SNE initialization.", basename(pca_manifest_path)),
    "",
    "Compared implementations:",
    "",
    sprintf("Backend filter for this run: %s. By default Benchmark #2 runs CPU, CPU FFT/FIt-SNE, and CUDA rows; other backends can be requested with `--backends=`.", paste(backend_filter, collapse = ", ")),
    "",
    "- `fastEmbedR_opentsne_cpu_grid128`: native fastEmbedR openTSNE-style optimizer from precomputed KNN, CPU backend, four CPU threads, FFT grid forced to 128.",
    "- `fastEmbedR_opentsne_cpu_grid256`: native fastEmbedR openTSNE-style optimizer from precomputed KNN, CPU backend, four CPU threads, FFT grid forced to 256.",
    "- `fastEmbedR_opentsne_cpu_grid512`: native fastEmbedR openTSNE-style optimizer from precomputed KNN, CPU backend, four CPU threads, FFT grid forced to 512.",
    "- `fastEmbedR_opentsne_cuda`: native fastEmbedR openTSNE-style optimizer from precomputed KNN, CUDA backend, FFT grid forced to 512.",
    "- `Rtsne_neighbors`: `Rtsne::Rtsne_neighbors()` from the saved precomputed KNN and saved ready PCA initialization.",
    "- `Rtsne_Rtsne`: direct `Rtsne::Rtsne()` from the centered raw matrix and saved ready PCA initialization; this row does not use the saved KNN and its timing includes Rtsne internal neighbour/affinity work.",
    "- `tsne_package`: `tsne::tsne()` from the centered raw matrix and saved ready PCA initialization; this row does not use the saved KNN and its timing includes internal package work.",
    "- `KlugerLab_FItSNE`: FIt-SNE through an installed `fftRtsne`/`Spectre` wrapper from the centered raw matrix, using saved ready PCA initialization when the wrapper supports it; this row does not use the saved KNN and its timing includes internal package work.",
    "",
    "Default parameter policy: each dataset receives one dataset-level perplexity/k pair from the benchmark policy table, unless the user explicitly supplies `--perplexity` or `--k`. Every t-SNE implementation uses the same perplexity for the same dataset. fastEmbedR openTSNE rows use the dataset k as saved non-self neighbour support. `Rtsne_neighbors` receives the same perplexity but may require extra neighbour columns, `ceiling(3 * perplexity) + 1`, only as support for Rtsne's own perplexity search. Perplexity is not lowered per method; datasets too small for the chosen value are marked `invalid_parameter`.",
    "",
    "Dataset defaults:",
    paste(utils::capture.output(print(dataset_parameter_defaults[, c("dataset", "perplexity", "k", "reason")], row.names = FALSE)), collapse = "\n"),
    "",
    sprintf("Other settings: max_iter = 1000, seed = %d, CPU threads = %d.", seed, n_threads),
    "",
    sprintf("Each method-dataset worker is executed with a %d second timeout. Failed, unavailable, or timed-out rows are retained in the result table.", timeout_sec),
    "",
    "Quality metrics are computed on a reproducible sample of up to 5000 cells/samples using `fastEmbedR::evaluate_embedding()`: trustworthiness, continuity, kNN preservation at 15/30/50, global distance correlations, stress, silhouette, label kNN accuracy, ARI, NMI, and rare-class recall when labels are available.",
    "",
    "Derived comparison scores are added per dataset: `runtime_score` and `memory_score` are scaled so higher is better; `structure_score` combines trustworthiness, continuity, kNN preservation, global distance correlations, and inverse stress; `label_score` combines silhouette, label kNN accuracy, ARI, NMI, and rare-class recall when available. `combined_score = 0.40 * structure_score + 0.25 * label_score + 0.20 * runtime_score + 0.15 * memory_score`. Raw metrics remain in the main CSV and should be inspected alongside the combined score.",
    "",
    "Outputs: `benchmark2_tsne_results.csv`, ranked speed-memory-quality CSVs, per-method `.RData` layout files, per-method PNG plots, runtime/quality/combined-score barplots, and a narrative result summary."
  )
  writeLines(txt, file.path(out_dir, "BENCHMARK2_MATERIALS_AND_METHODS.md"))
}

write_summary_file <- function(results, out_dir) {
  ok <- results[results$status == "success", , drop = FALSE]
  lines <- c("# BENCHMARK #2 Results Summary", "")
  if (!nrow(results)) {
    lines <- c(lines, "No worker rows were collected.")
  } else {
    lines <- c(lines, sprintf("Rows collected: %d. Successful rows: %d.", nrow(results), nrow(ok)), "")
    if (nrow(ok)) {
      best_time <- ok[order(ok$dataset, ok$embedding_sec), c("dataset", "method", "backend_requested", "embedding_sec", "trustworthiness", "label_knn_accuracy")]
      best_time <- do.call(rbind, lapply(split(best_time, best_time$dataset), head, 1L))
      lines <- c(lines, "Fastest successful method per dataset:", "", capture.output(print(best_time, row.names = FALSE)), "")
      best_quality <- ok[order(ok$dataset, -ok$trustworthiness), c("dataset", "method", "backend_requested", "embedding_sec", "trustworthiness", "label_knn_accuracy")]
      best_quality <- do.call(rbind, lapply(split(best_quality, best_quality$dataset), head, 1L))
      lines <- c(lines, "Highest trustworthiness per dataset:", "", capture.output(print(best_quality, row.names = FALSE)), "")
    if ("combined_score" %in% names(ok)) {
      best_combined <- ok[order(ok$dataset, -ok$combined_score), c("dataset", "method", "backend_requested", "embedding_sec", "peak_ram_gb", "trustworthiness", "label_knn_accuracy", "combined_score")]
      best_combined <- do.call(rbind, lapply(split(best_combined, best_combined$dataset), head, 1L))
      lines <- c(lines, "Best combined speed-memory-quality score per dataset:", "", capture.output(print(best_combined, row.names = FALSE)), "")
    }
    rtsne_direct <- ok[ok$method == "Rtsne_Rtsne", c("dataset", "method", "timing_scope", "embedding_sec", "total_sec", "trustworthiness", "label_knn_accuracy"), drop = FALSE]
    if (nrow(rtsne_direct)) {
      lines <- c(lines, "Direct Rtsne::Rtsne() speed rows:", "", capture.output(print(rtsne_direct, row.names = FALSE)), "")
    }
  }
    failed <- results[results$status != "success", c("dataset", "method", "status", "error_message"), drop = FALSE]
    if (nrow(failed)) {
      lines <- c(lines, "Failed/unavailable rows:", "", capture.output(print(failed, row.names = FALSE)), "")
    }
    lines <- c(lines, "Comment:", "",
               "Benchmark #2 now reports two timing scopes. `saved_knn_embedding_only` rows, including fastEmbedR openTSNE and Rtsne_neighbors, consume the saved centered Euclidean KNN cache and saved ready PCA initialization. `raw_x_internal_affinity_included` rows, including tsne::tsne() and KlugerLab FIt-SNE wrappers, run from the centered raw matrix because their public APIs do not consume the saved KNN object directly; their timing includes their own internal neighbour/affinity work.")
  }
  writeLines(lines, file.path(out_dir, "BENCHMARK2_RESULTS_SUMMARY.md"))
}

if (worker) {
  dataset_name <- args$dataset %||% stop("--dataset is required in worker mode", call. = FALSE)
  method_name <- args$method %||% stop("--method is required in worker mode", call. = FALSE)
  row_out <- args$row_out %||% stop("--row_out is required in worker mode", call. = FALSE)
  row <- tryCatch(
    run_one(dataset_name, method_name, row_out),
    error = function(e) {
      spec <- method_specs()[vapply(method_specs(), function(z) identical(z$method, method_name), logical(1))][[1L]]
      row <- result_template(dataset_name, method_name, spec$package, spec$backend, "failed", conditionMessage(e))
      utils::write.csv(row, row_out, row.names = FALSE)
      row
    }
  )
  quit(save = "no", status = if (identical(row$status[[1L]], "success")) 0L else 0L)
}

write_methods_file(out_dir)
worker_dir <- file.path(out_dir, "worker_rows")
dir.create(worker_dir, recursive = TRUE, showWarnings = FALSE)

specs_all <- method_specs()
methods <- vapply(
  specs_all[vapply(specs_all, function(z) tolower(z$backend) %in% backend_filter, logical(1L))],
  `[[`,
  character(1L),
  "method"
)
if (!include_raw_x) {
  specs <- specs_all[vapply(specs_all, function(z) tolower(z$backend) %in% backend_filter, logical(1L))]
  methods <- vapply(specs[vapply(specs, function(z) isTRUE(z$uses_precomputed_nn), logical(1))], `[[`, character(1L), "method")
}
if (!is.null(args$methods)) {
  method_filter <- trimws(strsplit(args$methods, ",", fixed = TRUE)[[1L]])
  method_filter[method_filter == "Rtsne_package"] <- "Rtsne_Rtsne"
  methods <- methods[methods %in% method_filter]
  if (!length(methods)) stop("No requested methods found in BENCHMARK #2 method list.", call. = FALSE)
}

log_msg("BENCHMARK #2 output: %s", out_dir)
log_msg("Datasets: %s", paste(manifest$dataset, collapse = ", "))
log_msg("Backends: %s", paste(backend_filter, collapse = ", "))
log_msg("Methods: %s", paste(methods, collapse = ", "))
if (is.finite(benchmark_perplexity) || is.finite(k_override)) {
  log_msg("User parameter override: perplexity=%s k=%s", as.character(benchmark_perplexity), as.character(k_override))
} else {
  log_msg("Dataset-aware parameter policy is active")
}

log_msg("Using per-worker KNN/PCA preparation so dataset failures do not stop the full benchmark")

cmd_args_all <- commandArgs(FALSE)
file_arg <- cmd_args_all[grepl("^--file=", cmd_args_all)]
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else "tools/benchmark2_tsne_speed_accuracy.R"
script_path <- normalizePath(script_path, mustWork = FALSE)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
for (dataset_name in manifest$dataset) {
  ds_row <- manifest[manifest$dataset == dataset_name, , drop = FALSE]
  ds_n <- nrow(load_dataset(dataset_name, ds_row$path[[1L]])$data)
  params <- dataset_parameters(dataset_name, ds_n)
  for (method_name in methods) {
    row_file <- file.path(worker_dir, paste0(sanitize(dataset_name), "__", sanitize(method_name), ".csv"))
    if (file.exists(row_file)) {
      log_msg("Skipping existing %s / %s", dataset_name, method_name)
      next
    }
    log_msg("Running %s / %s", dataset_name, method_name)
    worker_perplexity_arg <- if (is.finite(benchmark_perplexity)) as.character(params$perplexity) else "auto"
    worker_k_arg <- if (is.finite(k_override)) as.character(params$k) else "auto"
    worker_knn_k_arg <- if (is.finite(knn_cache_k_override)) as.character(params$knn_cache_k) else "auto"
    cmd <- sprintf(
      "timeout %d Rscript %s --worker=TRUE --data_root=%s --out_dir=%s --benchmark1_dir=%s --dataset=%s --method=%s --perplexity=%s --k=%s --knn_k=%s --threads=%d --timeout=%d --seed=%d --metric_n=%d --include_raw_x=%s --include_fitsne=%s --row_out=%s",
      timeout_sec,
      shQuote(script_path),
      shQuote(data_root),
      shQuote(out_dir),
      shQuote(benchmark1_dir),
      shQuote(dataset_name),
      shQuote(method_name),
      shQuote(worker_perplexity_arg),
      shQuote(worker_k_arg),
      shQuote(worker_knn_k_arg),
      n_threads,
      timeout_sec,
      seed,
      metric_n,
      if (include_raw_x) "TRUE" else "FALSE",
      if (include_fitsne) "TRUE" else "FALSE",
      shQuote(row_file)
    )
    code <- system(cmd)
    if (!file.exists(row_file)) {
      spec <- method_specs()[vapply(method_specs(), function(z) identical(z$method, method_name), logical(1))][[1L]]
      row <- result_template(dataset_name, method_name, spec$package, spec$backend, "timeout",
                             paste0("Worker exceeded timeout or terminated with code ", code, "."))
      utils::write.csv(row, row_file, row.names = FALSE)
    }
  }
  results_partial <- combine_worker_rows(worker_dir)
  results_partial <- add_benchmark_scores(results_partial)
  utils::write.csv(results_partial, file.path(out_dir, "benchmark2_tsne_results_partial.csv"), row.names = FALSE)
  if (nrow(results_partial)) {
    write_ranked_results(results_partial, out_dir)
    make_barplots(results_partial, out_dir)
    write_summary_file(results_partial, out_dir)
  }
}

results <- combine_worker_rows(worker_dir)
results <- add_benchmark_scores(results)
utils::write.csv(results, file.path(out_dir, "benchmark2_tsne_results.csv"), row.names = FALSE)
if (nrow(results)) {
  ok <- results[results$status == "success", , drop = FALSE]
  if (nrow(ok)) {
    best_by_dataset <- do.call(rbind, lapply(split(ok, ok$dataset), function(z) z[order(z$embedding_sec), , drop = FALSE][1L, , drop = FALSE]))
    utils::write.csv(best_by_dataset, file.path(out_dir, "benchmark2_best_by_dataset_runtime.csv"), row.names = FALSE)
  }
  write_ranked_results(results, out_dir)
  make_barplots(results, out_dir)
  write_summary_file(results, out_dir)
}

log_msg("BENCHMARK #2 finished: %s", out_dir)
