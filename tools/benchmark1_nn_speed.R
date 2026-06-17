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

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

args <- parse_args(commandArgs(trailingOnly = TRUE))

data_root <- args$data_root %||% "/mnt/sata_ssd/fastEmbedR_Data"
out_dir <- args$out_dir %||% file.path("/mnt/sata_ssd", paste0("faissR_BENCHMARK1_", format(Sys.time(), "%Y%m%d_%H%M%S")))
k <- as.integer(args$k %||% "50")
n_threads <- as.integer(args$threads %||% "4")
metric <- tolower(args$metric %||% "l2")
if (metric %in% c("euclidean", "l2")) metric <- "l2"
if (metric %in% c("ip", "innerproduct", "inner_product")) metric <- "inner_product"
if (!metric %in% c("l2", "cosine", "inner_product")) metric <- "l2"
timeout_sec <- as.integer(args$timeout %||% "600")
worker <- isTRUE(as.logical(args$worker %||% FALSE))
quality_eval_max_n <- as.integer(args$quality_n %||% Sys.getenv("FAISSR_BENCHMARK1_QUALITY_N", "512"))
if (length(quality_eval_max_n) != 1L || is.na(quality_eval_max_n) || !is.finite(quality_eval_max_n) || quality_eval_max_n < 1L) {
  quality_eval_max_n <- 512L
}
quality_eval_max_ops <- as.numeric(args$quality_max_ops %||% Sys.getenv("FAISSR_BENCHMARK1_QUALITY_MAX_OPS", "5e9"))
if (length(quality_eval_max_ops) != 1L || is.na(quality_eval_max_ops) || !is.finite(quality_eval_max_ops) || quality_eval_max_ops < 1) {
  quality_eval_max_ops <- 5e9
}
if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
  n_threads <- 4L
}
n_threads <- as.integer(n_threads)

configure_cpu_threads <- function(n_threads) {
  value <- as.character(as.integer(n_threads))
  Sys.setenv(
    OMP_NUM_THREADS = value,
    OPENBLAS_NUM_THREADS = value,
    MKL_NUM_THREADS = value,
    VECLIB_MAXIMUM_THREADS = value,
    NUMEXPR_NUM_THREADS = value,
    RCPP_PARALLEL_NUM_THREADS = value
  )
  options(Ncpus = as.integer(n_threads))
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    try(RhpcBLASctl::blas_set_num_threads(as.integer(n_threads)), silent = TRUE)
    try(RhpcBLASctl::omp_set_num_threads(as.integer(n_threads)), silent = TRUE)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    try(data.table::setDTthreads(as.integer(n_threads)), silent = TRUE)
  }
  invisible(as.integer(n_threads))
}
configure_cpu_threads(n_threads)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...), "\n", sep = "")
  flush.console()
}

write_csv_one <- function(path, row) {
  utils::write.csv(row, path, row.names = FALSE)
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

available_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

metric_arg_for_label <- function(metric) {
  if (identical(metric, "cosine")) "cosine" else "euclidean"
}

method_metric_applicable <- function(method, metric) {
  if (identical(metric, "l2")) return(list(ok = TRUE, reason = ""))
  if (identical(metric, "inner_product")) {
    ip_methods <- c(
      "faissR_faiss_flat_ip",
      "faissR_faiss_gpu_flat_ip"
    )
    if (method %in% ip_methods) return(list(ok = TRUE, reason = ""))
    return(list(ok = FALSE, reason = "inner-product search is benchmarked only with FAISS IP methods"))
  }
  cosine_methods <- c(
    "faissR_cpu_exact",
    "faissR_rcpphnsw",
    "faissR_faiss_gpu_ivf_flat",
    "RcppHNSW_hnsw",
    "BiocNeighbors_hnsw",
    "BiocNeighbors_annoy",
    "uwot_similarity_graph_fnn",
    "uwot_similarity_graph_annoy",
    "uwot_similarity_graph_hnsw",
    "uwot_similarity_graph_nndescent"
  )
  if (method %in% cosine_methods) return(list(ok = TRUE, reason = ""))
  list(ok = FALSE, reason = paste0("method `", method, "` does not expose a validated cosine/IP mode in this benchmark"))
}

method_is_exact <- function(method, metric) {
  if (identical(metric, "inner_product")) {
    return(method %in% c("faissR_faiss_flat_ip", "faissR_faiss_gpu_flat_ip"))
  }
  method %in% c(
    "faissR_cpu_exact",
    "faissR_faiss_flat_l2",
    "faissR_faiss_gpu_flat_l2",
    "faissR_cuda_exact",
    "Rnanoflann_standard",
    "RANN_kd",
    "RANN_bd",
    "rnndescent_bruteforce",
    "BiocNeighbors_exhaustive"
  )
}

coerce_matrix <- function(x) {
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_dataset <- function(dataset, data_path) {
  if (identical(dataset, "SimulatedUniform2D")) {
    set.seed(1)
    data <- matrix(runif(1000000), ncol = 2)
    colnames(data) <- c("x", "y")
    return(list(data = data, labels = NULL, source = "simulated runif(1000000), ncol=2"))
  }
  env <- new.env(parent = emptyenv())
  load(data_path, envir = env)
  if (!exists("dataset", envir = env, inherits = FALSE)) {
    stop("No object named `dataset` in ", data_path)
  }
  ds <- get("dataset", envir = env, inherits = FALSE)
  list(data = coerce_matrix(ds$data), labels = ds$labels, source = data_path)
}

standardize_knn <- function(obj) {
  if (is.null(obj)) return(list(indices = NULL, distances = NULL))
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
  list(indices = NULL, distances = NULL)
}

choose_quality_rows <- function(n, p) {
  if (n < 2L) return(integer())
  by_ops <- floor(quality_eval_max_ops / max(1, as.double(n) * as.double(p)))
  size <- min(quality_eval_max_n, n, max(16L, as.integer(by_ops)))
  if (!is.finite(size) || size < 1L) return(integer())
  set.seed(20260615 + n + p)
  sort(sample.int(n, size))
}

exact_subset_knn <- function(x, rows, k, metric) {
  n <- nrow(x)
  k <- min(k, n - 1L)
  idx <- matrix(NA_integer_, length(rows), k)
  dst <- matrix(NA_real_, length(rows), k)
  if (identical(metric, "cosine")) {
    norms <- sqrt(rowSums(x * x))
    norms[!is.finite(norms) | norms <= 0] <- 1
    z <- x / norms
  } else {
    z <- x
  }
  for (ii in seq_along(rows)) {
    r <- rows[[ii]]
    if (identical(metric, "cosine")) {
      score <- drop(z %*% z[r, ])
      dist <- 1 - score
      dist[r] <- Inf
      ord <- order(dist, decreasing = FALSE)[seq_len(k)]
      idx[ii, ] <- ord
      dst[ii, ] <- dist[ord]
    } else if (identical(metric, "inner_product")) {
      score <- drop(z %*% z[r, ])
      score[r] <- -Inf
      ord <- order(score, decreasing = TRUE)[seq_len(k)]
      idx[ii, ] <- ord
      dst[ii, ] <- score[ord]
    } else {
      diff <- sweep(z, 2L, z[r, ], FUN = "-")
      dist2 <- rowSums(diff * diff)
      dist2[r] <- Inf
      ord <- order(dist2, decreasing = FALSE)[seq_len(k)]
      idx[ii, ] <- ord
      dst[ii, ] <- sqrt(pmax(0, dist2[ord]))
    }
  }
  list(indices = idx, distances = dst)
}

knn_rank_correlation <- function(candidate, reference, k) {
  vals <- numeric(nrow(reference$indices))
  vals[] <- NA_real_
  for (i in seq_len(nrow(reference$indices))) {
    a <- candidate$indices[i, seq_len(k)]
    b <- reference$indices[i, seq_len(k)]
    universe <- unique(c(a, b))
    ra <- match(universe, a)
    rb <- match(universe, b)
    ra[is.na(ra)] <- k + 1L
    rb[is.na(rb)] <- k + 1L
    if (length(unique(ra)) > 1L && length(unique(rb)) > 1L) {
      vals[[i]] <- suppressWarnings(stats::cor(ra, rb, method = "spearman"))
    }
  }
  mean(vals, na.rm = TRUE)
}

evaluate_knn_quality <- function(x, obj, k, metric, exact) {
  empty <- list(
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    mean_relative_distance_error = NA_real_,
    rank_correlation = NA_real_,
    quality_eval_n = 0L,
    quality_exact_sec = NA_real_,
    quality_status = "not_evaluated",
    quality_error = ""
  )
  sx <- standardize_knn(obj)
  if (is.null(sx$indices) || is.null(sx$distances)) {
    empty$quality_error <- "method did not return a KNN index/distance matrix"
    return(empty)
  }
  if (isTRUE(exact)) {
    empty$recall_at_k <- 1
    empty$median_recall_at_k <- 1
    empty$min_recall_at_k <- 1
    empty$mean_relative_distance_error <- 0
    empty$rank_correlation <- 1
    empty$quality_status <- "exact_backend"
    return(empty)
  }
  rows <- choose_quality_rows(nrow(x), ncol(x))
  if (!length(rows)) {
    empty$quality_error <- "quality subset is empty"
    return(empty)
  }
  t0 <- proc.time()[["elapsed"]]
  ref <- tryCatch(exact_subset_knn(x, rows, k, metric), error = function(e) e)
  empty$quality_exact_sec <- proc.time()[["elapsed"]] - t0
  empty$quality_eval_n <- length(rows)
  if (inherits(ref, "error")) {
    empty$quality_status <- "failed"
    empty$quality_error <- conditionMessage(ref)
    return(empty)
  }
  cand <- list(
    indices = as.matrix(sx$indices[rows, seq_len(k), drop = FALSE]),
    distances = as.matrix(sx$distances[rows, seq_len(k), drop = FALSE])
  )
  rec <- faissR::knn_recall(cand, ref, k = k)
  denom <- mean(abs(ref$distances), na.rm = TRUE) + sqrt(.Machine$double.eps)
  empty$recall_at_k <- rec$recall_at_k[[1L]]
  empty$median_recall_at_k <- rec$median_recall_at_k[[1L]]
  empty$min_recall_at_k <- rec$min_recall_at_k[[1L]]
  empty$mean_relative_distance_error <- mean(abs(cand$distances - ref$distances), na.rm = TRUE) / denom
  empty$rank_correlation <- knn_rank_correlation(cand, ref, k)
  empty$quality_status <- "success"
  empty
}

drop_self_if_first <- function(indices, distances, target_k) {
  if (is.null(indices) || is.null(distances)) return(list(indices = indices, distances = distances))
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

save_cuvs_knn <- function(obj, dataset, out_dir) {
  knn_dir <- file.path(out_dir, "knn_cuvs_nndescent")
  dir.create(knn_dir, recursive = TRUE, showWarnings = FALSE)
  nn_cuvs_nndescent <- obj
  save(
    nn_cuvs_nndescent,
    file = file.path(knn_dir, paste0(dataset, "_cuvs_nndescent_k", k, ".RData")),
    compress = "gzip"
  )
}

annoy_knn <- function(x, k, n_trees = 50L, n_threads = 1L) {
  if (!available_pkg("RcppAnnoy")) stop("RcppAnnoy unavailable")
  p <- ncol(x)
  index <- new(RcppAnnoy::AnnoyEuclidean, p)
  for (i in seq_len(nrow(x))) index$addItem(i - 1L, x[i, ])
  index$build(as.integer(n_trees))
  query_one <- function(i) {
    ans <- index$getNNsByVectorList(x[i, ], k + 1L, search_k = -1L, include_distances = TRUE)
    ii <- as.integer(ans$item + 1L)
    dd <- as.numeric(ans$distance)
    keep <- ii != i
    ii <- ii[keep]
    dd <- dd[keep]
    if (length(ii) < k) {
      ii <- c(ii, rep(NA_integer_, k - length(ii)))
      dd <- c(dd, rep(NA_real_, k - length(dd)))
    }
    list(indices = ii[seq_len(k)], distances = dd[seq_len(k)])
  }
  n_threads <- as.integer(max(1L, n_threads))
  rows <- seq_len(nrow(x))
  if (n_threads > 1L && .Platform$OS.type != "windows") {
    chunks <- split(rows, cut(rows, breaks = min(n_threads, length(rows)), labels = FALSE))
    partial <- parallel::mclapply(
      chunks,
      function(ii) lapply(ii, query_one),
      mc.cores = min(n_threads, length(chunks))
    )
    rows_out <- unlist(partial, recursive = FALSE, use.names = FALSE)
  } else {
    rows_out <- lapply(rows, query_one)
  }
  idx <- do.call(rbind, lapply(rows_out, `[[`, "indices"))
  dst <- do.call(rbind, lapply(rows_out, `[[`, "distances"))
  list(indices = idx, distances = dst)
}

run_method <- function(method, x, k, n_threads, dataset, out_dir, metric) {
  configure_cpu_threads(n_threads)
  if (startsWith(method, "faissR_")) {
    if (!available_pkg("faissR")) stop("faissR unavailable")
    backend <- sub("^faissR_", "", method)
    backend <- switch(
      backend,
      cpu_exact = "cpu",
      rcpphnsw = "hnsw",
      faiss_flat_l2 = "faiss_flat_l2",
      faiss_flat_ip = "faiss_flat_ip",
      faiss_gpu_flat_l2 = "faiss_gpu_flat_l2",
      faiss_gpu_flat_ip = "faiss_gpu_flat_ip",
      faiss_ivf = "faiss_ivf",
      faiss_ivfpq = "faiss_ivfpq",
      faiss_gpu_ivf_flat = "faiss_gpu_ivf_flat",
      faiss_gpu_ivfpq = "faiss_gpu_ivfpq",
      faiss_hnsw = "faiss_hnsw",
      faiss_nsg = "faiss_nsg",
      faiss_nndescent = "faiss_nndescent",
      cuda_exact = "cuda_cuvs_bruteforce",
      cuda_ivf = "cuda_ivf",
      cuda_cuvs_ivf_flat = "cuda_cuvs_ivf_flat",
      cuda_cuvs_ivfpq = "cuda_cuvs_ivfpq",
      cuda_cuvs_bruteforce = "cuda_cuvs_bruteforce",
      cuda_cuvs_cagra = "cuda_cuvs_cagra",
      cuda_cuvs_nndescent = "cuda_cuvs_nndescent",
      backend
    )
    obj <- faissR::nn(x, k = k, backend = backend, n_threads = n_threads, metric = metric_arg_for_label(metric))
    if (identical(method, "faissR_cuda_cuvs_nndescent") && identical(metric, "l2")) {
      save_cuvs_knn(obj, dataset, out_dir)
    }
    return(obj)
  }

  switch(
    method,
    Rnanoflann_standard = {
      if (!available_pkg("Rnanoflann")) stop("Rnanoflann unavailable")
      out <- Rnanoflann::nn(x, x, k + 1L, parallel = TRUE, cores = n_threads, sorted = TRUE)
      keep <- drop_self_if_first(out$indices, out$distances, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_kd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      # RANN does not expose a thread argument; benchmark records requested threads,
      # but this call remains single-threaded inside RANN.
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "kd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    RANN_bd = {
      if (!available_pkg("RANN")) stop("RANN unavailable")
      # RANN does not expose a thread argument; benchmark records requested threads,
      # but this call remains single-threaded inside RANN.
      out <- RANN::nn2(x, x, k = k + 1L, treetype = "bd")
      keep <- drop_self_if_first(out$nn.idx, out$nn.dists, k)
      list(indices = keep$indices, distances = keep$distances)
    },
    rnndescent_rpf = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rpf_knn(x, k = k, n_threads = n_threads, include_self = FALSE, progress = "none")
    },
    rnndescent_rnnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::rnnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_nnd = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::nnd_knn(x, k = k, n_threads = n_threads, progress = "none")
    },
    rnndescent_bruteforce = {
      if (!available_pkg("rnndescent")) stop("rnndescent unavailable")
      rnndescent::brute_force_knn(x, k = k, n_threads = n_threads)
    },
    RcppHNSW_hnsw = {
      if (!available_pkg("RcppHNSW")) stop("RcppHNSW unavailable")
      RcppHNSW::hnsw_knn(x, k = k, distance = if (identical(metric, "cosine")) "cosine" else "euclidean", M = 16, ef_construction = 200, ef = max(50, 3 * k), n_threads = n_threads, progress = "none")
    },
    RcppAnnoy_euclidean = annoy_knn(x, k, n_threads = n_threads),
    BiocNeighbors_vptree = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::VptreeParam(distance = "Euclidean"), num.threads = n_threads)
    },
    BiocNeighbors_hnsw = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::HnswParam(distance = if (identical(metric, "cosine")) "Cosine" else "Euclidean", nlinks = 16, ef.construction = 200, ef.search = max(50, 3 * k)), num.threads = n_threads)
    },
    BiocNeighbors_annoy = {
      if (!available_pkg("BiocNeighbors")) stop("BiocNeighbors unavailable")
      BiocNeighbors::findKNN(x, k = k, BNPARAM = BiocNeighbors::AnnoyParam(distance = if (identical(metric, "cosine")) "Cosine" else "Euclidean", ntrees = 50), num.threads = n_threads)
    },
    uwot_similarity_graph_fnn = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "fnn", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_annoy = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "annoy", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_hnsw = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "hnsw", n_threads = n_threads, verbose = FALSE)
    },
    uwot_similarity_graph_nndescent = {
      if (!available_pkg("uwot")) stop("uwot unavailable")
      uwot::similarity_graph(x, n_neighbors = k, metric = metric_arg_for_label(metric), nn_method = "nndescent", n_threads = n_threads, verbose = FALSE)
    },
    umap_umap_knn_from_cuvs = {
      if (!available_pkg("umap")) stop("umap unavailable")
      if (!available_pkg("faissR")) stop("faissR unavailable")
      knn <- faissR::nn(x, k = k, backend = "cuda_cuvs_nndescent", n_threads = n_threads)
      sx <- standardize_knn(knn)
      umap::umap.knn(sx$indices, sx$distances)
    },
    Rtsne_neighbors = {
      stop("Rtsne::Rtsne_neighbors consumes precomputed neighbours and optimizes t-SNE; it is not a standalone KNN search method.")
    },
    stop("Unknown method: ", method)
  )
}

method_table <- function() {
  data.frame(
    method = c(
      "faissR_cpu_exact",
      "faissR_rcpphnsw",
      "faissR_faiss_flat_l2",
      "faissR_faiss_flat_ip",
      "faissR_faiss_gpu_flat_l2",
      "faissR_faiss_gpu_flat_ip",
      "faissR_faiss_ivf",
      "faissR_faiss_ivfpq",
      "faissR_faiss_gpu_ivf_flat",
      "faissR_faiss_gpu_ivfpq",
      "faissR_faiss_hnsw",
      "faissR_faiss_nsg",
      "faissR_faiss_nndescent",
      "faissR_cuda_exact",
      "faissR_cuda_ivf",
      "faissR_cuda_cuvs_ivf_flat",
      "faissR_cuda_cuvs_ivfpq",
      "faissR_cuda_cuvs_bruteforce",
      "faissR_cuda_cuvs_cagra",
      "faissR_cuda_cuvs_nndescent",
      "Rnanoflann_standard",
      "RANN_kd",
      "RANN_bd",
      "rnndescent_rpf",
      "rnndescent_rnnd",
      "rnndescent_nnd",
      "rnndescent_bruteforce",
      "RcppHNSW_hnsw",
      "RcppAnnoy_euclidean",
      "BiocNeighbors_vptree",
      "BiocNeighbors_hnsw",
      "BiocNeighbors_annoy",
      "uwot_similarity_graph_fnn",
      "uwot_similarity_graph_annoy",
      "uwot_similarity_graph_hnsw",
      "uwot_similarity_graph_nndescent",
      "umap_umap_knn_from_cuvs",
      "Rtsne_neighbors"
    ),
    implementation = c(
      rep("faissR", 20),
      "Rnanoflann", "RANN", "RANN",
      "rnndescent", "rnndescent", "rnndescent", "rnndescent",
      "RcppHNSW", "RcppAnnoy",
      "BiocNeighbors", "BiocNeighbors", "BiocNeighbors",
      "uwot", "uwot", "uwot", "uwot",
      "umap", "Rtsne"
    ),
    backend = c(
      "CPU", "CPU", "CPU", "CPU", "CUDA", "CUDA", "CPU", "CPU", "CUDA", "CUDA", "CPU", "CPU", "CPU",
      "CUDA", "CUDA", "CUDA", "CUDA", "CUDA", "CUDA", "CUDA",
      rep("CPU", 18)
    ),
    kind = c(
      rep("knn_search", 36),
      "knn_consumer",
      "not_applicable"
    ),
    stringsAsFactors = FALSE
  )
}

if (worker) {
  dataset <- args$dataset
  data_path <- args$data_path
  method <- args$method
  result_path <- args$result_path
  dir.create(dirname(result_path), recursive = TRUE, showWarnings = FALSE)
  meta <- method_table()
  mm <- meta[match(method, meta$method), , drop = FALSE]
  started_total <- proc.time()[["elapsed"]]
  row <- data.frame(
    dataset = dataset,
    method = method,
    implementation = mm$implementation %||% NA_character_,
    backend = mm$backend %||% NA_character_,
    kind = mm$kind %||% NA_character_,
    n = NA_integer_,
    p = NA_integer_,
    k = k,
    metric = metric,
    n_threads = n_threads,
    status = "failed",
    time_sec = NA_real_,
    load_sec = NA_real_,
    peak_rss_gb = NA_real_,
    recall_at_k = NA_real_,
    median_recall_at_k = NA_real_,
    min_recall_at_k = NA_real_,
    mean_relative_distance_error = NA_real_,
    rank_correlation = NA_real_,
    quality_eval_n = NA_integer_,
    quality_exact_sec = NA_real_,
    quality_status = NA_character_,
    quality_error = "",
    output_rows = NA_integer_,
    output_cols = NA_integer_,
    error = "",
    stringsAsFactors = FALSE
  )
  tryCatch({
    if (identical(mm$kind, "not_applicable")) {
      row$status <- "not_applicable"
      row$error <- "Rtsne::Rtsne_neighbors is not a standalone KNN search method."
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    applicable <- method_metric_applicable(method, metric)
    if (!isTRUE(applicable$ok)) {
      row$status <- "skipped"
      row$error <- applicable$reason
      row$quality_status <- "skipped"
      row$quality_error <- applicable$reason
      write_csv_one(result_path, row)
      quit(status = 0L)
    }
    load_start <- proc.time()[["elapsed"]]
    ds <- load_dataset(dataset, data_path)
    x <- ds$data
    row$n <- nrow(x)
    row$p <- ncol(x)
    row$load_sec <- proc.time()[["elapsed"]] - load_start
    gc()
    start <- proc.time()[["elapsed"]]
    obj <- run_method(method, x, k, n_threads, dataset, out_dir, metric)
    row$time_sec <- proc.time()[["elapsed"]] - start
    sx <- standardize_knn(obj)
    if (!is.null(sx$indices)) {
      row$output_rows <- nrow(sx$indices)
      row$output_cols <- ncol(sx$indices)
    }
    quality <- evaluate_knn_quality(
      x,
      obj,
      k,
      metric,
      exact = method_is_exact(method, metric)
    )
    row$recall_at_k <- quality$recall_at_k
    row$median_recall_at_k <- quality$median_recall_at_k
    row$min_recall_at_k <- quality$min_recall_at_k
    row$mean_relative_distance_error <- quality$mean_relative_distance_error
    row$rank_correlation <- quality$rank_correlation
    row$quality_eval_n <- quality$quality_eval_n
    row$quality_exact_sec <- quality$quality_exact_sec
    row$quality_status <- quality$quality_status
    row$quality_error <- quality$quality_error
    row$status <- "success"
    row$peak_rss_gb <- read_peak_rss_gb()
  }, error = function(e) {
    row$status <- "failed"
    row$error <- conditionMessage(e)
    row$time_sec <- proc.time()[["elapsed"]] - started_total
    row$peak_rss_gb <- read_peak_rss_gb()
  })
  write_csv_one(result_path, row)
  quit(status = 0L)
}

manifest_path <- file.path(data_root, "dataset_manifest.csv")
if (!file.exists(manifest_path)) stop("Missing dataset manifest: ", manifest_path)
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
manifest$path <- file.path(data_root, manifest$dataset, paste0(manifest$dataset, ".RData"))

datasets <- manifest[, c("dataset", "path", "n", "p")]
datasets <- rbind(
  datasets,
  data.frame(dataset = "SimulatedUniform2D", path = "SIMULATED", n = 500000L, p = 2L)
)
if (!is.null(args$datasets)) {
  wanted <- strsplit(args$datasets, ",", fixed = TRUE)[[1L]]
  datasets <- datasets[datasets$dataset %in% wanted, , drop = FALSE]
}

methods <- method_table()
if (!is.null(args$methods)) {
  wanted_methods <- strsplit(args$methods, ",", fixed = TRUE)[[1L]]
  methods <- methods[methods$method %in% wanted_methods, , drop = FALSE]
}

dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)

utils::write.csv(datasets, file.path(out_dir, "benchmark1_datasets.csv"), row.names = FALSE)
utils::write.csv(methods, file.path(out_dir, "benchmark1_methods.csv"), row.names = FALSE)
k_values <- as.integer(strsplit(args$k_values %||% Sys.getenv("FAISSR_BENCHMARK1_K_VALUES", "15,50,100"), ",", fixed = TRUE)[[1L]])
k_values <- unique(k_values[is.finite(k_values) & !is.na(k_values) & k_values > 0L])
if (!length(k_values)) k_values <- c(15L, 50L, 100L)
metric_values <- strsplit(args$metrics %||% Sys.getenv("FAISSR_BENCHMARK1_METRICS", "l2,cosine,inner_product"), ",", fixed = TRUE)[[1L]]
metric_values <- unique(tolower(trimws(metric_values)))
metric_values[metric_values %in% c("euclidean")] <- "l2"
metric_values[metric_values %in% c("ip", "innerproduct")] <- "inner_product"
metric_values <- unique(metric_values[metric_values %in% c("l2", "cosine", "inner_product")])
if (!length(metric_values)) metric_values <- c("l2", "cosine", "inner_product")
utils::write.csv(
  expand.grid(k = k_values, metric = metric_values, stringsAsFactors = FALSE),
  file.path(out_dir, "benchmark1_parameter_grid.csv"),
  row.names = FALSE
)

cmdline <- commandArgs(FALSE)
file_arg <- grep("^--file=", cmdline, value = TRUE)
script <- if (length(file_arg)) sub("^--file=", "", file_arg[[1L]]) else "tools/benchmark1_nn_speed.R"
if (!file.exists(script)) {
  script <- normalizePath("tools/benchmark1_nn_speed.R", mustWork = TRUE)
} else {
  script <- normalizePath(script, mustWork = TRUE)
}

results <- list()
job_id <- 0L
for (di in seq_len(nrow(datasets))) {
  for (mi in seq_len(nrow(methods))) {
    for (kk in k_values) {
      for (metric_i in metric_values) {
        job_id <- job_id + 1L
        dataset <- datasets$dataset[[di]]
        method <- methods$method[[mi]]
        result_path <- file.path(out_dir, "worker_results", sprintf("%03d_%s__%s__k%d__%s.csv", job_id, dataset, method, kk, metric_i))
        if (file.exists(result_path)) {
          log_msg("Skipping existing %s / %s / k=%d / metric=%s", dataset, method, kk, metric_i)
          next
        }
        log_msg("[%03d/%03d] %s / %s / k=%d / metric=%s", job_id, nrow(datasets) * nrow(methods) * length(k_values) * length(metric_values), dataset, method, kk, metric_i)
        cmd_args <- c(
          as.character(timeout_sec),
          "Rscript",
          script,
          "--worker=TRUE",
          paste0("--dataset=", dataset),
          paste0("--data_path=", datasets$path[[di]]),
          paste0("--method=", method),
          paste0("--result_path=", result_path),
          paste0("--out_dir=", out_dir),
          paste0("--k=", kk),
          paste0("--metric=", metric_i),
          paste0("--threads=", n_threads)
        )
        status <- system2("timeout", cmd_args, stdout = file.path(out_dir, "benchmark1_worker_stdout.log"), stderr = file.path(out_dir, "benchmark1_worker_stderr.log"))
        if (!file.exists(result_path)) {
          timeout_row <- data.frame(
            dataset = dataset,
            method = method,
            implementation = methods$implementation[[mi]],
            backend = methods$backend[[mi]],
            kind = methods$kind[[mi]],
            n = datasets$n[[di]],
            p = datasets$p[[di]],
            k = kk,
            metric = metric_i,
            n_threads = n_threads,
            status = if (identical(status, 124L)) "timeout" else "failed",
            time_sec = if (identical(status, 124L)) timeout_sec else NA_real_,
            load_sec = NA_real_,
            peak_rss_gb = NA_real_,
            recall_at_k = NA_real_,
            median_recall_at_k = NA_real_,
            min_recall_at_k = NA_real_,
            mean_relative_distance_error = NA_real_,
            rank_correlation = NA_real_,
            quality_eval_n = NA_integer_,
            quality_exact_sec = NA_real_,
            quality_status = "timeout",
            quality_error = paste("worker did not produce result; exit status", status),
            output_rows = NA_integer_,
            output_cols = NA_integer_,
            error = paste("worker did not produce result; exit status", status),
            stringsAsFactors = FALSE
          )
          write_csv_one(result_path, timeout_row)
        }
      }
    }
  }
}

files <- list.files(file.path(out_dir, "worker_results"), pattern = "[.]csv$", full.names = TRUE)
results <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
results <- results[order(results$dataset, results$backend, results$implementation, results$method), ]
utils::write.csv(results, file.path(out_dir, "benchmark1_nn_speed_results.csv"), row.names = FALSE)

success <- results[results$status == "success" & results$kind == "knn_search", , drop = FALSE]
best <- success[order(success$dataset, success$time_sec), ]
best <- best[!duplicated(best$dataset), ]
utils::write.csv(best, file.path(out_dir, "benchmark1_best_by_dataset.csv"), row.names = FALSE)
if (nrow(success)) {
  ranked_quality <- success[order(
    success$dataset,
    success$k,
    success$metric,
    -success$recall_at_k,
    success$time_sec,
    success$peak_rss_gb
  ), , drop = FALSE]
  utils::write.csv(ranked_quality, file.path(out_dir, "benchmark1_ranked_speed_quality_memory.csv"), row.names = FALSE)
}

png(file.path(out_dir, "benchmark1_nn_speed_barplot.png"), width = 2200, height = 1400, res = 160)
op <- par(mar = c(12, 5, 4, 2), mfrow = c(ceiling(length(unique(results$dataset)) / 2), 2))
for (dataset in unique(results$dataset)) {
  sub <- results[results$dataset == dataset & results$kind != "not_applicable", , drop = FALSE]
  sub$plot_time <- ifelse(sub$status == "success", sub$time_sec, timeout_sec)
  sub <- sub[order(sub$plot_time), ]
  cols <- ifelse(sub$status == "success", ifelse(sub$backend == "CUDA", "#2b8cbe", "#7bccc4"), "#d95f0e")
  barplot(
    sub$plot_time,
    names.arg = sub$method,
    las = 2,
    col = cols,
    main = dataset,
    ylab = "seconds (timeouts shown at cap)",
    cex.names = 0.45
  )
  legend("topright", fill = c("#7bccc4", "#2b8cbe", "#d95f0e"), legend = c("CPU success", "CUDA success", "failed/timeout"), cex = 0.7, bty = "n")
}
par(op)
dev.off()

materials <- c(
  "# BENCHMARK #1 Materials and Methods",
  "",
  "Benchmark #1 measures nearest-neighbour construction speed across faissR native/FAISS backends, optional CUDA/cuVS backends, and external R package implementations.",
  "",
  paste0("Datasets were read from `", data_root, "`. The manifest datasets were MNIST, FashionMNIST, USPS, COIL20, MetRef, and TabulaMuris. A simulated reference dataset was generated as `matrix(runif(1000000), ncol = 2)` with columns `x` and `y`, giving 500,000 observations and 2 variables."),
  paste0("All methods were tested over k = ", paste(k_values, collapse = ", "), " and metrics = ", paste(metric_values, collapse = ", "), ". CPU methods were run with n_threads/cores = ", n_threads, " when the package exposed a thread argument. Each dataset-method-parameter combination was executed in a separate R process with GNU `timeout` set to ", timeout_sec, " seconds."),
  paste0("Nearest-neighbour quality was evaluated against an exact subset reference where feasible. The reference subset used at most ", quality_eval_max_n, " rows and was automatically reduced when the estimated operation count exceeded ", format(quality_eval_max_ops, scientific = TRUE), ". Reported quality metrics are recall@k, median recall@k, minimum recall@k, mean relative distance error, and Spearman rank correlation of neighbour ranks."),
  "The faissR CUDA/cuVS NN-descent output was saved for every dataset where the method completed successfully.",
  "",
  "faissR methods tested: exact CPU, RcppHNSW wrapper, FAISS Flat L2, FAISS Flat IP, FAISS IVF, FAISS IVFPQ, FAISS HNSW, FAISS NSG, FAISS NNDescent, native CUDA exact, native CUDA IVF, direct cuVS IVF-Flat, explicit direct cuVS IVF-PQ, cuVS brute force, cuVS CAGRA, and cuVS NN-descent.",
  "External R package methods tested: Rnanoflann, RANN kd-tree and bd-tree, rnndescent RPF/RNND/NND/brute-force, RcppHNSW, RcppAnnoy, BiocNeighbors VP-tree/HNSW/Annoy, and uwot::similarity_graph with nn_method = fnn, annoy, hnsw, and nndescent.",
  "umap::umap.knn was included as a precomputed-neighbour consumer test, not as a standalone KNN search algorithm. Rtsne::Rtsne_neighbors was marked not applicable because it consumes precomputed neighbours and optimizes t-SNE rather than exporting a standalone KNN search.",
  "",
  "The benchmark records elapsed method time, load/conversion time, peak resident memory when available from `/proc/self/status`, output dimensions where an index matrix is returned, quality metrics, status, and error messages."
)
writeLines(materials, file.path(out_dir, "BENCHMARK1_MATERIALS_AND_METHODS.md"))

summary_lines <- c(
  "# BENCHMARK #1 Results Summary",
  "",
  paste0("Run directory: `", out_dir, "`"),
  "",
  "## Best Successful KNN Search Per Dataset",
  "",
  paste(capture.output(print(best[, c("dataset", "method", "implementation", "backend", "time_sec", "status")], row.names = FALSE)), collapse = "\n"),
  "",
  "## Comments",
  "",
  "This benchmark separates pure KNN search methods from graph/consumer functions. The fastest method can differ by dataset shape: low-dimensional simulated data favours tree/grid-like methods, while high-dimensional image matrices favour approximate graph or GPU methods. Exact brute-force methods are included as references but are expected to time out or be uncompetitive on the largest datasets. cuVS NN-descent outputs are saved to allow later embedding benchmarks to reuse the same neighbour graph rather than recomputing KNN."
)
writeLines(summary_lines, file.path(out_dir, "BENCHMARK1_RESULTS_SUMMARY.md"))

log_msg("DONE: %s", out_dir)
