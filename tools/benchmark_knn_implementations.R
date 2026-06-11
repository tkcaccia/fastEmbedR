#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (requireNamespace("pkgload", quietly = TRUE)) {
    pkgload::load_all(".", quiet = TRUE)
  } else {
    library(fastEmbedR)
  }
})

parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) next
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", parts[1L])
    value <- paste(parts[-1L], collapse = "=")
    out[[key]] <- value
  }
  out
}

split_arg <- function(x) {
  x <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  x[nzchar(x)]
}

as_logical_arg <- function(x) {
  tolower(trimws(as.character(x))) %in% c("1", "true", "yes", "y")
}

rss_mb <- function() {
  out <- suppressWarnings(system2(
    "ps",
    c("-o", "rss=", "-p", as.character(Sys.getpid())),
    stdout = TRUE,
    stderr = FALSE
  ))
  value <- suppressWarnings(as.numeric(trimws(out[1L])))
  if (length(value) != 1L || is.na(value)) return(NA_real_)
  value / 1024
}

make_dataset <- function(name, seed) {
  set.seed(seed)
  if (identical(name, "iris")) {
    x <- scale(as.matrix(datasets::iris[, 1:4]))
    labels <- datasets::iris$Species
    return(list(name = "iris", x = x, labels = labels))
  }

  m <- regexec("^synthetic_([0-9]+)(?:x([0-9]+))?$", name)
  hit <- regmatches(name, m)[[1L]]
  if (length(hit) >= 2L) {
    n <- as.integer(hit[2L])
    p <- if (length(hit) >= 3L && nzchar(hit[3L])) as.integer(hit[3L]) else 20L
    groups <- max(3L, min(10L, floor(sqrt(n / 20))))
    labels <- rep(seq_len(groups), length.out = n)
    centers <- matrix(rnorm(groups * p, sd = 3), groups, p)
    x <- matrix(rnorm(n * p, sd = 0.8), n, p)
    x <- x + centers[labels, , drop = FALSE]
    x <- scale(x)
    return(list(name = name, x = x, labels = factor(labels)))
  }

  stop("Unknown dataset `", name, "`. Use iris or synthetic_<n>[x<p>].", call. = FALSE)
}

strip_self_and_limit <- function(indices, distances, k, n) {
  strip_self_and_limit_by_query(indices, distances, k, seq_len(nrow(indices)))
}

strip_self_and_limit_by_query <- function(indices, distances, k, self_indices = NULL) {
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  storage.mode(indices) <- "integer"
  storage.mode(distances) <- "double"
  out_i <- matrix(NA_integer_, nrow(indices), k)
  out_d <- matrix(NA_real_, nrow(indices), k)
  if (is.null(self_indices)) self_indices <- rep(NA_integer_, nrow(indices))
  self_indices <- as.integer(self_indices)
  for (i in seq_len(nrow(indices))) {
    idx <- indices[i, ]
    dst <- distances[i, ]
    self <- self_indices[i]
    keep <- if (is.na(self)) seq_along(idx) else which(idx != self)
    if (length(keep) < k) keep <- seq_along(idx)
    keep <- keep[seq_len(min(k, length(keep)))]
    out_i[i, seq_along(keep)] <- idx[keep]
    out_d[i, seq_along(keep)] <- dst[keep]
  }
  list(indices = out_i, distances = out_d)
}

exact_reference_subset <- function(x, sample_rows, k) {
  raw <- fastEmbedR::nn(
    x,
    x[sample_rows, , drop = FALSE],
    k = k + 1L,
    backend = "cpu"
  )
  strip_self_and_limit_by_query(raw$indices, raw$distances, k, sample_rows)
}

finite_number_or <- function(x, default = NA_real_) {
  if (is.null(x) || length(x) == 0L) return(default)
  value <- suppressWarnings(as.numeric(x[1L]))
  if (is.finite(value)) value else default
}

safe_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

safe_median <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else median(x)
}

knn_recall_at_k <- function(candidate, reference, k, sample_rows = NULL) {
  if (is.null(candidate) || is.null(candidate$indices)) return(NA_real_)
  idx <- if (is.null(sample_rows)) candidate$indices else candidate$indices[sample_rows, seq_len(k), drop = FALSE]
  ref <- reference$indices
  if (!identical(dim(idx), dim(ref))) return(NA_real_)
  mean(vapply(seq_len(nrow(ref)), function(i) {
    length(intersect(idx[i, seq_len(k)], ref[i, seq_len(k)])) / k
  }, numeric(1L)))
}

mean_distance_error <- function(candidate, reference, sample_rows = NULL, k = ncol(reference$indices)) {
  if (is.null(candidate) || is.null(candidate$distances)) return(NA_real_)
  distances <- if (is.null(sample_rows)) candidate$distances else candidate$distances[sample_rows, seq_len(k), drop = FALSE]
  if (!identical(dim(distances), dim(reference$distances))) return(NA_real_)
  safe_mean(abs(distances - reference$distances))
}

knn_rank_correlation <- function(candidate, reference, k, sample_rows = NULL) {
  if (is.null(candidate) || is.null(candidate$indices)) return(NA_real_)
  idx <- if (is.null(sample_rows)) candidate$indices else candidate$indices[sample_rows, seq_len(k), drop = FALSE]
  ref <- reference$indices
  if (!identical(dim(idx), dim(ref))) return(NA_real_)
  values <- vapply(seq_len(nrow(ref)), function(i) {
    a <- idx[i, seq_len(k)]
    b <- ref[i, seq_len(k)]
    universe <- unique(c(a, b))
    rank_a <- match(universe, a)
    rank_b <- match(universe, b)
    rank_a[is.na(rank_a)] <- k + 1L
    rank_b[is.na(rank_b)] <- k + 1L
    if (length(unique(rank_a)) < 2L || length(unique(rank_b)) < 2L) return(NA_real_)
    suppressWarnings(stats::cor(rank_a, rank_b, method = "spearman"))
  }, numeric(1L))
  safe_mean(values)
}

profiled_result <- function(value, index_build_time_sec = NA_real_, query_time_sec = NA_real_) {
  attr(value, "index_build_time_sec") <- index_build_time_sec
  attr(value, "query_time_sec") <- query_time_sec
  value
}

timed_step <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, time = proc.time()[["elapsed"]] - start)
}

safe_time <- function(expr) {
  gc()
  before <- rss_mb()
  start <- proc.time()[["elapsed"]]
  result <- tryCatch(
    list(status = "success", value = force(expr), error = NA_character_),
    error = function(e) list(status = "failed", value = NULL, error = conditionMessage(e))
  )
  elapsed <- proc.time()[["elapsed"]] - start
  after <- rss_mb()
  result$elapsed <- elapsed
  result$rss_before_mb <- before
  result$rss_after_mb <- after
  result$delta_rss_mb <- if (is.na(before) || is.na(after)) NA_real_ else after - before
  result
}

with_fortran_nn <- function(enabled, code) {
  old <- Sys.getenv("FASTEMBEDR_USE_FORTRAN_NN", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("FASTEMBEDR_USE_FORTRAN_NN") else Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = old)
  }, add = TRUE)
  Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = if (isTRUE(enabled)) "1" else "0")
  force(code)
}

candidate_projection <- function(x, k, seed) {
  n <- nrow(x)
  n_centers <- fastEmbedR:::clustered_knn_center_count(n, k)
  centers <- fastEmbedR:::select_landmark_rows(x, n_centers, seed)
  projection_k <- min(n_centers, max(2L, min(12L, ceiling(k / 2))))
  projection <- fastEmbedR:::nn_compute(
    x[centers, , drop = FALSE],
    x,
    k = projection_k,
    backend = "cpu",
    points_missing = FALSE,
    exclude_self = FALSE
  )
  cols <- fastEmbedR:::clustered_knn_graph_columns(ncol(projection$indices), k)
  list(
    projection = projection,
    bucket_cols = cols$bucket_cols,
    query_cols = cols$query_cols
  )
}

normalize_fastembedr <- function(out, k, n) {
  strip_self_and_limit(out$indices, out$distances, k, n)
}

normalize_Rnanoflann <- function(out, k, n) {
  strip_self_and_limit(out$indices, out$distances, k, n)
}

normalize_FNN <- function(out, k, n) {
  strip_self_and_limit(out$nn.index, out$nn.dist, k, n)
}

normalize_RANN <- function(out, k, n) {
  strip_self_and_limit(out$nn.idx, out$nn.dists, k, n)
}

normalize_dbscan <- function(out, k, n) {
  strip_self_and_limit(out$id, out$dist, k, n)
}

normalize_RcppHNSW <- function(out, k, n) {
  strip_self_and_limit(out$idx, out$dist, k, n)
}

run_rcpphnsw <- function(x, k, distance = "euclidean", m = 16L, ef_construction = 200L, ef_search = 50L, n_threads = 0L, seed = 100L) {
  build <- timed_step(RcppHNSW::hnsw_build(
    x,
    distance = distance,
    M = as.integer(m),
    ef = as.integer(ef_construction),
    verbose = FALSE,
    progress = "none",
    n_threads = as.integer(n_threads),
    random_seed = as.integer(seed)
  ))
  query <- timed_step(RcppHNSW::hnsw_search(
    x,
    build$value,
    k = as.integer(k + 1L),
    ef = as.integer(ef_search),
    verbose = FALSE,
    progress = "none",
    n_threads = as.integer(n_threads)
  ))
  query$value$dist[query$value$dist < 0] <- 0
  profiled_result(query$value, build$time, query$time)
}

faiss_cpu_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) && tryCatch({
    reticulate::import("faiss", delay_load = FALSE)
    TRUE
  }, error = function(e) FALSE)
}

faiss_gpu_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) && tryCatch({
    faiss <- reticulate::import("faiss", delay_load = FALSE)
    reticulate::py_has_attr(faiss, "StandardGpuResources") &&
      reticulate::py_has_attr(faiss, "index_cpu_to_gpu") &&
      reticulate::py_has_attr(faiss, "get_num_gpus") &&
      as.integer(faiss$get_num_gpus()) > 0L
  }, error = function(e) FALSE)
}

cuml_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) && tryCatch({
    neighbors <- reticulate::import("cuml.neighbors", delay_load = FALSE)
    cupy <- reticulate::import("cupy", delay_load = FALSE)
    reticulate::py_has_attr(neighbors, "NearestNeighbors") &&
      as.integer(cupy$cuda$runtime$getDeviceCount()) > 0L
  }, error = function(e) FALSE)
}

faiss_effective_nlist <- function(n, nlist) {
  as.integer(max(1L, min(as.integer(nlist), max(1L, floor(as.integer(n) / 4L)))))
}

faiss_effective_nprobe <- function(nprobe, effective_nlist) {
  as.integer(max(1L, min(as.integer(nprobe), as.integer(effective_nlist))))
}

faiss_effective_nbits <- function(n, requested_nbits) {
  as.integer(max(1L, min(as.integer(requested_nbits), floor(log2(max(2L, as.integer(n)))))))
}

faiss_to_numpy <- function(x) {
  np <- reticulate::import("numpy", delay_load = FALSE)
  np$array(x, dtype = "float32", order = "C")
}

faiss_gpu_resources <- function(faiss) {
  faiss$StandardGpuResources()
}

faiss_to_gpu <- function(faiss, resources, device, index) {
  faiss$index_cpu_to_gpu(resources, as.integer(device), index)
}

faiss_search_has_missing <- function(search_result) {
  indices <- as.matrix(search_result[[2L]])
  any(!is.finite(indices)) || any(indices < 0L, na.rm = TRUE)
}

faiss_search <- function(index, xb, query_k, retry_nprobe = NA_integer_) {
  result <- index$search(xb, query_k)
  if (!is.na(retry_nprobe) && faiss_search_has_missing(result)) {
    index$nprobe <- as.integer(retry_nprobe)
    result <- index$search(xb, query_k)
  }
  result
}

normalize_FAISS <- function(out, k, n) {
  indices <- as.matrix(out[[2L]]) + 1L
  distances <- sqrt(pmax(as.matrix(out[[1L]]), 0))
  if (any(!is.finite(indices)) || any(indices < 1L, na.rm = TRUE)) {
    stop("FAISS returned invalid or missing neighbor indices.", call. = FALSE)
  }
  out_i <- matrix(NA_integer_, nrow(indices), k)
  out_d <- matrix(NA_real_, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    keep <- which(indices[i, ] != i)
    if (length(keep) < k) {
      stop("FAISS returned fewer non-self neighbors than requested.", call. = FALSE)
    }
    keep <- keep[seq_len(k)]
    out_i[i, ] <- indices[i, keep]
    out_d[i, ] <- distances[i, keep]
  }
  list(indices = out_i, distances = out_d)
}

cuml_to_matrix <- function(x) {
  if (reticulate::py_has_attr(x, "get")) {
    return(as.matrix(x$get()))
  }
  cupy <- reticulate::import("cupy", delay_load = FALSE)
  as.matrix(cupy$asnumpy(x))
}

normalize_cuml <- function(out, k, n) {
  indices <- cuml_to_matrix(out[[2L]]) + 1L
  distances <- pmax(cuml_to_matrix(out[[1L]]), 0)
  if (any(!is.finite(indices)) || any(indices < 1L, na.rm = TRUE)) {
    stop("cuML returned invalid or missing neighbor indices.", call. = FALSE)
  }
  out_i <- matrix(NA_integer_, nrow(indices), k)
  out_d <- matrix(NA_real_, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    keep <- which(indices[i, ] != i)
    if (length(keep) < k) {
      stop("cuML returned fewer non-self neighbors than requested.", call. = FALSE)
    }
    keep <- keep[seq_len(k)]
    out_i[i, ] <- indices[i, keep]
    out_d[i, ] <- distances[i, keep]
  }
  list(indices = out_i, distances = out_d)
}

run_faiss <- function(x,
                      k,
                      index_type,
                      nlist = NA_integer_,
                      nprobe = NA_integer_,
                      pq_m = NA_integer_,
                      pq_nbits = 8L,
                      hnsw_m = NA_integer_,
                      hnsw_ef_construction = NA_integer_,
                      hnsw_ef_search = NA_integer_,
                      gpu = FALSE,
                      device = 0L,
                      n_threads = 1L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  faiss <- reticulate::import("faiss", delay_load = FALSE)
  try(faiss$omp_set_num_threads(as.integer(n_threads)), silent = TRUE)
  if (isTRUE(gpu) && !isTRUE(faiss_gpu_available())) {
    stop("Python FAISS with CUDA GPU support is unavailable through reticulate.", call. = FALSE)
  }
  xb <- faiss_to_numpy(x)
  n <- nrow(x)
  d <- ncol(x)
  query_k <- as.integer(k + 1L)
  resources <- if (isTRUE(gpu)) faiss_gpu_resources(faiss) else NULL

  if (identical(index_type, "flat")) {
    build <- timed_step({
      index <- faiss$IndexFlatL2(as.integer(d))
      if (isTRUE(gpu)) index <- faiss_to_gpu(faiss, resources, device, index)
      index$add(xb)
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k))
    return(profiled_result(query$value, build$time, query$time))
  }

  if (identical(index_type, "ivf")) {
    build <- timed_step({
      effective_nlist <- faiss_effective_nlist(n, nlist)
      effective_nprobe <- faiss_effective_nprobe(nprobe, effective_nlist)
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      index <- faiss$IndexIVFFlat(quantizer, as.integer(d), effective_nlist, faiss$METRIC_L2)
      if (isTRUE(gpu)) index <- faiss_to_gpu(faiss, resources, device, index)
      index$train(xb)
      index$add(xb)
      index$nprobe <- effective_nprobe
      list(index = index, effective_nlist = effective_nlist)
    })
    query <- timed_step(faiss_search(build$value$index, xb, query_k, retry_nprobe = build$value$effective_nlist))
    return(profiled_result(query$value, build$time, query$time))
  }

  if (identical(index_type, "ivfpq")) {
    build <- timed_step({
      pq_m <- as.integer(pq_m)
      if (d %% pq_m != 0L) {
        stop("FAISS IVF-PQ requires dimension divisible by pq_m.", call. = FALSE)
      }
      effective_nlist <- faiss_effective_nlist(n, nlist)
      effective_nprobe <- faiss_effective_nprobe(nprobe, effective_nlist)
      effective_nbits <- faiss_effective_nbits(n, pq_nbits)
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      index <- faiss$IndexIVFPQ(
        quantizer,
        as.integer(d),
        effective_nlist,
        pq_m,
        effective_nbits,
        faiss$METRIC_L2
      )
      if (isTRUE(gpu)) index <- faiss_to_gpu(faiss, resources, device, index)
      index$train(xb)
      index$add(xb)
      index$nprobe <- effective_nprobe
      list(index = index, effective_nlist = effective_nlist)
    })
    query <- timed_step(faiss_search(build$value$index, xb, query_k, retry_nprobe = build$value$effective_nlist))
    return(profiled_result(query$value, build$time, query$time))
  }

  if (identical(index_type, "hnsw")) {
    if (isTRUE(gpu)) {
      stop("FAISS HNSW is not exposed as a CUDA GPU index in this benchmark.", call. = FALSE)
    }
    build <- timed_step({
      index <- faiss$IndexHNSWFlat(as.integer(d), as.integer(hnsw_m), faiss$METRIC_L2)
      index$hnsw$efConstruction <- as.integer(hnsw_ef_construction)
      index$hnsw$efSearch <- as.integer(hnsw_ef_search)
      index$add(xb)
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k))
    return(profiled_result(query$value, build$time, query$time))
  }

  stop("Unknown FAISS index type: ", index_type, call. = FALSE)
}

run_cuml_nn <- function(x,
                        k,
                        algorithm = "auto",
                        nlist = NA_integer_,
                        nprobe = NA_integer_,
                        device = 0L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  if (!isTRUE(cuml_available())) {
    stop("Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.", call. = FALSE)
  }
  cupy <- reticulate::import("cupy", delay_load = FALSE)
  neighbors <- reticulate::import("cuml.neighbors", delay_load = FALSE)
  try(cupy$cuda$Device(as.integer(device))$use(), silent = TRUE)
  xg <- cupy$array(x, dtype = "float32")
  base <- list(
    n_neighbors = as.integer(k + 1L),
    algorithm = algorithm,
    metric = "euclidean"
  )
  variants <- list(base)
  if (algorithm %in% c("ivfflat", "ivfpq")) {
    effective_nlist <- faiss_effective_nlist(nrow(x), nlist)
    effective_nprobe <- faiss_effective_nprobe(nprobe, effective_nlist)
    variants <- list(
      c(base, list(nlist = effective_nlist, nprobe = effective_nprobe)),
      c(base, list(n_clusters = effective_nlist, n_probes = effective_nprobe)),
      c(base, list(n_lists = effective_nlist, n_probes = effective_nprobe)),
      base
    )
  }
  last_error <- NULL
  model <- NULL
  for (args in variants) {
    model <- tryCatch(do.call(neighbors$NearestNeighbors, args), error = function(e) {
      last_error <<- conditionMessage(e)
      NULL
    })
    if (!is.null(model)) break
  }
  if (is.null(model)) {
    stop("Could not construct cuML NearestNeighbors model: ", last_error, call. = FALSE)
  }
  build <- timed_step(model$fit(xg))
  query <- timed_step(model$kneighbors(
    xg,
    n_neighbors = as.integer(k + 1L),
    return_distance = TRUE
  ))
  profiled_result(query$value, build$time, query$time)
}

normalize_BiocNeighbors <- function(out, k, n) {
  strip_self_and_limit(out$index, out$distance, k, n)
}

run_biocneighbors <- function(x, k, param, n_threads) {
  build <- timed_step(BiocNeighbors::buildIndex(x, BNPARAM = param))
  query <- timed_step(BiocNeighbors::queryKNN(
    build$value,
    x,
    k = as.integer(k + 1L),
    num.threads = as.integer(n_threads)
  ))
  profiled_result(query$value, build$time, query$time)
}

rcppannoy_search_k_value <- function(k, n_trees, search_multiplier) {
  if (is.null(search_multiplier) || is.na(search_multiplier)) return(-1L)
  as.integer(max(1L, ceiling(as.numeric(search_multiplier) * as.integer(n_trees) * (as.integer(k) + 1L))))
}

run_rcppannoy <- function(x, k, seed, n_trees, search_multiplier) {
  n <- nrow(x)
  query_k <- as.integer(k + 1L)
  build <- timed_step({
    index <- RcppAnnoy::AnnoyEuclidean$new(ncol(x))
    index$setSeed(as.integer(seed))
    for (i in seq_len(n)) {
      index$addItem(as.integer(i - 1L), x[i, ])
    }
    index$build(as.integer(n_trees))
    index
  })
  search_k <- rcppannoy_search_k_value(k, n_trees, search_multiplier)
  query <- timed_step({
    indices <- matrix(NA_integer_, n, query_k)
    distances <- matrix(NA_real_, n, query_k)
    for (i in seq_len(n)) {
      row <- build$value$getNNsByItemList(as.integer(i - 1L), query_k, search_k, TRUE)
      got <- min(query_k, length(row$item))
      if (got > 0L) {
        indices[i, seq_len(got)] <- as.integer(row$item[seq_len(got)] + 1L)
        distances[i, seq_len(got)] <- as.numeric(row$distance[seq_len(got)])
      }
    }
    list(indices = indices, distances = distances)
  })
  if (anyNA(query$value$indices) || anyNA(query$value$distances)) {
    stop("RcppAnnoy returned fewer neighbors than requested.", call. = FALSE)
  }
  profiled_result(query$value, build$time, query$time)
}

rcppannoy_impl_grid <- function() {
  out <- list()
  for (n_trees in c(10L, 25L, 50L, 100L)) {
    for (search_multiplier in c(NA_real_, 2, 5)) {
      nt <- as.integer(n_trees)
      sm <- search_multiplier
      search_label <- if (is.na(search_multiplier)) "default" else paste0(search_multiplier, "x")
      id_label <- gsub("[^A-Za-z0-9]+", "", search_label)
      out[[length(out) + 1L]] <- list(
        name = paste0("RcppAnnoy_t", nt, "_sk", id_label),
        family = "external_approx",
        backend = "cpu",
        exact = FALSE,
        kind = "direct",
        package = "RcppAnnoy",
        available = requireNamespace("RcppAnnoy", quietly = TRUE),
        run = function(x, k, seed, projection, n_trees = nt, search_multiplier = sm) {
          run_rcppannoy(x, k, seed, n_trees, search_multiplier)
        },
        normalize = normalize_Rnanoflann
      )
    }
  }
  out
}

make_impls <- function(include_external, n_threads) {
  impls <- list(
    list(
      name = "fastEmbedR_cpu_cpp_without_self",
      family = "fastEmbedR_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) with_fortran_nn(FALSE, fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_cpu_fortran_without_self",
      family = "fastEmbedR_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) with_fortran_nn(TRUE, fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_cpu_public_nn_self",
      family = "fastEmbedR_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) with_fortran_nn(FALSE, fastEmbedR::nn(x, k = k + 1L, backend = "cpu")),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_auto_without_self",
      family = "fastEmbedR_auto",
      backend = "auto",
      exact = NA,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) fastEmbedR:::nn_without_self(x, k = k, backend = "auto"),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_cpu_clustered_without_self",
      family = "fastEmbedR_approx",
      backend = "cpu_clustered",
      exact = FALSE,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_clustered"),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_cpu_nndescent_without_self",
      family = "fastEmbedR_approx",
      backend = "cpu_nndescent",
      exact = FALSE,
      kind = "direct",
      available = TRUE,
      run = function(x, k, seed, projection) fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_nndescent"),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_landmark_candidate_cpp_only",
      family = "fastEmbedR_approx",
      backend = "cpu",
      exact = FALSE,
      kind = "candidate_subroutine",
      available = TRUE,
      run = function(x, k, seed, projection) fastEmbedR:::landmark_candidate_knn_cpp(
        x,
        projection$projection$indices,
        as.integer(k),
        as.integer(projection$bucket_cols),
        as.integer(projection$query_cols),
        TRUE,
        as.integer(n_threads)
      ),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_metal_without_self",
      family = "fastEmbedR_exact",
      backend = "metal",
      exact = TRUE,
      kind = "direct",
      available = isTRUE(fastEmbedR::metal_available()),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Native Metal KNN is unavailable.",
      run = function(x, k, seed, projection) fastEmbedR:::nn_without_self(x, k = k, backend = "metal"),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_cuda_without_self",
      family = "fastEmbedR_exact",
      backend = "cuda",
      exact = TRUE,
      kind = "direct",
      available = isTRUE(fastEmbedR::cuda_available()),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Native CUDA KNN is unavailable.",
      run = function(x, k, seed, projection) fastEmbedR:::nn_without_self(x, k = k, backend = "cuda"),
      normalize = normalize_fastembedr
    ),
    list(
      name = "fastEmbedR_landmark_candidate_cuda_only",
      family = "fastEmbedR_approx",
      backend = "cuda",
      exact = FALSE,
      kind = "candidate_subroutine",
      available = isTRUE(fastEmbedR::cuda_available()),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Native CUDA candidate KNN is unavailable.",
      run = function(x, k, seed, projection) fastEmbedR:::landmark_candidate_knn_cuda_cpp(
        x,
        projection$projection$indices,
        as.integer(k),
        as.integer(projection$bucket_cols),
        as.integer(projection$query_cols)
      ),
      normalize = normalize_fastembedr
    )
  )

  if (!isTRUE(include_external)) return(impls)

  external <- list(
    list(
      name = "Rnanoflann_standard",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "Rnanoflann",
      available = requireNamespace("Rnanoflann", quietly = TRUE),
      run = function(x, k, seed, projection) Rnanoflann::nn(x, x, k = k + 1L, parallel = FALSE),
      normalize = normalize_Rnanoflann
    ),
    list(
      name = "Rnanoflann_parallel",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "Rnanoflann",
      available = requireNamespace("Rnanoflann", quietly = TRUE),
      run = function(x, k, seed, projection) Rnanoflann::nn(x, x, k = k + 1L, parallel = TRUE, cores = n_threads),
      normalize = normalize_Rnanoflann
    ),
    list(
      name = "FNN_kd_tree",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "FNN",
      available = requireNamespace("FNN", quietly = TRUE),
      run = function(x, k, seed, projection) FNN::get.knn(x, k = k, algorithm = "kd_tree"),
      normalize = normalize_FNN
    ),
    list(
      name = "FNN_cover_tree",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "FNN",
      available = requireNamespace("FNN", quietly = TRUE),
      run = function(x, k, seed, projection) FNN::get.knn(x, k = k, algorithm = "cover_tree"),
      normalize = normalize_FNN
    ),
    list(
      name = "FNN_CR",
      family = "external_cosine_unit_vector",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "FNN",
      available = requireNamespace("FNN", quietly = TRUE),
      run = function(x, k, seed, projection) FNN::get.knn(x, k = k, algorithm = "CR"),
      normalize = normalize_FNN
    ),
    list(
      name = "FNN_brute",
      family = "external_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "FNN",
      available = requireNamespace("FNN", quietly = TRUE),
      run = function(x, k, seed, projection) FNN::get.knn(x, k = k, algorithm = "brute"),
      normalize = normalize_FNN
    ),
    list(
      name = "RANN_kd",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "RANN",
      available = requireNamespace("RANN", quietly = TRUE),
      run = function(x, k, seed, projection) RANN::nn2(x, x, k = k + 1L, treetype = "kd", searchtype = "standard"),
      normalize = normalize_RANN
    ),
    list(
      name = "RANN_bd",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "RANN",
      available = requireNamespace("RANN", quietly = TRUE),
      run = function(x, k, seed, projection) RANN::nn2(x, x, k = k + 1L, treetype = "bd", searchtype = "standard"),
      normalize = normalize_RANN
    ),
    list(
      name = "dbscan_kdtree",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "dbscan",
      available = requireNamespace("dbscan", quietly = TRUE),
      run = function(x, k, seed, projection) dbscan::kNN(x, k = k, search = "kdtree"),
      normalize = normalize_dbscan
    ),
    list(
      name = "dbscan_linear",
      family = "external_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "dbscan",
      available = requireNamespace("dbscan", quietly = TRUE),
      run = function(x, k, seed, projection) dbscan::kNN(x, k = k, search = "linear"),
      normalize = normalize_dbscan
    ),
    list(
      name = "RcppHNSW",
      family = "external_approx",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "RcppHNSW",
      available = requireNamespace("RcppHNSW", quietly = TRUE),
      run = function(x, k, seed, projection) run_rcpphnsw(x, k, ef_search = max(50L, 4L * k), n_threads = n_threads, seed = seed),
      normalize = normalize_RcppHNSW
    ),
    list(
      name = "FAISS_IndexFlatL2",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "flat", n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_IVF_nlist16_nprobe4",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivf", nlist = 16L, nprobe = 4L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_IVF_nlist32_nprobe8",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivf", nlist = 32L, nprobe = 8L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_IVF_nlist64_nprobe16",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivf", nlist = 64L, nprobe = 16L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_IVFPQ_nlist16_nprobe4_m2",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivfpq", nlist = 16L, nprobe = 4L, pq_m = 2L, pq_nbits = 8L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_IVFPQ_nlist32_nprobe8_m2",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivfpq", nlist = 32L, nprobe = 8L, pq_m = 2L, pq_nbits = 8L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_GPU_IndexFlatL2",
      family = "external_faiss_gpu",
      backend = "cuda",
      exact = TRUE,
      kind = "direct",
      package = "faiss-gpu",
      available = faiss_gpu_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python FAISS with CUDA GPU support is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "flat", gpu = TRUE, device = 0L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_GPU_IVF_nlist64_nprobe16",
      family = "external_faiss_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "faiss-gpu",
      available = faiss_gpu_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python FAISS with CUDA GPU support is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivf", nlist = 64L, nprobe = 16L, gpu = TRUE, device = 0L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_GPU_IVF_nlist256_nprobe32",
      family = "external_faiss_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "faiss-gpu",
      available = faiss_gpu_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python FAISS with CUDA GPU support is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivf", nlist = 256L, nprobe = 32L, gpu = TRUE, device = 0L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_GPU_IVFPQ_nlist256_nprobe32_m2",
      family = "external_faiss_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "faiss-gpu",
      available = faiss_gpu_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python FAISS with CUDA GPU support is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivfpq", nlist = 256L, nprobe = 32L, pq_m = 2L, pq_nbits = 8L, gpu = TRUE, device = 0L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_GPU_IVFPQ_nlist1024_nprobe64_m2",
      family = "external_faiss_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "faiss-gpu",
      available = faiss_gpu_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python FAISS with CUDA GPU support is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "ivfpq", nlist = 1024L, nprobe = 64L, pq_m = 2L, pq_nbits = 8L, gpu = TRUE, device = 0L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "cuML_NearestNeighbors_auto",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = NA,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "auto", device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "cuML_NearestNeighbors_brute",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = TRUE,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "brute", device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "cuML_NearestNeighbors_ivfflat_nlist256_nprobe32",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "ivfflat", nlist = 256L, nprobe = 32L, device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "cuML_NearestNeighbors_ivfflat_nlist1024_nprobe64",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "ivfflat", nlist = 1024L, nprobe = 64L, device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "cuML_NearestNeighbors_ivfpq_nlist256_nprobe32",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "ivfpq", nlist = 256L, nprobe = 32L, device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "cuML_NearestNeighbors_ivfpq_nlist1024_nprobe64",
      family = "external_cuml_gpu",
      backend = "cuda",
      exact = FALSE,
      kind = "direct",
      package = "cuml",
      available = cuml_available(),
      unavailable_status = "backend_unavailable",
      unavailable_message = "Python RAPIDS cuML NearestNeighbors is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_cuml_nn(x, k, algorithm = "ivfpq", nlist = 1024L, nprobe = 64L, device = 0L),
      normalize = normalize_cuml
    ),
    list(
      name = "FAISS_HNSW_m16_efc200_efs50",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "hnsw", hnsw_m = 16L, hnsw_ef_construction = 200L, hnsw_ef_search = 50L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_HNSW_m16_efc200_efs100",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "hnsw", hnsw_m = 16L, hnsw_ef_construction = 200L, hnsw_ef_search = 100L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_HNSW_m32_efc200_efs50",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "hnsw", hnsw_m = 32L, hnsw_ef_construction = 200L, hnsw_ef_search = 50L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "FAISS_HNSW_m32_efc200_efs100",
      family = "external_faiss_cpu",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "faiss-cpu",
      available = faiss_cpu_available(),
      unavailable_status = "package_unavailable",
      unavailable_message = "Python package `faiss-cpu` is unavailable through reticulate.",
      run = function(x, k, seed, projection) run_faiss(x, k, "hnsw", hnsw_m = 32L, hnsw_ef_construction = 200L, hnsw_ef_search = 100L, n_threads = n_threads),
      normalize = normalize_FAISS
    ),
    list(
      name = "BiocNeighbors_exhaustive",
      family = "external_exact",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "BiocNeighbors",
      available = requireNamespace("BiocNeighbors", quietly = TRUE),
      run = function(x, k, seed, projection) run_biocneighbors(x, k, BiocNeighbors::ExhaustiveParam(), n_threads),
      normalize = normalize_BiocNeighbors
    ),
    list(
      name = "BiocNeighbors_kmknn",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "BiocNeighbors",
      available = requireNamespace("BiocNeighbors", quietly = TRUE),
      run = function(x, k, seed, projection) run_biocneighbors(x, k, BiocNeighbors::KmknnParam(), n_threads),
      normalize = normalize_BiocNeighbors
    ),
    list(
      name = "BiocNeighbors_vptree",
      family = "external_exact_or_index",
      backend = "cpu",
      exact = TRUE,
      kind = "direct",
      package = "BiocNeighbors",
      available = requireNamespace("BiocNeighbors", quietly = TRUE),
      run = function(x, k, seed, projection) run_biocneighbors(x, k, BiocNeighbors::VptreeParam(), n_threads),
      normalize = normalize_BiocNeighbors
    ),
    list(
      name = "BiocNeighbors_annoy",
      family = "external_approx",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "BiocNeighbors",
      available = requireNamespace("BiocNeighbors", quietly = TRUE),
      run = function(x, k, seed, projection) run_biocneighbors(x, k, BiocNeighbors::AnnoyParam(ntrees = 20L, search.mult = 50L), n_threads),
      normalize = normalize_BiocNeighbors
    ),
    list(
      name = "BiocNeighbors_hnsw",
      family = "external_approx",
      backend = "cpu",
      exact = FALSE,
      kind = "direct",
      package = "BiocNeighbors",
      available = requireNamespace("BiocNeighbors", quietly = TRUE),
      run = function(x, k, seed, projection) run_biocneighbors(x, k, BiocNeighbors::HnswParam(ef.search = max(50L, 4L * k)), n_threads),
      normalize = normalize_BiocNeighbors
    )
  )

  c(impls, rcppannoy_impl_grid(), external)
}

prewarm_native <- function(k) {
  x <- matrix(rnorm(64L * 6L), 64L, 6L)
  invisible(try(fastEmbedR:::nn_without_self(x, k = min(k, 10L), backend = "cpu"), silent = TRUE))
  if (isTRUE(fastEmbedR::metal_available())) {
    invisible(try(fastEmbedR:::nn_without_self(x, k = min(k, 10L), backend = "metal"), silent = TRUE))
  }
}

row_for_unavailable <- function(dataset, n, p, k, seed, repeat_id, impl, status, message) {
  data.frame(
    dataset = dataset,
    n = n,
    p = p,
    k = k,
    recall_sample_size = NA_integer_,
    seed = seed,
    repeat_id = repeat_id,
    implementation = impl$name,
    family = impl$family,
    kind = impl$kind,
    backend_requested = impl$backend,
    backend_used = NA_character_,
    exact_declared = impl$exact,
    status = status,
    error_message = message,
    elapsed_sec = NA_real_,
    index_build_time_sec = NA_real_,
    query_time_sec = NA_real_,
    rss_before_mb = NA_real_,
    rss_after_mb = NA_real_,
    delta_rss_mb = NA_real_,
    result_size_mb = NA_real_,
    recall_at_k_vs_cpu_exact = NA_real_,
    mean_distance_error_vs_cpu_exact = NA_real_,
    rank_correlation_vs_cpu_exact = NA_real_,
    neighbor_overlap_vs_cpu_exact = NA_real_,
    distance_mae_vs_cpu_exact = NA_real_,
    stringsAsFactors = FALSE
  )
}

run_one <- function(dataset, impl, reference, sample_rows, projection, k, seed, repeat_id) {
  x <- dataset$x
  if (!isTRUE(impl$available)) {
    status <- impl$unavailable_status %||% "package_unavailable"
    msg <- impl$unavailable_message %||% paste0("Implementation `", impl$name, "` is unavailable.")
    return(row_for_unavailable(dataset$name, nrow(x), ncol(x), k, seed, repeat_id, impl, status, msg))
  }

  timed <- safe_time(impl$run(x, k, seed, projection))
  normalized <- NULL
  backend_used <- impl$backend
  overlap <- NA_real_
  dist_mae <- NA_real_
  rank_cor <- NA_real_
  size <- NA_real_
  index_build_time_sec <- NA_real_
  query_time_sec <- NA_real_
  status <- timed$status
  error_message <- timed$error
  if (identical(status, "success")) {
    index_build_time_sec <- finite_number_or(attr(timed$value, "index_build_time_sec"))
    query_time_sec <- finite_number_or(attr(timed$value, "query_time_sec"), timed$elapsed)
    normalized <- tryCatch(
      impl$normalize(timed$value, k, nrow(x)),
      error = function(e) {
        status <<- "failed"
        error_message <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(timed$value)) size <- as.numeric(utils::object.size(timed$value)) / 1024^2
    if (!is.null(normalized)) {
      overlap <- knn_recall_at_k(normalized, reference, k, sample_rows)
      dist_mae <- mean_distance_error(normalized, reference, sample_rows, k)
      rank_cor <- knn_rank_correlation(normalized, reference, k, sample_rows)
    }
    if (inherits(timed$value, "fastEmbedR_nn")) {
      backend_used <- attr(timed$value, "backend")
    }
  }

  data.frame(
    dataset = dataset$name,
    n = nrow(x),
    p = ncol(x),
    k = k,
    recall_sample_size = length(sample_rows),
    seed = seed,
    repeat_id = repeat_id,
    implementation = impl$name,
    family = impl$family,
    kind = impl$kind,
    backend_requested = impl$backend,
    backend_used = backend_used,
    exact_declared = impl$exact,
    status = status,
    error_message = error_message,
    elapsed_sec = timed$elapsed,
    index_build_time_sec = index_build_time_sec,
    query_time_sec = query_time_sec,
    rss_before_mb = timed$rss_before_mb,
    rss_after_mb = timed$rss_after_mb,
    delta_rss_mb = timed$delta_rss_mb,
    result_size_mb = size,
    recall_at_k_vs_cpu_exact = overlap,
    mean_distance_error_vs_cpu_exact = dist_mae,
    rank_correlation_vs_cpu_exact = rank_cor,
    neighbor_overlap_vs_cpu_exact = overlap,
    distance_mae_vs_cpu_exact = dist_mae,
    stringsAsFactors = FALSE
  )
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x)) y else x
}

summarize_results <- function(raw) {
  split_key <- interaction(raw$dataset, raw$implementation, drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(raw, split_key), function(x) {
    ok <- x[x$status == "success", , drop = FALSE]
    first <- x[1L, , drop = FALSE]
    data.frame(
      dataset = first$dataset,
      n = first$n,
      p = first$p,
      k = first$k,
      recall_sample_size = first$recall_sample_size,
      implementation = first$implementation,
      family = first$family,
      kind = first$kind,
      backend_requested = first$backend_requested,
      backend_used = paste(unique(na.omit(x$backend_used)), collapse = ";"),
      exact_declared = first$exact_declared,
      repeats = nrow(x),
      successes = nrow(ok),
      status = if (nrow(ok) > 0L) "success" else first$status,
      median_elapsed_sec = if (nrow(ok) > 0L) safe_median(ok$elapsed_sec) else NA_real_,
      median_index_build_time_sec = if (nrow(ok) > 0L) safe_median(ok$index_build_time_sec) else NA_real_,
      median_query_time_sec = if (nrow(ok) > 0L) safe_median(ok$query_time_sec) else NA_real_,
      min_elapsed_sec = if (nrow(ok) > 0L) min(ok$elapsed_sec) else NA_real_,
      max_elapsed_sec = if (nrow(ok) > 0L) max(ok$elapsed_sec) else NA_real_,
      median_delta_rss_mb = if (nrow(ok) > 0L) safe_median(ok$delta_rss_mb) else NA_real_,
      median_result_size_mb = if (nrow(ok) > 0L) safe_median(ok$result_size_mb) else NA_real_,
      median_recall_at_k_vs_cpu_exact = if (nrow(ok) > 0L) safe_median(ok$recall_at_k_vs_cpu_exact) else NA_real_,
      median_mean_distance_error_vs_cpu_exact = if (nrow(ok) > 0L) safe_median(ok$mean_distance_error_vs_cpu_exact) else NA_real_,
      median_rank_correlation_vs_cpu_exact = if (nrow(ok) > 0L) safe_median(ok$rank_correlation_vs_cpu_exact) else NA_real_,
      median_neighbor_overlap_vs_cpu_exact = if (nrow(ok) > 0L) safe_median(ok$neighbor_overlap_vs_cpu_exact) else NA_real_,
      median_distance_mae_vs_cpu_exact = if (nrow(ok) > 0L) safe_median(ok$distance_mae_vs_cpu_exact) else NA_real_,
      error_message = paste(unique(na.omit(x$error_message[x$status != "success"])), collapse = " | "),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out[order(out$dataset, out$median_elapsed_sec, out$implementation), , drop = FALSE]
}

defaults <- list(
  datasets = "iris,synthetic_1000x20,synthetic_3000x20",
  k = "15",
  recall_sample_size = "1000",
  repeats = "3",
  seed = "4",
  out_dir = "results/knn_speed",
  include_external = "true",
  n_threads = as.character(max(1L, parallel::detectCores(logical = FALSE))),
  warmup = "true"
)

args <- parse_args(defaults)
datasets <- split_arg(args$datasets)
k <- as.integer(args$k)
recall_sample_size <- as.integer(args$recall_sample_size)
repeats <- as.integer(args$repeats)
seed <- as.integer(args$seed)
out_dir <- args$out_dir
include_external <- as_logical_arg(args$include_external)
n_threads <- as.integer(args$n_threads)
warmup <- as_logical_arg(args$warmup)

if (length(k) != 1L || is.na(k) || k < 1L) stop("`k` must be positive.", call. = FALSE)
if (length(recall_sample_size) != 1L || is.na(recall_sample_size) || recall_sample_size < 1L) {
  stop("`recall-sample-size` must be positive.", call. = FALSE)
}
if (length(repeats) != 1L || is.na(repeats) || repeats < 1L) stop("`repeats` must be positive.", call. = FALSE)
if (length(seed) != 1L || is.na(seed)) stop("`seed` must be an integer.", call. = FALSE)
if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) n_threads <- 1L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (warmup) prewarm_native(k)

impls <- make_impls(include_external = include_external, n_threads = n_threads)
dataset_objects <- lapply(seq_along(datasets), function(i) make_dataset(datasets[[i]], seed + i - 1L))

raw_rows <- list()
row_id <- 1L
for (dataset in dataset_objects) {
  x <- dataset$x
  kk <- min(k, nrow(x) - 1L)
  set.seed(seed)
  sample_n <- min(recall_sample_size, nrow(x))
  sample_rows <- sort(sample(seq_len(nrow(x)), sample_n))
  message(
    "Reference CPU exact KNN for ", dataset$name,
    " subset (n=", nrow(x), ", p=", ncol(x), ", k=", kk,
    ", recall_sample_size=", sample_n, ")"
  )
  reference <- exact_reference_subset(x, sample_rows, kk)
  projection <- candidate_projection(x, kk, seed)

  for (impl in impls) {
    for (repeat_id in seq_len(repeats)) {
      message("Running ", dataset$name, " / ", impl$name, " / repeat ", repeat_id)
      raw_rows[[row_id]] <- run_one(dataset, impl, reference, sample_rows, projection, kk, seed, repeat_id)
      row_id <- row_id + 1L
    }
  }
}

raw <- do.call(rbind, raw_rows)
summary <- summarize_results(raw)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
raw_file <- file.path(out_dir, paste0("knn_implementation_raw_", stamp, ".csv"))
summary_file <- file.path(out_dir, paste0("knn_implementation_summary_", stamp, ".csv"))
latest_raw <- file.path(out_dir, "latest_knn_implementation_raw.csv")
latest_summary <- file.path(out_dir, "latest_knn_implementation_summary.csv")

utils::write.csv(raw, raw_file, row.names = FALSE)
utils::write.csv(summary, summary_file, row.names = FALSE)
utils::write.csv(raw, latest_raw, row.names = FALSE)
utils::write.csv(summary, latest_summary, row.names = FALSE)

print(summary[, c(
  "dataset",
  "implementation",
  "status",
  "backend_used",
  "median_elapsed_sec",
  "median_index_build_time_sec",
  "median_query_time_sec",
  "median_recall_at_k_vs_cpu_exact",
  "median_mean_distance_error_vs_cpu_exact",
  "median_rank_correlation_vs_cpu_exact",
  "median_delta_rss_mb"
)], row.names = FALSE)

cat("\nSaved:\n")
cat("  ", normalizePath(latest_raw), "\n", sep = "")
cat("  ", normalizePath(latest_summary), "\n", sep = "")
