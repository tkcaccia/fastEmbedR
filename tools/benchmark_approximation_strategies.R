#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastEmbedR))

parse_csv_arg <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    value <- Sys.getenv(paste0("FASTEMBEDR_APPROX_", toupper(gsub("-", "_", name))), default)
  } else {
    value <- sub(prefix, "", value[[1L]], fixed = TRUE)
  }
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

parse_scalar <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 0L) {
    Sys.getenv(paste0("FASTEMBEDR_APPROX_", toupper(gsub("-", "_", name))), default)
  } else {
    sub(prefix, "", value[[1L]], fixed = TRUE)
  }
}

as_logical_arg <- function(x) {
  tolower(trimws(as.character(x)[1L])) %in% c("1", "true", "yes", "y", "on")
}

parse_knn_reuse_modes <- function(value) {
  modes <- tolower(gsub("-", "_", trimws(value)))
  if (any(modes %in% c("both", "all"))) {
    modes <- unique(c("method_specific", "across_methods", modes[!modes %in% c("both", "all")]))
  }
  allowed <- c("method_specific", "across_methods")
  bad <- modes[!modes %in% allowed]
  if (length(bad) > 0L) {
    stop("Invalid --knn-reuse value: ", paste(bad, collapse = ","), ". Use method_specific, across_methods, or both.", call. = FALSE)
  }
  unique(modes)
}

normalize_knn_cache_format <- function(value) {
  value <- tolower(trimws(as.character(value)[1L]))
  if (!nzchar(value) || is.na(value)) value <- "rds"
  switch(
    value,
    r = "rds",
    rds = "rds",
    npz = "npz",
    h5 = "hdf5",
    hdf5 = "hdf5",
    arrow = "parquet",
    parquet = "parquet",
    stop("Unsupported --knn-cache-format value: ", value, ". Use rds, npz, hdf5, or parquet.", call. = FALSE)
  )
}

json_or_text <- function(x) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null")))
  }
  paste(paste(names(x), unlist(x), sep = "="), collapse = ";")
}

standardize_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  keep <- apply(x, 2L, function(col) all(is.finite(col)) && stats::sd(col) > 0)
  x <- x[, keep, drop = FALSE]
  x <- scale(x)
  storage.mode(x) <- "double"
  x
}

dataset_record <- function(name, x, labels = NULL, source = "generated") {
  if (!is.null(labels)) labels <- factor(labels)
  list(name = name, x = standardize_matrix(x), labels = labels, source = source)
}

make_synthetic_dataset <- function(seed = 11L) {
  set.seed(seed)
  n_per <- 180L
  p <- 18L
  labels <- factor(rep(seq_len(4L), each = n_per))
  centers <- matrix(0, 4L, p)
  centers[2L, 1L:5L] <- 2
  centers[3L, 6L:10L] <- 2
  centers[4L, 11L:15L] <- 2
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.85), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record("synthetic_720", x, labels, "generated Gaussian clusters")
}

make_synthetic_large_dataset <- function(seed = 11L, n_per = 3000L, p = 32L) {
  set.seed(seed)
  labels <- factor(rep(seq_len(4L), each = n_per))
  centers <- matrix(0, 4L, p)
  add_center_block <- function(row, start, stop) {
    if (start <= p) centers[row, start:min(stop, p)] <<- 2.2
  }
  add_center_block(2L, 1L, 8L)
  add_center_block(3L, 9L, 16L)
  add_center_block(4L, 17L, 24L)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.9), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record(sprintf("synthetic_%d", length(labels)), x, labels, "generated Gaussian clusters")
}

make_synthetic_rare_dataset <- function(seed = 11L, class_sizes = c(9000L, 2200L, 650L, 150L), p = 32L) {
  set.seed(seed)
  labels <- factor(rep(seq_along(class_sizes), times = class_sizes))
  centers <- matrix(0, length(class_sizes), p)
  add_center_block <- function(row, start, stop, value = 2.4) {
    if (row <= nrow(centers) && start <= p) centers[row, start:min(stop, p)] <<- value
  }
  add_center_block(2L, 1L, 8L)
  add_center_block(3L, 9L, 16L)
  add_center_block(4L, 17L, 24L, value = 2.8)
  x <- matrix(stats::rnorm(length(labels) * p, sd = 0.9), length(labels), p)
  x <- x + centers[as.integer(labels), , drop = FALSE]
  dataset_record(sprintf("synthetic_rare_%d", length(labels)), x, labels, "generated imbalanced Gaussian clusters")
}

load_digits_dataset <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  tryCatch({
    sk <- reticulate::import("sklearn.datasets")
    digits <- sk$load_digits()
    dataset_record("sklearn_digits_1797", digits$data, as.integer(digits$target), "sklearn load_digits")
  }, error = function(e) NULL)
}

load_named_dataset <- function(name) {
  name <- tolower(name)
  if (identical(name, "iris")) {
    return(dataset_record("iris", iris[, 1L:4L], iris$Species, "R datasets"))
  }
  if (identical(name, "synthetic") || identical(name, "synthetic_720")) {
    return(make_synthetic_dataset())
  }
  if (name %in% c("synthetic_large", "synthetic_12000")) {
    return(make_synthetic_large_dataset())
  }
  if (name %in% c("synthetic_rare", "synthetic_rare_12000", "rare_synthetic")) {
    return(make_synthetic_rare_dataset())
  }
  if (identical(name, "digits") || identical(name, "sklearn_digits")) {
    return(load_digits_dataset())
  }
  stop("Unknown dataset: ", name, call. = FALSE)
}

current_rss_mb <- function() {
  out <- suppressWarnings(system2("ps", c("-o", "rss=", "-p", as.character(Sys.getpid())), stdout = TRUE, stderr = FALSE))
  if (length(out) == 0L) return(NA_real_)
  rss_kb <- suppressWarnings(as.numeric(trimws(out[[1L]])))
  if (!is.finite(rss_kb)) NA_real_ else rss_kb / 1024
}

gpu_memory_gb <- function(backend) {
  if (identical(backend, "cuda") && nzchar(Sys.which("nvidia-smi"))) {
    out <- suppressWarnings(system2(
      "nvidia-smi",
      c("--query-gpu=memory.used", "--format=csv,noheader,nounits"),
      stdout = TRUE,
      stderr = FALSE
    ))
    value <- suppressWarnings(sum(as.numeric(trimws(out)), na.rm = TRUE))
    if (is.finite(value)) return(value / 1024)
  }
  NA_real_
}

coerce_layout <- function(x, n) {
  if (inherits(x, "fastEmbedR_embedding")) x <- x$layout
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  cfg <- attr(x, "fastEmbedR_config")
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n && ncol(x) == n) x <- t(x)
  if (nrow(x) != n || ncol(x) < 2L) {
    stop("Embedding did not return an n x 2 layout.", call. = FALSE)
  }
  out <- x[, seq_len(2L), drop = FALSE]
  if (!is.null(cfg)) attr(out, "fastEmbedR_config") <- cfg
  out
}

measure_strategy <- function(expr, n, backend) {
  invisible(gc())
  rss_before <- current_rss_mb()
  gpu_before <- gpu_memory_gb(backend)
  elapsed <- system.time({
    value <- force(expr)
  })[["elapsed"]]
  gpu_after <- gpu_memory_gb(backend)
  rss_after <- current_rss_mb()
  list(
    value = value,
    layout = coerce_layout(value, n),
    total_time_sec = as.numeric(elapsed),
    rss_before_mb = rss_before,
    rss_after_mb = rss_after,
    peak_ram_mb = if (is.finite(rss_before) && is.finite(rss_after)) max(rss_before, rss_after) else NA_real_,
    gpu_before_gb = gpu_before,
    gpu_after_gb = gpu_after,
    peak_gpu_gb = suppressWarnings(max(c(gpu_before, gpu_after), na.rm = TRUE))
  )
}

metric_k <- function(n) {
  values <- c(15L, 30L, 50L)
  values[values < n]
}

safe_metric <- function(metrics, name) {
  if (name %in% names(metrics)) metrics[[name]] else NA_real_
}

safe_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

safe_number <- function(x, default = NA_real_) {
  if (is.null(x) || length(x) == 0L) return(default)
  value <- suppressWarnings(as.numeric(x[1L]))
  if (is.finite(value)) value else default
}

safe_character <- function(x, default = NA_character_) {
  if (is.null(x) || length(x) == 0L) return(default)
  value <- as.character(x[1L])
  if (!is.na(value) && nzchar(value)) value else default
}

safe_logical <- function(x, default = NA) {
  if (is.null(x) || length(x) == 0L || is.na(x[1L])) return(default)
  isTRUE(x[1L])
}

safe_numeric_cor <- function(x, y, method = "pearson") {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3L) return(NA_real_)
  x <- x[ok]
  y <- y[ok]
  if (stats::sd(x) == 0 || stats::sd(y) == 0) return(NA_real_)
  suppressWarnings(stats::cor(x, y, method = method))
}

normalized_stress <- function(high_dist, low_dist) {
  ok <- is.finite(high_dist) & is.finite(low_dist)
  if (sum(ok) < 3L) return(NA_real_)
  high_dist <- high_dist[ok]
  low_dist <- low_dist[ok]
  high_scale <- sqrt(sum(high_dist * high_dist))
  low_scale <- sqrt(sum(low_dist * low_dist))
  if (high_scale == 0 || low_scale == 0) return(NA_real_)
  high_dist <- high_dist / high_scale
  low_dist <- low_dist / low_scale
  sqrt(sum((high_dist - low_dist)^2))
}

sampled_pair_distances <- function(x, a, b) {
  p <- ncol(x)
  batch_size <- max(512L, as.integer(floor(2e6 / max(1L, p))))
  out <- numeric(length(a))
  starts <- seq.int(1L, length(a), by = batch_size)
  for (start in starts) {
    end <- min(length(a), start + batch_size - 1L)
    rows <- start:end
    diff <- x[a[rows], , drop = FALSE] - x[b[rows], , drop = FALSE]
    out[rows] <- sqrt(rowSums(diff * diff))
  }
  out
}

fast_tsne_available <- function() {
  nzchar(fastEmbedR:::fitsne_binary_path(required = FALSE))
}

backend_supported <- function(backend) {
  if (identical(backend, "cpu")) return(TRUE)
  if (identical(backend, "metal")) return(isTRUE(fastEmbedR::metal_available()))
  if (identical(backend, "cuda")) return(isTRUE(fastEmbedR::cuda_available()))
  FALSE
}

method_quality <- function(method, quality) {
  if (identical(method, "tsne")) quality else "auto"
}

normalize_knn_metric <- function(metric) {
  metric <- tolower(trimws(as.character(metric)[1L]))
  if (!nzchar(metric) || is.na(metric)) return("euclidean")
  switch(
    metric,
    l2 = "euclidean",
    l1 = "manhattan",
    cityblock = "manhattan",
    metric
  )
}

kdtree_space <- function(x, pca_dims) {
  pca_dims <- as.integer(pca_dims)
  rank <- min(ncol(x), nrow(x) - 1L, pca_dims)
  if (rank < 1L || rank >= ncol(x)) return(x)
  pca <- stats::prcomp(x, center = FALSE, scale. = FALSE, rank. = rank)
  scores <- pca$x[, seq_len(rank), drop = FALSE]
  storage.mode(scores) <- "double"
  scores
}

finish_external_knn <- function(indices, distances, backend, exact = TRUE) {
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  storage.mode(indices) <- "integer"
  storage.mode(distances) <- "double"
  out <- list(indices = indices, distances = distances)
  attr(out, "backend") <- backend
  attr(out, "exact") <- isTRUE(exact)
  class(out) <- c("fastEmbedR_nn", "list")
  out
}

knn_has_self_first <- function(knn) {
  idx <- knn$indices
  if (!is.matrix(idx) || ncol(idx) == 0L || nrow(idx) == 0L) return(FALSE)
  mean(idx[, 1L] == seq_len(nrow(idx)), na.rm = TRUE) > 0.9
}

knn_neighbor_cols <- function(knn) {
  if (knn_has_self_first(knn)) {
    if (ncol(knn$indices) < 2L) integer() else seq.int(2L, ncol(knn$indices))
  } else {
    seq_len(ncol(knn$indices))
  }
}

knn_effective_k <- function(knn) {
  length(knn_neighbor_cols(knn))
}

graph_knn_result <- function(base_knn, indices, distances, approximation, exact = FALSE) {
  backend <- attr(base_knn, "backend")
  if (is.null(backend) || !nzchar(backend)) backend <- "unknown"
  finish_external_knn(
    indices,
    distances,
    backend = paste(backend, "graph", approximation, sep = "_"),
    exact = isTRUE(exact)
  )
}

graph_limit_knn <- function(knn, keep_k, approximation, exact = TRUE) {
  keep_k <- max(1L, min(as.integer(keep_k), knn_effective_k(knn)))
  self <- knn_has_self_first(knn)
  cols <- if (self) c(1L, seq.int(2L, keep_k + 1L)) else seq_len(keep_k)
  list(
    knn = graph_knn_result(
      knn,
      knn$indices[, cols, drop = FALSE],
      knn$distances[, cols, drop = FALSE],
      approximation,
      exact = exact && isTRUE(attr(knn, "exact"))
    ),
    graph_approximation = approximation,
    graph_effective_k = keep_k,
    graph_edge_retention = keep_k / max(1L, knn_effective_k(knn))
  )
}

graph_distance_percentile_prune_knn <- function(knn, drop_fraction) {
  drop_fraction <- as.numeric(drop_fraction)
  if (!is.finite(drop_fraction) || drop_fraction <= 0 || drop_fraction >= 1) {
    stop("Distance-percentile graph pruning requires 0 < drop_fraction < 1.", call. = FALSE)
  }
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  if (base_k < 2L) {
    stop("Distance-percentile graph pruning requires at least two non-self neighbours.", call. = FALSE)
  }
  remove_k <- max(1L, min(base_k - 1L, ceiling(base_k * drop_fraction)))
  keep_k <- base_k - remove_k
  keep_cols <- cols[seq_len(keep_k)]
  removed_cols <- cols[seq.int(keep_k + 1L, base_k)]
  out_indices <- knn$indices[, keep_cols, drop = FALSE]
  out_distances <- knn$distances[, keep_cols, drop = FALSE]
  thresholds <- out_distances[, keep_k]
  removed_distances <- knn$distances[, removed_cols, drop = FALSE]
  if (self) {
    out_indices <- cbind(knn$indices[, 1L], out_indices)
    out_distances <- cbind(knn$distances[, 1L], out_distances)
  }
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("distance_top", round(100 * drop_fraction)),
      exact = isTRUE(attr(knn, "exact"))
    ),
    graph_approximation = paste0("distance_percentile_prune_top", round(100 * drop_fraction)),
    graph_effective_k = keep_k,
    graph_edge_retention = keep_k / base_k,
    graph_recall_at_k = keep_k / base_k,
    graph_mean_degree = keep_k,
    graph_min_degree = keep_k,
    graph_max_degree = keep_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_distance_prune_drop_fraction = drop_fraction,
    graph_distance_prune_percentile = 100 * (1 - drop_fraction),
    graph_distance_prune_removed_edges_mean = remove_k,
    graph_distance_prune_threshold_mean = safe_mean(thresholds),
    graph_distance_prune_threshold_min = if (length(thresholds) == 0L) NA_real_ else min(thresholds, na.rm = TRUE),
    graph_distance_prune_threshold_max = if (length(thresholds) == 0L) NA_real_ else max(thresholds, na.rm = TRUE),
    graph_distance_prune_removed_distance_mean = safe_mean(removed_distances)
  )
}

graph_undirected_edge_table <- function(knn) {
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  if (length(cols) == 0L) {
    return(list(a = integer(), b = integer(), distance = numeric()))
  }
  row_id <- rep(seq_len(n), each = length(cols))
  neigh <- as.integer(as.vector(t(knn$indices[, cols, drop = FALSE])))
  dist <- as.numeric(as.vector(t(knn$distances[, cols, drop = FALSE])))
  valid <- is.finite(dist) & is.finite(neigh) & neigh >= 1L & neigh <= n & neigh != row_id
  if (!any(valid)) {
    return(list(a = integer(), b = integer(), distance = numeric()))
  }
  a <- pmin(row_id[valid], neigh[valid])
  b <- pmax(row_id[valid], neigh[valid])
  dist <- dist[valid]
  key <- paste(a, b, sep = ":")
  ord <- order(key, dist)
  first <- !duplicated(key[ord])
  list(
    a = as.integer(a[ord][first]),
    b = as.integer(b[ord][first]),
    distance = as.numeric(dist[ord][first])
  )
}

graph_component_count <- function(n, a, b) {
  if (n <= 0L) return(0L)
  parent <- seq_len(n)
  find_root <- function(x) {
    while (parent[x] != x) {
      parent[x] <<- parent[parent[x]]
      x <- parent[x]
    }
    x
  }
  union_edge <- function(x, y) {
    rx <- find_root(x)
    ry <- find_root(y)
    if (rx != ry) parent[ry] <<- rx
  }
  if (length(a) > 0L) {
    for (i in seq_along(a)) union_edge(as.integer(a[i]), as.integer(b[i]))
  }
  length(unique(vapply(seq_len(n), find_root, integer(1L))))
}

graph_minimum_spanning_forest <- function(n, edges) {
  if (length(edges$a) == 0L) {
    return(list(a = integer(), b = integer(), distance = numeric()))
  }
  parent <- seq_len(n)
  find_root <- function(x) {
    while (parent[x] != x) {
      parent[x] <<- parent[parent[x]]
      x <- parent[x]
    }
    x
  }
  union_edge <- function(x, y) {
    rx <- find_root(x)
    ry <- find_root(y)
    if (rx == ry) return(FALSE)
    parent[ry] <<- rx
    TRUE
  }
  ord <- order(edges$distance, edges$a, edges$b)
  keep <- logical(length(ord))
  for (i in seq_along(ord)) {
    j <- ord[i]
    keep[i] <- union_edge(as.integer(edges$a[j]), as.integer(edges$b[j]))
  }
  kept <- ord[keep]
  list(
    a = as.integer(edges$a[kept]),
    b = as.integer(edges$b[kept]),
    distance = as.numeric(edges$distance[kept])
  )
}

graph_distance_prune_mst_rescue_knn <- function(knn, drop_fraction) {
  pruned_result <- graph_distance_percentile_prune_knn(knn, drop_fraction)
  pruned <- pruned_result$knn
  n <- nrow(knn$indices)
  self <- knn_has_self_first(pruned)
  cols <- knn_neighbor_cols(pruned)
  base_k <- max(1L, knn_effective_k(knn))
  base_edges <- graph_undirected_edge_table(knn)
  if (length(base_edges$a) == 0L) {
    stop("MST rescue requires at least one valid base KNN edge.", call. = FALSE)
  }
  pruned_edges <- graph_undirected_edge_table(pruned)
  base_components <- graph_component_count(n, base_edges$a, base_edges$b)
  before_components <- graph_component_count(n, pruned_edges$a, pruned_edges$b)
  forest <- graph_minimum_spanning_forest(n, base_edges)

  adj_indices <- vector("list", n)
  adj_distances <- vector("list", n)
  for (i in seq_len(n)) {
    idx <- as.integer(pruned$indices[i, cols])
    dst <- as.numeric(pruned$distances[i, cols])
    valid <- is.finite(dst) & is.finite(idx) & idx >= 1L & idx <= n & idx != i
    adj_indices[[i]] <- idx[valid]
    adj_distances[[i]] <- dst[valid]
  }

  added_directed <- 0L
  added_forest_edges <- 0L
  if (length(forest$a) > 0L) {
    for (e in seq_along(forest$a)) {
      a <- forest$a[e]
      b <- forest$b[e]
      d <- forest$distance[e]
      added_any <- FALSE
      if (!b %in% adj_indices[[a]]) {
        adj_indices[[a]] <- c(adj_indices[[a]], b)
        adj_distances[[a]] <- c(adj_distances[[a]], d)
        added_directed <- added_directed + 1L
        added_any <- TRUE
      }
      if (!a %in% adj_indices[[b]]) {
        adj_indices[[b]] <- c(adj_indices[[b]], a)
        adj_distances[[b]] <- c(adj_distances[[b]], d)
        added_directed <- added_directed + 1L
        added_any <- TRUE
      }
      if (isTRUE(added_any)) added_forest_edges <- added_forest_edges + 1L
    }
  }

  degrees <- lengths(adj_indices)
  after_edges_a <- integer(0)
  after_edges_b <- integer(0)
  for (i in seq_len(n)) {
    if (length(adj_indices[[i]]) > 0L) {
      after_edges_a <- c(after_edges_a, rep(i, length(adj_indices[[i]])))
      after_edges_b <- c(after_edges_b, adj_indices[[i]])
    }
  }
  after_components <- graph_component_count(n, after_edges_a, after_edges_b)
  out_k <- max(1L, max(degrees))
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  finite_dist <- knn$distances[is.finite(knn$distances) & knn$distances > 0]
  pad_distance <- if (length(finite_dist) == 0L) 1e6 else max(finite_dist) * 1e3 + 1
  if (self) {
    out_indices[, 1L] <- pruned$indices[, 1L]
    out_distances[, 1L] <- pruned$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  base_cols <- knn_neighbor_cols(knn)
  for (i in seq_len(n)) {
    if (degrees[i] > 0L) {
      ord <- order(adj_distances[[i]], adj_indices[[i]])
      idx <- adj_indices[[i]][ord]
      dst <- adj_distances[[i]][ord]
      out_indices[i, offset + seq_along(idx)] <- idx
      out_distances[i, offset + seq_along(dst)] <- dst
    } else {
      fallback_pos <- if (length(base_cols) > 0L) base_cols[1L] else NA_integer_
      idx <- if (is.na(fallback_pos)) if (i == 1L) 2L else 1L else knn$indices[i, fallback_pos]
      out_indices[i, offset + 1L] <- idx
      out_distances[i, offset + 1L] <- pad_distance
    }
    if (degrees[i] < out_k) {
      start <- max(1L, degrees[i] + 1L)
      pad_cols <- offset + seq.int(start, out_k)
      fallback <- out_indices[i, offset + 1L]
      if (is.na(fallback)) fallback <- if (i == 1L) 2L else 1L
      out_indices[i, pad_cols] <- fallback
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("distance_mst_top", round(100 * drop_fraction)),
      exact = FALSE
    ),
    graph_approximation = paste0("distance_percentile_prune_mst_rescue_top", round(100 * drop_fraction)),
    graph_effective_k = out_k,
    graph_edge_retention = mean(degrees) / base_k,
    graph_recall_at_k = mean(degrees) / base_k,
    graph_mean_degree = mean(degrees),
    graph_min_degree = min(degrees),
    graph_max_degree = max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k),
    graph_distance_prune_drop_fraction = pruned_result$graph_distance_prune_drop_fraction,
    graph_distance_prune_percentile = pruned_result$graph_distance_prune_percentile,
    graph_distance_prune_removed_edges_mean = pruned_result$graph_distance_prune_removed_edges_mean,
    graph_distance_prune_threshold_mean = pruned_result$graph_distance_prune_threshold_mean,
    graph_distance_prune_threshold_min = pruned_result$graph_distance_prune_threshold_min,
    graph_distance_prune_threshold_max = pruned_result$graph_distance_prune_threshold_max,
    graph_distance_prune_removed_distance_mean = pruned_result$graph_distance_prune_removed_distance_mean,
    graph_mst_rescue_enabled = 1,
    graph_mst_rescue_base_components = base_components,
    graph_mst_rescue_components_before = before_components,
    graph_mst_rescue_components_after = after_components,
    graph_mst_rescue_forest_edges = length(forest$a),
    graph_mst_rescue_added_forest_edges = added_forest_edges,
    graph_mst_rescue_added_directed_edges = added_directed,
    graph_mst_rescue_mean_degree_before = pruned_result$graph_mean_degree,
    graph_mst_rescue_mean_degree_after = mean(degrees)
  )
}

graph_local_density_scale <- function(knn, density_quantile = 0.5) {
  cols <- knn_neighbor_cols(knn)
  if (length(cols) == 0L) {
    return(rep(1, nrow(knn$indices)))
  }
  density_quantile <- max(0, min(1, as.numeric(density_quantile)))
  scale <- apply(knn$distances[, cols, drop = FALSE], 1L, function(row) {
    row <- row[is.finite(row) & row > 0]
    if (length(row) == 0L) return(NA_real_)
    as.numeric(stats::quantile(row, probs = density_quantile, names = FALSE, type = 7))
  })
  fallback <- stats::median(scale[is.finite(scale) & scale > 0], na.rm = TRUE)
  if (!is.finite(fallback) || fallback <= 0) fallback <- 1
  scale[!is.finite(scale) | scale <= 0] <- fallback
  scale
}

graph_density_corrected_knn <- function(knn,
                                        mode = c("geomean", "sparse_boost", "densmap_radius"),
                                        density_quantile = 0.5,
                                        strength = 1,
                                        clamp = c(0.25, 4)) {
  mode <- match.arg(mode)
  strength <- as.numeric(strength)
  if (!is.finite(strength) || strength <= 0) {
    stop("Density correction strength must be positive.", call. = FALSE)
  }
  clamp <- as.numeric(clamp)
  if (length(clamp) != 2L || any(!is.finite(clamp)) || clamp[1L] <= 0 || clamp[2L] <= clamp[1L]) {
    stop("Density correction clamp must be an increasing positive length-2 vector.", call. = FALSE)
  }
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  if (length(cols) < 2L) {
    stop("Density-corrected graph weights require at least two non-self neighbours.", call. = FALSE)
  }
  scale <- graph_local_density_scale(knn, density_quantile)
  global_scale <- stats::median(scale, na.rm = TRUE)
  if (!is.finite(global_scale) || global_scale <= 0) global_scale <- 1
  distances <- knn$distances
  corrections <- numeric(0)
  was_clamped <- logical(0)
  for (i in seq_len(n)) {
    idx <- as.integer(knn$indices[i, cols])
    dst <- as.numeric(knn$distances[i, cols])
    valid <- is.finite(dst) & idx >= 1L & idx <= n
    target_scale <- rep(scale[i], length(cols))
    target_scale[valid] <- scale[idx[valid]]
    pair_scale <- sqrt(scale[i] * target_scale)
    pair_scale[!is.finite(pair_scale) | pair_scale <= 0] <- global_scale
    correction <- switch(
      mode,
      geomean = (global_scale / pair_scale)^strength,
      sparse_boost = {
        denom <- pmax(scale[i], target_scale)
        denom[!is.finite(denom) | denom <= 0] <- global_scale
        (global_scale / denom)^strength
      },
      densmap_radius = (pair_scale / global_scale)^strength
    )
    correction[!is.finite(correction) | correction <= 0] <- 1
    unclamped <- correction
    correction <- pmin(clamp[2L], pmax(clamp[1L], correction))
    distances[i, cols] <- dst * correction
    corrections <- c(corrections, correction[is.finite(dst)])
    was_clamped <- c(was_clamped, (unclamped < clamp[1L] | unclamped > clamp[2L])[is.finite(dst)])
  }
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  clamp_fraction <- if (length(corrections) == 0L) {
    NA_real_
  } else {
    mean(was_clamped)
  }
  density_rank_cor <- suppressWarnings(stats::cor(
    scale,
    apply(distances[, cols, drop = FALSE], 1L, function(row) safe_mean(row[row > 0])),
    method = "spearman"
  ))
  list(
    knn = graph_knn_result(
      knn,
      knn$indices,
      distances,
      paste0("density_", mode),
      exact = FALSE
    ),
    graph_approximation = paste0("density_corrected_", mode),
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1,
    graph_recall_at_k = 1,
    graph_mean_degree = knn_effective_k(knn),
    graph_min_degree = knn_effective_k(knn),
    graph_max_degree = knn_effective_k(knn),
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_density_correction_method = mode,
    graph_density_correction_quantile = density_quantile,
    graph_density_correction_strength = strength,
    graph_density_scale_mean = safe_mean(scale),
    graph_density_scale_min = min(scale, na.rm = TRUE),
    graph_density_scale_max = max(scale, na.rm = TRUE),
    graph_density_scale_cv = stats::sd(scale, na.rm = TRUE) / max(abs(safe_mean(scale)), .Machine$double.eps),
    graph_density_sparse_fraction = mean(scale > global_scale),
    graph_density_correction_mean = safe_mean(corrections),
    graph_density_correction_min = if (length(corrections) == 0L) NA_real_ else min(corrections, na.rm = TRUE),
    graph_density_correction_max = if (length(corrections) == 0L) NA_real_ else max(corrections, na.rm = TRUE),
    graph_density_correction_clamp_fraction = clamp_fraction,
    graph_density_corrected_distance_scale_cor = density_rank_cor
  )
}

umap_row_membership <- function(row_distances, target_scale = 1, local_connectivity = 1) {
  d <- as.numeric(row_distances)
  finite <- is.finite(d)
  positive <- d[finite & d > 0]
  positive <- sort(positive)
  lc <- max(0, as.numeric(local_connectivity))
  lc_floor <- floor(lc)
  lc_interp <- lc - lc_floor
  rho <- 0
  if (length(positive) > 0L) {
    if (lc_floor > 0L) {
      if (lc_floor < length(positive)) {
        rho <- positive[lc_floor]
        if (lc_interp > 1e-12) {
          rho <- rho + lc_interp * (positive[lc_floor + 1L] - positive[lc_floor])
        }
      } else {
        rho <- positive[length(positive)]
      }
    } else if (lc_interp > 1e-12) {
      rho <- lc_interp * positive[1L]
    }
  }
  target <- max(1e-3, log2(max(2L, length(d))) * as.numeric(target_scale))
  lo <- 0
  hi <- Inf
  mid <- 1
  for (iter in seq_len(48L)) {
    adjusted <- d - rho
    psum <- sum(ifelse(finite & adjusted <= 0, 1, ifelse(finite, exp(-adjusted / mid), 0)))
    if (abs(psum - target) < 1e-5) break
    if (psum > target) {
      hi <- mid
      mid <- (lo + hi) / 2
    } else {
      lo <- mid
      mid <- if (is.infinite(hi)) mid * 2 else (lo + hi) / 2
    }
  }
  sigma <- max(mid, 1e-6)
  adjusted <- d - rho
  weight <- ifelse(finite & adjusted <= 0, 1, ifelse(finite, exp(-adjusted / sigma), 0))
  pmin(1, pmax(0, weight))
}

umap_fuzzy_affinity_knn <- function(knn,
                                    set_op_mix_ratio = 1,
                                    local_connectivity = 1,
                                    weight_power = 1,
                                    target_scale = 1,
                                    approximation = "umap_fuzzy_union") {
  if (!is.matrix(knn$indices)) knn$indices <- as.matrix(knn$indices)
  if (!is.matrix(knn$distances)) knn$distances <- as.matrix(knn$distances)
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  if (n < 2L || base_k < 1L) {
    stop("UMAP fuzzy graph transform requires at least one non-self neighbour.", call. = FALSE)
  }
  idx <- knn$indices[, cols, drop = FALSE]
  dist <- knn$distances[, cols, drop = FALSE]
  storage.mode(idx) <- "integer"
  storage.mode(dist) <- "double"
  min_idx <- suppressWarnings(min(idx, na.rm = TRUE))
  max_idx <- suppressWarnings(max(idx, na.rm = TRUE))
  offset <- if (is.finite(min_idx) && is.finite(max_idx) && min_idx >= 1 && max_idx <= n) 1L else 0L

  max_edges <- n * base_k
  from <- integer(max_edges)
  to <- integer(max_edges)
  weight <- numeric(max_edges)
  cursor <- 0L
  for (i in seq_len(n)) {
    nb0 <- as.integer(idx[i, ]) - offset
    valid <- is.finite(dist[i, ]) & nb0 >= 0L & nb0 < n & nb0 != (i - 1L)
    if (!any(valid)) next
    row_weight <- umap_row_membership(
      dist[i, ],
      target_scale = target_scale,
      local_connectivity = local_connectivity
    )
    row_weight[!valid] <- 0
    keep <- which(row_weight > 0)
    if (length(keep) == 0L) next
    rng <- seq.int(cursor + 1L, cursor + length(keep))
    from[rng] <- i - 1L
    to[rng] <- nb0[keep]
    weight[rng] <- row_weight[keep]
    cursor <- cursor + length(keep)
  }
  if (cursor == 0L) {
    stop("UMAP fuzzy graph transform produced no usable edges.", call. = FALSE)
  }
  from <- from[seq_len(cursor)]
  to <- to[seq_len(cursor)]
  weight <- weight[seq_len(cursor)]
  a <- pmin(from, to)
  b <- pmax(from, to)
  key <- a * n + b + 1
  ord <- order(key, from, to)
  key <- key[ord]
  from <- from[ord]
  to <- to[ord]
  weight <- weight[ord]

  mix <- min(1, max(0, as.numeric(set_op_mix_ratio)))
  power <- max(1e-6, as.numeric(weight_power))
  row_neighbors <- vector("list", n)
  row_weights <- vector("list", n)
  start <- 1L
  while (start <= length(key)) {
    end <- start
    while (end < length(key) && key[end + 1L] == key[start]) end <- end + 1L
    group_from <- from[start:end]
    group_to <- to[start:end]
    group_weight <- weight[start:end]
    aa <- min(group_from[1L], group_to[1L])
    bb <- max(group_from[1L], group_to[1L])
    forward_values <- group_weight[group_from == aa & group_to == bb]
    reverse_values <- group_weight[group_from == bb & group_to == aa]
    forward <- if (length(forward_values) > 0L) max(forward_values) else 0
    reverse <- if (length(reverse_values) > 0L) max(reverse_values) else 0
    union <- forward + reverse - forward * reverse
    intersection <- forward * reverse
    w <- mix * union + (1 - mix) * intersection
    w <- pmin(1, pmax(1e-12, w^power))
    ia <- aa + 1L
    ib <- bb + 1L
    row_neighbors[[ia]] <- c(row_neighbors[[ia]], ib)
    row_weights[[ia]] <- c(row_weights[[ia]], w)
    row_neighbors[[ib]] <- c(row_neighbors[[ib]], ia)
    row_weights[[ib]] <- c(row_weights[[ib]], w)
    start <- end + 1L
  }

  out_idx <- matrix(NA_integer_, n, base_k)
  out_dist <- matrix(NA_real_, n, base_k)
  fallback_distance <- -log(1e-12)
  for (i in seq_len(n)) {
    nb <- as.integer(row_neighbors[[i]])
    w <- as.numeric(row_weights[[i]])
    if (length(nb) > 0L) {
      ord_row <- order(-w, nb)
      nb <- nb[ord_row]
      w <- w[ord_row]
      keep <- !duplicated(nb) & nb != i
      nb <- nb[keep]
      w <- w[keep]
    }
    if (length(nb) < base_k) {
      base_nb <- as.integer(idx[i, ])
      base_nb <- base_nb[is.finite(base_nb) & base_nb >= 1L & base_nb <= n & base_nb != i]
      fill <- base_nb[!base_nb %in% nb]
      if (length(fill) < base_k - length(nb)) {
        fill2 <- seq_len(n)
        fill2 <- fill2[fill2 != i & !fill2 %in% c(nb, fill)]
        fill <- c(fill, fill2)
      }
      need <- base_k - length(nb)
      if (need > 0L && length(fill) > 0L) {
        add <- fill[seq_len(min(need, length(fill)))]
        nb <- c(nb, add)
        w <- c(w, rep(1e-12, length(add)))
      }
    }
    keep_n <- min(base_k, length(nb))
    out_idx[i, seq_len(keep_n)] <- nb[seq_len(keep_n)]
    out_dist[i, seq_len(keep_n)] <- -log(pmin(1, pmax(1e-12, w[seq_len(keep_n)])))
    if (keep_n < base_k) {
      fill <- seq_len(n)
      fill <- fill[fill != i & !fill %in% out_idx[i, seq_len(keep_n)]]
      extra <- seq.int(keep_n + 1L, base_k)
      out_idx[i, extra] <- fill[seq_len(length(extra))]
      out_dist[i, extra] <- fallback_distance
    }
  }
  transformed <- graph_knn_result(knn, out_idx, out_dist, approximation, exact = FALSE)
  finite_weights <- exp(-out_dist[is.finite(out_dist)])
  list(
    knn = transformed,
    graph_approximation = approximation,
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    umap_graph_set_op_mix_ratio = mix,
    umap_graph_local_connectivity = as.numeric(local_connectivity),
    umap_graph_weight_power = power,
    umap_graph_target_scale = as.numeric(target_scale),
    umap_graph_distance_transform = "negative_log_fuzzy_membership",
    umap_graph_mean_weight = safe_mean(finite_weights),
    umap_graph_min_weight = if (length(finite_weights) == 0L) NA_real_ else min(finite_weights, na.rm = TRUE),
    umap_graph_max_weight = if (length(finite_weights) == 0L) NA_real_ else max(finite_weights, na.rm = TRUE)
  )
}

graph_weighted_edge_sample_knn <- function(knn,
                                           keep_fraction = 0.5,
                                           weight_power = 1,
                                           include_top = 1L,
                                           target_scale = 1,
                                           seed = 1L) {
  if (!is.matrix(knn$indices)) knn$indices <- as.matrix(knn$indices)
  if (!is.matrix(knn$distances)) knn$distances <- as.matrix(knn$distances)
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  keep_fraction <- as.numeric(keep_fraction)
  if (!is.finite(keep_fraction) || keep_fraction <= 0 || keep_fraction > 1) {
    stop("Weighted edge sampling requires 0 < keep_fraction <= 1.", call. = FALSE)
  }
  if (base_k < 2L) {
    stop("Weighted edge sampling requires at least two non-self neighbours.", call. = FALSE)
  }
  keep_k <- max(1L, min(base_k, ceiling(base_k * keep_fraction)))
  include_top <- max(0L, min(as.integer(include_top), keep_k))
  weight_power <- max(1e-6, as.numeric(weight_power))

  native_sampler <- get0("weighted_edge_sample_knn_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (is.function(native_sampler)) {
    native <- tryCatch(
      native_sampler(
        knn$indices,
        knn$distances,
        keep_fraction,
        weight_power,
        include_top,
        as.numeric(target_scale),
        as.integer(seed)
      ),
      error = function(e) NULL
    )
    if (is.list(native) && !is.null(native$indices) && !is.null(native$distances)) {
      approximation <- paste0("weighted_edge_sample_", round(100 * keep_fraction))
      return(list(
        knn = graph_knn_result(knn, native$indices, native$distances, approximation, exact = FALSE),
        graph_approximation = approximation,
        graph_effective_k = safe_number(native$graph_effective_k),
        graph_edge_retention = safe_number(native$graph_edge_retention),
        graph_mean_degree = safe_number(native$graph_mean_degree),
        graph_min_degree = safe_number(native$graph_min_degree),
        graph_max_degree = safe_number(native$graph_max_degree),
        graph_isolated_fraction = safe_number(native$graph_isolated_fraction),
        graph_padding_fraction = safe_number(native$graph_padding_fraction),
        graph_edge_sampling_method = "umap_fuzzy_weighted_without_replacement_cpp",
        graph_edge_sampling_fraction = keep_fraction,
        graph_edge_sampling_weight_power = weight_power,
        graph_edge_sampling_include_top = include_top,
        graph_edge_sampling_target_scale = as.numeric(target_scale),
        graph_edge_sampling_mean_selected_weight =
          safe_number(native$graph_edge_sampling_mean_selected_weight),
        graph_edge_sampling_mean_candidate_weight =
          safe_number(native$graph_edge_sampling_mean_candidate_weight),
        graph_edge_sampling_selected_to_candidate_weight_ratio =
          safe_number(native$graph_edge_sampling_selected_to_candidate_weight_ratio)
      ))
    }
  }

  out_cols <- keep_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)

  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else {
    NULL
  }
  on.exit({
    if (is.null(old_seed)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))

  selected_weight_sum <- 0
  selected_weight_count <- 0L
  all_weight_sum <- 0
  all_weight_count <- 0L
  for (i in seq_len(n)) {
    idx <- as.integer(knn$indices[i, cols])
    dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(dst) & is.finite(idx) & idx >= 1L & idx <= n & idx != i)
    if (length(valid) == 0L) valid <- seq_len(base_k)
    membership <- umap_row_membership(dst, target_scale = target_scale)^weight_power
    membership[!is.finite(membership) | membership < 0] <- 0
    membership[setdiff(seq_len(base_k), valid)] <- 0
    valid_membership <- membership[valid]
    valid_membership <- valid_membership[is.finite(valid_membership)]
    if (length(valid_membership) > 0L) {
      all_weight_sum <- all_weight_sum + sum(valid_membership)
      all_weight_count <- all_weight_count + length(valid_membership)
    }

    nearest <- valid[order(dst[valid], valid)]
    chosen <- integer(0)
    if (include_top > 0L) {
      chosen <- nearest[seq_len(min(include_top, length(nearest)))]
    }
    remaining <- setdiff(valid, chosen)
    need <- keep_k - length(chosen)
    if (need > 0L && length(remaining) > 0L) {
      prob <- membership[remaining]
      if (!any(is.finite(prob) & prob > 0)) prob <- rep(1, length(remaining))
      prob[!is.finite(prob) | prob < 0] <- 0
      draw <- if (length(remaining) <= need) {
        remaining
      } else {
        remaining[sample.int(length(remaining), size = need, replace = FALSE, prob = prob)]
      }
      chosen <- c(chosen, draw)
    }
    if (length(chosen) < keep_k) {
      fill <- nearest[!nearest %in% chosen]
      if (length(fill) > 0L) {
        chosen <- c(chosen, fill[seq_len(min(keep_k - length(chosen), length(fill)))])
      }
    }
    chosen <- chosen[seq_len(min(keep_k, length(chosen)))]
    chosen <- chosen[order(dst[chosen], chosen)]
    out_indices[i, offset + seq_along(chosen)] <- idx[chosen]
    out_distances[i, offset + seq_along(chosen)] <- dst[chosen]
    chosen_membership <- membership[chosen]
    chosen_membership <- chosen_membership[is.finite(chosen_membership)]
    if (length(chosen_membership) > 0L) {
      selected_weight_sum <- selected_weight_sum + sum(chosen_membership)
      selected_weight_count <- selected_weight_count + length(chosen_membership)
    }
  }

  selected_weight_mean <- if (selected_weight_count > 0L) selected_weight_sum / selected_weight_count else NA_real_
  all_weight_mean <- if (all_weight_count > 0L) all_weight_sum / all_weight_count else NA_real_

  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("weighted_edge_sample_", round(100 * keep_fraction)),
      exact = FALSE
    ),
    graph_approximation = paste0("weighted_edge_sample_", round(100 * keep_fraction)),
    graph_effective_k = keep_k,
    graph_edge_retention = keep_k / base_k,
    graph_mean_degree = keep_k,
    graph_min_degree = keep_k,
    graph_max_degree = keep_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = mean(!is.finite(out_distances[, seq.int(offset + 1L, out_cols), drop = FALSE])),
    graph_edge_sampling_method = "umap_fuzzy_weighted_without_replacement",
    graph_edge_sampling_fraction = keep_fraction,
    graph_edge_sampling_weight_power = weight_power,
    graph_edge_sampling_include_top = include_top,
    graph_edge_sampling_target_scale = as.numeric(target_scale),
    graph_edge_sampling_mean_selected_weight = selected_weight_mean,
    graph_edge_sampling_mean_candidate_weight = all_weight_mean,
    graph_edge_sampling_selected_to_candidate_weight_ratio =
      selected_weight_mean / max(all_weight_mean, .Machine$double.eps)
  )
}

graph_effective_resistance_sparsify_knn <- function(knn, keep_fraction) {
  keep_fraction <- as.numeric(keep_fraction)
  if (!is.finite(keep_fraction) || keep_fraction <= 0 || keep_fraction >= 1) {
    stop("Effective-resistance sparsification requires 0 < keep_fraction < 1.", call. = FALSE)
  }
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  if (n < 3L || base_k < 2L) {
    stop("Effective-resistance sparsification requires at least three rows and two non-self neighbours.", call. = FALSE)
  }
  keep_k <- max(1L, min(base_k - 1L, ceiling(base_k * keep_fraction)))
  row_id <- rep(seq_len(n), each = base_k)
  neigh <- as.integer(as.vector(t(knn$indices[, cols, drop = FALSE])))
  dist <- as.numeric(as.vector(t(knn$distances[, cols, drop = FALSE])))
  valid <- is.finite(dist) & is.finite(neigh) & neigh >= 1L & neigh <= n & neigh != row_id
  if (sum(valid) < n) {
    stop("Effective-resistance sparsification found too few valid KNN edges.", call. = FALSE)
  }
  row_valid <- row_id[valid]
  neigh_valid <- neigh[valid]
  dist_valid <- dist[valid]
  a <- pmin(row_valid, neigh_valid)
  b <- pmax(row_valid, neigh_valid)
  edge_key <- paste(a, b, sep = ":")
  ord <- order(edge_key, dist_valid)
  first <- !duplicated(edge_key[ord])
  a_u <- a[ord][first]
  b_u <- b[ord][first]
  dist_u <- dist_valid[ord][first]
  scale <- stats::median(dist_u[is.finite(dist_u) & dist_u > 0], na.rm = TRUE)
  if (!is.finite(scale) || scale <= 0) scale <- 1
  weights <- exp(-pmin((dist_u / scale)^2, 50))
  weights[!is.finite(weights) | weights <= 0] <- .Machine$double.eps

  t0 <- proc.time()[["elapsed"]]
  wmat <- matrix(0, n, n)
  wmat[cbind(a_u, b_u)] <- weights
  wmat[cbind(b_u, a_u)] <- weights
  lap <- diag(rowSums(wmat), n, n) - wmat
  eig <- eigen(lap, symmetric = TRUE)
  tol <- max(n, 1L) * .Machine$double.eps * max(abs(eig$values), na.rm = TRUE)
  positive <- eig$values > tol
  rank <- sum(positive)
  if (rank < 1L) {
    stop("Effective-resistance sparsification could not find a non-zero Laplacian spectrum.", call. = FALSE)
  }
  vec <- eig$vectors[, positive, drop = FALSE]
  linv <- vec %*% (t(vec) / eig$values[positive])
  diag_linv <- diag(linv)
  resistance <- diag_linv[a_u] + diag_linv[b_u] - 2 * linv[cbind(a_u, b_u)]
  resistance[!is.finite(resistance) | resistance < 0] <- 0
  leverage <- weights * resistance
  spectral_time <- proc.time()[["elapsed"]] - t0
  leverage_key <- paste(a_u, b_u, sep = ":")
  leverage_by_key <- stats::setNames(leverage, leverage_key)

  out_cols <- keep_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  for (i in seq_len(n)) {
    idx <- as.integer(knn$indices[i, cols])
    dst <- as.numeric(knn$distances[i, cols])
    valid_i <- which(is.finite(dst) & is.finite(idx) & idx >= 1L & idx <= n & idx != i)
    if (length(valid_i) == 0L) valid_i <- seq_len(base_k)
    key_i <- paste(pmin(i, idx[valid_i]), pmax(i, idx[valid_i]), sep = ":")
    score_i <- as.numeric(leverage_by_key[key_i])
    score_i[!is.finite(score_i)] <- -Inf
    chosen_pos <- valid_i[order(-score_i, dst[valid_i], valid_i)]
    chosen_pos <- chosen_pos[seq_len(min(keep_k, length(chosen_pos)))]
    chosen_pos <- chosen_pos[order(dst[chosen_pos], chosen_pos)]
    out_indices[i, offset + seq_along(chosen_pos)] <- idx[chosen_pos]
    out_distances[i, offset + seq_along(chosen_pos)] <- dst[chosen_pos]
  }
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("effective_resistance_", round(100 * keep_fraction)),
      exact = FALSE
    ),
    graph_approximation = paste0("effective_resistance_sparsify_", round(100 * keep_fraction)),
    graph_effective_k = keep_k,
    graph_edge_retention = keep_k / base_k,
    graph_recall_at_k = keep_k / base_k,
    graph_mean_degree = keep_k,
    graph_min_degree = keep_k,
    graph_max_degree = keep_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_sparsification_method = "effective_resistance",
    graph_sparsification_keep_fraction = keep_fraction,
    graph_sparsification_target_k = keep_k,
    graph_sparsification_undirected_edges = length(weights),
    graph_sparsification_spectral_rank = rank,
    graph_sparsification_spectral_time_sec = spectral_time,
    graph_sparsification_leverage_mean = safe_mean(leverage),
    graph_sparsification_leverage_min = if (length(leverage) == 0L) NA_real_ else min(leverage, na.rm = TRUE),
    graph_sparsification_leverage_max = if (length(leverage) == 0L) NA_real_ else max(leverage, na.rm = TRUE),
    graph_sparsification_resistance_mean = safe_mean(resistance),
    graph_sparsification_weight_mean = safe_mean(weights)
  )
}

graph_rank_distances <- function(knn) {
  distances <- knn$distances
  cols <- knn_neighbor_cols(knn)
  if (length(cols) > 0L) {
    ranks <- seq_along(cols) / max(1L, length(cols))
    distances[, cols] <- matrix(rep(ranks, each = nrow(distances)), nrow = nrow(distances))
  }
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  list(
    knn = graph_knn_result(knn, knn$indices, distances, "rank_distances", exact = FALSE),
    graph_approximation = "rank_distances",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1
  )
}

graph_binary_distances <- function(knn) {
  distances <- knn$distances
  cols <- knn_neighbor_cols(knn)
  if (length(cols) > 0L) distances[, cols] <- 1
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  list(
    knn = graph_knn_result(knn, knn$indices, distances, "binary_distances", exact = FALSE),
    graph_approximation = "binary_distances",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1
  )
}

graph_local_scale_distances <- function(knn) {
  distances <- knn$distances
  cols <- knn_neighbor_cols(knn)
  if (length(cols) > 0L) {
    neighbor_dist <- distances[, cols, drop = FALSE]
    scale <- apply(neighbor_dist, 1L, function(row) {
      row <- row[is.finite(row) & row > 0]
      if (length(row) == 0L) 1 else stats::median(row)
    })
    scale[!is.finite(scale) | scale <= 0] <- 1
    distances[, cols] <- sweep(neighbor_dist, 1L, scale, "/")
  }
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  list(
    knn = graph_knn_result(knn, knn$indices, distances, "local_scale", exact = FALSE),
    graph_approximation = "local_scale",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1
  )
}

tsne_affinity_perplexities <- function(k, mode = c("auto", "multiscale_auto")) {
  mode <- match.arg(mode)
  k <- as.integer(k)
  cap <- max(1L, floor(k / 3L))
  if (cap < 2L) {
    return(integer())
  }
  if (identical(mode, "auto")) {
    return(as.integer(max(2L, min(30L, cap))))
  }
  unique(as.integer(c(
    max(2L, min(10L, floor(k / 9L))),
    max(2L, min(30L, cap))
  )))
}

tsne_conditional_affinity <- function(distances, perplexity, tol = 1e-5, max_iter = 50L) {
  distances <- as.numeric(distances)
  valid <- is.finite(distances) & distances >= 0
  out <- rep(0, length(distances))
  if (!any(valid)) {
    return(list(prob = out, entropy = NA_real_, perplexity = NA_real_, sigma = NA_real_))
  }
  d2 <- distances[valid]^2
  m <- length(d2)
  if (m < 2L || all(d2 <= .Machine$double.eps)) {
    out[valid] <- 1 / m
    return(list(prob = out, entropy = log(m), perplexity = m, sigma = Inf))
  }
  target_entropy <- log(max(1, min(as.numeric(perplexity), m)))
  beta <- 1
  beta_min <- -Inf
  beta_max <- Inf
  entropy <- NA_real_
  prob <- rep(1 / m, m)
  for (iter in seq_len(max_iter)) {
    weights <- exp(-d2 * beta)
    weights[!is.finite(weights)] <- 0
    sum_weights <- sum(weights)
    if (sum_weights <= 0) {
      weights <- rep(1, m)
      sum_weights <- m
    }
    prob <- weights / sum_weights
    entropy <- log(sum_weights) + beta * sum(d2 * weights) / sum_weights
    diff <- entropy - target_entropy
    if (abs(diff) <= tol) break
    if (diff > 0) {
      beta_min <- beta
      beta <- if (is.infinite(beta_max)) beta * 2 else (beta + beta_max) / 2
    } else {
      beta_max <- beta
      beta <- if (is.infinite(beta_min)) beta / 2 else (beta + beta_min) / 2
    }
    beta <- max(beta, .Machine$double.eps)
  }
  out[valid] <- prob
  list(
    prob = out,
    entropy = entropy,
    perplexity = exp(entropy),
    sigma = sqrt(1 / (2 * max(beta, .Machine$double.eps)))
  )
}

graph_tsne_affinity_knn <- function(knn,
                                    perplexities,
                                    temperature = 1,
                                    mode = "perplexity") {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  perplexities <- sort(unique(as.integer(perplexities)))
  perplexities <- perplexities[perplexities >= 1L]
  if (base_k < 3L || length(perplexities) == 0L) {
    stop("t-SNE affinity graph requires at least three non-self neighbours and one valid perplexity.", call. = FALSE)
  }
  if (max(perplexities) * 3L > base_k) {
    stop(
      "t-SNE affinity graph with perplexity ", max(perplexities),
      " requires --k >= ", max(perplexities) * 3L, ".",
      call. = FALSE
    )
  }
  temperature <- as.numeric(temperature)
  if (!is.finite(temperature) || temperature <= 0) {
    stop("t-SNE affinity temperature must be positive.", call. = FALSE)
  }

  prob_sum <- matrix(0, n, base_k)
  entropies <- numeric(0)
  effective_perplexities <- numeric(0)
  sigmas <- numeric(0)
  for (perplexity in perplexities) {
    for (i in seq_len(n)) {
      fit <- tsne_conditional_affinity(knn$distances[i, cols], perplexity)
      prob_sum[i, ] <- prob_sum[i, ] + fit$prob
      entropies <- c(entropies, fit$entropy)
      effective_perplexities <- c(effective_perplexities, fit$perplexity)
      sigmas <- c(sigmas, fit$sigma)
    }
  }
  prob <- prob_sum / length(perplexities)
  if (!identical(temperature, 1)) {
    prob <- prob^temperature
    row_sum <- rowSums(prob)
    row_sum[!is.finite(row_sum) | row_sum <= 0] <- 1
    prob <- sweep(prob, 1L, row_sum, "/")
  }
  positive_prob <- prob[is.finite(prob) & prob > 0]
  floor_prob <- if (length(positive_prob) == 0L) .Machine$double.eps else max(.Machine$double.eps, min(positive_prob) * 0.5)
  affinity_distances <- -log(pmax(prob, floor_prob))
  out_indices <- knn$indices[, cols, drop = FALSE]
  out_distances <- affinity_distances
  for (i in seq_len(n)) {
    ord <- order(out_distances[i, ], out_indices[i, ])
    out_indices[i, ] <- out_indices[i, ord]
    out_distances[i, ] <- out_distances[i, ord]
  }
  if (self) {
    out_indices <- cbind(knn$indices[, 1L], out_indices)
    out_distances <- cbind(knn$distances[, 1L], out_distances)
  }
  if (self) out_distances[, 1L] <- 0
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("tsne_affinity_", mode),
      exact = FALSE
    ),
    graph_approximation = paste0("tsne_affinity_", mode),
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = 1,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_tsne_affinity_mode = mode,
    graph_tsne_affinity_perplexities = paste(perplexities, collapse = ","),
    graph_tsne_affinity_num_scales = length(perplexities),
    graph_tsne_affinity_temperature = temperature,
    graph_tsne_affinity_entropy_mean = safe_mean(entropies),
    graph_tsne_affinity_effective_perplexity_mean = safe_mean(effective_perplexities),
    graph_tsne_affinity_sigma_mean = safe_mean(sigmas),
    graph_tsne_affinity_sigma_min = if (length(sigmas) == 0L) NA_real_ else min(sigmas, na.rm = TRUE),
    graph_tsne_affinity_sigma_max = if (length(sigmas) == 0L) NA_real_ else max(sigmas, na.rm = TRUE),
    graph_tsne_affinity_prob_min = if (length(positive_prob) == 0L) NA_real_ else min(positive_prob, na.rm = TRUE),
    graph_tsne_affinity_prob_max = if (length(positive_prob) == 0L) NA_real_ else max(positive_prob, na.rm = TRUE)
  )
}

graph_mutual_fill_knn <- function(knn, keep_k, approximation) {
  keep_k <- max(1L, min(as.integer(keep_k), knn_effective_k(knn)))
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    unique(x[is.finite(x) & x >= 1L & x <= n])
  })
  out_cols <- keep_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  for (i in seq_len(n)) {
    neighbors <- knn$indices[i, cols]
    valid <- which(is.finite(neighbors) & neighbors >= 1L & neighbors <= n)
    reciprocal <- valid[vapply(valid, function(pos) i %in% neighbor_sets[[neighbors[pos]]], logical(1L))]
    fill <- valid[!valid %in% reciprocal]
    chosen <- c(reciprocal, fill)
    if (length(chosen) > keep_k) chosen <- chosen[seq_len(keep_k)]
    out_indices[i, offset + seq_along(chosen)] <- knn$indices[i, cols[chosen]]
    out_distances[i, offset + seq_along(chosen)] <- knn$distances[i, cols[chosen]]
  }
  list(
    knn = graph_knn_result(knn, out_indices, out_distances, approximation, exact = FALSE),
    graph_approximation = approximation,
    graph_effective_k = keep_k,
    graph_edge_retention = keep_k / max(1L, knn_effective_k(knn))
  )
}

graph_mutual_only_knn <- function(knn) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    unique(x[is.finite(x) & x >= 1L & x <= n])
  })
  reciprocal_positions <- vector("list", n)
  degrees <- integer(n)
  for (i in seq_len(n)) {
    neighbors <- knn$indices[i, cols]
    valid <- which(is.finite(neighbors) & neighbors >= 1L & neighbors <= n)
    reciprocal_positions[[i]] <- valid[
      vapply(valid, function(pos) i %in% neighbor_sets[[neighbors[pos]]], logical(1L))
    ]
    degrees[i] <- length(reciprocal_positions[[i]])
  }
  out_k <- max(1L, max(degrees))
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  finite_dist <- knn$distances[is.finite(knn$distances) & knn$distances > 0]
  pad_distance <- if (length(finite_dist) == 0L) 1e6 else max(finite_dist) * 1e3 + 1
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  for (i in seq_len(n)) {
    chosen <- reciprocal_positions[[i]]
    if (length(chosen) == 0L) {
      valid <- which(is.finite(knn$indices[i, cols]) & knn$indices[i, cols] >= 1L & knn$indices[i, cols] <= n)
      fallback_pos <- if (length(valid) > 0L) valid[1L] else NA_integer_
      fallback_index <- if (is.na(fallback_pos)) {
        if (i == 1L) 2L else 1L
      } else {
        knn$indices[i, cols[fallback_pos]]
      }
      out_indices[i, offset + seq_len(out_k)] <- fallback_index
      out_distances[i, offset + seq_len(out_k)] <- pad_distance
      padding_count <- padding_count + out_k
      next
    }
    out_indices[i, offset + seq_along(chosen)] <- knn$indices[i, cols[chosen]]
    out_distances[i, offset + seq_along(chosen)] <- knn$distances[i, cols[chosen]]
    if (length(chosen) < out_k) {
      pad_cols <- offset + seq.int(length(chosen) + 1L, out_k)
      out_indices[i, pad_cols] <- knn$indices[i, cols[chosen[1L]]]
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  list(
    knn = graph_knn_result(knn, out_indices, out_distances, "mutual_only", exact = FALSE),
    graph_approximation = "mutual_only",
    graph_effective_k = out_k,
    graph_edge_retention = mean(degrees) / base_k,
    graph_recall_at_k = mean(degrees) / base_k,
    graph_mean_degree = mean(degrees),
    graph_min_degree = min(degrees),
    graph_max_degree = max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k)
  )
}

graph_symmetric_union_knn <- function(knn) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  incoming_indices <- vector("list", n)
  incoming_distances <- vector("list", n)
  outgoing_indices <- vector("list", n)
  outgoing_distances <- vector("list", n)
  for (i in seq_len(n)) {
    row_idx <- knn$indices[i, cols]
    row_dst <- knn$distances[i, cols]
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & is.finite(row_dst))
    outgoing_indices[[i]] <- as.integer(row_idx[valid])
    outgoing_distances[[i]] <- as.numeric(row_dst[valid])
    for (pos in valid) {
      j <- as.integer(row_idx[pos])
      incoming_indices[[j]] <- c(incoming_indices[[j]], i)
      incoming_distances[[j]] <- c(incoming_distances[[j]], as.numeric(row_dst[pos]))
    }
  }
  union_indices <- vector("list", n)
  union_distances <- vector("list", n)
  degrees <- integer(n)
  for (i in seq_len(n)) {
    idx <- outgoing_indices[[i]]
    dst <- outgoing_distances[[i]]
    inc_idx <- incoming_indices[[i]]
    inc_dst <- incoming_distances[[i]]
    if (length(inc_idx) > 0L) {
      for (p in seq_along(inc_idx)) {
        hit <- match(inc_idx[p], idx)
        if (is.na(hit)) {
          idx <- c(idx, inc_idx[p])
          dst <- c(dst, inc_dst[p])
        } else {
          dst[hit] <- min(dst[hit], inc_dst[p])
        }
      }
    }
    if (length(idx) > 1L) {
      ord <- order(dst, idx)
      idx <- idx[ord]
      dst <- dst[ord]
    }
    union_indices[[i]] <- idx
    union_distances[[i]] <- dst
    degrees[i] <- length(idx)
  }
  out_k <- max(1L, max(degrees))
  if (n * out_k > max(2000000L, n * base_k * 8L)) {
    stop(
      "Symmetric union graph would expand to ", out_k,
      " columns; skip or run on a smaller benchmark subset.",
      call. = FALSE
    )
  }
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  finite_dist <- knn$distances[is.finite(knn$distances) & knn$distances > 0]
  pad_distance <- if (length(finite_dist) == 0L) 1e6 else max(finite_dist) * 1e3 + 1
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  for (i in seq_len(n)) {
    idx <- union_indices[[i]]
    dst <- union_distances[[i]]
    if (length(idx) == 0L) {
      idx <- if (i == 1L) 2L else 1L
      dst <- pad_distance
    }
    out_indices[i, offset + seq_along(idx)] <- idx
    out_distances[i, offset + seq_along(dst)] <- dst
    if (length(idx) < out_k) {
      pad_cols <- offset + seq.int(length(idx) + 1L, out_k)
      out_indices[i, pad_cols] <- idx[1L]
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  list(
    knn = graph_knn_result(knn, out_indices, out_distances, "symmetric_union", exact = FALSE),
    graph_approximation = "symmetric_union",
    graph_effective_k = out_k,
    graph_edge_retention = mean(degrees) / base_k,
    graph_mean_degree = mean(degrees),
    graph_min_degree = min(degrees),
    graph_max_degree = max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k)
  )
}

graph_jaccard_weighted_knn <- function(knn) {
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  if (length(cols) == 0L) return(graph_rank_distances(knn))
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    sort(unique(x[is.finite(x) & x >= 1L & x <= n]))
  })
  distances <- knn$distances
  jaccard <- numeric(n * length(cols))
  write_at <- 0L
  eps <- 0.05
  for (i in seq_len(n)) {
    set_i <- neighbor_sets[[i]]
    for (rank in seq_along(cols)) {
      j <- knn$indices[i, cols[rank]]
      if (!is.finite(j) || j < 1L || j > n) next
      set_j <- neighbor_sets[[j]]
      union_count <- length(union(set_i, set_j))
      overlap <- if (union_count == 0L) 0 else length(intersect(set_i, set_j)) / union_count
      write_at <- write_at + 1L
      jaccard[write_at] <- overlap
      distances[i, cols[rank]] <- distances[i, cols[rank]] * (1 - overlap + eps)
    }
  }
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  jaccard <- jaccard[seq_len(write_at)]
  list(
    knn = graph_knn_result(knn, knn$indices, distances, "jaccard_weighted", exact = FALSE),
    graph_approximation = "jaccard_weighted",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1,
    graph_mean_jaccard = safe_mean(jaccard),
    graph_min_jaccard = if (length(jaccard) == 0L) NA_real_ else min(jaccard),
    graph_max_jaccard = if (length(jaccard) == 0L) NA_real_ else max(jaccard),
    graph_zero_jaccard_fraction = if (length(jaccard) == 0L) NA_real_ else mean(jaccard <= 0)
  )
}

graph_snn_reweight_knn <- function(knn) {
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  if (length(cols) == 0L) return(graph_rank_distances(knn))
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    sort(unique(x[is.finite(x) & x >= 1L & x <= n]))
  })
  distances <- knn$distances
  denom <- max(1L, length(cols))
  eps <- 1e-6
  for (i in seq_len(n)) {
    for (rank in seq_along(cols)) {
      j <- knn$indices[i, cols[rank]]
      if (!is.finite(j) || j < 1L || j > n) next
      overlap <- length(intersect(neighbor_sets[[i]], neighbor_sets[[j]])) / denom
      distances[i, cols[rank]] <- max(eps, 1 - overlap) + rank * eps
    }
  }
  if (knn_has_self_first(knn)) distances[, 1L] <- 0
  list(
    knn = graph_knn_result(knn, knn$indices, distances, "snn_reweight", exact = FALSE),
    graph_approximation = "snn_reweight",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1
  )
}

graph_localmap_false_neighbor_correct_knn <- function(knn,
                                                      jaccard_threshold = 0.05,
                                                      distance_quantile = 0.75,
                                                      distance_multiplier = 1.15,
                                                      min_keep_fraction = 0.50,
                                                      method = "localmap") {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  if (base_k < 4L) {
    stop("LocalMAP false-neighbour correction requires at least four non-self neighbours.", call. = FALSE)
  }
  jaccard_threshold <- max(0, min(1, as.numeric(jaccard_threshold)))
  distance_quantile <- max(0.50, min(0.95, as.numeric(distance_quantile)))
  distance_multiplier <- max(1.0, as.numeric(distance_multiplier))
  min_keep_fraction <- max(0.05, min(1, as.numeric(min_keep_fraction)))
  min_keep_k <- max(1L, min(base_k, as.integer(ceiling(base_k * min_keep_fraction))))

  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    sort(unique(as.integer(x[is.finite(x) & x >= 1L & x <= n & x != i])))
  })
  kept_positions <- vector("list", n)
  degrees <- integer(n)
  removed_counts <- integer(n)
  kept_jaccard <- numeric(0)
  removed_jaccard <- numeric(0)
  kept_distance_ratio <- numeric(0)
  removed_distance_ratio <- numeric(0)
  thresholds <- numeric(n)

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i & is.finite(row_dst))
    if (length(valid) == 0L) {
      kept_positions[[i]] <- integer(0)
      next
    }
    positive_dist <- row_dst[valid]
    positive_dist <- positive_dist[is.finite(positive_dist) & positive_dist > 0]
    row_scale <- if (length(positive_dist) == 0L) {
      max(1, safe_number(row_dst[valid], 1))
    } else {
      stats::quantile(positive_dist, distance_quantile, names = FALSE, type = 7)
    }
    if (!is.finite(row_scale) || row_scale <= 0) row_scale <- max(positive_dist, 1, na.rm = TRUE)
    threshold <- row_scale * distance_multiplier
    thresholds[i] <- threshold
    set_i <- neighbor_sets[[i]]
    jac <- numeric(length(valid))
    for (vv in seq_along(valid)) {
      j <- row_idx[valid[vv]]
      set_j <- neighbor_sets[[j]]
      union_count <- length(union(set_i, set_j))
      jac[vv] <- if (union_count == 0L) 0 else length(intersect(set_i, set_j)) / union_count
    }
    rank_position <- match(valid, seq_along(cols))
    suspicious <- row_dst[valid] > threshold & jac <= jaccard_threshold & rank_position > min_keep_k
    keep <- valid[!suspicious]
    protected <- valid[seq_len(min(min_keep_k, length(valid)))]
    keep <- unique(c(protected, keep))
    keep <- keep[order(match(keep, valid))]
    if (length(keep) == 0L) keep <- valid[1L]
    kept_positions[[i]] <- keep
    degrees[i] <- length(keep)
    removed_counts[i] <- length(valid) - length(keep)
    kept_match <- match(keep, valid)
    removed_match <- which(!(valid %in% keep))
    if (length(kept_match) > 0L) {
      kept_jaccard <- c(kept_jaccard, jac[kept_match])
      kept_distance_ratio <- c(kept_distance_ratio, row_dst[keep] / threshold)
    }
    if (length(removed_match) > 0L) {
      removed_jaccard <- c(removed_jaccard, jac[removed_match])
      removed_distance_ratio <- c(removed_distance_ratio, row_dst[valid[removed_match]] / threshold)
    }
  }

  out_k <- max(1L, max(degrees, na.rm = TRUE))
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  finite_dist <- knn$distances[is.finite(knn$distances) & knn$distances > 0]
  pad_distance <- if (length(finite_dist) == 0L) 1e6 else max(finite_dist) * 1e3 + 1
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  for (i in seq_len(n)) {
    keep <- kept_positions[[i]]
    if (length(keep) == 0L) {
      fallback <- if (i == 1L) 2L else 1L
      out_indices[i, offset + seq_len(out_k)] <- fallback
      out_distances[i, offset + seq_len(out_k)] <- pad_distance
      padding_count <- padding_count + out_k
      next
    }
    out_indices[i, offset + seq_along(keep)] <- knn$indices[i, cols[keep]]
    out_distances[i, offset + seq_along(keep)] <- knn$distances[i, cols[keep]]
    if (length(keep) < out_k) {
      pad_cols <- offset + seq.int(length(keep) + 1L, out_k)
      out_indices[i, pad_cols] <- knn$indices[i, cols[keep[1L]]]
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  transfer_mode <- switch(
    safe_character(method, "localmap"),
    umap = "umap_fuzzy_graph_false_neighbour_correction",
    tsne = "tsne_affinity_false_neighbour_filtering",
    pacmap = "pacmap_pair_false_neighbour_filtering",
    trimap = "trimap_inlier_false_neighbour_filtering",
    localmap = "localmap_native_false_neighbour_correction",
    "localmap_false_neighbour_transfer"
  )
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "localmap_false_neighbour_j",
        gsub("\\.", "p", format(jaccard_threshold, trim = TRUE, scientific = FALSE)),
        "_q",
        gsub("\\.", "p", format(distance_quantile, trim = TRUE, scientific = FALSE)),
        "_m",
        gsub("\\.", "p", format(distance_multiplier, trim = TRUE, scientific = FALSE))
      ),
      exact = FALSE
    ),
    graph_approximation = "localmap_false_neighbour_correction",
    graph_effective_k = out_k,
    graph_edge_retention = safe_mean(degrees / base_k),
    graph_recall_at_k = safe_mean(degrees / base_k),
    graph_mean_degree = safe_mean(degrees),
    graph_min_degree = if (length(degrees) == 0L) NA_real_ else min(degrees),
    graph_max_degree = if (length(degrees) == 0L) NA_real_ else max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k),
    graph_mean_jaccard = safe_mean(kept_jaccard),
    graph_min_jaccard = if (length(kept_jaccard) == 0L) NA_real_ else min(kept_jaccard, na.rm = TRUE),
    graph_max_jaccard = if (length(kept_jaccard) == 0L) NA_real_ else max(kept_jaccard, na.rm = TRUE),
    graph_zero_jaccard_fraction = if (length(kept_jaccard) == 0L) NA_real_ else mean(kept_jaccard <= 0),
    localmap_false_neighbor_enabled = TRUE,
    localmap_false_neighbor_mode = "distance_and_shared_neighbour_support",
    localmap_false_neighbor_transfer_mode = transfer_mode,
    localmap_false_neighbor_jaccard_threshold = jaccard_threshold,
    localmap_false_neighbor_distance_quantile = distance_quantile,
    localmap_false_neighbor_distance_multiplier = distance_multiplier,
    localmap_false_neighbor_min_keep_fraction = min_keep_fraction,
    localmap_false_neighbor_min_keep_k = min_keep_k,
    localmap_false_neighbor_removed_edges_mean = safe_mean(removed_counts),
    localmap_false_neighbor_removed_fraction = safe_mean(removed_counts / base_k),
    localmap_false_neighbor_kept_degree_mean = safe_mean(degrees),
    localmap_false_neighbor_kept_jaccard_mean = safe_mean(kept_jaccard),
    localmap_false_neighbor_removed_jaccard_mean = safe_mean(removed_jaccard),
    localmap_false_neighbor_kept_distance_ratio_mean = safe_mean(kept_distance_ratio),
    localmap_false_neighbor_removed_distance_ratio_mean = safe_mean(removed_distance_ratio),
    localmap_false_neighbor_threshold_mean = safe_mean(thresholds)
  )
}

localmap_local_weight_transfer_mode <- function(method) {
  switch(
    safe_character(method, "localmap"),
    umap = "umap_local_edge_loss_reweighting",
    tsne = "tsne_local_affinity_loss_reweighting",
    pacmap = "pacmap_near_pair_loss_reweighting",
    trimap = "trimap_inlier_loss_reweighting",
    localmap = "localmap_native_local_loss_reweighting",
    "localmap_local_loss_reweighting_transfer"
  )
}

graph_localmap_local_loss_reweight_knn <- function(knn,
                                                   local_weight = 1,
                                                   jaccard_blend = 0.35,
                                                   method = "localmap") {
  n <- nrow(knn$indices)
  cols <- knn_neighbor_cols(knn)
  base_k <- length(cols)
  if (base_k < 2L) {
    stop("LocalMAP local-loss reweighting requires at least two non-self neighbours.", call. = FALSE)
  }
  local_weight <- as.numeric(local_weight)
  if (!is.finite(local_weight) || local_weight <= 0) {
    stop("local_weight must be a positive finite number.", call. = FALSE)
  }
  local_weight <- max(0.05, min(20, local_weight))
  jaccard_blend <- max(0, min(1, as.numeric(jaccard_blend)))

  out_indices <- knn$indices
  out_distances <- knn$distances
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    sort(unique(as.integer(x[is.finite(x) & x >= 1L & x <= n & x != i])))
  })
  rank_template <- if (base_k == 1L) {
    1
  } else {
    1 - (seq_len(base_k) - 1) / max(1L, base_k - 1L)
  }
  trust_scores <- numeric(0)
  rank_scores <- numeric(0)
  jaccard_scores <- numeric(0)
  multipliers <- numeric(0)
  distance_ratios <- numeric(0)

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i & is.finite(row_dst))
    if (length(valid) == 0L) next
    set_i <- neighbor_sets[[i]]
    jac <- numeric(length(valid))
    for (vv in seq_along(valid)) {
      j <- row_idx[valid[vv]]
      set_j <- neighbor_sets[[j]]
      union_count <- length(union(set_i, set_j))
      jac[vv] <- if (union_count == 0L) 0 else length(intersect(set_i, set_j)) / union_count
    }
    rank_score <- rank_template[valid]
    trust <- (1 - jaccard_blend) * rank_score + jaccard_blend * jac
    trust <- pmin(1, pmax(0, trust))
    mult <- exp(log(local_weight) * trust)
    adjusted <- pmax(.Machine$double.eps, row_dst[valid] / pmax(mult, .Machine$double.eps))
    out_distances[i, cols[valid]] <- adjusted

    row_dist <- out_distances[i, cols]
    valid_order <- valid[order(row_dist[valid], seq_along(valid), na.last = TRUE)]
    invalid <- setdiff(seq_along(cols), valid)
    row_order <- c(valid_order, invalid)
    out_indices[i, cols] <- out_indices[i, cols[row_order]]
    out_distances[i, cols] <- out_distances[i, cols[row_order]]

    trust_scores <- c(trust_scores, trust)
    rank_scores <- c(rank_scores, rank_score)
    jaccard_scores <- c(jaccard_scores, jac)
    multipliers <- c(multipliers, mult)
    distance_ratios <- c(distance_ratios, adjusted / pmax(row_dst[valid], .Machine$double.eps))
  }
  if (knn_has_self_first(knn)) out_distances[, 1L] <- 0

  label <- gsub("\\.", "p", format(local_weight, trim = TRUE, scientific = FALSE))
  list(
    knn = graph_knn_result(knn, out_indices, out_distances, paste0("localmap_local_weight_", label), exact = FALSE),
    graph_approximation = "localmap_local_loss_reweighting",
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = 1,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_mean_jaccard = safe_mean(jaccard_scores),
    graph_min_jaccard = if (length(jaccard_scores) == 0L) NA_real_ else min(jaccard_scores, na.rm = TRUE),
    graph_max_jaccard = if (length(jaccard_scores) == 0L) NA_real_ else max(jaccard_scores, na.rm = TRUE),
    graph_zero_jaccard_fraction = if (length(jaccard_scores) == 0L) NA_real_ else mean(jaccard_scores <= 0),
    localmap_local_weight_enabled = TRUE,
    localmap_local_weight = local_weight,
    localmap_local_weight_mode = "rank_and_shared_neighbour_trust",
    localmap_local_weight_transfer_mode = localmap_local_weight_transfer_mode(method),
    localmap_local_weight_jaccard_blend = jaccard_blend,
    localmap_local_weight_mean_trust = safe_mean(trust_scores),
    localmap_local_weight_rank_component_mean = safe_mean(rank_scores),
    localmap_local_weight_jaccard_component_mean = safe_mean(jaccard_scores),
    localmap_local_weight_mean_multiplier = safe_mean(multipliers),
    localmap_local_weight_min_multiplier = if (length(multipliers) == 0L) NA_real_ else min(multipliers, na.rm = TRUE),
    localmap_local_weight_max_multiplier = if (length(multipliers) == 0L) NA_real_ else max(multipliers, na.rm = TRUE),
    localmap_local_weight_distance_scale_mean = safe_mean(distance_ratios)
  )
}

graph_snn_weighted_knn <- function(knn, snn_k, prune_threshold) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  snn_k <- as.integer(snn_k)
  prune_threshold <- as.numeric(prune_threshold)
  if (length(cols) < snn_k) {
    stop("SNN_k=", snn_k, " requires at least ", snn_k, " KNN columns.", call. = FALSE)
  }
  snn_cols <- cols[seq_len(snn_k)]
  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, snn_cols]
    sort(unique(x[is.finite(x) & x >= 1L & x <= n]))
  })
  kept_positions <- vector("list", n)
  kept_weights <- vector("list", n)
  degrees <- integer(n)
  weights_all <- numeric(n * snn_k)
  write_at <- 0L
  eps <- 1e-6
  for (i in seq_len(n)) {
    set_i <- neighbor_sets[[i]]
    row_positions <- integer(0L)
    row_weights <- numeric(0L)
    for (rank in seq_len(snn_k)) {
      j <- knn$indices[i, snn_cols[rank]]
      if (!is.finite(j) || j < 1L || j > n) next
      overlap <- length(intersect(set_i, neighbor_sets[[j]]))
      weight <- overlap / snn_k
      write_at <- write_at + 1L
      weights_all[write_at] <- weight
      if (weight >= prune_threshold) {
        row_positions <- c(row_positions, rank)
        row_weights <- c(row_weights, weight)
      }
    }
    kept_positions[[i]] <- row_positions
    kept_weights[[i]] <- row_weights
    degrees[i] <- length(row_positions)
  }
  out_k <- max(1L, max(degrees))
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  pad_distance <- 1 + prune_threshold + 100 * eps
  for (i in seq_len(n)) {
    chosen <- kept_positions[[i]]
    chosen_weights <- kept_weights[[i]]
    if (length(chosen) == 0L) {
      out_indices[i, offset + seq_len(out_k)] <- knn$indices[i, snn_cols[1L]]
      out_distances[i, offset + seq_len(out_k)] <- pad_distance
      padding_count <- padding_count + out_k
      next
    }
    out_indices[i, offset + seq_along(chosen)] <- knn$indices[i, snn_cols[chosen]]
    out_distances[i, offset + seq_along(chosen)] <- pmax(eps, 1 - chosen_weights) +
      seq_along(chosen) * eps
    if (length(chosen) < out_k) {
      pad_cols <- offset + seq.int(length(chosen) + 1L, out_k)
      out_indices[i, pad_cols] <- knn$indices[i, snn_cols[chosen[1L]]]
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  weights_all <- weights_all[seq_len(write_at)]
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("snn_k", snn_k, "_t", gsub("\\.", "", sprintf("%.2f", prune_threshold))),
      exact = FALSE
    ),
    graph_approximation = "snn_weighted",
    graph_effective_k = out_k,
    graph_edge_retention = mean(degrees) / snn_k,
    graph_recall_at_k = mean(degrees) / snn_k,
    graph_mean_degree = mean(degrees),
    graph_min_degree = min(degrees),
    graph_max_degree = max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k),
    graph_snn_k = snn_k,
    graph_snn_prune_threshold = prune_threshold,
    graph_mean_snn_weight = safe_mean(weights_all),
    graph_min_snn_weight = if (length(weights_all) == 0L) NA_real_ else min(weights_all),
    graph_max_snn_weight = if (length(weights_all) == 0L) NA_real_ else max(weights_all),
    graph_zero_snn_fraction = if (length(weights_all) == 0L) NA_real_ else mean(weights_all <= 0)
  )
}

graph_adaptive_k_knn <- function(knn, min_fraction, max_fraction, density_quantile = 0.5) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  min_fraction <- as.numeric(min_fraction)
  max_fraction <- as.numeric(max_fraction)
  min_k <- max(1L, min(base_k, ceiling(base_k * min_fraction)))
  max_k <- max(min_k, min(base_k, ceiling(base_k * max_fraction)))
  density_cols <- cols[seq_len(max_k)]
  local_scale <- apply(knn$distances[, density_cols, drop = FALSE], 1L, function(row) {
    row <- row[is.finite(row) & row > 0]
    if (length(row) == 0L) return(0)
    as.numeric(stats::quantile(row, probs = density_quantile, names = FALSE, type = 7))
  })
  if (max_k > min_k && length(unique(local_scale)) > 1L) {
    density_rank <- (rank(local_scale, ties.method = "average") - 1) / max(1L, n - 1L)
    degrees <- as.integer(round(min_k + density_rank * (max_k - min_k)))
  } else {
    degrees <- rep(as.integer(max_k), n)
  }
  degrees <- pmin(max_k, pmax(min_k, degrees))
  out_k <- max(1L, max(degrees))
  out_cols <- out_k + as.integer(self)
  out_indices <- matrix(NA_integer_, n, out_cols)
  out_distances <- matrix(NA_real_, n, out_cols)
  finite_dist <- knn$distances[is.finite(knn$distances) & knn$distances > 0]
  pad_distance <- if (length(finite_dist) == 0L) 1e6 else max(finite_dist) * 1e3 + 1
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  padding_count <- 0L
  for (i in seq_len(n)) {
    keep <- seq_len(degrees[i])
    out_indices[i, offset + keep] <- knn$indices[i, cols[keep]]
    out_distances[i, offset + keep] <- knn$distances[i, cols[keep]]
    if (degrees[i] < out_k) {
      pad_cols <- offset + seq.int(degrees[i] + 1L, out_k)
      out_indices[i, pad_cols] <- knn$indices[i, cols[1L]]
      out_distances[i, pad_cols] <- pad_distance
      padding_count <- padding_count + length(pad_cols)
    }
  }
  density_cor <- if (length(unique(local_scale)) > 1L && length(unique(degrees)) > 1L) {
    suppressWarnings(stats::cor(local_scale, degrees, method = "spearman"))
  } else {
    NA_real_
  }
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "adaptive_k_",
        gsub("\\.", "", sprintf("%.2f", min_fraction)),
        "_",
        gsub("\\.", "", sprintf("%.2f", max_fraction))
      ),
      exact = FALSE
    ),
    graph_approximation = "adaptive_k",
    graph_effective_k = out_k,
    graph_edge_retention = mean(degrees) / base_k,
    graph_recall_at_k = mean(degrees) / base_k,
    graph_mean_degree = mean(degrees),
    graph_min_degree = min(degrees),
    graph_max_degree = max(degrees),
    graph_isolated_fraction = mean(degrees == 0L),
    graph_padding_fraction = padding_count / max(1L, n * out_k),
    graph_adaptive_min_k = min_k,
    graph_adaptive_max_k = max_k,
    graph_adaptive_mean_k = mean(degrees),
    graph_adaptive_density_quantile = density_quantile,
    graph_adaptive_density_cor = density_cor,
    graph_adaptive_dense_fraction = mean(degrees == min_k),
    graph_adaptive_sparse_fraction = mean(degrees == max_k)
  )
}

graph_multi_k_knn <- function(knn, k_values) {
  k_values <- sort(unique(as.integer(k_values)))
  k_values <- k_values[k_values > 0L]
  if (length(k_values) < 2L) {
    stop("Multi-k graph requires at least two positive k values.", call. = FALSE)
  }
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  max_k <- max(k_values)
  if (length(cols) < max_k) {
    stop("Multi-k graph ", paste(k_values, collapse = ","),
         " requires benchmark --k >= ", max_k, ".", call. = FALSE)
  }
  keep_cols <- cols[seq_len(max_k)]
  out_indices <- knn$indices[, keep_cols, drop = FALSE]
  out_distances <- knn$distances[, keep_cols, drop = FALSE]
  rank <- seq_len(max_k)
  scale_weight <- vapply(rank, function(r) mean(r <= k_values), numeric(1L))
  out_distances <- sweep(out_distances, 2L, pmax(scale_weight, 1e-6), "/")
  if (self) {
    out_indices <- cbind(knn$indices[, 1L], out_indices)
    out_distances <- cbind(knn$distances[, 1L], out_distances)
  }
  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0("multi_k_", paste(k_values, collapse = "_")),
      exact = FALSE
    ),
    graph_approximation = "multi_k",
    graph_effective_k = max_k,
    graph_edge_retention = max_k / base_k,
    graph_recall_at_k = max_k / base_k,
    graph_mean_degree = max_k,
    graph_min_degree = max_k,
    graph_max_degree = max_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = 0,
    graph_multik_values = paste(k_values, collapse = ","),
    graph_multik_num_scales = length(k_values),
    graph_multik_min_k = min(k_values),
    graph_multik_max_k = max_k,
    graph_multik_mean_weight = mean(scale_weight),
    graph_multik_min_weight = min(scale_weight),
    graph_multik_max_weight = max(scale_weight)
  )
}

multiscale_perplexity_suffix <- function(perplexities) {
  paste0("p", paste(as.integer(perplexities), collapse = "_"))
}

multiscale_perplexity_transfer_mode <- function(method) {
  switch(
    method,
    tsne = "openTSNE_multiscale_affinity",
    umap = "umap_multi_k_graph",
    pacmap = "pacmap_multi_scale_near_pairs",
    trimap = "trimap_multi_scale_triplets_proxy",
    localmap = "localmap_multi_scale_local_graph",
    "multiscale_knn_graph"
  )
}

graph_multiscale_perplexity_knn <- function(knn, method, perplexities) {
  perplexities <- sort(unique(as.integer(perplexities)))
  perplexities <- perplexities[perplexities > 0L]
  if (length(perplexities) < 2L) {
    stop("Multiscale perplexity requires at least two positive perplexity values.", call. = FALSE)
  }
  base_k <- knn_effective_k(knn)
  transfer_mode <- multiscale_perplexity_transfer_mode(method)
  if (identical(method, "tsne")) {
    required_k <- 3L * max(perplexities)
    if (base_k < required_k) {
      stop(
        "t-SNE multiscale perplexities ", paste(perplexities, collapse = ","),
        " require --k >= ", required_k, ".",
        call. = FALSE
      )
    }
    result <- graph_tsne_affinity_knn(
      knn,
      perplexities = perplexities,
      temperature = 1,
      mode = paste0("multiscale_", multiscale_perplexity_suffix(perplexities))
    )
    effective_k_values <- 3L * perplexities
    uses_tsne_affinity <- 1
  } else {
    required_k <- max(perplexities)
    if (base_k < required_k) {
      stop(
        method, " multiscale transfer with perplexities ",
        paste(perplexities, collapse = ","),
        " requires --k >= ", required_k, ".",
        call. = FALSE
      )
    }
    result <- graph_multi_k_knn(knn, perplexities)
    effective_k_values <- perplexities
    uses_tsne_affinity <- 0
  }
  result$graph_approximation <- paste0(
    "multiscale_perplexity_",
    multiscale_perplexity_suffix(perplexities)
  )
  result$graph_multiscale_perplexities <- paste(perplexities, collapse = ",")
  result$graph_multiscale_num_scales <- length(perplexities)
  result$graph_multiscale_required_k <- required_k
  result$graph_multiscale_effective_k_values <- paste(effective_k_values, collapse = ",")
  result$graph_multiscale_transfer_mode <- transfer_mode
  result$graph_multiscale_uses_tsne_affinity <- uses_tsne_affinity
  result
}

graph_pacmap_mid_near_knn <- function(knn,
                                      mid_fraction = 0.25,
                                      mid_distance_scale = 1.35,
                                      seed = 1L) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  if (base_k < 3L) {
    stop("PaCMAP mid-near graph transfer requires at least three non-self neighbours.", call. = FALSE)
  }
  mid_fraction <- min(0.75, max(0.05, as.numeric(mid_fraction)))
  mid_slots <- max(1L, min(base_k - 1L, as.integer(ceiling(base_k * mid_fraction))))
  near_keep <- max(1L, base_k - mid_slots)
  mid_distance_scale <- max(1.01, as.numeric(mid_distance_scale))

  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    as.integer(unique(x[is.finite(x) & x >= 1L & x <= n & x != i]))
  })
  out_indices <- matrix(NA_integer_, n, base_k + as.integer(self))
  out_distances <- matrix(NA_real_, n, base_k + as.integer(self))
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  mid_added <- integer(n)
  fallback_slots <- integer(n)
  mid_rank_values <- numeric(0)
  eps <- 1e-7

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i)
    if (length(valid) == 0L) {
      valid <- seq_len(base_k)
      row_idx[!is.finite(row_idx) | row_idx < 1L | row_idx > n] <- if (i == 1L) 2L else 1L
      row_dst[!is.finite(row_dst)] <- max(1, safe_number(row_dst, 1))
    }
    near_positions <- valid[seq_len(min(near_keep, length(valid)))]
    direct <- unique(row_idx[valid])
    direct <- direct[is.finite(direct) & direct >= 1L & direct <= n & direct != i]
    direct_set <- c(i, direct)

    pool <- integer(0L)
    for (j in direct) {
      pool <- c(pool, neighbor_sets[[j]])
    }
    pool <- pool[is.finite(pool) & pool >= 1L & pool <= n & !pool %in% direct_set]
    if (length(pool) > 0L) {
      counts <- sort(table(pool), decreasing = TRUE)
      cand <- as.integer(names(counts))
      cand <- cand[order(-as.integer(counts), cand)]
    } else {
      cand <- integer(0L)
    }
    if (length(cand) > mid_slots) {
      start <- as.integer((as.double(i) * 1103515245 + as.double(seed) * 12345) %% length(cand)) + 1L
      cand <- cand[c(seq.int(start, length(cand)), if (start > 1L) seq_len(start - 1L) else integer())]
    }
    mid <- head(cand, mid_slots)
    fallback <- integer(0L)
    if (length(mid) < mid_slots) {
      remaining_direct <- row_idx[valid[!valid %in% near_positions]]
      remaining_direct <- remaining_direct[!remaining_direct %in% c(i, mid)]
      fallback <- head(remaining_direct, mid_slots - length(mid))
      mid <- c(mid, fallback)
    }
    if (length(mid) < mid_slots) {
      repeat_from <- if (length(near_positions) > 0L) row_idx[near_positions[1L]] else if (i == 1L) 2L else 1L
      mid <- c(mid, rep(repeat_from, mid_slots - length(mid)))
    }

    chosen <- c(row_idx[near_positions], mid)
    chosen <- chosen[seq_len(min(base_k, length(chosen)))]
    if (length(chosen) < base_k) {
      chosen <- c(chosen, rep(chosen[1L], base_k - length(chosen)))
    }

    near_dist <- row_dst[near_positions]
    near_dist[!is.finite(near_dist)] <- safe_number(row_dst, 1)
    row_positive <- row_dst[is.finite(row_dst) & row_dst > 0]
    row_scale <- if (length(row_positive) == 0L) {
      1
    } else {
      max(row_positive, stats::quantile(row_positive, 0.75, names = FALSE, type = 7))
    }
    mid_dist <- rep(row_scale * mid_distance_scale, length(mid)) + seq_along(mid) * eps
    chosen_dist <- c(near_dist, mid_dist)[seq_len(base_k)]
    if (length(chosen_dist) < base_k) {
      chosen_dist <- c(chosen_dist, rep(row_scale * mid_distance_scale, base_k - length(chosen_dist)))
    }

    out_indices[i, offset + seq_len(base_k)] <- as.integer(chosen)
    out_distances[i, offset + seq_len(base_k)] <- as.numeric(chosen_dist)
    mid_added[i] <- sum(!mid %in% direct)
    fallback_slots[i] <- length(fallback)
    if (length(mid) > 0L) {
      mid_rank_values <- c(mid_rank_values, match(mid, cand, nomatch = NA_integer_))
    }
  }

  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "pacmap_mid_near_f",
        gsub("\\.", "p", format(mid_fraction, trim = TRUE, scientific = FALSE)),
        "_s",
        gsub("\\.", "p", format(mid_distance_scale, trim = TRUE, scientific = FALSE))
      ),
      exact = FALSE
    ),
    graph_approximation = "pacmap_mid_near_transfer",
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = near_keep / base_k,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = safe_mean(fallback_slots / mid_slots),
    pacmap_transfer_mode = "mid_near_pairs_as_second_order_graph_edges",
    pacmap_auxiliary_pair_family = "mid_near_pairs",
    pacmap_mid_near_pairs_per_point = mid_slots,
    pacmap_mid_near_fraction = safe_mean(mid_added / base_k),
    pacmap_mid_near_requested_fraction = mid_fraction,
    pacmap_mid_near_distance_scale = mid_distance_scale,
    pacmap_mid_near_fallback_fraction = safe_mean(fallback_slots / mid_slots),
    pacmap_mid_near_rank_mean = safe_mean(mid_rank_values)
  )
}

graph_pacmap_mid_near_emphasis_knn <- function(knn,
                                               mid_fraction = 0.30,
                                               emphasis_strength = 1.50,
                                               seed = 1L) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  if (base_k < 6L) {
    stop("PaCMAP mid-near emphasis requires at least six non-self neighbours.", call. = FALSE)
  }
  mid_fraction <- min(0.75, max(0.05, as.numeric(mid_fraction)))
  emphasis_strength <- max(0.25, as.numeric(emphasis_strength))
  mid_slots <- max(1L, min(base_k - 1L, as.integer(ceiling(base_k * mid_fraction))))
  near_keep <- max(1L, base_k - mid_slots)
  mid_distance_multiplier <- max(0.20, min(2.00, 1 / emphasis_strength))

  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    as.integer(unique(x[is.finite(x) & x >= 1L & x <= n & x != i]))
  })
  out_indices <- matrix(NA_integer_, n, base_k + as.integer(self))
  out_distances <- matrix(NA_real_, n, base_k + as.integer(self))
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  mid_added <- integer(n)
  fallback_slots <- integer(n)
  mid_rank_values <- numeric(0)
  eps <- 1e-7

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i)
    if (length(valid) == 0L) {
      valid <- seq_len(base_k)
      row_idx[!is.finite(row_idx) | row_idx < 1L | row_idx > n] <- if (i == 1L) 2L else 1L
      row_dst[!is.finite(row_dst)] <- max(1, safe_number(row_dst, 1))
    }
    near_positions <- valid[seq_len(min(near_keep, length(valid)))]
    direct <- unique(row_idx[valid])
    direct <- direct[is.finite(direct) & direct >= 1L & direct <= n & direct != i]
    direct_set <- c(i, direct)

    pool <- integer(0L)
    for (j in direct) {
      pool <- c(pool, neighbor_sets[[j]])
    }
    pool <- pool[is.finite(pool) & pool >= 1L & pool <= n & !pool %in% direct_set]
    if (length(pool) > 0L) {
      counts <- sort(table(pool), decreasing = TRUE)
      cand <- as.integer(names(counts))
      cand <- cand[order(-as.integer(counts), cand)]
      start <- as.integer((as.double(i) * 1103515245 + as.double(seed) * 12345) %% length(cand)) + 1L
      cand <- cand[c(seq.int(start, length(cand)), if (start > 1L) seq_len(start - 1L) else integer())]
    } else {
      cand <- integer(0L)
    }
    mid <- head(cand, mid_slots)
    fallback <- integer(0L)
    if (length(mid) < mid_slots) {
      remaining_direct <- row_idx[valid[!valid %in% near_positions]]
      remaining_direct <- remaining_direct[!remaining_direct %in% c(i, mid)]
      fallback <- head(remaining_direct, mid_slots - length(mid))
      mid <- c(mid, fallback)
    }
    if (length(mid) < mid_slots) {
      repeat_from <- if (length(near_positions) > 0L) row_idx[near_positions[1L]] else if (i == 1L) 2L else 1L
      mid <- c(mid, rep(repeat_from, mid_slots - length(mid)))
    }

    chosen <- c(row_idx[near_positions], mid)
    chosen <- chosen[seq_len(min(base_k, length(chosen)))]
    if (length(chosen) < base_k) {
      chosen <- c(chosen, rep(chosen[1L], base_k - length(chosen)))
    }

    near_dist <- row_dst[near_positions]
    near_dist[!is.finite(near_dist)] <- safe_number(row_dst, 1)
    row_positive <- row_dst[is.finite(row_dst) & row_dst > 0]
    if (length(row_positive) == 0L) {
      row_scale <- 1
      near_floor <- 1e-6
    } else {
      row_scale <- stats::quantile(row_positive, 0.75, names = FALSE, type = 7)
      near_floor <- stats::quantile(row_positive, 0.25, names = FALSE, type = 7)
    }
    mid_dist_value <- max(near_floor, row_scale * mid_distance_multiplier)
    mid_dist <- rep(mid_dist_value, length(mid)) + seq_along(mid) * eps
    chosen_dist <- c(near_dist, mid_dist)[seq_len(base_k)]
    if (length(chosen_dist) < base_k) {
      chosen_dist <- c(chosen_dist, rep(mid_dist_value, base_k - length(chosen_dist)))
    }

    out_indices[i, offset + seq_len(base_k)] <- as.integer(chosen)
    out_distances[i, offset + seq_len(base_k)] <- as.numeric(chosen_dist)
    mid_added[i] <- sum(!mid %in% direct)
    fallback_slots[i] <- length(fallback)
    if (length(mid) > 0L) {
      mid_rank_values <- c(mid_rank_values, match(mid, cand, nomatch = NA_integer_))
    }
  }

  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "pacmap_mid_near_emphasis_f",
        gsub("\\.", "p", format(mid_fraction, trim = TRUE, scientific = FALSE)),
        "_e",
        gsub("\\.", "p", format(emphasis_strength, trim = TRUE, scientific = FALSE))
      ),
      exact = FALSE
    ),
    graph_approximation = "pacmap_mid_near_emphasis",
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = near_keep / base_k,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = safe_mean(fallback_slots / mid_slots),
    graph_multiscale_transfer_mode = "pacmap_mid_near_emphasis_as_multik_midrange_edges",
    graph_multiscale_uses_tsne_affinity = 0,
    pacmap_transfer_mode = "mid_near_pair_emphasis_graph",
    pacmap_auxiliary_pair_family = "emphasized_mid_near_pairs",
    pacmap_mid_near_pairs_per_point = mid_slots,
    pacmap_mid_near_fraction = safe_mean(mid_added / base_k),
    pacmap_mid_near_requested_fraction = mid_fraction,
    pacmap_mid_near_distance_scale = mid_distance_multiplier,
    pacmap_mid_near_fallback_fraction = safe_mean(fallback_slots / mid_slots),
    pacmap_mid_near_rank_mean = safe_mean(mid_rank_values),
    pacmap_mid_near_emphasis_strength = emphasis_strength,
    pacmap_mid_near_emphasis_distance_multiplier = mid_distance_multiplier
  )
}

graph_pacmap_pair_separation_knn <- function(knn,
                                             near_ratio = 0.50,
                                             mid_ratio = 0.30,
                                             far_ratio = 0.20,
                                             mid_distance_scale = 1.40,
                                             far_distance_scale = 3.00,
                                             seed = 1L) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  if (base_k < 6L) {
    stop("PaCMAP near/mid/far pair separation requires at least six non-self neighbours.", call. = FALSE)
  }
  ratios <- pmax(0, as.numeric(c(near_ratio, mid_ratio, far_ratio)))
  if (!any(is.finite(ratios)) || sum(ratios, na.rm = TRUE) <= 0) {
    stop("PaCMAP pair-separation ratios must contain at least one positive value.", call. = FALSE)
  }
  ratios[!is.finite(ratios)] <- 0
  ratios <- ratios / sum(ratios)
  near_ratio <- ratios[1L]
  mid_ratio <- ratios[2L]
  far_ratio <- ratios[3L]
  near_slots <- max(1L, as.integer(round(base_k * near_ratio)))
  mid_slots <- max(0L, as.integer(round(base_k * mid_ratio)))
  far_slots <- max(0L, base_k - near_slots - mid_slots)
  if (mid_ratio > 0 && mid_slots == 0L && base_k >= 2L) {
    mid_slots <- 1L
    near_slots <- max(1L, near_slots - 1L)
  }
  if (far_ratio > 0 && far_slots == 0L && base_k >= 3L) {
    far_slots <- 1L
    if (mid_slots > 0L) {
      mid_slots <- mid_slots - 1L
    } else {
      near_slots <- max(1L, near_slots - 1L)
    }
  }
  while (near_slots + mid_slots + far_slots > base_k) {
    if (mid_slots >= far_slots && mid_slots > 0L) {
      mid_slots <- mid_slots - 1L
    } else if (far_slots > 0L) {
      far_slots <- far_slots - 1L
    } else {
      near_slots <- near_slots - 1L
    }
  }
  while (near_slots + mid_slots + far_slots < base_k) {
    near_slots <- near_slots + 1L
  }
  mid_distance_scale <- max(1.01, as.numeric(mid_distance_scale))
  far_distance_scale <- max(mid_distance_scale + 0.01, as.numeric(far_distance_scale))

  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    as.integer(unique(x[is.finite(x) & x >= 1L & x <= n & x != i]))
  })
  out_indices <- matrix(NA_integer_, n, base_k + as.integer(self))
  out_distances <- matrix(NA_real_, n, base_k + as.integer(self))
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  mid_added <- integer(n)
  far_added <- integer(n)
  mid_fallback <- integer(n)
  far_fallback <- integer(n)
  eps <- 1e-7

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i)
    if (length(valid) == 0L) {
      valid <- seq_len(base_k)
      row_idx[!is.finite(row_idx) | row_idx < 1L | row_idx > n] <- if (i == 1L) 2L else 1L
      row_dst[!is.finite(row_dst)] <- max(1, safe_number(row_dst, 1))
    }
    near_positions <- valid[seq_len(min(near_slots, length(valid)))]
    direct <- unique(row_idx[valid])
    direct <- direct[is.finite(direct) & direct >= 1L & direct <= n & direct != i]
    direct_set <- c(i, direct)

    two_hop_pool <- integer(0L)
    for (j in direct) {
      two_hop_pool <- c(two_hop_pool, neighbor_sets[[j]])
    }
    two_hop_pool <- two_hop_pool[
      is.finite(two_hop_pool) & two_hop_pool >= 1L & two_hop_pool <= n &
        !two_hop_pool %in% direct_set
    ]
    if (length(two_hop_pool) > 0L) {
      counts <- sort(table(two_hop_pool), decreasing = TRUE)
      mid_candidates <- as.integer(names(counts))
      mid_candidates <- mid_candidates[order(-as.integer(counts), mid_candidates)]
      start <- as.integer((as.double(i) * 1103515245 + as.double(seed) * 12345) %% length(mid_candidates)) + 1L
      mid_candidates <- mid_candidates[c(seq.int(start, length(mid_candidates)), if (start > 1L) seq_len(start - 1L) else integer())]
    } else {
      mid_candidates <- integer(0L)
    }
    mid <- head(mid_candidates, mid_slots)
    if (length(mid) < mid_slots) {
      fallback_mid <- row_idx[valid[!valid %in% near_positions]]
      fallback_mid <- fallback_mid[!fallback_mid %in% c(i, mid)]
      fallback_mid <- head(fallback_mid, mid_slots - length(mid))
      mid <- c(mid, fallback_mid)
      mid_fallback[i] <- length(fallback_mid)
    }

    far_exclusion <- unique(c(i, direct, mid, two_hop_pool))
    far_candidates <- seq_len(n)
    far_candidates <- far_candidates[!far_candidates %in% far_exclusion]
    if (length(far_candidates) > 0L) {
      start <- as.integer((as.double(i) * 1664525 + as.double(seed + 17L) * 1013904223) %% length(far_candidates)) + 1L
      far_candidates <- far_candidates[c(seq.int(start, length(far_candidates)), if (start > 1L) seq_len(start - 1L) else integer())]
    }
    far <- head(far_candidates, far_slots)
    if (length(far) < far_slots) {
      fallback_far <- rev(row_idx[valid])
      fallback_far <- fallback_far[!fallback_far %in% c(i, mid, far)]
      fallback_far <- head(fallback_far, far_slots - length(far))
      far <- c(far, fallback_far)
      far_fallback[i] <- length(fallback_far)
    }

    chosen <- c(row_idx[near_positions], mid, far)
    if (length(chosen) == 0L) {
      chosen <- if (i == 1L) 2L else 1L
    }
    chosen <- chosen[seq_len(min(base_k, length(chosen)))]
    if (length(chosen) < base_k) {
      chosen <- c(chosen, rep(chosen[1L], base_k - length(chosen)))
    }

    near_dist <- row_dst[near_positions]
    near_dist[!is.finite(near_dist)] <- safe_number(row_dst, 1)
    row_positive <- row_dst[is.finite(row_dst) & row_dst > 0]
    row_scale <- if (length(row_positive) == 0L) {
      1
    } else {
      max(row_positive, stats::quantile(row_positive, 0.75, names = FALSE, type = 7))
    }
    mid_dist <- rep(row_scale * mid_distance_scale, length(mid)) + seq_along(mid) * eps
    far_dist <- rep(row_scale * far_distance_scale, length(far)) + seq_along(far) * eps
    chosen_dist <- c(near_dist, mid_dist, far_dist)
    chosen_dist <- chosen_dist[seq_len(min(base_k, length(chosen_dist)))]
    if (length(chosen_dist) < base_k) {
      chosen_dist <- c(chosen_dist, rep(row_scale * far_distance_scale, base_k - length(chosen_dist)))
    }

    out_indices[i, offset + seq_len(base_k)] <- as.integer(chosen)
    out_distances[i, offset + seq_len(base_k)] <- as.numeric(chosen_dist)
    mid_added[i] <- sum(!mid %in% direct)
    far_added[i] <- sum(!far %in% c(direct, mid))
  }

  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "pacmap_pair_sep_n", gsub("\\.", "p", format(near_ratio, trim = TRUE, scientific = FALSE)),
        "_m", gsub("\\.", "p", format(mid_ratio, trim = TRUE, scientific = FALSE)),
        "_f", gsub("\\.", "p", format(far_ratio, trim = TRUE, scientific = FALSE))
      ),
      exact = FALSE
    ),
    graph_approximation = "pacmap_near_mid_far_pair_separation",
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = near_slots / base_k,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = safe_mean((mid_fallback + far_fallback) / max(1L, mid_slots + far_slots)),
    pacmap_transfer_mode = "near_mid_far_pair_separation_graph",
    pacmap_auxiliary_pair_family = "near_mid_near_far_pairs",
    pacmap_near_ratio = near_slots / base_k,
    pacmap_mid_ratio = mid_slots / base_k,
    pacmap_far_ratio = far_slots / base_k,
    pacmap_near_pairs_per_point = near_slots,
    pacmap_mid_pairs_per_point = mid_slots,
    pacmap_far_pairs_per_point = far_slots,
    pacmap_mid_near_pairs_per_point = mid_slots,
    pacmap_mid_near_fraction = safe_mean(mid_added / base_k),
    pacmap_mid_near_requested_fraction = mid_ratio,
    pacmap_mid_near_distance_scale = mid_distance_scale,
    pacmap_mid_near_fallback_fraction = safe_mean(mid_fallback / max(1L, mid_slots)),
    pacmap_far_fallback_fraction = safe_mean(far_fallback / max(1L, far_slots)),
    pacmap_far_distance_scale = far_distance_scale,
    pacmap_far_pair_fraction = safe_mean(far_added / base_k)
	  )
	}

graph_trimap_triplet_proxy_knn <- function(knn,
                                           inlier_ratio = 0.70,
                                           semihard_ratio = 0.25,
                                           global_anchor_ratio = 0.05,
                                           semihard_distance_scale = 1.40,
                                           global_anchor_distance_scale = 3.50,
                                           family = "semi_hard",
                                           seed = 1L) {
  n <- nrow(knn$indices)
  self <- knn_has_self_first(knn)
  cols <- knn_neighbor_cols(knn)
  base_k <- max(1L, length(cols))
  if (base_k < 6L) {
    stop("TriMap triplet proxy requires at least six non-self neighbours.", call. = FALSE)
  }
  ratios <- pmax(0, as.numeric(c(inlier_ratio, semihard_ratio, global_anchor_ratio)))
  if (!any(is.finite(ratios)) || sum(ratios, na.rm = TRUE) <= 0) {
    stop("TriMap triplet proxy ratios must contain at least one positive value.", call. = FALSE)
  }
  ratios[!is.finite(ratios)] <- 0
  ratios <- ratios / sum(ratios)
  inlier_ratio <- ratios[1L]
  semihard_ratio <- ratios[2L]
  global_anchor_ratio <- ratios[3L]

  inlier_slots <- max(1L, as.integer(round(base_k * inlier_ratio)))
  semihard_slots <- max(0L, as.integer(round(base_k * semihard_ratio)))
  global_slots <- max(0L, base_k - inlier_slots - semihard_slots)
  if (semihard_ratio > 0 && semihard_slots == 0L && base_k >= 2L) {
    semihard_slots <- 1L
    inlier_slots <- max(1L, inlier_slots - 1L)
  }
  if (global_anchor_ratio > 0 && global_slots == 0L && base_k >= 3L) {
    global_slots <- 1L
    if (semihard_slots > 0L) {
      semihard_slots <- semihard_slots - 1L
    } else {
      inlier_slots <- max(1L, inlier_slots - 1L)
    }
  }
  while (inlier_slots + semihard_slots + global_slots > base_k) {
    if (global_slots > 0L) {
      global_slots <- global_slots - 1L
    } else if (semihard_slots > 0L) {
      semihard_slots <- semihard_slots - 1L
    } else {
      inlier_slots <- inlier_slots - 1L
    }
  }
  while (inlier_slots + semihard_slots + global_slots < base_k) {
    inlier_slots <- inlier_slots + 1L
  }

  semihard_distance_scale <- max(1.01, as.numeric(semihard_distance_scale))
  global_anchor_distance_scale <- max(semihard_distance_scale + 0.01, as.numeric(global_anchor_distance_scale))
  family <- safe_character(family, "semi_hard")

  neighbor_sets <- lapply(seq_len(n), function(i) {
    x <- knn$indices[i, cols]
    as.integer(unique(x[is.finite(x) & x >= 1L & x <= n & x != i]))
  })
  out_indices <- matrix(NA_integer_, n, base_k + as.integer(self))
  out_distances <- matrix(NA_real_, n, base_k + as.integer(self))
  if (self) {
    out_indices[, 1L] <- knn$indices[, 1L]
    out_distances[, 1L] <- knn$distances[, 1L]
  }
  offset <- as.integer(self)
  semihard_added <- integer(n)
  global_added <- integer(n)
  semihard_fallback <- integer(n)
  global_fallback <- integer(n)
  semihard_rank_values <- numeric(0)
  eps <- 1e-7

  for (i in seq_len(n)) {
    row_idx <- as.integer(knn$indices[i, cols])
    row_dst <- as.numeric(knn$distances[i, cols])
    valid <- which(is.finite(row_idx) & row_idx >= 1L & row_idx <= n & row_idx != i)
    if (length(valid) == 0L) {
      valid <- seq_len(base_k)
      row_idx[!is.finite(row_idx) | row_idx < 1L | row_idx > n] <- if (i == 1L) 2L else 1L
      row_dst[!is.finite(row_dst)] <- max(1, safe_number(row_dst, 1))
    }
    inlier_positions <- valid[seq_len(min(inlier_slots, length(valid)))]
    direct <- unique(row_idx[valid])
    direct <- direct[is.finite(direct) & direct >= 1L & direct <= n & direct != i]
    direct_set <- c(i, direct)

    two_hop_pool <- integer(0L)
    for (j in direct) {
      two_hop_pool <- c(two_hop_pool, neighbor_sets[[j]])
    }
    two_hop_pool <- two_hop_pool[
      is.finite(two_hop_pool) & two_hop_pool >= 1L & two_hop_pool <= n &
        !two_hop_pool %in% direct_set
    ]
    if (length(two_hop_pool) > 0L) {
      counts <- sort(table(two_hop_pool), decreasing = TRUE)
      semihard_candidates <- as.integer(names(counts))
      semihard_candidates <- semihard_candidates[order(-as.integer(counts), semihard_candidates)]
      start <- as.integer((as.double(i) * 1103515245 + as.double(seed) * 12345) %% length(semihard_candidates)) + 1L
      semihard_candidates <- semihard_candidates[
        c(seq.int(start, length(semihard_candidates)), if (start > 1L) seq_len(start - 1L) else integer())
      ]
    } else {
      semihard_candidates <- integer(0L)
    }
    semihard <- head(semihard_candidates, semihard_slots)
    if (length(semihard) < semihard_slots) {
      fallback <- row_idx[valid[!valid %in% inlier_positions]]
      fallback <- fallback[!fallback %in% c(i, semihard)]
      fallback <- head(fallback, semihard_slots - length(semihard))
      semihard <- c(semihard, fallback)
      semihard_fallback[i] <- length(fallback)
    }

    global_exclusion <- unique(c(i, direct, semihard, two_hop_pool))
    global_candidates <- seq_len(n)
    global_candidates <- global_candidates[!global_candidates %in% global_exclusion]
    if (length(global_candidates) > 0L) {
      start <- as.integer((as.double(i) * 1664525 + as.double(seed + 29L) * 1013904223) %% length(global_candidates)) + 1L
      global_candidates <- global_candidates[
        c(seq.int(start, length(global_candidates)), if (start > 1L) seq_len(start - 1L) else integer())
      ]
    }
    global <- head(global_candidates, global_slots)
    if (length(global) < global_slots) {
      fallback <- rev(row_idx[valid])
      fallback <- fallback[!fallback %in% c(i, semihard, global)]
      fallback <- head(fallback, global_slots - length(global))
      global <- c(global, fallback)
      global_fallback[i] <- length(fallback)
    }

    chosen <- c(row_idx[inlier_positions], semihard, global)
    if (length(chosen) == 0L) chosen <- if (i == 1L) 2L else 1L
    chosen <- chosen[seq_len(min(base_k, length(chosen)))]
    if (length(chosen) < base_k) chosen <- c(chosen, rep(chosen[1L], base_k - length(chosen)))

    inlier_dist <- row_dst[inlier_positions]
    inlier_dist[!is.finite(inlier_dist)] <- safe_number(row_dst, 1)
    row_positive <- row_dst[is.finite(row_dst) & row_dst > 0]
    row_scale <- if (length(row_positive) == 0L) {
      1
    } else {
      max(row_positive, stats::quantile(row_positive, 0.75, names = FALSE, type = 7))
    }
    semihard_dist <- rep(row_scale * semihard_distance_scale, length(semihard)) + seq_along(semihard) * eps
    global_dist <- rep(row_scale * global_anchor_distance_scale, length(global)) + seq_along(global) * eps
    chosen_dist <- c(inlier_dist, semihard_dist, global_dist)
    chosen_dist <- chosen_dist[seq_len(min(base_k, length(chosen_dist)))]
    if (length(chosen_dist) < base_k) {
      chosen_dist <- c(chosen_dist, rep(row_scale * global_anchor_distance_scale, base_k - length(chosen_dist)))
    }

    out_indices[i, offset + seq_len(base_k)] <- as.integer(chosen)
    out_distances[i, offset + seq_len(base_k)] <- as.numeric(chosen_dist)
    semihard_added[i] <- sum(!semihard %in% direct)
    global_added[i] <- sum(!global %in% c(direct, semihard))
    if (length(semihard) > 0L) {
      semihard_rank_values <- c(semihard_rank_values, match(semihard, semihard_candidates, nomatch = NA_integer_))
    }
  }

  list(
    knn = graph_knn_result(
      knn,
      out_indices,
      out_distances,
      paste0(
        "trimap_triplet_proxy_i", gsub("\\.", "p", format(inlier_ratio, trim = TRUE, scientific = FALSE)),
        "_s", gsub("\\.", "p", format(semihard_ratio, trim = TRUE, scientific = FALSE)),
        "_g", gsub("\\.", "p", format(global_anchor_ratio, trim = TRUE, scientific = FALSE))
      ),
      exact = FALSE
    ),
    graph_approximation = "trimap_triplet_candidate_proxy",
    graph_effective_k = base_k,
    graph_edge_retention = 1,
    graph_recall_at_k = inlier_slots / base_k,
    graph_mean_degree = base_k,
    graph_min_degree = base_k,
    graph_max_degree = base_k,
    graph_isolated_fraction = 0,
    graph_padding_fraction = safe_mean((semihard_fallback + global_fallback) / max(1L, semihard_slots + global_slots)),
    trimap_transfer_mode = "triplet_candidate_graph_proxy",
    trimap_triplet_family = family,
    trimap_inlier_ratio = inlier_slots / base_k,
    trimap_semihard_ratio = semihard_slots / base_k,
    trimap_global_anchor_ratio = global_slots / base_k,
    trimap_inlier_pairs_per_point = inlier_slots,
    trimap_semihard_pairs_per_point = semihard_slots,
    trimap_global_anchor_pairs_per_point = global_slots,
    trimap_semihard_fraction = safe_mean(semihard_added / base_k),
    trimap_global_anchor_fraction = safe_mean(global_added / base_k),
    trimap_semihard_distance_scale = semihard_distance_scale,
    trimap_global_anchor_distance_scale = global_anchor_distance_scale,
    trimap_semihard_fallback_fraction = safe_mean(semihard_fallback / max(1L, semihard_slots)),
    trimap_global_anchor_fallback_fraction = safe_mean(global_fallback / max(1L, global_slots)),
    trimap_semihard_rank_mean = safe_mean(semihard_rank_values),
    trimap_candidate_seed = as.integer(seed),
    trimap_native_explicit_triplets = 0,
    trimap_triplet_proxy_detail = "positive_candidate_graph_proxy_negatives_sampled_by_optimizer"
  )
}

graph_standard_knn <- function(knn) {
  list(
    knn = knn,
    graph_approximation = "standard_knn",
    graph_effective_k = knn_effective_k(knn),
    graph_edge_retention = 1
  )
}

profiled_knn_result <- function(value, index_build_time_sec = NA_real_, query_time_sec = NA_real_) {
  attr(value, "index_build_time_sec") <- as.numeric(index_build_time_sec)
  attr(value, "query_time_sec") <- as.numeric(query_time_sec)
  value
}

timed_step <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, time = proc.time()[["elapsed"]] - start)
}

strip_self_and_limit_by_query <- function(indices, distances, k, self_indices = NULL) {
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  storage.mode(indices) <- "integer"
  storage.mode(distances) <- "double"
  if (is.null(self_indices)) self_indices <- rep(NA_integer_, nrow(indices))
  self_indices <- as.integer(self_indices)
  out_indices <- matrix(NA_integer_, nrow(indices), k)
  out_distances <- matrix(NA_real_, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    self <- self_indices[i]
    keep <- if (is.na(self)) seq_len(ncol(indices)) else which(indices[i, ] != self)
    if (length(keep) < k) keep <- seq_len(ncol(indices))
    keep <- keep[seq_len(min(k, length(keep)))]
    out_indices[i, seq_along(keep)] <- indices[i, keep]
    out_distances[i, seq_along(keep)] <- distances[i, keep]
  }
  list(indices = out_indices, distances = out_distances)
}

knn_quality_reference <- function(x, k, sample_size, seed) {
  n <- nrow(x)
  sample_size <- min(as.integer(sample_size), n)
  set.seed(seed)
  rows <- sort(sample(seq_len(n), sample_size))
  exact <- fastEmbedR::nn(
    x,
    x[rows, , drop = FALSE],
    k = k + 1L,
    backend = "cpu"
  )
  list(
    rows = rows,
    reference = strip_self_and_limit_by_query(exact$indices, exact$distances, k, rows)
  )
}

knn_quality_rank_correlation <- function(candidate, reference, k) {
  idx <- candidate$indices
  ref <- reference$indices
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

evaluate_knn_quality <- function(knn, x, k, sample_size, seed) {
  if (is.null(knn) || !is.list(knn) || !all(c("indices", "distances") %in% names(knn))) {
    return(list(
      knn_recall_at_k = NA_real_,
      knn_mean_distance_error = NA_real_,
      knn_rank_correlation = NA_real_,
      knn_quality_sample_size = NA_integer_
    ))
  }
  ref <- knn_quality_reference(x, k, sample_size, seed)
  rows <- ref$rows
  candidate <- strip_self_and_limit_by_query(
    knn$indices[rows, , drop = FALSE],
    knn$distances[rows, , drop = FALSE],
    k,
    rows
  )
  reference <- ref$reference
  if (!identical(dim(candidate$indices), dim(reference$indices))) {
    return(list(
      knn_recall_at_k = NA_real_,
      knn_mean_distance_error = NA_real_,
      knn_rank_correlation = NA_real_,
      knn_quality_sample_size = length(rows)
    ))
  }
  recall <- mean(vapply(seq_len(nrow(reference$indices)), function(i) {
    length(intersect(candidate$indices[i, seq_len(k)], reference$indices[i, seq_len(k)])) / k
  }, numeric(1L)))
  list(
    knn_recall_at_k = recall,
    knn_mean_distance_error = safe_mean(abs(candidate$distances - reference$distances)),
    knn_rank_correlation = knn_quality_rank_correlation(candidate, reference, k),
    knn_quality_sample_size = length(rows)
  )
}

sklearn_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(list(available = FALSE, message = "R package `reticulate` is not installed."))
  }
  ok <- tryCatch({
    reticulate::import("sklearn.neighbors", delay_load = FALSE)
    TRUE
  }, error = function(e) FALSE)
  list(
    available = ok,
    message = "Python package `scikit-learn` is not available through reticulate."
  )
}

sklearn_knn <- function(ctx, algorithm, backend_name) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  neighbors <- tryCatch(
    reticulate::import("sklearn.neighbors", delay_load = FALSE),
    error = function(e) stop("Python package `scikit-learn` is not available through reticulate.", call. = FALSE)
  )
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  model <- neighbors$NearestNeighbors(
    n_neighbors = as.integer(ctx$k + 1L),
    algorithm = algorithm,
    metric = normalize_knn_metric(ctx$knn_metric)
  )
  build <- timed_step(model$fit(z))
  query <- timed_step(model$kneighbors(z, return_distance = TRUE))
  queried <- query$value
  distances <- as.matrix(queried[[1L]])
  indices <- as.matrix(queried[[2L]]) + 1L
  profiled_knn_result(finish_external_knn(indices, distances, backend_name, exact = TRUE), build$time, query$time)
}

kdtree_rnanoflann_knn <- function(ctx) {
  if (!requireNamespace("Rnanoflann", quietly = TRUE)) {
    stop("R package `Rnanoflann` is not installed.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  out <- Rnanoflann::nn(
    z,
    z,
    k = ctx$k + 1L,
    method = "euclidean",
    search = "standard",
    parallel = FALSE
  )
  finish_external_knn(out$indices, out$distances, "Rnanoflann_kdtree", exact = TRUE)
}

kdtree_fnn_knn <- function(ctx) {
  if (!requireNamespace("FNN", quietly = TRUE)) {
    stop("R package `FNN` is not installed.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  out <- FNN::get.knn(z, k = ctx$k, algorithm = "kd_tree")
  finish_external_knn(out$nn.index, out$nn.dist, "FNN_kdtree", exact = TRUE)
}

kdtree_sklearn_knn <- function(ctx) {
  sklearn_knn(ctx, "kd_tree", "sklearn_kdtree")
}

balltree_sklearn_knn <- function(ctx) {
  sklearn_knn(ctx, "ball_tree", "sklearn_balltree")
}

brute_sklearn_knn <- function(ctx) {
  sklearn_knn(ctx, "brute", "sklearn_brute")
}

annoy_available <- function() {
  list(
    available = requireNamespace("RcppAnnoy", quietly = TRUE),
    message = "R package `RcppAnnoy` is not installed."
  )
}

annoy_search_k_value <- function(k, n_trees, search_multiplier) {
  if (is.null(search_multiplier) || is.na(search_multiplier)) return(-1L)
  as.integer(max(1L, ceiling(as.numeric(search_multiplier) * n_trees * (k + 1L))))
}

annoy_knn <- function(ctx) {
  if (!requireNamespace("RcppAnnoy", quietly = TRUE)) {
    stop("R package `RcppAnnoy` is not installed.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  n <- nrow(z)
  query_k <- as.integer(ctx$k + 1L)
  build <- timed_step({
    index <- RcppAnnoy::AnnoyEuclidean$new(ncol(z))
    index$setSeed(as.integer(ctx$seed))
    for (i in seq_len(n)) {
      index$addItem(as.integer(i - 1L), z[i, ])
    }
    index$build(as.integer(ctx$annoy_n_trees))
    index
  })
  search_k <- annoy_search_k_value(ctx$k, ctx$annoy_n_trees, ctx$annoy_search_multiplier)
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
  indices <- query$value$indices
  distances <- query$value$distances
  if (anyNA(indices) || anyNA(distances)) {
    stop("Annoy returned fewer neighbors than requested.", call. = FALSE)
  }
  profiled_knn_result(finish_external_knn(indices, distances, "RcppAnnoy_euclidean", exact = FALSE), build$time, query$time)
}

annoy_strategy <- function(n_trees, search_multiplier) {
  search_label <- if (is.null(search_multiplier) || is.na(search_multiplier)) {
    "default"
  } else {
    paste0(search_multiplier, "x")
  }
  id_label <- gsub("[^A-Za-z0-9]+", "", search_label)
  list(
    id = paste0("annoy_t", n_trees, "_sk", id_label),
    family = "annoy_knn",
    description = "Approximate nearest neighbors using Spotify Annoy via RcppAnnoy. Good CPU baseline with moderate memory use.",
    availability = annoy_available,
    context_available = function(ctx) list(
      available = identical(normalize_knn_metric(ctx$knn_metric), "euclidean"),
      message = "Annoy strategy currently supports Euclidean KNN in this benchmark."
    ),
    compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
    build_knn = annoy_knn,
    params = function(ctx) list(
      k = ctx$k,
      knn = "annoy",
      implementation = "RcppAnnoy",
      metric = normalize_knn_metric(ctx$knn_metric),
      n_trees = as.integer(n_trees),
      search_k = if (is.null(search_multiplier) || is.na(search_multiplier)) {
        "default"
      } else {
        annoy_search_k_value(ctx$k, n_trees, search_multiplier)
      },
      search_multiplier = search_label,
      pca_dims = min(ctx$pca_dims, ctx$p),
      embedding_backend = ctx$backend,
      expected = "good speed, moderate memory, approximate CPU baseline",
      quality = method_quality(ctx$method, "auto")
    ),
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    annoy_n_trees = as.integer(n_trees),
    annoy_search_multiplier = if (is.null(search_multiplier) || is.na(search_multiplier)) NA_real_ else as.numeric(search_multiplier)
  )
}

annoy_strategy_grid <- function() {
  trees <- c(10L, 25L, 50L, 100L)
  multipliers <- list(NA_real_, 2, 5)
  out <- list()
  for (n_trees in trees) {
    for (search_multiplier in multipliers) {
      out[[length(out) + 1L]] <- annoy_strategy(n_trees, search_multiplier[[1L]])
    }
  }
  out
}

hnsw_available <- function() {
  list(
    available = requireNamespace("RcppHNSW", quietly = TRUE),
    message = "R package `RcppHNSW` is not installed."
  )
}

hnsw_distance <- function(metric) {
  metric <- normalize_knn_metric(metric)
  switch(
    metric,
    euclidean = "euclidean",
    cosine = "cosine",
    ip = "ip",
    inner_product = "ip",
    NA_character_
  )
}

hnsw_knn <- function(ctx) {
  if (!requireNamespace("RcppHNSW", quietly = TRUE)) {
    stop("R package `RcppHNSW` is not installed.", call. = FALSE)
  }
  distance <- hnsw_distance(ctx$knn_metric)
  if (is.na(distance)) {
    stop("HNSW strategy supports Euclidean, cosine, and inner-product distances in this benchmark.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  build <- timed_step(RcppHNSW::hnsw_build(
    z,
    distance = distance,
    M = as.integer(ctx$hnsw_m),
    ef = as.integer(ctx$hnsw_ef_construction),
    verbose = FALSE,
    progress = "none",
    n_threads = 0L,
    random_seed = as.integer(ctx$seed)
  ))
  query <- timed_step(RcppHNSW::hnsw_search(
    z,
    build$value,
    k = as.integer(ctx$k + 1L),
    ef = as.integer(ctx$hnsw_ef_search),
    verbose = FALSE,
    progress = "none",
    n_threads = 0L
  ))
  out <- query$value
  if (any(out$dist < -1e-6, na.rm = TRUE)) {
    stop("HNSW returned negative distances beyond numerical roundoff.", call. = FALSE)
  }
  out$dist[out$dist < 0] <- 0
  profiled_knn_result(finish_external_knn(out$idx, out$dist, paste0("RcppHNSW_", distance), exact = FALSE), build$time, query$time)
}

hnsw_strategy <- function(m, ef_construction, ef_search) {
  list(
    id = paste0("hnsw_m", m, "_efc", ef_construction, "_efs", ef_search),
    family = "hnsw_knn",
    description = "Approximate nearest neighbors using an HNSW index via RcppHNSW. Strong CPU speed/accuracy trade-off.",
    availability = hnsw_available,
    context_available = function(ctx) list(
      available = !is.na(hnsw_distance(ctx$knn_metric)),
      message = "HNSW strategy supports Euclidean, cosine, and inner-product distances in this benchmark."
    ),
    compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
    build_knn = hnsw_knn,
    params = function(ctx) list(
      k = ctx$k,
      knn = "hnsw",
      implementation = "RcppHNSW",
      metric = hnsw_distance(ctx$knn_metric),
      M = as.integer(m),
      ef_construction = as.integer(ef_construction),
      ef_search = as.integer(ef_search),
      pca_dims = min(ctx$pca_dims, ctx$p),
      embedding_backend = ctx$backend,
      expected = "very strong speed/accuracy CPU trade-off",
      quality = method_quality(ctx$method, "auto")
    ),
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    hnsw_m = as.integer(m),
    hnsw_ef_construction = as.integer(ef_construction),
    hnsw_ef_search = as.integer(ef_search)
  )
}

hnsw_strategy_grid <- function() {
  out <- list()
  for (m in c(8L, 16L, 32L)) {
    for (ef_construction in c(100L, 200L, 400L)) {
      for (ef_search in c(50L, 100L, 200L)) {
        out[[length(out) + 1L]] <- hnsw_strategy(m, ef_construction, ef_search)
      }
    }
  }
  out
}

nndescent_available <- function() {
  list(
    available = requireNamespace("rnndescent", quietly = TRUE),
    message = "R package `rnndescent` is not installed."
  )
}

nndescent_metric <- function(metric) {
  metric <- normalize_knn_metric(metric)
  supported <- c(
    "braycurtis", "canberra", "chebyshev", "correlation", "cosine",
    "dice", "euclidean", "hamming", "haversine", "hellinger",
    "jaccard", "jensenshannon", "kulsinski", "sqeuclidean",
    "manhattan", "rogerstanimoto", "russellrao", "sokalmichener",
    "sokalsneath", "spearmanr", "symmetrickl", "tsss", "yule"
  )
  if (metric %in% supported) metric else NA_character_
}

nndescent_label <- function(x) {
  if (is.null(x) || is.na(x)) return("auto")
  gsub("[^A-Za-z0-9]+", "p", format(x, scientific = FALSE, trim = TRUE))
}

nndescent_max_candidates_value <- function(k, rho, max_candidates) {
  if (!is.null(max_candidates) && !is.na(max_candidates)) {
    return(as.integer(max_candidates))
  }
  if (is.null(rho) || is.na(rho)) return(NA_integer_)
  as.integer(max(1L, ceiling(as.numeric(rho) * k)))
}

nndescent_knn <- function(ctx) {
  if (!requireNamespace("rnndescent", quietly = TRUE)) {
    stop("R package `rnndescent` is not installed.", call. = FALSE)
  }
  metric <- nndescent_metric(ctx$knn_metric)
  if (is.na(metric)) {
    stop("NN-descent strategy does not support this metric through rnndescent.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  max_candidates <- nndescent_max_candidates_value(ctx$k, ctx$nndescent_rho, ctx$nndescent_max_candidates)
  args <- list(
    data = z,
    k = as.integer(ctx$k + 1L),
    metric = metric,
    init = "tree",
    n_iters = as.integer(ctx$nndescent_n_iters),
    delta = as.numeric(ctx$nndescent_delta),
    low_memory = TRUE,
    n_threads = 0L,
    verbose = FALSE,
    progress = "bar"
  )
  if (!is.na(max_candidates)) {
    args$max_candidates <- as.integer(max_candidates)
  }
  out <- do.call(rnndescent::rnnd_knn, args)
  if (any(out$dist < -1e-6, na.rm = TRUE)) {
    stop("NN-descent returned negative distances beyond numerical roundoff.", call. = FALSE)
  }
  out$dist[out$dist < 0] <- 0
  finish_external_knn(out$idx, out$dist, paste0("rnndescent_", metric), exact = FALSE)
}

nndescent_strategy <- function(n_iters, delta, rho = NA_real_, max_candidates = NA_integer_) {
  rho_label <- nndescent_label(rho)
  candidates_label <- nndescent_label(max_candidates)
  id <- paste0(
    "nnd_i", n_iters,
    "_d", nndescent_label(delta),
    "_rho", rho_label,
    "_mc", candidates_label
  )
  list(
    id = id,
    family = "nndescent_knn",
    description = "Approximate nearest neighbors using nearest-neighbor descent via rnndescent. Important UMAP-style large-scale approximation.",
    availability = nndescent_available,
    context_available = function(ctx) list(
      available = !is.na(nndescent_metric(ctx$knn_metric)),
      message = "NN-descent strategy does not support this metric through rnndescent."
    ),
    compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
    build_knn = nndescent_knn,
    params = function(ctx) {
      effective_max_candidates <- nndescent_max_candidates_value(ctx$k, rho, max_candidates)
      list(
        k = ctx$k,
        knn = "nndescent",
        implementation = "rnndescent",
        metric = nndescent_metric(ctx$knn_metric),
        init = "tree",
        n_iters = as.integer(n_iters),
        delta = as.numeric(delta),
        rho = if (is.na(rho)) NA_real_ else as.numeric(rho),
        max_candidates = if (is.na(max_candidates)) NA_integer_ else as.integer(max_candidates),
        effective_max_candidates = effective_max_candidates,
        low_memory = TRUE,
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        expected = "good large-scale approximation, UMAP-style neighbor graph",
        quality = method_quality(ctx$method, "auto")
      )
    },
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    nndescent_n_iters = as.integer(n_iters),
    nndescent_delta = as.numeric(delta),
    nndescent_rho = if (is.null(rho) || is.na(rho)) NA_real_ else as.numeric(rho),
    nndescent_max_candidates = if (is.null(max_candidates) || is.na(max_candidates)) NA_integer_ else as.integer(max_candidates)
  )
}

nndescent_strategy_grid <- function() {
  out <- list()
  n_iters_values <- c(5L, 10L, 20L)
  delta_values <- c(0.01, 0.001)
  rho_values <- c(0.5, 1, 2)
  max_candidates_values <- c(30L, 60L, 120L)
  for (n_iters in n_iters_values) {
    for (delta in delta_values) {
      for (rho in rho_values) {
        out[[length(out) + 1L]] <- nndescent_strategy(n_iters, delta, rho = rho, max_candidates = NA_integer_)
      }
      for (max_candidates in max_candidates_values) {
        out[[length(out) + 1L]] <- nndescent_strategy(n_iters, delta, rho = NA_real_, max_candidates = max_candidates)
      }
    }
  }
  out
}

faiss_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(list(available = FALSE, message = "R package `reticulate` is not installed."))
  }
  ok <- tryCatch({
    reticulate::import("faiss", delay_load = FALSE)
    TRUE
  }, error = function(e) FALSE)
  list(
    available = ok,
    message = "Python package `faiss-cpu` is not available through reticulate."
  )
}

faiss_gpu_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(list(available = FALSE, message = "R package `reticulate` is not installed."))
  }
  out <- tryCatch({
    faiss <- reticulate::import("faiss", delay_load = FALSE)
    has_resources <- reticulate::py_has_attr(faiss, "StandardGpuResources")
    has_transfer <- reticulate::py_has_attr(faiss, "index_cpu_to_gpu")
    gpu_count <- if (reticulate::py_has_attr(faiss, "get_num_gpus")) {
      as.integer(faiss$get_num_gpus())
    } else {
      0L
    }
    message <- if (has_resources && has_transfer && gpu_count > 0L) {
      "FAISS GPU support is available."
    } else if (gpu_count > 0L) {
      "Python FAISS sees CUDA GPUs, but the FAISS GPU index API is not available."
    } else {
      "Python FAISS was found, but no FAISS CUDA GPU is available."
    }
    list(
      available = has_resources && has_transfer && gpu_count > 0L,
      message = message
    )
  }, error = function(e) {
    list(available = FALSE, message = "Python FAISS with CUDA GPU support is not available through reticulate.")
  })
  out
}

cuml_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(list(available = FALSE, message = "R package `reticulate` is not installed."))
  }
  out <- tryCatch({
    neighbors <- reticulate::import("cuml.neighbors", delay_load = FALSE)
    cupy <- reticulate::import("cupy", delay_load = FALSE)
    has_nn <- reticulate::py_has_attr(neighbors, "NearestNeighbors")
    gpu_count <- as.integer(cupy$cuda$runtime$getDeviceCount())
    if (has_nn && gpu_count > 0L) {
      list(available = TRUE, message = "cuML NearestNeighbors is available on CUDA.")
    } else if (gpu_count > 0L) {
      list(available = FALSE, message = "cuML was found, but cuml.neighbors.NearestNeighbors is not available.")
    } else {
      list(available = FALSE, message = "cuML/CuPy was found, but no CUDA GPU is available.")
    }
  }, error = function(e) {
    list(available = FALSE, message = "Python RAPIDS cuML NearestNeighbors is not available through reticulate.")
  })
  out
}

faiss_metric <- function(metric) {
  metric <- normalize_knn_metric(metric)
  switch(metric, euclidean = "l2", NA_character_)
}

cuml_metric <- function(metric) {
  metric <- normalize_knn_metric(metric)
  switch(metric, euclidean = "euclidean", NA_character_)
}

faiss_effective_dim <- function(ctx) {
  rank <- min(ctx$p, ctx$n - 1L, ctx$pca_dims)
  if (rank < 1L || rank >= ctx$p) ctx$p else rank
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

faiss_threads <- function() {
  threads <- parallel::detectCores(logical = TRUE)
  if (!is.finite(threads) || threads < 1L) 1L else as.integer(threads)
}

faiss_set_threads <- function(faiss) {
  threads <- faiss_threads()
  try(faiss$omp_set_num_threads(threads), silent = TRUE)
  threads
}

faiss_numpy_matrix <- function(x) {
  np <- reticulate::import("numpy", delay_load = FALSE)
  np$array(x, dtype = "float32", order = "C")
}

cuml_gpu_matrix <- function(x) {
  cupy <- reticulate::import("cupy", delay_load = FALSE)
  cupy$array(x, dtype = "float32")
}

cuml_to_matrix <- function(x) {
  if (reticulate::py_has_attr(x, "get")) {
    return(as.matrix(x$get()))
  }
  cupy <- reticulate::import("cupy", delay_load = FALSE)
  as.matrix(cupy$asnumpy(x))
}

faiss_gpu_resources <- function(faiss, temp_memory_mb = NA_integer_) {
  res <- faiss$StandardGpuResources()
  if (!is.null(temp_memory_mb) && !is.na(temp_memory_mb) && temp_memory_mb > 0L) {
    try(res$setTempMemory(as.integer(temp_memory_mb) * 1024L * 1024L), silent = TRUE)
  }
  res
}

faiss_to_gpu <- function(faiss, resources, device, index, use_float16 = FALSE) {
  device <- as.integer(device)
  if (reticulate::py_has_attr(faiss, "GpuClonerOptions")) {
    options <- faiss$GpuClonerOptions()
    if (reticulate::py_has_attr(options, "useFloat16")) {
      options$useFloat16 <- isTRUE(use_float16)
    }
    return(faiss$index_cpu_to_gpu(resources, device, index, options))
  }
  faiss$index_cpu_to_gpu(resources, device, index)
}

faiss_search_has_missing <- function(search_result) {
  indices <- as.matrix(search_result[[2L]])
  any(!is.finite(indices)) || any(indices < 0L, na.rm = TRUE)
}

faiss_search <- function(index, xb, query_k, retry_nprobe = NA_integer_) {
  search_result <- index$search(xb, query_k)
  if (!is.na(retry_nprobe) && faiss_search_has_missing(search_result)) {
    index$nprobe <- as.integer(retry_nprobe)
    search_result <- index$search(xb, query_k)
  }
  search_result
}

faiss_remove_self <- function(indices, distances, k) {
  n <- nrow(indices)
  out_indices <- matrix(0L, n, k)
  out_distances <- matrix(0, n, k)
  storage.mode(out_indices) <- "integer"
  storage.mode(out_distances) <- "double"
  for (i in seq_len(n)) {
    keep <- which(indices[i, ] != i)
    if (length(keep) < k) {
      stop("FAISS returned fewer non-self neighbors than requested.", call. = FALSE)
    }
    keep <- keep[seq_len(k)]
    out_indices[i, ] <- indices[i, keep]
    out_distances[i, ] <- distances[i, keep]
  }
  list(indices = out_indices, distances = out_distances)
}

faiss_search_to_knn <- function(search_result, k, backend, exact) {
  distances <- as.matrix(search_result[[1L]])
  indices <- as.matrix(search_result[[2L]]) + 1L
  if (any(!is.finite(indices)) || any(indices < 1L, na.rm = TRUE)) {
    stop("FAISS returned invalid or missing neighbor indices.", call. = FALSE)
  }
  distances <- sqrt(pmax(as.matrix(distances), 0))
  stripped <- faiss_remove_self(indices, distances, k)
  finish_external_knn(stripped$indices, stripped$distances, backend, exact = exact)
}

cuml_search_to_knn <- function(search_result, k, backend, exact) {
  distances <- cuml_to_matrix(search_result[[1L]])
  indices <- cuml_to_matrix(search_result[[2L]]) + 1L
  if (any(!is.finite(indices)) || any(indices < 1L, na.rm = TRUE)) {
    stop("cuML NearestNeighbors returned invalid or missing neighbor indices.", call. = FALSE)
  }
  distances <- pmax(as.matrix(distances), 0)
  stripped <- faiss_remove_self(indices, distances, k)
  finish_external_knn(stripped$indices, stripped$distances, backend, exact = exact)
}

cuml_nearest_neighbors_model <- function(neighbors, ctx) {
  algorithm <- ctx$cuml_algorithm
  base <- list(
    n_neighbors = as.integer(ctx$k + 1L),
    algorithm = algorithm,
    metric = cuml_metric(ctx$knn_metric)
  )
  variants <- list(base)
  if (algorithm %in% c("ivfflat", "ivfpq")) {
    nlist <- faiss_effective_nlist(ctx$n, ctx$cuml_nlist)
    nprobe <- faiss_effective_nprobe(ctx$cuml_nprobe, nlist)
    variants <- list(
      c(base, list(nlist = nlist, nprobe = nprobe)),
      c(base, list(n_clusters = nlist, n_probes = nprobe)),
      c(base, list(n_lists = nlist, n_probes = nprobe)),
      base
    )
  }
  last_error <- NULL
  for (args in variants) {
    model <- tryCatch(do.call(neighbors$NearestNeighbors, args), error = function(e) {
      last_error <<- conditionMessage(e)
      NULL
    })
    if (!is.null(model)) return(model)
  }
  stop("Could not construct cuML NearestNeighbors model: ", last_error, call. = FALSE)
}

faiss_knn <- function(ctx) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  faiss <- tryCatch(
    reticulate::import("faiss", delay_load = FALSE),
    error = function(e) stop("Python package `faiss-cpu` is not available through reticulate.", call. = FALSE)
  )
  if (is.na(faiss_metric(ctx$knn_metric))) {
    stop("FAISS CPU strategy currently supports Euclidean KNN in this benchmark.", call. = FALSE)
  }
  faiss_set_threads(faiss)
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  xb <- faiss_numpy_matrix(z)
  n <- nrow(z)
  d <- ncol(z)
  query_k <- as.integer(ctx$k + 1L)
  index_type <- ctx$faiss_index_type

  if (identical(index_type, "flat")) {
    build <- timed_step({
      index <- faiss$IndexFlatL2(as.integer(d))
      index$add(xb)
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_IndexFlatL2", exact = TRUE), build$time, query$time))
  }

  if (identical(index_type, "ivf")) {
    nlist <- faiss_effective_nlist(n, ctx$faiss_nlist)
    nprobe <- faiss_effective_nprobe(ctx$faiss_nprobe, nlist)
    build <- timed_step({
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      index <- faiss$IndexIVFFlat(quantizer, as.integer(d), nlist, faiss$METRIC_L2)
      index$train(xb)
      index$add(xb)
      index$nprobe <- nprobe
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k, retry_nprobe = nlist))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_IVFFlat", exact = FALSE), build$time, query$time))
  }

  if (identical(index_type, "ivfpq")) {
    pq_m <- as.integer(ctx$faiss_pq_m)
    if (d %% pq_m != 0L) {
      stop("FAISS IVF-PQ requires the PCA/input dimension to be divisible by `pq_m`.", call. = FALSE)
    }
    nlist <- faiss_effective_nlist(n, ctx$faiss_nlist)
    nprobe <- faiss_effective_nprobe(ctx$faiss_nprobe, nlist)
    nbits <- faiss_effective_nbits(n, ctx$faiss_pq_nbits)
    build <- timed_step({
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      index <- faiss$IndexIVFPQ(
        quantizer,
        as.integer(d),
        nlist,
        pq_m,
        nbits,
        faiss$METRIC_L2
      )
      index$train(xb)
      index$add(xb)
      index$nprobe <- nprobe
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k, retry_nprobe = nlist))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_IVFPQ", exact = FALSE), build$time, query$time))
  }

  if (identical(index_type, "hnsw")) {
    build <- timed_step({
      index <- faiss$IndexHNSWFlat(as.integer(d), as.integer(ctx$faiss_hnsw_m), faiss$METRIC_L2)
      index$hnsw$efConstruction <- as.integer(ctx$faiss_hnsw_ef_construction)
      index$hnsw$efSearch <- as.integer(ctx$faiss_hnsw_ef_search)
      index$add(xb)
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_HNSWFlat", exact = FALSE), build$time, query$time))
  }

  stop("Unknown FAISS index type: ", index_type, call. = FALSE)
}

faiss_gpu_knn <- function(ctx) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  faiss <- tryCatch(
    reticulate::import("faiss", delay_load = FALSE),
    error = function(e) stop("Python FAISS with CUDA GPU support is not available through reticulate.", call. = FALSE)
  )
  availability <- faiss_gpu_available()
  if (!isTRUE(availability$available)) {
    stop(availability$message, call. = FALSE)
  }
  if (is.na(faiss_metric(ctx$knn_metric))) {
    stop("FAISS GPU strategy currently supports Euclidean KNN in this benchmark.", call. = FALSE)
  }
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  xb <- faiss_numpy_matrix(z)
  n <- nrow(z)
  d <- ncol(z)
  query_k <- as.integer(ctx$k + 1L)
  index_type <- ctx$faiss_index_type
  resources <- faiss_gpu_resources(faiss, ctx$faiss_gpu_temp_memory_mb)
  device <- as.integer(ctx$faiss_gpu_device)
  use_float16 <- isTRUE(ctx$faiss_gpu_use_float16)

  if (identical(index_type, "flat")) {
    build <- timed_step({
      cpu_index <- faiss$IndexFlatL2(as.integer(d))
      index <- faiss_to_gpu(faiss, resources, device, cpu_index, use_float16)
      index$add(xb)
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_gpu_IndexFlatL2", exact = TRUE), build$time, query$time))
  }

  if (identical(index_type, "ivf")) {
    nlist <- faiss_effective_nlist(n, ctx$faiss_nlist)
    nprobe <- faiss_effective_nprobe(ctx$faiss_nprobe, nlist)
    build <- timed_step({
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      cpu_index <- faiss$IndexIVFFlat(quantizer, as.integer(d), nlist, faiss$METRIC_L2)
      index <- faiss_to_gpu(faiss, resources, device, cpu_index, use_float16)
      index$train(xb)
      index$add(xb)
      index$nprobe <- nprobe
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k, retry_nprobe = nlist))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_gpu_IVFFlat", exact = FALSE), build$time, query$time))
  }

  if (identical(index_type, "ivfpq")) {
    pq_m <- as.integer(ctx$faiss_pq_m)
    if (d %% pq_m != 0L) {
      stop("FAISS GPU IVF-PQ requires the PCA/input dimension to be divisible by `pq_m`.", call. = FALSE)
    }
    nlist <- faiss_effective_nlist(n, ctx$faiss_nlist)
    nprobe <- faiss_effective_nprobe(ctx$faiss_nprobe, nlist)
    nbits <- faiss_effective_nbits(n, ctx$faiss_pq_nbits)
    build <- timed_step({
      quantizer <- faiss$IndexFlatL2(as.integer(d))
      cpu_index <- faiss$IndexIVFPQ(
        quantizer,
        as.integer(d),
        nlist,
        pq_m,
        nbits,
        faiss$METRIC_L2
      )
      index <- faiss_to_gpu(faiss, resources, device, cpu_index, use_float16)
      index$train(xb)
      index$add(xb)
      index$nprobe <- nprobe
      index
    })
    query <- timed_step(faiss_search(build$value, xb, query_k, retry_nprobe = nlist))
    return(profiled_knn_result(faiss_search_to_knn(query$value, ctx$k, "faiss_gpu_IVFPQ", exact = FALSE), build$time, query$time))
  }

  stop("Unknown FAISS GPU index type: ", index_type, call. = FALSE)
}

cuml_knn <- function(ctx) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("R package `reticulate` is not installed.", call. = FALSE)
  }
  availability <- cuml_available()
  if (!isTRUE(availability$available)) {
    stop(availability$message, call. = FALSE)
  }
  if (is.na(cuml_metric(ctx$knn_metric))) {
    stop("cuML NearestNeighbors strategy currently supports Euclidean KNN in this benchmark.", call. = FALSE)
  }
  cupy <- reticulate::import("cupy", delay_load = FALSE)
  neighbors <- reticulate::import("cuml.neighbors", delay_load = FALSE)
  try(cupy$cuda$Device(as.integer(ctx$cuml_device))$use(), silent = TRUE)
  z <- kdtree_space(ctx$x, ctx$pca_dims)
  xg <- cuml_gpu_matrix(z)
  model <- cuml_nearest_neighbors_model(neighbors, ctx)
  build <- timed_step(model$fit(xg))
  query <- timed_step(model$kneighbors(
    xg,
    n_neighbors = as.integer(ctx$k + 1L),
    return_distance = TRUE
  ))
  profiled_knn_result(cuml_search_to_knn(
    query$value,
    ctx$k,
    paste0("cuml_NearestNeighbors_", ctx$cuml_algorithm),
    exact = identical(ctx$cuml_algorithm, "brute")
  ), build$time, query$time)
}

faiss_strategy <- function(index_type,
                           nlist = NA_integer_,
                           nprobe = NA_integer_,
                           pq_m = NA_integer_,
                           pq_nbits = 8L,
                           hnsw_m = NA_integer_,
                           hnsw_ef_construction = NA_integer_,
                           hnsw_ef_search = NA_integer_) {
  id <- switch(
    index_type,
    flat = "faiss_flat",
    ivf = paste0("faiss_ivf_nlist", nlist, "_nprobe", nprobe),
    ivfpq = paste0("faiss_ivfpq_nlist", nlist, "_nprobe", nprobe, "_m", pq_m),
    hnsw = paste0("faiss_hnsw_m", hnsw_m, "_efc", hnsw_ef_construction, "_efs", hnsw_ef_search),
    stop("Unknown FAISS index type.", call. = FALSE)
  )
  list(
    id = id,
    family = "faiss_knn",
    description = "FAISS CPU nearest-neighbor index. Tests IndexFlat, IVF, IVF-PQ, and HNSW-FAISS as scalable KNN sources.",
    availability = faiss_available,
    context_available = function(ctx) {
      metric_ok <- !is.na(faiss_metric(ctx$knn_metric))
      if (!metric_ok) {
        return(list(available = FALSE, message = "FAISS CPU strategy currently supports Euclidean KNN in this benchmark."))
      }
      if (identical(index_type, "ivfpq")) {
        d <- faiss_effective_dim(ctx)
        pq_ok <- d %% as.integer(pq_m) == 0L
        if (!pq_ok) {
          return(list(
            available = FALSE,
            message = paste0("FAISS IVF-PQ requires effective dimension ", d, " to be divisible by pq_m=", pq_m, ".")
          ))
        }
      }
      list(available = TRUE, message = NA_character_)
    },
    compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
    build_knn = faiss_knn,
    params = function(ctx) {
      effective_nlist <- if (is.na(nlist)) NA_integer_ else faiss_effective_nlist(ctx$n, nlist)
      effective_nprobe <- if (is.na(nprobe) || is.na(effective_nlist)) NA_integer_ else faiss_effective_nprobe(nprobe, effective_nlist)
      effective_nbits <- if (identical(index_type, "ivfpq")) faiss_effective_nbits(ctx$n, pq_nbits) else NA_integer_
      list(
        k = ctx$k,
        knn = "faiss_cpu",
        implementation = "faiss-cpu",
        index = index_type,
        metric = "euclidean",
        nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
        effective_nlist = effective_nlist,
        nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
        effective_nprobe = effective_nprobe,
        retry_full_nprobe_if_short = index_type %in% c("ivf", "ivfpq"),
        pq_m = if (is.na(pq_m)) NA_integer_ else as.integer(pq_m),
        pq_nbits = if (identical(index_type, "ivfpq")) as.integer(pq_nbits) else NA_integer_,
        effective_pq_nbits = effective_nbits,
        hnsw_m = if (is.na(hnsw_m)) NA_integer_ else as.integer(hnsw_m),
        hnsw_ef_construction = if (is.na(hnsw_ef_construction)) NA_integer_ else as.integer(hnsw_ef_construction),
        hnsw_ef_search = if (is.na(hnsw_ef_search)) NA_integer_ else as.integer(hnsw_ef_search),
        faiss_threads = faiss_threads(),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        expected = "good large-dataset CPU nearest-neighbor baseline",
        quality = method_quality(ctx$method, "auto")
      )
    },
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    faiss_index_type = index_type,
    faiss_nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
    faiss_nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
    faiss_pq_m = if (is.na(pq_m)) NA_integer_ else as.integer(pq_m),
    faiss_pq_nbits = as.integer(pq_nbits),
    faiss_hnsw_m = if (is.na(hnsw_m)) NA_integer_ else as.integer(hnsw_m),
    faiss_hnsw_ef_construction = if (is.na(hnsw_ef_construction)) NA_integer_ else as.integer(hnsw_ef_construction),
    faiss_hnsw_ef_search = if (is.na(hnsw_ef_search)) NA_integer_ else as.integer(hnsw_ef_search)
  )
}

faiss_strategy_grid <- function() {
  list(
    faiss_strategy("flat"),
    faiss_strategy("ivf", nlist = 16L, nprobe = 4L),
    faiss_strategy("ivf", nlist = 32L, nprobe = 8L),
    faiss_strategy("ivf", nlist = 64L, nprobe = 16L),
    faiss_strategy("ivfpq", nlist = 16L, nprobe = 4L, pq_m = 2L, pq_nbits = 8L),
    faiss_strategy("ivfpq", nlist = 32L, nprobe = 8L, pq_m = 2L, pq_nbits = 8L),
    faiss_strategy("hnsw", hnsw_m = 16L, hnsw_ef_construction = 200L, hnsw_ef_search = 50L),
    faiss_strategy("hnsw", hnsw_m = 16L, hnsw_ef_construction = 200L, hnsw_ef_search = 100L),
    faiss_strategy("hnsw", hnsw_m = 32L, hnsw_ef_construction = 200L, hnsw_ef_search = 50L),
    faiss_strategy("hnsw", hnsw_m = 32L, hnsw_ef_construction = 200L, hnsw_ef_search = 100L)
  )
}

faiss_gpu_strategy <- function(index_type,
                                nlist = NA_integer_,
                                nprobe = NA_integer_,
                                pq_m = NA_integer_,
                                pq_nbits = 8L,
                                device = 0L,
                                use_float16 = FALSE,
                                temp_memory_mb = NA_integer_) {
  id <- switch(
    index_type,
    flat = "faiss_gpu_flat",
    ivf = paste0("faiss_gpu_ivf_nlist", nlist, "_nprobe", nprobe),
    ivfpq = paste0("faiss_gpu_ivfpq_nlist", nlist, "_nprobe", nprobe, "_m", pq_m),
    stop("Unknown FAISS GPU index type.", call. = FALSE)
  )
  list(
    id = id,
    family = "faiss_gpu_knn",
    description = "FAISS CUDA nearest-neighbor index. Strong GPU KNN baseline using IndexFlat, IVF, and IVF-PQ; never falls back to CPU.",
    availability = faiss_gpu_available,
    unavailable_status = "backend_unavailable",
    context_available = function(ctx) {
      metric_ok <- !is.na(faiss_metric(ctx$knn_metric))
      if (!metric_ok) {
        return(list(available = FALSE, message = "FAISS GPU strategy currently supports Euclidean KNN in this benchmark."))
      }
      if (!identical(ctx$backend, "cuda")) {
        return(list(available = FALSE, message = "FAISS GPU KNN is CUDA-only; request `backend = cuda`."))
      }
      if (identical(index_type, "ivfpq")) {
        d <- faiss_effective_dim(ctx)
        pq_ok <- d %% as.integer(pq_m) == 0L
        if (!pq_ok) {
          return(list(
            available = FALSE,
            message = paste0("FAISS GPU IVF-PQ requires effective dimension ", d, " to be divisible by pq_m=", pq_m, ".")
          ))
        }
      }
      list(available = TRUE, message = NA_character_)
    },
    compatible = function(method, backend) identical(backend, "cuda"),
    build_knn = faiss_gpu_knn,
    params = function(ctx) {
      effective_nlist <- if (is.na(nlist)) NA_integer_ else faiss_effective_nlist(ctx$n, nlist)
      effective_nprobe <- if (is.na(nprobe) || is.na(effective_nlist)) NA_integer_ else faiss_effective_nprobe(nprobe, effective_nlist)
      effective_nbits <- if (identical(index_type, "ivfpq")) faiss_effective_nbits(ctx$n, pq_nbits) else NA_integer_
      list(
        k = ctx$k,
        knn = "faiss_gpu",
        implementation = "faiss_cuda_python",
        index = index_type,
        metric = "euclidean",
        device = as.integer(device),
        nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
        effective_nlist = effective_nlist,
        nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
        effective_nprobe = effective_nprobe,
        retry_full_nprobe_if_short = index_type %in% c("ivf", "ivfpq"),
        pq_m = if (is.na(pq_m)) NA_integer_ else as.integer(pq_m),
        pq_nbits = if (identical(index_type, "ivfpq")) as.integer(pq_nbits) else NA_integer_,
        effective_pq_nbits = effective_nbits,
        use_float16 = isTRUE(use_float16),
        temp_memory_mb = if (is.na(temp_memory_mb)) NA_integer_ else as.integer(temp_memory_mb),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        expected = "strong CUDA GPU KNN baseline",
        quality = method_quality(ctx$method, "auto")
      )
    },
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    faiss_index_type = index_type,
    faiss_nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
    faiss_nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
    faiss_pq_m = if (is.na(pq_m)) NA_integer_ else as.integer(pq_m),
    faiss_pq_nbits = as.integer(pq_nbits),
    faiss_gpu_device = as.integer(device),
    faiss_gpu_use_float16 = isTRUE(use_float16),
    faiss_gpu_temp_memory_mb = if (is.na(temp_memory_mb)) NA_integer_ else as.integer(temp_memory_mb)
  )
}

faiss_gpu_strategy_grid <- function() {
  list(
    faiss_gpu_strategy("flat"),
    faiss_gpu_strategy("ivf", nlist = 64L, nprobe = 16L),
    faiss_gpu_strategy("ivf", nlist = 256L, nprobe = 32L),
    faiss_gpu_strategy("ivfpq", nlist = 256L, nprobe = 32L, pq_m = 2L, pq_nbits = 8L),
    faiss_gpu_strategy("ivfpq", nlist = 1024L, nprobe = 64L, pq_m = 2L, pq_nbits = 8L)
  )
}

cuml_label <- function(x) {
  if (is.null(x) || is.na(x)) return("auto")
  gsub("[^A-Za-z0-9]+", "", format(x, scientific = FALSE, trim = TRUE))
}

cuml_strategy <- function(algorithm, nlist = NA_integer_, nprobe = NA_integer_, device = 0L) {
  id <- if (algorithm %in% c("ivfflat", "ivfpq")) {
    paste0("cuml_nn_", algorithm, "_nlist", cuml_label(nlist), "_nprobe", cuml_label(nprobe))
  } else {
    paste0("cuml_nn_", algorithm)
  }
  list(
    id = id,
    family = "cuml_knn",
    description = "RAPIDS cuML NearestNeighbors KNN source. CUDA-only and useful when the full cuML/RAPIDS pipeline is available.",
    availability = cuml_available,
    unavailable_status = "backend_unavailable",
    context_available = function(ctx) {
      metric_ok <- !is.na(cuml_metric(ctx$knn_metric))
      if (!metric_ok) {
        return(list(available = FALSE, message = "cuML NearestNeighbors strategy currently supports Euclidean KNN in this benchmark."))
      }
      if (!identical(ctx$backend, "cuda")) {
        return(list(available = FALSE, message = "cuML NearestNeighbors is CUDA-only; request `backend = cuda`."))
      }
      list(available = TRUE, message = NA_character_)
    },
    compatible = function(method, backend) identical(backend, "cuda"),
    build_knn = cuml_knn,
    params = function(ctx) {
      effective_nlist <- if (is.na(nlist)) NA_integer_ else faiss_effective_nlist(ctx$n, nlist)
      effective_nprobe <- if (is.na(nprobe) || is.na(effective_nlist)) NA_integer_ else faiss_effective_nprobe(nprobe, effective_nlist)
      list(
        k = ctx$k,
        knn = "cuml_nearest_neighbors",
        implementation = "rapids_cuml",
        algorithm = algorithm,
        metric = "euclidean",
        device = as.integer(device),
        nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
        effective_nlist = effective_nlist,
        nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
        effective_nprobe = effective_nprobe,
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        expected = "strong CUDA KNN baseline if RAPIDS cuML is installed",
        quality = method_quality(ctx$method, "auto")
      )
    },
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    ),
    cuml_algorithm = algorithm,
    cuml_nlist = if (is.na(nlist)) NA_integer_ else as.integer(nlist),
    cuml_nprobe = if (is.na(nprobe)) NA_integer_ else as.integer(nprobe),
    cuml_device = as.integer(device)
  )
}

cuml_strategy_grid <- function() {
  list(
    cuml_strategy("auto"),
    cuml_strategy("brute"),
    cuml_strategy("ivfflat", nlist = 256L, nprobe = 32L),
    cuml_strategy("ivfflat", nlist = 1024L, nprobe = 64L),
    cuml_strategy("ivfpq", nlist = 256L, nprobe = 32L),
    cuml_strategy("ivfpq", nlist = 1024L, nprobe = 64L)
  )
}

graph_strategy <- function(id, description, transform_knn, params = list(), context_available = NULL) {
  list(
    id = id,
    family = "graph_construction",
    description = description,
    compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
    context_available = context_available,
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    transform_knn = transform_knn,
    params = function(ctx) c(
      list(
        k = ctx$k,
        base_knn = "exact_cpu",
        graph_construction = id,
        graph_effective_k = ctx$graph_effective_k,
        graph_edge_retention = ctx$graph_edge_retention,
        graph_mean_degree = ctx$graph_mean_degree,
        graph_isolated_fraction = ctx$graph_isolated_fraction,
        graph_padding_fraction = ctx$graph_padding_fraction,
        graph_storage_format = ctx$graph_storage_format,
        graph_sparse_nnz = ctx$graph_sparse_nnz,
        graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
        graph_sparse_r_memory_mb = ctx$graph_sparse_r_memory_mb,
        graph_dense_knn_memory_mb = ctx$graph_dense_knn_memory_mb,
        graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
        graph_sparse_r_memory_ratio = ctx$graph_sparse_r_memory_ratio,
        graph_sparse_prune_weight = ctx$graph_sparse_prune_weight,
        graph_mean_jaccard = ctx$graph_mean_jaccard,
        graph_zero_jaccard_fraction = ctx$graph_zero_jaccard_fraction,
        graph_snn_k = ctx$graph_snn_k,
        graph_snn_prune_threshold = ctx$graph_snn_prune_threshold,
        graph_mean_snn_weight = ctx$graph_mean_snn_weight,
        graph_zero_snn_fraction = ctx$graph_zero_snn_fraction,
        graph_adaptive_min_k = ctx$graph_adaptive_min_k,
        graph_adaptive_max_k = ctx$graph_adaptive_max_k,
        graph_adaptive_mean_k = ctx$graph_adaptive_mean_k,
        graph_adaptive_density_cor = ctx$graph_adaptive_density_cor,
        graph_distance_prune_drop_fraction = ctx$graph_distance_prune_drop_fraction,
        graph_distance_prune_percentile = ctx$graph_distance_prune_percentile,
        graph_distance_prune_threshold_mean = ctx$graph_distance_prune_threshold_mean,
        graph_distance_prune_removed_edges_mean = ctx$graph_distance_prune_removed_edges_mean,
        graph_sparsification_method = ctx$graph_sparsification_method,
        graph_sparsification_keep_fraction = ctx$graph_sparsification_keep_fraction,
        graph_sparsification_spectral_rank = ctx$graph_sparsification_spectral_rank,
        graph_sparsification_leverage_mean = ctx$graph_sparsification_leverage_mean,
        graph_mst_rescue_enabled = ctx$graph_mst_rescue_enabled,
        graph_mst_rescue_components_before = ctx$graph_mst_rescue_components_before,
        graph_mst_rescue_components_after = ctx$graph_mst_rescue_components_after,
        graph_mst_rescue_added_directed_edges = ctx$graph_mst_rescue_added_directed_edges,
        graph_density_correction_method = ctx$graph_density_correction_method,
        graph_density_scale_cv = ctx$graph_density_scale_cv,
        graph_density_correction_mean = ctx$graph_density_correction_mean,
        graph_density_corrected_distance_scale_cor = ctx$graph_density_corrected_distance_scale_cor,
        umap_graph_set_op_mix_ratio = ctx$umap_graph_set_op_mix_ratio,
        umap_graph_local_connectivity = ctx$umap_graph_local_connectivity,
        umap_graph_weight_power = ctx$umap_graph_weight_power,
        umap_graph_target_scale = ctx$umap_graph_target_scale,
        umap_graph_distance_transform = ctx$umap_graph_distance_transform,
        umap_graph_mean_weight = ctx$umap_graph_mean_weight,
        graph_edge_sampling_method = ctx$graph_edge_sampling_method,
        graph_edge_sampling_fraction = ctx$graph_edge_sampling_fraction,
        graph_edge_sampling_weight_power = ctx$graph_edge_sampling_weight_power,
        graph_edge_sampling_include_top = ctx$graph_edge_sampling_include_top,
        graph_edge_sampling_target_scale = ctx$graph_edge_sampling_target_scale,
        graph_edge_sampling_mean_selected_weight = ctx$graph_edge_sampling_mean_selected_weight,
        graph_edge_sampling_mean_candidate_weight = ctx$graph_edge_sampling_mean_candidate_weight,
        graph_edge_sampling_selected_to_candidate_weight_ratio = ctx$graph_edge_sampling_selected_to_candidate_weight_ratio,
        graph_tsne_affinity_mode = ctx$graph_tsne_affinity_mode,
        graph_tsne_affinity_perplexities = ctx$graph_tsne_affinity_perplexities,
        graph_tsne_affinity_temperature = ctx$graph_tsne_affinity_temperature,
        graph_tsne_affinity_effective_perplexity_mean = ctx$graph_tsne_affinity_effective_perplexity_mean,
        graph_multiscale_perplexities = ctx$graph_multiscale_perplexities,
        graph_multiscale_transfer_mode = ctx$graph_multiscale_transfer_mode,
        graph_multiscale_required_k = ctx$graph_multiscale_required_k,
        graph_multiscale_effective_k_values = ctx$graph_multiscale_effective_k_values,
        pacmap_transfer_mode = ctx$pacmap_transfer_mode,
        pacmap_auxiliary_pair_family = ctx$pacmap_auxiliary_pair_family,
        pacmap_mid_near_pairs_per_point = ctx$pacmap_mid_near_pairs_per_point,
        pacmap_mid_near_fraction = ctx$pacmap_mid_near_fraction,
        pacmap_mid_near_requested_fraction = ctx$pacmap_mid_near_requested_fraction,
        pacmap_mid_near_distance_scale = ctx$pacmap_mid_near_distance_scale,
        pacmap_mid_near_fallback_fraction = ctx$pacmap_mid_near_fallback_fraction,
        pacmap_mid_near_rank_mean = ctx$pacmap_mid_near_rank_mean,
        pacmap_mid_near_emphasis_strength = ctx$pacmap_mid_near_emphasis_strength,
        pacmap_mid_near_emphasis_distance_multiplier = ctx$pacmap_mid_near_emphasis_distance_multiplier,
        pacmap_near_ratio = ctx$pacmap_near_ratio,
        pacmap_mid_ratio = ctx$pacmap_mid_ratio,
        pacmap_far_ratio = ctx$pacmap_far_ratio,
        pacmap_near_pairs_per_point = ctx$pacmap_near_pairs_per_point,
        pacmap_mid_pairs_per_point = ctx$pacmap_mid_pairs_per_point,
        pacmap_far_pairs_per_point = ctx$pacmap_far_pairs_per_point,
        pacmap_far_pair_fraction = ctx$pacmap_far_pair_fraction,
        pacmap_far_distance_scale = ctx$pacmap_far_distance_scale,
        pacmap_far_fallback_fraction = ctx$pacmap_far_fallback_fraction,
        trimap_transfer_mode = ctx$trimap_transfer_mode,
        trimap_triplet_family = ctx$trimap_triplet_family,
        trimap_inlier_ratio = ctx$trimap_inlier_ratio,
        trimap_semihard_ratio = ctx$trimap_semihard_ratio,
        trimap_global_anchor_ratio = ctx$trimap_global_anchor_ratio,
        trimap_inlier_pairs_per_point = ctx$trimap_inlier_pairs_per_point,
        trimap_semihard_pairs_per_point = ctx$trimap_semihard_pairs_per_point,
        trimap_global_anchor_pairs_per_point = ctx$trimap_global_anchor_pairs_per_point,
        trimap_semihard_fraction = ctx$trimap_semihard_fraction,
        trimap_global_anchor_fraction = ctx$trimap_global_anchor_fraction,
        trimap_semihard_distance_scale = ctx$trimap_semihard_distance_scale,
        trimap_global_anchor_distance_scale = ctx$trimap_global_anchor_distance_scale,
        trimap_native_explicit_triplets = ctx$trimap_native_explicit_triplets,
        trimap_triplet_proxy_detail = ctx$trimap_triplet_proxy_detail,
        localmap_false_neighbor_enabled = ctx$localmap_false_neighbor_enabled,
        localmap_false_neighbor_mode = ctx$localmap_false_neighbor_mode,
        localmap_false_neighbor_transfer_mode = ctx$localmap_false_neighbor_transfer_mode,
        localmap_false_neighbor_jaccard_threshold = ctx$localmap_false_neighbor_jaccard_threshold,
        localmap_false_neighbor_distance_quantile = ctx$localmap_false_neighbor_distance_quantile,
        localmap_false_neighbor_distance_multiplier = ctx$localmap_false_neighbor_distance_multiplier,
        localmap_false_neighbor_min_keep_fraction = ctx$localmap_false_neighbor_min_keep_fraction,
        localmap_false_neighbor_removed_fraction = ctx$localmap_false_neighbor_removed_fraction,
        localmap_local_weight_enabled = ctx$localmap_local_weight_enabled,
        localmap_local_weight = ctx$localmap_local_weight,
        localmap_local_weight_mode = ctx$localmap_local_weight_mode,
        localmap_local_weight_transfer_mode = ctx$localmap_local_weight_transfer_mode,
        localmap_local_weight_jaccard_blend = ctx$localmap_local_weight_jaccard_blend,
        localmap_local_weight_mean_trust = ctx$localmap_local_weight_mean_trust,
        localmap_local_weight_mean_multiplier = ctx$localmap_local_weight_mean_multiplier,
        localmap_local_weight_distance_scale_mean = ctx$localmap_local_weight_distance_scale_mean,
        embedding_backend = ctx$backend,
        quality = method_quality(ctx$method, "auto")
      ),
      params
    ),
    run = function(ctx) fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend,
      seed = ctx$seed
    )
  )
}

snn_graph_strategy <- function(snn_k, prune_threshold) {
  snn_k <- as.integer(snn_k)
  prune_threshold <- as.numeric(prune_threshold)
  suffix <- gsub("\\.", "", sprintf("%.2f", prune_threshold))
  graph_strategy(
    paste0("graph_snn_k", snn_k, "_t", suffix),
    paste0("Shared-nearest-neighbour weighted graph with SNN_k=", snn_k, " and prune threshold=", prune_threshold, "."),
    transform_knn = function(ctx) graph_snn_weighted_knn(ctx$knn, snn_k = snn_k, prune_threshold = prune_threshold),
    params = list(
      graph_weighting = "shared_nearest_neighbour",
      snn_k = snn_k,
      prune_threshold = prune_threshold,
      primary_methods = "umap,pacmap,localmap",
      partial_methods = "tsne_affinity_weighting",
      expected_benefit = "biological_cluster_separation",
      expected_risk = "may_reduce_continuous_trajectory_preservation"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= snn_k && ctx$n * snn_k <= 500000,
      message = paste0(
        "SNN graph requires benchmark --k >= ", snn_k,
        " and n * SNN_k <= 500000 for this R-side pilot."
      )
    )
  )
}

snn_graph_strategy_grid <- function() {
  out <- list()
  for (snn_k in c(15L, 30L, 50L)) {
    for (threshold in c(0, 0.1, 0.2, 0.3)) {
      out[[length(out) + 1L]] <- snn_graph_strategy(snn_k, threshold)
    }
  }
  out
}

localmap_false_neighbor_strategy <- function(id,
                                             jaccard_threshold,
                                             distance_quantile,
                                             distance_multiplier,
                                             min_keep_fraction) {
  graph_strategy(
    paste0("localmap_false_neighbors_", id),
    paste0(
      "LocalMAP-style false-neighbour correction: remove locally long KNN edges ",
      "with weak shared-neighbour support before running the embedding objective."
    ),
    transform_knn = function(ctx) graph_localmap_false_neighbor_correct_knn(
      ctx$knn,
      jaccard_threshold = jaccard_threshold,
      distance_quantile = distance_quantile,
      distance_multiplier = distance_multiplier,
      min_keep_fraction = min_keep_fraction,
      method = ctx$method
    ),
    params = list(
      graph_weighting = "localmap_false_neighbour_correction",
      jaccard_threshold = jaccard_threshold,
      distance_quantile = distance_quantile,
      distance_multiplier = distance_multiplier,
      min_keep_fraction = min_keep_fraction,
      original_method = "LocalMAP",
      primary_methods = "localmap,umap",
      transferred_methods = "tsne,pacmap,trimap",
      expected_benefit = "improve_trustworthiness_by_reducing_false_neighbours",
      expected_risk = "may_remove_continuous_or_rare_population_edges"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 8L && ctx$n * ctx$k <= 2000000,
      message = "LocalMAP false-neighbour correction requires --k >= 8 and n * k <= 2000000 for this R-side pilot."
    )
  )
}

localmap_false_neighbor_strategy_grid <- function() {
  list(
    localmap_false_neighbor_strategy(
      "mild_j0p03_q0p75_m1p05_keep0p67",
      jaccard_threshold = 0.03,
      distance_quantile = 0.75,
      distance_multiplier = 1.05,
      min_keep_fraction = 0.67
    ),
    localmap_false_neighbor_strategy(
      "balanced_j0p08_q0p65_m1p00_keep0p50",
      jaccard_threshold = 0.08,
      distance_quantile = 0.65,
      distance_multiplier = 1.00,
      min_keep_fraction = 0.50
    ),
    localmap_false_neighbor_strategy(
      "aggressive_j0p15_q0p55_m1p00_keep0p33",
      jaccard_threshold = 0.15,
      distance_quantile = 0.55,
      distance_multiplier = 1.00,
      min_keep_fraction = 0.33
    )
  )
}

localmap_local_weight_strategy <- function(local_weight) {
  local_weight <- as.numeric(local_weight)
  label <- gsub("\\.", "p", format(local_weight, trim = TRUE, scientific = FALSE))
  graph_strategy(
    paste0("localmap_local_weight_", label),
    paste0(
      "LocalMAP-style local-loss reweighting: multiply trustworthy local relations by ",
      local_weight,
      " before transferring the graph to the embedding objective."
    ),
    transform_knn = function(ctx) graph_localmap_local_loss_reweight_knn(
      ctx$knn,
      local_weight = local_weight,
      method = ctx$method
    ),
    params = list(
      graph_weighting = "localmap_local_loss_reweighting",
      local_weight = local_weight,
      trust_score = "0.65 * local_rank_score + 0.35 * shared_neighbour_jaccard",
      original_method = "LocalMAP",
      primary_methods = "localmap",
      transferred_methods = "umap,pacmap,trimap,tsne",
      expected_benefit = "emphasize_trustworthy_local_relations",
      expected_risk = "too_large_values_can_overcluster_or_reduce_global_structure"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$n * ctx$k <= 2000000,
      message = "LocalMAP local-loss reweighting requires --k >= 4 and n * k <= 2000000 for this R-side pilot."
    )
  )
}

localmap_local_weight_strategy_grid <- function() {
  lapply(c(0.5, 1, 2, 5), localmap_local_weight_strategy)
}

adaptive_k_strategy <- function(id, min_fraction, max_fraction = 1, density_quantile = 0.5) {
  graph_strategy(
    paste0("graph_adaptive_k_", id),
    paste0(
      "Adaptive per-point KNN degree using local density: dense rows use about ",
      round(100 * min_fraction), "% of k and sparse rows use up to ",
      round(100 * max_fraction), "% of k."
    ),
    transform_knn = function(ctx) graph_adaptive_k_knn(
      ctx$knn,
      min_fraction = min_fraction,
      max_fraction = max_fraction,
      density_quantile = density_quantile
    ),
    params = list(
      graph_weighting = "adaptive_k_by_local_density",
      min_fraction = min_fraction,
      max_fraction = max_fraction,
      density_quantile = density_quantile,
      primary_methods = "umap,pacmap,localmap,tsne,trimap",
      expected_benefit = "rare_population_preservation",
      expected_risk = "needs_careful_testing_for_density_bias"
    )
  )
}

adaptive_k_strategy_grid <- function() {
  list(
    adaptive_k_strategy("mild", min_fraction = 0.67),
    adaptive_k_strategy("balanced", min_fraction = 0.50),
    adaptive_k_strategy("aggressive", min_fraction = 0.33)
  )
}

density_corrected_strategy <- function(mode, density_quantile = 0.5, strength = 1) {
  mode <- match.arg(mode, c("geomean", "sparse_boost", "densmap_radius"))
  suffix <- switch(
    mode,
    geomean = "geomean",
    sparse_boost = "sparse_boost",
    densmap_radius = "densmap_radius"
  )
  id_suffix <- suffix
  if (!identical(as.numeric(strength), 1)) {
    strength_tag <- gsub("\\.", "p", format(strength, trim = TRUE, scientific = FALSE))
    id_suffix <- paste0(id_suffix, "_s", strength_tag)
  }
  benefit <- if (identical(mode, "densmap_radius")) {
    "density_radius_preservation_and_rare_population_preservation"
  } else {
    "sparse_population_preservation"
  }
  risk <- if (identical(mode, "densmap_radius")) {
    "can_reduce_cluster_compaction_or_global_separation_without_a_true_density_regularizer"
  } else {
    "can_expand_dense_clusters_or_overboost_sparse_bridges"
  }
  relation <- if (identical(mode, "densmap_radius")) {
    "graph_radius_preservation_pilot_not_full_density_regularized_objective"
  } else {
    "graph_weight_pilot_not_density_regularized_objective"
  }
  strategy <- graph_strategy(
    paste0("graph_density_", id_suffix),
    paste0(
      "Density-corrected graph weights using ", mode,
      " local KNN radius normalization."
    ),
    transform_knn = function(ctx) graph_density_corrected_knn(
      ctx$knn,
      mode = mode,
      density_quantile = density_quantile,
      strength = strength
    ),
    params = list(
      graph_weighting = "density_corrected_local_scale",
      density_mode = mode,
      density_quantile = density_quantile,
      correction_strength = strength,
      primary_methods = "umap,pacmap,trimap,localmap",
      partial_methods = "tsne_affinity_graph",
      expected_benefit = benefit,
      expected_risk = risk,
      densmap_relation = relation
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "trimap", "localmap", "tsne") && backend %in% c("cpu", "cuda", "metal")
  }
  strategy
}

density_corrected_strategy_grid <- function() {
  list(
    density_corrected_strategy("geomean"),
    density_corrected_strategy("sparse_boost"),
    density_corrected_strategy("densmap_radius", strength = 0.5),
    density_corrected_strategy("densmap_radius", strength = 1)
  )
}

umap_fuzzy_strategy <- function(id,
                                set_op_mix_ratio = 1,
                                local_connectivity = 1,
                                weight_power = 1,
                                target_scale = 1,
                                description) {
  graph_strategy(
    id,
    description,
    transform_knn = function(ctx) umap_fuzzy_affinity_knn(
      ctx$knn,
      set_op_mix_ratio = set_op_mix_ratio,
      local_connectivity = local_connectivity,
      weight_power = weight_power,
      target_scale = target_scale,
      approximation = id
    ),
    params = list(
      graph_weighting = "umap_fuzzy_membership",
      set_op_mix_ratio = set_op_mix_ratio,
      local_connectivity = local_connectivity,
      weight_power = weight_power,
      target_scale = target_scale,
      distance_transform = "negative_log_fuzzy_membership",
      primary_methods = "umap",
      transferred_methods = "tsne,pacmap,trimap,localmap",
      expected_benefit = "umap_fuzzy_local_connectivity_and_set_operation_transfer",
      expected_risk = "can_overcompress_or_disconnect weak asymmetric neighbourhoods"
    )
  )
}

umap_fuzzy_strategy_grid <- function() {
  list(
    umap_fuzzy_strategy(
      "umap_fuzzy_union",
      set_op_mix_ratio = 1,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy simplicial-set union weights converted back to affinity distances."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_intersection",
      set_op_mix_ratio = 0,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy set intersection transfer; keeps only mutually strong local evidence."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_mix_025",
      set_op_mix_ratio = 0.25,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy set union/intersection transfer with set_op_mix_ratio = 0.25."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_mix_05",
      set_op_mix_ratio = 0.5,
      weight_power = 1,
      target_scale = 1,
      description = "Balanced UMAP fuzzy set union/intersection transfer with set_op_mix_ratio = 0.5."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_mix_075",
      set_op_mix_ratio = 0.75,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy set union/intersection transfer with set_op_mix_ratio = 0.75."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_smooth_p05",
      set_op_mix_ratio = 1,
      weight_power = 0.5,
      target_scale = 1,
      description = "UMAP fuzzy union with smoothed membership weights, weight^0.5."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_sharp_p2",
      set_op_mix_ratio = 1,
      weight_power = 2,
      target_scale = 1,
      description = "UMAP fuzzy union with sharpened membership weights, weight^2."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_target_05",
      set_op_mix_ratio = 1,
      weight_power = 1,
      target_scale = 0.5,
      description = "UMAP smooth-kNN bandwidth target reduced to 50 percent for sharper local affinities."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_target_2",
      set_op_mix_ratio = 1,
      weight_power = 1,
      target_scale = 2,
      description = "UMAP smooth-kNN bandwidth target doubled for broader local affinities."
    )
  )
}

weighted_edge_sampling_strategy <- function(keep_fraction,
                                            weight_power = 1,
                                            include_top = 1L,
                                            target_scale = 1) {
  keep_fraction <- as.numeric(keep_fraction)
  pct <- round(100 * keep_fraction)
  graph_strategy(
    paste0("graph_edge_sample_weighted_", pct),
    paste0(
      "Sample ", pct,
      " percent of KNN graph edges per point, proportional to UMAP fuzzy membership weights."
    ),
    transform_knn = function(ctx) graph_weighted_edge_sample_knn(
      ctx$knn,
      keep_fraction = keep_fraction,
      weight_power = weight_power,
      include_top = include_top,
      target_scale = target_scale,
      seed = ctx$seed
    ),
    params = list(
      graph_weighting = "umap_fuzzy_weighted_edge_sampling",
      edge_sampling = "weighted_without_replacement",
      edge_fraction = keep_fraction,
      weight_power = weight_power,
      include_top = as.integer(include_top),
      target_scale = target_scale,
      primary_methods = "umap",
      transferred_methods = "tsne,pacmap,trimap,localmap",
      expected_benefit = "fewer_optimization_edges_while_biasing_toward_high_confidence_neighbours",
      expected_risk = "can_drop_medium_strength_edges_needed_for_global_or_rare_population_structure"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 3L,
      message = "Weighted edge sampling requires --k >= 3 so there are enough candidate edges to sample."
    )
  )
}

weighted_edge_sampling_strategy_grid <- function() {
  list(
    weighted_edge_sampling_strategy(0.25),
    weighted_edge_sampling_strategy(0.50),
    weighted_edge_sampling_strategy(0.75)
  )
}

sparse_fuzzy_graph_csr <- function(ctx,
                                   target_scale = 1,
                                   local_connectivity = 1,
                                   set_op_mix_ratio = 1,
                                   weight_power = 1,
                                   prune_weight = 0) {
  builder <- get0("knn_fuzzy_graph_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(builder)) {
    stop("Native sparse fuzzy CSR graph builder is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  csr <- builder(
    knn$indices,
    knn$distances,
    as.numeric(target_scale),
    as.numeric(local_connectivity),
    as.numeric(set_op_mix_ratio),
    as.numeric(weight_power),
    as.numeric(prune_weight)
  )
  mix <- min(1, max(0, as.numeric(set_op_mix_ratio)))
  list(
    graph_csr = csr,
    graph_approximation = "sparse_fuzzy_csr",
    graph_effective_k = safe_number(csr$graph_effective_k),
    graph_edge_retention = safe_number(csr$graph_effective_k) / max(1, ctx$k),
    graph_mean_degree = safe_number(csr$graph_mean_degree),
    graph_min_degree = safe_number(csr$graph_min_degree),
    graph_max_degree = safe_number(csr$graph_max_degree),
    graph_isolated_fraction = safe_number(csr$graph_isolated_fraction),
    graph_padding_fraction = 0,
    graph_storage_format = as.character(csr$graph_storage_format),
    graph_sparse_nnz = safe_number(csr$graph_sparse_nnz),
    graph_sparse_internal_memory_mb = safe_number(csr$graph_sparse_internal_memory_mb),
    graph_sparse_r_memory_mb = safe_number(csr$graph_sparse_r_memory_mb),
    graph_dense_knn_memory_mb = safe_number(csr$graph_dense_knn_memory_mb),
    graph_sparse_internal_memory_ratio = safe_number(csr$graph_sparse_internal_memory_ratio),
    graph_sparse_r_memory_ratio = safe_number(csr$graph_sparse_r_memory_ratio),
    graph_sparse_prune_weight = as.numeric(prune_weight),
    graph_sparse_mean_weight = safe_number(csr$graph_sparse_mean_weight),
    graph_sparse_min_weight = safe_number(csr$graph_sparse_min_weight),
    graph_sparse_max_weight = safe_number(csr$graph_sparse_max_weight),
    umap_graph_set_op_mix_ratio = mix,
    umap_graph_local_connectivity = as.numeric(local_connectivity),
    umap_graph_weight_power = as.numeric(weight_power),
    umap_graph_target_scale = as.numeric(target_scale),
    umap_graph_distance_transform = "csr_fuzzy_membership_weights",
    umap_graph_mean_weight = safe_number(csr$graph_sparse_mean_weight),
    umap_graph_min_weight = safe_number(csr$graph_sparse_min_weight),
    umap_graph_max_weight = safe_number(csr$graph_sparse_max_weight)
  )
}

sparse_fuzzy_graph_config <- function(ctx) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = "fast",
      backend = "cpu"
    )
  }
  cfg$backend <- "cpu"
  cfg$graph_prep_backend <- "cpu_sparse_fuzzy_csr"
  cfg$optimizer_backend <- "cpu_cpp_csr"
  cfg$graph_storage_format <- ctx$graph_storage_format
  cfg$graph_sparse_nnz <- ctx$graph_sparse_nnz
  cfg$graph_sparse_internal_memory_mb <- ctx$graph_sparse_internal_memory_mb
  cfg$graph_sparse_internal_memory_ratio <- ctx$graph_sparse_internal_memory_ratio
  cfg$local_connectivity <- ctx$umap_graph_local_connectivity
  if (identical(ctx$method, "umap")) {
    cfg$umap_transfer_mode <- "sparse_fuzzy_csr_objective_optimizer"
  } else if (identical(ctx$method, "tsne")) {
    cfg$tsne_mode <- "sparse_fuzzy_sampled_repulsion_experimental"
  } else {
    cfg$transfer_mode <- paste0(ctx$method, "_sparse_fuzzy_graph")
  }
  cfg
}

run_sparse_fuzzy_graph_csr <- function(ctx) {
  if (is.null(ctx$graph_csr)) {
    stop("Sparse fuzzy CSR graph was not prepared.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- sparse_fuzzy_graph_config(ctx)
  if (identical(ctx$method, "umap")) {
    runner <- get0("fast_knn_umap_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
    if (!is.function(runner)) {
      stop("Native sparse CSR UMAP optimizer is not available in this fastEmbedR build.", call. = FALSE)
    }
    layout <- runner(
      ctx$graph_csr$offsets,
      ctx$graph_csr$neighbors,
      ctx$graph_csr$weights,
      2L,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$spectral_n_iter),
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$init_backend <- "cpu_csr_spectral"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }
  runner <- get0("knn_objective_embed_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native sparse CSR optimizer is not available in this fastEmbedR build.", call. = FALSE)
  }
  init <- if (identical(ctx$method, "tsne")) {
    fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  } else {
    fastEmbedR:::spectral_knn_init(
      knn$indices,
      knn$distances,
      n_components = 2L,
      spectral_n_iter = as.integer(cfg$spectral_n_iter),
      seed = ctx$seed,
      backend = "cpu"
    )
  }
  cfg$init_backend <- attr(init, "backend")
  layout <- runner(
    ctx$graph_csr$offsets,
    ctx$graph_csr$neighbors,
    ctx$graph_csr$weights,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    "sgd",
    0,
    0,
    as.integer(cfg$n_epochs + 1L),
    0.9,
    0.999,
    1e-8,
    FALSE
  )
  prefix <- switch(
    ctx$method,
    umap = "UMAP",
    tsne = "TSNE",
    pacmap = "PACMAP",
    trimap = "TRIMAP",
    localmap = "LOCALMAP",
    "EMB"
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, prefix)
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

sparse_fuzzy_graph_strategy <- function(set_op_mix_ratio = 1,
                                        local_connectivity = 1,
                                        suffix = NULL) {
  mix <- min(1, max(0, as.numeric(set_op_mix_ratio)))
  lc <- max(0, as.numeric(local_connectivity))
  if (is.null(suffix)) {
    suffix <- switch(
      format(mix, trim = TRUE, scientific = FALSE),
      "0" = "intersection",
      "0.25" = "mix_025",
      "0.5" = "mix_05",
      "0.75" = "mix_075",
      "1" = "union",
      paste0("mix_", gsub("[^0-9]+", "", sprintf("%.3f", mix)))
    )
  }
  id <- if (identical(suffix, "union")) {
    "graph_sparse_fuzzy_csr"
  } else {
    paste0("graph_sparse_fuzzy_csr_", suffix)
  }
  strategy <- graph_strategy(
    id,
    paste0(
      "Build a UMAP fuzzy graph as CSR with set_op_mix_ratio = ",
      format(mix, trim = TRUE, scientific = FALSE),
      " and local_connectivity = ",
      format(lc, trim = TRUE, scientific = FALSE),
      " and run the optimizer directly from sparse edge storage."
    ),
    transform_knn = function(ctx) sparse_fuzzy_graph_csr(
      ctx,
      set_op_mix_ratio = mix,
      local_connectivity = lc
    ),
    params = list(
      graph_weighting = "umap_fuzzy_membership",
      graph_storage = "csr_only",
      set_op_mix_ratio = mix,
      local_connectivity = lc,
      target_scale = 1,
      weight_power = 1,
      prune_weight = 0,
      primary_methods = "umap,pacmap,localmap",
      transferred_methods = "trimap",
      expected_benefit = "lower_graph_storage_memory_and_no_padded_rectangular_graph_for_optimization",
      expected_risk = "csr_optimizer_path_can_differ_from_method_specific_umap_optimizers"
    ),
    context_available = function(ctx) {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_fuzzy_graph_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("fast_knn_umap_csr_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Sparse fuzzy CSR requires native CSR graph builder and optimizer symbols."
      )
    }
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "trimap", "localmap") &&
      identical(backend, "cpu")
  }
  strategy$run <- function(ctx) run_sparse_fuzzy_graph_csr(ctx)
  strategy
}

sparse_fuzzy_graph_strategy_grid <- function() {
  list(
    sparse_fuzzy_graph_strategy(set_op_mix_ratio = 0, suffix = "intersection"),
    sparse_fuzzy_graph_strategy(set_op_mix_ratio = 0.25, suffix = "mix_025"),
    sparse_fuzzy_graph_strategy(set_op_mix_ratio = 0.5, suffix = "mix_05"),
    sparse_fuzzy_graph_strategy(set_op_mix_ratio = 0.75, suffix = "mix_075"),
    sparse_fuzzy_graph_strategy(set_op_mix_ratio = 1, suffix = "union")
  )
}

sparse_local_connectivity_strategy_grid <- function() {
  list(
    sparse_fuzzy_graph_strategy(
      set_op_mix_ratio = 1,
      local_connectivity = 1,
      suffix = "local_connectivity_1"
    ),
    sparse_fuzzy_graph_strategy(
      set_op_mix_ratio = 1,
      local_connectivity = 2,
      suffix = "local_connectivity_2"
    ),
    sparse_fuzzy_graph_strategy(
      set_op_mix_ratio = 1,
      local_connectivity = 3,
      suffix = "local_connectivity_3"
    )
  )
}

umap_fuzzy_local_connectivity_strategy_grid <- function() {
  list(
    umap_fuzzy_strategy(
      "umap_fuzzy_local_connectivity_1",
      set_op_mix_ratio = 1,
      local_connectivity = 1,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy graph transfer with local_connectivity = 1."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_local_connectivity_2",
      set_op_mix_ratio = 1,
      local_connectivity = 2,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy graph transfer with local_connectivity = 2."
    ),
    umap_fuzzy_strategy(
      "umap_fuzzy_local_connectivity_3",
      set_op_mix_ratio = 1,
      local_connectivity = 3,
      weight_power = 1,
      target_scale = 1,
      description = "UMAP fuzzy graph transfer with local_connectivity = 3."
    )
  )
}

tsne_affinity_strategy <- function(mode = c("auto", "multiscale_auto"),
                                   temperature = 1,
                                   suffix = NULL) {
  mode <- match.arg(mode)
  temperature <- as.numeric(temperature)
  if (is.null(suffix)) {
    suffix <- if (identical(mode, "auto")) "auto" else "multiscale_auto"
  }
  strategy <- graph_strategy(
    paste0("graph_tsne_affinity_", suffix),
    paste0(
      "t-SNE-style KNN affinity transform using ",
      if (identical(mode, "auto")) "single-scale" else "multiscale",
      " perplexity matching",
      if (!identical(temperature, 1)) " with probability sharpening." else "."
    ),
    transform_knn = function(ctx) {
      perplexities <- tsne_affinity_perplexities(knn_effective_k(ctx$knn), mode)
      graph_tsne_affinity_knn(
        ctx$knn,
        perplexities = perplexities,
        temperature = temperature,
        mode = suffix
      )
    },
    params = list(
      graph_weighting = "tsne_conditional_perplexity_affinity",
      perplexity_mode = mode,
      probability_temperature = temperature,
      primary_methods = "tsne",
      transferred_methods = "umap,pacmap,trimap,localmap",
      expected_benefit = "local_entropy_equalization_and_multiscale_global_structure",
      expected_risk = "can_overwrite_method_specific_distance_semantics"
    ),
    context_available = function(ctx) {
      perplexities <- tsne_affinity_perplexities(ctx$k, mode)
      list(
        available = length(perplexities) > 0L && max(perplexities) * 3L <= ctx$k,
        message = paste0(
          "t-SNE affinity strategy requires --k >= 3 * max(perplexity); current k is ",
          ctx$k, "."
        )
      )
    }
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy
}

tsne_affinity_strategy_grid <- function() {
  list(
    tsne_affinity_strategy("auto", temperature = 1, suffix = "auto"),
    tsne_affinity_strategy("multiscale_auto", temperature = 1, suffix = "multiscale_auto"),
    tsne_affinity_strategy("auto", temperature = 1.5, suffix = "sharp_auto")
  )
}

multiscale_perplexity_strategy <- function(perplexities) {
  perplexities <- sort(unique(as.integer(perplexities)))
  suffix <- multiscale_perplexity_suffix(perplexities)
  strategy <- graph_strategy(
    paste0("multiscale_perplexity_", suffix),
    paste0(
      "Explicit openTSNE-style multiscale perplexity pilot with perplexities ",
      paste(perplexities, collapse = ","),
      ". t-SNE averages conditional affinities; other methods receive a transferred multi-k graph."
    ),
    transform_knn = function(ctx) graph_multiscale_perplexity_knn(
      ctx$knn,
      method = ctx$method,
      perplexities = perplexities
    ),
    params = list(
      graph_weighting = "multiscale_perplexity",
      perplexities = paste(perplexities, collapse = ","),
      primary_methods = "tsne",
      transferred_methods = "umap,pacmap,trimap,localmap",
      umap_transfer = "multi_k_graph",
      pacmap_transfer = "multi_scale_near_pairs",
      trimap_transfer = "multi_scale_triplets_proxy",
      localmap_transfer = "multi_scale_local_graph",
      expected_benefit = "local_global_balance_from_multiple_neighbourhood_scales",
      expected_risk = "large_k_increases_runtime_memory_and_can_blur_fine_clusters"
    ),
    context_available = function(ctx) {
      required_k <- if (identical(ctx$method, "tsne")) {
        3L * max(perplexities)
      } else {
        max(perplexities)
      }
      list(
        available = ctx$k >= required_k,
        message = paste0(
          ctx$method, " multiscale perplexities ",
          paste(perplexities, collapse = ","),
          " require --k >= ", required_k,
          "; current k is ", ctx$k, "."
        )
      )
    }
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy$multiscale_perplexities <- perplexities
  strategy
}

multiscale_perplexity_strategy_grid <- function() {
  list(
    multiscale_perplexity_strategy(c(30L, 100L)),
    multiscale_perplexity_strategy(c(30L, 100L, 300L))
  )
}

pacmap_mid_near_label <- function(mid_fraction, mid_distance_scale) {
  paste0(
    "f",
    gsub("\\.", "p", format(as.numeric(mid_fraction), trim = TRUE, scientific = FALSE)),
    "_s",
    gsub("\\.", "p", format(as.numeric(mid_distance_scale), trim = TRUE, scientific = FALSE))
  )
}

pacmap_mid_near_strategy <- function(mid_fraction, mid_distance_scale) {
  mid_fraction <- as.numeric(mid_fraction)
  mid_distance_scale <- as.numeric(mid_distance_scale)
  label <- pacmap_mid_near_label(mid_fraction, mid_distance_scale)
  strategy <- graph_strategy(
    paste0("pacmap_mid_near_", label),
    paste0(
      "PaCMAP-style mid-near transfer: keep nearest graph edges and replace the farthest KNN slots with ",
      "second-order neighbours. This tests whether PaCMAP's mid-near global anchors help other objectives."
    ),
    transform_knn = function(ctx) graph_pacmap_mid_near_knn(
      ctx$knn,
      mid_fraction = mid_fraction,
      mid_distance_scale = mid_distance_scale,
      seed = ctx$seed
    ),
    params = list(
      graph_weighting = "pacmap_mid_near_second_order_pairs",
      pacmap_transfer_mode = "mid_near_pairs_as_second_order_graph_edges",
      pacmap_auxiliary_pair_family = "mid_near_pairs",
      mid_fraction = mid_fraction,
      mid_distance_scale = mid_distance_scale,
      primary_methods = "pacmap",
      transferred_methods = "umap,tsne,trimap,localmap",
      expected_benefit = "better_local_global_balance_and_cluster_arrangement",
      expected_risk = "can_lower_exact_knn_recall_or_blur_strict_local_neighbourhoods"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 10L,
      message = "PaCMAP mid-near transfer requires --k >= 10 so near and mid-near slots can coexist."
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy$pacmap_transfer_mode <- "mid_near_pairs_as_second_order_graph_edges"
  strategy$pacmap_auxiliary_pair_family <- "mid_near_pairs"
  strategy$pacmap_mid_near_requested_fraction <- mid_fraction
  strategy$pacmap_mid_near_distance_scale <- mid_distance_scale
  strategy
}

pacmap_mid_near_strategy_grid <- function() {
  list(
    pacmap_mid_near_strategy(0.20, 1.25),
    pacmap_mid_near_strategy(0.33, 1.50),
    pacmap_mid_near_strategy(0.50, 1.75)
  )
}

pacmap_mid_near_emphasis_label <- function(mid_fraction, emphasis_strength) {
  paste0(
    "f",
    gsub("\\.", "p", format(as.numeric(mid_fraction), trim = TRUE, scientific = FALSE)),
    "_e",
    gsub("\\.", "p", format(as.numeric(emphasis_strength), trim = TRUE, scientific = FALSE))
  )
}

pacmap_mid_near_emphasis_strategy <- function(mid_fraction,
                                              emphasis_strength) {
  mid_fraction <- as.numeric(mid_fraction)
  emphasis_strength <- as.numeric(emphasis_strength)
  label <- pacmap_mid_near_emphasis_label(mid_fraction, emphasis_strength)
  distance_multiplier <- max(0.20, min(2.00, 1 / max(0.25, emphasis_strength)))
  strategy <- graph_strategy(
    paste0("pacmap_mid_near_emphasis_", label),
    paste0(
      "PaCMAP mid-near pair emphasis with mid_fraction=",
      format(mid_fraction, trim = TRUE, scientific = FALSE),
      " and emphasis_strength=",
      format(emphasis_strength, trim = TRUE, scientific = FALSE),
      ". This tests whether stronger second-order mid-range anchors improve global structure."
    ),
    transform_knn = function(ctx) graph_pacmap_mid_near_emphasis_knn(
      ctx$knn,
      mid_fraction = mid_fraction,
      emphasis_strength = emphasis_strength,
      seed = ctx$seed
    ),
    params = list(
      graph_weighting = "pacmap_mid_near_emphasis",
      pacmap_transfer_mode = "mid_near_pair_emphasis_graph",
      pacmap_auxiliary_pair_family = "emphasized_mid_near_pairs",
      mid_fraction = mid_fraction,
      emphasis_strength = emphasis_strength,
      mid_distance_multiplier = distance_multiplier,
      umap_transfer = "multi_k_edges_with_emphasized_midrange_anchors",
      tsne_transfer = "multiscale_affinity_proxy_with_midrange_constraints",
      trimap_transfer = "semi_hard_triplet_candidate_proxy",
      localmap_transfer = "local_plus_midrange_edges",
      primary_methods = "pacmap",
      transferred_methods = "umap,tsne,trimap,localmap",
      expected_benefit = "better_global_structure",
      expected_risk = "too_much_midrange_emphasis_can_reduce_local_neighbourhood_fidelity"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 12L,
      message = "PaCMAP mid-near emphasis requires --k >= 12 so local and mid-range anchors can coexist."
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy$pacmap_transfer_mode <- "mid_near_pair_emphasis_graph"
  strategy$pacmap_auxiliary_pair_family <- "emphasized_mid_near_pairs"
  strategy$pacmap_mid_near_requested_fraction <- mid_fraction
  strategy$pacmap_mid_near_emphasis_strength <- emphasis_strength
  strategy$pacmap_mid_near_emphasis_distance_multiplier <- distance_multiplier
  strategy
}

pacmap_mid_near_emphasis_strategy_grid <- function() {
  list(
    pacmap_mid_near_emphasis_strategy(0.25, 1.25),
    pacmap_mid_near_emphasis_strategy(0.33, 1.50),
    pacmap_mid_near_emphasis_strategy(0.40, 2.00)
  )
}

pacmap_pair_separation_label <- function(near_ratio, mid_ratio, far_ratio) {
  paste0(
    "n",
    gsub("\\.", "p", format(as.numeric(near_ratio), trim = TRUE, scientific = FALSE)),
    "_m",
    gsub("\\.", "p", format(as.numeric(mid_ratio), trim = TRUE, scientific = FALSE)),
    "_f",
    gsub("\\.", "p", format(as.numeric(far_ratio), trim = TRUE, scientific = FALSE))
  )
}

pacmap_pair_separation_strategy <- function(near_ratio,
                                            mid_ratio,
                                            far_ratio,
                                            mid_distance_scale = 1.40,
                                            far_distance_scale = 3.00) {
  ratios <- pmax(0, as.numeric(c(near_ratio, mid_ratio, far_ratio)))
  ratios <- ratios / sum(ratios)
  near_ratio <- ratios[1L]
  mid_ratio <- ratios[2L]
  far_ratio <- ratios[3L]
  label <- pacmap_pair_separation_label(near_ratio, mid_ratio, far_ratio)
  strategy <- graph_strategy(
    paste0("pacmap_pair_sep_", label),
    paste0(
      "PaCMAP near/mid/far pair-separation transfer with near_ratio=",
      format(near_ratio, trim = TRUE, scientific = FALSE),
      ", mid_ratio=",
      format(mid_ratio, trim = TRUE, scientific = FALSE),
      ", far_ratio=",
      format(far_ratio, trim = TRUE, scientific = FALSE),
      ". Near slots use direct KNN edges, mid slots use second-order neighbours, ",
      "and far slots use weak long-distance anchors."
    ),
    transform_knn = function(ctx) graph_pacmap_pair_separation_knn(
      ctx$knn,
      near_ratio = near_ratio,
      mid_ratio = mid_ratio,
      far_ratio = far_ratio,
      mid_distance_scale = mid_distance_scale,
      far_distance_scale = far_distance_scale,
      seed = ctx$seed
    ),
    params = list(
      graph_weighting = "pacmap_near_mid_far_pair_separation",
      pacmap_transfer_mode = "near_mid_far_pair_separation_graph",
      pacmap_auxiliary_pair_family = "near_mid_near_far_pairs",
      near_ratio = near_ratio,
      mid_ratio = mid_ratio,
      far_ratio = far_ratio,
      mid_distance_scale = mid_distance_scale,
      far_distance_scale = far_distance_scale,
      umap_transfer = "separate_local_midrange_and_weak_negative_proxy_edges",
      tsne_transfer = "local_affinities_with_midrange_constraints_and_weak_far_anchors",
      trimap_transfer = "stratified_triplet_candidate_proxy",
      localmap_transfer = "local_mid_far_map_graph",
      primary_methods = "pacmap",
      transferred_methods = "umap,tsne,trimap,localmap",
      expected_benefit = "explicit_local_global_pair_budget_control",
      expected_risk = "far pairs are weak graph anchors here, not true objective-level repulsion"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 12L && ctx$n > ctx$k + 3L,
      message = "PaCMAP near/mid/far pair separation requires --k >= 12 and enough non-neighbour candidates."
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy$pacmap_transfer_mode <- "near_mid_far_pair_separation_graph"
  strategy$pacmap_auxiliary_pair_family <- "near_mid_near_far_pairs"
  strategy$pacmap_near_ratio <- near_ratio
  strategy$pacmap_mid_ratio <- mid_ratio
  strategy$pacmap_far_ratio <- far_ratio
  strategy$pacmap_mid_near_requested_fraction <- mid_ratio
  strategy$pacmap_mid_near_distance_scale <- mid_distance_scale
  strategy$pacmap_far_distance_scale <- far_distance_scale
  strategy
}

pacmap_pair_separation_strategy_grid <- function() {
  list(
    pacmap_pair_separation_strategy(0.60, 0.25, 0.15),
    pacmap_pair_separation_strategy(0.50, 0.30, 0.20),
    pacmap_pair_separation_strategy(0.40, 0.40, 0.20),
    pacmap_pair_separation_strategy(0.50, 0.20, 0.30)
  )
}

trimap_triplet_proxy_label <- function(inlier_ratio, semihard_ratio, global_anchor_ratio) {
  paste0(
    "i",
    gsub("\\.", "p", format(as.numeric(inlier_ratio), trim = TRUE, scientific = FALSE)),
    "_s",
    gsub("\\.", "p", format(as.numeric(semihard_ratio), trim = TRUE, scientific = FALSE)),
    "_g",
    gsub("\\.", "p", format(as.numeric(global_anchor_ratio), trim = TRUE, scientific = FALSE))
  )
}

trimap_triplet_proxy_strategy <- function(id_suffix,
                                          family,
                                          inlier_ratio,
                                          semihard_ratio,
                                          global_anchor_ratio,
                                          semihard_distance_scale = 1.40,
                                          global_anchor_distance_scale = 3.50) {
  ratios <- pmax(0, as.numeric(c(inlier_ratio, semihard_ratio, global_anchor_ratio)))
  ratios <- ratios / sum(ratios)
  inlier_ratio <- ratios[1L]
  semihard_ratio <- ratios[2L]
  global_anchor_ratio <- ratios[3L]
  label <- trimap_triplet_proxy_label(inlier_ratio, semihard_ratio, global_anchor_ratio)
  strategy <- graph_strategy(
    paste0("trimap_", id_suffix, "_", label),
    paste0(
      "TriMap-inspired ", family,
      " transfer: keep direct inlier KNN edges, add second-order semi-hard candidates, ",
      "and optionally add weak global-anchor candidates. This is a graph proxy because the current ",
      "native optimizer samples outlier negatives internally."
    ),
    transform_knn = function(ctx) graph_trimap_triplet_proxy_knn(
      ctx$knn,
      inlier_ratio = inlier_ratio,
      semihard_ratio = semihard_ratio,
      global_anchor_ratio = global_anchor_ratio,
      semihard_distance_scale = semihard_distance_scale,
      global_anchor_distance_scale = global_anchor_distance_scale,
      family = family,
      seed = ctx$seed
    ),
    params = list(
      graph_weighting = "trimap_triplet_candidate_proxy",
      trimap_transfer_mode = "triplet_candidate_graph_proxy",
      trimap_triplet_family = family,
      trimap_inlier_ratio = inlier_ratio,
      trimap_semihard_ratio = semihard_ratio,
      trimap_global_anchor_ratio = global_anchor_ratio,
      trimap_semihard_distance_scale = semihard_distance_scale,
      trimap_global_anchor_distance_scale = global_anchor_distance_scale,
      trimap_native_explicit_triplets = 0,
      umap_transfer = "local_edges_with_semihard_and_global_anchor_positive_proxy_edges",
      pacmap_transfer = "near_pairs_plus_semihard_midrange_and_global_anchors",
      trimap_transfer = "inlier_triplet_candidate_proxy",
      localmap_transfer = "local_graph_with_semihard_and_global_anchor_edges",
      primary_methods = "trimap",
      transferred_methods = "umap,pacmap,localmap",
      expected_benefit = "better_global_structure_and_more_stable_triplet_candidates",
      expected_risk = "global_anchors_are_positive_proxy_edges_not_true_trimap_outlier_negatives"
    ),
    context_available = function(ctx) list(
      available = ctx$k >= 12L && ctx$n > ctx$k + 3L,
      message = "TriMap triplet proxies require --k >= 12 and enough non-neighbour candidates."
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "trimap", "localmap") &&
      backend %in% c("cpu", "cuda", "metal")
  }
  strategy$trimap_transfer_mode <- "triplet_candidate_graph_proxy"
  strategy$trimap_triplet_family <- family
  strategy$trimap_inlier_ratio <- inlier_ratio
  strategy$trimap_semihard_ratio <- semihard_ratio
  strategy$trimap_global_anchor_ratio <- global_anchor_ratio
  strategy$trimap_semihard_distance_scale <- semihard_distance_scale
  strategy$trimap_global_anchor_distance_scale <- global_anchor_distance_scale
  strategy$trimap_native_explicit_triplets <- 0
  strategy
}

trimap_triplet_proxy_strategy_grid <- function() {
  list(
    trimap_triplet_proxy_strategy(
      "semihard_mild",
      "semi_hard_candidates",
      inlier_ratio = 0.75,
      semihard_ratio = 0.25,
      global_anchor_ratio = 0.00,
      semihard_distance_scale = 1.25,
      global_anchor_distance_scale = 3.00
    ),
    trimap_triplet_proxy_strategy(
      "semihard_balanced",
      "semi_hard_plus_global_anchors",
      inlier_ratio = 0.60,
      semihard_ratio = 0.30,
      global_anchor_ratio = 0.10,
      semihard_distance_scale = 1.40,
      global_anchor_distance_scale = 3.50
    ),
    trimap_triplet_proxy_strategy(
      "global_anchor",
      "global_anchor_triplets",
      inlier_ratio = 0.50,
      semihard_ratio = 0.20,
      global_anchor_ratio = 0.30,
      semihard_distance_scale = 1.60,
      global_anchor_distance_scale = 4.50
    )
  )
}

distance_percentile_prune_strategy <- function(drop_fraction) {
  drop_fraction <- as.numeric(drop_fraction)
  drop_pct <- as.integer(round(100 * drop_fraction))
  keep_pct <- as.integer(round(100 * (1 - drop_fraction)))
  strategy <- graph_strategy(
    paste0("graph_prune_distance_top", drop_pct),
    paste0(
      "Remove the top ", drop_pct,
      "% longest KNN distances per node using the row-wise ",
      keep_pct, "th distance percentile."
    ),
    transform_knn = function(ctx) graph_distance_percentile_prune_knn(
      ctx$knn,
      drop_fraction = drop_fraction
    ),
    params = list(
      graph_weighting = "distance_percentile_prune",
      drop_longest_fraction = drop_fraction,
      keep_distance_percentile = keep_pct,
      primary_methods = "umap,pacmap,localmap",
      expected_benefit = "cleaner_local_clusters",
      expected_risk = "may_disconnect_sparse_or_rare_populations"
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "localmap") && backend %in% c("cpu", "cuda", "metal")
  }
  strategy
}

distance_percentile_prune_strategy_grid <- function() {
  list(
    distance_percentile_prune_strategy(0.05),
    distance_percentile_prune_strategy(0.10),
    distance_percentile_prune_strategy(0.20)
  )
}

mst_rescue_strategy <- function(drop_fraction) {
  drop_fraction <- as.numeric(drop_fraction)
  drop_pct <- as.integer(round(100 * drop_fraction))
  strategy <- graph_strategy(
    paste0("graph_prune_distance_top", drop_pct, "_mst"),
    paste0(
      "Remove the top ", drop_pct,
      "% longest KNN distances per node, then add the base KNN minimum spanning forest as a connectivity rescue."
    ),
    transform_knn = function(ctx) graph_distance_prune_mst_rescue_knn(
      ctx$knn,
      drop_fraction = drop_fraction
    ),
    params = list(
      graph_weighting = "distance_prune_with_mst_rescue",
      drop_longest_fraction = drop_fraction,
      rescue = "base_knn_minimum_spanning_forest",
      primary_methods = "umap,pacmap,localmap",
      expected_benefit = "prevents_disconnected_embeddings_after_pruning",
      expected_risk = "can_reintroduce_long_bridge_edges"
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "localmap") && backend %in% c("cpu", "cuda", "metal")
  }
  strategy
}

mst_rescue_strategy_grid <- function() {
  list(
    mst_rescue_strategy(0.05),
    mst_rescue_strategy(0.10),
    mst_rescue_strategy(0.20)
  )
}

effective_resistance_sparsify_strategy <- function(keep_fraction, max_n = 1200L) {
  keep_fraction <- as.numeric(keep_fraction)
  keep_pct <- as.integer(round(100 * keep_fraction))
  strategy <- graph_strategy(
    paste0("graph_sparsify_er_keep", keep_pct),
    paste0(
      "Exact effective-resistance spectral sparsification pilot retaining about ",
      keep_pct, "% of directed KNN edges per node."
    ),
    transform_knn = function(ctx) graph_effective_resistance_sparsify_knn(
      ctx$knn,
      keep_fraction = keep_fraction
    ),
    params = list(
      graph_weighting = "effective_resistance_spectral_sparsification",
      keep_fraction = keep_fraction,
      primary_methods = "umap,pacmap,localmap",
      partial_methods = "tsne_affinity_graph",
      expected_benefit = "lower_memory_and_bridge_preservation",
      expected_risk = "possible_loss_of_local_detail",
      implementation_note = "dense_laplacian_eigendecomposition_pilot"
    ),
    context_available = function(ctx) list(
      available = ctx$n <= max_n && ctx$k >= 5L,
      message = paste0(
        "Effective-resistance sparsification uses a dense Laplacian eigensolver in this pilot; ",
        "requires n <= ", max_n, " and k >= 5."
      )
    )
  )
  strategy$compatible <- function(method, backend) {
    method %in% c("umap", "pacmap", "localmap", "tsne") && backend %in% c("cpu", "cuda", "metal")
  }
  strategy
}

effective_resistance_sparsify_strategy_grid <- function() {
  list(
    effective_resistance_sparsify_strategy(0.50),
    effective_resistance_sparsify_strategy(0.75)
  )
}

graph_strategy_grid <- function() {
  c(list(
    graph_strategy(
      "graph_standard_knn",
      "Standard directed KNN graph baseline. Directly used by UMAP, PaCMAP, and LocalMAP; converted into affinities/triplets for t-SNE and TriMap.",
      transform_knn = function(ctx) graph_standard_knn(ctx$knn),
      params = list(
        graph_weighting = "standard_knn",
        primary_methods = "umap,pacmap,localmap",
        partial_methods = "tsne_affinities,trimap_triplets"
      )
    ),
    graph_strategy(
      "graph_prune_50",
      "Keep only the nearest 50 percent of KNN edges before graph weighting.",
      transform_knn = function(ctx) graph_limit_knn(ctx$knn, ceiling(ctx$k * 0.50), "prune_50", exact = TRUE),
      params = list(edge_fraction = 0.50, graph_weighting = "nearest_edge_prune")
    ),
    graph_strategy(
      "graph_prune_75",
      "Keep only the nearest 75 percent of KNN edges before graph weighting.",
      transform_knn = function(ctx) graph_limit_knn(ctx$knn, ceiling(ctx$k * 0.75), "prune_75", exact = TRUE),
      params = list(edge_fraction = 0.75, graph_weighting = "nearest_edge_prune")
    ),
    graph_strategy(
      "graph_mutual_50",
      "Prefer reciprocal neighbours, then fill with nearest non-reciprocal edges to 50 percent of KNN edges.",
      transform_knn = function(ctx) graph_mutual_fill_knn(ctx$knn, ceiling(ctx$k * 0.50), "mutual_50"),
      params = list(edge_fraction = 0.50, graph_weighting = "reciprocal_first")
    ),
    graph_strategy(
      "graph_mutual_75",
      "Prefer reciprocal neighbours, then fill with nearest non-reciprocal edges to 75 percent of KNN edges.",
      transform_knn = function(ctx) graph_mutual_fill_knn(ctx$knn, ceiling(ctx$k * 0.75), "mutual_75"),
      params = list(edge_fraction = 0.75, graph_weighting = "reciprocal_first")
    ),
    graph_strategy(
      "graph_mutual_only",
      "Keep only reciprocal KNN edges; rows with too few reciprocal edges are weakly padded only for rectangular KNN compatibility.",
      transform_knn = function(ctx) graph_mutual_only_knn(ctx$knn),
      params = list(
        graph_weighting = "mutual_knn_only",
        primary_methods = "umap,pacmap,localmap",
        partial_methods = "tsne_affinities,trimap_inlier_pairs",
        padding = "weak_finite_edges_for_rectangular_knn_input",
        expected_risk = "may_disconnect_rare_or_sparse_populations"
      )
    ),
    graph_strategy(
      "graph_symmetric_union",
      "Use the union of directed KNN edges, adding incoming reverse-neighbour edges to improve graph connectivity.",
      transform_knn = function(ctx) graph_symmetric_union_knn(ctx$knn),
      params = list(
        graph_weighting = "symmetric_union_knn",
        primary_methods = "umap,pacmap,localmap",
        partial_methods = "tsne_affinity_graph",
        expected_benefit = "more_connected_graph",
        expected_risk = "possibly_noisier_neighbourhoods"
      )
    ),
    graph_strategy(
      "graph_jaccard_weighted",
      "Reweight KNN distances by Jaccard neighbour-set overlap to strengthen shared-neighbour edges.",
      transform_knn = function(ctx) graph_jaccard_weighted_knn(ctx$knn),
      params = list(
        graph_weighting = "jaccard_neighbour_overlap",
        transform = "distance_times_one_minus_jaccard_plus_0.05",
        primary_methods = "umap,pacmap,localmap",
        partial_methods = "tsne_affinity_weighting",
        expected_benefit = "cluster_separation_single_cell_like_data",
        expected_risk = "can_overemphasize_dense_clusters"
      )
    ),
    graph_strategy(
      "graph_rank_distances",
      "Replace raw distances with normalized neighbour ranks to test rank-only graph construction.",
      transform_knn = function(ctx) graph_rank_distances(ctx$knn),
      params = list(graph_weighting = "rank_only")
    ),
    graph_strategy(
      "graph_binary_distances",
      "Use an unweighted KNN graph by replacing every non-self distance with one.",
      transform_knn = function(ctx) graph_binary_distances(ctx$knn),
      params = list(graph_weighting = "binary_unweighted")
    ),
    graph_strategy(
      "graph_local_scale",
      "Normalize each row by its local median KNN distance before graph weighting.",
      transform_knn = function(ctx) graph_local_scale_distances(ctx$knn),
      params = list(graph_weighting = "row_median_scaled")
    ),
    graph_strategy(
      "graph_snn_reweight",
      "Reweight distances by shared-nearest-neighbour overlap before embedding.",
      transform_knn = function(ctx) graph_snn_reweight_knn(ctx$knn),
      params = list(graph_weighting = "shared_nearest_neighbour"),
      context_available = function(ctx) list(
        available = ctx$n * ctx$k <= 250000,
        message = "SNN graph reweighting is an R-side pilot and is skipped when n * k exceeds 250000."
      )
    )
  ),
  snn_graph_strategy_grid(),
  adaptive_k_strategy_grid(),
  density_corrected_strategy_grid(),
  umap_fuzzy_strategy_grid(),
  sparse_fuzzy_graph_strategy_grid(),
  umap_fuzzy_local_connectivity_strategy_grid(),
  sparse_local_connectivity_strategy_grid(),
  weighted_edge_sampling_strategy_grid(),
  localmap_false_neighbor_strategy_grid(),
  localmap_local_weight_strategy_grid(),
  tsne_affinity_strategy_grid(),
  multiscale_perplexity_strategy_grid(),
  pacmap_mid_near_strategy_grid(),
  pacmap_mid_near_emphasis_strategy_grid(),
  pacmap_pair_separation_strategy_grid(),
  distance_percentile_prune_strategy_grid(),
  mst_rescue_strategy_grid(),
  effective_resistance_sparsify_strategy_grid())
}

strategy_availability <- function(strategy) {
  if (is.null(strategy$availability)) {
    return(list(available = TRUE, message = NA_character_))
  }
  tryCatch(strategy$availability(), error = function(e) {
    list(available = FALSE, message = conditionMessage(e))
  })
}

strategy_context_available <- function(strategy, ctx) {
  if (is.null(strategy$context_available)) {
    return(list(available = TRUE, message = NA_character_))
  }
  tryCatch(strategy$context_available(ctx), error = function(e) {
    list(available = FALSE, message = conditionMessage(e))
  })
}

knn_strategy_families <- function() {
  c(
    "exact_knn", "kdtree_knn", "balltree_knn", "brute_force_knn",
    "annoy_knn", "hnsw_knn", "nndescent_knn", "faiss_knn",
    "faiss_gpu_knn", "cuml_knn", "shared_knn", "optimization_budget",
    "optimization_schedule", "tsne_repulsion", "graph_construction",
    "sparse_edge_batching",
    "vectorized_edge_optimization",
    "atomic_sgd_optimization",
    "output_metric", "triplet_constraints", "artificial_neighbor_penalty",
    "false_neighbor_monitor", "initialization", "warm_start",
    "early_stopping",
    "coarse_to_fine"
  )
}

strategy_needs_knn <- function(strategy) {
  strategy$family %in% knn_strategy_families()
}

knn_cache_key <- function(ctx, strategy) {
  strategy_key <- if (!is.null(strategy$knn_cache_strategy_id)) strategy$knn_cache_strategy_id else strategy$id
  paste(
    ctx$dataset_name,
    paste0("n", ctx$n),
    paste0("p", ctx$p),
    strategy_key,
    ctx$backend,
    paste0("k", ctx$k),
    paste0("seed", ctx$seed),
    paste0("pca", ctx$pca_dims),
    paste0("metric", ctx$knn_metric),
    sep = "\r"
  )
}

safe_cache_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_.-]+", "_", x)
  substr(x, 1L, 220L)
}

knn_cache_extension <- function(format) {
  switch(format, rds = "rds", npz = "npz", hdf5 = "h5", parquet = "parquet")
}

knn_disk_cache_path <- function(options, graph_key) {
  format <- options$knn_cache_format
  file.path(
    options$knn_cache_dir,
    format,
    paste0(safe_cache_name(graph_key), ".", knn_cache_extension(format))
  )
}

metadata_json <- function(metadata) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package `jsonlite` is required for this KNN cache format.", call. = FALSE)
  }
  as.character(jsonlite::toJSON(metadata, auto_unbox = TRUE, null = "null"))
}

metadata_from_json <- function(x) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package `jsonlite` is required for this KNN cache format.", call. = FALSE)
  }
  jsonlite::fromJSON(as.character(x), simplifyVector = FALSE)
}

knn_payload_from_record <- function(record, ctx, strategy, graph_key) {
  list(
    indices = record$knn$indices,
    distances = record$knn$distances,
    metadata = list(
      graph_key = graph_key,
      dataset = ctx$dataset_name,
      n = ctx$n,
      p = ctx$p,
      k = ctx$k,
      strategy = strategy$id,
      strategy_family = strategy$family,
      backend_requested = ctx$backend,
      knn_backend = attr(record$knn, "backend"),
      exact = isTRUE(attr(record$knn, "exact")),
      seed = ctx$seed,
      pca_dims = ctx$pca_dims,
      knn_metric = ctx$knn_metric,
      graph_time_sec = record$graph_time,
      graph_index_build_time_sec = record$graph_build_time,
      graph_query_time_sec = record$graph_query_time,
      graph_memory_mb = record$memory_mb,
      created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
    )
  )
}

write_knn_payload <- function(payload, path, format) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (identical(format, "rds")) {
    saveRDS(payload, path)
    return(invisible(path))
  }
  if (identical(format, "npz")) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      stop("R package `reticulate` is required for NPZ KNN cache files.", call. = FALSE)
    }
    np <- reticulate::import("numpy", delay_load = FALSE)
    np$savez_compressed(
      path,
      indices = np$array(payload$indices, dtype = "int32", order = "C"),
      distances = np$array(payload$distances, dtype = "float64", order = "C"),
      metadata_json = metadata_json(payload$metadata)
    )
    return(invisible(path))
  }
  if (identical(format, "hdf5")) {
    if (!requireNamespace("hdf5r", quietly = TRUE)) {
      stop("R package `hdf5r` is required for HDF5 KNN cache files.", call. = FALSE)
    }
    h5 <- hdf5r::H5File$new(path, mode = "w")
    on.exit(h5$close_all(), add = TRUE)
    h5$create_dataset("indices", robj = payload$indices)
    h5$create_dataset("distances", robj = payload$distances)
    h5$create_dataset("metadata_json", robj = metadata_json(payload$metadata))
    return(invisible(path))
  }
  if (identical(format, "parquet")) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("R package `arrow` is required for Parquet KNN cache files.", call. = FALSE)
    }
    n <- nrow(payload$indices)
    k <- ncol(payload$indices)
    df <- data.frame(
      row = rep(seq_len(n), each = k),
      neighbor = rep(seq_len(k), times = n),
      index = as.integer(as.vector(t(payload$indices))),
      distance = as.numeric(as.vector(t(payload$distances))),
      metadata_json = NA_character_
    )
    df$metadata_json[1L] <- metadata_json(payload$metadata)
    arrow::write_parquet(df, path)
    return(invisible(path))
  }
  stop("Unsupported KNN cache format: ", format, call. = FALSE)
}

read_knn_payload <- function(path, format) {
  if (identical(format, "rds")) {
    return(readRDS(path))
  }
  if (identical(format, "npz")) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      stop("R package `reticulate` is required for NPZ KNN cache files.", call. = FALSE)
    }
    np <- reticulate::import("numpy", delay_load = FALSE)
    loaded <- np$load(path, allow_pickle = FALSE)
    metadata <- loaded[["metadata_json"]]
    metadata <- if (!is.character(metadata) && reticulate::py_has_attr(metadata, "item")) metadata$item() else as.character(metadata)
    return(list(
      indices = as.matrix(loaded[["indices"]]),
      distances = as.matrix(loaded[["distances"]]),
      metadata = metadata_from_json(metadata)
    ))
  }
  if (identical(format, "hdf5")) {
    if (!requireNamespace("hdf5r", quietly = TRUE)) {
      stop("R package `hdf5r` is required for HDF5 KNN cache files.", call. = FALSE)
    }
    h5 <- hdf5r::H5File$new(path, mode = "r")
    on.exit(h5$close_all(), add = TRUE)
    metadata <- h5[["metadata_json"]]$read()
    return(list(
      indices = as.matrix(h5[["indices"]]$read()),
      distances = as.matrix(h5[["distances"]]$read()),
      metadata = metadata_from_json(metadata)
    ))
  }
  if (identical(format, "parquet")) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("R package `arrow` is required for Parquet KNN cache files.", call. = FALSE)
    }
    df <- as.data.frame(arrow::read_parquet(path))
    n <- max(df$row)
    k <- max(df$neighbor)
    indices <- matrix(NA_integer_, n, k)
    distances <- matrix(NA_real_, n, k)
    idx <- cbind(df$row, df$neighbor)
    indices[idx] <- as.integer(df$index)
    distances[idx] <- as.numeric(df$distance)
    metadata <- df$metadata_json[which(!is.na(df$metadata_json) & nzchar(df$metadata_json))[1L]]
    return(list(
      indices = indices,
      distances = distances,
      metadata = metadata_from_json(metadata)
    ))
  }
  stop("Unsupported KNN cache format: ", format, call. = FALSE)
}

record_from_knn_payload <- function(payload, dataset, ctx, k, options, path, format, load_time) {
  metadata <- payload$metadata
  backend <- metadata$knn_backend
  if (is.null(backend) || !nzchar(backend)) backend <- paste0("disk_cache_", format)
  knn <- finish_external_knn(
    payload$indices,
    payload$distances,
    backend = backend,
    exact = isTRUE(metadata$exact)
  )
  quality <- evaluate_knn_quality(
    knn,
    dataset$x,
    k,
    options$knn_quality_sample_size,
    ctx$seed
  )
  list(
    knn = knn,
    time = as.numeric(load_time),
    build_time = 0,
    query_time = 0,
    graph_time = safe_number(metadata$graph_time_sec),
    graph_build_time = safe_number(metadata$graph_index_build_time_sec),
    graph_query_time = safe_number(metadata$graph_query_time_sec),
    memory_mb = NA_real_,
    quality = quality,
    disk_cache_hit = TRUE,
    disk_load_time = as.numeric(load_time),
    disk_save_time = 0,
    disk_cache_path = normalizePath(path, mustWork = FALSE),
    disk_cache_format = format,
    source = "disk_cache"
  )
}

build_knn_record <- function(dataset, ctx, strategy, backend, k, options) {
  rss_before <- current_rss_mb()
  knn_time <- system.time({
    knn <- if (!is.null(strategy$build_knn)) {
      ctx$knn_backend <- paste0("external_", strategy$family)
      strategy$build_knn(ctx)
    } else {
      knn_backend <- if (!is.null(strategy$knn_backend)) strategy$knn_backend(ctx) else backend
      ctx$knn_backend <- knn_backend
      if (!backend_supported(knn_backend)) {
        stop(paste0("KNN backend `", knn_backend, "` is unavailable or unsupported."), call. = FALSE)
      }
      fastEmbedR::nn(dataset$x, k = k + 1L, backend = knn_backend)
    }
  })[["elapsed"]]
  rss_after <- current_rss_mb()
  build_time <- safe_number(attr(knn, "index_build_time_sec"))
  query_time <- safe_number(attr(knn, "query_time_sec"), as.numeric(knn_time))
  quality <- evaluate_knn_quality(
    knn,
    dataset$x,
    k,
    options$knn_quality_sample_size,
    ctx$seed
  )
  memory_mb <- if (is.finite(rss_before) && is.finite(rss_after)) rss_after - rss_before else NA_real_
  list(
    knn = knn,
    time = as.numeric(knn_time),
    build_time = as.numeric(build_time),
    query_time = as.numeric(query_time),
    graph_time = as.numeric(knn_time),
    graph_build_time = as.numeric(build_time),
    graph_query_time = as.numeric(query_time),
    memory_mb = memory_mb,
    quality = quality,
    disk_cache_hit = FALSE,
    disk_load_time = 0,
    disk_save_time = 0,
    disk_cache_path = NA_character_,
    disk_cache_format = NA_character_,
    source = "computed"
  )
}

load_or_build_knn_record <- function(dataset, ctx, strategy, backend, k, options, graph_key) {
  use_disk <- isTRUE(options$knn_disk_cache)
  path <- if (use_disk) knn_disk_cache_path(options, graph_key) else NA_character_
  format <- options$knn_cache_format
  if (use_disk && file.exists(path) && !isTRUE(options$knn_cache_force_recompute)) {
    loaded <- timed_step(read_knn_payload(path, format))
    return(record_from_knn_payload(
      loaded$value,
      dataset,
      ctx,
      k,
      options,
      path,
      format,
      loaded$time
    ))
  }
  record <- build_knn_record(dataset, ctx, strategy, backend, k, options)
  if (use_disk) {
    payload <- knn_payload_from_record(record, ctx, strategy, graph_key)
    saved <- timed_step(write_knn_payload(payload, path, format))
    record$disk_save_time <- saved$time
    record$disk_cache_path <- normalizePath(path, mustWork = FALSE)
    record$disk_cache_format <- format
  }
  record
}

apply_knn_record <- function(ctx, record, reuse_mode, cache_hit, graph_key) {
  ctx$knn <- record$knn
  ctx$knn_reuse_mode <- reuse_mode
  ctx$knn_cache_hit <- isTRUE(cache_hit)
  ctx$knn_graph_key <- graph_key
  ctx$knn_graph_source <- if (isTRUE(cache_hit)) "memory_cache" else record$source
  ctx$knn_disk_cache_hit <- isTRUE(record$disk_cache_hit) && !isTRUE(cache_hit)
  ctx$knn_disk_cache_format <- record$disk_cache_format
  ctx$knn_disk_cache_path <- record$disk_cache_path
  ctx$knn_disk_load_time_sec <- if (isTRUE(cache_hit)) 0 else record$disk_load_time
  ctx$knn_disk_save_time_sec <- if (isTRUE(cache_hit)) 0 else record$disk_save_time
  ctx$knn_graph_time_sec <- safe_number(record$graph_time, record$time)
  ctx$knn_graph_index_build_time_sec <- safe_number(record$graph_build_time, record$build_time)
  ctx$knn_graph_query_time_sec <- safe_number(record$graph_query_time, record$query_time)
  ctx$knn_time_sec <- if (isTRUE(cache_hit)) 0 else record$time
  ctx$knn_index_build_time_sec <- if (isTRUE(cache_hit)) 0 else record$build_time
  ctx$knn_query_time_sec <- if (isTRUE(cache_hit)) 0 else record$query_time
  ctx$knn_memory_mb <- if (isTRUE(cache_hit)) 0 else record$memory_mb
  ctx$knn_recall_at_k <- record$quality$knn_recall_at_k
  ctx$knn_mean_distance_error <- record$quality$knn_mean_distance_error
  ctx$knn_rank_correlation <- record$quality$knn_rank_correlation
  ctx$knn_quality_sample_size <- record$quality$knn_quality_sample_size
  if (isTRUE(ctx$triplet_mining_approximate)) {
    ctx$triplet_mining_knn_backend <- safe_character(attr(record$knn, "backend"), ctx$triplet_mining_graph_source)
    ctx$triplet_mining_recall_at_k <- record$quality$knn_recall_at_k
    ctx$triplet_mining_rank_correlation <- record$quality$knn_rank_correlation
    ctx$triplet_mining_distance_error <- record$quality$knn_mean_distance_error
  }
  ctx
}

apply_graph_transform <- function(ctx, result, elapsed, dataset, sample_size, seed) {
  if (inherits(result, "fastEmbedR_nn")) result <- list(knn = result)
  if (!is.list(result) || (is.null(result$knn) && is.null(result$graph_csr))) {
    stop("Graph transform did not return a KNN object or CSR graph.", call. = FALSE)
  }
  transformed <- result$knn
  has_transformed_knn <- !is.null(transformed)
  if (has_transformed_knn && (!is.list(transformed) || !all(c("indices", "distances") %in% names(transformed)))) {
    stop("Graph transform returned an invalid KNN object.", call. = FALSE)
  }
  if (!is.null(result$graph_csr)) {
    ctx$graph_csr <- result$graph_csr
  }
  effective_k <- if (!is.null(result$graph_effective_k)) {
    as.integer(result$graph_effective_k)
  } else {
    knn_effective_k(ctx$knn)
  }
  effective_k <- max(0L, min(effective_k, max(knn_effective_k(ctx$knn), effective_k)))
  if (has_transformed_knn) {
    ctx$knn <- transformed
  }
  ctx$graph_approximation <- if (!is.null(result$graph_approximation)) as.character(result$graph_approximation) else "custom"
  ctx$graph_approximation_time_sec <- as.numeric(elapsed)
  ctx$graph_effective_k <- effective_k
  ctx$graph_edge_retention <- if (!is.null(result$graph_edge_retention)) {
    as.numeric(result$graph_edge_retention)
  } else {
    effective_k / max(1L, ctx$k)
  }
  if (has_transformed_knn && effective_k > 0L) {
    quality <- evaluate_knn_quality(
      transformed,
      dataset$x,
      ctx$k,
      sample_size,
      seed
    )
    } else {
      quality <- list(
        knn_recall_at_k = NA_real_,
        knn_mean_distance_error = NA_real_,
        knn_rank_correlation = NA_real_,
      knn_quality_sample_size = NA_integer_
    )
  }
  ctx$graph_recall_at_k <- if (!is.null(result$graph_recall_at_k)) {
    as.numeric(result$graph_recall_at_k)
  } else {
    quality$knn_recall_at_k
  }
  ctx$graph_mean_distance_error <- quality$knn_mean_distance_error
  ctx$graph_rank_correlation <- quality$knn_rank_correlation
  ctx$graph_quality_sample_size <- quality$knn_quality_sample_size
  ctx$graph_storage_format <- if (!is.null(result$graph_storage_format)) {
    as.character(result$graph_storage_format)
  } else {
    ctx$graph_storage_format
  }
  ctx$graph_sparse_nnz <- safe_number(result$graph_sparse_nnz)
  ctx$graph_sparse_internal_memory_mb <- safe_number(result$graph_sparse_internal_memory_mb)
  ctx$graph_sparse_r_memory_mb <- safe_number(result$graph_sparse_r_memory_mb)
  ctx$graph_dense_knn_memory_mb <- safe_number(result$graph_dense_knn_memory_mb)
  ctx$graph_sparse_internal_memory_ratio <- safe_number(result$graph_sparse_internal_memory_ratio)
  ctx$graph_sparse_r_memory_ratio <- safe_number(result$graph_sparse_r_memory_ratio)
  ctx$graph_sparse_prune_weight <- safe_number(result$graph_sparse_prune_weight)
  ctx$graph_sparse_mean_weight <- safe_number(result$graph_sparse_mean_weight)
  ctx$graph_sparse_min_weight <- safe_number(result$graph_sparse_min_weight)
  ctx$graph_sparse_max_weight <- safe_number(result$graph_sparse_max_weight)
  ctx$graph_mean_degree <- safe_number(result$graph_mean_degree)
  ctx$graph_min_degree <- safe_number(result$graph_min_degree)
  ctx$graph_max_degree <- safe_number(result$graph_max_degree)
  ctx$graph_isolated_fraction <- safe_number(result$graph_isolated_fraction)
  ctx$graph_padding_fraction <- safe_number(result$graph_padding_fraction)
  ctx$graph_mean_jaccard <- safe_number(result$graph_mean_jaccard)
  ctx$graph_min_jaccard <- safe_number(result$graph_min_jaccard)
  ctx$graph_max_jaccard <- safe_number(result$graph_max_jaccard)
  ctx$graph_zero_jaccard_fraction <- safe_number(result$graph_zero_jaccard_fraction)
  ctx$graph_snn_k <- safe_number(result$graph_snn_k)
  ctx$graph_snn_prune_threshold <- safe_number(result$graph_snn_prune_threshold)
  ctx$graph_mean_snn_weight <- safe_number(result$graph_mean_snn_weight)
  ctx$graph_min_snn_weight <- safe_number(result$graph_min_snn_weight)
  ctx$graph_max_snn_weight <- safe_number(result$graph_max_snn_weight)
  ctx$graph_zero_snn_fraction <- safe_number(result$graph_zero_snn_fraction)
  ctx$localmap_false_neighbor_enabled <- safe_logical(result$localmap_false_neighbor_enabled, ctx$localmap_false_neighbor_enabled)
  ctx$localmap_false_neighbor_mode <- if (!is.null(result$localmap_false_neighbor_mode)) {
    as.character(result$localmap_false_neighbor_mode)
  } else {
    ctx$localmap_false_neighbor_mode
  }
  ctx$localmap_false_neighbor_transfer_mode <- if (!is.null(result$localmap_false_neighbor_transfer_mode)) {
    as.character(result$localmap_false_neighbor_transfer_mode)
  } else {
    ctx$localmap_false_neighbor_transfer_mode
  }
  ctx$localmap_false_neighbor_jaccard_threshold <- safe_number(result$localmap_false_neighbor_jaccard_threshold)
  ctx$localmap_false_neighbor_distance_quantile <- safe_number(result$localmap_false_neighbor_distance_quantile)
  ctx$localmap_false_neighbor_distance_multiplier <- safe_number(result$localmap_false_neighbor_distance_multiplier)
  ctx$localmap_false_neighbor_min_keep_fraction <- safe_number(result$localmap_false_neighbor_min_keep_fraction)
  ctx$localmap_false_neighbor_min_keep_k <- safe_number(result$localmap_false_neighbor_min_keep_k)
  ctx$localmap_false_neighbor_removed_edges_mean <- safe_number(result$localmap_false_neighbor_removed_edges_mean)
  ctx$localmap_false_neighbor_removed_fraction <- safe_number(result$localmap_false_neighbor_removed_fraction)
  ctx$localmap_false_neighbor_kept_degree_mean <- safe_number(result$localmap_false_neighbor_kept_degree_mean)
  ctx$localmap_false_neighbor_kept_jaccard_mean <- safe_number(result$localmap_false_neighbor_kept_jaccard_mean)
  ctx$localmap_false_neighbor_removed_jaccard_mean <- safe_number(result$localmap_false_neighbor_removed_jaccard_mean)
  ctx$localmap_false_neighbor_kept_distance_ratio_mean <- safe_number(result$localmap_false_neighbor_kept_distance_ratio_mean)
  ctx$localmap_false_neighbor_removed_distance_ratio_mean <- safe_number(result$localmap_false_neighbor_removed_distance_ratio_mean)
  ctx$localmap_false_neighbor_threshold_mean <- safe_number(result$localmap_false_neighbor_threshold_mean)
  ctx$localmap_local_weight_enabled <- safe_logical(result$localmap_local_weight_enabled, ctx$localmap_local_weight_enabled)
  ctx$localmap_local_weight <- safe_number(result$localmap_local_weight)
  ctx$localmap_local_weight_mode <- if (!is.null(result$localmap_local_weight_mode)) {
    as.character(result$localmap_local_weight_mode)
  } else {
    ctx$localmap_local_weight_mode
  }
  ctx$localmap_local_weight_transfer_mode <- if (!is.null(result$localmap_local_weight_transfer_mode)) {
    as.character(result$localmap_local_weight_transfer_mode)
  } else {
    ctx$localmap_local_weight_transfer_mode
  }
  ctx$localmap_local_weight_jaccard_blend <- safe_number(result$localmap_local_weight_jaccard_blend)
  ctx$localmap_local_weight_mean_trust <- safe_number(result$localmap_local_weight_mean_trust)
  ctx$localmap_local_weight_rank_component_mean <- safe_number(result$localmap_local_weight_rank_component_mean)
  ctx$localmap_local_weight_jaccard_component_mean <- safe_number(result$localmap_local_weight_jaccard_component_mean)
  ctx$localmap_local_weight_mean_multiplier <- safe_number(result$localmap_local_weight_mean_multiplier)
  ctx$localmap_local_weight_min_multiplier <- safe_number(result$localmap_local_weight_min_multiplier)
  ctx$localmap_local_weight_max_multiplier <- safe_number(result$localmap_local_weight_max_multiplier)
  ctx$localmap_local_weight_distance_scale_mean <- safe_number(result$localmap_local_weight_distance_scale_mean)
  ctx$graph_adaptive_min_k <- safe_number(result$graph_adaptive_min_k)
  ctx$graph_adaptive_max_k <- safe_number(result$graph_adaptive_max_k)
  ctx$graph_adaptive_mean_k <- safe_number(result$graph_adaptive_mean_k)
  ctx$graph_adaptive_density_quantile <- safe_number(result$graph_adaptive_density_quantile)
  ctx$graph_adaptive_density_cor <- safe_number(result$graph_adaptive_density_cor)
  ctx$graph_adaptive_dense_fraction <- safe_number(result$graph_adaptive_dense_fraction)
  ctx$graph_adaptive_sparse_fraction <- safe_number(result$graph_adaptive_sparse_fraction)
  ctx$graph_distance_prune_drop_fraction <- safe_number(result$graph_distance_prune_drop_fraction)
  ctx$graph_distance_prune_percentile <- safe_number(result$graph_distance_prune_percentile)
  ctx$graph_distance_prune_removed_edges_mean <- safe_number(result$graph_distance_prune_removed_edges_mean)
  ctx$graph_distance_prune_threshold_mean <- safe_number(result$graph_distance_prune_threshold_mean)
  ctx$graph_distance_prune_threshold_min <- safe_number(result$graph_distance_prune_threshold_min)
  ctx$graph_distance_prune_threshold_max <- safe_number(result$graph_distance_prune_threshold_max)
  ctx$graph_distance_prune_removed_distance_mean <- safe_number(result$graph_distance_prune_removed_distance_mean)
  ctx$graph_sparsification_method <- if (!is.null(result$graph_sparsification_method)) {
    as.character(result$graph_sparsification_method)
  } else {
    NA_character_
  }
  ctx$graph_sparsification_keep_fraction <- safe_number(result$graph_sparsification_keep_fraction)
  ctx$graph_sparsification_target_k <- safe_number(result$graph_sparsification_target_k)
  ctx$graph_sparsification_undirected_edges <- safe_number(result$graph_sparsification_undirected_edges)
  ctx$graph_sparsification_spectral_rank <- safe_number(result$graph_sparsification_spectral_rank)
  ctx$graph_sparsification_spectral_time_sec <- safe_number(result$graph_sparsification_spectral_time_sec)
  ctx$graph_sparsification_leverage_mean <- safe_number(result$graph_sparsification_leverage_mean)
  ctx$graph_sparsification_leverage_min <- safe_number(result$graph_sparsification_leverage_min)
  ctx$graph_sparsification_leverage_max <- safe_number(result$graph_sparsification_leverage_max)
  ctx$graph_sparsification_resistance_mean <- safe_number(result$graph_sparsification_resistance_mean)
  ctx$graph_sparsification_weight_mean <- safe_number(result$graph_sparsification_weight_mean)
  ctx$graph_mst_rescue_enabled <- safe_number(result$graph_mst_rescue_enabled)
  ctx$graph_mst_rescue_base_components <- safe_number(result$graph_mst_rescue_base_components)
  ctx$graph_mst_rescue_components_before <- safe_number(result$graph_mst_rescue_components_before)
  ctx$graph_mst_rescue_components_after <- safe_number(result$graph_mst_rescue_components_after)
  ctx$graph_mst_rescue_forest_edges <- safe_number(result$graph_mst_rescue_forest_edges)
  ctx$graph_mst_rescue_added_forest_edges <- safe_number(result$graph_mst_rescue_added_forest_edges)
  ctx$graph_mst_rescue_added_directed_edges <- safe_number(result$graph_mst_rescue_added_directed_edges)
  ctx$graph_mst_rescue_mean_degree_before <- safe_number(result$graph_mst_rescue_mean_degree_before)
  ctx$graph_mst_rescue_mean_degree_after <- safe_number(result$graph_mst_rescue_mean_degree_after)
  ctx$graph_density_correction_method <- if (!is.null(result$graph_density_correction_method)) {
    as.character(result$graph_density_correction_method)
  } else {
    NA_character_
  }
  ctx$graph_density_correction_quantile <- safe_number(result$graph_density_correction_quantile)
  ctx$graph_density_correction_strength <- safe_number(result$graph_density_correction_strength)
  ctx$graph_density_scale_mean <- safe_number(result$graph_density_scale_mean)
  ctx$graph_density_scale_min <- safe_number(result$graph_density_scale_min)
  ctx$graph_density_scale_max <- safe_number(result$graph_density_scale_max)
  ctx$graph_density_scale_cv <- safe_number(result$graph_density_scale_cv)
  ctx$graph_density_sparse_fraction <- safe_number(result$graph_density_sparse_fraction)
  ctx$graph_density_correction_mean <- safe_number(result$graph_density_correction_mean)
  ctx$graph_density_correction_min <- safe_number(result$graph_density_correction_min)
  ctx$graph_density_correction_max <- safe_number(result$graph_density_correction_max)
  ctx$graph_density_correction_clamp_fraction <- safe_number(result$graph_density_correction_clamp_fraction)
  ctx$graph_density_corrected_distance_scale_cor <- safe_number(result$graph_density_corrected_distance_scale_cor)
  ctx$umap_graph_set_op_mix_ratio <- safe_number(result$umap_graph_set_op_mix_ratio)
  ctx$umap_graph_local_connectivity <- safe_number(result$umap_graph_local_connectivity)
  ctx$umap_graph_weight_power <- safe_number(result$umap_graph_weight_power)
  ctx$umap_graph_target_scale <- safe_number(result$umap_graph_target_scale)
  ctx$umap_graph_distance_transform <- if (!is.null(result$umap_graph_distance_transform)) {
    as.character(result$umap_graph_distance_transform)
  } else {
    NA_character_
  }
  ctx$umap_graph_mean_weight <- safe_number(result$umap_graph_mean_weight)
  ctx$umap_graph_min_weight <- safe_number(result$umap_graph_min_weight)
  ctx$umap_graph_max_weight <- safe_number(result$umap_graph_max_weight)
  ctx$graph_edge_sampling_method <- if (!is.null(result$graph_edge_sampling_method)) {
    as.character(result$graph_edge_sampling_method)
  } else {
    NA_character_
  }
  ctx$graph_edge_sampling_fraction <- safe_number(result$graph_edge_sampling_fraction)
  ctx$graph_edge_sampling_weight_power <- safe_number(result$graph_edge_sampling_weight_power)
  ctx$graph_edge_sampling_include_top <- safe_number(result$graph_edge_sampling_include_top)
  ctx$graph_edge_sampling_target_scale <- safe_number(result$graph_edge_sampling_target_scale)
  ctx$graph_edge_sampling_mean_selected_weight <- safe_number(result$graph_edge_sampling_mean_selected_weight)
  ctx$graph_edge_sampling_mean_candidate_weight <- safe_number(result$graph_edge_sampling_mean_candidate_weight)
  ctx$graph_edge_sampling_selected_to_candidate_weight_ratio <- safe_number(result$graph_edge_sampling_selected_to_candidate_weight_ratio)
  ctx$graph_tsne_affinity_mode <- if (!is.null(result$graph_tsne_affinity_mode)) {
    as.character(result$graph_tsne_affinity_mode)
  } else {
    NA_character_
  }
  ctx$graph_tsne_affinity_perplexities <- if (!is.null(result$graph_tsne_affinity_perplexities)) {
    as.character(result$graph_tsne_affinity_perplexities)
  } else {
    NA_character_
  }
  ctx$graph_tsne_affinity_num_scales <- safe_number(result$graph_tsne_affinity_num_scales)
  ctx$graph_tsne_affinity_temperature <- safe_number(result$graph_tsne_affinity_temperature)
  ctx$graph_tsne_affinity_entropy_mean <- safe_number(result$graph_tsne_affinity_entropy_mean)
  ctx$graph_tsne_affinity_effective_perplexity_mean <- safe_number(result$graph_tsne_affinity_effective_perplexity_mean)
  ctx$graph_tsne_affinity_sigma_mean <- safe_number(result$graph_tsne_affinity_sigma_mean)
  ctx$graph_tsne_affinity_sigma_min <- safe_number(result$graph_tsne_affinity_sigma_min)
  ctx$graph_tsne_affinity_sigma_max <- safe_number(result$graph_tsne_affinity_sigma_max)
  ctx$graph_tsne_affinity_prob_min <- safe_number(result$graph_tsne_affinity_prob_min)
  ctx$graph_tsne_affinity_prob_max <- safe_number(result$graph_tsne_affinity_prob_max)
  ctx$graph_multiscale_perplexities <- if (!is.null(result$graph_multiscale_perplexities)) {
    as.character(result$graph_multiscale_perplexities)
  } else {
    NA_character_
  }
  ctx$graph_multiscale_num_scales <- safe_number(result$graph_multiscale_num_scales)
  ctx$graph_multiscale_required_k <- safe_number(result$graph_multiscale_required_k)
  ctx$graph_multiscale_effective_k_values <- if (!is.null(result$graph_multiscale_effective_k_values)) {
    as.character(result$graph_multiscale_effective_k_values)
  } else {
    NA_character_
  }
  ctx$graph_multiscale_transfer_mode <- if (!is.null(result$graph_multiscale_transfer_mode)) {
    as.character(result$graph_multiscale_transfer_mode)
  } else {
    NA_character_
  }
  ctx$graph_multiscale_uses_tsne_affinity <- safe_number(result$graph_multiscale_uses_tsne_affinity)
  ctx$pacmap_transfer_mode <- if (!is.null(result$pacmap_transfer_mode)) {
    as.character(result$pacmap_transfer_mode)
  } else {
    ctx$pacmap_transfer_mode
  }
  ctx$pacmap_auxiliary_pair_family <- if (!is.null(result$pacmap_auxiliary_pair_family)) {
    as.character(result$pacmap_auxiliary_pair_family)
  } else {
    ctx$pacmap_auxiliary_pair_family
  }
  ctx$pacmap_mid_near_pairs_per_point <- safe_number(result$pacmap_mid_near_pairs_per_point, ctx$pacmap_mid_near_pairs_per_point)
  ctx$pacmap_mid_near_fraction <- safe_number(result$pacmap_mid_near_fraction, ctx$pacmap_mid_near_fraction)
  ctx$pacmap_mid_near_requested_fraction <- safe_number(result$pacmap_mid_near_requested_fraction, ctx$pacmap_mid_near_requested_fraction)
  ctx$pacmap_mid_near_distance_scale <- safe_number(result$pacmap_mid_near_distance_scale, ctx$pacmap_mid_near_distance_scale)
  ctx$pacmap_mid_near_fallback_fraction <- safe_number(result$pacmap_mid_near_fallback_fraction, ctx$pacmap_mid_near_fallback_fraction)
  ctx$pacmap_mid_near_rank_mean <- safe_number(result$pacmap_mid_near_rank_mean, ctx$pacmap_mid_near_rank_mean)
  ctx$pacmap_mid_near_emphasis_strength <- safe_number(result$pacmap_mid_near_emphasis_strength, ctx$pacmap_mid_near_emphasis_strength)
  ctx$pacmap_mid_near_emphasis_distance_multiplier <- safe_number(result$pacmap_mid_near_emphasis_distance_multiplier, ctx$pacmap_mid_near_emphasis_distance_multiplier)
  ctx$pacmap_near_ratio <- safe_number(result$pacmap_near_ratio, ctx$pacmap_near_ratio)
  ctx$pacmap_mid_ratio <- safe_number(result$pacmap_mid_ratio, ctx$pacmap_mid_ratio)
  ctx$pacmap_far_ratio <- safe_number(result$pacmap_far_ratio, ctx$pacmap_far_ratio)
  ctx$pacmap_near_pairs_per_point <- safe_number(result$pacmap_near_pairs_per_point, ctx$pacmap_near_pairs_per_point)
  ctx$pacmap_mid_pairs_per_point <- safe_number(result$pacmap_mid_pairs_per_point, ctx$pacmap_mid_pairs_per_point)
  ctx$pacmap_far_pairs_per_point <- safe_number(result$pacmap_far_pairs_per_point, ctx$pacmap_far_pairs_per_point)
  ctx$pacmap_far_pair_fraction <- safe_number(result$pacmap_far_pair_fraction, ctx$pacmap_far_pair_fraction)
  ctx$pacmap_far_distance_scale <- safe_number(result$pacmap_far_distance_scale, ctx$pacmap_far_distance_scale)
  ctx$pacmap_far_fallback_fraction <- safe_number(result$pacmap_far_fallback_fraction, ctx$pacmap_far_fallback_fraction)
  ctx$pacmap_far_repulsion_rate <- safe_number(result$pacmap_far_repulsion_rate, ctx$pacmap_far_repulsion_rate)
  ctx$pacmap_phase_schedule <- if (!is.null(result$pacmap_phase_schedule)) {
    as.character(result$pacmap_phase_schedule)
  } else {
    ctx$pacmap_phase_schedule
  }
  ctx$pacmap_phase_total_epochs <- safe_number(result$pacmap_phase_total_epochs, ctx$pacmap_phase_total_epochs)
  ctx$pacmap_phase_epoch_multiplier <- safe_number(result$pacmap_phase_epoch_multiplier, ctx$pacmap_phase_epoch_multiplier)
  ctx$pacmap_phase_warmup_epochs <- safe_number(result$pacmap_phase_warmup_epochs, ctx$pacmap_phase_warmup_epochs)
  ctx$pacmap_phase_refine_epochs <- safe_number(result$pacmap_phase_refine_epochs, ctx$pacmap_phase_refine_epochs)
  ctx$pacmap_phase_transfer_detail <- if (!is.null(result$pacmap_phase_transfer_detail)) {
    as.character(result$pacmap_phase_transfer_detail)
  } else {
    ctx$pacmap_phase_transfer_detail
  }
  ctx$trimap_transfer_mode <- if (!is.null(result$trimap_transfer_mode)) {
    as.character(result$trimap_transfer_mode)
  } else {
    ctx$trimap_transfer_mode
  }
  ctx$trimap_triplet_family <- if (!is.null(result$trimap_triplet_family)) {
    as.character(result$trimap_triplet_family)
  } else {
    ctx$trimap_triplet_family
  }
  ctx$trimap_inlier_ratio <- safe_number(result$trimap_inlier_ratio, ctx$trimap_inlier_ratio)
  ctx$trimap_semihard_ratio <- safe_number(result$trimap_semihard_ratio, ctx$trimap_semihard_ratio)
  ctx$trimap_global_anchor_ratio <- safe_number(result$trimap_global_anchor_ratio, ctx$trimap_global_anchor_ratio)
  ctx$trimap_inlier_pairs_per_point <- safe_number(result$trimap_inlier_pairs_per_point, ctx$trimap_inlier_pairs_per_point)
  ctx$trimap_semihard_pairs_per_point <- safe_number(result$trimap_semihard_pairs_per_point, ctx$trimap_semihard_pairs_per_point)
  ctx$trimap_global_anchor_pairs_per_point <- safe_number(result$trimap_global_anchor_pairs_per_point, ctx$trimap_global_anchor_pairs_per_point)
  ctx$trimap_semihard_fraction <- safe_number(result$trimap_semihard_fraction, ctx$trimap_semihard_fraction)
  ctx$trimap_global_anchor_fraction <- safe_number(result$trimap_global_anchor_fraction, ctx$trimap_global_anchor_fraction)
  ctx$trimap_semihard_distance_scale <- safe_number(result$trimap_semihard_distance_scale, ctx$trimap_semihard_distance_scale)
  ctx$trimap_global_anchor_distance_scale <- safe_number(result$trimap_global_anchor_distance_scale, ctx$trimap_global_anchor_distance_scale)
  ctx$trimap_semihard_fallback_fraction <- safe_number(result$trimap_semihard_fallback_fraction, ctx$trimap_semihard_fallback_fraction)
  ctx$trimap_global_anchor_fallback_fraction <- safe_number(result$trimap_global_anchor_fallback_fraction, ctx$trimap_global_anchor_fallback_fraction)
  ctx$trimap_semihard_rank_mean <- safe_number(result$trimap_semihard_rank_mean, ctx$trimap_semihard_rank_mean)
  ctx$trimap_candidate_seed <- safe_number(result$trimap_candidate_seed, ctx$trimap_candidate_seed)
  ctx$trimap_native_explicit_triplets <- safe_number(result$trimap_native_explicit_triplets, ctx$trimap_native_explicit_triplets)
  ctx$trimap_triplet_proxy_detail <- if (!is.null(result$trimap_triplet_proxy_detail)) {
    as.character(result$trimap_triplet_proxy_detail)
  } else {
    ctx$trimap_triplet_proxy_detail
  }
  ctx
}

tsne_barnes_hut_label <- function(theta) {
  theta <- as.numeric(theta)
  if (!is.finite(theta)) return("auto")
  if (abs(theta) < .Machine$double.eps) return("0")
  gsub("\\.", "", sprintf("%.1f", theta))
}

tsne_barnes_hut_config <- function(ctx, theta, k = ctx$k) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(k),
    objective = "tsne",
    quality = "fast",
    backend = "cpu"
  )
  cfg$tsne_mode <- "barnes_hut"
  cfg$theta <- as.numeric(theta)
  cfg$optimizer_backend <- "cpu_barnes_hut"
  cfg$affinity_backend <- "cpu_rtsne"
  cfg
}

run_tsne_barnes_hut <- function(ctx, theta) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- tsne_barnes_hut_config(ctx, theta, ncol(knn$indices))
  init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  layout <- fastEmbedR:::knn_tsne_neighbors_cpp(
    knn$indices,
    knn$distances,
    init,
    as.integer(cfg$n_epochs),
    cfg$perplexity,
    as.numeric(theta),
    cfg$learning_rate,
    as.integer(cfg$stop_lying_iter),
    as.integer(cfg$mom_switch_iter),
    cfg$momentum,
    cfg$final_momentum,
    cfg$exaggeration_factor,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    FALSE
  )
  colnames(layout) <- paste0("TSNE", seq_len(ncol(layout)))
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

tsne_barnes_hut_strategy <- function(theta) {
  theta <- as.numeric(theta)
  list(
    id = paste0("tsne_barnes_hut_theta", tsne_barnes_hut_label(theta)),
    family = "tsne_repulsion",
    description = paste0(
      "t-SNE Barnes-Hut repulsive-force approximation from KNN affinities with theta = ",
      theta,
      ". This is a t-SNE-only baseline; theta = 0 disables tree approximation."
    ),
    compatible = function(method, backend) identical(method, "tsne") && identical(backend, "cpu"),
    params = function(ctx) {
      cfg <- tsne_barnes_hut_config(ctx, theta)
      list(
        k = ctx$k,
        quality = "fast",
        optimizer = "barnes_hut",
        theta = theta,
        perplexity = cfg$perplexity,
        n_epochs = cfg$n_epochs,
        learning_rate = cfg$learning_rate,
        stop_lying_iter = cfg$stop_lying_iter,
        mom_switch_iter = cfg$mom_switch_iter,
        momentum = cfg$momentum,
        final_momentum = cfg$final_momentum,
        exaggeration_factor = cfg$exaggeration_factor,
        n_threads = cfg$n_threads,
        transfer_scope = "tsne_only_all_pair_repulsion"
      )
    },
    run = function(ctx) run_tsne_barnes_hut(ctx, theta),
    tsne_bh_theta = theta
  )
}

tsne_barnes_hut_strategy_grid <- function() {
  lapply(c(0, 0.3, 0.5, 0.8), tsne_barnes_hut_strategy)
}

early_exaggeration_label <- function(exaggeration, duration_fraction) {
  paste0(
    "x", as.integer(exaggeration),
    "_d", as.integer(round(100 * as.numeric(duration_fraction)))
  )
}

early_exaggeration_transfer_mode <- function(method) {
  switch(
    method,
    tsne = "tsne_true_early_exaggeration",
    umap = "umap_edge_weight_early_exaggeration",
    pacmap = "pacmap_near_pair_early_exaggeration",
    trimap = "trimap_inlier_triplet_proxy_early_exaggeration",
    localmap = "local_edge_early_exaggeration",
    "objective_attraction_early_exaggeration"
  )
}

early_exaggeration_config <- function(ctx,
                                      exaggeration,
                                      duration_fraction,
                                      k = ctx$k,
                                      n = ctx$n) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(
      n = as.integer(n),
      k = as.integer(k),
      backend = "cpu"
    )
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = as.integer(n),
      k = as.integer(k),
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  total_epochs <- max(2L, as.integer(cfg$n_epochs))
  warmup_epochs <- as.integer(ceiling(total_epochs * as.numeric(duration_fraction)))
  warmup_epochs <- max(1L, min(total_epochs - 1L, warmup_epochs))
  cfg$n_epochs <- total_epochs
  cfg$early_exaggeration_factor <- as.numeric(exaggeration)
  cfg$early_exaggeration_duration_fraction <- as.numeric(duration_fraction)
  cfg$early_exaggeration_warmup_epochs <- warmup_epochs
  cfg$early_exaggeration_refine_epochs <- total_epochs - warmup_epochs
  cfg$early_exaggeration_transfer_mode <- early_exaggeration_transfer_mode(ctx$method)
  cfg$early_exaggeration_distance_scale <- 1 / as.numeric(exaggeration)
  cfg$early_exaggeration_schedule_mode <- if (identical(ctx$method, "tsne")) {
    "native_tsne_lying"
  } else {
    "two_stage_distance_scaled_warmup"
  }
  cfg$backend <- "cpu"
  cfg
}

early_exaggerated_knn <- function(knn, exaggeration) {
  out <- knn
  out$distances <- knn$distances / as.numeric(exaggeration)
  out
}

run_early_exaggeration_schedule <- function(ctx, exaggeration, duration_fraction) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- early_exaggeration_config(
    ctx,
    exaggeration,
    duration_fraction,
    k = ncol(knn$indices),
    n = nrow(knn$indices)
  )

  if (identical(ctx$method, "tsne")) {
    init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    cfg$tsne_mode <- "barnes_hut"
    cfg$theta <- if (is.finite(safe_number(cfg$theta))) as.numeric(cfg$theta) else 0.5
    cfg$optimizer_backend <- "cpu_barnes_hut"
    cfg$affinity_backend <- "cpu_rtsne"
    cfg$stop_lying_iter <- as.integer(cfg$early_exaggeration_warmup_epochs)
    cfg$mom_switch_iter <- as.integer(cfg$early_exaggeration_warmup_epochs)
    cfg$exaggeration_factor <- as.numeric(exaggeration)
    layout <- fastEmbedR:::knn_tsne_neighbors_cpp(
      knn$indices,
      knn$distances,
      init,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    colnames(layout) <- paste0("TSNE", seq_len(ncol(layout)))
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = as.integer(cfg$spectral_n_iter),
    seed = as.integer(ctx$seed),
    backend = "cpu"
  )
  warm_knn <- early_exaggerated_knn(knn, exaggeration)
  if (identical(ctx$method, "umap")) {
    warm_layout <- fastEmbedR:::knn_umap_refine_cpp(
      warm_knn$indices,
      warm_knn$distances,
      init,
      as.integer(cfg$early_exaggeration_warmup_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::knn_umap_refine_cpp(
      knn$indices,
      knn$distances,
      warm_layout,
      as.integer(cfg$early_exaggeration_refine_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed + 1009L),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
  } else {
    warm_layout <- fastEmbedR:::knn_objective_embed_cpp(
      warm_knn$indices,
      warm_knn$distances,
      ctx$method,
      init,
      2L,
      as.integer(cfg$early_exaggeration_warmup_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE,
      FALSE
    )
    layout <- fastEmbedR:::knn_objective_embed_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      warm_layout,
      2L,
      as.integer(cfg$early_exaggeration_refine_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed + 1009L),
      FALSE,
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(
      layout,
      fastEmbedR:::objective_prefix(ctx$method)
    )
  }
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

early_exaggeration_strategy <- function(exaggeration, duration_fraction) {
  exaggeration <- as.numeric(exaggeration)
  duration_fraction <- as.numeric(duration_fraction)
  label <- early_exaggeration_label(exaggeration, duration_fraction)
  list(
    id = paste0("early_exag_", label),
    family = "optimization_schedule",
    description = paste0(
      "Early-exaggeration schedule with attraction factor ",
      exaggeration,
      " for the first ",
      round(100 * duration_fraction),
      "% of optimizer epochs. t-SNE uses native early exaggeration; other objectives use an experimental distance-scaled attraction warmup followed by normal refinement."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    params = function(ctx) {
      cfg <- early_exaggeration_config(ctx, exaggeration, duration_fraction)
      list(
        k = ctx$k,
        exaggeration = exaggeration,
        duration_fraction = duration_fraction,
        total_epochs = cfg$n_epochs,
        warmup_epochs = cfg$early_exaggeration_warmup_epochs,
        refine_epochs = cfg$early_exaggeration_refine_epochs,
        transfer_mode = cfg$early_exaggeration_transfer_mode,
        distance_scale = cfg$early_exaggeration_distance_scale,
        schedule_mode = cfg$early_exaggeration_schedule_mode,
        backend = "cpu"
      )
    },
    run = function(ctx) run_early_exaggeration_schedule(ctx, exaggeration, duration_fraction),
    early_exaggeration_factor = exaggeration,
    early_exaggeration_duration_fraction = duration_fraction
  )
}

early_exaggeration_strategy_grid <- function() {
  out <- list()
  factors <- c(4, 8, 12)
  durations <- c(0.10, 0.20, 0.30)
  for (factor in factors) {
    for (duration in durations) {
      out[[length(out) + 1L]] <- early_exaggeration_strategy(factor, duration)
    }
  }
  out
}

late_exaggeration_label <- function(exaggeration, duration_fraction) {
  paste0(
    "x", as.integer(exaggeration),
    "_d", as.integer(round(100 * as.numeric(duration_fraction)))
  )
}

late_exaggeration_transfer_mode <- function(method) {
  switch(
    method,
    tsne = "tsne_native_late_exaggeration",
    umap = "umap_edge_weight_late_exaggeration",
    pacmap = "pacmap_near_pair_late_exaggeration",
    trimap = "trimap_inlier_triplet_proxy_late_exaggeration",
    localmap = "local_edge_late_exaggeration",
    "objective_attraction_late_exaggeration"
  )
}

late_exaggeration_config <- function(ctx,
                                     exaggeration,
                                     duration_fraction,
                                     k = ctx$k,
                                     n = ctx$n) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(
      n = as.integer(n),
      k = as.integer(k),
      backend = "cpu"
    )
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = as.integer(n),
      k = as.integer(k),
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  total_epochs <- max(2L, as.integer(cfg$n_epochs))
  requested_late_epochs <- as.integer(ceiling(total_epochs * as.numeric(duration_fraction)))
  requested_late_epochs <- max(1L, min(total_epochs - 1L, requested_late_epochs))
  late_start_iter <- total_epochs - requested_late_epochs + 1L
  if (identical(ctx$method, "tsne")) {
    early_stop <- max(0L, as.integer(cfg$stop_lying_iter))
    late_start_iter <- max(late_start_iter, early_stop + 1L)
  }
  late_start_iter <- max(1L, min(total_epochs + 1L, late_start_iter))
  late_epochs <- max(0L, total_epochs - late_start_iter + 1L)
  normal_epochs <- max(1L, total_epochs - late_epochs)
  if (late_epochs < 1L) {
    late_epochs <- 1L
    normal_epochs <- total_epochs - 1L
    late_start_iter <- normal_epochs + 1L
  }
  cfg$n_epochs <- total_epochs
  cfg$late_exaggeration_factor <- as.numeric(exaggeration)
  cfg$late_exaggeration_duration_fraction <- as.numeric(duration_fraction)
  cfg$late_exaggeration_requested_epochs <- requested_late_epochs
  cfg$late_exaggeration_normal_epochs <- normal_epochs
  cfg$late_exaggeration_late_epochs <- late_epochs
  cfg$late_exaggeration_start_iter <- late_start_iter
  cfg$late_exaggeration_transfer_mode <- late_exaggeration_transfer_mode(ctx$method)
  cfg$late_exaggeration_distance_scale <- 1 / as.numeric(exaggeration)
  cfg$late_exaggeration_schedule_mode <- if (identical(ctx$method, "tsne")) {
    "native_tsne_late_lying"
  } else {
    "normal_then_distance_scaled_late_refine"
  }
  cfg$backend <- "cpu"
  cfg
}

run_late_exaggeration_schedule <- function(ctx, exaggeration, duration_fraction) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- late_exaggeration_config(
    ctx,
    exaggeration,
    duration_fraction,
    k = ncol(knn$indices),
    n = nrow(knn$indices)
  )

  if (identical(ctx$method, "tsne")) {
    late_runner <- get(
      "knn_tsne_neighbors_late_exag_cpp",
      envir = asNamespace("fastEmbedR"),
      inherits = FALSE
    )
    init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    cfg$tsne_mode <- "barnes_hut_late_exaggeration"
    cfg$theta <- if (is.finite(safe_number(cfg$theta))) as.numeric(cfg$theta) else 0.5
    cfg$optimizer_backend <- "cpu_barnes_hut_late_exaggeration"
    cfg$affinity_backend <- "cpu_rtsne"
    layout <- late_runner(
      knn$indices,
      knn$distances,
      init,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.numeric(exaggeration),
      as.integer(cfg$late_exaggeration_start_iter),
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    colnames(layout) <- paste0("TSNE", seq_len(ncol(layout)))
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = as.integer(cfg$spectral_n_iter),
    seed = as.integer(ctx$seed),
    backend = "cpu"
  )
  late_knn <- early_exaggerated_knn(knn, exaggeration)
  if (identical(ctx$method, "umap")) {
    normal_layout <- fastEmbedR:::knn_umap_refine_cpp(
      knn$indices,
      knn$distances,
      init,
      as.integer(cfg$late_exaggeration_normal_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::knn_umap_refine_cpp(
      late_knn$indices,
      late_knn$distances,
      normal_layout,
      as.integer(cfg$late_exaggeration_late_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed + 2003L),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
  } else {
    normal_layout <- fastEmbedR:::knn_objective_embed_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      init,
      2L,
      as.integer(cfg$late_exaggeration_normal_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE,
      FALSE
    )
    layout <- fastEmbedR:::knn_objective_embed_cpp(
      late_knn$indices,
      late_knn$distances,
      ctx$method,
      normal_layout,
      2L,
      as.integer(cfg$late_exaggeration_late_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed + 2003L),
      FALSE,
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(
      layout,
      fastEmbedR:::objective_prefix(ctx$method)
    )
  }
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

late_exaggeration_strategy <- function(exaggeration, duration_fraction) {
  exaggeration <- as.numeric(exaggeration)
  duration_fraction <- as.numeric(duration_fraction)
  label <- late_exaggeration_label(exaggeration, duration_fraction)
  list(
    id = paste0("late_exag_", label),
    family = "optimization_schedule",
    description = paste0(
      "Late-exaggeration schedule with attraction factor ",
      exaggeration,
      " for the final ",
      round(100 * duration_fraction),
      "% of optimizer epochs. Expected effect: sharper clusters, with possible loss of global or continuous structure."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      has_late_tsne <- exists(
        "knn_tsne_neighbors_late_exag_cpp",
        envir = asNamespace("fastEmbedR"),
        inherits = FALSE
      )
      list(
        available = has_late_tsne,
        message = "fastEmbedR must be reinstalled after adding the native late-exaggeration t-SNE C++ entry point."
      )
    },
    params = function(ctx) {
      cfg <- late_exaggeration_config(ctx, exaggeration, duration_fraction)
      list(
        k = ctx$k,
        exaggeration = exaggeration,
        duration_fraction = duration_fraction,
        total_epochs = cfg$n_epochs,
        normal_epochs = cfg$late_exaggeration_normal_epochs,
        late_epochs = cfg$late_exaggeration_late_epochs,
        late_start_iter = cfg$late_exaggeration_start_iter,
        transfer_mode = cfg$late_exaggeration_transfer_mode,
        distance_scale = cfg$late_exaggeration_distance_scale,
        schedule_mode = cfg$late_exaggeration_schedule_mode,
        expected_benefit = "sharper_cluster_separation",
        expected_risk = "possible_global_or_continuous_structure_loss",
        backend = "cpu"
      )
    },
    run = function(ctx) run_late_exaggeration_schedule(ctx, exaggeration, duration_fraction),
    late_exaggeration_factor = exaggeration,
    late_exaggeration_duration_fraction = duration_fraction
  )
}

late_exaggeration_strategy_grid <- function() {
  out <- list()
  factors <- c(2, 4, 8)
  durations <- c(0.10, 0.20, 0.30)
  for (factor in factors) {
    for (duration in durations) {
      out[[length(out) + 1L]] <- late_exaggeration_strategy(factor, duration)
    }
  }
  out
}

optimizer_transfer_mode <- function(method, optimizer) {
  if (identical(method, "tsne")) {
    return(paste0("tsne_native_", optimizer, "_optimizer"))
  }
  paste0(method, "_knn_objective_", optimizer, "_transfer")
}

optimizer_schedule_config <- function(ctx, optimizer) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  optimizer <- as.character(optimizer)
  cfg$optimizer_mode <- optimizer
  cfg$optimizer_schedule <- switch(
    optimizer,
    momentum = "momentum_0.5_to_0.8",
    nesterov = "nesterov_0.5_to_0.8",
    adam = "adam_beta1_0.9_beta2_0.999",
    adagrad = "adagrad_accumulated_squared_gradient",
    optimizer
  )
  cfg$optimizer_momentum <- 0.5
  cfg$optimizer_final_momentum <- 0.8
  cfg$optimizer_switch_iter <- if (identical(ctx$method, "tsne")) {
    as.integer(cfg$mom_switch_iter)
  } else {
    max(1L, as.integer(ceiling(as.integer(cfg$n_epochs) * 0.5)))
  }
  cfg$optimizer_adam_beta1 <- 0.9
  cfg$optimizer_adam_beta2 <- 0.999
  cfg$optimizer_adam_epsilon <- 1e-8
  cfg$optimizer_learning_rate_multiplier <- if (identical(optimizer, "adam")) {
    if (identical(ctx$method, "tsne")) 0.02 else 0.25
  } else if (identical(optimizer, "adagrad")) {
    if (identical(ctx$method, "tsne")) 0.01 else 0.10
  } else {
    1
  }
  cfg$optimizer_learning_rate <- as.numeric(cfg$learning_rate) * cfg$optimizer_learning_rate_multiplier
  cfg$optimizer_transfer_mode <- optimizer_transfer_mode(ctx$method, optimizer)
  cfg$optimizer_backend <- "cpu_cpp"
  cfg$backend <- "cpu"
  cfg
}

run_optimizer_schedule <- function(ctx, optimizer) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- optimizer_schedule_config(ctx, optimizer)
  if (identical(ctx$method, "tsne")) {
    runner <- get(
      "knn_tsne_neighbors_optimizer_cpp",
      envir = asNamespace("fastEmbedR"),
      inherits = FALSE
    )
    init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    cfg$tsne_mode <- paste0("barnes_hut_", optimizer)
    cfg$affinity_backend <- "cpu_rtsne"
    layout <- runner(
      knn$indices,
      knn$distances,
      init,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$optimizer_learning_rate,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$optimizer_switch_iter),
      cfg$optimizer_momentum,
      cfg$optimizer_final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      optimizer,
      cfg$optimizer_adam_beta1,
      cfg$optimizer_adam_beta2,
      cfg$optimizer_adam_epsilon,
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "TSNE")
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  runner <- get(
    "knn_objective_embed_optimizer_cpp",
    envir = asNamespace("fastEmbedR"),
    inherits = FALSE
  )
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = as.integer(cfg$spectral_n_iter),
    seed = as.integer(ctx$seed),
    backend = "cpu"
  )
  objective <- if (identical(ctx$method, "umap")) "umap" else ctx$method
  layout <- runner(
    knn$indices,
    knn$distances,
    objective,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$optimizer_learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    optimizer,
    cfg$optimizer_momentum,
    cfg$optimizer_final_momentum,
    as.integer(cfg$optimizer_switch_iter),
    cfg$optimizer_adam_beta1,
    cfg$optimizer_adam_beta2,
    cfg$optimizer_adam_epsilon,
    FALSE
  )
  layout <- fastEmbedR:::set_embedding_colnames(
    layout,
    if (identical(ctx$method, "umap")) "UMAP" else fastEmbedR:::objective_prefix(ctx$method)
  )
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

optimizer_schedule_strategy <- function(id, optimizer, description) {
  list(
    id = id,
    family = "optimization_schedule",
    description = description,
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_tsne_neighbors_optimizer_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_optimizer_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "fastEmbedR must be reinstalled after adding optimizer-schedule C++ entry points."
      )
    },
    params = function(ctx) {
      cfg <- optimizer_schedule_config(ctx, optimizer)
      list(
        k = ctx$k,
        optimizer = optimizer,
        optimizer_schedule = cfg$optimizer_schedule,
        momentum = cfg$optimizer_momentum,
        final_momentum = cfg$optimizer_final_momentum,
        switch_iter = cfg$optimizer_switch_iter,
        adam_beta1 = cfg$optimizer_adam_beta1,
        adam_beta2 = cfg$optimizer_adam_beta2,
        adam_epsilon = cfg$optimizer_adam_epsilon,
        learning_rate = cfg$optimizer_learning_rate,
        learning_rate_multiplier = cfg$optimizer_learning_rate_multiplier,
        transfer_mode = cfg$optimizer_transfer_mode,
        backend = "cpu"
      )
    },
    run = function(ctx) run_optimizer_schedule(ctx, optimizer),
    optimizer_mode = optimizer
  )
}

optimizer_schedule_strategy_grid <- function() {
  list(
    optimizer_schedule_strategy(
      "momentum_schedule_05_08",
      "momentum",
      "Momentum schedule transferred from t-SNE: momentum 0.5 before the switch iteration and 0.8 afterwards."
    ),
    optimizer_schedule_strategy(
      "optimizer_adam",
      "adam",
      "Adam optimizer transfer with beta1 = 0.9 and beta2 = 0.999. Uses a recorded conservative learning-rate multiplier for stability."
    ),
    optimizer_schedule_strategy(
      "optimizer_adagrad",
      "adagrad",
      "AdaGrad optimizer transfer with per-coordinate accumulated squared gradients and a conservative learning-rate multiplier."
    ),
    optimizer_schedule_strategy(
      "optimizer_nesterov",
      "nesterov",
      "Nesterov momentum transfer using the same 0.5 to 0.8 schedule as t-SNE momentum."
    )
  )
}

learning_rate_transfer_mode <- function(method, rule) {
  paste0(method, "_native_cpp_lr_", rule)
}

learning_rate_scaling_config <- function(ctx, rule) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  rule <- as.character(rule)
  base_learning_rate <- as.numeric(cfg$learning_rate)
  scaled_learning_rate <- switch(
    rule,
    fixed_1 = 1,
    fixed_10 = 10,
    n_over_12 = as.numeric(ctx$n) / 12,
    sqrt_n = sqrt(as.numeric(ctx$n)),
    method_default = base_learning_rate,
    stop("Unknown learning-rate rule: ", rule, call. = FALSE)
  )
  cfg$learning_rate_rule <- rule
  cfg$learning_rate_value <- as.numeric(scaled_learning_rate)
  cfg$learning_rate_base_default <- base_learning_rate
  cfg$learning_rate_scale <- if (base_learning_rate > 0) {
    as.numeric(scaled_learning_rate) / base_learning_rate
  } else {
    NA_real_
  }
  cfg$learning_rate_transfer_mode <- learning_rate_transfer_mode(ctx$method, rule)
  cfg$optimizer_backend <- "cpu_cpp_learning_rate_scaling"
  cfg$backend <- "cpu"
  cfg
}

run_learning_rate_scaling <- function(ctx, rule) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- learning_rate_scaling_config(ctx, rule)
  if (identical(ctx$method, "umap")) {
    layout <- fastEmbedR:::fast_knn_umap_cpp(
      knn$indices,
      knn$distances,
      2L,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate_value,
      cfg$repulsion_strength,
      as.integer(cfg$spectral_n_iter),
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$init_backend <- "cpu"
    cfg$graph_prep_backend <- "cpu"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  if (identical(ctx$method, "tsne")) {
    runner <- get(
      "knn_tsne_neighbors_cpp",
      envir = asNamespace("fastEmbedR"),
      inherits = FALSE
    )
    init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    layout <- runner(
      knn$indices,
      knn$distances,
      init,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate_value,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "TSNE")
    cfg$tsne_mode <- "rtsne_neighbors_learning_rate_scaling"
    cfg$affinity_backend <- "cpu_rtsne"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  runner <- get(
    "knn_objective_embed_cpp",
    envir = asNamespace("fastEmbedR"),
    inherits = FALSE
  )
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = as.integer(cfg$spectral_n_iter),
    seed = as.integer(ctx$seed),
    backend = "cpu"
  )
  layout <- runner(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate_value,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    FALSE,
    FALSE
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, fastEmbedR:::objective_prefix(ctx$method))
  cfg$init_backend <- "cpu"
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

learning_rate_scaling_strategy <- function(id, rule, description) {
  list(
    id = id,
    family = "optimization_schedule",
    description = description,
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("fast_knn_umap_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_tsne_neighbors_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "fastEmbedR must be installed with native CPU embedding entry points."
      )
    },
    params = function(ctx) {
      cfg <- learning_rate_scaling_config(ctx, rule)
      list(
        k = ctx$k,
        learning_rate_rule = rule,
        learning_rate = cfg$learning_rate_value,
        method_default_learning_rate = cfg$learning_rate_base_default,
        learning_rate_scale = cfg$learning_rate_scale,
        n = ctx$n,
        transfer_mode = cfg$learning_rate_transfer_mode,
        n_epochs = cfg$n_epochs,
        backend = "cpu"
      )
    },
    run = function(ctx) run_learning_rate_scaling(ctx, rule),
    learning_rate_rule = rule
  )
}

learning_rate_scaling_strategy_grid <- function() {
  list(
    learning_rate_scaling_strategy(
      "lr_fixed_1",
      "fixed_1",
      "Learning-rate scaling probe with lr = 1."
    ),
    learning_rate_scaling_strategy(
      "lr_fixed_10",
      "fixed_10",
      "Learning-rate scaling probe with lr = 10."
    ),
    learning_rate_scaling_strategy(
      "lr_n_over_12",
      "n_over_12",
      "Sample-size-scaled learning rate transferred from t-SNE-style heuristics: lr = n / 12."
    ),
    learning_rate_scaling_strategy(
      "lr_sqrt_n",
      "sqrt_n",
      "Sample-size-scaled learning rate: lr = sqrt(n)."
    ),
    learning_rate_scaling_strategy(
      "lr_method_default",
      "method_default",
      "Method-specific default learning rate from the current fastEmbedR configuration."
    )
  )
}

adaptive_lr_base_config <- function(ctx) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  cfg$backend <- "cpu"
  cfg
}

adaptive_lr_multiplier <- function(schedule, progress, final_multiplier = 0.05) {
  schedule <- safe_character(schedule, "constant")
  progress <- min(1, max(0, as.numeric(progress)))
  final_multiplier <- min(1, max(0, as.numeric(final_multiplier)))
  if (identical(schedule, "constant")) return(1)
  if (identical(schedule, "linear_decay")) {
    return(final_multiplier + (1 - final_multiplier) * (1 - progress))
  }
  if (identical(schedule, "cosine_decay")) {
    return(final_multiplier + 0.5 * (1 - final_multiplier) * (1 + cos(pi * progress)))
  }
  if (identical(schedule, "step_decay")) {
    if (progress >= 0.75) return(max(final_multiplier, 0.25))
    if (progress >= 0.50) return(max(final_multiplier, 0.50))
    return(1)
  }
  1
}

run_optimizer_with_initial_layout_lr <- function(ctx, knn, init_layout, n_epochs, learning_rate) {
  n_epochs <- max(1L, as.integer(n_epochs))
  learning_rate <- as.numeric(learning_rate)
  if (!is.finite(learning_rate) || learning_rate <= 0) {
    stop("Adaptive learning-rate chunk produced a non-positive learning rate.", call. = FALSE)
  }
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ncol(knn$indices), "cpu")
    cfg$n_epochs <- n_epochs
    cfg$learning_rate <- learning_rate
    layout <- fastEmbedR:::knn_umap_refine_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$optimizer_backend <- "cpu_umap_refine_chunked_lr"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = ncol(knn$indices),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- n_epochs
  cfg$learning_rate <- learning_rate
  if (identical(ctx$method, "tsne")) {
    layout <- fastEmbedR:::knn_tsne_neighbors_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    cfg$affinity_backend <- "cpu_rtsne"
    cfg$optimizer_backend <- "cpu_rtsne_neighbors_chunked_lr"
  } else {
    layout <- fastEmbedR:::knn_objective_embed_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      init_layout,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE,
      FALSE
    )
    cfg$optimizer_backend <- "cpu_knn_objective_chunked_lr"
  }
  layout <- fastEmbedR:::set_embedding_colnames(layout, fastEmbedR:::objective_prefix(ctx$method))
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

adaptive_lr_config <- function(ctx,
                               schedule,
                               total_epochs = NULL,
                               chunk_epochs = 25L,
                               final_multiplier = 0.05) {
  cfg <- adaptive_lr_base_config(ctx)
  schedule <- safe_character(schedule, "constant")
  total_epochs <- if (is.null(total_epochs)) {
    as.integer(cfg$n_epochs)
  } else {
    as.integer(total_epochs)
  }
  total_epochs <- max(1L, total_epochs)
  chunk_epochs <- max(1L, min(as.integer(chunk_epochs), total_epochs))
  cfg$adaptive_lr_enabled <- TRUE
  cfg$adaptive_lr_schedule <- schedule
  cfg$adaptive_lr_total_epochs <- total_epochs
  cfg$adaptive_lr_chunk_epochs <- chunk_epochs
  cfg$adaptive_lr_final_multiplier <- as.numeric(final_multiplier)
  cfg$adaptive_lr_base_learning_rate <- as.numeric(cfg$learning_rate)
  cfg$adaptive_lr_optimizer <- if (identical(schedule, "adam")) "adam" else if (identical(schedule, "adagrad")) "adagrad" else "sgd"
  cfg$adaptive_lr_native <- schedule %in% c("adam", "adagrad")
  cfg$adaptive_lr_chunked <- schedule %in% c("constant", "linear_decay", "cosine_decay", "step_decay")
  cfg$adaptive_lr_inner_decay <- if (identical(ctx$method, "tsne")) {
    "tsne_constant_eta_inside_chunk"
  } else {
    "native_chunk_linear_decay_inside_each_chunk"
  }
  cfg$adaptive_lr_backend <- if (schedule %in% c("adam", "adagrad")) "cpu_native_optimizer" else "cpu_chunked_schedule"
  cfg$adaptive_lr_note <- if (identical(schedule, "adam")) {
    "native Adam path with existing optimizer entry points"
  } else if (identical(schedule, "adagrad")) {
    "native AdaGrad path with per-coordinate accumulated squared gradients"
  } else {
    "chunked learning-rate schedule probe; native per-epoch schedule callback is not claimed"
  }
  cfg
}

run_adaptive_lr_schedule <- function(ctx,
                                     schedule,
                                     total_epochs = NULL,
                                     chunk_epochs = 25L,
                                     final_multiplier = 0.05,
                                     init_strategy = "pca") {
  schedule <- safe_character(schedule, "constant")
  if (schedule %in% c("adam", "adagrad")) {
    layout <- run_optimizer_schedule(ctx, schedule)
    cfg <- attr(layout, "fastEmbedR_config")
    if (is.null(cfg)) cfg <- list()
    base <- adaptive_lr_config(ctx, schedule, total_epochs = total_epochs, chunk_epochs = chunk_epochs, final_multiplier = final_multiplier)
    fields <- list(
      adaptive_lr_enabled = TRUE,
      adaptive_lr_schedule = schedule,
      adaptive_lr_total_epochs = safe_number(cfg$n_epochs, base$adaptive_lr_total_epochs),
      adaptive_lr_chunk_epochs = NA_real_,
      adaptive_lr_chunks_run = NA_real_,
      adaptive_lr_base_learning_rate = safe_number(cfg$optimizer_learning_rate, base$adaptive_lr_base_learning_rate),
      adaptive_lr_final_learning_rate = safe_number(cfg$optimizer_learning_rate, base$adaptive_lr_base_learning_rate),
      adaptive_lr_final_multiplier = 1,
      adaptive_lr_optimizer = schedule,
      adaptive_lr_native = TRUE,
      adaptive_lr_chunked = FALSE,
      adaptive_lr_backend = "cpu_native_optimizer",
      adaptive_lr_inner_decay = paste0("native_", schedule),
      adaptive_lr_trace_epochs = NA_character_,
      adaptive_lr_trace_learning_rate = NA_character_,
      adaptive_lr_trace_multiplier = NA_character_,
      adaptive_lr_note = safe_character(base$adaptive_lr_note)
    )
    attr(layout, "fastEmbedR_config") <- c(cfg, fields)
    return(layout)
  }

  full_knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  cfg <- adaptive_lr_config(
    ctx,
    schedule,
    total_epochs = total_epochs,
    chunk_epochs = chunk_epochs,
    final_multiplier = final_multiplier
  )
  total_epochs <- as.integer(cfg$adaptive_lr_total_epochs)
  chunk_epochs <- as.integer(cfg$adaptive_lr_chunk_epochs)
  base_lr <- as.numeric(cfg$adaptive_lr_base_learning_rate)
  init_used <- init_strategy
  init_time <- system.time({
    current <- tryCatch(
      build_initial_layout(ctx, full_knn, init_strategy, chunk_epochs),
      error = function(e) {
        init_used <<- "random"
        random_initial_layout(ctx$n, ctx$method, ctx$seed)
      }
    )
    current <- coerce_layout(current, ctx$n)
  })[["elapsed"]]

  epochs_completed <- 0L
  chunks_run <- 0L
  optimizer_time <- 0
  lr_trace <- numeric()
  multiplier_trace <- numeric()
  epoch_trace <- numeric()
  while (epochs_completed < total_epochs) {
    remaining <- total_epochs - epochs_completed
    this_chunk <- min(chunk_epochs, remaining)
    midpoint <- (epochs_completed + 0.5 * this_chunk) / max(1, total_epochs)
    multiplier <- adaptive_lr_multiplier(schedule, midpoint, final_multiplier)
    lr <- max(.Machine$double.eps, base_lr * multiplier)
    chunk_time <- system.time({
      current <- run_optimizer_with_initial_layout_lr(ctx, full_knn, current, this_chunk, lr)
      current <- coerce_layout(current, ctx$n)
    })[["elapsed"]]
    optimizer_time <- optimizer_time + as.numeric(chunk_time)
    epochs_completed <- epochs_completed + this_chunk
    chunks_run <- chunks_run + 1L
    lr_trace <- c(lr_trace, lr)
    multiplier_trace <- c(multiplier_trace, multiplier)
    epoch_trace <- c(epoch_trace, epochs_completed)
  }

  out_cfg <- attr(current, "fastEmbedR_config")
  if (is.null(out_cfg)) out_cfg <- list()
  fields <- list(
    adaptive_lr_enabled = TRUE,
    adaptive_lr_schedule = schedule,
    adaptive_lr_total_epochs = safe_number(total_epochs),
    adaptive_lr_chunk_epochs = safe_number(chunk_epochs),
    adaptive_lr_chunks_run = safe_number(chunks_run),
    adaptive_lr_base_learning_rate = safe_number(base_lr),
    adaptive_lr_final_learning_rate = safe_number(tail(lr_trace, 1L)),
    adaptive_lr_final_multiplier = safe_number(tail(multiplier_trace, 1L)),
    adaptive_lr_optimizer = "sgd",
    adaptive_lr_native = FALSE,
    adaptive_lr_chunked = TRUE,
    adaptive_lr_backend = "cpu_chunked_schedule",
    adaptive_lr_inner_decay = safe_character(cfg$adaptive_lr_inner_decay),
    adaptive_lr_init_strategy = init_used,
    adaptive_lr_init_time_sec = as.numeric(init_time),
    adaptive_lr_optimizer_time_sec = safe_number(optimizer_time),
    adaptive_lr_trace_epochs = early_stop_trace(epoch_trace, digits = 0L),
    adaptive_lr_trace_learning_rate = early_stop_trace(lr_trace),
    adaptive_lr_trace_multiplier = early_stop_trace(multiplier_trace),
    adaptive_lr_note = safe_character(cfg$adaptive_lr_note)
  )
  attr(current, "fastEmbedR_config") <- c(out_cfg, fields)
  current
}

adaptive_learning_rate_strategy <- function(schedule,
                                            chunk_epochs = 25L,
                                            total_epochs = NULL,
                                            final_multiplier = 0.05) {
  schedule <- safe_character(schedule, "constant")
  id <- paste0("adaptive_lr_", gsub("[^a-z0-9]+", "_", schedule))
  list(
    id = id,
    family = "optimization_schedule",
    knn_cache_strategy_id = "adaptive_lr_shared_knn",
    adaptive_lr_schedule = schedule,
    adaptive_lr_chunk_epochs = as.integer(chunk_epochs),
    adaptive_lr_total_epochs = if (is.null(total_epochs)) NA_real_ else as.integer(total_epochs),
    adaptive_lr_final_multiplier = as.numeric(final_multiplier),
    description = paste0("Adaptive learning-rate probe using schedule `", schedule, "`."),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) list(available = TRUE, message = NA_character_),
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_tsne_neighbors_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_umap_refine_cpp", envir = ns, inherits = FALSE)
      if (schedule %in% c("adam", "adagrad")) {
        ok <- ok &&
          exists("knn_tsne_neighbors_optimizer_cpp", envir = ns, inherits = FALSE) &&
          exists("knn_objective_embed_optimizer_cpp", envir = ns, inherits = FALSE)
      }
      list(
        available = ok,
        message = "fastEmbedR must be installed with native CPU embedding and optimizer entry points."
      )
    },
    params = function(ctx) {
      cfg <- adaptive_lr_config(
        ctx,
        schedule,
        total_epochs = total_epochs,
        chunk_epochs = chunk_epochs,
        final_multiplier = final_multiplier
      )
      list(
        k = ctx$k,
        adaptive_lr_schedule = schedule,
        total_epochs = cfg$adaptive_lr_total_epochs,
        chunk_epochs = cfg$adaptive_lr_chunk_epochs,
        base_learning_rate = cfg$adaptive_lr_base_learning_rate,
        final_multiplier = cfg$adaptive_lr_final_multiplier,
        optimizer = cfg$adaptive_lr_optimizer,
        native = cfg$adaptive_lr_native,
        chunked = cfg$adaptive_lr_chunked,
        backend = cfg$adaptive_lr_backend,
        note = cfg$adaptive_lr_note
      )
    },
    run = function(ctx) run_adaptive_lr_schedule(
      ctx,
      schedule = schedule,
      total_epochs = total_epochs,
      chunk_epochs = chunk_epochs,
      final_multiplier = final_multiplier
    )
  )
}

adaptive_learning_rate_strategy_grid <- function() {
  lapply(
    c("constant", "linear_decay", "cosine_decay", "step_decay", "adam", "adagrad"),
    adaptive_learning_rate_strategy
  )
}

mini_batch_label <- function(batch_fraction, chunks) {
  paste0(
    "f",
    as.integer(round(100 * as.numeric(batch_fraction))),
    "_c",
    as.integer(chunks)
  )
}

mini_batch_stage_epochs <- function(total_epochs, chunks) {
  chunks <- max(1L, as.integer(chunks))
  total_epochs <- max(chunks, as.integer(total_epochs))
  out <- rep.int(total_epochs %/% chunks, chunks)
  remainder <- total_epochs - sum(out)
  if (remainder > 0L) out[seq_len(remainder)] <- out[seq_len(remainder)] + 1L
  pmax(1L, out)
}

mini_batch_config <- function(ctx,
                              batch_fraction = 0.5,
                              chunks = 4L,
                              weight_power = 1,
                              include_top = 1L,
                              init_strategy = "pca") {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  batch_fraction <- max(0.01, min(1, as.numeric(batch_fraction)))
  chunks <- max(1L, as.integer(chunks))
  total_epochs <- max(chunks, as.integer(ctx$short_epochs))
  stage_epochs <- mini_batch_stage_epochs(total_epochs, chunks)
  cfg$n_epochs <- total_epochs
  cfg$mini_batch_enabled <- TRUE
  cfg$mini_batch_backend <- "cpu_chunked_weighted_edge_batches"
  cfg$mini_batch_mode <- "refreshed_weighted_knn_edge_batches"
  cfg$mini_batch_batch_fraction <- batch_fraction
  cfg$mini_batch_effective_k <- max(1L, ceiling(as.integer(ctx$k) * batch_fraction))
  cfg$mini_batch_chunks <- chunks
  cfg$mini_batch_chunk_epochs <- paste(stage_epochs, collapse = ",")
  cfg$mini_batch_total_epochs <- total_epochs
  cfg$mini_batch_refreshes <- max(0L, chunks - 1L)
  cfg$mini_batch_sampling <- "weighted_without_replacement_per_row"
  cfg$mini_batch_weight_power <- as.numeric(weight_power)
  cfg$mini_batch_include_top <- max(0L, as.integer(include_top))
  cfg$mini_batch_init_strategy <- safe_character(init_strategy, "pca")
  cfg$mini_batch_experimental <- identical(ctx$method, "tsne")
  cfg$mini_batch_note <- if (identical(ctx$method, "tsne")) {
    "experimental sparse-affinity mini-batch probe; early exaggeration is only applied in the first chunk"
  } else {
    "chunked refreshed edge-batch optimization probe; native per-epoch mini-batch kernel is not claimed"
  }
  cfg$optimizer_backend <- "cpu_cpp_mini_batch_probe"
  cfg$backend <- "cpu"
  cfg
}

run_mini_batch_chunk <- function(ctx,
                                 knn,
                                 init_layout,
                                 n_epochs,
                                 seed,
                                 first_chunk = FALSE) {
  n_epochs <- max(1L, as.integer(n_epochs))
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ncol(knn$indices), "cpu")
    cfg$n_epochs <- n_epochs
    layout <- fastEmbedR:::knn_umap_refine_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$optimizer_backend <- "cpu_umap_refine_mini_batch_chunk"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = ncol(knn$indices),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- n_epochs
  if (identical(ctx$method, "tsne")) {
    stop_lying_iter <- if (isTRUE(first_chunk)) {
      min(as.integer(cfg$stop_lying_iter), max(0L, n_epochs - 1L))
    } else {
      0L
    }
    mom_switch_iter <- if (isTRUE(first_chunk)) {
      min(as.integer(cfg$mom_switch_iter), max(1L, n_epochs))
    } else {
      1L
    }
    layout <- fastEmbedR:::knn_tsne_neighbors_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate,
      as.integer(stop_lying_iter),
      as.integer(mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(seed),
      FALSE
    )
    cfg$affinity_backend <- "cpu_rtsne_sparse_minibatch_experimental"
    cfg$optimizer_backend <- "cpu_rtsne_neighbors_mini_batch_chunk"
  } else {
    layout <- fastEmbedR:::knn_objective_embed_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      init_layout,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(seed),
      FALSE,
      FALSE
    )
    cfg$optimizer_backend <- "cpu_knn_objective_mini_batch_chunk"
  }
  layout <- fastEmbedR:::set_embedding_colnames(layout, fastEmbedR:::objective_prefix(ctx$method))
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

run_mini_batch_optimization <- function(ctx,
                                        batch_fraction = 0.5,
                                        chunks = 4L,
                                        weight_power = 1,
                                        include_top = 1L,
                                        init_strategy = "pca",
                                        seed_stride = 7919L) {
  base_knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  cfg <- mini_batch_config(
    ctx,
    batch_fraction = batch_fraction,
    chunks = chunks,
    weight_power = weight_power,
    include_top = include_top,
    init_strategy = init_strategy
  )
  stage_epochs <- as.integer(strsplit(cfg$mini_batch_chunk_epochs, ",", fixed = TRUE)[[1L]])
  init_used <- cfg$mini_batch_init_strategy
  init_time <- system.time({
    current <- tryCatch(
      build_initial_layout(ctx, base_knn, init_used, stage_epochs[1L]),
      error = function(e) {
        init_used <<- "random"
        random_initial_layout(ctx$n, ctx$method, ctx$seed)
      }
    )
    current <- coerce_layout(current, ctx$n)
  })[["elapsed"]]

  graph_time <- 0
  optimizer_time <- 0
  effective_k_trace <- numeric()
  retention_trace <- numeric()
  last <- NULL
  for (chunk in seq_along(stage_epochs)) {
    sampled_time <- system.time({
      last <- graph_weighted_edge_sample_knn(
        base_knn,
        keep_fraction = cfg$mini_batch_batch_fraction,
        weight_power = cfg$mini_batch_weight_power,
        include_top = cfg$mini_batch_include_top,
        target_scale = 1,
        seed = as.integer(ctx$seed + seed_stride * chunk)
      )
      stage_knn <- fastEmbedR:::normalize_supplied_knn(
        last$knn,
        ctx$n,
        max(1L, safe_number(last$graph_effective_k, cfg$mini_batch_effective_k))
      )
    })[["elapsed"]]
    graph_time <- graph_time + as.numeric(sampled_time)
    effective_k_trace <- c(effective_k_trace, safe_number(last$graph_effective_k))
    retention_trace <- c(retention_trace, safe_number(last$graph_edge_retention))

    chunk_time <- system.time({
      current <- run_mini_batch_chunk(
        ctx,
        stage_knn,
        current,
        stage_epochs[chunk],
        seed = as.integer(ctx$seed + seed_stride * chunk + 101L),
        first_chunk = chunk == 1L
      )
      current <- coerce_layout(current, ctx$n)
    })[["elapsed"]]
    optimizer_time <- optimizer_time + as.numeric(chunk_time)
  }

  out_cfg <- attr(current, "fastEmbedR_config")
  if (is.null(out_cfg)) out_cfg <- list()
  fields <- list(
    mini_batch_enabled = TRUE,
    mini_batch_backend = cfg$mini_batch_backend,
    mini_batch_mode = cfg$mini_batch_mode,
    mini_batch_batch_fraction = cfg$mini_batch_batch_fraction,
    mini_batch_effective_k = safe_number(tail(effective_k_trace, 1L), cfg$mini_batch_effective_k),
    mini_batch_chunks = safe_number(length(stage_epochs)),
    mini_batch_chunk_epochs = cfg$mini_batch_chunk_epochs,
    mini_batch_total_epochs = safe_number(sum(stage_epochs)),
    mini_batch_refreshes = safe_number(max(0L, length(stage_epochs) - 1L)),
    mini_batch_sampling = cfg$mini_batch_sampling,
    mini_batch_weight_power = cfg$mini_batch_weight_power,
    mini_batch_include_top = cfg$mini_batch_include_top,
    mini_batch_init_strategy = init_used,
    mini_batch_init_time_sec = as.numeric(init_time),
    mini_batch_graph_time_sec = safe_number(graph_time),
    mini_batch_optimizer_time_sec = safe_number(optimizer_time),
    mini_batch_trace_effective_k = early_stop_trace(effective_k_trace, digits = 0L),
    mini_batch_trace_retention = early_stop_trace(retention_trace),
    mini_batch_experimental = cfg$mini_batch_experimental,
    mini_batch_note = cfg$mini_batch_note,
    graph_approximation = if (!is.null(last$graph_approximation)) last$graph_approximation else "mini_batch_weighted_edges",
    graph_effective_k = safe_number(tail(effective_k_trace, 1L), cfg$mini_batch_effective_k),
    graph_edge_retention = safe_number(tail(retention_trace, 1L), cfg$mini_batch_batch_fraction),
    graph_edge_sampling_method = "mini_batch_refreshed_weighted_without_replacement",
    graph_edge_sampling_fraction = cfg$mini_batch_batch_fraction,
    graph_edge_sampling_weight_power = cfg$mini_batch_weight_power,
    graph_edge_sampling_include_top = cfg$mini_batch_include_top,
    graph_edge_sampling_target_scale = 1,
    graph_edge_sampling_mean_selected_weight = safe_number(last$graph_edge_sampling_mean_selected_weight),
    graph_edge_sampling_mean_candidate_weight = safe_number(last$graph_edge_sampling_mean_candidate_weight),
    graph_edge_sampling_selected_to_candidate_weight_ratio =
      safe_number(last$graph_edge_sampling_selected_to_candidate_weight_ratio)
  )
  attr(current, "fastEmbedR_config") <- c(out_cfg, fields)
  current
}

mini_batch_strategy <- function(batch_fraction = 0.5,
                                chunks = 4L,
                                weight_power = 1,
                                include_top = 1L) {
  batch_fraction <- as.numeric(batch_fraction)
  chunks <- as.integer(chunks)
  id <- paste0("mini_batch_", mini_batch_label(batch_fraction, chunks))
  list(
    id = id,
    family = "optimization_schedule",
    knn_cache_strategy_id = "mini_batch_shared_knn",
    description = paste0(
      "Mini-batch optimizer probe: refresh weighted KNN edge batches with fraction ",
      batch_fraction,
      " over ",
      chunks,
      " chunks. t-SNE rows are explicitly experimental."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_umap_refine_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_tsne_neighbors_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Mini-batch probes require native CPU KNN refinement entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= chunks,
      message = "Mini-batch optimization requires --k >= 4 and enough epochs for at least one epoch per chunk."
    ),
    params = function(ctx) {
      cfg <- mini_batch_config(
        ctx,
        batch_fraction = batch_fraction,
        chunks = chunks,
        weight_power = weight_power,
        include_top = include_top
      )
      list(
        k = ctx$k,
        batch_fraction = cfg$mini_batch_batch_fraction,
        effective_k = cfg$mini_batch_effective_k,
        chunks = cfg$mini_batch_chunks,
        chunk_epochs = cfg$mini_batch_chunk_epochs,
        total_epochs = cfg$mini_batch_total_epochs,
        refreshes = cfg$mini_batch_refreshes,
        sampling = cfg$mini_batch_sampling,
        weight_power = cfg$mini_batch_weight_power,
        include_top = cfg$mini_batch_include_top,
        experimental = cfg$mini_batch_experimental,
        backend = "cpu"
      )
    },
    run = function(ctx) run_mini_batch_optimization(
      ctx,
      batch_fraction = batch_fraction,
      chunks = chunks,
      weight_power = weight_power,
      include_top = include_top
    ),
    mini_batch_enabled = TRUE,
    mini_batch_batch_fraction = batch_fraction,
    mini_batch_chunks = chunks,
    mini_batch_weight_power = as.numeric(weight_power),
    mini_batch_include_top = as.integer(include_top)
  )
}

mini_batch_strategy_grid <- function() {
  list(
    mini_batch_strategy(0.25, 4L),
    mini_batch_strategy(0.50, 4L),
    mini_batch_strategy(0.75, 4L)
  )
}

deterministic_batch_label <- function(row_batch_size) {
  paste0("r", as.integer(row_batch_size))
}

deterministic_batch_config <- function(ctx,
                                       row_batch_size = 2048L,
                                       init_strategy = "spectral") {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  row_batch_size <- max(1L, as.integer(row_batch_size))
  chunks_per_epoch <- max(1L, ceiling(ctx$n / row_batch_size))
  cfg$n_epochs <- max(1L, as.integer(ctx$short_epochs))
  cfg$backend <- "cpu"
  cfg$graph_prep_backend <- if (identical(ctx$method, "tsne")) {
    "cpu_tsne_affinity_csr"
  } else {
    "cpu_sparse_fuzzy_csr"
  }
  cfg$optimizer_backend <- "cpu_cpp_deterministic_row_batches"
  cfg$mini_batch_enabled <- TRUE
  cfg$mini_batch_backend <- "cpu_csr_deterministic_batched_reduction"
  cfg$mini_batch_mode <- "fixed_row_batches_no_atomics"
  cfg$mini_batch_batch_fraction <- min(1, row_batch_size / max(1, ctx$n))
  cfg$mini_batch_effective_k <- safe_number(ctx$graph_effective_k, ctx$k)
  cfg$mini_batch_chunks <- chunks_per_epoch
  cfg$mini_batch_chunk_epochs <- paste0("row_batch_size=", row_batch_size)
  cfg$mini_batch_total_epochs <- cfg$n_epochs
  cfg$mini_batch_refreshes <- 0L
  cfg$mini_batch_sampling <- "none_full_csr_fixed_order"
  cfg$mini_batch_weight_power <- 1
  cfg$mini_batch_include_top <- NA_integer_
  cfg$mini_batch_init_strategy <- safe_character(init_strategy, "spectral")
  cfg$mini_batch_experimental <- identical(ctx$method, "tsne")
  cfg$mini_batch_note <- paste0(
    "Deterministic fixed row-batch CSR optimizer: per-thread deltas are combined ",
    "in thread-id order; no atomic coordinate updates are used."
  )
  cfg$deterministic_batch_row_batch_size <- row_batch_size
  cfg$deterministic_batch_chunks_per_epoch <- chunks_per_epoch
  cfg$deterministic_batch_reduction <- "fixed_thread_order"
  cfg$deterministic_batch_atomic_updates <- FALSE
  cfg$deterministic_batch_reproducible_given_threads <- TRUE
  cfg
}

deterministic_batch_csr_graph <- function(ctx,
                                          target_scale = 1,
                                          local_connectivity = 1,
                                          set_op_mix_ratio = 1,
                                          weight_power = 1,
                                          prune_weight = 0) {
  if (!identical(ctx$method, "tsne")) {
    return(sparse_fuzzy_graph_csr(
      ctx,
      target_scale = target_scale,
      local_connectivity = local_connectivity,
      set_op_mix_ratio = set_op_mix_ratio,
      weight_power = weight_power,
      prune_weight = prune_weight
    ))
  }
  runner <- get0("knn_tsne_affinity_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native t-SNE affinity CSR builder is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = ctx$k,
    objective = "tsne",
    quality = "fast",
    backend = "cpu"
  )
  csr <- runner(knn$indices, knn$distances, cfg$perplexity)
  degrees <- diff(as.integer(csr$offsets))
  nnz <- length(csr$neighbors)
  weights <- as.numeric(csr$probs)
  dense_mb <- (length(knn$indices) * 4 + length(knn$distances) * 8) / 1024^2
  internal_mb <- ((length(csr$offsets) + nnz) * 4 + nnz * 4) / 1024^2
  list(
    graph_csr = list(
      offsets = csr$offsets,
      neighbors = csr$neighbors,
      weights = csr$probs
    ),
    graph_approximation = "tsne_perplexity_affinity_csr",
    graph_storage_format = "csr_tsne_float_affinity_weights",
    graph_sparse_nnz = nnz,
    graph_sparse_internal_memory_mb = internal_mb,
    graph_sparse_r_memory_mb = ((length(csr$offsets) + nnz) * 4 + nnz * 8) / 1024^2,
    graph_dense_knn_memory_mb = dense_mb,
    graph_sparse_internal_memory_ratio = internal_mb / max(dense_mb, .Machine$double.eps),
    graph_sparse_r_memory_ratio = (((length(csr$offsets) + nnz) * 4 + nnz * 8) / 1024^2) /
      max(dense_mb, .Machine$double.eps),
    graph_sparse_prune_weight = 0,
    graph_sparse_mean_weight = safe_mean(weights),
    graph_sparse_min_weight = if (length(weights) == 0L) NA_real_ else min(weights, na.rm = TRUE),
    graph_sparse_max_weight = if (length(weights) == 0L) NA_real_ else max(weights, na.rm = TRUE),
    graph_effective_k = safe_mean(degrees),
    graph_edge_retention = safe_mean(degrees) / max(1, ctx$k),
    graph_recall_at_k = NA_real_,
    graph_mean_degree = safe_mean(degrees),
    graph_min_degree = if (length(degrees) == 0L) NA_real_ else min(degrees),
    graph_max_degree = if (length(degrees) == 0L) NA_real_ else max(degrees),
    graph_isolated_fraction = mean(degrees == 0),
    graph_padding_fraction = 0,
    graph_tsne_affinity_mode = "native_perplexity",
    graph_tsne_affinity_perplexities = as.character(cfg$perplexity),
    graph_tsne_affinity_num_scales = 1,
    graph_tsne_affinity_temperature = 1,
    graph_tsne_affinity_effective_perplexity_mean = cfg$perplexity
  )
}

run_deterministic_batch_optimization <- function(ctx,
                                                 row_batch_size = 2048L,
                                                 init_strategy = "spectral",
                                                 target_scale = 1,
                                                 local_connectivity = 1,
                                                 set_op_mix_ratio = 1,
                                                 weight_power = 1,
                                                 prune_weight = 0) {
  if (!identical(ctx$backend, "cpu")) {
    stop("Deterministic batched optimization is implemented for CPU only in this build.", call. = FALSE)
  }
  if (is.null(ctx$graph_csr)) {
    stop("CSR graph was not prepared.", call. = FALSE)
  }
  runner <- get0("knn_objective_embed_csr_deterministic_batch_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native deterministic batched CSR optimizer is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- deterministic_batch_config(
    ctx,
    row_batch_size = row_batch_size,
    init_strategy = init_strategy
  )
  init_used <- cfg$mini_batch_init_strategy
  init_time <- system.time({
    init <- if (identical(ctx$method, "tsne")) {
      fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    } else {
      tryCatch(
        fastEmbedR:::spectral_knn_init(
          knn$indices,
          knn$distances,
          n_components = 2L,
          spectral_n_iter = as.integer(cfg$spectral_n_iter),
          seed = ctx$seed,
          backend = "cpu"
        ),
        error = function(e) {
          init_used <<- "random"
          fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
        }
      )
    }
  })[["elapsed"]]
  optimizer_time <- system.time({
    layout <- runner(
      ctx$graph_csr$offsets,
      ctx$graph_csr$neighbors,
      ctx$graph_csr$weights,
      ctx$method,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      as.integer(cfg$deterministic_batch_row_batch_size),
      FALSE
    )
  })[["elapsed"]]
  prefix <- switch(
    ctx$method,
    umap = "UMAP",
    tsne = "TSNE",
    pacmap = "PACMAP",
    trimap = "TRIMAP",
    localmap = "LOCALMAP",
    "EMB"
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, prefix)
  fields <- list(
    graph_approximation = ctx$graph_approximation,
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_effective_k = ctx$graph_effective_k,
    graph_edge_retention = ctx$graph_edge_retention,
    init_backend = if (identical(ctx$method, "tsne")) "cpu_random" else attr(init, "backend"),
    init_strategy = init_used,
    mini_batch_enabled = TRUE,
    mini_batch_backend = cfg$mini_batch_backend,
    mini_batch_mode = cfg$mini_batch_mode,
    mini_batch_batch_fraction = cfg$mini_batch_batch_fraction,
    mini_batch_effective_k = safe_number(ctx$graph_effective_k, cfg$mini_batch_effective_k),
    mini_batch_chunks = cfg$mini_batch_chunks,
    mini_batch_chunk_epochs = cfg$mini_batch_chunk_epochs,
    mini_batch_total_epochs = cfg$mini_batch_total_epochs,
    mini_batch_refreshes = cfg$mini_batch_refreshes,
    mini_batch_sampling = cfg$mini_batch_sampling,
    mini_batch_weight_power = cfg$mini_batch_weight_power,
    mini_batch_include_top = cfg$mini_batch_include_top,
    mini_batch_init_strategy = init_used,
    mini_batch_init_time_sec = as.numeric(init_time),
    mini_batch_graph_time_sec = safe_number(ctx$graph_approximation_time_sec),
    mini_batch_optimizer_time_sec = as.numeric(optimizer_time),
    mini_batch_trace_effective_k = as.character(round(safe_number(ctx$graph_effective_k), 3)),
    mini_batch_trace_retention = as.character(round(safe_number(ctx$graph_edge_retention), 3)),
    mini_batch_experimental = cfg$mini_batch_experimental,
    mini_batch_note = cfg$mini_batch_note,
    deterministic_batch_row_batch_size = cfg$deterministic_batch_row_batch_size,
    deterministic_batch_chunks_per_epoch = cfg$deterministic_batch_chunks_per_epoch,
    deterministic_batch_reduction = cfg$deterministic_batch_reduction,
    deterministic_batch_atomic_updates = cfg$deterministic_batch_atomic_updates,
    deterministic_batch_reproducible_given_threads = cfg$deterministic_batch_reproducible_given_threads
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

deterministic_batch_strategy <- function(row_batch_size = 2048L) {
  row_batch_size <- max(1L, as.integer(row_batch_size))
  id <- paste0("deterministic_batch_", deterministic_batch_label(row_batch_size))
  list(
    id = id,
    family = "optimization_schedule",
    knn_cache_strategy_id = "deterministic_batch_shared_knn",
    description = paste0(
      "Deterministic batched CSR optimizer with fixed row batches of ",
      row_batch_size,
      " rows. Per-thread gradient buffers are reduced in fixed order; no atomics."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_objective_embed_csr_deterministic_batch_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_fuzzy_graph_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_tsne_affinity_csr_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Deterministic batch optimization requires native CSR graph and optimizer entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 3L,
      message = "Deterministic batch optimization requires --k >= 3."
    ),
    transform_knn = function(ctx) deterministic_batch_csr_graph(ctx),
    params = function(ctx) {
      cfg <- deterministic_batch_config(ctx, row_batch_size = row_batch_size)
      list(
        k = ctx$k,
        n_epochs = cfg$n_epochs,
        negative_sample_rate = cfg$negative_sample_rate,
        learning_rate = cfg$learning_rate,
        n_threads = cfg$n_threads,
        row_batch_size = cfg$deterministic_batch_row_batch_size,
        chunks_per_epoch = cfg$deterministic_batch_chunks_per_epoch,
        reduction = cfg$deterministic_batch_reduction,
        atomic_updates = cfg$deterministic_batch_atomic_updates,
        reproducible_given_threads = cfg$deterministic_batch_reproducible_given_threads,
        backend = "cpu"
      )
    },
    run = function(ctx) run_deterministic_batch_optimization(
      ctx,
      row_batch_size = row_batch_size
    )
  )
}

deterministic_batch_strategy_grid <- function() {
  list(
    deterministic_batch_strategy(512L),
    deterministic_batch_strategy(2048L),
    deterministic_batch_strategy(8192L)
  )
}

sparse_edge_batch_label <- function(edge_batch_size) {
  paste0("e", as.integer(edge_batch_size))
}

sparse_edge_batch_config <- function(ctx,
                                     edge_batch_size = 32768L,
                                     init_strategy = "spectral") {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  edge_batch_size <- max(1L, as.integer(edge_batch_size))
  estimated_edges <- max(1, safe_number(ctx$graph_sparse_nnz, ctx$n * ctx$k) / 2)
  chunks_per_epoch <- max(1L, ceiling(estimated_edges / edge_batch_size))
  touched_nodes <- min(
    ctx$n,
    edge_batch_size * if (identical(ctx$method, "trimap")) {
      max(3L, as.integer(safe_number(cfg$negative_sample_rate, 5L)) + 2L)
    } else {
      max(2L, as.integer(safe_number(cfg$negative_sample_rate, 5L)) + 2L)
    }
  )
  slot_mb <- ctx$n * 4 / 1024^2
  chunk_delta_mb <- touched_nodes * (4 + 4 + 4) / 1024^2

  cfg$n_epochs <- max(1L, as.integer(ctx$short_epochs))
  cfg$negative_sample_rate <- as.integer(safe_number(cfg$negative_sample_rate, 5L))
  cfg$learning_rate <- safe_number(cfg$learning_rate, if (identical(ctx$method, "tsne")) 50 else 1)
  cfg$backend <- "cpu"
  cfg$graph_prep_backend <- if (identical(ctx$method, "tsne")) {
    "cpu_tsne_affinity_csr"
  } else {
    "cpu_sparse_fuzzy_csr"
  }
  cfg$optimizer_backend <- "cpu_cpp_sparse_edge_batches"
  cfg$sparse_edge_batch_enabled <- TRUE
  cfg$sparse_edge_batch_backend <- "cpu"
  cfg$sparse_edge_batch_mode <- "stream_csr_edges_in_fixed_chunks"
  cfg$sparse_edge_batch_storage <- "csr_streamed_without_edge_list_copy"
  cfg$sparse_edge_batch_edge_batch_size <- edge_batch_size
  cfg$sparse_edge_batch_chunks_per_epoch <- chunks_per_epoch
  cfg$sparse_edge_batch_n_epochs <- cfg$n_epochs
  cfg$sparse_edge_batch_negative_sample_rate <- cfg$negative_sample_rate
  cfg$sparse_edge_batch_learning_rate <- cfg$learning_rate
  cfg$sparse_edge_batch_threads <- 1L
  cfg$sparse_edge_batch_atomic_updates <- FALSE
  cfg$sparse_edge_batch_edge_list_copy <- FALSE
  cfg$sparse_edge_batch_triplet_chunks <- identical(ctx$method, "trimap")
  cfg$sparse_edge_batch_affinity_chunks <- identical(ctx$method, "tsne")
  cfg$sparse_edge_batch_aux_memory_mb <- slot_mb + chunk_delta_mb
  cfg$sparse_edge_batch_init_strategy <- safe_character(init_strategy, "spectral")
  cfg$sparse_edge_batch_status <- "cpu_native_streamed_csr"
  cfg$sparse_edge_batch_note <- paste0(
    "CSR edges are streamed in fixed chunks of at most ",
    edge_batch_size,
    " undirected edges; chunk-local sparse deltas avoid full edge-list and dense per-thread delta copies."
  )
  cfg
}

sparse_edge_batch_normalized_knn <- function(ctx) {
  fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
}

sparse_edge_batch_csr_graph <- function(ctx,
                                        target_scale = 1,
                                        local_connectivity = 1,
                                        set_op_mix_ratio = 1,
                                        weight_power = 1,
                                        prune_weight = 0) {
  knn <- sparse_edge_batch_normalized_knn(ctx)
  if (!identical(ctx$method, "tsne")) {
    builder <- get0("knn_fuzzy_graph_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
    if (!is.function(builder)) {
      stop("Native sparse fuzzy CSR graph builder is not available in this fastEmbedR build.", call. = FALSE)
    }
    csr <- builder(
      knn$indices,
      knn$distances,
      as.numeric(target_scale),
      as.numeric(local_connectivity),
      as.numeric(set_op_mix_ratio),
      as.numeric(weight_power),
      as.numeric(prune_weight)
    )
    mix <- min(1, max(0, as.numeric(set_op_mix_ratio)))
    return(list(
      graph_csr = csr,
      graph_approximation = "sparse_fuzzy_csr_streamed_edge_batches",
      graph_effective_k = safe_number(csr$graph_effective_k),
      graph_edge_retention = safe_number(csr$graph_effective_k) / max(1, ctx$k),
      graph_mean_degree = safe_number(csr$graph_mean_degree),
      graph_min_degree = safe_number(csr$graph_min_degree),
      graph_max_degree = safe_number(csr$graph_max_degree),
      graph_isolated_fraction = safe_number(csr$graph_isolated_fraction),
      graph_padding_fraction = 0,
      graph_storage_format = as.character(csr$graph_storage_format),
      graph_sparse_nnz = safe_number(csr$graph_sparse_nnz),
      graph_sparse_internal_memory_mb = safe_number(csr$graph_sparse_internal_memory_mb),
      graph_sparse_r_memory_mb = safe_number(csr$graph_sparse_r_memory_mb),
      graph_dense_knn_memory_mb = safe_number(csr$graph_dense_knn_memory_mb),
      graph_sparse_internal_memory_ratio = safe_number(csr$graph_sparse_internal_memory_ratio),
      graph_sparse_r_memory_ratio = safe_number(csr$graph_sparse_r_memory_ratio),
      graph_sparse_prune_weight = as.numeric(prune_weight),
      graph_sparse_mean_weight = safe_number(csr$graph_sparse_mean_weight),
      graph_sparse_min_weight = safe_number(csr$graph_sparse_min_weight),
      graph_sparse_max_weight = safe_number(csr$graph_sparse_max_weight),
      graph_recall_at_k = NA_real_,
      graph_mean_distance_error = NA_real_,
      graph_rank_correlation = NA_real_,
      graph_quality_sample_size = NA_real_,
      graph_tsne_affinity_mode = NA_character_,
      graph_tsne_affinity_perplexities = NA_character_,
      graph_tsne_affinity_num_scales = NA_real_,
      umap_graph_set_op_mix_ratio = mix,
      umap_graph_local_connectivity = as.numeric(local_connectivity),
      umap_graph_weight_power = as.numeric(weight_power),
      umap_graph_target_scale = as.numeric(target_scale),
      umap_graph_mean_weight = safe_number(csr$graph_sparse_mean_weight),
      umap_graph_min_weight = safe_number(csr$graph_sparse_min_weight),
      umap_graph_max_weight = safe_number(csr$graph_sparse_max_weight)
    ))
  }

  runner <- get0("knn_tsne_affinity_csr_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native t-SNE affinity CSR builder is not available in this fastEmbedR build.", call. = FALSE)
  }
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = ctx$k,
    objective = "tsne",
    quality = "fast",
    backend = "cpu"
  )
  csr <- runner(knn$indices, knn$distances, cfg$perplexity)
  degrees <- diff(as.integer(csr$offsets))
  nnz <- length(csr$neighbors)
  weights <- as.numeric(csr$probs)
  dense_mb <- (length(knn$indices) * 4 + length(knn$distances) * 8) / 1024^2
  internal_mb <- ((length(csr$offsets) + nnz) * 4 + nnz * 4) / 1024^2
  list(
    graph_csr = list(
      offsets = csr$offsets,
      neighbors = csr$neighbors,
      weights = csr$probs
    ),
    graph_approximation = "tsne_perplexity_affinity_csr_streamed_chunks",
    graph_storage_format = "csr_tsne_float_affinity_weights",
    graph_sparse_nnz = nnz,
    graph_sparse_internal_memory_mb = internal_mb,
    graph_sparse_r_memory_mb = ((length(csr$offsets) + nnz) * 4 + nnz * 8) / 1024^2,
    graph_dense_knn_memory_mb = dense_mb,
    graph_sparse_internal_memory_ratio = internal_mb / max(dense_mb, .Machine$double.eps),
    graph_sparse_r_memory_ratio = (((length(csr$offsets) + nnz) * 4 + nnz * 8) / 1024^2) /
      max(dense_mb, .Machine$double.eps),
    graph_sparse_prune_weight = 0,
    graph_sparse_mean_weight = safe_mean(weights),
    graph_sparse_min_weight = if (length(weights) == 0L) NA_real_ else min(weights, na.rm = TRUE),
    graph_sparse_max_weight = if (length(weights) == 0L) NA_real_ else max(weights, na.rm = TRUE),
    graph_effective_k = safe_mean(degrees),
    graph_edge_retention = safe_mean(degrees) / max(1, ctx$k),
    graph_recall_at_k = NA_real_,
    graph_mean_degree = safe_mean(degrees),
    graph_min_degree = if (length(degrees) == 0L) NA_real_ else min(degrees),
    graph_max_degree = if (length(degrees) == 0L) NA_real_ else max(degrees),
    graph_isolated_fraction = mean(degrees == 0),
    graph_padding_fraction = 0,
    graph_tsne_affinity_mode = "native_perplexity_streamed_chunks",
    graph_tsne_affinity_perplexities = as.character(cfg$perplexity),
    graph_tsne_affinity_num_scales = 1,
    graph_tsne_affinity_temperature = 1,
    graph_tsne_affinity_effective_perplexity_mean = cfg$perplexity
  )
}

run_sparse_edge_batch_optimization <- function(ctx,
                                               edge_batch_size = 32768L,
                                               init_strategy = "spectral") {
  if (!identical(ctx$backend, "cpu")) {
    stop("Sparse edge-batch optimization is implemented for CPU only in this build.", call. = FALSE)
  }
  if (is.null(ctx$graph_csr)) {
    stop("CSR graph was not prepared.", call. = FALSE)
  }
  runner <- get0("knn_objective_embed_csr_sparse_edge_batch_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native sparse edge-batch optimizer is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- sparse_edge_batch_normalized_knn(ctx)
  cfg <- sparse_edge_batch_config(
    ctx,
    edge_batch_size = edge_batch_size,
    init_strategy = init_strategy
  )
  init_used <- cfg$sparse_edge_batch_init_strategy
  init_time <- system.time({
    init <- if (identical(ctx$method, "tsne")) {
      fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    } else {
      tryCatch(
        fastEmbedR:::spectral_knn_init(
          knn$indices,
          knn$distances,
          n_components = 2L,
          spectral_n_iter = as.integer(cfg$spectral_n_iter),
          seed = ctx$seed,
          backend = "cpu"
        ),
        error = function(e) {
          init_used <<- "random"
          fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
        }
      )
    }
  })[["elapsed"]]
  optimizer_time <- system.time({
    layout <- runner(
      ctx$graph_csr$offsets,
      ctx$graph_csr$neighbors,
      ctx$graph_csr$weights,
      ctx$method,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(ctx$seed),
      as.integer(cfg$sparse_edge_batch_edge_batch_size),
      FALSE
    )
  })[["elapsed"]]
  prefix <- switch(
    ctx$method,
    umap = "UMAP",
    tsne = "TSNE",
    pacmap = "PACMAP",
    trimap = "TRIMAP",
    localmap = "LOCALMAP",
    "EMB"
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, prefix)
  fields <- list(
    graph_approximation = ctx$graph_approximation,
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_effective_k = ctx$graph_effective_k,
    graph_edge_retention = ctx$graph_edge_retention,
    init_backend = if (identical(ctx$method, "tsne")) "cpu_random" else attr(init, "backend"),
    init_strategy = init_used,
    sparse_edge_batch_enabled = TRUE,
    sparse_edge_batch_backend = cfg$sparse_edge_batch_backend,
    sparse_edge_batch_mode = cfg$sparse_edge_batch_mode,
    sparse_edge_batch_storage = cfg$sparse_edge_batch_storage,
    sparse_edge_batch_edge_batch_size = cfg$sparse_edge_batch_edge_batch_size,
    sparse_edge_batch_chunks_per_epoch = cfg$sparse_edge_batch_chunks_per_epoch,
    sparse_edge_batch_n_epochs = cfg$sparse_edge_batch_n_epochs,
    sparse_edge_batch_negative_sample_rate = cfg$sparse_edge_batch_negative_sample_rate,
    sparse_edge_batch_learning_rate = cfg$sparse_edge_batch_learning_rate,
    sparse_edge_batch_threads = cfg$sparse_edge_batch_threads,
    sparse_edge_batch_atomic_updates = cfg$sparse_edge_batch_atomic_updates,
    sparse_edge_batch_edge_list_copy = cfg$sparse_edge_batch_edge_list_copy,
    sparse_edge_batch_triplet_chunks = cfg$sparse_edge_batch_triplet_chunks,
    sparse_edge_batch_affinity_chunks = cfg$sparse_edge_batch_affinity_chunks,
    sparse_edge_batch_aux_memory_mb = cfg$sparse_edge_batch_aux_memory_mb,
    sparse_edge_batch_graph_time_sec = safe_number(ctx$graph_approximation_time_sec),
    sparse_edge_batch_init_time_sec = as.numeric(init_time),
    sparse_edge_batch_optimizer_time_sec = as.numeric(optimizer_time),
    sparse_edge_batch_status = cfg$sparse_edge_batch_status,
    sparse_edge_batch_note = cfg$sparse_edge_batch_note
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

sparse_edge_batch_strategy <- function(edge_batch_size = 32768L) {
  edge_batch_size <- max(1L, as.integer(edge_batch_size))
  id <- paste0("sparse_edge_batch_", sparse_edge_batch_label(edge_batch_size))
  list(
    id = id,
    family = "sparse_edge_batching",
    knn_cache_strategy_id = "sparse_edge_batch_shared_knn",
    description = paste0(
      "Low-RAM streamed CSR edge optimizer with chunks of ",
      edge_batch_size,
      " edges. UMAP/PaCMAP/LocalMAP use edge chunks, TriMap expands triplets inside chunks, ",
      "and t-SNE streams sparse affinity chunks."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_objective_embed_csr_sparse_edge_batch_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_fuzzy_graph_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_tsne_affinity_csr_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Sparse edge batching requires native CSR graph and streamed optimizer entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 3L,
      message = "Sparse edge batching requires --k >= 3."
    ),
    transform_knn = function(ctx) sparse_edge_batch_csr_graph(ctx),
    params = function(ctx) {
      cfg <- sparse_edge_batch_config(ctx, edge_batch_size = edge_batch_size)
      list(
        k = ctx$k,
        n_epochs = cfg$sparse_edge_batch_n_epochs,
        negative_sample_rate = cfg$sparse_edge_batch_negative_sample_rate,
        learning_rate = cfg$sparse_edge_batch_learning_rate,
        edge_batch_size = cfg$sparse_edge_batch_edge_batch_size,
        chunks_per_epoch = cfg$sparse_edge_batch_chunks_per_epoch,
        storage = cfg$sparse_edge_batch_storage,
        edge_list_copy = cfg$sparse_edge_batch_edge_list_copy,
        atomic_updates = cfg$sparse_edge_batch_atomic_updates,
        aux_memory_mb = cfg$sparse_edge_batch_aux_memory_mb,
        backend = "cpu"
      )
    },
    run = function(ctx) run_sparse_edge_batch_optimization(
      ctx,
      edge_batch_size = edge_batch_size
    )
  )
}

sparse_edge_batch_strategy_grid <- function() {
  list(
    sparse_edge_batch_strategy(4096L),
    sparse_edge_batch_strategy(32768L),
    sparse_edge_batch_strategy(131072L)
  )
}

vectorized_edge_config <- function(ctx,
                                   batch_size = 4096L,
                                   target_scale = 1,
                                   local_connectivity = 1,
                                   set_op_mix_ratio = 1,
                                   weight_power = 1,
                                   prune_weight = 0) {
  cfg <- sparse_fuzzy_graph_config(ctx)
  cfg$n_epochs <- max(1L, as.integer(ctx$short_epochs))
  cfg$negative_sample_rate <- as.integer(safe_number(cfg$negative_sample_rate, 5L))
  cfg$learning_rate <- safe_number(cfg$learning_rate, 1)
  cfg$n_threads <- max(1L, as.integer(safe_number(cfg$n_threads, parallel::detectCores(logical = TRUE))))
  cfg$spectral_n_iter <- max(1L, as.integer(safe_number(cfg$spectral_n_iter, 10L)))
  cfg$batch_size <- max(64L, as.integer(batch_size))
  cfg$backend <- "cpu"
  cfg$graph_prep_backend <- "cpu_sparse_fuzzy_csr"
  cfg$optimizer_backend <- "cpu_cpp_vectorized_edges"
  cfg$vectorized_edge_enabled <- TRUE
  cfg$vectorized_edge_backend <- "cpu"
  cfg$vectorized_edge_storage <- "contiguous_edge_list_from_csr"
  cfg$vectorized_edge_batch_size <- cfg$batch_size
  cfg$vectorized_edge_n_epochs <- cfg$n_epochs
  cfg$vectorized_edge_negative_sample_rate <- cfg$negative_sample_rate
  cfg$vectorized_edge_threads <- cfg$n_threads
  cfg$vectorized_edge_simd <- "contiguous_batch_simd_friendly_scatter_loop"
  cfg$vectorized_edge_gpu_native <- FALSE
  cfg$vectorized_edge_target_scale <- as.numeric(target_scale)
  cfg$vectorized_edge_local_connectivity <- as.numeric(local_connectivity)
  cfg$vectorized_edge_set_op_mix_ratio <- as.numeric(set_op_mix_ratio)
  cfg$vectorized_edge_weight_power <- as.numeric(weight_power)
  cfg$vectorized_edge_prune_weight <- as.numeric(prune_weight)
  cfg$vectorized_edge_status <- "cpu_native"
  cfg$vectorized_edge_note <- "CSR graph is compacted once into contiguous from/to/weight vectors before optimization."
  cfg
}

run_vectorized_edge_optimization <- function(ctx,
                                             batch_size = 4096L,
                                             target_scale = 1,
                                             local_connectivity = 1,
                                             set_op_mix_ratio = 1,
                                             weight_power = 1,
                                             prune_weight = 0) {
  if (!identical(ctx$backend, "cpu")) {
    stop("Native vectorized edge optimization is implemented for CPU only in this build.", call. = FALSE)
  }
  if (is.null(ctx$graph_csr)) {
    stop("Sparse fuzzy CSR graph was not prepared.", call. = FALSE)
  }
  runner <- get0("knn_objective_embed_csr_vectorized_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native vectorized edge optimizer is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- vectorized_edge_config(
    ctx,
    batch_size = batch_size,
    target_scale = target_scale,
    local_connectivity = local_connectivity,
    set_op_mix_ratio = set_op_mix_ratio,
    weight_power = weight_power,
    prune_weight = prune_weight
  )
  init_used <- "spectral"
  init_time <- system.time({
    init <- tryCatch(
      fastEmbedR:::spectral_knn_init(
        knn$indices,
        knn$distances,
        n_components = 2L,
        spectral_n_iter = as.integer(cfg$spectral_n_iter),
        seed = ctx$seed,
        backend = "cpu"
      ),
      error = function(e) {
        init_used <<- "random"
        fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
      }
    )
  })[["elapsed"]]
  optimizer_time <- system.time({
    layout <- runner(
      ctx$graph_csr$offsets,
      ctx$graph_csr$neighbors,
      ctx$graph_csr$weights,
      ctx$method,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      as.integer(cfg$batch_size),
      FALSE
    )
  })[["elapsed"]]
  prefix <- switch(
    ctx$method,
    umap = "UMAP",
    pacmap = "PACMAP",
    localmap = "LOCALMAP",
    "EMB"
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, prefix)
  fields <- list(
    graph_approximation = "sparse_fuzzy_csr_vectorized_edges",
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_effective_k = ctx$graph_effective_k,
    graph_edge_retention = ctx$graph_edge_retention,
    init_backend = attr(init, "backend"),
    init_strategy = init_used,
    vectorized_edge_enabled = TRUE,
    vectorized_edge_backend = cfg$vectorized_edge_backend,
    vectorized_edge_storage = cfg$vectorized_edge_storage,
    vectorized_edge_batch_size = cfg$vectorized_edge_batch_size,
    vectorized_edge_n_edges = safe_number(ctx$graph_sparse_nnz, 0) / 2,
    vectorized_edge_n_epochs = cfg$vectorized_edge_n_epochs,
    vectorized_edge_negative_sample_rate = cfg$vectorized_edge_negative_sample_rate,
    vectorized_edge_threads = cfg$vectorized_edge_threads,
    vectorized_edge_learning_rate = cfg$learning_rate,
    vectorized_edge_simd = cfg$vectorized_edge_simd,
    vectorized_edge_gpu_native = cfg$vectorized_edge_gpu_native,
    vectorized_edge_graph_time_sec = safe_number(ctx$graph_approximation_time_sec),
    vectorized_edge_init_time_sec = as.numeric(init_time),
    vectorized_edge_optimizer_time_sec = as.numeric(optimizer_time),
    vectorized_edge_status = cfg$vectorized_edge_status,
    vectorized_edge_note = cfg$vectorized_edge_note
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

vectorized_edge_strategy <- function(batch_size = 4096L,
                                     target_scale = 1,
                                     local_connectivity = 1,
                                     set_op_mix_ratio = 1,
                                     weight_power = 1,
                                     prune_weight = 0) {
  batch_size <- max(64L, as.integer(batch_size))
  id <- paste0("vectorized_edge_b", batch_size)
  list(
    id = id,
    family = "vectorized_edge_optimization",
    knn_cache_strategy_id = "vectorized_edge_shared_knn",
    description = paste0(
      "CPU-native SIMD-friendly edge-list optimizer for UMAP, PaCMAP, and LocalMAP. ",
      "CSR graph rows are compacted into contiguous edge batches of size ", batch_size, "."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "localmap") &&
        backend %in% c("cpu", "cuda", "metal", "rocm")
    },
    knn_backend = function(ctx) "cpu",
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_fuzzy_graph_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_csr_vectorized_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Vectorized edge optimization requires the native CSR graph builder and vectorized edge optimizer."
      )
    },
    context_available = function(ctx) {
      if (!identical(ctx$backend, "cpu")) {
        return(list(
          available = FALSE,
          message = paste0(
            "Native vectorized edge optimization is not implemented for backend `",
            ctx$backend,
            "` yet; no CPU fallback is reported as GPU."
          )
        ))
      }
      list(
        available = ctx$k >= 3L,
        message = "Vectorized edge optimization requires --k >= 3."
      )
    },
    transform_knn = function(ctx) sparse_fuzzy_graph_csr(
      ctx,
      target_scale = target_scale,
      local_connectivity = local_connectivity,
      set_op_mix_ratio = set_op_mix_ratio,
      weight_power = weight_power,
      prune_weight = prune_weight
    ),
    params = function(ctx) {
      if (!identical(ctx$backend, "cpu")) {
        return(list(
          k = ctx$k,
          requested_backend = ctx$backend,
          backend = "not_supported",
          gpu_native = FALSE,
          unsupported_reason = "native_vectorized_edge_optimizer_not_implemented_for_requested_gpu_backend"
        ))
      }
      cfg <- vectorized_edge_config(
        ctx,
        batch_size = batch_size,
        target_scale = target_scale,
        local_connectivity = local_connectivity,
        set_op_mix_ratio = set_op_mix_ratio,
        weight_power = weight_power,
        prune_weight = prune_weight
      )
      list(
        k = ctx$k,
        batch_size = cfg$vectorized_edge_batch_size,
        n_epochs = cfg$vectorized_edge_n_epochs,
        negative_sample_rate = cfg$vectorized_edge_negative_sample_rate,
        learning_rate = cfg$learning_rate,
        n_threads = cfg$vectorized_edge_threads,
        graph_storage = cfg$vectorized_edge_storage,
        backend = cfg$vectorized_edge_backend,
        gpu_native = cfg$vectorized_edge_gpu_native
      )
    },
    run = function(ctx) run_vectorized_edge_optimization(
      ctx,
      batch_size = batch_size,
      target_scale = target_scale,
      local_connectivity = local_connectivity,
      set_op_mix_ratio = set_op_mix_ratio,
      weight_power = weight_power,
      prune_weight = prune_weight
    ),
    vectorized_edge_enabled = TRUE,
    vectorized_edge_batch_size = batch_size,
    vectorized_edge_backend = "cpu"
  )
}

vectorized_edge_strategy_grid <- function() {
  list(
    vectorized_edge_strategy(1024L),
    vectorized_edge_strategy(4096L),
    vectorized_edge_strategy(16384L)
  )
}

atomic_sgd_label <- function(learning_rate_scale) {
  gsub("[^0-9]+", "p", sprintf("%.4f", as.numeric(learning_rate_scale)))
}

atomic_sgd_config <- function(ctx,
                              learning_rate_scale = 0.005,
                              coordinate_clip = 25,
                              target_scale = 1,
                              local_connectivity = 1,
                              set_op_mix_ratio = 1,
                              weight_power = 1,
                              prune_weight = 0) {
  cfg <- sparse_fuzzy_graph_config(ctx)
  cfg$n_epochs <- max(1L, as.integer(ctx$short_epochs))
  cfg$negative_sample_rate <- as.integer(safe_number(cfg$negative_sample_rate, 5L))
  cfg$learning_rate <- safe_number(cfg$learning_rate, 1)
  cfg$n_threads <- max(1L, as.integer(safe_number(cfg$n_threads, parallel::detectCores(logical = TRUE))))
  cfg$spectral_n_iter <- max(1L, as.integer(safe_number(cfg$spectral_n_iter, 10L)))
  cfg$backend <- "cpu"
  cfg$graph_prep_backend <- "cpu_sparse_fuzzy_csr"
  cfg$optimizer_backend <- "cpu_cpp_atomic_sgd"
  cfg$atomic_sgd_enabled <- TRUE
  cfg$atomic_sgd_backend <- "cpu"
  cfg$atomic_sgd_update_mode <- "direct_atomic_coordinate_updates"
  cfg$atomic_sgd_storage <- "contiguous_edge_list_from_csr"
  cfg$atomic_sgd_n_epochs <- cfg$n_epochs
  cfg$atomic_sgd_negative_sample_rate <- cfg$negative_sample_rate
  cfg$atomic_sgd_threads <- cfg$n_threads
  cfg$atomic_sgd_learning_rate <- cfg$learning_rate
  cfg$atomic_sgd_learning_rate_scale <- safe_number(learning_rate_scale, 0.005)
  cfg$atomic_sgd_coordinate_clip <- safe_number(coordinate_clip, 25)
  cfg$atomic_sgd_openmp <- FALSE
  cfg$atomic_sgd_gpu_native <- FALSE
  cfg$atomic_sgd_nondeterministic <- cfg$n_threads > 1L
  cfg$atomic_sgd_target_scale <- as.numeric(target_scale)
  cfg$atomic_sgd_local_connectivity <- as.numeric(local_connectivity)
  cfg$atomic_sgd_set_op_mix_ratio <- as.numeric(set_op_mix_ratio)
  cfg$atomic_sgd_weight_power <- as.numeric(weight_power)
  cfg$atomic_sgd_prune_weight <- as.numeric(prune_weight)
  cfg$atomic_sgd_status <- "cpu_native_std_thread_atomics"
  cfg$atomic_sgd_note <- "Experimental direct SGD with atomic float coordinate updates; multi-threaded results are intentionally non-deterministic."
  cfg
}

run_atomic_sgd_optimization <- function(ctx,
                                        learning_rate_scale = 0.005,
                                        coordinate_clip = 25,
                                        target_scale = 1,
                                        local_connectivity = 1,
                                        set_op_mix_ratio = 1,
                                        weight_power = 1,
                                        prune_weight = 0) {
  if (!identical(ctx$backend, "cpu")) {
    stop("Native atomic SGD optimization is implemented for CPU only in this build.", call. = FALSE)
  }
  if (is.null(ctx$graph_csr)) {
    stop("Sparse fuzzy CSR graph was not prepared.", call. = FALSE)
  }
  runner <- get0("knn_objective_embed_csr_atomic_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
  if (!is.function(runner)) {
    stop("Native atomic SGD optimizer is not available in this fastEmbedR build.", call. = FALSE)
  }
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- atomic_sgd_config(
    ctx,
    learning_rate_scale = learning_rate_scale,
    coordinate_clip = coordinate_clip,
    target_scale = target_scale,
    local_connectivity = local_connectivity,
    set_op_mix_ratio = set_op_mix_ratio,
    weight_power = weight_power,
    prune_weight = prune_weight
  )
  init_used <- "spectral"
  init_time <- system.time({
    init <- tryCatch(
      fastEmbedR:::spectral_knn_init(
        knn$indices,
        knn$distances,
        n_components = 2L,
        spectral_n_iter = as.integer(cfg$spectral_n_iter),
        seed = ctx$seed,
        backend = "cpu"
      ),
      error = function(e) {
        init_used <<- "random"
        fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
      }
    )
  })[["elapsed"]]
  optimizer_time <- system.time({
    layout <- runner(
      ctx$graph_csr$offsets,
      ctx$graph_csr$neighbors,
      ctx$graph_csr$weights,
      ctx$method,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      cfg$atomic_sgd_learning_rate_scale,
      cfg$atomic_sgd_coordinate_clip,
      FALSE
    )
  })[["elapsed"]]
  prefix <- switch(
    ctx$method,
    umap = "UMAP",
    pacmap = "PACMAP",
    trimap = "TRIMAP",
    localmap = "LOCALMAP",
    "EMB"
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, prefix)
  fields <- list(
    graph_approximation = "sparse_fuzzy_csr_atomic_sgd",
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_effective_k = ctx$graph_effective_k,
    graph_edge_retention = ctx$graph_edge_retention,
    init_backend = attr(init, "backend"),
    init_strategy = init_used,
    atomic_sgd_enabled = TRUE,
    atomic_sgd_backend = cfg$atomic_sgd_backend,
    atomic_sgd_update_mode = cfg$atomic_sgd_update_mode,
    atomic_sgd_storage = cfg$atomic_sgd_storage,
    atomic_sgd_n_edges = safe_number(ctx$graph_sparse_nnz, 0) / 2,
    atomic_sgd_n_epochs = cfg$atomic_sgd_n_epochs,
    atomic_sgd_negative_sample_rate = cfg$atomic_sgd_negative_sample_rate,
    atomic_sgd_threads = cfg$atomic_sgd_threads,
    atomic_sgd_learning_rate = cfg$atomic_sgd_learning_rate,
    atomic_sgd_learning_rate_scale = cfg$atomic_sgd_learning_rate_scale,
    atomic_sgd_coordinate_clip = cfg$atomic_sgd_coordinate_clip,
    atomic_sgd_openmp = cfg$atomic_sgd_openmp,
    atomic_sgd_gpu_native = cfg$atomic_sgd_gpu_native,
    atomic_sgd_nondeterministic = cfg$atomic_sgd_nondeterministic,
    atomic_sgd_graph_time_sec = safe_number(ctx$graph_approximation_time_sec),
    atomic_sgd_init_time_sec = as.numeric(init_time),
    atomic_sgd_optimizer_time_sec = as.numeric(optimizer_time),
    atomic_sgd_status = cfg$atomic_sgd_status,
    atomic_sgd_note = cfg$atomic_sgd_note
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

atomic_sgd_strategy <- function(learning_rate_scale = 0.005,
                                coordinate_clip = 25,
                                target_scale = 1,
                                local_connectivity = 1,
                                set_op_mix_ratio = 1,
                                weight_power = 1,
                                prune_weight = 0) {
  learning_rate_scale <- safe_number(learning_rate_scale, 0.005)
  id <- paste0("atomic_sgd_s", atomic_sgd_label(learning_rate_scale))
  list(
    id = id,
    family = "atomic_sgd_optimization",
    knn_cache_strategy_id = "atomic_sgd_shared_knn",
    description = paste0(
      "Experimental parallel SGD with atomic coordinate updates; learning-rate scale ",
      learning_rate_scale,
      ". CPU uses std::thread atomics on this build; CUDA/Metal atomics are reported unsupported."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "trimap", "localmap") &&
        backend %in% c("cpu", "cuda", "metal", "rocm")
    },
    knn_backend = function(ctx) "cpu",
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("knn_fuzzy_graph_csr_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_csr_atomic_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Atomic SGD requires the native CSR graph builder and atomic optimizer."
      )
    },
    context_available = function(ctx) {
      if (!identical(ctx$backend, "cpu")) {
        return(list(
          available = FALSE,
          message = paste0(
            "Native atomic SGD is not implemented for backend `",
            ctx$backend,
            "` yet; no CPU fallback is reported as GPU."
          )
        ))
      }
      list(
        available = ctx$k >= 3L,
        message = "Atomic SGD requires --k >= 3."
      )
    },
    transform_knn = function(ctx) sparse_fuzzy_graph_csr(
      ctx,
      target_scale = target_scale,
      local_connectivity = local_connectivity,
      set_op_mix_ratio = set_op_mix_ratio,
      weight_power = weight_power,
      prune_weight = prune_weight
    ),
    params = function(ctx) {
      if (!identical(ctx$backend, "cpu")) {
        return(list(
          k = ctx$k,
          requested_backend = ctx$backend,
          backend = "not_supported",
          gpu_native = FALSE,
          nondeterministic = TRUE,
          unsupported_reason = "native_atomic_sgd_not_implemented_for_requested_gpu_backend"
        ))
      }
      cfg <- atomic_sgd_config(
        ctx,
        learning_rate_scale = learning_rate_scale,
        coordinate_clip = coordinate_clip,
        target_scale = target_scale,
        local_connectivity = local_connectivity,
        set_op_mix_ratio = set_op_mix_ratio,
        weight_power = weight_power,
        prune_weight = prune_weight
      )
      list(
        k = ctx$k,
        n_epochs = cfg$atomic_sgd_n_epochs,
        negative_sample_rate = cfg$atomic_sgd_negative_sample_rate,
        learning_rate = cfg$atomic_sgd_learning_rate,
        learning_rate_scale = cfg$atomic_sgd_learning_rate_scale,
        coordinate_clip = cfg$atomic_sgd_coordinate_clip,
        n_threads = cfg$atomic_sgd_threads,
        update_mode = cfg$atomic_sgd_update_mode,
        backend = cfg$atomic_sgd_backend,
        openmp = cfg$atomic_sgd_openmp,
        gpu_native = cfg$atomic_sgd_gpu_native,
        nondeterministic = cfg$atomic_sgd_nondeterministic
      )
    },
    run = function(ctx) run_atomic_sgd_optimization(
      ctx,
      learning_rate_scale = learning_rate_scale,
      coordinate_clip = coordinate_clip,
      target_scale = target_scale,
      local_connectivity = local_connectivity,
      set_op_mix_ratio = set_op_mix_ratio,
      weight_power = weight_power,
      prune_weight = prune_weight
    ),
    atomic_sgd_enabled = TRUE,
    atomic_sgd_learning_rate_scale = learning_rate_scale,
    atomic_sgd_coordinate_clip = coordinate_clip,
    atomic_sgd_backend = "cpu"
  )
}

atomic_sgd_strategy_grid <- function() {
  list(
    atomic_sgd_strategy(0.0025),
    atomic_sgd_strategy(0.0050),
    atomic_sgd_strategy(0.0100)
  )
}

umap_negative_sampling_config <- function(ctx, negative_sample_rate) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
    cfg$quality <- "auto"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  if (identical(ctx$method, "tsne")) {
    cfg$learning_rate <- 10
    cfg$tsne_mode <- "sampled_repulsion_experimental"
    cfg$tsne_sampled_repulsion_learning_rate_source <- "fixed_10_for_stability"
  }
  cfg$umap_negative_sample_rate <- as.integer(max(0L, negative_sample_rate))
  cfg$umap_negative_sample_rate_default <- as.integer(cfg$negative_sample_rate)
  cfg$negative_sample_rate <- cfg$umap_negative_sample_rate
  cfg$umap_transfer_mode <- if (identical(ctx$method, "umap")) {
    "umap_native_negative_sampling_rate"
  } else if (identical(ctx$method, "tsne")) {
    "tsne_sampled_repulsion_experimental"
  } else {
    paste0(ctx$method, "_umap_negative_sampling_transfer")
  }
  cfg$optimizer_backend <- "cpu_cpp_negative_sampling"
  cfg$backend <- "cpu"
  cfg
}

run_umap_negative_sampling <- function(ctx, negative_sample_rate) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- umap_negative_sampling_config(ctx, negative_sample_rate)
  if (identical(ctx$method, "umap")) {
    layout <- fastEmbedR:::fast_knn_umap_cpp(
      knn$indices,
      knn$distances,
      2L,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$spectral_n_iter),
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$init_backend <- "cpu"
    cfg$graph_prep_backend <- "cpu"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  if (identical(ctx$method, "tsne")) {
    runner <- get(
      "knn_objective_embed_optimizer_cpp",
      envir = asNamespace("fastEmbedR"),
      inherits = FALSE
    )
    init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
    layout <- runner(
      knn$indices,
      knn$distances,
      "tsne",
      init,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      "sgd",
      0,
      0,
      as.integer(cfg$n_epochs + 1L),
      0.9,
      0.999,
      1e-8,
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "TSNE")
    cfg$affinity_backend <- "cpu_umap_fuzzy_knn"
    cfg$optimizer_backend <- "cpu_cpp_sampled_repulsion"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  runner <- get(
    "knn_objective_embed_cpp",
    envir = asNamespace("fastEmbedR"),
    inherits = FALSE
  )
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = as.integer(cfg$spectral_n_iter),
    seed = as.integer(ctx$seed),
    backend = "cpu"
  )
  layout <- runner(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    FALSE,
    FALSE
  )
  layout <- fastEmbedR:::set_embedding_colnames(layout, fastEmbedR:::objective_prefix(ctx$method))
  cfg$init_backend <- "cpu"
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

umap_negative_sampling_strategy <- function(negative_sample_rate) {
  rate <- as.integer(negative_sample_rate)
  list(
    id = paste0("umap_neg_sample_", rate),
    family = "optimization_schedule",
    description = paste0(
      "UMAP-style negative sampling rate probe with negative_sample_rate = ",
      rate,
      ". Transferred to PaCMAP far-pair repulsion, TriMap outlier triplets, LocalMAP repulsive terms, ",
      "and t-SNE as an explicit sampled-repulsion experiment."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("fast_knn_umap_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_optimizer_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "fastEmbedR must be installed with native CPU embedding entry points."
      )
    },
    params = function(ctx) {
      cfg <- umap_negative_sampling_config(ctx, rate)
      list(
        k = ctx$k,
        negative_sample_rate = cfg$negative_sample_rate,
        method_default_negative_sample_rate = cfg$umap_negative_sample_rate_default,
        transfer_mode = cfg$umap_transfer_mode,
        n_epochs = cfg$n_epochs,
        learning_rate = cfg$learning_rate,
        tsne_sampled_repulsion_learning_rate_source = if (!is.null(cfg$tsne_sampled_repulsion_learning_rate_source)) {
          cfg$tsne_sampled_repulsion_learning_rate_source
        } else {
          NA_character_
        },
        backend = "cpu"
      )
    },
    run = function(ctx) run_umap_negative_sampling(ctx, rate),
    umap_negative_sample_rate = rate
  )
}

umap_negative_sampling_strategy_grid <- function() {
  list(
    umap_negative_sampling_strategy(2),
    umap_negative_sampling_strategy(5),
    umap_negative_sampling_strategy(10),
    umap_negative_sampling_strategy(20)
  )
}

run_pacmap_far_repulsion_transfer <- function(ctx, far_repulsion_rate) {
  layout <- run_umap_negative_sampling(ctx, far_repulsion_rate)
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  cfg$pacmap_transfer_mode <- "far_pair_repulsion_as_sampled_negative_edges"
  cfg$pacmap_auxiliary_pair_family <- "far_pairs_proxy"
  cfg$pacmap_far_repulsion_rate <- as.integer(far_repulsion_rate)
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

pacmap_far_repulsion_strategy <- function(far_repulsion_rate) {
  rate <- as.integer(far_repulsion_rate)
  list(
    id = paste0("pacmap_far_repulsion_", rate),
    family = "optimization_schedule",
    description = paste0(
      "PaCMAP far-pair repulsion transfer using sampled negative edges with rate = ",
      rate,
      ". This is a proxy for PaCMAP's explicit far pairs, not a native far-pair objective for every method."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("fast_knn_umap_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_optimizer_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "PaCMAP far-repulsion transfer requires native CPU embedding entry points."
      )
    },
    params = function(ctx) {
      cfg <- umap_negative_sampling_config(ctx, rate)
      list(
        k = ctx$k,
        pacmap_transfer_mode = "far_pair_repulsion_as_sampled_negative_edges",
        pacmap_auxiliary_pair_family = "far_pairs_proxy",
        far_repulsion_rate = rate,
        negative_sample_rate = cfg$negative_sample_rate,
        method_default_negative_sample_rate = cfg$umap_negative_sample_rate_default,
        n_epochs = cfg$n_epochs,
        learning_rate = cfg$learning_rate,
        backend = "cpu"
      )
    },
    run = function(ctx) run_pacmap_far_repulsion_transfer(ctx, rate),
    umap_negative_sample_rate = rate,
    pacmap_transfer_mode = "far_pair_repulsion_as_sampled_negative_edges",
    pacmap_auxiliary_pair_family = "far_pairs_proxy",
    pacmap_far_repulsion_rate = rate
  )
}

pacmap_far_repulsion_strategy_grid <- function() {
  list(
    pacmap_far_repulsion_strategy(5),
    pacmap_far_repulsion_strategy(10),
    pacmap_far_repulsion_strategy(20)
  )
}

pacmap_phase_config <- function(ctx,
                                mid_fraction,
                                mid_distance_scale,
                                warmup_fraction,
                                schedule = "custom",
                                epoch_multiplier = 1) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  schedule <- safe_character(schedule, "custom")
  epoch_multiplier <- max(0.1, as.numeric(epoch_multiplier))
  total_epochs <- max(2L, as.integer(ceiling(as.integer(ctx$short_epochs) * epoch_multiplier)))
  warmup_epochs <- max(1L, min(total_epochs - 1L, as.integer(ceiling(total_epochs * as.numeric(warmup_fraction)))))
  cfg$n_epochs <- total_epochs
  cfg$pacmap_transfer_mode <- "mid_near_warmup_then_near_refine"
  cfg$pacmap_auxiliary_pair_family <- "mid_near_plus_far_repulsion_schedule_proxy"
  cfg$pacmap_mid_near_requested_fraction <- as.numeric(mid_fraction)
  cfg$pacmap_mid_near_distance_scale <- as.numeric(mid_distance_scale)
  cfg$pacmap_phase_schedule <- schedule
  cfg$pacmap_phase_total_epochs <- total_epochs
  cfg$pacmap_phase_epoch_multiplier <- epoch_multiplier
  cfg$pacmap_phase_warmup_epochs <- warmup_epochs
  cfg$pacmap_phase_refine_epochs <- total_epochs - warmup_epochs
  cfg$pacmap_phase_warmup_fraction <- as.numeric(warmup_fraction)
  cfg$pacmap_phase_transfer_detail <- switch(
    ctx$method,
    umap = "staged_local_global_optimization_proxy",
    tsne = "staged_midrange_warm_start_proxy_for_perplexity_exaggeration",
    pacmap = "native_family_mid_near_warmup_then_refine_proxy",
    trimap = "staged_triplet_hardness_proxy",
    localmap = "staged_local_global_balance_proxy",
    "staged_midrange_warmup_proxy"
  )
  cfg$optimizer_backend <- "cpu_cpp_pacmap_phase_transfer"
  cfg$backend <- "cpu"
  cfg
}

run_pacmap_phase_schedule <- function(ctx,
                                      mid_fraction,
                                      mid_distance_scale,
                                      warmup_fraction,
                                      schedule = "custom",
                                      epoch_multiplier = 1) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  transformed <- graph_pacmap_mid_near_knn(
    knn,
    mid_fraction = mid_fraction,
    mid_distance_scale = mid_distance_scale,
    seed = ctx$seed
  )
  mid_knn <- fastEmbedR:::coerce_knn_input(transformed$knn)
  cfg <- pacmap_phase_config(
    ctx,
    mid_fraction,
    mid_distance_scale,
    warmup_fraction,
    schedule = schedule,
    epoch_multiplier = epoch_multiplier
  )
  if (identical(ctx$method, "umap")) {
    warm_layout <- fastEmbedR:::fast_knn_umap_core(
      mid_knn,
      backend = "cpu",
      seed = ctx$seed,
      n_epochs = as.integer(cfg$pacmap_phase_warmup_epochs)
    )
  } else {
    warm_layout <- fastEmbedR:::knn_embed_core(
      mid_knn,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu",
      seed = ctx$seed,
      n_epochs = as.integer(cfg$pacmap_phase_warmup_epochs)
    )
  }
  layout <- fastEmbedR:::refine_embedding_from_knn(
    method = ctx$method,
    indices = knn$indices,
    distances = knn$distances,
    init_layout = warm_layout,
    n_epochs = as.integer(cfg$pacmap_phase_refine_epochs),
    refinement = "pacmap_phase_transfer",
    seed = as.integer(ctx$seed + 3037L),
    backend = "cpu",
    verbose = FALSE
  )
  cfg$pacmap_mid_near_pairs_per_point <- transformed$pacmap_mid_near_pairs_per_point
  cfg$pacmap_mid_near_fraction <- transformed$pacmap_mid_near_fraction
  cfg$pacmap_mid_near_fallback_fraction <- transformed$pacmap_mid_near_fallback_fraction
  cfg$pacmap_mid_near_rank_mean <- transformed$pacmap_mid_near_rank_mean
  cfg$refinement_backend <- "cpu"
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

pacmap_phase_strategy <- function(mid_fraction = 0.33,
                                  mid_distance_scale = 1.50,
                                  warmup_fraction = 0.35,
                                  schedule = "custom",
                                  epoch_multiplier = 1) {
  label <- paste0(
    pacmap_mid_near_label(mid_fraction, mid_distance_scale),
    "_w",
    as.integer(round(100 * as.numeric(warmup_fraction)))
  )
  schedule <- safe_character(schedule, "custom")
  id <- if (identical(schedule, "custom")) {
    paste0("pacmap_phase_", label)
  } else {
    paste0("pacmap_phase_", schedule)
  }
  list(
    id = id,
    family = "optimization_schedule",
    description = paste0(
      "PaCMAP-like ", schedule,
      " staged schedule: warm up on second-order mid-near anchors for ",
      round(100 * warmup_fraction),
      "% of the optimization budget, then refine on the original KNN graph."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("fast_knn_umap_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_tsne_neighbors_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "PaCMAP phase transfer requires native CPU embedding entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 10L && ctx$short_epochs >= 4L,
      message = "PaCMAP phase transfer requires --k >= 10 and --short-epochs >= 4."
    ),
    params = function(ctx) {
      cfg <- pacmap_phase_config(
        ctx,
        mid_fraction,
        mid_distance_scale,
        warmup_fraction,
        schedule = schedule,
        epoch_multiplier = epoch_multiplier
      )
      list(
        k = ctx$k,
        schedule = cfg$pacmap_phase_schedule,
        pacmap_transfer_mode = cfg$pacmap_transfer_mode,
        pacmap_auxiliary_pair_family = cfg$pacmap_auxiliary_pair_family,
        mid_fraction = mid_fraction,
        mid_distance_scale = mid_distance_scale,
        epoch_multiplier = cfg$pacmap_phase_epoch_multiplier,
        total_epochs = cfg$n_epochs,
        warmup_epochs = cfg$pacmap_phase_warmup_epochs,
        refine_epochs = cfg$pacmap_phase_refine_epochs,
        transfer_detail = cfg$pacmap_phase_transfer_detail,
        backend = "cpu"
      )
    },
    run = function(ctx) run_pacmap_phase_schedule(
      ctx,
      mid_fraction,
      mid_distance_scale,
      warmup_fraction,
      schedule = schedule,
      epoch_multiplier = epoch_multiplier
    ),
    pacmap_transfer_mode = "mid_near_warmup_then_near_refine",
    pacmap_auxiliary_pair_family = "mid_near_plus_far_repulsion_schedule_proxy",
    pacmap_mid_near_requested_fraction = as.numeric(mid_fraction),
    pacmap_mid_near_distance_scale = as.numeric(mid_distance_scale),
    pacmap_phase_schedule = schedule,
    pacmap_phase_epoch_multiplier = as.numeric(epoch_multiplier),
    pacmap_phase_warmup_fraction = as.numeric(warmup_fraction)
  )
}

pacmap_phase_strategy_grid <- function() {
  list(
    pacmap_phase_strategy(0.25, 1.35, 0.25, schedule = "short", epoch_multiplier = 0.50),
    pacmap_phase_strategy(0.33, 1.50, 0.35, schedule = "default", epoch_multiplier = 1.00),
    pacmap_phase_strategy(0.40, 1.75, 0.40, schedule = "long", epoch_multiplier = 2.00)
  )
}

pair_resampling_transfer_mode <- function(method, mode) {
  if (identical(mode, "pacmap_pairs")) {
    return(switch(
      method,
      umap = "umap_near_mid_far_edge_resampling_proxy",
      pacmap = "pacmap_near_mid_far_pair_resampling",
      trimap = "trimap_triplet_candidate_resampling_proxy",
      localmap = "localmap_near_mid_far_neighbour_resampling_proxy",
      paste0(method, "_near_mid_far_pair_resampling_proxy")
    ))
  }
  switch(
    method,
    umap = "umap_weighted_edge_resampling",
    pacmap = "pacmap_weighted_pair_resampling_proxy",
    trimap = "trimap_triplet_candidate_resampling_proxy",
    localmap = "localmap_weighted_neighbour_resampling",
    paste0(method, "_weighted_pair_resampling_proxy")
  )
}

pair_resampling_stage_epochs <- function(total_epochs, stage_count) {
  stage_count <- max(2L, as.integer(stage_count))
  total_epochs <- max(stage_count, as.integer(total_epochs))
  out <- rep.int(total_epochs %/% stage_count, stage_count)
  remainder <- total_epochs - sum(out)
  if (remainder > 0L) out[seq_len(remainder)] <- out[seq_len(remainder)] + 1L
  pmax(1L, out)
}

pair_resampling_config <- function(ctx,
                                   mode = "weighted_edges",
                                   keep_fraction = 0.75,
                                   refreshes = 1L,
                                   weight_power = 1,
                                   include_top = 1L,
                                   near_ratio = 0.50,
                                   mid_ratio = 0.30,
                                   far_ratio = 0.20,
                                   mid_distance_scale = 1.40,
                                   far_distance_scale = 3.00,
                                   seed_stride = 7919L) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, backend = "cpu")
    cfg$objective <- "umap"
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu"
    )
  }
  mode <- safe_character(mode, "weighted_edges")
  refreshes <- max(1L, as.integer(refreshes))
  stage_count <- refreshes + 1L
  total_epochs <- max(stage_count, as.integer(ctx$short_epochs))
  stage_epochs <- pair_resampling_stage_epochs(total_epochs, stage_count)
  ratios <- pmax(0, as.numeric(c(near_ratio, mid_ratio, far_ratio)))
  if (!is.finite(sum(ratios)) || sum(ratios) <= 0) ratios <- c(0.50, 0.30, 0.20)
  ratios <- ratios / sum(ratios)

  cfg$n_epochs <- total_epochs
  cfg$pair_resampling_mode <- mode
  cfg$pair_resampling_pair_family <- if (identical(mode, "pacmap_pairs")) {
    "near_mid_far_pairs"
  } else {
    "weighted_knn_edges"
  }
  cfg$pair_resampling_transfer_mode <- pair_resampling_transfer_mode(ctx$method, mode)
  cfg$pair_resampling_refreshes <- refreshes
  cfg$pair_resampling_stage_count <- stage_count
  cfg$pair_resampling_stage_epochs <- paste(stage_epochs, collapse = ",")
  cfg$pair_resampling_warmup_epochs <- stage_epochs[1L]
  cfg$pair_resampling_refine_epochs <- sum(stage_epochs[-1L])
  cfg$pair_resampling_keep_fraction <- as.numeric(keep_fraction)
  cfg$pair_resampling_weight_power <- as.numeric(weight_power)
  cfg$pair_resampling_include_top <- as.integer(include_top)
  cfg$pair_resampling_seed_stride <- as.integer(seed_stride)
  cfg$pair_resampling_final_graph <- "resampled"
  cfg$pacmap_near_ratio <- ratios[1L]
  cfg$pacmap_mid_ratio <- ratios[2L]
  cfg$pacmap_far_ratio <- ratios[3L]
  cfg$pacmap_mid_near_distance_scale <- as.numeric(mid_distance_scale)
  cfg$pacmap_far_distance_scale <- as.numeric(far_distance_scale)
  cfg$optimizer_backend <- "cpu_cpp_pair_resampling_transfer"
  cfg$backend <- "cpu"
  cfg
}

pair_resampling_stage_knn <- function(ctx, base_knn, cfg, stage) {
  stage <- as.integer(stage)
  seed <- as.integer(ctx$seed + stage * cfg$pair_resampling_seed_stride)
  if (identical(cfg$pair_resampling_mode, "pacmap_pairs")) {
    return(graph_pacmap_pair_separation_knn(
      base_knn,
      near_ratio = cfg$pacmap_near_ratio,
      mid_ratio = cfg$pacmap_mid_ratio,
      far_ratio = cfg$pacmap_far_ratio,
      mid_distance_scale = cfg$pacmap_mid_near_distance_scale,
      far_distance_scale = cfg$pacmap_far_distance_scale,
      seed = seed
    ))
  }
  graph_weighted_edge_sample_knn(
    base_knn,
    keep_fraction = cfg$pair_resampling_keep_fraction,
    weight_power = cfg$pair_resampling_weight_power,
    include_top = cfg$pair_resampling_include_top,
    target_scale = 1,
    seed = seed
  )
}

run_pair_resampling <- function(ctx,
                                mode = "weighted_edges",
                                keep_fraction = 0.75,
                                refreshes = 1L,
                                weight_power = 1,
                                include_top = 1L,
                                near_ratio = 0.50,
                                mid_ratio = 0.30,
                                far_ratio = 0.20,
                                mid_distance_scale = 1.40,
                                far_distance_scale = 3.00,
                                seed_stride = 7919L) {
  base_knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- pair_resampling_config(
    ctx,
    mode = mode,
    keep_fraction = keep_fraction,
    refreshes = refreshes,
    weight_power = weight_power,
    include_top = include_top,
    near_ratio = near_ratio,
    mid_ratio = mid_ratio,
    far_ratio = far_ratio,
    mid_distance_scale = mid_distance_scale,
    far_distance_scale = far_distance_scale,
    seed_stride = seed_stride
  )
  stage_epochs <- as.integer(strsplit(cfg$pair_resampling_stage_epochs, ",", fixed = TRUE)[[1L]])
  first <- pair_resampling_stage_knn(ctx, base_knn, cfg, stage = 0L)
  stage_knn <- fastEmbedR:::coerce_knn_input(first$knn)
  if (identical(ctx$method, "umap")) {
    layout <- fastEmbedR:::fast_knn_umap_core(
      stage_knn,
      backend = "cpu",
      seed = ctx$seed,
      n_epochs = stage_epochs[1L]
    )
  } else {
    layout <- fastEmbedR:::knn_embed_core(
      stage_knn,
      objective = ctx$method,
      quality = method_quality(ctx$method, "fast"),
      backend = "cpu",
      seed = ctx$seed,
      n_epochs = stage_epochs[1L]
    )
  }

  last <- first
  if (length(stage_epochs) > 1L) {
    for (stage in seq.int(2L, length(stage_epochs))) {
      last <- pair_resampling_stage_knn(ctx, base_knn, cfg, stage = stage - 1L)
      stage_knn <- fastEmbedR:::coerce_knn_input(last$knn)
      layout <- fastEmbedR:::refine_embedding_from_knn(
        method = ctx$method,
        indices = stage_knn$indices,
        distances = stage_knn$distances,
        init_layout = layout,
        n_epochs = stage_epochs[stage],
        refinement = "pair_resampling",
        seed = as.integer(ctx$seed + cfg$pair_resampling_seed_stride * stage + 101L),
        backend = "cpu",
        verbose = FALSE
      )
    }
  }

  cfg$refinement_backend <- "cpu"
  cfg$graph_approximation <- if (!is.null(last$graph_approximation)) last$graph_approximation else "pair_resampling"
  cfg$graph_edge_retention <- safe_number(last$graph_edge_retention)
  cfg$graph_effective_k <- safe_number(last$graph_effective_k)
  cfg$graph_edge_sampling_method <- if (!is.null(last$graph_edge_sampling_method)) {
    as.character(last$graph_edge_sampling_method)
  } else {
    cfg$pair_resampling_pair_family
  }
  cfg$graph_edge_sampling_fraction <- safe_number(last$graph_edge_sampling_fraction, cfg$pair_resampling_keep_fraction)
  cfg$graph_edge_sampling_weight_power <- safe_number(last$graph_edge_sampling_weight_power, cfg$pair_resampling_weight_power)
  cfg$graph_edge_sampling_include_top <- safe_number(last$graph_edge_sampling_include_top, cfg$pair_resampling_include_top)
  cfg$graph_edge_sampling_target_scale <- safe_number(last$graph_edge_sampling_target_scale, 1)
  cfg$graph_edge_sampling_mean_selected_weight <- safe_number(last$graph_edge_sampling_mean_selected_weight)
  cfg$graph_edge_sampling_mean_candidate_weight <- safe_number(last$graph_edge_sampling_mean_candidate_weight)
  cfg$graph_edge_sampling_selected_to_candidate_weight_ratio <- safe_number(last$graph_edge_sampling_selected_to_candidate_weight_ratio)
  cfg$pacmap_transfer_mode <- "pair_resampling_during_optimization"
  cfg$pacmap_auxiliary_pair_family <- cfg$pair_resampling_pair_family
  for (name in c(
    "pacmap_near_pairs_per_point", "pacmap_mid_pairs_per_point",
    "pacmap_far_pairs_per_point", "pacmap_mid_near_pairs_per_point",
    "pacmap_mid_near_fraction", "pacmap_far_pair_fraction",
    "pacmap_mid_near_fallback_fraction", "pacmap_far_fallback_fraction",
    "pacmap_mid_near_rank_mean"
  )) {
    if (!is.null(last[[name]])) cfg[[name]] <- last[[name]]
  }
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

pair_resampling_label <- function(mode, keep_fraction, refreshes) {
  paste0(
    if (identical(mode, "pacmap_pairs")) "pacmap_pairs" else "weighted",
    "_k",
    as.integer(round(100 * as.numeric(keep_fraction))),
    "_r",
    as.integer(refreshes)
  )
}

pair_resampling_strategy <- function(mode = "weighted_edges",
                                     keep_fraction = 0.75,
                                     refreshes = 1L,
                                     weight_power = 1,
                                     include_top = 1L,
                                     near_ratio = 0.50,
                                     mid_ratio = 0.30,
                                     far_ratio = 0.20) {
  mode <- safe_character(mode, "weighted_edges")
  keep_fraction <- as.numeric(keep_fraction)
  refreshes <- as.integer(refreshes)
  label <- pair_resampling_label(mode, keep_fraction, refreshes)
  list(
    id = paste0("pair_resample_", label),
    family = "optimization_schedule",
    description = paste0(
      "Pair resampling during optimization: optimize on a sampled pair set, refresh it ",
      refreshes,
      if (refreshes == 1L) " time" else " times",
      ", then continue from the current layout."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "trimap", "localmap") &&
        identical(backend, "cpu")
    },
    availability = function() {
      ns <- asNamespace("fastEmbedR")
      ok <- exists("fast_knn_umap_cpp", envir = ns, inherits = FALSE) &&
        exists("knn_objective_embed_cpp", envir = ns, inherits = FALSE)
      list(
        available = ok,
        message = "Pair resampling requires native CPU embedding entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= (if (identical(mode, "pacmap_pairs")) 12L else 4L) &&
        ctx$short_epochs >= refreshes + 2L,
      message = "Pair resampling requires enough KNN slots and at least one epoch per optimization stage."
    ),
    params = function(ctx) {
      cfg <- pair_resampling_config(
        ctx,
        mode = mode,
        keep_fraction = keep_fraction,
        refreshes = refreshes,
        weight_power = weight_power,
        include_top = include_top,
        near_ratio = near_ratio,
        mid_ratio = mid_ratio,
        far_ratio = far_ratio
      )
      list(
        k = ctx$k,
        mode = cfg$pair_resampling_mode,
        pair_family = cfg$pair_resampling_pair_family,
        transfer_mode = cfg$pair_resampling_transfer_mode,
        refreshes = cfg$pair_resampling_refreshes,
        stage_count = cfg$pair_resampling_stage_count,
        stage_epochs = cfg$pair_resampling_stage_epochs,
        keep_fraction = cfg$pair_resampling_keep_fraction,
        weight_power = cfg$pair_resampling_weight_power,
        include_top = cfg$pair_resampling_include_top,
        near_ratio = cfg$pacmap_near_ratio,
        mid_ratio = cfg$pacmap_mid_ratio,
        far_ratio = cfg$pacmap_far_ratio,
        backend = "cpu"
      )
    },
    run = function(ctx) run_pair_resampling(
      ctx,
      mode = mode,
      keep_fraction = keep_fraction,
      refreshes = refreshes,
      weight_power = weight_power,
      include_top = include_top,
      near_ratio = near_ratio,
      mid_ratio = mid_ratio,
      far_ratio = far_ratio
    ),
    pair_resampling_mode = mode,
    pair_resampling_keep_fraction = keep_fraction,
    pair_resampling_refreshes = refreshes,
    pair_resampling_weight_power = weight_power,
    pair_resampling_include_top = include_top,
    pair_resampling_pair_family = if (identical(mode, "pacmap_pairs")) "near_mid_far_pairs" else "weighted_knn_edges",
    pacmap_near_ratio = near_ratio,
    pacmap_mid_ratio = mid_ratio,
    pacmap_far_ratio = far_ratio
  )
}

pair_resampling_strategy_grid <- function() {
  list(
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.75, refreshes = 1L),
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.75, refreshes = 2L),
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.50, refreshes = 2L),
    pair_resampling_strategy("pacmap_pairs", keep_fraction = 1.00, refreshes = 2L, near_ratio = 0.50, mid_ratio = 0.30, far_ratio = 0.20)
  )
}

triplet_aux_label <- function(weight, samples_per_edge) {
  paste0(
    "w",
    gsub("\\.", "p", format(as.numeric(weight), trim = TRUE, scientific = FALSE)),
    "_s",
    as.integer(samples_per_edge)
  )
}

triplet_aux_config <- function(ctx, weight, samples_per_edge) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(ctx$k),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- max(10L, min(as.integer(cfg$n_epochs), as.integer(ctx$short_epochs)))
  cfg$triplet_aux_enabled <- TRUE
  cfg$triplet_aux_weight <- as.numeric(weight)
  cfg$triplet_aux_samples_per_edge <- as.integer(samples_per_edge)
  cfg$triplet_aux_transfer_mode <- switch(
    ctx$method,
    umap = "umap_auxiliary_trimap_triplet_regularization",
    tsne = "tsne_sparse_affinity_with_auxiliary_trimap_triplet_regularization",
    pacmap = "pacmap_near_mid_far_with_auxiliary_trimap_triplet_regularization",
    localmap = "localmap_auxiliary_trimap_triplet_regularization",
    "auxiliary_trimap_triplet_regularization"
  )
  cfg$triplet_aux_native_backend <- "cpu_cpp"
  cfg$triplet_aux_optimizer_backend <- "cpu_cpp_triplet_aux"
  cfg$triplet_aux_n_epochs <- as.integer(cfg$n_epochs)
  cfg$triplet_aux_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$optimizer_backend <- "cpu_cpp_triplet_aux"
  cfg$graph_prep_backend <- "cpu"
  if (identical(ctx$method, "tsne")) {
    cfg$tsne_mode <- "sparse_negative_sampling_triplet_aux"
    cfg$affinity_backend <- "cpu_knn_sparse"
  }
  cfg
}

run_triplet_aux <- function(ctx, weight, samples_per_edge) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- triplet_aux_config(ctx, weight, samples_per_edge)
  init <- if (identical(ctx$method, "tsne")) {
    fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  } else {
    fastEmbedR:::spectral_knn_init(
      knn$indices,
      knn$distances,
      n_components = 2L,
      spectral_n_iter = cfg$spectral_n_iter,
      seed = ctx$seed,
      backend = "cpu"
    )
  }
  cfg$init_backend <- attr(init, "backend")
  optimizer <- if (identical(ctx$method, "tsne")) "momentum" else "sgd"
  momentum <- if (identical(optimizer, "momentum")) 0.5 else 0.0
  final_momentum <- if (identical(optimizer, "momentum")) 0.8 else 0.0
  momentum_switch <- if (identical(optimizer, "momentum")) {
    max(1L, min(250L, as.integer(floor(cfg$n_epochs / 2))))
  } else {
    cfg$n_epochs + 1L
  }
  layout <- fastEmbedR:::knn_objective_embed_triplet_aux_cpp(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    optimizer,
    momentum,
    final_momentum,
    as.integer(momentum_switch),
    0.9,
    0.999,
    1e-8,
    as.numeric(weight),
    as.integer(samples_per_edge),
    FALSE
  )
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- optimizer
  cfg$optimizer_momentum <- momentum
  cfg$optimizer_final_momentum <- final_momentum
  cfg$optimizer_switch_iter <- momentum_switch
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

triplet_aux_strategy <- function(weight, samples_per_edge = 1L) {
  weight <- as.numeric(weight)
  samples_per_edge <- as.integer(samples_per_edge)
  label <- triplet_aux_label(weight, samples_per_edge)
  list(
    id = paste0("triplet_aux_", label),
    family = "triplet_constraints",
    description = paste0(
      "TriMap-style auxiliary triplet regularization: for each near edge (i, j), ",
      "sample non-neighbour k and add a loss encouraging i to stay closer to j than k."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "localmap") && identical(backend, "cpu")
    },
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    availability = function() {
      ok <- exists("knn_objective_embed_triplet_aux_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = ok,
        message = "Triplet auxiliary loss requires the native CPU C++ triplet-auxiliary entry point."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= 10L,
      message = "Triplet auxiliary loss requires --k >= 4 and --short-epochs >= 10."
    ),
    params = function(ctx) {
      cfg <- triplet_aux_config(ctx, weight, samples_per_edge)
      list(
        k = ctx$k,
        triplet_aux_weight = cfg$triplet_aux_weight,
        triplet_aux_samples_per_edge = cfg$triplet_aux_samples_per_edge,
        triplet_aux_transfer_mode = cfg$triplet_aux_transfer_mode,
        triplet_aux_native_backend = cfg$triplet_aux_native_backend,
        n_epochs = cfg$triplet_aux_n_epochs,
        negative_sample_rate = cfg$triplet_aux_negative_sample_rate,
        backend = "cpu"
      )
    },
    run = function(ctx) run_triplet_aux(ctx, weight, samples_per_edge),
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = weight,
    triplet_aux_samples_per_edge = samples_per_edge,
    triplet_aux_native_backend = "cpu_cpp"
  )
}

triplet_aux_strategy_grid <- function() {
  list(
    triplet_aux_strategy(0.05, 1L),
    triplet_aux_strategy(0.10, 1L),
    triplet_aux_strategy(0.20, 2L)
  )
}

structured_triplet_label <- function(weight, n_inliers, n_outliers, n_random) {
  paste0(
    "w",
    gsub("\\.", "p", format(as.numeric(weight), trim = TRUE, scientific = FALSE)),
    "_i",
    as.integer(n_inliers),
    "_o",
    as.integer(n_outliers),
    "_r",
    as.integer(n_random)
  )
}

structured_triplet_config <- function(ctx, weight, n_inliers, n_outliers, n_random) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(ctx$k),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- max(10L, min(as.integer(cfg$n_epochs), as.integer(ctx$short_epochs)))
  cfg$triplet_aux_enabled <- TRUE
  cfg$triplet_aux_weight <- as.numeric(weight)
  cfg$triplet_aux_samples_per_edge <- 0L
  cfg$triplet_aux_transfer_mode <- switch(
    ctx$method,
    umap = "umap_positive_negative_edges_with_structured_trimap_triplets",
    pacmap = "pacmap_near_mid_far_pairs_with_structured_trimap_triplets",
    localmap = "localmap_local_nonlocal_contrast_with_structured_trimap_triplets",
    "structured_trimap_triplet_transfer"
  )
  cfg$triplet_aux_native_backend <- "cpu_cpp"
  cfg$triplet_aux_optimizer_backend <- "cpu_cpp_structured_triplets"
  cfg$triplet_aux_n_epochs <- as.integer(cfg$n_epochs)
  cfg$triplet_aux_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$triplet_structured_enabled <- TRUE
  cfg$triplet_structured_weight <- as.numeric(weight)
  cfg$triplet_structured_n_inliers <- as.integer(n_inliers)
  cfg$triplet_structured_n_outliers <- as.integer(n_outliers)
  cfg$triplet_structured_n_random <- as.integer(n_random)
  cfg$triplet_structured_total <- as.integer(n_inliers + n_outliers + n_random)
  cfg$triplet_structured_mode <- "inlier_outlier_random_triplets"
  cfg$optimizer_backend <- "cpu_cpp_structured_triplets"
  cfg$graph_prep_backend <- "cpu"
  cfg
}

run_structured_triplets <- function(ctx, weight, n_inliers, n_outliers, n_random) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- structured_triplet_config(ctx, weight, n_inliers, n_outliers, n_random)
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = cfg$spectral_n_iter,
    seed = ctx$seed,
    backend = "cpu"
  )
  cfg$init_backend <- attr(init, "backend")
  layout <- fastEmbedR:::knn_objective_embed_structured_triplets_cpp(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    "sgd",
    0,
    0,
    as.integer(cfg$n_epochs + 1L),
    0.9,
    0.999,
    1e-8,
    as.numeric(weight),
    as.integer(n_inliers),
    as.integer(n_outliers),
    as.integer(n_random),
    FALSE
  )
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- "sgd"
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

structured_triplet_strategy <- function(weight, n_inliers, n_outliers, n_random) {
  weight <- as.numeric(weight)
  n_inliers <- as.integer(n_inliers)
  n_outliers <- as.integer(n_outliers)
  n_random <- as.integer(n_random)
  label <- structured_triplet_label(weight, n_inliers, n_outliers, n_random)
  list(
    id = paste0("structured_triplets_", label),
    family = "triplet_constraints",
    description = paste0(
      "TriMap-style structured triplet transfer with n_inliers=", n_inliers,
      ", n_outliers=", n_outliers,
      ", n_random=", n_random,
      ". The native optimizer samples direct-neighbour positives and non-neighbour negatives ",
      "with separate deterministic streams for each triplet family."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "localmap") && identical(backend, "cpu")
    },
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    availability = function() {
      ok <- exists("knn_objective_embed_structured_triplets_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = ok,
        message = "Structured triplet transfer requires the native CPU C++ structured-triplet entry point."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= 10L,
      message = "Structured triplet transfer requires --k >= 4 and --short-epochs >= 10."
    ),
    params = function(ctx) {
      cfg <- structured_triplet_config(ctx, weight, n_inliers, n_outliers, n_random)
      list(
        k = ctx$k,
        triplet_structured_weight = cfg$triplet_structured_weight,
        n_inliers = cfg$triplet_structured_n_inliers,
        n_outliers = cfg$triplet_structured_n_outliers,
        n_random = cfg$triplet_structured_n_random,
        triplet_structured_mode = cfg$triplet_structured_mode,
        triplet_aux_transfer_mode = cfg$triplet_aux_transfer_mode,
        triplet_aux_native_backend = cfg$triplet_aux_native_backend,
        n_epochs = cfg$triplet_aux_n_epochs,
        negative_sample_rate = cfg$triplet_aux_negative_sample_rate,
        backend = "cpu"
      )
    },
    run = function(ctx) run_structured_triplets(ctx, weight, n_inliers, n_outliers, n_random),
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = weight,
    triplet_aux_samples_per_edge = 0L,
    triplet_aux_native_backend = "cpu_cpp",
    triplet_structured_enabled = TRUE,
    triplet_structured_weight = weight,
    triplet_structured_n_inliers = n_inliers,
    triplet_structured_n_outliers = n_outliers,
    triplet_structured_n_random = n_random,
    triplet_structured_total = n_inliers + n_outliers + n_random,
    triplet_structured_mode = "inlier_outlier_random_triplets"
  )
}

structured_triplet_strategy_grid <- function() {
  list(
    structured_triplet_strategy(0.05, 2L, 1L, 0L),
    structured_triplet_strategy(0.05, 2L, 1L, 1L),
    structured_triplet_strategy(0.10, 3L, 2L, 1L)
  )
}

global_random_triplet_label <- function(weight, samples_per_point, trimap_extra_negatives) {
  paste0(
    "w",
    gsub("\\.", "p", format(as.numeric(weight), trim = TRUE, scientific = FALSE)),
    "_r",
    as.integer(samples_per_point),
    "_t",
    as.integer(trimap_extra_negatives)
  )
}

global_random_triplet_config <- function(ctx,
                                         weight,
                                         samples_per_point,
                                         trimap_extra_negatives) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(ctx$k),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- max(10L, min(as.integer(cfg$n_epochs), as.integer(ctx$short_epochs)))
  samples_per_point <- as.integer(samples_per_point)
  trimap_extra_negatives <- as.integer(trimap_extra_negatives)
  if (identical(ctx$method, "trimap")) {
    cfg$negative_sample_rate <- as.integer(max(1L, cfg$negative_sample_rate + trimap_extra_negatives))
  }
  cfg$global_random_triplet_enabled <- TRUE
  cfg$global_random_triplet_weight <- as.numeric(weight)
  cfg$global_random_triplets_per_point <- samples_per_point
  cfg$global_random_trimap_extra_negatives <- if (identical(ctx$method, "trimap")) {
    trimap_extra_negatives
  } else {
    0L
  }
  cfg$global_random_negative_source <- "deterministic_random_non_neighbour"
  cfg$global_random_effective_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$global_random_transfer_mode <- switch(
    ctx$method,
    trimap = "trimap_extra_global_random_outlier_triplets",
    tsne = "tsne_sparse_affinity_with_global_random_triplets",
    pacmap = "pacmap_near_mid_far_with_global_random_triplets",
    localmap = "localmap_global_random_nonlocal_triplets",
    umap = "umap_auxiliary_global_random_triplets",
    "global_random_triplet_transfer"
  )
  cfg$triplet_aux_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_aux_weight <- as.numeric(weight)
  cfg$triplet_aux_samples_per_edge <- 0L
  cfg$triplet_aux_transfer_mode <- cfg$global_random_transfer_mode
  cfg$triplet_aux_native_backend <- "cpu_cpp"
  cfg$triplet_aux_optimizer_backend <- if (identical(ctx$method, "trimap")) {
    "cpu_cpp_trimap_extra_random_triplets"
  } else {
    "cpu_cpp_global_random_triplets"
  }
  cfg$triplet_aux_n_epochs <- as.integer(cfg$n_epochs)
  cfg$triplet_aux_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$triplet_structured_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_structured_weight <- as.numeric(weight)
  cfg$triplet_structured_n_inliers <- 0L
  cfg$triplet_structured_n_outliers <- 0L
  cfg$triplet_structured_n_random <- samples_per_point
  cfg$triplet_structured_total <- samples_per_point
  cfg$triplet_structured_mode <- "global_random_long_range_triplets"
  cfg$optimizer_backend <- cfg$triplet_aux_optimizer_backend
  cfg$graph_prep_backend <- "cpu"
  if (identical(ctx$method, "tsne")) {
    cfg$tsne_mode <- "sparse_negative_sampling_global_random_triplets"
    cfg$affinity_backend <- "cpu_knn_sparse"
  }
  cfg
}

run_global_random_triplets <- function(ctx,
                                       weight,
                                       samples_per_point,
                                       trimap_extra_negatives) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- global_random_triplet_config(ctx, weight, samples_per_point, trimap_extra_negatives)
  init <- if (identical(ctx$method, "tsne")) {
    fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  } else {
    fastEmbedR:::spectral_knn_init(
      knn$indices,
      knn$distances,
      n_components = 2L,
      spectral_n_iter = cfg$spectral_n_iter,
      seed = ctx$seed,
      backend = "cpu"
    )
  }
  cfg$init_backend <- attr(init, "backend")
  optimizer <- if (identical(ctx$method, "tsne")) "momentum" else "sgd"
  momentum <- if (identical(optimizer, "momentum")) 0.5 else 0.0
  final_momentum <- if (identical(optimizer, "momentum")) 0.8 else 0.0
  momentum_switch <- if (identical(optimizer, "momentum")) {
    max(1L, min(250L, as.integer(floor(cfg$n_epochs / 2))))
  } else {
    cfg$n_epochs + 1L
  }
  layout <- if (identical(ctx$method, "trimap")) {
    fastEmbedR:::knn_objective_embed_optimizer_cpp(
      knn$indices,
      knn$distances,
      "trimap",
      init,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      "sgd",
      0.0,
      0.0,
      as.integer(cfg$n_epochs + 1L),
      0.9,
      0.999,
      1e-8,
      FALSE
    )
  } else {
    fastEmbedR:::knn_objective_embed_structured_triplets_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      init,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      optimizer,
      momentum,
      final_momentum,
      as.integer(momentum_switch),
      0.9,
      0.999,
      1e-8,
      as.numeric(weight),
      0L,
      0L,
      as.integer(samples_per_point),
      FALSE
    )
  }
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- optimizer
  cfg$optimizer_momentum <- momentum
  cfg$optimizer_final_momentum <- final_momentum
  cfg$optimizer_switch_iter <- momentum_switch
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

global_random_triplet_strategy <- function(weight,
                                           samples_per_point,
                                           trimap_extra_negatives = samples_per_point) {
  weight <- as.numeric(weight)
  samples_per_point <- as.integer(samples_per_point)
  trimap_extra_negatives <- as.integer(trimap_extra_negatives)
  label <- global_random_triplet_label(weight, samples_per_point, trimap_extra_negatives)
  list(
    id = paste0("global_random_triplets_", label),
    family = "triplet_constraints",
    description = paste0(
      "Global random long-range triplets. TriMap receives extra random outlier triplets; ",
      "UMAP, t-SNE, PaCMAP, and LocalMAP receive auxiliary triplets with random non-neighbour negatives."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    availability = function() {
      structured_ok <- exists("knn_objective_embed_structured_triplets_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      optimizer_ok <- exists("knn_objective_embed_optimizer_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = structured_ok && optimizer_ok,
        message = "Global random triplets require native CPU C++ structured-triplet and optimizer entry points."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= 10L && samples_per_point >= 1L,
      message = "Global random triplets require --k >= 4, --short-epochs >= 10, and at least one random triplet."
    ),
    params = function(ctx) {
      cfg <- global_random_triplet_config(ctx, weight, samples_per_point, trimap_extra_negatives)
      list(
        k = ctx$k,
        global_random_triplet_weight = cfg$global_random_triplet_weight,
        global_random_triplets_per_point = cfg$global_random_triplets_per_point,
        global_random_trimap_extra_negatives = cfg$global_random_trimap_extra_negatives,
        global_random_negative_source = cfg$global_random_negative_source,
        global_random_transfer_mode = cfg$global_random_transfer_mode,
        global_random_effective_negative_sample_rate = cfg$global_random_effective_negative_sample_rate,
        n_epochs = cfg$n_epochs,
        backend = "cpu"
      )
    },
    run = function(ctx) run_global_random_triplets(ctx, weight, samples_per_point, trimap_extra_negatives),
    global_random_triplet_enabled = TRUE,
    global_random_triplet_weight = weight,
    global_random_triplets_per_point = samples_per_point,
    global_random_trimap_extra_negatives = trimap_extra_negatives,
    global_random_negative_source = "deterministic_random_non_neighbour",
    global_random_transfer_mode = "method_specific_global_random_triplets",
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = weight,
    triplet_aux_samples_per_edge = 0L,
    triplet_aux_native_backend = "cpu_cpp",
    triplet_structured_enabled = TRUE,
    triplet_structured_weight = weight,
    triplet_structured_n_inliers = 0L,
    triplet_structured_n_outliers = 0L,
    triplet_structured_n_random = samples_per_point,
    triplet_structured_total = samples_per_point,
    triplet_structured_mode = "global_random_long_range_triplets"
  )
}

global_random_triplet_strategy_grid <- function() {
  list(
    global_random_triplet_strategy(0.02, 1L, 1L),
    global_random_triplet_strategy(0.03, 2L, 2L),
    global_random_triplet_strategy(0.05, 4L, 3L)
  )
}

hard_negative_label <- function(rate,
                                weight_multiplier,
                                structured_weight,
                                n_inliers,
                                n_outliers,
                                n_random) {
  paste0(
    "r",
    gsub("\\.", "p", format(as.numeric(rate), trim = TRUE, scientific = FALSE)),
    "_m",
    gsub("\\.", "p", format(as.numeric(weight_multiplier), trim = TRUE, scientific = FALSE)),
    "_w",
    gsub("\\.", "p", format(as.numeric(structured_weight), trim = TRUE, scientific = FALSE)),
    "_i",
    as.integer(n_inliers),
    "_o",
    as.integer(n_outliers),
    "_x",
    as.integer(n_random)
  )
}

hard_negative_config <- function(ctx,
                                 rate,
                                 weight_multiplier,
                                 structured_weight,
                                 n_inliers,
                                 n_outliers,
                                 n_random) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(ctx$k),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- max(10L, min(as.integer(cfg$n_epochs), as.integer(ctx$short_epochs)))
  cfg$hard_negative_enabled <- TRUE
  cfg$hard_negative_rate <- as.numeric(rate)
  cfg$hard_negative_weight_multiplier <- as.numeric(weight_multiplier)
  cfg$hard_negative_candidate_source <- "second_order_non_neighbour_then_random_fallback"
  cfg$hard_negative_transfer_mode <- switch(
    ctx$method,
    trimap = "trimap_triplet_outlier_hard_negative_sampling",
    tsne = "tsne_repulsive_hard_negative_sampling",
    pacmap = "pacmap_far_pair_and_repulsive_hard_negative_sampling",
    localmap = "localmap_nonlocal_hard_negative_sampling",
    umap = "umap_negative_sample_hard_negative_sampling",
    "hard_negative_sampling"
  )
  cfg$triplet_aux_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_aux_weight <- as.numeric(structured_weight)
  cfg$triplet_aux_samples_per_edge <- 0L
  cfg$triplet_aux_transfer_mode <- cfg$hard_negative_transfer_mode
  cfg$triplet_aux_native_backend <- "cpu_cpp"
  cfg$triplet_aux_optimizer_backend <- "cpu_cpp_hard_negatives"
  cfg$triplet_aux_n_epochs <- as.integer(cfg$n_epochs)
  cfg$triplet_aux_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$triplet_structured_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_structured_weight <- as.numeric(structured_weight)
  cfg$triplet_structured_n_inliers <- as.integer(n_inliers)
  cfg$triplet_structured_n_outliers <- as.integer(n_outliers)
  cfg$triplet_structured_n_random <- as.integer(n_random)
  cfg$triplet_structured_total <- as.integer(n_inliers + n_outliers + n_random)
  cfg$triplet_structured_mode <- "hard_negative_triplets"
  cfg$optimizer_backend <- "cpu_cpp_hard_negatives"
  cfg$graph_prep_backend <- "cpu"
  if (identical(ctx$method, "tsne")) {
    cfg$tsne_mode <- "sparse_repulsive_hard_negative_sampling"
    cfg$affinity_backend <- "cpu_knn_sparse"
  }
  cfg
}

run_hard_negatives <- function(ctx,
                               rate,
                               weight_multiplier,
                               structured_weight,
                               n_inliers,
                               n_outliers,
                               n_random) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- hard_negative_config(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
  init <- if (identical(ctx$method, "tsne")) {
    fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  } else {
    fastEmbedR:::spectral_knn_init(
      knn$indices,
      knn$distances,
      n_components = 2L,
      spectral_n_iter = cfg$spectral_n_iter,
      seed = ctx$seed,
      backend = "cpu"
    )
  }
  cfg$init_backend <- attr(init, "backend")
  optimizer <- if (identical(ctx$method, "tsne")) "momentum" else "sgd"
  momentum <- if (identical(optimizer, "momentum")) 0.5 else 0.0
  final_momentum <- if (identical(optimizer, "momentum")) 0.8 else 0.0
  momentum_switch <- if (identical(optimizer, "momentum")) {
    max(1L, min(250L, as.integer(floor(cfg$n_epochs / 2))))
  } else {
    cfg$n_epochs + 1L
  }
  layout <- fastEmbedR:::knn_objective_embed_hard_negatives_cpp(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    optimizer,
    momentum,
    final_momentum,
    as.integer(momentum_switch),
    0.9,
    0.999,
    1e-8,
    as.numeric(structured_weight),
    as.integer(n_inliers),
    as.integer(n_outliers),
    as.integer(n_random),
    as.numeric(rate),
    as.numeric(weight_multiplier),
    FALSE
  )
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- optimizer
  cfg$optimizer_momentum <- momentum
  cfg$optimizer_final_momentum <- final_momentum
  cfg$optimizer_switch_iter <- momentum_switch
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

hard_negative_strategy <- function(rate,
                                   weight_multiplier,
                                   structured_weight,
                                   n_inliers,
                                   n_outliers,
                                   n_random) {
  rate <- as.numeric(rate)
  weight_multiplier <- as.numeric(weight_multiplier)
  structured_weight <- as.numeric(structured_weight)
  n_inliers <- as.integer(n_inliers)
  n_outliers <- as.integer(n_outliers)
  n_random <- as.integer(n_random)
  label <- hard_negative_label(rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
  list(
    id = paste0("hard_negatives_", label),
    family = "triplet_constraints",
    description = paste0(
      "Hard-negative transfer using second-order non-neighbours as difficult outliers. ",
      "rate=", format(rate, trim = TRUE, scientific = FALSE),
      ", multiplier=", format(weight_multiplier, trim = TRUE, scientific = FALSE),
      ", n_inliers=", n_inliers,
      ", n_outliers=", n_outliers,
      ", n_random=", n_random, "."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    availability = function() {
      ok <- exists("knn_objective_embed_hard_negatives_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = ok,
        message = "Hard-negative transfer requires the native CPU C++ hard-negative entry point."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= 10L,
      message = "Hard-negative transfer requires --k >= 4 and --short-epochs >= 10."
    ),
    params = function(ctx) {
      cfg <- hard_negative_config(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
      list(
        k = ctx$k,
        hard_negative_rate = cfg$hard_negative_rate,
        hard_negative_weight_multiplier = cfg$hard_negative_weight_multiplier,
        hard_negative_candidate_source = cfg$hard_negative_candidate_source,
        hard_negative_transfer_mode = cfg$hard_negative_transfer_mode,
        triplet_structured_weight = cfg$triplet_structured_weight,
        n_inliers = cfg$triplet_structured_n_inliers,
        n_outliers = cfg$triplet_structured_n_outliers,
        n_random = cfg$triplet_structured_n_random,
        backend = "cpu"
      )
    },
    run = function(ctx) run_hard_negatives(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random),
    hard_negative_enabled = TRUE,
    hard_negative_rate = rate,
    hard_negative_weight_multiplier = weight_multiplier,
    hard_negative_candidate_source = "second_order_non_neighbour_then_random_fallback",
    hard_negative_transfer_mode = "method_specific_hard_negative_sampling",
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = structured_weight,
    triplet_aux_samples_per_edge = 0L,
    triplet_aux_native_backend = "cpu_cpp",
    triplet_structured_enabled = TRUE,
    triplet_structured_weight = structured_weight,
    triplet_structured_n_inliers = n_inliers,
    triplet_structured_n_outliers = n_outliers,
    triplet_structured_n_random = n_random,
    triplet_structured_total = n_inliers + n_outliers + n_random,
    triplet_structured_mode = "hard_negative_triplets"
  )
}

hard_negative_strategy_grid <- function() {
  list(
    hard_negative_strategy(0.25, 0.75, 0.03, 1L, 1L, 0L),
    hard_negative_strategy(0.50, 1.00, 0.05, 2L, 1L, 0L),
    hard_negative_strategy(1.00, 0.75, 0.05, 2L, 1L, 1L)
  )
}

semihard_triplet_label <- function(rate,
                                   weight_multiplier,
                                   structured_weight,
                                   n_inliers,
                                   n_outliers,
                                   n_random) {
  paste0(
    "r",
    gsub("\\.", "p", format(as.numeric(rate), trim = TRUE, scientific = FALSE)),
    "_m",
    gsub("\\.", "p", format(as.numeric(weight_multiplier), trim = TRUE, scientific = FALSE)),
    "_w",
    gsub("\\.", "p", format(as.numeric(structured_weight), trim = TRUE, scientific = FALSE)),
    "_i",
    as.integer(n_inliers),
    "_o",
    as.integer(n_outliers),
    "_x",
    as.integer(n_random)
  )
}

semihard_triplet_config <- function(ctx,
                                    rate,
                                    weight_multiplier,
                                    structured_weight,
                                    n_inliers,
                                    n_outliers,
                                    n_random) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(ctx$k),
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu"
  )
  cfg$n_epochs <- max(10L, min(as.integer(cfg$n_epochs), as.integer(ctx$short_epochs)))
  cfg$semihard_triplet_enabled <- TRUE
  cfg$semihard_triplet_rate <- as.numeric(rate)
  cfg$semihard_triplet_weight_multiplier <- as.numeric(weight_multiplier)
  cfg$semihard_triplet_candidate_source <- "moderate_weight_second_order_non_neighbour_then_random_fallback"
  cfg$semihard_triplet_transfer_mode <- switch(
    ctx$method,
    trimap = "trimap_semihard_outlier_triplets",
    pacmap = "pacmap_semihard_far_pairs",
    localmap = "localmap_semihard_nonlocal_contrast",
    umap = "umap_semihard_negative_sampling",
    "semihard_triplet_sampling"
  )
  cfg$triplet_aux_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_aux_weight <- as.numeric(structured_weight)
  cfg$triplet_aux_samples_per_edge <- 0L
  cfg$triplet_aux_transfer_mode <- cfg$semihard_triplet_transfer_mode
  cfg$triplet_aux_native_backend <- "cpu_cpp"
  cfg$triplet_aux_optimizer_backend <- "cpu_cpp_semihard_triplets"
  cfg$triplet_aux_n_epochs <- as.integer(cfg$n_epochs)
  cfg$triplet_aux_negative_sample_rate <- as.integer(cfg$negative_sample_rate)
  cfg$triplet_structured_enabled <- !identical(ctx$method, "trimap")
  cfg$triplet_structured_weight <- as.numeric(structured_weight)
  cfg$triplet_structured_n_inliers <- as.integer(n_inliers)
  cfg$triplet_structured_n_outliers <- as.integer(n_outliers)
  cfg$triplet_structured_n_random <- as.integer(n_random)
  cfg$triplet_structured_total <- as.integer(n_inliers + n_outliers + n_random)
  cfg$triplet_structured_mode <- "semihard_triplets"
  cfg$optimizer_backend <- "cpu_cpp_semihard_triplets"
  cfg$graph_prep_backend <- "cpu"
  cfg
}

run_semihard_triplets <- function(ctx,
                                  rate,
                                  weight_multiplier,
                                  structured_weight,
                                  n_inliers,
                                  n_outliers,
                                  n_random) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- semihard_triplet_config(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = cfg$spectral_n_iter,
    seed = ctx$seed,
    backend = "cpu"
  )
  cfg$init_backend <- attr(init, "backend")
  layout <- fastEmbedR:::knn_objective_embed_semihard_triplets_cpp(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    "sgd",
    0.0,
    0.0,
    as.integer(cfg$n_epochs + 1L),
    0.9,
    0.999,
    1e-8,
    as.numeric(structured_weight),
    as.integer(n_inliers),
    as.integer(n_outliers),
    as.integer(n_random),
    as.numeric(rate),
    as.numeric(weight_multiplier),
    FALSE
  )
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- "sgd"
  cfg$optimizer_momentum <- 0.0
  cfg$optimizer_final_momentum <- 0.0
  cfg$optimizer_switch_iter <- cfg$n_epochs + 1L
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

approx_triplet_source_config <- function(ctx,
                                         source,
                                         source_detail,
                                         rate,
                                         weight_multiplier,
                                         structured_weight,
                                         n_inliers,
                                         n_outliers,
                                         n_random) {
  cfg <- semihard_triplet_config(
    ctx,
    rate,
    weight_multiplier,
    structured_weight,
    n_inliers,
    n_outliers,
    n_random
  )
  cfg$triplet_mining_approximate <- TRUE
  cfg$triplet_mining_graph_source <- as.character(source)
  cfg$triplet_mining_source_detail <- as.character(source_detail)
  cfg$triplet_mining_knn_backend <- safe_character(attr(ctx$knn, "backend"), as.character(source))
  cfg$triplet_mining_transfer_mode <- switch(
    ctx$method,
    trimap = "trimap_triplets_from_approximate_knn",
    umap = "umap_auxiliary_triplets_from_approximate_knn",
    pacmap = "pacmap_auxiliary_triplets_from_approximate_knn",
    "approximate_knn_triplet_mining"
  )
  cfg$triplet_mining_recall_at_k <- safe_number(ctx$knn_recall_at_k)
  cfg$triplet_mining_rank_correlation <- safe_number(ctx$knn_rank_correlation)
  cfg$triplet_mining_distance_error <- safe_number(ctx$knn_mean_distance_error)
  cfg$triplet_mining_quality_sample_size <- safe_number(ctx$knn_quality_sample_size)
  cfg$triplet_mining_candidate_source <- paste0(
    "approximate_",
    as.character(source),
    "_knn_second_order_non_neighbour"
  )
  cfg$semihard_triplet_candidate_source <- cfg$triplet_mining_candidate_source
  cfg$semihard_triplet_transfer_mode <- cfg$triplet_mining_transfer_mode
  cfg$triplet_aux_transfer_mode <- cfg$triplet_mining_transfer_mode
  cfg$triplet_aux_optimizer_backend <- "cpu_cpp_approx_knn_triplet_mining"
  cfg$triplet_structured_mode <- "approximate_knn_semihard_triplets"
  cfg$optimizer_backend <- "cpu_cpp_approx_knn_triplet_mining"
  cfg$graph_prep_backend <- safe_character(attr(ctx$knn, "backend"), "approximate_knn")
  cfg
}

run_approx_knn_triplet_mining <- function(ctx,
                                          source,
                                          source_detail,
                                          rate,
                                          weight_multiplier,
                                          structured_weight,
                                          n_inliers,
                                          n_outliers,
                                          n_random) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- approx_triplet_source_config(
    ctx,
    source,
    source_detail,
    rate,
    weight_multiplier,
    structured_weight,
    n_inliers,
    n_outliers,
    n_random
  )
  init <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = cfg$spectral_n_iter,
    seed = ctx$seed,
    backend = "cpu"
  )
  cfg$init_backend <- attr(init, "backend")
  layout <- fastEmbedR:::knn_objective_embed_semihard_triplets_cpp(
    knn$indices,
    knn$distances,
    ctx$method,
    init,
    2L,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    as.integer(cfg$n_threads),
    as.integer(ctx$seed),
    "sgd",
    0.0,
    0.0,
    as.integer(cfg$n_epochs + 1L),
    0.9,
    0.999,
    1e-8,
    as.numeric(structured_weight),
    as.integer(n_inliers),
    as.integer(n_outliers),
    as.integer(n_random),
    as.numeric(rate),
    as.numeric(weight_multiplier),
    FALSE
  )
  colnames(layout) <- paste0(toupper(ctx$method), seq_len(ncol(layout)))
  cfg$optimizer_mode <- "sgd"
  cfg$optimizer_momentum <- 0.0
  cfg$optimizer_final_momentum <- 0.0
  cfg$optimizer_switch_iter <- cfg$n_epochs + 1L
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

semihard_triplet_strategy <- function(rate,
                                      weight_multiplier,
                                      structured_weight,
                                      n_inliers,
                                      n_outliers,
                                      n_random) {
  rate <- as.numeric(rate)
  weight_multiplier <- as.numeric(weight_multiplier)
  structured_weight <- as.numeric(structured_weight)
  n_inliers <- as.integer(n_inliers)
  n_outliers <- as.integer(n_outliers)
  n_random <- as.integer(n_random)
  label <- semihard_triplet_label(rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
  list(
    id = paste0("semihard_triplets_", label),
    family = "triplet_constraints",
    description = paste0(
      "Semi-hard triplet transfer using moderate second-order non-neighbours. ",
      "rate=", format(rate, trim = TRUE, scientific = FALSE),
      ", multiplier=", format(weight_multiplier, trim = TRUE, scientific = FALSE),
      ", n_inliers=", n_inliers,
      ", n_outliers=", n_outliers,
      ", n_random=", n_random, "."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    knn_backend = function(ctx) "cpu",
    knn_cache_strategy_id = "exact_knn",
    availability = function() {
      ok <- exists("knn_objective_embed_semihard_triplets_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = ok,
        message = "Semi-hard triplet transfer requires the native CPU C++ semi-hard entry point."
      )
    },
    context_available = function(ctx) list(
      available = ctx$k >= 4L && ctx$short_epochs >= 10L,
      message = "Semi-hard triplet transfer requires --k >= 4 and --short-epochs >= 10."
    ),
    params = function(ctx) {
      cfg <- semihard_triplet_config(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random)
      list(
        k = ctx$k,
        semihard_triplet_rate = cfg$semihard_triplet_rate,
        semihard_triplet_weight_multiplier = cfg$semihard_triplet_weight_multiplier,
        semihard_triplet_candidate_source = cfg$semihard_triplet_candidate_source,
        semihard_triplet_transfer_mode = cfg$semihard_triplet_transfer_mode,
        triplet_structured_weight = cfg$triplet_structured_weight,
        n_inliers = cfg$triplet_structured_n_inliers,
        n_outliers = cfg$triplet_structured_n_outliers,
        n_random = cfg$triplet_structured_n_random,
        backend = "cpu"
      )
    },
    run = function(ctx) run_semihard_triplets(ctx, rate, weight_multiplier, structured_weight, n_inliers, n_outliers, n_random),
    semihard_triplet_enabled = TRUE,
    semihard_triplet_rate = rate,
    semihard_triplet_weight_multiplier = weight_multiplier,
    semihard_triplet_candidate_source = "moderate_weight_second_order_non_neighbour_then_random_fallback",
    semihard_triplet_transfer_mode = "method_specific_semihard_triplet_sampling",
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = structured_weight,
    triplet_aux_samples_per_edge = 0L,
    triplet_aux_native_backend = "cpu_cpp",
    triplet_structured_enabled = TRUE,
    triplet_structured_weight = structured_weight,
    triplet_structured_n_inliers = n_inliers,
    triplet_structured_n_outliers = n_outliers,
    triplet_structured_n_random = n_random,
    triplet_structured_total = n_inliers + n_outliers + n_random,
    triplet_structured_mode = "semihard_triplets"
  )
}

semihard_triplet_strategy_grid <- function() {
  list(
    semihard_triplet_strategy(0.25, 0.75, 0.03, 1L, 1L, 0L),
    semihard_triplet_strategy(0.50, 0.75, 0.05, 2L, 1L, 0L),
    semihard_triplet_strategy(0.50, 1.00, 0.05, 2L, 2L, 1L)
  )
}

approx_triplet_mining_strategy <- function(source,
                                           source_detail,
                                           build_knn,
                                           availability,
                                           context_available,
                                           rate = 0.25,
                                           weight_multiplier = 0.75,
                                           structured_weight = 0.03,
                                           n_inliers = 1L,
                                           n_outliers = 1L,
                                           n_random = 0L,
                                           extra_params = list(),
                                           strategy_fields = list()) {
  source <- as.character(source)
  source_detail <- as.character(source_detail)
  label <- semihard_triplet_label(
    rate,
    weight_multiplier,
    structured_weight,
    n_inliers,
    n_outliers,
    n_random
  )
  id <- paste0(
    "approx_triplet_mining_",
    gsub("[^A-Za-z0-9]+", "_", tolower(source_detail)),
    "_",
    label
  )
  out <- list(
    id = id,
    family = "triplet_constraints",
    description = paste0(
      "Triplet mining from an approximate KNN graph (",
      source_detail,
      "). TriMap uses the graph directly for triplets; UMAP and PaCMAP use auxiliary semi-hard triplets."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "pacmap", "trimap") && identical(backend, "cpu")
    },
    availability = function() {
      source_availability <- availability()
      native_ok <- exists("knn_objective_embed_semihard_triplets_cpp", envir = asNamespace("fastEmbedR"), inherits = FALSE)
      list(
        available = isTRUE(source_availability$available) && native_ok,
        message = if (!isTRUE(source_availability$available)) {
          source_availability$message
        } else {
          "Approximate triplet mining requires the native CPU C++ semi-hard triplet entry point."
        }
      )
    },
    context_available = function(ctx) {
      source_context <- context_available(ctx)
      if (!isTRUE(source_context$available)) return(source_context)
      list(
        available = ctx$k >= 4L && ctx$short_epochs >= 10L,
        message = "Approximate triplet mining requires --k >= 4 and --short-epochs >= 10."
      )
    },
    build_knn = build_knn,
    params = function(ctx) {
      cfg <- approx_triplet_source_config(
        ctx,
        source,
        source_detail,
        rate,
        weight_multiplier,
        structured_weight,
        n_inliers,
        n_outliers,
        n_random
      )
      c(
        list(
          k = ctx$k,
          triplet_mining_approximate = TRUE,
          triplet_mining_graph_source = source,
          triplet_mining_source_detail = source_detail,
          triplet_mining_transfer_mode = cfg$triplet_mining_transfer_mode,
          semihard_triplet_rate = cfg$semihard_triplet_rate,
          semihard_triplet_weight_multiplier = cfg$semihard_triplet_weight_multiplier,
          triplet_structured_weight = cfg$triplet_structured_weight,
          n_inliers = cfg$triplet_structured_n_inliers,
          n_outliers = cfg$triplet_structured_n_outliers,
          n_random = cfg$triplet_structured_n_random,
          n_epochs = cfg$n_epochs,
          backend = "cpu"
        ),
        extra_params
      )
    },
    run = function(ctx) run_approx_knn_triplet_mining(
      ctx,
      source,
      source_detail,
      rate,
      weight_multiplier,
      structured_weight,
      n_inliers,
      n_outliers,
      n_random
    ),
    triplet_mining_approximate = TRUE,
    triplet_mining_graph_source = source,
    triplet_mining_source_detail = source_detail,
    triplet_mining_transfer_mode = "method_specific_approximate_knn_triplet_mining",
    triplet_mining_candidate_source = paste0("approximate_", source, "_knn_second_order_non_neighbour"),
    semihard_triplet_enabled = TRUE,
    semihard_triplet_rate = as.numeric(rate),
    semihard_triplet_weight_multiplier = as.numeric(weight_multiplier),
    semihard_triplet_candidate_source = paste0("approximate_", source, "_knn_second_order_non_neighbour"),
    semihard_triplet_transfer_mode = "method_specific_approximate_knn_triplet_mining",
    triplet_aux_enabled = TRUE,
    triplet_aux_weight = as.numeric(structured_weight),
    triplet_aux_samples_per_edge = 0L,
    triplet_aux_native_backend = "cpu_cpp",
    triplet_structured_enabled = TRUE,
    triplet_structured_weight = as.numeric(structured_weight),
    triplet_structured_n_inliers = as.integer(n_inliers),
    triplet_structured_n_outliers = as.integer(n_outliers),
    triplet_structured_n_random = as.integer(n_random),
    triplet_structured_total = as.integer(n_inliers + n_outliers + n_random),
    triplet_structured_mode = "approximate_knn_semihard_triplets"
  )
  c(out, strategy_fields)
}

approx_triplet_mining_strategy_grid <- function() {
  list(
    approx_triplet_mining_strategy(
      source = "annoy",
      source_detail = "annoy_t25_sk2x",
      build_knn = annoy_knn,
      availability = annoy_available,
      context_available = function(ctx) list(
        available = identical(normalize_knn_metric(ctx$knn_metric), "euclidean"),
        message = "Approximate Annoy triplet mining currently supports Euclidean KNN."
      ),
      extra_params = list(
        approximate_knn = "annoy",
        implementation = "RcppAnnoy",
        n_trees = 25L,
        search_multiplier = 2
      ),
      strategy_fields = list(
        annoy_n_trees = 25L,
        annoy_search_multiplier = 2
      )
    ),
    approx_triplet_mining_strategy(
      source = "hnsw",
      source_detail = "hnsw_m16_efc200_efs100",
      build_knn = hnsw_knn,
      availability = hnsw_available,
      context_available = function(ctx) list(
        available = !is.na(hnsw_distance(ctx$knn_metric)),
        message = "Approximate HNSW triplet mining supports Euclidean, cosine, and inner-product KNN."
      ),
      extra_params = list(
        approximate_knn = "hnsw",
        implementation = "RcppHNSW",
        M = 16L,
        ef_construction = 200L,
        ef_search = 100L
      ),
      strategy_fields = list(
        hnsw_m = 16L,
        hnsw_ef_construction = 200L,
        hnsw_ef_search = 100L
      )
    ),
    approx_triplet_mining_strategy(
      source = "nndescent",
      source_detail = "nnd_i10_d0p001_rho1",
      build_knn = nndescent_knn,
      availability = nndescent_available,
      context_available = function(ctx) list(
        available = !is.na(nndescent_metric(ctx$knn_metric)),
        message = "Approximate NN-descent triplet mining does not support this metric through rnndescent."
      ),
      extra_params = list(
        approximate_knn = "nndescent",
        implementation = "rnndescent",
        n_iters = 10L,
        delta = 0.001,
        rho = 1
      ),
      strategy_fields = list(
        nndescent_n_iters = 10L,
        nndescent_delta = 0.001,
        nndescent_rho = 1,
        nndescent_max_candidates = NA_integer_
      )
    )
  )
}

fitsne_fft_config <- function(ctx, k = ctx$k, n_epochs = NULL) {
  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = as.integer(k),
    objective = "tsne",
    quality = "fft",
    backend = "cpu"
  )
  if (!is.null(n_epochs)) {
    cfg$n_epochs <- as.integer(n_epochs)
  }
  cfg$optimizer_backend <- "fitsne_fft"
  cfg$affinity_backend <- "cpu_knn_csr"
  cfg
}

fitsne_grid_label <- function(nterms, intervals_per_integer, min_num_intervals) {
  paste0(
    "o", as.integer(nterms),
    "_g", gsub("\\.", "p", format(as.numeric(intervals_per_integer), trim = TRUE, scientific = FALSE)),
    "_m", as.integer(min_num_intervals)
  )
}

run_fitsne_fft_base_layout <- function(ctx,
                                       n_epochs = NULL,
                                       nterms = ctx$fft_grid_nterms,
                                       intervals_per_integer = ctx$fft_grid_intervals_per_integer,
                                       min_num_intervals = ctx$fft_grid_min_num_intervals) {
  knn <- fastEmbedR:::coerce_knn_input(ctx$knn)
  cfg <- fitsne_fft_config(ctx, ncol(knn$indices), n_epochs)
  nterms <- as.integer(nterms)
  intervals_per_integer <- as.numeric(intervals_per_integer)
  min_num_intervals <- as.integer(min_num_intervals)
  init <- fastEmbedR:::tsne_random_init(nrow(knn$indices), 2L, ctx$seed)
  layout <- fastEmbedR:::knn_tsne_fitsne_fft(
    knn$indices,
    knn$distances,
    init,
    n_epochs = as.integer(cfg$n_epochs),
    perplexity = cfg$perplexity,
    theta = cfg$theta,
    learning_rate = cfg$learning_rate,
    stop_lying_iter = as.integer(cfg$stop_lying_iter),
    mom_switch_iter = as.integer(cfg$mom_switch_iter),
    momentum = cfg$momentum,
    final_momentum = cfg$final_momentum,
    exaggeration_factor = cfg$exaggeration_factor,
    n_threads = as.integer(cfg$n_threads),
    seed = as.integer(ctx$seed),
    verbose = FALSE,
    nterms = nterms,
    intervals_per_integer = intervals_per_integer,
    min_num_intervals = min_num_intervals
  )
  colnames(layout) <- paste0("TSNE", seq_len(ncol(layout)))
  cfg$fitsne_path <- attr(layout, "fitsne_path")
  cfg$fft_grid_nterms <- nterms
  cfg$fft_grid_intervals_per_integer <- intervals_per_integer
  cfg$fft_grid_min_num_intervals <- min_num_intervals
  attr(layout, "fitsne_path") <- NULL
  attr(layout, "fastEmbedR_config") <- cfg
  list(layout = layout, knn = knn, cfg = cfg)
}

run_fitsne_fft_experimental_transfer <- function(ctx) {
  refine_epochs <- if (identical(ctx$method, "tsne")) {
    0L
  } else {
    max(1L, min(as.integer(ctx$short_epochs), 80L))
  }
  base <- run_fitsne_fft_base_layout(ctx)
  if (identical(ctx$method, "tsne")) {
    cfg <- attr(base$layout, "fastEmbedR_config")
    cfg$fft_interpolation_mode <- "true_tsne_fitsne_fft"
    cfg$fft_transfer_experimental <- FALSE
    attr(base$layout, "fastEmbedR_config") <- cfg
    return(base$layout)
  }
  layout <- fastEmbedR:::refine_embedding_from_knn(
    method = ctx$method,
    indices = base$knn$indices,
    distances = base$knn$distances,
    init_layout = base$layout,
    n_epochs = refine_epochs,
    refinement = "fft_warm_start",
    seed = ctx$seed,
    backend = "cpu",
    verbose = FALSE
  )
  cfg <- attr(layout, "fastEmbedR_config")
  cfg$fft_interpolation_mode <- "experimental_fitsne_warm_start_refine"
  cfg$fft_transfer_experimental <- TRUE
  cfg$fft_transfer_native_repulsion <- FALSE
  cfg$fft_transfer_refine_epochs <- refine_epochs
  cfg$fft_transfer_base_optimizer <- "fitsne_fft"
  cfg$fitsne_path <- base$cfg$fitsne_path
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

fitsne_fft_experimental_strategy <- function() {
  list(
    id = "fitsne_fft_experimental",
    family = "tsne_repulsion",
    description = paste0(
      "FIt-SNE FFT interpolation. For t-SNE this is the true FFT repulsive-field baseline; ",
      "for UMAP, PaCMAP, TriMap, and LocalMAP this is only an experimental FFT t-SNE warm start ",
      "followed by method-specific CPU refinement."
    ),
    availability = function() list(
      available = fast_tsne_available(),
      message = "FIt-SNE FFT requires an executable `fast_tsne`; set FASTEMBEDR_FAST_TSNE_PATH or FAST_TSNE_PATH."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    params = function(ctx) {
      cfg <- fitsne_fft_config(ctx)
      refine_epochs <- if (identical(ctx$method, "tsne")) 0L else max(1L, min(as.integer(ctx$short_epochs), 80L))
      list(
        k = ctx$k,
        optimizer = "fitsne_fft",
        interpolation = "fft",
        mode = if (identical(ctx$method, "tsne")) "true_tsne_fft" else "experimental_warm_start_transfer",
        transfer_scope = "experimental_only",
        native_repulsive_field_for_method = identical(ctx$method, "tsne"),
        refine_objective = ctx$method,
        refine_epochs = refine_epochs,
        perplexity = cfg$perplexity,
        theta = cfg$theta,
        learning_rate = cfg$learning_rate,
        n_epochs = cfg$n_epochs,
        fast_tsne = fastEmbedR:::fitsne_binary_path(required = FALSE)
      )
    },
    run = run_fitsne_fft_experimental_transfer,
    fft_interpolation_mode = "fitsne_fft_experimental",
    fft_interpolation_experimental = TRUE,
    fft_grid_nterms = 3L,
    fft_grid_intervals_per_integer = 1,
    fft_grid_min_num_intervals = 50L
  )
}

fitsne_fft_grid_strategy <- function(id,
                                     nterms,
                                     intervals_per_integer,
                                     min_num_intervals,
                                     description) {
  base <- fitsne_fft_experimental_strategy()
  base$id <- paste0("fitsne_fft_grid_", id)
  base$description <- paste0(description, " Uses the same true t-SNE FFT / experimental warm-start transfer semantics as `fitsne_fft_experimental`.")
  base$fft_grid_nterms <- as.integer(nterms)
  base$fft_grid_intervals_per_integer <- as.numeric(intervals_per_integer)
  base$fft_grid_min_num_intervals <- as.integer(min_num_intervals)
  base$params <- function(ctx) {
    cfg <- fitsne_fft_config(ctx)
    refine_epochs <- if (identical(ctx$method, "tsne")) 0L else max(1L, min(as.integer(ctx$short_epochs), 80L))
    list(
      k = ctx$k,
      optimizer = "fitsne_fft",
      interpolation = "fft",
      interpolation_grid = id,
      interpolation_order_nterms = as.integer(nterms),
      intervals_per_integer = as.numeric(intervals_per_integer),
      min_num_intervals = as.integer(min_num_intervals),
      mode = if (identical(ctx$method, "tsne")) "true_tsne_fft" else "experimental_warm_start_transfer",
      transfer_scope = "experimental_only",
      native_repulsive_field_for_method = identical(ctx$method, "tsne"),
      refine_objective = ctx$method,
      refine_epochs = refine_epochs,
      perplexity = cfg$perplexity,
      theta = cfg$theta,
      learning_rate = cfg$learning_rate,
      n_epochs = cfg$n_epochs,
      fast_tsne = fastEmbedR:::fitsne_binary_path(required = FALSE)
    )
  }
  base
}

fitsne_fft_grid_strategy_grid <- function() {
  list(
    fitsne_fft_grid_strategy(
      "baseline_o3_g1_m50",
      nterms = 3L,
      intervals_per_integer = 1,
      min_num_intervals = 50L,
      description = "Baseline FIt-SNE interpolation grid: order 3, default grid spacing, minimum 50 intervals."
    ),
    fitsne_fft_grid_strategy(
      "coarse_o3_g2_m25",
      nterms = 3L,
      intervals_per_integer = 2,
      min_num_intervals = 25L,
      description = "Coarser FIt-SNE interpolation grid: same interpolation order, larger grid spacing, lower minimum grid resolution."
    ),
    fitsne_fft_grid_strategy(
      "loworder_o2_g1_m50",
      nterms = 2L,
      intervals_per_integer = 1,
      min_num_intervals = 50L,
      description = "Lower-order FIt-SNE interpolation: order 2 with baseline grid spacing and minimum resolution."
    ),
    fitsne_fft_grid_strategy(
      "fine_o4_g075_m75",
      nterms = 4L,
      intervals_per_integer = 0.75,
      min_num_intervals = 75L,
      description = "Finer FIt-SNE interpolation grid: order 4, denser grid spacing, and higher minimum grid resolution."
    )
  )
}

layout_row_norms <- function(layout) {
  sqrt(rowSums(layout * layout))
}

layout_metric_pair_distances <- function(layout, a, b, metric) {
  layout <- as.matrix(layout)
  metric <- safe_character(metric, "euclidean")
  if (identical(metric, "euclidean")) {
    return(sampled_pair_distances(layout, a, b))
  }
  if (identical(metric, "cosine")) {
    norms <- pmax(layout_row_norms(layout), .Machine$double.eps)
    dot <- rowSums(layout[a, , drop = FALSE] * layout[b, , drop = FALSE])
    cosine <- dot / (norms[a] * norms[b])
    return(pmax(0, pmin(2, 1 - pmax(-1, pmin(1, cosine)))))
  }
  if (identical(metric, "hyperbolic")) {
    radius <- layout_row_norms(layout)
    too_large <- is.finite(radius) & radius >= 0.999
    if (any(too_large)) {
      layout[too_large, ] <- layout[too_large, , drop = FALSE] * (0.999 / radius[too_large])
      radius[too_large] <- 0.999
    }
    duv <- layout[a, , drop = FALSE] - layout[b, , drop = FALSE]
    duv2 <- rowSums(duv * duv)
    ru2 <- pmin(0.998001, rowSums(layout[a, , drop = FALSE]^2))
    rv2 <- pmin(0.998001, rowSums(layout[b, , drop = FALSE]^2))
    arg <- 1 + 2 * duv2 / pmax(.Machine$double.eps, (1 - ru2) * (1 - rv2))
    return(acosh(pmax(1, arg)))
  }
  sampled_pair_distances(layout, a, b)
}

output_metric_global_metrics <- function(x_high,
                                         layout,
                                         metric,
                                         sample_size,
                                         seed) {
  metric <- safe_character(metric, "euclidean")
  n <- nrow(x_high)
  if (n < 3L) {
    return(list(
      output_metric_distance_spearman = NA_real_,
      output_metric_distance_pearson = NA_real_,
      output_metric_stress = NA_real_,
      output_metric_global_sample_size = n,
      output_metric_global_pair_count = 0L
    ))
  }
  sample_size <- min(as.integer(sample_size), n)
  set.seed(seed + 7919L)
  keep <- if (sample_size < n) sort(sample.int(n, sample_size)) else seq_len(n)
  pair_total <- length(keep) * (length(keep) - 1L) / 2
  max_pairs <- min(pair_total, 250000L)
  if (pair_total <= max_pairs) {
    pairs <- utils::combn(length(keep), 2L)
    a <- pairs[1L, ]
    b <- pairs[2L, ]
  } else {
    m <- as.integer(max_pairs)
    a <- sample.int(length(keep), m, replace = TRUE)
    b <- sample.int(length(keep) - 1L, m, replace = TRUE)
    b <- b + as.integer(b >= a)
  }
  x_sample <- x_high[keep, , drop = FALSE]
  layout_sample <- layout[keep, , drop = FALSE]
  high_dist <- sampled_pair_distances(x_sample, a, b)
  low_dist <- layout_metric_pair_distances(layout_sample, a, b, metric)
  list(
    output_metric_distance_spearman = safe_numeric_cor(high_dist, low_dist, method = "spearman"),
    output_metric_distance_pearson = safe_numeric_cor(high_dist, low_dist, method = "pearson"),
    output_metric_stress = normalized_stress(high_dist, low_dist),
    output_metric_global_sample_size = length(keep),
    output_metric_global_pair_count = length(high_dist)
  )
}

output_metric_transform_layout <- function(layout,
                                           metric,
                                           native = FALSE) {
  metric <- match.arg(metric, c("euclidean", "cosine", "hyperbolic"))
  out <- as.matrix(layout)
  storage.mode(out) <- "double"
  transform <- "none"
  radius <- layout_row_norms(out)
  projection_scale <- NA_real_
  curvature <- NA_real_
  if (identical(metric, "cosine")) {
    center <- apply(out, 2L, stats::median, na.rm = TRUE)
    out <- sweep(out, 2L, center, "-")
    radius <- layout_row_norms(out)
    zero <- !is.finite(radius) | radius <= .Machine$double.eps
    if (any(zero)) {
      out[zero, 1L] <- 1
      if (ncol(out) > 1L) out[zero, -1L] <- 0
      radius[zero] <- 1
    }
    out <- out / pmax(radius, .Machine$double.eps)
    transform <- "centered_row_l2_normalized_post_transform"
  } else if (identical(metric, "hyperbolic")) {
    center <- apply(out, 2L, stats::median, na.rm = TRUE)
    out <- sweep(out, 2L, center, "-")
    radius <- layout_row_norms(out)
    finite_radius <- radius[is.finite(radius) & radius > 0]
    projection_scale <- stats::median(finite_radius, na.rm = TRUE) +
      2 * stats::mad(finite_radius, constant = 1, na.rm = TRUE)
    if (!is.finite(projection_scale) || projection_scale <= 0) {
      projection_scale <- max(finite_radius, na.rm = TRUE)
    }
    if (!is.finite(projection_scale) || projection_scale <= 0) projection_scale <- 1
    direction <- out / pmax(radius, .Machine$double.eps)
    poincare_radius <- 0.999 * tanh(radius / projection_scale)
    poincare_radius[!is.finite(poincare_radius)] <- 0
    out <- direction * poincare_radius
    transform <- "poincare_disk_projection_post_transform"
    curvature <- -1
  }
  colnames(out) <- colnames(layout)
  stats_radius <- layout_row_norms(out)
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  cfg$output_metric <- metric
  cfg$output_metric_transform <- transform
  cfg$output_metric_native <- isTRUE(native)
  cfg$output_metric_projection_scale <- projection_scale
  cfg$output_metric_curvature <- curvature
  cfg$output_metric_radius_mean <- safe_mean(stats_radius)
  cfg$output_metric_radius_max <- if (all(!is.finite(stats_radius))) NA_real_ else max(stats_radius, na.rm = TRUE)
  cfg$output_metric_norm_mean <- cfg$output_metric_radius_mean
  attr(out, "fastEmbedR_config") <- cfg
  out
}

run_output_metric_strategy <- function(ctx, metric) {
  layout <- fastEmbedR::embed_knn(
    ctx$knn,
    method = ctx$method,
    quality = method_quality(ctx$method, "auto"),
    backend = ctx$backend,
    seed = ctx$seed
  )
  output_metric_transform_layout(
    layout,
    metric = metric,
    native = identical(metric, "euclidean")
  )
}

output_metric_strategy <- function(metric) {
  metric <- match.arg(metric, c("euclidean", "cosine", "hyperbolic"))
  id <- switch(
    metric,
    euclidean = "output_metric_euclidean",
    cosine = "output_metric_cosine_normalized",
    hyperbolic = "output_metric_hyperbolic_poincare"
  )
  description <- switch(
    metric,
    euclidean = "Native Euclidean output geometry baseline.",
    cosine = "Experimental cosine-like output geometry by centering and L2-normalizing the final layout rows.",
    hyperbolic = "Experimental Poincare-disk output geometry by projecting the final layout inside the unit disk."
  )
  list(
    id = id,
    family = "output_metric",
    description = description,
    compatible = function(method, backend) method %in% c("umap", "pacmap", "trimap", "localmap") && backend %in% c("cpu", "cuda", "metal"),
    knn_backend = function(ctx) "cpu",
    output_metric = metric,
    output_metric_transform = switch(
      metric,
      euclidean = "none",
      cosine = "centered_row_l2_normalized_post_transform",
      hyperbolic = "poincare_disk_projection_post_transform"
    ),
    output_metric_native = identical(metric, "euclidean"),
    params = function(ctx) list(
      k = ctx$k,
      output_metric = metric,
      output_metric_native = identical(metric, "euclidean"),
      output_metric_transform = switch(
        metric,
        euclidean = "none",
        cosine = "centered_row_l2_normalized_post_transform",
        hyperbolic = "poincare_disk_projection_post_transform"
      ),
      primary_methods = "umap",
      experimental_methods = "pacmap,trimap,localmap",
      caveat = if (identical(metric, "euclidean")) {
        "native_baseline"
      } else {
        "post_transform_probe_not_a_native_output_metric_optimizer"
      },
      quality = method_quality(ctx$method, "auto")
    ),
    run = function(ctx) run_output_metric_strategy(ctx, metric)
  )
}

output_metric_strategy_grid <- function() {
  list(
    output_metric_strategy("euclidean"),
    output_metric_strategy("cosine"),
    output_metric_strategy("hyperbolic")
  )
}

artificial_neighbor_transfer_mode <- function(method) {
  switch(
    safe_character(method, "localmap"),
    umap = "umap_post_embedding_artificial_neighbour_repulsion",
    tsne = "tsne_post_embedding_artificial_neighbour_repulsion",
    pacmap = "pacmap_post_embedding_artificial_neighbour_repulsion",
    trimap = "trimap_post_embedding_artificial_neighbour_repulsion",
    localmap = "localmap_native_concept_artificial_neighbour_penalty",
    "artificial_neighbour_penalty_transfer"
  )
}

artificial_neighbor_stats <- function(x_high,
                                      layout,
                                      high_knn,
                                      low_k = 15L,
                                      far_multiplier = 1.5,
                                      seed = 1L,
                                      max_pairs = 200000L) {
  n <- nrow(layout)
  low_k <- max(1L, min(as.integer(low_k), n - 1L))
  high_cols <- knn_neighbor_cols(high_knn)
  if (length(high_cols) == 0L || low_k < 1L) {
    return(list(
      false_rate = NA_real_,
      far_rate = NA_real_,
      total_edges = 0L,
      unsupported_edges = 0L,
      far_edges = 0L,
      pairs = data.frame(a = integer(), b = integer(), high_ratio = numeric(), low_distance = numeric()),
      mean_high_distance_ratio = NA_real_,
      mean_low_distance = NA_real_,
      target_distance = NA_real_
    ))
  }
  low_nn <- fastEmbedR::nn(layout, layout, low_k + 1L, backend = "cpu")
  low_cols <- knn_neighbor_cols(low_nn)
  low_cols <- low_cols[seq_len(min(length(low_cols), low_k))]
  high_sets <- lapply(seq_len(n), function(i) {
    x <- as.integer(high_knn$indices[i, high_cols])
    unique(x[is.finite(x) & x >= 1L & x <= n & x != i])
  })
  high_radius <- apply(high_knn$distances[, high_cols, drop = FALSE], 1L, function(row) {
    row <- row[is.finite(row) & row > 0]
    if (length(row) == 0L) NA_real_ else max(row)
  })
  fallback_radius <- stats::median(high_radius[is.finite(high_radius) & high_radius > 0], na.rm = TRUE)
  if (!is.finite(fallback_radius) || fallback_radius <= 0) fallback_radius <- 1
  high_radius[!is.finite(high_radius) | high_radius <= 0] <- fallback_radius

  low_distances <- low_nn$distances[, low_cols, drop = FALSE]
  low_radius <- apply(low_distances, 1L, function(row) {
    row <- row[is.finite(row) & row > 0]
    if (length(row) == 0L) NA_real_ else max(row)
  })
  target_distance <- stats::median(low_radius[is.finite(low_radius) & low_radius > 0], na.rm = TRUE)
  if (!is.finite(target_distance) || target_distance <= 0) {
    finite_low <- low_distances[is.finite(low_distances) & low_distances > 0]
    target_distance <- if (length(finite_low) == 0L) 1 else stats::median(finite_low)
  }

  total_edges <- n * length(low_cols)
  a <- rep(seq_len(n), each = length(low_cols))
  b <- as.integer(as.vector(t(low_nn$indices[, low_cols, drop = FALSE])))
  low_d <- as.numeric(as.vector(t(low_nn$distances[, low_cols, drop = FALSE])))
  valid <- is.finite(b) & b >= 1L & b <= n & b != a & is.finite(low_d)
  a <- a[valid]
  b <- b[valid]
  low_d <- low_d[valid]
  unsupported <- vapply(seq_along(a), function(ii) !(b[ii] %in% high_sets[[a[ii]]]), logical(1L))
  false_rate <- mean(unsupported)
  if (!any(unsupported)) {
    return(list(
      false_rate = false_rate,
      far_rate = 0,
      total_edges = length(a),
      unsupported_edges = 0L,
      far_edges = 0L,
      pairs = data.frame(a = integer(), b = integer(), high_ratio = numeric(), low_distance = numeric()),
      mean_high_distance_ratio = NA_real_,
      mean_low_distance = safe_mean(low_d),
      target_distance = target_distance
    ))
  }

  au <- a[unsupported]
  bu <- b[unsupported]
  low_u <- low_d[unsupported]
  high_dist <- sampled_pair_distances(x_high, au, bu)
  high_threshold <- far_multiplier * (high_radius[au] + high_radius[bu]) / 2
  high_ratio <- high_dist / pmax(high_threshold, .Machine$double.eps)
  far <- is.finite(high_ratio) & high_ratio > 1
  far_rate <- sum(far) / max(1L, length(a))
  if (any(far)) {
    pa <- pmin(au[far], bu[far])
    pb <- pmax(au[far], bu[far])
    keep <- !duplicated(paste(pa, pb, sep = ":"))
    pairs <- data.frame(
      a = as.integer(pa[keep]),
      b = as.integer(pb[keep]),
      high_ratio = as.numeric(high_ratio[far][keep]),
      low_distance = as.numeric(low_u[far][keep])
    )
    if (nrow(pairs) > max_pairs) {
      set.seed(seed + 1009L)
      pairs <- pairs[sort(sample.int(nrow(pairs), max_pairs)), , drop = FALSE]
    }
  } else {
    pairs <- data.frame(a = integer(), b = integer(), high_ratio = numeric(), low_distance = numeric())
  }
  list(
    false_rate = false_rate,
    far_rate = far_rate,
    total_edges = length(a),
    unsupported_edges = sum(unsupported),
    far_edges = sum(far),
    pairs = pairs,
    mean_high_distance_ratio = safe_mean(high_ratio),
    mean_low_distance = safe_mean(low_u),
    target_distance = target_distance
  )
}

refine_artificial_neighbors <- function(layout,
                                        pairs,
                                        target_distance,
                                        penalty_strength,
                                        n_iter,
                                        seed) {
  y <- as.matrix(layout)
  storage.mode(y) <- "double"
  if (nrow(pairs) == 0L || !is.finite(target_distance) || target_distance <= 0) {
    return(y)
  }
  penalty_strength <- max(0, min(5, as.numeric(penalty_strength)))
  n_iter <- max(1L, as.integer(n_iter))
  set.seed(seed + 3011L)
  y <- y + matrix(stats::rnorm(length(y), sd = 1e-8), nrow(y), ncol(y))
  center <- colMeans(y)
  centered <- sweep(y, 2L, center, "-")
  original_scale <- sqrt(mean(rowSums(centered * centered)))
  if (!is.finite(original_scale) || original_scale <= 0) original_scale <- 1
  a <- as.integer(pairs$a)
  b <- as.integer(pairs$b)
  weights <- pmin(4, pmax(1, as.numeric(pairs$high_ratio)))
  weights[!is.finite(weights)] <- 1
  degree <- tabulate(c(a, b), nbins = nrow(y))
  degree_scale <- pmax(1, sqrt(degree))
  for (iter in seq_len(n_iter)) {
    diff <- y[a, , drop = FALSE] - y[b, , drop = FALSE]
    dist <- sqrt(rowSums(diff * diff))
    tiny <- !is.finite(dist) | dist <= .Machine$double.eps
    if (any(tiny)) {
      diff[tiny, ] <- matrix(stats::rnorm(sum(tiny) * ncol(y), sd = 1e-6), sum(tiny), ncol(y))
      dist[tiny] <- sqrt(rowSums(diff[tiny, , drop = FALSE] * diff[tiny, , drop = FALSE]))
    }
    active <- is.finite(dist) & dist < target_distance
    if (!any(active)) break
    unit <- diff[active, , drop = FALSE] / pmax(dist[active], .Machine$double.eps)
    amount <- penalty_strength * weights[active] * (target_distance - dist[active])
    amount <- pmin(amount, target_distance * 0.25)
    delta <- unit * amount
    acc <- matrix(0, nrow(y), ncol(y))
    acc_a <- rowsum(delta, a[active], reorder = FALSE)
    acc_b <- rowsum(delta, b[active], reorder = FALSE)
    acc[as.integer(rownames(acc_a)), ] <- acc[as.integer(rownames(acc_a)), , drop = FALSE] + acc_a
    acc[as.integer(rownames(acc_b)), ] <- acc[as.integer(rownames(acc_b)), , drop = FALSE] - acc_b
    y <- y + acc / degree_scale
    y <- sweep(y, 2L, colMeans(y), "-")
    scale_now <- sqrt(mean(rowSums(y * y)))
    if (is.finite(scale_now) && scale_now > 0) {
      y <- y * (original_scale / scale_now)
    }
  }
  colnames(y) <- colnames(layout)
  y
}

run_artificial_neighbor_penalty <- function(ctx,
                                            penalty_strength,
                                            n_iter,
                                            low_k,
                                            far_multiplier) {
  base <- fastEmbedR::embed_knn(
    ctx$knn,
    method = ctx$method,
    quality = method_quality(ctx$method, "auto"),
    backend = ctx$backend,
    seed = ctx$seed
  )
  layout <- coerce_layout(base, ctx$n)
  before <- artificial_neighbor_stats(
    ctx$x,
    layout,
    ctx$knn,
    low_k = low_k,
    far_multiplier = far_multiplier,
    seed = ctx$seed
  )
  refined <- refine_artificial_neighbors(
    layout,
    before$pairs,
    before$target_distance,
    penalty_strength = penalty_strength,
    n_iter = n_iter,
    seed = ctx$seed
  )
  after <- artificial_neighbor_stats(
    ctx$x,
    refined,
    ctx$knn,
    low_k = low_k,
    far_multiplier = far_multiplier,
    seed = ctx$seed + 1L
  )
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  cfg$backend <- ctx$backend
  cfg$artificial_neighbor_penalty_enabled <- TRUE
  cfg$artificial_neighbor_transfer_mode <- artificial_neighbor_transfer_mode(ctx$method)
  cfg$artificial_neighbor_refinement_backend <- "cpu_post_embedding"
  cfg$artificial_neighbor_penalty_strength <- as.numeric(penalty_strength)
  cfg$artificial_neighbor_penalty_iterations <- as.integer(n_iter)
  cfg$artificial_neighbor_penalty_low_k <- as.integer(low_k)
  cfg$artificial_neighbor_penalty_far_multiplier <- as.numeric(far_multiplier)
  cfg$artificial_neighbor_penalty_target_distance <- safe_number(before$target_distance)
  cfg$artificial_neighbor_penalized_pairs <- nrow(before$pairs)
  cfg$artificial_neighbor_total_low_edges <- before$total_edges
  cfg$artificial_neighbor_false_rate_before <- safe_number(before$false_rate)
  cfg$artificial_neighbor_false_rate_after <- safe_number(after$false_rate)
  cfg$artificial_neighbor_false_rate_delta <- safe_number(after$false_rate - before$false_rate)
  cfg$artificial_neighbor_far_rate_before <- safe_number(before$far_rate)
  cfg$artificial_neighbor_far_rate_after <- safe_number(after$far_rate)
  cfg$artificial_neighbor_far_rate_delta <- safe_number(after$far_rate - before$far_rate)
  cfg$artificial_neighbor_mean_high_distance_ratio <- safe_number(before$mean_high_distance_ratio)
  cfg$artificial_neighbor_mean_low_distance <- safe_number(before$mean_low_distance)
  attr(refined, "fastEmbedR_config") <- cfg
  refined
}

artificial_neighbor_penalty_strategy <- function(id,
                                                 penalty_strength,
                                                 n_iter = 12L,
                                                 low_k = 15L,
                                                 far_multiplier = 1.5) {
  list(
    id = paste0("artificial_neighbor_penalty_", id),
    family = "artificial_neighbor_penalty",
    description = paste0(
      "LocalMAP-style artificial-neighbour penalty. Run the embedding, detect low-dimensional neighbours ",
      "that are not supported by high-dimensional KNN and are far in the original space, then apply a short ",
      "CPU repulsion pass with strength=", penalty_strength, "."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    context_available = function(ctx) list(
      available = ctx$n * min(as.integer(low_k), max(1L, ctx$n - 1L)) <= 2000000,
      message = "Artificial-neighbour penalty is an R-side CPU pilot and is skipped when n * low_k exceeds 2000000."
    ),
    params = function(ctx) list(
      k = ctx$k,
      low_k = as.integer(low_k),
      far_multiplier = as.numeric(far_multiplier),
      penalty_strength = as.numeric(penalty_strength),
      penalty_iterations = as.integer(n_iter),
      refinement_backend = "cpu_post_embedding",
      original_method = "LocalMAP",
      primary_methods = "localmap",
      transferred_methods = "umap,tsne,pacmap,trimap",
      metric = "trustworthiness,false_neighbor_rate",
      caveat = "post_embedding_refinement_not_native_objective_yet"
    ),
    run = function(ctx) run_artificial_neighbor_penalty(
      ctx,
      penalty_strength = penalty_strength,
      n_iter = n_iter,
      low_k = low_k,
      far_multiplier = far_multiplier
    ),
    artificial_neighbor_penalty_strength = as.numeric(penalty_strength),
    artificial_neighbor_penalty_iterations = as.integer(n_iter),
    artificial_neighbor_penalty_low_k = as.integer(low_k),
    artificial_neighbor_penalty_far_multiplier = as.numeric(far_multiplier)
  )
}

artificial_neighbor_penalty_strategy_grid <- function() {
  list(
    artificial_neighbor_penalty_strategy("mild_s0p15_f1p25_i8", penalty_strength = 0.15, n_iter = 8L, far_multiplier = 1.25),
    artificial_neighbor_penalty_strategy("balanced_s0p35_f1p00_i12", penalty_strength = 0.35, n_iter = 12L, far_multiplier = 1.00),
    artificial_neighbor_penalty_strategy("strong_s0p70_f0p85_i16", penalty_strength = 0.70, n_iter = 16L, far_multiplier = 0.85)
  )
}

false_neighbor_monitor_transfer_mode <- function(method) {
  switch(
    safe_character(method, "localmap"),
    umap = "umap_chunked_optimizer_false_neighbour_monitor",
    tsne = "tsne_chunked_optimizer_false_neighbour_monitor",
    pacmap = "pacmap_chunked_optimizer_false_neighbour_monitor",
    trimap = "trimap_chunked_optimizer_false_neighbour_monitor",
    localmap = "localmap_native_concept_chunked_false_neighbour_monitor",
    "chunked_false_neighbour_monitor"
  )
}

false_neighbor_monitor_score <- function(stats, far_weight = 2) {
  false_rate <- safe_number(stats$false_rate)
  far_rate <- safe_number(stats$far_rate)
  if (!is.finite(false_rate) && !is.finite(far_rate)) return(Inf)
  if (!is.finite(false_rate)) false_rate <- 0
  if (!is.finite(far_rate)) far_rate <- 0
  false_rate + as.numeric(far_weight) * far_rate
}

initial_false_neighbor_monitor_layout <- function(ctx, n_epochs) {
  if (identical(ctx$method, "umap")) {
    return(fastEmbedR:::fast_knn_umap_core(
      ctx$knn,
      backend = "cpu",
      seed = ctx$seed,
      n_epochs = n_epochs
    ))
  }
  fastEmbedR:::knn_embed_core(
    ctx$knn,
    objective = ctx$method,
    quality = method_quality(ctx$method, "fast"),
    backend = "cpu",
    seed = ctx$seed,
    n_epochs = n_epochs
  )
}

refine_false_neighbor_monitor_chunk <- function(ctx, layout, n_epochs, chunk_id) {
  fastEmbedR:::refine_embedding_from_knn(
    method = ctx$method,
    indices = ctx$knn$indices,
    distances = ctx$knn$distances,
    init_layout = layout,
    n_epochs = as.integer(n_epochs),
    refinement = "false_neighbor_monitor",
    seed = as.integer(ctx$seed + 7919L * chunk_id),
    backend = "cpu",
    verbose = FALSE
  )
}

run_false_neighbor_monitor <- function(ctx,
                                       chunk_epochs,
                                       max_chunks,
                                       tolerance,
                                       patience,
                                       action,
                                       start_mode = c("short_chunk", "full_auto"),
                                       low_k = 15L,
                                       far_multiplier = 1.25,
                                       far_weight = 2) {
  chunk_epochs <- max(5L, as.integer(chunk_epochs))
  next_chunk_epochs <- chunk_epochs
  max_chunks <- max(1L, as.integer(max_chunks))
  patience <- max(1L, as.integer(patience))
  tolerance <- max(0, as.numeric(tolerance))
  action <- match.arg(action, c("early_stop", "shrink_chunk"))
  start_mode <- match.arg(start_mode)
  low_k <- max(1L, as.integer(low_k))

  if (identical(start_mode, "full_auto")) {
    layout <- coerce_layout(fastEmbedR::embed_knn(
      ctx$knn,
      method = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = "cpu",
      seed = ctx$seed
    ), ctx$n)
    chunks_run <- 0L
    epochs_completed <- 0L
    chunk_ids <- seq_len(max_chunks)
    chunk_trace <- 0L
  } else {
    layout <- coerce_layout(initial_false_neighbor_monitor_layout(ctx, chunk_epochs), ctx$n)
    chunks_run <- 1L
    epochs_completed <- chunk_epochs
    chunk_ids <- if (max_chunks > 1L) seq.int(2L, max_chunks) else integer()
    chunk_trace <- chunk_epochs
  }
  current <- artificial_neighbor_stats(
    ctx$x,
    layout,
    ctx$knn,
    low_k = low_k,
    far_multiplier = far_multiplier,
    seed = ctx$seed
  )
  current_score <- false_neighbor_monitor_score(current, far_weight = far_weight)
  best_layout <- layout
  best <- current
  best_score <- current_score
  initial <- current
  initial_score <- current_score
  worsening_events <- 0L
  consecutive_worsening <- 0L
  adjustments <- 0L
  stopped_early <- FALSE
  score_trace <- current_score
  false_trace <- safe_number(current$false_rate)
  far_trace <- safe_number(current$far_rate)

  if (length(chunk_ids) > 0L) {
    for (chunk_id in chunk_ids) {
      candidate <- coerce_layout(
        refine_false_neighbor_monitor_chunk(ctx, layout, next_chunk_epochs, chunk_id),
        ctx$n
      )
      candidate_stats <- artificial_neighbor_stats(
        ctx$x,
        candidate,
        ctx$knn,
        low_k = low_k,
        far_multiplier = far_multiplier,
        seed = ctx$seed + chunk_id
      )
      candidate_score <- false_neighbor_monitor_score(candidate_stats, far_weight = far_weight)
      false_worse <- is.finite(candidate_stats$false_rate) &&
        is.finite(current$false_rate) &&
        candidate_stats$false_rate > current$false_rate + tolerance
      score_worse <- is.finite(candidate_score) &&
        is.finite(current_score) &&
        candidate_score > current_score + tolerance
      worsened <- isTRUE(false_worse || score_worse)

      chunks_run <- chunks_run + 1L
      epochs_completed <- epochs_completed + next_chunk_epochs
      score_trace <- c(score_trace, candidate_score)
      false_trace <- c(false_trace, safe_number(candidate_stats$false_rate))
      far_trace <- c(far_trace, safe_number(candidate_stats$far_rate))
      chunk_trace <- c(chunk_trace, next_chunk_epochs)

      if (worsened) {
        worsening_events <- worsening_events + 1L
        consecutive_worsening <- consecutive_worsening + 1L
        if (identical(action, "early_stop") && consecutive_worsening >= patience) {
          stopped_early <- TRUE
          break
        }
        if (identical(action, "shrink_chunk")) {
          adjustments <- adjustments + 1L
          next_chunk_epochs <- max(5L, as.integer(ceiling(next_chunk_epochs / 2)))
          layout <- best_layout
          current <- best
          current_score <- best_score
          next
        }
      } else {
        consecutive_worsening <- 0L
      }

      layout <- candidate
      current <- candidate_stats
      current_score <- candidate_score
      if (is.finite(candidate_score) && candidate_score < best_score) {
        best_layout <- candidate
        best <- candidate_stats
        best_score <- candidate_score
      }
    }
  }

  final <- artificial_neighbor_stats(
    ctx$x,
    best_layout,
    ctx$knn,
    low_k = low_k,
    far_multiplier = far_multiplier,
    seed = ctx$seed + 997L
  )
  final_score <- false_neighbor_monitor_score(final, far_weight = far_weight)

  cfg <- attr(best_layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  cfg$backend <- "cpu"
  cfg$false_neighbor_monitor_enabled <- TRUE
  cfg$false_neighbor_monitor_transfer_mode <- false_neighbor_monitor_transfer_mode(ctx$method)
  cfg$false_neighbor_monitor_backend <- "cpu_chunk_monitor"
  cfg$false_neighbor_monitor_action <- action
  cfg$false_neighbor_monitor_start_mode <- start_mode
  cfg$false_neighbor_monitor_chunk_epochs <- as.integer(chunk_epochs)
  cfg$false_neighbor_monitor_max_chunks <- as.integer(max_chunks)
  cfg$false_neighbor_monitor_chunks_run <- as.integer(chunks_run)
  cfg$false_neighbor_monitor_epochs_requested <- as.integer(chunk_epochs * max_chunks)
  cfg$false_neighbor_monitor_epochs_completed <- as.integer(epochs_completed)
  cfg$false_neighbor_monitor_patience <- as.integer(patience)
  cfg$false_neighbor_monitor_tolerance <- as.numeric(tolerance)
  cfg$false_neighbor_monitor_low_k <- as.integer(low_k)
  cfg$false_neighbor_monitor_far_multiplier <- as.numeric(far_multiplier)
  cfg$false_neighbor_monitor_far_weight <- as.numeric(far_weight)
  cfg$false_neighbor_monitor_initial_false_rate <- safe_number(initial$false_rate)
  cfg$false_neighbor_monitor_final_false_rate <- safe_number(final$false_rate)
  cfg$false_neighbor_monitor_best_false_rate <- safe_number(best$false_rate)
  cfg$false_neighbor_monitor_false_rate_delta <- safe_number(final$false_rate - initial$false_rate)
  cfg$false_neighbor_monitor_initial_far_rate <- safe_number(initial$far_rate)
  cfg$false_neighbor_monitor_final_far_rate <- safe_number(final$far_rate)
  cfg$false_neighbor_monitor_best_far_rate <- safe_number(best$far_rate)
  cfg$false_neighbor_monitor_far_rate_delta <- safe_number(final$far_rate - initial$far_rate)
  cfg$false_neighbor_monitor_score_initial <- safe_number(initial_score)
  cfg$false_neighbor_monitor_score_final <- safe_number(final_score)
  cfg$false_neighbor_monitor_score_best <- safe_number(best_score)
  cfg$false_neighbor_monitor_score_delta <- safe_number(final_score - initial_score)
  cfg$false_neighbor_monitor_worsening_events <- as.integer(worsening_events)
  cfg$false_neighbor_monitor_adjustments <- as.integer(adjustments)
  cfg$false_neighbor_monitor_stopped_early <- isTRUE(stopped_early)
  cfg$false_neighbor_monitor_score_trace <- paste(round(score_trace, 6), collapse = ",")
  cfg$false_neighbor_monitor_false_rate_trace <- paste(round(false_trace, 6), collapse = ",")
  cfg$false_neighbor_monitor_far_rate_trace <- paste(round(far_trace, 6), collapse = ",")
  cfg$false_neighbor_monitor_chunk_trace <- paste(chunk_trace, collapse = ",")
  attr(best_layout, "fastEmbedR_config") <- cfg
  best_layout
}

false_neighbor_monitor_strategy <- function(id,
                                            chunk_epochs = 20L,
                                            max_chunks = 5L,
                                            tolerance = 0.001,
                                            patience = 1L,
                                            action = c("early_stop", "shrink_chunk"),
                                            start_mode = c("short_chunk", "full_auto"),
                                            low_k = 15L,
                                            far_multiplier = 1.25,
                                            far_weight = 2) {
  action <- match.arg(action)
  start_mode <- match.arg(start_mode)
  list(
    id = paste0("false_neighbor_monitor_", id),
    family = "false_neighbor_monitor",
    description = paste0(
      "Chunked optimizer monitor. Start from ", start_mode, ", run ", chunk_epochs,
      "-epoch chunks, measure low-dimensional false-neighbour rate after each chunk, and ",
      gsub("_", " ", action), " when the rate worsens."
    ),
    compatible = function(method, backend) {
      method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu")
    },
    context_available = function(ctx) list(
      available = ctx$n * min(as.integer(low_k), max(1L, ctx$n - 1L)) <= 2000000,
      message = "False-neighbour monitor is an R-side CPU pilot and is skipped when n * low_k exceeds 2000000."
    ),
    params = function(ctx) list(
      k = ctx$k,
      chunk_epochs = as.integer(chunk_epochs),
      max_chunks = as.integer(max_chunks),
      epochs_requested = as.integer(chunk_epochs * max_chunks),
      tolerance = as.numeric(tolerance),
      patience = as.integer(patience),
      action = action,
      start_mode = start_mode,
      low_k = as.integer(low_k),
      far_multiplier = as.numeric(far_multiplier),
      far_weight = as.numeric(far_weight),
      monitor_backend = "cpu_chunk_monitor",
      original_method = "LocalMAP false-neighbour diagnostic",
      transferred_methods = "umap,tsne,pacmap,trimap,localmap",
      caveat = "chunked_cpu_refinement_monitor_not_native_gpu_optimizer_yet"
    ),
    run = function(ctx) run_false_neighbor_monitor(
      ctx,
      chunk_epochs = chunk_epochs,
      max_chunks = max_chunks,
      tolerance = tolerance,
      patience = patience,
      action = action,
      start_mode = start_mode,
      low_k = low_k,
      far_multiplier = far_multiplier,
      far_weight = far_weight
    ),
    false_neighbor_monitor_action = action,
    false_neighbor_monitor_start_mode = start_mode,
    false_neighbor_monitor_chunk_epochs = as.integer(chunk_epochs),
    false_neighbor_monitor_max_chunks = as.integer(max_chunks),
    false_neighbor_monitor_tolerance = as.numeric(tolerance),
    false_neighbor_monitor_patience = as.integer(patience),
    false_neighbor_monitor_low_k = as.integer(low_k),
    false_neighbor_monitor_far_multiplier = as.numeric(far_multiplier),
    false_neighbor_monitor_far_weight = as.numeric(far_weight)
  )
}

false_neighbor_monitor_strategy_grid <- function() {
  list(
    false_neighbor_monitor_strategy("early_stop_c20_p1", chunk_epochs = 20L, max_chunks = 5L, tolerance = 0.001, patience = 1L, action = "early_stop"),
    false_neighbor_monitor_strategy("shrink_c20_p1", chunk_epochs = 20L, max_chunks = 5L, tolerance = 0.001, patience = 1L, action = "shrink_chunk"),
    false_neighbor_monitor_strategy("conservative_c15_p1", chunk_epochs = 15L, max_chunks = 6L, tolerance = 0, patience = 1L, action = "early_stop"),
    false_neighbor_monitor_strategy("guarded_refine_c20_p1", chunk_epochs = 20L, max_chunks = 4L, tolerance = 0.001, patience = 1L, action = "early_stop", start_mode = "full_auto"),
    false_neighbor_monitor_strategy("guarded_shrink_c20_p1", chunk_epochs = 20L, max_chunks = 4L, tolerance = 0.001, patience = 1L, action = "shrink_chunk", start_mode = "full_auto")
  )
}

landmark_count_for_strategy <- function(ctx, fraction = NULL, count = NULL) {
  n <- as.integer(ctx$n)
  if (!is.null(count)) {
    out <- as.integer(round(as.numeric(count)))
  } else if (!is.null(fraction)) {
    out <- as.integer(ceiling(n * as.numeric(fraction)))
  } else {
    out <- fastEmbedR:::auto_landmark_count(n)
  }
  as.integer(max(2L, min(out, n - 1L)))
}

landmark_selection_cache <- new.env(parent = emptyenv())

landmark_selection_cache_key <- function(ctx,
                                         selection,
                                         count,
                                         density_alpha = NA_real_,
                                         density_k = NULL,
                                         diversity_beta = NA_real_,
                                         stratified_allocation = NULL,
                                         rare_tail_fraction = NA_real_,
                                         rare_tail_oversample = NA_real_,
                                         rare_cluster_fraction = NA_real_,
                                         rare_n_quantiles = NA_integer_) {
  paste(
    ctx$dataset_name,
    paste0("n", ctx$n),
    paste0("p", ctx$p),
    paste0("seed", ctx$seed),
    paste0("k", ctx$k),
    selection,
    paste0("count", count),
    paste0("density_alpha", ifelse(is.finite(density_alpha), density_alpha, "NA")),
    paste0("density_k", if (is.null(density_k)) "NA" else density_k),
    paste0("diversity_beta", ifelse(is.finite(diversity_beta), diversity_beta, "NA")),
    paste0("stratified_allocation", if (is.null(stratified_allocation)) "NA" else stratified_allocation),
    paste0("rare_tail_fraction", ifelse(is.finite(rare_tail_fraction), rare_tail_fraction, "NA")),
    paste0("rare_tail_oversample", ifelse(is.finite(rare_tail_oversample), rare_tail_oversample, "NA")),
    paste0("rare_cluster_fraction", ifelse(is.finite(rare_cluster_fraction), rare_cluster_fraction, "NA")),
    paste0("rare_n_quantiles", ifelse(is.finite(rare_n_quantiles), rare_n_quantiles, "NA")),
    sep = "|"
  )
}

stratified_subsample_indices <- function(labels,
                                         count,
                                         seed,
                                         source = "label",
                                         allocation = c("proportional", "balanced"),
                                         setup_time_sec = 0,
                                         cluster_k = NA_integer_,
                                         cluster_feature_dims = NA_integer_) {
  allocation <- match.arg(allocation)
  labels <- as.factor(labels)
  n <- length(labels)
  count <- max(2L, min(as.integer(count), n - 1L))
  if (count >= n) return(seq_len(n))
  set.seed(seed + 4201L)
  all_levels <- levels(labels)
  all_groups <- split(seq_len(n), labels)
  all_groups <- all_groups[all_levels]
  all_group_sizes <- vapply(all_groups, length, integer(1L))
  nonempty <- all_group_sizes > 0L
  groups <- all_groups[nonempty]
  group_sizes <- all_group_sizes[nonempty]
  if (length(groups) > count) {
    keep <- sample(names(groups), count, replace = FALSE, prob = group_sizes)
    groups <- groups[keep]
    group_sizes <- group_sizes[keep]
  }
  strata <- names(groups)
  n_strata <- length(groups)
  if (identical(allocation, "balanced")) {
    raw <- rep(max(1L, floor(count / n_strata)), n_strata)
    names(raw) <- strata
  } else {
    raw <- pmax(1L, floor(count * group_sizes / sum(group_sizes)))
  }
  raw <- pmin(raw, group_sizes)
  names(raw) <- strata
  while (sum(raw) < count) {
    room <- group_sizes - raw
    if (!any(room > 0L)) break
    score <- if (identical(allocation, "balanced")) {
      room / pmax(1L, raw)
    } else {
      group_sizes / pmax(1L, raw)
    }
    score[room <= 0L] <- -Inf
    raw[which.max(score)] <- raw[which.max(score)] + 1L
  }
  while (sum(raw) > count) {
    can_drop <- raw > 1L
    if (!any(can_drop)) break
    score <- if (identical(allocation, "balanced")) raw else raw / pmax(1L, group_sizes)
    score[!can_drop] <- -Inf
    raw[which.max(score)] <- raw[which.max(score)] - 1L
  }
  picked <- unlist(mapply(function(rows, m) {
    sort(sample(rows, min(length(rows), m)))
  }, groups[strata], raw[strata], SIMPLIFY = FALSE), use.names = FALSE)
  picked <- sort(unique(as.integer(picked)))
  if (length(picked) < count) {
    remaining <- setdiff(seq_len(n), picked)
    picked <- sort(c(picked, sample(remaining, count - length(picked))))
  }
  picked <- picked[seq_len(min(length(picked), count))]
  selected_counts <- table(factor(labels[picked], levels = all_levels))
  nonzero_selected <- as.integer(selected_counts[selected_counts > 0L])
  attr(picked, "stratified_landmark_source") <- source
  attr(picked, "stratified_landmark_allocation") <- allocation
  attr(picked, "stratified_landmark_time_sec") <- as.numeric(setup_time_sec)
  attr(picked, "stratified_landmark_n_strata") <- length(all_levels)
  attr(picked, "stratified_landmark_strata_sampled") <- sum(selected_counts > 0L)
  attr(picked, "stratified_landmark_missing_strata") <- sum(selected_counts == 0L)
  attr(picked, "stratified_landmark_min_stratum_size") <- min(all_group_sizes)
  attr(picked, "stratified_landmark_max_stratum_size") <- max(all_group_sizes)
  attr(picked, "stratified_landmark_min_selected_per_stratum") <- if (length(nonzero_selected)) min(nonzero_selected) else NA_real_
  attr(picked, "stratified_landmark_max_selected_per_stratum") <- if (length(nonzero_selected)) max(nonzero_selected) else NA_real_
  attr(picked, "stratified_landmark_balance_ratio") <- if (length(nonzero_selected)) {
    min(nonzero_selected) / max(nonzero_selected)
  } else {
    NA_real_
  }
  attr(picked, "stratified_landmark_cluster_k") <- cluster_k
  attr(picked, "stratified_landmark_cluster_feature_dims") <- cluster_feature_dims
  picked
}

stratified_cluster_assignments <- function(ctx, count, seed) {
  n <- as.integer(ctx$n)
  target_k <- min(50L, max(2L, as.integer(round(sqrt(max(2L, count))))))
  target_k <- min(target_k, n - 1L)
  elapsed <- system.time({
    z <- diversity_landmark_features(ctx, seed)
    z <- z[, seq_len(min(16L, ncol(z))), drop = FALSE]
    z <- scale(z, center = TRUE, scale = TRUE)
    z[!is.finite(z)] <- 0
    set.seed(seed + 4301L)
    init <- sample.int(n, target_k)
    clusters <- tryCatch(
      suppressWarnings(stats::kmeans(z, centers = z[init, , drop = FALSE], iter.max = 80L, algorithm = "Lloyd")$cluster),
      error = function(e) {
        # Deterministic fallback keeps the benchmark row explicit instead of pretending labels existed.
        fallback <- integer(n)
        fallback[order(rowSums(z * z), seq_len(n))] <- (seq_len(n) - 1L) %% target_k + 1L
        fallback
      }
    )
  })[["elapsed"]]
  list(
    labels = factor(clusters, levels = seq_len(target_k)),
    time_sec = as.numeric(elapsed),
    cluster_k = target_k,
    feature_dims = ncol(z)
  )
}

density_landmark_profile_cache <- new.env(parent = emptyenv())

density_landmark_profile_key <- function(ctx, alpha, density_k) {
  paste(
    ctx$dataset_name,
    paste0("n", ctx$n),
    paste0("p", ctx$p),
    paste0("k", ctx$k),
    paste0("alpha", alpha),
    paste0("density_k", density_k),
    sep = "|"
  )
}

density_landmark_profile <- function(ctx, alpha = 1, density_k = NULL) {
  n <- as.integer(ctx$n)
  alpha <- as.numeric(alpha)[1L]
  if (!is.finite(alpha) || alpha < 0) alpha <- 1
  density_k <- if (is.null(density_k)) ctx$k else as.integer(density_k)
  density_k <- max(1L, min(as.integer(density_k), n - 1L))
  cache_key <- density_landmark_profile_key(ctx, alpha, density_k)
  if (exists(cache_key, envir = density_landmark_profile_cache, inherits = FALSE)) {
    return(get(cache_key, envir = density_landmark_profile_cache, inherits = FALSE))
  }
  weights <- rep(1, n)
  mean_distance <- rep(NA_real_, n)
  density_time <- 0
  if (alpha > 0) {
    density_time <- system.time({
      raw <- tryCatch(
        fastEmbedR::nn(ctx$x, k = density_k + 1L, backend = "cpu"),
        error = function(e) fastEmbedR::nn(ctx$x, ctx$x, density_k + 1L, backend = "cpu")
      )
      distances <- as.matrix(raw$distances)
      if (ncol(distances) > density_k) {
        distances <- distances[, -1L, drop = FALSE]
      }
      distances <- distances[, seq_len(min(density_k, ncol(distances))), drop = FALSE]
      distances[!is.finite(distances)] <- NA_real_
      mean_distance <- rowMeans(distances, na.rm = TRUE)
      valid <- is.finite(mean_distance) & mean_distance > 0
      fallback <- if (any(valid)) stats::median(mean_distance[valid]) else 1
      mean_distance[!valid] <- fallback
      density <- 1 / pmax(mean_distance, .Machine$double.eps)
      weights <- density ^ alpha
    })[["elapsed"]]
  }
  weights[!is.finite(weights) | weights < 0] <- 0
  if (!any(weights > 0)) weights[] <- 1
  out <- list(
    alpha = alpha,
    density_k = density_k,
    weights = weights,
    mean_distance = mean_distance,
    time_sec = as.numeric(density_time)
  )
  assign(cache_key, out, envir = density_landmark_profile_cache)
  out
}

density_weighted_subsample_indices <- function(ctx, count, seed, alpha = 1, density_k = NULL) {
  n <- as.integer(ctx$n)
  count <- max(2L, min(as.integer(count), n - 1L))
  if (count >= n) return(seq_len(n))
  profile <- density_landmark_profile(ctx, alpha = alpha, density_k = density_k)
  weights <- profile$weights
  set.seed(seed + 5101L + as.integer(round(profile$alpha * 100)))
  idx <- sort(sample.int(n, count, replace = FALSE, prob = weights))
  attr(idx, "density_landmark_alpha") <- profile$alpha
  attr(idx, "density_landmark_k") <- profile$density_k
  attr(idx, "density_landmark_time_sec") <- profile$time_sec
  attr(idx, "density_landmark_weight_min") <- min(weights)
  attr(idx, "density_landmark_weight_median") <- stats::median(weights)
  attr(idx, "density_landmark_weight_max") <- max(weights)
  attr(idx, "density_landmark_weight_mean") <- mean(weights)
  attr(idx, "density_landmark_selected_weight_mean") <- mean(weights[idx])
  attr(idx, "density_landmark_selected_to_global_weight_ratio") <- mean(weights[idx]) / mean(weights)
  attr(idx, "density_landmark_mean_distance_median") <- stats::median(profile$mean_distance, na.rm = TRUE)
  idx
}

diversity_landmark_features <- function(ctx, seed) {
  z <- tryCatch(
    fastEmbedR:::landmark_selection_features(ctx$x, seed),
    error = function(e) {
      x <- as.matrix(ctx$x)
      x[, seq_len(min(8L, ncol(x))), drop = FALSE]
    }
  )
  z <- as.matrix(z)
  storage.mode(z) <- "double"
  z
}

diversity_whiten_features <- function(z) {
  z <- scale(z, center = TRUE, scale = FALSE)
  z[!is.finite(z)] <- 0
  p <- ncol(z)
  gram <- crossprod(z) / max(1L, nrow(z) - 1L)
  ridge <- 1e-8 * max(1, mean(diag(gram)))
  sv <- svd(gram + diag(ridge, p))
  keep <- is.finite(sv$d) & sv$d > ridge
  if (!any(keep)) return(z)
  z %*% sv$v[, keep, drop = FALSE] %*% diag(1 / sqrt(sv$d[keep]), nrow = sum(keep))
}

diversity_select_from_features <- function(z, count, seed, algorithm) {
  n <- nrow(z)
  count <- as.integer(min(max(2L, count), n - 1L))
  z_work <- if (identical(algorithm, "dpp_approx")) diversity_whiten_features(z) else z
  z_norm <- rowSums(z_work * z_work)
  selected <- integer(count)
  available <- rep(TRUE, n)
  min_dist <- rep(Inf, n)
  leverage <- rep(1, n)
  if (identical(algorithm, "dpp_approx")) {
    leverage <- pmax(rowSums(z_work * z_work), 0)
    if (!any(is.finite(leverage) & leverage > 0)) leverage[] <- 1
  }
  set.seed(seed + switch(algorithm, farthest_point = 6101L, kmeanspp = 6201L, dpp_approx = 6301L, 6401L))
  selected[1L] <- if (identical(algorithm, "farthest_point")) {
    which.min(z_norm)
  } else if (identical(algorithm, "dpp_approx")) {
    sample.int(n, 1L, prob = leverage)
  } else {
    sample.int(n, 1L)
  }
  for (i in seq_len(count)) {
    if (i > 1L) {
      scores <- min_dist
      scores[!available] <- 0
      scores[!is.finite(scores) | scores < 0] <- 0
      selected[i] <- if (identical(algorithm, "farthest_point")) {
        scores[!available] <- -Inf
        which.max(scores)
      } else {
        if (identical(algorithm, "dpp_approx")) {
          scores <- scores * pmax(leverage, .Machine$double.eps)
        }
        if (!any(scores > 0)) scores <- as.numeric(available)
        sample.int(n, 1L, prob = scores)
      }
    }
    available[selected[i]] <- FALSE
    center <- z_work[selected[i], , drop = FALSE]
    dist <- z_norm + z_norm[selected[i]] - 2 * drop(z_work %*% t(center))
    min_dist <- pmin(min_dist, pmax(0, dist))
    min_dist[!available] <- 0
  }
  cover <- sqrt(pmax(0, min_dist))
  list(
    indices = sort(unique(as.integer(selected))),
    feature_dims = ncol(z_work),
    cover_mean = mean(cover),
    cover_median = stats::median(cover),
    cover_max = max(cover),
    leverage_selected_to_global_ratio = mean(leverage[selected]) / mean(leverage)
  )
}

diversity_landmark_indices <- function(ctx, count, seed, algorithm) {
  n <- as.integer(ctx$n)
  count <- max(2L, min(as.integer(count), n - 1L))
  if (count >= n) return(seq_len(n))
  elapsed <- system.time({
    z <- diversity_landmark_features(ctx, seed)
    selected <- diversity_select_from_features(z, count, seed, algorithm)
  })[["elapsed"]]
  idx <- selected$indices
  if (length(idx) < count) {
    idx <- sort(c(idx, sample(setdiff(seq_len(n), idx), count - length(idx))))
  }
  idx <- sort(idx[seq_len(count)])
  attr(idx, "diversity_landmark_algorithm") <- algorithm
  attr(idx, "diversity_landmark_time_sec") <- as.numeric(elapsed)
  attr(idx, "diversity_landmark_feature_dims") <- selected$feature_dims
  attr(idx, "diversity_landmark_cover_mean") <- selected$cover_mean
  attr(idx, "diversity_landmark_cover_median") <- selected$cover_median
  attr(idx, "diversity_landmark_cover_max") <- selected$cover_max
  attr(idx, "diversity_landmark_leverage_selected_to_global_ratio") <- selected$leverage_selected_to_global_ratio
  idx
}

hybrid_density_diversity_indices <- function(ctx,
                                             count,
                                             seed,
                                             alpha = 1,
                                             beta = 1,
                                             density_k = NULL) {
  n <- as.integer(ctx$n)
  count <- max(2L, min(as.integer(count), n - 1L))
  if (count >= n) return(seq_len(n))
  alpha <- as.numeric(alpha)[1L]
  beta <- as.numeric(beta)[1L]
  if (!is.finite(alpha) || alpha < 0) alpha <- 1
  if (!is.finite(beta) || beta < 0) beta <- 1
  elapsed <- system.time({
    density <- density_landmark_profile(ctx, alpha = alpha, density_k = density_k)
    z <- diversity_landmark_features(ctx, seed)
    z[!is.finite(z)] <- 0
    z_norm <- rowSums(z * z)
    base_weight <- density$weights
    base_weight[!is.finite(base_weight) | base_weight < 0] <- 0
    if (!any(base_weight > 0)) base_weight[] <- 1
    selected <- integer(count)
    available <- rep(TRUE, n)
    min_dist <- rep(Inf, n)
    set.seed(seed + 7101L + as.integer(round(alpha * 100)) + as.integer(round(beta * 1000)))
    for (i in seq_len(count)) {
      if (i == 1L || beta == 0) {
        diversity_weight <- rep(1, n)
      } else {
        diversity_weight <- sqrt(pmax(min_dist, .Machine$double.eps)) ^ beta
      }
      score <- base_weight * diversity_weight
      score[!available] <- 0
      score[!is.finite(score) | score < 0] <- 0
      if (!any(score > 0)) score <- as.numeric(available)
      selected[i] <- sample.int(n, 1L, prob = score)
      available[selected[i]] <- FALSE
      center <- z[selected[i], , drop = FALSE]
      dist <- z_norm + z_norm[selected[i]] - 2 * drop(z %*% t(center))
      min_dist <- pmin(min_dist, pmax(0, dist))
      min_dist[!available] <- 0
    }
    cover <- sqrt(pmax(0, min_dist))
  })[["elapsed"]]
  idx <- sort(unique(as.integer(selected)))
  if (length(idx) < count) {
    idx <- sort(c(idx, sample(setdiff(seq_len(n), idx), count - length(idx))))
  }
  idx <- sort(idx[seq_len(count)])
  attr(idx, "hybrid_landmark_alpha") <- alpha
  attr(idx, "hybrid_landmark_beta") <- beta
  attr(idx, "hybrid_landmark_k") <- density$density_k
  attr(idx, "hybrid_landmark_time_sec") <- as.numeric(elapsed)
  attr(idx, "hybrid_landmark_density_time_sec") <- density$time_sec
  attr(idx, "hybrid_landmark_feature_dims") <- ncol(z)
  attr(idx, "hybrid_landmark_formula") <- "density^alpha * diversity^beta"
  attr(idx, "hybrid_landmark_density_weight_median") <- stats::median(base_weight)
  attr(idx, "hybrid_landmark_density_selected_to_global_weight_ratio") <- mean(base_weight[idx]) / mean(base_weight)
  attr(idx, "hybrid_landmark_mean_distance_median") <- stats::median(density$mean_distance, na.rm = TRUE)
  attr(idx, "hybrid_landmark_cover_mean") <- mean(cover)
  attr(idx, "hybrid_landmark_cover_median") <- stats::median(cover)
  attr(idx, "hybrid_landmark_cover_max") <- max(cover)
  attr(idx, "density_landmark_alpha") <- alpha
  attr(idx, "density_landmark_k") <- density$density_k
  attr(idx, "density_landmark_time_sec") <- density$time_sec
  attr(idx, "density_landmark_weight_min") <- min(base_weight)
  attr(idx, "density_landmark_weight_median") <- stats::median(base_weight)
  attr(idx, "density_landmark_weight_max") <- max(base_weight)
  attr(idx, "density_landmark_weight_mean") <- mean(base_weight)
  attr(idx, "density_landmark_selected_weight_mean") <- mean(base_weight[idx])
  attr(idx, "density_landmark_selected_to_global_weight_ratio") <- mean(base_weight[idx]) / mean(base_weight)
  attr(idx, "density_landmark_mean_distance_median") <- stats::median(density$mean_distance, na.rm = TRUE)
  idx
}

rare_protected_allocate_counts <- function(capacity, total, weights) {
  capacity <- pmax(0L, as.integer(capacity))
  total <- max(0L, min(as.integer(total), sum(capacity)))
  allocation <- integer(length(capacity))
  if (total <= 0L || !length(capacity) || !any(capacity > 0L)) return(allocation)
  weights <- as.numeric(weights)
  weights[!is.finite(weights) | weights <= 0] <- 0
  active <- capacity > 0L
  if (!any(weights[active] > 0)) weights[active] <- 1
  target <- numeric(length(capacity))
  target[active] <- total * weights[active] / sum(weights[active])
  if (total >= sum(active)) {
    allocation[active] <- pmin(capacity[active], pmax(1L, floor(target[active])))
  }
  while (sum(allocation) > total) {
    can_drop <- allocation > 0L
    if (!any(can_drop)) break
    score <- allocation
    score[!can_drop] <- -Inf
    allocation[which.max(score)] <- allocation[which.max(score)] - 1L
  }
  while (sum(allocation) < total) {
    room <- capacity - allocation
    if (!any(room > 0L)) break
    score <- target - allocation
    score[room <= 0L] <- -Inf
    if (!any(is.finite(score))) {
      score <- room
      score[room <= 0L] <- -Inf
    }
    allocation[which.max(score)] <- allocation[which.max(score)] + 1L
  }
  pmin(allocation, capacity)
}

rare_protected_weighted_sample <- function(candidates, count, weights = NULL) {
  candidates <- sort(unique(as.integer(candidates)))
  candidates <- candidates[is.finite(candidates)]
  count <- max(0L, min(as.integer(count), length(candidates)))
  if (count <= 0L || !length(candidates)) return(integer())
  prob <- NULL
  if (!is.null(weights)) {
    prob <- as.numeric(weights[candidates])
    prob[!is.finite(prob) | prob < 0] <- 0
    if (!any(prob > 0)) prob <- NULL
  }
  sort(sample(candidates, count, replace = FALSE, prob = prob))
}

rare_protected_cluster_sample <- function(labels, count, available) {
  labels <- as.factor(labels)
  rows <- which(available)
  count <- max(0L, min(as.integer(count), length(rows)))
  if (count <= 0L || !length(rows)) return(integer())
  groups <- split(rows, labels[rows], drop = TRUE)
  groups <- groups[vapply(groups, length, integer(1L)) > 0L]
  if (!length(groups)) return(integer())
  sizes <- vapply(groups, length, integer(1L))
  if (length(groups) > count) {
    keep <- sample(names(groups), count, replace = FALSE, prob = 1 / sqrt(pmax(1L, sizes)))
    groups <- groups[keep]
    sizes <- sizes[keep]
  }
  alloc <- rare_protected_allocate_counts(sizes, count, rep(1, length(groups)))
  picked <- unlist(mapply(function(rows_i, m_i) {
    rare_protected_weighted_sample(rows_i, m_i)
  }, groups, alloc, SIMPLIFY = FALSE), use.names = FALSE)
  sort(unique(as.integer(picked)))
}

rare_protected_landmark_indices <- function(ctx,
                                            count,
                                            seed,
                                            tail_fraction = 0.10,
                                            tail_oversample = 0.40,
                                            n_quantiles = 5L,
                                            cluster_fraction = 0.25,
                                            density_k = NULL) {
  n <- as.integer(ctx$n)
  count <- max(2L, min(as.integer(count), n - 1L))
  tail_fraction <- as.numeric(tail_fraction)[1L]
  tail_oversample <- as.numeric(tail_oversample)[1L]
  cluster_fraction <- as.numeric(cluster_fraction)[1L]
  n_quantiles <- as.integer(n_quantiles)[1L]
  if (!is.finite(n_quantiles) || n_quantiles < 2L) n_quantiles <- 5L
  if (!is.finite(tail_fraction) || tail_fraction <= 0 || tail_fraction >= 1) tail_fraction <- 0.10
  if (!is.finite(tail_oversample) || tail_oversample < 0) tail_oversample <- 0.40
  if (!is.finite(cluster_fraction) || cluster_fraction < 0) cluster_fraction <- 0.25
  tail_oversample <- min(0.90, tail_oversample)
  cluster_fraction <- min(0.80, cluster_fraction)
  if (count >= n) return(seq_len(n))

  cluster <- NULL
  selected <- integer()
  tail_pick <- integer()
  quantile_pick <- integer()
  cluster_pick <- integer()
  fill_pick <- integer()
  elapsed <- system.time({
    profile <- density_landmark_profile(ctx, alpha = 1, density_k = density_k)
    low_density_score <- as.numeric(profile$mean_distance)
    valid <- is.finite(low_density_score) & low_density_score > 0
    fallback <- if (any(valid)) stats::median(low_density_score[valid]) else 1
    low_density_score[!valid] <- fallback
    available <- rep(TRUE, n)
    set.seed(seed + 7601L + as.integer(round(1000 * tail_fraction)) + as.integer(round(100 * count)))

    tail_count <- min(count, max(1L, as.integer(round(count * tail_oversample))))
    cluster_count <- min(count - tail_count, max(0L, as.integer(round(count * cluster_fraction))))
    quantile_count <- max(0L, count - tail_count - cluster_count)

    tail_threshold <- as.numeric(stats::quantile(low_density_score, probs = 1 - tail_fraction, names = FALSE, type = 7))
    tail_candidates <- which(low_density_score >= tail_threshold)
    tail_pick <- rare_protected_weighted_sample(tail_candidates[available[tail_candidates]], tail_count, low_density_score)
    selected <- c(selected, tail_pick)
    available[tail_pick] <- FALSE

    density_rank <- rank(low_density_score, ties.method = "first")
    quantile_id <- pmin(n_quantiles, pmax(1L, as.integer(ceiling(density_rank / n * n_quantiles))))
    if (quantile_count > 0L) {
      capacity <- tabulate(quantile_id[available], nbins = n_quantiles)
      quantile_weights <- seq_len(n_quantiles)^1.5
      q_alloc <- rare_protected_allocate_counts(capacity, quantile_count, quantile_weights)
      for (q in seq_len(n_quantiles)) {
        if (q_alloc[q] <= 0L) next
        candidates <- which(available & quantile_id == q)
        pick <- rare_protected_weighted_sample(candidates, q_alloc[q], low_density_score)
        quantile_pick <- c(quantile_pick, pick)
        selected <- c(selected, pick)
        available[pick] <- FALSE
      }
    } else {
      q_alloc <- integer(n_quantiles)
    }

    if (cluster_count > 0L) {
      cluster <- stratified_cluster_assignments(ctx, count, seed)
      cluster_pick <- rare_protected_cluster_sample(cluster$labels, cluster_count, available)
      selected <- c(selected, cluster_pick)
      available[cluster_pick] <- FALSE
    }

    if (length(unique(selected)) < count) {
      fill_count <- count - length(unique(selected))
      fill_pick <- rare_protected_weighted_sample(which(available), fill_count, low_density_score)
      selected <- c(selected, fill_pick)
      available[fill_pick] <- FALSE
    }
  })[["elapsed"]]

  idx <- sort(unique(as.integer(selected)))
  if (length(idx) < count) {
    remaining <- setdiff(seq_len(n), idx)
    idx <- sort(c(idx, rare_protected_weighted_sample(remaining, count - length(idx), low_density_score)))
  }
  idx <- sort(idx[seq_len(min(length(idx), count))])
  selected_quantiles <- tabulate(quantile_id[idx], nbins = n_quantiles)
  global_score_mean <- mean(low_density_score)
  if (!is.finite(global_score_mean) || global_score_mean <= 0) global_score_mean <- 1

  attr(idx, "rare_protected_tail_fraction") <- tail_fraction
  attr(idx, "rare_protected_tail_oversample") <- tail_oversample
  attr(idx, "rare_protected_n_quantiles") <- n_quantiles
  attr(idx, "rare_protected_cluster_fraction") <- cluster_fraction
  attr(idx, "rare_protected_density_k") <- profile$density_k
  attr(idx, "rare_protected_time_sec") <- as.numeric(elapsed)
  attr(idx, "rare_protected_density_time_sec") <- profile$time_sec
  attr(idx, "rare_protected_cluster_time_sec") <- if (is.null(cluster)) NA_real_ else cluster$time_sec
  attr(idx, "rare_protected_tail_count") <- length(tail_pick)
  attr(idx, "rare_protected_quantile_count") <- length(quantile_pick)
  attr(idx, "rare_protected_cluster_count") <- length(cluster_pick)
  attr(idx, "rare_protected_fill_count") <- length(fill_pick)
  attr(idx, "rare_protected_tail_threshold") <- tail_threshold
  attr(idx, "rare_protected_selected_tail_fraction") <- mean(idx %in% tail_candidates)
  attr(idx, "rare_protected_selected_mean_distance_ratio") <- mean(low_density_score[idx]) / global_score_mean
  attr(idx, "rare_protected_selected_to_global_low_density_ratio") <- mean(low_density_score[idx]) / global_score_mean
  attr(idx, "rare_protected_quantile_min_selected") <- min(selected_quantiles)
  attr(idx, "rare_protected_quantile_max_selected") <- max(selected_quantiles)
  attr(idx, "rare_protected_cluster_k") <- if (is.null(cluster)) NA_real_ else cluster$cluster_k
  attr(idx, "rare_protected_cluster_feature_dims") <- if (is.null(cluster)) NA_real_ else cluster$feature_dims
  attr(idx, "density_landmark_k") <- profile$density_k
  attr(idx, "density_landmark_time_sec") <- profile$time_sec
  attr(idx, "density_landmark_weight_min") <- min(low_density_score)
  attr(idx, "density_landmark_weight_median") <- stats::median(low_density_score)
  attr(idx, "density_landmark_weight_max") <- max(low_density_score)
  attr(idx, "density_landmark_weight_mean") <- mean(low_density_score)
  attr(idx, "density_landmark_selected_weight_mean") <- mean(low_density_score[idx])
  attr(idx, "density_landmark_selected_to_global_weight_ratio") <- mean(low_density_score[idx]) / global_score_mean
  attr(idx, "density_landmark_mean_distance_median") <- stats::median(low_density_score, na.rm = TRUE)
  idx
}

landmark_indices_for_strategy <- function(ctx,
                                          selection,
                                          count,
                                          density_alpha = NA_real_,
                                          density_k = NULL,
                                          diversity_beta = NA_real_,
                                          rare_tail_fraction = NA_real_,
                                          rare_tail_oversample = NA_real_,
                                          rare_cluster_fraction = NA_real_,
                                          rare_n_quantiles = NA_integer_) {
  selection <- match.arg(selection, c(
    "projected_farthest", "random", "stratified", "stratified_label", "stratified_cluster",
    "density_weighted", "hybrid_density_diversity", "rare_protected",
    "farthest_point", "kmeanspp", "dpp_approx"
  ))
  stratified_allocation <- if (selection %in% c("stratified_label", "stratified_cluster")) "balanced" else "proportional"
  cache_key <- landmark_selection_cache_key(
    ctx,
    selection,
    count,
    density_alpha = density_alpha,
    density_k = density_k,
    diversity_beta = diversity_beta,
    stratified_allocation = if (selection %in% c("stratified", "stratified_label", "stratified_cluster")) {
      stratified_allocation
    } else {
      NULL
    },
    rare_tail_fraction = rare_tail_fraction,
    rare_tail_oversample = rare_tail_oversample,
    rare_cluster_fraction = rare_cluster_fraction,
    rare_n_quantiles = rare_n_quantiles
  )
  if (exists(cache_key, envir = landmark_selection_cache, inherits = FALSE)) {
    return(get(cache_key, envir = landmark_selection_cache, inherits = FALSE))
  }
  if (identical(selection, "projected_farthest")) {
    idx <- fastEmbedR:::select_landmark_rows(ctx$x, count, ctx$seed)
  } else if (identical(selection, "stratified") && !is.null(ctx$labels)) {
    idx <- stratified_subsample_indices(ctx$labels, count, ctx$seed, source = "label_auto", allocation = "proportional")
  } else if (identical(selection, "stratified") && is.null(ctx$labels)) {
    cluster <- stratified_cluster_assignments(ctx, count, ctx$seed)
    idx <- stratified_subsample_indices(
      cluster$labels,
      count,
      ctx$seed,
      source = "cluster_auto_kmeans",
      allocation = "balanced",
      setup_time_sec = cluster$time_sec,
      cluster_k = cluster$cluster_k,
      cluster_feature_dims = cluster$feature_dims
    )
  } else if (identical(selection, "stratified_label")) {
    if (is.null(ctx$labels)) stop("Label-stratified landmarking requires labels.", call. = FALSE)
    idx <- stratified_subsample_indices(ctx$labels, count, ctx$seed, source = "label", allocation = "balanced")
  } else if (identical(selection, "stratified_cluster")) {
    cluster <- stratified_cluster_assignments(ctx, count, ctx$seed)
    idx <- stratified_subsample_indices(
      cluster$labels,
      count,
      ctx$seed,
      source = "cluster_kmeans",
      allocation = "balanced",
      setup_time_sec = cluster$time_sec,
      cluster_k = cluster$cluster_k,
      cluster_feature_dims = cluster$feature_dims
    )
  } else if (identical(selection, "density_weighted")) {
    idx <- density_weighted_subsample_indices(ctx, count, ctx$seed, alpha = density_alpha, density_k = density_k)
  } else if (identical(selection, "hybrid_density_diversity")) {
    idx <- hybrid_density_diversity_indices(
      ctx,
      count,
      ctx$seed,
      alpha = density_alpha,
      beta = diversity_beta,
      density_k = density_k
    )
  } else if (identical(selection, "rare_protected")) {
    idx <- rare_protected_landmark_indices(
      ctx,
      count,
      ctx$seed,
      tail_fraction = rare_tail_fraction,
      tail_oversample = rare_tail_oversample,
      n_quantiles = rare_n_quantiles,
      cluster_fraction = rare_cluster_fraction,
      density_k = density_k
    )
  } else if (selection %in% c("farthest_point", "kmeanspp", "dpp_approx")) {
    idx <- diversity_landmark_indices(ctx, count, ctx$seed, selection)
  } else {
    set.seed(ctx$seed + 3001L)
    idx <- sort(sample.int(ctx$n, count))
  }
  attr(idx, "benchmark_selection") <- selection
  assign(cache_key, idx, envir = landmark_selection_cache)
  idx
}

patch_landmark_embedding_parameters <- function(value, fields) {
  if (inherits(value, "fastEmbedR_embedding")) {
    value$parameters <- c(value$parameters, fields)
    cfg <- attr(value$layout, "fastEmbedR_config")
    if (is.null(cfg)) cfg <- list()
    attr(value$layout, "fastEmbedR_config") <- c(cfg, fields)
    return(value)
  }
  cfg <- attr(value, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  attr(value, "fastEmbedR_config") <- c(cfg, fields)
  value
}

weighted_landmark_projection_layout <- function(landmark_layout,
                                                landmark_indices,
                                                projection_nn,
                                                n,
                                                weight = c("inverse_distance", "softmax", "gaussian")) {
  weight <- match.arg(weight)
  landmark_layout <- as.matrix(landmark_layout)
  storage.mode(landmark_layout) <- "double"
  indices <- as.matrix(projection_nn$indices)
  distances <- as.matrix(projection_nn$distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  if (nrow(indices) != n || !identical(dim(indices), dim(distances))) {
    stop("Projection KNN dimensions are inconsistent with the full data.", call. = FALSE)
  }
  n_components <- ncol(landmark_layout)
  k <- ncol(indices)
  out <- matrix(0, nrow = n, ncol = n_components)
  colnames(out) <- colnames(landmark_layout)
  eps <- sqrt(.Machine$double.eps)
  entropy <- numeric(n)
  bandwidth <- rep(NA_real_, n)
  zero_rows <- logical(n)
  for (i in seq_len(n)) {
    d <- pmax(0, distances[i, ])
    zero <- which(d <= eps)
    if (length(zero) > 0L) {
      w <- numeric(k)
      w[zero] <- 1 / length(zero)
      zero_rows[i] <- TRUE
    } else if (identical(weight, "inverse_distance")) {
      raw <- 1 / pmax(d, eps)
      w <- raw / sum(raw)
    } else {
      bw <- stats::median(d[is.finite(d) & d > eps])
      if (!is.finite(bw) || bw <= eps) bw <- max(eps, mean(d))
      bandwidth[i] <- bw
      shifted <- d - min(d)
      raw <- if (identical(weight, "softmax")) {
        exp(-shifted / bw)
      } else {
        exp(-0.5 * (shifted / bw)^2)
      }
      if (!any(is.finite(raw) & raw > 0)) raw <- rep(1, k)
      raw[!is.finite(raw) | raw < 0] <- 0
      w <- raw / sum(raw)
    }
    p <- pmax(w, .Machine$double.eps)
    entropy[i] <- -sum(p * log(p)) / log(k)
    rows <- indices[i, ]
    for (component in seq_len(n_components)) {
      out[i, component] <- sum(w * landmark_layout[rows, component])
    }
  }
  for (i in seq_along(landmark_indices)) {
    row <- as.integer(landmark_indices[i])
    if (row >= 1L && row <= n) out[row, ] <- landmark_layout[i, ]
  }
  attr(out, "projection_weight") <- weight
  attr(out, "projection_weight_entropy_mean") <- mean(entropy)
  attr(out, "projection_zero_neighbor_fraction") <- mean(zero_rows)
  attr(out, "projection_bandwidth_mean") <- if (all(is.na(bandwidth))) NA_real_ else mean(bandwidth, na.rm = TRUE)
  attr(out, "projection_bandwidth_rule") <- if (identical(weight, "inverse_distance")) "none" else "row_median_shifted_distance"
  out
}

landmark_projection_refinement_graph <- function(projection_nn, landmark_indices) {
  indices <- as.matrix(projection_nn$indices)
  distances <- as.matrix(projection_nn$distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  mapped <- matrix(
    landmark_indices[as.integer(indices)],
    nrow = nrow(indices),
    ncol = ncol(indices)
  )
  storage.mode(mapped) <- "integer"
  list(indices = mapped, distances = distances)
}

local_affine_landmark_projection_layout <- function(x,
                                                    landmark_layout,
                                                    landmark_indices,
                                                    projection_nn,
                                                    ridge = 1e-3,
                                                    weight = c("gaussian", "uniform"),
                                                    clip_multiplier = 3,
                                                    affine_blend = 0.25) {
  weight <- match.arg(weight)
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  landmark_layout <- as.matrix(landmark_layout)
  storage.mode(landmark_layout) <- "double"
  indices <- as.matrix(projection_nn$indices)
  distances <- as.matrix(projection_nn$distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  n <- nrow(x)
  if (nrow(indices) != n || !identical(dim(indices), dim(distances))) {
    stop("Projection KNN dimensions are inconsistent with the full data.", call. = FALSE)
  }
  x_landmarks <- x[landmark_indices, , drop = FALSE]
  n_components <- ncol(landmark_layout)
  k <- ncol(indices)
  out <- matrix(0, nrow = n, ncol = n_components)
  colnames(out) <- colnames(landmark_layout)
  eps <- sqrt(.Machine$double.eps)
  ridge <- as.numeric(ridge)[1L]
  if (!is.finite(ridge) || ridge < 0) ridge <- 1e-3
  clip_multiplier <- as.numeric(clip_multiplier)[1L]
  if (!is.finite(clip_multiplier) || clip_multiplier <= 0) clip_multiplier <- Inf
  affine_blend <- as.numeric(affine_blend)[1L]
  if (!is.finite(affine_blend) || affine_blend < 0) affine_blend <- 0.25
  affine_blend <- min(1, affine_blend)

  ranks <- numeric(n)
  conditions <- rep(NA_real_, n)
  entropy <- numeric(n)
  bandwidth <- rep(NA_real_, n)
  fallback <- logical(n)
  clipped <- logical(n)
  zero_rows <- logical(n)

  for (i in seq_len(n)) {
    rows <- indices[i, ]
    d <- pmax(0, distances[i, ])
    zero <- which(d <= eps)
    if (length(zero) > 0L) {
      w <- numeric(k)
      w[zero] <- 1 / length(zero)
      zero_rows[i] <- TRUE
    } else if (identical(weight, "uniform")) {
      w <- rep(1 / k, k)
    } else {
      bw <- stats::median(d[is.finite(d) & d > eps])
      if (!is.finite(bw) || bw <= eps) bw <- max(eps, mean(d))
      bandwidth[i] <- bw
      shifted <- d - min(d)
      raw <- exp(-0.5 * (shifted / bw)^2)
      raw[!is.finite(raw) | raw < 0] <- 0
      if (!any(raw > 0)) raw <- rep(1, k)
      w <- raw / sum(raw)
    }
    p <- pmax(w, .Machine$double.eps)
    entropy[i] <- -sum(p * log(p)) / log(k)

    x_neighbors <- x_landmarks[rows, , drop = FALSE]
    y_neighbors <- landmark_layout[rows, , drop = FALSE]
    x_center <- drop(crossprod(w, x_neighbors))
    y_center <- drop(crossprod(w, y_neighbors))
    weighted_average <- y_center
    x_centered <- sweep(x_neighbors, 2L, x_center, "-")
    y_centered <- sweep(y_neighbors, 2L, y_center, "-")
    sqrt_w <- sqrt(w)
    xw <- x_centered * sqrt_w
    yw <- y_centered * sqrt_w
    gram <- tcrossprod(xw)
    diag_mean <- mean(diag(gram))
    lambda <- ridge * if (is.finite(diag_mean) && diag_mean > eps) diag_mean else 1
    sv <- tryCatch(svd(gram, nu = 0L, nv = 0L)$d, error = function(e) numeric())
    if (length(sv)) {
      positive <- sv[is.finite(sv) & sv > max(sv) * 1e-8]
      ranks[i] <- length(positive)
      if (length(positive)) conditions[i] <- max(positive) / min(positive)
    }
    pred <- tryCatch({
      coef <- solve(gram + diag(lambda, k), yw)
      kernel <- drop((x[i, ] - x_center) %*% t(xw))
      y_center + drop(kernel %*% coef)
    }, error = function(e) rep(NA_real_, n_components))
    if (length(pred) != n_components || any(!is.finite(pred))) {
      pred <- weighted_average
      fallback[i] <- TRUE
    } else if (is.finite(clip_multiplier)) {
      radius <- sqrt(sum(w * rowSums(sweep(y_neighbors, 2L, y_center, "-")^2)))
      delta <- pred - y_center
      norm_delta <- sqrt(sum(delta * delta))
      max_delta <- clip_multiplier * max(radius, eps)
      if (is.finite(norm_delta) && norm_delta > max_delta) {
        pred <- y_center + delta * (max_delta / norm_delta)
        clipped[i] <- TRUE
      }
    }
    pred <- weighted_average + affine_blend * (pred - weighted_average)
    out[i, ] <- pred
  }
  for (i in seq_along(landmark_indices)) {
    row <- as.integer(landmark_indices[i])
    if (row >= 1L && row <= n) out[row, ] <- landmark_layout[i, ]
  }
  attr(out, "projection_weight") <- weight
  attr(out, "projection_weight_entropy_mean") <- mean(entropy)
  attr(out, "projection_zero_neighbor_fraction") <- mean(zero_rows)
  attr(out, "projection_bandwidth_mean") <- if (all(is.na(bandwidth))) NA_real_ else mean(bandwidth, na.rm = TRUE)
  attr(out, "projection_bandwidth_rule") <- if (identical(weight, "gaussian")) "row_median_shifted_distance" else "none"
  attr(out, "affine_ridge") <- ridge
  attr(out, "affine_rank_mean") <- mean(ranks)
  attr(out, "affine_condition_median") <- if (all(is.na(conditions))) NA_real_ else stats::median(conditions, na.rm = TRUE)
  attr(out, "affine_condition_max") <- if (all(is.na(conditions))) NA_real_ else max(conditions, na.rm = TRUE)
  attr(out, "affine_fallback_fraction") <- mean(fallback)
  attr(out, "affine_clipped_fraction") <- mean(clipped)
  attr(out, "affine_clip_multiplier") <- clip_multiplier
  attr(out, "affine_blend") <- affine_blend
  out
}

run_landmark_projection_strategy <- function(ctx,
                                             projection_k,
                                             weight,
                                             fraction = 0.10,
                                             count = NULL,
                                             selection = "projected_farthest") {
  count <- landmark_count_for_strategy(ctx, fraction = fraction, count = count)
  landmark_indices <- landmark_indices_for_strategy(ctx, selection, count)
  selection_used <- attr(landmark_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_landmarks <- ctx$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- max(1L, min(as.integer(ctx$k), nrow(x_landmarks) - 1L))
  projection_k <- max(1L, min(as.integer(projection_k), nrow(x_landmarks)))
  landmark_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = "cpu")
    landmark_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, nrow(x_landmarks), landmark_neighbors)
  })[["elapsed"]]
  landmark_embedding_time <- system.time({
    landmark_layout <- fastEmbedR:::embed_from_knn(
      ctx$method,
      landmark_knn$indices,
      landmark_knn$distances,
      2L,
      ctx$seed,
      ctx$backend,
      method_quality(ctx$method, "fast"),
      FALSE,
      n_epochs = ctx$short_epochs
    )
    landmark_layout <- coerce_layout(landmark_layout, nrow(x_landmarks))
  })[["elapsed"]]
  projection_nn <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_landmarks,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    layout <- weighted_landmark_projection_layout(
      landmark_layout,
      landmark_indices,
      projection_nn,
      ctx$n,
      weight = weight
    )
  })[["elapsed"]]
  fields <- list(
    landmark_approximation = "landmark_knn_projection",
    landmark_mode = "fast",
    landmark_selection_requested = selection,
    landmark_selection_used = selection_used,
    landmark_count_requested = as.integer(count),
    landmark_fraction_requested = if (is.null(fraction)) NA_real_ else as.numeric(fraction),
    landmark_n = length(landmark_indices),
    landmark_fraction = length(landmark_indices) / ctx$n,
    landmark_projection_k = projection_k,
    landmark_interpolation = paste0("knn_", weight),
    landmark_projection_weight = weight,
    landmark_projection_bandwidth_rule = safe_character(attr(layout, "projection_bandwidth_rule")),
    landmark_projection_bandwidth_mean = safe_number(attr(layout, "projection_bandwidth_mean")),
    landmark_projection_weight_entropy = safe_number(attr(layout, "projection_weight_entropy_mean")),
    landmark_projection_zero_neighbor_fraction = safe_number(attr(layout, "projection_zero_neighbor_fraction")),
    landmark_projection_time_sec = as.numeric(projection_time),
    landmark_projection_backend = safe_character(attr(projection_nn, "backend"), "cpu"),
    landmark_interpolation_backend = "cpu_r_weighted_average",
    landmark_interpolation_backend_reason = NA_character_,
    landmark_refinement = "none",
    landmark_refinement_epochs = 0L,
    landmark_refinement_backend = NA_character_,
    landmark_refinement_knn_backend = NA_character_,
    landmark_refinement_knn_backend_reason = NA_character_,
    landmark_landmark_knn_time_sec = as.numeric(landmark_knn_time),
    landmark_landmark_embedding_time_sec = as.numeric(landmark_embedding_time),
    subsample_strategy = selection,
    subsample_stratified = selection %in% c("stratified", "stratified_label", "stratified_cluster"),
    benchmark_forced_k = as.integer(ctx$k),
    benchmark_standardize = FALSE
  )
  fields <- c(fields, landmark_label_coverage_fields(ctx$labels, landmark_indices))
  cfg <- attr(landmark_layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

run_landmark_projection_refinement_strategy <- function(ctx,
                                                        refinement_epochs,
                                                        projection_k = NULL,
                                                        weight = "gaussian",
                                                        fraction = 0.10,
                                                        count = NULL,
                                                        selection = "projected_farthest") {
  count <- landmark_count_for_strategy(ctx, fraction = fraction, count = count)
  landmark_indices <- landmark_indices_for_strategy(ctx, selection, count)
  selection_used <- attr(landmark_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_landmarks <- ctx$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- max(1L, min(as.integer(ctx$k), nrow(x_landmarks) - 1L))
  projection_k <- if (is.null(projection_k)) ctx$k else projection_k
  projection_k <- max(1L, min(as.integer(projection_k), nrow(x_landmarks)))
  refinement_epochs <- max(1L, as.integer(refinement_epochs))

  landmark_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = "cpu")
    landmark_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, nrow(x_landmarks), landmark_neighbors)
  })[["elapsed"]]
  landmark_embedding_time <- system.time({
    landmark_layout <- fastEmbedR:::embed_from_knn(
      ctx$method,
      landmark_knn$indices,
      landmark_knn$distances,
      2L,
      ctx$seed,
      "cpu",
      method_quality(ctx$method, "fast"),
      FALSE,
      n_epochs = ctx$short_epochs
    )
    landmark_layout <- coerce_layout(landmark_layout, nrow(x_landmarks))
  })[["elapsed"]]

  projection_nn <- NULL
  projection_graph <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_landmarks,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    layout <- weighted_landmark_projection_layout(
      landmark_layout,
      landmark_indices,
      projection_nn,
      ctx$n,
      weight = weight
    )
    projection_graph <- landmark_projection_refinement_graph(projection_nn, landmark_indices)
  })[["elapsed"]]

  refinement_time <- system.time({
    refined <- fastEmbedR:::refine_embedding_from_knn(
      ctx$method,
      projection_graph$indices,
      projection_graph$distances,
      layout,
      n_epochs = refinement_epochs,
      refinement = "projection",
      seed = ctx$seed,
      backend = "cpu",
      verbose = FALSE
    )
    refined <- coerce_layout(refined, ctx$n)
  })[["elapsed"]]

  refined_cfg <- attr(refined, "fastEmbedR_config")
  fields <- list(
    landmark_enabled = TRUE,
    landmark_approximation = "landmark_projection_refinement",
    landmark_mode = "projection_refine",
    landmark_selection_requested = selection,
    landmark_selection_used = selection_used,
    landmark_count_requested = as.integer(count),
    landmark_fraction_requested = if (is.null(fraction)) NA_real_ else as.numeric(fraction),
    landmark_n = length(landmark_indices),
    landmark_fraction = length(landmark_indices) / ctx$n,
    landmark_projection_k = projection_k,
    landmark_interpolation = paste0("knn_", weight),
    landmark_projection_model = "weighted_average_then_projection_refinement",
    landmark_projection_weight = weight,
    landmark_projection_bandwidth_rule = safe_character(attr(layout, "projection_bandwidth_rule")),
    landmark_projection_bandwidth_mean = safe_number(attr(layout, "projection_bandwidth_mean")),
    landmark_projection_weight_entropy = safe_number(attr(layout, "projection_weight_entropy_mean")),
    landmark_projection_zero_neighbor_fraction = safe_number(attr(layout, "projection_zero_neighbor_fraction")),
    landmark_projection_time_sec = as.numeric(projection_time),
    landmark_projection_backend = safe_character(attr(projection_nn, "backend"), "cpu"),
    landmark_interpolation_backend = "cpu_r_weighted_average",
    landmark_interpolation_backend_reason = NA_character_,
    landmark_refinement = "projection",
    landmark_refinement_epochs = refinement_epochs,
    landmark_refinement_time_sec = as.numeric(refinement_time),
    landmark_refinement_backend = if (is.list(refined_cfg)) safe_character(refined_cfg$refinement_backend, "cpu") else "cpu",
    landmark_refinement_knn_backend = "projection_graph",
    landmark_refinement_knn_backend_reason = NA_character_,
    landmark_landmark_knn_time_sec = as.numeric(landmark_knn_time),
    landmark_landmark_embedding_time_sec = as.numeric(landmark_embedding_time),
    subsample_strategy = selection,
    subsample_stratified = selection %in% c("stratified", "stratified_label", "stratified_cluster"),
    benchmark_forced_k = as.integer(ctx$k),
    benchmark_standardize = FALSE
  )
  fields <- c(fields, landmark_label_coverage_fields(ctx$labels, landmark_indices))
  cfg <- attr(landmark_layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  attr(refined, "fastEmbedR_config") <- c(cfg, fields)
  refined
}

run_landmark_affine_projection_strategy <- function(ctx,
                                                    projection_k,
                                                    ridge = 1e-3,
                                                    fraction = 0.10,
                                                    count = NULL,
                                                    selection = "projected_farthest",
                                                    weight = "gaussian",
                                                    clip_multiplier = 3,
                                                    affine_blend = 0.25) {
  count <- landmark_count_for_strategy(ctx, fraction = fraction, count = count)
  landmark_indices <- landmark_indices_for_strategy(ctx, selection, count)
  selection_used <- attr(landmark_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_landmarks <- ctx$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- max(1L, min(as.integer(ctx$k), nrow(x_landmarks) - 1L))
  projection_k <- max(2L, min(as.integer(projection_k), nrow(x_landmarks)))
  landmark_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = "cpu")
    landmark_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, nrow(x_landmarks), landmark_neighbors)
  })[["elapsed"]]
  landmark_embedding_time <- system.time({
    landmark_layout <- fastEmbedR:::embed_from_knn(
      ctx$method,
      landmark_knn$indices,
      landmark_knn$distances,
      2L,
      ctx$seed,
      ctx$backend,
      method_quality(ctx$method, "fast"),
      FALSE,
      n_epochs = ctx$short_epochs
    )
    landmark_layout <- coerce_layout(landmark_layout, nrow(x_landmarks))
  })[["elapsed"]]
  projection_nn <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_landmarks,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    layout <- local_affine_landmark_projection_layout(
      ctx$x,
      landmark_layout,
      landmark_indices,
      projection_nn,
      ridge = ridge,
      weight = weight,
      clip_multiplier = clip_multiplier,
      affine_blend = affine_blend
    )
  })[["elapsed"]]
  fields <- list(
    landmark_approximation = "landmark_local_affine_projection",
    landmark_mode = "fast",
    landmark_selection_requested = selection,
    landmark_selection_used = selection_used,
    landmark_count_requested = as.integer(count),
    landmark_fraction_requested = if (is.null(fraction)) NA_real_ else as.numeric(fraction),
    landmark_n = length(landmark_indices),
    landmark_fraction = length(landmark_indices) / ctx$n,
    landmark_projection_k = projection_k,
    landmark_interpolation = "local_affine",
    landmark_projection_model = "local_affine_dual_ridge",
    landmark_projection_weight = weight,
    landmark_projection_bandwidth_rule = safe_character(attr(layout, "projection_bandwidth_rule")),
    landmark_projection_bandwidth_mean = safe_number(attr(layout, "projection_bandwidth_mean")),
    landmark_projection_weight_entropy = safe_number(attr(layout, "projection_weight_entropy_mean")),
    landmark_projection_zero_neighbor_fraction = safe_number(attr(layout, "projection_zero_neighbor_fraction")),
    landmark_projection_time_sec = as.numeric(projection_time),
    landmark_projection_backend = safe_character(attr(projection_nn, "backend"), "cpu"),
    landmark_interpolation_backend = "cpu_r_local_affine",
    landmark_interpolation_backend_reason = NA_character_,
    landmark_affine_ridge = safe_number(attr(layout, "affine_ridge")),
    landmark_affine_weight = weight,
    landmark_affine_rank_mean = safe_number(attr(layout, "affine_rank_mean")),
    landmark_affine_condition_median = safe_number(attr(layout, "affine_condition_median")),
    landmark_affine_condition_max = safe_number(attr(layout, "affine_condition_max")),
    landmark_affine_fallback_fraction = safe_number(attr(layout, "affine_fallback_fraction")),
    landmark_affine_clipped_fraction = safe_number(attr(layout, "affine_clipped_fraction")),
    landmark_affine_clip_multiplier = safe_number(attr(layout, "affine_clip_multiplier")),
    landmark_affine_blend = safe_number(attr(layout, "affine_blend")),
    landmark_refinement = "none",
    landmark_refinement_epochs = 0L,
    landmark_refinement_backend = NA_character_,
    landmark_refinement_knn_backend = NA_character_,
    landmark_refinement_knn_backend_reason = NA_character_,
    landmark_landmark_knn_time_sec = as.numeric(landmark_knn_time),
    landmark_landmark_embedding_time_sec = as.numeric(landmark_embedding_time),
    subsample_strategy = selection,
    subsample_stratified = selection %in% c("stratified", "stratified_label", "stratified_cluster"),
    benchmark_forced_k = as.integer(ctx$k),
    benchmark_standardize = FALSE
  )
  fields <- c(fields, landmark_label_coverage_fields(ctx$labels, landmark_indices))
  cfg <- attr(landmark_layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

landmark_label_coverage_fields <- function(labels, idx) {
  if (is.null(labels) || length(idx) == 0L) {
    return(list(
      landmark_label_classes_total = NA_real_,
      landmark_label_classes_present = NA_real_,
      landmark_label_missing_classes = NA_real_,
      landmark_label_min_count = NA_real_,
      landmark_label_min_fraction = NA_real_,
      landmark_rare_label_count = NA_real_,
      landmark_rare_label_present = NA
    ))
  }
  labels <- as.factor(labels)
  idx <- as.integer(idx)
  idx <- idx[is.finite(idx) & idx >= 1L & idx <= length(labels)]
  if (length(idx) == 0L) {
    return(landmark_label_coverage_fields(NULL, integer()))
  }
  selected_counts <- table(factor(labels[idx], levels = levels(labels)))
  dataset_counts <- table(labels)
  rare_levels <- names(dataset_counts)[dataset_counts == min(dataset_counts)]
  rare_counts <- as.integer(selected_counts[rare_levels])
  list(
    landmark_label_classes_total = length(selected_counts),
    landmark_label_classes_present = sum(selected_counts > 0L),
    landmark_label_missing_classes = sum(selected_counts == 0L),
    landmark_label_min_count = min(as.integer(selected_counts)),
    landmark_label_min_fraction = min(as.integer(selected_counts)) / length(idx),
    landmark_rare_label_count = min(rare_counts),
    landmark_rare_label_present = all(rare_counts > 0L)
  )
}

run_landmark_subsample_strategy <- function(ctx,
                                            selection,
                                            fraction = NULL,
                                            count = NULL,
                                            mode = c("fast", "balanced", "accurate"),
                                            density_alpha = NA_real_,
                                            density_k = NULL,
                                            diversity_beta = NA_real_,
                                            rare_tail_fraction = NA_real_,
                                            rare_tail_oversample = NA_real_,
                                            rare_cluster_fraction = NA_real_,
                                            rare_n_quantiles = NA_integer_,
                                            strategy_label = NULL) {
  mode <- match.arg(mode)
  count <- landmark_count_for_strategy(ctx, fraction = fraction, count = count)
  selection_requested <- selection
  explicit_indices <- landmark_indices_for_strategy(
    ctx,
    selection,
    count,
    density_alpha = density_alpha,
    density_k = density_k,
    diversity_beta = diversity_beta,
    rare_tail_fraction = rare_tail_fraction,
    rare_tail_oversample = rare_tail_oversample,
    rare_cluster_fraction = rare_cluster_fraction,
    rare_n_quantiles = rare_n_quantiles
  )
  selection_used <- attr(explicit_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection_requested
  value <- fastEmbedR:::embed(
    ctx$x,
    ctx$labels,
    method = ctx$method,
    mode = mode,
    n_neighbors = ctx$k,
    standardize = FALSE,
    pca_dims = NULL,
    landmarks = explicit_indices,
    backend = ctx$backend,
    quality = method_quality(ctx$method, "fast"),
    seed = ctx$seed,
    silhouette_sample = NULL,
    preserve_sample = NULL
  )
  fields <- list(
    landmark_approximation = if (is.null(strategy_label)) "landmark_subsample" else strategy_label,
    landmark_selection_requested = selection_requested,
    landmark_selection_used = selection_used,
    landmark_fraction_requested = if (is.null(fraction)) NA_real_ else as.numeric(fraction),
    landmark_count_requested = as.integer(count),
    landmark_mode = mode,
    subsample_strategy = if (selection_requested %in% c("random", "stratified", "stratified_label", "stratified_cluster", "density_weighted", "hybrid_density_diversity", "rare_protected")) selection_requested else NA_character_,
    subsample_stratified = selection_requested %in% c("stratified", "stratified_label", "stratified_cluster"),
    stratified_landmark_source = safe_character(attr(explicit_indices, "stratified_landmark_source")),
    stratified_landmark_allocation = safe_character(attr(explicit_indices, "stratified_landmark_allocation")),
    stratified_landmark_time_sec = safe_number(attr(explicit_indices, "stratified_landmark_time_sec")),
    stratified_landmark_n_strata = safe_number(attr(explicit_indices, "stratified_landmark_n_strata")),
    stratified_landmark_strata_sampled = safe_number(attr(explicit_indices, "stratified_landmark_strata_sampled")),
    stratified_landmark_missing_strata = safe_number(attr(explicit_indices, "stratified_landmark_missing_strata")),
    stratified_landmark_min_stratum_size = safe_number(attr(explicit_indices, "stratified_landmark_min_stratum_size")),
    stratified_landmark_max_stratum_size = safe_number(attr(explicit_indices, "stratified_landmark_max_stratum_size")),
    stratified_landmark_min_selected_per_stratum = safe_number(attr(explicit_indices, "stratified_landmark_min_selected_per_stratum")),
    stratified_landmark_max_selected_per_stratum = safe_number(attr(explicit_indices, "stratified_landmark_max_selected_per_stratum")),
    stratified_landmark_balance_ratio = safe_number(attr(explicit_indices, "stratified_landmark_balance_ratio")),
    stratified_landmark_cluster_k = safe_number(attr(explicit_indices, "stratified_landmark_cluster_k")),
    stratified_landmark_cluster_feature_dims = safe_number(attr(explicit_indices, "stratified_landmark_cluster_feature_dims")),
    density_landmark_alpha = safe_number(attr(explicit_indices, "density_landmark_alpha")),
    density_landmark_k = safe_number(attr(explicit_indices, "density_landmark_k")),
    density_landmark_time_sec = safe_number(attr(explicit_indices, "density_landmark_time_sec")),
    density_landmark_weight_min = safe_number(attr(explicit_indices, "density_landmark_weight_min")),
    density_landmark_weight_median = safe_number(attr(explicit_indices, "density_landmark_weight_median")),
    density_landmark_weight_max = safe_number(attr(explicit_indices, "density_landmark_weight_max")),
    density_landmark_weight_mean = safe_number(attr(explicit_indices, "density_landmark_weight_mean")),
    density_landmark_selected_weight_mean = safe_number(attr(explicit_indices, "density_landmark_selected_weight_mean")),
    density_landmark_selected_to_global_weight_ratio = safe_number(attr(explicit_indices, "density_landmark_selected_to_global_weight_ratio")),
    density_landmark_mean_distance_median = safe_number(attr(explicit_indices, "density_landmark_mean_distance_median")),
    hybrid_landmark_alpha = safe_number(attr(explicit_indices, "hybrid_landmark_alpha")),
    hybrid_landmark_beta = safe_number(attr(explicit_indices, "hybrid_landmark_beta")),
    hybrid_landmark_k = safe_number(attr(explicit_indices, "hybrid_landmark_k")),
    hybrid_landmark_time_sec = safe_number(attr(explicit_indices, "hybrid_landmark_time_sec")),
    hybrid_landmark_density_time_sec = safe_number(attr(explicit_indices, "hybrid_landmark_density_time_sec")),
    hybrid_landmark_feature_dims = safe_number(attr(explicit_indices, "hybrid_landmark_feature_dims")),
    hybrid_landmark_formula = safe_character(attr(explicit_indices, "hybrid_landmark_formula")),
    hybrid_landmark_density_weight_median = safe_number(attr(explicit_indices, "hybrid_landmark_density_weight_median")),
    hybrid_landmark_density_selected_to_global_weight_ratio = safe_number(attr(explicit_indices, "hybrid_landmark_density_selected_to_global_weight_ratio")),
    hybrid_landmark_mean_distance_median = safe_number(attr(explicit_indices, "hybrid_landmark_mean_distance_median")),
    hybrid_landmark_cover_mean = safe_number(attr(explicit_indices, "hybrid_landmark_cover_mean")),
    hybrid_landmark_cover_median = safe_number(attr(explicit_indices, "hybrid_landmark_cover_median")),
    hybrid_landmark_cover_max = safe_number(attr(explicit_indices, "hybrid_landmark_cover_max")),
    rare_protected_tail_fraction = safe_number(attr(explicit_indices, "rare_protected_tail_fraction")),
    rare_protected_tail_oversample = safe_number(attr(explicit_indices, "rare_protected_tail_oversample")),
    rare_protected_n_quantiles = safe_number(attr(explicit_indices, "rare_protected_n_quantiles")),
    rare_protected_cluster_fraction = safe_number(attr(explicit_indices, "rare_protected_cluster_fraction")),
    rare_protected_density_k = safe_number(attr(explicit_indices, "rare_protected_density_k")),
    rare_protected_time_sec = safe_number(attr(explicit_indices, "rare_protected_time_sec")),
    rare_protected_density_time_sec = safe_number(attr(explicit_indices, "rare_protected_density_time_sec")),
    rare_protected_cluster_time_sec = safe_number(attr(explicit_indices, "rare_protected_cluster_time_sec")),
    rare_protected_tail_count = safe_number(attr(explicit_indices, "rare_protected_tail_count")),
    rare_protected_quantile_count = safe_number(attr(explicit_indices, "rare_protected_quantile_count")),
    rare_protected_cluster_count = safe_number(attr(explicit_indices, "rare_protected_cluster_count")),
    rare_protected_fill_count = safe_number(attr(explicit_indices, "rare_protected_fill_count")),
    rare_protected_tail_threshold = safe_number(attr(explicit_indices, "rare_protected_tail_threshold")),
    rare_protected_selected_tail_fraction = safe_number(attr(explicit_indices, "rare_protected_selected_tail_fraction")),
    rare_protected_selected_mean_distance_ratio = safe_number(attr(explicit_indices, "rare_protected_selected_mean_distance_ratio")),
    rare_protected_selected_to_global_low_density_ratio = safe_number(attr(explicit_indices, "rare_protected_selected_to_global_low_density_ratio")),
    rare_protected_quantile_min_selected = safe_number(attr(explicit_indices, "rare_protected_quantile_min_selected")),
    rare_protected_quantile_max_selected = safe_number(attr(explicit_indices, "rare_protected_quantile_max_selected")),
    rare_protected_cluster_k = safe_number(attr(explicit_indices, "rare_protected_cluster_k")),
    rare_protected_cluster_feature_dims = safe_number(attr(explicit_indices, "rare_protected_cluster_feature_dims")),
    diversity_landmark_algorithm = safe_character(attr(explicit_indices, "diversity_landmark_algorithm")),
    diversity_landmark_time_sec = safe_number(attr(explicit_indices, "diversity_landmark_time_sec")),
    diversity_landmark_feature_dims = safe_number(attr(explicit_indices, "diversity_landmark_feature_dims")),
    diversity_landmark_cover_mean = safe_number(attr(explicit_indices, "diversity_landmark_cover_mean")),
    diversity_landmark_cover_median = safe_number(attr(explicit_indices, "diversity_landmark_cover_median")),
    diversity_landmark_cover_max = safe_number(attr(explicit_indices, "diversity_landmark_cover_max")),
    diversity_landmark_leverage_selected_to_global_ratio = safe_number(attr(explicit_indices, "diversity_landmark_leverage_selected_to_global_ratio")),
    benchmark_forced_k = as.integer(ctx$k),
    benchmark_standardize = FALSE
  )
  fields <- c(fields, landmark_label_coverage_fields(ctx$labels, explicit_indices))
  patch_landmark_embedding_parameters(value, fields)
}

landmark_subsample_strategy <- function(id,
                                        selection = c(
                                          "projected_farthest", "random", "stratified", "stratified_label",
                                          "stratified_cluster", "density_weighted",
                                          "hybrid_density_diversity", "rare_protected",
                                          "farthest_point", "kmeanspp", "dpp_approx"
                                        ),
                                        fraction = NULL,
                                        count = NULL,
                                        mode = c("fast", "balanced", "accurate"),
                                        density_alpha = NA_real_,
                                        density_k = NULL,
                                        diversity_beta = NA_real_,
                                        rare_tail_fraction = NA_real_,
                                        rare_tail_oversample = NA_real_,
                                        rare_cluster_fraction = NA_real_,
                                        rare_n_quantiles = NA_integer_,
                                        max_n = Inf) {
  selection <- match.arg(selection)
  mode <- match.arg(mode)
  list(
    id = id,
    family = if (identical(selection, "density_weighted")) {
      "density_landmarking"
    } else if (identical(selection, "hybrid_density_diversity")) {
      "hybrid_density_diversity_landmarking"
    } else if (identical(selection, "rare_protected")) {
      "rare_cell_protected_landmarking"
    } else if (selection %in% c("stratified_label", "stratified_cluster")) {
      "stratified_landmarking"
    } else if (selection %in% c("farthest_point", "kmeanspp", "dpp_approx")) {
      "diversity_landmarking"
    } else {
      "landmarking"
    },
    description = paste0(
      "Landmark/subsample approximation using ", selection,
      " landmarks and mode=", mode,
      if (!is.null(fraction)) paste0(" at fraction=", fraction) else "",
      if (identical(selection, "density_weighted")) paste0(" with alpha=", density_alpha) else "",
      if (identical(selection, "hybrid_density_diversity")) {
        paste0(" with alpha=", density_alpha, " and beta=", diversity_beta)
      } else {
        ""
      },
      if (identical(selection, "rare_protected")) {
        paste0(
          " with tail_fraction=", rare_tail_fraction,
          ", tail_oversample=", rare_tail_oversample,
          ", cluster_fraction=", rare_cluster_fraction,
          ", n_quantiles=", rare_n_quantiles
        )
      } else {
        ""
      },
      if (!is.null(count)) paste0(" at count=", count) else "."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && backend %in% c("cpu", "cuda", "metal"),
    context_available = function(ctx) {
      landmark_count <- landmark_count_for_strategy(ctx, fraction = fraction, count = count)
      list(
        available = ctx$n <= max_n &&
          landmark_count >= 2L &&
          landmark_count < ctx$n &&
          !(identical(selection, "stratified_label") && is.null(ctx$labels)),
        message = paste0(
          "Landmark/subsample strategy skipped because n exceeds limit, labels are unavailable for label-stratified sampling, or the requested landmark count is invalid. n=",
          ctx$n, ", max_n=", max_n, ", landmarks=", landmark_count,
          ", labels_available=", !is.null(ctx$labels)
        )
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      selection = selection,
      fraction = if (is.null(fraction)) NA_real_ else as.numeric(fraction),
      count = landmark_count_for_strategy(ctx, fraction = fraction, count = count),
      mode = mode,
      density_weight = if (selection %in% c("density_weighted", "hybrid_density_diversity")) "inverse mean kNN distance" else NA_character_,
      density_alpha = if (selection %in% c("density_weighted", "hybrid_density_diversity")) safe_number(density_alpha) else NA_real_,
      density_k = if (is.null(density_k)) ctx$k else as.integer(density_k),
      stratified_source_requested = if (selection %in% c("stratified", "stratified_label", "stratified_cluster")) selection else NA_character_,
      stratified_allocation = if (selection %in% c("stratified_label", "stratified_cluster")) "balanced" else if (identical(selection, "stratified")) "proportional_or_cluster_balanced" else NA_character_,
      hybrid_density_diversity = identical(selection, "hybrid_density_diversity"),
      hybrid_formula = if (identical(selection, "hybrid_density_diversity")) "p_i proportional to density_i^alpha * diversity_i^beta" else NA_character_,
      diversity_beta = if (identical(selection, "hybrid_density_diversity")) safe_number(diversity_beta) else NA_real_,
      rare_cell_protected = identical(selection, "rare_protected"),
      rare_protected_tail_fraction = if (identical(selection, "rare_protected")) safe_number(rare_tail_fraction) else NA_real_,
      rare_protected_tail_oversample = if (identical(selection, "rare_protected")) safe_number(rare_tail_oversample) else NA_real_,
      rare_protected_cluster_fraction = if (identical(selection, "rare_protected")) safe_number(rare_cluster_fraction) else NA_real_,
      rare_protected_n_quantiles = if (identical(selection, "rare_protected")) safe_number(rare_n_quantiles) else NA_real_,
      rare_protected_formula = if (identical(selection, "rare_protected")) "density quantiles + low-density tail oversampling + cluster-stratified coverage" else NA_character_,
      diversity_algorithm = if (selection %in% c("farthest_point", "kmeanspp", "dpp_approx")) selection else NA_character_,
      refinement = switch(mode, fast = "none", balanced = "bucketed", accurate = "full"),
      quality = method_quality(ctx$method, "fast"),
      standardize = FALSE,
      pca_dims = NULL,
      caveat = "subsample strategies fit only selected rows and project the full dataset"
    ),
    run = function(ctx) run_landmark_subsample_strategy(
      ctx,
      selection = selection,
      fraction = fraction,
      count = count,
      mode = mode,
      density_alpha = density_alpha,
      density_k = density_k,
      diversity_beta = diversity_beta,
      rare_tail_fraction = rare_tail_fraction,
      rare_tail_oversample = rare_tail_oversample,
      rare_cluster_fraction = rare_cluster_fraction,
      rare_n_quantiles = rare_n_quantiles,
      strategy_label = id
    )
  )
}

landmark_subsample_strategy_grid <- function() {
  list(
    landmark_subsample_strategy("landmark_auto_project", "projected_farthest", mode = "fast"),
    landmark_subsample_strategy("landmark_auto_bucketed", "projected_farthest", mode = "balanced"),
    landmark_subsample_strategy("landmark_frac10_project", "projected_farthest", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("landmark_frac10_bucketed", "projected_farthest", fraction = 0.10, mode = "balanced"),
    landmark_subsample_strategy("landmark_frac25_bucketed", "projected_farthest", fraction = 0.25, mode = "balanced"),
    landmark_subsample_strategy("landmark_frac25_full_refine", "projected_farthest", fraction = 0.25, mode = "accurate", max_n = 12000L),
    landmark_subsample_strategy("subsample_random10_project", "random", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("subsample_random25_bucketed", "random", fraction = 0.25, mode = "balanced"),
    landmark_subsample_strategy("subsample_stratified10_project", "stratified", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("subsample_stratified25_bucketed", "stratified", fraction = 0.25, mode = "balanced")
  )
}

random_landmark_strategy_grid <- function() {
  list(
    landmark_subsample_strategy("random_landmark_ratio01_project", "random", fraction = 0.01, mode = "fast"),
    landmark_subsample_strategy("random_landmark_ratio05_project", "random", fraction = 0.05, mode = "fast"),
    landmark_subsample_strategy("random_landmark_ratio10_project", "random", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("random_landmark_ratio20_project", "random", fraction = 0.20, mode = "fast")
  )
}

density_weighted_landmark_strategy_grid <- function() {
  list(
    landmark_subsample_strategy("density_landmark_alpha0_ratio10_project", "density_weighted", fraction = 0.10, mode = "fast", density_alpha = 0),
    landmark_subsample_strategy("density_landmark_alpha0p5_ratio10_project", "density_weighted", fraction = 0.10, mode = "fast", density_alpha = 0.5),
    landmark_subsample_strategy("density_landmark_alpha1_ratio10_project", "density_weighted", fraction = 0.10, mode = "fast", density_alpha = 1),
    landmark_subsample_strategy("density_landmark_alpha2_ratio10_project", "density_weighted", fraction = 0.10, mode = "fast", density_alpha = 2)
  )
}

stratified_landmark_strategy_grid <- function() {
  list(
    landmark_subsample_strategy("stratified_label_ratio10_project", "stratified_label", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("stratified_cluster_ratio10_project", "stratified_cluster", fraction = 0.10, mode = "fast")
  )
}

diversity_landmark_strategy_grid <- function() {
  list(
    landmark_subsample_strategy("diversity_farthest_ratio10_project", "farthest_point", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("diversity_kmeanspp_ratio10_project", "kmeanspp", fraction = 0.10, mode = "fast"),
    landmark_subsample_strategy("diversity_dpp_ratio10_project", "dpp_approx", fraction = 0.10, mode = "fast")
  )
}

hybrid_density_diversity_landmark_strategy_grid <- function() {
  fmt <- function(x) gsub("\\.", "p", format(x, trim = TRUE, scientific = FALSE))
  alphas <- c(0, 0.5, 1)
  betas <- c(0, 0.5, 1)
  out <- vector("list", length(alphas) * length(betas))
  pos <- 1L
  for (alpha in alphas) {
    for (beta in betas) {
      out[[pos]] <- landmark_subsample_strategy(
        paste0("hybrid_dd_alpha", fmt(alpha), "_beta", fmt(beta), "_ratio10_project"),
        "hybrid_density_diversity",
        fraction = 0.10,
        mode = "fast",
        density_alpha = alpha,
        diversity_beta = beta
      )
      pos <- pos + 1L
    }
  }
  out
}

rare_protected_landmark_strategy_grid <- function() {
  list(
    landmark_subsample_strategy(
      "rare_protected_tail20_cluster25_ratio10_project",
      "rare_protected",
      fraction = 0.10,
      mode = "fast",
      rare_tail_fraction = 0.20,
      rare_tail_oversample = 0.40,
      rare_cluster_fraction = 0.25,
      rare_n_quantiles = 5L
    ),
    landmark_subsample_strategy(
      "rare_protected_tail10_cluster25_ratio10_project",
      "rare_protected",
      fraction = 0.10,
      mode = "fast",
      rare_tail_fraction = 0.10,
      rare_tail_oversample = 0.40,
      rare_cluster_fraction = 0.25,
      rare_n_quantiles = 5L
    ),
    landmark_subsample_strategy(
      "rare_protected_tail20_cluster40_ratio10_project",
      "rare_protected",
      fraction = 0.10,
      mode = "fast",
      rare_tail_fraction = 0.20,
      rare_tail_oversample = 0.50,
      rare_cluster_fraction = 0.40,
      rare_n_quantiles = 5L
    )
  )
}

landmark_projection_strategy <- function(projection_k,
                                         weight = c("inverse_distance", "softmax", "gaussian"),
                                         fraction = 0.10,
                                         selection = "projected_farthest") {
  weight <- match.arg(weight)
  list(
    id = paste0("landmark_projection_k", as.integer(projection_k), "_", weight),
    family = "landmark_projection",
    description = paste0(
      "Landmark embedding followed by kNN weighted-average projection with projection_k=",
      as.integer(projection_k), " and weights=", weight, "."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      landmark_count <- landmark_count_for_strategy(ctx, fraction = fraction)
      list(
        available = landmark_count >= 2L && landmark_count < ctx$n && projection_k <= landmark_count,
        message = paste0(
          "Landmark projection skipped because the landmark count is invalid or smaller than projection_k. landmarks=",
          landmark_count, ", projection_k=", projection_k
        )
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      landmark_fraction = fraction,
      landmark_selection = selection,
      projection_k = as.integer(projection_k),
      projection_weights = weight,
      interpolation = paste0("knn_", weight),
      refinement = "none",
      backend_scope = "cpu projection benchmark; GPU-specific kernels are not claimed here",
      standardize = FALSE,
      pca_dims = NULL
    ),
    run = function(ctx) run_landmark_projection_strategy(
      ctx,
      projection_k = projection_k,
      weight = weight,
      fraction = fraction,
      selection = selection
    )
  )
}

landmark_projection_strategy_grid <- function() {
  ks <- c(5L, 15L, 30L)
  weights <- c("inverse_distance", "softmax", "gaussian")
  out <- vector("list", length(ks) * length(weights))
  pos <- 1L
  for (projection_k in ks) {
    for (weight in weights) {
      out[[pos]] <- landmark_projection_strategy(projection_k, weight)
      pos <- pos + 1L
    }
  }
  out
}

landmark_projection_refinement_strategy <- function(refinement_epochs,
                                                    projection_k = NULL,
                                                    fraction = 0.10,
                                                    selection = "projected_farthest",
                                                    weight = "gaussian") {
  id <- paste0("landmark_projection_refine_e", as.integer(refinement_epochs))
  list(
    id = id,
    family = "landmark_projection_refinement",
    description = paste0(
      "Landmark embedding followed by Gaussian kNN projection and ",
      as.integer(refinement_epochs), " epochs of projection-graph refinement."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      landmark_count <- landmark_count_for_strategy(ctx, fraction = fraction)
      effective_projection_k <- if (is.null(projection_k)) ctx$k else as.integer(projection_k)
      list(
        available = landmark_count >= 2L &&
          landmark_count < ctx$n &&
          effective_projection_k <= landmark_count,
        message = paste0(
          "Projection refinement skipped because the landmark count is invalid or smaller than projection_k. landmarks=",
          landmark_count, ", projection_k=", effective_projection_k
        )
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      landmark_fraction = fraction,
      landmark_selection = selection,
      projection_k = if (is.null(projection_k)) as.integer(ctx$k) else as.integer(projection_k),
      projection_weights = weight,
      interpolation = paste0("knn_", weight),
      refinement = "projection",
      refinement_epochs = as.integer(refinement_epochs),
      backend_scope = "cpu projection-refinement benchmark; GPU-specific kernels are not claimed here",
      standardize = FALSE,
      pca_dims = NULL
    ),
    run = function(ctx) run_landmark_projection_refinement_strategy(
      ctx,
      refinement_epochs = refinement_epochs,
      projection_k = projection_k,
      weight = weight,
      fraction = fraction,
      selection = selection
    )
  )
}

landmark_projection_refinement_strategy_grid <- function() {
  lapply(c(20L, 50L, 100L), landmark_projection_refinement_strategy)
}

landmark_affine_projection_strategy <- function(projection_k,
                                                ridge = 1e-3,
                                                fraction = 0.10,
                                                selection = "projected_farthest",
                                                weight = "gaussian",
                                                clip_multiplier = 3,
                                                affine_blend = 0.25) {
  list(
    id = paste0("landmark_affine_k", as.integer(projection_k), "_ridge", gsub("\\.", "p", format(ridge, scientific = FALSE, trim = TRUE))),
    family = "landmark_affine_projection",
    description = paste0(
      "Landmark embedding followed by a ridge-regularized local affine projection from high-dimensional landmark neighbours to the embedding, projection_k=",
      as.integer(projection_k), ", ridge=", ridge, "."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      landmark_count <- landmark_count_for_strategy(ctx, fraction = fraction)
      list(
        available = landmark_count >= 2L && landmark_count < ctx$n && projection_k <= landmark_count,
        message = paste0(
          "Local affine projection skipped because the landmark count is invalid or smaller than projection_k. landmarks=",
          landmark_count, ", projection_k=", projection_k
        )
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      landmark_fraction = fraction,
      landmark_selection = selection,
      projection_k = as.integer(projection_k),
      interpolation = "local_affine",
      affine_solver = "dual_ridge_k_by_k",
      affine_weight = weight,
      affine_ridge = ridge,
      affine_clip_multiplier = clip_multiplier,
      affine_blend = affine_blend,
      refinement = "none",
      backend_scope = "cpu projection benchmark; GPU-specific kernels are not claimed here",
      standardize = FALSE,
      pca_dims = NULL
    ),
    run = function(ctx) run_landmark_affine_projection_strategy(
      ctx,
      projection_k = projection_k,
      ridge = ridge,
      fraction = fraction,
      selection = selection,
      weight = weight,
      clip_multiplier = clip_multiplier,
      affine_blend = affine_blend
    )
  )
}

landmark_affine_projection_strategy_grid <- function() {
  list(
    landmark_affine_projection_strategy(15L, ridge = 1e-3),
    landmark_affine_projection_strategy(30L, ridge = 1e-3)
  )
}

normalize_initial_layout <- function(layout, method, seed, target_scale = 1) {
  layout <- as.matrix(layout)
  storage.mode(layout) <- "double"
  n <- nrow(layout)
  if (ncol(layout) < 2L) {
    set.seed(seed + 2719L)
    pad <- matrix(stats::rnorm(n * (2L - ncol(layout))), nrow = n)
    layout <- cbind(layout, pad)
  }
  layout <- layout[, seq_len(2L), drop = FALSE]
  for (j in seq_len(2L)) {
    layout[, j] <- layout[, j] - mean(layout[, j])
    sd_j <- stats::sd(layout[, j])
    if (!is.finite(sd_j) || sd_j <= 0) {
      set.seed(seed + 3000L + j)
      layout[, j] <- stats::rnorm(n)
      sd_j <- stats::sd(layout[, j])
    }
    layout[, j] <- layout[, j] / max(sd_j, .Machine$double.eps)
  }
  target_scale <- as.numeric(target_scale)[1L]
  if (!is.finite(target_scale) || target_scale <= 0) target_scale <- 1
  layout <- layout * target_scale
  colnames(layout) <- paste0(toupper(method), seq_len(2L))
  layout
}

random_initial_layout <- function(n, method, seed) {
  set.seed(seed + 401L)
  out <- matrix(stats::rnorm(n * 2L), nrow = n, ncol = 2L)
  attr(out, "init_backend") <- "cpu_random"
  normalize_initial_layout(out, method, seed)
}

pca_initial_layout <- function(ctx) {
  pca <- fastEmbedR:::fastpls_rsvd_pca_scores(
    ctx$x,
    rank = min(2L, ctx$p, ctx$n - 1L),
    seed = ctx$seed,
    backend = "cpu"
  )
  out <- normalize_initial_layout(pca$scores, ctx$method, ctx$seed)
  attr(out, "init_backend") <- safe_character(pca$backend, "cpu_pca")
  attr(out, "init_backend_reason") <- safe_character(pca$backend_reason)
  attr(out, "init_pca_method") <- safe_character(pca$method, "rsvd")
  attr(out, "init_pca_oversample") <- safe_number(pca$oversample)
  attr(out, "init_pca_power") <- safe_number(pca$power)
  out
}

spectral_initial_layout <- function(ctx, knn, spectral_n_iter = NULL) {
  spectral_n_iter <- if (is.null(spectral_n_iter)) {
    if (ctx$n >= 10000L) 80L else 50L
  } else {
    as.integer(spectral_n_iter)
  }
  out <- fastEmbedR:::spectral_knn_init(
    knn$indices,
    knn$distances,
    n_components = 2L,
    spectral_n_iter = spectral_n_iter,
    seed = ctx$seed,
    backend = "cpu"
  )
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- "cpu_spectral"
  attr(out, "init_spectral_n_iter") <- spectral_n_iter
  attr(out, "init_spectral_solver") <- "block_power_ritz_normalized_adjacency"
  attr(out, "init_spectral_graph") <- "fuzzy_knn_normalized_laplacian"
  out
}

spectral_sparse_normalized_adjacency <- function(knn) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("R package `Matrix` is required for sparse randomized spectral initialization.", call. = FALSE)
  }
  indices <- as.matrix(knn$indices)
  distances <- as.matrix(knn$distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  n <- nrow(indices)
  k <- ncol(indices)
  if (n < 3L || k < 1L) {
    stop("Sparse spectral initialization requires at least three rows and one neighbour.", call. = FALSE)
  }
  eps <- sqrt(.Machine$double.eps)
  row_ids <- rep(seq_len(n), each = k)
  col_ids <- as.integer(as.vector(t(indices)))
  dist <- as.numeric(as.vector(t(distances)))
  valid <- is.finite(col_ids) & is.finite(dist) & col_ids >= 1L & col_ids <= n &
    col_ids != row_ids & dist >= 0
  if (!any(valid)) {
    stop("Sparse spectral initialization could not build a non-empty graph.", call. = FALSE)
  }
  positive <- distances
  positive[!is.finite(positive) | positive <= eps] <- NA_real_
  sigma <- apply(positive, 1L, stats::median, na.rm = TRUE)
  fallback <- stats::median(sigma[is.finite(sigma) & sigma > eps], na.rm = TRUE)
  if (!is.finite(fallback) || fallback <= eps) {
    fallback <- mean(dist[valid & dist > eps], na.rm = TRUE)
  }
  if (!is.finite(fallback) || fallback <= eps) fallback <- 1
  sigma[!is.finite(sigma) | sigma <= eps] <- fallback
  w <- exp(-0.5 * (pmax(0, dist[valid]) / pmax(eps, sigma[row_ids[valid]]))^2)
  w[!is.finite(w) | w <= 0] <- eps
  adjacency <- Matrix::sparseMatrix(
    i = row_ids[valid],
    j = col_ids[valid],
    x = w,
    dims = c(n, n)
  )
  adjacency <- Matrix::drop0((adjacency + Matrix::t(adjacency)) * 0.5)
  degree <- Matrix::rowSums(adjacency)
  active <- is.finite(degree) & degree > 0
  if (sum(active) < 3L) {
    stop("Sparse spectral initialization graph has fewer than three active vertices.", call. = FALSE)
  }
  inv_sqrt_degree <- numeric(n)
  inv_sqrt_degree[active] <- 1 / sqrt(degree[active])
  normalized <- Matrix::Diagonal(n, x = inv_sqrt_degree) %*% adjacency %*%
    Matrix::Diagonal(n, x = inv_sqrt_degree)
  normalized <- Matrix::drop0(normalized)
  attr(normalized, "spectral_graph_nnz") <- length(normalized@x)
  attr(normalized, "spectral_graph_mean_degree") <- mean(degree[active])
  attr(normalized, "spectral_graph_active_fraction") <- mean(active)
  attr(normalized, "spectral_graph_degree") <- as.numeric(degree)
  normalized
}

spectral_sparse_vectors_to_layout <- function(vectors, values, ctx, backend, solver, graph) {
  vectors <- as.matrix(vectors)
  if (ncol(vectors) < 2L) {
    return(random_initial_layout(ctx$n, ctx$method, ctx$seed))
  }
  out <- normalize_initial_layout(vectors[, seq_len(2L), drop = FALSE], ctx$method, ctx$seed)
  attr(out, "init_backend") <- backend
  attr(out, "init_spectral_solver") <- solver
  attr(out, "init_spectral_graph") <- graph
  attr(out, "init_spectral_eigenvalues") <- paste(signif(values[seq_len(min(2L, length(values)))], 6), collapse = ";")
  out
}

spectral_irlba_initial_layout <- function(ctx, knn, spectral_n_iter = NULL) {
  if (!requireNamespace("irlba", quietly = TRUE)) {
    stop("R package `irlba` is not installed.", call. = FALSE)
  }
  adjacency <- spectral_sparse_normalized_adjacency(knn)
  n <- nrow(adjacency)
  if (n < 4L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  spectral_n_iter <- if (is.null(spectral_n_iter)) {
    if (ctx$n >= 10000L) 120L else 80L
  } else {
    as.integer(spectral_n_iter)
  }
  set.seed(ctx$seed + 9101L)
  fit <- irlba::irlba(
    adjacency,
    nv = min(3L, n - 1L),
    nu = min(3L, n - 1L),
    maxit = spectral_n_iter,
    tol = 1e-4,
    reorth = TRUE
  )
  vectors <- fit$u
  values <- fit$d
  if (ncol(vectors) >= 3L) {
    vectors <- vectors[, 2:3, drop = FALSE]
    values <- values[2:3]
  } else {
    vectors <- vectors[, seq_len(min(2L, ncol(vectors))), drop = FALSE]
    values <- values[seq_len(min(2L, length(values)))]
  }
  out <- spectral_sparse_vectors_to_layout(
    vectors,
    values,
    ctx,
    backend = "cpu_irlba_sparse_spectral",
    solver = "irlba_sparse_normalized_adjacency_svd",
    graph = "symmetric_local_gaussian_sparse_normalized_adjacency"
  )
  attr(out, "init_spectral_n_iter") <- spectral_n_iter
  attr(out, "init_spectral_graph_nnz") <- safe_number(attr(adjacency, "spectral_graph_nnz"))
  attr(out, "init_spectral_graph_active_fraction") <- safe_number(attr(adjacency, "spectral_graph_active_fraction"))
  out
}

spectral_rspectra_initial_layout <- function(ctx, knn, spectral_n_iter = NULL) {
  if (!requireNamespace("RSpectra", quietly = TRUE)) {
    stop("R package `RSpectra` is not installed.", call. = FALSE)
  }
  adjacency <- spectral_sparse_normalized_adjacency(knn)
  n <- nrow(adjacency)
  if (n < 4L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  spectral_n_iter <- if (is.null(spectral_n_iter)) {
    if (ctx$n >= 10000L) 120L else 80L
  } else {
    as.integer(spectral_n_iter)
  }
  set.seed(ctx$seed + 9201L)
  fit <- RSpectra::eigs_sym(
    adjacency,
    k = min(3L, n - 1L),
    which = "LA",
    opts = list(maxitr = spectral_n_iter, tol = 1e-4)
  )
  ord <- order(fit$values, decreasing = TRUE)
  vectors <- fit$vectors[, ord, drop = FALSE]
  values <- fit$values[ord]
  if (ncol(vectors) >= 3L) {
    vectors <- vectors[, 2:3, drop = FALSE]
    values <- values[2:3]
  } else {
    vectors <- vectors[, seq_len(min(2L, ncol(vectors))), drop = FALSE]
    values <- values[seq_len(min(2L, length(values)))]
  }
  out <- spectral_sparse_vectors_to_layout(
    vectors,
    values,
    ctx,
    backend = "cpu_rspectra_sparse_spectral",
    solver = "rspectra_sparse_normalized_adjacency_eigs",
    graph = "symmetric_local_gaussian_sparse_normalized_adjacency"
  )
  attr(out, "init_spectral_n_iter") <- spectral_n_iter
  attr(out, "init_spectral_graph_nnz") <- safe_number(attr(adjacency, "spectral_graph_nnz"))
  attr(out, "init_spectral_graph_active_fraction") <- safe_number(attr(adjacency, "spectral_graph_active_fraction"))
  out
}

diffusion_block_power_eigen <- function(adjacency, rank = 3L, n_iter = 100L, seed = 1L) {
  n <- nrow(adjacency)
  rank <- max(2L, min(as.integer(rank), n - 1L))
  n_iter <- max(5L, as.integer(n_iter))
  set.seed(seed + 9301L)
  q <- matrix(stats::rnorm(n * rank), nrow = n, ncol = rank)
  q <- qr.Q(qr(q))
  for (iter in seq_len(n_iter)) {
    q <- as.matrix(adjacency %*% q)
    q <- qr.Q(qr(q))
  }
  aq <- as.matrix(adjacency %*% q)
  small <- crossprod(q, aq)
  small <- (small + t(small)) * 0.5
  eig <- eigen(small, symmetric = TRUE)
  ord <- order(eig$values, decreasing = TRUE)
  vectors <- q %*% eig$vectors[, ord, drop = FALSE]
  list(values = eig$values[ord], vectors = vectors[, seq_len(rank), drop = FALSE])
}

diffusion_map_initial_layout <- function(ctx,
                                         knn,
                                         diffusion_time = 1,
                                         diffusion_n_iter = NULL) {
  adjacency <- spectral_sparse_normalized_adjacency(knn)
  n <- nrow(adjacency)
  if (n < 4L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  diffusion_n_iter <- if (is.null(diffusion_n_iter)) {
    if (ctx$n >= 10000L) 120L else 80L
  } else {
    as.integer(diffusion_n_iter)
  }
  rank <- min(3L, n - 1L)
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    set.seed(ctx$seed + 9401L)
    fit <- RSpectra::eigs_sym(
      adjacency,
      k = rank,
      which = "LA",
      opts = list(maxitr = diffusion_n_iter, tol = 1e-4)
    )
    ord <- order(fit$values, decreasing = TRUE)
    values <- fit$values[ord]
    vectors <- fit$vectors[, ord, drop = FALSE]
    solver <- "rspectra_diffusion_map_conjugate_markov"
    backend <- "cpu_rspectra_diffusion_map"
  } else {
    fit <- diffusion_block_power_eigen(
      adjacency,
      rank = rank,
      n_iter = diffusion_n_iter,
      seed = ctx$seed
    )
    values <- fit$values
    vectors <- fit$vectors
    solver <- "block_power_ritz_diffusion_map_conjugate_markov"
    backend <- "cpu_block_power_diffusion_map"
  }
  if (ncol(vectors) >= 3L) {
    chosen <- 2:3
  } else {
    chosen <- seq_len(min(2L, ncol(vectors)))
  }
  if (length(chosen) < 2L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  degree <- attr(adjacency, "spectral_graph_degree")
  if (is.null(degree) || length(degree) != n) degree <- rep(1, n)
  degree[!is.finite(degree) | degree <= 0] <- stats::median(degree[is.finite(degree) & degree > 0], na.rm = TRUE)
  degree[!is.finite(degree) | degree <= 0] <- 1
  diffusion_time <- as.numeric(diffusion_time)[1L]
  if (!is.finite(diffusion_time) || diffusion_time < 0) diffusion_time <- 1
  phi <- sweep(vectors[, chosen, drop = FALSE], 1L, sqrt(pmax(degree, .Machine$double.eps)), "/")
  lambda <- values[chosen]
  scale <- sign(lambda) * (abs(lambda) ^ diffusion_time)
  scale[!is.finite(scale)] <- 1
  out <- sweep(phi, 2L, scale, "*")
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- backend
  attr(out, "init_diffusion_time") <- diffusion_time
  attr(out, "init_diffusion_n_iter") <- diffusion_n_iter
  attr(out, "init_diffusion_solver") <- solver
  attr(out, "init_diffusion_graph") <- "symmetric_local_gaussian_conjugate_markov_knn"
  attr(out, "init_diffusion_eigenvalues") <- paste(signif(lambda, 6), collapse = ";")
  attr(out, "init_diffusion_graph_nnz") <- safe_number(attr(adjacency, "spectral_graph_nnz"))
  attr(out, "init_diffusion_graph_active_fraction") <- safe_number(attr(adjacency, "spectral_graph_active_fraction"))
  out
}

laplacian_eigenmaps_initial_layout <- function(ctx,
                                               knn,
                                               laplacian_n_iter = NULL,
                                               normalized = TRUE) {
  adjacency <- spectral_sparse_normalized_adjacency(knn)
  n <- nrow(adjacency)
  if (n < 4L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  laplacian_n_iter <- if (is.null(laplacian_n_iter)) {
    if (ctx$n >= 10000L) 120L else 80L
  } else {
    as.integer(laplacian_n_iter)
  }
  rank <- min(3L, n - 1L)
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    laplacian <- Matrix::Diagonal(n) - adjacency
    set.seed(ctx$seed + 9501L)
    fit <- RSpectra::eigs_sym(
      laplacian,
      k = rank,
      which = "SA",
      opts = list(maxitr = laplacian_n_iter, tol = 1e-4)
    )
    ord <- order(fit$values, decreasing = FALSE)
    values <- fit$values[ord]
    vectors <- fit$vectors[, ord, drop = FALSE]
    solver <- "rspectra_sparse_normalized_laplacian_smallest_eigen"
    backend <- "cpu_rspectra_laplacian_eigenmaps"
  } else {
    fit <- diffusion_block_power_eigen(
      adjacency,
      rank = rank,
      n_iter = laplacian_n_iter,
      seed = ctx$seed
    )
    ord <- order(fit$values, decreasing = TRUE)
    adjacency_values <- fit$values[ord]
    values <- 1 - adjacency_values
    vectors <- fit$vectors[, ord, drop = FALSE]
    solver <- "block_power_ritz_laplacian_eigenmaps_from_normalized_adjacency"
    backend <- "cpu_block_power_laplacian_eigenmaps"
  }
  if (ncol(vectors) >= 3L) {
    chosen <- 2:3
  } else {
    chosen <- seq_len(min(2L, ncol(vectors)))
  }
  if (length(chosen) < 2L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  out <- vectors[, chosen, drop = FALSE]
  degree <- attr(adjacency, "spectral_graph_degree")
  if (isTRUE(normalized) && !is.null(degree) && length(degree) == n) {
    degree[!is.finite(degree) | degree <= 0] <- stats::median(degree[is.finite(degree) & degree > 0], na.rm = TRUE)
    degree[!is.finite(degree) | degree <= 0] <- 1
    out <- sweep(out, 1L, sqrt(pmax(degree, .Machine$double.eps)), "/")
  }
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- backend
  attr(out, "init_laplacian_n_iter") <- laplacian_n_iter
  attr(out, "init_laplacian_solver") <- solver
  attr(out, "init_laplacian_graph") <- "symmetric_local_gaussian_sparse_normalized_laplacian"
  attr(out, "init_laplacian_eigenvalues") <- paste(signif(values[chosen], 6), collapse = ";")
  attr(out, "init_laplacian_graph_nnz") <- safe_number(attr(adjacency, "spectral_graph_nnz"))
  attr(out, "init_laplacian_graph_active_fraction") <- safe_number(attr(adjacency, "spectral_graph_active_fraction"))
  attr(out, "init_laplacian_normalized_coordinates") <- isTRUE(normalized)
  out
}

spectral_exact_initial_layout <- function(ctx, knn) {
  indices <- as.matrix(knn$indices)
  distances <- as.matrix(knn$distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  n <- nrow(indices)
  if (n < 3L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  w <- matrix(0, nrow = n, ncol = n)
  for (i in seq_len(n)) {
    idx <- as.integer(indices[i, ])
    dst <- as.numeric(distances[i, ])
    valid <- is.finite(idx) & is.finite(dst) & idx >= 1L & idx <= n & idx != i & dst >= 0
    if (!any(valid)) next
    d <- dst[valid]
    j <- idx[valid]
    scale <- stats::median(d[d > sqrt(.Machine$double.eps)])
    if (!is.finite(scale) || scale <= 0) scale <- max(mean(d), sqrt(.Machine$double.eps))
    row_weight <- exp(-0.5 * (pmax(0, d) / scale)^2)
    row_weight[!is.finite(row_weight) | row_weight <= 0] <- .Machine$double.eps
    w[cbind(rep.int(i, length(j)), j)] <- pmax(w[cbind(rep.int(i, length(j)), j)], row_weight)
  }
  w <- pmax(w, t(w))
  degree <- rowSums(w)
  active <- degree > 0 & is.finite(degree)
  if (sum(active) < 3L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  inv_sqrt_degree <- numeric(n)
  inv_sqrt_degree[active] <- 1 / sqrt(degree[active])
  normalized_adjacency <- w * tcrossprod(inv_sqrt_degree)
  laplacian <- diag(1, n)
  laplacian[active, active] <- diag(1, sum(active)) - normalized_adjacency[active, active, drop = FALSE]
  eig <- eigen(laplacian, symmetric = TRUE)
  order_idx <- order(eig$values, seq_along(eig$values))
  chosen <- order_idx[seq_len(min(length(order_idx), 3L))]
  chosen <- chosen[-1L]
  if (length(chosen) < 2L) {
    return(random_initial_layout(n, ctx$method, ctx$seed))
  }
  out <- eig$vectors[, chosen[seq_len(2L)], drop = FALSE]
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- "cpu_dense_laplacian_eigen"
  attr(out, "init_spectral_solver") <- "dense_normalized_laplacian_eigen"
  attr(out, "init_spectral_graph") <- "symmetric_local_gaussian_knn"
  attr(out, "init_spectral_eigenvalues") <- paste(signif(eig$values[chosen[seq_len(2L)]], 6), collapse = ";")
  attr(out, "init_spectral_exact_max_n") <- 2500L
  out
}

spectral_nystrom_initial_layout <- function(ctx,
                                            n_epochs,
                                            fraction = NULL,
                                            count = NULL,
                                            projection_k = NULL,
                                            selection = "projected_farthest",
                                            weight = "gaussian") {
  nystrom_count <- if (is.null(fraction) && is.null(count)) {
    min(ctx$n - 1L, max(ctx$k + 1L, 64L, as.integer(ceiling(sqrt(ctx$n) * 8L))))
  } else {
    landmark_count_for_strategy(ctx, fraction = fraction, count = count)
  }
  nystrom_count <- max(3L, min(as.integer(nystrom_count), ctx$n - 1L))
  landmark_indices <- landmark_indices_for_strategy(ctx, selection, nystrom_count)
  selection_used <- attr(landmark_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_landmarks <- ctx$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- max(2L, min(as.integer(ctx$k), nrow(x_landmarks) - 1L))
  projection_k <- if (is.null(projection_k)) min(15L, nrow(x_landmarks)) else projection_k
  projection_k <- max(1L, min(as.integer(projection_k), nrow(x_landmarks)))
  landmark_ctx <- ctx
  landmark_ctx$x <- x_landmarks
  landmark_ctx$n <- nrow(x_landmarks)
  landmark_ctx$p <- ncol(x_landmarks)
  landmark_ctx$seed <- ctx$seed + 9701L
  landmark_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = "cpu")
    landmark_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, nrow(x_landmarks), landmark_neighbors)
  })[["elapsed"]]
  landmark_spectral_time <- system.time({
    landmark_layout <- spectral_initial_layout(
      landmark_ctx,
      landmark_knn,
      spectral_n_iter = if (ctx$n >= 10000L) 60L else 40L
    )
  })[["elapsed"]]
  projection_nn <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_landmarks,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    out <- weighted_landmark_projection_layout(
      landmark_layout,
      landmark_indices,
      projection_nn,
      ctx$n,
      weight = weight
    )
  })[["elapsed"]]
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- "cpu_nystrom_spectral"
  attr(out, "init_spectral_solver") <- "nystrom_landmark_block_power_extension"
  attr(out, "init_spectral_graph") <- "landmark_fuzzy_knn_plus_gaussian_out_of_sample_extension"
  attr(out, "init_spectral_nystrom_landmarks") <- length(landmark_indices)
  attr(out, "init_spectral_nystrom_fraction") <- length(landmark_indices) / ctx$n
  attr(out, "init_spectral_nystrom_selection_requested") <- selection
  attr(out, "init_spectral_nystrom_selection_used") <- selection_used
  attr(out, "init_spectral_nystrom_projection_k") <- projection_k
  attr(out, "init_spectral_nystrom_weight") <- weight
  attr(out, "init_spectral_nystrom_landmark_knn_time_sec") <- as.numeric(landmark_knn_time)
  attr(out, "init_spectral_nystrom_landmark_spectral_time_sec") <- as.numeric(landmark_spectral_time)
  attr(out, "init_spectral_nystrom_projection_time_sec") <- as.numeric(projection_time)
  attr(out, "init_projection_backend") <- safe_character(attr(projection_nn, "backend"), "cpu")
  attr(out, "init_projection_weight") <- weight
  attr(out, "init_projection_k") <- projection_k
  out
}

landmark_projection_initial_layout <- function(ctx,
                                               n_epochs,
                                               fraction = 0.10,
                                               projection_k = NULL,
                                               selection = "projected_farthest",
                                               weight = "gaussian") {
  count <- landmark_count_for_strategy(ctx, fraction = fraction)
  landmark_indices <- landmark_indices_for_strategy(ctx, selection, count)
  selection_used <- attr(landmark_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_landmarks <- ctx$x[landmark_indices, , drop = FALSE]
  landmark_neighbors <- max(1L, min(as.integer(ctx$k), nrow(x_landmarks) - 1L))
  projection_k <- if (is.null(projection_k)) ctx$k else projection_k
  projection_k <- max(1L, min(as.integer(projection_k), nrow(x_landmarks)))
  landmark_epochs <- max(5L, min(30L, floor(as.integer(n_epochs) / 2L)))

  landmark_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_landmarks, k = landmark_neighbors, backend = "cpu")
    landmark_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, nrow(x_landmarks), landmark_neighbors)
  })[["elapsed"]]
  landmark_embedding_time <- system.time({
    landmark_layout <- fastEmbedR:::embed_from_knn(
      ctx$method,
      landmark_knn$indices,
      landmark_knn$distances,
      2L,
      ctx$seed,
      "cpu",
      method_quality(ctx$method, "fast"),
      FALSE,
      n_epochs = landmark_epochs
    )
    landmark_layout <- coerce_layout(landmark_layout, nrow(x_landmarks))
  })[["elapsed"]]
  projection_nn <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_landmarks,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    out <- weighted_landmark_projection_layout(
      landmark_layout,
      landmark_indices,
      projection_nn,
      ctx$n,
      weight = weight
    )
  })[["elapsed"]]
  out <- normalize_initial_layout(out, ctx$method, ctx$seed)
  attr(out, "init_backend") <- "cpu_landmark_projection"
  attr(out, "init_landmark_selection_requested") <- selection
  attr(out, "init_landmark_selection_used") <- selection_used
  attr(out, "init_landmark_n") <- length(landmark_indices)
  attr(out, "init_landmark_fraction") <- length(landmark_indices) / ctx$n
  attr(out, "init_projection_k") <- projection_k
  attr(out, "init_projection_weight") <- weight
  attr(out, "init_landmark_epochs") <- landmark_epochs
  attr(out, "init_landmark_knn_time_sec") <- as.numeric(landmark_knn_time)
  attr(out, "init_landmark_embedding_time_sec") <- as.numeric(landmark_embedding_time)
  attr(out, "init_projection_time_sec") <- as.numeric(projection_time)
  attr(out, "init_projection_backend") <- safe_character(attr(projection_nn, "backend"), "cpu")
  out
}

build_initial_layout <- function(ctx, knn, init_strategy, n_epochs) {
  if (identical(init_strategy, "random")) {
    return(random_initial_layout(ctx$n, ctx$method, ctx$seed))
  }
  if (identical(init_strategy, "pca")) {
    return(pca_initial_layout(ctx))
  }
  if (identical(init_strategy, "spectral")) {
    return(spectral_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "spectral_irlba")) {
    return(spectral_irlba_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "spectral_rspectra")) {
    return(spectral_rspectra_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "diffusion_map")) {
    return(diffusion_map_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "laplacian_eigenmaps")) {
    return(laplacian_eigenmaps_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "spectral_exact")) {
    return(spectral_exact_initial_layout(ctx, knn))
  }
  if (identical(init_strategy, "spectral_nystrom")) {
    return(spectral_nystrom_initial_layout(ctx, n_epochs = n_epochs))
  }
  if (identical(init_strategy, "landmark_projection")) {
    return(landmark_projection_initial_layout(ctx, n_epochs = n_epochs))
  }
  stop("Unknown initialization strategy: ", init_strategy, call. = FALSE)
}

run_optimizer_with_initial_layout <- function(ctx, knn, init_layout, n_epochs) {
  n_epochs <- max(1L, as.integer(n_epochs))
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ncol(knn$indices), "cpu")
    cfg$n_epochs <- n_epochs
    layout <- fastEmbedR:::knn_umap_refine_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    layout <- fastEmbedR:::set_embedding_colnames(layout, "UMAP")
    cfg$optimizer_backend <- "cpu_umap_refine"
    attr(layout, "fastEmbedR_config") <- cfg
    return(layout)
  }

  cfg <- fastEmbedR:::knn_embed_config(
    n = ctx$n,
    k = ncol(knn$indices),
    objective = ctx$method,
    quality = "fast",
    backend = "cpu"
  )
  cfg$n_epochs <- n_epochs
  if (identical(ctx$method, "tsne")) {
    layout <- fastEmbedR:::knn_tsne_neighbors_cpp(
      knn$indices,
      knn$distances,
      init_layout,
      as.integer(cfg$n_epochs),
      cfg$perplexity,
      cfg$theta,
      cfg$learning_rate,
      as.integer(cfg$stop_lying_iter),
      as.integer(cfg$mom_switch_iter),
      cfg$momentum,
      cfg$final_momentum,
      cfg$exaggeration_factor,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE
    )
    cfg$affinity_backend <- "cpu_rtsne"
    cfg$optimizer_backend <- "cpu_rtsne_neighbors"
  } else {
    layout <- fastEmbedR:::knn_objective_embed_cpp(
      knn$indices,
      knn$distances,
      ctx$method,
      init_layout,
      2L,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      as.integer(cfg$n_threads),
      as.integer(ctx$seed),
      FALSE,
      FALSE
    )
    cfg$optimizer_backend <- "cpu_knn_objective"
  }
  layout <- fastEmbedR:::set_embedding_colnames(layout, fastEmbedR:::objective_prefix(ctx$method))
  attr(layout, "fastEmbedR_config") <- cfg
  layout
}

run_initialization_strategy <- function(ctx,
                                        init_strategy,
                                        n_epochs = NULL) {
  n_epochs <- if (is.null(n_epochs)) ctx$short_epochs else as.integer(n_epochs)
  knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  init_record <- NULL
  init_time <- system.time({
    init_layout <- build_initial_layout(ctx, knn, init_strategy, n_epochs)
  })[["elapsed"]]
  optimizer_time <- system.time({
    layout <- run_optimizer_with_initial_layout(ctx, knn, init_layout, n_epochs)
    layout <- coerce_layout(layout, ctx$n)
  })[["elapsed"]]
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  fields <- list(
    init_strategy = init_strategy,
    init_backend = safe_character(attr(init_layout, "init_backend"), "cpu"),
    init_backend_reason = safe_character(attr(init_layout, "init_backend_reason")),
    init_time_sec = as.numeric(init_time),
    init_optimizer_epochs = as.integer(n_epochs),
    init_optimizer_time_sec = as.numeric(optimizer_time),
    init_scale = 1,
    init_spectral_n_iter = safe_number(attr(init_layout, "init_spectral_n_iter")),
    init_spectral_solver = safe_character(attr(init_layout, "init_spectral_solver")),
    init_spectral_graph = safe_character(attr(init_layout, "init_spectral_graph")),
    init_spectral_eigenvalues = safe_character(attr(init_layout, "init_spectral_eigenvalues")),
    init_spectral_exact_max_n = safe_number(attr(init_layout, "init_spectral_exact_max_n")),
    init_spectral_graph_nnz = safe_number(attr(init_layout, "init_spectral_graph_nnz")),
    init_spectral_graph_active_fraction = safe_number(attr(init_layout, "init_spectral_graph_active_fraction")),
    init_spectral_nystrom_landmarks = safe_number(attr(init_layout, "init_spectral_nystrom_landmarks")),
    init_spectral_nystrom_fraction = safe_number(attr(init_layout, "init_spectral_nystrom_fraction")),
    init_spectral_nystrom_projection_k = safe_number(attr(init_layout, "init_spectral_nystrom_projection_k")),
    init_spectral_nystrom_weight = safe_character(attr(init_layout, "init_spectral_nystrom_weight")),
    init_spectral_nystrom_selection_requested = safe_character(attr(init_layout, "init_spectral_nystrom_selection_requested")),
    init_spectral_nystrom_selection_used = safe_character(attr(init_layout, "init_spectral_nystrom_selection_used")),
    init_spectral_nystrom_landmark_knn_time_sec = safe_number(attr(init_layout, "init_spectral_nystrom_landmark_knn_time_sec")),
    init_spectral_nystrom_landmark_spectral_time_sec = safe_number(attr(init_layout, "init_spectral_nystrom_landmark_spectral_time_sec")),
    init_spectral_nystrom_projection_time_sec = safe_number(attr(init_layout, "init_spectral_nystrom_projection_time_sec")),
    init_diffusion_time = safe_number(attr(init_layout, "init_diffusion_time")),
    init_diffusion_n_iter = safe_number(attr(init_layout, "init_diffusion_n_iter")),
    init_diffusion_solver = safe_character(attr(init_layout, "init_diffusion_solver")),
    init_diffusion_graph = safe_character(attr(init_layout, "init_diffusion_graph")),
    init_diffusion_eigenvalues = safe_character(attr(init_layout, "init_diffusion_eigenvalues")),
    init_diffusion_graph_nnz = safe_number(attr(init_layout, "init_diffusion_graph_nnz")),
    init_diffusion_graph_active_fraction = safe_number(attr(init_layout, "init_diffusion_graph_active_fraction")),
    init_laplacian_n_iter = safe_number(attr(init_layout, "init_laplacian_n_iter")),
    init_laplacian_solver = safe_character(attr(init_layout, "init_laplacian_solver")),
    init_laplacian_graph = safe_character(attr(init_layout, "init_laplacian_graph")),
    init_laplacian_eigenvalues = safe_character(attr(init_layout, "init_laplacian_eigenvalues")),
    init_laplacian_graph_nnz = safe_number(attr(init_layout, "init_laplacian_graph_nnz")),
    init_laplacian_graph_active_fraction = safe_number(attr(init_layout, "init_laplacian_graph_active_fraction")),
    init_laplacian_normalized_coordinates = safe_logical(attr(init_layout, "init_laplacian_normalized_coordinates")),
    init_pca_method = safe_character(attr(init_layout, "init_pca_method")),
    init_pca_oversample = safe_number(attr(init_layout, "init_pca_oversample")),
    init_pca_power = safe_number(attr(init_layout, "init_pca_power")),
    init_landmark_n = safe_number(attr(init_layout, "init_landmark_n")),
    init_landmark_fraction = safe_number(attr(init_layout, "init_landmark_fraction")),
    init_landmark_selection_requested = safe_character(attr(init_layout, "init_landmark_selection_requested")),
    init_landmark_selection_used = safe_character(attr(init_layout, "init_landmark_selection_used")),
    init_projection_k = safe_number(attr(init_layout, "init_projection_k")),
    init_projection_weight = safe_character(attr(init_layout, "init_projection_weight")),
    init_landmark_epochs = safe_number(attr(init_layout, "init_landmark_epochs")),
    init_landmark_knn_time_sec = safe_number(attr(init_layout, "init_landmark_knn_time_sec")),
    init_landmark_embedding_time_sec = safe_number(attr(init_layout, "init_landmark_embedding_time_sec")),
    init_projection_time_sec = safe_number(attr(init_layout, "init_projection_time_sec")),
    init_projection_backend = safe_character(attr(init_layout, "init_projection_backend"))
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

initialization_strategy <- function(init_strategy) {
  list(
    id = paste0("init_", init_strategy),
    family = "initialization",
    knn_cache_strategy_id = "initialization_shared_knn",
    description = paste0(
      "Use ", init_strategy,
      " as the initial layout, then run the same short CPU optimizer on the shared KNN graph."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      if (identical(init_strategy, "landmark_projection")) {
        landmark_count <- landmark_count_for_strategy(ctx, fraction = 0.10)
        return(list(
          available = landmark_count >= 2L && landmark_count < ctx$n && ctx$k <= landmark_count,
          message = paste0(
            "Landmark initialization skipped because the landmark count is invalid or smaller than k. landmarks=",
            landmark_count, ", k=", ctx$k
          )
        ))
      }
      if (identical(init_strategy, "spectral_exact")) {
        return(list(
          available = ctx$n <= 2500L,
          message = paste0(
            "Exact dense graph-Laplacian spectral initialization is limited to n <= 2500 in this benchmark. n=",
            ctx$n
          )
        ))
      }
      if (identical(init_strategy, "spectral_nystrom")) {
        landmark_count <- max(3L, min(ctx$n - 1L, max(ctx$k + 1L, 64L, as.integer(ceiling(sqrt(ctx$n) * 8L)))))
        return(list(
          available = landmark_count >= 3L && ctx$k <= landmark_count,
          message = paste0(
            "Nyström spectral initialization skipped because the landmark count is invalid or smaller than k. landmarks=",
            landmark_count, ", k=", ctx$k
          )
        ))
      }
      list(available = TRUE, message = NA_character_)
    },
    availability = function() {
      if (identical(init_strategy, "spectral_irlba")) {
        return(list(
          available = requireNamespace("Matrix", quietly = TRUE) && requireNamespace("irlba", quietly = TRUE),
          message = "R packages `Matrix` and `irlba` are required for init_spectral_irlba."
        ))
      }
      if (identical(init_strategy, "spectral_rspectra")) {
        return(list(
          available = requireNamespace("Matrix", quietly = TRUE) && requireNamespace("RSpectra", quietly = TRUE),
          message = "R packages `Matrix` and `RSpectra` are required for init_spectral_rspectra."
        ))
      }
      if (identical(init_strategy, "diffusion_map")) {
        return(list(
          available = requireNamespace("Matrix", quietly = TRUE),
          message = "R package `Matrix` is required for init_diffusion_map."
        ))
      }
      if (identical(init_strategy, "laplacian_eigenmaps")) {
        return(list(
          available = requireNamespace("Matrix", quietly = TRUE),
          message = "R package `Matrix` is required for init_laplacian_eigenmaps."
        ))
      }
      list(available = TRUE, message = NA_character_)
    },
    params = function(ctx) list(
      k = ctx$k,
      init = init_strategy,
      n_epochs = ctx$short_epochs,
      optimizer = if (identical(ctx$method, "tsne")) "rtsne_neighbors" else "native_knn_objective",
      spectral_solver = if (identical(init_strategy, "spectral")) {
        "block_power_ritz_normalized_adjacency"
      } else if (identical(init_strategy, "spectral_irlba")) {
        "irlba_sparse_normalized_adjacency_svd"
      } else if (identical(init_strategy, "spectral_rspectra")) {
        "rspectra_sparse_normalized_adjacency_eigs"
      } else if (identical(init_strategy, "diffusion_map")) {
        "diffusion_components_conjugate_markov_operator"
      } else if (identical(init_strategy, "laplacian_eigenmaps")) {
        "sparse_normalized_laplacian_eigenmaps"
      } else if (identical(init_strategy, "spectral_exact")) {
        "dense_normalized_laplacian_eigen"
      } else if (identical(init_strategy, "spectral_nystrom")) {
        "nystrom_landmark_block_power_extension"
      } else {
        NA_character_
      },
      backend_scope = "cpu initialization benchmark; GPU-specific initialization kernels are not claimed here",
      standardize = FALSE,
      pca_dims = NULL
    ),
    run = function(ctx) run_initialization_strategy(ctx, init_strategy)
  )
}

initialization_strategy_grid <- function() {
  lapply(
    c(
      "random", "pca", "spectral", "spectral_irlba", "spectral_rspectra",
      "diffusion_map", "laplacian_eigenmaps",
      "spectral_nystrom", "spectral_exact", "landmark_projection"
    ),
    initialization_strategy
  )
}

warm_start_embedding_cache <- new.env(parent = emptyenv())

warm_start_cache_key <- function(ctx, previous_epochs, previous_init) {
  paste(
    ctx$dataset_name,
    paste0("n", ctx$n),
    paste0("p", ctx$p),
    ctx$method,
    paste0("k", ctx$k),
    paste0("seed", ctx$seed),
    paste0("previous_epochs", previous_epochs),
    paste0("previous_init", previous_init),
    sep = "|"
  )
}

warm_start_previous_embedding <- function(ctx, knn, previous_epochs, previous_init) {
  previous_epochs <- max(1L, as.integer(previous_epochs))
  key <- warm_start_cache_key(ctx, previous_epochs, previous_init)
  if (exists(key, envir = warm_start_embedding_cache, inherits = FALSE)) {
    cached <- get(key, envir = warm_start_embedding_cache, inherits = FALSE)
    cached$layout <- as.matrix(cached$layout)
    cached$cache_hit <- TRUE
    cached$this_row_setup_time_sec <- 0
    return(cached)
  }
  init_time <- system.time({
    init_layout <- build_initial_layout(ctx, knn, previous_init, previous_epochs)
  })[["elapsed"]]
  embedding_time <- system.time({
    layout <- run_optimizer_with_initial_layout(ctx, knn, init_layout, previous_epochs)
    layout <- coerce_layout(layout, ctx$n)
  })[["elapsed"]]
  record <- list(
    layout = layout,
    key = key,
    cache_hit = FALSE,
    previous_init = previous_init,
    previous_epochs = previous_epochs,
    previous_init_time_sec = as.numeric(init_time),
    previous_embedding_time_sec = as.numeric(embedding_time),
    previous_build_time_sec = as.numeric(init_time) + as.numeric(embedding_time),
    this_row_setup_time_sec = as.numeric(init_time) + as.numeric(embedding_time)
  )
  assign(key, record, envir = warm_start_embedding_cache)
  record
}

run_warm_start_strategy <- function(ctx,
                                    refinement_epochs,
                                    previous_epochs = NULL,
                                    previous_init = "pca") {
  knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  refinement_epochs <- max(1L, as.integer(refinement_epochs))
  previous_epochs <- if (is.null(previous_epochs)) {
    max(1L, as.integer(ctx$short_epochs))
  } else {
    max(1L, as.integer(previous_epochs))
  }
  previous <- warm_start_previous_embedding(ctx, knn, previous_epochs, previous_init)
  init_layout <- as.matrix(previous$layout)
  refinement_time <- system.time({
    layout <- run_optimizer_with_initial_layout(ctx, knn, init_layout, refinement_epochs)
    layout <- coerce_layout(layout, ctx$n)
  })[["elapsed"]]
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  fields <- list(
    warm_start_enabled = TRUE,
    warm_start_cache_hit = isTRUE(previous$cache_hit),
    warm_start_cache_key = safe_character(previous$key),
    warm_start_previous_init = safe_character(previous$previous_init),
    warm_start_previous_epochs = safe_number(previous$previous_epochs),
    warm_start_refinement_epochs = as.integer(refinement_epochs),
    warm_start_previous_init_time_sec = safe_number(previous$previous_init_time_sec),
    warm_start_previous_embedding_time_sec = safe_number(previous$previous_embedding_time_sec),
    warm_start_previous_build_time_sec = safe_number(previous$previous_build_time_sec),
    warm_start_this_row_setup_time_sec = safe_number(previous$this_row_setup_time_sec),
    warm_start_refinement_time_sec = as.numeric(refinement_time),
    warm_start_total_if_cache_miss_sec = safe_number(previous$previous_build_time_sec) + as.numeric(refinement_time),
    warm_start_reuse_mode = "previous_embedding_cache",
    warm_start_use_case = "parameter_grid_or_interactive_reuse",
    warm_start_parameter_delta = "optimizer_continuation_proxy",
    warm_start_bias_risk = "biased_toward_previous_solution"
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

warm_start_strategy <- function(refinement_epochs, previous_init = "pca") {
  refinement_epochs <- as.integer(refinement_epochs)
  list(
    id = paste0("warm_start_refine_e", refinement_epochs),
    family = "warm_start",
    knn_cache_strategy_id = "warm_start_shared_knn",
    description = paste0(
      "Reuse a cached previous embedding and refine for ",
      refinement_epochs,
      " epochs. Intended for parameter-grid and interactive update workflows."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    params = function(ctx) list(
      k = ctx$k,
      warm_start = TRUE,
      previous_init = previous_init,
      previous_epochs = max(1L, as.integer(ctx$short_epochs)),
      refinement_epochs = refinement_epochs,
      reuse_mode = "previous_embedding_cache",
      expected_gain = "large_after_cache_hit",
      risk = "bias_toward_previous_solution",
      backend_scope = "cpu warm-start benchmark; no GPU warm-start path is claimed here"
    ),
    run = function(ctx) run_warm_start_strategy(
      ctx,
      refinement_epochs = refinement_epochs,
      previous_init = previous_init
    )
  )
}

warm_start_strategy_grid <- function() {
  list(
    warm_start_strategy(10L),
    warm_start_strategy(20L),
    warm_start_strategy(40L)
  )
}

default_epoch_budget <- function(ctx) {
  if (identical(ctx$method, "umap")) {
    cfg <- fastEmbedR:::fast_knn_umap_config(ctx$n, ctx$k, ctx$backend)
  } else {
    cfg <- fastEmbedR:::knn_embed_config(
      n = ctx$n,
      k = ctx$k,
      objective = ctx$method,
      quality = method_quality(ctx$method, "auto"),
      backend = ctx$backend
    )
  }
  as.integer(cfg$n_epochs)
}

run_epoch_budget_strategy <- function(ctx, epochs = NULL) {
  requested <- if (is.null(epochs)) "default" else as.character(as.integer(epochs))
  default_epochs <- default_epoch_budget(ctx)
  quality <- method_quality(ctx$method, "auto")
  if (identical(ctx$method, "umap")) {
    layout <- fastEmbedR:::fast_knn_umap_core(
      ctx$knn,
      backend = ctx$backend,
      seed = ctx$seed,
      n_epochs = if (is.null(epochs)) NULL else as.integer(epochs)
    )
  } else {
    layout <- fastEmbedR:::knn_embed_core(
      ctx$knn,
      objective = ctx$method,
      quality = quality,
      backend = ctx$backend,
      seed = ctx$seed,
      n_epochs = if (is.null(epochs)) NULL else as.integer(epochs)
    )
  }
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  effective_epochs <- safe_number(cfg$n_epochs)
  fields <- list(
    epoch_budget_enabled = TRUE,
    epoch_budget_requested = requested,
    epoch_budget_effective = effective_epochs,
    epoch_budget_default_epochs = default_epochs,
    epoch_budget_ratio_to_default = if (is.finite(effective_epochs) && default_epochs > 0) {
      effective_epochs / default_epochs
    } else {
      NA_real_
    },
    epoch_budget_quality = safe_character(cfg$quality, quality),
    epoch_budget_tsne_mode = safe_character(cfg$tsne_mode),
    epoch_budget_optimizer_backend = safe_character(cfg$optimizer_backend, cfg$backend),
    epoch_budget_speed_quality_tradeoff = "lower epochs reduce runtime but may reduce local/global quality",
    epoch_budget_is_default = is.null(epochs)
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

epoch_budget_strategy <- function(epochs = NULL) {
  is_default <- is.null(epochs)
  id <- if (is_default) "epoch_budget_default" else paste0("epoch_budget_e", as.integer(epochs))
  list(
    id = id,
    family = "optimization_budget",
    knn_cache_strategy_id = "epoch_budget_shared_knn",
    description = if (is_default) {
      "Default optimizer epoch budget using the same KNN graph."
    } else {
      paste0("Fixed optimizer budget with n_epochs = ", as.integer(epochs), " using the same KNN graph.")
    },
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && backend %in% c("cpu", "cuda", "metal"),
    params = function(ctx) {
      default_epochs <- default_epoch_budget(ctx)
      effective <- if (is_default) default_epochs else as.integer(epochs)
      list(
        k = ctx$k,
        epochs = if (is_default) "default" else as.integer(epochs),
        default_epochs = default_epochs,
        ratio_to_default = effective / default_epochs,
        quality = method_quality(ctx$method, "auto"),
        variable_under_test = "optimizer_epochs",
        fixed = "same KNN graph, method, backend, seed, and quality preset",
        report = "quality-speed tradeoff"
      )
    },
    run = function(ctx) run_epoch_budget_strategy(ctx, epochs = epochs)
  )
}

epoch_budget_strategy_grid <- function() {
  c(
    lapply(c(50L, 100L, 200L, 500L), epoch_budget_strategy),
    list(epoch_budget_strategy(NULL))
  )
}

early_stop_default_max_epochs <- function(ctx) {
  default <- tryCatch(default_epoch_budget(ctx), error = function(e) NA_integer_)
  if (!is.finite(default) || default <= 0L) default <- as.integer(ctx$short_epochs)
  as.integer(min(500L, max(200L, as.integer(ctx$short_epochs), default)))
}

early_stop_default_tolerance <- function(criterion) {
  switch(
    safe_character(criterion, "displacement"),
    displacement = 0.002,
    trustworthiness = 0.001,
    neighbour_stability = 0.002,
    neighbor_stability = 0.002,
    combined = 0.002,
    0.002
  )
}

early_stop_trace <- function(x, digits = 6L) {
  if (length(x) == 0L) return(NA_character_)
  values <- vapply(x, function(value) {
    if (!is.finite(safe_number(value))) "NA" else formatC(safe_number(value), digits = digits, format = "fg")
  }, character(1L))
  paste(values, collapse = "|")
}

early_stop_sample_indices <- function(ctx, sample_size) {
  n <- as.integer(ctx$n)
  sample_size <- max(2L, min(as.integer(sample_size), n))
  if (sample_size >= n) return(seq_len(n))
  set.seed(as.integer(ctx$seed) + 87011L)
  sort(sample.int(n, sample_size))
}

embedding_mean_displacement <- function(previous, current) {
  previous <- as.matrix(previous)
  current <- as.matrix(current)
  if (!identical(dim(previous), dim(current))) return(NA_real_)
  delta <- sqrt(rowSums((current - previous)^2))
  centered <- sweep(current, 2L, colMeans(current), "-")
  scale <- stats::median(sqrt(rowSums(centered^2)), na.rm = TRUE)
  if (!is.finite(scale) || scale <= 0) {
    finite_current <- abs(current[is.finite(current)])
    scale <- if (length(finite_current) == 0L) 1 else stats::median(finite_current, na.rm = TRUE)
  }
  if (!is.finite(scale) || scale <= 0) scale <- 1
  mean(delta[is.finite(delta)], na.rm = TRUE) / scale
}

embedding_neighbour_indices <- function(layout, k) {
  layout <- as.matrix(layout)
  n <- nrow(layout)
  k <- max(1L, min(as.integer(k), n - 1L))
  raw <- tryCatch(
    fastEmbedR::nn(layout, layout, k + 1L, backend = "cpu"),
    error = function(e) fastEmbedR::nn(layout, k = k + 1L, backend = "cpu")
  )
  cols <- knn_neighbor_cols(raw)
  cols <- cols[seq_len(min(length(cols), k))]
  if (length(cols) == 0L) return(matrix(integer(), nrow = n, ncol = 0L))
  raw$indices[, cols, drop = FALSE]
}

embedding_neighbour_overlap <- function(previous, current, k, sample_indices) {
  previous <- as.matrix(previous)
  current <- as.matrix(current)
  sample_indices <- as.integer(sample_indices)
  sample_indices <- sample_indices[is.finite(sample_indices) & sample_indices >= 1L & sample_indices <= nrow(current)]
  if (length(sample_indices) < 3L) return(NA_real_)
  k <- max(1L, min(as.integer(k), length(sample_indices) - 1L))
  a <- embedding_neighbour_indices(previous[sample_indices, , drop = FALSE], k)
  b <- embedding_neighbour_indices(current[sample_indices, , drop = FALSE], k)
  if (ncol(a) == 0L || ncol(b) == 0L) return(NA_real_)
  overlaps <- numeric(nrow(a))
  for (i in seq_len(nrow(a))) {
    ai <- unique(as.integer(a[i, ]))
    bi <- unique(as.integer(b[i, ]))
    ai <- ai[is.finite(ai) & ai >= 1L & ai <= length(sample_indices)]
    bi <- bi[is.finite(bi) & bi >= 1L & bi <= length(sample_indices)]
    denom <- max(1L, min(length(ai), length(bi), k))
    overlaps[i] <- length(intersect(ai, bi)) / denom
  }
  mean(overlaps[is.finite(overlaps)], na.rm = TRUE)
}

early_stop_trustworthiness <- function(ctx, layout, sample_indices, k) {
  sample_indices <- as.integer(sample_indices)
  sample_indices <- sample_indices[is.finite(sample_indices) & sample_indices >= 1L & sample_indices <= ctx$n]
  if (length(sample_indices) < 3L) return(NA_real_)
  x_sample <- ctx$x[sample_indices, , drop = FALSE]
  layout_sample <- as.matrix(layout)[sample_indices, , drop = FALSE]
  labels_sample <- if (is.null(ctx$labels)) NULL else ctx$labels[sample_indices]
  k <- max(1L, min(as.integer(k), nrow(x_sample) - 1L))
  metric <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x_sample,
      layout_sample,
      labels = labels_sample,
      k = k,
      sample_size_for_global_metrics = nrow(x_sample),
      sample_size_for_local_metrics = nrow(x_sample),
      use_cache = FALSE,
      seed = ctx$seed,
      method = ctx$method,
      backend = ctx$backend,
      dataset = ctx$dataset_name
    ),
    error = function(e) NULL
  )
  if (is.null(metric) || !("trustworthiness" %in% names(metric))) return(NA_real_)
  safe_number(metric$trustworthiness)
}

early_stop_plateau <- function(criterion,
                               displacement,
                               trust_delta,
                               neighbour_delta,
                               neighbour_stability,
                               tolerance,
                               chunk_index) {
  criterion <- safe_character(criterion, "displacement")
  if (identical(criterion, "neighbor_stability")) criterion <- "neighbour_stability"
  if (identical(criterion, "displacement")) {
    return(is.finite(displacement) && displacement <= tolerance)
  }
  if (identical(criterion, "trustworthiness")) {
    return(chunk_index > 1L && is.finite(trust_delta) && trust_delta <= tolerance)
  }
  if (identical(criterion, "neighbour_stability")) {
    return(chunk_index > 1L && is.finite(neighbour_delta) &&
      (neighbour_delta <= tolerance || (is.finite(neighbour_stability) && (1 - neighbour_stability) <= tolerance)))
  }
  if (identical(criterion, "combined")) {
    disp_ok <- is.finite(displacement) && displacement <= tolerance
    trust_ok <- chunk_index > 1L && is.finite(trust_delta) && trust_delta <= tolerance
    neigh_ok <- chunk_index > 1L && is.finite(neighbour_delta) &&
      (neighbour_delta <= tolerance || (is.finite(neighbour_stability) && (1 - neighbour_stability) <= tolerance))
    return((disp_ok && (trust_ok || neigh_ok)) || (trust_ok && neigh_ok))
  }
  FALSE
}

run_early_stopping_strategy <- function(ctx,
                                        criterion,
                                        chunk_epochs = 25L,
                                        max_epochs = NULL,
                                        patience = 2L,
                                        tolerance = NULL,
                                        init_strategy = "pca",
                                        monitor_k = 15L,
                                        monitor_sample_size = NULL) {
  criterion <- safe_character(criterion, "displacement")
  if (identical(criterion, "neighbor_stability")) criterion <- "neighbour_stability"
  if (identical(criterion, "loss_change")) {
    stop("Loss-change early stopping is not supported because the native optimizers do not expose per-epoch loss.", call. = FALSE)
  }
  full_knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  chunk_epochs <- max(1L, as.integer(chunk_epochs))
  max_epochs <- if (is.null(max_epochs)) early_stop_default_max_epochs(ctx) else as.integer(max_epochs)
  max_epochs <- max(chunk_epochs, max_epochs)
  patience <- max(1L, as.integer(patience))
  tolerance <- if (is.null(tolerance)) early_stop_default_tolerance(criterion) else as.numeric(tolerance)
  if (!is.finite(tolerance) || tolerance < 0) tolerance <- early_stop_default_tolerance(criterion)
  monitor_sample_size <- if (is.null(monitor_sample_size)) min(500L, ctx$n) else as.integer(monitor_sample_size)
  sample_indices <- early_stop_sample_indices(ctx, monitor_sample_size)
  monitor_k <- max(1L, min(as.integer(monitor_k), length(sample_indices) - 1L, ctx$n - 1L))
  if (monitor_k < 1L) stop("Early stopping monitor needs at least three sampled points.", call. = FALSE)

  init_used <- init_strategy
  init_time <- system.time({
    init_layout <- tryCatch(
      build_initial_layout(ctx, full_knn, init_strategy, chunk_epochs),
      error = function(e) {
        init_used <<- "random"
        random_initial_layout(ctx$n, ctx$method, ctx$seed)
      }
    )
    init_layout <- coerce_layout(init_layout, ctx$n)
  })[["elapsed"]]

  current <- init_layout
  previous_trust <- NA_real_
  previous_stability <- NA_real_
  bad_count <- 0L
  epochs_completed <- 0L
  chunks_run <- 0L
  stop_status <- "max_epochs"
  stop_reason <- "maximum_epoch_budget_reached"
  optimizer_time <- 0
  displacement_trace <- numeric()
  trust_trace <- numeric()
  trust_delta_trace <- numeric()
  stability_trace <- numeric()
  stability_delta_trace <- numeric()
  epochs_trace <- numeric()

  while (epochs_completed < max_epochs) {
    remaining <- max_epochs - epochs_completed
    this_chunk <- min(chunk_epochs, remaining)
    previous <- current
    chunk_time <- system.time({
      current <- run_optimizer_with_initial_layout(ctx, full_knn, previous, this_chunk)
      current <- coerce_layout(current, ctx$n)
    })[["elapsed"]]
    optimizer_time <- optimizer_time + as.numeric(chunk_time)
    epochs_completed <- epochs_completed + this_chunk
    chunks_run <- chunks_run + 1L

    displacement <- embedding_mean_displacement(previous, current)
    trust <- if (criterion %in% c("trustworthiness", "combined")) {
      early_stop_trustworthiness(ctx, current, sample_indices, monitor_k)
    } else {
      NA_real_
    }
    stability <- if (criterion %in% c("neighbour_stability", "combined")) {
      embedding_neighbour_overlap(previous, current, monitor_k, sample_indices)
    } else {
      NA_real_
    }
    trust_delta <- if (is.finite(trust) && is.finite(previous_trust)) trust - previous_trust else NA_real_
    stability_delta <- if (is.finite(stability) && is.finite(previous_stability)) stability - previous_stability else NA_real_
    previous_trust <- if (is.finite(trust)) trust else previous_trust
    previous_stability <- if (is.finite(stability)) stability else previous_stability

    displacement_trace <- c(displacement_trace, displacement)
    trust_trace <- c(trust_trace, trust)
    trust_delta_trace <- c(trust_delta_trace, trust_delta)
    stability_trace <- c(stability_trace, stability)
    stability_delta_trace <- c(stability_delta_trace, stability_delta)
    epochs_trace <- c(epochs_trace, epochs_completed)

    plateau <- early_stop_plateau(
      criterion,
      displacement = displacement,
      trust_delta = trust_delta,
      neighbour_delta = stability_delta,
      neighbour_stability = stability,
      tolerance = tolerance,
      chunk_index = chunks_run
    )
    if (isTRUE(plateau)) {
      bad_count <- bad_count + 1L
    } else {
      bad_count <- 0L
    }
    if (bad_count >= patience && chunks_run >= max(2L, patience)) {
      stop_status <- "stopped"
      stop_reason <- paste0(criterion, "_plateau")
      break
    }
  }

  cfg <- attr(current, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  fields <- list(
    early_stop_enabled = TRUE,
    early_stop_criterion = criterion,
    early_stop_status = stop_status,
    early_stop_reason = stop_reason,
    early_stop_max_epochs = safe_number(max_epochs),
    early_stop_chunk_epochs = safe_number(chunk_epochs),
    early_stop_epochs_run = safe_number(epochs_completed),
    early_stop_chunks_run = safe_number(chunks_run),
    early_stop_patience = safe_number(patience),
    early_stop_tolerance = safe_number(tolerance),
    early_stop_displacement_final = safe_number(tail(displacement_trace, 1L)),
    early_stop_trustworthiness_final = safe_number(tail(trust_trace, 1L)),
    early_stop_trustworthiness_delta_final = safe_number(tail(trust_delta_trace, 1L)),
    early_stop_neighbour_stability_final = safe_number(tail(stability_trace, 1L)),
    early_stop_neighbour_stability_delta_final = safe_number(tail(stability_delta_trace, 1L)),
    early_stop_monitor_sample_size = safe_number(length(sample_indices)),
    early_stop_monitor_k = safe_number(monitor_k),
    early_stop_init_strategy = init_used,
    early_stop_init_time_sec = as.numeric(init_time),
    early_stop_optimizer_time_sec = safe_number(optimizer_time),
    early_stop_loss_available = FALSE,
    early_stop_loss_reason = "native_optimizer_loss_callback_unavailable",
    early_stop_chunked_optimizer = TRUE,
    early_stop_trace_epochs = early_stop_trace(epochs_trace, digits = 0L),
    early_stop_trace_displacement = early_stop_trace(displacement_trace),
    early_stop_trace_trustworthiness = early_stop_trace(trust_trace),
    early_stop_trace_trustworthiness_delta = early_stop_trace(trust_delta_trace),
    early_stop_trace_neighbour_stability = early_stop_trace(stability_trace),
    early_stop_trace_neighbour_stability_delta = early_stop_trace(stability_delta_trace),
    early_stop_backend_scope = "cpu chunked benchmark monitor; native callback support is not claimed",
    early_stop_risk = "chunked restarts approximate early stopping until native per-epoch callbacks are available"
  )
  attr(current, "fastEmbedR_config") <- c(cfg, fields)
  current
}

early_stopping_strategy <- function(criterion,
                                    chunk_epochs = 25L,
                                    max_epochs = NULL,
                                    patience = 2L,
                                    tolerance = NULL) {
  criterion <- safe_character(criterion, "displacement")
  if (identical(criterion, "neighbor_stability")) criterion <- "neighbour_stability"
  label <- gsub("[^a-z0-9]+", "_", criterion)
  id <- paste0("early_stop_", label, "_c", as.integer(chunk_epochs), "_p", as.integer(patience))
  if (identical(criterion, "loss_change")) id <- "early_stop_loss_change"
  list(
    id = id,
    family = "early_stopping",
    knn_cache_strategy_id = "early_stopping_shared_knn",
    early_stop_criterion = criterion,
    early_stop_chunk_epochs = as.integer(chunk_epochs),
    early_stop_max_epochs = if (is.null(max_epochs)) NA_real_ else as.integer(max_epochs),
    early_stop_patience = as.integer(patience),
    early_stop_tolerance = if (is.null(tolerance)) NA_real_ else as.numeric(tolerance),
    description = paste0("Chunked optimizer early stopping using the ", criterion, " monitor."),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      if (identical(criterion, "loss_change")) {
        return(list(
          available = FALSE,
          message = "Loss-change early stopping is not supported because the native optimizers do not expose per-epoch loss."
        ))
      }
      list(available = TRUE, message = NA_character_)
    },
    params = function(ctx) list(
      k = ctx$k,
      criterion = criterion,
      chunk_epochs = as.integer(chunk_epochs),
      max_epochs = if (is.null(max_epochs)) early_stop_default_max_epochs(ctx) else as.integer(max_epochs),
      patience = as.integer(patience),
      tolerance = if (is.null(tolerance)) early_stop_default_tolerance(criterion) else as.numeric(tolerance),
      monitor_k = min(15L, ctx$k, ctx$n - 1L),
      monitor_sample_size = min(500L, ctx$n),
      init = "pca",
      loss_callback_available = FALSE,
      backend_scope = "cpu chunked benchmark monitor"
    ),
    run = function(ctx) run_early_stopping_strategy(
      ctx,
      criterion = criterion,
      chunk_epochs = chunk_epochs,
      max_epochs = max_epochs,
      patience = patience,
      tolerance = tolerance
    )
  )
}

early_stopping_strategy_grid <- function() {
  list(
    early_stopping_strategy("displacement", chunk_epochs = 25L, patience = 2L),
    early_stopping_strategy("trustworthiness", chunk_epochs = 25L, patience = 2L),
    early_stopping_strategy("neighbour_stability", chunk_epochs = 25L, patience = 2L),
    early_stopping_strategy("combined", chunk_epochs = 25L, patience = 2L),
    early_stopping_strategy("loss_change", chunk_epochs = 25L, patience = 2L)
  )
}

coarse_to_fine_fields <- function(ctx,
                                  mode,
                                  selection_requested = NA_character_,
                                  selection_used = NA_character_,
                                  count_requested = NA_real_,
                                  fraction_requested = NA_real_,
                                  coarse_n = NA_real_,
                                  coarse_k = NA_real_,
                                  projection_k = NA_real_,
                                  projection_weight = NA_character_,
                                  coarse_epochs = NA_real_,
                                  refinement_epochs = NA_real_,
                                  selection_time = NA_real_,
                                  coarse_knn_time = NA_real_,
                                  coarse_embedding_time = NA_real_,
                                  projection_time = NA_real_,
                                  refinement_time = NA_real_,
                                  projection_layout = NULL,
                                  projection_backend = NA_character_,
                                  refinement_backend = NA_character_,
                                  init_backend = NA_character_) {
  entropy <- if (is.null(projection_layout)) NA_real_ else safe_number(attr(projection_layout, "projection_weight_entropy_mean"))
  zero_fraction <- if (is.null(projection_layout)) NA_real_ else safe_number(attr(projection_layout, "projection_zero_neighbor_fraction"))
  bandwidth <- if (is.null(projection_layout)) NA_real_ else safe_number(attr(projection_layout, "projection_bandwidth_mean"))
  total_setup <- sum(
    c(selection_time, coarse_knn_time, coarse_embedding_time, projection_time),
    na.rm = TRUE
  )
  list(
    coarse_to_fine_enabled = TRUE,
    coarse_to_fine_mode = mode,
    coarse_to_fine_selection_requested = selection_requested,
    coarse_to_fine_selection_used = selection_used,
    coarse_to_fine_count_requested = safe_number(count_requested),
    coarse_to_fine_fraction_requested = safe_number(fraction_requested),
    coarse_to_fine_n = safe_number(coarse_n),
    coarse_to_fine_fraction = if (is.finite(coarse_n)) coarse_n / ctx$n else NA_real_,
    coarse_to_fine_k = safe_number(coarse_k),
    coarse_to_fine_projection_k = safe_number(projection_k),
    coarse_to_fine_projection_weight = safe_character(projection_weight),
    coarse_to_fine_coarse_epochs = safe_number(coarse_epochs),
    coarse_to_fine_refinement_epochs = safe_number(refinement_epochs),
    coarse_to_fine_selection_time_sec = safe_number(selection_time),
    coarse_to_fine_knn_time_sec = safe_number(coarse_knn_time),
    coarse_to_fine_embedding_time_sec = safe_number(coarse_embedding_time),
    coarse_to_fine_projection_time_sec = safe_number(projection_time),
    coarse_to_fine_refinement_time_sec = safe_number(refinement_time),
    coarse_to_fine_setup_time_sec = total_setup,
    coarse_to_fine_projection_entropy = entropy,
    coarse_to_fine_projection_zero_neighbor_fraction = zero_fraction,
    coarse_to_fine_projection_bandwidth_mean = bandwidth,
    coarse_to_fine_projection_backend = safe_character(projection_backend),
    coarse_to_fine_refinement_backend = safe_character(refinement_backend),
    coarse_to_fine_init_backend = safe_character(init_backend),
    coarse_to_fine_expected_gain = "fast_stable_initialization_then_short_full_refinement",
    coarse_to_fine_risk = "coarse_subset_or_low_resolution_graph_can_bias_global_layout"
  )
}

run_coarse_to_fine_subset_strategy <- function(ctx,
                                               fraction,
                                               refinement_epochs,
                                               coarse_epochs = NULL,
                                               projection_k = NULL,
                                               selection = "projected_farthest",
                                               weight = "gaussian") {
  full_knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  count <- landmark_count_for_strategy(ctx, fraction = fraction)
  selection_time <- system.time({
    coarse_indices <- landmark_indices_for_strategy(ctx, selection, count)
  })[["elapsed"]]
  selection_used <- attr(coarse_indices, "benchmark_selection")
  if (is.null(selection_used)) selection_used <- selection
  x_coarse <- ctx$x[coarse_indices, , drop = FALSE]
  coarse_n <- nrow(x_coarse)
  coarse_k <- max(1L, min(as.integer(ctx$k), coarse_n - 1L))
  projection_k <- if (is.null(projection_k)) min(as.integer(ctx$k), coarse_n) else projection_k
  projection_k <- max(1L, min(as.integer(projection_k), coarse_n))
  refinement_epochs <- max(1L, as.integer(refinement_epochs))
  coarse_epochs <- if (is.null(coarse_epochs)) {
    max(5L, min(as.integer(ctx$short_epochs), refinement_epochs))
  } else {
    max(1L, as.integer(coarse_epochs))
  }

  coarse_knn_time <- system.time({
    raw_knn <- fastEmbedR:::nn_without_self(x_coarse, k = coarse_k, backend = "cpu")
    coarse_knn <- fastEmbedR:::normalize_supplied_knn(raw_knn, coarse_n, coarse_k)
  })[["elapsed"]]
  coarse_embedding_time <- system.time({
    coarse_layout <- fastEmbedR:::embed_from_knn(
      ctx$method,
      coarse_knn$indices,
      coarse_knn$distances,
      2L,
      ctx$seed,
      "cpu",
      method_quality(ctx$method, "fast"),
      FALSE,
      n_epochs = coarse_epochs
    )
    coarse_layout <- coerce_layout(coarse_layout, coarse_n)
  })[["elapsed"]]

  projection_nn <- NULL
  projected_layout <- NULL
  projection_time <- system.time({
    projection_nn <- fastEmbedR::nn(
      x_coarse,
      ctx$x,
      k = projection_k,
      backend = "cpu"
    )
    projected_layout <- weighted_landmark_projection_layout(
      coarse_layout,
      coarse_indices,
      projection_nn,
      ctx$n,
      weight = weight
    )
  })[["elapsed"]]

  refinement_time <- system.time({
    layout <- run_optimizer_with_initial_layout(ctx, full_knn, projected_layout, refinement_epochs)
    layout <- coerce_layout(layout, ctx$n)
  })[["elapsed"]]
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  fields <- coarse_to_fine_fields(
    ctx,
    mode = "subset_embed_project_full_graph_refine",
    selection_requested = selection,
    selection_used = selection_used,
    count_requested = count,
    fraction_requested = fraction,
    coarse_n = length(coarse_indices),
    coarse_k = coarse_k,
    projection_k = projection_k,
    projection_weight = weight,
    coarse_epochs = coarse_epochs,
    refinement_epochs = refinement_epochs,
    selection_time = as.numeric(selection_time),
    coarse_knn_time = as.numeric(coarse_knn_time),
    coarse_embedding_time = as.numeric(coarse_embedding_time),
    projection_time = as.numeric(projection_time),
    refinement_time = as.numeric(refinement_time),
    projection_layout = projected_layout,
    projection_backend = safe_character(attr(projection_nn, "backend"), "cpu"),
    refinement_backend = cfg$optimizer_backend,
    init_backend = "cpu_subset_embedding_projection"
  )
  landmark_fields <- c(
    list(
      landmark_enabled = TRUE,
      landmark_approximation = "coarse_to_fine_subset",
      landmark_mode = "coarse_to_fine_full_refine",
      landmark_selection_requested = selection,
      landmark_selection_used = selection_used,
      landmark_count_requested = as.integer(count),
      landmark_fraction_requested = as.numeric(fraction),
      landmark_n = length(coarse_indices),
      landmark_fraction = length(coarse_indices) / ctx$n,
      landmark_projection_k = projection_k,
      landmark_interpolation = paste0("knn_", weight),
      landmark_projection_model = "coarse_subset_weighted_projection_then_full_graph_refinement",
      landmark_projection_weight = weight,
      landmark_projection_bandwidth_rule = safe_character(attr(projected_layout, "projection_bandwidth_rule")),
      landmark_projection_bandwidth_mean = safe_number(attr(projected_layout, "projection_bandwidth_mean")),
      landmark_projection_weight_entropy = safe_number(attr(projected_layout, "projection_weight_entropy_mean")),
      landmark_projection_zero_neighbor_fraction = safe_number(attr(projected_layout, "projection_zero_neighbor_fraction")),
      landmark_projection_time_sec = as.numeric(projection_time),
      landmark_projection_backend = safe_character(attr(projection_nn, "backend"), "cpu"),
      landmark_interpolation_backend = "cpu_r_weighted_average",
      landmark_refinement = "full_graph",
      landmark_refinement_epochs = refinement_epochs,
      landmark_refinement_time_sec = as.numeric(refinement_time),
      landmark_refinement_backend = safe_character(cfg$optimizer_backend, "cpu"),
      landmark_refinement_knn_backend = "full_knn_graph",
      landmark_landmark_knn_time_sec = as.numeric(coarse_knn_time),
      landmark_landmark_embedding_time_sec = as.numeric(coarse_embedding_time),
      subsample_strategy = selection,
      subsample_stratified = selection %in% c("stratified", "stratified_label", "stratified_cluster"),
      benchmark_forced_k = as.integer(ctx$k),
      benchmark_standardize = FALSE
    ),
    landmark_label_coverage_fields(ctx$labels, coarse_indices)
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields, landmark_fields)
  layout
}

run_coarse_to_fine_graph_strategy <- function(ctx,
                                              coarse_k,
                                              refinement_epochs,
                                              coarse_epochs = NULL,
                                              init_strategy = "pca") {
  full_knn <- fastEmbedR:::normalize_supplied_knn(ctx$knn, ctx$n, ctx$k)
  coarse_k <- max(1L, min(as.integer(coarse_k), ncol(full_knn$indices)))
  coarse_knn <- list(
    indices = full_knn$indices[, seq_len(coarse_k), drop = FALSE],
    distances = full_knn$distances[, seq_len(coarse_k), drop = FALSE]
  )
  coarse_knn <- fastEmbedR:::normalize_supplied_knn(coarse_knn, ctx$n, coarse_k)
  refinement_epochs <- max(1L, as.integer(refinement_epochs))
  coarse_epochs <- if (is.null(coarse_epochs)) {
    max(5L, min(as.integer(ctx$short_epochs), refinement_epochs))
  } else {
    max(1L, as.integer(coarse_epochs))
  }
  init_time <- system.time({
    init_layout <- build_initial_layout(ctx, coarse_knn, init_strategy, coarse_epochs)
  })[["elapsed"]]
  coarse_embedding_time <- system.time({
    coarse_layout <- run_optimizer_with_initial_layout(ctx, coarse_knn, init_layout, coarse_epochs)
    coarse_layout <- coerce_layout(coarse_layout, ctx$n)
  })[["elapsed"]]
  refinement_time <- system.time({
    layout <- run_optimizer_with_initial_layout(ctx, full_knn, coarse_layout, refinement_epochs)
    layout <- coerce_layout(layout, ctx$n)
  })[["elapsed"]]
  cfg <- attr(layout, "fastEmbedR_config")
  if (is.null(cfg)) cfg <- list()
  fields <- coarse_to_fine_fields(
    ctx,
    mode = "low_resolution_knn_graph_then_full_graph_refine",
    coarse_n = ctx$n,
    coarse_k = coarse_k,
    coarse_epochs = coarse_epochs,
    refinement_epochs = refinement_epochs,
    selection_time = as.numeric(init_time),
    coarse_knn_time = 0,
    coarse_embedding_time = as.numeric(coarse_embedding_time),
    projection_time = 0,
    refinement_time = as.numeric(refinement_time),
    refinement_backend = cfg$optimizer_backend,
    init_backend = paste0("cpu_", init_strategy, "_initial_layout")
  )
  attr(layout, "fastEmbedR_config") <- c(cfg, fields)
  layout
}

coarse_to_fine_subset_strategy <- function(fraction,
                                           refinement_epochs,
                                           selection = "projected_farthest",
                                           weight = "gaussian",
                                           coarse_epochs = NULL) {
  fraction <- as.numeric(fraction)[1L]
  refinement_epochs <- as.integer(refinement_epochs)
  id <- paste0("coarse_to_fine_subset_f", nndescent_label(fraction), "_e", refinement_epochs)
  list(
    id = id,
    family = "coarse_to_fine",
    knn_cache_strategy_id = "coarse_to_fine_shared_knn",
    description = paste0(
      "Embed a ", format(100 * fraction, trim = TRUE),
      "% coarse subset, project all points, then refine on the full KNN graph for ",
      refinement_epochs, " epochs."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      count <- landmark_count_for_strategy(ctx, fraction = fraction)
      list(
        available = count >= 2L && count < ctx$n,
        message = paste0("Coarse-to-fine subset skipped because the coarse subset is invalid. coarse_n=", count)
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      mode = "subset_embed_project_full_graph_refine",
      coarse_fraction = fraction,
      coarse_n = landmark_count_for_strategy(ctx, fraction = fraction),
      selection = selection,
      projection_weight = weight,
      coarse_epochs = if (is.null(coarse_epochs)) max(5L, min(as.integer(ctx$short_epochs), refinement_epochs)) else as.integer(coarse_epochs),
      refinement_epochs = refinement_epochs,
      backend_scope = "cpu coarse-to-fine benchmark; GPU-specific kernels are not claimed here",
      expected = "fast and stable initialization",
      risk = "coarse subset can bias global layout or underrepresent rare classes"
    ),
    run = function(ctx) run_coarse_to_fine_subset_strategy(
      ctx,
      fraction = fraction,
      refinement_epochs = refinement_epochs,
      coarse_epochs = coarse_epochs,
      selection = selection,
      weight = weight
    )
  )
}

coarse_to_fine_graph_strategy <- function(coarse_k,
                                          refinement_epochs,
                                          coarse_epochs = NULL,
                                          init_strategy = "pca") {
  coarse_k <- as.integer(coarse_k)
  refinement_epochs <- as.integer(refinement_epochs)
  list(
    id = paste0("coarse_to_fine_graph_k", coarse_k, "_e", refinement_epochs),
    family = "coarse_to_fine",
    knn_cache_strategy_id = "coarse_to_fine_shared_knn",
    description = paste0(
      "Run an early pass on a low-resolution k=", coarse_k,
      " graph, then refine on the full KNN graph for ", refinement_epochs, " epochs."
    ),
    compatible = function(method, backend) method %in% c("umap", "tsne", "pacmap", "trimap", "localmap") && identical(backend, "cpu"),
    context_available = function(ctx) {
      list(
        available = ctx$k >= 2L && coarse_k < ctx$k,
        message = paste0("Low-resolution graph skipped because coarse_k must be smaller than k. coarse_k=", coarse_k, ", k=", ctx$k)
      )
    },
    params = function(ctx) list(
      k = ctx$k,
      mode = "low_resolution_knn_graph_then_full_graph_refine",
      coarse_k = min(coarse_k, ctx$k),
      init = init_strategy,
      coarse_epochs = if (is.null(coarse_epochs)) max(5L, min(as.integer(ctx$short_epochs), refinement_epochs)) else as.integer(coarse_epochs),
      refinement_epochs = refinement_epochs,
      backend_scope = "cpu coarse-to-fine benchmark; GPU-specific kernels are not claimed here",
      expected = "fast stable initialization from cheap low-resolution graph",
      risk = "low-resolution graph can lose local detail before refinement"
    ),
    run = function(ctx) run_coarse_to_fine_graph_strategy(
      ctx,
      coarse_k = coarse_k,
      refinement_epochs = refinement_epochs,
      coarse_epochs = coarse_epochs,
      init_strategy = init_strategy
    )
  )
}

coarse_to_fine_strategy_grid <- function() {
  list(
    coarse_to_fine_subset_strategy(0.05, 20L),
    coarse_to_fine_subset_strategy(0.10, 40L),
    coarse_to_fine_graph_strategy(10L, 20L)
  )
}

strategy_registry <- function() {
  c(
  list(
    list(
      id = "exact_knn",
      family = "exact_knn",
      description = "Baseline exact nearest-neighbor graph. Use as the small-dataset reference; slow for large data.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      knn_backend = function(ctx) "cpu",
      params = function(ctx) list(
        k = ctx$k,
        knn = "exact",
        knn_backend = "cpu",
        embedding_backend = ctx$backend,
        recommended_for = "small_datasets",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "kdtree_rnanoflann",
      family = "kdtree_knn",
      description = "Exact KD-tree KNN from Rnanoflann, optionally after PCA. Best for low-dimensional tabular data and PCA-reduced inputs.",
      availability = function() list(
        available = requireNamespace("Rnanoflann", quietly = TRUE),
        message = "R package `Rnanoflann` is not installed."
      ),
      context_available = function(ctx) list(
        available = identical(normalize_knn_metric(ctx$knn_metric), "euclidean"),
        message = "Rnanoflann KD-tree strategy currently supports only Euclidean KNN in this benchmark."
      ),
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      build_knn = kdtree_rnanoflann_knn,
      params = function(ctx) list(
        k = ctx$k,
        knn = "kdtree",
        implementation = "Rnanoflann",
        metric = normalize_knn_metric(ctx$knn_metric),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        works_best = "PCA 10-50 dimensions or low-dimensional tabular data",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "kdtree_fnn",
      family = "kdtree_knn",
      description = "Exact KD-tree KNN from FNN::get.knn, optionally after PCA. Best for low-dimensional tabular data and PCA-reduced inputs.",
      availability = function() list(
        available = requireNamespace("FNN", quietly = TRUE),
        message = "R package `FNN` is not installed."
      ),
      context_available = function(ctx) list(
        available = identical(normalize_knn_metric(ctx$knn_metric), "euclidean"),
        message = "FNN KD-tree strategy currently supports only Euclidean KNN in this benchmark."
      ),
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      build_knn = kdtree_fnn_knn,
      params = function(ctx) list(
        k = ctx$k,
        knn = "kdtree",
        implementation = "FNN",
        metric = normalize_knn_metric(ctx$knn_metric),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        works_best = "PCA 10-50 dimensions or low-dimensional tabular data",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "kdtree_sklearn",
      family = "kdtree_knn",
      description = "Exact KD-tree KNN from scikit-learn NearestNeighbors, optionally after PCA. Best for low-dimensional tabular data and PCA-reduced inputs.",
      availability = sklearn_available,
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      build_knn = kdtree_sklearn_knn,
      params = function(ctx) list(
        k = ctx$k,
        knn = "kdtree",
        implementation = "scikit-learn",
        metric = normalize_knn_metric(ctx$knn_metric),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        works_best = "PCA 10-50 dimensions or low-dimensional tabular data",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "balltree_sklearn",
      family = "balltree_knn",
      description = "Exact Ball-tree KNN from scikit-learn NearestNeighbors, optionally after PCA. Useful for medium dimensionality and non-Euclidean metrics supported by scikit-learn.",
      availability = sklearn_available,
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      build_knn = balltree_sklearn_knn,
      params = function(ctx) list(
        k = ctx$k,
        knn = "balltree",
        implementation = "scikit-learn",
        metric = normalize_knn_metric(ctx$knn_metric),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        works_best = "medium dimensionality or non-Euclidean metrics",
        compare_to = "exact_knn,kdtree_sklearn,brute_sklearn",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "brute_sklearn",
      family = "brute_force_knn",
      description = "Exact brute-force KNN from scikit-learn NearestNeighbors for direct KD-tree vs Ball-tree vs brute-force comparisons.",
      availability = sklearn_available,
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      build_knn = brute_sklearn_knn,
      params = function(ctx) list(
        k = ctx$k,
        knn = "brute_force",
        implementation = "scikit-learn",
        metric = normalize_knn_metric(ctx$knn_metric),
        pca_dims = min(ctx$pca_dims, ctx$p),
        embedding_backend = ctx$backend,
        compare_to = "exact_knn,kdtree_sklearn,balltree_sklearn",
        quality = method_quality(ctx$method, "auto")
      ),
      run = function(ctx) fastEmbedR::embed_knn(
        ctx$knn,
        method = ctx$method,
        quality = method_quality(ctx$method, "auto"),
        backend = ctx$backend,
        seed = ctx$seed
      )
    ),
    list(
      id = "full_knn_auto",
      family = "shared_knn",
      description = "Full embedding from one precomputed KNN graph; t-SNE uses quality='auto'.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      params = function(ctx) list(k = ctx$k, mode = "full_knn", quality = method_quality(ctx$method, "auto")),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = ctx$method, quality = method_quality(ctx$method, "auto"), backend = ctx$backend, seed = ctx$seed)
    ),
    list(
      id = "short_epochs_knn",
      family = "optimization_budget",
      description = "Same KNN graph but reduced optimizer epochs.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      params = function(ctx) list(k = ctx$k, n_epochs = ctx$short_epochs, quality = method_quality(ctx$method, "fast")),
      run = function(ctx) {
        if (identical(ctx$method, "umap")) {
          return(fastEmbedR:::fast_knn_umap_core(ctx$knn, backend = ctx$backend, seed = ctx$seed, n_epochs = ctx$short_epochs))
        }
        fastEmbedR:::knn_embed_core(ctx$knn, objective = ctx$method, quality = method_quality(ctx$method, "fast"), backend = ctx$backend, seed = ctx$seed, n_epochs = ctx$short_epochs)
      }
    ),
    epoch_budget_strategy(50L),
    epoch_budget_strategy(100L),
    epoch_budget_strategy(200L),
    epoch_budget_strategy(500L),
    epoch_budget_strategy(NULL),
    output_metric_strategy("euclidean"),
    output_metric_strategy("cosine"),
    output_metric_strategy("hyperbolic"),
    artificial_neighbor_penalty_strategy("mild_s0p15_f1p25_i8", penalty_strength = 0.15, n_iter = 8L, far_multiplier = 1.25),
    artificial_neighbor_penalty_strategy("balanced_s0p35_f1p00_i12", penalty_strength = 0.35, n_iter = 12L, far_multiplier = 1.00),
    artificial_neighbor_penalty_strategy("strong_s0p70_f0p85_i16", penalty_strength = 0.70, n_iter = 16L, far_multiplier = 0.85),
    false_neighbor_monitor_strategy("early_stop_c20_p1", chunk_epochs = 20L, max_chunks = 5L, tolerance = 0.001, patience = 1L, action = "early_stop"),
    false_neighbor_monitor_strategy("shrink_c20_p1", chunk_epochs = 20L, max_chunks = 5L, tolerance = 0.001, patience = 1L, action = "shrink_chunk"),
    false_neighbor_monitor_strategy("conservative_c15_p1", chunk_epochs = 15L, max_chunks = 6L, tolerance = 0, patience = 1L, action = "early_stop"),
    false_neighbor_monitor_strategy("guarded_refine_c20_p1", chunk_epochs = 20L, max_chunks = 4L, tolerance = 0.001, patience = 1L, action = "early_stop", start_mode = "full_auto"),
    false_neighbor_monitor_strategy("guarded_shrink_c20_p1", chunk_epochs = 20L, max_chunks = 4L, tolerance = 0.001, patience = 1L, action = "shrink_chunk", start_mode = "full_auto"),
    early_exaggeration_strategy(4, 0.10),
    early_exaggeration_strategy(4, 0.20),
    early_exaggeration_strategy(4, 0.30),
    early_exaggeration_strategy(8, 0.10),
    early_exaggeration_strategy(8, 0.20),
    early_exaggeration_strategy(8, 0.30),
    early_exaggeration_strategy(12, 0.10),
    early_exaggeration_strategy(12, 0.20),
    early_exaggeration_strategy(12, 0.30),
    late_exaggeration_strategy(2, 0.10),
    late_exaggeration_strategy(2, 0.20),
    late_exaggeration_strategy(2, 0.30),
    late_exaggeration_strategy(4, 0.10),
    late_exaggeration_strategy(4, 0.20),
    late_exaggeration_strategy(4, 0.30),
    late_exaggeration_strategy(8, 0.10),
    late_exaggeration_strategy(8, 0.20),
    late_exaggeration_strategy(8, 0.30),
    optimizer_schedule_strategy(
      "momentum_schedule_05_08",
      "momentum",
      "Momentum schedule transferred from t-SNE: momentum 0.5 before the switch iteration and 0.8 afterwards."
    ),
    optimizer_schedule_strategy(
      "optimizer_adam",
      "adam",
      "Adam optimizer transfer with beta1 = 0.9 and beta2 = 0.999. Uses a recorded conservative learning-rate multiplier for stability."
    ),
    optimizer_schedule_strategy(
      "optimizer_adagrad",
      "adagrad",
      "AdaGrad optimizer transfer with per-coordinate accumulated squared gradients and a conservative learning-rate multiplier."
    ),
    optimizer_schedule_strategy(
      "optimizer_nesterov",
      "nesterov",
      "Nesterov momentum transfer using the same 0.5 to 0.8 schedule as t-SNE momentum."
    ),
    learning_rate_scaling_strategy(
      "lr_fixed_1",
      "fixed_1",
      "Learning-rate scaling probe with lr = 1."
    ),
    learning_rate_scaling_strategy(
      "lr_fixed_10",
      "fixed_10",
      "Learning-rate scaling probe with lr = 10."
    ),
    learning_rate_scaling_strategy(
      "lr_n_over_12",
      "n_over_12",
      "Sample-size-scaled learning rate transferred from t-SNE-style heuristics: lr = n / 12."
    ),
    learning_rate_scaling_strategy(
      "lr_sqrt_n",
      "sqrt_n",
      "Sample-size-scaled learning rate: lr = sqrt(n)."
    ),
    learning_rate_scaling_strategy(
      "lr_method_default",
      "method_default",
      "Method-specific default learning rate from the current fastEmbedR configuration."
    ),
    adaptive_learning_rate_strategy("constant"),
    adaptive_learning_rate_strategy("linear_decay"),
    adaptive_learning_rate_strategy("cosine_decay"),
    adaptive_learning_rate_strategy("step_decay"),
    adaptive_learning_rate_strategy("adam"),
    adaptive_learning_rate_strategy("adagrad"),
    mini_batch_strategy(0.25, 4L),
    mini_batch_strategy(0.50, 4L),
    mini_batch_strategy(0.75, 4L),
    umap_negative_sampling_strategy(2),
    umap_negative_sampling_strategy(5),
    umap_negative_sampling_strategy(10),
    umap_negative_sampling_strategy(20),
    pacmap_far_repulsion_strategy(5),
    pacmap_far_repulsion_strategy(10),
    pacmap_far_repulsion_strategy(20),
    pacmap_phase_strategy(0.25, 1.35, 0.25, schedule = "short", epoch_multiplier = 0.50),
    pacmap_phase_strategy(0.33, 1.50, 0.35, schedule = "default", epoch_multiplier = 1.00),
    pacmap_phase_strategy(0.40, 1.75, 0.40, schedule = "long", epoch_multiplier = 2.00),
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.75, refreshes = 1L),
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.75, refreshes = 2L),
    pair_resampling_strategy("weighted_edges", keep_fraction = 0.50, refreshes = 2L),
	    pair_resampling_strategy("pacmap_pairs", keep_fraction = 1.00, refreshes = 2L, near_ratio = 0.50, mid_ratio = 0.30, far_ratio = 0.20),
	    triplet_aux_strategy(0.05, 1L),
	    triplet_aux_strategy(0.10, 1L),
	    triplet_aux_strategy(0.20, 2L),
	    structured_triplet_strategy(0.05, 2L, 1L, 0L),
	    structured_triplet_strategy(0.05, 2L, 1L, 1L),
	    structured_triplet_strategy(0.10, 3L, 2L, 1L),
	    global_random_triplet_strategy(0.02, 1L, 1L),
	    global_random_triplet_strategy(0.03, 2L, 2L),
	    global_random_triplet_strategy(0.05, 4L, 3L),
	    hard_negative_strategy(0.25, 0.75, 0.03, 1L, 1L, 0L),
	    hard_negative_strategy(0.50, 1.00, 0.05, 2L, 1L, 0L),
	    hard_negative_strategy(1.00, 0.75, 0.05, 2L, 1L, 1L),
    semihard_triplet_strategy(0.25, 0.75, 0.03, 1L, 1L, 0L),
    semihard_triplet_strategy(0.50, 0.75, 0.05, 2L, 1L, 0L),
    semihard_triplet_strategy(0.50, 1.00, 0.05, 2L, 2L, 1L)
  ),
  approx_triplet_mining_strategy_grid(),
  landmark_subsample_strategy_grid(),
  random_landmark_strategy_grid(),
  stratified_landmark_strategy_grid(),
  density_weighted_landmark_strategy_grid(),
  diversity_landmark_strategy_grid(),
  hybrid_density_diversity_landmark_strategy_grid(),
  rare_protected_landmark_strategy_grid(),
  landmark_projection_strategy_grid(),
  landmark_projection_refinement_strategy_grid(),
  landmark_affine_projection_strategy_grid(),
  initialization_strategy_grid(),
  warm_start_strategy_grid(),
  coarse_to_fine_strategy_grid(),
  early_stopping_strategy_grid(),
  list(
    trimap_triplet_proxy_strategy(
      "semihard_mild",
      "semi_hard_candidates",
      inlier_ratio = 0.75,
      semihard_ratio = 0.25,
      global_anchor_ratio = 0.00,
      semihard_distance_scale = 1.25,
      global_anchor_distance_scale = 3.00
    ),
    trimap_triplet_proxy_strategy(
      "semihard_balanced",
      "semi_hard_plus_global_anchors",
      inlier_ratio = 0.60,
      semihard_ratio = 0.30,
      global_anchor_ratio = 0.10,
      semihard_distance_scale = 1.40,
      global_anchor_distance_scale = 3.50
    ),
    trimap_triplet_proxy_strategy(
      "global_anchor",
      "global_anchor_triplets",
      inlier_ratio = 0.50,
      semihard_ratio = 0.20,
      global_anchor_ratio = 0.30,
      semihard_distance_scale = 1.60,
      global_anchor_distance_scale = 4.50
    ),
    list(
      id = "pca_then_full",
      family = "preprocessing",
      description = "RSVD/PCA preprocessing before KNN and full embedding; transferred across all objectives.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      params = function(ctx) list(pca_dims = min(ctx$pca_dims, ctx$p), mode = "accurate", quality = method_quality(ctx$method, "auto")),
      run = function(ctx) fastEmbedR:::embed(ctx$x, ctx$labels, method = ctx$method, mode = "accurate", pca_dims = min(ctx$pca_dims, ctx$p), backend = ctx$backend, quality = method_quality(ctx$method, "auto"), seed = ctx$seed, silhouette_sample = NULL, preserve_sample = NULL)
    ),
    list(
      id = "landmark_fast",
      family = "landmarking",
      description = "Landmark fit plus interpolation only.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      params = function(ctx) list(landmarks = ctx$landmarks, mode = "fast", k = ctx$k, quality = method_quality(ctx$method, "fast"), standardize = FALSE),
      run = function(ctx) patch_landmark_embedding_parameters(
        fastEmbedR:::embed(ctx$x, ctx$labels, method = ctx$method, mode = "fast", n_neighbors = ctx$k, standardize = FALSE, pca_dims = NULL, landmarks = ctx$landmarks, backend = ctx$backend, quality = method_quality(ctx$method, "fast"), seed = ctx$seed, silhouette_sample = NULL, preserve_sample = NULL),
        list(
          landmark_approximation = "landmark_fast",
          landmark_mode = "fast",
          landmark_selection_requested = "auto_or_cli",
          landmark_selection_used = "auto_or_cli",
          benchmark_forced_k = as.integer(ctx$k),
          benchmark_standardize = FALSE
        )
      )
    ),
    list(
      id = "landmark_balanced_refine",
      family = "landmarking",
      description = "Landmark fit plus bucketed local-KNN refinement.",
      compatible = function(method, backend) backend %in% c("cpu", "cuda", "metal"),
      params = function(ctx) list(landmarks = ctx$landmarks, mode = "balanced", k = ctx$k, quality = method_quality(ctx$method, "fast"), standardize = FALSE),
      run = function(ctx) patch_landmark_embedding_parameters(
        fastEmbedR:::embed(ctx$x, ctx$labels, method = ctx$method, mode = "balanced", n_neighbors = ctx$k, standardize = FALSE, pca_dims = NULL, landmarks = ctx$landmarks, backend = ctx$backend, quality = method_quality(ctx$method, "fast"), seed = ctx$seed, silhouette_sample = NULL, preserve_sample = NULL),
        list(
          landmark_approximation = "landmark_balanced_refine",
          landmark_mode = "balanced",
          landmark_selection_requested = "auto_or_cli",
          landmark_selection_used = "auto_or_cli",
          benchmark_forced_k = as.integer(ctx$k),
          benchmark_standardize = FALSE
        )
      )
    ),
    list(
      id = "tsne_rtsne_neighbors",
      family = "tsne_repulsion",
      description = "Rtsne-neighbors-style Barnes-Hut optimizer from KNN affinities.",
      compatible = function(method, backend) identical(method, "tsne") && identical(backend, "cpu"),
      params = function(ctx) list(k = ctx$k, quality = "fast", theta = 0.5),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "tsne", quality = "fast", backend = "cpu", seed = ctx$seed)
    ),
    tsne_barnes_hut_strategy(0),
    tsne_barnes_hut_strategy(0.3),
    tsne_barnes_hut_strategy(0.5),
    tsne_barnes_hut_strategy(0.8),
    list(
      id = "tsne_fitsne_fft",
      family = "tsne_repulsion",
      description = "FIt-SNE FFT optimizer from the same KNN-derived sparse affinities.",
      compatible = function(method, backend) identical(method, "tsne") && identical(backend, "cpu") && fast_tsne_available(),
      params = function(ctx) list(k = ctx$k, quality = "fft", fast_tsne = fastEmbedR:::fitsne_binary_path(required = FALSE)),
      run = function(ctx) fastEmbedR::embed_knn(ctx$knn, method = "tsne", quality = "fft", backend = "cpu", seed = ctx$seed)
    ),
    fitsne_fft_experimental_strategy(),
    fitsne_fft_grid_strategy(
      "baseline_o3_g1_m50",
      nterms = 3L,
      intervals_per_integer = 1,
      min_num_intervals = 50L,
      description = "Baseline FIt-SNE interpolation grid: order 3, default grid spacing, minimum 50 intervals."
    ),
    fitsne_fft_grid_strategy(
      "coarse_o3_g2_m25",
      nterms = 3L,
      intervals_per_integer = 2,
      min_num_intervals = 25L,
      description = "Coarser FIt-SNE interpolation grid: same interpolation order, larger grid spacing, lower minimum grid resolution."
    ),
    fitsne_fft_grid_strategy(
      "loworder_o2_g1_m50",
      nterms = 2L,
      intervals_per_integer = 1,
      min_num_intervals = 50L,
      description = "Lower-order FIt-SNE interpolation: order 2 with baseline grid spacing and minimum resolution."
    ),
    fitsne_fft_grid_strategy(
      "fine_o4_g075_m75",
      nterms = 4L,
      intervals_per_integer = 0.75,
      min_num_intervals = 75L,
      description = "Finer FIt-SNE interpolation grid: order 4, denser grid spacing, and higher minimum grid resolution."
    ),
    list(
      id = "rocm_requested",
      family = "backend_probe",
      description = "Explicit ROCm request; currently records unsupported unless a ROCm path is implemented.",
      compatible = function(method, backend) identical(backend, "rocm"),
      params = function(ctx) list(backend = "rocm"),
      run = function(ctx) stop("ROCm backend is not implemented in fastEmbedR yet.", call. = FALSE)
    )
  )
  )
}

empty_row <- function(ctx, strategy, status, error_message) {
  params <- tryCatch(strategy$params(ctx), error = function(e) list())
  ctx_default <- function(name, default) {
    value <- ctx[[name]]
    if (is.null(value)) default else value
  }
  data.frame(
    method = ctx$method,
    approximation = strategy$id,
    approximation_family = strategy$family,
    knn_reuse_mode = ctx$knn_reuse_mode,
    knn_cache_hit = ctx$knn_cache_hit,
    knn_graph_key = ctx$knn_graph_key,
    knn_graph_source = ctx$knn_graph_source,
    knn_disk_cache_hit = ctx$knn_disk_cache_hit,
    knn_disk_cache_format = ctx$knn_disk_cache_format,
    knn_disk_cache_path = ctx$knn_disk_cache_path,
    backend = ctx$backend,
    backend_used = NA_character_,
    gpu_transfer_policy = ctx_default("gpu_transfer_policy", NA_character_),
    gpu_transfer_backend = ctx_default("gpu_transfer_backend", NA_character_),
    gpu_transfer_host_to_device_count = ctx_default("gpu_transfer_host_to_device_count", NA_real_),
    gpu_transfer_device_to_host_count = ctx_default("gpu_transfer_device_to_host_count", NA_real_),
    gpu_transfer_host_to_device_bytes = ctx_default("gpu_transfer_host_to_device_bytes", NA_real_),
    gpu_transfer_device_to_host_bytes = ctx_default("gpu_transfer_device_to_host_bytes", NA_real_),
    gpu_transfer_host_to_device_mb = ctx_default("gpu_transfer_host_to_device_mb", NA_real_),
    gpu_transfer_device_to_host_mb = ctx_default("gpu_transfer_device_to_host_mb", NA_real_),
    gpu_transfer_knn_uploaded_once = ctx_default("gpu_transfer_knn_uploaded_once", NA),
    gpu_transfer_init_uploaded_once = ctx_default("gpu_transfer_init_uploaded_once", NA),
    gpu_transfer_embedding_returned_only_at_end = ctx_default("gpu_transfer_embedding_returned_only_at_end", NA),
    gpu_transfer_init_roundtrip = ctx_default("gpu_transfer_init_roundtrip", NA),
    gpu_transfer_graph_metadata_roundtrip = ctx_default("gpu_transfer_graph_metadata_roundtrip", NA),
    gpu_transfer_graph_prepared_on_device = ctx_default("gpu_transfer_graph_prepared_on_device", NA),
    gpu_transfer_note = ctx_default("gpu_transfer_note", NA_character_),
    dataset = ctx$dataset_name,
    n = ctx$n,
    p = ctx$p,
    seed = ctx$seed,
    parameter_settings = json_or_text(params),
    status = status,
    error_message = error_message,
    preprocessing_time_sec = NA_real_,
    knn_time_sec = ctx$knn_time_sec,
    knn_graph_time_sec = ctx$knn_graph_time_sec,
    knn_index_build_time_sec = ctx$knn_index_build_time_sec,
    knn_query_time_sec = ctx$knn_query_time_sec,
    knn_graph_index_build_time_sec = ctx$knn_graph_index_build_time_sec,
    knn_graph_query_time_sec = ctx$knn_graph_query_time_sec,
    knn_disk_load_time_sec = ctx$knn_disk_load_time_sec,
    knn_disk_save_time_sec = ctx$knn_disk_save_time_sec,
    knn_memory_mb = ctx$knn_memory_mb,
    knn_recall_at_k = ctx$knn_recall_at_k,
    knn_mean_distance_error = ctx$knn_mean_distance_error,
    knn_rank_correlation = ctx$knn_rank_correlation,
    knn_quality_sample_size = ctx$knn_quality_sample_size,
    graph_approximation = ctx$graph_approximation,
    graph_approximation_time_sec = ctx$graph_approximation_time_sec,
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_r_memory_mb = ctx$graph_sparse_r_memory_mb,
    graph_dense_knn_memory_mb = ctx$graph_dense_knn_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_sparse_r_memory_ratio = ctx$graph_sparse_r_memory_ratio,
    graph_sparse_prune_weight = ctx$graph_sparse_prune_weight,
    graph_sparse_mean_weight = ctx$graph_sparse_mean_weight,
    graph_sparse_min_weight = ctx$graph_sparse_min_weight,
    graph_sparse_max_weight = ctx$graph_sparse_max_weight,
    graph_effective_k = ctx$graph_effective_k,
    graph_edge_retention = ctx$graph_edge_retention,
    graph_recall_at_k = ctx$graph_recall_at_k,
    graph_mean_distance_error = ctx$graph_mean_distance_error,
    graph_rank_correlation = ctx$graph_rank_correlation,
    graph_quality_sample_size = ctx$graph_quality_sample_size,
    graph_mean_degree = ctx$graph_mean_degree,
    graph_min_degree = ctx$graph_min_degree,
    graph_max_degree = ctx$graph_max_degree,
    graph_isolated_fraction = ctx$graph_isolated_fraction,
    graph_padding_fraction = ctx$graph_padding_fraction,
    graph_mean_jaccard = ctx$graph_mean_jaccard,
    graph_min_jaccard = ctx$graph_min_jaccard,
    graph_max_jaccard = ctx$graph_max_jaccard,
    graph_zero_jaccard_fraction = ctx$graph_zero_jaccard_fraction,
    graph_snn_k = ctx$graph_snn_k,
    graph_snn_prune_threshold = ctx$graph_snn_prune_threshold,
    graph_mean_snn_weight = ctx$graph_mean_snn_weight,
    graph_min_snn_weight = ctx$graph_min_snn_weight,
    graph_max_snn_weight = ctx$graph_max_snn_weight,
    graph_zero_snn_fraction = ctx$graph_zero_snn_fraction,
    localmap_false_neighbor_enabled = ctx$localmap_false_neighbor_enabled,
    localmap_false_neighbor_mode = ctx$localmap_false_neighbor_mode,
    localmap_false_neighbor_transfer_mode = ctx$localmap_false_neighbor_transfer_mode,
    localmap_false_neighbor_jaccard_threshold = ctx$localmap_false_neighbor_jaccard_threshold,
    localmap_false_neighbor_distance_quantile = ctx$localmap_false_neighbor_distance_quantile,
    localmap_false_neighbor_distance_multiplier = ctx$localmap_false_neighbor_distance_multiplier,
    localmap_false_neighbor_min_keep_fraction = ctx$localmap_false_neighbor_min_keep_fraction,
    localmap_false_neighbor_min_keep_k = ctx$localmap_false_neighbor_min_keep_k,
    localmap_false_neighbor_removed_edges_mean = ctx$localmap_false_neighbor_removed_edges_mean,
    localmap_false_neighbor_removed_fraction = ctx$localmap_false_neighbor_removed_fraction,
    localmap_false_neighbor_kept_degree_mean = ctx$localmap_false_neighbor_kept_degree_mean,
    localmap_false_neighbor_kept_jaccard_mean = ctx$localmap_false_neighbor_kept_jaccard_mean,
    localmap_false_neighbor_removed_jaccard_mean = ctx$localmap_false_neighbor_removed_jaccard_mean,
    localmap_false_neighbor_kept_distance_ratio_mean = ctx$localmap_false_neighbor_kept_distance_ratio_mean,
    localmap_false_neighbor_removed_distance_ratio_mean = ctx$localmap_false_neighbor_removed_distance_ratio_mean,
    localmap_false_neighbor_threshold_mean = ctx$localmap_false_neighbor_threshold_mean,
    localmap_local_weight_enabled = ctx$localmap_local_weight_enabled,
    localmap_local_weight = ctx$localmap_local_weight,
    localmap_local_weight_mode = ctx$localmap_local_weight_mode,
    localmap_local_weight_transfer_mode = ctx$localmap_local_weight_transfer_mode,
    localmap_local_weight_jaccard_blend = ctx$localmap_local_weight_jaccard_blend,
    localmap_local_weight_mean_trust = ctx$localmap_local_weight_mean_trust,
    localmap_local_weight_rank_component_mean = ctx$localmap_local_weight_rank_component_mean,
    localmap_local_weight_jaccard_component_mean = ctx$localmap_local_weight_jaccard_component_mean,
    localmap_local_weight_mean_multiplier = ctx$localmap_local_weight_mean_multiplier,
    localmap_local_weight_min_multiplier = ctx$localmap_local_weight_min_multiplier,
    localmap_local_weight_max_multiplier = ctx$localmap_local_weight_max_multiplier,
    localmap_local_weight_distance_scale_mean = ctx$localmap_local_weight_distance_scale_mean,
    artificial_neighbor_penalty_enabled = ctx$artificial_neighbor_penalty_enabled,
    artificial_neighbor_transfer_mode = ctx$artificial_neighbor_transfer_mode,
    artificial_neighbor_refinement_backend = ctx$artificial_neighbor_refinement_backend,
    artificial_neighbor_penalty_strength = ctx$artificial_neighbor_penalty_strength,
    artificial_neighbor_penalty_iterations = ctx$artificial_neighbor_penalty_iterations,
    artificial_neighbor_penalty_low_k = ctx$artificial_neighbor_penalty_low_k,
    artificial_neighbor_penalty_far_multiplier = ctx$artificial_neighbor_penalty_far_multiplier,
    artificial_neighbor_penalty_target_distance = ctx$artificial_neighbor_penalty_target_distance,
    artificial_neighbor_penalized_pairs = ctx$artificial_neighbor_penalized_pairs,
    artificial_neighbor_total_low_edges = ctx$artificial_neighbor_total_low_edges,
    artificial_neighbor_false_rate_before = ctx$artificial_neighbor_false_rate_before,
    artificial_neighbor_false_rate_after = ctx$artificial_neighbor_false_rate_after,
    artificial_neighbor_false_rate_delta = ctx$artificial_neighbor_false_rate_delta,
    artificial_neighbor_far_rate_before = ctx$artificial_neighbor_far_rate_before,
    artificial_neighbor_far_rate_after = ctx$artificial_neighbor_far_rate_after,
    artificial_neighbor_far_rate_delta = ctx$artificial_neighbor_far_rate_delta,
    artificial_neighbor_mean_high_distance_ratio = ctx$artificial_neighbor_mean_high_distance_ratio,
    artificial_neighbor_mean_low_distance = ctx$artificial_neighbor_mean_low_distance,
    false_neighbor_monitor_enabled = ctx$false_neighbor_monitor_enabled,
    false_neighbor_monitor_transfer_mode = ctx$false_neighbor_monitor_transfer_mode,
    false_neighbor_monitor_backend = ctx$false_neighbor_monitor_backend,
    false_neighbor_monitor_action = ctx$false_neighbor_monitor_action,
    false_neighbor_monitor_start_mode = ctx$false_neighbor_monitor_start_mode,
    false_neighbor_monitor_chunk_epochs = ctx$false_neighbor_monitor_chunk_epochs,
    false_neighbor_monitor_max_chunks = ctx$false_neighbor_monitor_max_chunks,
    false_neighbor_monitor_chunks_run = ctx$false_neighbor_monitor_chunks_run,
    false_neighbor_monitor_epochs_requested = ctx$false_neighbor_monitor_epochs_requested,
    false_neighbor_monitor_epochs_completed = ctx$false_neighbor_monitor_epochs_completed,
    false_neighbor_monitor_patience = ctx$false_neighbor_monitor_patience,
    false_neighbor_monitor_tolerance = ctx$false_neighbor_monitor_tolerance,
    false_neighbor_monitor_low_k = ctx$false_neighbor_monitor_low_k,
    false_neighbor_monitor_far_multiplier = ctx$false_neighbor_monitor_far_multiplier,
    false_neighbor_monitor_far_weight = ctx$false_neighbor_monitor_far_weight,
    false_neighbor_monitor_initial_false_rate = ctx$false_neighbor_monitor_initial_false_rate,
    false_neighbor_monitor_final_false_rate = ctx$false_neighbor_monitor_final_false_rate,
    false_neighbor_monitor_best_false_rate = ctx$false_neighbor_monitor_best_false_rate,
    false_neighbor_monitor_false_rate_delta = ctx$false_neighbor_monitor_false_rate_delta,
    false_neighbor_monitor_initial_far_rate = ctx$false_neighbor_monitor_initial_far_rate,
    false_neighbor_monitor_final_far_rate = ctx$false_neighbor_monitor_final_far_rate,
    false_neighbor_monitor_best_far_rate = ctx$false_neighbor_monitor_best_far_rate,
    false_neighbor_monitor_far_rate_delta = ctx$false_neighbor_monitor_far_rate_delta,
    false_neighbor_monitor_score_initial = ctx$false_neighbor_monitor_score_initial,
    false_neighbor_monitor_score_final = ctx$false_neighbor_monitor_score_final,
    false_neighbor_monitor_score_best = ctx$false_neighbor_monitor_score_best,
    false_neighbor_monitor_score_delta = ctx$false_neighbor_monitor_score_delta,
    false_neighbor_monitor_worsening_events = ctx$false_neighbor_monitor_worsening_events,
    false_neighbor_monitor_adjustments = ctx$false_neighbor_monitor_adjustments,
    false_neighbor_monitor_stopped_early = ctx$false_neighbor_monitor_stopped_early,
    false_neighbor_monitor_score_trace = ctx$false_neighbor_monitor_score_trace,
    false_neighbor_monitor_false_rate_trace = ctx$false_neighbor_monitor_false_rate_trace,
    false_neighbor_monitor_far_rate_trace = ctx$false_neighbor_monitor_far_rate_trace,
    false_neighbor_monitor_chunk_trace = ctx$false_neighbor_monitor_chunk_trace,
    init_strategy = ctx$init_strategy,
    init_backend = ctx$init_backend,
    init_backend_reason = ctx$init_backend_reason,
    init_time_sec = ctx$init_time_sec,
    init_optimizer_epochs = ctx$init_optimizer_epochs,
    init_optimizer_time_sec = ctx$init_optimizer_time_sec,
    init_scale = ctx$init_scale,
    init_spectral_n_iter = ctx$init_spectral_n_iter,
    init_spectral_solver = ctx$init_spectral_solver,
    init_spectral_graph = ctx$init_spectral_graph,
    init_spectral_eigenvalues = ctx$init_spectral_eigenvalues,
    init_spectral_exact_max_n = ctx$init_spectral_exact_max_n,
    init_spectral_graph_nnz = ctx$init_spectral_graph_nnz,
    init_spectral_graph_active_fraction = ctx$init_spectral_graph_active_fraction,
    init_spectral_nystrom_landmarks = ctx$init_spectral_nystrom_landmarks,
    init_spectral_nystrom_fraction = ctx$init_spectral_nystrom_fraction,
    init_spectral_nystrom_projection_k = ctx$init_spectral_nystrom_projection_k,
    init_spectral_nystrom_weight = ctx$init_spectral_nystrom_weight,
    init_spectral_nystrom_selection_requested = ctx$init_spectral_nystrom_selection_requested,
    init_spectral_nystrom_selection_used = ctx$init_spectral_nystrom_selection_used,
    init_spectral_nystrom_landmark_knn_time_sec = ctx$init_spectral_nystrom_landmark_knn_time_sec,
    init_spectral_nystrom_landmark_spectral_time_sec = ctx$init_spectral_nystrom_landmark_spectral_time_sec,
    init_spectral_nystrom_projection_time_sec = ctx$init_spectral_nystrom_projection_time_sec,
    init_diffusion_time = ctx$init_diffusion_time,
    init_diffusion_n_iter = ctx$init_diffusion_n_iter,
    init_diffusion_solver = ctx$init_diffusion_solver,
    init_diffusion_graph = ctx$init_diffusion_graph,
    init_diffusion_eigenvalues = ctx$init_diffusion_eigenvalues,
    init_diffusion_graph_nnz = ctx$init_diffusion_graph_nnz,
    init_diffusion_graph_active_fraction = ctx$init_diffusion_graph_active_fraction,
    init_laplacian_n_iter = ctx$init_laplacian_n_iter,
    init_laplacian_solver = ctx$init_laplacian_solver,
    init_laplacian_graph = ctx$init_laplacian_graph,
    init_laplacian_eigenvalues = ctx$init_laplacian_eigenvalues,
    init_laplacian_graph_nnz = ctx$init_laplacian_graph_nnz,
    init_laplacian_graph_active_fraction = ctx$init_laplacian_graph_active_fraction,
    init_laplacian_normalized_coordinates = ctx$init_laplacian_normalized_coordinates,
    init_pca_method = ctx$init_pca_method,
    init_pca_oversample = ctx$init_pca_oversample,
    init_pca_power = ctx$init_pca_power,
    init_landmark_n = ctx$init_landmark_n,
    init_landmark_fraction = ctx$init_landmark_fraction,
    init_landmark_selection_requested = ctx$init_landmark_selection_requested,
    init_landmark_selection_used = ctx$init_landmark_selection_used,
    init_projection_k = ctx$init_projection_k,
    init_projection_weight = ctx$init_projection_weight,
    init_landmark_epochs = ctx$init_landmark_epochs,
    init_landmark_knn_time_sec = ctx$init_landmark_knn_time_sec,
    init_landmark_embedding_time_sec = ctx$init_landmark_embedding_time_sec,
    init_projection_time_sec = ctx$init_projection_time_sec,
    init_projection_backend = ctx$init_projection_backend,
    warm_start_enabled = ctx$warm_start_enabled,
    warm_start_cache_hit = ctx$warm_start_cache_hit,
    warm_start_cache_key = ctx$warm_start_cache_key,
    warm_start_previous_init = ctx$warm_start_previous_init,
    warm_start_previous_epochs = ctx$warm_start_previous_epochs,
    warm_start_refinement_epochs = ctx$warm_start_refinement_epochs,
    warm_start_previous_init_time_sec = ctx$warm_start_previous_init_time_sec,
    warm_start_previous_embedding_time_sec = ctx$warm_start_previous_embedding_time_sec,
    warm_start_previous_build_time_sec = ctx$warm_start_previous_build_time_sec,
    warm_start_this_row_setup_time_sec = ctx$warm_start_this_row_setup_time_sec,
    warm_start_refinement_time_sec = ctx$warm_start_refinement_time_sec,
    warm_start_total_if_cache_miss_sec = ctx$warm_start_total_if_cache_miss_sec,
    warm_start_reuse_mode = ctx$warm_start_reuse_mode,
    warm_start_use_case = ctx$warm_start_use_case,
    warm_start_parameter_delta = ctx$warm_start_parameter_delta,
    warm_start_bias_risk = ctx$warm_start_bias_risk,
    epoch_budget_enabled = ctx$epoch_budget_enabled,
    epoch_budget_requested = ctx$epoch_budget_requested,
    epoch_budget_effective = ctx$epoch_budget_effective,
    epoch_budget_default_epochs = ctx$epoch_budget_default_epochs,
    epoch_budget_ratio_to_default = ctx$epoch_budget_ratio_to_default,
    epoch_budget_quality = ctx$epoch_budget_quality,
    epoch_budget_tsne_mode = ctx$epoch_budget_tsne_mode,
    epoch_budget_optimizer_backend = ctx$epoch_budget_optimizer_backend,
    epoch_budget_speed_quality_tradeoff = ctx$epoch_budget_speed_quality_tradeoff,
    epoch_budget_is_default = ctx$epoch_budget_is_default,
    early_stop_enabled = ctx$early_stop_enabled,
    early_stop_criterion = ctx$early_stop_criterion,
    early_stop_status = ctx$early_stop_status,
    early_stop_reason = ctx$early_stop_reason,
    early_stop_max_epochs = ctx$early_stop_max_epochs,
    early_stop_chunk_epochs = ctx$early_stop_chunk_epochs,
    early_stop_epochs_run = ctx$early_stop_epochs_run,
    early_stop_chunks_run = ctx$early_stop_chunks_run,
    early_stop_patience = ctx$early_stop_patience,
    early_stop_tolerance = ctx$early_stop_tolerance,
    early_stop_displacement_final = ctx$early_stop_displacement_final,
    early_stop_trustworthiness_final = ctx$early_stop_trustworthiness_final,
    early_stop_trustworthiness_delta_final = ctx$early_stop_trustworthiness_delta_final,
    early_stop_neighbour_stability_final = ctx$early_stop_neighbour_stability_final,
    early_stop_neighbour_stability_delta_final = ctx$early_stop_neighbour_stability_delta_final,
    early_stop_monitor_sample_size = ctx$early_stop_monitor_sample_size,
    early_stop_monitor_k = ctx$early_stop_monitor_k,
    early_stop_init_strategy = ctx$early_stop_init_strategy,
    early_stop_init_time_sec = ctx$early_stop_init_time_sec,
    early_stop_optimizer_time_sec = ctx$early_stop_optimizer_time_sec,
    early_stop_loss_available = ctx$early_stop_loss_available,
    early_stop_loss_reason = ctx$early_stop_loss_reason,
    early_stop_chunked_optimizer = ctx$early_stop_chunked_optimizer,
    early_stop_trace_epochs = ctx$early_stop_trace_epochs,
    early_stop_trace_displacement = ctx$early_stop_trace_displacement,
    early_stop_trace_trustworthiness = ctx$early_stop_trace_trustworthiness,
    early_stop_trace_trustworthiness_delta = ctx$early_stop_trace_trustworthiness_delta,
    early_stop_trace_neighbour_stability = ctx$early_stop_trace_neighbour_stability,
    early_stop_trace_neighbour_stability_delta = ctx$early_stop_trace_neighbour_stability_delta,
    early_stop_backend_scope = ctx$early_stop_backend_scope,
    early_stop_risk = ctx$early_stop_risk,
    coarse_to_fine_enabled = ctx$coarse_to_fine_enabled,
    coarse_to_fine_mode = ctx$coarse_to_fine_mode,
    coarse_to_fine_selection_requested = ctx$coarse_to_fine_selection_requested,
    coarse_to_fine_selection_used = ctx$coarse_to_fine_selection_used,
    coarse_to_fine_count_requested = ctx$coarse_to_fine_count_requested,
    coarse_to_fine_fraction_requested = ctx$coarse_to_fine_fraction_requested,
    coarse_to_fine_n = ctx$coarse_to_fine_n,
    coarse_to_fine_fraction = ctx$coarse_to_fine_fraction,
    coarse_to_fine_k = ctx$coarse_to_fine_k,
    coarse_to_fine_projection_k = ctx$coarse_to_fine_projection_k,
    coarse_to_fine_projection_weight = ctx$coarse_to_fine_projection_weight,
    coarse_to_fine_coarse_epochs = ctx$coarse_to_fine_coarse_epochs,
    coarse_to_fine_refinement_epochs = ctx$coarse_to_fine_refinement_epochs,
    coarse_to_fine_selection_time_sec = ctx$coarse_to_fine_selection_time_sec,
    coarse_to_fine_knn_time_sec = ctx$coarse_to_fine_knn_time_sec,
    coarse_to_fine_embedding_time_sec = ctx$coarse_to_fine_embedding_time_sec,
    coarse_to_fine_projection_time_sec = ctx$coarse_to_fine_projection_time_sec,
    coarse_to_fine_refinement_time_sec = ctx$coarse_to_fine_refinement_time_sec,
    coarse_to_fine_setup_time_sec = ctx$coarse_to_fine_setup_time_sec,
    coarse_to_fine_projection_entropy = ctx$coarse_to_fine_projection_entropy,
    coarse_to_fine_projection_zero_neighbor_fraction = ctx$coarse_to_fine_projection_zero_neighbor_fraction,
    coarse_to_fine_projection_bandwidth_mean = ctx$coarse_to_fine_projection_bandwidth_mean,
    coarse_to_fine_projection_backend = ctx$coarse_to_fine_projection_backend,
    coarse_to_fine_refinement_backend = ctx$coarse_to_fine_refinement_backend,
    coarse_to_fine_init_backend = ctx$coarse_to_fine_init_backend,
    coarse_to_fine_expected_gain = ctx$coarse_to_fine_expected_gain,
    coarse_to_fine_risk = ctx$coarse_to_fine_risk,
    landmark_enabled = ctx$landmark_enabled,
    landmark_approximation = ctx$landmark_approximation,
    landmark_mode = ctx$landmark_mode,
    landmark_selection = ctx$landmark_selection,
    landmark_selection_requested = ctx$landmark_selection_requested,
    landmark_selection_used = ctx$landmark_selection_used,
    landmark_count_requested = ctx$landmark_count_requested,
    landmark_fraction_requested = ctx$landmark_fraction_requested,
    landmark_n = ctx$landmark_n,
    landmark_fraction = ctx$landmark_fraction,
    landmark_label_classes_total = ctx$landmark_label_classes_total,
    landmark_label_classes_present = ctx$landmark_label_classes_present,
    landmark_label_missing_classes = ctx$landmark_label_missing_classes,
    landmark_label_min_count = ctx$landmark_label_min_count,
    landmark_label_min_fraction = ctx$landmark_label_min_fraction,
    landmark_rare_label_count = ctx$landmark_rare_label_count,
    landmark_rare_label_present = ctx$landmark_rare_label_present,
    stratified_landmark_source = ctx$stratified_landmark_source,
    stratified_landmark_allocation = ctx$stratified_landmark_allocation,
    stratified_landmark_time_sec = ctx$stratified_landmark_time_sec,
    stratified_landmark_n_strata = ctx$stratified_landmark_n_strata,
    stratified_landmark_strata_sampled = ctx$stratified_landmark_strata_sampled,
    stratified_landmark_missing_strata = ctx$stratified_landmark_missing_strata,
    stratified_landmark_min_stratum_size = ctx$stratified_landmark_min_stratum_size,
    stratified_landmark_max_stratum_size = ctx$stratified_landmark_max_stratum_size,
    stratified_landmark_min_selected_per_stratum = ctx$stratified_landmark_min_selected_per_stratum,
    stratified_landmark_max_selected_per_stratum = ctx$stratified_landmark_max_selected_per_stratum,
    stratified_landmark_balance_ratio = ctx$stratified_landmark_balance_ratio,
    stratified_landmark_cluster_k = ctx$stratified_landmark_cluster_k,
    stratified_landmark_cluster_feature_dims = ctx$stratified_landmark_cluster_feature_dims,
    density_landmark_alpha = ctx$density_landmark_alpha,
    density_landmark_k = ctx$density_landmark_k,
    density_landmark_time_sec = ctx$density_landmark_time_sec,
    density_landmark_weight_min = ctx$density_landmark_weight_min,
    density_landmark_weight_median = ctx$density_landmark_weight_median,
    density_landmark_weight_max = ctx$density_landmark_weight_max,
    density_landmark_weight_mean = ctx$density_landmark_weight_mean,
    density_landmark_selected_weight_mean = ctx$density_landmark_selected_weight_mean,
    density_landmark_selected_to_global_weight_ratio = ctx$density_landmark_selected_to_global_weight_ratio,
    density_landmark_mean_distance_median = ctx$density_landmark_mean_distance_median,
    hybrid_landmark_alpha = ctx$hybrid_landmark_alpha,
    hybrid_landmark_beta = ctx$hybrid_landmark_beta,
    hybrid_landmark_k = ctx$hybrid_landmark_k,
    hybrid_landmark_time_sec = ctx$hybrid_landmark_time_sec,
    hybrid_landmark_density_time_sec = ctx$hybrid_landmark_density_time_sec,
    hybrid_landmark_feature_dims = ctx$hybrid_landmark_feature_dims,
    hybrid_landmark_formula = ctx$hybrid_landmark_formula,
    hybrid_landmark_density_weight_median = ctx$hybrid_landmark_density_weight_median,
    hybrid_landmark_density_selected_to_global_weight_ratio = ctx$hybrid_landmark_density_selected_to_global_weight_ratio,
    hybrid_landmark_mean_distance_median = ctx$hybrid_landmark_mean_distance_median,
    hybrid_landmark_cover_mean = ctx$hybrid_landmark_cover_mean,
    hybrid_landmark_cover_median = ctx$hybrid_landmark_cover_median,
    hybrid_landmark_cover_max = ctx$hybrid_landmark_cover_max,
    rare_protected_tail_fraction = ctx$rare_protected_tail_fraction,
    rare_protected_tail_oversample = ctx$rare_protected_tail_oversample,
    rare_protected_n_quantiles = ctx$rare_protected_n_quantiles,
    rare_protected_cluster_fraction = ctx$rare_protected_cluster_fraction,
    rare_protected_density_k = ctx$rare_protected_density_k,
    rare_protected_time_sec = ctx$rare_protected_time_sec,
    rare_protected_density_time_sec = ctx$rare_protected_density_time_sec,
    rare_protected_cluster_time_sec = ctx$rare_protected_cluster_time_sec,
    rare_protected_tail_count = ctx$rare_protected_tail_count,
    rare_protected_quantile_count = ctx$rare_protected_quantile_count,
    rare_protected_cluster_count = ctx$rare_protected_cluster_count,
    rare_protected_fill_count = ctx$rare_protected_fill_count,
    rare_protected_tail_threshold = ctx$rare_protected_tail_threshold,
    rare_protected_selected_tail_fraction = ctx$rare_protected_selected_tail_fraction,
    rare_protected_selected_mean_distance_ratio = ctx$rare_protected_selected_mean_distance_ratio,
    rare_protected_selected_to_global_low_density_ratio = ctx$rare_protected_selected_to_global_low_density_ratio,
    rare_protected_quantile_min_selected = ctx$rare_protected_quantile_min_selected,
    rare_protected_quantile_max_selected = ctx$rare_protected_quantile_max_selected,
    rare_protected_cluster_k = ctx$rare_protected_cluster_k,
    rare_protected_cluster_feature_dims = ctx$rare_protected_cluster_feature_dims,
    diversity_landmark_algorithm = ctx$diversity_landmark_algorithm,
    diversity_landmark_time_sec = ctx$diversity_landmark_time_sec,
    diversity_landmark_feature_dims = ctx$diversity_landmark_feature_dims,
    diversity_landmark_cover_mean = ctx$diversity_landmark_cover_mean,
    diversity_landmark_cover_median = ctx$diversity_landmark_cover_median,
    diversity_landmark_cover_max = ctx$diversity_landmark_cover_max,
    diversity_landmark_leverage_selected_to_global_ratio = ctx$diversity_landmark_leverage_selected_to_global_ratio,
    landmark_projection_k = ctx$landmark_projection_k,
    landmark_interpolation = ctx$landmark_interpolation,
    landmark_projection_model = ctx$landmark_projection_model,
    landmark_projection_weight = ctx$landmark_projection_weight,
    landmark_projection_bandwidth_rule = ctx$landmark_projection_bandwidth_rule,
    landmark_projection_bandwidth_mean = ctx$landmark_projection_bandwidth_mean,
    landmark_projection_weight_entropy = ctx$landmark_projection_weight_entropy,
    landmark_projection_zero_neighbor_fraction = ctx$landmark_projection_zero_neighbor_fraction,
    landmark_projection_time_sec = ctx$landmark_projection_time_sec,
    landmark_landmark_knn_time_sec = ctx$landmark_landmark_knn_time_sec,
    landmark_landmark_embedding_time_sec = ctx$landmark_landmark_embedding_time_sec,
    landmark_affine_ridge = ctx$landmark_affine_ridge,
    landmark_affine_weight = ctx$landmark_affine_weight,
    landmark_affine_rank_mean = ctx$landmark_affine_rank_mean,
    landmark_affine_condition_median = ctx$landmark_affine_condition_median,
    landmark_affine_condition_max = ctx$landmark_affine_condition_max,
    landmark_affine_fallback_fraction = ctx$landmark_affine_fallback_fraction,
    landmark_affine_clipped_fraction = ctx$landmark_affine_clipped_fraction,
    landmark_affine_clip_multiplier = ctx$landmark_affine_clip_multiplier,
    landmark_affine_blend = ctx$landmark_affine_blend,
    landmark_refinement = ctx$landmark_refinement,
    landmark_refinement_epochs = ctx$landmark_refinement_epochs,
    landmark_refinement_time_sec = ctx$landmark_refinement_time_sec,
    landmark_projection_backend = ctx$landmark_projection_backend,
    landmark_interpolation_backend = ctx$landmark_interpolation_backend,
    landmark_interpolation_backend_reason = ctx$landmark_interpolation_backend_reason,
    landmark_refinement_backend = ctx$landmark_refinement_backend,
    landmark_refinement_knn_backend = ctx$landmark_refinement_knn_backend,
    landmark_refinement_knn_backend_reason = ctx$landmark_refinement_knn_backend_reason,
    subsample_strategy = ctx$subsample_strategy,
    subsample_stratified = ctx$subsample_stratified,
    benchmark_forced_k = ctx$benchmark_forced_k,
    benchmark_standardize = ctx$benchmark_standardize,
    graph_adaptive_min_k = ctx$graph_adaptive_min_k,
    graph_adaptive_max_k = ctx$graph_adaptive_max_k,
    graph_adaptive_mean_k = ctx$graph_adaptive_mean_k,
    graph_adaptive_density_quantile = ctx$graph_adaptive_density_quantile,
    graph_adaptive_density_cor = ctx$graph_adaptive_density_cor,
    graph_adaptive_dense_fraction = ctx$graph_adaptive_dense_fraction,
    graph_adaptive_sparse_fraction = ctx$graph_adaptive_sparse_fraction,
    graph_distance_prune_drop_fraction = ctx$graph_distance_prune_drop_fraction,
    graph_distance_prune_percentile = ctx$graph_distance_prune_percentile,
    graph_distance_prune_removed_edges_mean = ctx$graph_distance_prune_removed_edges_mean,
    graph_distance_prune_threshold_mean = ctx$graph_distance_prune_threshold_mean,
    graph_distance_prune_threshold_min = ctx$graph_distance_prune_threshold_min,
    graph_distance_prune_threshold_max = ctx$graph_distance_prune_threshold_max,
    graph_distance_prune_removed_distance_mean = ctx$graph_distance_prune_removed_distance_mean,
    graph_sparsification_method = ctx$graph_sparsification_method,
    graph_sparsification_keep_fraction = ctx$graph_sparsification_keep_fraction,
    graph_sparsification_target_k = ctx$graph_sparsification_target_k,
    graph_sparsification_undirected_edges = ctx$graph_sparsification_undirected_edges,
    graph_sparsification_spectral_rank = ctx$graph_sparsification_spectral_rank,
    graph_sparsification_spectral_time_sec = ctx$graph_sparsification_spectral_time_sec,
    graph_sparsification_leverage_mean = ctx$graph_sparsification_leverage_mean,
    graph_sparsification_leverage_min = ctx$graph_sparsification_leverage_min,
    graph_sparsification_leverage_max = ctx$graph_sparsification_leverage_max,
    graph_sparsification_resistance_mean = ctx$graph_sparsification_resistance_mean,
    graph_sparsification_weight_mean = ctx$graph_sparsification_weight_mean,
    graph_mst_rescue_enabled = ctx$graph_mst_rescue_enabled,
    graph_mst_rescue_base_components = ctx$graph_mst_rescue_base_components,
    graph_mst_rescue_components_before = ctx$graph_mst_rescue_components_before,
    graph_mst_rescue_components_after = ctx$graph_mst_rescue_components_after,
    graph_mst_rescue_forest_edges = ctx$graph_mst_rescue_forest_edges,
    graph_mst_rescue_added_forest_edges = ctx$graph_mst_rescue_added_forest_edges,
    graph_mst_rescue_added_directed_edges = ctx$graph_mst_rescue_added_directed_edges,
    graph_mst_rescue_mean_degree_before = ctx$graph_mst_rescue_mean_degree_before,
    graph_mst_rescue_mean_degree_after = ctx$graph_mst_rescue_mean_degree_after,
    graph_density_correction_method = ctx$graph_density_correction_method,
    graph_density_correction_quantile = ctx$graph_density_correction_quantile,
    graph_density_correction_strength = ctx$graph_density_correction_strength,
    graph_density_scale_mean = ctx$graph_density_scale_mean,
    graph_density_scale_min = ctx$graph_density_scale_min,
    graph_density_scale_max = ctx$graph_density_scale_max,
    graph_density_scale_cv = ctx$graph_density_scale_cv,
    graph_density_sparse_fraction = ctx$graph_density_sparse_fraction,
    graph_density_correction_mean = ctx$graph_density_correction_mean,
    graph_density_correction_min = ctx$graph_density_correction_min,
    graph_density_correction_max = ctx$graph_density_correction_max,
    graph_density_correction_clamp_fraction = ctx$graph_density_correction_clamp_fraction,
    graph_density_corrected_distance_scale_cor = ctx$graph_density_corrected_distance_scale_cor,
    umap_graph_set_op_mix_ratio = ctx$umap_graph_set_op_mix_ratio,
    umap_graph_local_connectivity = ctx$umap_graph_local_connectivity,
    umap_graph_weight_power = ctx$umap_graph_weight_power,
    umap_graph_target_scale = ctx$umap_graph_target_scale,
    umap_graph_distance_transform = ctx$umap_graph_distance_transform,
    umap_graph_mean_weight = ctx$umap_graph_mean_weight,
    umap_graph_min_weight = ctx$umap_graph_min_weight,
    umap_graph_max_weight = ctx$umap_graph_max_weight,
    graph_edge_sampling_method = ctx$graph_edge_sampling_method,
    graph_edge_sampling_fraction = ctx$graph_edge_sampling_fraction,
    graph_edge_sampling_weight_power = ctx$graph_edge_sampling_weight_power,
    graph_edge_sampling_include_top = ctx$graph_edge_sampling_include_top,
    graph_edge_sampling_target_scale = ctx$graph_edge_sampling_target_scale,
    graph_edge_sampling_mean_selected_weight = ctx$graph_edge_sampling_mean_selected_weight,
    graph_edge_sampling_mean_candidate_weight = ctx$graph_edge_sampling_mean_candidate_weight,
    graph_edge_sampling_selected_to_candidate_weight_ratio = ctx$graph_edge_sampling_selected_to_candidate_weight_ratio,
    pair_resampling_mode = ctx$pair_resampling_mode,
    pair_resampling_pair_family = ctx$pair_resampling_pair_family,
    pair_resampling_transfer_mode = ctx$pair_resampling_transfer_mode,
    pair_resampling_refreshes = ctx$pair_resampling_refreshes,
    pair_resampling_stage_count = ctx$pair_resampling_stage_count,
    pair_resampling_stage_epochs = ctx$pair_resampling_stage_epochs,
    pair_resampling_warmup_epochs = ctx$pair_resampling_warmup_epochs,
    pair_resampling_refine_epochs = ctx$pair_resampling_refine_epochs,
    pair_resampling_keep_fraction = ctx$pair_resampling_keep_fraction,
    pair_resampling_weight_power = ctx$pair_resampling_weight_power,
    pair_resampling_include_top = ctx$pair_resampling_include_top,
    pair_resampling_seed_stride = ctx$pair_resampling_seed_stride,
    pair_resampling_final_graph = ctx$pair_resampling_final_graph,
    triplet_aux_enabled = ctx$triplet_aux_enabled,
    triplet_aux_weight = ctx$triplet_aux_weight,
    triplet_aux_samples_per_edge = ctx$triplet_aux_samples_per_edge,
    triplet_aux_transfer_mode = ctx$triplet_aux_transfer_mode,
    triplet_aux_native_backend = ctx$triplet_aux_native_backend,
	    triplet_aux_optimizer_backend = ctx$triplet_aux_optimizer_backend,
	    triplet_aux_n_epochs = ctx$triplet_aux_n_epochs,
	    triplet_aux_negative_sample_rate = ctx$triplet_aux_negative_sample_rate,
	    triplet_structured_enabled = ctx$triplet_structured_enabled,
	    triplet_structured_weight = ctx$triplet_structured_weight,
	    triplet_structured_n_inliers = ctx$triplet_structured_n_inliers,
	    triplet_structured_n_outliers = ctx$triplet_structured_n_outliers,
	    triplet_structured_n_random = ctx$triplet_structured_n_random,
	    triplet_structured_total = ctx$triplet_structured_total,
	    triplet_structured_mode = ctx$triplet_structured_mode,
	    global_random_triplet_enabled = ctx$global_random_triplet_enabled,
	    global_random_triplet_weight = ctx$global_random_triplet_weight,
	    global_random_triplets_per_point = ctx$global_random_triplets_per_point,
	    global_random_trimap_extra_negatives = ctx$global_random_trimap_extra_negatives,
	    global_random_negative_source = ctx$global_random_negative_source,
	    global_random_transfer_mode = ctx$global_random_transfer_mode,
	    global_random_effective_negative_sample_rate = ctx$global_random_effective_negative_sample_rate,
		    hard_negative_enabled = ctx$hard_negative_enabled,
		    hard_negative_rate = ctx$hard_negative_rate,
		    hard_negative_weight_multiplier = ctx$hard_negative_weight_multiplier,
		    hard_negative_candidate_source = ctx$hard_negative_candidate_source,
		    hard_negative_transfer_mode = ctx$hard_negative_transfer_mode,
		    semihard_triplet_enabled = ctx$semihard_triplet_enabled,
		    semihard_triplet_rate = ctx$semihard_triplet_rate,
		    semihard_triplet_weight_multiplier = ctx$semihard_triplet_weight_multiplier,
		    semihard_triplet_candidate_source = ctx$semihard_triplet_candidate_source,
		    semihard_triplet_transfer_mode = ctx$semihard_triplet_transfer_mode,
		    triplet_mining_approximate = ctx$triplet_mining_approximate,
		    triplet_mining_graph_source = ctx$triplet_mining_graph_source,
		    triplet_mining_source_detail = ctx$triplet_mining_source_detail,
		    triplet_mining_knn_backend = ctx$triplet_mining_knn_backend,
		    triplet_mining_candidate_source = ctx$triplet_mining_candidate_source,
		    triplet_mining_transfer_mode = ctx$triplet_mining_transfer_mode,
		    triplet_mining_recall_at_k = ctx$triplet_mining_recall_at_k,
		    triplet_mining_rank_correlation = ctx$triplet_mining_rank_correlation,
		    triplet_mining_distance_error = ctx$triplet_mining_distance_error,
		    umap_negative_sample_rate = ctx$umap_negative_sample_rate,
    umap_transfer_mode = ctx$umap_transfer_mode,
    graph_tsne_affinity_mode = ctx$graph_tsne_affinity_mode,
    graph_tsne_affinity_perplexities = ctx$graph_tsne_affinity_perplexities,
    graph_tsne_affinity_num_scales = ctx$graph_tsne_affinity_num_scales,
    graph_tsne_affinity_temperature = ctx$graph_tsne_affinity_temperature,
    graph_tsne_affinity_entropy_mean = ctx$graph_tsne_affinity_entropy_mean,
    graph_tsne_affinity_effective_perplexity_mean = ctx$graph_tsne_affinity_effective_perplexity_mean,
    graph_tsne_affinity_sigma_mean = ctx$graph_tsne_affinity_sigma_mean,
    graph_tsne_affinity_sigma_min = ctx$graph_tsne_affinity_sigma_min,
    graph_tsne_affinity_sigma_max = ctx$graph_tsne_affinity_sigma_max,
    graph_tsne_affinity_prob_min = ctx$graph_tsne_affinity_prob_min,
    graph_tsne_affinity_prob_max = ctx$graph_tsne_affinity_prob_max,
    graph_multiscale_perplexities = ctx$graph_multiscale_perplexities,
    graph_multiscale_num_scales = ctx$graph_multiscale_num_scales,
    graph_multiscale_required_k = ctx$graph_multiscale_required_k,
    graph_multiscale_effective_k_values = ctx$graph_multiscale_effective_k_values,
    graph_multiscale_transfer_mode = ctx$graph_multiscale_transfer_mode,
    graph_multiscale_uses_tsne_affinity = ctx$graph_multiscale_uses_tsne_affinity,
    pacmap_transfer_mode = ctx$pacmap_transfer_mode,
    pacmap_auxiliary_pair_family = ctx$pacmap_auxiliary_pair_family,
    pacmap_mid_near_pairs_per_point = ctx$pacmap_mid_near_pairs_per_point,
    pacmap_mid_near_fraction = ctx$pacmap_mid_near_fraction,
    pacmap_mid_near_requested_fraction = ctx$pacmap_mid_near_requested_fraction,
    pacmap_mid_near_distance_scale = ctx$pacmap_mid_near_distance_scale,
    pacmap_mid_near_fallback_fraction = ctx$pacmap_mid_near_fallback_fraction,
    pacmap_mid_near_rank_mean = ctx$pacmap_mid_near_rank_mean,
    pacmap_mid_near_emphasis_strength = ctx$pacmap_mid_near_emphasis_strength,
    pacmap_mid_near_emphasis_distance_multiplier = ctx$pacmap_mid_near_emphasis_distance_multiplier,
    pacmap_near_ratio = ctx$pacmap_near_ratio,
    pacmap_mid_ratio = ctx$pacmap_mid_ratio,
    pacmap_far_ratio = ctx$pacmap_far_ratio,
    pacmap_near_pairs_per_point = ctx$pacmap_near_pairs_per_point,
    pacmap_mid_pairs_per_point = ctx$pacmap_mid_pairs_per_point,
    pacmap_far_pairs_per_point = ctx$pacmap_far_pairs_per_point,
    pacmap_far_pair_fraction = ctx$pacmap_far_pair_fraction,
    pacmap_far_distance_scale = ctx$pacmap_far_distance_scale,
    pacmap_far_fallback_fraction = ctx$pacmap_far_fallback_fraction,
    pacmap_far_repulsion_rate = ctx$pacmap_far_repulsion_rate,
    pacmap_phase_schedule = ctx$pacmap_phase_schedule,
    pacmap_phase_total_epochs = ctx$pacmap_phase_total_epochs,
    pacmap_phase_epoch_multiplier = ctx$pacmap_phase_epoch_multiplier,
    pacmap_phase_warmup_epochs = ctx$pacmap_phase_warmup_epochs,
    pacmap_phase_refine_epochs = ctx$pacmap_phase_refine_epochs,
    pacmap_phase_transfer_detail = ctx$pacmap_phase_transfer_detail,
    trimap_transfer_mode = ctx$trimap_transfer_mode,
    trimap_triplet_family = ctx$trimap_triplet_family,
    trimap_inlier_ratio = ctx$trimap_inlier_ratio,
    trimap_semihard_ratio = ctx$trimap_semihard_ratio,
    trimap_global_anchor_ratio = ctx$trimap_global_anchor_ratio,
    trimap_inlier_pairs_per_point = ctx$trimap_inlier_pairs_per_point,
    trimap_semihard_pairs_per_point = ctx$trimap_semihard_pairs_per_point,
    trimap_global_anchor_pairs_per_point = ctx$trimap_global_anchor_pairs_per_point,
    trimap_semihard_fraction = ctx$trimap_semihard_fraction,
    trimap_global_anchor_fraction = ctx$trimap_global_anchor_fraction,
    trimap_semihard_distance_scale = ctx$trimap_semihard_distance_scale,
    trimap_global_anchor_distance_scale = ctx$trimap_global_anchor_distance_scale,
    trimap_semihard_fallback_fraction = ctx$trimap_semihard_fallback_fraction,
    trimap_global_anchor_fallback_fraction = ctx$trimap_global_anchor_fallback_fraction,
    trimap_semihard_rank_mean = ctx$trimap_semihard_rank_mean,
    trimap_candidate_seed = ctx$trimap_candidate_seed,
    trimap_native_explicit_triplets = ctx$trimap_native_explicit_triplets,
    trimap_triplet_proxy_detail = ctx$trimap_triplet_proxy_detail,
    early_exaggeration_factor = ctx$early_exaggeration_factor,
    early_exaggeration_duration_fraction = ctx$early_exaggeration_duration_fraction,
    early_exaggeration_total_epochs = ctx$early_exaggeration_total_epochs,
    early_exaggeration_warmup_epochs = ctx$early_exaggeration_warmup_epochs,
    early_exaggeration_refine_epochs = ctx$early_exaggeration_refine_epochs,
    early_exaggeration_transfer_mode = ctx$early_exaggeration_transfer_mode,
    early_exaggeration_distance_scale = ctx$early_exaggeration_distance_scale,
    early_exaggeration_schedule_mode = ctx$early_exaggeration_schedule_mode,
    late_exaggeration_factor = ctx$late_exaggeration_factor,
    late_exaggeration_duration_fraction = ctx$late_exaggeration_duration_fraction,
    late_exaggeration_total_epochs = ctx$late_exaggeration_total_epochs,
    late_exaggeration_requested_epochs = ctx$late_exaggeration_requested_epochs,
    late_exaggeration_normal_epochs = ctx$late_exaggeration_normal_epochs,
    late_exaggeration_late_epochs = ctx$late_exaggeration_late_epochs,
    late_exaggeration_start_iter = ctx$late_exaggeration_start_iter,
    late_exaggeration_transfer_mode = ctx$late_exaggeration_transfer_mode,
    late_exaggeration_distance_scale = ctx$late_exaggeration_distance_scale,
    late_exaggeration_schedule_mode = ctx$late_exaggeration_schedule_mode,
    optimizer_mode = ctx$optimizer_mode,
    optimizer_schedule = ctx$optimizer_schedule,
    optimizer_momentum = ctx$optimizer_momentum,
    optimizer_final_momentum = ctx$optimizer_final_momentum,
    optimizer_switch_iter = ctx$optimizer_switch_iter,
    optimizer_learning_rate = ctx$optimizer_learning_rate,
    optimizer_learning_rate_multiplier = ctx$optimizer_learning_rate_multiplier,
    optimizer_adam_beta1 = ctx$optimizer_adam_beta1,
    optimizer_adam_beta2 = ctx$optimizer_adam_beta2,
    optimizer_adam_epsilon = ctx$optimizer_adam_epsilon,
    optimizer_transfer_mode = ctx$optimizer_transfer_mode,
    learning_rate_rule = ctx$learning_rate_rule,
    learning_rate_value = ctx$learning_rate_value,
    learning_rate_base_default = ctx$learning_rate_base_default,
    learning_rate_scale = ctx$learning_rate_scale,
    learning_rate_transfer_mode = ctx$learning_rate_transfer_mode,
    adaptive_lr_enabled = ctx$adaptive_lr_enabled,
    adaptive_lr_schedule = ctx$adaptive_lr_schedule,
    adaptive_lr_total_epochs = ctx$adaptive_lr_total_epochs,
    adaptive_lr_chunk_epochs = ctx$adaptive_lr_chunk_epochs,
    adaptive_lr_chunks_run = ctx$adaptive_lr_chunks_run,
    adaptive_lr_base_learning_rate = ctx$adaptive_lr_base_learning_rate,
    adaptive_lr_final_learning_rate = ctx$adaptive_lr_final_learning_rate,
    adaptive_lr_final_multiplier = ctx$adaptive_lr_final_multiplier,
    adaptive_lr_optimizer = ctx$adaptive_lr_optimizer,
    adaptive_lr_native = ctx$adaptive_lr_native,
    adaptive_lr_chunked = ctx$adaptive_lr_chunked,
    adaptive_lr_backend = ctx$adaptive_lr_backend,
    adaptive_lr_inner_decay = ctx$adaptive_lr_inner_decay,
    adaptive_lr_init_strategy = ctx$adaptive_lr_init_strategy,
    adaptive_lr_init_time_sec = ctx$adaptive_lr_init_time_sec,
    adaptive_lr_optimizer_time_sec = ctx$adaptive_lr_optimizer_time_sec,
    adaptive_lr_trace_epochs = ctx$adaptive_lr_trace_epochs,
    adaptive_lr_trace_learning_rate = ctx$adaptive_lr_trace_learning_rate,
    adaptive_lr_trace_multiplier = ctx$adaptive_lr_trace_multiplier,
    adaptive_lr_note = ctx$adaptive_lr_note,
    mini_batch_enabled = ctx$mini_batch_enabled,
    mini_batch_backend = ctx$mini_batch_backend,
    mini_batch_mode = ctx$mini_batch_mode,
    mini_batch_batch_fraction = ctx$mini_batch_batch_fraction,
    mini_batch_effective_k = ctx$mini_batch_effective_k,
    mini_batch_chunks = ctx$mini_batch_chunks,
    mini_batch_chunk_epochs = ctx$mini_batch_chunk_epochs,
    mini_batch_total_epochs = ctx$mini_batch_total_epochs,
    mini_batch_refreshes = ctx$mini_batch_refreshes,
    mini_batch_sampling = ctx$mini_batch_sampling,
    mini_batch_weight_power = ctx$mini_batch_weight_power,
    mini_batch_include_top = ctx$mini_batch_include_top,
    mini_batch_init_strategy = ctx$mini_batch_init_strategy,
    mini_batch_init_time_sec = ctx$mini_batch_init_time_sec,
    mini_batch_graph_time_sec = ctx$mini_batch_graph_time_sec,
    mini_batch_optimizer_time_sec = ctx$mini_batch_optimizer_time_sec,
    mini_batch_trace_effective_k = ctx$mini_batch_trace_effective_k,
    mini_batch_trace_retention = ctx$mini_batch_trace_retention,
    mini_batch_experimental = ctx$mini_batch_experimental,
    mini_batch_note = ctx$mini_batch_note,
    deterministic_batch_row_batch_size = ctx$deterministic_batch_row_batch_size,
    deterministic_batch_chunks_per_epoch = ctx$deterministic_batch_chunks_per_epoch,
    deterministic_batch_reduction = ctx$deterministic_batch_reduction,
    deterministic_batch_atomic_updates = ctx$deterministic_batch_atomic_updates,
    deterministic_batch_reproducible_given_threads = ctx$deterministic_batch_reproducible_given_threads,
    sparse_edge_batch_enabled = ctx_default("sparse_edge_batch_enabled", NA),
    sparse_edge_batch_backend = ctx_default("sparse_edge_batch_backend", NA_character_),
    sparse_edge_batch_mode = ctx_default("sparse_edge_batch_mode", NA_character_),
    sparse_edge_batch_storage = ctx_default("sparse_edge_batch_storage", NA_character_),
    sparse_edge_batch_edge_batch_size = ctx_default("sparse_edge_batch_edge_batch_size", NA_real_),
    sparse_edge_batch_chunks_per_epoch = ctx_default("sparse_edge_batch_chunks_per_epoch", NA_real_),
    sparse_edge_batch_n_epochs = ctx_default("sparse_edge_batch_n_epochs", NA_real_),
    sparse_edge_batch_negative_sample_rate = ctx_default("sparse_edge_batch_negative_sample_rate", NA_real_),
    sparse_edge_batch_learning_rate = ctx_default("sparse_edge_batch_learning_rate", NA_real_),
    sparse_edge_batch_threads = ctx_default("sparse_edge_batch_threads", NA_real_),
    sparse_edge_batch_atomic_updates = ctx_default("sparse_edge_batch_atomic_updates", NA),
    sparse_edge_batch_edge_list_copy = ctx_default("sparse_edge_batch_edge_list_copy", NA),
    sparse_edge_batch_triplet_chunks = ctx_default("sparse_edge_batch_triplet_chunks", NA),
    sparse_edge_batch_affinity_chunks = ctx_default("sparse_edge_batch_affinity_chunks", NA),
    sparse_edge_batch_aux_memory_mb = ctx_default("sparse_edge_batch_aux_memory_mb", NA_real_),
    sparse_edge_batch_graph_time_sec = ctx_default("sparse_edge_batch_graph_time_sec", NA_real_),
    sparse_edge_batch_init_time_sec = ctx_default("sparse_edge_batch_init_time_sec", NA_real_),
    sparse_edge_batch_optimizer_time_sec = ctx_default("sparse_edge_batch_optimizer_time_sec", NA_real_),
    sparse_edge_batch_status = ctx_default("sparse_edge_batch_status", NA_character_),
    sparse_edge_batch_note = ctx_default("sparse_edge_batch_note", NA_character_),
    vectorized_edge_enabled = ctx$vectorized_edge_enabled,
    vectorized_edge_backend = ctx$vectorized_edge_backend,
    vectorized_edge_storage = ctx$vectorized_edge_storage,
    vectorized_edge_batch_size = ctx$vectorized_edge_batch_size,
    vectorized_edge_n_edges = ctx$vectorized_edge_n_edges,
    vectorized_edge_n_epochs = ctx$vectorized_edge_n_epochs,
    vectorized_edge_negative_sample_rate = ctx$vectorized_edge_negative_sample_rate,
    vectorized_edge_threads = ctx$vectorized_edge_threads,
    vectorized_edge_learning_rate = ctx$vectorized_edge_learning_rate,
    vectorized_edge_simd = ctx$vectorized_edge_simd,
    vectorized_edge_gpu_native = ctx$vectorized_edge_gpu_native,
    vectorized_edge_graph_time_sec = ctx$vectorized_edge_graph_time_sec,
    vectorized_edge_init_time_sec = ctx$vectorized_edge_init_time_sec,
    vectorized_edge_optimizer_time_sec = ctx$vectorized_edge_optimizer_time_sec,
    vectorized_edge_status = ctx$vectorized_edge_status,
    vectorized_edge_note = ctx$vectorized_edge_note,
    atomic_sgd_enabled = ctx$atomic_sgd_enabled,
    atomic_sgd_backend = ctx$atomic_sgd_backend,
    atomic_sgd_update_mode = ctx$atomic_sgd_update_mode,
    atomic_sgd_storage = ctx$atomic_sgd_storage,
    atomic_sgd_n_edges = ctx$atomic_sgd_n_edges,
    atomic_sgd_n_epochs = ctx$atomic_sgd_n_epochs,
    atomic_sgd_negative_sample_rate = ctx$atomic_sgd_negative_sample_rate,
    atomic_sgd_threads = ctx$atomic_sgd_threads,
    atomic_sgd_learning_rate = ctx$atomic_sgd_learning_rate,
    atomic_sgd_learning_rate_scale = ctx$atomic_sgd_learning_rate_scale,
    atomic_sgd_coordinate_clip = ctx$atomic_sgd_coordinate_clip,
    atomic_sgd_openmp = ctx$atomic_sgd_openmp,
    atomic_sgd_gpu_native = ctx$atomic_sgd_gpu_native,
    atomic_sgd_nondeterministic = ctx$atomic_sgd_nondeterministic,
    atomic_sgd_graph_time_sec = ctx$atomic_sgd_graph_time_sec,
    atomic_sgd_init_time_sec = ctx$atomic_sgd_init_time_sec,
    atomic_sgd_optimizer_time_sec = ctx$atomic_sgd_optimizer_time_sec,
    atomic_sgd_status = ctx$atomic_sgd_status,
    atomic_sgd_note = ctx$atomic_sgd_note,
    tsne_bh_theta = ctx$tsne_bh_theta,
    tsne_bh_perplexity = ctx$tsne_bh_perplexity,
    tsne_bh_n_epochs = ctx$tsne_bh_n_epochs,
    tsne_bh_learning_rate = ctx$tsne_bh_learning_rate,
    tsne_bh_n_threads = ctx$tsne_bh_n_threads,
    tsne_bh_stop_lying_iter = ctx$tsne_bh_stop_lying_iter,
    tsne_bh_mom_switch_iter = ctx$tsne_bh_mom_switch_iter,
    fft_interpolation_mode = ctx$fft_interpolation_mode,
    fft_interpolation_experimental = ctx$fft_interpolation_experimental,
    fft_interpolation_backend = ctx$fft_interpolation_backend,
    fft_interpolation_transfer_scope = ctx$fft_interpolation_transfer_scope,
    fft_interpolation_native_repulsive_field = ctx$fft_interpolation_native_repulsive_field,
    fft_interpolation_refine_epochs = ctx$fft_interpolation_refine_epochs,
    fft_grid_nterms = ctx$fft_grid_nterms,
    fft_grid_intervals_per_integer = ctx$fft_grid_intervals_per_integer,
    fft_grid_min_num_intervals = ctx$fft_grid_min_num_intervals,
    output_metric = ctx$output_metric,
    output_metric_transform = ctx$output_metric_transform,
    output_metric_native = ctx$output_metric_native,
    output_metric_projection_scale = ctx$output_metric_projection_scale,
    output_metric_curvature = ctx$output_metric_curvature,
    output_metric_radius_mean = ctx$output_metric_radius_mean,
    output_metric_radius_max = ctx$output_metric_radius_max,
    output_metric_norm_mean = ctx$output_metric_norm_mean,
    embedding_time_sec = NA_real_,
    total_time_sec = NA_real_,
    total_with_knn_time_sec = NA_real_,
    peak_ram_mb = NA_real_,
    peak_gpu_gb = NA_real_,
    trustworthiness = NA_real_,
    continuity = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    distance_spearman = NA_real_,
    distance_pearson = NA_real_,
    stress = NA_real_,
    output_metric_distance_spearman = NA_real_,
    output_metric_distance_pearson = NA_real_,
    output_metric_stress = NA_real_,
    output_metric_global_sample_size = NA_real_,
    output_metric_global_pair_count = NA_real_,
    density_spearman = NA_real_,
    density_pearson = NA_real_,
    density_log_radius_rmse = NA_real_,
    density_radius_high_mean = NA_real_,
    density_radius_embedding_mean = NA_real_,
    density_sample_size = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    rare_class_recall = NA_real_,
    ari = NA_real_,
    nmi = NA_real_,
    procrustes_rmsd = NA_real_,
    neighbour_stability = NA_real_,
    cluster_stability_ari = NA_real_,
    cluster_stability_nmi = NA_real_,
    layout_path = NA_character_,
    stringsAsFactors = FALSE
  )
}

success_row <- function(ctx, strategy, measured, metrics, layout_path) {
  cfg <- attr(measured$layout, "fastEmbedR_config")
  if (inherits(measured$value, "fastEmbedR_embedding")) {
    cfg <- measured$value$parameters
  }
  backend_used <- if (is.list(cfg) && !is.null(cfg$backend)) as.character(cfg$backend) else ctx$backend
  params <- tryCatch(strategy$params(ctx), error = function(e) list())
  cfg_chr <- function(name, default = NA_character_) {
    if (is.list(cfg) && !is.null(cfg[[name]])) safe_character(cfg[[name]], default) else default
  }
  cfg_num <- function(name, default = NA_real_) {
    if (is.list(cfg) && !is.null(cfg[[name]])) safe_number(cfg[[name]], default) else default
  }
  cfg_log <- function(name, default = NA) {
    if (is.list(cfg) && !is.null(cfg[[name]])) safe_logical(cfg[[name]], default) else default
  }
  data.frame(
    method = ctx$method,
    approximation = strategy$id,
    approximation_family = strategy$family,
    knn_reuse_mode = ctx$knn_reuse_mode,
    knn_cache_hit = ctx$knn_cache_hit,
    knn_graph_key = ctx$knn_graph_key,
    knn_graph_source = ctx$knn_graph_source,
    knn_disk_cache_hit = ctx$knn_disk_cache_hit,
    knn_disk_cache_format = ctx$knn_disk_cache_format,
    knn_disk_cache_path = ctx$knn_disk_cache_path,
    backend = ctx$backend,
    backend_used = backend_used,
    gpu_transfer_policy = cfg_chr("gpu_transfer_policy", ctx$gpu_transfer_policy),
    gpu_transfer_backend = cfg_chr("gpu_transfer_backend", ctx$gpu_transfer_backend),
    gpu_transfer_host_to_device_count = cfg_num("gpu_transfer_host_to_device_count", ctx$gpu_transfer_host_to_device_count),
    gpu_transfer_device_to_host_count = cfg_num("gpu_transfer_device_to_host_count", ctx$gpu_transfer_device_to_host_count),
    gpu_transfer_host_to_device_bytes = cfg_num("gpu_transfer_host_to_device_bytes", ctx$gpu_transfer_host_to_device_bytes),
    gpu_transfer_device_to_host_bytes = cfg_num("gpu_transfer_device_to_host_bytes", ctx$gpu_transfer_device_to_host_bytes),
    gpu_transfer_host_to_device_mb = cfg_num("gpu_transfer_host_to_device_mb", ctx$gpu_transfer_host_to_device_mb),
    gpu_transfer_device_to_host_mb = cfg_num("gpu_transfer_device_to_host_mb", ctx$gpu_transfer_device_to_host_mb),
    gpu_transfer_knn_uploaded_once = cfg_log("gpu_transfer_knn_uploaded_once", ctx$gpu_transfer_knn_uploaded_once),
    gpu_transfer_init_uploaded_once = cfg_log("gpu_transfer_init_uploaded_once", ctx$gpu_transfer_init_uploaded_once),
    gpu_transfer_embedding_returned_only_at_end = cfg_log("gpu_transfer_embedding_returned_only_at_end", ctx$gpu_transfer_embedding_returned_only_at_end),
    gpu_transfer_init_roundtrip = cfg_log("gpu_transfer_init_roundtrip", ctx$gpu_transfer_init_roundtrip),
    gpu_transfer_graph_metadata_roundtrip = cfg_log("gpu_transfer_graph_metadata_roundtrip", ctx$gpu_transfer_graph_metadata_roundtrip),
    gpu_transfer_graph_prepared_on_device = cfg_log("gpu_transfer_graph_prepared_on_device", ctx$gpu_transfer_graph_prepared_on_device),
    gpu_transfer_note = cfg_chr("gpu_transfer_note", ctx$gpu_transfer_note),
    dataset = ctx$dataset_name,
    n = ctx$n,
    p = ctx$p,
    seed = ctx$seed,
    parameter_settings = json_or_text(params),
    status = "success",
    error_message = NA_character_,
    preprocessing_time_sec = if (inherits(measured$value, "fastEmbedR_embedding")) measured$value$timings["preprocess", "elapsed"] else NA_real_,
    knn_time_sec = if (is.finite(ctx$knn_time_sec)) ctx$knn_time_sec else if (inherits(measured$value, "fastEmbedR_embedding")) measured$value$timings["knn", "elapsed"] else NA_real_,
    knn_graph_time_sec = ctx$knn_graph_time_sec,
    knn_index_build_time_sec = ctx$knn_index_build_time_sec,
    knn_query_time_sec = ctx$knn_query_time_sec,
    knn_graph_index_build_time_sec = ctx$knn_graph_index_build_time_sec,
    knn_graph_query_time_sec = ctx$knn_graph_query_time_sec,
    knn_disk_load_time_sec = ctx$knn_disk_load_time_sec,
    knn_disk_save_time_sec = ctx$knn_disk_save_time_sec,
    knn_memory_mb = ctx$knn_memory_mb,
    knn_recall_at_k = ctx$knn_recall_at_k,
    knn_mean_distance_error = ctx$knn_mean_distance_error,
    knn_rank_correlation = ctx$knn_rank_correlation,
    knn_quality_sample_size = ctx$knn_quality_sample_size,
    graph_approximation = cfg_chr("graph_approximation", ctx$graph_approximation),
    graph_approximation_time_sec = ctx$graph_approximation_time_sec,
    graph_storage_format = ctx$graph_storage_format,
    graph_sparse_nnz = ctx$graph_sparse_nnz,
    graph_sparse_internal_memory_mb = ctx$graph_sparse_internal_memory_mb,
    graph_sparse_r_memory_mb = ctx$graph_sparse_r_memory_mb,
    graph_dense_knn_memory_mb = ctx$graph_dense_knn_memory_mb,
    graph_sparse_internal_memory_ratio = ctx$graph_sparse_internal_memory_ratio,
    graph_sparse_r_memory_ratio = ctx$graph_sparse_r_memory_ratio,
    graph_sparse_prune_weight = ctx$graph_sparse_prune_weight,
    graph_sparse_mean_weight = ctx$graph_sparse_mean_weight,
    graph_sparse_min_weight = ctx$graph_sparse_min_weight,
    graph_sparse_max_weight = ctx$graph_sparse_max_weight,
    graph_effective_k = cfg_num("graph_effective_k", ctx$graph_effective_k),
    graph_edge_retention = cfg_num("graph_edge_retention", ctx$graph_edge_retention),
    graph_recall_at_k = ctx$graph_recall_at_k,
    graph_mean_distance_error = ctx$graph_mean_distance_error,
    graph_rank_correlation = ctx$graph_rank_correlation,
    graph_quality_sample_size = ctx$graph_quality_sample_size,
    graph_mean_degree = ctx$graph_mean_degree,
    graph_min_degree = ctx$graph_min_degree,
    graph_max_degree = ctx$graph_max_degree,
    graph_isolated_fraction = ctx$graph_isolated_fraction,
    graph_padding_fraction = ctx$graph_padding_fraction,
    graph_mean_jaccard = ctx$graph_mean_jaccard,
    graph_min_jaccard = ctx$graph_min_jaccard,
    graph_max_jaccard = ctx$graph_max_jaccard,
    graph_zero_jaccard_fraction = ctx$graph_zero_jaccard_fraction,
    graph_snn_k = ctx$graph_snn_k,
    graph_snn_prune_threshold = ctx$graph_snn_prune_threshold,
    graph_mean_snn_weight = ctx$graph_mean_snn_weight,
    graph_min_snn_weight = ctx$graph_min_snn_weight,
    graph_max_snn_weight = ctx$graph_max_snn_weight,
    graph_zero_snn_fraction = ctx$graph_zero_snn_fraction,
    localmap_false_neighbor_enabled = cfg_log("localmap_false_neighbor_enabled", ctx$localmap_false_neighbor_enabled),
    localmap_false_neighbor_mode = cfg_chr("localmap_false_neighbor_mode", ctx$localmap_false_neighbor_mode),
    localmap_false_neighbor_transfer_mode = cfg_chr("localmap_false_neighbor_transfer_mode", ctx$localmap_false_neighbor_transfer_mode),
    localmap_false_neighbor_jaccard_threshold = cfg_num("localmap_false_neighbor_jaccard_threshold", ctx$localmap_false_neighbor_jaccard_threshold),
    localmap_false_neighbor_distance_quantile = cfg_num("localmap_false_neighbor_distance_quantile", ctx$localmap_false_neighbor_distance_quantile),
    localmap_false_neighbor_distance_multiplier = cfg_num("localmap_false_neighbor_distance_multiplier", ctx$localmap_false_neighbor_distance_multiplier),
    localmap_false_neighbor_min_keep_fraction = cfg_num("localmap_false_neighbor_min_keep_fraction", ctx$localmap_false_neighbor_min_keep_fraction),
    localmap_false_neighbor_min_keep_k = cfg_num("localmap_false_neighbor_min_keep_k", ctx$localmap_false_neighbor_min_keep_k),
    localmap_false_neighbor_removed_edges_mean = cfg_num("localmap_false_neighbor_removed_edges_mean", ctx$localmap_false_neighbor_removed_edges_mean),
    localmap_false_neighbor_removed_fraction = cfg_num("localmap_false_neighbor_removed_fraction", ctx$localmap_false_neighbor_removed_fraction),
    localmap_false_neighbor_kept_degree_mean = cfg_num("localmap_false_neighbor_kept_degree_mean", ctx$localmap_false_neighbor_kept_degree_mean),
    localmap_false_neighbor_kept_jaccard_mean = cfg_num("localmap_false_neighbor_kept_jaccard_mean", ctx$localmap_false_neighbor_kept_jaccard_mean),
    localmap_false_neighbor_removed_jaccard_mean = cfg_num("localmap_false_neighbor_removed_jaccard_mean", ctx$localmap_false_neighbor_removed_jaccard_mean),
    localmap_false_neighbor_kept_distance_ratio_mean = cfg_num("localmap_false_neighbor_kept_distance_ratio_mean", ctx$localmap_false_neighbor_kept_distance_ratio_mean),
    localmap_false_neighbor_removed_distance_ratio_mean = cfg_num("localmap_false_neighbor_removed_distance_ratio_mean", ctx$localmap_false_neighbor_removed_distance_ratio_mean),
    localmap_false_neighbor_threshold_mean = cfg_num("localmap_false_neighbor_threshold_mean", ctx$localmap_false_neighbor_threshold_mean),
    localmap_local_weight_enabled = cfg_log("localmap_local_weight_enabled", ctx$localmap_local_weight_enabled),
    localmap_local_weight = cfg_num("localmap_local_weight", ctx$localmap_local_weight),
    localmap_local_weight_mode = cfg_chr("localmap_local_weight_mode", ctx$localmap_local_weight_mode),
    localmap_local_weight_transfer_mode = cfg_chr("localmap_local_weight_transfer_mode", ctx$localmap_local_weight_transfer_mode),
    localmap_local_weight_jaccard_blend = cfg_num("localmap_local_weight_jaccard_blend", ctx$localmap_local_weight_jaccard_blend),
    localmap_local_weight_mean_trust = cfg_num("localmap_local_weight_mean_trust", ctx$localmap_local_weight_mean_trust),
    localmap_local_weight_rank_component_mean = cfg_num("localmap_local_weight_rank_component_mean", ctx$localmap_local_weight_rank_component_mean),
    localmap_local_weight_jaccard_component_mean = cfg_num("localmap_local_weight_jaccard_component_mean", ctx$localmap_local_weight_jaccard_component_mean),
    localmap_local_weight_mean_multiplier = cfg_num("localmap_local_weight_mean_multiplier", ctx$localmap_local_weight_mean_multiplier),
    localmap_local_weight_min_multiplier = cfg_num("localmap_local_weight_min_multiplier", ctx$localmap_local_weight_min_multiplier),
    localmap_local_weight_max_multiplier = cfg_num("localmap_local_weight_max_multiplier", ctx$localmap_local_weight_max_multiplier),
    localmap_local_weight_distance_scale_mean = cfg_num("localmap_local_weight_distance_scale_mean", ctx$localmap_local_weight_distance_scale_mean),
    artificial_neighbor_penalty_enabled = cfg_log("artificial_neighbor_penalty_enabled", ctx$artificial_neighbor_penalty_enabled),
    artificial_neighbor_transfer_mode = cfg_chr("artificial_neighbor_transfer_mode", ctx$artificial_neighbor_transfer_mode),
    artificial_neighbor_refinement_backend = cfg_chr("artificial_neighbor_refinement_backend", ctx$artificial_neighbor_refinement_backend),
    artificial_neighbor_penalty_strength = cfg_num("artificial_neighbor_penalty_strength", ctx$artificial_neighbor_penalty_strength),
    artificial_neighbor_penalty_iterations = cfg_num("artificial_neighbor_penalty_iterations", ctx$artificial_neighbor_penalty_iterations),
    artificial_neighbor_penalty_low_k = cfg_num("artificial_neighbor_penalty_low_k", ctx$artificial_neighbor_penalty_low_k),
    artificial_neighbor_penalty_far_multiplier = cfg_num("artificial_neighbor_penalty_far_multiplier", ctx$artificial_neighbor_penalty_far_multiplier),
    artificial_neighbor_penalty_target_distance = cfg_num("artificial_neighbor_penalty_target_distance", ctx$artificial_neighbor_penalty_target_distance),
    artificial_neighbor_penalized_pairs = cfg_num("artificial_neighbor_penalized_pairs", ctx$artificial_neighbor_penalized_pairs),
    artificial_neighbor_total_low_edges = cfg_num("artificial_neighbor_total_low_edges", ctx$artificial_neighbor_total_low_edges),
    artificial_neighbor_false_rate_before = cfg_num("artificial_neighbor_false_rate_before", ctx$artificial_neighbor_false_rate_before),
    artificial_neighbor_false_rate_after = cfg_num("artificial_neighbor_false_rate_after", ctx$artificial_neighbor_false_rate_after),
    artificial_neighbor_false_rate_delta = cfg_num("artificial_neighbor_false_rate_delta", ctx$artificial_neighbor_false_rate_delta),
    artificial_neighbor_far_rate_before = cfg_num("artificial_neighbor_far_rate_before", ctx$artificial_neighbor_far_rate_before),
    artificial_neighbor_far_rate_after = cfg_num("artificial_neighbor_far_rate_after", ctx$artificial_neighbor_far_rate_after),
    artificial_neighbor_far_rate_delta = cfg_num("artificial_neighbor_far_rate_delta", ctx$artificial_neighbor_far_rate_delta),
    artificial_neighbor_mean_high_distance_ratio = cfg_num("artificial_neighbor_mean_high_distance_ratio", ctx$artificial_neighbor_mean_high_distance_ratio),
    artificial_neighbor_mean_low_distance = cfg_num("artificial_neighbor_mean_low_distance", ctx$artificial_neighbor_mean_low_distance),
    false_neighbor_monitor_enabled = cfg_log("false_neighbor_monitor_enabled", ctx$false_neighbor_monitor_enabled),
    false_neighbor_monitor_transfer_mode = cfg_chr("false_neighbor_monitor_transfer_mode", ctx$false_neighbor_monitor_transfer_mode),
    false_neighbor_monitor_backend = cfg_chr("false_neighbor_monitor_backend", ctx$false_neighbor_monitor_backend),
    false_neighbor_monitor_action = cfg_chr("false_neighbor_monitor_action", ctx$false_neighbor_monitor_action),
    false_neighbor_monitor_start_mode = cfg_chr("false_neighbor_monitor_start_mode", ctx$false_neighbor_monitor_start_mode),
    false_neighbor_monitor_chunk_epochs = cfg_num("false_neighbor_monitor_chunk_epochs", ctx$false_neighbor_monitor_chunk_epochs),
    false_neighbor_monitor_max_chunks = cfg_num("false_neighbor_monitor_max_chunks", ctx$false_neighbor_monitor_max_chunks),
    false_neighbor_monitor_chunks_run = cfg_num("false_neighbor_monitor_chunks_run", ctx$false_neighbor_monitor_chunks_run),
    false_neighbor_monitor_epochs_requested = cfg_num("false_neighbor_monitor_epochs_requested", ctx$false_neighbor_monitor_epochs_requested),
    false_neighbor_monitor_epochs_completed = cfg_num("false_neighbor_monitor_epochs_completed", ctx$false_neighbor_monitor_epochs_completed),
    false_neighbor_monitor_patience = cfg_num("false_neighbor_monitor_patience", ctx$false_neighbor_monitor_patience),
    false_neighbor_monitor_tolerance = cfg_num("false_neighbor_monitor_tolerance", ctx$false_neighbor_monitor_tolerance),
    false_neighbor_monitor_low_k = cfg_num("false_neighbor_monitor_low_k", ctx$false_neighbor_monitor_low_k),
    false_neighbor_monitor_far_multiplier = cfg_num("false_neighbor_monitor_far_multiplier", ctx$false_neighbor_monitor_far_multiplier),
    false_neighbor_monitor_far_weight = cfg_num("false_neighbor_monitor_far_weight", ctx$false_neighbor_monitor_far_weight),
    false_neighbor_monitor_initial_false_rate = cfg_num("false_neighbor_monitor_initial_false_rate", ctx$false_neighbor_monitor_initial_false_rate),
    false_neighbor_monitor_final_false_rate = cfg_num("false_neighbor_monitor_final_false_rate", ctx$false_neighbor_monitor_final_false_rate),
    false_neighbor_monitor_best_false_rate = cfg_num("false_neighbor_monitor_best_false_rate", ctx$false_neighbor_monitor_best_false_rate),
    false_neighbor_monitor_false_rate_delta = cfg_num("false_neighbor_monitor_false_rate_delta", ctx$false_neighbor_monitor_false_rate_delta),
    false_neighbor_monitor_initial_far_rate = cfg_num("false_neighbor_monitor_initial_far_rate", ctx$false_neighbor_monitor_initial_far_rate),
    false_neighbor_monitor_final_far_rate = cfg_num("false_neighbor_monitor_final_far_rate", ctx$false_neighbor_monitor_final_far_rate),
    false_neighbor_monitor_best_far_rate = cfg_num("false_neighbor_monitor_best_far_rate", ctx$false_neighbor_monitor_best_far_rate),
    false_neighbor_monitor_far_rate_delta = cfg_num("false_neighbor_monitor_far_rate_delta", ctx$false_neighbor_monitor_far_rate_delta),
    false_neighbor_monitor_score_initial = cfg_num("false_neighbor_monitor_score_initial", ctx$false_neighbor_monitor_score_initial),
    false_neighbor_monitor_score_final = cfg_num("false_neighbor_monitor_score_final", ctx$false_neighbor_monitor_score_final),
    false_neighbor_monitor_score_best = cfg_num("false_neighbor_monitor_score_best", ctx$false_neighbor_monitor_score_best),
    false_neighbor_monitor_score_delta = cfg_num("false_neighbor_monitor_score_delta", ctx$false_neighbor_monitor_score_delta),
    false_neighbor_monitor_worsening_events = cfg_num("false_neighbor_monitor_worsening_events", ctx$false_neighbor_monitor_worsening_events),
    false_neighbor_monitor_adjustments = cfg_num("false_neighbor_monitor_adjustments", ctx$false_neighbor_monitor_adjustments),
    false_neighbor_monitor_stopped_early = cfg_log("false_neighbor_monitor_stopped_early", ctx$false_neighbor_monitor_stopped_early),
    false_neighbor_monitor_score_trace = cfg_chr("false_neighbor_monitor_score_trace", ctx$false_neighbor_monitor_score_trace),
    false_neighbor_monitor_false_rate_trace = cfg_chr("false_neighbor_monitor_false_rate_trace", ctx$false_neighbor_monitor_false_rate_trace),
    false_neighbor_monitor_far_rate_trace = cfg_chr("false_neighbor_monitor_far_rate_trace", ctx$false_neighbor_monitor_far_rate_trace),
    false_neighbor_monitor_chunk_trace = cfg_chr("false_neighbor_monitor_chunk_trace", ctx$false_neighbor_monitor_chunk_trace),
    init_strategy = cfg_chr("init_strategy", ctx$init_strategy),
    init_backend = cfg_chr("init_backend", ctx$init_backend),
    init_backend_reason = cfg_chr("init_backend_reason", ctx$init_backend_reason),
    init_time_sec = cfg_num("init_time_sec", ctx$init_time_sec),
    init_optimizer_epochs = cfg_num("init_optimizer_epochs", ctx$init_optimizer_epochs),
    init_optimizer_time_sec = cfg_num("init_optimizer_time_sec", ctx$init_optimizer_time_sec),
    init_scale = cfg_num("init_scale", ctx$init_scale),
    init_spectral_n_iter = cfg_num("init_spectral_n_iter", ctx$init_spectral_n_iter),
    init_spectral_solver = cfg_chr("init_spectral_solver", ctx$init_spectral_solver),
    init_spectral_graph = cfg_chr("init_spectral_graph", ctx$init_spectral_graph),
    init_spectral_eigenvalues = cfg_chr("init_spectral_eigenvalues", ctx$init_spectral_eigenvalues),
    init_spectral_exact_max_n = cfg_num("init_spectral_exact_max_n", ctx$init_spectral_exact_max_n),
    init_spectral_graph_nnz = cfg_num("init_spectral_graph_nnz", ctx$init_spectral_graph_nnz),
    init_spectral_graph_active_fraction = cfg_num("init_spectral_graph_active_fraction", ctx$init_spectral_graph_active_fraction),
    init_spectral_nystrom_landmarks = cfg_num("init_spectral_nystrom_landmarks", ctx$init_spectral_nystrom_landmarks),
    init_spectral_nystrom_fraction = cfg_num("init_spectral_nystrom_fraction", ctx$init_spectral_nystrom_fraction),
    init_spectral_nystrom_projection_k = cfg_num("init_spectral_nystrom_projection_k", ctx$init_spectral_nystrom_projection_k),
    init_spectral_nystrom_weight = cfg_chr("init_spectral_nystrom_weight", ctx$init_spectral_nystrom_weight),
    init_spectral_nystrom_selection_requested = cfg_chr("init_spectral_nystrom_selection_requested", ctx$init_spectral_nystrom_selection_requested),
    init_spectral_nystrom_selection_used = cfg_chr("init_spectral_nystrom_selection_used", ctx$init_spectral_nystrom_selection_used),
    init_spectral_nystrom_landmark_knn_time_sec = cfg_num("init_spectral_nystrom_landmark_knn_time_sec", ctx$init_spectral_nystrom_landmark_knn_time_sec),
    init_spectral_nystrom_landmark_spectral_time_sec = cfg_num("init_spectral_nystrom_landmark_spectral_time_sec", ctx$init_spectral_nystrom_landmark_spectral_time_sec),
    init_spectral_nystrom_projection_time_sec = cfg_num("init_spectral_nystrom_projection_time_sec", ctx$init_spectral_nystrom_projection_time_sec),
    init_diffusion_time = cfg_num("init_diffusion_time", ctx$init_diffusion_time),
    init_diffusion_n_iter = cfg_num("init_diffusion_n_iter", ctx$init_diffusion_n_iter),
    init_diffusion_solver = cfg_chr("init_diffusion_solver", ctx$init_diffusion_solver),
    init_diffusion_graph = cfg_chr("init_diffusion_graph", ctx$init_diffusion_graph),
    init_diffusion_eigenvalues = cfg_chr("init_diffusion_eigenvalues", ctx$init_diffusion_eigenvalues),
    init_diffusion_graph_nnz = cfg_num("init_diffusion_graph_nnz", ctx$init_diffusion_graph_nnz),
    init_diffusion_graph_active_fraction = cfg_num("init_diffusion_graph_active_fraction", ctx$init_diffusion_graph_active_fraction),
    init_laplacian_n_iter = cfg_num("init_laplacian_n_iter", ctx$init_laplacian_n_iter),
    init_laplacian_solver = cfg_chr("init_laplacian_solver", ctx$init_laplacian_solver),
    init_laplacian_graph = cfg_chr("init_laplacian_graph", ctx$init_laplacian_graph),
    init_laplacian_eigenvalues = cfg_chr("init_laplacian_eigenvalues", ctx$init_laplacian_eigenvalues),
    init_laplacian_graph_nnz = cfg_num("init_laplacian_graph_nnz", ctx$init_laplacian_graph_nnz),
    init_laplacian_graph_active_fraction = cfg_num("init_laplacian_graph_active_fraction", ctx$init_laplacian_graph_active_fraction),
    init_laplacian_normalized_coordinates = cfg_log("init_laplacian_normalized_coordinates", ctx$init_laplacian_normalized_coordinates),
    init_pca_method = cfg_chr("init_pca_method", ctx$init_pca_method),
    init_pca_oversample = cfg_num("init_pca_oversample", ctx$init_pca_oversample),
    init_pca_power = cfg_num("init_pca_power", ctx$init_pca_power),
    init_landmark_n = cfg_num("init_landmark_n", ctx$init_landmark_n),
    init_landmark_fraction = cfg_num("init_landmark_fraction", ctx$init_landmark_fraction),
    init_landmark_selection_requested = cfg_chr("init_landmark_selection_requested", ctx$init_landmark_selection_requested),
    init_landmark_selection_used = cfg_chr("init_landmark_selection_used", ctx$init_landmark_selection_used),
    init_projection_k = cfg_num("init_projection_k", ctx$init_projection_k),
    init_projection_weight = cfg_chr("init_projection_weight", ctx$init_projection_weight),
    init_landmark_epochs = cfg_num("init_landmark_epochs", ctx$init_landmark_epochs),
    init_landmark_knn_time_sec = cfg_num("init_landmark_knn_time_sec", ctx$init_landmark_knn_time_sec),
    init_landmark_embedding_time_sec = cfg_num("init_landmark_embedding_time_sec", ctx$init_landmark_embedding_time_sec),
    init_projection_time_sec = cfg_num("init_projection_time_sec", ctx$init_projection_time_sec),
    init_projection_backend = cfg_chr("init_projection_backend", ctx$init_projection_backend),
    warm_start_enabled = cfg_log("warm_start_enabled", ctx$warm_start_enabled),
    warm_start_cache_hit = cfg_log("warm_start_cache_hit", ctx$warm_start_cache_hit),
    warm_start_cache_key = cfg_chr("warm_start_cache_key", ctx$warm_start_cache_key),
    warm_start_previous_init = cfg_chr("warm_start_previous_init", ctx$warm_start_previous_init),
    warm_start_previous_epochs = cfg_num("warm_start_previous_epochs", ctx$warm_start_previous_epochs),
    warm_start_refinement_epochs = cfg_num("warm_start_refinement_epochs", ctx$warm_start_refinement_epochs),
    warm_start_previous_init_time_sec = cfg_num("warm_start_previous_init_time_sec", ctx$warm_start_previous_init_time_sec),
    warm_start_previous_embedding_time_sec = cfg_num("warm_start_previous_embedding_time_sec", ctx$warm_start_previous_embedding_time_sec),
    warm_start_previous_build_time_sec = cfg_num("warm_start_previous_build_time_sec", ctx$warm_start_previous_build_time_sec),
    warm_start_this_row_setup_time_sec = cfg_num("warm_start_this_row_setup_time_sec", ctx$warm_start_this_row_setup_time_sec),
    warm_start_refinement_time_sec = cfg_num("warm_start_refinement_time_sec", ctx$warm_start_refinement_time_sec),
    warm_start_total_if_cache_miss_sec = cfg_num("warm_start_total_if_cache_miss_sec", ctx$warm_start_total_if_cache_miss_sec),
    warm_start_reuse_mode = cfg_chr("warm_start_reuse_mode", ctx$warm_start_reuse_mode),
    warm_start_use_case = cfg_chr("warm_start_use_case", ctx$warm_start_use_case),
    warm_start_parameter_delta = cfg_chr("warm_start_parameter_delta", ctx$warm_start_parameter_delta),
    warm_start_bias_risk = cfg_chr("warm_start_bias_risk", ctx$warm_start_bias_risk),
    epoch_budget_enabled = cfg_log("epoch_budget_enabled", ctx$epoch_budget_enabled),
    epoch_budget_requested = cfg_chr("epoch_budget_requested", ctx$epoch_budget_requested),
    epoch_budget_effective = cfg_num("epoch_budget_effective", ctx$epoch_budget_effective),
    epoch_budget_default_epochs = cfg_num("epoch_budget_default_epochs", ctx$epoch_budget_default_epochs),
    epoch_budget_ratio_to_default = cfg_num("epoch_budget_ratio_to_default", ctx$epoch_budget_ratio_to_default),
    epoch_budget_quality = cfg_chr("epoch_budget_quality", ctx$epoch_budget_quality),
    epoch_budget_tsne_mode = cfg_chr("epoch_budget_tsne_mode", ctx$epoch_budget_tsne_mode),
    epoch_budget_optimizer_backend = cfg_chr("epoch_budget_optimizer_backend", ctx$epoch_budget_optimizer_backend),
    epoch_budget_speed_quality_tradeoff = cfg_chr("epoch_budget_speed_quality_tradeoff", ctx$epoch_budget_speed_quality_tradeoff),
    epoch_budget_is_default = cfg_log("epoch_budget_is_default", ctx$epoch_budget_is_default),
    early_stop_enabled = cfg_log("early_stop_enabled", ctx$early_stop_enabled),
    early_stop_criterion = cfg_chr("early_stop_criterion", ctx$early_stop_criterion),
    early_stop_status = cfg_chr("early_stop_status", ctx$early_stop_status),
    early_stop_reason = cfg_chr("early_stop_reason", ctx$early_stop_reason),
    early_stop_max_epochs = cfg_num("early_stop_max_epochs", ctx$early_stop_max_epochs),
    early_stop_chunk_epochs = cfg_num("early_stop_chunk_epochs", ctx$early_stop_chunk_epochs),
    early_stop_epochs_run = cfg_num("early_stop_epochs_run", ctx$early_stop_epochs_run),
    early_stop_chunks_run = cfg_num("early_stop_chunks_run", ctx$early_stop_chunks_run),
    early_stop_patience = cfg_num("early_stop_patience", ctx$early_stop_patience),
    early_stop_tolerance = cfg_num("early_stop_tolerance", ctx$early_stop_tolerance),
    early_stop_displacement_final = cfg_num("early_stop_displacement_final", ctx$early_stop_displacement_final),
    early_stop_trustworthiness_final = cfg_num("early_stop_trustworthiness_final", ctx$early_stop_trustworthiness_final),
    early_stop_trustworthiness_delta_final = cfg_num("early_stop_trustworthiness_delta_final", ctx$early_stop_trustworthiness_delta_final),
    early_stop_neighbour_stability_final = cfg_num("early_stop_neighbour_stability_final", ctx$early_stop_neighbour_stability_final),
    early_stop_neighbour_stability_delta_final = cfg_num("early_stop_neighbour_stability_delta_final", ctx$early_stop_neighbour_stability_delta_final),
    early_stop_monitor_sample_size = cfg_num("early_stop_monitor_sample_size", ctx$early_stop_monitor_sample_size),
    early_stop_monitor_k = cfg_num("early_stop_monitor_k", ctx$early_stop_monitor_k),
    early_stop_init_strategy = cfg_chr("early_stop_init_strategy", ctx$early_stop_init_strategy),
    early_stop_init_time_sec = cfg_num("early_stop_init_time_sec", ctx$early_stop_init_time_sec),
    early_stop_optimizer_time_sec = cfg_num("early_stop_optimizer_time_sec", ctx$early_stop_optimizer_time_sec),
    early_stop_loss_available = cfg_log("early_stop_loss_available", ctx$early_stop_loss_available),
    early_stop_loss_reason = cfg_chr("early_stop_loss_reason", ctx$early_stop_loss_reason),
    early_stop_chunked_optimizer = cfg_log("early_stop_chunked_optimizer", ctx$early_stop_chunked_optimizer),
    early_stop_trace_epochs = cfg_chr("early_stop_trace_epochs", ctx$early_stop_trace_epochs),
    early_stop_trace_displacement = cfg_chr("early_stop_trace_displacement", ctx$early_stop_trace_displacement),
    early_stop_trace_trustworthiness = cfg_chr("early_stop_trace_trustworthiness", ctx$early_stop_trace_trustworthiness),
    early_stop_trace_trustworthiness_delta = cfg_chr("early_stop_trace_trustworthiness_delta", ctx$early_stop_trace_trustworthiness_delta),
    early_stop_trace_neighbour_stability = cfg_chr("early_stop_trace_neighbour_stability", ctx$early_stop_trace_neighbour_stability),
    early_stop_trace_neighbour_stability_delta = cfg_chr("early_stop_trace_neighbour_stability_delta", ctx$early_stop_trace_neighbour_stability_delta),
    early_stop_backend_scope = cfg_chr("early_stop_backend_scope", ctx$early_stop_backend_scope),
    early_stop_risk = cfg_chr("early_stop_risk", ctx$early_stop_risk),
    coarse_to_fine_enabled = cfg_log("coarse_to_fine_enabled", ctx$coarse_to_fine_enabled),
    coarse_to_fine_mode = cfg_chr("coarse_to_fine_mode", ctx$coarse_to_fine_mode),
    coarse_to_fine_selection_requested = cfg_chr("coarse_to_fine_selection_requested", ctx$coarse_to_fine_selection_requested),
    coarse_to_fine_selection_used = cfg_chr("coarse_to_fine_selection_used", ctx$coarse_to_fine_selection_used),
    coarse_to_fine_count_requested = cfg_num("coarse_to_fine_count_requested", ctx$coarse_to_fine_count_requested),
    coarse_to_fine_fraction_requested = cfg_num("coarse_to_fine_fraction_requested", ctx$coarse_to_fine_fraction_requested),
    coarse_to_fine_n = cfg_num("coarse_to_fine_n", ctx$coarse_to_fine_n),
    coarse_to_fine_fraction = cfg_num("coarse_to_fine_fraction", ctx$coarse_to_fine_fraction),
    coarse_to_fine_k = cfg_num("coarse_to_fine_k", ctx$coarse_to_fine_k),
    coarse_to_fine_projection_k = cfg_num("coarse_to_fine_projection_k", ctx$coarse_to_fine_projection_k),
    coarse_to_fine_projection_weight = cfg_chr("coarse_to_fine_projection_weight", ctx$coarse_to_fine_projection_weight),
    coarse_to_fine_coarse_epochs = cfg_num("coarse_to_fine_coarse_epochs", ctx$coarse_to_fine_coarse_epochs),
    coarse_to_fine_refinement_epochs = cfg_num("coarse_to_fine_refinement_epochs", ctx$coarse_to_fine_refinement_epochs),
    coarse_to_fine_selection_time_sec = cfg_num("coarse_to_fine_selection_time_sec", ctx$coarse_to_fine_selection_time_sec),
    coarse_to_fine_knn_time_sec = cfg_num("coarse_to_fine_knn_time_sec", ctx$coarse_to_fine_knn_time_sec),
    coarse_to_fine_embedding_time_sec = cfg_num("coarse_to_fine_embedding_time_sec", ctx$coarse_to_fine_embedding_time_sec),
    coarse_to_fine_projection_time_sec = cfg_num("coarse_to_fine_projection_time_sec", ctx$coarse_to_fine_projection_time_sec),
    coarse_to_fine_refinement_time_sec = cfg_num("coarse_to_fine_refinement_time_sec", ctx$coarse_to_fine_refinement_time_sec),
    coarse_to_fine_setup_time_sec = cfg_num("coarse_to_fine_setup_time_sec", ctx$coarse_to_fine_setup_time_sec),
    coarse_to_fine_projection_entropy = cfg_num("coarse_to_fine_projection_entropy", ctx$coarse_to_fine_projection_entropy),
    coarse_to_fine_projection_zero_neighbor_fraction = cfg_num("coarse_to_fine_projection_zero_neighbor_fraction", ctx$coarse_to_fine_projection_zero_neighbor_fraction),
    coarse_to_fine_projection_bandwidth_mean = cfg_num("coarse_to_fine_projection_bandwidth_mean", ctx$coarse_to_fine_projection_bandwidth_mean),
    coarse_to_fine_projection_backend = cfg_chr("coarse_to_fine_projection_backend", ctx$coarse_to_fine_projection_backend),
    coarse_to_fine_refinement_backend = cfg_chr("coarse_to_fine_refinement_backend", ctx$coarse_to_fine_refinement_backend),
    coarse_to_fine_init_backend = cfg_chr("coarse_to_fine_init_backend", ctx$coarse_to_fine_init_backend),
    coarse_to_fine_expected_gain = cfg_chr("coarse_to_fine_expected_gain", ctx$coarse_to_fine_expected_gain),
    coarse_to_fine_risk = cfg_chr("coarse_to_fine_risk", ctx$coarse_to_fine_risk),
    landmark_enabled = cfg_log("landmark", ctx$landmark_enabled),
    landmark_approximation = cfg_chr("landmark_approximation", ctx$landmark_approximation),
    landmark_mode = cfg_chr("landmark_mode", ctx$landmark_mode),
    landmark_selection = cfg_chr("landmark_selection", ctx$landmark_selection),
    landmark_selection_requested = cfg_chr("landmark_selection_requested", ctx$landmark_selection_requested),
    landmark_selection_used = cfg_chr("landmark_selection_used", ctx$landmark_selection_used),
    landmark_count_requested = cfg_num("landmark_count_requested", ctx$landmark_count_requested),
    landmark_fraction_requested = cfg_num("landmark_fraction_requested", ctx$landmark_fraction_requested),
    landmark_n = cfg_num("n_landmarks", ctx$landmark_n),
    landmark_fraction = cfg_num("landmark_fraction", ctx$landmark_fraction),
    landmark_label_classes_total = cfg_num("landmark_label_classes_total", ctx$landmark_label_classes_total),
    landmark_label_classes_present = cfg_num("landmark_label_classes_present", ctx$landmark_label_classes_present),
    landmark_label_missing_classes = cfg_num("landmark_label_missing_classes", ctx$landmark_label_missing_classes),
    landmark_label_min_count = cfg_num("landmark_label_min_count", ctx$landmark_label_min_count),
    landmark_label_min_fraction = cfg_num("landmark_label_min_fraction", ctx$landmark_label_min_fraction),
    landmark_rare_label_count = cfg_num("landmark_rare_label_count", ctx$landmark_rare_label_count),
    landmark_rare_label_present = cfg_log("landmark_rare_label_present", ctx$landmark_rare_label_present),
    stratified_landmark_source = cfg_chr("stratified_landmark_source", ctx$stratified_landmark_source),
    stratified_landmark_allocation = cfg_chr("stratified_landmark_allocation", ctx$stratified_landmark_allocation),
    stratified_landmark_time_sec = cfg_num("stratified_landmark_time_sec", ctx$stratified_landmark_time_sec),
    stratified_landmark_n_strata = cfg_num("stratified_landmark_n_strata", ctx$stratified_landmark_n_strata),
    stratified_landmark_strata_sampled = cfg_num("stratified_landmark_strata_sampled", ctx$stratified_landmark_strata_sampled),
    stratified_landmark_missing_strata = cfg_num("stratified_landmark_missing_strata", ctx$stratified_landmark_missing_strata),
    stratified_landmark_min_stratum_size = cfg_num("stratified_landmark_min_stratum_size", ctx$stratified_landmark_min_stratum_size),
    stratified_landmark_max_stratum_size = cfg_num("stratified_landmark_max_stratum_size", ctx$stratified_landmark_max_stratum_size),
    stratified_landmark_min_selected_per_stratum = cfg_num("stratified_landmark_min_selected_per_stratum", ctx$stratified_landmark_min_selected_per_stratum),
    stratified_landmark_max_selected_per_stratum = cfg_num("stratified_landmark_max_selected_per_stratum", ctx$stratified_landmark_max_selected_per_stratum),
    stratified_landmark_balance_ratio = cfg_num("stratified_landmark_balance_ratio", ctx$stratified_landmark_balance_ratio),
    stratified_landmark_cluster_k = cfg_num("stratified_landmark_cluster_k", ctx$stratified_landmark_cluster_k),
    stratified_landmark_cluster_feature_dims = cfg_num("stratified_landmark_cluster_feature_dims", ctx$stratified_landmark_cluster_feature_dims),
    density_landmark_alpha = cfg_num("density_landmark_alpha", ctx$density_landmark_alpha),
    density_landmark_k = cfg_num("density_landmark_k", ctx$density_landmark_k),
    density_landmark_time_sec = cfg_num("density_landmark_time_sec", ctx$density_landmark_time_sec),
    density_landmark_weight_min = cfg_num("density_landmark_weight_min", ctx$density_landmark_weight_min),
    density_landmark_weight_median = cfg_num("density_landmark_weight_median", ctx$density_landmark_weight_median),
    density_landmark_weight_max = cfg_num("density_landmark_weight_max", ctx$density_landmark_weight_max),
    density_landmark_weight_mean = cfg_num("density_landmark_weight_mean", ctx$density_landmark_weight_mean),
    density_landmark_selected_weight_mean = cfg_num("density_landmark_selected_weight_mean", ctx$density_landmark_selected_weight_mean),
    density_landmark_selected_to_global_weight_ratio = cfg_num("density_landmark_selected_to_global_weight_ratio", ctx$density_landmark_selected_to_global_weight_ratio),
    density_landmark_mean_distance_median = cfg_num("density_landmark_mean_distance_median", ctx$density_landmark_mean_distance_median),
    hybrid_landmark_alpha = cfg_num("hybrid_landmark_alpha", ctx$hybrid_landmark_alpha),
    hybrid_landmark_beta = cfg_num("hybrid_landmark_beta", ctx$hybrid_landmark_beta),
    hybrid_landmark_k = cfg_num("hybrid_landmark_k", ctx$hybrid_landmark_k),
    hybrid_landmark_time_sec = cfg_num("hybrid_landmark_time_sec", ctx$hybrid_landmark_time_sec),
    hybrid_landmark_density_time_sec = cfg_num("hybrid_landmark_density_time_sec", ctx$hybrid_landmark_density_time_sec),
    hybrid_landmark_feature_dims = cfg_num("hybrid_landmark_feature_dims", ctx$hybrid_landmark_feature_dims),
    hybrid_landmark_formula = cfg_chr("hybrid_landmark_formula", ctx$hybrid_landmark_formula),
    hybrid_landmark_density_weight_median = cfg_num("hybrid_landmark_density_weight_median", ctx$hybrid_landmark_density_weight_median),
    hybrid_landmark_density_selected_to_global_weight_ratio = cfg_num("hybrid_landmark_density_selected_to_global_weight_ratio", ctx$hybrid_landmark_density_selected_to_global_weight_ratio),
    hybrid_landmark_mean_distance_median = cfg_num("hybrid_landmark_mean_distance_median", ctx$hybrid_landmark_mean_distance_median),
    hybrid_landmark_cover_mean = cfg_num("hybrid_landmark_cover_mean", ctx$hybrid_landmark_cover_mean),
    hybrid_landmark_cover_median = cfg_num("hybrid_landmark_cover_median", ctx$hybrid_landmark_cover_median),
    hybrid_landmark_cover_max = cfg_num("hybrid_landmark_cover_max", ctx$hybrid_landmark_cover_max),
    rare_protected_tail_fraction = cfg_num("rare_protected_tail_fraction", ctx$rare_protected_tail_fraction),
    rare_protected_tail_oversample = cfg_num("rare_protected_tail_oversample", ctx$rare_protected_tail_oversample),
    rare_protected_n_quantiles = cfg_num("rare_protected_n_quantiles", ctx$rare_protected_n_quantiles),
    rare_protected_cluster_fraction = cfg_num("rare_protected_cluster_fraction", ctx$rare_protected_cluster_fraction),
    rare_protected_density_k = cfg_num("rare_protected_density_k", ctx$rare_protected_density_k),
    rare_protected_time_sec = cfg_num("rare_protected_time_sec", ctx$rare_protected_time_sec),
    rare_protected_density_time_sec = cfg_num("rare_protected_density_time_sec", ctx$rare_protected_density_time_sec),
    rare_protected_cluster_time_sec = cfg_num("rare_protected_cluster_time_sec", ctx$rare_protected_cluster_time_sec),
    rare_protected_tail_count = cfg_num("rare_protected_tail_count", ctx$rare_protected_tail_count),
    rare_protected_quantile_count = cfg_num("rare_protected_quantile_count", ctx$rare_protected_quantile_count),
    rare_protected_cluster_count = cfg_num("rare_protected_cluster_count", ctx$rare_protected_cluster_count),
    rare_protected_fill_count = cfg_num("rare_protected_fill_count", ctx$rare_protected_fill_count),
    rare_protected_tail_threshold = cfg_num("rare_protected_tail_threshold", ctx$rare_protected_tail_threshold),
    rare_protected_selected_tail_fraction = cfg_num("rare_protected_selected_tail_fraction", ctx$rare_protected_selected_tail_fraction),
    rare_protected_selected_mean_distance_ratio = cfg_num("rare_protected_selected_mean_distance_ratio", ctx$rare_protected_selected_mean_distance_ratio),
    rare_protected_selected_to_global_low_density_ratio = cfg_num("rare_protected_selected_to_global_low_density_ratio", ctx$rare_protected_selected_to_global_low_density_ratio),
    rare_protected_quantile_min_selected = cfg_num("rare_protected_quantile_min_selected", ctx$rare_protected_quantile_min_selected),
    rare_protected_quantile_max_selected = cfg_num("rare_protected_quantile_max_selected", ctx$rare_protected_quantile_max_selected),
    rare_protected_cluster_k = cfg_num("rare_protected_cluster_k", ctx$rare_protected_cluster_k),
    rare_protected_cluster_feature_dims = cfg_num("rare_protected_cluster_feature_dims", ctx$rare_protected_cluster_feature_dims),
    diversity_landmark_algorithm = cfg_chr("diversity_landmark_algorithm", ctx$diversity_landmark_algorithm),
    diversity_landmark_time_sec = cfg_num("diversity_landmark_time_sec", ctx$diversity_landmark_time_sec),
    diversity_landmark_feature_dims = cfg_num("diversity_landmark_feature_dims", ctx$diversity_landmark_feature_dims),
    diversity_landmark_cover_mean = cfg_num("diversity_landmark_cover_mean", ctx$diversity_landmark_cover_mean),
    diversity_landmark_cover_median = cfg_num("diversity_landmark_cover_median", ctx$diversity_landmark_cover_median),
    diversity_landmark_cover_max = cfg_num("diversity_landmark_cover_max", ctx$diversity_landmark_cover_max),
    diversity_landmark_leverage_selected_to_global_ratio = cfg_num("diversity_landmark_leverage_selected_to_global_ratio", ctx$diversity_landmark_leverage_selected_to_global_ratio),
    landmark_projection_k = cfg_num("landmark_projection_k", ctx$landmark_projection_k),
    landmark_interpolation = cfg_chr("landmark_interpolation", ctx$landmark_interpolation),
    landmark_projection_model = cfg_chr("landmark_projection_model", ctx$landmark_projection_model),
    landmark_projection_weight = cfg_chr("landmark_projection_weight", ctx$landmark_projection_weight),
    landmark_projection_bandwidth_rule = cfg_chr("landmark_projection_bandwidth_rule", ctx$landmark_projection_bandwidth_rule),
    landmark_projection_bandwidth_mean = cfg_num("landmark_projection_bandwidth_mean", ctx$landmark_projection_bandwidth_mean),
    landmark_projection_weight_entropy = cfg_num("landmark_projection_weight_entropy", ctx$landmark_projection_weight_entropy),
    landmark_projection_zero_neighbor_fraction = cfg_num("landmark_projection_zero_neighbor_fraction", ctx$landmark_projection_zero_neighbor_fraction),
    landmark_projection_time_sec = cfg_num("landmark_projection_time_sec", ctx$landmark_projection_time_sec),
    landmark_landmark_knn_time_sec = cfg_num("landmark_landmark_knn_time_sec", ctx$landmark_landmark_knn_time_sec),
    landmark_landmark_embedding_time_sec = cfg_num("landmark_landmark_embedding_time_sec", ctx$landmark_landmark_embedding_time_sec),
    landmark_affine_ridge = cfg_num("landmark_affine_ridge", ctx$landmark_affine_ridge),
    landmark_affine_weight = cfg_chr("landmark_affine_weight", ctx$landmark_affine_weight),
    landmark_affine_rank_mean = cfg_num("landmark_affine_rank_mean", ctx$landmark_affine_rank_mean),
    landmark_affine_condition_median = cfg_num("landmark_affine_condition_median", ctx$landmark_affine_condition_median),
    landmark_affine_condition_max = cfg_num("landmark_affine_condition_max", ctx$landmark_affine_condition_max),
    landmark_affine_fallback_fraction = cfg_num("landmark_affine_fallback_fraction", ctx$landmark_affine_fallback_fraction),
    landmark_affine_clipped_fraction = cfg_num("landmark_affine_clipped_fraction", ctx$landmark_affine_clipped_fraction),
    landmark_affine_clip_multiplier = cfg_num("landmark_affine_clip_multiplier", ctx$landmark_affine_clip_multiplier),
    landmark_affine_blend = cfg_num("landmark_affine_blend", ctx$landmark_affine_blend),
    landmark_refinement = cfg_chr("landmark_refinement", ctx$landmark_refinement),
    landmark_refinement_epochs = cfg_num("landmark_refinement_epochs", ctx$landmark_refinement_epochs),
    landmark_refinement_time_sec = cfg_num("landmark_refinement_time_sec", ctx$landmark_refinement_time_sec),
    landmark_projection_backend = cfg_chr("landmark_projection_backend", ctx$landmark_projection_backend),
    landmark_interpolation_backend = cfg_chr("landmark_interpolation_backend", ctx$landmark_interpolation_backend),
    landmark_interpolation_backend_reason = cfg_chr("landmark_interpolation_backend_reason", ctx$landmark_interpolation_backend_reason),
    landmark_refinement_backend = cfg_chr("landmark_refinement_backend", ctx$landmark_refinement_backend),
    landmark_refinement_knn_backend = cfg_chr("landmark_refinement_knn_backend", ctx$landmark_refinement_knn_backend),
    landmark_refinement_knn_backend_reason = cfg_chr("landmark_refinement_knn_backend_reason", ctx$landmark_refinement_knn_backend_reason),
    subsample_strategy = cfg_chr("subsample_strategy", ctx$subsample_strategy),
    subsample_stratified = cfg_log("subsample_stratified", ctx$subsample_stratified),
    benchmark_forced_k = cfg_num("benchmark_forced_k", ctx$benchmark_forced_k),
    benchmark_standardize = cfg_log("benchmark_standardize", ctx$benchmark_standardize),
    graph_adaptive_min_k = ctx$graph_adaptive_min_k,
    graph_adaptive_max_k = ctx$graph_adaptive_max_k,
    graph_adaptive_mean_k = ctx$graph_adaptive_mean_k,
    graph_adaptive_density_quantile = ctx$graph_adaptive_density_quantile,
    graph_adaptive_density_cor = ctx$graph_adaptive_density_cor,
    graph_adaptive_dense_fraction = ctx$graph_adaptive_dense_fraction,
    graph_adaptive_sparse_fraction = ctx$graph_adaptive_sparse_fraction,
    graph_distance_prune_drop_fraction = ctx$graph_distance_prune_drop_fraction,
    graph_distance_prune_percentile = ctx$graph_distance_prune_percentile,
    graph_distance_prune_removed_edges_mean = ctx$graph_distance_prune_removed_edges_mean,
    graph_distance_prune_threshold_mean = ctx$graph_distance_prune_threshold_mean,
    graph_distance_prune_threshold_min = ctx$graph_distance_prune_threshold_min,
    graph_distance_prune_threshold_max = ctx$graph_distance_prune_threshold_max,
    graph_distance_prune_removed_distance_mean = ctx$graph_distance_prune_removed_distance_mean,
    graph_sparsification_method = ctx$graph_sparsification_method,
    graph_sparsification_keep_fraction = ctx$graph_sparsification_keep_fraction,
    graph_sparsification_target_k = ctx$graph_sparsification_target_k,
    graph_sparsification_undirected_edges = ctx$graph_sparsification_undirected_edges,
    graph_sparsification_spectral_rank = ctx$graph_sparsification_spectral_rank,
    graph_sparsification_spectral_time_sec = ctx$graph_sparsification_spectral_time_sec,
    graph_sparsification_leverage_mean = ctx$graph_sparsification_leverage_mean,
    graph_sparsification_leverage_min = ctx$graph_sparsification_leverage_min,
    graph_sparsification_leverage_max = ctx$graph_sparsification_leverage_max,
    graph_sparsification_resistance_mean = ctx$graph_sparsification_resistance_mean,
    graph_sparsification_weight_mean = ctx$graph_sparsification_weight_mean,
    graph_mst_rescue_enabled = ctx$graph_mst_rescue_enabled,
    graph_mst_rescue_base_components = ctx$graph_mst_rescue_base_components,
    graph_mst_rescue_components_before = ctx$graph_mst_rescue_components_before,
    graph_mst_rescue_components_after = ctx$graph_mst_rescue_components_after,
    graph_mst_rescue_forest_edges = ctx$graph_mst_rescue_forest_edges,
    graph_mst_rescue_added_forest_edges = ctx$graph_mst_rescue_added_forest_edges,
    graph_mst_rescue_added_directed_edges = ctx$graph_mst_rescue_added_directed_edges,
    graph_mst_rescue_mean_degree_before = ctx$graph_mst_rescue_mean_degree_before,
    graph_mst_rescue_mean_degree_after = ctx$graph_mst_rescue_mean_degree_after,
    graph_density_correction_method = ctx$graph_density_correction_method,
    graph_density_correction_quantile = ctx$graph_density_correction_quantile,
    graph_density_correction_strength = ctx$graph_density_correction_strength,
    graph_density_scale_mean = ctx$graph_density_scale_mean,
    graph_density_scale_min = ctx$graph_density_scale_min,
    graph_density_scale_max = ctx$graph_density_scale_max,
    graph_density_scale_cv = ctx$graph_density_scale_cv,
    graph_density_sparse_fraction = ctx$graph_density_sparse_fraction,
    graph_density_correction_mean = ctx$graph_density_correction_mean,
    graph_density_correction_min = ctx$graph_density_correction_min,
    graph_density_correction_max = ctx$graph_density_correction_max,
    graph_density_correction_clamp_fraction = ctx$graph_density_correction_clamp_fraction,
    graph_density_corrected_distance_scale_cor = ctx$graph_density_corrected_distance_scale_cor,
    umap_graph_set_op_mix_ratio = ctx$umap_graph_set_op_mix_ratio,
    umap_graph_local_connectivity = ctx$umap_graph_local_connectivity,
    umap_graph_weight_power = ctx$umap_graph_weight_power,
    umap_graph_target_scale = ctx$umap_graph_target_scale,
    umap_graph_distance_transform = ctx$umap_graph_distance_transform,
    umap_graph_mean_weight = ctx$umap_graph_mean_weight,
    umap_graph_min_weight = ctx$umap_graph_min_weight,
    umap_graph_max_weight = ctx$umap_graph_max_weight,
    graph_edge_sampling_method = cfg_chr("graph_edge_sampling_method", ctx$graph_edge_sampling_method),
    graph_edge_sampling_fraction = cfg_num("graph_edge_sampling_fraction", ctx$graph_edge_sampling_fraction),
    graph_edge_sampling_weight_power = cfg_num("graph_edge_sampling_weight_power", ctx$graph_edge_sampling_weight_power),
    graph_edge_sampling_include_top = cfg_num("graph_edge_sampling_include_top", ctx$graph_edge_sampling_include_top),
    graph_edge_sampling_target_scale = cfg_num("graph_edge_sampling_target_scale", ctx$graph_edge_sampling_target_scale),
    graph_edge_sampling_mean_selected_weight = cfg_num("graph_edge_sampling_mean_selected_weight", ctx$graph_edge_sampling_mean_selected_weight),
    graph_edge_sampling_mean_candidate_weight = cfg_num("graph_edge_sampling_mean_candidate_weight", ctx$graph_edge_sampling_mean_candidate_weight),
    graph_edge_sampling_selected_to_candidate_weight_ratio = cfg_num("graph_edge_sampling_selected_to_candidate_weight_ratio", ctx$graph_edge_sampling_selected_to_candidate_weight_ratio),
    pair_resampling_mode = cfg_chr("pair_resampling_mode", ctx$pair_resampling_mode),
    pair_resampling_pair_family = cfg_chr("pair_resampling_pair_family", ctx$pair_resampling_pair_family),
    pair_resampling_transfer_mode = cfg_chr("pair_resampling_transfer_mode", ctx$pair_resampling_transfer_mode),
    pair_resampling_refreshes = cfg_num("pair_resampling_refreshes", ctx$pair_resampling_refreshes),
    pair_resampling_stage_count = cfg_num("pair_resampling_stage_count", ctx$pair_resampling_stage_count),
    pair_resampling_stage_epochs = cfg_chr("pair_resampling_stage_epochs", ctx$pair_resampling_stage_epochs),
    pair_resampling_warmup_epochs = cfg_num("pair_resampling_warmup_epochs", ctx$pair_resampling_warmup_epochs),
    pair_resampling_refine_epochs = cfg_num("pair_resampling_refine_epochs", ctx$pair_resampling_refine_epochs),
    pair_resampling_keep_fraction = cfg_num("pair_resampling_keep_fraction", ctx$pair_resampling_keep_fraction),
    pair_resampling_weight_power = cfg_num("pair_resampling_weight_power", ctx$pair_resampling_weight_power),
    pair_resampling_include_top = cfg_num("pair_resampling_include_top", ctx$pair_resampling_include_top),
    pair_resampling_seed_stride = cfg_num("pair_resampling_seed_stride", ctx$pair_resampling_seed_stride),
    pair_resampling_final_graph = cfg_chr("pair_resampling_final_graph", ctx$pair_resampling_final_graph),
    triplet_aux_enabled = cfg_log("triplet_aux_enabled", ctx$triplet_aux_enabled),
    triplet_aux_weight = cfg_num("triplet_aux_weight", ctx$triplet_aux_weight),
    triplet_aux_samples_per_edge = cfg_num("triplet_aux_samples_per_edge", ctx$triplet_aux_samples_per_edge),
    triplet_aux_transfer_mode = cfg_chr("triplet_aux_transfer_mode", ctx$triplet_aux_transfer_mode),
    triplet_aux_native_backend = cfg_chr("triplet_aux_native_backend", ctx$triplet_aux_native_backend),
	    triplet_aux_optimizer_backend = cfg_chr("triplet_aux_optimizer_backend", ctx$triplet_aux_optimizer_backend),
	    triplet_aux_n_epochs = cfg_num("triplet_aux_n_epochs", ctx$triplet_aux_n_epochs),
	    triplet_aux_negative_sample_rate = cfg_num("triplet_aux_negative_sample_rate", ctx$triplet_aux_negative_sample_rate),
	    triplet_structured_enabled = cfg_log("triplet_structured_enabled", ctx$triplet_structured_enabled),
	    triplet_structured_weight = cfg_num("triplet_structured_weight", ctx$triplet_structured_weight),
	    triplet_structured_n_inliers = cfg_num("triplet_structured_n_inliers", ctx$triplet_structured_n_inliers),
	    triplet_structured_n_outliers = cfg_num("triplet_structured_n_outliers", ctx$triplet_structured_n_outliers),
	    triplet_structured_n_random = cfg_num("triplet_structured_n_random", ctx$triplet_structured_n_random),
	    triplet_structured_total = cfg_num("triplet_structured_total", ctx$triplet_structured_total),
	    triplet_structured_mode = cfg_chr("triplet_structured_mode", ctx$triplet_structured_mode),
	    global_random_triplet_enabled = cfg_log("global_random_triplet_enabled", ctx$global_random_triplet_enabled),
	    global_random_triplet_weight = cfg_num("global_random_triplet_weight", ctx$global_random_triplet_weight),
	    global_random_triplets_per_point = cfg_num("global_random_triplets_per_point", ctx$global_random_triplets_per_point),
	    global_random_trimap_extra_negatives = cfg_num("global_random_trimap_extra_negatives", ctx$global_random_trimap_extra_negatives),
	    global_random_negative_source = cfg_chr("global_random_negative_source", ctx$global_random_negative_source),
	    global_random_transfer_mode = cfg_chr("global_random_transfer_mode", ctx$global_random_transfer_mode),
	    global_random_effective_negative_sample_rate = cfg_num("global_random_effective_negative_sample_rate", ctx$global_random_effective_negative_sample_rate),
		    hard_negative_enabled = cfg_log("hard_negative_enabled", ctx$hard_negative_enabled),
		    hard_negative_rate = cfg_num("hard_negative_rate", ctx$hard_negative_rate),
		    hard_negative_weight_multiplier = cfg_num("hard_negative_weight_multiplier", ctx$hard_negative_weight_multiplier),
		    hard_negative_candidate_source = cfg_chr("hard_negative_candidate_source", ctx$hard_negative_candidate_source),
		    hard_negative_transfer_mode = cfg_chr("hard_negative_transfer_mode", ctx$hard_negative_transfer_mode),
		    semihard_triplet_enabled = cfg_log("semihard_triplet_enabled", ctx$semihard_triplet_enabled),
		    semihard_triplet_rate = cfg_num("semihard_triplet_rate", ctx$semihard_triplet_rate),
		    semihard_triplet_weight_multiplier = cfg_num("semihard_triplet_weight_multiplier", ctx$semihard_triplet_weight_multiplier),
		    semihard_triplet_candidate_source = cfg_chr("semihard_triplet_candidate_source", ctx$semihard_triplet_candidate_source),
		    semihard_triplet_transfer_mode = cfg_chr("semihard_triplet_transfer_mode", ctx$semihard_triplet_transfer_mode),
		    triplet_mining_approximate = cfg_log("triplet_mining_approximate", ctx$triplet_mining_approximate),
		    triplet_mining_graph_source = cfg_chr("triplet_mining_graph_source", ctx$triplet_mining_graph_source),
		    triplet_mining_source_detail = cfg_chr("triplet_mining_source_detail", ctx$triplet_mining_source_detail),
		    triplet_mining_knn_backend = cfg_chr("triplet_mining_knn_backend", ctx$triplet_mining_knn_backend),
		    triplet_mining_candidate_source = cfg_chr("triplet_mining_candidate_source", ctx$triplet_mining_candidate_source),
		    triplet_mining_transfer_mode = cfg_chr("triplet_mining_transfer_mode", ctx$triplet_mining_transfer_mode),
		    triplet_mining_recall_at_k = cfg_num("triplet_mining_recall_at_k", ctx$triplet_mining_recall_at_k),
		    triplet_mining_rank_correlation = cfg_num("triplet_mining_rank_correlation", ctx$triplet_mining_rank_correlation),
		    triplet_mining_distance_error = cfg_num("triplet_mining_distance_error", ctx$triplet_mining_distance_error),
		    umap_negative_sample_rate = ctx$umap_negative_sample_rate,
    umap_transfer_mode = ctx$umap_transfer_mode,
    graph_tsne_affinity_mode = ctx$graph_tsne_affinity_mode,
    graph_tsne_affinity_perplexities = ctx$graph_tsne_affinity_perplexities,
    graph_tsne_affinity_num_scales = ctx$graph_tsne_affinity_num_scales,
    graph_tsne_affinity_temperature = ctx$graph_tsne_affinity_temperature,
    graph_tsne_affinity_entropy_mean = ctx$graph_tsne_affinity_entropy_mean,
    graph_tsne_affinity_effective_perplexity_mean = ctx$graph_tsne_affinity_effective_perplexity_mean,
    graph_tsne_affinity_sigma_mean = ctx$graph_tsne_affinity_sigma_mean,
    graph_tsne_affinity_sigma_min = ctx$graph_tsne_affinity_sigma_min,
    graph_tsne_affinity_sigma_max = ctx$graph_tsne_affinity_sigma_max,
    graph_tsne_affinity_prob_min = ctx$graph_tsne_affinity_prob_min,
    graph_tsne_affinity_prob_max = ctx$graph_tsne_affinity_prob_max,
    graph_multiscale_perplexities = ctx$graph_multiscale_perplexities,
    graph_multiscale_num_scales = ctx$graph_multiscale_num_scales,
    graph_multiscale_required_k = ctx$graph_multiscale_required_k,
    graph_multiscale_effective_k_values = ctx$graph_multiscale_effective_k_values,
    graph_multiscale_transfer_mode = ctx$graph_multiscale_transfer_mode,
    graph_multiscale_uses_tsne_affinity = ctx$graph_multiscale_uses_tsne_affinity,
    pacmap_transfer_mode = cfg_chr("pacmap_transfer_mode", ctx$pacmap_transfer_mode),
    pacmap_auxiliary_pair_family = cfg_chr("pacmap_auxiliary_pair_family", ctx$pacmap_auxiliary_pair_family),
    pacmap_mid_near_pairs_per_point = cfg_num("pacmap_mid_near_pairs_per_point", ctx$pacmap_mid_near_pairs_per_point),
    pacmap_mid_near_fraction = cfg_num("pacmap_mid_near_fraction", ctx$pacmap_mid_near_fraction),
    pacmap_mid_near_requested_fraction = cfg_num("pacmap_mid_near_requested_fraction", ctx$pacmap_mid_near_requested_fraction),
    pacmap_mid_near_distance_scale = cfg_num("pacmap_mid_near_distance_scale", ctx$pacmap_mid_near_distance_scale),
    pacmap_mid_near_fallback_fraction = cfg_num("pacmap_mid_near_fallback_fraction", ctx$pacmap_mid_near_fallback_fraction),
    pacmap_mid_near_rank_mean = cfg_num("pacmap_mid_near_rank_mean", ctx$pacmap_mid_near_rank_mean),
    pacmap_mid_near_emphasis_strength = cfg_num("pacmap_mid_near_emphasis_strength", ctx$pacmap_mid_near_emphasis_strength),
    pacmap_mid_near_emphasis_distance_multiplier = cfg_num("pacmap_mid_near_emphasis_distance_multiplier", ctx$pacmap_mid_near_emphasis_distance_multiplier),
    pacmap_near_ratio = cfg_num("pacmap_near_ratio", ctx$pacmap_near_ratio),
    pacmap_mid_ratio = cfg_num("pacmap_mid_ratio", ctx$pacmap_mid_ratio),
    pacmap_far_ratio = cfg_num("pacmap_far_ratio", ctx$pacmap_far_ratio),
    pacmap_near_pairs_per_point = cfg_num("pacmap_near_pairs_per_point", ctx$pacmap_near_pairs_per_point),
    pacmap_mid_pairs_per_point = cfg_num("pacmap_mid_pairs_per_point", ctx$pacmap_mid_pairs_per_point),
    pacmap_far_pairs_per_point = cfg_num("pacmap_far_pairs_per_point", ctx$pacmap_far_pairs_per_point),
    pacmap_far_pair_fraction = cfg_num("pacmap_far_pair_fraction", ctx$pacmap_far_pair_fraction),
    pacmap_far_distance_scale = cfg_num("pacmap_far_distance_scale", ctx$pacmap_far_distance_scale),
    pacmap_far_fallback_fraction = cfg_num("pacmap_far_fallback_fraction", ctx$pacmap_far_fallback_fraction),
    pacmap_far_repulsion_rate = cfg_num("pacmap_far_repulsion_rate", ctx$pacmap_far_repulsion_rate),
    pacmap_phase_schedule = cfg_chr("pacmap_phase_schedule", ctx$pacmap_phase_schedule),
    pacmap_phase_total_epochs = cfg_num("pacmap_phase_total_epochs", ctx$pacmap_phase_total_epochs),
    pacmap_phase_epoch_multiplier = cfg_num("pacmap_phase_epoch_multiplier", ctx$pacmap_phase_epoch_multiplier),
    pacmap_phase_warmup_epochs = cfg_num("pacmap_phase_warmup_epochs", ctx$pacmap_phase_warmup_epochs),
    pacmap_phase_refine_epochs = cfg_num("pacmap_phase_refine_epochs", ctx$pacmap_phase_refine_epochs),
    pacmap_phase_transfer_detail = cfg_chr("pacmap_phase_transfer_detail", ctx$pacmap_phase_transfer_detail),
    trimap_transfer_mode = cfg_chr("trimap_transfer_mode", ctx$trimap_transfer_mode),
    trimap_triplet_family = cfg_chr("trimap_triplet_family", ctx$trimap_triplet_family),
    trimap_inlier_ratio = cfg_num("trimap_inlier_ratio", ctx$trimap_inlier_ratio),
    trimap_semihard_ratio = cfg_num("trimap_semihard_ratio", ctx$trimap_semihard_ratio),
    trimap_global_anchor_ratio = cfg_num("trimap_global_anchor_ratio", ctx$trimap_global_anchor_ratio),
    trimap_inlier_pairs_per_point = cfg_num("trimap_inlier_pairs_per_point", ctx$trimap_inlier_pairs_per_point),
    trimap_semihard_pairs_per_point = cfg_num("trimap_semihard_pairs_per_point", ctx$trimap_semihard_pairs_per_point),
    trimap_global_anchor_pairs_per_point = cfg_num("trimap_global_anchor_pairs_per_point", ctx$trimap_global_anchor_pairs_per_point),
    trimap_semihard_fraction = cfg_num("trimap_semihard_fraction", ctx$trimap_semihard_fraction),
    trimap_global_anchor_fraction = cfg_num("trimap_global_anchor_fraction", ctx$trimap_global_anchor_fraction),
    trimap_semihard_distance_scale = cfg_num("trimap_semihard_distance_scale", ctx$trimap_semihard_distance_scale),
    trimap_global_anchor_distance_scale = cfg_num("trimap_global_anchor_distance_scale", ctx$trimap_global_anchor_distance_scale),
    trimap_semihard_fallback_fraction = cfg_num("trimap_semihard_fallback_fraction", ctx$trimap_semihard_fallback_fraction),
    trimap_global_anchor_fallback_fraction = cfg_num("trimap_global_anchor_fallback_fraction", ctx$trimap_global_anchor_fallback_fraction),
    trimap_semihard_rank_mean = cfg_num("trimap_semihard_rank_mean", ctx$trimap_semihard_rank_mean),
    trimap_candidate_seed = cfg_num("trimap_candidate_seed", ctx$trimap_candidate_seed),
    trimap_native_explicit_triplets = cfg_num("trimap_native_explicit_triplets", ctx$trimap_native_explicit_triplets),
    trimap_triplet_proxy_detail = cfg_chr("trimap_triplet_proxy_detail", ctx$trimap_triplet_proxy_detail),
    early_exaggeration_factor = ctx$early_exaggeration_factor,
    early_exaggeration_duration_fraction = ctx$early_exaggeration_duration_fraction,
    early_exaggeration_total_epochs = ctx$early_exaggeration_total_epochs,
    early_exaggeration_warmup_epochs = ctx$early_exaggeration_warmup_epochs,
    early_exaggeration_refine_epochs = ctx$early_exaggeration_refine_epochs,
    early_exaggeration_transfer_mode = ctx$early_exaggeration_transfer_mode,
    early_exaggeration_distance_scale = ctx$early_exaggeration_distance_scale,
    early_exaggeration_schedule_mode = ctx$early_exaggeration_schedule_mode,
    late_exaggeration_factor = ctx$late_exaggeration_factor,
    late_exaggeration_duration_fraction = ctx$late_exaggeration_duration_fraction,
    late_exaggeration_total_epochs = ctx$late_exaggeration_total_epochs,
    late_exaggeration_requested_epochs = ctx$late_exaggeration_requested_epochs,
    late_exaggeration_normal_epochs = ctx$late_exaggeration_normal_epochs,
    late_exaggeration_late_epochs = ctx$late_exaggeration_late_epochs,
    late_exaggeration_start_iter = ctx$late_exaggeration_start_iter,
    late_exaggeration_transfer_mode = ctx$late_exaggeration_transfer_mode,
    late_exaggeration_distance_scale = ctx$late_exaggeration_distance_scale,
    late_exaggeration_schedule_mode = ctx$late_exaggeration_schedule_mode,
    optimizer_mode = ctx$optimizer_mode,
    optimizer_schedule = ctx$optimizer_schedule,
    optimizer_momentum = ctx$optimizer_momentum,
    optimizer_final_momentum = ctx$optimizer_final_momentum,
    optimizer_switch_iter = ctx$optimizer_switch_iter,
    optimizer_learning_rate = ctx$optimizer_learning_rate,
    optimizer_learning_rate_multiplier = ctx$optimizer_learning_rate_multiplier,
    optimizer_adam_beta1 = ctx$optimizer_adam_beta1,
    optimizer_adam_beta2 = ctx$optimizer_adam_beta2,
    optimizer_adam_epsilon = ctx$optimizer_adam_epsilon,
    optimizer_transfer_mode = ctx$optimizer_transfer_mode,
    learning_rate_rule = ctx$learning_rate_rule,
    learning_rate_value = ctx$learning_rate_value,
    learning_rate_base_default = ctx$learning_rate_base_default,
    learning_rate_scale = ctx$learning_rate_scale,
    learning_rate_transfer_mode = ctx$learning_rate_transfer_mode,
    adaptive_lr_enabled = cfg_log("adaptive_lr_enabled", ctx$adaptive_lr_enabled),
    adaptive_lr_schedule = cfg_chr("adaptive_lr_schedule", ctx$adaptive_lr_schedule),
    adaptive_lr_total_epochs = cfg_num("adaptive_lr_total_epochs", ctx$adaptive_lr_total_epochs),
    adaptive_lr_chunk_epochs = cfg_num("adaptive_lr_chunk_epochs", ctx$adaptive_lr_chunk_epochs),
    adaptive_lr_chunks_run = cfg_num("adaptive_lr_chunks_run", ctx$adaptive_lr_chunks_run),
    adaptive_lr_base_learning_rate = cfg_num("adaptive_lr_base_learning_rate", ctx$adaptive_lr_base_learning_rate),
    adaptive_lr_final_learning_rate = cfg_num("adaptive_lr_final_learning_rate", ctx$adaptive_lr_final_learning_rate),
    adaptive_lr_final_multiplier = cfg_num("adaptive_lr_final_multiplier", ctx$adaptive_lr_final_multiplier),
    adaptive_lr_optimizer = cfg_chr("adaptive_lr_optimizer", ctx$adaptive_lr_optimizer),
    adaptive_lr_native = cfg_log("adaptive_lr_native", ctx$adaptive_lr_native),
    adaptive_lr_chunked = cfg_log("adaptive_lr_chunked", ctx$adaptive_lr_chunked),
    adaptive_lr_backend = cfg_chr("adaptive_lr_backend", ctx$adaptive_lr_backend),
    adaptive_lr_inner_decay = cfg_chr("adaptive_lr_inner_decay", ctx$adaptive_lr_inner_decay),
    adaptive_lr_init_strategy = cfg_chr("adaptive_lr_init_strategy", ctx$adaptive_lr_init_strategy),
    adaptive_lr_init_time_sec = cfg_num("adaptive_lr_init_time_sec", ctx$adaptive_lr_init_time_sec),
    adaptive_lr_optimizer_time_sec = cfg_num("adaptive_lr_optimizer_time_sec", ctx$adaptive_lr_optimizer_time_sec),
    adaptive_lr_trace_epochs = cfg_chr("adaptive_lr_trace_epochs", ctx$adaptive_lr_trace_epochs),
    adaptive_lr_trace_learning_rate = cfg_chr("adaptive_lr_trace_learning_rate", ctx$adaptive_lr_trace_learning_rate),
    adaptive_lr_trace_multiplier = cfg_chr("adaptive_lr_trace_multiplier", ctx$adaptive_lr_trace_multiplier),
    adaptive_lr_note = cfg_chr("adaptive_lr_note", ctx$adaptive_lr_note),
    mini_batch_enabled = cfg_log("mini_batch_enabled", ctx$mini_batch_enabled),
    mini_batch_backend = cfg_chr("mini_batch_backend", ctx$mini_batch_backend),
    mini_batch_mode = cfg_chr("mini_batch_mode", ctx$mini_batch_mode),
    mini_batch_batch_fraction = cfg_num("mini_batch_batch_fraction", ctx$mini_batch_batch_fraction),
    mini_batch_effective_k = cfg_num("mini_batch_effective_k", ctx$mini_batch_effective_k),
    mini_batch_chunks = cfg_num("mini_batch_chunks", ctx$mini_batch_chunks),
    mini_batch_chunk_epochs = cfg_chr("mini_batch_chunk_epochs", ctx$mini_batch_chunk_epochs),
    mini_batch_total_epochs = cfg_num("mini_batch_total_epochs", ctx$mini_batch_total_epochs),
    mini_batch_refreshes = cfg_num("mini_batch_refreshes", ctx$mini_batch_refreshes),
    mini_batch_sampling = cfg_chr("mini_batch_sampling", ctx$mini_batch_sampling),
    mini_batch_weight_power = cfg_num("mini_batch_weight_power", ctx$mini_batch_weight_power),
    mini_batch_include_top = cfg_num("mini_batch_include_top", ctx$mini_batch_include_top),
    mini_batch_init_strategy = cfg_chr("mini_batch_init_strategy", ctx$mini_batch_init_strategy),
    mini_batch_init_time_sec = cfg_num("mini_batch_init_time_sec", ctx$mini_batch_init_time_sec),
    mini_batch_graph_time_sec = cfg_num("mini_batch_graph_time_sec", ctx$mini_batch_graph_time_sec),
    mini_batch_optimizer_time_sec = cfg_num("mini_batch_optimizer_time_sec", ctx$mini_batch_optimizer_time_sec),
    mini_batch_trace_effective_k = cfg_chr("mini_batch_trace_effective_k", ctx$mini_batch_trace_effective_k),
    mini_batch_trace_retention = cfg_chr("mini_batch_trace_retention", ctx$mini_batch_trace_retention),
    mini_batch_experimental = cfg_log("mini_batch_experimental", ctx$mini_batch_experimental),
    mini_batch_note = cfg_chr("mini_batch_note", ctx$mini_batch_note),
    deterministic_batch_row_batch_size = cfg_num("deterministic_batch_row_batch_size", ctx$deterministic_batch_row_batch_size),
    deterministic_batch_chunks_per_epoch = cfg_num("deterministic_batch_chunks_per_epoch", ctx$deterministic_batch_chunks_per_epoch),
    deterministic_batch_reduction = cfg_chr("deterministic_batch_reduction", ctx$deterministic_batch_reduction),
    deterministic_batch_atomic_updates = cfg_log("deterministic_batch_atomic_updates", ctx$deterministic_batch_atomic_updates),
    deterministic_batch_reproducible_given_threads = cfg_log("deterministic_batch_reproducible_given_threads", ctx$deterministic_batch_reproducible_given_threads),
    sparse_edge_batch_enabled = cfg_log("sparse_edge_batch_enabled", ctx$sparse_edge_batch_enabled),
    sparse_edge_batch_backend = cfg_chr("sparse_edge_batch_backend", ctx$sparse_edge_batch_backend),
    sparse_edge_batch_mode = cfg_chr("sparse_edge_batch_mode", ctx$sparse_edge_batch_mode),
    sparse_edge_batch_storage = cfg_chr("sparse_edge_batch_storage", ctx$sparse_edge_batch_storage),
    sparse_edge_batch_edge_batch_size = cfg_num("sparse_edge_batch_edge_batch_size", ctx$sparse_edge_batch_edge_batch_size),
    sparse_edge_batch_chunks_per_epoch = cfg_num("sparse_edge_batch_chunks_per_epoch", ctx$sparse_edge_batch_chunks_per_epoch),
    sparse_edge_batch_n_epochs = cfg_num("sparse_edge_batch_n_epochs", ctx$sparse_edge_batch_n_epochs),
    sparse_edge_batch_negative_sample_rate = cfg_num("sparse_edge_batch_negative_sample_rate", ctx$sparse_edge_batch_negative_sample_rate),
    sparse_edge_batch_learning_rate = cfg_num("sparse_edge_batch_learning_rate", ctx$sparse_edge_batch_learning_rate),
    sparse_edge_batch_threads = cfg_num("sparse_edge_batch_threads", ctx$sparse_edge_batch_threads),
    sparse_edge_batch_atomic_updates = cfg_log("sparse_edge_batch_atomic_updates", ctx$sparse_edge_batch_atomic_updates),
    sparse_edge_batch_edge_list_copy = cfg_log("sparse_edge_batch_edge_list_copy", ctx$sparse_edge_batch_edge_list_copy),
    sparse_edge_batch_triplet_chunks = cfg_log("sparse_edge_batch_triplet_chunks", ctx$sparse_edge_batch_triplet_chunks),
    sparse_edge_batch_affinity_chunks = cfg_log("sparse_edge_batch_affinity_chunks", ctx$sparse_edge_batch_affinity_chunks),
    sparse_edge_batch_aux_memory_mb = cfg_num("sparse_edge_batch_aux_memory_mb", ctx$sparse_edge_batch_aux_memory_mb),
    sparse_edge_batch_graph_time_sec = cfg_num("sparse_edge_batch_graph_time_sec", ctx$sparse_edge_batch_graph_time_sec),
    sparse_edge_batch_init_time_sec = cfg_num("sparse_edge_batch_init_time_sec", ctx$sparse_edge_batch_init_time_sec),
    sparse_edge_batch_optimizer_time_sec = cfg_num("sparse_edge_batch_optimizer_time_sec", ctx$sparse_edge_batch_optimizer_time_sec),
    sparse_edge_batch_status = cfg_chr("sparse_edge_batch_status", ctx$sparse_edge_batch_status),
    sparse_edge_batch_note = cfg_chr("sparse_edge_batch_note", ctx$sparse_edge_batch_note),
    vectorized_edge_enabled = cfg_log("vectorized_edge_enabled", ctx$vectorized_edge_enabled),
    vectorized_edge_backend = cfg_chr("vectorized_edge_backend", ctx$vectorized_edge_backend),
    vectorized_edge_storage = cfg_chr("vectorized_edge_storage", ctx$vectorized_edge_storage),
    vectorized_edge_batch_size = cfg_num("vectorized_edge_batch_size", ctx$vectorized_edge_batch_size),
    vectorized_edge_n_edges = cfg_num("vectorized_edge_n_edges", ctx$vectorized_edge_n_edges),
    vectorized_edge_n_epochs = cfg_num("vectorized_edge_n_epochs", ctx$vectorized_edge_n_epochs),
    vectorized_edge_negative_sample_rate = cfg_num("vectorized_edge_negative_sample_rate", ctx$vectorized_edge_negative_sample_rate),
    vectorized_edge_threads = cfg_num("vectorized_edge_threads", ctx$vectorized_edge_threads),
    vectorized_edge_learning_rate = cfg_num("vectorized_edge_learning_rate", ctx$vectorized_edge_learning_rate),
    vectorized_edge_simd = cfg_chr("vectorized_edge_simd", ctx$vectorized_edge_simd),
    vectorized_edge_gpu_native = cfg_log("vectorized_edge_gpu_native", ctx$vectorized_edge_gpu_native),
    vectorized_edge_graph_time_sec = cfg_num("vectorized_edge_graph_time_sec", ctx$vectorized_edge_graph_time_sec),
    vectorized_edge_init_time_sec = cfg_num("vectorized_edge_init_time_sec", ctx$vectorized_edge_init_time_sec),
    vectorized_edge_optimizer_time_sec = cfg_num("vectorized_edge_optimizer_time_sec", ctx$vectorized_edge_optimizer_time_sec),
    vectorized_edge_status = cfg_chr("vectorized_edge_status", ctx$vectorized_edge_status),
    vectorized_edge_note = cfg_chr("vectorized_edge_note", ctx$vectorized_edge_note),
    atomic_sgd_enabled = cfg_log("atomic_sgd_enabled", ctx$atomic_sgd_enabled),
    atomic_sgd_backend = cfg_chr("atomic_sgd_backend", ctx$atomic_sgd_backend),
    atomic_sgd_update_mode = cfg_chr("atomic_sgd_update_mode", ctx$atomic_sgd_update_mode),
    atomic_sgd_storage = cfg_chr("atomic_sgd_storage", ctx$atomic_sgd_storage),
    atomic_sgd_n_edges = cfg_num("atomic_sgd_n_edges", ctx$atomic_sgd_n_edges),
    atomic_sgd_n_epochs = cfg_num("atomic_sgd_n_epochs", ctx$atomic_sgd_n_epochs),
    atomic_sgd_negative_sample_rate = cfg_num("atomic_sgd_negative_sample_rate", ctx$atomic_sgd_negative_sample_rate),
    atomic_sgd_threads = cfg_num("atomic_sgd_threads", ctx$atomic_sgd_threads),
    atomic_sgd_learning_rate = cfg_num("atomic_sgd_learning_rate", ctx$atomic_sgd_learning_rate),
    atomic_sgd_learning_rate_scale = cfg_num("atomic_sgd_learning_rate_scale", ctx$atomic_sgd_learning_rate_scale),
    atomic_sgd_coordinate_clip = cfg_num("atomic_sgd_coordinate_clip", ctx$atomic_sgd_coordinate_clip),
    atomic_sgd_openmp = cfg_log("atomic_sgd_openmp", ctx$atomic_sgd_openmp),
    atomic_sgd_gpu_native = cfg_log("atomic_sgd_gpu_native", ctx$atomic_sgd_gpu_native),
    atomic_sgd_nondeterministic = cfg_log("atomic_sgd_nondeterministic", ctx$atomic_sgd_nondeterministic),
    atomic_sgd_graph_time_sec = cfg_num("atomic_sgd_graph_time_sec", ctx$atomic_sgd_graph_time_sec),
    atomic_sgd_init_time_sec = cfg_num("atomic_sgd_init_time_sec", ctx$atomic_sgd_init_time_sec),
    atomic_sgd_optimizer_time_sec = cfg_num("atomic_sgd_optimizer_time_sec", ctx$atomic_sgd_optimizer_time_sec),
    atomic_sgd_status = cfg_chr("atomic_sgd_status", ctx$atomic_sgd_status),
    atomic_sgd_note = cfg_chr("atomic_sgd_note", ctx$atomic_sgd_note),
    tsne_bh_theta = ctx$tsne_bh_theta,
    tsne_bh_perplexity = ctx$tsne_bh_perplexity,
    tsne_bh_n_epochs = ctx$tsne_bh_n_epochs,
    tsne_bh_learning_rate = ctx$tsne_bh_learning_rate,
    tsne_bh_n_threads = ctx$tsne_bh_n_threads,
    tsne_bh_stop_lying_iter = ctx$tsne_bh_stop_lying_iter,
    tsne_bh_mom_switch_iter = ctx$tsne_bh_mom_switch_iter,
    fft_interpolation_mode = ctx$fft_interpolation_mode,
    fft_interpolation_experimental = ctx$fft_interpolation_experimental,
    fft_interpolation_backend = ctx$fft_interpolation_backend,
    fft_interpolation_transfer_scope = ctx$fft_interpolation_transfer_scope,
    fft_interpolation_native_repulsive_field = ctx$fft_interpolation_native_repulsive_field,
    fft_interpolation_refine_epochs = ctx$fft_interpolation_refine_epochs,
    fft_grid_nterms = ctx$fft_grid_nterms,
    fft_grid_intervals_per_integer = ctx$fft_grid_intervals_per_integer,
    fft_grid_min_num_intervals = ctx$fft_grid_min_num_intervals,
    output_metric = cfg_chr("output_metric", ctx$output_metric),
    output_metric_transform = cfg_chr("output_metric_transform", ctx$output_metric_transform),
    output_metric_native = cfg_log("output_metric_native", ctx$output_metric_native),
    output_metric_projection_scale = cfg_num("output_metric_projection_scale", ctx$output_metric_projection_scale),
    output_metric_curvature = cfg_num("output_metric_curvature", ctx$output_metric_curvature),
    output_metric_radius_mean = cfg_num("output_metric_radius_mean", ctx$output_metric_radius_mean),
    output_metric_radius_max = cfg_num("output_metric_radius_max", ctx$output_metric_radius_max),
    output_metric_norm_mean = cfg_num("output_metric_norm_mean", ctx$output_metric_norm_mean),
    embedding_time_sec = if (inherits(measured$value, "fastEmbedR_embedding")) measured$value$timings["embedding", "elapsed"] else measured$total_time_sec,
    total_time_sec = measured$total_time_sec,
    total_with_knn_time_sec = measured$total_time_sec +
      (if (is.finite(ctx$knn_time_sec)) ctx$knn_time_sec else 0) +
      (if (is.finite(ctx$graph_approximation_time_sec)) ctx$graph_approximation_time_sec else 0),
    peak_ram_mb = measured$peak_ram_mb,
    peak_gpu_gb = if (is.infinite(measured$peak_gpu_gb)) NA_real_ else measured$peak_gpu_gb,
    trustworthiness = metrics$trustworthiness,
    continuity = metrics$continuity,
    knn_preservation_15 = safe_metric(metrics, "knn_preservation_15"),
    knn_preservation_30 = safe_metric(metrics, "knn_preservation_30"),
    knn_preservation_50 = safe_metric(metrics, "knn_preservation_50"),
    distance_spearman = metrics$distance_spearman,
    distance_pearson = metrics$distance_pearson,
    stress = metrics$stress,
    output_metric_distance_spearman = safe_metric(metrics, "output_metric_distance_spearman"),
    output_metric_distance_pearson = safe_metric(metrics, "output_metric_distance_pearson"),
    output_metric_stress = safe_metric(metrics, "output_metric_stress"),
    output_metric_global_sample_size = safe_metric(metrics, "output_metric_global_sample_size"),
    output_metric_global_pair_count = safe_metric(metrics, "output_metric_global_pair_count"),
    density_spearman = safe_metric(metrics, "density_spearman"),
    density_pearson = safe_metric(metrics, "density_pearson"),
    density_log_radius_rmse = safe_metric(metrics, "density_log_radius_rmse"),
    density_radius_high_mean = safe_metric(metrics, "density_radius_high_mean"),
    density_radius_embedding_mean = safe_metric(metrics, "density_radius_embedding_mean"),
    density_sample_size = safe_metric(metrics, "density_sample_size"),
    silhouette = metrics$silhouette,
    label_knn_accuracy = metrics$label_knn_accuracy,
    rare_class_recall = metrics$rare_class_recall,
    ari = metrics$ari,
    nmi = metrics$nmi,
    procrustes_rmsd = NA_real_,
    neighbour_stability = NA_real_,
    cluster_stability_ari = NA_real_,
    cluster_stability_nmi = NA_real_,
    layout_path = layout_path,
    stringsAsFactors = FALSE
  )
}

adjusted_rand_index <- function(x, y) {
  x <- as.factor(x)
  y <- as.factor(y)
  tab <- table(x, y)
  choose2 <- function(z) z * (z - 1) / 2
  sum_comb <- sum(choose2(tab))
  row_comb <- sum(choose2(rowSums(tab)))
  col_comb <- sum(choose2(colSums(tab)))
  total_comb <- choose2(sum(tab))
  if (total_comb == 0) return(NA_real_)
  expected <- row_comb * col_comb / total_comb
  denom <- 0.5 * (row_comb + col_comb) - expected
  if (denom == 0) return(NA_real_)
  (sum_comb - expected) / denom
}

normalized_mutual_info <- function(x, y) {
  x <- as.factor(x)
  y <- as.factor(y)
  tab <- table(x, y)
  n <- sum(tab)
  if (n == 0) return(NA_real_)
  pij <- tab / n
  pi <- rowSums(pij)
  pj <- colSums(pij)
  nz <- pij > 0
  mi <- sum(pij[nz] * log(pij[nz] / outer(pi, pj)[nz]))
  hx <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  hy <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  if (hx == 0 || hy == 0) return(NA_real_)
  mi / sqrt(hx * hy)
}

procrustes_rmsd <- function(a, b) {
  a <- as.matrix(a)
  b <- as.matrix(b)
  a <- sweep(a, 2L, colMeans(a), "-")
  b <- sweep(b, 2L, colMeans(b), "-")
  a_scale <- sqrt(sum(a * a))
  b_scale <- sqrt(sum(b * b))
  if (a_scale == 0 || b_scale == 0) return(NA_real_)
  a <- a / a_scale
  b <- b / b_scale
  sv <- svd(t(b) %*% a)
  rot <- sv$u %*% t(sv$v)
  sqrt(mean((a - b %*% rot)^2))
}

neighbor_overlap <- function(a, b, k = 15L) {
  k <- min(k, nrow(a) - 1L)
  if (k < 1L) return(NA_real_)
  nn_a <- fastEmbedR::nn(a, a, k + 1L, backend = "cpu")$indices[, -1L, drop = FALSE]
  nn_b <- fastEmbedR::nn(b, b, k + 1L, backend = "cpu")$indices[, -1L, drop = FALSE]
  mean(vapply(seq_len(nrow(a)), function(i) {
    length(intersect(nn_a[i, seq_len(k)], nn_b[i, seq_len(k)])) / k
  }, numeric(1)))
}

cluster_labels <- function(layout, labels, seed) {
  if (is.null(labels)) return(NULL)
  labels <- as.factor(labels)
  centers <- length(levels(labels))
  if (centers < 2L || centers >= nrow(layout)) return(NULL)
  set.seed(seed)
  tryCatch(stats::kmeans(layout, centers = centers, nstart = 5L, iter.max = 50L)$cluster, error = function(e) NULL)
}

add_stability_metrics <- function(results, dataset_labels) {
  ok <- which(results$status == "success" & file.exists(results$layout_path))
  if (length(ok) == 0L) return(results)
  groups <- split(ok, paste(results$dataset[ok], results$method[ok], results$approximation[ok], results$knn_reuse_mode[ok], results$backend[ok], results$parameter_settings[ok], sep = "\r"))
  for (idx in groups) {
    if (length(unique(results$seed[idx])) < 2L) next
    layouts <- lapply(results$layout_path[idx], readRDS)
    labels <- dataset_labels[[results$dataset[idx[1L]]]]
    rmsd <- overlap <- ari <- nmi <- numeric(0)
    clusters <- lapply(seq_along(layouts), function(i) cluster_labels(layouts[[i]], labels, results$seed[idx[i]]))
    for (i in seq_len(length(layouts) - 1L)) {
      for (j in (i + 1L):length(layouts)) {
        rmsd <- c(rmsd, procrustes_rmsd(layouts[[i]], layouts[[j]]))
        overlap <- c(overlap, neighbor_overlap(layouts[[i]], layouts[[j]], 15L))
        if (!is.null(clusters[[i]]) && !is.null(clusters[[j]])) {
          ari <- c(ari, adjusted_rand_index(clusters[[i]], clusters[[j]]))
          nmi <- c(nmi, normalized_mutual_info(clusters[[i]], clusters[[j]]))
        }
      }
    }
    results$procrustes_rmsd[idx] <- mean(rmsd, na.rm = TRUE)
    results$neighbour_stability[idx] <- mean(overlap, na.rm = TRUE)
    results$cluster_stability_ari[idx] <- mean(ari, na.rm = TRUE)
    results$cluster_stability_nmi[idx] <- mean(nmi, na.rm = TRUE)
  }
  results
}

layout_file <- function(layout_dir, ctx, strategy) {
  safe <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
  file.path(
    layout_dir,
    paste(
      safe(ctx$dataset_name),
      safe(ctx$method),
      safe(strategy$id),
      safe(ctx$knn_reuse_mode),
      safe(ctx$backend),
      paste0("k", ctx$k),
      paste0("seed", ctx$seed),
      sep = "_"
    )
  )
}

run_one <- function(dataset, method, strategy, backend, seed, k, out_dir, options, knn_reuse_mode = "method_specific", knn_cache = NULL) {
  n <- nrow(dataset$x)
  k <- min(as.integer(k), n - 1L)
  ctx <- list(
    dataset_name = dataset$name,
    x = dataset$x,
    labels = dataset$labels,
    method = method,
    backend = backend,
    seed = as.integer(seed),
    k = k,
    n = n,
    p = ncol(dataset$x),
    pca_dims = as.integer(options$pca_dims),
    knn_metric = normalize_knn_metric(options$knn_metric),
    short_epochs = as.integer(options$short_epochs),
    landmarks = if (options$landmarks <= 0) TRUE else min(as.integer(options$landmarks), n - 1L),
    knn = NULL,
    knn_reuse_mode = knn_reuse_mode,
    knn_cache_hit = FALSE,
    knn_graph_key = NA_character_,
    knn_graph_source = NA_character_,
    knn_disk_cache_hit = FALSE,
    knn_disk_cache_format = NA_character_,
    knn_disk_cache_path = NA_character_,
    knn_disk_load_time_sec = NA_real_,
    knn_disk_save_time_sec = NA_real_,
    knn_time_sec = NA_real_,
    knn_graph_time_sec = NA_real_,
      knn_index_build_time_sec = NA_real_,
      knn_query_time_sec = NA_real_,
      knn_graph_index_build_time_sec = NA_real_,
      knn_graph_query_time_sec = NA_real_,
      knn_memory_mb = NA_real_,
      knn_recall_at_k = NA_real_,
      knn_mean_distance_error = NA_real_,
      knn_rank_correlation = NA_real_,
      knn_quality_sample_size = NA_integer_,
      graph_approximation = "none",
      graph_approximation_time_sec = 0,
      graph_csr = NULL,
      graph_storage_format = NA_character_,
      graph_sparse_nnz = NA_real_,
      graph_sparse_internal_memory_mb = NA_real_,
      graph_sparse_r_memory_mb = NA_real_,
      graph_dense_knn_memory_mb = NA_real_,
      graph_sparse_internal_memory_ratio = NA_real_,
      graph_sparse_r_memory_ratio = NA_real_,
      graph_sparse_prune_weight = NA_real_,
      graph_sparse_mean_weight = NA_real_,
      graph_sparse_min_weight = NA_real_,
      graph_sparse_max_weight = NA_real_,
      graph_effective_k = NA_integer_,
      graph_edge_retention = NA_real_,
      graph_recall_at_k = NA_real_,
      graph_mean_distance_error = NA_real_,
      graph_rank_correlation = NA_real_,
      graph_quality_sample_size = NA_integer_,
      graph_mean_degree = NA_real_,
      graph_min_degree = NA_real_,
      graph_max_degree = NA_real_,
      graph_isolated_fraction = NA_real_,
      graph_padding_fraction = NA_real_,
      graph_mean_jaccard = NA_real_,
      graph_min_jaccard = NA_real_,
      graph_max_jaccard = NA_real_,
      graph_zero_jaccard_fraction = NA_real_,
      graph_snn_k = NA_real_,
      graph_snn_prune_threshold = NA_real_,
      graph_mean_snn_weight = NA_real_,
      graph_min_snn_weight = NA_real_,
      graph_max_snn_weight = NA_real_,
      graph_zero_snn_fraction = NA_real_,
      localmap_false_neighbor_enabled = NA,
      localmap_false_neighbor_mode = NA_character_,
      localmap_false_neighbor_transfer_mode = NA_character_,
      localmap_false_neighbor_jaccard_threshold = NA_real_,
      localmap_false_neighbor_distance_quantile = NA_real_,
      localmap_false_neighbor_distance_multiplier = NA_real_,
      localmap_false_neighbor_min_keep_fraction = NA_real_,
      localmap_false_neighbor_min_keep_k = NA_real_,
      localmap_false_neighbor_removed_edges_mean = NA_real_,
      localmap_false_neighbor_removed_fraction = NA_real_,
      localmap_false_neighbor_kept_degree_mean = NA_real_,
      localmap_false_neighbor_kept_jaccard_mean = NA_real_,
      localmap_false_neighbor_removed_jaccard_mean = NA_real_,
      localmap_false_neighbor_kept_distance_ratio_mean = NA_real_,
      localmap_false_neighbor_removed_distance_ratio_mean = NA_real_,
      localmap_false_neighbor_threshold_mean = NA_real_,
      localmap_local_weight_enabled = NA,
      localmap_local_weight = NA_real_,
      localmap_local_weight_mode = NA_character_,
      localmap_local_weight_transfer_mode = NA_character_,
      localmap_local_weight_jaccard_blend = NA_real_,
      localmap_local_weight_mean_trust = NA_real_,
      localmap_local_weight_rank_component_mean = NA_real_,
      localmap_local_weight_jaccard_component_mean = NA_real_,
      localmap_local_weight_mean_multiplier = NA_real_,
      localmap_local_weight_min_multiplier = NA_real_,
      localmap_local_weight_max_multiplier = NA_real_,
      localmap_local_weight_distance_scale_mean = NA_real_,
      artificial_neighbor_penalty_enabled = NA,
      artificial_neighbor_transfer_mode = NA_character_,
      artificial_neighbor_refinement_backend = NA_character_,
      artificial_neighbor_penalty_strength = NA_real_,
      artificial_neighbor_penalty_iterations = NA_real_,
      artificial_neighbor_penalty_low_k = NA_real_,
      artificial_neighbor_penalty_far_multiplier = NA_real_,
      artificial_neighbor_penalty_target_distance = NA_real_,
      artificial_neighbor_penalized_pairs = NA_real_,
      artificial_neighbor_total_low_edges = NA_real_,
      artificial_neighbor_false_rate_before = NA_real_,
      artificial_neighbor_false_rate_after = NA_real_,
      artificial_neighbor_false_rate_delta = NA_real_,
      artificial_neighbor_far_rate_before = NA_real_,
      artificial_neighbor_far_rate_after = NA_real_,
      artificial_neighbor_far_rate_delta = NA_real_,
      artificial_neighbor_mean_high_distance_ratio = NA_real_,
      artificial_neighbor_mean_low_distance = NA_real_,
      false_neighbor_monitor_enabled = NA,
      false_neighbor_monitor_transfer_mode = NA_character_,
      false_neighbor_monitor_backend = NA_character_,
      false_neighbor_monitor_action = NA_character_,
      false_neighbor_monitor_start_mode = NA_character_,
      false_neighbor_monitor_chunk_epochs = NA_real_,
      false_neighbor_monitor_max_chunks = NA_real_,
      false_neighbor_monitor_chunks_run = NA_real_,
      false_neighbor_monitor_epochs_requested = NA_real_,
      false_neighbor_monitor_epochs_completed = NA_real_,
      false_neighbor_monitor_patience = NA_real_,
      false_neighbor_monitor_tolerance = NA_real_,
      false_neighbor_monitor_low_k = NA_real_,
      false_neighbor_monitor_far_multiplier = NA_real_,
      false_neighbor_monitor_far_weight = NA_real_,
      false_neighbor_monitor_initial_false_rate = NA_real_,
      false_neighbor_monitor_final_false_rate = NA_real_,
      false_neighbor_monitor_best_false_rate = NA_real_,
      false_neighbor_monitor_false_rate_delta = NA_real_,
      false_neighbor_monitor_initial_far_rate = NA_real_,
      false_neighbor_monitor_final_far_rate = NA_real_,
      false_neighbor_monitor_best_far_rate = NA_real_,
      false_neighbor_monitor_far_rate_delta = NA_real_,
      false_neighbor_monitor_score_initial = NA_real_,
      false_neighbor_monitor_score_final = NA_real_,
      false_neighbor_monitor_score_best = NA_real_,
      false_neighbor_monitor_score_delta = NA_real_,
      false_neighbor_monitor_worsening_events = NA_real_,
      false_neighbor_monitor_adjustments = NA_real_,
      false_neighbor_monitor_stopped_early = NA,
      false_neighbor_monitor_score_trace = NA_character_,
      false_neighbor_monitor_false_rate_trace = NA_character_,
      false_neighbor_monitor_far_rate_trace = NA_character_,
      false_neighbor_monitor_chunk_trace = NA_character_,
      early_stop_enabled = NA,
      early_stop_criterion = NA_character_,
      early_stop_status = NA_character_,
      early_stop_reason = NA_character_,
      early_stop_max_epochs = NA_real_,
      early_stop_chunk_epochs = NA_real_,
      early_stop_epochs_run = NA_real_,
      early_stop_chunks_run = NA_real_,
      early_stop_patience = NA_real_,
      early_stop_tolerance = NA_real_,
      early_stop_displacement_final = NA_real_,
      early_stop_trustworthiness_final = NA_real_,
      early_stop_trustworthiness_delta_final = NA_real_,
      early_stop_neighbour_stability_final = NA_real_,
      early_stop_neighbour_stability_delta_final = NA_real_,
      early_stop_monitor_sample_size = NA_real_,
      early_stop_monitor_k = NA_real_,
      early_stop_init_strategy = NA_character_,
      early_stop_init_time_sec = NA_real_,
      early_stop_optimizer_time_sec = NA_real_,
      early_stop_loss_available = NA,
      early_stop_loss_reason = NA_character_,
      early_stop_chunked_optimizer = NA,
      early_stop_trace_epochs = NA_character_,
      early_stop_trace_displacement = NA_character_,
      early_stop_trace_trustworthiness = NA_character_,
      early_stop_trace_trustworthiness_delta = NA_character_,
      early_stop_trace_neighbour_stability = NA_character_,
      early_stop_trace_neighbour_stability_delta = NA_character_,
      early_stop_backend_scope = NA_character_,
      early_stop_risk = NA_character_,
      init_strategy = NA_character_,
      init_backend = NA_character_,
      init_backend_reason = NA_character_,
      init_time_sec = NA_real_,
      init_optimizer_epochs = NA_real_,
      init_optimizer_time_sec = NA_real_,
      init_scale = NA_real_,
      init_spectral_n_iter = NA_real_,
      init_spectral_solver = NA_character_,
      init_spectral_graph = NA_character_,
      init_spectral_eigenvalues = NA_character_,
      init_spectral_exact_max_n = NA_real_,
      init_spectral_graph_nnz = NA_real_,
      init_spectral_graph_active_fraction = NA_real_,
      init_spectral_nystrom_landmarks = NA_real_,
      init_spectral_nystrom_fraction = NA_real_,
      init_spectral_nystrom_projection_k = NA_real_,
      init_spectral_nystrom_weight = NA_character_,
      init_spectral_nystrom_selection_requested = NA_character_,
      init_spectral_nystrom_selection_used = NA_character_,
      init_spectral_nystrom_landmark_knn_time_sec = NA_real_,
      init_spectral_nystrom_landmark_spectral_time_sec = NA_real_,
      init_spectral_nystrom_projection_time_sec = NA_real_,
      init_diffusion_time = NA_real_,
      init_diffusion_n_iter = NA_real_,
      init_diffusion_solver = NA_character_,
      init_diffusion_graph = NA_character_,
      init_diffusion_eigenvalues = NA_character_,
      init_diffusion_graph_nnz = NA_real_,
      init_diffusion_graph_active_fraction = NA_real_,
      init_laplacian_n_iter = NA_real_,
      init_laplacian_solver = NA_character_,
      init_laplacian_graph = NA_character_,
      init_laplacian_eigenvalues = NA_character_,
      init_laplacian_graph_nnz = NA_real_,
      init_laplacian_graph_active_fraction = NA_real_,
      init_laplacian_normalized_coordinates = NA,
      init_pca_method = NA_character_,
      init_pca_oversample = NA_real_,
      init_pca_power = NA_real_,
      init_landmark_n = NA_real_,
      init_landmark_fraction = NA_real_,
      init_landmark_selection_requested = NA_character_,
      init_landmark_selection_used = NA_character_,
      init_projection_k = NA_real_,
      init_projection_weight = NA_character_,
      init_landmark_epochs = NA_real_,
      init_landmark_knn_time_sec = NA_real_,
      init_landmark_embedding_time_sec = NA_real_,
      init_projection_time_sec = NA_real_,
      init_projection_backend = NA_character_,
      warm_start_enabled = NA,
      warm_start_cache_hit = NA,
      warm_start_cache_key = NA_character_,
      warm_start_previous_init = NA_character_,
      warm_start_previous_epochs = NA_real_,
      warm_start_refinement_epochs = NA_real_,
      warm_start_previous_init_time_sec = NA_real_,
      warm_start_previous_embedding_time_sec = NA_real_,
      warm_start_previous_build_time_sec = NA_real_,
      warm_start_this_row_setup_time_sec = NA_real_,
      warm_start_refinement_time_sec = NA_real_,
      warm_start_total_if_cache_miss_sec = NA_real_,
      warm_start_reuse_mode = NA_character_,
      warm_start_use_case = NA_character_,
      warm_start_parameter_delta = NA_character_,
      warm_start_bias_risk = NA_character_,
      epoch_budget_enabled = NA,
      epoch_budget_requested = NA_character_,
      epoch_budget_effective = NA_real_,
      epoch_budget_default_epochs = NA_real_,
      epoch_budget_ratio_to_default = NA_real_,
      epoch_budget_quality = NA_character_,
      epoch_budget_tsne_mode = NA_character_,
      epoch_budget_optimizer_backend = NA_character_,
      epoch_budget_speed_quality_tradeoff = NA_character_,
      epoch_budget_is_default = NA,
      coarse_to_fine_enabled = NA,
      coarse_to_fine_mode = NA_character_,
      coarse_to_fine_selection_requested = NA_character_,
      coarse_to_fine_selection_used = NA_character_,
      coarse_to_fine_count_requested = NA_real_,
      coarse_to_fine_fraction_requested = NA_real_,
      coarse_to_fine_n = NA_real_,
      coarse_to_fine_fraction = NA_real_,
      coarse_to_fine_k = NA_real_,
      coarse_to_fine_projection_k = NA_real_,
      coarse_to_fine_projection_weight = NA_character_,
      coarse_to_fine_coarse_epochs = NA_real_,
      coarse_to_fine_refinement_epochs = NA_real_,
      coarse_to_fine_selection_time_sec = NA_real_,
      coarse_to_fine_knn_time_sec = NA_real_,
      coarse_to_fine_embedding_time_sec = NA_real_,
      coarse_to_fine_projection_time_sec = NA_real_,
      coarse_to_fine_refinement_time_sec = NA_real_,
      coarse_to_fine_setup_time_sec = NA_real_,
      coarse_to_fine_projection_entropy = NA_real_,
      coarse_to_fine_projection_zero_neighbor_fraction = NA_real_,
      coarse_to_fine_projection_bandwidth_mean = NA_real_,
      coarse_to_fine_projection_backend = NA_character_,
      coarse_to_fine_refinement_backend = NA_character_,
      coarse_to_fine_init_backend = NA_character_,
      coarse_to_fine_expected_gain = NA_character_,
      coarse_to_fine_risk = NA_character_,
      landmark_enabled = NA,
      landmark_approximation = NA_character_,
      landmark_mode = NA_character_,
      landmark_selection = NA_character_,
      landmark_selection_requested = NA_character_,
      landmark_selection_used = NA_character_,
      landmark_count_requested = NA_real_,
      landmark_fraction_requested = NA_real_,
      landmark_n = NA_real_,
      landmark_fraction = NA_real_,
      landmark_label_classes_total = NA_real_,
      landmark_label_classes_present = NA_real_,
      landmark_label_missing_classes = NA_real_,
      landmark_label_min_count = NA_real_,
      landmark_label_min_fraction = NA_real_,
      landmark_rare_label_count = NA_real_,
      landmark_rare_label_present = NA,
      stratified_landmark_source = NA_character_,
      stratified_landmark_allocation = NA_character_,
      stratified_landmark_time_sec = NA_real_,
      stratified_landmark_n_strata = NA_real_,
      stratified_landmark_strata_sampled = NA_real_,
      stratified_landmark_missing_strata = NA_real_,
      stratified_landmark_min_stratum_size = NA_real_,
      stratified_landmark_max_stratum_size = NA_real_,
      stratified_landmark_min_selected_per_stratum = NA_real_,
      stratified_landmark_max_selected_per_stratum = NA_real_,
      stratified_landmark_balance_ratio = NA_real_,
      stratified_landmark_cluster_k = NA_real_,
      stratified_landmark_cluster_feature_dims = NA_real_,
      density_landmark_alpha = NA_real_,
      density_landmark_k = NA_real_,
      density_landmark_time_sec = NA_real_,
      density_landmark_weight_min = NA_real_,
      density_landmark_weight_median = NA_real_,
      density_landmark_weight_max = NA_real_,
      density_landmark_weight_mean = NA_real_,
      density_landmark_selected_weight_mean = NA_real_,
      density_landmark_selected_to_global_weight_ratio = NA_real_,
      density_landmark_mean_distance_median = NA_real_,
      hybrid_landmark_alpha = NA_real_,
      hybrid_landmark_beta = NA_real_,
      hybrid_landmark_k = NA_real_,
      hybrid_landmark_time_sec = NA_real_,
      hybrid_landmark_density_time_sec = NA_real_,
      hybrid_landmark_feature_dims = NA_real_,
      hybrid_landmark_formula = NA_character_,
      hybrid_landmark_density_weight_median = NA_real_,
      hybrid_landmark_density_selected_to_global_weight_ratio = NA_real_,
      hybrid_landmark_mean_distance_median = NA_real_,
      hybrid_landmark_cover_mean = NA_real_,
      hybrid_landmark_cover_median = NA_real_,
      hybrid_landmark_cover_max = NA_real_,
      rare_protected_tail_fraction = NA_real_,
      rare_protected_tail_oversample = NA_real_,
      rare_protected_n_quantiles = NA_real_,
      rare_protected_cluster_fraction = NA_real_,
      rare_protected_density_k = NA_real_,
      rare_protected_time_sec = NA_real_,
      rare_protected_density_time_sec = NA_real_,
      rare_protected_cluster_time_sec = NA_real_,
      rare_protected_tail_count = NA_real_,
      rare_protected_quantile_count = NA_real_,
      rare_protected_cluster_count = NA_real_,
      rare_protected_fill_count = NA_real_,
      rare_protected_tail_threshold = NA_real_,
      rare_protected_selected_tail_fraction = NA_real_,
      rare_protected_selected_mean_distance_ratio = NA_real_,
      rare_protected_selected_to_global_low_density_ratio = NA_real_,
      rare_protected_quantile_min_selected = NA_real_,
      rare_protected_quantile_max_selected = NA_real_,
      rare_protected_cluster_k = NA_real_,
      rare_protected_cluster_feature_dims = NA_real_,
      diversity_landmark_algorithm = NA_character_,
      diversity_landmark_time_sec = NA_real_,
      diversity_landmark_feature_dims = NA_real_,
      diversity_landmark_cover_mean = NA_real_,
      diversity_landmark_cover_median = NA_real_,
      diversity_landmark_cover_max = NA_real_,
      diversity_landmark_leverage_selected_to_global_ratio = NA_real_,
      landmark_projection_k = NA_real_,
      landmark_interpolation = NA_character_,
      landmark_projection_model = NA_character_,
      landmark_projection_weight = NA_character_,
      landmark_projection_bandwidth_rule = NA_character_,
      landmark_projection_bandwidth_mean = NA_real_,
      landmark_projection_weight_entropy = NA_real_,
      landmark_projection_zero_neighbor_fraction = NA_real_,
      landmark_projection_time_sec = NA_real_,
      landmark_landmark_knn_time_sec = NA_real_,
      landmark_landmark_embedding_time_sec = NA_real_,
      landmark_affine_ridge = NA_real_,
      landmark_affine_weight = NA_character_,
      landmark_affine_rank_mean = NA_real_,
      landmark_affine_condition_median = NA_real_,
      landmark_affine_condition_max = NA_real_,
      landmark_affine_fallback_fraction = NA_real_,
      landmark_affine_clipped_fraction = NA_real_,
      landmark_affine_clip_multiplier = NA_real_,
      landmark_affine_blend = NA_real_,
      landmark_refinement = NA_character_,
      landmark_refinement_epochs = NA_real_,
      landmark_refinement_time_sec = NA_real_,
      landmark_projection_backend = NA_character_,
      landmark_interpolation_backend = NA_character_,
      landmark_interpolation_backend_reason = NA_character_,
      landmark_refinement_backend = NA_character_,
      landmark_refinement_knn_backend = NA_character_,
      landmark_refinement_knn_backend_reason = NA_character_,
      subsample_strategy = NA_character_,
      subsample_stratified = NA,
      benchmark_forced_k = NA_real_,
      benchmark_standardize = NA,
      graph_adaptive_min_k = NA_real_,
      graph_adaptive_max_k = NA_real_,
      graph_adaptive_mean_k = NA_real_,
      graph_adaptive_density_quantile = NA_real_,
      graph_adaptive_density_cor = NA_real_,
      graph_adaptive_dense_fraction = NA_real_,
      graph_adaptive_sparse_fraction = NA_real_,
      graph_distance_prune_drop_fraction = NA_real_,
      graph_distance_prune_percentile = NA_real_,
      graph_distance_prune_removed_edges_mean = NA_real_,
      graph_distance_prune_threshold_mean = NA_real_,
      graph_distance_prune_threshold_min = NA_real_,
      graph_distance_prune_threshold_max = NA_real_,
      graph_distance_prune_removed_distance_mean = NA_real_,
      graph_sparsification_method = NA_character_,
      graph_sparsification_keep_fraction = NA_real_,
      graph_sparsification_target_k = NA_real_,
      graph_sparsification_undirected_edges = NA_real_,
      graph_sparsification_spectral_rank = NA_real_,
      graph_sparsification_spectral_time_sec = NA_real_,
      graph_sparsification_leverage_mean = NA_real_,
      graph_sparsification_leverage_min = NA_real_,
      graph_sparsification_leverage_max = NA_real_,
      graph_sparsification_resistance_mean = NA_real_,
      graph_sparsification_weight_mean = NA_real_,
      graph_mst_rescue_enabled = NA_real_,
      graph_mst_rescue_base_components = NA_real_,
      graph_mst_rescue_components_before = NA_real_,
      graph_mst_rescue_components_after = NA_real_,
      graph_mst_rescue_forest_edges = NA_real_,
      graph_mst_rescue_added_forest_edges = NA_real_,
      graph_mst_rescue_added_directed_edges = NA_real_,
      graph_mst_rescue_mean_degree_before = NA_real_,
      graph_mst_rescue_mean_degree_after = NA_real_,
      graph_density_correction_method = NA_character_,
      graph_density_correction_quantile = NA_real_,
      graph_density_correction_strength = NA_real_,
      graph_density_scale_mean = NA_real_,
      graph_density_scale_min = NA_real_,
      graph_density_scale_max = NA_real_,
      graph_density_scale_cv = NA_real_,
      graph_density_sparse_fraction = NA_real_,
      graph_density_correction_mean = NA_real_,
      graph_density_correction_min = NA_real_,
      graph_density_correction_max = NA_real_,
      graph_density_correction_clamp_fraction = NA_real_,
      graph_density_corrected_distance_scale_cor = NA_real_,
      umap_graph_set_op_mix_ratio = NA_real_,
      umap_graph_local_connectivity = NA_real_,
      umap_graph_weight_power = NA_real_,
      umap_graph_target_scale = NA_real_,
      umap_graph_distance_transform = NA_character_,
      umap_graph_mean_weight = NA_real_,
      umap_graph_min_weight = NA_real_,
      umap_graph_max_weight = NA_real_,
      graph_edge_sampling_method = NA_character_,
      graph_edge_sampling_fraction = NA_real_,
      graph_edge_sampling_weight_power = NA_real_,
      graph_edge_sampling_include_top = NA_real_,
      graph_edge_sampling_target_scale = NA_real_,
      graph_edge_sampling_mean_selected_weight = NA_real_,
      graph_edge_sampling_mean_candidate_weight = NA_real_,
      graph_edge_sampling_selected_to_candidate_weight_ratio = NA_real_,
      pair_resampling_mode = NA_character_,
      pair_resampling_pair_family = NA_character_,
      pair_resampling_transfer_mode = NA_character_,
      pair_resampling_refreshes = NA_real_,
      pair_resampling_stage_count = NA_real_,
      pair_resampling_stage_epochs = NA_character_,
      pair_resampling_warmup_epochs = NA_real_,
      pair_resampling_refine_epochs = NA_real_,
      pair_resampling_keep_fraction = NA_real_,
      pair_resampling_weight_power = NA_real_,
      pair_resampling_include_top = NA_real_,
      pair_resampling_seed_stride = NA_real_,
      pair_resampling_final_graph = NA_character_,
      triplet_aux_enabled = NA,
      triplet_aux_weight = NA_real_,
      triplet_aux_samples_per_edge = NA_real_,
      triplet_aux_transfer_mode = NA_character_,
	      triplet_aux_native_backend = NA_character_,
	      triplet_aux_optimizer_backend = NA_character_,
	      triplet_aux_n_epochs = NA_real_,
	      triplet_aux_negative_sample_rate = NA_real_,
	      triplet_structured_enabled = NA,
	      triplet_structured_weight = NA_real_,
	      triplet_structured_n_inliers = NA_real_,
	      triplet_structured_n_outliers = NA_real_,
	      triplet_structured_n_random = NA_real_,
	      triplet_structured_total = NA_real_,
	      triplet_structured_mode = NA_character_,
	      global_random_triplet_enabled = NA,
	      global_random_triplet_weight = NA_real_,
	      global_random_triplets_per_point = NA_real_,
	      global_random_trimap_extra_negatives = NA_real_,
	      global_random_negative_source = NA_character_,
	      global_random_transfer_mode = NA_character_,
	      global_random_effective_negative_sample_rate = NA_real_,
		      hard_negative_enabled = NA,
		      hard_negative_rate = NA_real_,
		      hard_negative_weight_multiplier = NA_real_,
		      hard_negative_candidate_source = NA_character_,
		      hard_negative_transfer_mode = NA_character_,
		      semihard_triplet_enabled = NA,
		      semihard_triplet_rate = NA_real_,
		      semihard_triplet_weight_multiplier = NA_real_,
		      semihard_triplet_candidate_source = NA_character_,
		      semihard_triplet_transfer_mode = NA_character_,
		      triplet_mining_approximate = NA,
		      triplet_mining_graph_source = NA_character_,
		      triplet_mining_source_detail = NA_character_,
		      triplet_mining_knn_backend = NA_character_,
		      triplet_mining_candidate_source = NA_character_,
		      triplet_mining_transfer_mode = NA_character_,
		      triplet_mining_recall_at_k = NA_real_,
		      triplet_mining_rank_correlation = NA_real_,
		      triplet_mining_distance_error = NA_real_,
		      umap_negative_sample_rate = NA_real_,
      umap_transfer_mode = NA_character_,
      graph_tsne_affinity_mode = NA_character_,
      graph_tsne_affinity_perplexities = NA_character_,
      graph_tsne_affinity_num_scales = NA_real_,
      graph_tsne_affinity_temperature = NA_real_,
      graph_tsne_affinity_entropy_mean = NA_real_,
      graph_tsne_affinity_effective_perplexity_mean = NA_real_,
      graph_tsne_affinity_sigma_mean = NA_real_,
      graph_tsne_affinity_sigma_min = NA_real_,
      graph_tsne_affinity_sigma_max = NA_real_,
      graph_tsne_affinity_prob_min = NA_real_,
      graph_tsne_affinity_prob_max = NA_real_,
      graph_multiscale_perplexities = NA_character_,
      graph_multiscale_num_scales = NA_real_,
      graph_multiscale_required_k = NA_real_,
      graph_multiscale_effective_k_values = NA_character_,
      graph_multiscale_transfer_mode = NA_character_,
      graph_multiscale_uses_tsne_affinity = NA_real_,
      pacmap_transfer_mode = NA_character_,
      pacmap_auxiliary_pair_family = NA_character_,
      pacmap_mid_near_pairs_per_point = NA_real_,
      pacmap_mid_near_fraction = NA_real_,
      pacmap_mid_near_requested_fraction = NA_real_,
      pacmap_mid_near_distance_scale = NA_real_,
      pacmap_mid_near_fallback_fraction = NA_real_,
      pacmap_mid_near_rank_mean = NA_real_,
      pacmap_mid_near_emphasis_strength = NA_real_,
      pacmap_mid_near_emphasis_distance_multiplier = NA_real_,
      pacmap_near_ratio = NA_real_,
      pacmap_mid_ratio = NA_real_,
      pacmap_far_ratio = NA_real_,
      pacmap_near_pairs_per_point = NA_real_,
      pacmap_mid_pairs_per_point = NA_real_,
      pacmap_far_pairs_per_point = NA_real_,
      pacmap_far_pair_fraction = NA_real_,
      pacmap_far_distance_scale = NA_real_,
      pacmap_far_fallback_fraction = NA_real_,
      pacmap_far_repulsion_rate = NA_real_,
      pacmap_phase_schedule = NA_character_,
      pacmap_phase_total_epochs = NA_real_,
      pacmap_phase_epoch_multiplier = NA_real_,
      pacmap_phase_warmup_epochs = NA_real_,
      pacmap_phase_refine_epochs = NA_real_,
      pacmap_phase_transfer_detail = NA_character_,
      trimap_transfer_mode = NA_character_,
      trimap_triplet_family = NA_character_,
      trimap_inlier_ratio = NA_real_,
      trimap_semihard_ratio = NA_real_,
      trimap_global_anchor_ratio = NA_real_,
      trimap_inlier_pairs_per_point = NA_real_,
      trimap_semihard_pairs_per_point = NA_real_,
      trimap_global_anchor_pairs_per_point = NA_real_,
      trimap_semihard_fraction = NA_real_,
      trimap_global_anchor_fraction = NA_real_,
      trimap_semihard_distance_scale = NA_real_,
      trimap_global_anchor_distance_scale = NA_real_,
      trimap_semihard_fallback_fraction = NA_real_,
      trimap_global_anchor_fallback_fraction = NA_real_,
      trimap_semihard_rank_mean = NA_real_,
      trimap_candidate_seed = NA_real_,
      trimap_native_explicit_triplets = NA_real_,
      trimap_triplet_proxy_detail = NA_character_,
      early_exaggeration_factor = NA_real_,
      early_exaggeration_duration_fraction = NA_real_,
      early_exaggeration_total_epochs = NA_real_,
      early_exaggeration_warmup_epochs = NA_real_,
      early_exaggeration_refine_epochs = NA_real_,
      early_exaggeration_transfer_mode = NA_character_,
      early_exaggeration_distance_scale = NA_real_,
      early_exaggeration_schedule_mode = NA_character_,
      late_exaggeration_factor = NA_real_,
      late_exaggeration_duration_fraction = NA_real_,
      late_exaggeration_total_epochs = NA_real_,
      late_exaggeration_requested_epochs = NA_real_,
      late_exaggeration_normal_epochs = NA_real_,
      late_exaggeration_late_epochs = NA_real_,
      late_exaggeration_start_iter = NA_real_,
      late_exaggeration_transfer_mode = NA_character_,
      late_exaggeration_distance_scale = NA_real_,
      late_exaggeration_schedule_mode = NA_character_,
      optimizer_mode = NA_character_,
      optimizer_schedule = NA_character_,
      optimizer_momentum = NA_real_,
      optimizer_final_momentum = NA_real_,
      optimizer_switch_iter = NA_real_,
      optimizer_learning_rate = NA_real_,
      optimizer_learning_rate_multiplier = NA_real_,
      optimizer_adam_beta1 = NA_real_,
      optimizer_adam_beta2 = NA_real_,
      optimizer_adam_epsilon = NA_real_,
      optimizer_transfer_mode = NA_character_,
      learning_rate_rule = NA_character_,
      learning_rate_value = NA_real_,
      learning_rate_base_default = NA_real_,
      learning_rate_scale = NA_real_,
      learning_rate_transfer_mode = NA_character_,
      adaptive_lr_enabled = NA,
      adaptive_lr_schedule = NA_character_,
      adaptive_lr_total_epochs = NA_real_,
      adaptive_lr_chunk_epochs = NA_real_,
      adaptive_lr_chunks_run = NA_real_,
      adaptive_lr_base_learning_rate = NA_real_,
      adaptive_lr_final_learning_rate = NA_real_,
      adaptive_lr_final_multiplier = NA_real_,
      adaptive_lr_optimizer = NA_character_,
      adaptive_lr_native = NA,
      adaptive_lr_chunked = NA,
      adaptive_lr_backend = NA_character_,
      adaptive_lr_inner_decay = NA_character_,
      adaptive_lr_init_strategy = NA_character_,
      adaptive_lr_init_time_sec = NA_real_,
      adaptive_lr_optimizer_time_sec = NA_real_,
      adaptive_lr_trace_epochs = NA_character_,
      adaptive_lr_trace_learning_rate = NA_character_,
      adaptive_lr_trace_multiplier = NA_character_,
      adaptive_lr_note = NA_character_,
      mini_batch_enabled = NA,
      mini_batch_backend = NA_character_,
      mini_batch_mode = NA_character_,
      mini_batch_batch_fraction = NA_real_,
      mini_batch_effective_k = NA_real_,
      mini_batch_chunks = NA_real_,
      mini_batch_chunk_epochs = NA_character_,
      mini_batch_total_epochs = NA_real_,
      mini_batch_refreshes = NA_real_,
      mini_batch_sampling = NA_character_,
      mini_batch_weight_power = NA_real_,
      mini_batch_include_top = NA_real_,
      mini_batch_init_strategy = NA_character_,
      mini_batch_init_time_sec = NA_real_,
      mini_batch_graph_time_sec = NA_real_,
      mini_batch_optimizer_time_sec = NA_real_,
      mini_batch_trace_effective_k = NA_character_,
      mini_batch_trace_retention = NA_character_,
      mini_batch_experimental = NA,
      mini_batch_note = NA_character_,
      deterministic_batch_row_batch_size = NA_real_,
      deterministic_batch_chunks_per_epoch = NA_real_,
      deterministic_batch_reduction = NA_character_,
      deterministic_batch_atomic_updates = NA,
      deterministic_batch_reproducible_given_threads = NA,
      sparse_edge_batch_enabled = NA,
      sparse_edge_batch_backend = NA_character_,
      sparse_edge_batch_mode = NA_character_,
      sparse_edge_batch_storage = NA_character_,
      sparse_edge_batch_edge_batch_size = NA_real_,
      sparse_edge_batch_chunks_per_epoch = NA_real_,
      sparse_edge_batch_n_epochs = NA_real_,
      sparse_edge_batch_negative_sample_rate = NA_real_,
      sparse_edge_batch_learning_rate = NA_real_,
      sparse_edge_batch_threads = NA_real_,
      sparse_edge_batch_atomic_updates = NA,
      sparse_edge_batch_edge_list_copy = NA,
      sparse_edge_batch_triplet_chunks = NA,
      sparse_edge_batch_affinity_chunks = NA,
      sparse_edge_batch_aux_memory_mb = NA_real_,
      sparse_edge_batch_graph_time_sec = NA_real_,
      sparse_edge_batch_init_time_sec = NA_real_,
      sparse_edge_batch_optimizer_time_sec = NA_real_,
      sparse_edge_batch_status = NA_character_,
      sparse_edge_batch_note = NA_character_,
      vectorized_edge_enabled = NA,
      vectorized_edge_backend = NA_character_,
      vectorized_edge_storage = NA_character_,
      vectorized_edge_batch_size = NA_real_,
      vectorized_edge_n_edges = NA_real_,
      vectorized_edge_n_epochs = NA_real_,
      vectorized_edge_negative_sample_rate = NA_real_,
      vectorized_edge_threads = NA_real_,
      vectorized_edge_learning_rate = NA_real_,
      vectorized_edge_simd = NA_character_,
      vectorized_edge_gpu_native = NA,
      vectorized_edge_graph_time_sec = NA_real_,
      vectorized_edge_init_time_sec = NA_real_,
      vectorized_edge_optimizer_time_sec = NA_real_,
      vectorized_edge_status = NA_character_,
      vectorized_edge_note = NA_character_,
      atomic_sgd_enabled = NA,
      atomic_sgd_backend = NA_character_,
      atomic_sgd_update_mode = NA_character_,
      atomic_sgd_storage = NA_character_,
      atomic_sgd_n_edges = NA_real_,
      atomic_sgd_n_epochs = NA_real_,
      atomic_sgd_negative_sample_rate = NA_real_,
      atomic_sgd_threads = NA_real_,
      atomic_sgd_learning_rate = NA_real_,
      atomic_sgd_learning_rate_scale = NA_real_,
      atomic_sgd_coordinate_clip = NA_real_,
      atomic_sgd_openmp = NA,
      atomic_sgd_gpu_native = NA,
      atomic_sgd_nondeterministic = NA,
      atomic_sgd_graph_time_sec = NA_real_,
      atomic_sgd_init_time_sec = NA_real_,
      atomic_sgd_optimizer_time_sec = NA_real_,
      atomic_sgd_status = NA_character_,
      atomic_sgd_note = NA_character_,
      tsne_bh_theta = NA_real_,
      tsne_bh_perplexity = NA_real_,
      tsne_bh_n_epochs = NA_real_,
      tsne_bh_learning_rate = NA_real_,
      tsne_bh_n_threads = NA_real_,
      tsne_bh_stop_lying_iter = NA_real_,
      tsne_bh_mom_switch_iter = NA_real_,
      fft_interpolation_mode = NA_character_,
      fft_interpolation_experimental = NA_real_,
      fft_interpolation_backend = NA_character_,
      fft_interpolation_transfer_scope = NA_character_,
      fft_interpolation_native_repulsive_field = NA_real_,
      fft_interpolation_refine_epochs = NA_real_,
      fft_grid_nterms = 3L,
      fft_grid_intervals_per_integer = 1,
      fft_grid_min_num_intervals = 50L,
      output_metric = NA_character_,
      output_metric_transform = NA_character_,
      output_metric_native = NA,
      output_metric_projection_scale = NA_real_,
      output_metric_curvature = NA_real_,
      output_metric_radius_mean = NA_real_,
      output_metric_radius_max = NA_real_,
      output_metric_norm_mean = NA_real_
    )
  if (!is.null(strategy$annoy_n_trees)) {
    ctx$annoy_n_trees <- as.integer(strategy$annoy_n_trees)
    ctx$annoy_search_multiplier <- strategy$annoy_search_multiplier
  }
  if (!is.null(strategy$hnsw_m)) {
    ctx$hnsw_m <- as.integer(strategy$hnsw_m)
    ctx$hnsw_ef_construction <- as.integer(strategy$hnsw_ef_construction)
    ctx$hnsw_ef_search <- as.integer(strategy$hnsw_ef_search)
  }
  if (!is.null(strategy$nndescent_n_iters)) {
    ctx$nndescent_n_iters <- as.integer(strategy$nndescent_n_iters)
    ctx$nndescent_delta <- as.numeric(strategy$nndescent_delta)
    ctx$nndescent_rho <- strategy$nndescent_rho
    ctx$nndescent_max_candidates <- strategy$nndescent_max_candidates
  }
  if (!is.null(strategy$faiss_index_type)) {
    ctx$faiss_index_type <- strategy$faiss_index_type
    ctx$faiss_nlist <- strategy$faiss_nlist
    ctx$faiss_nprobe <- strategy$faiss_nprobe
    ctx$faiss_pq_m <- strategy$faiss_pq_m
    ctx$faiss_pq_nbits <- strategy$faiss_pq_nbits
    ctx$faiss_hnsw_m <- strategy$faiss_hnsw_m
    ctx$faiss_hnsw_ef_construction <- strategy$faiss_hnsw_ef_construction
    ctx$faiss_hnsw_ef_search <- strategy$faiss_hnsw_ef_search
    ctx$faiss_gpu_device <- if (!is.null(strategy$faiss_gpu_device)) strategy$faiss_gpu_device else 0L
    ctx$faiss_gpu_use_float16 <- isTRUE(strategy$faiss_gpu_use_float16)
    ctx$faiss_gpu_temp_memory_mb <- if (!is.null(strategy$faiss_gpu_temp_memory_mb)) strategy$faiss_gpu_temp_memory_mb else NA_integer_
  }
  if (!is.null(strategy$cuml_algorithm)) {
    ctx$cuml_algorithm <- strategy$cuml_algorithm
    ctx$cuml_nlist <- strategy$cuml_nlist
    ctx$cuml_nprobe <- strategy$cuml_nprobe
    ctx$cuml_device <- strategy$cuml_device
  }
  if (!is.null(strategy$tsne_bh_theta)) {
    ctx$tsne_bh_theta <- as.numeric(strategy$tsne_bh_theta)
    bh_cfg <- tsne_barnes_hut_config(ctx, ctx$tsne_bh_theta)
    ctx$tsne_bh_perplexity <- safe_number(bh_cfg$perplexity)
    ctx$tsne_bh_n_epochs <- safe_number(bh_cfg$n_epochs)
    ctx$tsne_bh_learning_rate <- safe_number(bh_cfg$learning_rate)
    ctx$tsne_bh_n_threads <- safe_number(bh_cfg$n_threads)
    ctx$tsne_bh_stop_lying_iter <- safe_number(bh_cfg$stop_lying_iter)
    ctx$tsne_bh_mom_switch_iter <- safe_number(bh_cfg$mom_switch_iter)
  }
  if (!is.null(strategy$fft_interpolation_mode)) {
    ctx$fft_interpolation_mode <- as.character(strategy$fft_interpolation_mode)
    ctx$fft_interpolation_experimental <- as.numeric(isTRUE(strategy$fft_interpolation_experimental))
    ctx$fft_interpolation_backend <- "cpu_external_fitsne"
    ctx$fft_grid_nterms <- if (!is.null(strategy$fft_grid_nterms)) {
      as.integer(strategy$fft_grid_nterms)
    } else {
      3L
    }
    ctx$fft_grid_intervals_per_integer <- if (!is.null(strategy$fft_grid_intervals_per_integer)) {
      as.numeric(strategy$fft_grid_intervals_per_integer)
    } else {
      1
    }
    ctx$fft_grid_min_num_intervals <- if (!is.null(strategy$fft_grid_min_num_intervals)) {
      as.integer(strategy$fft_grid_min_num_intervals)
    } else {
      50L
    }
    ctx$fft_interpolation_transfer_scope <- if (identical(method, "tsne")) {
      "true_tsne_fft"
    } else {
      "experimental_fft_warm_start_refine"
    }
    ctx$fft_interpolation_native_repulsive_field <- as.numeric(identical(method, "tsne"))
    ctx$fft_interpolation_refine_epochs <- if (identical(method, "tsne")) {
      0
    } else {
      max(1L, min(as.integer(ctx$short_epochs), 80L))
    }
  }
  if (!is.null(strategy$multiscale_perplexities)) {
    requested_perplexities <- sort(unique(as.integer(strategy$multiscale_perplexities)))
    ctx$graph_multiscale_perplexities <- paste(requested_perplexities, collapse = ",")
    ctx$graph_multiscale_num_scales <- length(requested_perplexities)
    ctx$graph_multiscale_required_k <- if (identical(method, "tsne")) {
      3L * max(requested_perplexities)
    } else {
      max(requested_perplexities)
    }
    ctx$graph_multiscale_effective_k_values <- if (identical(method, "tsne")) {
      paste(3L * requested_perplexities, collapse = ",")
    } else {
      paste(requested_perplexities, collapse = ",")
    }
    ctx$graph_multiscale_transfer_mode <- multiscale_perplexity_transfer_mode(method)
    ctx$graph_multiscale_uses_tsne_affinity <- as.numeric(identical(method, "tsne"))
  }
  if (!is.null(strategy$early_exaggeration_factor)) {
    ee_cfg <- early_exaggeration_config(
      ctx,
      strategy$early_exaggeration_factor,
      strategy$early_exaggeration_duration_fraction
    )
    ctx$early_exaggeration_factor <- safe_number(ee_cfg$early_exaggeration_factor)
    ctx$early_exaggeration_duration_fraction <- safe_number(ee_cfg$early_exaggeration_duration_fraction)
    ctx$early_exaggeration_total_epochs <- safe_number(ee_cfg$n_epochs)
    ctx$early_exaggeration_warmup_epochs <- safe_number(ee_cfg$early_exaggeration_warmup_epochs)
    ctx$early_exaggeration_refine_epochs <- safe_number(ee_cfg$early_exaggeration_refine_epochs)
    ctx$early_exaggeration_transfer_mode <- as.character(ee_cfg$early_exaggeration_transfer_mode)
    ctx$early_exaggeration_distance_scale <- safe_number(ee_cfg$early_exaggeration_distance_scale)
    ctx$early_exaggeration_schedule_mode <- as.character(ee_cfg$early_exaggeration_schedule_mode)
  }
  if (!is.null(strategy$late_exaggeration_factor)) {
    le_cfg <- late_exaggeration_config(
      ctx,
      strategy$late_exaggeration_factor,
      strategy$late_exaggeration_duration_fraction
    )
    ctx$late_exaggeration_factor <- safe_number(le_cfg$late_exaggeration_factor)
    ctx$late_exaggeration_duration_fraction <- safe_number(le_cfg$late_exaggeration_duration_fraction)
    ctx$late_exaggeration_total_epochs <- safe_number(le_cfg$n_epochs)
    ctx$late_exaggeration_requested_epochs <- safe_number(le_cfg$late_exaggeration_requested_epochs)
    ctx$late_exaggeration_normal_epochs <- safe_number(le_cfg$late_exaggeration_normal_epochs)
    ctx$late_exaggeration_late_epochs <- safe_number(le_cfg$late_exaggeration_late_epochs)
    ctx$late_exaggeration_start_iter <- safe_number(le_cfg$late_exaggeration_start_iter)
    ctx$late_exaggeration_transfer_mode <- as.character(le_cfg$late_exaggeration_transfer_mode)
    ctx$late_exaggeration_distance_scale <- safe_number(le_cfg$late_exaggeration_distance_scale)
    ctx$late_exaggeration_schedule_mode <- as.character(le_cfg$late_exaggeration_schedule_mode)
  }
  if (!is.null(strategy$optimizer_mode)) {
    opt_cfg <- optimizer_schedule_config(ctx, strategy$optimizer_mode)
    ctx$optimizer_mode <- as.character(opt_cfg$optimizer_mode)
    ctx$optimizer_schedule <- as.character(opt_cfg$optimizer_schedule)
    ctx$optimizer_momentum <- safe_number(opt_cfg$optimizer_momentum)
    ctx$optimizer_final_momentum <- safe_number(opt_cfg$optimizer_final_momentum)
    ctx$optimizer_switch_iter <- safe_number(opt_cfg$optimizer_switch_iter)
    ctx$optimizer_learning_rate <- safe_number(opt_cfg$optimizer_learning_rate)
    ctx$optimizer_learning_rate_multiplier <- safe_number(opt_cfg$optimizer_learning_rate_multiplier)
    ctx$optimizer_adam_beta1 <- safe_number(opt_cfg$optimizer_adam_beta1)
    ctx$optimizer_adam_beta2 <- safe_number(opt_cfg$optimizer_adam_beta2)
    ctx$optimizer_adam_epsilon <- safe_number(opt_cfg$optimizer_adam_epsilon)
    ctx$optimizer_transfer_mode <- as.character(opt_cfg$optimizer_transfer_mode)
  }
  if (!is.null(strategy$learning_rate_rule)) {
    lr_cfg <- learning_rate_scaling_config(ctx, strategy$learning_rate_rule)
    ctx$learning_rate_rule <- as.character(lr_cfg$learning_rate_rule)
    ctx$learning_rate_value <- safe_number(lr_cfg$learning_rate_value)
    ctx$learning_rate_base_default <- safe_number(lr_cfg$learning_rate_base_default)
    ctx$learning_rate_scale <- safe_number(lr_cfg$learning_rate_scale)
    ctx$learning_rate_transfer_mode <- as.character(lr_cfg$learning_rate_transfer_mode)
  }
  if (!is.null(strategy$adaptive_lr_schedule)) {
    alr_cfg <- adaptive_lr_config(
      ctx,
      strategy$adaptive_lr_schedule,
      total_epochs = if (is.finite(safe_number(strategy$adaptive_lr_total_epochs))) {
        as.integer(strategy$adaptive_lr_total_epochs)
      } else {
        NULL
      },
      chunk_epochs = as.integer(strategy$adaptive_lr_chunk_epochs),
      final_multiplier = safe_number(strategy$adaptive_lr_final_multiplier, 0.05)
    )
    ctx$adaptive_lr_enabled <- TRUE
    ctx$adaptive_lr_schedule <- as.character(alr_cfg$adaptive_lr_schedule)
    ctx$adaptive_lr_total_epochs <- safe_number(alr_cfg$adaptive_lr_total_epochs)
    ctx$adaptive_lr_chunk_epochs <- safe_number(alr_cfg$adaptive_lr_chunk_epochs)
    ctx$adaptive_lr_base_learning_rate <- safe_number(alr_cfg$adaptive_lr_base_learning_rate)
    ctx$adaptive_lr_final_multiplier <- safe_number(alr_cfg$adaptive_lr_final_multiplier)
    ctx$adaptive_lr_optimizer <- as.character(alr_cfg$adaptive_lr_optimizer)
    ctx$adaptive_lr_native <- safe_logical(alr_cfg$adaptive_lr_native)
    ctx$adaptive_lr_chunked <- safe_logical(alr_cfg$adaptive_lr_chunked)
    ctx$adaptive_lr_backend <- as.character(alr_cfg$adaptive_lr_backend)
    ctx$adaptive_lr_inner_decay <- as.character(alr_cfg$adaptive_lr_inner_decay)
    ctx$adaptive_lr_note <- as.character(alr_cfg$adaptive_lr_note)
  }
  if (!is.null(strategy$mini_batch_enabled)) {
    mb_cfg <- mini_batch_config(
      ctx,
      batch_fraction = safe_number(strategy$mini_batch_batch_fraction, 0.5),
      chunks = as.integer(safe_number(strategy$mini_batch_chunks, 4L)),
      weight_power = safe_number(strategy$mini_batch_weight_power, 1),
      include_top = as.integer(safe_number(strategy$mini_batch_include_top, 1L))
    )
    ctx$mini_batch_enabled <- TRUE
    ctx$mini_batch_backend <- as.character(mb_cfg$mini_batch_backend)
    ctx$mini_batch_mode <- as.character(mb_cfg$mini_batch_mode)
    ctx$mini_batch_batch_fraction <- safe_number(mb_cfg$mini_batch_batch_fraction)
    ctx$mini_batch_effective_k <- safe_number(mb_cfg$mini_batch_effective_k)
    ctx$mini_batch_chunks <- safe_number(mb_cfg$mini_batch_chunks)
    ctx$mini_batch_chunk_epochs <- as.character(mb_cfg$mini_batch_chunk_epochs)
    ctx$mini_batch_total_epochs <- safe_number(mb_cfg$mini_batch_total_epochs)
    ctx$mini_batch_refreshes <- safe_number(mb_cfg$mini_batch_refreshes)
    ctx$mini_batch_sampling <- as.character(mb_cfg$mini_batch_sampling)
    ctx$mini_batch_weight_power <- safe_number(mb_cfg$mini_batch_weight_power)
    ctx$mini_batch_include_top <- safe_number(mb_cfg$mini_batch_include_top)
    ctx$mini_batch_init_strategy <- as.character(mb_cfg$mini_batch_init_strategy)
    ctx$mini_batch_experimental <- safe_logical(mb_cfg$mini_batch_experimental)
    ctx$mini_batch_note <- as.character(mb_cfg$mini_batch_note)
  }
  if (!is.null(strategy$vectorized_edge_enabled)) {
    ve_cfg <- vectorized_edge_config(
      ctx,
      batch_size = as.integer(safe_number(strategy$vectorized_edge_batch_size, 4096L))
    )
    ctx$vectorized_edge_enabled <- TRUE
    ctx$vectorized_edge_backend <- if (identical(ctx$backend, "cpu")) {
      as.character(ve_cfg$vectorized_edge_backend)
    } else {
      as.character(ctx$backend)
    }
    ctx$vectorized_edge_storage <- as.character(ve_cfg$vectorized_edge_storage)
    ctx$vectorized_edge_batch_size <- safe_number(ve_cfg$vectorized_edge_batch_size)
    ctx$vectorized_edge_n_epochs <- safe_number(ve_cfg$vectorized_edge_n_epochs)
    ctx$vectorized_edge_negative_sample_rate <- safe_number(ve_cfg$vectorized_edge_negative_sample_rate)
    ctx$vectorized_edge_threads <- safe_number(ve_cfg$vectorized_edge_threads)
    ctx$vectorized_edge_learning_rate <- safe_number(ve_cfg$learning_rate)
    ctx$vectorized_edge_simd <- as.character(ve_cfg$vectorized_edge_simd)
    ctx$vectorized_edge_gpu_native <- safe_logical(ve_cfg$vectorized_edge_gpu_native)
    ctx$vectorized_edge_status <- if (identical(ctx$backend, "cpu")) {
      as.character(ve_cfg$vectorized_edge_status)
    } else {
      "not_supported"
    }
    ctx$vectorized_edge_note <- if (identical(ctx$backend, "cpu")) {
      as.character(ve_cfg$vectorized_edge_note)
    } else {
      paste0(
        "Native vectorized edge optimization is not implemented for backend `",
        ctx$backend,
        "` yet; no CPU fallback is reported as GPU."
      )
    }
  }
  if (!is.null(strategy$atomic_sgd_enabled)) {
    asgd_cfg <- atomic_sgd_config(
      ctx,
      learning_rate_scale = safe_number(strategy$atomic_sgd_learning_rate_scale, 0.005),
      coordinate_clip = safe_number(strategy$atomic_sgd_coordinate_clip, 25)
    )
    ctx$atomic_sgd_enabled <- TRUE
    ctx$atomic_sgd_backend <- if (identical(ctx$backend, "cpu")) {
      as.character(asgd_cfg$atomic_sgd_backend)
    } else {
      as.character(ctx$backend)
    }
    ctx$atomic_sgd_update_mode <- as.character(asgd_cfg$atomic_sgd_update_mode)
    ctx$atomic_sgd_storage <- as.character(asgd_cfg$atomic_sgd_storage)
    ctx$atomic_sgd_n_epochs <- safe_number(asgd_cfg$atomic_sgd_n_epochs)
    ctx$atomic_sgd_negative_sample_rate <- safe_number(asgd_cfg$atomic_sgd_negative_sample_rate)
    ctx$atomic_sgd_threads <- safe_number(asgd_cfg$atomic_sgd_threads)
    ctx$atomic_sgd_learning_rate <- safe_number(asgd_cfg$atomic_sgd_learning_rate)
    ctx$atomic_sgd_learning_rate_scale <- safe_number(asgd_cfg$atomic_sgd_learning_rate_scale)
    ctx$atomic_sgd_coordinate_clip <- safe_number(asgd_cfg$atomic_sgd_coordinate_clip)
    ctx$atomic_sgd_openmp <- safe_logical(asgd_cfg$atomic_sgd_openmp)
    ctx$atomic_sgd_gpu_native <- safe_logical(asgd_cfg$atomic_sgd_gpu_native)
    ctx$atomic_sgd_nondeterministic <- safe_logical(asgd_cfg$atomic_sgd_nondeterministic)
    ctx$atomic_sgd_status <- if (identical(ctx$backend, "cpu")) {
      as.character(asgd_cfg$atomic_sgd_status)
    } else {
      "not_supported"
    }
    ctx$atomic_sgd_note <- if (identical(ctx$backend, "cpu")) {
      as.character(asgd_cfg$atomic_sgd_note)
    } else {
      paste0(
        "Native atomic SGD is not implemented for backend `",
        ctx$backend,
        "` yet; no CPU fallback is reported as GPU."
      )
    }
  }
  if (!is.null(strategy$umap_negative_sample_rate)) {
    ns_cfg <- umap_negative_sampling_config(ctx, strategy$umap_negative_sample_rate)
    ctx$umap_negative_sample_rate <- safe_number(ns_cfg$umap_negative_sample_rate)
    ctx$umap_transfer_mode <- as.character(ns_cfg$umap_transfer_mode)
  }
  if (!is.null(strategy$pacmap_transfer_mode)) {
    ctx$pacmap_transfer_mode <- safe_character(strategy$pacmap_transfer_mode)
  }
  if (!is.null(strategy$pacmap_auxiliary_pair_family)) {
    ctx$pacmap_auxiliary_pair_family <- safe_character(strategy$pacmap_auxiliary_pair_family)
  }
  if (!is.null(strategy$pacmap_mid_near_requested_fraction)) {
    ctx$pacmap_mid_near_requested_fraction <- safe_number(strategy$pacmap_mid_near_requested_fraction)
  }
  if (!is.null(strategy$pacmap_mid_near_distance_scale)) {
    ctx$pacmap_mid_near_distance_scale <- safe_number(strategy$pacmap_mid_near_distance_scale)
  }
  if (!is.null(strategy$pacmap_mid_near_emphasis_strength)) {
    ctx$pacmap_mid_near_emphasis_strength <- safe_number(strategy$pacmap_mid_near_emphasis_strength)
  }
  if (!is.null(strategy$pacmap_mid_near_emphasis_distance_multiplier)) {
    ctx$pacmap_mid_near_emphasis_distance_multiplier <- safe_number(strategy$pacmap_mid_near_emphasis_distance_multiplier)
  }
  if (!is.null(strategy$pacmap_near_ratio)) {
    ctx$pacmap_near_ratio <- safe_number(strategy$pacmap_near_ratio)
  }
  if (!is.null(strategy$pacmap_mid_ratio)) {
    ctx$pacmap_mid_ratio <- safe_number(strategy$pacmap_mid_ratio)
  }
  if (!is.null(strategy$pacmap_far_ratio)) {
    ctx$pacmap_far_ratio <- safe_number(strategy$pacmap_far_ratio)
  }
  if (!is.null(strategy$pacmap_far_distance_scale)) {
    ctx$pacmap_far_distance_scale <- safe_number(strategy$pacmap_far_distance_scale)
  }
  if (!is.null(strategy$pacmap_far_repulsion_rate)) {
    ctx$pacmap_far_repulsion_rate <- safe_number(strategy$pacmap_far_repulsion_rate)
  }
  if (!is.null(strategy$pacmap_phase_warmup_fraction)) {
    phase_schedule <- if (!is.null(strategy$pacmap_phase_schedule)) strategy$pacmap_phase_schedule else "custom"
    phase_epoch_multiplier <- if (!is.null(strategy$pacmap_phase_epoch_multiplier)) {
      strategy$pacmap_phase_epoch_multiplier
    } else {
      1
    }
    phase_cfg <- pacmap_phase_config(
      ctx,
      ctx$pacmap_mid_near_requested_fraction,
      ctx$pacmap_mid_near_distance_scale,
      strategy$pacmap_phase_warmup_fraction,
      schedule = phase_schedule,
      epoch_multiplier = phase_epoch_multiplier
    )
    ctx$pacmap_phase_schedule <- safe_character(phase_cfg$pacmap_phase_schedule)
    ctx$pacmap_phase_total_epochs <- safe_number(phase_cfg$pacmap_phase_total_epochs)
    ctx$pacmap_phase_epoch_multiplier <- safe_number(phase_cfg$pacmap_phase_epoch_multiplier)
    ctx$pacmap_phase_warmup_epochs <- safe_number(phase_cfg$pacmap_phase_warmup_epochs)
    ctx$pacmap_phase_refine_epochs <- safe_number(phase_cfg$pacmap_phase_refine_epochs)
    ctx$pacmap_phase_transfer_detail <- safe_character(phase_cfg$pacmap_phase_transfer_detail)
  }
	  if (!is.null(strategy$pair_resampling_mode)) {
	    resample_cfg <- pair_resampling_config(
	      ctx,
      mode = strategy$pair_resampling_mode,
      keep_fraction = if (!is.null(strategy$pair_resampling_keep_fraction)) strategy$pair_resampling_keep_fraction else 0.75,
      refreshes = if (!is.null(strategy$pair_resampling_refreshes)) strategy$pair_resampling_refreshes else 1L,
      weight_power = if (!is.null(strategy$pair_resampling_weight_power)) strategy$pair_resampling_weight_power else 1,
      include_top = if (!is.null(strategy$pair_resampling_include_top)) strategy$pair_resampling_include_top else 1L,
      near_ratio = if (!is.null(strategy$pacmap_near_ratio)) strategy$pacmap_near_ratio else 0.50,
      mid_ratio = if (!is.null(strategy$pacmap_mid_ratio)) strategy$pacmap_mid_ratio else 0.30,
      far_ratio = if (!is.null(strategy$pacmap_far_ratio)) strategy$pacmap_far_ratio else 0.20
    )
    ctx$pair_resampling_mode <- safe_character(resample_cfg$pair_resampling_mode)
    ctx$pair_resampling_pair_family <- safe_character(resample_cfg$pair_resampling_pair_family)
    ctx$pair_resampling_transfer_mode <- safe_character(resample_cfg$pair_resampling_transfer_mode)
    ctx$pair_resampling_refreshes <- safe_number(resample_cfg$pair_resampling_refreshes)
    ctx$pair_resampling_stage_count <- safe_number(resample_cfg$pair_resampling_stage_count)
    ctx$pair_resampling_stage_epochs <- safe_character(resample_cfg$pair_resampling_stage_epochs)
    ctx$pair_resampling_warmup_epochs <- safe_number(resample_cfg$pair_resampling_warmup_epochs)
    ctx$pair_resampling_refine_epochs <- safe_number(resample_cfg$pair_resampling_refine_epochs)
    ctx$pair_resampling_keep_fraction <- safe_number(resample_cfg$pair_resampling_keep_fraction)
    ctx$pair_resampling_weight_power <- safe_number(resample_cfg$pair_resampling_weight_power)
    ctx$pair_resampling_include_top <- safe_number(resample_cfg$pair_resampling_include_top)
    ctx$pair_resampling_seed_stride <- safe_number(resample_cfg$pair_resampling_seed_stride)
    ctx$pair_resampling_final_graph <- safe_character(resample_cfg$pair_resampling_final_graph)
    ctx$pacmap_near_ratio <- safe_number(resample_cfg$pacmap_near_ratio)
	    ctx$pacmap_mid_ratio <- safe_number(resample_cfg$pacmap_mid_ratio)
	    ctx$pacmap_far_ratio <- safe_number(resample_cfg$pacmap_far_ratio)
	  }
  if (!is.null(strategy$trimap_transfer_mode)) {
    ctx$trimap_transfer_mode <- safe_character(strategy$trimap_transfer_mode)
  }
  if (!is.null(strategy$trimap_triplet_family)) {
    ctx$trimap_triplet_family <- safe_character(strategy$trimap_triplet_family)
  }
  if (!is.null(strategy$trimap_inlier_ratio)) {
    ctx$trimap_inlier_ratio <- safe_number(strategy$trimap_inlier_ratio)
  }
  if (!is.null(strategy$trimap_semihard_ratio)) {
    ctx$trimap_semihard_ratio <- safe_number(strategy$trimap_semihard_ratio)
  }
  if (!is.null(strategy$trimap_global_anchor_ratio)) {
    ctx$trimap_global_anchor_ratio <- safe_number(strategy$trimap_global_anchor_ratio)
  }
  if (!is.null(strategy$trimap_semihard_distance_scale)) {
    ctx$trimap_semihard_distance_scale <- safe_number(strategy$trimap_semihard_distance_scale)
  }
  if (!is.null(strategy$trimap_global_anchor_distance_scale)) {
    ctx$trimap_global_anchor_distance_scale <- safe_number(strategy$trimap_global_anchor_distance_scale)
  }
  if (!is.null(strategy$trimap_native_explicit_triplets)) {
    ctx$trimap_native_explicit_triplets <- safe_number(strategy$trimap_native_explicit_triplets)
  }
  if (!is.null(strategy$triplet_aux_enabled)) {
    ctx$triplet_aux_enabled <- safe_logical(strategy$triplet_aux_enabled)
  }
  if (!is.null(strategy$triplet_aux_weight)) {
    ctx$triplet_aux_weight <- safe_number(strategy$triplet_aux_weight)
  }
  if (!is.null(strategy$triplet_aux_samples_per_edge)) {
    ctx$triplet_aux_samples_per_edge <- safe_number(strategy$triplet_aux_samples_per_edge)
  }
	  if (!is.null(strategy$triplet_aux_native_backend)) {
	    ctx$triplet_aux_native_backend <- safe_character(strategy$triplet_aux_native_backend)
	  }
	  if (!is.null(strategy$triplet_structured_enabled)) {
	    ctx$triplet_structured_enabled <- safe_logical(strategy$triplet_structured_enabled)
	  }
	  if (!is.null(strategy$triplet_structured_weight)) {
	    ctx$triplet_structured_weight <- safe_number(strategy$triplet_structured_weight)
	  }
	  if (!is.null(strategy$triplet_structured_n_inliers)) {
	    ctx$triplet_structured_n_inliers <- safe_number(strategy$triplet_structured_n_inliers)
	  }
	  if (!is.null(strategy$triplet_structured_n_outliers)) {
	    ctx$triplet_structured_n_outliers <- safe_number(strategy$triplet_structured_n_outliers)
	  }
	  if (!is.null(strategy$triplet_structured_n_random)) {
	    ctx$triplet_structured_n_random <- safe_number(strategy$triplet_structured_n_random)
	  }
	  if (!is.null(strategy$triplet_structured_total)) {
	    ctx$triplet_structured_total <- safe_number(strategy$triplet_structured_total)
	  }
	  if (!is.null(strategy$triplet_structured_mode)) {
	    ctx$triplet_structured_mode <- safe_character(strategy$triplet_structured_mode)
	  }
	  if (!is.null(strategy$global_random_triplet_enabled)) {
	    ctx$global_random_triplet_enabled <- safe_logical(strategy$global_random_triplet_enabled)
	  }
	  if (!is.null(strategy$global_random_triplet_weight)) {
	    ctx$global_random_triplet_weight <- safe_number(strategy$global_random_triplet_weight)
	  }
	  if (!is.null(strategy$global_random_triplets_per_point)) {
	    ctx$global_random_triplets_per_point <- safe_number(strategy$global_random_triplets_per_point)
	  }
	  if (!is.null(strategy$global_random_trimap_extra_negatives)) {
	    ctx$global_random_trimap_extra_negatives <- safe_number(strategy$global_random_trimap_extra_negatives)
	  }
	  if (!is.null(strategy$global_random_negative_source)) {
	    ctx$global_random_negative_source <- safe_character(strategy$global_random_negative_source)
	  }
	  if (!is.null(strategy$global_random_transfer_mode)) {
	    ctx$global_random_transfer_mode <- safe_character(strategy$global_random_transfer_mode)
	  }
	  if (!is.null(strategy$global_random_effective_negative_sample_rate)) {
	    ctx$global_random_effective_negative_sample_rate <- safe_number(strategy$global_random_effective_negative_sample_rate)
	  }
	  if (!is.null(strategy$hard_negative_enabled)) {
	    ctx$hard_negative_enabled <- safe_logical(strategy$hard_negative_enabled)
	  }
	  if (!is.null(strategy$hard_negative_rate)) {
	    ctx$hard_negative_rate <- safe_number(strategy$hard_negative_rate)
	  }
	  if (!is.null(strategy$hard_negative_weight_multiplier)) {
	    ctx$hard_negative_weight_multiplier <- safe_number(strategy$hard_negative_weight_multiplier)
	  }
	  if (!is.null(strategy$hard_negative_candidate_source)) {
	    ctx$hard_negative_candidate_source <- safe_character(strategy$hard_negative_candidate_source)
	  }
		  if (!is.null(strategy$hard_negative_transfer_mode)) {
		    ctx$hard_negative_transfer_mode <- safe_character(strategy$hard_negative_transfer_mode)
		  }
		  if (!is.null(strategy$semihard_triplet_enabled)) {
		    ctx$semihard_triplet_enabled <- safe_logical(strategy$semihard_triplet_enabled)
		  }
		  if (!is.null(strategy$semihard_triplet_rate)) {
		    ctx$semihard_triplet_rate <- safe_number(strategy$semihard_triplet_rate)
		  }
		  if (!is.null(strategy$semihard_triplet_weight_multiplier)) {
		    ctx$semihard_triplet_weight_multiplier <- safe_number(strategy$semihard_triplet_weight_multiplier)
		  }
		  if (!is.null(strategy$semihard_triplet_candidate_source)) {
		    ctx$semihard_triplet_candidate_source <- safe_character(strategy$semihard_triplet_candidate_source)
		  }
		  if (!is.null(strategy$semihard_triplet_transfer_mode)) {
		    ctx$semihard_triplet_transfer_mode <- safe_character(strategy$semihard_triplet_transfer_mode)
		  }
		  if (!is.null(strategy$triplet_mining_approximate)) {
		    ctx$triplet_mining_approximate <- safe_logical(strategy$triplet_mining_approximate)
		  }
		  if (!is.null(strategy$triplet_mining_graph_source)) {
		    ctx$triplet_mining_graph_source <- safe_character(strategy$triplet_mining_graph_source)
		  }
		  if (!is.null(strategy$triplet_mining_source_detail)) {
		    ctx$triplet_mining_source_detail <- safe_character(strategy$triplet_mining_source_detail)
		  }
		  if (!is.null(strategy$triplet_mining_candidate_source)) {
		    ctx$triplet_mining_candidate_source <- safe_character(strategy$triplet_mining_candidate_source)
		  }
		  if (!is.null(strategy$triplet_mining_transfer_mode)) {
		    ctx$triplet_mining_transfer_mode <- safe_character(strategy$triplet_mining_transfer_mode)
		  }
  if (!is.null(strategy$early_stop_criterion)) {
    ctx$early_stop_enabled <- TRUE
    ctx$early_stop_criterion <- safe_character(strategy$early_stop_criterion)
    ctx$early_stop_chunk_epochs <- safe_number(strategy$early_stop_chunk_epochs)
    ctx$early_stop_max_epochs <- safe_number(strategy$early_stop_max_epochs)
    ctx$early_stop_patience <- safe_number(strategy$early_stop_patience)
    ctx$early_stop_tolerance <- safe_number(strategy$early_stop_tolerance)
    if (identical(ctx$early_stop_criterion, "loss_change")) {
      ctx$early_stop_loss_available <- FALSE
      ctx$early_stop_loss_reason <- "native_optimizer_loss_callback_unavailable"
    }
  }
			  if (!is.null(strategy$output_metric)) {
	    ctx$output_metric <- safe_character(strategy$output_metric)
	    ctx$output_metric_transform <- safe_character(strategy$output_metric_transform)
    ctx$output_metric_native <- safe_logical(strategy$output_metric_native)
  }
  if (!strategy$compatible(method, backend)) {
    return(empty_row(ctx, strategy, "not_supported", "Approximation is not compatible with this method/backend combination."))
  }
  availability <- strategy_availability(strategy)
  if (!isTRUE(availability$available)) {
    status <- if (!is.null(strategy$unavailable_status)) strategy$unavailable_status else "not_supported"
    return(empty_row(ctx, strategy, status, availability$message))
  }
  context_availability <- strategy_context_available(strategy, ctx)
  if (!isTRUE(context_availability$available)) {
    return(empty_row(ctx, strategy, "not_supported", context_availability$message))
  }
  if (!backend_supported(backend)) {
    return(empty_row(ctx, strategy, "backend_unavailable", paste0("Backend `", backend, "` is unavailable or unsupported.")))
  }
  if (strategy_needs_knn(strategy)) {
    knn_attempt <- tryCatch({
      graph_key <- knn_cache_key(ctx, strategy)
      use_cache <- identical(knn_reuse_mode, "across_methods") && !is.null(knn_cache)
      cache_hit <- FALSE
      if (use_cache && exists(graph_key, envir = knn_cache, inherits = FALSE)) {
        record <- get(graph_key, envir = knn_cache, inherits = FALSE)
        cache_hit <- TRUE
      } else {
        record <- load_or_build_knn_record(dataset, ctx, strategy, backend, k, options, graph_key)
        if (use_cache) assign(graph_key, record, envir = knn_cache)
      }
      list(
        ok = TRUE,
        record = record,
        cache_hit = cache_hit,
        graph_key = graph_key,
        error = NA_character_
      )
    }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
    if (!isTRUE(knn_attempt$ok)) {
      return(empty_row(ctx, strategy, "failed", knn_attempt$error))
      }
      ctx <- apply_knn_record(ctx, knn_attempt$record, knn_reuse_mode, knn_attempt$cache_hit, knn_attempt$graph_key)
    }
    if (!is.null(strategy$transform_knn)) {
      graph_attempt <- tryCatch({
        transformed <- timed_step(strategy$transform_knn(ctx))
        list(
          ok = TRUE,
          ctx = apply_graph_transform(
            ctx,
            transformed$value,
            transformed$time,
            dataset,
            options$knn_quality_sample_size,
            seed
          ),
          error = NA_character_
        )
      }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
      if (!isTRUE(graph_attempt$ok)) {
        return(empty_row(ctx, strategy, "failed", graph_attempt$error))
      }
      ctx <- graph_attempt$ctx
    }
    tryCatch({
      measured <- measure_strategy(strategy$run(ctx), n, backend)
    layout_path <- paste0(layout_file(file.path(out_dir, "layouts"), ctx, strategy), ".rds")
    dir.create(dirname(layout_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(measured$layout, layout_path)
    metrics <- fastEmbedR::evaluate_embedding(
      dataset$x,
      measured$layout,
      labels = dataset$labels,
      k = metric_k(n),
      sample_size_for_global_metrics = min(as.integer(options$global_sample_size), n),
      sample_size_for_local_metrics = min(as.integer(options$local_sample_size), n),
      use_cache = FALSE,
      seed = seed,
      method = method,
      backend = backend,
      dataset = dataset$name
    )
    layout_cfg <- attr(measured$layout, "fastEmbedR_config")
    output_metric <- if (is.list(layout_cfg) && !is.null(layout_cfg$output_metric)) {
      safe_character(layout_cfg$output_metric)
    } else {
      safe_character(ctx$output_metric)
    }
    if (!is.na(output_metric) && nzchar(output_metric)) {
      output_metrics <- output_metric_global_metrics(
        dataset$x,
        measured$layout,
        metric = output_metric,
        sample_size = min(as.integer(options$global_sample_size), n),
        seed = seed
      )
      for (metric_name in names(output_metrics)) {
        metrics[[metric_name]] <- output_metrics[[metric_name]]
      }
    }
    success_row(ctx, strategy, measured, metrics, layout_path)
  }, error = function(e) {
    empty_row(ctx, strategy, "failed", conditionMessage(e))
  })
}

write_best_tables <- function(results, out_dir) {
  success <- results[results$status == "success", , drop = FALSE]
  if (nrow(success) == 0L) return(invisible(NULL))
  scale01 <- function(x, inverse = FALSE) {
    if (all(!is.finite(x))) return(rep(NA_real_, length(x)))
    rng <- range(x[is.finite(x)], na.rm = TRUE)
    if (diff(rng) == 0) out <- rep(1, length(x)) else out <- (x - rng[1L]) / diff(rng)
    if (inverse) out <- 1 - out
    out
  }
  success$combined_score <-
    0.25 * scale01(success$trustworthiness) +
    0.20 * scale01(success$knn_preservation_15) +
    0.15 * scale01(success$distance_spearman) +
    0.15 * scale01(success$label_knn_accuracy) +
    0.10 * scale01(success$neighbour_stability) +
    0.10 * scale01(success$total_with_knn_time_sec, inverse = TRUE) +
    0.05 * scale01(success$peak_ram_mb, inverse = TRUE)
  best <- success[order(success$dataset, success$method, -success$combined_score, success$total_with_knn_time_sec), , drop = FALSE]
  best <- do.call(rbind, lapply(split(best, paste(best$dataset, best$method, sep = "\r")), function(x) x[1L, , drop = FALSE]))
  utils::write.csv(best, file.path(out_dir, "best_by_dataset_method.csv"), row.names = FALSE)
  if ("density_spearman" %in% names(success)) {
    density_success <- success[is.finite(success$density_spearman), , drop = FALSE]
    if (nrow(density_success) > 0L) {
      density_best <- density_success[
        order(density_success$dataset, density_success$method, -density_success$density_spearman, density_success$density_log_radius_rmse),
        ,
        drop = FALSE
      ]
      density_best <- do.call(rbind, lapply(
        split(density_best, paste(density_best$dataset, density_best$method, sep = "\r")),
        function(x) x[1L, , drop = FALSE]
      ))
      utils::write.csv(density_best, file.path(out_dir, "best_by_density_preservation.csv"), row.names = FALSE)
    }
  }
  if ("output_metric_distance_spearman" %in% names(success)) {
    output_success <- success[
      is.finite(success$output_metric_distance_spearman) &
        !is.na(success$output_metric) &
        nzchar(success$output_metric),
      ,
      drop = FALSE
    ]
    if (nrow(output_success) > 0L) {
      output_best <- output_success[
        order(
          output_success$dataset,
          output_success$method,
          -output_success$output_metric_distance_spearman,
          output_success$output_metric_stress,
          output_success$total_with_knn_time_sec
        ),
        ,
        drop = FALSE
      ]
      output_best <- do.call(rbind, lapply(
        split(output_best, paste(output_best$dataset, output_best$method, sep = "\r")),
        function(x) x[1L, , drop = FALSE]
      ))
      utils::write.csv(output_best, file.path(out_dir, "best_by_output_metric.csv"), row.names = FALSE)
    }
  }
  invisible(best)
}

annoy_strategy_ids <- function() {
  vapply(annoy_strategy_grid(), function(strategy) strategy$id, character(1L))
}

hnsw_strategy_ids <- function() {
  vapply(hnsw_strategy_grid(), function(strategy) strategy$id, character(1L))
}

nndescent_strategy_ids <- function() {
  vapply(nndescent_strategy_grid(), function(strategy) strategy$id, character(1L))
}

faiss_strategy_ids <- function() {
  vapply(faiss_strategy_grid(), function(strategy) strategy$id, character(1L))
}

faiss_gpu_strategy_ids <- function() {
  vapply(faiss_gpu_strategy_grid(), function(strategy) strategy$id, character(1L))
}

cuml_strategy_ids <- function() {
  vapply(cuml_strategy_grid(), function(strategy) strategy$id, character(1L))
}

graph_strategy_ids <- function() {
  vapply(graph_strategy_grid(), function(strategy) strategy$id, character(1L))
}

snn_graph_strategy_ids <- function() {
  vapply(snn_graph_strategy_grid(), function(strategy) strategy$id, character(1L))
}

localmap_false_neighbor_strategy_ids <- function() {
  vapply(localmap_false_neighbor_strategy_grid(), function(strategy) strategy$id, character(1L))
}

localmap_local_weight_strategy_ids <- function() {
  vapply(localmap_local_weight_strategy_grid(), function(strategy) strategy$id, character(1L))
}

artificial_neighbor_penalty_strategy_ids <- function() {
  vapply(artificial_neighbor_penalty_strategy_grid(), function(strategy) strategy$id, character(1L))
}

false_neighbor_monitor_strategy_ids <- function() {
  vapply(false_neighbor_monitor_strategy_grid(), function(strategy) strategy$id, character(1L))
}

landmark_subsample_strategy_ids <- function() {
  vapply(landmark_subsample_strategy_grid(), function(strategy) strategy$id, character(1L))
}

random_landmark_strategy_ids <- function() {
  vapply(random_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

stratified_landmark_strategy_ids <- function() {
  vapply(stratified_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

density_weighted_landmark_strategy_ids <- function() {
  vapply(density_weighted_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

diversity_landmark_strategy_ids <- function() {
  vapply(diversity_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

hybrid_density_diversity_landmark_strategy_ids <- function() {
  vapply(hybrid_density_diversity_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

rare_protected_landmark_strategy_ids <- function() {
  vapply(rare_protected_landmark_strategy_grid(), function(strategy) strategy$id, character(1L))
}

landmark_projection_strategy_ids <- function() {
  vapply(landmark_projection_strategy_grid(), function(strategy) strategy$id, character(1L))
}

landmark_projection_refinement_strategy_ids <- function() {
  vapply(landmark_projection_refinement_strategy_grid(), function(strategy) strategy$id, character(1L))
}

landmark_affine_projection_strategy_ids <- function() {
  vapply(landmark_affine_projection_strategy_grid(), function(strategy) strategy$id, character(1L))
}

initialization_strategy_ids <- function() {
  vapply(initialization_strategy_grid(), function(strategy) strategy$id, character(1L))
}

warm_start_strategy_ids <- function() {
  vapply(warm_start_strategy_grid(), function(strategy) strategy$id, character(1L))
}

epoch_budget_strategy_ids <- function() {
  vapply(epoch_budget_strategy_grid(), function(strategy) strategy$id, character(1L))
}

early_stopping_strategy_ids <- function() {
  vapply(early_stopping_strategy_grid(), function(strategy) strategy$id, character(1L))
}

coarse_to_fine_strategy_ids <- function() {
  vapply(coarse_to_fine_strategy_grid(), function(strategy) strategy$id, character(1L))
}

adaptive_k_strategy_ids <- function() {
  vapply(adaptive_k_strategy_grid(), function(strategy) strategy$id, character(1L))
}

density_corrected_strategy_ids <- function() {
  vapply(density_corrected_strategy_grid(), function(strategy) strategy$id, character(1L))
}

umap_fuzzy_strategy_ids <- function() {
  vapply(umap_fuzzy_strategy_grid(), function(strategy) strategy$id, character(1L))
}

weighted_edge_sampling_strategy_ids <- function() {
  vapply(weighted_edge_sampling_strategy_grid(), function(strategy) strategy$id, character(1L))
}

sparse_fuzzy_graph_strategy_ids <- function() {
  vapply(sparse_fuzzy_graph_strategy_grid(), function(strategy) strategy$id, character(1L))
}

sparse_local_connectivity_strategy_ids <- function() {
  vapply(sparse_local_connectivity_strategy_grid(), function(strategy) strategy$id, character(1L))
}

umap_fuzzy_local_connectivity_strategy_ids <- function() {
  vapply(umap_fuzzy_local_connectivity_strategy_grid(), function(strategy) strategy$id, character(1L))
}

tsne_affinity_strategy_ids <- function() {
  vapply(tsne_affinity_strategy_grid(), function(strategy) strategy$id, character(1L))
}

multiscale_perplexity_strategy_ids <- function() {
  vapply(multiscale_perplexity_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pacmap_mid_near_strategy_ids <- function() {
  vapply(pacmap_mid_near_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pacmap_mid_near_emphasis_strategy_ids <- function() {
  vapply(pacmap_mid_near_emphasis_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pacmap_pair_separation_strategy_ids <- function() {
  vapply(pacmap_pair_separation_strategy_grid(), function(strategy) strategy$id, character(1L))
}

early_exaggeration_strategy_ids <- function() {
  vapply(early_exaggeration_strategy_grid(), function(strategy) strategy$id, character(1L))
}

late_exaggeration_strategy_ids <- function() {
  vapply(late_exaggeration_strategy_grid(), function(strategy) strategy$id, character(1L))
}

optimizer_schedule_strategy_ids <- function() {
  vapply(optimizer_schedule_strategy_grid(), function(strategy) strategy$id, character(1L))
}

learning_rate_scaling_strategy_ids <- function() {
  vapply(learning_rate_scaling_strategy_grid(), function(strategy) strategy$id, character(1L))
}

adaptive_learning_rate_strategy_ids <- function() {
  vapply(adaptive_learning_rate_strategy_grid(), function(strategy) strategy$id, character(1L))
}

mini_batch_strategy_ids <- function() {
  vapply(mini_batch_strategy_grid(), function(strategy) strategy$id, character(1L))
}

deterministic_batch_strategy_ids <- function() {
  vapply(deterministic_batch_strategy_grid(), function(strategy) strategy$id, character(1L))
}

sparse_edge_batch_strategy_ids <- function() {
  vapply(sparse_edge_batch_strategy_grid(), function(strategy) strategy$id, character(1L))
}

vectorized_edge_strategy_ids <- function() {
  vapply(vectorized_edge_strategy_grid(), function(strategy) strategy$id, character(1L))
}

atomic_sgd_strategy_ids <- function() {
  vapply(atomic_sgd_strategy_grid(), function(strategy) strategy$id, character(1L))
}

output_metric_strategy_ids <- function() {
  vapply(output_metric_strategy_grid(), function(strategy) strategy$id, character(1L))
}

umap_negative_sampling_strategy_ids <- function() {
  vapply(umap_negative_sampling_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pacmap_far_repulsion_strategy_ids <- function() {
  vapply(pacmap_far_repulsion_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pacmap_phase_strategy_ids <- function() {
  vapply(pacmap_phase_strategy_grid(), function(strategy) strategy$id, character(1L))
}

pair_resampling_strategy_ids <- function() {
  vapply(pair_resampling_strategy_grid(), function(strategy) strategy$id, character(1L))
}

triplet_aux_strategy_ids <- function() {
  vapply(triplet_aux_strategy_grid(), function(strategy) strategy$id, character(1L))
}

structured_triplet_strategy_ids <- function() {
  vapply(structured_triplet_strategy_grid(), function(strategy) strategy$id, character(1L))
}

global_random_triplet_strategy_ids <- function() {
  vapply(global_random_triplet_strategy_grid(), function(strategy) strategy$id, character(1L))
}

hard_negative_strategy_ids <- function() {
  vapply(hard_negative_strategy_grid(), function(strategy) strategy$id, character(1L))
}

semihard_triplet_strategy_ids <- function() {
  vapply(semihard_triplet_strategy_grid(), function(strategy) strategy$id, character(1L))
}

approx_triplet_mining_strategy_ids <- function() {
  vapply(approx_triplet_mining_strategy_grid(), function(strategy) strategy$id, character(1L))
}

trimap_triplet_proxy_strategy_ids <- function() {
  vapply(trimap_triplet_proxy_strategy_grid(), function(strategy) strategy$id, character(1L))
}

tsne_barnes_hut_strategy_ids <- function() {
  vapply(tsne_barnes_hut_strategy_grid(), function(strategy) strategy$id, character(1L))
}

fitsne_fft_experimental_strategy_ids <- function() {
  "fitsne_fft_experimental"
}

fitsne_fft_grid_strategy_ids <- function() {
  vapply(fitsne_fft_grid_strategy_grid(), function(strategy) strategy$id, character(1L))
}

distance_percentile_prune_strategy_ids <- function() {
  vapply(distance_percentile_prune_strategy_grid(), function(strategy) strategy$id, character(1L))
}

mst_rescue_strategy_ids <- function() {
  vapply(mst_rescue_strategy_grid(), function(strategy) strategy$id, character(1L))
}

effective_resistance_sparsify_strategy_ids <- function() {
  vapply(effective_resistance_sparsify_strategy_grid(), function(strategy) strategy$id, character(1L))
}

expand_strategy_ids <- function(ids) {
  annoy_markers <- c("annoy", "annoy_grid", "annoy_rcppannoy")
  hnsw_markers <- c("hnsw", "hnsw_grid", "hnsw_rcpphnsw")
  nndescent_markers <- c("nnd", "nndescent", "nndescent_grid", "rnndescent")
  faiss_markers <- c("faiss", "faiss_cpu", "faiss_grid")
  faiss_gpu_markers <- c("faiss_gpu", "faiss_gpu_grid", "faiss_cuda")
  cuml_markers <- c("cuml", "cuml_grid", "cuml_nn", "rapids_cuml")
  graph_markers <- c("graph", "graph_grid", "graph_construction", "graph_approximations")
  snn_markers <- c("snn", "snn_grid", "graph_snn", "snn_graph", "snn_graph_grid")
  localmap_false_neighbor_markers <- c(
    "localmap_false_neighbors", "localmap_false_neighbors_grid",
    "localmap_false_neighbours", "localmap_false_neighbours_grid",
    "false_neighbor_correction", "false_neighbor_correction_grid",
    "false_neighbour_correction", "false_neighbour_correction_grid",
    "localmap_false_neighbor_correction", "localmap_false_neighbour_correction",
    "false_neighbor_filtering", "false_neighbour_filtering"
  )
  localmap_local_weight_markers <- c(
    "localmap_local_weight", "localmap_local_weight_grid",
    "localmap_local_loss", "localmap_local_loss_grid",
    "local_loss_reweighting", "local_loss_reweighting_grid",
    "local_loss_weight", "local_loss_weight_grid",
    "local_relation_reweighting", "local_relation_reweighting_grid",
    "localmap_local_reweighting", "localmap_local_reweighting_grid"
  )
  artificial_neighbor_markers <- c(
    "artificial_neighbors", "artificial_neighbors_grid",
    "artificial_neighbours", "artificial_neighbours_grid",
    "artificial_neighbor_penalty", "artificial_neighbor_penalty_grid",
    "artificial_neighbour_penalty", "artificial_neighbour_penalty_grid",
    "false_neighbor_penalty", "false_neighbor_penalty_grid",
    "false_neighbour_penalty", "false_neighbour_penalty_grid",
    "post_embedding_false_neighbors", "post_embedding_false_neighbours",
    "localmap_artificial_neighbors", "localmap_artificial_neighbours"
  )
  false_neighbor_monitor_markers <- c(
    "false_neighbor_monitor", "false_neighbor_monitor_grid",
    "false_neighbour_monitor", "false_neighbour_monitor_grid",
    "optimization_monitor", "optimization_monitor_grid",
    "fnn_monitor", "fnn_monitor_grid",
    "false_neighbor_early_stop", "false_neighbour_early_stop",
    "localmap_monitor", "localmap_false_neighbor_monitor",
    "false_neighbor_rate_monitor", "false_neighbour_rate_monitor"
  )
  landmark_subsample_markers <- c(
    "landmark", "landmarks", "landmark_grid", "landmarking",
    "landmark_subsample", "landmark_subsample_grid",
    "subsample", "subsampling", "subsample_grid",
    "subsample_project", "subsample_projection",
    "random_subsample", "stratified_subsample",
    "landmark_and_subsampling", "landmark_subsampling"
  )
  random_landmark_markers <- c(
    "random_landmark", "random_landmarks", "random_landmarking",
    "random_landmark_grid", "random_landmarking_grid",
    "random_landmark_ratio", "random_landmark_ratios"
  )
  stratified_landmark_markers <- c(
    "stratified_landmark", "stratified_landmarks", "stratified_landmarking",
    "stratified_landmark_grid", "stratified_landmarking_grid",
    "label_stratified_landmark", "label_stratified_landmarks",
    "cluster_stratified_landmark", "cluster_stratified_landmarks",
    "unsupervised_stratified_landmark", "unsupervised_stratified_landmarks"
  )
  density_landmark_markers <- c(
    "density_landmark", "density_landmarks", "density_landmarking",
    "density_landmark_grid", "density_landmarking_grid",
    "density_weighted_landmark", "density_weighted_landmarks",
    "density_weighted_landmarking", "density_weighted_landmark_grid",
    "density_weighted_landmarking_grid", "density_landmark_alpha",
    "density_landmark_alpha_grid"
  )
  diversity_landmark_markers <- c(
    "diversity_landmark", "diversity_landmarks", "diversity_landmarking",
    "diversity_landmark_grid", "diversity_landmarking_grid",
    "farthest_point_landmark", "farthest_point_landmarks",
    "kmeanspp_landmark", "kmeansplusplus_landmark",
    "dpp_landmark", "dpp_landmarks", "dpp_approx_landmark",
    "landmark_diversity", "landmark_diversity_grid"
  )
  hybrid_density_diversity_markers <- c(
    "hybrid_density_diversity", "hybrid_density_diversity_landmark",
    "hybrid_density_diversity_landmarks", "hybrid_density_diversity_landmarking",
    "hybrid_density_diversity_grid", "hybrid_density_diversity_landmark_grid",
    "density_diversity_landmark", "density_diversity_landmarks",
    "density_diversity_landmarking", "density_diversity_grid",
    "hybrid_landmark", "hybrid_landmarks", "hybrid_landmarking"
  )
  rare_protected_landmark_markers <- c(
    "rare_protected", "rare_protected_landmark", "rare_protected_landmarks",
    "rare_protected_landmarking", "rare_cell_protected",
    "rare_cell_protected_landmark", "rare_cell_protected_landmarks",
    "rare_cell_protected_landmarking", "rare_cell_landmark",
    "rare_cell_landmarks", "rare_cell_landmarking",
    "rare_landmark", "rare_landmarks", "rare_landmarking"
  )
  landmark_projection_markers <- c(
    "landmark_projection", "landmark_projection_grid",
    "landmark_knn_projection", "landmark_knn_projection_grid",
    "knn_landmark_projection", "projection_interpolation",
    "projection_interpolation_grid", "landmark_weighted_projection",
    "landmark_weighted_projection_grid"
  )
  landmark_projection_refinement_markers <- c(
    "landmark_projection_refinement", "landmark_projection_refinement_grid",
    "landmark_project_refine", "landmark_project_refine_grid",
    "project_then_refine", "project_then_refine_grid",
    "landmark_refine_projection", "landmark_refine_projection_grid",
    "landmark_projection_followed_by_refinement"
  )
  landmark_affine_projection_markers <- c(
    "landmark_affine_projection", "landmark_affine_projection_grid",
    "local_affine_projection", "local_affine_projection_grid",
    "landmark_local_affine", "landmark_local_affine_projection",
    "affine_landmark_projection", "affine_projection",
    "affine_projection_grid"
  )
  initialization_markers <- c(
    "initialization", "initialization_grid",
    "init", "init_grid",
    "initialization_strategies", "initialization_strategy",
    "init_strategies", "init_strategy",
    "random_pca_spectral_init", "embedding_initialization"
  )
  random_initialization_markers <- c(
    "random_initialization", "random_initialization_baseline",
    "random_init", "random_init_baseline",
    "random_start", "random_start_baseline",
    "random_layout", "random_layout_init",
    "baseline_random_init"
  )
  spectral_initialization_markers <- c(
    "spectral_initialization", "spectral_initialization_grid",
    "spectral_init", "spectral_init_grid",
    "laplacian_initialization", "laplacian_initialization_grid",
    "graph_laplacian_init", "graph_laplacian_initialization"
  )
  fast_spectral_initialization_markers <- c(
    "fast_spectral_initialization", "fast_spectral_init",
    "approx_spectral_initialization", "approx_spectral_init",
    "block_power_spectral_init", "ritz_spectral_init"
  )
  randomized_spectral_initialization_markers <- c(
    "randomized_spectral_initialization", "randomized_spectral_initialization_grid",
    "randomized_spectral_init", "randomized_spectral_init_grid",
    "randomized_laplacian_initialization", "randomized_laplacian_init",
    "sparse_spectral_initialization", "sparse_spectral_init",
    "irlba_spectral", "irlba_spectral_init", "spectral_irlba",
    "rspectra_spectral", "rspectra_spectral_init", "spectral_rspectra",
    "nystrom_spectral", "nystrom_spectral_init", "spectral_nystrom",
    "nystrom_initialization", "nystrom_spectral_initialization"
  )
  irlba_spectral_initialization_markers <- c(
    "irlba_spectral_initialization", "irlba_spectral_init",
    "sparse_irlba_spectral", "sparse_irlba_spectral_init",
    "init_spectral_irlba_only"
  )
  rspectra_spectral_initialization_markers <- c(
    "rspectra_spectral_initialization", "rspectra_spectral_init",
    "sparse_rspectra_spectral", "sparse_rspectra_spectral_init",
    "init_spectral_rspectra_only"
  )
  nystrom_spectral_initialization_markers <- c(
    "nystrom_spectral_initialization", "nystrom_spectral_init",
    "nystrom_laplacian_initialization", "nystrom_laplacian_init",
    "init_spectral_nystrom_only"
  )
  diffusion_map_initialization_markers <- c(
    "diffusion_map_initialization", "diffusion_map_init",
    "diffusion_initialization", "diffusion_init",
    "diffusion_components", "diffusion_components_init",
    "diffusion_components_initialization",
    "diffusion_map", "diffusion_maps",
    "trajectory_initialization", "trajectory_init"
  )
  laplacian_eigenmaps_initialization_markers <- c(
    "laplacian_eigenmaps_initialization", "laplacian_eigenmaps_init",
    "laplacian_eigenmap_initialization", "laplacian_eigenmap_init",
    "laplacian_eigenmaps", "laplacian_eigenmap",
    "le_initialization", "le_init",
    "graph_laplacian_eigenmaps", "sparse_laplacian_eigenmaps"
  )
  initialization_comparison_markers <- c(
    "initialization_comparison", "initialization_compare",
    "compare_initialization", "compare_initializations",
    "pca_vs_spectral_vs_diffusion_vs_random",
    "random_vs_pca_vs_spectral_vs_diffusion",
    "random_pca_spectral_diffusion_laplacian",
    "random_pca_spectral_diffusion_laplacian_comparison",
    "laplacian_initialization_comparison"
  )
  exact_spectral_initialization_markers <- c(
    "exact_spectral_initialization", "exact_spectral_init",
    "dense_laplacian_initialization", "dense_laplacian_init",
    "dense_spectral_initialization", "dense_spectral_init"
  )
  warm_start_markers <- c(
    "warm_start", "warm_start_grid", "warm_start_previous_embedding",
    "warm_start_previous", "reuse_embedding", "reuse_previous_embedding",
    "previous_embedding", "interactive_warm_start",
    "parameter_grid_warm_start", "grid_warm_start"
  )
  coarse_to_fine_markers <- c(
    "coarse_to_fine", "coarse_to_fine_grid",
    "coarse_fine", "coarse_fine_grid",
    "coarse_to_fine_initialization", "coarse_to_fine_init",
    "subset_then_refine", "subset_project_refine",
    "low_resolution_graph", "low_resolution_graph_refine",
    "lowres_graph", "lowres_graph_refine"
  )
  adaptive_markers <- c("adaptive_k", "adaptive_k_grid", "graph_adaptive_k", "density_adaptive_k")
  density_markers <- c(
    "density_corrected", "density_corrected_grid", "graph_density",
    "density_weights", "densmap_like", "densmap", "density_preservation",
    "density_radius", "density_radius_preservation"
  )
  umap_fuzzy_markers <- c(
    "umap_fuzzy", "umap_fuzzy_grid", "umap_graph", "umap_graph_grid",
    "fuzzy_simplicial_set", "umap_set_op", "umap_set_op_mix",
    "set_op_mix", "set_op_mix_ratio", "umap_specific_graph"
  )
  local_connectivity_markers <- c(
    "local_connectivity", "local_connectivity_grid",
    "umap_local_connectivity", "umap_local_connectivity_grid",
    "graph_local_connectivity", "graph_local_connectivity_grid"
  )
  sparse_local_connectivity_markers <- c(
    "sparse_local_connectivity", "sparse_local_connectivity_grid",
    "csr_local_connectivity", "csr_local_connectivity_grid",
    "sparse_fuzzy_local_connectivity"
  )
  weighted_edge_sampling_markers <- c(
    "edge_sampling", "weighted_edge_sampling", "edge_sampling_weight",
    "graph_edge_sampling", "umap_edge_sampling", "weighted_edge_sample_grid"
  )
  sparse_fuzzy_graph_markers <- c(
    "sparse_fuzzy", "sparse_fuzzy_graph", "sparse_fuzzy_csr",
    "csr_fuzzy", "csr_fuzzy_graph", "sparse_graph", "graph_sparse_fuzzy",
    "sparse_set_op_mix", "sparse_set_op_mix_ratio"
  )
  tsne_affinity_markers <- c(
    "tsne_affinity", "tsne_affinity_grid", "perplexity_affinity",
    "early_exaggeration_transfer"
  )
  multiscale_perplexity_markers <- c(
    "multiscale_perplexity", "multiscale_perplexity_grid",
    "opentsne_multiscale", "perplexities_30_100",
    "perplexities_30_100_300", "multiscale_30_100",
    "multiscale_30_100_300"
  )
  pacmap_mid_near_markers <- c(
    "pacmap_mid_near", "pacmap_mid_near_grid", "mid_near",
    "mid_near_pairs", "pacmap_middle_near", "pacmap_aux_mid"
  )
  pacmap_mid_near_emphasis_markers <- c(
    "pacmap_mid_near_emphasis", "pacmap_mid_near_emphasis_grid",
    "mid_near_emphasis", "mid_near_pair_emphasis",
    "pacmap_mid_emphasis", "midrange_emphasis", "mid_range_emphasis"
  )
  pacmap_pair_separation_markers <- c(
    "pacmap_pair_separation", "pacmap_pair_separation_grid",
    "pacmap_pair_sep", "near_mid_far", "near_mid_far_pairs",
    "near_pair_mid_near_far_pair", "pair_categories",
    "near_ratio", "mid_ratio", "far_ratio", "near_mid_far_ratio",
    "near_ratio_grid", "mid_ratio_grid", "far_ratio_grid"
  )
  pacmap_far_markers <- c(
    "pacmap_far", "pacmap_far_grid", "pacmap_far_repulsion",
    "far_pairs", "far_pair_repulsion", "pacmap_aux_far"
  )
  pacmap_phase_markers <- c(
    "pacmap_phase", "pacmap_phase_grid", "pacmap_schedule",
    "pacmap_phase_schedule", "phase_schedule", "staged_optimization",
    "short_schedule", "default_schedule", "long_schedule",
    "pacmap_warmup", "pacmap_two_phase"
  )
  pair_resampling_markers <- c(
    "pair_resampling", "pair_resampling_grid", "pair_refresh",
    "refresh_pairs", "resample_pairs", "edge_resampling",
    "umap_edge_resampling", "triplet_resampling",
    "neighbour_resampling", "neighbor_resampling",
    "localmap_neighbour_resampling", "pacmap_pair_resampling"
  )
	  triplet_aux_markers <- c(
	    "triplet_constraints", "triplet_constraints_grid",
	    "triplet_regularization", "triplet_regularisation",
	    "triplet_aux", "triplet_aux_grid", "auxiliary_triplet_loss",
	    "aux_triplet_loss", "trimap_aux_loss", "trimap_triplet_loss"
	  )
		  structured_triplet_markers <- c(
		    "structured_triplets", "structured_triplets_grid",
		    "inlier_outlier_random_triplets", "inlier_outlier_random_triplets_grid",
		    "trimap_structured_triplets", "trimap_structured_triplets_grid",
		    "n_inliers_n_outliers_n_random", "triplet_family_budget",
		    "triplet_family_budget_grid"
		  )
			  global_random_triplet_markers <- c(
			    "global_random_triplets", "global_random_triplets_grid",
			    "global_random_triplet", "global_random_triplet_grid",
			    "random_long_range_triplets", "random_long_range_triplets_grid",
			    "long_range_triplets", "long_range_triplets_grid",
			    "random_outlier_triplets", "random_outlier_triplets_grid"
			  )
			  hard_negative_markers <- c(
			    "hard_negatives", "hard_negatives_grid",
			    "hard_negative_triplets", "hard_negative_triplets_grid",
			    "hard_negative_sampling", "hard_negative_sampling_grid",
			    "hard_outliers", "hard_outliers_grid",
			    "difficult_outliers", "difficult_outliers_grid",
			    "second_order_negatives", "near_outlier_negatives"
			  )
			  semihard_triplet_markers <- c(
			    "semihard_triplets", "semihard_triplets_grid",
			    "semi_hard_triplets", "semi_hard_triplets_grid",
			    "semihard_negatives", "semihard_negatives_grid",
			    "semi_hard_negatives", "semi_hard_negatives_grid",
			    "semihard_negative_sampling", "semihard_negative_sampling_grid",
			    "semi_hard_negative_sampling", "semi_hard_negative_sampling_grid",
			    "semi_hard_candidates", "semihard_candidates"
			  )
			  approx_triplet_mining_markers <- c(
			    "approx_triplet_mining", "approx_triplet_mining_grid",
			    "approximate_triplet_mining", "approximate_triplet_mining_grid",
			    "triplet_mining_approx_knn", "triplet_mining_approx_knn_grid",
			    "approx_knn_triplets", "approx_knn_triplets_grid",
			    "triplets_from_approx_knn", "approximate_knn_triplet_mining"
			  )
			  trimap_triplet_markers <- c(
	    "trimap_specific", "trimap_transfer", "trimap_specific_approximations",
	    "trimap_approximations", "trimap_transfer_grid",
	    "trimap_triplet_proxy", "trimap_triplet_proxy_grid",
	    "global_anchor_triplets", "trimap_global_anchors",
	    "triplet_candidate_graph", "triplet_candidate_proxy"
	  )
  pacmap_specific_markers <- c(
    "pacmap_specific", "pacmap_transfer", "pacmap_specific_approximations",
    "pacmap_approximations", "pacmap_transfer_grid"
  )
  early_exaggeration_markers <- c(
    "early_exaggeration", "early_exaggeration_grid", "early_exag",
    "exaggeration_grid", "edge_exaggeration", "near_pair_exaggeration",
    "inlier_exaggeration", "local_edge_exaggeration"
  )
  late_exaggeration_markers <- c(
    "late_exaggeration", "late_exaggeration_grid", "late_exag",
    "cluster_sharpening", "late_edge_exaggeration",
    "late_near_pair_exaggeration", "late_inlier_exaggeration",
    "late_local_edge_exaggeration"
  )
  optimizer_schedule_markers <- c(
    "momentum_schedule", "momentum", "optimizer_schedule",
    "optimizer_grid", "adam", "nesterov", "nesterov_momentum"
  )
  epoch_budget_markers <- c(
    "fewer_epochs", "fewer_epochs_grid", "reduced_epochs",
    "reduced_epochs_grid", "epoch_budget", "epoch_budget_grid",
    "epochs_grid", "optimization_budget_grid",
    "optimizer_budget", "optimizer_budget_grid",
    "epochs_50_100_200_500", "quality_speed_epochs"
  )
  early_stopping_markers <- c(
    "early_stopping", "early_stopping_grid",
    "early_stop", "early_stop_grid",
    "stopping", "stopping_grid",
    "embedding_stability_stop", "embedding_stability_plateau",
    "displacement_stop", "displacement_plateau",
    "trustworthiness_plateau", "trustworthiness_stop",
    "neighbour_stability_plateau", "neighbour_stability_stop",
    "neighbor_stability_plateau", "neighbor_stability_stop",
    "loss_change_stop", "loss_plateau", "optimizer_early_stop"
  )
  learning_rate_markers <- c(
    "learning_rate_scaling", "learning_rate_grid", "lr_scaling",
    "lr_grid", "auto_lr", "learning_rate_auto_scaling"
  )
  adaptive_lr_markers <- c(
    "adaptive_lr", "adaptive_lr_grid",
    "adaptive_learning_rate", "adaptive_learning_rate_grid",
    "lr_schedule", "lr_schedule_grid",
    "learning_rate_schedule", "learning_rate_schedule_grid",
    "constant_lr", "linear_decay_lr", "cosine_decay_lr",
    "step_decay_lr", "adam_lr", "adagrad_lr",
    "adam", "adagrad"
  )
  mini_batch_markers <- c(
    "mini_batch", "mini_batch_grid",
    "minibatch", "minibatch_grid",
    "mini_batch_optimization", "mini_batch_optimizer",
    "mini_batch_sgd", "batched_optimization",
    "edge_batching", "edge_batching_grid",
    "pair_batching", "triplet_batching",
    "gpu_friendly_batches", "batch_optimizer"
  )
  deterministic_batch_markers <- c(
    "deterministic_batch", "deterministic_batch_grid",
    "deterministic_batched", "deterministic_batched_grid",
    "deterministic_batched_optimization", "deterministic_batch_optimizer",
    "reproducible_batch", "reproducible_batch_grid",
    "no_atomic_batch", "no_atomic_batches",
    "fixed_order_batches", "fixed_row_batches"
  )
  sparse_edge_batch_markers <- c(
    "sparse_edge_batch", "sparse_edge_batch_grid",
    "sparse_edge_batches", "sparse_edge_batches_grid",
    "sparse_edge_batching", "sparse_edge_batching_grid",
    "streamed_csr_edges", "streamed_csr_edge_grid",
    "edge_chunks", "edge_chunks_grid",
    "affinity_chunks", "triplet_chunks",
    "low_ram_edge_batches", "low_memory_edge_batches"
  )
  vectorized_edge_markers <- c(
    "vectorized_edge", "vectorized_edge_grid",
    "vectorized_edges", "vectorized_edges_grid",
    "edge_vectorized", "edge_vectorized_grid",
    "simd_edge_optimizer", "simd_edge_optimizer_grid",
    "cpu_simd_edges", "cpu_simd_edges_grid",
    "edge_list_optimizer", "edge_list_optimizer_grid",
    "vectorized_edge_optimization", "edge_vectorization"
  )
  atomic_sgd_markers <- c(
    "atomic_sgd", "atomic_sgd_grid",
    "parallel_sgd_atomics", "parallel_sgd_atomics_grid",
    "atomic_edge_sgd", "atomic_edge_sgd_grid",
    "cpu_atomic_sgd", "cpu_atomic_sgd_grid",
    "cuda_atomics", "metal_atomics",
    "parallel_edge_updates", "lock_free_sgd"
  )
  output_metric_markers <- c(
    "output_metric", "output_metrics", "output_metric_grid",
    "umap_output_metric", "cosine_output", "hyperbolic_output",
    "cosine_like_output", "poincare_output"
  )
  umap_negative_markers <- c(
    "umap_negative_sampling", "umap_negative_sampling_grid",
    "negative_sampling", "negative_sample_rate", "umap_neg_sampling"
  )
  umap_specific_markers <- c(
    "umap_specific", "umap_transfer", "umap_specific_approximations",
    "umap_approximations"
  )
  tsne_bh_markers <- c("barnes_hut", "barnes_hut_grid", "tsne_barnes_hut", "bh_tsne", "rtsne_theta_grid")
  fitsne_fft_markers <- c(
    "fitsne_fft", "fft_interpolation", "fitsne_interpolation",
    "fitsne_fft_experimental", "fft_transfer", "fitsne_transfer"
  )
  fitsne_fft_grid_markers <- c(
    "fft_grid", "fitsne_grid", "interpolation_grid",
    "grid_coarseness", "fitsne_grid_coarseness", "fft_interpolation_grid"
  )
  distance_prune_markers <- c(
    "distance_prune", "distance_prune_grid", "graph_distance_prune",
    "graph_prune_distance", "distance_percentile_prune"
  )
  graph_sparsify_markers <- c(
    "graph_sparsify", "graph_sparsification", "spectral_sparsify",
    "effective_resistance", "effective_resistance_sparsify", "er_sparsify"
  )
  mst_rescue_markers <- c(
    "mst_rescue", "graph_mst_rescue", "minimum_spanning_tree_rescue",
    "prune_mst_rescue", "distance_prune_mst"
  )
  out <- ids
  if (any(out %in% annoy_markers)) {
    out <- unique(c(out[!out %in% annoy_markers], annoy_strategy_ids()))
  }
  if (any(out %in% hnsw_markers)) {
    out <- unique(c(out[!out %in% hnsw_markers], hnsw_strategy_ids()))
  }
  if (any(out %in% nndescent_markers)) {
    out <- unique(c(out[!out %in% nndescent_markers], nndescent_strategy_ids()))
  }
  if (any(out %in% faiss_markers)) {
    out <- unique(c(out[!out %in% faiss_markers], faiss_strategy_ids()))
  }
  if (any(out %in% faiss_gpu_markers)) {
    out <- unique(c(out[!out %in% faiss_gpu_markers], faiss_gpu_strategy_ids()))
  }
  if (any(out %in% cuml_markers)) {
    out <- unique(c(out[!out %in% cuml_markers], cuml_strategy_ids()))
  }
  if (any(out %in% graph_markers)) {
    out <- unique(c(out[!out %in% graph_markers], graph_strategy_ids()))
  }
  if (any(out %in% snn_markers)) {
    out <- unique(c(out[!out %in% snn_markers], snn_graph_strategy_ids()))
  }
  if (any(out %in% localmap_false_neighbor_markers)) {
    out <- unique(c(out[!out %in% localmap_false_neighbor_markers], localmap_false_neighbor_strategy_ids()))
  }
  if (any(out %in% localmap_local_weight_markers)) {
    out <- unique(c(out[!out %in% localmap_local_weight_markers], localmap_local_weight_strategy_ids()))
  }
  if (any(out %in% artificial_neighbor_markers)) {
    out <- unique(c(out[!out %in% artificial_neighbor_markers], artificial_neighbor_penalty_strategy_ids()))
  }
  if (any(out %in% false_neighbor_monitor_markers)) {
    out <- unique(c(out[!out %in% false_neighbor_monitor_markers], false_neighbor_monitor_strategy_ids()))
  }
  if (any(out %in% landmark_subsample_markers)) {
    out <- unique(c(out[!out %in% landmark_subsample_markers], c("landmark_fast", "landmark_balanced_refine", landmark_subsample_strategy_ids())))
  }
  if (any(out %in% random_landmark_markers)) {
    out <- unique(c(out[!out %in% random_landmark_markers], random_landmark_strategy_ids()))
  }
  if (any(out %in% stratified_landmark_markers)) {
    out <- unique(c(out[!out %in% stratified_landmark_markers], stratified_landmark_strategy_ids()))
  }
  if (any(out %in% density_landmark_markers)) {
    out <- unique(c(out[!out %in% density_landmark_markers], density_weighted_landmark_strategy_ids()))
  }
  if (any(out %in% diversity_landmark_markers)) {
    out <- unique(c(out[!out %in% diversity_landmark_markers], diversity_landmark_strategy_ids()))
  }
  if (any(out %in% hybrid_density_diversity_markers)) {
    out <- unique(c(out[!out %in% hybrid_density_diversity_markers], hybrid_density_diversity_landmark_strategy_ids()))
  }
  if (any(out %in% rare_protected_landmark_markers)) {
    out <- unique(c(out[!out %in% rare_protected_landmark_markers], rare_protected_landmark_strategy_ids()))
  }
  if (any(out %in% landmark_projection_markers)) {
    out <- unique(c(out[!out %in% landmark_projection_markers], landmark_projection_strategy_ids()))
  }
  if (any(out %in% landmark_projection_refinement_markers)) {
    out <- unique(c(out[!out %in% landmark_projection_refinement_markers], landmark_projection_refinement_strategy_ids()))
  }
  if (any(out %in% landmark_affine_projection_markers)) {
    out <- unique(c(out[!out %in% landmark_affine_projection_markers], landmark_affine_projection_strategy_ids()))
  }
  if (any(out %in% initialization_markers)) {
    out <- unique(c(out[!out %in% initialization_markers], initialization_strategy_ids()))
  }
  if (any(out %in% random_initialization_markers)) {
    out <- unique(c(out[!out %in% random_initialization_markers], "init_random"))
  }
  if (any(out %in% spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% spectral_initialization_markers], c("init_spectral", "init_spectral_exact")))
  }
  if (any(out %in% fast_spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% fast_spectral_initialization_markers], "init_spectral"))
  }
  if (any(out %in% randomized_spectral_initialization_markers)) {
    out <- unique(c(
      out[!out %in% randomized_spectral_initialization_markers],
      c("init_spectral_irlba", "init_spectral_rspectra", "init_spectral_nystrom")
    ))
  }
  if (any(out %in% irlba_spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% irlba_spectral_initialization_markers], "init_spectral_irlba"))
  }
  if (any(out %in% rspectra_spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% rspectra_spectral_initialization_markers], "init_spectral_rspectra"))
  }
  if (any(out %in% nystrom_spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% nystrom_spectral_initialization_markers], "init_spectral_nystrom"))
  }
  if (any(out %in% diffusion_map_initialization_markers)) {
    out <- unique(c(out[!out %in% diffusion_map_initialization_markers], "init_diffusion_map"))
  }
  if (any(out %in% laplacian_eigenmaps_initialization_markers)) {
    out <- unique(c(out[!out %in% laplacian_eigenmaps_initialization_markers], "init_laplacian_eigenmaps"))
  }
  if (any(out %in% initialization_comparison_markers)) {
    out <- unique(c(
      out[!out %in% initialization_comparison_markers],
      c("init_random", "init_pca", "init_spectral", "init_diffusion_map", "init_laplacian_eigenmaps")
    ))
  }
  if (any(out %in% exact_spectral_initialization_markers)) {
    out <- unique(c(out[!out %in% exact_spectral_initialization_markers], "init_spectral_exact"))
  }
  if (any(out %in% warm_start_markers)) {
    out <- unique(c(out[!out %in% warm_start_markers], warm_start_strategy_ids()))
  }
  if (any(out %in% coarse_to_fine_markers)) {
    out <- unique(c(out[!out %in% coarse_to_fine_markers], coarse_to_fine_strategy_ids()))
  }
  if (any(out %in% adaptive_markers)) {
    out <- unique(c(out[!out %in% adaptive_markers], adaptive_k_strategy_ids()))
  }
  if (any(out %in% density_markers)) {
    out <- unique(c(out[!out %in% density_markers], density_corrected_strategy_ids()))
  }
  if (any(out %in% umap_fuzzy_markers)) {
    out <- unique(c(out[!out %in% umap_fuzzy_markers], umap_fuzzy_strategy_ids()))
  }
  if (any(out %in% weighted_edge_sampling_markers)) {
    out <- unique(c(out[!out %in% weighted_edge_sampling_markers], weighted_edge_sampling_strategy_ids()))
  }
  if (any(out %in% sparse_fuzzy_graph_markers)) {
    out <- unique(c(out[!out %in% sparse_fuzzy_graph_markers], sparse_fuzzy_graph_strategy_ids()))
  }
  if (any(out %in% local_connectivity_markers)) {
    out <- unique(c(
      out[!out %in% local_connectivity_markers],
      umap_fuzzy_local_connectivity_strategy_ids(),
      sparse_local_connectivity_strategy_ids()
    ))
  }
  if (any(out %in% sparse_local_connectivity_markers)) {
    out <- unique(c(out[!out %in% sparse_local_connectivity_markers], sparse_local_connectivity_strategy_ids()))
  }
  if (any(out %in% tsne_affinity_markers)) {
    out <- unique(c(out[!out %in% tsne_affinity_markers], tsne_affinity_strategy_ids()))
  }
  if (any(out %in% multiscale_perplexity_markers)) {
    out <- unique(c(out[!out %in% multiscale_perplexity_markers], multiscale_perplexity_strategy_ids()))
  }
  if (any(out %in% pacmap_mid_near_markers)) {
    out <- unique(c(out[!out %in% pacmap_mid_near_markers], pacmap_mid_near_strategy_ids()))
  }
  if (any(out %in% pacmap_mid_near_emphasis_markers)) {
    out <- unique(c(out[!out %in% pacmap_mid_near_emphasis_markers], pacmap_mid_near_emphasis_strategy_ids()))
  }
  if (any(out %in% pacmap_pair_separation_markers)) {
    out <- unique(c(out[!out %in% pacmap_pair_separation_markers], pacmap_pair_separation_strategy_ids()))
  }
  if (any(out %in% pacmap_far_markers)) {
    out <- unique(c(out[!out %in% pacmap_far_markers], pacmap_far_repulsion_strategy_ids()))
  }
  if (any(out %in% pacmap_phase_markers)) {
    out <- unique(c(out[!out %in% pacmap_phase_markers], pacmap_phase_strategy_ids()))
  }
  if (any(out %in% pair_resampling_markers)) {
    out <- unique(c(out[!out %in% pair_resampling_markers], pair_resampling_strategy_ids()))
  }
	  if (any(out %in% triplet_aux_markers)) {
	    out <- unique(c(out[!out %in% triplet_aux_markers], triplet_aux_strategy_ids()))
	  }
	  if (any(out %in% structured_triplet_markers)) {
	    out <- unique(c(out[!out %in% structured_triplet_markers], structured_triplet_strategy_ids()))
	  }
		  if (any(out %in% global_random_triplet_markers)) {
		    out <- unique(c(out[!out %in% global_random_triplet_markers], global_random_triplet_strategy_ids()))
		  }
		  if (any(out %in% hard_negative_markers)) {
		    out <- unique(c(out[!out %in% hard_negative_markers], hard_negative_strategy_ids()))
		  }
			  if (any(out %in% semihard_triplet_markers)) {
			    out <- unique(c(out[!out %in% semihard_triplet_markers], semihard_triplet_strategy_ids()))
			  }
			  if (any(out %in% approx_triplet_mining_markers)) {
			    out <- unique(c(out[!out %in% approx_triplet_mining_markers], approx_triplet_mining_strategy_ids()))
			  }
			  if (any(out %in% trimap_triplet_markers)) {
	    out <- unique(c(out[!out %in% trimap_triplet_markers], trimap_triplet_proxy_strategy_ids()))
	  }
  if (any(out %in% pacmap_specific_markers)) {
    out <- unique(c(
      out[!out %in% pacmap_specific_markers],
      pacmap_mid_near_strategy_ids(),
      pacmap_mid_near_emphasis_strategy_ids(),
      pacmap_pair_separation_strategy_ids(),
      pacmap_far_repulsion_strategy_ids(),
      pacmap_phase_strategy_ids(),
      pair_resampling_strategy_ids()
    ))
  }
  if (any(out %in% early_exaggeration_markers)) {
    out <- unique(c(out[!out %in% early_exaggeration_markers], early_exaggeration_strategy_ids()))
  }
  if (any(out %in% late_exaggeration_markers)) {
    out <- unique(c(out[!out %in% late_exaggeration_markers], late_exaggeration_strategy_ids()))
  }
  if (any(out %in% optimizer_schedule_markers)) {
    out <- unique(c(out[!out %in% optimizer_schedule_markers], optimizer_schedule_strategy_ids()))
  }
  if (any(out %in% epoch_budget_markers)) {
    out <- unique(c(out[!out %in% epoch_budget_markers], epoch_budget_strategy_ids()))
  }
  if (any(out %in% early_stopping_markers)) {
    out <- unique(c(out[!out %in% early_stopping_markers], early_stopping_strategy_ids()))
  }
  if (any(out %in% learning_rate_markers)) {
    out <- unique(c(out[!out %in% learning_rate_markers], learning_rate_scaling_strategy_ids()))
  }
  if (any(out %in% adaptive_lr_markers)) {
    out <- unique(c(out[!out %in% adaptive_lr_markers], adaptive_learning_rate_strategy_ids()))
  }
  if (any(out %in% mini_batch_markers)) {
    out <- unique(c(out[!out %in% mini_batch_markers], mini_batch_strategy_ids()))
  }
  if (any(out %in% deterministic_batch_markers)) {
    out <- unique(c(out[!out %in% deterministic_batch_markers], deterministic_batch_strategy_ids()))
  }
  if (any(out %in% sparse_edge_batch_markers)) {
    out <- unique(c(out[!out %in% sparse_edge_batch_markers], sparse_edge_batch_strategy_ids()))
  }
  if (any(out %in% vectorized_edge_markers)) {
    out <- unique(c(out[!out %in% vectorized_edge_markers], vectorized_edge_strategy_ids()))
  }
  if (any(out %in% atomic_sgd_markers)) {
    out <- unique(c(out[!out %in% atomic_sgd_markers], atomic_sgd_strategy_ids()))
  }
  if (any(out %in% output_metric_markers)) {
    out <- unique(c(out[!out %in% output_metric_markers], output_metric_strategy_ids()))
  }
  if (any(out %in% umap_negative_markers)) {
    out <- unique(c(out[!out %in% umap_negative_markers], umap_negative_sampling_strategy_ids()))
  }
  if (any(out %in% umap_specific_markers)) {
    out <- unique(c(
      out[!out %in% umap_specific_markers],
      umap_fuzzy_strategy_ids(),
      sparse_fuzzy_graph_strategy_ids(),
      umap_fuzzy_local_connectivity_strategy_ids(),
      sparse_local_connectivity_strategy_ids(),
      weighted_edge_sampling_strategy_ids(),
      umap_negative_sampling_strategy_ids()
    ))
  }
  if (any(out %in% tsne_bh_markers)) {
    out <- unique(c(out[!out %in% tsne_bh_markers], tsne_barnes_hut_strategy_ids()))
  }
  if (any(out %in% fitsne_fft_markers)) {
    out <- unique(c(out[!out %in% fitsne_fft_markers], fitsne_fft_experimental_strategy_ids()))
  }
  if (any(out %in% fitsne_fft_grid_markers)) {
    out <- unique(c(out[!out %in% fitsne_fft_grid_markers], fitsne_fft_grid_strategy_ids()))
  }
  if (any(out %in% distance_prune_markers)) {
    out <- unique(c(out[!out %in% distance_prune_markers], distance_percentile_prune_strategy_ids()))
  }
  if (any(out %in% graph_sparsify_markers)) {
    out <- unique(c(out[!out %in% graph_sparsify_markers], effective_resistance_sparsify_strategy_ids()))
  }
  if (any(out %in% mst_rescue_markers)) {
    out <- unique(c(out[!out %in% mst_rescue_markers], mst_rescue_strategy_ids()))
  }
  out
}

datasets <- Filter(Negate(is.null), lapply(parse_csv_arg("datasets", "iris,synthetic"), load_named_dataset))
methods <- parse_csv_arg("methods", "umap,tsne,pacmap,trimap,localmap")
backends <- parse_csv_arg("backends", "cpu")
default_strategy_ids <- paste(c(
  "exact_knn",
  "kdtree_rnanoflann",
  "kdtree_fnn",
  "kdtree_sklearn",
  "balltree_sklearn",
  "brute_sklearn",
  annoy_strategy_ids(),
  hnsw_strategy_ids(),
  nndescent_strategy_ids(),
  faiss_strategy_ids(),
  "full_knn_auto",
  "short_epochs_knn",
  "pca_then_full",
  "landmark_fast",
  "landmark_balanced_refine",
  "tsne_rtsne_neighbors",
  "tsne_fitsne_fft"
), collapse = ",")
strategy_ids <- expand_strategy_ids(parse_csv_arg("approximations", default_strategy_ids))
seeds <- as.integer(parse_csv_arg("seeds", "4,5,6"))
k_values <- as.integer(parse_csv_arg("k", "15"))
knn_reuse_modes <- parse_knn_reuse_modes(parse_csv_arg("knn-reuse", "method_specific"))
out_dir <- parse_scalar("out-dir", file.path("results", "approximation_strategies"))
options <- list(
  pca_dims = as.integer(parse_scalar("pca-dims", "30")),
  knn_metric = normalize_knn_metric(parse_scalar("knn-metric", "euclidean")),
  short_epochs = as.integer(parse_scalar("short-epochs", "80")),
  landmarks = as.integer(parse_scalar("landmarks", "0")),
  global_sample_size = as.integer(parse_scalar("global-sample-size", "1000")),
  local_sample_size = as.integer(parse_scalar("local-sample-size", "1000")),
  knn_quality_sample_size = as.integer(parse_scalar("knn-quality-sample-size", "1000")),
  knn_disk_cache = as_logical_arg(parse_scalar("knn-disk-cache", "false")),
  knn_cache_dir = parse_scalar("knn-cache-dir", file.path(out_dir, "knn_cache")),
  knn_cache_format = normalize_knn_cache_format(parse_scalar("knn-cache-format", "rds")),
  knn_cache_force_recompute = as_logical_arg(parse_scalar("knn-cache-force-recompute", "false"))
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
strategies <- c(
  strategy_registry(),
  annoy_strategy_grid(),
  hnsw_strategy_grid(),
  nndescent_strategy_grid(),
  faiss_strategy_grid(),
  faiss_gpu_strategy_grid(),
  cuml_strategy_grid(),
  deterministic_batch_strategy_grid(),
  sparse_edge_batch_strategy_grid(),
  vectorized_edge_strategy_grid(),
  atomic_sgd_strategy_grid(),
  graph_strategy_grid()
)
strategies <- strategies[vapply(strategies, function(x) x$id %in% strategy_ids, logical(1))]
if (length(datasets) == 0L) stop("No datasets loaded.", call. = FALSE)
if (length(strategies) == 0L) stop("No approximation strategies selected.", call. = FALSE)

rows <- list()
for (knn_reuse_mode in knn_reuse_modes) {
  knn_cache <- new.env(parent = emptyenv())
  for (dataset in datasets) {
    for (method in methods) {
      for (strategy in strategies) {
        for (backend in backends) {
          for (k in k_values) {
            for (seed in seeds) {
              message("Running ", dataset$name, " / ", method, " / ", strategy$id, " / ", knn_reuse_mode, " / ", backend, " / seed ", seed)
              rows[[length(rows) + 1L]] <- run_one(
                dataset,
                method,
                strategy,
                backend,
                seed,
                k,
                out_dir,
                options,
                knn_reuse_mode = knn_reuse_mode,
                knn_cache = if (identical(knn_reuse_mode, "across_methods")) knn_cache else NULL
              )
            }
          }
        }
      }
    }
  }
}

results <- do.call(rbind, rows)
labels <- setNames(lapply(datasets, function(x) x$labels), vapply(datasets, function(x) x$name, character(1)))
results <- add_stability_metrics(results, labels)

stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- file.path(out_dir, paste0("approximation_strategy_results_", stamp, ".csv"))
latest_file <- file.path(out_dir, "latest_approximation_strategy_results.csv")
utils::write.csv(results, results_file, row.names = FALSE)
utils::write.csv(results, latest_file, row.names = FALSE)
best <- write_best_tables(results, out_dir)

summary_columns <- intersect(c(
  "dataset", "method", "approximation", "knn_reuse_mode", "knn_cache_hit",
  "knn_graph_source", "knn_disk_cache_hit", "knn_disk_load_time_sec",
  "backend", "seed", "status", "knn_time_sec", "knn_graph_time_sec",
  "gpu_transfer_policy", "gpu_transfer_host_to_device_count",
  "gpu_transfer_device_to_host_count", "gpu_transfer_host_to_device_mb",
  "gpu_transfer_device_to_host_mb", "gpu_transfer_knn_uploaded_once",
  "gpu_transfer_embedding_returned_only_at_end",
  "gpu_transfer_graph_metadata_roundtrip",
  "knn_recall_at_k", "knn_mean_distance_error", "knn_rank_correlation",
  "graph_approximation", "graph_approximation_time_sec", "graph_effective_k",
  "graph_edge_retention", "graph_recall_at_k", "graph_mean_distance_error",
  "graph_rank_correlation", "graph_mean_degree", "graph_isolated_fraction",
  "graph_padding_fraction", "graph_mean_jaccard", "graph_zero_jaccard_fraction",
  "graph_storage_format", "graph_sparse_nnz", "graph_sparse_internal_memory_mb",
  "graph_dense_knn_memory_mb", "graph_sparse_internal_memory_ratio",
  "graph_snn_k", "graph_snn_prune_threshold", "graph_mean_snn_weight",
  "graph_zero_snn_fraction",
  "localmap_false_neighbor_enabled", "localmap_false_neighbor_mode",
  "localmap_false_neighbor_transfer_mode",
  "localmap_false_neighbor_jaccard_threshold",
  "localmap_false_neighbor_distance_quantile",
  "localmap_false_neighbor_distance_multiplier",
  "localmap_false_neighbor_min_keep_fraction",
  "localmap_false_neighbor_min_keep_k",
  "localmap_false_neighbor_removed_edges_mean",
  "localmap_false_neighbor_removed_fraction",
  "localmap_false_neighbor_kept_degree_mean",
  "localmap_false_neighbor_kept_jaccard_mean",
  "localmap_false_neighbor_removed_jaccard_mean",
  "localmap_false_neighbor_kept_distance_ratio_mean",
  "localmap_false_neighbor_removed_distance_ratio_mean",
  "localmap_false_neighbor_threshold_mean",
  "localmap_local_weight_enabled", "localmap_local_weight",
  "localmap_local_weight_mode", "localmap_local_weight_transfer_mode",
  "localmap_local_weight_jaccard_blend",
  "localmap_local_weight_mean_trust",
  "localmap_local_weight_rank_component_mean",
  "localmap_local_weight_jaccard_component_mean",
  "localmap_local_weight_mean_multiplier",
  "localmap_local_weight_min_multiplier",
  "localmap_local_weight_max_multiplier",
  "localmap_local_weight_distance_scale_mean",
  "artificial_neighbor_penalty_enabled", "artificial_neighbor_transfer_mode",
  "artificial_neighbor_refinement_backend",
  "artificial_neighbor_penalty_strength",
  "artificial_neighbor_penalty_iterations",
  "artificial_neighbor_penalty_low_k",
  "artificial_neighbor_penalty_far_multiplier",
  "artificial_neighbor_penalty_target_distance",
  "artificial_neighbor_penalized_pairs",
  "artificial_neighbor_total_low_edges",
  "artificial_neighbor_false_rate_before",
  "artificial_neighbor_false_rate_after",
  "artificial_neighbor_false_rate_delta",
  "artificial_neighbor_far_rate_before",
  "artificial_neighbor_far_rate_after",
  "artificial_neighbor_far_rate_delta",
  "artificial_neighbor_mean_high_distance_ratio",
  "artificial_neighbor_mean_low_distance",
  "false_neighbor_monitor_enabled",
  "false_neighbor_monitor_transfer_mode",
  "false_neighbor_monitor_backend",
  "false_neighbor_monitor_action",
  "false_neighbor_monitor_start_mode",
  "false_neighbor_monitor_chunk_epochs",
  "false_neighbor_monitor_max_chunks",
  "false_neighbor_monitor_chunks_run",
  "false_neighbor_monitor_epochs_requested",
  "false_neighbor_monitor_epochs_completed",
  "false_neighbor_monitor_patience",
  "false_neighbor_monitor_tolerance",
  "false_neighbor_monitor_initial_false_rate",
  "false_neighbor_monitor_final_false_rate",
  "false_neighbor_monitor_best_false_rate",
  "false_neighbor_monitor_false_rate_delta",
  "false_neighbor_monitor_initial_far_rate",
  "false_neighbor_monitor_final_far_rate",
  "false_neighbor_monitor_best_far_rate",
  "false_neighbor_monitor_far_rate_delta",
  "false_neighbor_monitor_score_initial",
  "false_neighbor_monitor_score_final",
  "false_neighbor_monitor_score_best",
  "false_neighbor_monitor_score_delta",
  "false_neighbor_monitor_worsening_events",
  "false_neighbor_monitor_adjustments",
  "false_neighbor_monitor_stopped_early",
  "init_strategy", "init_backend", "init_time_sec",
  "init_optimizer_epochs", "init_optimizer_time_sec",
  "init_spectral_n_iter", "init_spectral_solver", "init_spectral_graph",
  "init_spectral_eigenvalues", "init_spectral_graph_nnz",
  "init_spectral_graph_active_fraction", "init_spectral_nystrom_landmarks",
  "init_spectral_nystrom_fraction", "init_spectral_nystrom_projection_k",
  "init_spectral_nystrom_weight", "init_spectral_nystrom_selection_used",
  "init_spectral_nystrom_landmark_knn_time_sec",
  "init_spectral_nystrom_landmark_spectral_time_sec",
  "init_spectral_nystrom_projection_time_sec",
  "init_diffusion_time", "init_diffusion_n_iter", "init_diffusion_solver",
  "init_diffusion_graph", "init_diffusion_eigenvalues",
  "init_diffusion_graph_nnz", "init_diffusion_graph_active_fraction",
  "init_laplacian_n_iter", "init_laplacian_solver",
  "init_laplacian_graph", "init_laplacian_eigenvalues",
  "init_laplacian_graph_nnz", "init_laplacian_graph_active_fraction",
  "init_laplacian_normalized_coordinates",
  "init_pca_method", "init_landmark_n",
  "init_landmark_fraction", "init_projection_k", "init_projection_weight",
  "warm_start_enabled", "warm_start_cache_hit", "warm_start_previous_init",
  "warm_start_previous_epochs", "warm_start_refinement_epochs",
  "warm_start_this_row_setup_time_sec", "warm_start_refinement_time_sec",
  "warm_start_total_if_cache_miss_sec", "warm_start_reuse_mode",
  "warm_start_bias_risk",
  "epoch_budget_enabled", "epoch_budget_requested",
  "epoch_budget_effective", "epoch_budget_default_epochs",
  "epoch_budget_ratio_to_default", "epoch_budget_quality",
  "epoch_budget_tsne_mode", "epoch_budget_optimizer_backend",
  "coarse_to_fine_enabled", "coarse_to_fine_mode", "coarse_to_fine_n",
  "coarse_to_fine_fraction", "coarse_to_fine_k",
  "coarse_to_fine_projection_k", "coarse_to_fine_coarse_epochs",
  "coarse_to_fine_refinement_epochs", "coarse_to_fine_setup_time_sec",
  "coarse_to_fine_refinement_time_sec",
  "coarse_to_fine_projection_zero_neighbor_fraction",
  "landmark_enabled", "landmark_approximation", "landmark_mode",
  "landmark_selection", "landmark_selection_requested",
  "landmark_selection_used", "landmark_count_requested",
  "landmark_fraction_requested", "landmark_n", "landmark_fraction",
  "landmark_label_classes_total", "landmark_label_classes_present",
  "landmark_label_missing_classes", "landmark_label_min_count",
  "landmark_label_min_fraction", "landmark_rare_label_count",
  "landmark_rare_label_present",
  "stratified_landmark_source", "stratified_landmark_allocation",
  "stratified_landmark_time_sec", "stratified_landmark_n_strata",
  "stratified_landmark_strata_sampled", "stratified_landmark_missing_strata",
  "stratified_landmark_min_stratum_size",
  "stratified_landmark_max_stratum_size",
  "stratified_landmark_min_selected_per_stratum",
  "stratified_landmark_max_selected_per_stratum",
  "stratified_landmark_balance_ratio", "stratified_landmark_cluster_k",
  "stratified_landmark_cluster_feature_dims",
  "density_landmark_alpha", "density_landmark_k",
  "density_landmark_time_sec", "density_landmark_selected_to_global_weight_ratio",
  "density_landmark_weight_median", "density_landmark_mean_distance_median",
  "hybrid_landmark_alpha", "hybrid_landmark_beta", "hybrid_landmark_time_sec",
  "hybrid_landmark_density_time_sec", "hybrid_landmark_cover_mean",
  "hybrid_landmark_cover_max", "hybrid_landmark_density_selected_to_global_weight_ratio",
  "rare_protected_tail_fraction", "rare_protected_tail_oversample",
  "rare_protected_n_quantiles", "rare_protected_cluster_fraction",
  "rare_protected_time_sec", "rare_protected_density_time_sec",
  "rare_protected_cluster_time_sec", "rare_protected_tail_count",
  "rare_protected_quantile_count", "rare_protected_cluster_count",
  "rare_protected_fill_count", "rare_protected_selected_tail_fraction",
  "rare_protected_selected_mean_distance_ratio",
  "rare_protected_selected_to_global_low_density_ratio",
  "rare_protected_quantile_min_selected", "rare_protected_quantile_max_selected",
  "rare_protected_cluster_k",
  "diversity_landmark_algorithm", "diversity_landmark_time_sec",
  "diversity_landmark_cover_mean", "diversity_landmark_cover_max",
  "diversity_landmark_leverage_selected_to_global_ratio",
  "landmark_projection_k", "landmark_interpolation",
  "landmark_projection_model",
  "landmark_projection_weight", "landmark_projection_bandwidth_rule",
  "landmark_projection_bandwidth_mean", "landmark_projection_weight_entropy",
  "landmark_projection_zero_neighbor_fraction", "landmark_projection_time_sec",
  "landmark_landmark_knn_time_sec", "landmark_landmark_embedding_time_sec",
  "landmark_affine_ridge", "landmark_affine_weight",
  "landmark_affine_rank_mean", "landmark_affine_condition_median",
  "landmark_affine_condition_max", "landmark_affine_fallback_fraction",
  "landmark_affine_clipped_fraction", "landmark_affine_clip_multiplier",
  "landmark_affine_blend",
  "landmark_refinement", "landmark_refinement_epochs",
  "landmark_refinement_time_sec", "landmark_projection_backend",
  "landmark_interpolation_backend", "landmark_refinement_backend",
  "landmark_refinement_knn_backend", "subsample_strategy",
  "subsample_stratified", "benchmark_forced_k", "benchmark_standardize",
  "graph_adaptive_mean_k", "graph_adaptive_density_cor",
  "graph_adaptive_dense_fraction", "graph_adaptive_sparse_fraction",
  "graph_distance_prune_drop_fraction", "graph_distance_prune_percentile",
  "graph_distance_prune_threshold_mean", "graph_distance_prune_removed_edges_mean",
  "graph_sparsification_method", "graph_sparsification_keep_fraction",
  "graph_sparsification_spectral_rank", "graph_sparsification_spectral_time_sec",
  "graph_sparsification_leverage_mean",
  "graph_mst_rescue_components_before", "graph_mst_rescue_components_after",
  "graph_mst_rescue_added_directed_edges", "graph_mst_rescue_mean_degree_after",
  "graph_density_correction_method", "graph_density_scale_cv",
  "graph_density_correction_mean", "graph_density_correction_clamp_fraction",
  "graph_density_corrected_distance_scale_cor",
  "umap_graph_set_op_mix_ratio", "umap_graph_local_connectivity",
  "umap_graph_weight_power",
  "umap_graph_target_scale", "umap_graph_distance_transform",
  "umap_graph_mean_weight", "graph_edge_sampling_fraction",
  "graph_edge_sampling_mean_selected_weight",
  "graph_edge_sampling_selected_to_candidate_weight_ratio",
  "pair_resampling_mode", "pair_resampling_pair_family",
  "pair_resampling_transfer_mode", "pair_resampling_refreshes",
  "pair_resampling_stage_epochs", "pair_resampling_keep_fraction",
  "pair_resampling_final_graph",
	  "triplet_aux_enabled", "triplet_aux_weight",
	  "triplet_aux_samples_per_edge", "triplet_aux_transfer_mode",
	  "triplet_aux_native_backend", "triplet_aux_n_epochs",
	  "triplet_structured_enabled", "triplet_structured_weight",
	  "triplet_structured_n_inliers", "triplet_structured_n_outliers",
	  "triplet_structured_n_random", "triplet_structured_total",
	  "triplet_structured_mode",
	  "global_random_triplet_enabled", "global_random_triplet_weight",
	  "global_random_triplets_per_point", "global_random_trimap_extra_negatives",
	  "global_random_negative_source", "global_random_transfer_mode",
	  "global_random_effective_negative_sample_rate",
	  "hard_negative_enabled", "hard_negative_rate",
	  "hard_negative_weight_multiplier", "hard_negative_candidate_source",
	  "hard_negative_transfer_mode",
	  "semihard_triplet_enabled", "semihard_triplet_rate",
	  "semihard_triplet_weight_multiplier", "semihard_triplet_candidate_source",
	  "semihard_triplet_transfer_mode",
	  "triplet_mining_approximate", "triplet_mining_graph_source",
	  "triplet_mining_source_detail", "triplet_mining_knn_backend",
	  "triplet_mining_candidate_source", "triplet_mining_transfer_mode",
	  "triplet_mining_recall_at_k", "triplet_mining_rank_correlation",
	  "triplet_mining_distance_error",
	  "umap_negative_sample_rate",
  "umap_transfer_mode",
  "graph_tsne_affinity_mode", "graph_tsne_affinity_perplexities",
  "graph_tsne_affinity_temperature", "graph_tsne_affinity_effective_perplexity_mean",
  "graph_multiscale_perplexities", "graph_multiscale_transfer_mode",
  "graph_multiscale_required_k", "graph_multiscale_effective_k_values",
  "pacmap_transfer_mode", "pacmap_auxiliary_pair_family",
  "pacmap_mid_near_pairs_per_point", "pacmap_mid_near_fraction",
  "pacmap_mid_near_requested_fraction", "pacmap_mid_near_distance_scale",
  "pacmap_mid_near_fallback_fraction",
  "pacmap_mid_near_emphasis_strength", "pacmap_mid_near_emphasis_distance_multiplier",
  "pacmap_near_ratio", "pacmap_mid_ratio", "pacmap_far_ratio",
  "pacmap_near_pairs_per_point", "pacmap_mid_pairs_per_point",
  "pacmap_far_pairs_per_point", "pacmap_far_pair_fraction",
  "pacmap_far_distance_scale", "pacmap_far_fallback_fraction",
  "pacmap_far_repulsion_rate",
  "pacmap_phase_schedule", "pacmap_phase_total_epochs",
  "pacmap_phase_epoch_multiplier", "pacmap_phase_warmup_epochs",
  "pacmap_phase_refine_epochs", "pacmap_phase_transfer_detail",
  "trimap_transfer_mode", "trimap_triplet_family",
  "trimap_inlier_ratio", "trimap_semihard_ratio",
  "trimap_global_anchor_ratio", "trimap_inlier_pairs_per_point",
  "trimap_semihard_pairs_per_point", "trimap_global_anchor_pairs_per_point",
  "trimap_semihard_fraction", "trimap_global_anchor_fraction",
  "trimap_semihard_distance_scale", "trimap_global_anchor_distance_scale",
  "trimap_semihard_fallback_fraction", "trimap_global_anchor_fallback_fraction",
  "trimap_native_explicit_triplets",
  "early_exaggeration_factor", "early_exaggeration_duration_fraction",
  "early_exaggeration_warmup_epochs", "early_exaggeration_transfer_mode",
  "late_exaggeration_factor", "late_exaggeration_duration_fraction",
  "late_exaggeration_late_epochs", "late_exaggeration_transfer_mode",
  "optimizer_mode", "optimizer_schedule", "optimizer_switch_iter",
  "optimizer_learning_rate", "optimizer_transfer_mode",
  "learning_rate_rule", "learning_rate_value", "learning_rate_base_default",
  "learning_rate_scale", "learning_rate_transfer_mode",
  "adaptive_lr_schedule", "adaptive_lr_total_epochs", "adaptive_lr_chunk_epochs",
  "adaptive_lr_chunks_run", "adaptive_lr_base_learning_rate",
  "adaptive_lr_final_learning_rate", "adaptive_lr_final_multiplier",
  "adaptive_lr_optimizer", "adaptive_lr_native", "adaptive_lr_chunked",
  "adaptive_lr_backend", "adaptive_lr_inner_decay", "adaptive_lr_note",
  "mini_batch_backend", "mini_batch_mode", "mini_batch_batch_fraction",
  "mini_batch_effective_k", "mini_batch_chunks", "mini_batch_chunk_epochs",
  "mini_batch_total_epochs", "mini_batch_refreshes", "mini_batch_sampling",
  "mini_batch_graph_time_sec", "mini_batch_init_time_sec",
  "mini_batch_optimizer_time_sec", "mini_batch_experimental",
  "deterministic_batch_row_batch_size", "deterministic_batch_chunks_per_epoch",
  "deterministic_batch_reduction", "deterministic_batch_atomic_updates",
  "deterministic_batch_reproducible_given_threads",
  "sparse_edge_batch_backend", "sparse_edge_batch_mode",
  "sparse_edge_batch_storage", "sparse_edge_batch_edge_batch_size",
  "sparse_edge_batch_chunks_per_epoch", "sparse_edge_batch_n_epochs",
  "sparse_edge_batch_negative_sample_rate", "sparse_edge_batch_learning_rate",
  "sparse_edge_batch_threads", "sparse_edge_batch_atomic_updates",
  "sparse_edge_batch_edge_list_copy", "sparse_edge_batch_triplet_chunks",
  "sparse_edge_batch_affinity_chunks", "sparse_edge_batch_aux_memory_mb",
  "sparse_edge_batch_graph_time_sec", "sparse_edge_batch_init_time_sec",
  "sparse_edge_batch_optimizer_time_sec", "sparse_edge_batch_status",
  "vectorized_edge_backend", "vectorized_edge_storage",
  "vectorized_edge_batch_size", "vectorized_edge_n_edges",
  "vectorized_edge_n_epochs", "vectorized_edge_negative_sample_rate",
  "vectorized_edge_threads", "vectorized_edge_learning_rate",
  "vectorized_edge_simd", "vectorized_edge_gpu_native",
  "vectorized_edge_graph_time_sec", "vectorized_edge_init_time_sec",
  "vectorized_edge_optimizer_time_sec", "vectorized_edge_status",
  "atomic_sgd_backend", "atomic_sgd_update_mode", "atomic_sgd_storage",
  "atomic_sgd_n_edges", "atomic_sgd_n_epochs",
  "atomic_sgd_negative_sample_rate", "atomic_sgd_threads",
  "atomic_sgd_learning_rate", "atomic_sgd_learning_rate_scale",
  "atomic_sgd_coordinate_clip", "atomic_sgd_openmp",
  "atomic_sgd_gpu_native", "atomic_sgd_nondeterministic",
  "atomic_sgd_graph_time_sec", "atomic_sgd_init_time_sec",
  "atomic_sgd_optimizer_time_sec", "atomic_sgd_status",
  "tsne_bh_theta", "tsne_bh_perplexity", "tsne_bh_n_epochs",
  "fft_interpolation_mode", "fft_interpolation_transfer_scope",
  "fft_interpolation_native_repulsive_field", "fft_interpolation_refine_epochs",
  "fft_grid_nterms", "fft_grid_intervals_per_integer", "fft_grid_min_num_intervals",
  "output_metric", "output_metric_transform", "output_metric_native",
  "output_metric_radius_mean", "output_metric_radius_max",
  "output_metric_distance_spearman", "output_metric_stress",
  "total_time_sec", "total_with_knn_time_sec", "peak_ram_mb", "trustworthiness", "continuity",
  "knn_preservation_15", "distance_spearman", "density_spearman",
  "density_log_radius_rmse", "silhouette",
  "label_knn_accuracy", "rare_class_recall", "procrustes_rmsd",
  "neighbour_stability", "cluster_stability_ari", "error_message"
), names(results))
print(results[, summary_columns, drop = FALSE], row.names = FALSE)
cat("\nSaved:\n")
cat("  ", normalizePath(results_file), "\n", sep = "")
cat("  ", normalizePath(latest_file), "\n", sep = "")
if (!is.null(best)) cat("  ", normalizePath(file.path(out_dir, "best_by_dataset_method.csv")), "\n", sep = "")
