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
  max_index <- 0.5 * (row_comb + col_comb)
  denom <- max_index - expected
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

embedding_clusters <- function(embedding, labels = NULL, seed = 4L) {
  if (is.null(labels)) return(rep(NA_integer_, nrow(embedding)))
  labels <- as.factor(labels)
  n_clusters <- length(levels(labels))
  if (n_clusters < 2L || n_clusters >= nrow(embedding)) return(rep(NA_integer_, nrow(embedding)))
  set.seed(seed)
  out <- tryCatch(
    stats::kmeans(embedding, centers = n_clusters, nstart = 5L, iter.max = 50L)$cluster,
    error = function(e) rep(NA_integer_, nrow(embedding))
  )
  as.integer(out)
}

majority_vote <- function(values) {
  tab <- table(values)
  names(tab)[which.max(tab)]
}

classification_from_embedding_nn <- function(embed_indices, labels, k) {
  labels <- as.factor(labels)
  k <- min(as.integer(k), ncol(embed_indices))
  pred <- majority_vote_knn_labels_cpp(
    embed_indices,
    as.integer(labels),
    as.integer(k),
    as.integer(length(levels(labels)))
  )
  factor(levels(labels)[pred], levels = levels(labels))
}

class_recall_metrics <- function(truth, pred) {
  truth <- as.factor(truth)
  pred <- factor(pred, levels = levels(truth))
  levels_truth <- levels(truth)
  recall <- vapply(levels_truth, function(level) {
    keep <- truth == level
    if (!any(keep)) return(NA_real_)
    mean(pred[keep] == level, na.rm = TRUE)
  }, numeric(1))
  counts <- as.integer(table(truth)[levels_truth])
  rare_cutoff <- stats::quantile(counts, probs = 0.25, type = 1, na.rm = TRUE)
  rare <- counts <= rare_cutoff
  list(
    table = data.frame(label = levels_truth, n = counts, recall = unname(recall), stringsAsFactors = FALSE),
    rare_class_recall = if (any(rare, na.rm = TRUE)) mean(recall[rare], na.rm = TRUE) else NA_real_
  )
}

class_recall_json <- function(recall_table) {
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(as.character(jsonlite::toJSON(recall_table, auto_unbox = TRUE, dataframe = "rows", null = "null")))
  }
  paste(paste(recall_table$label, recall_table$recall, sep = ":"), collapse = ";")
}

batch_entropy_metrics <- function(embed_indices, batch, k) {
  if (is.null(batch)) {
    return(list(batch_entropy = NA_real_, batch_mixing_score = NA_real_))
  }
  batch <- as.factor(batch)
  if (length(levels(batch)) < 2L) {
    return(list(batch_entropy = NA_real_, batch_mixing_score = NA_real_))
  }
  k <- min(as.integer(k), ncol(embed_indices))
  out <- batch_entropy_cpp(
    embed_indices,
    as.integer(batch),
    as.integer(k),
    as.integer(length(levels(batch)))
  )
  list(
    batch_entropy = unname(out["batch_entropy"]),
    batch_mixing_score = unname(out["batch_mixing_score"])
  )
}

centroid_distance_correlation <- function(x_high, embedding, labels) {
  if (is.null(labels)) return(NA_real_)
  labels <- as.factor(labels)
  if (length(levels(labels)) < 3L) return(NA_real_)
  high_centers <- do.call(rbind, lapply(levels(labels), function(level) colMeans(x_high[labels == level, , drop = FALSE])))
  low_centers <- do.call(rbind, lapply(levels(labels), function(level) colMeans(embedding[labels == level, , drop = FALSE])))
  safe_numeric_cor(stats::dist(high_centers), stats::dist(low_centers), method = "pearson")
}

mean_neighbor_rank_error <- function(high_indices, embed_indices, k) {
  k <- min(as.integer(k), ncol(high_indices), ncol(embed_indices))
  errs <- numeric(nrow(high_indices) * k)
  pos <- 1L
  for (i in seq_len(nrow(high_indices))) {
    ranks <- seq_len(k)
    names(ranks) <- as.character(high_indices[i, seq_len(k)])
    emb <- as.character(embed_indices[i, seq_len(k)])
    high_rank <- unname(ranks[emb])
    high_rank[is.na(high_rank)] <- k + 1L
    errs[pos:(pos + k - 1L)] <- abs(high_rank - seq_len(k))
    pos <- pos + k
  }
  mean(errs, na.rm = TRUE)
}

finite_sample_size <- function(sample_size, n) {
  if (is.null(sample_size)) return(n)
  sample_size <- as.integer(sample_size)
  if (length(sample_size) != 1L || is.na(sample_size) || sample_size < 1L) return(n)
  min(sample_size, n)
}

evaluation_reference_cache_path <- function(cache_dir,
                                            dataset,
                                            n,
                                            p,
                                            max_k,
                                            backend = "cpu") {
  dataset <- as.character(dataset)
  if (length(dataset) != 1L || is.na(dataset) || !nzchar(dataset)) dataset <- "dataset"
  cache_file(cache_dir, "eval_nn", dataset, n, p, paste0("k", max_k, "_euclidean_", backend))
}

normalize_evaluation_reference <- function(reference_nn, n, max_k) {
  out <- normalize_supplied_knn(reference_nn, n, max_k)
  out$backend <- if (is.null(reference_nn$backend)) {
    attr(reference_nn, "backend")
  } else {
    reference_nn$backend
  }
  if (is.null(out$backend) || length(out$backend) == 0L || is.na(out$backend)) {
    out$backend <- "precomputed"
  }
  out$cache_hit <- isTRUE(reference_nn$cache_hit)
  out$cache_path <- if (is.null(reference_nn$cache_path)) NA_character_ else as.character(reference_nn$cache_path)
  out
}

get_or_compute_evaluation_reference <- function(x_high,
                                                max_k,
                                                dataset = "dataset",
                                                use_cache = TRUE,
                                                cache_dir = file.path("results", "cache"),
                                                force_recompute = FALSE,
                                                backend = "cpu",
                                                n_threads = NULL) {
  n <- nrow(x_high)
  max_k <- min(as.integer(max_k), n - 1L)
  if (length(max_k) != 1L || is.na(max_k) || max_k < 1L) {
    stop("`max_k` must be a positive integer smaller than `nrow(x_high)`.", call. = FALSE)
  }
  cache_path <- evaluation_reference_cache_path(cache_dir, dataset, n, ncol(x_high), max_k, backend = backend)
  if (isTRUE(use_cache) && !isTRUE(force_recompute) && file.exists(cache_path)) {
    cached <- readRDS(cache_path)
    cached$cache_hit <- TRUE
    return(normalize_evaluation_reference(cached, n, max_k))
  }
  raw <- fastEmbedR::nn(
    x_high,
    x_high,
    k = max_k + 1L,
    backend = backend,
    n_threads = n_threads
  )
  out <- normalize_supplied_knn(raw, n, max_k)
  out$backend <- attr(raw, "backend")
  out$cache_hit <- FALSE
  out$cache_path <- cache_path
  if (isTRUE(use_cache)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(out, cache_path)
  }
  out
}

named_metric_or_na <- function(x, name) {
  if (!name %in% names(x)) NA_real_ else unname(x[[name]])
}

append_metric_backend_reason <- function(current, message) {
  values <- c(current, message)
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) == 0L) NA_character_ else paste(values, collapse = "; ")
}

sampled_pair_distances <- function(x, a, b, n_threads = NULL) {
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  sampled_pair_distances_cpp(
    x,
    as.integer(a),
    as.integer(b),
    as.integer(max(1L, n_threads))
  )
}

global_distance_metrics <- function(x_high, embedding, sample_size, seed, n_threads = NULL) {
  n <- nrow(x_high)
  if (n < 3L) {
    return(list(
      distance_spearman = NA_real_,
      distance_pearson = NA_real_,
      stress = NA_real_,
      global_sample_size = n,
      global_pair_count = 0L
    ))
  }
  sample_size <- min(as.integer(sample_size), n)
  set.seed(seed)
  keep <- if (sample_size < n) sort(sample.int(n, sample_size)) else seq_len(n)
  pair_total <- length(keep) * (length(keep) - 1L) / 2
  max_pairs <- min(pair_total, 250000L)
  if (pair_total <= max_pairs) {
    high_dist <- as.numeric(stats::dist(x_high[keep, , drop = FALSE]))
    low_dist <- as.numeric(stats::dist(embedding[keep, , drop = FALSE]))
  } else {
    set.seed(seed + 104729L)
    m <- as.integer(max_pairs)
    a <- sample.int(length(keep), m, replace = TRUE)
    b <- sample.int(length(keep) - 1L, m, replace = TRUE)
    b <- b + as.integer(b >= a)
    x_sample <- x_high[keep, , drop = FALSE]
    embedding_sample <- embedding[keep, , drop = FALSE]
    high_dist <- sampled_pair_distances(x_sample, a, b, n_threads = n_threads)
    low_dist <- sampled_pair_distances(embedding_sample, a, b, n_threads = n_threads)
  }
  list(
    distance_spearman = safe_numeric_cor(high_dist, low_dist, method = "spearman"),
    distance_pearson = safe_numeric_cor(high_dist, low_dist, method = "pearson"),
    stress = normalized_stress(high_dist, low_dist),
    global_sample_size = length(keep),
    global_pair_count = length(high_dist)
  )
}

standardize_log_radius <- function(x) {
  center <- stats::median(x, na.rm = TRUE)
  scale <- stats::mad(x, center = center, constant = 1, na.rm = TRUE)
  if (!is.finite(scale) || scale <= 0) {
    scale <- stats::sd(x, na.rm = TRUE)
  }
  if (!is.finite(scale) || scale <= 0) {
    return(rep(0, length(x)))
  }
  (x - center) / scale
}

local_density_radius_metrics <- function(high_distances,
                                         embedding_distances,
                                         k,
                                         keep = NULL) {
  k <- min(as.integer(k), ncol(high_distances), ncol(embedding_distances))
  if (length(k) != 1L || is.na(k) || k < 1L) {
    return(list(
      density_spearman = NA_real_,
      density_pearson = NA_real_,
      density_log_radius_rmse = NA_real_,
      density_radius_high_mean = NA_real_,
      density_radius_embedding_mean = NA_real_,
      density_sample_size = 0L
    ))
  }
  high_radius <- as.numeric(high_distances[, k])
  embedding_radius <- as.numeric(embedding_distances[, k])
  if (!is.null(keep) && length(keep) > 0L) {
    high_radius <- high_radius[keep]
    embedding_radius <- embedding_radius[keep]
  }
  ok <- is.finite(high_radius) & is.finite(embedding_radius) &
    high_radius >= 0 & embedding_radius >= 0
  if (sum(ok) < 3L) {
    return(list(
      density_spearman = NA_real_,
      density_pearson = NA_real_,
      density_log_radius_rmse = NA_real_,
      density_radius_high_mean = mean(high_radius[ok], na.rm = TRUE),
      density_radius_embedding_mean = mean(embedding_radius[ok], na.rm = TRUE),
      density_sample_size = sum(ok)
    ))
  }
  positive <- c(high_radius[ok & high_radius > 0], embedding_radius[ok & embedding_radius > 0])
  eps <- if (length(positive) == 0L) .Machine$double.eps else min(positive, na.rm = TRUE) * 1e-6
  if (!is.finite(eps) || eps <= 0) eps <- .Machine$double.eps
  high_log_radius <- log(pmax(high_radius[ok], eps))
  embedding_log_radius <- log(pmax(embedding_radius[ok], eps))
  high_z <- standardize_log_radius(high_log_radius)
  embedding_z <- standardize_log_radius(embedding_log_radius)
  list(
    density_spearman = safe_numeric_cor(high_log_radius, embedding_log_radius, method = "spearman"),
    density_pearson = safe_numeric_cor(high_log_radius, embedding_log_radius, method = "pearson"),
    density_log_radius_rmse = sqrt(mean((high_z - embedding_z)^2, na.rm = TRUE)),
    density_radius_high_mean = mean(high_radius[ok], na.rm = TRUE),
    density_radius_embedding_mean = mean(embedding_radius[ok], na.rm = TRUE),
    density_sample_size = sum(ok)
  )
}

#' Evaluate an embedding against high-dimensional structure
#'
#' @param x_high High-dimensional input, usually scaled data or PCA scores.
#' @param embedding Low-dimensional embedding matrix.
#' @param labels Optional biological/class labels.
#' @param batch Optional batch labels.
#' @param k Neighbor sizes used for structure metrics.
#' @param primary_k Neighbor size used for scalar trustworthiness, continuity,
#'   label accuracy, rare-class recall, density summaries, and rank-error
#'   metrics. The default `NULL` preserves the historical k = 15 behavior when
#'   possible.
#' @param reference_nn Optional precomputed high-dimensional nearest-neighbor
#'   list. It may include a self-neighbor column, like \code{nn(x, x, k + 1)}.
#' @param sample_size_for_global_metrics Maximum deterministic subsample size
#'   for all-pairs global distance metrics.
#' @param sample_size_for_local_metrics Maximum deterministic subsample size for
#'   local trustworthiness, continuity, and neighbour preservation metrics.
#' @param use_cache Cache the high-dimensional reference neighbours used by
#'   quality metrics.
#' @param cache_dir Directory for quality-metric reference-neighbour cache files.
#' @param force_recompute Ignore cached quality reference neighbours.
#' @param seed Random seed recorded in the output and used for subsampling.
#' @param method Method name recorded in the output.
#' @param backend Backend name recorded in the output.
#' @param n_threads Number of CPU worker threads used when quality metrics need
#'   nearest-neighbor searches. Ignored by native GPU metric backends.
#' @param dataset Dataset name recorded in the output.
#' @return A one-row data frame with local, global, label-aware, batch-aware,
#'   and metadata columns. Per-class recall is also attached as an attribute.
#' @export
evaluate_embedding <- function(x_high,
                               embedding,
                               labels = NULL,
                               batch = NULL,
                               k = c(15L, 30L, 50L),
                               primary_k = NULL,
                               reference_nn = NULL,
                               sample_size_for_global_metrics = min(5000L, nrow(x_high)),
                               sample_size_for_local_metrics = min(5000L, nrow(x_high)),
                               use_cache = FALSE,
                               cache_dir = file.path("results", "cache"),
                               force_recompute = FALSE,
                               seed = NA_integer_,
                               method = NA_character_,
                               backend = NA_character_,
                               n_threads = NULL,
                               dataset = NA_character_) {
  x_high <- as.matrix(x_high)
  embedding <- as.matrix(embedding)
  if (nrow(x_high) != nrow(embedding)) stop("`x_high` and `embedding` must have the same row count.", call. = FALSE)
  if (nrow(x_high) < 2L) stop("`x_high` must contain at least two rows.", call. = FALSE)
  if (!is.null(labels) && length(labels) != nrow(x_high)) stop("`labels` must have one entry per row.", call. = FALSE)
  if (!is.null(batch) && length(batch) != nrow(x_high)) stop("`batch` must have one entry per row.", call. = FALSE)

  requested_k <- as.integer(k)
  requested_k <- requested_k[is.finite(requested_k) & requested_k > 0L]
  if (length(requested_k) == 0L) requested_k <- 15L
  if (!is.null(primary_k)) {
    primary_k <- as.integer(primary_k[[1L]])
    if (is.na(primary_k) || !is.finite(primary_k) || primary_k < 1L) {
      stop("`primary_k` must be NULL or a positive integer.", call. = FALSE)
    }
    requested_k <- unique(c(requested_k, primary_k))
  }
  eval_k <- pmin(requested_k, nrow(x_high) - 1L)
  names(eval_k) <- paste0("knn_preservation_", requested_k)
  max_k <- max(eval_k)
  primary_k <- if (is.null(primary_k)) {
    min(15L, max_k)
  } else {
    min(as.integer(primary_k), max_k)
  }
  n_threads <- normalize_nn_threads(n_threads)
  metric_resolution <- resolve_metric_backend(backend)
  metric_backend <- metric_resolution$backend
  metric_backend_reason <- metric_resolution$reason

  high_nn <- if (is.null(reference_nn)) {
    tryCatch(
      get_or_compute_evaluation_reference(
        x_high,
        max_k = max_k,
        dataset = dataset,
        use_cache = use_cache,
        cache_dir = cache_dir,
        force_recompute = force_recompute,
        backend = metric_backend,
        n_threads = n_threads
      ),
      error = function(e) {
        metric_backend_reason <<- append_metric_backend_reason(
          metric_backend_reason,
          conditionMessage(e)
        )
        metric_backend <<- "cpu"
        get_or_compute_evaluation_reference(
          x_high,
          max_k = max_k,
          dataset = dataset,
          use_cache = use_cache,
          cache_dir = cache_dir,
          force_recompute = force_recompute,
          backend = "cpu",
          n_threads = n_threads
        )
      }
    )
  } else {
    normalize_evaluation_reference(reference_nn, nrow(x_high), max_k)
  }
  embed_nn <- tryCatch(
    fastEmbedR::nn(
      embedding,
      embedding,
      max_k + 1L,
      backend = metric_backend,
      n_threads = n_threads
    ),
    error = function(e) {
      metric_backend_reason <<- append_metric_backend_reason(
        metric_backend_reason,
        conditionMessage(e)
      )
      metric_backend <<- "cpu"
      fastEmbedR::nn(
        embedding,
        embedding,
        max_k + 1L,
        backend = "cpu",
        n_threads = n_threads
      )
    }
  )
  high_indices <- high_nn$indices
  embed_indices <- embed_nn$indices[, -1L, drop = FALSE]
  embed_distances <- embed_nn$distances[, -1L, drop = FALSE]
  local_sample_size <- finite_sample_size(sample_size_for_local_metrics, nrow(embedding))
  local_keep <- sample_indices(nrow(embedding), local_sample_size, if (is.na(seed)) 4L else seed)

  labels_factor <- if (is.null(labels)) NULL else as.factor(labels)
  labels_int <- if (is.null(labels_factor)) integer(0L) else as.integer(labels_factor)
  n_label_levels <- if (is.null(labels_factor)) 0L else length(levels(labels_factor))

  structure_by_k <- lapply(eval_k, function(kk) {
    structure_score_with_backend(
      embedding,
      high_indices[, seq_len(kk), drop = FALSE],
      as.integer(local_keep),
      as.integer(kk),
      labels_int,
      as.integer(n_label_levels),
      backend = metric_backend
    )$values
  })
  primary_idx <- which.min(abs(eval_k - primary_k))
  primary_structure <- structure_by_k[[primary_idx]]
  preservation <- vapply(structure_by_k, function(z) unname(z["knn_preservation"]), numeric(1))

  global <- global_distance_metrics(
    x_high,
    embedding,
    sample_size = sample_size_for_global_metrics,
    seed = if (is.na(seed)) 4L else seed,
    n_threads = n_threads
  )
  density <- local_density_radius_metrics(
    high_nn$distances,
    embed_distances,
    primary_k,
    keep = local_keep
  )

  silhouette <- if (is.null(labels_factor) || n_label_levels < 2L) {
    NA_real_
  } else {
    silhouette_score_with_backend(
      labels_int,
      embedding,
      n_label_levels,
      backend = metric_backend
    )$value
  }
  label_knn_accuracy <- NA_real_
  ari <- nmi <- rare_class_recall <- NA_real_
  per_class_recall <- data.frame(label = character(), n = integer(), recall = numeric(), stringsAsFactors = FALSE)
  if (!is.null(labels_factor) && n_label_levels >= 2L) {
    pred <- classification_from_embedding_nn(embed_indices, labels_factor, primary_k)
    label_knn_accuracy <- mean(pred == labels_factor, na.rm = TRUE)
    recalls <- class_recall_metrics(labels_factor, pred)
    rare_class_recall <- recalls$rare_class_recall
    per_class_recall <- recalls$table
    clusters <- embedding_clusters(embedding, labels_factor, if (is.na(seed)) 4L else seed)
    if (!all(is.na(clusters))) {
      ari <- adjusted_rand_index(labels_factor, clusters)
      nmi <- normalized_mutual_info(labels_factor, clusters)
    }
  }

  batch_metrics <- batch_entropy_metrics(embed_indices, batch, primary_k)
  label_batch_tradeoff <- if (is.finite(label_knn_accuracy) && is.finite(batch_metrics$batch_mixing_score)) {
    0.5 * label_knn_accuracy + 0.5 * batch_metrics$batch_mixing_score
  } else {
    NA_real_
  }

  out <- data.frame(
    dataset = dataset,
    method = method,
    backend = backend,
    metric_backend = metric_backend,
    metric_backend_reason = metric_backend_reason,
    high_nn_backend = if (is.null(high_nn$backend)) NA_character_ else as.character(high_nn$backend),
    embedding_nn_backend = attr(embed_nn, "backend"),
    n_threads = if (identical(metric_backend, "cpu")) as.integer(n_threads) else NA_integer_,
    seed = as.integer(seed),
    primary_k = as.integer(primary_k),
    local_sample_size = length(local_keep),
    trustworthiness = unname(primary_structure["local_trustworthiness"]),
    continuity = unname(primary_structure["local_continuity"]),
    knn_preservation = unname(primary_structure["knn_preservation"]),
    knn_preservation_15 = named_metric_or_na(preservation, "knn_preservation_15"),
    knn_preservation_30 = named_metric_or_na(preservation, "knn_preservation_30"),
    knn_preservation_50 = named_metric_or_na(preservation, "knn_preservation_50"),
    mean_neighbor_rank_error = mean_neighbor_rank_error_cpp(high_indices, embed_indices, primary_k),
    distance_spearman = global$distance_spearman,
    distance_pearson = global$distance_pearson,
    stress = global$stress,
    global_sample_size = global$global_sample_size,
    global_pair_count = global$global_pair_count,
    density_spearman = density$density_spearman,
    density_pearson = density$density_pearson,
    density_log_radius_rmse = density$density_log_radius_rmse,
    density_radius_high_mean = density$density_radius_high_mean,
    density_radius_embedding_mean = density$density_radius_embedding_mean,
    density_sample_size = density$density_sample_size,
    evaluation_reference_cache_hit = isTRUE(high_nn$cache_hit),
    evaluation_reference_cache_path = if (is.null(high_nn$cache_path)) NA_character_ else as.character(high_nn$cache_path),
    centroid_distance_correlation = centroid_distance_correlation(x_high, embedding, labels_factor),
    silhouette = silhouette,
    label_knn_accuracy = label_knn_accuracy,
    nn_accuracy = label_knn_accuracy,
    ari = ari,
    nmi = nmi,
    rare_class_recall = rare_class_recall,
    per_class_recall_json = class_recall_json(per_class_recall),
    batch_entropy = batch_metrics$batch_entropy,
    batch_mixing_score = batch_metrics$batch_mixing_score,
    label_batch_tradeoff = label_batch_tradeoff,
    stringsAsFactors = FALSE
  )
  for (metric_name in names(preservation)) {
    if (!metric_name %in% names(out)) {
      out[[metric_name]] <- unname(preservation[[metric_name]])
    }
  }
  attr(out, "per_class_recall") <- per_class_recall
  out
}
