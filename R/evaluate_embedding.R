safe_numeric_cor <- function(x, y, method = "pearson") {
    ok <- is.finite(x) & is.finite(y)
    if (sum(ok) < 3L) {
        return(NA_real_)
    }
    x <- x[ok]
    y <- y[ok]
    if (stats::sd(x) == 0 || stats::sd(y) == 0) {
        return(NA_real_)
    }
    stats::cor(x, y, method = method)
}

normalized_stress <- function(high_dist, low_dist) {
    ok <- is.finite(high_dist) & is.finite(low_dist)
    if (sum(ok) < 3L) {
        return(NA_real_)
    }
    high_dist <- high_dist[ok]
    low_dist <- low_dist[ok]
    high_scale <- sqrt(sum(high_dist * high_dist))
    low_scale <- sqrt(sum(low_dist * low_dist))
    if (high_scale == 0 || low_scale == 0) {
        return(NA_real_)
    }
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
    if (total_comb == 0) {
        return(NA_real_)
    }
    expected <- row_comb * col_comb / total_comb
    max_index <- 0.5 * (row_comb + col_comb)
    denom <- max_index - expected
    if (denom == 0) {
        return(NA_real_)
    }
    (sum_comb - expected) / denom
}

normalized_mutual_info <- function(x, y) {
    x <- as.factor(x)
    y <- as.factor(y)
    tab <- table(x, y)
    n <- sum(tab)
    if (n == 0) {
        return(NA_real_)
    }
    pij <- tab / n
    pi <- rowSums(pij)
    pj <- colSums(pij)
    nz <- pij > 0
    mi <- sum(pij[nz] * log(pij[nz] / outer(pi, pj)[nz]))
    hx <- -sum(pi[pi > 0] * log(pi[pi > 0]))
    hy <- -sum(pj[pj > 0] * log(pj[pj > 0]))
    if (hx == 0 || hy == 0) {
        return(NA_real_)
    }
    mi / sqrt(hx * hy)
}

embedding_clusters <- function(embedding, labels = NULL, seed = 4L) {
    if (is.null(labels)) {
        return(rep(NA_integer_, nrow(embedding)))
    }
    labels <- as.factor(labels)
    n_clusters <- length(levels(labels))
    if (n_clusters < 2L || n_clusters >= nrow(embedding)) {
        return(rep(
            NA_integer_, nrow(embedding)
        ))
    }
    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    out <- tryCatch(
        stats::kmeans(embedding,
            centers = n_clusters, nstart = 5L,
            iter.max = 50L
        )$cluster,
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
        if (!any(keep)) {
            return(NA_real_)
        }
        mean(pred[keep] == level, na.rm = TRUE)
    }, numeric(1))
    counts <- as.integer(table(truth)[levels_truth])
    rare_cutoff <- stats::quantile(counts, probs = 0.25, type = 1, na.rm = TRUE)
    rare <- counts <= rare_cutoff
    list(
        table = data.frame(
            label = levels_truth,
            n = counts,
            recall = unname(recall),
            stringsAsFactors = FALSE
        ),
        rare_class_recall = if (any(rare, na.rm = TRUE)) {
            mean(recall[rare],
                na.rm = TRUE
            )
        } else {
            NA_real_
        }
    )
}

class_recall_json <- function(recall_table) {
    if (requireNamespace("jsonlite", quietly = TRUE)) {
        json <- jsonlite::toJSON(
            recall_table,
            auto_unbox = TRUE,
            dataframe = "rows",
            null = "null"
        )
        return(as.character(json))
    }
    paste(paste(recall_table$label, recall_table$recall, sep = ":"),
        collapse = ";"
    )
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
    if (is.null(labels)) {
        return(NA_real_)
    }
    labels <- as.factor(labels)
    if (length(levels(labels)) < 3L) {
        return(NA_real_)
    }
    class_centroid <- function(x, level) {
        colMeans(x[labels == level, , drop = FALSE])
    }
    high_centers <- do.call(
        rbind,
        lapply(levels(labels), function(level) class_centroid(x_high, level))
    )
    low_centers <- do.call(
        rbind,
        lapply(levels(labels), function(level) class_centroid(embedding, level))
    )
    safe_numeric_cor(
        stats::dist(high_centers),
        stats::dist(low_centers),
        method = "pearson"
    )
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
    if (is.null(sample_size)) {
        return(n)
    }
    sample_size <- integer_scalar(sample_size)
    if (is.na(sample_size) || sample_size < 1L) {
        stop("Sample sizes must be positive integers.", call. = FALSE)
    }
    min(sample_size, n)
}

evaluation_reference_cache_path <- function(cache_dir,
                                            dataset,
                                            n,
                                            p,
                                            max_k,
                                            backend = "cpu",
                                            data_fingerprint = "unknown",
                                            metric = "euclidean") {
    dataset <- as.character(dataset)
    if (length(dataset) != 1L || is.na(dataset) || !nzchar(dataset)) {
        dataset <-
            "dataset"
    }
    cache_file(
        cache_dir,
        "eval_nn",
        dataset,
        n,
        p,
        paste0("k", max_k, "_", metric, "_", backend, "_", data_fingerprint)
    )
}

evaluation_data_fingerprint <- function(x) {
    nr <- nrow(x)
    nc <- ncol(x)
    rows <- unique(as.integer(round(seq.int(1L, nr, length.out = min(
        17L,
        nr
    )))))
    cols <- unique(as.integer(round(seq.int(1L, nc, length.out = min(
        17L,
        nc
    )))))
    payload <- list(
        dim = c(nr, nc),
        sample = as.numeric(x[rows, cols, drop = FALSE]),
        row_sums = as.numeric(rowSums(x[rows, , drop = FALSE])),
        col_sums = as.numeric(colSums(x[, cols, drop = FALSE]))
    )
    path <- tempfile("fastembedr-eval-fingerprint-", fileext = ".bin")
    on.exit(unlink(path), add = TRUE)
    con <- file(path, open = "wb")
    tryCatch(
        writeBin(serialize(payload, NULL, version = 2L), con),
        finally = close(con)
    )
    as.character(unname(tools::md5sum(path)))
}

normalize_evaluation_reference <- function(reference_nn, n, max_k) {
    out <- normalize_supplied_knn(reference_nn, n, max_k)
    out$backend <- if (is.null(reference_nn$backend)) {
        attr(reference_nn, "backend")
    } else {
        reference_nn$backend
    }
    if (is.null(out$backend) || length(out$backend) == 0L || is.na(
        out$backend
    )) {
        out$backend <- "precomputed"
    }
    out$cache_hit <- isTRUE(reference_nn$cache_hit)
    out$cache_path <- if (is.null(reference_nn$cache_path)) {
        NA_character_
    } else {
        as.character(reference_nn$cache_path)
    }
    out
}

knn_backend_label <- function(knn) {
    label <- attr(knn, "backend", exact = TRUE)
    if (is.null(label) && is.list(knn)) label <- knn$backend
    if (is.null(label) || length(label) == 0L || is.na(label[[1L]])) {
        NA_character_
    } else {
        as.character(label[[1L]])
    }
}

get_or_compute_evaluation_reference <- function(x_high,
                                                max_k,
                                                dataset = "dataset",
                                                use_cache = TRUE,
                                                cache_dir = file.path(
                                                    "results",
                                                    "cache"
                                                ),
                                                force_recompute = FALSE,
                                                backend = "cpu",
                                                n_threads = NULL) {
    max_k <- validate_evaluation_max_k(max_k, nrow(x_high))
    cache_path <- evaluation_reference_path(
        x_high, max_k, dataset, cache_dir, backend
    )
    if (isTRUE(use_cache) && !isTRUE(force_recompute) && file.exists(
        cache_path
    )) {
        cached <- readRDS(cache_path)
        cached$cache_hit <- TRUE
        return(normalize_evaluation_reference(cached, nrow(x_high), max_k))
    }
    raw <- fastembedr_nn_without_self(
        x_high,
        k = max_k,
        backend = backend,
        method = "auto",
        metric = "euclidean",
        n_threads = n_threads,
        target_recall = 0.99
    )
    out <- normalize_supplied_knn(raw, nrow(x_high), max_k)
    out$backend <- knn_backend_label(raw)
    out$cache_hit <- FALSE
    out$cache_path <- cache_path
    if (isTRUE(use_cache)) {
        dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
        saveRDS(out, cache_path)
    }
    out
}

validate_evaluation_max_k <- function(max_k, n) {
    max_k <- min(as.integer(max_k), n - 1L)
    if (length(max_k) != 1L || is.na(max_k) || max_k < 1L) {
        stop(
            "`max_k` must be positive and smaller than `nrow(x_high)`.",
            call. = FALSE
        )
    }
    max_k
}

evaluation_reference_path <- function(x, max_k, dataset,
                                        cache_dir, backend) {
    evaluation_reference_cache_path(
        cache_dir, dataset, nrow(x), ncol(x), max_k,
        backend = backend,
        data_fingerprint = evaluation_data_fingerprint(x)
    )
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

global_metric_sample <- function(x_high, embedding, sample_size, seed) {
    n <- nrow(x_high)
    sample_size <- min(as.integer(sample_size), n)
    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    keep <- if (sample_size < n) {
        sort(sample.int(n, sample_size))
    } else {
        seq_len(
            n
        )
    }
    list(
        high = embedding_dense_double_matrix(
            x_high[keep, , drop = FALSE]
        ),
        embedding = embedding_dense_double_matrix(
            embedding[keep, , drop = FALSE]
        ),
        keep = keep
    )
}

global_metric_distances <- function(sample, seed, n_threads) {
    keep <- sample$keep
    pair_total <- length(keep) * (length(keep) - 1L) / 2
    max_pairs <- min(pair_total, 250000L)
    if (pair_total <= max_pairs) {
        return(list(
            high = as.numeric(stats::dist(sample$high)),
            low = as.numeric(stats::dist(sample$embedding))
        ))
    }
    restore_seed <- set_local_seed(seed + 104729L)
    on.exit(restore_seed(), add = TRUE)
    a <- sample.int(length(keep), as.integer(max_pairs), replace = TRUE)
    b <- sample.int(length(keep) - 1L, length(a), replace = TRUE)
    b <- b + as.integer(b >= a)
    list(
        high = sampled_pair_distances(
            sample$high, a, b,
            n_threads = n_threads
        ),
        low = sampled_pair_distances(
            sample$embedding, a, b,
            n_threads = n_threads
        )
    )
}

global_distance_metrics <- function(x_high, embedding, sample_size, seed,
                                    n_threads = NULL) {
    if (nrow(x_high) < 3L) {
        return(list(
            distance_spearman = NA_real_, distance_pearson = NA_real_,
            stress = NA_real_, global_sample_size = nrow(x_high),
            global_pair_count = 0L
        ))
    }
    sample <- global_metric_sample(x_high, embedding, sample_size, seed)
    distances <- global_metric_distances(sample, seed, n_threads)
    list(
        distance_spearman = safe_numeric_cor(
            distances$high, distances$low,
            method = "spearman"
        ),
        distance_pearson = safe_numeric_cor(
            distances$high, distances$low,
            method = "pearson"
        ),
        stress = normalized_stress(distances$high, distances$low),
        global_sample_size = length(sample$keep),
        global_pair_count = length(distances$high)
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

empty_density_metrics <- function(high_mean = NA_real_,
                                    embedding_mean = NA_real_, n = 0L) {
    list(
        density_spearman = NA_real_, density_pearson = NA_real_,
        density_log_radius_rmse = NA_real_,
        density_radius_high_mean = high_mean,
        density_radius_embedding_mean = embedding_mean,
        density_sample_size = n
    )
}

density_radius_epsilon <- function(high_radius, embedding_radius, ok) {
    positive <- c(
        high_radius[ok & high_radius > 0],
        embedding_radius[ok & embedding_radius > 0]
    )
    value <- if (length(positive) == 0L) {
        .Machine$double.eps
    } else {
        min(positive, na.rm = TRUE) * 1e-6
    }
    if (!is.finite(value) || value <= 0) .Machine$double.eps else value
}

summarize_density_radius <- function(high_radius, embedding_radius, ok) {
    eps <- density_radius_epsilon(high_radius, embedding_radius, ok)
    high_log <- log(pmax(high_radius[ok], eps))
    embedding_log <- log(pmax(embedding_radius[ok], eps))
    high_z <- standardize_log_radius(high_log)
    embedding_z <- standardize_log_radius(embedding_log)
    list(
        density_spearman = safe_numeric_cor(
            high_log, embedding_log,
            method = "spearman"
        ),
        density_pearson = safe_numeric_cor(
            high_log, embedding_log,
            method = "pearson"
        ),
        density_log_radius_rmse = sqrt(mean(
            (high_z - embedding_z)^2,
            na.rm = TRUE
        )),
        density_radius_high_mean = mean(high_radius[ok], na.rm = TRUE),
        density_radius_embedding_mean = mean(
            embedding_radius[ok],
            na.rm = TRUE
        ),
        density_sample_size = sum(ok)
    )
}

local_density_radius_metrics <- function(high_distances,
                                            embedding_distances, k,
                                            keep = NULL) {
    k <- min(as.integer(k), ncol(high_distances), ncol(embedding_distances))
    if (length(k) != 1L || is.na(k) || k < 1L) {
        return(empty_density_metrics())
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
        return(empty_density_metrics(
            mean(high_radius[ok], na.rm = TRUE),
            mean(embedding_radius[ok], na.rm = TRUE), sum(ok)
        ))
    }
    summarize_density_radius(high_radius, embedding_radius, ok)
}

#' Evaluate an embedding against high-dimensional structure
#'
#' Trustworthiness and continuity use their standard rank-penalty definitions.
#' Local rank metrics are computed exactly on a deterministic subsample so the
#' function does not allocate all-pairs matrices for the complete dataset.
#' Float32 inputs are sampled before conversion to a dense double matrix.
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
#'   list. It may include a self-neighbor column.
#' @param sample_size_for_global_metrics Maximum deterministic subsample size
#'   for all-pairs global distance metrics.
#' @param sample_size_for_local_metrics Maximum deterministic subsample size for
#'   local trustworthiness, continuity, and neighbor preservation metrics.
#' @param use_cache Cache the high-dimensional reference neighbors used by
#'   quality metrics.
#' @param cache_dir Directory for quality-metric reference-neighbor cache files.
#' @param force_recompute Ignore cached quality reference neighbors.
#' @param seed Random seed recorded in the output and used for subsampling.
#' @param method Method name recorded in the output.
#' @param backend Backend name recorded in the output.
#' @param n.cores Number of CPU cores used when quality metrics need
#'   nearest-neighbor searches. Ignored by native GPU metric backends.
#' @param dataset Dataset name recorded in the output.
#' @return A one-row data frame with local, global, label-aware, batch-aware,
#'   and metadata columns. Per-class recall is also attached as an attribute.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' metrics <- evaluate_embedding(
#'     x, x[, 1:2],
#'     labels = iris$Species, k = 5,
#'     sample_size_for_local_metrics = 100,
#'     sample_size_for_global_metrics = 100,
#'     seed = 1
#' )
#' @export
evaluate_embedding <- function(x_high, embedding, labels = NULL,
                                batch = NULL, k = c(15L, 30L, 50L),
                                primary_k = NULL, reference_nn = NULL,
                                sample_size_for_global_metrics = min(
                                    5000L, nrow(x_high)
                                ),
                                sample_size_for_local_metrics = min(
                                    2000L, nrow(x_high)
                                ),
                                use_cache = FALSE,
                                cache_dir = file.path("results", "cache"),
                                force_recompute = FALSE, seed = NA_integer_,
                                method = NA_character_, backend = NA_character_,
                                n.cores = NULL, dataset = NA_character_) {
    input <- validate_evaluation_inputs(
        x_high, embedding, labels, batch
    )
    sample <- prepare_evaluation_sample(
        input, k, primary_k, sample_size_for_local_metrics, seed
    )
    neighbors <- resolve_evaluation_neighbors(
        input, sample, reference_nn, backend, dataset,
        use_cache, cache_dir, force_recompute, n.cores
    )
    metrics <- compute_evaluation_metrics(
        input, sample, neighbors, sample_size_for_global_metrics, seed
    )
    assemble_evaluation_output(
        input, sample, neighbors, metrics, dataset,
        method, backend, seed
    )
}

validate_finite_evaluation_matrix <- function(x, name) {
    if (!is.matrix(x) && !is_float32_matrix(x)) {
        x <- as.matrix(x)
    }
    if (!is_float32_matrix(x) && !is.numeric(x)) {
        stop("`", name, "` must be a numeric matrix.", call. = FALSE)
    }
    if (length(dim(x)) != 2L || ncol(x) < 1L) {
        stop("`", name, "` must have at least one column.", call. = FALSE)
    }
    finite <- if (is_float32_matrix(x)) {
        float32_all_finite_cpp(x)
    } else {
        all(is.finite(x))
    }
    if (!isTRUE(finite)) {
        stop("`", name, "` must contain only finite values.", call. = FALSE)
    }
    x
}

validate_evaluation_inputs <- function(x_high, embedding, labels, batch) {
    x_high <- validate_finite_evaluation_matrix(x_high, "x_high")
    embedding <- validate_finite_evaluation_matrix(embedding, "embedding")
    if (nrow(x_high) != nrow(embedding)) {
        stop(
            "`x_high` and `embedding` must have the same row count.",
            call. = FALSE
        )
    }
    if (nrow(x_high) < 3L) {
        stop("`x_high` must contain at least three rows.", call. = FALSE)
    }
    if (!is.null(labels) && length(labels) != nrow(x_high)) {
        stop("`labels` must have one entry per row.", call. = FALSE)
    }
    if (!is.null(batch) && length(batch) != nrow(x_high)) {
        stop("`batch` must have one entry per row.", call. = FALSE)
    }
    list(
        x = x_high, embedding = embedding,
        labels = labels, batch = batch
    )
}

normalize_evaluation_k <- function(k, primary_k) {
    if (length(k) == 0L) {
        requested <- 15L
    } else {
        requested <- vapply(k, integer_scalar, integer(1L))
        if (anyNA(requested) || any(requested < 1L)) {
            stop("`k` must contain positive integers.", call. = FALSE)
        }
    }
    if (!is.null(primary_k)) {
        primary_k <- integer_scalar(primary_k)
        if (is.na(primary_k) || primary_k < 1L) {
            stop(
                "`primary_k` must be NULL or a positive integer.",
                call. = FALSE
            )
        }
        requested <- unique(c(requested, primary_k))
    }
    list(requested = requested, primary = primary_k)
}

prepare_evaluation_sample <- function(input, k, primary_k,
                                        sample_size, seed) {
    k_state <- normalize_evaluation_k(k, primary_k)
    metric_seed <- if (is.na(seed)) 4L else as.integer(seed)
    requested_size <- finite_sample_size(sample_size, nrow(input$x))
    minimum_rows <- floor((3 * max(k_state$requested) + 1) / 2) + 1L
    local_size <- min(
        nrow(input$x), max(requested_size, minimum_rows)
    )
    keep <- sample_indices(nrow(input$x), local_size, metric_seed)
    local_x <- embedding_dense_double_matrix(
        input$x[keep, , drop = FALSE]
    )
    local_embedding <- embedding_dense_double_matrix(
        input$embedding[keep, , drop = FALSE]
    )
    max_standard_k <- floor((2L * nrow(local_x) - 2L) / 3L)
    eval_k <- pmin(k_state$requested, max_standard_k)
    names(eval_k) <- paste0("knn_preservation_", k_state$requested)
    max_k <- max(eval_k)
    primary <- if (is.null(k_state$primary)) {
        min(15L, max_k)
    } else {
        min(as.integer(k_state$primary), max_k)
    }
    list(
        keep = keep, x = local_x, embedding = local_embedding,
        labels = if (is.null(input$labels)) NULL else input$labels[keep],
        batch = if (is.null(input$batch)) NULL else input$batch[keep],
        n = nrow(local_x), eval_k = eval_k, max_k = max_k,
        primary_k = primary, metric_seed = metric_seed
    )
}

resolve_high_evaluation_reference <- function(input, sample, reference_nn,
                                                backend, reason, dataset,
                                                use_cache, cache_dir,
                                                force_recompute, n_threads) {
    supplied <- !is.null(reference_nn) && sample$n == nrow(input$x)
    if (!is.null(reference_nn) && !supplied) {
        reason <- append_metric_backend_reason(
            reason, "reference_nn_recomputed_for_local_subsample"
        )
    }
    if (supplied) {
        return(list(
            knn = normalize_evaluation_reference(
                reference_nn, sample$n, sample$max_k
            ),
            backend = backend, reason = reason, supplied = TRUE
        ))
    }
    args <- list(
        x_high = sample$x, max_k = sample$max_k, dataset = dataset,
        use_cache = use_cache, cache_dir = cache_dir,
        force_recompute = force_recompute, backend = backend,
        n_threads = n_threads
    )
    attempt <- capture_error(do.call(
        get_or_compute_evaluation_reference, args
    ))
    if (is.na(attempt$error)) {
        return(list(
            knn = attempt$value, backend = backend,
            reason = reason, supplied = FALSE
        ))
    }
    args$backend <- "cpu"
    list(
        knn = do.call(get_or_compute_evaluation_reference, args),
        backend = "cpu",
        reason = append_metric_backend_reason(reason, attempt$error),
        supplied = FALSE
    )
}

resolve_embedding_evaluation_reference <- function(sample, backend,
                                                    reason, n_threads) {
    args <- list(
        data = sample$embedding, k = sample$max_k, backend = backend,
        method = "auto", metric = "euclidean", n_threads = n_threads,
        target_recall = 0.99
    )
    attempt <- capture_error(do.call(fastembedr_nn_without_self, args))
    if (is.na(attempt$error)) {
        raw <- attempt$value
    } else {
        backend <- "cpu"
        reason <- append_metric_backend_reason(reason, attempt$error)
        args$backend <- "cpu"
        args$method <- "hnsw"
        raw <- do.call(fastembedr_nn_without_self, args)
    }
    list(
        knn = normalize_evaluation_reference(
            raw, sample$n, sample$max_k
        ),
        backend = backend, reason = reason
    )
}

resolve_evaluation_neighbors <- function(input, sample, reference_nn,
                                            backend, dataset, use_cache,
                                            cache_dir, force_recompute,
                                            n.cores) {
    n_threads <- normalize_nn_threads(n.cores)
    resolution <- resolve_metric_backend(backend)
    high <- resolve_high_evaluation_reference(
        input, sample, reference_nn, resolution$backend,
        resolution$reason, dataset, use_cache, cache_dir,
        force_recompute, n_threads
    )
    embedding <- resolve_embedding_evaluation_reference(
        sample, high$backend, high$reason, n_threads
    )
    list(
        high = high$knn, embedding = embedding$knn,
        backend = embedding$backend, reason = embedding$reason,
        reference_supplied = high$supplied, n_threads = n_threads
    )
}

evaluation_label_state <- function(labels) {
    factor <- if (is.null(labels)) NULL else as.factor(labels)
    list(
        factor = factor,
        integer = if (is.null(factor)) integer(0L) else as.integer(factor),
        n_levels = if (is.null(factor)) 0L else length(levels(factor))
    )
}

evaluation_structure_metrics <- function(sample, neighbors) {
    by_k <- exact_structure_metrics_cpp(
        sample$x, sample$embedding, as.integer(sample$eval_k),
        as.integer(neighbors$n_threads)
    )
    primary_idx <- which.min(abs(sample$eval_k - sample$primary_k))
    preservation <- by_k[, "knn_preservation"]
    names(preservation) <- names(sample$eval_k)
    list(
        primary = by_k[primary_idx, , drop = TRUE],
        preservation = preservation
    )
}

evaluation_class_metrics <- function(sample, neighbors, label_state) {
    empty <- data.frame(
        label = character(), n = integer(), recall = numeric()
    )
    silhouette <- if (label_state$n_levels < 2L) {
        NA_real_
    } else {
        silhouette_score_with_backend(
            label_state$integer, sample$embedding,
            label_state$n_levels,
            backend = neighbors$backend
        )$value
    }
    if (label_state$n_levels < 2L) {
        return(list(
            silhouette = silhouette, accuracy = NA_real_,
            ari = NA_real_, nmi = NA_real_,
            rare_class_recall = NA_real_, per_class_recall = empty
        ))
    }
    pred <- classification_from_embedding_nn(
        neighbors$embedding$indices, label_state$factor,
        sample$primary_k
    )
    recalls <- class_recall_metrics(label_state$factor, pred)
    clusters <- embedding_clusters(
        sample$embedding, label_state$factor, sample$metric_seed
    )
    clustered <- !all(is.na(clusters))
    list(
        silhouette = silhouette,
        accuracy = mean(pred == label_state$factor, na.rm = TRUE),
        ari = if (clustered) {
            adjusted_rand_index(label_state$factor, clusters)
        } else {
            NA_real_
        },
        nmi = if (clustered) {
            normalized_mutual_info(label_state$factor, clusters)
        } else {
            NA_real_
        },
        rare_class_recall = recalls$rare_class_recall,
        per_class_recall = recalls$table
    )
}

compute_evaluation_metrics <- function(input, sample, neighbors,
                                        global_sample_size, seed) {
    labels <- evaluation_label_state(sample$labels)
    structure <- evaluation_structure_metrics(sample, neighbors)
    global <- global_distance_metrics(
        input$x, input$embedding, global_sample_size,
        if (is.na(seed)) 4L else seed,
        n_threads = neighbors$n_threads
    )
    density <- local_density_radius_metrics(
        neighbors$high$distances, neighbors$embedding$distances,
        sample$primary_k
    )
    classes <- evaluation_class_metrics(sample, neighbors, labels)
    batch <- batch_entropy_metrics(
        neighbors$embedding$indices, sample$batch, sample$primary_k
    )
    finite <- is.finite(classes$accuracy) &&
        is.finite(batch$batch_mixing_score)
    list(
        labels = labels, structure = structure, global = global,
        density = density, classes = classes, batch = batch,
        label_batch_tradeoff = if (finite) {
            0.5 * classes$accuracy + 0.5 * batch$batch_mixing_score
        } else {
            NA_real_
        }
    )
}

evaluation_identity_row <- function(sample, neighbors, dataset,
                                    method, backend, seed) {
    data.frame(
        dataset = dataset, method = method, backend = backend,
        metric_backend = neighbors$backend,
        metric_backend_reason = neighbors$reason,
        rank_metric_backend = "cpu_exact",
        high_nn_backend = neighbors$high$backend %||% NA_character_,
        embedding_nn_backend = knn_backend_label(neighbors$embedding),
        n_threads = as.integer(neighbors$n_threads),
        seed = as.integer(seed), primary_k = as.integer(sample$primary_k),
        local_sample_size = sample$n
    )
}

evaluation_structure_row <- function(metrics) {
    primary <- metrics$structure$primary
    preservation <- metrics$structure$preservation
    data.frame(
        trustworthiness = unname(primary["trustworthiness"]),
        continuity = unname(primary["continuity"]),
        knn_preservation = unname(primary["knn_preservation"]),
        knn_preservation_15 = named_metric_or_na(
            preservation, "knn_preservation_15"
        ),
        knn_preservation_30 = named_metric_or_na(
            preservation, "knn_preservation_30"
        ),
        knn_preservation_50 = named_metric_or_na(
            preservation, "knn_preservation_50"
        ),
        mean_neighbor_rank_error = unname(
            primary["mean_neighbor_rank_error"]
        )
    )
}

evaluation_geometry_row <- function(input, sample, neighbors, metrics) {
    global <- metrics$global
    density <- metrics$density
    data.frame(
        distance_spearman = global$distance_spearman,
        distance_pearson = global$distance_pearson,
        stress = global$stress,
        global_sample_size = global$global_sample_size,
        global_pair_count = global$global_pair_count,
        density_spearman = density$density_spearman,
        density_pearson = density$density_pearson,
        density_log_radius_rmse = density$density_log_radius_rmse,
        density_radius_high_mean = density$density_radius_high_mean,
        density_radius_embedding_mean =
            density$density_radius_embedding_mean,
        density_sample_size = density$density_sample_size,
        evaluation_reference_supplied_used =
            neighbors$reference_supplied,
        evaluation_reference_cache_hit =
            isTRUE(neighbors$high$cache_hit),
        evaluation_reference_cache_path =
            neighbors$high$cache_path %||% NA_character_,
        centroid_distance_correlation = centroid_distance_correlation(
            sample$x, sample$embedding, metrics$labels$factor
        )
    )
}

evaluation_class_row <- function(metrics) {
    classes <- metrics$classes
    batch <- metrics$batch
    data.frame(
        silhouette = classes$silhouette,
        label_knn_accuracy = classes$accuracy,
        nn_accuracy = classes$accuracy,
        ari = classes$ari, nmi = classes$nmi,
        rare_class_recall = classes$rare_class_recall,
        per_class_recall_json = class_recall_json(
            classes$per_class_recall
        ),
        batch_entropy = batch$batch_entropy,
        batch_mixing_score = batch$batch_mixing_score,
        label_batch_tradeoff = metrics$label_batch_tradeoff
    )
}

assemble_evaluation_output <- function(input, sample, neighbors, metrics,
                                        dataset, method, backend, seed) {
    out <- cbind(
        evaluation_identity_row(
            sample, neighbors, dataset, method, backend, seed
        ),
        evaluation_structure_row(metrics),
        evaluation_geometry_row(input, sample, neighbors, metrics),
        evaluation_class_row(metrics)
    )
    preservation <- metrics$structure$preservation
    for (name in names(preservation)) {
        if (!name %in% names(out)) {
            out[[name]] <- unname(preservation[[name]])
        }
    }
    attr(out, "per_class_recall") <- metrics$classes$per_class_recall
    out
}
