fastembedr_cpu_needs_blocking <- function(dimensions) {
    as.double(dimensions[[1L]]) * dimensions[[2L]] >
        .Machine$integer.max
}

resident_pca_float_loadings <- function(loadings) {
    if (!requireNamespace("float", quietly = TRUE)) stop(
        "Resident float32 PCA requires `float`.", call. = FALSE)
    float::fl(loadings)
}

fastembedr_cpu_blocked_pca <- function(data, ncomp, center,
        scale, seed, n.cores) {
    if (!is_float32_matrix(data)) stop(
        "Large resident CPU PCA requires float32 input.",
        call. = FALSE)
    rows <- nrow(data)
    columns <- ncol(data)
    rank <- min(as.integer(ncomp), rows - 1L, columns)
    if (rows < 2L || rank < 1L || columns > 1024L) stop(
        "Blocked CPU PCA needs two rows, usable rank and at most ",
        "1024 columns.", call. = FALSE)
    started <- proc.time()[[3L]]
    source <- list(data = data, format = "memory_float32",
        n_threads = normalize_pca_threads(n.cores))
    chunk_rows <- resident_pca_block_rows(data)
    statistics <- massive_pca_covariance_cpp(source, chunk_rows,
        normalize_pca_threads(n.cores), center)
    statistics_seconds <- unname(proc.time()[[3L]] - started)
    deviations <- if (scale) sqrt(diag(statistics$covariance)) else
        rep(1, columns)
    if (any(!is.finite(deviations)) || any(deviations <= 0)) stop(
        "PCA scaling requires positive column variance.")
    covariance <- statistics$covariance /
        outer(deviations, deviations)
    eigenfit <- eigen(covariance, symmetric = TRUE)
    loadings <- eigenfit$vectors[, seq_len(rank), drop = FALSE]
    projection_started <- proc.time()[[3L]]
    projected <- resident_float32_project_cpu_cpp(data, loadings,
        statistics$center, deviations, normalize_pca_threads(n.cores))
    projection_seconds <- unname(proc.time()[[3L]] -
        projection_started)
    columns_names <- paste0("PC", seq_len(rank))
    colnames(projected$scores) <- columns_names
    colnames(loadings) <- columns_names
    fit <- list(scores = projected$scores,
        loadings = resident_pca_float_loadings(loadings),
        singular_values = sqrt(pmax(eigenfit$values[seq_len(rank)], 0) *
            (rows - 1)), center = statistics$center, scale = deviations,
        ncomp = rank, method = "blocked_covariance", backend =
            "cpu_covariance", backend_reason = NA_character_,
        engine = "native_cpu_float32", precision = "float32",
        oversample = 0L, power = 0L,
        n_threads = projected$n_threads_effective,
        seed = as.integer(seed), timing = c(
            covariance = statistics_seconds,
            scores = projection_seconds,
            total = unname(proc.time()[[3L]] - started)))
    class(fit) <- "fastEmbedR_pca"
    fit
}
