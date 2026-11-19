fastembedr_narrow_covariance_route <- function(dimensions) {
    columns <- dimensions[[2L]]
    columns <= 64L && as.double(dimensions[[1L]]) * columns >= 5e6
}

fastembedr_cpu_covariance_route <- function(dimensions, float32) {
    elements <- as.double(dimensions[[1L]]) * dimensions[[2L]]
    elements > .Machine$integer.max ||
        (float32 && fastembedr_narrow_covariance_route(dimensions))
}

resident_pca_float_loadings <- function(loadings) {
    if (!requireNamespace("float", quietly = TRUE)) stop(
        "Resident float32 PCA requires `float`.", call. = FALSE)
    float::fl(loadings)
}

fastembedr_blocked_pca <- function(data, ncomp, center,
        scale, seed, n.cores, backend = "cpu") {
    if (!is_float32_matrix(data))
        stop("Blocked PCA requires float32 input.", call. = FALSE)
    rows <- nrow(data)
    columns <- ncol(data)
    rank <- min(as.integer(ncomp), rows - 1L, columns)
    if (rows < 2L || rank < 1L || columns > 1024L) stop(
        "Blocked PCA needs two rows, usable rank and up to 1024 columns.")
    started <- proc.time()[[3L]]
    n.cores <- normalize_pca_threads(n.cores)
    source <- list(data = data, format = "memory_float32", n_threads = n.cores)
    chunk_rows <- resident_pca_block_rows(data)
    cuda <- identical(backend, "cuda")
    covariance_fn <- if (cuda) massive_pca_covariance_cuda_cpp else
        massive_pca_covariance_cpp
    statistics <- covariance_fn(source, chunk_rows, n.cores, center)
    statistics_seconds <- proc.time()[[3L]] - started
    deviations <- if (scale) sqrt(diag(statistics$covariance)) else
        rep(1, columns)
    if (cuda) deviations[deviations == 0] <- 1
    if (any(!is.finite(deviations)) || any(deviations <= 0)) stop(
        "PCA scaling requires positive column variance.")
    eigenfit <- eigen(statistics$covariance /
        outer(deviations, deviations), symmetric = TRUE)
    loadings <- eigenfit$vectors[, seq_len(rank), drop = FALSE]
    projection_started <- proc.time()[[3L]]
    projected <- if (cuda) list(scores = massive_pca_project_cuda_cpp(
        source, "", loadings, statistics$center, deviations, chunk_rows)) else
        resident_float32_project_cpu_cpp(data, loadings, statistics$center,
            deviations, n.cores)
    projection_seconds <- proc.time()[[3L]] - projection_started
    columns_names <- paste0("PC", seq_len(rank))
    colnames(projected$scores) <- columns_names
    colnames(loadings) <- columns_names
    fit <- list(decomposition_backend = "cpu_eigen", scores = projected$scores,
        loadings = resident_pca_float_loadings(loadings),
        singular_values = sqrt(pmax(eigenfit$values[seq_len(rank)], 0) *
            (rows - 1)), center = statistics$center, scale = deviations,
        ncomp = rank, method = "blocked_covariance",
        backend = paste0(backend, "_covariance"),
        backend_reason = NA_character_,
        engine = paste0("native_", backend, "_float32"), precision = "float32",
        oversample = 0L, power = 0L,
        n_threads = projected$n_threads_effective,
        seed = as.integer(seed), timing = c(covariance = statistics_seconds,
            scores = projection_seconds, total = proc.time()[[3L]] - started))
    class(fit) <- "fastEmbedR_pca"
    fit
}
