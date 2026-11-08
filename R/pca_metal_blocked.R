fastembedr_metal_needs_blocking <- function(dimensions) {
    rows <- as.double(dimensions[[1L]])
    columns <- as.double(dimensions[[2L]])
    columns * (rows + 3) > .Machine$integer.max ||
        rows * (min(rows - 1, columns) + 3) >
            .Machine$integer.max
}

resident_pca_block_rows <- function(x) {
    as.integer(max(1, floor(64 * 1024^2 / (4 * ncol(x)))))
}

metal_resident_covariance <- function(x, chunk_rows, center) {
    rows <- nrow(x)
    columns <- ncol(x)
    cross <- matrix(0, columns, columns)
    means <- numeric(columns)
    completed <- 0
    allocated <- 0
    started <- proc.time()[[3L]]
    for (first in seq.int(1, rows, by = chunk_rows)) {
        take <- as.integer(min(chunk_rows, rows - first + 1))
        block <- resident_float32_row_major_block_cpp(x, first, take)
        moment <- pca_crossprod_block_metal_cpp(block, center)
        observed <- moment$metal_allocated_bytes_observed
        if (length(observed) != 1L || !is.finite(observed) ||
            observed <= 0) stop("Metal covariance allocation missing.")
        allocated <- max(allocated, observed)
        updated <- completed + take
        if (center) {
            delta <- moment$center - means
            cross <- cross + moment$cross +
                tcrossprod(delta) * (completed * take / updated)
            means <- means + delta * (take / updated)
        } else cross <- cross + moment$cross
        completed <- updated
    }
    covariance <- cross / (rows - 1)
    if (any(!is.finite(covariance))) stop(
        "Metal covariance contains nonfinite values.")
    list(center = means, covariance = (covariance + t(covariance)) / 2,
        allocated = allocated,
        seconds = unname(proc.time()[[3L]] - started))
}

metal_resident_project <- function(x, loadings, centers, scales,
        chunk_rows) {
    rows <- nrow(x)
    scores <- resident_float32_score_matrix_cpp(rows, ncol(loadings))
    allocated <- 0
    started <- proc.time()[[3L]]
    for (first in seq.int(1, rows, by = chunk_rows)) {
        take <- as.integer(min(chunk_rows, rows - first + 1))
        block <- resident_float32_column_major_block_cpp(x, first, take)
        projected <- pca_project_block_metal_cpp(block, loadings,
            centers, scales)
        observed <- attr(projected, "metal_allocated_bytes_observed")
        if (length(observed) != 1L || !is.finite(observed) ||
            observed <= 0) stop("Metal projection allocation missing.")
        allocated <- max(allocated, observed)
        scores <- resident_float32_score_assign_cpp(scores,
            first, projected)
    }
    list(scores = scores, allocated = allocated,
        seconds = unname(proc.time()[[3L]] - started))
}

fastembedr_metal_blocked_pca <- function(data, ncomp, center,
        scale, seed) {
    if (!is_float32_matrix(data)) stop(
        "Large resident Metal PCA requires float32 input.",
        call. = FALSE)
    rows <- nrow(data)
    columns <- ncol(data)
    rank <- min(as.integer(ncomp), rows - 1L, columns)
    if (rows < 2L || rank < 1L) stop(
        "Metal PCA needs two rows and a usable component.")
    started <- proc.time()[[3L]]
    chunk_rows <- resident_pca_block_rows(data)
    statistics <- metal_resident_covariance(data, chunk_rows, center)
    deviations <- if (scale) sqrt(diag(statistics$covariance)) else
        rep(1, columns)
    if (any(!is.finite(deviations)) || any(deviations <= 0)) stop(
        "PCA scaling requires positive column variance.")
    covariance <- statistics$covariance /
        outer(deviations, deviations)
    eigenfit <- eigen(covariance, symmetric = TRUE)
    loadings <- eigenfit$vectors[, seq_len(rank), drop = FALSE]
    projected <- metal_resident_project(data, loadings,
        statistics$center, deviations, chunk_rows)
    fit <- list(scores = projected$scores,
        loadings = resident_pca_float_loadings(loadings),
        singular_values = sqrt(pmax(eigenfit$values[seq_len(rank)], 0) *
            (rows - 1)), center = statistics$center, scale = deviations,
        ncomp = rank, method = "blocked_covariance", backend =
            "metal_mps_covariance", selection = "blocked_covariance",
        backend_reason = NA_character_, engine = "native_metal_mps",
        precision = "float32", metal_allocated_bytes_observed =
            max(statistics$allocated, projected$allocated),
        oversample = 0L, power = 0L, seed = as.integer(seed),
        timing = c(covariance = statistics$seconds,
            scores = projected$seconds,
            total = unname(proc.time()[[3L]] - started)))
    class(fit) <- "fastEmbedR_pca"
    fit
}
