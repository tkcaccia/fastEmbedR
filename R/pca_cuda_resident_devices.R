resident_cuda_pca_matrix <- function(x) {
    if (is_float32_matrix(x)) {
        return(length(fastembedr_matrix_dimensions(x)) == 2L)
    }
    is.matrix(x) && is.double(x)
}

pca_resident_cuda_input <- function(x, ncomp, center, scale,
        backend, devices) {
    backend <- validate_pca_backend(resolve_embedding_backend(backend))
    if (backend != "cuda") stop(
        "Resident PCA `devices` requires CUDA; no fallback was used.",
        call. = FALSE)
    if (!resident_cuda_pca_matrix(x)) stop(
        "Resident multi-GPU PCA requires a double or float32 matrix.",
        call. = FALSE)
    if (!is.logical(center) || length(center) != 1L || is.na(center) ||
        !is.logical(scale) || length(scale) != 1L || is.na(scale)) stop(
        "`center` and `scale` must be TRUE or FALSE.", call. = FALSE)
    rank <- integer_scalar(ncomp)
    if (is.na(rank) || rank < 1L || rank > ncol(x) ||
        rank >= nrow(x) || ncol(x) > 2048L) stop(
        "Resident multi-GPU PCA needs rank < nrow and <=2048 columns.",
        call. = FALSE)
    if (length(devices) < 1L) stop(
        "Resident CUDA PCA needs at least one CUDA device.",
        call. = FALSE)
    devices <- massive_validate_devices(devices, backend, nrow(x))
    list(rank = rank, devices = devices)
}

pca_resident_cuda_fit <- function(x, ncomp, xtest, center, scale,
        backend, seed, tsne_init, devices) {
    request <- pca_resident_cuda_input(x, ncomp, center, scale,
        backend, devices)
    devices <- request$devices
    identity <- massive_cuda_device_uuids(devices)
    seed <- integer_scalar(seed)
    if (is.na(seed)) stop("`seed` must be an integer.", call. = FALSE)
    started <- proc.time()[["elapsed"]]
    stats <- pca_cuda_resident_covariance_cpp(x, devices, center)
    covariance_done <- proc.time()[["elapsed"]]
    if (any(!is.finite(stats$covariance))) stop(
        "Resident CUDA PCA covariance is nonfinite.", call. = FALSE)
    deviations <- if (scale) sqrt(diag(stats$covariance)) else
        rep(1, ncol(x))
    if (any(!is.finite(deviations)) || any(deviations <= 0)) stop(
        "PCA scaling requires positive variance in every column.",
        call. = FALSE)
    covariance <- stats$covariance / outer(deviations, deviations)
    eig <- eigen((covariance + t(covariance)) / 2, symmetric = TRUE)
    loadings <- eig$vectors[, seq_len(request$rank), drop = FALSE]
    colnames(loadings) <- paste0("PC", seq_len(request$rank))
    decomposition_done <- proc.time()[["elapsed"]]
    projection <- pca_cuda_resident_project_cpp(x, devices,
        stats$center, deviations, loadings)
    projection_done <- proc.time()[["elapsed"]]
    fit <- list(scores = projection$scores, loadings = loadings,
        singular_values = sqrt(pmax(eig$values[
            seq_len(request$rank)], 0) * (nrow(x) - 1)),
        center = stats$center, scale = deviations,
        ncomp = request$rank, method = "resident_covariance",
        backend = "cuda", engine = if (length(devices) == 1L)
            "native_cuda_covariance" else "native_cuda_multigpu_covariance",
        decomposition_backend = "cpu_eigen",
        precision = "float32_gpu_float64_output",
        gpu_devices = devices, gpu_identity = identity,
        device_rows = list(covariance = stats$device_rows,
            projection = projection$device_rows),
        seed = seed, timing = c(covariance = covariance_done - started,
            decomposition = decomposition_done - covariance_done,
            scores = projection_done - decomposition_done,
            total = projection_done - started))
    class(fit) <- "fastEmbedR_pca"
    if (!is.null(xtest)) {
        fit <- resident_cuda_pca_xtest(fit, xtest, devices)
    }
    attach_opentsne_pca_init(fit, tsne_init)
}

resident_cuda_pca_xtest <- function(fit, xtest, devices) {
    if (!resident_cuda_pca_matrix(xtest) || nrow(xtest) < 1L ||
        ncol(xtest) != nrow(fit$loadings)) stop(
        "Resident multi-GPU PCA `xtest` needs a double or float32 matrix ",
        "with matching columns.", call. = FALSE)
    projection_devices <- head(devices, nrow(xtest))
    projected <- pca_cuda_resident_project_cpp(xtest,
        projection_devices, fit$center, fit$scale, fit$loadings)
    fit$scores_test <- projected$scores
    fit$test_projection_devices <- projection_devices
    fit$test_projection_device_rows <- projected$device_rows
    fit
}
