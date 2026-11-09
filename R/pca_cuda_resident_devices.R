pca_resident_cuda_input <- function(x, ncomp, center, scale,
        backend, devices) {
    backend <- validate_pca_backend(resolve_embedding_backend(backend))
    if (backend != "cuda") stop(
        "Resident PCA `devices` requires CUDA; no fallback was used.",
        call. = FALSE)
    if (!is.matrix(x) || !is.double(x)) stop(
        "Resident multi-GPU PCA requires a double R matrix.",
        call. = FALSE)
    if (!is.logical(center) || length(center) != 1L || is.na(center) ||
        !is.logical(scale) || length(scale) != 1L || is.na(scale)) stop(
        "`center` and `scale` must be TRUE or FALSE.", call. = FALSE)
    rank <- integer_scalar(ncomp)
    if (is.na(rank) || rank < 1L || rank > ncol(x) ||
        rank >= nrow(x) || ncol(x) > 2048L) stop(
        "Resident multi-GPU PCA needs rank < nrow and <=2048 columns.",
        call. = FALSE)
    if (length(devices) < 2L) stop(
        "Resident multi-GPU PCA needs at least two CUDA devices.",
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
    stats <- pca_cuda_resident_covariance_cpp(x, devices, center)
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
    scores <- pca_cuda_resident_project_cpp(x, devices,
        stats$center, deviations, loadings)
    colnames(scores) <- colnames(loadings)
    fit <- list(scores = scores, loadings = loadings,
        singular_values = sqrt(pmax(eig$values[
            seq_len(request$rank)], 0) * (nrow(x) - 1)),
        center = stats$center, scale = deviations,
        ncomp = request$rank, method = "resident_covariance",
        backend = "cuda", engine = "native_cuda_multigpu_covariance",
        decomposition_backend = "cpu_eigen",
        precision = "float32_gpu_float64_output",
        gpu_devices = devices, gpu_identity = identity,
        seed = seed)
    class(fit) <- "fastEmbedR_pca"
    if (!is.null(xtest)) {
        if (!is.matrix(xtest) || !is.double(xtest) ||
            ncol(xtest) != ncol(x) || nrow(xtest) < 1L ||
            any(!is.finite(xtest))) stop(
            "Resident multi-GPU PCA `xtest` needs a double matrix ",
            "with matching columns.", call. = FALSE)
        projection_devices <- head(devices, nrow(xtest))
        fit$scores_test <- pca_cuda_resident_project_cpp(xtest,
            projection_devices, stats$center, deviations, loadings)
        colnames(fit$scores_test) <- colnames(loadings)
        fit$test_projection_devices <- projection_devices
    }
    attach_opentsne_pca_init(fit, tsne_init)
}
