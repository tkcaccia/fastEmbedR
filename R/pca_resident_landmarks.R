pca_resident_landmarks <- function(x, ncomp, xtest, center,
        scale, backend, n.cores, seed, tsne_init, devices,
        landmarks, massive, checkpoint, resume) {
    if (massive != "off") stop(
        "Resident PCA landmarks require `massive = \"off\"`; ",
        "use `massive_project_pca()` for file-backed projection.",
        call. = FALSE)
    if (!identical(checkpoint, FALSE) || !identical(resume, FALSE)) stop(
        "PCA landmark checkpoint controls require file-backed mode.",
        call. = FALSE)
    selected <- select_landmarks(x, landmarks, seed, n.cores)
    rank <- integer_scalar(ncomp)
    if (is.na(rank) || selected$n_landmarks <= rank) stop(
        "PCA needs more selected landmarks than components.",
        call. = FALSE)
    if (selected$n_landmarks == selected$n) {
        fit <- pca(x, ncomp, xtest, center, scale, backend,
            n.cores, seed, tsne_init, devices = devices)
        fit$fit_scope <- "all_rows_optimized"
        fit$landmarks <- selected
        return(fit)
    }
    reference <- x[selected$indices, , drop = FALSE]
    multi_cuda <- length(devices) > 1L
    fit <- pca(reference, ncomp, xtest = if (multi_cuda) x else NULL,
        center = center,
        scale = scale, backend = backend, n.cores = n.cores,
        seed = seed, tsne_init = FALSE, devices = devices)
    fit$scores_reference <- fit$scores
    fit$scores <- if (multi_cuda) fit$scores_test else
        pca_resident_project_scores(x, fit)
    fit$scores_test <- NULL
    if (multi_cuda) {
        fit$projection_devices <- fit$test_projection_devices
        fit$test_projection_devices <- NULL
    }
    fit$fit_scope <- "landmark_fit_plus_projection"
    fit$landmarks <- selected
    fit$projection_backend <- if (multi_cuda) "cuda" else
        if (startsWith(fit$backend, "metal")) "metal" else "cpu"
    if (!is.null(xtest)) {
        fit <- pca_resident_test_projection(fit, xtest, devices,
            multi_cuda)
    }
    attach_opentsne_pca_init(fit, tsne_init)
}

pca_resident_test_projection <- function(fit, xtest, devices,
        multi_cuda) {
    if (!multi_cuda) {
        fit$scores_test <- pca_resident_project_scores(xtest, fit)
        fit$test_projection_backend <- if (startsWith(fit$backend,
            "metal")) "metal" else "cpu"
        return(fit)
    }
    data <- prepare_pca_test_matrix(xtest, fit)$data
    data <- embedding_dense_double_matrix(data)
    projection_devices <- head(devices, nrow(data))
    fit$scores_test <- pca_cuda_resident_project_cpp(data,
        projection_devices, fit$center, fit$scale, fit$loadings)
    colnames(fit$scores_test) <- colnames(fit$loadings)
    fit$test_projection_backend <- "cuda"
    fit$test_projection_devices <- projection_devices
    fit
}

pca_resident_project_scores <- function(x, fit) {
    if (!startsWith(fit$backend, "metal"))
        return(project_pca_test_scores(x, fit))
    input <- prepare_pca_test_matrix(x, fit)
    data <- input$data
    loadings <- embedding_dense_double_matrix(fit$loadings)
    scores <- matrix(0, nrow(data), ncol(loadings))
    chunk <- as.integer(max(1, floor(64 * 1024^2 /
        (8 * ncol(data)))))
    for (first in seq.int(1L, nrow(data), by = chunk)) {
        last <- min(nrow(data), first + chunk - 1L)
        block <- embedding_dense_double_matrix(
            data[first:last, , drop = FALSE])
        block <- sweep(block, 2L, as.numeric(fit$center), "-")
        block <- sweep(block, 2L, as.numeric(fit$scale), "/")
        scores[first:last, ] <- rsvd_multiply_metal_cpp(
            block, loadings, FALSE)
    }
    if (input$float32) scores <- float::fl(scores)
    colnames(scores) <- colnames(fit$loadings)
    scores
}
