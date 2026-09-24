# Internal backend-aware PCA initialization for t-SNE workflows.

annotate_opentsne_pca_init <- function(pca, n_components) {
    init <- normalize_opentsne_pca_scores(pca$scores, n_components)
    attr(init, "fastEmbedR_init_method") <- paste0("pca_", pca$method)
    attr(init, "fastEmbedR_init_backend") <- pca$backend
    attr(init, "fastEmbedR_init_backend_reason") <- pca$backend_reason
    if (!is.null(pca$package)) {
        attr(init, "fastEmbedR_init_package") <- pca$package
        attr(init, "fastEmbedR_init_package_version") <- pca$package_version
    }
    if (!is.null(pca$timing)) {
        attr(init, "fastEmbedR_init_timing") <- pca$timing
    }
    init
}

make_metal_opentsne_pca_init <- function(x, n_components, seed) {
    pca <- fastembedr_metal_rsvd_pca(
        x,
        ncomp = n_components, center = TRUE, scale = FALSE, seed = seed
    )
    pca$package <- "fastEmbedR native Metal/MPS"
    pca$package_version <- as.character(utils::packageVersion("fastEmbedR"))
    annotate_opentsne_pca_init(pca, n_components)
}

make_cuda_opentsne_pca_init <- function(x, n_components, seed) {
    if (!exists("pca_tsvd_cuda_cpp", mode = "function")) {
        stop(
            "CUDA PCA initialization requires a native CUDA PCA backend, ",
            "but the package was not built with that capability.",
            call. = FALSE
        )
    }
    result <- capture_error(fastembedr_cuda_pca(
        x,
        ncomp = n_components, center = TRUE, scale = FALSE, seed = seed
    ))
    if (is.null(result$value)) {
        stop(
            "CUDA PCA initialization failed in the native CUDA backend: ",
            result$error,
            call. = FALSE
        )
    }
    result$value$package <- "fastEmbedR native CUDA PCA"
    result$value$package_version <- NA_character_
    annotate_opentsne_pca_init(result$value, n_components)
}

make_cpu_opentsne_pca_init <- function(x, n_components, seed, n_threads) {
    requested_threads <- if (is.null(n_threads)) {
        default_tsne_threads()
    } else {
        normalize_pca_threads(n_threads)
    }
    pca <- fastembedr_cpu_rsvd_pca(
        x,
        ncomp = n_components,
        center = TRUE,
        scale = FALSE,
        seed = seed,
        n.cores = requested_threads
    )
    pca$package <- "fastEmbedR native CPU RSVD"
    pca$package_version <- as.character(utils::packageVersion("fastEmbedR"))
    annotate_opentsne_pca_init(pca, n_components)
}

make_opentsne_pca_init <- function(
    x, n_components, seed, backend = "cpu", n_threads = NULL,
    .thread_controlled = FALSE
) {
    x <- opentsne_pca_input_matrix(x)
    n_components <- as.integer(n_components)
    if (length(n_components) != 1L || is.na(n_components) ||
        n_components < 1L) {
        stop("`n_components` must be a positive integer.", call. = FALSE)
    }
    if (identical(backend, "cpu") && !isTRUE(.thread_controlled)) {
        requested <- n_threads %||% default_tsne_threads()
        return(with_pca_cpu_threads(
            requested,
            make_opentsne_pca_init(
                x, n_components, seed, backend, requested, TRUE
            )
        )$value)
    }
    if (identical(backend, "metal")) {
        return(make_metal_opentsne_pca_init(x, n_components, seed))
    }
    if (identical(backend, "cuda")) {
        return(make_cuda_opentsne_pca_init(x, n_components, seed))
    }
    make_cpu_opentsne_pca_init(x, n_components, seed, n_threads)
}

resolve_opentsne_y_init <- function(Y_init, n, n_components) {
    if (is.null(Y_init)) {
        return(NULL)
    }
    if (is.character(Y_init) && length(Y_init) == 1L) {
        path <- path.expand(Y_init)
        if (!file.exists(path)) {
            stop("`Y_init` file does not exist: ", path, call. = FALSE)
        }
        Y_init <- readRDS(path)
        attr(Y_init, "fastEmbedR_init_cache_file") <- path
        attr(Y_init, "fastEmbedR_init_cache_hit") <- TRUE
    }
    float_init <- is_float32_matrix(Y_init)
    if (!float_init) {
        Y_init <- opentsne_dense_numeric_matrix(Y_init)
    }
    if (nrow(Y_init) != n || ncol(Y_init) != n_components) {
        stop(
            "`Y_init` must have ", n, " rows and ", n_components,
            " columns.",
            call. = FALSE
        )
    }
    finite_init <- if (float_init) {
        isTRUE(float32_all_finite_cpp(Y_init))
    } else {
        all(is.finite(Y_init))
    }
    if (!finite_init) {
        stop("`Y_init` must contain only finite values.", call. = FALSE)
    }
    Y_init
}

make_opentsne_pca_init_from_data <- function(init_data,
                                                n,
                                                n_components,
                                                seed,
                                                backend = "cpu",
                                                n_threads = NULL) {
    x <- opentsne_pca_input_matrix(init_data)
    if (nrow(x) != n) {
        stop("`init_data` must have one row per KNN row.", call. = FALSE)
    }
    make_opentsne_pca_init(
        x,
        n_components = n_components,
        seed = seed,
        backend = backend,
        n_threads = n_threads
    )
}
