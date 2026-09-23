#' Choose a default neighborhood size
#'
#' @param x A data matrix/data frame, or an integer row count.
#' @param include_self If `TRUE`, return the value to request from `nn()` when
#'   the query points are the data itself.
#' @return An integer neighborhood size.
#' @noRd
auto_k <- function(x, include_self = FALSE) {
    n <- if (length(x) == 1L && is.numeric(x)) {
        as.integer(x)
    } else {
        nrow(x)
    }
    if (length(n) != 1L || is.na(n) || !is.finite(n) || n < 2L) {
        stop("`x` must describe at least two observations.", call. = FALSE)
    }

    k <- if (n < 500L) {
        15L
    } else if (n < 10000L) {
        30L
    } else {
        50L
    }
    k <- max(1L, min(k, n - 1L))
    if (isTRUE(include_self)) k + 1L else k
}

auto_embedding_k <- function(x, method = "tsne", include_self = FALSE) {
    n <- if (length(x) == 1L && is.numeric(x)) {
        as.integer(x)
    } else {
        nrow(x)
    }
    auto_k(n, include_self = include_self)
}

resolve_embedding_metric <- function(metric, data = NULL) {
    match.arg(metric, c("euclidean", "cosine", "correlation", "inner_product"))
}

coerce_embedding_data <- function(data) {
    input_float32 <- is_float32_matrix(data)
    x <- if (input_float32) {
        data
    } else {
        x <- as.matrix(data)
        storage.mode(x) <- "double"
        x
    }
    if (nrow(x) < 2L || ncol(x) < 1L) {
        stop("`data` must have at least two rows and one column.",
            call. = FALSE
        )
    }
    finite <- if (input_float32) {
        float32_all_finite_cpp(x)
    } else {
        all(is.finite(x))
    }
    if (!isTRUE(finite)) {
        stop("`data` must contain only finite values.", call. = FALSE)
    }
    list(data = x, float32 = input_float32)
}

initial_preprocess_metadata <- function(standardize, input_float32) {
    active <- isTRUE(standardize)
    list(
        standardize = active,
        pca_dims = NA_integer_,
        standardize_backend = if (active) "cpu" else "none",
        pca_backend = "none",
        pca_method = "none",
        pca_oversample = NA_integer_,
        pca_power = NA_integer_,
        pca_backend_reason = NA_character_,
        preprocess_backend = if (active) "cpu" else "none",
        preprocess_backend_reason = if (input_float32) {
            "float32_input_preserved"
        } else {
            NA_character_
        }
    )
}

try_accelerated_standardization <- function(x, backend) {
    if (backend == "cpu") {
        return(list(value = NULL, error = NA_character_))
    }
    available <- switch(backend,
        cuda = cuda_metric_available(),
        metal = metal_metric_available(),
        FALSE
    )
    if (!available) {
        return(list(
            value = NULL,
            error = paste0(backend, "_preprocessing_unavailable")
        ))
    }
    fun <- if (backend == "cuda") {
        standardize_cuda_cpp
    } else {
        standardize_metal_cpp
    }
    capture_error(fun(x))
}

standardize_embedding_data <- function(x, metadata, backend, float32) {
    if (!metadata$standardize) {
        return(list(data = x, metadata = metadata))
    }
    if (float32) {
        result <- standardize_float32_cpp(x)
        metadata$standardize_backend <- "cpu_float32"
        metadata$preprocess_backend <- "cpu_float32"
        metadata$preprocess_backend_reason <-
            "native_float32_column_standardization"
        return(list(data = result$data, metadata = metadata))
    }
    accelerated <- try_accelerated_standardization(x, backend)
    if (!is.null(accelerated$value)) {
        metadata$standardize_backend <- backend
        metadata$preprocess_backend <- backend
        return(list(data = accelerated$value$data, metadata = metadata))
    }
    metadata$preprocess_backend_reason <- accelerated$error
    result <- standardize_cpu_cpp(x)
    metadata$standardize_backend <- "cpu"
    metadata$preprocess_backend <- "cpu"
    list(data = result$data, metadata = metadata)
}

validate_embedding_pca_dims <- function(pca_dims, x) {
    if (is.null(pca_dims)) {
        return(NULL)
    }
    pca_dims <- as.integer(pca_dims)
    if (length(pca_dims) != 1L ||
        is.na(pca_dims) ||
        !is.finite(pca_dims) ||
        pca_dims < 1L) {
        stop("`pca_dims` must be NULL or a positive integer.",
            call. = FALSE
        )
    }
    min(pca_dims, nrow(x) - 1L, ncol(x))
}

run_embedding_pca <- function(x, rank, backend, seed) {
    args <- list(
        data = x,
        ncomp = rank,
        center = FALSE,
        scale = FALSE,
        seed = seed
    )
    fun <- switch(backend,
        cuda = fastembedr_cuda_pca,
        metal = fastembedr_metal_rsvd_pca,
        cpu = fastembedr_cpu_rsvd_pca
    )
    if (backend == "cpu") {
        args$n.cores <- 1L
    }
    do.call(fun, args)
}

apply_embedding_pca <- function(x, metadata, pca_dims, backend, seed) {
    rank <- validate_embedding_pca_dims(pca_dims, x)
    if (is.null(rank) || rank < 1L || rank >= ncol(x)) {
        return(list(data = x, metadata = metadata))
    }
    fit <- run_embedding_pca(x, rank, backend, seed)
    metadata$pca_dims <- as.integer(ncol(fit$scores))
    metadata$pca_backend <- fit$backend
    metadata$pca_method <- fit$method
    metadata$pca_oversample <- fit$oversample
    metadata$pca_power <- fit$power
    metadata$pca_backend_reason <- fit$backend_reason
    metadata$preprocess_backend <- combine_preprocess_backends(
        metadata$standardize_backend,
        fit$backend
    )
    list(data = fit$scores, metadata = metadata)
}

prepare_embedding_data <- function(data,
                                    standardize,
                                    pca_dims,
                                    seed,
                                    backend = "cpu") {
    input <- coerce_embedding_data(data)
    metadata <- initial_preprocess_metadata(
        standardize,
        input$float32
    )
    standardized <- standardize_embedding_data(
        input$data,
        metadata,
        backend,
        input$float32
    )
    reduced <- apply_embedding_pca(
        standardized$data,
        standardized$metadata,
        pca_dims,
        backend,
        seed
    )
    list(data = reduced$data, preprocess = reduced$metadata)
}

resolve_preprocess_backend <- function(backend) {
    backend <- as.character(backend)[1L]
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
        return("cpu")
    }
    if (identical(backend, "gpu")) {
        return(resolve_backend_request("gpu", need_embedding = TRUE))
    }
    if (backend %in% c("cuda", "metal")) {
        return(backend)
    }
    "cpu"
}

combine_preprocess_backends <- function(standardize_backend, pca_backend) {
    standardize_backend <- if (is.null(
        standardize_backend
    )) {
        "none"
    } else {
        standardize_backend
    }
    pca_backend <- if (is.null(pca_backend)) "none" else pca_backend
    if (identical(standardize_backend, "none")) {
        return(pca_backend)
    }
    if (identical(pca_backend, "none")) {
        return(standardize_backend)
    }
    if (identical(standardize_backend, pca_backend)) {
        return(standardize_backend)
    }
    paste(standardize_backend, pca_backend, sep = "_")
}

validate_pca_backend <- function(backend) {
    backend <- as.character(backend)[1L]
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
        return("cpu")
    }
    if (identical(backend, "gpu")) {
        return(resolve_backend_request(backend, need_embedding = TRUE))
    }
    if (!backend %in% c("cpu", "cuda", "metal")) {
        stop("`backend` must be one of 'cpu', 'cuda', or 'metal'.",
            call. = FALSE
        )
    }
    backend
}

fastembedr_metal_rsvd_pca <- function(data,
                                      ncomp,
                                      center = TRUE,
                                      scale = FALSE,
                                      seed = 4L) {
    x <- if (is_float32_matrix(data)) {
        data
    } else {
        x <- as.matrix(data)
        storage.mode(x) <- "double"
        x
    }
    dimensions <- fastembedr_matrix_dimensions(x)
    tuning <- fastembedr_rsvd_tuning(
        dimensions[[1L]], dimensions[[2L]], as.integer(ncomp), "metal"
    )
    fit <- pca_tsvd_metal_cpp(
        x,
        as.integer(ncomp),
        isTRUE(center),
        isTRUE(scale),
        as.integer(seed),
        tuning$oversample,
        tuning$power
    )
    out <- list(
        scores = fit$scores,
        loadings = fit$loadings,
        singular_values = fit$singular_values,
        center = fit$center,
        scale = fit$scale,
        ncomp = as.integer(ncol(fit$scores)),
        method = fit$method,
        backend = fit$backend,
        selection = "fixed_rsvd",
        backend_reason = NA_character_,
        engine = "native_metal_mps",
        precision = fit$precision,
        oversample = as.integer(fit$oversample),
        power = as.integer(fit$power),
        seed = as.integer(seed),
        timing = fit$timing
    )
    class(out) <- "fastEmbedR_pca"
    out
}

fastembedr_metal_tsvd_pca <- fastembedr_metal_rsvd_pca

fastembedr_cuda_pca <- function(data,
                                ncomp,
                                center = TRUE,
                                scale = FALSE,
                                seed = 4L,
                                method = c("auto", "rsvd", "tsvd")) {
    method <- match.arg(method)
    dimensions <- fastembedr_matrix_dimensions(data)
    tuning <- fastembedr_rsvd_tuning(
        dimensions[[1L]], dimensions[[2L]], as.integer(ncomp), "cuda"
    )
    method_code <- match(method, c("auto", "rsvd", "tsvd")) - 1L
    fit <- pca_tsvd_cuda_cpp(
        data,
        as.integer(ncomp),
        isTRUE(center),
        isTRUE(scale),
        as.integer(seed),
        as.integer(method_code),
        tuning$oversample,
        tuning$power
    )
    colnames(fit$scores) <- paste0("PC", seq_len(ncol(fit$scores)))
    colnames(fit$loadings) <- paste0("PC", seq_len(ncol(fit$loadings)))
    out <- list(
        scores = fit$scores,
        loadings = fit$loadings,
        singular_values = fit$singular_values,
        center = fit$center,
        scale = fit$scale,
        ncomp = as.integer(ncol(fit$scores)),
        method = fit$method,
        backend = fit$backend,
        selection = fit$selection,
        backend_reason = NA_character_,
        engine = if (identical(fit$method, "rsvd")) {
            "native_cuda_rsvd"
        } else {
            "native_cuda_raft"
        },
        precision = fit$precision,
        oversample = fit$oversample,
        power = fit$power,
        seed = as.integer(seed),
        timing = fit$timing
    )
    class(out) <- "fastEmbedR_pca"
    out
}

fastembedr_cuda_tsvd_pca <- fastembedr_cuda_pca

attach_opentsne_pca_init <- function(fit, requested) {
    if (!is.logical(requested) || length(requested) != 1L || is.na(requested)) {
        stop("`tsne_init` must be TRUE or FALSE.", call. = FALSE)
    }
    if (!isTRUE(requested)) {
        return(fit)
    }

    init <- normalize_opentsne_pca_scores(fit$scores, fit$ncomp)
    attr(init, "fastEmbedR_init_method") <- paste0("pca_", fit$method)
    attr(init, "fastEmbedR_init_backend") <- fit$backend
    attr(init, "fastEmbedR_init_backend_reason") <-
        fit$backend_reason %||% NA_character_
    attr(init, "fastEmbedR_init_seed") <- fit$seed
    fit$tsne_init <- init
    fit
}

prepare_pca_test_matrix <- function(xtest, fit) {
    use_float32 <- is_float32_matrix(xtest) && is_float32_matrix(fit$loadings)
    x_test <- if (use_float32) {
        if (!requireNamespace("float", quietly = TRUE)) {
            stop("The float package is required to project float32 `xtest`.",
                call. = FALSE
            )
        }
        xtest
    } else if (is_float32_matrix(xtest)) {
        float::dbl(xtest)
    } else {
        x <- as.matrix(xtest)
        storage.mode(x) <- "double"
        x
    }
    if (nrow(x_test) < 1L || ncol(x_test) != nrow(fit$loadings)) {
        stop(
            "`xtest` must have at least one row and the same number ",
            "of columns as `x`.",
            call. = FALSE
        )
    }
    finite <- if (use_float32) {
        isTRUE(float32_all_finite_cpp(x_test))
    } else {
        all(is.finite(x_test))
    }
    if (!finite) {
        stop("`xtest` must contain only finite values.", call. = FALSE)
    }
    list(data = x_test, float32 = use_float32)
}

project_pca_float_scores <- function(x, fit) {
    center <- float::fl(matrix(as.numeric(fit$center), nrow = 1L))
    scale <- float::fl(matrix(as.numeric(fit$scale), nrow = 1L))
    x <- float::sweep(x, 2L, center, "-", check.margin = FALSE)
    x <- float::sweep(x, 2L, scale, "/", check.margin = FALSE)
    x %*% fit$loadings
}

project_pca_double_scores <- function(x, fit) {
    loadings <- if (is_float32_matrix(fit$loadings)) {
        float::dbl(fit$loadings)
    } else {
        as.matrix(fit$loadings)
    }
    x <- sweep(
        x, 2L, as.numeric(fit$center), "-",
        check.margin = FALSE
    )
    x <- sweep(
        x, 2L, as.numeric(fit$scale), "/",
        check.margin = FALSE
    )
    x %*% loadings
}

project_pca_test_scores <- function(xtest, fit) {
    input <- prepare_pca_test_matrix(xtest, fit)
    scores <- if (input$float32) {
        project_pca_float_scores(input$data, fit)
    } else {
        project_pca_double_scores(input$data, fit)
    }
    colnames(scores) <- colnames(fit$scores)
    scores
}

finalize_pca_fit <- function(fit, xtest, tsne_init) {
    if (!is.null(xtest)) {
        fit$scores_test <- project_pca_test_scores(xtest, fit)
    }
    attach_opentsne_pca_init(fit, tsne_init)
}

normalize_pca_threads <- function(n.cores) {
    n.cores <- integer_scalar(n.cores)
    if (length(n.cores) != 1L || is.na(n.cores) ||
        !is.finite(n.cores) || n.cores < 1L) {
        stop("`n.cores` must be a positive integer.", call. = FALSE)
    }
    n.cores
}

set_pca_thread_environment <- function(n_threads) {
    variables <- c(
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RCPP_PARALLEL_NUM_THREADS"
    )
    previous <- Sys.getenv(variables, unset = NA_character_)
    values <- stats::setNames(
        rep(list(as.character(n_threads)), length(variables)),
        variables
    )
    do.call(Sys.setenv, values)
    function() restore_pca_thread_environment(previous)
}

restore_pca_thread_environment <- function(previous) {
    for (name in names(previous)) {
        value <- previous[[name]]
        if (is.na(value)) {
            Sys.unsetenv(name)
        } else {
            do.call(Sys.setenv, stats::setNames(list(value), name))
        }
    }
    invisible(NULL)
}

set_pca_rhpc_threads <- function(n_threads) {
    if (!requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        return(list(
            control = "environment",
            effective = NA_integer_,
            restore = function() invisible(NULL)
        ))
    }
    previous_blas <- tryCatch(
        RhpcBLASctl::blas_get_num_procs(),
        error = function(e) NA_integer_
    )
    previous_omp <- tryCatch(
        RhpcBLASctl::omp_get_max_threads(),
        error = function(e) NA_integer_
    )
    try(RhpcBLASctl::blas_set_num_threads(n_threads), silent = TRUE)
    try(RhpcBLASctl::omp_set_num_threads(n_threads), silent = TRUE)
    effective <- tryCatch(
        as.integer(RhpcBLASctl::blas_get_num_procs()),
        error = function(e) NA_integer_
    )
    list(
        control = "RhpcBLASctl+environment",
        effective = effective,
        restore = function() {
            restore_pca_rhpc_threads(previous_blas, previous_omp)
        }
    )
}

restore_pca_rhpc_threads <- function(blas, omp) {
    if (length(blas) == 1L && !is.na(blas)) {
        try(RhpcBLASctl::blas_set_num_threads(blas), silent = TRUE)
    }
    if (length(omp) == 1L && !is.na(omp)) {
        try(RhpcBLASctl::omp_set_num_threads(omp), silent = TRUE)
    }
    invisible(NULL)
}

with_pca_cpu_threads <- function(n_threads, code) {
    n_threads <- normalize_pca_threads(n_threads)
    restore_env <- set_pca_thread_environment(n_threads)
    on.exit(restore_env(), add = TRUE)
    control <- set_pca_rhpc_threads(n_threads)
    on.exit(control$restore(), add = TRUE)
    list(
        value = force(code),
        control = control$control,
        effective = control$effective
    )
}

prepare_cpu_pca_input <- function(data, ncomp) {
    x <- if (is_float32_matrix(data)) {
        data
    } else {
        x <- as.matrix(data)
        if (!is.numeric(x) && !is.integer(x)) {
            storage.mode(x) <- "double"
        }
        x
    }
    dimensions <- fastembedr_matrix_dimensions(x)
    if (length(dimensions) != 2L || dimensions[[1L]] < 2L ||
        dimensions[[2L]] < 1L) {
        stop("`data` must have at least two rows and one column.",
            call. = FALSE
        )
    }
    rank <- min(
        as.integer(ncomp),
        as.integer(dimensions[[1L]] - 1L),
        as.integer(dimensions[[2L]])
    )
    if (rank < 1L) stop("`data` has no usable PCA rank.", call. = FALSE)
    list(data = x, dimensions = dimensions, rank = rank)
}

cpu_pca_sketch <- function(input, seed) {
    tuning <- fastembedr_rsvd_tuning(
        input$dimensions[[1L]],
        input$dimensions[[2L]],
        input$rank,
        "cpu"
    )
    sketch_rank <- min(
        as.integer(min(input$dimensions)),
        input$rank + tuning$oversample
    )
    power <- if (sketch_rank >= min(input$dimensions)) {
        0L
    } else {
        tuning$power
    }
    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    omega <- matrix(
        stats::rnorm(input$dimensions[[2L]] * sketch_rank),
        nrow = input$dimensions[[2L]],
        ncol = sketch_rank
    )
    list(omega = omega, power = power)
}

finish_cpu_pca_fit <- function(fit, seed) {
    colnames(fit$scores) <- paste0("PC", seq_len(ncol(fit$scores)))
    colnames(fit$loadings) <- paste0(
        "PC", seq_len(ncol(fit$loadings))
    )
    out <- list(
        scores = fit$scores,
        loadings = fit$loadings,
        singular_values = fit$singular_values,
        center = fit$center,
        scale = fit$scale,
        ncomp = as.integer(ncol(fit$scores)),
        method = "rsvd",
        backend = "cpu_rsvd",
        backend_reason = NA_character_,
        engine = "native_cpu_float32",
        precision = "float32",
        oversample = as.integer(fit$oversample),
        power = as.integer(fit$power),
        seed = as.integer(seed),
        timing = fit$timing
    )
    class(out) <- "fastEmbedR_pca"
    out
}

fastembedr_cpu_rsvd_pca <- function(data,
                                    ncomp,
                                    center = TRUE,
                                    scale = FALSE,
                                    seed = 4L,
                                    n.cores = 1L) {
    input <- prepare_cpu_pca_input(data, ncomp)
    sketch <- cpu_pca_sketch(input, seed)
    fit <- pca_rsvd_cpu_cpp(
        input$data,
        input$rank,
        isTRUE(center),
        isTRUE(scale),
        sketch$omega,
        sketch$power,
        normalize_pca_threads(n.cores)
    )
    finish_cpu_pca_fit(fit, seed)
}

run_pca_backend <- function(x, ncomp, center, scale, backend,
                            n_threads, seed) {
    args <- list(
        data = x,
        ncomp = ncomp,
        center = center,
        scale = scale,
        seed = seed
    )
    fun <- switch(backend,
        cpu = fastembedr_cpu_rsvd_pca,
        metal = fastembedr_metal_rsvd_pca,
        cuda = fastembedr_cuda_pca
    )
    if (backend == "cpu") {
        args$n.cores <- n_threads
    }
    do.call(fun, args)
}

validate_pca_request <- function(ncomp, tsne_init, n.cores) {
    if (!is.logical(tsne_init) ||
        length(tsne_init) != 1L ||
        is.na(tsne_init)) {
        stop("`tsne_init` must be TRUE or FALSE.", call. = FALSE)
    }
    ncomp <- as.integer(ncomp)
    if (length(ncomp) != 1L ||
        is.na(ncomp) ||
        !is.finite(ncomp) ||
        ncomp < 1L) {
        stop("`ncomp` must be a positive integer.", call. = FALSE)
    }
    list(
        ncomp = ncomp,
        n_threads = normalize_pca_threads(n.cores)
    )
}

#' Backend-native truncated PCA
#'
#' `pca()` computes principal component scores with a backend-native truncated
#' decomposition. CPU uses a package-native float32 blocked randomized SVD
#' (rSVD). Metal uses a package-native float32 block-subspace rSVD whose large
#' matrix products are executed with Metal Performance Shaders. CUDA selects
#' between package-native float32 rSVD and RAPIDS RAFT TSVD from the matrix
#' shape, requested rank, and estimated arithmetic cost. Wide, low-rank
#' problems favor rSVD; smaller, narrower, or less-truncated problems favor
#' TSVD. CUDA requests fail explicitly when the required native support is
#' unavailable; they never fall back to CPU PCA.
#'
#' CPU matrix products and factorizations use the BLAS/LAPACK linked to R.
#' Set `n.cores` to control their CPU core limit. The requested value is
#' applied temporarily through standard BLAS/OpenMP environment variables and,
#' when installed, `RhpcBLASctl`; the prior process settings are restored after
#' the call. A single-threaded BLAS can still report one effective thread. With
#' `float::float32` input, native Metal and CUDA preprocessing and the returned
#' scores/loadings also remain float32. The API intentionally has no
#' decomposition method menu and does not call Python. Explicit unavailable GPU
#' requests fail instead of being reported as GPU work.
#'
#' @param x Numeric matrix/data frame or `float::float32` matrix with
#'   observations in rows.
#' @param ncomp Number of principal components.
#' @param xtest Optional test/query matrix with the same columns as `x`.
#'   Projected coordinates are returned in `scores_test`.
#' @param center If `TRUE`, mean-center columns before decomposition.
#' @param scale If `TRUE`, scale centered columns to unit sample standard
#'   deviation before decomposition.
#' @param backend PCA backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Positive integer CPU core limit. It controls the linked
#'   BLAS/OpenMP numerical kernels when `backend = "cpu"` and is ignored by
#'   Metal and CUDA.
#' @param seed Random seed for backends that use a Gaussian subspace sketch.
#'   The RAFT covariance-eigensolver route records this value but does not
#'   consume random numbers.
#' @param tsne_init If `TRUE`, add `tsne_init` to the returned PCA
#'   object. This matrix is centered and rescaled so its largest component
#'   standard deviation is `1e-4`, ready to pass as `Y_init` to [tsne()]
#'   or [tsne_knn()].
#' @return A `fastEmbedR_pca` list with `scores`, `loadings`,
#'   `singular_values`, centering/scaling vectors, backend metadata, and
#'   decomposition metadata. When `tsne_init = TRUE`, the list also
#'   contains `tsne_init`; when `xtest` is supplied, it also contains
#'   `scores_test`. Metal and CUDA preserve `float::float32` scores, loadings,
#'   initialization, and compatible test projections when the input is
#'   float32. CPU results also record `n.cores_requested`,
#'   `n.cores_effective`, and `core_control`.
#' @examples
#' fit <- pca(
#'     as.matrix(iris[, 1:4]),
#'     ncomp = 2,
#'     n.cores = 2,
#'     seed = 1,
#'     tsne_init = TRUE
#' )
#' plot(fit$scores, pch = 21, bg = iris$Species)
#' plot(fit$tsne_init, pch = 21, bg = iris$Species)
#' @export

pca <- function(x,
                ncomp = 2L,
                xtest = NULL,
                center = TRUE,
                scale = FALSE,
                backend = NULL,
                n.cores = 1L,
                seed = 4L,
                tsne_init = FALSE) {
    backend <- validate_pca_backend(resolve_embedding_backend(backend))
    request <- validate_pca_request(ncomp, tsne_init, n.cores)
    run_pca <- function() {
        fit <- run_pca_backend(
            x, request$ncomp, center, scale, backend,
            request$n_threads, seed
        )
        finalize_pca_fit(fit, xtest, tsne_init)
    }
    if (!identical(backend, "cpu")) {
        return(run_pca())
    }
    threaded <- with_pca_cpu_threads(request$n_threads, run_pca())
    fit <- threaded$value
    fit$n.cores_requested <- request$n_threads
    fit$n.cores_effective <- threaded$effective
    fit$core_control <- threaded$control
    fit
}

fastembedr_rsvd_tuning <- function(n, p, rank, backend) {
    backend <- if (is.null(backend)) "cpu" else backend
    if (identical(backend, "cuda")) {
        oversample <- 16L
        power <- 2L
    } else if (identical(backend, "metal")) {
        oversample <- if (n * p >= 5e6) 16L else 10L
        power <- if (rank <= 20L) 1L else 2L
    } else {
        oversample <- 20L
        power <- if (rank <= 20L) 1L else 2L
    }
    list(
        oversample = as.integer(max(0L, oversample)),
        power = as.integer(max(0L, power))
    )
}

coerce_supplied_knn <- function(nn) {
    if (!all(c("indices", "distances") %in% names(nn))) {
        stop("`nn` must contain `indices` and `distances`.", call. = FALSE)
    }
    indices <- nn$indices
    distances <- nn$distances
    if (!is.matrix(indices)) indices <- as.matrix(indices)
    distance_is_float32 <- is_float32_matrix(distances)
    if (!distance_is_float32 && !is.matrix(distances)) {
        distances <- as.matrix(
            distances
        )
    }
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    if (!distance_is_float32 && !identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    list(
        indices = indices,
        distances = distances,
        float32 = distance_is_float32
    )
}

validate_supplied_knn <- function(knn, n) {
    indices <- knn$indices
    distances <- knn$distances
    if (!identical(dim(indices), dim(distances))) {
        stop("KNN `indices` and `distances` must have the same dimensions.",
            call. = FALSE
        )
    }
    if (nrow(indices) != n) {
        stop("KNN matrix row count must match `nrow(data)`.", call. = FALSE)
    }
    if (ncol(indices) < 1L) {
        stop("KNN matrices must have at least one neighbor column.",
            call. = FALSE
        )
    }
    if (!knn$float32 && (any(!is.finite(distances)) || any(
        distances < 0
    ))) {
        stop("KNN `distances` must be finite and non-negative.", call. = FALSE)
    }
    invisible(NULL)
}

strip_supplied_self_neighbors <- function(knn, keep_self) {
    stripped <- if (knn$float32) {
        strip_self_neighbors_float_cpp(knn$indices, knn$distances)
    } else {
        strip_self_neighbors(knn$indices, knn$distances)
    }
    has_self <- stripped$has_self
    knn_with_self <- if (isTRUE(keep_self) && has_self) {
        list(indices = knn$indices, distances = knn$distances)
    } else {
        NULL
    }
    if (has_self) {
        knn$indices <- stripped$indices
        knn$distances <- stripped$distances
    }
    if (ncol(knn$indices) < 1L) {
        stop("KNN matrices must contain at least one non-self neighbor.",
            call. = FALSE
        )
    }
    knn$has_self <- isTRUE(has_self)
    knn$knn_with_self <- knn_with_self
    knn
}

trim_supplied_knn <- function(knn, n_neighbors) {
    if (is.null(n_neighbors)) {
        n_neighbors <- ncol(knn$indices)
    } else {
        n_neighbors <- as.integer(n_neighbors)
        invalid_n_neighbors <- length(n_neighbors) != 1L ||
            is.na(n_neighbors) ||
            !is.finite(n_neighbors) ||
            n_neighbors < 1L
        if (invalid_n_neighbors) {
            stop("`n_neighbors` must be NULL or a positive integer.",
                call. = FALSE
            )
        }
        if (n_neighbors > ncol(knn$indices)) {
            stop("`n_neighbors` is larger than the supplied KNN width.",
                call. = FALSE
            )
        }
        knn$indices <- knn$indices[, seq_len(n_neighbors), drop = FALSE]
        knn$distances <- knn$distances[
            , seq_len(n_neighbors),
            drop = FALSE
        ]
    }
    list(
        indices = knn$indices,
        distances = knn$distances,
        n_neighbors = as.integer(n_neighbors),
        has_self = knn$has_self,
        knn_with_self = knn$knn_with_self
    )
}

normalize_supplied_knn <- function(
    nn, n, n_neighbors = NULL,
    keep_self = FALSE
) {
    knn <- coerce_supplied_knn(nn)
    validate_supplied_knn(knn, n)
    knn <- strip_supplied_self_neighbors(knn, keep_self)
    trim_supplied_knn(knn, n_neighbors)
}

embedding_label_info <- function(labels) {
    labels_factor <- if (is.null(labels)) NULL else as.factor(labels)
    labels_int <- if (is.null(labels_factor)) {
        NULL
    } else {
        as.integer(
            labels_factor
        )
    }
    levels <- if (is.null(labels_factor)) {
        0L
    } else {
        length(levels(labels_factor))
    }
    list(integer = labels_int, n_levels = levels)
}

embedding_silhouette_result <- function(layout, labels, keep, backend) {
    if (is.null(labels$integer) || !length(keep)) {
        return(list(
            value = NA_real_,
            backend = if (is.null(labels$integer)) "none" else "skipped",
            reason = NA_character_
        ))
    }
    silhouette_score_with_backend(
        labels$integer[keep],
        layout[keep, , drop = FALSE],
        labels$n_levels,
        backend = backend
    )
}

embedding_score_frame <- function(silhouette, structure) {
    values <- structure$values
    out <- data.frame(
        silhouette = silhouette$value,
        knn_preservation = unname(values["knn_preservation"]),
        local_trustworthiness = unname(values["local_trustworthiness"]),
        local_continuity = unname(values["local_continuity"]),
        structure_score = unname(values["structure_score"]),
        embedding_knn_accuracy = unname(values["embedding_knn_accuracy"]),
        stringsAsFactors = FALSE
    )
    attr(out, "structure_backend") <- structure$backend
    attr(out, "structure_backend_reason") <- structure$reason
    attr(out, "silhouette_backend") <- silhouette$backend
    attr(out, "silhouette_backend_reason") <- silhouette$reason
    attr(out, "backend") <- paste(
        paste0("structure:", structure$backend),
        paste0("silhouette:", silhouette$backend),
        sep = ";"
    )
    out
}

embedding_scores <- function(layout, labels, indices,
                                silhouette_sample, preserve_sample,
                                preserve_k, seed, preserve_keep = NULL,
                                backend = "cpu") {
    label_info <- embedding_label_info(labels)
    silhouette_keep <- sample_indices(
        nrow(layout), silhouette_sample, seed
    )
    if (is.null(preserve_keep)) {
        preserve_keep <- sample_indices(
            nrow(layout), preserve_sample, seed
        )
    }
    preserve_k <- if (is.null(preserve_k)) {
        ncol(indices)
    } else {
        min(as.integer(preserve_k), ncol(indices))
    }

    silhouette <- embedding_silhouette_result(
        layout, label_info, silhouette_keep, backend
    )
    structure <- structure_score_with_backend(
        layout,
        indices,
        preserve_keep,
        preserve_k,
        label_info$integer %||% integer(0L),
        label_info$n_levels,
        backend = backend
    )
    embedding_score_frame(silhouette, structure)
}

auto_landmark_count <- function(n) {
    if (n <= 3L) {
        return(n)
    }
    if (n <= 80L) {
        return(as.integer(min(n - 1L, max(2L, ceiling(n * 0.5)))))
    }
    if (n <= 1000L) {
        return(as.integer(min(n - 1L, max(80L, ceiling(sqrt(n) * 7)))))
    }
    as.integer(min(n - 1L, max(300L, ceiling(sqrt(n) * 7))))
}

landmark_argument_error <- function() {
    stop(
        "`landmarks` must be NULL, TRUE, a positive count, ",
        "a fraction in (0, 1), or row indices.",
        call. = FALSE
    )
}

resolve_landmark_count <- function(value, n) {
    value <- as.numeric(value)
    if (!is.finite(value) || value <= 0) {
        landmark_argument_error()
    }
    count <- if (value < 1) {
        ceiling(n * value)
    } else {
        as.integer(round(value))
    }
    if (count < 2L) {
        stop("Landmark mode requires at least two landmarks.",
            call. = FALSE
        )
    }
    if (count >= n) NULL else count
}

resolve_landmark_indices <- function(landmarks, n) {
    if (!is.numeric(landmarks)) {
        landmark_argument_error()
    }
    idx <- sort(unique(as.integer(landmarks)))
    invalid <- length(idx) < 2L ||
        any(is.na(idx)) ||
        any(idx < 1L) ||
        any(idx > n)
    if (invalid) {
        stop(
            "Landmark row indices must contain at least two valid row numbers.",
            call. = FALSE
        )
    }
    if (length(idx) >= n) NULL else idx
}

resolve_landmarks <- function(landmarks, x, seed, n_threads = NULL) {
    n <- nrow(x)
    if (is.null(landmarks) || identical(landmarks, FALSE)) {
        return(NULL)
    }
    if (identical(landmarks, TRUE)) {
        count <- auto_landmark_count(n)
        return(select_landmark_rows(x, count, seed, n_threads))
    }
    if (length(landmarks) == 1L && is.numeric(landmarks)) {
        count <- resolve_landmark_count(landmarks, n)
        if (is.null(count)) {
            return(NULL)
        }
        return(select_landmark_rows(x, count, seed, n_threads))
    }
    resolve_landmark_indices(landmarks, n)
}

select_landmark_rows <- function(x, count, seed, n_threads = NULL) {
    n <- nrow(x)
    count <- as.integer(min(max(2L, count), n))
    if (count >= n) {
        return(seq_len(n))
    }

    z <- landmark_selection_features(x, seed, n_threads = n_threads)
    if (count <= 2000L) {
        candidate_count <- min(n, max(count, min(12000L, max(500L, 4L *
            count))))
        candidates <- projection_quantile_rows(z, candidate_count, seed)
        picked <- farthest_landmark_subset(z[candidates, , drop = FALSE], count)
        selected <- candidates[picked]
        method <- "projected_farthest"
    } else {
        selected <- projection_quantile_rows(z, count, seed)
        method <- "multi_projection_quantiles"
    }
    selected <- sort(unique(as.integer(selected)))
    if (length(selected) < count) {
        selected <- fill_landmark_rows(selected, n, count, seed)
    }
    selected <- sort(selected[seq_len(count)])
    attr(selected, "selection_method") <- method
    selected
}

landmark_selection_features <- function(x, seed, n_threads = NULL) {
    n <- nrow(x)
    p <- ncol(x)
    n_direct <- min(4L, p)
    n_random <- min(4L, p)

    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    directions <- matrix(stats::rnorm(p * n_random), nrow = p, ncol = n_random)
    norms <- sqrt(colSums(directions * directions))
    norms[!is.finite(norms) | norms == 0] <- 1
    directions <- sweep(directions, 2L, norms, "/")
    if (is_float32_matrix(x)) {
        z <- landmark_projection_float32_cpp(
            x,
            directions,
            n_direct = n_direct,
            n_threads = as.integer(normalize_nn_threads(n_threads))
        )
    } else {
        x <- as.matrix(x)
        direct <- x[, seq_len(n_direct), drop = FALSE]
        z <- cbind(direct, x %*% directions)
    }
    z <- as.matrix(z)
    storage.mode(z) <- "double"

    center <- colMeans(z)
    z <- sweep(z, 2L, center, "-")
    scale <- sqrt(colSums(z * z) / max(1L, n - 1L))
    keep <- is.finite(scale) & scale > 0
    if (!any(keep)) {
        return(matrix(seq_len(n), ncol = 1L))
    }
    sweep(z[, keep, drop = FALSE], 2L, scale[keep], "/")
}

projection_quantile_rows <- function(z, count, seed) {
    n <- nrow(z)
    count <- as.integer(min(max(1L, count), n))
    n_axes <- ncol(z)
    per_axis <- max(2L, ceiling(1.25 * count / max(1L, n_axes)))
    selected <- integer(0)

    center_order <- order(rowSums(z * z), seq_len(n))
    selected <- c(selected, center_order[1L])
    for (axis in seq_len(n_axes)) {
        ordered <- order(z[, axis], seq_len(n))
        positions <- unique(round(seq(1, n, length.out = per_axis)))
        selected <- c(selected, ordered[positions])
    }
    selected <- unique(as.integer(selected))

    if (length(selected) < count) {
        selected <- fill_landmark_rows(selected, n, count, seed)
    }
    if (length(selected) > count) {
        ordered <- order(z[selected, 1L], selected)
        positions <- unique(round(seq(1, length(selected), length.out = count)))
        selected <- selected[ordered[positions]]
    }
    sort(unique(as.integer(selected)))[seq_len(count)]
}

farthest_landmark_subset <- function(z, count) {
    n <- nrow(z)
    count <- as.integer(min(max(1L, count), n))
    if (count >= n) {
        return(seq_len(n))
    }

    z_norm <- rowSums(z * z)
    selected <- integer(count)
    selected[1L] <- which.min(z_norm)
    min_dist <- rep(Inf, n)

    for (i in seq_len(count)) {
        if (i > 1L) {
            selected[i] <- which.max(min_dist)
        }
        center <- z[selected[i], , drop = FALSE]
        dist <- z_norm + z_norm[selected[i]] - 2 * drop(z %*% t(center))
        min_dist <- pmin(min_dist, pmax(0, dist))
        min_dist[selected[seq_len(i)]] <- -Inf
    }
    selected
}

fill_landmark_rows <- function(selected, n, count, seed) {
    selected <- unique(as.integer(selected))
    if (length(selected) >= count) {
        return(selected[seq_len(count)])
    }
    restore_seed <- set_local_seed(seed + 1009L)
    on.exit(restore_seed(), add = TRUE)
    remaining <- setdiff(seq_len(n), selected)
    need <- count - length(selected)
    c(selected, sort(sample(remaining, need)))
}

sampled_score_indices <- function(x,
                                    keep,
                                    preserve_k,
                                    backend,
                                    n_threads = NULL) {
    n <- nrow(x)
    preserve_k <- as.integer(max(1L, min(preserve_k, n - 1L)))
    keep <- as.integer(keep)
    if (length(keep) == 0L) {
        return(matrix(integer(0L), nrow = 0L, ncol = preserve_k))
    }
    out <- matrix(1L, nrow = length(keep), ncol = preserve_k)
    query_k <- min(n, preserve_k + 1L)
    batch_size <- max(256L, min(length(keep), as.integer(floor(2e6 / max(
        1L,
        query_k
    )))))
    starts <- seq.int(1L, length(keep), by = batch_size)
    for (start in starts) {
        end <- min(length(keep), start + batch_size - 1L)
        rows <- start:end
        batch_keep <- keep[rows]
        raw <- fastembedr_native_query_knn(
            x,
            x[batch_keep, , drop = FALSE],
            k = query_k,
            metric = "euclidean",
            n_threads = n_threads
        )
        for (local_i in seq_along(batch_keep)) {
            row <- unique(raw$indices[local_i, raw$indices[local_i, ] !=
                batch_keep[
                    local_i
                ]])
            if (length(row) < preserve_k) {
                fill <- setdiff(seq_len(n), c(batch_keep[local_i], row))
                row <- c(row, fill)
            }
            out[rows[local_i], ] <- row[seq_len(preserve_k)]
        }
        rm(raw)
        if (length(starts) > 1L) gc(FALSE)
    }
    out
}

#' Print or plot an embedding result
#'
#' These methods summarize a `fastEmbedR_embedding` object or plot its first
#' two layout dimensions.
#'
#' @param x A `fastEmbedR_embedding` object.
#' @param labels Optional labels used to color plotted observations.
#' @param pch Plotting symbol passed to [graphics::plot()].
#' @param bg Point background colors. When `NULL`, integer colors are derived
#'   from `labels`, or white is used when labels are absent.
#' @param col Point border or foreground color.
#' @param xlab,ylab Axis labels.
#' @param main Plot title. `NULL` derives a title from the fitted method.
#' @param ... Additional arguments passed to [graphics::plot()] by the plot
#'   method. The print method ignores them.
#' @return The input object, invisibly.
#' @name fastEmbedR_embedding_methods
NULL

#' @rdname fastEmbedR_embedding_methods
#' @export
print.fastEmbedR_embedding <- function(x, ...) {
    cat("fastEmbedR embedding\n")
    cat("  method: ", x$parameters$method, "\n", sep = "")
    cat("  observations: ", x$parameters$n, "\n", sep = "")
    cat("  dimensions: ", ncol(x$layout), "\n", sep = "")
    cat("  neighbors: ", x$parameters$n_neighbors, "\n", sep = "")
    if (isTRUE(x$parameters$landmark)) {
        cat("  landmarks: ", x$parameters$n_landmarks, "\n", sep = "")
    }
    cat("  embedding backend: ", x$parameters$backend, "\n", sep = "")
    cat("  KNN backend: ", x$parameters$nn_backend, "\n", sep = "")
    cat("  elapsed: ", format(round(x$metrics$elapsed, 3L), nsmall = 3L),
        " sec\n",
        sep = ""
    )
    silhouette <- if ("silhouette" %in% names(x$metrics)) {
        x$metrics$silhouette[[
            1L
        ]]
    } else {
        NA_real_
    }
    if (length(silhouette) == 1L && is.finite(silhouette)) {
        cat("  silhouette: ", format(round(silhouette, 4L), nsmall = 4L), "\n",
            sep = ""
        )
    }
    knn_preservation <- if ("knn_preservation" %in% names(x$metrics)) {
        x$metrics$knn_preservation[[1L]]
    } else {
        NA_real_
    }
    if (length(knn_preservation) == 1L && is.finite(knn_preservation)) {
        cat("  KNN preservation: ", format(round(knn_preservation, 4L),
            nsmall = 4L
        ), "\n", sep = "")
    }
    invisible(x)
}

#' @rdname fastEmbedR_embedding_methods
#' @export
plot.fastEmbedR_embedding <- function(x,
                                        labels = x$labels,
                                        pch = 21,
                                        bg = NULL,
                                        col = "black",
                                        xlab = "Component 1",
                                        ylab = "Component 2",
                                        main = NULL,
                                        ...) {
    layout <- embedding_dense_double_matrix(x$layout)
    if (ncol(layout) < 2L) {
        stop("Plotting requires at least two embedding dimensions.",
            call. = FALSE
        )
    }
    if (is.null(main)) {
        main <- paste0("fastEmbedR ", x$parameters$method)
    }
    if (is.null(bg)) {
        bg <- if (is.null(labels)) {
            "white"
        } else {
            as.integer(as.factor(labels))
        }
    }
    graphics::plot(
        layout[, 1L],
        layout[, 2L],
        pch = pch,
        bg = bg,
        col = col,
        xlab = xlab,
        ylab = ylab,
        main = main,
        ...
    )
    invisible(x)
}
