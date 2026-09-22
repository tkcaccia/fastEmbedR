#' Transform query points into an existing t-SNE embedding
#'
#' `transform_tsne()` places query observations into a fixed reference t-SNE
#' embedding. It follows a fixed-reference t-SNE transform design:
#' query-to-reference affinities are computed from KNN distances, query points
#' are initialized from nearby reference points, and optimization moves only the
#' query points.
#'
#' @param reference_layout Numeric reference embedding matrix, or a
#'   `fastEmbedR_embedding` object.
#' @param knn Optional query-to-reference KNN list with `indices` and
#'   `distances`.
#' @param reference_data Reference observations in the same preprocessing space
#'   used to fit `reference_layout`. Required only when `knn` is `NULL`.
#' @param new_data Query observations in the same preprocessing space as
#'   `reference_data`. Required only when `knn` is `NULL`.
#' @param k Number of reference neighbors used for transform KNN. If `NULL`,
#'   uses `ceiling(perplexity)`, matching the package's compact t-SNE support.
#' @param perplexity Transform perplexity. The default is 5.
#' @param initialization Initial query placement: `"median"` from reference
#'   neighbors, inverse-distance `"weighted"`, or `"random"`.
#' @param Y_init Optional initial query embedding matrix. When supplied, it
#'   overrides `initialization`.
#' @param n_iter Number of normal transform optimization iterations.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param learning_rate Transform learning rate.
#' @param early_exaggeration Early exaggeration multiplier.
#' @param exaggeration Normal transform exaggeration multiplier.
#' @param initial_momentum Momentum used during early exaggeration.
#' @param final_momentum Momentum used during normal optimization.
#' @param max_grad_norm Maximum per-query gradient norm. Use `Inf` to disable.
#' @param max_step_norm Maximum per-query step norm. Use `Inf` to disable.
#' @param n_negatives Number of sampled reference points for repulsion when
#'   the reference set is larger than `exact_repulsion_threshold`. GPU sampled
#'   repulsion is native and experimental for large reference sets.
#' @param exact_repulsion_threshold Use exact query-reference repulsion at or
#'   below this reference count.
#' @param n.cores Number of CPU cores for the native optimizer.
#' @param seed Random seed.
#' @param backend Backend used for query KNN when `knn` is `NULL`; `"metal"`
#'   and `"cuda"` run their native fixed-reference transform optimizers when
#'   those backends were compiled. Explicit unavailable GPU requests fail; CPU
#'   work is never reported as Metal or CUDA.
#' @param verbose Print native optimizer progress.
#' @return A numeric matrix with one row per query observation.
#' @examples
#' reference <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), 4L, 2L, byrow = TRUE)
#' query_knn <- list(
#'     indices = matrix(c(1L, 2L, 3L, 4L), 2L, 2L, byrow = TRUE),
#'     distances = matrix(c(0.1, 0.2, 0.15, 0.25), 2L, 2L, byrow = TRUE)
#' )
#' query_layout <- transform_tsne(
#'     reference,
#'     knn = query_knn, perplexity = 0.5,
#'     n_iter = 2, exact_repulsion_threshold = 10, seed = 1
#' )
#' @export
transform_tsne <- function(
    reference_layout, knn = NULL, reference_data = NULL, new_data = NULL,
    k = NULL, perplexity = 5,
    initialization = c("median", "weighted", "random"), Y_init = NULL,
    n_iter = 250L, early_exaggeration_iter = 0L, learning_rate = 0.1,
    early_exaggeration = 4, exaggeration = 1.5,
    initial_momentum = 0.8, final_momentum = 0.8,
    max_grad_norm = 0.25, max_step_norm = Inf, n_negatives = NULL,
    exact_repulsion_threshold = 4096L, n.cores = NULL, seed = 4L,
    backend = NULL, verbose = FALSE
) {
    request <- list(
        knn = knn, reference_data = reference_data, new_data = new_data,
        k = k, perplexity = perplexity,
        initialization = match.arg(initialization), Y_init = Y_init,
        n_iter = n_iter,
        early_exaggeration_iter = early_exaggeration_iter,
        learning_rate = learning_rate,
        early_exaggeration = early_exaggeration,
        exaggeration = exaggeration, initial_momentum = initial_momentum,
        final_momentum = final_momentum, max_grad_norm = max_grad_norm,
        max_step_norm = max_step_norm, n_negatives = n_negatives,
        exact_repulsion_threshold = exact_repulsion_threshold,
        n_threads = n.cores, seed = seed, backend = backend,
        verbose = verbose
    )
    reference_layout <- prepare_tsne_transform_reference(reference_layout)
    request$backend <- resolve_embedding_backend(request$backend)
    request$optimizer_backend <- resolve_tsne_transform_backend(
        request$backend
    )
    prepared <- prepare_tsne_transform_projection(
        request, reference_layout
    )
    controls <- prepare_tsne_transform_controls(
        request, reference_layout
    )
    init <- prepare_tsne_transform_init(
        request$Y_init, prepared$projection, reference_layout
    )
    native <- run_tsne_transform_optimizer(
        reference_layout, prepared$projection, request, controls, init
    )
    finalize_tsne_transform(
        native, reference_layout, prepared, request, controls
    )
}

prepare_tsne_transform_reference <- function(reference_layout) {
    if (inherits(reference_layout, "fastEmbedR_embedding")) {
        reference_layout <- reference_layout$layout
    }
    transform_embedding_matrix(
        reference_layout, "reference_layout",
        min_rows = 1L
    )
}

validate_tsne_transform_data <- function(
    reference_data, new_data, reference_layout
) {
    reference_data <- transform_embedding_matrix(
        reference_data, "reference_data",
        min_rows = 1L
    )
    new_data <- transform_embedding_matrix(
        new_data, "new_data",
        min_rows = 1L
    )
    if (nrow(reference_data) != nrow(reference_layout)) {
        stop(
            "`reference_data` and `reference_layout` must have the same ",
            "number of rows.",
            call. = FALSE
        )
    }
    if (ncol(reference_data) != ncol(new_data)) {
        stop(
            "`reference_data` and `new_data` must have the same number ",
            "of columns.",
            call. = FALSE
        )
    }
    list(reference = reference_data, query = new_data)
}

prepare_tsne_transform_projection <- function(request, reference_layout) {
    if (!is.null(request$knn)) {
        projection <- transform_projection_knn(
            request$knn, nrow(reference_layout), request$k
        )
        return(list(
            projection = projection,
            backend = attr(request$knn, "backend") %||% "precomputed",
            exact = attr(request$knn, "exact")
        ))
    }
    if (is.null(request$reference_data) || is.null(request$new_data)) {
        stop(
            "Supply either `knn`, or both `reference_data` and `new_data`.",
            call. = FALSE
        )
    }
    data <- validate_tsne_transform_data(
        request$reference_data, request$new_data, reference_layout
    )
    k <- request$k %||% min(
        nrow(data$reference), ceiling(request$perplexity)
    )
    k <- transform_embedding_k(k, nrow(data$reference))
    policy <- fastembedr_query_nn_policy(
        request$optimizer_backend, nrow(data$reference),
        nrow(data$query), ncol(data$reference)
    )
    raw <- fastembedr_native_query_knn(
        data$reference, data$query, k, "euclidean", request$n_threads,
        policy$target_recall, policy$backend, policy$method, FALSE
    )
    list(
        projection = transform_projection_knn(
            raw, nrow(reference_layout), k
        ),
        backend = attr(raw, "backend") %||% "precomputed",
        exact = attr(raw, "exact")
    )
}

validate_tsne_transform_iterations <- function(request) {
    n_iter <- as.integer(request$n_iter)
    early <- as.integer(request$early_exaggeration_iter)
    invalid <- length(n_iter) != 1L || is.na(n_iter) || n_iter < 0L ||
        length(early) != 1L || is.na(early) || early < 0L ||
        n_iter + early < 1L
    if (invalid) {
        stop(
            "Transform iteration counts must be non-negative and sum ",
            "to at least one.",
            call. = FALSE
        )
    }
    list(n_iter = n_iter, early = early)
}

prepare_tsne_transform_controls <- function(request, reference_layout) {
    perplexity <- as.numeric(request$perplexity)
    if (length(perplexity) != 1L || is.na(perplexity) ||
        !is.finite(perplexity) || perplexity <= 0) {
        stop("`perplexity` must be a positive number.", call. = FALSE)
    }
    n_threads <- request$n_threads %||% default_tsne_threads()
    n_threads <- as.integer(n_threads)
    if (length(n_threads) != 1L || is.na(n_threads) ||
        !is.finite(n_threads) || n_threads < 0L) {
        stop("`n.cores` must be NULL or a non-negative integer.",
            call. = FALSE
        )
    }
    iterations <- validate_tsne_transform_iterations(request)
    threshold <- as.integer(request$exact_repulsion_threshold)
    if (length(threshold) != 1L || is.na(threshold)) threshold <- 4096L
    n_negatives <- request$n_negatives
    if (is.null(n_negatives)) {
        n_negatives <- if (nrow(reference_layout) <= threshold) {
            nrow(reference_layout)
        } else {
            min(256L, nrow(reference_layout))
        }
    }
    n_negatives <- as.integer(n_negatives)
    if (length(n_negatives) != 1L || is.na(n_negatives) ||
        n_negatives < 1L) {
        stop("`n_negatives` must be NULL or a positive integer.",
            call. = FALSE
        )
    }
    list(
        perplexity = perplexity, n_threads = n_threads,
        n_iter = iterations$n_iter, early_iter = iterations$early,
        threshold = threshold, n_negatives = n_negatives
    )
}

prepare_tsne_transform_init <- function(
    Y_init, projection, reference_layout
) {
    if (is.null(Y_init)) {
        return(list(value = matrix(0, 0L, 0L), supplied = FALSE))
    }
    value <- transform_embedding_matrix(
        Y_init, "Y_init",
        min_rows = nrow(projection$indices)
    )
    valid <- nrow(value) == nrow(projection$indices) &&
        ncol(value) == ncol(reference_layout)
    if (!valid) {
        stop(
            "`Y_init` must have one row per query and the same columns as ",
            "`reference_layout`.",
            call. = FALSE
        )
    }
    list(value = value, supplied = TRUE)
}

run_tsne_transform_gpu <- function(
    backend, reference_layout, projection, request, controls, init
) {
    if (ncol(reference_layout) != 2L) {
        stop(
            backend, " t-SNE transform currently supports only ",
            "two-dimensional reference layouts.",
            call. = FALSE
        )
    }
    call <- if (backend == "metal") {
        transform_tsne_metal_cpp
    } else {
        transform_tsne_cuda_cpp
    }
    call(
        reference_layout, projection$indices, projection$distances,
        init$value, init$supplied, request$initialization,
        controls$perplexity, controls$n_iter, controls$early_iter,
        as.numeric(request$learning_rate),
        as.numeric(request$early_exaggeration),
        as.numeric(request$exaggeration),
        as.numeric(request$initial_momentum),
        as.numeric(request$final_momentum),
        as.numeric(request$max_grad_norm),
        as.numeric(request$max_step_norm), controls$n_negatives,
        controls$threshold, as.integer(request$seed)
    )
}

run_tsne_transform_cpu <- function(
    reference_layout, projection, request, controls, init
) {
    transform_tsne_cpp(
        reference_layout, projection$indices, projection$distances,
        init$value, init$supplied, request$initialization,
        controls$perplexity, controls$n_iter, controls$early_iter,
        as.numeric(request$learning_rate),
        as.numeric(request$early_exaggeration),
        as.numeric(request$exaggeration),
        as.numeric(request$initial_momentum),
        as.numeric(request$final_momentum),
        as.numeric(request$max_grad_norm),
        as.numeric(request$max_step_norm), controls$n_negatives,
        controls$threshold, controls$n_threads, as.integer(request$seed),
        isTRUE(request$verbose)
    )
}

run_tsne_transform_optimizer <- function(
    reference_layout, projection, request, controls, init
) {
    if (request$optimizer_backend %in% c("metal", "cuda")) {
        return(run_tsne_transform_gpu(
            request$optimizer_backend, reference_layout, projection,
            request, controls, init
        ))
    }
    run_tsne_transform_cpu(
        reference_layout, projection, request, controls, init
    )
}

tsne_transform_config <- function(
    out, layout, reference_layout, prepared, request, controls
) {
    list(
        method = "transform_tsne", backend = request$optimizer_backend,
        backend_requested = request$backend,
        nn_backend = prepared$backend,
        n_reference = nrow(reference_layout), n_query = nrow(layout),
        k = as.integer(ncol(prepared$projection$indices)),
        perplexity = controls$perplexity, n_iter = controls$n_iter,
        early_exaggeration_iter = controls$early_iter,
        learning_rate = as.numeric(request$learning_rate),
        early_exaggeration = as.numeric(request$early_exaggeration),
        exaggeration = as.numeric(request$exaggeration),
        initial_momentum = as.numeric(request$initial_momentum),
        final_momentum = as.numeric(request$final_momentum),
        max_grad_norm = as.numeric(request$max_grad_norm),
        max_step_norm = as.numeric(request$max_step_norm),
        n_negatives = out$n_negatives,
        exact_repulsion_threshold = controls$threshold,
        initialization = out$initialization, optimizer = out$optimizer,
        repulsion = out$repulsion, affinities = out$affinities,
        affinity_storage = out$affinity_storage,
        transform_batch_size = out$transform_batch_size,
        transform_batches = out$transform_batches,
        n.cores = out$n_threads %||% NA_integer_,
        seed = as.integer(request$seed),
        provenance = "fixed_reference_tsne_native_cpp"
    )
}

finalize_tsne_transform <- function(
    out, reference_layout, prepared, request, controls
) {
    layout <- out$Y
    colnames(layout) <- colnames(reference_layout)
    attr(layout, "backend") <- request$optimizer_backend
    attr(layout, "nn_backend") <- prepared$backend
    attr(layout, "exact") <- if (is.null(prepared$exact)) {
        NA
    } else {
        isTRUE(prepared$exact)
    }
    attr(layout, "k") <- as.integer(ncol(prepared$projection$indices))
    attr(layout, "transform") <- "tsne_fixed_reference"
    attr(layout, "fastEmbedR_config") <- tsne_transform_config(
        out, layout, reference_layout, prepared, request, controls
    )
    layout
}

resolve_tsne_transform_backend <- function(backend) {
    backend <- resolve_embedding_backend(backend)
    if (identical(backend, "cpu")) {
        return("cpu")
    }
    if (identical(backend, "metal")) {
        if (!metal_metric_available()) {
            stop(
                "Metal t-SNE transform backend is not available on ",
                "this system.",
                call. = FALSE
            )
        }
        return("metal")
    }
    if (identical(backend, "cuda")) {
        if (!cuda_metric_available()) {
            stop(
                "CUDA t-SNE transform backend is not available on this system.",
                call. = FALSE
            )
        }
        return("cuda")
    }
    stop("Unsupported t-SNE transform backend: ", backend, ".", call. = FALSE)
}

landmark_projection_mode <- function() {
    mode <- getOption("fastEmbedR.landmark_projection", "auto")
    mode <- tolower(as.character(mode)[1L])
    if (!mode %in% c("auto", "exact", "approx")) {
        mode <- "auto"
    }
    mode
}

landmark_projection_min_work <- function() {
    value <- numeric_scalar(getOption(
        "fastEmbedR.landmark_projection_min_work",
        1e10
    ))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0) {
        value <- 1e10
    }
    value
}

landmark_projection_min_rows <- function() {
    value <- integer_scalar(getOption(
        "fastEmbedR.landmark_projection_min_rows",
        2000L
    ))
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        value < 1L) {
        value <- 2000L
    }
    value
}

landmark_projection_approx_params <- function(n_landmarks, k) {
    n_landmarks <- as.integer(n_landmarks)
    k <- as.integer(k)
    n_projections <- getOption(
        "fastEmbedR.landmark_projection_n_projections",
        NULL
    )
    if (is.null(n_projections)) {
        n_projections <- max(8L, min(24L, 2L * ceiling(log2(max(
            2L,
            n_landmarks
        )))))
    } else {
        n_projections <- integer_scalar(n_projections)
        if (length(n_projections) != 1L || is.na(n_projections) || !is.finite(
            n_projections
        )) {
            n_projections <- max(8L, min(24L, 2L * ceiling(log2(max(
                2L,
                n_landmarks
            )))))
        }
    }
    n_projections <- as.integer(max(1L, min(64L, n_projections)))

    window <- getOption("fastEmbedR.landmark_projection_window", NULL)
    if (is.null(window)) {
        window <- max(
            64L,
            20L * k,
            ceiling(0.015 * n_landmarks),
            ceiling(sqrt(max(1L, n_landmarks)))
        )
    } else {
        window <- integer_scalar(window)
        if (length(window) != 1L || is.na(window) || !is.finite(window)) {
            window <- max(
                64L,
                20L * k,
                ceiling(0.015 * n_landmarks),
                ceiling(sqrt(max(1L, n_landmarks)))
            )
        }
    }
    window <- as.integer(max(k, min(n_landmarks, window)))

    list(n_projections = n_projections, window = window)
}

landmark_reference_knn <- function(x_landmarks,
                                    k,
                                    backend,
                                    n_threads,
                                    metric = "euclidean") {
    knn_backend <- embedding_knn_backend(backend)
    fastembedr_nn_without_self(
        x_landmarks,
        k = k,
        backend = knn_backend,
        method = "auto",
        metric = metric,
        output = fastembedr_knn_output_type(x_landmarks, knn_backend),
        n_threads = n_threads,
        tuning = "auto",
        target_recall = 0.99,
        keep_gpu = identical(knn_backend, "cuda")
    )
}

landmark_projection_knn <- function(x_landmarks,
                                    x_query,
                                    k,
                                    backend,
                                    seed,
                                    n_threads = NULL,
                                    landmark_layout = NULL,
                                    all_data = NULL,
                                    landmark_indices = NULL,
                                    query_rows = NULL) {
    backend <- as.character(backend)[1L]
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
        backend <- "cpu"
    }
    n_threads <- normalize_nn_threads(n_threads)
    policy <- fastembedr_query_nn_policy(
        backend,
        n_reference = nrow(x_landmarks),
        n_query = nrow(x_query),
        p = ncol(x_landmarks)
    )
    result <- fastembedr_native_query_knn(
        x_landmarks,
        x_query,
        k = k,
        metric = "euclidean",
        output = fastembedr_knn_output_type(x_landmarks, policy$backend),
        n_threads = n_threads,
        target_recall = policy$target_recall,
        backend = policy$backend,
        method = policy$method,
        keep_gpu = identical(policy$backend, "cuda")
    )
    attr(result, "approximation") <- list(
        strategy = paste0(
            "native_", policy$method, "_reference_query_knn"
        ),
        backend = policy$backend,
        target_recall = policy$target_recall,
        seed = as.integer(seed)
    )
    result
}

annotate_affine_projection <- function(out, backend, threads, method) {
    layout <- out$layout
    attr(layout, "projection_method") <- out$method %||% method
    attr(layout, "projection_backend") <- out$backend %||% backend
    attr(layout, "projection_confidence") <- out$confidence
    attr(layout, "projection_fallback") <- out$fallback
    attr(layout, "projection_max_neighbors") <- out$max_neighbors
    attr(layout, "projection_ridge") <- out$ridge
    attr(layout, "projection_max_extrapolation") <- out$max_extrapolation
    attr(layout, "projection_threads") <- threads
    layout
}

try_metal_affine_projection <- function(
    x_landmarks, x_query, landmark_layout, projection_knn,
    max_neighbors, ridge, max_extrapolation, required
) {
    out <- tryCatch(
        project_embedding_affine_metal_cpp(
            x_landmarks, x_query, landmark_layout,
            projection_knn$indices, projection_knn$distances,
            as.integer(max_neighbors), as.numeric(ridge),
            as.numeric(max_extrapolation)
        ),
        error = function(e) {
            if (required) {
                stop(
                    "Metal affine landmark projection failed: ",
                    conditionMessage(e),
                    call. = FALSE
                )
            }
            NULL
        }
    )
    if (is.null(out) || is.null(out$layout)) {
        return(NULL)
    }
    annotate_affine_projection(
        out, "metal", NA_integer_, "local_affine_knn_projection_metal"
    )
}

run_cpu_affine_projection <- function(
    x_landmarks, x_query, landmark_layout, projection_knn,
    max_neighbors, ridge, max_extrapolation, n_threads
) {
    parallel <- exists(
        "project_embedding_affine_parallel_cpp",
        envir = asNamespace("fastEmbedR"), inherits = FALSE
    )
    call <- if (parallel) {
        project_embedding_affine_parallel_cpp
    } else {
        project_embedding_affine_cpp
    }
    args <- list(
        x_landmarks, x_query, landmark_layout, projection_knn$indices,
        projection_knn$distances, as.integer(max_neighbors),
        as.numeric(ridge), as.numeric(max_extrapolation)
    )
    if (parallel) args <- c(args, list(as.integer(n_threads)))
    out <- tryCatch(do.call(call, args), error = function(e) NULL)
    if (is.null(out) || is.null(out$layout)) {
        layout <- project_embedding_knn_cpp(
            landmark_layout, projection_knn$indices,
            projection_knn$distances
        )
        attr(layout, "projection_method") <- "weighted_knn_fallback"
        attr(layout, "projection_backend") <- "cpu"
        return(layout)
    }
    annotate_affine_projection(
        out, "cpu", out$n_threads %||% 1L,
        "local_affine_knn_projection"
    )
}

landmark_affine_projection <- function(
    x_landmarks, x_query, landmark_layout, projection_knn,
    max_neighbors = NULL, ridge = 1e-3, max_extrapolation = 2.5,
    n_threads = NULL, backend = "auto"
) {
    if (is.null(max_neighbors)) {
        max_neighbors <- min(12L, ncol(projection_knn$indices))
    }
    max_neighbors <- as.integer(max_neighbors)
    if (length(max_neighbors) != 1L || is.na(max_neighbors) ||
        max_neighbors < 3L) {
        max_neighbors <- min(12L, ncol(projection_knn$indices))
    }
    backend <- as.character(backend)[1L]
    metal_work <- as.double(nrow(x_query)) * as.double(ncol(x_query))
    use_metal <- ncol(landmark_layout) == 2L &&
        backend %in% c("metal", "gpu") &&
        isTRUE(embedding_metal_available_cpp()) &&
        metal_work >= 5e6
    if (use_metal) {
        layout <- try_metal_affine_projection(
            x_landmarks, x_query, landmark_layout, projection_knn,
            max_neighbors, ridge, max_extrapolation, backend == "metal"
        )
        if (!is.null(layout)) {
            return(layout)
        }
    }
    run_cpu_affine_projection(
        x_landmarks, x_query, landmark_layout, projection_knn,
        max_neighbors, ridge, max_extrapolation,
        normalize_nn_threads(n_threads)
    )
}

zero_proc_time <- function() {
    structure(rep(0, 5), names = names(system.time({})))
}

scalar_or_default <- function(values, name, default) {
    value <- values[[name]]
    if (is.null(value)) {
        return(default)
    }
    value
}

scalar_numeric_or_default <- function(values, name, default) {
    value <- scalar_or_default(values, name, default)
    if (length(value) != 1L || is.na(value)) {
        return(default)
    }
    if (is.character(value) && identical(tolower(value), "auto")) {
        return(
            default
        )
    }
    out <- numeric_scalar(value)
    if (length(out) != 1L || is.na(out) || !is.finite(out)) default else out
}

scalar_integer_or_default <- function(values, name, default) {
    value <- scalar_numeric_or_default(values, name, default)
    out <- integer_scalar(value)
    if (length(out) != 1L || is.na(out)) as.integer(default) else out
}

resident_transform_backend <- function(backend, k, keep_knn) {
    if (isTRUE(keep_knn)) {
        return(NA_character_)
    }
    backend <- as.character(backend)[1L]
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
        backend <- "auto"
    }
    if (k > 128L) {
        return(NA_character_)
    }
    if (backend %in% c("metal", "gpu") &&
        isTRUE(embedding_metal_available_cpp())) {
        return("metal")
    }
    if (backend %in% c("cuda", "gpu") &&
        isTRUE(embedding_cuda_available_cpp())) {
        return("cuda")
    }
    NA_character_
}

resident_projected_layout <- function(
    resident, backend, backend_requested, n_reference, k, perplexity,
    n_iter, early_exaggeration_iter, learning_rate, early_exaggeration,
    exaggeration, initial_momentum, final_momentum, max_grad_norm,
    max_step_norm, exact_repulsion_threshold, seed, reference_layout
) {
    layout <- resident$Y
    colnames(layout) <- colnames(reference_layout)
    attr(layout, "backend") <- backend
    attr(layout, "nn_backend") <- paste0(backend, "_resident_projection")
    attr(layout, "exact") <- TRUE
    attr(layout, "k") <- as.integer(k)
    attr(layout, "transform") <- "tsne_fixed_reference"
    attr(layout, "fastEmbedR_config") <- list(
        method = "transform_tsne",
        backend = backend,
        backend_requested = backend_requested,
        nn_backend = paste0(backend, "_resident_projection"),
        n_reference = as.integer(n_reference),
        n_query = nrow(layout),
        k = as.integer(k),
        perplexity = perplexity,
        n_iter = as.integer(n_iter),
        early_exaggeration_iter = as.integer(early_exaggeration_iter),
        learning_rate = as.numeric(learning_rate),
        early_exaggeration = as.numeric(early_exaggeration),
        exaggeration = as.numeric(exaggeration),
        initial_momentum = as.numeric(initial_momentum),
        final_momentum = as.numeric(final_momentum),
        max_grad_norm = as.numeric(max_grad_norm),
        max_step_norm = as.numeric(max_step_norm),
        n_negatives = resident$n_negatives,
        exact_repulsion_threshold = as.integer(exact_repulsion_threshold),
        initialization = resident$initialization,
        optimizer = resident$optimizer,
        repulsion = resident$repulsion,
        affinities = resident$affinities,
        affinity_storage = resident$affinity_storage,
        transform_batch_size = NA_integer_,
        transform_batches = NA_integer_,
        n.cores = NA_integer_,
        seed = as.integer(seed),
        resident = TRUE,
        returned_intermediates = resident$returned_intermediates,
        provenance = "fixed_reference_tsne_native_cpp_gpu_resident"
    )
    layout
}

resident_projection_result <- function(backend, k) {
    out <- list(
        indices = matrix(integer(0L), nrow = 0L, ncol = 0L),
        distances = matrix(numeric(0L), nrow = 0L, ncol = 0L)
    )
    result <- finish_nn_result(out, paste0(backend, "_resident_projection"), k,
        FALSE,
        exact = TRUE
    )
    attr(result, "approximation") <- list(
        strategy = "gpu_resident_landmark_projection_transform",
        backend = backend,
        returned_intermediates = "final_layout_only"
    )
    result
}

normalize_landmark_tsne_request <- function(request) {
    request$reference_method <- match.arg(
        request$reference_method,
        "tsne"
    )
    request$initialization <- match.arg(
        request$initialization,
        c("median", "weighted", "random")
    )
    request$backend <- resolve_embedding_backend(request$backend)
    request$n_threads <- request$n.cores
    request
}

prepare_landmark_tsne_data <- function(data, request) {
    prepared <- timed_do_call(prepare_embedding_data, list(
        data,
        request$standardize,
        request$pca_dims,
        request$seed,
        backend = resolve_preprocess_backend(request$backend)
    ))
    x <- prepared$value$data
    validate_landmark_tsne_neighbors(request$n_neighbors, nrow(x))
    selection <- select_landmarks(
        x,
        request$landmarks,
        seed = request$seed,
        n.cores = request$n_threads
    )
    list(
        x = x,
        n = nrow(x),
        prepared = prepared$value,
        preprocess_time = prepared$time,
        selection = selection
    )
}

validate_landmark_tsne_neighbors <- function(n_neighbors, n) {
    if (is.null(n_neighbors)) {
        return(invisible(NULL))
    }
    n_neighbors <- as.integer(n_neighbors)
    invalid <- length(n_neighbors) != 1L ||
        is.na(n_neighbors) ||
        n_neighbors < 1L ||
        n_neighbors >= n
    if (invalid) {
        stop(
            "`n_neighbors` must be positive and smaller than `nrow(data)`.",
            call. = FALSE
        )
    }
    invisible(NULL)
}

run_full_landmark_tsne <- function(state, request) {
    args <- list(
        data = state$x,
        perplexity = request$perplexity,
        n_components = request$n_components,
        standardize = FALSE,
        pca_dims = NULL,
        seed = request$seed,
        backend = request$backend,
        keep_knn = request$keep_knn,
        verbose = request$verbose,
        n.cores = request$n_threads
    )
    do.call(tsne, c(args, request$extra))
}

partition_landmark_tsne <- function(state, request) {
    indices <- state$selection$indices
    query_indices <- state$selection$query_indices
    partition <- split_landmark_data(
        state$x,
        indices,
        query_indices,
        n_threads = request$n_threads
    )
    state$landmark_indices <- indices
    state$query_indices <- query_indices
    state$x_landmarks <- partition$landmarks
    state$x_query <- partition$query
    state$n_landmarks <- nrow(partition$landmarks)
    state
}

landmark_tsne_reference_policy <- function(state, request) {
    policy <- opentsne_neighbor_policy(
        state$n_landmarks,
        perplexity = request$perplexity
    )
    perplexity <- request$perplexity %||% policy$perplexity
    n_neighbors <- request$n_neighbors %||% policy$n_neighbors
    if (n_neighbors >= state$n_landmarks) {
        stop(
            "`n_neighbors` must be smaller than selected landmarks.",
            call. = FALSE
        )
    }
    required <- opentsne_support_width(perplexity)
    if (n_neighbors != required) {
        stop(
            "Compact t-SNE affinity support requires `n_neighbors = ",
            required, "` for this perplexity.",
            call. = FALSE
        )
    }
    list(
        perplexity = perplexity,
        n_neighbors = as.integer(n_neighbors)
    )
}

landmark_tsne_reference_args <- function(knn, state, request, policy) {
    args <- list(
        indices = knn,
        n_neighbors = policy$n_neighbors,
        perplexity = policy$perplexity,
        n_components = request$n_components,
        init_data = state$x_landmarks,
        seed = request$seed,
        backend = request$backend,
        verbose = request$verbose,
        n.cores = request$n_threads
    )
    c(args, request$extra)
}

run_landmark_tsne_reference <- function(state, request, policy) {
    reference_knn <- landmark_reference_knn(
        state$x_landmarks,
        k = policy$n_neighbors,
        backend = request$backend,
        n_threads = request$n_threads
    )
    layout <- do.call(
        tsne_knn,
        landmark_tsne_reference_args(
            reference_knn,
            state,
            request,
            policy
        )
    )
    list(knn = reference_knn, layout = layout)
}

landmark_tsne_reference_fit <- function(result, state, request, policy) {
    config <- attr(result$value$layout, "fastEmbedR_config")
    nn_backend <- attr(result$value$knn, "backend") %||% "supplied"
    parameters <- c(list(
        method = "tsne",
        input = "knn",
        n = state$n_landmarks,
        p = ncol(state$x_landmarks),
        n_neighbors = policy$n_neighbors,
        k = policy$n_neighbors + 1L,
        n_components = as.integer(request$n_components),
        seed = as.integer(request$seed),
        nn_backend = nn_backend,
        keep_knn = request$keep_knn
    ), config, list(preprocess = "none_precomputed_knn"))
    list(
        layout = result$value$layout,
        metrics = landmark_tsne_reference_metrics(
            state,
            policy,
            config,
            result$time
        ),
        parameters = parameters,
        knn = if (request$keep_knn) result$value$knn else NULL
    )
}

landmark_tsne_reference_metrics <- function(
    state,
    policy,
    config,
    elapsed
) {
    data.frame(
        method = "tsne",
        n = state$n_landmarks,
        p = ncol(state$x_landmarks),
        n_neighbors = policy$n_neighbors,
        perplexity = config$perplexity,
        elapsed = elapsed[["elapsed"]],
        preprocess_elapsed = 0,
        knn_elapsed = NA_real_,
        embedding_elapsed = NA_real_,
        stringsAsFactors = FALSE
    )
}

fit_landmark_tsne_reference <- function(state, request) {
    policy <- landmark_tsne_reference_policy(state, request)
    result <- timed_do_call(
        run_landmark_tsne_reference,
        list(state, request, policy)
    )
    state$policy <- policy
    state$reference_time <- result$time
    state$reference_fit <- landmark_tsne_reference_fit(
        result,
        state,
        request,
        policy
    )
    state
}

landmark_tsne_transform_controls <- function(state, request) {
    transform_k <- request$transform_k
    if (is.null(transform_k)) {
        transform_k <- ceiling(request$transform_perplexity)
        transform_k <- min(state$n_landmarks, transform_k)
    }
    transform_iter <- as.integer(request$transform_iter)
    if (length(transform_iter) != 1L ||
        is.na(transform_iter) ||
        transform_iter < 0L) {
        stop(
            "`transform_iter` must be a non-negative integer.",
            call. = FALSE
        )
    }
    threshold <- scalar_integer_or_default(
        request$extra,
        "exact_repulsion_threshold",
        4096L
    )
    if (length(threshold) != 1L || is.na(threshold)) {
        threshold <- 4096L
    }
    landmark_tsne_transform_scalars(
        state,
        request,
        transform_embedding_k(transform_k, state$n_landmarks),
        transform_iter,
        threshold
    )
}

landmark_tsne_transform_scalars <- function(
    state,
    request,
    transform_k,
    transform_iter,
    threshold
) {
    negatives <- request$transform_n_negatives
    if (is.null(negatives)) {
        negatives <- if (state$n_landmarks <= threshold) {
            state$n_landmarks
        } else {
            min(256L, state$n_landmarks)
        }
    }
    list(
        k = transform_k,
        n_iter = transform_iter,
        threshold = threshold,
        n_negatives = as.integer(negatives),
        learning_rate = scalar_numeric_or_default(
            request$extra, "transform_learning_rate", 0.1
        ),
        early_exaggeration = scalar_numeric_or_default(
            request$extra, "transform_early_exaggeration", 4
        ),
        exaggeration = scalar_numeric_or_default(
            request$extra, "transform_exaggeration", 1.5
        ),
        initial_momentum = scalar_numeric_or_default(
            request$extra, "transform_initial_momentum", 0.8
        ),
        final_momentum = scalar_numeric_or_default(
            request$extra, "transform_final_momentum", 0.8
        ),
        max_grad_norm = scalar_numeric_or_default(
            request$extra, "transform_max_grad_norm", 0.25
        ),
        max_step_norm = scalar_numeric_or_default(
            request$extra, "transform_max_step_norm", Inf
        )
    )
}

compute_landmark_tsne_projection_knn <- function(
    state,
    request,
    controls
) {
    landmark_projection_knn(
        state$x_landmarks,
        state$x_query,
        k = controls$k,
        backend = request$backend,
        seed = request$seed + 503L,
        n_threads = request$n_threads,
        landmark_layout = state$reference_fit$layout,
        all_data = state$x,
        landmark_indices = state$landmark_indices,
        query_rows = state$query_indices
    )
}

run_landmark_tsne_cuda_projection <- function(
    state,
    request,
    controls,
    projection_knn
) {
    resident <- landmark_tsne_transform_cuda_gpu_cpp(
        projection_knn,
        state$x_landmarks,
        state$x_query,
        state$reference_fit$layout,
        as.numeric(request$transform_perplexity),
        controls$n_iter,
        as.integer(request$transform_early_exaggeration_iter),
        controls$learning_rate,
        controls$early_exaggeration,
        controls$exaggeration,
        controls$initial_momentum,
        controls$final_momentum,
        controls$max_grad_norm,
        controls$max_step_norm,
        controls$n_negatives,
        controls$threshold,
        as.integer(request$seed + 1009L),
        12L,
        1e-3,
        2.5
    )
    layout <- resident$Y
    attr(layout, "backend") <- "cuda"
    attr(layout, "fastEmbedR_config") <- resident$config
    layout
}

landmark_tsne_projection_init <- function(
    state,
    request,
    projection_knn
) {
    init <- attr(projection_knn, "projected_layout", exact = TRUE)
    affine <- landmark_affine_projection(
        state$x_landmarks,
        state$x_query,
        state$reference_fit$layout,
        projection_knn,
        n_threads = request$n_threads,
        backend = request$backend
    )
    valid_affine <- embedding_layout_dims_match(
        affine,
        length(state$query_indices),
        request$n_components
    )
    if (valid_affine) {
        return(list(
            layout = affine,
            backend = attr(affine, "projection_backend") %||% NA_character_,
            method = attr(affine, "projection_method") %||% NA_character_
        ))
    }
    valid_init <- embedding_layout_dims_match(
        init,
        length(state$query_indices),
        request$n_components
    )
    list(
        layout = if (valid_init) init else NULL,
        backend = NA_character_,
        method = NA_character_
    )
}

landmark_tsne_projection_only <- function(
    state,
    request,
    projection_knn,
    init
) {
    if (is.null(init$layout)) {
        init$layout <- project_embedding_knn_cpp(
            state$reference_fit$layout,
            projection_knn$indices,
            projection_knn$distances
        )
        init$backend <- "cpu"
        init$method <- "weighted_knn_fallback"
    }
    layout <- init$layout
    attr(layout, "backend") <- attr(projection_knn, "backend") %||%
        request$backend
    attr(layout, "fastEmbedR_config") <- list(
        optimizer = "projection_only",
        repulsion = "none",
        n_negatives = 0L,
        initialization = request$initialization,
        backend = attr(layout, "backend")
    )
    list(layout = layout, init = init)
}

landmark_tsne_transform_args <- function(
    state,
    request,
    controls,
    projection_knn,
    init
) {
    list(
        reference_layout = state$reference_fit$layout,
        knn = projection_knn,
        perplexity = request$transform_perplexity,
        initialization = request$initialization,
        Y_init = init$layout,
        n_iter = controls$n_iter,
        early_exaggeration_iter =
            request$transform_early_exaggeration_iter,
        learning_rate = controls$learning_rate,
        early_exaggeration = controls$early_exaggeration,
        exaggeration = controls$exaggeration,
        initial_momentum = controls$initial_momentum,
        final_momentum = controls$final_momentum,
        max_grad_norm = controls$max_grad_norm,
        max_step_norm = controls$max_step_norm,
        n_negatives = request$transform_n_negatives,
        n.cores = request$n_threads,
        seed = request$seed + 1009L,
        backend = request$backend,
        verbose = request$verbose
    )
}

run_landmark_tsne_host_projection <- function(
    state,
    request,
    controls,
    projection_knn
) {
    init <- landmark_tsne_projection_init(
        state,
        request,
        projection_knn
    )
    if (controls$n_iter == 0L) {
        projected <- landmark_tsne_projection_only(
            state,
            request,
            projection_knn,
            init
        )
        return(list(
            layout = projected$layout,
            time = zero_proc_time(),
            init = projected$init
        ))
    }
    transformed <- timed_do_call(
        transform_tsne,
        landmark_tsne_transform_args(
            state,
            request,
            controls,
            projection_knn,
            init
        )
    )
    list(
        layout = transformed$value,
        time = transformed$time,
        init = init
    )
}

project_landmark_tsne <- function(state, request) {
    controls <- landmark_tsne_transform_controls(state, request)
    projection <- timed_do_call(
        compute_landmark_tsne_projection_knn,
        list(state, request, controls)
    )
    projection_knn <- projection$value
    use_cuda <- identical(request$backend, "cuda") &&
        fastembedr_is_gpu_knn(projection_knn)
    if (use_cuda) {
        transformed <- timed_do_call(
            run_landmark_tsne_cuda_projection,
            list(state, request, controls, projection_knn)
        )
        init <- list(
            backend = "cuda",
            method = "local_affine_knn_projection_cuda_resident"
        )
    } else {
        transformed <- run_landmark_tsne_host_projection(
            state,
            request,
            controls,
            projection_knn
        )
        init <- transformed$init
    }
    state$projection <- list(
        knn = projection_knn,
        projected = transformed$layout,
        projection_time = projection$time,
        transform_time = transformed$time,
        init = init,
        controls = controls
    )
    state
}

landmark_tsne_projection_strategy <- function(knn) {
    approximation <- attr(knn, "approximation", exact = TRUE)
    if (!is.null(approximation$strategy)) {
        return(as.character(approximation$strategy))
    }
    if (isTRUE(attr(knn, "exact"))) {
        return("exact")
    }
    NA_character_
}

landmark_tsne_reference_times <- function(state) {
    metrics <- state$reference_fit$metrics
    list(
        total = if ("elapsed" %in% names(metrics)) {
            metrics$elapsed[[1L]]
        } else {
            state$reference_time[["elapsed"]]
        },
        knn = if ("knn_elapsed" %in% names(metrics)) {
            metrics$knn_elapsed[[1L]]
        } else {
            NA_real_
        },
        optimizer = if ("embedding_elapsed" %in% names(metrics)) {
            metrics$embedding_elapsed[[1L]]
        } else {
            NA_real_
        }
    )
}

landmark_tsne_timings <- function(state) {
    rbind(
        preprocess = state$preprocess_time,
        reference_embedding = state$reference_time,
        landmark_projection_knn =
            state$projection$projection_time,
        transform = state$projection$transform_time
    )
}

landmark_tsne_metrics <- function(state, request, timings) {
    reference <- landmark_tsne_reference_times(state)
    data.frame(
        method = "landmark_tsne",
        reference_method = request$reference_method,
        n = state$n,
        p = ncol(state$x),
        n_neighbors = state$policy$n_neighbors,
        perplexity = state$policy$perplexity,
        elapsed = sum(timings[, "elapsed"]),
        preprocess_elapsed = state$preprocess_time[["elapsed"]],
        reference_embedding_elapsed =
            state$reference_time[["elapsed"]],
        reference_total_elapsed = reference$total,
        reference_knn_elapsed = reference$knn,
        reference_optimizer_elapsed = reference$optimizer,
        landmark_projection_knn_elapsed =
            state$projection$projection_time[["elapsed"]],
        transform_elapsed =
            state$projection$transform_time[["elapsed"]],
        landmark = TRUE,
        n_landmarks = state$n_landmarks,
        landmark_fraction = state$n_landmarks / state$n,
        transform_k = state$projection$controls$k,
        stringsAsFactors = FALSE
    )
}

landmark_tsne_reference_parameters <- function(state) {
    params <- state$reference_fit$parameters
    list(
        perplexity = params$perplexity,
        affinity_support = params$affinity_support,
        affinity_support_policy = params$affinity_support_policy,
        affinity_support_k = params$affinity_support_k,
        affinity_support_multiplier =
            params$affinity_support_multiplier,
        compact_affinity_support = params$compact_affinity_support,
        nn_backend = params$nn_backend
    )
}

landmark_tsne_transform_parameters <- function(state, request) {
    config <- attr(
        state$projection$projected,
        "fastEmbedR_config"
    )
    list(
        transform_k = state$projection$controls$k,
        transform_perplexity = request$transform_perplexity,
        transform_iter = state$projection$controls$n_iter,
        transform_early_exaggeration_iter = as.integer(
            request$transform_early_exaggeration_iter
        ),
        transform_optimizer = config$optimizer,
        transform_repulsion = config$repulsion,
        transform_n_negatives = config$n_negatives,
        transform_initialization = config$initialization
    )
}

landmark_tsne_parameters <- function(state, request) {
    projection <- state$projection
    base <- list(
        method = "landmark_tsne",
        reference_method = request$reference_method,
        n = state$n,
        p = ncol(state$x),
        n_neighbors = state$policy$n_neighbors,
        k = state$policy$n_neighbors + 1L,
        n_components = as.integer(request$n_components),
        seed = as.integer(request$seed),
        projection_nn_backend = attr(projection$knn, "backend"),
        projection_strategy =
            landmark_tsne_projection_strategy(projection$knn),
        projection_init_backend = projection$init$backend,
        projection_init_method = projection$init$method,
        backend = attr(projection$projected, "backend"),
        transform_backend = attr(projection$projected, "backend"),
        n.cores = normalize_nn_threads(request$n_threads),
        landmark = TRUE,
        n_landmarks = state$n_landmarks,
        landmark_fraction = state$n_landmarks / state$n,
        landmark_selection = state$selection$method,
        keep_knn = request$keep_knn,
        provenance = "landmark_tsne_fixed_reference_native_cpp"
    )
    c(
        base,
        landmark_tsne_reference_parameters(state),
        landmark_tsne_transform_parameters(state, request),
        state$prepared$preprocess
    )
}

assemble_landmark_tsne_output <- function(state, request) {
    projection <- state$projection
    layout <- assemble_landmark_layout(
        state$reference_fit$layout,
        projection$projected,
        state$landmark_indices,
        state$query_indices,
        state$n,
        prefix = "TSNE",
        return_float32 = is_float32_matrix(state$x)
    )
    timings <- landmark_tsne_timings(state)
    out <- list(
        layout = layout,
        labels = NULL,
        method = "landmark_tsne",
        metrics = landmark_tsne_metrics(state, request, timings),
        parameters = landmark_tsne_parameters(state, request),
        timings = timings,
        knn = NULL,
        landmarks = list(
            indices = state$landmark_indices,
            layout = state$reference_fit$layout,
            reference_fit = state$reference_fit,
            projection_knn = if (request$keep_knn) {
                projection$knn
            } else {
                NULL
            },
            transform = attr(
                projection$projected,
                "fastEmbedR_config"
            )
        ),
        preprocess = state$prepared$preprocess
    )
    class(out) <- "fastEmbedR_embedding"
    out
}

#' Landmark t-SNE with fixed-reference transform
#'
#' `landmark_tsne()` embeds a subset of observations with [tsne()], then
#' places the remaining observations with `transform_tsne()`. Projection KNN
#' searches only the
#' fixed landmark reference: CPU uses native HNSW, Metal uses native exact or
#' recall-tuned IVF-Flat, and CUDA keeps native exact or IVF-Flat results
#' resident on the device.
#'
#' @param data Numeric matrix/data frame with observations in rows.
#' @param landmarks `TRUE` for an automatic subset, a fraction such as `0.5`, a
#'   landmark count, or explicit row indices.
#' @param reference_method Kept for compatibility. Only `"tsne"` is
#'   accepted in the cleaned package API.
#' @inheritParams tsne
#' @param transform_k Number of landmark neighbors used to place non-landmarks.
#' @param transform_perplexity Perplexity used by `transform_tsne()`.
#' @param transform_iter Number of normal transform iterations. Use `0` for
#'   projection-only landmarking with no transform refinement.
#' @param n_neighbors Number of non-self neighbors used to embed the landmark
#'   reference set. If `NULL`, it follows the same neighbor policy as
#'   [tsne()]: `ceiling(perplexity)` under compact support.
#' @param perplexity t-SNE perplexity for the landmark reference embedding. If
#'   `NULL`, the optimizer chooses a safe value from the reference KNN width and
#'   sample size.
#' @param standardize Center and scale columns before landmark selection and
#'   neighbor search. Unlike [tsne()], landmark t-SNE defaults to `TRUE`.
#' @param transform_early_exaggeration_iter Number of transform early
#'   exaggeration iterations.
#' @param transform_n_negatives Number of sampled reference negatives used by
#'   `transform_tsne()` on large landmark sets. GPU sampled transform repulsion
#'   is native and experimental for large reference sets.
#' @param initialization Initial placement for transformed observations.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores used by CPU KNN and CPU
#'   transform optimization. Native GPU stages ignore this argument.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' fit <- landmark_tsne(
#'     as.matrix(iris[, 1:4]),
#'     landmarks = 0.5, perplexity = 5,
#'     early_exaggeration_iter = 5, n_iter = 10,
#'     transform_iter = 5, seed = 1
#' )
#' plot(fit, labels = iris$Species)
#' @export
landmark_tsne <- function(
    data, landmarks = TRUE, reference_method = c("tsne"),
    n_neighbors = NULL, perplexity = NULL,
    n_components = 2L,
    standardize = TRUE, pca_dims = NULL, seed = 4L, backend = NULL,
    transform_k = NULL, transform_perplexity = 5,
    transform_iter = 250L, transform_early_exaggeration_iter = 0L,
    transform_n_negatives = NULL,
    initialization = c("median", "weighted", "random"),
    keep_knn = FALSE, verbose = FALSE, n.cores = NULL, ...
) {
    request <- list(
        landmarks = landmarks, reference_method = reference_method,
        n_neighbors = n_neighbors, perplexity = perplexity,
        n_components = n_components,
        standardize = standardize, pca_dims = pca_dims, seed = seed,
        backend = backend, transform_k = transform_k,
        transform_perplexity = transform_perplexity,
        transform_iter = transform_iter,
        transform_early_exaggeration_iter =
            transform_early_exaggeration_iter,
        transform_n_negatives = transform_n_negatives,
        initialization = initialization, keep_knn = keep_knn,
        verbose = verbose, n.cores = n.cores, extra = list(...)
    )
    request <- normalize_landmark_tsne_request(request)
    state <- prepare_landmark_tsne_data(data, request)
    if (!length(state$selection$query_indices)) {
        return(run_full_landmark_tsne(state, request))
    }
    state <- partition_landmark_tsne(state, request)
    state <- fit_landmark_tsne_reference(state, request)
    state <- project_landmark_tsne(state, request)
    assemble_landmark_tsne_output(state, request)
}
