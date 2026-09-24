# KNN normalization, sparse-affinity construction, and backend dispatch.

normalize_opentsne_knn_input <- function(
    indices, distances = NULL,
    n_neighbors = NULL
) {
    knn <- coerce_knn_input(indices, distances)
    n <- nrow(knn$indices)
    available <- knn$n_neighbors
    if (is.null(n_neighbors)) {
        n_neighbors <- available
    } else {
        n_neighbors <- integer_scalar(n_neighbors)
        if (is.na(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
            stop(
                "`n_neighbors` must be a positive integer smaller than ",
                "the number of rows.",
                call. = FALSE
            )
        }
        if (n_neighbors > available) {
            stop("`n_neighbors` is larger than the supplied KNN width.",
                call. = FALSE
            )
        }
    }
    materialized <- materialize_knn_range(
        knn$indices, knn$distances,
        knn$col_start,
        n_neighbors
    )
    distance_type <- knn$distance_type
    list(
        indices = materialized$indices,
        distances = materialized$distances,
        n = n,
        n_neighbors = as.integer(n_neighbors),
        has_self = isTRUE(knn$has_self),
        input_backend = knn$input_backend,
        distance_type = distance_type
    )
}

prepare_opentsne_materialized_run <- function(request) {
    request$backend <- resolve_embedding_backend(request$backend)
    request$optimizer_backend <- resolve_opentsne_optimizer_backend(
        request$backend
    )
    request$gpu_resident_knn <- !is.null(request$gpu_knn)
    request$n <- if (request$gpu_resident_knn) {
        as.integer(request$gpu_n)
    } else {
        nrow(request$indices)
    }
    request$k <- if (request$gpu_resident_knn) {
        as.integer(request$gpu_k)
    } else {
        ncol(request$indices)
    }
    request$Y_init <- resolve_opentsne_y_init(
        request$Y_init,
        request$n,
        request$n_components
    )
    request$n_threads <- request$n_threads %||% default_tsne_threads()
    request$gradient_method <- normalize_tsne_negative_gradient_method(
        request$negative_gradient_method
    )
    request$auto_params <- resolve_opentsne_auto_parameters(
        request$n, request$k, request$perplexity,
        request$early_exaggeration_iter, request$n_iter,
        request$learning_rate, request$optimizer_backend,
        request$gradient_method, request$auto_config
    )
    request$iterations <- validate_opentsne_iteration_counts(
        request$auto_params
    )
    request$gradient_method <- resolve_opentsne_gradient_method(
        request$gradient_method, request$optimizer_backend,
        request$n, request$n_components
    )
    request
}

prepare_opentsne_materialized_controls <- function(request) {
    init_info <- prepare_opentsne_initialization(
        request$Y_init, request$gpu_resident_knn,
        request$cuda_init_data, request$optimizer_backend,
        request$indices, request$distances, request$n_components,
        request$seed, request$gradient_method
    )
    controls <- resolve_opentsne_controls(
        request$n, request$n_components, request$auto_params$perplexity,
        request$theta, request$iterations$early, request$iterations$normal,
        request$verbose, init_info$Y_init, request$initial_momentum,
        request$final_momentum, request$learning_rate,
        request$early_exaggeration, request$exaggeration,
        request$auto_params, request$min_gain, request$max_step_norm,
        request$optimizer_backend, request$gradient_method,
        request$record_costs
    )
    list(init_info = init_info, controls = controls)
}

run_opentsne_materialized_request <- function(request) {
    request <- prepare_opentsne_materialized_run(request)
    prepared <- prepare_opentsne_materialized_controls(request)
    out <- run_opentsne_native_optimizer(
        request$optimizer_backend, request$indices, request$distances,
        request$gpu_knn, request$gpu_resident_knn, request$cuda_init_data,
        request$k, prepared$controls, request$iterations$early,
        request$iterations$normal, request$gradient_method,
        request$auto_params, request$n_threads, request$seed
    )
    finalize_opentsne_native_result(
        out, request$optimizer_backend, request$gpu_resident_knn,
        request$distances, request$n, request$k, prepared$controls,
        request$auto_params, prepared$init_info, request$gradient_method,
        request$iterations$early, request$iterations$normal,
        request$input_had_self, request$input_backend
    )
}

fast_knn_opentsne_materialized <- function(
    indices, distances, n_components = 2L, perplexity = NULL, theta = 0.5,
    early_exaggeration_iter = NULL, n_iter = NULL, learning_rate = "auto",
    early_exaggeration = "auto", exaggeration = NULL, Y_init = NULL,
    initial_momentum = 0.8, final_momentum = 0.8, min_gain = 0.01,
    max_step_norm = "auto", negative_gradient_method = "auto",
    record_costs = FALSE, n_threads = NULL, seed = 42L, verbose = FALSE,
    backend = NULL, auto_config = TRUE, input_had_self = FALSE,
    input_backend = NA_character_, gpu_knn = NULL, gpu_n = NULL,
    gpu_k = NULL, cuda_init_data = NULL
) {
    request <- list(
        indices = indices, distances = distances,
        n_components = n_components, perplexity = perplexity, theta = theta,
        early_exaggeration_iter = early_exaggeration_iter, n_iter = n_iter,
        learning_rate = learning_rate,
        early_exaggeration = early_exaggeration, exaggeration = exaggeration,
        Y_init = Y_init, initial_momentum = initial_momentum,
        final_momentum = final_momentum, min_gain = min_gain,
        max_step_norm = max_step_norm,
        negative_gradient_method = negative_gradient_method,
        record_costs = record_costs, n_threads = n_threads, seed = seed,
        verbose = verbose, backend = backend, auto_config = auto_config,
        input_had_self = input_had_self, input_backend = input_backend,
        gpu_knn = gpu_knn, gpu_n = gpu_n, gpu_k = gpu_k,
        cuda_init_data = cuda_init_data
    )
    run_opentsne_materialized_request(request)
}

fast_knn_opentsne_core <- function(
    indices, distances = NULL, n_components = 2L, perplexity = NULL,
    theta = 0.5, early_exaggeration_iter = NULL, n_iter = NULL,
    learning_rate = "auto", early_exaggeration = "auto",
    exaggeration = NULL, Y_init = NULL, initial_momentum = 0.8,
    final_momentum = 0.8, min_gain = 0.01, max_step_norm = 5,
    negative_gradient_method = "auto", record_costs = FALSE,
    n_threads = NULL, seed = 42L, verbose = FALSE, backend = NULL,
    auto_config = TRUE
) {
    backend <- resolve_embedding_backend(backend)
    if (inherits(indices, "fastEmbedR_tsne_prepared")) {
        if (!is.null(distances)) {
            stop(
                "Do not pass `distances` when `indices` is a prepared ",
                "t-SNE object.",
                call. = FALSE
            )
        }
        knn <- indices$knn
    } else {
        knn <- normalize_opentsne_knn_input(indices, distances)
    }
    Y_init <- resolve_opentsne_y_init(Y_init, knn$n, n_components)
    fast_knn_opentsne_materialized(
        knn$indices, knn$distances,
        n_components = n_components,
        perplexity = perplexity,
        theta = theta,
        early_exaggeration_iter = early_exaggeration_iter,
        n_iter = n_iter,
        learning_rate = learning_rate,
        early_exaggeration = early_exaggeration,
        exaggeration = exaggeration,
        Y_init = Y_init,
        initial_momentum = initial_momentum,
        final_momentum = final_momentum,
        min_gain = min_gain,
        max_step_norm = max_step_norm,
        negative_gradient_method = negative_gradient_method,
        record_costs = record_costs,
        n_threads = n_threads,
        seed = seed,
        verbose = verbose,
        backend = backend,
        auto_config = auto_config,
        input_had_self = knn$has_self,
        input_backend = knn$input_backend
    )
}

#' Precompute reusable t-SNE KNN state
#'
#' `prepare_tsne_knn()` strips self-neighbors, trims to the requested
#' non-self width, and stores compact KNN matrices once. Pass the returned
#' object to [tsne_knn()] for repeated seeds or backend comparisons without
#' repeating KNN normalization/materialization.
#'
#' @inheritParams tsne_knn
#' @return A prepared t-SNE KNN object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 15L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'     nrow = nrow(d), byrow = TRUE
#' )
#' prep <- prepare_tsne_knn(idx, dst, perplexity = 5)
#' y1 <- tsne_knn(prep, seed = 1, early_exaggeration_iter = 5, n_iter = 10)
#' y2 <- tsne_knn(prep, seed = 2, early_exaggeration_iter = 5, n_iter = 10)
#' @export
prepare_tsne_knn <- function(indices,
                                distances = NULL,
                                n_neighbors = NULL,
                                perplexity = NULL) {
    knn0 <- coerce_knn_input(indices, distances)
    policy <- opentsne_neighbor_policy(
        nrow(knn0$indices),
        perplexity = perplexity,
        available = knn0$n_neighbors
    )
    if (is.null(n_neighbors)) {
        n_neighbors <- policy$n_neighbors
    }
    required_k <- opentsne_support_width(policy$perplexity)
    if (n_neighbors != required_k) {
        stop(
            "Compact t-SNE affinity support requires `n_neighbors = ",
            required_k, "` for this perplexity.",
            call. = FALSE
        )
    }
    knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
    out <- list(
        knn = knn,
        perplexity = policy$perplexity,
        n_neighbors = as.integer(n_neighbors),
        affinity_support = "compact",
        affinity_state = "knn_materialized_affinity_builder_internal"
    )
    class(out) <- c("fastEmbedR_tsne_prepared", "list")
    out
}
