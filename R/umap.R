#' Run opinionated fixed-policy UMAP from data or precomputed KNN
#'
#' `umap()` is a small convenience wrapper. If `data` is already a list of KNN
#' `indices` and `distances`, it calls [umap_knn()] directly. Otherwise it
#' preprocesses the data, computes KNN once with fastEmbedR's native backend,
#' and embeds from that KNN graph.
#'
#' @param data Numeric matrix/data frame, or a list containing KNN `indices`
#'   and `distances`.
#' @param n_neighbors Number of non-self neighbors. `NULL` chooses the package
#'   default for the data size.
#' @param n_components Output dimensionality. The CPU backend supports positive
#'   dimensions, including three-dimensional embeddings. The current Metal and
#'   CUDA optimizers support only `2L`.
#' @param standardize Center and scale columns before KNN when `data` is a
#'   matrix. Defaults to `FALSE` so one-call results match a KNN object computed
#'   from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input: `"euclidean"`,
#'   `"cosine"`, `"correlation"`, or `"inner_product"`.
#' @param nn Optional precomputed KNN result when `data` is a matrix.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. CPU KNN
#'   uses package-native HNSW. Metal uses package-native exact or recall-tuned
#'   IVF-Flat search. CUDA uses cuVS brute-force exact search below 100,000
#'   rows and cuVS IVF-Flat above that threshold; an explicitly enabled FAISS
#'   GPU build may provide exact search. The KNN result stays on the device
#'   through graph construction and optimization.
#'   GPU requests must resolve to a real native backend; the package does not
#'   relabel CPU work as GPU.
#' @param n.cores Requested CPU core count. For matrix input, the value is
#'   passed to native CPU KNN and CPU UMAP; UMAP currently uses at most four
#'   workers and records both requested and effective values. `NULL` uses one
#'   KNN worker and a size-aware UMAP default from one to four workers. Native
#'   GPU stages ignore this argument.
#' @param keep_knn Keep KNN matrices in the returned object.
#' @param graph_mode Graph weighting mode. `"fuzzy"` (the default) uses the
#'   standard UMAP fuzzy graph. `"binary"` uses a symmetric unit-weight graph
#'   as an adjacency-only sensitivity approximation; it is not standard UMAP
#'   or necessarily faster.
#' @param verbose Print progress.
#' @details
#' `umap()` is an opinionated high-throughput UMAP implementation rather than
#' a drop-in interface for every UMAP hyperparameter. The public matrix-input
#' API exposes neighborhood size, distance metric, graph weighting, output
#' dimension, preprocessing, backend, seed, and CPU thread count. In the
#' current release, epochs, minimum distance, spread, learning rate, repulsion
#' strength, negative-sample rate, spectral iteration count, and asynchronous
#' update mode are selected internally. The resolved values and rule names are
#' stored in `fit$parameters`.
#'
#' The default policy uses 500 epochs below 10,000 observations and 200 epochs
#' for larger inputs, increasing to 300 for high-variability distance profiles;
#' five negative samples, spread 1, and repulsion strength 1 are fixed. Minimum
#' distance is normally 0.01 and becomes 0.1 only under the documented
#' wide-shell profile; learning rate is correspondingly 1 or 1.25. Use
#' [umap_init()] to precompute and reuse package-native initial coordinates,
#' or [umap_knn()] to supply an externally controlled KNN graph. These controls
#' are scientifically consequential; fixing them is a scope
#' decision made to keep one tested objective and update schedule aligned across
#' CPU, Metal, and CUDA, not a claim that the values are universally optimal.
#' Arbitrary `min_dist`, `spread`, epoch, learning-rate, negative-sampling, or
#' optimizer-mode sweeps should use a
#' general-purpose UMAP implementation; fastEmbedR does not claim API or
#' parameter interchangeability with those packages.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' fit <- umap(x, n_neighbors = 15, seed = 1)
#' plot(fit)
#' @name umap
NULL

validate_umap_request <- function(backend, graph_mode, n_components,
                                    n.cores, keep_knn) {
    backend <- resolve_embedding_backend(backend)
    graph_mode <- match.arg(graph_mode, c("fuzzy", "binary"))
    n_components <- validate_n_components(n_components)
    if (n_components != 2L && backend %in% c("metal", "cuda")) {
        stop(
            "Native Metal and CUDA UMAP optimizers currently support only ",
            "`n_components = 2`.",
            call. = FALSE
        )
    }
    list(
        backend = backend, graph_mode = graph_mode,
        n_components = n_components, n_threads = n.cores,
        keep_knn = isTRUE(keep_knn)
    )
}

resolve_umap_layout_k <- function(data, cfg) {
    resolved_k <- cfg$knn_n_neighbors %||% cfg$n_neighbors
    if (!is.null(resolved_k) && length(resolved_k) > 0L && !is.na(resolved_k)) {
        return(as.integer(resolved_k))
    }
    if (fastembedr_is_gpu_knn(data)) {
        return(as.integer(fastembedr_gpu_knn_info(data)$k))
    }
    as.integer(ncol(coerce_knn_input(data)$indices))
}

umap_knn_input_fit <- function(data, nn, state, seed, verbose) {
    if (fastembedr_is_gpu_knn(data)) data <- fastembedr_as_gpu_knn(data)
    if (!is.null(nn)) {
        stop("When `data` is a KNN object, do not also pass `nn`.",
            call. = FALSE
        )
    }
    timed <- timed_do_call(fast_knn_umap, list(
        indices = data, n_components = state$n_components, seed = seed,
        verbose = verbose, backend = state$backend,
        n_threads = state$n_threads, graph_mode = state$graph_mode
    ))
    layout <- timed$value
    cfg <- attr(layout, "fastEmbedR_config")
    metrics <- data.frame(
        method = "umap", n = nrow(layout), p = NA_integer_,
        n_neighbors = resolve_umap_layout_k(data, cfg),
        elapsed = timed$time[["elapsed"]], preprocess_elapsed = 0,
        knn_elapsed = 0, embedding_elapsed = timed$time[["elapsed"]]
    )
    new_embedding_result(
        layout, "umap", metrics, cfg,
        rbind(
            preprocess = timed$time * 0, knn = timed$time * 0,
            embedding = timed$time
        ),
        if (state$keep_knn) data else NULL
    )
}

prepare_umap_matrix <- function(data, standardize, pca_dims, seed, backend) {
    timed <- timed_do_call(prepare_embedding_data, list(
        data = data, standardize = standardize, pca_dims = pca_dims,
        seed = seed, backend = backend
    ))
    list(prepared = timed$value, time = timed$time)
}

validate_umap_supplied_knn <- function(nn, backend, n) {
    if (!fastembedr_is_gpu_knn(nn)) {
        return(coerce_knn_input(nn))
    }
    nn <- fastembedr_as_gpu_knn(nn)
    if (!identical(backend, "cuda")) {
        stop("A GPU-resident KNN object requires `backend = \"cuda\"`.",
            call. = FALSE
        )
    }
    info <- fastembedr_gpu_knn_info(nn)
    if (info$n != n || isTRUE(info$has_self)) {
        stop("Supplied GPU-resident KNN output is incompatible with UMAP.",
            call. = FALSE
        )
    }
    nn
}

compute_umap_matrix_knn <- function(x, nn, n_neighbors, metric, state) {
    engine <- "supplied"
    timed <- system.time({
        result <- if (is.null(nn)) {
            policy <- fastembedr_embedding_nn_policy(state$backend, nrow(x))
            keep_gpu <- identical(state$backend, "cuda")
            engine <- fastembedr_nn_policy_engine(policy, keep_gpu = keep_gpu)
            fastembedr_nn_without_self(
                x,
                k = as.integer(n_neighbors), backend = policy$backend,
                method = policy$method, metric = metric,
                output = fastembedr_knn_output_type(x, policy$backend),
                n_threads = state$n_threads, tuning = policy$tuning,
                target_recall = policy$target_recall, keep_gpu = keep_gpu
            )
        } else {
            validate_umap_supplied_knn(nn, state$backend, nrow(x))
        }
    })
    list(knn = result, engine = engine, time = timed)
}

new_embedding_result <- function(layout, method, metrics, parameters,
                                    timings, knn = NULL, extras = list()) {
    out <- c(list(
        layout = layout, labels = NULL, method = method, metrics = metrics,
        parameters = parameters, timings = timings, knn = knn
    ), extras)
    class(out) <- "fastEmbedR_embedding"
    out
}

assemble_umap_matrix_fit <- function(input, prepared, knn_state,
                                        embedding, state, standardize,
                                        metric) {
    layout <- finalize_embedding_layout(
        embedding$value, "UMAP",
        return_float32 = is_float32_matrix(input) &&
            is_float32_matrix(prepared$prepared$data)
    )
    cfg <- attr(layout, "fastEmbedR_config")
    times <- list(prepared$time, knn_state$time, embedding$time)
    elapsed <- sum(vapply(times, function(x) x[["elapsed"]], numeric(1)))
    metrics <- data.frame(
        method = "umap", n = nrow(layout), p = ncol(prepared$prepared$data),
        n_neighbors = as.integer(state$n_neighbors), elapsed = elapsed,
        preprocess_elapsed = times[[1L]][["elapsed"]],
        knn_elapsed = times[[2L]][["elapsed"]],
        embedding_elapsed = times[[3L]][["elapsed"]]
    )
    params <- c(cfg, list(
        standardize = isTRUE(standardize),
        pca_dims = prepared$prepared$preprocess$pca_dims,
        preprocess = prepared$prepared$preprocess,
        graph_mode = state$graph_mode, metric = metric,
        nn_engine = knn_state$engine,
        nn_backend = knn_state$knn$backend_used %||%
            attr(knn_state$knn, "backend") %||% "supplied"
    ))
    new_embedding_result(
        layout, "umap", metrics, params,
        rbind(
            preprocess = times[[1L]], knn = times[[2L]],
            embedding = times[[3L]]
        ),
        if (state$keep_knn) knn_state$knn else NULL
    )
}

#' @rdname umap
#' @export
umap <- function(data, n_neighbors = NULL, n_components = 2L,
                    standardize = FALSE, pca_dims = NULL,
                    metric = c(
                        "euclidean", "cosine", "correlation",
                        "inner_product"
                    ), nn = NULL, seed = 4L,
                    backend = NULL, n.cores = NULL, keep_knn = FALSE,
                    graph_mode = c("fuzzy", "binary"), verbose = FALSE) {
    state <- validate_umap_request(
        backend, graph_mode, n_components, n.cores, keep_knn
    )
    if (is_knn_input(data)) {
        return(umap_knn_input_fit(data, nn, state, seed, verbose))
    }
    prepared <- prepare_umap_matrix(
        data, standardize, pca_dims, seed, state$backend
    )
    x <- prepared$prepared$data
    metric <- resolve_embedding_metric(metric, x)
    if (is.null(n_neighbors)) {
        n_neighbors <- auto_embedding_k(nrow(x), "umap", include_self = FALSE)
    }
    state$n_neighbors <- as.integer(n_neighbors)
    knn_state <- compute_umap_matrix_knn(x, nn, n_neighbors, metric, state)
    embedding <- timed_do_call(fast_knn_umap, list(
        indices = knn_state$knn, n_components = state$n_components,
        seed = seed, verbose = verbose, backend = state$backend,
        n_threads = state$n_threads, graph_mode = state$graph_mode
    ))
    assemble_umap_matrix_fit(
        data, prepared, knn_state, embedding, state, standardize, metric
    )
}

#' Run landmark UMAP from a data matrix
#'
#' `landmark_umap()` embeds a landmark subset with [umap()] and projects the
#' remaining observations against the fixed landmark embedding using
#' query-to-reference KNN, local affine placement, and optional fixed-reference
#' refinement. It is an explicit landmark approximation: the UMAP objective
#' and parameters for the landmark subset are unchanged.
#'
#' @inheritParams umap
#' @param standardize Center and scale columns before landmark selection and
#'   neighbor search. Unlike [umap()], landmark UMAP defaults to `TRUE`.
#' @param landmarks `TRUE` for an automatic subset, a fraction such as `0.5`, a
#'   landmark count, or explicit row indices.
#' @param transform_k Number of landmark neighbors used to project
#'   non-landmark observations. Defaults to `n_neighbors`.
#' @param graph_mode Graph weighting mode passed unchanged to the reference
#'   [umap()] fit. `"fuzzy"` (the default) uses standard UMAP fuzzy graph
#'   weights; `"binary"` uses a symmetric unit-weight sensitivity graph. The
#'   fixed-reference projection algorithm is shared, but the fitted reference
#'   layout and its recorded metadata retain the selected graph mode.
#' @return A `fastEmbedR_embedding` object containing the full layout,
#'   landmark and query indices, timings, and resolved parameters.
#' @examples
#' x <- scale(as.matrix(iris[1:60, 1:4]))
#' fit <- landmark_umap(x, landmarks = 0.5, n_neighbors = 10, seed = 1)
#' plot(fit)
#' @export
landmark_umap <- function(data, landmarks = 0.5, n_neighbors = NULL,
                            n_components = 2L, standardize = TRUE,
                            pca_dims = NULL, seed = 4L, backend = NULL,
                            transform_k = NULL, n.cores = NULL,
                            keep_knn = FALSE,
                            graph_mode = c("fuzzy", "binary"),
                            verbose = FALSE) {
    state <- prepare_landmark_umap(
        data, landmarks, n_neighbors, n_components, standardize,
        pca_dims, seed, backend, n.cores, graph_mode
    )
    if (state$all_landmarks) {
        return(run_full_landmark_umap(
            state, seed, keep_knn, verbose
        ))
    }
    reference <- fit_landmark_umap_reference(
        state, seed, keep_knn, verbose
    )
    projection <- project_landmark_umap_rows(
        state, reference$fit, transform_k, seed
    )
    refinement <- refine_landmark_umap(
        state, reference$fit, projection, seed, verbose
    )
    assemble_landmark_umap_fit(
        state, reference, projection, refinement,
        keep_knn
    )
}

validate_landmark_umap_k <- function(n_neighbors, n) {
    if (is.null(n_neighbors)) {
        n_neighbors <- auto_embedding_k(
            n,
            method = "umap", include_self = FALSE
        )
    }
    value <- as.integer(n_neighbors)
    invalid <- length(value) != 1L || is.na(value) ||
        value < 1L || value >= n
    if (invalid) {
        stop(
            "`n_neighbors` must be positive and smaller than `nrow(data)`.",
            call. = FALSE
        )
    }
    value
}

prepare_landmark_umap <- function(data, landmarks, n_neighbors,
                                    n_components, standardize, pca_dims,
                                    seed, backend, n.cores, graph_mode) {
    backend <- resolve_embedding_backend(backend)
    graph_mode <- match.arg(graph_mode, c("fuzzy", "binary"))
    n_components <- validate_n_components(n_components)
    prepared <- timed_do_call(prepare_embedding_data, list(
        data = data, standardize = standardize, pca_dims = pca_dims,
        seed = seed, backend = backend
    ))
    x <- prepared$value$data
    n_neighbors <- validate_landmark_umap_k(n_neighbors, nrow(x))
    selection <- select_landmarks(
        x, landmarks,
        seed = seed, n.cores = n.cores
    )
    all_landmarks <- length(selection$query_indices) == 0L
    partition <- if (all_landmarks) {
        NULL
    } else {
        split_landmark_data(
            x, selection$indices, selection$query_indices,
            n_threads = n.cores
        )
    }
    list(
        x = x, prepared = prepared$value, preprocess_time = prepared$time,
        selection = selection, partition = partition,
        all_landmarks = all_landmarks, n = nrow(x),
        n_neighbors = n_neighbors, n_components = n_components,
        backend = backend, n_threads = n.cores,
        graph_mode = graph_mode, seed = seed
    )
}

run_full_landmark_umap <- function(state, seed, keep_knn, verbose) {
    umap(
        state$x,
        n_neighbors = state$n_neighbors,
        n_components = state$n_components, standardize = FALSE,
        pca_dims = NULL, seed = seed, backend = state$backend,
        n.cores = state$n_threads, keep_knn = keep_knn,
        graph_mode = state$graph_mode, verbose = verbose
    )
}

fit_landmark_umap_reference <- function(state, seed, keep_knn, verbose) {
    x_landmarks <- state$partition$landmarks
    k <- min(state$n_neighbors, nrow(x_landmarks) - 1L)
    timed <- system.time({
        knn <- landmark_reference_knn(
            x_landmarks,
            k = k, backend = state$backend,
            n_threads = state$n_threads
        )
        fit <- umap(
            x_landmarks,
            n_neighbors = k,
            n_components = state$n_components, standardize = FALSE,
            pca_dims = NULL, seed = seed, backend = state$backend,
            nn = knn, n.cores = state$n_threads, keep_knn = keep_knn,
            graph_mode = state$graph_mode, verbose = verbose
        )
    })
    list(fit = fit, knn = knn, time = timed, n_neighbors = k)
}

project_landmark_umap_host <- function(state, fit, knn) {
    projected <- attr(knn, "projected_layout", exact = TRUE)
    affine <- landmark_affine_projection(
        state$partition$landmarks, state$partition$query,
        fit$layout, knn,
        n_threads = state$n_threads,
        backend = state$backend
    )
    if (embedding_layout_dims_match(
        affine, length(state$selection$query_indices), state$n_components
    )) {
        return(affine)
    }
    if (!is.null(projected)) {
        return(projected)
    }
    use_metal <- identical(state$backend, "metal") &&
        isTRUE(embedding_metal_available_cpp())
    projection_fun <- if (use_metal) {
        project_embedding_knn_metal_cpp
    } else {
        project_embedding_knn_cpp
    }
    projection_fun(fit$layout, knn$indices, knn$distances)
}

project_landmark_umap_rows <- function(state, fit, transform_k, seed) {
    n_landmarks <- nrow(state$partition$landmarks)
    if (is.null(transform_k)) {
        transform_k <- min(n_landmarks, state$n_neighbors)
    }
    transform_k <- transform_embedding_k(transform_k, n_landmarks)
    timed <- system.time({
        knn <- landmark_projection_knn(
            state$partition$landmarks, state$partition$query,
            k = transform_k, backend = state$backend,
            seed = seed + 503L, n_threads = state$n_threads,
            landmark_layout = fit$layout, all_data = state$x,
            landmark_indices = state$selection$indices,
            query_rows = state$selection$query_indices
        )
        resident <- identical(state$backend, "cuda") &&
            fastembedr_is_gpu_knn(knn)
        projected <- if (resident) {
            NULL
        } else {
            project_landmark_umap_host(state, fit, knn)
        }
    })
    list(
        knn = knn, projected = projected, resident = resident,
        transform_k = transform_k, time = timed
    )
}

landmark_umap_refinement_epochs <- function() {
    value <- integer_scalar(getOption(
        "fastEmbedR.landmark_umap_refine_epochs", 50L
    ))
    if (length(value) != 1L || is.na(value) || value < 0L) 50L else value
}

landmark_umap_optimizer_parameters <- function(fit) {
    params <- fit$parameters
    values <- list(
        min_dist = as.numeric(params$min_dist %||% 0.01),
        negative_rate = as.integer(params$negative_sample_rate %||% 5L),
        learning_rate = as.numeric(params$learning_rate %||% 1),
        repulsion = as.numeric(params$repulsion_strength %||% 1)
    )
    if (!is.finite(values$min_dist) || values$min_dist < 0) {
        values$min_dist <- 0.01
    }
    if (is.na(values$negative_rate) || values$negative_rate < 0L) {
        values$negative_rate <- 5L
    }
    if (!is.finite(values$learning_rate) || values$learning_rate <= 0) {
        values$learning_rate <- 1
    }
    if (!is.finite(values$repulsion) || values$repulsion <= 0) {
        values$repulsion <- 1
    }
    values
}

run_landmark_umap_cuda_refinement <- function(state, fit, projection,
                                                epochs, params, seed) {
    result <- landmark_umap_project_refine_cuda_gpu_cpp(
        projection$knn, state$partition$landmarks,
        state$partition$query, fit$layout,
        as.integer(state$selection$indices),
        as.integer(state$selection$query_indices), as.integer(state$n),
        as.integer(epochs), params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion,
        as.integer(seed + 2003L), 12L, 1e-3, 2.5
    )
    list(layout = result$layout, backend = "cuda")
}

landmark_projection_global_indices <- function(state, projection) {
    indices <- state$selection$indices[
        as.integer(projection$knn$indices)
    ]
    matrix(
        as.integer(indices),
        nrow = nrow(projection$knn$indices),
        ncol = ncol(projection$knn$indices)
    )
}

run_landmark_umap_metal_refinement <- function(state, fit, projection,
                                                layout, indices, epochs,
                                                params, seed) {
    tryCatch(
        {
            refined <- knn_umap_refine_rows_metal_cpp(
                indices, projection$knn$distances,
                as.integer(state$selection$query_indices), layout,
                as.integer(epochs), params$min_dist,
                params$negative_rate, params$learning_rate,
                params$repulsion, as.integer(seed + 2003L)
            )
            list(layout = refined, backend = "metal")
        },
        error = function(e) {
            stop(
                "Metal UMAP landmark refinement failed: ",
                conditionMessage(e),
                call. = FALSE
            )
        }
    )
}

run_landmark_umap_cpu_refinement <- function(state, projection, layout,
                                                indices, epochs, params,
                                                seed, verbose) {
    refined <- knn_umap_refine_rows_cpp(
        indices, projection$knn$distances,
        as.integer(state$selection$query_indices), layout,
        as.integer(epochs), params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion,
        as.integer(normalize_nn_threads(state$n_threads)),
        as.integer(seed + 2003L), isTRUE(verbose)
    )
    list(layout = refined, backend = "cpu")
}

run_landmark_umap_host_refinement <- function(state, fit, projection,
                                                layout, epochs, params,
                                                seed, verbose) {
    indices <- landmark_projection_global_indices(state, projection)
    use_metal <- state$n_components == 2L &&
        identical(state$backend, "metal") &&
        isTRUE(embedding_metal_available_cpp())
    result <- if (use_metal) {
        run_landmark_umap_metal_refinement(
            state, fit, projection, layout, indices, epochs, params, seed
        )
    } else {
        run_landmark_umap_cpu_refinement(
            state, projection, layout, indices, epochs, params, seed, verbose
        )
    }
    result$layout[state$selection$indices, ] <-
        embedding_dense_double_matrix(fit$layout)
    colnames(result$layout) <- colnames(fit$layout)
    result
}

refine_landmark_umap <- function(state, fit, projection, seed, verbose) {
    epochs <- landmark_umap_refinement_epochs()
    params <- landmark_umap_optimizer_parameters(fit)
    layout <- if (projection$resident) {
        NULL
    } else {
        assemble_landmark_layout(
            fit$layout, projection$projected, state$selection$indices,
            state$selection$query_indices, state$n,
            prefix = "UMAP",
            return_float32 = FALSE
        )
    }
    if (!projection$resident && epochs == 0L) {
        return(list(
            layout = layout, backend = NA_character_,
            epochs = epochs, time = zero_proc_time()
        ))
    }
    timed <- system.time({
        result <- if (projection$resident) {
            run_landmark_umap_cuda_refinement(
                state, fit, projection, epochs, params, seed
            )
        } else {
            run_landmark_umap_host_refinement(
                state, fit, projection, layout, epochs, params,
                seed, verbose
            )
        }
    })
    list(
        layout = result$layout, backend = result$backend,
        epochs = epochs, time = timed
    )
}

landmark_umap_metrics <- function(state, reference, projection, refinement) {
    fit_metrics <- reference$fit$metrics
    knn_elapsed <- if ("knn_elapsed" %in% names(fit_metrics)) {
        fit_metrics$knn_elapsed[[1L]]
    } else {
        NA_real_
    }
    optimizer_elapsed <- if ("embedding_elapsed" %in% names(fit_metrics)) {
        fit_metrics$embedding_elapsed[[1L]]
    } else {
        NA_real_
    }
    times <- list(
        state$preprocess_time, reference$time,
        projection$time, refinement$time
    )
    data.frame(
        method = "landmark_umap", n = state$n, p = ncol(state$x),
        n_neighbors = state$n_neighbors,
        elapsed = sum(vapply(times, function(x) x[["elapsed"]], numeric(1))),
        preprocess_elapsed = times[[1L]][["elapsed"]],
        reference_embedding_elapsed = times[[2L]][["elapsed"]],
        reference_knn_elapsed = knn_elapsed,
        reference_optimizer_elapsed = optimizer_elapsed,
        landmark_projection_knn_elapsed = times[[3L]][["elapsed"]],
        landmark_refinement_elapsed = times[[4L]][["elapsed"]],
        transform_elapsed = 0, landmark = TRUE,
        n_landmarks = nrow(state$partition$landmarks),
        landmark_fraction = nrow(state$partition$landmarks) / state$n,
        transform_k = projection$transform_k,
        silhouette = NA_real_, knn_preservation = NA_real_,
        local_trustworthiness = NA_real_, local_continuity = NA_real_,
        structure_score = NA_real_, embedding_knn_accuracy = NA_real_
    )
}

landmark_umap_parameters <- function(state, reference, projection,
                                        refinement, keep_knn) {
    approximation <- attr(projection$knn, "approximation", exact = TRUE)
    strategy <- approximation$strategy %||% if (
        isTRUE(attr(projection$knn, "exact"))
    ) {
        "exact"
    } else {
        NA_character_
    }
    n_landmarks <- nrow(state$partition$landmarks)
    c(list(
        method = "landmark_umap", n = state$n, p = ncol(state$x),
        n_neighbors = state$n_neighbors,
        n_components = as.integer(state$n_components),
        seed = as.integer(state$seed),
        nn_backend = reference$fit$parameters$nn_backend,
        projection_nn_backend = attr(projection$knn, "backend"),
        projection_strategy = as.character(strategy),
        backend = reference$fit$parameters$backend %||% state$backend,
        n.cores = normalize_nn_threads(state$n_threads), landmark = TRUE,
        n_landmarks = n_landmarks,
        landmark_fraction = n_landmarks / state$n,
        landmark_selection = state$selection$method,
        graph_mode = state$graph_mode, transform_k = projection$transform_k,
        landmark_refinement = if (refinement$epochs > 0L) {
            "fixed_landmark_umap_rows"
        } else {
            "none"
        },
        landmark_refinement_epochs = as.integer(refinement$epochs),
        landmark_refinement_backend = if (refinement$epochs > 0L) {
            refinement$backend
        } else {
            NA_character_
        },
        keep_knn = keep_knn,
        provenance = "UMAP_landmark_projection_native_cpp"
    ), state$prepared$preprocess)
}

assemble_landmark_umap_fit <- function(state, reference, projection,
                                        refinement, keep_knn) {
    layout <- finalize_embedding_layout(
        refinement$layout, "UMAP",
        return_float32 = is_float32_matrix(state$x)
    )
    timings <- rbind(
        preprocess = state$preprocess_time,
        reference_embedding = reference$time,
        landmark_projection_knn = projection$time,
        landmark_refinement = refinement$time,
        transform = zero_proc_time()
    )
    extras <- list(
        landmarks = list(
            indices = state$selection$indices,
            layout = reference$fit$layout,
            reference_fit = reference$fit,
            projection_knn = if (keep_knn) projection$knn else NULL
        ),
        preprocess = state$prepared$preprocess
    )
    new_embedding_result(
        layout, "landmark_umap",
        landmark_umap_metrics(state, reference, projection, refinement),
        landmark_umap_parameters(
            state, reference, projection, refinement, keep_knn
        ),
        timings,
        knn = NULL, extras = extras
    )
}
