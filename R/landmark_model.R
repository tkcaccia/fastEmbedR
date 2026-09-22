#' Select representative landmark observations
#'
#' `select_landmarks()` separates row indices into a representative landmark
#' reference and its complementary query set. It does not embed or project the
#' data.
#'
#' @param data Numeric matrix, data frame, or `float::float32` matrix.
#' @param landmarks `TRUE` for an automatic count, a fraction in `(0, 1)`, a
#'   landmark count, or explicit row indices.
#' @param seed Random seed used by projection-based landmark selection.
#' @param n.cores Number of CPU cores used for float32 selection features.
#' @return A `fastEmbedR_landmark_selection` object containing `indices` and
#'   `query_indices`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, landmarks = 0.5, seed = 1)
#' selection$indices
#' @export
select_landmarks <- function(data,
                                landmarks = TRUE,
                                seed = 4L,
                                n.cores = NULL) {
    n_threads <- n.cores
    prepared <- prepare_embedding_data(
        data,
        standardize = FALSE,
        pca_dims = NULL,
        seed = seed,
        backend = "cpu"
    )
    x <- prepared$data
    indices <- resolve_landmarks(
        landmarks,
        x,
        seed,
        n_threads = n_threads
    )
    if (is.null(indices)) indices <- seq_len(nrow(x))
    query_indices <- setdiff(seq_len(nrow(x)), indices)
    method <- attr(indices, "selection_method", exact = TRUE)
    if (is.null(method)) {
        method <- if (length(indices) == nrow(x)) "all_rows" else "indices"
    }
    out <- list(
        indices = as.integer(indices),
        query_indices = as.integer(query_indices),
        n = as.integer(nrow(x)),
        p = as.integer(ncol(x)),
        n_landmarks = as.integer(length(indices)),
        landmark_fraction = length(indices) / nrow(x),
        method = method,
        seed = as.integer(seed)
    )
    class(out) <- c("fastEmbedR_landmark_selection", "list")
    out
}

normalize_landmark_selection <- function(selection, data) {
    n <- nrow(data)
    selection_method <- NULL
    selection_seed <- NA_integer_
    if (inherits(selection, "fastEmbedR_landmark_selection")) {
        if (!identical(as.integer(selection$n), as.integer(n))) {
            stop(
                "`selection` was created for a different number of rows.",
                call. = FALSE
            )
        }
        indices <- selection$indices
        selection_method <- selection$method
        selection_seed <- selection$seed
    } else if (is.numeric(selection)) {
        indices <- sort(unique(as.integer(selection)))
    } else {
        stop(
            sprintf(
                "%s%s",
                "`selection` must be returned by select_landmarks() or ",
                "contain row indices."
            ),
            call. = FALSE
        )
    }
    if (length(indices) < 2L || anyNA(indices) ||
        any(indices < 1L) || any(indices > n)) {
        stop("Landmark indices must identify at least two valid rows.",
            call. = FALSE
        )
    }
    query_indices <- setdiff(seq_len(n), indices)
    method <- selection_method %||%
        attr(indices, "selection_method") %||% "indices"
    out <- list(
        indices = as.integer(indices),
        query_indices = as.integer(query_indices),
        n = as.integer(n),
        p = as.integer(ncol(data)),
        n_landmarks = as.integer(length(indices)),
        landmark_fraction = length(indices) / n,
        method = method,
        seed = as.integer(selection_seed %||% NA_integer_)
    )
    class(out) <- c("fastEmbedR_landmark_selection", "list")
    out
}

#' Fit an embedding model on selected landmarks
#'
#' `fit_landmark_model()` runs the ordinary production [umap()] or
#' [tsne()] implementation on the selected reference rows. The returned
#' object stores the reference data and embedding needed to project the
#' complementary rows or genuinely new observations.
#'
#' @param data Data in the analysis space used for KNN. Preprocess once before
#'   calling this staged API when centering, scaling, or PCA is required.
#' @param selection A result from [select_landmarks()] or explicit landmark row
#'   indices.
#' @param method `"umap"` or `"tsne"`.
#' @param n_neighbors UMAP neighborhood size. For t-SNE this is the
#'   precomputed KNN support width; `NULL` derives it from `perplexity`.
#' @param perplexity t-SNE perplexity.
#' @param n_components Embedding dimensionality.
#' @param metric KNN metric.
#' @param seed Random seed.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores.
#' @param graph_mode UMAP graph mode, passed unchanged to [umap()]. The
#'   standard `"fuzzy"` graph is the default.
#' @param keep_knn Retain reference KNN output.
#' @param verbose Print optimizer progress.
#' @param ... Additional optimizer arguments passed to [tsne()].
#' @return A `fastEmbedR_landmark_model`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, 0.5, seed = 1)
#' model <- fit_landmark_model(
#'     x, selection,
#'     method = "umap", n_neighbors = 10,
#'     graph_mode = "fuzzy", seed = 1
#' )
#' @export
fit_landmark_model <- function(data, selection,
                                method = c("umap", "tsne"),
                                n_neighbors = NULL, perplexity = NULL,
                                n_components = 2L,
                                metric = c(
                                    "euclidean", "cosine",
                                    "correlation", "inner_product"
                                ),
                                seed = 4L, backend = NULL, n.cores = NULL,
                                graph_mode = c("fuzzy", "binary"),
                                keep_knn = FALSE, verbose = FALSE, ...) {
    state <- prepare_landmark_model(
        data, selection, method, n_components, metric, seed,
        backend, n.cores, graph_mode
    )
    timed <- system.time({
        fitted <- if (identical(state$method, "umap")) {
            fit_landmark_umap_model(
                state, n_neighbors, keep_knn, verbose
            )
        } else {
            fit_landmark_tsne_model(
                state, n_neighbors, perplexity, keep_knn,
                verbose, list(...)
            )
        }
    })
    assemble_landmark_model(state, fitted, timed)
}

landmark_model_reference_data <- function(x, selection, n.cores) {
    if (length(selection$query_indices) == 0L) {
        return(x)
    }
    if (is_float32_matrix(x)) {
        return(split_float32_rows_cpp(
            x, selection$indices, selection$query_indices,
            as.integer(normalize_nn_threads(n.cores))
        )$landmarks)
    }
    x[selection$indices, , drop = FALSE]
}

prepare_landmark_model <- function(data, selection, method, n_components,
                                    metric, seed, backend, n.cores,
                                    graph_mode) {
    method <- match.arg(method, c("umap", "tsne"))
    backend <- resolve_embedding_backend(backend)
    graph_mode <- match.arg(graph_mode, c("fuzzy", "binary"))
    metric <- resolve_embedding_metric(metric, data)
    n_components <- validate_n_components(n_components)
    prepared <- prepare_embedding_data(
        data,
        standardize = FALSE, pca_dims = NULL,
        seed = seed, backend = backend
    )
    selection <- normalize_landmark_selection(selection, prepared$data)
    reference <- landmark_model_reference_data(
        prepared$data, selection, n.cores
    )
    if (nrow(reference) < 2L) {
        stop("At least two landmark rows are required.", call. = FALSE)
    }
    list(
        x = prepared$data, selection = selection,
        reference_data = reference, method = method,
        n_components = n_components, metric = metric, seed = seed,
        backend = backend, n.cores = n.cores, graph_mode = graph_mode,
        affinity_support = "compact"
    )
}

fit_landmark_umap_model <- function(state, n_neighbors,
                                    keep_knn, verbose) {
    if (is.null(n_neighbors)) {
        n_neighbors <- auto_embedding_k(nrow(state$x), method = "umap")
    }
    n_neighbors <- as.integer(min(
        n_neighbors, nrow(state$reference_data) - 1L
    ))
    knn <- precompute_knn(
        state$reference_data,
        k = n_neighbors, metric = state$metric,
        backend = state$backend, n.cores = state$n.cores
    )
    fit <- umap(
        state$reference_data,
        n_neighbors = n_neighbors,
        n_components = state$n_components, standardize = FALSE,
        metric = state$metric, nn = knn, seed = state$seed,
        backend = state$backend, n.cores = state$n.cores,
        keep_knn = keep_knn, graph_mode = state$graph_mode,
        verbose = verbose
    )
    list(
        fit = fit, knn = knn, n_neighbors = n_neighbors,
        perplexity = NULL
    )
}

resolve_landmark_model_tsne_neighbors <- function(n_neighbors, n_reference,
                                                    perplexity) {
    policy <- opentsne_neighbor_policy(
        n_reference,
        perplexity = perplexity
    )
    if (is.null(perplexity)) perplexity <- policy$perplexity
    if (is.null(n_neighbors)) {
        return(list(
            n_neighbors = policy$n_neighbors, perplexity = perplexity
        ))
    }
    n_neighbors <- as.integer(n_neighbors)
    invalid <- length(n_neighbors) != 1L || is.na(n_neighbors) ||
        n_neighbors < 1L || n_neighbors >= n_reference
    if (invalid) {
        stop(
            "`n_neighbors` must be positive and smaller than the landmarks.",
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
    list(n_neighbors = n_neighbors, perplexity = perplexity)
}

fit_landmark_tsne_model <- function(state, n_neighbors, perplexity,
                                    keep_knn, verbose, dots) {
    policy <- resolve_landmark_model_tsne_neighbors(
        n_neighbors, nrow(state$reference_data), perplexity
    )
    knn <- precompute_knn(
        state$reference_data,
        k = policy$n_neighbors,
        metric = state$metric, backend = state$backend,
        n.cores = state$n.cores
    )
    args <- c(list(
        data = state$reference_data, perplexity = policy$perplexity,
        n_components = state$n_components,
        init_data = state$reference_data, standardize = FALSE,
        metric = state$metric, nn = knn, seed = state$seed,
        backend = state$backend, keep_knn = keep_knn,
        verbose = verbose, n.cores = state$n.cores
    ), dots)
    fit <- do.call(tsne, args)
    list(
        fit = fit, knn = knn, n_neighbors = policy$n_neighbors,
        perplexity = fit$parameters$perplexity %||% policy$perplexity
    )
}

assemble_landmark_model <- function(state, fitted, elapsed) {
    out <- list(
        method = state$method, fit = fitted$fit,
        reference_data = state$reference_data,
        selection = state$selection, n_total = as.integer(nrow(state$x)),
        p = as.integer(ncol(state$x)),
        n_components = as.integer(state$n_components),
        n_neighbors = as.integer(fitted$n_neighbors),
        perplexity = fitted$perplexity,
        affinity_support = if (state$method == "tsne") {
            state$affinity_support
        } else {
            NA_character_
        },
        metric = state$metric,
        graph_mode = if (state$method == "umap") {
            state$graph_mode
        } else {
            NA_character_
        },
        backend = state$backend, seed = as.integer(state$seed),
        n.cores = normalize_nn_threads(state$n.cores),
        elapsed_sec = unname(elapsed[["elapsed"]])
    )
    class(out) <- c("fastEmbedR_landmark_model", "list")
    out
}

landmark_query_data <- function(model, data, query_indices = NULL) {
    prepared <- prepare_embedding_data(
        data,
        standardize = FALSE,
        pca_dims = NULL,
        seed = model$seed,
        backend = model$backend
    )
    x <- prepared$data
    # Automatic reassembly is meaningful only for a model fitted on a strict
    # landmark subset. An all-reference model must treat every supplied matrix
    # as
    # new data, including a query matrix with the same row count as the training
    # matrix.
    full_input <- is.null(query_indices) &&
        length(model$selection$query_indices) > 0L &&
        nrow(x) == model$n_total
    if (full_input) query_indices <- model$selection$query_indices
    selected <- select_landmark_query_rows(x, query_indices, model$n.cores)
    query <- selected$data
    query_indices <- selected$indices
    if (ncol(query) != ncol(model$reference_data)) {
        stop("Query data are not in the model's reference feature space.",
            call. = FALSE
        )
    }
    list(
        data = query,
        indices = query_indices,
        full_input = full_input,
        input = x
    )
}

select_landmark_query_rows <- function(x, query_indices, n.cores) {
    if (is.null(query_indices)) {
        return(list(data = x, indices = seq_len(nrow(x))))
    }
    query_indices <- as.integer(query_indices)
    invalid <- anyNA(query_indices) || any(query_indices < 1L) ||
        any(query_indices > nrow(x))
    if (invalid) {
        stop("`query_indices` contains invalid rows.", call. = FALSE)
    }
    query <- if (length(query_indices) == nrow(x)) {
        x
    } else if (is_float32_matrix(x)) {
        split_float32_rows_cpp(
            x, query_indices, setdiff(seq_len(nrow(x)), query_indices),
            as.integer(n.cores)
        )$landmarks
    } else {
        x[query_indices, , drop = FALSE]
    }
    list(data = query, indices = query_indices)
}

landmark_model_projection_k <- function(model,
                                        transform_k,
                                        transform_perplexity = 5) {
    if (is.null(transform_k)) {
        transform_k <- if (identical(model$method, "umap")) {
            model$n_neighbors
        } else {
            ceiling(transform_perplexity)
        }
    }
    transform_embedding_k(
        min(as.integer(transform_k), nrow(model$reference_data)),
        nrow(model$reference_data)
    )
}

project_landmark_umap <- function(model, query, projection_knn,
                                    refinement_epochs, n_threads, verbose) {
    params <- landmark_umap_optimizer_parameters(model$fit)
    if (identical(model$backend, "cuda") &&
        fastembedr_is_gpu_knn(projection_knn)) {
        return(project_landmark_umap_cuda(
            model, query, projection_knn, refinement_epochs, params
        ))
    }
    initialized <- initialize_landmark_umap_query(
        model, query, projection_knn, n_threads
    )
    if (refinement_epochs <= 0L) {
        return(initialized)
    }
    refine_landmark_umap_query(
        model, query, projection_knn, initialized$layout,
        refinement_epochs, params, n_threads, verbose
    )
}

project_landmark_umap_cuda <- function(model, query, knn,
                                        epochs, params) {
    n_reference <- nrow(model$reference_data)
    query_rows <- n_reference + seq_len(nrow(query))
    result <- landmark_umap_project_refine_cuda_gpu_cpp(
        knn, model$reference_data, query, model$fit$layout,
        seq_len(n_reference), query_rows, n_reference + nrow(query),
        as.integer(epochs), params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion,
        as.integer(model$seed + 2003L), 12L, 1e-3, 2.5
    )
    list(
        layout = result$layout[query_rows, , drop = FALSE],
        backend = "cuda"
    )
}

initialize_landmark_umap_query <- function(model, query, knn, n_threads) {
    reference_layout <- model$fit$layout
    affine <- landmark_affine_projection(
        model$reference_data, query, reference_layout, knn,
        n_threads = n_threads, backend = model$backend
    )
    if (!embedding_layout_dims_match(
        affine, nrow(query), ncol(reference_layout)
    )) {
        affine <- project_embedding_knn_cpp(
            reference_layout, knn$indices, knn$distances
        )
    }
    list(
        layout = affine,
        backend = attr(affine, "projection_backend") %||% "cpu"
    )
}

refine_landmark_umap_query_metal <- function(model, knn, rows,
                                                combined, epochs, params) {
    knn_umap_refine_rows_metal_cpp(
        knn$indices, knn$distances, as.integer(rows), combined,
        as.integer(epochs), params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion,
        as.integer(model$seed + 2003L)
    )
}

refine_landmark_umap_query_cpu <- function(model, knn, rows, combined,
                                            epochs, params, n_threads,
                                            verbose) {
    knn_umap_refine_rows_cpp(
        knn$indices, knn$distances, as.integer(rows), combined,
        as.integer(epochs), params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion,
        as.integer(normalize_nn_threads(n_threads)),
        as.integer(model$seed + 2003L), isTRUE(verbose)
    )
}

refine_landmark_umap_query <- function(model, query, knn, initial,
                                        epochs, params, n_threads, verbose) {
    n_reference <- nrow(model$reference_data)
    rows <- n_reference + seq_len(nrow(query))
    combined <- rbind(
        embedding_dense_double_matrix(model$fit$layout),
        embedding_dense_double_matrix(initial)
    )
    use_metal <- identical(model$backend, "metal")
    combined <- if (use_metal) {
        refine_landmark_umap_query_metal(
            model, knn, rows, combined, epochs, params
        )
    } else {
        refine_landmark_umap_query_cpu(
            model, knn, rows, combined, epochs, params,
            n_threads, verbose
        )
    }
    combined[seq_len(n_reference), ] <-
        embedding_dense_double_matrix(model$fit$layout)
    list(
        layout = combined[rows, , drop = FALSE],
        backend = if (use_metal) "metal" else "cpu"
    )
}

project_landmark_model_tsne <- function(model, query, projection_knn,
                                        transform_perplexity, transform_iter,
                                        transform_early_exaggeration_iter,
                                        transform_n_negatives, initialization,
                                        n_threads, verbose, dots) {
    controls <- landmark_model_tsne_controls(
        model, transform_n_negatives, dots
    )
    if (identical(model$backend, "cuda") &&
        fastembedr_is_gpu_knn(projection_knn)) {
        return(project_landmark_model_tsne_cuda(
            model, query, projection_knn, transform_perplexity,
            transform_iter, transform_early_exaggeration_iter, controls
        ))
    }
    project_landmark_model_tsne_host(
        model, query, projection_knn, transform_perplexity,
        transform_iter, transform_early_exaggeration_iter,
        initialization, n_threads, verbose, controls
    )
}

landmark_model_tsne_controls <- function(model, n_negatives, dots) {
    n_reference <- nrow(model$reference_data)
    threshold <- as.integer(dots$exact_repulsion_threshold %||% 4096L)
    if (is.null(n_negatives)) {
        n_negatives <- if (n_reference <= threshold) {
            n_reference
        } else {
            min(256L, n_reference)
        }
    }
    list(
        exact_threshold = threshold,
        n_negatives = as.integer(n_negatives),
        learning_rate = as.numeric(
            dots$transform_learning_rate %||% 0.1
        ),
        early_exaggeration = as.numeric(
            dots$transform_early_exaggeration %||% 4
        ),
        exaggeration = as.numeric(dots$transform_exaggeration %||% 1.5),
        initial_momentum = as.numeric(
            dots$transform_initial_momentum %||% 0.8
        ),
        final_momentum = as.numeric(
            dots$transform_final_momentum %||% 0.8
        ),
        max_grad_norm = as.numeric(
            dots$transform_max_grad_norm %||% 0.25
        ),
        max_step_norm = as.numeric(dots$transform_max_step_norm %||% Inf)
    )
}

project_landmark_model_tsne_cuda <- function(model, query, knn, perplexity,
                                                n_iter, exaggeration_iter,
                                                controls) {
    result <- landmark_tsne_transform_cuda_gpu_cpp(
        knn, model$reference_data, query, model$fit$layout,
        as.numeric(perplexity), as.integer(n_iter),
        as.integer(exaggeration_iter), controls$learning_rate,
        controls$early_exaggeration, controls$exaggeration,
        controls$initial_momentum, controls$final_momentum,
        controls$max_grad_norm, controls$max_step_norm,
        controls$n_negatives, controls$exact_threshold,
        as.integer(model$seed + 1009L), 12L, 1e-3, 2.5
    )
    list(layout = result$Y, backend = "cuda")
}

landmark_tsne_initial_layout <- function(model, query, knn, n_threads) {
    initial <- landmark_affine_projection(
        model$reference_data, query, model$fit$layout, knn,
        n_threads = n_threads, backend = model$backend
    )
    if (!embedding_layout_dims_match(
        initial, nrow(query), ncol(model$fit$layout)
    )) {
        return(NULL)
    }
    initial
}

project_landmark_model_tsne_host <- function(model, query, knn, perplexity,
                                                n_iter, exaggeration_iter,
                                                initialization, n_threads,
                                                verbose,
                                                controls) {
    initial <- landmark_tsne_initial_layout(
        model, query, knn, n_threads
    )
    layout <- transform_tsne(
        model$fit$layout,
        knn = knn, perplexity = perplexity,
        initialization = initialization, Y_init = initial,
        n_iter = n_iter, early_exaggeration_iter = exaggeration_iter,
        learning_rate = controls$learning_rate,
        early_exaggeration = controls$early_exaggeration,
        exaggeration = controls$exaggeration,
        initial_momentum = controls$initial_momentum,
        final_momentum = controls$final_momentum,
        max_grad_norm = controls$max_grad_norm,
        max_step_norm = controls$max_step_norm,
        n_negatives = controls$n_negatives,
        exact_repulsion_threshold = controls$exact_threshold,
        n.cores = n_threads, seed = model$seed + 1009L,
        backend = model$backend, verbose = verbose
    )
    list(
        layout = layout,
        backend = attr(layout, "backend") %||% model$backend
    )
}

#' Project observations into a fitted landmark embedding
#'
#' `project_landmark_model()` computes query-to-reference KNN with the same
#' native backend policy used by the full embedding functions, then applies the
#' method-specific fixed-reference transform. Landmark coordinates remain
#' fixed.
#'
#' @param model A model returned by [fit_landmark_model()].
#' @param data The original full data matrix for a model fitted on a strict
#'   landmark subset, or a matrix containing only new query observations in the
#'   same feature space. A model fitted with all training rows as references
#'   always treats `data` as new query observations.
#' @param query_indices Optional rows of `data` to project.
#' @param transform_k Number of reference neighbors.
#' @param refinement_epochs Fixed-reference UMAP refinement epochs.
#' @param transform_perplexity Perplexity of the t-SNE transform.
#' @param transform_iter Number of t-SNE transform iterations.
#' @param transform_early_exaggeration_iter Transform exaggeration iterations.
#' @param transform_n_negatives Number of reference negatives for t-SNE.
#' @param initialization t-SNE query initialization.
#' @param keep_knn Retain query-to-reference KNN output.
#' @param n.cores Number of CPU cores.
#' @param verbose Print optimizer progress.
#' @param ... Low-level t-SNE transform controls.
#' @return A `fastEmbedR_embedding`. For a strict landmark-subset model, passing
#'   the original full matrix reassembles `layout` in original row order.
#'   Otherwise `layout` contains only the supplied query rows. The returned
#'   parameters record `projection_scope` as `"original_reconstruction"` or
#'   `"held_out_query"`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, 0.5, seed = 1)
#' model <- fit_landmark_model(
#'     x, selection,
#'     method = "umap", n_neighbors = 10, seed = 1
#' )
#' fit <- project_landmark_model(model, x, refinement_epochs = 2)
#' @export
project_landmark_model <- function(model, data, query_indices = NULL,
                                    transform_k = NULL,
                                    refinement_epochs = 50L,
                                    transform_perplexity = 5,
                                    transform_iter = 250L,
                                    transform_early_exaggeration_iter = 0L,
                                    transform_n_negatives = NULL,
                                    initialization = c(
                                        "median", "weighted", "random"
                                    ),
                                    keep_knn = FALSE, n.cores = NULL,
                                    verbose = FALSE, ...) {
    request <- prepare_landmark_projection(
        model, data, query_indices, transform_k,
        transform_perplexity, initialization, n.cores
    )
    if (nrow(request$query) < 1L) {
        return(model$fit)
    }
    projected <- compute_landmark_projection(
        request, refinement_epochs, transform_perplexity,
        transform_iter, transform_early_exaggeration_iter,
        transform_n_negatives, verbose, list(...)
    )
    assemble_landmark_projection_fit(
        request, projected, keep_knn, transform_perplexity,
        transform_iter
    )
}

prepare_landmark_projection <- function(model, data, query_indices,
                                        transform_k,
                                        transform_perplexity,
                                        initialization, n.cores) {
    if (!inherits(model, "fastEmbedR_landmark_model")) {
        stop("`model` must be returned by fit_landmark_model().",
            call. = FALSE
        )
    }
    initialization <- match.arg(
        initialization, c("median", "weighted", "random")
    )
    n_threads <- normalize_nn_threads(n.cores %||% model$n.cores)
    query_info <- landmark_query_data(model, data, query_indices)
    transform_k <- landmark_model_projection_k(
        model, transform_k,
        transform_perplexity = transform_perplexity
    )
    list(
        model = model, query_info = query_info,
        query = query_info$data, transform_k = transform_k,
        initialization = initialization, n_threads = n_threads
    )
}

validate_landmark_refinement_epochs <- function(value) {
    value <- as.integer(value)
    if (length(value) != 1L || is.na(value) || value < 0L) {
        stop("`refinement_epochs` must be non-negative.", call. = FALSE)
    }
    value
}

run_landmark_projection_method <- function(request, knn,
                                            refinement_epochs,
                                            transform_perplexity,
                                            transform_iter,
                                            exaggeration_iter,
                                            transform_n_negatives,
                                            verbose, dots) {
    model <- request$model
    if (identical(model$method, "umap")) {
        epochs <- validate_landmark_refinement_epochs(refinement_epochs)
        result <- project_landmark_umap(
            model, request$query, knn, epochs,
            request$n_threads, verbose
        )
        return(list(result = result, refinement_epochs = epochs))
    }
    result <- project_landmark_model_tsne(
        model, request$query, knn, transform_perplexity,
        as.integer(transform_iter), as.integer(exaggeration_iter),
        transform_n_negatives, request$initialization,
        request$n_threads, verbose, dots
    )
    list(result = result, refinement_epochs = NA_integer_)
}

compute_landmark_projection <- function(request, refinement_epochs,
                                        transform_perplexity,
                                        transform_iter,
                                        exaggeration_iter,
                                        transform_n_negatives,
                                        verbose, dots) {
    model <- request$model
    knn_timed <- timed_do_call(precompute_query_knn, list(
        reference = model$reference_data, query = request$query,
        k = request$transform_k, metric = model$metric,
        backend = model$backend, n.cores = request$n_threads
    ))
    projection_timed <- timed_do_call(
        run_landmark_projection_method,
        list(
            request = request, knn = knn_timed$value,
            refinement_epochs = refinement_epochs,
            transform_perplexity = transform_perplexity,
            transform_iter = transform_iter,
            exaggeration_iter = exaggeration_iter,
            transform_n_negatives = transform_n_negatives,
            verbose = verbose, dots = dots
        )
    )
    list(
        knn = knn_timed$value, knn_time = knn_timed$time,
        projection = projection_timed$value$result,
        projection_time = projection_timed$time,
        refinement_epochs =
            projection_timed$value$refinement_epochs
    )
}

assemble_projected_landmark_layout <- function(request, projected) {
    model <- request$model
    prefix <- if (identical(model$method, "umap")) "UMAP" else "TSNE"
    query_layout <- finalize_embedding_layout(
        projected$projection$layout, prefix,
        return_float32 = is_float32_matrix(request$query)
    )
    layout <- if (request$query_info$full_input) {
        assemble_landmark_layout(
            model$fit$layout, query_layout, model$selection$indices,
            model$selection$query_indices, model$n_total,
            prefix = prefix,
            return_float32 = is_float32_matrix(
                request$query_info$input
            )
        )
    } else {
        query_layout
    }
    list(layout = layout, query_layout = query_layout)
}

landmark_projection_parameters <- function(request, projected,
                                            transform_iter) {
    model <- request$model
    approximation <- attr(projected$knn, "approximation", exact = TRUE)
    list(
        method = paste0("landmark_", model$method),
        backend = model$backend, metric = model$metric,
        n_neighbors = model$n_neighbors, perplexity = model$perplexity,
        graph_mode = model$graph_mode, seed = model$seed, landmark = TRUE,
        n_landmarks = nrow(model$reference_data),
        landmark_fraction = model$selection$landmark_fraction,
        landmark_selection = model$selection$method,
        transform_k = request$transform_k,
        projection_nn_backend = projected$knn$execution_backend %||%
            attr(projected$knn, "backend") %||% model$backend,
        projection_strategy = approximation$strategy %||%
            projected$knn$engine %||% projected$knn$method %||%
            attr(projected$knn, "method"),
        projection_backend = projected$projection$backend,
        projection_scope = if (request$query_info$full_input) {
            "original_reconstruction"
        } else {
            "held_out_query"
        },
        refinement_epochs = projected$refinement_epochs,
        transform_iter = if (model$method == "tsne") {
            as.integer(transform_iter)
        } else {
            NA_integer_
        }
    )
}

landmark_projection_metrics <- function(request, projected, layout) {
    model <- request$model
    data.frame(
        method = paste0("landmark_", model$method),
        n = nrow(layout), p = model$p,
        elapsed = model$elapsed_sec +
            projected$knn_time[["elapsed"]] +
            projected$projection_time[["elapsed"]],
        reference_embedding_elapsed = model$elapsed_sec,
        projection_knn_elapsed = projected$knn_time[["elapsed"]],
        projection_transform_elapsed =
            projected$projection_time[["elapsed"]]
    )
}

assemble_landmark_projection_fit <- function(request, projected,
                                                keep_knn,
                                                transform_perplexity,
                                                transform_iter) {
    model <- request$model
    layouts <- assemble_projected_landmark_layout(request, projected)
    timings <- rbind(
        reference_embedding =
            model$fit$timings["embedding", , drop = FALSE],
        projection_knn = projected$knn_time,
        projection_transform = projected$projection_time
    )
    retained_knn <- if (isTRUE(keep_knn)) projected$knn else NULL
    extras <- list(
        query_layout = layouts$query_layout, model = model,
        landmarks = list(
            indices = model$selection$indices,
            layout = model$fit$layout, reference_fit = model$fit,
            projection_knn = retained_knn
        )
    )
    new_embedding_result(
        layouts$layout, paste0("landmark_", model$method),
        landmark_projection_metrics(request, projected, layouts$layout),
        landmark_projection_parameters(
            request, projected, transform_iter
        ),
        timings, retained_knn, extras
    )
}
