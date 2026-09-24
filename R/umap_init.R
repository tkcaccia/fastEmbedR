#' Precompute a reusable UMAP initialization
#'
#' `umap_init()` separates UMAP neighbor search, graph construction, and
#' initialization from stochastic optimization. The returned object can be
#' passed directly to [umap_knn()] so repeated or cross-backend runs reuse the
#' same KNN graph and initial coordinates.
#'
#' @param x Numeric data, a KNN object, or an object returned by
#'   [prepare_umap_knn()].
#' @param distances Optional KNN distance matrix when `x` is an index matrix.
#' @param n_neighbors Number of non-self neighbors when `x` is data. `NULL`
#'   uses the package's size-aware default.
#' @param n_components Output dimensionality.
#' @param metric Distance metric used only when KNN must be computed.
#' @param backend Initialization backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param seed Random seed.
#' @param n.cores Number of CPU cores.
#' @param graph_mode UMAP graph weighting. `"fuzzy"` is the standard default;
#'   `"binary"` requests the adjacency-only sensitivity graph.
#'
#' @details
#' For the fuzzy graph, CPU initialization is computed from the prepared CSR
#' graph, while native Metal or CUDA spectral kernels are used when requested
#' and available. Binary initialization is computed from the same unit-weight
#' CSR graph used by the optimizer, so its mathematical input is unchanged.
#' Explicit GPU requests fail when the corresponding native backend is
#' unavailable.
#'
#' CUDA KNN input is materialized on the host only for this standalone
#' diagnostic API. The ordinary one-call CUDA UMAP path remains device
#' resident.
#'
#' @return A `fastEmbedR_umap_initialization` object with `layout`, `prepared`,
#'   resolved `parameters`, and component timings. Pass the complete object to
#'   [umap_knn()] or use its `layout` component directly for diagnostics.
#'
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' initialization <- umap_init(
#'     x,
#'     n_neighbors = 15,
#'     backend = "cpu",
#'     graph_mode = "fuzzy",
#'     seed = 1
#' )
#' layout <- umap_knn(initialization, backend = "cpu", seed = 1)
#' plot(layout, pch = 21, bg = iris$Species)
#' @export
umap_init <- function(x, distances = NULL, n_neighbors = NULL,
                        n_components = 2L,
                        metric = c("euclidean", "cosine", "correlation"),
                        backend = NULL, seed = 4L, n.cores = NULL,
                        graph_mode = c("fuzzy", "binary")) {
    graph_mode_missing <- missing(graph_mode)
    request <- validate_umap_init_request(
        x, backend, n_components, graph_mode, graph_mode_missing
    )
    graph <- prepare_umap_init_graph(
        x, distances, n_neighbors, metric, request$backend,
        n.cores, request$graph_mode
    )
    cfg <- configure_umap_init(
        graph$prepared$config, request, n.cores
    )
    initialized <- compute_umap_initialization(
        graph$prepared, cfg, seed
    )
    validate_umap_initialization(
        initialized$layout, graph$prepared, request$n_components
    )
    assemble_umap_initialization(
        graph, initialized, cfg, seed, request$n_components
    )
}

validate_umap_init_request <- function(x, backend, n_components,
                                        graph_mode, graph_mode_missing) {
    backend <- resolve_embedding_backend(backend)
    n_components <- validate_n_components(n_components)
    supplied <- inherits(x, "fastEmbedR_umap_prepared")
    graph_mode <- match.arg(graph_mode, c("fuzzy", "binary"))
    if (supplied && graph_mode_missing) {
        graph_mode <- x$config$graph_mode %||% graph_mode
    }
    if (supplied &&
        !identical(as.character(x$config$graph_mode), graph_mode)) {
        stop(
            "`graph_mode` does not match the prepared UMAP graph.",
            call. = FALSE
        )
    }
    if (n_components != 2L && backend %in% c("cuda", "metal")) {
        stop(
            "Native Metal and CUDA UMAP initialization supports ",
            "two dimensions.",
            call. = FALSE
        )
    }
    list(
        backend = backend, n_components = n_components,
        graph_mode = graph_mode, supplied = supplied
    )
}

prepare_umap_init_from_knn <- function(x, distances, backend,
                                        n.cores, graph_mode) {
    timed <- timed_do_call(prepare_umap_knn, list(
        indices = x, distances = distances, backend = backend,
        n.cores = n.cores, graph_mode = graph_mode
    ))
    list(
        prepared = timed$value, knn_elapsed = 0,
        graph_elapsed = unname(timed$time[["elapsed"]])
    )
}

prepare_umap_init_from_data <- function(x, n_neighbors, metric,
                                        backend, n.cores, graph_mode) {
    metric <- resolve_embedding_metric(metric, x)
    if (is.null(n_neighbors)) {
        n_neighbors <- auto_embedding_k(
            nrow(x),
            method = "umap", include_self = FALSE
        )
    }
    knn_timed <- timed_do_call(precompute_knn, list(
        data = x, k = as.integer(n_neighbors), metric = metric,
        backend = backend, n.cores = n.cores
    ))
    knn <- knn_timed$value
    if (fastembedr_is_gpu_knn(knn)) {
        knn <- fastembedr_gpu_knn_to_host(knn)
    }
    graph_timed <- timed_do_call(prepare_umap_knn, list(
        indices = knn, backend = backend, n.cores = n.cores,
        graph_mode = graph_mode
    ))
    list(
        prepared = graph_timed$value,
        knn_elapsed = unname(knn_timed$time[["elapsed"]]),
        graph_elapsed = unname(graph_timed$time[["elapsed"]])
    )
}

prepare_umap_init_graph <- function(x, distances, n_neighbors, metric,
                                    backend, n.cores, graph_mode) {
    if (inherits(x, "fastEmbedR_umap_prepared")) {
        if (!is.null(distances)) {
            stop(
                "Do not pass `distances` with a prepared UMAP object.",
                call. = FALSE
            )
        }
        return(list(
            prepared = x, knn_elapsed = 0, graph_elapsed = 0
        ))
    }
    if (!is.null(distances) || is_knn_input(x)) {
        return(prepare_umap_init_from_knn(
            x, distances, backend, n.cores, graph_mode
        ))
    }
    prepare_umap_init_from_data(
        x, n_neighbors, metric, backend, n.cores, graph_mode
    )
}

configure_umap_init_threads <- function(cfg, n.cores) {
    if (is.null(n.cores)) {
        cfg$n.cores_requested <- cfg$n_threads
        cfg$n.cores_effective <- cfg$n_threads
        return(cfg)
    }
    requested <- integer_scalar(n.cores)
    if (length(requested) != 1L || is.na(requested) || requested < 1L) {
        stop("`n.cores` must be NULL or a positive integer.", call. = FALSE)
    }
    cfg$n.cores_requested <- as.integer(requested)
    cfg$n_threads <- max(1L, min(4L, requested))
    cfg$n.cores_effective <- cfg$n_threads
    cfg
}

configure_umap_init <- function(cfg, request, n.cores) {
    cfg$backend <- request$backend
    cfg$graph_mode <- request$graph_mode
    cfg$initialization_n_components <- request$n_components
    configure_umap_init_threads(cfg, n.cores)
}

native_umap_initialization <- function(prepared, cfg, seed, n_components) {
    knn <- prepared$knn
    materialized <- materialize_knn_range(
        knn$indices, knn$distances, knn$col_start, knn$n_neighbors
    )
    distances <- if (is_float32_matrix(materialized$distances)) {
        matrix(
            as.numeric(materialized$distances),
            nrow = nrow(materialized$indices),
            ncol = ncol(materialized$indices)
        )
    } else {
        materialized$distances
    }
    result <- spectral_knn_init(
        materialized$indices, distances,
        n_components = n_components,
        min_dist = cfg$min_dist, spectral_n_iter = cfg$spectral_n_iter,
        seed = seed, backend = cfg$backend, n_threads = cfg$n_threads
    )
    scale_embedding_sdev_r(result, cfg$init_scale)
}

compute_umap_initialization <- function(prepared, cfg, seed) {
    timed <- system.time({
        native <- cfg$backend %in% c("cuda", "metal") &&
            identical(cfg$graph_mode, "fuzzy")
        layout <- if (native) {
            native_umap_initialization(
                prepared, cfg, seed, cfg$initialization_n_components %||% 2L
            )
        } else {
            umap_init_from_csr_graph(
                prepared$graph,
                n_components = cfg$initialization_n_components %||% 2L,
                cfg = cfg, seed = seed, verbose = FALSE
            )
        }
    })
    list(layout = layout, time = timed)
}

validate_umap_initialization <- function(layout, prepared, n_components) {
    expected <- c(
        nrow(prepared$knn$indices), as.integer(n_components)
    )
    if (!is.matrix(layout) || !identical(dim(layout), expected) ||
        any(!is.finite(layout))) {
        stop("UMAP initialization produced an invalid layout.", call. = FALSE)
    }
    invisible(TRUE)
}

assemble_umap_initialization <- function(graph, initialized, cfg,
                                            seed, n_components) {
    cfg$initialization_n_components <- as.integer(n_components)
    cfg$init_backend <- attr(initialized$layout, "backend") %||% "cpu"
    cfg$initialization_reusable <- TRUE
    cfg$initialization_seed <- as.integer(seed)
    attr(initialized$layout, "fastEmbedR_config") <- public_core_config(cfg)
    graph$prepared$initialization <- initialized$layout
    graph$prepared$initialization_parameters <- cfg
    elapsed <- unname(initialized$time[["elapsed"]])
    out <- list(
        layout = initialized$layout, prepared = graph$prepared,
        parameters = public_core_config(cfg),
        timings = data.frame(
            stage = c("knn", "graph", "initialization", "total"),
            elapsed_sec = c(
                graph$knn_elapsed, graph$graph_elapsed, elapsed,
                graph$knn_elapsed + graph$graph_elapsed + elapsed
            ),
            stringsAsFactors = FALSE
        )
    )
    class(out) <- c("fastEmbedR_umap_initialization", "list")
    out
}
#' Print a reusable UMAP initialization
#'
#' @param x A `fastEmbedR_umap_initialization` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.fastEmbedR_umap_initialization <- function(x, ...) {
    cat("<fastEmbedR_umap_initialization>\n")
    cat("  observations: ", nrow(x$layout), "\n", sep = "")
    cat("  components:   ", ncol(x$layout), "\n", sep = "")
    cat("  graph mode:   ", x$parameters$graph_mode, "\n", sep = "")
    cat("  init backend: ", x$parameters$init_backend, "\n", sep = "")
    total <- x$timings$elapsed_sec[x$timings$stage == "total"]
    if (length(total) && is.finite(total)) {
        cat("  elapsed:      ", format(round(total, 3L), nsmall = 3L), " s\n",
            sep = ""
        )
    }
    invisible(x)
}
