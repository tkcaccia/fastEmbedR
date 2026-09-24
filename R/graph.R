graph_build_settings <- function(k, backend, metric, weight,
                                    mutual, prune, n.cores) {
    mutual <- isTRUE(mutual)
    prune <- as.numeric(prune)
    if (length(prune) != 1L || !is.finite(prune) || prune < 0) {
        stop("`prune` must be one finite non-negative number.",
            call. = FALSE
        )
    }
    list(
        k = graph_positive_integer(k, "k"),
        backend = resolve_embedding_backend(backend),
        metric = match.arg(
            metric,
            c("euclidean", "cosine", "correlation")
        ),
        weight = match.arg(
            weight,
            c("snn", "distance", "binary")
        ),
        mutual = mutual,
        prune = prune,
        n_threads = resolve_n_cores(n.cores)
    )
}

graph_input_source <- function(x) {
    if (inherits(x, "fastEmbedR_embedding")) {
        return(list(value = x$layout, name = "embedding"))
    }
    list(value = x, name = "precomputed_knn")
}

compute_graph_input_knn <- function(x, source, settings) {
    if (is_knn_input(x)) {
        return(list(
            value = coerce_knn_input(x),
            backend = attr(x, "backend") %||% NA_character_,
            source = source
        ))
    }
    if (!(is.matrix(x) || is.data.frame(x) || is_float32_matrix(x))) {
        stop("`x` must be data, an embedding, or a KNN object.",
            call. = FALSE
        )
    }
    if (is.null(nrow(x)) || nrow(x) < 2L) {
        stop("`x` must contain at least two rows.", call. = FALSE)
    }
    policy <- fastembedr_embedding_nn_policy(
        settings$backend,
        nrow(x)
    )
    found <- graph_data_knn(x, settings, policy)
    list(
        value = coerce_knn_input(found),
        backend = attr(found, "backend") %||% policy$backend,
        source = if (source == "embedding") "embedding" else "data"
    )
}

graph_data_knn <- function(x, settings, policy) {
    fastembedr_nn_without_self(
        x,
        k = min(settings$k, nrow(x) - 1L),
        backend = policy$backend,
        method = policy$method,
        metric = settings$metric,
        output = "double",
        n_threads = settings$n_threads,
        tuning = policy$tuning,
        target_recall = policy$target_recall,
        keep_gpu = FALSE
    )
}

materialize_graph_knn <- function(knn, requested_k) {
    k <- min(requested_k, knn$n_neighbors)
    if (k < 1L) {
        stop("The KNN input has no usable non-self neighbors.",
            call. = FALSE
        )
    }
    selected <- materialize_knn_range(
        knn$indices,
        knn$distances,
        col_start = knn$col_start,
        n_neighbors = k
    )
    indices <- as.matrix(selected$indices)
    storage.mode(indices) <- "integer"
    list(
        indices = indices,
        distances = embedding_dense_double_matrix(selected$distances),
        k = k
    )
}

finish_knn_graph <- function(graph, input, selected, settings, started) {
    requested <- if (input$source == "precomputed_knn") {
        NA_character_
    } else {
        settings$backend
    }
    graph$parameters <- list(
        k = selected$k,
        metric = settings$metric,
        weight = settings$weight,
        mutual = settings$mutual,
        prune = settings$prune,
        source = input$source,
        requested_knn_backend = requested,
        knn_backend = graph_backend_name(
            input$value$input_backend,
            input$backend
        ),
        n.cores = settings$n_threads,
        elapsed_sec = unname(proc.time()[[3L]] - started)
    )
    class(graph) <- c("fastEmbedR_graph", "list")
    graph
}

#' Build a compact nearest-neighbor graph
#'
#' `knn_graph()` converts a data matrix, a `fastEmbedR_embedding`, or a
#' precomputed KNN object into a native undirected edge graph. When neighbors
#' must be computed, the function uses fastEmbedR's existing native KNN route;
#' it does not expose a second nearest-neighbor tuning API.
#'
#' @param x Numeric data matrix, a `fastEmbedR_embedding`, a list containing
#'   `indices` and `distances`, or an existing `fastEmbedR_graph`.
#' @param k Number of non-self neighbors retained.
#' @param backend KNN-search backend used only when `x` is data: `"cpu"`,
#'   `"cuda"`, or `"metal"`.
#' @param metric Distance metric used only when KNN must be computed:
#'   `"euclidean"`, `"cosine"`, or `"correlation"`.
#' @param weight Edge weighting: shared-neighbor Jaccard (`"snn"`),
#'   inverse distance (`"distance"`), or unit weights (`"binary"`).
#' @param mutual Keep only reciprocal KNN edges.
#' @param prune Remove edges whose final weight is less than or equal to this
#'   non-negative threshold.
#' @param n.cores CPU cores used by KNN search and graph construction.
#' @return A compact `fastEmbedR_graph` list with `from`, `to`, `weight`,
#'   `n_vertices`, and construction metadata.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. J Stat Mech. 2008;2008:P10008.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' graph <- knn_graph(x, k = 15, weight = "snn")
#' graph
#' @export
knn_graph <- function(x,
                        k = 30L,
                        backend = NULL,
                        metric = c("euclidean", "cosine", "correlation"),
                        weight = c("snn", "distance", "binary"),
                        mutual = FALSE,
                        prune = 0,
                        n.cores = NULL) {
    if (inherits(x, "fastEmbedR_graph")) {
        return(validate_fastembedr_graph(x))
    }
    if (length(mutual) != 1L || is.na(mutual)) {
        stop("`mutual` must be TRUE or FALSE.", call. = FALSE)
    }
    started <- proc.time()[[3L]]
    settings <- graph_build_settings(
        k, backend, metric, weight, mutual, prune, n.cores
    )
    source <- graph_input_source(x)
    input <- compute_graph_input_knn(
        source$value,
        source$name,
        settings
    )
    selected <- materialize_graph_knn(input$value, settings$k)
    graph <- fastembedr_graph_from_knn_cpp(
        selected$indices,
        selected$distances,
        weight_type = settings$weight,
        mutual = settings$mutual,
        prune = settings$prune,
        n_threads = settings$n_threads
    )
    finish_knn_graph(graph, input, selected, settings, started)
}

# Validate and normalize graph-clustering controls.
graph_cluster_settings <- function(method, backend, resolution,
                                    n_iterations, n_runs, steps, seed) {
    method <- match.arg(method, c("leiden", "louvain", "walktrap"))
    backend <- resolve_embedding_backend(backend)
    resolution <- as.numeric(resolution)
    if (length(resolution) != 1L ||
        !is.finite(resolution) ||
        resolution <= 0) {
        stop("`resolution` must be one finite positive number.",
            call. = FALSE
        )
    }
    seed <- as.integer(seed)
    if (length(seed) != 1L || is.na(seed)) {
        stop("`seed` must be one integer.", call. = FALSE)
    }
    settings <- list(
        method = method,
        backend = backend,
        resolution = resolution,
        n_iterations = graph_positive_integer(
            n_iterations, "n_iterations"
        ),
        n_runs = graph_positive_integer(n_runs, "n_runs"),
        steps = graph_positive_integer(steps, "steps"),
        seed = seed
    )
    validate_walktrap_settings(settings)
}

validate_walktrap_settings <- function(settings) {
    if (settings$method != "walktrap") {
        return(settings)
    }
    if (!isTRUE(all.equal(settings$resolution, 1))) {
        stop("Walktrap uses its reference resolution of 1.",
            call. = FALSE
        )
    }
    if (settings$backend != "cpu") {
        stop(
            "Walktrap is supported only with `backend = \"cpu\"`.",
            call. = FALSE
        )
    }
    settings
}

graph_cluster_call_args <- function(graph, settings) {
    list(
        graph$from,
        graph$to,
        graph$weight,
        n_vertices = graph$n_vertices,
        method = settings$method,
        resolution = settings$resolution,
        n_iterations = settings$n_iterations,
        n_runs = settings$n_runs,
        seed = settings$seed
    )
}

run_graph_cluster_backend <- function(graph, settings) {
    if (settings$method == "walktrap") {
        return(fastembedr_walktrap_cpp(
            graph$from,
            graph$to,
            graph$weight,
            n_vertices = graph$n_vertices,
            steps = settings$steps
        ))
    }
    fun <- switch(settings$backend,
        cpu = fastembedr_graph_cluster_cpp,
        cuda = fastembedr_graph_cluster_cuda_cpp,
        metal = fastembedr_graph_cluster_metal_cpp
    )
    do.call(fun, graph_cluster_call_args(graph, settings))
}

graph_cluster_parameters <- function(graph, settings, started) {
    walktrap <- settings$method == "walktrap"
    list(
        resolution = if (walktrap) 1 else settings$resolution,
        n_iterations = if (walktrap) {
            NA_integer_
        } else {
            settings$n_iterations
        },
        n_runs = if (walktrap) 1L else settings$n_runs,
        steps = if (walktrap) settings$steps else NA_integer_,
        seed = if (walktrap) NA_integer_ else settings$seed,
        n_vertices = graph$n_vertices,
        n_edges = graph$n_edges,
        elapsed_sec = unname(proc.time()[[3L]] - started)
    )
}

finish_graph_cluster <- function(result, graph, settings, started) {
    result$method <- settings$method
    result$backend_requested <- settings$backend
    result$backend <- settings$backend
    result$parameters <- graph_cluster_parameters(
        graph,
        settings,
        started
    )
    class(result) <- c("fastEmbedR_graph_cluster", "list")
    result
}

#' Native graph community detection
#'
#' `graph_cluster()` clusters a compact `fastEmbedR_graph` with a native
#' multilevel Louvain implementation, a Leiden local-move/refine/aggregate
#' implementation, or the Pons-Latapy Walktrap algorithm. Louvain and Leiden
#' have CPU, CUDA, and Metal implementations. Walktrap is CPU-only. No Python,
#' `igraph`, cuGraph, or external clustering function is called at run time.
#'
#' @param graph A `fastEmbedR_graph`, or an edge-list list containing `from`,
#'   `to`, `weight`, and `n_vertices`.
#' @param method `"leiden"`, `"louvain"`, or `"walktrap"`. Walktrap is the
#'   package's true random-walk community method.
#' @param backend Execution backend. `"cuda"` and `"metal"` are available for
#'   Louvain and Leiden only and fail explicitly if the requested accelerator
#'   was not compiled or is unavailable. They never fall back to CPU.
#' @param resolution Positive modularity resolution for Louvain and Leiden.
#'   Walktrap uses its reference resolution of 1.
#' @param n_iterations Maximum local-moving passes for Louvain and Leiden.
#' @param n_runs Independent seeded Louvain/Leiden runs; the highest-modularity
#'   result is returned.
#' @param steps Random-walk length for Walktrap.
#' @param seed Reproducible random seed for Louvain and Leiden.
#' @return A `fastEmbedR_graph_cluster` containing one-based `membership`,
#'   modularity, method, implementation, and run metadata.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. J Stat Mech. 2008;2008:P10008.
#'
#' Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
#' well-connected communities. Sci Rep. 2019;9:5233.
#'
#' Pons P, Latapy M. Computing communities in large networks using random
#' walks. J Graph Algorithms Appl. 2006;10:191-218.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' graph <- knn_graph(x, k = 15)
#' clusters <- graph_cluster(graph, method = "leiden", seed = 1)
#' table(clusters$membership)
#' @export
graph_cluster <- function(graph,
                            method = c("leiden", "louvain", "walktrap"),
                            backend = NULL,
                            resolution = 1,
                            n_iterations = 10L,
                            n_runs = 1L,
                            steps = 4L,
                            seed = 1L) {
    graph <- validate_fastembedr_graph(graph)
    settings <- graph_cluster_settings(
        method, backend, resolution, n_iterations, n_runs, steps, seed
    )
    started <- proc.time()[[3L]]
    result <- run_graph_cluster_backend(graph, settings)
    finish_graph_cluster(result, graph, settings, started)
}

validate_fastembedr_graph <- function(graph) {
    required <- c("from", "to", "weight", "n_vertices")
    if (!is.list(graph) || !all(required %in% names(graph))) {
        stop(
            sprintf(
                "%s%s",
                "`graph` must be a fastEmbedR graph or an edge list with ",
                "from, to, weight, and n_vertices."
            ),
            call. = FALSE
        )
    }
    graph$from <- as.integer(graph$from)
    graph$to <- as.integer(graph$to)
    graph$weight <- as.numeric(graph$weight)
    graph$n_vertices <- graph_positive_integer(
        graph$n_vertices,
        "graph$n_vertices"
    )
    if (length(graph$from) != length(graph$to) ||
        length(graph$from) != length(graph$weight)) {
        stop("Graph edge vectors must have equal lengths.", call. = FALSE)
    }
    if (anyNA(graph$from) || anyNA(graph$to) || anyNA(graph$weight) ||
        any(!is.finite(graph$weight)) || any(graph$weight < 0)) {
        stop(
            "Graph edges must have valid vertices and finite ",
            "non-negative weights.",
            call. = FALSE
        )
    }
    if (length(graph$from) > 0L &&
        (min(graph$from) < 1L || min(graph$to) < 1L ||
            max(graph$from) > graph$n_vertices || max(
            graph$to
        ) > graph$n_vertices)) {
        stop("Graph vertices must be between 1 and n_vertices.", call. = FALSE)
    }
    graph$n_edges <- length(graph$from)
    class(graph) <- c("fastEmbedR_graph", "list")
    graph
}

graph_positive_integer <- function(x, name) {
    numeric_value <- numeric_scalar(x)
    value <- integer_scalar(numeric_value)
    if (length(numeric_value) != 1L || !is.finite(numeric_value) ||
        numeric_value != value || value < 1L) {
        stop("`", name, "` must be one positive integer.", call. = FALSE)
    }
    value
}

graph_backend_name <- function(primary, fallback) {
    valid <- function(value) {
        length(value) == 1L && !is.na(value) && nzchar(value)
    }
    if (valid(primary)) as.character(primary) else as.character(fallback)
}

#' Print a nearest-neighbor graph
#'
#' @param x A `fastEmbedR_graph` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.fastEmbedR_graph <- function(x, ...) {
    cat("fastEmbedR KNN graph\n")
    cat("  vertices: ", x$n_vertices, "\n", sep = "")
    cat("  edges: ", x$n_edges, "\n", sep = "")
    if (!is.null(x$weight_type)) {
        cat("  weights: ", x$weight_type, "\n",
            sep = ""
        )
    }
    invisible(x)
}

#' Print a graph-clustering result
#'
#' @param x A `fastEmbedR_graph_cluster` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.fastEmbedR_graph_cluster <- function(x, ...) {
    cat("fastEmbedR graph clustering\n")
    cat("  method: ", x$method, "\n", sep = "")
    cat("  backend: ", x$backend, "\n", sep = "")
    cat("  communities: ", x$n_communities, "\n", sep = "")
    cat("  modularity: ", format(x$modularity, digits = 5), "\n", sep = "")
    invisible(x)
}
