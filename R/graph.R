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
                                    n_iterations, n_runs, steps, seed,
                                    vectors = FALSE) {
    method <- match.arg(method, c("leiden", "louvain", "walktrap"))
    backend <- resolve_embedding_backend(backend)
    resolution <- as.numeric(resolution)
    if (!length(resolution) || any(!is.finite(resolution)) ||
        any(resolution <= 0)) {
        stop("`resolution` must contain positive finite numbers.",
            call. = FALSE
        )
    }
    seed <- as.numeric(seed)
    if (!length(seed) || any(!is.finite(seed)) ||
        any(seed != floor(seed)) || any(abs(seed) > .Machine$integer.max)) {
        stop("`seed` must contain finite integers.", call. = FALSE)
    }
    if (!vectors && (length(resolution) != 1L || length(seed) != 1L)) {
        stop("Vector controls require graph_cluster().", call. = FALSE)
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
        seed = as.integer(seed),
        consensus = length(resolution) > 1L || length(seed) > 1L
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
    enhanced <- settings$consensus || any(!vapply(settings$options, is.null,
        FALSE))
    if (enhanced && (settings$backend != "cpu" ||
        settings$method == "walktrap")) {
        stop("Graph options require CPU Louvain or Leiden.", call. = FALSE)
    }
    if (settings$consensus && settings$n_runs != 1L) {
        stop("With vector controls use n_runs = 1; seed defines repeats.",
            call. = FALSE)
    }
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
    args <- graph_cluster_call_args(graph, settings)
    if (settings$backend == "cpu") args <- c(args, settings$options)
    do.call(fun, args)
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
    result$parameters <- c(graph_cluster_parameters(graph, settings, started),
        result$parameters %||% list())
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
#' @param resolution Positive finite modularity resolution(s). A vector
#'   automatically enables consensus across all resolution-seed combinations.
#'   The first value controls final clustering on the reweighted graph.
#'   Walktrap requires the single value 1.
#' @param n_iterations Maximum local-moving passes for Louvain and Leiden.
#' @param n_runs Independent seeded Louvain/Leiden runs; the highest-modularity
#'   result is returned for scalar controls. Must be 1 with vector controls;
#'   use `seed` to specify consensus repeats.
#' @param steps Random-walk length for Walktrap.
#' @param seed Finite integer seed(s). A vector automatically enables
#'   consensus; each seed is tested once per supplied resolution. The first
#'   seed controls the final clustering. Scalars preserve ordinary clustering.
#' @param layout_merge Optional list with a two-column `layout` (matrix or
#'   `fastEmbedR_embedding`), `min_connectivity` (default 0.2),
#'   `max_stretch` (default 2), and `min_cross_edges` (default 2). Merging
#'   requires both original-graph connectivity and layout-edge continuity.
#'   Float32 layouts are converted to double for edge scoring. This is an
#'   exploratory PAGA-inspired postprocessing rule, not PAGA.
#' @param projection Optional list with an observation-aligned, finite
#'   n-by-2 `layout` (matrix or `fastEmbedR_embedding`), local-radius `k`
#'   (default 30), `stretch` (default 4), and `cutoff` (default 0.001).
#'   Native exact double-precision 2D search finds each point's kth non-self
#'   radius. Edge length is divided by the geometric mean of endpoint radii;
#'   weights are multiplied by `exp(-max(ratio / stretch - 1, 0)^2)`.
#'   Penalties below `cutoff` are removed. Float32 layouts are explicitly
#'   converted to double; distance calculations remain double precision.
#'   The default NULL skips projection weighting. When enabled, it runs
#'   after consensus and before final clustering.
#' @param small_groups Optional list with precomputed `knn` indices and
#'   distances, `min_size` (default 20), and optional neighbor `k` (default
#'   graph construction k, otherwise 30). Small communities are assigned to
#'   their nearest supplied neighbor belonging to a large raw community.
#'   Distance ties use neighbor-column order. Groups are renumbered by size.
#'   Rows without a core neighbor receive label 0 and are reported separately;
#'   modularity and connectivity are then NA. `min_size = 1` preserves all
#'   communities. No original feature matrix is used by these options.
#'
#' @details When either `resolution` or `seed` has length greater than one,
#'   native consensus runs `length(resolution) * length(seed)` partitions
#'   using the requested `method`. Existing edge weights are multiplied by
#'   squared coassignment frequencies. Optional projection weighting follows.
#'   Final clustering uses `resolution[1]` and `seed[1]`; no additional
#'   resolution or seed is inferred. `n_iterations` applies to every partition.
#'   `small_groups`, when supplied, is applied last.
#'
#'   Vector controls, `projection`, and `small_groups` currently require CPU
#'   Louvain or Leiden and cannot be combined with `layout_merge`.
#'   Accelerator requests fail explicitly. These procedures do not request
#'   a fixed group count. Scalar controls with `projection = NULL` and no
#'   small-group postprocessing retain the ordinary clustering workflow.
#' @section Methodological inspiration:
#' Vector-triggered consensus draws on coassignment-based clustering
#' (Lancichinetti and Fortunato, 2012), multiresolution ensembles
#' (Jeub et al., 2018), and sparse consensus construction (Tandon et al.,
#' 2019). This package retains only original graph edges and multiplies their
#' weights by squared coassignment frequencies. It does not implement the
#' iterative thresholding, hierarchical consensus/null model, or additional
#' node-pair sampling of those published algorithms.
#'
#' `projection` is a custom layout-based edge penalty. Its local distance
#' normalization is related to the local-scaling principle of Zelnik-Manor
#' and Perona (2004); PARC (Stassen et al., 2020) motivates refining graph
#' edges before community detection. The soft-margin penalty on supplied
#' UMAP/t-SNE coordinates is package-specific. It does not implement
#' self-tuning spectral clustering or PARC's pruning procedure.
#' @return A `fastEmbedR_graph_cluster` containing one-based `membership`,
#'   modularity, method, implementation, and run metadata. With
#'   `layout_merge`, `membership_original`, `modularity_original`, and
#'   `layout_merge_pairs` preserve the unmerged result and decisions.
#'   Graph reweighting adds `reweighted_graph` and records the native controls
#'   and weighted edge count in `parameters`. `small_groups` adds
#'   `membership_original`, `modularity_original`, `reassigned` and
#'   `unassigned` (one-based row IDs). Reported modularity uses the weighted
#'   graph when reweighting is enabled.
#'   Enhanced workflows record `final_resolution`, `final_seed` and
#'   `processing_order`; consensus metadata records the exact ensemble size.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. J Stat Mech. 2008;2008:P10008.
#'
#' Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
#' well-connected communities. Sci Rep. 2019;9:5233.
#'
#' Pons P, Latapy M. Computing communities in large networks using random
#' walks. J Graph Algorithms Appl. 2006;10:191-218.
#'
#' Wolf FA et al. PAGA: graph abstraction reconciles clustering with
#' trajectory inference through a topology preserving map of single cells.
#' Genome Biol. 2019;20:59. doi:10.1186/s13059-019-1663-x.
#'
#' Lancichinetti A, Fortunato S. Consensus clustering in complex networks.
#' Sci Rep. 2012;2:336. \doi{10.1038/srep00336}.
#'
#' Jeub LGS, Sporns O, Fortunato S. Multiresolution consensus clustering
#' in networks. Sci Rep. 2018;8:3259. \doi{10.1038/s41598-018-21352-7}.
#'
#' Tandon A, Albeshri A, Thayananthan V, Alhalabi W, Fortunato S. Fast
#' consensus clustering in complex networks. Phys Rev E. 2019;99:042301.
#' \doi{10.1103/PhysRevE.99.042301}.
#'
#' Zelnik-Manor L, Perona P. Self-tuning spectral clustering.
#' Adv Neural Inf Process Syst. 2004;17:1601-1608.
#'
#' Stassen SV et al. PARC: ultrafast and accurate clustering of phenotypic
#' data of millions of single cells. Bioinformatics. 2020;36:2778-2786.
#' \doi{10.1093/bioinformatics/btaa042}.
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
                            seed = 1L,
                            layout_merge = NULL,
                            projection = NULL,
                            small_groups = NULL) {
    graph <- validate_fastembedr_graph(graph)
    settings <- graph_cluster_settings(
        method, backend, resolution, n_iterations, n_runs, steps, seed,
        vectors = TRUE
    )
    options <- list(projection = projection, small_groups = small_groups)
    if (!is.null(layout_merge) && (settings$consensus ||
        any(!vapply(options, is.null, FALSE)))) {
        stop("New graph options cannot be combined with layout_merge.",
            call. = FALSE)
    }
    started <- proc.time()[[3L]]
    settings$options <- graph_cluster_options(options, graph)
    result <- run_graph_cluster_backend(graph, settings)
    result <- finish_graph_cluster(result, graph, settings, started)
    if (!is.null(layout_merge)) {
        result <- graph_cluster_layout_merge(result, graph, layout_merge)
    }
    result
}

graph_cluster_options <- function(options, graph) {
    for (name in names(options)) {
        if (!is.null(options[[name]]) && !is.list(options[[name]])) {
            stop(name, " must be NULL or a list.", call. = FALSE)
        }
    }
    if (!is.null(options$projection)) {
        layout <- options$projection[["layout"]]
        if (is.null(layout)) stop("projection requires layout.",
            call. = FALSE)
        if (inherits(layout, "fastEmbedR_embedding")) layout <- layout$layout
        options$projection$layout_input_type <- if (
            is_float32_matrix(layout)) "float32" else typeof(layout)
        options$projection$layout <- embedding_dense_double_matrix(layout)
    }
    if (!is.null(options$small_groups)) {
        controls <- options$small_groups
        if (is.null(controls$knn)) {
            stop("small_groups requires knn indices and distances.",
                call. = FALSE)
        }
        nn <- coerce_knn_input(controls$knn)
        k <- controls[["k"]] %||% graph$parameters[["k"]] %||% 30L
        nn <- materialize_graph_knn(nn, graph_positive_integer(k, "k"))
        controls[c("knn", "k")] <- NULL
        controls$indices <- nn$indices
        controls$distances <- nn$distances
        options$small_groups <- controls
    }
    options
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
