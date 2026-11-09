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
        nrow(x), ncol(x)
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
    if (fastembedr_file_backed_input(x)) stop(
        "Use precompute_knn(x, output = ...) for file-backed KNN, then ",
        "massive_umap_fuzzy_graph() for an undirected fuzzy graph.",
        call. = FALSE)
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
    graph_check_knn_capacity(source$value, settings$k,
        settings$backend)
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
                                    vectors = FALSE, n.cores = 1L) {
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
        n.cores = graph_positive_integer(n.cores, "n.cores"),
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

run_graph_cluster_multigpu <- function(graph, settings, options) {
    if (settings$n.cores != 1L)
        stop("Resident multi-GPU clustering cannot use CPU workers.",
            call. = FALSE)
    graph_cluster_require_gpu("cuda")
    identity <- massive_cuda_device_uuids(settings$devices)
    if (settings$consensus || options) {
        result <- graph_cluster_gpu_enhanced(graph, settings)
    } else {
        args <- c(graph_cluster_call_args(graph, settings),
            list(devices = settings$devices))
        result <- do.call(fastembedr_graph_cluster_cuda_devices_cpp,
            args)
    }
    result$gpu_identity <- identity
    result
}

run_graph_cluster_backend <- function(graph, settings) {
    options <- any(!vapply(settings$options, is.null, FALSE))
    if (!is.null(settings$devices)) return(
        run_graph_cluster_multigpu(graph, settings, options))
    if (settings$n.cores > 1L && (settings$backend != "cpu" ||
        settings$method == "walktrap" || settings$consensus ||
        (!settings$parallel.fit &&
        settings$n_runs < settings$n.cores))) stop(
        "CPU restart workers or fit workers require scalar ",
        "Louvain/Leiden controls and ",
        "`backend = \"cpu\"`; restart mode also needs ",
        "`n_runs >= n.cores`.", call. = FALSE)
    if ((settings$consensus || options) &&
        settings$method == "walktrap") {
        stop("Graph options require Louvain or Leiden.",
            call. = FALSE)
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
    if ((settings$consensus || options) &&
        settings$backend %in% c("cuda", "metal")) {
        graph_cluster_require_gpu(settings$backend)
        return(graph_cluster_gpu_enhanced(graph, settings))
    }
    fun <- switch(settings$backend,
        cpu = fastembedr_graph_cluster_cpp,
        cuda = fastembedr_graph_cluster_cuda_cpp,
        metal = fastembedr_graph_cluster_metal_cpp
    )
    args <- graph_cluster_call_args(graph, settings)
    if (settings$backend == "cpu") {
        args <- c(args, settings$options,
            list(n_threads = settings$n.cores,
                parallel_fit = settings$parallel.fit))
    }
    do.call(fun, args)
}

graph_cluster_gpu_consensus <- function(edges, settings, cluster_fun) {
    agreement <- numeric(length(edges$weight))
    count <- length(settings$resolution) * length(settings$seed)
    for (resolution in settings$resolution) for (seed in settings$seed) {
        partition <- cluster_fun(
            edges$from, edges$to, edges$weight, edges$n_vertices,
            settings$method, resolution, settings$n_iterations, 1L, seed)
        labels <- partition$membership
        agreement <- agreement +
            (labels[edges$from] == labels[edges$to])
    }
    edges$weight <- edges$weight * (agreement / count)^2
    keep <- edges$weight > 0
    edges[c("from", "to", "weight")] <- lapply(
        edges[c("from", "to", "weight")], `[`, keep)
    edges$n_edges <- length(edges$weight)
    list(graph = edges, metadata = list(
        method = settings$method, resolutions = settings$resolution,
        seeds = settings$seed, partitions = count, agreement_power = 2L,
        implementation = paste0(
            settings$backend, "_partitions_host_sparse_edge_consensus"
        ),
        weighted_edges = edges$n_edges))
}

graph_cluster_gpu_enhanced <- function(graph, settings) {
    cluster_fun <- graph_cluster_accel_function(settings)
    edges <- fastembedr_graph_canonical_cpp(
        graph$from, graph$to, graph$weight, graph$n_vertices)
    metadata <- list(consensus = NULL, projection = NULL,
        small_groups = NULL)
    order <- character()
    if (settings$consensus) {
        step <- graph_cluster_gpu_consensus(edges, settings, cluster_fun)
        edges <- step$graph
        metadata$consensus <- step$metadata
        order <- c(order, "consensus")
    }
    if (!is.null(settings$options$projection)) {
        step <- fastembedr_graph_projection_cpp(
            edges$from, edges$to, edges$weight, edges$n_vertices,
            settings$options$projection)
        edges <- step$graph
        metadata$projection <- c(step$metadata,
            execution = "host_native_cpp")
        order <- c(order, "projection")
    }
    if (!edges$n_edges) stop("Reweighting removed all graph edges.",
        call. = FALSE)
    result <- cluster_fun(
        edges$from, edges$to, edges$weight, edges$n_vertices,
        settings$method, settings$resolution[1L], settings$n_iterations,
        settings$n_runs, settings$seed[1L])
    order <- c(order, "clustering")
    if (settings$consensus || !is.null(settings$options$projection)) {
        result$reweighted_graph <- edges
    }
    if (!is.null(settings$options$small_groups)) {
        post <- fastembedr_graph_small_groups_cpp(
            edges$from, edges$to, edges$weight, edges$n_vertices,
            result$membership, settings$resolution[1L],
            settings$options$small_groups)
        metadata$small_groups <- c(post$metadata,
            execution = "host_native_cpp")
        post$metadata <- NULL
        result[names(post)] <- post
        order <- c(order, "small_groups")
    }
    result$parameters <- c(metadata, list(weighted_edges = edges$n_edges,
        final_resolution = settings$resolution[1L],
        final_seed = settings$seed[1L], processing_order = order))
    result
}

graph_cluster_accel_function <- function(settings) {
    if (!is.null(settings$devices)) {
        return(function(from, to, weight, n_vertices, method,
                resolution, n_iterations, n_runs, seed) {
            fastembedr_graph_cluster_cuda_devices_cpp(
                from, to, weight, n_vertices, method,
                resolution, n_iterations, n_runs, seed,
                settings$devices)
        })
    }
    switch(settings$backend,
        cuda = fastembedr_graph_cluster_cuda_cpp,
        metal = fastembedr_graph_cluster_metal_cpp)
}

graph_cluster_require_gpu <- function(backend) {
    ready <- switch(backend,
        cuda = graph_clustering_cuda_available_cpp(),
        metal = graph_clustering_metal_available_cpp())
    label <- if (identical(backend, "cuda")) "CUDA" else "Metal"
    if (!isTRUE(ready)) stop(
        label, " graph clustering unavailable; no fallback.",
        call. = FALSE)
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
        n.cores = settings$n.cores,
        parallel.mode = settings$parallel.mode,
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

graph_cluster_input_controls <- function(graph, landmarks, k,
        vote_k, k_given, vote_given, dots) {
    matrix_input <- is.matrix(graph) || is_float32_matrix(graph) ||
        inherits(graph, "fastEmbedR_massive_matrix")
    if (identical(landmarks, FALSE)) {
        if (inherits(graph, "fastEmbedR_massive_matrix")) stop(
            "Full file-backed matrix clustering needs a prepared graph.",
            call. = FALSE)
        landmarks <- NULL
    }
    routed_input <- matrix_input ||
        inherits(graph, "fastEmbedR_massive_graph")
    if (!matrix_input && (!is.null(landmarks) || k_given ||
        vote_given)) stop(
        "Landmark controls require a data matrix, not a prepared graph.",
        call. = FALSE)
    if (vote_given && is.null(landmarks)) stop(
        "`vote_k` requires `landmarks`.", call. = FALSE)
    if (!is.null(landmarks)) dots$landmarks <- landmarks
    if (k_given) dots$k <- k
    if (vote_given) dots$vote_k <- vote_k
    if (fastembedr_file_backed_input(graph) &&
        !is.null(dots$vote_k)) stop(
        "`vote_k` requires a resident data matrix.", call. = FALSE)
    list(data_input = routed_input, dots = dots)
}

#' Native graph community detection
#'
#' `graph_cluster()` clusters a matrix, compact graph, or file-backed input
#' with a native
#' multilevel Louvain implementation, a Leiden local-move/refine/aggregate
#' implementation, or the Pons-Latapy Walktrap algorithm. Louvain and Leiden
#' have CPU, CUDA, and Metal implementations. Walktrap is CPU-only. No Python,
#' `igraph`, cuGraph, or external clustering function is called at run time.
#'
#' @param graph A resident data matrix, `fastEmbedR_graph`, file-backed
#'   matrix or prepared file-backed graph. An edge-list list with `from`,
#'   `to`, `weight`, and `n_vertices` is also accepted. Data matrices build
#'   a KNN graph first; prepared graphs reuse their existing edges.
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
#' @param n.cores CPU workers for Louvain/Leiden. With
#'   `parallel.mode = "restarts"`, requires scalar controls and
#'   `n_runs >= n.cores`. With `parallel.mode = "fit"`, workers
#'   propose moves for one shared partition; coarsening and Leiden
#'   refinement remain serial.
#' @param parallel.mode `"restarts"` preserves parallel independent
#'   runs; `"fit"` parallelizes local moves within one CPU fit.
#' @param massive Optional data mode: `"auto"`, `"landmark"`, or
#'   `"out_of_core_graph"`. A file-backed graph selects full-graph mode;
#'   a matrix selects auto mode unless `storage = "file_backed"` requests
#'   a landmark fit. Resident graphs reject it.
#' @param landmarks Optional reference-row fraction, count, or indices for
#'   resident data; a count for file-backed data. `FALSE` explicitly selects
#'   full clustering for resident data. Prepared graphs reject a landmark
#'   selection.
#' @param k Neighbors in a landmark reference graph. Ignored unless
#'   `landmarks` is supplied. Prepared graphs reject it.
#' @param vote_k Neighbors for assigning non-landmark resident rows.
#'   File-backed data and prepared graphs reject it.
#' @param ... For file-backed data, storage
#'   controls such as `output`, `chunk_rows`,
#'   `memory_limit`, `checkpoint`, `resume`, and `gpu_storage`.
#'   When omitted for file-backed input, the memory budget uses
#'   `options(memory_limit = ...)`, or `"8GB"`.
#' @param storage `"auto"`, `"resident"`, or `"file_backed"`. When omitted,
#'   uses `options(storage = ...)`. Explicit `"resident"` on a
#'   [massive_matrix()] file descriptor loads float32 input after a RAM and
#'   `memory_limit` check; the limit does not bound the subsequent fit.
#'   A file-backed matrix needs `landmarks`; a full file-backed fit needs
#'   a prepared graph. These controls are rejected for a resident graph.
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
#'   With `devices`, CUDA partitions graph rows across the selected GPUs.
#'   Every device updates the same partition, with labels synchronized
#'   through host RAM after each color phase. Separate `n_runs` remain
#'   independent repeats. Consensus and graph reweighting apply to
#'   coordinated partitions and are recorded as host stages.
#'   This experimental route requires device qualification.
#'
#'   CUDA and Metal run Louvain/Leiden partitions on their devices; consensus
#'   edge agreement, exact projection weighting, and small-group assignment
#'   run on the host and are recorded as such. Enhanced options cannot be
#'   combined with `layout_merge`. Unsupported requests fail explicitly.
#'   These procedures do not request
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
#'   CUDA returns `repeatability = "not_bitwise_deterministic"`: parallel
#'   community updates can yield different partitions for the same seed.
#'   Matrix input with explicit `landmarks` returns a
#'   `fastEmbedR_landmark_clusters` result with all-row labels.
#' @param devices Optional vector of at least two distinct CUDA device IDs
#'   for a coordinated resident fit. Requires `backend = "cuda"`.
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
                            backend = NULL, resolution = 1,
                            n_iterations = 10L,
                            n_runs = 1L,
                            steps = 4L,
                            seed = 1L,
                            layout_merge = NULL,
                            projection = NULL,
                            small_groups = NULL,
                            n.cores = 1L, devices = NULL,
                            parallel.mode = c("restarts", "fit"),
                            massive = NULL, storage = NULL,
                            landmarks = NULL, k = 30L, vote_k = k, ...) {
    prepared <- graph_cluster_prepare_input(graph, storage,
        list(...), massive)
    graph <- prepared$graph
    dots <- prepared$dots
    storage <- fastembedr_storage_option(graph, storage)
    controls <- graph_cluster_input_controls(graph, landmarks,
        k, vote_k, !missing(k), !missing(vote_k), dots)
    data_input <- controls$data_input
    dots <- controls$dots
    if (is.matrix(graph) || is_float32_matrix(graph)) {
        if (!is.null(dots$landmarks)) return(
            graph_cluster_resident_landmarks(graph, method, backend,
                resolution, n_iterations, n_runs, steps, seed,
                layout_merge, projection, small_groups, n.cores,
                devices, parallel.mode, massive, dots))
    }
    if (data_input) {
        if (!is.null(layout_merge) || !is.null(projection) ||
            !is.null(small_groups)) stop(
            "These graph options need a resident graph.",
            call. = FALSE)
        return(do.call(graph_cluster_data_input, c(list(graph = graph,
            method = method, backend = backend,
            resolution = resolution, n_iterations = n_iterations,
            n_runs = n_runs, steps = steps, seed = seed,
            n.cores = n.cores, devices = devices,
            parallel.mode = parallel.mode, massive = massive,
            storage = storage), dots)))
    }
    if (!is.null(massive) || length(dots)) stop(
        "File-backed controls require a file-backed input.",
        call. = FALSE)
    graph_cluster_resident(graph, method, backend, resolution,
        n_iterations, n_runs, steps, seed, layout_merge, projection,
        small_groups, n.cores, devices, parallel.mode)
}

graph_cluster_prepare_input <- function(graph, storage, dots,
        massive) {
    if (fastembedr_resident_file_requested(graph, storage)) {
        backed <- c("output", "chunk_rows", "checkpoint", "resume",
            "gpu_storage")
        if (!is.null(massive) || any(names(dots) %in% backed)) stop(
            "Resident file loading conflicts with file-backed ",
            "clustering controls.", call. = FALSE)
        graph <- fastembedr_explicit_resident_data(graph, storage,
            dots$memory_limit %||% fastembedr_memory_option("8GB"))
        dots$memory_limit <- NULL
    }
    list(graph = graph, dots = dots)
}

graph_cluster_resident_landmarks <- function(graph, method, backend,
        resolution, n_iterations, n_runs, steps, seed,
        layout_merge, projection, small_groups, n.cores,
        devices, parallel.mode, massive, dots) {
    method <- match.arg(method, c("leiden", "louvain", "walktrap"))
    if (method == "walktrap") stop(
        "Resident landmark clustering supports Leiden or Louvain.",
        call. = FALSE)
    if (!is.null(massive) || !is.null(layout_merge) ||
        !is.null(projection) || !is.null(small_groups)) stop(
        "Resident landmark clustering cannot use file-backed or ",
        "graph postprocessing controls.", call. = FALSE)
    if (!identical(steps, 4L) || length(seed) != 1L) stop(
        "Resident landmark clustering needs scalar seed and ",
        "default Walktrap steps.", call. = FALSE)
    if (any(!nzchar(names(dots)))) stop(
        "Resident landmark controls must be named.",
        call. = FALSE)
    args <- list(data = graph, landmarks = dots$landmarks,
        method = method, backend = backend,
        resolution = resolution, n_iterations = n_iterations,
        n_runs = n_runs, seed = seed, n.cores = n.cores,
        devices = devices, parallel.mode = parallel.mode)
    do.call(cluster_landmarks,
        c(args, dots[setdiff(names(dots), "landmarks")]))
}

graph_cluster_resident <- function(graph, method, backend,
        resolution, n_iterations, n_runs, steps, seed,
        layout_merge, projection, small_groups, n.cores,
        devices, parallel.mode) {
    graph_check_resident_capacity(graph, backend)
    graph <- validate_fastembedr_graph(graph)
    settings <- graph_cluster_settings(
        method, backend, resolution, n_iterations, n_runs, steps, seed,
        vectors = TRUE, n.cores = n.cores
    )
    settings$parallel.mode <- match.arg(parallel.mode,
        c("restarts", "fit"))
    settings$parallel.fit <- settings$parallel.mode == "fit"
    if (settings$parallel.fit && (settings$backend != "cpu" ||
        settings$method == "walktrap")) stop(
        "`parallel.mode = \"fit\"` requires CPU Louvain or Leiden.",
        call. = FALSE)
    if (settings$parallel.fit && settings$n.cores > 32L) stop(
        "CPU fit mode supports at most 32 workers.", call. = FALSE)
    if (!is.null(devices)) {
        if (length(devices) < 2L) stop(
            "Resident multi-GPU clustering needs at least two devices.",
            call. = FALSE)
        settings$devices <- massive_validate_devices(devices,
            settings$backend, graph$n_vertices)
    }
    if (settings$backend == "cuda" &&
        isTRUE(graph_clustering_cuda_available_cpp()))
        graph_check_resident_vram(graph, settings$devices)
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

graph_check_resident_capacity <- function(graph, backend,
        memory_limit = fastembedr_memory_option("8GB"),
        available = massive_available_ram_bytes()) {
    if (!is.list(graph) || is.null(graph$from) ||
        is.null(graph$n_vertices)) return(invisible(NULL))
    graph_check_resident_counts(length(graph$from), graph$n_vertices,
        backend, memory_limit, available)
}

graph_check_knn_capacity <- function(x, k, backend,
        memory_limit = fastembedr_memory_option("8GB"),
        available = massive_available_ram_bytes()) {
    rows <- nrow(x)
    if (is.null(rows) && is.list(x)) rows <- nrow(x$indices)
    if (is.null(rows) || rows < 2L) return(invisible(NULL))
    graph_check_resident_counts(as.double(rows) *
        min(k, rows - 1), rows, backend, memory_limit, available)
}

graph_check_resident_counts <- function(edges, vertices, backend,
        memory_limit, available) {
    edges <- as.double(edges)
    vertices <- as.double(vertices)
    if (!is.finite(edges) || edges < 0 ||
        !is.finite(vertices) || vertices < 1) stop(
        "Resident graph counts must be finite and non-negative.",
        call. = FALSE)
    if (resolve_embedding_backend(backend) != "cpu" &&
        2 * edges > .Machine$integer.max) stop(
        "Resident GPU clustering may exceed signed-32-bit CSR offsets; ",
        "use a bounded file-backed graph.", call. = FALSE)
    estimate <- 256 * 1024^2 + 128 * edges + 64 * vertices
    limit <- massive_memory_bytes(memory_limit)
    if (!is.finite(available) || available <= 0 ||
        !is.finite(estimate) ||
        estimate > 0.7 * min(limit, available)) stop(
        "Resident clustering exceeds the RAM budget; use a bounded ",
        "file-backed graph or increase memory_limit when RAM permits.",
        call. = FALSE)
    invisible(list(max_csr_edges = 2 * edges,
        peak_ram_estimate_bytes = estimate))
}

graph_check_resident_vram <- function(graph, devices = NULL,
        free = massive_auto_cuda_free(devices, graph$n_vertices)) {
    required <- 128 * 1024^2 +
        128 * as.double(length(graph$from)) +
        256 * graph$n_vertices
    if (!is.finite(free) || free <= 0 ||
        !is.finite(required) || required > 0.65 * free) stop(
        "Resident CUDA clustering exceeds the free-VRAM budget; ",
        "use a bounded file-backed graph.", call. = FALSE)
    invisible(list(peak_vram_estimate_bytes = required,
        free_vram_bytes = free))
}

graph_cluster_data_input <- function(graph, method, backend,
        resolution, n_iterations, n_runs, steps, seed, n.cores,
        devices, parallel.mode, massive, storage, ...) {
    dots <- list(...)
    massive <- fastembedr_cluster_storage_mode(graph, massive,
        storage, dots$landmarks)
    if (fastembedr_file_backed_input(graph) &&
        is.null(dots$memory_limit))
        dots$memory_limit <- fastembedr_memory_option("8GB")
    if (is.null(massive)) massive <- if (
        inherits(graph, "fastEmbedR_massive_graph")) {
        "out_of_core_graph"
    } else "auto"
    method <- match.arg(method, c("leiden", "louvain", "walktrap"))
    default_mode <- length(parallel.mode) > 1L
    parallel.mode <- match.arg(parallel.mode,
        c("restarts", "fit"))
    if (massive == "out_of_core_graph" && default_mode &&
        n.cores > 1L) parallel.mode <- "fit"
    args <- list(x = graph, method = method, massive = massive,
        backend = backend %||% "cpu", resolution = resolution,
        n_iterations = n_iterations, seed = seed, n.cores = n.cores,
        devices = devices, parallel.mode = parallel.mode)
    if (massive != "out_of_core_graph") {
        args$n_runs <- n_runs
        args$steps <- steps
    } else if (method == "walktrap") args$steps <- steps
    do.call(massive_cluster, c(args, dots))
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
        is.na(value) || numeric_value != value || value < 1L) {
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
