massive_modularity_resources <- function(graph, chunk_rows, memory_limit,
        n_communities) {
    limit <- massive_memory_bytes(memory_limit)
    rows <- integer_scalar(chunk_rows)
    if (is.na(rows) || rows < 1L) {
        stop("`chunk_rows` must be a positive integer.", call. = FALSE)
    }
    fixed <- 32 * 1024^2 + 16 * n_communities
    available <- floor(0.7 * limit - fixed - 8 * (rows + 1))
    edges <- min(floor(available / 8), 16e6, 65536 * rows)
    if (!is.finite(edges) || edges < graph$max_degree) {
        stop("Graph scan buffers exceed `memory_limit`.", call. = FALSE)
    }
    list(chunk_rows = rows, edge_budget = edges,
        estimated_heap_bytes = fixed + 8 * (rows + 1) + 8 * edges,
        label_mmap_bytes = 4 * graph$n_vertices,
        memory_limit_bytes = limit)
}

#' Measure modularity of a file-backed fuzzy graph and disk-backed labels
#'
#' Scans each undirected edge of a symmetric CSR fuzzy graph once, using
#' bounded buffers. Labels are one-based little-endian `uint32` values and
#' are memory-mapped; their pages may contribute to process RSS. This is a
#' CPU-only quality diagnostic, not a full-graph clustering algorithm.
#'
#' @param graph A result of [massive_umap_fuzzy_graph()].
#' @param membership Path to a `uint32` label file, or a
#'   `fastEmbedR_massive_clusters` descriptor.
#' @param n_communities Number of labels for a file path; inferred from a
#'   cluster descriptor.
#' @param resolution Positive modularity resolution.
#' @param chunk_rows Maximum CSR rows scanned per chunk.
#' @param memory_limit RAM budget for native scan buffers. Memory-mapped
#'   label pages are tracked separately and are not a strict RSS cap.
#' @param backend Only `"cpu"` is supported; other requests fail explicitly.
#' @return Modularity, internal and total edge weights, edge-pair count,
#'   backend, and a buffer/mapping resource estimate.
#' @export
massive_graph_modularity <- function(graph, membership,
        n_communities = NULL, resolution = 1,
        chunk_rows = 8192L, memory_limit = "1GB", backend = "cpu") {
    if (!identical(backend, "cpu")) stop(
        "Massive modularity requires CPU; no backend fallback was used.",
        call. = FALSE)
    if (.Platform$OS.type != "unix") stop(
        "Massive modularity requires POSIX memory mapping.",
        call. = FALSE)
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !identical(graph$weight_method, "umap_fuzzy_union") ||
        !isTRUE(graph$symmetrized)) stop(
        "`graph` must be a symmetric massive UMAP fuzzy graph.",
        call. = FALSE)
    if (inherits(membership, "fastEmbedR_massive_clusters")) {
        if (membership$nrow != graph$n_vertices) stop(
            "Membership and graph vertex counts differ.", call. = FALSE)
        n_communities <- membership$n_communities
        membership <- membership$membership_path
    }
    if (!is.character(membership) || length(membership) != 1L ||
        is.na(membership)) stop(
        "`membership` must be one label file path.", call. = FALSE)
    groups <- integer_scalar(n_communities)
    if (is.na(groups) || groups < 1L || groups > graph$n_vertices ||
        !is.numeric(resolution) || length(resolution) != 1L ||
        !is.finite(resolution) || resolution <= 0) stop(
        "Invalid community count or resolution.", call. = FALSE)
    resources <- massive_modularity_resources(graph, chunk_rows,
        memory_limit, groups)
    path <- normalizePath(membership, mustWork = TRUE)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path, path)
    before <- massive_checkpoint_file_identity(paths)
    if (!identical(massive_checkpoint_file_identity(paths[1:3]),
        graph$file_identity) || before$bytes[[4L]] !=
        4 * graph$n_vertices) stop(
        "Graph or membership files changed or have invalid size.",
        call. = FALSE)
    value <- massive_graph_modularity_cpp(paths[[1L]], paths[[2L]],
        paths[[3L]], path, graph$n_vertices, groups, resolution,
        resources$chunk_rows, resources$edge_budget)
    if (!identical(before, massive_checkpoint_file_identity(paths))) {
        stop("Graph or membership files changed during scan.",
            call. = FALSE)
    }
    c(value, list(resources = resources, experimental = TRUE))
}
