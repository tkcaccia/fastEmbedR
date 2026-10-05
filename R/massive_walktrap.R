massive_walktrap_plan <- function(graph, output, chunk_rows,
        memory_limit, backend, steps) {
    massive_louvain_graph_check(graph, backend)
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) stop(
        "`output` must be one new label-file path.", call. = FALSE)
    steps <- graph_positive_integer(steps, "steps")
    requested <- integer_scalar(chunk_rows)
    if (is.na(requested) || requested < 2L) stop(
        "`chunk_rows` must be at least two for Walktrap.",
        call. = FALSE)
    limit <- massive_memory_bytes(memory_limit)
    fixed <- 64 * 1024^2
    available <- 0.6 * limit - fixed
    if (available <= 0) stop(
        "Walktrap partition exceeds the memory budget.",
        call. = FALSE)
    dense <- floor(sqrt(available / 16))
    edges <- floor(min(128e6, 0.2 * limit) /
        (20 * graph$max_degree))
    rows <- min(requested, 4000L, dense, edges)
    if (!is.finite(rows) || rows < 2L) stop(
        "Walktrap partition exceeds the memory budget.",
        call. = FALSE)
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    if (file.exists(path) || length(Sys.glob(paste0(path, ".*")))) {
        stop("Walktrap output or partial state already exists.",
            call. = FALSE)
    }
    if (4 * graph$n_vertices + 64 * graph$n_edges >
        0.8 * massive_disk_available_cpp(path)) stop(
        "Walktrap contraction exceeds the free-disk budget.",
        call. = FALSE)
    list(path = path, rows = as.integer(rows), steps = steps,
        memory_limit_bytes = limit)
}

massive_walktrap_block <- function(graph, first, count, steps) {
    edges <- massive_read_graph_edges(graph, first, count)
    last <- first + count - 1
    inside <- edges$to >= first & edges$to <= last
    keep <- edges$from < edges$to & inside
    local <- list(from = as.integer(edges$from[keep] - first + 1),
        to = as.integer(edges$to[keep] - first + 1),
        weight = edges$weight[keep], n_vertices = count)
    labels <- graph_cluster(local, method = "walktrap",
        backend = "cpu", steps = steps)$membership
    list(membership = labels,
        retained_weight = sum(edges$weight[inside]),
        total_weight = sum(edges$weight))
}

massive_walktrap_level <- function(graph, output, plan) {
    con <- file(output, "wb")
    on.exit(close(con))
    first <- 1
    groups <- 0L
    blocks <- 0L
    retained <- 0
    total <- 0
    while (first <= graph$n_vertices) {
        count <- as.integer(min(plan$rows,
            graph$n_vertices - first + 1))
        block <- massive_walktrap_block(graph, first, count, plan$steps)
        labels <- block$membership
        writeBin(as.integer(labels + groups), con, size = 4L,
            endian = "little")
        groups <- groups + max(labels)
        retained <- retained + block$retained_weight
        total <- total + block$total_weight
        first <- first + count
        blocks <- blocks + 1L
        if (blocks %% 1000L == 0L) message(
            "EXPERIMENTAL Walktrap: ", first - 1, "/",
            graph$n_vertices, " vertices")
    }
    close(con)
    on.exit(NULL, add = FALSE)
    if (file.info(output)$size != 4 * graph$n_vertices) stop(
        "Walktrap label file has an incorrect size.", call. = FALSE)
    score <- massive_graph_modularity(graph, output, groups,
        chunk_rows = plan$rows,
        memory_limit = plan$memory_limit_bytes)
    list(membership_path = output,
        membership_identity = massive_checkpoint_file_identity(output),
        n_communities = groups,
        total_edge_weight = score$total_edge_weight,
        resources = score$resources,
        modularity = score$modularity, blocks = blocks,
        retained_weight_fraction = if (total > 0) {
            retained / total
        } else NA_real_)
}

massive_walktrap_merge <- function(contracted, part, original,
        plan, prefix) {
    estimate <- 64 * 1024^2 + 128 * contracted$n_vertices +
        192 * contracted$n_edges + 16 * contracted$n_vertices^2
    if (contracted$n_vertices > 4000L ||
        estimate > 0.6 * plan$memory_limit_bytes) return(NULL)
    graph <- massive_read_contracted_cpp(contracted$path,
        contracted$n_vertices, contracted$n_edges)
    fit <- graph_cluster(graph, method = "walktrap",
        backend = "cpu", steps = plan$steps)
    mapping <- paste0(prefix, ".merge.u32")
    writeBin(as.integer(fit$membership), mapping,
        size = 4L, endian = "little")
    massive_remap_louvain_cpp(part, mapping,
        original, contracted$n_vertices)
    fit$n_communities
}

massive_walktrap_hierarchy <- function(graph, plan) {
    part <- paste0(plan$path, ".part")
    current <- graph
    levels <- list()
    for (index in seq_len(64L)) {
        prefix <- paste0(plan$path, ".level", index - 1L)
        mapping <- if (index == 1L) part else
            paste0(prefix, ".labels.u32")
        level <- massive_walktrap_level(current, mapping, plan)
        levels[[index]] <- list(vertices = current$n_vertices,
            communities = level$n_communities, blocks = level$blocks,
            retained_weight_fraction = level$retained_weight_fraction)
        if (index > 1L) massive_remap_louvain_cpp(part, mapping,
            graph$n_vertices, current$n_vertices)
        if (current$n_vertices <= plan$rows) return(list(
            part = part, groups = level$n_communities,
            levels = levels))
        if (level$n_communities >= current$n_vertices) stop(
            "Partitioned Walktrap made no contraction progress; ",
            "partial labels retained.", call. = FALSE)
        contracted <- massive_louvain_contract(current, level,
            prefix, plan)
        merged <- massive_walktrap_merge(contracted, part,
            graph$n_vertices, plan, prefix)
        if (!is.null(merged)) return(list(part = part,
            groups = merged, levels = levels))
        current <- massive_louvain_coarse_csr(contracted,
            paste0(prefix, ".csr"), plan$memory_limit_bytes)
    }
    stop("Partitioned Walktrap exceeded 64 levels; partial labels ",
        "retained.", call. = FALSE)
}

#' EXPERIMENTAL partitioned Walktrap on a file-backed fuzzy graph
#'
#' Runs the existing exact Walktrap kernel on bounded induced graph blocks,
#' contracts their communities, and repeats on the contracted graph. This
#' is an approximation when more than one block is needed: cross-block
#' walks are absent from the first level, and results depend on vertex
#' order. A graph that cannot contract
#' fails explicitly, leaving partial labels for inspection. Only CPU is
#' supported. Exact in-memory `graph_cluster()` is unchanged.
#'
#' @param graph A symmetric file-backed fuzzy graph.
#' @param output New `.clusters.u32` path.
#' @param steps Random-walk length.
#' @param chunk_rows Maximum vertices per induced graph block.
#' @param memory_limit Host working-memory budget.
#' @param backend Must be `"cpu"`.
#' @return File-backed Walktrap labels and approximation metadata.
#' @export
massive_walktrap <- function(graph, output, steps = 4L,
        chunk_rows = 4000L, memory_limit = "1GB", backend = "cpu") {
    plan <- massive_walktrap_plan(graph, output, chunk_rows,
        memory_limit, backend, steps)
    message("EXPERIMENTAL partitioned Walktrap: ",
        graph$n_vertices, " vertices; block rows=", plan$rows)
    hierarchy <- massive_walktrap_hierarchy(graph, plan)
    score <- massive_graph_modularity(graph, hierarchy$part,
        hierarchy$groups, chunk_rows = plan$rows,
        memory_limit = plan$memory_limit_bytes)
    if (!file.rename(hierarchy$part, plan$path)) stop(
        "Cannot finalize Walktrap label file.", call. = FALSE)
    result <- list(membership_path = plan$path,
        membership_identity = massive_checkpoint_file_identity(
            plan$path), n_vertices = graph$n_vertices,
        n_communities = hierarchy$groups,
        modularity_final = score$modularity,
        method = "walktrap", backend = "cpu",
        implementation = "partitioned_walktrap",
        approximate = length(hierarchy$levels) > 1L ||
            hierarchy$levels[[1L]]$blocks > 1L,
        initial_retained_weight_fraction =
            hierarchy$levels[[1L]]$retained_weight_fraction,
        graph_identity = graph$file_identity,
        steps = plan$steps, block_rows = plan$rows,
        memory_limit_bytes = plan$memory_limit_bytes,
        levels = hierarchy$levels, mode = "out_of_core_graph",
        experimental = TRUE)
    class(result) <- "fastEmbedR_massive_walktrap"
    result
}

#' Read bounded labels from an experimental Walktrap fit
#'
#' @param x A `massive_walktrap()` result.
#' @param first First one-based vertex row.
#' @param n Number of labels to read; capped at one million.
#' @return One-based integer community labels.
#' @export
massive_read_walktrap_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_walktrap")) stop(
        "`x` must be a massive_walktrap() result.", call. = FALSE)
    massive_read_label_rows(x, first, n)
}

#' @export
print.fastEmbedR_massive_walktrap <- function(x, ...) {
    cat("EXPERIMENTAL partitioned Walktrap\n")
    cat("  vertices: ", format(x$n_vertices, scientific = FALSE),
        "; communities: ", x$n_communities, "\n", sep = "")
    cat("  approximate: ", x$approximate,
        "; modularity: ", signif(x$modularity_final, 6), "\n",
        sep = "")
    cat("  first-level retained edge weight: ",
        signif(x$initial_retained_weight_fraction, 4), "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_walktrap <- function(x, n = 6L, ...) {
    massive_read_walktrap_rows(x, n = min(n, x$n_vertices))
}
