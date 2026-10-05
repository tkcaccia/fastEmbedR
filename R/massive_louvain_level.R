massive_louvain_resources <- function(graph, chunk_rows, memory_limit) {
    rows <- integer_scalar(chunk_rows)
    if (is.na(rows) || rows < 1L) stop(
        "`chunk_rows` must be a positive integer.", call. = FALSE)
    limit <- massive_memory_bytes(memory_limit)
    fixed <- 64 * 1024^2 + 64 * graph$max_degree
    available <- floor(0.7 * limit - fixed - 16 * (rows + 1))
    edges <- min(floor(available / 16), 16e6,
        max(graph$max_degree, 65536 * rows))
    if (!is.finite(edges) || edges < graph$max_degree) stop(
        "Louvain graph buffers exceed `memory_limit`.",
        call. = FALSE)
    list(chunk_rows = rows, edge_budget = edges,
        estimated_heap_bytes = fixed + 16 * (rows + 1) + 16 * edges,
        mapped_state_bytes = 16 * graph$n_vertices,
        memory_limit_bytes = limit)
}

massive_louvain_output <- function(output, bytes) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) stop(
        "`output` must be one new label-file path.", call. = FALSE)
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    pending <- paste0(path, c("", ".part", ".counts.part",
        ".volumes.part"))
    if (any(file.exists(pending))) stop(
        "Louvain output or partial state already exists.",
        call. = FALSE)
    if (bytes > 0.8 * massive_disk_available_cpp(path)) stop(
        "Louvain mapped state exceeds the free-disk budget.",
        call. = FALSE)
    path
}

massive_louvain_graph_check <- function(graph, backend) {
    if (!identical(backend, "cpu")) stop(
        "EXPERIMENTAL file-backed Louvain requires CPU; no fallback.",
        call. = FALSE)
    if (.Platform$OS.type != "unix") stop(
        "File-backed Louvain requires POSIX memory mapping.",
        call. = FALSE)
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !identical(graph$weight_method, "umap_fuzzy_union") ||
        !isTRUE(graph$symmetrized)) stop(
        "`graph` must be a symmetric massive UMAP fuzzy graph.",
        call. = FALSE)
}

massive_louvain_controls <- function(max_passes, resolution, seed) {
    passes <- integer_scalar(max_passes)
    run_seed <- integer_scalar(seed)
    if (is.na(passes) || passes < 1L || is.na(run_seed) ||
        run_seed < 0L || !is.numeric(resolution) ||
        length(resolution) != 1L || !is.finite(resolution) ||
        resolution <= 0) stop(
        "Invalid Louvain pass count, resolution, or seed.",
        call. = FALSE)
    list(passes = passes, resolution = resolution, seed = run_seed)
}

massive_louvain_run_level <- function(graph, output, passes,
        resolution, seed, chunk_rows, memory_limit, contracted,
        level_index, initial = NULL) {
    resources <- massive_louvain_resources(graph, chunk_rows,
        memory_limit)
    path <- massive_louvain_output(output,
        resources$mapped_state_bytes)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    before <- massive_checkpoint_file_identity(paths)
    if (!identical(before, graph$file_identity)) stop(
        "Massive graph files changed since construction.",
        call. = FALSE)
    initial_path <- if (is.null(initial)) "" else
        initial$parent_mapping_path
    initial_count <- if (is.null(initial)) 0L else initial$n_parent
    if (!is.null(initial) &&
        (!isTRUE(all.equal(initial$n_refined, graph$n_vertices)) ||
        !identical(massive_checkpoint_file_identity(initial_path),
            initial$parent_mapping_identity))) stop(
        "Initial Leiden partition changed or has wrong size.",
        call. = FALSE)
    message("EXPERIMENTAL Louvain level ", level_index, ": ",
        graph$n_vertices, " vertices; ", graph$n_edges,
        " directed edges; mapped state=",
        round(resources$mapped_state_bytes / 1024^3, 2), " GiB")
    result <- massive_louvain_level_cpp(paths[[1L]], paths[[2L]],
        paths[[3L]], graph$n_vertices, path, passes, resolution,
        resources$chunk_rows, resources$edge_budget, seed,
        contracted, initial_path, initial_count)
    if (!identical(before, massive_checkpoint_file_identity(paths))) {
        stop("Massive graph files changed during Louvain.",
            call. = FALSE)
    }
    result$membership_identity <- massive_checkpoint_file_identity(path)
    result$graph_identity <- before
    result$method <- paste0("louvain_local_move_level_", level_index)
    result$resources <- resources
    result$seed <- seed
    result$resolution <- resolution
    result$experimental <- TRUE
    class(result) <- "fastEmbedR_massive_louvain_level"
    result
}

#' EXPERIMENTAL first local-moving level of file-backed Louvain
#'
#' Applies the existing Louvain modularity move rule to every vertex of a
#' symmetric file-backed UMAP fuzzy graph. CSR blocks are read from a seeded
#' start in cyclic order; rows within each block are shuffled. Labels,
#' counts, and volumes are memory-mapped. This function returns one
#' local-moving level; use
#' [massive_louvain()] for memory-gated multilevel Louvain. Mapped pages
#' may contribute to RSS, so `memory_limit` bounds scan buffers rather than
#' total process memory. Interrupted runs retain partial files for inspection
#' and cannot yet resume.
#'
#' @param graph A result of [massive_umap_fuzzy_graph()].
#' @param output New path for one-based `uint32` cluster labels.
#' @param max_passes Maximum local-moving sweeps.
#' @param resolution Positive modularity resolution.
#' @param seed Nonnegative seed controlling the scan start of each pass.
#' @param chunk_rows Maximum consecutive CSR rows per scan block.
#' @param memory_limit RAM budget for graph scan buffers.
#' @param backend Only `"cpu"` is implemented; other requests fail.
#' @return A lightweight file-backed first-level Louvain partition.
#' @export
massive_louvain_level <- function(graph, output, max_passes = 10L,
        resolution = 1, seed = 1L, chunk_rows = 8192L,
        memory_limit = "1GB", backend = "cpu") {
    massive_louvain_graph_check(graph, backend)
    controls <- massive_louvain_controls(max_passes, resolution,
        seed)
    massive_louvain_run_level(graph, output, controls$passes,
        controls$resolution, controls$seed, chunk_rows,
        memory_limit, FALSE, 0L)
}

massive_read_label_rows <- function(x, first, n) {
    start <- as.numeric(first)
    count <- integer_scalar(n)
    if (length(start) != 1L || !is.finite(start) || start < 1 ||
        start != floor(start) || is.na(count) || count < 1L ||
        count > 1e6 || count > x$n_vertices - start + 1) stop(
        "Requested label rows are invalid or too large.",
        call. = FALSE)
    identity <- tryCatch(
        massive_checkpoint_file_identity(x$membership_path),
        error = function(e) NULL)
    if (!identical(identity, x$membership_identity)) stop(
        "Cluster label file changed since fitting.", call. = FALSE)
    con <- file(x$membership_path, "rb")
    on.exit(close(con))
    seek(con, where = 4 * (start - 1), origin = "start")
    readBin(con, integer(), n = count, size = 4L,
        endian = "little")
}

#' Read bounded labels from an experimental Louvain level
#'
#' @param x A result of [massive_louvain_level()].
#' @param first First one-based vertex row.
#' @param n Number of labels to read; capped at one million.
#' @return One-based integer community labels.
#' @export
massive_read_louvain_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_louvain_level")) stop(
        "`x` must be a massive_louvain_level() result.",
        call. = FALSE)
    massive_read_label_rows(x, first, n)
}

#' @export
print.fastEmbedR_massive_louvain_level <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR ", x$method, "\n", sep = "")
    cat("  vertices: ", format(x$n_vertices, scientific = FALSE),
        "; communities: ", x$n_communities, "\n", sep = "")
    cat("  modularity: ", signif(x$modularity_final, 6),
        "; passes: ", x$passes, "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_louvain_level <- function(x, n = 6L, ...) {
    massive_read_louvain_rows(x, n = min(n, x$n_vertices))
}

massive_louvain_contract_resources <- function(graph, output,
        memory_limit, stage = "new") {
    limit <- massive_memory_bytes(memory_limit)
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    paths <- paste0(path, c("", ".part",
        ".level0.clusters.u32", ".coarse.edges.bin",
        ".coarse.edges.bin.part", ".coarse.edges.bin.work",
        ".checkpoint.rds"))
    existing <- file.exists(paths)
    expected <- switch(stage, new = rep(FALSE, 7L),
        level0 = c(FALSE, FALSE, TRUE, FALSE,
            FALSE, FALSE, TRUE),
        contracted = c(FALSE, FALSE, TRUE, TRUE,
            FALSE, FALSE, TRUE),
        stop("Invalid Louvain checkpoint stage.", call. = FALSE))
    if (!identical(existing, expected)) stop(
        "Louvain output or intermediate files conflict with resume.",
        call. = FALSE)
    pairs <- graph$n_edges / 2
    disk <- 64 * pairs + 4 * graph$n_vertices
    if (stage != "contracted" &&
        disk > 0.8 * massive_disk_available_cpp(path)) stop(
        "Louvain contraction exceeds the free-disk budget.",
        call. = FALSE)
    list(path = path, memory_limit_bytes = limit,
        estimated_contraction_disk_bytes = disk)
}

massive_louvain_signature <- function(graph, path, controls,
        chunk_rows, memory_limit) {
    path <- file.path(normalizePath(dirname(path), mustWork = TRUE),
        basename(path))
    dll <- getLoadedDLLs()[["fastEmbedR"]]
    if (is.null(dll)) stop(
        "Cannot identify the native Louvain implementation.",
        call. = FALSE)
    native <- unname(tools::md5sum(dll[["path"]]))
    if (is.na(native)) stop(
        "Cannot hash the native Louvain implementation.",
        call. = FALSE)
    massive_checkpoint_signature(list(version = 1L, path = path,
        graph = graph$file_identity, vertices = graph$n_vertices,
        edges = graph$n_edges, controls = controls,
        chunk_rows = chunk_rows,
        memory_limit = massive_memory_bytes(memory_limit),
        native = native))
}

massive_louvain_save_stage <- function(path, signature, stage,
        level, contracted = NULL) {
    state <- list(signature = signature, stage = stage,
        level = level, contracted = contracted,
        contracted_identity = if (!is.null(contracted)) {
            massive_checkpoint_file_identity(contracted$path)
        } else NULL)
    massive_checkpoint_write(state, paste0(path, ".checkpoint.rds"))
    state
}

massive_louvain_identity_matches <- function(path, saved, bytes) {
    current <- tryCatch(massive_checkpoint_file_identity(path),
        error = function(e) NULL)
    identical(current, saved) && length(current$bytes) == 1L &&
        identical(current$bytes, as.numeric(bytes))
}

massive_louvain_resume_stage <- function(graph, path, signature) {
    state <- tryCatch(readRDS(paste0(path, ".checkpoint.rds")),
        error = function(e) NULL)
    if (!is.list(state) ||
        !identical(state$signature, signature) ||
        !is.character(state$stage) ||
        length(state$stage) != 1L ||
        is.na(state$stage) ||
        !state$stage %in% c("level0", "contracted")) stop(
        "Louvain checkpoint does not match this graph or controls.",
        call. = FALSE)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    level <- state$level
    current <- tryCatch(massive_checkpoint_file_identity(paths),
        error = function(e) NULL)
    if (!identical(current,
            graph$file_identity) ||
        !inherits(level, "fastEmbedR_massive_louvain_level") ||
        !identical(level$membership_path,
            paste0(path, ".level0.clusters.u32")) ||
        !identical(level$n_vertices, graph$n_vertices) ||
        !massive_louvain_identity_matches(level$membership_path,
            level$membership_identity, 4 * graph$n_vertices)) stop(
        "Louvain graph or completed first level changed.",
        call. = FALSE)
    if (identical(state$stage, "contracted")) {
        coarse <- state$contracted
        if (!is.list(coarse) ||
            !identical(coarse$path,
                paste0(path, ".coarse.edges.bin")) ||
            !identical(coarse$n_vertices, level$n_communities) ||
            !is.numeric(coarse$n_edges) ||
            length(coarse$n_edges) != 1L ||
            !is.finite(coarse$n_edges) ||
            !massive_louvain_identity_matches(coarse$path,
                state$contracted_identity,
                16 * coarse$n_edges)) stop(
            "Completed Louvain contraction changed.",
            call. = FALSE)
    }
    state
}

massive_louvain_coarse_graph <- function(contracted, memory_limit) {
    estimate <- 64 * 1024^2 + 128 * contracted$n_vertices +
        192 * contracted$n_edges
    if (estimate > memory_limit) stop(
        "Contracted Louvain graph exceeds `memory_limit`; ",
        "the first-level result remains available. No fallback.",
        call. = FALSE)
    graph <- massive_read_contracted_cpp(contracted$path,
        contracted$n_vertices, contracted$n_edges)
    list(graph = graph, estimated_resident_bytes = estimate)
}

massive_louvain_contract <- function(graph, level, path, resources) {
    graph_paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    if (!identical(massive_checkpoint_file_identity(graph_paths),
        graph$file_identity)) stop(
        "Massive graph files changed before Louvain contraction.",
        call. = FALSE)
    if (64 * graph$n_edges > 0.8 *
        massive_disk_available_cpp(path)) stop(
        "Louvain contraction exceeds the free-disk budget.",
        call. = FALSE)
    contracted <- massive_contract_louvain_cpp(graph$offsets_path,
        graph$indices_path, graph$weights_path,
        level$membership_path, graph$n_vertices,
        level$n_communities, paste0(path, ".coarse.edges.bin"),
        resources$memory_limit_bytes, level$resources$chunk_rows,
        level$resources$edge_budget,
        identical(graph$weight_dtype, "float64"))
    if (!identical(massive_checkpoint_file_identity(graph_paths),
        graph$file_identity) || !identical(
        massive_checkpoint_file_identity(level$membership_path),
        level$membership_identity)) stop(
        "Louvain contraction inputs changed.", call. = FALSE)
    if (!isTRUE(all.equal(contracted$total_edge_weight,
        level$total_edge_weight, tolerance = 1e-6))) stop(
        "Contracted Louvain graph lost edge weight.",
        call. = FALSE)
    contracted
}

massive_louvain_coarse_csr <- function(contracted, prefix,
        memory_limit) {
    before <- massive_checkpoint_file_identity(contracted$path)
    csr <- massive_coarse_csr_cpp(contracted$path,
        contracted$n_vertices, contracted$n_edges, prefix,
        memory_limit)
    if (!identical(before, massive_checkpoint_file_identity(
        contracted$path)) || !isTRUE(all.equal(
        csr$total_edge_weight, contracted$total_edge_weight,
        tolerance = 1e-9))) stop(
        "Contracted Louvain CSR changed or lost edge weight.",
        call. = FALSE)
    csr$file_identity <- massive_checkpoint_file_identity(c(
        csr$offsets_path, csr$indices_path, csr$weights_path))
    csr$weight_dtype <- "float64"
    csr$storage <- "csr"
    csr$weight_method <- "umap_fuzzy_union"
    csr$symmetrized <- TRUE
    class(csr) <- "fastEmbedR_massive_graph"
    csr
}

massive_louvain_resident_level <- function(contracted, part,
        output, index, resources, resolution, max_passes, seed,
        original_rows) {
    coarse <- massive_louvain_coarse_graph(contracted,
        resources$memory_limit_bytes)
    groups <- graph_cluster(coarse$graph, method = "louvain",
        backend = "cpu", resolution = resolution,
        n_iterations = max_passes,
        seed = as.integer((as.double(seed) + index - 1) %%
            (.Machine$integer.max + 1)))
    mapping <- paste0(output, ".level", index, ".clusters.u32")
    if (file.exists(mapping)) stop(
        "Resident Louvain mapping already exists.", call. = FALSE)
    writeBin(as.integer(groups$membership), mapping, size = 4L,
        endian = "little")
    massive_remap_louvain_cpp(part, mapping, original_rows,
        contracted$n_vertices)
    list(n_communities = groups$n_communities,
        estimated_resident_bytes = coarse$estimated_resident_bytes)
}

massive_louvain_disk_level <- function(contracted, part,
        output, index, resources, max_passes, resolution, seed,
        chunk_rows, original_rows) {
    prefix <- paste0(output, ".level", index)
    csr <- massive_louvain_coarse_csr(contracted,
        paste0(prefix, ".csr"), resources$memory_limit_bytes)
    level <- massive_louvain_run_level(csr,
        paste0(prefix, ".clusters.u32"), max_passes,
        resolution,
        as.integer((as.double(seed) + index - 1) %%
            (.Machine$integer.max + 1)), chunk_rows,
        resources$memory_limit_bytes, TRUE, index)
    if (!isTRUE(all.equal(level$total_edge_weight,
        contracted$total_edge_weight, tolerance = 1e-9))) stop(
        "Contracted Louvain level lost edge weight.",
        call. = FALSE)
    massive_remap_louvain_cpp(part, level$membership_path,
        original_rows, contracted$n_vertices)
    next_graph <- if (level$n_communities < contracted$n_vertices) {
        massive_louvain_contract(csr, level, prefix, resources)
    } else NULL
    list(level = level, next_graph = next_graph)
}

massive_louvain_hierarchy <- function(graph, level, contracted,
        output, resources, max_passes, resolution, seed, chunk_rows) {
    part <- paste0(output, ".part")
    if (!file.copy(level$membership_path, part)) stop(
        "Cannot copy first-level Louvain labels.", call. = FALSE)
    levels <- list()
    for (index in seq_len(64L)) {
        estimate <- 64 * 1024^2 + 128 * contracted$n_vertices +
            192 * contracted$n_edges
        if (estimate <= 0.5 * resources$memory_limit_bytes) {
            resident <- massive_louvain_resident_level(contracted,
                part, output, index, resources, resolution,
                max_passes, seed, graph$n_vertices)
            return(list(part = part,
                n_communities = resident$n_communities,
                resident_bytes = resident$estimated_resident_bytes,
                levels = levels))
        }
        step <- massive_louvain_disk_level(contracted, part,
            output, index, resources, max_passes, resolution,
            seed, chunk_rows, graph$n_vertices)
        levels[[index]] <- list(path = step$level$membership_path,
            n_vertices = contracted$n_vertices,
            n_communities = step$level$n_communities,
            modularity = step$level$modularity_final)
        if (is.null(step$next_graph)) return(list(part = part,
            n_communities = step$level$n_communities,
            resident_bytes = 0, levels = levels))
        contracted <- step$next_graph
    }
    stop("Louvain exceeded 64 contracted levels; partial files retained.",
        call. = FALSE)
}

massive_louvain_prepare <- function(graph, output, controls,
        chunk_rows, memory_limit, checkpoint, resume, initial = NULL) {
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    signature <- if (checkpoint) massive_louvain_signature(
        graph, path, controls, chunk_rows, memory_limit) else NULL
    saved <- if (resume) massive_louvain_resume_stage(
        graph, path, signature) else NULL
    stage <- if (is.null(saved)) "new" else saved$stage
    resources <- massive_louvain_contract_resources(
        graph, path, memory_limit, stage)
    level <- if (is.null(saved)) massive_louvain_run_level(graph,
        paste0(path, ".level0.clusters.u32"), controls$passes,
        controls$resolution, controls$seed, chunk_rows,
        memory_limit, FALSE, 0L, initial) else saved$level
    if (checkpoint && is.null(saved)) saved <-
        massive_louvain_save_stage(path, signature, "level0", level)
    contracted <- if (identical(stage, "contracted")) {
        saved$contracted
    } else {
        massive_louvain_contract(graph, level, path, resources)
    }
    if (checkpoint && !identical(stage, "contracted"))
        massive_louvain_save_stage(path, signature, "contracted",
            level, contracted)
    list(path = path, resources = resources,
        level = level, contracted = contracted)
}

#' EXPERIMENTAL multilevel Louvain on a file-backed fuzzy graph
#'
#' Runs a file-backed local-moving level, externally contracts its edges,
#' and keeps higher contracted levels file-backed until a level passes a
#' conservative resident-memory check. At that point it reuses
#' [graph_cluster()]. Original graph and final labels remain on disk.
#' With `checkpoint = TRUE`, completed first-level and contraction files
#' can be reused by `resume = TRUE` after validating input, control, and
#' native-library identities. An interrupted individual scan or higher
#' level is not yet resumable; its partial files remain for inspection.
#' Mapped label pages may contribute to RSS beyond scan buffers.
#'
#' @param graph A symmetric result of [massive_umap_fuzzy_graph()].
#' @param output New path for one-based `uint32` final cluster labels.
#' @param max_passes Maximum local-moving sweeps per Louvain level.
#' @param resolution Positive modularity resolution.
#' @param seed Nonnegative random seed.
#' @param chunk_rows Maximum consecutive graph rows per scan block.
#' @param memory_limit RAM budget for scan buffers and resident levels.
#' @param backend Only `"cpu"` is implemented; other requests fail.
#' @param checkpoint Save completed first-level and contraction stages.
#' @param resume Reuse a matching checkpoint; requires `checkpoint = TRUE`.
#' @param initial Optional `massive_cluster_landmarks()` result whose saved
#'   labels initialize local moves. Its row order must match `graph`.
#' @return Lightweight file-backed Louvain result with modularity.
#' @export
massive_louvain <- function(graph, output, max_passes = 10L,
        resolution = 1, seed = 1L, chunk_rows = 8192L,
        memory_limit = "1GB", backend = "cpu",
        checkpoint = FALSE, resume = FALSE, initial = NULL) {
    massive_louvain_graph_check(graph, backend)
    massive_checkpoint_validate_controls(checkpoint, resume)
    controls <- massive_louvain_controls(max_passes, resolution, seed)
    initial <- massive_cluster_initial(graph, initial)
    if (!is.null(initial))
        controls$initial <- initial$parent_mapping_identity
    prepared <- massive_louvain_prepare(graph, output, controls,
        chunk_rows, memory_limit, checkpoint, resume, initial)
    path <- prepared$path
    level <- prepared$level
    contracted <- prepared$contracted
    resources <- prepared$resources
    hierarchy <- massive_louvain_hierarchy(graph, level,
        contracted, path, resources, controls$passes,
        controls$resolution, controls$seed, chunk_rows)
    score <- massive_graph_modularity(graph, hierarchy$part,
        hierarchy$n_communities, resolution, chunk_rows, memory_limit)
    if (score$modularity < level$modularity_final - 1e-7) stop(
        "Contracted Louvain reduced original-graph modularity.",
        call. = FALSE)
    if (!file.rename(hierarchy$part, path)) stop(
        "Cannot finalize Louvain label file.", call. = FALSE)
    if (checkpoint) unlink(paste0(path, ".checkpoint.rds"))
    first_level_modularity <- level$modularity_final
    level$membership_path <- path
    level$membership_identity <- massive_checkpoint_file_identity(path)
    level$n_communities <- hierarchy$n_communities
    level$modularity_final <- score$modularity
    level$method <- "louvain_multilevel"
    level$first_level <- list(path = paste0(path,
        ".level0.clusters.u32"),
        n_communities = contracted$n_vertices,
        modularity = first_level_modularity)
    level$contracted_edges <- contracted$n_edges
    level$estimated_coarse_bytes <- hierarchy$resident_bytes
    level$file_backed_levels <- hierarchy$levels
    level$initial_partition_path <- if (is.null(initial)) NULL else
        initial$parent_mapping_path
    level
}
