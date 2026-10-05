massive_leiden_refinement_output <- function(output, bytes) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) stop(
        "`output` must be one new label-file path.", call. = FALSE)
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    pending <- paste0(path, c("", ".part", ".parent.u32",
        ".parent.u32.part", ".sizes.part", ".volumes.part",
        ".cuts.part", ".parent_volumes.part"))
    if (any(file.exists(pending))) stop(
        "Leiden output or partial state already exists.",
        call. = FALSE)
    if (bytes > 0.8 * massive_disk_available_cpp(path)) stop(
        "Leiden mapped state exceeds the free-disk budget.",
        call. = FALSE)
    path
}

#' EXPERIMENTAL file-backed Leiden refinement stage
#'
#' Refines one Louvain local-move partition using Leiden's constrained
#' singleton-merge rule. Graph edges are streamed in bounded blocks and
#' labels and refinement state are memory-mapped. This is one stage; use
#' [massive_leiden()] for the multilevel fit. Mapped pages can contribute
#' to process RSS.
#'
#' @param graph A result of [massive_umap_fuzzy_graph()].
#' @param parent A first-level [massive_louvain_level()] result for `graph`.
#' @param output New path for one-based `uint32` refined labels.
#' @param resolution Positive modularity resolution.
#' @param seed Nonnegative refinement seed.
#' @param chunk_rows Maximum consecutive CSR rows per scan block.
#' @param memory_limit RAM budget for graph scan buffers.
#' @param backend Only `"cpu"` is implemented; no fallback is used.
#' @return A file-backed refinement and refined-to-parent label map.
#' @export
massive_leiden_refine_level <- function(graph, parent, output,
        resolution = 1, seed = 1L, chunk_rows = 8192L,
        memory_limit = "1GB", backend = "cpu") {
    massive_louvain_graph_check(graph, backend)
    if (!inherits(parent, "fastEmbedR_massive_louvain_level") ||
        parent$n_vertices != graph$n_vertices ||
        !identical(parent$resolution, resolution)) stop(
        "`parent` must be a matching Louvain level.", call. = FALSE)
    run_seed <- integer_scalar(seed)
    if (is.na(run_seed) || run_seed < 0L) stop(
        "`seed` must be a nonnegative integer.", call. = FALSE)
    resources <- massive_louvain_resources(graph, chunk_rows,
        memory_limit)
    mapped <- 32 * graph$n_vertices + 8 * parent$n_communities
    path <- massive_leiden_refinement_output(output, mapped)
    files <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path, parent$membership_path)
    before <- massive_checkpoint_file_identity(files)
    if (!identical(massive_checkpoint_file_identity(files[1:3]),
            parent$graph_identity) ||
        !identical(massive_checkpoint_file_identity(files[1:3]),
            graph$file_identity) ||
        !identical(massive_checkpoint_file_identity(files[[4L]]),
            parent$membership_identity) ||
        before$bytes[[4L]] != 4 * graph$n_vertices) stop(
        "Leiden graph or parent labels changed since fitting.",
        call. = FALSE)
    result <- massive_leiden_refine_cpp(files[[1L]], files[[2L]],
        files[[3L]], files[[4L]], graph$n_vertices,
        parent$n_communities, path, resolution,
        resources$chunk_rows, resources$edge_budget, run_seed,
        identical(graph$weight_dtype, "float64"))
    if (!identical(before, massive_checkpoint_file_identity(files))) {
        stop("Leiden inputs changed during refinement.",
            call. = FALSE)
    }
    result$membership_identity <- massive_checkpoint_file_identity(path)
    result$parent_mapping_identity <- massive_checkpoint_file_identity(
        result$parent_mapping_path)
    result$graph_identity <- graph$file_identity
    result$n_communities <- result$n_refined
    result$total_edge_weight <- parent$total_edge_weight
    result$resources <- resources
    result$resources$mapped_state_bytes <- mapped
    result$method <- "leiden_refinement_level"
    result$experimental <- TRUE
    class(result) <- "fastEmbedR_massive_leiden_refinement"
    result
}

massive_leiden_output <- function(graph, output, memory_limit,
        resume) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) stop(
        "`output` must be one new label-file path.", call. = FALSE)
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    pending <- paste0(path, c("", ".part",
        ".level0.parent.u32", ".level0.refined.u32",
        ".checkpoint.rds"))
    if (file.exists(path) || (!resume &&
            any(file.exists(pending)))) stop(
        "Leiden output or intermediate files already exist.",
        call. = FALSE)
    if (resume && !all(file.exists(pending[c(2L, 5L)]))) stop(
        "Leiden checkpoint or mapped labels are missing.",
        call. = FALSE)
    required <- 64 * graph$n_edges + 40 * graph$n_vertices
    if (!resume && required > 0.8 *
            massive_disk_available_cpp(path)) stop(
        "Leiden graph contraction exceeds the free-disk budget.",
        call. = FALSE)
    list(path = path,
        memory_limit_bytes = massive_memory_bytes(memory_limit))
}

massive_leiden_signature <- function(graph, path, controls,
        chunk_rows, memory_limit) {
    massive_checkpoint_signature(list(method = "leiden", version = 1L,
        graph = massive_louvain_signature(graph, path, controls,
            chunk_rows, memory_limit)))
}

massive_leiden_save <- function(path, signature, stage, index,
        current, previous, part, levels, first_modularity,
        n_communities = NULL) {
    state <- list(signature = signature, stage = stage,
        index = index, current = current, previous = previous,
        part_identity = massive_checkpoint_file_identity(part),
        levels = levels, first_modularity = first_modularity,
        n_communities = n_communities)
    massive_checkpoint_write(state,
        paste0(path, ".checkpoint.rds"))
    state
}

massive_leiden_resume_state <- function(graph, path, signature) {
    state <- tryCatch(readRDS(paste0(path, ".checkpoint.rds")),
        error = function(e) NULL)
    if (!is.list(state) || !identical(state$signature, signature) ||
        !is.character(state$stage) ||
        length(state$stage) != 1L ||
        !state$stage %in% c("next", "ready") ||
        !is.numeric(state$index) || length(state$index) != 1L ||
        is.na(state$index) || state$index < 1L ||
        state$index > 64L ||
        (state$stage == "next" && state$index == 64L) ||
        length(state$levels) != state$index ||
        !is.numeric(state$first_modularity) ||
        length(state$first_modularity) != 1L ||
        !is.finite(state$first_modularity) ||
        !is.list(state$current)) stop(
        "Leiden checkpoint does not match completed levels.",
        call. = FALSE)
    massive_leiden_resume_inputs(graph, path, state)
    state
}

massive_leiden_resume_inputs <- function(graph, path, state) {
    current <- state$current
    previous <- state$previous
    paths <- c(current$offsets_path, current$indices_path,
        current$weights_path)
    if (!identical(massive_checkpoint_file_identity(c(
            graph$offsets_path, graph$indices_path,
            graph$weights_path)), graph$file_identity) ||
        !identical(massive_checkpoint_file_identity(paths),
            current$file_identity) ||
        !massive_louvain_identity_matches(paste0(path, ".part"),
            state$part_identity, 4 * graph$n_vertices)) stop(
        "Leiden graph, mapping, or completed labels changed.",
        call. = FALSE)
    if (state$stage == "next" &&
        (!is.list(previous) ||
        !massive_louvain_identity_matches(
            previous$parent_mapping_path,
            previous$parent_mapping_identity,
            4 * previous$n_refined) ||
        !isTRUE(all.equal(previous$n_refined,
            current$n_vertices)))) stop(
        "Leiden refined-to-parent map changed.", call. = FALSE)
    if (state$stage == "ready" &&
        (!is.numeric(state$n_communities) ||
        length(state$n_communities) != 1L ||
        !is.finite(state$n_communities) ||
        state$n_communities < 1L ||
        state$n_communities > graph$n_vertices)) stop(
        "Leiden final checkpoint has invalid community count.",
        call. = FALSE)
    if (state$stage == "next" && length(Sys.glob(paste0(
            path, ".level", state$index, ".*")))) stop(
        "An incomplete Leiden level conflicts with resume.",
        call. = FALSE)
    invisible(NULL)
}

massive_leiden_finish_level <- function(step, output,
        index, signature, current, previous, part, levels) {
    done <- is.null(step$refined) ||
        step$refined$n_refined >= current$n_vertices ||
        step$refined$n_refined == 1L
    if (!done) return(NULL)
    first <- levels[[1L]]$parent_modularity
    if (!is.null(signature)) massive_leiden_save(output,
        signature, "ready", index + 1L, current,
        step$refined %||% previous, part, levels, first,
        step$n_communities)
    list(part = part, n_communities = step$n_communities,
        levels = levels, first_modularity = first)
}

massive_leiden_run_level <- function(graph, initial, prefix,
        index, passes, resolution, seed, chunk_rows, memory_limit) {
    run_seed <- as.integer((as.double(seed) + index) %%
        (.Machine$integer.max + 1))
    parent <- massive_louvain_run_level(graph,
        paste0(prefix, ".parent.u32"), passes, resolution,
        run_seed, chunk_rows, memory_limit, index > 0L,
        index, initial)
    if (parent$n_communities == graph$n_vertices) return(list(
        parent = parent, refined = NULL,
        mapping = parent$membership_path,
        n_communities = parent$n_communities))
    refined <- massive_leiden_refine_level(graph, parent,
        paste0(prefix, ".refined.u32"), resolution,
        run_seed, chunk_rows, memory_limit)
    mapping <- if (refined$n_refined == graph$n_vertices) {
        parent$membership_path
    } else refined$membership_path
    list(parent = parent, refined = refined, mapping = mapping,
        n_communities = if (identical(mapping,
            parent$membership_path)) parent$n_communities else
            refined$n_refined)
}

massive_leiden_hierarchy <- function(graph, output, controls,
        chunk_rows, memory_limit, resources, signature = NULL,
        saved = NULL, initial = NULL) {
    current <- if (is.null(saved)) graph else saved$current
    previous <- if (is.null(saved)) initial else saved$previous
    part <- paste0(output, ".part")
    levels <- if (is.null(saved)) list() else saved$levels
    if (!is.null(saved) && saved$stage == "ready") return(list(
        part = part, n_communities = saved$n_communities,
        levels = levels,
        first_modularity = saved$first_modularity))
    start <- if (is.null(saved)) 0L else saved$index
    for (index in seq.int(start, 63L)) {
        prefix <- paste0(output, ".level", index)
        step <- massive_leiden_run_level(current, previous, prefix,
            index, controls$passes, controls$resolution,
            controls$seed, chunk_rows, memory_limit)
        levels[[index + 1L]] <- list(
            vertices = current$n_vertices,
            parent = step$parent$n_communities,
            parent_modularity = step$parent$modularity_final,
            refined = if (is.null(step$refined)) NA_integer_ else
                step$refined$n_refined)
        if (index == 0L) {
            if (!file.copy(step$mapping, part)) stop(
                "Cannot initialize Leiden labels.", call. = FALSE)
        } else {
            massive_remap_louvain_cpp(part, step$mapping,
                graph$n_vertices, current$n_vertices)
        }
        done <- massive_leiden_finish_level(step, output,
            index, signature, current, previous, part, levels)
        if (!is.null(done)) return(done)
        contracted <- massive_louvain_contract(current,
            step$refined, prefix, resources)
        current <- massive_louvain_coarse_csr(contracted,
            paste0(prefix, ".csr"),
            resources$memory_limit_bytes)
        previous <- step$refined
        if (!is.null(signature)) massive_leiden_save(output,
            signature, "next", index + 1L, current,
            previous, part, levels,
            levels[[1L]]$parent_modularity)
    }
    stop("Leiden exceeded 64 levels; partial files retained.",
        call. = FALSE)
}

#' EXPERIMENTAL multilevel Leiden on a file-backed fuzzy graph
#'
#' Applies disk-backed modularity local moves and the existing Leiden
#' refinement rule, then externally contracts each refined graph. Labels,
#' graph edges, and intermediate mappings remain file-backed. Currently
#' CPU-only and single-worker. With checkpoints, completed graph levels
#' can resume after identity validation; interrupted levels retain partial
#' files and require separate recovery. The memory budget does not cover
#' mapped pages or the R process itself. One hierarchy is run per call;
#' `max_passes` controls local moves within each level.
#'
#' @param graph A symmetric result of [massive_umap_fuzzy_graph()].
#' @param output New path for one-based `uint32` final labels.
#' @param max_passes Maximum local-moving sweeps per level.
#' @param resolution Positive modularity resolution.
#' @param seed Nonnegative random seed.
#' @param chunk_rows Maximum CSR rows scanned per block.
#' @param memory_limit RAM budget for graph scan buffers.
#' @param backend Only `"cpu"` is implemented; no fallback is used.
#' @param checkpoint Save completed graph levels for later resumption.
#' @param resume Resume matching completed levels; requires `checkpoint`.
#' @param initial Optional `massive_cluster_landmarks()` result whose saved
#'   labels initialize local moves. Its row order must match `graph`.
#' @return Lightweight file-backed Leiden result with modularity.
#' @export
massive_leiden <- function(graph, output, max_passes = 10L,
        resolution = 1, seed = 1L, chunk_rows = 8192L,
        memory_limit = "1GB", backend = "cpu",
        checkpoint = FALSE, resume = FALSE, initial = NULL) {
    massive_louvain_graph_check(graph, backend)
    massive_checkpoint_validate_controls(checkpoint, resume)
    controls <- massive_louvain_controls(max_passes, resolution, seed)
    initial <- massive_cluster_initial(graph, initial)
    if (!is.null(initial))
        controls$initial <- initial$parent_mapping_identity
    resources <- massive_leiden_output(graph, output, memory_limit,
        resume)
    before <- massive_checkpoint_file_identity(c(
        graph$offsets_path, graph$indices_path, graph$weights_path))
    if (!identical(before, graph$file_identity)) stop(
        "Massive graph changed since construction.", call. = FALSE)
    signature <- if (checkpoint) massive_leiden_signature(graph,
        resources$path, controls, chunk_rows, memory_limit) else NULL
    saved <- if (resume) massive_leiden_resume_state(graph,
        resources$path, signature) else NULL
    fitted <- massive_leiden_hierarchy(graph, resources$path,
        controls, chunk_rows, memory_limit, resources,
        signature, saved, initial)
    score <- massive_graph_modularity(graph, fitted$part,
        fitted$n_communities, resolution, chunk_rows, memory_limit)
    if (score$modularity < fitted$first_modularity - 1e-7) stop(
        "Contracted Leiden reduced original-graph modularity.",
        call. = FALSE)
    if (!identical(before, massive_checkpoint_file_identity(c(
            graph$offsets_path, graph$indices_path,
            graph$weights_path)))) stop(
        "Massive graph changed during Leiden.", call. = FALSE)
    if (!file.rename(fitted$part, resources$path)) stop(
        "Cannot finalize Leiden label file.", call. = FALSE)
    if (checkpoint) unlink(paste0(resources$path,
        ".checkpoint.rds"))
    result <- list(membership_path = resources$path,
        membership_identity = massive_checkpoint_file_identity(
            resources$path), n_vertices = graph$n_vertices,
        n_communities = fitted$n_communities,
        modularity_final = score$modularity,
        levels = fitted$levels, graph_identity = before,
        method = "leiden_multilevel", backend = "cpu",
        initial_partition_path = if (is.null(initial)) NULL else
            initial$parent_mapping_path,
        experimental = TRUE)
    class(result) <- "fastEmbedR_massive_leiden"
    result
}

#' Read bounded labels from experimental file-backed Leiden
#'
#' @param x A result of [massive_leiden()].
#' @param first First one-based vertex row.
#' @param n Number of labels to read; capped at one million.
#' @return One-based integer community labels.
#' @export
massive_read_leiden_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_leiden")) stop(
        "`x` must be a massive_leiden() result.", call. = FALSE)
    massive_read_label_rows(x, first, n)
}

#' @export
print.fastEmbedR_massive_leiden <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR file-backed Leiden\n")
    cat("  vertices: ", format(x$n_vertices, scientific = FALSE),
        "; communities: ", x$n_communities, "\n", sep = "")
    cat("  modularity: ", signif(x$modularity_final, 6),
        "; levels: ", length(x$levels), "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_leiden <- function(x, n = 6L, ...) {
    massive_read_leiden_rows(x, n = min(n, x$n_vertices))
}
