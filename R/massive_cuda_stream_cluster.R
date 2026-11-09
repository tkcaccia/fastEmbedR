massive_cuda_stream_graph_identity <- function(graph) {
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    current <- massive_checkpoint_file_identity(paths)
    if (!identical(current, graph$file_identity)) stop(
        "Streamed CUDA graph changed since construction.",
        call. = FALSE)
    if (!is.null(graph$validated_input_identity))
        massive_checkpoint_validated_graph_identity(graph, paths)
    current
}

massive_cuda_stream_plan <- function(graph, output, devices,
        chunk_rows, memory_limit) {
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !identical(graph$weight_method, "umap_fuzzy_union") ||
        !isTRUE(graph$symmetrized)) stop(
        "Streamed multi-GPU clustering needs a symmetric fuzzy CSR graph.",
        call. = FALSE)
    devices <- massive_validate_devices(devices, "cuda", graph$n_vertices)
    if (length(devices) < 2L) stop(
        "Streamed clustering requires at least two CUDA GPUs.",
        call. = FALSE)
    rows <- integer_scalar(chunk_rows %||% 8192L)
    limit <- massive_memory_bytes(memory_limit)
    free <- massive_auto_cuda_free(devices, graph$n_vertices)
    budget <- min(2e6, floor(0.2 * free / 48))
    if (is.na(rows) || rows < 1L ||
        budget < graph$max_degree || budget < 1) stop(
        "A CSR row exceeds the available bounded GPU batch.",
        call. = FALSE)
    if (32 * graph$n_vertices + 512 * 1024^2 > 0.7 * limit) stop(
        "Streamed clustering host state exceeds `memory_limit`.",
        call. = FALSE)
    path <- massive_louvain_output(output,
        32 * graph$n_vertices + 1024^2)
    artifacts <- paste0(path, c(".part", ".stream.json",
        ".stream.out", ".stream.err"))
    if (any(file.exists(artifacts))) stop(
        "Streamed clustering output files already exist.",
        call. = FALSE)
    list(path = path, devices = devices,
        uuids = massive_cuda_device_uuids(devices),
        chunk_rows = rows, edge_budget = budget,
        host_limit = limit, free_vram = free)
}

massive_cuda_stream_invoke <- function(graph, plan, controls) {
    python <- Sys.getenv("FASTEMBEDR_CUGRAPH_PYTHON", "python3")
    if (!nzchar(Sys.which(python))) stop(
        "CUDA Python is unavailable; no fallback.", call. = FALSE)
    script <- system.file("python", "massive_cuda_stream_cluster.py",
        package = "fastEmbedR", mustWork = TRUE)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    args <- c(script, "--paths", paths, "--vertices",
        format(graph$n_vertices, scientific = FALSE, trim = TRUE),
        "--devices", length(plan$devices), "--chunk-rows",
        plan$chunk_rows, "--edge-budget", plan$edge_budget,
        "--passes", controls$passes, "--resolution",
        controls$resolution, "--seed", controls$seed,
        "--output", plan$path, "--metadata",
        paste0(plan$path, ".stream.json"))
    if (nzchar(controls$initial_path)) args <- c(args,
        "--initial", controls$initial_path,
        "--initial-count", controls$initial_count)
    status <- system2(python, shQuote(args),
        stdout = paste0(plan$path, ".stream.out"),
        stderr = paste0(plan$path, ".stream.err"),
        env = paste0("CUDA_VISIBLE_DEVICES=",
            paste(plan$uuids, collapse = ",")))
    if (!identical(status, 0L)) stop(
        "Streamed CUDA clustering failed; inspect .stream.err.",
        call. = FALSE)
    jsonlite::fromJSON(paste0(plan$path, ".stream.json"))
}

massive_cuda_stream_initial <- function(initial, graph) {
    if (is.null(initial)) return(list(path = "", count = 0L,
        identity = NULL))
    if (!inherits(initial, "fastEmbedR_massive_leiden_refinement") ||
        initial$n_refined != graph$n_vertices ||
        !identical(massive_checkpoint_file_identity(
            initial$parent_mapping_path),
            initial$parent_mapping_identity)) stop(
        "Streamed CUDA initial Leiden mapping changed.",
        call. = FALSE)
    list(path = initial$parent_mapping_path,
        count = initial$n_parent,
        identity = initial$parent_mapping_identity)
}

massive_cuda_stream_audit <- function(graph, plan, metadata,
        resolution, memory_limit, before, initial) {
    part <- paste0(plan$path, ".part")
    if (!identical(metadata$backend, "cuda_streamed_multigpu_local") ||
        metadata$vertices != graph$n_vertices ||
        !identical(as.integer(metadata$gpu_devices_used),
            seq_along(plan$devices) - 1L) ||
        length(unique(metadata$gpu_pci_bus_ids)) !=
            length(plan$devices) ||
        !identical(metadata$initial_partition,
            nzchar(initial$path)) ||
        file.info(part)$size != 4 * graph$n_vertices) stop(
        "Streamed clustering output is incomplete.", call. = FALSE)
    score <- massive_graph_modularity(graph, part,
        metadata$communities, resolution, plan$chunk_rows,
        memory_limit)
    if (!isTRUE(all.equal(score$modularity,
            metadata$modularity, tolerance = 1e-5))) stop(
        "Streamed CUDA modularity failed independent audit.",
        call. = FALSE)
    if (score$modularity < metadata$modularity_initial - 1e-7)
        stop("Streamed CUDA local moves reduced modularity.",
            call. = FALSE)
    if (!identical(before, massive_cuda_stream_graph_identity(graph)))
        stop("Massive graph changed during clustering.", call. = FALSE)
    if (!file.rename(part, plan$path)) stop(
        "Cannot finalize streamed CUDA labels.", call. = FALSE)
    score
}

massive_cuda_stream_level <- function(graph, output, devices,
        chunk_rows, memory_limit, resolution, passes, seed,
        initial = NULL) {
    controls <- massive_louvain_controls(passes, resolution, seed)
    initial <- massive_cuda_stream_initial(initial, graph)
    controls$initial_path <- initial$path
    controls$initial_count <- initial$count
    plan <- massive_cuda_stream_plan(graph, output, devices,
        chunk_rows, memory_limit)
    before <- massive_cuda_stream_graph_identity(graph)
    worker <- system.file("python", "massive_cuda_stream_cluster.py",
        package = "fastEmbedR", mustWork = TRUE)
    hash <- unname(tools::md5sum(worker))
    metadata <- massive_cuda_stream_invoke(graph, plan, controls)
    if (!identical(before, massive_cuda_stream_graph_identity(graph)) ||
        !identical(hash, unname(tools::md5sum(worker))) ||
        (!is.null(initial$identity) && !identical(
            initial$identity,
            massive_checkpoint_file_identity(initial$path)))) stop(
        "Streamed clustering source changed during fitting.",
        call. = FALSE)
    score <- massive_cuda_stream_audit(graph, plan, metadata,
        resolution, memory_limit, before, initial)
    structure(list(membership_path = plan$path,
        membership_identity = massive_checkpoint_file_identity(
            plan$path), n_vertices = graph$n_vertices,
        n_communities = metadata$communities,
        resolution = resolution,
        modularity_initial = metadata$modularity_initial,
        modularity_final = score$modularity,
        total_edge_weight = score$total_edge_weight,
        resources = list(chunk_rows = plan$chunk_rows,
            edge_budget = plan$edge_budget),
        method = "louvain_local_move_level", backend = "cuda",
        gpu_devices = plan$devices, gpu_identity = plan$uuids,
        graph_identity = graph$file_identity,
        validated_input_identity = before, graph_audit = score,
        stream = metadata, python_worker_md5 = hash,
        mode = "out_of_core_graph", experimental = TRUE),
        class = "fastEmbedR_massive_louvain_level")
}

massive_cuda_stream_hierarchy <- function(graph, first, path,
        devices, chunk_rows, memory_limit, resolution,
        passes, seed, resources) {
    part <- paste0(path, ".part")
    if (!file.copy(first$membership_path, part)) stop(
        "Cannot stage streamed Louvain labels.", call. = FALSE)
    current <- graph
    level <- first
    levels <- list()
    for (index in 0:63) {
        if (level$n_communities >= current$n_vertices ||
            level$n_communities < 2L) break
        prefix <- if (index == 0L) path else
            paste0(path, ".level", index)
        contracted <- massive_louvain_contract(current, level,
            prefix, resources)
        csr <- massive_louvain_coarse_csr(contracted,
            paste0(path, ".level", index + 1L, ".csr"),
            resources$memory_limit_bytes)
        next_level <- massive_cuda_stream_level(csr,
            paste0(path, ".level", index + 1L,
                ".clusters.u32"), devices, chunk_rows,
            memory_limit, resolution, passes,
            as.integer((as.double(seed) + index + 1) %%
                (.Machine$integer.max + 1)))
        massive_remap_louvain_cpp(part,
            next_level$membership_path, graph$n_vertices,
            csr$n_vertices)
        levels[[index + 1L]] <- next_level
        current <- csr
        level <- next_level
    }
    if (level$n_communities < current$n_vertices &&
        level$n_communities >= 2L) stop(
        "Streamed Louvain exceeded 64 levels; partial files remain.",
        call. = FALSE)
    list(part = part, level = level, levels = levels)
}

massive_cuda_stream_cluster <- function(graph, output, devices,
        chunk_rows, memory_limit, resolution, passes, seed) {
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    resources <- massive_louvain_contract_resources(graph,
        path, memory_limit)
    before <- massive_cuda_stream_graph_identity(graph)
    first <- massive_cuda_stream_level(graph,
        paste0(path, ".level0.clusters.u32"), devices,
        chunk_rows, memory_limit, resolution, passes, seed)
    hierarchy <- massive_cuda_stream_hierarchy(graph, first,
        path, devices, chunk_rows, memory_limit, resolution,
        passes, seed, resources)
    score <- massive_graph_modularity(graph, hierarchy$part,
        hierarchy$level$n_communities, resolution,
        chunk_rows %||% 8192L, memory_limit)
    if (score$modularity < first$modularity_final - 1e-7 ||
        !identical(before, massive_cuda_stream_graph_identity(graph))) stop(
        "Streamed multilevel Louvain failed its final audit.",
        call. = FALSE)
    if (!file.rename(hierarchy$part, path)) stop(
        "Cannot finalize streamed multilevel Louvain labels.",
        call. = FALSE)
    first$membership_path <- path
    first$membership_identity <- massive_checkpoint_file_identity(path)
    first$n_communities <- hierarchy$level$n_communities
    first$modularity_final <- score$modularity
    first$method <- "louvain_multilevel_cuda_stream"
    first$levels <- hierarchy$levels
    first$contraction_backend <- "cpu"
    first$objective_audit_backend <- "cpu"
    first$graph_audit <- score
    first
}

massive_cuda_stream_leiden_step <- function(graph, prefix,
        previous, devices, chunk_rows, memory_limit,
        resolution, passes, seed) {
    parent <- massive_cuda_stream_level(graph,
        paste0(prefix, ".parent.u32"), devices, chunk_rows,
        memory_limit, resolution, passes, seed, previous)
    if (parent$n_communities == graph$n_vertices) return(list(
        parent = parent, refined = NULL,
        mapping = parent$membership_path,
        n_communities = parent$n_communities))
    refined <- massive_leiden_refine_level(graph, parent,
        paste0(prefix, ".refined.u32"), resolution, seed,
        chunk_rows, memory_limit, backend = "cpu")
    mapping <- if (refined$n_refined == graph$n_vertices)
        parent$membership_path else refined$membership_path
    list(parent = parent, refined = refined, mapping = mapping,
        n_communities = if (identical(mapping,
            parent$membership_path)) parent$n_communities else
            refined$n_refined)
}

massive_cuda_stream_leiden_hierarchy <- function(graph, path,
        devices, chunk_rows, memory_limit, resolution,
        passes, seed, resources) {
    current <- graph
    previous <- NULL
    part <- paste0(path, ".part")
    levels <- list()
    first_modularity <- NULL
    for (index in 0:63) {
        prefix <- paste0(path, ".level", index)
        run_seed <- as.integer((as.double(seed) + index) %%
            (.Machine$integer.max + 1))
        step <- massive_cuda_stream_leiden_step(current,
            prefix, previous, devices, chunk_rows,
            memory_limit, resolution, passes, run_seed)
        if (is.null(first_modularity)) first_modularity <-
            step$parent$modularity_final
        if (index == 0L) {
            if (!file.copy(step$mapping, part)) stop(
                "Cannot stage streamed Leiden labels.",
                call. = FALSE)
        } else massive_remap_louvain_cpp(part, step$mapping,
            graph$n_vertices, current$n_vertices)
        levels[[index + 1L]] <- list(
            parent = step$parent, refined = step$refined,
            vertices = current$n_vertices)
        done <- is.null(step$refined) ||
            step$refined$n_refined >= current$n_vertices ||
            step$refined$n_refined == 1L
        if (done) return(list(part = part,
            n_communities = step$n_communities,
            levels = levels, first_modularity = first_modularity))
        contracted <- massive_louvain_contract(current,
            step$refined, prefix, resources)
        current <- massive_louvain_coarse_csr(contracted,
            paste0(prefix, ".csr"),
            resources$memory_limit_bytes)
        previous <- step$refined
    }
    stop("Streamed Leiden exceeded 64 levels; partial files remain.",
        call. = FALSE)
}

massive_cuda_stream_leiden <- function(graph, output, devices,
        chunk_rows, memory_limit, resolution, passes, seed) {
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    resources <- massive_leiden_output(graph, path,
        memory_limit, FALSE)
    before <- massive_cuda_stream_graph_identity(graph)
    fitted <- massive_cuda_stream_leiden_hierarchy(graph, path,
        devices, chunk_rows, memory_limit, resolution,
        passes, seed, resources)
    score <- massive_graph_modularity(graph, fitted$part,
        fitted$n_communities, resolution,
        chunk_rows %||% 8192L, memory_limit,
        connectivity = TRUE)
    if (score$modularity < fitted$first_modularity - 1e-7 ||
        score$disconnected_communities != 0L ||
        !identical(before, massive_cuda_stream_graph_identity(graph))) stop(
        "Streamed Leiden failed modularity or connectivity audit.",
        call. = FALSE)
    if (!file.rename(fitted$part, path)) stop(
        "Cannot finalize streamed Leiden labels.",
        call. = FALSE)
    structure(list(membership_path = path,
        membership_identity = massive_checkpoint_file_identity(path),
        n_vertices = graph$n_vertices,
        n_communities = fitted$n_communities,
        modularity_final = score$modularity,
        method = "leiden_multilevel_cuda_stream",
        backend = "cuda", gpu_devices = devices,
        gpu_identity = massive_cuda_device_uuids(devices),
        refinement_backend = "cpu",
        contraction_backend = "cpu", graph_audit = score,
        graph_identity = graph$file_identity,
        levels = fitted$levels, mode = "out_of_core_graph",
        experimental = TRUE),
        class = "fastEmbedR_massive_leiden")
}
