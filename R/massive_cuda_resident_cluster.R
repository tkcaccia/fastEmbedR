massive_cuda_resident_plan <- function(graph, output, devices,
        chunk_rows, memory_limit) {
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !identical(graph$weight_method, "umap_fuzzy_union") ||
        !isTRUE(graph$symmetrized)) stop(
        "Resident CUDA clustering needs symmetric fuzzy CSR.",
        call. = FALSE)
    devices <- massive_validate_devices(devices, "cuda",
        graph$n_vertices)
    if (length(devices) < 2L) stop(
        "Resident CUDA clustering needs at least two GPUs.",
        call. = FALSE)
    rows <- integer_scalar(chunk_rows %||% 8192L)
    limit <- massive_memory_bytes(memory_limit)
    bytes <- 128 * graph$n_vertices + 64 * graph$n_edges +
        512 * 1024^2
    free <- massive_auto_cuda_free(devices, graph$n_vertices)
    vram <- 28 * graph$n_vertices +
        8 * ceiling(graph$n_edges / length(devices))
    if (is.na(rows) || rows < 1L ||
        bytes > 0.7 * limit || vram > 0.65 * free) stop(
        "Resident CUDA graph exceeds admitted RAM or VRAM.",
        call. = FALSE)
    path <- massive_louvain_output(output,
        4 * graph$n_vertices + 1024^2)
    if (any(file.exists(paste0(path, c("", ".part"))))) stop(
        "Resident CUDA output already exists.", call. = FALSE)
    list(path = path, devices = devices,
        uuids = massive_cuda_device_uuids(devices),
        rows = rows, edge_budget = min(2e6,
            max(1, floor(0.2 * free / 48))),
        memory_limit = limit, host_estimate = bytes,
        vram_estimate_per_device = vram)
}

massive_cuda_resident_cluster <- function(graph, output, method,
        devices, chunk_rows, memory_limit, resolution,
        n_iterations, seed) {
    if (!method %in% c("louvain", "leiden")) stop(
        "Resident CUDA supports Louvain and Leiden.",
        call. = FALSE)
    plan <- massive_cuda_resident_plan(graph, output,
        devices, chunk_rows, memory_limit)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    before <- massive_checkpoint_file_identity(paths)
    if (!identical(before, graph$file_identity)) stop(
        "Resident CUDA graph identity changed.", call. = FALSE)
    raw <- massive_cluster_cuda_resident_csr_cpp(paths[1L],
        paths[2L], paths[3L], graph$n_vertices,
        plan$path, method, plan$devices,
        n_iterations, resolution, plan$rows,
        plan$edge_budget, plan$memory_limit, seed,
        contracted = identical(graph$weight_dtype, "float64"))
    if (!identical(raw$backend,
            "cuda_resident_multigpu_native") ||
        !identical(as.integer(raw$gpu_devices_used),
            plan$devices) ||
        !identical(before,
            massive_checkpoint_file_identity(paths))) stop(
        "Native resident CUDA result failed identity audit.",
        call. = FALSE)
    score <- massive_graph_modularity(graph, plan$path,
        raw$n_communities, resolution, plan$rows,
        memory_limit)
    if (!isTRUE(all.equal(score$modularity,
            raw$modularity, tolerance = 1e-5))) stop(
        "Native resident CUDA modularity failed audit.",
        call. = FALSE)
    structure(list(membership_path = plan$path,
        membership_identity = massive_checkpoint_file_identity(
            plan$path), n_vertices = graph$n_vertices,
        n_communities = raw$n_communities,
        modularity_final = score$modularity,
        total_edge_weight = score$total_edge_weight,
        method = method, backend = "cuda",
        gpu_devices = plan$devices, gpu_identity = plan$uuids,
        resources = plan, graph_identity = before,
        graph_audit = score,
        implementation = "native_resident_cuda_multigpu",
        mode = "out_of_core_graph", experimental = TRUE),
        class = "fastEmbedR_massive_cluster")
}
