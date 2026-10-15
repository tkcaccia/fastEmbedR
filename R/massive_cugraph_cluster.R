massive_cugraph_graph_identity <- function(graph) {
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    massive_checkpoint_validated_graph_identity(graph, paths)
}

massive_cugraph_plan <- function(graph, output, devices,
        chunk_rows, memory_limit) {
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !identical(graph$weight_method, "umap_fuzzy_union") ||
        !isTRUE(graph$symmetrized)) stop(
        "cuGraph needs a symmetric massive UMAP fuzzy graph.",
        call. = FALSE)
    devices <- massive_validate_devices(devices, "cuda",
        graph$n_vertices)
    if (!(length(devices) %in% 2:4)) stop(
        "cuGraph requires two to four CUDA-visible devices; no fallback.",
        call. = FALSE)
    rows <- integer_scalar(chunk_rows %||% 8192L)
    limit <- massive_memory_bytes(memory_limit)
    staging <- 48 * graph$n_vertices +
        40 * rows * graph$max_degree * length(devices)
    if (is.na(rows) || rows < 1L ||
        staging > 0.7 * limit) stop(
        "cuGraph CSR staging or label audit exceeds RAM budget.",
        call. = FALSE)
    estimate <- 512 * 1024^2 +
        96 * graph$n_edges / length(devices)
    free <- massive_auto_cuda_free(devices, graph$n_vertices)
    if (estimate > 0.7 * free) stop(
        "cuGraph full graph exceeds free GPU memory; no fallback.",
        call. = FALSE)
    path <- massive_louvain_output(output,
        12 * graph$n_vertices + 1024^2)
    if (any(file.exists(paste0(path, c(".part", ".cugraph.json",
            ".cugraph.out", ".cugraph.err"))))) stop(
        "cuGraph output files already exist; choose a new output.",
        call. = FALSE)
    list(path = path, devices = devices, uuids =
        massive_cuda_device_uuids(devices), chunk_rows = rows,
        limit = limit, ram_staging_estimate_bytes = staging,
        vram_estimate = estimate, free_vram = free)
}

massive_cugraph_invoke <- function(graph, plan, method,
        controls) {
    python <- Sys.getenv("FASTEMBEDR_CUGRAPH_PYTHON", "python3")
    if (!nzchar(Sys.which(python))) stop(
        "cuGraph Python is unavailable; no CPU fallback.",
        call. = FALSE)
    script <- system.file("python", "massive_cugraph_cluster.py",
        package = "fastEmbedR", mustWork = TRUE)
    metadata <- paste0(plan$path, ".cugraph.json")
    paths <- c(graph$offsets_path, graph$indices_path, graph$weights_path)
    args <- c(script, "--paths", paths, "--vertices",
        format(graph$n_vertices, scientific = FALSE, trim = TRUE),
        "--chunk-rows", plan$chunk_rows,
        "--method", method, "--iterations", controls$passes,
        "--resolution", controls$resolution, "--seed",
        controls$seed, "--uuids", plan$uuids,
        "--vram-estimate", format(ceiling(plan$vram_estimate),
            scientific = FALSE, trim = TRUE),
        "--ram-limit", format(floor(plan$limit),
            scientific = FALSE, trim = TRUE),
        "--output", paste0(plan$path, ".part"),
        "--metadata", metadata)
    status <- system2(python, shQuote(args),
        stdout = paste0(plan$path, ".cugraph.out"),
        stderr = paste0(plan$path, ".cugraph.err"),
        env = paste0("CUDA_VISIBLE_DEVICES=",
            paste(plan$uuids, collapse = ",")))
    if (!identical(status, 0L)) stop(
        "cuGraph failed; inspect the .cugraph.err log. No fallback.",
        call. = FALSE)
    if (!file.exists(metadata) || !file.exists(
            paste0(plan$path, ".part"))) stop(
        "cuGraph did not write complete labels and metadata.",
        call. = FALSE)
    jsonlite::fromJSON(metadata, simplifyVector = FALSE)
}

massive_cugraph_worker_map <- function(workers, devices) {
    valid <- function(item) is.list(item) &&
        is.character(item$address) && length(item$address) == 1L &&
        nzchar(item$address) && is.numeric(item$pid) &&
        length(item$pid) == 1L && is.finite(item$pid) &&
        item$pid > 0 && item$pid == floor(item$pid) &&
        is.character(item$pci_bus_id) &&
        length(item$pci_bus_id) == 1L &&
        nzchar(item$pci_bus_id) && is.character(item$uuid) &&
        length(item$uuid) == 1L && nzchar(item$uuid)
    if (!is.list(workers) || length(workers) != length(devices) ||
        !all(vapply(workers, valid, TRUE))) stop(
        "cuGraph worker GPU map is incomplete.", call. = FALSE)
    uuids <- vapply(workers, `[[`, "", "uuid")
    buses <- vapply(workers, `[[`, "", "pci_bus_id")
    pids <- vapply(workers, `[[`, 0, "pid")
    expected <- stats::setNames(vapply(devices, `[[`, "",
        "pci_bus_id"), vapply(devices, `[[`, "", "uuid"))
    if (anyDuplicated(uuids) || anyDuplicated(buses) ||
        anyDuplicated(pids) || !setequal(uuids, names(expected)) ||
        !isTRUE(all(buses == unname(expected[uuids])))) stop(
        "cuGraph worker GPU map is incomplete.", call. = FALSE)
    invisible(NULL)
}

massive_cugraph_memory <- function(metadata, plan) {
    memory <- metadata$gpu_memory
    host <- if (is.list(memory)) memory$host_process_tree else NULL
    if (!is.list(memory) || !is.numeric(memory$samples) ||
        length(memory$samples) != 1L || memory$samples < 2L ||
        !identical(memory$scope, "whole_physical_gpu") ||
        !identical(metadata$distributed_graph, TRUE) ||
        !identical(as.integer(metadata$worker_count),
            length(plan$devices)) ||
        length(memory$devices) != length(plan$devices) ||
        !identical(vapply(memory$devices, `[[`, "", "uuid"),
            unname(plan$uuids)) ||
        !identical(metadata$device_spilling, FALSE)) stop(
        "cuGraph GPU memory telemetry is incomplete.", call. = FALSE)
    massive_cugraph_worker_map(memory$workers, metadata$devices)
    if (!is.list(host) ||
        !identical(host$scope, "launcher_and_workers") ||
        !is.numeric(host$samples) || length(host$samples) != 1L ||
        !is.finite(host$samples) ||
        host$samples != memory$samples ||
        !is.numeric(host$tracked_samples) ||
        length(host$tracked_samples) != 1L ||
        host$tracked_samples < 2L ||
        host$tracked_samples > host$samples ||
        !is.numeric(host$baseline_rss_bytes) ||
        !is.numeric(host$peak_rss_bytes) ||
        length(host$baseline_rss_bytes) != 1L ||
        length(host$peak_rss_bytes) != 1L ||
        !is.finite(host$baseline_rss_bytes) ||
        !is.finite(host$peak_rss_bytes) ||
        host$baseline_rss_bytes < 0 ||
        host$peak_rss_bytes < host$baseline_rss_bytes) stop(
        "cuGraph host RAM telemetry is incomplete.", call. = FALSE)
    if (host$peak_rss_bytes > plan$limit) stop(
        "cuGraph exceeded the requested RAM budget.", call. = FALSE)
    peaks <- vapply(memory$devices, function(item) {
        values <- c(item$baseline_used_bytes,
            item$peak_used_bytes, item$peak_delta_bytes)
        if (length(values) != 3L || !is.numeric(values) ||
            any(!is.finite(values)) || any(values < 0) ||
            values[[2L]] < values[[1L]] ||
            values[[3L]] != values[[2L]] - values[[1L]]) stop(
            "cuGraph GPU peak is invalid.", call. = FALSE)
        values[[2L]]
    }, 0)
    memory$peak <- max(peaks)
    memory
}

massive_cugraph_controls_check <- function(metadata, method, controls) {
    expected_semantics <- if (method == "leiden")
        "max_iter" else "max_level"
    scalar <- function(value, expected) is.numeric(value) &&
        length(value) == 1L && is.finite(value) && value == expected
    if (!identical(metadata$method, method) ||
        !identical(metadata$backend, "cugraph_dask") ||
        !identical(metadata$iteration_semantics,
            expected_semantics) ||
        !identical(metadata$seed_applied, method == "leiden") ||
        !scalar(metadata$iterations_requested, controls$passes) ||
        !scalar(metadata$seed_requested, controls$seed) ||
        !is.numeric(metadata$resolution) ||
        length(metadata$resolution) != 1L ||
        !is.finite(metadata$resolution) ||
        !isTRUE(all.equal(metadata$resolution,
            controls$resolution, tolerance = 1e-12))) stop(
        "cuGraph did not report the requested clustering controls.",
        call. = FALSE)
    invisible(NULL)
}

massive_cugraph_finish <- function(graph, plan, method,
        before, metadata, resolution, memory_limit) {
    part <- paste0(plan$path, ".part")
    if (!identical(before, massive_cugraph_graph_identity(graph))) stop(
        "Massive graph changed during cuGraph fitting.", call. = FALSE)
    if (file.info(part)$size != graph$n_vertices * 4 ||
        metadata$rows != graph$n_vertices ||
        length(metadata$devices) != length(plan$devices) ||
        !identical(vapply(metadata$devices, `[[`, "", "uuid"),
            unname(plan$uuids))) stop(
        "cuGraph output omitted rows or changed devices.", call. = FALSE)
    memory <- massive_cugraph_memory(metadata, plan)
    score <- massive_graph_modularity(graph, part, metadata$communities,
        resolution, plan$chunk_rows, memory_limit,
        connectivity = method == "leiden")
    if (method == "leiden" && (!isTRUE(score$connectivity_audited) ||
        !identical(score$disconnected_communities, 0L))) stop(
        "cuGraph Leiden returned disconnected communities.", call. = FALSE)
    gap <- abs(score$modularity - metadata$modularity_reported)
    if (!is.finite(gap) ||
        (method == "louvain" && gap > 0.005)) stop(
        sprintf("cuGraph modularity disagrees (source=%.6f, reported=%.6f).",
            score$modularity, metadata$modularity_reported),
        call. = FALSE)
    if (gap > 0.005) warning("verified graph score used.", call. = FALSE)
    if (!identical(before, massive_cugraph_graph_identity(graph)))
        stop("Massive graph changed during cuGraph fitting.", call. = FALSE)
    if (!file.rename(part, plan$path)) stop("Cannot finalize cuGraph labels.")
    result <- list(membership_path = plan$path,
        membership_identity = massive_checkpoint_file_identity(plan$path),
        n_vertices = graph$n_vertices, n_communities = metadata$communities,
        modularity_final = score$modularity, modularity_gap = gap,
        modularity_reported = metadata$modularity_reported,
        passes = if (method == "louvain") metadata$iterations_requested
            else NULL,
        iterations_requested = metadata$iterations_requested,
        method = paste0(method, "_cugraph_dask"), backend = "cuda",
        gpu_devices = plan$devices, gpu_identity = metadata$devices,
        graph_identity = before, cugraph = metadata, graph_audit = score,
        resources = list(free_vram_preflight_bytes = plan$free_vram,
            vram_estimate_bytes = plan$vram_estimate,
            ram_staging_estimate_bytes = plan$ram_staging_estimate_bytes,
            measured_vram_peak_bytes = memory$peak,
            measured_vram_peak_per_device = memory$devices,
            measured_vram_scope = memory$scope,
            measured_ram_peak_bytes = memory$host_process_tree$peak_rss_bytes,
            measured_ram_scope = "launcher_and_workers"), experimental = TRUE)
    structure(result, class = if (method == "leiden")
        "fastEmbedR_massive_leiden" else "fastEmbedR_massive_louvain_level")
}

massive_cugraph_cluster <- function(graph, output, method,
        devices, chunk_rows, memory_limit, resolution,
        n_iterations, seed) {
    implementation <- massive_checkpoint_implementation_identity()
    worker <- system.file("python", "massive_cugraph_cluster.py",
        package = "fastEmbedR", mustWork = TRUE)
    worker_hash <- unname(tools::md5sum(worker))
    if (is.na(worker_hash)) stop(
        "Cannot hash the cuGraph worker.", call. = FALSE)
    controls <- massive_louvain_controls(n_iterations,
        resolution, seed)
    if (method == "louvain" && controls$seed != 1L) stop(
        "cuGraph Louvain has no seed control; use default seed = 1.",
        call. = FALSE)
    plan <- massive_cugraph_plan(graph, output, devices,
        chunk_rows, memory_limit)
    before <- massive_cugraph_graph_identity(graph)
    metadata <- massive_cugraph_invoke(graph, plan, method,
        controls)
    massive_cugraph_controls_check(metadata, method, controls)
    if (!identical(implementation,
            massive_checkpoint_implementation_identity()) ||
        !identical(worker_hash, unname(tools::md5sum(worker)))) stop(
        "cuGraph implementation changed during fitting.",
        call. = FALSE)
    result <- massive_cugraph_finish(graph, plan, method,
        before, metadata, resolution, memory_limit)
    result$implementation_identity <- implementation
    result$python_worker_md5 <- worker_hash
    result
}
