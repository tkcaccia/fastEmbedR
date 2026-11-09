#' Optimize experimental disk-backed t-SNE affinities
#'
#' Optimizes a normalized symmetric CSR affinity graph from disk.
#' The CPU optimizer retains only its float32 coordinates and update state
#' in RAM, reuses the package's 2D or 3D FFT repulsion, and writes the
#' final coordinates to a file. CUDA streams affinities through a 2D or 3D
#' FFT optimizer. Metal maps the complete read-only graph into shared memory
#' for its 2D or 3D FFT optimizer, or streams bounded graph chunks from SSD.
#' Metal retains coordinates and update state in memory and reserves space
#' for its FFT workspace. It checks the device working-set budget before
#' allocation and during fitting. Graph streaming is not a fully streamed fit.
#' This is a full-graph fit, not landmark projection. Optimization uses
#' fixed iteration counts, without automatic KL stopping. Disk reads can
#' be substantial. All routes can checkpoint optimizer state.
#'
#' @param graph A result of [massive_tsne_affinities()].
#' @param init File-backed `.f32` initial coordinates with two or three columns.
#' @param output New `.f32` output path.
#' @param early_exaggeration_iter,n_iter Iterations in each phase.
#' @param early_exaggeration,exaggeration Phase affinity multipliers.
#' @param learning_rate Positive value or `"auto"` for `n / exaggeration`.
#' @param initial_momentum,final_momentum Phase momentum values.
#' @param min_gain Minimum per-coordinate adaptive gain.
#' @param max_step_norm Maximum update norm; `Inf` disables clipping.
#' @param memory_limit Host RAM budget. Coordinated CUDA fits reserve one
#'   bounded attraction-edge buffer per requested device.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"`. CUDA supports
#'   2D or 3D output with NCCL for multiple GPUs.
#' @param n.cores CPU worker count for FFT, sparse attraction, and updates.
#' @param devices CUDA device IDs. Multiple IDs run one coordinated fit.
#' @param graph_access Metal graph access: `"auto"`, `"mmap"`, or
#'   `"stream"`. Auto streams graphs larger than one edge chunk.
#' @param edge_chunk_capacity Maximum attraction edges per Metal SSD chunk.
#' @param checkpoint Save coordinates and optimizer state periodically.
#' @param resume Continue a matching interrupted fit.
#' @param checkpoint_every Iterations between state snapshots.
#' @return An experimental file-backed `fastEmbedR_massive_matrix`.
#'   CUDA results include host-call stage times. `attraction_sync`
#'   can include previously queued FFT work.
#' @keywords internal
massive_tsne_optimize <- function(graph, init, output,
        early_exaggeration_iter = 250L, n_iter = 500L,
        early_exaggeration = 12, exaggeration = 1,
        learning_rate = "auto", initial_momentum = 0.8,
        final_momentum = 0.8, min_gain = 0.01,
        max_step_norm = Inf, memory_limit = "8GB",
        backend = "cpu", n.cores = 1L, checkpoint = FALSE,
        resume = FALSE, checkpoint_every = 100L, devices = NULL,
        graph_access = "auto", edge_chunk_capacity = 8388608L) {
    plan <- massive_tsne_optimize_plan(graph, init, output,
        early_exaggeration_iter, n_iter, early_exaggeration,
        exaggeration, learning_rate, initial_momentum,
        final_momentum, min_gain, max_step_norm, memory_limit,
        backend, n.cores, devices, graph_access,
        edge_chunk_capacity)
    message("EXPERIMENTAL full-graph t-SNE: ", graph$n_vertices,
        " rows; ", graph$n_edges, " edges; backend=", backend)
    saved <- massive_tsne_checkpoint_prepare(plan, checkpoint,
        resume, checkpoint_every)
    args <- list(graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$access, init$path, plan$output,
        as.integer(graph$n_vertices), as.integer(init$ncol),
        plan$early_iter, plan$normal_iter, early_exaggeration,
        exaggeration, plan$learning_rate, plan$learning_rate_auto,
        initial_momentum, final_momentum, min_gain, max_step_norm)
    fit <- massive_tsne_native_fit(args, plan, saved, backend)
    if (backend == "cuda" && !identical(
        as.integer(fit$fft_grid_size), plan$fft_grid_size)) stop(
        "CUDA t-SNE changed its planned FFT grid.", call. = FALSE)
    if (!identical(plan$source_identity,
        massive_checkpoint_input_files_identity(plan$source_paths)))
        stop("t-SNE inputs changed during fitting.", call. = FALSE)
    if (length(plan$devices) > 1L) massive_tsne_validate_cuda_fit(
        plan, graph, fit, saved)
    result <- massive_tsne_result(plan, graph, init, fit, saved)
    if (!is.null(saved$sidecar)) unlink(c(saved$latest(), saved$sidecar))
    result
}

massive_tsne_native_fit <- function(args, plan, saved, backend) {
    if (backend == "cuda") {
        previous_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", NA_character_)
        on.exit(if (is.na(previous_grid))
            Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID") else
            Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = previous_grid),
            add = TRUE)
        Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = as.character(plan$fft_grid_size))
        return(do.call(massive_tsne_optimize_cuda_cpp, c(args,
            list(saved$iteration, saved$state_path, saved$every,
                saved$progress, plan$edge_capacity, plan$devices))))
    }
    if (backend == "metal") {
        return(do.call(massive_tsne_optimize_metal_cpp, c(args[-4L],
            list(saved$iteration, saved$state_path, saved$every,
                saved$progress, plan$graph_access,
                plan$edge_chunk_capacity))))
    }
    do.call(massive_tsne_optimize_cpp, c(args,
        list(plan$workers, saved$iteration, saved$state_path,
            saved$every, saved$progress)))
}

massive_tsne_validate_cuda_fit <- function(plan, graph, fit, saved) {
    total <- plan$early_iter + plan$normal_iter
    expected <- as.double(graph$n_edges) *
        as.double(total - saved$iteration)
    owned <- massive_graph_owner_edge_counts(graph, plan$devices) *
        as.double(total - saved$iteration)
    work <- fit$per_device_attraction_edges
    expected_work <- fit$per_device_expected_edges
    if (!identical(fit$backend_used,
            "native_cuda_fft_graph_coordinated_nccl") ||
        !identical(as.integer(fit$devices), plan$devices) ||
        !identical(as.integer(fit$iterations), as.integer(total)) ||
        !identical(as.integer(fit$resumed_from_iteration),
            as.integer(saved$iteration)) ||
        !is.numeric(work) || length(work) != length(plan$devices) ||
        any(!is.finite(work)) || any(work < 0) ||
        !is.numeric(expected_work) ||
        length(expected_work) != length(plan$devices) ||
        any(!is.finite(expected_work)) ||
        !identical(as.double(work), as.double(expected_work)) ||
        !identical(as.double(work), as.double(owned)) ||
        sum(expected_work) != expected ||
        sum(work) != expected) stop(
        "Coordinated CUDA t-SNE did not visit every graph edge.",
        call. = FALSE)
    stage <- fit$stage_seconds
    timing <- if (is.list(stage)) unlist(stage[c(
        "repulsion_enqueue", "graph_visit", "attraction_sync",
        "update_sync", "output_write")], use.names = FALSE)
    else numeric()
    if (length(timing) != 5L || !is.numeric(timing) ||
        any(!is.finite(timing)) || any(timing < 0)) stop(
        "Coordinated CUDA t-SNE timing is incomplete.", call. = FALSE)
    invisible(NULL)
}

massive_tsne_result <- function(plan, graph, init, fit, saved) {
    result <- massive_matrix(plan$output, graph$n_vertices, init$ncol)
    result$method <- "tsne"
    result$mode <- "out_of_core_graph"
    result$implementation_identity <-
        massive_checkpoint_implementation_identity()
    result$backend <- fit$backend_used
    result$graph_index <- list(
        offset_dtype = graph$offset_dtype,
        row_id_dtype = graph$index_dtype,
        edge_positions = as.double(graph$n_edges))
    result$optimizer <- fit$repulsion
    result$gpu_devices <- plan$devices
    result$gpu_identity <- plan$gpu_identity
    result$parameters <- plan$parameters
    result$parameters$checkpoint_every <- saved$every
    result$parameters$resumed_from_iteration <- saved$iteration
    result$resources <- massive_tsne_result_resources(
        plan, graph, fit, saved)
    if (plan$backend == "metal")
        result$metal_allocated_bytes_observed <-
            fit$metal_allocated_bytes_observed %||% NA_real_
    result$elapsed_seconds_current_call <- fit$elapsed_seconds
    if (!is.null(fit$stage_seconds)) result$stage_seconds <- fit$stage_seconds
    if (!is.null(fit$repulsion_gpu_seconds))
        result$gpu_stage_seconds <- list(
            repulsion = fit$repulsion_gpu_seconds)
    if (length(plan$devices) > 1L) {
        if (length(fit$owner_wall_seconds) != 1L ||
            !is.finite(fit$owner_wall_seconds) ||
            fit$owner_wall_seconds <= 0) stop(
            "Coordinated CUDA t-SNE lacks owner wall time.", call. = FALSE)
        result$resources$owner_wall_seconds <- fit$owner_wall_seconds
        result$resources$fft_owner_device <- plan$devices[[1L]]
    }
    result$fft_grid_size <- fit$fft_grid_size
    result$elapsed_seconds <- saved$prior_elapsed_seconds +
        fit$elapsed_seconds
    result$source_identity <- plan$source_identity
    result
}

massive_tsne_result_resources <- function(plan, graph, fit, saved) {
    resources <- list(estimated_peak_ram_bytes = plan$peak_ram,
        estimated_peak_vram_bytes = plan$peak_vram,
        graph_bytes = graph$n_edges * 8 + (graph$n_vertices + 1) * 8,
        checkpoint_disk_bytes = if (saved$every > 0L)
            2 * plan$state_bytes else 0,
        effective_cpu_workers = if (identical(plan$backend, "cpu"))
            fit$n_threads else NULL)
    if (identical(plan$backend, "cuda")) {
        resources$graph_feeder <-
            if (length(plan$devices) > 1L) "concurrent_owner" else "serial"
        resources$edge_batch_capacity <- plan$edge_capacity
    }
    if (identical(plan$backend, "metal"))
        resources$graph_access <- if (plan$graph_access == "stream")
            "bounded_stream" else "read_only_mmap"
    if (identical(plan$backend, "metal")) {
        resources$streaming_scope <- if (plan$graph_access ==
            "stream") "graph_upload_only" else "none"
        resources$optimizer_state_residency <-
            "admission_bounded_resident"
        resources$fft_workspace_residency <-
            "resident_with_reserved_headroom"
        resources$metal_state_admission_bytes <-
            fit$metal_state_admission_bytes %||% NA_real_
        resources$metal_working_set_cap_bytes <-
            fit$metal_working_set_cap_bytes %||% NA_real_
        resources$graph_upload_peak_bytes_estimate <-
            if (plan$graph_access == "stream")
                min(graph$n_edges, plan$edge_chunk_capacity) * 8 +
                    (min(graph$n_vertices, 262144) + 1) * 8
            else resources$graph_bytes
        resources$layout_state_bytes_estimate <-
            graph$n_vertices * plan$dimensions * 24 +
                graph$n_vertices * 8
        resources$fft_and_misc_reserve_bytes <- 1024 * 1024^2
    }
    if (!is.null(fit$per_device_attraction_edges))
        resources$per_device_attraction_edges <-
            fit$per_device_attraction_edges
    if (!is.null(fit$per_device_expected_edges))
        resources$per_device_expected_edges <-
            fit$per_device_expected_edges
    resources
}

massive_tsne_checkpoint_prepare <- function(plan, checkpoint,
        resume, checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    sidecar <- paste0(plan$output, ".checkpoint.rds")
    snapshots <- Sys.glob(paste0(plan$output,
        ".iter_*.state.f32*"))
    if (!checkpoint) {
        if (file.exists(sidecar) || length(snapshots)) stop(
            "t-SNE checkpoint files already exist.", call. = FALSE)
        return(list(iteration = 0L, state_path = "",
            every = 0L, progress = NULL, sidecar = NULL,
            prior_elapsed_seconds = 0))
    }
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L ||
        !identical(as.double(every),
            as.double(checkpoint_every))) stop(
        "`checkpoint_every` must be a positive integer.",
        call. = FALSE)
    extra <- plan$state_bytes * if (resume) 1 else 2
    if (extra + plan$state_bytes / 3 >
        0.8 * massive_disk_available_cpp(plan$output)) stop(
        "t-SNE snapshots exceed the free-disk budget.",
        call. = FALSE)
    signature <- massive_checkpoint_signature(list(version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        output = plan$output, source = plan$source_identity,
        n_vertices = plan$n_vertices,
        dimensions = plan$dimensions,
        parameters = plan$parameters, checkpoint_every = every))
    saved <- if (resume) {
        massive_tsne_checkpoint_load(plan, sidecar, signature)
    } else {
        if (file.exists(sidecar) || length(snapshots)) stop(
            "t-SNE checkpoint files already exist.", call. = FALSE)
        list(iteration = 0L, state_path = "", elapsed_seconds = 0)
    }
    state <- new.env(parent = emptyenv())
    state$previous <- saved$state_path
    state$prior_elapsed <- saved$elapsed_seconds
    progress <- function(iteration, snapshot, elapsed_seconds) {
        massive_tsne_checkpoint_commit(plan, sidecar, signature,
            state, iteration, snapshot, elapsed_seconds)
    }
    list(iteration = as.integer(saved$iteration),
        state_path = saved$state_path, every = every,
        progress = progress, sidecar = sidecar,
        prior_elapsed_seconds = saved$elapsed_seconds,
        latest = function() state$previous)
}

massive_tsne_checkpoint_load <- function(plan, sidecar,
        signature) {
    if (!file.exists(sidecar)) stop(
        "No t-SNE checkpoint exists to resume.", call. = FALSE)
    saved <- readRDS(sidecar)
    total <- plan$early_iter + plan$normal_iter
    valid <- is.numeric(saved$iteration) &&
        length(saved$iteration) == 1L &&
        is.finite(saved$iteration) &&
        saved$iteration >= 1L && saved$iteration <= total &&
        saved$iteration == floor(saved$iteration)
    expected <- if (valid) paste0(plan$output, ".iter_",
        saved$iteration, ".state.f32") else NULL
    if (!identical(saved$signature, signature) ||
        !valid ||
        !massive_checkpoint_same_paths(saved$state_path, expected) ||
        !is.numeric(saved$elapsed_seconds) ||
        length(saved$elapsed_seconds) != 1L ||
        !is.finite(saved$elapsed_seconds) ||
        saved$elapsed_seconds < 0 ||
        !file.exists(saved$state_path) ||
        !identical(saved$state_identity,
            massive_checkpoint_file_identity(saved$state_path)) ||
        !identical(saved$state_md5,
            unname(tools::md5sum(saved$state_path))) ||
        file.info(saved$state_path)$size != plan$state_bytes) stop(
        "t-SNE checkpoint does not match inputs or controls.",
        call. = FALSE)
    snapshots <- normalizePath(Sys.glob(paste0(plan$output,
        ".iter_*.state.f32*")), winslash = "/", mustWork = TRUE)
    if (length(setdiff(snapshots, saved$state_path))) stop(
        "Uncommitted t-SNE snapshot exists; inspect it before ",
        "resuming.", call. = FALSE)
    saved
}

massive_tsne_checkpoint_commit <- function(plan, sidecar,
        signature, state, iteration, snapshot,
        elapsed_seconds) {
    snapshot <- normalizePath(snapshot, winslash = "/", mustWork = TRUE)
    expected <- paste0(plan$output, ".iter_", iteration,
        ".state.f32")
    if (!massive_checkpoint_same_paths(snapshot, expected)) stop(
        "t-SNE snapshot path does not match its iteration.",
        call. = FALSE)
    if (!identical(plan$source_identity,
        massive_checkpoint_input_files_identity(plan$source_paths))) stop(
        "t-SNE inputs changed before checkpoint.", call. = FALSE)
    identity <- massive_checkpoint_file_identity(snapshot)
    if (identity$bytes != plan$state_bytes) stop(
        "t-SNE snapshot size does not match optimizer state.",
        call. = FALSE)
    hash <- unname(tools::md5sum(snapshot))
    if (length(hash) != 1L || is.na(hash)) stop(
        "Cannot hash t-SNE snapshot.", call. = FALSE)
    if (!is.numeric(elapsed_seconds) ||
        length(elapsed_seconds) != 1L ||
        !is.finite(elapsed_seconds) || elapsed_seconds < 0) stop(
        "Invalid t-SNE checkpoint elapsed time.", call. = FALSE)
    saved <- list(signature = signature, iteration = iteration,
        state_path = snapshot, state_identity = identity,
        state_md5 = hash,
        elapsed_seconds = state$prior_elapsed + elapsed_seconds)
    massive_checkpoint_write(saved, sidecar)
    if (nzchar(state$previous)) unlink(state$previous)
    state$previous <- snapshot
}

massive_tsne_devices <- function(devices, backend, rows) {
    selected <- massive_validate_devices(devices, backend, rows)
    if (backend != "cuda") return(NULL)
    if (is.null(selected)) selected <- as.integer(
        massive_cuda_memory_cpp()$device)
    if (length(selected) > 1L &&
            !isTRUE(massive_nccl_available_cpp())) stop(
        "Multi-GPU t-SNE requires NCCL; ",
        "no fallback was used.", call. = FALSE)
    if (length(selected) > 1L && rows <=
        (length(selected) - 1L) * ceiling(rows / length(selected))) stop(
        "Coordinated CUDA t-SNE needs rows on every device.",
        call. = FALSE)
    selected
}

massive_tsne_validate_graph_init <- function(graph, init, backend) {
    if (length(backend) != 1L || is.na(backend) ||
        !backend %in% c("cpu", "cuda", "metal"))
        stop("Full-graph t-SNE requires CPU, CUDA or Metal; ",
            "no backend fallback was used.", call. = FALSE)
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$weight_method, "tsne_compact_affinity") ||
        !isTRUE(graph$symmetrized)) stop(
        "Expected a normalized t-SNE affinity graph.", call. = FALSE)
    if (!inherits(init, "fastEmbedR_massive_matrix") ||
        !identical(init$format, "f32") ||
        init$nrow != graph$n_vertices ||
        !init$ncol %in% c(2, 3)) stop(
        "`init` must be a matching file-backed 2D/3D `.f32` matrix.",
        call. = FALSE)
    if (backend == "cuda" &&
        !isTRUE(embedding_cuda_available_cpp())) stop(
        "Native CUDA t-SNE unavailable; no backend fallback.",
        call. = FALSE)
    if (backend == "metal" &&
        !isTRUE(embedding_metal_available_cpp())) stop(
        "Native Metal t-SNE unavailable; no backend fallback.",
        call. = FALSE)
}

massive_tsne_metal_access <- function(graph, backend,
        graph_access, edge_chunk_capacity) {
    if (backend != "metal") {
        if (!identical(graph_access, "auto") ||
            !is.numeric(edge_chunk_capacity) ||
            length(edge_chunk_capacity) != 1L ||
            is.na(edge_chunk_capacity) ||
            edge_chunk_capacity != 8388608) stop(
            "Metal graph controls require `backend = 'metal'`.",
            call. = FALSE)
        return(list(mode = "auto", capacity = NULL))
    }
    if (!is.character(graph_access) || length(graph_access) != 1L ||
        is.na(graph_access) ||
        !graph_access %in% c("auto", "mmap", "stream")) stop(
        "Metal `graph_access` must be auto, mmap, or stream.",
        call. = FALSE)
    capacity <- integer_scalar(edge_chunk_capacity)
    if (is.na(capacity) || capacity < 1L ||
        !identical(as.double(capacity),
            as.double(edge_chunk_capacity))) stop(
        "`edge_chunk_capacity` must be a positive integer.",
        call. = FALSE)
    if (graph_access == "auto") graph_access <-
        if (graph$n_edges > capacity) "stream" else "mmap"
    list(mode = graph_access, capacity = capacity)
}

massive_tsne_optimize_plan <- function(graph, init, output,
        early_iter, normal_iter, early_exag, exag, learning_rate,
        momentum, final_momentum, min_gain, max_step,
        memory_limit, backend, n.cores, devices = NULL,
        graph_access = "auto", edge_chunk_capacity = 8388608L) {
    massive_tsne_validate_graph_init(graph, init, backend)
    access <- massive_tsne_metal_access(graph, backend,
        graph_access, edge_chunk_capacity)
    graph_access <- access$mode
    capacity <- access$capacity
    selected <- massive_tsne_devices(devices, backend,
        graph$n_vertices)
    identity <- if (!is.null(devices))
        massive_cuda_device_uuids(selected) else NULL
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !grepl("\\.f32$", output) ||
        !dir.exists(dirname(output))) stop(
        "`output` must be a new `.f32` path in an existing directory.",
        call. = FALSE)
    path <- file.path(normalizePath(dirname(output)), basename(output))
    if (file.exists(path) || file.exists(paste0(path, ".part"))) stop(
        "t-SNE output or partial file already exists.", call. = FALSE)
    controls <- massive_tsne_validate_controls(early_iter,
        normal_iter, early_exag, exag, learning_rate, momentum,
        final_momentum, min_gain, max_step, n.cores)
    if (backend == "metal" && controls$workers != 1L) stop(
        "Metal t-SNE requires `n.cores = 1`; no backend fallback.",
        call. = FALSE)
    controls$parameters$devices <- selected
    controls$parameters$gpu_identity <- identity
    if (backend == "metal") {
        controls$parameters$graph_access <- graph_access
        controls$parameters$edge_chunk_capacity <- capacity
    }
    resources <- massive_tsne_resource_plan(graph, init, path,
        memory_limit, backend, selected, graph_access, capacity)
    controls$parameters$fft_grid_size <- resources$fft_grid_size
    controls$parameters$graph_offset_dtype <- graph$offset_dtype
    controls$parameters$graph_index_dtype <- graph$index_dtype
    controls$parameters$graph_edge_positions <- as.double(graph$n_edges)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path, init$path)
    massive_checkpoint_validated_graph_identity(graph, paths[1:3])
    c(list(output = path, source_paths = paths,
        source_identity = massive_checkpoint_input_files_identity(paths),
        n_vertices = graph$n_vertices, dimensions = init$ncol,
        devices = selected, gpu_identity = identity,
        backend = backend, graph_access = graph_access,
        edge_chunk_capacity = capacity), resources, controls)
}

massive_tsne_resource_plan <- function(graph, init, path,
        memory_limit, backend, devices, graph_access = "auto",
        edge_chunk_capacity = NULL) {
    limit <- massive_auto_memory_limit(memory_limit)
    coords <- graph$n_vertices * init$ncol
    feeders <- max(1L, length(devices))
    if (coords > .Machine$integer.max) stop(
        "FFT coordinate count exceeds the native integer limit.",
        call. = FALSE)
    capacity <- if (backend == "cuda") {
        max(65536L, min(1000000L,
            floor((0.8 * limit - 64 * 1024^2) / (12 * feeders))))
    } else NULL
    peak <- if (backend == "cuda") {
        64 * 1024^2 + capacity * 12 * feeders
    } else if (backend == "metal") {
        graph_bytes <- if (graph_access == "stream")
            min(graph$n_edges, edge_chunk_capacity) * 8 +
                (min(graph$n_vertices, 262144) + 1) * 8 else
            graph$n_edges * 8 + (graph$n_vertices + 1) * 8
        graph_bytes +
            coords * 24 + graph$n_vertices * 8 +
            1024 * 1024^2
    } else {
        coords * 16 +
            as.integer(init$ncol == 3) * graph$n_vertices * 4 +
            512 * 1024^2
    }
    if (peak > 0.8 * limit) stop(
        "t-SNE optimizer exceeds `memory_limit`.", call. = FALSE)
    if (coords * 4 >
        0.8 * massive_disk_available_cpp(path)) stop(
        "t-SNE output exceeds free disk space.", call. = FALSE)
    grid <- massive_tsne_fft_grid(graph, backend, init$ncol)
    vram <- if (backend == "cuda") {
        fft_bytes <- if (init$ncol == 3L)
            2 * (2 * grid)^3 * 8 else 9 * (2 * grid)^2 * 8
        coords * 16 + graph$n_vertices * 8 +
            fft_bytes + capacity * 12 + 256 * 1024^2
    } else NA_real_
    if (backend == "cuda" &&
        vram > 0.8 * massive_auto_cuda_free(devices,
            graph$n_vertices)) stop(
        "CUDA t-SNE layout and FFT workspace exceed free VRAM.",
        call. = FALSE)
    list(peak_ram = peak, peak_vram = vram, fft_grid_size = grid,
        edge_capacity = capacity, state_bytes = coords * 12)
}

massive_tsne_fft_grid <- function(graph, backend, dimensions = 3L) {
    if (backend == "cuda" && dimensions == 3L) {
        if (graph$n_vertices < 5000) 16L else 64L
    } else if (backend == "cuda") {
        requested <- suppressWarnings(as.integer(
            Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", "")))
        if (!is.na(requested) && requested %in%
            c(32L, 64L, 128L, 256L, 512L, 1024L)) requested
        else if (graph$n_vertices < 20000) 256L
        else if (graph$n_vertices >= 100000) 1024L
        else 512L
    } else if (backend == "metal" && dimensions == 2L) {
        fallback <- if (graph$n_vertices >= 10000) 256L else 128L
        requested <- suppressWarnings(as.integer(
            Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", "")))
        if (is.na(requested) || requested < 1L) requested <- fallback
        grid <- 32L
        while (grid < requested && grid < 512L) grid <- grid * 2L
        grid
    } else if (backend == "metal") {
        if (graph$n_vertices < 5000) 16L else 64L
    } else NULL
}

massive_tsne_validate_controls <- function(early_iter, normal_iter,
        early_exag, exag, learning_rate, momentum,
        final_momentum, min_gain, max_step, n.cores) {
    if (!is.numeric(early_iter) || !is.numeric(normal_iter) ||
        length(early_iter) != 1L || length(normal_iter) != 1L) stop(
        "t-SNE iterations must be numeric scalar counts.",
        call. = FALSE)
    iterations <- vapply(list(early_iter, normal_iter),
        integer_scalar, integer(1L))
    if (anyNA(iterations) || any(iterations < 0L) ||
        !identical(as.double(iterations),
            as.double(c(early_iter, normal_iter))) ||
        sum(as.double(iterations)) < 1 ||
        sum(as.double(iterations)) > .Machine$integer.max) stop(
        "t-SNE iterations must be non-negative with a positive total.",
        call. = FALSE)
    values <- c(early_exag, exag, momentum, final_momentum, min_gain)
    if (!is.numeric(values) || length(values) != 5L ||
        any(!is.finite(values)) || any(values[c(1, 2, 5)] <= 0) ||
        any(values[c(3, 4)] < 0) ||
        !is.numeric(max_step) || length(max_step) != 1L ||
        is.na(max_step) || max_step <= 0) stop(
        "Invalid t-SNE phase or step controls.", call. = FALSE)
    automatic <- identical(learning_rate, "auto")
    if (!automatic && (!is.numeric(learning_rate) ||
        length(learning_rate) != 1L ||
        !is.finite(learning_rate) || learning_rate <= 0)) stop(
        "`learning_rate` must be positive or `auto`.", call. = FALSE)
    workers <- integer_scalar(n.cores)
    if (is.na(workers) || workers < 1L || workers > 256L ||
        !identical(as.double(workers), as.double(n.cores))) stop(
        "`n.cores` must be an integer from 1 to 256.", call. = FALSE)
    list(early_iter = iterations[[1L]],
        normal_iter = iterations[[2L]], workers = workers,
        learning_rate_auto = automatic,
        learning_rate = if (automatic) 1 else learning_rate,
        parameters = list(
            early_exaggeration_iter = iterations[[1L]],
            n_iter = iterations[[2L]],
            early_exaggeration = early_exag, exaggeration = exag,
            learning_rate = learning_rate,
            initial_momentum = momentum,
            final_momentum = final_momentum,
            min_gain = min_gain, max_step_norm = max_step,
            n.cores = workers, auto_kld_stop = FALSE))
}
