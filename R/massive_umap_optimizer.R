#' Optimize an experimental disk-backed UMAP graph
#'
#' Streams a symmetric fuzzy CSR graph for each epoch. CPU coordinates use
#' float32 RAM or an explicitly requested writable memory-mapped file.
#' CUDA streams bounded edge batches. Its default layout must fit VRAM;
#' explicit managed storage allows CUDA paging from host RAM.
#' Sampling times are derived from edge weights instead of storing a
#' per-edge schedule. CPU supports file-backed 2D or 3D initialization;
#' Completed CPU, Metal, or CUDA epochs can be checkpointed and resumed without
#' rescanning earlier epochs.
#'
#' @param graph A result of [massive_umap_fuzzy_graph()].
#' @param init A file-backed [massive_matrix()] of initial coordinates.
#' @param output New `.f32` file for the resulting coordinates.
#' @param n_epochs Number of optimization epochs, at least two because
#'   the first epoch initializes the edge-sampling schedule.
#' @param negative_sample_rate Negative samples per positive edge.
#' @param learning_rate Initial learning rate.
#' @param min_dist UMAP minimum-distance curve parameter.
#' @param repulsion_strength Repulsive-force multiplier.
#' @param seed Random seed for negative sampling.
#' @param chunk_rows Maximum graph rows per sequential read.
#' @param memory_limit Host working-buffer budget. The full layout counts in
#'   CPU RAM mode; OS page cache for a mapped layout is not capped by this
#'   value. CUDA preflights device allocations against free device memory;
#'   managed CUDA coordinates count against the host budget.
#' @param backend `"cpu"`, `"metal"`, or `"cuda"`; GPU requests have no
#'   CPU fallback. Metal streams edges but keeps one 2D layout in shared
#'   unified memory.
#' @param n.cores Only one worker is currently supported.
#' @param layout_storage `"memory"` (default), `"mmap"` for a writable
#'   mapping, or `"managed"` for explicit CUDA unified memory. Managed
#'   storage may page between host and GPU and requires a supporting GPU.
#'   Metal `"mmap"` maps the host staging/output file, but still requires
#'   one complete 2D layout in Metal shared memory.
#' @param devices Optional one to four CUDA device IDs. Explicit devices
#'   use NCCL to reduce edge updates to coordinate owners at each epoch;
#'   fixed-point accumulation makes the reduction order-independent.
#'   Omitting this argument retains the in-place single-GPU optimizer.
#'   Each coordinated device stores a full read-only layout, so this
#'   route does not pool VRAM for inputs larger than one device.
#' @param checkpoint Save a full coordinate snapshot at completed epochs.
#' @param resume Continue a matching interrupted fit on the same backend.
#' @param checkpoint_every Number of epochs between snapshots.
#' @return An experimental file-backed `fastEmbedR_massive_matrix`.
#' @export
massive_umap_optimize <- function(graph, init, output,
        n_epochs = 200L, negative_sample_rate = 5L, learning_rate = 1,
        min_dist = 0.1, repulsion_strength = 1, seed = 1L,
        chunk_rows = 8192L, memory_limit = "16GB", backend = "cpu",
        n.cores = 1L, checkpoint = FALSE, resume = FALSE,
        checkpoint_every = 10L, layout_storage = "memory",
        devices = NULL) {
    plan <- massive_umap_optimize_plan(graph, init, output,
        n_epochs, negative_sample_rate, learning_rate, min_dist,
        repulsion_strength, seed, chunk_rows, memory_limit,
        backend, n.cores, layout_storage, resume, devices)
    message("EXPERIMENTAL full-graph UMAP: ", graph$n_vertices,
        " rows; ", graph$n_edges, " edges; backend=", backend,
        "; layout storage=", if (backend == "cuda" &&
            layout_storage == "memory") "device" else layout_storage)
    saved <- massive_umap_checkpoint_prepare(plan, checkpoint, resume,
        checkpoint_every)
    built <- massive_umap_native_fit(graph, init, plan, saved,
        learning_rate, min_dist, repulsion_strength, layout_storage)
    if (!identical(plan$source_identity,
        massive_checkpoint_input_files_identity(plan$source_paths))) {
        stop("UMAP graph or initialization changed during fitting.",
            call. = FALSE)
    }
    if (plan$backend == "cuda" && is.null(plan$devices) &&
        !identical(built$backend_used, "native_cuda_umap_stream")) stop(
        "CUDA UMAP used an unexpected native route.", call. = FALSE)
    if (!is.null(plan$devices))
        massive_umap_validate_cuda_fit(plan, graph, built, saved)
    massive_umap_result(plan, graph, init, built, saved, checkpoint,
        layout_storage)
}

massive_umap_validate_cuda_fit <- function(plan, graph, built, saved) {
    work <- built$per_device_edge_visits
    rows <- built$per_device_row_visits
    previous <- saved$edge_visits
    active_edges <- massive_graph_owner_active_counts(graph,
        plan$devices, built$max_weight, plan$n_epochs)
    expected_work <- active_edges * (plan$n_epochs - saved$epoch)
    block <- ceiling(graph$n_vertices / length(plan$devices))
    owned <- pmin(block, pmax(0, graph$n_vertices -
        (seq_along(plan$devices) - 1L) * block))
    expected_rows <- owned * (plan$n_epochs - saved$epoch)
    if (!identical(built$backend_used,
            "native_cuda_umap_coordinated_nccl") ||
        !identical(as.integer(built$devices), plan$devices) ||
        !identical(as.double(built$n_vertices),
            as.double(graph$n_vertices)) ||
        !identical(as.double(built$n_edges),
            as.double(graph$n_edges)) ||
        !is.numeric(work) || length(work) != length(plan$devices) ||
        any(!is.finite(work)) || any(work < 0) ||
        !identical(as.double(work), as.double(expected_work)) ||
        !is.numeric(rows) || length(rows) != length(plan$devices) ||
        any(!is.finite(rows)) ||
        !identical(as.double(rows), as.double(expected_rows)) ||
        !is.numeric(previous) || length(previous) != 1L ||
        !is.finite(previous) || previous < 0 ||
        !is.numeric(built$edge_visits) ||
        length(built$edge_visits) != 1L ||
        !is.finite(built$edge_visits) ||
        sum(work) != built$edge_visits - previous) stop(
        "Coordinated CUDA UMAP work accounting is inconsistent.",
        call. = FALSE)
    timing <- unlist(built[c("graph_visit_seconds",
        "device_step_seconds", "synchronization_seconds",
        "finalization_block_seconds")], use.names = FALSE)
    if (length(timing) != 4L || !is.numeric(timing) ||
        any(!is.finite(timing)) || any(timing < 0)) stop(
        "Coordinated CUDA UMAP timing is incomplete.", call. = FALSE)
    invisible(NULL)
}

massive_umap_native_fit <- function(graph, init, plan, saved,
        learning_rate, min_dist, repulsion_strength, layout_storage) {
    backend <- plan$backend
    built <- if (backend == "cuda") {
        massive_umap_optimize_cuda_cpp(graph$offsets_path,
            graph$indices_path, graph$weights_path, graph$n_vertices,
            saved$start_path, saved$format, plan$output,
            plan$n_epochs, plan$negative_sample_rate,
            learning_rate, min_dist, repulsion_strength, plan$seed,
            plan$chunk_rows, plan$edge_capacity, plan$limit,
            saved$epoch, saved$every, saved$progress,
            saved$edge_visits, init$ncol, layout_storage,
            plan$devices %||% integer())
    } else if (backend == "metal") {
        massive_umap_optimize_metal_cpp(graph$offsets_path,
            graph$indices_path, graph$weights_path, graph$n_vertices,
            saved$start_path, saved$format, plan$output,
            plan$n_epochs, plan$negative_sample_rate,
            learning_rate, min_dist, repulsion_strength, plan$seed,
            plan$chunk_rows, plan$edge_capacity, plan$limit,
            saved$epoch, saved$positive, saved$negative,
            saved$every, saved$progress, layout_storage)
    } else massive_umap_optimize_cpp(graph$offsets_path,
        graph$indices_path, graph$weights_path, graph$n_vertices,
        saved$start_path, saved$format, init$ncol, plan$output,
        plan$n_epochs, plan$negative_sample_rate, learning_rate,
        min_dist, repulsion_strength, plan$seed,
        plan$chunk_rows, plan$limit, saved$epoch,
        saved$positive, saved$negative, saved$every,
        saved$progress, layout_storage)
    built
}

massive_umap_result <- function(plan, graph, init, built, saved,
                                checkpoint, layout_storage) {
    result <- massive_matrix(plan$output, graph$n_vertices, init$ncol)
    result$method <- "umap"
    result$mode <- "out_of_core_graph"
    result$implementation_identity <-
        massive_checkpoint_implementation_identity()
    result$backend <- plan$backend
    result$optimizer <- if (plan$backend == "metal")
        "streamed_metal_atomic" else if (is.null(plan$devices))
        paste0("streamed_", plan$backend, "_sgd") else "synchronized_cuda_nccl"
    result$layout_storage <- if (plan$backend == "cuda" &&
        layout_storage == "memory") "device" else layout_storage
    result$parameters <- plan$parameters
    result$parameters$checkpoint_every <- saved$every
    result$parameters$resumed_from_epoch <- saved$epoch
    layout_ram <- if (plan$backend == "cuda" &&
        layout_storage == "managed") {
        plan$layout_bytes + plan$edge_capacity * 16
    } else if (plan$backend == "cuda") {
        min(plan$chunk_rows, 8192L) * init$ncol * 4 +
            plan$edge_capacity * 16 * max(1L, length(plan$devices))
    } else if (plan$backend == "metal")
        (if (layout_storage == "memory") 2 * plan$layout_bytes
        else plan$layout_bytes + min(plan$chunk_rows, 8192L) *
            init$ncol * 4) + plan$edge_capacity * 32
    else if (layout_storage == "memory") plan$layout_bytes
    else min(plan$chunk_rows, 8192L) * init$ncol * 4
    result$resources <- list(layout_ram_bytes = layout_ram,
        layout_mmap_bytes = plan$layout_bytes * (layout_storage == "mmap"),
        graph_bytes = graph$n_edges * 8 + (graph$n_vertices + 1) * 8,
        checkpoint_disk_bytes = plan$layout_bytes *
            (2 + as.integer(layout_storage == "mmap")) *
            as.integer(checkpoint))
    if (plan$backend == "cuda") result <-
        massive_umap_cuda_result(result, plan, built, layout_storage)
    if (plan$backend == "metal") result$resources <- c(result$resources,
        list(metal_memory_model = "unified_shared",
            metal_buffer_bytes_estimate = plan$layout_bytes +
                16 * plan$edge_capacity,
            edge_batch_capacity = plan$edge_capacity,
            peak_vram_bytes = NA_real_))
    result$updates <- built[c("positive_updates", "negative_updates")]
    if (plan$backend == "cuda") result$updates$edge_visits <-
        built$edge_visits
    result$source_identity <- plan$source_identity
    if (!is.null(saved$sidecar)) massive_umap_checkpoint_cleanup(
        plan$output, saved$sidecar)
    result
}

massive_umap_cuda_result <- function(result, plan, built,
        layout_storage) {
    result$gpu_identity <- plan$gpu_identity
    result$native_backend_used <- built$backend_used
    device_layout <- massive_umap_cuda_layout_bytes(result$nrow,
        result$ncol, plan$devices)
    result$resources$device_buffer_bytes <-
        plan$edge_capacity * 16 +
        if (!is.null(plan$devices)) 3 * device_layout else
        if (layout_storage == "memory") plan$layout_bytes else 0
    result$resources$managed_layout_bytes <-
        if (layout_storage == "managed") plan$layout_bytes else 0
    result$resources$edge_batch_capacity <- plan$edge_capacity
    result$stage_seconds <- list(
        graph_visit = built$graph_visit_seconds,
        device_step = built$device_step_seconds,
        synchronization = built$synchronization_seconds,
        finalization = built$finalization_block_seconds)
    result$gpu_devices <- if (is.null(plan$devices)) {
        as.integer(massive_cuda_memory_cpp()$device)
    } else built$devices
    result$resources$graph_feeder <- if (is.null(plan$devices)) {
        "serial"
    } else "concurrent_owner"
    if (is.null(plan$devices)) return(result)
    if (length(plan$devices) > 1L &&
        (length(built$owner_wall_seconds) != 1L ||
        !is.finite(built$owner_wall_seconds) ||
        built$owner_wall_seconds <= 0)) stop(
        "Coordinated CUDA UMAP lacks owner wall time.",
        call. = FALSE)
    result$resources$per_device_edge_visits <-
        built$per_device_edge_visits
    result$resources$per_device_row_visits <-
        built$per_device_row_visits
    result$resources$owner_wall_seconds <- built$owner_wall_seconds
    result$resources$delta_scale <- built$delta_scale
    result$resources$synchronization_seconds <-
        built$synchronization_seconds
    result$resources$finalization_block_seconds <-
        built$finalization_block_seconds
    result$resources$layout_replication_bytes <-
        if (length(plan$devices) > 1L) {
            length(plan$devices) * device_layout
        } else 0
    result
}

massive_umap_checkpoint_cleanup <- function(output, sidecar) {
    snapshots <- Sys.glob(paste0(output, ".epoch_*.f32"))
    checkpoint_files <- c(snapshots, sidecar)
    unlink(checkpoint_files)
    if (any(file.exists(checkpoint_files))) stop(
        "Could not remove completed UMAP checkpoints.",
        call. = FALSE)
}

massive_umap_checkpoint_prepare <- function(plan, checkpoint,
                                            resume, checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    sidecar <- paste0(plan$output, ".checkpoint.rds")
    orphaned <- Sys.glob(paste0(plan$output, ".epoch_*.f32*"))
    if (!checkpoint) {
        if (file.exists(sidecar) || length(orphaned)) stop(
            "UMAP checkpoint files exist; use `resume = TRUE` ",
            "or inspect them before a new fit.",
            call. = FALSE)
        return(list(start_path = utils::tail(plan$source_paths, 1L),
            format = plan$init_format, epoch = 0L,
            positive = 0, negative = 0, every = 0L,
            edge_visits = 0, progress = NULL, sidecar = NULL))
    }
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L) stop(
        "`checkpoint_every` must be positive.", call. = FALSE)
    snapshots <- 2 + as.integer(identical(plan$backend, "cpu") &&
        identical(plan$parameters$layout_storage, "mmap"))
    additional <- if (resume) plan$layout_bytes else
        snapshots * plan$layout_bytes
    if (additional >
        0.8 * massive_disk_available_cpp(plan$output)) stop(
        "UMAP snapshots exceed the free-disk budget.", call. = FALSE)
    signature <- massive_checkpoint_signature(list(version = 2L,
        implementation = massive_checkpoint_implementation_identity(),
        output = plan$output, source = plan$source_identity,
        backend = plan$backend, parameters = plan$parameters,
        checkpoint_every = every))
    massive_umap_checkpoint_state(plan, sidecar, signature,
        resume, every)
}

massive_umap_checkpoint_state <- function(plan, sidecar,
        signature, resume, every) {
    if (resume) {
        saved <- massive_umap_checkpoint_load(plan, sidecar, signature)
    } else {
        if (file.exists(sidecar) || length(Sys.glob(paste0(
            plan$output, ".epoch_*.f32*")))) stop(
            "UMAP checkpoint files already exist.", call. = FALSE)
        saved <- list(epoch = 0L, snapshot = NULL,
            positive = 0, negative = 0, edge_visits = 0)
    }
    state <- new.env(parent = emptyenv())
    state$previous <- saved$snapshot
    progress <- function(epoch, snapshot, positive, negative,
            edge_visits = 0) {
        massive_umap_checkpoint_commit(plan, sidecar, signature,
            state, epoch, snapshot, positive, negative,
            edge_visits)
    }
    list(start_path = saved$snapshot %||%
            utils::tail(plan$source_paths, 1L),
        format = if (resume) "f32" else plan$init_format,
        epoch = as.integer(saved$epoch), positive = saved$positive,
        negative = saved$negative, edge_visits = saved$edge_visits,
        every = every,
        progress = progress, sidecar = sidecar,
        latest = function() state$previous)
}

massive_umap_checkpoint_load <- function(plan, sidecar, signature) {
    if (!file.exists(sidecar)) stop(
        "No UMAP checkpoint exists to resume.", call. = FALSE)
    saved <- readRDS(sidecar)
    valid_epoch <- is.numeric(saved$epoch) &&
        length(saved$epoch) == 1L &&
        is.finite(saved$epoch) && saved$epoch >= 1L &&
        saved$epoch <= plan$n_epochs &&
        saved$epoch == floor(saved$epoch)
    expected <- if (valid_epoch) paste0(plan$output,
        ".epoch_", saved$epoch, ".f32") else NULL
    valid_count <- function(value) is.numeric(value) &&
        length(value) == 1L && is.finite(value) &&
        value >= 0 && value == floor(value)
    if (!identical(saved$signature, signature) ||
        !valid_epoch ||
        !massive_checkpoint_same_paths(saved$snapshot, expected) ||
        !valid_count(saved$positive) ||
        !valid_count(saved$negative) ||
        !valid_count(saved$edge_visits) ||
        !file.exists(saved$snapshot) ||
        !identical(saved$snapshot_identity,
            massive_checkpoint_file_identity(saved$snapshot)) ||
        !identical(saved$snapshot_md5,
            unname(tools::md5sum(saved$snapshot))) ||
        file.info(saved$snapshot)$size != plan$layout_bytes) {
        stop("UMAP checkpoint does not match inputs or controls.",
            call. = FALSE)
    }
    snapshots <- normalizePath(Sys.glob(paste0(
        plan$output, ".epoch_*.f32*")), winslash = "/", mustWork = TRUE)
    if (length(setdiff(snapshots, saved$snapshot))) stop(
        "Uncommitted UMAP snapshot exists; inspect it before ",
        "resuming.", call. = FALSE)
    saved
}

massive_umap_checkpoint_commit <- function(plan, sidecar, signature,
        state, epoch, snapshot, positive, negative,
        edge_visits = 0) {
    snapshot <- normalizePath(snapshot, winslash = "/", mustWork = TRUE)
    expected <- paste0(plan$output, ".epoch_", epoch, ".f32")
    if (!massive_checkpoint_same_paths(snapshot, expected)) stop(
        "UMAP snapshot path does not match its epoch: ",
        snapshot, " != ", expected, call. = FALSE)
    if (!identical(plan$source_identity,
        massive_checkpoint_input_files_identity(plan$source_paths))) stop(
        "UMAP inputs changed before checkpoint.", call. = FALSE)
    identity <- massive_checkpoint_file_identity(snapshot)
    if (identity$bytes != plan$layout_bytes) stop(
        "UMAP snapshot size does not match layout.", call. = FALSE)
    hash <- unname(tools::md5sum(snapshot))
    if (length(hash) != 1L || is.na(hash)) stop(
        "Cannot hash UMAP snapshot.", call. = FALSE)
    saved <- list(signature = signature, epoch = epoch,
        snapshot = snapshot, snapshot_identity = identity,
        snapshot_md5 = hash,
        positive = positive, negative = negative,
        edge_visits = edge_visits)
    massive_checkpoint_write(saved, sidecar)
    if (!is.null(state$previous)) unlink(state$previous)
    state$previous <- snapshot
}

massive_umap_plan_layout <- function(graph, init, controls,
        memory_limit, backend, layout_storage) {
    bytes <- graph$n_vertices * init$ncol * 4
    limit <- massive_memory_bytes(memory_limit)
    capacity <- if (backend == "metal") as.integer(
        min(1e6, max(65536, graph$max_degree))) else NULL
    staged <- min(controls$chunk_rows, 8192L) * init$ncol * 4
    resident <- if (backend == "metal") {
        1.5 * bytes + capacity * 32 +
            if (layout_storage == "memory") bytes else staged
    } else if (layout_storage == "managed" ||
        (layout_storage == "memory" && backend == "cpu"))
        bytes else staged
    required <- 128 * 1024^2 + if (backend == "metal") resident
        else resident * 1.25
    if (!is.finite(bytes) || required > limit) stop(
        "UMAP layout exceeds `memory_limit`.", call. = FALSE)
    list(bytes = bytes, limit = limit, metal_capacity = capacity)
}

massive_umap_optimize_plan <- function(graph, init, output,
        n_epochs, negative_sample_rate, learning_rate, min_dist,
        repulsion_strength, seed, chunk_rows, memory_limit,
        backend, n.cores, layout_storage = "memory",
        resume = FALSE, devices = NULL) {
    massive_umap_validate_inputs(graph, init, backend, n.cores)
    devices <- massive_umap_plan_storage(devices, backend,
        graph, init, layout_storage)
    identity <- if (length(devices) > 1L)
        massive_cuda_device_uuids(devices) else NULL
    controls <- massive_umap_validate_controls(n_epochs, negative_sample_rate,
        learning_rate, min_dist, repulsion_strength, seed, chunk_rows)
    layout <- massive_umap_plan_layout(graph, init, controls,
        memory_limit, backend, layout_storage)
    layout_bytes <- layout$bytes
    limit <- layout$limit
    path <- massive_float_output(output, layout_bytes,
        resume = resume && layout_storage == "mmap")
    graph_paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path)
    massive_checkpoint_validated_graph_identity(graph, graph_paths)
    paths <- c(graph_paths, init$path)
    edge_capacity <- if (backend == "cuda") {
        massive_umap_cuda_capacity(graph, layout_bytes,
            controls$chunk_rows, limit, init$ncol, layout_storage,
            devices)
    } else layout$metal_capacity
    list(output = path, layout_bytes = layout_bytes, backend = backend,
        edge_capacity = edge_capacity, limit = limit,
        devices = devices, gpu_identity = identity,
        source_paths = paths, init_format = init$format,
        source_identity = massive_checkpoint_input_files_identity(paths),
        n_epochs = controls$n_epochs,
        negative_sample_rate = controls$negative_sample_rate,
        seed = controls$seed, chunk_rows = controls$chunk_rows,
        parameters = c(controls, list(learning_rate = learning_rate,
            min_dist = min_dist, repulsion_strength = repulsion_strength,
            n.cores = 1L, layout_storage = layout_storage),
            if (!is.null(devices)) list(devices = devices,
                gpu_identity = identity)))
}

massive_umap_plan_storage <- function(devices, backend, graph,
        init, layout_storage) {
    devices <- massive_validate_devices(devices, backend,
        graph$n_vertices)
    if (!is.null(devices) && (length(devices) > 4L ||
        layout_storage != "memory")) stop(
        "Coordinated CUDA UMAP supports at most four devices, ",
        "2D/3D output, and device storage.", call. = FALSE)
    if (length(devices) > 1L && graph$n_vertices <=
        (length(devices) - 1L) *
            ceiling(graph$n_vertices / length(devices))) stop(
        "Coordinated CUDA UMAP needs rows on every device.",
        call. = FALSE)
    if (!is.null(devices) && !massive_nccl_available_cpp()) stop(
        "Coordinated CUDA UMAP requires an NCCL-enabled build.",
        call. = FALSE)
    if (!layout_storage %in% c("memory", "mmap", "managed")) stop(
        "`layout_storage` must be 'memory', 'mmap', or 'managed'.",
        call. = FALSE)
    if (backend == "cuda" && layout_storage == "mmap") stop(
        "CUDA full-graph UMAP does not support mmap; ",
        "no CPU fallback was used.", call. = FALSE)
    if (backend == "cpu" && layout_storage == "managed") stop(
        "Managed layout storage requires CUDA; ",
        "no backend fallback was used.", call. = FALSE)
    if (backend == "metal" && layout_storage == "managed") stop(
        "Full-graph Metal UMAP does not support managed storage; ",
        "no backend fallback was used.", call. = FALSE)
    devices
}

massive_umap_cuda_layout_bytes <- function(vertices, dimensions,
        devices) {
    count <- max(1L, length(devices))
    ceiling(vertices / count) * count * dimensions * 4
}

massive_umap_cuda_capacity <- function(graph, layout_bytes,
                                        chunk_rows, limit,
                                        dimensions, layout_storage,
                                        devices = NULL) {
    free_bytes <- if (is.null(devices)) {
        massive_cuda_memory_cpp()$free_bytes
    } else massive_auto_cuda_free(devices, graph$n_vertices)
    device_layout <- if (layout_storage == "memory") {
        massive_umap_cuda_layout_bytes(graph$n_vertices,
            dimensions, devices) *
            if (is.null(devices)) 1 else 3
    } else 0
    available <- 0.65 * free_bytes - device_layout - 64 * 1024^2
    host_layout <- if (layout_storage == "managed") {
        layout_bytes
    } else 0
    host_batches <- max(1L, length(devices))
    capacity <- floor(min(graph$n_edges, 1e6, available / 16,
        (limit - 128 * 1024^2 - host_layout -
            min(chunk_rows, 8192L) * dimensions * 4) /
            (16 * host_batches)))
    if (!is.finite(capacity) || capacity < max(1, graph$max_degree)) {
        stop("CUDA layout or one graph row exceeds the RAM/VRAM ",
            "budget; no CPU fallback was used.", call. = FALSE)
    }
    as.integer(capacity)
}

massive_umap_validate_inputs <- function(graph, init, backend,
                                        n.cores) {
    if (!backend %in% c("cpu", "cuda", "metal") ||
        !identical(integer_scalar(n.cores), 1L)) {
        stop("EXPERIMENTAL full-graph UMAP needs one worker; ",
            "no backend fallback was used.", call. = FALSE)
    }
    if (!inherits(graph, "fastEmbedR_massive_graph") ||
        !identical(graph$storage, "csr") ||
        !isTRUE(graph$symmetrized) ||
        !identical(graph$weight_method, "umap_fuzzy_union")) {
        stop("`graph` must be a symmetric fuzzy massive UMAP CSR graph.",
            call. = FALSE)
    }
    if (!inherits(init, "fastEmbedR_massive_matrix") ||
        !init$format %in% c("f32", "fbin") ||
        init$nrow != graph$n_vertices ||
        !init$ncol %in% c(2, 3)) {
        stop("`init` must be file-backed 2D/3D coordinates ",
            "for every graph vertex.", call. = FALSE)
    }
    if (backend == "cuda" &&
        !isTRUE(embedding_cuda_available_cpp())) stop(
        "CUDA full-graph UMAP requires native CUDA; ",
        "no CPU fallback was used.", call. = FALSE)
    if (backend == "metal" &&
        (!isTRUE(embedding_metal_available_cpp()) ||
            init$ncol != 2L)) stop(
        "Metal full-graph UMAP needs native Metal and 2D output; ",
        "no CPU fallback was used.", call. = FALSE)
}

massive_umap_validate_controls <- function(n_epochs,
        negative_sample_rate, learning_rate, min_dist,
        repulsion_strength, seed, chunk_rows) {
    controls <- list(n_epochs = integer_scalar(n_epochs),
        negative_sample_rate = integer_scalar(negative_sample_rate),
        seed = integer_scalar(seed),
        chunk_rows = integer_scalar(chunk_rows))
    if (anyNA(unlist(controls)) || controls$n_epochs < 2L ||
        controls$negative_sample_rate < 0L ||
        controls$negative_sample_rate > 1000L ||
        controls$chunk_rows < 1L ||
        any(!is.finite(c(learning_rate, min_dist,
            repulsion_strength))) || learning_rate <= 0 ||
        min_dist < 0 || min_dist > 1 ||
        repulsion_strength <= 0) {
        stop("Invalid experimental UMAP optimizer controls.",
            call. = FALSE)
    }
    controls
}
