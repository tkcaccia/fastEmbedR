#' Optimize experimental disk-backed t-SNE affinities
#'
#' Streams a normalized symmetric CSR affinity graph on every iteration.
#' The CPU optimizer retains only its float32 coordinates and update state
#' in RAM, reuses the package's 2D or 3D FFT repulsion, and writes the
#' final coordinates to a file. The CUDA route streams the same affinities
#' through its 2D FFT optimizer and retains coordinates on the GPU.
#' This is a full-graph fit, not landmark projection. Optimization uses
#' fixed iteration counts, without automatic KL stopping. Disk reads can
#' be substantial. Both routes can checkpoint optimizer state.
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
#' @param memory_limit RAM budget, including coordinates, optimizer state,
#'   and a conservative FFT workspace allowance.
#' @param backend `"cpu"` or `"cuda"`; CUDA currently requires 2D output.
#' @param n.cores CPU worker count for FFT and coordinate updates.
#' @param checkpoint Save coordinates and optimizer state periodically.
#' @param resume Continue a matching interrupted fit.
#' @param checkpoint_every Iterations between state snapshots.
#' @return An experimental file-backed `fastEmbedR_massive_matrix`.
#'   CUDA results include host-call stage times. `attraction_sync`
#'   can include previously queued FFT work.
#' @export
massive_tsne_optimize <- function(graph, init, output,
        early_exaggeration_iter = 250L, n_iter = 500L,
        early_exaggeration = 12, exaggeration = 1,
        learning_rate = "auto", initial_momentum = 0.8,
        final_momentum = 0.8, min_gain = 0.01,
        max_step_norm = Inf, memory_limit = "8GB",
        backend = "cpu", n.cores = 1L, checkpoint = FALSE,
        resume = FALSE, checkpoint_every = 100L) {
    plan <- massive_tsne_optimize_plan(graph, init, output,
        early_exaggeration_iter, n_iter, early_exaggeration,
        exaggeration, learning_rate, initial_momentum,
        final_momentum, min_gain, max_step_norm, memory_limit,
        backend, n.cores)
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
    fit <- if (backend == "cuda") {
        do.call(massive_tsne_optimize_cuda_cpp, c(args,
            list(saved$iteration, saved$state_path, saved$every,
                saved$progress, plan$edge_capacity)))
    } else {
        do.call(massive_tsne_optimize_cpp, c(args,
            list(plan$workers, saved$iteration, saved$state_path,
                saved$every, saved$progress)))
    }
    if (!identical(plan$source_identity,
        massive_checkpoint_file_identity(plan$source_paths))) stop(
        "t-SNE graph or initialization changed during fitting.",
        call. = FALSE)
    result <- massive_tsne_result(plan, graph, init, fit, saved)
    if (!is.null(saved$sidecar))
        unlink(c(saved$latest(), saved$sidecar))
    result
}

massive_tsne_result <- function(plan, graph, init, fit, saved) {
    result <- massive_matrix(plan$output, graph$n_vertices, init$ncol)
    result$method <- "tsne"
    result$mode <- "out_of_core_graph"
    result$backend <- fit$backend_used
    result$optimizer <- fit$repulsion
    result$parameters <- plan$parameters
    result$parameters$checkpoint_every <- saved$every
    result$parameters$resumed_from_iteration <- saved$iteration
    result$resources <- list(estimated_peak_ram_bytes = plan$peak_ram,
        estimated_peak_vram_bytes = plan$peak_vram,
        graph_bytes = graph$n_edges * 8 +
            (graph$n_vertices + 1) * 8,
        checkpoint_disk_bytes = if (saved$every > 0L)
            2 * plan$state_bytes else 0)
    result$elapsed_seconds_current_call <- fit$elapsed_seconds
    if (!is.null(fit$stage_seconds))
        result$stage_seconds <- fit$stage_seconds
    result$fft_grid_size <- fit$fft_grid_size
    result$elapsed_seconds <- saved$prior_elapsed_seconds +
        fit$elapsed_seconds
    result$source_identity <- plan$source_identity
    result
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
        !valid || !identical(saved$state_path, expected) ||
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
        ".iter_*.state.f32*")), mustWork = TRUE)
    if (length(setdiff(snapshots, saved$state_path))) stop(
        "Uncommitted t-SNE snapshot exists; inspect it before ",
        "resuming.", call. = FALSE)
    saved
}

massive_tsne_checkpoint_commit <- function(plan, sidecar,
        signature, state, iteration, snapshot,
        elapsed_seconds) {
    snapshot <- normalizePath(snapshot, mustWork = TRUE)
    expected <- paste0(plan$output, ".iter_", iteration,
        ".state.f32")
    if (!identical(snapshot, expected)) stop(
        "t-SNE snapshot path does not match its iteration.",
        call. = FALSE)
    if (!identical(plan$source_identity,
        massive_checkpoint_file_identity(plan$source_paths))) stop(
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

massive_tsne_optimize_plan <- function(graph, init, output,
        early_iter, normal_iter, early_exag, exag, learning_rate,
        momentum, final_momentum, min_gain, max_step,
        memory_limit, backend, n.cores) {
    if (length(backend) != 1L || is.na(backend) ||
        !backend %in% c("cpu", "cuda"))
        stop("Full-graph t-SNE requires CPU or CUDA; ",
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
    if (backend == "cuda" && init$ncol != 2L) stop(
        "EXPERIMENTAL full-graph CUDA t-SNE supports 2D only; ",
        "no backend fallback was used.", call. = FALSE)
    if (backend == "cuda" &&
        !isTRUE(embedding_cuda_available_cpp())) stop(
        "Native CUDA t-SNE is unavailable; ",
        "no backend fallback was used.",
        call. = FALSE)
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
    resources <- massive_tsne_resource_plan(graph, init, path,
        memory_limit, backend)
    paths <- c(graph$offsets_path, graph$indices_path,
        graph$weights_path, init$path)
    if (!identical(graph$file_identity,
        massive_checkpoint_file_identity(paths[1:3]))) stop(
        "t-SNE affinity files changed.", call. = FALSE)
    c(list(output = path, source_paths = paths,
        source_identity = massive_checkpoint_file_identity(paths),
        n_vertices = graph$n_vertices, dimensions = init$ncol),
        resources, controls)
}

massive_tsne_resource_plan <- function(graph, init, path,
        memory_limit, backend) {
    limit <- massive_memory_bytes(memory_limit)
    coords <- graph$n_vertices * init$ncol
    if (coords > .Machine$integer.max) stop(
        "FFT coordinate count exceeds the native integer limit.",
        call. = FALSE)
    capacity <- if (backend == "cuda") {
        max(65536L, min(1000000L,
            floor((0.8 * limit - 64 * 1024^2) / 12)))
    } else NULL
    peak <- if (backend == "cuda") {
        64 * 1024^2 + capacity * 12
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
    grid <- if (backend == "cuda") {
        requested <- suppressWarnings(as.integer(
            Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", "")))
        if (!is.na(requested) && requested %in%
            c(32L, 64L, 128L, 256L, 512L, 1024L)) requested
        else if (graph$n_vertices < 20000) 256L
        else if (graph$n_vertices >= 100000) 1024L
        else 512L
    } else NULL
    vram <- if (backend == "cuda") {
        coords * 16 + graph$n_vertices * 8 +
            9 * (2 * grid)^2 * 8 + capacity * 12 +
            64 * 1024^2
    } else NA_real_
    if (backend == "cuda" &&
        vram > 0.8 * massive_cuda_memory_cpp()$free_bytes) stop(
        "CUDA t-SNE layout and FFT workspace exceed free VRAM.",
        call. = FALSE)
    list(peak_ram = peak, peak_vram = vram,
        edge_capacity = capacity, state_bytes = coords * 12)
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
