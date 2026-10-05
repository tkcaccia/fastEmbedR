massive_cluster_reference_budget <- function(rows, cols, k,
        backend, workers, memory_limit, devices, method = "leiden") {
    limit <- massive_memory_bytes(memory_limit)
    per_landmark <- 64 * cols + 512 * k + 2048 + 8 * workers
    available <- max(0, 0.7 * limit - 168 * 1024^2)
    ram_cap <- if (method == "walktrap") {
        floor((sqrt(per_landmark^2 + 64 * available) -
            per_landmark) / 32)
    } else floor(available / per_landmark)
    vram_cap <- Inf
    if (backend == "cuda") {
        free <- massive_auto_cuda_free(devices, rows)
        vram_cap <- floor((0.6 * free - 128 * 1024^2) /
            (16 * cols + 256 * k + 512))
    }
    cap <- min(ram_cap, vram_cap, floor(rows / 2),
        .Machine$integer.max)
    if (method == "walktrap") cap <- min(cap, 4000L)
    list(cap = cap, limit = limit,
        per_landmark = per_landmark,
        free_vram_bytes = if (backend == "cuda") free else NA_real_)
}

massive_cluster_auto_plan <- function(x, landmarks, k, backend,
        workers, memory_limit, mode, output, devices,
        method = "leiden") {
    source <- inherits(x, "fastEmbedR_massive_matrix")
    if (!source && !is.matrix(x) && !is_float32_matrix(x)) {
        stop("EXPERIMENTAL clustering needs a matrix source.",
            call. = FALSE)
    }
    rows <- if (source) x$nrow else nrow(x)
    cols <- if (source) x$ncol else ncol(x)
    if (!is.null(devices) && mode != "landmark") stop(
        "`devices` requires landmark clustering.", call. = FALSE)
    devices <- massive_validate_devices(devices, backend, rows)
    if (is.na(k) || k < 1L || k >= rows) stop(
        "`k` must be smaller than the source row count.",
        call. = FALSE)
    if (mode == "auto") memory_limit <- massive_auto_memory_limit(memory_limit)
    budget <- massive_cluster_reference_budget(rows, cols, k,
        backend, workers, memory_limit, devices, method)
    normal <- 256 * 1024^2 + rows *
        (32 * cols + 512 * k + 2048 + 8 * workers)
    if (method == "walktrap") normal <- normal + 16 * rows^2
    gpu_ok <- backend != "cuda" ||
        128 * 1024^2 + rows * (16 * cols + 256 * k + 512) <=
        0.5 * budget$free_vram_bytes
    if (mode == "auto" && is.null(landmarks) &&
        rows <= .Machine$integer.max &&
        (method != "walktrap" || rows <= 4000L) &&
        normal <= 0.5 * budget$limit && gpu_ok) {
        if (!is.null(output)) stop(
            "Auto clustering fits in memory; remove `output`.",
            call. = FALSE)
        return(list(mode = "in_memory", estimate = normal))
    }
    if (!source || x$format == "memory") stop(
        "Landmark clustering needs a file-backed massive_matrix().",
        call. = FALSE)
    count <- if (is.null(landmarks)) min(budget$cap, 1e6) else
        integer_scalar(landmarks)
    if (is.na(count) || !is.finite(count) || count <= k ||
        count >= rows || count > budget$cap) stop(
        "Landmark count exceeds the RAM/VRAM or Walktrap ",
        "4,000-vertex budget, or cannot support `k`.",
        call. = FALSE)
    if (is.null(output)) stop(
        "Landmark clustering needs an output prefix.", call. = FALSE)
    if (mode == "auto") message("EXPERIMENTAL auto: landmark clustering")
    list(mode = "landmark", count = as.integer(count),
        budget = budget, estimate = normal, devices = devices)
}

massive_cluster_reuse <- function(x, nn, landmarks, k,
        backend, mode) {
    if (is.null(nn)) return(NULL)
    if (mode != "landmark") stop(
        "`nn` requires explicit landmark clustering.", call. = FALSE)
    if (!inherits(nn, "fastEmbedR_massive_projection") ||
        is.null(nn$selection$indices)) stop(
        "`nn` must be a previous massive landmark embedding.",
        call. = FALSE)
    count <- integer_scalar(landmarks %||%
        length(nn$selection$indices))
    if (is.na(count)) stop("Invalid reused landmark count.",
        call. = FALSE)
    massive_embedding_reused_graph(x, nn, count, k, backend)
}

massive_cluster_output_plan <- function(x, count, k, output,
        resume = FALSE, checkpoint = FALSE, reuse = FALSE) {
    paths <- massive_cluster_paths(output, x$nrow * 8, resume)
    prefix <- sub("\\.clusters\\.u32$", "", paths[["membership"]])
    landmark_path <- paste0(prefix, ".landmarks.f32")
    knn_prefix <- paste0(prefix, ".knn")
    if (!resume && !reuse) {
        massive_float_output(landmark_path, count * x$ncol * 4)
        massive_knn_paths(knn_prefix, x$nrow * k * 8)
    }
    disk <- x$nrow * 8
    if (!reuse) disk <- disk + count * x$ncol * 4 +
        x$nrow * 8 * k
    if (checkpoint) disk <- disk + 64 * count * k
    if (resume) {
        files <- c(landmark_path, paste0(knn_prefix,
            c(".indices.u32", ".distances.f32")),
            paste0(unname(paths), ".part"),
            paste0(prefix, ".workflow.checkpoint.rds"))
        files <- files[file.exists(files)]
        sizes <- file.info(files)$size
        if (anyNA(sizes)) stop("Cannot inspect cluster resume files.",
            call. = FALSE)
        disk <- max(0, disk - sum(sizes))
    }
    if (disk > 0.8 * massive_disk_available_cpp(paths[[1L]])) {
        stop("Cluster workflow exceeds the free-disk budget.",
            call. = FALSE)
    }
    list(prefix = prefix, landmarks = landmark_path,
        knn = knn_prefix, disk_bytes = disk)
}

massive_cluster_workflow <- function(x, plan, paths, k, method,
        backend, workers, landmark_method, knn_method,
        chunk_rows, memory_limit,
        resolution, n_iterations, n_runs, steps, seed,
        checkpoint, checkpoint_every, resume, reused) {
    started <- proc.time()[[3L]]
    estimates <- massive_cluster_workflow_report(x, plan, paths,
        backend, method)
    if (!is.null(plan$devices)) {
        original <- massive_cuda_memory_cpp()$device
        on.exit(massive_cuda_select_cpp(original))
        massive_cuda_select_cpp(plan$devices[1L])
    }
    controls <- massive_cluster_workflow_controls(k, method, backend,
        workers, plan$devices, landmark_method, knn_method,
        chunk_rows, memory_limit, resolution, n_iterations,
        n_runs, steps, seed, checkpoint_every)
    workflow <- massive_cluster_workflow_checkpoint(x, plan, paths,
        controls, checkpoint, resume, reused)
    prepared <- massive_cluster_precompute(x, plan, paths, k, backend,
        workers, landmark_method, knn_method, chunk_rows,
        memory_limit, seed, workflow, checkpoint, resume, reused)
    low_sidecar <- paste0(paths$prefix, ".checkpoint.rds")
    result <- massive_cluster_landmarks(prepared$graph, prepared$knn,
        paths$prefix, method = method, selection = prepared$selection,
        backend = backend,
        n.cores = workers, chunk_rows = chunk_rows,
        memory_limit = memory_limit, resolution = resolution,
        n_iterations = n_iterations, n_runs = n_runs,
        steps = steps, seed = seed, checkpoint = checkpoint,
        checkpoint_every = checkpoint_every,
        resume = resume && file.exists(low_sidecar))
    if (!is.null(prepared$workflow)) unlink(prepared$workflow$path)
    result$mode <- "landmark"
    result$selection <- prepared$selection
    result$knn <- prepared$knn
    result$graph_reused <- !is.null(reused)
    result$gpu_devices <- plan$devices
    result$reference_graph_parameters <- prepared$graph$parameters
    result$workflow_elapsed_sec <- unname(proc.time()[[3L]] - started)
    result$resources$workflow_disk_estimate_bytes <- paths$disk_bytes
    result$resources$reference_ram_estimate_bytes <-
        estimates$reference_ram_estimate_bytes
    if (backend == "cuda")
        result$resources$free_vram_preflight_bytes <-
            plan$budget$free_vram_bytes
    result
}

massive_cluster_in_memory <- function(x, k, method, backend,
        workers, resolution, n_iterations, n_runs, steps,
        seed, estimate) {
    message("EXPERIMENTAL auto clustering selected in-memory graph")
    data <- if (inherits(x, "fastEmbedR_massive_matrix"))
        massive_auto_materialize(x) else x
    graph <- knn_graph(data, k, backend = backend,
        n.cores = workers)
    result <- graph_cluster(graph, method, backend, resolution,
        n_iterations, n_runs, steps, seed)
    result$massive_auto <- "in_memory"
    result$massive_auto_estimated_ram_bytes <- estimate
    result$mode <- "in_memory"
    result$experimental <- TRUE
    result
}

massive_cluster_full_graph <- function(x, method, backend, output,
        n.cores, chunk_rows, memory_limit, resolution,
        n_iterations, steps, seed, unused, checkpoint, resume, initial) {
    if (any(unused)) stop(
        "Landmark and KNN controls do not apply to a full graph.",
        call. = FALSE)
    if (normalize_nn_threads(n.cores) != 1L) stop(
        "EXPERIMENTAL full-graph clustering uses one CPU worker.",
        call. = FALSE)
    result <- if (identical(method, "walktrap")) {
        if (!is.null(initial) || checkpoint || resume ||
            !isTRUE(all.equal(resolution, 1)) ||
            n_iterations != 10L) stop(
            "Partitioned Walktrap has no initialization or checkpoint; ",
            "resolution and iteration controls do not apply.",
            call. = FALSE)
        massive_walktrap(x, output, steps = steps,
            chunk_rows = chunk_rows %||% 4000L,
            memory_limit = memory_limit, backend = backend)
    } else if (identical(method, "leiden")) {
        massive_leiden(x, output, max_passes = n_iterations,
            resolution = resolution, seed = seed,
            chunk_rows = chunk_rows %||% 8192L,
            memory_limit = memory_limit, backend = backend,
            checkpoint = checkpoint, resume = resume,
            initial = initial)
    } else {
        massive_louvain(x, output, max_passes = n_iterations,
            resolution = resolution, seed = seed,
            chunk_rows = chunk_rows %||% 8192L,
            memory_limit = memory_limit, backend = backend,
            checkpoint = checkpoint, resume = resume,
            initial = initial)
    }
    result$mode <- "out_of_core_graph"
    result
}

massive_cluster_full_request <- function(x, method, backend, output,
        n.cores, chunk_rows, memory_limit, resolution,
        n_iterations, steps, seed, checkpoint, resume, checkpoint_every,
        unused, initial) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (!identical(checkpoint_every, 100L)) stop(
        "`checkpoint_every` applies only to landmark labels.",
        call. = FALSE)
    massive_cluster_full_graph(x, method, backend, output,
        n.cores, chunk_rows, memory_limit, resolution,
        n_iterations, steps, seed, unused, checkpoint, resume, initial)
}

massive_cluster_auto_result <- function(result, massive, plan) {
    if (massive == "auto") {
        result$massive_auto <- "landmark"
        result$massive_auto_estimated_ram_bytes <- plan$estimate
    }
    result
}

#' EXPERIMENTAL massive-data community clustering
#'
#' Selects bounded resident landmarks from a file-backed matrix, builds their
#' graph with `knn_graph()`, and runs the existing native Walktrap, Louvain,
#' or Leiden implementation. Query-to-landmark neighbors and weighted-vote
#' labels remain file-backed. This is a landmark approximation, not clustering
#' of the complete graph. With a symmetric file-backed fuzzy graph and
#' `massive = "out_of_core_graph"`, the function runs CPU multilevel
#' Louvain or Leiden on the file-backed graph. Full-graph Leiden runs one
#' hierarchy with `n_iterations` local-move passes per level. Full-graph
#' Walktrap runs exact walks within bounded blocks and contracts communities;
#' it is an explicit approximation when more than one block is needed.
#' `massive = "auto"` uses
#' ordinary in-memory graph clustering when resource estimates permit it.
#' Exact Walktrap is CPU-only and limited to 4,000 landmark vertices;
#' automatic planning includes its quadratic transition-matrix memory.
#'
#' @param x A `massive_matrix()` descriptor, a symmetric file-backed fuzzy
#'   graph in full-graph mode, or a resident matrix in auto mode.
#' @param method `"leiden"`, `"louvain"`, or `"walktrap"` (random walks).
#' @param massive `"landmark"`, `"auto"`, or `"out_of_core_graph"`.
#' @param landmarks Landmark count; auto mode chooses a bounded count if NULL.
#' @param nn Previous massive landmark UMAP or t-SNE result whose verified
#'   selection and query KNN should be reused instead of recomputed.
#' @param k Number of non-self neighbors in the landmark graph and projection.
#' @param output New prefix for landmark files, or a new `.clusters.u32`
#'   label-file path in full-graph mode.
#' @param backend `"cpu"` or `"cuda"`; full-graph clustering is CPU-only.
#' @param devices Optional CUDA device indices for landmark clustering.
#'   The first builds the reference graph; all search disjoint query shards.
#' @param n.cores CPU worker count.
#' @param landmark_method `"reservoir"` or `"random"` landmark selection.
#' @param knn_method Query-to-landmark nearest-neighbor search method.
#' @param chunk_rows Maximum source rows per streamed batch.
#' @param memory_limit Conservative RAM budget; CUDA also checks free VRAM.
#' @param resolution,n_iterations,n_runs,steps,seed Graph clustering controls.
#' @param checkpoint,resume Save or reuse completed landmark workflow stages,
#'   label batches, or completed full-graph Louvain/Leiden stages. Full-graph
#'   Walktrap does not yet support checkpoints.
#' @param checkpoint_every Landmark label batches between checkpoints.
#' @param initial Optional saved landmark-clustering result used to
#'   initialize full-graph Louvain or Leiden. Other modes reject it.
#' @return An experimental file-backed clustering result, a file-backed
#'   Walktrap, Louvain, or Leiden result in full-graph mode, or an in-memory
#'   `graph_cluster()` result when auto mode selects the resident route.
#' @export
massive_cluster <- function(x, method = c("leiden", "louvain", "walktrap"),
        massive = c("landmark", "auto", "out_of_core_graph"), landmarks = NULL,
        k = 30L, output = NULL, backend = "cpu", n.cores = 1L,
        devices = NULL, landmark_method = c("reservoir", "random"),
        knn_method = c("auto", "exact", "hnsw", "ivf"),
        chunk_rows = NULL, memory_limit = "8GB", resolution = 1,
        n_iterations = 10L, n_runs = 1L, steps = 4L, seed = 1L,
        checkpoint = FALSE, resume = FALSE, checkpoint_every = 100L,
        initial = NULL, nn = NULL) {
    method <- match.arg(method)
    massive <- match.arg(massive)
    if (massive == "out_of_core_graph") {
        unused <- c(!missing(landmarks), !missing(k), !is.null(nn),
            !missing(landmark_method), !missing(knn_method), !missing(n_runs),
            !missing(steps) && method != "walktrap", !is.null(devices))
        return(massive_cluster_full_request(x, method, backend, output,
            n.cores, chunk_rows, memory_limit, resolution, n_iterations,
            steps, seed, checkpoint, resume, checkpoint_every, unused,
            initial))
    }
    if (!is.null(initial)) stop("`initial` requires out_of_core_graph.")
    checkpoint_every <- massive_cluster_checkpoint_controls(
        checkpoint, resume, checkpoint_every)
    backend <- massive_cluster_backend(method, backend)
    if (backend == "cuda" && !native_cuda_knn_available_cpp())
        stop("CUDA KNN unavailable; no CPU fallback.", call. = FALSE)
    workers <- normalize_nn_threads(n.cores)
    landmark_method <- match.arg(landmark_method)
    knn_method <- match.arg(knn_method)
    k <- integer_scalar(k)
    graph_cluster_settings(method, backend, resolution, n_iterations,
        n_runs, steps, seed)
    reused <- massive_cluster_reuse(x, nn, landmarks, k, backend, massive)
    if (!is.null(reused)) landmarks <- length(reused$selection$indices)
    plan <- massive_cluster_auto_plan(x, landmarks, k, backend, workers,
        memory_limit, massive, output, devices, method)
    if (plan$mode == "in_memory") {
        if (checkpoint || resume) stop("In-memory mode cannot checkpoint.")
        return(massive_cluster_in_memory(x, k, method, backend, workers,
            resolution, n_iterations, n_runs, steps, seed, plan$estimate))
    }
    massive_knn_method(knn_method, backend, plan$count, k)
    paths <- massive_cluster_output_plan(x, plan$count, k, output,
        resume, checkpoint, !is.null(reused))
    result <- massive_cluster_workflow(x, plan, paths, k, method,
        backend, workers, landmark_method, knn_method, chunk_rows,
        memory_limit, resolution, n_iterations, n_runs, steps, seed,
        checkpoint, checkpoint_every, resume, reused)
    massive_cluster_auto_result(result, massive, plan)
}
