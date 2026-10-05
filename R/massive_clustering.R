massive_cluster_cuda_resources <- function(reference_graph) {
    info <- massive_cuda_memory_cpp()
    required <- 128 * 1024^2 +
        128 * length(reference_graph$weight) +
        256 * reference_graph$n_vertices
    if (required > 0.65 * info$free_bytes) {
        stop("CUDA landmark graph exceeds the free-VRAM budget.",
            call. = FALSE)
    }
    list(peak_vram_bytes = required,
        free_vram_bytes = info$free_bytes, gpu_device = info$device)
}

massive_cluster_resources <- function(graph, reference_graph, workers,
        chunk_rows, memory_limit, backend = "cpu", method = "leiden") {
    limit <- massive_memory_bytes(memory_limit)
    if (method == "walktrap" &&
        reference_graph$n_vertices > 4000L) {
        stop("Exact Walktrap supports at most 4,000 landmarks.",
            call. = FALSE)
    }
    # Include R edges and native sorting, adjacency, and contraction buffers.
    graph_bytes <- 128 * length(reference_graph$weight) +
        256 * reference_graph$n_vertices
    fixed <- 64 * 1024^2 + graph_bytes +
        8 * graph$n_reference * workers
    if (method == "walktrap") {
        fixed <- fixed + 16 * reference_graph$n_vertices^2
    }
    per_row <- 20 * graph$ncol + 24
    maximum <- floor((0.7 * limit - fixed) / per_row)
    maximum <- min(maximum, floor(128e6 / (20 * graph$ncol)))
    requested <- chunk_rows %||% 250000L
    if (!is.numeric(requested) || length(requested) != 1L ||
        !is.finite(requested) || requested < 1 ||
        requested != floor(requested) ||
        requested > .Machine$integer.max) {
        stop("`chunk_rows` must be one positive integer.", call. = FALSE)
    }
    if (!is.finite(maximum) || maximum < 1) {
        stop("Cluster buffers exceed `memory_limit`.", call. = FALSE)
    }
    rows <- as.integer(min(requested, maximum))
    result <- list(chunk_rows = rows,
        peak_ram_bytes = fixed + rows * per_row,
        reference_graph_estimate_bytes = graph_bytes,
        memory_limit_bytes = limit, output_bytes = graph$nrow * 8)
    if (backend == "cuda") {
        result <- c(result, massive_cluster_cuda_resources(
            reference_graph))
    }
    result
}

massive_cluster_paths <- function(output, bytes, resume = FALSE) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) {
        stop("`output` must be one file prefix.", call. = FALSE)
    }
    parent <- normalizePath(dirname(output), mustWork = TRUE)
    prefix <- file.path(parent, basename(output))
    paths <- c(membership = paste0(prefix, ".clusters.u32"),
        confidence = paste0(prefix, ".confidence.f32"))
    parts <- paste0(paths, ".part")
    part_exists <- file.exists(parts)
    checkpoint <- paste0(prefix, ".checkpoint.rds")
    if (any(file.exists(paths)) ||
        (!resume && file.exists(checkpoint)) ||
        (resume && any(part_exists) && !all(part_exists)) ||
        (!resume && any(part_exists))) {
        stop("Cluster output or partial files conflict with `resume`.",
            call. = FALSE)
    }
    written <- if (resume && all(part_exists)) {
        sum(file.info(parts)$size)
    } else 0
    remaining <- bytes - written
    if (!is.finite(remaining) || remaining < 0 ||
        remaining > 0.9 * massive_disk_available_cpp(paths[[1L]])) {
        stop("Cluster output exceeds the free-disk budget.",
            call. = FALSE)
    }
    paths
}

massive_cluster_checkpoint <- function(graph, knn, reference_rows,
        paths, controls, resume) {
    path <- sub("\\.clusters\\.u32$", ".checkpoint.rds",
        paths[["membership"]])
    signature <- massive_checkpoint_signature(list(
        version = 1L,
        graph = graph[c("from", "to", "weight", "n_vertices")],
        knn = list(nrow = knn$nrow, ncol = knn$ncol,
            n_reference = knn$n_reference,
            indices = massive_checkpoint_file_identity(
                knn$indices_path),
            distances = massive_checkpoint_file_identity(
                knn$distances_path)),
        reference_rows = reference_rows, controls = controls))
    if (!resume) {
        if (file.exists(path)) {
            stop("Cluster checkpoint exists; use `resume = TRUE`.",
                call. = FALSE)
        }
        return(list(path = path, signature = signature, state = NULL))
    }
    state <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(state) || !is.list(state$clustered)) {
        stop("Cluster checkpoint is unreadable.", call. = FALSE)
    }
    parts <- paste0(paths, ".part")
    exists <- file.exists(parts)
    sizes <- if (all(exists)) file.info(parts)$size else c(0, 0)
    rows <- state$completed_rows
    labels <- state$clustered$membership
    communities <- state$clustered$n_communities
    if (!identical(state$signature, signature) ||
        !is.numeric(rows) || length(rows) != 1L ||
        !is.finite(rows) || rows != floor(rows) ||
        rows < 0 || rows > knn$nrow ||
        (rows > 0 && !all(exists)) ||
        anyNA(sizes) || any(sizes < rows * 4) ||
        any(sizes > knn$nrow * 4) ||
        !is.integer(labels) ||
        length(labels) != knn$n_reference || anyNA(labels) ||
        any(labels < 1) ||
        !is.numeric(communities) || length(communities) != 1L ||
        !is.finite(communities) || communities < 1 ||
        any(labels > communities)) {
        stop("Cluster checkpoint or partial output does not match ",
            "the current inputs.", call. = FALSE)
    }
    list(path = path, signature = signature, state = state)
}

massive_cluster_save_checkpoint <- function(checkpoint, clustered) {
    checkpoint$state <- list(signature = checkpoint$signature,
        completed_rows = 0, clustered = clustered)
    massive_checkpoint_write(checkpoint$state, checkpoint$path)
    checkpoint
}

massive_cluster_fit <- function(graph, controls, checkpoint, resume) {
    if (resume) return(checkpoint$state$clustered)
    graph_cluster(graph, method = controls$method,
        backend = controls$backend,
        resolution = controls$resolution,
        n_iterations = controls$n_iterations,
        n_runs = controls$n_runs, steps = controls$steps,
        seed = controls$seed)
}

massive_cluster_result <- function(knn, clustered, paths, controls,
        reference_rows, resources, checkpoint, started) {
    result <- list(membership_path = paths[["membership"]],
        membership_identity = massive_checkpoint_file_identity(
            paths[["membership"]]),
        confidence_path = paths[["confidence"]], nrow = knn$nrow,
        n_reference = knn$n_reference,
        n_communities = clustered$n_communities,
        reference_membership = clustered$membership,
        reference_modularity = clustered$modularity,
        method = controls$method, backend = controls$backend,
        assignment_backend = "cpu",
        graph_backend = knn$backend,
        n.cores = controls$workers,
        reference_rows_preserved = !is.null(reference_rows),
        resources = resources, checkpoint = checkpoint,
        elapsed_sec = unname(proc.time()[[3L]] - started),
        experimental = TRUE)
    class(result) <- "fastEmbedR_massive_clusters"
    result
}

massive_cluster_initial <- function(graph, initial) {
    if (is.null(initial)) return(NULL)
    if (!inherits(initial, "fastEmbedR_massive_clusters")) stop(
        "`initial` must be saved landmark clusters.", call. = FALSE)
    count <- initial$n_communities
    if (!is.numeric(initial$nrow) || length(initial$nrow) != 1L ||
        !is.finite(initial$nrow) ||
        initial$nrow != graph$n_vertices ||
        !is.numeric(count) || length(count) != 1L ||
        !is.finite(count) || count < 1 || count != floor(count) ||
        count > min(graph$n_vertices, .Machine$integer.max)) stop(
        "Initial landmark clusters do not match graph vertices.",
        call. = FALSE)
    identity <- tryCatch(massive_checkpoint_file_identity(
        initial$membership_path), error = function(e) NULL)
    if (is.null(identity) ||
        !identical(identity, initial$membership_identity) ||
        !identical(identity$bytes, 4 * graph$n_vertices)) stop(
        "Initial landmark label file changed or has wrong size.",
        call. = FALSE)
    list(parent_mapping_path = initial$membership_path,
        parent_mapping_identity = identity, n_parent = as.integer(count),
        n_refined = graph$n_vertices)
}

massive_cluster_finish <- function(membership, confidence, parts,
        paths, rows) {
    truncate(membership)
    truncate(confidence)
    close(membership)
    close(confidence)
    if (any(file.info(parts)$size != rows * 4)) {
        stop("Cluster output size mismatch; .part retained.",
            call. = FALSE)
    }
    if (!file.rename(parts[["membership"]], paths[["membership"]])) {
        stop("Could not finalize cluster labels; .part retained.",
            call. = FALSE)
    }
    if (!file.rename(parts[["confidence"]], paths[["confidence"]])) {
        if (!file.rename(paths[["membership"]],
            parts[["membership"]])) {
            stop("Could not restore cluster labels after rename failure.",
                call. = FALSE)
        }
        stop("Could not finalize cluster confidence; .part retained.",
            call. = FALSE)
    }
}

massive_cluster_batch <- function(graph, first, count,
        labels, workers, reference_rows) {
    knn <- massive_read_knn_rows(graph, first, count)
    assigned <- massive_vote_landmarks_cpp(
        knn$indices, knn$distances, labels, workers)
    if (!is.null(reference_rows)) {
        low <- findInterval(first - 1, reference_rows) + 1L
        high <- findInterval(first + count - 1, reference_rows)
        if (low <= high) {
            selected <- seq.int(low, high)
            rows <- as.integer(reference_rows[selected] - first + 1)
            assigned$membership[rows] <- labels[selected]
            assigned$confidence[rows] <- 1
        }
    }
    assigned
}

massive_cluster_stream <- function(graph, labels, paths, resources,
        workers, reference_rows, checkpoint, checkpoint_every) {
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    saved <- checkpoint$state
    completed <- saved$completed_rows %||% 0
    mode <- if (completed > 0) "r+b" else "wb"
    membership <- file(parts[["membership"]], mode)
    on.exit(try(close(membership), silent = TRUE))
    confidence <- file(parts[["confidence"]], mode)
    on.exit(try(close(confidence), silent = TRUE), add = TRUE)
    if (completed > 0) {
        seek(membership, completed * 4, rw = "write")
        seek(confidence, completed * 4, rw = "write")
    }
    first <- completed + 1
    reported <- -1L
    batches <- 0L
    while (first <= graph$nrow) {
        count <- as.integer(min(resources$chunk_rows,
            graph$nrow - first + 1))
        assigned <- massive_cluster_batch(graph, first, count,
            labels, workers, reference_rows)
        writeBin(assigned$membership, membership, size = 4L,
            endian = "little")
        writeBin(assigned$confidence, confidence, size = 4L,
            endian = "little")
        first <- first + count
        batches <- batches + 1L
        if (!is.null(saved) && (batches %% checkpoint_every == 0L ||
            first > graph$nrow)) {
            flush(membership)
            flush(confidence)
            saved$completed_rows <- first - 1
            massive_checkpoint_write(saved, checkpoint$path)
        }
        progress <- floor(20 * (first - 1) / graph$nrow)
        if (progress > reported) {
            message("EXPERIMENTAL landmark clustering: ",
                first - 1, "/", graph$nrow, " rows")
            reported <- progress
        }
    }
    massive_cluster_finish(membership, confidence, parts, paths,
        graph$nrow)
    if (!is.null(saved)) unlink(checkpoint$path)
}

massive_cluster_backend <- function(method, backend) {
    if (!is.character(backend) || length(backend) != 1L ||
        is.na(backend) || !backend %in% c("cpu", "cuda")) {
        stop("EXPERIMENTAL clustering requires CPU or CUDA.",
            call. = FALSE)
    }
    if (backend == "cuda" && method == "walktrap") {
        stop("Walktrap is CPU-only; no CUDA fallback was used.",
            call. = FALSE)
    }
    if (backend == "cuda" &&
        (!isTRUE(graph_clustering_cuda_available_cpp()) ||
            !isTRUE(embedding_cuda_available_cpp()))) {
        stop("CUDA graph clustering is unavailable; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    backend
}

#' EXPERIMENTAL file-backed landmark graph clustering
#'
#' Runs the existing native Leiden, Louvain, or Walktrap implementation on
#' a resident landmark graph. Saved CPU- or CUDA-built KNN rows are streamed
#' through inverse-distance weighted community voting. A supplied landmark
#' selection preserves reference labels at their original source rows.
#'
#' @param reference_graph A `knn_graph()` result for the landmark rows.
#' @param knn Saved CPU- or CUDA-built query-to-landmark graph from
#'   `massive_landmark_knn()`.
#' @param output New prefix for label and confidence output files.
#' @param method `"leiden"`, `"louvain"`, or `"walktrap"`.
#' @param selection Optional result of `massive_select_landmarks()` from the
#'   query source; selected rows retain their fitted community labels.
#' @param backend `"cpu"` for all methods, or `"cuda"` for landmark-graph
#'   Louvain and Leiden. Streamed community voting remains on CPU. A CUDA
#'   request fails explicitly if the backend is unavailable.
#' @param n.cores CPU workers used for weighted community voting.
#' @param chunk_rows Maximum query rows per batch.
#' @param memory_limit Conservative RAM budget for voting buffers.
#' @param resolution Modularity resolution for Leiden or Louvain.
#' @param n_iterations Maximum local-moving passes.
#' @param n_runs Independent seeded graph-clustering runs.
#' @param steps Walktrap random-walk length.
#' @param seed Random seed for Leiden or Louvain.
#' @param checkpoint Save landmark communities and completed output rows.
#' @param checkpoint_every Save progress after this many query batches.
#' @param resume Continue from matching partial outputs and checkpoint.
#' @return A lightweight `fastEmbedR_massive_clusters` descriptor with
#'   one-based `uint32` labels, float32 confidence, and landmark metadata.
#' @export
massive_cluster_landmarks <- function(
    reference_graph, knn, output,
    method = c("leiden", "louvain", "walktrap"),
    selection = NULL, backend = "cpu", n.cores = 1L,
    chunk_rows = NULL, memory_limit = "8GB",
    resolution = 1, n_iterations = 10L, n_runs = 1L,
    steps = 4L, seed = 1L, checkpoint = FALSE,
    checkpoint_every = 100L, resume = FALSE
) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    checkpoint_every <- integer_scalar(checkpoint_every)
    if (is.na(checkpoint_every) || checkpoint_every < 1L) {
        stop("`checkpoint_every` must be a positive integer.",
            call. = FALSE)
    }
    if (!inherits(knn, "fastEmbedR_massive_knn") ||
        !knn$backend %in% c("cpu", "cuda")) {
        stop("EXPERIMENTAL clustering requires a saved CPU or CUDA KNN.",
            call. = FALSE)
    }
    reference_graph <- validate_fastembedr_graph(reference_graph)
    if (reference_graph$n_vertices != knn$n_reference) {
        stop("Landmark graph and KNN reference sizes differ.",
            call. = FALSE)
    }
    method <- match.arg(method)
    backend <- massive_cluster_backend(method, backend)
    workers <- normalize_nn_threads(n.cores)
    reference_rows <- massive_reference_indices(selection, knn)
    resources <- massive_cluster_resources(knn, reference_graph,
        workers, chunk_rows, memory_limit, backend, method)
    paths <- massive_cluster_paths(output, resources$output_bytes, resume)
    controls <- list(method = method, backend = backend, workers = workers,
        chunk_rows = resources$chunk_rows, resolution = resolution,
        n_iterations = n_iterations, n_runs = n_runs,
        steps = steps, seed = seed)
    saved <- if (checkpoint) massive_cluster_checkpoint(
        reference_graph, knn, reference_rows, paths, controls,
        resume) else NULL
    started <- proc.time()[[3L]]
    clustered <- massive_cluster_fit(reference_graph, controls,
        saved, resume)
    if (checkpoint && !resume)
        saved <- massive_cluster_save_checkpoint(saved, clustered)
    massive_cluster_stream(knn, clustered$membership,
        paths, resources, workers, reference_rows, saved,
        checkpoint_every)
    massive_cluster_result(knn, clustered, paths, controls,
        reference_rows, resources, checkpoint, started)
}

#' Read bounded rows from an experimental clustering result
#'
#' @param x A `fastEmbedR_massive_clusters` descriptor.
#' @param first First one-based query row.
#' @param n Number of consecutive rows to read.
#' @return A list with `membership` and `confidence` vectors.
#' @export
massive_read_cluster_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_clusters")) {
        stop("`x` must be a massive_cluster_landmarks() result.",
            call. = FALSE)
    }
    first <- as.numeric(first)
    count <- integer_scalar(n)
    if (length(first) != 1L || !is.finite(first) ||
        first < 1 || first != floor(first) ||
        is.na(count) || count < 1L) {
        stop("`first` and `n` must be positive integer counts.",
            call. = FALSE)
    }
    massive_read_cluster_rows_cpp(
        x$membership_path, x$confidence_path, x$nrow,
        first, count, 128e6)
}

#' @export
print.fastEmbedR_massive_clusters <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR landmark clusters\n")
    cat("  rows: ", format(x$nrow, scientific = FALSE),
        "; communities: ", x$n_communities, "\n", sep = "")
    cat("  method: ", x$method, "; backend: ", x$backend,
        "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_clusters <- function(x, n = 6L, ...) {
    massive_read_cluster_rows(x, n = min(n, x$nrow))
}
