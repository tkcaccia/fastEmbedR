massive_projection_cuda_resources <- function(graph, dimensions,
        rows) {
    info <- massive_cuda_memory_cpp()
    fixed <- 128 * 1024^2 + 8 * graph$n_reference * dimensions
    per_row <- 12 * graph$ncol + 16 * dimensions
    maximum <- floor((0.65 * info$free_bytes - fixed) / per_row)
    if (!is.finite(maximum) || maximum < 1) {
        stop("CUDA projection buffers exceed the free-VRAM budget.",
            call. = FALSE)
    }
    rows <- as.integer(min(rows, maximum))
    list(rows = rows, peak_vram_bytes = fixed + rows * per_row,
        free_vram_bytes = info$free_bytes, gpu_device = info$device)
}

massive_cuda_projection_engine <- function(reference_rows, dimensions,
        neighbors, batch_rows) {
    reference_bytes <- reference_rows * dimensions * 8
    batch_bytes <- batch_rows * (neighbors * 12 + dimensions * 8)
    if (reference_bytes > batch_bytes) {
        "persistent_cuda"
    } else {
        "batch_cuda"
    }
}

massive_projection_resources <- function(graph, dimensions,
        chunk_rows, memory_limit, backend) {
    limit <- massive_memory_bytes(memory_limit)
    fixed <- 64 * 1024^2 + 32 * graph$n_reference * dimensions
    per_row <- 20 * graph$ncol + 64 * dimensions
    maximum <- floor((0.7 * limit - fixed) / per_row)
    maximum <- min(maximum, floor(128e6 / (20 * graph$ncol)),
        .Machine$integer.max - graph$n_reference)
    requested <- chunk_rows %||% 250000L
    if (!is.numeric(requested) || length(requested) != 1L ||
        !is.finite(requested) || requested < 1 ||
        requested != floor(requested) ||
        requested > .Machine$integer.max) {
        stop("`chunk_rows` must be one positive integer.", call. = FALSE)
    }
    if (!is.finite(maximum) || maximum < 1) {
        stop("Projection buffers exceed `memory_limit` or R row limits.",
            call. = FALSE)
    }
    rows <- as.integer(min(requested, maximum))
    gpu <- if (backend == "cuda") {
        massive_projection_cuda_resources(graph, dimensions, rows)
    } else NULL
    if (!is.null(gpu)) rows <- gpu$rows
    result <- list(chunk_rows = rows,
        peak_ram_bytes = fixed + rows * per_row,
        peak_vram_bytes = gpu$peak_vram_bytes %||% 0,
        memory_limit_bytes = limit,
        output_bytes = graph$nrow * dimensions * 4)
    if (!is.null(gpu)) {
        result$free_vram_bytes <- gpu$free_vram_bytes
        result$gpu_device <- gpu$gpu_device
    }
    result
}

massive_projection_range <- function(graph, row_range) {
    if (is.null(row_range)) return(c(1, graph$nrow))
    if (!is.numeric(row_range) || length(row_range) != 2L ||
        any(!is.finite(row_range)) || any(row_range != floor(row_range)) ||
        row_range[1L] < 1 || row_range[1L] > row_range[2L] ||
        row_range[2L] > graph$nrow) {
        stop("`row_range` must be two valid inclusive row indices.",
            call. = FALSE)
    }
    as.double(row_range)
}

massive_project_umap_rows <- function(layout, knn, epochs, params,
        workers, seed) {
    initial <- project_embedding_knn_cpp(
        layout, knn$indices, knn$distances)
    if (epochs == 0L) return(initial)
    reference_rows <- nrow(layout)
    combined <- rbind(layout, initial)
    rows <- reference_rows + seq_len(nrow(initial))
    refined <- knn_umap_refine_rows_cpp(
        knn$indices, knn$distances, as.integer(rows), combined,
        epochs, params$min_dist, params$negative_rate,
        params$learning_rate, params$repulsion, workers, seed, FALSE)
    refined[rows, , drop = FALSE]
}

massive_project_tsne_rows <- function(layout, knn, iterations,
        perplexity, workers, seed, backend) {
    transform_tsne(
        layout, knn = knn, perplexity = perplexity,
        n_iter = iterations, backend = backend, n.cores = workers,
        seed = seed
    )
}

massive_project_batch <- function(fit, layout, graph, first,
        count, backend, workers, epochs, iterations,
        perplexity, params, seed, reference_rows, projector) {
    knn <- massive_read_knn_rows(graph, first, count)
    batch_seed <- as.integer((as.double(seed) + first - 1) %%
        .Machine$integer.max)
    projected <- if (fit$method == "umap") {
        if (backend == "cuda") {
            if (is.null(projector)) {
                project_embedding_knn_cuda_cpp(layout,
                    knn$indices, knn$distances)
            } else {
                massive_cuda_projector_batch_cpp(projector,
                    knn$indices, knn$distances)
            }
        } else {
            massive_project_umap_rows(layout, knn, epochs, params,
                workers, batch_seed)
        }
    } else {
        massive_project_tsne_rows(layout, knn, iterations,
            perplexity, workers, batch_seed, backend)
    }
    if (!identical(dim(projected), c(count, ncol(layout))) ||
        any(!is.finite(projected))) {
        stop("Projection returned invalid coordinates.", call. = FALSE)
    }
    if (!is.null(reference_rows)) {
        low <- findInterval(first - 1, reference_rows) + 1L
        high <- findInterval(first + count - 1, reference_rows)
        if (low <= high) {
            selected <- seq.int(low, high)
            rows <- as.integer(reference_rows[selected] - first + 1)
            projected[rows, ] <- layout[selected, , drop = FALSE]
        }
    }
    projected
}

massive_projection_signature <- function(fit, graph, resources,
        settings, reference_rows) {
    files <- massive_checkpoint_file_identity(c(
        graph$indices_path, graph$distances_path))
    params <- if (fit$method == "umap") {
        landmark_umap_optimizer_parameters(fit)
    } else NULL
    massive_checkpoint_signature(list(
        version = 1L, layout = fit$layout, method = fit$method,
        parameters = params, graph = list(nrow = graph$nrow,
            ncol = graph$ncol, n_reference = graph$n_reference,
            metric = graph$metric), files = files,
        reference_rows = reference_rows,
        output_bytes = resources$output_bytes,
        settings = settings))
}

massive_projection_checkpoint <- function(path, fit, graph,
        resources, settings, reference_rows, resume, rows_total) {
    sidecar <- paste0(path, ".checkpoint.rds")
    signature <- massive_projection_signature(fit, graph,
        resources, settings, reference_rows)
    if (resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        rows <- if (is.list(state)) state$completed_rows else NULL
        part_exists <- file.exists(paste0(path, ".part"))
        size <- if (part_exists) {
            file.info(paste0(path, ".part"))$size
        } else 0
        if (!is.list(state) ||
            !identical(state$signature, signature) ||
            !is.numeric(rows) || length(rows) != 1L ||
            !is.finite(rows) || rows != floor(rows) ||
            rows < 0 || rows > rows_total ||
            (!part_exists && rows > 0) ||
            is.na(size) || size < rows * ncol(fit$layout) * 4 ||
            size > resources$output_bytes) {
            stop("Projection checkpoint or partial output does not ",
                "match the current inputs.", call. = FALSE)
        }
        return(list(state = state, path = sidecar))
    }
    if (file.exists(sidecar)) {
        stop("Projection checkpoint already exists; use `resume = TRUE`.",
            call. = FALSE)
    }
    state <- list(signature = signature, completed_rows = 0)
    massive_checkpoint_write(state, sidecar)
    list(state = state, path = sidecar)
}

massive_next_projection_batch <- function(fit, layout, graph, first,
        count, backend, workers, epochs, iterations, perplexity,
        params, seed, reference_rows, local, projector) {
    projected <- if (is.null(local)) {
        massive_project_batch(fit, layout, graph, first,
            count, backend, workers, epochs, iterations, perplexity,
            params, seed, reference_rows, projector)
    } else {
        batch_seed <- as.integer((as.double(seed) + first - 1) %%
            .Machine$integer.max)
        massive_project_local_batch(local, graph, layout,
            params, first, count, epochs, workers, batch_seed,
            reference_rows)
    }
    if (!identical(dim(projected), c(count, ncol(layout))) ||
        any(!is.finite(projected))) {
        stop("Projection returned invalid coordinates.",
            call. = FALSE)
    }
    projected
}

massive_projection_finalize <- function(connection, part, path,
                                        bytes, checkpoint) {
    truncate(connection)
    close(connection)
    if (file.info(part)$size != bytes || !file.rename(part, path)) {
        stop("Could not finalize projected coordinates; .part retained.",
            call. = FALSE)
    }
    if (!is.null(checkpoint)) unlink(checkpoint$path)
}

massive_projection_release <- function(projector) {
    if (!is.null(projector)) massive_cuda_projector_release_cpp(projector)
}

massive_project_stream <- function(fit, graph, path, resources,
        backend, workers, epochs, iterations, perplexity, seed,
        reference_rows, checkpoint, row_range, local = NULL) {
    part <- paste0(path, ".part")
    state <- checkpoint$state
    completed <- if (is.null(state)) 0 else state$completed_rows
    first <- row_range[1L] + completed
    connection <- file(part, if (completed > 0) "r+b" else "wb")
    on.exit(try(close(connection), silent = TRUE))
    layout <- embedding_dense_double_matrix(fit$layout)
    projector <- if (identical(resources$projection_engine,
            "persistent_cuda")) {
        massive_cuda_projector_create_cpp(layout, graph$ncol,
            resources$chunk_rows)
    } else NULL
    on.exit(massive_projection_release(projector), add = TRUE)
    if (completed > 0) {
        seek(connection, where = completed * ncol(layout) * 4,
            origin = "start", rw = "write")
    }
    params <- if (fit$method == "umap") {
        landmark_umap_optimizer_parameters(fit)
    } else NULL
    reported <- -1L
    while (first <= row_range[2L]) {
        count <- as.integer(min(resources$chunk_rows,
            row_range[2L] - first + 1))
        projected <- massive_next_projection_batch(fit, layout,
            graph, first, count, backend, workers, epochs, iterations,
            perplexity, params, seed, reference_rows, local, projector)
        writeBin(as.vector(t(projected)), connection,
            size = 4L, endian = "little")
        first <- first + count
        if (!is.null(state)) {
            flush(connection)
            state$completed_rows <- first - row_range[1L]
            massive_checkpoint_write(state, checkpoint$path)
        }
        done <- first - row_range[1L]
        total <- diff(row_range) + 1
        progress <- floor(20 * done / total)
        if (progress > reported) {
            message("EXPERIMENTAL landmark projection: ",
                done, "/", total, " rows")
            reported <- progress
        }
    }
    massive_projection_finalize(connection, part, path,
        resources$output_bytes, checkpoint)
}

massive_reference_indices <- function(selection, graph) {
    if (is.null(selection)) return(NULL)
    if (!is.list(selection)) {
        stop("`selection` must be a landmark selection result.",
            call. = FALSE)
    }
    indices <- selection$indices
    data <- selection$data
    if (!inherits(data, "fastEmbedR_massive_matrix") ||
        !is.numeric(indices) ||
        length(indices) != graph$n_reference ||
        data$nrow != graph$n_reference ||
        any(!is.finite(indices)) ||
        any(indices != floor(indices)) ||
        any(indices < 1) || any(indices > graph$nrow) ||
        any(diff(indices) <= 0)) {
        stop("`selection` must map the reference to sorted query rows.",
            call. = FALSE)
    }
    if (!identical(selection$indices_signature,
        massive_checkpoint_signature(indices))) {
        stop("Landmark row indices changed after selection.",
            call. = FALSE)
    }
    if (!is.null(graph$reference_identity) &&
        !identical(graph$reference_identity,
            massive_checkpoint_source_identity(data))) {
        stop("Landmark selection differs from the KNN reference.",
            call. = FALSE)
    }
    if (!is.null(selection$source_identity) &&
        !identical(selection$source_identity, graph$source_identity)) {
        stop("Landmark selection differs from the KNN query source.",
            call. = FALSE)
    }
    indices
}

massive_projection_cuda_controls <- function(fit, graph, epochs) {
    if (fit$method == "umap" && epochs != 0L) {
        stop("EXPERIMENTAL CUDA UMAP projection requires ",
            "`refinement_epochs = 0`; no CPU fallback was used.",
            call. = FALSE)
    }
    if (fit$method == "tsne" &&
        (!(ncol(fit$layout) %in% c(2L, 3L)) ||
        graph$ncol > 128L)) {
        stop("EXPERIMENTAL CUDA t-SNE projection requires a 2D or ",
            "3D reference and k <= 128; no CPU fallback was used.",
            call. = FALSE)
    }
    if (!isTRUE(embedding_cuda_available_cpp())) {
        stop("EXPERIMENTAL CUDA projection is unavailable; ",
            "no CPU fallback was used.", call. = FALSE)
    }
}

massive_projection_controls <- function(fit, graph, backend,
                                        refinement_epochs,
                                        transform_iter,
                                        transform_perplexity, seed) {
    if (inherits(fit$layout, "float32") &&
        !requireNamespace("float", quietly = TRUE)) {
        stop("A saved float32 reference requires `float`.",
            call. = FALSE)
    }
    if (!inherits(fit, "fastEmbedR_embedding") ||
        !(fit$method %in% c("umap", "tsne")) ||
        !inherits(graph, "fastEmbedR_massive_knn") ||
        graph$n_reference != nrow(fit$layout) ||
        graph$metric != "euclidean") {
        stop("Fit and graph must share a landmark reference.",
            call. = FALSE)
    }
    if (!(backend %in% c("cpu", "cuda")) ||
        !(graph$backend %in% c("cpu", "cuda"))) {
        stop("EXPERIMENTAL projection requires CPU or CUDA inputs.",
            call. = FALSE)
    }
    epochs <- integer_scalar(refinement_epochs)
    iterations <- integer_scalar(transform_iter)
    seed <- integer_scalar(seed)
    if (anyNA(c(epochs, iterations, seed)) ||
        epochs < 0L || iterations < 0L || seed < 0L) {
        stop("Epochs, iterations, and seed must be non-negative integers.",
            call. = FALSE)
    }
    if (fit$method == "tsne" &&
        (!is.numeric(transform_perplexity) ||
        length(transform_perplexity) != 1L ||
        !is.finite(transform_perplexity) ||
        transform_perplexity <= 0 ||
        transform_perplexity > graph$ncol)) {
        stop("`transform_perplexity` must be in (0, graph k].",
            call. = FALSE)
    }
    if (fit$method == "tsne" && iterations == 0L) {
        stop("t-SNE projection requires `transform_iter` > 0.",
            call. = FALSE)
    }
    if (backend == "cuda") {
        massive_projection_cuda_controls(fit, graph, epochs)
    }
    list(epochs = epochs, iterations = iterations, seed = seed)
}

massive_projection_result <- function(path, graph, fit, dimensions,
        backend, workers, resources, controls, checkpoint, local,
        reference_rows, elapsed, row_range) {
    result <- list(layout = massive_matrix(path,
            nrow = diff(row_range) + 1,
            ncol = dimensions), method = fit$method, backend = backend,
        mode = "landmark",
        n.cores = workers, graph = graph,
        reference_backend = fit$parameters$backend,
        reference_rows_preserved = !is.null(reference_rows),
        resources = resources, refinement_epochs = if (
            fit$method == "umap") controls$epochs else NA_integer_,
        transform_iter = if (fit$method == "tsne")
            controls$iterations else NA_integer_,
        checkpoint = checkpoint,
        local_refine = !is.null(local),
        local_neighbors = if (is.null(local)) NA_integer_ else
            local$neighbors,
        overlap_rows = if (is.null(local)) NA_integer_ else
            local$overlap,
        elapsed_sec = elapsed, experimental = TRUE,
        row_start = row_range[1L], row_end = row_range[2L])
    class(result) <- "fastEmbedR_massive_projection"
    result
}

#' @export
print.fastEmbedR_massive_projection <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR ", x$method,
        " landmark projection\n", sep = "")
    cat("  rows: ", x$layout$nrow,
        "; reference rows: ", x$graph$n_reference,
        "; backend: ", x$backend, "\n", sep = "")
    cat("  output: ", x$layout$path, "\n", sep = "")
    invisible(x)
}

massive_projection_complete <- function(result) {
    path <- result$layout$path
    graph <- result$graph
    files <- c(path, graph$indices_path, graph$distances_path)
    identity <- massive_checkpoint_file_identity(files)
    expected <- c(result$layout$nrow * result$layout$ncol,
        rep(graph$nrow * graph$ncol, 2L)) * 4
    if (!identical(identity$bytes, as.numeric(expected))) {
        stop("Completed projection files have unexpected sizes.",
            call. = FALSE)
    }
    graph_identity <- massive_checkpoint_file_identity(files[-1L])
    if (!is.null(graph$output_identity) &&
        !identical(graph_identity, graph$output_identity)) {
        stop("KNN files changed since graph construction.",
            call. = FALSE)
    }
    result$file_identity <- identity
    result$manifest_path <- paste0(path, ".manifest.rds")
    massive_checkpoint_write(list(version = 1L, result = result),
        result$manifest_path)
    result
}

#' EXPERIMENTAL reopen of completed landmark projections
#'
#' Reopens a file-backed projection without loading its coordinates. The
#' coordinate and KNN files must retain their paths, sizes, and modification
#' times. This local completion record is not a portable content hash.
#'
#' @param output The `.f32` path supplied to `massive_project_landmarks()`.
#' @return A `fastEmbedR_massive_projection` with file-backed `layout`.
#' @export
massive_open_projection <- function(output) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || tolower(tools::file_ext(output)) != "f32") {
        stop("`output` must be one .f32 projection path.",
            call. = FALSE)
    }
    path <- file.path(normalizePath(dirname(output), mustWork = TRUE),
        basename(output))
    manifest <- paste0(path, ".manifest.rds")
    if (!file.exists(manifest)) stop(
        "Projection completion manifest is missing.", call. = FALSE)
    saved <- tryCatch(suppressWarnings(readRDS(manifest)),
        error = function(e) NULL)
    result <- if (is.list(saved)) saved$result else NULL
    valid <- is.list(saved) && identical(saved$version, 1L) &&
        is.list(result) &&
        inherits(result, "fastEmbedR_massive_projection") &&
        inherits(result$layout, "fastEmbedR_massive_matrix") &&
        inherits(result$graph, "fastEmbedR_massive_knn") &&
        identical(result$layout$path, path) &&
        identical(result$layout$format, "f32") &&
        identical(result$manifest_path, manifest) &&
        isTRUE(result$experimental) &&
        isTRUE(result$backend %in% c("cpu", "cuda")) &&
        isTRUE(result$method %in% c("umap", "tsne")) &&
        identical(result$layout$nrow,
            diff(c(result$row_start, result$row_end)) + 1)
    if (!isTRUE(valid)) stop(
        "Projection completion manifest is invalid.", call. = FALSE)
    files <- c(path, result$graph$indices_path,
        result$graph$distances_path)
    identity <- tryCatch(massive_checkpoint_file_identity(files),
        error = function(e) NULL)
    expected <- c(result$layout$nrow * result$layout$ncol,
        rep(result$graph$nrow * result$graph$ncol, 2L)) * 4
    if (!identical(identity, result$file_identity) ||
        !identical(identity$bytes, as.numeric(expected))) stop(
        "Projection or KNN files changed since completion.",
        call. = FALSE)
    result
}

massive_projection_setup <- function(fit, graph, output,
        selection, backend, n.cores, chunk_rows, memory_limit,
        transform_perplexity, checkpoint, resume, controls,
        local, range) {
    reference_rows <- massive_reference_indices(selection, graph)
    dimensions <- ncol(fit$layout)
    workers <- normalize_nn_threads(n.cores)
    if (backend == "cuda" && workers != 1L) {
        stop("CUDA projection requires `n.cores = 1`.", call. = FALSE)
    }
    resources <- massive_projection_resources(
        graph, dimensions, chunk_rows, memory_limit, backend)
    resources$projection_engine <- if (backend == "cuda" &&
        fit$method == "umap") {
        massive_cuda_projection_engine(graph$n_reference, dimensions,
            graph$ncol, resources$chunk_rows)
    } else "batch"
    resources$output_bytes <- (diff(range) + 1) * dimensions * 4
    if (!is.null(local)) {
        resources <- massive_local_resources(resources, graph,
            dimensions, local)
    }
    path <- massive_float_output(output, resources$output_bytes,
        resume = resume && file.exists(paste0(output, ".part")))
    if (file.exists(paste0(path, ".manifest.rds"))) stop(
        "Projection completion manifest already exists.",
        call. = FALSE)
    settings <- list(backend = backend, workers = workers,
        chunk_rows = resources$chunk_rows, epochs = controls$epochs,
        iterations = controls$iterations, perplexity = transform_perplexity,
        seed = controls$seed, row_range = range)
    if (!is.null(local)) settings$local <- list(
        source = local$identity, neighbors = local$neighbors,
        overlap = local$overlap)
    saved <- if (checkpoint) massive_projection_checkpoint(
        path, fit, graph, resources, settings, reference_rows,
        resume, diff(range) + 1) else NULL
    list(path = path, resources = resources, saved = saved,
        reference_rows = reference_rows, workers = workers,
        dimensions = dimensions)
}

#' Experimental file-backed landmark embedding projection
#'
#' Projects each saved query-to-landmark KNN batch into a fixed reference
#' embedding. Fit `umap()` or `tsne()` on the same landmark rows used to
#' construct `graph`. Only the query coordinates are written; reference
#' coordinates are not moved. Query batches are independent, so changing
#' `chunk_rows` can change stochastic refinement results.
#'
#' @param fit A `fastEmbedR_embedding` fitted on the landmark reference.
#' @param graph File-backed query-to-landmark KNN from
#'   `massive_landmark_knn()`.
#' @param selection Optional result of `massive_select_landmarks()` from
#'   the same query source. Its rows retain the fitted reference coordinates.
#' @param output New row-major float32 `.f32` output path.
#' @param backend `"cpu"` or `"cuda"`. CUDA supports zero-epoch UMAP
#'   projection and 2D or 3D t-SNE fixed-reference transformation.
#'   Unsupported CUDA routes fail without CPU fallback.
#' @param n.cores Number of CPU workers.
#' @param chunk_rows Maximum query rows per projection batch.
#' @param memory_limit Conservative RAM budget.
#' @param refinement_epochs UMAP fixed-reference refinement epochs;
#'   zero uses weighted KNN projection only.
#' @param source Query `massive_matrix()` descriptor. Required only when
#'   `local_refine = TRUE`; it must match the saved KNN source identity.
#' @param local_refine Reconstruct a bounded CPU query-to-query graph in
#'   overlapping windows before UMAP refinement. EXPERIMENTAL and opt-in.
#' @param local_neighbors Non-self query neighbors per local window.
#' @param overlap_rows Extra query rows on each side of a window; must be at
#'   least `local_neighbors`.
#' @param transform_iter t-SNE fixed-reference transform iterations.
#' @param transform_perplexity t-SNE transform perplexity, at most graph k.
#' @param seed Random seed for batch-specific optimizer streams.
#' @param checkpoint Save completed row counts after each batch. With
#'   `devices`, also retain a parent manifest for shard recovery.
#' @param resume Continue from matching checkpoint and partial output.
#'   A missing `.part` file is allowed only before the first row is written;
#'   completed CUDA shards are reused before the final merge.
#' @param row_range Optional inclusive query-row range, used for independent
#'   projection shards. Output contains only that range.
#' @param devices Optional distinct zero-based CUDA device indices. Only
#'   independent projection batches are distributed; reference fitting and
#'   KNN construction remain on one device. Shards remain on disk.
#' @return An experimental file-backed result with `layout` as a
#'   `fastEmbedR_massive_matrix` descriptor. A `.manifest.rds` sidecar
#'   supports `massive_open_projection()` on the same files.
#' @export
massive_project_landmarks <- function(
    fit, graph, output, selection = NULL,
    backend = "cpu", n.cores = 1L,
    chunk_rows = NULL, memory_limit = "8GB",
    refinement_epochs = 50L, transform_iter = 250L,
    transform_perplexity = 5, seed = 4L,
    checkpoint = FALSE, resume = FALSE,
    source = NULL, local_refine = FALSE,
    local_neighbors = 15L, overlap_rows = 500L,
    row_range = NULL, devices = NULL
) {
    if (!is.null(devices)) {
        arguments <- list(fit = fit, graph = graph, output = output,
            selection = selection, backend = backend, n.cores = n.cores,
            chunk_rows = chunk_rows, memory_limit = memory_limit,
            refinement_epochs = refinement_epochs,
            transform_iter = transform_iter,
            transform_perplexity = transform_perplexity, seed = seed)
        return(massive_project_devices(arguments, devices, row_range,
            checkpoint, resume, local_refine))
    }
    massive_checkpoint_validate_controls(checkpoint, resume)
    controls <- massive_projection_controls(
        fit, graph, backend, refinement_epochs,
        transform_iter, transform_perplexity, seed)
    local <- massive_local_controls(fit, graph, backend, source,
        local_refine, local_neighbors, overlap_rows, controls$epochs)
    range <- massive_projection_range(graph, row_range)
    if (!is.null(local) && !is.null(row_range)) {
        stop("Local refinement does not support projection shards.",
            call. = FALSE)
    }
    prepared <- massive_projection_setup(fit, graph, output,
        selection, backend, n.cores, chunk_rows, memory_limit,
        transform_perplexity, checkpoint, resume, controls,
        local, range)
    message("EXPERIMENTAL landmark projection: ", fit$method,
        "; backend=", backend, "; rows=", prepared$resources$chunk_rows)
    started <- proc.time()[[3L]]
    massive_project_stream(fit, graph, prepared$path, prepared$resources,
        backend, prepared$workers, controls$epochs, controls$iterations,
        transform_perplexity, controls$seed, prepared$reference_rows,
        prepared$saved, range, local)
    result <- massive_projection_result(prepared$path, graph, fit,
        prepared$dimensions, backend, prepared$workers,
        prepared$resources, controls, checkpoint, local,
        prepared$reference_rows,
        unname(proc.time()[[3L]] - started), range)
    massive_projection_complete(result)
}
