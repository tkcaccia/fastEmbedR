massive_knn_resources <- function(x, reference, k, method,
                                    chunk_rows, memory_limit) {
    limit <- massive_memory_bytes(memory_limit)
    reference_items <- as.double(reference$nrow) * reference$ncol
    graph_estimate <- if (method == "hnsw") {
        reference$nrow * 2048
    } else {
        0
    }
    fixed <- reference_items * 16 + graph_estimate + 64 * 1024^2
    if (reference$format == "memory") fixed <- fixed +
        reference_items * 8
    if (x$format == "memory") fixed <- fixed +
        as.double(x$nrow) * x$ncol * 8
    per_row <- 12 * x$ncol + 32 * k
    maximum <- floor((0.7 * limit - fixed) / per_row)
    requested <- chunk_rows %||% 250000L
    if (!is.numeric(requested) || length(requested) != 1L ||
        !is.finite(requested) || requested < 1 ||
        requested != floor(requested) ||
        requested > .Machine$integer.max) {
        stop("`chunk_rows` must be one positive integer.", call. = FALSE)
    }
    if (!is.finite(maximum) || maximum < 1) {
        stop("Landmark index and query buffers exceed `memory_limit`.",
            call. = FALSE)
    }
    rows <- as.integer(min(requested, maximum))
    list(chunk_rows = rows, peak_ram_bytes = fixed + rows * per_row,
        index_estimate_bytes = graph_estimate,
        output_bytes = as.double(x$nrow) * k * 8,
        memory_limit_bytes = limit)
}

massive_knn_cuda_resources <- function(resources, reference, k,
                                        method, info) {
    reference_items <- as.double(reference$nrow) * reference$ncol
    fixed <- 128 * 1024^2 + reference_items *
        if (method == "ivf") 12 else 8
    per_row <- 4 * reference$ncol + 20 * k
    budget <- 0.65 * info$free_bytes
    maximum <- floor((budget - fixed) / per_row)
    if (!is.finite(maximum) || maximum < 1) {
        stop("Estimated CUDA index and query buffers exceed the ",
            "conservative free-VRAM budget.", call. = FALSE)
    }
    rows <- as.integer(min(resources$chunk_rows, 32768, maximum))
    resources$chunk_rows <- rows
    resources$peak_vram_bytes <- fixed + rows * per_row
    resources$free_vram_bytes <- info$free_bytes
    resources$vram_budget_bytes <- budget
    resources$gpu_device <- info$device
    resources
}

massive_knn_paths <- function(output, output_bytes, resume = FALSE,
        allow_final_resume = FALSE, allow_mixed_resume = FALSE) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) {
        stop("`output` must be one file prefix.", call. = FALSE)
    }
    parent <- normalizePath(dirname(output), mustWork = TRUE)
    prefix <- file.path(parent, basename(output))
    paths <- c(indices = paste0(prefix, ".indices.u32"),
        distances = paste0(prefix, ".distances.f32"))
    parts <- paste0(paths, ".part")
    part_exists <- file.exists(parts)
    finished <- file.exists(paths)
    mixed <- resume && allow_mixed_resume &&
        sum(finished) == 1L && sum(part_exists) == 1L &&
        all(finished != part_exists)
    if ((any(finished) && !(resume && allow_final_resume)) ||
        any(finished & part_exists) ||
        (resume && any(part_exists) && !all(part_exists) && !mixed) ||
        (!resume && any(part_exists))) {
        stop("KNN output or partial files conflict with `resume`.",
            call. = FALSE)
    }
    written <- if (resume) sum(file.info(parts)$size[part_exists]) else 0
    if (any(finished & file.info(paths)$size != output_bytes / 2)) {
        stop("Completed KNN file has the wrong byte count.",
            call. = FALSE)
    }
    remaining <- output_bytes - written -
        sum(file.info(paths)$size[finished])
    if (!is.finite(remaining) || remaining < 0 ||
        remaining > 0.9 * massive_disk_available_cpp(paths[[1L]])) {
        stop("KNN output exceeds the conservative free-disk budget.",
            call. = FALSE)
    }
    paths
}

massive_knn_reference <- function(reference, k, method,
        n.cores, backend, resources = NULL, hnsw_build = NULL) {
    if (method == "stream_exact") return(list(reference = reference,
        reference_chunk_rows = resources$reference_chunk_rows,
        build_seconds = 0,
        reference_storage = "file_backed_float32"))
    if (method %in% c("faiss_hnsw", "hnsw")) return(
        massive_faiss_hnsw_reference(reference, k, n.cores, hnsw_build))
    if (backend == "metal") return(massive_metal_reference(reference))
    started <- proc.time()[[3L]]
    native_buffer <- reference$format != "memory" &&
        (backend == "cuda" || method == "exact")
    matrix <- if (native_buffer) {
        massive_reference_buffer_cpp(reference)
    } else if (reference$format == "memory" &&
        backend == "cpu" && method == "exact") {
        reference$data
    } else {
        massive_read_rows_cpp(reference, 1,
            as.integer(reference$nrow),
            8 * reference$nrow * reference$ncol)
    }
    storage <- if (native_buffer) "native_float32" else "R_double"
    if (backend == "cuda") {
        index <- native_cuda_index_build_cpp(matrix, k, method, 0.99)
        return(list(matrix = NULL, index = index,
            build_seconds = unname(proc.time()[[3L]] - started),
            reference_storage = storage))
    }
    if (method == "exact") {
        index <- native_exact_index_build_cpp(matrix)
        return(list(matrix = NULL, index = index,
            build_seconds = unname(proc.time()[[3L]] - started),
            reference_storage = storage))
    }
    stop("Unsupported landmark KNN index method.", call. = FALSE)
}

massive_metal_reference <- function(reference) {
    if (!requireNamespace("float", quietly = TRUE)) stop(
        "File-backed Metal KNN requires `float`.", call. = FALSE)
    started <- proc.time()[[3L]]
    data <- massive_read_rows_cpp(reference, 1L,
        as.integer(reference$nrow),
        4 * as.double(reference$nrow) * reference$ncol, TRUE)
    list(matrix = data, build_seconds = unname(
        proc.time()[[3L]] - started),
        reference_storage = "R_float32")
}

massive_faiss_hnsw_reference <- function(reference, k, n.cores,
        hnsw_build = NULL) {
    if (!requireNamespace("float", quietly = TRUE)) stop(
        "Persistent FAISS HNSW requires `float`.", call. = FALSE)
    started <- proc.time()[[3L]]
    data <- massive_read_rows_cpp(reference, 1L,
        as.integer(reference$nrow),
        4 * reference$nrow * reference$ncol, TRUE)
    index <- faissr_hnsw_index_build_cpp(data,
        as.integer(reference$nrow), as.integer(reference$ncol),
        k, n.cores, hnsw_build$m %||% 0L,
        hnsw_build$ef_construction %||% 0L)
    list(index = index, build_seconds = unname(
        proc.time()[[3L]] - started),
        reference_storage = "R_float32", build_threads = n.cores)
}

massive_knn_query <- function(state, query, k, method,
                                n.cores, backend, ef_search = 0L) {
    if (backend == "cuda") {
        return(native_cuda_index_search_cpp(state$index, query, k))
    }
    if (backend == "metal") return(native_metal_query_knn_cpp(
        state$matrix, query, k, "exact", "euclidean", 0.99))
    if (method %in% c("faiss_hnsw", "hnsw")) return(
        faissr_hnsw_index_search_cpp(state$index$pointer, query,
            k, if (ef_search == 0L) state$index$efSearch else ef_search,
            n.cores))
    if (method == "exact") {
        return(native_exact_index_search_cpp(state$index, query,
            k, n_threads = n.cores))
    }
    stop("Unsupported landmark KNN search method.", call. = FALSE)
}

massive_knn_finish <- function(indices, distances, parts,
                                paths, nrow, k) {
    if (!is.null(indices)) {
        truncate(indices)
        truncate(distances)
        close(indices)
        close(distances)
    }
    expected <- as.double(nrow) * k * 4
    sizes <- file.info(parts)$size
    if (anyNA(sizes) || any(sizes != expected)) {
        observed <- paste(ifelse(is.na(sizes), "missing",
            format(sizes, scientific = FALSE, trim = TRUE)),
            collapse = ", ")
        stop("Landmark KNN output size mismatch: expected ",
            format(expected, scientific = FALSE, trim = TRUE),
            " bytes per file; observed ", observed, ".",
            call. = FALSE)
    }
    if (!file.rename(parts[["indices"]], paths[["indices"]])) {
        stop("Could not finalize landmark KNN indices.", call. = FALSE)
    }
    if (!file.rename(parts[["distances"]], paths[["distances"]])) {
        if (!file.rename(paths[["indices"]], parts[["indices"]])) {
            stop("Could not restore KNN indices after rename failure.",
                call. = FALSE)
        }
        stop("Could not finalize landmark KNN distances.",
            call. = FALSE)
    }
}

massive_knn_checkpoint <- function(x, setup, resume) {
    path <- sub("\\.indices\\.u32$", ".checkpoint.rds",
        setup$paths[["indices"]])
    settings <- list(
        version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        faissr = if (setup$method == "hnsw")
            massive_faissr_identity() else NULL,
        query = massive_checkpoint_source_identity(x),
        reference = massive_checkpoint_source_identity(setup$reference),
        k = setup$k, method = setup$method,
        backend = setup$backend, workers = setup$workers,
        build_threads = if (setup$method == "hnsw") 1L else NULL,
        chunk_rows = setup$resources$chunk_rows)
    if (setup$method == "stream_exact") {
        settings$reference_chunk_rows <-
            setup$resources$reference_chunk_rows
    }
    signature <- massive_checkpoint_signature(settings)
    if (resume) {
        state <- tryCatch(readRDS(path), error = function(e) NULL)
        rows <- if (is.list(state)) state$completed_rows else NULL
        parts <- paste0(setup$paths, ".part")
        part_exists <- file.exists(parts)
        sizes <- if (all(part_exists)) file.info(parts)$size else c(0, 0)
        if (!is.list(state) ||
            !identical(state$signature, signature) ||
            !is.numeric(rows) || length(rows) != 1L ||
            !is.finite(rows) || rows != floor(rows) ||
            rows < 0 || rows > x$nrow ||
            (rows > 0 && !all(part_exists)) ||
            anyNA(sizes) || any(sizes < rows * setup$k * 4) ||
            any(sizes > x$nrow * setup$k * 4) ||
            (rows > 0 && !is.character(state$pilot_signature))) {
            stop("KNN checkpoint or partial output does not match ",
                "the current inputs.", call. = FALSE)
        }
        return(list(state = state, path = path))
    }
    if (file.exists(path)) {
        stop("KNN checkpoint already exists; use `resume = TRUE`.",
            call. = FALSE)
    }
    state <- list(signature = signature, completed_rows = 0,
        pilot_signature = NULL, pilot_metadata = NULL)
    massive_checkpoint_write(state, path)
    list(state = state, path = path)
}

massive_knn_batch <- function(x, state, first, rows, k, method,
                                n.cores, backend) {
    if (method == "stream_exact") {
        return(massive_exact_reference_batch_cpp(x,
            state$reference, first, rows, k,
            state$reference_chunk_rows, n.cores))
    }
    query <- if (identical(method, "hnsw") || backend == "metal") {
        massive_read_rows_cpp(x, first, rows,
            4 * rows * x$ncol, TRUE)
    } else if (x$format == "memory") {
        massive_read_rows_cpp(x, first, rows, 8 * rows * x$ncol)
    } else {
        massive_reference_buffer_cpp(massive_matrix_rows(x,
            first, rows))
    }
    found <- massive_knn_query(state, query, k, method,
        n.cores, backend)
    if (backend == "metal" &&
        (!identical(found$backend, "metal") ||
            !identical(found$method, "native_metal_exact_query") ||
            !isTRUE(found$exact))) stop(
        "Metal landmark KNN changed backend or algorithm.",
        call. = FALSE)
    found
}

massive_knn_probe_signature <- function(neighbors) {
    massive_checkpoint_signature(list(
        indices = neighbors$indices,
        distances = writeBin(as.vector(t(neighbors$distances)), raw(),
            size = 4L, endian = "little"),
        metadata = massive_knn_metadata(neighbors)))
}

massive_knn_verify_probe <- function(x, state, k, method, n.cores,
                                        backend, resources, checkpoint) {
    saved <- checkpoint$state
    if (is.null(saved) || saved$completed_rows == 0) {
        return(invisible(NULL))
    }
    rows <- as.integer(min(resources$chunk_rows, x$nrow))
    observed <- massive_knn_batch(x, state, 1, rows, k, method,
        n.cores, backend)
    if (!identical(massive_knn_probe_signature(observed),
        saved$pilot_signature)) {
        stop("Rebuilt KNN index differs from the checkpoint pilot; ",
            "partial output was not changed.", call. = FALSE)
    }
    invisible(NULL)
}

massive_knn_open_parts <- function(parts, completed, k) {
    mode <- if (completed > 0) "r+b" else "wb"
    indices <- file(parts[["indices"]], mode)
    distances <- tryCatch(file(parts[["distances"]], mode),
        error = function(e) {
            close(indices)
            stop(e)
        })
    if (completed > 0) {
        offset <- completed * k * 4
        seek(indices, where = offset, origin = "start", rw = "write")
        seek(distances, where = offset, origin = "start", rw = "write")
    }
    list(indices = indices, distances = distances)
}

massive_knn_metadata <- function(neighbors) {
    list(pilot_recall = neighbors$pilot_recall %||% NA_real_,
        pilot_id_recall = neighbors$pilot_id_recall %||% NA_real_,
        pilot_min_row_recall =
            neighbors$pilot_min_row_recall %||% NA_real_,
        pilot_recall_metric = neighbors$pilot_recall_metric %||% "id",
        pilot_rows = neighbors$pilot_rows %||% 0L,
        nlist = neighbors$nlist %||% 0L,
        nprobe = neighbors$nprobe %||% 0L)
}

massive_knn_commit_batch <- function(saved, checkpoint, neighbors,
                                        completed, indices, distances) {
    if (is.null(saved)) return(NULL)
    if (is.null(saved$pilot_signature)) {
        saved$pilot_signature <- massive_knn_probe_signature(neighbors)
        saved$pilot_metadata <- massive_knn_metadata(neighbors)
    }
    flush(indices)
    flush(distances)
    saved$completed_rows <- completed
    massive_checkpoint_write(saved, checkpoint$path)
    saved
}

massive_knn_stream <- function(x, state, k, method, n.cores, backend,
                                resources, paths, checkpoint) {
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    massive_knn_verify_probe(x, state, k, method, n.cores,
        backend, resources, checkpoint)
    saved <- checkpoint$state
    completed <- saved$completed_rows %||% 0
    connections <- massive_knn_open_parts(parts, completed, k)
    indices <- connections$indices
    distances <- connections$distances
    committed <- FALSE
    on.exit({
        try(close(indices), silent = TRUE)
        try(close(distances), silent = TRUE)
        if (!committed && is.null(saved)) unlink(parts)
    })
    started <- proc.time()[[3L]]
    first <- completed + 1
    reported <- -1L
    neighbors <- saved$pilot_metadata
    while (first <= x$nrow) {
        rows <- as.integer(min(resources$chunk_rows, x$nrow - first + 1))
        neighbors <- massive_knn_batch(x, state, first, rows, k,
            method, n.cores, backend)
        writeBin(as.integer(t(neighbors$indices)), indices,
            size = 4L, endian = "little")
        writeBin(as.vector(t(neighbors$distances)), distances,
            size = 4L, endian = "little")
        first <- first + rows
        saved <- massive_knn_commit_batch(saved, checkpoint,
            neighbors, first - 1, indices, distances)
        progress <- floor(20 * (first - 1) / x$nrow)
        if (progress > reported) {
            message("EXPERIMENTAL landmark KNN: ", first - 1,
                "/", x$nrow, " rows")
            reported <- progress
        }
    }
    massive_knn_finish(
        indices, distances, parts, paths, x$nrow, k)
    committed <- TRUE
    if (!is.null(saved)) unlink(checkpoint$path)
    c(list(seconds = unname(proc.time()[[3L]] - started)),
        massive_knn_metadata(neighbors))
}

massive_knn_method <- function(method, backend, reference_rows, k) {
    method <- match.arg(method,
        c("auto", "exact", "stream_exact", "hnsw", "ivf"))
    if (method == "auto") {
        method <- if (backend == "metal") "exact" else if (
                backend == "cuda") {
            if (reference_rows < 100000) "exact" else "ivf"
        } else fastembedr_cpu_knn_method(reference_rows)
    }
    if (backend == "metal" && method != "exact") stop(
        "EXPERIMENTAL Metal landmark KNN supports exact search only.",
        call. = FALSE)
    if (backend == "cuda" && !method %in% c("exact", "ivf")) {
        stop("EXPERIMENTAL CUDA KNN supports exact or IVF-Flat; ",
            "no algorithm fallback was used.", call. = FALSE)
    }
    if (backend == "cpu" && method == "ivf") {
        stop("EXPERIMENTAL CPU landmark KNN has no IVF route; ",
            "no algorithm fallback was used.", call. = FALSE)
    }
    if (backend == "cuda" && k > 256L) {
        stop("EXPERIMENTAL CUDA KNN requires k <= 256.",
            call. = FALSE)
    }
    if (backend == "metal" && k > 128L) stop(
        "EXPERIMENTAL Metal landmark KNN requires k <= 128.",
        call. = FALSE)
    method
}

massive_knn_setup <- function(x, reference, k, output, backend,
                                n.cores, method, chunk_rows,
                                memory_limit, resume, cuda_memory = TRUE,
                                reference_chunk_rows = NULL,
                                allow_final_resume = FALSE) {
    if (!inherits(x, "fastEmbedR_massive_matrix"))
        stop("`x` must be a massive_matrix() descriptor.", call. = FALSE)
    if (is.list(reference) && inherits(reference$data,
        "fastEmbedR_massive_matrix")) reference <- reference$data
    if (!inherits(reference, "fastEmbedR_massive_matrix") ||
        x$ncol != reference$ncol ||
        reference$nrow > .Machine$integer.max) stop(
        "Reference dimensions must match and fit R integer IDs.", call. = FALSE)
    backend <- match.arg(backend, c("cpu", "cuda", "metal"))
    if (backend == "cuda" && !native_cuda_knn_available_cpp()) stop(
        "EXPERIMENTAL CUDA KNN requires native cuVS; ",
        "no CPU fallback was used.", call. = FALSE)
    if (backend == "metal" &&
        (!native_metal_knn_available_cpp() || x$ncol > 16384L))
        stop("Metal KNN needs Metal and <=16,384 columns.", call. = FALSE)
    k <- integer_scalar(k)
    if (is.na(k) || k < 1L || k > reference$nrow)
        stop("`k` must be a positive landmark count.", call. = FALSE)
    method <- massive_knn_method(method, backend, reference$nrow, k)
    if (!is.null(reference_chunk_rows) && method != "stream_exact")
        stop("`reference_chunk_rows` requires `stream_exact`.", call. = FALSE)
    if (method == "stream_exact" &&
        (reference$format == "memory" || k > 65536L))
        stop("Stream exact: file-backed reference; k <= 65,536.", call. = FALSE)
    workers <- normalize_nn_threads(n.cores)
    if (backend == "metal" && workers != 1L)
        stop("Metal KNN needs `n.cores = 1`.", call. = FALSE)
    resources <- if (method == "stream_exact")
        massive_full_knn_resources(x, k, chunk_rows,
            reference_chunk_rows, memory_limit, reference) else
            massive_knn_resources(
        x, reference, k, method, chunk_rows, memory_limit)
    if (backend == "cuda" && cuda_memory)
        resources <- massive_knn_cuda_resources(resources, reference,
            k, method, native_cuda_memory_info_cpp())
    if (backend == "metal") resources <- c(resources, list(
        metal_memory_model = "unified_shared",
        metal_buffer_bytes_estimate =
            8 * reference$nrow * reference$ncol +
            8 * resources$chunk_rows * x$ncol))
    paths <- massive_knn_paths(output, resources$output_bytes,
        resume, allow_final_resume)
    list(reference = reference, k = k, backend = backend, method = method,
        workers = workers, resources = resources, paths = paths)
}

massive_knn_result <- function(x, setup, state, elapsed,
                                checkpoint, devices) {
    paths <- setup$paths
    out <- list(indices_path = paths[["indices"]],
        distances_path = paths[["distances"]],
        nrow = x$nrow, ncol = setup$k, n_reference = setup$reference$nrow,
        source_identity = if (x$format != "memory") {
            massive_checkpoint_source_identity(x)
        } else NULL,
        reference_identity = if (setup$reference$format != "memory") {
            massive_checkpoint_source_identity(setup$reference)
        } else NULL,
        output_identity = massive_checkpoint_file_identity(unname(paths)),
        implementation_identity =
            massive_checkpoint_implementation_identity(),
        faissr_identity = if (setup$method == "hnsw")
            massive_faissr_identity() else NULL,
        metric = "euclidean", backend = setup$backend,
        method = setup$method, exact = setup$method %in% c(
            "exact", "stream_exact", "sharded_exact"),
        index_reused = setup$backend != "metal" &&
            (setup$backend == "cuda" ||
                setup$method %in% c("hnsw", "exact")),
        target_recall = 0.99, recall_audited = setup$method %in% c(
            "exact", "stream_exact", "sharded_exact"),
        pilot_recall = elapsed$pilot_recall,
        pilot_id_recall = elapsed$pilot_id_recall,
        pilot_min_row_recall = elapsed$pilot_min_row_recall,
        pilot_recall_metric = elapsed$pilot_recall_metric,
        pilot_rows = elapsed$pilot_rows, nlist = elapsed$nlist,
        nprobe = elapsed$nprobe, resources = setup$resources,
        build_seconds = state$build_seconds,
        reference_storage = state$reference_storage,
        spanning_links = isTRUE(state$spanning_links),
        hnsw_settings = if (setup$method == "hnsw")
            c(state$index[c("M", "efConstruction", "efSearch", "tuning_rule")],
                list(build_threads = state$build_threads)) else NULL,
        query_storage = if (setup$method == "stream_exact") {
            "file_backed_float32"
        } else if (setup$method == "hnsw" ||
            setup$backend == "metal") "R_float32" else if (
            x$format == "memory") {
            "R_double"
        } else "native_float32",
        query_seconds = elapsed$seconds, checkpoint = checkpoint,
        experimental = TRUE)
    class(out) <- "fastEmbedR_massive_knn"
    if (!is.null(devices)) out$gpu_devices <- devices
    out
}

#' Experimental file-backed query-to-landmark nearest neighbors
#'
#' Searches a disk-backed matrix in bounded query batches. The default
#' search keeps the landmark reference in RAM or VRAM. Explicit CPU
#' `stream_exact` scans bounded reference blocks without materializing the
#' full landmark matrix; this exact route is quadratic in query and
#' landmark counts. CPU exact and persistent faissR HNSW, and CUDA exact
#' or IVF-Flat search retain references across query batches. Metal exact
#' search retains a host float32 reference but reuploads it each batch.
#' CUDA needs cuVS support. IVF-Flat calibrates on the first query batch;
#' pilot recall is not guaranteed for later rows. For CUDA IVF,
#' `pilot_recall` counts distance-equivalent ties at the exact kth distance;
#' `pilot_id_recall` reports mean neighbor-ID overlap. Persistent sharded
#' IVF also requires `pilot_min_row_recall` to meet its target; stricter
#' probing can increase query time substantially.
#' CPU uses exact search by default below 5,000 landmarks.
#' Output files contain row-major, one-based `uint32` landmark indices and
#' float32 distances. HNSW's 0.99 recall is a target, not a measured result.
#' Checkpointed HNSW uses one index-build thread for reproducible resume;
#' `n.cores` still controls query search. The effective build thread count
#' is recorded in `hnsw_settings`.
#' Explicit CUDA `sharded_exact` builds one bounded reference shard at a
#' time and merges exact candidates on disk. It rereads every query for
#' each shard and can be much slower than an in-VRAM index.
#'
#' @param x Query `massive_matrix()` descriptor.
#' @param reference Landmark `massive_matrix()` descriptor or the result of
#'   `massive_select_landmarks()`.
#' @param k Number of nearest landmarks.
#' @param output New file prefix for `.indices.u32` and `.distances.f32`.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"`. Metal uses exact
#'   search with bounded query batches and a resident reference.
#' @param n.cores CPU worker count.
#' @param method `"auto"`, `"exact"` (also Metal), `"stream_exact"` (CPU with a
#'   file-backed reference), `"hnsw"` (CPU faissR), `"ivf"` (CUDA), or
#'   `"sharded_exact"` or `"sharded_ivf"` (CUDA with a file-backed
#'   reference). Sharded IVF is approximate and needs a recall audit.
#' @param chunk_rows Maximum query rows per batch.
#' @param reference_chunk_rows Maximum reference rows per batch for
#'   `"stream_exact"`, `"sharded_exact"`, or `"sharded_ivf"`;
#'   otherwise must be `NULL`.
#' @param memory_limit Conservative RAM budget. CUDA also caps batches to a
#'   conservative fraction of currently free VRAM.
#' @param checkpoint Save completed KNN batches to a sidecar file.
#'   Multi-GPU `"sharded_ivf"` checkpoints only completed GPU query shards.
#'   Single-GPU `"sharded_ivf"` does not support checkpointing.
#' @param resume Continue from matching partial KNN files and checkpoint.
#'   Sharded exact CUDA search commits every 100 query blocks and at each
#'   reference-shard boundary. Multi-GPU sharded IVF reuses completed GPU
#'   shards but restarts an interrupted shard from its first query row.
#' @param devices Optional CUDA device indices. Multiple devices search
#'   disjoint query ranges, including with `sharded_exact`. Checkpointing
#'   requires file-backed query and reference sources.
#' @param checkpoint_every Commit every this many query blocks for sharded
#'   CUDA search. Other search methods use their existing checkpoint policy.
#' @return A lightweight `fastEmbedR_massive_knn` descriptor.
#' @export
massive_landmark_knn <- function(x, reference, k, output,
        backend = "cpu", n.cores = 1L,
        method = c("auto", "exact", "stream_exact",
            "hnsw", "ivf", "sharded_exact", "sharded_ivf"),
        chunk_rows = NULL, reference_chunk_rows = NULL,
        memory_limit = "8GB", checkpoint = FALSE,
        resume = FALSE, devices = NULL,
        checkpoint_every = 100L) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    method <- match.arg(method)
    checkpoint_every <- integer_scalar(checkpoint_every)
    if (is.na(checkpoint_every) || checkpoint_every < 1L) stop(
        "`checkpoint_every` must be a positive integer.", call. = FALSE)
    if (method %in% c("sharded_exact", "sharded_ivf")) return(
        massive_landmark_knn_sharded(x, reference, k, output,
            backend, n.cores, chunk_rows, reference_chunk_rows,
            memory_limit, checkpoint, resume, devices, method,
            checkpoint_every))
    if (checkpoint_every != 100L) stop(
        "`checkpoint_every` requires sharded CUDA search.",
        call. = FALSE)
    if (!is.null(reference_chunk_rows) && backend == "cuda") stop(
        "`reference_chunk_rows` requires CPU stream_exact; ",
        "no CUDA fallback was used.", call. = FALSE)
    if (!is.null(devices)) {
        devices <- massive_validate_devices(devices, backend, x$nrow)
        if (length(devices) > 1L) return(massive_knn_devices(x,
            reference, k, output, n.cores, method, chunk_rows,
            memory_limit, checkpoint, resume, devices))
        massive_cuda_select_cpp(devices[1L])
    }
    setup <- massive_knn_setup(x, reference, k, output, backend,
        n.cores, method, chunk_rows, memory_limit, resume,
        reference_chunk_rows = reference_chunk_rows)
    saved <- if (resume) massive_knn_checkpoint(x, setup, TRUE) else NULL
    message("EXPERIMENTAL landmark KNN: ", setup$method,
        "; backend=", setup$backend, "; CPU workers=", setup$workers,
        "; chunk rows=", setup$resources$chunk_rows)
    build_workers <- if (checkpoint && setup$method == "hnsw")
        1L else setup$workers
    state <- massive_knn_reference(setup$reference, setup$k,
        setup$method, build_workers, setup$backend, setup$resources)
    if (checkpoint && !resume)
        saved <- massive_knn_checkpoint(x, setup, FALSE)
    elapsed <- massive_knn_stream(x, state, setup$k, setup$method,
        setup$workers, setup$backend, setup$resources,
        setup$paths, saved)
    massive_knn_result(x, setup, state, elapsed, checkpoint, devices)
}

massive_landmark_shard_batch <- function(x, reference, setup,
        state, shard_first, shard_rows, query_first, first_shard,
        capture_pilot = FALSE) {
    rows <- as.integer(min(setup$resources$chunk_rows,
        x$nrow - query_first + 1))
    query <- massive_reference_buffer_cpp(
        massive_matrix_rows(x, query_first, rows))
    found <- native_cuda_index_search_cpp(state$index, query,
        as.integer(min(setup$k, shard_rows)))
    expected <- if (setup$method == "sharded_ivf") {
        "native_cuda_cuvs_ivf_flat"
    } else "native_cuda_cuvs_exact"
    search <- if (setup$method == "sharded_ivf") "ivf" else "exact"
    if (!identical(found$backend, expected) ||
        !identical(found$method, search)) stop(
        "Sharded landmark KNN changed CUDA algorithm.",
        call. = FALSE)
    massive_merge_knn_shard_cpp(
        paste0(setup$paths[["indices"]], ".part"),
        paste0(setup$paths[["distances"]], ".part"),
        found$indices, found$distances, query_first,
        shard_first, shard_rows, reference$nrow, setup$k,
        first_shard, x$nrow, FALSE)
    list(rows = rows, pilot = if (capture_pilot) {
        massive_knn_probe_signature(found)
    } else NULL, metadata = massive_knn_metadata(found))
}

massive_landmark_shard_probe <- function(x, setup, state,
        saved, shard_rows) {
    if (is.null(saved) || saved$state$completed_rows == 0)
        return(invisible(NULL))
    rows <- as.integer(min(setup$resources$chunk_rows, x$nrow))
    query <- massive_reference_buffer_cpp(
        massive_matrix_rows(x, 1, rows))
    found <- native_cuda_index_search_cpp(state$index, query,
        as.integer(min(setup$k, shard_rows)))
    if (!identical(massive_knn_probe_signature(found),
        saved$state$pilot_signature)) stop(
        "Rebuilt landmark shard differs from checkpoint pilot.",
        call. = FALSE)
    invisible(NULL)
}

massive_landmark_shard_commit <- function(saved, batch, done,
        total, setup) {
    if (is.null(saved)) return(NULL)
    if (is.null(saved$state$pilot_signature))
        saved$state$pilot_signature <- batch$pilot
    number <- ceiling(done / setup$resources$chunk_rows)
    if (done == total || number %% setup$checkpoint_every == 0) {
        saved$state$completed_rows <- done
        massive_checkpoint_write(saved$state, saved$path)
    }
    saved
}

massive_landmark_assert_sources <- function(x, reference, identity) {
    if (!identical(identity$query,
        massive_checkpoint_source_identity(x)) ||
        !identical(identity$reference,
        massive_checkpoint_source_identity(reference))) stop(
        "Sharded KNN source changed during search.", call. = FALSE)
}

massive_landmark_run_shard <- function(x, reference, setup,
        identity, shard, saved) {
    n_shards <- setup$resources$n_shards
    base <- floor(reference$nrow / n_shards)
    extra <- reference$nrow %% n_shards
    first <- 1 + shard * base + min(shard, extra)
    count <- base + as.integer(shard < extra)
    view <- massive_matrix_rows(reference, first, count)
    search <- if (setup$method == "sharded_ivf") "ivf" else "exact"
    state <- massive_knn_reference(view,
        as.integer(min(setup$k, count)), search, setup$workers, "cuda")
    massive_landmark_shard_probe(x, setup, state, saved, count)
    if (shard == 0L && !is.null(saved))
        massive_sharded_restore_first(setup$parts,
            saved$state$completed_rows, setup$k)
    query_first <- (saved$state$completed_rows %||% 0) + 1
    query_seconds <- 0
    metadata <- NULL
    while (query_first <= x$nrow) {
        massive_landmark_assert_sources(x, reference, identity)
        started <- proc.time()[[3L]]
        batch <- massive_landmark_shard_batch(x, reference,
            setup, state, first, count, query_first, shard == 0L,
            !is.null(saved) &&
                is.null(saved$state$pilot_signature))
        if (is.null(metadata)) metadata <- batch$metadata
        query_seconds <- query_seconds +
            unname(proc.time()[[3L]] - started)
        done <- query_first + batch$rows - 1
        saved <- massive_landmark_shard_commit(saved, batch, done,
            x$nrow, setup)
        query_first <- done + 1
        if (floor(20 * done / x$nrow) >
            floor(20 * (done - batch$rows) / x$nrow)) {
            message("EXPERIMENTAL CUDA landmark shard ", shard + 1L,
                "/", n_shards, ": ", done, "/", x$nrow, " query rows")
        }
    }
    if (!is.null(saved)) {
        saved$state$shard <- shard + 1L
        saved$state$completed_rows <- 0
        saved$state$pilot_signature <- NULL
        massive_checkpoint_write(saved$state, saved$path)
    }
    build_seconds <- state$build_seconds
    rm(state)
    invisible(gc(FALSE))
    list(build_seconds = build_seconds, query_seconds = query_seconds,
        saved = saved, metadata = metadata)
}

massive_landmark_shard_stream <- function(x, reference, setup,
        identity, saved) {
    build_seconds <- 0
    query_seconds <- 0
    settings <- list()
    start <- saved$state$shard %||% 0L
    if (start < setup$resources$n_shards) {
        for (shard in seq.int(start,
            setup$resources$n_shards - 1L)) {
            step <- massive_landmark_run_shard(x, reference, setup,
                identity, shard, saved)
            build_seconds <- build_seconds + step$build_seconds
            query_seconds <- query_seconds + step$query_seconds
            settings[[length(settings) + 1L]] <- step$metadata
            saved <- step$saved
        }
    }
    list(build_seconds = build_seconds,
        query_seconds = query_seconds, saved = saved,
        settings = settings)
}

massive_landmark_sharded_setup <- function(x, reference, k, output,
        backend, n.cores, chunk_rows, reference_chunk_rows,
        memory_limit, checkpoint, resume, devices,
        allow_final_resume = FALSE, method = "sharded_exact",
        checkpoint_every = 100L) {
    if (backend != "cuda") stop(
        "Sharded exact KNN needs CUDA; no fallback.", call. = FALSE)
    if (method == "sharded_ivf" && (checkpoint || resume)) stop(
        "Sharded IVF checkpoint and resume are unavailable.",
        call. = FALSE)
    if (is.list(reference) &&
        inherits(reference$data, "fastEmbedR_massive_matrix"))
        reference <- reference$data
    if (!inherits(x, "fastEmbedR_massive_matrix") ||
        !inherits(reference, "fastEmbedR_massive_matrix") ||
        x$format == "memory" || reference$format == "memory" ||
        x$ncol != reference$ncol || x$nrow > .Machine$integer.max ||
        reference$nrow > .Machine$integer.max) stop(
        "Sharded CUDA KNN needs matching file-backed matrices.",
        call. = FALSE)
    k <- integer_scalar(k)
    if (is.na(k) || k < 1L || k > min(256L, reference$nrow)) stop(
        "Sharded CUDA KNN requires 1 <= k <= min(256, landmarks).",
        call. = FALSE)
    devices <- massive_validate_devices(devices, backend, x$nrow)
    if (length(devices) > 1L) stop(
        "Sharded CUDA KNN supports one device.", call. = FALSE)
    if (length(devices) == 1L) massive_cuda_select_cpp(devices)
    resources <- massive_cuda_sharded_resources(reference, k,
        chunk_rows, reference_chunk_rows, memory_limit, x$nrow,
        exclude_self = FALSE,
        method = if (method == "sharded_ivf") "ivf_sharded" else
            "exact")
    paths <- massive_knn_paths(output, resources$output_bytes, resume,
        allow_final_resume)
    setup <- list(paths = paths, reference = reference, k = k,
        backend = backend, method = method,
        workers = normalize_nn_threads(n.cores),
        resources = resources, checkpoint_every = checkpoint_every)
    identity <- list(query = massive_checkpoint_source_identity(x),
        reference = massive_checkpoint_source_identity(reference))
    parts <- stats::setNames(paste0(paths, ".part"), names(paths))
    setup$parts <- parts
    saved <- if (checkpoint) massive_sharded_checkpoint(x, setup,
        paths, backend, resume, identity) else NULL
    list(setup = setup, identity = identity, saved = saved,
        devices = devices, reference = reference)
}

massive_landmark_knn_sharded <- function(x, reference, k, output,
        backend, n.cores, chunk_rows, reference_chunk_rows,
        memory_limit, checkpoint, resume, devices, method,
        checkpoint_every) {
    if (!is.null(devices)) {
        devices <- massive_validate_devices(devices, backend, x$nrow)
        if (length(devices) > 1L) return(
            massive_knn_sharded_devices(x, reference, k, output,
                n.cores, chunk_rows, reference_chunk_rows,
                memory_limit, checkpoint, resume, devices, method,
                checkpoint_every))
    }
    request <- massive_landmark_sharded_setup(x, reference, k,
        output, backend, n.cores, chunk_rows, reference_chunk_rows,
        memory_limit, checkpoint, resume, devices,
        method = method, checkpoint_every = checkpoint_every)
    setup <- request$setup
    parts <- setup$parts
    committed <- FALSE
    on.exit(if (!committed && is.null(request$saved)) unlink(parts))
    elapsed <- massive_landmark_shard_stream(x, request$reference,
        setup, request$identity, request$saved)
    massive_landmark_assert_sources(x, request$reference,
        request$identity)
    massive_knn_finish(NULL, NULL, parts, setup$paths, x$nrow,
        setup$k)
    committed <- TRUE
    if (!is.null(elapsed$saved)) unlink(elapsed$saved$path)
    state <- list(build_seconds = elapsed$build_seconds,
        reference_storage = "native_float32")
    timing <- list(seconds = elapsed$query_seconds,
        pilot_recall = NA_real_, pilot_rows = 0L,
        nlist = if (method == "sharded_ivf") NA_integer_ else 0L,
        nprobe = if (method == "sharded_ivf") NA_integer_ else 0L)
    result <- massive_knn_result(x, setup, state, timing,
        checkpoint, request$devices)
    if (method == "sharded_ivf") {
        result$ivf_settings <- elapsed$settings
    }
    result$resumed <- resume
    result
}

#' Read a bounded range of experimental landmark KNN rows
#'
#' @param x A `fastEmbedR_massive_knn` descriptor.
#' @param first First one-based query row.
#' @param n Number of consecutive query rows to inspect.
#' @return A list of resident `indices` and `distances` matrices.
#' @export
massive_read_knn_rows <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_knn")) {
        stop("`x` must be a massive_landmark_knn() result.",
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
    massive_read_knn_rows_cpp(
        x$indices_path, x$distances_path, x$nrow, x$ncol,
        x$n_reference, first, count, 128e6
    )
}

massive_knn_audit_identity <- function(knn, query, reference, count) {
    if (!inherits(knn, "fastEmbedR_massive_knn") ||
        !inherits(query, "fastEmbedR_massive_matrix") ||
        !inherits(reference, "fastEmbedR_massive_matrix") ||
        query$format == "memory" || reference$format == "memory" ||
        is.na(count) || count < 1L ||
        count > min(knn$nrow, 10000L) ||
        knn$nrow != query$nrow || knn$n_reference != reference$nrow ||
        query$ncol != reference$ncol) stop(
        "Recall audit needs matching file-backed query and reference.",
        call. = FALSE)
    list(query = massive_checkpoint_source_identity(query),
        reference = massive_checkpoint_source_identity(reference),
        output = massive_checkpoint_file_identity(
            c(knn$indices_path, knn$distances_path)))
}

massive_knn_audit_verify <- function(knn, query, reference,
                                        identity) {
    current <- massive_knn_audit_identity(knn, query, reference, 1L)
    if (!identical(identity, current) ||
        !identical(identity$query, knn$source_identity) ||
        !identical(identity$reference, knn$reference_identity) ||
        !identical(identity$output, knn$output_identity)) stop(
        "Recall audit inputs or neighbor files changed since fitting.",
        call. = FALSE)
}

#' EXPERIMENTAL measured recall for file-backed landmark neighbors
#'
#' Compares evenly spaced query rows with an independent streamed exact
#' CPU search. Sampled queries share bounded reference scans; the number
#' of passes is reported in `audit_reference_passes`. This audit can
#' still be expensive and does not certify unsampled rows or change the
#' saved neighbor files. `observed_recall` is strict ID overlap;
#' `distance_equivalent_recall` also accepts ties at the exact kth distance.
#'
#' @param knn A result of [massive_landmark_knn()].
#' @param query,reference Original file-backed matrix descriptors.
#' @param sample_rows Number of query rows to audit (at most 10,000).
#' @param memory_limit RAM budget for exact-search buffers.
#' @return `knn` with sampled mean and minimum row recall, a count of
#'   sampled rows below the target, and sampled row IDs in metadata.
#' @export
massive_audit_landmark_knn <- function(knn, query, reference,
                                        sample_rows = 32L,
                                        memory_limit = "1GB") {
    if (is.list(reference) &&
        inherits(reference$data, "fastEmbedR_massive_matrix")) {
        reference <- reference$data
    }
    count <- integer_scalar(sample_rows)
    identity <- massive_knn_audit_identity(knn, query, reference, count)
    massive_knn_audit_verify(knn, query, reference, identity)
    resources <- massive_full_knn_resources(query, knn$ncol, count,
        NULL, memory_limit, reference)
    rows <- floor((seq_len(count) - 0.5) * knn$nrow / count) + 1
    recall <- numeric(count)
    distance_recall <- numeric(count)
    started <- proc.time()[[3L]]
    groups <- split(seq_along(rows), ceiling(seq_along(rows) /
        resources$chunk_rows))
    message("EXPERIMENTAL landmark KNN recall audit: ", count,
        " rows in ", length(groups), " reference pass(es)")
    for (group in groups) {
        exact <- massive_exact_sample_batch_cpp(query, reference,
            rows[group], knn$ncol,
            resources$reference_chunk_rows, 1L, FALSE)
        if (!identical(exact$backend_used,
            "native_cpu_exact_stream") || !isTRUE(exact$exact)) stop(
            "Recall reference changed algorithm.", call. = FALSE)
        for (i in seq_along(group)) {
            observed <- massive_read_knn_rows(knn,
                rows[group[[i]]], 1L)
            recall[group[[i]]] <- sum(
                observed$indices[1L, ] %in%
                exact$indices[i, ]) / knn$ncol
            distance_recall[group[[i]]] <-
                massive_recall_distance_row(exact$distances[i, ],
                    observed$distances[1L, ])
        }
    }
    massive_knn_audit_verify(knn, query, reference, identity)
    knn$recall_audited <- TRUE
    knn$audit_reference_backend <- "cpu_exact_stream"
    knn$audit_sample_rows <- rows
    knn$audit_reference_passes <- length(groups)
    stats <- massive_recall_summary(recall, knn$target_recall)
    knn[names(stats)] <- stats
    knn$distance_equivalent_recall <- mean(distance_recall)
    knn$distance_equivalent_row_recall <- distance_recall
    knn$audit_seconds <- unname(proc.time()[[3L]] - started)
    knn
}

massive_recall_summary <- function(recall, target) {
    list(observed_recall = mean(recall),
        minimum_row_recall = min(recall),
        sampled_rows_below_target = sum(recall < target),
        sample_target_met = all(recall >= target))
}

massive_recall_distance_row <- function(exact, observed) {
    cutoff <- max(exact)
    tolerance <- 1e-5 * max(1, cutoff)
    mean(is.finite(observed) & observed >= -tolerance &
        observed <= cutoff + tolerance)
}

#' @export
print.fastEmbedR_massive_knn <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR landmark KNN\n")
    cat("  queries: ", format(x$nrow, scientific = FALSE),
        "; landmarks: ", format(x$n_reference, scientific = FALSE),
        "; k: ", x$ncol, "\n", sep = "")
    cat("  backend: ", x$backend, "; method: ", x$method,
        "\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_knn <- function(x, n = 6L, ...) {
    massive_read_knn_rows(x, n = min(n, x$nrow))
}

massive_full_knn_resources <- function(x, k, chunk_rows,
        reference_chunk_rows, memory_limit, reference_source = x) {
    limit <- massive_memory_bytes(memory_limit)
    reference <- reference_chunk_rows %||% 8192L
    query <- chunk_rows %||% 256L
    valid <- function(value) is.numeric(value) && length(value) == 1L &&
        is.finite(value) && value >= 1 && value == floor(value) &&
        value <= .Machine$integer.max
    if (!valid(reference) || !valid(query)) {
        stop("KNN chunk sizes must be positive integer counts.",
            call. = FALSE)
    }
    resident <- if (x$format == "memory") {
        x$nrow * x$ncol * 8
    } else 0
    fixed <- resident + 64 * 1024^2
    reference <- as.integer(min(reference,
        floor((0.7 * limit - fixed) / (4 * reference_source$ncol)),
        floor(256e6 / (4 * reference_source$ncol))))
    query <- as.integer(min(query,
        floor((0.7 * limit - fixed - reference * x$ncol * 4) /
            (4 * x$ncol + 40 * k)),
        floor(256e6 / (4 * x$ncol)),
        floor(128e6 / (12 * k))))
    if (is.na(reference) || is.na(query) ||
        reference < 1L || query < 1L) {
        stop("Exact streamed KNN buffers exceed `memory_limit`.",
            call. = FALSE)
    }
    list(chunk_rows = query, reference_chunk_rows = reference,
        peak_ram_bytes = fixed + reference * x$ncol * 4 +
            query * (4 * x$ncol + 40 * k),
        output_bytes = as.double(x$nrow) * k * 8,
        memory_limit_bytes = limit)
}

massive_full_knn_checkpoint <- function(x, setup, resume) {
    path <- paste0(setup$output, ".checkpoint.rds")
    identity <- list(version = 2L,
        implementation = massive_checkpoint_implementation_identity(),
        source = massive_checkpoint_source_identity(x),
        output = setup$output, k = setup$k,
        query_rows = setup$resources$chunk_rows,
        reference_rows = setup$resources$reference_chunk_rows,
        workers = setup$workers, backend = "cpu",
        method = setup$method,
        checkpoint_every = setup$checkpoint_every)
    if (!is.null(setup$posting_identity)) {
        identity$posting <- setup$posting_identity
    }
    signature <- massive_checkpoint_signature(identity)
    if (!resume) {
        if (file.exists(path)) {
            stop("Full-graph checkpoint exists; use `resume = TRUE`.",
                call. = FALSE)
        }
        state <- list(signature = signature, completed_rows = 0)
        massive_checkpoint_write(state, path)
        return(list(state = state, path = path))
    }
    state <- tryCatch(readRDS(path), error = function(e) NULL)
    rows <- state$completed_rows
    parts <- paste0(setup$paths, ".part")
    sizes <- if (all(file.exists(parts))) {
        file.info(parts)$size
    } else c(0, 0)
    if (!is.list(state) || !identical(state$signature, signature) ||
        !is.numeric(rows) || length(rows) != 1L ||
        !is.finite(rows) || rows != floor(rows) ||
        rows < 0 || rows > x$nrow ||
        (rows < x$nrow && rows %% setup$resources$chunk_rows != 0) ||
        (rows > 0 && !all(file.exists(parts))) ||
        anyNA(sizes) || any(sizes < rows * setup$k * 4) ||
        any(sizes > x$nrow * setup$k * 4)) {
        stop("Full-graph checkpoint or partial files do not match.",
            call. = FALSE)
    }
    list(state = state, path = path)
}

massive_full_knn_commit <- function(batch, streams, completed, saved,
        setup, posting_bytes = NULL, save_now = TRUE) {
    if (!identical(batch$backend_used, setup$backend_used) ||
        !identical(batch$exact, setup$exact)) {
        stop("Full-graph backend metadata is invalid.",
            call. = FALSE)
    }
    writeBin(as.integer(t(batch$indices)), streams$indices,
        size = 4L, endian = "little")
    writeBin(as.vector(t(batch$distances)), streams$distances,
        size = 4L, endian = "little")
    if (is.null(saved) || !save_now) return(saved)
    flush(streams$indices)
    flush(streams$distances)
    saved$state$completed_rows <- completed
    if (!is.null(posting_bytes))
        saved$state$posting_read_bytes <- posting_bytes
    massive_checkpoint_write(saved$state, saved$path)
    saved
}

massive_full_knn_assert_source <- function(x, identity) {
    if (!is.null(identity) && !identical(identity,
        massive_checkpoint_source_identity(x))) stop(
        "Full-graph source changed during search.",
        call. = FALSE)
}

massive_full_knn_finish_stream <- function(indices, distances,
        parts, setup, rows, measurement) {
    truncate(indices)
    truncate(distances)
    close(indices)
    close(distances)
    if (is.function(setup$before_finish)) {
        setup$before_finish(parts, measurement)
    }
    massive_knn_finish(NULL, NULL, parts, setup$paths,
        rows, setup$k)
}

massive_full_knn_stream <- function(x, setup, saved) {
    parts <- stats::setNames(paste0(unname(setup$paths), ".part"),
        names(setup$paths))
    completed <- saved$state$completed_rows %||% 0
    streams <- massive_knn_open_parts(parts, completed, setup$k)
    committed <- FALSE
    on.exit({
        try(close(streams$indices), silent = TRUE)
        try(close(streams$distances), silent = TRUE)
        if (!committed && is.null(saved)) unlink(parts)
    })
    identity <- if (is.null(x$path)) NULL else
        massive_checkpoint_source_identity(x)
    started <- proc.time()[[3L]]
    posting_bytes <- saved$state$posting_read_bytes %||%
        if (completed > 0 && !is.null(setup$posting_identity))
            NA_real_ else 0
    first <- completed + 1
    reported <- -1L
    while (first <= x$nrow) {
        massive_full_knn_assert_source(x, identity)
        rows <- as.integer(min(setup$resources$chunk_rows,
            x$nrow - first + 1))
        batch <- setup$search_batch(first, rows)
        posting_bytes <- posting_bytes +
            (batch$posting_read_bytes %||% 0)
        massive_full_knn_assert_source(x, identity)
        first <- first + rows
        done <- first - 1
        save_now <- done == x$nrow ||
            ceiling(done / setup$resources$chunk_rows) %%
                setup$checkpoint_every == 0
        saved <- massive_full_knn_commit(batch, streams, first - 1,
            saved, setup, posting_bytes, save_now)
        progress <- floor(20 * done / x$nrow)
        if (progress > reported) {
            message("EXPERIMENTAL full KNN: ", done, "/",
                x$nrow, " rows")
            reported <- progress
        }
    }
    measurement <- list(seconds = unname(proc.time()[[3L]] - started),
        posting_read_bytes = posting_bytes)
    massive_full_knn_finish_stream(streams$indices,
        streams$distances, parts, setup, x$nrow, measurement)
    committed <- TRUE
    if (!is.null(saved)) unlink(saved$path)
    measurement
}

massive_hnsw_sharded_resources <- function(x, k, chunk_rows,
        reference_chunk_rows, memory_limit) {
    if (x$format == "memory") stop(
        "Sharded HNSW needs a file-backed matrix.", call. = FALSE)
    limit <- massive_memory_bytes(memory_limit)
    per_reference <- 16 * x$ncol + 2048
    max_reference <- floor((0.7 * limit - 64 * 1024^2) /
        per_reference)
    requested <- reference_chunk_rows %||% min(x$nrow, 250000L)
    if (is.na(integer_scalar(requested)) || requested < 2L ||
        max_reference < 2) stop(
        "HNSW reference shard exceeds `memory_limit`.", call. = FALSE)
    shard_rows <- min(requested, max_reference, x$nrow)
    shards <- min(ceiling(x$nrow / shard_rows), floor(x$nrow / 2))
    shard_rows <- ceiling(x$nrow / shards)
    if (shard_rows > min(requested, max_reference)) stop(
        "Balanced HNSW shards exceed the reference-row limit.",
        call. = FALSE)
    fixed <- 64 * 1024^2 + shard_rows * per_reference
    per_query <- 12 * x$ncol + 16 * (k + 1) + 8 * k
    maximum <- floor((0.7 * limit - fixed) / per_query)
    requested <- chunk_rows %||% 8192L
    if (is.na(integer_scalar(requested)) || requested < 1L ||
        maximum < 1) stop(
        "HNSW query buffers exceed `memory_limit`.", call. = FALSE)
    rows <- as.integer(min(requested, maximum,
        floor(128e6 / (8 * x$ncol)),
        floor(128e6 / (12 * (k + 1)))))
    if (is.na(rows) || rows < 1L) stop(
        "HNSW query batch cannot fit in R.", call. = FALSE)
    list(chunk_rows = rows, reference_chunk_rows = shard_rows,
        n_shards = as.integer(shards),
        query_batches = ceiling(x$nrow / rows) * shards,
        minimum_query_read_bytes =
            x$nrow * x$ncol * 4 * shards,
        peak_ram_bytes = fixed + rows * per_query,
        output_bytes = as.double(x$nrow) * k * 8,
        memory_limit_bytes = limit)
}

massive_cuda_sharded_resources <- function(x, k, chunk_rows,
        reference_chunk_rows, memory_limit, query_rows = x$nrow,
        exclude_self = TRUE, method = "exact") {
    if (x$format == "memory") stop(
        "Sharded CUDA search needs a file source.", call. = FALSE)
    if (!native_cuda_knn_available_cpp()) stop(
        "Full CUDA KNN requires native cuVS; no CPU fallback was used.",
        call. = FALSE)
    if (exclude_self && k > 255L) stop("CUDA k exceeds 255.", call. = FALSE)
    info <- native_cuda_memory_info_cpp()
    limit <- massive_memory_bytes(memory_limit)
    requested <- reference_chunk_rows %||% min(x$nrow, 250000L)
    requested_query <- chunk_rows %||% 8192L
    if (is.na(integer_scalar(requested)) || requested < 2L ||
        is.na(integer_scalar(requested_query)) ||
        requested_query < 1L) stop(
        "CUDA shard sizes must be positive integer counts.",
        call. = FALSE)
    fixed <- 128 * 1024^2
    index_bytes <- if (method == "ivf_sharded") 12 else 8
    host_max <- floor((0.7 * limit - fixed) / (16 * x$ncol))
    gpu_max <- floor((0.65 * info$free_bytes - fixed) /
        (index_bytes * x$ncol))
    shard_rows <- min(requested, host_max, gpu_max, x$nrow)
    if (!is.finite(shard_rows) || shard_rows < 2L) stop(
        "CUDA reference shard exceeds RAM or free VRAM budget.",
        call. = FALSE)
    shards <- min(ceiling(x$nrow / shard_rows), floor(x$nrow / 2))
    shard_rows <- ceiling(x$nrow / shards)
    if (shard_rows > min(requested, host_max, gpu_max))
        stop("Balanced CUDA shards exceed the limit.", call. = FALSE)
    host_free <- 0.7 * limit - fixed - 16 * shard_rows * x$ncol
    gpu_free <- 0.65 * info$free_bytes - fixed -
        index_bytes * shard_rows * x$ncol
    per_query <- 12 * x$ncol + 32 * (k + 1)
    rows <- as.integer(min(requested_query, floor(host_free / per_query),
        floor(gpu_free / per_query), floor(128e6 / (8 * x$ncol)),
        floor(128e6 / (12 * (k + 1)))))
    if (is.na(rows) || rows < 1L) stop(
        "CUDA query batch exceeds memory budget.", call. = FALSE)
    list(chunk_rows = rows, reference_chunk_rows = shard_rows,
        n_shards = as.integer(shards),
        query_batches = ceiling(query_rows / rows) * shards,
        minimum_query_read_bytes = query_rows * x$ncol * 4 * shards,
        peak_ram_bytes = fixed + 16 * shard_rows * x$ncol + rows * per_query,
        peak_vram_bytes = fixed + index_bytes * shard_rows * x$ncol +
            rows * per_query,
        free_vram_bytes = info$free_bytes, gpu_device = info$device,
        output_bytes = query_rows * k * 8, memory_limit_bytes = limit)
}

massive_metal_sharded_resources <- function(x, k, chunk_rows,
        reference_chunk_rows, memory_limit) {
    if (x$format == "memory") stop(
        "Sharded Metal search needs a file source.", call. = FALSE)
    if (!native_metal_knn_available_cpp()) stop(
        "Full Metal KNN requires native Metal; no CPU fallback.",
        call. = FALSE)
    if (x$ncol > 16384L || k > 127L) stop(
        "Metal exact KNN needs at most 16,384 columns and k <= 127.",
        call. = FALSE)
    requested <- reference_chunk_rows %||% min(x$nrow, 250000L)
    query <- chunk_rows %||% 8192L
    if (is.na(integer_scalar(requested)) || requested < 2L ||
        is.na(integer_scalar(query)) || query < 1L) stop(
        "Metal shard sizes must be positive integer counts.",
        call. = FALSE)
    limit <- massive_memory_bytes(memory_limit)
    fixed <- 64 * 1024^2
    max_ref <- floor(min((0.7 * limit - fixed) /
        (20 * x$ncol), 128 * 1024^2 / (4 * x$ncol)))
    shard_rows <- min(requested, max_ref, x$nrow)
    if (!is.finite(shard_rows) || shard_rows < 2L) stop(
        "Metal reference shard exceeds the RAM budget.", call. = FALSE)
    shards <- min(ceiling(x$nrow / shard_rows), floor(x$nrow / 2))
    shard_rows <- ceiling(x$nrow / shards)
    if (shard_rows > min(requested, max_ref)) stop(
        "Balanced Metal shards exceed the RAM budget.", call. = FALSE)
    per_query <- 20 * x$ncol + 32 * (k + 1)
    rows <- as.integer(min(query, floor((0.7 * limit - fixed -
        20 * shard_rows * x$ncol) / per_query),
        floor(64 * 1024^2 / per_query)))
    if (is.na(rows) || rows < 1L) stop(
        "Metal query batch exceeds the RAM budget.", call. = FALSE)
    list(chunk_rows = rows, reference_chunk_rows = shard_rows,
        n_shards = as.integer(shards),
        query_batches = ceiling(x$nrow / rows) * shards,
        minimum_query_read_bytes = as.double(x$nrow) * x$ncol *
            4 * shards,
        peak_ram_bytes = fixed + 20 * shard_rows * x$ncol +
            rows * per_query,
        metal_buffer_bytes_estimate = 4 * shard_rows * x$ncol +
            rows * (4 * x$ncol + 8 * (k + 1)),
        metal_memory_model = "unified_shared",
        output_bytes = as.double(x$nrow) * k * 8,
        memory_limit_bytes = limit)
}

massive_sharded_checkpoint <- function(x, setup, paths, backend,
        resume, source_identity = massive_checkpoint_source_identity(x)) {
    path <- sub("\\.indices\\.u32$", ".checkpoint.rds", paths[["indices"]])
    signature <- massive_checkpoint_signature(Filter(Negate(is.null), list(
        version = if (backend == "cpu") 4L else 2L, source = source_identity,
        implementation = massive_checkpoint_implementation_identity(),
        faissr = if (backend == "cpu") massive_faissr_identity() else NULL,
        ef_search = setup$ef_search %||% 0L, hnsw_build = setup$hnsw_build,
        k = setup$k, backend = backend, method = setup$method,
        workers = setup$workers, build_threads =
            setup$build_workers %||% setup$workers,
        query_rows = setup$resources$chunk_rows,
        reference_rows = setup$resources$reference_chunk_rows,
        n_shards = setup$resources$n_shards,
        memory_limit = setup$resources$memory_limit_bytes,
        checkpoint_every = setup$checkpoint_every, gpu_device =
            setup$resources$gpu_device %||% NA_integer_)))
    if (!resume) {
        if (file.exists(path)) stop("Sharded checkpoint exists; use resume.")
        state <- list(signature = signature, shard = 0L,
            completed_rows = 0, pilot_signature = NULL)
        massive_checkpoint_write(state, path)
        return(list(path = path, state = state))
    }
    state <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(state)) stop("Sharded KNN checkpoint is unreadable.")
    exists <- file.exists(paste0(paths, ".part"))
    sizes <- if (all(exists)) file.info(paste0(paths, ".part"))$size else 0
    full <- as.double(x$nrow) * setup$k * 4
    rows <- state$completed_rows
    shard <- state$shard
    valid <- identical(state$signature, signature) &&
        is.numeric(shard) && length(shard) == 1L &&
        is.finite(shard) && shard == floor(shard) &&
        shard >= 0 && shard <= setup$resources$n_shards &&
        is.numeric(rows) && length(rows) == 1L &&
        is.finite(rows) && rows == floor(rows) &&
        rows >= 0 && rows <= x$nrow &&
        (rows == x$nrow || rows %% setup$resources$chunk_rows == 0) &&
        (rows == 0 || (is.character(state$pilot_signature) &&
            length(state$pilot_signature) == 1L)) &&
        (shard < setup$resources$n_shards || rows == 0) &&
        (!any(exists) || all(exists)) && !anyNA(sizes) &&
        all(sizes >= if (shard == 0) as.double(rows) *
            setup$k * 4 else full) &&
        all(sizes <= full) && (shard == 0 || all(exists)) &&
        (rows == 0 || all(exists))
    if (!isTRUE(valid)) stop("Sharded KNN checkpoint or files do not match.")
    list(path = path, state = state)
}

massive_sharded_restore_first <- function(parts, completed, k) {
    bytes <- completed * k * 4
    for (path in parts[file.exists(parts)]) {
        stream <- file(path, "r+b")
        seek(stream, where = bytes, origin = "start", rw = "write")
        truncate(stream)
        close(stream)
    }
}

massive_sharded_probe <- function(x, state, setup, saved, method,
        backend, count) {
    if (is.null(saved) || saved$state$completed_rows == 0) return()
    rows <- as.integer(min(setup$resources$chunk_rows, x$nrow))
    query <- massive_sharded_query(x, 1L, rows, method, backend)
    found <- massive_knn_query(state, query,
        as.integer(min(setup$k + 1, count)),
        method, setup$workers, backend, setup$ef_search %||% 0L)
    if (!identical(massive_knn_probe_signature(found),
        saved$state$pilot_signature)) stop(
        "Rebuilt sharded KNN index differs from checkpoint pilot; ",
        "partial output was not changed.", call. = FALSE)
}

massive_sharded_query <- function(x, first, rows, method,
        backend = "cpu") {
    view <- massive_matrix_rows(x, first, rows)
    if (method == "faiss_hnsw" || backend == "metal") return(
        massive_read_rows_cpp(
        view, 1L, rows, 4 * rows * x$ncol, TRUE))
    massive_reference_buffer_cpp(view)
}

massive_saved_ivf_setting <- function(saved, shard) {
    if (length(saved$state$ivf_settings) <= shard) return(NULL)
    saved$state$ivf_settings[[shard + 1L]]
}

massive_sharded_batch <- function(x, setup, parts, state, shard,
        first, count, query_first, method, backend, saved) {
    rows <- as.integer(min(setup$resources$chunk_rows,
        x$nrow - query_first + 1))
    query <- massive_sharded_query(x, query_first, rows, method,
        backend)
    local_k <- as.integer(min(setup$k + 1, count))
    found <- massive_knn_query(state, query, local_k,
        method, setup$workers, backend, setup$ef_search %||% 0L)
    valid <- if (backend == "metal") {
        identical(found$backend, "metal") &&
            identical(found$method, "native_metal_exact_query") &&
            isTRUE(found$exact)
    } else if (backend == "cuda") {
        expected <- if (method == "ivf")
            "native_cuda_cuvs_ivf_flat" else "native_cuda_cuvs_exact"
        identical(found$backend, expected) &&
            identical(found$method, method)
    } else identical(found$backend, "cpu") &&
        identical(found$method, "faissr_hnsw_query_reused") &&
        isTRUE(found$index_reused)
    if (!valid) stop("Sharded KNN changed algorithm or backend.",
        call. = FALSE)
    ivf_setting <- if (method == "ivf") {
        massive_sharded_ivf_setting(found, shard, first, count)
    } else NULL
    if (!is.null(saved) && !is.null(ivf_setting)) {
        previous <- massive_saved_ivf_setting(saved, shard)
        if (!is.null(previous) && !identical(previous, ivf_setting))
            stop("Checkpointed IVF settings changed.", call. = FALSE)
        saved$state$ivf_settings[[shard + 1L]] <- ivf_setting
    }
    massive_merge_knn_shard_cpp(parts[["indices"]],
        parts[["distances"]], found$indices, found$distances,
        query_first, first, count, x$nrow, setup$k, shard == 0L)
    if (!is.null(saved)) {
        if (is.null(saved$state$pilot_signature)) {
            saved$state$pilot_signature <-
                massive_knn_probe_signature(found)
        }
        done <- query_first + rows - 1
        batch_number <- ceiling(done / setup$resources$chunk_rows)
        if (done == x$nrow ||
            batch_number %% setup$checkpoint_every == 0) {
            saved$state$completed_rows <- done
            massive_checkpoint_write(saved$state, saved$path)
        }
    }
    list(rows = rows, saved = saved, ivf_setting = ivf_setting)
}

massive_sharded_ivf_setting <- function(found, shard, first, count) {
    values <- found[c("nlist", "nprobe", "pilot_rows",
        "tuning_attempts")]
    minimum <- found$pilot_min_row_recall
    if (length(values) != 4L || any(vapply(values, function(value)
        length(value) != 1L || !is.numeric(value) ||
            !is.finite(value) || value < 0 ||
            value != floor(value), logical(1L))) ||
        values$nlist < 1L || values$nprobe < 1L ||
        values$nprobe > values$nlist ||
        length(minimum) != 1L || !is.numeric(minimum) ||
        !is.finite(minimum) || minimum < 0.99 || minimum > 1) stop(
        "Sharded IVF query omitted effective tuning settings.",
        call. = FALSE)
    c(list(shard = shard + 1L, first_row = first,
        reference_rows = count), values,
        list(pilot_recall = found$pilot_recall,
            pilot_id_recall = found$pilot_id_recall,
            pilot_min_row_recall = minimum,
            pilot_recall_metric = found$pilot_recall_metric))
}

massive_hnsw_add_setting <- function(settings, observed) {
    if (is.null(observed)) return(settings)
    c(settings, list(observed))
}

massive_sharded_hnsw_setting <- function(state, setup, shard,
        first, count, backend) {
    if (backend != "cpu") return(NULL)
    list(shard = shard + 1L, first_row = first,
        reference_rows = count, m = state$index$M,
        ef_construction = state$index$efConstruction,
        ef_search = if ((setup$ef_search %||% 0L) > 0L)
            setup$ef_search else state$index$efSearch,
        tuning_rule = state$index$tuning_rule,
        build_threads = state$build_threads)
}

massive_index_shard <- function(x, setup, parts, shard,
        source_identity, backend, saved) {
    base <- floor(x$nrow / setup$resources$n_shards)
    extra <- x$nrow %% setup$resources$n_shards
    first <- 1 + shard * base + min(shard, extra)
    count <- base + as.integer(shard < extra)
    reference <- massive_matrix_rows(x, first, count)
    tuning_k <- as.integer(min(setup$k + (backend != "cpu"), count))
    method <- if (setup$method == "ivf_sharded") "ivf" else
        if (backend == "cpu") "faiss_hnsw" else "exact"
    state <- massive_knn_reference(reference, tuning_k,
        method, setup$build_workers %||% setup$workers, backend,
        hnsw_build = setup$hnsw_build)
    if (backend == "cpu" &&
        !identical(state$index$method, "faissr_hnsw_persistent")) stop(
        "Expected CPU HNSW index was not built.", call. = FALSE)
    hnsw_settings <- massive_sharded_hnsw_setting(state, setup,
        shard, first, count, backend)
    ivf_setting <- massive_saved_ivf_setting(saved, shard)
    massive_sharded_probe(x, state, setup, saved, method,
        backend, count)
    if (shard == 0L && !is.null(saved))
        massive_sharded_restore_first(parts,
            saved$state$completed_rows, setup$k)
    query_first <- (saved$state$completed_rows %||% 0) + 1
    while (query_first <= x$nrow) {
        if (!identical(source_identity,
            massive_checkpoint_source_identity(x))) stop(
            "Full-graph source changed during sharded search.",
            call. = FALSE)
        batch <- massive_sharded_batch(x, setup, parts, state,
            shard, first, count, query_first, method, backend, saved)
        saved <- batch$saved
        if (is.null(ivf_setting)) ivf_setting <- batch$ivf_setting
        query_first <- query_first + batch$rows
    }
    if (method == "ivf" && is.null(ivf_setting)) stop(
        "Sharded IVF result has no effective settings.", call. = FALSE)
    if (!is.null(saved)) {
        saved$state$hnsw_settings <- massive_hnsw_add_setting(
            saved$state$hnsw_settings %||% list(), hnsw_settings)
        saved$state$shard <- shard + 1L
        saved$state$completed_rows <- 0
        saved$state$pilot_signature <- NULL
        massive_checkpoint_write(saved$state, saved$path)
    }
    message("KNN shard ", shard + 1L, "/", setup$resources$n_shards)
    list(build_seconds = state$build_seconds, saved = saved,
        hnsw_settings = hnsw_settings, ivf_setting = ivf_setting)
}

massive_sharded_result <- function(x, paths, setup, backend,
        identity, build_seconds, started, audit_rows, checkpoint,
        resume, hnsw_settings, ivf_settings) {
    graph <- massive_knn_graph(sub("\\.indices\\.u32$", "",
        paths[["indices"]]), x$nrow, setup$k)
    graph$backend <- backend
    graph$method <- if (setup$method == "ivf_sharded") {
        "native_cuda_cuvs_ivf_sharded"
    } else if (backend == "cuda") {
        "native_cuda_cuvs_exact_sharded"
    } else if (backend == "metal") {
        "native_metal_exact_sharded"
    } else "faissr_cpu_hnsw_sharded"
    graph$exact <- backend != "cpu" && setup$method == "exact"
    graph$spanning_links <- FALSE
    graph$ef_search_requested <- setup$ef_search
    graph$hnsw_build_requested <- setup$hnsw_build
    graph$hnsw_settings <- hnsw_settings
    graph$ivf_settings <- ivf_settings
    graph$target_recall <- if (backend == "cpu" ||
            setup$method == "ivf_sharded") {
        0.99
    } else NA_real_
    graph$recall_audited <- FALSE
    graph$query_storage <- if (backend != "cuda") "R_float32" else
        "native_float32"
    graph$source_identity <- identity
    graph$implementation_identity <-
        massive_checkpoint_implementation_identity()
    graph$faissr_identity <- if (backend == "cpu")
        massive_faissr_identity() else NULL
    graph$resources <- setup$resources
    graph$build_seconds <- build_seconds
    graph$total_seconds <- unname(proc.time()[[3L]] - started)
    graph$search_seconds <- graph$total_seconds - build_seconds
    graph$checkpoint <- checkpoint
    graph$checkpoint_every <- setup$checkpoint_every
    graph$resumed <- resume
    graph$timing_scope <- "current_invocation"
    massive_full_knn_audit(graph, x, audit_rows, setup)
}

massive_sharded_graph <- function(x, k, output, n.cores,
        chunk_rows, reference_chunk_rows, memory_limit,
        audit_rows, backend, method, checkpoint, resume,
        checkpoint_every, ef_search, hnsw_build) {
    if (backend == "metal" && normalize_nn_threads(n.cores) != 1L)
        stop("Metal full KNN requires `n.cores = 1`.", call. = FALSE)
    resources <- if (backend == "cuda") {
        massive_cuda_sharded_resources(x, k, chunk_rows,
            reference_chunk_rows, memory_limit, method = method)
    } else if (backend == "metal") {
        massive_metal_sharded_resources(x, k, chunk_rows,
            reference_chunk_rows, memory_limit)
    } else massive_hnsw_sharded_resources(x, k,
        chunk_rows, reference_chunk_rows, memory_limit)
    if (ef_search > x$nrow / resources$n_shards)
        stop("`ef_search` exceeds the smallest reference shard.")
    paths <- massive_knn_paths(output, resources$output_bytes, resume)
    parts <- stats::setNames(paste0(unname(paths), ".part"), names(paths))
    setup <- list(k = k, workers = normalize_nn_threads(n.cores),
        resources = resources, method = method, ef_search = ef_search,
        hnsw_build = hnsw_build, checkpoint_every = checkpoint_every,
        build_workers = if (checkpoint && backend == "cpu") 1L else
            normalize_nn_threads(n.cores))
    saved <- if (checkpoint) massive_sharded_checkpoint(
        x, setup, paths, backend, resume) else NULL
    identity <- massive_checkpoint_source_identity(x)
    completed <- FALSE
    on.exit(if (!completed && is.null(saved)) unlink(parts))
    started <- proc.time()[[3L]]
    build_seconds <- 0
    hnsw <- saved$state$hnsw_settings %||% list()
    ivf <- saved$state$ivf_settings %||% list()
    start_shard <- saved$state$shard %||% 0L
    if (start_shard < resources$n_shards)
        for (shard in seq.int(start_shard, resources$n_shards - 1L)) {
            step <- massive_index_shard(x, setup, parts, shard,
                identity, backend, saved)
            build_seconds <- build_seconds + step$build_seconds
            hnsw <- massive_hnsw_add_setting(hnsw, step$hnsw_settings)
            if (length(step$ivf_setting)) ivf[[shard + 1L]] <- step$ivf_setting
            saved <- step$saved
        }
    if (!identical(identity, massive_checkpoint_source_identity(x)))
        stop("Full-graph source changed during sharded search.")
    massive_knn_finish(NULL, NULL, parts, paths, x$nrow, k)
    completed <- TRUE
    if (!is.null(saved)) unlink(saved$path)
    massive_sharded_result(x, paths, setup, backend, identity,
        build_seconds, started, audit_rows, checkpoint, resume, hnsw, ivf)
}

massive_full_knn_audit <- function(graph, x, audit_rows, setup,
        rows = NULL) {
    if (audit_rows == 0L) return(graph)
    if (is.null(rows)) rows <- as.integer(floor(
        (seq_len(audit_rows) - 0.5) * x$nrow / audit_rows) + 1)
    if (length(rows) != audit_rows || anyNA(rows) ||
        any(rows < 1L | rows > x$nrow) || anyDuplicated(rows)) stop(
        "Full-graph audit rows are invalid.", call. = FALSE)
    identity <- if (!is.null(x$path))
        massive_checkpoint_source_identity(x) else NULL
    recall <- numeric(audit_rows)
    distance_recall <- numeric(audit_rows)
    started <- proc.time()[[3L]]
    groups <- split(seq_along(rows), ceiling(seq_along(rows) /
        setup$resources$chunk_rows))
    message("Full KNN audit: ", audit_rows, " rows; ",
        length(groups), " reference passes")
    for (group in groups) {
        exact <- massive_exact_sample_batch_cpp(x, x, rows[group],
            graph$k, setup$resources$reference_chunk_rows,
            setup$workers, TRUE)
        if (!identical(exact$backend_used,
            "native_cpu_exact_stream") || !isTRUE(exact$exact)) stop(
            "Exact recall reference changed algorithm.", call. = FALSE)
        for (i in seq_along(group)) {
            observed <- massive_read_graph_edges(graph,
                rows[group[[i]]], 1L)
            recall[group[[i]]] <- sum(
                observed$to %in% exact$indices[i, ]) / graph$k
            distance_recall[group[[i]]] <-
                massive_recall_distance_row(exact$distances[i, ],
                    observed$distance)
        }
    }
    if (!is.null(identity) &&
        !identical(identity, massive_checkpoint_source_identity(x)))
        stop("Full-graph source changed during recall audit.",
            call. = FALSE)
    graph$recall_audited <- TRUE
    graph$audit_rows <- audit_rows
    graph$audit_sample_rows <- rows
    graph$audit_row_recall <- recall
    graph$distance_equivalent_recall <- mean(distance_recall)
    graph$distance_equivalent_row_recall <- distance_recall
    graph$audit_reference_passes <- length(groups)
    stats <- massive_recall_summary(recall, 0.99)
    graph[names(stats)] <- stats
    graph$audit_seconds <- unname(proc.time()[[3L]] - started)
    graph
}

massive_full_knn_exact <- function(x, k, output, n.cores,
        chunk_rows, reference_chunk_rows, memory_limit,
        checkpoint, resume, audit_rows, checkpoint_every) {
    workers <- normalize_nn_threads(n.cores)
    resources <- massive_full_knn_resources(x, k, chunk_rows,
        reference_chunk_rows, memory_limit)
    paths <- massive_knn_paths(output, resources$output_bytes, resume)
    setup <- list(output = sub("\\.indices\\.u32$", "", paths[["indices"]]),
        paths = paths, k = k, workers = workers, resources = resources,
        checkpoint_every = checkpoint_every, method = "exact",
        backend_used = "native_cpu_exact_stream", exact = TRUE)
    setup$search_batch <- function(first, rows) {
        massive_exact_graph_batch_cpp(x, first, rows, k,
            resources$reference_chunk_rows, workers)
    }
    saved <- if (checkpoint) massive_full_knn_checkpoint(
        x, setup, resume) else NULL
    message("EXPERIMENTAL full exact KNN: ", x$nrow, " rows; k=", k,
        "; CPU workers=", workers, "; query rows=", resources$chunk_rows,
        "; reference rows=", resources$reference_chunk_rows)
    measurement <- massive_full_knn_stream(x, setup, saved)
    graph <- massive_knn_graph(setup$output, x$nrow, k)
    graph$backend <- "cpu"
    graph$method <- "native_cpu_exact_stream"
    graph$exact <- TRUE
    graph$source_identity <- massive_checkpoint_source_identity(x)
    graph$implementation_identity <-
        massive_checkpoint_implementation_identity()
    graph$resources <- resources
    graph$search_seconds <- measurement$seconds
    graph$checkpoint <- checkpoint
    graph$checkpoint_every <- checkpoint_every
    massive_full_knn_audit(graph, x, audit_rows, setup)
}

massive_hnsw_build_controls <- function(value, backend, method) {
    if (is.null(value)) return(NULL)
    if (backend != "cpu" || method != "hnsw_sharded") stop(
        "`hnsw_build` requires CPU HNSW sharded search.")
    if (!is.list(value) || length(value) != 2L ||
        !setequal(names(value), c("m", "ef_construction"))) stop(
        "`hnsw_build` needs named `m` and `ef_construction`.")
    m <- integer_scalar(value$m)
    efc <- integer_scalar(value$ef_construction)
    if (is.na(m) || m < 2L || is.na(efc) || efc < m) stop(
        "Invalid `hnsw_build` settings.")
    list(m = m, ef_construction = efc)
}

massive_full_knn_controls <- function(x, k, backend, method,
        audit_rows, checkpoint_every, checkpoint, resume,
        query_order, ef_search, hnsw_build = NULL) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    method <- match.arg(method, c("exact", "hnsw_sharded",
        "ivf_sharded", "coarse_postings"))
    query_order <- match.arg(query_order, c("original", "posting"))
    backend <- match.arg(backend, c("cpu", "cuda", "metal"))
    k <- integer_scalar(k)
    audit_rows <- integer_scalar(audit_rows)
    ef_search <- integer_scalar(ef_search)
    checkpoint_every <- integer_scalar(checkpoint_every)
    if (is.na(checkpoint_every) || checkpoint_every < 1L) stop(
        "`checkpoint_every` must be a positive integer.")
    if (!inherits(x, "fastEmbedR_massive_matrix") || x$nrow < 2 ||
        x$nrow > .Machine$integer.max) stop(
        "Full KNN requires 2 to 2^31 - 1 matrix rows.")
    if (checkpoint) {
        origin <- x
        while (identical(origin$format, "view")) {
            origin <- origin$source
        }
        if (is.null(origin$path)) stop("Checkpoint needs a file source.")
    }
    if (is.na(k) || k < 1L || k >= x$nrow || k > 65536L)
        stop("`k` must be a valid non-self neighbor count.")
    if (is.na(audit_rows) || audit_rows < 0L ||
        audit_rows > min(x$nrow, 10000L)) stop(
        "`audit_rows` must be between zero and 10,000 rows.")
    if (is.na(ef_search) || ef_search < 0L ||
        (ef_search > 0L && ef_search < k + 1L)) stop(
        "`ef_search` must be zero or at least `k + 1`.")
    if (ef_search > 0L && method != "hnsw_sharded") stop(
        "`ef_search` requires CPU HNSW sharded search.")
    hnsw_build <- massive_hnsw_build_controls(
        hnsw_build, backend, method)
    if (backend == "cuda" && method == "hnsw_sharded")
        stop("CUDA HNSW unavailable; no CPU fallback.", call. = FALSE)
    if (backend == "cpu" && method == "ivf_sharded")
        stop("IVF-Flat needs CUDA; no CPU fallback.", call. = FALSE)
    if (backend == "metal" && method != "exact") stop(
        "File-backed Metal KNN supports exact shards only; ",
        "no algorithm fallback was used.", call. = FALSE)
    list(k = k, backend = backend, method = method,
        audit_rows = audit_rows, ef_search = ef_search,
        hnsw_build = hnsw_build,
        checkpoint_every = checkpoint_every,
        query_order = query_order)
}

#' Build an experimental file-backed full KNN graph
#'
#' Exact search scans all reference blocks for each query block.
#' `method = "hnsw_sharded"` builds one persistent faissR CPU HNSW index
#' per reference shard, searches all query blocks against it, and merges
#' candidates into the file-backed output. It is approximate; its target
#' recall is not measured unless `audit_rows` is positive. Multiple shard
#' passes can be I/O-heavy.
#' Checkpointed HNSW builds each shard with one thread for reproducible
#' resume; queries still use `n.cores` threads. Each shard records the
#' effective build thread count in `hnsw_settings`.
#' CUDA `method = "exact"` builds and queries bounded cuVS reference shards.
#' It also scans the full query source per shard, so this exact route is a
#' correctness baseline rather than a billion-row ANN solution. A CUDA
#' `method = "ivf_sharded"` uses the existing native cuVS IVF-Flat index
#' within each shard. It remains approximate across shards; request
#' `audit_rows` to measure sampled full-graph ID recall. Each per-shard
#' pilot gates both distance recall and minimum row ID recall; it does
#' not guarantee untouched rows. CUDA requests never fall back to CPU.
#' Metal exact search reads bounded float32 reference and query batches,
#' executes the existing native Metal query kernel, and merges global IDs
#' into the same file-backed graph. It re-uploads each reference shard for
#' every query batch; this is a correctness route, not an ANN index.
#'
#' @param x A `massive_matrix()` descriptor.
#' @param k Number of non-self neighbors, at most 65,536.
#' @param output New prefix for graph files.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"`. Metal supports exact
#'   shards only and never falls back to CPU.
#' @param n.cores CPU workers for each query block.
#' @param chunk_rows Maximum query rows per block; reduced if needed to
#'   respect `memory_limit`. Posting search defaults to at most 65,536
#'   rows per block to reduce repeated posting scans.
#' @param reference_chunk_rows Maximum reference rows per block.
#'   For sharded CPU HNSW or CUDA exact/IVF search, the requested maximum
#'   resident index-shard size.
#' @param memory_limit Conservative RAM budget.
#' @param method `"exact"`, `"hnsw_sharded"`, `"ivf_sharded"`, or
#'   `"coarse_postings"`. The posting route requires the original
#'   file-backed source, `postings`, and matching `centers`.
#' @param postings Physical posting descriptor for `"coarse_postings"`.
#' @param centers Centers used to construct `postings`.
#' @param nprobe Posting lists examined per row; defaults to at most eight.
#' @param query_order For posting search, `"original"` retains source-row
#'   order; `"posting"` scans grouped posting rows and externally restores
#'   original graph order with bounded memory and additional disk space.
#' @param checkpoint Save completed query blocks, including sharded search.
#'   Multi-GPU IVF checkpoints completed GPU query shards only.
#' @param resume Continue a matching checkpoint and partial graph.
#'   An interrupted multi-GPU IVF shard restarts from its first query row.
#' @param checkpoint_every Commit every this many query blocks. A shard
#'   boundary is always committed. Defaults to 100.
#' @param audit_rows Query rows to compare with streamed exact search
#'   (maximum 10,000). Posting search also samples small posting lists
#'   and distant-from-center rows; other methods use evenly spaced rows.
#'   Multi-GPU IVF excludes each device's first tuning query batch and
#'   samples only untouched rows.
#'   Defaults to zero. Reported recall covers only sampled rows, not
#'   the full graph. Strict ID `sample_target_met` requires every sampled
#'   row to meet 0.99. `distance_equivalent_recall` also accepts exact
#'   kth-distance ties; the CUDA IVF pilot additionally gates strict IDs.
#' @param ef_search CPU HNSW search effort per shard. Zero uses the tuned
#'   shape-based default. A positive value must be at least `k + 1` and
#'   cannot exceed the smallest reference shard. Other methods reject it.
#' @param hnsw_build Optional named list with `m` and `ef_construction` for
#'   an explicitly calibrated CPU HNSW build. `NULL` uses faissR's
#'   shape-based policy. Other methods and backends reject this setting.
#' @param devices Optional CUDA device indices. Two or more devices require
#'   exact or IVF sharded search and a file-backed source. CPU and Metal
#'   requests fail explicitly; the graph is never moved to another route.
#' @return A validated `fastEmbedR_massive_graph` descriptor.
#' @export
massive_full_knn_graph <- function(x, k, output, backend = "cpu",
        n.cores = 1L, chunk_rows = NULL, reference_chunk_rows = NULL,
        memory_limit = "8GB", checkpoint = FALSE, resume = FALSE,
        method = c("exact", "hnsw_sharded", "ivf_sharded",
            "coarse_postings"),
        audit_rows = 0L, checkpoint_every = 100L,
        postings = NULL, centers = NULL, nprobe = NULL,
        query_order = c("original", "posting"), ef_search = 0L,
        hnsw_build = NULL, devices = NULL) {
    controls <- massive_full_knn_controls(x, k, backend, method,
        audit_rows, checkpoint_every, checkpoint, resume,
        query_order, ef_search, hnsw_build)
    k <- controls$k
    backend <- controls$backend
    method <- controls$method
    audit_rows <- controls$audit_rows
    ef_search <- controls$ef_search
    hnsw_build <- controls$hnsw_build
    checkpoint_every <- controls$checkpoint_every
    query_order <- controls$query_order
    if (!is.null(devices) &&
        !method %in% c("exact", "ivf_sharded")) stop(
        "Full KNN devices require exact or IVF search; no fallback.",
        call. = FALSE)
    if (method == "coarse_postings") {
        if (backend != "cpu")
            stop("Posting KNN is CPU-only; no CUDA fallback.")
        return(massive_full_knn_postings(x, k, output, n.cores,
            chunk_rows, reference_chunk_rows, memory_limit, checkpoint,
            resume, audit_rows, checkpoint_every, postings, centers,
            nprobe, query_order))
    }
    if (query_order != "original") stop("Order requires posting search.")
    if (!is.null(postings) || !is.null(centers) || !is.null(nprobe))
        stop("Posting controls require `coarse_postings`.", call. = FALSE)
    if (!is.null(devices)) return(massive_full_knn_device_request(
        x, k, output, backend, n.cores, chunk_rows,
        reference_chunk_rows, memory_limit, checkpoint, resume,
        audit_rows, checkpoint_every, devices, method))
    if (method %in% c("hnsw_sharded", "ivf_sharded") ||
        backend %in% c("cuda", "metal")) {
        return(massive_sharded_graph(x, k, output, n.cores,
            chunk_rows, reference_chunk_rows, memory_limit, audit_rows,
            backend, method, checkpoint, resume, checkpoint_every,
            ef_search, hnsw_build))
    }
    massive_full_knn_exact(x, k, output, n.cores, chunk_rows,
        reference_chunk_rows, memory_limit, checkpoint, resume,
        audit_rows, checkpoint_every)
}
