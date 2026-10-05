massive_posting_plan <- function(x, centers, chunk_rows,
        memory_limit) {
    limit <- massive_memory_bytes(memory_limit)
    fixed <- 64 * 1024^2 + 12 * length(centers) +
        16 * nrow(centers)
    per_row <- 8 * x$ncol + 8
    maximum <- floor((0.7 * limit - fixed) / per_row)
    requested <- chunk_rows %||% 10000L
    rows <- integer_scalar(requested)
    if (is.na(rows) || rows < 1L || maximum < 1) stop(
        "Posting buffers exceed `memory_limit`.", call. = FALSE)
    list(chunk_rows = as.integer(min(rows, maximum)),
        estimated_ram_bytes = fixed + min(rows, maximum) * per_row,
        memory_limit_bytes = limit)
}

massive_posting_check_state <- function(state, signature, paths,
        rows, columns, lists, chunk_rows) {
    valid <- is.list(state) && identical(state$signature, signature) &&
        is.numeric(state$completed_rows) &&
        length(state$completed_rows) == 1L &&
        is.numeric(state$counts) && length(state$counts) == lists &&
        is.numeric(state$positions) &&
        length(state$positions) == lists
    if (!valid) stop("Posting checkpoint does not match inputs.",
        call. = FALSE)
    done <- state$completed_rows
    counts <- state$counts
    offsets <- c(0, cumsum(counts))
    valid <- is.finite(done) && done >= 0 && done <= rows &&
        done == floor(done) &&
        (done == rows || done %% chunk_rows == 0) &&
        all(is.finite(counts)) && all(counts >= 0) &&
        all(counts == floor(counts)) && sum(counts) == rows &&
        all(is.finite(state$positions)) &&
        all(state$positions == floor(state$positions)) &&
        all(state$positions >= utils::head(offsets, -1L)) &&
        all(state$positions <= utils::tail(offsets, -1L)) &&
        sum(state$positions - utils::head(offsets, -1L)) == done
    parts <- paste0(paths[1:2], ".part")
    final <- file.exists(paths[1:2])
    staged <- file.exists(parts)
    files <- ifelse(final, paths[1:2], parts)
    sizes <- file.info(files)$size
    if (!valid || anyNA(sizes) ||
        any(final & staged) ||
        (done != rows && (any(final) ||
            file.exists(paths[[3L]]) ||
            file.exists(paste0(paths[[3L]], ".part")))) ||
        !identical(as.numeric(sizes), c(rows * columns * 4,
            rows * 4))) stop("Posting partial files disagree with ",
        "checkpoint.", call. = FALSE)
    invisible(NULL)
}

massive_posting_prepare <- function(x, centers, prefix, paths, plan,
        workers, checkpoint, resume, checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L) stop(
        "`checkpoint_every` must be a positive integer.",
        call. = FALSE)
    sidecar <- paste0(prefix, ".checkpoint.rds")
    if (file.exists(paste0(prefix, ".manifest.rds")) ||
        (!resume && any(file.exists(c(paths,
            paste0(paths, ".part"), sidecar))))) stop(
        "Posting output or partial files already exist.",
        call. = FALSE)
    if (!checkpoint) return(list(state = NULL, progress = NULL,
        sidecar = NULL, every = 0L))
    signature <- massive_checkpoint_signature(list(version = 1L,
        source = massive_checkpoint_source_identity(x),
        centers = centers, prefix = prefix, plan = plan,
        workers = workers, every = every))
    state <- NULL
    if (resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        massive_posting_check_state(state, signature, paths,
            x$nrow, x$ncol, nrow(centers), plan$chunk_rows)
    } else if (file.exists(sidecar)) stop(
        "Posting checkpoint exists; use `resume = TRUE`.",
        call. = FALSE)
    progress <- function(done, counts, positions) {
        saved <- list(signature = signature, completed_rows = done,
            counts = counts, positions = positions)
        massive_checkpoint_write(saved, sidecar)
    }
    list(state = state, progress = progress, sidecar = sidecar,
        every = every)
}

massive_posting_prefix <- function(output) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) stop(
        "`output` must be one posting-file prefix.", call. = FALSE)
    file.path(normalizePath(dirname(output), winslash = "/",
        mustWork = TRUE),
        basename(output))
}

massive_posting_finalize <- function(x, centers, prefix, paths,
        built, before, plan) {
    result <- list(features = massive_matrix(paths[[1L]], x$nrow,
        x$ncol), ids_path = paths[[2L]],
        offsets_path = paths[[3L]], counts = built$counts,
        offsets = built$offsets, nlist = nrow(centers),
        nrow = x$nrow, ncol = x$ncol, backend = "cpu",
        method = "nearest_center_physical_postings",
        centers_hash = massive_checkpoint_signature(centers),
        source_identity = before, resources = plan,
        file_identity = massive_checkpoint_file_identity(paths),
        manifest_path = paste0(prefix, ".manifest.rds"),
        experimental = TRUE)
    class(result) <- "fastEmbedR_massive_postings"
    massive_checkpoint_write(list(version = 1L,
        postings = result, centers = centers), result$manifest_path)
    result
}

#' EXPERIMENTAL physical posting lists for a file-backed matrix
#'
#' Assigns each row to its nearest supplied center in two streaming passes.
#' The resulting float32 rows are contiguous by posting and retain one-based
#' original row IDs. This builds storage only; it does not perform KNN search
#' or certify the quality of the supplied centers.
#'
#' @param x A file-backed `massive_matrix()` descriptor.
#' @param centers Finite matrix with one center per row and one column per
#'   input feature.
#' @param output New prefix for `.features.f32`, `.ids.u32`,
#'   `.offsets.u64`, and `.manifest.rds` files.
#' @param chunk_rows Maximum rows in one read block.
#' @param n.cores CPU workers used for nearest-center assignment.
#' @param memory_limit Conservative algorithm-buffer RAM budget.
#' @param checkpoint Save a resumable posting-list checkpoint.
#' @param resume Continue a matching interrupted build.
#' @param checkpoint_every Save after this many write-pass blocks.
#' @return A lightweight posting-list descriptor.
#' @export
massive_coarse_postings <- function(x, centers, output,
        chunk_rows = NULL, n.cores = 1L, memory_limit = "1GB",
        checkpoint = FALSE, resume = FALSE, checkpoint_every = 100L) {
    if (!inherits(x, "fastEmbedR_massive_matrix") ||
        x$format == "memory" || x$nrow > .Machine$integer.max)
        stop("Postings need a file-backed source with <= 2^31-1 rows.",
            call. = FALSE)
    if (!is.matrix(centers) || !is.numeric(centers) ||
        ncol(centers) != x$ncol || nrow(centers) < 2L ||
        nrow(centers) > 65536L || any(!is.finite(centers)))
        stop("`centers` must be a finite feature-compatible matrix.",
            call. = FALSE)
    prefix <- massive_posting_prefix(output)
    paths <- paste0(prefix, c(".features.f32", ".ids.u32",
        ".offsets.u64"))
    plan <- massive_posting_plan(x, centers, chunk_rows, memory_limit)
    before <- massive_checkpoint_source_identity(x)
    workers <- normalize_nn_threads(n.cores)
    saved <- massive_posting_prepare(x, centers, prefix, paths,
        plan, workers, checkpoint, resume, checkpoint_every)
    disk <- x$nrow * (4 * x$ncol + 4) +
        8 * (nrow(centers) + 1) +
        16 * length(centers) + 1024^2
    if (!resume && disk >
        0.8 * massive_disk_available_cpp(paths[[1L]]))
        stop("Postings exceed the free-disk budget.", call. = FALSE)
    built <- massive_coarse_postings_cpp(x, centers, prefix,
        plan$chunk_rows, workers, saved$state, saved$progress,
        saved$every)
    if (!identical(before, massive_checkpoint_source_identity(x)))
        stop("Posting input changed during construction.",
            call. = FALSE)
    result <- massive_posting_finalize(x, centers, prefix, paths,
        built, before, plan)
    if (!is.null(saved$sidecar)) unlink(saved$sidecar)
    result
}

#' EXPERIMENTAL reopen of completed physical posting lists
#'
#' Loads the saved centers and posting descriptor without rescanning the
#' source matrix or rebuilding the posting files. File identity, dimensions,
#' and center identity are checked before the descriptor is returned.
#'
#' @param output Prefix passed to [massive_coarse_postings()].
#' @return A list containing `postings` and `centers` for subsequent search
#'   or full-graph construction.
#' @export
massive_open_postings <- function(output) {
    prefix <- massive_posting_prefix(output)
    manifest <- paste0(prefix, ".manifest.rds")
    if (!file.exists(manifest)) stop("Posting manifest is missing or invalid.")
    saved <- tryCatch(readRDS(manifest), error = function(e) NULL)
    if (!is.list(saved)) stop("Posting manifest is missing or invalid.")
    index <- saved$postings
    centers <- saved$centers
    valid <- identical(saved$version, 1L) &&
        inherits(index, "fastEmbedR_massive_postings") &&
        inherits(index$features, "fastEmbedR_massive_matrix") &&
        massive_checkpoint_same_paths(index$manifest_path, manifest) &&
        identical(index$backend, "cpu") &&
        identical(index$method,
            "nearest_center_physical_postings") &&
        is.matrix(centers) && is.numeric(centers) &&
        identical(as.numeric(dim(centers)),
            as.numeric(c(index$nlist, index$ncol))) &&
        all(is.finite(centers)) &&
        identical(index$centers_hash,
            massive_checkpoint_signature(centers))
    if (!valid) stop("Posting manifest is missing or invalid.", call. = FALSE)
    paths <- c(index$features$path, index$ids_path,
        index$offsets_path)
    if (!massive_checkpoint_same_paths(paths, paste0(prefix,
        c(".features.f32", ".ids.u32", ".offsets.u64"))) ||
        !identical(index$features$nrow, index$nrow) ||
        !identical(index$features$ncol, index$ncol) ||
        !is.numeric(index$counts) ||
        !is.numeric(index$offsets) ||
        length(index$counts) != index$nlist ||
        length(index$offsets) != index$nlist + 1L ||
        any(!is.finite(index$counts)) ||
        any(index$counts < 0) ||
        index$offsets[[1L]] != 0 ||
        !identical(as.numeric(diff(index$offsets)),
            as.numeric(index$counts)) ||
        !identical(as.numeric(utils::tail(index$offsets, 1L)),
            as.numeric(index$nrow))) stop(
        "Posting manifest dimensions or paths are invalid.",
        call. = FALSE)
    current <- tryCatch(massive_checkpoint_file_identity(paths),
        error = function(e) NULL)
    expected <- c(4 * index$nrow * index$ncol,
        4 * index$nrow, 8 * (index$nlist + 1L))
    if (!identical(current, index$file_identity) ||
        !identical(current$bytes, as.numeric(expected))) stop(
        "Posting files changed since construction.", call. = FALSE)
    list(postings = index, centers = centers)
}

#' Read a bounded part of an experimental physical posting list
#'
#' @param x A result of [massive_coarse_postings()].
#' @param list One-based posting number.
#' @param first One-based row within the posting.
#' @param n Number of rows to inspect.
#' @return Original one-based row IDs and a double matrix of features.
#' @export
massive_read_posting <- function(x, list, first = 1L, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_postings")) stop(
        "`x` must be a massive posting descriptor.", call. = FALSE)
    bucket <- integer_scalar(list)
    start <- integer_scalar(first)
    count <- integer_scalar(n)
    if (is.na(bucket) || bucket < 1L || bucket > x$nlist ||
        is.na(start) || start < 1L || is.na(count) || count < 1L ||
        count > x$counts[[bucket]] - start + 1L) stop(
        "Requested posting rows are invalid.", call. = FALSE)
    paths <- c(x$features$path, x$ids_path, x$offsets_path)
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) stop("Posting files changed.", call. = FALSE)
    row <- x$offsets[[bucket]] + start
    data <- massive_read_rows(x$features, row, count)
    con <- file(x$ids_path, "rb")
    on.exit(close(con))
    seek(con, where = 4 * (row - 1), origin = "start")
    ids <- readBin(con, integer(), count, size = 4L,
        endian = "little")
    if (length(ids) != count || any(ids < 1L | ids > x$nrow))
        stop("Posting row IDs are invalid.", call. = FALSE)
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) stop("Posting files changed during read.",
            call. = FALSE)
    list(row_id = ids, data = data)
}

massive_posting_search_plan <- function(query, postings, count, k,
        probes, reference_chunk, workers, memory_limit,
        shrink_query = FALSE) {
    limit <- massive_memory_bytes(memory_limit)
    resident <- if (query$format == "memory") {
        query$nrow * query$ncol * 8
    } else 0
    fixed <- resident + 64 * 1024^2 +
        12 * postings$nlist * query$ncol +
        (32 + 16 * workers) * postings$nlist
    per_query <- 4 * query$ncol + 32 * k + 16 * probes + 64
    if (shrink_query) {
        # Reserve space for useful reference blocks during a graph scan.
        count <- floor(min(count,
            max(1, 0.5 * (0.7 * limit - fixed) / per_query),
            (0.7 * limit - fixed - 4 * query$ncol - 4) /
                per_query,
            128e6 / (12 * k), 128e6 / (4 * probes),
            256e6 / (4 * query$ncol)))
    }
    available <- floor((0.7 * limit - fixed - count * per_query) /
        (4 * query$ncol + 4))
    requested <- integer_scalar(reference_chunk)
    if (is.na(requested) || requested < 1L || count < 1 ||
        available < 1) stop(
        "Posting search buffers exceed `memory_limit`.",
        call. = FALSE)
    rows <- as.integer(min(requested, available, 256e6 /
        (4 * query$ncol + 4)))
    if (rows < 1L || count * k > 128e6 / 12 ||
        count * probes > 128e6 / 4 ||
        count * query$ncol * 4 > 256e6) stop(
        "Posting search buffers exceed native limits.",
        call. = FALSE)
    list(query_rows = as.integer(count), reference_chunk = rows,
        estimated_ram_bytes = fixed + count * per_query +
            rows * (4 * query$ncol + 4),
        memory_limit_bytes = limit)
}

massive_posting_assert_files <- function(postings) {
    paths <- c(postings$features$path, postings$ids_path,
        postings$offsets_path)
    if (!identical(massive_checkpoint_file_identity(paths),
        postings$file_identity)) stop(
        "Posting files changed since construction.", call. = FALSE)
    paths
}

massive_posting_search_controls <- function(query, postings, centers,
        first, n, k, nprobe, exclude_self) {
    if (!inherits(query, "fastEmbedR_massive_matrix") ||
        !inherits(postings, "fastEmbedR_massive_postings") ||
        query$ncol != postings$ncol) stop(
        "Query and posting dimensions are incompatible.",
        call. = FALSE)
    if (!is.matrix(centers) || !is.numeric(centers) ||
        !identical(massive_checkpoint_signature(centers),
            postings$centers_hash)) stop(
        "Posting centers do not match their build.", call. = FALSE)
    if (!is.logical(exclude_self) || length(exclude_self) != 1L ||
        is.na(exclude_self)) stop(
        "`exclude_self` must be TRUE or FALSE.", call. = FALSE)
    start <- as.numeric(first)
    count <- integer_scalar(n)
    neighbors <- integer_scalar(k)
    probes <- integer_scalar(nprobe %||% min(8L,
        postings$nlist))
    if (length(start) != 1L || !is.finite(start) ||
        start < 1 || start != floor(start) ||
        is.na(count) || count < 1L ||
        count > query$nrow - start + 1 ||
        is.na(neighbors) || neighbors < 1L ||
        neighbors > postings$nrow - as.integer(exclude_self) ||
        is.na(probes) || probes < 1L || probes > postings$nlist) stop(
        "Invalid posting search range or controls.", call. = FALSE)
    list(first = start, count = count, k = neighbors,
        probes = probes)
}

#' EXPERIMENTAL bounded search over physical posting lists
#'
#' Searches only the nearest `nprobe` center lists for each query. Every
#' queried list is scanned exactly; with `nprobe = postings$nlist`, results
#' match full exact Euclidean search. Fewer probes are approximate and have
#' no recall guarantee. This returns one bounded batch, not a full graph.
#'
#' @param query A `massive_matrix()` query source.
#' @param postings A result of [massive_coarse_postings()].
#' @param centers The same centers used to build `postings`.
#' @param first First one-based query row.
#' @param n Number of query rows in this batch.
#' @param k Number of nearest rows to return.
#' @param nprobe Number of posting lists to scan per query; defaults to
#'   at most eight lists.
#' @param reference_chunk Maximum posting rows read at once.
#' @param n.cores CPU workers for distance calculations.
#' @param memory_limit Conservative RAM budget for the batch.
#' @param exclude_self Exclude matching row IDs. `query` must be the
#'   original source or the physical posting feature file; the latter
#'   uses the saved row-ID file to identify each query.
#' @return Neighbor IDs, distances, candidate counts, and backend metadata.
#' @export
massive_search_postings <- function(query, postings, centers,
        first = 1, n = 256L, k = 30L, nprobe = NULL,
        reference_chunk = 8192L, n.cores = 1L,
        memory_limit = "1GB", exclude_self = FALSE) {
    controls <- massive_posting_search_controls(query, postings,
        centers, first, n, k, nprobe, exclude_self)
    source <- massive_checkpoint_source_identity(query)
    original <- identical(source, postings$source_identity)
    grouped <- identical(source,
        massive_checkpoint_source_identity(postings$features))
    if (exclude_self && !original && !grouped) stop(
        "Self exclusion requires the original or grouped posting source.",
        call. = FALSE)
    paths <- massive_posting_assert_files(postings)
    workers <- normalize_nn_threads(n.cores)
    plan <- massive_posting_search_plan(query, postings,
        controls$count, controls$k, controls$probes,
        reference_chunk, workers, memory_limit)
    native_query <- query
    if (exclude_self && grouped && !original)
        native_query$query_ids_path <- postings$ids_path
    result <- massive_posting_search_cpp(native_query, paths[[1L]],
        paths[[2L]], paths[[3L]], centers, controls$first,
        controls$count, controls$k, controls$probes,
        plan$reference_chunk,
        workers, exclude_self, postings$nrow, TRUE)
    if (!identical(source, massive_checkpoint_source_identity(query)) ||
        !identical(massive_checkpoint_file_identity(paths),
            postings$file_identity)) stop(
        "Posting search inputs changed during execution.",
        call. = FALSE)
    result$nprobe <- controls$probes
    result$nlist <- postings$nlist
    result$resources <- plan
    result$experimental <- TRUE
    result
}

#' EXPERIMENTAL recall preflight for physical posting search
#'
#' Checks evenly spaced, distant-from-center, and small-posting source rows
#' against streamed exact neighbors before writing a full graph. At least one
#' audit row represents a small nonempty posting list; larger audits reserve
#' up to one quarter for these lists.
#' Center scoring uses bounded candidates when a full scan is too costly.
#' The sampled recall cannot certify unsampled rows.
#'
#' @param x Original file-backed matrix used to build `postings`.
#' @param postings A result of [massive_coarse_postings()].
#' @param centers Centers used to build `postings`.
#' @param k Number of non-self neighbors.
#' @param nprobe Number of posting lists searched per sampled row.
#' @param sample_rows Number of sampled rows, at most 256.
#' @param target_recall Minimum required recall for each sampled row.
#' @param reference_chunk_rows Maximum reference rows read at once.
#' @param n.cores CPU workers for distance calculations.
#' @param memory_limit Conservative algorithm-buffer RAM budget.
#' @return Sampled row IDs and recalls, summary, backend, and estimated
#'   exact-reference read and comparison counts.
#' @export
massive_posting_recall_pilot <- function(x, postings, centers,
        k = 30L, nprobe = NULL, sample_rows = 32L,
        target_recall = 0.99, reference_chunk_rows = 8192L,
        n.cores = 1L, memory_limit = "1GB") {
    if (!inherits(x, "fastEmbedR_massive_matrix")) stop(
        "Recall pilot needs a massive matrix.", call. = FALSE)
    count <- integer_scalar(sample_rows)
    if (is.na(count) || count < 1L || count > min(x$nrow, 256L) ||
        !is.numeric(target_recall) || length(target_recall) != 1L ||
        !is.finite(target_recall) || target_recall <= 0 ||
        target_recall > 1) stop("Invalid recall pilot controls.",
            call. = FALSE)
    controls <- massive_posting_search_controls(x, postings,
        centers, 1, count, k, nprobe, TRUE)
    identity <- massive_checkpoint_source_identity(x)
    if (x$format == "memory" ||
        !identical(identity, postings$source_identity)) stop(
        "Recall pilot needs the original file-backed source.",
        call. = FALSE)
    paths <- massive_posting_assert_files(postings)
    workers <- normalize_nn_threads(n.cores)
    plan <- massive_posting_search_plan(x, postings, count,
        controls$k, controls$probes, reference_chunk_rows,
        workers, memory_limit)
    exact_plan <- massive_full_knn_resources(x, controls$k,
        count, reference_chunk_rows, memory_limit)
    if (exact_plan$chunk_rows < count) stop(
        "Exact pilot exceeds `memory_limit`; reduce `sample_rows`.",
        call. = FALSE)
    selection <- massive_posting_audit_rows(x, centers, count,
        plan$reference_chunk, workers, postings)
    query <- x
    query$selected_rows <- selection$rows
    started <- proc.time()[[3L]]
    observed <- massive_posting_search_cpp(query, paths[[1L]],
        paths[[2L]], paths[[3L]], centers, 1, count,
        controls$k, controls$probes, plan$reference_chunk,
        workers, TRUE, postings$nrow, FALSE)
    exact <- massive_exact_sample_batch_cpp(x, x, selection$rows,
        controls$k, exact_plan$reference_chunk_rows,
        workers, TRUE)
    massive_posting_pilot_summary(x, postings, paths, identity,
        selection, observed, exact, controls, target_recall, started)
}

massive_posting_pilot_summary <- function(x, postings, paths,
        identity, selection, observed, exact, controls,
        target_recall, started) {
    if (!identical(exact$backend_used, "native_cpu_exact_stream") ||
        !isTRUE(exact$exact) ||
        !identical(observed$backend_used,
            "native_cpu_posting_scan")) stop(
        "Recall pilot backend changed.", call. = FALSE)
    rows <- selection$rows
    recall <- vapply(seq_along(rows), function(i) sum(
        observed$indices[i, ] %in% exact$indices[i, ]) /
        controls$k, numeric(1L))
    if (!identical(identity, massive_checkpoint_source_identity(x)) ||
        !identical(postings$file_identity,
            massive_checkpoint_file_identity(paths))) stop(
        "Recall pilot inputs changed during execution.",
        call. = FALSE)
    c(list(sample_rows = rows, row_recall = recall,
        candidate_count = observed$candidate_count,
        posting_read_bytes = observed$posting_read_bytes,
        exact_reference_bytes = x$nrow * x$ncol * 4,
        selection_source_bytes = selection$scored_rows * x$ncol * 4,
        selection_source_rows = selection$scored_rows,
        posting_lists_sampled = selection$posting_lists_sampled,
        exact_comparisons = x$nrow * length(rows),
        audit_sampling = selection$sampling,
        nprobe = controls$probes, nlist = postings$nlist,
        k = controls$k, target_recall = target_recall,
        posting_backend = observed$backend_used,
        reference_backend = exact$backend_used,
        experimental = TRUE,
        elapsed_seconds = unname(proc.time()[[3L]] - started)),
        massive_recall_summary(recall, target_recall))
}

massive_posting_graph_batch <- function(query, postings, centers,
        first, count, k, probes, reference_chunk, workers) {
    massive_posting_assert_files(postings)
    result <- massive_posting_search_cpp(query,
        postings$features$path, postings$ids_path,
        postings$offsets_path, centers, first, count, k,
        probes, reference_chunk, workers, TRUE,
        postings$nrow, FALSE)
    massive_posting_assert_files(postings)
    result
}

massive_posting_graph_setup <- function(x, postings, centers,
        k, output, nprobe, workers, chunk_rows,
        reference_chunk_rows, memory_limit, resume,
        checkpoint_every, query_order = "original") {
    if (x$format == "memory" ||
        !identical(massive_checkpoint_source_identity(x),
            postings$source_identity)) stop(
        "Posting graph needs its original file-backed source.", call. = FALSE)
    controls <- massive_posting_search_controls(x, postings,
        centers, 1, 1L, k, nprobe, TRUE)
    query <- if (query_order == "posting") postings$features else x
    rows <- integer_scalar(chunk_rows %||% 65536L)
    if (is.na(rows) || rows < 1L) stop(
        "`chunk_rows` must be a positive integer.", call. = FALSE)
    plan <- massive_posting_search_plan(query, postings,
        min(rows, x$nrow), k, controls$probes,
        reference_chunk_rows %||% 8192L, workers, memory_limit,
        shrink_query = TRUE)
    resources <- list(chunk_rows = plan$query_rows,
        reference_chunk_rows = plan$reference_chunk,
        peak_ram_bytes = plan$estimated_ram_bytes,
        output_bytes = x$nrow * k * 8,
        memory_limit_bytes = plan$memory_limit_bytes,
        scan_bytes_upper = ceiling(x$nrow / plan$query_rows) *
            x$nrow * (4 * x$ncol + 4))
    paths <- massive_knn_paths(output, resources$output_bytes, resume,
        allow_final_resume = query_order == "posting",
        allow_mixed_resume = query_order == "posting")
    prefix <- sub("\\.indices\\.u32$", "", paths[["indices"]])
    setup <- list(output = prefix, paths = paths, k = k,
        workers = workers, resources = resources,
        checkpoint_every = checkpoint_every,
        method = "coarse_postings", backend_used = "native_cpu_posting_scan",
        exact = controls$probes == postings$nlist,
        query_source = query,
        posting_identity = c(list(files = postings$file_identity,
            centers = postings$centers_hash,
            nprobe = controls$probes),
            if (query_order == "posting")
                list(query_order = query_order)))
    native_query <- query
    if (query_order == "posting")
        native_query$query_ids_path <- postings$ids_path
    setup$search_batch <- function(first, count) {
        massive_posting_graph_batch(native_query, postings,
            centers, first, count, k, controls$probes,
            plan$reference_chunk, workers)
    }
    setup
}

massive_posting_reorder_plan <- function(x, k, grouped, final,
        memory_limit) {
    limit <- massive_memory_bytes(memory_limit)
    record_bytes <- 4 + 8 * k
    bucket_rows <- floor(min(x$nrow, 250000,
        0.15 * limit / (record_bytes + 8)))
    read_rows <- floor(min(65536,
        0.1 * limit / record_bytes))
    if (bucket_rows < 1 || read_rows < 1) stop(
        "Grouped graph reorder exceeds `memory_limit`.",
        call. = FALSE)
    existing <- c(grouped, final, paste0(final, ".part"))
    existing <- existing[file.exists(existing)]
    written <- sum(file.info(existing)$size)
    needed <- 2 * x$nrow * k * 8 +
        x$nrow * record_bytes * (1 + 1 / 32) - written
    if (needed > 0.85 * massive_disk_available_cpp(final[[1L]]))
        stop("Grouped graph needs more free disk space.",
            call. = FALSE)
    list(bucket_rows = as.integer(bucket_rows),
        read_rows = as.integer(read_rows),
        peak_ram_bytes = bucket_rows * (record_bytes + 8) +
            read_rows * record_bytes + 32 * 65536 + 64 * 1024^2,
        disk_required_bytes = needed)
}

massive_posting_grouped_saved <- function(manifest, signature,
        paths, resume) {
    saved <- tryCatch(readRDS(manifest), error = function(e) NULL)
    if (!resume || !is.list(saved) ||
        !identical(saved$signature, signature) ||
        !is.list(saved$measurement) ||
        !is.numeric(saved$measurement$seconds) ||
        length(saved$measurement$seconds) != 1L ||
        !is.finite(saved$measurement$seconds) ||
        !is.list(saved$files)) stop(
        "Grouped graph manifest does not match its inputs.",
        call. = FALSE)
    current <- massive_checkpoint_file_identity(paths)
    staged <- stats::setNames(paste0(paths, ".part"), names(paths))
    prior <- saved$files
    valid <- identical(prior, current) ||
        (massive_checkpoint_same_names(prior$paths, unname(staged)) &&
            identical(prior$bytes, current$bytes) &&
            identical(prior$modified, current$modified))
    if (!valid) stop("Completed grouped graph changed.",
        call. = FALSE)
    saved$measurement
}

massive_posting_restore_pair <- function(paths, manifest,
        signature, resume) {
    finished <- file.exists(paths)
    if (!any(finished) || all(finished)) return(invisible(NULL))
    parts <- paste0(paths, ".part")
    if (!resume || sum(finished) != 1L ||
        !all(finished != file.exists(parts))) stop(
        "Graph files need manual review.", call. = FALSE)
    saved <- tryCatch(readRDS(manifest), error = function(e) NULL)
    if (!is.list(saved) ||
        !identical(saved$signature, signature) ||
        !is.list(saved$files)) stop(
        "Graph commit manifest does not match its inputs.",
        call. = FALSE)
    current <- ifelse(finished, paths, parts)
    identity <- massive_checkpoint_file_identity(current)
    expected <- saved$files
    if (!massive_checkpoint_same_names(expected$paths,
        unname(parts)) ||
        !identical(expected$bytes, identity$bytes) ||
        !identical(expected$modified, identity$modified)) stop(
        "Graph mixed files changed.", call. = FALSE)
    if (!file.rename(paths[finished], parts[finished])) stop(
        "Could not restore graph partial file.",
        call. = FALSE)
    invisible(NULL)
}

massive_posting_grouped_measure <- function(x, setup, postings,
        checkpoint, resume) {
    signature <- massive_checkpoint_signature(list(version = 1L,
        source = massive_checkpoint_source_identity(x),
        query = massive_checkpoint_source_identity(setup$query_source),
        posting = setup$posting_identity, output = setup$output,
        k = setup$k, workers = setup$workers,
        resources = setup$resources,
        checkpoint_every = setup$checkpoint_every))
    manifest <- paste0(setup$output, ".grouped-manifest.rds")
    massive_posting_restore_pair(setup$paths, manifest,
        signature, resume)
    if (all(file.exists(setup$paths))) {
        return(massive_posting_grouped_saved(manifest,
            signature, setup$paths, resume))
    }
    if (file.exists(manifest) && !resume) stop(
        "Grouped graph manifest exists; use `resume = TRUE`.",
        call. = FALSE)
    if (file.exists(manifest)) {
        saved <- tryCatch(readRDS(manifest), error = function(e) NULL)
        if (!is.list(saved) ||
            !identical(saved$signature, signature)) stop(
            "Grouped graph manifest does not match its inputs.",
            call. = FALSE)
        parts <- paste0(setup$paths, ".part")
        if (!all(file.exists(parts)) ||
            !identical(saved$files,
                massive_checkpoint_file_identity(parts))) stop(
            "Grouped graph partial files changed.", call. = FALSE)
    }
    setup$before_finish <- function(parts, measurement) {
        prior <- if (file.exists(manifest)) readRDS(manifest) else NULL
        if (!is.null(prior$measurement))
            measurement <- prior$measurement
        massive_checkpoint_write(list(signature = signature,
            files = massive_checkpoint_file_identity(parts),
            measurement = measurement), manifest)
    }
    state <- if (checkpoint) massive_full_knn_checkpoint(
        setup$query_source, setup, resume) else NULL
    massive_full_knn_stream(setup$query_source, setup, state)
    measurement <- readRDS(manifest)$measurement
    massive_checkpoint_write(list(signature = signature,
        files = massive_checkpoint_file_identity(setup$paths),
        measurement = measurement), manifest)
    measurement
}

massive_posting_reorder_commit <- function(x, k, prefix, final,
        setup, postings, plan, resume) {
    marker <- paste0(prefix, ".reorder-manifest.rds")
    parts <- stats::setNames(paste0(final, ".part"), names(final))
    signature <- massive_checkpoint_signature(list(version = 1L,
        source = massive_checkpoint_source_identity(x),
        postings = postings$file_identity,
        grouped = massive_checkpoint_file_identity(setup$paths),
        output = final, k = k,
        plan = plan[c("bucket_rows", "read_rows")]))
    massive_posting_restore_pair(final, marker, signature, resume)
    if (all(file.exists(final))) {
        saved <- massive_posting_grouped_saved(marker, signature,
            final, resume)
        return(saved$seconds)
    }
    if (file.exists(marker)) {
        if (!all(file.exists(parts))) stop(
            "Reordered graph partial files changed.", call. = FALSE)
        seconds <- massive_posting_grouped_saved(marker,
            signature, parts, resume)$seconds
    } else {
        message("EXPERIMENTAL posting KNN: reordering grouped graph")
        started <- proc.time()[[3L]]
        massive_reorder_posting_graph_cpp(postings$ids_path,
            setup$paths[["indices"]], setup$paths[["distances"]],
            parts[["indices"]], parts[["distances"]], x$nrow,
            k, plan$bucket_rows, plan$read_rows, resume)
        seconds <- unname(proc.time()[[3L]] - started)
        massive_checkpoint_write(list(signature = signature,
            files = massive_checkpoint_file_identity(parts),
            measurement = list(seconds = seconds)), marker)
    }
    massive_posting_assert_files(postings)
    massive_full_knn_assert_source(x, postings$source_identity)
    massive_knn_finish(NULL, NULL, parts, final, x$nrow, k)
    seconds
}

massive_posting_completion_signature <- function(x, k, prefix,
        postings, nprobe, workers, chunk_rows,
        reference_chunk_rows, memory_limit, checkpoint_every) {
    massive_checkpoint_signature(list(version = 1L,
        source = massive_checkpoint_source_identity(x),
        posting = postings$file_identity,
        centers = postings$centers_hash,
        output = prefix, k = k, nprobe = nprobe,
        workers = workers, chunk_rows = chunk_rows,
        reference_chunk_rows = reference_chunk_rows,
        memory_limit = memory_limit,
        checkpoint_every = checkpoint_every))
}

massive_posting_completed <- function(marker, signature, final,
        resume, postings, centers) {
    if (!file.exists(marker)) return(NULL)
    saved <- tryCatch(readRDS(marker), error = function(e) NULL)
    if (!identical(saved$stage, "complete")) return(NULL)
    if (!resume || !all(file.exists(final)) ||
        !identical(saved$signature, signature) ||
        !identical(saved$files,
            massive_checkpoint_file_identity(final)) ||
        !is.list(saved$setup) ||
        !massive_checkpoint_same_names(saved$setup$paths, final) ||
        !is.list(saved$measurement)) stop(
        "Completed posting graph does not match its inputs.",
        call. = FALSE)
    massive_posting_assert_files(postings)
    massive_posting_search_controls(postings$features,
        postings, centers, 1, 1L, saved$setup$k,
        saved$setup$posting_identity$nprobe, TRUE)
    grouped <- paste0(sub("\\.indices\\.u32$", "",
        final[["indices"]]), ".grouped",
        c(".indices.u32", ".distances.f32"))
    if (!massive_checkpoint_same_names(saved$grouped$paths,
        grouped)) stop(
        "Completed posting graph has invalid work paths.",
        call. = FALSE)
    present <- file.exists(grouped)
    if (any(present)) {
        current <- massive_checkpoint_file_identity(grouped[present])
        if (!identical(current$bytes,
            saved$grouped$bytes[present]) ||
            !identical(current$modified,
                saved$grouped$modified[present])) stop(
            "Completed posting graph work files changed.",
            call. = FALSE)
        unlink(grouped[present])
    }
    setup <- saved$setup
    query <- postings$features
    query$query_ids_path <- postings$ids_path
    setup$search_batch <- function(first, count) {
        massive_posting_graph_batch(query, postings, centers,
            first, count, setup$k,
            setup$posting_identity$nprobe,
            setup$resources$reference_chunk_rows,
            setup$workers)
    }
    list(setup = setup, measurement = saved$measurement)
}

massive_posting_complete_mark <- function(marker, signature,
        final, grouped, setup, measurement) {
    setup$search_batch <- NULL
    setup$before_finish <- NULL
    massive_checkpoint_write(list(stage = "complete",
        signature = signature,
        files = massive_checkpoint_file_identity(final),
        grouped = massive_checkpoint_file_identity(grouped),
        setup = setup, measurement = measurement), marker)
}

massive_posting_grouped_run <- function(x, k, output, postings,
        centers, nprobe, workers, chunk_rows, reference_chunk_rows,
        memory_limit, checkpoint, resume, checkpoint_every) {
    final <- massive_knn_paths(output, x$nrow * k * 8, resume,
        allow_final_resume = TRUE, allow_mixed_resume = TRUE)
    prefix <- sub("\\.indices\\.u32$", "", final[["indices"]])
    marker <- paste0(prefix, ".reorder-manifest.rds")
    signature <- massive_posting_completion_signature(x, k,
        prefix, postings, nprobe, workers, chunk_rows,
        reference_chunk_rows, memory_limit, checkpoint_every)
    completed <- massive_posting_completed(marker, signature,
        final, resume, postings, centers)
    if (!is.null(completed)) return(completed)
    setup <- massive_posting_graph_setup(x, postings, centers,
        k, paste0(prefix, ".grouped"), nprobe, workers,
        chunk_rows, reference_chunk_rows, memory_limit, resume,
        checkpoint_every, query_order = "posting")
    grouped <- setup$paths
    plan <- massive_posting_reorder_plan(x, k, grouped,
        final, memory_limit)
    manifest <- paste0(setup$output, ".grouped-manifest.rds")
    parts <- stats::setNames(paste0(final, ".part"), names(final))
    committed <- FALSE
    on.exit(if (!committed && !checkpoint)
        unlink(c(grouped, parts, final, manifest, marker)))
    massive_posting_graph_message(x, k, setup, postings,
        "posting")
    measurement <- massive_posting_grouped_measure(x, setup,
        postings, checkpoint, resume)
    massive_posting_assert_files(postings)
    measurement$reorder_seconds <- massive_posting_reorder_commit(
        x, k, prefix, final, setup, postings, plan, resume)
    setup$resources$reorder <- plan
    setup$resources$peak_ram_bytes <- max(
        setup$resources$peak_ram_bytes, plan$peak_ram_bytes)
    setup$output <- prefix
    setup$paths <- final
    massive_posting_complete_mark(marker, signature, final,
        grouped, setup, measurement)
    committed <- TRUE
    unlink(c(grouped, manifest))
    list(setup = setup, measurement = measurement)
}

massive_posting_graph_message <- function(x, k, setup, postings,
        query_order) {
    message("EXPERIMENTAL posting KNN: ", x$nrow,
        " rows; k=", k, "; probes=",
        setup$posting_identity$nprobe, "/", postings$nlist,
        "; query rows=", setup$resources$chunk_rows,
        "; query order=", query_order,
        "; scan upper bound=",
        signif(setup$resources$scan_bytes_upper / 1024^3, 3),
        " GiB")
}

massive_posting_representatives <- function(postings, count) {
    if (is.null(postings) || count == 0L) return(integer())
    nonempty <- which(postings$counts > 0)
    chosen <- utils::head(nonempty[order(postings$counts[nonempty],
        nonempty)], count)
    paths <- massive_posting_assert_files(postings)
    con <- file(paths[[2L]], "rb")
    on.exit(close(con))
    rows <- vapply(chosen, function(bucket) {
        offset <- postings$offsets[[bucket]] +
            floor((postings$counts[[bucket]] - 1) / 2)
        seek(con, where = 4 * offset, origin = "start")
        id <- readBin(con, integer(), n = 1L, size = 4L,
            endian = "little")
        if (length(id) != 1L || id < 1L || id > postings$nrow)
            stop("Posting representative ID is invalid.", call. = FALSE)
        id
    }, integer(1L))
    massive_posting_assert_files(postings)
    rows
}

massive_posting_audit_rows <- function(x, centers, audit_rows,
        chunk_rows, workers, postings = NULL) {
    if (audit_rows == 0L) return(list(rows = integer(),
        scored_rows = 0, sampling = "none",
        posting_lists_sampled = 0L))
    reserved <- if (is.null(postings)) 0L else
        max(1L, floor(audit_rows / 4L))
    representatives <- massive_posting_representatives(postings,
        reserved)
    spread_count <- floor((audit_rows - length(representatives)) / 2L)
    spread <- if (spread_count > 0L) as.integer(floor(
        (seq_len(spread_count) - 0.5) * x$nrow /
            spread_count) + 1) else integer()
    bounded <- x$nrow * x$ncol * nrow(centers) > 1e9
    candidates <- if (bounded) as.integer(min(x$nrow,
        max(audit_rows, min(512, 4 * audit_rows)))) else 0L
    distant <- massive_posting_distant_rows_cpp(x, centers,
        audit_rows, min(8192L, chunk_rows), workers, candidates)
    selected <- unique(c(representatives, spread))
    rows <- sort(c(selected, utils::head(
        distant[!distant %in% selected], audit_rows - length(selected))))
    sampling <- if (bounded) {
        "evenly_spaced_and_distant_candidates"
    } else "evenly_spaced_and_distant_center"
    if (length(representatives))
        sampling <- paste0(sampling, "_and_posting_lists")
    list(rows = rows,
        scored_rows = if (bounded) candidates else x$nrow,
        posting_lists_sampled = length(representatives),
        sampling = sampling)
}

massive_full_knn_postings <- function(x, k, output, n.cores,
        chunk_rows, reference_chunk_rows, memory_limit,
        checkpoint, resume, audit_rows, checkpoint_every,
        postings, centers, nprobe, query_order) {
    workers <- normalize_nn_threads(n.cores)
    if (query_order == "posting") {
        grouped <- massive_posting_grouped_run(x, k, output,
            postings, centers, nprobe, workers, chunk_rows,
            reference_chunk_rows, memory_limit, checkpoint,
            resume, checkpoint_every)
        setup <- grouped$setup
        measurement <- grouped$measurement
    } else {
        setup <- massive_posting_graph_setup(x, postings, centers,
            k, output, nprobe, workers, chunk_rows,
            reference_chunk_rows, memory_limit, resume,
            checkpoint_every)
        saved <- if (checkpoint) massive_full_knn_checkpoint(
            x, setup, resume) else NULL
        massive_posting_graph_message(x, k, setup, postings,
            "original")
    }
    if (query_order == "original")
        measurement <- massive_full_knn_stream(x, setup, saved)
    graph <- massive_knn_graph(setup$output, x$nrow, k)
    graph$backend <- "cpu"
    graph$method <- setup$backend_used
    graph$exact <- setup$exact
    graph$nprobe <- setup$posting_identity$nprobe
    graph$nlist <- postings$nlist
    graph$posting_identity <- setup$posting_identity
    graph$source_identity <- massive_checkpoint_source_identity(x)
    graph$resources <- setup$resources
    graph$resources$posting_read_bytes <- measurement$posting_read_bytes
    graph$search_seconds <- measurement$seconds
    graph$query_order <- query_order
    if (query_order == "posting")
        graph$reorder_seconds <- measurement$reorder_seconds
    graph$checkpoint <- checkpoint
    graph$checkpoint_every <- checkpoint_every
    if (audit_rows == 0L) return(graph)
    selection <- massive_posting_audit_rows(x, centers, audit_rows,
        setup$resources$chunk_rows, setup$workers, postings)
    graph <- massive_full_knn_audit(graph, x, audit_rows, setup,
        selection$rows)
    graph$audit_sampling <- selection$sampling
    graph$audit_selection_source_rows <- selection$scored_rows
    graph$audit_posting_lists_sampled <- selection$posting_lists_sampled
    graph
}
