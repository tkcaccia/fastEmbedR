massive_graph_import <- function(prefix, n_vertices, k, access, kind) {
    if (!is.character(prefix) || length(prefix) != 1L ||
        is.na(prefix) || !nzchar(prefix)) {
        stop("`prefix` must be one graph file prefix.", call. = FALSE)
    }
    if (!is.numeric(n_vertices) || length(n_vertices) != 1L ||
        !is.finite(n_vertices) || n_vertices < 2 ||
        n_vertices != floor(n_vertices) ||
        n_vertices > .Machine$integer.max) {
        stop("`n_vertices` must be between 2 and 2^31 - 1.",
            call. = FALSE)
    }
    if (!is.numeric(k) || length(k) != 1L || !is.finite(k) ||
        k < 1 || k != floor(k) || k > 65536 || k >= n_vertices) {
        stop("`k` must be a valid non-self neighbor count.",
            call. = FALSE)
    }
    storage <- massive_graph_storage_cpp(n_vertices, as.integer(k))
    suffix <- if (kind == "weight") ".weights.f32" else ".distances.f32"
    paths <- normalizePath(paste0(prefix, c(".indices.u32", suffix)),
        mustWork = TRUE)
    identity <- massive_checkpoint_input_files_identity(paths)
    checked <- massive_validate_graph_cpp(paths[[1L]], paths[[2L]],
        n_vertices, as.integer(k), "stream", kind)
    if (!identical(identity, massive_checkpoint_input_files_identity(paths))) {
        stop("Massive graph files changed during validation.",
            call. = FALSE)
    }
    graph <- list(indices_path = paths[[1L]],
        values_path = paths[[2L]],
        n_vertices = checked$n_vertices, n_edges = checked$n_edges,
        k = as.integer(k), storage = "fixed_knn",
        metric = if (kind == "distance") {
            "euclidean"
        } else "external",
        file_bytes_each = storage$file_bytes,
        last_row_offset_bytes = storage$last_row_offset_bytes,
        index_dtype = "uint32", value_dtype = "float32",
        directed = TRUE, symmetrized = FALSE,
        edge_value = kind, backend = "external",
        access = access, validation_access = "stream",
        validated = checked$validated,
        file_identity = identity[c("paths", "bytes", "modified")],
        validated_input_identity = identity,
        experimental = TRUE)
    graph[[paste0(kind, "s_path")]] <- paths[[2L]]
    graph[[paste0(kind, "_dtype")]] <- "float32"
    class(graph) <- "fastEmbedR_massive_graph"
    graph
}

#' Import an experimental file-backed full KNN graph
#'
#' The input is a pair of existing row-major files named
#' `prefix.indices.u32` and `prefix.distances.f32`. Each vertex must have
#' exactly `k` distinct, non-self neighbors with one-based unsigned 32-bit
#' IDs and finite non-negative float32 distances. Import scans and validates
#' every edge with bounded sequential reads. `access = "mmap"` applies to
#' later bounded edge reads, not the full validation scan. Import does not
#' compute neighbors, symmetrize edges, construct UMAP weights, or cluster.
#'
#' @param prefix Prefix of the existing KNN file pair.
#' @param n_vertices Number of vertices; currently at most 2^31 - 1.
#' @param k Number of non-self neighbors per vertex; at most 65,536.
#' @param access Sequential file reads or memory-mapped access.
#' @return A lightweight `fastEmbedR_massive_graph` descriptor. The source
#'   files are reused without copying them.
#' @export
massive_knn_graph <- function(prefix, n_vertices, k,
                                access = c("stream", "mmap")) {
    massive_graph_import(prefix, n_vertices, k, match.arg(access),
        "distance")
}

#' Calculate experimental directed UMAP memberships from disk-backed KNN
#'
#' Scans validated full-graph distances twice in bounded native buffers.
#' The resulting `.weights.f32` file shares the existing index file. These
#' are directed memberships, not the symmetrized fuzzy UMAP graph, and are
#' not yet accepted by full-data UMAP optimization.
#'
#' @param x A distance-valued result of [massive_knn_graph()] or
#'   [massive_full_knn_graph()].
#' @param checkpoint Save completed mean-scan and weight blocks for resume.
#' @param resume Resume an interrupted checkpointed run.
#' @param checkpoint_every Save progress after this many native blocks.
#' @param backend This graph transformation currently runs on CPU only.
#' @param n.cores CPU workers for independent per-row weights.
#' @return A file-backed directed weighted-graph descriptor.
#' @export
massive_umap_memberships <- function(x, checkpoint = FALSE,
        resume = FALSE, checkpoint_every = 100L,
        backend = "cpu", n.cores = 1L) {
    if (!identical(backend, "cpu")) {
        stop("EXPERIMENTAL directed UMAP weighting requires CPU; ",
            "no backend fallback was used.", call. = FALSE)
    }
    if (!inherits(x, "fastEmbedR_massive_graph") ||
        !identical(x$storage, "fixed_knn") ||
        !identical(x$edge_value, "distance")) {
        stop("`x` must be a distance-valued fixed-KNN massive graph.",
            call. = FALSE)
    }
    workers <- integer_scalar(n.cores)
    if (is.na(workers) || workers < 1L || workers > 256L) stop(
        "`n.cores` must be an integer from 1 to 256.", call. = FALSE)
    paths <- c(x$indices_path, x$values_path)
    massive_checkpoint_validated_graph_identity(x, paths)
    prefix <- sub("\\.indices\\.u32$", "", x$indices_path)
    if (identical(prefix, x$indices_path)) stop(
        "Massive KNN indices path has an invalid suffix.",
        call. = FALSE)
    output <- paste0(prefix, ".weights.f32")
    prepared <- massive_umap_membership_prepare(x, output,
        checkpoint, resume, checkpoint_every)
    message("EXPERIMENTAL directed UMAP memberships: ",
        format(x$n_edges, scientific = FALSE),
        " edges; ", if (resume) "resuming checkpoint" else
            "two bounded graph scans")
    massive_umap_memberships_cpp(x$indices_path, x$values_path,
        x$n_vertices, x$k, output, prepared$mean, prepared$rows,
        prepared$progress, prepared$resume_part, checkpoint_every,
        workers)
    massive_checkpoint_validated_graph_identity(x, paths)
    graph <- massive_weighted_graph(prefix, x$n_vertices, x$k,
        access = x$access)
    graph$backend <- "cpu"
    graph$n.cores <- workers
    graph$weight_method <- "umap_directed_membership"
    graph$source_identity <- x$file_identity
    if (!is.null(prepared$sidecar)) unlink(prepared$sidecar)
    graph
}

massive_umap_membership_prepare <- function(x, output, checkpoint,
        resume, checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    if (!is.numeric(checkpoint_every) ||
        length(checkpoint_every) != 1L ||
        !is.finite(checkpoint_every) || checkpoint_every < 1L ||
        checkpoint_every != floor(checkpoint_every) ||
        checkpoint_every > .Machine$integer.max) {
        stop("`checkpoint_every` must be a positive integer.",
            call. = FALSE)
    }
    part <- paste0(output, ".part")
    massive_float_output(output, x$n_edges * 4,
        resume = resume && file.exists(part))
    sidecar <- if (checkpoint) paste0(output, ".checkpoint.rds")
    if (!checkpoint) return(list(
        mean = massive_umap_global_mean_cpp(x$indices_path,
            x$values_path, x$n_vertices, x$k),
        rows = 0, progress = NULL, resume_part = FALSE,
        sidecar = NULL))
    signature <- massive_checkpoint_signature(list(version = 2L,
        source = x$file_identity, output = output,
        n_vertices = x$n_vertices, k = x$k,
        checkpoint_every = checkpoint_every))
    state <- massive_umap_membership_checkpoint(x, sidecar, part,
        signature, resume, checkpoint_every)
    progress <- function(rows) {
        state$completed_rows <- rows
        massive_checkpoint_write(state, sidecar)
    }
    list(mean = state$mean, rows = state$completed_rows,
        progress = progress, resume_part = file.exists(part),
        sidecar = sidecar)
}

massive_umap_membership_checkpoint <- function(x, sidecar, part,
        signature, resume, checkpoint_every) {
    if (resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        massive_umap_membership_check_state(state, signature, x, part)
    } else {
        if (file.exists(sidecar)) stop("UMAP checkpoint exists; use ",
            "`resume = TRUE`.", call. = FALSE)
        state <- list(signature = signature, stage = "mean",
            completed_rows = 0, mean_rows = 0, mean_sum = "0")
        state$mean_hash <- massive_checkpoint_signature(list(
            rows = state$mean_rows, sum = state$mean_sum))
        massive_checkpoint_write(state, sidecar)
    }
    if (identical(state$stage, "write")) return(state)
    progress <- function(rows, sum) {
        state$mean_rows <- rows
        state$mean_sum <- sum
        state$mean_hash <- massive_checkpoint_signature(list(
            rows = rows, sum = sum))
        massive_checkpoint_write(state, sidecar)
    }
    mean <- massive_umap_global_mean_resume_cpp(x$indices_path,
        x$values_path, x$n_vertices, x$k, state$mean_rows,
        state$mean_sum, progress, as.integer(checkpoint_every))
    state <- list(signature = signature, stage = "write",
        completed_rows = 0, mean = mean)
    massive_checkpoint_write(state, sidecar)
    state
}

massive_umap_membership_check_state <- function(state, signature,
        x, part) {
    if (!is.list(state) || !identical(state$signature, signature) ||
        !is.character(state$stage) || length(state$stage) != 1L ||
        is.na(state$stage) || !state$stage %in% c("mean", "write")) {
        stop("UMAP membership checkpoint does not match inputs.",
            call. = FALSE)
    }
    if (identical(state$stage, "mean")) {
        block <- max(1L, 65536L %/% x$k)
        rows <- state$mean_rows
        if (!is.numeric(rows) || length(rows) != 1L ||
            !is.finite(rows) || rows != floor(rows) || rows < 0 ||
            rows > x$n_vertices ||
            (rows != x$n_vertices && rows %% block != 0) ||
            !is.character(state$mean_sum) ||
            length(state$mean_sum) != 1L ||
            !identical(state$completed_rows, 0) ||
            file.exists(part)) {
            stop("UMAP membership mean checkpoint does not match inputs.",
                call. = FALSE)
        }
        expected <- massive_checkpoint_signature(list(
            rows = rows, sum = state$mean_sum))
        if (!identical(state$mean_hash, expected)) stop(
            "UMAP membership mean checkpoint does not match inputs.",
            call. = FALSE)
        return(invisible(NULL))
    }
    if (!is.numeric(state$mean) || length(state$mean) != 1L ||
        !is.finite(state$mean) || state$mean < 0 ||
        !is.numeric(state$completed_rows) ||
        length(state$completed_rows) != 1L) {
        stop("UMAP membership checkpoint does not match inputs.",
            call. = FALSE)
    }
    rows <- state$completed_rows
    block <- max(1L, 65536L %/% x$k)
    bytes <- if (file.exists(part)) file.info(part)$size else 0
    if (!is.finite(rows) || rows != floor(rows) || rows < 0 ||
        rows > x$n_vertices ||
        (rows != x$n_vertices && rows %% block != 0) ||
        is.na(bytes) || bytes < rows * x$k * 4 ||
        bytes > x$n_edges * 4 ||
        (rows > 0 && !file.exists(part))) {
        stop("UMAP membership partial file disagrees with checkpoint.",
            call. = FALSE)
    }
}

#' Import an experimental file-backed weighted graph
#'
#' Imports `prefix.indices.u32` and `prefix.weights.f32` without copying
#' them. Each row must have `k` distinct non-self neighbors and float32
#' weights between 0 and 1. This is a directed, externally supplied graph;
#' import
#' does not construct or verify a symmetric UMAP fuzzy graph.
#'
#' @inheritParams massive_knn_graph
#' @return A lightweight `fastEmbedR_massive_graph` descriptor with
#'   `edge_value = "weight"`.
#' @export
massive_weighted_graph <- function(prefix, n_vertices, k,
                                    access = c("stream", "mmap")) {
    massive_graph_import(prefix, n_vertices, k, match.arg(access),
        "weight")
}

#' Symmetrize an experimental disk-backed UMAP graph
#'
#' Externally sorts directed membership pairs in bounded native buffers,
#' applies fuzzy union, and writes a symmetric weighted CSR graph and a
#' content-hashed manifest. With `sort_method = "bucketed"`, distances
#' from a KNN graph are converted to memberships while routing pairs into
#' bounded vertex buckets; no directed weight file is written. The bucketed
#' path does not yet support checkpoints. Both sorts are CPU-only.
#' Use [massive_fuzzy_graph()] after a restart.
#'
#' @param x Directed memberships from [massive_umap_memberships()] for the
#'   global sort, or KNN distances from [massive_knn_graph()] for the
#'   bucketed sort.
#' @param output New prefix for `.offsets.u64`, `.indices.u32`, and
#'   `.weights.f32` output files.
#' @param memory_limit RAM budget for external sorting, at least 64 MB.
#' @param backend Only `"cpu"` is currently supported.
#' @param checkpoint Save completed sorted-pair runs for resume.
#' @param resume Continue a matching interrupted graph sort.
#' @param checkpoint_every Save progress after this many input blocks.
#' @param sort_method Global restartable sort or lower-SSD bucketed sort.
#' @param consume_knn Explicitly delete package-built KNN files after
#'   bucket staging. The KNN graph cannot then be reused or resumed;
#'   rebuilding it is required if later graph construction fails.
#' @return An experimental symmetric `fastEmbedR_massive_graph` descriptor.
#' @export
massive_umap_fuzzy_graph <- function(x, output,
        memory_limit = "8GB", backend = "cpu", checkpoint = FALSE,
        resume = FALSE, checkpoint_every = 100L,
        sort_method = c("global", "bucketed"),
        consume_knn = FALSE) {
    sort_method <- match.arg(sort_method)
    if (!identical(backend, "cpu")) stop(
        "EXPERIMENTAL fuzzy graph sorting requires CPU; ",
        "no backend fallback was used.", call. = FALSE)
    massive_fuzzy_sort_input(x, sort_method, checkpoint, resume)
    release_input <- massive_graph_release_knn(x, consume_knn, sort_method)
    plan <- massive_umap_fuzzy_preflight(x, output, memory_limit,
        resume = resume, bucketed = sort_method == "bucketed",
        consume_knn = consume_knn)
    prefix <- plan$prefix
    paths <- c(x$indices_path, x$values_path)
    massive_checkpoint_validated_graph_identity(x, paths)
    message("EXPERIMENTAL fuzzy graph sort: ", x$n_edges,
        " directed edges; initial disk admission=",
        round(plan$disk_bytes / 1024^3, 2), " GiB")
    method <- if (sort_method == "bucketed") "umap_bucketed" else "umap"
    sorted <- massive_graph_sort_run(x, plan, method, 0, 1L,
        checkpoint, resume, checkpoint_every, release_input)
    built <- sorted$built
    if (!consume_knn) massive_checkpoint_validated_graph_identity(x, paths)
    graph <- massive_csr_graph(prefix, x$n_vertices, x$access)
    if (graph$n_edges != built$n_edges) stop(
        "Fuzzy graph edge count changed after construction.",
        call. = FALSE)
    graph$symmetrized <- TRUE; graph$backend <- "cpu"
    graph$weight_method <- "umap_fuzzy_union"
    graph$sort_method <- sort_method; graph$source_k <- x$k
    graph$source_identity <- x$file_identity
    graph$consumed_knn_bytes <- built$released_input_bytes
    graph$sort_resources <- massive_graph_sort_resources(
        built$peak_sort_payload_bytes, built$disk_budget_bytes)
    graph$peak_graph_stage_bytes <- built$peak_with_knn_bytes
    graph$manifest_path <- paste0(prefix, ".fuzzy.rds")
    manifest <- list(version = 1L, method = graph$weight_method,
        n_vertices = graph$n_vertices, n_edges = graph$n_edges,
        max_degree = graph$max_degree, source_k = graph$source_k,
        sort_method = sort_method,
        consumed_knn_bytes = graph$consumed_knn_bytes,
        peak_graph_stage_bytes = graph$peak_graph_stage_bytes,
        sort_resources = graph$sort_resources,
        hashes = massive_graph_hash(graph$file_identity))
    massive_checkpoint_write(manifest, graph$manifest_path)
    if (!is.null(sorted$sidecar)) unlink(sorted$sidecar)
    graph
}

massive_graph_release_knn <- function(x, consume_knn, sort_method) {
    if (!is.logical(consume_knn) || length(consume_knn) != 1L ||
        is.na(consume_knn)) stop("`consume_knn` must be TRUE or FALSE.",
            call. = FALSE)
    if (!consume_knn) return(NULL)
    if (sort_method != "bucketed" || is.null(x$source_identity) ||
        !identical(x$implementation_identity,
            massive_checkpoint_implementation_identity()) ||
        !identical(x$edge_value, "distance")) stop(
        "KNN consumption requires a package-built bucketed KNN graph.",
        call. = FALSE)
    paths <- c(x$indices_path, x$values_path)
    massive_checkpoint_validated_graph_identity(x, paths)
    bytes <- sum(file.info(paths)$size)
    function() {
        sizes <- file.info(paths)$size
        if (anyNA(sizes) || any(sizes != 0)) stop(
            "Consumed KNN files were not fully staged.",
            call. = FALSE)
        if (unlink(paths) != 0L || any(file.exists(paths))) stop(
            "Could not retire the consumed KNN files.", call. = FALSE)
        as.double(bytes)
    }
}

massive_fuzzy_sort_input <- function(x, method, checkpoint, resume) {
    weighted <- identical(x$weight_method,
        "umap_directed_membership")
    distances <- identical(x$edge_value, "distance")
    if (!inherits(x, "fastEmbedR_massive_graph") ||
        !identical(x$storage, "fixed_knn") ||
        !(if (method == "global") weighted else distances)) stop(
        "Global sorting needs directed UMAP memberships; ",
        "bucketed sorting needs KNN distances.", call. = FALSE)
    if (method == "bucketed" && (checkpoint || resume)) stop(
        "Bucketed graph sorting does not support checkpoints.",
        call. = FALSE)
    invisible(NULL)
}

massive_graph_hash <- function(identity) {
    hashes <- unname(tools::md5sum(identity$paths))
    if (length(hashes) != 3L || anyNA(hashes) ||
        !identical(identity,
            massive_checkpoint_file_identity(identity$paths))) {
        stop("Massive graph files changed during hashing.",
            call. = FALSE)
    }
    hashes
}

massive_graph_sort_resources <- function(peak, budget) {
    if (!is.numeric(peak) || length(peak) != 1L ||
        !is.finite(peak) || peak < 0 ||
        !is.numeric(budget) || length(budget) != 1L ||
        !is.finite(budget) || budget < peak) stop(
        "Graph sort disk accounting is invalid.", call. = FALSE)
    list(peak_payload_bytes = peak, budget_bytes = budget)
}

massive_graph_manifest_knn_bytes <- function(manifest, graph) {
    bytes <- manifest$consumed_knn_bytes %||% 0
    if (!is.numeric(bytes) || length(bytes) != 1L ||
        !is.finite(bytes) ||
        !bytes %in% c(0, 8 * graph$n_vertices * manifest$source_k)) {
        stop("Invalid consumed KNN byte count in graph manifest.",
            call. = FALSE)
    }
    bytes
}

massive_graph_manifest_stage_peak <- function(manifest, graph) {
    peak <- manifest$peak_graph_stage_bytes
    if (is.null(peak)) return(NULL)
    lower <- graph$sort_resources$peak_payload_bytes
    upper <- lower + 8 * graph$n_vertices * manifest$source_k
    if (!is.numeric(peak) || length(peak) != 1L ||
        !is.finite(peak) || peak < lower || peak > upper) stop(
        "Invalid graph-stage peak in manifest.", call. = FALSE)
    peak
}

#' Reload a package-built experimental fuzzy UMAP graph
#'
#' Validates every CSR row and checks content hashes against the manifest
#' written by [massive_umap_fuzzy_graph()]. Externally supplied CSR files
#' without that manifest remain unverified. Reimport scans the graph twice
#' but does not reconstruct nearest neighbors or fuzzy memberships.
#'
#' @inheritParams massive_csr_graph
#' @return A verified `fastEmbedR_massive_graph` descriptor.
#' @export
massive_fuzzy_graph <- function(prefix, n_vertices,
                                access = c("stream", "mmap")) {
    graph <- massive_csr_graph(prefix, n_vertices, access)
    path <- paste0(sub("\\.offsets\\.u64$", "",
        graph$offsets_path), ".fuzzy.rds")
    manifest <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(manifest) ||
        !identical(manifest$version, 1L) ||
        !identical(manifest$method, "umap_fuzzy_union") ||
        !identical(manifest$n_vertices, graph$n_vertices) ||
        !identical(manifest$n_edges, graph$n_edges) ||
        !identical(manifest$max_degree, graph$max_degree) ||
        !is.numeric(manifest$source_k) || length(manifest$source_k) != 1L ||
        !is.finite(manifest$source_k) ||
        manifest$source_k < 1 ||
        manifest$source_k != floor(manifest$source_k) ||
        !is.character(manifest$hashes) ||
        length(manifest$hashes) != 3L ||
        anyNA(manifest$hashes)) {
        stop("Missing or invalid fuzzy graph manifest.",
            call. = FALSE)
    }
    if (!identical(manifest$hashes,
        massive_graph_hash(graph$file_identity))) stop(
        "Fuzzy graph content differs from its manifest.",
        call. = FALSE)
    graph$symmetrized <- TRUE
    graph$consumed_knn_bytes <- massive_graph_manifest_knn_bytes(
        manifest, graph)
    graph$backend <- "cpu"
    graph$weight_method <- manifest$method
    graph$sort_method <- manifest$sort_method %||% "global"
    if (!graph$sort_method %in% c("global", "bucketed")) stop(
        "Invalid fuzzy graph sort method.", call. = FALSE)
    graph$source_k <- manifest$source_k
    if (!is.null(manifest$sort_resources)) {
        reported <- manifest$sort_resources
        graph$sort_resources <- massive_graph_sort_resources(
            reported$peak_payload_bytes, reported$budget_bytes)
        if (!identical(reported, graph$sort_resources)) stop(
            "Graph sort disk accounting is invalid.", call. = FALSE)
    }
    graph$peak_graph_stage_bytes <- massive_graph_manifest_stage_peak(
        manifest, graph)
    graph$manifest_path <- path; graph
}

#' Construct experimental file-backed t-SNE affinities
#'
#' Streams every row of a validated full KNN graph, computes the same
#' conditional probabilities as native float32 t-SNE, externally sorts
#' directed pairs, and writes symmetric normalized CSR affinities. The
#' complete graph is never materialized in memory. A content-hashed manifest
#' permits [massive_affinity_graph()] to reload the result. The saved graph
#' can feed the experimental CPU [massive_tsne_optimize()] route.
#'
#' @param x A distance-valued [massive_knn_graph()] descriptor.
#' @param perplexity Positive perplexity no larger than the KNN width.
#' @param output New CSR output prefix.
#' @param memory_limit RAM budget for external sorting, at least 64 MB.
#' @param backend Only `"cpu"` is supported for graph construction.
#' @param n.cores CPU workers for independent conditional-probability rows.
#' @param checkpoint Save completed sorted-pair runs for resume.
#' @param resume Continue a matching interrupted graph sort.
#' @param checkpoint_every Save progress after this many input blocks.
#' @param sort_method Global restartable sort or lower-SSD bucketed sort.
#' @param consume_knn Explicitly delete package-built KNN files after
#'   bucket staging. Rebuilding KNN is required if construction fails.
#' @return An experimental symmetric `fastEmbedR_massive_graph` descriptor.
#' @export
massive_tsne_affinities <- function(x, perplexity, output,
        memory_limit = "8GB", backend = "cpu", n.cores = 1L,
        checkpoint = FALSE, resume = FALSE,
        checkpoint_every = 100L,
        sort_method = c("global", "bucketed"),
        consume_knn = FALSE) {
    sort_method <- match.arg(sort_method)
    if (sort_method == "bucketed" && (checkpoint || resume)) stop(
        "Bucketed graph sorting does not support checkpoints.",
        call. = FALSE)
    workers <- massive_tsne_affinity_controls(x, perplexity,
        backend, n.cores)
    release_input <- massive_graph_release_knn(x, consume_knn,
        sort_method)
    paths <- c(x$indices_path, x$values_path)
    massive_checkpoint_validated_graph_identity(x, paths)
    plan <- massive_umap_fuzzy_preflight(x, output, memory_limit,
        ".affinity.rds", resume, sort_method == "bucketed",
        consume_knn)
    method <- if (sort_method == "bucketed") "tsne_bucketed" else "tsne"
    sorted <- massive_graph_sort_run(x, plan, method, perplexity,
        workers, checkpoint, resume, checkpoint_every, release_input)
    if (!consume_knn) massive_checkpoint_validated_graph_identity(x, paths)
    graph <- massive_csr_graph(plan$prefix, x$n_vertices, x$access)
    if (graph$n_edges != sorted$built$n_edges) stop(
        "t-SNE affinity edge count changed after construction.",
        call. = FALSE)
    graph$symmetrized <- TRUE
    graph$weight_method <- "tsne_compact_affinity"
    graph$perplexity <- perplexity; graph$source_k <- x$k
    graph$source_identity <- x$file_identity
    graph$consumed_knn_bytes <- sorted$built$released_input_bytes
    graph$backend <- "cpu"; graph$n.cores <- workers
    graph$sort_method <- sort_method
    graph$sort_resources <- massive_graph_sort_resources(
        sorted$built$peak_sort_payload_bytes, sorted$built$disk_budget_bytes)
    graph$peak_graph_stage_bytes <- sorted$built$peak_with_knn_bytes
    graph$manifest_path <- paste0(plan$prefix, ".affinity.rds")
    manifest <- list(version = 1L, method = graph$weight_method,
        n_vertices = graph$n_vertices, n_edges = graph$n_edges,
        max_degree = graph$max_degree, source_k = graph$source_k,
        perplexity = perplexity, sort_method = sort_method,
        consumed_knn_bytes = graph$consumed_knn_bytes,
        peak_graph_stage_bytes = graph$peak_graph_stage_bytes,
        sort_resources = graph$sort_resources,
        hashes = massive_graph_hash(graph$file_identity))
    massive_checkpoint_write(manifest, graph$manifest_path)
    if (!is.null(sorted$sidecar)) unlink(sorted$sidecar)
    graph
}

massive_tsne_affinity_controls <- function(x, perplexity,
        backend, n.cores) {
    if (!identical(backend, "cpu")) stop(
        "EXPERIMENTAL t-SNE affinities require CPU; ",
        "no backend fallback was used.", call. = FALSE)
    if (!inherits(x, "fastEmbedR_massive_graph") ||
        !identical(x$storage, "fixed_knn") ||
        !identical(x$edge_value, "distance")) stop(
        "`x` must be a distance-valued fixed-KNN massive graph.",
        call. = FALSE)
    if (!is.numeric(perplexity) || length(perplexity) != 1L ||
        !is.finite(perplexity) || perplexity <= 0 ||
        perplexity > x$k) stop(
        "`perplexity` must be positive and at most the KNN width.",
        call. = FALSE)
    workers <- integer_scalar(n.cores)
    if (is.na(workers) || workers < 1L || workers > 256L) stop(
        "`n.cores` must be an integer from 1 to 256.", call. = FALSE)
    workers
}

massive_umap_fuzzy_preflight <- function(x, output, memory_limit,
        manifest_suffix = ".fuzzy.rds", resume = FALSE,
        bucketed = FALSE, consume_knn = FALSE) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) {
        stop("`output` must be one new graph prefix.", call. = FALSE)
    }
    limit <- massive_memory_bytes(memory_limit)
    if (limit < 64 * 1024^2) stop(
        "Graph sorting needs at least 64 MB RAM budget.",
        call. = FALSE)
    prefix <- file.path(normalizePath(dirname(output),
        mustWork = TRUE), basename(output))
    outputs <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32", manifest_suffix))
    work <- paste0(prefix, ".work")
    sidecar <- paste0(prefix, ".sort-checkpoint.rds")
    pending <- c(paste0(outputs, ".part"), work, sidecar)
    if (any(file.exists(outputs)) ||
        (!resume && any(file.exists(pending))) ||
        (resume && (!file.exists(sidecar) ||
            (file.exists(work) && !dir.exists(work))))) stop(
        "Graph output or work files already exist.", call. = FALSE)
    needed <- (if (bucketed && consume_knn) 4 else if (bucketed)
        12 else 19) * x$n_edges + 8 * (x$n_vertices + 1) +
        (if (bucketed && consume_knn) 12 * 65536 else 0)
    existing <- c(list.files(work, full.names = TRUE),
        pending[file.exists(pending) & !dir.exists(pending)])
    occupied <- if (length(existing)) sum(file.info(existing)$size)
        else 0
    if (max(0, needed - occupied) >
        0.8 * massive_disk_available_cpp(prefix)) stop(
        "Graph temporary files exceed the free-disk budget.",
        call. = FALSE)
    paths <- c(x$indices_path, x$values_path)
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) stop(
        "Massive membership files changed since import.",
        call. = FALSE)
    list(prefix = prefix, limit = limit, disk_bytes = needed)
}

massive_graph_sort_signature <- function(x, plan, method,
        perplexity, workers, checkpoint_every) {
    dll <- getLoadedDLLs()[["fastEmbedR"]]
    if (is.null(dll)) stop(
        "Cannot identify the native graph sorter.", call. = FALSE)
    native <- unname(tools::md5sum(dll[["path"]]))
    if (is.na(native)) stop(
        "Cannot hash the native graph sorter.", call. = FALSE)
    massive_checkpoint_signature(list(version = 1L,
        input = x$file_identity, n_vertices = x$n_vertices,
        k = x$k, output = plan$prefix, memory_limit = plan$limit,
        method = method, perplexity = perplexity,
        workers = workers, checkpoint_every = checkpoint_every,
        native = native))
}

massive_graph_sort_run_identity <- function(state, x, work) {
    runs <- state$runs
    phase <- state$phase %||% "pairs_input"
    names <- basename(runs$paths)
    reduced <- identical(phase, "pairs_reduced")
    generated <- identical(phase, "rows_generated")
    rows_reduced <- identical(phase, "rows_reduced")
    expected <- if (reduced || generated || rows_reduced) {
        paste0(work, names)
    } else if (length(names)) {
        paste0(work, "pairs_run_", seq_along(names) - 1L,
            ".bin")
    } else character()
    names_ok <- if (reduced) {
        all(grepl("^pairs_(run_[0-9]+|merge_[0-9]+_0)[.]bin$",
            names))
    } else if (generated) {
        all(grepl("^rows_run_[0-9]+[.]bin$", names))
    } else if (rows_reduced) {
        all(grepl("^rows_(run_[0-9]+|merge_[0-9]+_0)[.]bin$",
            names))
    } else TRUE
    bytes <- sum(runs$bytes)
    sizes_ok <- if (generated || rows_reduced) {
        bytes > 0 && bytes %% 12 == 0 &&
            bytes <= 24 * x$n_edges
    } else bytes == state$completed_rows * x$k * 12
    phase %in% c("pairs_input", "pairs_reduced",
        "rows_generated", "rows_reduced") && names_ok &&
        !anyDuplicated(names) &&
        (phase == "pairs_input" ||
            (state$completed_rows == x$n_vertices && length(names))) &&
        massive_checkpoint_same_paths(runs$paths, expected) &&
        is.numeric(runs$bytes) && all(is.finite(runs$bytes)) &&
        all(runs$bytes > 0) && sizes_ok &&
        identical(runs, tryCatch(massive_checkpoint_file_identity(
            runs$paths), error = function(e) NULL))
}

massive_graph_sort_state <- function(x, plan, signature, resume) {
    sidecar <- paste0(plan$prefix, ".sort-checkpoint.rds")
    if (!resume) {
        state <- list(signature = signature, completed_rows = 0,
            runs = list(paths = character(), bytes = numeric(),
                modified = numeric()))
        massive_checkpoint_write(state, sidecar)
        return(state)
    }
    state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
    if (!is.list(state) || !is.list(state$runs)) stop(
        "Graph sort checkpoint is unreadable.", call. = FALSE)
    rows <- state$completed_rows
    phase <- state$phase %||% "pairs_input"
    block <- max(1L, 65536L %/% x$k)
    work <- paste0(plan$prefix, ".work/")
    valid <- identical(state$signature, signature) &&
        is.numeric(rows) && length(rows) == 1L &&
        is.finite(rows) && rows >= 0 && rows <= x$n_vertices &&
        rows == floor(rows) &&
        (rows == x$n_vertices || rows %% block == 0) &&
        (!phase %in% c("rows_generated", "rows_reduced") ||
            (is.numeric(state$total_mass) &&
            length(state$total_mass) == 1L &&
            is.finite(state$total_mass) &&
            state$total_mass >= 0)) &&
        massive_graph_sort_run_identity(state, x, work)
    if (!isTRUE(valid)) stop(
        "Graph sort checkpoint does not match inputs or runs.",
        call. = FALSE)
    state
}

massive_graph_sort_phase_progress <- function(state, sidecar, done,
        paths, phase, total_mass, k) {
    previous <- state$phase %||% "pairs_input"
    allowed <- switch(phase,
        pairs_reduced = previous %in% c("pairs_input",
            "pairs_reduced"),
        rows_generated = previous %in% c("pairs_input",
            "pairs_reduced"),
        rows_reduced = previous %in% c("rows_generated",
            "rows_reduced"), FALSE)
    previous_bytes <- sum(state$runs$bytes)
    if (!allowed || done != state$completed_rows ||
        !length(paths)) stop(
        "Graph sort phase is inconsistent.", call. = FALSE)
    state$runs <- massive_checkpoint_file_identity(paths)
    expected <- if (phase == "pairs_reduced") {
        done * k * 12
    } else if (phase == "rows_reduced") {
        previous_bytes
    } else sum(state$runs$bytes)
    if (sum(state$runs$bytes) != expected ||
        (phase %in% c("rows_generated", "rows_reduced") &&
        (expected <= 0 || expected %% 12 != 0 ||
        expected > 24 * done * k ||
        !is.numeric(total_mass) || length(total_mass) != 1L ||
        !is.finite(total_mass) || total_mass < 0))) stop(
        "Reduced graph runs have invalid sizes.", call. = FALSE)
    state$phase <- phase
    if (phase %in% c("rows_generated", "rows_reduced")) {
        state$total_mass <- total_mass
    }
    massive_checkpoint_write(state, sidecar)
    state
}

massive_graph_sort_progress <- function(plan, state, k) {
    sidecar <- paste0(plan$prefix, ".sort-checkpoint.rds")
    function(done, paths, phase = "pairs_input", total_mass = NULL) {
        next_state <- state
        if (phase %in% c("pairs_reduced", "rows_generated",
            "rows_reduced")) {
            state <<- massive_graph_sort_phase_progress(next_state,
                sidecar, done, paths, phase, total_mass, k)
            return(invisible(NULL))
        }
        old <- next_state$runs$paths
        if (done < next_state$completed_rows ||
            length(paths) < length(old) ||
            !massive_checkpoint_same_paths(
                utils::head(paths, length(old)), old)) stop(
            "Graph sort progress is inconsistent.", call. = FALSE)
        fresh <- if (length(paths) > length(old)) {
            paths[seq.int(length(old) + 1L, length(paths))]
        } else character()
        if (length(fresh)) {
            added <- massive_checkpoint_file_identity(fresh)
            for (key in names(next_state$runs)) {
                next_state$runs[[key]] <- c(next_state$runs[[key]],
                    added[[key]])
            }
        }
        if (sum(next_state$runs$bytes) != done * k * 12) stop(
            "Graph sort run sizes disagree with input progress.",
            call. = FALSE)
        next_state$completed_rows <- done
        massive_checkpoint_write(next_state, sidecar)
        state <<- next_state
    }
}

massive_graph_sort_run <- function(x, plan, method, perplexity,
        workers, checkpoint, resume, checkpoint_every,
        release_input = NULL) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L) stop(
        "`checkpoint_every` must be positive.", call. = FALSE)
    state <- if (checkpoint) {
        signature <- massive_graph_sort_signature(x, plan,
            method, perplexity, workers, every)
        massive_graph_sort_state(x, plan, signature, resume)
    } else NULL
    if (!is.null(release_input)) on.exit({
        paths <- c(x$indices_path, x$values_path)
        sizes <- file.info(paths)$size
        if (any(!is.na(sizes) & sizes != x$file_bytes_each)) {
            unlink(paths)
        }
    }, add = TRUE)
    progress <- if (checkpoint) massive_graph_sort_progress(
        plan, state, x$k) else NULL
    built <- massive_symmetrize_graph_cpp(x$indices_path,
        x$values_path, x$n_vertices, x$k, plan$prefix,
        plan$limit, method, perplexity, workers, resume,
        state$completed_rows %||% 0, state$runs$paths %||%
            character(), state$phase %||% "pairs_input",
        state$total_mass %||% 0,
        if (checkpoint) every else 0L,
        progress, release_input)
    list(built = built, sidecar = if (checkpoint)
        paste0(plan$prefix, ".sort-checkpoint.rds") else NULL)
}

#' Reload package-built experimental t-SNE affinities
#'
#' Validates every CSR row and checks file contents against the manifest
#' written by [massive_tsne_affinities()]. A generic external CSR graph
#' cannot be treated as a package-built t-SNE affinity graph.
#'
#' @inheritParams massive_csr_graph
#' @return A verified `fastEmbedR_massive_graph` descriptor.
#' @export
massive_affinity_graph <- function(prefix, n_vertices,
                                    access = c("stream", "mmap")) {
    graph <- massive_csr_graph(prefix, n_vertices, access)
    path <- paste0(sub("\\.offsets\\.u64$", "",
        graph$offsets_path), ".affinity.rds")
    manifest <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(manifest) ||
        !identical(manifest$version, 1L) ||
        !identical(manifest$method, "tsne_compact_affinity") ||
        !identical(manifest$n_vertices, graph$n_vertices) ||
        !identical(manifest$n_edges, graph$n_edges) ||
        !identical(manifest$max_degree, graph$max_degree) ||
        !is.numeric(manifest$source_k) || length(manifest$source_k) != 1L ||
        !is.finite(manifest$source_k) ||
        manifest$source_k < 1 ||
        manifest$source_k != floor(manifest$source_k) ||
        !is.numeric(manifest$perplexity) ||
        length(manifest$perplexity) != 1L ||
        !is.finite(manifest$perplexity) ||
        manifest$perplexity <= 0 ||
        manifest$perplexity > manifest$source_k ||
        !is.character(manifest$hashes) ||
        length(manifest$hashes) != 3L ||
        anyNA(manifest$hashes)) {
        stop("Missing or invalid t-SNE affinity manifest.",
            call. = FALSE)
    }
    if (!identical(manifest$hashes,
        massive_graph_hash(graph$file_identity))) stop(
        "t-SNE affinity content differs from its manifest.",
        call. = FALSE)
    graph$symmetrized <- TRUE; graph$backend <- "cpu"
    graph$consumed_knn_bytes <- massive_graph_manifest_knn_bytes(
        manifest, graph)
    graph$weight_method <- manifest$method
    graph$source_k <- manifest$source_k; graph$perplexity <- manifest$perplexity
    graph$sort_method <- manifest$sort_method %||% "global"
    if (!graph$sort_method %in% c("global", "bucketed")) stop(
        "Invalid affinity graph sort method.", call. = FALSE)
    if (!is.null(manifest$sort_resources)) {
        reported <- manifest$sort_resources
        graph$sort_resources <- massive_graph_sort_resources(
            reported$peak_payload_bytes, reported$budget_bytes)
        if (!identical(reported, graph$sort_resources)) stop(
            "Graph sort disk accounting is invalid.", call. = FALSE)
    }
    graph$peak_graph_stage_bytes <- massive_graph_manifest_stage_peak(
        manifest, graph)
    graph$manifest_path <- path; graph
}

#' Import an experimental file-backed weighted CSR graph
#'
#' Reads `prefix.offsets.u64`, `prefix.indices.u32`, and
#' `prefix.weights.f32`. Offsets are zero-based edge positions; target IDs
#' are one-based. Neighbor IDs must be strictly increasing within each row,
#' and weights must lie between zero and one. Every row and edge is
#' validated by a
#' bounded sequential scan. Import does not construct or certify a symmetric
#' fuzzy UMAP graph. At least one edge is required.
#'
#' @param prefix Prefix of the existing CSR file triple.
#' @param n_vertices Number of vertices; currently at most 2^31 - 1.
#' @param access Sequential file reads or memory-mapped access for later
#'   bounded edge reads. Import validation always streams.
#' @return A lightweight `fastEmbedR_massive_graph` descriptor.
#' @export
massive_csr_graph <- function(prefix, n_vertices,
                                access = c("stream", "mmap")) {
    if (!is.character(prefix) || length(prefix) != 1L ||
        is.na(prefix) || !nzchar(prefix)) {
        stop("`prefix` must be one graph file prefix.", call. = FALSE)
    }
    if (!is.numeric(n_vertices) || length(n_vertices) != 1L ||
        !is.finite(n_vertices) || n_vertices < 2 ||
        n_vertices != floor(n_vertices) ||
        n_vertices > .Machine$integer.max) {
        stop("`n_vertices` must be between 2 and 2^31 - 1.",
            call. = FALSE)
    }
    paths <- normalizePath(paste0(prefix, c(".offsets.u64",
        ".indices.u32", ".weights.f32")), mustWork = TRUE)
    validated_input_identity <-
        massive_checkpoint_input_files_identity(paths)
    checked <- massive_validate_csr_graph_cpp(paths[[1L]],
        paths[[2L]], paths[[3L]], n_vertices)
    if (!identical(validated_input_identity,
        massive_checkpoint_input_files_identity(paths))) {
        stop("Massive graph files changed during validation.",
            call. = FALSE)
    }
    identity <- validated_input_identity[c("paths", "bytes", "modified")]
    graph <- list(offsets_path = paths[[1L]],
        indices_path = paths[[2L]], weights_path = paths[[3L]],
        values_path = paths[[3L]], n_vertices = checked$n_vertices,
        n_edges = checked$n_edges, max_degree = checked$max_degree,
        storage = "csr", metric = "external", index_dtype = "uint32",
        offset_dtype = "uint64", value_dtype = "float32",
        weight_dtype = "float32", directed = TRUE,
        symmetrized = FALSE, edge_value = "weight",
        backend = "external", access = match.arg(access),
        validation_access = "stream", validated = checked$validated,
        file_identity = identity,
        validated_input_identity = validated_input_identity,
        experimental = TRUE)
    class(graph) <- "fastEmbedR_massive_graph"
    graph
}

massive_graph_owner_edge_counts <- function(graph, devices) {
    count <- length(devices)
    block <- ceiling(graph$n_vertices / count)
    boundaries <- pmin(graph$n_vertices, seq.int(0, count) * block)
    size <- file.info(graph$offsets_path)$size
    if (is.na(size) || size != 8 * (graph$n_vertices + 1)) stop(
        "Graph offsets cannot verify GPU edge ownership.",
        call. = FALSE)
    input <- file(graph$offsets_path, "rb")
    on.exit(close(input), add = TRUE)
    offsets <- vapply(boundaries, function(row) {
        seek(input, where = 8 * row, origin = "start")
        bytes <- readBin(input, raw(), n = 8L)
        if (length(bytes) != 8L) stop(
            "Graph offsets cannot verify GPU edge ownership.",
            call. = FALSE)
        sum(as.double(as.integer(bytes)) * 256^(0:7))
    }, numeric(1L))
    if (offsets[[1L]] != 0 ||
        utils::tail(offsets, 1L) != graph$n_edges ||
        any(diff(offsets) < 0)) stop(
        "Graph offsets cannot verify GPU edge ownership.",
        call. = FALSE)
    diff(offsets)
}

massive_graph_owner_active_counts <- function(graph, devices,
        max_weight, n_epochs) {
    owners <- massive_graph_owner_edge_counts(graph, devices)
    bytes <- file.info(graph$weights_path)$size
    if (is.na(bytes) || bytes != 4 * graph$n_edges ||
        !is.numeric(max_weight) || length(max_weight) != 1L ||
        !is.finite(max_weight) || max_weight <= 0) stop(
        "Graph weights cannot verify GPU edge ownership.",
        call. = FALSE)
    threshold <- readBin(writeBin(max_weight / n_epochs, raw(),
        size = 4L, endian = "little"), numeric(), n = 1L,
        size = 4L, endian = "little")
    input <- file(graph$weights_path, "rb")
    on.exit(close(input), add = TRUE)
    active <- numeric(length(owners))
    largest <- -Inf
    for (i in seq_along(owners)) {
        remaining <- owners[i]
        while (remaining > 0) {
            count <- as.integer(min(1048576, remaining))
            values <- readBin(input, numeric(), n = count,
                size = 4L, endian = "little")
            if (length(values) != count || any(!is.finite(values))) stop(
                "Graph weights cannot verify GPU edge ownership.",
                call. = FALSE)
            largest <- max(largest, values)
            active[i] <- active[i] + sum(values >= threshold)
            remaining <- remaining - count
        }
    }
    if (largest != max_weight) stop(
        "Graph weights cannot verify GPU edge ownership.",
        call. = FALSE)
    active
}

#' Read a bounded edge range from an experimental full KNN graph
#'
#' Reads complete source rows. The result is a directed edge list with
#' one-based `from` and `to` IDs plus either distances or weights, according
#' to the imported graph's `edge_value` metadata.
#'
#' @param x A `fastEmbedR_massive_graph` descriptor.
#' @param first First one-based source row.
#' @param n Number of consecutive source rows. Output is capped at 128 MB.
#' @return A list of `from`, `to`, and `distance` or `weight` vectors.
#' @export
massive_read_graph_edges <- function(x, first = 1, n = 6L) {
    if (!inherits(x, "fastEmbedR_massive_graph")) {
        stop("`x` must be an imported massive graph.",
            call. = FALSE)
    }
    first <- as.numeric(first)
    if (length(first) != 1L || !is.finite(first) || first < 1 ||
        first != floor(first) || !is.numeric(n) ||
        length(n) != 1L || !is.finite(n) || n < 1 ||
        n != floor(n) || n > .Machine$integer.max) {
        stop("`first` and `n` must be positive integer counts.",
            call. = FALSE)
    }
    paths <- if (identical(x$storage, "csr")) {
        c(x$offsets_path, x$indices_path, x$values_path)
    } else c(x$indices_path, x$values_path)
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) {
        stop("Massive graph files changed since import.",
            call. = FALSE)
    }
    if (identical(x$storage, "csr")) {
        return(massive_read_csr_graph_edges_cpp(paths[[1L]],
            paths[[2L]], paths[[3L]], x$n_vertices, x$access,
            first, as.integer(n), 128e6))
    }
    massive_read_graph_edges_cpp(paths[[1L]], paths[[2L]],
        x$n_vertices, x$k, x$access, first, as.integer(n),
        128e6, x$edge_value)
}

#' Plan bounded partitions of an experimental full KNN graph
#'
#' The plan stores only partition arithmetic, not a vector of all partition
#' boundaries or graph edges. The byte limit includes the R edge-list output
#' (one double source, one integer target, and one double value per edge).
#'
#' @param x A `fastEmbedR_massive_graph` descriptor.
#' @param max_bytes Maximum resident edge-list bytes per partition, up to
#'   128 MB. One complete source row must fit.
#' @return A constant-size `fastEmbedR_massive_graph_partitions` descriptor.
#' @export
massive_graph_partitions <- function(x, max_bytes = 64e6) {
    if (!inherits(x, "fastEmbedR_massive_graph")) {
        stop("`x` must be an imported massive graph.",
            call. = FALSE)
    }
    degree <- if (identical(x$storage, "csr")) {
        x$max_degree
    } else x$k
    if (!is.numeric(max_bytes) || length(max_bytes) != 1L ||
        !is.finite(max_bytes) || max_bytes < 20 * degree ||
        max_bytes > 128e6) {
        stop("`max_bytes` must fit one row and not exceed 128 MB.",
            call. = FALSE)
    }
    rows <- floor(max_bytes / (20 * degree))
    plan <- list(graph = x, rows_per_partition = rows,
        n_partitions = ceiling(x$n_vertices / rows),
        max_bytes = max_bytes)
    class(plan) <- "fastEmbedR_massive_graph_partitions"
    plan
}

#' Read one bounded experimental graph partition
#'
#' @param x A `massive_graph_partitions()` result.
#' @param partition One-based partition number.
#' @return A directed edge list with `from`, `to`, and value vectors.
#' @export
massive_read_graph_partition <- function(x, partition) {
    if (!inherits(x, "fastEmbedR_massive_graph_partitions")) {
        stop("`x` must be a massive_graph_partitions() result.",
            call. = FALSE)
    }
    if (!is.numeric(partition) || length(partition) != 1L ||
        !is.finite(partition) || partition < 1 ||
        partition != floor(partition) ||
        partition > x$n_partitions) {
        stop("`partition` is outside the graph partition range.",
            call. = FALSE)
    }
    first <- (partition - 1) * x$rows_per_partition + 1
    count <- as.integer(min(x$rows_per_partition,
        x$graph$n_vertices - first + 1))
    massive_read_graph_edges(x$graph, first, count)
}

#' @export
print.fastEmbedR_massive_graph <- function(x, ...) {
    cat("EXPERIMENTAL fastEmbedR file-backed graph\n")
    cat("  vertices: ", format(x$n_vertices, scientific = FALSE),
        "; directed edges: ", format(x$n_edges, scientific = FALSE),
        "; ", if (identical(x$storage, "csr")) {
            "max degree"
        } else "k", ": ", if (identical(x$storage, "csr")) {
            x$max_degree
        } else x$k, "\n", sep = "")
    cat("  access: ", x$access, "; values: ", x$edge_value,
        "s\n", sep = "")
    invisible(x)
}

#' @export
head.fastEmbedR_massive_graph <- function(x, n = 6L, ...) {
    massive_read_graph_edges(x, n = min(n, x$n_vertices))
}
