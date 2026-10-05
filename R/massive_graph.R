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
    identity <- massive_checkpoint_file_identity(paths)
    checked <- massive_validate_graph_cpp(paths[[1L]], paths[[2L]],
        n_vertices, as.integer(k), "stream", kind)
    if (!identical(identity, massive_checkpoint_file_identity(paths))) {
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
        validated = checked$validated, file_identity = identity,
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
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) stop(
            "Massive KNN graph files changed since import.",
            call. = FALSE)
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
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) {
        stop("Massive KNN graph files changed during weighting.",
            call. = FALSE)
    }
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

#' Symmetrize an experimental disk-backed UMAP membership graph
#'
#' Externally sorts directed membership pairs in bounded native buffers,
#' applies fuzzy union, and writes a symmetric weighted CSR graph and a
#' content-hashed manifest. The sort needs substantial temporary disk space
#' and is currently CPU-only. Use [massive_fuzzy_graph()] after a restart.
#'
#' @param x A result of [massive_umap_memberships()].
#' @param output New prefix for `.offsets.u64`, `.indices.u32`, and
#'   `.weights.f32` output files.
#' @param memory_limit RAM budget for external sorting, at least 64 MB.
#' @param backend Only `"cpu"` is currently supported.
#' @param checkpoint Save completed sorted-pair runs for resume.
#' @param resume Continue a matching interrupted graph sort.
#' @param checkpoint_every Save progress after this many input blocks.
#' @return An experimental symmetric `fastEmbedR_massive_graph` descriptor.
#' @export
massive_umap_fuzzy_graph <- function(x, output,
        memory_limit = "8GB", backend = "cpu", checkpoint = FALSE,
        resume = FALSE, checkpoint_every = 100L) {
    if (!identical(backend, "cpu")) stop(
        "EXPERIMENTAL fuzzy graph sorting requires CPU; ",
        "no backend fallback was used.", call. = FALSE)
    if (!inherits(x, "fastEmbedR_massive_graph") ||
        !identical(x$storage, "fixed_knn") ||
        !identical(x$weight_method, "umap_directed_membership")) {
        stop("`x` must contain directed UMAP memberships.",
            call. = FALSE)
    }
    plan <- massive_umap_fuzzy_preflight(x, output, memory_limit,
        resume = resume)
    prefix <- plan$prefix
    paths <- c(x$indices_path, x$values_path)
    message("EXPERIMENTAL fuzzy graph sort: ",
        format(x$n_edges, scientific = FALSE),
        " directed edges; temporary disk budget=",
        round(plan$disk_bytes / 1024^3, 2), " GiB")
    sorted <- massive_graph_sort_run(x, plan, "umap", 0, 1L,
        checkpoint, resume, checkpoint_every)
    built <- sorted$built
    if (!identical(massive_checkpoint_file_identity(paths),
        x$file_identity)) stop(
        "Massive membership files changed during sorting.",
        call. = FALSE)
    graph <- massive_csr_graph(prefix, x$n_vertices, x$access)
    if (graph$n_edges != built$n_edges) stop(
        "Fuzzy graph edge count changed after construction.",
        call. = FALSE)
    graph$symmetrized <- TRUE
    graph$backend <- "cpu"
    graph$weight_method <- "umap_fuzzy_union"
    graph$source_k <- x$k
    graph$source_identity <- x$file_identity
    graph$manifest_path <- paste0(prefix, ".fuzzy.rds")
    manifest <- list(version = 1L, method = graph$weight_method,
        n_vertices = graph$n_vertices, n_edges = graph$n_edges,
        max_degree = graph$max_degree, source_k = graph$source_k,
        hashes = massive_graph_hash(graph$file_identity))
    massive_checkpoint_write(manifest, graph$manifest_path)
    if (!is.null(sorted$sidecar)) unlink(sorted$sidecar)
    graph
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
    manifest <- if (file.exists(path)) {
        tryCatch(readRDS(path), error = function(e) NULL)
    } else NULL
    if (!is.list(manifest) ||
        !identical(manifest$version, 1L) ||
        !identical(manifest$method, "umap_fuzzy_union") ||
        !identical(manifest$n_vertices, graph$n_vertices) ||
        !identical(manifest$n_edges, graph$n_edges) ||
        !identical(manifest$max_degree, graph$max_degree) ||
        !is.numeric(manifest$source_k) ||
        length(manifest$source_k) != 1L ||
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
    graph$backend <- "cpu"
    graph$weight_method <- manifest$method
    graph$source_k <- manifest$source_k
    graph$manifest_path <- path
    graph
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
#' @return An experimental symmetric `fastEmbedR_massive_graph` descriptor.
#' @export
massive_tsne_affinities <- function(x, perplexity, output,
        memory_limit = "8GB", backend = "cpu", n.cores = 1L,
        checkpoint = FALSE, resume = FALSE,
        checkpoint_every = 100L) {
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
    plan <- massive_umap_fuzzy_preflight(x, output, memory_limit,
        ".affinity.rds", resume)
    sorted <- massive_graph_sort_run(x, plan, "tsne",
        perplexity, workers, checkpoint, resume,
        checkpoint_every)
    if (!identical(massive_checkpoint_file_identity(
        c(x$indices_path, x$values_path)), x$file_identity)) stop(
        "Massive KNN files changed during affinity construction.",
        call. = FALSE)
    graph <- massive_csr_graph(plan$prefix, x$n_vertices, x$access)
    if (graph$n_edges != sorted$built$n_edges) stop(
        "t-SNE affinity edge count changed after construction.",
        call. = FALSE)
    graph$symmetrized <- TRUE
    graph$weight_method <- "tsne_compact_affinity"
    graph$perplexity <- perplexity
    graph$source_k <- x$k
    graph$source_identity <- x$file_identity
    graph$backend <- "cpu"
    graph$n.cores <- workers
    graph$manifest_path <- paste0(plan$prefix, ".affinity.rds")
    manifest <- list(version = 1L, method = graph$weight_method,
        n_vertices = graph$n_vertices, n_edges = graph$n_edges,
        max_degree = graph$max_degree, source_k = graph$source_k,
        perplexity = perplexity,
        hashes = massive_graph_hash(graph$file_identity))
    massive_checkpoint_write(manifest, graph$manifest_path)
    if (!is.null(sorted$sidecar)) unlink(sorted$sidecar)
    graph
}

massive_umap_fuzzy_preflight <- function(x, output, memory_limit,
        manifest_suffix = ".fuzzy.rds", resume = FALSE) {
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
    needed <- 96 * x$n_edges + 8 * (x$n_vertices + 1)
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
    runs <- state$runs
    block <- max(1L, 65536L %/% x$k)
    expected <- if (length(runs$paths)) paste0(plan$prefix,
        ".work/pairs_run_", seq_along(runs$paths) - 1L,
        ".bin") else character()
    current <- tryCatch(massive_checkpoint_file_identity(
        runs$paths), error = function(e) NULL)
    valid <- identical(state$signature, signature) &&
        is.numeric(rows) && length(rows) == 1L &&
        is.finite(rows) && rows >= 0 && rows <= x$n_vertices &&
        rows == floor(rows) &&
        (rows == x$n_vertices || rows %% block == 0) &&
        massive_checkpoint_same_paths(runs$paths, expected) &&
        is.numeric(runs$bytes) &&
        all(is.finite(runs$bytes)) &&
        all(runs$bytes > 0) &&
        sum(runs$bytes) == rows * x$k * 16 &&
        identical(runs, current)
    if (!isTRUE(valid)) stop(
        "Graph sort checkpoint does not match inputs or runs.",
        call. = FALSE)
    state
}

massive_graph_sort_progress <- function(plan, state, k) {
    sidecar <- paste0(plan$prefix, ".sort-checkpoint.rds")
    function(done, paths) {
        old <- state$runs$paths
        if (done < state$completed_rows || length(paths) < length(old) ||
            !massive_checkpoint_same_paths(
                utils::head(paths, length(old)), old)) stop(
            "Graph sort progress is inconsistent.", call. = FALSE)
        fresh <- if (length(paths) > length(old)) {
            paths[seq.int(length(old) + 1L, length(paths))]
        } else character()
        if (length(fresh)) {
            added <- massive_checkpoint_file_identity(fresh)
            for (key in names(state$runs)) {
                state$runs[[key]] <- c(state$runs[[key]],
                    added[[key]])
            }
        }
        if (sum(state$runs$bytes) != done * k * 16) stop(
            "Graph sort run sizes disagree with input progress.",
            call. = FALSE)
        state$completed_rows <- done
        massive_checkpoint_write(state, sidecar)
    }
}

massive_graph_sort_run <- function(x, plan, method, perplexity,
        workers, checkpoint, resume, checkpoint_every) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    every <- integer_scalar(checkpoint_every)
    if (is.na(every) || every < 1L) stop(
        "`checkpoint_every` must be positive.", call. = FALSE)
    state <- if (checkpoint) {
        signature <- massive_graph_sort_signature(x, plan,
            method, perplexity, workers, every)
        massive_graph_sort_state(x, plan, signature, resume)
    } else NULL
    progress <- if (checkpoint) massive_graph_sort_progress(
        plan, state, x$k) else NULL
    built <- massive_symmetrize_graph_cpp(x$indices_path,
        x$values_path, x$n_vertices, x$k, plan$prefix,
        plan$limit, method, perplexity, workers, resume,
        state$completed_rows %||% 0, state$runs$paths %||%
            character(), if (checkpoint) every else 0L,
        progress)
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
    manifest <- if (file.exists(path)) {
        tryCatch(readRDS(path), error = function(e) NULL)
    } else NULL
    if (!is.list(manifest) ||
        !identical(manifest$version, 1L) ||
        !identical(manifest$method, "tsne_compact_affinity") ||
        !identical(manifest$n_vertices, graph$n_vertices) ||
        !identical(manifest$n_edges, graph$n_edges) ||
        !identical(manifest$max_degree, graph$max_degree) ||
        !is.numeric(manifest$source_k) ||
        length(manifest$source_k) != 1L ||
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
    graph$symmetrized <- TRUE
    graph$backend <- "cpu"
    graph$weight_method <- manifest$method
    graph$source_k <- manifest$source_k
    graph$perplexity <- manifest$perplexity
    graph$manifest_path <- path
    graph
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
    identity <- massive_checkpoint_file_identity(paths)
    checked <- massive_validate_csr_graph_cpp(paths[[1L]],
        paths[[2L]], paths[[3L]], n_vertices)
    if (!identical(identity, massive_checkpoint_file_identity(paths))) {
        stop("Massive graph files changed during validation.",
            call. = FALSE)
    }
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
        file_identity = identity, experimental = TRUE)
    class(graph) <- "fastEmbedR_massive_graph"
    graph
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
