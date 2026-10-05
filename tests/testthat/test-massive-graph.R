write_massive_graph_fixture <- function(prefix, indices, distances) {
    writeBin(as.integer(t(indices)), paste0(prefix, ".indices.u32"),
        size = 4L, endian = "little")
    writeBin(as.vector(t(distances)), paste0(prefix, ".distances.f32"),
        size = 4L, endian = "little")
    invisible(prefix)
}

write_massive_matrix_fixture <- function(x) {
    path <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(x)), path, size = 4L, endian = "little")
    path
}

test_that("streamed full exact KNN matches an independent distance matrix", {
    set.seed(7103)
    x <- rbind(matrix(stats::rnorm(48), ncol = 4L),
        c(0, 0, 0, 0), c(0, 0, 0, 0))
    source_path <- write_massive_matrix_fixture(x)
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(c(source_path, paths)))
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    graph <- massive_full_knn_graph(source, k = 3L, output = prefix,
        n.cores = 2L, chunk_rows = 4L, reference_chunk_rows = 5L,
        memory_limit = "256MB")
    expect_s3_class(graph, "fastEmbedR_massive_graph")
    expect_identical(graph$method, "native_cpu_exact_stream")
    expect_identical(graph$backend, "cpu")
    expect_true(graph$exact)
    expect_true(graph$validated)
    observed <- massive_read_graph_edges(graph, 1, nrow(x))
    rounded <- matrix(readBin(source_path, numeric(),
        n = length(x), size = 4L, endian = "little"),
        ncol = ncol(x), byrow = TRUE)
    distances <- as.matrix(stats::dist(rounded))
    diag(distances) <- Inf
    for (row in seq_len(nrow(x))) {
        expected <- order(distances[row, ], seq_len(nrow(x)))[1:3]
        span <- (row - 1L) * 3L + seq_len(3L)
        expect_identical(observed$to[span], as.integer(expected))
        expect_equal(observed$distance[span],
            unname(distances[row, expected]), tolerance = 1e-5)
    }
    fbin_path <- tempfile(fileext = ".fbin")
    fbin_prefix <- tempfile()
    fbin_paths <- paste0(fbin_prefix,
        c(".indices.u32", ".distances.f32"))
    on.exit(unlink(c(fbin_path, fbin_paths)), add = TRUE)
    stream <- file(fbin_path, "wb")
    writeBin(as.integer(dim(x)), stream, size = 4L,
        endian = "little")
    writeBin(as.vector(t(x)), stream, size = 4L, endian = "little")
    close(stream)
    mapped <- massive_matrix(fbin_path, access = "mmap")
    fbin_graph <- massive_full_knn_graph(mapped, 3L, fbin_prefix,
        n.cores = 1L, chunk_rows = 5L, reference_chunk_rows = 6L,
        memory_limit = "256MB")
    expect_equal(massive_read_graph_edges(fbin_graph, 1, nrow(x)),
        observed)
    if (!fastEmbedR:::native_cuda_knn_available_cpp()) {
        expect_error(massive_full_knn_graph(source, 3L, tempfile(),
            backend = "cuda"), "no CPU fallback")
    }
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        chunk_rows = 100000L, memory_limit = "64MB"),
        "memory_limit")
})

test_that("sharded native HNSW merges file-backed global neighbors", {
    set.seed(9231)
    x <- matrix(stats::rnorm(23L * 4L), ncol = 4L)
    source_path <- write_massive_matrix_fixture(x)
    exact_prefix <- tempfile()
    shard_prefix <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(exact_prefix, suffix),
        paste0(shard_prefix, suffix),
        paste0(shard_prefix, suffix, ".part"))))
    source <- massive_matrix(source_path, 23L, 4L)
    exact <- massive_full_knn_graph(source, k = 3L,
        output = exact_prefix, chunk_rows = 5L,
        reference_chunk_rows = 6L, memory_limit = "256MB")
    sharded <- massive_full_knn_graph(source, k = 3L,
        output = shard_prefix, method = "hnsw_sharded",
        chunk_rows = 4L, reference_chunk_rows = 6L,
        memory_limit = "256MB")
    observed <- massive_read_graph_edges(sharded, 1L, 23L)
    reference <- massive_read_graph_edges(exact, 1L, 23L)
    expect_identical(sharded$method, "native_cpu_hnsw_sharded")
    expect_identical(sharded$query_storage, "native_float32")
    expect_false(sharded$exact)
    expect_false(sharded$recall_audited)
    expect_gt(sharded$resources$n_shards, 1L)
    expect_equal(sharded$resources$query_batches,
        ceiling(23 / 4) * sharded$resources$n_shards)
    expect_equal(sharded$resources$minimum_query_read_bytes,
        23 * 4 * 4 * sharded$resources$n_shards)
    expect_gte(sharded$total_seconds, sharded$build_seconds)
    expect_gte(sharded$search_seconds, 0)
    expect_identical(observed$to, reference$to)
    expect_equal(observed$distance, reference$distance,
        tolerance = 1e-5)
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        backend = "cuda", method = "hnsw_sharded"),
        "no CPU fallback")
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        backend = "cpu", method = "ivf_sharded"),
        "no CPU fallback")
    audited_prefix <- tempfile()
    on.exit(unlink(paste0(audited_prefix, suffix)), add = TRUE)
    audited <- massive_full_knn_graph(source, 3L,
        output = audited_prefix, method = "hnsw_sharded",
        chunk_rows = 4L, reference_chunk_rows = 6L,
        memory_limit = "256MB", audit_rows = 7L)
    expect_true(audited$recall_audited)
    expect_identical(audited$audit_rows, 7L)
    expect_equal(audited$observed_recall, 1)
    expect_equal(audited$minimum_row_recall, 1)
    expect_true(audited$sample_target_met)
    expect_identical(length(audited$audit_sample_rows), 7L)
    expect_identical(audited$audit_reference_passes, 2L)
    expect_gte(audited$audit_seconds, 0)
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        method = "hnsw_sharded", audit_rows = 1.5),
        "audit_rows")
})

test_that("shard balancing cannot exceed a reference-row limit", {
    path <- write_massive_matrix_fixture(
        matrix(as.numeric(seq_len(10)), ncol = 2L))
    on.exit(unlink(path))
    source <- massive_matrix(path, 5L, 2L)
    expect_error(fastEmbedR:::massive_hnsw_sharded_resources(
        source, 2L, 2L, 2L, "256MB"),
        "Balanced HNSW shards exceed")
    prefix <- tempfile()
    expect_error(massive_full_knn_graph(source, 2L, prefix,
        method = "hnsw_sharded", chunk_rows = 2L,
        reference_chunk_rows = 2L, memory_limit = "256MB"),
        "Balanced HNSW shards exceed")
    expect_false(file.exists(paste0(prefix, ".indices.u32")))
    if (fastEmbedR:::native_cuda_knn_available_cpp()) {
        expect_error(fastEmbedR:::massive_cuda_sharded_resources(
            source, 2L, 2L, 2L, "256MB"),
            "Balanced CUDA shards exceed")
    }
})

test_that("sharded HNSW queries avoid R double materialization", {
    set.seed(917)
    x <- matrix(rnorm(19L * 4L), ncol = 4L)
    path <- write_massive_matrix_fixture(x)
    prefix <- tempfile()
    on.exit(unlink(c(path, paste0(prefix,
        c(".indices.u32", ".distances.f32")))))
    source <- massive_matrix(path, nrow(x), ncol(x))
    testthat::local_mocked_bindings(
        massive_read_rows_cpp = function(...) {
            stop("R double query path was used.")
        }, .package = "fastEmbedR")
    graph <- massive_full_knn_graph(source, 2L, prefix,
        method = "hnsw_sharded", chunk_rows = 4L,
        reference_chunk_rows = 6L, memory_limit = "256MB")
    expect_identical(graph$query_storage, "native_float32")
    expect_identical(graph$method, "native_cpu_hnsw_sharded")
    edges <- massive_read_graph_edges(graph, 1L, nrow(x))
    expect_length(edges$to, nrow(x) * 2L)
    expect_true(all(is.finite(edges$distance)))
})

test_that("sampled exact KNN shares one reference scan", {
    set.seed(1093)
    values <- matrix(rnorm(37L * 5L), ncol = 5L)
    path <- write_massive_matrix_fixture(values)
    on.exit(unlink(path))
    source <- massive_matrix(path, 37L, 5L)
    ids <- c(2, 11, 29, 36)
    sampled <- fastEmbedR:::massive_exact_sample_batch_cpp(
        source, source, ids, 4L, 7L, 2L, TRUE)
    for (i in seq_along(ids)) {
        single <- fastEmbedR:::massive_exact_graph_batch_cpp(
            source, ids[[i]], 1L, 4L, 7L, 2L)
        expect_identical(sampled$indices[i, ],
            single$indices[1L, ])
        expect_equal(sampled$distances[i, ],
            single$distances[1L, ])
    }
    expect_error(fastEmbedR:::massive_exact_sample_batch_cpp(
        source, source, c(0, 2), 4L, 7L, 1L, TRUE),
        "one-based")
})

test_that("sharded HNSW resumes a partially written first shard", {
    set.seed(1528)
    values <- matrix(stats::rnorm(23L * 4L), ncol = 4L)
    source_path <- write_massive_matrix_fixture(values)
    output <- tempfile()
    expected_path <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    checkpoint_path <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(c(source_path, paste0(output, suffix),
        paste0(output, suffix, ".part"), checkpoint_path,
        paste0(expected_path, suffix))))
    source <- massive_matrix(source_path, 23L, 4L)
    expected <- massive_full_knn_graph(source, 3L, expected_path,
        method = "hnsw_sharded", n.cores = 2L, chunk_rows = 4L,
        reference_chunk_rows = 6L, memory_limit = "256MB")
    resources <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 3L, 4L, 6L, "256MB")
    paths <- fastEmbedR:::massive_knn_paths(output,
        resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    setup <- list(k = 3L, workers = 2L, resources = resources,
        method = "hnsw_sharded", checkpoint_every = 100L)
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cpu", FALSE)
    fastEmbedR:::massive_index_shard(source, setup, parts, 0L,
        fastEmbedR:::massive_checkpoint_source_identity(source),
        "cpu", saved)
    reference <- fastEmbedR:::massive_matrix_rows(source, 1L, 6L)
    index <- fastEmbedR:::massive_knn_reference(reference,
        4L, "hnsw", 2L, "cpu")
    query <- fastEmbedR:::massive_read_rows_cpp(source, 1L,
        4L, 128)
    pilot <- fastEmbedR:::massive_knn_query(index, query,
        4L, "hnsw", 2L, "cpu")
    state <- readRDS(checkpoint_path)
    state$shard <- 0L
    state$completed_rows <- 4
    state$pilot_signature <-
        fastEmbedR:::massive_knn_probe_signature(pilot)
    saveRDS(state, checkpoint_path)
    expect_error(massive_full_knn_graph(source, 3L, output,
        method = "hnsw_sharded", checkpoint = TRUE, resume = TRUE,
        n.cores = 2L, chunk_rows = 5L, reference_chunk_rows = 6L,
        memory_limit = "256MB"), "checkpoint")
    state$pilot_signature <- "wrong"
    saveRDS(state, checkpoint_path)
    expect_error(massive_full_knn_graph(source, 3L, output,
        method = "hnsw_sharded", checkpoint = TRUE, resume = TRUE,
        n.cores = 2L, chunk_rows = 4L, reference_chunk_rows = 6L,
        memory_limit = "256MB"), "pilot")
    expect_equal(file.info(parts)$size, rep(23L * 3L * 4L, 2L))
    state$pilot_signature <-
        fastEmbedR:::massive_knn_probe_signature(pilot)
    saveRDS(state, checkpoint_path)
    observed <- massive_full_knn_graph(source, 3L, output,
        method = "hnsw_sharded", checkpoint = TRUE, resume = TRUE,
        n.cores = 2L, chunk_rows = 4L, reference_chunk_rows = 6L,
        memory_limit = "256MB")
    expect_identical(observed$method, expected$method)
    expect_true(observed$resumed)
    expect_false(file.exists(checkpoint_path))
    expect_equal(massive_read_graph_edges(observed, 1L, 23L),
        massive_read_graph_edges(expected, 1L, 23L))
})

test_that("sharded HNSW replays an uncommitted merge idempotently", {
    set.seed(1528)
    values <- matrix(stats::rnorm(23L * 4L), ncol = 4L)
    source_path <- write_massive_matrix_fixture(values)
    output <- tempfile()
    expected_path <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(output, suffix),
        paste0(output, suffix, ".part"),
        paste0(output, ".checkpoint.rds"),
        paste0(expected_path, suffix))))
    source <- massive_matrix(source_path, 23L, 4L)
    expected <- massive_full_knn_graph(source, 3L, expected_path,
        method = "hnsw_sharded", chunk_rows = 4L,
        reference_chunk_rows = 6L, memory_limit = "256MB")
    resources <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 3L, 4L, 6L, "256MB")
    paths <- fastEmbedR:::massive_knn_paths(output,
        resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    setup <- list(k = 3L, workers = 1L, resources = resources,
        method = "hnsw_sharded", checkpoint_every = 100L)
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cpu", FALSE)
    identity <- fastEmbedR:::massive_checkpoint_source_identity(source)
    first <- fastEmbedR:::massive_index_shard(source, setup,
        parts, 0L, identity, "cpu", saved)
    fastEmbedR:::massive_index_shard(source, setup, parts,
        1L, identity, "cpu", first$saved)
    reference <- fastEmbedR:::massive_matrix_rows(source, 7L, 6L)
    index <- fastEmbedR:::massive_knn_reference(reference,
        4L, "hnsw", 1L, "cpu")
    query <- fastEmbedR:::massive_read_rows_cpp(source, 1L,
        4L, 128)
    pilot <- fastEmbedR:::massive_knn_query(index, query,
        4L, "hnsw", 1L, "cpu")
    state <- readRDS(paste0(output, ".checkpoint.rds"))
    state$shard <- 1L
    state$completed_rows <- 4
    state$pilot_signature <-
        fastEmbedR:::massive_knn_probe_signature(pilot)
    saveRDS(state, paste0(output, ".checkpoint.rds"))
    observed <- massive_full_knn_graph(source, 3L, output,
        method = "hnsw_sharded", checkpoint = TRUE, resume = TRUE,
        chunk_rows = 4L, reference_chunk_rows = 6L,
        memory_limit = "256MB")
    expect_equal(massive_read_graph_edges(observed, 1L, 23L),
        massive_read_graph_edges(expected, 1L, 23L))
})

test_that("sharded checkpoint interval replays pending query blocks", {
    set.seed(1529)
    values <- matrix(stats::rnorm(23L * 4L), ncol = 4L)
    source_path <- write_massive_matrix_fixture(values)
    output <- tempfile()
    expected_path <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(output, suffix),
        paste0(output, suffix, ".part"),
        paste0(output, ".checkpoint.rds"),
        paste0(expected_path, suffix))))
    source <- massive_matrix(source_path, 23L, 4L)
    expected <- massive_full_knn_graph(source, 3L, expected_path,
        method = "hnsw_sharded", chunk_rows = 4L,
        reference_chunk_rows = 6L, memory_limit = "256MB")
    resources <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 3L, 4L, 6L, "256MB")
    paths <- fastEmbedR:::massive_knn_paths(output,
        resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    setup <- list(k = 3L, workers = 1L, resources = resources,
        method = "hnsw_sharded", checkpoint_every = 2L)
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cpu", FALSE)
    reference <- fastEmbedR:::massive_matrix_rows(source, 1L, 6L)
    index <- fastEmbedR:::massive_knn_reference(reference,
        4L, "hnsw", 1L, "cpu")
    first <- fastEmbedR:::massive_sharded_batch(source, setup,
        parts, index, 0L, 1L, 6L, 1L, "hnsw", "cpu", saved)
    expect_equal(readRDS(saved$path)$completed_rows, 0)
    fastEmbedR:::massive_sharded_batch(source, setup,
        parts, index, 0L, 1L, 6L, 5L, "hnsw", "cpu", first$saved)
    expect_equal(readRDS(saved$path)$completed_rows, 8)
    observed <- massive_full_knn_graph(source, 3L, output,
        method = "hnsw_sharded", chunk_rows = 4L,
        reference_chunk_rows = 6L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE, checkpoint_every = 2L)
    expect_true(observed$resumed)
    expect_identical(observed$checkpoint_every, 2L)
    expect_equal(massive_read_graph_edges(observed, 1L, 23L),
        massive_read_graph_edges(expected, 1L, 23L))
})

test_that("sharded CUDA exact KNN matches the streamed CPU graph", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(1603)
    x <- matrix(stats::rnorm(41L * 6L), ncol = 6L)
    source_path <- write_massive_matrix_fixture(x)
    cpu_prefix <- tempfile()
    gpu_prefix <- tempfile()
    resumed_prefix <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(cpu_prefix, suffix),
        paste0(gpu_prefix, suffix),
        paste0(gpu_prefix, suffix, ".part"),
        paste0(resumed_prefix, suffix),
        paste0(resumed_prefix, suffix, ".part"),
        paste0(resumed_prefix, ".checkpoint.rds"))))
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    cpu <- massive_full_knn_graph(source, 5L, cpu_prefix,
        chunk_rows = 7L, reference_chunk_rows = 13L,
        memory_limit = "512MB")
    gpu <- massive_full_knn_graph(source, 5L, gpu_prefix,
        backend = "cuda", chunk_rows = 7L,
        reference_chunk_rows = 13L, memory_limit = "512MB")
    expect_identical(gpu$backend, "cuda")
    expect_identical(gpu$method,
        "native_cuda_cuvs_exact_sharded")
    expect_identical(gpu$query_storage, "native_float32")
    expect_true(gpu$exact)
    expect_gt(gpu$resources$n_shards, 1L)
    expect_gt(gpu$resources$peak_vram_bytes, 0)
    observed <- massive_read_graph_edges(gpu, 1L, nrow(x))
    reference <- massive_read_graph_edges(cpu, 1L, nrow(x))
    expect_identical(observed$to, reference$to)
    expect_equal(observed$distance, reference$distance,
        tolerance = 1e-4)
    resources <- fastEmbedR:::massive_cuda_sharded_resources(
        source, 5L, 7L, 13L, "512MB")
    paths <- fastEmbedR:::massive_knn_paths(resumed_prefix,
        resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    setup <- list(k = 5L, workers = 1L, resources = resources,
        method = "exact", checkpoint_every = 2L)
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cuda", FALSE)
    fastEmbedR:::massive_index_shard(source, setup, parts, 0L,
        fastEmbedR:::massive_checkpoint_source_identity(source),
        "cuda", saved)
    resumed <- massive_full_knn_graph(source, 5L, resumed_prefix,
        backend = "cuda", chunk_rows = 7L,
        reference_chunk_rows = 13L, memory_limit = "512MB",
        checkpoint = TRUE, resume = TRUE, checkpoint_every = 2L)
    expect_identical(resumed$backend, "cuda")
    expect_identical(resumed$checkpoint_every, 2L)
    expect_true(resumed$resumed)
    expect_equal(massive_read_graph_edges(resumed, 1L, nrow(x)),
        observed, tolerance = 1e-4)
})

test_that("sharded CUDA IVF reports approximate full-graph recall", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(1604)
    x <- matrix(stats::rnorm(120L * 6L), ncol = 6L)
    source_path <- write_massive_matrix_fixture(x)
    prefix <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(prefix, suffix),
        paste0(prefix, suffix, ".part"))))
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    resources <- fastEmbedR:::massive_cuda_sharded_resources(
        source, 5L, NULL, 70L, "512MB", method = "ivf_sharded")
    expect_equal(resources$chunk_rows, 8192L)
    expect_equal(resources$query_batches, resources$n_shards)
    expect_equal(resources$minimum_query_read_bytes,
        120 * 6 * 4 * resources$n_shards)
    graph <- massive_full_knn_graph(source, 5L, prefix,
        backend = "cuda", method = "ivf_sharded",
        chunk_rows = 10L, reference_chunk_rows = 70L,
        memory_limit = "512MB", audit_rows = 12L)
    expect_identical(graph$backend, "cuda")
    expect_identical(graph$method, "native_cuda_cuvs_ivf_sharded")
    expect_false(graph$exact)
    expect_true(graph$recall_audited)
    expect_identical(graph$audit_rows, 12L)
    expect_equal(graph$target_recall, 0.99)
    expect_gt(graph$resources$n_shards, 1L)
    expect_true(all(graph$audit_row_recall >= 0))
    expect_true(all(graph$audit_row_recall <= 1))
    edges <- massive_read_graph_edges(graph, 1L, nrow(x))
    expect_true(all(is.finite(edges$distance)))
    expect_false(any(edges$from == edges$to))
})

test_that("billion-row sharded plans keep large counts numeric", {
    source <- list(format = "file", nrow = 1e9, ncol = 96L)
    plan <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, 8192L, 250000L, "8GB")
    expect_equal(plan$n_shards, 4000L)
    expect_equal(plan$output_bytes, 1e9 * 30 * 8)
    expect_equal(plan$query_batches,
        ceiling(1e9 / plan$chunk_rows) * 4000)
    expect_equal(plan$minimum_query_read_bytes,
        1e9 * 96 * 4 * 4000)
    expect_gt(plan$minimum_query_read_bytes, 2^32)
    tasks <- fastEmbedR:::massive_knn_device_tasks(source, 30L,
        tempfile(), c(0L, 1L, 2L))
    ranges <- do.call(rbind, lapply(tasks, `[[`, "row_range"))
    expect_equal(ranges[1L, 1L], 1)
    expect_equal(ranges[3L, 2L], 1e9)
    expect_equal(ranges[-1L, 1L], ranges[-3L, 2L] + 1)
    expect_equal(sum(vapply(tasks, `[[`, 0, "bytes")),
        1e9 * 30 * 4)
})

test_that("full exact KNN resumes a committed file-backed query block", {
    x <- matrix(seq_len(80) / 17, ncol = 5L)
    source_path <- write_massive_matrix_fixture(x)
    prefix <- file.path(normalizePath(tempdir()), basename(tempfile()))
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    parts <- paste0(paths, ".part")
    on.exit(unlink(c(source_path, paths, parts,
        paste0(prefix, ".checkpoint.rds"))))
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    resources <- fastEmbedR:::massive_full_knn_resources(
        source, 2L, 4L, 5L, "256MB")
    setup <- list(output = prefix,
        paths = c(indices = paths[[1L]], distances = paths[[2L]]),
        k = 2L, workers = 1L, resources = resources,
        checkpoint_every = 100L, method = "exact",
        backend_used = "native_cpu_exact_stream", exact = TRUE)
    saved <- fastEmbedR:::massive_full_knn_checkpoint(source,
        setup, FALSE)
    first <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 1, 4L, 2L, 5L, 1L)
    streams <- fastEmbedR:::massive_knn_open_parts(
        c(indices = parts[[1L]], distances = parts[[2L]]), 0, 2L)
    fastEmbedR:::massive_full_knn_commit(first, streams, 4,
        saved, setup)
    close(streams$indices)
    close(streams$distances)
    graph <- massive_full_knn_graph(source, 2L, prefix,
        n.cores = 1L, chunk_rows = 4L, reference_chunk_rows = 5L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE)
    expect_equal(graph$n_edges, nrow(x) * 2L)
    expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
    expect_equal(massive_read_graph_edges(graph, 1, 4)$to,
        as.integer(t(first$indices)))
})

test_that("exact checkpoint interval replays uncommitted query blocks", {
    values <- matrix(seq_len(80) / 17, ncol = 5L)
    source_path <- write_massive_matrix_fixture(values)
    output <- file.path(normalizePath(tempdir()),
        basename(tempfile()))
    expected_path <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32")
    on.exit(unlink(c(source_path, paste0(output, suffix),
        paste0(output, suffix, ".part"),
        paste0(output, ".checkpoint.rds"),
        paste0(expected_path, suffix))))
    source <- massive_matrix(source_path, nrow(values), ncol(values))
    expected <- massive_full_knn_graph(source, 2L, expected_path,
        chunk_rows = 4L, reference_chunk_rows = 5L,
        memory_limit = "256MB")
    resources <- fastEmbedR:::massive_full_knn_resources(
        source, 2L, 4L, 5L, "256MB")
    paths <- fastEmbedR:::massive_knn_paths(output,
        resources$output_bytes)
    parts <- stats::setNames(paste0(unname(paths), ".part"),
        names(paths))
    setup <- list(output = output, paths = paths, k = 2L,
        workers = 1L, resources = resources,
        checkpoint_every = 2L, method = "exact",
        backend_used = "native_cpu_exact_stream", exact = TRUE)
    saved <- fastEmbedR:::massive_full_knn_checkpoint(source,
        setup, FALSE)
    streams <- fastEmbedR:::massive_knn_open_parts(parts, 0, 2L)
    first <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 1L, 4L, 2L, 5L, 1L)
    fastEmbedR:::massive_full_knn_commit(first, streams, 4,
        saved, setup, save_now = FALSE)
    expect_equal(readRDS(saved$path)$completed_rows, 0)
    second <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 5L, 4L, 2L, 5L, 1L)
    fastEmbedR:::massive_full_knn_commit(second, streams, 8,
        saved, setup, save_now = TRUE)
    expect_equal(readRDS(saved$path)$completed_rows, 8)
    close(streams$indices)
    close(streams$distances)
    observed <- massive_full_knn_graph(source, 2L, output,
        chunk_rows = 4L, reference_chunk_rows = 5L,
        memory_limit = "256MB", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L)
    expect_identical(observed$checkpoint_every, 2L)
    expect_equal(massive_read_graph_edges(observed, 1L, 16L),
        massive_read_graph_edges(expected, 1L, 16L))
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        checkpoint_every = 0L), "checkpoint_every")
    expect_error(massive_full_knn_graph(source, 2L, tempfile(),
        checkpoint_every = 1.5), "checkpoint_every")
})

test_that("full graph import streams existing files without copying", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(paths))
    indices <- outer(1:7, 1:3, function(i, j) (i + j - 1) %% 7 + 1)
    distances <- matrix(rep(c(0.5, 1, 1.5), 7), nrow = 7,
        byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    streamed <- massive_knn_graph(prefix, 7, 3, access = "stream")
    mapped <- massive_knn_graph(prefix, 7, 3, access = "mmap")
    expect_s3_class(streamed, "fastEmbedR_massive_graph")
    expect_identical(streamed$n_edges, 21)
    expect_true(streamed$directed)
    expect_false(streamed$symmetrized)
    expect_identical(streamed$edge_value, "distance")
    expect_identical(streamed$distance_dtype, "float32")
    expect_identical(streamed$backend, "external")
    expect_identical(mapped$validation_access, "stream")
    expect_identical(streamed$indices_path,
        normalizePath(paths[[1L]], mustWork = TRUE))
    expected <- list(from = rep(3:5, each = 3),
        to = as.integer(t(indices[3:5, ])),
        distance = as.numeric(t(distances[3:5, ])))
    expect_equal(massive_read_graph_edges(streamed, 3, 3), expected)
    expect_equal(massive_read_graph_edges(mapped, 3, 3), expected)
    expect_equal(head(streamed, 1)$to, as.integer(indices[1, ]))
    expect_error(massive_read_graph_edges(streamed, 7, 2), "range")
    expect_error(massive_read_graph_edges(streamed, 1, 0), "counts")
    writeBin(as.raw(0), paths[[1L]], useBytes = TRUE)
    expect_error(massive_read_graph_edges(streamed, 1, 1), "changed")
})

test_that("full graph import rejects corrupt and self-neighbor edges", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(paths))
    indices <- matrix(c(2L, 3L, 1L, 3L, 1L, 2L),
        nrow = 3L, byrow = TRUE)
    distances <- matrix(1, nrow = 3L, ncol = 2L)
    write_massive_graph_fixture(prefix, indices, distances)
    expect_true(massive_knn_graph(prefix, 3L, 2L)$validated)
    invalid <- indices
    invalid[2L, 1L] <- 2L
    write_massive_graph_fixture(prefix, invalid, distances)
    expect_error(massive_knn_graph(prefix, 3L, 2L), "self edges")
    invalid <- indices
    invalid[1L, 2L] <- 2L
    write_massive_graph_fixture(prefix, invalid, distances)
    expect_error(massive_knn_graph(prefix, 3L, 2L), "duplicate")
    invalid[1L, 2L] <- 4L
    write_massive_graph_fixture(prefix, invalid, distances)
    expect_error(massive_knn_graph(prefix, 3L, 2L), "invalid")
    distances[1L, 1L] <- NaN
    write_massive_graph_fixture(prefix, indices, distances)
    expect_error(massive_knn_graph(prefix, 3L, 2L), "invalid")
    expect_error(massive_knn_graph(prefix, 3L, 2.5), "k")
    expect_error(massive_knn_graph(prefix, 2^31, 2), "n_vertices")
    writeBin(1L, paths[[1L]], size = 4L)
    expect_error(massive_knn_graph(prefix, 3L, 2L), "size")
})

test_that("full graph offsets remain exact beyond signed 32-bit", {
    storage <- fastEmbedR:::massive_graph_storage_cpp(1e9, 30L)
    expect_identical(storage$n_edges, 30e9)
    expect_identical(storage$file_bytes, 120e9)
    expect_identical(storage$last_row_offset_bytes,
        (1e9 - 1) * 30 * 4)
    expect_gt(storage$last_row_offset_bytes, 2^31)
})

test_that("full graph partitions read bounded complete rows", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(paths))
    indices <- outer(1:7, 1:3, function(i, j) (i + j - 1) %% 7 + 1)
    distances <- matrix(rep(c(0.5, 1, 1.5), 7), nrow = 7,
        byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    graph <- massive_knn_graph(prefix, 7, 3)
    plan <- massive_graph_partitions(graph, max_bytes = 120)
    expect_identical(plan$rows_per_partition, 2)
    expect_identical(plan$n_partitions, 4)
    for (part in 1:4) {
        first <- (part - 1) * 2 + 1
        count <- min(2, 7 - first + 1)
        expect_equal(massive_read_graph_partition(plan, part),
            massive_read_graph_edges(graph, first, count))
    }
    expect_error(massive_read_graph_partition(plan, 5), "range")
    expect_error(massive_graph_partitions(graph, 59), "one row")
    expect_error(massive_graph_partitions(graph, 129e6), "128 MB")
})

test_that("billion-row partition plans remain constant-size", {
    graph <- structure(list(n_vertices = 1e9, k = 30L),
        class = "fastEmbedR_massive_graph")
    plan <- massive_graph_partitions(graph)
    expect_lt(as.numeric(object.size(plan)), 2048)
    expect_identical(plan$rows_per_partition, floor(64e6 / 600))
    expect_identical(plan$n_partitions,
        ceiling(1e9 / plan$rows_per_partition))
    last <- (plan$n_partitions - 1) * plan$rows_per_partition + 1
    expect_lte(last, 1e9)
    expect_gt(last, 1e9 - plan$rows_per_partition)
})

test_that("weighted graph import reuses the bounded graph reader", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".weights.f32"))
    on.exit(unlink(paths))
    indices <- matrix(c(2L, 3L, 3L, 4L, 4L, 1L, 1L, 2L),
        nrow = 4L, byrow = TRUE)
    weights <- matrix(c(0, 0.25, 0.5, 0.75, 1, 0.25, 0.5, 0.75),
        nrow = 4L, byrow = TRUE)
    writeBin(as.integer(t(indices)), paths[[1L]], size = 4L,
        endian = "little")
    writeBin(as.vector(t(weights)), paths[[2L]], size = 4L,
        endian = "little")
    graph <- massive_weighted_graph(prefix, 4, 2, access = "mmap")
    expect_identical(graph$edge_value, "weight")
    expect_identical(graph$weight_dtype, "float32")
    expect_false(graph$symmetrized)
    expect_identical(graph$values_path, graph$weights_path)
    expected <- list(from = rep(1:4, each = 2),
        to = as.integer(t(indices)),
        weight = as.numeric(t(weights)))
    expect_equal(massive_read_graph_edges(graph, 1, 4), expected)
    plan <- massive_graph_partitions(graph, max_bytes = 80)
    expect_equal(massive_read_graph_partition(plan, 1),
        list(from = expected$from[1:4], to = expected$to[1:4],
            weight = expected$weight[1:4]))
    weights[1, 1] <- 1.25
    writeBin(as.vector(t(weights)), paths[[2L]], size = 4L,
        endian = "little")
    expect_error(massive_weighted_graph(prefix, 4, 2),
        "invalid edge values")
    weights[1, 1] <- -0.25
    writeBin(as.vector(t(weights)), paths[[2L]], size = 4L,
        endian = "little")
    expect_error(massive_weighted_graph(prefix, 4, 2),
        "invalid edge values")
})

test_that("directed UMAP memberships stream without changing KNN IDs", {
    prefix <- tempfile()
    parallel_prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32",
        ".weights.f32", ".weights.f32.part"))
    parallel_paths <- paste0(parallel_prefix,
        c(".indices.u32", ".distances.f32", ".weights.f32"))
    on.exit(unlink(c(paths, parallel_paths)))
    indices <- outer(1:8, 1:3,
        function(i, j) (i + j - 1L) %% 8L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), 8L),
        nrow = 8L, byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    write_massive_graph_fixture(parallel_prefix, indices, distances)
    knn <- massive_knn_graph(prefix, 8L, 3L, access = "mmap")
    weighted <- massive_umap_memberships(knn)
    parallel <- massive_umap_memberships(
        massive_knn_graph(parallel_prefix, 8L, 3L), n.cores = 4L)
    expect_identical(parallel$n.cores, 4L)
    expect_identical(readBin(paths[[3L]], "raw", n = 96L),
        readBin(parallel_paths[[3L]], "raw", n = 96L))
    expect_identical(weighted$weight_method,
        "umap_directed_membership")
    expect_identical(weighted$backend, "cpu")
    expect_false(weighted$symmetrized)
    expect_identical(weighted$indices_path, knn$indices_path)
    expect_false(file.exists(paths[[4L]]))
    observed <- massive_read_graph_edges(weighted, 1, 8L)
    expect_identical(observed$to, as.integer(t(indices)))
    rho <- distances[1L, 1L]
    target <- log2(3)
    objective <- function(sigma) {
        sum(exp(-pmax(distances[1L, ] - rho, 0) / sigma)) -
            target
    }
    sigma <- stats::uniroot(objective, c(1e-5, 10))$root
    expected <- exp(-pmax(distances[1L, ] - rho, 0) / sigma)
    expect_equal(observed$weight[1:3], expected, tolerance = 1e-4)
    expect_equal(observed$weight[seq(1L, 24L, by = 3L)],
        rep(1, 8L))
    native <- fastEmbedR:::umap_graph_csr_cpp(
        indices, distances, 0L, 3L, 3L, 1L)
    for (row in seq_len(8L)) {
        positions <- seq.int(native$offsets[row] + 1L,
            native$offsets[row + 1L])
        for (position in positions) {
            neighbor <- native$neighbors[position] + 1L
            forward <- observed$weight[observed$from == row &
                observed$to == neighbor]
            backward <- observed$weight[observed$from == neighbor &
                observed$to == row]
            a <- if (length(forward)) forward else 0
            b <- if (length(backward)) backward else 0
            expect_equal(native$weights[position], a + b - a * b,
                tolerance = 1e-5)
        }
    }
    expect_error(massive_umap_memberships(knn), "already exists")
    expect_error(massive_umap_memberships(weighted),
        "distance-valued")
    expect_error(massive_umap_memberships(knn, backend = "cuda"),
        "no backend fallback")
    expect_error(massive_umap_memberships(knn, n.cores = 0L),
        "n.cores")
    expect_error(massive_umap_memberships(knn, n.cores = 1.5),
        "n.cores")
})

test_that("directed memberships resume a committed output block", {
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32",
        ".weights.f32", ".weights.f32.part",
        ".weights.f32.checkpoint.rds"))
    on.exit(unlink(paths))
    rows <- 22000L
    indices <- outer(seq_len(rows), 1:3,
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), rows),
        nrow = rows, byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    graph <- massive_knn_graph(prefix, rows, 3L)
    output <- paste0(sub("\\.indices\\.u32$", "",
        graph$indices_path), ".weights.f32")
    prepared <- fastEmbedR:::massive_umap_membership_prepare(
        graph, output, TRUE, FALSE, 1L)
    interrupt <- function(done) {
        prepared$progress(done)
        stop("deliberate interruption")
    }
    expect_error(fastEmbedR:::massive_umap_memberships_cpp(
        graph$indices_path, graph$values_path, rows, 3L,
        output, prepared$mean, 0, interrupt, FALSE, 1L),
        "deliberate interruption")
    committed <- 65536L %/% 3L
    expect_identical(readRDS(paths[[5L]])$completed_rows,
        as.numeric(committed))
    expect_equal(file.info(paths[[4L]])$size,
        committed * 3L * 4L)
    expect_error(massive_umap_memberships(graph, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 2L), "does not match")
    weighted <- massive_umap_memberships(graph, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 1L, n.cores = 4L)
    expect_identical(weighted$n.cores, 4L)
    expect_false(file.exists(paths[[4L]]))
    expect_false(file.exists(paths[[5L]]))
    expect_equal(file.info(paths[[3L]])$size, rows * 3L * 4L)
    boundary <- massive_read_graph_edges(weighted, committed, 2L)
    expect_identical(boundary$to,
        as.integer(t(indices[committed:(committed + 1L), ])))
    expect_equal(boundary$weight[1:3],
        boundary$weight[4:6], tolerance = 1e-7)
})

test_that("directed memberships resume the global-mean scan", {
    prefix <- tempfile()
    baseline_prefix <- tempfile()
    suffix <- c(".indices.u32", ".distances.f32", ".weights.f32",
        ".weights.f32.part", ".weights.f32.checkpoint.rds")
    on.exit(unlink(c(paste0(prefix, suffix),
        paste0(baseline_prefix, suffix))))
    rows <- 22000L
    indices <- outer(seq_len(rows), 1:3,
        function(i, j) (i + j - 1L) %% rows + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), rows),
        nrow = rows, byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    write_massive_graph_fixture(baseline_prefix, indices, distances)
    graph <- massive_knn_graph(prefix, rows, 3L)
    native <- fastEmbedR:::massive_umap_global_mean_resume_cpp
    interrupted <- function(indices_path, distances_path,
            n_vertices, k, completed_rows, sum, progress,
            checkpoint_every) {
        stop_after_block <- function(done, partial_sum) {
            progress(done, partial_sum)
            stop("deliberate mean interruption")
        }
        native(indices_path, distances_path, n_vertices, k,
            completed_rows, sum, stop_after_block, checkpoint_every)
    }
    expect_error(with_mocked_bindings(
        massive_umap_memberships(graph, checkpoint = TRUE,
            checkpoint_every = 1L),
        massive_umap_global_mean_resume_cpp = interrupted),
        "deliberate mean interruption")
    sidecar <- paste0(prefix, ".weights.f32.checkpoint.rds")
    state <- readRDS(sidecar)
    expect_identical(state$stage, "mean")
    expect_identical(state$mean_rows, as.numeric(65536L %/% 3L))
    expect_false(file.exists(paste0(prefix, ".weights.f32.part")))
    tampered <- state
    tampered$mean_sum <- "1"
    saveRDS(tampered, sidecar)
    expect_error(massive_umap_memberships(graph, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 1L), "does not match")
    saveRDS(state, sidecar)
    resumed_from <- new.env(parent = emptyenv())
    record_resume <- function(indices_path, distances_path,
            n_vertices, k, completed_rows, sum, progress,
            checkpoint_every) {
        resumed_from$rows <- completed_rows
        native(indices_path, distances_path, n_vertices, k,
            completed_rows, sum, progress, checkpoint_every)
    }
    resumed <- with_mocked_bindings(
        massive_umap_memberships(graph, checkpoint = TRUE,
            resume = TRUE, checkpoint_every = 1L),
        massive_umap_global_mean_resume_cpp = record_resume)
    expect_identical(resumed_from$rows,
        as.numeric(65536L %/% 3L))
    baseline <- massive_umap_memberships(massive_knn_graph(
        baseline_prefix, rows, 3L))
    expect_identical(unname(tools::md5sum(resumed$values_path)),
        unname(tools::md5sum(baseline$values_path)))
    expect_false(file.exists(sidecar))
})

test_that("disk-backed fuzzy union agrees with native UMAP CSR", {
    prefix <- tempfile()
    output <- tempfile()
    source_paths <- paste0(prefix,
        c(".indices.u32", ".distances.f32", ".weights.f32"))
    result_paths <- paste0(output,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    on.exit(unlink(c(source_paths, result_paths,
        paste0(result_paths, ".part"),
        paste0(output, c(".work", ".fuzzy.rds"))),
        recursive = TRUE))
    indices <- outer(1:8, 1:3,
        function(i, j) (i + j - 1L) %% 8L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), 8L),
        nrow = 8L, byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    knn <- massive_knn_graph(prefix, 8L, 3L)
    directed <- massive_umap_memberships(knn)
    fuzzy <- massive_umap_fuzzy_graph(directed, output,
        memory_limit = "64MB")
    expect_true(fuzzy$symmetrized)
    expect_identical(fuzzy$backend, "cpu")
    expect_identical(fuzzy$weight_method, "umap_fuzzy_union")
    reopened <- massive_fuzzy_graph(output, 8L)
    expect_true(reopened$symmetrized)
    expect_identical(reopened$weight_method, fuzzy$weight_method)
    expect_identical(reopened$n_edges, fuzzy$n_edges)
    native <- fastEmbedR:::umap_graph_csr_cpp(
        indices, distances, 0L, 3L, 3L, 1L)
    expect_equal(fuzzy$n_edges, native$nnz)
    for (row in seq_len(8L)) {
        positions <- seq.int(native$offsets[row] + 1L,
            native$offsets[row + 1L])
        observed <- massive_read_graph_edges(reopened, row, 1L)
        expect_identical(observed$to,
            native$neighbors[positions] + 1L)
        expect_equal(observed$weight, native$weights[positions],
            tolerance = 1e-5)
    }
    expect_error(massive_umap_fuzzy_graph(directed, output),
        "already exist")
    expect_error(massive_umap_fuzzy_graph(directed, tempfile(),
        backend = "cuda"), "no backend fallback")
    expect_error(massive_umap_fuzzy_graph(knn, tempfile()),
        "directed UMAP memberships")
    sidecar_only <- tempfile()
    on.exit(unlink(paste0(sidecar_only, ".fuzzy.rds")), add = TRUE)
    saveRDS(list(), paste0(sidecar_only, ".fuzzy.rds"))
    expect_error(massive_umap_fuzzy_graph(directed, sidecar_only),
        "already exist")
    weight_path <- paste0(output, ".weights.f32")
    stamp <- file.info(weight_path)$mtime
    con <- file(weight_path, "r+b")
    writeBin(0.1234567, con, size = 4L, endian = "little")
    close(con)
    Sys.setFileTime(weight_path, stamp)
    expect_error(massive_fuzzy_graph(output, 8L),
        "content differs")
    unlink(paste0(output, ".fuzzy.rds"))
    expect_error(massive_fuzzy_graph(output, 8L), "manifest")
    expect_false(massive_csr_graph(output, 8L)$symmetrized)
})

test_that("fuzzy union accepts a hub larger than the input KNN", {
    n <- 65538L
    source <- tempfile()
    output <- tempfile()
    fitted_path <- tempfile(fileext = ".f32")
    source_paths <- paste0(source,
        c(".indices.u32", ".distances.f32", ".weights.f32"))
    result_paths <- paste0(output,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    on.exit(unlink(c(source_paths, result_paths, fitted_path,
        paste0(output, c(".work", ".fuzzy.rds"))),
        recursive = TRUE))
    neighbors <- matrix(c(2L, rep.int(1L, n - 1L)), ncol = 1L)
    distances <- matrix(rep.int(1, n), ncol = 1L)
    write_massive_graph_fixture(source, neighbors, distances)
    knn <- massive_knn_graph(source, n, 1L)
    directed <- massive_umap_memberships(knn)
    fuzzy <- massive_umap_fuzzy_graph(directed, output,
        memory_limit = "64MB")
    expect_equal(fuzzy$max_degree, n - 1L)
    expect_equal(fuzzy$n_edges, 2 * (n - 1L))
    expect_identical(massive_read_graph_edges(fuzzy, 1L, 1L)$to,
        seq.int(2L, n))
    initial <- matrix(c(seq_len(n) %% 997L,
        seq_len(n) %% 991L) / 1000, ncol = 2L)
    initial_path <- write_massive_matrix_fixture(initial)
    on.exit(unlink(initial_path), add = TRUE)
    fitted <- massive_umap_optimize(fuzzy,
        massive_matrix(initial_path, n, 2L), fitted_path,
        n_epochs = 2L, negative_sample_rate = 1L,
        chunk_rows = 64L, memory_limit = "256MB")
    expect_gt(fitted$updates$positive_updates, 0)
    expect_true(all(is.finite(massive_read_rows(fitted, n, 1L))))
    louvain_path <- tempfile(fileext = ".clusters.u32")
    leiden_path <- tempfile(fileext = ".clusters.u32")
    on.exit(unlink(c(louvain_path, leiden_path)), add = TRUE)
    if (.Platform$OS.type == "windows") {
        expect_error(massive_louvain_level(fuzzy, louvain_path,
            max_passes = 1L, chunk_rows = 64L,
            memory_limit = "256MB"), "POSIX memory mapping")
        return(invisible(NULL))
    }
    louvain <- massive_louvain_level(fuzzy, louvain_path,
        max_passes = 1L, chunk_rows = 64L,
        memory_limit = "256MB")
    expect_equal(louvain$n_vertices, n)
    expect_true(all(massive_read_louvain_rows(louvain, n, 1L) > 0L))
    leiden <- massive_leiden_refine_level(fuzzy, louvain,
        leiden_path, chunk_rows = 64L, memory_limit = "256MB")
    on.exit(unlink(leiden$parent_mapping_path), add = TRUE)
    expect_equal(leiden$n_vertices, n)
    expect_true(all(fastEmbedR:::massive_read_label_rows(
        leiden, n, 1L) > 0L))
})

test_that("graph sorting resumes committed pair runs", {
    n <- 45000L
    source <- tempfile()
    output <- tempfile()
    baseline <- tempfile()
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32", ".weights.f32")),
        paste0(output, c(".offsets.u64", ".indices.u32",
            ".weights.f32", ".fuzzy.rds",
            ".sort-checkpoint.rds", ".work")),
        paste0(baseline, c(".offsets.u64", ".indices.u32",
            ".weights.f32", ".fuzzy.rds", ".work"))),
        recursive = TRUE))
    neighbors <- outer(seq_len(n), 1:3, function(i, j)
        (i + j - 1L) %% n + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), n),
        nrow = n, byrow = TRUE)
    write_massive_graph_fixture(source, neighbors, distances)
    directed <- massive_umap_memberships(
        massive_knn_graph(source, n, 3L))
    plan <- fastEmbedR:::massive_umap_fuzzy_preflight(
        directed, output, "64MB")
    signature <- fastEmbedR:::massive_graph_sort_signature(
        directed, plan, "umap", 0, 1L, 1L)
    state <- fastEmbedR:::massive_graph_sort_state(
        directed, plan, signature, FALSE)
    progress <- fastEmbedR:::massive_graph_sort_progress(
        plan, state, 3L)
    interrupted <- function(done, paths) {
        progress(done, paths)
        stop("deliberate graph-sort interruption")
    }
    expect_error(fastEmbedR:::massive_symmetrize_graph_cpp(
        directed$indices_path, directed$values_path, n, 3L,
        output, plan$limit, "umap", 0, 1L, FALSE, 0,
        character(), 1L, interrupted), "deliberate")
    checkpoint <- readRDS(paste0(output,
        ".sort-checkpoint.rds"))
    expect_equal(checkpoint$completed_rows, 65536L %/% 3L)
    expect_length(checkpoint$runs$paths, 1L)
    sidecar <- paste0(output, ".sort-checkpoint.rds")
    altered <- checkpoint
    altered$runs$bytes[1L] <- altered$runs$bytes[1L] - 16
    saveRDS(altered, sidecar)
    expect_error(massive_umap_fuzzy_graph(directed, output,
        "64MB", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 1L), "checkpoint")
    saveRDS(checkpoint, sidecar)
    expect_error(massive_umap_fuzzy_graph(directed, output,
        "64MB", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L), "checkpoint")
    resumed <- massive_umap_fuzzy_graph(directed, output,
        "64MB", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 1L)
    fresh <- massive_umap_fuzzy_graph(directed, baseline,
        "64MB")
    expect_equal(resumed$n_edges, fresh$n_edges)
    suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
    expect_identical(unname(tools::md5sum(paste0(output, suffix))),
        unname(tools::md5sum(paste0(baseline, suffix))))
    expect_false(file.exists(paste0(output,
        ".sort-checkpoint.rds")))
})

test_that("graph sorting resumes a checkpoint without a work directory", {
    source <- tempfile()
    output <- tempfile()
    baseline <- tempfile()
    suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32")), paste0(output, c(suffix,
        ".fuzzy.rds", ".sort-checkpoint.rds", ".work")),
        paste0(baseline, c(suffix, ".fuzzy.rds", ".work"))),
        recursive = TRUE))
    indices <- outer(1:8, 1:3, function(i, j)
        (i + j - 1L) %% 8L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), 8L),
        nrow = 8L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    directed <- massive_umap_memberships(
        massive_knn_graph(source, 8L, 3L))
    on.exit(unlink(directed$values_path), add = TRUE)
    plan <- fastEmbedR:::massive_umap_fuzzy_preflight(
        directed, output, "64MB")
    signature <- fastEmbedR:::massive_graph_sort_signature(
        directed, plan, "umap", 0, 1L, 1L)
    fastEmbedR:::massive_graph_sort_state(
        directed, plan, signature, FALSE)
    expect_false(dir.exists(paste0(output, ".work")))
    resumed <- massive_umap_fuzzy_graph(directed, output,
        "64MB", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 1L)
    fresh <- massive_umap_fuzzy_graph(directed, baseline, "64MB")
    expect_equal(resumed$n_edges, fresh$n_edges)
    expect_identical(unname(tools::md5sum(paste0(output, suffix))),
        unname(tools::md5sum(paste0(baseline, suffix))))
})

test_that("disk-backed modularity matches the resident native score", {
    skip_on_os("windows")
    prefix <- tempfile()
    output <- tempfile()
    label_path <- tempfile(fileext = ".u32")
    source_paths <- paste0(prefix,
        c(".indices.u32", ".distances.f32", ".weights.f32"))
    result_paths <- paste0(output,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    on.exit(unlink(c(source_paths, result_paths, label_path,
        paste0(output, ".work")), recursive = TRUE))
    indices <- outer(1:8, 1:3,
        function(i, j) (i + j - 1L) %% 8L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.9), 8L),
        nrow = 8L, byrow = TRUE)
    write_massive_graph_fixture(prefix, indices, distances)
    knn <- massive_knn_graph(prefix, 8L, 3L)
    fuzzy <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn), output,
        memory_limit = "64MB")
    fuzzy <- massive_fuzzy_graph(output, 8L)
    labels <- rep(1:2, each = 4L)
    writeBin(as.integer(labels), label_path, size = 4L,
        endian = "little")
    edges <- massive_read_graph_edges(fuzzy, 1L, 8L)
    unique <- edges$from < edges$to
    reference <- fastEmbedR:::fastembedr_graph_modularity_cpp(
        as.integer(edges$from[unique]),
        as.integer(edges$to[unique]), edges$weight[unique],
        8L, as.integer(labels), 1)
    result <- massive_graph_modularity(fuzzy, label_path, 2L,
        chunk_rows = 1L, memory_limit = "64MB")
    expect_equal(result$modularity, reference, tolerance = 1e-6)
    expect_equal(result$n_edge_pairs, sum(unique))
    expect_equal(massive_graph_modularity(fuzzy, label_path, 2L,
        chunk_rows = 8L, memory_limit = "64MB")$modularity,
        result$modularity)
    expect_identical(result$backend, "cpu")
    expect_equal(result$resources$label_mmap_bytes, 32)
    descriptor <- structure(list(nrow = 8L,
        n_communities = 2L, membership_path = label_path),
        class = "fastEmbedR_massive_clusters")
    expect_equal(massive_graph_modularity(fuzzy, descriptor,
        chunk_rows = 2L, memory_limit = "64MB")$modularity,
        reference, tolerance = 1e-6)
    descriptor$nrow <- 7L
    expect_error(massive_graph_modularity(fuzzy, descriptor),
        "vertex counts differ")
    expect_error(massive_graph_modularity(fuzzy, label_path, 2L,
        backend = "cuda"), "no backend fallback")
    expect_error(massive_graph_modularity(knn, label_path, 2L),
        "symmetric massive UMAP")
    expect_error(massive_graph_modularity(fuzzy, label_path, 2L,
        memory_limit = "16MB"), "memory_limit")
    writeBin(as.integer(c(1L, rep(3L, 7L))), label_path,
        size = 4L, endian = "little")
    expect_error(massive_graph_modularity(fuzzy, label_path, 2L),
        "exceeds")
    writeBin(1L, label_path, size = 4L, endian = "little")
    expect_error(massive_graph_modularity(fuzzy, label_path, 2L),
        "invalid size")
})

test_that("disk-backed t-SNE affinities match native float32 affinities", {
    source <- tempfile()
    output <- tempfile()
    output4 <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    outputs <- paste0(output, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    outputs4 <- paste0(output4, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    on.exit(unlink(c(inputs, outputs, outputs4,
        paste0(c(outputs, outputs4), ".part"),
        paste0(c(output, output4), ".work"),
        paste0(c(output, output4), ".affinity.rds")),
        recursive = TRUE))
    indices <- outer(1:8, 1:3,
        function(i, j) (i + j - 1L) %% 8L + 1L)
    distances <- matrix(rep(c(0.15, 0.5, 1.1), 8L),
        nrow = 8L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 8L, 3L)
    graph <- massive_tsne_affinities(knn, 2.5, output,
        memory_limit = "64MB")
    graph4 <- massive_tsne_affinities(knn, 2.5, output4,
        memory_limit = "64MB", n.cores = 4L)
    expect_identical(graph$n.cores, 1L)
    expect_identical(graph4$n.cores, 4L)
    expect_identical(unname(tools::md5sum(outputs)),
        unname(tools::md5sum(outputs4)))
    restored <- massive_affinity_graph(output, 8L)
    expect_identical(restored$file_identity, graph$file_identity)
    expect_identical(restored$perplexity, 2.5)
    expect_true(restored$symmetrized)
    reference <- fastEmbedR:::opentsne_force_diagnostic_cpp(
        indices, distances, matrix(seq_len(16L) / 19,
            ncol = 2L), 2.5, 1, 32L, 1L)
    expect_true(graph$symmetrized)
    expect_identical(graph$weight_method, "tsne_compact_affinity")
    for (row in seq_len(8L)) {
        first <- reference$affinity_row_ptr0[row] + 1L
        last <- reference$affinity_row_ptr0[row + 1L]
        expected <- seq.int(first, last)
        observed <- massive_read_graph_edges(graph, row, 1L)
        order <- order(reference$affinity_col1[expected])
        expect_identical(observed$to,
            reference$affinity_col1[expected][order])
        expect_equal(observed$weight,
            reference$affinity_weight[expected][order],
            tolerance = 1e-6)
    }
    expect_equal(sum(reference$affinity_weight), 1,
        tolerance = 1e-6)
    expect_error(massive_tsne_affinities(knn, 4, tempfile()),
        "perplexity")
    expect_error(massive_tsne_affinities(knn, 2.5, tempfile(),
        n.cores = 0L), "n.cores")
    expect_error(massive_tsne_affinities(knn, 2.5, tempfile(),
        n.cores = 1.5), "n.cores")
    expect_error(massive_tsne_affinities(knn, 2.5, tempfile(),
        backend = "cuda"), "no backend fallback")
    expect_error(massive_tsne_affinities(knn, 2.5, output),
        "already exist")
    manifest <- paste0(output, ".affinity.rds")
    saved <- readRDS(manifest)
    saved$perplexity <- 4
    saveRDS(saved, manifest)
    expect_error(massive_affinity_graph(output, 8L), "manifest")
    saved$perplexity <- 2.5
    saveRDS(saved, manifest)
    stamp <- file.info(outputs[[3L]])$mtime
    con <- file(outputs[[3L]], "r+b")
    writeBin(0.1234567, con, size = 4L, endian = "little")
    close(con)
    Sys.setFileTime(outputs[[3L]], stamp)
    expect_error(massive_affinity_graph(output, 8L), "content differs")
    unlink(manifest)
    expect_error(massive_affinity_graph(output, 8L), "manifest")
})

test_that("full-graph t-SNE streams affinities without changing FFT steps", {
    source <- tempfile()
    affinity <- tempfile()
    knn_files <- paste0(source,
        c(".indices.u32", ".distances.f32"))
    graph_files <- paste0(affinity,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    graph_work <- paste0(affinity,
        c(".affinity.rds", ".work"))
    layouts <- character()
    on.exit(unlink(c(knn_files, graph_files, graph_work,
        layouts), recursive = TRUE))
    indices <- outer(1:16, 1:3,
        function(i, j) (i + j - 1L) %% 16L + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), 16L),
        nrow = 16L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 16L, 3L)
    graph <- massive_tsne_affinities(knn, 3, affinity,
        memory_limit = "64MB")
    for (dims in 2:3) {
        init <- matrix(seq_len(16L * dims) / 10000,
            nrow = 16L, ncol = dims)
        init_path <- write_massive_matrix_fixture(init)
        output <- tempfile(fileext = ".f32")
        layouts <- c(layouts, init_path, output)
        start <- massive_matrix(init_path, 16L, dims)
        actual <- massive_tsne_optimize(graph, start, output,
            early_exaggeration_iter = 2L, n_iter = 3L,
            learning_rate = 1, memory_limit = "1GB", n.cores = 1L)
        reference <- fastEmbedR:::knn_tsne_opentsne_float_cpp(
            indices, distances, init, TRUE, dims, 3, 0.5,
            2L, 3L, 12, 1, 1, FALSE, 0.8, 0.8, 0.01, Inf,
            "fft", 1L, 4L, FALSE, FALSE, FALSE, 5000)
        expect_identical(actual$backend,
            "native_cpu_fft_graph_stream")
        expect_identical(actual$mode, "out_of_core_graph")
        expect_equal(massive_read_rows(actual, 1L, 16L),
            reference$Y, tolerance = 1e-4)
        public_path <- tempfile(fileext = ".f32")
        layouts <- c(layouts, public_path)
        public <- tsne(graph, perplexity = 3,
            n_components = dims, Y_init = start,
            early_exaggeration_iter = 2L, n_iter = 3L,
            learning_rate = 1, max_step_norm = Inf,
            backend = "cpu", massive = "out_of_core_graph",
            output = public_path, memory_limit = "1GB",
            checkpoint = TRUE)
        expect_equal(massive_read_rows(public, 1L, 16L),
            reference$Y, tolerance = 1e-4)
        expect_false(file.exists(paste0(public_path,
            ".checkpoint.rds")))
        if (dims == 3L ||
            !fastEmbedR:::embedding_cuda_available_cpp())
            expect_error(massive_tsne_optimize(graph, start,
                tempfile(fileext = ".f32"), backend = "cuda"),
                "no backend fallback")
        expect_error(massive_tsne_optimize(graph, start,
            tempfile(fileext = ".f32"), memory_limit = "64MB"),
            "memory_limit")
    }
})

test_that("streamed CUDA t-SNE uses its FFT optimizer", {
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
        skip("CUDA is unavailable")
    source <- tempfile()
    affinity <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    edges <- paste0(affinity,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    init_path <- tempfile(fileext = ".f32")
    cpu_path <- tempfile(fileext = ".f32")
    cuda_path <- tempfile(fileext = ".f32")
    public_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(inputs, edges, init_path, cpu_path,
        cuda_path, public_path,
        paste0(affinity, ".affinity.rds"))))
    n <- 32L
    ids <- outer(seq_len(n), 1:4,
        function(i, j) (i + j - 1L) %% n + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9, 1.2), n),
        nrow = n, byrow = TRUE)
    write_massive_graph_fixture(source, ids, distances)
    knn <- massive_knn_graph(source, n, 4L)
    graph <- massive_tsne_affinities(knn, 4, affinity,
        memory_limit = "64MB")
    writeBin(as.vector(t(matrix(seq_len(n * 2L) / 10000,
        nrow = n))), init_path, size = 4L)
    start <- massive_matrix(init_path, n, 2L)
    args <- list(graph, start, early_exaggeration_iter = 2L,
        n_iter = 3L, learning_rate = 1, memory_limit = "1GB")
    cpu <- do.call(massive_tsne_optimize,
        c(args, list(output = cpu_path)))
    cuda <- do.call(massive_tsne_optimize,
        c(args, list(output = cuda_path, backend = "cuda")))
    observed <- massive_read_rows(cuda, 1L, n)
    expect_identical(cuda$backend, "native_cuda_fft_graph_stream")
    expect_identical(cuda$optimizer, "fft_grid_cuda_cufft")
    stages <- unlist(cuda$stage_seconds, use.names = TRUE)
    expect_named(stages, c("repulsion_enqueue", "graph_visit",
        "attraction_sync", "update_sync", "output_write"))
    expect_true(all(is.finite(stages) & stages >= 0))
    expect_lte(sum(stages), cuda$elapsed_seconds_current_call + 0.01)
    expect_true(all(is.finite(observed)))
    expect_equal(observed, massive_read_rows(cpu, 1L, n),
        tolerance = 1e-4)
    public <- tsne(graph, perplexity = 4, n_components = 2L,
        Y_init = start, backend = "cuda",
        massive = "out_of_core_graph", output = public_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, max_step_norm = Inf,
        memory_limit = "1GB")
    expect_identical(public$backend, "native_cuda_fft_graph_stream")
    expect_equal(massive_read_rows(public, 1L, n), observed,
        tolerance = 1e-4)
})

test_that("streamed CUDA t-SNE resumes saved optimizer state", {
    if (!isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
        skip("CUDA is unavailable")
    source <- tempfile()
    affinity <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    edges <- paste0(affinity,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    init_path <- write_massive_matrix_fixture(
        matrix(seq_len(32L) / 10000, nrow = 16L))
    reference_path <- tempfile(fileext = ".f32")
    resumed_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(inputs, edges, init_path, reference_path,
        resumed_path, paste0(affinity, ".affinity.rds"),
        paste0(resumed_path, ".checkpoint.rds"),
        Sys.glob(paste0(resumed_path, ".iter_*.state.f32*")))))
    ids <- outer(1:16, 1:3,
        function(i, j) (i + j - 1L) %% 16L + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), 16L),
        nrow = 16L, byrow = TRUE)
    write_massive_graph_fixture(source, ids, distances)
    knn <- massive_knn_graph(source, 16L, 3L)
    graph <- massive_tsne_affinities(knn, 3, affinity,
        memory_limit = "64MB")
    init <- massive_matrix(init_path, 16L, 2L)
    args <- list(graph, init, early_exaggeration_iter = 2L,
        n_iter = 3L, learning_rate = 1, memory_limit = "1GB",
        backend = "cuda")
    reference <- do.call(massive_tsne_optimize,
        c(args, list(output = reference_path)))
    plan <- fastEmbedR:::massive_tsne_optimize_plan(graph, init,
        resumed_path, 2L, 3L, 12, 1, 1, 0.8, 0.8, 0.01,
        Inf, "1GB", "cuda", 1L)
    saved <- fastEmbedR:::massive_tsne_checkpoint_prepare(plan,
        TRUE, FALSE, 2L)
    interrupt <- function(iteration, snapshot, elapsed) {
        saved$progress(iteration, snapshot, elapsed)
        stop("forced interruption")
    }
    expect_error(fastEmbedR:::massive_tsne_optimize_cuda_cpp(
        graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$access, init$path,
        plan$output, 16L, 2L, 2L, 3L, 12, 1, 1,
        FALSE, 0.8, 0.8, 0.01, Inf, 0L, "", 2L,
        interrupt, plan$edge_capacity), "forced interruption")
    resumed <- do.call(massive_tsne_optimize, c(args,
        list(output = resumed_path, checkpoint = TRUE,
            resume = TRUE, checkpoint_every = 2L)))
    expect_identical(resumed$parameters$resumed_from_iteration, 2L)
    expect_equal(massive_read_rows(resumed, 1L, 16L),
        massive_read_rows(reference, 1L, 16L), tolerance = 1e-4)
})

test_that("full-graph t-SNE resumes the same optimizer trajectory", {
    source <- tempfile()
    affinity <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    graph_files <- paste0(affinity,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    init_path <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    resumed_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(inputs, graph_files, init_path,
        reference_path, resumed_path,
        paste0(affinity, c(".affinity.rds", ".work")),
        paste0(resumed_path, ".checkpoint.rds"),
        Sys.glob(paste0(resumed_path, ".iter_*.state.f32*"))),
        recursive = TRUE))
    indices <- outer(1:16, 1:3,
        function(i, j) (i + j - 1L) %% 16L + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), 16L),
        nrow = 16L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 16L, 3L)
    graph <- massive_tsne_affinities(knn, 3, affinity,
        memory_limit = "64MB")
    init <- matrix(seq_len(32L) / 10000, nrow = 16L)
    writeBin(as.numeric(t(init)), init_path,
        size = 4L, endian = "little")
    start <- massive_matrix(init_path, 16L, 2L)
    reference <- massive_tsne_optimize(graph, start, reference_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "1GB")
    plan <- massive_tsne_optimize_plan(graph, start, resumed_path,
        2L, 3L, 12, 1, 1, 0.8, 0.8, 0.01, Inf,
        "1GB", "cpu", 1L)
    saved <- massive_tsne_checkpoint_prepare(plan,
        TRUE, FALSE, 2L)
    interrupt <- function(iteration, snapshot, elapsed_seconds) {
        saved$progress(iteration, snapshot, elapsed_seconds)
        stop("forced interruption")
    }
    expect_error(massive_tsne_optimize_cpp(graph$offsets_path,
        graph$indices_path, graph$weights_path, graph$access,
        start$path, plan$output, 16L, 2L, 2L, 3L, 12, 1,
        1, FALSE, 0.8, 0.8, 0.01, Inf, 1L, 0L, "", 2L,
        interrupt), "forced interruption")
    expect_true(file.exists(paste0(resumed_path,
        ".checkpoint.rds")))
    sidecar <- paste0(resumed_path, ".checkpoint.rds")
    state <- readRDS(sidecar)
    original <- readBin(state$state_path, "raw",
        n = file.info(state$state_path)$size)
    corrupted <- original
    corrupted[[1L]] <- as.raw(bitwXor(
        as.integer(original[[1L]]), 1L))
    writeBin(corrupted, state$state_path)
    state$state_identity <- massive_checkpoint_file_identity(
        state$state_path)
    saveRDS(state, sidecar)
    expect_error(massive_tsne_optimize(graph, start, resumed_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "1GB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L), "does not match")
    writeBin(original, state$state_path)
    state$state_identity <- massive_checkpoint_file_identity(
        state$state_path)
    saveRDS(state, sidecar)
    expect_error(massive_tsne_optimize(graph, start, resumed_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 2, memory_limit = "1GB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L), "does not match")
    resumed <- massive_tsne_optimize(graph, start, resumed_path,
        early_exaggeration_iter = 2L, n_iter = 3L,
        learning_rate = 1, memory_limit = "1GB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L)
    expect_identical(resumed$parameters$resumed_from_iteration, 2L)
    expect_gte(resumed$elapsed_seconds,
        resumed$elapsed_seconds_current_call)
    expect_identical(unname(tools::md5sum(reference$path)),
        unname(tools::md5sum(resumed$path)))
    expect_false(file.exists(paste0(resumed_path,
        ".checkpoint.rds")))
})

test_that("t-SNE affinity sorting spills multiple bounded runs", {
    n <- 80000L
    k <- 16L
    source <- tempfile()
    output <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    outputs <- paste0(output, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    on.exit(unlink(c(inputs, outputs, paste0(outputs, ".part"),
        paste0(output, c(".work", ".affinity.rds"))),
        recursive = TRUE))
    indices <- outer(seq_len(n), seq_len(k),
        function(i, j) (i + j - 1L) %% n + 1L)
    writeBin(as.integer(t(indices)), inputs[[1L]],
        size = 4L, endian = "little")
    writeBin(rep(1, n * k), inputs[[2L]],
        size = 4L, endian = "little")
    knn <- massive_knn_graph(source, n, k)
    graph <- massive_tsne_affinities(knn, k, output,
        memory_limit = "64MB")
    expect_equal(graph$n_edges, 2 * n * k)
    expect_equal(graph$max_degree, 2 * k)
    expect_equal(sum(massive_read_graph_edges(graph, 1, 1)$weight),
        1 / n, tolerance = 1e-6)
})

test_that("t-SNE affinity sorting resumes across a run merge", {
    n <- 135000L
    k <- 16L
    source <- tempfile()
    output <- tempfile()
    baseline <- tempfile()
    inputs <- paste0(source, c(".indices.u32", ".distances.f32"))
    suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
    on.exit(unlink(c(inputs, paste0(output, suffix),
        paste0(output, c(".work", ".affinity.rds",
            ".sort-checkpoint.rds")),
        paste0(baseline, c(suffix, ".work", ".affinity.rds"))),
        recursive = TRUE))
    indices <- outer(seq_len(n), seq_len(k), function(i, j)
        (i + j - 1L) %% n + 1L)
    writeBin(as.integer(t(indices)), inputs[[1L]],
        size = 4L, endian = "little")
    writeBin(rep(1, n * k), inputs[[2L]],
        size = 4L, endian = "little")
    knn <- massive_knn_graph(source, n, k)
    plan <- fastEmbedR:::massive_umap_fuzzy_preflight(
        knn, output, "64MB", ".affinity.rds")
    signature <- fastEmbedR:::massive_graph_sort_signature(
        knn, plan, "tsne", k, 2L, 1L)
    state <- fastEmbedR:::massive_graph_sort_state(
        knn, plan, signature, FALSE)
    progress <- fastEmbedR:::massive_graph_sort_progress(
        plan, state, k)
    interrupted <- function(done, paths) {
        progress(done, paths)
        if (done == n) stop("deliberate final-pair interruption")
    }
    expect_error(fastEmbedR:::massive_symmetrize_graph_cpp(
        knn$indices_path, knn$values_path, n, k, output,
        plan$limit, "tsne", k, 2L, FALSE, 0, character(),
        1L, interrupted), "deliberate")
    saved <- readRDS(paste0(output, ".sort-checkpoint.rds"))
    expect_equal(saved$completed_rows, n)
    expect_gt(length(saved$runs$paths), 32L)
    writeBin(as.raw(1), paste0(output,
        ".work/rows_run_0.bin"))
    expect_error(massive_tsne_affinities(knn, k - 1, output,
        "64MB", n.cores = 2L, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 1L), "checkpoint")
    resumed <- massive_tsne_affinities(knn, k, output,
        "64MB", n.cores = 2L, checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 1L)
    fresh <- massive_tsne_affinities(knn, k, baseline,
        "64MB", n.cores = 2L)
    expect_equal(resumed$n_edges, fresh$n_edges)
    expect_identical(unname(tools::md5sum(paste0(output, suffix))),
        unname(tools::md5sum(paste0(baseline, suffix))))
    expect_false(file.exists(paste0(output, ".work")))
})

test_that("fuzzy graph sorting merges multiple disk runs", {
    rows <- 80000L
    k <- 16L
    source <- tempfile()
    output <- tempfile()
    source_paths <- paste0(source,
        c(".indices.u32", ".weights.f32"))
    result_paths <- paste0(output,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    on.exit(unlink(c(source_paths, result_paths,
        paste0(result_paths, ".part"), paste0(output, ".work")),
        recursive = TRUE))
    indices <- outer(seq_len(rows), seq_len(k),
        function(i, j) (i + j - 1L) %% rows + 1L)
    writeBin(as.integer(t(indices)), source_paths[[1L]],
        size = 4L, endian = "little")
    writeBin(rep(0.5, rows * k), source_paths[[2L]],
        size = 4L, endian = "little")
    directed <- massive_weighted_graph(source, rows, k)
    directed$weight_method <- "umap_directed_membership"
    fuzzy <- massive_umap_fuzzy_graph(directed, output,
        memory_limit = "64MB")
    expect_equal(fuzzy$n_edges, 2 * rows * k)
    expect_equal(fuzzy$max_degree, 2 * k)
    expect_false(file.exists(paste0(output, ".work")))
    first <- massive_read_graph_edges(fuzzy, 1, 1)
    expect_identical(first$to, c(2:17, 79985:80000))
    expect_equal(first$weight, rep(0.5, 2 * k))
})

test_that("streamed UMAP optimizer matches resident CSR on a small graph", {
    source <- tempfile()
    fuzzy_prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    output_neg <- tempfile(fileext = ".f32")
    output_short <- tempfile(fileext = ".f32")
    init_3d_path <- tempfile(fileext = ".f32")
    output_3d <- tempfile(fileext = ".f32")
    output_mmap <- tempfile(fileext = ".f32")
    source_paths <- paste0(source,
        c(".indices.u32", ".distances.f32", ".weights.f32"))
    fuzzy_paths <- paste0(fuzzy_prefix,
        c(".offsets.u64", ".indices.u32", ".weights.f32"))
    on.exit(unlink(c(source_paths, fuzzy_paths, init_path, output,
        output_neg, output_short, init_3d_path, output_3d,
        output_mmap,
        paste0(c(output, output_neg, output_short, output_3d,
            output_mmap),
            ".part"),
        paste0(fuzzy_prefix, ".work")),
        recursive = TRUE))
    indices <- outer(1:12, 1:4,
        function(i, j) (i + j - 1L) %% 12L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.8, 1.4), 12L),
        nrow = 12L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 12L, 4L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy_prefix,
        memory_limit = "64MB")
    graph <- massive_fuzzy_graph(fuzzy_prefix, 12L)
    set.seed(49)
    start <- matrix(stats::rnorm(24L, sd = 0.2), ncol = 2L)
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, 12L, 2L)
    result <- massive_umap_optimize(graph, init, output,
        n_epochs = 30L, negative_sample_rate = 0L,
        memory_limit = "256MB", chunk_rows = 3L)
    observed <- massive_read_rows(result, 1L, 12L)
    native <- fastEmbedR:::umap_graph_csr_cpp(
        indices, distances, 0L, 4L, 4L, 1L)
    full_edges <- massive_read_graph_edges(graph, 1L, 12L)
    expect_identical(full_edges$to, native$neighbors + 1L)
    expect_equal(full_edges$weight, native$weights,
        tolerance = 1e-5)
    reference <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        native$offsets, native$neighbors, native$weights,
        start, 30L, 0.1, 0L, 1, 1, 1L, 1L, FALSE)
    expect_s3_class(result, "fastEmbedR_massive_matrix")
    expect_identical(result$mode, "out_of_core_graph")
    expect_equal(dim(observed), c(12L, 2L))
    expect_true(all(is.finite(observed)))
    expect_equal(observed, reference, tolerance = 0.02)
    if (.Platform$OS.type != "windows") {
        mapped <- massive_umap_optimize(graph, init, output_mmap,
            n_epochs = 30L, negative_sample_rate = 0L,
            memory_limit = "130MB", chunk_rows = 3L,
            layout_storage = "mmap")
        expect_identical(readBin(output_mmap, "raw", n = 96L),
            readBin(output, "raw", n = 96L))
        expect_identical(mapped$layout_storage, "mmap")
        expect_equal(mapped$resources$layout_mmap_bytes, 96)
        expect_lt(mapped$resources$layout_ram_bytes, 96)
        expect_false(file.exists(paste0(output_mmap, ".part")))
    }
    expect_error(massive_umap_optimize(graph, init, tempfile(
        fileext = ".f32"), layout_storage = "unsupported"),
        "must be 'memory', 'mmap', or 'managed'")
    expect_error(massive_umap_optimize(graph, init, tempfile(
        fileext = ".f32"), n_epochs = 1L), "optimizer controls")
    short <- massive_umap_optimize(graph, init, output_short,
        n_epochs = 2L, negative_sample_rate = 2L,
        memory_limit = "256MB", chunk_rows = 4L)
    reference_short <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        native$offsets, native$neighbors, native$weights,
        start, 2L, 0.1, 2L, 1, 1, 1L, 1L, FALSE)
    expect_equal(massive_read_rows(short, 1L, 12L),
        reference_short, tolerance = 0.001)
    start_3d <- cbind(start, seq_len(12L) / 30)
    writeBin(as.vector(t(start_3d)), init_3d_path,
        size = 4L, endian = "little")
    init_3d <- massive_matrix(init_3d_path, 12L, 3L)
    short_3d <- massive_umap_optimize(graph, init_3d, output_3d,
        n_epochs = 2L, negative_sample_rate = 2L,
        memory_limit = "256MB", chunk_rows = 4L)
    reference_3d <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        native$offsets, native$neighbors, native$weights,
        start_3d, 2L, 0.1, 2L, 1, 1, 1L, 1L, FALSE)
    expect_equal(massive_read_rows(short_3d, 1L, 12L),
        reference_3d, tolerance = 0.001)
    with_negatives <- massive_umap_optimize(graph, init, output_neg,
        n_epochs = 30L, negative_sample_rate = 2L,
        learning_rate = 0.05, memory_limit = "256MB",
        chunk_rows = 4L)
    reference_neg <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        native$offsets, native$neighbors, native$weights,
        start, 30L, 0.1, 2L, 0.05, 1, 1L, 1L, FALSE)
    expect_equal(massive_read_rows(with_negatives, 1L, 12L),
        reference_neg, tolerance = 0.15)
    expect_gt(with_negatives$updates$negative_updates, 0)
    if (!fastEmbedR:::embedding_cuda_available_cpp()) {
        expect_error(massive_umap_optimize(graph, init, tempfile(
            fileext = ".f32"), backend = "cuda"), "no CPU fallback")
    }
    expect_error(massive_umap_optimize(graph, init, tempfile(
        fileext = ".f32"), n.cores = 2L), "one worker")
    expect_error(massive_umap_optimize(graph, init, tempfile(
        fileext = ".f32"), memory_limit = "64MB"), "memory_limit")
})

test_that("public UMAP runs the prepared full graph without fallback", {
    source <- tempfile()
    prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    mapped_path <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32", ".weights.f32")),
        paste0(prefix, c(".offsets.u64", ".indices.u32",
            ".weights.f32")), init_path, output, mapped_path,
        reference_path,
        paste0(prefix, ".work")), recursive = TRUE))
    indices <- outer(1:12, 1:4,
        function(i, j) (i + j - 1L) %% 12L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.8, 1.4), 12L),
        nrow = 12L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 12L, 4L)
    graph <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn), prefix,
        memory_limit = "64MB")
    start <- matrix(seq_len(36L) / 100, ncol = 3L)
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, 12L, 3L)
    public <- umap(graph, n_components = 3L,
        massive = "out_of_core_graph", init = init,
        output = output, memory_limit = "256MB", checkpoint = TRUE)
    reference <- massive_umap_optimize(graph, init, reference_path,
        seed = 4L, memory_limit = "256MB")
    expect_identical(public$mode, "out_of_core_graph")
    expect_identical(public$backend, "cpu")
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    expect_equal(massive_read_rows(public, 1L, 12L),
        massive_read_rows(reference, 1L, 12L), tolerance = 0)
    if (.Platform$OS.type != "windows") {
        mapped <- umap(graph, n_components = 3L,
            massive = "out_of_core_graph", init = init,
            output = mapped_path, memory_limit = "130MB",
            layout_storage = "mmap")
        expect_identical(mapped$layout_storage, "mmap")
        expect_identical(readBin(mapped_path, "raw", n = 144L),
            readBin(output, "raw", n = 144L))
    }
    if (!fastEmbedR:::embedding_cuda_available_cpp()) {
        expect_error(umap(graph, n_components = 3L,
            backend = "cuda", massive = "out_of_core_graph",
            init = init, output = tempfile(fileext = ".f32")),
        "no CPU fallback")
    }
    expect_error(umap(graph, n_components = 3L, n.cores = 2L,
        massive = "out_of_core_graph", init = init,
        output = tempfile(fileext = ".f32")), "one worker")
    expect_error(umap(graph, n_components = 3L, n_neighbors = 4L,
        massive = "out_of_core_graph", init = init,
        output = tempfile(fileext = ".f32")), "prepared fuzzy graph")
    expect_error(umap(graph, init = init), "full-graph UMAP")
    expect_error(umap(graph, layout_storage = "mmap"),
        "full-graph UMAP")
    expect_error(umap(graph, massive = "landmark",
        layout_storage = "mmap"), "full-graph UMAP")
    cuda_error <- if (fastEmbedR:::embedding_cuda_available_cpp()) {
        "no CPU fallback"
    } else "no CPU fallback"
    expect_error(umap(graph, n_components = 3L,
        backend = "cuda", massive = "out_of_core_graph",
        init = init, output = tempfile(fileext = ".f32"),
        layout_storage = "mmap"), cuda_error)
})

test_that("CUDA streams fuzzy UMAP edges without CPU fallback", {
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    source <- tempfile()
    prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    managed_path <- tempfile(fileext = ".f32")
    one_batch_path <- tempfile(fileext = ".f32")
    public_path <- tempfile(fileext = ".f32")
    checkpoint_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32")), paste0(prefix, c(".offsets.u64",
        ".indices.u32", ".weights.f32")), init_path, output,
        one_batch_path, managed_path, public_path, checkpoint_path,
        paste0(checkpoint_path, ".checkpoint.rds"),
        Sys.glob(paste0(checkpoint_path, ".epoch_*.f32")),
        paste0(prefix, ".work")),
        recursive = TRUE))
    n <- 24L
    indices <- outer(seq_len(n), 1:4,
        function(i, j) (i + j - 1L) %% n + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.8, 1.4), n),
        nrow = n, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, n, 4L)), prefix,
        memory_limit = "64MB")
    set.seed(42)
    start <- matrix(stats::rnorm(n * 2L, sd = 0.2), ncol = 2L)
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, n, 2L)
    fitted <- massive_umap_optimize(graph, init, output,
        n_epochs = 30L, negative_sample_rate = 2L,
        learning_rate = 0.05, chunk_rows = 3L,
        memory_limit = "256MB", backend = "cuda")
    layout <- massive_read_rows(fitted, 1L, n)
    expect_identical(fitted$backend, "cuda")
    expect_identical(fitted$layout_storage, "device")
    expect_identical(fitted$optimizer, "streamed_cuda_sgd")
    expect_true(all(is.finite(layout)))
    expect_gt(max(abs(layout - start)), 1e-4)
    expect_gt(fitted$updates$edge_visits, graph$n_edges)
    expect_equal(file.info(output)$size, n * 2L * 4L)
    edges <- massive_read_graph_edges(graph, 1L, n)
    periods <- max(edges$weight) / pmax(edges$weight, 1e-6)
    reference <- fastEmbedR:::umap_cuda_optimize_coo_cpp(
        as.integer(edges$from - 1L), as.integer(edges$to - 1L),
        edges$weight, periods, start, 30L, 2L, 0.05, 0.1,
        1, 1L, 0L)
    one_batch <- massive_umap_optimize(graph, init, one_batch_path,
        n_epochs = 30L, negative_sample_rate = 2L,
        learning_rate = 0.05, chunk_rows = n,
        memory_limit = "256MB", backend = "cuda")
    one_layout <- massive_read_rows(one_batch, 1L, n)
    expect_gt(stats::cor(as.vector(stats::dist(one_layout)),
        as.vector(stats::dist(reference))), 0.9)
    expect_gt(stats::cor(as.vector(stats::dist(layout)),
        as.vector(stats::dist(reference))), 0.65)
    managed <- massive_umap_optimize(graph, init, managed_path,
        n_epochs = 30L, negative_sample_rate = 2L,
        learning_rate = 0.05, chunk_rows = 3L,
        memory_limit = "256MB", backend = "cuda",
        layout_storage = "managed")
    managed_layout <- massive_read_rows(managed, 1L, n)
    expect_identical(managed$backend, "cuda")
    expect_identical(managed$layout_storage, "managed")
    expect_equal(managed$resources$managed_layout_bytes,
        n * 2L * 4L)
    expect_equal(managed$resources$device_buffer_bytes,
        managed$resources$edge_batch_capacity * 16)
    expect_true(all(is.finite(managed_layout)))
    expect_gt(stats::cor(as.vector(stats::dist(layout)),
        as.vector(stats::dist(managed_layout))), 0.9)
    expect_lt(fitted$resources$device_buffer_bytes,
        graph$n_edges * 16 + n * 2L * 4L + 1)
    public <- umap(graph, init = init, output = public_path,
        massive = "out_of_core_graph", backend = "cuda",
        memory_limit = "256MB")
    expect_identical(public$backend, "cuda")
    expect_true(all(is.finite(massive_read_rows(public, 1L, n))))
    plan <- fastEmbedR:::massive_umap_optimize_plan(graph, init,
        checkpoint_path, 30L, 2L, 0.05, 0.1, 1, 1L, 3L,
        "256MB", "cuda", 1L)
    saved <- fastEmbedR:::massive_umap_checkpoint_prepare(
        plan, TRUE, FALSE, 5L)
    interrupt <- function(epoch, snapshot, positive, negative,
            edge_visits) {
        saved$progress(epoch, snapshot, positive, negative,
            edge_visits)
        stop("simulated CUDA interruption")
    }
    expect_error(fastEmbedR:::massive_umap_optimize_cuda_cpp(
        graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$n_vertices,
        saved$start_path, saved$format, checkpoint_path,
        30L, 2L, 0.05, 0.1, 1, 1L, 3L,
        plan$edge_capacity, plan$limit, 0L, 5L,
        interrupt, 0), "simulated CUDA interruption")
    state <- readRDS(paste0(checkpoint_path, ".checkpoint.rds"))
    expect_equal(state$epoch, 5L)
    expect_gt(state$edge_visits, 0)
    expect_error(massive_umap_optimize(graph, init,
        checkpoint_path, n_epochs = 30L,
        negative_sample_rate = 2L, learning_rate = 0.05,
        chunk_rows = 3L, memory_limit = "256MB",
        backend = "cpu", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 5L), "does not match")
    resumed <- massive_umap_optimize(graph, init,
        checkpoint_path, n_epochs = 30L,
        negative_sample_rate = 2L, learning_rate = 0.05,
        chunk_rows = 3L, memory_limit = "256MB",
        backend = "cuda", checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 5L)
    expect_identical(resumed$parameters$resumed_from_epoch, 5L)
    expect_equal(resumed$updates$edge_visits,
        fitted$updates$edge_visits)
    expect_gt(stats::cor(as.vector(stats::dist(
        massive_read_rows(resumed, 1L, n))),
        as.vector(stats::dist(layout))), 0.95)
    expect_false(file.exists(paste0(checkpoint_path,
        ".checkpoint.rds")))
    expect_error(massive_umap_optimize(graph, init,
        tempfile(fileext = ".f32"),
        backend = "cuda", layout_storage = "mmap"), "no CPU fallback")
    expect_error(massive_umap_optimize(graph, init,
        tempfile(fileext = ".f32"),
        backend = "cpu", layout_storage = "managed"),
        "no backend fallback")
})

test_that("CUDA streams three-coordinate UMAP and resumes epochs", {
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    source <- tempfile()
    prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    managed_path <- tempfile(fileext = ".f32")
    public_path <- tempfile(fileext = ".f32")
    resumed_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(Sys.glob(paste0(source, ".*")),
        Sys.glob(paste0(prefix, ".*")), init_path, output,
        public_path, managed_path,
        Sys.glob(paste0(resumed_path, "*"))), recursive = TRUE))
    n <- 18L
    ids <- outer(seq_len(n), 1:3,
        function(i, j) (i + j - 1L) %% n + 1L)
    distances <- matrix(rep(c(0.2, 0.5, 0.9), n),
        nrow = n, byrow = TRUE)
    write_massive_graph_fixture(source, ids, distances)
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, n, 3L)), prefix,
        memory_limit = "64MB")
    start <- cbind(sin(seq_len(n)), cos(seq_len(n)),
        sin(seq_len(n) / 2)) * 0.2
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, n, 3L)
    fitted <- massive_umap_optimize(graph, init, output,
        n_epochs = 20L, negative_sample_rate = 1L,
        learning_rate = 0.05, chunk_rows = 3L,
        memory_limit = "256MB", backend = "cuda")
    observed <- massive_read_rows(fitted, 1L, n)
    expect_identical(fitted$backend, "cuda")
    expect_identical(dim(observed), c(n, 3L))
    expect_true(all(is.finite(observed)))
    expect_gt(max(abs(observed - start)), 1e-4)
    expect_equal(file.info(output)$size, n * 3L * 4L)
    managed <- massive_umap_optimize(graph, init, managed_path,
        n_epochs = 20L, negative_sample_rate = 1L,
        learning_rate = 0.05, chunk_rows = 3L,
        memory_limit = "256MB", backend = "cuda",
        layout_storage = "managed")
    managed_layout <- massive_read_rows(managed, 1L, n)
    expect_identical(managed$layout_storage, "managed")
    expect_identical(dim(managed_layout), c(n, 3L))
    expect_true(all(is.finite(managed_layout)))
    expect_gt(stats::cor(as.vector(stats::dist(observed)),
        as.vector(stats::dist(managed_layout))), 0.9)
    public <- umap(graph, n_components = 3L,
        massive = "out_of_core_graph", init = init,
        output = public_path, backend = "cuda")
    expect_identical(public$backend, "cuda")
    expect_true(all(is.finite(massive_read_rows(public, 1L, n))))
    plan <- fastEmbedR:::massive_umap_optimize_plan(graph, init,
        resumed_path, 20L, 1L, 0.05, 0.1, 1, 1L, 3L,
        "256MB", "cuda", 1L)
    saved <- fastEmbedR:::massive_umap_checkpoint_prepare(
        plan, TRUE, FALSE, 5L)
    interrupt <- function(epoch, snapshot, positive, negative,
            edge_visits) {
        saved$progress(epoch, snapshot, positive, negative,
            edge_visits)
        stop("simulated 3D interruption")
    }
    expect_error(fastEmbedR:::massive_umap_optimize_cuda_cpp(
        graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$n_vertices,
        saved$start_path, saved$format, plan$output,
        20L, 1L, 0.05, 0.1, 1, 1L, 3L,
        plan$edge_capacity, plan$limit, 0L, 5L,
        interrupt, 0, 3L), "simulated 3D interruption")
    resumed <- massive_umap_optimize(graph, init, resumed_path,
        n_epochs = 20L, negative_sample_rate = 1L,
        learning_rate = 0.05, chunk_rows = 3L,
        memory_limit = "256MB", backend = "cuda",
        checkpoint = TRUE, resume = TRUE, checkpoint_every = 5L)
    expect_identical(resumed$parameters$resumed_from_epoch, 5L)
    expect_equal(resumed$updates$edge_visits,
        fitted$updates$edge_visits)
    expect_gt(stats::cor(as.vector(stats::dist(observed)),
        as.vector(stats::dist(massive_read_rows(resumed, 1L, n)))),
        0.9)
})

test_that("streamed UMAP retains neighborhood quality on iris", {
    x <- scale(as.matrix(iris[, 1:4]))
    n <- nrow(x)
    k <- 15L
    distances <- as.matrix(stats::dist(x))
    diag(distances) <- Inf
    indices <- t(vapply(seq_len(n), function(row) {
        as.integer(order(distances[row, ], seq_len(n))[seq_len(k)])
    }, integer(k)))
    knn_distances <- matrix(distances[cbind(
        rep(seq_len(n), each = k), as.vector(t(indices)))],
        nrow = n, byrow = TRUE)
    source <- tempfile()
    prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    mapped_output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32", ".weights.f32")),
        paste0(prefix, c(".offsets.u64", ".indices.u32",
            ".weights.f32")), init_path, output, mapped_output,
        paste0(mapped_output, ".part"))))
    write_massive_graph_fixture(source, indices, knn_distances)
    knn <- massive_knn_graph(source, n, k)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, prefix,
        memory_limit = "64MB")
    start <- scale(x[, 1:2]) * 0.1
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, n, 2L)
    fitted <- massive_umap_optimize(graph, init, output,
        n_epochs = 100L, memory_limit = "256MB")
    layout <- massive_read_rows(fitted, 1L, n)
    if (.Platform$OS.type != "windows") {
        mapped <- massive_umap_optimize(graph, init, mapped_output,
            n_epochs = 100L, memory_limit = "130MB",
            layout_storage = "mmap")
        expect_identical(readBin(output, "raw", n = n * 8L),
            readBin(mapped_output, "raw", n = n * 8L))
        expect_equal(massive_read_rows(mapped, 1L, n), layout)
    }
    native <- fastEmbedR:::umap_graph_csr_cpp(
        indices, knn_distances, 0L, 4L, 4L, 1L)
    reference <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        native$offsets, native$neighbors, native$weights,
        start, 100L, 0.1, 5L, 1, 1, 1L, 1L, FALSE)
    quality <- evaluate_embedding(x, layout, k = k)
    reference_quality <- evaluate_embedding(x, reference, k = k)
    expect_gt(quality$trustworthiness, 0.9)
    expect_gt(quality$knn_preservation, 0.5)
    expect_gte(quality$trustworthiness,
        reference_quality$trustworthiness - 0.05)
    expect_gte(quality$knn_preservation,
        reference_quality$knn_preservation - 0.1)
})

test_that("file-backed PCA scores feed one full KNN and UMAP graph", {
    x <- scale(as.matrix(iris[seq_len(60L), 1:4]))
    source_path <- write_massive_matrix_fixture(x)
    scores_path <- tempfile(fileext = ".f32")
    knn_prefix <- tempfile()
    fuzzy_prefix <- tempfile()
    output <- tempfile(fileext = ".f32")
    on.exit(unlink(c(source_path, scores_path, output,
        paste0(knn_prefix, c(".indices.u32", ".distances.f32",
            ".weights.f32")),
        paste0(fuzzy_prefix, c(".offsets.u64", ".indices.u32",
            ".weights.f32")))))
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    scores <- pca(source, ncomp = 2L, backend = "cpu",
        n.cores = 2L, massive = "out_of_core",
        output = scores_path, memory_limit = "256MB")$scores
    knn <- massive_full_knn_graph(scores, k = 10L,
        output = knn_prefix, n.cores = 2L,
        memory_limit = "256MB", chunk_rows = 10L)
    directed <- massive_umap_memberships(knn)
    graph <- massive_umap_fuzzy_graph(directed, fuzzy_prefix,
        memory_limit = "64MB")
    layout <- massive_umap_optimize(graph, scores, output,
        n_epochs = 5L, memory_limit = "256MB")
    expect_s3_class(layout, "fastEmbedR_massive_matrix")
    expect_equal(c(layout$nrow, layout$ncol), c(60, 2))
    expect_identical(knn$n_vertices, graph$n_vertices)
    observed <- massive_read_rows(layout, 1L, 60L)
    expect_identical(dim(observed), c(60L, 2L))
    expect_true(all(is.finite(observed)))
})

test_that("saved PCA and fuzzy graph feed UMAP and Louvain", {
    skip_on_os("windows")
    root <- tempfile()
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE))
    set.seed(812)
    x <- rbind(
        matrix(rnorm(120L, mean = -3, sd = 0.3), ncol = 4L),
        matrix(rnorm(120L, mean = 3, sd = 0.3), ncol = 4L))
    input <- file.path(root, "input.f32")
    writeBin(as.vector(t(x)), input, size = 4L, endian = "little")
    source <- massive_matrix(input, nrow(x), ncol(x))
    scores <- file.path(root, "scores.f32")
    pca(source, ncomp = 2L, massive = "out_of_core",
        backend = "cpu", output = scores, memory_limit = "256MB")
    unlink(input)
    fit <- massive_open_pca(scores)
    knn <- massive_full_knn_graph(fit$scores, k = 5L,
        output = file.path(root, "knn"), method = "exact",
        chunk_rows = 10L, reference_chunk_rows = 20L,
        memory_limit = "256MB")
    massive_umap_fuzzy_graph(massive_umap_memberships(knn),
        file.path(root, "fuzzy"), memory_limit = "64MB")
    graph <- massive_fuzzy_graph(file.path(root, "fuzzy"), 60L)
    layout <- massive_umap_optimize(graph, fit$scores,
        file.path(root, "umap.f32"), n_epochs = 5L,
        chunk_rows = 15L, layout_storage = "mmap",
        memory_limit = "256MB")
    clusters <- massive_cluster(graph, method = "louvain",
        massive = "out_of_core_graph",
        output = file.path(root, "labels.clusters.u32"),
        n_iterations = 5L, chunk_rows = 15L,
        memory_limit = "128MB")
    labels <- massive_read_louvain_rows(clusters, 1L, 60L)
    expect_true(all(is.finite(massive_read_rows(layout, 1L, 60L))))
    expect_equal(clusters$n_communities, length(unique(labels)))
    expect_gte(clusters$modularity_final,
        clusters$first_level$modularity - 1e-7)
})

test_that("full-graph UMAP resumes a complete epoch snapshot", {
    source <- tempfile()
    fuzzy_prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32", ".weights.f32")),
        paste0(fuzzy_prefix, c(".offsets.u64", ".indices.u32",
            ".weights.f32")), init_path, output, reference_path,
        paste0(output, ".checkpoint.rds"),
        Sys.glob(paste0(output, ".epoch_*.f32")))))
    indices <- outer(1:12, 1:4,
        function(i, j) (i + j - 1L) %% 12L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.8, 1.4), 12L),
        nrow = 12L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 12L, 4L)
    graph <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn), fuzzy_prefix,
        memory_limit = "64MB")
    set.seed(49)
    start <- matrix(stats::rnorm(24L, sd = 0.2), ncol = 2L)
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, 12L, 2L)
    plan <- fastEmbedR:::massive_umap_optimize_plan(graph, init,
        output, 8L, 2L, 1, 0.1, 1, 1L, 4L,
        "256MB", "cpu", 1L)
    saved <- fastEmbedR:::massive_umap_checkpoint_prepare(
        plan, TRUE, FALSE, 3L)
    interrupt <- function(epoch, snapshot, positive, negative) {
        saved$progress(epoch, snapshot, positive, negative)
        stop("simulated interruption")
    }
    expect_error(fastEmbedR:::massive_umap_optimize_cpp(
        graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$n_vertices,
        init$path, init$format, 2L, output, 8L, 2L,
        1, 0.1, 1, 1L, 4L, plan$limit,
        0L, 0, 0, 3L, interrupt), "simulated interruption")
    state <- readRDS(paste0(output, ".checkpoint.rds"))
    expect_equal(state$epoch, 3L)
    expect_true(file.exists(state$snapshot))
    tampered <- state
    tampered$snapshot <- normalizePath(init_path)
    tampered$snapshot_identity <-
        fastEmbedR:::massive_checkpoint_file_identity(init_path)
    saveRDS(tampered, paste0(output, ".checkpoint.rds"))
    expect_error(massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        chunk_rows = 4L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 3L), "does not match")
    expect_true(file.exists(init_path))
    saveRDS(state, paste0(output, ".checkpoint.rds"))
    orphan <- paste0(output, ".epoch_6.f32")
    writeBin(1, orphan, size = 4L, endian = "little")
    expect_error(massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        chunk_rows = 4L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 3L), "Uncommitted UMAP snapshot")
    unlink(orphan)
    expect_error(massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        learning_rate = 0.5, chunk_rows = 4L,
        memory_limit = "256MB", checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 3L), "does not match")
    resumed <- massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        chunk_rows = 4L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 3L)
    reference <- massive_umap_optimize(graph, init,
        reference_path, n_epochs = 8L,
        negative_sample_rate = 2L, chunk_rows = 4L,
        memory_limit = "256MB")
    expect_identical(readBin(resumed$path, "raw",
        n = file.info(resumed$path)$size),
        readBin(reference$path, "raw",
            n = file.info(reference$path)$size))
    expect_equal(resumed$updates, reference$updates)
    expect_identical(resumed$parameters$resumed_from_epoch, 3L)
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    expect_false(file.exists(state$snapshot))
})

test_that("mapped UMAP reserves space for overlapping snapshots", {
    plan <- list(output = tempfile(fileext = ".f32"),
        layout_bytes = 100, backend = "cpu",
        parameters = list(layout_storage = "mmap"))
    with_mocked_bindings(
        massive_disk_available_cpp = function(...) 300,
        expect_error(fastEmbedR:::massive_umap_checkpoint_prepare(
            plan, TRUE, FALSE, 10L), "free-disk budget"),
        .package = "fastEmbedR"
    )
    expect_false(file.exists(plan$output))
    plan$parameters$layout_storage <- "memory"
    accepted <- with_mocked_bindings(
        massive_disk_available_cpp = function(...) 300,
        massive_umap_checkpoint_state = function(...) TRUE,
        fastEmbedR:::massive_umap_checkpoint_prepare(
            plan, TRUE, FALSE, 10L),
        .package = "fastEmbedR"
    )
    expect_true(accepted)
})

test_that("mapped UMAP resumes from an owned epoch snapshot", {
    skip_on_os("windows")
    source <- tempfile()
    fuzzy_prefix <- tempfile()
    init_path <- tempfile(fileext = ".f32")
    output <- tempfile(fileext = ".f32")
    reference_path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(paste0(source, c(".indices.u32",
        ".distances.f32", ".weights.f32")),
        paste0(fuzzy_prefix, c(".offsets.u64", ".indices.u32",
            ".weights.f32")), init_path, output, reference_path,
        paste0(output, c(".part", ".checkpoint.rds")),
        Sys.glob(paste0(output, ".epoch_*.f32")))))
    indices <- outer(1:12, 1:4,
        function(i, j) (i + j - 1L) %% 12L + 1L)
    distances <- matrix(rep(c(0.1, 0.4, 0.8, 1.4), 12L),
        nrow = 12L, byrow = TRUE)
    write_massive_graph_fixture(source, indices, distances)
    knn <- massive_knn_graph(source, 12L, 4L)
    graph <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn), fuzzy_prefix,
        memory_limit = "64MB")
    set.seed(49)
    start <- matrix(stats::rnorm(24L, sd = 0.2), ncol = 2L)
    writeBin(as.vector(t(start)), init_path, size = 4L,
        endian = "little")
    init <- massive_matrix(init_path, 12L, 2L)
    plan <- fastEmbedR:::massive_umap_optimize_plan(graph, init,
        output, 8L, 2L, 1, 0.1, 1, 1L, 4L,
        "256MB", "cpu", 1L, "mmap")
    saved <- fastEmbedR:::massive_umap_checkpoint_prepare(
        plan, TRUE, FALSE, 3L)
    interrupt <- function(epoch, snapshot, positive, negative) {
        saved$progress(epoch, snapshot, positive, negative)
        stop("simulated mapped interruption")
    }
    expect_error(fastEmbedR:::massive_umap_optimize_cpp(
        graph$offsets_path, graph$indices_path,
        graph$weights_path, graph$n_vertices,
        init$path, init$format, 2L, output, 8L, 2L,
        1, 0.1, 1, 1L, 4L, plan$limit,
        0L, 0, 0, 3L, interrupt, "mmap"),
        "simulated mapped interruption")
    state <- readRDS(paste0(output, ".checkpoint.rds"))
    expect_identical(state$epoch, 3L)
    expect_equal(file.info(paste0(output, ".part"))$size, 96)
    expect_true(file.exists(state$snapshot))
    expect_error(massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        chunk_rows = 4L, memory_limit = "256MB",
        checkpoint = TRUE, checkpoint_every = 3L,
        layout_storage = "mmap"), "already exists")
    resumed <- massive_umap_optimize(graph, init, output,
        n_epochs = 8L, negative_sample_rate = 2L,
        chunk_rows = 4L, memory_limit = "256MB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 3L, layout_storage = "mmap")
    reference <- massive_umap_optimize(graph, init,
        reference_path, n_epochs = 8L,
        negative_sample_rate = 2L, chunk_rows = 4L,
        memory_limit = "256MB")
    expect_identical(readBin(resumed$path, "raw", n = 96L),
        readBin(reference$path, "raw", n = 96L))
    expect_equal(resumed$updates, reference$updates)
    expect_identical(resumed$parameters$resumed_from_epoch, 3L)
    expect_false(file.exists(paste0(output, ".part")))
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    expect_false(file.exists(state$snapshot))
})

write_massive_csr_fixture <- function(prefix, offsets, indices,
                                        weights) {
    paths <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    offset_words <- as.integer(rbind(offsets, rep(0, length(offsets))))
    writeBin(offset_words, paths[[1L]], size = 4L, endian = "little")
    writeBin(as.integer(indices), paths[[2L]], size = 4L,
        endian = "little")
    writeBin(as.numeric(weights), paths[[3L]], size = 4L,
        endian = "little")
    paths
}

test_that("variable-degree CSR graphs stream bounded edge partitions", {
    prefix <- tempfile()
    offsets <- c(0, 2, 3, 5, 6)
    indices <- c(2L, 3L, 1L, 1L, 4L, 3L)
    weights <- c(0.5, 0.25, 0.5, 0.25, 0.75, 0.75)
    paths <- write_massive_csr_fixture(prefix, offsets, indices, weights)
    on.exit(unlink(paths))
    streamed <- massive_csr_graph(prefix, 4, access = "stream")
    mapped <- massive_csr_graph(prefix, 4, access = "mmap")
    expect_identical(streamed$storage, "csr")
    expect_identical(streamed$n_edges, 6)
    expect_identical(streamed$max_degree, 2)
    expect_identical(streamed$offset_dtype, "uint64")
    expect_false(streamed$symmetrized)
    expected <- list(from = c(1, 1, 2, 3, 3, 4),
        to = indices, weight = weights)
    expect_equal(massive_read_graph_edges(streamed, 1, 4), expected)
    expect_equal(massive_read_graph_edges(mapped, 1, 4), expected)
    plan <- massive_graph_partitions(streamed, max_bytes = 80)
    expect_identical(plan$n_partitions, 2)
    expect_equal(massive_read_graph_partition(plan, 2),
        list(from = c(3, 3, 4), to = indices[4:6],
            weight = weights[4:6]))
    expect_error(massive_graph_partitions(streamed, 39), "one row")
    expect_error(massive_read_graph_edges(streamed, 4, 2), "range")
    writeBin(1L, paths[[2L]], size = 4L)
    expect_error(massive_read_graph_edges(streamed, 1, 1), "changed")
})

test_that("CSR validation streams hubs beyond one edge block", {
    n <- 65538L
    prefix <- tempfile()
    offsets <- c(0, n - 1L, seq.int(n, 2L * (n - 1L)))
    indices <- c(seq.int(2L, n), rep.int(1L, n - 1L))
    weights <- rep.int(0.5, 2L * (n - 1L))
    paths <- write_massive_csr_fixture(prefix, offsets,
        indices, weights)
    on.exit(unlink(paths))
    graph <- massive_csr_graph(prefix, n, access = "stream")
    expect_equal(graph$max_degree, n - 1L)
    expect_identical(graph$n_edges, 2 * (n - 1L))
    hub <- massive_read_graph_edges(graph, 1L, 1L)
    expect_identical(hub$to[c(65536L, 65537L)],
        c(65537L, 65538L))
    expect_equal(hub$weight, rep.int(0.5, n - 1L))
    indices[n - 1L] <- indices[n - 2L]
    write_massive_csr_fixture(prefix, offsets, indices, weights)
    expect_error(massive_csr_graph(prefix, n), "targets")
})

test_that("CSR edge blocks preserve row boundaries and empty rows", {
    n <- 2100L
    k <- 40L
    prefix <- tempfile()
    neighbors <- lapply(seq_len(n), function(row) {
        if (row %in% c(1L, 1640L, n)) return(integer())
        sort(((row + seq_len(k) - 1L) %% n) + 1L)
    })
    lengths <- lengths(neighbors)
    offsets <- c(0, cumsum(lengths))
    indices <- unlist(neighbors, use.names = FALSE)
    weights <- rep.int(0.5, length(indices))
    paths <- write_massive_csr_fixture(prefix, offsets,
        indices, weights)
    on.exit(unlink(paths))
    expected <- list(from = rep(seq_len(n), lengths),
        to = indices, weight = weights)
    streamed <- massive_csr_graph(prefix, n, access = "stream")
    mapped <- massive_csr_graph(prefix, n, access = "mmap")
    expect_gt(streamed$n_edges, 65536)
    expect_equal(massive_read_graph_edges(streamed, 1L, n),
        expected)
    expect_equal(massive_read_graph_edges(mapped, 1L, n),
        expected)
})

test_that("CSR validation rejects malformed offsets and edges", {
    prefix <- tempfile()
    offsets <- c(0, 2, 3, 5, 6)
    indices <- c(2L, 3L, 1L, 1L, 4L, 3L)
    weights <- c(0.5, 0.25, 0.5, 0.25, 0.75, 0.75)
    paths <- write_massive_csr_fixture(prefix, offsets, indices, weights)
    on.exit(unlink(paths))
    expect_true(massive_csr_graph(prefix, 4)$validated)
    write_massive_csr_fixture(prefix, c(0, 2, 3, 5, 7),
        indices, weights)
    expect_error(massive_csr_graph(prefix, 4), "span")
    write_massive_csr_fixture(prefix, c(0, 2, 1, 5, 6),
        indices, weights)
    expect_error(massive_csr_graph(prefix, 4), "offsets")
    write_massive_csr_fixture(prefix, offsets,
        c(3L, 2L, indices[-(1:2)]), weights)
    expect_error(massive_csr_graph(prefix, 4), "targets")
    write_massive_csr_fixture(prefix, offsets,
        c(2L, 2L, indices[-(1:2)]), weights)
    expect_error(massive_csr_graph(prefix, 4), "targets")
    write_massive_csr_fixture(prefix, offsets,
        c(1L, indices[-1L]), weights)
    expect_error(massive_csr_graph(prefix, 4), "targets")
    write_massive_csr_fixture(prefix, offsets, indices,
        c(1.5, weights[-1L]))
    expect_error(massive_csr_graph(prefix, 4), "weights")
    write_massive_csr_fixture(prefix, offsets, indices, weights)
    writeBin(1L, paths[[3L]], size = 4L)
    expect_error(massive_csr_graph(prefix, 4), "size")
})

test_that("CSR readers accept zero-degree rows", {
    prefix <- tempfile()
    paths <- write_massive_csr_fixture(prefix, c(0, 1, 1, 2),
        c(2L, 1L), c(0.5, 0.5))
    on.exit(unlink(paths))
    graph <- massive_csr_graph(prefix, 3)
    expect_identical(graph$max_degree, 1)
    expect_equal(massive_read_graph_edges(graph, 2, 1),
        list(from = numeric(), to = integer(), weight = numeric()))
})
