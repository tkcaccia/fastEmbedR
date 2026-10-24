test_that("sharded IVF records effective settings", {
    found <- list(backend = "native_cuda_cuvs_ivf_flat",
        method = "ivf", nlist = 64L, nprobe = 16L,
        pilot_rows = 32L, tuning_attempts = 2L,
        pilot_recall = 1, pilot_id_recall = 1,
        pilot_min_row_recall = 1,
        pilot_recall_metric = "distance_at_k")
    setting <- fastEmbedR:::massive_sharded_ivf_setting(
        found, 1L, 101L, 500L)
    expect_identical(setting$shard, 2L)
    expect_identical(setting$first_row, 101L)
    expect_identical(setting$reference_rows, 500L)
    expect_identical(setting$nlist, 64L)
    expect_identical(setting$nprobe, 16L)
    expect_identical(setting$tuning_attempts, 2L)
    expect_identical(setting$pilot_min_row_recall, 1)
    found$pilot_min_row_recall <- NULL
    expect_error(fastEmbedR:::massive_sharded_ivf_setting(
        found, 1L, 101L, 500L), "effective tuning settings")
    found$pilot_min_row_recall <- 0.97
    expect_error(fastEmbedR:::massive_sharded_ivf_setting(
        found, 1L, 101L, 500L), "effective tuning settings")
    found$pilot_min_row_recall <- 1
    found$nprobe <- 65L
    expect_error(fastEmbedR:::massive_sharded_ivf_setting(
        found, 1L, 101L, 500L), "effective tuning settings")
    found$nprobe <- NULL
    expect_error(fastEmbedR:::massive_sharded_ivf_setting(
        found, 1L, 101L, 500L), "effective tuning settings")
})

test_that("sharded IVF rejects changed settings on resume", {
    root <- tempfile("ivf_settings_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE))
    found <- list(backend = "native_cuda_cuvs_ivf_flat",
        method = "ivf", nlist = 64L, nprobe = 16L,
        pilot_rows = 32L, tuning_attempts = 2L,
        pilot_recall = 1, pilot_id_recall = 1,
        pilot_min_row_recall = 1,
        pilot_recall_metric = "distance_at_k")
    setup <- list(k = 2L, workers = 1L,
        resources = list(chunk_rows = 2L),
        checkpoint_every = 1L)
    saved <- list(state = list(completed_rows = 0L,
        pilot_signature = "already-recorded"),
        path = file.path(root, "checkpoint.rds"))
    expect_null(fastEmbedR:::massive_saved_ivf_setting(saved, 0L))
    run <- function(state) fastEmbedR:::massive_sharded_batch(
        list(nrow = 4L), setup, NULL, state, 0L, 1L, 4L,
        1L, "ivf", "cuda", saved)
    with_mocked_bindings(
        massive_sharded_query = function(...) NULL,
        massive_knn_query = function(...) found,
        massive_merge_knn_shard_cpp = function(...) NULL,
        {
            batch <- run(NULL)
            expect_identical(batch$ivf_setting$nprobe, 16L)
            expect_identical(readRDS(saved$path)$ivf_settings[[1L]],
                batch$ivf_setting)
            saved$state$ivf_settings <- list(batch$ivf_setting)
            expect_identical(fastEmbedR:::massive_saved_ivf_setting(
                saved, 0L), batch$ivf_setting)
            found$nprobe <- 32L
            expect_error(run(NULL), "Checkpointed IVF settings changed")
        }, .package = "fastEmbedR")
})

test_that("single-GPU IVF graph exposes per-shard settings", {
    setting <- list(shard = 1L, nlist = 64L, nprobe = 16L)
    setup <- list(k = 2L, method = "ivf_sharded", ef_search = 0L,
        hnsw_build = NULL, resources = list(),
        checkpoint_every = 100L)
    paths <- c(indices = "graph.indices.u32",
        distances = "graph.distances.f32")
    with_mocked_bindings(
        massive_knn_graph = function(...) list(),
        massive_checkpoint_implementation_identity = function(...)
            list(source = "test"),
        {
            graph <- fastEmbedR:::massive_sharded_result(
                list(nrow = 4L), paths, setup, "cuda", list(),
                1, proc.time()[[3L]], 0L, FALSE, FALSE,
                list(), list(setting))
            expect_identical(graph$ivf_settings, list(setting))
            expect_identical(graph$method,
                "native_cuda_cuvs_ivf_sharded")
        }, .package = "fastEmbedR")
})

test_that("single-GPU IVF resumes a completed reference shard", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    root <- tempfile("ivf_shard_resume_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE))
    input <- file.path(root, "input.f32")
    set.seed(19L)
    writeBin(as.vector(t(matrix(rnorm(240L), 40L, 6L))),
        input, size = 4L)
    source <- massive_matrix(input, 40L, 6L)
    expected <- massive_full_knn_graph(source, 3L,
        file.path(root, "expected"), backend = "cuda",
        method = "ivf_sharded", chunk_rows = 7L,
        reference_chunk_rows = 15L, memory_limit = "512MB")
    resources <- fastEmbedR:::massive_cuda_sharded_resources(
        source, 3L, 7L, 15L, "512MB", method = "ivf_sharded")
    prefix <- file.path(root, "resumed")
    paths <- fastEmbedR:::massive_knn_paths(prefix,
        resources$output_bytes)
    parts <- stats::setNames(paste0(paths, ".part"), names(paths))
    setup <- list(k = 3L, workers = 1L, build_workers = 1L,
        resources = resources, method = "ivf_sharded",
        ef_search = 0L, hnsw_build = NULL, checkpoint_every = 2L)
    saved <- fastEmbedR:::massive_sharded_checkpoint(source,
        setup, paths, "cuda", FALSE)
    fastEmbedR:::massive_index_shard(source, setup, parts, 0L,
        fastEmbedR:::massive_checkpoint_source_identity(source),
        "cuda", saved)
    expect_identical(readRDS(saved$path)$shard, 1L)
    observed <- massive_full_knn_graph(source, 3L, prefix,
        backend = "cuda", method = "ivf_sharded",
        chunk_rows = 7L, reference_chunk_rows = 15L,
        memory_limit = "512MB", checkpoint = TRUE,
        resume = TRUE, checkpoint_every = 2L)
    expect_true(observed$resumed)
    expect_equal(massive_read_graph_edges(observed, 1L, 40L),
        massive_read_graph_edges(expected, 1L, 40L))
})
