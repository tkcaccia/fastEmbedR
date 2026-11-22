test_that("sharded HNSW uses a memory-bounded query batch", {
    source <- list(nrow = 1000000, ncol = 100L, format = "fbin")
    default <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, NULL, 20000L, "4GB")
    small <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, 256L, 20000L, "4GB")
    expect_identical(default$chunk_rows, 8192L)
    expect_identical(small$chunk_rows, 256L)
    expect_lt(default$query_batches, small$query_batches)
    expect_lt(default$peak_ram_bytes, default$memory_limit_bytes)
})

test_that("default HNSW shard size uses the available RAM budget", {
    source <- list(nrow = 1000021L, ncol = 11L, format = "f32")
    roomy <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, NULL, NULL, "8GB")
    tight <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, NULL, NULL, "512MB")
    capped <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, NULL, 250000L, "8GB")
    expect_identical(roomy$n_shards, 1L)
    expect_equal(roomy$reference_chunk_rows, 1000021)
    expect_gt(tight$n_shards, 1L)
    expect_lte(tight$peak_ram_bytes, tight$memory_limit_bytes)
    expect_identical(capped$n_shards, 5L)
    expect_lte(capped$reference_chunk_rows, 250000L)
})

test_that("file HNSW budgets two vector copies for a single bulk add", {
    source <- list(nrow = 1000000L, ncol = 1024L, format = "f32")
    plan <- fastEmbedR:::massive_hnsw_sharded_resources
    expect_identical(plan(source, 30L, NULL, NULL, "20GB")$n_shards, 1L)
    expect_gt(plan(source, 30L, NULL, NULL, "10GB")$n_shards, 1L)
    expect_gt(plan(source, 30L, NULL, 500000L, "20GB")$n_shards, 1L)
})

test_that("file HNSW sends recall and an explicit build degree to faissR", {
    source <- list(nrow = 10000000L, ncol = 96L, format = "f32")
    testthat::local_mocked_bindings(
        faissr_knn_select_cpp = function(request) {
            expect_identical(request$target_recall, 0.95)
            expect_identical(request$hnsw_build_m, 128L)
            list(resources = list(n_shards = 4L))
        }, .package = "fastEmbedR")
    plan <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 30L, NULL, NULL, "22GB", 0.95,
        list(m = 128L, ef_construction = 300L))
    expect_identical(plan$n_shards, 4L)
})

test_that("sharded HNSW tunes for output k, not the self candidate", {
    set.seed(215)
    reference <- matrix(rnorm(90L * 6L), ncol = 6L)
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(path))
    writeBin(as.vector(t(reference)), path, size = 4L,
        endian = "little")
    source <- massive_matrix(path, 90L, 6L, format = "f32")
    setup <- list(k = 30L, workers = 1L,
        resources = list(n_shards = 1L), method = "hnsw_sharded")
    observed <- new.env(parent = emptyenv())
    testthat::local_mocked_bindings(
        massive_knn_reference = function(reference, k, method,
                n.cores, backend, hnsw_build = NULL,
                cuda_provider = "faissR", target_recall = 0.99,
                policy_k = k) {
            observed$k <- k
            observed$policy_k <- policy_k
            observed$method <- method
            observed$hnsw_build <- hnsw_build
            stop("captured index build")
        }, .package = "fastEmbedR")
    expect_error(fastEmbedR:::massive_index_shard(
        source, setup, NULL, 0L, NULL, "cpu", NULL),
        "captured index build")
    expect_identical(observed$k, 30L)
    expect_identical(observed$policy_k, 30L)
    expect_identical(observed$method, "faiss_hnsw")
    expect_null(observed$hnsw_build)
})

test_that("the next CPU shard starts after the previous index is finalized", {
    released <- 0L
    finalize_index <- function(index) released <<- released + 1L
    testthat::local_mocked_bindings(
        massive_matrix_rows = function(...) NULL,
        massive_index_shard_state = function(x, reference, setup, shard, ...) {
            expect_identical(released, shard)
            pointer <- new.env()
            reg.finalizer(pointer, finalize_index)
            list(state = list(index = list(pointer = pointer),
                build_seconds = 0))
        },
        massive_sharded_probe = function(...) NULL,
        massive_checkpoint_source_identity = function(...) "stable",
        massive_sharded_batch = function(...) list(rows = 128L,
            timings = c(search = 0, write = 0)),
        .package = "fastEmbedR")
    setup <- list(k = 5L, method = "hnsw_sharded",
        resources = list(n_shards = 2L))
    for (shard in 0:1) {
        fastEmbedR:::massive_index_shard(list(nrow = 128L), setup,
            NULL, shard, "stable", "cpu", NULL)
    }
    expect_identical(released, 2L)
})

test_that("massive CPU graph honors explicit HNSW search effort", {
    set.seed(213)
    x <- matrix(rnorm(80L * 6L), ncol = 6L)
    path <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    paths <- paste0(prefix, c(".indices.u32", ".distances.f32"))
    on.exit(unlink(c(path, paths)))
    writeBin(as.vector(t(x)), path, size = 4L, endian = "little")
    source <- massive_matrix(path, 80L, 6L, format = "f32")
    graph <- massive_full_knn_graph(source, 5L, prefix,
        backend = "cpu", method = "hnsw_sharded", ef_search = 80L,
        n.cores = 2L, reference_chunk_rows = 80L,
        memory_limit = "256MB", audit_rows = 8L)
    expect_identical(graph$method, "faissr_cpu_hnsw_sharded")
    expect_equal(graph$target_recall, 0.99)
    expect_false(graph$spanning_links)
    expect_identical(graph$ef_search_requested, 80L)
    expect_true(graph$sample_target_met)
    expect_equal(graph$minimum_row_recall, 1)
    expect_error(massive_full_knn_graph(source, 5L, tempfile(),
        method = "hnsw_sharded", ef_search = 5L), "ef_search")
    expect_error(massive_full_knn_graph(source, 5L, tempfile(),
        method = "hnsw_sharded", ef_search = 81L,
        reference_chunk_rows = 80L), "smallest reference shard")
    expect_error(massive_full_knn_graph(source, 5L, tempfile(),
        method = "exact", ef_search = 80L), "requires CPU HNSW")
})

test_that("sharded CPU k30 default retains recall with shared tuning", {
    set.seed(216)
    reference <- matrix(rnorm(90L * 6L), ncol = 6L)
    path <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    small_prefix <- tempfile()
    on.exit(unlink(c(path, paste0(rep(c(prefix, small_prefix),
        each = 2L), c(".indices.u32", ".distances.f32")))))
    writeBin(as.vector(t(reference)), path, size = 4L,
        endian = "little")
    source <- massive_matrix(path, 90L, 6L, format = "f32")
    graph <- massive_full_knn_graph(source, 30L, prefix,
        method = "hnsw_sharded", reference_chunk_rows = 90L,
        memory_limit = "256MB", audit_rows = 8L)
    expect_identical(graph$method, "faissr_cpu_hnsw_sharded")
    expect_equal(graph$target_recall, 0.99)
    expect_true(graph$sample_target_met)
    small <- massive_full_knn_graph(source, 30L, small_prefix,
        method = "hnsw_sharded", reference_chunk_rows = 90L,
        chunk_rows = 7L, memory_limit = "256MB")
    read_graph <- function(item) fastEmbedR:::massive_read_knn_rows_cpp(
        item$indices_path, item$distances_path, item$n_vertices,
        item$k, item$n_vertices, 1L, 90L, 128e6)
    expect_equal(read_graph(graph), read_graph(small))
})

test_that("recall reporting retains a measured miss after explicit opt-in", {
    pilot <- list(recall_audited = TRUE, target_met = FALSE,
        recall_status = "pilot_exact_reference_missed_guard",
        pilot_recall = 0.97, target_recall = 0.99)
    check <- fastEmbedR:::faissr_hnsw_recall_check
    old <- options(faissR.knn.report_recall_miss = FALSE)
    on.exit(options(old))
    expect_error(check(pilot, "euclidean"), "did not meet the recall")
    options(faissR.knn.report_recall_miss = TRUE)
    expect_identical(check(pilot, "euclidean"), pilot)
    pilot$recall_audited <- FALSE
    expect_error(check(pilot, "euclidean"), "did not meet the recall")
})
