test_that("persistent faissR HNSW reports the actual search", {
    set.seed(244)
    reference <- matrix(rnorm(90L * 6L), ncol = 6L)
    data <- float::fl(reference)
    index <- fastEmbedR:::faissr_hnsw_index_build_cpp(
        data, 90L, 6L, 5L, 2L)
    expected <- faissR:::nn_tune_faiss_hnsw_cpp(
        90L, 6L, 5L, "euclidean", 0.99)
    expect_identical(index$method, "faissr_hnsw_persistent")
    expect_identical(index$M, expected$m)
    expect_identical(index$efConstruction, expected$ef_construction)
    expect_identical(index$efSearch, expected$ef_search)
    expect_identical(index$tuning_rule, expected$rule)
    expect_true(inherits(index$pointer, "faissR_faiss_hnsw_index"))
    calibrated <- fastEmbedR:::faissr_hnsw_index_build_cpp(
        data, 90L, 6L, 5L, 2L, 16L, 80L)
    expect_identical(calibrated$M, 16L)
    expect_identical(calibrated$efConstruction, 80L)
    expect_identical(calibrated$tuning_rule, "explicit")
    expect_error(fastEmbedR:::faissr_hnsw_index_build_cpp(
        data, 90L, 6L, 5L, 2L, 16L, 0L), "together")
    found <- fastEmbedR:::faissr_hnsw_index_search_cpp(
        index$pointer, data[1:3, , drop = FALSE], 6L, 80L, 2L)
    expect_identical(found$backend, "cpu")
    expect_identical(found$method, "faissr_hnsw_query_reused")
    expect_true(found$index_reused)
    expect_identical(found$ef_search, 80L)
    expect_true(all(found$indices >= 1L & found$indices <= 90L))
    expect_true(all(is.finite(found$distances)))
    expect_error(fastEmbedR:::faissr_hnsw_index_search_cpp(
        new("externalptr"), data[1:2, , drop = FALSE],
        6L, 80L, 2L))
})

test_that("HNSW build timing includes reference staging", {
    source <- list(nrow = 4L, ncol = 2L)
    data <- float::fl(matrix(seq_len(8L), ncol = 2L))
    result <- with_mocked_bindings(
        fastEmbedR:::massive_faiss_hnsw_reference(source, 2L, 1L),
        massive_read_rows_cpp = function(...) {
            Sys.sleep(0.1)
            data
        },
        faissr_hnsw_index_build_cpp = function(...) list(),
        .package = "fastEmbedR")
    expect_gt(result$build_seconds, 0.07)
    expect_identical(result$reference_storage, "R_float32")
})

test_that("full CPU HNSW exposes faissR backend and audited recall", {
    set.seed(245)
    reference <- matrix(rnorm(90L * 6L), ncol = 6L)
    path <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    on.exit(unlink(c(path, paste0(prefix,
        c(".indices.u32", ".distances.f32")))))
    writeBin(as.vector(t(reference)), path, size = 4L,
        endian = "little")
    source <- massive_matrix(path, 90L, 6L, format = "f32")
    graph <- massive_full_knn_graph(source, 5L, prefix,
        method = "hnsw_sharded", n.cores = 2L,
        reference_chunk_rows = 90L, chunk_rows = 16L,
        ef_search = 80L,
        hnsw_build = list(m = 16L, ef_construction = 80L),
        memory_limit = "256MB", audit_rows = 8L)
    expect_identical(graph$backend, "cpu")
    expect_identical(graph$method, "faissr_cpu_hnsw_sharded")
    expect_false(graph$exact)
    expect_false(graph$spanning_links)
    expect_true(graph$recall_audited)
    expect_equal(graph$target_recall, 0.99)
    expect_identical(graph$hnsw_build_requested,
        list(m = 16L, ef_construction = 80L))
    expect_length(graph$hnsw_settings, 1L)
    expect_identical(graph$hnsw_settings[[1L]]$m, 16L)
    expect_identical(graph$hnsw_settings[[1L]]$ef_construction, 80L)
    expect_identical(graph$hnsw_settings[[1L]]$ef_search, 80L)
    expect_identical(graph$hnsw_settings[[1L]]$tuning_rule,
        "explicit")
    expect_true(all(graph$audit_row_recall >= 0 &
        graph$audit_row_recall <= 1))
    expect_error(massive_full_knn_graph(source, 5L, tempfile(),
        method = "exact", hnsw_build = list(
            m = 16L, ef_construction = 80L)), "requires CPU HNSW")
    expect_error(massive_full_knn_graph(source, 5L, tempfile(),
        method = "hnsw_sharded", hnsw_build = list(m = 16L)),
        "named")
})

test_that("HNSW checkpoint identity includes explicit build settings", {
    path <- tempfile(fileext = ".f32")
    output <- tempfile()
    on.exit(unlink(c(path, paste0(output, ".checkpoint.rds"))))
    writeBin(as.vector(t(matrix(seq_len(40L), ncol = 4L))),
        path, size = 4L, endian = "little")
    source <- massive_matrix(path, 10L, 4L, format = "f32")
    resources <- fastEmbedR:::massive_hnsw_sharded_resources(
        source, 3L, 4L, 10L, "256MB")
    paths <- fastEmbedR:::massive_knn_paths(output,
        resources$output_bytes)
    setup <- list(k = 3L, workers = 1L, resources = resources,
        method = "hnsw_sharded", ef_search = 80L,
        hnsw_build = list(m = 16L, ef_construction = 80L),
        checkpoint_every = 100L)
    fastEmbedR:::massive_sharded_checkpoint(source, setup,
        paths, "cpu", FALSE)
    setup$hnsw_build$m <- 24L
    expect_error(fastEmbedR:::massive_sharded_checkpoint(
        source, setup, paths, "cpu", TRUE), "checkpoint")
})
