test_that("persistent HNSW reports explicit search effort", {
    set.seed(211)
    reference <- matrix(rnorm(180L * 6L), ncol = 6L)
    query <- matrix(rnorm(24L * 6L), ncol = 6L)
    index <- fastEmbedR:::native_hnsw_index_build_cpp(
        reference, 5L, n_threads = 2L)
    search <- fastEmbedR:::native_hnsw_index_search_cpp
    default <- search(index$pointer, query, 5L, 2L)
    tuned <- search(index$pointer, query, 5L, 2L, 120L)
    expect_identical(tuned$backend, "cpu")
    expect_identical(tuned$method, "native_hnsw_query_reused")
    expect_true(index$spanning_links)
    expect_true(tuned$spanning_links)
    expect_identical(tuned$ef_search_used, 120L)
    expect_true(default$ef_search_used >= 5L)
    expect_identical(dim(tuned$indices), c(24L, 5L))
    expect_error(search(index$pointer, query, 5L, 2L, -1L),
        "Invalid persistent HNSW query")
    expect_error(search(index$pointer, query, 5L, 2L, 181L),
        "Invalid persistent HNSW query")
})

test_that("persistent HNSW reaches every reference at full effort", {
    set.seed(212)
    reference <- matrix(rnorm(300L * 20L), ncol = 20L)
    query <- matrix(rnorm(24L * 20L), ncol = 20L)
    index <- fastEmbedR:::native_hnsw_index_build_cpp(
        reference, 8L, n_threads = 2L)
    exact <- fastEmbedR:::native_exact_index_build_cpp(reference)
    observed <- fastEmbedR:::native_hnsw_index_search_cpp(
        index$pointer, query, 8L, n_threads = 2L,
        ef_search = nrow(reference))
    expected <- fastEmbedR:::native_exact_index_search_cpp(
        exact, query, 8L, n_threads = 2L)
    expect_true(observed$spanning_links)
    expect_identical(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-5)
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
    expect_identical(graph$method, "native_cpu_hnsw_sharded")
    expect_true(graph$spanning_links)
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
