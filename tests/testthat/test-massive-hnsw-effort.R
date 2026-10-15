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

test_that("persistent HNSW shares ordinary CPU tuning", {
    set.seed(214)
    reference <- matrix(rnorm(90L * 6L), ncol = 6L)
    ordinary <- fastEmbedR:::native_hnsw_knn_cpp(
        reference, 30L, 1L, "euclidean", 0.99)
    persistent <- fastEmbedR:::native_hnsw_index_build_cpp(
        reference, 30L, n_threads = 1L)
    for (name in c("tuning_rule", "tuning_shape_group",
            "tuning_k_bucket", "M", "efConstruction", "efSearch")) {
        expect_identical(persistent[[name]], ordinary[[name]])
    }
    query <- fastEmbedR:::native_hnsw_index_search_cpp(
        persistent$pointer, reference[1:2, , drop = FALSE], 30L, 1L)
    expect_identical(query$ef_search_used, persistent$efSearch)
})

test_that("persistent HNSW batch size leaves neighbors unchanged", {
    set.seed(217)
    reference <- matrix(rnorm(120L * 12L), ncol = 12L)
    index <- fastEmbedR:::native_hnsw_index_build_cpp(
        reference, 10L, n_threads = 2L)
    query <- reference[1:24, , drop = FALSE]
    all <- fastEmbedR:::native_hnsw_index_search_cpp(
        index$pointer, query, 11L, 2L, 60L)
    pieces <- lapply(seq.int(1L, 24L, by = 3L), function(first) {
        fastEmbedR:::native_hnsw_index_search_cpp(index$pointer,
            query[first:(first + 2L), , drop = FALSE],
            11L, 2L, 60L)
    })
    expect_identical(all$indices,
        do.call(rbind, lapply(pieces, `[[`, "indices")))
    expect_equal(all$distances,
        do.call(rbind, lapply(pieces, `[[`, "distances")))
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
                n.cores, backend, hnsw_build = NULL) {
            observed$k <- k
            observed$method <- method
            observed$hnsw_build <- hnsw_build
            stop("captured index build")
        }, .package = "fastEmbedR")
    expect_error(fastEmbedR:::massive_index_shard(
        source, setup, NULL, 0L, NULL, "cpu", NULL),
        "captured index build")
    expect_identical(observed$k, 30L)
    expect_identical(observed$method, "faiss_hnsw")
    expect_null(observed$hnsw_build)
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
