test_that("file-backed Metal shards report estimated and observed memory", {
    source <- list(nrow = 16L, ncol = 3L)
    setup <- list(k = 5L, method = "exact", target_recall = 0.99,
        resources = list(metal_buffer_bytes_estimate = 4096))
    timings <- c(build = 0.01, load = 0, search = 0.02, write = 0.03,
        metal_peak = 1024)
    graph <- testthat::with_mocked_bindings(
        fastEmbedR:::massive_sharded_result(source,
            c(indices = "unused.indices.u32"), setup, "metal", "input",
            timings[["build"]], unname(proc.time()[[3L]]),
            0L, FALSE, FALSE, NULL, NULL, timings = timings),
        massive_knn_graph = function(...) list(),
        massive_knn_residency = function(...) "file-backed",
        massive_checkpoint_implementation_identity = function(...) "impl",
        massive_faissr_identity = function(...) "faissR",
        .package = "fastEmbedR")
    expect_identical(graph$method, "faissr_native_metal_flat_sharded")
    expect_identical(graph$knn_provider, "faissR")
    expect_identical(graph$device_ids, 0L)
    expect_equal(graph$resources$peak_device_bytes, 4096)
    expect_equal(graph$resources$observed_metal_owned_buffer_peak_bytes, 1024)
    expect_identical(graph$resources$device_memory_model,
        "unified_shared_estimate")
    expect_equal(graph$build_seconds, 0.01)
    expect_equal(graph$knn_search_seconds, 0.02)
    expect_equal(graph$graph_write_seconds, 0.03)
})
