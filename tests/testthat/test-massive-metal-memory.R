test_that("file-backed Metal retains faissR device memory metadata", {
    setup <- list(k = 5L, backend = "metal", workers = 1L,
        devices = NULL, resources = list(reference_chunk_rows = 64L,
            memory_limit_bytes = 256 * 1024^2))
    stats <- new.env(parent = emptyenv())
    stats$build <- stats$search <- stats$peak_device <- 0
    stats$metal_owned_peak <- 0
    stats$device_model <- "none"
    found <- list(device_ids = 0L,
        device_memory_model = "unified_shared_estimate",
        peak_device_bytes = 4096, index_build_seconds = 0.01,
        search_seconds = 0.02,
        observed_metal_owned_live_buffer_bytes = 1024)
    testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_batch(
            NULL, 1L, 16L, setup, stats),
        faissr_matrix_knn_batch_cpp = function(...) found,
        .package = "fastEmbedR")
    expect_equal(stats$peak_device, 4096)
    expect_equal(stats$metal_owned_peak, 1024)
    expect_equal(stats$build, 0.01)
    expect_equal(stats$search, 0.02)
    expect_identical(stats$device_model, "unified_shared_estimate")
    graph <- testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_graph(list(nrow = 16L),
            c(setup, list(output = "unused", backend_used = "metal_exact",
                target_recall = 0.99)), list(seconds = 0.1),
            stats, FALSE, FALSE, 100L),
        massive_knn_graph = function(...) list(),
        massive_knn_residency = function(...) "file-backed",
        massive_checkpoint_source_identity = function(...) "input",
        massive_checkpoint_implementation_identity = function(...) "impl",
        massive_faissr_identity = function(...) "faissR",
        .package = "fastEmbedR")
    expect_equal(graph$resources$observed_metal_owned_buffer_peak_bytes,
        1024)
    found$observed_metal_owned_live_buffer_bytes <- NULL
    expect_error(testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_batch(
            NULL, 1L, 16L, setup, stats),
        faissr_matrix_knn_batch_cpp = function(...) found,
        .package = "fastEmbedR"), "omitted device memory")
    found$observed_metal_owned_live_buffer_bytes <- 1024
    found$peak_device_bytes <- 0
    expect_error(testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_batch(
            NULL, 1L, 16L, setup, stats),
        faissr_matrix_knn_batch_cpp = function(...) found,
        .package = "fastEmbedR"), "omitted device memory")
})
