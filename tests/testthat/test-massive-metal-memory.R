test_that("file-backed Metal retains faissR device memory metadata", {
    setup <- list(k = 5L, backend = "metal", workers = 1L,
        devices = NULL, resources = list(reference_chunk_rows = 64L,
            memory_limit_bytes = 256 * 1024^2))
    stats <- new.env(parent = emptyenv())
    stats$build <- stats$search <- stats$peak_device <- 0
    stats$device_model <- "none"
    found <- list(device_ids = 0L,
        device_memory_model = "unified_shared_estimate",
        peak_device_bytes = 4096, index_build_seconds = 0.01,
        search_seconds = 0.02)
    testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_batch(
            NULL, 1L, 16L, setup, stats),
        faissr_matrix_knn_batch_cpp = function(...) found,
        .package = "fastEmbedR")
    expect_equal(stats$peak_device, 4096)
    expect_equal(stats$build, 0.01)
    expect_equal(stats$search, 0.02)
    expect_identical(stats$device_model, "unified_shared_estimate")
    found$peak_device_bytes <- 0
    expect_error(testthat::with_mocked_bindings(
        fastEmbedR:::massive_full_exact_batch(
            NULL, 1L, 16L, setup, stats),
        faissr_matrix_knn_batch_cpp = function(...) found,
        .package = "fastEmbedR"), "omitted device memory")
})
