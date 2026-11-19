test_that("file-backed Metal IVF completes one native graph", {
    skip_if_not(fastEmbedR:::fastembedr_metal_knn_available(),
        "Metal is required")
    set.seed(42)
    x <- matrix(rnorm(4096L * 3L), ncol = 3L)
    directory <- tempfile("metal-file-ivf-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE), add = TRUE)
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(t(x)), input, size = 4L, endian = "little")
    source <- massive_matrix(input, nrow(x), ncol(x))
    expect_error(massive_full_knn_graph(source, 5L,
        file.path(directory, "too-small"), backend = "metal",
        method = "metal_ivf", memory_limit = "1MB"),
        "budget")
    graph <- massive_full_knn_graph(source, 5L,
        file.path(directory, "graph"), backend = "metal",
        method = "metal_ivf", memory_limit = "512MB",
        target_recall = 0.9, audit_rows = 16L)
    expect_identical(graph$method, "faissr_native_metal_ivf_file")
    expect_identical(graph$index_residency,
        "transient_metal_buffers")
    expect_false(graph$exact)
    expect_true(graph$recall_audited)
    expect_gte(graph$observed_recall, 0.9)
    expect_gt(graph$resources$observed_metal_owned_buffer_peak_bytes, 0)
    expect_equal(file.info(graph$indices_path)$size,
        4096 * 5 * 4)
    expect_equal(file.info(graph$distances_path)$size,
        4096 * 5 * 4)
    automatic <- precompute_knn(source, k = 5L,
        backend = "metal", output = file.path(directory, "automatic"),
        memory_limit = "512MB", target_recall = 0.9)
    expect_identical(automatic$method,
        "faissr_native_metal_ivf_file")
    bounded <- precompute_knn(source, k = 5L,
        backend = "metal", output = file.path(directory, "bounded"),
        memory_limit = "256MB", target_recall = 0.9)
    expect_identical(bounded$method,
        "faissr_native_metal_exact_file_batch")
})
