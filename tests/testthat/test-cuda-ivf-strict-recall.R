test_that("CUDA IVF pilot guards the strict sampled ID recall", {
    skip_if_not(fastEmbedR:::native_cuda_knn_available_cpp())
    root <- tempfile("cuda_ivf_strict_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    set.seed(24L)
    data <- matrix(rnorm(2000L * 16L), ncol = 16L)
    input <- file.path(root, "input.f32")
    writeBin(as.vector(t(data)), input, size = 4L)
    source <- massive_matrix(input, 2000L, 16L)
    graph <- massive_full_knn_graph(source, 30L,
        file.path(root, "graph"), backend = "cuda",
        method = "ivf_sharded", chunk_rows = 256L,
        reference_chunk_rows = 2000L, memory_limit = "1GB",
        audit_rows = 32L)
    setting <- graph$ivf_settings[[1L]]
    expect_gte(setting$pilot_min_row_recall, 0.99)
    expect_gte(setting$pilot_id_recall, setting$pilot_min_row_recall)
    expect_true(graph$sample_target_met)
})
