test_that("CUDA file CAGRA searches disjoint reference shards", {
    skip_if_not_installed("faissR")
    skip_if_not(faissR::cuvs_available())
    skip_if_not(fastEmbedR:::massive_cuda_device_count_cpp() >= 2L)

    old <- options(fastEmbedR.cuda_knn_method = "cagra_width4")
    on.exit(options(old), add = TRUE)
    dir <- tempfile("cagra-shards-")
    dir.create(dir)
    on.exit(unlink(dir, recursive = TRUE), add = TRUE)
    rows <- 500001L
    set.seed(177)
    x <- matrix(rnorm(4L * rows), rows)
    path <- file.path(dir, "source.f32")
    writeBin(as.double(t(x)), path, size = 4L, endian = "little")
    data <- massive_matrix(path, rows, 4L)
    graph <- precompute_knn(data, k = 8L, backend = "cuda",
        devices = 0:1, output = file.path(dir, "neighbors"),
        n.cores = 1L, audit_rows = 16L, target_recall = 0.99)
    expect_identical(graph$method, "cuda_cuvs_cagra_file_multi_sharded")
    expect_identical(graph$device_ids, 0:1)
    expect_equal(graph$cagra_settings$shard_rows, c(250001, 250000))
    expect_identical(graph$knn_provider, "faissR")
    expect_true(graph$validated)
    expect_true(is.finite(graph$observed_recall))
})
