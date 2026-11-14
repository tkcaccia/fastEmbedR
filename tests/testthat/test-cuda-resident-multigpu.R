test_that("resident t-SNE selects the coordinated CUDA optimizer", {
    n <- 8L
    indices <- cbind(c(2:n, 1L), c(3:n, 1L, 2L))
    distances <- matrix(rep(c(0.2, 0.4), n), nrow = n)
    initial <- matrix(seq_len(2L * n) / 10000, nrow = n)
    calls <- 0L
    with_mocked_bindings({
        fit <- tsne_knn(indices, distances, n_neighbors = 2L,
            perplexity = 2, Y_init = initial, backend = "cuda",
            devices = c(0L, 1L), early_exaggeration_iter = 1L,
            n_iter = 1L, auto_config = FALSE)
        cfg <- attr(fit, "fastEmbedR_config")
        expect_identical(calls, 1L)
        expect_identical(cfg$optimizer,
            "cuda_resident_coordinated_nccl")
        expect_identical(cfg$gpu_devices, c(0L, 1L))
        expect_identical(cfg$gpu_identity, c("GPU-A", "GPU-B"))
        expect_identical(cfg$affinity_upload_mode,
            "pinned_pipeline_cuda_shards")
        expect_identical(cfg$cuda_stage_seconds$knn_shard_upload, 0.2)
        expect_true(all(cfg$per_device_attraction_edges > 0L))
        expect_identical(cfg$graph_cache$graph_cache_mode,
            "resident_device_shards")
        expect_equal(as.vector(fit), as.vector(initial))
        one_call <- tsne(list(indices = indices,
            distances = distances), perplexity = 2,
            Y_init = initial, backend = "cuda", devices = c(0L, 1L),
            early_exaggeration_iter = 1L, n_iter = 1L,
            auto_config = FALSE)
        expect_identical(calls, 2L)
        expect_identical(one_call$parameters$gpu_devices,
            c(0L, 1L))
        expect_identical(one_call$parameters$graph_cache$graph_cache_bytes,
            192)
    },
    embedding_cuda_available_cpp = function() TRUE,
    cuda_opentsne_native_available = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...)
        c("GPU-A", "GPU-B"),
    knn_tsne_opentsne_cuda_multigpu_cpp = function(...) {
        args <- list(...)
        calls <<- calls + 1L
        list(Y = args[[3L]],
            optimizer = "cuda_resident_coordinated_nccl",
            repulsion = "fft_grid", precision = "float32",
            n_threads = 1L, gpu_devices = args[[17L]],
            per_device_attraction_edges = c(8L, 8L),
            affinity_upload_mode = "pinned_pipeline_cuda_shards",
            cuda_stage_seconds = list(knn_shard_upload = 0.2),
            graph_cache_mode = "resident_device_shards",
            graph_cache_bytes = 192)
    }, .package = "fastEmbedR")
})

test_that("resident t-SNE device requests reject a CPU backend", {
    indices <- cbind(c(2:8, 1L), c(3:8, 1L, 2L))
    distances <- matrix(0.2, 8L, 2L)
    expect_error(tsne_knn(indices, distances, n_neighbors = 2L,
        perplexity = 2, backend = "cpu", devices = c(0L, 1L),
        early_exaggeration_iter = 1L, n_iter = 1L,
        auto_config = FALSE), "requires CUDA")
})
