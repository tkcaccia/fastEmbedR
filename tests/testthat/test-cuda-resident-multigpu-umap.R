test_that("resident UMAP uses coordinated CUDA for both public APIs", {
    n <- 8L
    indices <- cbind(c(2:n, 1L), c(3:n, 1L, 2L))
    distances <- matrix(rep(c(0.2, 0.4), n), nrow = n)
    initial <- matrix(seq_len(2L * n) / 10000, nrow = n)
    calls <- 0L
    with_mocked_bindings({
        fit <- umap_knn(indices, distances, backend = "cuda",
            devices = c(0L, 1L))
        cfg <- attr(fit, "fastEmbedR_config")
        expect_identical(calls, 1L)
        expect_identical(cfg$optimizer,
            "cuda_resident_umap_coordinated_nccl")
        expect_identical(cfg$gpu_devices, c(0L, 1L))
        expect_identical(cfg$gpu_identity, c("GPU-A", "GPU-B"))
        expect_lt(max(abs(as.vector(as.numeric(fit)) -
            as.vector(initial))), 1e-6)
        one <- umap(list(indices = indices, distances = distances),
            backend = "cuda", devices = c(0L, 1L))
        expect_identical(calls, 2L)
        expect_identical(one$parameters$gpu_devices, c(0L, 1L))
    },
    massive_nccl_available_cpp = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...)
        c("GPU-A", "GPU-B"),
    spectral_knn_init = function(...) initial,
    umap_cuda_optimize_csr_multigpu_cpp = function(...) {
        args <- list(...)
        calls <<- calls + 1L
        list(Y = args[[4L]],
            optimizer = "cuda_resident_umap_coordinated_nccl",
            gpu_devices = args[[11L]],
            per_device_edge_visits = c(10, 10),
            delta_scale = 256,
            synchronization_seconds = 0.01)
    }, .package = "fastEmbedR")
})

test_that("resident UMAP materializes native GPU KNN in host RAM", {
    n <- 8L
    indices <- cbind(c(2:n, 1L), c(3:n, 1L, 2L))
    distances <- matrix(rep(c(0.2, 0.4), n), nrow = n)
    initial <- matrix(seq_len(2L * n) / 10000, nrow = n)
    fake <- structure(list(), class = "fake_gpu_knn")
    conversions <- 0L
    with_mocked_bindings({
        fit <- umap_knn(fake, backend = "cuda",
            devices = c(0L, 1L))
        cfg <- attr(fit, "fastEmbedR_config")
        expect_identical(conversions, 1L)
        expect_true(cfg$knn_materialized_for_gpu)
        expect_identical(cfg$optimizer,
            "cuda_resident_umap_coordinated_nccl")
    },
    fastembedr_is_gpu_knn = function(x)
        inherits(x, "fake_gpu_knn"),
    fastembedr_gpu_knn_info = function(x)
        list(n = n, k = 2L),
    fastembedr_gpu_knn_to_host = function(x) {
        conversions <<- conversions + 1L
        list(indices = indices, distances = distances)
    },
    massive_nccl_available_cpp = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...)
        c("GPU-A", "GPU-B"),
    spectral_knn_init = function(...) initial,
    umap_cuda_optimize_csr_multigpu_cpp = function(...) {
        args <- list(...)
        list(Y = args[[4L]],
            optimizer = "cuda_resident_umap_coordinated_nccl",
            gpu_devices = args[[11L]],
            per_device_edge_visits = c(10, 10),
            delta_scale = 256,
            synchronization_seconds = 0.01)
    }, .package = "fastEmbedR")
})

test_that("resident UMAP devices reject a CPU backend", {
    indices <- cbind(c(2:8, 1L), c(3:8, 1L, 2L))
    distances <- matrix(0.2, 8L, 2L)
    expect_error(umap_knn(indices, distances, backend = "cpu",
        devices = c(0L, 1L)), "requires CUDA")
})

test_that("coordinated CUDA receives wide binary 3D graph types", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    distances <- matrix(rep(seq_len(width), each = rows),
        rows, width)
    prepared <- prepare_umap_knn(indices, distances,
        graph_mode = "binary", backend = "cpu")
    prepared$graph <- fastEmbedR:::umap_build_csr_graph(
        indices, NULL, 0L, width, width, 1L,
        graph_mode = "binary", allow_wide = TRUE,
        force_wide = TRUE)
    prepared$config$csr_offset_dtype <- "uint64"
    prepared$config$graph_offset_dtype <- "uint64"
    observed <- FALSE
    with_mocked_bindings({
        fit <- umap_knn(prepared, n_components = 3L,
            backend = "cuda", devices = c(0L, 1L))
        expect_true(observed)
        expect_identical(dim(fit), c(rows, 3L))
        expect_s4_class(fit, "float32")
        expect_identical(attr(fit,
            "fastEmbedR_config")$csr_offset_dtype, "uint64")
    },
    massive_nccl_available_cpp = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...)
        c("GPU-A", "GPU-B"),
    umap_cuda_optimize_csr_multigpu_cpp = function(...) {
        args <- list(...)
        observed <<- is.double(args[[1L]]) &&
            inherits(args[[3L]], "float32")
        list(Y = float::fl(args[[4L]]),
            optimizer = "cuda_resident_umap_coordinated_nccl",
            gpu_devices = args[[11L]],
            per_device_edge_visits = c(10, 10),
            delta_scale = 256,
            synchronization_seconds = 0.01)
    }, .package = "fastEmbedR")
})

test_that("wide prepared CUDA selects the current GPU automatically", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    distances <- matrix(rep(seq_len(width), each = rows),
        rows, width)
    prepared <- prepare_umap_knn(indices, distances,
        graph_mode = "binary", backend = "cpu")
    prepared$graph <- fastEmbedR:::umap_build_csr_graph(
        indices, NULL, 0L, width, width, 1L,
        graph_mode = "binary", allow_wide = TRUE,
        force_wide = TRUE)
    initial <- matrix(seq_len(rows * 3L) / 10000,
        rows, 3L)
    called <- FALSE
    with_mocked_bindings({
        fit <- umap_knn(prepared, n_components = 3L,
            backend = "cuda")
        cfg <- attr(fit, "fastEmbedR_config")
        expect_true(called)
        expect_identical(cfg$csr_offset_dtype, "uint64")
        expect_identical(cfg$gpu_devices, 1L)
        expect_identical(cfg$gpu_identity, "GPU-X")
        expect_s4_class(fit, "float32")
    },
    massive_cuda_memory_cpp = function() list(device = 1L),
    massive_nccl_available_cpp = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...) "GPU-X",
    umap_cuda_optimize_csr_multigpu_cpp = function(...) {
        args <- list(...)
        called <<- is.double(args[[1L]]) &&
            inherits(args[[3L]], "float32") &&
            identical(args[[11L]], 1L)
        list(Y = float::fl(initial),
            optimizer = "cuda_resident_umap_coordinated_nccl",
            gpu_devices = 1L,
            per_device_edge_visits = 10,
            delta_scale = 256,
            synchronization_seconds = 0.01)
    }, .package = "fastEmbedR")
})

test_that("CUDA wide dispatch checks large edge bounds without allocation", {
    small <- structure(list(n = 1000000L, k = 30L),
        class = "fake_gpu_knn")
    large <- structure(list(n = 50000000L, k = 30L),
        class = "fake_gpu_knn")
    with_mocked_bindings({
        expect_false(fastEmbedR:::umap_cuda_wide_needed(small, NULL))
        expect_true(fastEmbedR:::umap_cuda_wide_needed(large, NULL))
    },
    fastembedr_is_gpu_knn = function(x)
        inherits(x, "fake_gpu_knn"),
    fastembedr_gpu_knn_info = function(x) x,
    .package = "fastEmbedR")
})
