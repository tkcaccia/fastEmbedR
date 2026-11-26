test_that("UMAP and file KNN select the admitted CUDA reference shards", {
    devices <- c(1L, 3L)
    profile <- list(graph_degree = 32L, intermediate_degree = 64L,
        batch_rows = 512L, memory_limit_bytes = 32 * 2^30,
        builder = "nn_descent")
    request <- NULL
    native_request <- NULL
    available <- TRUE
    multi_selector <- TRUE
    testthat::local_mocked_bindings(cuvs_available = function() available,
        .package = "faissR")
    testthat::local_mocked_bindings(
        resolve_embedding_backend = identity,
        massive_available_ram_bytes = function() 40 * 2^30,
        massive_auto_cuda_free = function(selected, rows) {
            expect_identical(selected, devices)
            8 * 2^30
        },
        faissr_knn_select_cpp = function(spec) {
            request <<- spec
            enabled <- spec$allow_cuda_profile ||
                (multi_selector && !is.null(spec$cuda_device_count))
            list(backend = "cuda", method = if (enabled) "cagra_width4" else
                "ivf", target_recall = .99, cuda_cagra_profile = profile)
        },
        faissr_cuvs_multi_cagra_cpp = function(spec) {
            native_request <<- spec
            list(backend_used = "cuda_cuvs_multi_cagra", devices = devices,
                indices = matrix(1L, nrow(spec$data), spec$k),
                distances = matrix(1, nrow(spec$data), spec$k), exact = FALSE)
        },
        massive_validate_devices = function(selected, ...) selected,
        precompute_knn_file_cagra = function(data, k, output, budget,
                selection, audit_rows, selected, checkpoint, workers) {
            expect_identical(selected, devices)
            expect_identical(selection$cuda_cagra_profile, profile)
            expect_equal(budget, 32 * 2^30)
            list(method = "cuda_cuvs_cagra_file_multi_sharded")
        },
        .package = "fastEmbedR")
    old <- options(fastEmbedR.cuda_knn_method = "auto")
    on.exit(options(old), add = TRUE)
    fit <- fastEmbedR:::compute_umap_matrix_knn(matrix(0, 500000L, 2L),
        NULL, 1L, "euclidean", list(backend = "cuda", target_recall = .99,
            n_threads = 8L), devices)
    expect_identical(fit$engine, "cuda_cuvs_multi_cagra")
    expect_identical(native_request$devices, devices)
    expect_identical(native_request$merge_workers, 2L)
    expect_identical(native_request$batch_rows, 512L)
    expect_identical(native_request$memory_limit_bytes, 32 * 2^30)
    expect_true(native_request$nn_descent)
    expect_identical(request$cuda_device_count, 2L)
    expect_equal(request$cuda_free_bytes, 8 * 2^30)
    src <- list(nrow = 10000000L, ncol = 96L, format = "view",
        source = list(format = "fbin"))
    graph <- fastEmbedR:::precompute_knn_backed(src, 30L, "euclidean", "cuda",
        1L, "unused", "32GB", devices, FALSE, FALSE, 100L, .99, 0L)
    expect_identical(graph$method, "cuda_cuvs_cagra_file_multi_sharded")
    expect_true(request$file_backed)
    expect_false(request$allow_cuda_profile)
    expect_identical(request$cuda_device_count, 2L)
    multi_selector <- FALSE
    old_policy <- fastEmbedR:::fastembedr_embedding_nn_policy("cuda",
        10000000L, 96L, .99, cuda_devices = devices)
    expect_identical(old_policy$method, "ivf")
    multi_selector <- TRUE
    available <- FALSE
    policy <- fastEmbedR:::fastembedr_embedding_nn_policy("cuda", 10000000L,
        96L, .99, cuda_devices = devices)
    expect_identical(policy$method, "ivf")
    expect_false(request$allow_cuda_profile)
    expect_null(request$cuda_device_count)
})

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
