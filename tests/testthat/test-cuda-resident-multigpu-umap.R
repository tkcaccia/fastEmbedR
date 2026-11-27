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
        expect_identical(cfg$graph_cache$graph_cache_mode,
            "resident_device_shards")
        expect_identical(cfg$graph_cache$graph_cache_bytes, 128)
        expect_lt(max(abs(as.vector(as.numeric(fit)) -
            as.vector(initial))), 1e-6)
        one <- umap(list(indices = indices, distances = distances),
            backend = "cuda", devices = c(0L, 1L))
        expect_identical(calls, 2L)
        expect_identical(one$parameters$gpu_devices, c(0L, 1L))
        expect_identical(one$parameters$graph_cache$graph_cache_mode,
            "resident_device_shards")
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
            graph_cache_mode = "resident_device_shards",
            graph_cache_bytes = 128,
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
        expect_identical(cfg$graph_cache$graph_cache_mode,
            "staged_batches")
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
            graph_cache_mode = "staged_batches",
            graph_cache_bytes = 0,
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

test_that("wide fuzzy CUDA initializes from the existing CSR graph", {
    skip_if_not_installed("float")
    rows <- 36L
    width <- 6L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + 2L * j - 1L) %% rows + 1L)
    distances <- float::fl(matrix(rep(seq_len(width), each = rows),
        rows, width))
    prepared <- prepare_umap_knn(indices, distances, backend = "cpu")
    prepared$graph <- fastEmbedR:::umap_build_csr_graph(
        indices, distances, 0L, width, width, 1L,
        allow_wide = TRUE, force_wide = TRUE)
    cfg <- fastEmbedR:::configure_prepared_umap(prepared, "cuda", 2L, 1L, NULL)
    expected <- fastEmbedR:::umap_init_from_csr_graph(
        prepared$graph, 2L, cfg, 42L)
    native_limit <- FALSE
    run <- function() {
        previous <- options(fastEmbedR.cuda_wide_csr_build = TRUE)
        on.exit(options(previous))
        umap_knn(prepared, backend = "cuda", n.cores = 1L,
            devices = c(0L, 1L))
    }
    with_mocked_bindings({
        fit <- run()
        meta <- attr(fit, "fastEmbedR_config")
        expect_identical(meta$init_backend, "cuda_wide_csr_diffusion")
        expect_identical(meta$init_resources$device_rows, c(18L, 18L))
        expect_false(meta$initialization_reuse_hit)

        native_limit <- TRUE
        expect_error(run(), "initialization exceeds the device budget")
    },
    massive_nccl_available_cpp = function() TRUE,
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
    spectral_knn_init = function(...) stop("KNN affinity rebuild"),
    umap_cuda_init_csr_wide_cpp = function(offsets, neighbors, weights,
            iterations, seed, devices, device_limit) {
        expect_identical(neighbors, prepared$graph$neighbors)
        expect_identical(weights, prepared$graph$weights)
        expect_identical(devices, c(0L, 1L))
        if (!native_limit)
            list(layout = expected, device_rows = c(18L, 18L))
    },
    umap_cuda_optimize_csr_multigpu_cpp = function(...) {
        args <- list(...)
        expect_identical(args[[4L]], expected)
        expect_identical(args[[11L]], c(0L, 1L))
        list(Y = args[[4L]], optimizer = "cuda_resident_umap_coordinated_nccl",
            gpu_devices = args[[11L]], per_device_edge_visits = c(10, 10),
            delta_scale = 256, synchronization_seconds = 0.01)
    }, .package = "fastEmbedR")
})

test_that("native CUDA wide diffusion matches the CPU initialization", {
    skip_if_not(fastEmbedR:::massive_nccl_available_cpp())
    rows <- 257L
    indices <- outer(seq_len(rows), seq_len(4L), function(i, j) {
        (i + j - 1L) %% rows + 1L
    })
    distances <- float::fl(matrix(rep(seq_len(4L), each = rows), rows))
    graph <- fastEmbedR:::umap_build_csr_graph(
        indices, distances, 0L, 4L, 4L, 1L,
        allow_wide = TRUE, force_wide = TRUE)
    expected <- fastEmbedR:::fast_knn_umap_csr_wide_diffusion_cpp(
        graph$offsets, graph$neighbors, graph$weights, 20L, 1L, 42L)
    cards <- min(2L, fastEmbedR:::massive_cuda_device_count_cpp())
    skip_if(cards < 1L)
    devices <- seq_len(cards) - 1L
    actual <- fastEmbedR:::umap_cuda_init_csr_wide_cpp(
        graph$offsets, graph$neighbors, graph$weights, 20L, 42L,
        devices, Inf)
    expect_equal(actual$layout, unname(as.matrix(expected)),
        tolerance = 1e-6, ignore_attr = TRUE)
    expect_equal(sum(actual$device_rows), rows)
    expect_true(all(actual$device_rows > 0L))
    expect_equal(sum(actual$device_edges), graph$nnz)
    expect_null(fastEmbedR:::umap_cuda_init_csr_wide_cpp(
        graph$offsets, graph$neighbors, graph$weights, 20L, 42L,
        devices, 64))
})

test_that("retained CUDA layout matches exported diffusion and scale", {
    skip_if_not(fastEmbedR:::massive_nccl_available_cpp())
    cards <- min(2L, fastEmbedR:::massive_cuda_device_count_cpp())
    skip_if(cards < 1L)
    n <- 257L
    k <- 4L
    devices <- seq_len(cards) - 1L
    ids <- outer(seq_len(n), seq_len(k), function(i, j)
        (i + j - 1L) %% n + 1L)
    distances <- float::fl(matrix(rep(seq_len(k), each = n), n))
    graph <- fastEmbedR:::umap_build_csr_graph(ids, distances, 0L, k, k,
        1L, allow_wide = TRUE, force_wide = TRUE)
    for (iterations in c(3L, 4L)) {
        initial <- fastEmbedR:::umap_cuda_init_csr_wide_cpp(
            graph$offsets, graph$neighbors, graph$weights, iterations,
            42L, devices, Inf)$layout
        for (target in c(NA_real_, 5)) {
            # Epoch zero clamps coordinates without applying edge updates.
            fit <- fastEmbedR:::umap_cuda_optimize_csr_multigpu_cpp(
                graph$offsets, graph$neighbors, graph$weights, NULL,
                1L, 5L, 1, 0.1, 1, 42L, devices, iterations, target, Inf)
            expected <- if (is.na(target)) initial else scale(initial) * target
            expected <- pmin(pmax(expected, -16), 16)
            expect_true(fit$initialization_resources$layout_reused_on_device)
            expect_equal(as.matrix(float::dbl(fit$Y)), expected,
                tolerance = 2e-6, ignore_attr = TRUE)
        }
    }
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

test_that("CUDA wide UMAP keeps initializer CSR for the optimizer", {
    skip_if_not(fastEmbedR:::massive_nccl_available_cpp())
    skip_if(fastEmbedR:::massive_cuda_device_count_cpp() < 1L)
    skip_if_not_installed("float")
    n <- 257L
    k <- 4L
    indices <- outer(seq_len(n), seq_len(k), function(i, j)
        (i + j - 1L) %% n + 1L)
    distances <- float::fl(matrix(rep(seq_len(k), each = n), n))
    prepared <- prepare_umap_knn(indices, distances, backend = "cpu")
    prepared$graph <- fastEmbedR:::umap_build_csr_graph(
        indices, distances, 0L, k, k, 1L,
        allow_wide = TRUE, force_wide = TRUE)
    run <- function(retain, limit = "available", build = TRUE) {
        previous <- options(fastEmbedR.cuda_wide_csr_reuse = retain,
            fastEmbedR.cuda_wide_csr_build = build, memory_limit = limit)
        on.exit(options(previous))
        umap_knn(prepared, backend = "cuda", devices = 0L, n.cores = 1L,
            seed = 44L)
    }
    baseline <- run(FALSE)
    retained <- run(TRUE)
    a <- attr(baseline, "fastEmbedR_config")
    b <- attr(retained, "fastEmbedR_config")
    expect_identical(b$optimizer, "cuda_resident_umap_coordinated")
    expect_identical(b$optimizer, a$optimizer)
    expect_identical(a$synchronization_seconds, 0)
    expect_identical(b$synchronization_seconds, 0)
    expect_identical(b$init_backend, "cuda_wide_csr_diffusion")
    expect_identical(b$graph_cache$graph_cache_mode, "retained_device_csr")
    expect_lt(b$graph_cache$graph_cache_bytes, a$graph_cache$graph_cache_bytes)
    expect_equal(b$per_device_edge_visits, a$per_device_edge_visits)
    expect_equal(b$init_resources$device_edges, prepared$graph$nnz)
    expect_equal(float::dbl(retained), float::dbl(baseline), tolerance = 1e-5)
    expect_error(run(TRUE, 1), "exceeds the device budget")
    expect_error(run(FALSE, 1, build = FALSE),
        "optimizer exceeds the device budget")
})

test_that("CUDA fuzzy row owners pass their graph straight to diffusion", {
    skip_if_not(fastEmbedR:::massive_nccl_available_cpp())
    skip_if(fastEmbedR:::massive_cuda_device_count_cpp() < 1L)
    skip_if_not_installed("float")
    n <- 257L
    ids <- outer(seq_len(n), c(0L, 1L, 2L, 2L, 4L), function(i, j)
        (i + j - 1L) %% n + 1L)
    d <- float::fl(matrix(rep(c(0, 1, 2, 3, 5), each = n), n))
    knn <- fastEmbedR:::coerce_knn_input(ids, d)
    prepared <- prepare_umap_knn(ids, d, backend = "cuda")
    prepared$graph <- fastEmbedR:::umap_build_csr_graph(
        knn$indices, knn$distances, knn$col_start, 4L, 4L, 1L,
        allow_wide = TRUE, force_wide = TRUE)
    run <- function(build, bytes = "available") {
        previous <- options(fastEmbedR.cuda_wide_csr_build = build,
            fastEmbedR.cuda_wide_csr_reuse = TRUE, memory_limit = bytes)
        on.exit(options(previous))
        input <- if (build) ids else prepared
        distances <- if (build) d else NULL
        fastEmbedR:::fast_knn_umap_core(input, distances, backend = "cuda",
            devices = 0L, n_threads = 1L, n_epochs = 5L, seed = 44L)
    }
    host <- run(FALSE)
    device <- run(TRUE)
    cfg <- attr(device, "fastEmbedR_config")
    expected <- attr(host, "fastEmbedR_config")
    expect_identical(cfg$graph_prep_backend, "cuda_fuzzy_csr_host_staged")
    expect_identical(cfg$init_backend, "cuda_wide_csr_diffusion")
    expect_identical(cfg$graph_cache$graph_cache_mode, "retained_device_csr")
    expect_equal(cfg$graph_nnz, prepared$graph$nnz)
    expect_gt(cfg$graph_cache$graph_cache_bytes,
        8 * (cfg$graph_nnz + n + 1L))
    expect_equal(cfg$graph_build$device_edges, prepared$graph$nnz)
    expect_equal(cfg$graph_build$cross_owner_records_via_host, 0)
    expect_equal(cfg$per_device_edge_visits, expected$per_device_edge_visits)
    expect_equal(float::dbl(device), float::dbl(host), tolerance = 1e-3)
    expect_equal(cfg$graph_build$host_record_upper_bytes, 12 * n * 4)
    expect_true(cfg$graph_build$outgoing_records_regenerated)
    expect_error(run(TRUE, 1),
        "Host-staged CUDA fuzzy graph exceeds the RAM budget",
        fixed = TRUE)
})
