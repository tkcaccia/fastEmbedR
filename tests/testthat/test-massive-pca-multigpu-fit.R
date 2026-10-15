test_that("CUDA PCA shard covariance combines into one fit", {
    set.seed(4404)
    x <- matrix(rnorm(44L), ncol = 4L)
    shards <- list(x[1:5, , drop = FALSE],
        x[6:11, , drop = FALSE])
    for (center in c(TRUE, FALSE)) {
        parts <- lapply(seq_along(shards), function(i) {
            block <- shards[[i]]
            mean <- if (center) colMeans(block) else rep(0, 4L)
            adjusted <- sweep(block, 2L, mean)
            list(rows = as.double(nrow(block)), device = i - 1L,
                row_range = if (i == 1L) c(1, 5) else c(6, 11),
                center = mean,
                covariance = crossprod(adjusted) / (nrow(block) - 1),
                seconds = 1, estimated_ram_bytes = 2048,
                estimated_vram_bytes = 1024)
        })
        combined <- fastEmbedR:::massive_pca_combine_covariance(
            parts, center)
        expected <- if (center) stats::cov(x) else
            crossprod(x) / (nrow(x) - 1)
        expect_equal(combined$covariance, expected, tolerance = 1e-12)
        expect_equal(combined$center,
            if (center) colMeans(x) else rep(0, 4L))
        expect_identical(combined$fit_devices, c(0L, 1L))
        expect_equal(vapply(combined$fit_per_device, `[[`, 0,
            "rows"), c(5, 6))
    }
})

test_that("multi-GPU PCA rejects truncated fit shards", {
    source <- fastEmbedR:::massive_synthetic_matrix(8L, 4L)
    resources <- list(chunk_rows = 2L,
        memory_limit_bytes = 1e9, peak_ram_bytes = 1e8,
        gpu_devices = c(0L, 1L))
    with_mocked_bindings(
        {
            expect_error(fastEmbedR:::massive_pca_covariance_devices(
                source, 2L, TRUE, resources),
                "fit did not cover every row")
            expect_error(fastEmbedR:::massive_pca_wide_action_devices(
                source, 2L, numeric(4L), rep(1, 4L),
                matrix(0, 4L, 2L), resources),
                "sketch did not cover every row")
        },
        massive_cuda_parallel_tasks = function(tasks, worker,
                arguments) lapply(tasks, function(task) list(
            device = task$device,
            row_range = task$row_range,
            rows = as.double(diff(task$row_range)),
            center = numeric(4L), covariance = diag(4L),
            action = matrix(0, 4L, 2L))),
        .package = "fastEmbedR"
    )
})

test_that("multi-GPU PCA rejects swapped equal-sized fit shards", {
    source <- fastEmbedR:::massive_synthetic_matrix(8L, 4L)
    resources <- list(chunk_rows = 2L,
        memory_limit_bytes = 1e9, peak_ram_bytes = 1e8,
        gpu_devices = c(0L, 1L))
    with_mocked_bindings(
        {
            expect_error(fastEmbedR:::massive_pca_covariance_devices(
                source, 2L, TRUE, resources),
                "fit did not cover every row")
            expect_error(fastEmbedR:::massive_pca_wide_action_devices(
                source, 2L, numeric(4L), rep(1, 4L),
                matrix(0, 4L, 2L), resources),
                "sketch did not cover every row")
        },
        massive_cuda_parallel_tasks = function(tasks, worker,
                arguments) lapply(seq_along(tasks), function(i) list(
            device = tasks[[i]]$device,
            row_range = tasks[[3L - i]]$row_range,
            rows = 4, center = numeric(4L),
            covariance = diag(4L),
            action = matrix(0, 4L, 2L))),
        .package = "fastEmbedR"
    )
})

test_that("multi-GPU PCA reports the larger process-tree stage", {
    path <- tempfile(fileext = ".f32")
    on.exit(unlink(c(path, paste0(path, ".manifest.rds"))))
    writeBin(rep(0, 4L), path, size = 4L)
    source <- fastEmbedR:::massive_synthetic_matrix(4L, 2L)
    fit <- list(loadings = matrix(0, 2L, 1L),
        center = numeric(2L), scale = rep(1, 2L),
        fit_per_device = list(
            list(estimated_ram_bytes = 20),
            list(estimated_ram_bytes = 30)))
    resources <- list(peak_ram_bytes = 170,
        parent_peak_ram_bytes = 100,
        projection_peak_ram_bytes = 170,
        gpu_devices = c(0L, 1L))
    result <- with_mocked_bindings(
        fastEmbedR:::finish_massive_pca(fit, path,
            source, 1L, 1L, 4L, resources,
            proc.time()[[3L]], "cuda"),
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        .package = "fastEmbedR")
    expect_equal(result$resources$fit_peak_ram_bytes, 150)
    expect_equal(result$resources$peak_ram_bytes, 170)
    expect_identical(result$resources$peak_ram_scope,
        "process_tree_estimate")
    expect_identical(result$gpu_identity, c("GPU-A", "GPU-B"))
})

test_that("multi-GPU PCA fit routes statistics to distinct devices", {
    set.seed(4405)
    x <- matrix(rnorm(44L), ncol = 4L)
    raw <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(x)), raw, size = 4L,
        endian = "little")
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow(x), ncol(x))
    resources <- list(chunk_rows = 4L, memory_limit_bytes = 1e9,
        peak_ram_bytes = 1e8,
        gpu_devices = c(0L, 1L))
    with_mocked_bindings(
        {
            fit <- fastEmbedR:::massive_pca_decomposition(
                source, 2L, TRUE, FALSE, resources, 1L,
                "cuda", 4L)
            expect_identical(fit$fit_devices, c(0L, 1L))
            expect_identical(fit$fit_statistics_backend,
                "distributed_cuda_covariance")
            expect_identical(fit$decomposition_backend,
                "cpu_eigen")
            expect_true(is.finite(fit$fit_statistics_seconds))
            expect_equal(fit$singular_values,
                svd(scale(x, center = TRUE, scale = FALSE))$d[1:2],
                tolerance = 1e-6)
            scaled <- fastEmbedR:::massive_pca_decomposition(
                source, 2L, TRUE, TRUE, resources, 1L,
                "cuda", 4L)
            expect_equal(scaled$singular_values,
                stats::prcomp(x, scale. = TRUE)$sdev[1:2] *
                    sqrt(nrow(x) - 1), tolerance = 1e-6)
        },
        massive_cuda_parallel_tasks = function(tasks, worker,
                arguments) {
            expect_identical(vapply(tasks, `[[`, 0L, "device"),
                c(0L, 1L))
            expect_equal(arguments$memory_limit, 4.5e8)
            lapply(tasks, function(task) {
                rows <- seq.int(task$row_range[1L],
                    task$row_range[2L])
                block <- x[rows, , drop = FALSE]
                list(center = colMeans(block),
                    covariance = stats::cov(block),
                    rows = as.double(nrow(block)),
                    row_range = task$row_range,
                    device = task$device, seconds = 1,
                    estimated_ram_bytes = 2048,
                    estimated_vram_bytes = 1024)
            })
        },
        massive_pca_covariance_cuda_cpp = function(...) stop(
            "single-device covariance was used"),
        .package = "fastEmbedR"
    )
})

test_that("CUDA PCA fit worker reads only its assigned rows", {
    x <- matrix(seq_len(44L) / 10, ncol = 4L)
    raw <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(x)), raw, size = 4L,
        endian = "little")
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow(x), ncol(x))
    task <- list(device = 1L, row_range = c(6, 11))
    args <- list(source = source, rank = 2L, center = TRUE,
        chunk_rows = 3L, memory_limit = 1e8)
    worker <- fastEmbedR:::massive_pca_covariance_device_worker
    with_mocked_bindings(
        {
            result <- worker(task, args,
                normalizePath(find.package("fastEmbedR")),
                fastEmbedR:::
                    massive_checkpoint_implementation_identity())
            expect_identical(result$device, 1L)
            expect_equal(result$rows, 6)
            expect_identical(result$row_range, c(6, 11))
            expect_equal(result$center, colMeans(x[6:11, ,
                drop = FALSE]), tolerance = 1e-6)
        },
        massive_cuda_select_cpp = function(device) {
            expect_identical(device, 1L)
            device
        },
        massive_pca_resources = function(source, ...) list(
            gpu_device = 1L, chunk_rows = 3L,
            peak_ram_bytes = 2048,
            peak_vram_bytes = 1024),
        massive_pca_covariance_cuda_cpp = function(source, ...) {
            values <- massive_read_rows(source, 1L, source$nrow)
            list(center = colMeans(values),
                covariance = stats::cov(values))
        },
        .package = "fastEmbedR"
    )
})

test_that("multi-GPU PCA checkpoint records the distributed fit", {
    source <- fastEmbedR:::massive_synthetic_matrix(12L, 4L)
    output <- tempfile(fileext = ".f32")
    sidecar <- paste0(output, ".checkpoint.rds")
    on.exit(unlink(sidecar))
    resources <- list(chunk_rows = 4L, gpu_devices = c(0L, 1L))
    signature <- with_mocked_bindings(
        fastEmbedR:::massive_pca_checkpoint_signature(source,
            output, 2L, TRUE, FALSE, "cuda", 1L, 4L, resources),
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        .package = "fastEmbedR")
    remapped <- with_mocked_bindings(
        fastEmbedR:::massive_pca_checkpoint_signature(source,
            output, 2L, TRUE, FALSE, "cuda", 1L, 4L, resources),
        massive_cuda_device_uuids = function(...) c("GPU-B", "GPU-A"),
        .package = "fastEmbedR")
    expect_false(identical(signature, remapped))
    resources$gpu_devices <- 0L
    old <- fastEmbedR:::massive_pca_checkpoint_signature(
        source, output, 2L, TRUE, FALSE, "cuda", 1L, 4L,
        resources)
    expect_false(identical(signature, old))
    resources$gpu_devices <- c(0L, 1L)
    state <- list(signature = signature, stage = "pending",
        completed_rows = 0)
    with_mocked_bindings(
        {
            saved <- fastEmbedR:::massive_pca_checkpoint_fit(
                source, 2L, TRUE, FALSE, "cuda", 1L, 4L,
                resources, state, sidecar)
            expect_identical(saved$stage, "fit")
            expect_identical(readRDS(sidecar)$fit$fit_devices,
                c(0L, 1L))
        },
        massive_pca_decomposition = function(...) list(
            fit_devices = c(0L, 1L),
            fit_per_device = list(list(device = 0L, rows = 6,
                row_range = c(1, 6)), list(device = 1L, rows = 6,
                row_range = c(7, 12))),
            loadings = matrix(0, 4L, 2L),
            center = numeric(4L), scale = rep(1, 4L)),
        .package = "fastEmbedR"
    )
})

test_that("multi-GPU PCA refuses a fit with missing GPU work", {
    source <- fastEmbedR:::massive_synthetic_matrix(12L, 4L)
    resources <- list(chunk_rows = 4L,
        gpu_devices = c(0L, 1L))
    with_mocked_bindings(
        expect_error(fastEmbedR:::massive_pca_prepare(source,
            tempfile(fileext = ".f32"), 2L, TRUE, FALSE,
            "cuda", 1L, 4L, resources, FALSE, FALSE),
            "invalid GPU row ownership"),
        massive_pca_decomposition = function(...) list(
            fit_devices = c(0L, 1L),
            fit_per_device = list(list(device = 0L, rows = 12))),
        .package = "fastEmbedR")
})

test_that("multi-GPU PCA resume rejects changed fit row ownership", {
    source <- fastEmbedR:::massive_synthetic_matrix(12L, 4L)
    sidecar <- tempfile(fileext = ".rds")
    on.exit(unlink(sidecar))
    resources <- list(chunk_rows = 4L, output_bytes = 96,
        gpu_devices = c(0L, 1L))
    fit <- list(fit_devices = c(0L, 1L),
        fit_per_device = list(list(device = 0L, rows = 6,
            row_range = c(1, 6)), list(device = 1L, rows = 6,
            row_range = c(7, 12))))
    state <- list(signature = "source-locked", stage = "fit",
        completed_rows = 0, fit = fit,
        fit_hash = fastEmbedR:::massive_checkpoint_signature(fit))
    load <- function() fastEmbedR:::massive_pca_checkpoint_load(
        sidecar, "source-locked", source, 2L, resources,
        tempfile(fileext = ".part"), 1L, "cuda")
    saveRDS(state, sidecar)
    expect_identical(load()$fit, fit)
    state$fit$fit_per_device[[1L]]$row_range <- c(7, 12)
    state$fit$fit_per_device[[2L]]$row_range <- c(1, 6)
    state$fit_hash <- fastEmbedR:::massive_checkpoint_signature(
        state$fit)
    saveRDS(state, sidecar)
    expect_error(load(), "invalid GPU row ownership")
    state$fit <- fit
    state$fit$fit_per_device[[1L]]$rows <- 7
    state$fit$fit_per_device[[2L]]$rows <- 5
    state$fit_hash <- fastEmbedR:::massive_checkpoint_signature(
        state$fit)
    saveRDS(state, sidecar)
    expect_error(load(), "invalid GPU row ownership")
})

test_that("multi-GPU randomized PCA combines both GPU actions", {
    set.seed(4406)
    x <- matrix(rnorm(12L * 2050L), ncol = 2050L)
    raw <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(x)), raw, size = 4L,
        endian = "little")
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow(x), ncol(x))
    resources <- list(chunk_rows = 4L,
        memory_limit_bytes = 1e9, peak_ram_bytes = 1e8,
        gpu_devices = c(0L, 1L))
    with_mocked_bindings(
        {
            fit <- fastEmbedR:::massive_pca_wide_decomposition(
                source, 2L, TRUE, FALSE, resources, 1L,
                4L, "cuda")
            expect_identical(fit$fit_devices, c(0L, 1L))
            expect_identical(fit$fit_sketch_backend,
                "distributed_cuda_action")
            expect_identical(fit$moments_backend, "cpu")
            expect_identical(fit$decomposition_backend,
                "cpu_qr_eigen")
            expect_equal(fit$singular_values,
                svd(scale(x, center = TRUE, scale = FALSE),
                    nu = 0L, nv = 0L)$d[1:2],
                tolerance = 1e-4)
            expect_true(all(vapply(fit$fit_per_device,
                function(part) is.finite(part$second_action_seconds),
                logical(1L))))
        },
        massive_cuda_parallel_tasks = function(tasks, worker,
                arguments) {
            expect_identical(vapply(tasks, `[[`, 0L, "device"),
                c(0L, 1L))
            lapply(tasks, function(task) {
                rows <- seq.int(task$row_range[1L],
                    task$row_range[2L])
                block <- x[rows, , drop = FALSE]
                scaled <- sweep(sweep(block, 2L,
                    arguments$center), 2L, arguments$scale, "/")
                list(action = crossprod(scaled,
                    scaled %*% arguments$basis),
                    rows = as.double(nrow(block)),
                    row_range = task$row_range,
                    device = task$device, seconds = 1,
                    estimated_ram_bytes = 2048,
                    estimated_vram_bytes = 1024)
            })
        },
        massive_pca_wide_action_cuda_cpp = function(...) stop(
            "single-device sketch was used"),
        .package = "fastEmbedR"
    )
})

test_that("wide PCA worker restricts its CUDA action to owned rows", {
    x <- matrix(seq_len(8L * 2050L) / 1000, nrow = 8L)
    raw <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(x)), raw, size = 4L,
        endian = "little")
    on.exit(unlink(raw))
    source <- massive_matrix(raw, nrow(x), ncol(x))
    task <- list(device = 1L, row_range = c(5, 8))
    basis <- matrix(0, ncol(x), 2L)
    basis[1L, 1L] <- basis[2L, 2L] <- 1
    args <- list(source = source, rank = 2L,
        center = numeric(ncol(x)), scale = rep(1, ncol(x)),
        basis = basis, chunk_rows = 2L, memory_limit = 1e8)
    worker <- fastEmbedR:::massive_pca_wide_action_device_worker
    with_mocked_bindings(
        {
            result <- worker(task, args,
                normalizePath(find.package("fastEmbedR")),
                fastEmbedR:::
                    massive_checkpoint_implementation_identity())
            expect_identical(result$device, 1L)
            expect_equal(result$rows, 4)
            expect_identical(result$row_range, c(5, 8))
            block <- x[5:8, , drop = FALSE]
            expect_equal(result$action,
                crossprod(block, block %*% basis),
                tolerance = 1e-5)
        },
        massive_cuda_select_cpp = function(device) {
            expect_identical(device, 1L)
            device
        },
        massive_pca_resources = function(source, ...) list(
            gpu_device = 1L, chunk_rows = 2L,
            peak_ram_bytes = 2048,
            peak_vram_bytes = 1024),
        massive_pca_wide_action_cuda_cpp = function(source,
                center, scale, basis, ...) {
            block <- massive_read_rows(source, 1L, source$nrow)
            scaled <- sweep(sweep(block, 2L, center),
                2L, scale, "/")
            crossprod(scaled, scaled %*% basis)
        },
        .package = "fastEmbedR"
    )
})
