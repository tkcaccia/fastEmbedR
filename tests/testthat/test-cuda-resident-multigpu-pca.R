test_that("resident PCA selects both CUDA fit and score projection", {
    x <- matrix(seq_len(24L) / 24, nrow = 8L)
    covariance <- diag(c(3, 2, 1))
    calls <- character()
    with_mocked_bindings({
        fit <- pca(x, ncomp = 2L, backend = "cuda",
            devices = c(0L, 1L), xtest = x[1:2, , drop = FALSE])
        expect_identical(calls, c("fit", "project", "project"))
        expect_identical(fit$gpu_devices, c(0L, 1L))
        expect_identical(fit$gpu_identity, c("GPU-A", "GPU-B"))
        expect_identical(fit$engine,
            "native_cuda_multigpu_covariance")
        expect_identical(dim(fit$scores), c(8L, 2L))
        expect_identical(dim(fit$scores_test), c(2L, 2L))
        expect_identical(fit$test_projection_devices,
            c(0L, 1L))
    },
    massive_validate_devices = function(devices, ...) devices,
    massive_cuda_device_uuids = function(...)
        c("GPU-A", "GPU-B"),
    pca_cuda_resident_covariance_cpp = function(...) {
        calls <<- c(calls, "fit")
        list(center = colMeans(x), covariance = covariance)
    },
    pca_cuda_resident_project_cpp = function(input, devices, center,
            scale, loadings) {
        calls <<- c(calls, "project")
        matrix(1, nrow(input), ncol(loadings))
    }, .package = "fastEmbedR")
})

test_that("resident PCA device requests do not fall back", {
    x <- matrix(seq_len(24L) / 24, nrow = 8L)
    expect_error(pca(x, backend = "cpu", devices = c(0L, 1L)),
        "requires CUDA")
    expect_error(pca(x, backend = "cuda", devices = 0L),
        "at least two")
    expect_error(pca(x, backend = "cuda", devices = c(0L, 1L),
        checkpoint = TRUE), "out_of_core")
})

test_that("auto PCA retains the resident CUDA device request", {
    x <- matrix(seq_len(24L) / 24, nrow = 8L)
    with_mocked_bindings({
        fit <- fastEmbedR:::run_massive_auto_pca(x, 2L, NULL,
            TRUE, FALSE, "cuda", 1L, 4L, FALSE, NULL, NULL,
            "512MB", FALSE, FALSE, c(0L, 1L))
        expect_identical(fit$requested_devices, c(0L, 1L))
        expect_identical(fit$massive_auto, "in_memory")
    },
    massive_pca_auto_route = function(...)
        list(mode = "off", data = x, estimate = 1024),
    pca = function(..., devices) {
        list(requested_devices = devices)
    }, .package = "fastEmbedR")
})
