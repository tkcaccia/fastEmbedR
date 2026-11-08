test_that("large resident CPU PCA selects bounded covariance", {
    expect_true(fastEmbedR:::fastembedr_cpu_needs_blocking(
        c(100000000L, 96L)))
    expect_false(fastEmbedR:::fastembedr_cpu_needs_blocking(
        c(1000000L, 96L)))
})

test_that("blocked CPU PCA keeps 16 workers and covariance parity", {
    set.seed(19L)
    x <- matrix(rnorm(840L), 210L, 4L)
    local_mocked_bindings(
        is_float32_matrix = function(x) TRUE,
        resident_pca_float_loadings = identity,
        resident_pca_block_rows = function(x) 37L,
        massive_pca_covariance_cpp = function(spec, chunk_rows,
                n_threads, center, ...) {
            expect_identical(spec$format, "memory_float32")
            expect_identical(spec$n_threads, 16L)
            expect_identical(n_threads, 16L)
            means <- if (center) colMeans(spec$data) else
                rep(0, ncol(spec$data))
            adjusted <- sweep(spec$data, 2L, means)
            list(center = means, covariance =
                crossprod(adjusted) / (nrow(spec$data) - 1))
        },
        resident_float32_project_cpu_cpp = function(input,
                loadings, centers, scales, n_threads) {
            expect_identical(n_threads, 16L)
            list(scores = sweep(sweep(input, 2L, centers),
                2L, scales, "/") %*% loadings,
                n_threads_effective = n_threads)
        }, .package = "fastEmbedR"
    )
    fit <- fastEmbedR:::fastembedr_cpu_blocked_pca(x, 2L,
        TRUE, FALSE, 19L, 16L)
    reference <- prcomp(x, center = TRUE, scale. = FALSE)
    expect_true(all(abs(cor(fit$scores,
        reference$x[, 1:2]))[c(1, 4)] > 0.999999))
    expect_identical(fit$n_threads, 16L)
    expect_identical(fit$backend, "cpu_covariance")
})

test_that("native resident float32 covariance and projection use 16", {
    skip_if_not_installed("float")
    set.seed(41L)
    values <- matrix(rnorm(820L), 205L, 4L)
    data <- float::fl(values)
    source <- list(data = data, format = "memory_float32")
    moments <- fastEmbedR:::massive_pca_covariance_cpp(
        source, 37L, 16L, TRUE)
    expect_equal(moments$center, colMeans(values), tolerance = 1e-5)
    expect_equal(moments$covariance, cov(values), tolerance = 1e-5)
    loadings <- eigen(moments$covariance, symmetric = TRUE)$vectors[, 1:2]
    fit <- fastEmbedR:::resident_float32_project_cpu_cpp(
        data, loadings, moments$center, rep(1, 4L), 16L)
    expect_identical(fit$n_threads_effective, 16L)
    expect_s4_class(fit$scores, "float32")
    expected <- sweep(values, 2L, colMeans(values)) %*% loadings
    expect_equal(float::dbl(fit$scores), expected, tolerance = 1e-4)
    complete <- fastEmbedR:::fastembedr_cpu_blocked_pca(
        data, 2L, TRUE, FALSE, 41L, 16L)
    expect_identical(complete$n_threads, 16L)
    expect_s4_class(complete$scores, "float32")
    expect_s4_class(complete$loadings, "float32")
    expect_true(all(abs(cor(float::dbl(complete$scores),
        prcomp(values)$x[, 1:2]))[c(1, 4)] > 0.999))
})
