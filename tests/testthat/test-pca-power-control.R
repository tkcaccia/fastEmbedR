test_that("resident CPU PCA honors a power-iteration count", {
    set.seed(204L)
    u <- qr.Q(qr(scale(matrix(rnorm(256L * 80L), 256L), scale = FALSE)))
    v <- qr.Q(qr(matrix(rnorm(80L * 80L), 80L)))
    sigmas <- c(1.7, 1.6, seq(1.1, 0.5, length.out = 78L))
    x <- sweep(u, 2L, sigmas, "*") %*% t(v)
    reference <- prcomp(x, rank. = 2L)$rotation
    baseline <- pca(x, backend = "cpu", seed = 42L, n.cores = 1L)
    refined <- pca(x, backend = "cpu", seed = 42L, n.cores = 1L,
        power_iterations = 3L)
    alignment <- function(fit) min(svd(crossprod(reference,
        fit$loadings), nu = 0L, nv = 0L)$d)
    expect_identical(baseline$power, 1L)
    expect_identical(refined$power, 3L)
    expect_identical(refined$backend, "cpu_rsvd")
    expect_gt(alignment(refined), 0.99)
    expect_gt(alignment(refined), alignment(baseline))
    expect_identical(pca(x, backend = "cpu", n.cores = 1L,
        power_iterations = 0L)$power, 0L)
})

test_that("PCA declines invalid iteration counts and incompatible routes", {
    x <- as.matrix(iris[, 1:4])
    expect_error(pca(x, backend = "cpu", power_iterations = -1L),
        "must be a nonnegative integer")
    expect_error(pca(x, backend = "cpu", power_iterations = 0.5),
        "must be a nonnegative integer")
    expect_error(pca(x, backend = "cpu", power_iterations = c(1L, 3L)),
        "must be a nonnegative integer")
    expect_error(pca(x, backend = "cpu", massive = "out_of_core",
        power_iterations = 3L), "requires a full resident CPU rSVD fit")
    expect_error(pca(x, backend = "cpu", landmarks = 50L,
        power_iterations = 3L), "requires a full resident CPU rSVD fit")
    expect_error(pca(x, devices = 0L, power_iterations = 3L),
        "requires a full resident CPU rSVD fit")
    expect_error(pca(x, backend = "cuda", power_iterations = 3L),
        "requires the CPU backend")
    expect_identical(pca(x, backend = "cpu", power_iterations = 3L)$power,
        0L)
    local_mocked_bindings(fastembedr_cpu_covariance_route = function(...) TRUE,
        .package = "fastEmbedR")
    expect_error(pca(x, backend = "cpu", power_iterations = 3L),
        "unavailable for CPU covariance PCA")
})
