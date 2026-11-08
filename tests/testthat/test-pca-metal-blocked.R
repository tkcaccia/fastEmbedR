test_that("large Metal PCA selects the bounded covariance route", {
    expect_true(fastEmbedR:::fastembedr_metal_needs_blocking(
        c(100000000L, 96L)))
    expect_true(fastEmbedR:::fastembedr_metal_needs_blocking(
        c(70000000L, 30L)))
    expect_false(fastEmbedR:::fastembedr_metal_needs_blocking(
        c(10000L, 96L)))
})

test_that("blocked Metal PCA algebra matches a covariance reference", {
    set.seed(87L)
    x <- matrix(rnorm(820L), 205L, 4L)
    local_mocked_bindings(
        is_float32_matrix = function(x) TRUE,
        resident_pca_float_loadings = identity,
        resident_pca_block_rows = function(x) 37L,
        resident_float32_score_matrix_cpp = function(rows, columns)
            matrix(0, rows, columns),
        resident_float32_score_assign_cpp = function(scores,
                first_row, block) {
            scores[first_row:(first_row + nrow(block) - 1), ] <- block
            scores
        },
        resident_float32_row_major_block_cpp = function(input,
                first_row, count) input[seq.int(first_row,
                    length.out = count), , drop = FALSE],
        pca_crossprod_block_metal_cpp = function(block, center) {
            means <- if (center) colMeans(block) else rep(0, ncol(block))
            adjusted <- sweep(block, 2L, means)
            list(center = means, cross = crossprod(adjusted),
                metal_allocated_bytes_observed = 4096)
        },
        pca_project_block_metal_cpp = function(block, loadings,
                centers, scales) {
            scores <- sweep(sweep(block, 2L, centers), 2L,
                scales, "/") %*% loadings
            attr(scores, "metal_allocated_bytes_observed") <- 4096
            scores
        }, .package = "fastEmbedR"
    )
    fit <- fastEmbedR:::fastembedr_metal_blocked_pca(x, 2L,
        TRUE, FALSE, 87L)
    reference <- prcomp(x, center = TRUE, scale. = FALSE)
    expect_equal(fit$center, reference$center, tolerance = 1e-10)
    expect_equal(fit$singular_values, reference$sdev[1:2] *
        sqrt(nrow(x) - 1), tolerance = 1e-8)
    expect_true(all(abs(cor(fit$scores,
        reference$x[, 1:2]))[c(1, 4)] > 0.999999))
    expect_identical(fit$backend, "metal_mps_covariance")
    expect_gt(fit$metal_allocated_bytes_observed, 0)
    scaled <- fastEmbedR:::fastembedr_metal_blocked_pca(x, 2L,
        TRUE, TRUE, 87L)
    scaled_reference <- prcomp(x, center = TRUE, scale. = TRUE)
    expect_equal(scaled$scale, scaled_reference$scale,
        tolerance = 1e-8)
    expect_true(all(abs(cor(scaled$scores,
        scaled_reference$x[, 1:2]))[c(1, 4)] > 0.999999))
})

test_that("blocked resident float32 PCA uses the real Metal device", {
    skip_if_not_installed("float")
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    set.seed(61L)
    values <- matrix(rnorm(820L), 205L, 4L)
    data <- float::fl(values)
    fit <- fastEmbedR:::fastembedr_metal_blocked_pca(data,
        2L, TRUE, FALSE, 61L)
    reference <- prcomp(values, center = TRUE, scale. = FALSE)
    expect_identical(fit$backend, "metal_mps_covariance")
    expect_s4_class(fit$scores, "float32")
    expect_s4_class(fit$loadings, "float32")
    expect_gt(fit$metal_allocated_bytes_observed, 0)
    expect_true(all(abs(cor(float::dbl(fit$scores),
        reference$x[, 1:2]))[c(1, 4)] > 0.995))
})
