write_metal_pca_fbin <- function(x) {
    path <- tempfile(fileext = ".fbin")
    connection <- file(path, "wb")
    on.exit(close(connection))
    writeBin(as.integer(dim(x)), connection, size = 4L,
        endian = "little")
    writeBin(as.vector(t(x)), connection, size = 4L,
        endian = "little")
    path
}

test_that("tiled double reader preserves row-major fbin values", {
    x <- matrix(seq_len(137L * 37L) / 16, nrow = 137L)
    path <- write_metal_pca_fbin(x)
    on.exit(unlink(path))
    source <- massive_matrix(path, access = "stream")
    observed <- fastEmbedR:::massive_read_rows_cpp(
        source, 1L, 137L, 8 * 137 * 37, FALSE)
    expect_identical(observed, x)
    raw <- fastEmbedR:::massive_read_rows_cpp(
        source, 1L, 137L, 4 * 137 * 37, TRUE, TRUE)
    expect_type(raw, "raw")
    expect_identical(attr(raw, "nrow"), 137L)
    expect_identical(attr(raw, "ncol"), 37L)
    values <- readBin(raw, "numeric", n = 137L * 37L,
        size = 4L, endian = "little")
    expect_identical(values, as.vector(t(x)))
})

test_that("streamed Metal PCA combines centered chunks stably", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    x <- outer(seq_len(137L), seq_len(7L), function(i, j) {
        1e4 + sin(i * j) / 4
    })
    path <- write_metal_pca_fbin(x)
    on.exit(unlink(path))
    source <- massive_matrix(path, access = "stream")
    observed <- fastEmbedR:::massive_read_rows_cpp(
        source, 1L, 137L, 8 * 137 * 7, FALSE)
    result <- fastEmbedR:::massive_pca_covariance_metal(
        source, 31L, TRUE)
    centered <- sweep(observed, 2L, colMeans(observed))
    expected <- crossprod(centered) / (nrow(observed) - 1L)
    expect_equal(result$center, colMeans(observed), tolerance = 1e-8)
    expect_equal(result$covariance, expected, tolerance = 1e-3)
    expect_gt(result$metal_allocated_bytes_observed, 0)
})

test_that("streamed Metal PCA projects float32 blocks", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    x <- matrix(seq_len(137L * 7L) / 16, nrow = 137L)
    path <- write_metal_pca_fbin(x)
    on.exit(unlink(path))
    source <- massive_matrix(path, access = "stream")
    block <- fastEmbedR:::massive_read_rows_cpp(
        source, 1L, 137L, 4 * 137 * 7, TRUE, TRUE)
    loadings <- matrix(seq_len(14L) / 32, nrow = 7L)
    centers <- rep(0.5, 7L)
    scales <- rep(2, 7L)
    scores <- fastEmbedR:::pca_project_block_metal_cpp(
        block, loadings, centers, scales)
    raw <- fastEmbedR:::pca_crossprod_block_metal_cpp(block, FALSE)
    expected <- sweep(sweep(x, 2L, centers), 2L, scales, "/") %*%
        loadings
    expect_equal(dim(scores), dim(expected))
    expect_equal(as.vector(scores), as.vector(expected),
        tolerance = 1e-4)
    expect_gt(attr(scores, "metal_allocated_bytes_observed"), 0)
    expect_equal(raw$center, rep(0, 7L))
    expect_equal(raw$cross, crossprod(x), tolerance = 1e-4)
    expect_error(fastEmbedR:::pca_project_block_metal_cpp(
        x, loadings, centers, scales), "row-major float32")
    expect_error(fastEmbedR:::pca_crossprod_block_metal_cpp(
        x, FALSE), "row-major float32")
    expect_error(fastEmbedR:::pca_project_block_metal_cpp(
        block, loadings, centers, rep(0, 7L)), "invalid scaling")
})
