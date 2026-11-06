test_that("substantial Metal PCA uses a deeper randomized sketch", {
    large <- fastEmbedR:::fastembedr_rsvd_tuning(
        1000000L, 96L, 2L, "metal")
    expect_equal(large$oversample, 48L)
    expect_equal(large$power, 4L)
    expect_equal(fastEmbedR:::fastembedr_rsvd_tuning(
        10000000L, 96L, 2L, "metal")$oversample, 48L)
    expect_error(fastEmbedR:::fastembedr_rsvd_tuning(
        100000000L, 96L, 2L, "metal"), "safe MPS size")
    small <- fastEmbedR:::fastembedr_rsvd_tuning(
        256L, 8L, 2L, "metal")
    expect_equal(small$power, 1L)
})
