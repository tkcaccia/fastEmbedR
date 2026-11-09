test_that("large UMAP distance profiles use bounded deterministic rows", {
    pick <- fastEmbedR:::umap_profile_row_indices
    rows <- pick(50000000L)
    expect_length(rows, 100000L)
    expect_identical(rows[[1L]], 1L)
    expect_identical(tail(rows, 1L), 50000000L)
    expect_true(all(diff(rows) > 0L))
    expect_identical(rows, pick(50000000L))
    expect_identical(pick(10L), seq_len(10L))
})

test_that("UMAP profiles sample before converting float32 distances", {
    skip_if_not_installed("float")
    rows <- 100050L
    values <- matrix(rep(seq_len(30L), each = rows),
        nrow = rows, ncol = 30L)
    distances <- float::fl(values)
    profile <- fastEmbedR:::fast_knn_umap_distance_profile(distances)
    expect_identical(profile$sampled_rows, 100000L)
    expect_true(is.finite(profile$cv))
    expect_true(is.finite(profile$ratio_30_15))
    expect_identical(
        fastEmbedR:::fast_knn_umap_distance_profile(
            distances[1:1000, , drop = FALSE])$sampled_rows,
        1000L)
})
