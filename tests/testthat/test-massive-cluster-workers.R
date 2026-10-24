test_that("streamed community votes agree across worker counts", {
    labels <- rep.int(1L, 500000L)
    labels[c(1L, 2L, 3L)] <- c(1L, 200000L, 500000L)
    indices <- matrix(rep(c(1L, 2L, 3L), 48L), ncol = 3L,
        byrow = TRUE)
    distances <- matrix(rep(c(0.1, 0.2, 0.3), 48L), ncol = 3L,
        byrow = TRUE)
    single <- fastEmbedR:::massive_vote_landmarks_cpp(
        indices, distances, labels, 1L)
    parallel <- fastEmbedR:::massive_vote_landmarks_cpp(
        indices, distances, labels, 16L)
    expect_identical(parallel, single)
    expect_true(all(single$membership == 1L))
    expect_true(all(is.finite(single$confidence)))
    dense_labels <- c(3L, 2L, 1L)
    dense <- fastEmbedR:::massive_vote_landmarks_cpp(
        indices, distances, dense_labels, 8L)
    expect_true(all(dense$membership == 3L))
})
