test_that("KNN objective variants return finite layouts", {
  set.seed(5)
  x <- rbind(matrix(rnorm(60), 20, 3), matrix(rnorm(60, 2), 20, 3))
  d <- as.matrix(dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[, 1:6]
  dst <- matrix(0, nrow(d), 6)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }

  for (objective in c("tsne", "pacmap", "trimap", "localmap")) {
    layout <- knn_embed(
      idx,
      dst,
      objective = objective,
      n_epochs = 3,
      n_threads = 2,
      seed = 9
    )
    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
  }
})
