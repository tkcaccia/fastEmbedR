make_exact_knn <- function(x, k) {
  d <- as.matrix(stats::dist(x))
  diag(d) <- 0
  idx <- t(apply(d, 1, order))[, seq_len(k), drop = FALSE]
  dst <- matrix(0, nrow(d), k)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }
  list(idx = idx, dist = dst)
}

mean_silhouette <- function(layout, labels) {
  fastknnumap:::silhouette_score(labels, layout)
}

knn_preservation <- function(layout, reference_idx, k) {
  d <- as.matrix(stats::dist(layout))
  diag(d) <- Inf
  layout_idx <- t(apply(d, 1, order))[, seq_len(k), drop = FALSE]
  reference_idx <- reference_idx[, seq_len(k), drop = FALSE]
  mean(vapply(
    seq_len(nrow(reference_idx)),
    function(i) length(intersect(reference_idx[i, ], layout_idx[i, ])) / k,
    numeric(1)
  ))
}

test_that("fast_knn_umap returns a finite layout", {
  set.seed(1)
  x <- matrix(rnorm(60), ncol = 3)
  d <- as.matrix(dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[ , 1:5]
  dst <- matrix(seq_len(nrow(d)), nrow(d), 5)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }

  layout <- fast_knn_umap(idx, dst, n_epochs = 10, seed = 7)
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
})

test_that("SGD preserves cluster structure from the same KNN input", {
  set.seed(44)
  n_per_class <- 40L
  labels <- rep(seq_len(3L), each = n_per_class)
  x <- matrix(rnorm(length(labels) * 8L, sd = 0.75), ncol = 8L)
  x <- x + matrix(rep(c(-3, 0, 3)[labels], 8L), ncol = 8L)

  n_neighbors <- 15L
  nn <- make_exact_knn(x, n_neighbors + 1L)
  curve_a <- 1.576943
  curve_b <- 0.895061

  candidate <- fast_knn_umap(
    nn$idx[, -1L, drop = FALSE],
    nn$dist[, -1L, drop = FALSE],
    n_epochs = 120L,
    init = "random",
    min_dist = 0.1,
    a = curve_a,
    b = curve_b,
    negative_sample_rate = 5L,
    learning_rate = 1,
    repulsion_strength = 1,
    prune_epochs = TRUE,
    seed = 4L
  )

  candidate_sil <- mean_silhouette(candidate, labels)
  candidate_pres <- knn_preservation(candidate, nn$idx[, -1L, drop = FALSE], n_neighbors)

  expect_gt(candidate_sil, 0.55)
  expect_gt(candidate_pres, 0.35)
})

test_that("spectral mode returns a finite layout without SGD epochs", {
  set.seed(2)
  x <- matrix(rnorm(80), ncol = 4)
  d <- as.matrix(dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[, 1:4]
  dst <- matrix(0, nrow(d), 4)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }

  layout <- fast_knn_umap(idx, dst, mode = "spectral", spectral_n_iter = 5, seed = 8)
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
})

test_that("UMAP-style SGD controls are accepted", {
  set.seed(3)
  x <- matrix(rnorm(90), ncol = 3)
  d <- as.matrix(dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[, 1:5]
  dst <- matrix(0, nrow(d), 5)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }

  layout <- fast_knn_umap(
    idx,
    dst,
    n_epochs = 10,
    a = 1.576943,
    b = 0.895061,
    repulsion_strength = 1.2,
    init_sdev = "range",
    prune_epochs = TRUE,
    seed = 9
  )
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
})
