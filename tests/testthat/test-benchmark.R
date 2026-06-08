test_that("benchmark_knn_umap compares native KNN-based implementations", {
  set.seed(3)
  x <- rbind(matrix(rnorm(60), 20, 3), matrix(rnorm(60, 3), 20, 3))
  labels <- rep(1:2, each = 20)

  result <- benchmark_knn_umap(
    x,
    labels,
    k = 6,
    implementations = c("fastknnumap_spectral", "fastknnumap_hybrid"),
    hybrid_epochs = 5,
    spectral_n_iter = 3,
    silhouette_sample = 20,
    preserve_sample = 20,
    verbose = FALSE
  )

  expect_equal(nrow(result$metrics), 2L)
  expect_true(all(result$metrics$status == "ok"))
  expect_true(all(is.finite(result$metrics$elapsed)))
  expect_true(all(is.finite(result$metrics$silhouette)))
  expect_true(all(is.finite(result$metrics$knn_preservation)))
})

test_that("benchmark_knn_umap can compare umap and Rtsne references", {
  skip_if_not_installed("umap")
  skip_if_not_installed("Rtsne")

  result <- benchmark_knn_umap(
    as.matrix(iris[, 1:4]),
    iris$Species,
    k = 10,
    implementations = c("umap", "rtsne"),
    n_epochs = 250,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    verbose = FALSE
  )

  expect_equal(sort(result$metrics$implementation), c("rtsne", "umap"))
  expect_true(all(result$metrics$status == "ok"))
  expect_true(all(is.finite(result$metrics$elapsed)))
  expect_true(all(is.finite(result$metrics$silhouette)))
})

test_that("benchmark_embedding_datasets combines dataset results", {
  skip_if_not_installed("Rtsne")

  result <- benchmark_embedding_datasets(
    datasets = "iris",
    subsets = c(iris = NA),
    implementations = c("fastknnumap_spectral", "rtsne"),
    n_epochs = 250,
    spectral_n_iter = 3,
    silhouette_sample = NULL,
    preserve_sample = 50,
    verbose = FALSE
  )

  expect_equal(unique(result$metrics$dataset), "iris")
  expect_equal(nrow(result$metrics), 2L)
  expect_true(all(result$metrics$status == "ok"))
  expect_true(all(is.finite(result$metrics$elapsed)))
})

test_that("landmark_knn_umap embeds from a reduced KNN graph", {
  set.seed(4)
  x <- matrix(rnorm(90), ncol = 3)
  d <- as.matrix(dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[, 1:6]
  dst <- matrix(0, nrow(d), 6)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }

  graph <- landmark_knn_graph(idx, dst, landmark_ratio = 0.25, landmark_k = 3, local_k = 2)
  expect_equal(dim(graph$indices), c(nrow(x), 5L))
  expect_true(length(graph$landmarks) >= 2L)

  layout <- landmark_knn_umap(
    idx,
    dst,
    landmark_ratio = 0.25,
    landmark_k = 3,
    local_k = 2,
    n_epochs = 5,
    spectral_n_iter = 3
  )
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
})
