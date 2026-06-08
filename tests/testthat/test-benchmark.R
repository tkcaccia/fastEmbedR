test_that("benchmark_knn_umap compares native KNN-based implementations", {
  set.seed(3)
  x <- rbind(matrix(rnorm(60), 20, 3), matrix(rnorm(60, 3), 20, 3))
  labels <- rep(1:2, each = 20)

  result <- benchmark_knn_umap(
    x,
    labels,
    k = 6,
    implementations = c("fastknnumap_spectral", "fastknnumap_hybrid"),
    n_epochs = 5,
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
  expect_equal(unique(result$metrics$k), 6L)
  expect_equal(unique(result$metrics$n_epochs), 5L)
})

test_that("benchmark_knn_umap rejects unfair epoch overrides across methods", {
  set.seed(3)
  x <- rbind(matrix(rnorm(60), 20, 3), matrix(rnorm(60, 3), 20, 3))
  labels <- rep(1:2, each = 20)

  expect_error(
    benchmark_knn_umap(
      x,
      labels,
      k = 6,
      implementations = c("fastknnumap_sgd", "fastknnumap_hybrid"),
      n_epochs = 10,
      hybrid_epochs = 5,
      verbose = FALSE
    ),
    "must equal `n_epochs`"
  )
})

test_that("benchmark_knn_umap maps compact method aliases to native implementations", {
  result <- benchmark_knn_umap(
    as.matrix(iris[, 1:4]),
    iris$Species,
    k = 10,
    implementations = c("umap", "tsne", "pacmap"),
    n_epochs = 20,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    verbose = FALSE
  )

  expect_equal(sort(result$metrics$implementation), c("fastknnumap_sgd", "knn_pacmap", "knn_tsne"))
  expect_true(all(result$metrics$status == "ok"))
  expect_true(all(is.finite(result$metrics$elapsed)))
  expect_true(all(is.finite(result$metrics$silhouette)))
})

test_that("benchmark_embedding_datasets combines dataset results", {
  result <- benchmark_embedding_datasets(
    datasets = "iris",
    subsets = c(iris = NA),
    implementations = c("fastknnumap_spectral", "knn_tsne"),
    repeats = 2,
    n_epochs = 20,
    spectral_n_iter = 3,
    silhouette_sample = NULL,
    preserve_sample = 50,
    verbose = FALSE
  )

  expect_equal(unique(result$metrics$dataset), "iris")
  expect_equal(nrow(result$metrics), 4L)
  expect_true(all(result$metrics$status == "ok"))
  expect_true(all(is.finite(result$metrics$elapsed)))
  expect_true(any(is.finite(result$metrics$stability)))
})

test_that("benchmark_embed exposes a minimal benchmark interface", {
  result <- benchmark_embed(
    datasets = "iris",
    methods = c("fast", "tsne"),
    preset = "quick",
    output_csv = NULL
  )

  expect_equal(unique(result$metrics$dataset), "iris")
  expect_equal(sort(result$metrics$implementation), c("fastknnumap_sgd", "knn_tsne"))
  expect_true(all(result$metrics$status == "ok"))
})

test_that("benchmark_embed writes plots when output_csv is set", {
  out <- tempfile(fileext = ".csv")
  result <- benchmark_embed(
    datasets = "iris",
    methods = c("fast", "tsne"),
    preset = "quick",
    output_csv = out
  )

  expect_true(file.exists(out))
  expect_true(length(result$plot_paths) >= 1L)
  expect_true(all(file.exists(result$plot_paths)))
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
