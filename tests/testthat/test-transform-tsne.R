test_that("transform_tsne places query rows from supplied reference neighbours", {
  set.seed(401)
  x <- matrix(rnorm(70L * 5L), 70L, 5L)
  ref <- x[1:50, , drop = FALSE]
  qry <- x[51:60, , drop = FALSE]

  fit <- infotsne(
    ref,
    n_neighbors = 15L,
    perplexity = 5,
    max_iter = 8L,
    n_negatives = 10L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL
  )
  query_knn <- nn(ref, qry, k = 15L, backend = "cpu")
  layout <- transform_tsne(
    fit$layout,
    knn = query_knn,
    perplexity = 5,
    n_iter = 6L,
    n_negatives = 10L,
    n_threads = 2L,
    seed = 401L
  )

  expect_equal(dim(layout), c(nrow(qry), 2L))
  expect_true(all(is.finite(layout)))
  expect_equal(attr(layout, "transform"), "opentsne_style_fixed_reference")
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "transform_tsne")
  expect_equal(cfg$optimizer, "opentsne_style_fixed_reference_transform")
  expect_equal(cfg$nn_backend, "cpu")
})

test_that("transform_tsne reports GPU transform backends honestly", {
  set.seed(403)
  x <- matrix(rnorm(60L * 4L), 60L, 4L)
  ref <- x[1:42, , drop = FALSE]
  qry <- x[43:50, , drop = FALSE]
  fit <- infotsne(
    ref,
    n_neighbors = 12L,
    perplexity = 4,
    max_iter = 6L,
    n_negatives = 8L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL
  )
  query_knn <- nn(ref, qry, k = 12L, backend = "cpu")

  expect_error(
    transform_tsne(
      fit$layout,
      knn = query_knn,
      perplexity = 4,
      n_iter = 3L,
      n_negatives = 8L,
      backend = "cuda"
    ),
    "CUDA t-SNE transform backend is planned"
  )

  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
  layout <- transform_tsne(
    fit$layout,
    knn = query_knn,
    perplexity = 4,
    n_iter = 3L,
    n_negatives = 42L,
    backend = "metal",
    seed = 403L
  )
  expect_equal(dim(layout), c(nrow(qry), 2L))
  expect_true(all(is.finite(layout)))
  expect_equal(attr(layout, "backend"), "metal")
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$backend, "metal")
  expect_equal(cfg$optimizer, "opentsne_style_fixed_reference_transform_metal")
  expect_equal(cfg$repulsion, "exact_reference_metal")
})

test_that("landmark_tsne returns a compact full embedding object", {
  set.seed(402)
  x <- rbind(
    matrix(rnorm(60L, 0, 0.25), 20L, 3L),
    matrix(rnorm(60L, 2, 0.25), 20L, 3L)
  )
  labels <- rep(1:2, each = 20L)

  fit <- landmark_tsne(
    x,
    labels = labels,
    landmarks = 20L,
    reference_method = "infotsne",
    n_neighbors = 12L,
    perplexity = 4,
    max_iter = 6L,
    n_negatives = 8L,
    transform_iter = 5L,
    transform_n_negatives = 8L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL,
    keep_knn = TRUE
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_true(all(is.finite(fit$layout)))
  expect_equal(fit$parameters$method, "landmark_tsne")
  expect_true(isTRUE(fit$parameters$landmark))
  expect_equal(fit$parameters$n_landmarks, 20L)
  expect_equal(fit$parameters$transform_optimizer, "opentsne_style_fixed_reference_transform")
  expect_true(is.list(fit$landmarks$projection_knn))
})
