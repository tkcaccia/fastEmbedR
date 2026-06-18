test_that("auto_k chooses bounded size-aware neighborhoods", {
  expect_equal(fastEmbedR:::auto_k(100L), 15L)
  expect_equal(fastEmbedR:::auto_k(100L, include_self = TRUE), 16L)
  expect_equal(fastEmbedR:::auto_k(1000L), 30L)
  expect_equal(fastEmbedR:::auto_k(20000L), 50L)
  expect_equal(fastEmbedR:::auto_k(matrix(0, 8L, 3L)), 7L)
  expect_error(fastEmbedR:::auto_k(1L), "at least two")
})

test_that("automatic embedding K is openTSNE focused", {
  expect_equal(fastEmbedR:::auto_embedding_k(1000L, "opentsne"), 30L)
  expect_equal(fastEmbedR:::auto_embedding_k(1000L, include_self = TRUE), 31L)
})

test_that("preprocessing PCA uses fastPLS-style RSVD", {
  set.seed(39)
  x <- matrix(rnorm(80L * 40L), 80L, 40L)
  pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = 5L,
    seed = 39L,
    backend = "cpu"
  )

  expect_equal(dim(pre$data), c(80L, 5L))
  expect_true(all(is.finite(pre$data)))
  expect_equal(pre$preprocess$pca_method, "rsvd")
  expect_equal(pre$preprocess$pca_backend, "cpu_rsvd")
})

test_that("opentsne convenience wrapper runs the automatic KNN workflow", {
  set.seed(43)
  x <- rbind(matrix(rnorm(30), 10L, 3L), matrix(rnorm(30, 2), 10L, 3L))
  labels <- rep(1:2, each = 10L)

  fit <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 43L
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(fit$parameters$method, "opentsne")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(colnames(fit$layout), c("openTSNE1", "openTSNE2"))
})

test_that("high-level embeddings avoid retaining KNN matrices by default", {
  set.seed(47)
  x <- matrix(rnorm(90), 30L, 3L)

  compact <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 47L
  )
  retained <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 47L,
    keep_knn = TRUE
  )

  expect_null(compact$knn)
  expect_equal(dim(retained$knn$indices), c(nrow(x), 1L))
  expect_equal(dim(retained$knn$distances), c(nrow(x), 1L))
})
