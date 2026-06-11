test_that("auto_k chooses bounded size-aware neighborhoods", {
  expect_equal(fastEmbedR:::auto_k(100L), 15L)
  expect_equal(fastEmbedR:::auto_k(100L, include_self = TRUE), 16L)
  expect_equal(fastEmbedR:::auto_k(1000L), 30L)
  expect_equal(fastEmbedR:::auto_k(20000L), 50L)
  expect_equal(fastEmbedR:::auto_k(matrix(0, 8L, 3L)), 7L)
  expect_error(fastEmbedR:::auto_k(1L), "at least two")
})

test_that("automatic embedding K is UMAP focused", {
  expect_equal(fastEmbedR:::auto_embedding_k(1000L, "umap"), 30L)
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

test_that("umap convenience wrapper runs the automatic workflow", {
  set.seed(43)
  x <- rbind(matrix(rnorm(30), 10L, 3L), matrix(rnorm(30, 2), 10L, 3L))
  labels <- rep(1:2, each = 10L)

  fit <- umap(
    x,
    labels = labels,
    n_neighbors = 4L,
    seed = 43L,
    silhouette_sample = NULL,
    preserve_sample = NULL
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(fit$parameters$method, "umap")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(colnames(fit$layout), c("UMAP1", "UMAP2"))
})

test_that("landmark mode runs UMAP only", {
  set.seed(44)
  x <- rbind(
    matrix(rnorm(48, 0, 0.2), 16L, 3L),
    matrix(rnorm(48, 2, 0.2), 16L, 3L),
    matrix(rnorm(48, 4, 0.2), 16L, 3L)
  )
  labels <- rep(1:3, each = 16L)

  fit <- umap(
    x,
    labels = labels,
    n_neighbors = 5L,
    landmarks = 12L,
    seed = 44L,
    silhouette_sample = NULL,
    preserve_sample = 20L
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_true(all(is.finite(fit$layout)))
  expect_true(isTRUE(fit$parameters$landmark))
  expect_equal(fit$parameters$n_landmarks, 12L)
  expect_equal(length(fit$landmarks$indices), fit$parameters$n_landmarks)
  expect_true(all(is.finite(fit$landmarks$layout)))
})

test_that("projection confidence selects only low-confidence non-landmarks", {
  n <- 5001L
  indices <- matrix(1:3, nrow = n, ncol = 3L, byrow = TRUE)
  distances <- matrix(rep(c(4, 4.2, 4.4), n), nrow = n, byrow = TRUE)
  distances[3000:n, ] <- matrix(rep(c(0.05, 2, 3), n - 2999L), ncol = 3L, byrow = TRUE)
  distances[c(1L, 10L), ] <- matrix(rep(c(0, 1, 2), 2L), ncol = 3L, byrow = TRUE)
  projection_nn <- list(indices = indices, distances = distances)

  old_enabled <- Sys.getenv("FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT", NA_character_)
  old_fraction <- Sys.getenv("FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT_FRACTION", NA_character_)
  Sys.setenv(
    FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT = "true",
    FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT_FRACTION = "0.20"
  )
  on.exit({
    if (is.na(old_enabled)) {
      Sys.unsetenv("FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT")
    } else {
      Sys.setenv(FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT = old_enabled)
    }
    if (is.na(old_fraction)) {
      Sys.unsetenv("FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT_FRACTION")
    } else {
      Sys.setenv(FASTEMBEDR_SELECTIVE_LANDMARK_REFINEMENT_FRACTION = old_fraction)
    }
  }, add = TRUE)

  scores <- fastEmbedR:::projection_confidence_scores(projection_nn)
  selected <- fastEmbedR:::select_landmark_refinement_rows(projection_nn, c(1L, 10L))

  expect_equal(scores[1L], 1)
  expect_lt(mean(scores[2:2500]), mean(scores[3000:n]))
  expect_equal(selected$policy, "low_confidence")
  expect_equal(selected$selection_backend, "r_confidence_mask")
  expect_false(any(selected$rows %in% c(1L, 10L)))
  expect_equal(length(selected$rows), ceiling((n - 2L) * 0.20))

  attr(projection_nn, "confidence") <- scores
  selected_native <- fastEmbedR:::select_landmark_refinement_rows(projection_nn, c(1L, 10L))
  expect_equal(selected_native$policy, "low_confidence")
  expect_equal(selected_native$selection_backend, "cpp_confidence_mask")
  expect_false(any(selected_native$rows %in% c(1L, 10L)))
  expect_equal(length(selected_native$rows), ceiling((n - 2L) * 0.20))
})

test_that("high-level embeddings avoid retaining KNN matrices by default", {
  set.seed(47)
  x <- matrix(rnorm(90), 30L, 3L)

  compact <- umap(
    x,
    n_neighbors = 5L,
    seed = 47L,
    silhouette_sample = NULL,
    preserve_sample = NULL
  )
  retained <- umap(
    x,
    n_neighbors = 5L,
    seed = 47L,
    keep_knn = TRUE,
    silhouette_sample = NULL,
    preserve_sample = NULL
  )

  expect_null(compact$knn)
  expect_equal(dim(retained$knn$indices), c(nrow(x), 5L))
  expect_equal(dim(retained$knn$distances), c(nrow(x), 5L))
})

test_that("supervised_umap explicitly uses labels for graph adjustment", {
  set.seed(49)
  x <- rbind(matrix(rnorm(60), 20L, 3L), matrix(rnorm(60, 1), 20L, 3L))
  labels <- rep(1:2, each = 20L)

  fit <- supervised_umap(
    x,
    labels = labels,
    n_neighbors = 6L,
    target_weight = 0.6,
    seed = 49L,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    keep_knn = TRUE
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_true(isTRUE(fit$parameters$supervised))
  expect_equal(fit$parameters$target_metric, "categorical")
  expect_equal(fit$parameters$target_weight, 0.6)
  expect_equal(dim(fit$knn$indices), c(nrow(x), 6L))
})
