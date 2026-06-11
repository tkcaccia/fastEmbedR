test_that("CUDA preprocessing, projection, interpolation, and scoring match CPU", {
  skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())

  set.seed(71)
  x <- matrix(rnorm(160), nrow = 40L, ncol = 4L)
  cpu_pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = NULL,
    seed = 71L,
    backend = "cpu"
  )
  cuda_pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = NULL,
    seed = 71L,
    backend = "cuda"
  )
  expect_equal(cuda_pre$preprocess$standardize_backend, "cuda")
  expect_equal(cuda_pre$data, cpu_pre$data, tolerance = 1e-10)

  x_pca <- matrix(rnorm(60L * 30L), nrow = 60L, ncol = 30L)
  cuda_pca <- fastEmbedR:::prepare_embedding_data(
    x_pca,
    standardize = TRUE,
    pca_dims = 4L,
    seed = 71L,
    backend = "cuda"
  )
  expect_equal(dim(cuda_pca$data), c(60L, 4L))
  expect_equal(cuda_pca$preprocess$pca_backend, "cuda_rsvd")
  expect_equal(cuda_pca$preprocess$pca_method, "rsvd")

  reference_layout <- cbind(rnorm(8L), rnorm(8L))
  projection_indices <- matrix(
    c(
      1L, 2L, 3L,
      3L, 4L, 5L,
      5L, 6L, 7L,
      2L, 4L, 8L,
      1L, 7L, 8L
    ),
    nrow = 5L,
    byrow = TRUE
  )
  projection_distances <- matrix(
    c(
      0.0, 0.4, 0.7,
      0.2, 0.3, 0.9,
      0.1, 0.6, 0.8,
      0.5, 0.6, 0.7,
      0.3, 0.4, 0.5
    ),
    nrow = 5L,
    byrow = TRUE
  )
  cpu_project <- fastEmbedR:::project_embedding_knn_cpp(
    reference_layout,
    projection_indices,
    projection_distances
  )
  cuda_project <- fastEmbedR:::project_embedding_knn_cuda_cpp(
    reference_layout,
    projection_indices,
    projection_distances
  )
  expect_equal(cuda_project, cpu_project, tolerance = 1e-7)

  landmark_indices <- c(2L, 5L, 8L, 11L, 14L, 17L, 20L, 23L)
  landmark_projection_indices <- matrix(
    rep(c(1L, 2L, 3L, 4L), length.out = 24L * 4L),
    nrow = 24L,
    ncol = 4L
  )
  landmark_projection_distances <- matrix(
    abs(rnorm(24L * 4L)) + 0.01,
    nrow = 24L,
    ncol = 4L
  )
  for (i in seq_along(landmark_indices)) {
    landmark_projection_indices[landmark_indices[i], 1L] <- i
    landmark_projection_distances[landmark_indices[i], 1L] <- 0
  }
  cpu_interp <- fastEmbedR:::interpolate_landmark_layout_cpp(
    reference_layout,
    as.integer(landmark_indices),
    landmark_projection_indices,
    landmark_projection_distances,
    24L
  )
  cuda_interp <- fastEmbedR:::interpolate_landmark_layout_cuda_cpp(
    reference_layout,
    as.integer(landmark_indices),
    landmark_projection_indices,
    landmark_projection_distances,
    24L
  )
  expect_equal(cuda_interp, cpu_interp, tolerance = 1e-7)

  layout <- cbind(rnorm(30L), rnorm(30L))
  labels <- rep(1:3, each = 10L)
  knn <- nn(layout, layout, k = 7L, backend = "cpu")
  high_indices <- knn$indices[, -1L, drop = FALSE]
  keep <- seq_len(nrow(layout))
  cpu_score <- fastEmbedR:::knn_structure_score_cpp(
    layout,
    high_indices,
    as.integer(keep),
    6L,
    as.integer(labels),
    3L
  )
  cuda_score <- fastEmbedR:::knn_structure_score_cuda_cpp(
    layout,
    high_indices,
    as.integer(keep),
    6L,
    as.integer(labels),
    3L
  )
  expect_equal(cuda_score, cpu_score, tolerance = 1e-8)

  cpu_sil <- fastEmbedR:::silhouette_score_cpp(layout, as.integer(labels))
  cuda_sil <- fastEmbedR:::silhouette_score_cuda_cpp(layout, as.integer(labels), 3L)
  expect_equal(cuda_sil, cpu_sil, tolerance = 1e-10)

  fit <- umap(
    x,
    labels = rep(1:2, each = 20L),
    n_neighbors = 6L,
    landmarks = 16L,
    backend = "cuda",
    seed = 72L,
    silhouette_sample = 20L,
    preserve_sample = 20L
  )
  expect_equal(fit$parameters$standardize_backend, "cuda")
  expect_equal(fit$parameters$landmark_interpolation_backend, "cuda")
  expect_equal(fit$parameters$scoring_structure_backend, "cuda")
  expect_equal(fit$parameters$scoring_silhouette_backend, "cuda")
})
