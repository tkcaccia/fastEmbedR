test_that("Metal public paths stay native and do not depend on Python bridges", {
  desc <- utils::packageDescription("fastEmbedR")
  dependency_fields <- paste(
    unlist(desc[c("Depends", "Imports", "LinkingTo")], use.names = FALSE),
    collapse = " "
  )
  expect_false(grepl("\\b(reticulate|torch|mlx)\\b", dependency_fields, ignore.case = TRUE))

  nn_body <- paste(deparse(body(fastEmbedR:::nn_compute)), collapse = "\n")
  prepare_body <- paste(deparse(body(fastEmbedR:::prepare_embedding_data)), collapse = "\n")
  transform_body <- paste(deparse(body(fastEmbedR::transform_tsne)), collapse = "\n")
  opentsne_body <- paste(deparse(body(fastEmbedR:::fast_knn_opentsne_materialized)), collapse = "\n")

  expect_match(nn_body, "nn_metal_cpp", fixed = TRUE)
  expect_match(prepare_body, "standardize_metal_cpp", fixed = TRUE)
  expect_match(transform_body, "transform_tsne_metal_cpp", fixed = TRUE)
  expect_match(opentsne_body, "knn_tsne_opentsne_metal_cpp", fixed = TRUE)
  expect_false(grepl("reticulate|py_|python|torch|mlx", paste(nn_body, prepare_body, transform_body, opentsne_body), ignore.case = TRUE))
})

test_that("Metal UMAP exposes only the validated atomic-inplace optimizer", {
  expect_equal(fastEmbedR:::fast_knn_umap_metal_optimizer_mode(), "atomic_inplace")
})

test_that("Metal UMAP auto policy keeps the visually validated atomic-inplace path", {
  old_hybrid <- getOption("fastEmbedR.gpu_hybrid_refine", NULL)
  on.exit({
    if (is.null(old_hybrid)) {
      options(fastEmbedR.gpu_hybrid_refine = NULL)
    } else {
      options(fastEmbedR.gpu_hybrid_refine = old_hybrid)
    }
  }, add = TRUE)

  options(fastEmbedR.gpu_hybrid_refine = "auto")

  cfg <- fastEmbedR:::fast_knn_umap_config(
    n = 70000L,
    k = 50L,
    backend = "metal"
  )
  expect_equal(fastEmbedR:::fast_knn_umap_metal_optimizer_mode(), "atomic_inplace")
  expect_false(fastEmbedR:::fast_knn_umap_gpu_hybrid_auto_selected(cfg))
  expect_equal(fastEmbedR:::fast_knn_umap_gpu_hybrid_plan(cfg)$reason, "auto_policy_keeps_pure_gpu")
})

test_that("Metal preprocessing, projection, interpolation, and scoring match CPU", {
  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())

  set.seed(71)
  x <- matrix(rnorm(160), nrow = 40L, ncol = 4L)
  cpu_pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = NULL,
    seed = 71L,
    backend = "cpu"
  )
  metal_pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = NULL,
    seed = 71L,
    backend = "metal"
  )
  expect_equal(metal_pre$preprocess$standardize_backend, "metal")
  expect_equal(metal_pre$data, cpu_pre$data, tolerance = 1e-5)

  x_pca <- matrix(rnorm(60L * 30L), nrow = 60L, ncol = 30L)
  metal_pca <- fastEmbedR:::prepare_embedding_data(
    x_pca,
    standardize = TRUE,
    pca_dims = 4L,
    seed = 71L,
    backend = "metal"
  )
  expect_equal(dim(metal_pca$data), c(60L, 4L))
  expect_equal(metal_pca$preprocess$pca_backend, "metal_rsvd")
  expect_equal(metal_pca$preprocess$pca_method, "rsvd")

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
  metal_project <- fastEmbedR:::project_embedding_knn_metal_cpp(
    reference_layout,
    projection_indices,
    projection_distances
  )
  expect_equal(metal_project, cpu_project, tolerance = 1e-5)

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
  metal_interp <- fastEmbedR:::interpolate_landmark_layout_metal_cpp(
    reference_layout,
    as.integer(landmark_indices),
    landmark_projection_indices,
    landmark_projection_distances,
    24L
  )
  expect_equal(metal_interp, cpu_interp, tolerance = 1e-5)

  x_query <- matrix(rnorm(36L * 5L), nrow = 36L, ncol = 5L)
  fused_landmark_indices <- c(1L, 4L, 9L, 13L, 18L, 22L, 29L, 35L)
  x_landmarks <- x_query[fused_landmark_indices, , drop = FALSE]
  fused_layout_reference <- cbind(rnorm(length(fused_landmark_indices)), rnorm(length(fused_landmark_indices)))
  fused_projection <- fastEmbedR::nn(x_landmarks, x_query, k = 4L, backend = "cpu")
  cpu_fused_reference <- fastEmbedR:::interpolate_landmark_layout_cpp(
    fused_layout_reference,
    as.integer(fused_landmark_indices),
    fused_projection$indices,
    fused_projection$distances,
    nrow(x_query)
  )
  metal_fused <- fastEmbedR:::landmark_project_interpolate_metal_cpp(
    x_landmarks,
    x_query,
    fused_layout_reference,
    as.integer(fused_landmark_indices),
    4L
  )
  expect_equal(metal_fused, cpu_fused_reference, tolerance = 1e-4)

  metal_fused_knn <- fastEmbedR:::landmark_project_interpolate_knn_confidence_metal_cpp(
    x_landmarks,
    x_query,
    fused_layout_reference,
    as.integer(fused_landmark_indices),
    4L
  )
  expect_equal(metal_fused_knn$layout, cpu_fused_reference, tolerance = 1e-4)
  expect_equal(metal_fused_knn$indices, fused_projection$indices)
  expect_equal(metal_fused_knn$distances, fused_projection$distances, tolerance = 1e-4)
  expect_equal(length(metal_fused_knn$confidence), nrow(x_query))
  expect_true(all(is.finite(metal_fused_knn$confidence)))
  expect_true(all(metal_fused_knn$confidence >= 0 & metal_fused_knn$confidence <= 1))
  expect_true(all(metal_fused_knn$confidence[fused_landmark_indices] > 0.99))

  layout <- cbind(rnorm(30L), rnorm(30L))
  labels <- rep(1:3, each = 10L)
  knn <- fastEmbedR::nn(layout, layout, k = 7L, backend = "cpu")
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
  metal_score <- fastEmbedR:::knn_structure_score_metal_cpp(
    layout,
    high_indices,
    as.integer(keep),
    6L,
    as.integer(labels),
    3L
  )
  expect_equal(metal_score, cpu_score, tolerance = 1e-5)

  cpu_sil <- fastEmbedR:::silhouette_score_cpp(layout, as.integer(labels))
  metal_sil <- fastEmbedR:::silhouette_score_metal_cpp(layout, as.integer(labels), 3L)
  expect_equal(metal_sil, cpu_sil, tolerance = 1e-5)
})
