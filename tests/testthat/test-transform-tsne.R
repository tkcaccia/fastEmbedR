test_that("transform_tsne places query rows from supplied reference neighbours", {
  set.seed(401)
  x <- matrix(rnorm(70L * 5L), 70L, 5L)
  ref <- x[1:50, , drop = FALSE]
  qry <- x[51:60, , drop = FALSE]

  fit <- opentsne(
    ref,
    n_neighbors = 15L,
    perplexity = 5,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
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
  expect_equal(cfg$affinities, "precomputed_query_conditional")
  expect_equal(cfg$affinity_storage, "flat_row_major_double")
  expect_equal(cfg$transform_batches, 1L)
})

test_that("transform_tsne CPU transform is deterministic across query-parallel threads", {
  set.seed(406)
  n_ref <- 96L
  n_query <- 32L
  k <- 18L
  reference_layout <- matrix(rnorm(n_ref * 2L), ncol = 2L)
  indices <- matrix(sample.int(n_ref, n_query * k, replace = TRUE), nrow = n_query)
  distances <- matrix(rexp(n_query * k), nrow = n_query)
  distances <- t(apply(distances, 1L, sort))
  query_knn <- list(indices = indices, distances = distances)
  attr(query_knn, "backend") <- "synthetic_precomputed"

  one_thread <- transform_tsne(
    reference_layout,
    knn = query_knn,
    perplexity = 5,
    n_iter = 5L,
    n_negatives = 12L,
    exact_repulsion_threshold = 8L,
    n_threads = 1L,
    backend = "cpu",
    seed = 406L
  )
  four_threads <- transform_tsne(
    reference_layout,
    knn = query_knn,
    perplexity = 5,
    n_iter = 5L,
    n_negatives = 12L,
    exact_repulsion_threshold = 8L,
    n_threads = 4L,
    backend = "cpu",
    seed = 406L
  )

  expect_equal(dim(four_threads), dim(one_thread))
  expect_equal(as.numeric(four_threads), as.numeric(one_thread), tolerance = 0)
  expect_equal(attr(four_threads, "fastEmbedR_config")$n_threads, 4L)
})

test_that("transform_tsne CPU batching preserves fixed-reference results", {
  old_batch <- Sys.getenv("FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE", unset = NA_character_)
  on.exit({
    if (is.na(old_batch)) {
      Sys.unsetenv("FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE")
    } else {
      Sys.setenv(FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE = old_batch)
    }
  }, add = TRUE)

  set.seed(407)
  n_ref <- 80L
  n_query <- 30L
  k <- 15L
  reference_layout <- matrix(rnorm(n_ref * 2L), ncol = 2L)
  indices <- matrix(sample.int(n_ref, n_query * k, replace = TRUE), nrow = n_query)
  distances <- matrix(rexp(n_query * k), nrow = n_query)
  distances <- t(apply(distances, 1L, sort))
  query_knn <- list(indices = indices, distances = distances)
  attr(query_knn, "backend") <- "synthetic_precomputed"

  Sys.setenv(FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE = "1000")
  full_batch <- transform_tsne(
    reference_layout,
    knn = query_knn,
    perplexity = 5,
    n_iter = 4L,
    n_negatives = 10L,
    exact_repulsion_threshold = 8L,
    n_threads = 4L,
    backend = "cpu",
    seed = 407L
  )

  Sys.setenv(FASTEMBEDR_TSNE_TRANSFORM_BATCH_SIZE = "7")
  small_batch <- transform_tsne(
    reference_layout,
    knn = query_knn,
    perplexity = 5,
    n_iter = 4L,
    n_negatives = 10L,
    exact_repulsion_threshold = 8L,
    n_threads = 4L,
    backend = "cpu",
    seed = 407L
  )

  expect_equal(as.numeric(small_batch), as.numeric(full_batch), tolerance = 0)
  cfg <- attr(small_batch, "fastEmbedR_config")
  expect_equal(cfg$transform_batch_size, 7L)
  expect_equal(cfg$transform_batches, ceiling(n_query / 7))
})

test_that("transform_tsne reports GPU transform backends honestly", {
  set.seed(403)
  x <- matrix(rnorm(60L * 4L), 60L, 4L)
  ref <- x[1:42, , drop = FALSE]
  qry <- x[43:50, , drop = FALSE]
  fit <- opentsne(
    ref,
    n_neighbors = 12L,
    perplexity = 4,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL
  )
  query_knn <- nn(ref, qry, k = 12L, backend = "cpu")

  if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
    cuda_layout <- transform_tsne(
      fit$layout,
      knn = query_knn,
      perplexity = 4,
      n_iter = 3L,
      n_negatives = 8L,
      exact_repulsion_threshold = 1L,
      backend = "cuda",
      seed = 403L
    )
    expect_equal(dim(cuda_layout), c(nrow(qry), 2L))
    expect_true(all(is.finite(cuda_layout)))
    expect_equal(attr(cuda_layout, "backend"), "cuda")
    cfg <- attr(cuda_layout, "fastEmbedR_config")
    expect_equal(cfg$backend, "cuda")
    expect_equal(cfg$optimizer, "opentsne_style_fixed_reference_transform_cuda")
    expect_equal(cfg$repulsion, "sampled_reference_cuda")
  } else {
    expect_error(
      transform_tsne(
        fit$layout,
        knn = query_knn,
        perplexity = 4,
        n_iter = 3L,
        n_negatives = 8L,
        backend = "cuda"
      ),
      "CUDA t-SNE transform backend is not available"
    )
  }

  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
  layout <- transform_tsne(
    fit$layout,
    knn = query_knn,
    perplexity = 4,
    n_iter = 3L,
    n_negatives = 8L,
    exact_repulsion_threshold = 1L,
    backend = "metal",
    seed = 403L
  )
  expect_equal(dim(layout), c(nrow(qry), 2L))
  expect_true(all(is.finite(layout)))
  expect_equal(attr(layout, "backend"), "metal")
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$backend, "metal")
  expect_equal(cfg$optimizer, "opentsne_style_fixed_reference_transform_metal")
  expect_equal(cfg$repulsion, "sampled_reference_metal")
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
    reference_method = "opentsne",
    n_neighbors = 12L,
    perplexity = 4,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
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

test_that("native affine landmark projection returns finite local placements", {
  reference_data <- matrix(
    c(
      0, 0,
      1, 0,
      0, 1,
      1, 1,
      2, 0
    ),
    ncol = 2,
    byrow = TRUE
  )
  reference_layout <- reference_data
  query_data <- matrix(
    c(
      0, 0,
      0.8, 0.2,
      1.2, 0.7
    ),
    ncol = 2,
    byrow = TRUE
  )
  indices <- matrix(
    c(
      1, 2, 3, 4,
      2, 4, 1, 3,
      4, 5, 2, 3
    ),
    nrow = 3,
    byrow = TRUE
  )
  distances <- matrix(
    c(
      0, 1, 1, sqrt(2),
      sqrt(0.08), sqrt(0.68), sqrt(0.68), sqrt(1.28),
      sqrt(0.13), sqrt(0.53), sqrt(0.53), sqrt(1.13)
    ),
    nrow = 3,
    byrow = TRUE
  )

  projected <- fastEmbedR:::project_embedding_affine_cpp(
    reference_data,
    query_data,
    reference_layout,
    indices,
    distances,
    4L,
    1e-3,
    2.5
  )

  expect_equal(dim(projected$layout), c(3L, 2L))
  expect_true(all(is.finite(projected$layout)))
  expect_equal(as.numeric(projected$layout[1L, ]), c(0, 0), tolerance = 1e-12)
  expect_true(all(projected$confidence >= 0 & projected$confidence <= 1))
  expect_equal(projected$method, "local_affine_knn_projection")
})

test_that("landmark_tsne can use projection-specific approximate KNN", {
  old <- options(
    fastEmbedR.landmark_projection = "auto",
    fastEmbedR.landmark_projection_min_rows = 1L,
    fastEmbedR.landmark_projection_min_work = 0
  )
  on.exit(options(old), add = TRUE)

  set.seed(404)
  x <- rbind(
    matrix(rnorm(90L, 0, 0.25), 30L, 3L),
    matrix(rnorm(90L, 2, 0.25), 30L, 3L)
  )

  fit <- landmark_tsne(
    x,
    landmarks = 30L,
    n_neighbors = 12L,
    perplexity = 4,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    transform_k = 10L,
    transform_perplexity = 2,
    transform_iter = 3L,
    transform_n_negatives = 8L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL,
    keep_knn = TRUE,
    backend = "cpu",
    n_threads = 2L
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(fit$parameters$projection_nn_backend, "cpu_projection_approx")
  expect_equal(fit$parameters$projection_strategy, "random_projection_landmark_query_knn")
  expect_true(is.list(attr(fit$landmarks$projection_knn, "approximation")))
})

test_that("landmark_tsne uses fused Metal projection when requested", {
  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
  skip_if_not(fastEmbedR:::metal_opentsne_native_available())
  skip_if_not(metal_available())

  set.seed(405)
  x <- rbind(
    matrix(rnorm(72L, 0, 0.25), 24L, 3L),
    matrix(rnorm(72L, 2, 0.25), 24L, 3L)
  )

  fit <- landmark_tsne(
    x,
    landmarks = 24L,
    n_neighbors = 10L,
    perplexity = 3,
    early_exaggeration_iter = 1L,
    n_iter = 1L,
    transform_k = 8L,
    transform_iter = 1L,
    transform_n_negatives = 8L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL,
    keep_knn = TRUE,
    backend = "metal",
    n_threads = 2L,
    negative_gradient_method = "exact"
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(fit$parameters$projection_nn_backend, "metal_fused_projection")
  expect_equal(fit$parameters$projection_strategy, "query_only_exact_fused_landmark_projection_knn_confidence")
  expect_equal(attr(fit$landmarks$projection_knn, "metal_kernel"), "landmark_project_interpolate_knn_confidence")
})

test_that("landmark_tsne keeps Metal projection and transform native when intermediates are not requested", {
  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
  skip_if_not(fastEmbedR:::metal_opentsne_native_available())
  skip_if_not(metal_available())

  set.seed(406)
  x <- rbind(
    matrix(rnorm(72L, 0, 0.25), 24L, 3L),
    matrix(rnorm(72L, 2, 0.25), 24L, 3L)
  )

  fit <- landmark_tsne(
    x,
    landmarks = 24L,
    n_neighbors = 10L,
    perplexity = 3,
    early_exaggeration_iter = 1L,
    n_iter = 1L,
    transform_k = 8L,
    transform_iter = 1L,
    transform_n_negatives = 8L,
    standardize = FALSE,
    preserve_sample = NULL,
    silhouette_sample = NULL,
    keep_knn = FALSE,
    backend = "metal",
    n_threads = 2L,
    record_costs = FALSE,
    negative_gradient_method = "exact"
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(fit$parameters$projection_nn_backend, "metal_fused_projection")
  expect_equal(fit$parameters$projection_strategy, "query_only_exact_fused_landmark_projection_knn_confidence")
  expect_equal(fit$parameters$transform_optimizer, "opentsne_style_fixed_reference_transform_metal")
  expect_true(is.null(fit$landmarks$projection_knn))
})
