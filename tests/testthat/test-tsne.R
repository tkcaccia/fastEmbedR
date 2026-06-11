test_that("embed_knn runs native t-SNE from supplied neighbours", {
  set.seed(310)
  x <- matrix(rnorm(50L * 4L), 50L, 4L)
  knn <- nn(x, k = 16L, backend = "cpu")
  init <- matrix(rnorm(nrow(x) * 2L, sd = 1e-4), ncol = 2L)

  layout <- embed_knn(
    knn,
    method = "tsne",
    perplexity = 5,
    max_iter = 10L,
    Y_init = init,
    n_threads = 2L,
    seed = 310L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "tsne")
  expect_equal(cfg$optimizer, "barnes_hut_sparse_knn")
  expect_equal(cfg$repulsion, "barnes_hut")
  expect_equal(cfg$perplexity, 5)
  expect_true(length(attr(layout, "itercosts")) >= 1L)
})

test_that("native t-SNE keeps exact repulsion available when theta is zero", {
  set.seed(311)
  x <- matrix(rnorm(40L * 4L), 40L, 4L)
  knn <- nn(x, k = 13L, backend = "cpu")

  layout <- embed_knn(
    knn,
    method = "tsne",
    perplexity = 4,
    theta = 0,
    max_iter = 5L,
    n_threads = 2L,
    seed = 311L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$optimizer, "exact_sparse_knn")
  expect_equal(cfg$repulsion, "pair_symmetric")
})

test_that("native t-SNE exposes openTSNE-style negative gradient selection", {
  set.seed(312)
  x <- matrix(rnorm(42L * 4L), 42L, 4L)
  knn <- nn(x, k = 13L, backend = "cpu")

  bh <- embed_knn(
    knn,
    method = "tsne",
    perplexity = 4,
    negative_gradient_method = "bh",
    max_iter = 5L,
    n_threads = 2L,
    seed = 312L
  )
  expect_equal(attr(bh, "fastEmbedR_config")$repulsion, "barnes_hut")

  exact <- embed_knn(
    knn,
    method = "tsne",
    perplexity = 4,
    negative_gradient_method = "exact",
    max_iter = 5L,
    n_threads = 2L,
    seed = 312L
  )
  expect_equal(attr(exact, "fastEmbedR_config")$repulsion, "pair_symmetric")

  expect_error(
    embed_knn(
      knn,
      method = "tsne",
      perplexity = 4,
      negative_gradient_method = "fft",
      max_iter = 5L
    ),
    "not yet ported",
    fixed = TRUE
  )
})

test_that("native openTSNE-style t-SNE uses two-phase optimizer", {
  set.seed(321)
  x <- matrix(rnorm(50L * 5L), 50L, 5L)
  knn <- nn(x, k = 16L, backend = "cpu")

  layout <- embed_knn(
    knn,
    method = "opentsne",
    perplexity = 5,
    early_exaggeration_iter = 3L,
    n_iter = 4L,
    learning_rate = "auto",
    negative_gradient_method = "bh",
    n_threads = 2L,
    seed = 321L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "opentsne")
  expect_equal(cfg$optimizer, "opentsne_barnes_hut_sparse_knn")
  expect_equal(cfg$repulsion, "barnes_hut")
  expect_equal(cfg$early_exaggeration_iter, 3L)
  expect_equal(cfg$n_iter, 4L)
  expect_equal(cfg$learning_rate, "auto")
  expect_equal(cfg$learning_rate_early, nrow(x) / cfg$early_exaggeration)
  expect_equal(cfg$learning_rate_normal, nrow(x) / cfg$exaggeration)

  expect_error(
    embed_knn(
      knn,
      method = "opentsne",
      perplexity = 5,
      negative_gradient_method = "fft",
      early_exaggeration_iter = 1L,
      n_iter = 1L
    ),
    "not yet ported",
    fixed = TRUE
  )
})

test_that("opentsne has direct KNN input functions", {
  set.seed(322)
  x <- matrix(rnorm(54L * 5L), 54L, 5L)
  labels <- rep(1:3, length.out = nrow(x))
  knn <- nn(x, k = 19L, backend = "cpu")

  layout <- opentsne_knn(
    knn$indices,
    knn$distances,
    n_neighbors = 12L,
    perplexity = 3,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    n_threads = 2L,
    seed = 322L
  )
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "opentsne")
  expect_equal(cfg$n_neighbors, 12L)
  expect_equal(cfg$perplexity, 3)

  fit <- opentsne(
    knn,
    labels = labels,
    n_neighbors = 12L,
    perplexity = 3,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    seed = 322L
  )
  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(fit$parameters$input, "knn")
  expect_equal(fit$metrics$n_neighbors, 12L)
  expect_equal(fit$metrics$preprocess_elapsed, 0)
  expect_equal(fit$metrics$knn_elapsed, 0)
})

test_that("t-SNE CUDA backend is explicit and never reports CPU as GPU", {
  set.seed(319)
  x <- matrix(rnorm(32L * 4L), 32L, 4L)
  knn <- nn(x, k = 10L, backend = "cpu")
  init <- matrix(rnorm(nrow(x) * 2L, sd = 1e-4), ncol = 2L)

  expect_error(
    embed_knn(
      knn,
      method = "tsne",
      perplexity = 3,
      max_iter = 2L,
      Y_init = init,
      backend = "metal"
    ),
    "not supported on Metal",
    fixed = TRUE
  )

  if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
    layout <- embed_knn(
      knn,
      method = "tsne",
      perplexity = 3,
      max_iter = 2L,
      Y_init = init,
      backend = "cuda"
    )
    expect_equal(dim(layout), c(nrow(x), 2L))
    expect_true(all(is.finite(layout)))
    expect_equal(attr(layout, "fastEmbedR_config")$backend, "cuda")
    expect_equal(
      attr(layout, "fastEmbedR_config")$optimizer,
      "cuda_exact_dense_from_knn"
    )
  } else {
    expect_error(
      embed_knn(
        knn,
        method = "tsne",
        perplexity = 3,
        max_iter = 2L,
        Y_init = init,
        backend = "cuda"
      ),
      "CUDA exact t-SNE",
      fixed = TRUE
    )
    expect_error(
      embed_knn(
        knn,
        method = "tsne",
        perplexity = 3,
        max_iter = 2L,
        Y_init = init,
        backend = "gpu"
      ),
      "will not run CPU code and report it as GPU",
      fixed = TRUE
    )
  }
})

test_that("native t-SNE can use KeOps-style blocked exact repulsion", {
  old <- Sys.getenv("FASTEMBEDR_TSNE_REPULSION", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTEMBEDR_TSNE_REPULSION")
    } else {
      Sys.setenv(FASTEMBEDR_TSNE_REPULSION = old)
    }
  }, add = TRUE)
  Sys.setenv(FASTEMBEDR_TSNE_REPULSION = "keops_blocked")

  set.seed(314)
  x <- matrix(rnorm(48L * 4L), 48L, 4L)
  knn <- nn(x, k = 16L, backend = "cpu")

  layout <- embed_knn(
    knn,
    method = "tsne",
    perplexity = 5,
    max_iter = 5L,
    n_threads = 2L,
    seed = 314L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$optimizer, "exact_sparse_knn_keops_blocked")
  expect_equal(cfg$repulsion, "keops_blocked")
  expect_true(cfg$repulsion_block_size >= 32L)
})

test_that("embed_knn runs native InfoTSNE from supplied neighbours", {
  set.seed(315)
  x <- matrix(rnorm(55L * 5L), 55L, 5L)
  knn <- nn(x, k = 16L, backend = "cpu")

  layout <- embed_knn(
    knn,
    method = "infotsne",
    perplexity = 5,
    max_iter = 8L,
    n_negatives = 12L,
    n_threads = 2L,
    seed = 315L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "infotsne")
  expect_equal(cfg$optimizer, "infotsne_negative_sampling")
  expect_equal(cfg$n_negatives, 12L)
})

test_that("tsne convenience wrapper returns a compact embedding object", {
  set.seed(311)
  x <- rbind(matrix(rnorm(60L, 0, 0.3), 20L, 3L),
             matrix(rnorm(60L, 2, 0.3), 20L, 3L))
  labels <- rep(1:2, each = 20L)

  fit <- tsne(
    x,
    labels = labels,
    n_neighbors = 15L,
    perplexity = 5,
    max_iter = 10L,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    keep_knn = TRUE
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_true(all(is.finite(fit$layout)))
  expect_equal(fit$parameters$method, "tsne")
  expect_equal(dim(fit$knn$indices), c(nrow(x), 15L))
})

test_that("infotsne convenience wrapper returns a compact embedding object", {
  set.seed(316)
  x <- rbind(matrix(rnorm(60L, 0, 0.3), 20L, 3L),
             matrix(rnorm(60L, 2, 0.3), 20L, 3L))
  labels <- rep(1:2, each = 20L)

  fit <- infotsne(
    x,
    labels = labels,
    n_neighbors = 15L,
    perplexity = 5,
    max_iter = 8L,
    n_negatives = 10L,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    keep_knn = TRUE
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_true(all(is.finite(fit$layout)))
  expect_equal(fit$parameters$method, "infotsne")
  expect_equal(dim(fit$knn$indices), c(nrow(x), 15L))
})

test_that("tsne can use cuVS for KNN without labelling the optimizer as GPU", {
  set.seed(313)
  x <- matrix(rnorm(36L * 4L), 36L, 4L)

  if (isTRUE(cuvs_available())) {
    fit <- tsne(
      x,
      n_neighbors = 6L,
      perplexity = 2,
      max_iter = 5L,
      backend = "cuda_cuvs_bruteforce",
      silhouette_sample = NULL,
      preserve_sample = NULL,
      keep_knn = TRUE
    )
    expect_equal(fit$parameters$method, "tsne")
    expect_equal(fit$parameters$backend, "cpu")
    expect_equal(fit$parameters$nn_backend, "cuda_cuvs_bruteforce")
    expect_equal(dim(fit$knn$indices), c(nrow(x), 6L))
  } else {
    expect_error(
      tsne(
        x,
        n_neighbors = 6L,
        perplexity = 2,
        max_iter = 5L,
        backend = "cuda_cuvs_bruteforce",
        silhouette_sample = NULL,
        preserve_sample = NULL
      ),
      "cuVS"
    )
  }
})

test_that("native t-SNE is visually/structurally close to Rtsne_neighbors on a fixed start", {
  skip_if_not_installed("Rtsne")
  set.seed(312)
  x <- matrix(rnorm(70L * 5L), 70L, 5L)
  knn <- nn(x, k = 16L, backend = "cpu")
  idx <- knn$indices[, -1L, drop = FALSE]
  dst <- knn$distances[, -1L, drop = FALSE]
  init <- matrix(rnorm(nrow(x) * 2L, sd = 1e-4), ncol = 2L)

  fast <- embed_knn(
    list(indices = idx, distances = dst),
    method = "tsne",
    perplexity = 5,
    max_iter = 20L,
    Y_init = init,
    n_threads = 2L,
    seed = 312L
  )
  ref <- Rtsne::Rtsne_neighbors(
    idx,
    dst,
    perplexity = 5,
    max_iter = 20L,
    Y_init = init,
    num_threads = 2L,
    verbose = FALSE
  )$Y

  expect_true(all(is.finite(fast)))
  expect_true(all(is.finite(ref)))
  expect_gt(suppressWarnings(cor(as.vector(stats::dist(fast)), as.vector(stats::dist(ref)))), 0.75)
})
